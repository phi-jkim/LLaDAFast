"""
BlockSoftmaxLinearHybrid: complementary combination of two attention mechanisms.

Design
──────
For each block n (tokens at positions [n·S, (n+1)·S)):

  softmax_out_n:  full bidirectional softmax attention **within block n only**
                  (exact, cheap — only S×S tokens; no cross-block context)

  linear_out_n:   kernel linear attention using the cumulative state from
                  **all previous blocks 0 … n-1** (approximate, O(D·F) state;
                  provides long-range cross-block context)

  out_n = sigmoid(alpha) · softmax_out_n  +  (1 − sigmoid(alpha)) · linear_out_n

The two streams are complementary:
  · softmax handles exact token interactions within the current denoising block.
  · linear carries the "memory" of everything denoised so far.

State update order: linear_out_n is computed BEFORE the state is updated with
block n, so the linear stream is strictly causal across blocks.

Stage-1 distillation usage
───────────────────────────
  · alpha (learnable scalar): init = 4.0 → sigmoid ≈ 0.98 (start softmax-dominant).
    Anneals toward 0 (pure linear) as hedgehog_weights converge.
  · In hybrid mode the parent LLaDA2MoeAttention freezes Q/K/V (only
    linear_attention.{hedgehog_weights, alpha} have requires_grad=True).
  · The first block has no prior context, so linear_out_0 = 0 and the output
    is sigmoid(alpha) · softmax_out_0.

API — same as OrderInvariantKernelLinearAttention:
  forward(q, k, v, attention_mask=None, key_padding_mask=None) → (B, H, L, D)
  attention_mask is NOT used (not needed; softmax is intra-block, linear is
  handled internally).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear_attention import OrderInvariantKernelLinearAttention


class BlockSoftmaxLinearHybrid(nn.Module):

    def __init__(self, config, block_size: int | None = None):
        super().__init__()
        self.num_heads  = int(config.num_attention_heads)
        self.head_dim   = config.head_dim
        self.block_size = int(block_size or getattr(config, "block_size", 32))
        self.feature_dim = int(config.feature_dim)
        self.eps        = 1e-6
        self.scaling    = self.head_dim ** -0.5

        # Hedgehog feature map (shared with OrderInvariantKernelLinearAttention internals).
        weight = torch.eye(self.head_dim, self.feature_dim).unsqueeze(0).expand(
            self.num_heads, -1, -1
        )
        self.hedgehog_weights = nn.Parameter(weight.clone())   # (H, D, F)

        # Mixing scalar: sigmoid(4.0) ≈ 0.982 → starts softmax-dominant.
        self.alpha = nn.Parameter(torch.tensor(4.0))

    # ── Hedgehog feature map ─────────────────────────────────────────────────

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:      (B, H, N, S, D)   [blocks × tokens × head_dim]
        return: (B, H, N, S, 2F)  [cat(softmax(u), softmax(-u))]
        """
        weights = self.hedgehog_weights.to(dtype=x.dtype)
        u = torch.einsum("bhnsd,hdf->bhnsf", x, weights)
        u_f32 = u.float()
        return torch.cat(
            [F.softmax(u_f32, dim=-1), F.softmax(-u_f32, dim=-1)], dim=-1
        ).to(dtype=x.dtype)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        query_states: torch.Tensor,     # (B, H, L, D)
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask=None,            # ignored (not needed by design)
        key_padding_mask=None,          # (B, L) float/bool, 1=real token
    ) -> torch.Tensor:
        """Returns (B, H, L, D)."""
        B, H, L, D = query_states.shape
        out_dtype = query_states.dtype

        # ── Pad to an integer number of blocks ──────────────────────────────
        num_blocks = (L + self.block_size - 1) // self.block_size
        padded_L   = num_blocks * self.block_size
        pad_len    = padded_L - L

        q = query_states
        k = key_states
        v = value_states
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))

        # Reshape to (B, H, N, S, D) — N blocks of S tokens each.
        S = self.block_size
        q = q.view(B, H, num_blocks, S, D)
        k = k.view(B, H, num_blocks, S, D)
        v = v.view(B, H, num_blocks, S, D)

        # ── Build key validity mask for padding ──────────────────────────────
        # key_valid: (B, 1, N, S, 1) float32
        if key_padding_mask is not None:
            kpm = key_padding_mask.float()[:, :padded_L]
            if kpm.shape[-1] < padded_L:
                kpm = F.pad(kpm, (0, padded_L - kpm.shape[-1]))
            key_valid = kpm.view(B, 1, num_blocks, S, 1)
        elif pad_len > 0:
            valid_1d = torch.ones(padded_L, device=q.device)
            valid_1d[L:] = 0.0
            key_valid = valid_1d.view(1, 1, num_blocks, S, 1)
        else:
            key_valid = None

        # ── Compute Hedgehog feature maps ────────────────────────────────────
        phi_q = self._feature_map(q)   # (B, H, N, S, 2F)
        phi_k = self._feature_map(k)   # (B, H, N, S, 2F)
        D_feat = 2 * self.feature_dim

        # Mask padding in keys/values for both linear and padding zeroing.
        if key_valid is not None:
            kv = key_valid.to(device=phi_k.device, dtype=phi_q.dtype)
            phi_k = phi_k * kv
            v = v * kv.to(v.dtype)

        # ── Block-by-block computation ───────────────────────────────────────
        # Linear state (fp32 for numerical stability).
        S_state = torch.zeros(B, H, D_feat, D, dtype=torch.float32, device=q.device)
        Z_state = torch.zeros(B, H, D_feat,    dtype=torch.float32, device=q.device)

        w = torch.sigmoid(self.alpha)   # mixing weight, differentiable

        out_blocks = []
        for n in range(num_blocks):
            phi_k_n = phi_k[:, :, n]   # (B, H, S, 2F)
            v_n     = v[:, :, n]        # (B, H, S, D)
            phi_q_n = phi_q[:, :, n]   # (B, H, S, 2F)
            q_n     = q[:, :, n]        # (B, H, S, D)
            k_n     = k[:, :, n]        # (B, H, S, D)

            # ── Linear path: prior-blocks context only ───────────────────────
            # Query the state accumulated up to block n-1 (BEFORE this block's update).
            # For block 0, S_state = Z_state = 0, so linear_out_n = 0.
            denom = torch.matmul(phi_q_n.float(), Z_state.unsqueeze(-1)).clamp_min(self.eps)
            linear_out_n = (
                torch.matmul(phi_q_n.float(), S_state) / denom
            ).to(out_dtype)           # (B, H, S, D)

            # Update state with block n AFTER computing linear output.
            S_state = S_state + torch.matmul(
                phi_k_n.transpose(-2, -1).float(), v_n.float()
            )
            Z_state = Z_state + phi_k_n.float().sum(dim=-2)

            # ── Softmax path: within-block only ─────────────────────────────
            # Full bidirectional SDPA among the S tokens of block n.
            # q_n, k_n: (B, H, S, D) — treat the S-dim as the sequence dim.
            softmax_out_n = F.scaled_dot_product_attention(
                q_n, k_n, v_n,
                attn_mask=None,    # no causal or cross-block masking needed
                dropout_p=0.0,
                scale=self.scaling,
            )  # (B, H, S, D)

            # ── Mix ──────────────────────────────────────────────────────────
            out_blocks.append(w * softmax_out_n + (1.0 - w) * linear_out_n)

        # Reassemble and strip padding.
        out = torch.stack(out_blocks, dim=2).view(B, H, padded_L, D)
        if pad_len > 0:
            out = out[:, :, :L, :]
        return out

    def forward_bd3lm(
        self,
        query_states: torch.Tensor,      # (B, H, 2L, D)
        key_states: torch.Tensor,        # (B, H, 2L, D)
        value_states: torch.Tensor,      # (B, H, 2L, D)
        half_len: int,                   # L — original single-half length
        key_padding_mask=None,           # (B, 2L) float/bool, 1=real token
    ) -> torch.Tensor:
        """
        BD3LM staircase forward for the hybrid attention.

        Same semantics as OrderInvariantKernelLinearAttention.forward_bd3lm:
          - Noisy block i queries linear state from clean blocks 0..i-1 only.
          - State updated with clean block i keys/values.
          - Clean block i queries updated state (includes self).
          - Noisy keys are NEVER added to the recurrent state.

        Per block, each stream is:
          softmax:  bidirectional within-block only (S×S, exact).
          linear:   cross-block causal context via recurrent state (approximate).
          output:   sigmoid(alpha) * softmax  +  (1-sigmoid(alpha)) * linear

        Returns (B, H, 2L, D) — [noisy outputs | clean outputs].
        """
        B, H, total_L, D = query_states.shape
        assert total_L == 2 * half_len

        out_dtype = query_states.dtype
        S = self.block_size

        # ── Split halves ──────────────────────────────────────────────────────
        q_noisy = query_states[:, :, :half_len, :]   # (B, H, L, D)
        q_clean = query_states[:, :, half_len:, :]
        k_noisy = key_states  [:, :, :half_len, :]   # needed for noisy intra-block softmax
        k_clean = key_states  [:, :, half_len:, :]   # only clean keys enter linear state
        v_noisy = value_states[:, :, :half_len, :]
        v_clean = value_states[:, :, half_len:, :]

        num_blocks = (half_len + S - 1) // S
        padded_L   = num_blocks * S
        pad_len    = padded_L - half_len

        def _pad_block(x):
            if pad_len > 0:
                x = F.pad(x, (0, 0, 0, pad_len))
            return x.view(B, H, num_blocks, S, D)

        q_noisy = _pad_block(q_noisy)
        q_clean = _pad_block(q_clean)
        k_noisy = _pad_block(k_noisy)
        k_clean = _pad_block(k_clean)
        v_noisy = _pad_block(v_noisy)
        v_clean = _pad_block(v_clean)

        # ── Feature maps (phi) for linear stream ─────────────────────────────
        phi_q_noisy = self._feature_map(q_noisy)   # (B, H, N, S, 2F)
        phi_q_clean = self._feature_map(q_clean)
        phi_k_clean = self._feature_map(k_clean)
        D_feat = 2 * self.feature_dim

        # ── Apply key_padding_mask to clean side only ─────────────────────────
        if key_padding_mask is not None:
            kpm = key_padding_mask[:, half_len:].float()   # (B, L)
            if kpm.shape[-1] < padded_L:
                kpm = F.pad(kpm, (0, padded_L - kpm.shape[-1]), value=0.0)
            kv_valid = kpm.view(B, 1, num_blocks, S, 1).to(
                device=phi_k_clean.device, dtype=phi_k_clean.dtype
            )
            phi_k_clean = phi_k_clean * kv_valid
            v_clean     = v_clean     * kv_valid.to(v_clean.dtype)

        # ── Zero out within-block pad positions ───────────────────────────────
        if pad_len > 0:
            valid_1d = torch.ones(padded_L, device=phi_k_clean.device, dtype=phi_k_clean.dtype)
            valid_1d[half_len:] = 0.0
            pad_mask_blk = valid_1d.view(1, 1, num_blocks, S, 1)
            phi_k_clean = phi_k_clean * pad_mask_blk
            v_clean     = v_clean     * pad_mask_blk.to(v_clean.dtype)

        # ── Recurrent state (fp32) ────────────────────────────────────────────
        S_state = torch.zeros(B, H, D_feat, D, dtype=torch.float32, device=q_noisy.device)
        Z_state = torch.zeros(B, H, D_feat,    dtype=torch.float32, device=q_noisy.device)

        w = torch.sigmoid(self.alpha)   # mixing weight

        out_noisy_blocks = []
        out_clean_blocks = []

        for n in range(num_blocks):
            q_n_noisy = q_noisy[:, :, n]       # (B, H, S, D)
            k_n_noisy = k_noisy[:, :, n]
            v_n_noisy = v_noisy[:, :, n]
            q_n_clean = q_clean[:, :, n]
            k_n_clean = k_clean[:, :, n]
            v_n_clean = v_clean[:, :, n]
            phi_qn_noisy = phi_q_noisy[:, :, n]   # (B, H, S, 2F)
            phi_qn_clean = phi_q_clean[:, :, n]
            phi_kn_clean = phi_k_clean[:, :, n]

            # ── Step A: noisy block queries pre-update state ──────────────────
            denom_n = torch.matmul(phi_qn_noisy.float(), Z_state.unsqueeze(-1)).clamp_min(self.eps)
            linear_noisy = (
                torch.matmul(phi_qn_noisy.float(), S_state) / denom_n
            ).to(out_dtype)                     # (B, H, S, D)

            # Intra-block softmax over noisy tokens only (bidirectional, exact).
            softmax_noisy = F.scaled_dot_product_attention(
                q_n_noisy, k_n_noisy, v_n_noisy,
                attn_mask=None, dropout_p=0.0, scale=self.scaling,
            )                                   # (B, H, S, D)

            out_noisy_blocks.append(w * softmax_noisy + (1.0 - w) * linear_noisy)

            # ── Step B: update state with clean block n ───────────────────────
            S_state = S_state + torch.matmul(
                phi_kn_clean.transpose(-2, -1).float(), v_n_clean.float()
            )
            Z_state = Z_state + phi_kn_clean.float().sum(dim=-2)

            # ── Step C: clean block queries post-update state ─────────────────
            denom_c = torch.matmul(phi_qn_clean.float(), Z_state.unsqueeze(-1)).clamp_min(self.eps)
            linear_clean = (
                torch.matmul(phi_qn_clean.float(), S_state) / denom_c
            ).to(out_dtype)                     # (B, H, S, D)

            # Intra-block softmax over clean tokens only (bidirectional, exact).
            softmax_clean = F.scaled_dot_product_attention(
                q_n_clean, k_n_clean, v_n_clean,
                attn_mask=None, dropout_p=0.0, scale=self.scaling,
            )                                   # (B, H, S, D)

            out_clean_blocks.append(w * softmax_clean + (1.0 - w) * linear_clean)

        # ── Reconstruct and return (B, H, 2L, D) ─────────────────────────────
        out_n = torch.stack(out_noisy_blocks, dim=2).reshape(B, H, padded_L, D)
        out_c = torch.stack(out_clean_blocks, dim=2).reshape(B, H, padded_L, D)
        if pad_len > 0:
            out_n = out_n[:, :, :half_len, :]
            out_c = out_c[:, :, :half_len, :]

        return torch.cat([out_n, out_c], dim=2)   # (B, H, 2L, D)
