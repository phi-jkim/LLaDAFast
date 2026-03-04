# llada_fast/modeling/linear_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrderInvariantKernelLinearAttention(nn.Module):
    """
    Block-causal, bidirectional-within-block kernel linear attention.
    Implemented with the Hedgehog feature map (LoLCATs style).

    Contract with LLaDA2MoeAttention:
      - query/key/value: (B, H, L, D)
      - attention_mask (optional): (B, 1, L, L) float/bool, usually 0/1 block mask
      - key_padding_mask (optional): (B, L) bool/0-1 float (1/True = real token)
    """

    def __init__(self, config, block_size: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.hidden_size = int(config.hidden_size)
        assert self.hidden_size % self.num_heads == 0
        self.head_dim = self.hidden_size // self.num_heads
        self.block_size = int(block_size or getattr(config, "block_size", 32))
        self.eps = float(eps)

        # Hedgehog feature map params
        self.feature_dim = int(config.feature_dim)
        # Learnable per-head projection: (H, D, F), initialized as truncated identity
        weight = torch.eye(self.head_dim, self.feature_dim).unsqueeze(0).expand(
            self.num_heads, -1, -1
        )
        self.hedgehog_weights = nn.Parameter(weight.clone())

    @staticmethod
    def _infer_key_valid_from_4d_mask(attention_mask: torch.Tensor, padded_L: int) -> torch.Tensor:
        """
        Returns (B, 1, padded_L, 1) float mask in {0,1}.
        """
        attn = attention_mask[..., :padded_L, :padded_L]

        if attn.dtype == torch.bool:
            attn_bool = attn
        else:
            if torch.isinf(attn).any():
                attn_bool = torch.isfinite(attn)
            else:
                attn_bool = attn != 0

        key_valid = attn_bool.any(dim=-2).float()  # (B, 1, padded_L)

        if key_valid.shape[-1] < padded_L:
            key_valid = F.pad(key_valid, (0, padded_L - key_valid.shape[-1]), value=0.0)

        return key_valid.unsqueeze(-1)  # (B, 1, padded_L, 1)

    def _feature_map(self, x: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
        """
        Hedgehog feature map:
        1. u = x @ W
        2. phi(x) = [softmax(u), softmax(-u)]
        """
        # x: (B, H, N, S, D)
        weights = self.hedgehog_weights.to(dtype=compute_dtype)
        x = x.to(dtype=compute_dtype)

        # Linear projection: (B, H, N, S, D) @ (H, D, F) -> (B, H, N, S, F)
        u = torch.einsum("bhnsd,hdf->bhnsf", x, weights)

        # Activation: concat(softmax(u), softmax(-u)) over the feature dimension
        # Use float32 for softmax stability
        u_f32 = u.float()
        phi = torch.cat(
            [F.softmax(u_f32, dim=-1), F.softmax(-u_f32, dim=-1)], dim=-1
        ).to(dtype=compute_dtype)

        return phi

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        query/key/value: (B, H, L, D)
        Returns: (B, H, L, D) in original dtype of query_states
        """
        B, H, L, D = query_states.shape
        assert H == self.num_heads and D == self.head_dim

        out_dtype = query_states.dtype
        compute_dtype = out_dtype

        num_blocks = (L + self.block_size - 1) // self.block_size
        padded_L = num_blocks * self.block_size
        pad_len = padded_L - L

        q = query_states
        k = key_states
        v = value_states

        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))

        # (B, H, N, S, D)
        q = q.view(B, H, num_blocks, self.block_size, D)
        k = k.view(B, H, num_blocks, self.block_size, D)
        v = v.view(B, H, num_blocks, self.block_size, D)

        phi_q = self._feature_map(q, compute_dtype)  # (B, H, N, S, 2*F)
        phi_k = self._feature_map(k, compute_dtype)  # (B, H, N, S, 2*F)
        v = v.to(dtype=compute_dtype)                # (B, H, N, S, D)

        D_feat = 2 * self.feature_dim

        # key-valid mask priority: key_padding_mask > infer from 4D attention_mask > None
        key_valid = None
        if key_padding_mask is not None:
            kpm = key_padding_mask
            if kpm.dtype == torch.bool:
                kpm = kpm.float()
            else:
                kpm = kpm.to(dtype=torch.float32)
            kpm = kpm[:, :padded_L]
            if kpm.shape[-1] < padded_L:
                kpm = F.pad(kpm, (0, padded_L - kpm.shape[-1]), value=0.0)
            key_valid = kpm.unsqueeze(1).unsqueeze(-1)  # (B,1,padded_L,1)
        elif attention_mask is not None:
            key_valid = OrderInvariantKernelLinearAttention._infer_key_valid_from_4d_mask(
                attention_mask, padded_L
            )

        if key_valid is not None:
            key_valid = key_valid.to(device=phi_k.device, dtype=compute_dtype)
            key_valid_blocks = key_valid.view(B, 1, num_blocks, self.block_size, 1)
            phi_k = phi_k * key_valid_blocks
            v = v * key_valid_blocks

        # Correctness: mask internal pad-to-block padding
        if pad_len > 0:
            valid_1d = torch.ones(padded_L, device=phi_k.device, dtype=compute_dtype)
            valid_1d[L:] = 0
            pad_valid = valid_1d.view(1, 1, num_blocks, self.block_size, 1)
            phi_k = phi_k * pad_valid
            v = v * pad_valid

        # Recurrent block-causal computation with O(D_feat * D) state
        S_state = torch.zeros(B, H, D_feat, D, dtype=torch.float32, device=phi_k.device)
        Z_state = torch.zeros(B, H, D_feat, dtype=torch.float32, device=phi_k.device)

        out_blocks = []
        for n in range(num_blocks):
            # (B,H,S,D_feat)
            phi_k_n = phi_k[:, :, n]
            v_n = v[:, :, n]
            phi_q_n = phi_q[:, :, n]

            # --- update states in fp32 ---
            # S += (D_feat,S) @ (S,D) => (D_feat,D)
            S_state = S_state + torch.matmul(
                phi_k_n.transpose(-2, -1).float(), v_n.float()
            )
            # Z += sum over S
            Z_state = Z_state + phi_k_n.float().sum(dim=-2)

            # --- compute output ---
            num = torch.matmul(phi_q_n.float(), S_state)  # (B,H,S,D)
            den = torch.matmul(phi_q_n.float(), Z_state.unsqueeze(-1))  # (B,H,S,1)
            den = den.clamp_min(self.eps)

            block_out = (num / den).to(out_dtype)
            out_blocks.append(block_out)

        out = torch.stack(out_blocks, dim=2).reshape(B, H, padded_L, D)
        if pad_len > 0:
            out = out[:, :, :L, :]

        return out

