import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# FLA ops (full-seq + recurrent)
# ----------------------------
try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )
    HAS_FLA = True
except Exception as e:
    HAS_FLA = False
    _FLA_IMPORT_ERR = e


# ============================================================
# 1) FLA-backed Gated DeltaNet scan:
#    - forward(): full-sequence chunk kernel (training/denoising)
#    - step(): single-token recurrent kernel (generation) with cached state
# ============================================================
class GatedDeltaNetScanFLA(nn.Module):
    """
    Full-sequence scan (chunked Triton) + single-step recurrent update.

    Conventions:
      - q,k: [B, T, H, K]
      - v,o: [B, T, H, V]
      - alpha,beta: [B, T, H]
      - state S: [B, H, V, K]
    """

    def __init__(self, d_model: int, num_heads: int, d_k: int, d_v: int):
        super().__init__()
        if not HAS_FLA:
            raise ImportError(
                f"FLA not available. Install flash-linear-attention or fla-core. "
                f"Original import error: {_FLA_IMPORT_ERR}"
            )

        self.H = int(num_heads)
        self.K = int(d_k)
        self.V = int(d_v)

        self.q_proj = nn.Linear(d_model, self.H * self.K, bias=False)
        self.k_proj = nn.Linear(d_model, self.H * self.K, bias=False)
        self.v_proj = nn.Linear(d_model, self.H * self.V, bias=False)

        self.alpha_proj = nn.Linear(d_model, self.H)  # decay gate
        self.beta_proj = nn.Linear(d_model, self.H)   # step size

    @staticmethod
    def _make_cu_seqlens(padding_mask: torch.Tensor) -> torch.Tensor:
        """
        padding_mask: [B,T] bool (True=valid)
        returns: cu_seqlens [B+1] int32 (FlashAttention style)
        Assumes right-padding.
        """
        lengths = padding_mask.sum(dim=1, dtype=torch.int32)  # [B]
        cu = torch.zeros((lengths.numel() + 1,), device=padding_mask.device, dtype=torch.int32)
        cu[1:] = torch.cumsum(lengths, dim=0)
        return cu

    def _project(self, x: torch.Tensor):
        """
        x: [B,T,d]
        returns:
          q,k: [B,T,H,K]
          v,o: [B,T,H,V]
          alpha,beta: [B,T,H]
        """
        B, T, _ = x.shape
        H, K, V = self.H, self.K, self.V

        q = self.q_proj(x).view(B, T, H, K)
        k = self.k_proj(x).view(B, T, H, K)
        v = self.v_proj(x).view(B, T, H, V)

        # Stabilization: L2 norm keys/queries
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        alpha = torch.sigmoid(self.alpha_proj(x)).view(B, T, H)              # (0,1)
        beta = F.softplus(self.beta_proj(x)).view(B, T, H) * (K ** -0.5)     # >=0 scaled

        return q, k, v, alpha, beta

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
    ):
        """
        Full-sequence path (training / denoising / full-context evaluation).

        x: [B,T,d_model]
        padding_mask: [B,T] bool True=valid (optional)
        initial_state: [B,H,V,K] (optional)
        output_final_state: bool

        Returns:
          - out: [B,T,H,V]
          - maybe final_state depending on FLA version when output_final_state=True
        """
        q, k, v, alpha, beta = self._project(x)
        cu_seqlens = None if padding_mask is None else self._make_cu_seqlens(padding_mask)

        # NOTE: signatures vary slightly across versions; this is the common pattern.
        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=alpha,
            beta=beta,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )

    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor | None,
    ):
        """
        One-token recurrent update (generation).

        x_t: [B, d] or [B,1,d]
        state: [B,H,V,K] or None

        Returns:
          o_t: [B,1,H,V]
          next_state: [B,H,V,K]
        """
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)  # [B,1,d]

        q, k, v, alpha, beta = self._project(x_t)  # [B,1,H,*]
        q1 = q[:, 0]         # [B,H,K]
        k1 = k[:, 0]         # [B,H,K]
        v1 = v[:, 0]         # [B,H,V]
        a1 = alpha[:, 0]     # [B,H]
        b1 = beta[:, 0]      # [B,H]

        # Recurrent kernel; returns (out, final_state) in many versions.
        # If your version returns only out, you must update state another way.
        out = fused_recurrent_gated_delta_rule(
            q=q1,
            k=k1,
            v=v1,
            g=a1,
            beta=b1,
            initial_state=state,
            output_final_state=True,
        )

        if isinstance(out, tuple) and len(out) >= 2:
            o1, next_state = out[0], out[1]
        else:
            raise RuntimeError(
                "Your fused_recurrent_gated_delta_rule did not return (o, next_state). "
                "Check your FLA version / signature."
            )

        # Ensure [B,1,H,V]
        if o1.dim() == 3:
            o1 = o1.unsqueeze(1)

        return o1, next_state


# ============================================================
# 2) Bidirectional layer (training/full-seq only)
#    NOTE: For autoregressive generation, you should use forward-only + cached state.
# ============================================================
class BidirectionalGatedDeltaNetLayerFLA(nn.Module):
    """
    Bidirectional wrapper using two full-seq scans:
      - forward scan on x
      - forward scan on reversed x
      - fuse
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        fuse_mode: str = "gate",
        dropout: float = 0.0,
        use_norm: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_norm = use_norm
        self.use_residual = use_residual

        if use_norm:
            self.norm = nn.RMSNorm(d_model)

        self.scan = GatedDeltaNetScanFLA(d_model, num_heads, d_k, d_v)

        out_dim = num_heads * d_v
        self.fuse_mode = fuse_mode
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if fuse_mode == "concat":
            self.out_proj = nn.Linear(2 * out_dim, d_model)
        elif fuse_mode == "gate":
            self.gamma_proj = nn.Linear(d_model, out_dim)
            self.out_proj = nn.Linear(out_dim, d_model)
        else:
            raise ValueError(f"Unknown fuse_mode: {fuse_mode}")

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None):
        residual = x
        x_norm = self.norm(x) if self.use_norm else x

        # forward direction
        o_fwd = self.scan(x_norm, padding_mask=padding_mask)         # [B,T,H,V] or (out, state)
        if isinstance(o_fwd, tuple):
            o_fwd = o_fwd[0]
        B, T, H, V = o_fwd.shape
        o_fwd = o_fwd.reshape(B, T, H * V)

        # backward direction (reverse input, scan, reverse output)
        x_rev = torch.flip(x_norm, dims=[1])
        mask_rev = torch.flip(padding_mask, dims=[1]) if padding_mask is not None else None

        o_bwd = self.scan(x_rev, padding_mask=mask_rev)
        if isinstance(o_bwd, tuple):
            o_bwd = o_bwd[0]
        o_bwd = torch.flip(o_bwd, dims=[1]).reshape(B, T, H * V)

        # fuse
        if self.fuse_mode == "concat":
            y = torch.cat([o_fwd, o_bwd], dim=-1)
            out = self.out_proj(y)
        else:
            gamma = torch.sigmoid(self.gamma_proj(x_norm))
            y = gamma * o_fwd + (1.0 - gamma) * o_bwd
            out = self.out_proj(y)

        out = self.dropout(out)
        if self.use_residual:
            return residual + out
        return out


# ============================================================
# 3) Causal (generation-friendly) layer:
#    Uses scan.step() with cached state for O(1) per token.
# ============================================================
class CausalGatedDeltaNetLayerFLA(nn.Module):
    """
    Forward-only Gated DeltaNet with state caching for generation.
    Use this for autoregressive generation. (Bidirectional doesn't apply.)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.0,
        use_norm: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_norm = use_norm
        self.use_residual = use_residual

        if use_norm:
            self.norm = nn.RMSNorm(d_model)

        self.scan = GatedDeltaNetScanFLA(d_model, num_heads, d_k, d_v)

        out_dim = num_heads * d_v
        self.out_proj = nn.Linear(out_dim, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward_full(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None):
        """
        Full-sequence forward (e.g., training).
        """
        residual = x
        x_norm = self.norm(x) if self.use_norm else x

        o = self.scan(x_norm, padding_mask=padding_mask)
        if isinstance(o, tuple):
            o = o[0]
        B, T, H, V = o.shape
        o = o.reshape(B, T, H * V)

        out = self.out_proj(o)
        out = self.dropout(out)
        if self.use_residual:
            return residual + out
        return out

    def forward_step(self, x_t: torch.Tensor, state: torch.Tensor | None):
        """
        One token step for generation.

        x_t: [B,d] or [B,1,d]
        state: [B,H,V,K] or None
        returns:
          y_t: [B,1,d]
          next_state: [B,H,V,K]
        """
        residual = x_t
        if x_t.dim() == 2:
            x_t_in = x_t.unsqueeze(1)
        else:
            x_t_in = x_t

        x_norm = self.norm(x_t_in) if self.use_norm else x_t_in

        o_t, next_state = self.scan.step(x_norm, state)  # o_t: [B,1,H,V]
        B, _, H, V = o_t.shape
        o_t = o_t.reshape(B, 1, H * V)

        out = self.out_proj(o_t)
        out = self.dropout(out)
        if self.use_residual:
            # residual should match [B,1,d]
            if residual.dim() == 2:
                residual = residual.unsqueeze(1)
            out = residual + out
        return out, next_state


# ============================================================
# 4) LLaDA wrappers
#    - Bidirectional wrapper for training/denoising
#    - Causal wrapper for generation with cached state
# ============================================================
class LLaDA2BidirectionalGatedDeltaNetLayerFLAWrapper(nn.Module):
    """
    Drop BidirectionalGatedDeltaNetLayerFLA into LLaDA2MoeDecoderLayer in place of attention.
    Use in training (full sequences). Not suitable for autoregressive generation.
    """

    def __init__(self, config, layer_idx=None):
        super().__init__()
        if not HAS_FLA:
            raise ImportError(
                "fla.ops.gated_delta_rule required. "
                f"Original import error: {_FLA_IMPORT_ERR}"
            )
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        dropout = getattr(config, "attention_dropout", 0.0)

        # disable norm/residual inside, outer layer handles it
        self.gdn = BidirectionalGatedDeltaNetLayerFLA(
            d_model=self.hidden_size,
            num_heads=self.num_heads,
            d_k=self.head_dim,
            d_v=self.head_dim,
            fuse_mode="gate",
            dropout=dropout,
            use_norm=False,
            use_residual=False,
        )

    @staticmethod
    def _infer_padding_mask(hidden_states, attention_mask, key_padding_mask):
        B, L, _ = hidden_states.shape
        padding_mask = None

        if key_padding_mask is not None:
            padding_mask = key_padding_mask.bool()[:, :L]
        elif attention_mask is not None and attention_mask.dim() == 4:
            attn = attention_mask[..., :L, :L]
            if torch.isinf(attn).any():
                attn_bool = torch.isfinite(attn)
            else:
                attn_bool = attn != 0
            padding_mask = attn_bool.any(dim=-2).squeeze(1)  # [B,L]
        return padding_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings=None,
        key_padding_mask: torch.Tensor | None = None,
        block_attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        padding_mask = self._infer_padding_mask(hidden_states, attention_mask, key_padding_mask)
        out = self.gdn(hidden_states, padding_mask=padding_mask)
        return out, None, None


class LLaDA2CausalGatedDeltaNetLayerFLAWrapper(nn.Module):
    """
    Forward-only version suitable for autoregressive generation using cached state.

    We store the fast matrix state in past_key_value:
      past_key_value = {"gdn_state": Tensor[B,H,V,K]}
    """

    def __init__(self, config, layer_idx=None):
        super().__init__()
        if not HAS_FLA:
            raise ImportError(
                "fla.ops.gated_delta_rule required. "
                f"Original import error: {_FLA_IMPORT_ERR}"
            )
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        dropout = getattr(config, "attention_dropout", 0.0)

        self.gdn = CausalGatedDeltaNetLayerFLA(
            d_model=self.hidden_size,
            num_heads=self.num_heads,
            d_k=self.head_dim,
            d_v=self.head_dim,
            dropout=dropout,
            use_norm=False,
            use_residual=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings=None,
        key_padding_mask: torch.Tensor | None = None,
        block_attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        B, L, _ = hidden_states.shape

        # If generating token-by-token, L is typically 1.
        # If L > 1 during prompt prefill, use full forward and return final state if needed.

        state = None
        if past_key_value is not None:
            # Accept dict or tuple-like; adjust as your infra expects
            if isinstance(past_key_value, dict):
                state = past_key_value.get("gdn_state", None)
            else:
                # If your framework uses a tuple, you can encode state at index 0, etc.
                state = past_key_value

        if L == 1 and use_cache:
            # Fast step
            out, next_state = self.gdn.forward_step(hidden_states, state)
            new_past = {"gdn_state": next_state}
            return out, None, new_past

        # Prefill / training path: full forward
        # (no bidir here; causal only)
        out = self.gdn.forward_full(hidden_states, padding_mask=None)

        # Optionally: if you want to produce a state after prefill, you can run the scan
        # with output_final_state=True in the underlying scan and return it here.
        # That requires plumbing through chunk_gated_delta_rule returning final_state
        # in your installed FLA version.
        return out, None, past_key_value


# ============================================================
# 5) Hybrid (local softmax + global GDN) you included, unchanged
#    (Generation would still need careful caching; this is training-oriented.)
# ============================================================
class LLaDA2HybridLinearSoftmaxFLAWrapper(nn.Module):
    """
    Hybrid Attention:
    - Intra-block: exact softmax attention
    - Inter-block: gated delta rule (FLA)
    NOTE: This forward is full-sequence oriented.
    """

    def __init__(self, config, layer_idx=None):
        super().__init__()
        if not HAS_FLA:
            raise ImportError(
                "fla.ops.gated_delta_rule required. "
                f"Original import error: {_FLA_IMPORT_ERR}"
            )

        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.block_size = int(getattr(config, "block_size", 32))

        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        self.alpha_proj = nn.Linear(self.hidden_size, self.num_heads)
        self.beta_proj = nn.Linear(self.hidden_size, self.num_heads)

        # Plain Linear inter-block params (ELU + 1.0 feature map)
        self.phi_scale = nn.Parameter(torch.ones(self.num_heads, 1, self.head_dim))
        self.phi_bias = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))
        self.eps = 1e-6

        self.dense = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size,
            bias=getattr(config, "use_bias", True),
        )
        self.dropout = nn.Dropout(getattr(config, "attention_dropout", 0.0))

    def _local_softmax_attention(self, q, k, v, block_size):
        """
        q, k, v: [B, H, T, D]
        returns: [B, H, T, D]
        """
        B, H, T, D = q.shape
        num_blocks = (T + block_size - 1) // block_size
        pad_len = num_blocks * block_size - T

        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))

        q_b = q.view(B, H, num_blocks, block_size, D)
        k_b = k.view(B, H, num_blocks, block_size, D)
        v_b = v.view(B, H, num_blocks, block_size, D)

        scores = torch.matmul(q_b, k_b.transpose(-2, -1)) * self.scaling
        attn_weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        attn_weights = self.dropout(attn_weights)
        local_out = torch.matmul(attn_weights, v_b)

        local_out = local_out.view(B, H, num_blocks * block_size, D)
        if pad_len > 0:
            local_out = local_out[:, :, :T, :]
        return local_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings=None,
        key_padding_mask: torch.Tensor | None = None,
        block_attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        x = hidden_states
        B, T, _ = x.shape
        H, K = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, K)
        k = self.k_proj(x).view(B, T, H, K)
        v = self.v_proj(x).view(B, T, H, K)

        # Apply RoPE if provided (essential for teacher-student parity)
        if position_embeddings is not None:
            from .modeling_llada2_moe import apply_rotary_pos_emb
            q_pe, k_pe = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), *position_embeddings)
            q, k = q_pe.transpose(1, 2), k_pe.transpose(1, 2)

        # 1. Local Softmax (intra-block)
        local_out = self._local_softmax_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            self.block_size,
        ).transpose(1, 2)  # [B,T,H,K]

        # 2. Plain Linear (inter-block global)
        num_blocks = (T + self.block_size - 1) // self.block_size
        padded_L = num_blocks * self.block_size
        pad_len = padded_L - T

        # Map to blocks for recurrence
        q_l = q.transpose(1, 2)
        k_l = k.transpose(1, 2)
        v_l = v.transpose(1, 2)
        
        if pad_len > 0:
            q_l = F.pad(q_l, (0, 0, 0, pad_len))
            k_l = F.pad(k_l, (0, 0, 0, pad_len))
            v_l = F.pad(v_l, (0, 0, 0, pad_len))

        q_b = q_l.view(B, H, num_blocks, self.block_size, K)
        k_b = k_l.view(B, H, num_blocks, self.block_size, K)
        v_b = v_l.view(B, H, num_blocks, self.block_size, K)

        # Feature Map: ELU + 1
        phi_q = (F.elu(q_b * self.phi_scale.unsqueeze(1) + self.phi_bias.unsqueeze(1)) + 1.0).clamp(0.0, 10.0)
        phi_k = (F.elu(k_b * self.phi_scale.unsqueeze(1) + self.phi_bias.unsqueeze(1)) + 1.0).clamp(0.0, 10.0)

        # Linear Recurrence (O(D^2) state)
        S_state = torch.zeros(B, H, K, K, dtype=torch.float32, device=x.device)
        Z_state = torch.zeros(B, H, K, dtype=torch.float32, device=x.device)
        global_out_blocks = []

        for n in range(num_blocks):
            phi_k_n = phi_k[:, :, n]
            v_n     = v_b[:, :, n]
            phi_q_n = phi_q[:, :, n]

            # Update state with block n
            S_state = S_state + torch.matmul(phi_k_n.transpose(-2, -1).float(), v_n.float())
            Z_state = Z_state + phi_k_n.float().sum(dim=-2)

            # Compute global output for block n
            num = torch.matmul(phi_q_n.float(), S_state)
            den = torch.matmul(phi_q_n.float(), Z_state.unsqueeze(-1)).clamp_min(self.eps)
            global_out_blocks.append((num / den).to(x.dtype))

        global_out = torch.stack(global_out_blocks, dim=2).reshape(B, H, padded_L, K)
        if pad_len > 0:
            global_out = global_out[:, :, :T, :]
        global_out = global_out.transpose(1, 2) # [B,T,H,K]

        # Combine
        hybrid_out = local_out + global_out
        hybrid_out = hybrid_out.reshape(B, T, H * K)
        out = self.dense(hybrid_out)

        return out, None, None