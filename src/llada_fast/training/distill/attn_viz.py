"""
Attention matrix visualization for LLaDAFast distillation.

Saves side-by-side teacher vs. student attention heatmaps every N steps.

Teacher: softmax attention weights (B, H, L, L) captured via output_attentions=True.
Student: for linear attention, the "effective" attention kernel matrix is computed as
         the normalized kernel scores phi_q_i^T phi_k_j / (phi_q_i^T Z_i), where Z_i
         is the cumulative key sum up to (but not including) position i (causal prefix).
"""

import os
from typing import List, Optional

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for cluster training)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Teacher: capture softmax attention weights ────────────────────────────────

def _get_teacher_attn_weights(
    teacher,
    input_ids: torch.Tensor,           # (1, L)
    attention_mask_4d: torch.Tensor,   # (1, 1, L, L)
    key_padding_mask: torch.Tensor,    # (1, L)
    layer_ids: List[int],
    device: torch.device,
) -> dict:
    """
    Run a teacher forward with output_attentions=True.
    Returns {layer_id: attn_weights (H, L, L)} (first batch element, cpu).
    """
    store = {}
    handles = []

    for li in layer_ids:
        def _hook(module, inp, out, _li=li):
            # out = (attn_output, attn_weights, past_kv)
            if isinstance(out, (tuple, list)) and len(out) > 1 and out[1] is not None:
                store[_li] = out[1][0].detach().cpu()   # (H, L, L)
        h = teacher.model.layers[li].attention.register_forward_hook(_hook)
        handles.append(h)

    with torch.no_grad():
        teacher.model(
            input_ids.to(device),
            attention_mask=attention_mask_4d.to(device),
            key_padding_mask=key_padding_mask.to(device).bool(),
            output_attentions=True,
        )

    for h in handles:
        h.remove()

    return store


# ── Student: compute effective linear-attention kernel matrix ─────────────────

def _compute_student_kernel_matrix(
    student,
    input_ids: torch.Tensor,
    attention_mask_4d: torch.Tensor,
    key_padding_mask: torch.Tensor,
    layer_ids: List[int],
    device: torch.device,
) -> dict:
    """
    For each active linear attention layer, computes the L×L effective attention
    kernel matrix:
        A[i, j] = phi_q_i^T phi_k_j / (phi_q_i^T Z_i)
    where Z_i = cumulative sum of phi_k_{0..i-1} (causal prefix, excluding self).

    Returns {layer_id: attn_matrix (H, L, L)} on cpu.
    """
    store = {}
    handles = []

    for li in layer_ids:
        attn_mod = student.model.layers[li].attention
        if not (getattr(attn_mod, "is_linear_active", False) and hasattr(attn_mod, "linear_attention")):
            continue
        lin_attn = attn_mod.linear_attention

        def _pre_hook(module, args, _li=li):
            if len(args) < 2:
                return
            q, k = args[0], args[1]
            with torch.no_grad():
                B, H, L_in, D = q.shape
                num_blocks = (L_in + module.block_size - 1) // module.block_size
                padded_L   = num_blocks * module.block_size
                pad_len    = padded_L - L_in
                S          = module.block_size

                q_p = F.pad(q, (0, 0, 0, pad_len)) if pad_len > 0 else q
                k_p = F.pad(k, (0, 0, 0, pad_len)) if pad_len > 0 else k

                # Reshape to (B, H, N, S, D) as expected by _feature_map
                q_b = q_p.view(B, H, num_blocks, S, D)
                k_b = k_p.view(B, H, num_blocks, S, D)

                # Feature maps — handle both module signatures:
                #   OrderInvariantKernelLinearAttention._feature_map(x, compute_dtype)
                #   BlockSoftmaxLinearHybrid._feature_map(x)
                try:
                    phi_q = module._feature_map(q_b, q.dtype)
                    phi_k = module._feature_map(k_b, k.dtype)
                except TypeError:
                    phi_q = module._feature_map(q_b)
                    phi_k = module._feature_map(k_b)

                Df = phi_q.shape[-1]
                phi_q_flat = phi_q.reshape(B, H, padded_L, Df).float()
                phi_k_flat = phi_k.reshape(B, H, padded_L, Df).float()

                # ── Numerator: full (i, j) kernel scores ──────────────────────
                # K_num[i, j] = phi_q_i · phi_k_j
                K_num = torch.matmul(phi_q_flat, phi_k_flat.transpose(-1, -2))  # (B,H,L,L)

                # ── Denominator: per query position (i,) ──────────────────────
                # Z_causal[i] = sum_{k=0}^{i-1} phi_k_k  (prefix, excludes self)
                Z_cum    = phi_k_flat.cumsum(dim=2)                  # (B, H, padded_L, Df)
                Z_causal = torch.cat([
                    torch.zeros(B, H, 1, Df, device=Z_cum.device),
                    Z_cum[:, :, :-1, :]
                ], dim=2)                                             # (B, H, padded_L, Df)

                # denom_q[i] = phi_q_i · Z_causal_i  →  (B, H, padded_L, 1)
                denom_q = (phi_q_flat * Z_causal).sum(dim=-1, keepdim=True).clamp_min(1e-6)

                # ── Causal mask ────────────────────────────────────────────────
                causal = torch.tril(torch.ones(padded_L, padded_L, device=q.device))
                K_num  = K_num * causal

                # ── Normalize and crop ─────────────────────────────────────────
                A = (K_num / denom_q)[:, :, :L_in, :L_in]    # (B, H, L_in, L_in)
                A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-6)

                store[_li] = A[0].detach().cpu()   # (H, L_in, L_in)

        h = lin_attn.register_forward_pre_hook(_pre_hook)
        handles.append(h)

    with torch.no_grad():
        student.model(
            input_ids.to(device),
            attention_mask=attention_mask_4d.to(device),
            key_padding_mask=key_padding_mask.to(device).bool(),
        )

    for h in handles:
        h.remove()

    return store


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_and_save_attention(
    teacher_weights: dict,      # {layer_id: (H, L, L)}
    student_weights: dict,      # {layer_id: (H, L, L)}
    layer_ids: List[int],
    step: int,
    out_dir: str,
    max_len: int = 128,         # truncate to this many tokens for readability
) -> None:
    """
    Saves a PNG grid: rows = layers, cols = [teacher_avg, student_avg, diff].
    """
    os.makedirs(out_dir, exist_ok=True)

    n_layers = len(layer_ids)
    if n_layers == 0:
        return

    fig = plt.figure(figsize=(15, 4 * n_layers))
    gs = gridspec.GridSpec(n_layers, 3, figure=fig, hspace=0.4, wspace=0.3)

    for row, li in enumerate(sorted(layer_ids)):
        t_w = teacher_weights.get(li)   # (H, L, L) or None
        s_w = student_weights.get(li)   # (H, L, L) or None

        # Average over heads and truncate
        def _prep(w):
            if w is None:
                return None
            L = min(w.shape[-1], max_len)
            return w.float().mean(0)[:L, :L].numpy()

        t_mat = _prep(t_w)
        s_mat = _prep(s_w)

        # Teacher
        ax_t = fig.add_subplot(gs[row, 0])
        if t_mat is not None:
            im = ax_t.imshow(t_mat, aspect="auto", cmap="viridis", vmin=0)
            plt.colorbar(im, ax=ax_t, fraction=0.03)
        else:
            ax_t.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax_t.transAxes)
        ax_t.set_title(f"L{li} Teacher (step {step})", fontsize=8)
        ax_t.set_xlabel("Key pos"); ax_t.set_ylabel("Query pos")

        # Student
        ax_s = fig.add_subplot(gs[row, 1])
        if s_mat is not None:
            im = ax_s.imshow(s_mat, aspect="auto", cmap="viridis", vmin=0)
            plt.colorbar(im, ax=ax_s, fraction=0.03)
        else:
            ax_s.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax_s.transAxes)
        ax_s.set_title(f"L{li} Student kernel (step {step})", fontsize=8)
        ax_s.set_xlabel("Key pos"); ax_s.set_ylabel("Query pos")

        # Difference
        ax_d = fig.add_subplot(gs[row, 2])
        if t_mat is not None and s_mat is not None:
            L = min(t_mat.shape[0], s_mat.shape[0])
            diff = t_mat[:L, :L] - s_mat[:L, :L]
            vmax = max(abs(diff.min()), abs(diff.max()), 1e-6)
            im = ax_d.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            plt.colorbar(im, ax=ax_d, fraction=0.03)
        else:
            ax_d.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax_d.transAxes)
        ax_d.set_title(f"L{li} Diff (T − S)", fontsize=8)
        ax_d.set_xlabel("Key pos"); ax_d.set_ylabel("Query pos")

    fig.suptitle(f"Attention Maps — Step {step}", fontsize=12, y=1.01)
    fpath = os.path.join(out_dir, f"attn_step_{step:07d}.png")
    fig.savefig(fpath, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"\n[VIZ] Saved attention map → {fpath}")
