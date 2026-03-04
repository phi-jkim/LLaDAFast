"""
Attention matrix visualization for LLaDAFast distillation.

Saves two PNGs per visualization step:
  teacher_step_XXXXXXX.png  —  all N layers, softmax attention (head-averaged)
  student_step_XXXXXXX.png  —  all is_linear_active layers, linear kernel matrix
                                computed on a "clean past, noisy present" input

Teacher: softmax attn_weights (B, H, L, L) captured via forward hook.
Student: effective linear-attention kernel A[i,j] = phi_q_i·phi_k_j / (phi_q_i·Z_i)
         where Z_i is the causal prefix sum of phi_k up to (but not including) i.
"""

import os
import random
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mask_to_additive(mask_0_1: torch.Tensor) -> torch.Tensor:
    """Convert 0/1 block-causal mask → additive (-1e9 / 0.0) for softmax."""
    return (1.0 - mask_0_1.float()) * -1e9


def _capture_teacher_attn(
    teacher,
    input_ids: torch.Tensor,
    attn_mask_0_1: torch.Tensor,
    key_padding_mask: torch.Tensor,
    layer_ids: List[int],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """
    Returns {li: (H, L, L) float32 cpu tensor} for every requested layer.
    Uses 'output_attentions=True' to trigger weight computation in the model.
    Note: Pass the RAW 0/1 mask; LLaDA2MoeModel.forward converts it to additive.
    """
    store: Dict[int, torch.Tensor] = {}
    handles = []

    for li in layer_ids:
        def _hook(module, inp, out, _li=li):
            try:
                # out is (attn_output, attn_weights, past_key_value)
                w = out[1] if (isinstance(out, (tuple, list)) and len(out) > 1) else None
                if w is not None and isinstance(w, torch.Tensor) and w.dim() == 4:
                    w_cpu = w[0].detach().float().cpu()
                    # Softmax weights should be non-negative.
                    if w_cpu.min() >= -0.1: 
                        store[_li] = w_cpu
            except Exception:
                pass
        h = teacher.model.layers[li].attention.register_forward_hook(_hook)
        handles.append(h)

    try:
        with torch.no_grad():
            teacher.model(
                input_ids.to(device),
                attention_mask=attn_mask_0_1.to(device),
                key_padding_mask=key_padding_mask.to(device).bool(),
                output_attentions=True,
            )
    finally:
        for h in handles:
            h.remove()

    missing = [li for li in layer_ids if li not in store]
    if missing:
        print(f"[VIZ] Warning: teacher attn N/A for layers {missing}. "
              f"Check model configuration or output_attentions support.")
    return store


def _compute_student_kernel(
    student,
    input_ids: torch.Tensor,           # (1, L)  noisy (clean-past/noisy-present)
    attn_mask_4d: torch.Tensor,        # (1, 1, L, L)  0/1 block-causal
    key_padding_mask: torch.Tensor,    # (1, L)
    layer_ids: List[int],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """
    For every is_linear_active layer in layer_ids, computes the L×L effective
    linear-attention kernel matrix:
        A[i, j] = phi_q_i · phi_k_j / (phi_q_i · Z_i)   (causal, normalized)

    Returns {li: (H, L, L) float32 cpu tensor}.
    """
    store: Dict[int, torch.Tensor] = {}
    handles = []

    for li in layer_ids:
        attn_mod = student.model.layers[li].attention
        if not hasattr(attn_mod, "linear_attention"):
            continue
        lin_attn = attn_mod.linear_attention

        def _pre_hook(module, args, _li=li):
            if len(args) < 2:
                return
            q, k = args[0], args[1]
            with torch.no_grad():
                B, H, L_in, D = q.shape
                S = module.block_size
                num_blocks = (L_in + S - 1) // S
                padded_L   = num_blocks * S
                pad_len    = padded_L - L_in

                q_p = F.pad(q, (0, 0, 0, pad_len)) if pad_len > 0 else q
                k_p = F.pad(k, (0, 0, 0, pad_len)) if pad_len > 0 else k
                q_b = q_p.view(B, H, num_blocks, S, D)
                k_b = k_p.view(B, H, num_blocks, S, D)

                # Handle both _feature_map(x, dtype) and _feature_map(x)
                try:
                    phi_q = module._feature_map(q_b, q.dtype)
                    phi_k = module._feature_map(k_b, k.dtype)
                except TypeError:
                    phi_q = module._feature_map(q_b)
                    phi_k = module._feature_map(k_b)

                Df = phi_q.shape[-1]
                phi_q_f = phi_q.reshape(B, H, padded_L, Df).float()
                phi_k_f = phi_k.reshape(B, H, padded_L, Df).float()

                # Numerator: phi_q_i · phi_k_j  →  (B, H, L, L)
                K_num = torch.matmul(phi_q_f, phi_k_f.transpose(-1, -2))

                # Use the block-causal mask from the model.
                mask = attn_mask_4d[0, 0, :padded_L, :padded_L].to(device=q.device).clone()
                
                # If in Hybrid mode, the linear part is purposefully blind to the current block
                # (handled by softmax instead). Mask out the diagonal blocks to show this.
                if lin_attn.__class__.__name__ == "BlockSoftmaxLinearHybrid":
                    S = lin_attn.block_size
                    b_idx_q = torch.arange(padded_L, device=q.device) // S
                    b_idx_k = torch.arange(padded_L, device=q.device) // S
                    mask *= (b_idx_k.unsqueeze(0) < b_idx_q.unsqueeze(1)).float()
                
                K_num = K_num.masked_fill(mask == 0, float("-inf"))

                # Softmax over keys (makes it directly comparable to teacher softmax)
                A = torch.softmax(K_num.float(), dim=-1)  # (B, H, L, L)
                A = A[:, :, :L_in, :L_in]

                store[_li] = A[0].detach().cpu()   # (H, L_in, L_in)

        h = lin_attn.register_forward_pre_hook(_pre_hook)
        handles.append(h)

    with torch.no_grad():
        student.model(
            input_ids.to(device),
            attention_mask=attn_mask_4d.to(device),
            key_padding_mask=key_padding_mask.to(device).bool(),
        )
    for h in handles:
        h.remove()
    return store


# ─────────────────────────────────────────────────────────────────────────────
# Public plotting API
# ─────────────────────────────────────────────────────────────────────────────

def _draw_heatmap(ax, mat, title: str, cmap: str = "viridis", vmin=0, vmax=None):
    """Draw a single heatmap on ax with a colorbar."""
    kw = dict(aspect="auto", cmap=cmap, vmin=vmin)
    if vmax is not None:
        kw["vmax"] = vmax
    im = ax.imshow(mat, **kw)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    ax.set_title(title, fontsize=7)
    ax.set_xlabel("Key", fontsize=6)
    ax.set_ylabel("Query", fontsize=6)
    ax.tick_params(labelsize=5)


def plot_teacher_all_layers(
    teacher,
    input_ids: torch.Tensor,
    attn_mask_0_1: torch.Tensor,
    key_padding_mask: torch.Tensor,
    n_layers: int,
    step: int,
    out_dir: str,
    max_len: int = 128,
    layers_per_page: int = 4,
    device: torch.device = torch.device("cuda:0"),
) -> None:
    """
    Saves teacher_step_XXXXXXX_p{page}.png per page of `layers_per_page` rows.
    """
    os.makedirs(out_dir, exist_ok=True)
    layer_ids = list(range(n_layers))
    # Pass 0/1 mask directly; teacher.model will convert to additive.
    weights = _capture_teacher_attn(teacher, input_ids, attn_mask_0_1,
                                     key_padding_mask, layer_ids, device)

    pages = [layer_ids[i:i+layers_per_page] for i in range(0, n_layers, layers_per_page)]
    for page_idx, page_layers in enumerate(pages):
        n = len(page_layers)
        # Use 2 columns for two random heads Side-by-Side
        fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n))
        if n == 1:
            axes = axes.reshape(1, 2)
            
        for row, li in enumerate(page_layers):
            w = weights.get(li)
            if w is not None:
                num_heads = w.shape[0]
                # Pick two random unique heads
                rng = random.Random(step + li)
                h1, h2 = rng.sample(range(num_heads), 2) if num_heads > 1 else (0, 0)
                
                L = min(w.shape[-1], max_len)
                for col, h_idx in enumerate([h1, h2]):
                    ax = axes[row, col]
                    mat = w[h_idx].float()[:L, :L].numpy()
                    _draw_heatmap(ax, mat, f"Layer {li} (Head {h_idx})")
            else:
                for col in range(2):
                    ax = axes[row, col]
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"Layer {li}", fontsize=7)
        layers_str = f"L{page_layers[0]}–{page_layers[-1]}"
        fig.suptitle(f"Teacher Attention ({layers_str}) — Step {step}", fontsize=10)
        plt.tight_layout()
        fpath = os.path.join(out_dir, f"teacher_step_{step:07d}_p{page_idx:02d}.png")
        fig.savefig(fpath, bbox_inches="tight", dpi=80)
        plt.close(fig)
        print(f"[VIZ] Teacher p{page_idx} → {fpath}")


def plot_student_all_layers(
    student,
    noisy_input_ids: torch.Tensor,
    attn_mask_0_1: torch.Tensor,
    key_padding_mask: torch.Tensor,
    n_layers: int,
    step: int,
    out_dir: str,
    max_len: int = 128,
    layers_per_page: int = 4,
    device: torch.device = torch.device("cuda:1"),
) -> None:
    """
    Saves student_step_XXXXXXX_p{page}.png per page of `layers_per_page` rows.
    Linear attention stays active; kernel scores phi_q·phi_k are softmax'd
    post-hoc so the display is a proper probability distribution comparable
    to the teacher softmax plot.
    Frozen (not-yet-active) layers show 'softmax frozen' placeholder.
    """
    os.makedirs(out_dir, exist_ok=True)
    layer_ids = list(range(n_layers))
    kernel = _compute_student_kernel(student, noisy_input_ids, attn_mask_0_1,
                                      key_padding_mask, layer_ids, device)

    pages = [layer_ids[i:i+layers_per_page] for i in range(0, n_layers, layers_per_page)]
    for page_idx, page_layers in enumerate(pages):
        n = len(page_layers)
        # Use 2 columns for two random heads Side-by-Side
        fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n))
        if n == 1:
            axes = axes.reshape(1, 2)
            
        for row, li in enumerate(page_layers):
            w = kernel.get(li)
            is_active = getattr(student.model.layers[li].attention, "is_linear_active", False)
            if w is not None:
                num_heads = w.shape[0]
                # Pick two random unique heads
                rng = random.Random(step + li + 99) # different seed than teacher
                h1, h2 = rng.sample(range(num_heads), 2) if num_heads > 1 else (0, 0)
                
                L = min(w.shape[-1], max_len)
                tag = "[kernel softmax]" if is_active else "[kernel frozen]"
                for col, h_idx in enumerate([h1, h2]):
                    ax = axes[row, col]
                    mat = w[h_idx].float()[:L, :L].numpy()
                    _draw_heatmap(ax, mat, f"Layer {li} (Head {h_idx}) {tag}")
            else:
                for col in range(2):
                    ax = axes[row, col]
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color="grey")
                    ax.set_title(f"Layer {li}", fontsize=7)
        layers_str = f"L{page_layers[0]}–{page_layers[-1]}"
        fig.suptitle(f"Student Kernel (softmax'd) ({layers_str}) — Step {step}", fontsize=10)
        plt.tight_layout()
        fpath = os.path.join(out_dir, f"student_step_{step:07d}_p{page_idx:02d}.png")
        fig.savefig(fpath, bbox_inches="tight", dpi=80)
        plt.close(fig)
        print(f"[VIZ] Student p{page_idx} → {fpath}")


def plot_loss_curve(
    loss_history: list,             # list of (step, loss, loss_m2t, loss_t2t, ...)
    seq_len_history: list,          # list of (step, avg_seq_len)
    out_dir: str,
    step: int,
    ppl_history: Optional[list] = None,  # list of (step, perplexity)
) -> None:
    """
    Saves loss_curve.png with subplots:
      1. Student masked-denoising perplexity (if ppl_history provided)
      2. Combined / M2T / T2T distillation loss
      3. Average effective sequence length per batch (if seq_len_history provided)
    """
    if not loss_history:
        return
    os.makedirs(out_dir, exist_ok=True)

    steps_l, losses, losses_m2t, losses_t2t = (
        zip(*loss_history) if loss_history else ([], [], [], [])
    )

    has_ppl = bool(ppl_history)
    has_len = bool(seq_len_history)
    n_plots = 1 + int(has_ppl) + int(has_len)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    ax_idx = 0

    # ── Perplexity (primary signal) ───────────────────────────────────────────
    if has_ppl:
        import math as _math
        steps_p, ppls = zip(*ppl_history)
        # Filter out non-finite values for a clean plot.
        valid = [(s, p) for s, p in zip(steps_p, ppls) if _math.isfinite(p)]
        if valid:
            vs, vp = zip(*valid)
            ax = axes[ax_idx]
            ax.plot(vs, vp, color="#e84c6a", linewidth=1.4, marker="o",
                    markersize=3, label="Student PPL")
            ax.set_ylabel("Perplexity")
            ax.set_title(f"Student Masked-Denoising Perplexity — Step {step}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        ax_idx += 1

    # ── Distillation loss curves ──────────────────────────────────────────────
    ax = axes[ax_idx]
    ax.plot(steps_l, losses,     label="Combined", color="#4c9be8", linewidth=1.2)
    ax.plot(steps_l, losses_m2t, label="M2T",      color="#e8874c", linewidth=0.8, alpha=0.8)
    ax.plot(steps_l, losses_t2t, label="T2T",      color="#6ae84c", linewidth=0.8, alpha=0.8)
    ax.set_ylabel("Loss")
    ax.set_title(f"Distillation Loss — Step {step}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax_idx += 1

    # ── Average sequence length ───────────────────────────────────────────────
    if has_len:
        steps_s, avg_lens = zip(*seq_len_history)
        ax = axes[ax_idx]
        ax.plot(steps_s, avg_lens, color="#a84ce8", linewidth=1.0)
        ax.set_ylabel("Avg non-pad tokens")
        ax.set_xlabel("Step")
        ax.set_title("Average Effective Sequence Length per Batch")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fpath = os.path.join(out_dir, "loss_curve.png")
    fig.savefig(fpath, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"\n[VIZ] Loss curve → {fpath} (step {step})")
