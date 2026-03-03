#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLaDAFast Stage-1 Distillation: LoLCATs Teacher Forcing

Trains a student model (linear / hybrid attention) to match the hidden-state
outputs of a frozen teacher (standard softmax attention) layer by layer.

Key paradigms
─────────────
1. Teacher Forcing: student attention outputs are replaced with the teacher's
   exact activations, preventing error accumulation across layers.  The forcing
   probability per layer decays from 1.0 → 0.0 over `force_decay_length` steps.

2. Clean Past, Noisy Present: one random block within each training sequence is
   corrupted with [MASK] tokens; all past blocks are kept clean.  This trains
   the student to handle the block-parallel decoding regime.

3. Loss: MSE on per-layer attention outputs (hidden-state matching) +
   KL divergence on final logits (output distribution matching).

4. Curriculum: layers are activated one at a time in a configurable order
   (default: middle-out).  Progression can be fixed-interval or LLM-judge-gated.
"""

import argparse
import glob
import os
import random
import shutil
import warnings
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from llada_fast.modeling.configuration_llada2_moe import LLaDA2MoeConfig
from llada_fast.modeling.modeling_llada2_moe import LLaDA2MoeModelLM
from .config import DistillConfig
from .curriculum import CurriculumManager, DEFAULT_PROG_SEQ
from .data import build_block_causal_mask, corrupt_one_block, corrupt_one_block_t2t, StreamingTextLoader
from .hooks import TeacherHooks, StudentHooks
from .llm_judge import evaluate_with_llm
from .attn_viz import plot_teacher_all_layers, plot_student_all_layers, plot_loss_curve

warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# ── Weight synchronisation ────────────────────────────────────────────────────

def sync_layer_weights(s_layer, t_layer, layer_idx: int, s_config, t_config,
                       skip_standard: bool = False) -> None:
    """
    Copy teacher weights into the student layer.

    For standard LLaDA2MoeAttention layers, load_state_dict handles it
    (strict=False allows the student's extra linear-attention params to survive).
    For non-standard layers (e.g. GatedDeltaNet), manually slice the fused QKV weight.
    """
    from llada_fast.modeling.modeling_llada2_moe import LLaDA2MoeAttention
    if isinstance(s_layer.attention, LLaDA2MoeAttention):
        if not skip_standard:
            s_layer.load_state_dict(t_layer.state_dict(), strict=False)
        return

    print(f"[INIT] Mapping weights for non-standard layer {layer_idx} ({type(s_layer.attention).__name__})")
    t_attn = t_layer.attention
    s_attn = s_layer.attention
    num_heads    = t_config.num_attention_heads
    num_kv_heads = t_config.num_key_value_heads
    head_dim     = t_config.head_dim
    hidden_size  = t_config.hidden_size

    if hasattr(t_attn, "query_key_value"):
        t_qkv = t_attn.query_key_value.weight.data
        q, k, v = t_qkv.split(
            [num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=0
        )

        def _set(container, name, data):
            if not hasattr(container, name):
                return
            proj = getattr(container, name)
            if proj.weight.shape != data.shape:
                if name in ("k_proj", "v_proj") and proj.weight.shape[0] > data.shape[0]:
                    n_rep = num_heads // num_kv_heads
                    data = (
                        data.view(num_kv_heads, head_dim, hidden_size)
                            .repeat_interleave(n_rep, dim=0)
                            .view(-1, hidden_size)
                    )
            proj.weight.data.copy_(data.to(proj.weight.device, dtype=proj.weight.dtype))

        if hasattr(s_attn, "q_proj"):
            _set(s_attn, "q_proj", q)
            _set(s_attn, "k_proj", k)
            _set(s_attn, "v_proj", v)
        elif hasattr(s_attn, "gdn"):
            scan = s_attn.gdn.scan
            _set(scan, "q_proj", q)
            _set(scan, "k_proj", k)
            _set(scan, "v_proj", v)

    if not skip_standard:
        s_layer.mlp.load_state_dict(t_layer.mlp.state_dict(), strict=False)
        s_layer.input_layernorm.load_state_dict(t_layer.input_layernorm.state_dict(), strict=False)
        s_layer.post_attention_layernorm.load_state_dict(
            t_layer.post_attention_layernorm.state_dict(), strict=False
        )


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_teacher(cfg: DistillConfig, device: torch.device) -> LLaDA2MoeModelLM:
    teacher = LLaDA2MoeModelLM.from_pretrained(
        cfg.teacher_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"[INIT] Teacher loaded on {device}")
    return teacher


def _build_student(
    cfg: DistillConfig, device: torch.device, teacher: LLaDA2MoeModelLM
) -> tuple[LLaDA2MoeModelLM, LLaDA2MoeConfig]:
    s_config = LLaDA2MoeConfig.from_pretrained(cfg.teacher_model_path, trust_remote_code=True)
    if cfg.use_gated_deltanet:
        s_config.use_bidirectional_gated_deltanet = True
    if cfg.use_hybrid_local_softmax:
        s_config.use_hybrid_local_softmax = True
    if cfg.use_block_softmax_hybrid:
        s_config.use_block_softmax_hybrid = True
    if cfg.block_size_override is not None:
        s_config.block_size = cfg.block_size_override

    s_config.use_linear_attention = True
    if cfg.progressive_interval > 0 or cfg.use_llm_curriculum_eval:
        s_config.linear_attention_layers = None     # gate activation dynamically
    elif cfg.linear_layers is not None:
        s_config.linear_attention_layers = cfg.linear_layers

    student = LLaDA2MoeModelLM(s_config).to(torch.bfloat16).to(device)
    print(f"[INIT] Student created on {device}. Syncing base weights...")
    ret = student.load_state_dict(teacher.state_dict(), strict=False)
    for i, (s_layer, t_layer) in enumerate(zip(student.model.layers, teacher.model.layers)):
        sync_layer_weights(s_layer, t_layer, i, s_config, teacher.config, skip_standard=True)
    print(f"[INIT] Weights synced. Missing: {len(ret.missing_keys)}, "
          f"Unexpected: {len(ret.unexpected_keys)}")
    return student, s_config


def _initial_active_layers(cfg: DistillConfig, n_layers: int, prog_seq: List[int]) -> List[int]:
    """Determine which layers are active at step 0 (before any resume)."""
    if cfg.joint or (not cfg.progressive_interval and not cfg.use_llm_curriculum_eval):
        return cfg.linear_layers if cfg.linear_layers is not None else list(range(n_layers))
    return [prog_seq[0]]


def _apply_active_layers(
    student: LLaDA2MoeModelLM,
    active_layers: List[int],
    hybrid_mode: bool = False,
) -> None:
    """
    Set requires_grad and is_linear_active for the given active layers.

    hybrid_mode (use_block_softmax_hybrid):
      Only the linear_attention sub-module's parameters are trainable
      (hedgehog_weights + alpha).  Q/K/V projections stay frozen so the
      softmax path uses the exact teacher initialization throughout stage-1.

    default (pure linear attention):
      All parameters inside the attention module are trainable.
    """
    active_set = set(active_layers)
    # Reset all layers — prevents inactive layers from running linear attention.
    for li, layer in enumerate(student.model.layers):
        layer.attention.is_linear_active = (li in active_set)

    if hybrid_mode:
        # Train only the kernel feature map (hedgehog_weights) and alpha mixer.
        prefixes = [f"model.layers.{li}.attention.linear_attention." for li in active_layers]
    else:
        prefixes = [f"model.layers.{li}.attention." for li in active_layers]

    for name, p in student.named_parameters():
        p.requires_grad = any(name.startswith(pref) for pref in prefixes)


def _build_optimizer(
    cfg: DistillConfig, student: LLaDA2MoeModelLM
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        fused=True if torch.cuda.is_available() else False,
    )


def _build_scheduler(
    cfg: DistillConfig, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.lr_factor,
        patience=cfg.lr_patience,
        min_lr=cfg.min_lr,
        verbose=True,
    )


# ── Checkpoint save / resume ──────────────────────────────────────────────────

def _save_checkpoint(
    student: LLaDA2MoeModelLM,
    optimizer: torch.optim.Optimizer,
    curriculum: CurriculumManager,
    tokenizer,
    step: int,
    cfg: DistillConfig,
) -> None:
    parent_dir = cfg.output_dir
    os.makedirs(parent_dir, exist_ok=True)
    
    # Keep only the last two checkpoints (the most recent one + the one we are about to save)
    pths = sorted(glob.glob(os.path.join(parent_dir, "step_*")), 
                  key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0)
    # If we have 2 or more, delete all but the most recent one.
    if len(pths) >= 2:
        for i in range(len(pths) - 1):
            try:
                shutil.rmtree(pths[i])
                print(f"\n[SAVE] Removed old checkpoint: {pths[i]}")
            except Exception as e:
                print(f"[SAVE] Could not remove {pths[i]}: {e}")

    out_dir = os.path.join(parent_dir, f"step_{step}")
    os.makedirs(out_dir, exist_ok=True)
    delta = {n: p.cpu() for n, p in student.named_parameters() if p.requires_grad}
    torch.save(delta, os.path.join(out_dir, "linear_attn_delta.pt"))
    torch.save(optimizer.state_dict(), os.path.join(out_dir, "optimizer.pt"))
    torch.save(
        {
            "step": step,
            "active_layers": curriculum.state.active_layers,
            "prog_seq_cursor": curriculum.state.prog_seq_cursor,
            "hybrid_ratio_step": curriculum.state.hybrid_ratio_step,
        },
        os.path.join(out_dir, "train_state.pt"),
    )
    tokenizer.save_pretrained(out_dir)
    print(f"\n[SAVE] Saved delta ({len(delta)} tensors) to {out_dir}")


def _save_final(student: LLaDA2MoeModelLM, tokenizer, cfg: DistillConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    delta = {n: p.cpu() for n, p in student.named_parameters() if p.requires_grad}
    torch.save(delta, os.path.join(cfg.output_dir, "linear_attn_delta.pt"))
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"\nDistillation complete. Saved delta ({len(delta)} tensors) to {cfg.output_dir}")


def _resume(
    cfg: DistillConfig,
    student: LLaDA2MoeModelLM,
    curriculum: CurriculumManager,
    dev1: torch.device,
) -> tuple[int, Optional[torch.optim.Optimizer]]:
    """
    Load a checkpoint.  Returns (start_step, new_optimizer_or_None).
    A new optimizer is returned when the saved checkpoint has multiple param groups
    (progressive mode) that require a matching group structure.
    """
    if not cfg.resume_from or not os.path.exists(cfg.resume_from):
        return 0, None

    print(f"\n[RESUME] Loading from {cfg.resume_from}...")
    ts_path = os.path.join(cfg.resume_from, "train_state.pt")
    step = 0
    if os.path.exists(ts_path):
        ts = torch.load(ts_path, map_location="cpu")
        step = ts.get("step", 0)
        curriculum.state.active_layers    = ts.get("active_layers", curriculum.state.active_layers)
        curriculum.state.prog_seq_cursor  = ts.get("prog_seq_cursor", curriculum.state.prog_seq_cursor)
        curriculum.state.hybrid_ratio_step = ts.get("hybrid_ratio_step", 0)
        curriculum.state.already_in_optimizer = set(curriculum.state.active_layers)
        print(f"[RESUME] Step={step}, ActiveLayers={curriculum.state.active_layers}")

    delta_path = os.path.join(cfg.resume_from, "linear_attn_delta.pt")
    if os.path.exists(delta_path):
        ret = student.load_state_dict(torch.load(delta_path, map_location="cpu"), strict=False)
        print(f"[RESUME] Loaded delta. Missing: {len(ret.missing_keys)}")

    # Apply resumed active layers now so the optimizer sees the correct requires_grad flags.
    _apply_active_layers(student, curriculum.state.active_layers, hybrid_mode=cfg.use_block_softmax_hybrid)

    opt_path = os.path.join(cfg.resume_from, "optimizer.pt")
    if not os.path.exists(opt_path):
        return step, None

    saved_opt   = torch.load(opt_path, map_location="cpu")
    n_groups    = len(saved_opt["param_groups"])
    active      = curriculum.state.active_layers

    if n_groups > 1:
        # Rebuild optimizer with one group per active layer to match the checkpoint.
        first_prefix = f"model.layers.{active[0]}.attention."
        first_params = [p for n, p in student.named_parameters()
                        if n.startswith(first_prefix) and p.requires_grad]
        new_opt = torch.optim.AdamW(first_params, lr=cfg.learning_rate)
        for li in active[1:n_groups]:
            prefix = f"model.layers.{li}.attention."
            params = [p for n, p in student.named_parameters()
                      if n.startswith(prefix) and p.requires_grad]
            if params:
                new_opt.add_param_group({"params": params, "lr": 5e-5})
        print(f"[RESUME] Rebuilt optimizer with {len(new_opt.param_groups)} groups.")
        new_opt.load_state_dict(torch.load(opt_path, map_location=dev1))
        return step, new_opt
    else:
        # Single group: create a fresh single-group optimizer and load state.
        new_opt = _build_optimizer(cfg, student)
        new_opt.load_state_dict(torch.load(opt_path, map_location=dev1))
        print("[RESUME] Optimizer loaded.")
        return step, new_opt


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _generate_eval(
    model: LLaDA2MoeModelLM,
    tokenizer,
    prompts: Sequence[str],
    device: torch.device,
    cfg: DistillConfig,
) -> List[str]:
    """Run block-parallel generation for qualitative evaluation."""
    model.eval()
    mask_id = tokenizer.mask_token_id
    eos_id  = tokenizer.eos_token_id
    outs = []
    for p in prompts:
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(device)
        input_ids = enc["input_ids"] if isinstance(enc, dict) else enc
        gen_ids = model.generate(
            inputs=input_ids,
            temperature=0.2,
            block_length=cfg.eval_block_length,
            steps=cfg.eval_steps,
            gen_length=cfg.eval_gen_len,
            eos_early_stop=True,
            threshold=cfg.eval_threshold,
            editing_threshold=cfg.eval_editing_threshold,
            max_post_steps=cfg.eval_max_post_steps,
            eos_id=eos_id,
            mask_id=mask_id,
            repetition_penalty=1.1,
        )
        outs.append(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
    model.train()
    return outs


# ── Main training loop ────────────────────────────────────────────────────────

def distill_step1(cfg: DistillConfig) -> None:
    """Stage-1 LoLCATs distillation."""
    dev0 = torch.device(cfg.device_teacher)
    dev1 = torch.device(cfg.device_student)

    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model_path, trust_remote_code=True)
    mask_id   = tokenizer.mask_token_id
    assert mask_id is not None, "tokenizer must expose mask_token_id"

    # ── Models ──────────────────────────────────────────────────────────────
    teacher          = _load_teacher(cfg, dev0)
    student, s_config = _build_student(cfg, dev1, teacher)

    n_layers  = len(teacher.model.layers)
    layer_ids = cfg.distill_layers or list(range(n_layers))
    prog_seq  = []   # CurriculumManager will generate middle-out default for n_layers

    # ── Initial active layers & optimizer ───────────────────────────────────
    initial_active = _initial_active_layers(cfg, n_layers, prog_seq)
    _apply_active_layers(student, initial_active, hybrid_mode=cfg.use_block_softmax_hybrid)
    optimizer = _build_optimizer(cfg, student)
    scheduler = _build_scheduler(cfg, optimizer)

    trainable_count = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[INIT] Trainable parameters: {trainable_count:,}  (initial active: {initial_active})")

    # ── Curriculum ──────────────────────────────────────────────────────────
    def _revert_layer(li: int) -> None:
        """Revert a layer to teacher weights and freeze its parameters."""
        sync_layer_weights(student.model.layers[li], teacher.model.layers[li],
                           li, s_config, teacher.config, skip_standard=False)
        student.model.layers[li].attention.is_linear_active = False
        for p in student.model.layers[li].attention.parameters():
            p.requires_grad = False

    curriculum = CurriculumManager(
        cfg=cfg, n_layers=n_layers, optimizer=optimizer,
        student=student, revert_layer_fn=_revert_layer, prog_seq=prog_seq,
    )

    # ── Resume ──────────────────────────────────────────────────────────────
    step, resumed_opt = _resume(cfg, student, curriculum, dev1)
    if resumed_opt is not None:
        optimizer = resumed_opt
        curriculum.optimizer = optimizer   # keep curriculum in sync

    # Re-apply flags after resume (active_layers may have changed).
    _apply_active_layers(student, curriculum.active_layers, hybrid_mode=cfg.use_block_softmax_hybrid)

    # ── Hooks ───────────────────────────────────────────────────────────────
    teacher_hooks = TeacherHooks(teacher, layer_ids)
    student_hooks = StudentHooks(student, layer_ids, curriculum.p_force_dict)

    # ── Masks, vocab & data ────────────────────────────────────────────────────
    block_size   = s_config.block_size
    mask_proto_t = build_block_causal_mask(cfg.seq_len, block_size, dev0, teacher.dtype)
    mask_proto_s = build_block_causal_mask(cfg.seq_len, block_size, dev1, student.dtype)
    loader       = StreamingTextLoader(cfg.dataset_name, cfg.dataset_subset, tokenizer, cfg.seq_len)

    # Valid vocab IDs for T2T random-token substitution (exclude special tokens).
    special_ids = set(tokenizer.all_special_ids)
    vocab_ids_cpu = torch.tensor(
        [i for i in range(tokenizer.vocab_size) if i not in special_ids], dtype=torch.long
    )
    vocab_ids_t = vocab_ids_cpu.to(dev0)   # for T2T corruption (on teacher device)

    # lm_head in fp32 for numerically stable logit computation.
    teacher.lm_head.float()
    student.lm_head.float()

    pbar = tqdm(total=cfg.num_steps, desc="Distillation", initial=step)
    skip_count = 0           # blocks skipped due to NaN/Inf
    loss_history: list = []  # (step, avg_loss, avg_loss_m2t, avg_loss_t2t)
    seq_len_history: list = []  # (step, avg_real_tokens)
    prev_n_active = len(curriculum.active_layers)  # track layer activations
    prev_layers   = set(curriculum.active_layers)  # for identifying newly added layer

    while step < cfg.num_steps:
        optimizer.zero_grad(set_to_none=True)
        window_loss_sum = 0.0
        window_m2t_sum  = 0.0
        window_t2t_sum  = 0.0
        window_blocks_done = 0
        microbatches_done = 0

        # Accumulate gradients sequentially to avoid OOM
        while microbatches_done < cfg.grad_accum_steps:
            # ── Load a microbatch of sequences ──────────────────────────────────
            # (User: if OOMing, set --batch_size 1 and increase --grad_accum_steps)
            input_ids, pad_mask = loader.next_batch(cfg.batch_size)
            B = input_ids.shape[0]

            # Use the maximum sequence length in the batch to determine num_blocks
            real_lens  = pad_mask.float().sum(dim=-1).cpu().long()
            max_real_L = int(real_lens.max().item())
            num_blocks = max(1, (max_real_L + block_size - 1) // block_size)
            
            attn4d_t   = mask_proto_t.expand(B, -1, -1, -1)
            attn4d_s   = mask_proto_s.expand(B, -1, -1, -1)

            # Track avg real (non-padding) token count across the batch
            avg_real_tok = float(real_lens.float().mean().item())
            seq_len_history.append((step, avg_real_tok))

            block_pbar = tqdm(range(num_blocks), desc=f"Seq {microbatches_done+1}/{cfg.grad_accum_steps}", leave=False)
            for block_idx in block_pbar:
                s_tok = block_idx * block_size
                e_tok = min((block_idx + 1) * block_size, cfg.seq_len)
                n_tokens = e_tok - s_tok
                if n_tokens <= 0:
                    continue

                t_noise = 0.05 + 0.90 * random.random()   # t ∈ [0.05, 0.95]

                # ── M2T: corrupt with [MASK] ──────────────────────────────────────
                noisy_m2t = corrupt_one_block(
                    input_ids.to(dev0), pad_mask.to(dev0), mask_id, block_size, t_noise, block_idx
                )

                # ── T2T: corrupt with random vocab tokens ─────────────────────────
                noisy_t2t, _ = corrupt_one_block_t2t(
                    input_ids.to(dev0), pad_mask.to(dev0), vocab_ids_t,
                    mask_id, block_size, t_noise, block_idx
                )

                # ── Teacher forward (M2T) ─────────────────────────────────────────
                teacher_hooks.clear()
                with torch.no_grad():
                    t_out_m2t = teacher.model(
                        noisy_m2t, attention_mask=attn4d_t,
                        key_padding_mask=pad_mask.to(dev0).bool(),
                    )
                    with torch.autocast("cuda", enabled=False):
                        t_logits_m2t = teacher.lm_head(
                            t_out_m2t.last_hidden_state[:, s_tok:e_tok].float()
                        ).to(dev1, non_blocking=True)

                t_targets_m2t = {
                    li: ten.to(dev1, dtype=student.dtype, non_blocking=True)
                    for li, ten in teacher_hooks.store.items()
                }

                # ── Teacher forward (T2T) ─────────────────────────────────────────
                teacher_hooks.clear()
                with torch.no_grad():
                    t_out_t2t = teacher.model(
                        noisy_t2t, attention_mask=attn4d_t,
                        key_padding_mask=pad_mask.to(dev0).bool(),
                    )
                    with torch.autocast("cuda", enabled=False):
                        t_logits_t2t = teacher.lm_head(
                            t_out_t2t.last_hidden_state[:, s_tok:e_tok].float()
                        ).to(dev1, non_blocking=True)

                t_targets_t2t = {
                    li: ten.to(dev1, dtype=student.dtype, non_blocking=True)
                    for li, ten in teacher_hooks.store.items()
                }

                # ── Student forward (M2T) ─────────────────────────────────────────
                active_set = set(curriculum.active_layers)

                student_hooks.teacher_targets.clear()
                student_hooks.teacher_targets.update(t_targets_m2t)
                student_hooks.clear()

                s_out_m2t = student.model(
                    noisy_m2t.to(dev1), attention_mask=attn4d_s,
                    key_padding_mask=pad_mask.to(dev1).bool(),
                )
                with torch.autocast("cuda", enabled=False):
                    s_logits_m2t = student.lm_head(
                        s_out_m2t.last_hidden_state[:, s_tok:e_tok].float()
                    )

                # NaN guard — if M2T student is unstable, skip this block entirely
                nan_layers_m2t = [
                    i for i in layer_ids
                    if not torch.isfinite(student_hooks.record.get(i, torch.ones(1, device=dev1))).all()
                ]
                if nan_layers_m2t:
                    print(
                        f"\n[NaN] step={step} block={block_idx}/{num_blocks} "
                        f"M2T: NaN/Inf in layers {nan_layers_m2t} — skipping block."
                    )
                    skip_count += 1
                    continue

                # ── Student metrics (M2T) ─────────────────────────────────────────
                block_mask = pad_mask[:, s_tok:e_tok].to(dev1, dtype=torch.float32)

                mse_m2t = torch.zeros((), device=dev1, dtype=torch.float32)
                for i in layer_ids:
                    if i in active_set and i in student_hooks.record and i in t_targets_m2t:
                        s_val = student_hooks.record[i][:, s_tok:e_tok]
                        t_val = t_targets_m2t[i][:, s_tok:e_tok]
                        # Average over D, then masked sum over (B, L_block), then divide by valid token count
                        err_sq = torch.mean((s_val.float() - t_val.float())**2, dim=-1)
                        mse_m2t = mse_m2t + (err_sq * block_mask).sum() / (block_mask.sum() + 1e-9)

                kl_m2t = torch.zeros((), device=dev1, dtype=torch.float32)
                if cfg.beta != 0.0:
                    kl_full = F.kl_div(
                        F.log_softmax(s_logits_m2t.float() / cfg.temperature, dim=-1),
                        F.softmax(t_logits_m2t.float() / cfg.temperature, dim=-1),
                        reduction="none",
                    ).sum(dim=-1) # (B, L_block)
                    kl_m2t = (kl_full * block_mask).sum() / (block_mask.sum() + 1e-9) * (cfg.temperature ** 2)

                loss_m2t = cfg.alpha * mse_m2t + cfg.beta * kl_m2t

                # ── Student forward (T2T) ─────────────────────────────────────────
                student_hooks.teacher_targets.clear()
                student_hooks.teacher_targets.update(t_targets_t2t)
                student_hooks.clear()

                s_out_t2t = student.model(
                    noisy_t2t.to(dev1), attention_mask=attn4d_s,
                    key_padding_mask=pad_mask.to(dev1).bool(),
                )
                with torch.autocast("cuda", enabled=False):
                    s_logits_t2t = student.lm_head(
                        s_out_t2t.last_hidden_state[:, s_tok:e_tok].float()
                    )

                mse_t2t = torch.zeros((), device=dev1, dtype=torch.float32)

                # NaN guard for T2T student forward
                nan_layers_t2t = [
                    i for i in layer_ids
                    if not torch.isfinite(student_hooks.record.get(i, torch.ones(1, device=dev1))).all()
                ]
                if nan_layers_t2t:
                    print(
                        f"\n[NaN] step={step} block={block_idx}/{num_blocks} "
                        f"T2T: NaN/Inf in layers {nan_layers_t2t} — falling back to M2T-only loss."
                    )
                    # T2T is unstable; fallback to scaled M2T loss only
                    loss_t2t = loss_m2t.detach() # dummy for logging
                    loss = (cfg.omega_mask * loss_m2t) / (cfg.grad_accum_steps * num_blocks)
                else:
                    mse_t2t = torch.zeros((), device=dev1, dtype=torch.float32)
                    for i in layer_ids:
                        if i in active_set and i in student_hooks.record and i in t_targets_t2t:
                            s_val = student_hooks.record[i][:, s_tok:e_tok]
                            t_val = t_targets_t2t[i][:, s_tok:e_tok]
                            err_sq = torch.mean((s_val.float() - t_val.float())**2, dim=-1)
                            mse_t2t = mse_t2t + (err_sq * block_mask).sum() / (block_mask.sum() + 1e-9)

                    kl_t2t = torch.zeros((), device=dev1, dtype=torch.float32)
                    if cfg.beta != 0.0:
                        kl_full = F.kl_div(
                            F.log_softmax(s_logits_t2t.float() / cfg.temperature, dim=-1),
                            F.softmax(t_logits_t2t.float() / cfg.temperature, dim=-1),
                            reduction="none",
                        ).sum(dim=-1)
                        kl_t2t = (kl_full * block_mask).sum() / (block_mask.sum() + 1e-9) * (cfg.temperature ** 2)

                    loss_t2t = cfg.alpha * mse_t2t + cfg.beta * kl_t2t
                    loss = (cfg.omega_mask * loss_m2t + cfg.omega_edit * loss_t2t) / (cfg.grad_accum_steps * num_blocks)

                # ── Combined loss & backprop ──────────────────────────────────────
                # This call happens INSIDE the block loop: it clears activations for THIS block immediately.
                if torch.isfinite(loss):
                    loss.backward()
                    # Correct sum for logging: Mean Sequence Loss
                    window_loss_sum += float(loss) * (cfg.grad_accum_steps * num_blocks)
                    window_m2t_sum  += float(loss_m2t)
                    window_t2t_sum  += float(loss_t2t)
                    window_blocks_done += 1
                    
                    block_pbar.set_description(f"Seq {microbatches_done+1}/{cfg.grad_accum_steps} | block {block_idx+1}/{num_blocks}")
                    block_pbar.set_postfix({"loss": f"{float(loss)*(cfg.grad_accum_steps*num_blocks):.3f}"})
                else:
                    skip_count += 1

            # ── Nan guard fallback ──────────────────
            # (In-loop logic for Nan guard was previously duplicated, cleaned up here)

        # ── End of Microbatch ───────────────────
        microbatches_done += 1
        step += B
        pbar.update(B)

        # ── End of Accumulation Window (Update Optimizer) ─────────────────────
        # We normalized loss by grad_accum_steps * num_blocks during backward.
        # Now we step and log.
        if window_blocks_done > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            # optimizer.zero_grad handled at start of window
            
            # window_loss_sum is the sum of (BatchMeanLoss per block)
            # Dividing by window_blocks_done gives the final mean over the tokens in the window.
            avg_loss     = window_loss_sum / max(1, window_blocks_done)
            avg_loss_m2t = window_m2t_sum  / max(1, window_blocks_done)
            avg_loss_t2t = window_t2t_sum  / max(1, window_blocks_done)

            # Interval-based events
            loss_history.append((step, avg_loss, avg_loss_m2t, avg_loss_t2t))
            scheduler.step(avg_loss)
            curriculum.step(step)

            pbar.set_description(
                f"loss={avg_loss:.4f}  "
                f"m2t(mse={avg_loss_m2t:.3f})  "
                f"t2t(mse={avg_loss_t2t:.3f})  "
                f"layers={len(curriculum.active_layers)}  "
                f"skips={skip_count}"
            )

            # Mid-training events
            # Robust triggers for non-unit batch sizes
            def _triggered(every, current_step, step_inc):
                if every <= 0: return False
                return (current_step // every) > ((current_step - step_inc) // every)

            inc = B * cfg.grad_accum_steps
            if _triggered(cfg.plot_attn_every, step, inc):
                from .attn_viz import plot_loss_curve, plot_teacher_all_layers, plot_student_all_layers
                _viz_out = os.path.join(cfg.output_dir, "attn_plots")
                plot_teacher_all_layers(
                    teacher, input_ids.to(dev0), attn4d_t, pad_mask.to(dev0),
                    n_layers=n_layers, step=step, out_dir=_viz_out,
                    max_len=cfg.plot_attn_max_len, device=dev0,
                )
                student.eval()
                plot_student_all_layers(
                    student, input_ids.to(dev1), attn4d_s, pad_mask.to(dev1),
                    n_layers=n_layers, step=step, out_dir=_viz_out,
                    max_len=cfg.plot_attn_max_len, device=dev1,
                )
                student.train()
                plot_loss_curve(loss_history, seq_len_history, _viz_out, step)

            if _triggered(cfg.eval_every, step, inc):
                print(f"\n--- Eval @ step {step} ---")
                student.eval()
                gens = _generate_eval(student, tokenizer, cfg.eval_prompts, dev1, cfg)
                student.train()
                gen_path = os.path.join(cfg.output_dir, "generations.txt")
                with open(gen_path, "a") as _gf:
                    _gf.write(f"\n=== Step {step} ===\n")
                    for prompt, gen in zip(cfg.eval_prompts, gens):
                        _gf.write(f"Prompt: {prompt}\nStudent: {gen}\n")
                print(f"  Sample: {gens[0]}")

            if _triggered(cfg.save_every, step, inc):
                _save_checkpoint(student, optimizer, curriculum, tokenizer, step, cfg)

    # ── Cleanup & final save ─────────────────────────────────────────────────
    pbar.close()
    teacher_hooks.remove()
    student_hooks.remove()
    _save_final(student, tokenizer, cfg)


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LLaDAFast Stage-1 Distillation")
    ap.add_argument("--teacher_model",            type=str,   default="inclusionAI/LLaDA2.1-mini")
    ap.add_argument("--steps",                    type=int,   default=15000)
    ap.add_argument("--lr",                       type=float, default=2e-5)
    ap.add_argument("--device_teacher",           type=str,   default="cuda:0")
    ap.add_argument("--device_student",           type=str,   default="cuda:1")
    ap.add_argument("--alpha",                    type=float, default=1.0,  help="MSE loss weight")
    ap.add_argument("--beta",                     type=float, default=1.0,  help="KL loss weight")
    ap.add_argument("--temp",                     type=float, default=1.0,  help="KL temperature")
    ap.add_argument("--omega_mask",               type=float, default=0.5,  help="M2T loss weight (mask-to-token)")
    ap.add_argument("--omega_edit",               type=float, default=0.5,  help="T2T loss weight (token editing)")
    ap.add_argument("--linear_layers",            type=str,   default="",
                    help="Comma-separated layer IDs to linearize (default: all)")
    ap.add_argument("--progressive_interval",     type=int,   default=0,
                    help="Steps between activating each additional layer")
    ap.add_argument("--force_decay_length",       type=int,   default=1000,
                    help="Steps for teacher-forcing probability to decay to 0")
    ap.add_argument("--use_llm_curriculum",       action="store_true",
                    help="Use LLM judge for quality-gated layer progression")
    ap.add_argument("--eval_every",               type=int,   default=100)
    ap.add_argument("--save_every",               type=int,   default=2000)
    ap.add_argument("--weight_decay",             type=float, default=0.0)
    ap.add_argument("--batch_size",               type=int,   default=1)
    ap.add_argument("--grad_accum_steps",         type=int,   default=1)
    ap.add_argument("--lr_patience",              type=int,   default=20)
    ap.add_argument("--lr_factor",               type=float, default=0.1)
    ap.add_argument("--min_lr",                   type=float, default=1e-5)
    ap.add_argument("--use_gated_deltanet",        action="store_true")
    ap.add_argument("--use_hybrid_local_softmax",  action="store_true")
    ap.add_argument("--use_block_softmax_hybrid",  action="store_true",
                    help="BlockSoftmaxLinearHybrid: learnable softmax+linear blend")
    ap.add_argument("--resume_from",              type=str,   default="")
    ap.add_argument("--joint",                    action="store_true",
                    help="Distill all layers jointly (Attention Surgery style)")
    ap.add_argument("--hybrid_ratio_start",       type=float, default=0.5)
    ap.add_argument("--hybrid_ratio_end",         type=float, default=0.0)
    ap.add_argument("--hybrid_anneal_steps",      type=int,   default=5000)
    ap.add_argument("--output_dir",               type=str,   default="./student_forcing_final")
    ap.add_argument("--plot_attn_every",           type=int,   default=0,
                    help="Save attention maps every N steps (0=off). Starts at step 0.")
    ap.add_argument("--plot_attn_max_layers",      type=int,   default=4)
    ap.add_argument("--plot_attn_max_len",         type=int,   default=128)
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = DistillConfig(
        teacher_model_path     = args.teacher_model,
        num_steps              = args.steps,
        learning_rate          = args.lr,
        device_teacher         = args.device_teacher,
        device_student         = args.device_student,
        alpha                  = args.alpha,
        beta                   = args.beta,
        temperature            = args.temp,
        omega_mask             = args.omega_mask,
        omega_edit             = args.omega_edit,
        linear_layers          = [int(x) for x in args.linear_layers.split(",")]
                                  if args.linear_layers else None,
        progressive_interval   = args.progressive_interval,
        force_decay_length     = args.force_decay_length,
        use_llm_curriculum_eval= args.use_llm_curriculum,
        eval_every             = args.eval_every,
        save_every             = args.save_every,
        weight_decay           = args.weight_decay,
        lr_patience            = args.lr_patience,
        batch_size             = args.batch_size,
        grad_accum_steps       = args.grad_accum_steps,
        lr_factor              = args.lr_factor,
        min_lr                 = args.min_lr,
        use_gated_deltanet     = args.use_gated_deltanet,
        use_hybrid_local_softmax = args.use_hybrid_local_softmax,
        use_block_softmax_hybrid = args.use_block_softmax_hybrid,
        resume_from            = args.resume_from,
        joint                  = args.joint,
        hybrid_ratio_start     = args.hybrid_ratio_start,
        hybrid_ratio_end       = args.hybrid_ratio_end,
        hybrid_anneal_steps    = args.hybrid_anneal_steps,
        output_dir             = args.output_dir,
        plot_attn_every        = args.plot_attn_every,
        plot_attn_max_layers   = args.plot_attn_max_layers,
        plot_attn_max_len      = args.plot_attn_max_len,
    )
    distill_step1(cfg)
