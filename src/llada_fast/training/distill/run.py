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
from .attn_viz import (
    _get_teacher_attn_weights,
    _compute_student_kernel_matrix,
    plot_and_save_attention,
)

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
    out_dir = os.path.join(cfg.output_dir, f"step_{step}")
    parent_dir = cfg.output_dir
    os.makedirs(parent_dir, exist_ok=True)
    for old in glob.glob(os.path.join(parent_dir, "step_*")):
        try:
            shutil.rmtree(old)
            print(f"\n[SAVE] Removed old checkpoint: {old}")
        except Exception as e:
            print(f"[SAVE] Could not remove {old}: {e}")

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
    prog_seq  = DEFAULT_PROG_SEQ[:n_layers] if n_layers <= len(DEFAULT_PROG_SEQ) else list(range(n_layers))

    # ── Initial active layers & optimizer ───────────────────────────────────
    initial_active = _initial_active_layers(cfg, n_layers, prog_seq)
    _apply_active_layers(student, initial_active, hybrid_mode=cfg.use_block_softmax_hybrid)
    optimizer = _build_optimizer(cfg, student)

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
    skip_count = 0  # counts optimizer steps skipped due to NaN/Inf

    while step < cfg.num_steps:

        # ── Attention visualization ──────────────────────────────────────────────────
        if cfg.plot_attn_every > 0 and step % cfg.plot_attn_every == 0:
            _viz_layers = sorted(curriculum.active_layers)[:cfg.plot_attn_max_layers]
            _viz_ids, _viz_pad = loader.next_batch()          # grab a sample sequence
            _viz_attn4d_t = mask_proto_t.expand(_viz_ids.shape[0], -1, -1, -1)
            _viz_attn4d_s = mask_proto_s.expand(_viz_ids.shape[0], -1, -1, -1)
            student.eval()
            teacher_attn = _get_teacher_attn_weights(
                teacher, _viz_ids, _viz_attn4d_t, _viz_pad, _viz_layers, dev0
            )
            student_attn = _compute_student_kernel_matrix(
                student, _viz_ids.to(dev1), _viz_attn4d_s, _viz_pad, _viz_layers, dev1
            )
            student.train()
            plot_and_save_attention(
                teacher_attn, student_attn, _viz_layers, step,
                out_dir=os.path.join(cfg.output_dir, "attn_plots"),
                max_len=cfg.plot_attn_max_len,
            )

        # ── Evaluation ──────────────────────────────────────────────────────
        if cfg.eval_every > 0 and step % cfg.eval_every == 0 and step > 0:
            print(f"\n--- Eval @ step {step} ---")
            gens = _generate_eval(student, tokenizer, cfg.eval_prompts, dev1, cfg)
            for prompt, gen in zip(cfg.eval_prompts, gens):
                print(f"  Prompt:  {prompt}\n  Student: {gen}\n")

            if cfg.use_llm_curriculum_eval and step > 0:
                current_layer = curriculum.active_layers[-1]
                curriculum.state.gen_history.append({"layer": current_layer, "text": gens[0]})
                if len(curriculum.state.gen_history) > 8:
                    curriculum.state.gen_history.pop(0)
                verdict = evaluate_with_llm(
                    curriculum.state.gen_history,
                    cfg.eval_prompts[0],
                    step,
                    curriculum.state.consecutive_failures,
                    current_layer,
                )
                print(f"  LLM Judge: {verdict}")
                curriculum.record_verdict(verdict)

        # ── Checkpointing ───────────────────────────────────────────────────
        if cfg.save_every > 0 and step > 0 and step % cfg.save_every == 0:
            _save_checkpoint(student, optimizer, curriculum, tokenizer, step, cfg)

        # ── Data: load one sequence, iterate over all blocks ─────────────────
        input_ids, pad_mask = loader.next_batch()   # (1, L)
        B = input_ids.shape[0]

        real_len   = int(pad_mask[0].sum().item())
        num_blocks = max(1, (real_len + block_size - 1) // block_size)
        attn4d_t   = mask_proto_t.expand(B, -1, -1, -1)
        attn4d_s   = mask_proto_s.expand(B, -1, -1, -1)

        for block_idx in range(num_blocks):
            s_tok = block_idx * block_size
            e_tok = min((block_idx + 1) * block_size, real_len)
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

            # M2T loss
            mse_m2t = torch.zeros((), device=dev1, dtype=torch.float32)
            for i in layer_ids:
                if i in active_set and i in student_hooks.record and i in t_targets_m2t:
                    s_val = student_hooks.record[i][:, s_tok:e_tok]
                    t_val = t_targets_m2t[i][:, s_tok:e_tok]
                    mse_m2t = mse_m2t + F.mse_loss(
                        s_val.float(), t_val.float(), reduction="sum"
                    ) / (n_tokens * s_val.shape[-1])

            kl_m2t = torch.zeros((), device=dev1, dtype=torch.float32)
            if cfg.beta != 0.0:
                kl_m2t = F.kl_div(
                    F.log_softmax(s_logits_m2t.float() / cfg.temperature, dim=-1),
                    F.softmax(t_logits_m2t.float() / cfg.temperature, dim=-1),
                    reduction="batchmean",
                ) * (cfg.temperature ** 2)

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
                # T2T is unstable; still use M2T-only loss for this block
                loss = cfg.omega_mask * loss_m2t
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in student.parameters() if p.requires_grad], 1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    curriculum.step(step)
                    step += 1
                    pbar.update(1)
                else:
                    skip_count += 1
                    print(
                        f"\n[SKIP] step={step} block={block_idx}/{num_blocks} "
                        f"T2T fallback M2T loss also non-finite — skipping optimizer step. "
                        f"(total skips: {skip_count})"
                    )
                if step >= cfg.num_steps:
                    break
                continue

            for i in layer_ids:
                if i in active_set and i in student_hooks.record and i in t_targets_t2t:
                    s_val = student_hooks.record[i][:, s_tok:e_tok]
                    t_val = t_targets_t2t[i][:, s_tok:e_tok]
                    mse_t2t = mse_t2t + F.mse_loss(
                        s_val.float(), t_val.float(), reduction="sum"
                    ) / (n_tokens * s_val.shape[-1])

            kl_t2t = torch.zeros((), device=dev1, dtype=torch.float32)
            if cfg.beta != 0.0:
                kl_t2t = F.kl_div(
                    F.log_softmax(s_logits_t2t.float() / cfg.temperature, dim=-1),
                    F.softmax(t_logits_t2t.float() / cfg.temperature, dim=-1),
                    reduction="batchmean",
                ) * (cfg.temperature ** 2)

            loss_t2t = cfg.alpha * mse_t2t + cfg.beta * kl_t2t

            # ── Combined loss & backprop ──────────────────────────────────────
            loss = cfg.omega_mask * loss_m2t + cfg.omega_edit * loss_t2t

            if not torch.isfinite(loss):
                skip_count += 1
                print(
                    f"\n[SKIP] step={step} block={block_idx}/{num_blocks} "
                    f"Combined loss non-finite (m2t={loss_m2t.item():.4f}, t2t={loss_t2t.item():.4f}) "
                    f"— skipping optimizer step. (total skips: {skip_count})"
                )
                optimizer.zero_grad(set_to_none=True)  # clear any partial grads
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # ── Curriculum step ───────────────────────────────────────────────
            curriculum.step(step)

            pbar.set_description(
                f"loss={loss.item():.4f}  "
                f"m2t(mse={float(mse_m2t):.3f},kl={float(kl_m2t):.3f})  "
                f"t2t(mse={float(mse_t2t):.3f},kl={float(kl_t2t):.3f})  "
                f"layers={len(curriculum.active_layers)}  "
                f"skips={skip_count}"
            )
            step += 1
            pbar.update(1)

            if step >= cfg.num_steps:
                break

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
