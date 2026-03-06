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
import json
import math
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
from .data import (
    build_block_causal_mask,
    build_bd3lm_mask,
    corrupt_one_block, corrupt_one_block_t2t,       # kept for reference / lora train
    corrupt_all_blocks, corrupt_all_blocks_t2t,     # BD3LM single-pass corruption
    TestSetBuffer,
    StreamingTextLoader,
)
from .hooks import TeacherHooks, StudentHooks

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
    if cfg.progressive_interval > 0:
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
    if cfg.joint or not cfg.progressive_interval:
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
    cfg: DistillConfig, optimizer: torch.optim.Optimizer, last_epoch: int = 0
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear warmup then cosine decay from learning_rate → min_lr.

    All units are GRADIENT UPDATES (one per optimizer.step() call).
    num_steps and warmup_steps from cfg are in sequences, so we divide by
    sequences_per_update = batch_size * grad_accum_steps to get gradient-update
    counts.  scheduler.step() is called once per optimizer.step(), so
    current_step in the lambda correctly tracks gradient updates.

    Pass last_epoch=step (sequences) when resuming; the conversion is applied here.
    """
    seq_per_update  = max(1, cfg.batch_size * cfg.grad_accum_steps)
    total_updates   = max(1, cfg.num_steps    // seq_per_update)
    warmup_updates  = max(1, cfg.warmup_steps // seq_per_update)
    done_updates    = last_epoch // seq_per_update   # gradient updates already done

    eta_min_ratio = cfg.min_lr / cfg.learning_rate

    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_updates:
            return float(current_step) / warmup_updates
        progress = float(current_step - warmup_updates) / max(1, total_updates - warmup_updates)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    # last_epoch - 1: LambdaLR.__init__ calls step() once, advancing to done_updates.
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch=done_updates - 1)


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

    if cfg.gradient_checkpointing:
        # GC is incompatible with our forward-hook-based MSE loss (GC runs the
        # forward under no_grad → hook-captured tensors have no grad_fn).
        # Memory is instead saved by backpropping M2T and T2T separately so only
        # one forward graph lives in memory at a time.
        print("[INIT] --gradient_checkpointing ignored: split-backward handles memory.")

    n_layers  = len(teacher.model.layers)
    layer_ids = cfg.distill_layers or list(range(n_layers))
    prog_seq  = []   # CurriculumManager will generate middle-out default for n_layers

    # ── Initial active layers & optimizer ───────────────────────────────────
    initial_active = _initial_active_layers(cfg, n_layers, prog_seq)
    _apply_active_layers(student, initial_active, hybrid_mode=cfg.use_block_softmax_hybrid)
    optimizer = _build_optimizer(cfg, student)
    # Scheduler is built AFTER resume so it references the final optimizer
    # and can be fast-forwarded to the correct cosine position.

    trainable_count = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[INIT] Trainable parameters: {trainable_count:,}  (initial active: {initial_active})")

    # ── Curriculum ──────────────────────────────────────────────────────────
    curriculum = CurriculumManager(
        cfg=cfg, n_layers=n_layers, optimizer=optimizer,
        student=student, prog_seq=prog_seq,
    )

    # ── Resume ──────────────────────────────────────────────────────────────
    step, resumed_opt = _resume(cfg, student, curriculum, dev1)
    if resumed_opt is not None:
        optimizer = resumed_opt
        curriculum.optimizer = optimizer   # keep curriculum in sync

    # Build scheduler now, bound to the final optimizer.
    # Pass step so LambdaLR fast-forwards to the correct LR position on resume.
    scheduler = _build_scheduler(cfg, optimizer, last_epoch=step)

    # Re-apply flags after resume (active_layers may have changed).
    _apply_active_layers(student, curriculum.active_layers, hybrid_mode=cfg.use_block_softmax_hybrid)

    # ── Hooks ───────────────────────────────────────────────────────────────
    teacher_hooks = TeacherHooks(teacher, layer_ids)
    student_hooks = StudentHooks(student, layer_ids)

    # ── Masks, vocab & data ────────────────────────────────────────────────────
    block_size = s_config.block_size

    # BD3LM 2L staircase mask: [x_t noisy | x_0 clean], shape (1,1,2L,2L).
    # Both halves share position IDs 0..L-1 so RoPE sees the same positions.
    mask_proto_t = build_bd3lm_mask(cfg.seq_len, block_size, dev0, teacher.dtype)
    mask_proto_s = build_bd3lm_mask(cfg.seq_len, block_size, dev1, student.dtype)

    # Shared position IDs for the 2L sequence: [0..L-1, 0..L-1]
    _pos = torch.arange(cfg.seq_len, dtype=torch.long)
    pos_ids_2L = torch.cat([_pos, _pos], dim=0).unsqueeze(0)  # (1, 2L)

    # Reserve test set FIRST so training never sees those records.
    test_buffer = TestSetBuffer(
        cfg.dataset_name, cfg.dataset_subset, tokenizer, cfg.seq_len,
        test_size=cfg.test_size,
    )
    loader = StreamingTextLoader(
        cfg.dataset_name, cfg.dataset_subset, tokenizer, cfg.seq_len,
        skip_first=test_buffer.raw_consumed,
    )

    # Valid vocab IDs for T2T random-token substitution (exclude special tokens).
    special_ids = set(tokenizer.all_special_ids)
    vocab_ids_cpu = torch.tensor(
        [i for i in range(tokenizer.vocab_size) if i not in special_ids], dtype=torch.long
    )
    vocab_ids_t = vocab_ids_cpu.to(dev0)   # for T2T corruption (on teacher device)

    # lm_head in fp32 for numerically stable logit computation.
    teacher.lm_head.float()
    student.lm_head.float()

    # ── Test-loss evaluation helper ───────────────────────────────────────────
    @torch.no_grad()
    def _eval_test_loss() -> tuple:
        """
        Compute CARD distillation loss on the reserved test set.
        Returns (avg_loss, avg_m2t, avg_t2t).
        """
        student.eval()
        tot_loss = tot_m2t = tot_t2t = 0.0
        n_ok = 0
        n_batches = (test_buffer._n + cfg.batch_size - 1) // cfg.batch_size
        for b in range(n_batches):
            s_idx = b * cfg.batch_size
            e_idx = min(s_idx + cfg.batch_size, test_buffer._n)
            ids = torch.cat(test_buffer._ids[s_idx:e_idx], dim=0).to(dev0)
            pmask = torch.cat(test_buffer._masks[s_idx:e_idx], dim=0).to(dev0)
            B_t   = ids.shape[0]

            full_nb   = (cfg.seq_len + block_size - 1) // block_size
            t_blk     = 0.05 + 0.90 * torch.rand(full_nb, device=dev0)

            noisy_m2t_e, _ = corrupt_all_blocks(ids, pmask, mask_id, block_size, t_blk)
            noisy_t2t_e, _ = corrupt_all_blocks_t2t(
                ids, pmask, vocab_ids_t, mask_id, block_size, t_blk
            )

            in_m2t_e = torch.cat([noisy_m2t_e, ids],    dim=1)
            in_t2t_e = torch.cat([noisy_t2t_e, ids],    dim=1)
            pad_2L_e = torch.cat([pmask, pmask],         dim=1).bool()
            pos_e_t  = pos_ids_2L.expand(B_t, -1).to(dev0)
            pos_e_s  = pos_ids_2L.expand(B_t, -1).to(dev1)
            attn_e_t = mask_proto_t.expand(B_t, -1, -1, -1)
            attn_e_s = mask_proto_s.expand(B_t, -1, -1, -1)

            # Teacher — M2T
            teacher_hooks.clear()
            t_out_m2t_e = teacher.model(
                in_m2t_e, attention_mask=attn_e_t,
                position_ids=pos_e_t, key_padding_mask=pad_2L_e,
            )
            with torch.autocast("cuda", enabled=False):
                t_logits_m2t_e = teacher.lm_head(
                    t_out_m2t_e.last_hidden_state[:, :cfg.seq_len].float()
                ).to(dev1)
            t_tgt_m2t_e = {
                li: v.to(dev1, dtype=student.dtype)
                for li, v in teacher_hooks.store.items()
            }

            # Teacher — T2T
            teacher_hooks.clear()
            t_out_t2t_e = teacher.model(
                in_t2t_e, attention_mask=attn_e_t,
                position_ids=pos_e_t, key_padding_mask=pad_2L_e,
            )
            with torch.autocast("cuda", enabled=False):
                t_logits_t2t_e = teacher.lm_head(
                    t_out_t2t_e.last_hidden_state[:, :cfg.seq_len].float()
                ).to(dev1)
            t_tgt_t2t_e = {
                li: v.to(dev1, dtype=student.dtype)
                for li, v in teacher_hooks.store.items()
            }

            # Student — M2T
            # StudentHooks short-circuits when module.training=False, so switch to
            # train() to allow hooks to record activations.  teacher_targets is left
            # empty so no teacher forcing actually occurs (pure student output).
            student_hooks.teacher_targets.clear()
            student_hooks.clear()
            student.train()
            s_out_m2t_e = student.model(
                in_m2t_e.to(dev1), attention_mask=attn_e_s,
                position_ids=pos_e_s, key_padding_mask=pad_2L_e.to(dev1),
                half_len=cfg.seq_len,
            )
            student.eval()
            with torch.autocast("cuda", enabled=False):
                s_logits_m2t_e = student.lm_head(
                    s_out_m2t_e.last_hidden_state[:, :cfg.seq_len].float()
                )

            rtm = pmask.to(dev1, dtype=torch.float32)   # (B, L) real-token mask
            nr  = rtm.sum() + 1e-9
            active_set_e = set(curriculum.active_layers)

            L_e = cfg.seq_len
            mse_m = torch.zeros((), device=dev1, dtype=torch.float32)
            n_contrib_m = 0
            for i in layer_ids:
                if i in active_set_e and i in student_hooks.record and i in t_tgt_m2t_e:
                    err = (student_hooks.record[i][:, :L_e].float()
                           - t_tgt_m2t_e[i][:, :L_e].float()).pow(2).mean(-1)
                    mse_m = mse_m + (err * rtm).sum() / nr
                    n_contrib_m += 1
            mse_m = mse_m / max(1, n_contrib_m)
            kl_m = torch.zeros((), device=dev1, dtype=torch.float32)
            if cfg.beta != 0.0:
                kf = F.kl_div(
                    F.log_softmax(s_logits_m2t_e.float() / cfg.temperature, dim=-1),
                    F.softmax(t_logits_m2t_e.float() / cfg.temperature, dim=-1),
                    reduction="none",
                ).sum(-1)
                kl_m = (kf * rtm).sum() / nr * cfg.temperature ** 2
            l_m2t = float(cfg.alpha * mse_m + cfg.beta * kl_m)

            # Student — T2T
            student_hooks.teacher_targets.clear()
            student_hooks.clear()
            student.train()
            s_out_t2t_e = student.model(
                in_t2t_e.to(dev1), attention_mask=attn_e_s,
                position_ids=pos_e_s, key_padding_mask=pad_2L_e.to(dev1),
                half_len=cfg.seq_len,
            )
            student.eval()
            with torch.autocast("cuda", enabled=False):
                s_logits_t2t_e = student.lm_head(
                    s_out_t2t_e.last_hidden_state[:, :cfg.seq_len].float()
                )

            mse_t = torch.zeros((), device=dev1, dtype=torch.float32)
            n_contrib_t = 0
            for i in layer_ids:
                if i in active_set_e and i in student_hooks.record and i in t_tgt_t2t_e:
                    err = (student_hooks.record[i][:, :L_e].float()
                           - t_tgt_t2t_e[i][:, :L_e].float()).pow(2).mean(-1)
                    mse_t = mse_t + (err * rtm).sum() / nr
                    n_contrib_t += 1
            mse_t = mse_t / max(1, n_contrib_t)
            kl_t = torch.zeros((), device=dev1, dtype=torch.float32)
            if cfg.beta != 0.0:
                kf = F.kl_div(
                    F.log_softmax(s_logits_t2t_e.float() / cfg.temperature, dim=-1),
                    F.softmax(t_logits_t2t_e.float() / cfg.temperature, dim=-1),
                    reduction="none",
                ).sum(-1)
                kl_t = (kf * rtm).sum() / nr * cfg.temperature ** 2
            l_t2t = float(cfg.alpha * mse_t + cfg.beta * kl_t)

            l_tot = cfg.omega_mask * l_m2t + cfg.omega_edit * l_t2t
            if torch.isfinite(torch.tensor(l_tot)):
                tot_loss += l_tot
                tot_m2t  += l_m2t
                tot_t2t  += l_t2t
                n_ok += 1

        student.train()
        if n_ok == 0:
            return float("nan"), float("nan"), float("nan")
        return tot_loss / n_ok, tot_m2t / n_ok, tot_t2t / n_ok

    @torch.no_grad()
    def _eval_perplexity() -> float:
        """
        Rigorous Blockwise Pseudo-Perplexity.

        For each sequence:
        1. Parse sequence into blocks (block_size).
        2. Randomly select one block `i` to evaluate.
        3. Past blocks < i: kept clean (ground truth).
        4. Current block i: ~50% of tokens masked.
        5. Future blocks > i: completely masked out.
        6. Compute cross-entropy purely on the masked tokens of block `i`.
        """
        student.eval()
        attn_1L = build_block_causal_mask(cfg.seq_len, block_size, dev1, student.dtype)  # (1,1,L,L)
        ces = []
        for i in range(test_buffer._n):
            ids_t  = test_buffer._ids[i].to(dev1)   # (1, L)
            mask_t = test_buffer._masks[i].to(dev1)  # (1, L)
            real   = mask_t.bool()

            valid_len = real[0].sum().item()
            if valid_len == 0:
                continue
                
            num_blocks = (valid_len + block_size - 1) // block_size
            if num_blocks == 0:
                continue
                
            # Randomly select one block to evaluate
            target_block = int(torch.randint(0, num_blocks, (1,)).item())
            start_idx = target_block * block_size
            end_idx = min(start_idx + block_size, valid_len)
            
            noisy = ids_t.clone()
            
            # Future blocks > i: completely masked
            if end_idx < cfg.seq_len:
                noisy[0, end_idx:] = mask_id

            # Current block i: ~50% random masking
            block_len = end_idx - start_idx
            noise_mask = torch.zeros_like(real)
            block_noise = torch.rand((1, block_len), device=dev1) < 0.5
            if not block_noise.any():
                block_noise[0, 0] = True  # guarantee at least 1 mask
                
            noise_mask[0, start_idx:end_idx] = block_noise
            noisy[noise_mask] = mask_id

            out = student.model(
                noisy,
                attention_mask=attn_1L.expand(1, -1, -1, -1),
                key_padding_mask=real,
            )
            with torch.autocast("cuda", enabled=False):
                logits = student.lm_head(out.last_hidden_state[0].float())  # (L, V)

            log_p = F.log_softmax(logits, dim=-1)        # (L, V)
            masked_pos  = noise_mask[0]                  # (L,) bool
            target_ids  = ids_t[0]                       # (L,)
            ce = -log_p[masked_pos, target_ids[masked_pos]].mean()
            
            if torch.isfinite(ce):
                ces.append(ce.item())

        student.train()
        if not ces:
            return float("nan")
        return math.exp(sum(ces) / len(ces))

    pbar = tqdm(total=cfg.num_steps, desc="Distillation", initial=step)
    skip_count = 0           # microbatches skipped due to NaN/Inf
    loss_history: list = []  # (step, train_loss, train_m2t, train_t2t, test_loss, test_m2t, test_t2t)
    ppl_history:  list = []  # (step, perplexity)
    seq_len_history: list = []  # (step, avg_real_tokens)
    prev_n_active = len(curriculum.active_layers)  # track layer activations
    prev_layers   = set(curriculum.active_layers)  # for identifying newly added layer

    while step < cfg.num_steps:
        optimizer.zero_grad(set_to_none=True)
        window_loss_sum   = 0.0
        window_m2t_sum    = 0.0
        window_t2t_sum    = 0.0
        window_steps_done = 0      # microbatches that produced a finite loss
        microbatches_done = 0

        # ── Accumulate gradients over cfg.grad_accum_steps microbatches ───────
        while microbatches_done < cfg.grad_accum_steps:
            # ── Load a microbatch ───────────────────────────────────────────────
            input_ids, pad_mask = loader.next_batch(cfg.batch_size)
            B = input_ids.shape[0]

            real_lens  = pad_mask.float().sum(dim=-1).cpu().long()
            max_real_L = int(real_lens.max().item())
            num_blocks = max(1, (max_real_L + block_size - 1) // block_size)

            avg_real_tok = float(real_lens.float().mean().item())
            seq_len_history.append((step + microbatches_done * B, avg_real_tok))

            # ── BD3LM single-pass: build 2L inputs ─────────────────────────────
            # Corrupt ALL blocks simultaneously (independent noise per block).
            # Must use the FULL seq_len block count — corrupt_all_blocks internally
            # computes num_blocks = ceil(seq_len / block_size) and needs t_per_block
            # to have at least that many entries to avoid a broadcast shape error.
            full_num_blocks = (cfg.seq_len + block_size - 1) // block_size
            t_per_block = 0.05 + 0.90 * torch.rand(full_num_blocks, device=dev0)

            ids_dev0 = input_ids.to(dev0)
            pm_dev0  = pad_mask.to(dev0)

            noisy_m2t, _ = corrupt_all_blocks(
                ids_dev0, pm_dev0, mask_id, block_size, t_per_block
            )
            noisy_t2t, _ = corrupt_all_blocks_t2t(
                ids_dev0, pm_dev0, vocab_ids_t, mask_id, block_size, t_per_block
            )

            # 2L inputs: [x_t (noisy) | x_0 (clean)] — staircase mask ensures
            # noisy block i attends only to CLEAN blocks < i (truly clean past).
            in_m2t_2L = torch.cat([noisy_m2t, ids_dev0], dim=1)   # (B, 2L) on dev0
            in_t2t_2L = torch.cat([noisy_t2t, ids_dev0], dim=1)   # (B, 2L) on dev0
            pad_2L_0  = torch.cat([pm_dev0,   pm_dev0],  dim=1).bool()  # (B, 2L)

            # Position IDs: both halves share 0..L-1 so RoPE/embeddings align.
            pos_2L_t  = pos_ids_2L.expand(B, -1).to(dev0)
            pos_2L_s  = pos_ids_2L.expand(B, -1).to(dev1)

            # Expand 2L mask to batch
            attn4d_t = mask_proto_t.expand(B, -1, -1, -1)
            attn4d_s = mask_proto_s.expand(B, -1, -1, -1)

            # ── Teacher forward — M2T ───────────────────────────────────────────
            teacher_hooks.clear()
            with torch.no_grad():
                t_out_m2t = teacher.model(
                    in_m2t_2L, attention_mask=attn4d_t,
                    position_ids=pos_2L_t,
                    key_padding_mask=pad_2L_0,
                )
                with torch.autocast("cuda", enabled=False):
                    # Only the noisy half (first L) produces the denoising logits.
                    t_logits_m2t = teacher.lm_head(
                        t_out_m2t.last_hidden_state[:, :cfg.seq_len].float()
                    ).to(dev1, non_blocking=True)

            # Store FULL (B, 2L, D) for teacher forcing; slice at loss time.
            t_targets_m2t = {
                li: ten.to(dev1, dtype=student.dtype, non_blocking=True)
                for li, ten in teacher_hooks.store.items()
            }

            # ── Teacher forward — T2T ───────────────────────────────────────────
            teacher_hooks.clear()
            with torch.no_grad():
                t_out_t2t = teacher.model(
                    in_t2t_2L, attention_mask=attn4d_t,
                    position_ids=pos_2L_t,
                    key_padding_mask=pad_2L_0,
                )
                with torch.autocast("cuda", enabled=False):
                    t_logits_t2t = teacher.lm_head(
                        t_out_t2t.last_hidden_state[:, :cfg.seq_len].float()
                    ).to(dev1, non_blocking=True)

            t_targets_t2t = {
                li: ten.to(dev1, dtype=student.dtype, non_blocking=True)
                for li, ten in teacher_hooks.store.items()
            }

            active_set = set(curriculum.active_layers)

            # ── Student forward — M2T ───────────────────────────────────────────
            student_hooks.teacher_targets.clear()
            student_hooks.teacher_targets.update(t_targets_m2t)
            student_hooks.clear()

            s_out_m2t = student.model(
                in_m2t_2L.to(dev1), attention_mask=attn4d_s,
                position_ids=pos_2L_s,
                key_padding_mask=pad_2L_0.to(dev1),
                half_len=cfg.seq_len,
            )
            with torch.autocast("cuda", enabled=False):
                s_logits_m2t = student.lm_head(
                    s_out_m2t.last_hidden_state[:, :cfg.seq_len].float()
                )

            # NaN guard on noisy-half hidden states (what the student needs to learn)
            nan_layers_m2t = [
                i for i in layer_ids
                if not torch.isfinite(
                    student_hooks.record.get(i, torch.ones(1, device=dev1))[:, :cfg.seq_len]
                ).all()
            ]
            if nan_layers_m2t:
                print(
                    f"\n[NaN] step={step} mb={microbatches_done} "
                    f"M2T: NaN/Inf in layers {nan_layers_m2t} — skipping microbatch."
                )
                skip_count += 1
                microbatches_done += 1
                continue

            # ── M2T loss — noisy half only, masked by real tokens ───────────────
            # real_tok_mask covers the first L positions (x_t = what we're training).
            real_tok_mask = pad_mask.to(dev1, dtype=torch.float32)  # (B, L)
            n_real = real_tok_mask.sum() + 1e-9

            L = cfg.seq_len
            mse_m2t = torch.zeros((), device=dev1, dtype=torch.float32)
            n_contrib_m2t = 0
            for i in layer_ids:
                if i in active_set and i in student_hooks.record and i in t_targets_m2t:
                    s_val = student_hooks.record[i][:, :L].float()   # (B, L, D) noisy half
                    t_val = t_targets_m2t[i][:, :L].float()          # (B, L, D) noisy half
                    err_sq = (s_val - t_val).pow(2).mean(dim=-1)     # (B, L)
                    mse_m2t = mse_m2t + (err_sq * real_tok_mask).sum() / n_real
                    n_contrib_m2t += 1
            mse_m2t = mse_m2t / max(1, n_contrib_m2t)

            kl_m2t = torch.zeros((), device=dev1, dtype=torch.float32)
            if cfg.beta != 0.0:
                kl_full = F.kl_div(
                    F.log_softmax(s_logits_m2t.float() / cfg.temperature, dim=-1),
                    F.softmax(t_logits_m2t.float() / cfg.temperature, dim=-1),
                    reduction="none",
                ).sum(dim=-1)  # (B, L)
                kl_m2t = (kl_full * real_tok_mask).sum() / n_real * (cfg.temperature ** 2)

            loss_m2t = cfg.alpha * mse_m2t + cfg.beta * kl_m2t

            # ── Student forward — T2T ───────────────────────────────────────────
            student_hooks.teacher_targets.clear()
            student_hooks.teacher_targets.update(t_targets_t2t)
            student_hooks.clear()

            s_out_t2t = student.model(
                in_t2t_2L.to(dev1), attention_mask=attn4d_s,
                position_ids=pos_2L_s,
                key_padding_mask=pad_2L_0.to(dev1),
                half_len=cfg.seq_len,
            )
            with torch.autocast("cuda", enabled=False):
                s_logits_t2t = student.lm_head(
                    s_out_t2t.last_hidden_state[:, :cfg.seq_len].float()
                )

            # NaN guard for T2T — fall back to M2T-only loss
            nan_layers_t2t = [
                i for i in layer_ids
                if not torch.isfinite(
                    student_hooks.record.get(i, torch.ones(1, device=dev1))[:, :cfg.seq_len]
                ).all()
            ]
            if nan_layers_t2t:
                print(
                    f"\n[NaN] step={step} mb={microbatches_done} "
                    f"T2T: NaN/Inf in layers {nan_layers_t2t} — falling back to M2T-only loss."
                )
                loss_t2t = loss_m2t.detach()
                loss = (cfg.omega_mask * loss_m2t) / cfg.grad_accum_steps
            else:
                # ── T2T loss — noisy half only ──────────────────────────────────
                mse_t2t = torch.zeros((), device=dev1, dtype=torch.float32)
                n_contrib_t2t = 0
                for i in layer_ids:
                    if i in active_set and i in student_hooks.record and i in t_targets_t2t:
                        s_val = student_hooks.record[i][:, :L].float()
                        t_val = t_targets_t2t[i][:, :L].float()
                        err_sq = (s_val - t_val).pow(2).mean(dim=-1)
                        mse_t2t = mse_t2t + (err_sq * real_tok_mask).sum() / n_real
                        n_contrib_t2t += 1
                mse_t2t = mse_t2t / max(1, n_contrib_t2t)

                kl_t2t = torch.zeros((), device=dev1, dtype=torch.float32)
                if cfg.beta != 0.0:
                    kl_full = F.kl_div(
                        F.log_softmax(s_logits_t2t.float() / cfg.temperature, dim=-1),
                        F.softmax(t_logits_t2t.float() / cfg.temperature, dim=-1),
                        reduction="none",
                    ).sum(dim=-1)
                    kl_t2t = (kl_full * real_tok_mask).sum() / n_real * (cfg.temperature ** 2)

                loss_t2t = cfg.alpha * mse_t2t + cfg.beta * kl_t2t
                loss = (cfg.omega_mask * loss_m2t + cfg.omega_edit * loss_t2t) / cfg.grad_accum_steps

            # ── Backprop ────────────────────────────────────────────────────────
            if torch.isfinite(loss):
                loss.backward()
                window_loss_sum   += float(loss) * cfg.grad_accum_steps
                window_m2t_sum    += float(loss_m2t)
                window_t2t_sum    += float(loss_t2t)
                window_steps_done += 1
            else:
                skip_count += 1

            # ── End of microbatch ───────────────────────────────────────────────
            microbatches_done += 1

        # ── End of accumulation window: optimizer step + logging ───────────────
        step += B * cfg.grad_accum_steps
        pbar.update(B * cfg.grad_accum_steps)

        if window_steps_done > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            # optimizer.zero_grad handled at start of window
            
            avg_loss     = window_loss_sum / max(1, window_steps_done)
            avg_loss_m2t = window_m2t_sum  / max(1, window_steps_done)
            avg_loss_t2t = window_t2t_sum  / max(1, window_steps_done)

            # ── Interval-based events ───────────────────────────────────────────
            def _triggered(every, current_step, step_inc):
                if every <= 0: return False
                return (current_step // every) > ((current_step - step_inc) // every)

            inc = B * cfg.grad_accum_steps

            # Test-set evaluation + perplexity (same trigger)
            test_loss = test_m2t = test_t2t = float("nan")
            student_ppl = float("nan")
            if _triggered(cfg.eval_every, step, inc):
                test_loss, test_m2t, test_t2t = _eval_test_loss()
                student_ppl = _eval_perplexity()
                ppl_history.append((step, student_ppl))

            loss_history.append((step, avg_loss, avg_loss_m2t, avg_loss_t2t,
                                  test_loss, test_m2t, test_t2t))
            scheduler.step()
            curriculum.step(step)

            test_str = (
                f"  test={test_loss:.4f}" if not (test_loss != test_loss) else ""
            )
            ppl_str = (
                f"  ppl={student_ppl:.1f}" if math.isfinite(student_ppl) else ""
            )
            pbar.set_description(
                f"train={avg_loss:.4f}  "
                f"m2t={avg_loss_m2t:.3f}  "
                f"t2t={avg_loss_t2t:.3f}"
                f"{test_str}{ppl_str}  "
                f"layers={len(curriculum.active_layers)}  "
                f"skips={skip_count}"
            )


            if _triggered(cfg.eval_every, step, inc):
                ppl_display = f"{student_ppl:.1f}" if math.isfinite(student_ppl) else "nan"
                print(
                    f"\n--- Eval @ step {step} | "
                    f"train={avg_loss:.4f}  test={test_loss:.4f} "
                    f"(m2t={test_m2t:.3f} t2t={test_t2t:.3f})  "
                    f"ppl={ppl_display} ---"
                )
                gens = _generate_eval(student, tokenizer, cfg.eval_prompts, dev1, cfg)
                os.makedirs(cfg.output_dir, exist_ok=True)
                gen_path = os.path.join(cfg.output_dir, "generations.txt")
                with open(gen_path, "a") as _gf:
                    _gf.write(f"\n=== Step {step} ===\n")
                    for prompt, gen in zip(cfg.eval_prompts, gens):
                        _gf.write(f"Prompt: {prompt}\nStudent: {gen}\n")
                print(f"  Sample: {gens[0]}")

                # ── Persist curves incrementally ──────────────────────────────────
                if ppl_history:
                    ppl_path = os.path.join(cfg.output_dir, "ppl_curve.json")
                    with open(ppl_path, "w") as _pf:
                        json.dump({"ppl_curve": [{"step": s, "ppl": p} for s, p in ppl_history]}, _pf, indent=2)
                
                if loss_history:
                    loss_path = os.path.join(cfg.output_dir, "loss_curve.json")
                    keys = ["step", "train_loss", "train_m2t", "train_t2t", "test_loss", "test_m2t", "test_t2t"]
                    with open(loss_path, "w") as _lf:
                        json.dump({"loss_curve": [dict(zip(keys, row)) for row in loss_history]}, _lf, indent=2)

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
    ap.add_argument("--seq_len",                  type=int,   default=1024)
    ap.add_argument("--gradient_checkpointing",   action="store_true",
                    help="Recompute student activations during backward (saves ~60%% activation memory)")
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
    ap.add_argument("--eval_every",               type=int,   default=100)
    ap.add_argument("--test_size",                type=int,   default=256,
                    help="Number of examples reserved as a fixed test set")
    ap.add_argument("--test_eval_batches",        type=int,   default=8,
                    help="Mini-batches to average per test-loss evaluation")
    ap.add_argument("--save_every",               type=int,   default=2000)
    ap.add_argument("--weight_decay",             type=float, default=0.0)
    ap.add_argument("--batch_size",               type=int,   default=1)
    ap.add_argument("--grad_accum_steps",         type=int,   default=1)
    ap.add_argument("--warmup_steps",             type=int,   default=200,
                    help="Linear LR warmup steps before cosine decay begins.")
    ap.add_argument("--min_lr",                   type=float, default=1e-5,
                    help="Cosine decay floor LR.")
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
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = DistillConfig(
        teacher_model_path     = args.teacher_model,
        num_steps              = args.steps,
        seq_len                = args.seq_len,
        gradient_checkpointing = args.gradient_checkpointing,
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
        eval_every             = args.eval_every,
        test_size              = args.test_size,
        test_eval_batches      = args.test_eval_batches,
        save_every             = args.save_every,
        weight_decay           = args.weight_decay,
        warmup_steps           = args.warmup_steps,
        batch_size             = args.batch_size,
        grad_accum_steps       = args.grad_accum_steps,
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
    )
    distill_step1(cfg)
