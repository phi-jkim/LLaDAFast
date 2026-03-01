#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLaDAFast Distillation Script (Step 1: Feature Distillation)

This script performs "Teacher-Forcing" distillation where a student model (using Linear Attention)
learns to match the attention outputs of a frozen teacher model (LLaDA2.1 with Softmax Attention).

Key Paradigms:
1. **LOLCATS Teacher Forcing**: 
   - We run the Teacher on cuda:0 and the Student on cuda:1.
   - For every supervised layer, we "force" the student's forward pass by replacing its 
     attention sublayer output with the Teacher's ground-truth output.
   - This ensures the Student stays on the "clean manifold" of the teacher during training,
     preventing error accumulation across layers.
2. **Clean Past, Noisy Present**: 
   - We simulate the state of the model during Block-Parallel Decoding.
   - We pick one random block (32 tokens) to be the "noisy present" (corrupted with masks).
   - Everything *before* this block is kept as "clean past" (ground truth tokens).
3. **Numerical Stability**: 
   - We accumulate losses in fp32 and use explicit NaN/Inf guards to skip divergent steps.
   - Noise (t) is capped at [0.05, 0.95] to avoid degenerate all-mask or no-mask scenarios.
"""

import argparse
import math
import os
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from llada_fast.modeling.modeling_llada2_moe import LLaDA2MoeModelLM
from llada_fast.modeling.configuration_llada2_moe import LLaDA2MoeConfig

# Silence Transformers logging to keep the terminal clean from formatting warnings
warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()

# Fix #4: Enable TF32 + “high” matmul precision (Free speed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# -----------------------------
# Mask Building (The "Staircase")
# -----------------------------
def build_block_causal_mask_proto(
    seq_len: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Builds a 4D Block-Causal Mask (B, 1, L, L).
    
    This is the "Staircase" mask that enables block-parallel decoding:
    - Within a 32-token block: Tokens see each other bidirectionally (Parallel Denoising).
    - Across blocks: Tokens can ONLY see the past, never the future (Causal constraint).
    """
    num_blocks = (seq_len + block_size - 1) // block_size
    # Create the block-level causal triangle
    blk = torch.tril(torch.ones(num_blocks, num_blocks, device=device, dtype=torch.float32))
    # Interleave to stretch the 1x1 block causality into a 32x32 square causality
    attend = (
        blk.repeat_interleave(block_size, dim=0)
           .repeat_interleave(block_size, dim=1)
           [:seq_len, :seq_len]
    )
    # Ensure dtype matches the model's query dtype to satisfy SDPA's bias requirements
    return attend.to(dtype=dtype).unsqueeze(0).unsqueeze(0)  # (1,1,L,L)


# -----------------------------
# Denoising Paradigm: Clean Past, Noisy Present
# -----------------------------
def corrupt_one_block_with_mask(
    input_ids: torch.Tensor,          # (B,L)
    attn_2d: torch.Tensor,            # (B,L) 1=real token, 0=padding
    mask_id: int,
    block_size: int,
    t: float,
    block_idx: Optional[int] = None,
) -> torch.Tensor:
    """
    Simulates the model's state during generation:
    1. Past blocks: Already fully denoised (Ground Truth).
    2. Current block (The "Present"): Corrupted with [MASK] tokens at rate `t`.
    3. Future blocks: Unseen (masked by the block-causal mask during forward).
    
    This teaches the student that it can trust its clean past and must reconstruct the noisy current block.
    """
    B, L = input_ids.shape
    noisy = input_ids.clone()

    # Pre-denoise any padding with mask_id so the model ignores them
    for b in range(B):
        real_len = int(attn_2d[b].sum().item())
        if real_len < L:
            noisy[b, real_len:] = mask_id

        # Pick one random block (if not provided)
        if block_idx is None:
            num_blocks_real = max(1, (real_len + block_size - 1) // block_size)
            bi = random.randrange(num_blocks_real)
        else:
            bi = block_idx
            
        s = bi * block_size
        e = min((bi + 1) * block_size, real_len)

        if e <= s:
            continue

        # CORRUPTION: Replace tokens with [MASK] according to probability t
        n = e - s
        m = (torch.rand(n, device=noisy.device) < t)
        
        # Stability check: Ensure at least one token stays unmasked so linear attention state (Z) has signal
        if m.all():
            m[0] = False
            
        noisy[b, s:e][m] = mask_id

        # MASK FUTURE: Everything after the noisy block is also masked out
        # This aligns the training distribution with the generation assumption.
        if e < real_len:
            noisy[b, e:real_len] = mask_id

    return noisy


# -----------------------------
# Evaluation: Blockwise Generation
# -----------------------------
@torch.no_grad()
def eval_generation(
    model: LLaDA2MoeModelLM,
    tokenizer,
    prompts: Sequence[str],
    device: torch.device,
    max_new_tokens: int,
    block_length: int,
    steps: int,
    threshold: float = 0.7,
    editing_threshold: float = 0.5,
    max_post_steps: int = 16,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> List[str]:
    """
    Uses the model's custom .generate() implementation to verify quality progress.
    This simulates the real production-time decoding loop.
    """
    model.eval()
    outs: List[str] = []
    
    # Extract ground-truth IDs from tokenizer
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    print(f"tokenizer eos/pad/mask: {eos_id}, {pad_id}, {mask_id}")
    assert mask_id is not None

    for p in prompts:
        # Fix: Apply chat template for instruction-tuned LLaDA (Fix #16)
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(device)
        input_ids = enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "input_ids") else enc

        gen_ids = model.generate(
            inputs=input_ids,
            temperature=temperature,
            block_length=block_length,
            steps=steps,
            gen_length=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            eos_early_stop=True,  # Enable for quality eval
            threshold=threshold,
            editing_threshold=editing_threshold,
            max_post_steps=max_post_steps,
            eos_id=eos_id,
            mask_id=mask_id,
        )
        # Decode the full generated continuation
        outs.append(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
    model.train()
    return outs


# -----------------------------
# LOLCATS: Teacher Collection Hooks
# -----------------------------
def _get_attn_out(out):
    return out[0] if isinstance(out, (tuple, list)) else out

def install_teacher_collection_hooks(
    teacher: LLaDA2MoeModelLM,
    layer_ids: Sequence[int],
    store: Dict[int, torch.Tensor],
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Records the 'Correct' attention behavior from the Teacher.
    We capture out[0], which is the projected attention result BEFORE the residual add.
    """
    handles = []

    def make_hook(li: int):
        def hook(module, inp, out):
            # store[li] becomes the target for the student's MSE loss
            store[li] = _get_attn_out(out).detach() # detach to stop gradients from flowing to teacher
            return None
        return hook

    for li in layer_ids:
        h = teacher.model.layers[li].attention.register_forward_hook(make_hook(li))
        handles.append(h)

    return handles


# -----------------------------
# LOLCATS: Student Teacher-Forcing Hooks
# -----------------------------
def install_student_hooks(
    student: LLaDA2MoeModelLM,
    layer_ids: Sequence[int],
    student_record: Dict[int, torch.Tensor],
    teacher_targets: Dict[int, torch.Tensor], # Ground truth activations
    p_force_dict: Dict[int, float],
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Activation Teacher Forcing (LoLCATs):
    1. Records the student's raw predicted attention output for MSE loss.
    2. REPLACES it with the teacher's target to prevent error compounding.
    """
    handles = []

    def make_hook(li: int):
        def hook(module, inp, out):
            if not module.training:
                return out
            
            raw_out = _get_attn_out(out)
            # Save the student's current prediction (Raw) for loss calculation
            student_record[li] = raw_out
            
            # Activation Teacher Forcing: Replace student output with teacher's GROUND TRUTH
            # Probability per-layer determines if we apply forcing this step
            force_prob = p_force_dict.get(li, 1.0)
            if li in teacher_targets and random.random() < force_prob:
                t_val = teacher_targets[li]
                # Match shape/device if necessary, though they should match by design
                t_val_cast = t_val.to(raw_out.device, dtype=raw_out.dtype)
                if isinstance(out, (tuple, list)):
                    return (t_val_cast,) + tuple(out)[1:]
                return t_val_cast
            
            return out
        return hook

    for li in layer_ids:
        h = student.model.layers[li].attention.register_forward_hook(make_hook(li))
        handles.append(h)

    return handles


# -----------------------------
# Distillation Pipeline
# -----------------------------
def distill_step1_teacher_forcing(
    teacher_model_path: str,
    dataset_name: str,
    dataset_subset: Optional[str],
    seq_len: int,
    num_steps: int,
    learning_rate: float,
    device_teacher: str,
    device_student: str,
    distill_layers: Optional[List[int]],
    linear_layers: Optional[List[int]],
    block_size_override: Optional[int],
    progressive_interval: int,
    force_decay_length: int,
    eval_every: int,
    eval_prompts: List[str],
    eval_gen_len: int,
    eval_block_length: int,
    eval_steps: int,
    eval_threshold: float = 0.95,
    eval_editing_threshold: float = 0.9,
    eval_max_post_steps: int = 16,
    alpha: float = 1.0,
    beta: float = 1.0,
    temperature: float = 1.0,
):
    # Setup
    dev0 = torch.device(device_teacher)
    dev1 = torch.device(device_student)

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, trust_remote_code=True)
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    print(f"tokenizer eos/pad/mask: {eos_id}, {pad_id}, {mask_id}")

    # 1. Load Teacher (Frozen)
    teacher = LLaDA2MoeModelLM.from_pretrained(
        teacher_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(dev0).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 2. Setup Student (Linear Attention Student)
    s_config = LLaDA2MoeConfig.from_pretrained(teacher_model_path, trust_remote_code=True)
    if progressive_interval > 0:
        s_config.use_linear_attention = True
        s_config.linear_attention_layers = None # Init all linear, we'll gate them dynamically
    elif linear_layers is not None:
        s_config.use_linear_attention = False
        s_config.linear_attention_layers = linear_layers
    else:
        s_config.use_linear_attention = True
        
    # Do NOT unilaterally set use_qk_norm unless teacher has it
    if block_size_override is not None:
        s_config.block_size = block_size_override

    student = LLaDA2MoeModelLM(s_config).to(torch.bfloat16).to(dev1)
    ret = student.load_state_dict(teacher.state_dict(), strict=False)
    print(f"Student loaded. Missing keys: {len(ret.missing_keys)}, Unexpected keys: {len(ret.unexpected_keys)}")
    if len(ret.missing_keys) > 0:
        print(f"Sample missing keys: {ret.missing_keys[:10]}")


    # Param filtering & Progressive State Init
    n_layers = len(teacher.model.layers)
    layer_activation_steps = {} # When did each layer become linear?
    p_force_dict = {}           # Current forcing probability per layer
    
    if progressive_interval > 0:
        active_layers = [n_layers - 1] # Start with just the last layer
        layer_activation_steps[n_layers-1] = 0
    else:
        active_layers = linear_layers if linear_layers is not None else list(range(n_layers))
        for li in active_layers:
            layer_activation_steps[li] = 0

    active_attention_prefixes = [f"model.layers.{i}.attention." for i in active_layers]
    for i, layer in enumerate(student.model.layers):
        layer.attention.is_linear_active = (i in active_layers)

    for name, p in student.named_parameters():
        p.requires_grad = any(name.startswith(prefix) for prefix in active_attention_prefixes)

        
    trainable = [(n, p.numel()) for n, p in student.named_parameters() if p.requires_grad]
    print(f"\nTrainable parameter groups: {len(trainable)}")
    print(f"Total trainable params: {sum(x for _, x in trainable):,}")
    print(f"Sample trainable params: {trainable[:5]}\n")

    optimizer = torch.optim.AdamW(
        (p for p in student.parameters() if p.requires_grad),
        lr=learning_rate,
    )

    # Handle layer selection
    layer_ids = distill_layers or list(range(n_layers))

    # Build Mask Prototypes (must match and expanded appropriately per batch)
    block_size = int(getattr(s_config, "block_size", 32))
    mask_proto0 = build_block_causal_mask_proto(seq_len, block_size, dev0, teacher.dtype)
    mask_proto1 = build_block_causal_mask_proto(seq_len, block_size, dev1, student.dtype)

    # Dataset
    data = load_dataset(dataset_name, name=dataset_subset, split="train", streaming=True)
    data_iter = iter(data)

    # LOLCATS Store
    t_store_cpu: Dict[int, torch.Tensor] = {} # Teacher outputs on dev0
    t_store_student: Dict[int, torch.Tensor] = {} # Teacher outputs moved to dev1
    s_record: Dict[int, torch.Tensor] = {}

    # Fix: Ensure heads are in fp32 for stable logit computation (Fix #19)
    teacher.lm_head.float()
    student.lm_head.float()

    # Initial Hook Installation
    t_handles = install_teacher_collection_hooks(teacher, layer_ids, t_store_cpu)
    s_handles = install_student_hooks(student, layer_ids, s_record, t_store_student, p_force_dict)

    pbar = tqdm(total=num_steps, desc="Distillation")

    step = 0
    while step < num_steps:
        # STEP E: PERIODIC EVAL
        # Trigger eval at start (step 0) and then every eval_every steps
        if eval_every > 0 and step % eval_every == 0:
            print(f"\n--- Evaluation Step {step} ---")
            
            # TEACHER Eval (Sanity Check)
            print("TEACHER Generation:")
            with torch.no_grad():
                t_gens = eval_generation(teacher, tokenizer, eval_prompts, dev0, eval_gen_len, 
                                        eval_block_length, eval_steps, 
                                        threshold=eval_threshold, editing_threshold=eval_editing_threshold, 
                                        max_post_steps=eval_max_post_steps, temperature=0.0)
                for p, g in zip(eval_prompts, t_gens):
                    print(f" Prompt: {p}\n Teacher: {g}\n")

            # STUDENT Eval
            print("STUDENT Generation:")
            student.eval()
            with torch.no_grad():
                gens = eval_generation(student, tokenizer, eval_prompts, dev1, eval_gen_len, 
                                       eval_block_length, eval_steps,
                                       threshold=eval_threshold, editing_threshold=eval_editing_threshold, 
                                       max_post_steps=eval_max_post_steps, temperature=0.0)
                for p, g in zip(eval_prompts, gens):
                    print(f" Prompt: {p}\n Student: {g}\n")
            student.train()

        # Data preparation
        try:
            ex = next(data_iter)
        except StopIteration:
            break
            
        text = ex.get("text") or next(v for v in ex.values() if isinstance(v, str))
        if not text.strip():
            continue

        # ... (rest of data prep and noisy block logic) ...
        enc = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True, padding="max_length")
        input_ids = enc.input_ids
        attn2d = enc.attention_mask
        B = input_ids.shape[0]

        # NOISE: Pick a block and corrupt it inside the "Clean Past, Noisy Present" paradigm
        # We pick a single block for the batch for efficient KL slicing
        real_len = int(attn2d[0].sum().item())
        num_blocks_total = max(1, (real_len + block_size - 1) // block_size)
        block_idx = random.randrange(num_blocks_total)
        s, e = block_idx * block_size, min((block_idx + 1) * block_size, real_len)

        t_noise = 0.05 + 0.90 * random.random() # Clamp t to [0.05, 0.95] for numerical health
        noisy0 = corrupt_one_block_with_mask(input_ids.to(dev0), attn2d.to(dev0), mask_id, block_size, t_noise, block_idx=block_idx)
        attn4d0 = mask_proto0[:, :, :seq_len, :seq_len].expand(B, -1, -1, -1)

        t_store_cpu.clear()
        with torch.no_grad():
            # Call the underlying model to get hidden states (avoids full head projection)
            t_model_out = teacher.model(noisy0, attention_mask=attn4d0, key_padding_mask=attn2d.to(dev0).bool())
            t_hs = t_model_out.last_hidden_state
            
            # Fix: Compute teacher logits in explicit fp32 without autocast (Fix #19)
            # lm_head is now in float()
            with torch.autocast(device_type="cuda", enabled=False):
                t_logits_block = teacher.lm_head(t_hs[:, s:e, :].float())
            t_logits_block = t_logits_block.to(dev1, non_blocking=True)

        # Optimization: Pre-copy all teacher tensors to student device once per step
        t_store_student.clear()
        for li, ten in t_store_cpu.items():
            t_store_student[li] = ten.to(device=dev1, dtype=student.dtype, non_blocking=True)

        # STEP B: STUDENT PASS (Trains student while forced by t_store_student)
        s_record.clear()
        noisy1 = noisy0.to(dev1)
        attn4d1 = mask_proto1[:, :, :seq_len, :seq_len].expand(B, -1, -1, -1)
        # Use student.model to save memory (no full sequence head projection)
        s_model_out = student.model(noisy1, attention_mask=attn4d1, key_padding_mask=attn2d.to(dev1).bool())
        s_hs = s_model_out.last_hidden_state
        
        # Fix: Compute student logits in explicit fp32 without autocast (Fix #19)
        # lm_head is now in float()
        with torch.autocast(device_type="cuda", enabled=False):
            s_logits_block = student.lm_head(s_hs[:, s:e, :].float())
            
        # Hook shape validation (to ensure Activation Forcing didn't silently shape mismatch)
        if len(layer_ids) > 0:
            chk_li = layer_ids[0]
            if chk_li in t_store_student and chk_li in s_record:
                t_shape = t_store_student[chk_li].shape
                s_shape = s_record[chk_li].shape
                assert t_shape == s_shape, f"Hook shape mismatch at Layer {chk_li}! Teacher: {t_shape}, Student: {s_shape}"


        # DEBUG: Check Layer 0 stats for NaNs (Root cause detection)
        if 0 in s_record:
            x_test = s_record[0]
            if torch.isnan(x_test).any() or torch.isinf(x_test).any():
                print(f"\n[DEBUG] Layer 0 NaN/Inf Alert! max_abs={x_test.float().abs().max().item():.2e}")

        # Fix: Pinpoint stability checks for logits (Fix #21)
        if not torch.isfinite(t_logits_block).all():
            print(f"Teacher logits have inf/nan: max_abs={t_logits_block.float().abs().max().item():.2e}")
        if not torch.isfinite(s_logits_block).all():
            print(f"Student logits have inf/nan: max_abs={s_logits_block.float().abs().max().item():.2e}")

        # STEP C: CURRICULUM & STABILITY
        # Progressive Activation Curriculum
        if progressive_interval > 0:
            expected_active_count = 1 + step // progressive_interval
            expected_active_count = min(expected_active_count, n_layers)
            while len(active_layers) < expected_active_count:
                next_layer = n_layers - 1 - len(active_layers)
                if next_layer < 0:
                    break
                active_layers.append(next_layer)
                student.model.layers[next_layer].attention.is_linear_active = True
                layer_activation_steps[next_layer] = step
                
                # Unfreeze parameters dynamically
                new_params = []
                for name, p in student.model.layers[next_layer].attention.named_parameters():
                    p.requires_grad = True
                    new_params.append(p)
                
                if new_params:
                    optimizer.add_param_group({'params': new_params})
                    
                print(f"\n[CURRICULUM] Step {step}: Layer {next_layer} activated for Linear Attention!")
                
        # Forcing Decay Curriculum
        for li in active_layers:
            if force_decay_length > 0:
                age = step - layer_activation_steps.get(li, 0)
                p_force_dict[li] = max(0.0, 1.0 - (age / force_decay_length))
            else:
                p_force_dict[li] = 1.0

        skip = False
        for i in layer_ids:
            if i in s_record and not torch.isfinite(s_record[i]).all():
                print(f"skip due to NaN at effective step {step} (Layer {i})")
                skip = True
                break
        if skip:
            continue

        # STEP D: LOSS & BACKPROP
        # 1. MSE over attention outputs, masked only for real tokens
        valid = attn2d.to(dev1).unsqueeze(-1).to(dtype=student.dtype)
        real_len = int(attn2d.sum().item())
        mse_loss = 0.0
        
        computed_layers = [i for i in layer_ids if i in active_layers]
        for i in computed_layers:
            s_val = s_record[i]
            t_val = t_store_student[i]
            # Fix: Compute MSE loss in fp32 for stability (Fix #22)
            mse_loss_i = F.mse_loss(
                (s_val * valid).float(), 
                (t_val * valid).float(), 
                reduction="sum"
            ) / (real_len * s_val.shape[-1])
            mse_loss = mse_loss + mse_loss_i
        
        # 2. Logits KL Distillation (Only computed on the active noise block)
        # Fix: Skip KL computation entirely if beta=0.0 to avoid NaN poisoning (Fix #20)
        if beta != 0.0:
            kl_loss = F.kl_div(
                F.log_softmax(s_logits_block.float() / temperature, dim=-1),
                F.softmax(t_logits_block.float() / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature * temperature)
        else:
            kl_loss = torch.zeros((), device=dev1, dtype=torch.float32)

        loss = alpha * mse_loss + beta * kl_loss

        if not torch.isfinite(loss):
            print(f"skip due to NaN/Inf loss at effective step {step}")
            continue

        loss.backward()
        # Correct Gradient Clipping: Only target parameters with requires_grad=True
        torch.nn.utils.clip_grad_norm_((p for p in student.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Progress tracking
        pbar.set_description(f"loss={float(loss.item()):.6f}")
        step += 1
        pbar.update(1)



    # Cleanup
    pbar.close()
    for h in t_handles + s_handles: h.remove()
    student.save_pretrained("./student_forcing_final")
    print("\nDistillation Complete.")


# -----------------------------
# CLI Entrypoint
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_model", type=str, default="inclusionAI/LLaDA2.1-mini", help="Path to teacher")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--device_teacher", type=str, default="cuda:0")
    ap.add_argument("--device_student", type=str, default="cuda:1")
    ap.add_argument("--alpha", type=float, default=1.0, help="MSE Hidden weight")
    ap.add_argument("--beta", type=float, default=1.0, help="KL Logits weight")
    ap.add_argument("--temp", type=float, default=1.0, help="KL Temperature")
    ap.add_argument("--linear_layers", type=str, default="", help="Comma separated list of layer IDs to swap to linear attention. If empty, all layers are swapped.")
    ap.add_argument("--progressive_interval", type=int, default=0, help="Steps between activating new layers top-down")
    ap.add_argument("--force_decay_length", type=int, default=0, help="Steps over which a layer's force probability decays to 0")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    distill_step1_teacher_forcing(
        teacher_model_path=args.teacher_model,
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_subset="sample-10BT",
        seq_len=1024,
        num_steps=args.steps,
        learning_rate=args.lr,
        device_teacher=args.device_teacher,
        device_student=args.device_student,
        distill_layers=None,
        linear_layers=[int(x) for x in args.linear_layers.split(",")] if args.linear_layers else None,
        block_size_override=None,
        progressive_interval=args.progressive_interval,
        force_decay_length=args.force_decay_length,
        eval_every=100,
        eval_prompts=["Write a story about a fast cat."],
        eval_gen_len=128,
        eval_block_length=32,
        eval_steps=32,
        eval_threshold=0.7,
        eval_editing_threshold=0.5,
        eval_max_post_steps=16,
        alpha=args.alpha,
        beta=args.beta,
        temperature=args.temp,
    )