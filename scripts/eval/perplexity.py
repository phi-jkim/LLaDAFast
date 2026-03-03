#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-1 Distillation Evaluation: Masked-Diffusion Perplexity
=============================================================

Evaluates a LLaDAFast student checkpoint (or a plain teacher) on a held-out
text corpus by estimating the MDLM NELBO (masked-diffusion perplexity).

MDLM NELBO (per token)
----------------------
For a masked-diffusion model the variational lower bound on log p(x) is:

    NELBO = E_{t ~ U[0,1]}  E_{m ~ Bern(t)^L}
                [ (1/|m|) * sum_{i: m_i=1} -log p_θ(x_i | x^m) ]

We estimate it with N_MC (t, m) samples per sequence and report
PPL = exp(NELBO).

Model loading
-------------
Two modes are supported:

  1. Baseline (teacher only):
       --model_path inclusionAI/LLaDA2.1-mini

  2. Stage-1 student checkpoint (delta applied on top of teacher):
       --model_path inclusionAI/LLaDA2.1-mini
       --checkpoint  llada_fast_distilled_step_4000

     The checkpoint directory must contain linear_attn_delta.pt produced by
     distill.py.  The teacher weights are loaded first; then the delta is
     patched in (strict=False), exactly mirroring the resume logic in distill.py.

Usage
-----
python scripts/eval/perplexity.py \\
    --model_path inclusionAI/LLaDA2.1-mini \\
    --checkpoint  llada_fast_distilled_step_4000 \\
    --dataset_name HuggingFaceFW/fineweb-edu \\
    --dataset_subset sample-10BT \\
    --split train \\
    --num_samples 512 \\
    --seq_len 512 \\
    --batch_size 4 \\
    --n_mc 4 \\
    --device cuda
"""

import argparse
import math
import os
import random
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from llada_fast.modeling.configuration_llada2_moe import LLaDA2MoeConfig
from llada_fast.modeling.modeling_llada2_moe import LLaDA2MoeModelLM

warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    # Model
    model_path: str = "inclusionAI/LLaDA2.1-mini"
    checkpoint: Optional[str] = None          # path to distilled checkpoint dir (delta mode)

    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: Optional[str] = "sample-10BT"
    split: str = "train"
    num_samples: int = 512                    # number of sequences to evaluate
    seq_len: int = 512                        # tokens per sequence

    # Eval
    batch_size: int = 4
    n_mc: int = 4                             # Monte Carlo (t, mask) samples per sequence
    t_min: float = 0.05                       # min noise level (matches distill.py clamp)
    t_max: float = 0.95                       # max noise level

    # Model architecture
    block_size: int = 32                      # block size for block-causal mask

    # Hardware
    device: str = "cuda"
    seed: int = 42

    # Output
    output_file: Optional[str] = None         # optional .txt path to save results


# ---------------------------------------------------------------------------
# Mask utilities  (mirrors distill.py)
# ---------------------------------------------------------------------------

def build_block_causal_mask(
    seq_len: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    4D block-causal mask (1, 1, seq_len, seq_len).

    Within a block: all positions attend bidirectionally.
    Across blocks: strictly causal (earlier blocks only).
    """
    num_blocks = (seq_len + block_size - 1) // block_size
    block_causal = torch.tril(
        torch.ones(num_blocks, num_blocks, device=device, dtype=torch.float32)
    )
    attend = (
        block_causal
        .repeat_interleave(block_size, dim=0)
        .repeat_interleave(block_size, dim=1)
        [:seq_len, :seq_len]
    )
    return attend.to(dtype=dtype).unsqueeze(0).unsqueeze(0)   # (1, 1, L, L)


def mask_tokens(
    input_ids: torch.Tensor,      # (B, L)
    pad_mask: torch.Tensor,       # (B, L) 1=real token, 0=padding
    mask_id: int,
    t: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly masks real (non-pad) tokens with probability t.

    Returns:
        noisy_ids:      (B, L)  input_ids with masked positions replaced by mask_id
        is_masked:      (B, L)  bool tensor, True at positions that were masked
    """
    B, L = input_ids.shape
    # Sample mask: Bernoulli(t) but only for real (non-padding) positions
    rand = torch.rand(B, L, device=input_ids.device)
    is_masked = (rand < t) & pad_mask.bool()

    # Stability: guarantee at least one real token unmasked per row so Z != 0
    for b in range(B):
        real_positions = pad_mask[b].bool().nonzero(as_tuple=True)[0]
        if len(real_positions) > 0 and is_masked[b, real_positions].all():
            is_masked[b, real_positions[0]] = False

    noisy_ids = input_ids.clone()
    noisy_ids[is_masked] = mask_id
    return noisy_ids, is_masked


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(cfg: EvalConfig) -> Tuple[LLaDA2MoeModelLM, AutoTokenizer]:
    """
    Loads a LLaDAFast model in one of two modes:

    Baseline mode  (cfg.checkpoint is None):
        Loads the raw teacher directly from cfg.model_path.

    Delta mode  (cfg.checkpoint is set):
        1. Loads teacher weights from cfg.model_path.
        2. Reconfigures the model with use_linear_attention=True.
        3. Copies teacher weights into the student.
        4. Patches in the saved linear-attention delta (strict=False).

    This is the exact inverse of the save logic in distill.py.
    """
    device = torch.device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)

    if cfg.checkpoint is None:
        # ---- Baseline: plain teacher ----------------------------------------
        print(f"[load] Loading teacher from {cfg.model_path}")
        model = LLaDA2MoeModelLM.from_pretrained(
            cfg.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device).eval()

    else:
        # ---- Student: teacher base + linear-attention delta -----------------
        # Support both naming conventions used in distill.py
        _candidates = ["linear_attn_delta.pt", "model_delta.pt"]
        delta_path = next(
            (os.path.join(cfg.checkpoint, n) for n in _candidates
             if os.path.exists(os.path.join(cfg.checkpoint, n))),
            None,
        )
        if delta_path is None:
            raise FileNotFoundError(
                f"No delta file found in {cfg.checkpoint}. "
                f"Expected one of: {_candidates}"
            )

        print(f"[load] Loading teacher base from {cfg.model_path}")
        teacher = LLaDA2MoeModelLM.from_pretrained(
            cfg.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()

        print(f"[load] Building student config with use_linear_attention=True")
        s_cfg = LLaDA2MoeConfig.from_pretrained(cfg.model_path, trust_remote_code=True)
        s_cfg.use_linear_attention = True
        s_cfg.block_size = cfg.block_size

        student = LLaDA2MoeModelLM(s_cfg).to(torch.bfloat16)
        student.load_state_dict(teacher.state_dict(), strict=False)
        del teacher

        print(f"[load] Patching delta from {delta_path}")
        delta = torch.load(delta_path, map_location="cpu")
        missing, unexpected = student.load_state_dict(delta, strict=False)
        print(f"[load] Delta applied. Missing={len(missing)}, Unexpected={len(unexpected)}")

        model = student.to(device).eval()

    print(f"[load] Model ready on {device}. "
          f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_text_batches(
    cfg: EvalConfig,
    tokenizer: AutoTokenizer,
) -> List[torch.Tensor]:
    """
    Loads `cfg.num_samples` tokenized sequences from a HuggingFace streaming
    dataset.  Returns a list of (input_ids, pad_mask) tuples, each of shape
    (cfg.batch_size, cfg.seq_len).
    """
    print(f"[data] Streaming {cfg.dataset_name} / {cfg.dataset_subset} ({cfg.split})")
    raw = load_dataset(
        cfg.dataset_name,
        name=cfg.dataset_subset,
        split=cfg.split,
        streaming=True,
    )

    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    batch_ids: List[torch.Tensor] = []
    batch_pads: List[torch.Tensor] = []
    collected = 0

    for ex in raw:
        if collected >= cfg.num_samples:
            break

        # Extract text (handles datasets where the text field may vary)
        text = ex.get("text") or next(
            (v for v in ex.values() if isinstance(v, str)), ""
        )
        if not text.strip():
            continue

        enc = tokenizer(
            text,
            return_tensors="pt",
            max_length=cfg.seq_len,
            truncation=True,
            padding="max_length",
        )
        batch_ids.append(enc.input_ids[0])
        batch_pads.append(enc.attention_mask[0])
        collected += 1

        if len(batch_ids) == cfg.batch_size:
            batches.append((
                torch.stack(batch_ids),   # (B, L)
                torch.stack(batch_pads),  # (B, L)
            ))
            batch_ids, batch_pads = [], []

    # Flush a partial final batch
    if batch_ids:
        batches.append((torch.stack(batch_ids), torch.stack(batch_pads)))

    print(f"[data] Loaded {collected} sequences in {len(batches)} batches")
    return batches


# ---------------------------------------------------------------------------
# Perplexity estimation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_perplexity(
    model: LLaDA2MoeModelLM,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    tokenizer: AutoTokenizer,
    cfg: EvalConfig,
) -> dict:
    """
    Estimates MDLM NELBO perplexity with Monte Carlo sampling.

    For each sequence we draw cfg.n_mc (t, mask) pairs and average the
    per-token NLL over masked positions. The final PPL = exp(mean NLL).

    Returns a dict with keys:
        ppl          – overall masked-diffusion PPL
        mean_nll     – mean per-token NLL
        n_sequences  – number of sequences evaluated
        n_tokens     – total real (non-pad) tokens seen
    """
    device = torch.device(cfg.device)
    dtype  = next(model.parameters()).dtype
    mask_id = tokenizer.mask_token_id
    assert mask_id is not None, "Tokenizer must have a mask_token_id"

    # Pre-build the block-causal mask prototype (reused across all batches)
    mask_proto = build_block_causal_mask(cfg.seq_len, cfg.block_size, device, dtype)

    total_nll   = 0.0
    total_tokens = 0
    n_sequences = 0

    for input_ids, pad_mask in tqdm(batches, desc="Evaluating", unit="batch"):
        B = input_ids.shape[0]
        input_ids = input_ids.to(device)
        pad_mask  = pad_mask.to(device)
        attn4d    = mask_proto.expand(B, -1, -1, -1)

        batch_nll   = 0.0
        batch_tokens = 0

        for _ in range(cfg.n_mc):
            # Sample noise level t ~ U[t_min, t_max]
            t = cfg.t_min + (cfg.t_max - cfg.t_min) * random.random()

            noisy_ids, is_masked = mask_tokens(input_ids, pad_mask, mask_id, t)

            # Forward pass
            out = model(
                input_ids=noisy_ids,
                attention_mask=attn4d,
                key_padding_mask=pad_mask.bool(),
            )
            logits = out.logits.float()          # (B, L, V) in fp32 for stable CE

            # Cross-entropy on masked positions only
            # Flatten to (B*L, V) and compute per-token NLL
            nll_flat = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                input_ids.view(-1),
                reduction="none",
            )                                    # (B*L,)

            masked_flat = is_masked.view(-1)     # (B*L,) bool
            if masked_flat.any():
                batch_nll    += nll_flat[masked_flat].sum().item()
                batch_tokens += masked_flat.sum().item()

        # Average over MC samples for this batch
        if batch_tokens > 0:
            total_nll    += batch_nll
            total_tokens += batch_tokens
        n_sequences += B

    mean_nll = total_nll / max(total_tokens, 1)
    ppl      = math.exp(min(mean_nll, 100))      # cap to avoid overflow in degenerate cases

    return {
        "ppl":         ppl,
        "mean_nll":    mean_nll,
        "n_sequences": n_sequences,
        "n_tokens":    total_tokens,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(results: dict, cfg: EvalConfig) -> None:
    lines = [
        "",
        "=" * 52,
        "  LLaDAFast Stage-1 Perplexity Evaluation",
        "=" * 52,
        f"  Model:        {cfg.model_path}",
        f"  Checkpoint:   {cfg.checkpoint or '(none — baseline teacher)'}",
        f"  Dataset:      {cfg.dataset_name} / {cfg.dataset_subset}",
        f"  Split:        {cfg.split}",
        f"  Sequences:    {results['n_sequences']}",
        f"  Masked tokens:{results['n_tokens']:,}",
        f"  MC samples:   {cfg.n_mc}  (t ~ U[{cfg.t_min}, {cfg.t_max}])",
        "-" * 52,
        f"  Mean NLL:     {results['mean_nll']:.4f}",
        f"  PPL:          {results['ppl']:.2f}",
        "=" * 52,
        "",
    ]
    output = "\n".join(lines)
    print(output)

    if cfg.output_file:
        os.makedirs(os.path.dirname(cfg.output_file) or ".", exist_ok=True)
        with open(cfg.output_file, "w") as f:
            f.write(output)
        print(f"[report] Results saved to {cfg.output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(
        description="MDLM perplexity evaluation for LLaDAFast stage-1 checkpoints"
    )
    p.add_argument("--model_path",      default="inclusionAI/LLaDA2.1-mini")
    p.add_argument("--checkpoint",      default=None,
                   help="Path to distilled checkpoint dir (contains linear_attn_delta.pt). "
                        "Omit for baseline teacher evaluation.")
    p.add_argument("--dataset_name",    default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_subset",  default="sample-10BT")
    p.add_argument("--split",           default="train")
    p.add_argument("--num_samples",     type=int,   default=512)
    p.add_argument("--seq_len",         type=int,   default=512)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--n_mc",            type=int,   default=4,
                   help="Monte Carlo (t, mask) samples per sequence")
    p.add_argument("--t_min",           type=float, default=0.05)
    p.add_argument("--t_max",           type=float, default=0.95)
    p.add_argument("--block_size",      type=int,   default=32)
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--output_file",     default=None,
                   help="Optional path to save the results as a .txt file")

    a = p.parse_args()
    return EvalConfig(**vars(a))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = parse_args()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    model, tokenizer = load_model(cfg)
    batches = load_text_batches(cfg, tokenizer)
    results = estimate_perplexity(model, batches, tokenizer, cfg)
    report(results, cfg)


if __name__ == "__main__":
    main()
