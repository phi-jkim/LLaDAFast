"""Stage-2 LoRA fine-tuning for linearized LLaDA2 (hybrid block from stage 1).

Single-pass M2T + T2T over ALL blocks simultaneously, batch_size=1 with
gradient accumulation, reserved test set for perplexity. No teacher required.
Uses Alpaca (instruction data) to match LLaDA2.1-mini's instruct training.
"""
import argparse
import glob
import math
import os
import random
import shutil

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer

from llada_fast.modeling.configuration_llada2_moe import LLaDA2MoeConfig
from llada_fast.modeling.modeling_llada2_moe import LLaDA2MoeModelLM
from llada_fast.training.distill.data import (
    build_block_causal_mask,
    corrupt_all_blocks,
    corrupt_all_blocks_t2t,
)


# ── Alpaca data loader ────────────────────────────────────────────────────────

class AlpacaLoader:
    """
    Loads tatsu-lab/alpaca, formats each example with the tokenizer's chat
    template (or a plain fallback), and tokenizes to seq_len.

    First `test_size` examples are reserved as a fixed test set; the remaining
    examples are shuffled and cycled for training.
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int,
        test_size: int = 256,
        dataset_name: str = "tatsu-lab/alpaca",
    ):
        ds = load_dataset(dataset_name, split="train")
        has_template = getattr(tokenizer, "chat_template", None) is not None

        texts = []
        for ex in ds:
            instruction = ex.get("instruction", "")
            inp         = ex.get("input", "")
            output      = ex.get("output", "")
            content = instruction + ("\n" + inp if inp.strip() else "")
            if has_template:
                text = tokenizer.apply_chat_template(
                    [{"role": "user",      "content": content},
                     {"role": "assistant", "content": output}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                text = f"### Instruction:\n{content}\n\n### Response:\n{output}"
            texts.append(text)

        self._texts     = texts
        self._tok       = tokenizer
        self._seq_len   = seq_len
        self._n         = len(texts)
        self._test_size = min(test_size, self._n // 8)

        train_idx = list(range(self._test_size, self._n))
        random.shuffle(train_idx)
        self._train_order = train_idx
        self._pos = 0

        print(f"[Alpaca] {self._n} examples  "
              f"test={self._test_size}  train={len(self._train_order)}")

    def _encode(self, text):
        enc = self._tok(
            text,
            truncation=True,
            max_length=self._seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        return enc.input_ids, enc.attention_mask   # (1, L), (1, L)

    def next_train(self):
        if self._pos >= len(self._train_order):
            random.shuffle(self._train_order)
            self._pos = 0
        idx = self._train_order[self._pos]
        self._pos += 1
        return self._encode(self._texts[idx])

    def next_test(self):
        idx = random.randint(0, self._test_size - 1)
        return self._encode(self._texts[idx])


# ── LR scheduler ──────────────────────────────────────────────────────────────

def _build_scheduler(optimizer, num_steps, warmup_steps, min_lr, lr, last_epoch=0):
    def _lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, num_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr / lr) + (1.0 - min_lr / lr) * cosine
    # last_epoch - 1: LambdaLR calls step() once on init, advancing to last_epoch.
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch=last_epoch - 1)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def _save(model, optimizer, scheduler, step, output_dir):
    """Save trainable delta + optimizer. Keeps only the two most recent checkpoints."""
    pths = sorted(
        glob.glob(os.path.join(output_dir, "step_*")),
        key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0,
    )
    if len(pths) >= 2:
        for p in pths[:-1]:
            shutil.rmtree(p, ignore_errors=True)

    out = os.path.join(output_dir, f"step_{step}")
    os.makedirs(out, exist_ok=True)
    delta = {n: p.cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(delta,                       os.path.join(out, "lora_delta.pt"))
    torch.save(optimizer.state_dict(),      os.path.join(out, "optimizer.pt"))
    torch.save({"step": step},              os.path.join(out, "train_state.pt"))
    print(f"\n[SAVE] step_{step}  ({len(delta)} trainable tensors)")


# ── Perplexity eval ───────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_ppl(model, loader, mask_id, block_size, attn_proto, device, n_batches=16):
    model.eval()
    ces = []
    for _ in range(n_batches):
        ids, pmask = loader.next_test()
        ids   = ids.to(device)
        pmask = pmask.to(device)

        num_blocks  = (ids.shape[1] + block_size - 1) // block_size
        t_per_block = 0.05 + 0.90 * torch.rand(num_blocks, device=device)

        noisy, corrupted = corrupt_all_blocks(ids, pmask, mask_id, block_size, t_per_block)
        if not corrupted.any():
            continue

        out = model(
            noisy,
            attention_mask=attn_proto.expand(1, -1, -1, -1),
            key_padding_mask=pmask.bool(),
        )
        ce = F.cross_entropy(out.logits[0][corrupted[0]], ids[0][corrupted[0]])
        if torch.isfinite(ce):
            ces.append(ce.item())

    model.train()
    return math.exp(sum(ces) / len(ces)) if ces else float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def train(cfg):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model_path, trust_remote_code=True)
    mask_id = tokenizer.mask_token_id
    assert mask_id is not None, "tokenizer must expose mask_token_id"

    special_ids = set(tokenizer.all_special_ids)
    vocab_ids = torch.tensor(
        [i for i in range(tokenizer.vocab_size) if i not in special_ids],
        dtype=torch.long, device=device,
    )

    # ── Build student model ────────────────────────────────────────────────────
    config = LLaDA2MoeConfig.from_pretrained(cfg.teacher_model_path)
    config.use_linear_attention    = True
    config.use_qk_norm             = True
    config.use_block_softmax_hybrid = True   # stage 1 used hybrid block
    block_size = int(getattr(config, "block_size", 32))

    model = LLaDA2MoeModelLM.from_pretrained(
        cfg.teacher_model_path, config=config,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    # Load stage-1 trained delta (hedgehog_weights + alpha)
    if cfg.stage1_checkpoint:
        delta_path = os.path.join(cfg.stage1_checkpoint, "linear_attn_delta.pt")
        if os.path.exists(delta_path):
            ret = model.load_state_dict(
                torch.load(delta_path, map_location="cpu"), strict=False
            )
            print(f"[INIT] Loaded stage-1 delta. Missing keys: {len(ret.missing_keys)}")
        else:
            print(f"[WARN] No delta at {delta_path} — starting from teacher weights.")

    # ── LoRA on base projections ───────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_rank * 2,
        target_modules=["query_key_value", "dense", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    # Keep linear-attention-specific params trainable (trained in stage 1).
    for name, p in model.named_parameters():
        if "hedgehog_weights" in name or name.endswith(".alpha"):
            p.requires_grad = True

    model = model.to(device)
    model.train()
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=0.01,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    loader = AlpacaLoader(
        tokenizer, cfg.seq_len,
        test_size=cfg.test_size,
        dataset_name=cfg.alpaca_dataset,
    )

    attn_proto = build_block_causal_mask(cfg.seq_len, block_size, device, torch.bfloat16)

    # ── Resume ────────────────────────────────────────────────────────────────
    step = 0
    if cfg.resume_from and os.path.exists(cfg.resume_from):
        ts_path = os.path.join(cfg.resume_from, "train_state.pt")
        if os.path.exists(ts_path):
            step = torch.load(ts_path, map_location="cpu").get("step", 0)
        delta_path = os.path.join(cfg.resume_from, "lora_delta.pt")
        if os.path.exists(delta_path):
            model.load_state_dict(
                torch.load(delta_path, map_location="cpu"), strict=False
            )
            print(f"[RESUME] Loaded LoRA delta from {cfg.resume_from}")
        opt_path = os.path.join(cfg.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        print(f"[RESUME] Starting from step {step}")

    scheduler = _build_scheduler(
        optimizer, cfg.num_steps, cfg.warmup_steps, cfg.min_lr, cfg.lr,
        last_epoch=step,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    os.makedirs(cfg.output_dir, exist_ok=True)
    pbar   = tqdm(total=cfg.num_steps, initial=step)
    m2t_ema = t2t_ema = None

    while step < cfg.num_steps:
        optimizer.zero_grad(set_to_none=True)
        acc_m2t = acc_t2t = 0.0
        acc_count = 0

        for _ in range(cfg.grad_accum_steps):
            ids, pmask = loader.next_train()
            ids   = ids.to(device)
            pmask = pmask.to(device)

            num_blocks  = (cfg.seq_len + block_size - 1) // block_size
            t_per_block = 0.05 + 0.90 * torch.rand(num_blocks, device=device)
            attn        = attn_proto.expand(1, -1, -1, -1)
            kpm         = pmask.bool()

            # ── M2T ───────────────────────────────────────────────────────────
            noisy_m2t, corrupted_m2t = corrupt_all_blocks(
                ids, pmask, mask_id, block_size, t_per_block
            )
            out_m2t  = model(noisy_m2t, attention_mask=attn, key_padding_mask=kpm)
            loss_m2t = (
                F.cross_entropy(out_m2t.logits[0][corrupted_m2t[0]], ids[0][corrupted_m2t[0]])
                if corrupted_m2t.any() else out_m2t.logits.sum() * 0.0
            )

            # ── T2T ───────────────────────────────────────────────────────────
            noisy_t2t, corrupted_t2t = corrupt_all_blocks_t2t(
                ids, pmask, vocab_ids, mask_id, block_size, t_per_block
            )
            out_t2t  = model(noisy_t2t, attention_mask=attn, key_padding_mask=kpm)
            loss_t2t = (
                F.cross_entropy(out_t2t.logits[0][corrupted_t2t[0]], ids[0][corrupted_t2t[0]])
                if corrupted_t2t.any() else out_t2t.logits.sum() * 0.0
            )

            loss = (cfg.omega_m2t * loss_m2t + cfg.omega_t2t * loss_t2t) / cfg.grad_accum_steps
            if torch.isfinite(loss):
                loss.backward()
                acc_m2t   += float(loss_m2t)
                acc_t2t   += float(loss_t2t)
                acc_count += 1

        if acc_count > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()

            avg_m2t = acc_m2t / acc_count
            avg_t2t = acc_t2t / acc_count
            m2t_ema = avg_m2t if m2t_ema is None else 0.95 * m2t_ema + 0.05 * avg_m2t
            t2t_ema = avg_t2t if t2t_ema is None else 0.95 * t2t_ema + 0.05 * avg_t2t

        step += 1
        pbar.update(1)
        pbar.set_description(
            f"m2t={m2t_ema:.3f}  t2t={t2t_ema:.3f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if step % cfg.eval_every == 0:
            ppl = _eval_ppl(
                model, loader, mask_id, block_size, attn_proto, device,
                n_batches=cfg.test_eval_batches,
            )
            tqdm.write(f"--- Eval @ step {step} | ppl={ppl:.1f} ---")

        if step % cfg.save_every == 0:
            _save(model, optimizer, scheduler, step, cfg.output_dir)

    pbar.close()
    _save(model, optimizer, scheduler, step, cfg.output_dir)
    print("Stage-2 LoRA training complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_model_path",  default="inclusionAI/LLaDA2.1-mini")
    ap.add_argument("--stage1_checkpoint",   default="",
                    help="Dir containing linear_attn_delta.pt from stage-1 distillation")
    ap.add_argument("--alpaca_dataset",      default="tatsu-lab/alpaca")
    ap.add_argument("--output_dir",          default="./runs/lora_stage2")
    ap.add_argument("--resume_from",         default="")
    ap.add_argument("--device",              default="cuda:0")
    ap.add_argument("--seq_len",             type=int,   default=1024)
    ap.add_argument("--num_steps",           type=int,   default=10000)
    ap.add_argument("--grad_accum_steps",    type=int,   default=8)
    ap.add_argument("--lr",                  type=float, default=1e-4)
    ap.add_argument("--min_lr",              type=float, default=1e-6)
    ap.add_argument("--warmup_steps",        type=int,   default=200)
    ap.add_argument("--lora_rank",           type=int,   default=16)
    ap.add_argument("--omega_m2t",           type=float, default=0.5)
    ap.add_argument("--omega_t2t",           type=float, default=0.5)
    ap.add_argument("--eval_every",          type=int,   default=100)
    ap.add_argument("--save_every",          type=int,   default=1000)
    ap.add_argument("--test_size",           type=int,   default=256)
    ap.add_argument("--test_eval_batches",   type=int,   default=16)
    cfg = ap.parse_args()
    train(cfg)
