"""Stage-2 LoRA fine-tuning for linearized LLaDA2 (hybrid block from stage 1).

BD3LM block-wise masked diffusion over Tulu-v2 SFT mixture (multi-turn).
- Only assistant tokens are corrupted / used in the loss.
- Prompt and user tokens are always preserved in the noisy input.
- Block-causal attention (block-wise AR) via build_block_causal_mask.
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


# ── Tulu v2 data loader ───────────────────────────────────────────────────────

class TuluLoader:
    """
    Loads allenai/tulu-v2-sft-mixture (multi-turn chat), tokenizes with the
    model's chat template, and builds a resp_mask that is 1 ONLY on assistant
    tokens across all turns. User / system / padding tokens are always 0.

    First `test_size` examples are reserved as a fixed test set; the remaining
    examples are shuffled and cycled for training.
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int,
        block_size: int,
        test_size: int = 256,
        dataset_name: str = "allenai/tulu-v2-sft-mixture",
    ):
        ds = load_dataset(dataset_name, split="train")
        self._raw          = list(ds)
        self._tok          = tokenizer
        self._seq_len      = seq_len
        self._block_size   = block_size
        self._n            = len(self._raw)
        self._test_size    = min(test_size, self._n // 8)
        self._has_template = getattr(tokenizer, "chat_template", None) is not None

        train_idx = list(range(self._test_size, self._n))
        random.shuffle(train_idx)
        self._train_order = train_idx
        self._pos = 0

        print(f"[Tulu] {self._n} examples  "
              f"test={self._test_size}  train={len(self._train_order)}")

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _encode_example(self, idx):
        messages = self._raw[idx]["messages"]
        messages = [m for m in messages if m.get("content", "").strip()]
        if not messages:
            return self._encode_example((idx + 1) % self._n)

        if self._has_template:
            full_text = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        else:
            full_text = "".join(
                f"### {m['role'].capitalize()}:\n{m['content']}\n\n"
                for m in messages
            )

        # Prefer offset_mapping (fast tokenizers) for precise span detection.
        try:
            enc     = self._tok(
                full_text, truncation=False, return_tensors="pt",
                return_offsets_mapping=True,
            )
            offsets = enc.pop("offset_mapping")[0]   # (L, 2)
            ids     = enc.input_ids
            resp_mask, user_turn_ids = self._build_masks_offsets(messages, offsets)
        except Exception:
            enc  = self._tok(
                full_text, truncation=False, return_tensors="pt",
            )
            ids  = enc.input_ids
            resp_mask, user_turn_ids = self._build_masks_prefix(messages, ids.shape[1])

        ids_1d = ids[0].tolist()
        resp_1d = resp_mask[0].tolist()
        
        new_ids = []
        new_resp = []
        new_amsk = []
        
        L = len(ids_1d)
        for i in range(L):
            if i > 0 and resp_1d[i-1] == 1 and resp_1d[i] == 0:
                # Transition from assistant to non-assistant (e.g. next user prompt)
                # Pad to the next block boundary to avoid prompt leakage
                rem = len(new_ids) % self._block_size
                if rem != 0:
                    pad_len = self._block_size - rem
                    new_ids.extend([self._tok.pad_token_id] * pad_len)
                    new_resp.extend([0] * pad_len)
                    new_amsk.extend([0] * pad_len)
                    
            if len(new_ids) >= self._seq_len:
                break
                
            new_ids.append(ids_1d[i])
            new_resp.append(resp_1d[i])
            new_amsk.append(1) # 1 for real tokens
            
        # If still shorter than max_length, pad the end
        if len(new_ids) < self._seq_len:
            pad_len = self._seq_len - len(new_ids)
            new_ids.extend([self._tok.pad_token_id] * pad_len)
            new_resp.extend([0] * pad_len)
            new_amsk.extend([0] * pad_len)
            
        new_ids = new_ids[:self._seq_len]
        new_resp = new_resp[:self._seq_len]
        new_amsk = new_amsk[:self._seq_len]
        
        ids_t = torch.tensor([new_ids], dtype=torch.long)
        amsk_t = torch.tensor([new_amsk], dtype=torch.long)
        resp_t = torch.tensor([new_resp], dtype=torch.long)
        
        return ids_t, amsk_t, resp_t, torch.zeros_like(ids_t)   # user_turn_ids no longer needed

    def _build_masks_offsets(self, messages, offsets):
        """
        Build resp_mask and user_turn_ids using character-level offsets.

        resp_mask:     (1, L)  1 = assistant token
        user_turn_ids: (1, L)  positive int = which user turn; 0 = not a user token

        User turn IDs are 1-indexed per user message in order of appearance.
        """
        L             = offsets.shape[0]
        resp_mask     = torch.zeros(1, L, dtype=torch.long)
        user_turn_ids = torch.zeros(1, L, dtype=torch.long)
        if not self._has_template:
            return resp_mask, user_turn_ids

        user_turn_idx = 0
        for i, msg in enumerate(messages):
            role = msg["role"]
            if role not in ("assistant", "user"):
                continue
            try:
                prefix_text = self._tok.apply_chat_template(
                    messages[:i], tokenize=False, add_generation_prompt=(role == "assistant"),
                )
                until_text  = self._tok.apply_chat_template(
                    messages[:i + 1], tokenize=False, add_generation_prompt=False,
                )
                c0, c1   = len(prefix_text), len(until_text)
                tok_mask = (offsets[:, 1] > c0) & (offsets[:, 0] < c1)
                if role == "assistant":
                    resp_mask[0][tok_mask] = 1
                else:
                    user_turn_idx += 1
                    user_turn_ids[0][tok_mask] = user_turn_idx
            except Exception:
                pass
        return resp_mask, user_turn_ids

    def _build_masks_prefix(self, messages, L):
        """Fallback: estimate spans via prefix token lengths."""
        resp_mask     = torch.zeros(1, L, dtype=torch.long)
        user_turn_ids = torch.zeros(1, L, dtype=torch.long)
        if not self._has_template:
            return resp_mask, user_turn_ids

        user_turn_idx = 0
        for i, msg in enumerate(messages):
            role = msg["role"]
            if role not in ("assistant", "user"):
                continue
            try:
                prefix_text = self._tok.apply_chat_template(
                    messages[:i], tokenize=False, add_generation_prompt=(role == "assistant"),
                )
                until_text  = self._tok.apply_chat_template(
                    messages[:i + 1], tokenize=False, add_generation_prompt=False,
                )
                s = self._tok(prefix_text, return_tensors="pt",
                              add_special_tokens=False).input_ids.shape[1]
                e = self._tok(until_text,  return_tensors="pt",
                              add_special_tokens=False).input_ids.shape[1]
                if role == "assistant":
                    resp_mask[0, s:min(e, L)] = 1
                else:
                    user_turn_idx += 1
                    user_turn_ids[0, s:min(e, L)] = user_turn_idx
            except Exception:
                pass
        return resp_mask, user_turn_ids

    # ── Iteration ─────────────────────────────────────────────────────────────

    def next_train(self):
        if self._pos >= len(self._train_order):
            random.shuffle(self._train_order)
            self._pos = 0
        idx = self._train_order[self._pos]
        self._pos += 1
        return self._encode_example(idx)

    def next_test(self):
        idx = random.randint(0, self._test_size - 1)
        return self._encode_example(idx)


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
def _eval_ppl(model, loader, mask_id, block_size, device, n_batches=16):
    model.eval()
    ces = []
    for _ in range(n_batches):
        ids, pmask, resp_mask, _ = loader.next_test()   # user_turn_ids unused
        ids       = ids.to(device)
        pmask     = pmask.to(device)
        resp_mask = resp_mask.to(device)

        num_blocks  = (ids.shape[1] + block_size - 1) // block_size
        t_per_block = 0.05 + 0.90 * torch.rand(num_blocks, device=device)

        noisy, corrupted = corrupt_all_blocks(
            ids, resp_mask, mask_id, block_size, t_per_block
        )
        noisy[pmask.bool() & ~resp_mask.bool()] = ids[pmask.bool() & ~resp_mask.bool()]

        eligible = corrupted[0] & resp_mask[0].bool()
        if not eligible.any():
            continue

        out = model(noisy, attention_mask=None, key_padding_mask=pmask.bool())
        logits_dev = out.logits.device
        eligible   = eligible.to(logits_dev)
        ce = F.cross_entropy(out.logits[0][eligible], ids[0].to(logits_dev)[eligible])
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
    config.use_linear_attention     = True
    config.use_qk_norm              = True
    config.use_block_softmax_hybrid = cfg.use_block_softmax_hybrid
    block_size = int(getattr(config, "block_size", 32))

    model = LLaDA2MoeModelLM.from_pretrained(
        cfg.teacher_model_path, config=config,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Load stage-1 trained delta (hedgehog_weights + alpha).
    # Accepts either:
    #   - a step dir:   ./runs/hybrid_15k/step_2000   (has linear_attn_delta.pt)
    #   - an output dir: ./runs/hybrid_15k             (final linear_attn_delta.pt at root)
    if cfg.stage1_checkpoint:
        delta_path = os.path.join(cfg.stage1_checkpoint, "linear_attn_delta.pt")
        if not os.path.exists(delta_path):
            # Try to find the latest step_* subdirectory.
            step_dirs = sorted(
                glob.glob(os.path.join(cfg.stage1_checkpoint, "step_*")),
                key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0,
            )
            for sd in reversed(step_dirs):
                candidate = os.path.join(sd, "linear_attn_delta.pt")
                if os.path.exists(candidate):
                    delta_path = candidate
                    break
        if os.path.exists(delta_path):
            ret = model.load_state_dict(
                torch.load(delta_path, map_location="cpu"), strict=False
            )
            print(f"[INIT] Loaded stage-1 delta from {delta_path}. "
              f"Missing: {len(ret.missing_keys)}  Unexpected: {len(ret.unexpected_keys)}")
        else:
            print(f"[WARN] No stage-1 delta found under {cfg.stage1_checkpoint} — "
                  f"starting from teacher weights.")

    # ── LoRA on base projections ──────────────────────────────────────────
    # LoLCATs Stage 2: ONLY LoRA parameters are trained.
    # The linear attention weights (hedgehog_weights, alpha) are FROZEN here —
    # they stay at their Stage-1 converged values and are not updated further.
    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_rank * 2,   # alpha = 2*r (= 16 when r=8)
        target_modules=["query_key_value", "dense"],
        lora_dropout=0.0,               # LoLCATs paper: 0.0 dropout
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    # After get_peft_model all original weights are frozen. make sure
    # hedgehog_weights and alpha are ALSO frozen (not re-enabled here).
    # This matches the canonical LoLCATs Stage 2 behaviour.

    # Shard model across all available GPUs (pipeline parallelism).
    # Must happen after get_peft_model so LoRA layers are included in the dispatch.
    from accelerate import dispatch_model, infer_auto_device_map
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        device_map = infer_auto_device_map(model, dtype=torch.bfloat16)
        model = dispatch_model(model, device_map=device_map)
        print(f"[INIT] Sharded model across {n_gpu} GPUs")
    else:
        model = model.to(device)

    model.train()
    model.print_trainable_parameters()

    raw_model = model  # reference for save/load (no DataParallel wrapper)

    optimizer = torch.optim.AdamW(
        [p for p in raw_model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=0.0,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    loader = TuluLoader(
        tokenizer, 
        cfg.seq_len,
        block_size=block_size,
        test_size=cfg.test_size,
        dataset_name=cfg.dataset,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    # step counts gradient updates (one per outer loop = one optimizer.step()).
    # This keeps eval_every / save_every / warmup_steps / num_steps all in the
    # same units regardless of grad_accum_steps.
    step = 0
    if cfg.resume_from and os.path.exists(cfg.resume_from):
        ts_path = os.path.join(cfg.resume_from, "train_state.pt")
        if os.path.exists(ts_path):
            step = torch.load(ts_path, map_location="cpu").get("step", 0)
        delta_path = os.path.join(cfg.resume_from, "lora_delta.pt")
        if os.path.exists(delta_path):
            raw_model.load_state_dict(
                torch.load(delta_path, map_location="cpu"), strict=False
            )
            print(f"[RESUME] Loaded LoRA delta from {cfg.resume_from}")
        opt_path = os.path.join(cfg.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        print(f"[RESUME] Starting from step {step}")

    # Scheduler counts gradient updates; convert sequence-based step and hyperparams.
    grad_steps_total   = cfg.num_steps    // cfg.grad_accum_steps
    grad_steps_warmup  = cfg.warmup_steps // cfg.grad_accum_steps
    grad_steps_done    = step             // cfg.grad_accum_steps
    scheduler = _build_scheduler(
        optimizer, grad_steps_total, grad_steps_warmup, cfg.min_lr, cfg.lr,
        last_epoch=grad_steps_done,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    # step counts total sequences processed (batch_size * grad_accum_steps per
    # outer loop).  eval_every / save_every / num_steps are also in sequences.
    # Floor-division trigger (same as distill) fires even if inc > every.
    #
    # NOTE: attention_mask is NOT passed to the model — BlockSoftmaxLinearHybrid
    # ignores it entirely (softmax is intra-block by architecture; linear is a
    # causal recurrence).  key_padding_mask IS used by linear attention.
    def _triggered(every, cur, inc):
        return every > 0 and (cur // every) > ((cur - inc) // every)

    B   = cfg.batch_size
    inc = B * cfg.grad_accum_steps   # sequences per outer loop

    os.makedirs(cfg.output_dir, exist_ok=True)
    pbar    = tqdm(total=cfg.num_steps, initial=step)
    m2t_ema = t2t_ema = None

    while step < cfg.num_steps:
        optimizer.zero_grad(set_to_none=True)
        acc_m2t = acc_t2t = 0.0
        acc_count = 0

        for _ in range(cfg.grad_accum_steps):
            # ── Batch assembly (B examples stacked) ───────────────────────────
            batch = [loader.next_train() for _ in range(B)]
            ids       = torch.cat([x[0] for x in batch], dim=0).to(device)   # (B, L)
            pmask     = torch.cat([x[1] for x in batch], dim=0).to(device)   # (B, L)
            resp_mask = torch.cat([x[2] for x in batch], dim=0).to(device)   # (B, L)
            user_turn_ids = torch.cat([x[3] for x in batch], dim=0).to(device) # (B, L)
            kpm         = pmask.bool()                          # (B, L)  1=real token
            prompt_real = pmask.bool() & ~resp_mask.bool()      # (B, L)

            # ── 2L Shared Inputs ──────────────────────────────────────────────
            num_blocks = (cfg.seq_len + block_size - 1) // block_size
            pos_2L = torch.cat([torch.arange(cfg.seq_len, device=device), torch.arange(cfg.seq_len, device=device)], dim=0).unsqueeze(0).expand(B, -1)
            pad_2L = torch.cat([kpm, kpm], dim=1)

            # ── M2T (BD3LM) ───────────────────────────────────────────────────
            t_per_block_m2t = 0.05 + 0.90 * torch.rand(num_blocks, device=device)
            noisy_m2t, corrupted_m2t = corrupt_all_blocks(
                ids, resp_mask, mask_id, block_size, t_per_block_m2t
            )
            noisy_m2t[prompt_real] = ids[prompt_real]
            
            in_m2t = torch.cat([noisy_m2t, ids], dim=1)
            out_m2t = model(
                in_m2t,
                attention_mask=None,
                position_ids=pos_2L,
                key_padding_mask=pad_2L,
                half_len=cfg.seq_len,
            )
            logits_dev = out_m2t.logits.device
            logits_m2t = out_m2t.logits[:, :cfg.seq_len]  # Only the noisy half
            
            # Average loss across all examples in the batch.
            losses_m2t = []
            for b in range(B):
                elig = (corrupted_m2t[b] & resp_mask[b].bool()).to(logits_dev)
                if elig.any():
                    losses_m2t.append(
                        F.cross_entropy(logits_m2t[b][elig], ids[b].to(logits_dev)[elig])
                    )
            loss_m2t = (torch.stack(losses_m2t).mean() if losses_m2t
                        else out_m2t.logits.sum() * 0.0)

            # Backward M2T immediately to free activations before T2T forward.
            loss_m2t_scaled = cfg.omega_m2t * loss_m2t / cfg.grad_accum_steps
            if torch.isfinite(loss_m2t_scaled):
                loss_m2t_scaled.backward()
                acc_m2t += float(loss_m2t)
            del out_m2t, logits_m2t, in_m2t, noisy_m2t, corrupted_m2t
            del loss_m2t, loss_m2t_scaled

            # ── T2T ───────────────────────────────────────────────────────────
            t_per_block_t2t = 0.05 + 0.90 * torch.rand(num_blocks, device=device)
            noisy_t2t, corrupted_t2t = corrupt_all_blocks_t2t(
                ids, resp_mask, vocab_ids, mask_id, block_size, t_per_block_t2t
            )
            noisy_t2t[prompt_real] = ids[prompt_real]

            in_t2t = torch.cat([noisy_t2t, ids], dim=1)
            out_t2t = model(
                in_t2t,
                attention_mask=None,
                position_ids=pos_2L,
                key_padding_mask=pad_2L,
                half_len=cfg.seq_len,
            )
            t2t_dev  = out_t2t.logits.device
            logits_t2t = out_t2t.logits[:, :cfg.seq_len]  # Only the noisy half
            
            losses_t2t = []
            for b in range(B):
                elig = (corrupted_t2t[b] & resp_mask[b].bool()).to(t2t_dev)
                if elig.any():
                    losses_t2t.append(
                        F.cross_entropy(logits_t2t[b][elig], ids[b].to(t2t_dev)[elig])
                    )
            loss_t2t = (torch.stack(losses_t2t).mean() if losses_t2t
                        else out_t2t.logits.sum() * 0.0)

            # Backward T2T immediately.
            loss_t2t_scaled = cfg.omega_t2t * loss_t2t / cfg.grad_accum_steps
            if torch.isfinite(loss_t2t_scaled):
                loss_t2t_scaled.backward()
                acc_t2t += float(loss_t2t)
            del out_t2t, logits_t2t, in_t2t, noisy_t2t, corrupted_t2t
            del loss_t2t, loss_t2t_scaled

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

        step += inc   # sequences processed this outer loop
        pbar.update(inc)
        pbar.set_description(
            f"m2t={m2t_ema:.3f}  t2t={t2t_ema:.3f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if _triggered(cfg.eval_every, step, inc):
            ppl = _eval_ppl(
                model, loader, mask_id, block_size, device,
                n_batches=cfg.test_eval_batches,
            )
            tqdm.write(f"--- Eval @ step {step} | ppl={ppl:.1f} ---")

        if _triggered(cfg.save_every, step, inc):
            _save(raw_model, optimizer, scheduler, step, cfg.output_dir)

    pbar.close()
    _save(raw_model, optimizer, scheduler, step, cfg.output_dir)
    print("Stage-2 LoRA training complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_model_path",      default="inclusionAI/LLaDA2.1-mini")
    ap.add_argument("--stage1_checkpoint",       default="",
                    help="Dir containing linear_attn_delta.pt (or a step_* subdir) from Stage 1")
    ap.add_argument("--dataset",                 default="allenai/tulu-v2-sft-mixture")
    ap.add_argument("--output_dir",              default="./runs/lora_stage2")
    ap.add_argument("--resume_from",             default="")
    ap.add_argument("--device",                  default="cuda:0")
    ap.add_argument("--seq_len",                 type=int,   default=1024)
    ap.add_argument("--num_steps",               type=int,   default=10000)
    ap.add_argument("--batch_size",               type=int,   default=2,
                    help="Sequences per micro-batch. With 2-GPU sharding, batch=2 fits in VRAM.")
    ap.add_argument("--grad_accum_steps",        type=int,   default=8)
    ap.add_argument("--lr",                      type=float, default=1e-4)
    ap.add_argument("--min_lr",                  type=float, default=1e-6)
    ap.add_argument("--warmup_steps",            type=int,   default=200)
    ap.add_argument("--lora_rank",               type=int,   default=8,   # LoLCATs paper: r=8
                    help="LoRA rank. Paper uses r=8, lora_alpha=16 (2*r). dropout=0.0.")
    ap.add_argument("--omega_m2t",               type=float, default=0.5)
    ap.add_argument("--omega_t2t",               type=float, default=0.5)
    ap.add_argument("--eval_every",              type=int,   default=100)
    ap.add_argument("--save_every",              type=int,   default=1000)
    ap.add_argument("--test_size",               type=int,   default=256)
    ap.add_argument("--test_eval_batches",       type=int,   default=16)
    # Architecture flags — must match Stage 1.
    ap.add_argument("--use_block_softmax_hybrid", action="store_true", default=True,
                    help="Use BlockSoftmaxLinearHybrid (must match Stage 1; default: True)")
    cfg = ap.parse_args()
    train(cfg)
