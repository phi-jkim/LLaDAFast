"""Data utilities for LLaDAFast distillation.

Exports:
  build_block_causal_mask  — (1,1,L,L) 0/1 block-causal attention mask
  corrupt_one_block        — "Clean Past, Noisy Present" token corruption
  StreamingTextLoader      — HF streaming dataset wrapper with auto-reset
"""

import random
from typing import Optional, Tuple

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


def build_block_causal_mask(
    seq_len: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Returns (1, 1, seq_len, seq_len) block-causal 0/1 mask.
    Within each block: tokens attend bidirectionally (parallel denoising).
    Across blocks:     strictly causal (each block sees all past blocks).
    """
    num_blocks = (seq_len + block_size - 1) // block_size
    blk = torch.tril(torch.ones(num_blocks, num_blocks, device=device, dtype=torch.float32))
    attend = (
        blk.repeat_interleave(block_size, dim=0)
           .repeat_interleave(block_size, dim=1)
           [:seq_len, :seq_len]
    )
    return attend.to(dtype=dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)


def corrupt_one_block(
    input_ids: torch.Tensor,        # (B, L)
    pad_mask: torch.Tensor,         # (B, L)  1=real token, 0=padding
    mask_id: int,
    block_size: int,
    t: float,
    block_idx: Optional[int] = None,
) -> torch.Tensor:
    """
    Simulates "Clean Past, Noisy Present" training distribution:
      - Past blocks:    kept clean (ground truth tokens).
      - Current block:  tokens randomly replaced with [MASK] at rate t.
      - Future blocks:  also masked out (model cannot see the future).

    Guarantees at least one unmasked token in the current block so that
    the linear attention state Z never collapses to zero.

    Returns: corrupted input_ids (B, L), same device as input.
    """
    B, L = input_ids.shape
    noisy = input_ids.clone()

    for b in range(B):
        real_len = int(pad_mask[b].sum().item())
        # Fill padding region with mask_id so the model treats it as masked.
        if real_len < L:
            noisy[b, real_len:] = mask_id

        # Choose the "present" block.
        bi = block_idx if block_idx is not None else random.randrange(
            max(1, (real_len + block_size - 1) // block_size)
        )
        s = bi * block_size
        e = min((bi + 1) * block_size, real_len)
        if e <= s:
            continue

        # Corrupt present block.
        mask_flags = torch.rand(e - s, device=noisy.device) < t
        if mask_flags.all():
            mask_flags[0] = False   # keep at least one key for linear attention
        noisy[b, s:e][mask_flags] = mask_id

        # Mask future tokens (beyond the present block).
        if e < real_len:
            noisy[b, e:real_len] = mask_id

    return noisy


def corrupt_one_block_t2t(
    input_ids: torch.Tensor,        # (B, L)
    pad_mask: torch.Tensor,         # (B, L)  1=real token, 0=padding
    vocab_ids: torch.Tensor,        # 1-D tensor of valid (non-special) vocab token IDs
    mask_id: int,
    block_size: int,
    t: float,
    block_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Token-to-Token (T2T) "editing" corruption:
      - Past blocks:    kept clean (ground truth tokens).
      - Current block:  tokens randomly replaced with random vocab tokens at rate t.
      - Future blocks:  replaced with mask_id (model cannot see future).

    Returns:
      noisy      — corrupted input_ids (B, L)
      labels_t2t — (B, L) long tensor, original token IDs at corrupted positions,
                   -100 everywhere else (ignored in cross-entropy / KL loss).
    """
    B, L = input_ids.shape
    noisy  = input_ids.clone()
    labels = torch.full((B, L), -100, dtype=torch.long, device=input_ids.device)

    for b in range(B):
        real_len = int(pad_mask[b].sum().item())
        if real_len < L:
            noisy[b, real_len:] = mask_id

        bi = block_idx if block_idx is not None else random.randrange(
            max(1, (real_len + block_size - 1) // block_size)
        )
        s = bi * block_size
        e = min((bi + 1) * block_size, real_len)
        if e <= s:
            continue

        n_pos = e - s
        edit_flags = torch.rand(n_pos, device=noisy.device) < t
        if edit_flags.all():
            edit_flags[0] = False   # keep at least one original token

        k = int(edit_flags.sum().item())
        if k > 0:
            rand_tokens = vocab_ids[torch.randint(0, vocab_ids.numel(), (k,), device=noisy.device)]
            labels[b, s:e][edit_flags] = noisy[b, s:e][edit_flags]   # save originals
            noisy[b, s:e][edit_flags]  = rand_tokens

        # Mask future tokens.
        if e < real_len:
            noisy[b, e:real_len] = mask_id

    return noisy, labels



class StreamingTextLoader:
    """
    Wraps a HuggingFace streaming dataset and tokenizes on-the-fly.

    Automatically resets the iterator when it is exhausted, so training
    never silently stops mid-run (fixes the StopIteration bug in the
    original distill.py).
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_subset: Optional[str],
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
    ):
        self._name = dataset_name
        self._subset = dataset_subset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self._reset()

    def _reset(self) -> None:
        data = load_dataset(self._name, name=self._subset, split="train", streaming=True)
        self._iter = iter(data)

    def next_batch(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (input_ids, attention_mask), both shape (batch_size, seq_len).
        Skips empty examples; blocks until batch_size non-empty examples are found.
        """
        all_ids = []
        all_masks = []
        while len(all_ids) < batch_size:
            try:
                ex = next(self._iter)
            except StopIteration:
                self._reset()
                ex = next(self._iter)

            text = ex.get("text") or next(
                (v for v in ex.values() if isinstance(v, str)), ""
            )
            if not text.strip():
                continue

            enc = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.seq_len,
                truncation=True,
                padding="max_length",
            )
            all_ids.append(enc.input_ids)
            all_masks.append(enc.attention_mask)
        
        return torch.cat(all_ids, dim=0), torch.cat(all_masks, dim=0)
