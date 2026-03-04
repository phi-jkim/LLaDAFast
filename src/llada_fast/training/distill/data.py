"""Data utilities for LLaDAFast distillation.

Exports:
  build_block_causal_mask   — (1,1,L,L) 0/1 block-causal attention mask
  build_bd3lm_mask          — (1,1,2L,2L) BD3LM staircase attention mask
  corrupt_one_block         — "Clean Past, Noisy Present" token corruption (single block)
  corrupt_one_block_t2t     — T2T variant of the above
  corrupt_all_blocks        — CARD-style: corrupt ALL blocks in one shot (M2T)
  corrupt_all_blocks_t2t    — CARD-style: corrupt ALL blocks in one shot (T2T)
  TestSetBuffer             — fixed reserved test set buffered at init
  StreamingTextLoader       — HF streaming dataset wrapper with auto-reset
"""

import random
from typing import Optional, Tuple

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


# Re-export mask utilities from the standalone masks module (no heavy deps).
from llada_fast.modeling.masks import build_block_causal_mask, build_bd3lm_mask

__all__ = [
    "build_block_causal_mask",
    "build_bd3lm_mask",
]


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



def corrupt_all_blocks(
    input_ids: torch.Tensor,        # (B, L)
    pad_mask: torch.Tensor,         # (B, L)  1=real token, 0=padding
    mask_id: int,
    block_size: int,
    t_per_block: Optional[torch.Tensor] = None,  # (num_blocks,) noise levels; None → random
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CARD-style single-pass M2T corruption: corrupt ALL blocks simultaneously
    with independent per-block noise levels.

    Each block gets its tokens randomly replaced with [MASK] at rate t_i.
    Padding positions are always set to mask_id.
    Guarantees at least one real unmasked token per block (linear attention stability).

    Returns:
      noisy     — corrupted input_ids (B, L)
      corrupted — (B, L) bool tensor, True at positions that were replaced with mask_id
                  (does NOT include padding; use pad_mask for those)
    """
    B, L = input_ids.shape
    num_blocks = (L + block_size - 1) // block_size
    dev = input_ids.device

    if t_per_block is None:
        t_per_block = 0.05 + 0.90 * torch.rand(num_blocks, device=dev)

    # Expand per-block noise levels to per-position: (L,)
    t_pos = t_per_block.repeat_interleave(block_size)[:L]

    # Sample corruption mask: (B, L)
    corrupted = torch.rand(B, L, device=dev) < t_pos.unsqueeze(0)

    real_mask = pad_mask.bool()  # (B, L) True = real token

    # Safety: ensure at least one real token per block remains uncorrupted.
    for bi in range(num_blocks):
        s = bi * block_size
        e = min(s + block_size, L)
        blk_c = corrupted[:, s:e]    # (B, n)
        blk_r = real_mask[:, s:e]    # (B, n)
        # Rows where every real position in this block is about to be masked.
        all_gone = (blk_c & blk_r).all(dim=1) & blk_r.any(dim=1)  # (B,)
        if all_gone.any():
            first_real = blk_r.float().argmax(dim=1)  # (B,) index of first real pos
            bad = all_gone.nonzero(as_tuple=True)[0]
            corrupted[bad, s + first_real[bad]] = False

    noisy = input_ids.clone()
    noisy[corrupted] = mask_id     # mask corrupted real positions
    noisy[~real_mask] = mask_id    # mask padding positions

    return noisy, corrupted


def corrupt_all_blocks_t2t(
    input_ids: torch.Tensor,        # (B, L)
    pad_mask: torch.Tensor,         # (B, L)  1=real token, 0=padding
    vocab_ids: torch.Tensor,        # 1-D tensor of valid (non-special) vocab token IDs
    mask_id: int,
    block_size: int,
    t_per_block: Optional[torch.Tensor] = None,  # (num_blocks,) noise levels; None → random
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CARD-style single-pass T2T corruption: corrupt ALL blocks simultaneously
    with independent per-block noise levels.

    Each block gets its tokens randomly replaced with random vocab tokens at rate t_i.
    Padding positions are always set to mask_id.
    Guarantees at least one real uncorrupted token per block.

    Returns:
      noisy     — corrupted input_ids (B, L)
      corrupted — (B, L) bool tensor, True at positions that were replaced with random tokens
    """
    B, L = input_ids.shape
    num_blocks = (L + block_size - 1) // block_size
    dev = input_ids.device

    if t_per_block is None:
        t_per_block = 0.05 + 0.90 * torch.rand(num_blocks, device=dev)

    t_pos = t_per_block.repeat_interleave(block_size)[:L]
    corrupted = torch.rand(B, L, device=dev) < t_pos.unsqueeze(0)  # (B, L)

    real_mask = pad_mask.bool()

    # Safety: at least one real uncorrupted token per block.
    for bi in range(num_blocks):
        s = bi * block_size
        e = min(s + block_size, L)
        blk_c = corrupted[:, s:e]
        blk_r = real_mask[:, s:e]
        all_gone = (blk_c & blk_r).all(dim=1) & blk_r.any(dim=1)
        if all_gone.any():
            first_real = blk_r.float().argmax(dim=1)
            bad = all_gone.nonzero(as_tuple=True)[0]
            corrupted[bad, s + first_real[bad]] = False

    noisy = input_ids.clone()
    n_corrupted = int(corrupted.sum().item())
    if n_corrupted > 0:
        rand_tokens = vocab_ids[torch.randint(0, vocab_ids.numel(), (n_corrupted,), device=dev)]
        noisy[corrupted] = rand_tokens
    noisy[~real_mask] = mask_id    # mask padding positions

    return noisy, corrupted


class TestSetBuffer:
    """
    Eagerly buffers the first `test_size` non-empty examples from the stream
    as a fixed, reserved test set.  Training must skip the same raw records.

    Attributes:
      raw_consumed  — total raw dataset records consumed (including empties).
                      Pass this to StreamingTextLoader as `skip_first` so the
                      training stream never overlaps with the test set.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_subset: Optional[str],
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
        test_size: int = 256,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self._ids: list = []
        self._masks: list = []
        self.raw_consumed: int = 0

        data = load_dataset(dataset_name, name=dataset_subset, split="train", streaming=True)
        it = iter(data)
        n_buffered = 0
        while n_buffered < test_size:
            try:
                ex = next(it)
                self.raw_consumed += 1
            except StopIteration:
                break
            text = ex.get("text") or next(
                (v for v in ex.values() if isinstance(v, str)), ""
            )
            if not text.strip():
                continue
            enc = tokenizer(
                text,
                return_tensors="pt",
                max_length=seq_len,
                truncation=True,
                padding="max_length",
            )
            self._ids.append(enc.input_ids)
            self._masks.append(enc.attention_mask)
            n_buffered += 1

        self._n = len(self._ids)
        print(
            f"[TestSet] Buffered {self._n} examples "
            f"(consumed {self.raw_consumed} raw records — training will skip these)."
        )

    def next_batch(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a random mini-batch sampled (with replacement) from the buffer."""
        idxs = torch.randint(0, self._n, (batch_size,)).tolist()
        return (
            torch.cat([self._ids[i] for i in idxs], dim=0),
            torch.cat([self._masks[i] for i in idxs], dim=0),
        )

    def __len__(self) -> int:
        return self._n


class StreamingTextLoader:
    """
    Wraps a HuggingFace streaming dataset and tokenizes on-the-fly.

    Pass `skip_first=test_buffer.raw_consumed` so the training stream
    never overlaps with the reserved test set.

    Automatically resets the iterator when it is exhausted.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_subset: Optional[str],
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
        skip_first: int = 0,
    ):
        self._name = dataset_name
        self._subset = dataset_subset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self._skip_first = skip_first
        self._reset()

    def _reset(self) -> None:
        data = load_dataset(self._name, name=self._subset, split="train", streaming=True)
        if self._skip_first > 0:
            data = data.skip(self._skip_first)
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
