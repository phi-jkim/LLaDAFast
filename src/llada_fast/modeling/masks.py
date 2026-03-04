"""
Attention mask utilities for LLaDAFast.

Standalone module with NO external dependencies beyond torch — importable
without datasets/transformers so tests run in lightweight environments.
"""

import torch


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


def build_bd3lm_mask(
    seq_len: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    BD3LM "staircase" attention mask for single-pass block-diffusion training.

    Input layout (length 2L):
        [ x_t  (noisy)  |  x_0  (clean) ]
          pos 0 .. L-1     pos L .. 2L-1

    Three permitted attention patterns (OR-combined):

      M_BD  — within-block bidirectional, same half.
              noisy_i ↔ noisy_i  (parallel denoising within current block)
              clean_i ↔ clean_i  (self-conditioning within the clean copy)

      M_OBC — noisy query → clean key from a STRICTLY EARLIER block.
              noisy_i → clean_j  where j < i  (clean past conditioning, no cheating)

      M_BC  — clean query → clean key, causal (block_q >= block_kv).
              clean_i → clean_j  where j <= i  (builds KV context for clean half)

    Position IDs for both halves should be 0..L-1 (shared), so the model
    treats the two copies as the same sequence positions.

    Returns (1, 1, 2*seq_len, 2*seq_len) float mask (1=attend, 0=block).
    """
    n = seq_len
    total = 2 * n

    idx = torch.arange(total, device=device)   # 0 .. 2n-1

    is_clean = idx >= n                         # True = clean half  (2L,)
    blk      = (idx % n) // block_size         # block index within half (2L,)

    # Broadcast to (2L, 2L): row=query, col=key
    is_clean_q  = is_clean.unsqueeze(1).expand(total, total)   # (2L,2L) query half
    is_clean_kv = is_clean.unsqueeze(0).expand(total, total)   # (2L,2L) key   half
    blk_q       = blk.unsqueeze(1).expand(total, total)        # (2L,2L) query block
    blk_kv      = blk.unsqueeze(0).expand(total, total)        # (2L,2L) key   block

    M_BD  = (blk_q == blk_kv) & (is_clean_q == is_clean_kv)   # same block, same half
    M_OBC = (~is_clean_q) & (is_clean_kv) & (blk_q > blk_kv)  # noisy→clean, past only
    M_BC  = ( is_clean_q) & (is_clean_kv) & (blk_q >= blk_kv) # clean→clean, causal

    attend = (M_BD | M_OBC | M_BC).to(dtype=dtype)
    return attend.unsqueeze(0).unsqueeze(0)    # (1, 1, 2L, 2L)
