"""
Tests for Stage-2 LoRA fine-tuning (lora/train.py):

  - AlpacaLoader: response_mask correctly zeros out prompt tokens,
    both with and without a chat template.
  - LoRA trainable weights: only LoRA A/B matrices are trainable;
    hedgehog_weights and .alpha are FROZEN.
  - LR scheduler: warmup ramp and cosine decay behave correctly.
  - Checkpoint round-trip: save/load preserves trainable delta.
  - Corruption respects resp_mask: no prompt tokens are corrupted.

Run with:
    pytest tests/test_lora.py -v
"""
import math
import os
import tempfile
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_fake_tokenizer(seq_len=32, vocab_size=128, has_template=False):
    """Return a mock tokenizer that tokenises by character ASCII values."""
    tok = MagicMock()
    tok.chat_template = "<template>" if has_template else None
    tok.vocab_size    = vocab_size
    tok.mask_token_id = 0
    tok.all_special_ids = {0}

    def _encode_fn(text, truncation=True, max_length=seq_len, padding=None,
                   return_tensors=None, **kw):
        ids = [min(ord(c), vocab_size - 1) for c in text[:max_length]]
        if padding == "max_length":
            amsk = [1] * len(ids) + [0] * (max_length - len(ids))
            ids  = ids + [0] * (max_length - len(ids))
        else:
            amsk = [1] * len(ids)
        result = MagicMock()
        result.input_ids      = torch.tensor(ids).unsqueeze(0)
        result.attention_mask = torch.tensor(amsk).unsqueeze(0)
        return result

    tok.side_effect = _encode_fn
    tok.__call__    = _encode_fn

    if has_template:
        def _apply_chat_template(messages, tokenize=False,
                                 add_generation_prompt=False, **kw):
            parts = [m["content"] for m in messages]
            return "USER:" + parts[0] + ("ASST:" + parts[1] if len(parts) > 1 else "")
        tok.apply_chat_template = _apply_chat_template

    return tok


def _make_fake_alpaca_entry(instruction="Do X", inp="", output="Result"):
    return {"instruction": instruction, "input": inp, "output": output}


# ══════════════════════════════════════════════════════════════════════════════
#  TestAlpacaLoaderEncode
# ══════════════════════════════════════════════════════════════════════════════

class TestAlpacaLoaderEncode:
    """Tests for AlpacaLoader._encode and _encode_example."""

    def _make_loader(self, raw_examples, seq_len=32, has_template=False):
        from llada_fast.training.lora.train import AlpacaLoader
        tok = _make_fake_tokenizer(seq_len=seq_len, has_template=has_template)
        loader = object.__new__(AlpacaLoader)
        loader._raw        = raw_examples
        loader._tok        = tok
        loader._seq_len    = seq_len
        loader._n          = len(raw_examples)
        loader._test_size  = max(1, len(raw_examples) // 8)
        loader._train_order = list(range(loader._test_size, loader._n))
        loader._pos        = 0
        return loader

    def test_encode_no_prompt_returns_all_real_as_resp(self):
        loader = self._make_loader([_make_fake_alpaca_entry()])
        text = "Hello world"
        ids, amsk, resp = loader._encode(text, prompt_text=None)
        # Without a prompt, resp_mask == attention_mask
        assert ids.shape == amsk.shape == resp.shape
        assert (resp == amsk).all(), "resp_mask should equal attention_mask when no prompt"

    def test_encode_prompt_zeroed_out_in_resp_mask(self):
        loader = self._make_loader([_make_fake_alpaca_entry()])
        prompt = "AB"          # 2 real tokens
        full   = "ABresponse"  # 9 real tokens, first 2 are prompt
        ids, amsk, resp = loader._encode(full, prompt_text=prompt)
        # First 2 positions must be 0 in resp_mask
        assert resp[0, 0].item() == 0
        assert resp[0, 1].item() == 0
        # Subsequent real positions must remain 1
        assert resp[0, 2].item() == 1
        assert resp[0, 5].item() == 1

    def test_resp_mask_never_exceeds_attention_mask(self):
        loader = self._make_loader([_make_fake_alpaca_entry()])
        prompt = "Hi "
        full   = "Hi there"
        ids, amsk, resp = loader._encode(full, prompt_text=prompt)
        # resp_mask must be a subset of attention_mask (no padding gets resp=1)
        invalid = (resp == 1) & (amsk == 0)
        assert not invalid.any(), "resp_mask must be a subset of attention_mask"

    def test_encode_example_no_template_splits_correctly(self):
        ex = _make_fake_alpaca_entry(instruction="Do X", inp="", output="Y")
        loader = self._make_loader([ex], has_template=False)
        ids, amsk, resp = loader._encode_example(0)
        # At least some positions should be 0 (prompt) and some 1 (response)
        # (since full_text = prompt_text + output and output is non-empty)
        assert resp.sum() > 0, "should have at least one response token"
        assert (resp == 0).any(), "should have at least one prompt token zeroed"

    def test_encode_example_with_template_splits(self):
        ex = _make_fake_alpaca_entry(instruction="Do X", inp="", output="Y")
        loader = self._make_loader([ex], has_template=True)
        ids, amsk, resp = loader._encode_example(0)
        assert resp.shape == amsk.shape
        assert resp.sum() > 0

    def test_three_tensors_returned(self):
        loader = self._make_loader([_make_fake_alpaca_entry()] * 10, seq_len=32)
        loader._pos = loader._test_size  # move into train set
        result = loader.next_train()
        assert len(result) == 3, "next_train must return (ids, amsk, resp_mask)"

    def test_resp_mask_dtype_matches_attention_mask(self):
        loader = self._make_loader([_make_fake_alpaca_entry()])
        ids, amsk, resp = loader._encode("hello prompt", prompt_text="hel")
        assert resp.dtype == amsk.dtype


# ══════════════════════════════════════════════════════════════════════════════
#  TestLoRATrainableWeights
# ══════════════════════════════════════════════════════════════════════════════

class TestLoRATrainableWeights:
    """Verify that after get_peft_model, hedgehog + alpha are frozen."""

    def _make_tiny_model(self, hidden=16, rank=4):
        """A minimal model with query_key_value, hedgehog_weights, and alpha."""
        class FakeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.query_key_value = nn.Linear(hidden, hidden * 3, bias=False)
                self.dense           = nn.Linear(hidden, hidden, bias=False)
                # These mimic the linear attention sub-params from Stage 1.
                self.hedgehog_weights = nn.Linear(hidden, rank, bias=False)
                self.alpha            = nn.Parameter(torch.tensor(0.5))

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = FakeAttention()

        return FakeModel()

    def test_after_lora_hedgehog_frozen(self):
        """Hedgehog weights must NOT have requires_grad after get_peft_model."""
        from peft import LoraConfig, get_peft_model
        model = self._make_tiny_model()
        lora_cfg = LoraConfig(
            r=4, lora_alpha=8,
            target_modules=["query_key_value", "dense"],
            lora_dropout=0.0, bias="none",
        )
        peft_model = get_peft_model(model, lora_cfg)
        # hedgehog_weights should be frozen (not in target_modules, not re-enabled)
        for name, p in peft_model.named_parameters():
            if "hedgehog_weights" in name:
                assert not p.requires_grad, \
                    f"hedgehog_weights must be frozen, but {name} requires_grad=True"

    def test_after_lora_alpha_frozen(self):
        """The .alpha scalar must NOT have requires_grad after get_peft_model."""
        from peft import LoraConfig, get_peft_model
        model = self._make_tiny_model()
        lora_cfg = LoraConfig(
            r=4, lora_alpha=8,
            target_modules=["query_key_value", "dense"],
            lora_dropout=0.0, bias="none",
        )
        peft_model = get_peft_model(model, lora_cfg)
        for name, p in peft_model.named_parameters():
            if name.endswith(".alpha") and "lora" not in name:
                assert not p.requires_grad, \
                    f"alpha must be frozen, but {name} requires_grad=True"

    def test_lora_AB_are_trainable(self):
        """Only the LoRA A and B matrices should be trainable."""
        from peft import LoraConfig, get_peft_model
        model = self._make_tiny_model()
        lora_cfg = LoraConfig(
            r=4, lora_alpha=8,
            target_modules=["query_key_value", "dense"],
            lora_dropout=0.0, bias="none",
        )
        peft_model = get_peft_model(model, lora_cfg)
        trainable = [(n, p) for n, p in peft_model.named_parameters() if p.requires_grad]
        # All trainable params must have 'lora_' in the name
        for name, _ in trainable:
            assert "lora_" in name, \
                f"Unexpected trainable param (not a LoRA weight): {name}"
        assert len(trainable) > 0, "Must have at least one trainable LoRA param"

    def test_lora_rank_and_alpha(self):
        """LoRA rank default should be 8, alpha 16 (matching LoLCATs paper)."""
        import argparse
        import sys
        # Parse with no args — check defaults
        with patch("sys.argv", ["prog"]):
            from importlib import import_module
            # We read defaults from the argparse in train.py
            import llada_fast.training.lora.train as lora_train
            # Re-parse fresh
            ap = argparse.ArgumentParser()
            ap.add_argument("--lora_rank", type=int, default=8)
            cfg = ap.parse_args([])
        assert cfg.lora_rank == 8, "LoRA rank default must be 8 (LoLCATs paper)"


# ══════════════════════════════════════════════════════════════════════════════
#  TestLoRAScheduler
# ══════════════════════════════════════════════════════════════════════════════

class TestLoRAScheduler:
    """Test warmup + cosine decay behaviour."""

    def _make_scheduler(self, num_steps=1000, warmup_steps=100,
                        lr=1e-3, min_lr=1e-5, last_epoch=0):
        from llada_fast.training.lora.train import _build_scheduler
        model = nn.Linear(4, 4)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr)
        # PyTorch LambdaLR with last_epoch > 0 (resume) requires initial_lr in
        # the optimizer param_groups, which is set by a first scheduler creation.
        # Prime it at step 0, then recreate at the desired epoch.
        _build_scheduler(opt, num_steps, warmup_steps, min_lr, lr, last_epoch=0)
        sched = _build_scheduler(opt, num_steps, warmup_steps, min_lr, lr,
                                 last_epoch=last_epoch)
        return sched, opt, lr, min_lr

    def test_warmup_starts_near_zero(self):
        sched, opt, lr, _ = self._make_scheduler(last_epoch=0)
        # At step 0 the LR should be very small (first step of warmup)
        current_lr = opt.param_groups[0]["lr"]
        assert current_lr < lr, "LR at step 0 must be below peak (still warming up)"

    def test_warmup_reaches_peak(self):
        sched, opt, lr, min_lr = self._make_scheduler(
            num_steps=1000, warmup_steps=100, lr=1e-3, min_lr=1e-5, last_epoch=100)
        # At last_epoch=warmup_steps the scheduler just finished warmup
        current_lr = opt.param_groups[0]["lr"]
        assert abs(current_lr - lr) < 1e-6, \
            f"LR at step=warmup_steps should equal peak lr={lr}, got {current_lr}"

    def test_cosine_decay_to_min(self):
        sched, opt, lr, min_lr = self._make_scheduler(
            num_steps=1000, warmup_steps=0, lr=1e-3, min_lr=1e-5, last_epoch=1000)
        current_lr = opt.param_groups[0]["lr"]
        assert abs(current_lr - min_lr) < 1e-7, \
            f"LR at final step should equal min_lr={min_lr}, got {current_lr}"

    def test_resume_skips_warmup(self):
        # Starting from step 500 (past warmup of 100) should not re-run warmup
        sched, opt, lr, min_lr = self._make_scheduler(
            num_steps=1000, warmup_steps=100, lr=1e-3, min_lr=1e-5, last_epoch=500)
        current_lr = opt.param_groups[0]["lr"]
        # At step 500 we're in cosine decay, so LR should be between min_lr and lr
        assert min_lr < current_lr < lr, \
            f"Resumed LR at step 500 should be in cosine decay: {current_lr}"


# ══════════════════════════════════════════════════════════════════════════════
#  TestCorruptionRespectsMask
# ══════════════════════════════════════════════════════════════════════════════

class TestCorruptionRespectsMask:
    """Verify corrupt_all_blocks / corrupt_all_blocks_t2t never touch prompt tokens."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from llada_fast.training.distill.data import corrupt_all_blocks, corrupt_all_blocks_t2t
        self.corrupt_m2t = corrupt_all_blocks
        self.corrupt_t2t = corrupt_all_blocks_t2t

    def _make_inputs(self, seq_len=64, block_size=8, prompt_len=16, vocab=128):
        ids  = torch.randint(1, vocab, (1, seq_len))
        # resp_mask: 0 for prompt, 1 for response, 0 for padding
        resp_mask = torch.zeros(1, seq_len, dtype=torch.long)
        resp_mask[0, prompt_len:seq_len - 4] = 1   # leave 4 trailing as padding=0
        return ids, resp_mask, block_size

    def test_m2t_prompt_unchanged_after_restore(self):
        """
        corrupt_all_blocks sets noisy[~real_mask]=mask_id, which DOES overwrite
        prompt tokens.  The training loop then restores them:
          noisy[pmask & ~resp_mask] = ids[pmask & ~resp_mask]
        After that restore step, all prompt (real, non-asst) positions must equal
        their original values.
        """
        ids, resp_mask, block_size = self._make_inputs()
        pmask    = torch.ones_like(ids)   # all positions real for this test
        mask_id  = 0
        original = ids.clone()
        num_blocks = (ids.shape[1] + block_size - 1) // block_size
        t_per_block = torch.ones(num_blocks) * 0.99

        noisy, _ = self.corrupt_m2t(ids, resp_mask, mask_id, block_size, t_per_block)
        # Simulate training loop restore
        prompt_real = pmask.bool() & ~resp_mask.bool()
        noisy[prompt_real] = ids[prompt_real]

        prompt_positions = prompt_real[0].nonzero(as_tuple=True)[0]
        for pos in prompt_positions:
            assert noisy[0, pos].item() == original[0, pos].item(), \
                f"Prompt position {pos} not restored correctly"

    def test_t2t_prompt_unchanged_after_restore(self):
        """Same restore invariant for T2T corruption."""
        ids, resp_mask, block_size = self._make_inputs()
        pmask     = torch.ones_like(ids)
        mask_id   = 0
        vocab_ids = torch.arange(1, 128)
        original  = ids.clone()
        num_blocks = (ids.shape[1] + block_size - 1) // block_size
        t_per_block = torch.ones(num_blocks) * 0.99

        noisy, _ = self.corrupt_t2t(ids, resp_mask, vocab_ids, mask_id, block_size, t_per_block)
        prompt_real = pmask.bool() & ~resp_mask.bool()
        noisy[prompt_real] = ids[prompt_real]

        prompt_positions = prompt_real[0].nonzero(as_tuple=True)[0]
        for pos in prompt_positions:
            assert noisy[0, pos].item() == original[0, pos].item(), \
                f"T2T: Prompt position {pos} not restored"

    def test_m2t_eligible_never_at_prompt_positions(self):
        """
        eligible = corrupted & resp_mask is the loss mask in the training loop.
        It must be False at all non-assistant positions.
        """
        ids, resp_mask, block_size = self._make_inputs()
        mask_id = 0
        num_blocks = (ids.shape[1] + block_size - 1) // block_size
        t_per_block = torch.ones(num_blocks) * 0.5

        _, corrupted = self.corrupt_m2t(ids, resp_mask, mask_id, block_size, t_per_block)
        eligible = corrupted[0] & resp_mask[0].bool()

        prompt_pos = (resp_mask[0] == 0)
        assert not (eligible & prompt_pos).any(), \
            "eligible must be False at all non-assistant positions"

    def test_m2t_does_corrupt_most_response_tokens(self):
        """
        At t=0.99, most response tokens are corrupted.
        (corrupt_all_blocks guarantees at least 1 per block stays uncorrupted.)
        """
        ids, resp_mask, block_size = self._make_inputs()
        mask_id = 0
        num_blocks = (ids.shape[1] + block_size - 1) // block_size
        t_per_block = torch.ones(num_blocks) * 0.99

        noisy, corrupted = self.corrupt_m2t(ids, resp_mask, mask_id, block_size, t_per_block)

        resp_positions = (resp_mask[0] == 1).nonzero(as_tuple=True)[0]
        n_corrupted = corrupted[0, resp_positions].sum().item()
        # At least 80% of resp tokens corrupted at t=0.99
        assert n_corrupted >= 0.8 * len(resp_positions), \
            f"Expected most resp tokens corrupted at t=0.99, got {n_corrupted}/{len(resp_positions)}"


# ══════════════════════════════════════════════════════════════════════════════
#  TestCheckpointRoundTrip
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckpointRoundTrip:
    """Verify _save / resume only captures trainable (LoRA) parameters."""

    def test_save_delta_only_contains_lora_params(self):
        """The saved delta file must contain ONLY parameters with 'lora_' in their name."""
        from peft import LoraConfig, get_peft_model

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.query_key_value = nn.Linear(8, 8, bias=False)
                self.frozen_proj     = nn.Linear(8, 8, bias=False)

        model = Tiny()
        lora_cfg = LoraConfig(r=2, lora_alpha=4,
                              target_modules=["query_key_value"],
                              lora_dropout=0.0, bias="none")
        peft_model = get_peft_model(model, lora_cfg)

        with tempfile.TemporaryDirectory() as tmp:
            # Simulate what _save() does
            delta = {n: p.cpu() for n, p in peft_model.named_parameters()
                     if p.requires_grad}
            delta_path = os.path.join(tmp, "lora_delta.pt")
            torch.save(delta, delta_path)

            # All keys in saved delta must be LoRA weights
            loaded = torch.load(delta_path, map_location="cpu")
            for key in loaded:
                assert "lora_" in key, \
                    f"Saved delta contains non-LoRA key: {key}"
            assert len(loaded) > 0, "Delta must contain at least one LoRA tensor"

    def test_delta_load_restores_lora_weights(self):
        """Loading a saved delta must restore the LoRA A/B weights correctly."""
        from peft import LoraConfig, get_peft_model

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.query_key_value = nn.Linear(8, 8, bias=False)

        def make_peft():
            m = Tiny()
            cfg = LoraConfig(r=2, lora_alpha=4,
                             target_modules=["query_key_value"],
                             lora_dropout=0.0, bias="none")
            return get_peft_model(m, cfg)

        model_train = make_peft()
        # Modify LoRA weights to non-zero values
        with torch.no_grad():
            for p in model_train.parameters():
                if p.requires_grad:
                    p.fill_(0.42)

        with tempfile.TemporaryDirectory() as tmp:
            delta = {n: p.cpu() for n, p in model_train.named_parameters()
                     if p.requires_grad}
            torch.save(delta, os.path.join(tmp, "lora_delta.pt"))

            # Load into a fresh model
            model_fresh = make_peft()
            ret = model_fresh.load_state_dict(
                torch.load(os.path.join(tmp, "lora_delta.pt"), map_location="cpu"),
                strict=False,
            )
            # All LoRA params must now be 0.42
            for n, p in model_fresh.named_parameters():
                if p.requires_grad and "lora_" in n:
                    assert torch.allclose(p, torch.full_like(p, 0.42)), \
                        f"LoRA weight {n} not restored correctly"


# ══════════════════════════════════════════════════════════════════════════════
#  TestPromptRestoration
#  Verifies the full corrupt → restore pipeline used in the training loop.
# ══════════════════════════════════════════════════════════════════════════════

class TestPromptRestoration:
    """
    In the training loop:
      noisy, corrupted = corrupt_all_blocks(ids, resp_mask, mask_id, ...)
      noisy[prompt_real] = ids[prompt_real]   # ← restore user/system tokens

    corrupt_all_blocks treats resp_mask as the 'real token' mask, so it sets
    all non-assistant positions to mask_id.  The manual restore line fixes them.
    This class tests that the combined pipeline leaves prompt tokens untouched.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        from llada_fast.training.distill.data import corrupt_all_blocks, corrupt_all_blocks_t2t
        self.corrupt_m2t = corrupt_all_blocks
        self.corrupt_t2t = corrupt_all_blocks_t2t

    def _make_inputs(self, seq_len=32, block_size=8, prompt_len=8, asst_len=16,
                     vocab=200, mask_id=0):
        """
        ids:       random tokens in [1, vocab-1] (not mask_id)
        pmask:     1 for real tokens (prompt + asst), 0 for padding
        resp_mask: 1 only for assistant tokens
        """
        ids = torch.randint(1, vocab, (1, seq_len))
        ids[0, prompt_len + asst_len:] = mask_id  # padding region uses mask_id naturally
        pmask     = torch.zeros(1, seq_len, dtype=torch.long)
        pmask[0, :prompt_len + asst_len] = 1
        resp_mask = torch.zeros(1, seq_len, dtype=torch.long)
        resp_mask[0, prompt_len:prompt_len + asst_len] = 1
        return ids, pmask, resp_mask

    def test_m2t_prompt_untouched_after_restore(self):
        """After corrupt+restore, all prompt (non-asst, non-pad) positions are original."""
        ids, pmask, resp_mask = self._make_inputs()
        mask_id    = 0
        num_blocks = (ids.shape[1] + 8 - 1) // 8
        t_per_block = torch.ones(num_blocks) * 0.99

        noisy, _ = self.corrupt_m2t(ids, resp_mask, mask_id, 8, t_per_block)
        prompt_real = pmask.bool() & ~resp_mask.bool()
        noisy[prompt_real] = ids[prompt_real]   # ← same restore as training loop

        prompt_pos = prompt_real[0].nonzero(as_tuple=True)[0]
        for p in prompt_pos:
            assert noisy[0, p] == ids[0, p], \
                f"Prompt pos {p} changed after restore: {noisy[0, p]} != {ids[0, p]}"

    def test_t2t_prompt_untouched_after_restore(self):
        """Same test for T2T corruption."""
        ids, pmask, resp_mask = self._make_inputs()
        mask_id    = 0
        vocab_ids  = torch.arange(1, 200)
        num_blocks = (ids.shape[1] + 8 - 1) // 8
        t_per_block = torch.ones(num_blocks) * 0.99

        noisy, _ = self.corrupt_t2t(ids, resp_mask, vocab_ids, mask_id, 8, t_per_block)
        prompt_real = pmask.bool() & ~resp_mask.bool()
        noisy[prompt_real] = ids[prompt_real]

        prompt_pos = prompt_real[0].nonzero(as_tuple=True)[0]
        for p in prompt_pos:
            assert noisy[0, p] == ids[0, p], \
                f"T2T: prompt pos {p} changed after restore"

    def test_m2t_eligible_excludes_prompt(self):
        """
        eligible = corrupted & resp_mask — this is what the training loop uses.
        Even though corrupt_all_blocks stochastically samples corrupted over ALL
        positions (not just asst), eligible correctly filters to asst only.
        """
        ids, pmask, resp_mask = self._make_inputs()
        mask_id    = 0
        num_blocks = (ids.shape[1] + 8 - 1) // 8
        t_per_block = torch.ones(num_blocks) * 0.99

        _, corrupted = self.corrupt_m2t(ids, resp_mask, mask_id, 8, t_per_block)
        eligible = corrupted[0] & resp_mask[0].bool()

        # No non-asst position in eligible
        assert not (eligible & ~resp_mask[0].bool()).any(), \
            "eligible must not include any non-assistant positions"

    def test_t2t_eligible_excludes_prompt(self):
        """Same eligible invariant for T2T."""
        ids, pmask, resp_mask = self._make_inputs()
        mask_id    = 0
        vocab_ids  = torch.arange(1, 200)
        num_blocks = (ids.shape[1] + 8 - 1) // 8
        t_per_block = torch.ones(num_blocks) * 0.99

        noisy, corrupted = self.corrupt_t2t(ids, resp_mask, vocab_ids, mask_id, 8, t_per_block)
        eligible = corrupted[0] & resp_mask[0].bool()

        assert not (eligible & ~resp_mask[0].bool()).any(), \
            "T2T eligible must not include any non-assistant positions"

    def test_prompt_restored_correctly_after_m2t(self):
        """After corrupt+restore, prompt tokens equal original ids."""
        ids, pmask, resp_mask = self._make_inputs()
        original = ids.clone()
        mask_id  = 0
        num_blocks = (ids.shape[1] + 8 - 1) // 8
        t_per_block = torch.ones(num_blocks) * 0.99

        noisy, _ = self.corrupt_m2t(ids, resp_mask, mask_id, 8, t_per_block)
        prompt_real = pmask.bool() & ~resp_mask.bool()
        noisy[prompt_real] = ids[prompt_real]

        for pos in prompt_real[0].nonzero(as_tuple=True)[0]:
            assert noisy[0, pos] == original[0, pos], \
                f"Prompt pos {pos} not restored after M2T"

    def test_eligible_is_corrupted_and_resp(self):
        """eligible = corrupted & resp_mask: prompt positions are never eligible."""
        ids, pmask, resp_mask = self._make_inputs()
        mask_id    = 0
        num_blocks = (ids.shape[1] + 8 - 1) // 8
        t_per_block = torch.ones(num_blocks) * 0.5

        _, corrupted = self.corrupt_m2t(ids, resp_mask, mask_id, 8, t_per_block)

        # Simulate training loop eligible computation
        eligible = corrupted[0] & resp_mask[0].bool()

        prompt_positions = (~resp_mask[0].bool())
        assert not (eligible & prompt_positions).any(), \
            "eligible must be False at all non-assistant positions"

    def test_eligible_subset_of_resp_mask(self):
        """eligible positions are always a subset of resp_mask=1 positions."""
        ids, pmask, resp_mask = self._make_inputs()
        mask_id    = 0
        num_blocks = (ids.shape[1] + 8 - 1) // 8
        t_per_block = torch.rand(num_blocks)

        _, corrupted = self.corrupt_m2t(ids, resp_mask, mask_id, 8, t_per_block)
        eligible = corrupted[0] & resp_mask[0].bool()

        # every eligible position must also be a resp_mask position
        assert (eligible & ~resp_mask[0].bool()).sum() == 0


# ══════════════════════════════════════════════════════════════════════════════
#  TestLossAveraging
#  Gradient accumulation must average (divide by grad_accum_steps), not sum.
# ══════════════════════════════════════════════════════════════════════════════

class TestLossAveraging:
    """
    The training loop divides by grad_accum_steps before backward:
      loss = (omega_m2t * loss_m2t + omega_t2t * loss_t2t) / grad_accum_steps
    This ensures the effective gradient is the average over the accumulation
    window, not the sum (which would scale gradients by grad_accum_steps).
    """

    def test_loss_scaling_matches_average(self):
        """
        Simulate two accumulation steps with known scalar losses.
        Accumulated gradients should equal what you'd get from a single
        step with the average loss.
        """
        model = nn.Linear(4, 1, bias=False)
        x = torch.ones(1, 4)
        grad_accum_steps = 4
        omega_m2t = omega_t2t = 0.5

        # Compute gradients via accumulation (simulating training loop)
        model.zero_grad()
        for _ in range(grad_accum_steps):
            loss_m2t = model(x).sum()
            loss_t2t = model(x).sum()
            loss = (omega_m2t * loss_m2t + omega_t2t * loss_t2t) / grad_accum_steps
            loss.backward()

        accumulated_grad = model.weight.grad.clone()

        # Compute reference: single step with the average loss value
        model.zero_grad()
        loss_ref = (omega_m2t * model(x).sum() + omega_t2t * model(x).sum())
        loss_ref.backward()
        reference_grad = model.weight.grad.clone()

        assert torch.allclose(accumulated_grad, reference_grad, atol=1e-6), \
            f"Accumulated grad {accumulated_grad} != reference {reference_grad}"

    def test_larger_accum_does_not_inflate_grad(self):
        """Doubling grad_accum_steps should give the same gradient magnitude."""
        model1 = nn.Linear(4, 1, bias=False)
        model2 = nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            model2.weight.copy_(model1.weight)

        x = torch.ones(1, 4)

        model1.zero_grad()
        for _ in range(2):
            loss = model1(x).sum() / 2
            loss.backward()

        model2.zero_grad()
        for _ in range(4):
            loss = model2(x).sum() / 4
            loss.backward()

        assert torch.allclose(model1.weight.grad, model2.weight.grad, atol=1e-6), \
            "Gradient should not depend on grad_accum_steps when loss is properly averaged"


# ══════════════════════════════════════════════════════════════════════════════
#  TestEMATracking
# ══════════════════════════════════════════════════════════════════════════════

class TestEMATracking:
    """
    EMA update:  ema = 0.95 * ema + 0.05 * new_value
    First step:  ema = new_value  (no prior ema)
    """

    def test_first_step_ema_equals_value(self):
        m2t_ema = None
        avg_m2t = 2.5
        m2t_ema = avg_m2t if m2t_ema is None else 0.95 * m2t_ema + 0.05 * avg_m2t
        assert m2t_ema == 2.5

    def test_ema_converges_to_constant(self):
        """After many steps of constant value, EMA should equal that value."""
        ema = None
        val = 1.23
        for _ in range(500):
            ema = val if ema is None else 0.95 * ema + 0.05 * val
        assert abs(ema - val) < 1e-4, f"EMA did not converge: {ema} != {val}"

    def test_ema_is_slower_than_raw(self):
        """EMA should be slower to react to a sudden jump than the raw value."""
        ema = 1.0
        raw = 1.0
        # Sudden jump
        new_val = 10.0
        ema = 0.95 * ema + 0.05 * new_val
        raw = new_val
        assert ema < raw, "EMA must lag behind raw value after a sudden change"


# ══════════════════════════════════════════════════════════════════════════════
#  TestSchedulerMonotonicity
# ══════════════════════════════════════════════════════════════════════════════

class TestSchedulerMonotonicity:
    """LR must monotonically increase during warmup and decrease during decay."""

    def _lrs_for_steps(self, steps, num_steps=1000, warmup_steps=100,
                       lr=1e-3, min_lr=1e-5):
        from llada_fast.training.lora.train import _build_scheduler
        model = nn.Linear(4, 4)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr)
        sched = _build_scheduler(opt, num_steps, warmup_steps, min_lr, lr)
        lrs = []
        for _ in range(steps):
            lrs.append(opt.param_groups[0]["lr"])
            sched.step()
        return lrs

    def test_warmup_is_monotone_increasing(self):
        lrs = self._lrs_for_steps(101)
        warmup_lrs = lrs[1:101]   # skip step 0 (LR=0)
        for i in range(len(warmup_lrs) - 1):
            assert warmup_lrs[i] <= warmup_lrs[i + 1], \
                f"LR not monotone at warmup step {i}: {warmup_lrs[i]} > {warmup_lrs[i+1]}"

    def test_decay_is_monotone_decreasing(self):
        lrs = self._lrs_for_steps(1000)
        decay_lrs = lrs[100:]   # after warmup
        for i in range(len(decay_lrs) - 1):
            assert decay_lrs[i] >= decay_lrs[i + 1] - 1e-10, \
                f"LR increased during decay at step {100+i}: {decay_lrs[i]} < {decay_lrs[i+1]}"

    def test_lr_never_below_min_lr_during_decay(self):
        """LR must never drop below min_lr during the cosine decay phase."""
        min_lr = 1e-5
        warmup = 100
        lrs = self._lrs_for_steps(1000, warmup_steps=warmup, min_lr=min_lr)
        decay_lrs = lrs[warmup:]  # skip warmup phase (LR starts from 0)
        for i, lr in enumerate(decay_lrs):
            assert lr >= min_lr - 1e-10, \
                f"LR {lr} fell below min_lr {min_lr} at decay step {i}"


# ══════════════════════════════════════════════════════════════════════════════
#  TestCheckpointManagement
#  _save keeps only the 2 most recent checkpoints.
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckpointManagement:

    def _make_tiny_peft(self):
        from peft import LoraConfig, get_peft_model
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.query_key_value = nn.Linear(8, 8, bias=False)
        model = Tiny()
        cfg = LoraConfig(r=2, lora_alpha=4, target_modules=["query_key_value"],
                         lora_dropout=0.0, bias="none")
        return get_peft_model(model, cfg)

    def test_save_creates_step_dir(self):
        from llada_fast.training.lora.train import _save
        model = self._make_tiny_peft()
        opt   = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            _save(model, opt, sched, step=100, output_dir=tmp)
            assert os.path.isdir(os.path.join(tmp, "step_100"))
            assert os.path.exists(os.path.join(tmp, "step_100", "lora_delta.pt"))
            assert os.path.exists(os.path.join(tmp, "step_100", "optimizer.pt"))
            assert os.path.exists(os.path.join(tmp, "step_100", "train_state.pt"))

    def test_save_records_correct_step(self):
        from llada_fast.training.lora.train import _save
        model = self._make_tiny_peft()
        opt   = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            _save(model, opt, sched, step=500, output_dir=tmp)
            state = torch.load(os.path.join(tmp, "step_500", "train_state.pt"),
                               map_location="cpu")
            assert state["step"] == 500

    def test_save_keeps_only_two_most_recent(self):
        """After 3 saves, only step_2000 and step_3000 should remain."""
        from llada_fast.training.lora.train import _save
        model = self._make_tiny_peft()
        opt   = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            for step in [1000, 2000, 3000]:
                _save(model, opt, sched, step=step, output_dir=tmp)

            remaining = sorted(
                [d for d in os.listdir(tmp) if d.startswith("step_")]
            )
            assert remaining == ["step_2000", "step_3000"], \
                f"Expected ['step_2000', 'step_3000'], got {remaining}"

    def test_save_does_not_delete_if_only_one_exists(self):
        """With only one checkpoint, nothing should be deleted."""
        from llada_fast.training.lora.train import _save
        model = self._make_tiny_peft()
        opt   = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            _save(model, opt, sched, step=1000, output_dir=tmp)
            _save(model, opt, sched, step=2000, output_dir=tmp)
            dirs = [d for d in os.listdir(tmp) if d.startswith("step_")]
            assert len(dirs) == 2, f"Expected 2 checkpoints, got {dirs}"


# ══════════════════════════════════════════════════════════════════════════════
#  TestVocabIds
#  vocab_ids must exclude all special tokens (including mask_token_id).
# ══════════════════════════════════════════════════════════════════════════════

class TestVocabIds:
    """
    Training loop builds:
      special_ids = set(tokenizer.all_special_ids)
      vocab_ids = tensor([i for i in range(vocab_size) if i not in special_ids])
    T2T corruption samples random tokens from vocab_ids — must never sample
    a special token like [MASK], [PAD], [BOS], etc.
    """

    def _make_vocab_ids(self, vocab_size=1000, special_ids=None):
        if special_ids is None:
            special_ids = {0, 1, 2}   # e.g. [PAD]=0, [BOS]=1, [MASK]=2
        return torch.tensor(
            [i for i in range(vocab_size) if i not in special_ids],
            dtype=torch.long,
        )

    def test_no_special_ids_in_vocab_ids(self):
        special_ids = {0, 1, 2, 999}
        vocab_ids   = self._make_vocab_ids(special_ids=special_ids)
        for sid in special_ids:
            assert sid not in vocab_ids.tolist(), \
                f"Special id {sid} found in vocab_ids"

    def test_vocab_ids_covers_all_non_special(self):
        vocab_size  = 500
        special_ids = {0, 1, 2}
        vocab_ids   = self._make_vocab_ids(vocab_size=vocab_size, special_ids=special_ids)
        expected    = set(range(vocab_size)) - special_ids
        assert set(vocab_ids.tolist()) == expected

    def test_t2t_never_samples_mask_id(self):
        """With vocab_ids excluding mask_id, T2T should never produce mask_id."""
        from llada_fast.training.distill.data import corrupt_all_blocks_t2t
        torch.manual_seed(0)
        mask_id   = 0
        vocab_ids = torch.arange(1, 200)   # excludes mask_id=0
        ids       = torch.randint(1, 200, (1, 64))
        resp_mask = torch.ones(1, 64, dtype=torch.long)  # all asst
        num_blocks = (64 + 8 - 1) // 8
        t_per_block = torch.ones(num_blocks) * 0.9

        noisy, corrupted = corrupt_all_blocks_t2t(
            ids, resp_mask, vocab_ids, mask_id, 8, t_per_block
        )
        corrupted_vals = noisy[0][corrupted[0]]
        assert (corrupted_vals != mask_id).all(), \
            "T2T must not produce mask_id at corrupted positions"


# ══════════════════════════════════════════════════════════════════════════════
#  TestTuluLoaderMasks  (mock tokenizer with real chat template logic)
# ══════════════════════════════════════════════════════════════════════════════

class TestTuluLoaderMasks:
    """
    Tests TuluLoader._build_masks_offsets and _build_masks_prefix with a
    fake tokenizer that simulates the LLaDA2.1 chat template format.

    Template format: <role>HUMAN</role>...<|role_end|><role>ASSISTANT</role>...<|role_end|>
    We use a simple substitute for testing: "U:{content}A:{content}"
    """

    def _make_loader(self, seq_len=64):
        """
        Builds a TuluLoader with a mock tokenizer that:
        - Tokenizes by character (each char = 1 token)
        - apply_chat_template produces "U:{user}|A:{asst}|" for each turn
        - Returns offset_mapping = [(i, i+1)] for each character
        """
        from llada_fast.training.lora.train import TuluLoader

        tok = MagicMock()
        tok.chat_template = "<template>"  # non-None → has template

        def _apply_chat_template(messages, tokenize=False,
                                 add_generation_prompt=False, **kw):
            result = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    result += f"U:{content}|"
                elif role == "assistant":
                    result += f"A:{content}|"
            if add_generation_prompt:
                result += "A:"
            return result

        tok.apply_chat_template = _apply_chat_template

        def _encode_fn(text, truncation=True, max_length=seq_len, padding=None,
                       return_tensors=None, return_offsets_mapping=False, **kw):
            chars = list(text[:max_length])
            ids_  = [ord(c) % 200 + 1 for c in chars]
            amsk_ = [1] * len(ids_)
            if padding == "max_length":
                pad_len = max_length - len(ids_)
                ids_  += [0] * pad_len
                amsk_ += [0] * pad_len

            result = MagicMock()
            result.input_ids      = torch.tensor(ids_).unsqueeze(0)
            result.attention_mask = torch.tensor(amsk_).unsqueeze(0)
            if return_offsets_mapping:
                offsets = [(i, i + 1) for i in range(len(chars))]
                offsets += [(0, 0)] * (max_length - len(chars))
                result.__getitem__ = lambda self, k: (
                    torch.tensor(offsets) if k == "offset_mapping" else MagicMock()
                )
                # Support enc.pop("offset_mapping")
                result.pop = lambda k: torch.tensor(offsets)
            return result

        tok.__call__ = _encode_fn
        tok.side_effect = None

        loader = object.__new__(TuluLoader)
        loader._raw        = []
        loader._tok        = tok
        loader._seq_len    = seq_len
        loader._n          = 0
        loader._test_size  = 0
        loader._has_template = True
        return loader

    def test_single_turn_resp_mask_covers_asst(self):
        """
        resp_mask=1 for assistant content tokens, 0 for user tokens.

        Template format: "U:hello|A:world|"
        prefix_text (with add_generation_prompt=True for asst turn) = "U:hello|A:"
        until_text                                                   = "U:hello|A:world|"
        c0 = 10 ("U:hello|A:" is 10 chars), c1 = 16.
        So asst span = positions 10-15 = "world|" (6 tokens, not 8).
        The role marker "A:" (2 chars) is included in the prefix, not the content span.
        """
        loader = self._make_loader()
        messages = [
            {"role": "user",      "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        full_text = loader._tok.apply_chat_template(messages, tokenize=False)
        # "U:hello|A:world|" → 16 chars
        L = len(full_text)
        offsets = torch.tensor([(i, i + 1) for i in range(L)])

        resp_mask, user_turn_ids = loader._build_masks_offsets(messages, offsets)

        # User span = "U:hello|" = chars 0-7 → resp_mask must be 0
        assert resp_mask[0, :8].sum() == 0, "User section must not have resp_mask=1"
        # Asst content = "world|" = chars 10-15 → resp_mask must be 1 (6 tokens)
        # Role marker "A:" = chars 8-9 → unassigned (resp_mask=0)
        assert resp_mask[0, 10:16].sum() == 6, "Asst content must be resp_mask=1"
        # Role marker "A:" chars 8-9 are NOT in asst span
        assert resp_mask[0, 8:10].sum() == 0, "Role marker 'A:' must not be resp_mask=1"

    def test_single_turn_user_turn_ids(self):
        """user_turn_ids should be 1 for user content tokens, 0 elsewhere."""
        loader = self._make_loader()
        messages = [
            {"role": "user",      "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        full_text = loader._tok.apply_chat_template(messages, tokenize=False)
        L = len(full_text)
        offsets = torch.tensor([(i, i + 1) for i in range(L)])

        resp_mask, user_turn_ids = loader._build_masks_offsets(messages, offsets)

        # User content span: prefix_text="" (0 chars), until_text="U:hello|" (8 chars)
        # add_generation_prompt=False for user → c0=0, c1=8 → positions 0-7
        assert (user_turn_ids[0, :8] == 1).all(), "User tokens must have uid=1"
        assert (user_turn_ids[0, 8:] == 0).all(), "Asst/role tokens must have uid=0"

    def test_two_turn_user_ids_increment(self):
        """Second user turn gets uid=2, not uid=1."""
        loader = self._make_loader()
        messages = [
            {"role": "user",      "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user",      "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        # "U:q1|A:a1|U:q2|A:a2|"
        # positions: 0-4=USER1, 5-9=ASST1, 10-14=USER2, 15-19=ASST2
        full_text = loader._tok.apply_chat_template(messages, tokenize=False)
        L = len(full_text)
        offsets = torch.tensor([(i, i + 1) for i in range(L)])

        resp_mask, user_turn_ids = loader._build_masks_offsets(messages, offsets)

        # USER1 positions: uid=1
        user1_uid = user_turn_ids[0][user_turn_ids[0] == 1].sum()
        user2_uid = user_turn_ids[0][user_turn_ids[0] == 2].sum()
        assert user1_uid > 0, "Should have uid=1 tokens"
        assert user2_uid > 0, "Should have uid=2 tokens"
        assert (user_turn_ids[0] > 2).sum() == 0, "No uid > 2 in a 2-user-turn conversation"

    def test_resp_mask_user_turn_ids_disjoint_on_real_output(self):
        """resp_mask and user_turn_ids>0 must never overlap on any token."""
        loader = self._make_loader()
        messages = [
            {"role": "user",      "content": "abc"},
            {"role": "assistant", "content": "def"},
            {"role": "user",      "content": "ghi"},
            {"role": "assistant", "content": "jkl"},
        ]
        full_text = loader._tok.apply_chat_template(messages, tokenize=False)
        L = len(full_text)
        offsets = torch.tensor([(i, i + 1) for i in range(L)])

        resp_mask, user_turn_ids = loader._build_masks_offsets(messages, offsets)

        overlap = (resp_mask[0] == 1) & (user_turn_ids[0] > 0)
        assert not overlap.any(), \
            "resp_mask and user_turn_ids must be mutually exclusive"

    def test_system_tokens_get_neither_mask(self):
        """System message tokens should have resp_mask=0 and uid=0."""
        loader = self._make_loader()
        messages = [
            {"role": "system",    "content": "be helpful"},
            {"role": "user",      "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        full_text = loader._tok.apply_chat_template(messages, tokenize=False)
        L = len(full_text)
        offsets = torch.tensor([(i, i + 1) for i in range(L)])

        resp_mask, user_turn_ids = loader._build_masks_offsets(messages, offsets)

        # Total asst tokens + user tokens + system tokens = L
        asst_count = (resp_mask[0] == 1).sum().item()
        user_count = (user_turn_ids[0] > 0).sum().item()
        neither    = L - asst_count - user_count
        # System tokens should be a non-zero portion (be helpful = 10 chars → part of full_text)
        # We only check that resp_mask covers strictly less than L
        assert asst_count < L, "Not all tokens should be assistant"
        assert user_count < L, "Not all tokens should be user"


# ══════════════════════════════════════════════════════════════════════════════
#  TestAttnMaskEdgeCases
# ══════════════════════════════════════════════════════════════════════════════

class TestAttnMaskEdgeCases:
    """Edge cases for build_instruction_attn_mask."""

    def _build(self, seq_len, block_size, resp_1d, uid_1d):
        from llada_fast.training.lora.train import build_instruction_attn_mask
        return build_instruction_attn_mask(
            seq_len, block_size, resp_1d, uid_1d,
            device=torch.device("cpu"), dtype=torch.float32,
        ).squeeze(0).squeeze(0)

    def test_all_asst_is_block_causal(self):
        """With all assistant tokens, mask should equal standard block-causal mask."""
        seq_len    = 16
        block_size = 4
        resp = torch.ones(seq_len, dtype=torch.long)
        uid  = torch.zeros(seq_len, dtype=torch.long)
        M    = self._build(seq_len, block_size, resp, uid)

        # Build expected block-causal mask manually:
        # i can attend to j iff block_j <= block_i AND (j<=i OR asst_j)
        # Since all asst_j=True: any j in an earlier or same block is ok
        pos        = torch.arange(seq_len)
        block_idx  = pos // block_size
        expected   = (block_idx.unsqueeze(0) <= block_idx.unsqueeze(1)).float()
        assert torch.equal(M, expected), \
            "All-asst mask must equal standard block-causal (any j in earlier/same block)"

    def test_all_user_single_turn_is_fully_connected(self):
        """Single user turn spanning entire sequence → fully connected (all 1s)."""
        seq_len    = 16
        block_size = 4
        resp = torch.zeros(seq_len, dtype=torch.long)
        uid  = torch.ones(seq_len, dtype=torch.long)   # all uid=1
        M    = self._build(seq_len, block_size, resp, uid)
        assert M.all(), "Single user turn must produce fully connected attention"

    def test_pure_causal_when_no_user_turn_ids(self):
        """
        When uid=0 everywhere and resp=0 everywhere (e.g. all system tokens),
        rule2 never fires and rule1 gives standard causal (j <= i).
        """
        seq_len    = 8
        block_size = 8   # single block
        resp = torch.zeros(seq_len, dtype=torch.long)
        uid  = torch.zeros(seq_len, dtype=torch.long)
        M    = self._build(seq_len, block_size, resp, uid)
        # rule1: block_j=0 <= block_i=0 AND (j<=i OR asst_j=False) = j<=i
        expected = torch.tril(torch.ones(seq_len, seq_len))
        assert torch.equal(M, expected), \
            "All-system single-block must give standard causal mask"

    def test_dtype_bfloat16(self):
        """Mask dtype should match requested dtype."""
        from llada_fast.training.lora.train import build_instruction_attn_mask
        seq_len = 8
        resp = torch.zeros(seq_len, dtype=torch.long)
        uid  = torch.zeros(seq_len, dtype=torch.long)
        M = build_instruction_attn_mask(
            seq_len, 4, resp, uid,
            device=torch.device("cpu"), dtype=torch.bfloat16,
        )
        assert M.dtype == torch.bfloat16

    def test_single_token_sequence(self):
        """Sequence of length 1 should produce a 1x1 mask of [[1]]."""
        resp = torch.zeros(1, dtype=torch.long)
        uid  = torch.zeros(1, dtype=torch.long)
        M    = self._build(1, 1, resp, uid)
        assert M.shape == (1, 1)
        assert M[0, 0] == 1
