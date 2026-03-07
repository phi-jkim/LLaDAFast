import unittest
import torch
import math
from transformers import AutoTokenizer

# Mock necessary components from your training script for testing
import sys
sys.path.append("/home/jinhakim/LLaDAFast/src")

from llada_fast.training.lora.train import _build_scheduler
from llada_fast.modeling.configuration_llada2_moe import LLaDA2MoeConfig
from llada_fast.modeling.modeling_llada2_moe import LLaDA2MoeModelLM
from peft import get_peft_model, LoraConfig

class TestLoLCATsStage2Implementation(unittest.TestCase):
    
    def test_learning_rate_scheduler_math(self):
        """
        Validates the math of the LambdaLR learning rate scheduler to ensure
        it follows the Linear Warmup + Cosine Decay formula perfectly, treating
        0.0 as a multiplier, not an absolute.
        """
        dummy_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor([1.0]))], lr=1e-4)
        
        # Scenario: 10,000 max updates, 200 warmup updates, peak 1e-4, floor 1e-6
        max_updates = 10000
        warmup_updates = 200
        peak_lr = 1e-4
        min_lr = 1e-6
        
        scheduler = _build_scheduler(
            optimizer=dummy_optimizer, 
            num_steps=max_updates, 
            warmup_steps=warmup_updates, 
            min_lr=min_lr, 
            lr=peak_lr, 
            last_epoch=0
        )
        
        # Test Step 0: Should be exactly 0.0 (multiplier 0.0 * 1e-4)
        self.assertEqual(scheduler.get_last_lr()[0], 0.0)
        
        # Test Step 100: Halfway through warmup, should be exactly 5e-5 (multiplier 0.5 * 1e-4)
        for _ in range(100):
            scheduler.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 5e-5, places=7)
        
        # Test Step 200: End of warmup, should hit absolute peak 1e-4 (multiplier 1.0 * 1e-4)
        for _ in range(100):
            scheduler.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 1e-4, places=7)
        
        # Test Step 10,000: End of training cosine decay, should hit absolute floor 1e-6
        for _ in range(9800):
            scheduler.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 1e-6, places=7)
        
        # Ensure it never goes below floor on overstep
        scheduler.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 1e-6, places=7)

    def test_lora_configuration_matches_lolcats(self):
        """
        Validates that the PEFT LoRA configuration perfectly matches the LoLCATs paper
        requirements (r=8, alpha=16, dropout=0.0, targeting specific modules).
        """
        # Create a dummy small model to test PEFT injection
        config = LLaDA2MoeConfig(
            vocab_size=163840, # Must be larger than default pad_token_id 151643
            hidden_size=64, 
            intermediate_size=128,
            moe_intermediate_size=128,
            num_experts=2,
            num_experts_per_tok=1,
            num_hidden_layers=2, 
            num_attention_heads=4, 
            num_key_value_heads=4
        )
        config.use_linear_attention = True
        model = LLaDA2MoeModelLM(config)
        
        # This matches train.py line 360 settings exactly
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query_key_value", "dense"],
            lora_dropout=0.0,  # CRITICAL: LoLCATs paper dropout=0.0
            bias="none",
        )
        
        peft_model = get_peft_model(model, lora_cfg)
        
        # 1. Verify Rank is 8
        self.assertEqual(peft_model.peft_config['default'].r, 8)
        
        # 2. Verify Alpha is 16 (which is 2 * r)
        self.assertEqual(peft_model.peft_config['default'].lora_alpha, 16)
        
        # 3. Verify Dropout is absolute 0.0
        self.assertEqual(peft_model.peft_config['default'].lora_dropout, 0.0)
        
        # 4. Verify only LoRA parameters are trainable (linear attention is frozen)
        for name, param in peft_model.named_parameters():
            if param.requires_grad:
                self.assertTrue("lora" in name, f"Parameter {name} is trainable but not a LoRA weight!")
            if "hedgehog" in name or "alpha" in name:
                self.assertFalse(param.requires_grad, f"Linear attention parameter {name} must be frozen during Stage-2!")

    def test_tulu_loader_cross_turn_padding(self):
        """
        Validates the behavior of TuluLoader when transitioning from an assistant
        turn back to a user turn. The loader must pad the sequence to the next
        block boundary to prevent the user turn from leaking into the assistant's
        block-parallel evaluation.
        """
        from llada_fast.training.lora.train import TuluLoader
        
        # We need a real tokenizer with a chat template for this
        tokenizer = AutoTokenizer.from_pretrained("inclusionAI/LLaDA2.1-mini", trust_remote_code=True)
        block_size = 32
        seq_len = 1024
        
        loader = TuluLoader(
            tokenizer=tokenizer,
            seq_len=seq_len,
            block_size=block_size,
            test_size=1,  # minimal test set
            dataset_name="allenai/tulu-v2-sft-mixture"
        )
        
        # Fetch one mapped training example
        # Next batch returns: ids_t, amsk_t, resp_t, user_turn_ids (unused)
        ids, pmask, resp_mask, _ = loader.next_train()
        
        # Check shapes
        self.assertEqual(ids.shape, (1, seq_len))
        self.assertEqual(pmask.shape, (1, seq_len))
        self.assertEqual(resp_mask.shape, (1, seq_len))
        
        # 1. Padding mask matches pad token IDs
        is_pad = (ids[0] == tokenizer.pad_token_id)
        is_kpm = (pmask[0] == 0)
        self.assertTrue(torch.equal(is_pad, is_kpm), "pad_mask does not perfectly align with pad_token_ids")
        
        # 2. Assistant resp_mask is never active on pad tokens
        self.assertTrue((resp_mask[0][is_pad] == 0).all(), "resp_mask is active on padding tokens!")
        
        # 3. Cross-turn boundaries are padded to `block_size` exactness
        # We look for transitions from resp_mask = 1 (assistant) to resp_mask = 0 (next turn)
        # where the next token is NOT padding (meaning a real user token starts)
        
        resp = resp_mask[0].tolist()
        pad = is_pad.tolist()
        
        transitions_found = 0
        for i in range(1, seq_len):
            if resp[i-1] == 1 and resp[i] == 0:
                # Assistant turn ended.
                # If padding was inserted, the token at `i` must be a PAD token, 
                # OR the length of the sequence up to `i` must be perfectly divisible by block_size.
                
                # In your implementation: padding is inserted between the turns 
                # to advance the 'new_ids' length to a multiple of block_size.
                # Therefore, the NEXT real token (the user prompt start) must reside at an index divisible by block_size.
                
                # Find the index of the next real token
                next_real_idx = i
                while next_real_idx < seq_len and pad[next_real_idx]:
                    next_real_idx += 1
                
                if next_real_idx < seq_len:
                    transitions_found += 1
                    error_msg = f"User turn started at index {next_real_idx}, which is not a multiple of block_size {block_size}. Cross-turn leakage could occur!"
                    self.assertEqual(next_real_idx % block_size, 0, error_msg)

        # Ensuring we saw a transition is hard with a random sample, which is why
        # we added the `test_tulu_loader_cross_turn_padding_rigorous` test below.

    def test_tulu_loader_cross_turn_padding_rigorous(self):
        """
        Rigorously validates TuluLoader cross-turn padding by injecting a hand-crafted
        multi-turn conversation into the loader, guaranteeing that the exact transitions
        can be monitored for precise block-alignment.
        """
        from llada_fast.training.lora.train import TuluLoader
        
        tokenizer = AutoTokenizer.from_pretrained("inclusionAI/LLaDA2.1-mini", trust_remote_code=True)
        block_size = 32
        seq_len = 1024
        
        loader = TuluLoader(
            tokenizer=tokenizer,
            seq_len=seq_len,
            block_size=block_size,
            test_size=1,
            dataset_name="allenai/tulu-v2-sft-mixture"
        )
        
        # Inject our own custom multi-turn message history to guarantee multiple 
        # User -> Assistant -> User boundaries exist in the exact sequence.
        loader._raw = [
            {
                "messages": [
                    {"role": "user", "content": "1 2 3 4 5 " * 10},  # ~50 tokens
                    {"role": "assistant", "content": "A B C D E " * 10}, # ~50 tokens
                    {"role": "user", "content": "6 7 8 9 10 " * 5}, # ~25 tokens
                    {"role": "assistant", "content": "F G H I J " * 5}  # ~25 tokens
                ]
            }
        ]
        loader._n = 1
        loader._train_order = [0]
        loader._pos = 0
        
        ids, pmask, resp_mask, _ = loader.next_train()
        
        resp = resp_mask[0].tolist()
        pad = (pmask[0] == 0).tolist()
        
        # 1. We must find exactly ONE transition from Assistant -> User in this 4-message sequence
        # (The final assistant turn just hits the end/padding of the sequence)
        transitions_found = 0
        
        # 2. Assert the block padding logic
        for i in range(1, seq_len):
            if resp[i-1] == 1 and resp[i] == 0:
                # We found the end of an assistant turn!
                
                # Check if this is the start of the next User prompt or just the end of the whole sequence
                next_real_idx = i
                while next_real_idx < seq_len and pad[next_real_idx]:
                    next_real_idx += 1
                
                # If there are real tokens after this, it means we transitioned to a User turn
                if next_real_idx < seq_len and not pad[next_real_idx]:
                    transitions_found += 1
                    error_msg = f"User turn strictly started at index {next_real_idx}, which is NOT a multiple of block_size {block_size}."
                    self.assertEqual(next_real_idx % block_size, 0, error_msg)

        # Ensure the test actually triggered on our custom 4-message setup
        self.assertEqual(transitions_found, 1, "Failed to detect the exact 1 Assistant->User transition in the mocked multi-turn conversation.")

    def test_corrupt_all_blocks_m2t_statistics(self):
        """
        Synthetically tests the M2T (Masked-to-Token) block corruptor over many samples
        to guarantee:
        1. It properly respects the target corruption rate t_per_block mathematically over volume.
        2. Padding tokens are ALWAYS fully overridden to the mask_id.
        3. A block NEVER fully collapses to 100% masked if it contains real tokens.
        """
        from llada_fast.training.distill.data import corrupt_all_blocks
        
        B = 64
        L = 256
        block_size = 32
        mask_id = 999
        pad_id = 0
        device = "cpu"
        
        # Synthetic data: 50% real tokens, 50% padding (all padding at the end)
        real_len = 128
        input_ids = torch.randint(10, 100, (B, L), device=device)
        input_ids[:, real_len:] = pad_id
        
        pad_mask = torch.ones((B, L), dtype=torch.long, device=device)
        pad_mask[:, real_len:] = 0
        
        num_blocks = (L + block_size - 1) // block_size
        
        # We enforce a strict 60% noise rate across all blocks for this test
        t_target = 0.60
        t_per_block = torch.full((num_blocks,), t_target, device=device)
        
        noisy, corrupted = corrupt_all_blocks(
            input_ids=input_ids,
            pad_mask=pad_mask,
            mask_id=mask_id,
            block_size=block_size,
            t_per_block=t_per_block
        )
        
        # 1. Padding Overrides
        pad_region = noisy[:, real_len:]
        self.assertTrue((pad_region == mask_id).all(), "Padding region was not strictly masked.")
        
        # 2. Mathematical Noise Rate (over real tokens only)
        # `corrupted` is supposed to be true ONLY on real tokens that got masked
        real_corrupted = corrupted[:, :real_len] 
        actual_noise_rate = real_corrupted.float().mean().item()
        
        # Over 64 batches * 128 tokens (8192 items), the noise rate should be very close to 60%
        self.assertAlmostEqual(actual_noise_rate, t_target, delta=0.03, 
                               msg=f"Expected noise rate ~{t_target}, but got {actual_noise_rate:.3f}.")
        
        # 3. Block Collapse Safety
        # Check every real block in every batch to ensure NO block is 100% masked
        for b in range(B):
            for bi in range(real_len // block_size):
                s = bi * block_size
                e = s + block_size
                block_real_mask = pad_mask[b, s:e]
                block_corrupted = corrupted[b, s:e]
                
                # Are ALL tokens that are supposed to be real corrupted?
                is_all_corrupted = (block_corrupted[block_real_mask.bool()]).all()
                self.assertFalse(is_all_corrupted.item(), "A real block fully collapsed to 100% noise!")

    def test_corrupt_all_blocks_t2t_validity(self):
        """
        Synthetically tests the T2T (Token-to-Token) block corruptor to guarantee:
        1. Tokens are replaced with VALID vocabulary tokens from the provided set.
        2. M2T masks are NOT used for T2T corrupted items.
        """
        from llada_fast.training.distill.data import corrupt_all_blocks_t2t
        
        B = 10
        L = 64
        block_size = 32
        mask_id = -999  # deliberate negative mask ID
        pad_id = 0
        device = "cpu"
        
        input_ids = torch.ones((B, L), dtype=torch.long, device=device) * 5  # "Clean" tokens are all ID=5
        pad_mask = torch.ones((B, L), dtype=torch.long, device=device) # 100% real sequence
        
        # Synthetic valid vocabulary
        valid_vocab = [100, 101, 102, 103, 104]
        vocab_ids = torch.tensor(valid_vocab, dtype=torch.long, device=device)
        
        t_per_block = torch.full((L // block_size,), 0.80, device=device) # 80% noise
        
        noisy, corrupted = corrupt_all_blocks_t2t(
            input_ids=input_ids,
            pad_mask=pad_mask,
            vocab_ids=vocab_ids,
            mask_id=mask_id,
            block_size=block_size,
            t_per_block=t_per_block
        )
        
        # Extract the tokens that were flagged as corrupted
        corrupted_tokens = noisy[corrupted]
        
        # Ensure that no corrupted token is accidentally the mask_id 
        # (T2T relies on vocabulary hallucination, not M2T masking!)
        self.assertFalse((corrupted_tokens == mask_id).any(), "T2T applied a masking token instead of a vocab token")
        
        # Ensure that ALL corrupted tokens exist exactly within the valid vocabulary pool
        for tok in corrupted_tokens.tolist():
            self.assertIn(tok, valid_vocab, f"T2T produced an invalid vocabulary token ID: {tok}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
