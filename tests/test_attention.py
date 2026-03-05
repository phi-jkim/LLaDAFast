"""
Rigorous tests for:
  1. Masks (build_bd3lm_mask, build_block_causal_mask) — imported from lightweight masks.py
  2. OrderInvariantKernelLinearAttention — forward()
  3. BlockSoftmaxLinearHybrid — forward()
  4. BlockSoftmaxLinearHybrid — forward_bd3lm() staircase semantics
  5. Alpha mixing behavior
  6. Gradient correctness

Run with:
    pytest tests/test_attention.py -v
"""
import math
import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace


# ── Config helper ────────────────────────────────────────────────────────────

def make_cfg(hidden_size=128, num_heads=4, num_kv_heads=4, feature_dim=32, block_size=8):
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=hidden_size // num_heads,
        feature_dim=feature_dim,
        block_size=block_size,
    )

def rand_qkv(B, H, L, D, dtype=torch.float32, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return (
        torch.randn(B, H, L, D, dtype=dtype),
        torch.randn(B, H, L, D, dtype=dtype),
        torch.randn(B, H, L, D, dtype=dtype),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  1. Mask tests — import from lightweight masks.py (no datasets/transformers)
# ══════════════════════════════════════════════════════════════════════════════

class TestBlockCausalMask:

    @pytest.fixture(autouse=True)
    def setup(self):
        from llada_fast.modeling.masks import build_block_causal_mask
        self.build = build_block_causal_mask

    def _mask(self, seq_len=32, block_size=8):
        return self.build(seq_len, block_size, torch.device("cpu"), torch.float32).squeeze()

    def test_shape(self):
        m = self.build(32, 8, torch.device("cpu"), torch.float32)
        assert m.shape == (1, 1, 32, 32)

    def test_values_binary(self):
        m = self._mask()
        assert set(m.unique().tolist()) <= {0.0, 1.0}

    def test_within_block_bidirectional(self):
        m = self._mask()
        for i in range(8):
            for j in range(8):
                assert m[i, j] == 1.0

    def test_causal_block0_cannot_see_block1(self):
        m = self._mask()
        for i in range(8):
            for j in range(8, 16):
                assert m[i, j] == 0.0, f"block0 row {i} must not see block1 col {j}"

    def test_block1_sees_block0(self):
        m = self._mask()
        for i in range(8, 16):
            for j in range(8):
                assert m[i, j] == 1.0

    def test_non_multiple_length(self):
        m = self.build(30, 8, torch.device("cpu"), torch.float32)
        assert m.shape == (1, 1, 30, 30)
        assert set(m.squeeze().unique().tolist()) <= {0.0, 1.0}

    def test_single_block_fully_attended(self):
        m = self._mask(8, 8)
        assert (m == 1.0).all()

    def test_last_block_sees_all_previous(self):
        m = self._mask(32, 8)  # 4 blocks
        # Last block (rows 24..31) must see all earlier tokens
        for i in range(24, 32):
            for j in range(24):
                assert m[i, j] == 1.0


class TestBD3LMMask:

    @pytest.fixture(autouse=True)
    def setup(self):
        from llada_fast.modeling.masks import build_bd3lm_mask
        self.build = build_bd3lm_mask

    def _mask(self, seq_len=32, block_size=8):
        return self.build(seq_len, block_size, torch.device("cpu"), torch.float32).squeeze()

    def test_shape(self):
        m = self.build(32, 8, torch.device("cpu"), torch.float32)
        assert m.shape == (1, 1, 64, 64)

    def test_values_binary(self):
        m = self._mask()
        assert set(m.unique().tolist()) <= {0.0, 1.0}

    def test_M_BD_noisy_within_block(self):
        m = self._mask(); L = 32
        for i in range(8):
            for j in range(8):
                assert m[i, j] == 1.0

    def test_M_BD_clean_within_block(self):
        m = self._mask(); L = 32
        for i in range(L, L + 8):
            for j in range(L, L + 8):
                assert m[i, j] == 1.0

    def test_M_OBC_noisy_block1_sees_clean_block0(self):
        m = self._mask(); L = 32
        for i in range(8, 16):
            for j in range(L, L + 8):
                assert m[i, j] == 1.0, f"noisy_b1 row {i} should see clean_b0 col {j}"

    def test_M_OBC_noisy_block1_not_see_clean_block1(self):
        m = self._mask(); L = 32
        for i in range(8, 16):
            for j in range(L + 8, L + 16):
                assert m[i, j] == 0.0, f"noisy_b1 must NOT see clean_b1"

    def test_M_OBC_noisy_block0_sees_no_clean(self):
        m = self._mask(); L = 32
        for i in range(8):
            for j in range(L, 2 * L):
                assert m[i, j] == 0.0

    def test_M_BC_clean_causal(self):
        m = self._mask(); L = 32
        # clean_b1 sees clean_b0
        for i in range(L + 8, L + 16):
            for j in range(L, L + 8):
                assert m[i, j] == 1.0
        # clean_b0 does NOT see clean_b1
        for i in range(L, L + 8):
            for j in range(L + 8, L + 16):
                assert m[i, j] == 0.0

    def test_no_noisy_cross_block_attention(self):
        m = self._mask()
        for i in range(8):
            for j in range(8, 16):
                assert m[i, j] == 0.0

    def test_no_clean_to_noisy(self):
        m = self._mask(); L = 32
        for i in range(L, 2 * L):
            for j in range(L):
                assert m[i, j] == 0.0

    def test_non_power_of_two_length(self):
        m = self.build(30, 8, torch.device("cpu"), torch.float32)
        assert m.shape == (1, 1, 60, 60)
        assert set(m.squeeze().unique().tolist()) <= {0.0, 1.0}

    def test_mask_row_sums_increase_with_block(self):
        """Each successive noisy block should have >= as many attended positions."""
        m = self._mask(32, 8); L = 32
        row_sums = [m[i * 8, :].sum().item() for i in range(4)]  # first token of each noisy block
        for a, b in zip(row_sums, row_sums[1:]):
            assert b >= a, "Later noisy blocks should attend to at least as much"


# ══════════════════════════════════════════════════════════════════════════════
#  2. OrderInvariantKernelLinearAttention
# ══════════════════════════════════════════════════════════════════════════════

class TestOrderInvariantLinear:

    @pytest.fixture(autouse=True)
    def setup(self):
        from llada_fast.modeling.linear_attention import OrderInvariantKernelLinearAttention
        self.cls = OrderInvariantKernelLinearAttention
        self.cfg = make_cfg()

    def test_output_shape(self):
        attn = self.cls(self.cfg, block_size=8)
        q, k, v = rand_qkv(2, 4, 40, 32)
        assert attn(q, k, v).shape == (2, 4, 40, 32)

    def test_non_multiple_seq_len(self):
        attn = self.cls(self.cfg, block_size=8)
        q, k, v = rand_qkv(1, 4, 30, 32)
        assert attn(q, k, v).shape == (1, 4, 30, 32)

    def test_finite_outputs(self):
        attn = self.cls(self.cfg, block_size=8)
        q, k, v = rand_qkv(2, 4, 32, 32)
        assert attn(q, k, v).isfinite().all()

    def test_no_future_block_leakage(self):
        attn = self.cls(self.cfg, block_size=8)
        q, k, v = rand_qkv(1, 4, 32, 32)
        out1 = attn(q, k, v)
        v2 = v.clone(); v2[:, :, 8:16, :] += 5.0
        out2 = attn(q, k, v2)
        assert torch.allclose(out1[:, :, :8, :], out2[:, :, :8, :], atol=1e-5)
        assert not torch.allclose(out1[:, :, 8:, :], out2[:, :, 8:, :], atol=1e-5)

    def test_hedgehog_grad_flows(self):
        attn = self.cls(self.cfg, block_size=8)
        q, k, v = rand_qkv(1, 4, 16, 32)
        attn(q, k, v).sum().backward()
        assert attn.hedgehog_weights.grad is not None
        assert attn.hedgehog_weights.grad.abs().max() > 0


# ══════════════════════════════════════════════════════════════════════════════
#  3. BlockSoftmaxLinearHybrid — forward()
# ══════════════════════════════════════════════════════════════════════════════

class TestHybridForward:

    @pytest.fixture(autouse=True)
    def setup(self):
        from llada_fast.modeling.hybrid_attention import BlockSoftmaxLinearHybrid
        self.cls = BlockSoftmaxLinearHybrid
        self.cfg = make_cfg()

    def _attn(self, **kw):
        return self.cls(self.cfg, block_size=8)

    def test_output_shape(self):
        q, k, v = rand_qkv(2, 4, 40, 32)
        assert self._attn()(q, k, v).shape == (2, 4, 40, 32)

    def test_non_multiple_seq_len(self):
        q, k, v = rand_qkv(1, 4, 30, 32)
        assert self._attn()(q, k, v).shape == (1, 4, 30, 32)

    def test_finite_outputs(self):
        q, k, v = rand_qkv(2, 4, 32, 32)
        assert self._attn()(q, k, v).isfinite().all()

    def test_no_cross_block_future_leakage(self):
        """Block 0 output must not change when block 1's V changes."""
        attn = self._attn()
        q, k, v = rand_qkv(1, 4, 32, 32)
        out1 = attn(q, k, v)
        v2 = v.clone(); v2[:, :, 8:16, :] += 5.0
        out2 = attn(q, k, v2)
        assert torch.allclose(out1[:, :, :8, :], out2[:, :, :8, :], atol=1e-5)

    def test_within_block_softmax_bidirectionality(self):
        """Changing v[j] in block n must change output at i in same block (softmax dominant)."""
        attn = self._attn()
        attn.alpha.data.fill_(10.0)  # softmax dominant
        q, k, v = rand_qkv(1, 4, 32, 32)
        out1 = attn(q, k, v)
        v2 = v.clone(); v2[:, :, 5, :] += 5.0   # pos 5 in block 0
        out2 = attn(q, k, v2)
        # pos 3 (same block) changes
        assert not torch.allclose(out1[:, :, 3, :], out2[:, :, 3, :], atol=1e-5)

    def test_linear_cross_block_always_present(self):
        """With shared normalization, linear stream always provides cross-block context.
        Even with high alpha (softmax dominant), block 1 output changes when block 0 V
        changes, because linear_factor=1 is independent of alpha."""
        attn = self._attn()
        attn.alpha.data.fill_(10.0)  # strongly softmax-dominant
        q, k, v = rand_qkv(1, 4, 32, 32, seed=42)
        out1 = attn(q, k, v)
        v2 = v.clone(); v2[:, :, :8, :] += 5.0   # change entire block 0 V
        out2 = attn(q, k, v2)
        # Block 1 must see the update via linear state (cross-block always present)
        assert not torch.allclose(out1[:, :, 8:16, :], out2[:, :, 8:16, :], atol=1e-4), \
            "Block 1 should respond to block 0 V change even with high alpha (linear stream)"

    def test_cross_block_context_via_linear(self):
        """When alpha=0 (pure linear), block 1 output changes when block 0 v changes."""
        attn = self._attn()
        attn.alpha.data.fill_(-100.0)  # pure linear (softmax weight ≈ 0)
        q, k, v = rand_qkv(1, 4, 32, 32)
        out1 = attn(q, k, v)
        v2 = v.clone(); v2[:, :, :8, :] += 5.0   # change block 0 v
        out2 = attn(q, k, v2)
        # Block 1 should see the updated block 0 context
        assert not torch.allclose(out1[:, :, 8:16, :], out2[:, :, 8:16, :], atol=1e-4)

    def test_param_set(self):
        """Only hedgehog_weights and alpha should be parameters."""
        attn = self._attn()
        params = {n for n, _ in attn.named_parameters()}
        assert params == {"hedgehog_weights", "alpha"}

    def test_alpha_grad_flows(self):
        attn = self._attn()
        q, k, v = rand_qkv(1, 4, 16, 32)
        attn(q, k, v).sum().backward()
        assert attn.alpha.grad is not None
        assert attn.hedgehog_weights.grad is not None

    def test_key_padding_mask_suppresses_padding(self):
        attn = self._attn()
        q, k, v = rand_qkv(1, 4, 16, 32)
        out_full = attn(q, k, v)
        kpm = torch.ones(1, 16); kpm[:, 12:] = 0.0
        out_masked = attn(q, k, v, key_padding_mask=kpm)
        assert not torch.allclose(out_full, out_masked, atol=1e-5)
        assert out_masked.isfinite().all()

    def test_batch_dimension_independent(self):
        """Two items in a batch with different values should give independent outputs."""
        attn = self._attn()
        q, k, v = rand_qkv(2, 4, 16, 32)
        q[1] = q[0];  k[1] = k[0];  v[1] = v[0]  # make batch identical
        v[1, :, 5, :] += 5.0                        # perturb item 1
        out = attn(q, k, v)
        assert not torch.allclose(out[0], out[1], atol=1e-5)

    def test_deterministic(self):
        attn = self._attn()
        q, k, v = rand_qkv(1, 4, 16, 32, seed=0)
        out1 = attn(q, k, v)
        out2 = attn(q, k, v)
        assert torch.allclose(out1, out2)


# ══════════════════════════════════════════════════════════════════════════════
#  4. BlockSoftmaxLinearHybrid — forward_bd3lm()
# ══════════════════════════════════════════════════════════════════════════════

class TestHybridBD3LM:

    @pytest.fixture(autouse=True)
    def setup(self):
        from llada_fast.modeling.hybrid_attention import BlockSoftmaxLinearHybrid
        self.cls = BlockSoftmaxLinearHybrid
        self.cfg = make_cfg()

    def _attn(self):
        return self.cls(self.cfg, block_size=8)

    def test_output_shape(self):
        attn = self._attn()
        q, k, v = rand_qkv(2, 4, 64, 32)
        assert attn.forward_bd3lm(q, k, v, half_len=32).shape == (2, 4, 64, 32)

    def test_finite_outputs(self):
        attn = self._attn()
        q, k, v = rand_qkv(1, 4, 64, 32)
        assert attn.forward_bd3lm(q, k, v, half_len=32).isfinite().all()

    def test_non_multiple_half_len(self):
        attn = self._attn()
        q, k, v = rand_qkv(1, 4, 60, 32)
        out = attn.forward_bd3lm(q, k, v, half_len=30)
        assert out.shape == (1, 4, 60, 32)
        assert out.isfinite().all()

    def test_staircase_noisy_block0_ignores_all_clean(self):
        """Noisy block 0 has no prior clean state — clean V changes must not affect it."""
        attn = self._attn()
        q, k, v = rand_qkv(1, 4, 64, 32, seed=1)
        out1 = attn.forward_bd3lm(q, k, v, half_len=32)
        v2 = v.clone(); v2[:, :, 32:, :] += 5.0
        out2 = attn.forward_bd3lm(q, k, v2, half_len=32)
        assert torch.allclose(out1[:, :, :8, :], out2[:, :, :8, :], atol=1e-5), \
            "Noisy block 0 changed when only clean values changed"

    def test_staircase_noisy_block1_sees_clean_block0_only(self):
        """Noisy block 1 output responds to clean block 0 changes but not clean block 1+."""
        attn = self._attn()
        attn.alpha.data.fill_(0.0)  # 50/50 mix, ensures linear is visible
        q, k, v = rand_qkv(1, 4, 64, 32, seed=2)
        base = attn.forward_bd3lm(q, k, v, half_len=32)

        v_c0 = v.clone(); v_c0[:, :, 32:40, :] += 5.0   # clean block 0
        v_c1 = v.clone(); v_c1[:, :, 40:, :] += 5.0     # clean block 1+

        out_c0 = attn.forward_bd3lm(q, k, v_c0, half_len=32)
        out_c1 = attn.forward_bd3lm(q, k, v_c1, half_len=32)

        nb1 = slice(8, 16)
        assert not torch.allclose(base[:, :, nb1, :], out_c0[:, :, nb1, :], atol=1e-4), \
            "Noisy b1 should respond to clean b0 change"
        assert torch.allclose(base[:, :, nb1, :], out_c1[:, :, nb1, :], atol=1e-5), \
            "Noisy b1 must NOT respond to clean b1+ change"

    def test_noisy_keys_never_enter_state(self):
        """Changing noisy K/V must not affect clean outputs."""
        attn = self._attn()
        q, k, v = rand_qkv(1, 4, 64, 32, seed=3)
        out1 = attn.forward_bd3lm(q, k, v, half_len=32)

        k2 = k.clone(); k2[:, :, :32, :] += 5.0
        v2 = v.clone(); v2[:, :, :32, :] += 5.0
        out2 = attn.forward_bd3lm(q, k2, v2, half_len=32)

        assert torch.allclose(out1[:, :, 32:, :], out2[:, :, 32:, :], atol=1e-5), \
            "Clean outputs changed when only noisy K/V changed"

    def test_clean_linear_state_causally_accumulates(self):
        """Clean block 2 should see clean block 0 and 1 context via linear state."""
        attn = self._attn()
        attn.alpha.data.fill_(-100.0)   # pure linear
        q, k, v = rand_qkv(1, 4, 64, 32, seed=4)
        base = attn.forward_bd3lm(q, k, v, half_len=32)

        # Change clean block 0 v only
        v2 = v.clone(); v2[:, :, 32:40, :] += 10.0
        out2 = attn.forward_bd3lm(q, k, v2, half_len=32)

        # Clean block 1 (pos L+8..L+15) should change (it reads state including block 0)
        assert not torch.allclose(
            base[:, :, 40:48, :], out2[:, :, 40:48, :], atol=1e-4
        ), "Clean block 1 should see changed clean block 0 context"

    def test_clean_block_intra_softmax_exact(self):
        """Clean block sees itself via intra-block softmax (softmax dominant)."""
        attn = self._attn()
        attn.alpha.data.fill_(10.0)   # softmax dominant
        q, k, v = rand_qkv(1, 4, 64, 32, seed=5)
        base = attn.forward_bd3lm(q, k, v, half_len=32)

        v2 = v.clone(); v2[:, :, 32:40, :] += 5.0   # change clean block 0 V
        out2 = attn.forward_bd3lm(q, k, v2, half_len=32)

        assert not torch.allclose(
            base[:, :, 32:40, :], out2[:, :, 32:40, :], atol=1e-5
        ), "Clean block 0 output should change when its own V changes"

    def test_noisy_intra_softmax_bidirectional(self):
        """Noisy block i is bidirectional within itself via softmax stream."""
        attn = self._attn()
        attn.alpha.data.fill_(10.0)   # softmax dominant
        q, k, v = rand_qkv(1, 4, 64, 32, seed=6)
        base = attn.forward_bd3lm(q, k, v, half_len=32)

        v2 = v.clone(); v2[:, :, 5, :] += 5.0   # change pos 5 in noisy block 0
        out2 = attn.forward_bd3lm(q, k, v2, half_len=32)

        # pos 3 (also noisy block 0) should change
        assert not torch.allclose(base[:, :, 3, :], out2[:, :, 3, :], atol=1e-5)

    def test_key_padding_mask_applied_to_clean_only(self):
        attn = self._attn()
        q, k, v = rand_qkv(1, 4, 64, 32, seed=7)
        kpm_full = torch.ones(1, 64)
        out_full = attn.forward_bd3lm(q, k, v, half_len=32, key_padding_mask=kpm_full)

        kpm_part = kpm_full.clone(); kpm_part[:, 56:] = 0.0
        out_part = attn.forward_bd3lm(q, k, v, half_len=32, key_padding_mask=kpm_part)

        assert not torch.allclose(out_full, out_part, atol=1e-5)
        assert out_part.isfinite().all()

    def test_gradient_flows_through_bd3lm(self):
        attn = self._attn()
        q, k, v = rand_qkv(1, 4, 32, 32)
        attn.forward_bd3lm(q, k, v, half_len=16).sum().backward()
        assert attn.hedgehog_weights.grad is not None
        assert attn.alpha.grad is not None
        assert attn.hedgehog_weights.grad.abs().max() > 0

    def test_clean_block_queries_include_self_in_linear_state(self):
        """
        In forward_bd3lm, clean block n reads LINEAR state AFTER updating with itself.
        This differs from forward() (which reads state BEFORE update).
        Verify: changing clean block 0 v changes clean block 0 linear output (self-context).
        """
        attn = self._attn()
        attn.alpha.data.fill_(-100.0)  # pure linear
        q, k, v = rand_qkv(1, 4, 64, 32, seed=8)
        base = attn.forward_bd3lm(q, k, v, half_len=32)

        v2 = v.clone(); v2[:, :, 32:40, :] += 10.0   # change clean block 0
        out2 = attn.forward_bd3lm(q, k, v2, half_len=32)

        # Clean block 0 (rows 32..39) should change even in pure-linear mode
        # because state is updated with self BEFORE clean block 0 queries it
        assert not torch.allclose(
            base[:, :, 32:40, :], out2[:, :, 32:40, :], atol=1e-4
        ), "Clean block 0 output should reflect self-context (post-self-update state)"

    def test_batch_independence(self):
        attn = self._attn()
        q, k, v = rand_qkv(2, 4, 64, 32, seed=9)
        q[1] = q[0]; k[1] = k[0]; v[1] = v[0]
        v[1, :, 32, :] += 5.0  # perturb clean block 0 of item 1
        out = attn.forward_bd3lm(q, k, v, half_len=32)
        assert not torch.allclose(out[0], out[1], atol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
#  5. Alpha mixing
# ══════════════════════════════════════════════════════════════════════════════

class TestAlphaMixing:

    @pytest.fixture(autouse=True)
    def setup(self):
        from llada_fast.modeling.hybrid_attention import BlockSoftmaxLinearHybrid
        self.cls = BlockSoftmaxLinearHybrid
        self.cfg = make_cfg()

    def test_high_alpha_matches_pure_softmax_single_block(self):
        """sigma(100) ≈ 1: single block (no linear cross-block) → output ≈ sdpa."""
        attn = self.cls(self.cfg, block_size=8)
        attn.alpha.data.fill_(100.0)
        q, k, v = rand_qkv(1, 4, 8, 32, seed=10)
        with torch.no_grad():
            out_hybrid  = attn(q, k, v)
            out_softmax = F.scaled_dot_product_attention(q, k, v, scale=attn.scaling)
        assert torch.allclose(out_hybrid, out_softmax, atol=1e-5)

    def test_low_alpha_first_block_near_zero(self):
        """sigma(-100) ≈ 0: block 0 has no prior state → output ≈ 0 in pure-linear."""
        attn = self.cls(self.cfg, block_size=8)
        attn.alpha.data.fill_(-100.0)
        q, k, v = rand_qkv(1, 4, 8, 32)
        with torch.no_grad():
            out = attn(q, k, v)
        assert out.abs().max() < 1e-3

    def test_alpha_linearly_interpolates_outputs(self):
        """Output for intermediate alpha should lie between high-alpha and low-alpha outputs."""
        attn = self.cls(self.cfg, block_size=8)
        q, k, v = rand_qkv(1, 4, 16, 32, seed=11)
        with torch.no_grad():
            attn.alpha.data.fill_(10.0);  out_hi = attn(q, k, v).clone()
            attn.alpha.data.fill_(-10.0); out_lo = attn(q, k, v).clone()
            attn.alpha.data.fill_(0.0);   out_mid = attn(q, k, v).clone()
        # mid should be between hi and lo element-wise (within variance)
        lo_err = (out_mid - out_lo).abs().mean().item()
        hi_err = (out_mid - out_hi).abs().mean().item()
        all_err = (out_hi - out_lo).abs().mean().item()
        assert lo_err < all_err and hi_err < all_err, \
            "alpha=0 output should be closer to both extremes than they are to each other"

    def test_alpha_gradient(self):
        attn = self.cls(self.cfg, block_size=8)
        q, k, v = rand_qkv(1, 4, 8, 32)
        attn(q, k, v).sum().backward()
        assert attn.alpha.grad is not None
        assert attn.alpha.grad.abs().max() > 0


# ══════════════════════════════════════════════════════════════════════════════
#  6. Softmax (teacher) + BD3LM mask — end-to-end staircase enforcement
# ══════════════════════════════════════════════════════════════════════════════

class TestSoftmaxWithBD3LMMask:
    """
    Applies the BD3LM mask to vanilla scaled_dot_product_attention and
    verifies all three staircase rules hold under real softmax computation.
    This mirrors how the teacher processes the 2L sequence.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from llada_fast.modeling.masks import build_bd3lm_mask
        self.build_mask = build_bd3lm_mask
        self.L = 32
        self.S = 8   # block_size

    def _attn_with_mask(self, q, k, v, mask):
        """Run SDPA with a 0/1 mask (1=attend) → convert to additive mask for sdpa."""
        # sdpa expects additive mask: 0=attend, -inf=block
        additive = (1.0 - mask) * -1e9
        return F.scaled_dot_product_attention(q, k, v, attn_mask=additive)

    def _mask_and_call(self, q, k, v):
        """Build mask and run sdpa on 2L input."""
        mask = self.build_mask(self.L, self.S, q.device, q.dtype)  # (1,1,2L,2L)
        return self._attn_with_mask(q, k, v, mask)

    def test_output_shape(self):
        q, k, v = rand_qkv(1, 4, 2 * self.L, 32, seed=20)
        out = self._mask_and_call(q, k, v)
        assert out.shape == (1, 4, 2 * self.L, 32)

    def test_finite_outputs(self):
        q, k, v = rand_qkv(1, 4, 2 * self.L, 32, seed=21)
        assert self._mask_and_call(q, k, v).isfinite().all()

    def test_noisy_block0_cannot_see_clean(self):
        """
        Noisy block 0 (rows 0..7) must be independent of all clean values.
        Change all clean V → noisy block 0 output must be unchanged.
        """
        q, k, v = rand_qkv(1, 4, 2 * self.L, 32, seed=22)
        out1 = self._mask_and_call(q, k, v)
        v2 = v.clone(); v2[:, :, self.L:, :] += 5.0
        out2 = self._mask_and_call(q, k, v2)
        assert torch.allclose(out1[:, :, :self.S, :], out2[:, :, :self.S, :], atol=1e-5), \
            "Noisy block 0 changed when clean V changed"

    def test_noisy_block1_sees_clean_block0(self):
        """
        Noisy block 1 output must change when only clean block 0 V changes.
        """
        q, k, v = rand_qkv(1, 4, 2 * self.L, 32, seed=23)
        out1 = self._mask_and_call(q, k, v)
        v2 = v.clone(); v2[:, :, self.L:self.L + self.S, :] += 5.0   # clean block 0
        out2 = self._mask_and_call(q, k, v2)
        assert not torch.allclose(
            out1[:, :, self.S:2*self.S, :], out2[:, :, self.S:2*self.S, :], atol=1e-5
        ), "Noisy block 1 should be affected by clean block 0"

    def test_noisy_block1_ignores_clean_block1(self):
        """
        Noisy block 1 output must NOT change when only clean block 1 V changes.
        (M_OBC allows noisy_i → clean_j only for j < i, not j = i.)
        """
        q, k, v = rand_qkv(1, 4, 2 * self.L, 32, seed=24)
        out1 = self._mask_and_call(q, k, v)
        # Change clean block 1 only (global rows L+S .. L+2S)
        v2 = v.clone(); v2[:, :, self.L + self.S : self.L + 2*self.S, :] += 5.0
        out2 = self._mask_and_call(q, k, v2)
        assert torch.allclose(
            out1[:, :, self.S:2*self.S, :], out2[:, :, self.S:2*self.S, :], atol=1e-5
        ), "Noisy block 1 changed when only clean block 1 (same index) changed"

    def test_clean_causal_block0_ignores_clean_block1(self):
        """
        Clean block 0 (M_BC: block_q >= block_kv) must not see clean block 1.
        """
        q, k, v = rand_qkv(1, 4, 2 * self.L, 32, seed=25)
        out1 = self._mask_and_call(q, k, v)
        v2 = v.clone(); v2[:, :, self.L + self.S : self.L + 2*self.S, :] += 5.0
        out2 = self._mask_and_call(q, k, v2)
        assert torch.allclose(
            out1[:, :, self.L:self.L + self.S, :],
            out2[:, :, self.L:self.L + self.S, :], atol=1e-5
        ), "Clean block 0 changed when only clean block 1 V changed"

    def test_clean_block1_sees_clean_block0(self):
        """
        Clean block 1 output changes when clean block 0 V changes (causal cross-block).
        """
        q, k, v = rand_qkv(1, 4, 2 * self.L, 32, seed=26)
        out1 = self._mask_and_call(q, k, v)
        v2 = v.clone(); v2[:, :, self.L:self.L + self.S, :] += 5.0   # clean block 0
        out2 = self._mask_and_call(q, k, v2)
        assert not torch.allclose(
            out1[:, :, self.L + self.S : self.L + 2*self.S, :],
            out2[:, :, self.L + self.S : self.L + 2*self.S, :], atol=1e-5
        ), "Clean block 1 should see clean block 0 context"

    def test_clean_does_not_see_noisy(self):
        """
        Clean outputs must be fully independent of noisy V (no clean→noisy attention).
        """
        q, k, v = rand_qkv(1, 4, 2 * self.L, 32, seed=27)
        out1 = self._mask_and_call(q, k, v)
        v2 = v.clone(); v2[:, :, :self.L, :] += 5.0   # change all noisy V
        out2 = self._mask_and_call(q, k, v2)
        assert torch.allclose(
            out1[:, :, self.L:, :], out2[:, :, self.L:, :], atol=1e-5
        ), "Clean outputs changed when only noisy V changed"

    def test_noisy_within_block_bidirectional(self):
        """
        Softmax within a noisy block is bidirectional: changing v[j] in noisy block n
        changes output at i in the same block.
        """
        q, k, v = rand_qkv(1, 4, 2 * self.L, 32, seed=28)
        out1 = self._mask_and_call(q, k, v)
        v2 = v.clone(); v2[:, :, 5, :] += 5.0   # noisy block 0, pos 5
        out2 = self._mask_and_call(q, k, v2)
        # pos 3 (same noisy block 0) should change
        assert not torch.allclose(out1[:, :, 3, :], out2[:, :, 3, :], atol=1e-5)

    def test_noisy_no_cross_noisy_block_attention(self):
        """
        Noisy block 0 output must not change when noisy block 1 V changes.
        (No cross-block noisy attention.)
        """
        q, k, v = rand_qkv(1, 4, 2 * self.L, 32, seed=29)
        out1 = self._mask_and_call(q, k, v)
        v2 = v.clone(); v2[:, :, self.S:2*self.S, :] += 5.0   # noisy block 1
        out2 = self._mask_and_call(q, k, v2)
        assert torch.allclose(
            out1[:, :, :self.S, :], out2[:, :, :self.S, :], atol=1e-5
        ), "Noisy block 0 changed when only noisy block 1 V changed"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
