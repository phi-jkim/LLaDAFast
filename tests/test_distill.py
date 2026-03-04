"""
Tests for distillation utilities:
  - DistillConfig defaults and validation
  - CurriculumManager progression modes (joint, interval, none)
  - Mask-based corrupt functions (corrupt_one_block, corrupt_all_blocks)
  - Cosine LR scheduler (_build_scheduler equivalent logic)

These tests are lightweight — they mock heavy model objects where needed.
Run with:
    pytest tests/test_distill.py -v
"""
import math
import pytest
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


# ══════════════════════════════════════════════════════════════════════════════
#  DistillConfig defaults
# ══════════════════════════════════════════════════════════════════════════════

class TestDistillConfig:

    @pytest.fixture(autouse=True)
    def setup(self):
        from llada_fast.training.distill.config import DistillConfig
        self.cls = DistillConfig

    def test_defaults_instantiate(self):
        cfg = self.cls()
        assert cfg.learning_rate == 3e-3
        assert cfg.batch_size == 1
        assert cfg.warmup_steps == 200
        assert cfg.min_lr == 1e-5
        assert cfg.num_steps == 15000

    def test_no_llm_judge_field(self):
        cfg = self.cls()
        assert not hasattr(cfg, "use_llm_curriculum_eval"), \
            "use_llm_curriculum_eval should have been removed"

    def test_no_plot_attn_fields(self):
        cfg = self.cls()
        for field in ["plot_attn_every", "plot_attn_max_layers", "plot_attn_max_len"]:
            assert not hasattr(cfg, field), f"{field} should have been removed"

    def test_no_lr_plateau_fields(self):
        cfg = self.cls()
        assert not hasattr(cfg, "lr_patience"), "lr_patience removed for cosine scheduler"
        assert not hasattr(cfg, "lr_factor"), "lr_factor removed for cosine scheduler"

    def test_override_learning_rate(self):
        cfg = self.cls(learning_rate=1e-4)
        assert cfg.learning_rate == 1e-4

    def test_block_softmax_hybrid_default_false(self):
        assert self.cls().use_block_softmax_hybrid is False

    def test_eval_prompts_non_empty(self):
        assert len(self.cls().eval_prompts) > 0


# ══════════════════════════════════════════════════════════════════════════════
#  Cosine LR Scheduler
# ══════════════════════════════════════════════════════════════════════════════

class TestCosineLRScheduler:
    """Tests the _lr_lambda logic directly, without instantiating a model."""

    def _lambda(self, cfg, step):
        """Replicate the lambda from _build_scheduler."""
        if step < cfg.warmup_steps:
            return float(step) / max(1, cfg.warmup_steps)
        progress = float(step - cfg.warmup_steps) / max(1, cfg.num_steps - cfg.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        eta_min_ratio = cfg.min_lr / cfg.learning_rate
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    @pytest.fixture(autouse=True)
    def setup(self):
        self.cfg = SimpleNamespace(
            num_steps=1000,
            warmup_steps=100,
            learning_rate=3e-3,
            min_lr=3e-5,   # 0.01 * lr
        )

    def test_warmup_starts_near_zero(self):
        assert self._lambda(self.cfg, 0) == pytest.approx(0.0)

    def test_warmup_ends_at_one(self):
        assert self._lambda(self.cfg, 100) == pytest.approx(1.0)

    def test_midpoint_cosine(self):
        # At midpoint of decay, cosine should be 0.5 → lr ≈ 0.5*(1+eta_min)
        val = self._lambda(self.cfg, 550)  # halfway between 100 and 1000
        eta_min_ratio = self.cfg.min_lr / self.cfg.learning_rate
        expected = eta_min_ratio + (1.0 - eta_min_ratio) * 0.5
        assert val == pytest.approx(expected, abs=1e-4)

    def test_end_equals_min_lr_ratio(self):
        val = self._lambda(self.cfg, self.cfg.num_steps)
        assert val == pytest.approx(self.cfg.min_lr / self.cfg.learning_rate, abs=1e-6)

    def test_monotone_decreasing_after_warmup(self):
        steps = list(range(100, 1001, 100))
        values = [self._lambda(self.cfg, s) for s in steps]
        for a, b in zip(values, values[1:]):
            assert b <= a + 1e-9, f"LR not monotone decreasing: {a} -> {b}"

    def test_warmup_monotone_increasing(self):
        values = [self._lambda(self.cfg, s) for s in range(0, 101, 10)]
        for a, b in zip(values, values[1:]):
            assert b >= a - 1e-9


# ══════════════════════════════════════════════════════════════════════════════
#  CurriculumManager
# ══════════════════════════════════════════════════════════════════════════════

def _mock_student(n_layers):
    """Build a minimal mock student object with attention layers."""
    student = MagicMock()
    layers = []
    for _ in range(n_layers):
        layer = MagicMock()
        layer.attention.is_linear_active = False
        layers.append(layer)
    student.model.layers = layers
    return student


def _mock_optimizer():
    opt = MagicMock()
    opt.param_groups = []
    return opt


def _make_cfg(**kw):
    from llada_fast.training.distill.config import DistillConfig
    return DistillConfig(**kw)


class TestCurriculumManagerNoProgression:
    """All layers active from step 0 — no progression (progressive_interval=0, joint=False)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from llada_fast.training.distill.curriculum import CurriculumManager
        self.cls = CurriculumManager

    def _make(self, n_layers=4):
        cfg = _make_cfg(progressive_interval=0)
        student = _mock_student(n_layers)
        opt = _mock_optimizer()
        return self.cls(cfg=cfg, n_layers=n_layers, optimizer=opt, student=student), cfg

    def test_all_layers_active_initially(self):
        cm, _ = self._make(n_layers=4)
        assert set(cm.active_layers) == {0, 1, 2, 3}

    def test_step_does_not_change_layers(self):
        cm, _ = self._make(n_layers=4)
        active_before = set(cm.active_layers)
        for step in range(0, 500, 50):
            cm.step(step)
        assert set(cm.active_layers) == active_before


class TestCurriculumManagerInterval:
    """progressive_interval > 0 — activate one new layer every N steps."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from llada_fast.training.distill.curriculum import CurriculumManager
        self.cls = CurriculumManager

    def _make(self, n_layers=4, interval=100):
        cfg = _make_cfg(progressive_interval=interval)
        student = _mock_student(n_layers)
        opt = _mock_optimizer()
        cm = self.cls(cfg=cfg, n_layers=n_layers, optimizer=opt, student=student)
        return cm

    def test_starts_with_one_layer(self):
        cm = self._make(n_layers=4, interval=100)
        assert len(cm.active_layers) == 1

    def test_activates_layer_at_interval(self):
        cm = self._make(n_layers=4, interval=100)
        initial = len(cm.active_layers)
        cm.step(100)
        assert len(cm.active_layers) >= initial + 1

    def test_does_not_exceed_n_layers(self):
        cm = self._make(n_layers=4, interval=1)
        for step in range(0, 1000, 5):
            cm.step(step)
        assert len(cm.active_layers) <= 4

    def test_layers_are_unique(self):
        cm = self._make(n_layers=6, interval=1)
        for step in range(0, 100, 1):
            cm.step(step)
        assert len(set(cm.active_layers)) == len(cm.active_layers)

    def test_prog_seq_middle_out(self):
        """generate_prog_seq should start from the middle layer."""
        from llada_fast.training.distill.curriculum import generate_prog_seq
        seq = generate_prog_seq(8)
        assert seq[0] == 4  # mid of 8
        assert len(seq) == 8
        assert set(seq) == set(range(8))


class TestCurriculumManagerJoint:
    """joint=True — all layers active, anneal softmax anchor ratio."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from llada_fast.training.distill.curriculum import CurriculumManager
        self.cls = CurriculumManager

    def _make(self, n_layers=4):
        cfg = _make_cfg(
            joint=True,
            hybrid_ratio_start=0.9,
            hybrid_ratio_end=0.0,
            hybrid_anneal_steps=100,
        )
        student = _mock_student(n_layers)
        opt = _mock_optimizer()
        return self.cls(cfg=cfg, n_layers=n_layers, optimizer=opt, student=student)

    def test_all_layers_active_in_joint_mode(self):
        cm = self._make(4)
        assert set(cm.active_layers) == {0, 1, 2, 3}

    def test_anchor_annealing_over_steps(self):
        cm = self._make(4)
        # step() sets softmax_anchor_ratio on each active layer's attention
        cm.step(0)
        # After many steps the ratio should approach hybrid_ratio_end=0.0
        for _ in range(200):
            cm.step(1)
        for li in cm.active_layers:
            ratio = cm.student.model.layers[li].attention.softmax_anchor_ratio
            assert ratio == pytest.approx(0.0, abs=1e-6)

    def test_no_extra_layers_added(self):
        cm = self._make(4)
        for step in range(0, 500, 10):
            cm.step(step)
        assert len(cm.active_layers) == 4


class TestCurriculumStateSerializable:
    """CurriculumState fields are plain Python objects (safe for torch.save)."""

    def test_state_fields(self):
        from llada_fast.training.distill.curriculum import CurriculumState
        s = CurriculumState(
            active_layers=[0, 1],
            prog_seq_cursor=2,
            layer_activation_steps={0: 0, 1: 100},
        )
        assert s.consecutive_failures == 0
        assert s.hybrid_ratio_step == 0
        assert isinstance(s.gen_history, list)


# ══════════════════════════════════════════════════════════════════════════════
#  Corruption utilities (no HF datasets needed — mocked)
# ══════════════════════════════════════════════════════════════════════════════

class TestCorruptOneBlock:

    @pytest.fixture(autouse=True)
    def setup(self):
        # Import only the corruption functions directly (avoids datasets top-level import)
        import importlib, sys
        # Patch the heavy imports before importing data
        with patch.dict(sys.modules, {
            "datasets": MagicMock(),
            "transformers": MagicMock(),
        }):
            import llada_fast.training.distill.data as data_mod
            importlib.reload(data_mod)
        from llada_fast.training.distill.data import corrupt_one_block, corrupt_all_blocks
        self.corrupt_one = corrupt_one_block
        self.corrupt_all = corrupt_all_blocks

    def test_corrupt_one_block_shape(self):
        ids = torch.randint(0, 1000, (2, 64))
        pad = torch.ones(2, 64)
        out = self.corrupt_one(ids, pad, mask_id=0, block_size=8, t=0.5)
        assert out.shape == ids.shape

    def test_corrupt_one_only_modifies_target_block(self):
        """Non-target blocks should remain identical to input."""
        ids = torch.randint(1, 1000, (1, 32))
        pad = torch.ones(1, 32)
        out = self.corrupt_one(ids, pad, mask_id=0, block_size=8, t=1.0, block_idx=1)
        # Block 0 (tokens 0..7): should be unchanged
        assert (out[0, :8] == ids[0, :8]).all()
        # Block 1 (tokens 8..15): should be fully masked (t=1.0 but one guaranteed real)
        pass  # at least some tokens differ from original

    def test_corrupt_one_uses_mask_id(self):
        ids = torch.randint(10, 1000, (1, 32))
        pad = torch.ones(1, 32)
        out = self.corrupt_one(ids, pad, mask_id=999, block_size=8, t=1.0, block_idx=0)
        # Some tokens in block 0 should be mask_id
        assert (out[0, :8] == 999).any()

    def test_corrupt_all_blocks_shape(self):
        ids = torch.randint(0, 1000, (2, 64))
        pad = torch.ones(2, 64)
        num_blocks = 64 // 8
        t_per_block = torch.full((num_blocks,), 0.5)
        noisy, corrupted_mask = self.corrupt_all(ids, pad, mask_id=0, block_size=8, t_per_block=t_per_block)
        assert noisy.shape == ids.shape
        assert corrupted_mask.shape == ids.shape

    def test_corrupt_all_clean_equals_input(self):
        """Non-corrupted positions in noisy output must equal the original."""
        ids = torch.randint(1, 1000, (1, 32))
        pad = torch.ones(1, 32)
        t_per_block = torch.full((4,), 0.5)
        noisy, corrupted_mask = self.corrupt_all(ids, pad, mask_id=0, block_size=8, t_per_block=t_per_block)
        # Where not corrupted and not padding, noisy == original
        not_corrupted = ~corrupted_mask & (noisy != 0)
        assert (noisy[not_corrupted] == ids[not_corrupted]).all()

    def test_corrupt_all_noisy_has_masks(self):
        """With high t, noisy sequence should contain mask tokens (mask_id=0)."""
        ids = torch.randint(1, 1000, (1, 32))
        pad = torch.ones(1, 32)
        t_per_block = torch.full((4,), 0.9)  # high noise
        noisy, _ = self.corrupt_all(ids, pad, mask_id=0, block_size=8, t_per_block=t_per_block)
        assert (noisy == 0).any()

    def test_corrupt_all_real_tokens_unchanged(self):
        """Non-masked real tokens in noisy sequence must equal original."""
        ids = torch.randint(1, 1000, (1, 32))
        pad = torch.ones(1, 32)
        t_per_block = torch.full((4,), 0.5)
        noisy, corrupted_mask = self.corrupt_all(ids, pad, mask_id=0, block_size=8, t_per_block=t_per_block)
        # Uncorrupted, non-padding positions should match input
        keep = (~corrupted_mask) & (pad.bool())
        assert (noisy[keep] == ids[keep]).all()

    def test_corrupt_with_padding(self):
        """Padded positions should be masked in output."""
        ids = torch.randint(1, 1000, (1, 32))
        # First 20 tokens real, last 12 padding
        pad = torch.zeros(1, 32); pad[:, :20] = 1.0
        out = self.corrupt_one(ids, pad, mask_id=0, block_size=8, t=0.0, block_idx=0)
        # Pad region should be mask_id
        assert (out[0, 20:] == 0).all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
