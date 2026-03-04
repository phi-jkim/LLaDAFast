"""
Progressive layer-activation curriculum for LoLCATs distillation.

CurriculumManager handles:
  - Which layers are currently active (being distilled).
  - Layer activation ordering (outward-from-middle by default).
  - Teacher-forcing probability decay per layer.
  - Joint mode: softmax anchor ratio annealing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set

import torch

def generate_prog_seq(n_layers: int) -> List[int]:
    """Generates a sequence starting from the middle and expanding outwards."""
    mid = n_layers // 2
    seq = [mid]
    for i in range(1, max(mid + 1, n_layers - mid)):
        # Try higher first, then lower
        hi = mid + i
        lo = mid - i
        if hi < n_layers:
            seq.append(hi)
        if lo >= 0:
            seq.append(lo)
    return seq


# Default order: dynamically generated based on n_layers in CurriculumManager.
DEFAULT_PROG_SEQ: List[int] = []


@dataclass
class CurriculumState:
    """Serialisable snapshot of curriculum progress (saved/loaded with train_state.pt)."""
    active_layers: List[int]
    prog_seq_cursor: int
    layer_activation_steps: Dict[int, int]
    consecutive_failures: int = 0
    gen_history: List[Dict] = field(default_factory=list)
    hybrid_ratio_step: int = 0
    already_in_optimizer: Set[int] = field(default_factory=set)


class CurriculumManager:
    """
    Drives the LoLCATs progressive distillation curriculum.

    Modes:
      joint                — all layers active from step 0; anneal softmax anchor ratio.
      progressive_interval — activate one new layer every N steps.
      (none)               — all target layers active from step 0, no progression.
    """

    def __init__(
        self,
        cfg,                            # DistillConfig
        n_layers: int,
        optimizer: torch.optim.Optimizer,
        student,
        prog_seq: List[int] = DEFAULT_PROG_SEQ,
    ):
        self._hybrid_mode = getattr(cfg, "use_block_softmax_hybrid", False)
        self.cfg = cfg
        self.n_layers = n_layers
        self.optimizer = optimizer
        self.student = student
        if not prog_seq:
            self.prog_seq = generate_prog_seq(n_layers)
        else:
            self.prog_seq = prog_seq
        self.state = self._build_initial_state()

    # ── Public API ──────────────────────────────────────────────────────────

    def step(self, current_step: int) -> None:
        """Advance curriculum state by one training step."""
        if self.cfg.joint:
            self._step_joint(current_step)
        elif self.cfg.progressive_interval > 0:
            self._step_interval(current_step)

    @property
    def active_layers(self) -> List[int]:
        return self.state.active_layers

    @property
    def p_force_dict(self) -> Dict[int, float]:
        """Teacher forcing is always on (p=1.0) for all active layers."""
        return {li: 1.0 for li in self.state.active_layers}

    # ── Internal helpers ────────────────────────────────────────────────────

    def _build_initial_state(self) -> CurriculumState:
        cfg = self.cfg
        if cfg.joint or not cfg.progressive_interval:
            active = cfg.linear_layers if cfg.linear_layers is not None else list(range(self.n_layers))
            return CurriculumState(
                active_layers=list(active),
                prog_seq_cursor=0,
                layer_activation_steps={li: 0 for li in active},
                already_in_optimizer=set(active),
            )
        else:
            first = self.prog_seq[0]
            return CurriculumState(
                active_layers=[first],
                prog_seq_cursor=1,
                layer_activation_steps={first: 0},
                already_in_optimizer={first},
            )

    def _step_joint(self, step: int) -> None:
        cfg = self.cfg
        if cfg.hybrid_anneal_steps > 0:
            t = min(1.0, self.state.hybrid_ratio_step / cfg.hybrid_anneal_steps)
            anchor = cfg.hybrid_ratio_start + t * (cfg.hybrid_ratio_end - cfg.hybrid_ratio_start)
        else:
            anchor = cfg.hybrid_ratio_end
        for li in self.state.active_layers:
            self.student.model.layers[li].attention.softmax_anchor_ratio = anchor
        self.state.hybrid_ratio_step += 1


    def _step_interval(self, step: int) -> None:
        expected = 1 + step // self.cfg.progressive_interval
        self._activate_up_to(expected, step)

    def _activate_up_to(self, target_count: int, step: int) -> None:
        target_count = min(target_count, self.n_layers)
        while len(self.state.active_layers) < target_count:
            cursor = self.state.prog_seq_cursor
            if cursor >= len(self.prog_seq):
                break
            next_layer = self.prog_seq[cursor]
            self.state.prog_seq_cursor += 1
            self.state.active_layers.append(next_layer)
            self.state.layer_activation_steps[next_layer] = step
            self.student.model.layers[next_layer].attention.is_linear_active = True

            attn = self.student.model.layers[next_layer].attention
            if self._hybrid_mode and hasattr(attn, "linear_attention"):
                # Hybrid mode: only train the kernel feature map + alpha.
                new_params = list(attn.linear_attention.parameters())
            else:
                new_params = list(attn.parameters())
            for p in new_params:
                p.requires_grad = True

            if new_params and next_layer not in self.state.already_in_optimizer:
                self.optimizer.add_param_group({"params": new_params, "lr": self.cfg.learning_rate})
            self.state.already_in_optimizer.add(next_layer)
            print(f"\n[CURRICULUM] Step {step}: layer {next_layer} activated for linear attention!")
