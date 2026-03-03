"""
Progressive layer-activation curriculum for LoLCATs distillation.

CurriculumManager handles:
  - Which layers are currently active (being distilled).
  - Layer activation ordering (outward-from-middle by default).
  - Teacher-forcing probability decay per layer.
  - LLM-judge integration for quality-gated progression.
  - Joint mode: softmax anchor ratio annealing.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import torch

# Default order: start from the middle of a 24-layer model, expand outward.
DEFAULT_PROG_SEQ = [
    12, 13, 11, 14, 10, 15, 9, 16, 8, 17, 7, 18, 6, 19, 5, 20, 4, 21, 3, 22, 2, 23, 1, 0
]


@dataclass
class CurriculumState:
    """Serialisable snapshot of curriculum progress (saved/loaded with train_state.pt)."""
    active_layers: List[int]
    prog_seq_cursor: int
    layer_activation_steps: Dict[int, int]
    p_force_dict: Dict[int, float] = field(default_factory=dict)
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
      use_llm_curriculum_eval — quality-gated progression via LLM judge.
      (none)               — all target layers active from step 0, no progression.

    The `revert_layer_fn` callback is called when a layer is skipped (SKIP verdict).
    It should revert that layer's weights to the teacher's and freeze its parameters.
    """

    def __init__(
        self,
        cfg,                            # DistillConfig
        n_layers: int,
        optimizer: torch.optim.Optimizer,
        student,
        revert_layer_fn: Callable[[int], None],
        prog_seq: List[int] = DEFAULT_PROG_SEQ,
    ):
        self._hybrid_mode = getattr(cfg, "use_block_softmax_hybrid", False)
        self.cfg = cfg
        self.n_layers = n_layers
        self.optimizer = optimizer
        self.student = student
        self.revert_layer_fn = revert_layer_fn
        self.prog_seq = prog_seq
        self.state = self._build_initial_state()
        self._llm_verdict = "FAIL"

    # ── Public API ──────────────────────────────────────────────────────────

    def record_verdict(self, verdict: str) -> None:
        """Call after each eval step with the LLM judge verdict (or 'FAIL' if no eval)."""
        self._llm_verdict = verdict

    def step(self, current_step: int) -> None:
        """Advance curriculum state by one training step."""
        if self.cfg.joint:
            self._step_joint(current_step)
        elif self.cfg.use_llm_curriculum_eval:
            self._step_llm(current_step)
        elif self.cfg.progressive_interval > 0:
            self._step_interval(current_step)
        # Always recompute forcing probabilities.
        self._update_p_force(current_step)

    @property
    def active_layers(self) -> List[int]:
        return self.state.active_layers

    @property
    def p_force_dict(self) -> Dict[int, float]:
        return self.state.p_force_dict

    # ── Internal helpers ────────────────────────────────────────────────────

    def _build_initial_state(self) -> CurriculumState:
        cfg = self.cfg
        if cfg.joint or (not cfg.progressive_interval and not cfg.use_llm_curriculum_eval):
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

    def _step_llm(self, step: int) -> None:
        cfg = self.cfg
        expected = len(self.state.active_layers)
        if cfg.eval_every > 0 and step % cfg.eval_every == 0 and step > 0:
            v = self._llm_verdict
            if v == "PASS":
                expected += 1
                self.state.consecutive_failures = 0
                self.state.gen_history.clear()
                print(f"\n[CURRICULUM] Step {step}: layer passed LLM judge, activating next.")
            elif v == "SKIP" or self.state.consecutive_failures >= 12:
                reason = "SKIP verdict" if v == "SKIP" else "12 consecutive failures"
                skipped = self.state.active_layers.pop()
                print(f"\n[CURRICULUM] Step {step}: layer {skipped} skipped ({reason}), reverting.")
                self.revert_layer_fn(skipped)
                expected = len(self.state.active_layers) + 1
                self.state.consecutive_failures = 0
                self.state.gen_history.clear()
            else:
                self.state.consecutive_failures += 1
                print(f"\n[CURRICULUM] Layer {self.state.active_layers[-1]} "
                      f"failed ({self.state.consecutive_failures}/12).")
        self._activate_up_to(expected, step)

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
                self.optimizer.add_param_group({"params": new_params, "lr": 5e-5})
            self.state.already_in_optimizer.add(next_layer)
            print(f"\n[CURRICULUM] Step {step}: layer {next_layer} activated for linear attention!")

    def _update_p_force(self, step: int) -> None:
        """Recompute teacher-forcing probability for every active layer."""
        decay = self.cfg.force_decay_length
        is_joint = self.cfg.joint
        for li in self.state.active_layers:
            if decay > 0:
                age = step - self.state.layer_activation_steps.get(li, 0)
                self.state.p_force_dict[li] = max(0.0, 1.0 - age / decay)
            else:
                # joint mode: no forcing (rely on softmax anchor); otherwise full forcing
                self.state.p_force_dict[li] = 0.0 if is_joint else 1.0
