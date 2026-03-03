"""
LoLCATs forward hooks for teacher collection and student teacher-forcing.

TeacherHooks  — captures ground-truth attention outputs from the frozen teacher.
StudentHooks  — records student attention outputs and (optionally) replaces them
                with the teacher's ground truth to prevent error compounding.
"""

import random
from typing import Dict, List, Sequence

import torch


def _get_attn_out(out) -> torch.Tensor:
    """Extract the primary attention output tensor from a module's return value."""
    return out[0] if isinstance(out, (tuple, list)) else out


class TeacherHooks:
    """
    Registers forward hooks on the teacher's attention modules to collect
    ground-truth attention outputs.

    Usage::

        hooks = TeacherHooks(teacher, layer_ids)
        with torch.no_grad():
            teacher(...)              # populates hooks.store
        targets = {li: t.to(dev1) for li, t in hooks.store.items()}
        hooks.clear()                 # ready for next step
        hooks.remove()                # cleanup at end of training
    """

    def __init__(self, model, layer_ids: Sequence[int]):
        self.store: Dict[int, torch.Tensor] = {}
        self._handles: List = []
        for li in layer_ids:
            def _hook(module, inp, out, _li=li):
                self.store[_li] = _get_attn_out(out).detach()
            h = model.model.layers[li].attention.register_forward_hook(_hook)
            self._handles.append(h)

    def clear(self) -> None:
        self.store.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


class StudentHooks:
    """
    Registers forward hooks on the student's attention modules to:
      1. Record the student's raw attention output (used for MSE loss).
      2. Replace the output with the teacher's ground truth (teacher forcing).

    For BlockSoftmaxLinearHybrid layers, `module._raw_linear_out` is read
    instead of the blended output so that MSE directly trains the Hedgehog
    kernel weights (not the already-anchored blended signal).

    Teacher forcing is stochastic: each layer has a force probability in
    `p_force_dict` that decays from 1.0 → 0.0 over training.

    Usage::

        student_hooks = StudentHooks(student, layer_ids, p_force_dict)
        student_hooks.teacher_targets.update({li: t for li, t in teacher_targets})
        student(...)           # hooks fire; student_hooks.record is populated
        student_hooks.clear()  # ready for next step
        student_hooks.remove() # cleanup at end of training
    """

    def __init__(
        self,
        model,
        layer_ids: Sequence[int],
        p_force_dict: Dict[int, float],
    ):
        self.record: Dict[int, torch.Tensor] = {}
        self.teacher_targets: Dict[int, torch.Tensor] = {}
        self.p_force_dict = p_force_dict
        self._handles: List = []

        for li in layer_ids:
            def _hook(module, inp, out, _li=li):
                if not module.training:
                    return out

                # Prefer the pure linear output (set on the attention module by
                # LLaDA2MoeAttention.forward after the linear/hybrid computation).
                raw_linear = getattr(module, "_raw_linear_out", None)
                raw_out = raw_linear if raw_linear is not None else _get_attn_out(out)
                self.record[_li] = raw_out

                # Teacher forcing: replace student output with teacher ground truth.
                force_prob = self.p_force_dict.get(_li, 1.0)
                if _li in self.teacher_targets and random.random() < force_prob:
                    t_val = self.teacher_targets[_li].to(
                        device=raw_out.device, dtype=raw_out.dtype
                    )
                    if isinstance(out, (tuple, list)):
                        return (t_val,) + tuple(out)[1:]
                    return t_val

                return out

            h = model.model.layers[li].attention.register_forward_hook(_hook)
            self._handles.append(h)

    def clear(self) -> None:
        self.record.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
