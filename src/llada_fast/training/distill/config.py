"""DistillConfig: single dataclass holding all stage-1 distillation hyperparameters."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DistillConfig:
    # ── Model / data ──────────────────────────────────────────────────────────
    teacher_model_path: str = "inclusionAI/LLaDA2.1-mini"
    dataset_name: str = "yahma/alpaca-cleaned"
    dataset_subset: Optional[str] = None
    seq_len: int = 1024

    # ── Training ──────────────────────────────────────────────────────────────
    num_steps: int = 15000
    gradient_checkpointing: bool = False    # recompute activations during backward (saves ~60% activation memory)
    learning_rate: float = 3e-3
    batch_size: int = 1
    grad_accum_steps: int = 1
    device_teacher: str = "cuda:0"
    device_student: str = "cuda:1"
    alpha: float = 1.0          # MSE hidden-state loss weight
    beta: float = 1.0           # KL logit distillation loss weight
    temperature: float = 1.0    # KL temperature
    omega_mask: float = 0.5     # M2T objective weight within a block step
    omega_edit: float = 0.5     # T2T objective weight within a block step
    weight_decay: float = 0.0   # Weight decay for AdamW
    warmup_steps: int = 200     # Linear warmup steps before cosine decay begins
    min_lr: float = 1e-5        # Cosine decay floor LR

    # ── Architecture ──────────────────────────────────────────────────────────
    distill_layers: Optional[List[int]] = None      # which layers to supervise (default: all)
    linear_layers: Optional[List[int]] = None       # which layers to linearize (default: all)
    block_size_override: Optional[int] = None
    use_gated_deltanet: bool = False
    use_hybrid_local_softmax: bool = False
    use_block_softmax_hybrid: bool = False          # BlockSoftmaxLinearHybrid variant

    # ── Curriculum ────────────────────────────────────────────────────────────
    progressive_interval: int = 0           # steps between activating each new layer

    # ── Joint / Attention-Surgery mode ────────────────────────────────────────
    joint: bool = False
    hybrid_ratio_start: float = 0.5         # initial softmax anchor fraction
    hybrid_ratio_end: float = 0.0           # final softmax anchor fraction
    hybrid_anneal_steps: int = 5000

    # ── Test set (reserved) ───────────────────────────────────────────────────
    test_size: int = 256            # examples to reserve from the stream head
    test_eval_batches: int = 8      # mini-batches to average per test evaluation

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_every: int = 100
    eval_prompts: List[str] = field(
        default_factory=lambda: ["Write a story about a fast cat."]
    )
    eval_gen_len: int = 128
    eval_block_length: int = 32
    eval_steps: int = 32
    eval_threshold: float = 0.95
    eval_editing_threshold: float = 0.9
    eval_max_post_steps: int = 16

    # ── Checkpointing ─────────────────────────────────────────────────────────
    save_every: int = 2000
    output_dir: str = "./student_forcing_final"
    resume_from: str = ""


