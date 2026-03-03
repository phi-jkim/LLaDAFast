"""DistillConfig: single dataclass holding all stage-1 distillation hyperparameters."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DistillConfig:
    # ── Model / data ──────────────────────────────────────────────────────────
    teacher_model_path: str = "inclusionAI/LLaDA2.1-mini"
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: Optional[str] = "sample-10BT"
    seq_len: int = 1024

    # ── Training ──────────────────────────────────────────────────────────────
    num_steps: int = 15000
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
    lr_patience: int = 20       # Patience for ReduceLROnPlateau (measured in sequences)
    lr_factor: float = 0.1      # Factor for ReduceLROnPlateau
    min_lr: float = 1e-5        # Minimum LR for ReduceLROnPlateau

    # ── Architecture ──────────────────────────────────────────────────────────
    distill_layers: Optional[List[int]] = None      # which layers to supervise (default: all)
    linear_layers: Optional[List[int]] = None       # which layers to linearize (default: all)
    block_size_override: Optional[int] = None
    use_gated_deltanet: bool = False
    use_hybrid_local_softmax: bool = False
    use_block_softmax_hybrid: bool = False          # BlockSoftmaxLinearHybrid variant

    # ── Curriculum ────────────────────────────────────────────────────────────
    progressive_interval: int = 0           # steps between activating each new layer
    force_decay_length: int = 1000          # steps for teacher-forcing prob to decay to 0
    use_llm_curriculum_eval: bool = False   # use LLM judge for layer progression

    # ── Joint / Attention-Surgery mode ────────────────────────────────────────
    joint: bool = False
    hybrid_ratio_start: float = 0.5         # initial softmax anchor fraction
    hybrid_ratio_end: float = 0.0           # final softmax anchor fraction
    hybrid_anneal_steps: int = 5000

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

    # ── Attention visualization ────────────────────────────────────────────────
    plot_attn_every: int = 0           # 0 = disabled; N = save every N steps (from step 0)
    plot_attn_max_layers: int = 4      # how many active layers to visualize
    plot_attn_max_len: int = 128       # truncate sequence to this many tokens in plot
