# LLaDAFast

LLaDAFast distills [LLaDA 2.1-mini](https://huggingface.co/inclusionAI/LLaDA2.1-mini) into a faster model using **kernel linear attention** and **hybrid block attention**, trained with a LoLCATs-style teacher-forcing curriculum.

## Method

### Stage 1 — Attention Distillation

A frozen teacher (softmax attention) supervises a student (linear/hybrid attention) layer-by-layer via:

- **Teacher Forcing**: student attention outputs are replaced with the teacher's exact activations during the forward pass, preventing error compounding across layers. Forcing probability decays 1.0 → 0.0 per layer over `--force_decay_length` steps.
- **M2T (Mask-to-Token)**: present block tokens are randomly masked with `[MASK]`; trains denoising.
- **T2T (Token-to-Token)**: present block tokens are replaced with random vocab tokens; trains editing/self-correction.
- **All-blocks training**: every 32-token block in a sequence is used for gradient updates (not just one random block per sequence).
- **Progressive curriculum**: layers are activated one at a time in middle-out order (layer 12 → 13 → 11 → 14 …), each for `--progressive_interval` steps.

### Stage 2 — LoRA Recovery

Restores generation quality via LoRA fine-tuning on UltraChat with the LLaDA denoising objective.

---

## Repository Structure

```
LLaDAFast/
├── src/llada_fast/
│   ├── modeling/
│   │   ├── linear_attention.py          # OrderInvariantKernelLinearAttention (Hedgehog)
│   │   ├── hybrid_attention.py          # BlockSoftmaxLinearHybrid (past=linear, current=softmax)
│   │   ├── bidirectional_gated_deltanet.py
│   │   └── modeling_llada2_moe.py       # LLaDA2MoeModelLM (teacher + student)
│   └── training/
│       ├── distill/
│       │   ├── run.py                   # Stage-1 training loop (entrypoint)
│       │   ├── config.py                # DistillConfig dataclass
│       │   ├── data.py                  # corrupt_one_block, corrupt_one_block_t2t, StreamingTextLoader
│       │   ├── hooks.py                 # TeacherHooks, StudentHooks (forward hooks)
│       │   ├── curriculum.py            # Progressive layer activation curriculum
│       │   ├── attn_viz.py              # Attention map visualization (PNG grid)
│       │   └── llm_judge.py             # LLM-judge quality-gated curriculum (optional)
│       └── lora/
│           └── train.py                 # Stage-2 LoRA fine-tuning
└── scripts/eval/
    └── perplexity.py
```

---

## Installation

```bash
pip install -e .
pip install matplotlib  # for attention visualization
```

---

## Stage 1: Distillation

**Requirements**: 2 GPUs (teacher on `cuda:0`, student on `cuda:1`).

### Pure linear attention (progressive, no LLM judge)
```bash
python -m llada_fast.training.distill.run \
  --teacher_model inclusionAI/LLaDA2.1-mini \
  --progressive_interval 500 \
  --force_decay_length 1000 \
  --steps 15000 \
  --lr 2e-5 \
  --output_dir ./distilled_linear
```

### Hybrid attention (past blocks = linear, current block = softmax)
```bash
python -m llada_fast.training.distill.run \
  --teacher_model inclusionAI/LLaDA2.1-mini \
  --use_block_softmax_hybrid \
  --progressive_interval 500 \
  --force_decay_length 1000 \
  --steps 15000 \
  --lr 2e-5 \
  --omega_mask 0.5 \
  --omega_edit 0.5 \
  --output_dir ./distilled_hybrid \
  --plot_attn_every 200 \
  --plot_attn_max_layers 20 \
  --plot_attn_max_len 128
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--use_block_softmax_hybrid` | off | Hybrid: softmax within current block, linear over past blocks |
| `--progressive_interval N` | 0 | Activate one new layer every N steps (0 = all layers at once) |
| `--force_decay_length N` | 1000 | Steps for teacher-forcing probability to decay 1.0 → 0.0 |
| `--omega_mask` | 0.5 | Weight of M2T (masking) loss objective |
| `--omega_edit` | 0.5 | Weight of T2T (random-token editing) loss objective |
| `--alpha` | 1.0 | MSE hidden-state loss weight |
| `--beta` | 1.0 | KL logit distillation loss weight |
| `--plot_attn_every N` | 0 | Save attention heatmaps every N steps (0 = off) |
| `--resume_from PATH` | — | Resume from a saved checkpoint directory |

Checkpoints are saved to `{output_dir}/step_{N}/` every `--save_every` steps.
Attention maps (PNG) are saved to `{output_dir}/attn_plots/`.

---

## Stage 2: LoRA Recovery

```bash
python -m llada_fast.training.lora.train --config configs/lora_step2.yaml
```

---

## Acknowledgments

Based on [LLaDA 2.1](https://github.com/inclusionAI/LLaDA2.X) by InclusionAI, with distillation design inspired by [LoLCATs](https://github.com/HazyResearch/lolcats) (Hedgehog feature map + teacher-forcing curriculum).
