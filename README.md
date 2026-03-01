# LLaDAFast

LLaDAFast is an optimized implementation of [LLaDA 2.1-mini](https://huggingface.co/inclusionAI/LLaDA2.1-mini), featuring **Fully Linear Attention** and **Block-Bidirectional** processing.

## Highlights
- **Linear Efficiency**: Replaces standard softmax attention with an order-invariant kernel linear attention.
- **Frozen-Teacher Distillation**: Learns from the original LLaDA 2.1-mini using a truly frozen teacher strategy on FineWeb-Edu.
- **LoRA Recovery**: Restores performance via LoRA post-training on UltraChat with the LLaDA denoising/diffusion objective.

## Repository Structure
```text
LLaDAFast/
├── src/
│   └── llada_fast/
│       ├── modeling/      # Optimized linear attention and model definitions
│       ├── data/          # Dataset loading and packing utilities
│       ├── training/      # Distillation and LoRA training loops
│       └── utils/         # Helper functions and configs
├── configs/               # Training and model configurations
├── scripts/               # Entrypoint scripts for training/eval
├── tests/                 # Unit and integration tests
├── docs/                  # Design documents and benchmarks
└── README.md
```

## Getting Started

### Installation
```bash
conda env create -f environment.yml
conda activate llada_fast
```

### Distillation (Step 1)
**Requirements**: This script requires **2 GPUs** (80GB+ each). The Frozen Teacher is loaded on `cuda:0` and the Student on `cuda:1`.

Run the progressive layer-swapping curriculum:
```bash
python src/llada_fast/training/distill.py \
    --steps 15000 \
    --lr 5e-5 \
    --progressive_interval 300 \
    --force_decay_length 1500
```

### LoRA Recovery (Step 2)
```bash
python scripts/train_lora.py --config configs/lora_step2.yaml
```

## Acknowledgments
Based on the [LLaDA 2.1](https://github.com/inclusionAI/LLaDA2.X) framework by the InclusionAI team.
