#!/bin/bash
# End-to-end test for LLaDAFast Distillation
# Verifies: Full-layer activation, sequence triggers, and checkpoint rotation.

# 1. Clean up old test runs
rm -rf ./distilled_test
mkdir -p ./distilled_test

# 2. Add src to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "Starting E2E test (3 sequences, 100 steps total)..."

/home/jinhakim/.conda/envs/llada_fast/bin/python3 -m llada_fast.training.distill.run \
    --teacher_model inclusionAI/LLaDA2.1-mini \
    --use_block_softmax_hybrid \
    --lr 0.01 \
    --progressive_interval 0 \
    --force_decay_length 0 \
    --steps 100 \
    --save_every 1 \
    --plot_attn_every 1 \
    --output_dir ./distilled_test \
    --device_teacher cuda:0 \
    --device_student cuda:1

echo "--------------------------------------------------"
echo "Verifying outputs in ./distilled_test:"
ls -d ./distilled_test/step_*
ls ./distilled_test/attn_plots/
echo "--------------------------------------------------"
