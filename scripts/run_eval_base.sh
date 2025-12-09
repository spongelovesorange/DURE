#!/bin/bash
# Script to evaluate the Base Model on Forget Set and Retain Set

# Ensure we are in the root directory
cd "$(dirname "$0")"

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

DATASET="Digital_Music"
MODEL="GPT2"
QUANT="rqvae"
CKPT_PATH="ckpt/recommendation/$DATASET/${MODEL}_${QUANT}/best_model.pth"

echo "=== Evaluating Base Model on Forget Set (Should be HIGH if dirty) ==="
python recommendation/eval_base.py \
    --model $MODEL \
    --dataset $DATASET \
    --quant_method $QUANT \
    --ckpt_path $CKPT_PATH \
    --test_json datasets/$DATASET/$DATASET.forget.jsonl

echo ""
echo "=== Evaluating Base Model on Retain Set (Baseline Performance) ==="
python recommendation/eval_base.py \
    --model $MODEL \
    --dataset $DATASET \
    --quant_method $QUANT \
    --ckpt_path $CKPT_PATH \
    --test_json datasets/$DATASET/$DATASET.test.jsonl
