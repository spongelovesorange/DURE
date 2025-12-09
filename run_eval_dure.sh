#!/bin/bash
# Script to evaluate DURE Model (Unlearning Performance)

# Ensure we are in the root directory
cd "$(dirname "$0")"

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

DATASET="Digital_Music"
BASE_CKPT="ckpt/recommendation/$DATASET/GPT2_rqvae/best_model.pth"
ADAPTER_CKPT="ckpt/dure/$DATASET/adapter.pth"

echo "=== Evaluating DURE on Forget Set (Should be LOW/ZERO) ==="
# Note: eval_dure.py is hardcoded to use mask=1.0 for Forget Set
python recommendation/eval_dure.py \
    --dataset $DATASET \
    --base_ckpt $BASE_CKPT \
    --adapter_ckpt $ADAPTER_CKPT
