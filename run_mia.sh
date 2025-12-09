#!/bin/bash
# Script to run MIA Attack on DURE Model

# Ensure we are in the root directory
cd "$(dirname "$0")"

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

DATASET="Digital_Music"
BASE_CKPT="ckpt/recommendation/$DATASET/GPT2_rqvae/best_model.pth"
ADAPTER_CKPT="ckpt/dure/$DATASET/adapter.pth"
FORGET_FILE="datasets/$DATASET/$DATASET.forget.jsonl"
TEST_FILE="datasets/$DATASET/$DATASET.test.jsonl"

echo "=== Running MIA Attack Experiment ==="
python recommendation/mia_attack.py \
    --dataset $DATASET \
    --base_ckpt $BASE_CKPT \
    --adapter_ckpt $ADAPTER_CKPT \
    --forget_file $FORGET_FILE \
    --test_file $TEST_FILE
