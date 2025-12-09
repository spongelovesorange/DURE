#!/bin/bash
# Script to run full DURE experiment on ml-1m (T5)

# Ensure we are in the root directory
cd "$(dirname "$0")"

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

DATASET="ml-1m"
MODEL="T5"
DATA_ROOT="/data/DURE/datasets"
OUTPUT_DIR="ckpt/dure/$DATASET"

echo "=== 1. Training DURE on $DATASET ($MODEL) ==="
python recommendation/dure_main_v2.py \
    --dataset $DATASET \
    --model_type $MODEL \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --sharding_rate 0.5 \
    --lambda_ecl 0.1 \
    --lambda_dol 0.1 \
    --epochs 20

echo "=== 2. Evaluating DURE on $DATASET ==="
BASE_CKPT="ckpt/recommendation/$DATASET/${MODEL}_rqvae/best_model.pth"
ADAPTER_CKPT="$OUTPUT_DIR/adapter.pth"

python recommendation/eval_dure_v2.py \
    --dataset $DATASET \
    --model_type $MODEL \
    --base_ckpt $BASE_CKPT \
    --adapter_ckpt $ADAPTER_CKPT

echo "=== Full Experiment Complete ==="
