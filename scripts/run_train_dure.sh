#!/bin/bash
# Script to train DURE Adapter for Unlearning

# Ensure we are in the root directory
cd "$(dirname "$0")"

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

DATASET="Digital_Music"
DATA_ROOT="/data/DURE/datasets"
OUTPUT_DIR="ckpt/dure/$DATASET"

echo "Starting DURE Training on $DATASET..."
python recommendation/dure_main.py \
    --dataset $DATASET \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR

echo "DURE Training complete. Adapter saved to $OUTPUT_DIR/adapter.pth"
