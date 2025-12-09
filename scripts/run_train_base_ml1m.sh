#!/bin/bash
# Script to train Base T5 on ml-1m

cd "$(dirname "$0")"
export PYTHONPATH=$PYTHONPATH:.

DATASET="ml-1m"
MODEL="T5"
QUANT="rqvae"

echo "Starting Base Model Training on $DATASET ($MODEL)..."
python recommendation/main.py --model $MODEL --dataset $DATASET --quant_method $QUANT
