#!/bin/bash
# Script to train the Base Model (GPT2) on the mixed dataset (Train + Forget)
# This simulates a "dirty" model that needs unlearning.

# Ensure we are in the root directory
cd "$(dirname "$0")"

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Dataset and Model Config
DATASET="Digital_Music"
MODEL="GPT2"
QUANT="rqvae"

echo "Starting Base Model Training on $DATASET..."
python recommendation/main.py --model $MODEL --dataset $DATASET --quant_method $QUANT

echo "Training complete. Checkpoints saved in ckpt/recommendation/$DATASET/${MODEL}_${QUANT}/"
