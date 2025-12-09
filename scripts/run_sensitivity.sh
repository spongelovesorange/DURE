#!/bin/bash
# Script to run Sensitivity Analysis for DURE

# Ensure we are in the root directory
cd "$(dirname "$0")"
export PYTHONPATH=$PYTHONPATH:.

DATASET="Digital_Music"
DATA_ROOT="/data/DURE/datasets"
BASE_CKPT="ckpt/recommendation/$DATASET/GPT2_rqvae/best_model.pth"

# Function to run experiment
run_exp() {
    SHARDING=$1
    ECL=$2
    DOL=$3
    NAME="shard${SHARDING}_ecl${ECL}_dol${DOL}"
    OUTPUT_DIR="ckpt/dure_sensitivity/$NAME"
    
    echo "========================================================"
    echo "Running Experiment: $NAME"
    echo "Sharding: $SHARDING | ECL: $ECL | DOL: $DOL"
    echo "========================================================"
    
    # Train
    python recommendation/dure_main.py \
        --dataset $DATASET \
        --data_root $DATA_ROOT \
        --output_dir $OUTPUT_DIR \
        --sharding_rate $SHARDING \
        --lambda_ecl $ECL \
        --lambda_dol $DOL \
        --epochs 5 # Use fewer epochs for speed, enough to see trend
        
    # Eval
    echo "Evaluating $NAME..."
    python recommendation/eval_dure.py \
        --dataset $DATASET \
        --base_ckpt $BASE_CKPT \
        --adapter_ckpt "$OUTPUT_DIR/adapter.pth"
}

# 1. Baseline (Current Best)
# run_exp 0.5 0.1 0.1

# 2. Ablation: No Sharding (Random Sharding = 0.0 means update all parameters? No, rate is "frozen" rate usually.
# In code: mask = bernoulli(1 - sharding_rate). 
# If rate=0.0, mask=1.0 (Update All).
# If rate=0.9, mask=0.1 (Update Few).
run_exp 0.0 0.1 0.1

# 3. Ablation: No Aux Losses (Only DPO)
run_exp 0.5 0.0 0.0

# 4. High Sharding (Very Sparse)
run_exp 0.9 0.1 0.1

