#!/bin/bash
set -e # Exit on error

echo "========================================================"
echo "STEP 1: Training Base Model (Dirty)"
echo "========================================================"
./run_train_base.sh

echo "========================================================"
echo "STEP 2: Evaluating Base Model"
echo "========================================================"
./run_eval_base.sh

echo "========================================================"
echo "STEP 3: Training DURE Adapter (Unlearning)"
echo "========================================================"
./run_train_dure.sh

echo "========================================================"
echo "STEP 4: Evaluating DURE Model"
echo "========================================================"
./run_eval_dure.sh
