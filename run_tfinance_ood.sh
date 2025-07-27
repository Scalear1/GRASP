#!/bin/bash

# Set your logits file path here
LOGITS_PATH="path/to/your/logits.pt"

# Run different OOD detection methods
echo "Running OOD detection on TFinance dataset..."

# GRASP method
echo "Testing GRASP..."
python test_ood_tfinance.py \
    --logits_path $LOGITS_PATH \
    --ood GRASP \
    --device 0 \
    --runs 5 \
    --K 8 \
    --alpha 0.1 \
    --delta 0.1 \
    --tau1 10 \
    --tau2 50 \
    --st top \
    --ood_budget 0.1

# MSP baseline
echo "Testing MSP..."
python test_ood_tfinance.py \
    --logits_path $LOGITS_PATH \
    --ood MSP \
    --device 0 \
    --runs 5 \
    --ood_budget 0.1

# Energy baseline
echo "Testing Energy..."
python test_ood_tfinance.py \
    --logits_path $LOGITS_PATH \
    --ood Energy \
    --device 0 \
    --runs 5 \
    --T 1.0 \
    --ood_budget 0.1

# KNN baseline
echo "Testing KNN..."
python test_ood_tfinance.py \
    --logits_path $LOGITS_PATH \
    --ood KNN \
    --device 0 \
    --runs 5 \
    --ood_budget 0.1

echo "All experiments completed!"