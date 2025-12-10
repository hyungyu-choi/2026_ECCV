#!/bin/bash

# ==============================================================================
# AutoLaparo Phase Recognition Evaluation Script
# ==============================================================================

# 1. CONFIGURATION
# ----------------
# Path to your LMDB dataset folder
LMDB_PATH="./datasets/AutoLaparo"

# Path to the JSON file containing validation labels
# (Make sure this JSON follows the structure expected by the code: dict with fold keys)
LABELS_JSON="./datasets/AutoLaparo.json"

# Directory containing your trained model checkpoints
# The script expects files named: fold_0_model_best.pt, fold_1_model_best.pt, ...
MODEL_DIR="your model"

# Evaluation settings
BATCH_SIZE=32

# 2. EXECUTION
# ----------------
echo "----------------------------------------------------------------"
echo "Starting Evaluation for AutoLaparo Phase Recognition"
echo "----------------------------------------------------------------"
echo "Data:   $LMDB_PATH"
echo "Labels: $LABELS_JSON"
echo "Models: $MODEL_DIR"
echo "----------------------------------------------------------------"

python ../downstream/test_phase_recognition_autolaparo.py \
    --lmdb "$LMDB_PATH" \
    --labels "$LABELS_JSON" \
    --models "$MODEL_DIR" \
    --bs "$BATCH_SIZE" \
