#!/bin/bash

# ==============================================================================
# k-NN Evaluation Script for Cholec80
# ==============================================================================

# Path to pretrained checkpoint
CHECKPOINT="pl_stitch/pl_vitbase16_cholec80_global/checkpoint0010.pth"

# Architecture settings
ARCH="vit_base"
PATCH_SIZE=16

# Data paths
TRAIN_DIR="../code/Dataset/cholec80/frames/extract_1fps/training_set"
TEST_DIR="../code/Dataset/cholec80/frames/extract_1fps/test_set"
TRAIN_LABEL_DIR="../code/Dataset/cholec80/phase_annotations/training_set_1fps"
TEST_LABEL_DIR="../code/Dataset/cholec80/phase_annotations/test_set_1fps"

# Evaluation settings
K=20
BATCH_SIZE=64

# Output directory
OUTPUT_DIR="./knn_results_cholec80"

echo "================================================================"
echo "k-NN Evaluation for Cholec80 Phase Recognition"
echo "================================================================"
echo "Checkpoint:   $CHECKPOINT"
echo "Architecture: $ARCH (patch_size=$PATCH_SIZE)"
echo "k:            $K"
echo "Batch size:   $BATCH_SIZE"
echo "================================================================"

python pl_stitch/eval_knn_cholec80.py \
    --checkpoint "$CHECKPOINT" \
    --arch "$ARCH" \
    --patch_size "$PATCH_SIZE" \
    --train_dir "$TRAIN_DIR" \
    --test_dir "$TEST_DIR" \
    --train_label_dir "$TRAIN_LABEL_DIR" \
    --test_label_dir "$TEST_LABEL_DIR" \
    --k "$K" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "================================================================"
echo "Evaluation complete! Results saved to $OUTPUT_DIR"
echo "================================================================"