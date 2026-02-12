#!/bin/bash
# ==============================================================================
# Convert Cholec80 training frames to LMDB for PL-Stitch pretraining
# ==============================================================================

# Path to Cholec80 training frames (relative to this script's location in scripts/)
FRAMES_ROOT="../code/Dataset/cholec80/frames/extract_1fps/training_set"

# Output LMDB path (will be created)
LMDB_OUTPUT="./datasets/cholec80_pretrain.lmdb"

mkdir -p "$(dirname "$LMDB_OUTPUT")"

echo "Converting Cholec80 frames to LMDB..."
echo "  Source: $FRAMES_ROOT"
echo "  Output: $LMDB_OUTPUT"

python create_cholec80_lmdb.py \
    --frames-root "$FRAMES_ROOT" \
    --lmdb-path "$LMDB_OUTPUT" \
    --image-size 224 \
    --map-size 5e11

echo "Done!"