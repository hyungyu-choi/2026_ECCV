#!/usr/bin/env bash
# ==============================================================================
# PL-Stitch pretraining on Cholec80 with BOTH local + global temporal PL losses
# ==============================================================================

cd pl_stitch

DATA_PATH="../datasets/cholec80_pretrain.lmdb"

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run \
  --nproc_per_node=2 \
  --nnodes=1 \
  --master_port="${MASTER_PORT:-29502}" \
  main_pl_global.py \
      --arch vit_base \
      --output_dir pl_vitbase16_cholec80_global \
      --data_path "$DATA_PATH" \
      --epochs 30 \
      --warmup_epochs 3 \
      --lambda_puzzle 0.4 \
      --batch_size_per_gpu 16 \
      --batch_size_temporal_per_gpu 16 \
      --batch_size_global_per_gpu 16 \
      --saveckp_freq 1 \
      --lr 0.0004 \
      --lr_head 0.0004 \
      --global_crops_scale 0.14 1 \
      --local_crops_scale 0.05 0.25 \
      --lambda_video 1.0 \
      --lambda_puzzle 0.4 \
      --lambda_temporal_global 1.0 \
      --start_epoch_global 0 \
      --momentum_teacher 0.998