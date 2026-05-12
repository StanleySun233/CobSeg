#!/bin/bash
# CobSeg supervised training on VHF dataset

# Set PyTorch memory management to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python inference.py \
  --model_name cobseg \
  --dataset vhf \
  --exp_name cobseg_supervised_vhf
