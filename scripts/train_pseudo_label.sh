#!/bin/bash
# CobSeg training with pseudo labels on VHF dataset

# Set PyTorch memory management to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python inference.py \
  --model_name cobseg \
  --dataset vhf_pseudo_50 \
  --exp_name cobseg_pseudolabel_vhf
