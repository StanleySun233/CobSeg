# CobSeg: Coherence Boundary Modeling for Dialogue Topic Segmentation

EMNLP 26' CobSeg Implementation Details.

Code for paper: *CobSeg: Coherence Boundary Modeling for Dialogue Topic Segmentation*

## Quick Start
### Environment Setup
conda & pytorch

Note: this is only valid in: ubuntu22.04 with cuda 12.8 environment. You can modify if there is some environment conflict.

```bash
conda create -n cobseg python=3.11 -y
conda activate cobseg
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### Supervised Training
```bash
bash scripts/train_supervised.sh
```

### Pseudo-Label Training
```bash
# 1. view pseudo_label's readme.

# 2. Train with pseudo labels
bash scripts/train_pseudo_label.sh
```

## Requirements
```bash
pip install -r requirements.txt
```
