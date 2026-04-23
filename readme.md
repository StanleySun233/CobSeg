```bash
python inference.py \
  --model_name dud \
  --dataset superseg \
  --exp_name dud_nsp_twostage
```

固定训练配置已经写死在 `inference.py`：

- `encoder=roberta-base`
- `finetune_main_encoder=True`
- `two_stage_training=True`
- `use_nsp_cross_encoder=True`
- `stage1_epochs=5`
- `stage1_lr=5e-4`
- `stage1_main_encoder_lr=2e-5`
- `nsp_stage2_aux_weight=0.2`

也就是说：

1. 第一阶段默认训练 RoBERTa NSP cross-encoder
2. 第一阶段最佳 checkpoint 默认作为第二阶段初始化
3. 第二阶段默认训练完整 DUD 下游分割

监督数据比例实验可以用单独脚本跑：

```bash
python scripts/run_supervision_sweep.py \
  --dataset superseg \
  --baseline_json checkpoints/superseg/nsp_texttiling/results.json \
  --exp_prefix sup_vs_unsup \
  --model_name dud \
  --encoder roberta-base \
  --epochs 50
```

说明：

- 脚本按“先划分，再在 train split 内抽样”解释比例
- 默认比例为 `1%, 3%, 5%, 10%, 25%, 50%, 75%`
- `train/valid/test` 切分不变，只对子采样 `train`
- 默认比较 `test.Score` 是否达到 `--baseline_json` 中的无监督结果
- 汇总结果默认写到 `data/experiments/<dataset>_<exp_prefix>.csv`
