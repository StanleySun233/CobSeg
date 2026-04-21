python /home/sijin/maritime/dts/inference.py \
  --model_name dud \
  --dataset tiage \
  --encoder roberta-base \
  --exp_name cs_score

# Unfreeze the main RoBERTa/BERT branch so utterance hierarchy vectors are fine-tuned.
python /home/sijin/maritime/dts/inference.py \
  --model_name dud \
  --dataset tiage \
  --encoder roberta-base \
  --finetune_main_encoder \
  --main_encoder_lr 2e-5 \
  --exp_name cs_score_ft

# Two-stage DUD: stage1 warms up transition-aware RoBERTa representations,
# stage2 trains the full DUD decoder on top.
python /home/sijin/maritime/dts/inference.py \
  --model_name dud \
  --dataset tiage \
  --encoder roberta-base \
  --finetune_main_encoder \
  --main_encoder_lr 2e-5 \
  --two_stage_training \
  --stage1_epochs 5 \
  --stage1_lr 5e-4 \
  --stage1_main_encoder_lr 2e-5 \
  --stage1_aux_weight 0.5 \
  --exp_name cs_score_ft_stage2

# True RoBERTa NSP-style cross-encoder in stage1:
# adjacent utterance pairs are jointly encoded, then the learned pair embedding
# and boundary probability are injected back into stage2 DUD.
python /home/sijin/maritime/dts/inference.py \
  --model_name dud \
  --dataset tiage \
  --encoder roberta-base \
  --finetune_main_encoder \
  --main_encoder_lr 2e-5 \
  --two_stage_training \
  --use_nsp_cross_encoder \
  --stage1_epochs 5 \
  --stage1_lr 5e-4 \
  --stage1_main_encoder_lr 2e-5 \
  --nsp_stage2_aux_weight 0.2 \
  --exp_name cs_score_ft_nsp_stage2

# Also fine-tune the SIM/topic branch in stage1.
# Here stage1 updates both the main RoBERTa pair branch and the SimCSE/topic branch;
# stage2 keeps the topic encoder frozen unless --finetune_topic_encoder is added.
python /home/sijin/maritime/dts/inference.py \
  --model_name dud \
  --dataset tiage \
  --encoder roberta-base \
  --finetune_main_encoder \
  --main_encoder_lr 2e-5 \
  --two_stage_training \
  --use_nsp_cross_encoder \
  --stage1_epochs 5 \
  --stage1_lr 5e-4 \
  --stage1_main_encoder_lr 2e-5 \
  --stage1_finetune_topic_encoder \
  --stage1_topic_encoder_lr 2e-5 \
  --nsp_stage2_aux_weight 0.2 \
  --exp_name cs_score_ft_nsp_stage1_topic
