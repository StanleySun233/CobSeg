python /home/sijin/maritime/dts/inference.py \
  --model_name bert_bilstm \
  --dataset tiage \
  --encoder BAAI/bge-m3 \
  --topic_json_path ./data/topic/topic_keywords.json \
  --rank_loss_weight 0.1 \
  --rank_margin 0.1 \
  --rank_kw_gap 0.05 \
  --exp_name kw_rank_v2