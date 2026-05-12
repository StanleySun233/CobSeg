1. pesudo seg: BERT NSP + TextTiling full-run, then select 100 dialogues
```bash
python scripts/build_pesudo_seg.py \
  --dataset data/dataset/vhf.json \
  --sample-size 100 \
  --output-dir data/dataset/vhf_pesudo_100_artifacts
```
this step first cuts the full train split, then samples 100 dialogues, and writes `pesudo_seg_sampled_train_100.json` in the original dialogue JSON shape with only `segments` updated.

2. pesudo label: OpenAI pesudo label for whole dialogues
```bash
python scripts/build_pesudo_label.py \
  --artifact-dir data/dataset/vhf_pesudo_100_artifacts \
  --openai-workers 16
```
this step takes one whole dialogue as input and outputs one dialogue-level JSON object with `utterances` and a `segments` array. Each segment carries `pesudo_label`, `pesudo_summary`, and the segment length.

3. pesudo utterance: OpenAI generation from whole-dialogue specs
```bash
python scripts/build_pesudo_utterance.py \
  --dataset data/dataset/vhf.json \
  --artifact-dir data/dataset/vhf_pesudo_100_artifacts \
  --label-file data/dataset/vhf_pesudo_100_artifacts/pesudo_label_dialogue.jsonl \
  --openai-workers 16 \
  --output data/dataset/vhf_pesudo_100.json
```
this step takes one whole dialogue's `pesudo_label + pesudo_summary + random length` segment specs and generates the full dialogue in one call. The generated output is validated against the sampled per-segment lengths.

4. output the pesudo train split
```bash
python scripts/build_pesudo_utterance.py \
  --dataset data/dataset/vhf.json \
  --artifact-dir data/dataset/vhf_pesudo_100_artifacts \
  --label-file data/dataset/vhf_pesudo_100_artifacts/pesudo_label_dialogue.jsonl \
  --output data/dataset/vhf_pesudo_100.json
```
