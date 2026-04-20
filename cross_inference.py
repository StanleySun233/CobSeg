import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from inference import ExperimentConfig, build_model, predict, resolve_dataset_limits
from utils.dialogue_dataset import DialogueDataset
from utils.dts_data import collate_fn, topic_channel_sets_from_info, EmbeddedDialogueDataset
from utils.dts_utils import (
    dialogues_used_for_stream,
    evaluate_all,
    print_metrics,
    save_sample_predictions,
)
from utils.utils import resolve_dataset_path


@dataclass
class CrossInferenceConfig:
    ckpt_dataset: str
    dataset: str
    ckpt_path: str
    out_dir: str
    encoder: str
    emb_batch: int
    batch_size: int
    seed: int
    num_samples: int
    eval_val: bool
    model_name: str
    exp_name: str
    stream_mode: str
    disable_token_transformer: bool
    disable_crf: bool
    disable_ubiw: bool
    topic_json_path: str
    edge_gate_alpha: float
    edge_gate_gamma: float


def default_ckpt_path(ckpt_dataset: str, exp_name: str) -> Path:
    root = Path(__file__).resolve().parent / "checkpoints"
    stem = Path(resolve_dataset_path(ckpt_dataset)).stem
    return root / stem / exp_name / "best.pt"


def default_out_dir(ckpt_dataset: str, exp_name: str, target_dataset: str) -> Path:
    root = Path(__file__).resolve().parent / "checkpoints" / "cross_eval"
    ck_stem = Path(resolve_dataset_path(ckpt_dataset)).stem
    tg_stem = Path(resolve_dataset_path(target_dataset)).stem
    return root / f"{ck_stem}__{exp_name}__on__{tg_stem}"


def run_cross_eval(cfg: CrossInferenceConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    max_tok_ckpt, max_u_ckpt = resolve_dataset_limits(cfg.ckpt_dataset)
    max_tok_tgt, max_u_tgt = resolve_dataset_limits(cfg.dataset)
    if max_tok_tgt > max_tok_ckpt:
        raise SystemExit(
            f"Target max_utt_tokens ({max_tok_tgt}) > checkpoint dataset ({cfg.ckpt_dataset}) "
            f"max_utt_tokens ({max_tok_ckpt}). Use a checkpoint trained with >= target token budget, "
            f"or trim target limits in data/dataset_config.yaml for this experiment."
        )

    print(
        f"Checkpoint layout: max_utt_tokens={max_tok_ckpt} max_utterances={max_u_ckpt}  ({cfg.ckpt_dataset})"
    )
    print(
        f"Target embedding: max_utt_tokens={max_tok_tgt} max_utterances={max_u_tgt}  ({cfg.dataset})"
    )

    ckpt_path = Path(cfg.ckpt_path)
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    os.environ.setdefault("HF_HUB_READ_TIMEOUT", "120")

    tgt_path = resolve_dataset_path(cfg.dataset)
    print(f"Target dataset: {tgt_path}")

    full_dataset = DialogueDataset(tgt_path)
    val_dialogues = [d for d in full_dataset if d.set in ("valid", "val", "dev")]
    test_dialogues = [d for d in full_dataset if d.set == "test"]
    print(f"Target valid: {len(val_dialogues)}  test: {len(test_dialogues)}")

    print(f"Loading HF model: {cfg.encoder}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder, trust_remote_code=True, local_files_only=True)
    enc_model = AutoModel.from_pretrained(cfg.encoder, trust_remote_code=True, local_files_only=True).to(device).eval()

    topic_channels_by_ds: dict[str, dict[str, set[str]]] = {}
    if os.path.exists(cfg.topic_json_path):
        with open(cfg.topic_json_path, "r", encoding="utf-8") as f:
            topic_data = json.load(f)
        topic_channels_by_ds = {
            ds: topic_channel_sets_from_info(info) for ds, info in topic_data.items()
        }
        print(f"Loaded topic keywords from {cfg.topic_json_path}")
    topic_channels = topic_channels_by_ds.get(cfg.dataset, topic_channel_sets_from_info({}))

    def make_embedded(dialogues: list):
        return EmbeddedDialogueDataset(
            dialogues,
            enc_model,
            tokenizer,
            device,
            batch_size=cfg.emb_batch,
            max_utterances=max_u_tgt,
            max_utt_tokens=max_tok_tgt,
            dataset_name=cfg.dataset,
            topic_channels=topic_channels,
        )

    test_data = make_embedded(test_dialogues)
    if not test_data.samples:
        raise SystemExit("No test dialogues after encoding.")

    val_loader = None
    if cfg.eval_val and val_dialogues:
        val_data = make_embedded(val_dialogues)
        if val_data.samples:
            val_loader = DataLoader(
                val_data,
                batch_size=cfg.batch_size,
                shuffle=False,
                collate_fn=lambda b: collate_fn(b, max_utterances=max_u_tgt),
            )

    test_loader = DataLoader(
        test_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, max_utterances=max_u_tgt),
    )

    input_dim = test_data.samples[0][0].shape[-1]
    token_len = test_data.samples[0][1].shape[1]
    if token_len != max_tok_tgt:
        raise ValueError(f"token length {token_len} != max_tok_tgt {max_tok_tgt}")

    exp_cfg = ExperimentConfig(
        model_name=cfg.model_name,
        dataset=cfg.dataset,
        encoder=cfg.encoder,
        emb_batch=cfg.emb_batch,
        epochs=0,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        lr=0.0,
        lr_patience=0,
        lr_factor=0.0,
        min_lr=0.0,
        early_stop=0,
        num_samples=cfg.num_samples,
        eval_only=True,
        exp_name=cfg.exp_name,
        stream_mode=cfg.stream_mode,
        disable_token_transformer=cfg.disable_token_transformer,
        disable_crf=cfg.disable_crf,
        disable_ubiw=cfg.disable_ubiw,
        topic_json_path=cfg.topic_json_path,
        rank_loss_weight=0.0,
        rank_margin=0.0,
        rank_kw_gap=0.0,
        end_loss_weight=0.0,
        start_loss_weight=0.0,
        edge_gate_alpha=cfg.edge_gate_alpha,
        edge_gate_gamma=cfg.edge_gate_gamma,
        ubiw_aux_weight=0.0,
        ubiw_aux_tau=2.0,
    )

    model, use_crf = build_model(exp_cfg, input_dim=input_dim, max_utt_tokens=max_tok_ckpt)
    model = model.to(device)

    print(f"Loading checkpoint: {ckpt_path}")
    load_res = model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    if load_res.missing_keys:
        print(f"[warn] missing_keys ({len(load_res.missing_keys)}): first 8 -> {load_res.missing_keys[:8]}")
    if load_res.unexpected_keys:
        print(f"[warn] unexpected_keys ({len(load_res.unexpected_keys)}): first 8 -> {load_res.unexpected_keys[:8]}")
    if hasattr(model, "sync_start_heads_from_end"):
        if any(k.startswith("head_s_start") or k.startswith("head_w_start") for k in load_res.missing_keys):
            model.sync_start_heads_from_end()

    metrics_val = None
    if val_loader is not None:
        preds_v, labels_v = predict(model, val_loader, device, use_crf=use_crf)
        metrics_val = evaluate_all(preds_v, labels_v)
        print_metrics(metrics_val, prefix="Val (cross)")

    preds, labels = predict(model, test_loader, device, use_crf=use_crf)
    metrics_test = evaluate_all(preds, labels)
    print_metrics(metrics_test, prefix="Test (cross)")

    record = {
        "ckpt_dataset": cfg.ckpt_dataset,
        "target_dataset": cfg.dataset,
        "ckpt_path": str(ckpt_path),
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
        "config": asdict(cfg),
        "max_utt_tokens_ckpt": max_tok_ckpt,
        "max_utt_tokens_target_embed": max_tok_tgt,
    }
    results_path = out_dir / "results_cross.json"
    with open(results_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Results saved to {results_path}")

    save_sample_predictions(
        dialogues_used_for_stream(test_dialogues, max_u_tgt),
        preds,
        labels,
        out_path=out_dir / "sample_predictions_cross.csv",
        n=cfg.num_samples,
        seed=cfg.seed,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint trained on one dataset on another (generalization)."
    )
    parser.add_argument(
        "--ckpt_dataset",
        default="doc2dial",
        help="Dataset name used when training (determines default ckpt path and model max_utt_tokens).",
    )
    parser.add_argument(
        "--dataset",
        default="vhf",
        help="Target dataset to evaluate on (embedding limits from dataset_config.yaml).",
    )
    parser.add_argument(
        "--ckpt_path",
        default="",
        help="Explicit path to best.pt. If empty, uses checkpoints/<ckpt_dataset_stem>/<exp_name>/best.pt",
    )
    parser.add_argument(
        "--out_dir",
        default="",
        help="Directory for results_cross.json and sample CSV. Default: checkpoints/cross_eval/<ck>__<exp>__on__<tgt>/",
    )
    parser.add_argument("--exp_name", type=str, default="topic_kw_v1")
    parser.add_argument("--encoder", default="BAAI/bge-m3")
    parser.add_argument("--emb_batch", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--no_val", action="store_true", help="Skip validation split metrics.")
    parser.add_argument("--model_name", default="dud", choices=("dud", "bert_bilstm", "bert"))
    parser.add_argument("--stream_mode", choices=("dual", "sentence", "token"), default="dual")
    parser.add_argument("--disable_token_transformer", action="store_true")
    parser.add_argument("--disable_crf", action="store_true")
    parser.add_argument("--disable_ubiw", action="store_true")
    parser.add_argument("--topic_json_path", type=str, default="./data/topic/topic_keywords.json")
    parser.add_argument("--edge_gate_alpha", type=float, default=0.25)
    parser.add_argument("--edge_gate_gamma", type=float, default=1.5)

    args = parser.parse_args()

    ckpt_path = args.ckpt_path.strip() or str(default_ckpt_path(args.ckpt_dataset, args.exp_name))
    out_dir = args.out_dir.strip() or str(default_out_dir(args.ckpt_dataset, args.exp_name, args.dataset))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = CrossInferenceConfig(
        ckpt_dataset=args.ckpt_dataset,
        dataset=args.dataset,
        ckpt_path=ckpt_path,
        out_dir=out_dir,
        encoder=args.encoder,
        emb_batch=args.emb_batch,
        batch_size=args.batch_size,
        seed=args.seed,
        num_samples=args.num_samples,
        eval_val=not args.no_val,
        model_name=args.model_name,
        exp_name=args.exp_name,
        stream_mode=args.stream_mode,
        disable_token_transformer=args.disable_token_transformer,
        disable_crf=args.disable_crf,
        disable_ubiw=args.disable_ubiw,
        topic_json_path=args.topic_json_path,
        edge_gate_alpha=args.edge_gate_alpha,
        edge_gate_gamma=args.edge_gate_gamma,
    )
    run_cross_eval(cfg)


if __name__ == "__main__":
    main()
