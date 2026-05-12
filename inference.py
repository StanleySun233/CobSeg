import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from model.base_model import BaseModel
from model.cobseg import CobSeg
from model.bert_dts import PureBertSegmenter
from utils.dialogue_dataset import DialogueDataset
from utils.dts_data import (
    EmbeddedDialogueDataset,
    MAX_UTT_TOKENS,
    MAX_UTTERANCES,
    collate_fn,
    collate_finetune_fn,
    topic_channel_sets_from_info,
)
from utils.dts_utils import run_checkpoint_dir
from utils.utils import resolve_dataset_path

CS_ENCODER_NAME = "princeton-nlp/sup-simcse-bert-base-uncased"



@dataclass
class ExperimentConfig:
    model_name: str = "cobseg"
    dataset: str = "vhf"
    exp_name: str = "baseline"
    encoder: str = "roberta-base"
    emb_batch: int = 16
    epochs: int = 50
    batch_size: int = 2
    seed: int = 42
    lr: float = BaseModel.default_lr
    lr_patience: int = BaseModel.default_lr_patience
    lr_factor: float = BaseModel.default_lr_factor
    min_lr: float = BaseModel.default_min_lr
    early_stop: int = BaseModel.default_early_stop
    num_samples: int = -1
    eval_only: bool = False
    stream_mode: str = "dual"
    disable_token_transformer: bool = False
    disable_crf: bool = False
    disable_ubiw: bool = False
    topic_json_path: str = "./data/topic/topic_keywords.json"
    rank_loss_weight: float = 0.2
    rank_margin: float = 0.1
    rank_kw_gap: float = 0.05
    end_loss_weight: float = 1.0
    start_loss_weight: float = 1.0
    edge_gate_alpha: float = 0.25
    edge_gate_gamma: float = 1.5
    ubiw_aux_weight: float = 0.2
    ubiw_aux_tau: float = 2.0
    coh_aux_weight: float = 0.2
    finetune_main_encoder: bool = True
    main_encoder_lr: float = 2e-5
    two_stage_training: bool = True
    stage1_epochs: int = 5
    stage1_lr: float = 5e-4
    stage1_main_encoder_lr: float = 2e-5
    stage1_aux_weight: float = 0.5
    use_nsp_cross_encoder: bool = True
    nsp_max_pair_tokens: int = 0
    nsp_stage2_aux_weight: float = 0.2
    train_subset_count: int = 0
    train_subset_seed: int = 42


_DATASET_CFG_CACHE: dict = {}
_DATASET_CONFIG_PATH = Path(__file__).resolve().parent / "data" / "dataset_config.yaml"


def _load_dataset_config(config_path: str) -> dict:
    global _DATASET_CFG_CACHE
    if config_path not in _DATASET_CFG_CACHE:
        p = Path(config_path)
        if p.exists():
            with open(p) as f:
                _DATASET_CFG_CACHE[config_path] = yaml.safe_load(f) or {}
        else:
            print(f"[warn] dataset_config not found: {config_path}  (using built-in defaults)")
            _DATASET_CFG_CACHE[config_path] = {}
    return _DATASET_CFG_CACHE[config_path]


def resolve_dataset_limits(dataset: str) -> tuple[int, int]:
    ds_stem = Path(resolve_dataset_path(dataset)).stem
    ds_cfg = _load_dataset_config(str(_DATASET_CONFIG_PATH)).get(ds_stem, {})
    return (
        int(ds_cfg.get("max_utt_tokens", MAX_UTT_TOKENS)),
        int(ds_cfg.get("max_utterances", MAX_UTTERANCES)),
    )


def compute_pos_weight(dataset: EmbeddedDialogueDataset) -> float:
    total_neg, total_pos = 0, 0
    for sample in dataset.samples:
        labels = sample[-2]
        pos = labels.sum().item()
        neg = len(labels) - pos
        total_pos += pos
        total_neg += neg
    if total_pos == 0:
        return 1.0
    return total_neg / total_pos


def build_model(cfg: ExperimentConfig, input_dim: int, max_utt_tokens: int) -> tuple[BaseModel, bool]:
    if cfg.model_name == "cobseg":
        model = CobSeg(
            input_dim=input_dim,
            max_utt_tokens=max_utt_tokens,
            stream_mode=cfg.stream_mode,
            use_token_transformer=not cfg.disable_token_transformer,
            use_crf=not cfg.disable_crf,
            use_ubiw=not cfg.disable_ubiw,
            edge_gate_alpha=cfg.edge_gate_alpha,
            edge_gate_gamma=cfg.edge_gate_gamma,
            topic_json_path=cfg.topic_json_path,
        )
        return model, not cfg.disable_crf

    if cfg.model_name == "bert":
        model = PureBertSegmenter(input_dim=input_dim)
        return model, False

    raise ValueError(f"Unsupported model_name: {cfg.model_name}")


def _load_hf_encoder(
    model_name: str,
    device: torch.device,
    *,
    eval_mode: bool = True,
):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        enc_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to(device)
        if eval_mode:
            enc_model.eval()
        return tokenizer, enc_model
    except OSError as exc:
        raise RuntimeError(
            f"Failed to load encoder '{model_name}'. "
            "The loader uses local cache when available and otherwise tries to "
            "download from Hugging Face."
        ) from exc


def _build_data_loader(
    data: EmbeddedDialogueDataset,
    *,
    batch_size: int,
    max_utterances: int,
    finetune_main_encoder: bool,
    shuffle: bool,
) -> DataLoader:
    collate_builder = collate_finetune_fn if finetune_main_encoder else collate_fn
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_builder(b, max_utterances=max_utterances),
    )


def _load_topic_channels(topic_json_path: str, dataset: str) -> dict[str, set[str]]:
    topic_channels_by_ds: dict[str, dict[str, set[str]]] = {}
    if os.path.exists(topic_json_path):
        with open(topic_json_path, "r", encoding="utf-8") as f:
            topic_data = json.load(f)
        topic_channels_by_ds = {
            ds: topic_channel_sets_from_info(info)
            for ds, info in topic_data.items()
        }
        print(f"Loaded topic keywords from {topic_json_path}")
    return topic_channels_by_ds.get(dataset, topic_channel_sets_from_info({}))


def _select_train_subset(train_dialogues: list, subset_count: int, subset_seed: int) -> list:
    if subset_count <= 0 or subset_count >= len(train_dialogues):
        return train_dialogues

    rng = np.random.default_rng(subset_seed)
    order = rng.permutation(len(train_dialogues))
    chosen = sorted(order[:subset_count].tolist())
    return [train_dialogues[idx] for idx in chosen]


def run_single(cfg: ExperimentConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if cfg.use_nsp_cross_encoder and not cfg.finetune_main_encoder:
        raise SystemExit(
            "--use_nsp_cross_encoder requires --finetune_main_encoder so the "
            "stage1 RoBERTa cross-encoder can be reused in stage-2."
        )
    if cfg.use_nsp_cross_encoder and cfg.model_name != "cobseg":
        raise SystemExit("--use_nsp_cross_encoder is currently only supported with --model_name cobseg.")

    max_utt_tokens, max_utterances = resolve_dataset_limits(cfg.dataset)
    nsp_max_pair_tokens = (
        int(cfg.nsp_max_pair_tokens)
        if cfg.nsp_max_pair_tokens > 0
        else min(max(max_utt_tokens * 2, max_utt_tokens + 8), 256)
    )
    print(
        f"Limits  max_utt_tokens={max_utt_tokens}  max_utterances={max_utterances}"
        f"  (data/dataset_config.yaml)"
    )
    if cfg.use_nsp_cross_encoder:
        print(f"NSP cross-encoder pair max tokens: {nsp_max_pair_tokens}")

    os.environ.setdefault("HF_HUB_READ_TIMEOUT", "120")

    ds_path = resolve_dataset_path(cfg.dataset)
    print(f"Loading dataset: {ds_path}")
    ckpt_dir = run_checkpoint_dir(__file__, ds_path, cfg.exp_name)
    print(f"Run checkpoint directory: {ckpt_dir}")

    full_dataset = DialogueDataset(ds_path)
    train_dialogues = [d for d in full_dataset if d.set == "train"]
    val_dialogues = [d for d in full_dataset if d.set in ("valid", "val", "dev")]
    test_dialogues = [d for d in full_dataset if d.set == "test"]
    print(f"Train: {len(train_dialogues)}  Valid: {len(val_dialogues)}  Test: {len(test_dialogues)}")

    original_train_count = len(train_dialogues)
    if cfg.train_subset_count > 0:
        if cfg.train_subset_count > original_train_count:
            raise SystemExit(
                f"--train_subset_count={cfg.train_subset_count} exceeds available "
                f"train dialogues ({original_train_count})."
            )
        train_dialogues = _select_train_subset(
            train_dialogues,
            subset_count=cfg.train_subset_count,
            subset_seed=cfg.train_subset_seed,
        )
        pct = 100.0 * len(train_dialogues) / max(original_train_count, 1)
        print(
            "Using supervised train subset: "
            f"{len(train_dialogues)}/{original_train_count} dialogues "
            f"({pct:.2f}% of train split, seed={cfg.train_subset_seed})"
        )

    print(f"Loading main encoder: {cfg.encoder}")
    tokenizer, enc_model = _load_hf_encoder(
        cfg.encoder,
        device,
        eval_mode=not cfg.finetune_main_encoder,
    )
    enc_model.requires_grad_(cfg.finetune_main_encoder)
    print(f"Loading CS encoder: {CS_ENCODER_NAME}")
    cs_tokenizer, cs_enc_model = _load_hf_encoder(CS_ENCODER_NAME, device, eval_mode=True)
    cs_enc_model.requires_grad_(False)

    topic_channels = _load_topic_channels(cfg.topic_json_path, cfg.dataset)

    dataset_kw = dict(
        cs_enc_model=cs_enc_model,
        cs_tokenizer=cs_tokenizer,
        batch_size=cfg.emb_batch,
        max_utterances=max_utterances,
        max_utt_tokens=max_utt_tokens,
        dataset_name=cfg.dataset,
        topic_channels=topic_channels,
        finetune_main_encoder=cfg.finetune_main_encoder,
        use_nsp_cross_encoder=cfg.use_nsp_cross_encoder,
        nsp_max_pair_tokens=nsp_max_pair_tokens,
    )

    train_data = EmbeddedDialogueDataset(
        train_dialogues,
        enc_model,
        tokenizer,
        device,
        **dataset_kw,
    )
    test_data = EmbeddedDialogueDataset(
        test_dialogues,
        enc_model,
        tokenizer,
        device,
        **dataset_kw,
    )

    if not cfg.eval_only and len(train_data.samples) == 0:
        raise SystemExit("No training dialogues after encoding.")

    ref_data = train_data if train_data.samples else test_data
    if not ref_data.samples:
        raise SystemExit("No dialogues after encoding.")

    input_dim = (
        int(enc_model.config.hidden_size)
        if cfg.finetune_main_encoder
        else ref_data.samples[0][0].shape[-1]
    )
    token_len = ref_data.samples[0][1].shape[1]
    if token_len != max_utt_tokens:
        raise ValueError(f"token length {token_len} != max_utt_tokens {max_utt_tokens}")

    train_loader = _build_data_loader(
        train_data,
        batch_size=cfg.batch_size,
        max_utterances=max_utterances,
        finetune_main_encoder=cfg.finetune_main_encoder,
        shuffle=True,
    )
    test_loader = _build_data_loader(
        test_data,
        batch_size=cfg.batch_size,
        max_utterances=max_utterances,
        finetune_main_encoder=cfg.finetune_main_encoder,
        shuffle=False,
    )

    val_loader = None
    if val_dialogues:
        val_data = EmbeddedDialogueDataset(
            val_dialogues,
            enc_model,
            tokenizer,
            device,
            **dataset_kw,
        )
        val_loader = _build_data_loader(
            val_data,
            batch_size=cfg.batch_size,
            max_utterances=max_utterances,
            finetune_main_encoder=cfg.finetune_main_encoder,
            shuffle=False,
        )

    model, use_crf = build_model(cfg, input_dim=input_dim, max_utt_tokens=max_utt_tokens)
    model = model.to(device)
    main_encoder = enc_model if cfg.finetune_main_encoder else None
    model.configure_runtime(
        main_encoder=main_encoder,
        use_crf=use_crf,
        device=device,
    )

    if hasattr(model, "set_class_balance"):
        pos_w = compute_pos_weight(train_data) if train_data.samples else 1.0
        model.set_class_balance(pos_w)
        print(f"Train token pos_weight (neg/pos): {pos_w:.4f}")

    total_params = model.count_trainable_parameters()
    if main_encoder is not None:
        seg_params = model.count_trainable_parameters(include_main_encoder=False)
        enc_params = model.count_trainable_parameters(include_model=False)
        print(
            f"Trainable parameters: total={total_params:,}  "
            f"segmenter={seg_params:,}  main_encoder={enc_params:,}"
        )
    else:
        print(f"Trainable parameters: {total_params:,}")

    model.run_experiment(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        ckpt_dir=ckpt_dir,
        test_dialogues=test_dialogues,
        max_utterances=max_utterances,
    )
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="cobseg", choices=["cobseg", "bert"])
    parser.add_argument("--dataset", default="vhf")
    parser.add_argument("--exp_name", default="baseline")
    parser.add_argument("--encoder", default="roberta-base")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--emb_batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = ExperimentConfig(
        model_name=args.model_name,
        dataset=args.dataset,
        exp_name=args.exp_name,
        encoder=args.encoder,
        batch_size=args.batch_size,
        emb_batch=args.emb_batch,
        epochs=args.epochs,
        seed=args.seed,
    )
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    run_single(cfg)


if __name__ == "__main__":
    main()
