"""
NSP-based TextTiling benchmark for dialogue topic segmentation.

Pairwise approach: computes NSP probabilities between adjacent utterances,
then applies TextTiling (similarity → depth → threshold → boundaries).

Supports:
  - Unsupervised (zero-shot): use pretrained BERT NSP head directly.
  - Supervised: fine-tune AutoModelForSequenceClassification on sentence pairs.
  - SC mode: bi-encoder cosine similarity (contrastive learning).

Usage:
  python -m model.nsp_texttiling --dataset dialseg711 --encoder bert-base-uncased --epochs 0
  python -m model.nsp_texttiling --dataset vhf --encoder bert-base-uncased --epochs 10
  python -m model.nsp_texttiling --dataset doc2dial --encoder BAAI/bge-m3 --mode SC --epochs 0
"""
import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForNextSentencePrediction,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from model.base_model import BaseModel
from utils.dialogue_dataset import DialogueDataset
from utils.dts_utils import (
    evaluate_all,
    print_metrics,
    save_sample_predictions,
    segments_to_boundaries,
)
from utils.tet import (
    alpha_search,
    depth_computing,
    similarity_computing,
    texttiling_segment,
)
from utils.utils import resolve_dataset_path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class NSPExperimentConfig:
    dataset: str
    encoder: str
    mode: str
    epochs: int
    batch_size: int
    lr: float
    seed: int
    max_length: int
    alpha_lower: float
    alpha_upper: float
    alpha_step: float
    num_samples: int
    exp_name: str
    eval_only: bool


# ---------------------------------------------------------------------------
# Dataset: sentence pairs for NSP training
# ---------------------------------------------------------------------------

class SentencePairDataset(Dataset):
    """(sent1, sent2) tokenized as a cross-encoder input + label."""

    def __init__(self, pairs: list[tuple[str, str, int]], tokenizer, max_length: int = 128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        s1, s2, label = self.pairs[idx]
        enc = self.tokenizer(
            s1, s2,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Pair construction from Utterance objects
# ---------------------------------------------------------------------------

def build_pairs(dialogues: list) -> list[tuple[str, str, int]]:
    """Construct (sent_a, sent_b, label) pairs from Utterance objects.

    label=1 if both utterances are in the same segment, 0 otherwise.
    """
    pairs: list[tuple[str, str, int]] = []
    for dial in dialogues:
        utts = dial.utterances
        segs = dial.segments
        seg_ids: list[int] = []
        for seg_idx, seg_len in enumerate(segs):
            seg_ids.extend([seg_idx] * seg_len)
        for i in range(len(utts) - 1):
            if i + 1 < len(seg_ids):
                label = 1 if seg_ids[i] == seg_ids[i + 1] else 0
            else:
                label = 1
            pairs.append((utts[i], utts[i + 1], label))
    return pairs


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class NSPTextTilingModel(BaseModel):
    """Wraps a HuggingFace NSP / SequenceClassification / AutoModel."""

    default_lr = 2e-5
    default_lr_patience = 3
    default_early_stop = 5

    def __init__(self, model_name_or_path: str, mode: str = "NSP", max_length: int = 128):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.mode = mode
        self.max_length = max_length
        self.use_nsp_head = False

        if mode == "NSP":
            self.model, self.use_nsp_head = self._load_nsp_model(model_name_or_path)
        elif mode == "SC":
            self.model = AutoModel.from_pretrained(model_name_or_path)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    @staticmethod
    def _load_nsp_model(name: str) -> tuple[nn.Module, bool]:
        config = AutoConfig.from_pretrained(name)
        archs = config.architectures or []
        if any("SequenceClassification" in a for a in archs):
            return AutoModelForSequenceClassification.from_pretrained(name), False
        try:
            return AutoModelForNextSentencePrediction.from_pretrained(name), True
        except (ValueError, OSError):
            return (
                AutoModelForSequenceClassification.from_pretrained(name, num_labels=2),
                False,
            )

    def to_device(self):
        self.model = self.model.to(self.device)
        return self


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_nsp(
    model: NSPTextTilingModel,
    train_dialogues: list,
    dev_dialogues: list,
    test_dialogues: list,
    device: torch.device,
    ckpt_dir: Path,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 128,
    alpha_lower: float = -2.0,
    alpha_upper: float = 2.0,
    alpha_step: float = 0.1,
) -> dict:
    """Fine-tune NSP model with epoch-wise evaluation on dev/test.

    Returns:
        dict with keys: best_epoch, best_alpha, best_dev_score, history
    """
    print("[NSP] Building training pairs...")
    pairs = build_pairs(train_dialogues)
    pos = sum(1 for _, _, l in pairs if l == 1)
    neg = len(pairs) - pos
    print(f"[NSP] Total pairs: {len(pairs)}  (pos={pos}, neg={neg})")

    if model.mode == "NSP":
        train_model = AutoModelForSequenceClassification.from_pretrained(
            model.model_name_or_path, num_labels=2
        ).to(device)
    else:
        train_model = model.model.to(device)

    dataset = SentencePairDataset(pairs, model.tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(train_model.parameters(), lr=lr)

    # Fallback: if no dev, use test for alpha search (but still report test separately)
    search_dialogues = dev_dialogues if dev_dialogues else test_dialogues
    if not dev_dialogues:
        print("[WARN] No dev set, using test for alpha search (not recommended)")

    best_epoch = 0
    best_alpha = 0.0
    best_dev_score = float("inf")  # minimize PK
    history = []

    for epoch in range(1, epochs + 1):
        # --- Training ---
        train_model.train()
        total_loss = 0.0
        for batch_enc, labels in tqdm(loader, desc=f"Epoch {epoch}/{epochs} train"):
            batch_enc = {k: v.to(device) for k, v in batch_enc.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            out = train_model(**batch_enc, labels=labels)
            out.loss.backward()
            optimizer.step()
            total_loss += out.loss.item()
        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch}/{epochs}] train_loss={avg_loss:.4f}")

        # --- Evaluation on dev ---
        train_model.eval()
        # Temporarily update model wrapper
        model.model = train_model
        model.use_nsp_head = False

        # Alpha search on dev (or test if no dev)
        alpha, dev_pk = alpha_search(
            search_dialogues, model.tokenizer, model.model, device,
            use_nsp_head=model.use_nsp_head, mode=model.mode,
            batch_size=batch_size,
            lower=alpha_lower, upper=alpha_upper, step=alpha_step,
        )
        print(f"[Epoch {epoch}] alpha={alpha:.2f}  dev_PK={dev_pk:.4f}")

        # Full dev metrics
        metrics_dev, _, _ = evaluate_nsp_texttiling(
            model, search_dialogues, alpha, device, batch_size=batch_size,
        )
        print_metrics(metrics_dev, prefix=f"Epoch {epoch} Dev")

        # Test metrics (always evaluate on test, even if we used it for alpha search)
        metrics_test, _, _ = evaluate_nsp_texttiling(
            model, test_dialogues, alpha, device, batch_size=batch_size,
        )
        print_metrics(metrics_test, prefix=f"Epoch {epoch} Test")

        # Track history
        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "alpha": alpha,
            "dev": metrics_dev,
            "test": metrics_test,
        })

        # Save checkpoint if best dev PK
        if dev_pk < best_dev_score:
            best_dev_score = dev_pk
            best_epoch = epoch
            best_alpha = alpha
            save_dir = ckpt_dir / "best"
            save_dir.mkdir(parents=True, exist_ok=True)
            train_model.save_pretrained(save_dir)
            model.tokenizer.save_pretrained(save_dir)
            # Save metrics alongside model
            metrics_path = save_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump({
                    "epoch": epoch,
                    "alpha": alpha,
                    "dev": metrics_dev,
                    "test": metrics_test,
                }, f, indent=2)
            print(f"[Epoch {epoch}] New best! Saved to {save_dir}")

        # Save latest checkpoint
        latest_dir = ckpt_dir / "last"
        latest_dir.mkdir(parents=True, exist_ok=True)
        train_model.save_pretrained(latest_dir)
        model.tokenizer.save_pretrained(latest_dir)
        # Save latest metrics
        latest_metrics_path = latest_dir / "metrics.json"
        with open(latest_metrics_path, "w") as f:
            json.dump({
                "epoch": epoch,
                "alpha": alpha,
                "dev": metrics_dev,
                "test": metrics_test,
            }, f, indent=2)

    # Load best model
    print(f"\n[Training done] Best epoch: {best_epoch}  best_dev_PK: {best_dev_score:.4f}")
    best_model_dir = ckpt_dir / "best"
    if best_model_dir.exists():
        print(f"Loading best model from {best_model_dir}")
        if model.mode == "NSP":
            model.model = AutoModelForSequenceClassification.from_pretrained(best_model_dir).to(device)
        else:
            model.model = AutoModel.from_pretrained(best_model_dir).to(device)
        model.use_nsp_head = False
        model.model.eval()

    return {
        "best_epoch": best_epoch,
        "best_alpha": best_alpha,
        "best_dev_score": best_dev_score,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_nsp_texttiling(
    model: NSPTextTilingModel,
    dialogues: list,
    alpha: float,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[dict, list[list[int]], list[list[int]]]:
    """Evaluate NSP TextTiling on a set of dialogues.

    Returns:
        (metrics_dict, all_preds, all_labels)
    """
    model.model.eval()
    all_preds: list[list[int]] = []
    all_labels: list[list[int]] = []

    for dial in tqdm(dialogues, desc="NSP eval"):
        utts = dial.utterances
        n = len(utts)
        if n < 2:
            gt = segments_to_boundaries(dial.segments)[:n]
            all_preds.append([0] * n)
            all_labels.append(gt)
            continue

        sim = similarity_computing(
            utts, model.tokenizer, model.model, device,
            batch_size=batch_size,
            use_nsp_head=model.use_nsp_head,
            mode=model.mode,
        )
        depth = depth_computing(sim)
        pred = texttiling_segment(depth, n, alpha)
        gt = segments_to_boundaries(dial.segments)[:n]
        all_preds.append(pred)
        all_labels.append(gt)

    metrics = evaluate_all(all_preds, all_labels)
    return metrics, all_preds, all_labels


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def _ckpt_dir(ds_path: str, encoder_name: str, exp_name: str) -> Path:
    """Checkpoint directory: checkpoints/bert-finetune/{dataset}/{encoder}/{exp_name}/"""
    ds_stem = Path(ds_path).stem
    encoder_stem = encoder_name.replace("/", "_")  # e.g. BAAI/bge-m3 -> BAAI_bge-m3
    return (
        Path(__file__).resolve().parent.parent
        / "checkpoints"
        / "bert-finetune"
        / ds_stem
        / encoder_stem
        / exp_name
    )


def run_nsp_texttiling(cfg: NSPExperimentConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    ds_path = resolve_dataset_path(cfg.dataset)
    print(f"Dataset: {ds_path}")

    full_dataset = DialogueDataset(ds_path)
    train_dialogues = [d for d in full_dataset if d.set == "train"]
    dev_dialogues = [d for d in full_dataset if d.set in ("valid", "val", "dev")]
    test_dialogues = [d for d in full_dataset if d.set == "test"]
    print(f"Train: {len(train_dialogues)}  Dev: {len(dev_dialogues)}  Test: {len(test_dialogues)}")

    ckpt_dir = _ckpt_dir(ds_path, cfg.encoder, cfg.exp_name)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint dir: {ckpt_dir}")

    # --- Build model ---
    print(f"Loading model: {cfg.encoder}  mode={cfg.mode}")
    model = NSPTextTilingModel(cfg.encoder, mode=cfg.mode, max_length=cfg.max_length)
    model.model = model.model.to(device)

    # --- Supervised fine-tuning with epoch-wise evaluation ---
    train_info = None
    if not cfg.eval_only and cfg.epochs > 0:
        if not train_dialogues:
            raise SystemExit("No training dialogues found.")
        train_info = train_nsp(
            model, train_dialogues, dev_dialogues, test_dialogues, device, ckpt_dir,
            epochs=cfg.epochs, batch_size=cfg.batch_size,
            lr=cfg.lr, max_length=cfg.max_length,
            alpha_lower=cfg.alpha_lower, alpha_upper=cfg.alpha_upper, alpha_step=cfg.alpha_step,
        )
        best_alpha = train_info["best_alpha"]
        print(f"\n[Final] Using best model from epoch {train_info['best_epoch']}, alpha={best_alpha:.2f}")
    else:
        # --- Unsupervised: alpha search on dev ---
        if not dev_dialogues:
            print("[WARN] No dev set found, using test data for alpha search.")
            dev_dialogues = test_dialogues

        best_alpha, best_pk = alpha_search(
            dev_dialogues, model.tokenizer, model.model, device,
            use_nsp_head=model.use_nsp_head, mode=model.mode,
            batch_size=cfg.batch_size,
            lower=cfg.alpha_lower, upper=cfg.alpha_upper, step=cfg.alpha_step,
        )
        print(f"[NSP] Best alpha={best_alpha:.2f}  dev PK={best_pk:.4f}")

    # --- Final evaluation on dev and test with best alpha ---
    search_dialogues = dev_dialogues if dev_dialogues else test_dialogues
    metrics_dev, _, _ = evaluate_nsp_texttiling(
        model, search_dialogues, best_alpha, device, batch_size=cfg.batch_size,
    )
    print_metrics(metrics_dev, prefix="Final Dev")

    metrics_test, preds_test, labels_test = evaluate_nsp_texttiling(
        model, test_dialogues, best_alpha, device, batch_size=cfg.batch_size,
    )
    print_metrics(metrics_test, prefix="Final Test")

    # --- Save results ---
    results = {
        "config": asdict(cfg),
        "best_alpha": best_alpha,
        "metrics_dev": metrics_dev,
        "metrics_test": metrics_test,
    }
    if train_info:
        results["training"] = {
            "best_epoch": train_info["best_epoch"],
            "best_dev_score": train_info["best_dev_score"],
            "history": train_info["history"],
        }
    results_path = ckpt_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    save_sample_predictions(
        test_dialogues,
        preds_test,
        labels_test,
        out_path=ckpt_dir / "sample_predictions.csv",
        n=cfg.num_samples,
        seed=cfg.seed,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NSP TextTiling benchmark for DTS")
    parser.add_argument("--dataset", default="dialseg711",
                        help="vhf | dialseg711 | doc2dial | tiage | superseg, or path to .json")
    parser.add_argument("--encoder", default="bert-base-uncased",
                        help="HuggingFace model name or path")
    parser.add_argument("--mode", default="NSP", choices=("NSP", "SC"),
                        help="NSP (cross-encoder) or SC (bi-encoder cosine)")
    parser.add_argument("--epochs", type=int, default=0,
                        help="0 = unsupervised (zero-shot), >0 = supervised fine-tuning")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--alpha_lower", type=float, default=-2.0)
    parser.add_argument("--alpha_upper", type=float, default=2.0)
    parser.add_argument("--alpha_step", type=float, default=0.1)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--exp_name", default="nsp_texttiling")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    cfg = NSPExperimentConfig(**vars(args))
    run_nsp_texttiling(cfg)


if __name__ == "__main__":
    main()
