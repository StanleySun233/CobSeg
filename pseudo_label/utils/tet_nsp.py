from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForNextSentencePrediction, AutoModelForSequenceClassification, AutoTokenizer


def load_nsp_model(model_name: str, device: torch.device) -> tuple[AutoTokenizer, torch.nn.Module, bool]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    archs = config.architectures or []
    if any("SequenceClassification" in arch for arch in archs):
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        use_nsp_head = False
    else:
        try:
            model = AutoModelForNextSentencePrediction.from_pretrained(model_name).to(device)
            use_nsp_head = True
        except (OSError, ValueError):
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
            use_nsp_head = False
    model.eval()
    return tokenizer, model, use_nsp_head


def similarity_computing(
    texts: list[str],
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
    use_nsp_head: bool = False,
) -> list[float]:
    model.eval()
    if len(texts) < 2:
        return []
    pairs = [(texts[i], texts[i + 1]) for i in range(len(texts) - 1)]
    all_scores: list[float] = []
    prob_idx = 0 if use_nsp_head else 1
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            sents1, sents2 = zip(*batch)
            tokenized = tokenizer(
                list(sents1),
                list(sents2),
                padding=True,
                max_length=128,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**tokenized).logits
            probs = torch.softmax(logits, dim=1)
            all_scores.extend(probs[:, prob_idx].detach().cpu().tolist())
    return all_scores


def depth_computing(scores: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    num = len(arr)
    depth = np.zeros(num, dtype=np.float64)
    for i in range(num):
        left_flag = arr[i]
        for li in range(i - 1, -1, -1):
            if arr[li] >= left_flag:
                left_flag = arr[li]
            else:
                break
        right_flag = arr[i]
        for ri in range(i + 1, num):
            if arr[ri] >= right_flag:
                right_flag = arr[ri]
            else:
                break
        depth[i] = 0.5 * (left_flag + right_flag - 2.0 * arr[i])
    return depth


def texttiling_segment(
    depth_scores: np.ndarray,
    n_utterances: int,
    alpha: float,
) -> list[int]:
    threshold = depth_scores.mean() + alpha * depth_scores.std()
    boundaries = np.zeros(n_utterances, dtype=np.int32)
    nd = len(depth_scores)
    boundaries[:nd] = (depth_scores > threshold).astype(np.int32)
    boundaries[-1] = 0
    return boundaries.tolist()


def cache_nsp_depths(
    dialogues: list,
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
    use_nsp_head: bool = True,
) -> list[dict]:
    cached: list[dict] = []
    for dial in tqdm(dialogues, desc="NSP score cache"):
        utts = dial["utterances"] if isinstance(dial, dict) else dial.utterances
        if len(utts) < 2:
            continue
        sim = similarity_computing(
            utts,
            tokenizer,
            model,
            device,
            batch_size=batch_size,
            use_nsp_head=use_nsp_head,
        )
        depth = depth_computing(sim)
        segments = dial["segments"] if isinstance(dial, dict) else dial.segments
        cached.append(
            {
                "dialogue": dial,
                "similarity": np.asarray(sim, dtype=np.float64),
                "depth": depth,
                "mean": float(depth.mean()) if depth.size else 0.0,
                "std": float(depth.std()) if depth.size else 0.0,
                "n": len(utts),
                "segments": segments,
            }
        )
    return cached
