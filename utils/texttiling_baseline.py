from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Iterable

import torch
from utils.dts_utils import segments_to_boundaries

_TOKEN_RE = re.compile(r"\b[a-z]+\b")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def build_vocabulary(utterances: list[str]) -> list[str]:
    vocab = sorted({tok for utt in utterances for tok in tokenize(utt)})
    return vocab


def build_tf_matrix(utterances: list[str], vocab: list[str], *, device: torch.device | None = None) -> torch.Tensor:
    vocab_index = {term: idx for idx, term in enumerate(vocab)}
    matrix = torch.zeros((len(utterances), len(vocab)), dtype=torch.float32, device=device)
    if not utterances or not vocab:
        return matrix
    for row_idx, utt in enumerate(utterances):
        counts = Counter(tokenize(utt))
        if not counts:
            continue
        for token, count in counts.items():
            col_idx = vocab_index.get(token)
            if col_idx is not None:
                matrix[row_idx, col_idx] = float(count)
    return matrix


def cosine_similarities(tf_matrix: torch.Tensor) -> torch.Tensor:
    if tf_matrix.size(0) < 2:
        return torch.zeros(0, dtype=torch.float32)
    left = tf_matrix[:-1]
    right = tf_matrix[1:]
    left_norm = torch.linalg.norm(left, dim=1).clamp(min=1e-8)
    right_norm = torch.linalg.norm(right, dim=1).clamp(min=1e-8)
    dots = (left * right).sum(dim=1)
    return dots / (left_norm * right_norm)


def depth_scores(similarities: torch.Tensor, window: int = 2) -> torch.Tensor:
    if similarities.numel() == 0:
        return torch.zeros(0, dtype=torch.float32)
    scores = torch.zeros_like(similarities)
    n = similarities.numel()
    for idx in range(n):
        left_start = max(0, idx - window)
        left_slice = similarities[left_start : idx + 1]
        right_end = min(n, idx + window + 1)
        right_slice = similarities[idx:right_end]
        left_peak = left_slice.max() if left_slice.numel() else similarities[idx]
        right_peak = right_slice.max() if right_slice.numel() else similarities[idx]
        scores[idx] = (left_peak - similarities[idx]) + (right_peak - similarities[idx])
    return scores


def compute_texttiling_scores(
    utterances: list[str],
    *,
    device: torch.device | None = None,
    window: int = 2,
) -> torch.Tensor:
    if len(utterances) <= 1:
        return torch.zeros(0, dtype=torch.float32, device=device)
    vocab = build_vocabulary(utterances)
    tf_matrix = build_tf_matrix(utterances, vocab, device=device)
    sims = cosine_similarities(tf_matrix)
    return depth_scores(sims, window=window)


def scores_to_boundaries(scores: torch.Tensor, alpha: float) -> list[int]:
    if scores.numel() == 0:
        return []
    mean = scores.mean()
    std = scores.std(unbiased=False)
    threshold = mean + float(alpha) * std
    boundaries: list[int] = []
    for idx, score in enumerate(scores.tolist()):
        left_ok = idx == 0 or score >= scores[idx - 1].item()
        right_ok = idx == scores.numel() - 1 or score >= scores[idx + 1].item()
        if score >= threshold.item() and left_ok and right_ok:
            boundaries.append(idx)
    return boundaries


def scores_to_segments(scores: torch.Tensor, alpha: float, total_utterances: int) -> list[int]:
    return boundaries_to_segments_from_indices(scores_to_boundaries(scores, alpha), total_utterances)


def select_boundaries(
    utterances: list[str],
    alpha: float,
    *,
    window: int = 2,
) -> list[int]:
    scores = compute_texttiling_scores(utterances, window=window)
    return scores_to_boundaries(scores, alpha)


def boundaries_to_segments_from_indices(boundaries: list[int], total_utterances: int) -> list[int]:
    if total_utterances <= 0:
        return []
    if not boundaries:
        return [total_utterances]
    cleaned = sorted({b for b in boundaries if 0 <= b < total_utterances - 1})
    if not cleaned:
        return [total_utterances]
    segments: list[int] = []
    start = 0
    for boundary in cleaned:
        end = boundary + 1
        if end > start:
            segments.append(end - start)
            start = end
    if start < total_utterances:
        segments.append(total_utterances - start)
    return segments


@dataclass(frozen=True)
class TextTilingResult:
    boundaries: list[int]
    segments: list[int]
    scores: list[float]
    threshold: float


def run_texttiling(utterances: list[str], alpha: float, *, window: int = 2) -> TextTilingResult:
    scores = compute_texttiling_scores(utterances, window=window)
    if scores.numel() == 0:
        segments = [len(utterances)] if utterances else []
        return TextTilingResult(boundaries=[], segments=segments, scores=[], threshold=0.0)

    mean = scores.mean()
    std = scores.std(unbiased=False)
    threshold = float((mean + float(alpha) * std).item())
    boundaries = scores_to_boundaries(scores, alpha)
    segments = boundaries_to_segments_from_indices(boundaries, len(utterances))
    return TextTilingResult(
        boundaries=boundaries,
        segments=segments,
        scores=[float(v) for v in scores.tolist()],
        threshold=threshold,
    )


def segment_texts(utterances: list[str], segments: list[int]) -> list[list[str]]:
    chunks: list[list[str]] = []
    start = 0
    for seg_len in segments:
        if seg_len <= 0:
            continue
        end = start + seg_len
        chunks.append(utterances[start:end])
        start = end
    if start < len(utterances):
        chunks.append(utterances[start:])
    return chunks


def average_segment_length(segments: Iterable[list[str]]) -> int:
    lengths = [len(seg) for seg in segments if seg]
    if not lengths:
        return 1
    return max(int(math.floor(sum(lengths) / len(lengths))), 1)


def flatten_segment_samples(dialogue_id: str, segments: list[list[str]]) -> list[dict]:
    samples: list[dict] = []
    for seg_idx, seg in enumerate(segments):
        samples.append(
            {
                "source_dial_id": dialogue_id,
                "source_segment_index": seg_idx,
                "utterances": seg,
                "segment_len": len(seg),
            }
        )
    return samples


def segment_lengths_to_boundaries(segment_lengths: list[int]) -> list[int]:
    return segments_to_boundaries(segment_lengths)


def boundaries_to_segment_lengths(boundaries: list[int]) -> list[int]:
    total = len(boundaries)
    if total == 0:
        return []
    return boundaries_to_segments_from_indices([idx for idx, flag in enumerate(boundaries) if flag], total)
