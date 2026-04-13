"""
TextTiling algorithm for dialogue topic segmentation.

Ported from ~/DTS/DTSr/neural_texttiling.py, adapted to the project's
boundary convention (boundary[i]=1 means utterance i is the last of a
non-final segment; last utterance is always 0).

Performance notes:
  - depth_computing uses numba JIT for ~50x speedup over pure Python loops.
  - alpha_search is vectorized: pre-computes depth stats, then sweeps all
    alphas with a fast PK-only evaluator (skips sklearn.f1_score which
    dominates ~90% of the original evaluate_all cost).
"""
import numpy as np
import torch
from tqdm import tqdm
from numba import njit

from utils.dts_utils import evaluate_all, segments_to_boundaries


# ---------------------------------------------------------------------------
# 1. Similarity computing  (NSP / SC pairwise scores)
# ---------------------------------------------------------------------------

def similarity_computing(
    texts: list[str],
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
    use_nsp_head: bool = False,
    mode: str = "NSP",
) -> list[float]:
    """Compute pairwise coherence scores between adjacent utterances.

    For NSP mode: feeds (u_i, u_{i+1}) as sentence pairs into a cross-encoder
    and returns the softmax probability of "same segment".

    For SC mode: encodes each utterance individually (bi-encoder), then
    computes cosine similarity between adjacent embeddings.

    Args:
        texts: N utterance strings.
        tokenizer: HuggingFace tokenizer.
        model: NSP/SequenceClassification model (NSP mode) or AutoModel (SC mode).
        device: torch device.
        batch_size: batch size for inference.
        use_nsp_head: if True, prob index 0 is "isNext" (native BERT NSP);
                      if False, prob index 1 is the "same segment" class.
        mode: "NSP" for cross-encoder, "SC" for bi-encoder cosine.

    Returns:
        List of N-1 similarity scores in [0, 1].
    """
    model.eval()

    if mode == "SC":
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
            )
            mask = inputs["attention_mask"].unsqueeze(-1).to(device).float()
            embeddings = torch.sum(outputs[0] * mask, dim=1) / torch.clamp(
                mask.sum(dim=1), min=1e-9
            )
        scores = (
            torch.cosine_similarity(embeddings[:-1], embeddings[1:])
            .cpu()
            .tolist()
        )
        return scores

    # --- NSP mode: cross-encoder ---
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
            )
            logits = model(**tokenized.to(device)).logits  # (B, 2)
            probs = torch.softmax(logits, dim=1)
            all_scores.extend(probs[:, prob_idx].cpu().tolist())

    return all_scores


# ---------------------------------------------------------------------------
# 2. Depth computing  (valley detection, numba-accelerated)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _depth_computing_numba(scores: np.ndarray) -> np.ndarray:
    """Numba-JIT depth computing: find left/right peaks for each position."""
    num = len(scores)
    depth = np.zeros(num, dtype=np.float64)
    for i in range(num):
        left_flag = scores[i]
        for li in range(i - 1, -1, -1):
            if scores[li] >= left_flag:
                left_flag = scores[li]
            else:
                break

        right_flag = scores[i]
        for ri in range(i + 1, num):
            if scores[ri] >= right_flag:
                right_flag = scores[ri]
            else:
                break

        depth[i] = 0.5 * (left_flag + right_flag - 2.0 * scores[i])
    return depth


def depth_computing(scores: list[float] | np.ndarray) -> np.ndarray:
    """Convert similarity scores to depth scores.

    For each position i, find the nearest peak to the left and right.
    Depth = 0.5 * (left_peak + right_peak - 2 * scores[i]).
    Higher depth indicates a likely topic boundary.

    Uses numba JIT for acceleration.
    """
    arr = np.asarray(scores, dtype=np.float64)
    return _depth_computing_numba(arr)


# ---------------------------------------------------------------------------
# 3. TextTiling segmentation  (depth → boundaries, vectorized)
# ---------------------------------------------------------------------------

def texttiling_segment(
    depth_scores: np.ndarray,
    n_utterances: int,
    alpha: float,
) -> list[int]:
    """Apply threshold to depth scores and produce boundary labels.

    threshold = mean(depth) + alpha * std(depth)
    Positions where depth > threshold are segment boundaries.

    Returns:
        list of N ints (0 or 1). boundary[i]=1 means utterance i is the
        last utterance of a non-final segment.  boundary[N-1] is always 0.
        Compatible with segments_to_boundaries() and evaluate_all().
    """
    threshold = depth_scores.mean() + alpha * depth_scores.std()
    boundaries = np.zeros(n_utterances, dtype=np.int32)
    nd = len(depth_scores)
    boundaries[:nd] = (depth_scores > threshold).astype(np.int32)
    boundaries[-1] = 0
    return boundaries.tolist()


# ---------------------------------------------------------------------------
# 4. Fast PK-only evaluator for alpha search (skips slow sklearn.f1_score)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _pk_single(pred_segs: np.ndarray, true_segs: np.ndarray, k: int) -> float:
    """Compute Pk metric for a single dialogue (numba-accelerated).

    pred_segs / true_segs: 1-D arrays of segment lengths (e.g. [3, 5, 2]).
    k: half-window size (default: mean segment length / 2).
    """
    # Expand segment lengths to per-utterance segment IDs
    n_pred = int(pred_segs.sum())
    n_true = int(true_segs.sum())
    n = min(n_pred, n_true)
    if n <= 1:
        return 0.0

    pred_ids = np.empty(n, dtype=np.int64)
    idx = 0
    sid = 0
    for s in range(len(pred_segs)):
        seg_len = int(pred_segs[s])
        for _ in range(seg_len):
            if idx < n:
                pred_ids[idx] = sid
                idx += 1
        sid += 1

    true_ids = np.empty(n, dtype=np.int64)
    idx = 0
    sid = 0
    for s in range(len(true_segs)):
        seg_len = int(true_segs[s])
        for _ in range(seg_len):
            if idx < n:
                true_ids[idx] = sid
                idx += 1
        sid += 1

    if k <= 0:
        k = 1
    if k >= n:
        k = n - 1

    n_windows = n - k
    if n_windows <= 0:
        return 0.0

    misses = 0
    for i in range(n_windows):
        j = i + k
        pred_same = 1 if pred_ids[i] == pred_ids[j] else 0
        true_same = 1 if true_ids[i] == true_ids[j] else 0
        if pred_same != true_same:
            misses += 1

    return misses / n_windows


def _boundaries_to_seglens(boundaries: list[int] | np.ndarray) -> np.ndarray:
    """Convert boundary labels [0,0,1,0,0] -> segment lengths [3,2]."""
    segs = []
    count = 0
    for b in boundaries:
        count += 1
        if b == 1:
            segs.append(count)
            count = 0
    if count > 0:
        segs.append(count)
    return np.array(segs, dtype=np.int64)


def _fast_pk(all_preds: list[np.ndarray], all_gt_segs: list[np.ndarray]) -> float:
    """Compute mean Pk over all dialogues, using numba-accelerated _pk_single.

    k = max(2, round(mean(reference_segment_lengths) / 2)), matching segeval.
    """
    total_pk = 0.0
    n = len(all_preds)
    for pred_boundaries, gt_segs in zip(all_preds, all_gt_segs):
        pred_segs = _boundaries_to_seglens(pred_boundaries)
        # k = round(mean_segment_length / 2), min 2  (same as segeval)
        mean_seg = gt_segs.sum() / max(len(gt_segs), 1)
        k = max(2, round(mean_seg / 2))
        total_pk += _pk_single(pred_segs, gt_segs, k)
    return total_pk / max(n, 1)


# ---------------------------------------------------------------------------
# 5. Alpha search  (vectorized, fast PK-only evaluation)
# ---------------------------------------------------------------------------

def alpha_search(
    dialogues: list,
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
    use_nsp_head: bool = False,
    mode: str = "NSP",
    batch_size: int = 64,
    lower: float = -2.0,
    upper: float = 2.0,
    step: float = 0.1,
) -> tuple[float, float]:
    """Search for the best alpha threshold on a dev set.

    Pre-computes similarity and depth scores for all dialogues, then sweeps
    alpha values using a fast PK-only evaluator (no sklearn overhead).

    Args:
        dialogues: list of Utterance objects (from DialogueDataset).
        tokenizer: HuggingFace tokenizer.
        model: NSP/SC model.
        device: torch device.
        use_nsp_head: whether the model has a native NSP head.
        mode: "NSP" or "SC".
        batch_size: inference batch size.
        lower, upper, step: alpha search range.

    Returns:
        (best_alpha, best_pk)
    """
    print(f"[TET] Pre-computing scores for {len(dialogues)} dev dialogues...")

    # Pre-compute depth stats and ground truth for each dialogue
    cached: list[tuple[np.ndarray, float, float, int, np.ndarray]] = []
    for dial in tqdm(dialogues, desc="TET scores"):
        utts = dial.utterances
        n = len(utts)
        if n < 2:
            continue
        sim = similarity_computing(utts, tokenizer, model, device, batch_size,
                                   use_nsp_head=use_nsp_head, mode=mode)
        depth = depth_computing(sim)
        gt = segments_to_boundaries(dial.segments)[:n]
        gt_segs = _boundaries_to_seglens(gt)
        # Cache depth array, its mean/std, n_utterances, and ground truth segments
        cached.append((depth, float(depth.mean()), float(depth.std()), n, gt_segs))

    alphas = np.arange(lower, upper, step)
    print(f"[TET] Sweeping {len(alphas)} alpha values over {len(cached)} dialogues...")

    # Warm up numba JIT on a tiny input
    _pk_single(np.array([3, 2], dtype=np.int64), np.array([3, 2], dtype=np.int64), 1)

    best_alpha = float(alphas[0])
    best_pk = float("inf")

    for alpha in tqdm(alphas, desc="TET alpha search"):
        alpha_f = float(alpha)
        all_preds = []
        all_gt_segs = []
        for depth, d_mean, d_std, n, gt_segs in cached:
            threshold = d_mean + alpha_f * d_std
            boundaries = np.zeros(n, dtype=np.int32)
            nd = len(depth)
            boundaries[:nd] = (depth > threshold).astype(np.int32)
            boundaries[-1] = 0
            all_preds.append(boundaries)
            all_gt_segs.append(gt_segs)

        pk = _fast_pk(all_preds, all_gt_segs)
        if pk < best_pk:
            best_pk = pk
            best_alpha = alpha_f

    return best_alpha, best_pk
