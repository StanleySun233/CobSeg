import segeval
from sklearn.metrics import f1_score, precision_score, recall_score


def boundaries_to_seg_points_dydts(boundaries):
    """
    按 DyDTS 的 convert_to_binary_segments 输入格式，boundary[i]=1 认为在 utterance i 处发生切分，
    DyDTS 的 seg_points 用的是 seg_p_labels 的索引，因此这里映射为 idx=i+1。
    """
    n = len(boundaries)
    seg_points = []
    for i, b in enumerate(boundaries):
        if b == 1:
            idx = i + 1
            # 与 DyDTS 代码一致：只允许落在 (0, n) 之间的切分点
            if 0 < idx < n:
                seg_points.append(idx)
    return seg_points


def convert_to_binary_segments_dydts(seg_points, n):
    """
    直接复刻 DyDTS/inference.py 的 convert_to_binary_segments 逻辑。

    DyDTS: seg_p_labels 长度为 len(contents)+1，也就是 n+1；切分点在 seg_p_labels 上打 1。
    """
    seg_p_labels = [0] * (n + 1)
    for idx in seg_points:
        if 0 <= idx < len(seg_p_labels):
            seg_p_labels[idx] = 1

    results_p = []
    tmp = 0
    for fake in seg_p_labels:
        if fake == 1:
            tmp += 1
            results_p.append(tmp)
            tmp = 0
        else:
            tmp += 1
    results_p.append(tmp)

    # DyDTS 里：results_p[0] = results_p[0] - 1
    if results_p:
        results_p[0] = results_p[0] - 1
    return results_p


def evaluate_wd_pk_f1(pred_boundaries, true_boundaries):
    # 与 DyDTS/inference.py 保持一致：先把 boundary 转为 seg_points，再转成段长度/masses。
    n = len(pred_boundaries)
    pred_seg_points = boundaries_to_seg_points_dydts(pred_boundaries)
    true_seg_points = boundaries_to_seg_points_dydts(true_boundaries)

    pred_seg = convert_to_binary_segments_dydts(pred_seg_points, n)
    true_seg = convert_to_binary_segments_dydts(true_seg_points, n)
    wd = segeval.window_diff(pred_seg, true_seg)
    pk = segeval.pk(pred_seg, true_seg)
    pred_labels = [1 if b == 1 else 0 for b in pred_boundaries]
    true_labels = [1 if b == 1 else 0 for b in true_boundaries]
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    return wd, pk, f1


def evaluate_segmentation(reference, hypothesis):
    # Calculate WD, PK, F1 using segeval
    wd, pk, f1 = evaluate_wd_pk_f1(hypothesis, reference)

    # Calculate Precision and Recall for compatibility
    min_length = min(len(reference), len(hypothesis))
    ref_seq = reference[:min_length]
    hyp_seq = hypothesis[:min_length]
    precision = precision_score(ref_seq, hyp_seq, pos_label=1, zero_division=0)
    recall = recall_score(ref_seq, hyp_seq, pos_label=1, zero_division=0)

    # Ensure all values are native Python float types (not Decimal, numpy types, etc.)
    return {
        'PK': float(pk),
        'WD': float(wd),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1': float(f1)
    }
