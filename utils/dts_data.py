import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.dts_utils import segments_to_boundaries


MAX_UTT_TOKENS = 64
MAX_UTTERANCES = 128


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z]+\b", text.lower())


def _build_kw_scores(
    utterances: list[str],
    keyword_set: set[str],
) -> torch.Tensor:
    scores = []
    for utt in utterances:
        tokens = _tokenize(utt)
        if not tokens:
            scores.append(0.0)
            continue
        hit = sum(1 for t in tokens if t in keyword_set)
        scores.append(hit / len(tokens))
    return torch.tensor(scores, dtype=torch.float32)


def topic_channel_sets_from_info(info: dict) -> dict[str, set[str]]:
    legacy = set(info.get("all_top_words", []))
    ae = info.get("keywords_ae")
    bd = info.get("keywords_bd")
    if not isinstance(ae, dict):
        ae = {}
    if not isinstance(bd, dict):
        bd = {}
    distinctive = set(ae.get("distinctive", []))
    if not distinctive and legacy:
        distinctive = set(legacy)
    return {
        "coh_salient": distinctive,
        "coh_ambient": set(ae.get("ubiquitous", [])),
        "bnd_core": set(bd.get("prototype", [])),
        "bnd_marker": set(bd.get("boundary", [])),
    }


KW_COH_AMBIENT_W = 0.25
KW_BND_CORE_PENALTY = 0.15


def _build_kw_channel_scores(
    utterances: list[str],
    channels: dict[str, set[str]],
) -> torch.Tensor:
    d_set = channels.get("coh_salient", set())
    u_set = channels.get("coh_ambient", set())
    p_set = channels.get("bnd_core", set())
    b_set = channels.get("bnd_marker", set())
    t = len(utterances)
    out = torch.zeros(t, 2, dtype=torch.float32)
    for idx, utt in enumerate(utterances):
        tokens = _tokenize(utt)
        if not tokens:
            continue
        n = len(tokens)

        def dens(st: set[str]) -> float:
            return sum(1 for x in tokens if x in st) / n

        out[idx, 0] = dens(d_set) + KW_COH_AMBIENT_W * dens(u_set)
        out[idx, 1] = dens(b_set) - KW_BND_CORE_PENALTY * dens(p_set)
    return out


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


def tokenize_utterances_hf(
    tokenizer,
    texts: list[str],
    batch_size: int,
    max_utt_tokens: int = MAX_UTT_TOKENS,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    ids_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    n = len(texts)
    steps = range(0, n, batch_size)
    it = tqdm(steps, desc="HF tokenize", leave=False) if show_progress else steps
    for i in it:
        batch = texts[i : i + batch_size]
        fe = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_utt_tokens,
            return_tensors="pt",
        )
        ids_chunks.append(fe["input_ids"].cpu())
        mask_chunks.append(fe["attention_mask"].cpu())
    ids_all = torch.cat(ids_chunks, dim=0).numpy().astype(np.int64)
    mask_all = torch.cat(mask_chunks, dim=0).numpy().astype(np.int64)
    return ids_all, mask_all


def tokenize_sentence_pairs_hf(
    tokenizer,
    first_texts: list[str],
    second_texts: list[str],
    batch_size: int,
    max_pair_tokens: int,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    ids_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    n = len(first_texts)
    steps = range(0, n, batch_size)
    it = tqdm(steps, desc="HF pair tokenize", leave=False) if show_progress else steps
    for i in it:
        first_batch = first_texts[i : i + batch_size]
        second_batch = second_texts[i : i + batch_size]
        fe = tokenizer(
            first_batch,
            second_batch,
            padding="max_length",
            truncation=True,
            max_length=max_pair_tokens,
            return_tensors="pt",
        )
        ids_chunks.append(fe["input_ids"].cpu())
        mask_chunks.append(fe["attention_mask"].cpu())
    ids_all = torch.cat(ids_chunks, dim=0).numpy().astype(np.int64)
    mask_all = torch.cat(mask_chunks, dim=0).numpy().astype(np.int64)
    return ids_all, mask_all


def encode_utterances_hf(
    enc_model: nn.Module,
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    max_utt_tokens: int = MAX_UTT_TOKENS,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    enc_model.eval()
    sent_chunks: list[torch.Tensor] = []
    hid_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    n = len(texts)
    steps = range(0, n, batch_size)
    it = tqdm(steps, desc="HF encode", leave=False) if show_progress else steps
    with torch.no_grad():
        for i in it:
            batch = texts[i : i + batch_size]
            fe = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=max_utt_tokens,
                return_tensors="pt",
            )
            fe = {k: v.to(device) for k, v in fe.items()}
            out = enc_model(**fe)
            hidden = out.last_hidden_state
            mask = fe["attention_mask"].float()
            sent = mean_pool(hidden, mask)
            sent_chunks.append(sent.float().cpu())
            hid_chunks.append(hidden.float().cpu())
            mask_chunks.append(mask.cpu())
    sent_all = torch.cat(sent_chunks, dim=0).numpy().astype(np.float32)
    hid_all = torch.cat(hid_chunks, dim=0).numpy().astype(np.float32)
    m_all = torch.cat(mask_chunks, dim=0).numpy().astype(np.float32)
    return sent_all, hid_all, m_all


class EmbeddedDialogueDataset(Dataset):
    def __init__(
        self,
        dialogues,
        enc_model: nn.Module,
        tokenizer,
        device: torch.device,
        cs_enc_model: nn.Module | None = None,
        cs_tokenizer=None,
        batch_size: int = 32,
        max_utterances: int = MAX_UTTERANCES,
        max_utt_tokens: int = MAX_UTT_TOKENS,
        dataset_name: str = "",
        topic_channels: dict[str, set[str]] | None = None,
        finetune_main_encoder: bool = False,
        finetune_topic_encoder: bool = False,
        use_nsp_cross_encoder: bool = False,
        nsp_max_pair_tokens: int | None = None,
    ):
        self.max_utterances = max_utterances
        self.max_utt_tokens = max_utt_tokens
        self.finetune_main_encoder = bool(finetune_main_encoder)
        self.finetune_topic_encoder = bool(finetune_topic_encoder)
        self.use_nsp_cross_encoder = bool(use_nsp_cross_encoder)
        self.nsp_max_pair_tokens = int(
            nsp_max_pair_tokens
            if nsp_max_pair_tokens is not None
            else min(max(max_utt_tokens * 2, max_utt_tokens + 8), 256)
        )
        self.samples: list[tuple[torch.Tensor, ...]] = []
        channels = topic_channels or {}

        all_texts: list[str] = []
        meta: list[tuple[int, int]] = []
        pair_first_texts: list[str] = []
        pair_second_texts: list[str] = []
        pair_meta: list[tuple[int, int]] = []
        labels_list: list[torch.Tensor] = []
        kw_list: list[torch.Tensor] = []

        for dialogue in dialogues:
            utts = dialogue.utterances[:max_utterances]
            n = len(utts)
            if n == 0:
                continue
            full_b = segments_to_boundaries(dialogue.segments)
            labels_list.append(torch.tensor(full_b[:n], dtype=torch.float32))
            kw_list.append(_build_kw_channel_scores(utts, channels))
            start = len(all_texts)
            all_texts.extend(utts)
            meta.append((start, n))
            pair_start = len(pair_first_texts)
            if self.use_nsp_cross_encoder and n > 1:
                pair_first_texts.extend(utts[:-1])
                pair_second_texts.extend(utts[1:])
            pair_meta.append((pair_start, max(n - 1, 0)))

        if not all_texts:
            return

        sent_np = None
        tok_np = None
        mask_np = None
        ids_np = None
        topic_ids_np = None
        topic_mask_np = None
        if self.finetune_main_encoder:
            print(
                f"Tokenizing {len(all_texts)} utterances "
                f"for trainable main encoder (U×L, L={max_utt_tokens}, U≤{max_utterances}) …"
            )
            ids_np, mask_np = tokenize_utterances_hf(
                tokenizer,
                all_texts,
                batch_size=batch_size,
                max_utt_tokens=max_utt_tokens,
                show_progress=True,
            )
        else:
            print(
                f"Encoding {len(all_texts)} utterances "
                f"(HF mean-pool (U×D) + tokens (U×L×D), L={max_utt_tokens}, U≤{max_utterances}) …"
            )
            sent_np, tok_np, mask_np = encode_utterances_hf(
                enc_model,
                tokenizer,
                all_texts,
                device,
                batch_size=batch_size,
                max_utt_tokens=max_utt_tokens,
                show_progress=True,
            )
        cs_sent_np = sent_np
        if self.finetune_topic_encoder:
            if cs_tokenizer is None:
                raise ValueError("finetune_topic_encoder=True requires cs_tokenizer.")
            print(
                f"Tokenizing {len(all_texts)} utterances for trainable topic encoder "
                f"(L={max_utt_tokens}) …"
            )
            topic_ids_np, topic_mask_np = tokenize_utterances_hf(
                cs_tokenizer,
                all_texts,
                batch_size=batch_size,
                max_utt_tokens=max_utt_tokens,
                show_progress=True,
            )
        elif cs_enc_model is not None and cs_tokenizer is not None:
            print(
                f"Encoding {len(all_texts)} utterances for CS "
                f"(SimCSE mean-pool, L={max_utt_tokens}) …"
            )
            cs_sent_np, _, _ = encode_utterances_hf(
                cs_enc_model,
                cs_tokenizer,
                all_texts,
                device,
                batch_size=batch_size,
                max_utt_tokens=max_utt_tokens,
                show_progress=True,
            )
        elif self.finetune_main_encoder:
            raise ValueError(
                "finetune_main_encoder=True requires a separate CS/topic encoder "
                "for static topic features."
            )

        pair_ids_np = None
        pair_mask_np = None
        if self.use_nsp_cross_encoder:
            if not self.finetune_main_encoder:
                raise ValueError(
                    "use_nsp_cross_encoder=True requires finetune_main_encoder=True "
                    "so the trained RoBERTa pair encoder can be reused in stage-2."
                )
            if pair_first_texts:
                print(
                    f"Tokenizing {len(pair_first_texts)} adjacent utterance pairs "
                    f"for NSP cross-encoder (L={self.nsp_max_pair_tokens}) …"
                )
                pair_ids_np, pair_mask_np = tokenize_sentence_pairs_hf(
                    tokenizer,
                    pair_first_texts,
                    pair_second_texts,
                    batch_size=batch_size,
                    max_pair_tokens=self.nsp_max_pair_tokens,
                    show_progress=True,
                )
            else:
                pair_ids_np = np.zeros((0, self.nsp_max_pair_tokens), dtype=np.int64)
                pair_mask_np = np.zeros((0, self.nsp_max_pair_tokens), dtype=np.int64)

        for (start, n), (pair_start, n_pairs), labels, kw_scores in zip(
            meta,
            pair_meta,
            labels_list,
            kw_list,
        ):
            if self.finetune_main_encoder:
                input_ids = torch.tensor(ids_np[start : start + n], dtype=torch.long)
                attn_mask = torch.tensor(mask_np[start : start + n], dtype=torch.long)
                if self.finetune_topic_encoder:
                    topic_ids = torch.tensor(topic_ids_np[start : start + n], dtype=torch.long)
                    topic_attn = torch.tensor(topic_mask_np[start : start + n], dtype=torch.long)
                    if self.use_nsp_cross_encoder:
                        pair_ids = torch.zeros(n, self.nsp_max_pair_tokens, dtype=torch.long)
                        pair_attn = torch.zeros(n, self.nsp_max_pair_tokens, dtype=torch.long)
                        if n_pairs > 0:
                            pair_ids[:n_pairs] = torch.tensor(
                                pair_ids_np[pair_start : pair_start + n_pairs], dtype=torch.long
                            )
                            pair_attn[:n_pairs] = torch.tensor(
                                pair_mask_np[pair_start : pair_start + n_pairs], dtype=torch.long
                            )
                        self.samples.append(
                            (
                                input_ids,
                                attn_mask,
                                pair_ids,
                                pair_attn,
                                topic_ids,
                                topic_attn,
                                labels,
                                kw_scores,
                            )
                        )
                    else:
                        self.samples.append(
                            (input_ids, attn_mask, topic_ids, topic_attn, labels, kw_scores)
                        )
                elif self.use_nsp_cross_encoder:
                    cs_sent_slice = cs_sent_np[start : start + n]
                    et = torch.tensor(cs_sent_slice, dtype=torch.float32)
                    pair_ids = torch.zeros(n, self.nsp_max_pair_tokens, dtype=torch.long)
                    pair_attn = torch.zeros(n, self.nsp_max_pair_tokens, dtype=torch.long)
                    if n_pairs > 0:
                        pair_ids[:n_pairs] = torch.tensor(
                            pair_ids_np[pair_start : pair_start + n_pairs], dtype=torch.long
                        )
                        pair_attn[:n_pairs] = torch.tensor(
                            pair_mask_np[pair_start : pair_start + n_pairs], dtype=torch.long
                        )
                    self.samples.append(
                        (input_ids, attn_mask, pair_ids, pair_attn, et, labels, kw_scores)
                    )
                else:
                    cs_sent_slice = cs_sent_np[start : start + n]
                    et = torch.tensor(cs_sent_slice, dtype=torch.float32)
                    self.samples.append((input_ids, attn_mask, et, labels, kw_scores))
            else:
                cs_sent_slice = cs_sent_np[start : start + n]
                et = torch.tensor(cs_sent_slice, dtype=torch.float32)
                sent_slice = sent_np[start : start + n]
                es = torch.tensor(sent_slice, dtype=torch.float32)
                ew = torch.tensor(tok_np[start : start + n], dtype=torch.float32)
                tm = torch.tensor(mask_np[start : start + n], dtype=torch.float32)
                self.samples.append((es, ew, tm, et, labels, kw_scores))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_fn(batch, max_utterances: int = MAX_UTTERANCES):
    emb_s, emb_w, tok_m, emb_t, labels, kw_scores = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in emb_s], dtype=torch.long)
    bsz = len(batch)
    d = emb_s[0].shape[1]
    lt = emb_w[0].shape[1]
    pad_s = torch.zeros(bsz, max_utterances, d, dtype=torch.float32)
    pad_w = torch.zeros(bsz, max_utterances, lt, d, dtype=torch.float32)
    pad_m = torch.zeros(bsz, max_utterances, lt, dtype=torch.float32)
    pad_t = torch.zeros(bsz, max_utterances, d, dtype=torch.float32)
    pad_y = torch.full((bsz, max_utterances), -1.0)
    kw_dim = kw_scores[0].shape[-1] if kw_scores[0].dim() > 1 else 1
    pad_kw = torch.zeros(bsz, max_utterances, kw_dim, dtype=torch.float32)
    for i, (s, w, m, t_emb, y, kw) in enumerate(zip(emb_s, emb_w, tok_m, emb_t, labels, kw_scores)):
        t = int(lengths[i].item())
        pad_s[i, :t] = s
        pad_w[i, :t] = w
        pad_m[i, :t] = m
        pad_t[i, :t] = t_emb
        pad_y[i, :t] = y
        if kw.dim() == 1:
            pad_kw[i, :t, 0] = kw
        else:
            pad_kw[i, :t, :] = kw
    return pad_s, pad_w, pad_m, pad_t, pad_y, lengths, pad_kw


def collate_finetune_fn(batch, max_utterances: int = MAX_UTTERANCES):
    sample_len = len(batch[0])
    if sample_len == 5:
        input_ids, attn_masks, emb_t, labels, kw_scores = zip(*batch)
        pair_ids = None
        pair_attn_masks = None
        topic_ids = None
        topic_attn_masks = None
    elif sample_len == 6:
        input_ids, attn_masks, topic_ids, topic_attn_masks, labels, kw_scores = zip(*batch)
        pair_ids = None
        pair_attn_masks = None
        emb_t = None
    elif sample_len == 7:
        input_ids, attn_masks, pair_ids, pair_attn_masks, emb_t, labels, kw_scores = zip(*batch)
        topic_ids = None
        topic_attn_masks = None
    elif sample_len == 8:
        input_ids, attn_masks, pair_ids, pair_attn_masks, topic_ids, topic_attn_masks, labels, kw_scores = zip(*batch)
        emb_t = None
    else:
        raise ValueError(f"Unsupported finetune sample format with {sample_len} fields.")

    lengths = torch.tensor([ids.shape[0] for ids in input_ids], dtype=torch.long)
    bsz = len(batch)
    lt = input_ids[0].shape[1]
    pad_ids = torch.zeros(bsz, max_utterances, lt, dtype=torch.long)
    pad_attn = torch.zeros(bsz, max_utterances, lt, dtype=torch.long)
    pad_pair_ids = None
    pad_pair_attn = None
    if pair_ids is not None and pair_attn_masks is not None:
        lp = pair_ids[0].shape[1]
        pad_pair_ids = torch.zeros(bsz, max_utterances, lp, dtype=torch.long)
        pad_pair_attn = torch.zeros(bsz, max_utterances, lp, dtype=torch.long)
    pad_topic_ids = None
    pad_topic_attn = None
    if topic_ids is not None and topic_attn_masks is not None:
        ltopic = topic_ids[0].shape[1]
        pad_topic_ids = torch.zeros(bsz, max_utterances, ltopic, dtype=torch.long)
        pad_topic_attn = torch.zeros(bsz, max_utterances, ltopic, dtype=torch.long)
    pad_t = None
    if emb_t is not None:
        d = emb_t[0].shape[1]
        pad_t = torch.zeros(bsz, max_utterances, d, dtype=torch.float32)
    pad_y = torch.full((bsz, max_utterances), -1.0)
    kw_dim = kw_scores[0].shape[-1] if kw_scores[0].dim() > 1 else 1
    pad_kw = torch.zeros(bsz, max_utterances, kw_dim, dtype=torch.float32)
    for i, (ids, attn, y, kw) in enumerate(zip(input_ids, attn_masks, labels, kw_scores)):
        t = int(lengths[i].item())
        pad_ids[i, :t] = ids
        pad_attn[i, :t] = attn
        if pad_t is not None:
            pad_t[i, :t] = emb_t[i]
        pad_y[i, :t] = y
        if kw.dim() == 1:
            pad_kw[i, :t, 0] = kw
        else:
            pad_kw[i, :t, :] = kw
        if pad_pair_ids is not None and pad_pair_attn is not None:
            pad_pair_ids[i, :t] = pair_ids[i]
            pad_pair_attn[i, :t] = pair_attn_masks[i]
        if pad_topic_ids is not None and pad_topic_attn is not None:
            pad_topic_ids[i, :t] = topic_ids[i]
            pad_topic_attn[i, :t] = topic_attn_masks[i]
    if pad_pair_ids is not None and pad_pair_attn is not None and pad_topic_ids is not None and pad_topic_attn is not None:
        return (
            pad_ids,
            pad_attn,
            pad_pair_ids,
            pad_pair_attn,
            pad_topic_ids,
            pad_topic_attn,
            pad_y,
            lengths,
            pad_kw,
        )
    if pad_pair_ids is not None and pad_pair_attn is not None:
        return pad_ids, pad_attn, pad_pair_ids, pad_pair_attn, pad_t, pad_y, lengths, pad_kw
    if pad_topic_ids is not None and pad_topic_attn is not None:
        return pad_ids, pad_attn, pad_topic_ids, pad_topic_attn, pad_y, lengths, pad_kw
    return pad_ids, pad_attn, pad_t, pad_y, lengths, pad_kw
