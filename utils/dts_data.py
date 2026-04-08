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


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


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
        batch_size: int = 32,
        max_utterances: int = MAX_UTTERANCES,
        max_utt_tokens: int = MAX_UTT_TOKENS,
        dataset_name: str = "",
        topic_word_set: dict[str, set[str]] | None = None,
    ):
        self.max_utterances = max_utterances
        self.max_utt_tokens = max_utt_tokens
        self.samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        topic_word_set = topic_word_set or {}
        keyword_set = topic_word_set.get(dataset_name, set())

        all_texts: list[str] = []
        meta: list[tuple[int, int]] = []
        labels_list: list[torch.Tensor] = []
        kw_list: list[torch.Tensor] = []

        for dialogue in dialogues:
            utts = dialogue.utterances[:max_utterances]
            n = len(utts)
            if n == 0:
                continue
            full_b = segments_to_boundaries(dialogue.segments)
            labels_list.append(torch.tensor(full_b[:n], dtype=torch.float32))
            kw_list.append(_build_kw_scores(utts, keyword_set))
            start = len(all_texts)
            all_texts.extend(utts)
            meta.append((start, n))

        if not all_texts:
            return

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

        for (start, n), labels, kw_scores in zip(meta, labels_list, kw_list):
            es = torch.tensor(sent_np[start : start + n], dtype=torch.float32)
            ew = torch.tensor(tok_np[start : start + n], dtype=torch.float32)
            tm = torch.tensor(mask_np[start : start + n], dtype=torch.float32)
            self.samples.append((es, ew, tm, labels, kw_scores))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_fn(batch, max_utterances: int = MAX_UTTERANCES):
    emb_s, emb_w, tok_m, labels, kw_scores = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in emb_s], dtype=torch.long)
    bsz = len(batch)
    d = emb_s[0].shape[1]
    lt = emb_w[0].shape[1]
    pad_s = torch.zeros(bsz, max_utterances, d, dtype=torch.float32)
    pad_w = torch.zeros(bsz, max_utterances, lt, d, dtype=torch.float32)
    pad_m = torch.zeros(bsz, max_utterances, lt, dtype=torch.float32)
    pad_y = torch.full((bsz, max_utterances), -1.0)
    pad_kw = torch.zeros(bsz, max_utterances, dtype=torch.float32)
    for i, (s, w, m, y, kw) in enumerate(zip(emb_s, emb_w, tok_m, labels, kw_scores)):
        t = int(lengths[i].item())
        pad_s[i, :t] = s
        pad_w[i, :t] = w
        pad_m[i, :t] = m
        pad_y[i, :t] = y
        pad_kw[i, :t] = kw
    return pad_s, pad_w, pad_m, pad_y, lengths, pad_kw

