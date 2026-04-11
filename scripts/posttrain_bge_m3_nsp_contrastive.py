import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.dialogue_dataset import DialogueDataset
from utils.dts_data import mean_pool
from utils.utils import resolve_dataset_path


DATASET_KEYS = ("vfh", "dialseg711", "doc2dial", "tiage", "superseg")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_train_dialogues() -> list[list[str]]:
    out: list[list[str]] = []
    for key in DATASET_KEYS:
        path = resolve_dataset_path(key)
        ds = DialogueDataset(path)
        for d in ds.data:
            if d.set != "train":
                continue
            utts = [u for u in d.utterances if u and str(u).strip()]
            if len(utts) >= 2:
                out.append(utts)
    return out


def build_positive_pairs(dialogues: list[list[str]]) -> list[tuple[str, str, int]]:
    pairs: list[tuple[str, str, int]] = []
    for di, utts in enumerate(dialogues):
        for i in range(len(utts) - 1):
            a, b = utts[i], utts[i + 1]
            if a.strip() and b.strip():
                pairs.append((a, b, di))
    return pairs


def sample_negative_tail(dialogues: list[list[str]], d_idx: int, rng: random.Random) -> str:
    n = len(dialogues)
    if n <= 1:
        return dialogues[0][0]
    d2 = rng.randrange(n - 1)
    if d2 >= d_idx:
        d2 += 1
    utts2 = dialogues[d2]
    return utts2[rng.randrange(len(utts2))]


class AdjacentPairDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str, int]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, str, int]:
        return self.pairs[idx]


def collate_pairs(batch: list[tuple[str, str, int]]):
    return list(zip(*batch))


class PostTrainNSPContrastive(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.nsp_head = nn.Linear(hidden_size, 2)

    def forward_nsp(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = mean_pool(out.last_hidden_state, attention_mask)
        return self.nsp_head(h)

    def encode_single(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return mean_pool(out.last_hidden_state, attention_mask)


def contrastive_in_batch_loss(z_a: torch.Tensor, z_b: torch.Tensor, tau: float) -> torch.Tensor:
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = (z_a @ z_b.T) / tau
    targets = torch.arange(z_a.size(0), device=z_a.device, dtype=torch.long)
    la = F.cross_entropy(logits, targets)
    lb = F.cross_entropy(logits.T, targets)
    return 0.5 * (la + lb)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_name", type=str, default="bge-m3-nsp-contrastive")
    p.add_argument("--model_name", type=str, default="BAAI/bge-m3")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--lambda_ctr", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--neg_prob", type=float, default=0.5, help="NSP: prob of replacing b with cross-dialogue random utterance")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    print("Loading train dialogues from all datasets...")
    dialogues = load_train_dialogues()
    print(f"Train dialogues (len>=2): {len(dialogues)}")
    pairs = build_positive_pairs(dialogues)
    print(f"Adjacent positive pairs: {len(pairs)}")
    if not pairs:
        print("No pairs; exit.")
        return

    out_dir = _REPO_ROOT / "data" / "posttrain" / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, local_files_only=True
    )
    encoder = AutoModel.from_pretrained(
        args.model_name, trust_remote_code=True, local_files_only=True
    )
    hidden = encoder.config.hidden_size
    model = PostTrainNSPContrastive(encoder, hidden).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce_nsp = nn.CrossEntropyLoss()

    ds = AdjacentPairDataset(pairs)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_pairs,
    )

    rng = random.Random(args.seed)
    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            texts_a, texts_b, d_indices = batch
            bs = len(texts_a)
            b_nsp: list[str] = []
            labels = torch.zeros(bs, dtype=torch.long, device=device)
            for i in range(bs):
                if rng.random() < args.neg_prob:
                    b_nsp.append(
                        sample_negative_tail(dialogues, int(d_indices[i]), rng)
                    )
                    labels[i] = 0
                else:
                    b_nsp.append(texts_b[i])
                    labels[i] = 1

            enc_nsp = tokenizer(
                list(texts_a),
                b_nsp,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            enc_nsp = {k: v.to(device) for k, v in enc_nsp.items()}
            logits_nsp = model.forward_nsp(
                enc_nsp["input_ids"], enc_nsp["attention_mask"]
            )
            loss_nsp = ce_nsp(logits_nsp, labels)

            enc_a = tokenizer(
                list(texts_a),
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            enc_b = tokenizer(
                list(texts_b),
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            enc_a = {k: v.to(device) for k, v in enc_a.items()}
            enc_b = {k: v.to(device) for k, v in enc_b.items()}
            z_a = model.encode_single(enc_a["input_ids"], enc_a["attention_mask"])
            z_b = model.encode_single(enc_b["input_ids"], enc_b["attention_mask"])
            loss_ctr = contrastive_in_batch_loss(z_a, z_b, args.tau)

            loss = loss_nsp + args.lambda_ctr * loss_ctr
            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step += 1
            pbar.set_postfix(
                nsp=float(loss_nsp.item()),
                ctr=float(loss_ctr.item()),
                total=float(loss.item()),
            )

    model.eval()
    save_root = str(out_dir)
    model.encoder.save_pretrained(save_root)
    tokenizer.save_pretrained(save_root)
    torch.save(model.nsp_head.state_dict(), os.path.join(save_root, "nsp_head.pt"))
    print(f"Saved encoder and tokenizer to {save_root}")
    print(f"Saved NSP head to {os.path.join(save_root, 'nsp_head.pt')}")


if __name__ == "__main__":
    main()
