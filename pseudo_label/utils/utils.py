from __future__ import annotations

import re
from pathlib import Path


def normalize_dataset_name(dataset_name_or_path: str) -> str:
    if not dataset_name_or_path:
        return "vhf"
    text = str(dataset_name_or_path).strip()
    if text.lower().endswith(".json"):
        text = Path(text).stem
    text = text.lower()
    if text == "vfh":
        return "vhf"
    return text


def base_dataset_name(dataset_name_or_path: str) -> str:
    name = normalize_dataset_name(dataset_name_or_path)
    return re.sub(r"_weak_\d+$", "", name)


def resolve_dataset_path(dataset_name_or_path: str) -> str:
    if not dataset_name_or_path:
        dataset_name_or_path = "vhf"
    text = str(dataset_name_or_path).strip()
    if text.lower().endswith(".json"):
        return text
    mapping = {
        "vhf": "./data/dataset/vhf.json",
        "dialseg711": "./data/dataset/dialseg711.json",
        "dialseg711_weak_50": "./data/dataset/dialseg711_weak_50.json",
        "doc2dial": "./data/dataset/doc2dial.json",
        "tiage": "./data/dataset/tiage.json",
        "superseg": "./data/dataset/superseg.json",
    }
    key = normalize_dataset_name(text)
    if key in mapping:
        return mapping[key]
    candidate = Path("./data/dataset") / f"{key}.json"
    if candidate.exists():
        return str(candidate)
    return mapping["vhf"]
