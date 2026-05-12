from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


_SPEAKER_PREFIX_RE = re.compile(r"^\s*([^:：]{1,40})\s*[:：]\s*(\S.*)$")


def strip_speaker_label(text: str) -> str:
    match = _SPEAKER_PREFIX_RE.match(text)
    if not match:
        return text.strip()
    prefix = match.group(1).strip()
    lowered = prefix.lower()
    if lowered.startswith("speaker") or prefix.startswith("说话人") or (
        any(ch.isalpha() for ch in prefix) and prefix.upper() == prefix
    ):
        return match.group(2).strip()
    return text.strip()


def load_dataset(dataset_path: str) -> list[dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list dataset at {dataset_path}")
    return data


def split_dataset(data: list[dict]) -> dict[str, list[dict]]:
    split_map: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    for item in data:
        split = str(item.get("set", "train")).strip().lower()
        if split in {"val", "dev"}:
            split = "valid"
        if split not in split_map:
            split = "train"
        split_map[split].append(item)
    return split_map


def sample_train_dialogues(train_dialogues: list[dict], seed: int, sample_size: int) -> list[dict]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(train_dialogues):
        raise ValueError(f"sample_size={sample_size} exceeds train size={len(train_dialogues)}")
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(train_dialogues), size=sample_size, replace=False).tolist())
    return [train_dialogues[idx] for idx in indices]


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class FlanT5Summarizer:
    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.model.eval()

    @staticmethod
    def _clean_summary(text: str) -> str:
        text = text.strip()
        if not text:
            return text
        text = re.sub(r"(?i)^\s*(summarize|summary|topic)\s*[:\-]\s*", "", text)
        text = re.sub(r"(?i)\b(summarize the topic of this dialogue segment in one short sentence)\b", "", text)
        text = re.sub(r"\s+", " ", text).strip(" -:;")
        if not text:
            return ""
        for sep in [".", "!", "?", "\n"]:
            if sep in text:
                text = text.split(sep, 1)[0].strip()
                break
        return text.strip(" -:;")

    def summarize(self, utterances: list[str], max_new_tokens: int = 32) -> str:
        return self.summarize_batch([utterances], max_new_tokens=max_new_tokens)[0]

    def summarize_batch(self, batch_utterances: list[list[str]], max_new_tokens: int = 32) -> list[str]:
        prompts = [
            (
                "Write a one-sentence topic summary in third person. "
                "Do not use first person or second person. "
                + " </s> ".join(utterances)
            )
            for utterances in batch_utterances
        ]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=1.15,
                length_penalty=1.0,
            )
        raw_summaries = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        cleaned: list[str] = []
        for raw in raw_summaries:
            text = raw.strip()
            summary = self._clean_summary(text)
            cleaned.append(summary or text)
        return cleaned


def get_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required for synthetic utterance generation.") from exc

    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("openai_base_url") or None
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key") or None
    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    return OpenAI(**kwargs)


def extract_json_object(text: str) -> dict:
    text = text.strip()
    fenced_patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
    ]
    for pattern in fenced_patterns:
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return json.loads(match.group(1))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def validate_pesudo_label_payload(payload: dict) -> dict[str, str]:
    pesudo_label = payload.get("pesudo_label")
    pesudo_summary = payload.get("pesudo_summary")
    if not isinstance(pesudo_label, str):
        raise ValueError("`pesudo_label` must be a string")
    if not isinstance(pesudo_summary, str):
        raise ValueError("`pesudo_summary` must be a string")
    pesudo_label = re.sub(r"\s+", " ", pesudo_label.strip()).strip(" -:;,.")
    pesudo_summary = re.sub(r"\s+", " ", pesudo_summary.strip()).strip(" -:;,.")
    if not pesudo_label:
        raise ValueError("`pesudo_label` cannot be empty")
    if not pesudo_summary:
        raise ValueError("`pesudo_summary` cannot be empty")
    return {"pesudo_label": pesudo_label, "pesudo_summary": pesudo_summary}


def validate_pesudo_seg_labels(payload: dict, expected_segments: int) -> list[dict[str, str]]:
    pesudo_seg = payload.get("pesudo_seg")
    if not isinstance(pesudo_seg, list):
        raise ValueError("`pesudo_seg` must be a list")
    if len(pesudo_seg) != expected_segments:
        raise ValueError(f"Expected {expected_segments} segments, got {len(pesudo_seg)}")
    cleaned: list[dict[str, str]] = []
    for item in pesudo_seg:
        if not isinstance(item, dict):
            raise ValueError("Each segment must be an object")
        cleaned.append(validate_pesudo_label_payload(item))
    return cleaned


def validate_pesudo_utterance_payload(payload: dict, segment_len: int) -> list[str]:
    pesudo_utterance = payload.get("pesudo_utterance")
    returned_len = payload.get("pesudo_seg_len")
    if not isinstance(pesudo_utterance, list):
        raise ValueError("`pesudo_utterance` must be a list")
    if len(pesudo_utterance) != segment_len:
        raise ValueError(f"Expected {segment_len} utterances, got {len(pesudo_utterance)}")
    if returned_len is not None and int(returned_len) != int(segment_len):
        raise ValueError("`pesudo_seg_len` must match the target length")
    cleaned: list[str] = []
    for item in pesudo_utterance:
        if not isinstance(item, str):
            raise ValueError("All utterances must be strings")
        text = strip_speaker_label(item)
        if not text:
            raise ValueError("Empty utterance is not allowed")
        cleaned.append(text)
    return cleaned


def generate_pesudo_label_with_openai(
    client,
    *,
    model: str,
    utterances: list[str],
    retries: int,
) -> tuple[dict[str, str] | None, str]:
    prompt_variants = [
        (
            "You label a dialogue segment. Return JSON only. Do not explain.",
            "Return a JSON object with exactly two keys: pesudo_label and pesudo_summary.\n"
            "Segment:\n"
            + "\n".join(f"- {utt}" for utt in utterances),
        ),
        (
            "You label dialogue segments. Return JSON only.",
            "Output JSON with exactly two keys: pesudo_label and pesudo_summary.\n"
            "Segment:\n"
            + "\n".join(f"- {utt}" for utt in utterances),
        ),
    ]
    for system_text, user_text in prompt_variants:
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        for _ in range(max(retries, 1)):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content or ""
            try:
                payload = extract_json_object(text)
                return validate_pesudo_label_payload(payload), text
            except Exception:
                continue
    return None, text if 'text' in locals() else ""


def generate_pesudo_seg_labels_with_openai(
    client,
    *,
    model: str,
    segments: list[list[str]],
    retries: int,
    debug_label: str = "",
) -> tuple[list[dict[str, str]] | None, str, str]:
    messages = [
        {
            "role": "system",
            "content": (
                "You label each dialogue segment with a pesudo label and a pesudo summary. "
                "Return JSON only. Do not explain. "
                "Return exactly one JSON object with a pesudo_seg key."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return JSON with exactly one key: pesudo_seg.\n"
                "pesudo_seg must be an array of objects, one per dialogue segment.\n"
                "Each object must have exactly two string keys: pesudo_label and pesudo_summary.\n"
                "The output must preserve the input segment order.\n"
                "The output must contain exactly the same number of segments as the input.\n"
                "Each output item must correspond to one input segment and must not merge or split segments.\n"
                "Input JSON:\n"
                + json.dumps(
                    {
                        "pesudo_seg": [
                            {
                                "pesudo_seg_index": idx,
                                "pesudo_seg_len": len(seg),
                                "utterances": seg,
                            }
                            for idx, seg in enumerate(segments)
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            ),
        },
    ]
    last_error = ""
    for attempt in range(1, max(retries, 1) + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or ""
        try:
            payload = extract_json_object(text)
            return validate_pesudo_seg_labels(payload, len(segments)), text, ""
        except Exception as exc:
            last_error = str(exc)
            label = f" for {debug_label}" if debug_label else ""
            tqdm.write(
                f"[pesudo label retry {attempt}/{max(retries, 1)}{label}] "
                f"{last_error}\nRaw output:\n{text}",
            )
            continue
    return None, text if 'text' in locals() else "", last_error


def generate_pesudo_seg_utterance_with_openai(
    client,
    *,
    model: str,
    pesudo_label: str,
    pesudo_summary: str,
    segment_len: int,
    style_hint: str,
    retries: int,
) -> tuple[list[str] | None, str]:
    messages = [
        {
            "role": "system",
            "content": (
                "You generate a natural two-speaker dialogue segment. "
                "Return JSON only. Do not explain. "
                "The JSON must contain exactly two keys: pesudo_utterance and pesudo_seg_len. "
                "The pesudo_utterance value must be a JSON array of plain strings only. "
                "Do not return speaker/text objects. "
                "Do not prefix any utterance with speaker names or labels such as "
                "VTS:, TRAPICHE EMERALD:, Speaker A:, speaker:, or 说话人:."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Pesudo label: {pesudo_label}\n"
                f"Pesudo summary: {pesudo_summary}\n"
                f"Target length: {segment_len} utterances\n"
                f"Style constraints: {style_hint}\n"
                'Return JSON with exactly two keys: {"pesudo_utterance": ["..."], "pesudo_seg_len": <int>}\n'
                "Every utterance must be a plain string. "
                "Do not prefix utterances with speaker identifiers, ship names, role tags, Speaker A/B, speaker:, or 说话人:. "
                "Do not output objects. Do not output explanations."
            ),
        },
    ]
    for _ in range(retries):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or ""
        try:
            payload = extract_json_object(text)
            return validate_pesudo_utterance_payload(payload, segment_len), text
        except Exception:
            continue
    return None, text if 'text' in locals() else ""


def generate_pesudo_utterance_with_openai(
    client,
    *,
    model: str,
    pesudo_summary: str,
    segment_len: int,
    style_hint: str,
    retries: int,
) -> tuple[list[str] | None, str]:
    return generate_pesudo_seg_utterance_with_openai(
        client,
        model=model,
        pesudo_label="generic",
        pesudo_summary=pesudo_summary,
        segment_len=segment_len,
        style_hint=style_hint,
        retries=retries,
    )


def validate_pesudo_dialogue_payload(payload: dict, expected_segments: list[int]) -> tuple[list[str], list[int]]:
    pesudo_utterance = payload.get("pesudo_utterance")
    pesudo_seg = payload.get("pesudo_seg")
    if not isinstance(pesudo_utterance, list):
        raise ValueError("`pesudo_utterance` must be a list")
    if not isinstance(pesudo_seg, list):
        raise ValueError("`pesudo_seg` must be a list")
    if len(pesudo_seg) != len(expected_segments):
        raise ValueError(f"Expected {len(expected_segments)} segments, got {len(pesudo_seg)}")
    cleaned_utterances: list[str] = []
    for item in pesudo_utterance:
        if not isinstance(item, str):
            raise ValueError("All utterances must be strings")
        text = strip_speaker_label(item)
        if not text:
            raise ValueError("Empty utterance is not allowed")
        cleaned_utterances.append(text)
    cleaned_segments: list[int] = []
    for seg_len, expected in zip(pesudo_seg, expected_segments):
        if int(seg_len) != int(expected):
            raise ValueError("Segment length mismatch")
        if int(seg_len) <= 0:
            raise ValueError("Segment lengths must be positive integers")
        cleaned_segments.append(int(seg_len))
    if sum(cleaned_segments) != len(cleaned_utterances):
        raise ValueError("Utterances count does not match segments")
    return cleaned_utterances, cleaned_segments


def generate_pesudo_dialogue_with_openai(
    client,
    *,
    model: str,
    pesudo_seg_specs: list[dict],
    retries: int,
    debug_label: str = "",
) -> tuple[tuple[list[str], list[int]] | None, str, str]:
    expected_segments = [int(spec["pesudo_seg_len"]) for spec in pesudo_seg_specs]
    total_target_utterances = int(sum(expected_segments))
    messages = [
        {
            "role": "system",
            "content": (
                "You generate a full coherent dialogue from segment specifications. "
                "Return JSON only. Do not explain. "
                "Return exactly two keys: pesudo_utterance and pesudo_seg."
            ),
        },
        {
            "role": "user",
            "content": (
                "Generate one continuous dialogue.\n"
                "Return JSON with exactly two keys: pesudo_utterance and pesudo_seg.\n"
                "pesudo_utterance must be a JSON array of plain strings.\n"
                "Do not prefix utterances with speaker identifiers, ship names, role tags, Speaker A/B, speaker:, or 说话人:.\n"
                "pesudo_seg must be a JSON array of integers, and each integer is a segment LENGTH, not a segment index.\n"
                "pesudo_seg must list the utterance counts for each segment in order.\n"
                "Each integer must match the requested target segment length.\n"
                f"The number of segments must be exactly {len(expected_segments)}.\n"
                f"The pesudo_seg array must be exactly {expected_segments}.\n"
                f"The total number of utterances must be exactly {total_target_utterances}.\n"
                "The pesudo_utterance array length must equal the sum of the pesudo_seg array.\n"
                "Input JSON:\n"
                + json.dumps(
                    {
                        "pesudo_seg_count": len(expected_segments),
                        "target_pesudo_seg": expected_segments,
                        "total_target_pesudo_utterance": total_target_utterances,
                        "pesudo_seg_specs": [
                            {
                                "pesudo_seg_index": idx,
                                "pesudo_label": spec["pesudo_label"],
                                "pesudo_summary": spec["pesudo_summary"],
                                "pesudo_seg_len": spec["pesudo_seg_len"],
                            }
                            for idx, spec in enumerate(pesudo_seg_specs)
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            ),
        },
    ]
    last_error = ""
    for attempt in range(1, max(retries, 1) + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or ""
        try:
            payload = extract_json_object(text)
            return validate_pesudo_dialogue_payload(payload, expected_segments), text, ""
        except Exception as exc:
            last_error = str(exc)
            label = f" for {debug_label}" if debug_label else ""
            tqdm.write(
                f"[dialogue rebuild retry {attempt}/{max(retries, 1)}{label}] "
                f"{last_error}\nTarget segments: {expected_segments}\nRaw output:\n{text}",
            )
            continue
    return None, text if 'text' in locals() else "", last_error


def summaries_to_candidate_matrix(summaries: list[str]):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(summaries)
    return vectorizer, matrix


def choose_replacement(
    *,
    candidates: list[dict],
    pesudo_summary: str,
    target_len: int,
    budget_remaining: int,
    source_dial_id: str,
    source_segment_index: int,
) -> dict | None:
    if not candidates:
        return None
    summaries = [c["pesudo_summary"] for c in candidates]
    vectorizer, matrix = summaries_to_candidate_matrix(summaries)
    target_vec = vectorizer.transform([pesudo_summary])
    scores = (matrix @ target_vec.T).toarray().ravel()
    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: (
            0 if item[0]["source_dial_id"] == source_dial_id else 1,
            abs(int(item[0]["segment_len"]) - int(target_len)),
            -float(item[1]),
            int(item[0]["source_segment_index"]),
        ),
    )
    for candidate, _ in ranked:
        if candidate["source_dial_id"] == source_dial_id and candidate["source_segment_index"] == source_segment_index:
            continue
        candidate_len = int(candidate["segment_len"])
        if candidate_len > budget_remaining:
            continue
        if candidate_len <= target_len and abs(candidate_len - int(target_len)) <= max(2, target_len // 2):
            return candidate
    return ranked[0][0]


def rebuild_synthetic_dialogue(
    source_segments: list[dict],
    generated_by_key: dict[tuple[str, int], dict],
    candidate_pool: list[dict],
    *,
    max_utterances: int,
) -> tuple[list[str], list[int], dict]:
    synthetic_utterances: list[str] = []
    synthetic_segments: list[int] = []
    same_source = 0
    cross_source = 0
    first_cross_used = False
    last_cross_used = False
    n_segments = len(source_segments)
    allow_cross = n_segments >= 4
    for idx, segment in enumerate(source_segments):
        remaining_budget = max_utterances - len(synthetic_utterances)
        remaining_min = sum(len(seg["utterances"]) for seg in source_segments[idx + 1 :])
        budget_for_current = max(remaining_budget - remaining_min, 1)
        key = (segment["source_dial_id"], segment["source_segment_index"])
        replacement = generated_by_key.get(key)
        is_cross = False
        if replacement is None:
            replacement = choose_replacement(
                candidates=candidate_pool,
                pesudo_summary=segment["pesudo_summary"],
                target_len=segment["segment_len"],
                budget_remaining=budget_for_current,
                source_dial_id=segment["source_dial_id"],
                source_segment_index=segment["source_segment_index"],
            )
            is_cross = replacement is not None and replacement["source_dial_id"] != segment["source_dial_id"]
        if replacement is None:
            replacement = segment
            is_cross = False

        if is_cross:
            if not allow_cross:
                replacement = segment
                is_cross = False
            elif cross_source >= 1:
                replacement = segment
                is_cross = False
            elif idx == 0 and last_cross_used:
                replacement = segment
                is_cross = False
            elif idx == n_segments - 1 and first_cross_used:
                replacement = segment
                is_cross = False

        chosen = replacement.get("utterances", segment["utterances"])
        if not isinstance(chosen, list) or not chosen:
            chosen = segment["utterances"]
            is_cross = False
        if len(chosen) > budget_for_current:
            chosen = segment["utterances"]
            is_cross = False
        synthetic_utterances.extend([str(u) for u in chosen])
        synthetic_segments.append(len(chosen))
        if is_cross:
            cross_source += 1
            if idx == 0:
                first_cross_used = True
            if idx == n_segments - 1:
                last_cross_used = True
        else:
            same_source += 1
        if len(synthetic_utterances) > max_utterances:
            raise ValueError("Synthetic dialogue exceeds max_utterances")

    if len(synthetic_segments) < 2:
        raise ValueError("Synthetic dialogue must keep at least 2 segments")
    if same_source / max(len(synthetic_segments), 1) < 0.7:
        raise ValueError("Synthetic dialogue violates same-source ratio")
    if cross_source > 1:
        raise ValueError("Synthetic dialogue violates cross-source cap")
    if first_cross_used and last_cross_used:
        raise ValueError("Synthetic dialogue violates first/last cross-source constraint")
    return synthetic_utterances, synthetic_segments, {
        "same_source_segments": same_source,
        "cross_source_segments": cross_source,
    }


def build_pesudo_summary_rows(
    summarizer: FlanT5Summarizer,
    segments: list[dict],
    *,
    desc: str,
    batch_size: int = 8,
) -> list[dict]:
    rows: list[dict] = []
    for start in tqdm(range(0, len(segments), batch_size), desc=desc, unit="batch"):
        batch = segments[start : start + batch_size]
        summaries = summarizer.summarize_batch([sample["utterances"] for sample in batch])
        for sample, summary in zip(batch, summaries):
            row = dict(sample)
            row["pesudo_summary"] = summary
            rows.append(row)
    return rows


def filter_pesudo_seg_rows(
    segment_rows: list[dict],
    *,
    min_segment_len: int,
    max_segment_len: int = 10,
) -> list[dict]:
    if max_segment_len <= 0:
        raise ValueError("max_segment_len must be positive")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in segment_rows:
        grouped[str(row["source_dial_id"])].append(row)

    filtered: list[dict] = []
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["source_segment_index"]))
        for row in rows:
            segment_len = int(row["segment_len"])
            if segment_len >= int(min_segment_len) and segment_len < int(max_segment_len):
                filtered.append(row)
    return filtered
