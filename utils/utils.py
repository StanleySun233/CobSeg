import json
import os
from decimal import Decimal
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv


def parse_llm_response(response_text):
    if "```json" in response_text:
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        response_text = response_text[start:end].strip()
    elif "```" in response_text:
        start = response_text.find("```") + 3
        end = response_text.find("```", start)
        response_text = response_text[start:end].strip()

    parsed_response = json.loads(response_text)
    if "result" in parsed_response:
        return parsed_response["result"]
    else:
        return parsed_response


def convert_numpy_types(obj):
    # Handle numpy types
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle Decimal types (from segeval library)
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        # Recursively process dictionary, but handle numeric string values
        result = {}
        for key, value in obj.items():
            # For metrics keys (PK, WD, Precision, Recall, F1), ensure numeric values are floats
            if key in ['PK', 'WD', 'Precision', 'Recall', 'F1'] and isinstance(value, str):
                try:
                    result[key] = float(value)
                except (ValueError, TypeError):
                    result[key] = convert_numpy_types(value)
            else:
                result[key] = convert_numpy_types(value)
        return result
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def convert_segments_to_boundary(segments, total_length):
    boundary = [0] * total_length

    current_pos = 0
    for i, segment_length in enumerate(segments):
        if segment_length > 0:
            current_pos += segment_length
            if i < len(segments) - 1 and current_pos < total_length:
                boundary[current_pos - 1] = 1

    return boundary


def convert_predictions_to_boundary(predictions, total_length=None):
    boundary = []

    for i, pred in enumerate(predictions):
        # 第一个位置永远不应该是分割点（对话开始前没有边界）
        if i == 0:
            boundary.append(0)
            continue

        if isinstance(pred, dict) and pred.get('success', False):
            parsed = pred.get('parsed_response', {})
            if parsed and parsed.get('result') == 'SEGMENT':
                boundary.append(1)
            else:
                boundary.append(0)
        else:
            boundary.append(0)

    return boundary


def load_config(path="config.yaml"):
    env_path = Path(path).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=False)
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
        # Expand environment variables (supports $VAR and ${VAR} formats)
        content = os.path.expandvars(content)
        config = yaml.load(content, Loader=yaml.FullLoader)
    return config


def _normalize_ollama_base_url(base_url: str) -> str:
    if not isinstance(base_url, str):
        return base_url
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def resolve_llm_settings(config, provider, model_override=None):
    supported = ("ollama", "openai", "openrouter")
    if provider not in supported:
        raise ValueError(f"Unsupported llm provider: {provider}")

    api_key = config["api_key"][provider]
    base_url = config["base_url"][provider]
    if provider == "ollama":
        base_url = _normalize_ollama_base_url(base_url)

    model_value = model_override if model_override else config["model"][provider]
    if isinstance(model_value, list):
        if not model_value:
            raise ValueError(f"No model configured for provider: {provider}")
        model_name = model_value[0]
    else:
        model_name = model_value

    return api_key, base_url, model_name


def resolve_embedding_settings(config, provider, model_override=None):
    supported = ("ollama", "openai", "openrouter", "sentence-transformers")
    if provider not in supported:
        raise ValueError(f"Unsupported embedding provider: {provider}")

    if provider == "sentence-transformers":
        api_key = ""
        base_url = ""
    else:
        api_key = config["api_key"][provider]
        base_url = config["base_url"][provider]
        if provider == "ollama":
            base_url = _normalize_ollama_base_url(base_url)

    model_value = model_override if model_override else config["embeddings_model"][provider]
    if isinstance(model_value, list):
        if not model_value:
            raise ValueError(f"No embedding model configured for provider: {provider}")
        model_name = model_value[0]
    else:
        model_name = model_value

    return api_key, base_url, model_name


def resolve_dataset_path(dataset_name_or_path: str) -> str:
    if not dataset_name_or_path:
        dataset_name_or_path = 'vfh'
    if dataset_name_or_path.lower().endswith('.json'):
        return dataset_name_or_path
    mapping = {
        'vfh': './data/dataset/vhf.json',
        'dialseg711': './data/dataset/dialseg711.json',
        'doc2dial': './data/dataset/doc2dial.json',
        'tiage': './data/dataset/tiage.json',
        'superseg': './data/dataset/superseg.json',
        'vhf_pseudo_100': './data/dataset/vhf_pseudo_100.json',
        'dialseg711_pseudo_100': './data/dataset/dialseg711_pseudo_100.json',
    }
    key = dataset_name_or_path.lower()
    return mapping.get(key, mapping['vfh'])
