def _short_model_name(model_name: str) -> str:
    if model_name is None:
        return ""
    return str(model_name).split("/")[-1]


def resolve_llm_pricing(config: dict, provider: str, model_name: str) -> dict | None:
    if not config or not provider or not model_name:
        return None

    pricing_root = config.get("llm_pricing") or {}
    provider_pricing = pricing_root.get(provider) or {}
    if not isinstance(provider_pricing, dict):
        return None

    if model_name in provider_pricing:
        v = provider_pricing.get(model_name)
        return v if isinstance(v, dict) else None

    short = _short_model_name(model_name)
    if short in provider_pricing:
        v = provider_pricing.get(short)
        return v if isinstance(v, dict) else None

    return None


def compute_cost_usd(input_tokens: int, output_tokens: int, pricing: dict | None) -> float | None:
    if pricing is None:
        return None

    in_price = pricing.get("input_price_per_1m", None)
    out_price = pricing.get("output_price_per_1m", None)
    if in_price is None and out_price is None:
        return None

    in_price_f = float(in_price or 0)
    out_price_f = float(out_price or 0)
    in_f = int(input_tokens or 0)
    out_f = int(output_tokens or 0)

    return round((in_f / 1_000_000) * in_price_f + (out_f / 1_000_000) * out_price_f, 8)


def sum_tokens(items: list) -> tuple[int, int]:
    ti = 0
    to = 0
    for it in items or []:
        if isinstance(it, dict):
            ti += int(it.get("input_tokens", 0) or 0)
            to += int(it.get("output_tokens", 0) or 0)
    return ti, to


def sum_dialogue_tokens(dialogues: list) -> tuple[int, int]:
    ti = 0
    to = 0
    for dialogue in dialogues or []:
        if isinstance(dialogue, list):
            d_ti, d_to = sum_tokens(dialogue)
            ti += d_ti
            to += d_to
    return ti, to

