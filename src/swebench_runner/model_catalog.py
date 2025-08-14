from __future__ import annotations

from datetime import datetime
from typing import Any

import requests


class ModelCatalogError(Exception):
    pass


def _parse_date_from_id(model_id: str) -> datetime | None:
    """Parse a YYYY-MM-DD suffix anywhere in the id string."""
    try:
        # Common pattern: ...-2024-12-17
        parts = model_id.split("-")
        for i in range(len(parts) - 2):
            token = "-".join(parts[i:i+3])
            if len(token) == 10 and token[4] == "-" and token[7] == "-":
                return datetime.strptime(token, "%Y-%m-%d")
    except Exception:
        return None
    return None


def _sort_models_by_recentness(items: list[dict[str, Any]], id_key: str = "id",
                               created_key_candidates: list[str] | None = None) -> list[str]:
    created_key_candidates = created_key_candidates or ["created", "updated", "created_at", "updated_at"]

    def extract_created(obj: dict[str, Any]) -> datetime | None:
        # Prefer explicit timestamps
        for key in created_key_candidates:
            if key in obj and obj[key] is not None:
                try:
                    val = obj[key]
                    # Accept unix seconds or iso8601
                    if isinstance(val, (int, float)):
                        return datetime.fromtimestamp(int(val))
                    if isinstance(val, str):
                        # Try ISO
                        return datetime.fromisoformat(val.replace("Z", "+00:00"))
                except Exception:
                    continue
        # Fallback: parse from id
        model_id = obj.get(id_key, "")
        return _parse_date_from_id(model_id)

    def sort_key(obj: dict[str, Any]):
        created = extract_created(obj)
        # Newest first => sort by negative timestamp; None goes last
        ts = created.timestamp() if created else float("-inf")
        # For None created, push very old by using extremely low ts
        if created is None:
            ts = -10**18
        return (-ts, obj.get(id_key, ""))

    # Sort and return ids
    return [obj.get(id_key, "") for obj in sorted(items, key=sort_key)]


def list_openai_models(api_key: str) -> list[str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    r = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=15)
    if r.status_code != 200:
        raise ModelCatalogError(f"OpenAI list models failed: {r.status_code} {r.text[:200]}")
    data = r.json()
    items = [m for m in data.get("data", []) if isinstance(m, dict)]
    # Sort by created desc, then id
    return _sort_models_by_recentness(items)


def list_anthropic_models(api_key: str) -> list[str]:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    r = requests.get("https://api.anthropic.com/v1/models", headers=headers, timeout=15)
    if r.status_code != 200:
        raise ModelCatalogError(f"Anthropic list models failed: {r.status_code} {r.text[:200]}")
    data = r.json()
    items = [m for m in data.get("data", []) if isinstance(m, dict)]
    # Anthropic does not reliably include created timestamps.
    # Many ids contain a YYYY-MM-DD suffix; parse that, else fall back to id.
    return _sort_models_by_recentness(items)


essential_fields = ("id",)


def list_openrouter_models(api_key: str | None) -> list[str]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=15)
    if r.status_code != 200:
        raise ModelCatalogError(f"OpenRouter list models failed: {r.status_code} {r.text[:200]}")
    data = r.json()
    items = [m for m in data.get("data", []) if isinstance(m, dict)]
    # OpenRouter sometimes includes created/updated; otherwise parse date in id
    return _sort_models_by_recentness(items)
