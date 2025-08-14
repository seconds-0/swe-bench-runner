import json

from swebench_runner import model_catalog as mc


def test_sort_openai_by_created_then_id(monkeypatch):
    payload = {
        "data": [
            {"id": "gpt-X-2024-05-01", "created": 1714521600},
            {"id": "gpt-A-2024-05-01", "created": 1714521600},
            {"id": "gpt-Y-2023-12-01", "created": 1701388800},
            {"id": "older-model"},
        ]
    }

    class R:
        status_code = 200
        text = json.dumps(payload)
        def json(self):
            return payload

    monkeypatch.setattr(mc, "requests", type("Rq", (), {"get": lambda *a, **k: R()}))

    ids = mc.list_openai_models("secret")
    # Newest (same date) sorted by id, then older, then fallback id
    assert ids[:3] == ["gpt-A-2024-05-01", "gpt-X-2024-05-01", "gpt-Y-2023-12-01"]
    assert ids[-1] == "older-model"


def test_sort_anthropic_by_date_in_id(monkeypatch):
    payload = {
        "data": [
            {"id": "claude-3-5-sonnet-2024-07-18"},
            {"id": "claude-3-opus-2024-02-29"},
            {"id": "claude-2"},
        ]
    }

    class R:
        status_code = 200
        text = json.dumps(payload)
        def json(self):
            return payload

    monkeypatch.setattr(mc, "requests", type("Rq", (), {"get": lambda *a, **k: R()}))

    ids = mc.list_anthropic_models("secret")
    assert ids[0].startswith("claude-3-5-sonnet-2024-07-18")
    assert ids[1].startswith("claude-3-opus-2024-02-29")
    assert ids[-1] == "claude-2"


def test_sort_openrouter_by_timestamp_or_id(monkeypatch):
    payload = {
        "data": [
            {"id": "openrouter/some-model-2024-06-01", "updated_at": "2024-06-02T00:00:00Z"},
            {"id": "openrouter/older-2023-01-01"},
            {"id": "openrouter/alpha"},
        ]
    }

    class R:
        status_code = 200
        text = json.dumps(payload)
        def json(self):
            return payload

    monkeypatch.setattr(mc, "requests", type("Rq", (), {"get": lambda *a, **k: R()}))

    ids = mc.list_openrouter_models(None)
    assert ids[0].startswith("openrouter/some-model-2024-06-01")
    assert ids[-1] == "openrouter/alpha"
