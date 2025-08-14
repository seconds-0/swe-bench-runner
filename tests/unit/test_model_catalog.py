import json

import pytest

from swebench_runner import model_catalog


class DummyResponse:
    def __init__(self, status_code: int, payload: dict | list):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


@pytest.mark.parametrize(
    "func,url_key,headers_key,ok_payload,expected",
    [
        (
            model_catalog.list_openai_models,
            "https://api.openai.com/v1/models",
            "Authorization",
            {"data": [{"id": "gpt-4o"}, {"id": "o3"}]},
            ["gpt-4o", "o3"],
        ),
        (
            model_catalog.list_anthropic_models,
            "https://api.anthropic.com/v1/models",
            "x-api-key",
            {"data": [{"id": "claude-3-5-sonnet"}]},
            ["claude-3-5-sonnet"],
        ),
    ],
)
def test_model_list_success(monkeypatch, func, url_key, headers_key, ok_payload, expected):
    calls: list[tuple[str, dict]] = []

    def fake_get(url: str, headers: dict, timeout: int):
        calls.append((url, headers))
        return DummyResponse(200, ok_payload)

    monkeypatch.setattr(model_catalog, "requests", type("R", (), {"get": fake_get}))

    out: list[str] = func("secret")
    assert out == sorted(expected)
    assert calls and calls[0][0] == url_key
    assert headers_key in calls[0][1]


def test_openrouter_list_success(monkeypatch):
    def fake_get(url: str, headers: dict, timeout: int):
        assert url == "https://openrouter.ai/api/v1/models"
        payload = {"data": [{"id": "openrouter/anything"}]}
        return DummyResponse(200, payload)

    monkeypatch.setattr(model_catalog, "requests", type("R", (), {"get": fake_get}))
    out = model_catalog.list_openrouter_models("maybe")
    assert out == ["openrouter/anything"]


@pytest.mark.parametrize("func", [
    model_catalog.list_openai_models,
    model_catalog.list_anthropic_models,
])
def test_model_list_error(monkeypatch, func):
    def fake_get(url: str, headers: dict, timeout: int):
        return DummyResponse(401, {"error": "unauthorized"})

    monkeypatch.setattr(model_catalog, "requests", type("R", (), {"get": fake_get}))

    with pytest.raises(model_catalog.ModelCatalogError):
        func("bad-secret")
