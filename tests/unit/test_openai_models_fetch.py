import asyncio
import json
from unittest import mock

import pytest

from swebench_runner.providers.base import ProviderConfig
from swebench_runner.providers.openai import OpenAIProvider


@pytest.mark.asyncio
async def test_openai_models_fetch_on_init_default_base(monkeypatch):
    # Ensure default base URL (no override) and allow fetch (default behavior)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("SWEBENCH_OPENAI_FETCH_MODELS", raising=False)

    config = ProviderConfig(name="openai", api_key="test_key", model="gpt-4o")

    class DummyResp:
        def __init__(self, status: int, payload: dict):
            self.status = status
            self._payload = payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self):
            return self._payload

        async def text(self):
            return json.dumps(self._payload)

    class DummySession:
        def __init__(self, *_, **__):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url, headers=None):
            assert url == "https://api.openai.com/v1/models"
            return DummyResp(200, {"data": [{"id": "gpt-5-nano"}]})

    with mock.patch("aiohttp.ClientSession", DummySession):
        provider = OpenAIProvider(config)
        # Let scheduled task run
        await asyncio.sleep(0.05)
        assert "gpt-5-nano" in provider.supported_models


@pytest.mark.asyncio
async def test_openai_models_fetch_on_init_disabled(monkeypatch):
    # Disable fetch explicitly
    monkeypatch.setenv("SWEBENCH_OPENAI_FETCH_MODELS", "0")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    config = ProviderConfig(
        name="openai", api_key="test_key", model="gpt-4o",
        endpoint="https://custom.example.com/v1"
    )

    # Ensure no network call by asserting ClientSession is not created
    with mock.patch("aiohttp.ClientSession") as session_mock:
        OpenAIProvider(config)
        await asyncio.sleep(0.02)
        session_mock.assert_not_called()


@pytest.mark.asyncio
async def test_openai_models_fetch_on_init_non_default_base_requires_opt_in(monkeypatch):
    # Custom base URL without opt-in should not fetch
    monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.example.com/v1")
    monkeypatch.delenv("SWEBENCH_OPENAI_FETCH_MODELS", raising=False)

    config = ProviderConfig(
        name="openai", api_key="test_key", model="gpt-4o",
        endpoint="https://custom.example.com/v1"
    )

    with mock.patch("aiohttp.ClientSession") as session_mock:
        OpenAIProvider(config)
        await asyncio.sleep(0.02)
        session_mock.assert_not_called()

    # With opt-in it should fetch
    monkeypatch.setenv("SWEBENCH_OPENAI_FETCH_MODELS", "1")

    class DummyResp2:
        def __init__(self, status: int, payload: dict):
            self.status = status
            self._payload = payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self):
            return self._payload

        async def text(self):
            return json.dumps(self._payload)

    class DummySession2:
        def __init__(self, *_, **__):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url, headers=None):
            assert url == "https://custom.example.com/v1/models"
            return DummyResp2(200, {"data": [{"id": "gpt-5-nano-2"}]})

    with mock.patch("aiohttp.ClientSession", DummySession2):
        provider2 = OpenAIProvider(config)
        await asyncio.sleep(0.05)
        assert "gpt-5-nano-2" in provider2.supported_models
