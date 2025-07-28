"""Configuration for integration tests.

This module sets up the infrastructure for running true integration tests
that make real API calls to external services. Tests are marked with
@pytest.mark.integration and are skipped by default unless explicitly run.
"""

import os

import pytest


def pytest_configure(config):
    """Register integration marker."""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test (requires API keys)"
    )


@pytest.fixture
def skip_without_openai_key():
    """Skip test if OpenAI API key is not available."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set - skipping OpenAI integration test")


@pytest.fixture
def skip_without_anthropic_key():
    """Skip test if Anthropic API key is not available."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set - skipping Anthropic integration test")


@pytest.fixture
def skip_without_ollama():
    """Skip test if Ollama is not running."""
    import asyncio

    import aiohttp

    async def check_ollama():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    return resp.status == 200
        except:
            return False

    if not asyncio.run(check_ollama()):
        pytest.skip("Ollama not running on localhost:11434 - skipping Ollama integration test")


@pytest.fixture
def openai_test_model() -> str:
    """Get the OpenAI model to use for testing."""
    return os.environ.get("OPENAI_TEST_MODEL", "gpt-3.5-turbo")


@pytest.fixture
def anthropic_test_model() -> str:
    """Get the Anthropic model to use for testing."""
    return os.environ.get("ANTHROPIC_TEST_MODEL", "claude-3-haiku-20240307")


@pytest.fixture
def ollama_test_model() -> str:
    """Get the Ollama model to use for testing."""
    return os.environ.get("OLLAMA_TEST_MODEL", "llama3.2:1b")


@pytest.fixture
def minimal_test_prompt() -> str:
    """Get a minimal prompt for cost-effective testing."""
    return "Say 'test' and nothing else."


@pytest.fixture
def streaming_test_prompt() -> str:
    """Get a prompt for testing streaming responses."""
    return "Count from 1 to 5, one number per line."


@pytest.fixture
def cost_test_prompt() -> str:
    """Get a prompt with predictable token count for cost testing."""
    # This prompt should generate ~10-15 tokens in response
    return "Complete this sentence with exactly 5 words: The sky is"


@pytest.fixture
def error_test_prompt() -> str:
    """Get a prompt that's safe but tests error boundaries."""
    return "a" * 10000  # Very long prompt to potentially trigger token limits
