"""Tests for authentication strategies module."""

from __future__ import annotations

import pytest

from swebench_runner.providers.auth_strategies import (
    ApiKeyAuth,
    AuthConfig,
    AuthStrategyFactory,
    AuthType,
    BearerTokenAuth,
    NoAuth,
)


class TestAuthConfig:
    """Test AuthConfig dataclass."""

    def test_valid_bearer_token_config(self) -> None:
        """Test creating valid bearer token config."""
        config = AuthConfig(
            auth_type=AuthType.BEARER_TOKEN,
            credentials={"api_key": "sk-test123"}
        )

        assert config.auth_type == AuthType.BEARER_TOKEN
        assert config.credentials == {"api_key": "sk-test123"}
        assert config.additional_headers is None

    def test_valid_config_with_additional_headers(self) -> None:
        """Test config with additional headers."""
        additional_headers = {"Custom-Header": "value"}
        config = AuthConfig(
            auth_type=AuthType.API_KEY,
            credentials={"api_key": "test-key"},
            additional_headers=additional_headers
        )

        assert config.additional_headers == additional_headers

    def test_none_auth_with_empty_credentials(self) -> None:
        """Test that NoAuth can have empty credentials."""
        config = AuthConfig(
            auth_type=AuthType.NONE,
            credentials={}
        )

        assert config.auth_type == AuthType.NONE
        assert config.credentials == {}

    def test_bearer_token_requires_credentials(self) -> None:
        """Test that bearer token auth requires credentials."""
        with pytest.raises(ValueError, match="Credentials required for bearer_token auth"):
            AuthConfig(
                auth_type=AuthType.BEARER_TOKEN,
                credentials={}
            )

    def test_api_key_requires_credentials(self) -> None:
        """Test that API key auth requires credentials."""
        with pytest.raises(ValueError, match="Credentials required for api_key auth"):
            AuthConfig(
                auth_type=AuthType.API_KEY,
                credentials={}
            )


class TestBearerTokenAuth:
    """Test BearerTokenAuth strategy."""

    def test_prepare_headers_basic(self) -> None:
        """Test basic header preparation with bearer token."""
        config = AuthConfig(
            auth_type=AuthType.BEARER_TOKEN,
            credentials={"api_key": "sk-test123"}
        )
        auth = BearerTokenAuth(config)

        base_headers = {"Content-Type": "application/json"}
        result = auth.prepare_headers(base_headers)

        expected = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-test123"
        }
        assert result == expected

    def test_prepare_headers_with_organization(self) -> None:
        """Test header preparation with OpenAI organization ID."""
        config = AuthConfig(
            auth_type=AuthType.BEARER_TOKEN,
            credentials={
                "api_key": "sk-test123",
                "organization_id": "org-test456"
            }
        )
        auth = BearerTokenAuth(config)

        result = auth.prepare_headers({})

        assert result["Authorization"] == "Bearer sk-test123"
        assert result["OpenAI-Organization"] == "org-test456"

    def test_prepare_headers_with_project(self) -> None:
        """Test header preparation with OpenAI project ID."""
        config = AuthConfig(
            auth_type=AuthType.BEARER_TOKEN,
            credentials={
                "api_key": "sk-test123",
                "project_id": "proj-test789"
            }
        )
        auth = BearerTokenAuth(config)

        result = auth.prepare_headers({})

        assert result["Authorization"] == "Bearer sk-test123"
        assert result["OpenAI-Project"] == "proj-test789"

    def test_prepare_headers_with_additional_headers(self) -> None:
        """Test header preparation with additional headers."""
        config = AuthConfig(
            auth_type=AuthType.BEARER_TOKEN,
            credentials={"api_key": "sk-test123"},
            additional_headers={"Custom-Header": "custom-value"}
        )
        auth = BearerTokenAuth(config)

        result = auth.prepare_headers({"Existing": "header"})

        expected = {
            "Existing": "header",
            "Authorization": "Bearer sk-test123",
            "Custom-Header": "custom-value"
        }
        assert result == expected

    def test_prepare_headers_missing_api_key(self) -> None:
        """Test error when API key is missing."""
        config = AuthConfig(
            auth_type=AuthType.BEARER_TOKEN,
            credentials={}  # This will be allowed by __post_init__ but fail in prepare_headers
        )
        # Override the validation to test the specific error
        config.credentials = {}
        auth = BearerTokenAuth(config)

        with pytest.raises(ValueError, match="API key required for bearer token auth"):
            auth.prepare_headers({})

    def test_validate_credentials_valid(self) -> None:
        """Test credential validation with valid API key."""
        config = AuthConfig(
            auth_type=AuthType.BEARER_TOKEN,
            credentials={"api_key": "sk-test123"}
        )
        auth = BearerTokenAuth(config)

        assert auth.validate_credentials() is True

    def test_validate_credentials_missing(self) -> None:
        """Test credential validation with missing API key."""
        config = AuthConfig(
            auth_type=AuthType.BEARER_TOKEN,
            credentials={}
        )
        # Override to test validation
        config.credentials = {}
        auth = BearerTokenAuth(config)

        assert auth.validate_credentials() is False

    def test_validate_credentials_empty(self) -> None:
        """Test credential validation with empty API key."""
        config = AuthConfig(
            auth_type=AuthType.BEARER_TOKEN,
            credentials={"api_key": ""}
        )
        # Override to test validation
        config.credentials = {"api_key": ""}
        auth = BearerTokenAuth(config)

        assert auth.validate_credentials() is False

    def test_validate_credentials_whitespace(self) -> None:
        """Test credential validation with whitespace-only API key."""
        config = AuthConfig(
            auth_type=AuthType.BEARER_TOKEN,
            credentials={"api_key": "   "}
        )
        # Override to test validation
        config.credentials = {"api_key": "   "}
        auth = BearerTokenAuth(config)

        assert auth.validate_credentials() is False


class TestApiKeyAuth:
    """Test ApiKeyAuth strategy."""

    def test_prepare_headers_basic(self) -> None:
        """Test basic header preparation with API key."""
        config = AuthConfig(
            auth_type=AuthType.API_KEY,
            credentials={"api_key": "sk-ant-test123"}
        )
        auth = ApiKeyAuth(config)

        base_headers = {"Content-Type": "application/json"}
        result = auth.prepare_headers(base_headers)

        expected = {
            "Content-Type": "application/json",
            "x-api-key": "sk-ant-test123",
            "anthropic-version": "2023-06-01"
        }
        assert result == expected

    def test_prepare_headers_with_beta_features(self) -> None:
        """Test header preparation with Anthropic beta features."""
        config = AuthConfig(
            auth_type=AuthType.API_KEY,
            credentials={
                "api_key": "sk-ant-test123",
                "beta_features": "tools-2024-04-04"
            }
        )
        auth = ApiKeyAuth(config)

        result = auth.prepare_headers({})

        assert result["x-api-key"] == "sk-ant-test123"
        assert result["anthropic-version"] == "2023-06-01"
        assert result["anthropic-beta"] == "tools-2024-04-04"

    def test_prepare_headers_with_additional_headers(self) -> None:
        """Test header preparation with additional headers."""
        config = AuthConfig(
            auth_type=AuthType.API_KEY,
            credentials={"api_key": "sk-ant-test123"},
            additional_headers={"Custom-Header": "custom-value"}
        )
        auth = ApiKeyAuth(config)

        result = auth.prepare_headers({"Existing": "header"})

        assert result["x-api-key"] == "sk-ant-test123"
        assert result["anthropic-version"] == "2023-06-01"
        assert result["Custom-Header"] == "custom-value"
        assert result["Existing"] == "header"

    def test_prepare_headers_missing_api_key(self) -> None:
        """Test error when API key is missing."""
        config = AuthConfig(
            auth_type=AuthType.API_KEY,
            credentials={}
        )
        # Override to test the specific error
        config.credentials = {}
        auth = ApiKeyAuth(config)

        with pytest.raises(ValueError, match="API key required for API key auth"):
            auth.prepare_headers({})

    def test_validate_credentials_valid(self) -> None:
        """Test credential validation with valid API key."""
        config = AuthConfig(
            auth_type=AuthType.API_KEY,
            credentials={"api_key": "sk-ant-test123"}
        )
        auth = ApiKeyAuth(config)

        assert auth.validate_credentials() is True

    def test_validate_credentials_missing(self) -> None:
        """Test credential validation with missing API key."""
        config = AuthConfig(
            auth_type=AuthType.API_KEY,
            credentials={}
        )
        # Override to test validation
        config.credentials = {}
        auth = ApiKeyAuth(config)

        assert auth.validate_credentials() is False


class TestNoAuth:
    """Test NoAuth strategy."""

    def test_prepare_headers_basic(self) -> None:
        """Test basic header preparation without authentication."""
        config = AuthConfig(
            auth_type=AuthType.NONE,
            credentials={}
        )
        auth = NoAuth(config)

        base_headers = {"Content-Type": "application/json"}
        result = auth.prepare_headers(base_headers)

        expected = {"Content-Type": "application/json"}
        assert result == expected

    def test_prepare_headers_with_additional_headers(self) -> None:
        """Test header preparation with additional headers."""
        config = AuthConfig(
            auth_type=AuthType.NONE,
            credentials={},
            additional_headers={"Custom-Header": "custom-value"}
        )
        auth = NoAuth(config)

        result = auth.prepare_headers({"Existing": "header"})

        expected = {
            "Existing": "header",
            "Custom-Header": "custom-value"
        }
        assert result == expected

    def test_validate_credentials_always_true(self) -> None:
        """Test that credential validation always returns True."""
        config = AuthConfig(
            auth_type=AuthType.NONE,
            credentials={}
        )
        auth = NoAuth(config)

        assert auth.validate_credentials() is True


class TestAuthStrategyFactory:
    """Test AuthStrategyFactory."""

    def test_create_bearer_token_strategy(self) -> None:
        """Test creating bearer token strategy."""
        config = AuthConfig(
            auth_type=AuthType.BEARER_TOKEN,
            credentials={"api_key": "sk-test123"}
        )

        strategy = AuthStrategyFactory.create(config)

        assert isinstance(strategy, BearerTokenAuth)
        assert strategy.config == config

    def test_create_api_key_strategy(self) -> None:
        """Test creating API key strategy."""
        config = AuthConfig(
            auth_type=AuthType.API_KEY,
            credentials={"api_key": "sk-ant-test123"}
        )

        strategy = AuthStrategyFactory.create(config)

        assert isinstance(strategy, ApiKeyAuth)
        assert strategy.config == config

    def test_create_no_auth_strategy(self) -> None:
        """Test creating no auth strategy."""
        config = AuthConfig(
            auth_type=AuthType.NONE,
            credentials={}
        )

        strategy = AuthStrategyFactory.create(config)

        assert isinstance(strategy, NoAuth)
        assert strategy.config == config

    def test_create_unsupported_auth_type(self) -> None:
        """Test error with unsupported auth type."""
        # Create a mock enum value that doesn't exist
        class FakeAuthType:
            value = "fake_auth"

        config = AuthConfig(
            auth_type=FakeAuthType(),  # type: ignore
            credentials={"key": "value"}
        )

        with pytest.raises(ValueError, match="Unsupported auth type"):
            AuthStrategyFactory.create(config)

    def test_create_from_provider_openai(self) -> None:
        """Test creating strategy for OpenAI provider."""
        credentials = {"api_key": "sk-test123"}
        strategy = AuthStrategyFactory.create_from_provider("openai", credentials)

        assert isinstance(strategy, BearerTokenAuth)
        assert strategy.config.auth_type == AuthType.BEARER_TOKEN
        assert strategy.config.credentials == credentials

    def test_create_from_provider_openai_case_insensitive(self) -> None:
        """Test creating strategy for OpenAI provider (case insensitive)."""
        credentials = {"api_key": "sk-test123"}
        strategy = AuthStrategyFactory.create_from_provider("OpenAI", credentials)

        assert isinstance(strategy, BearerTokenAuth)
        assert strategy.config.auth_type == AuthType.BEARER_TOKEN

    def test_create_from_provider_anthropic(self) -> None:
        """Test creating strategy for Anthropic provider."""
        credentials = {"api_key": "sk-ant-test123"}
        strategy = AuthStrategyFactory.create_from_provider("anthropic", credentials)

        assert isinstance(strategy, ApiKeyAuth)
        assert strategy.config.auth_type == AuthType.API_KEY
        assert strategy.config.credentials == credentials

    def test_create_from_provider_ollama(self) -> None:
        """Test creating strategy for Ollama provider."""
        credentials = {}  # Ollama doesn't need credentials
        strategy = AuthStrategyFactory.create_from_provider("ollama", credentials)

        assert isinstance(strategy, NoAuth)
        assert strategy.config.auth_type == AuthType.NONE
        assert strategy.config.credentials == {}

    def test_create_from_provider_with_additional_headers(self) -> None:
        """Test creating strategy with additional headers."""
        credentials = {"api_key": "sk-test123"}
        additional_headers = {"Custom-Header": "value"}

        strategy = AuthStrategyFactory.create_from_provider(
            "openai",
            credentials,
            additional_headers
        )

        assert strategy.config.additional_headers == additional_headers

    def test_create_from_provider_unknown(self) -> None:
        """Test error with unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider: unknown"):
            AuthStrategyFactory.create_from_provider("unknown", {})

    def test_get_supported_auth_types(self) -> None:
        """Test getting supported auth types."""
        supported_types = AuthStrategyFactory.get_supported_auth_types()

        expected = [AuthType.BEARER_TOKEN, AuthType.API_KEY, AuthType.NONE]
        assert set(supported_types) == set(expected)

    def test_get_supported_providers(self) -> None:
        """Test getting supported providers."""
        supported_providers = AuthStrategyFactory.get_supported_providers()

        expected = ["openai", "anthropic", "ollama"]
        assert set(supported_providers) == set(expected)


class TestAuthStrategyIntegration:
    """Integration tests for authentication strategies."""

    def test_openai_full_workflow(self) -> None:
        """Test complete workflow for OpenAI authentication."""
        # Create strategy
        strategy = AuthStrategyFactory.create_from_provider(
            "openai",
            {
                "api_key": "sk-test123",
                "organization_id": "org-456",
                "project_id": "proj-789"
            }
        )

        # Validate credentials
        assert strategy.validate_credentials() is True

        # Prepare headers
        base_headers = {"Content-Type": "application/json"}
        headers = strategy.prepare_headers(base_headers)

        expected = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-test123",
            "OpenAI-Organization": "org-456",
            "OpenAI-Project": "proj-789"
        }
        assert headers == expected

    def test_anthropic_full_workflow(self) -> None:
        """Test complete workflow for Anthropic authentication."""
        # Create strategy
        strategy = AuthStrategyFactory.create_from_provider(
            "anthropic",
            {
                "api_key": "sk-ant-test123",
                "beta_features": "tools-2024-04-04"
            }
        )

        # Validate credentials
        assert strategy.validate_credentials() is True

        # Prepare headers
        base_headers = {"Content-Type": "application/json"}
        headers = strategy.prepare_headers(base_headers)

        expected = {
            "Content-Type": "application/json",
            "x-api-key": "sk-ant-test123",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "tools-2024-04-04"
        }
        assert headers == expected

    def test_ollama_full_workflow(self) -> None:
        """Test complete workflow for Ollama authentication."""
        # Create strategy
        strategy = AuthStrategyFactory.create_from_provider(
            "ollama",
            {},
            additional_headers={"Custom-Header": "local-api"}
        )

        # Validate credentials
        assert strategy.validate_credentials() is True

        # Prepare headers
        base_headers = {"Content-Type": "application/json"}
        headers = strategy.prepare_headers(base_headers)

        expected = {
            "Content-Type": "application/json",
            "Custom-Header": "local-api"
        }
        assert headers == expected

    def test_multiple_strategies_isolation(self) -> None:
        """Test that multiple strategies don't interfere with each other."""
        # Create different strategies
        openai_strategy = AuthStrategyFactory.create_from_provider(
            "openai",
            {"api_key": "sk-openai-123"}
        )

        anthropic_strategy = AuthStrategyFactory.create_from_provider(
            "anthropic",
            {"api_key": "sk-ant-456"}
        )

        # Prepare headers for each
        base_headers = {"Content-Type": "application/json"}

        openai_headers = openai_strategy.prepare_headers(base_headers)
        anthropic_headers = anthropic_strategy.prepare_headers(base_headers)

        # Verify they're different and don't interfere
        assert "Authorization" in openai_headers
        assert "x-api-key" in anthropic_headers

        assert "x-api-key" not in openai_headers
        assert "Authorization" not in anthropic_headers

        # Verify original base_headers wasn't modified
        assert base_headers == {"Content-Type": "application/json"}
