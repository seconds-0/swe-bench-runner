"""Authentication strategies for model providers.

This module provides a flexible authentication strategy pattern that supports
different authentication methods used by various LLM providers.

Supported authentication types:
- Bearer Token: Used by OpenAI-style APIs
- API Key: Used by Anthropic-style APIs
- None: For local APIs like Ollama

Example usage:
    # Create auth strategy for OpenAI
    config = AuthConfig(
        auth_type=AuthType.BEARER_TOKEN,
        credentials={"api_key": "sk-..."}
    )
    auth = AuthStrategyFactory.create(config)
    headers = auth.prepare_headers({"Content-Type": "application/json"})

    # Or use factory method for known providers
    auth = AuthStrategyFactory.create_from_provider(
        "openai",
        {"api_key": "sk-..."}
    )
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AuthType(Enum):
    """Supported authentication types for model providers."""

    BEARER_TOKEN = "bearer_token"  # noqa: S105
    API_KEY = "api_key"  # noqa: S105
    NONE = "none"


@dataclass
class AuthConfig:
    """Authentication configuration for providers.

    Args:
        auth_type: The type of authentication to use
        credentials: Dictionary containing authentication credentials
        additional_headers: Optional additional headers to include

    Raises:
        ValueError: If credentials are required but not provided
    """

    auth_type: AuthType
    credentials: dict[str, str]
    additional_headers: dict[str, str] | None = None

    def __post_init__(self) -> None:
        """Validate auth configuration."""
        if self.auth_type != AuthType.NONE and not self.credentials:
            raise ValueError(f"Credentials required for {self.auth_type.value} auth")


class AuthStrategy(ABC):
    """Abstract base class for authentication strategies.

    This class defines the interface that all authentication strategies
    must implement. Each strategy handles the specifics of how to
    authenticate with different provider APIs.
    """

    def __init__(self, config: AuthConfig) -> None:
        """Initialize the authentication strategy.

        Args:
            config: Authentication configuration
        """
        self.config = config

    @abstractmethod
    def prepare_headers(self, base_headers: dict[str, str]) -> dict[str, str]:
        """Prepare headers with authentication information.

        Args:
            base_headers: Base headers to augment with auth info

        Returns:
            Headers dictionary with authentication information added

        Raises:
            ValueError: If required credentials are missing
        """
        pass

    @abstractmethod
    def validate_credentials(self) -> bool:
        """Validate that credentials are properly configured.

        Returns:
            True if credentials are valid, False otherwise
        """
        pass


class BearerTokenAuth(AuthStrategy):
    """Bearer token authentication for OpenAI-style APIs.

    This strategy adds an Authorization header with a Bearer token.
    It also supports OpenAI-specific headers like organization and project IDs.
    """

    def prepare_headers(self, base_headers: dict[str, str]) -> dict[str, str]:
        """Prepare headers with Bearer token authentication.

        Args:
            base_headers: Base headers to augment

        Returns:
            Headers with Authorization header and optional OpenAI headers

        Raises:
            ValueError: If API key is missing
        """
        headers = base_headers.copy()

        token = self.config.credentials.get("api_key")
        if not token:
            raise ValueError("API key required for bearer token auth")

        headers["Authorization"] = f"Bearer {token}"

        # Add optional organization/project headers for OpenAI
        if org_id := self.config.credentials.get("organization_id"):
            headers["OpenAI-Organization"] = org_id
        if project_id := self.config.credentials.get("project_id"):
            headers["OpenAI-Project"] = project_id

        # Add any additional headers
        if self.config.additional_headers:
            headers.update(self.config.additional_headers)

        logger.debug("Prepared Bearer token authentication headers")
        return headers

    def validate_credentials(self) -> bool:
        """Validate that the API key is present.

        Returns:
            True if API key is present and non-empty
        """
        api_key = self.config.credentials.get("api_key")
        is_valid = bool(api_key and api_key.strip())

        if not is_valid:
            logger.warning(
                "Bearer token authentication validation failed: missing API key"
            )

        return is_valid


class ApiKeyAuth(AuthStrategy):
    """API key authentication for Anthropic-style APIs.

    This strategy adds API key authentication via custom headers.
    It includes Anthropic-specific headers like version and beta features.
    """

    def prepare_headers(self, base_headers: dict[str, str]) -> dict[str, str]:
        """Prepare headers with API key authentication.

        Args:
            base_headers: Base headers to augment

        Returns:
            Headers with x-api-key header and Anthropic-specific headers

        Raises:
            ValueError: If API key is missing
        """
        headers = base_headers.copy()

        api_key = self.config.credentials.get("api_key")
        if not api_key:
            raise ValueError("API key required for API key auth")

        headers["x-api-key"] = api_key

        # Anthropic requires version header
        headers["anthropic-version"] = "2023-06-01"

        # Add beta features if specified
        if beta_features := self.config.credentials.get("beta_features"):
            headers["anthropic-beta"] = beta_features

        # Add any additional headers
        if self.config.additional_headers:
            headers.update(self.config.additional_headers)

        logger.debug("Prepared API key authentication headers")
        return headers

    def validate_credentials(self) -> bool:
        """Validate that the API key is present.

        Returns:
            True if API key is present and non-empty
        """
        api_key = self.config.credentials.get("api_key")
        is_valid = bool(api_key and api_key.strip())

        if not is_valid:
            logger.warning(
                "API key authentication validation failed: missing API key"
            )

        return is_valid


class NoAuth(AuthStrategy):
    """No authentication for local APIs like Ollama.

    This strategy doesn't add any authentication headers but can
    still include additional custom headers for local API configuration.
    """

    def prepare_headers(self, base_headers: dict[str, str]) -> dict[str, str]:
        """Prepare headers without authentication.

        Args:
            base_headers: Base headers to augment

        Returns:
            Headers with optional additional headers but no authentication
        """
        headers = base_headers.copy()

        # Add any additional headers (like custom headers for local APIs)
        if self.config.additional_headers:
            headers.update(self.config.additional_headers)

        logger.debug("Prepared headers without authentication")
        return headers

    def validate_credentials(self) -> bool:
        """Validate credentials (always returns True for no auth).

        Returns:
            Always True since no credentials are required
        """
        return True


class AuthStrategyFactory:
    """Factory for creating authentication strategies.

    This factory provides methods to create appropriate authentication
    strategies based on configuration or provider name.
    """

    _strategies: dict[AuthType, type[AuthStrategy]] = {
        AuthType.BEARER_TOKEN: BearerTokenAuth,
        AuthType.API_KEY: ApiKeyAuth,
        AuthType.NONE: NoAuth,
    }

    @classmethod
    def create(cls, config: AuthConfig) -> AuthStrategy:
        """Create appropriate auth strategy based on config.

        Args:
            config: Authentication configuration

        Returns:
            Configured authentication strategy

        Raises:
            ValueError: If auth type is not supported
        """
        strategy_class = cls._strategies.get(config.auth_type)
        if not strategy_class:
            supported_types = list(cls._strategies.keys())
            raise ValueError(
                f"Unsupported auth type: {config.auth_type}. "
                f"Supported types: {supported_types}"
            )

        logger.debug(f"Creating {strategy_class.__name__} for {config.auth_type}")
        return strategy_class(config)

    @classmethod
    def create_from_provider(
        cls,
        provider_name: str,
        credentials: dict[str, str],
        additional_headers: dict[str, str] | None = None
    ) -> AuthStrategy:
        """Create auth strategy based on provider name.

        Args:
            provider_name: Name of the provider (case-insensitive)
            credentials: Authentication credentials
            additional_headers: Optional additional headers

        Returns:
            Configured authentication strategy for the provider

        Raises:
            ValueError: If provider is not recognized
        """
        provider_lower = provider_name.lower()

        if provider_lower == "openai":
            config = AuthConfig(
                auth_type=AuthType.BEARER_TOKEN,
                credentials=credentials,
                additional_headers=additional_headers
            )
        elif provider_lower == "anthropic":
            config = AuthConfig(
                auth_type=AuthType.API_KEY,
                credentials=credentials,
                additional_headers=additional_headers
            )
        elif provider_lower == "ollama":
            config = AuthConfig(
                auth_type=AuthType.NONE,
                credentials={},  # Ollama doesn't need credentials
                additional_headers=additional_headers
            )
        else:
            supported_providers = ["openai", "anthropic", "ollama"]
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Supported providers: {supported_providers}"
            )

        logger.debug(f"Creating auth strategy for provider: {provider_name}")
        return cls.create(config)

    @classmethod
    def get_supported_auth_types(cls) -> list[AuthType]:
        """Get list of supported authentication types.

        Returns:
            List of supported AuthType values
        """
        return list(cls._strategies.keys())

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Get list of providers supported by create_from_provider.

        Returns:
            List of supported provider names
        """
        return ["openai", "anthropic", "ollama"]

