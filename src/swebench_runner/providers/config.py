"""Configuration management for providers."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    keyring = None  # type: ignore[assignment]
    KEYRING_AVAILABLE = False

from .base import ProviderConfig
from .exceptions import ProviderConfigurationError

logger = logging.getLogger(__name__)


class ProviderConfigManager:
    """Manages provider configurations from multiple sources.

    Configuration precedence (highest to lowest):
    1. Environment variables
    2. Keyring (secure storage)
    3. Configuration file
    4. Default values
    """

    # Environment variable mappings for common providers
    ENV_MAPPING = {
        "openai": {
            "api_key": "OPENAI_API_KEY",
            "org_id": "OPENAI_ORG_ID",
            "base_url": "OPENAI_BASE_URL",
            "model": "OPENAI_MODEL",
        },
        "openrouter": {
            "api_key": "OPENROUTER_API_KEY",
            "model": "OPENROUTER_MODEL",
        },
        # Future provider implementations:
        "anthropic": {
            "api_key": "ANTHROPIC_API_KEY",
            "model": "ANTHROPIC_MODEL",
        },
        "groq": {
            "api_key": "GROQ_API_KEY",
            "model": "GROQ_MODEL",
        },
        "vertex": {
            "project": "GCP_PROJECT",
            "location": "GCP_LOCATION",
            "model": "VERTEX_MODEL",
        },
        "huggingface": {
            "api_key": "HF_API_KEY",
            "model": "HF_MODEL",
        },
    }

    # Default models for providers
    DEFAULT_MODELS = {
        "openai": "gpt-4-turbo-preview",
        "openrouter": "anthropic/claude-3-sonnet",
        # Future provider defaults:
        "anthropic": "claude-3-sonnet-20240229",
        "groq": "mixtral-8x7b-32768",
        "vertex": "gemini-pro",
        "huggingface": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    }

    SERVICE_NAME = "swebench-runner"

    def __init__(self, config_dir: Path | None = None):
        """Initialize config manager.

        Args:
            config_dir: Directory for config files (default: ~/.swebench)
        """
        self.config_dir = config_dir or (Path.home() / ".swebench")
        self.config_file = self.config_dir / "providers.json"
        self._cache: dict[str, ProviderConfig] = {}

    def load_config(self, provider_name: str) -> ProviderConfig:
        """Load configuration for a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            ProviderConfig instance

        Raises:
            ProviderConfigurationError: If no valid configuration found
        """
        # Check cache first
        if provider_name in self._cache:
            return self._cache[provider_name]

        # Try loading from various sources
        config = None

        # 1. Check environment variables
        config = self._load_from_env(provider_name)

        # 2. Check keyring if env didn't provide everything
        if not config or not config.api_key:
            keyring_config = self._load_from_keyring(provider_name)
            if keyring_config:
                if config:
                    # Merge with env config
                    if not config.api_key and keyring_config.api_key:
                        config.api_key = keyring_config.api_key
                else:
                    config = keyring_config

        # 3. Check config file
        if not config:
            config = self._load_from_file(provider_name)

        if not config:
            env_var = (
                self.ENV_MAPPING.get(provider_name, {})
                .get("api_key", f"{provider_name.upper()}_API_KEY")
            )
            raise ProviderConfigurationError(
                f"No configuration found for provider '{provider_name}'. "
                f"Set {env_var} environment variable or run "
                f"'swebench init {provider_name}'"
            )

        # Cache the config
        self._cache[provider_name] = config
        return config

    def _load_from_env(self, provider_name: str) -> ProviderConfig | None:
        """Load configuration from environment variables.

        Args:
            provider_name: Provider name

        Returns:
            ProviderConfig if found, None otherwise
        """
        env_mapping = self.ENV_MAPPING.get(provider_name, {})

        # Check for API key (required for most providers)
        api_key_var = env_mapping.get("api_key", f"{provider_name.upper()}_API_KEY")
        api_key = os.getenv(api_key_var)

        if not api_key and provider_name != "vertex":  # Vertex uses gcloud auth
            return None

        # Build config from environment
        config = ProviderConfig(
            name=provider_name,
            api_key=api_key,
        )

        # Check for optional settings
        if "model" in env_mapping:
            config.model = (
                os.getenv(env_mapping["model"]) or
                self.DEFAULT_MODELS.get(provider_name)
            )

        if "base_url" in env_mapping:
            config.endpoint = os.getenv(env_mapping["base_url"])

        if "org_id" in env_mapping and provider_name == "openai":
            org_id = os.getenv(env_mapping["org_id"])
            if org_id:
                config.extra_params = {"organization": org_id}

        # Vertex-specific handling
        if provider_name == "vertex":
            project = os.getenv(env_mapping.get("project", "GCP_PROJECT"))
            location = os.getenv(
                env_mapping.get("location", "GCP_LOCATION"), "us-central1"
            )
            if project:
                config.extra_params = {
                    "project": project,
                    "location": location
                }

        logger.debug(f"Loaded {provider_name} config from environment")
        return config

    def _load_from_keyring(self, provider_name: str) -> ProviderConfig | None:
        """Load configuration from system keyring.

        Args:
            provider_name: Provider name

        Returns:
            ProviderConfig if found, None otherwise
        """
        if not KEYRING_AVAILABLE:
            return None

        try:
            # Try to get API key from keyring
            api_key = keyring.get_password(
                self.SERVICE_NAME, f"{provider_name}_api_key"
            )
            if not api_key:
                return None

            config = ProviderConfig(
                name=provider_name,
                api_key=api_key,
                model=self.DEFAULT_MODELS.get(provider_name)
            )

            # Check for other stored values
            model = keyring.get_password(self.SERVICE_NAME, f"{provider_name}_model")
            if model:
                config.model = model

            endpoint = keyring.get_password(
                self.SERVICE_NAME, f"{provider_name}_endpoint"
            )
            if endpoint:
                config.endpoint = endpoint

            logger.debug(f"Loaded {provider_name} config from keyring")
            return config

        except Exception as e:
            logger.warning(f"Failed to load from keyring: {e}")
            return None

    def _load_from_file(self, provider_name: str) -> ProviderConfig | None:
        """Load configuration from file.

        Args:
            provider_name: Provider name

        Returns:
            ProviderConfig if found, None otherwise
        """
        if not self.config_file.exists():
            return None

        try:
            with open(self.config_file) as f:
                data = json.load(f)

            if provider_name not in data:
                return None

            provider_data = data[provider_name]
            config = ProviderConfig(
                name=provider_name,
                api_key=provider_data.get("api_key"),
                endpoint=provider_data.get("endpoint"),
                model=(
                    provider_data.get("model") or
                    self.DEFAULT_MODELS.get(provider_name)
                ),
                temperature=provider_data.get("temperature", 0.0),
                max_tokens=provider_data.get("max_tokens", 4000),
                timeout=provider_data.get("timeout", 120),
                extra_params=provider_data.get("extra_params", {})
            )

            logger.debug(f"Loaded {provider_name} config from file")
            return config

        except Exception as e:
            logger.warning(f"Failed to load config file: {e}")
            return None

    def save_config(self, config: ProviderConfig, save_to_keyring: bool = True) -> None:
        """Save provider configuration.

        Args:
            config: Provider configuration to save
            save_to_keyring: Whether to save API key to keyring
        """
        provider_name = config.name

        # Save to keyring if available
        if save_to_keyring and KEYRING_AVAILABLE and config.api_key:
            try:
                keyring.set_password(
                    self.SERVICE_NAME,
                    f"{provider_name}_api_key",
                    config.api_key
                )
                if config.model:
                    keyring.set_password(
                        self.SERVICE_NAME,
                        f"{provider_name}_model",
                        config.model
                    )
                logger.info(f"Saved {provider_name} credentials to keyring")
            except Exception as e:
                logger.warning(f"Failed to save to keyring: {e}")

        # Save to file (without API key if using keyring)
        self._save_to_file(
            config,
            include_api_key=not (save_to_keyring and KEYRING_AVAILABLE)
        )

        # Update cache
        self._cache[provider_name] = config

    def _save_to_file(
        self, config: ProviderConfig, include_api_key: bool = False
    ) -> None:
        """Save configuration to file.

        Args:
            config: Configuration to save
            include_api_key: Whether to include API key in file
        """
        # Ensure directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Load existing data
        data = {}
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    data = json.load(f)
            except Exception as e:
                logger.debug(f"Failed to load existing config file: {e}")

        # Update with new config
        provider_data: dict[str, Any] = {
            "model": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "timeout": config.timeout,
        }

        if config.endpoint:
            provider_data["endpoint"] = config.endpoint

        if config.extra_params:
            provider_data["extra_params"] = config.extra_params

        if include_api_key and config.api_key:
            provider_data["api_key"] = config.api_key

        data[config.name] = provider_data

        # Write to file
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {config.name} config to {self.config_file}")

    def list_configured_providers(self) -> list[str]:
        """List all configured providers.

        Returns:
            List of provider names with configurations
        """
        providers = set()

        # Check environment
        for provider in self.ENV_MAPPING:
            if self._load_from_env(provider):
                providers.add(provider)

        # Check keyring
        if KEYRING_AVAILABLE:
            try:
                # This is a bit hacky but keyring doesn't provide a list method
                for provider in self.ENV_MAPPING:
                    if keyring.get_password(self.SERVICE_NAME, f"{provider}_api_key"):
                        providers.add(provider)
            except Exception as e:
                logger.debug(f"Failed to check keyring for provider {provider}: {e}")

        # Check file
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    data = json.load(f)
                providers.update(data.keys())
            except Exception as e:
                logger.debug(f"Failed to load config file for listing providers: {e}")

        return sorted(providers)

    def clear_config(self, provider_name: str) -> None:
        """Clear configuration for a provider.

        Args:
            provider_name: Provider to clear
        """
        # Clear from cache
        self._cache.pop(provider_name, None)

        # Clear from keyring
        if KEYRING_AVAILABLE:
            try:
                keyring.delete_password(self.SERVICE_NAME, f"{provider_name}_api_key")
                keyring.delete_password(self.SERVICE_NAME, f"{provider_name}_model")
            except Exception as e:
                logger.debug(
                    f"Failed to clear keyring credentials for {provider_name}: {e}"
                )

        # Clear from file
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    data = json.load(f)

                if provider_name in data:
                    del data[provider_name]

                    with open(self.config_file, 'w') as f:
                        json.dump(data, f, indent=2)

            except Exception as e:
                logger.warning(f"Failed to clear config from file: {e}")

        logger.info(f"Cleared configuration for {provider_name}")
