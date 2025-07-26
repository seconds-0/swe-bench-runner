"""Base classes for model providers."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Standard response from any model provider."""

    content: str
    model: str
    usage: dict[str, int] | None = None
    cost: float | None = None
    latency_ms: int | None = None
    provider: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    finish_reason: str | None = None
    raw_response: dict[str, Any] | None = None


@dataclass
class ProviderCapabilities:
    """Declare provider capabilities for smart routing."""

    max_context_length: int = 4096
    supports_streaming: bool = False
    supports_json_mode: bool = False
    supports_function_calling: bool = False
    rate_limits: dict[str, int] | None = None
    supported_models: list[str] = field(default_factory=list)
    cost_per_1k_prompt_tokens: float | None = None
    cost_per_1k_completion_tokens: float | None = None


@dataclass
class ProviderConfig:
    """Configuration for a provider."""

    name: str
    api_key: str | None = None
    endpoint: str | None = None
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4000
    timeout: int = 120
    extra_params: dict[str, Any] | None = None

    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


class ModelProvider(ABC):
    """Base class for all model providers with production features."""

    # Provider metadata
    name: str = ""
    description: str = ""
    api_version: str = "1.0"
    requires_api_key: bool = True
    supported_models: list[str] = []
    supports_streaming: bool = False
    default_model: str | None = None

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.capabilities = self._init_capabilities()
        self._initialized = False
        self._health_status = "unknown"
        self._validate_config()

    def _validate_config(self):
        """Validate provider configuration."""
        if self.requires_api_key and not self.config.api_key:
            from .exceptions import ProviderConfigurationError

            raise ProviderConfigurationError(
                f"{self.name} provider requires an API key. "
                f"Set it via environment variable or configuration."
            )

        if self.config.model and self.supported_models:
            if self.config.model not in self.supported_models:
                from .exceptions import ProviderConfigurationError

                raise ProviderConfigurationError(
                    f"Model '{self.config.model}' is not supported by {self.name}. "
                    f"Supported models: {', '.join(self.supported_models)}"
                )

    @abstractmethod
    def _init_capabilities(self) -> ProviderCapabilities:
        """Initialize provider-specific capabilities.

        Returns:
            ProviderCapabilities instance with provider-specific settings
        """
        pass

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response from the model.

        Args:
            prompt: The input prompt
            **kwargs: Additional provider-specific parameters

        Returns:
            ModelResponse containing the generated text and metadata

        Raises:
            ProviderError: For various provider-specific errors
        """
        pass

    async def validate_connection(self) -> bool:
        """Test if the provider is properly configured and accessible.

        Returns:
            True if connection is valid, False otherwise

        Note:
            Default implementation tries a minimal generation.
            Providers can override for more specific validation.
        """
        try:
            # Default implementation - try a minimal generation
            test_response = await self.generate(
                "Respond with 'OK' if you receive this.",
                max_tokens=10
            )
            self._health_status = "healthy"
            return bool(test_response.content)
        except Exception as e:
            logger.warning(f"Provider {self.name} validation failed: {e}")
            self._health_status = "unhealthy"
            return False


    def get_token_limit(self) -> int:
        """Get the maximum token limit for this provider/model.

        Returns:
            Maximum number of tokens supported
        """
        # Default conservative limit
        return 4096

    @classmethod
    def get_required_env_vars(cls) -> list[str]:
        """Get list of required environment variables.

        Returns:
            List of environment variable names this provider needs
        """
        return []

    @classmethod
    def from_env(cls, model: str | None = None) -> "ModelProvider":
        """Create provider instance from environment variables.

        Args:
            model: Optional model override

        Returns:
            Configured provider instance

        Raises:
            ProviderConfigurationError: If required env vars are missing
        """
        import os

        from .exceptions import ProviderConfigurationError

        env_vars = {}
        for var in cls.get_required_env_vars():
            value = os.environ.get(var)
            if not value:
                raise ProviderConfigurationError(
                    f"Missing required environment variable: {var}"
                )
            env_vars[var] = value

        # Create config from environment
        config = cls._config_from_env(env_vars, model)
        return cls(config)

    @classmethod
    @abstractmethod
    def _config_from_env(
        cls, env_vars: dict[str, str], model: str | None = None
    ) -> ProviderConfig:
        """Create provider config from environment variables.

        Args:
            env_vars: Dictionary of environment variables
            model: Optional model override

        Returns:
            ProviderConfig instance
        """
        pass

    @abstractmethod
    def estimate_cost(self, prompt_tokens: int, max_tokens: int) -> float:
        """Estimate the cost of a generation.

        Args:
            prompt_tokens: Number of tokens in the prompt
            max_tokens: Maximum number of completion tokens

        Returns:
            Estimated cost in USD
        """
        pass

    def get_generation_params(self) -> dict[str, Any]:
        """Get provider-specific generation parameters.

        Returns:
            Dictionary of default generation parameters
        """
        return {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

    async def health_check(self) -> dict[str, Any]:
        """Get current health status.

        Returns:
            Dictionary with health status information
        """
        return {
            "status": self._health_status,
            "provider": self.name,
            "model": self.config.model,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def generate_patch(self, instance: dict[str, Any]) -> str:
        """Generate a patch for a SWE-bench instance.

        This is a convenience method that formats the prompt appropriately
        for patch generation and extracts the patch from the response.

        Args:
            instance: SWE-bench instance data

        Returns:
            The generated patch as a string

        Note:
            Providers can override this for custom patch generation logic.
            Default implementation delegates to generate() with appropriate prompting.
        """
        # TODO: Implement in Phase 2 with proper prompt engineering
        # For now, add a comment that this will be implemented when
        # PatchGenerator is created
        raise NotImplementedError(
            "generate_patch will be implemented with PatchGenerator in Phase 2. "
            "For now, use generate() directly with appropriate prompts."
        )
