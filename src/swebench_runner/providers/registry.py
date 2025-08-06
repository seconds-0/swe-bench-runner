"""Provider registration and discovery system."""

import importlib
import logging
from pathlib import Path
from threading import Lock
from typing import Optional

from .base import ModelProvider, ProviderConfig
from .exceptions import ProviderConfigurationError, ProviderNotFoundError

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Thread-safe registry for model providers with validation."""

    def __init__(self):
        self._providers: dict[str, type[ModelProvider]] = {}
        self._instances: dict = {}  # Cached instances
        self._configs: dict = {}
        self._initialized = False
        self._lock = Lock()  # Thread safety

    def register(self, provider_class: type[ModelProvider], validate: bool = True):
        """Register a provider class with validation.

        Args:
            provider_class: The provider class to register
            validate: Whether to validate the provider class

        Raises:
            ValueError: If provider is invalid
        """
        if not provider_class.name:
            raise ValueError(f"Provider {provider_class.__name__} must have a name")

        if validate:
            self._validate_provider_class(provider_class)

        with self._lock:
            self._providers[provider_class.name] = provider_class
            logger.info(f"Registered provider: {provider_class.name}")

    def _validate_provider_class(self, provider_class: type[ModelProvider]):
        """Validate a provider class has required attributes and methods.

        Args:
            provider_class: Provider class to validate

        Raises:
            ValueError: If validation fails
        """
        # Check required class attributes
        required_attrs = ["name", "description"]
        for attr in required_attrs:
            if not hasattr(provider_class, attr) or not getattr(provider_class, attr):
                raise ValueError(f"Provider {provider_class.__name__} must have {attr} attribute")

        # Check required methods are implemented
        required_methods = ["generate", "_init_capabilities", "_config_from_env"]
        for method in required_methods:
            if not hasattr(provider_class, method):
                raise ValueError(f"Provider {provider_class.__name__} must implement {method} method")

    def get_provider_class(self, name: str) -> type[ModelProvider]:
        """Get a provider class by name.

        Args:
            name: Provider name

        Returns:
            Provider class

        Raises:
            ProviderNotFoundError: If provider not found
        """
        with self._lock:
            if not self._initialized:
                self._auto_discover()

            if name not in self._providers:
                available = ", ".join(sorted(self._providers.keys()))
                raise ProviderNotFoundError(
                    f"Provider '{name}' not found. Available providers: {available}"
                )

            return self._providers[name]

    def get_provider(
        self, name: str, config: Optional[ProviderConfig] = None, cache: bool = True
    ) -> ModelProvider:
        """Get an initialized provider instance with caching.

        Args:
            name: Provider name
            config: Optional provider configuration
            cache: Whether to cache the instance

        Returns:
            Initialized provider instance
        """
        # Check cache first
        if cache and name in self._instances:
            return self._instances[name]

        provider_class = self.get_provider_class(name)

        if config is None:
            # Try to create from environment
            try:
                provider = provider_class.from_env()
            except Exception as e:
                # Fall back to stored config if available
                with self._lock:
                    if name in self._configs:
                        config = self._configs[name]
                    else:
                        # Try to load from config manager
                        from .config import ProviderConfigManager
                        config_manager = ProviderConfigManager()
                        try:
                            config = config_manager.load_config(name)
                        except ProviderConfigurationError:
                            raise e

        provider = provider_class(config or self._configs.get(name))

        # Cache the instance
        if cache:
            with self._lock:
                self._instances[name] = provider

        return provider

    def list_providers(self) -> list:
        """List all available providers.

        Returns:
            List of provider information dictionaries
        """
        with self._lock:
            if not self._initialized:
                self._auto_discover()

            return [
                {
                    "name": cls.name,
                    "description": cls.description,
                    "models": cls.supported_models,
                    "requires_api_key": cls.requires_api_key,
                    "supports_streaming": cls.supports_streaming,
                    "api_version": getattr(cls, "api_version", "1.0"),
                    "configured": cls.name in self._configs,
                    "cached": cls.name in self._instances,
                }
                for cls in self._providers.values()
            ]

    def list_provider_names(self) -> list:
        """List all available provider names.

        Returns:
            List of provider names
        """
        with self._lock:
            if not self._initialized:
                self._auto_discover()
            return list(self._providers.keys())

    def _auto_discover(self):
        """Auto-discover providers in the providers directory."""
        self._initialized = True

        # Get the providers directory
        providers_dir = Path(__file__).parent

        # Look for Python files
        for file_path in providers_dir.glob("*.py"):
            if file_path.name.startswith("_") or file_path.name in [
                "base.py",
                "registry.py",
                "exceptions.py",
            ]:
                continue

            module_name = file_path.stem
            try:
                # Import the module
                module = importlib.import_module(
                    f".{module_name}", package="swebench_runner.providers"
                )

                # Look for ModelProvider subclasses
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, ModelProvider)
                        and attr is not ModelProvider
                        and hasattr(attr, "name")
                        and attr.name
                    ):
                        self.register(attr, validate=False)  # Skip validation during auto-discovery
            except Exception as e:
                # Log error but continue discovery
                logger.warning(f"Failed to load provider from {module_name}: {e}")

    def save_config(self, name: str, config: ProviderConfig):
        """Save a provider configuration.

        Args:
            name: Provider name
            config: Provider configuration
        """
        with self._lock:
            self._configs[name] = config
            # Clear cached instance to force reload with new config
            self._instances.pop(name, None)

    def clear(self):
        """Clear all registered providers and cached instances."""
        with self._lock:
            self._providers.clear()
            self._configs.clear()
            self._instances.clear()
            self._initialized = False

    def clear_cache(self, name: Optional[str] = None):
        """Clear cached provider instances.

        Args:
            name: Specific provider to clear, or None to clear all
        """
        with self._lock:
            if name:
                self._instances.pop(name, None)
            else:
                self._instances.clear()


# Global registry instance
_registry = ProviderRegistry()


def register_provider(provider_class: type[ModelProvider]):
    """Decorator to register a provider class.

    Usage:
        @register_provider
        class MyProvider(ModelProvider):
            name = "my_provider"
            ...
    """
    _registry.register(provider_class)
    return provider_class


def get_registry() -> ProviderRegistry:
    """Get the global provider registry."""
    return _registry
