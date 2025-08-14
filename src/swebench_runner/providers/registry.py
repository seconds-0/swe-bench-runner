"""Provider registration and discovery system."""

from __future__ import annotations

import ast
import importlib
import logging
from pathlib import Path
from threading import Lock
from typing import Any

from .base import ModelProvider, ProviderConfig
from .exceptions import (
    ProviderAlreadyRegisteredError,
    ProviderConfigurationError,
    ProviderNotFoundError,
)

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Thread-safe singleton registry for model providers with validation."""

    _instance: ProviderRegistry | None = None
    _lock = Lock()  # Class-level lock for singleton creation

    def __new__(cls) -> ProviderRegistry:
        """Implement singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    # Initialize instance attributes only once
                    cls._instance._providers: dict[str, type[ModelProvider]] = {}
                    cls._instance._instances: dict[str, ModelProvider] = {}  # Cached instances
                    cls._instance._configs: dict[str, ProviderConfig] = {}
                    cls._instance._initialized = False
                    cls._instance._instance_lock = Lock()  # Instance-level lock for operations
        return cls._instance

    def __init__(self) -> None:
        """Initialize the registry (called every time, but singleton ensures single instance)."""
        # Initialization is handled in __new__ to ensure it happens only once
        pass

    def register(
        self, name: str, provider_class: type[ModelProvider], force: bool = False
    ) -> None:
        """Register a provider class.

        Args:
            name: The name to register the provider under
            provider_class: The provider class to register
            force: If True, overwrite existing provider

        Raises:
            ProviderAlreadyRegisteredError: If provider already registered and force=False
        """
        with self._instance_lock:
            if name in self._providers and not force:
                raise ProviderAlreadyRegisteredError(
                    f"Provider '{name}' is already registered. Use force=True to overwrite.",
                    provider=name
                )
            self._providers[name] = provider_class
            logger.info(f"Registered provider: {name}")

    def _validate_provider_class(self, provider_class: type[ModelProvider]) -> None:
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
                raise ValueError(
                    f"Provider {provider_class.__name__} must have {attr} attribute"
                )

        # Check required methods are implemented
        required_methods = ["generate", "_init_capabilities", "_config_from_env"]
        for method in required_methods:
            if not hasattr(provider_class, method):
                raise ValueError(
                    f"Provider {provider_class.__name__} must implement {method} method"
                )

    def get(self, name: str) -> type[ModelProvider]:
        """Get a provider class by name.

        Args:
            name: Provider name

        Returns:
            Provider class

        Raises:
            ProviderNotFoundError: If provider not found
        """
        return self.get_provider_class(name)

    def get_provider_class(self, name: str) -> type[ModelProvider]:
        """Get a provider class by name.

        Args:
            name: Provider name

        Returns:
            Provider class

        Raises:
            ProviderNotFoundError: If provider not found
        """
        with self._instance_lock:
            if not self._initialized:
                self._auto_discover()

            # Check if already loaded
            if name in self._providers:
                return self._providers[name]

            # Try lazy loading
            if hasattr(self, '_lazy_providers') and name in self._lazy_providers:
                module_path, class_name = self._lazy_providers[name]
                try:
                    module = importlib.import_module(module_path)
                    provider_class = getattr(module, class_name)
                    # Cache for future use
                    self._providers[name] = provider_class
                    return provider_class
                except (ImportError, AttributeError) as e:
                    logger.error(f"Failed to load provider {name}: {e}")
                    raise ProviderNotFoundError(
                        f"Failed to load provider '{name}': {e}"
                    ) from e

            # Provider not found
            available_lazy = (
                list(self._lazy_providers.keys())
                if hasattr(self, '_lazy_providers')
                else []
            )
            available_loaded = list(self._providers.keys())
            all_available = sorted(set(available_lazy + available_loaded))
            available_str = ', '.join(all_available)
            raise ProviderNotFoundError(
                f"Provider '{name}' not found. Available providers: {available_str}"
            )

    def get_provider(
        self, name: str, config: ProviderConfig | None = None, cache: bool = True
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
                with self._instance_lock:
                    if name in self._configs:
                        config = self._configs[name]
                    else:
                        # Try to load from config manager
                        from .config import ProviderConfigManager
                        config_manager = ProviderConfigManager()
                        try:
                            config = config_manager.load_config(name)
                        except ProviderConfigurationError:
                            raise e from None

        # Ensure we have a valid config
        final_config = config or self._configs.get(name)
        if final_config is None:
            raise ProviderConfigurationError(
                f"No configuration available for provider '{name}'. "
                "Set environment variables or use config file."
            )

        provider = provider_class(final_config)

        # Cache the instance
        if cache:
            with self._instance_lock:
                self._instances[name] = provider

        return provider

    def list_providers(self) -> list[str]:
        """List all available provider names.

        Returns:
            List of provider names
        """
        return self.list_provider_names()

    def list_provider_details(self) -> list[dict[str, Any]]:
        """List detailed information about all available providers.

        Returns:
            List of provider information dictionaries
        """
        with self._instance_lock:
            if not self._initialized:
                self._auto_discover()

            results = []

            # First add already loaded providers
            for name, cls in self._providers.items():
                # Handle both provider classes and mock objects (for tests)
                if hasattr(cls, 'name'):
                    results.append({
                        "name": cls.name,
                        "description": getattr(cls, 'description', ''),
                        "models": getattr(cls, 'supported_models', []),
                        "requires_api_key": getattr(cls, 'requires_api_key', True),
                        "supports_streaming": getattr(cls, 'supports_streaming', False),
                        "api_version": getattr(cls, "api_version", "1.0"),
                        "configured": cls.name in self._configs,
                        "cached": cls.name in self._instances,
                    })
                else:
                    # Mock or simplified provider
                    results.append({
                        "name": name,
                        "description": f"{name} provider",
                        "models": [],
                        "requires_api_key": True,
                        "supports_streaming": False,
                        "api_version": "1.0",
                        "configured": name in self._configs,
                        "cached": name in self._instances,
                    })

            # Then add lazy providers (but don't load them yet)
            if hasattr(self, '_lazy_providers'):
                for name in self._lazy_providers:
                    if name not in self._providers:  # Skip if already loaded
                        # Return minimal info without loading
                        results.append({
                            "name": name,
                            "description": f"{name.title()} provider (not loaded)",
                            "models": [],
                            "requires_api_key": True,  # Safe default
                            "supports_streaming": False,  # Safe default
                            "api_version": "1.0",
                            "configured": name in self._configs,
                            "cached": name in self._instances,
                        })

            return results

    def list_provider_names(self) -> list[str]:
        """List all available provider names.

        Returns:
            List of provider names
        """
        with self._instance_lock:
            if not self._initialized:
                self._auto_discover()

            # Combine loaded and lazy provider names
            loaded_names = list(self._providers.keys())
            lazy_names = (
                list(self._lazy_providers.keys())
                if hasattr(self, '_lazy_providers')
                else []
            )
            return sorted(set(loaded_names + lazy_names))

    def _auto_discover(self) -> None:
        """Auto-discover providers in the providers directory WITHOUT importing them.

        Uses AST parsing to find provider classes without executing module code,
        preventing side effects like async initialization that cause hanging.
        """
        self._initialized = True

        # Store provider info for lazy loading
        if not hasattr(self, '_lazy_providers'):
            self._lazy_providers = {}

        # Get the providers directory
        providers_dir = Path(__file__).parent

        # Scan for Python files (excluding __pycache__, tests, and special files)
        for py_file in providers_dir.glob("*.py"):
            if py_file.stem.startswith("_") or py_file.stem in (
                "base", "registry", "exceptions", "config",
                "async_bridge", "auth_strategies", "circuit_breaker",
                "rate_limiters", "streaming_adapters", "token_counters",
                "transform_pipeline", "unified_models", "wrappers",
                "openai_errors"  # Helper modules, not providers
            ):
                continue

            # Use AST to find provider classes without importing
            try:
                with open(py_file, encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=str(py_file))

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Look for classes that inherit from ModelProvider
                        # or have Provider in the name
                        if "Provider" in node.name and not node.name.startswith("_"):
                            # Check if it has a 'name' attribute
                            has_name = False
                            provider_name = None

                            for item in node.body:
                                if isinstance(item, ast.Assign):
                                    for target in item.targets:
                                        if isinstance(target, ast.Name) and target.id == "name":
                                            # Try to get the literal value
                                            if isinstance(item.value, ast.Constant):
                                                provider_name = item.value.value
                                                has_name = True
                                                break

                            if has_name and provider_name:
                                # Store for lazy loading
                                module_path = f"swebench_runner.providers.{py_file.stem}"
                                self._lazy_providers[provider_name] = (
                                    module_path, node.name
                                )
                                logger.debug(
                                    f"Discovered provider: {provider_name} "
                                    f"({module_path}.{node.name})"
                                )
            except Exception as e:
                logger.warning(f"Failed to parse {py_file}: {e}")
                continue

        # If no providers found via AST, fall back to known providers
        # This ensures the system works even if AST parsing fails
        if not self._lazy_providers:
            logger.warning("No providers discovered via AST, using fallback list")
            fallback_providers = [
                ("mock", "swebench_runner.providers.mock", "MockProvider"),
                ("openai", "swebench_runner.providers.openai", "OpenAIProvider"),
                ("anthropic", "swebench_runner.providers.anthropic", "AnthropicProvider"),
                ("ollama", "swebench_runner.providers.ollama", "OllamaProvider"),
                ("openrouter", "swebench_runner.providers.openrouter", "OpenRouterProvider"),
            ]
            for name, module_path, class_name in fallback_providers:
                self._lazy_providers[name] = (module_path, class_name)

    def save_config(self, name: str, config: ProviderConfig) -> None:
        """Save a provider configuration.

        Args:
            name: Provider name
            config: Provider configuration
        """
        with self._instance_lock:
            self._configs[name] = config
            # Clear cached instance to force reload with new config
            self._instances.pop(name, None)

    def clear(self) -> None:
        """Clear all registered providers and cached instances."""
        with self._instance_lock:
            self._providers.clear()
            self._configs.clear()
            self._instances.clear()
            self._initialized = False

    def clear_cache(self, name: str | None = None) -> None:
        """Clear cached provider instances.

        Args:
            name: Specific provider to clear, or None to clear all
        """
        with self._instance_lock:
            if name:
                self._instances.pop(name, None)
            else:
                self._instances.clear()

    def clear_instances(self) -> None:
        """Clear all cached provider instances."""
        self.clear_cache()

    def unregister(self, name: str) -> None:
        """Unregister a provider.

        Args:
            name: Provider name to unregister
        """
        with self._instance_lock:
            self._providers.pop(name, None)
            self._instances.pop(name, None)
            self._configs.pop(name, None)

    def is_registered(self, name: str) -> bool:
        """Check if a provider is registered.

        Args:
            name: Provider name to check

        Returns:
            True if registered, False otherwise
        """
        with self._instance_lock:
            if not self._initialized:
                self._auto_discover()
            return name in self._providers or (
                hasattr(self, '_lazy_providers') and name in self._lazy_providers
            )

    def get_provider_info(self, name: str) -> dict[str, Any] | None:
        """Get information about a provider.

        Args:
            name: Provider name

        Returns:
            Provider information dictionary or None if not found
        """
        with self._instance_lock:
            if name not in self._providers:
                # Try to load it if it's a lazy provider
                if hasattr(self, '_lazy_providers') and name in self._lazy_providers:
                    try:
                        # Temporarily release lock to avoid deadlock
                        self._instance_lock.release()
                        try:
                            self.get_provider_class(name)
                        finally:
                            self._instance_lock.acquire()
                    except ProviderNotFoundError:
                        return None

            if name in self._providers:
                provider_class = self._providers[name]
                return {
                    "name": name,
                    "class": provider_class,
                    "description": getattr(provider_class, 'description', ''),
                    "models": getattr(provider_class, 'supported_models', []),
                    "requires_api_key": getattr(provider_class, 'requires_api_key', True),
                    "supports_streaming": getattr(provider_class, 'supports_streaming', False),
                }
            return None


# Global registry instance
_registry = ProviderRegistry()


def register_provider(provider_class: type[ModelProvider]) -> type[ModelProvider]:
    """Decorator to register a provider class.

    Usage:
        @register_provider
        class MyProvider(ModelProvider):
            name = "my_provider"
            ...
    """
    # Use the provider's name attribute for registration
    if hasattr(provider_class, 'name') and provider_class.name:
        _registry.register(provider_class.name, provider_class)
    else:
        # Fallback to class name
        _registry.register(provider_class.__name__.lower(), provider_class)
    return provider_class


def get_registry() -> ProviderRegistry:
    """Get the global provider registry."""
    return _registry
