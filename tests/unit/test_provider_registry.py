"""Test provider registry functionality."""
import threading
from unittest.mock import Mock

import pytest

from swebench_runner.providers.registry import ProviderRegistry


class TestProviderRegistry:
    """Test ProviderRegistry functionality."""

    def test_singleton_pattern(self):
        """Registry should be a singleton."""
        registry1 = ProviderRegistry()
        registry2 = ProviderRegistry()
        assert registry1 is registry2

    def test_register_provider_class(self):
        """Should register provider classes."""
        registry = ProviderRegistry()
        registry._providers.clear()  # Clear any existing providers

        mock_provider_class = Mock()
        registry.register("test_provider", mock_provider_class)

        assert "test_provider" in registry._providers
        assert registry._providers["test_provider"] == mock_provider_class

    def test_register_duplicate_provider_raises_error(self):
        """Should raise error when registering duplicate provider."""
        registry = ProviderRegistry()
        registry._providers.clear()

        mock_provider_class = Mock()
        registry.register("test_provider", mock_provider_class)

        from swebench_runner.providers.exceptions import ProviderAlreadyRegisteredError
        with pytest.raises(ProviderAlreadyRegisteredError):
            registry.register("test_provider", mock_provider_class)

    def test_register_with_force_overwrites(self):
        """Should overwrite provider when force=True."""
        registry = ProviderRegistry()
        registry._providers.clear()

        mock_provider1 = Mock()
        mock_provider2 = Mock()

        registry.register("test_provider", mock_provider1)
        registry.register("test_provider", mock_provider2, force=True)

        assert registry._providers["test_provider"] == mock_provider2

    def test_get_nonexistent_provider_raises_error(self):
        """Should raise error for nonexistent provider."""
        registry = ProviderRegistry()
        registry._providers.clear()

        from swebench_runner.providers.exceptions import ProviderNotFoundError
        with pytest.raises(ProviderNotFoundError):
            registry.get("nonexistent")

    def test_list_providers(self):
        """Should list all registered providers."""
        registry = ProviderRegistry()
        registry._providers.clear()

        mock_provider1 = Mock()
        mock_provider2 = Mock()

        registry.register("provider1", mock_provider1)
        registry.register("provider2", mock_provider2)

        providers = registry.list_providers()
        assert "provider1" in providers
        assert "provider2" in providers

    def test_clear_instances(self):
        """Should clear cached instances."""
        registry = ProviderRegistry()
        registry._instances.clear()

        # Add a mock instance
        registry._instances["test"] = Mock()
        assert "test" in registry._instances

        registry.clear_instances()
        assert len(registry._instances) == 0

    def test_unregister_provider(self):
        """Should unregister provider."""
        registry = ProviderRegistry()
        registry._providers.clear()

        mock_provider = Mock()
        registry.register("test_provider", mock_provider)
        assert "test_provider" in registry._providers

        registry.unregister("test_provider")
        assert "test_provider" not in registry._providers

    def test_thread_safety_for_registration(self):
        """Registry registration should be thread-safe."""
        registry = ProviderRegistry()
        registry._providers.clear()

        def register_provider(name):
            mock_provider = Mock()
            try:
                registry.register(name, mock_provider)
            except Exception:
                pass  # Ignore duplicate registration errors

        threads = []
        for i in range(10):
            t = threading.Thread(target=register_provider, args=(f"provider_{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All unique providers should be registered
        assert len(registry._providers) == 10

    def test_is_registered(self):
        """Should check if provider is registered."""
        registry = ProviderRegistry()
        registry._providers.clear()

        mock_provider = Mock()
        registry.register("test_provider", mock_provider)

        assert registry.is_registered("test_provider")
        assert not registry.is_registered("nonexistent")

    def test_get_provider_info(self):
        """Should get provider information."""
        registry = ProviderRegistry()
        registry._providers.clear()

        mock_provider = Mock()
        mock_provider.__name__ = "MockProvider"
        mock_provider.__doc__ = "Test provider"

        registry.register("test_provider", mock_provider)

        info = registry.get_provider_info("test_provider")
        assert info is not None
        assert info["class"] == mock_provider
        assert info["name"] == "test_provider"

    def test_registry_with_real_providers(self):
        """Test registry with actual provider classes."""
        registry = ProviderRegistry()

        # Import real providers
        from swebench_runner.providers.anthropic import AnthropicProvider
        from swebench_runner.providers.openai import OpenAIProvider

        # Clear and register
        registry._providers.clear()
        registry.register("openai", OpenAIProvider)
        registry.register("anthropic", AnthropicProvider)

        # Check they're registered
        assert "openai" in registry.list_providers()
        assert "anthropic" in registry.list_providers()

        # Check info
        openai_info = registry.get_provider_info("openai")
        assert openai_info["class"] == OpenAIProvider

    def test_default_initialization(self):
        """Test that registry initializes with default providers."""
        # Get a fresh registry instance
        registry = ProviderRegistry()

        # Should have some default providers registered
        providers = registry.list_providers()

        # Common providers that should be registered by default
        expected_providers = ["openai", "anthropic", "ollama", "openrouter"]

        for provider in expected_providers:
            assert provider in providers, f"Expected {provider} to be registered by default"
