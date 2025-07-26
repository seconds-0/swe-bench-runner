"""Utility functions for provider operations in CLI."""

from __future__ import annotations

import os
import sys

import click

from . import exit_codes
from .providers import (
    ProviderConfig,
    ProviderConfigManager,
    ProviderConfigurationError,
    ProviderNotFoundError,
    SyncProviderWrapper,
    get_registry,
)


def get_provider_for_cli(
    provider_name: str | None = None,
    model: str | None = None
) -> SyncProviderWrapper:
    """Get a sync-wrapped provider for CLI usage.

    Priority order:
    1. Provided provider_name parameter
    2. SWEBENCH_PROVIDER environment variable
    3. Default to "openai"

    Args:
        provider_name: Optional provider name to use
        model: Optional model override

    Returns:
        SyncProviderWrapper: Ready-to-use sync provider

    Raises:
        SystemExit: If provider is not found or not configured
    """
    registry = get_registry()
    config_manager = ProviderConfigManager()

    # Determine provider name
    if not provider_name:
        provider_name = os.environ.get('SWEBENCH_PROVIDER', 'openai')

    # Check if provider exists
    if provider_name not in registry.list_provider_names():
        click.echo(f"❌ Provider '{provider_name}' not found", err=True)
        click.echo(f"Available providers: {', '.join(registry.list_provider_names())}")
        click.echo("\n💡 Set a default provider with: export SWEBENCH_PROVIDER=<name>")
        sys.exit(exit_codes.GENERAL_ERROR)

    # Get provider configuration
    try:
        config = config_manager.load_config(provider_name)
    except ProviderConfigurationError:
        config = None

    if not config:
        click.echo(f"❌ Provider '{provider_name}' is not configured", err=True)
        click.echo(f"\n🔧 Configure it with: swebench provider init {provider_name}")
        sys.exit(exit_codes.GENERAL_ERROR)

    # Override model if specified
    if model:
        # Create a new config with the overridden model
        # It's always a ProviderConfig object from load_config
        config_dict = vars(config).copy()
        config_dict['model'] = model
        config = ProviderConfig(**config_dict)

    try:
        # Create provider instance
        provider_class = registry.get_provider_class(provider_name)
        provider = provider_class(config)

        # Wrap with sync wrapper
        return SyncProviderWrapper(provider)

    except ProviderConfigurationError as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
        click.echo(f"\n🔧 Reconfigure with: swebench provider init {provider_name}")
        sys.exit(exit_codes.GENERAL_ERROR)
    except Exception as e:
        click.echo(f"❌ Failed to initialize provider: {e}", err=True)
        sys.exit(exit_codes.GENERAL_ERROR)


def ensure_provider_configured(provider_name: str) -> None:
    """Ensure a provider is configured, prompting user if not.

    Args:
        provider_name: Name of the provider to check

    Raises:
        SystemExit: If provider is not configured and user chooses not to configure
    """
    config_manager = ProviderConfigManager()
    registry = get_registry()

    # Check if provider exists
    if provider_name not in registry.list_provider_names():
        click.echo(f"❌ Provider '{provider_name}' not found", err=True)
        click.echo(f"Available providers: {', '.join(registry.list_provider_names())}")
        sys.exit(exit_codes.GENERAL_ERROR)

    # Check if configured
    try:
        config = config_manager.load_config(provider_name)
    except ProviderConfigurationError:
        config = None

    if config:
        return  # Already configured

    # Not configured - prompt user
    click.echo(f"⚠️  Provider '{provider_name}' is not configured")

    if click.confirm(f"Would you like to configure {provider_name} now?"):
        # Import here to avoid circular dependency
        from .cli_provider import init

        # Invoke the init command
        ctx = click.get_current_context()
        ctx.invoke(init, provider_name=provider_name)
    else:
        click.echo("\n❌ Provider configuration is required to continue")
        click.echo(f"Run: swebench provider init {provider_name}")
        sys.exit(exit_codes.GENERAL_ERROR)


def validate_provider_setup() -> bool:
    """Check if at least one provider is configured.

    Returns:
        bool: True if at least one provider is configured
    """
    registry = get_registry()
    config_manager = ProviderConfigManager()

    for provider_name in registry.list_provider_names():
        try:
            if config_manager.load_config(provider_name):
                return True
        except ProviderConfigurationError:
            pass

    return False


def get_default_provider_name() -> str | None:
    """Get the default provider name.

    Returns the first configured provider, or the SWEBENCH_PROVIDER
    environment variable if set.

    Returns:
        Optional[str]: Default provider name or None if none configured
    """
    # Check environment variable first
    env_provider = os.environ.get('SWEBENCH_PROVIDER')
    if env_provider:
        registry = get_registry()
        if env_provider in registry.list_provider_names():
            config_manager = ProviderConfigManager()
            if config_manager.get_config(env_provider):
                return env_provider

    # Otherwise, return first configured provider
    registry = get_registry()
    config_manager = ProviderConfigManager()

    for provider_name in sorted(registry.list_provider_names()):
        try:
            if config_manager.load_config(provider_name):
                return provider_name
        except ProviderConfigurationError:
            pass

    return None


def format_provider_error(error: Exception, provider_name: str) -> str:
    """Format a provider error message for display.

    Args:
        error: The exception that occurred
        provider_name: Name of the provider

    Returns:
        str: Formatted error message with helpful suggestions
    """
    if isinstance(error, ProviderNotFoundError):
        return (
            f"❌ Provider '{provider_name}' not found\n"
            f"Available providers: {', '.join(get_registry().list_provider_names())}"
        )
    elif isinstance(error, ProviderConfigurationError):
        return (
            f"❌ Provider configuration error: {error}\n"
            f"🔧 Reconfigure with: swebench provider init {provider_name}"
        )
    elif "api" in str(error).lower() and "key" in str(error).lower():
        return (
            f"❌ API key error: {error}\n"
            f"🔧 Configure API key with: swebench provider init {provider_name}"
        )
    elif "rate limit" in str(error).lower():
        return (
            f"⚠️  Rate limit exceeded for {provider_name}\n"
            f"💡 Try again later or use a different provider"
        )
    elif "connection" in str(error).lower() or "network" in str(error).lower():
        return (
            f"❌ Network error: {error}\n"
            f"💡 Check your internet connection and try again"
        )
    else:
        return f"❌ Provider error: {error}"
