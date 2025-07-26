"""CLI commands for managing model providers."""

from __future__ import annotations

import sys
import time

import click
from rich.console import Console
from rich.table import Table

from . import exit_codes
from .providers import (
    ProviderConfig,
    ProviderConfigManager,
    ProviderConfigurationError,
    SyncProviderWrapper,
    get_registry,
)

console = Console()


@click.group(name="provider")
def provider_cli() -> None:
    """Manage model providers for patch generation."""
    pass


@provider_cli.command()
def list() -> None:
    """List all available providers and their configuration status."""
    registry = get_registry()
    config_manager = ProviderConfigManager()

    # Create a table
    table = Table(title="Available Model Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Status", style="green")
    table.add_column("API Key", style="yellow")

    providers_info = registry.list_providers()

    # Sort by provider name
    for provider_info in sorted(providers_info, key=lambda x: x['name']):
        provider_name = provider_info['name']
        provider_class = registry.get_provider_class(provider_name)

        # Check if configured
        try:
            config = config_manager.load_config(provider_name)
            is_configured = bool(config)
        except ProviderConfigurationError:
            is_configured = False

        # Check for API key requirement and status
        has_api_key = False
        if is_configured and config:
            has_api_key = bool(getattr(config, 'api_key', None))

        # Get provider description
        description = getattr(provider_class, '__doc__', '').strip().split('\n')[0]
        if not description:
            description = f"{provider_name.title()} model provider"

        # Determine status
        if is_configured:
            status = "✅ Configured"
        else:
            status = "❌ Not configured"

        # API key status
        if provider_name in ['openai', 'openrouter']:  # Providers that need API keys
            api_key_status = "✅ Set" if has_api_key else "❌ Required"
        else:
            api_key_status = "N/A"

        table.add_row(provider_name, description, status, api_key_status)

    console.print(table)
    console.print(
        "\n💡 Use 'swebench provider init <provider>' to configure a provider"
    )


@provider_cli.command()
@click.argument('provider_name')
def init(provider_name: str) -> None:
    """Initialize a provider configuration interactively."""
    registry = get_registry()
    config_manager = ProviderConfigManager()

    # Check if provider exists
    if provider_name not in registry.list_provider_names():
        click.echo(f"❌ Provider '{provider_name}' not found", err=True)
        click.echo(f"Available providers: {', '.join(registry.list_provider_names())}")
        sys.exit(exit_codes.GENERAL_ERROR)

    click.echo(f"🔧 Configuring {provider_name} provider")
    click.echo()

    config = {}

    # Provider-specific configuration
    if provider_name == 'openai':
        # API Key
        try:
            existing_config = config_manager.load_config(provider_name)
            default_api_key = getattr(existing_config, 'api_key', '')
        except ProviderConfigurationError:
            default_api_key = ''

        api_key = click.prompt(
            "OpenAI API Key",
            hide_input=True,
            default=default_api_key,
            show_default=False
        )
        if api_key:
            config['api_key'] = api_key

        # Model selection
        click.echo("\nAvailable models:")
        click.echo("  1. gpt-4-turbo-preview (Latest, best quality)")
        click.echo("  2. gpt-4 (Stable, proven)")
        click.echo("  3. gpt-3.5-turbo (Fast, cost-effective)")

        model_choice = click.prompt(
            "Select model (1-3)",
            type=int,
            default=1
        )

        model_map = {
            1: "gpt-4-turbo-preview",
            2: "gpt-4",
            3: "gpt-3.5-turbo"
        }
        config['model'] = model_map.get(model_choice, "gpt-4-turbo-preview")

        # Temperature
        temperature = click.prompt(
            "Temperature (0.0-1.0)",
            type=float,
            default=0.1
        )
        config['temperature'] = temperature

    elif provider_name == 'openrouter':
        # API Key
        try:
            existing_config = config_manager.load_config(provider_name)
            default_api_key = getattr(existing_config, 'api_key', '')
        except ProviderConfigurationError:
            default_api_key = ''

        api_key = click.prompt(
            "OpenRouter API Key",
            hide_input=True,
            default=default_api_key,
            show_default=False
        )
        if api_key:
            config['api_key'] = api_key

        # Model selection
        click.echo("\nPopular models on OpenRouter:")
        click.echo("  1. openai/gpt-4-turbo-preview ($10.00/1M tokens)")
        click.echo("  2. anthropic/claude-3-opus ($15.00/1M tokens)")
        click.echo("  3. anthropic/claude-3-sonnet ($3.00/1M tokens)")
        click.echo("  4. google/gemini-pro-1.5 ($2.50/1M tokens)")
        click.echo("  5. meta-llama/llama-3-70b-instruct ($0.59/1M tokens)")
        click.echo("  6. Custom (enter model name)")

        model_choice = click.prompt(
            "Select model (1-6)",
            type=int,
            default=1
        )

        if model_choice == 6:
            config['model'] = click.prompt("Enter model name")
        else:
            model_map = {
                1: "openai/gpt-4-turbo-preview",
                2: "anthropic/claude-3-opus",
                3: "anthropic/claude-3-sonnet",
                4: "google/gemini-pro-1.5",
                5: "meta-llama/llama-3-70b-instruct"
            }
            config['model'] = model_map.get(model_choice, "openai/gpt-4-turbo-preview")

        # Temperature
        temperature = click.prompt(
            "Temperature (0.0-1.0)",
            type=float,
            default=0.1
        )
        config['temperature'] = temperature

    elif provider_name == 'mock':
        # Mock provider doesn't need configuration
        click.echo("Mock provider doesn't require configuration")
        config = {'name': 'mock'}  # Minimal config for mock provider

    # Test the configuration
    click.echo("\n🧪 Testing provider connection...")
    try:
        # Create provider instance with test config
        provider_class = registry.get_provider_class(provider_name)

        # Create ProviderConfig instance
        # config is always a dict from the init flow above
        if 'name' not in config:
            config['name'] = provider_name
        provider_config = ProviderConfig(**config)

        provider = provider_class(provider_config)

        # Wrap with sync wrapper for testing
        sync_provider = SyncProviderWrapper(provider)

        # Test with simple prompt
        test_prompt = "Please respond with 'OK' if you receive this message."
        start_time = time.time()

        response = sync_provider.generate_sync(
            prompt=test_prompt,
            max_tokens=50
        )

        elapsed = time.time() - start_time

        if response.content.strip():
            click.echo(f"✅ Connection successful! Response received in {elapsed:.2f}s")
            click.echo(f"   Model: {response.model}")
            if response.cost:
                click.echo(f"   Cost: ${response.cost:.4f}")

            # Save configuration
            config_manager.save_config(provider_config)
            click.echo(f"\n✅ {provider_name} provider configured successfully!")

        else:
            click.echo("❌ Test failed: No response received", err=True)
            sys.exit(exit_codes.GENERAL_ERROR)

    except Exception as e:
        click.echo(f"❌ Test failed: {e}", err=True)
        click.echo("\nPlease check your configuration and try again.")
        sys.exit(exit_codes.GENERAL_ERROR)


@provider_cli.command()
@click.argument('provider_name')
@click.option('--prompt', '-p',
              default='Hello! Please respond with "OK" if you receive this.',
              help='Test prompt')
@click.option('--max-tokens', '-m', default=50, help='Maximum tokens for response')
def test(provider_name: str, prompt: str, max_tokens: int) -> None:
    """Test a provider with a simple prompt."""
    registry = get_registry()
    config_manager = ProviderConfigManager()

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

    if not config:
        click.echo(f"❌ Provider '{provider_name}' is not configured", err=True)
        click.echo(f"Run: swebench provider init {provider_name}")
        sys.exit(exit_codes.GENERAL_ERROR)

    click.echo(f"🧪 Testing {provider_name} provider...")
    click.echo(f"Prompt: {prompt}")
    click.echo()

    try:
        # Create provider instance
        provider_class = registry.get_provider_class(provider_name)
        provider = provider_class(config)

        # Wrap with sync wrapper
        sync_provider = SyncProviderWrapper(provider)

        # Generate response
        start_time = time.time()
        response = sync_provider.generate_sync(
            prompt=prompt,
            max_tokens=max_tokens
        )
        elapsed = time.time() - start_time

        # Display results
        click.echo("📝 Response:")
        click.echo(response.content)
        click.echo()
        click.echo(f"⏱️  Latency: {elapsed:.2f}s")
        click.echo(f"🤖 Model: {response.model}")
        if response.cost:
            click.echo(f"💰 Cost: ${response.cost:.4f}")
        if response.usage:
            click.echo(f"📊 Tokens: {response.usage['total_tokens']} "
                      f"(prompt: {response.usage['prompt_tokens']}, "
                      f"completion: {response.usage['completion_tokens']})")

    except ProviderConfigurationError as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
        click.echo(f"Run: swebench provider init {provider_name}")
        sys.exit(exit_codes.GENERAL_ERROR)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(exit_codes.GENERAL_ERROR)


@provider_cli.command()
@click.argument('provider_name')
def info(provider_name: str) -> None:
    """Show detailed information about a provider."""
    registry = get_registry()
    config_manager = ProviderConfigManager()

    # Check if provider exists
    if provider_name not in registry.list_provider_names():
        click.echo(f"❌ Provider '{provider_name}' not found", err=True)
        click.echo(f"Available providers: {', '.join(registry.list_provider_names())}")
        sys.exit(exit_codes.GENERAL_ERROR)

    provider_class = registry.get_provider_class(provider_name)
    try:
        config = config_manager.load_config(provider_name)
    except ProviderConfigurationError:
        config = None

    click.echo(f"\n📊 Provider: {provider_name}")
    click.echo("=" * 50)

    # Description
    description = getattr(provider_class, '__doc__', '').strip()
    if description:
        click.echo("\n📝 Description:")
        click.echo(f"   {description}")

    # Configuration status
    click.echo("\n⚙️  Configuration:")
    if config:
        click.echo("   Status: ✅ Configured")
        # Show non-sensitive config
        # Convert ProviderConfig to dict
        config_dict = vars(config) if hasattr(config, '__dict__') else {}
        safe_config = {k: v for k, v in config_dict.items() if k != 'api_key'}
        if 'api_key' in config_dict and config_dict['api_key']:
            key_len = len(config_dict['api_key'])
            safe_config['api_key'] = (
                '***' + config_dict['api_key'][-4:] if key_len > 4 else '****'
            )
        for key, value in safe_config.items():
            click.echo(f"   {key}: {value}")
    else:
        click.echo("   Status: ❌ Not configured")
        click.echo(f"   Run: swebench provider init {provider_name}")

    # Provider-specific information
    if provider_name == 'openai':
        click.echo("\n🌐 Additional Info:")
        click.echo("   API Docs: https://platform.openai.com/docs")
        click.echo("   Models: gpt-4-turbo-preview, gpt-4, gpt-3.5-turbo")
        click.echo("   Rate Limits: Varies by tier")

    elif provider_name == 'openrouter':
        click.echo("\n🌐 Additional Info:")
        click.echo("   API Docs: https://openrouter.ai/docs")
        click.echo("   Browse Models: https://openrouter.ai/models")
        click.echo("   Features: Access to 100+ models from various providers")

        if config and getattr(config, 'api_key', None):
            click.echo("\n   Checking available models...")
            try:
                # Create provider to get models
                # provider = provider_class(**config)
                # sync_provider = SyncProviderWrapper(provider)

                # Note: This would require implementing a list_models method
                # For now, just show the common ones
                click.echo("   Popular models: claude-3, gpt-4, llama-3, gemini-pro")
            except Exception:  # noqa: S110
                pass

    elif provider_name == 'mock':
        click.echo("\n🧪 Mock Provider:")
        click.echo("   Purpose: Testing and development")
        click.echo("   Behavior: Returns canned responses")
        click.echo("   Cost: Always $0.00")


if __name__ == "__main__":
    provider_cli()
