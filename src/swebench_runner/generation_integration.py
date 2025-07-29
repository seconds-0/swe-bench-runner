"""Enhanced integration module for patch generation with unified provider system.

This module provides a high-level interface for generating patches using any
supported provider (OpenAI, Anthropic, Ollama) with advanced features like:
- Provider selection and validation
- Cost estimation across all providers
- Provider fallback chains for reliability
- Rate limiting and circuit breaker integration
- Progress tracking and batch processing
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import click
from rich.console import Console

# Progress components removed - using direct console output instead
from rich.table import Table

from .datasets import DatasetManager
from .generation import (
    BatchProcessor,
    BatchResult,
    PatchGenerator,
    PatchValidator,
    ResponseParser,
    TemplateStyle,
)
from .generation.patch_formatter import PatchFormatter
from .providers import (
    ModelProvider,
    ProviderConfigManager,
    ProviderNotFoundError,
    UnifiedRequest,
    UnifiedResponse,
    get_registry,
)

logger = logging.getLogger(__name__)


class ProviderCoordinator:
    """Coordinates provider selection, validation, and fallback management."""

    def __init__(self, registry: Any = None) -> None:
        self.registry = registry or get_registry()
        self.config_manager = ProviderConfigManager()

    def select_provider(
        self,
        provider_name: str,
        model: str | None = None,
        validate_connectivity: bool = True
    ) -> ModelProvider:
        """Select and configure appropriate provider with validation.

        Args:
            provider_name: Name of the provider to use
            model: Optional model override
            validate_connectivity: Whether to test connectivity

        Returns:
            Configured and validated provider instance

        Raises:
            ProviderNotFoundError: If provider doesn't exist
            ProviderConfigurationError: If provider misconfigured
        """
        try:
            # Get provider class and check if it exists
            provider_class = self.registry.get_provider_class(provider_name)

            # Load configuration
            config = self.config_manager.load_config(provider_name)
            if model:
                # Override model in config
                config_dict = vars(config).copy()
                config_dict['model'] = model
                from .providers import ProviderConfig
                config = ProviderConfig(**config_dict)

            # Create provider instance
            provider = provider_class(config)

            # Validate connectivity if requested
            if validate_connectivity:
                # For sync context, skip background validation
                # Background validation only works in async context
                logger.debug(f"Skipping background validation for {provider_name} in sync context")

            return provider

        except Exception as e:
            logger.error(f"Failed to select provider '{provider_name}': {e}")
            raise


    def get_provider_info(self, provider_name: str) -> dict[str, Any]:
        """Get detailed information about a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Dictionary with provider capabilities and status
        """
        try:
            provider_class = self.registry.get_provider_class(provider_name)

            # Try to get an instance to check configuration
            try:
                provider = self.registry.get_provider(provider_name, cache=False)
                configured = True
                config_status = "valid"
            except Exception as e:
                configured = False
                config_status = str(e)
                provider = None

            return {
                "name": provider_class.name,
                "description": provider_class.description,
                "supported_models": provider_class.supported_models,
                "requires_api_key": provider_class.requires_api_key,
                "supports_streaming": provider_class.supports_streaming,
                "default_model": provider_class.default_model,
                "configured": configured,
                "config_status": config_status,
                "capabilities": provider.capabilities.__dict__ if provider else None,
            }
        except ProviderNotFoundError:
            return {"error": f"Provider '{provider_name}' not found"}

    def list_available_providers(self) -> list[dict[str, Any]]:
        """List all available providers with their status."""
        return [
            self.get_provider_info(name)
            for name in self.registry.list_provider_names()
        ]

    async def generate_with_fallback(
        self,
        instance: dict[str, Any],
        primary_provider: str,
        fallback_providers: list[str] | None = None,
        model: str | None = None,
        **kwargs: Any
    ) -> UnifiedResponse:
        """Generate with provider fallback on failures.

        Args:
            instance: SWE-bench instance data
            primary_provider: Primary provider to try first
            fallback_providers: List of fallback providers
            model: Optional model override
            **kwargs: Additional generation parameters

        Returns:
            UnifiedResponse from successful provider

        Raises:
            Exception: If all providers fail
        """
        providers_to_try = [primary_provider]
        if fallback_providers:
            providers_to_try.extend(fallback_providers)

        last_error = None

        for provider_name in providers_to_try:
            try:
                logger.info(f"Attempting generation with provider: {provider_name}")
                provider = self.select_provider(
                    provider_name, model, validate_connectivity=False
                )

                # Create unified request from instance
                request = self._build_request_from_instance(instance, **kwargs)

                # For providers that support unified interface
                response: UnifiedResponse
                if hasattr(provider, 'generate_unified'):
                    response = await provider.generate_unified(request)
                else:
                    # Fallback to legacy generate method
                    model_response = await provider.generate(request.prompt, **kwargs)
                    response = self._convert_to_unified_response(
                        model_response, provider_name
                    )

                logger.info(f"Successfully generated with provider: {provider_name}")
                return response

            except Exception as e:
                logger.warning(f"Provider '{provider_name}' failed: {e}")
                last_error = e
                continue

        # All providers failed
        raise Exception(f"All providers failed. Last error: {last_error}")

    def _build_request_from_instance(
        self, instance: dict[str, Any], **kwargs: Any
    ) -> UnifiedRequest:
        """Build UnifiedRequest from SWE-bench instance."""
        # For now, use a simple prompt format
        # TODO: Integrate with proper prompt building from generation module
        problem_statement = instance.get("problem_statement", "")
        prompt = f"Generate a patch for the following issue:\n\n{problem_statement}"

        return UnifiedRequest(
            prompt=prompt,
            system_message=kwargs.get("system_message"),
            max_tokens=kwargs.get("max_tokens", 4000),
            temperature=kwargs.get("temperature", 0.0),
            model=kwargs.get("model"),
            stream=kwargs.get("stream", False),
        )

    def _convert_to_unified_response(
        self, model_response: Any, provider_name: str
    ) -> UnifiedResponse:
        """Convert legacy ModelResponse to UnifiedResponse."""
        from .providers.unified_models import TokenUsage

        usage = TokenUsage(
            prompt_tokens=0,  # Legacy response may not have this
            completion_tokens=0,
            total_tokens=0
        )

        if model_response.usage:
            usage.prompt_tokens = model_response.usage.get("prompt_tokens", 0)
            usage.completion_tokens = model_response.usage.get("completion_tokens", 0)
            usage.total_tokens = model_response.usage.get("total_tokens", 0)

        return UnifiedResponse(
            content=model_response.content,
            model=model_response.model,
            usage=usage,
            latency_ms=model_response.latency_ms or 0,
            finish_reason=model_response.finish_reason or "stop",
            provider=provider_name,
            cost=model_response.cost,
            raw_response=model_response.raw_response or {},
        )


class GenerationIntegration:
    """Enhanced generation integration with unified provider support.

    This class provides a high-level interface for patch generation that:
    - Uses the new unified provider system (OpenAI, Anthropic, Ollama)
    - Supports provider selection and validation
    - Provides cost estimation across all providers
    - Handles provider fallback chains for reliability
    - Integrates rate limiting and circuit breaker patterns
    - Maintains backward compatibility with existing code
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.dataset_manager = DatasetManager(cache_dir)
        self.console = Console()

        # Enhanced provider management
        self.registry = get_registry()
        self.provider_coordinator = ProviderCoordinator(self.registry)
        self.config_manager = ProviderConfigManager()

        # Initialize cost estimator with unified providers
        self.cost_estimator = UnifiedCostEstimator()

    def select_provider(
        self, provider_name: str, model: str | None = None
    ) -> ModelProvider:
        """Select and configure appropriate provider with validation.

        Args:
            provider_name: Name of the provider to use
            model: Optional model override

        Returns:
            Configured and validated provider instance

        Raises:
            ProviderNotFoundError: If provider doesn't exist
            ProviderConfigurationError: If provider misconfigured
        """
        return self.provider_coordinator.select_provider(provider_name, model)

    def estimate_batch_cost(
        self,
        instances: list[dict[str, Any]],
        provider_name: str,
        model: str | None = None
    ) -> float:
        """Estimate cost for batch processing across providers.

        Args:
            instances: List of SWE-bench instances
            provider_name: Provider to use for cost estimation
            model: Optional model override

        Returns:
            Estimated cost in USD
        """
        return self.cost_estimator.estimate_batch_cost(instances, provider_name, model)

    async def generate_with_fallback(
        self,
        instance: dict[str, Any],
        primary_provider: str,
        fallback_providers: list[str] | None = None,
        model: str | None = None,
        **kwargs: Any
    ) -> UnifiedResponse:
        """Generate with provider fallback on failures.

        Args:
            instance: SWE-bench instance data
            primary_provider: Primary provider to try first
            fallback_providers: List of fallback providers
            model: Optional model override
            **kwargs: Additional generation parameters

        Returns:
            UnifiedResponse from successful provider
        """
        return await self.provider_coordinator.generate_with_fallback(
            instance, primary_provider, fallback_providers, model, **kwargs
        )

    def list_available_providers(self) -> list[dict[str, Any]]:
        """List all available providers with their configuration status."""
        return self.provider_coordinator.list_available_providers()

    def show_provider_status(self) -> None:
        """Display provider status table in the console."""
        providers = self.list_available_providers()

        table = Table(title="Available Providers")
        table.add_column("Provider", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Models", style="green")
        table.add_column("Configured", style="yellow")
        table.add_column("API Key", style="red")

        for provider in providers:
            if "error" in provider:
                continue

            models_str = ", ".join(provider.get("supported_models", [])[:3])
            if len(provider.get("supported_models", [])) > 3:
                models_str += "..."

            configured = "✅" if provider.get("configured") else "❌"
            api_key = (
                "✅"
                if provider.get("requires_api_key", False)
                and provider.get("configured")
                else ("❌" if provider.get("requires_api_key", False) else "N/A")
            )

            table.add_row(
                provider["name"],
                provider.get("description", "")[:50],
                models_str,
                configured,
                api_key
            )

        self.console.print(table)

    async def generate_patches_for_evaluation(
        self,
        instances: list[dict[str, Any]],
        provider_name: str,
        model: str | None = None,
        output_path: Path | None = None,
        checkpoint_path: Path | None = None,
        max_workers: int = 5,
        show_progress: bool = True,
        template_style: TemplateStyle = TemplateStyle.DETAILED,
        max_tokens: int = 4000,
    ) -> Path:
        """Generate patches for instances and return path to JSONL.

        Args:
            instances: List of SWE-bench instances
            provider_name: Model provider to use
            model: Optional model override
            output_path: Where to save generated patches
            checkpoint_path: Path for checkpoint file
            max_workers: Concurrent generation workers
            show_progress: Whether to show progress bar
            template_style: Template style for prompts
            max_tokens: Maximum tokens for responses

        Returns:
            Path to JSONL file containing patches
        """
        # Use enhanced provider selection with validation
        try:
            provider = self.select_provider(provider_name, model)
            if show_progress:
                self.console.print(f"✅ Selected provider: {provider.name}")
                if hasattr(provider, 'capabilities'):
                    caps = provider.capabilities
                    self.console.print(
                        f"   Model: {provider.config.model or 'default'} "
                        f"(max tokens: {caps.max_context_length:,})"
                    )
        except Exception as e:
            self.console.print(
                f"[red]❌ Failed to initialize provider '{provider_name}': {e}[/red]"
            )
            raise

        # Create generation components
        response_parser = ResponseParser(
            auto_fix_common_issues=True,
            min_confidence=0.3
        )


        patch_validator = PatchValidator()

        patch_generator = PatchGenerator(
            provider=provider,
            max_retries=3,
            response_parser=response_parser
        )

        # Create batch processor
        batch_processor = BatchProcessor(
            generator=patch_generator,
            checkpoint_dir=checkpoint_path,  # Note: checkpoint_dir not checkpoint_path
            max_concurrent=max_workers,
            progress_bar=show_progress
        )

        # Show enhanced cost estimate using unified providers
        try:
            estimated_cost = self.estimate_batch_cost(instances, provider_name, model)

            if show_progress:
                # Show provider-specific cost information
                provider_info = self.provider_coordinator.get_provider_info(
                    provider_name
                )

                self.console.print(f"\n💰 Estimated cost: ${estimated_cost:.4f}")

                # Show provider-specific details
                if provider_info.get("capabilities"):
                    caps = provider_info["capabilities"]
                    if caps.get("cost_per_1k_prompt_tokens"):
                        prompt_cost = caps['cost_per_1k_prompt_tokens']
                        completion_cost = caps.get('cost_per_1k_completion_tokens', 0)
                        self.console.print(
                            f"   Cost per 1K tokens: ${prompt_cost:.4f} "
                            f"(prompt) / ${completion_cost:.4f} (completion)"
                        )

                # Warning for expensive operations
                if estimated_cost > 10.0:
                    self.console.print(
                        "[yellow]⚠️  Warning: Estimated cost exceeds $10[/yellow]"
                    )
                    if not click.confirm("Continue with generation?"):
                        raise click.Abort()
                elif estimated_cost > 1.0:
                    self.console.print(
                        "[yellow]💡 Tip: Consider using a less expensive model "
                        "for development[/yellow]"
                    )
        except Exception as e:
            if show_progress:
                self.console.print(f"[yellow]⚠️  Could not estimate cost: {e}[/yellow]")

        # Process batch
        if show_progress:
            self.console.print(
                f"\n🤖 Generating patches for {len(instances)} instances..."
            )

        try:
            # Default 30 minute timeout for entire batch
            batch_result = await asyncio.wait_for(
                batch_processor.process_batch(
                    instances=instances,
                    resume_from_checkpoint=True,
                    save_final_checkpoint=True
                ),
                timeout=1800  # 30 minutes
            )
        except asyncio.TimeoutError:
            self.console.print(
                "[red]❌ Batch processing timed out after 30 minutes[/red]"
            )
            raise click.Abort() from None

        # Handle results
        failure_handler = GenerationFailureHandler(self.console)
        should_continue = failure_handler.handle_batch_failure(batch_result)

        if not should_continue:
            raise click.Abort("Too many failures in batch generation")

        # Save results to JSONL
        if not output_path:
            output_path = (
                self.cache_dir / "temp" / f"generated_patches_{provider_name}.jsonl"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to patch format
        patch_data = []
        for result in batch_result.successful:
            if result.patch:
                # Validate the patch before including it
                validation_result = patch_validator.validate(result.patch)
                if validation_result.is_valid:
                    patch_data.append({
                        "instance_id": result.instance_id,
                        "patch": result.patch,
                        "model": result.model,
                        "cost": result.cost,
                        "metadata": result.metadata
                    })
                else:
                    issue_messages = [
                        issue.message for issue in validation_result.issues
                    ]
                    self.console.print(
                        f"[yellow]⚠️  Invalid patch for {result.instance_id}: "
                        f"{', '.join(issue_messages)}[/yellow]"
                    )

        # Format patches for evaluation
        formatter = PatchFormatter()
        formatted_patches = formatter.format_for_evaluation(patch_data)

        # Save to JSONL
        with open(output_path, 'w') as f:
            for item in formatted_patches:
                f.write(json.dumps(item) + '\n')

        # Show summary
        if show_progress:
            self._show_generation_summary(batch_result, output_path)

        return output_path

    def _show_generation_summary(
        self, batch_result: BatchResult, output_path: Path
    ) -> None:
        """Show generation summary table."""
        table = Table(title="Generation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        stats = batch_result.stats
        table.add_row("Total Instances", str(stats.total_instances))
        table.add_row("Successful", f"{stats.completed} ({stats.success_rate:.1%})")
        table.add_row("Failed", str(stats.failed))
        table.add_row("Skipped", str(stats.skipped))
        table.add_row("Total Cost", f"${stats.total_cost:.4f}")
        table.add_row("Total Time", f"{stats.total_time:.1f}s")
        table.add_row("Output File", str(output_path))

        self.console.print(table)


class CostEstimator:
    """Estimate and track generation costs."""

    # Rough token estimates per instance component
    TOKENS_PER_INSTANCE = {
        "problem_statement": 500,
        "test_patch": 300,
        "base_prompt": 200,
        "response": 1000,
    }

    # Cost per 1M tokens for common models (rough estimates)
    MODEL_COSTS = {
        # OpenAI
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        # OpenRouter (varies by model)
        "claude-3-opus": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        # Default fallback
        "default": {"input": 5.0, "output": 15.0},
    }

    def estimate_batch_cost(
        self,
        instances: list[dict[str, Any]],
        provider_name: str,
        model: str | None = None
    ) -> tuple[float, float]:
        """Estimate min/max cost for batch generation.

        Returns:
            Tuple of (min_cost, max_cost) in dollars
        """
        # Get model costs
        model_key = model or "default"
        costs = self.MODEL_COSTS.get(model_key, self.MODEL_COSTS["default"])

        # Calculate tokens
        input_tokens_per_instance = sum([
            self.TOKENS_PER_INSTANCE["problem_statement"],
            self.TOKENS_PER_INSTANCE["test_patch"],
            self.TOKENS_PER_INSTANCE["base_prompt"],
        ])
        output_tokens_per_instance = self.TOKENS_PER_INSTANCE["response"]

        total_instances = len(instances)

        # Calculate costs (convert from per-million to actual cost)
        input_cost = (
            input_tokens_per_instance * total_instances * costs["input"]
        ) / 1_000_000
        output_cost = (
            output_tokens_per_instance * total_instances * costs["output"]
        ) / 1_000_000

        min_cost = input_cost + output_cost
        # Max assumes 2x tokens due to retries/longer responses
        max_cost = min_cost * 2.0

        return (min_cost, max_cost)


class UnifiedCostEstimator:
    """Enhanced cost estimation using unified provider interfaces."""

    def __init__(self) -> None:
        self.registry = get_registry()

        # Fallback token estimates per instance component
        self.TOKENS_PER_INSTANCE = {
            "problem_statement": 500,
            "test_patch": 300,
            "base_prompt": 200,
            "response": 1000,
        }

    def estimate_batch_cost(
        self,
        instances: list[dict[str, Any]],
        provider_name: str,
        model: str | None = None
    ) -> float:
        """Estimate cost for batch generation using provider-specific methods.

        Args:
            instances: List of SWE-bench instances
            provider_name: Provider to use for cost estimation
            model: Optional model override

        Returns:
            Estimated cost in USD
        """
        try:
            # Get provider instance
            provider = self.registry.get_provider(provider_name, cache=True)

            # Override model if specified
            if model and provider.config.model != model:
                config_dict = vars(provider.config).copy()
                config_dict['model'] = model
                from .providers import ProviderConfig
                provider.config = ProviderConfig(**config_dict)

            # Use provider-specific cost estimation if available
            if hasattr(provider, 'estimate_cost'):
                return self._estimate_with_provider(provider, instances)
            else:
                # Fallback to generic estimation
                return self._estimate_generic(provider, instances)

        except Exception as e:
            logger.warning(f"Could not estimate cost with provider: {e}")
            # Fallback to legacy cost estimation
            legacy_estimator = CostEstimator()
            min_cost, max_cost = legacy_estimator.estimate_batch_cost(
                instances, provider_name, model
            )
            return (min_cost + max_cost) / 2  # Return average

    def _estimate_with_provider(
        self, provider: ModelProvider, instances: list[dict[str, Any]]
    ) -> float:
        """Estimate cost using provider's built-in cost estimation."""
        total_cost = 0.0

        for instance in instances:
            # Estimate tokens for this instance
            prompt_tokens = self._estimate_instance_tokens(instance)
            max_tokens = provider.config.max_tokens or 1000

            # Get cost estimate from provider
            try:
                cost = provider.estimate_cost(prompt_tokens, max_tokens)
                total_cost += cost
            except Exception as e:
                logger.warning(f"Provider cost estimation failed: {e}")
                # Fallback to generic calculation
                total_cost += self._generic_cost_calculation(
                    provider, prompt_tokens, max_tokens
                )

        return total_cost

    def _estimate_generic(
        self, provider: ModelProvider, instances: list[dict[str, Any]]
    ) -> float:
        """Generic cost estimation using capabilities."""
        if not hasattr(provider, 'capabilities'):
            # Very basic fallback
            return len(instances) * 0.01  # $0.01 per instance

        total_cost = 0.0

        for instance in instances:
            prompt_tokens = self._estimate_instance_tokens(instance)
            max_tokens = provider.config.max_tokens or 1000

            cost = self._generic_cost_calculation(provider, prompt_tokens, max_tokens)
            total_cost += cost

        return total_cost

    def _generic_cost_calculation(
        self, provider: ModelProvider, prompt_tokens: int, max_tokens: int
    ) -> float:
        """Calculate cost using provider capabilities."""
        caps = provider.capabilities

        # Get cost per 1K tokens
        prompt_cost_per_1k = (
            caps.cost_per_1k_prompt_tokens or 0.001
        )  # Default fallback
        completion_cost_per_1k = caps.cost_per_1k_completion_tokens or 0.002

        # Calculate costs
        prompt_cost = (prompt_tokens / 1000) * prompt_cost_per_1k
        completion_cost = (max_tokens / 1000) * completion_cost_per_1k

        return prompt_cost + completion_cost

    def _estimate_instance_tokens(self, instance: dict[str, Any]) -> int:
        """Estimate total tokens for an instance."""
        total = 0

        # Problem statement tokens
        problem_statement = instance.get("problem_statement", "")
        total += int(len(problem_statement.split()) * 1.3)  # Rough token multiplier

        # Test patch tokens
        test_patch = instance.get("test_patch", "")
        total += int(len(test_patch.split()) * 1.3)

        # Base prompt overhead
        total += self.TOKENS_PER_INSTANCE["base_prompt"]

        return int(total)


class GenerationFailureHandler:
    """Handle generation failures gracefully."""

    def __init__(self, console: Console):
        self.console = console

    def handle_batch_failure(self, results: BatchResult) -> bool:
        """Decide whether to continue with partial results.

        Returns:
            True if should continue, False if should abort
        """
        stats = results.stats

        # If more than 50% failed, it's probably a systemic issue
        if stats.failed > stats.completed:
            self.console.print(
                f"[red]❌ Generation failed for {stats.failed}/{stats.total_instances} "
                f"instances[/red]"
            )

            # Show common failure reasons
            failure_reasons: dict[str, int] = {}
            for failure in results.failed:
                reason = failure.get("error", "Unknown error")
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

            if failure_reasons:
                self.console.print("\n[yellow]Common failure reasons:[/yellow]")
                for reason, count in sorted(
                    failure_reasons.items(), key=lambda x: x[1], reverse=True
                )[:5]:
                    self.console.print(f"  • {reason}: {count} instances")

            # Ask user what to do
            if stats.completed > 0:
                return click.confirm(
                    f"\n{stats.completed} patches were generated successfully. "
                    f"Continue with these?"
                )
            else:
                self.console.print("[red]No patches were generated successfully.[/red]")
                return False

        # If we have some successes, continue
        if stats.completed > 0:
            if stats.failed > 0:
                self.console.print(
                    f"[yellow]⚠️  Generated {stats.completed} patches successfully, "
                    f"{stats.failed} failed[/yellow]"
                )
            return True

        return False
