"""Integration module for patch generation with evaluation pipeline."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console
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
from .providers import ProviderConfigManager, get_registry


class GenerationIntegration:
    """Integrates patch generation with evaluation pipeline."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.dataset_manager = DatasetManager(cache_dir)
        self.console = Console()

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
        # Get provider
        registry = get_registry()
        config_manager = ProviderConfigManager()

        # Load provider config
        config = config_manager.load_config(provider_name)
        if model:
            # Override model in config - config is always a ProviderConfig
            # from load_config
            config_dict = vars(config).copy()
            config_dict['model'] = model
            from .providers import ProviderConfig
            config = ProviderConfig(**config_dict)

        # Create provider instance
        provider_class = registry.get_provider_class(provider_name)
        provider = provider_class(config)

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

        # Show cost estimate
        cost_estimator = CostEstimator()
        min_cost, max_cost = cost_estimator.estimate_batch_cost(
            instances, provider_name, model
        )

        if show_progress:
            self.console.print(
                f"\n💰 Estimated cost: ${min_cost:.2f} - ${max_cost:.2f}"
            )

            if max_cost > 10.0:
                self.console.print(
                    "[yellow]⚠️  Warning: Estimated cost exceeds $10[/yellow]"
                )
                if not click.confirm("Continue with generation?"):
                    raise click.Abort()

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
