"""Patch generation orchestrator for SWE-bench instances."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from swebench_runner.generation.response_parser import ResponseParser
from swebench_runner.providers import ModelProvider
from swebench_runner.providers.exceptions import (
    ProviderError,
    ProviderRateLimitError,
    ProviderTokenLimitError,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from patch generation."""

    patch: str | None
    instance_id: str
    model: str
    attempts: int
    truncated: bool
    cost: float
    success: bool
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PatchGenerator:
    """Orchestrates patch generation from SWE-bench instances."""

    def __init__(
        self,
        provider: ModelProvider,
        max_retries: int = 3,
        initial_temperature: float = 0.0,
        temperature_increment: float = 0.1,
        context_reduction_factor: float = 0.8,
        rate_limit_wait_seconds: int = 60,
        response_parser: ResponseParser | None = None,
    ):
        """Initialize the patch generator.

        Args:
            provider: The model provider to use for generation
            max_retries: Maximum number of generation attempts
            initial_temperature: Starting temperature for generation
            temperature_increment: How much to increase temperature on parse failures
            context_reduction_factor: Factor to reduce context by on token limit errors
            rate_limit_wait_seconds: How long to wait on rate limit errors
            response_parser: Optional custom ResponseParser instance
        """
        self.provider = provider
        self.max_retries = max_retries
        self.initial_temperature = initial_temperature
        self.temperature_increment = temperature_increment
        self.context_reduction_factor = context_reduction_factor
        self.rate_limit_wait_seconds = rate_limit_wait_seconds
        self.response_parser = response_parser or ResponseParser(
            auto_fix_common_issues=True,
            min_confidence=0.3
        )

    async def generate_patch(self, instance: dict[str, Any]) -> GenerationResult:
        """Generate a patch for a SWE-bench instance.

        Args:
            instance: SWE-bench instance data containing problem_statement, repo, etc.

        Returns:
            GenerationResult with the generated patch or error information
        """
        instance_id = instance.get("instance_id", "unknown")
        model = self.provider.config.model or self.provider.default_model or "unknown"

        logger.info(
            f"Starting patch generation for instance {instance_id} using {model}"
        )

        attempts = 0
        total_cost = 0.0
        current_temperature = self.initial_temperature
        truncated = False

        # Build initial prompt
        prompt = self._build_basic_prompt(instance)
        current_prompt = prompt

        while attempts < self.max_retries:
            attempts += 1
            logger.debug(
                f"Generation attempt {attempts}/{self.max_retries} for {instance_id}"
            )

            try:
                # Attempt generation with current parameters
                response = await self.provider.generate(
                    current_prompt,
                    temperature=current_temperature,
                    max_tokens=self.provider.config.max_tokens,
                )

                # Track cost
                if response.cost:
                    total_cost += response.cost

                # Extract patch from response using ResponseParser
                parse_result = self.response_parser.extract_patch(
                    response.content, instance
                )

                if (parse_result.patch and
                        parse_result.confidence >= self.response_parser.min_confidence):
                    logger.info(
                        f"Successfully generated patch for {instance_id} on attempt "
                        f"{attempts} (format: {parse_result.format_detected.value}, "
                        f"confidence: {parse_result.confidence:.2f})"
                    )
                    return GenerationResult(
                        patch=parse_result.patch,
                        instance_id=instance_id,
                        model=model,
                        attempts=attempts,
                        truncated=truncated,
                        cost=total_cost,
                        success=True,
                        metadata={
                            "final_temperature": current_temperature,
                            "response_length": len(response.content),
                            "finish_reason": response.finish_reason,
                            "format_detected": parse_result.format_detected.value,
                            "extraction_confidence": parse_result.confidence,
                            "extraction_issues": parse_result.issues,
                            "extraction_warnings": parse_result.metadata.get(
                                "validation_warnings", []
                            ),
                        }
                    )
                else:
                    # Parse failure - adjust temperature and retry
                    issues_str = (
                        ", ".join(parse_result.issues)
                        if parse_result.issues else "unknown"
                    )
                    logger.warning(
                        f"Failed to extract valid patch from response for "
                        f"{instance_id} (issues: {issues_str}), adjusting temperature"
                    )
                    current_temperature = min(
                        1.0, current_temperature + self.temperature_increment
                    )

            except ProviderTokenLimitError as e:
                logger.warning(f"Token limit exceeded for {instance_id}: {e}")
                # Reduce context and retry
                truncated = True
                current_prompt = self._reduce_context(current_prompt, instance)

            except ProviderRateLimitError as e:
                logger.warning(f"Rate limit hit for {instance_id}: {e}")
                # Wait and retry
                if attempts < self.max_retries:
                    logger.info(
                        f"Waiting {self.rate_limit_wait_seconds} seconds "
                        f"before retry..."
                    )
                    await asyncio.sleep(self.rate_limit_wait_seconds)

            except ProviderError as e:
                # Other provider errors - log and continue
                logger.error(f"Provider error for {instance_id}: {e}")

            except Exception as e:
                # Unexpected errors
                logger.exception(
                    f"Unexpected error generating patch for {instance_id}: {e}"
                )
                return GenerationResult(
                    patch=None,
                    instance_id=instance_id,
                    model=model,
                    attempts=attempts,
                    truncated=truncated,
                    cost=total_cost,
                    success=False,
                    error=f"Unexpected error: {str(e)}",
                )

        # All attempts failed
        logger.error(
            f"Failed to generate patch for {instance_id} after {attempts} attempts"
        )
        return GenerationResult(
            patch=None,
            instance_id=instance_id,
            model=model,
            attempts=attempts,
            truncated=truncated,
            cost=total_cost,
            success=False,
            error=f"Failed after {attempts} attempts",
        )

    async def generate_batch(
        self,
        instances: list[dict[str, Any]],
        concurrency: int = 5,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> list[GenerationResult]:
        """Generate patches for multiple instances.

        Args:
            instances: List of SWE-bench instances
            concurrency: Maximum number of concurrent generations
            progress_callback: Optional callback(instance_id, current, total)
                for progress updates

        Returns:
            List of GenerationResult objects, one per instance
        """
        logger.info(
            f"Starting batch generation for {len(instances)} instances with "
            f"concurrency={concurrency}"
        )

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrency)

        async def generate_with_semaphore(
            instance: dict[str, Any], index: int
        ) -> GenerationResult:
            """Generate patch with semaphore control."""
            async with semaphore:
                instance_id = instance.get("instance_id", f"unknown_{index}")

                if progress_callback:
                    progress_callback(instance_id, index, len(instances))

                result = await self.generate_patch(instance)

                if progress_callback:
                    progress_callback(instance_id, index + 1, len(instances))

                return result

        # Create tasks for all instances
        tasks = [
            generate_with_semaphore(instance, i)
            for i, instance in enumerate(instances)
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to GenerationResult objects
        final_results: list[GenerationResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                instance_id = instances[i].get("instance_id", f"unknown_{i}")
                logger.error(
                    f"Exception during batch generation for {instance_id}: {result}"
                )
                final_results.append(
                    GenerationResult(
                        patch=None,
                        instance_id=instance_id,
                        model=self.provider.config.model or "unknown",
                        attempts=0,
                        truncated=False,
                        cost=0.0,
                        success=False,
                        error=f"Batch generation exception: {str(result)}",
                    )
                )
            else:
                # result is a GenerationResult here
                assert isinstance(result, GenerationResult)
                final_results.append(result)

        # Log summary
        successful = sum(1 for r in final_results if r.success)
        total_cost = sum(r.cost for r in final_results)
        logger.info(
            f"Batch generation complete: {successful}/{len(instances)} successful, "
            f"total cost: ${total_cost:.4f}"
        )

        return final_results

    def _build_basic_prompt(self, instance: dict[str, Any]) -> str:
        """Build a basic prompt until PromptBuilder is available.

        Args:
            instance: SWE-bench instance data

        Returns:
            Formatted prompt string
        """
        repo = instance.get("repo", "the repository")
        problem = instance.get("problem_statement", "No problem statement provided")
        instance_id = instance.get("instance_id", "unknown")

        # Include additional context if available
        hints = instance.get("hints_text", "")
        test_patch = instance.get("test_patch", "")

        prompt_parts = [
            f"Fix the following issue in {repo}:",
            "",
            f"Problem: {problem}",
            "",
            f"Instance ID: {instance_id}",
        ]

        if hints:
            prompt_parts.extend(["", "Additional Context:", hints])

        if test_patch:
            prompt_parts.extend([
                "",
                "The following test was written to reproduce this issue:",
                "```diff",
                test_patch,
                "```"
            ])

        prompt_parts.extend([
            "",
            "Generate a patch in unified diff format that fixes this issue. "
            "The patch should:",
            "1. Be a valid unified diff",
            "2. Fix the described problem",
            "3. Be minimal and focused",
            "",
            "Return only the patch in unified diff format, starting with "
            "'diff --git' or '--- a/'."
        ])

        return "\n".join(prompt_parts)


    def _reduce_context(self, prompt: str, instance: dict[str, Any]) -> str:
        """Reduce prompt context when hitting token limits.

        Args:
            prompt: Current prompt that exceeded limits
            instance: Original instance data

        Returns:
            Reduced prompt
        """
        # For now, use a simple strategy: truncate the problem statement
        lines = prompt.split('\n')

        # Find and truncate the problem statement
        for i, line in enumerate(lines):
            if line.startswith("Problem:"):
                # Get the problem text
                problem_start = i
                problem_end = i + 1
                while (problem_end < len(lines) and
                       lines[problem_end].strip() and
                       not lines[problem_end].startswith((
                           "Instance ID:", "Additional Context:",
                           "The following test"
                       ))):
                    problem_end += 1

                # Calculate how much to keep
                problem_lines = lines[problem_start+1:problem_end]
                problem_text = '\n'.join(problem_lines)

                if len(problem_text) > 1000:  # Only truncate if it's long
                    # Keep a portion based on reduction factor
                    keep_chars = int(len(problem_text) * self.context_reduction_factor)
                    truncated_problem = problem_text[:keep_chars] + "\n... (truncated)"

                    # Rebuild the prompt
                    new_lines = (
                        lines[:problem_start+1] +
                        [truncated_problem] +
                        lines[problem_end:]
                    )

                    new_prompt_len = len('\n'.join(new_lines))
                    logger.info(
                        f"Reduced prompt from {len(prompt)} to {new_prompt_len} "
                        f"characters"
                    )
                    return '\n'.join(new_lines)

        # If we couldn't truncate the problem, try removing optional sections
        filtered_lines = []
        skip_section = False

        for line in lines:
            if line.startswith(
                ("Additional Context:", "The following test")
            ):
                skip_section = True
            elif line.strip() == "" and skip_section:
                skip_section = False
            elif not skip_section:
                filtered_lines.append(line)

        reduced = '\n'.join(filtered_lines)
        logger.info(
            f"Reduced prompt from {len(prompt)} to {len(reduced)} characters by "
            f"removing optional sections"
        )

        return reduced
