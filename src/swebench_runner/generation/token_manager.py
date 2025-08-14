"""Token management for model context windows."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .prompt_builder import PromptContext

logger = logging.getLogger(__name__)


class TruncationStrategy(Enum):
    """Strategies for truncating content to fit token limits."""

    BALANCED = "balanced"  # Reduce all sections proportionally
    PRESERVE_TESTS = "preserve_tests"  # Keep test info, reduce code
    PRESERVE_RECENT = "preserve_recent"  # Keep recent code, reduce old
    SUMMARY = "summary"  # Summarize removed content


@dataclass
class FitStats:
    """Statistics about prompt fitting process."""

    original_tokens: int
    final_tokens: int
    truncated: bool
    sections_truncated: dict[str, int] = field(default_factory=dict)
    strategy_used: TruncationStrategy | None = None


class TokenManager:
    """Manages token counting and context window limits."""

    # Model context limits (as of early 2024)
    MODEL_LIMITS = {
        # OpenAI
        "gpt-4-turbo-preview": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-3.5-turbo": 16385,
        "gpt-3.5-turbo-16k": 16385,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,

        # Anthropic (via OpenRouter or direct)
        "anthropic/claude-3-opus": 200000,
        "anthropic/claude-3-sonnet": 200000,
        "anthropic/claude-3-haiku": 200000,
        "anthropic/claude-2.1": 200000,
        "anthropic/claude-2": 100000,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "claude-2.1": 200000,
        "claude-2": 100000,

        # Default fallback
        "default": 4096
    }

    # Cost per 1M tokens (approximate, as of early 2024)
    MODEL_COSTS = {
        # OpenAI - input/output costs
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},

        # Anthropic
        "claude-3-opus": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
    }

    def __init__(
        self,
        cache_enabled: bool = True,
        cache_size: int = 10000,
        fallback_estimation: bool = True,
        default_chars_per_token: float = 4.0
    ):
        """Initialize the token manager.

        Args:
            cache_enabled: Whether to cache token counts
            cache_size: Maximum size of the token count cache
            fallback_estimation: Whether to use estimation when tokenizer unavailable
            default_chars_per_token: Default character-to-token ratio for estimation
        """
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size
        self.fallback_estimation = fallback_estimation
        self.default_chars_per_token = default_chars_per_token

        # Track total tokens counted
        self.total_tokens_counted = 0

        # Initialize tiktoken support
        self._tiktoken_available = False
        self._tiktoken_encodings: dict[str, Any] = {}
        self._init_tiktoken()

        # Set up caching if enabled
        if self.cache_enabled:
            # Create cached version of the method
            self._count_tokens_cached = lru_cache(maxsize=cache_size)(self._count_tokens_uncached)
        else:
            # Use uncached version
            self._count_tokens_cached = self._count_tokens_uncached

    def _init_tiktoken(self) -> None:
        """Initialize tiktoken for OpenAI models."""
        try:
            import tiktoken
            self._tiktoken_available = True
            logger.debug("tiktoken is available for accurate token counting")

            # Pre-load common encodings
            try:
                # cl100k_base is used by GPT-4 and GPT-3.5-turbo models
                self._tiktoken_encodings["cl100k_base"] = (
                    tiktoken.get_encoding("cl100k_base")
                )
            except Exception as e:
                logger.warning(f"Failed to load cl100k_base encoding: {e}")

        except ImportError:
            self._tiktoken_available = False
            logger.info(
                "tiktoken not available, will use estimation for token counting"
            )

    def count_tokens(self, text: str, model: str = "gpt-3.5-turbo", use_tiktoken: bool | None = None) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for
            model: Model name for tokenizer selection
            use_tiktoken: Force use/skip tiktoken (None = auto)

        Returns:
            Number of tokens
        """
        if self.cache_enabled:
            count = self._count_tokens_cached(text, model, use_tiktoken)
        else:
            count = self._count_tokens_uncached(text, model, use_tiktoken)

        # Defensive: ensure we always return an integer token count
        if not isinstance(count, int):
            count = self._estimate_tokens(text)

        # Track usage
        self.total_tokens_counted += count
        return count

    # Backward-compatibility shim for tests that patch this symbol
    def _count_tokens(self, text: str, model: str = "gpt-3.5-turbo", use_tiktoken: bool | None = None) -> int:  # noqa: D401
        """Alias to the uncached counter; exists for test patching."""
        return self._count_tokens_uncached(text, model, use_tiktoken)

    def _count_tokens_uncached(self, text: str, model: str = "gpt-3.5-turbo", use_tiktoken: bool | None = None) -> int:
        """Count tokens without caching."""
        # Decide whether to use tiktoken
        if use_tiktoken is False:
            # Explicitly skip tiktoken
            return self._estimate_tokens(text)
        elif use_tiktoken is True:
            # Explicitly require tiktoken
            if not self._tiktoken_available:
                raise ValueError("tiktoken requested but not available")
            return self._try_tiktoken(text, model)
        else:
            # Auto mode - try tiktoken first if available
            if self._tiktoken_available and self._is_openai_model(model):
                try:
                    tk_count = self._try_tiktoken(text, model)
                    if isinstance(tk_count, int):
                        return tk_count
                    # Fall through to estimation if tiktoken returned non-int (e.g., mocked None)
                    logger.warning("tiktoken returned non-integer count; falling back to estimation")
                except Exception as e:
                    logger.warning(f"Failed to count tokens with tiktoken: {e}")

            # Fall back to estimation
            if self.fallback_estimation:
                return self._estimate_tokens(text)
            else:
                raise ValueError(f"No tokenizer available for model {model}")

    def _try_tiktoken(self, text: str, model: str) -> int:
        """Try to count tokens using tiktoken."""
        if not self._tiktoken_available:
            raise ValueError("tiktoken not available")
        return self._count_openai_tokens(text, model)

    def _is_openai_model(self, model: str) -> bool:
        """Check if model is an OpenAI model."""
        openai_prefixes = [
            "gpt-", "text-davinci", "text-curie", "text-babbage", "text-ada"
        ]
        return any(model.startswith(prefix) for prefix in openai_prefixes)

    def _count_openai_tokens(self, text: str, model: str) -> int:
        """Count tokens using tiktoken for OpenAI models."""
        if not self._tiktoken_available:
            raise ValueError("tiktoken not available")

        # Get the appropriate encoding
        encoding_name = self._get_tiktoken_encoding_name(model)

        if encoding_name not in self._tiktoken_encodings:
            import tiktoken
            try:
                self._tiktoken_encodings[encoding_name] = (
                    tiktoken.get_encoding(encoding_name)
                )
            except Exception as e:
                logger.warning(f"Failed to load encoding {encoding_name}: {e}")
                raise

        encoding = self._tiktoken_encodings[encoding_name]

        # Count tokens
        tokens = encoding.encode(text)
        return len(tokens)

    def _get_tiktoken_encoding_name(self, model: str) -> str:
        """Get the tiktoken encoding name for a model."""
        # GPT-4 and GPT-3.5-turbo use cl100k_base
        if "gpt-4" in model or "gpt-3.5-turbo" in model:
            return "cl100k_base"
        # Older models use different encodings
        elif "text-davinci" in model:
            return "p50k_base"
        else:
            # Default to cl100k_base for unknown models
            return "cl100k_base"

    def _estimate_tokens(self, text: str, chars_per_token: float | None = None) -> int:
        """Estimate token count based on character count.

        Args:
            text: The text to estimate tokens for
            chars_per_token: Characters per token ratio (default: 4.0)

        Returns:
            Estimated number of tokens
        """
        if not text:
            return 0

        chars_per_token = chars_per_token or self.default_chars_per_token

        # Basic estimation
        base_estimate = len(text) / chars_per_token

        # Adjust for special tokens and whitespace
        # Count newlines and special punctuation as extra tokens
        special_chars = text.count('\n') + text.count('.') + text.count(',')

        # Add some buffer for tokenizer overhead
        overhead = 0.1 * base_estimate

        return int(base_estimate + special_chars * 0.5 + overhead)

    # Keep original signature expected by many tests: get_model_limit(model: str, provider: str = "")
    def get_model_limit(self, model: str, provider: str = "") -> int:
        """Get token limit for a model.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            model: The model name

        Returns:
            Token limit for the model
        """
        # Normalize provider/model argument order from tests which may pass (provider, model)
        def _looks_like_model(name: str) -> bool:
            return (
                name in self.MODEL_LIMITS or
                name.startswith("gpt-") or
                "claude" in name or
                "/" in name
            )
        def _looks_like_provider(name: str) -> bool:
            return name in ("openai", "anthropic", "ollama", "openrouter", "any", "unknown")

        # Handle cases where first arg is empty and second carries the model
        if not model and provider:
            model, provider = provider, ""

        if _looks_like_provider(model) and _looks_like_model(provider):
            model, provider = provider, model

        # Check exact match first
        if model in self.MODEL_LIMITS:
            return self.MODEL_LIMITS[model]

        # Check with provider prefix if provided
        if provider:
            provider_model = f"{provider}/{model}"
            if provider_model in self.MODEL_LIMITS:
                return self.MODEL_LIMITS[provider_model]
            # Also allow reverse (model without provider) if the base model exists
            if model in self.MODEL_LIMITS:
                return self.MODEL_LIMITS[model]

        # Check for partial matches (e.g., "gpt-4" in "gpt-4-0613")
        if model:
            for known_model, limit in self.MODEL_LIMITS.items():
                if known_model in model or model in known_model:
                    return limit

        # Special-case fallback for common aliases
        if model.startswith("gpt-3.5-turbo"):
            return self.MODEL_LIMITS.get("gpt-3.5-turbo", self.MODEL_LIMITS["default"])

        # Return conservative default
        logger.warning(f"Unknown model {model}, using default limit")
        return self.MODEL_LIMITS["default"]

    def fit_context(
        self,
        context: PromptContext,
        model: str,
        reserve_tokens: int = 1000,
        strategy: TruncationStrategy = TruncationStrategy.BALANCED
    ) -> tuple[str, bool, FitStats]:
        """Fit context within model limits.

        Args:
            context: The prompt context to fit
            model: The model name
            reserve_tokens: Tokens to reserve for response
            strategy: Truncation strategy to use

        Returns:
            Tuple of (prompt, truncated, stats)
        """
        # Import here to avoid circular dependency
        from .prompt_builder import PromptBuilder

        # Get model limit
        provider = self._extract_provider(model)
        max_tokens = self.get_model_limit(model, provider) - reserve_tokens

        # Build initial prompt
        builder = PromptBuilder()
        initial_prompt = builder.build_prompt(context)
        initial_tokens = self.count_tokens(initial_prompt, model)

        # Check if it fits
        if initial_tokens <= max_tokens:
            stats = FitStats(
                original_tokens=initial_tokens,
                final_tokens=initial_tokens,
                truncated=False
            )
            return initial_prompt, False, stats

        # Need to truncate
        logger.info(
            f"Context too large ({initial_tokens} tokens), "
            f"truncating to fit {max_tokens}"
        )

        # Truncate context
        truncated_context = self.truncate_context(
            context, max_tokens, strategy
        )

        # Build truncated prompt
        final_prompt = builder.build_prompt(truncated_context)
        final_tokens = self.count_tokens(final_prompt, model)

        # Create stats
        stats = FitStats(
            original_tokens=initial_tokens,
            final_tokens=final_tokens,
            truncated=True,
            strategy_used=strategy,
            sections_truncated=self._calculate_section_differences(
                context, truncated_context
            )
        )

        return final_prompt, True, stats

    def truncate_context(
        self,
        context: PromptContext,
        max_tokens: int,
        strategy: TruncationStrategy = TruncationStrategy.BALANCED
    ) -> PromptContext:
        """Truncate context to fit token limit.

        Args:
            context: The context to truncate
            max_tokens: Maximum tokens allowed
            strategy: Truncation strategy to use

        Returns:
            Truncated context
        """
        # Estimate current token usage by section
        section_tokens = self._estimate_section_tokens(context)
        total_tokens = sum(section_tokens.values())

        if total_tokens <= max_tokens:
            return context

        tokens_to_remove = total_tokens - max_tokens

        # Apply strategy
        if strategy == TruncationStrategy.BALANCED:
            return self._truncate_balanced(context, tokens_to_remove, section_tokens)
        elif strategy == TruncationStrategy.PRESERVE_TESTS:
            return self._truncate_preserve_tests(
                context, tokens_to_remove, section_tokens
            )
        elif strategy == TruncationStrategy.PRESERVE_RECENT:
            return self._truncate_preserve_recent(
                context, tokens_to_remove, section_tokens
            )
        else:
            # Default to balanced
            return self._truncate_balanced(context, tokens_to_remove, section_tokens)

    def _truncate_balanced(
        self,
        context: PromptContext,
        tokens_to_remove: int,
        section_tokens: dict[str, int]
    ) -> PromptContext:
        """Remove tokens proportionally from each section."""
        # Import here to avoid circular dependency
        from .prompt_builder import PromptContext

        # Calculate reduction factor
        total_tokens = sum(section_tokens.values())
        reduction_factor = 1.0 - (tokens_to_remove / total_tokens)

        # Create new context with reduced content
        new_context = PromptContext(
            system_prompt=context.system_prompt,
            user_prompt=context.user_prompt,
            problem_statement=self._truncate_text(
                context.problem_statement,
                int(section_tokens.get("problem_statement", 0) * reduction_factor),
                preserve_structure=True
            ) if context.problem_statement else "",
            instance_id=context.instance_id,
            repo_name=context.repo_name,
            base_commit=context.base_commit,
            hints=context.hints[:max(1, int(len(context.hints) * reduction_factor))],
            metadata=context.metadata.copy()
        )

        # Truncate code context
        if context.code_context:
            code_budget = int(section_tokens.get("code_context", 0) * reduction_factor)
            new_context.code_context = self._truncate_file_dict(
                context.code_context, code_budget, preserve_structure=True
            )

        # Truncate test context
        if context.test_context:
            test_budget = int(section_tokens.get("test_context", 0) * reduction_factor)
            new_context.test_context = self._truncate_file_dict(
                context.test_context, test_budget, preserve_structure=False
            )

        return new_context

    def _truncate_preserve_tests(
        self,
        context: PromptContext,
        tokens_to_remove: int,
        section_tokens: dict[str, int]
    ) -> PromptContext:
        """Preserve test information, truncate code context."""
        # Import here to avoid circular dependency
        from .prompt_builder import PromptContext

        # Calculate what we can remove from non-test sections
        removable_tokens = (
            section_tokens.get("code_context", 0) +
            section_tokens.get("problem_statement", 0) // 2
        )

        if removable_tokens < tokens_to_remove:
            # Still need to truncate tests somewhat
            test_reduction = (
                (tokens_to_remove - removable_tokens) /
                section_tokens.get("test_context", 1)
            )
            test_reduction = min(0.5, test_reduction)  # Keep at least 50% of tests
        else:
            test_reduction = 0

        # Create new context
        new_context = PromptContext(
            system_prompt=context.system_prompt,
            user_prompt=context.user_prompt,
            problem_statement=self._truncate_text(
                context.problem_statement,
                section_tokens.get("problem_statement", 0) * 3 // 4,  # Keep 75%
                preserve_structure=True
            ) if context.problem_statement else "",
            instance_id=context.instance_id,
            repo_name=context.repo_name,
            base_commit=context.base_commit,
            hints=context.hints,  # Keep all hints
            metadata=context.metadata.copy()
        )

        # Aggressively truncate code context
        if context.code_context:
            # Calculate remaining budget after preserving tests
            test_size = int(
                section_tokens.get("test_context", 0) * (1 - test_reduction)
            )
            problem_size = len(new_context.problem_statement) // 4

            # Calculate total tokens and ensure we have a positive budget
            total_current = sum(section_tokens.values())
            available_tokens = max(100, total_current - tokens_to_remove)

            code_budget = max(10, available_tokens - test_size - problem_size)
            new_context.code_context = self._truncate_file_dict(
                context.code_context, max(10, code_budget // 2), preserve_structure=True
            )

        # Preserve test context
        if context.test_context:
            if test_reduction > 0:
                test_budget = int(
                    section_tokens.get("test_context", 0) * (1 - test_reduction)
                )
                new_context.test_context = self._truncate_file_dict(
                    context.test_context, test_budget, preserve_structure=False
                )
            else:
                new_context.test_context = context.test_context.copy()

        return new_context

    def _truncate_preserve_recent(
        self,
        context: PromptContext,
        tokens_to_remove: int,
        section_tokens: dict[str, int]
    ) -> PromptContext:
        """Preserve recent/relevant code, truncate older parts."""
        # For now, implement similar to balanced but prioritize keeping
        # files that appear to be more relevant (smaller files, test files)
        return self._truncate_balanced(context, tokens_to_remove, section_tokens)

    def _truncate_text(
        self,
        text: str,
        max_tokens: int,
        preserve_structure: bool = True
    ) -> str:
        """Truncate text to fit token limit.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            preserve_structure: Whether to preserve code structure

        Returns:
            Truncated text
        """
        if not text:
            return text

        # Estimate current tokens
        current_tokens = self._estimate_tokens(text)
        if current_tokens <= max_tokens:
            return text

        # Calculate character budget (rough approximation)
        char_budget = int(max_tokens * self.default_chars_per_token)

        if not preserve_structure:
            # Simple truncation
            if len(text) <= char_budget:
                return text
            return text[:char_budget-20] + "\n[... truncated ...]"

        # Preserve structure (imports, signatures, etc.)
        lines = text.split('\n')

        # Separate structural and body lines
        structure_lines = []
        body_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            # Identify structural elements
            if (stripped.startswith(('import ', 'from ', 'class ', 'def ', '@')) or
                stripped.endswith(':') or
                (i < 10 and not stripped)):  # Keep early empty lines for structure
                structure_lines.append((i, line))
            else:
                body_lines.append((i, line))

        # Build truncated version
        result_lines = []

        # Sort structure lines by original position to maintain order
        structure_lines = sorted(structure_lines, key=lambda x: x[0])

        # Add structure lines first
        structure_chars = sum(len(line) + 1 for _, line in structure_lines)
        if structure_chars < char_budget:
            for _, line in structure_lines:
                result_lines.append(line)
            remaining_budget = char_budget - structure_chars
        else:
            # Even structure is too large, truncate it
            budget = char_budget // 2
            chars = 0
            for _, line in structure_lines:
                if chars + len(line) + 1 < budget:
                    result_lines.append(line)
                    chars += len(line) + 1
                else:
                    break
            remaining_budget = char_budget - chars

        # Add body lines
        if remaining_budget > 100 and body_lines:
            # Add from beginning and end
            chars = 0
            added_truncation = False

            # Add from beginning
            for i, (_, line) in enumerate(body_lines):
                if chars + len(line) + 1 < remaining_budget // 2:
                    result_lines.append(line)
                    chars += len(line) + 1
                else:
                    truncated_count = len(body_lines) - i * 2
                    result_lines.append(
                        f"\n[... {truncated_count} lines truncated ...]"
                    )
                    added_truncation = True
                    break

            # Add from end if space remains
            if remaining_budget - chars > 100:
                end_lines: list[str] = []
                end_chars = 0
                for _, line in reversed(body_lines):
                    if end_chars + len(line) + 1 < (remaining_budget - chars):
                        end_lines.insert(0, line)
                        end_chars += len(line) + 1
                    else:
                        break

                if end_lines and not added_truncation:
                    result_lines.append("\n[... truncated ...]")
                result_lines.extend(end_lines)
        elif body_lines and remaining_budget <= 100:
            # Not enough budget for body lines, but we had body content
            result_lines.append("\n[... truncated ...]")

        return '\n'.join(result_lines)

    def _truncate_file_dict(
        self,
        files: dict[str, str],
        token_budget: int,
        preserve_structure: bool = True
    ) -> dict[str, str]:
        """Truncate a dictionary of files to fit token budget."""
        if not files:
            return {}

        # Sort files by importance (smaller files first, test files first)
        sorted_files = sorted(
            files.items(),
            key=lambda x: (
                "test" not in x[0].lower(),  # Prioritize test files
                len(x[1]),  # Prioritize smaller files
                x[0]  # Alphabetical as tiebreaker
            )
        )

        result = {}
        used_tokens = 0

        for filename, content in sorted_files:
            file_tokens = self._estimate_tokens(content)

            if used_tokens + file_tokens <= token_budget:
                # File fits completely
                result[filename] = content
                used_tokens += file_tokens
            elif used_tokens < token_budget * 0.8:  # Still have significant budget
                # Truncate file to fit remaining budget
                remaining = token_budget - used_tokens
                truncated = self._truncate_text(
                    content, remaining, preserve_structure
                )
                result[filename] = truncated
                used_tokens += self._estimate_tokens(truncated)
            else:
                # No more budget
                break

        return result

    def _estimate_section_tokens(self, context: PromptContext) -> dict[str, int]:
        """Estimate tokens used by each section of the context."""
        tokens = {}

        # System and user prompts
        tokens["system_prompt"] = self._estimate_tokens(context.system_prompt)
        tokens["user_prompt"] = self._estimate_tokens(context.user_prompt)

        # Problem statement
        tokens["problem_statement"] = self._estimate_tokens(context.problem_statement)

        # Code context
        code_tokens = sum(
            self._estimate_tokens(content)
            for content in context.code_context.values()
        )
        tokens["code_context"] = code_tokens

        # Test context
        test_tokens = sum(
            self._estimate_tokens(content)
            for content in context.test_context.values()
        )
        tokens["test_context"] = test_tokens

        # Hints
        hint_tokens = sum(
            self._estimate_tokens(hint)
            for hint in context.hints
        )
        tokens["hints"] = hint_tokens

        return tokens

    def _calculate_section_differences(
        self,
        original: PromptContext,
        truncated: PromptContext
    ) -> dict[str, int]:
        """Calculate tokens removed from each section."""
        original_tokens = self._estimate_section_tokens(original)
        truncated_tokens = self._estimate_section_tokens(truncated)

        differences = {}
        for section in original_tokens:
            diff = original_tokens[section] - truncated_tokens.get(section, 0)
            if diff > 0:
                differences[section] = diff

        return differences

    def _extract_provider(self, model: str) -> str:
        """Extract provider from model name."""
        # Check for explicit provider prefix
        if "/" in model:
            return model.split("/")[0]

        # Infer from model name
        if model.startswith("gpt-"):
            return "openai"
        elif "claude" in model:
            return "anthropic"
        else:
            return "unknown"

    def estimate_cost(self, *args, **kwargs) -> float | None:
        """Estimate generation cost.

        Supports both keyword usage and legacy positional usage.

        Returns:
            Estimated cost in USD (0.0 if model not found)
        """
        # Normalize arguments
        if kwargs and 'model' in kwargs:
            model = kwargs.get("model")
            prompt_tokens = kwargs.get("prompt_tokens")
            max_completion_tokens = kwargs.get("max_completion_tokens")
        else:
            # Legacy overloads:
            # (model, input_tokens, output_tokens)
            if len(args) >= 3 and isinstance(args[0], str):
                model = args[0]
                prompt_tokens = args[1]
                max_completion_tokens = args[2]
            else:
                # (input_tokens, output_tokens, provider, model)
                prompt_tokens = args[0] if len(args) > 0 else 0
                max_completion_tokens = args[1] if len(args) > 1 else 0
                model = args[3] if len(args) > 3 else None

        if model is None:
            return None

        # Look up model costs
        cost_info = None

        # Check exact match
        if model in self.MODEL_COSTS:
            cost_info = self.MODEL_COSTS[model]
        else:
            # Check partial matches
            for known_model, costs in self.MODEL_COSTS.items():
                if known_model in model or model in known_model:
                    cost_info = costs
                    break

        if not cost_info:
            return None

        # Calculate cost (per million tokens)
        input_cost = (prompt_tokens / 1_000_000) * cost_info["input"]
        output_cost = (max_completion_tokens / 1_000_000) * cost_info["output"]

        return input_cost + output_cost

    def check_cost_warnings(self, model: str, input_tokens: int, output_tokens: int) -> list[str]:
        """Check for cost warnings based on token usage.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            List of warning messages if operation would be expensive
        """
        warnings = []

        # Calculate estimated cost
        cost = self.estimate_cost(model, input_tokens, output_tokens)

        # Warn if cost exceeds thresholds
        if cost > 10.0:
            warnings.append(f"High cost warning: Estimated cost ${cost:.2f} exceeds $10")
        elif cost > 5.0:
            warnings.append(f"Cost warning: Estimated cost ${cost:.2f} exceeds $5")
        elif cost > 1.0:
            warnings.append(f"Moderate cost: Estimated cost ${cost:.2f}")

        # Warn about large token counts
        total_tokens = input_tokens + output_tokens
        if total_tokens > 100000:
            warnings.append(f"Very high token usage: {total_tokens:,} tokens")
        elif total_tokens > 50000:
            warnings.append(f"High token usage: {total_tokens:,} tokens")

        # Warn if approaching context limits
        if model in self.MODEL_LIMITS:
            limit = self.MODEL_LIMITS[model]
            usage_pct = (input_tokens / limit) * 100
            if usage_pct > 90:
                warnings.append(f"Near context limit: {usage_pct:.1f}% of {limit:,} token limit")
            elif usage_pct > 75:
                warnings.append(f"High context usage: {usage_pct:.1f}% of {limit:,} token limit")

        # Add expensive model warning
        expensive_models = ["gpt-4", "gpt-4-turbo", "claude-3-opus"]
        if any(exp in model.lower() for exp in expensive_models):
            if total_tokens > 10000:
                warnings.append(f"Using expensive model {model} with {total_tokens:,} tokens")

        return warnings

    def truncate_content(
        self,
        content: dict[str, str],
        target_tokens: int,
        strategy: TruncationStrategy = TruncationStrategy.BALANCED,
        model: str = "gpt-3.5-turbo"
    ) -> dict[str, str]:
        """Truncate structured content to fit within token limit.

        Args:
            content: Dictionary of content sections
            target_tokens: Target total token count
            strategy: Truncation strategy to use
            model: Model name for token counting

        Returns:
            Truncated content dictionary
        """
        if not content:
            return {}

        # Count tokens in each section
        section_tokens = {}
        total_tokens = 0
        for key, text in content.items():
            tokens = self.count_tokens(text, model)
            section_tokens[key] = tokens
            total_tokens += tokens

        # If it fits, return as-is
        if total_tokens <= target_tokens:
            return content.copy()

        # Apply truncation strategy
        truncated = {}

        if strategy == TruncationStrategy.BALANCED:
            # Reduce all sections proportionally
            reduction_ratio = target_tokens / total_tokens

            for key, text in content.items():
                section_target = int(section_tokens[key] * reduction_ratio)
                if section_target > 0:
                    # Truncate this section
                    char_estimate = int(len(text) * reduction_ratio * 0.95)
                    truncated[key] = text[:char_estimate]
                else:
                    truncated[key] = ""

        elif strategy == TruncationStrategy.PRESERVE_TESTS:
            # Keep test sections, reduce others
            test_keys = [k for k in content.keys() if 'test' in k.lower()]
            other_keys = [k for k in content.keys() if 'test' not in k.lower()]

            # Reserve tokens for tests
            test_tokens = sum(section_tokens[k] for k in test_keys if k in section_tokens)
            remaining_tokens = max(0, target_tokens - test_tokens)

            # Add all test content
            for key in test_keys:
                if key in content:
                    truncated[key] = content[key]

            # Truncate other content to fit remaining space
            if remaining_tokens > 0 and other_keys:
                other_total = sum(section_tokens[k] for k in other_keys if k in section_tokens)
                if other_total > 0:
                    reduction_ratio = remaining_tokens / other_total
                    for key in other_keys:
                        if key in content:
                            char_estimate = int(len(content[key]) * reduction_ratio * 0.95)
                            truncated[key] = content[key][:char_estimate]
        else:
            # Default: use balanced
            return self.truncate_content(content, target_tokens, TruncationStrategy.BALANCED, model)

        return truncated

    def fit_to_limit(self, text: str, max_tokens: int, model: str = "gpt-3.5-turbo", reserve_tokens: int = 0, **kwargs) -> FitStats:
        """Fit text within a token limit by truncating if necessary.

        Args:
            text: Text to fit
            max_tokens: Maximum number of tokens allowed
            model: Model name for token counting
            reserve_tokens: Additional tokens to reserve (reduces effective limit)

        Returns:
            FitStats object with truncation information
        """
        # Adjust max tokens for reservation
        effective_limit = max_tokens - reserve_tokens
        if effective_limit <= 0:
            raise ValueError(f"No space after reserving {reserve_tokens} tokens from {max_tokens}")
        if not text:
            return FitStats(
                original_tokens=0,
                final_tokens=0,
                truncated=False
            )

        # Count original tokens
        # Use estimation for deterministic budget enforcement across environments
        original_tokens = self.count_tokens(text, model, use_tiktoken=False)

        # If it fits, return as-is
        if original_tokens <= effective_limit:
            return FitStats(
                original_tokens=original_tokens,
                final_tokens=original_tokens,
                truncated=False
            )

        # Need to truncate
        # Estimate how much to keep based on character ratio
        char_ratio = effective_limit / original_tokens
        target_chars = int(len(text) * char_ratio * 0.95)  # Keep 95% to ensure we're under limit

        truncated_text = text[:target_chars]

        # Verify the truncated text fits
        final_tokens = self.count_tokens(truncated_text, model, use_tiktoken=False)

        # If still too large, truncate more aggressively
        while final_tokens > effective_limit and len(truncated_text) > 100:
            truncated_text = truncated_text[:int(len(truncated_text) * 0.9)]
            final_tokens = self.count_tokens(truncated_text, model)

        return FitStats(
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            truncated=True,
            sections_truncated={"text": original_tokens - final_tokens},
            strategy_used=TruncationStrategy.BALANCED
        )

    def fit_to_limit_with_stats(self, text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> FitStats:
        """Fit text within token limit and return statistics.

        This is an alias for fit_to_limit that explicitly returns stats.

        Args:
            text: Text to fit
            max_tokens: Maximum number of tokens allowed
            model: Model name for token counting

        Returns:
            FitStats object with truncation information
        """
        return self.fit_to_limit(text, max_tokens, model)

    def get_context_window_usage(self, prompt_tokens: int, model: str) -> float:
        """Get percentage of context window used.

        Args:
            prompt_tokens: Number of tokens in prompt
            model: Model name

        Returns:
            Percentage of context window used (0.0-1.0)
        """
        provider = self._extract_provider(model)
        limit = self.get_model_limit(model, provider)
        return prompt_tokens / limit

    def suggest_model_for_context(
        self,
        context_size: int,
        provider: str = "any"
    ) -> list[str]:
        """Suggest models that can handle the context size.

        Args:
            context_size: Required context size in tokens
            provider: Provider to filter by, or "any"

        Returns:
            List of suitable model names
        """
        suitable_models = []

        # Add buffer for response
        required_size = context_size + 2000

        for model, limit in self.MODEL_LIMITS.items():
            if limit >= required_size:
                # Check provider filter
                if provider == "any":
                    suitable_models.append(model)
                elif "/" in model and model.startswith(provider + "/"):
                    suitable_models.append(model)
                elif provider == "openai" and model.startswith("gpt-"):
                    suitable_models.append(model)
                elif provider == "anthropic" and "claude" in model:
                    suitable_models.append(model)

        # Sort by context size (smaller first, as they're usually cheaper)
        suitable_models.sort(key=lambda m: self.MODEL_LIMITS.get(m, 0))

        return suitable_models

    def format_truncation_summary(self, stats: FitStats) -> str:
        """Format a human-readable truncation summary.

        Args:
            stats: Truncation statistics

        Returns:
            Formatted summary string
        """
        if not stats.truncated:
            return f"Prompt fits within limit ({stats.final_tokens:,} tokens used)"

        lines = [
            f"Prompt truncated from {stats.original_tokens:,} to "
            f"{stats.final_tokens:,} tokens",
            f"Strategy: "
            f"{stats.strategy_used.value if stats.strategy_used else 'unknown'}"
        ]

        if stats.sections_truncated:
            lines.append("\nSections reduced:")
            for section, tokens_removed in stats.sections_truncated.items():
                percentage = (tokens_removed / stats.original_tokens) * 100
                lines.append(
                    f"  - {section}: {tokens_removed:,} tokens removed "
                    f"({percentage:.1f}%)"
                )

        return "\n".join(lines)
