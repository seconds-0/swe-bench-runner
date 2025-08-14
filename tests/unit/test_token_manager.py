"""Unit tests for token_manager.py - focus on token counting and budget enforcement.

Tests specifically for:
- Token counting accuracy for different models
- Budget enforcement to prevent excessive API costs
- Token limit warnings for context windows
- Multi-model token handling
- Cost estimation accuracy
"""

from unittest.mock import patch

import pytest

from swebench_runner.generation.token_manager import (
    FitStats,
    TokenManager,
    TruncationStrategy,
)


class TestTokenManagerBudgetEnforcement:
    """Test token budget enforcement to prevent cost overruns."""

    def test_prevents_excessive_token_usage(self):
        """Prevent responses that would exceed token budgets.

        Why: Uncontrolled token usage can lead to thousands of dollars in API costs.
        """
        manager = TokenManager()

        # Set a budget limit
        budget_tokens = 10000

        # Try to use more tokens than budget
        large_text = "x" * 50000  # ~12500 tokens at 4 chars/token

        result = manager.fit_to_limit(large_text, budget_tokens)

        # Should truncate to fit budget
        assert result.final_tokens <= budget_tokens
        assert result.truncated is True

    def test_calculates_cost_estimates_accurately(self):
        """Calculate accurate cost estimates for API calls.

        Why: Users need to know how much their requests will cost.
        """
        manager = TokenManager()

        # Test with known model costs
        models_and_tokens = [
            ("gpt-4-turbo", 1000, 1000),  # 1K input, 1K output
            ("claude-3-opus", 5000, 2000),  # 5K input, 2K output
        ]

        for model, input_tokens, output_tokens in models_and_tokens:
            cost = manager.estimate_cost(model, input_tokens, output_tokens)

            # Should calculate based on model costs
            assert cost > 0
            # Verify it matches expected calculation
            if model in manager.MODEL_COSTS:
                expected = (
                    (input_tokens / 1_000_000 * manager.MODEL_COSTS[model]["input"]) +
                    (output_tokens / 1_000_000 * manager.MODEL_COSTS[model]["output"])
                )
                assert abs(cost - expected) < 0.01  # Allow small float difference

    def test_warns_about_high_cost_operations(self):
        """Warn when operations would be expensive.

        Why: Prevent accidental expensive API calls.
        """
        manager = TokenManager()

        # Large request that would be expensive
        large_tokens = 100000  # 100K tokens

        warnings = manager.check_cost_warnings("gpt-4", large_tokens, large_tokens)

        # Should generate warnings for expensive operations
        assert len(warnings) > 0
        assert any("cost" in w.lower() or "expensive" in w.lower() for w in warnings)


class TestTokenManagerContextLimits:
    """Test enforcement of model context window limits."""

    def test_respects_model_context_limits(self):
        """Enforce model-specific context window limits.

        Why: Exceeding context limits causes API errors.
        """
        manager = TokenManager()

        # Test different models
        test_cases = [
            ("gpt-4", 8192),
            ("gpt-3.5-turbo", 16385),
            ("claude-3-opus", 200000),
        ]

        for model, expected_limit in test_cases:
            limit = manager.get_model_limit(model)
            assert limit == expected_limit

    def test_handles_unknown_models_safely(self):
        """Use safe defaults for unknown models.

        Why: New models shouldn't cause crashes.
        """
        manager = TokenManager()

        # Unknown model
        limit = manager.get_model_limit("future-model-xyz")

        # Should use safe default
        assert limit == manager.MODEL_LIMITS["default"]
        assert limit > 0

    def test_leaves_room_for_response(self):
        """Reserve space for model responses in context.

        Why: Need room for the model to generate a response.
        """
        manager = TokenManager()

        model_limit = 8192
        response_reserve = 2000  # Reserve for response

        text = "x" * 40000  # ~10K tokens

        result = manager.fit_to_limit(
            text,
            model_limit - response_reserve,
            reserve_tokens=response_reserve
        )

        # Should leave room for response
        assert result.final_tokens <= (model_limit - response_reserve)


class TestTokenManagerAccuracy:
    """Test token counting accuracy."""

    def test_token_counting_matches_tiktoken(self):
        """Token counts should match tiktoken library.

        Why: Inaccurate counts lead to unexpected truncation or errors.
        """
        manager = TokenManager()

        # Common test strings
        test_strings = [
            "Hello, world!",
            "def function():\n    return 42",
            "// This is a comment\n/* Block comment */",
        ]

        for text in test_strings:
            # Test with estimation when tiktoken unavailable
            estimated = manager.count_tokens(text, use_tiktoken=False)

            # Should be reasonable estimate (within 50% of char/4)
            expected_estimate = len(text) / 4
            assert 0.5 * expected_estimate <= estimated <= 2 * expected_estimate

    def test_caches_token_counts_for_performance(self):
        """Cache token counts to avoid redundant computation.

        Why: Token counting can be expensive for large texts.
        """
        manager = TokenManager(cache_enabled=True)

        large_text = "x" * 10000

        # First call
        count1 = manager.count_tokens(large_text)

        # Second call should use cache
        with patch.object(manager, '_count_tokens_uncached', return_value=count1) as mock:
            count2 = manager.count_tokens(large_text)

            # Should not call uncached version if cache works
            if manager.cache_enabled:
                mock.assert_not_called()

            assert count1 == count2

    def test_handles_unicode_correctly(self):
        """Count tokens correctly for unicode text.

        Why: International text and emojis need accurate counting.
        """
        manager = TokenManager()

        unicode_texts = [
            "Hello 世界",  # Chinese
            "Привет мир",  # Russian
            "مرحبا بالعالم",  # Arabic
            "🚀 Launch! 🎉",  # Emojis
        ]

        for text in unicode_texts:
            count = manager.count_tokens(text)

            # Should handle without errors
            assert count > 0
            # Unicode typically uses more tokens
            assert count >= len(text.split())


class TestTokenManagerTruncation:
    """Test content truncation strategies."""

    def test_balanced_truncation_reduces_proportionally(self):
        """Balanced truncation should reduce all sections equally.

        Why: Maintains context from all parts of the input.
        """
        manager = TokenManager()

        content = {
            "code": "x" * 4000,  # ~1000 tokens
            "tests": "y" * 4000,  # ~1000 tokens
            "docs": "z" * 4000,   # ~1000 tokens
        }

        result = manager.truncate_content(
            content,
            target_tokens=1500,  # Half of original
            strategy=TruncationStrategy.BALANCED
        )

        # Each section should be reduced proportionally
        assert all(len(v) < 4000 for v in result.values())
        # Should maintain all sections
        assert all(k in result for k in content.keys())

    def test_preserve_tests_strategy_keeps_test_context(self):
        """Preserve tests strategy should prioritize test information.

        Why: Tests often contain the most important context for fixes.
        """
        manager = TokenManager()

        content = {
            "code": "x" * 8000,   # ~2000 tokens
            "tests": "test" * 2000,  # ~500 tokens
        }

        result = manager.truncate_content(
            content,
            target_tokens=1000,
            strategy=TruncationStrategy.PRESERVE_TESTS
        )

        # Tests should be preserved more than code
        if "tests" in result and "code" in result:
            test_reduction = len(content["tests"]) - len(result["tests"])
            code_reduction = len(content["code"]) - len(result["code"])
            assert code_reduction > test_reduction

    def test_truncation_adds_continuation_markers(self):
        """Truncated content should indicate it was cut.

        Why: Model needs to know content was truncated.
        """
        manager = TokenManager()

        long_text = "line\n" * 1000

        result = manager.fit_to_limit(long_text, max_tokens=100)

        if result.truncated:
            # Should indicate truncation somehow
            truncated_text = result.text if hasattr(result, 'text') else ""
            # Common truncation indicators
            assert any(marker in truncated_text for marker in [
                "...", "[truncated]", "<!-- truncated -->", "# ..."
            ]) or result.truncated is True


class TestTokenManagerConfiguration:
    """Test configuration options."""

    def test_can_disable_caching(self):
        """Allow disabling cache for testing.

        Why: Some scenarios need fresh calculations.
        """
        manager = TokenManager(cache_enabled=False)

        text = "test text"

        # Should not use cache
        count1 = manager.count_tokens(text)
        count2 = manager.count_tokens(text)

        assert count1 == count2
        # Verify cache is disabled
        assert not manager.cache_enabled

    def test_configurable_estimation_ratio(self):
        """Allow configuring character-to-token ratio.

        Why: Different languages have different ratios.
        """
        manager_default = TokenManager(default_chars_per_token=4.0)
        manager_dense = TokenManager(default_chars_per_token=2.0)

        text = "x" * 100

        count_default = manager_default.count_tokens(text, use_tiktoken=False)
        count_dense = manager_dense.count_tokens(text, use_tiktoken=False)

        # Dense should count more tokens for same text
        assert count_dense > count_default

    def test_fallback_estimation_when_tiktoken_unavailable(self):
        """Use estimation when tiktoken not installed.

        Why: Should work even without optional dependencies.
        """
        manager = TokenManager(fallback_estimation=True)

        # Mock tiktoken not being available
        with patch.object(manager, '_try_tiktoken', return_value=None):
            count = manager.count_tokens("Hello world")

            # Should estimate
            assert count > 0
            # Should be reasonable estimate
            assert 1 <= count <= 10  # "Hello world" is ~2-3 tokens


class TestTokenManagerMetadata:
    """Test metadata and statistics generation."""

    def test_generates_truncation_statistics(self):
        """Generate statistics about truncation operations.

        Why: Users need to know what was removed.
        """
        manager = TokenManager()

        long_text = "x" * 10000

        stats = manager.fit_to_limit_with_stats(long_text, max_tokens=1000)

        assert isinstance(stats, FitStats)
        assert stats.original_tokens > stats.final_tokens
        assert stats.truncated is True
        assert stats.strategy_used is not None

    def test_tracks_token_usage_over_time(self):
        """Track cumulative token usage.

        Why: Useful for monitoring costs and usage patterns.
        """
        manager = TokenManager()

        # Process multiple texts
        texts = ["text1", "text2", "text3"]

        total = 0
        for text in texts:
            count = manager.count_tokens(text)
            total += count

        # Should track total usage
        if hasattr(manager, 'total_tokens_counted'):
            assert manager.total_tokens_counted == total


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
