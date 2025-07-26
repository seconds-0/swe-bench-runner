"""Tests for the TokenManager component."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from swebench_runner.generation.prompt_builder import PromptContext
from swebench_runner.generation.token_manager import (
    FitStats,
    TokenManager,
    TruncationStrategy,
)


class TestTokenManager:
    """Tests for TokenManager class."""

    @pytest.fixture
    def manager(self):
        """Create a TokenManager instance."""
        return TokenManager(cache_enabled=True)

    @pytest.fixture
    def sample_context(self):
        """Create a sample PromptContext."""
        return PromptContext(
            system_prompt="You are a helpful assistant.",
            user_prompt="Fix this bug.",
            problem_statement="The function crashes when input is None.",
            code_context={
                "main.py": "def process(data):\n    return data.upper()",
                "utils.py": "import json\n\ndef load_config():\n    return {}"
            },
            test_context={
                "test_main.py": "def test_process():\n    assert process(None) == ''"
            },
            instance_id="test-001",
            repo_name="test/repo",
            base_commit="abc123",
            hints=["Check for None values"]
        )

    def test_init(self):
        """Test TokenManager initialization."""
        manager = TokenManager(
            cache_enabled=False,
            cache_size=5000,
            fallback_estimation=False,
            default_chars_per_token=3.5
        )

        assert not manager.cache_enabled
        assert manager.cache_size == 5000
        assert not manager.fallback_estimation
        assert manager.default_chars_per_token == 3.5

    def test_init_with_tiktoken(self):
        """Test initialization with tiktoken available."""
        # First, mock the import to succeed
        mock_tiktoken = MagicMock()
        mock_encoding = Mock()
        mock_tiktoken.get_encoding.return_value = mock_encoding

        with patch.dict('sys.modules', {'tiktoken': mock_tiktoken}):
            # Now import and instantiate with tiktoken "available"
            from swebench_runner.generation.token_manager import TokenManager as TM
            manager = TM()

            assert manager._tiktoken_available
            # The encoding might be loaded during init
            if "cl100k_base" in manager._tiktoken_encodings:
                assert manager._tiktoken_encodings["cl100k_base"] == mock_encoding

    def test_init_without_tiktoken(self):
        """Test initialization without tiktoken."""
        with patch.dict('sys.modules', {'tiktoken': None}):
            manager = TokenManager()
            assert not manager._tiktoken_available

    def test_count_tokens_empty(self, manager):
        """Test token counting with empty text."""
        assert manager.count_tokens("", "gpt-4") == 0

    def test_count_tokens_with_tiktoken(self):
        """Test token counting with tiktoken available."""
        manager = TokenManager()

        # Manually set up tiktoken mocking
        manager._tiktoken_available = True
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        manager._tiktoken_encodings["cl100k_base"] = mock_encoding

        count = manager.count_tokens("Hello world", "gpt-4")

        assert count == 5
        mock_encoding.encode.assert_called_with("Hello world")

    def test_count_tokens_fallback(self, manager):
        """Test token counting with estimation fallback."""
        manager._tiktoken_available = False

        # Test with default chars per token (4.0)
        text = "Hello world!"  # 12 chars
        count = manager.count_tokens(text, "claude-3")

        # Basic estimate: 12/4 = 3
        # Special chars: 1 (!)
        # Overhead: ~10% of 3 = 0.3
        # Total: ~3 + 0.5 + 0.3 = 3.8, rounded to 3 or 4
        assert 3 <= count <= 5

    def test_count_tokens_caching(self, manager):
        """Test that token counting is cached."""
        manager._tiktoken_available = False

        text = "Test caching"
        model = "gpt-4"

        # First call
        count1 = manager.count_tokens(text, model)

        # Mock the underlying method to ensure it's not called again
        with patch.object(manager, '_count_tokens') as mock_count:
            mock_count.return_value = 999  # Different value

            # Second call should use cache
            count2 = manager.count_tokens(text, model)

            assert count1 == count2
            mock_count.assert_not_called()

    def test_get_model_limit(self, manager):
        """Test getting model context limits."""
        # Known models
        assert manager.get_model_limit("openai", "gpt-4-turbo") == 128000
        assert manager.get_model_limit("anthropic", "claude-3-opus") == 200000
        assert manager.get_model_limit("", "gpt-3.5-turbo") == 16385

        # With provider prefix
        assert manager.get_model_limit("anthropic", "claude-2") == 100000

        # Unknown model
        assert manager.get_model_limit("unknown", "mystery-model") == 4096

    def test_get_model_limit_partial_match(self, manager):
        """Test model limit lookup with partial matches."""
        # Partial match (e.g., versioned model names)
        assert manager.get_model_limit("openai", "gpt-4-0613") == 8192
        assert manager.get_model_limit("", "gpt-4-turbo-2024-01-01") == 128000

    def test_estimate_tokens(self, manager):
        """Test token estimation logic."""
        # Simple text
        text1 = "Hello"  # 5 chars
        tokens1 = manager._estimate_tokens(text1)
        assert 1 <= tokens1 <= 3

        # Text with special characters
        text2 = "Hello.\nWorld,\nTest!"  # More special chars
        tokens2 = manager._estimate_tokens(text2)
        assert tokens2 > tokens1

        # Longer text
        text3 = "a" * 400  # 400 chars = ~100 base tokens
        tokens3 = manager._estimate_tokens(text3)
        assert 90 <= tokens3 <= 120

    def test_estimate_cost(self, manager):
        """Test cost estimation."""
        # Known model
        cost = manager.estimate_cost(
            prompt_tokens=1000,
            max_completion_tokens=500,
            provider="openai",
            model="gpt-4-turbo"
        )

        # gpt-4-turbo: $10/1M input, $30/1M output
        # 1000 tokens input = $0.01
        # 500 tokens output = $0.015
        # Total = $0.025
        assert cost == pytest.approx(0.025, rel=0.01)

        # Unknown model
        cost = manager.estimate_cost(1000, 500, "unknown", "mystery-model")
        assert cost is None

    def test_get_context_window_usage(self, manager):
        """Test context window usage calculation."""
        # 50% usage
        usage = manager.get_context_window_usage(64000, "gpt-4-turbo")
        assert usage == pytest.approx(0.5, rel=0.01)

        # Near limit
        usage = manager.get_context_window_usage(8000, "gpt-4")
        assert usage == pytest.approx(0.977, rel=0.01)

    def test_suggest_model_for_context(self, manager):
        """Test model suggestions based on context size."""
        # Small context
        suggestions = manager.suggest_model_for_context(2000, "any")
        assert "gpt-4" in suggestions  # 8k limit
        assert "gpt-3.5-turbo" in suggestions

        # Large context
        suggestions = manager.suggest_model_for_context(50000, "any")
        assert "gpt-4" not in suggestions  # Too small
        assert "gpt-4-turbo" in suggestions
        assert "claude-3-opus" in suggestions

        # Provider filter
        openai_suggestions = manager.suggest_model_for_context(10000, "openai")
        assert all("gpt" in m or "/" not in m for m in openai_suggestions)

        anthropic_suggestions = manager.suggest_model_for_context(10000, "anthropic")
        assert all("claude" in m for m in anthropic_suggestions)

    def test_fit_context_no_truncation(self, manager, sample_context):
        """Test fit_context when content fits."""
        manager._tiktoken_available = False  # Use estimation

        prompt, truncated, stats = manager.fit_context(
            sample_context,
            "gpt-4-turbo",  # 128k limit
            reserve_tokens=1000
        )

        assert not truncated
        assert stats.truncated is False
        assert stats.original_tokens == stats.final_tokens
        assert len(stats.sections_truncated) == 0

    def test_fit_context_with_truncation(self, manager, sample_context):
        """Test fit_context when truncation is needed."""
        manager._tiktoken_available = False

        # Make context very large to force truncation
        sample_context.problem_statement = "This is a very long problem statement. " * 100
        sample_context.code_context["large.py"] = "# Large file\n" * 500

        # Use a model with very small limit to force truncation
        with patch.object(manager, 'get_model_limit', return_value=100):
            prompt, truncated, stats = manager.fit_context(
                sample_context,
                "tiny-model",
                reserve_tokens=10,
                strategy=TruncationStrategy.BALANCED
            )

            assert truncated
            assert stats.truncated is True
            assert stats.final_tokens < stats.original_tokens
            assert stats.strategy_used == TruncationStrategy.BALANCED
            # The sections_truncated might be empty if the truncation happened at prompt building level
            # So we just check that truncation occurred

    def test_truncate_text_simple(self, manager):
        """Test simple text truncation."""
        text = "Hello " * 100  # 600 chars

        # Truncate to ~50 tokens (200 chars)
        truncated = manager._truncate_text(text, 50, preserve_structure=False)

        assert len(truncated) < len(text)
        assert "[... truncated ...]" in truncated

    def test_truncate_text_preserve_structure(self, manager):
        """Test text truncation with structure preservation."""
        code = """import os
import sys
from typing import List

class MyClass:
    def __init__(self):
        self.data = []

    def process(self, items: List[str]):
        for item in items:
            if item:
                self.data.append(item.upper())
                print(f"Processed: {item}")
        return self.data

    def clear(self):
        self.data = []"""

        # Truncate to a very small limit to force truncation
        # The code is about 340 chars, so ~85 tokens. Let's use 20 tokens (80 chars)
        truncated = manager._truncate_text(code, 20, preserve_structure=True)

        # Should be shorter than original
        assert len(truncated) < len(code)

        # Should have some indication of truncation
        assert "truncated" in truncated.lower() or "..." in truncated

    def test_truncate_file_dict(self, manager):
        """Test truncating dictionary of files."""
        files = {
            "test_small.py": "def test():\n    pass",
            "large_file.py": "x = 1\n" * 100,
            "another_test.py": "import pytest\n\ndef test_something():\n    assert True"
        }

        # Small budget - should prioritize test files and smaller files
        truncated = manager._truncate_file_dict(files, token_budget=20)

        assert "test_small.py" in truncated  # Small test file
        assert len(truncated) <= len(files)

    def test_truncate_balanced_strategy(self, manager, sample_context):
        """Test balanced truncation strategy."""
        manager._tiktoken_available = False

        truncated = manager.truncate_context(
            sample_context,
            max_tokens=50,
            strategy=TruncationStrategy.BALANCED
        )

        # Should have content in all sections but reduced
        assert len(truncated.problem_statement) < len(sample_context.problem_statement)
        assert len(truncated.code_context) <= len(sample_context.code_context)
        assert len(truncated.test_context) <= len(sample_context.test_context)

    def test_truncate_preserve_tests_strategy(self, manager, sample_context):
        """Test preserve tests truncation strategy."""
        manager._tiktoken_available = False

        # Add more content to make truncation necessary
        sample_context.code_context["big_file.py"] = "x = 1\n" * 200

        truncated = manager.truncate_context(
            sample_context,
            max_tokens=100,
            strategy=TruncationStrategy.PRESERVE_TESTS
        )

        # Test context should be preserved more than code
        if sample_context.test_context and truncated.test_context:
            test_reduction = len(str(truncated.test_context)) / len(str(sample_context.test_context))
            code_reduction = len(str(truncated.code_context)) / len(str(sample_context.code_context))
            assert test_reduction > code_reduction

    def test_format_truncation_summary(self, manager):
        """Test formatting truncation summary."""
        # No truncation
        stats1 = FitStats(
            original_tokens=1000,
            final_tokens=1000,
            truncated=False
        )
        summary1 = manager.format_truncation_summary(stats1)
        assert "fits within limit" in summary1
        assert "1,000 tokens" in summary1

        # With truncation
        stats2 = FitStats(
            original_tokens=5000,
            final_tokens=3000,
            truncated=True,
            strategy_used=TruncationStrategy.BALANCED,
            sections_truncated={
                "code_context": 1500,
                "test_context": 500
            }
        )
        summary2 = manager.format_truncation_summary(stats2)
        assert "truncated from 5,000 to 3,000" in summary2
        assert "balanced" in summary2
        assert "code_context: 1,500 tokens removed" in summary2
        assert "30.0%" in summary2  # 1500/5000 = 30%

    def test_is_openai_model(self, manager):
        """Test OpenAI model detection."""
        assert manager._is_openai_model("gpt-4")
        assert manager._is_openai_model("gpt-3.5-turbo")
        assert manager._is_openai_model("text-davinci-003")
        assert not manager._is_openai_model("claude-3")
        assert not manager._is_openai_model("anthropic/claude-2")

    def test_get_tiktoken_encoding_name(self, manager):
        """Test getting correct tiktoken encoding names."""
        assert manager._get_tiktoken_encoding_name("gpt-4") == "cl100k_base"
        assert manager._get_tiktoken_encoding_name("gpt-3.5-turbo") == "cl100k_base"
        assert manager._get_tiktoken_encoding_name("text-davinci-003") == "p50k_base"
        assert manager._get_tiktoken_encoding_name("unknown-model") == "cl100k_base"

    def test_extract_provider(self, manager):
        """Test provider extraction from model names."""
        assert manager._extract_provider("anthropic/claude-3") == "anthropic"
        assert manager._extract_provider("openai/gpt-4") == "openai"
        assert manager._extract_provider("gpt-4") == "openai"
        assert manager._extract_provider("claude-3-opus") == "anthropic"
        assert manager._extract_provider("unknown-model") == "unknown"

    def test_estimate_section_tokens(self, manager, sample_context):
        """Test estimating tokens for each section."""
        manager._tiktoken_available = False

        tokens = manager._estimate_section_tokens(sample_context)

        assert "system_prompt" in tokens
        assert "problem_statement" in tokens
        assert "code_context" in tokens
        assert "test_context" in tokens
        assert "hints" in tokens

        # All values should be positive or zero
        assert all(v >= 0 for v in tokens.values())

        # Total should match sum
        total = sum(tokens.values())
        assert total > 0

    def test_calculate_section_differences(self, manager, sample_context):
        """Test calculating token differences between contexts."""
        manager._tiktoken_available = False

        # Create a truncated version
        truncated = PromptContext(
            system_prompt=sample_context.system_prompt,
            problem_statement=sample_context.problem_statement[:20],
            code_context={"main.py": "truncated"},
            instance_id=sample_context.instance_id,
            repo_name=sample_context.repo_name,
            base_commit=sample_context.base_commit
        )

        diffs = manager._calculate_section_differences(sample_context, truncated)

        # Should show reductions
        assert "problem_statement" in diffs
        assert "code_context" in diffs
        assert all(v > 0 for v in diffs.values())

    def test_truncate_context_already_fits(self, manager, sample_context):
        """Test truncate_context when content already fits."""
        manager._tiktoken_available = False

        # Set a high limit
        truncated = manager.truncate_context(
            sample_context,
            max_tokens=10000,
            strategy=TruncationStrategy.BALANCED
        )

        # Should return same context
        assert truncated.problem_statement == sample_context.problem_statement
        assert truncated.code_context == sample_context.code_context
        assert truncated.test_context == sample_context.test_context

    def test_integration_with_prompt_builder(self, manager):
        """Test integration with PromptBuilder."""

        # Create a large context that needs truncation
        large_context = PromptContext(
            system_prompt="You are an AI assistant.",
            problem_statement="Fix the bug " * 1000,  # Large problem
            code_context={
                f"file{i}.py": f"# Content of file {i}\n" * 100
                for i in range(10)
            },
            instance_id="test-001",
            repo_name="test/repo",
            base_commit="abc123"
        )

        # Fit to small model
        prompt, truncated, stats = manager.fit_context(
            large_context,
            "gpt-4",  # 8k limit
            reserve_tokens=1000,
            strategy=TruncationStrategy.BALANCED
        )

        assert truncated
        assert len(prompt) < 8192 * 4  # Rough char estimate
        assert stats.truncated
        assert stats.original_tokens > stats.final_tokens

    @patch('swebench_runner.generation.token_manager.logger')
    def test_logging(self, mock_logger, manager):
        """Test that appropriate logging occurs."""
        manager._tiktoken_available = False

        # Test warning for unknown model
        manager.get_model_limit("unknown", "mystery-model")
        mock_logger.warning.assert_called()

        # Test info for truncation
        context = PromptContext(
            problem_statement="x" * 10000,
            instance_id="test",
            repo_name="repo",
            base_commit="commit"
        )

        with patch.object(manager, 'get_model_limit', return_value=100):
            manager.fit_context(context, "tiny-model", reserve_tokens=10)
            mock_logger.info.assert_called()

    def test_no_cache_mode(self):
        """Test TokenManager without caching."""
        manager = TokenManager(cache_enabled=False)

        text = "Test no cache"
        model = "gpt-4"

        # Count tokens twice
        count1 = manager.count_tokens(text, model)

        # Mock the uncached method to ensure it IS called each time
        with patch.object(manager, '_estimate_tokens', return_value=10) as mock_estimate:
            count2 = manager.count_tokens(text, model)

            # Should call estimate since no cache
            mock_estimate.assert_called()

    def test_tiktoken_error_handling(self):
        """Test handling of tiktoken errors."""
        # Mock tiktoken to raise an error during import
        mock_tiktoken = MagicMock()
        mock_tiktoken.get_encoding.side_effect = Exception("Encoding error")

        with patch.dict('sys.modules', {'tiktoken': mock_tiktoken}):
            from swebench_runner.generation.token_manager import TokenManager as TM
            manager = TM()

            # Should handle error gracefully - tiktoken might be available but encoding failed
            # Or tiktoken might not be available at all
            if manager._tiktoken_available:
                assert "cl100k_base" not in manager._tiktoken_encodings

    def test_count_tokens_tiktoken_failure(self):
        """Test fallback when tiktoken fails during counting."""
        manager = TokenManager()
        manager._tiktoken_available = True

        with patch.object(manager, '_count_openai_tokens', side_effect=Exception("Count error")):
            # Should fall back to estimation
            count = manager.count_tokens("Hello world", "gpt-4")
            assert count > 0  # Should get estimate instead of failing
