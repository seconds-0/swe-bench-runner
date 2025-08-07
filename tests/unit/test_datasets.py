"""Unit tests for datasets.py - focus on error messages.

Tests specifically for:
- Helpful error message generation
- Authentication error guidance
- Network error suggestions
"""

import pytest

from swebench_runner.datasets import (
    get_helpful_error_message
)
from swebench_runner.exceptions import (
    DatasetAuthenticationError,
    DatasetNetworkError,
    DatasetNotFoundError,
    RegexValidationError
)


class TestDatasetErrorMessages:
    """Test helpful error message generation."""

    def test_authentication_error_includes_token_setup(self):
        """Authentication errors should explain token setup.
        
        Why: Users need clear instructions to fix auth issues quickly.
        """
        error = DatasetAuthenticationError("Auth failed")
        message = get_helpful_error_message(error, {"dataset": "full"})
        
        # Should include HuggingFace token URL
        assert "https://huggingface.co/settings/tokens" in message
        # Should include export command
        assert "export HF_TOKEN=" in message
        # Should mention it's free
        assert "free" in message.lower()

    def test_network_error_suggests_offline_mode(self):
        """Network errors should suggest offline mode.
        
        Why: Users may have cached data and not need network.
        """
        error = DatasetNetworkError("Connection failed")
        message = get_helpful_error_message(error, {"dataset": "lite", "offline": False})
        
        # Should suggest offline flag
        assert "--offline" in message
        # Should suggest checking cache
        assert "cache" in message.lower()

    def test_dataset_not_found_lists_available(self):
        """Dataset not found should list available datasets.
        
        Why: Users may not know valid dataset names.
        """
        error = DatasetNotFoundError("Dataset 'invalid' not found")
        message = get_helpful_error_message(error)
        
        # Should list available datasets
        assert "lite" in message
        assert "verified" in message
        assert "full" in message
        # Should include sizes
        assert "300" in message or "1.2MB" in message

    def test_regex_error_suggests_simpler_patterns(self):
        """Regex errors should suggest alternatives.
        
        Why: Complex regex can cause ReDoS vulnerabilities.
        """
        error = RegexValidationError("Pattern too complex")
        message = get_helpful_error_message(error)
        
        # Should suggest glob patterns
        assert "django__*" in message or "glob" in message.lower()
        # Should warn about nested quantifiers
        assert "(a+)+" in message or "nested" in message.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])