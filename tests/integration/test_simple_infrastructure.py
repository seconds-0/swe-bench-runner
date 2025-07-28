"""Simple integration test infrastructure validation.

This test file has minimal dependencies and validates basic functionality.
"""

import os

import pytest


@pytest.mark.integration
def test_integration_marker():
    """Test that integration marker works."""
    # This test should only run with -m integration
    assert True


@pytest.mark.integration
def test_skip_mechanism():
    """Test that skip mechanism works."""
    if not os.environ.get("DUMMY_API_KEY"):
        pytest.skip("DUMMY_API_KEY not set - skip working correctly")
    assert False  # Should not reach here


@pytest.mark.integration
def test_basic_math():
    """Test basic functionality in integration context."""
    assert 2 + 2 == 4
    assert len("test") == 4
