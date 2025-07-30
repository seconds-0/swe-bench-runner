"""Test the integration test infrastructure itself.

This test file validates that the integration test framework is properly
set up, even without provider packages installed. It ensures that:
- Integration marker works correctly
- Skip mechanisms function properly
- Test discovery operates as expected
"""

import os

import pytest

# Disable the autouse fixture from main conftest for integration tests
pytestmark = [pytest.mark.integration, pytest.mark.no_autouse]


@pytest.mark.integration
class TestInfrastructure:
    """Test the integration test infrastructure."""

    def test_marker_applied(self):
        """Test that integration marker is properly applied."""
        # This test should only run when -m integration is specified
        assert True, "Integration marker is working"

    def test_skip_without_env_var(self):
        """Test that we can skip based on missing environment variables."""
        if not os.environ.get("INTEGRATION_TEST_DUMMY_KEY"):
            pytest.skip("INTEGRATION_TEST_DUMMY_KEY not set - skip mechanism working")

        # This line should not be reached in normal testing
        raise AssertionError("This should have been skipped")

    def test_basic_assertion(self):
        """Test that basic assertions work in integration context."""
        assert 1 + 1 == 2
        assert "integration" in "integration test"

    @pytest.mark.skipif(
        not os.environ.get("RUN_EXPENSIVE_TESTS"),
        reason="Expensive tests disabled by default"
    )
    def test_conditional_skip(self):
        """Test conditional skip based on environment."""
        # This simulates expensive tests that should be skipped by default
        assert True, "Expensive test ran when explicitly enabled"
