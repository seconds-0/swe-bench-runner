"""Tests for exit codes module."""

from swebench_runner import exit_codes


class TestExitCodes:
    """Test exit code constants and functions."""
    
    def test_exit_code_constants(self):
        """Test that exit code constants have expected values."""
        assert exit_codes.SUCCESS == 0
        assert exit_codes.GENERAL_ERROR == 1
        assert exit_codes.DOCKER_NOT_FOUND == 2
        assert exit_codes.NETWORK_ERROR == 3
        assert exit_codes.RESOURCE_ERROR == 4
    
    def test_get_exit_code_name(self):
        """Test getting exit code names."""
        assert exit_codes.get_exit_code_name(0) == "SUCCESS"
        assert exit_codes.get_exit_code_name(1) == "GENERAL_ERROR"
        assert exit_codes.get_exit_code_name(2) == "DOCKER_NOT_FOUND"
        assert exit_codes.get_exit_code_name(3) == "NETWORK_ERROR"
        assert exit_codes.get_exit_code_name(4) == "RESOURCE_ERROR"
        
        # Test unknown code
        assert exit_codes.get_exit_code_name(99) == "UNKNOWN (99)"
    
    def test_get_exit_code_description(self):
        """Test getting exit code descriptions."""
        assert "successfully" in exit_codes.get_exit_code_description(0)
        assert "General error" in exit_codes.get_exit_code_description(1)
        assert "Docker" in exit_codes.get_exit_code_description(2)
        assert "Network" in exit_codes.get_exit_code_description(3)
        assert "resources" in exit_codes.get_exit_code_description(4)
        
        # Test unknown code
        assert exit_codes.get_exit_code_description(99) == "Unknown exit code: 99"
    
    def test_exit_code_names_dict(self):
        """Test that EXIT_CODE_NAMES dict is complete."""
        assert len(exit_codes.EXIT_CODE_NAMES) == 5
        assert all(isinstance(k, int) for k in exit_codes.EXIT_CODE_NAMES.keys())
        assert all(isinstance(v, str) for v in exit_codes.EXIT_CODE_NAMES.values())
    
    def test_exit_code_descriptions_dict(self):
        """Test that EXIT_CODE_DESCRIPTIONS dict is complete."""
        assert len(exit_codes.EXIT_CODE_DESCRIPTIONS) == 5
        assert all(isinstance(k, int) for k in exit_codes.EXIT_CODE_DESCRIPTIONS.keys())
        assert all(isinstance(v, str) for v in exit_codes.EXIT_CODE_DESCRIPTIONS.values())