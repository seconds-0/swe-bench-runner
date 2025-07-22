"""Critical tests for CLI functionality that actually matters."""

import json
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from swebench_runner import cli, exit_codes
from swebench_runner.models import EvaluationResult


class TestCLIJSONOutput:
    """Test JSON output format - this is an API contract."""
    
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()
    
    def test_json_output_success(self, cli_runner, tmp_path):
        """Test JSON output for successful evaluation."""
        patches_file = tmp_path / "test.jsonl" 
        patches_file.write_text('{"instance_id": "test-123", "patch": "fix"}')
        
        # Mock all the checks that happen before evaluation
        with patch("swebench_runner.cli.check_and_prompt_first_run") as mock_bootstrap:
            mock_bootstrap.return_value = False  # Not first run
            
            # Mock at the CLI level to avoid Docker checks entirely
            with patch("swebench_runner.cli.run_evaluation") as mock_run:
                mock_run.return_value = EvaluationResult(
                    instance_id="test-123",
                    passed=True,
                    error=None
                )
                
                result = cli_runner.invoke(cli.run, [
                    "--patches", str(patches_file),
                    "--json"
                ])
                
                output = json.loads(result.output)
                assert output["instance_id"] == "test-123"
                assert output["passed"] is True
                assert output["error"] is None
                assert result.exit_code == exit_codes.SUCCESS
    
    def test_json_output_failure(self, cli_runner, tmp_path):
        """Test JSON output for failed evaluation."""
        patches_file = tmp_path / "test.jsonl"
        patches_file.write_text('{"instance_id": "test-456", "patch": "fix"}')
        
        with patch("swebench_runner.cli.check_and_prompt_first_run") as mock_bootstrap:
            mock_bootstrap.return_value = False  # Not first run
            
            with patch("swebench_runner.cli.run_evaluation") as mock_run:
                mock_run.return_value = EvaluationResult(
                    instance_id="test-456", 
                    passed=False,
                    error="Tests failed"
                )
                
                result = cli_runner.invoke(cli.run, [
                    "--patches", str(patches_file),
                    "--json"
                ])
                
                output = json.loads(result.output)
                assert output["instance_id"] == "test-456"
                assert output["passed"] is False
                assert output["error"] == "Tests failed"
    
    def test_json_output_exception(self, cli_runner, tmp_path):
        """Test JSON output when exception occurs."""
        patches_file = tmp_path / "test.jsonl"
        patches_file.write_text('{"instance_id": "test", "patch": "fix"}')
        
        with patch("swebench_runner.cli.check_and_prompt_first_run") as mock_bootstrap:
            mock_bootstrap.return_value = False  # Not first run
            
            with patch("swebench_runner.cli.run_evaluation") as mock_run:
                mock_run.side_effect = Exception("Unexpected error")
                
                result = cli_runner.invoke(cli.run, [
                    "--patches", str(patches_file),
                    "--json"
                ])
                
                output = json.loads(result.output)
                assert output["error"] == "Unexpected error"
                assert output["passed"] is False
                assert result.exit_code == exit_codes.GENERAL_ERROR