from unittest.mock import patch


def test_generate_command_basic(tmp_path):
    """Test generate command works with mock provider."""
    from click.testing import CliRunner

    from swebench_runner.cli import cli

    runner = CliRunner()

    with patch('swebench_runner.generation_integration.GenerationIntegration') as mock_integration, \
         patch('swebench_runner.provider_utils.get_provider_for_cli'), \
         patch('swebench_runner.datasets.DatasetManager') as mock_dm:

        # Setup minimal mocks
        mock_dm.return_value.get_instances.return_value = [
            {"instance_id": "test-1", "problem_statement": "Test", "repo": "test/repo"}
        ]

        # Mock the async run to return a path
        output_file = tmp_path / "output.jsonl"
        output_file.write_text('{"instance_id": "test-1", "patch": "diff"}\n')

        # Use a regular function that returns the path directly
        mock_integration.return_value.generate_patches_for_evaluation = lambda *args, **kwargs: output_file

        # Run command
        result = runner.invoke(cli, ['generate', '-i', 'test-1', '-p', 'mock'])

        assert result.exit_code == 0
        assert "Patch generated successfully" in result.output
