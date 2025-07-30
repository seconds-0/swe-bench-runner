from unittest.mock import AsyncMock, MagicMock, patch


def test_generate_command_basic(tmp_path):
    """Test generate command works with mock provider."""
    from click.testing import CliRunner

    from swebench_runner.cli import cli

    runner = CliRunner()

    # Create output file
    output_file = tmp_path / "output.jsonl"
    output_file.write_text('{"instance_id": "test-1", "patch": "diff"}\n')

    # We need to mock the imports at their original locations
    with patch('swebench_runner.generation_integration.GenerationIntegration') as mock_integration_class, \
         patch('swebench_runner.provider_utils.ensure_provider_configured'), \
         patch('swebench_runner.provider_utils.get_provider_for_cli') as mock_get_provider, \
         patch('swebench_runner.datasets.DatasetManager') as mock_dm_class:

        # Setup dataset mock
        mock_dm = MagicMock()
        mock_dm.get_instances.return_value = [
            {"instance_id": "test-1", "problem_statement": "Test", "repo": "test/repo"}
        ]
        mock_dm_class.return_value = mock_dm

        # Setup provider mock
        mock_provider = MagicMock()
        mock_provider.name = "mock"
        mock_provider._async_provider = MagicMock()
        mock_provider._async_provider.name = "mock"
        mock_provider._async_provider.config = MagicMock(model="mock-model")
        mock_get_provider.return_value = mock_provider

        # Setup integration mock
        mock_integration = MagicMock()
        # Use AsyncMock for the async method
        mock_integration.generate_patches_for_evaluation = AsyncMock(return_value=output_file)
        mock_integration_class.return_value = mock_integration

        # Run command
        result = runner.invoke(cli, ['generate', '-i', 'test-1', '-p', 'mock', '-d', 'lite'])

        assert result.exit_code == 0
        assert "Generating patch for test-1" in result.output
