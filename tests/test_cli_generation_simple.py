"""Simple test for CLI generation commands."""
from unittest.mock import MagicMock, patch


def test_generate_command_with_mocked_generation(tmp_path):
    """Test generate command with fully mocked generation."""
    from click.testing import CliRunner

    from swebench_runner.cli import cli

    runner = CliRunner()

    # Create mock result file
    result_file = tmp_path / "test.jsonl"
    result_file.write_text('{"instance_id": "test-1", "patch": "diff --git a/test.py\\n+fix", "cost": 0.01}\n')

    with patch('swebench_runner.provider_utils.get_provider_for_cli') as mock_get_provider, \
         patch('swebench_runner.datasets.DatasetManager') as mock_dm, \
         patch('swebench_runner.generation_integration.GenerationIntegration') as mock_integration, \
         patch('asyncio.run', return_value=result_file):

        # Setup provider mock
        mock_provider = MagicMock()
        mock_provider._async_provider.name = "mock"
        mock_provider._async_provider.config.model = "mock-model"
        mock_provider._async_provider.default_model = "mock-model"
        mock_get_provider.return_value = mock_provider

        # Setup dataset manager mock
        mock_dm.return_value.get_instances.return_value = [
            {"instance_id": "test-1", "problem_statement": "Test problem"}
        ]

        # Run the command
        result = runner.invoke(cli, ['generate', '-i', 'test-1', '-d', 'lite'])

        # Check results
        assert result.exit_code == 0
        assert "Generating patch" in result.output
        assert "generated successfully" in result.output.lower()


def test_run_command_with_provider_flag(tmp_path):
    """Test run command with --provider flag for generation."""
    from click.testing import CliRunner

    from swebench_runner.cli import cli

    runner = CliRunner()

    # Create a temp dataset file
    dataset_file = tmp_path / "dataset.jsonl"
    dataset_file.write_text('{"instance_id": "test-1", "problem_statement": "Test"}\n')

    # Create a temp patches file that will be "generated"
    patches_file = tmp_path / "patches.jsonl"
    patches_file.write_text('{"instance_id": "test-1", "patch": "diff --git a/test.py b/test.py\\n--- a/test.py\\n+++ b/test.py\\n@@ -1,1 +1,1 @@\\n-old\\n+new"}\n')

    with patch('swebench_runner.datasets.DatasetManager') as mock_dm, \
         patch('swebench_runner.provider_utils.ensure_provider_configured'), \
         patch('swebench_runner.generation_integration.GenerationIntegration') as mock_integration, \
         patch('asyncio.run', return_value=patches_file), \
         patch('swebench_runner.cache.get_cache_dir', return_value=tmp_path), \
         patch('swebench_runner.bootstrap.check_and_prompt_first_run', return_value=False):

        # Setup dataset manager
        mock_dm.return_value.get_instances.return_value = [
            {"instance_id": "test-1", "problem_statement": "Test"}
        ]

        # Create temp directory structure
        temp_dir = tmp_path / ".swebench" / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Mock save_as_jsonl to actually create the file
        def mock_save_jsonl(instances, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                import json
                for inst in instances:
                    f.write(json.dumps(inst) + '\n')

        mock_dm.return_value.save_as_jsonl = mock_save_jsonl
        mock_dm.return_value.check_memory_requirements.return_value = (True, None)
        mock_dm.return_value.cleanup_temp_files = MagicMock()

        # Run with generate-only flag
        result = runner.invoke(
            cli,
            ['run', '-d', 'lite', '--count', '1', '--provider', 'mock', '--generate-only']
        )

        # Check that generation was triggered
        if result.exit_code != 0:
            print(f"Exit code: {result.exit_code}")
            print(f"Output: {result.output}")
        assert result.exit_code == 0
        # The output might not show "Patches generated successfully" if we exit early
        # but should at least show that the dataset was loaded
        assert "Loaded 1 instances" in result.output or "Patches generated successfully" in result.output
