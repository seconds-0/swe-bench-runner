"""Integration tests for CLI dataset functionality."""

from __future__ import annotations

from unittest.mock import Mock, patch

from click.testing import CliRunner

from swebench_runner.cli import cli


class TestCLIDatasets:
    """Test suite for CLI dataset functionality."""

    def test_info_command_help(self) -> None:
        """Test that info command shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['info', '--help'])
        assert result.exit_code == 0
        assert 'Get information about a SWE-bench dataset' in result.output
        assert '-d, --dataset [lite|verified|full]' in result.output

    def test_info_command_missing_dataset(self) -> None:
        """Test that info command requires dataset argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ['info'])
        assert result.exit_code != 0
        assert 'Missing option' in result.output

    @patch('datasets.load_dataset_builder')
    def test_info_command_success(self, mock_load_builder: Mock) -> None:
        """Test successful info command execution."""
        runner = CliRunner()

        # Mock dataset builder and info
        mock_builder = Mock()
        mock_info = Mock()
        mock_info.splits = {'test': Mock(num_examples=300)}
        mock_info.download_size = 1200000  # 1.2MB
        mock_info.dataset_size = 3600000   # 3.6MB
        mock_info.description = "Test dataset description"
        mock_builder.info = mock_info
        mock_load_builder.return_value = mock_builder

        result = runner.invoke(cli, ['info', '-d', 'lite'])

        assert result.exit_code == 0
        assert 'SWE-bench lite dataset:' in result.output
        assert 'Total instances: 300' in result.output
        assert 'Download size: 1.' in result.output  # Tolerant of rounding
        assert 'On-disk size: 3.' in result.output

    def test_run_command_shows_dataset_options(self) -> None:
        """Test that run command shows dataset options in help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['run', '--help'])
        assert result.exit_code == 0
        assert '-d, --dataset [lite|verified|full]' in result.output
        assert '--instances TEXT' in result.output
        assert '--count INTEGER' in result.output
        assert '--sample TEXT' in result.output
        assert '--subset TEXT' in result.output
        assert '--regex' in result.output
        assert '--rerun-failed PATH' in result.output

    def test_run_command_dataset_conflicts(self) -> None:
        """Test that run command rejects conflicting source options."""
        runner = CliRunner()

        # Test dataset + patches conflict
        with runner.isolated_filesystem():
            # Create a dummy patches file
            with open('patches.jsonl', 'w') as f:
                f.write('{"instance_id": "test", "patch": "test"}\n')

            result = runner.invoke(
                cli, ['run', '--patches', 'patches.jsonl', '-d', 'lite']
            )
            assert result.exit_code != 0
            assert 'Cannot provide multiple sources' in result.output

    @patch('datasets.load_dataset')
    def test_run_command_dataset_no_matches(self, mock_load_dataset: Mock) -> None:
        """Test run command when no instances match criteria."""
        runner = CliRunner()

        # Mock empty dataset
        mock_dataset = Mock()
        mock_dataset.filter = Mock(return_value=mock_dataset)
        mock_dataset.__iter__ = Mock(return_value=iter([]))
        mock_dataset.__len__ = Mock(return_value=0)
        mock_load_dataset.return_value = mock_dataset

        result = runner.invoke(cli, ['run', '-d', 'lite', '--instances', 'nonexistent'])

        assert result.exit_code != 0
        assert 'No instances matched your criteria' in result.output

    @patch('swebench_runner.cli.run_evaluation')
    @patch('datasets.load_dataset')
    def test_run_command_dataset_success_basic(
        self, mock_load_dataset: Mock, mock_run_eval: Mock
    ) -> None:
        """Test successful dataset loading and basic run."""
        runner = CliRunner()

        # Mock dataset with one instance
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([
            {
                'instance_id': 'test__test-123',
                'patch': 'diff --git a/test.py b/test.py\n...',
                'repo': 'test/test',
                'base_commit': 'abc123',
                'problem_statement': 'Test issue'
            }
        ]))
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset.filter = Mock(return_value=mock_dataset)
        mock_dataset.select = Mock(return_value=mock_dataset)
        mock_load_dataset.return_value = mock_dataset

        # Mock evaluation result
        mock_result = Mock()
        mock_result.instance_id = 'test__test-123'
        mock_result.passed = True
        mock_result.error = None
        mock_run_eval.return_value = mock_result

        result = runner.invoke(cli, ['run', '-d', 'lite', '--count', '1', '--no-input'])

        # Should load dataset and attempt to run evaluation
        assert 'Loading lite dataset from HuggingFace' in result.output
        assert 'Loaded 1 instances from lite dataset' in result.output
        # Mocked evaluation should succeed
        assert result.exit_code == 0

    @patch('datasets.load_dataset')
    def test_run_command_dataset_with_filters(self, mock_load_dataset: Mock) -> None:
        """Test run command with various filtering options."""
        runner = CliRunner()

        # Mock dataset - need separate mock for each filter result
        def create_mock_dataset(items):
            mock_dataset = Mock()
            mock_dataset.__iter__ = Mock(return_value=iter(items))
            mock_dataset.__len__ = Mock(return_value=len(items))
            mock_dataset.filter = Mock(return_value=create_mock_dataset(items))
            mock_dataset.select = Mock(return_value=create_mock_dataset(items))
            return mock_dataset

        mock_load_dataset.return_value = create_mock_dataset([
            {'instance_id': 'django__django-123', 'patch': 'test patch'}
        ])

        # Test with specific instances
        result = runner.invoke(cli, [
            'run', '-d', 'lite',
            '--instances', 'django__django-123,django__django-456',
            '--no-input'
        ])

        assert (
            'Specific instances: django__django-123, django__django-456' 
            in result.output
        )

        # Test with pattern filtering
        result = runner.invoke(cli, [
            'run', '-d', 'lite',
            '--subset', 'django__*',
            '--no-input'
        ])

        assert 'Filtered by pattern: django__*' in result.output

        # Test with regex filtering
        result = runner.invoke(cli, [
            'run', '-d', 'lite',
            '--subset', 'django__django-[0-9]+',
            '--regex',
            '--no-input'
        ])

        assert 'Filtered by regex: django__django-[0-9]+' in result.output

        # Test with percentage sampling
        result = runner.invoke(cli, [
            'run', '-d', 'lite',
            '--sample', '10%',
            '--no-input'
        ])

        assert 'Random 10.0% sample' in result.output

    def test_run_command_requires_source(self) -> None:
        """Test that run command requires some source of patches."""
        runner = CliRunner()

        # Should fail if no patches, dataset, or auto-detection
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['run', '--no-input'])
            assert result.exit_code != 0
            assert (
                'Must provide --patches, --patches-dir, or --dataset' in result.output
            )
