"""Tests for dataset management functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from swebench_runner.datasets import DatasetManager, get_helpful_error_message
from swebench_runner.exceptions import (
    DatasetAuthenticationError,
    DatasetNetworkError,
    DatasetNotFoundError,
    DatasetValidationError,
    InstanceValidationError,
    RegexValidationError,
)


class TestDatasetManager:
    """Test suite for DatasetManager class."""

    def test_init(self) -> None:
        """Test DatasetManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))
            assert manager.cache_dir.exists()
            assert manager.cache_dir.name == "datasets"

    def test_unknown_dataset_raises_error(self) -> None:
        """Test that unknown dataset raises ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))
            with pytest.raises(DatasetNotFoundError, match="Unknown dataset: invalid"):
                manager.fetch_dataset("invalid")

    @patch('datasets.load_dataset')
    def test_fetch_dataset_success(self, mock_load_dataset: Mock) -> None:
        """Test successful dataset fetching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))
            mock_dataset = Mock()
            mock_load_dataset.return_value = mock_dataset

            result = manager.fetch_dataset("lite")

            assert result == mock_dataset
            mock_load_dataset.assert_called_once_with(
                "princeton-nlp/SWE-bench_Lite",
                split="test",
                download_mode="reuse_dataset_if_exists"
            )

    @patch('datasets.load_dataset')
    def test_get_instances_basic(self, mock_load_dataset: Mock) -> None:
        """Test basic instance retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            # Mock dataset with sample data
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

            instances = manager.get_instances("lite")

            assert len(instances) == 1
            assert instances[0]['instance_id'] == 'test__test-123'
            assert instances[0]['patch'] == 'diff --git a/test.py b/test.py\n...'

    @patch('datasets.load_dataset')
    def test_get_instances_with_specific_ids(self, mock_load_dataset: Mock) -> None:
        """Test filtering by specific instance IDs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            mock_dataset = Mock()
            mock_dataset.filter = Mock(return_value=mock_dataset)
            mock_dataset.__iter__ = Mock(return_value=iter([
                {'instance_id': 'test__test-123', 'patch': 'test patch'}
            ]))
            mock_dataset.__len__ = Mock(return_value=1)
            mock_load_dataset.return_value = mock_dataset

            manager.get_instances("lite", instances=['test__test-123'])

            # Verify filter was called with correct lambda function
            mock_dataset.filter.assert_called_once()
            filter_func = mock_dataset.filter.call_args[0][0]

            # Test the filter function
            assert filter_func({'instance_id': 'test__test-123'}) is True
            assert filter_func({'instance_id': 'other__other-456'}) is False

    @patch('datasets.load_dataset')
    def test_get_instances_with_glob_pattern(self, mock_load_dataset: Mock) -> None:
        """Test filtering with glob pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            mock_dataset = Mock()
            mock_dataset.filter = Mock(return_value=mock_dataset)
            mock_dataset.__iter__ = Mock(return_value=iter([]))
            mock_dataset.__len__ = Mock(return_value=0)
            mock_load_dataset.return_value = mock_dataset

            manager.get_instances("lite", subset_pattern="django__*", use_regex=False)

            # Verify filter was called
            mock_dataset.filter.assert_called_once()
            filter_func = mock_dataset.filter.call_args[0][0]

            # Test the filter function uses fnmatch
            assert filter_func({'instance_id': 'django__django-123'}) is True
            assert filter_func({'instance_id': 'flask__flask-456'}) is False

    @patch('datasets.load_dataset')
    def test_get_instances_with_regex_pattern(self, mock_load_dataset: Mock) -> None:
        """Test filtering with regex pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            mock_dataset = Mock()
            mock_dataset.filter = Mock(return_value=mock_dataset)
            mock_dataset.__iter__ = Mock(return_value=iter([]))
            mock_dataset.__len__ = Mock(return_value=0)
            mock_load_dataset.return_value = mock_dataset

            manager.get_instances(
                "lite", subset_pattern=r"django__django-\d+", use_regex=True
            )

            # Verify filter was called
            mock_dataset.filter.assert_called_once()
            filter_func = mock_dataset.filter.call_args[0][0]

            # Test the filter function uses regex
            assert filter_func({'instance_id': 'django__django-123'}) is True
            assert filter_func({'instance_id': 'django__django-abc'}) is False

    @patch('datasets.load_dataset')
    def test_get_instances_with_count_sampling(self, mock_load_dataset: Mock) -> None:
        """Test count-based sampling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=100)
            mock_dataset.select = Mock(return_value=mock_dataset)
            mock_dataset.__iter__ = Mock(return_value=iter([]))
            mock_load_dataset.return_value = mock_dataset

            manager.get_instances("lite", count=10)

            # Verify select was called with appropriate indices
            mock_dataset.select.assert_called_once()
            indices = mock_dataset.select.call_args[0][0]
            assert len(indices) == 10

    @patch('datasets.load_dataset')
    def test_get_instances_with_percentage_sampling(
        self, mock_load_dataset: Mock
    ) -> None:
        """Test percentage-based sampling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=100)
            mock_dataset.select = Mock(return_value=mock_dataset)
            mock_dataset.__iter__ = Mock(return_value=iter([]))
            mock_load_dataset.return_value = mock_dataset

            manager.get_instances("lite", sample_percent=10.0)

            # Verify select was called with 10% of instances (10 out of 100)
            mock_dataset.select.assert_called_once()
            indices = mock_dataset.select.call_args[0][0]
            assert len(indices) == 10

    def test_save_as_jsonl(self) -> None:
        """Test saving instances to JSONL format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))
            output_path = Path(temp_dir) / "test.jsonl"

            instances = [
                {'instance_id': 'test1', 'patch': 'patch1'},
                {'instance_id': 'test2', 'patch': 'patch2'}
            ]

            manager.save_as_jsonl(instances, output_path)

            assert output_path.exists()
            content = output_path.read_text()
            lines = content.strip().split('\n')
            assert len(lines) == 2
            assert '"instance_id": "test1"' in lines[0]
            assert '"patch": "patch1"' in lines[0]

    def test_cleanup_temp_files(self) -> None:
        """Test temporary file cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))
            temp_dir_path = manager.cache_dir.parent / "temp"

            # Create a temp file
            temp_dir_path.mkdir(exist_ok=True)
            test_file = temp_dir_path / "test.jsonl"
            test_file.write_text("test content")

            assert test_file.exists()

            manager.cleanup_temp_files()

            assert temp_dir_path.exists()  # Directory should still exist
            assert not test_file.exists()  # But file should be gone

    @patch('datasets.load_dataset_builder')
    def test_get_dataset_info(self, mock_load_builder: Mock) -> None:
        """Test getting dataset information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            # Mock dataset builder and info
            mock_builder = Mock()
            mock_info = Mock()
            mock_info.splits = {'test': Mock(num_examples=300)}
            mock_info.download_size = 1200000  # 1.2MB
            mock_info.dataset_size = 3600000   # 3.6MB
            mock_info.description = "Test dataset description"
            mock_builder.info = mock_info
            mock_load_builder.return_value = mock_builder

            info = manager.get_dataset_info("lite")

            assert info['name'] == 'lite'
            assert info['total_instances'] == 300
            assert info['download_size_mb'] == pytest.approx(1.2, rel=0.1)
            assert info['dataset_size_mb'] == pytest.approx(3.6, rel=0.1)
            assert info['description'] == "Test dataset description"


class TestEnhancedErrorMessages:
    """Test suite for enhanced error message functionality."""

    def test_authentication_error_message(self) -> None:
        """Test enhanced message for authentication errors."""
        error = DatasetAuthenticationError("Authentication required for dataset lite")
        context = {'dataset': 'lite'}

        message = get_helpful_error_message(error, context)

        assert "❌ Authentication required for lite dataset" in message
        assert "🔧 How to fix:" in message
        assert "huggingface.co/settings/tokens" in message
        assert "export HF_TOKEN=" in message

    def test_network_error_message_online(self) -> None:
        """Test enhanced message for network errors in online mode."""
        error = DatasetNetworkError("Network error accessing dataset verified")
        context = {'dataset': 'verified', 'offline': False}

        message = get_helpful_error_message(error, context)

        assert "❌ Network error downloading dataset" in message
        assert "Check your internet connection" in message
        assert "--offline flag" in message

    def test_network_error_message_offline(self) -> None:
        """Test enhanced message for network errors in offline mode."""
        error = DatasetNetworkError("Dataset not cached")
        context = {'dataset': 'full', 'offline': True}

        message = get_helpful_error_message(error, context)

        assert "❌ Dataset full not available in offline mode" in message
        assert "Remove --offline flag" in message
        assert "Download once, then use --offline" in message

    def test_dataset_not_found_error(self) -> None:
        """Test enhanced message for dataset not found errors."""
        error = DatasetNotFoundError("Unknown dataset: invalid")

        message = get_helpful_error_message(error)

        assert "❌ Unknown dataset: invalid" in message
        assert "🔧 Available datasets:" in message
        assert "lite: 300 instances" in message
        assert "verified: 500 instances" in message
        assert "full: 2294 instances" in message

    def test_regex_validation_error(self) -> None:
        """Test enhanced message for regex validation errors."""
        error = RegexValidationError("Invalid regex pattern")

        message = get_helpful_error_message(error)

        assert "❌ Invalid regex pattern" in message
        assert "🔧 How to fix:" in message
        assert "django__*" in message
        assert "regex101.com" in message

    def test_instance_validation_error(self) -> None:
        """Test enhanced message for instance validation errors."""
        error = InstanceValidationError("Invalid instance ID format")

        message = get_helpful_error_message(error)

        assert "❌ Invalid instance ID format" in message
        assert "🔧 Valid instance ID format:" in message
        assert "django__django-123" in message
        assert "swebench info -d lite" in message

    def test_dataset_validation_error(self) -> None:
        """Test enhanced message for dataset validation errors."""
        error = DatasetValidationError("Invalid parameter")

        message = get_helpful_error_message(error)

        assert "❌ Invalid parameter" in message
        assert "🔧 Parameter guidelines:" in message
        assert "--count: 1-10000" in message
        assert "--sample: 1-100" in message

    def test_generic_error_fallback(self) -> None:
        """Test fallback for unrecognized errors."""
        error = ValueError("Some generic error")

        message = get_helpful_error_message(error)

        assert message == "Some generic error"
