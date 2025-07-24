"""Comprehensive test coverage for datasets module to reach 85% threshold."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from swebench_runner.datasets import (
    DatasetManager,
    _validate_instance_ids,
    _validate_numeric_params,
    _validate_regex_pattern,
    configure_hf_auth,
    get_helpful_error_message,
    get_hf_token,
)

# Mock datasets module for tests
try:
    import datasets
except ImportError:
    datasets = None

try:
    import huggingface_hub
except ImportError:
    huggingface_hub = None
from swebench_runner.exceptions import (
    DatasetAuthenticationError,
    DatasetError,
    DatasetNetworkError,
    DatasetNotFoundError,
    DatasetValidationError,
    InstanceValidationError,
    RegexValidationError,
)


class TestRegexValidationEdgeCases:
    """Test regex validation with various edge cases for security."""

    def test_regex_max_length_validation(self) -> None:
        """Test regex pattern length limits."""
        # Test a reasonable pattern under the limit
        pattern_under_limit = "django__django-[0-9]+" + "x" * 100  # Well under 1000
        result = _validate_regex_pattern(pattern_under_limit)
        assert result is not None

        # Test over limit (should fail)
        pattern_over_limit = "a" * 1001
        with pytest.raises(RegexValidationError, match="Pattern too long"):
            _validate_regex_pattern(pattern_over_limit)

    def test_regex_dangerous_patterns(self) -> None:
        """Test detection of ReDoS vulnerable patterns."""
        dangerous_patterns = [
            r"(a+)+$",  # Nested quantifiers
            r"(.*)*",   # Nested star quantifiers
            r"(?=.*)+", # Positive lookahead with quantifier
            r"(?!.*)+", # Negative lookahead with quantifier
            r"(a*)*b",  # Nested star quantifiers
            r"(.*+.*)+", # Nested quantifiers in groups
        ]

        for pattern in dangerous_patterns:
            with pytest.raises(RegexValidationError) as exc_info:
                _validate_regex_pattern(pattern)
            assert "dangerous pattern" in str(exc_info.value).lower()

    def test_regex_timeout_protection(self) -> None:
        """Test regex patterns that could cause timeouts."""
        # This pattern is designed to be slow on certain inputs
        slow_patterns = [
            r"(a+)+" + "b",  # Classic exponential backtracking
            r"(a*)*" + "c",  # Nested quantifiers with suffix
        ]

        for pattern in slow_patterns:
            with pytest.raises(RegexValidationError):
                _validate_regex_pattern(pattern)

    def test_regex_timeout_environment_variable(self) -> None:
        """Test custom timeout configuration via environment variable."""
        # Test with custom timeout value
        import os
        original_timeout = os.environ.get('SWEBENCH_REGEX_TIMEOUT_MS')

        try:
            # Set a custom timeout (not too short, as the test runs quickly)
            os.environ['SWEBENCH_REGEX_TIMEOUT_MS'] = '50'

            # Test that the environment variable is used in the validation
            # This pattern is safe but we're testing the timeout mechanism
            result = _validate_regex_pattern("test.*pattern")
            assert result is not None

            # Test with 0 timeout should cause immediate timeout on slow patterns
            os.environ['SWEBENCH_REGEX_TIMEOUT_MS'] = '0.1'

            # This pattern causes exponential backtracking
            with pytest.raises(RegexValidationError):
                _validate_regex_pattern("(a+)+" + "b")

        finally:
            # Restore original value
            if original_timeout is None:
                os.environ.pop('SWEBENCH_REGEX_TIMEOUT_MS', None)
            else:
                os.environ['SWEBENCH_REGEX_TIMEOUT_MS'] = original_timeout

    def test_regex_exception_during_search(self) -> None:
        """Test handling of exceptions during regex search."""
        # Mock a regex that raises an exception during search
        with patch('re.compile') as mock_compile:
            mock_pattern = Mock()
            # Make search() raise an exception after a delay
            def slow_exception_search(text):
                import time
                time.sleep(0.02)  # 20ms delay
                raise RuntimeError("Search failed")

            mock_pattern.search = Mock(side_effect=slow_exception_search)
            mock_compile.return_value = mock_pattern

            # Set very short timeout to trigger timeout in exception handler
            import os
            original_timeout = os.environ.get('SWEBENCH_REGEX_TIMEOUT_MS')
            try:
                os.environ['SWEBENCH_REGEX_TIMEOUT_MS'] = '10'  # 10ms timeout

                with pytest.raises(RegexValidationError) as exc_info:
                    _validate_regex_pattern("test_pattern")
                assert "exponential complexity" in str(exc_info.value)

            finally:
                if original_timeout is None:
                    os.environ.pop('SWEBENCH_REGEX_TIMEOUT_MS', None)
                else:
                    os.environ['SWEBENCH_REGEX_TIMEOUT_MS'] = original_timeout

    def test_regex_valid_patterns(self) -> None:
        """Test that valid patterns are accepted."""
        valid_patterns = [
            r"django__.*",
            r"test_\d+",
            r"[a-zA-Z]+__[a-zA-Z]+",
            r"^requests__.*-\d+$",
            r"(test|prod)_instance",
        ]

        for pattern in valid_patterns:
            result = _validate_regex_pattern(pattern)
            assert result is not None


class TestInstanceIDValidationEdgeCases:
    """Test instance ID validation security measures."""

    def test_instance_id_empty_list(self) -> None:
        """Test validation with empty list."""
        result = _validate_instance_ids([])
        assert result == []

    def test_instance_id_length_limits(self) -> None:
        """Test instance ID length validation."""
        # Valid length
        valid_id = "a" * 50 + "__" + "b" * 50
        result = _validate_instance_ids([valid_id])
        assert result == [valid_id]

        # Too long
        long_id = "a" * 201 + "__test"
        with pytest.raises(InstanceValidationError, match="too long"):
            _validate_instance_ids([long_id])

    def test_instance_id_path_traversal(self) -> None:
        """Test prevention of path traversal attacks."""
        malicious_ids = [
            "../../../etc/passwd",
            "test../secret",
            "/absolute/path",
            "C:\\windows\\system32",
        ]

        for bad_id in malicious_ids:
            with pytest.raises(InstanceValidationError) as exc_info:
                _validate_instance_ids([bad_id])
            assert "invalid" in str(exc_info.value).lower()

    def test_instance_id_format_validation(self) -> None:
        """Test instance ID format requirements."""
        # Invalid formats
        invalid_formats = [
            "123starts_with_number",  # Must start with letter
            "has spaces",
            "special!chars",
            "unicode_🚀",
            "",  # Empty string
        ]

        for invalid_id in invalid_formats:
            with pytest.raises(InstanceValidationError):
                _validate_instance_ids([invalid_id])

        # Valid formats - note these don't require double underscore
        valid_formats = [
            "django_django-12345",
            "requests_requests-456",
            "numpy.numpy-789",
            "test_TEST-123",
            "repo.feature-branch.fix",
        ]

        for valid_id in valid_formats:
            result = _validate_instance_ids([valid_id])
            assert result == [valid_id]


class TestNumericParamValidation:
    """Test numeric parameter validation."""

    def test_count_validation(self) -> None:
        """Test count parameter validation."""
        # Valid counts
        _validate_numeric_params(count=1)
        _validate_numeric_params(count=100)
        _validate_numeric_params(count=10000)

        # Invalid counts
        with pytest.raises(DatasetValidationError, match="positive integer"):
            _validate_numeric_params(count=0)

        with pytest.raises(DatasetValidationError, match="positive integer"):
            _validate_numeric_params(count=-5)

        with pytest.raises(DatasetValidationError, match="positive integer"):
            _validate_numeric_params(count="not a number")  # type: ignore

        with pytest.raises(DatasetValidationError, match="Count too large"):
            _validate_numeric_params(count=10001)

    def test_sample_percent_validation(self) -> None:
        """Test sample percentage validation."""
        # Valid percentages
        _validate_numeric_params(sample_percent=1.0)
        _validate_numeric_params(sample_percent=50.5)
        _validate_numeric_params(sample_percent=100.0)

        # Invalid percentages - error says "between 0-100" not "between 0 and 100"
        with pytest.raises(DatasetValidationError) as exc_info:
            _validate_numeric_params(sample_percent=0.0)
        assert "between 0-100" in str(exc_info.value)

        with pytest.raises(DatasetValidationError) as exc_info:
            _validate_numeric_params(sample_percent=-10.0)
        assert "between 0-100" in str(exc_info.value)

        with pytest.raises(DatasetValidationError) as exc_info:
            _validate_numeric_params(sample_percent=101.0)
        assert "between 0-100" in str(exc_info.value)

        with pytest.raises(DatasetValidationError) as exc_info:
            _validate_numeric_params(sample_percent="fifty")  # type: ignore
        assert "must be number" in str(exc_info.value)

    def test_random_seed_validation(self) -> None:
        """Test random seed validation."""
        # Valid seeds
        _validate_numeric_params(random_seed=0)
        _validate_numeric_params(random_seed=42)
        _validate_numeric_params(random_seed=2**31 - 1)

        # Invalid seeds - check actual error message
        with pytest.raises(DatasetValidationError) as exc_info:
            _validate_numeric_params(random_seed=-1)
        assert "Random seed must be" in str(exc_info.value)

        with pytest.raises(DatasetValidationError) as exc_info:
            _validate_numeric_params(random_seed=2**32)  # Actually 2^32 not 2^31
        assert "Random seed must be" in str(exc_info.value)


class TestDatasetNetworkErrors:
    """Test network error handling in dataset operations."""

    @patch('datasets.load_dataset')
    def test_fetch_dataset_connection_error(self, mock_load_dataset: Mock) -> None:
        """Test handling of connection errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            mock_load_dataset.side_effect = Exception("Connection timeout")

            with pytest.raises(DatasetNetworkError) as exc_info:
                manager.fetch_dataset("lite")
            assert "Network error" in str(exc_info.value)

    @patch('datasets.load_dataset')
    def test_fetch_dataset_authentication_error(self, mock_load_dataset: Mock) -> None:
        """Test handling of authentication errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            # Test various auth error formats
            auth_errors = [
                "401 Unauthorized",
                "HTTP 401: Unauthorized",
                "401 Client Error",
            ]

            for error_msg in auth_errors:
                mock_load_dataset.side_effect = Exception(error_msg)
                with pytest.raises(DatasetAuthenticationError):
                    manager.fetch_dataset("lite")

    @patch('datasets.load_dataset')
    def test_fetch_dataset_offline_mode(self, mock_load_dataset: Mock) -> None:
        """Test offline mode behavior."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            # Test offline with missing cache
            mock_load_dataset.side_effect = Exception("Dataset not found locally")

            with pytest.raises(DatasetError) as exc_info:
                manager.fetch_dataset("lite", offline=True)
            assert "not available in offline mode" in str(exc_info.value)

    @patch('datasets.load_dataset')
    def test_fetch_dataset_generic_error(self, mock_load_dataset: Mock) -> None:
        """Test handling of generic errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            mock_load_dataset.side_effect = Exception("Unknown error occurred")

            with pytest.raises(DatasetError) as exc_info:
                manager.fetch_dataset("lite")
            assert "Failed to load dataset" in str(exc_info.value)


class TestMemoryManagement:
    """Test memory estimation and requirements checking."""

    def test_estimate_memory_usage_with_info(self) -> None:
        """Test memory estimation when dataset info is available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            with patch.object(manager, 'get_dataset_info') as mock_info:
                mock_info.return_value = {
                    'total_instances': 300,
                    'dataset_size_mb': 1.2
                }

                # Test without count
                usage = manager.estimate_memory_usage("lite")
                assert usage['instances'] == 300
                assert usage['estimated_ram_mb'] > 0
                assert usage['download_size_mb'] == 1.2
                assert usage['total_mb'] > usage['estimated_ram_mb']

                # Test with count
                usage = manager.estimate_memory_usage("lite", count=50)
                assert usage['instances'] == 50
                assert usage['estimated_ram_mb'] < 1  # Should be small for 50 instances

    def test_estimate_memory_usage_fallback(self) -> None:
        """Test memory estimation when dataset info fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            with patch.object(manager, 'get_dataset_info') as mock_info:
                mock_info.side_effect = Exception("Network error")

                # Should use conservative defaults
                usage = manager.estimate_memory_usage("lite", count=100)
                assert usage['instances'] == 100
                assert usage['estimated_ram_mb'] > 0
                assert usage['download_size_mb'] == 5.0  # Conservative default

    @patch('psutil.virtual_memory')
    def test_check_memory_requirements_sufficient(self, mock_memory: Mock) -> None:
        """Test memory check with sufficient memory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            # Mock 8GB available
            mock_memory.return_value.available = 8 * 1024 * 1024 * 1024

            can_proceed, msg = manager.check_memory_requirements("lite")
            assert can_proceed is True
            assert msg == "" or "Memory usage:" in msg

    @patch('psutil.virtual_memory')
    def test_check_memory_requirements_insufficient(self, mock_memory: Mock) -> None:
        """Test memory check with insufficient memory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            # Mock only 100MB available
            mock_memory.return_value.available = 100 * 1024 * 1024

            with patch.object(manager, 'estimate_memory_usage') as mock_estimate:
                mock_estimate.return_value = {
                    'instances': 2000,
                    'estimated_ram_mb': 500,
                    'download_size_mb': 10,
                    'total_mb': 510
                }

                can_proceed, msg = manager.check_memory_requirements("full")
                assert can_proceed is False
                assert "High memory usage warning" in msg
                assert "Available RAM:" in msg

    def test_check_memory_requirements_no_psutil(self) -> None:
        """Test memory check when psutil is not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            # Mock psutil import to fail
            import sys
            psutil_backup = sys.modules.get('psutil')
            sys.modules['psutil'] = None

            try:
                # Should assume sufficient memory
                can_proceed, msg = manager.check_memory_requirements("lite")
                assert can_proceed is True
                assert msg == ""
            finally:
                # Restore psutil
                if psutil_backup is not None:
                    sys.modules['psutil'] = psutil_backup
                else:
                    sys.modules.pop('psutil', None)


class TestStreamingSupport:
    """Test streaming functionality for large datasets."""

    @patch('datasets.load_dataset')
    def test_get_instances_streaming_basic(self, mock_load: Mock) -> None:
        """Test basic streaming functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            # Create mock dataset
            mock_instances = [
                {
                    'instance_id': f'test_{i}',
                    'patch': f'patch_{i}',
                    'repo': f'repo_{i}',
                    'base_commit': f'commit_{i}',
                    'problem_statement': f'problem_{i}'
                }
                for i in range(250)
            ]

            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=250)
            mock_dataset.filter = Mock(return_value=mock_dataset)

            # Create a mock that returns the correct slice of instances
            def mock_select(indices):
                selected = [mock_instances[i] for i in indices]
                selected_mock = Mock()
                selected_mock.__iter__ = Mock(return_value=iter(selected))
                return selected_mock

            mock_dataset.select = Mock(side_effect=mock_select)
            mock_load.return_value = mock_dataset

            # Test streaming with batches
            batches = list(manager.get_instances_streaming("lite", batch_size=100))
            assert len(batches) == 3  # 250 instances / 100 batch size
            assert len(batches[0]) == 100
            assert len(batches[1]) == 100
            assert len(batches[2]) == 50

            # Verify content structure
            first_item = batches[0][0]
            assert 'instance_id' in first_item
            assert 'patch' in first_item
            assert first_item['instance_id'] == 'test_0'

    @patch('datasets.load_dataset')
    def test_get_instances_streaming_with_filters(self, mock_load: Mock) -> None:
        """Test streaming with various filters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=100)
            mock_dataset.filter = Mock(return_value=mock_dataset)

            # Create empty select result
            empty_mock = Mock()
            empty_mock.__iter__ = Mock(return_value=iter([]))
            mock_dataset.select = Mock(return_value=empty_mock)

            mock_load.return_value = mock_dataset

            # Test with specific instances
            list(manager.get_instances_streaming(
                "lite",
                batch_size=50,
                instances=['test_1', 'test_2']
            ))
            assert mock_dataset.filter.called

            # Test with pattern
            list(manager.get_instances_streaming(
                "lite",
                batch_size=50,
                subset_pattern="django__*"
            ))
            assert mock_dataset.filter.call_count >= 2

    def test_save_streaming_as_jsonl(self) -> None:
        """Test saving streaming instances to JSONL."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))
            output_file = Path(temp_dir) / "output.jsonl"

            # Create streaming generator
            def instance_generator():
                for batch_num in range(3):
                    yield [
                        {'instance_id': f'test_{i}', 'patch': f'patch_{i}'}
                        for i in range(batch_num * 10, (batch_num + 1) * 10)
                    ]

            # Save streaming instances
            total = manager.save_streaming_as_jsonl(
                instance_generator(),
                output_file
            )

            assert total == 30
            assert output_file.exists()

            # Verify content
            lines = output_file.read_text().strip().split('\n')
            assert len(lines) == 30

            # Check first and last items
            first_item = json.loads(lines[0])
            assert first_item['instance_id'] == 'test_0'
            assert first_item['patch'] == 'patch_0'

            last_item = json.loads(lines[29])
            assert last_item['instance_id'] == 'test_29'
            assert last_item['patch'] == 'patch_29'

    def test_save_streaming_empty_stream(self) -> None:
        """Test saving empty streaming instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))
            output_file = Path(temp_dir) / "empty.jsonl"

            # Empty generator
            def empty_generator():
                return
                yield  # Make it a generator

            total = manager.save_streaming_as_jsonl(
                empty_generator(),
                output_file
            )

            assert total == 0
            assert output_file.exists()
            assert output_file.read_text() == ""


class TestAuthentication:
    """Test HuggingFace authentication functionality."""

    def test_get_hf_token_from_env(self) -> None:
        """Test token retrieval from environment variables."""
        # Test HF_TOKEN
        with patch.dict('os.environ', {'HF_TOKEN': 'test_token_123'}):
            token = get_hf_token()
            assert token == 'test_token_123'

        # Test HUGGING_FACE_HUB_TOKEN as fallback
        with patch.dict('os.environ', {'HUGGING_FACE_HUB_TOKEN': 'fallback_token'}):
            token = get_hf_token()
            assert token == 'fallback_token'

        # Test no token
        with patch.dict('os.environ', {}, clear=True):
            token = get_hf_token()
            assert token is None

    def test_configure_hf_auth_with_token(self) -> None:
        """Test authentication configuration with token."""
        with patch(
            'swebench_runner.datasets.get_hf_token', return_value="test_token_123"
        ):
            with patch('huggingface_hub.login') as mock_login:
                result = configure_hf_auth()
                assert result is True
                mock_login.assert_called_once_with(
                    token="test_token_123",
                    add_to_git_credential=False
                )

    @patch('swebench_runner.datasets.get_hf_token')
    def test_configure_hf_auth_without_token(self, mock_get_token: Mock) -> None:
        """Test authentication configuration without token."""
        mock_get_token.return_value = None

        result = configure_hf_auth()
        assert result is False

    def test_configure_hf_auth_import_error(self) -> None:
        """Test authentication when huggingface_hub not available."""
        with patch('swebench_runner.datasets.get_hf_token', return_value="test_token"):
            # Mock the huggingface_hub import to fail
            import sys
            original_modules = sys.modules.copy()

            # Remove huggingface_hub if it exists
            if 'huggingface_hub' in sys.modules:
                del sys.modules['huggingface_hub']

            # Mock the import to fail
            def mock_import(name, *args, **kwargs):
                if name == 'huggingface_hub':
                    raise ImportError("No module named 'huggingface_hub'")
                return __import__(name, *args, **kwargs)

            with patch('builtins.__import__', side_effect=mock_import):
                result = configure_hf_auth()
                assert result is False

            # Restore original modules
            sys.modules.update(original_modules)

    def test_save_as_jsonl_basic(self) -> None:
        """Test basic JSONL save functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))
            output_file = Path(temp_dir) / "test.jsonl"

            instances = [
                {
                    'instance_id': 'test_1',
                    'patch': 'patch_1',
                    'repo': 'repo_1',
                    'base_commit': 'commit_1',
                    'problem_statement': 'problem_1'
                },
                {
                    'instance_id': 'test_2',
                    'patch': 'patch_2',
                    'repo': 'repo_2',
                    'base_commit': 'commit_2',
                    'problem_statement': 'problem_2'
                }
            ]

            manager.save_as_jsonl(instances, output_file)

            # Verify file exists and content
            assert output_file.exists()
            lines = output_file.read_text().strip().split('\n')
            assert len(lines) == 2

            # Check only instance_id and patch are saved
            first_item = json.loads(lines[0])
            assert first_item == {'instance_id': 'test_1', 'patch': 'patch_1'}
            assert 'repo' not in first_item
            assert 'base_commit' not in first_item


class TestGetInstancesComprehensive:
    """Comprehensive tests for get_instances functionality."""

    @patch('datasets.load_dataset')
    def test_get_instances_with_all_filters(self, mock_load: Mock) -> None:
        """Test get_instances with all filtering options."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            # Create comprehensive mock dataset
            mock_instances = [
                {
                    'instance_id': f'django__{i:03d}',
                    'patch': f'patch_{i}',
                    'repo': 'django/django',
                    'base_commit': f'commit_{i}',
                    'problem_statement': f'Fix issue {i}'
                }
                for i in range(50)
            ]

            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=50)
            mock_dataset.__iter__ = Mock(return_value=iter(mock_instances))

            # Simple mock that returns the same dataset
            mock_dataset.filter = Mock(return_value=mock_dataset)

            # Mock select to return a subset
            def mock_select(indices):
                selected_instances = [
                    mock_instances[i] for i in indices if i < len(mock_instances)
                ]
                selected_mock = Mock()
                selected_mock.__iter__ = Mock(return_value=iter(selected_instances))
                return selected_mock

            mock_dataset.select = Mock(side_effect=mock_select)
            mock_load.return_value = mock_dataset

            # Test with regex pattern, count, and seed
            instances = manager.get_instances(
                "lite",
                count=5,
                subset_pattern="django__.*",  # Simpler pattern
                use_regex=True,
                random_seed=42
            )

            # Should get 5 instances (due to count)
            assert len(instances) <= 5

    @patch('datasets.load_dataset')
    def test_get_instances_percentage_edge_cases(self, mock_load: Mock) -> None:
        """Test percentage sampling edge cases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            # Dataset with only 5 instances
            mock_instances = [
                {'instance_id': f'test_{i}', 'patch': f'patch_{i}'}
                for i in range(5)
            ]

            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=5)
            mock_dataset.__iter__ = Mock(return_value=iter(mock_instances))
            mock_dataset.filter = Mock(return_value=mock_dataset)

            def mock_select(indices):
                selected = [
                    mock_instances[i] for i in indices if i < len(mock_instances)
                ]
                selected_mock = Mock()
                selected_mock.__iter__ = Mock(return_value=iter(selected))
                return selected_mock

            mock_dataset.select = Mock(side_effect=mock_select)
            mock_load.return_value = mock_dataset

            # Test 10% of 5 instances (should give at least 1)
            instances = manager.get_instances(
                "lite",
                sample_percent=10.0
            )
            assert len(instances) >= 1

            # Test 100% sampling
            instances = manager.get_instances(
                "lite",
                sample_percent=100.0
            )
            assert len(instances) == 5

    @patch('datasets.load_dataset')
    def test_get_instances_count_exceeds_dataset(self, mock_load: Mock) -> None:
        """Test when requested count exceeds dataset size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            # Small dataset
            mock_instances = [
                {'instance_id': f'test_{i}', 'patch': f'patch_{i}'}
                for i in range(3)
            ]

            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=3)
            mock_dataset.__iter__ = Mock(return_value=iter(mock_instances))
            mock_dataset.filter = Mock(return_value=mock_dataset)
            mock_dataset.select = Mock(return_value=mock_dataset)
            mock_load.return_value = mock_dataset

            # Request more than available - should log warning
            # Since we changed from print to logging, we need to check logs
            with patch('swebench_runner.datasets.logger') as mock_logger:
                instances = manager.get_instances(
                    "lite",
                    count=10
                )
                assert len(instances) == 3
                mock_logger.warning.assert_called_once_with(
                    "Requested %d instances but dataset only has %d",
                    10, 3
                )


class TestDatasetManagerInit:
    """Test DatasetManager initialization."""

    def test_init_creates_cache_dir(self) -> None:
        """Test that DatasetManager creates cache directory on init."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            manager = DatasetManager(base_path)

            assert manager.cache_dir == base_path / "datasets"
            assert manager.cache_dir.exists()
            assert manager.cache_dir.is_dir()


class TestDatasetInfoErrors:
    """Test error handling in dataset info retrieval."""

    def test_get_dataset_info_invalid_dataset(self) -> None:
        """Test dataset info with invalid dataset name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            with pytest.raises(ValueError, match="Unknown dataset"):
                manager.get_dataset_info("invalid_dataset")

    @patch('datasets.load_dataset_builder')
    def test_get_dataset_info_network_error(self, mock_builder: Mock) -> None:
        """Test dataset info with network errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            mock_builder.side_effect = Exception("Network timeout")

            with pytest.raises(DatasetNetworkError) as exc_info:
                manager.get_dataset_info("lite")
            assert "Network error" in str(exc_info.value)

    @patch('datasets.load_dataset_builder')
    def test_get_dataset_info_missing_splits(self, mock_builder: Mock) -> None:
        """Test dataset info with missing test split."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            mock_info = Mock()
            mock_info.splits = {'train': Mock(num_examples=100)}  # Missing 'test'
            mock_info.description = "Test dataset"
            mock_info.download_size = 1024 * 1024
            mock_info.dataset_size = 2 * 1024 * 1024
            mock_builder.return_value.info = mock_info

            with pytest.raises(DatasetError) as exc_info:
                manager.get_dataset_info("lite")
            assert "Failed to get info" in str(exc_info.value)

    def test_get_dataset_info_import_error(self) -> None:
        """Test dataset info when datasets library not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))

            # Simulate the ImportError in the try block
            with patch('builtins.__import__', side_effect=ImportError("No module")):
                with pytest.raises(ImportError, match="HuggingFace datasets library"):
                    manager.get_dataset_info("lite")


class TestFileLockingAndCleanup:
    """Test file cleanup with proper locking."""

    def test_cleanup_temp_files_basic(self) -> None:
        """Test basic temp file cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))
            temp_files_dir = manager.cache_dir.parent / "temp"
            temp_files_dir.mkdir(exist_ok=True)

            # Create test files
            test_file1 = temp_files_dir / "test1.jsonl"
            test_file2 = temp_files_dir / "test2.jsonl"
            test_file1.write_text("test content 1")
            test_file2.write_text("test content 2")

            assert test_file1.exists()
            assert test_file2.exists()

            # Clean up
            manager.cleanup_temp_files()

            # Temp directory should exist but be empty
            assert temp_files_dir.exists()
            assert not test_file1.exists()
            assert not test_file2.exists()

    def test_cleanup_temp_files_concurrent_access(self) -> None:
        """Test cleanup with simulated concurrent access."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))
            temp_files_dir = manager.cache_dir.parent / "temp"
            lock_file = temp_files_dir.parent / ".temp_cleanup.lock"

            # Simulate locked file (another process cleaning)
            import fcntl
            temp_files_dir.parent.mkdir(parents=True, exist_ok=True)

            with open(lock_file, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                # This should not raise an error, just skip the cleanup
                manager.cleanup_temp_files()

                # When locked, cleanup is skipped, so temp dir may not be created
                # The function returns early when it can't acquire the lock
                pass

    def test_cleanup_temp_files_missing_directory(self) -> None:
        """Test cleanup when temp directory doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))
            temp_files_dir = manager.cache_dir.parent / "temp"

            # Ensure it doesn't exist
            assert not temp_files_dir.exists()

            # Cleanup should create it
            manager.cleanup_temp_files()

            # Now it should exist
            assert temp_files_dir.exists()

    def test_cleanup_temp_files_permission_error(self) -> None:
        """Test cleanup with permission errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatasetManager(Path(temp_dir))
            temp_files_dir = manager.cache_dir.parent / "temp"
            temp_files_dir.mkdir(exist_ok=True)

            # Create a file
            test_file = temp_files_dir / "test.jsonl"
            test_file.write_text("test content")

            # Simulate rename failure by mocking
            with patch.object(Path, 'rename', side_effect=OSError("Permission denied")):
                # Should handle error gracefully
                manager.cleanup_temp_files()

                # File should still exist (cleanup failed)
                assert test_file.exists()


class TestEnhancedErrorMessagesIntegration:
    """Test enhanced error message generation with various contexts."""

    def test_error_message_with_context(self) -> None:
        """Test error messages with different contexts."""
        # Authentication error
        error = DatasetAuthenticationError("Auth required")
        context = {'dataset': 'verified'}
        msg = get_helpful_error_message(error, context)
        assert "verified dataset" in msg
        assert "HF_TOKEN" in msg

        # Network error online
        error = DatasetNetworkError("Connection failed")
        context = {'dataset': 'lite', 'offline': False}
        msg = get_helpful_error_message(error, context)
        assert "Network error downloading" in msg
        assert "--offline flag" in msg

        # Network error offline
        context = {'dataset': 'full', 'offline': True}
        msg = get_helpful_error_message(error, context)
        assert "not available in offline mode" in msg
        assert "Remove --offline flag" in msg

    def test_error_message_fallback(self) -> None:
        """Test error message for unknown error types."""
        # Unknown error type
        error = RuntimeError("Some runtime error")
        msg = get_helpful_error_message(error)
        assert msg == "Some runtime error"

        # DatasetError (generic)
        error = DatasetError("Generic dataset error")
        msg = get_helpful_error_message(error)
        assert msg == "Generic dataset error"

    def test_error_message_all_types(self) -> None:
        """Test all error message types."""
        # DatasetNotFoundError
        error = DatasetNotFoundError("Dataset xyz not found")
        msg = get_helpful_error_message(error)
        assert "Dataset xyz not found" in msg
        assert "Available datasets:" in msg
        assert "lite:" in msg
        assert "verified:" in msg
        assert "full:" in msg

        # RegexValidationError
        error = RegexValidationError("Invalid regex pattern")
        msg = get_helpful_error_message(error)
        assert "Invalid regex pattern" in msg
        assert "How to fix:" in msg
        assert "simpler patterns" in msg

        # InstanceValidationError
        error = InstanceValidationError("Invalid instance ID format")
        msg = get_helpful_error_message(error)
        assert "Invalid instance ID format" in msg
        assert "Valid instance ID format:" in msg
        assert "django__django-123" in msg

        # DatasetValidationError
        error = DatasetValidationError("Invalid parameter")
        msg = get_helpful_error_message(error)
        assert "Invalid parameter" in msg
        assert "Parameter guidelines:" in msg
        assert "--count:" in msg
        assert "--sample:" in msg
