"""Tests for cache module."""

from datetime import datetime
from unittest.mock import Mock, patch

from swebench_runner import cache


class TestGetCacheDir:
    """Test the get_cache_dir function."""

    def test_get_cache_dir_default(self, monkeypatch, tmp_path):
        """Test get_cache_dir with default location."""
        # Remove env var to use default
        monkeypatch.delenv("SWEBENCH_CACHE_DIR", raising=False)

        with patch("swebench_runner.cache.Path.home") as mock_home:
            mock_home.return_value = tmp_path

            cache_dir = cache.get_cache_dir()

            assert cache_dir == tmp_path / ".swebench"
            assert (tmp_path / ".swebench").exists()
            assert (tmp_path / ".swebench" / "datasets").exists()
            assert (tmp_path / ".swebench" / "logs").exists()
            assert (tmp_path / ".swebench" / "results").exists()

    def test_get_cache_dir_with_env_var(self, monkeypatch, tmp_path):
        """Test get_cache_dir with environment variable."""
        custom_cache = tmp_path / "custom_cache"
        monkeypatch.setenv("SWEBENCH_CACHE_DIR", str(custom_cache))

        cache_dir = cache.get_cache_dir()

        assert cache_dir == custom_cache
        assert custom_cache.exists()
        assert (custom_cache / "datasets").exists()
        assert (custom_cache / "logs").exists()
        assert (custom_cache / "results").exists()

    def test_get_cache_dir_existing(self, tmp_path):
        """Test get_cache_dir when directories already exist."""
        cache_dir = tmp_path / ".swebench"
        cache_dir.mkdir(exist_ok=True)
        (cache_dir / "datasets").mkdir(exist_ok=True)
        (cache_dir / "logs").mkdir(exist_ok=True)
        (cache_dir / "results").mkdir(exist_ok=True)

        with patch("swebench_runner.cache.Path.home") as mock_home:
            mock_home.return_value = tmp_path

            result = cache.get_cache_dir()

            assert result == cache_dir
            # Should not raise even if directories exist


class TestIsFirstRun:
    """Test the is_first_run function."""

    @patch("swebench_runner.cache.get_cache_dir")
    def test_is_first_run_true(self, mock_get_cache_dir, tmp_path):
        """Test when config file doesn't exist."""
        mock_get_cache_dir.return_value = tmp_path

        assert cache.is_first_run() is True

    @patch("swebench_runner.cache.get_cache_dir")
    def test_is_first_run_false(self, mock_get_cache_dir, tmp_path):
        """Test when config file exists."""
        mock_get_cache_dir.return_value = tmp_path
        config_file = tmp_path / "config.toml"
        config_file.write_text("# Config")

        assert cache.is_first_run() is False


class TestMarkFirstRunComplete:
    """Test the mark_first_run_complete function."""

    @patch("swebench_runner.cache.get_cache_dir")
    def test_mark_first_run_complete(self, mock_get_cache_dir, tmp_path):
        """Test marking first run as complete."""
        mock_get_cache_dir.return_value = tmp_path

        cache.mark_first_run_complete()

        config_file = tmp_path / "config.toml"
        assert config_file.exists()

        content = config_file.read_text()
        assert "SWE-bench Runner Configuration" in content
        assert str(tmp_path) in content
        assert "version = \"0.1.0\"" in content
        # Check timestamp format - find the line with first_run_completed
        for line in content.split('\n'):
            if 'first_run_completed' in line and 'T' in line:
                # Extract date from line like: first_run_completed = "2024-01-20T..."
                date_part = line.split('"')[1].split('T')[0]
                year = int(date_part.split('-')[0])
                assert datetime.now().year == year
                break


class TestGetCacheUsage:
    """Test the get_cache_usage function."""

    @patch("swebench_runner.cache.get_cache_dir")
    def test_get_cache_usage_empty(self, mock_get_cache_dir, tmp_path):
        """Test cache usage for empty directories."""
        cache_dir = tmp_path / ".swebench"
        cache_dir.mkdir(exist_ok=True)
        (cache_dir / "datasets").mkdir(exist_ok=True)
        (cache_dir / "logs").mkdir(exist_ok=True)
        (cache_dir / "results").mkdir(exist_ok=True)
        mock_get_cache_dir.return_value = cache_dir

        usage = cache.get_cache_usage()

        assert usage == {"datasets": 0, "logs": 0, "results": 0}

    @patch("swebench_runner.cache.get_cache_dir")
    def test_get_cache_usage_with_files(self, mock_get_cache_dir, tmp_path):
        """Test cache usage with files."""
        cache_dir = tmp_path / ".swebench"
        datasets_dir = cache_dir / "datasets"
        logs_dir = cache_dir / "logs"
        results_dir = cache_dir / "results"

        # Create directories and files
        datasets_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)
        results_dir.mkdir(parents=True)

        (datasets_dir / "data1.json").write_text("x" * 1000)
        (datasets_dir / "data2.json").write_text("x" * 2000)
        (logs_dir / "log1.txt").write_text("x" * 500)
        (results_dir / "result1.json").write_text("x" * 1500)

        mock_get_cache_dir.return_value = cache_dir

        usage = cache.get_cache_usage()

        assert usage["datasets"] == 3000
        assert usage["logs"] == 500
        assert usage["results"] == 1500

    @patch("swebench_runner.cache.get_cache_dir")
    def test_get_cache_usage_with_subdirs(self, mock_get_cache_dir, tmp_path):
        """Test cache usage with subdirectories."""
        cache_dir = tmp_path / ".swebench"
        datasets_dir = cache_dir / "datasets"
        subdir = datasets_dir / "sub"

        subdir.mkdir(parents=True)
        (cache_dir / "logs").mkdir(parents=True)
        (cache_dir / "results").mkdir(parents=True)

        (subdir / "file1.json").write_text("x" * 1000)
        (datasets_dir / "file2.json").write_text("x" * 500)

        mock_get_cache_dir.return_value = cache_dir

        usage = cache.get_cache_usage()

        assert usage["datasets"] == 1500  # Both files counted

    @patch("swebench_runner.cache.get_cache_dir")
    def test_get_cache_usage_missing_dirs(self, mock_get_cache_dir, tmp_path):
        """Test cache usage when some directories don't exist."""
        cache_dir = tmp_path / ".swebench"
        cache_dir.mkdir(exist_ok=True)
        (cache_dir / "datasets").mkdir(exist_ok=True)
        # logs and results don't exist

        mock_get_cache_dir.return_value = cache_dir

        usage = cache.get_cache_usage()

        assert usage["datasets"] == 0
        assert usage["logs"] == 0
        assert usage["results"] == 0


class TestCleanCache:
    """Test the clean_cache function."""

    @patch("swebench_runner.cache.get_cache_dir")
    def test_clean_cache_datasets_only(self, mock_get_cache_dir, tmp_path):
        """Test cleaning only datasets."""
        cache_dir = tmp_path / ".swebench"
        datasets_dir = cache_dir / "datasets"
        logs_dir = cache_dir / "logs"
        results_dir = cache_dir / "results"

        # Create directories and files
        datasets_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)
        results_dir.mkdir(parents=True)

        (datasets_dir / "data.json").write_text("x" * 1000)
        (logs_dir / "log.txt").write_text("x" * 500)
        (results_dir / "result.json").write_text("x" * 750)

        mock_get_cache_dir.return_value = cache_dir

        removed = cache.clean_cache(clean_datasets=True)

        assert removed["datasets"] == 1000
        assert removed["logs"] == 0
        assert removed["results"] == 0
        assert not (datasets_dir / "data.json").exists()
        assert (logs_dir / "log.txt").exists()
        assert (results_dir / "result.json").exists()

    @patch("swebench_runner.cache.get_cache_dir")
    def test_clean_cache_all(self, mock_get_cache_dir, tmp_path):
        """Test cleaning all cache directories."""
        cache_dir = tmp_path / ".swebench"
        datasets_dir = cache_dir / "datasets"
        logs_dir = cache_dir / "logs"
        results_dir = cache_dir / "results"

        # Create directories and files
        datasets_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)
        results_dir.mkdir(parents=True)

        (datasets_dir / "data.json").write_text("x" * 1000)
        (logs_dir / "log.txt").write_text("x" * 500)
        (results_dir / "result.json").write_text("x" * 750)

        mock_get_cache_dir.return_value = cache_dir

        removed = cache.clean_cache(
            clean_datasets=True,
            clean_logs=True,
            clean_results=True
        )

        assert removed["datasets"] == 1000
        assert removed["logs"] == 500
        assert removed["results"] == 750
        assert len(list(datasets_dir.iterdir())) == 0
        assert len(list(logs_dir.iterdir())) == 0
        assert len(list(results_dir.iterdir())) == 0

    @patch("swebench_runner.cache.get_cache_dir")
    def test_clean_cache_dry_run(self, mock_get_cache_dir, tmp_path):
        """Test dry run mode."""
        cache_dir = tmp_path / ".swebench"
        datasets_dir = cache_dir / "datasets"

        datasets_dir.mkdir(parents=True)
        (cache_dir / "logs").mkdir(parents=True)
        (cache_dir / "results").mkdir(parents=True)

        (datasets_dir / "data.json").write_text("x" * 1000)

        mock_get_cache_dir.return_value = cache_dir

        removed = cache.clean_cache(clean_datasets=True, dry_run=True)

        assert removed["datasets"] == 1000
        assert (datasets_dir / "data.json").exists()  # File not removed

    @patch("swebench_runner.cache.get_cache_dir")
    def test_clean_cache_missing_dirs(self, mock_get_cache_dir, tmp_path):
        """Test cleaning when directories don't exist."""
        cache_dir = tmp_path / ".swebench"
        cache_dir.mkdir(exist_ok=True)
        # No subdirectories

        mock_get_cache_dir.return_value = cache_dir

        removed = cache.clean_cache(
            clean_datasets=True,
            clean_logs=True,
            clean_results=True
        )

        assert removed == {"datasets": 0, "logs": 0, "results": 0}


class TestGetDirectoryFunctions:
    """Test the directory getter functions."""

    @patch("swebench_runner.cache.get_cache_dir")
    def test_get_results_dir(self, mock_get_cache_dir, tmp_path):
        """Test get_results_dir."""
        mock_get_cache_dir.return_value = tmp_path

        results_dir = cache.get_results_dir()

        assert results_dir == tmp_path / "results"

    @patch("swebench_runner.cache.get_cache_dir")
    def test_get_logs_dir(self, mock_get_cache_dir, tmp_path):
        """Test get_logs_dir."""
        mock_get_cache_dir.return_value = tmp_path

        logs_dir = cache.get_logs_dir()

        assert logs_dir == tmp_path / "logs"

    @patch("swebench_runner.cache.get_cache_dir")
    def test_get_datasets_dir(self, mock_get_cache_dir, tmp_path):
        """Test get_datasets_dir."""
        mock_get_cache_dir.return_value = tmp_path

        datasets_dir = cache.get_datasets_dir()

        assert datasets_dir == tmp_path / "datasets"


class TestAutoDetectPatchesFile:
    """Test the auto_detect_patches_file function."""

    def test_auto_detect_finds_predictions_jsonl(self, tmp_path):
        """Test finding predictions.jsonl (highest priority)."""
        (tmp_path / "predictions.jsonl").write_text('{"test": 1}')
        (tmp_path / "patches.jsonl").write_text('{"test": 2}')

        result = cache.auto_detect_patches_file(tmp_path)

        assert result == tmp_path / "predictions.jsonl"

    def test_auto_detect_finds_patches_jsonl(self, tmp_path):
        """Test finding patches.jsonl."""
        (tmp_path / "patches.jsonl").write_text('{"test": 1}')
        (tmp_path / "model_patches.jsonl").write_text('{"test": 2}')

        result = cache.auto_detect_patches_file(tmp_path)

        assert result == tmp_path / "patches.jsonl"

    def test_auto_detect_skips_empty_files(self, tmp_path):
        """Test that empty files are skipped."""
        (tmp_path / "predictions.jsonl").write_text('')  # Empty
        (tmp_path / "patches.jsonl").write_text('{"test": 1}')

        result = cache.auto_detect_patches_file(tmp_path)

        assert result == tmp_path / "patches.jsonl"

    def test_auto_detect_none_found(self, tmp_path):
        """Test when no patches files are found."""
        (tmp_path / "other.txt").write_text('test')

        result = cache.auto_detect_patches_file(tmp_path)

        assert result is None

    def test_auto_detect_uses_cwd_default(self):
        """Test that current directory is used by default."""
        with patch("swebench_runner.cache.Path.cwd") as mock_cwd:
            mock_path = Mock()
            mock_cwd.return_value = mock_path

            # Mock the directory iteration
            mock_path.__truediv__ = Mock(
                side_effect=lambda x: Mock(exists=Mock(return_value=False))
            )

            result = cache.auto_detect_patches_file()

            mock_cwd.assert_called_once()
            assert result is None

    def test_auto_detect_checks_all_candidates(self, tmp_path):
        """Test that all candidate names are checked."""
        # Only create the last candidate
        (tmp_path / "predictions.json").write_text('{"test": 1}')

        result = cache.auto_detect_patches_file(tmp_path)

        assert result == tmp_path / "predictions.json"
