"""Dataset management for SWE-bench datasets from HuggingFace."""

from __future__ import annotations

import fcntl
import fnmatch
import json
import logging
import os
import random
import re
import shutil
import string
import time
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from re import Pattern
from typing import Any

from .exceptions import (
    DatasetAuthenticationError,
    DatasetError,
    DatasetNetworkError,
    DatasetNotFoundError,
    DatasetValidationError,
    InstanceValidationError,
    RegexValidationError,
)

# Configure logger
logger = logging.getLogger(__name__)


def get_helpful_error_message(
    error: Exception, context: dict[str, Any] | None = None
) -> str:
    """Generate contextual error messages with fix suggestions."""
    if context is None:
        context = {}

    if isinstance(error, DatasetAuthenticationError):
        dataset = context.get('dataset', 'unknown')
        return (
            f"❌ Authentication required for {dataset} dataset\n"
            "\n"
            "   🔧 How to fix:\n"
            "   1. Get a free token at: https://huggingface.co/settings/tokens\n"
            "   2. Set environment variable: export HF_TOKEN=your_token_here\n"
            "   3. Restart your terminal/IDE after setting the token\n"
            "\n"
            "   💡 The token is free and only takes 30 seconds to set up!"
        )
    elif isinstance(error, DatasetNetworkError):
        dataset = context.get('dataset', 'lite')
        offline_mode = context.get('offline', False)
        if offline_mode:
            return (
                f"❌ Dataset {dataset} not available in offline mode\n"
                "\n"
                "   🔧 How to fix:\n"
                "   1. Remove --offline flag to download the dataset\n"
                f"   2. Run: swebench info -d {dataset} (to check if cached)\n"
                "   3. Or try a different dataset that might be cached\n"
                "\n"
                "   💡 Download once, then use --offline for faster runs!"
            )
        else:
            return (
                "❌ Network error downloading dataset\n"
                "\n"
                "   🔧 How to fix:\n"
                "   1. Check your internet connection\n"
                "   2. Try again in a few minutes (might be temporary outage)\n"
                "   3. Use --offline flag if dataset is already cached\n"
                f"   4. Check cache status: swebench info -d {dataset}\n"
                "\n"
                "   💡 Most network issues resolve themselves quickly!"
            )
    elif isinstance(error, DatasetNotFoundError):
        return (
            f"❌ {str(error)}\n"
            "\n"
            "   🔧 Available datasets:\n"
            "   • lite: 300 instances (1.2MB) - Great for testing\n"
            "   • verified: 500 instances (2MB) - Human-verified fixes\n"
            "   • full: 2294 instances (8MB) - Complete benchmark\n"
            "\n"
            "   💡 Start with 'lite' for fastest results!"
        )
    elif isinstance(error, RegexValidationError):
        return (
            f"❌ {str(error)}\n"
            "\n"
            "   🔧 How to fix:\n"
            "   1. Use simpler patterns like 'django__*' or 'requests__*'\n"
            "   2. Avoid nested quantifiers like (a+)+ or (a*)*\n"
            "   3. Test your pattern at: https://regex101.com\n"
            "\n"
            "   💡 Glob patterns (django__*) are often simpler than regex!"
        )
    elif isinstance(error, InstanceValidationError):
        return (
            f"❌ {str(error)}\n"
            "\n"
            "   🔧 Valid instance ID format:\n"
            "   • Must start with a letter (a-z, A-Z)\n"
            "   • Can contain letters, digits, underscore, hyphen, dot\n"
            "   • Examples: django__django-123, requests__requests-456\n"
            "\n"
            "   💡 Use 'swebench info -d lite' to see valid instance IDs!"
        )
    elif isinstance(error, DatasetValidationError):
        return (
            f"❌ {str(error)}\n"
            "\n"
            "   🔧 Parameter guidelines:\n"
            "   • --count: 1-10000 (number of instances)\n"
            "   • --sample: 1-100 (percentage like '10%')\n"
            "   • --instances: comma-separated list\n"
            "\n"
            "   💡 Try --count 5 for a quick test run!"
        )
    else:
        return str(error)

# Input validation constants
MAX_REGEX_LENGTH = 1000
MAX_INSTANCE_COUNT = 10000
REGEX_TEST_TIMEOUT_MS = 10  # Default timeout for ReDoS testing
VALID_INSTANCE_ID_CHARS = set(string.ascii_letters + string.digits + '_-.')


def _validate_regex_pattern(pattern: str) -> Pattern[str]:
    """Validate regex pattern and prevent ReDoS attacks."""
    if not pattern:
        raise RegexValidationError("Pattern cannot be empty")

    if len(pattern) > MAX_REGEX_LENGTH:
        raise RegexValidationError(
            f"Pattern too long ({len(pattern)} chars). Maximum: {MAX_REGEX_LENGTH}"
        )

    # Check for dangerous patterns that could cause ReDoS
    dangerous_patterns = [
        (r'\(\?\=.*\)\+', 'Positive lookahead with quantifier'),
        (r'\(\?\!.*\)\+', 'Negative lookahead with quantifier'),
        (r'.*\*.*\*', 'Nested quantifiers'),
        (r'\(.*\)\+.*\(.*\)\+', 'Multiple groups with quantifiers'),
        (r'\[.*\]\*.*\[.*\]\*', 'Multiple character classes with quantifiers'),
        (r'\(.*\+.*\)\+', 'Nested quantifiers in groups (classic ReDoS)'),
        (r'\(.*\*.*\)\*', 'Nested star quantifiers in groups'),
        (r'.{500,}', 'Extremely large quantifiers (potential ReDoS)'),
        (r'[a-zA-Z0-9_]\{[0-9]{3,},\}', 'Large numeric quantifiers (potential ReDoS)'),
    ]

    for danger_pattern, description in dangerous_patterns:
        if re.search(danger_pattern, pattern):
            raise RegexValidationError(
                f"Potentially dangerous pattern detected: {description}. "
                f"Pattern: {pattern[:50]}{'...' if len(pattern) > 50 else ''}"
            )

    try:
        # Compile with timeout protection by limiting complexity
        compiled = re.compile(pattern)

        # Test with a simple string to catch some ReDoS patterns
        # Use shorter test to avoid getting stuck
        test_strings = ["a" * 10, "b" * 10, "abc123", "test_string"]

        for test_string in test_strings:
            start_time = time.time()
            try:
                compiled.search(test_string)
                timeout_ms = os.environ.get(
                    'SWEBENCH_REGEX_TIMEOUT_MS', REGEX_TEST_TIMEOUT_MS
                )
                timeout = float(timeout_ms) / 1000
                if time.time() - start_time > timeout:
                    raise RegexValidationError(
                        "Pattern appears to have exponential complexity (ReDoS risk)"
                    )
            except Exception as ex:
                # If any error occurs during search, consider it suspicious
                timeout_ms = os.environ.get(
                    'SWEBENCH_REGEX_TIMEOUT_MS', REGEX_TEST_TIMEOUT_MS
                )
                timeout = float(timeout_ms) / 1000
                if time.time() - start_time > timeout:
                    raise RegexValidationError(
                        "Pattern appears to have exponential complexity (ReDoS risk)"
                    ) from ex
                # Otherwise continue to next test

        return compiled
    except re.error as e:
        raise RegexValidationError(f"Invalid regex pattern: {e}") from e


@lru_cache(maxsize=128)
def _validate_regex_pattern_cached(pattern: str) -> Pattern[str]:
    """Cached version of regex pattern validation.

    Note: This caches the compiled pattern, improving performance
    for repeated use of the same regex patterns.
    """
    return _validate_regex_pattern(pattern)


def _validate_instance_ids(instances: list[str]) -> list[str]:
    """Validate instance ID format and prevent injection attacks."""
    if not instances:
        return instances

    if len(instances) > MAX_INSTANCE_COUNT:
        raise InstanceValidationError(
            f"Too many instances specified ({len(instances)}). "
            f"Maximum: {MAX_INSTANCE_COUNT}"
        )

    validated = []
    for instance_id in instances:
        if not instance_id or not isinstance(instance_id, str):
            raise InstanceValidationError(
                f"Invalid instance ID: {instance_id!r} (must be non-empty string)"
            )

        # Length validation
        if len(instance_id) > 200:  # Reasonable max length
            raise InstanceValidationError(
                f"Instance ID too long: {instance_id[:50]}... "
                f"({len(instance_id)} chars, max 200)"
            )

        # Character validation - only allow safe characters
        invalid_chars = set(instance_id) - VALID_INSTANCE_ID_CHARS
        if invalid_chars:
            raise InstanceValidationError(
                f"Instance ID contains invalid characters: {sorted(invalid_chars)}. "
                f"Allowed: letters, digits, underscore, hyphen, dot"
            )

        # Pattern validation - must look like actual instance IDs
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_.-]*$', instance_id):
            raise InstanceValidationError(
                f"Instance ID has invalid format: {instance_id!r}. "
                f"Must start with letter and contain only letters, digits, _, -, ."
            )

        validated.append(instance_id.strip())

    return validated


def _validate_numeric_params(
    count: int | None = None,
    sample_percent: float | None = None,
    random_seed: int | None = None
) -> None:
    """Validate numeric parameters."""
    if count is not None:
        if not isinstance(count, int) or count < 1:
            raise DatasetValidationError(
                f"Count must be positive integer, got: {count!r}"
            )
        if count > MAX_INSTANCE_COUNT:
            raise DatasetValidationError(
                f"Count too large ({count}). Maximum: {MAX_INSTANCE_COUNT}"
            )

    if sample_percent is not None:
        if not isinstance(sample_percent, (int, float)):  # noqa: UP038
            raise DatasetValidationError(
                f"Sample percent must be number, got: {sample_percent!r}"
            )
        if sample_percent <= 0 or sample_percent > 100:
            raise DatasetValidationError(
                f"Sample percent must be between 0-100, got: {sample_percent}"
            )

    if random_seed is not None:
        if not isinstance(random_seed, int):
            raise DatasetValidationError(
                f"Random seed must be integer, got: {random_seed!r}"
            )
        if random_seed < 0 or random_seed > 2**32 - 1:
            raise DatasetValidationError(
                f"Random seed must be 0 <= seed <= {2**32-1}, got: {random_seed}"
            )


class DatasetManager:
    """Manages SWE-bench dataset downloads and access."""

    DATASET_MAPPING = {
        'lite': 'princeton-nlp/SWE-bench_Lite',
        'verified': 'princeton-nlp/SWE-bench_Verified',
        'full': 'princeton-nlp/SWE-bench'
    }

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir / "datasets"
        self.cache_dir.mkdir(exist_ok=True)

    def fetch_dataset(
        self, dataset_name: str, force_download: bool = False, offline: bool = False
    ) -> Any:
        """Fetch dataset from HuggingFace or cache."""
        if dataset_name not in self.DATASET_MAPPING:
            raise DatasetNotFoundError(
                f"Unknown dataset: {dataset_name}. "
                f"Choose from: {list(self.DATASET_MAPPING.keys())}"
            )

        hf_dataset_name = self.DATASET_MAPPING[dataset_name]

        # Set cache directory for HuggingFace datasets
        os.environ['HF_DATASETS_CACHE'] = str(self.cache_dir)

        try:
            from datasets import load_dataset  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "HuggingFace datasets library is required for dataset auto-fetch. "
                "Install with: pip install datasets"
            ) from e

        # Load dataset (will download if not cached)
        try:
            if offline:
                # In offline mode, only use cached data
                download_mode = 'reuse_dataset_if_exists'
            elif force_download:
                download_mode = 'force_redownload'
            else:
                download_mode = 'reuse_dataset_if_exists'

            dataset = load_dataset(  # type: ignore[import-not-found]
                hf_dataset_name,
                split='test',  # SWE-bench uses 'test' split
                download_mode=download_mode
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "401" in error_msg or "unauthorized" in error_msg:
                raise DatasetAuthenticationError(
                    f"Authentication required for dataset {dataset_name}"
                ) from e
            elif offline and ("not found" in error_msg or "not available" in error_msg):
                raise DatasetError(
                    f"Dataset {dataset_name} not available in offline mode. "
                    f"Run without --offline to download it first."
                ) from e
            elif any(
                term in error_msg for term in ["connection", "network", "timeout"]
            ):
                if offline:
                    raise DatasetError(
                        f"Dataset {dataset_name} not cached locally. "
                        f"Run without --offline to download it first."
                    ) from e
                else:
                    raise DatasetNetworkError(
                        f"Network error accessing dataset {dataset_name}: {e}"
                    ) from e
            else:
                raise DatasetError(f"Failed to load dataset {dataset_name}: {e}") from e

        return dataset

    def get_instances(
        self,
        dataset_name: str,
        count: int | None = None,
        subset_pattern: str | None = None,
        random_seed: int | None = None,
        instances: list[str] | None = None,
        sample_percent: float | None = None,
        use_regex: bool = False,
        offline: bool = False
    ) -> list[dict[str, Any]]:
        """Get instances from dataset with optional filtering."""
        # Validate all inputs first
        _validate_numeric_params(count, sample_percent, random_seed)

        if instances:
            instances = _validate_instance_ids(instances)

        compiled_pattern = None
        if subset_pattern:
            if use_regex:
                compiled_pattern = _validate_regex_pattern_cached(subset_pattern)
            elif len(subset_pattern) > MAX_REGEX_LENGTH:
                raise DatasetValidationError(
                    f"Pattern too long ({len(subset_pattern)} chars). "
                    f"Maximum: {MAX_REGEX_LENGTH}"
                )

        dataset = self.fetch_dataset(dataset_name, offline=offline)

        # Filter by specific instance IDs first
        if instances:
            # Convert to set for O(1) lookups
            instance_set = set(instances)
            dataset = dataset.filter(
                lambda x: x['instance_id'] in instance_set
            )

        # Apply pattern filtering
        if subset_pattern:
            if use_regex:
                assert compiled_pattern is not None  # Should be set above
                dataset = dataset.filter(
                    lambda x: compiled_pattern.match(x['instance_id']) is not None
                )
            else:
                dataset = dataset.filter(
                    lambda x: fnmatch.fnmatch(x['instance_id'], subset_pattern)
                )

        # Handle percentage sampling
        if sample_percent:
            count = int(len(dataset) * sample_percent / 100)
            if count == 0:
                count = 1  # At least one instance

        # Apply random sampling if count specified
        if count and count < len(dataset):
            if random_seed is not None:
                random.seed(random_seed)
            indices = random.sample(range(len(dataset)), count)
            dataset = dataset.select(indices)
        elif count and count >= len(dataset):
            logger.warning(
                "Requested %d instances but dataset only has %d",
                count, len(dataset)
            )

        # Convert to list of dicts
        result_instances = []
        for item in dataset:
            result_instances.append({
                'instance_id': item['instance_id'],
                # Handle different field names
                'patch': item.get('patch', item.get('diff', '')),
                # Include other metadata if needed
                'repo': item.get('repo', ''),
                'base_commit': item.get('base_commit', ''),
                'problem_statement': item.get('problem_statement', '')
            })

        return result_instances

    def estimate_memory_usage(
        self, dataset_name: str, count: int | None = None
    ) -> dict[str, float]:
        """Estimate memory usage in MB for dataset operations."""
        try:
            dataset_info = self.get_dataset_info(dataset_name)
        except Exception:
            # If we can't get info, use conservative estimates
            dataset_info = {
                'total_instances': count or 300,  # Default to lite size
                'dataset_size_mb': 5.0  # Conservative estimate
            }

        # Estimate instance count
        instance_count = count or dataset_info['total_instances']

        # Estimate memory usage (roughly 1-2KB per instance)
        estimated_mb = (instance_count * 1.5) / 1024  # 1.5KB per instance average
        download_mb = dataset_info.get('dataset_size_mb', estimated_mb)

        return {
            'instances': instance_count,
            'estimated_ram_mb': estimated_mb,
            'download_size_mb': download_mb,
            'total_mb': estimated_mb + download_mb
        }

    def check_memory_requirements(
        self, dataset_name: str, count: int | None = None
    ) -> tuple[bool, str]:
        """Check if system has enough memory and provide warnings."""
        try:
            import psutil
            available_ram_mb = psutil.virtual_memory().available / (1024 * 1024)
        except ImportError:
            # If psutil not available, assume we have enough memory
            return True, ""

        usage = self.estimate_memory_usage(dataset_name, count)

        if usage['total_mb'] > available_ram_mb * 0.8:  # Use max 80% of available RAM
            warning = (
                f"⚠️  High memory usage warning:\n"
                f"   Estimated usage: {usage['total_mb']:.1f} MB\n"
                f"   Available RAM: {available_ram_mb:.1f} MB\n"
                f"   Consider using --count to limit instances"
            )
            return False, warning
        elif usage['total_mb'] > 100:  # Warn for large datasets
            info = (
                f"📊 Memory usage: ~{usage['total_mb']:.1f} MB "
                f"({usage['instances']} instances)"
            )
            return True, info
        else:
            return True, ""

    def get_instances_streaming(
        self,
        dataset_name: str,
        batch_size: int = 100,
        count: int | None = None,
        subset_pattern: str | None = None,
        random_seed: int | None = None,
        instances: list[str] | None = None,
        sample_percent: float | None = None,
        use_regex: bool = False,
        offline: bool = False
    ) -> Iterator[list[dict[str, Any]]]:
        """Stream instances in batches to avoid memory issues."""
        # Validate inputs first
        _validate_numeric_params(count, sample_percent, random_seed)

        if instances:
            instances = _validate_instance_ids(instances)

        compiled_pattern = None
        if subset_pattern:
            if use_regex:
                compiled_pattern = _validate_regex_pattern_cached(subset_pattern)
            elif len(subset_pattern) > MAX_REGEX_LENGTH:
                raise DatasetValidationError(
                    f"Pattern too long ({len(subset_pattern)} chars). "
                    f"Maximum: {MAX_REGEX_LENGTH}"
                )

        dataset = self.fetch_dataset(dataset_name, offline=offline)

        # Apply filters that don't require full dataset loading
        if instances:
            # Convert to set for O(1) lookups
            instance_set = set(instances)
            dataset = dataset.filter(
                lambda x: x['instance_id'] in instance_set
            )

        # Apply pattern filtering
        if subset_pattern:
            if use_regex:
                assert compiled_pattern is not None
                dataset = dataset.filter(
                    lambda x: compiled_pattern.match(x['instance_id']) is not None
                )
            else:
                dataset = dataset.filter(
                    lambda x: fnmatch.fnmatch(x['instance_id'], subset_pattern)
                )

        # Handle percentage sampling
        if sample_percent:
            count = int(len(dataset) * sample_percent / 100)
            if count == 0:
                count = 1

        # Apply random sampling if count specified
        if count and count < len(dataset):
            if random_seed is not None:
                random.seed(random_seed)
            indices = random.sample(range(len(dataset)), count)
            dataset = dataset.select(indices)

        # Stream in batches
        total_size = len(dataset)
        for i in range(0, total_size, batch_size):
            end_idx = min(i + batch_size, total_size)
            batch = dataset.select(range(i, end_idx))

            # Convert batch to our format
            result_batch = []
            for item in batch:
                result_batch.append({
                    'instance_id': item['instance_id'],
                    'patch': item.get('patch', item.get('diff', '')),
                    'repo': item.get('repo', ''),
                    'base_commit': item.get('base_commit', ''),
                    'problem_statement': item.get('problem_statement', '')
                })

            yield result_batch

    def save_as_jsonl(self, instances: list[dict[str, Any]], output_path: Path) -> None:
        """Save instances as JSONL file for compatibility."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for instance in instances:
                # Only save instance_id and patch for compatibility
                json.dump({
                    'instance_id': instance['instance_id'],
                    'patch': instance['patch']
                }, f)
                f.write('\n')

    def save_streaming_as_jsonl(
        self,
        instance_stream: Iterator[list[dict[str, Any]]],
        output_path: Path
    ) -> int:
        """Save streaming instances as JSONL file, returns total count."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        total_count = 0

        with open(output_path, 'w') as f:
            for batch in instance_stream:
                for instance in batch:
                    json.dump({
                        'instance_id': instance['instance_id'],
                        'patch': instance['patch']
                    }, f)
                    f.write('\n')
                    total_count += 1

        return total_count

    def cleanup_temp_files(self) -> None:
        """Clean up temporary files with atomic operations and proper locking."""
        temp_dir = self.cache_dir.parent / "temp"
        lock_file = temp_dir.parent / ".temp_cleanup.lock"

        # Ensure parent directory exists
        temp_dir.parent.mkdir(exist_ok=True)

        try:
            # Use file locking to prevent race conditions
            with open(lock_file, 'w') as lock_fd:
                try:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    # Another process is cleaning up, skip
                    return

                if temp_dir.exists():
                    # Move to a temporary name first (atomic operation)
                    cleanup_target = temp_dir.with_name(
                        f"{temp_dir.name}.cleanup.{os.getpid()}"
                    )
                    try:
                        temp_dir.rename(cleanup_target)
                        try:
                            shutil.rmtree(cleanup_target)
                        except (OSError, FileNotFoundError):
                            # Directory was already cleaned
                            pass
                    except (OSError, FileNotFoundError):
                        # Another process already cleaned up
                        if cleanup_target.exists():
                            shutil.rmtree(cleanup_target, ignore_errors=True)

                # Recreate the temp directory
                temp_dir.mkdir(exist_ok=True)

        finally:
            # Clean up lock file
            try:
                lock_file.unlink(missing_ok=True)
            except (OSError, FileNotFoundError):
                pass

    def get_dataset_info(self, dataset_name: str) -> dict[str, Any]:
        """Get information about a dataset without loading it."""
        if dataset_name not in self.DATASET_MAPPING:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        try:
            from datasets import load_dataset_builder  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "HuggingFace datasets library is required for dataset info. "
                "Install with: pip install datasets"
            ) from e

        try:
            builder = load_dataset_builder(self.DATASET_MAPPING[dataset_name])
            info = builder.info

            return {
                'name': dataset_name,
                'total_instances': info.splits['test'].num_examples,
                'download_size_mb': info.download_size / 1024 / 1024,
                'dataset_size_mb': info.dataset_size / 1024 / 1024,
                'description': info.description
            }
        except Exception as e:
            error_msg = str(e).lower()
            if "401" in error_msg or "unauthorized" in error_msg:
                raise DatasetAuthenticationError(
                    f"Authentication required for dataset {dataset_name}"
                ) from e
            elif any(
                term in error_msg for term in ["connection", "network", "timeout"]
            ):
                raise DatasetNetworkError(
                    f"Network error accessing dataset {dataset_name}: {e}"
                ) from e
            else:
                raise DatasetError(
                    f"Failed to get info for dataset {dataset_name}: {e}"
                ) from e


def get_hf_token() -> str | None:
    """Get HuggingFace token from environment."""
    return os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')


def configure_hf_auth() -> bool:
    """Configure HuggingFace authentication if token available."""
    token = get_hf_token()
    if token:
        try:
            from huggingface_hub import login  # type: ignore[import-not-found]
            login(token=token, add_to_git_credential=False)
            return True
        except ImportError:
            # huggingface_hub not available, skip auth
            pass
    return False
