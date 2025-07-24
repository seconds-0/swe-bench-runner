# Phase 2: Code Quality & Performance Improvements

**Task ID**: FIX-Phase2-CodeQuality
**Status**: Not Started
**Priority**: High - PR Review Feedback
**Target**: Address all PR review comments for production-ready code

## Context

PR #9 received a comprehensive review with the following key feedback:
1. Code quality issues (type annotations, print statements)
2. Performance considerations (ReDoS validation, filtering efficiency)
3. Edge cases in error handling and race conditions
4. Suggestions for future enhancements

## Phase 2 Implementation Plan

### Task 2.1: Code Quality Fixes (Est: 30 min)

#### 2.1.1: Fix Type Annotation Inconsistency
**File**: `src/swebench_runner/datasets.py:238`
```python
# Current (with noqa comment):
if not isinstance(sample_percent, (int, float)):  # noqa: UP038

# Fix to:
if not isinstance(sample_percent, int | float):
```
**Note**: Need to verify Python 3.9 compatibility for union syntax

#### 2.1.2: Replace print() with Logging
**File**: `src/swebench_runner/datasets.py:409-412`
```python
# Current:
print(
    f"Warning: Requested {count} instances but "
    f"dataset only has {len(dataset)}"
)

# Fix to:
import logging
logger = logging.getLogger(__name__)

logger.warning(
    "Requested %d instances but dataset only has %d",
    count, len(dataset)
)
```

#### 2.1.3: Add Logger Configuration
**File**: `src/swebench_runner/datasets.py` (top)
```python
import logging

# Configure logger
logger = logging.getLogger(__name__)
```

### Task 2.2: Performance Optimizations (Est: 45 min)

#### 2.2.1: Optimize Instance Filtering
**File**: `src/swebench_runner/datasets.py:376-426`
```python
# Current:
if instances:
    dataset = dataset.filter(
        lambda x: x['instance_id'] in instances
    )

# Optimize to:
if instances:
    # Convert to set for O(1) lookups
    instance_set = set(instances)
    dataset = dataset.filter(
        lambda x: x['instance_id'] in instance_set
    )
```

#### 2.2.2: Add Pattern Validation Caching
**File**: `src/swebench_runner/datasets.py`
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def _validate_regex_pattern_cached(pattern: str) -> Pattern[str]:
    """Cached version of regex pattern validation."""
    return _validate_regex_pattern(pattern)

# Update usage in get_instances() and get_instances_streaming()
if use_regex:
    compiled_pattern = _validate_regex_pattern_cached(subset_pattern)
```

#### 2.2.3: Make ReDoS Test Timeout Configurable
```python
# Add class constant
REGEX_TEST_TIMEOUT_MS = 10  # Default 10ms, configurable via env

# In _validate_regex_pattern():
timeout = float(os.environ.get('SWEBENCH_REGEX_TIMEOUT_MS', REGEX_TEST_TIMEOUT_MS)) / 1000
if time.time() - start_time > timeout:
    raise RegexValidationError(...)
```

### Task 2.3: Edge Case Improvements (Est: 30 min)

#### 2.3.1: Improve Atomic File Operations
**File**: `src/swebench_runner/datasets.py:622-624`
```python
# Current has potential race condition between rename and rmtree
# Improve to:
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
```

#### 2.3.2: More Granular Exception Handling
**File**: `src/swebench_runner/datasets.py:320-344`
```python
# Add specific exception checks before generic catch-all
except ConnectionError as e:
    raise DatasetNetworkError(
        f"Connection failed for dataset {dataset_name}: {e}"
    ) from e
except TimeoutError as e:
    raise DatasetNetworkError(
        f"Timeout downloading dataset {dataset_name}: {e}"
    ) from e
except PermissionError as e:
    raise DatasetAuthenticationError(
        f"Permission denied for dataset {dataset_name}: {e}"
    ) from e
except Exception as e:
    # Existing string-based detection as fallback
    ...
```

### Task 2.4: Future Enhancement Preparation (Est: 15 min)

#### 2.4.1: Add Progress Bar Support Structure
```python
# Add optional progress callback parameter
def fetch_dataset(
    self,
    dataset_name: str,
    force_download: bool = False,
    offline: bool = False,
    progress_callback: Callable[[int, int], None] | None = None
) -> Any:
    """
    Fetch dataset from HuggingFace or cache.

    Args:
        progress_callback: Optional callback(current_bytes, total_bytes)
    """
```

#### 2.4.2: Add Metadata Caching Structure
```python
# Add to DatasetManager
def _get_cached_metadata_path(self, dataset_name: str) -> Path:
    """Get path for cached dataset metadata."""
    return self.cache_dir / f"{dataset_name}_metadata.json"

def _load_cached_metadata(self, dataset_name: str) -> dict[str, Any] | None:
    """Load cached metadata if available and fresh."""
    cache_path = self._get_cached_metadata_path(dataset_name)
    if cache_path.exists():
        # Check if cache is less than 24 hours old
        if time.time() - cache_path.stat().st_mtime < 86400:
            with open(cache_path) as f:
                return json.load(f)
    return None
```

## Testing Plan

### New Tests to Add:
1. Test logging output instead of print
2. Test performance with large instance lists (1000+ items)
3. Test regex pattern caching behavior
4. Test atomic file operation interruption scenarios
5. Test specific exception types in error handling

### Update Existing Tests:
1. Mock logging instead of capturing print output
2. Add performance benchmarks for filtering operations
3. Test cache invalidation for metadata

## Success Metrics

1. **No print() statements** in library code
2. **Type annotations** consistent throughout (no noqa comments)
3. **Performance improvement** for large instance filtering (>10x faster)
4. **Zero race conditions** in file operations
5. **All tests passing** with no regressions

## Risk Mitigation

1. **Python 3.9 Compatibility**: Test union syntax on Python 3.9 before changing
2. **Logging Configuration**: Ensure logging doesn't interfere with CLI output
3. **Cache Invalidation**: Make sure pattern cache doesn't grow unbounded
4. **Backward Compatibility**: Ensure changes don't break existing API

## Implementation Checklist

- [ ] Set up logging configuration
- [ ] Fix type annotation inconsistency
- [ ] Replace print() with logger.warning()
- [ ] Optimize instance filtering with sets
- [ ] Add regex pattern caching
- [ ] Make ReDoS timeout configurable
- [ ] Improve atomic file operations
- [ ] Add granular exception handling
- [ ] Add progress callback structure
- [ ] Add metadata caching structure
- [ ] Update tests for logging
- [ ] Add performance tests
- [ ] Run full test suite
- [ ] Check Python 3.9 compatibility
- [ ] Update documentation if needed

## Validation Steps

```bash
# Run tests with coverage
pytest tests/ --cov=swebench_runner --cov-report=term-missing

# Check for print statements
grep -r "print(" src/swebench_runner/datasets.py

# Check type consistency
mypy src/swebench_runner/datasets.py

# Performance test
python -m timeit -s "instances = list(range(10000))" "set(instances)"

# Check Python 3.9 compatibility
python3.9 -m pytest tests/test_datasets.py
```
