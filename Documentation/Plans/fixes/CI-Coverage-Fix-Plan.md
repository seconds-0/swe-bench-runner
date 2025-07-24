# CI Coverage Fix Plan - Dataset Auto-Fetch Feature

**Task ID**: FIX-Coverage-Dataset-AutoFetch
**Status**: Not Started
**Priority**: Critical - Blocking PR Merge
**Target**: Increase test coverage from 77% to 85%+

## Problem Statement

The Dataset Auto-Fetch implementation has reduced overall test coverage to 77%, below the required 85% threshold. The main culprit is `datasets.py` with only 56% coverage (128 uncovered lines out of 290).

### Current Coverage Breakdown
```
src/swebench_runner/datasets.py    290    128    56%   (Missing 128 lines)
src/swebench_runner/cli.py         252     82    67%   (Missing 82 lines)
Overall:                           1121    261    77%
```

## Detailed Analysis of Uncovered Code

### 1. DatasetManager - Core Functions (Priority: HIGH)
**Uncovered Lines**: 131, 134, 153, 171-177, 183-184, 190, 193, 201, 207, 215, 222, 240, 244, 250, 254

These lines cover:
- Regex validation edge cases
- Instance ID validation edge cases
- Error handling paths in fetch_dataset
- Network error scenarios
- Authentication error paths

### 2. Memory & Resource Management (Priority: HIGH)
**Uncovered Lines**: 433-449, 462, 467-484

These lines cover:
- `estimate_memory_usage()` function
- `check_memory_requirements()` function
- psutil import error handling
- Memory warning generation

### 3. Streaming Support (Priority: MEDIUM)
**Uncovered Lines**: 500-565

These lines cover:
- `get_instances_streaming()` function
- `save_streaming_as_jsonl()` function
- Batch processing logic

### 4. Authentication & Config (Priority: MEDIUM)
**Uncovered Lines**: 689-698

These lines cover:
- `configure_hf_auth()` function
- HuggingFace login logic
- Token validation

### 5. Dataset Info & Error Paths (Priority: LOW)
**Uncovered Lines**: 664-677, 684

These lines cover:
- Error handling in `get_dataset_info()`
- Missing dataset builder scenarios

## Test Implementation Plan

### Phase 1.1: Core Function Coverage (Est: 1 hour)

#### Test: Regex Validation Edge Cases
```python
def test_regex_validation_edge_cases():
    """Test regex validation with various edge cases."""
    # Test maximum length validation
    long_pattern = "a" * 1001
    with pytest.raises(RegexValidationError):
        _validate_regex_pattern(long_pattern)

    # Test dangerous patterns
    dangerous_patterns = [
        r"(a+)+$",  # Exponential backtracking
        r"(.*)*",   # Nested quantifiers
        r"(?=.*)+", # Lookahead with quantifier
    ]

    for pattern in dangerous_patterns:
        with pytest.raises(RegexValidationError):
            _validate_regex_pattern(pattern)

    # Test timeout protection
    slow_pattern = r"(a+)+" + "b"
    with pytest.raises(RegexValidationError):
        _validate_regex_pattern(slow_pattern)
```

#### Test: Instance ID Validation Edge Cases
```python
def test_instance_id_validation_edge_cases():
    """Test instance ID validation with edge cases."""
    # Test empty list
    assert _validate_instance_ids([]) == []

    # Test long instance IDs
    long_id = "a" * 201 + "__test"
    with pytest.raises(InstanceValidationError):
        _validate_instance_ids([long_id])

    # Test path traversal attempts
    malicious_ids = [
        "../../../etc/passwd",
        "test__../../../secret",
        "test\\..\\..\\windows",
    ]

    for bad_id in malicious_ids:
        with pytest.raises(InstanceValidationError):
            _validate_instance_ids([bad_id])
```

#### Test: Network Error Scenarios
```python
@patch('datasets.load_dataset')
def test_fetch_dataset_network_errors(mock_load_dataset):
    """Test dataset fetching with network errors."""
    manager = DatasetManager(tmp_path)

    # Test connection error
    mock_load_dataset.side_effect = Exception("Connection timeout")
    with pytest.raises(DatasetNetworkError):
        manager.fetch_dataset("lite")

    # Test authentication error
    mock_load_dataset.side_effect = Exception("401 Unauthorized")
    with pytest.raises(DatasetAuthenticationError):
        manager.fetch_dataset("lite")

    # Test offline mode with missing cache
    mock_load_dataset.side_effect = Exception("Dataset not found locally")
    with pytest.raises(DatasetError):
        manager.fetch_dataset("lite", offline=True)
```

### Phase 1.2: Memory Management Coverage (Est: 45 min)

#### Test: Memory Estimation
```python
@patch('swebench_runner.datasets.DatasetManager.get_dataset_info')
def test_memory_estimation(mock_get_info):
    """Test memory usage estimation."""
    manager = DatasetManager(tmp_path)

    # Test with dataset info available
    mock_get_info.return_value = {
        'total_instances': 300,
        'dataset_size_mb': 1.2
    }

    usage = manager.estimate_memory_usage("lite")
    assert usage['instances'] == 300
    assert usage['estimated_ram_mb'] > 0
    assert usage['download_size_mb'] == 1.2

    # Test with count parameter
    usage = manager.estimate_memory_usage("lite", count=50)
    assert usage['instances'] == 50

    # Test when dataset info fails
    mock_get_info.side_effect = Exception("Network error")
    usage = manager.estimate_memory_usage("lite", count=100)
    assert usage['instances'] == 100
```

#### Test: Memory Requirements Check
```python
@patch('psutil.virtual_memory')
def test_check_memory_requirements(mock_memory):
    """Test memory requirements checking."""
    manager = DatasetManager(tmp_path)

    # Test sufficient memory
    mock_memory.return_value.available = 8 * 1024 * 1024 * 1024  # 8GB
    can_proceed, msg = manager.check_memory_requirements("lite")
    assert can_proceed is True

    # Test insufficient memory
    mock_memory.return_value.available = 100 * 1024 * 1024  # 100MB
    can_proceed, msg = manager.check_memory_requirements("full")
    assert can_proceed is False
    assert "High memory usage warning" in msg

    # Test without psutil
    with patch('swebench_runner.datasets.psutil', side_effect=ImportError):
        can_proceed, msg = manager.check_memory_requirements("lite")
        assert can_proceed is True  # Should assume enough memory
```

### Phase 1.3: Streaming Support Coverage (Est: 45 min)

#### Test: Streaming Instances
```python
@patch('datasets.load_dataset')
def test_get_instances_streaming(mock_load_dataset):
    """Test streaming instance retrieval."""
    manager = DatasetManager(tmp_path)

    # Create mock dataset
    mock_instances = [
        {'instance_id': f'test_{i}', 'patch': f'patch_{i}'}
        for i in range(250)
    ]
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=250)
    mock_dataset.select = Mock(side_effect=lambda indices:
        [mock_instances[i] for i in indices]
    )
    mock_load_dataset.return_value = mock_dataset

    # Test streaming with batches
    batches = list(manager.get_instances_streaming("lite", batch_size=100))
    assert len(batches) == 3  # 250 instances / 100 batch size
    assert len(batches[0]) == 100
    assert len(batches[1]) == 100
    assert len(batches[2]) == 50

    # Test with filters
    mock_dataset.filter = Mock(return_value=mock_dataset)
    batches = list(manager.get_instances_streaming(
        "lite",
        batch_size=50,
        subset_pattern="test_1*"
    ))
    assert mock_dataset.filter.called
```

#### Test: Streaming Save
```python
def test_save_streaming_as_jsonl():
    """Test saving streaming instances to JSONL."""
    manager = DatasetManager(tmp_path)
    output_file = tmp_path / "output.jsonl"

    # Create streaming generator
    def instance_generator():
        for i in range(3):
            yield [
                {'instance_id': f'test_{j}', 'patch': f'patch_{j}'}
                for j in range(i*10, (i+1)*10)
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
    first_item = json.loads(lines[0])
    assert first_item['instance_id'] == 'test_0'
```

### Phase 1.4: Authentication Coverage (Est: 30 min)

#### Test: HuggingFace Authentication
```python
@patch('swebench_runner.datasets.get_hf_token')
@patch('huggingface_hub.login')
def test_configure_hf_auth(mock_login, mock_get_token):
    """Test HuggingFace authentication configuration."""
    # Test with token available
    mock_get_token.return_value = "test_token_123"
    result = configure_hf_auth()
    assert result is True
    mock_login.assert_called_once_with(
        token="test_token_123",
        add_to_git_credential=False
    )

    # Test without token
    mock_get_token.return_value = None
    result = configure_hf_auth()
    assert result is False

    # Test with import error
    mock_get_token.return_value = "test_token"
    with patch('swebench_runner.datasets.login', side_effect=ImportError):
        result = configure_hf_auth()
        assert result is False
```

### Phase 1.5: Error Path Coverage (Est: 30 min)

#### Test: Dataset Info Error Handling
```python
@patch('datasets.load_dataset_builder')
def test_get_dataset_info_errors(mock_builder):
    """Test dataset info with various errors."""
    manager = DatasetManager(tmp_path)

    # Test network error
    mock_builder.side_effect = Exception("Network error")
    with pytest.raises(DatasetError):
        manager.get_dataset_info("lite")

    # Test missing splits
    mock_builder.return_value.info.splits = {}
    with pytest.raises(KeyError):
        manager.get_dataset_info("lite")

    # Test invalid dataset
    with pytest.raises(ValueError):
        manager.get_dataset_info("invalid_dataset")
```

## Integration Testing Plan

### CLI Integration Tests (Est: 30 min)

#### Test: Dataset Flag Combinations
```python
def test_cli_dataset_combinations():
    """Test various dataset flag combinations."""
    runner = CliRunner()

    # Test count with dataset
    result = runner.invoke(cli, ['run', '-d', 'lite', '--count', '5'])
    assert "Loading lite dataset" in result.output

    # Test sample percentage
    result = runner.invoke(cli, ['run', '-d', 'lite', '--sample', '10%'])
    assert "Random 10.0% sample" in result.output

    # Test offline mode
    result = runner.invoke(cli, ['run', '-d', 'lite', '--offline'])
    assert "offline" in result.output.lower() or result.exit_code != 0
```

## Success Metrics

### Coverage Targets
- `datasets.py`: From 56% → 85%+ (reduce uncovered from 128 → <44 lines)
- Overall: From 77% → 85%+

### Test Count
- Add ~25-30 new test cases
- Focus on error paths and edge cases
- Ensure all public methods have tests

## Risk Mitigation

### Potential Issues
1. **Mock complexity**: HuggingFace datasets API is complex to mock
   - **Mitigation**: Use minimal mocks, focus on interface not implementation

2. **CI environment differences**: Tests may behave differently in CI
   - **Mitigation**: Test with CI=true environment variable locally

3. **Time constraints**: Full implementation may take longer than estimated
   - **Mitigation**: Prioritize high-impact tests first

## Implementation Checklist

- [ ] Create test file: `tests/test_datasets_coverage.py`
- [ ] Implement regex validation edge case tests
- [ ] Implement instance ID validation tests
- [ ] Implement network error scenario tests
- [ ] Implement memory management tests
- [ ] Implement streaming support tests
- [ ] Implement authentication tests
- [ ] Implement error path tests
- [ ] Add CLI integration tests
- [ ] Run coverage report and verify 85%+
- [ ] Run all tests on multiple Python versions
- [ ] Update existing tests if needed

## Validation

```bash
# Check coverage locally
pytest tests/ --cov=swebench_runner --cov-report=term-missing --cov-fail-under=85

# Check specific module coverage
pytest tests/test_datasets*.py --cov=swebench_runner.datasets --cov-report=term-missing

# Verify no regressions
pytest tests/ -v
```
