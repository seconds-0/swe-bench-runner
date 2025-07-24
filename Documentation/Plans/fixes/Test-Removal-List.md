# Specific Tests to Remove - Test Theater Cleanup

## Tests to DELETE (No Real Value)

### 1. Trivial Happy Path Tests
- `test_regex_max_length_validation` - Just tests that valid patterns work
- `test_regex_valid_patterns` - Tests obvious valid patterns
- `test_instance_id_empty_list` - Tests [] returns []
- `test_init_creates_cache_dir` - Tests basic Python mkdir
- `test_save_as_jsonl_basic` - Tests basic file writing

### 2. Testing Python Stdlib
- `test_get_hf_token_from_env` - Tests os.environ.get()
- `test_configure_hf_auth_without_token` - Tests None is falsy

### 3. Over-Mocked Tests (Testing Mocks, Not Code)
- `test_get_instances_streaming_basic` - 90% mock setup
- `test_get_instances_with_all_filters` - Complex mocks testing mocks
- `test_cleanup_temp_files_concurrent_access` - Tests fcntl, not our logic
- `test_cleanup_temp_files_permission_error` - Mocks Path.rename to fail

### 4. Redundant Validation Tests
- `test_count_validation` - We don't need to test every edge of numeric validation
- `test_random_seed_validation` - Testing int bounds is not valuable

## Tests to KEEP (High Value)

### Security Tests
- `test_regex_dangerous_patterns` ✓ - ReDoS protection
- `test_instance_id_path_traversal` ✓ - Security vulnerability
- `test_instance_id_format_validation` ✓ - But simplify to just test SQL injection chars

### Error Handling Tests
- `test_fetch_dataset_connection_error` ✓ - Network errors users will hit
- `test_fetch_dataset_authentication_error` ✓ - Common user issue
- `test_error_message_with_context` ✓ - Ensures helpful error messages

### Performance Tests
- Keep ONE test that verifies set() optimization for O(1) lookups

## Tests to SIMPLIFY

### 1. Regex Timeout Tests
- Combine `test_regex_timeout_protection` and `test_regex_timeout_environment_variable`
- Just test that catastrophic patterns are caught

### 2. Memory Tests
- `test_estimate_memory_usage_with_info` - Simplify to just test it returns reasonable numbers
- `test_check_memory_requirements_insufficient` - Just test warning message is helpful

### 3. Error Message Tests
- Combine all error message tests into one test with subtests

## New Test Structure

```python
# test_datasets_security.py (50 lines instead of 300)
class TestDatasetsSecurity:
    def test_prevents_regex_dos(self):
        """Test that catastrophic regex patterns are rejected."""
        bad_patterns = ["(a+)+", "(.*)*"]
        for pattern in bad_patterns:
            with pytest.raises(RegexValidationError):
                _validate_regex_pattern(pattern)

    def test_prevents_path_injection(self):
        """Test that path traversal attempts are blocked."""
        bad_ids = ["../../../etc/passwd", "'; DROP TABLE--"]
        for bad_id in bad_ids:
            with pytest.raises(InstanceValidationError):
                _validate_instance_ids([bad_id])

# test_datasets_errors.py (100 lines instead of 500)
class TestDatasetsErrorHandling:
    def test_network_errors_have_helpful_messages(self):
        """Test that network errors guide users to solutions."""
        # Just test the message, not elaborate mocking

    def test_auth_errors_explain_token_setup(self):
        """Test auth errors tell users how to get a token."""
        error = DatasetAuthenticationError("401")
        msg = get_helpful_error_message(error)
        assert "https://huggingface.co/settings/tokens" in msg
```

## Expected Outcome

- Remove ~30 tests (about 600 lines)
- Keep ~15 tests (about 300 lines)
- Coverage drops from 85% to ~70%
- But the 70% actually protects against real issues
- Tests run 50% faster
- Tests are actually maintainable

## Implementation Steps

1. Delete the obvious test theater tests
2. Run coverage to see what we lost
3. If we lost coverage of actual important paths, write ONE good test for that path
4. Consolidate similar tests
5. Simplify mock-heavy tests to test actual behavior, not mock behavior

## The Key Question for Each Test

**"If this test fails, what bug did we just prevent?"**

If the answer is "none" or "Python would be broken", delete it.
