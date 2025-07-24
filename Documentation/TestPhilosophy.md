# SWE-bench Runner Test Philosophy

## Core Principle: Test for Real Bugs, Not Coverage Metrics

We intentionally maintain ~70% test coverage instead of chasing arbitrary metrics. Every test must answer: **"What specific bug does this prevent?"**

## What We Test

### 1. Security Vulnerabilities ✅
```python
def test_prevents_regex_dos_attacks(self):
    """Prevent catastrophic backtracking that could hang the system."""
    dangerous_patterns = ["(a+)+$", "(.*)*"]  # Real ReDoS vulnerabilities
```
**Why**: These patterns can cause exponential time complexity, hanging user systems.

### 2. Critical User Errors ✅
```python
def test_auth_error_explains_token_setup(self):
    """Ensure auth errors include complete setup instructions."""
    assert "https://huggingface.co/settings/tokens" in msg
```
**Why**: Users need actionable error messages to solve problems quickly.

### 3. Performance Optimizations ✅
```python
def test_uses_set_for_instance_filtering(self):
    """Verify O(1) lookups instead of O(n) searches."""
```
**Why**: The difference between O(1) and O(n) matters with thousands of instances.

### 4. Core Functionality ✅
```python
def test_count_parameter_limits_results(self):
    """Ensure --count actually limits instances returned."""
```
**Why**: If this breaks, users get unexpected behavior.

## What We DON'T Test

### 1. Python Standard Library ❌
```python
# BAD: Testing os.environ.get()
def test_get_token_from_env(self):
    with patch.dict('os.environ', {'TOKEN': 'test'}):
        assert get_token() == 'test'
```
**Why**: We trust Python's stdlib works correctly.

### 2. Obvious Happy Paths ❌
```python
# BAD: Testing that valid input works
def test_valid_regex_patterns_compile(self):
    assert _validate_regex_pattern("django__.*") is not None
```
**Why**: If valid patterns don't work, we have bigger problems.

### 3. Mock Theater ❌
```python
# BAD: 40 lines of mock setup for 2 lines of test
def test_complex_scenario(self):
    mock1 = Mock()
    mock2 = Mock()
    # ... 30 more lines of setup
    assert mock1.called
```
**Why**: This tests the mocks, not our code.

### 4. Trivial Getters/Setters ❌
```python
# BAD: Testing basic attribute access
def test_cache_dir_property(self):
    assert manager.cache_dir == Path(temp) / "datasets"
```
**Why**: This is testing Python's basic functionality.

## Coverage Philosophy

- **85% coverage with test theater** < **70% coverage with real tests**
- Coverage is a tool, not a goal
- Missing coverage is OK if it's for unlikely edge cases
- Each test should prevent a specific, realistic bug

## Good Test Checklist

Before writing a test, ask:
1. ✅ What specific bug will this catch?
2. ✅ Is this testing our logic or Python/libraries?
3. ✅ Would this bug realistically impact users?
4. ✅ Is the test readable and maintainable?
5. ✅ Does it run fast (no unnecessary I/O)?

If you answer "no" to any of these, reconsider the test.

## Examples

### Good Test ✅
```python
def test_prevents_path_traversal(self):
    """Security: Prevent reading /etc/passwd via instance IDs."""
    with pytest.raises(ValidationError):
        validate_instance_id("../../../etc/passwd")
```
- Prevents real security vulnerability
- Tests our validation logic
- Clear what it's protecting against

### Bad Test ❌
```python
def test_empty_list_returns_empty(self):
    """Test that empty list validation returns empty list."""
    assert validate_instance_ids([]) == []
```
- No real bug prevented
- Tests obvious Python behavior
- Exists only for coverage

## Test File Organization

- `test_<module>_security.py` - Security vulnerability tests
- `test_<module>_errors.py` - Error handling and messages
- `test_<module>.py` - Core functionality tests

Keep test files small and focused. 200 lines of good tests > 1000 lines of theater.

## Maintenance

When adding new tests:
1. Write the test name as a question: "What happens if..."
2. If the answer is "Python breaks", don't write the test
3. If the answer is "users lose data/security/time", write the test

When reviewing tests:
1. Can you understand what bug it prevents?
2. Is the test testing the right thing?
3. Would you want this test to catch a regression?

Remember: **We're not testing to make coverage tools happy. We're testing to keep users happy.**
