# FINAL-CI-GREEN-Plan

## Task ID: FINAL-CI-GREEN
**Status**: Not Started

## Executive Summary

We need to achieve 100% CI green status for the feat/dataset-auto-fetch branch TODAY. Analysis shows we have approximately:
- **51 Ruff violations** (E501, B904, S110 across multiple files)
- **13 MyPy violations** (missing return type annotations, import errors, type issues)

All issues are surgical fixes that can be completed in parallel by multiple coding agents. No architectural changes required.

## Problem Statement

The feat/dataset-auto-fetch branch is failing CI with linting and type checking violations. The Python version compatibility issue has been resolved (updated to 3.10). We need to systematically fix all remaining violations to achieve green status.

## Proposed Solution

Break down all violations into small, surgical tasks that can be executed in parallel:
1. **Ruff E501 fixes**: Line length violations (split long lines)
2. **Ruff B904 fixes**: Add exception chaining (`from err` or `from None`)
3. **Ruff S110 fixes**: Replace try-except-pass with proper logging
4. **MyPy return type annotations**: Add `-> None` to functions
5. **MyPy type issues**: Fix argument types and imports

## Categories and Task Breakdown

### Category A: Ruff E501 Line Length Violations (OpenAI Provider)
**Priority**: High (blocking CI)
**Estimated time**: 15 minutes total (5 tasks × 3 minutes each)

#### Task A1: Fix E501 in openai.py lines 302, 349
**File**: `src/swebench_runner/providers/openai.py`
**Lines**: 302, 349
**Fix**:
```python
# Line 302: Split the long line
error_message = (
    error_data.get("error", {}).get("message", "")
)

# Line 349: Split the long error message
raise ProviderResponseError(
    f"Unexpected status {response.status}: "
    f"{error_data.get('error', {}).get('message', 'Unknown error')}"
)
```
**Why**: Line length violations block CI

#### Task A2: Fix E501 in openai.py lines 353, 358, 361, 367
**File**: `src/swebench_runner/providers/openai.py`
**Lines**: 353, 358, 361, 367
**Fix**:
```python
# Line 353: Split timeout error message
raise ProviderTimeoutError(
    f"Request to {url} timed out after {self.config.timeout}s"
)

# Line 358: Split warning message
logger.warning(
    f"Connection error on attempt {attempt + 1}, "
    f"retrying in {delay}s: {e}"
)

# Line 361: Split exception types
except (
    ProviderAuthenticationError,
    ProviderTokenLimitError,
    ProviderResponseError
):

# Line 367: Split rate limit message
logger.warning(
    f"Rate limited, waiting {e.retry_after}s before retry"
)
```
**Why**: Line length violations block CI

#### Task A3: Fix E501 in openai.py lines 431, 433
**File**: `src/swebench_runner/providers/openai.py`
**Lines**: 431, 433
**Fix**:
```python
# Line 431: Split header access
self._rate_limit_info["requests_limit"] = int(
    headers["x-ratelimit-limit-requests"]
)

# Line 433: Split header access
self._rate_limit_info["requests_remaining"] = int(
    headers["x-ratelimit-remaining-requests"]
)
```
**Why**: Line length violations block CI

#### Task A4: Fix E501 in openrouter.py
**File**: `src/swebench_runner/providers/openrouter.py`
**Lines**: Check for similar violations and fix
**Fix**: Split any lines >88 characters following same patterns as A1-A3
**Why**: Line length violations block CI

#### Task A5: Fix E501 in config.py and other provider files
**File**: Check all files in `src/swebench_runner/providers/` for E501 violations
**Fix**: Split any lines >88 characters following same patterns
**Why**: Line length violations block CI

### Category B: Ruff B904 Exception Chaining Violations
**Priority**: High (blocking CI)
**Estimated time**: 10 minutes total (2 tasks × 5 minutes each)

#### Task B1: Fix B904 in openai.py line 353
**File**: `src/swebench_runner/providers/openai.py`
**Lines**: 353
**Fix**:
```python
except asyncio.TimeoutError as e:
    raise ProviderTimeoutError(
        f"Request to {url} timed out after {self.config.timeout}s"
    ) from e
```
**Why**: Exception chaining required for proper error tracking

#### Task B2: Fix B904 in other provider files
**File**: Search all provider files for similar B904 violations
**Fix**: Add `from e` or `from None` to all raise statements in except blocks
**Why**: Exception chaining required for proper error tracking

### Category C: Ruff S110 Try-Except-Pass Violations
**Priority**: Medium (blocking CI)
**Estimated time**: 10 minutes total

#### Task C1: Fix S110 violations across codebase
**File**: Search for try-except-pass patterns
**Fix**: Replace `pass` with appropriate logging:
```python
except SomeException as e:
    logger.debug(f"Expected exception: {e}")
    # or logger.warning() depending on severity
```
**Why**: Silent exceptions hide important debugging information

### Category D: MyPy Return Type Annotations
**Priority**: High (blocking CI)
**Estimated time**: 15 minutes total (5 tasks × 3 minutes each)

#### Task D1: Fix return type annotations in circuit_breaker.py
**File**: `src/swebench_runner/providers/circuit_breaker.py`
**Lines**: 88, 95, 158, 171, 207
**Fix**:
```python
def _check_state(self) -> None:

def _transition_to(self, new_state: CircuitState) -> None:

def _on_success(self) -> None:

def _on_failure(self) -> None:

def reset(self) -> None:
```
**Why**: MyPy requires return type annotations

#### Task D2: Fix return type annotations in base.py
**File**: `src/swebench_runner/providers/base.py`
**Lines**: 54, 83
**Fix**:
```python
def __post_init__(self) -> None:

def _validate_config(self) -> None:
```
**Why**: MyPy requires return type annotations

#### Task D3: Fix argument type annotation in base.py
**File**: `src/swebench_runner/providers/base.py`
**Lines**: 112
**Fix**:
```python
async def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
```
**Why**: MyPy requires type annotations for all arguments

#### Task D4: Fix argument type annotation in circuit_breaker.py
**File**: `src/swebench_runner/providers/circuit_breaker.py`
**Lines**: 110
**Fix**:
```python
async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
```
**Why**: MyPy requires type annotations for all arguments

### Category E: MyPy Type and Import Issues
**Priority**: High (blocking CI)
**Estimated time**: 15 minutes total (3 tasks × 5 minutes each)

#### Task E1: Fix keyring import issue in config.py
**File**: `src/swebench_runner/providers/config.py`
**Lines**: 10-14
**Fix**:
```python
try:
    import keyring
except ImportError:
    keyring = None  # type: ignore[assignment]
```
**Why**: Handle optional dependency properly

#### Task E2: Fix circuit breaker type issues
**File**: `src/swebench_runner/providers/circuit_breaker.py`
**Lines**: 131, 153
**Fix**:
```python
# Line 131: Handle None case
if self._last_failure_time is not None:
    time_since_failure = time.time() - self._last_failure_time

# Line 153: Fix exception type
if isinstance(self.config.expected_exception, type):
    except self.config.expected_exception:
else:
    except self.config.expected_exception:
```
**Why**: Fix type checking errors

#### Task E3: Fix registry provider argument type
**File**: `src/swebench_runner/providers/registry.py`
**Lines**: 127
**Fix**:
```python
# Add null check before provider instantiation
provider_config = config or self._configs.get(name)
if provider_config is None:
    raise ValueError(f"No configuration found for provider {name}")
provider = provider_class(provider_config)
```
**Why**: Ensure non-None config passed to provider

### Category F: Generation Files Type Issues
**Priority**: Medium (if present)
**Estimated time**: 10 minutes total

#### Task F1: Fix generation_integration.py issues
**File**: `src/swebench_runner/generation_integration.py`
**Fix**: Add any missing type annotations found in mypy output
**Why**: Complete type safety

#### Task F2: Fix other generation files
**File**: Check `src/swebench_runner/generation/` for type issues
**Fix**: Add missing type annotations
**Why**: Complete type safety

## Parallel Execution Strategy

**Agent Assignment**:
- **Agent 1**: Category A (OpenAI E501 fixes) - Tasks A1, A2, A3
- **Agent 2**: Category B + C (Exception chaining and S110) - Tasks B1, B2, C1
- **Agent 3**: Category D (Return type annotations) - Tasks D1, D2, D3, D4
- **Agent 4**: Category E (Type/import issues) - Tasks E1, E2, E3
- **Agent 5**: Category F (Generation files) - Tasks F1, F2

**Dependencies**: None - all tasks are independent and can run in parallel

## Implementation Checklist

### Phase 1: Parallel Execution (30 minutes)
- [ ] Agent 1: Complete all E501 line length fixes
- [ ] Agent 2: Complete all B904 and S110 fixes
- [ ] Agent 3: Complete all return type annotations
- [ ] Agent 4: Complete all type and import fixes
- [ ] Agent 5: Complete generation file fixes

### Phase 2: Integration Testing (10 minutes)
- [ ] Run `ruff check src/swebench_runner tests` - should pass
- [ ] Run `python3 -m mypy src/swebench_runner` - should pass
- [ ] Run `make pre-pr` - should show green checkmarks

### Phase 3: Final Verification (5 minutes)
- [ ] Commit all changes with descriptive message
- [ ] Run final `make pre-pr` to confirm 100% green
- [ ] Update this plan status to "Completed"

## Verification Steps

1. **Ruff Verification**:
   ```bash
   ruff check src/swebench_runner tests
   # Expected: No errors
   ```

2. **MyPy Verification**:
   ```bash
   python3 -m mypy src/swebench_runner
   # Expected: Success: no issues found
   ```

3. **Full CI Simulation**:
   ```bash
   make pre-pr
   # Expected: All critical checks passed!
   ```

## Success Criteria

- [ ] Zero Ruff violations in `src/swebench_runner`
- [ ] Zero MyPy errors in `src/swebench_runner`
- [ ] `make pre-pr` shows "All critical checks passed!"
- [ ] All changes maintain existing functionality
- [ ] No new errors introduced

## Decision Authority

**Engineering Manager can decide independently**:
- How to split long lines (style choices)
- Whether to use `from e` or `from None` for exception chaining
- Log levels for replaced pass statements
- Specific type annotation choices

**Requires user input**:
- Major architectural changes (not needed)
- Changes to error handling behavior (not needed)
- Performance optimization decisions (not needed)

## Acceptable Tradeoffs

- **Code style over performance**: Prioritize readability in line splits
- **Explicit over implicit**: Add type annotations even for obvious cases
- **Safe over optimal**: Use conservative exception chaining
- **Working over perfect**: Get CI green first, optimize later

## Questions/Uncertainties

### Blocking
None - all fixes are straightforward surgical changes

### Non-blocking
- **Line split style**: Using consistent parenthetical grouping
- **Log levels**: Using DEBUG for expected exceptions, WARNING for retries
- **Exception chaining**: Using `from e` for external errors, `from None` for internal logic

## Notes

- All tasks maintain backward compatibility
- No functional changes, only style and type safety fixes
- Each task is independently testable
- Total estimated time: 45 minutes with 5 parallel agents
- No new dependencies required
- All fixes follow existing codebase patterns

## Dependencies

- Python 3.10+ (already satisfied)
- ruff (already installed)
- mypy (already installed)
- No new external dependencies required

## Components Involved

- Provider system (OpenAI, OpenRouter, Circuit Breaker, Base, Config, Registry)
- Generation system (minimal fixes if needed)
- Build/CI system (verification only)
- Type checking system (MyPy configuration)

## Automated Test Plan

Each task will be verified by:
1. **Immediate verification**: Run ruff/mypy on modified file
2. **Integration verification**: Run full linting after each category
3. **Final verification**: Full `make pre-pr` simulation

No new tests required - existing functionality preserved.
