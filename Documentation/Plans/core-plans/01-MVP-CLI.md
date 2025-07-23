# Work Plan: MVP-CLI - Basic CLI Structure

**Task ID**: MVP-CLI
**Status**: Completed

## Problem Statement

We need a basic CLI structure that allows users to run SWE-bench evaluations with a simple command. This is the foundation for all other functionality - without a working CLI entry point, nothing else can function.

## Proposed Solution

Create a minimal Click-based CLI application with:
1. Main command group (`swebench`)
2. Primary `run` command with `--patches` flag
3. Basic project structure with entry point
4. Error handling for missing arguments
5. Version display functionality

### Technical Approach
- Use Click 8.1.x for CLI framework (battle-tested, great UX)
- Create minimal project structure with pyproject.toml
- Set up entry point for `swebench` command
- Implement stub `run` command that validates inputs
- Use sys.exit(0) for success (proper exit codes in Phase 2)

### Expected CLI Structure
```python
# cli.py
import sys
import click
from . import __version__

@click.group()
@click.version_option(version=__version__)
def cli():
    """SWE-bench evaluation runner."""
    pass

@cli.command()
@click.option('--patches', required=True, type=click.Path(exists=True),
              help='Path to JSONL file containing patches')
def run(patches):
    """Run SWE-bench evaluation."""
    click.echo(f"Would run evaluation with {patches}")
    sys.exit(0)

if __name__ == '__main__':
    cli()
```

### Interface Design
```bash
# Primary usage
swebench run --patches predictions.jsonl

# Version check
swebench --version

# Help
swebench --help
swebench run --help
```

## Automated Test Plan

1. **Unit Tests** (`tests/test_cli.py`):
   ```python
   from click.testing import CliRunner
   from swebench_runner.cli import cli

   def test_version():
       runner = CliRunner()
       result = runner.invoke(cli, ['--version'])
       assert result.exit_code == 0
       assert '0.1.0' in result.output
   ```
   - Test CLI initialization
   - Test --version flag
   - Test --help output
   - Test missing --patches argument error
   - Test invalid patches file error
   - Test empty file error
   - Test successful run returns exit code 0

2. **Integration Tests**:
   - Test full command execution (stub)
   - Test entry point installation

## Components Involved

- `src/swebench_runner/__init__.py` - Package initialization and version
- `src/swebench_runner/__main__.py` - Entry point for python -m execution
- `src/swebench_runner/cli.py` - Main CLI logic
- `pyproject.toml` - Project configuration and dependencies
- `tests/test_cli.py` - CLI tests

## Dependencies

- **External**:
  - Click 8.1.x (CLI framework)
  - pytest >=7.0 (testing)
  - mypy >=1.0 (type checking)
  - ruff >=0.1 (linting)
- **Internal**: None (first work plan)
- **Knowledge**: Click documentation, Python packaging, Click testing with CliRunner

## Implementation Checklist

- [ ] Create project structure
  - [ ] Verify .gitignore includes Python patterns
  - [ ] Create src/swebench_runner directory
  - [ ] Create tests directory
  - [ ] Create pyproject.toml with project metadata
  - [ ] Set package name as "swebench-runner" (PyPI name)
  - [ ] Set Python requirement to >=3.8
  - [ ] Add development dependencies (pytest, mypy, ruff)
  - [ ] Add build-system configuration
  - [ ] Ensure wheel size will be <1 MB
- [ ] Set up package basics
  - [ ] Create `__init__.py` with `__version__ = "0.1.0"`
  - [ ] Create `__main__.py` for `python -m swebench_runner` execution
  - [ ] Configure entry point: `swebench = "swebench_runner.cli:cli"`
- [ ] Implement CLI foundation
  - [ ] Create cli.py with Click app
  - [ ] Add main command group using @click.group()
  - [ ] Add @click.version_option(version=__version__)
  - [ ] Add run command with --patches flag (required)
  - [ ] Use click.Path(exists=True) for validation
  - [ ] Implement basic sys.exit(0) for success
  - [ ] Structure to allow future subcommands (validate, clean)
- [ ] Add error handling
  - [ ] Handle missing --patches with helpful message
  - [ ] Handle non-existent patches file
  - [ ] Check file is not empty
  - [ ] Use consistent error format: "Error: {message}"
- [ ] Write tests
  - [ ] Set up test_cli.py with Click's CliRunner
  - [ ] Test successful run (exit code 0)
  - [ ] Test --version output
  - [ ] Test --help output
  - [ ] Test missing --patches error
  - [ ] Test non-existent file error
  - [ ] Test empty file handling
  - [ ] Verify package size <1 MB
- [ ] Documentation
  - [ ] Add docstrings to all functions
  - [ ] Update README with basic usage
  - [ ] Document package vs module naming convention

## Verification Steps

1. **Manual Testing**:
   ```bash
   # Install in development mode
   pip install -e .

   # Test commands
   swebench --version  # Should show version
   swebench --help     # Should show help
   swebench run --help # Should show run command help
   swebench run        # Should error about missing --patches
   swebench run --patches nonexistent.jsonl  # Should error gracefully
   ```

2. **Automated Testing**:
   ```bash
   pytest tests/test_cli.py -v
   ```

3. **Success Criteria**:
   - [ ] `swebench` command is available after pip install
   - [ ] --version shows correct version
   - [ ] --help provides useful information
   - [ ] Missing arguments produce helpful errors
   - [ ] All tests pass

## Decision Authority

**Can decide independently**:
- Project structure details
- Test organization
- Error message wording
- Code style (following Python conventions)

**Need user input**:
- Additional CLI flags beyond --patches
- Command naming if different from 'run'
- Version numbering scheme

## Questions/Uncertainties

**Blocking**:
- None - version will be 0.1.0, using pyproject.toml only

**Non-blocking**:
- Exact help text wording (can refine later)
- Whether to add --verbose flag in MVP (assuming no for simplicity)
- Should we stub future commands (validate, clean) or add later? (assuming add later)

## Acceptable Tradeoffs

- **No configuration file support** - just CLI flags for MVP
- **No colored output** - can add Rich later
- **Minimal validation** - just check file exists and not empty
- **No progress indication** - will add in later phases
- **No subcommands yet** - only `run` command, others added in Phase 2+
- **Simple version scheme** - starting at 0.1.0, not 0.4.0 as shown in examples
- **Basic exit codes** - only 0 (success) or 1 (error), specific codes in Phase 2
- **No JSON validation** - check file exists/not empty, parse validation in Phase 2

## Notes

This is the foundation work plan. Every other feature builds on having a working CLI structure. Keeping it minimal ensures we can iterate quickly and get to a working demo as fast as possible.

### Naming Convention Decision
- **Package name** (PyPI): `swebench-runner` (with hyphen)
- **Module name** (Python): `swebench_runner` (with underscore)
- **Command name**: `swebench` (no suffix)
- This follows Python packaging conventions where PyPI names use hyphens but Python module names use underscores.

### Documentation References
- [Click Command Groups](https://click.palletsprojects.com/commands/)
- [Click Options vs Arguments](https://click.palletsprojects.com/options/)
- [Testing Click Applications](https://click.palletsprojects.com/testing/)
- [pyproject.toml specification](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [Python Entry Points](https://packaging.python.org/en/latest/specifications/entry-points/)

### Basic pyproject.toml Structure
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "swebench-runner"
version = "0.1.0"
description = "A delightfully simple CLI for running SWE-bench evaluations"
requires-python = ">=3.8"
dependencies = [
    "click>=8.1.0,<9.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "mypy>=1.0",
    "ruff>=0.1",
]

[project.scripts]
swebench = "swebench_runner.cli:cli"
```
