# Simplified CLI Integration Remediation Plan

## Critical Fixes Only - No Overengineering

### Phase 1: Fix Integration Errors (4 tasks)

#### Task 1: Fix BatchProcessor Constructor
**File**: `src/swebench_runner/generation_integration.py`
**Lines**: 114-120
**Fix**:
```python
# WRONG (current):
batch_processor = BatchProcessor(
    generator=patch_generator,
    prompt_builder=prompt_builder,  # REMOVE - doesn't exist
    checkpoint_path=checkpoint_path  # WRONG parameter name
)

# CORRECT:
batch_processor = BatchProcessor(
    generator=patch_generator,
    checkpoint_dir=checkpoint_path,  # Note: checkpoint_dir not checkpoint_path
    max_concurrent=max_workers,
    progress_bar=show_progress
)
```

#### Task 2: Add Patch Validation
**File**: `src/swebench_runner/generation_integration.py`
**Lines**: 159-167 (in the loop processing results)
**Add validation**:
```python
# After line 160 where we check if result.patch exists:
if result.patch:
    # Validate the patch before including it
    validation_result = patch_validator.validate(result.patch, result.instance_id)
    if validation_result.is_valid:
        patch_data.append({
            "instance_id": result.instance_id,
            "patch": result.patch,
            "model": result.model,
            "cost": result.cost,
            "metadata": result.metadata
        })
    else:
        logger.warning(f"Invalid patch for {result.instance_id}: {validation_result.issues}")
```

#### Task 3: Create PatchFormatter
**File**: Create `src/swebench_runner/generation/patch_formatter.py`
**Complete implementation**:
```python
"""Format patches for evaluation compatibility."""
from typing import List, Dict, Any

class PatchFormatter:
    """Ensures patches are in the correct format for evaluation."""

    @staticmethod
    def format_for_evaluation(generation_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert generation format to evaluation format.

        The evaluation expects:
        - 'instance_id': str
        - 'patch': str (the actual diff)
        - No 'prediction' field (common mistake)
        """
        formatted = []
        for result in generation_results:
            # Ensure we have required fields
            if 'instance_id' not in result or 'patch' not in result:
                continue

            formatted_entry = {
                'instance_id': result['instance_id'],
                'patch': result['patch']
            }

            # Optionally include model info as metadata
            if 'model' in result:
                formatted_entry['model'] = result['model']

            formatted.append(formatted_entry)

        return formatted
```

#### Task 4: Fix Output Format
**File**: `src/swebench_runner/generation_integration.py`
**Lines**: 169-172 (before saving to JSONL)
**Changes**:
```python
# Import at top
from .patch_formatter import PatchFormatter

# Replace lines 169-172 with:
# Format patches for evaluation
formatter = PatchFormatter()
formatted_patches = formatter.format_for_evaluation(patch_data)

# Save to JSONL
with open(output_path, 'w') as f:
    for item in formatted_patches:
        f.write(json.dumps(item) + '\n')
```

### Phase 2: Core Functionality (3 tasks)

#### Task 5: Fix Progress Callback
**File**: `src/swebench_runner/generation_integration.py`
**Issue**: BatchProcessor needs a progress callback
**Solution**: Create a simple callback that updates console
```python
# Add before calling process_batch (around line 140):
def progress_callback(instance_id: str, current: int, total: int):
    if show_progress:
        self.console.print(f"[{current}/{total}] Processing {instance_id}...")

# Update process_batch call:
batch_result = await batch_processor.process_batch(
    instances=instances,
    resume_from_checkpoint=True if checkpoint_path else False,
    save_final_checkpoint=True
)
```

#### Task 6: Add Basic Timeout
**File**: `src/swebench_runner/generation_integration.py`
**Add timeout wrapper**:
```python
# At top of file
import asyncio

# In generate_patches_for_evaluation, wrap the process_batch call:
try:
    # Default 30 minute timeout for entire batch
    batch_result = await asyncio.wait_for(
        batch_processor.process_batch(
            instances=instances,
            resume_from_checkpoint=True if checkpoint_path else False,
            save_final_checkpoint=True
        ),
        timeout=1800  # 30 minutes
    )
except asyncio.TimeoutError:
    self.console.print("[red]❌ Batch processing timed out after 30 minutes[/red]")
    raise click.Abort()
```

#### Task 7: Add Environment Variable Support
**File**: `src/swebench_runner/cli.py`
**In run() function, after parameter definitions**:
```python
# Around line 180, before any processing:
# Use environment variables as defaults
if not provider and 'SWEBENCH_PROVIDER' in os.environ:
    provider = os.environ['SWEBENCH_PROVIDER']
if not model and 'SWEBENCH_MODEL' in os.environ:
    model = os.environ['SWEBENCH_MODEL']
if max_workers == 5 and 'SWEBENCH_MAX_WORKERS' in os.environ:
    max_workers = int(os.environ['SWEBENCH_MAX_WORKERS'])
```

### Phase 3: Testing (2 tasks)

#### Task 8: Create Working Integration Test
**File**: `tests/test_generation_integration.py`
**Replace the skipped test with**:
```python
@pytest.mark.asyncio
async def test_generate_patches_basic(temp_cache_dir, sample_instances):
    """Test basic patch generation flow."""
    # Use real components, not mocks
    from swebench_runner.providers import MockProvider, ProviderConfig
    from swebench_runner.generation_integration import GenerationIntegration

    # Setup
    integration = GenerationIntegration(temp_cache_dir)

    # Configure mock provider to return a valid patch
    mock_config = ProviderConfig(
        name="mock",
        model="mock-model",
        extra_params={
            "responses": {
                "1": "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-old\n+new",
                "2": "diff --git a/file2.py b/file2.py\n--- a/file2.py\n+++ b/file2.py\n@@ -1,1 +1,1 @@\n-old\n+new"
            }
        }
    )

    # Mock the provider registry to return our configured mock
    with patch('swebench_runner.generation_integration.get_registry') as mock_registry, \
         patch('swebench_runner.generation_integration.ProviderConfigManager') as mock_config_mgr:

        mock_registry.return_value.get_provider_class.return_value = MockProvider
        mock_config_mgr.return_value.load_config.return_value = mock_config

        # Generate patches
        output_path = await integration.generate_patches_for_evaluation(
            instances=sample_instances[:2],
            provider_name="mock",
            show_progress=False
        )

        # Verify output
        assert output_path.exists()
        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) == 2

        # Verify format
        for line in lines:
            data = json.loads(line)
            assert 'instance_id' in data
            assert 'patch' in data
            assert data['patch'].startswith('diff --git')
```

#### Task 9: Create CLI Test
**File**: Create `tests/test_cli_generation_simple.py`
**Simple test without Docker**:
```python
def test_generate_command_basic(tmp_path):
    """Test generate command works with mock provider."""
    from click.testing import CliRunner
    from swebench_runner.cli import cli

    runner = CliRunner()

    with patch('swebench_runner.generation_integration.GenerationIntegration') as mock_integration, \
         patch('swebench_runner.provider_utils.get_provider_for_cli') as mock_get_provider, \
         patch('swebench_runner.datasets.DatasetManager') as mock_dm:

        # Setup minimal mocks
        mock_dm.return_value.get_instances.return_value = [
            {"instance_id": "test-1", "problem_statement": "Test", "repo": "test/repo"}
        ]

        # Mock the async run to return a path
        output_file = tmp_path / "output.jsonl"
        output_file.write_text('{"instance_id": "test-1", "patch": "diff"}\n')

        async def mock_generate(*args, **kwargs):
            return output_file

        mock_integration.return_value.generate_patches_for_evaluation = mock_generate

        # Run command
        result = runner.invoke(cli, ['generate', '-i', 'test-1', '-p', 'mock'])

        assert result.exit_code == 0
        assert "Patch generated successfully" in result.output
```

## What We're NOT Doing (YAGNI):
- ❌ No adapter classes
- ❌ No complex error recovery strategies
- ❌ No configuration file support
- ❌ No fancy progress displays beyond what Rich already provides
- ❌ No separate ValidationStats class
- ❌ No complex checkpoint system (BatchProcessor already has it)

## Total Tasks: 9 (down from 24)
Each task is focused on fixing a specific broken thing, not adding new complexity.
