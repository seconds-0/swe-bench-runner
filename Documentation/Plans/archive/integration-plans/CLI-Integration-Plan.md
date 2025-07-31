# Work Plan: Main CLI Integration for Model Providers

**Task ID**: FEAT-CLI-Integration
**Status**: In Planning
**Priority**: High
**Last Updated**: 2025-01-25

## Problem Statement

The model provider system is complete with all core components (providers, generation engine, CLI commands) but it's not integrated with the main SWE-bench runner evaluation flow. The `swebench run` command needs to be enhanced to use the model providers for patch generation when no patches are provided.

Currently:
- `swebench run` requires pre-generated patches (JSONL file)
- The `generate` command is a demo placeholder that doesn't actually generate real patches
- No connection between the generation engine and the evaluation flow

## Proposed Solution

### 1. Enhanced Run Command
Modify the `run` command to support model-based patch generation:

```bash
# Current usage (unchanged)
swebench run --dataset lite --count 5          # Error: requires patches
swebench run --patches my_patches.jsonl         # Works: uses existing patches

# New usage (with providers)
swebench run --dataset lite --count 5 --provider openai    # Generate & evaluate
swebench run --dataset lite --provider openai --model gpt-4 # Specific model
swebench run --dataset lite --generate-only                 # Just generate, no eval
```

### 2. Integration Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLI (run command)                     │
├─────────────────────────────────────────────────────────┤
│  1. Load dataset instances                               │
│  2. Check if patches provided                            │
│  3. If not: Generate patches using provider              │
│  4. Save patches to temp/output location                 │
│  5. Run evaluation with generated patches                │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│               Patch Generation Pipeline                  │
├─────────────────────────────────────────────────────────┤
│  • Load full instance data (with code context)           │
│  • Use PromptBuilder to create prompts                   │
│  • Generate via configured provider                      │
│  • Parse and validate patches                            │
│  • Handle retries and errors                             │
│  • Track costs and performance                           │
└─────────────────────────────────────────────────────────┘
```

### 3. Key Components to Modify

#### 3.1 CLI Command Enhancement (cli.py)
```python
@cli.command()
# Add provider options
@click.option('--provider', '-p', help='Model provider for patch generation')
@click.option('--model', '-m', help='Specific model to use')
@click.option('--generate-only', is_flag=True, help='Only generate patches, skip evaluation')
@click.option('--generation-output', type=Path, help='Save generated patches to file')
@click.option('--max-workers', default=5, help='Concurrent patch generations')
@click.option('--checkpoint', type=Path, help='Resume from generation checkpoint')
def run(...existing params..., provider, model, generate_only, generation_output, max_workers, checkpoint):
    # Enhanced logic to handle generation
```

#### 3.2 Generation Integration Module
Create `src/swebench_runner/generation_integration.py`:
```python
class GenerationIntegration:
    """Integrates patch generation with evaluation pipeline."""

    async def generate_patches_for_evaluation(
        self,
        instances: list[dict[str, Any]],
        provider_name: str,
        model: Optional[str] = None,
        output_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        max_workers: int = 5
    ) -> Path:
        """Generate patches for instances and return path to JSONL."""
        # 1. Get provider
        # 2. Load full instance data with code context
        # 3. Use BatchProcessor for concurrent generation
        # 4. Save as JSONL in evaluation format
        # 5. Return path for evaluation
```

#### 3.3 Instance Data Enhancement
The current dataset loading only provides basic fields. We need to enhance it to include:
- Repository structure/files
- Test files content
- Related code context
- Build/test commands

This requires integration with the SWE-bench data loading:
```python
class EnhancedDatasetManager(DatasetManager):
    """Enhanced dataset manager with full instance data."""

    def get_instances_with_context(
        self,
        dataset_name: str,
        instances: Optional[list[str]] = None,
        **kwargs
    ) -> list[dict[str, Any]]:
        """Load instances with full code context for generation."""
        # Load base instances
        base_instances = self.get_instances(dataset_name, instances, **kwargs)

        # Enhance with code context
        # This might require downloading additional data or
        # using the SWE-bench API to get full instance details
        return self._enhance_with_context(base_instances)
```

### 4. Implementation Details

#### 4.1 Progress and Status Display
```python
def display_generation_progress(stats: BatchStats, current: int, total: int):
    """Rich display of generation progress."""
    # Show:
    # - Progress bar with ETA
    # - Success/failure counts
    # - Current cost estimate
    # - Current instance being processed
```

#### 4.2 Error Handling
```python
class GenerationFailureHandler:
    """Handle generation failures gracefully."""

    def handle_batch_failure(self, results: BatchResult) -> bool:
        """Decide whether to continue with partial results."""
        if results.stats.success_rate > 0.5:
            # Offer to continue with successful patches
            return click.confirm(
                f"⚠️  Generated {len(results.successful)}/{results.stats.total_instances} patches. "
                "Continue with partial results?"
            )
        return False
```

#### 4.3 Cost Management
```python
class CostEstimator:
    """Estimate and track generation costs."""

    def estimate_batch_cost(
        self,
        instances: list[dict],
        provider: ModelProvider
    ) -> tuple[float, float]:
        """Estimate min/max cost for batch generation."""
        # Calculate based on:
        # - Average instance size
        # - Provider pricing
        # - Retry assumptions

    def should_warn_cost(self, estimated_cost: float) -> bool:
        """Check if cost warrants a warning."""
        return estimated_cost > 10.0  # $10 threshold
```

### 5. Configuration and Defaults

#### 5.1 Environment Variables
```bash
SWEBENCH_PROVIDER=openai              # Default provider
SWEBENCH_MODEL=gpt-4-turbo-preview   # Default model
SWEBENCH_MAX_WORKERS=5               # Concurrent generations
SWEBENCH_GENERATION_TIMEOUT=300      # 5 minute timeout per instance
```

#### 5.2 CLI Configuration File
Support `.swebench.yaml` for project defaults:
```yaml
generation:
  provider: openai
  model: gpt-4-turbo-preview
  max_workers: 5
  temperature: 0.1
  checkpoint_dir: .swebench/checkpoints

evaluation:
  max_patch_size_mb: 5
  timeout: 900
```

### 6. Testing Strategy

#### 6.1 Unit Tests
- Test generation integration logic
- Test cost estimation
- Test checkpoint/resume functionality
- Test error handling paths

#### 6.2 Integration Tests
- Test full pipeline with mock provider
- Test with real instances (limited set)
- Test checkpoint recovery
- Test partial batch handling

#### 6.3 End-to-End Tests
```python
def test_full_generation_and_evaluation():
    """Test complete flow from dataset to evaluation."""
    # 1. Load small dataset subset
    # 2. Generate patches with mock provider
    # 3. Run evaluation
    # 4. Verify results
```

### 7. Documentation Updates

#### 7.1 README.md Updates
- Add generation examples
- Document provider setup
- Show cost considerations
- Include performance tips

#### 7.2 CLI Help Text
- Update run command help
- Add generation examples
- Document new flags
- Include troubleshooting

### 8. Migration Considerations

#### 8.1 Backward Compatibility
- Existing `--patches` workflow must continue to work
- Default behavior without provider should remain unchanged
- Clear error messages when provider not configured

#### 8.2 Feature Flags
Consider adding feature flags for gradual rollout:
```python
FEATURES = {
    'generation_enabled': os.environ.get('SWEBENCH_GENERATION_ENABLED', 'true') == 'true',
    'batch_checkpointing': os.environ.get('SWEBENCH_BATCH_CHECKPOINTING', 'true') == 'true',
}
```

## Acceptance Criteria

1. **Basic Generation Flow**
   - [ ] `swebench run -d lite --count 5 --provider openai` generates and evaluates
   - [ ] Progress is displayed during generation
   - [ ] Costs are tracked and displayed
   - [ ] Generated patches are saved for inspection

2. **Error Handling**
   - [ ] Clear error when provider not configured
   - [ ] Graceful handling of generation failures
   - [ ] Option to continue with partial results
   - [ ] Proper exit codes for different failure modes

3. **Performance**
   - [ ] Concurrent generation with configurable workers
   - [ ] Checkpoint/resume for large batches
   - [ ] Memory-efficient for large datasets
   - [ ] Reasonable timeouts with retry logic

4. **User Experience**
   - [ ] Rich progress display during generation
   - [ ] Cost warnings for expensive operations
   - [ ] Clear success/failure reporting
   - [ ] Helpful error messages with fixes

5. **Testing**
   - [ ] 90%+ code coverage for new code
   - [ ] Integration tests pass
   - [ ] Documentation is complete
   - [ ] Examples work as documented

## Research Required

### 1. Full Instance Data Access (RESOLVED)
The basic SWE-bench dataset contains:
- `instance_id`: Unique identifier
- `problem_statement`: The issue description
- `repo`: Repository name
- `base_commit`: Git commit hash
- `patch`: The reference patch (for validation)

For enhanced context, there are BM25 retrieval datasets available:
- `princeton-nlp/SWE-bench_bm25_13K` (13K token context)
- `princeton-nlp/SWE-bench_bm25_27K` (27K token context)
- `princeton-nlp/SWE-bench_bm25_40K` (40K token context)

These contain pre-retrieved relevant code files for each instance, which we can load using:
```python
from datasets import load_dataset
context_dataset = load_dataset('princeton-nlp/SWE-bench_bm25_13K', split='test')
```

### 2. Optimal Prompt Engineering
Based on the available data, we should:
- Use the problem_statement as the core issue description
- Include repository and commit info for context
- Add retrieved code files from BM25 datasets when available
- Use different template styles based on problem complexity
- Implement smart truncation to fit within token limits

### 3. Cost Optimization
- Default batch size: 5 concurrent requests (configurable)
- Implement exponential backoff for retries (max 3 attempts)
- Start with smaller context (13K) and increase if needed
- Cache successful generations to avoid re-running

## Implementation Order

1. **Phase 1: Core Integration** (Priority: HIGH)
   - Create GenerationIntegration class
   - Modify run command to accept provider flags
   - Basic generation flow without optimization
   - Simple progress display

2. **Phase 2: Enhanced Context** (Priority: HIGH)
   - Research and implement full instance data loading
   - Enhance PromptBuilder with code context
   - Implement smart context truncation
   - Add validation for generated patches

3. **Phase 3: Production Features** (Priority: MEDIUM)
   - Batch processing with checkpoints
   - Cost estimation and warnings
   - Rich progress display
   - Configuration file support

4. **Phase 4: Optimization** (Priority: LOW)
   - Smart instance ordering
   - Adaptive retry strategies
   - Result caching
   - Performance profiling

## Notes

- The integration should feel seamless - users shouldn't need to understand the internal complexity
- Generation should be optional - the tool still works without any providers configured
- We should track and report costs prominently to avoid surprises
- The generated patches should be saved in a format that allows manual inspection and editing

## Questions/Uncertainties

**Blocking**:
1. How to access full instance data with code context? The current dataset only has problem statements.
2. Should generation be blocking or allow background processing with status checking?

**Non-blocking**:
1. Optimal default temperature for patch generation? (Assumption: 0.1 for consistency)
2. Should we support multiple providers in one run for comparison? (Assumption: No for v1)
3. Default timeout per instance? (Assumption: 5 minutes)

## Success Metrics

- Users can go from `pip install` to evaluated results in <10 minutes
- Generation success rate >70% on lite dataset
- Cost per instance <$0.50 on average
- No regression in existing functionality
- Positive user feedback on ease of use

## Remediation Plan for Integration Issues (2025-01-25)

After implementing the initial CLI integration, a critical review revealed several fundamental issues that need to be fixed. This section breaks down the remediation work into small, focused tasks suitable for individual agents.

### Issue 1: BatchProcessor Integration Mismatch

The current implementation incorrectly instantiates BatchProcessor with wrong parameters.

#### Task 1.1: Fix BatchProcessor Constructor Call
**File**: `src/swebench_runner/generation_integration.py`
**Function**: `generate_patches_for_evaluation()` lines 114-120
**Specific Changes**:
1. Remove the incorrect parameters: `prompt_builder`, `checkpoint_path`
2. Change to use correct parameters: `checkpoint_dir` instead of `checkpoint_path`
3. Update to match actual BatchProcessor constructor signature
4. Test that BatchProcessor instantiates correctly

#### Task 1.2: Create BatchProcessorAdapter Class
**File**: Create new `src/swebench_runner/generation/batch_adapter.py`
**Purpose**: Adapter to connect PromptBuilder with BatchProcessor
**Implementation**:
```python
class BatchProcessorAdapter:
    """Adapts batch processing to use PromptBuilder for each instance."""
    def __init__(self, batch_processor: BatchProcessor, prompt_builder: PromptBuilder):
        self.batch_processor = batch_processor
        self.prompt_builder = prompt_builder

    async def process_with_prompts(self, instances: List[Dict]) -> BatchResult:
        # Implementation to inject prompt building into the process
```

#### Task 1.3: Update PatchGenerator to Use PromptBuilder
**File**: `src/swebench_runner/generation/patch_generator.py`
**Function**: `_build_basic_prompt()` method
**Changes**:
1. Add optional `prompt_builder` parameter to PatchGenerator.__init__
2. If prompt_builder provided, use it instead of _build_basic_prompt
3. Update the prompt building logic to use PromptBuilder.build_prompt()
4. Add tests for both modes (with and without PromptBuilder)

### Issue 2: Missing Patch Validation

Generated patches are not validated before saving.

#### Task 2.1: Add Validation to GenerationResult Processing
**File**: `src/swebench_runner/generation_integration.py`
**Function**: `generate_patches_for_evaluation()` lines 159-167
**Changes**:
1. After line 160, add patch validation using self.patch_validator
2. Only include patches that pass validation
3. Log validation failures with reasons
4. Update summary to show validated vs invalid patches

#### Task 2.2: Create ValidationStats Class
**File**: `src/swebench_runner/generation/validation_stats.py`
**Purpose**: Track validation statistics during batch processing
**Implementation**:
```python
@dataclass
class ValidationStats:
    total_generated: int = 0
    valid_patches: int = 0
    invalid_patches: int = 0
    validation_errors: Dict[str, List[str]] = field(default_factory=dict)
```

### Issue 3: Checkpoint/Resume Not Implemented

Checkpoint functionality is incomplete despite being passed through.

#### Task 3.1: Implement Checkpoint Loading in GenerationIntegration
**File**: `src/swebench_runner/generation_integration.py`
**Function**: Add new method `_load_checkpoint()`
**Implementation**:
1. Check if checkpoint_path exists
2. Load previous results if found
3. Filter out already-completed instances
4. Return remaining instances to process

#### Task 3.2: Implement Checkpoint Saving in GenerationIntegration
**File**: `src/swebench_runner/generation_integration.py`
**Function**: Add checkpoint saving after batch processing
**Implementation**:
1. After BatchProcessor completes, save checkpoint
2. Include completed instances, failed instances, and statistics
3. Use atomic write (write to temp, then rename)
4. Add error handling for checkpoint save failures

### Issue 4: Environment Variable Support

Missing support for environment variables as specified in plan.

#### Task 4.1: Add Environment Variable Loading
**File**: Create `src/swebench_runner/generation/config.py`
**Implementation**:
```python
class GenerationConfig:
    """Load generation configuration from environment variables."""

    @classmethod
    def from_env(cls) -> 'GenerationConfig':
        return cls(
            default_provider=os.getenv('SWEBENCH_PROVIDER', 'openai'),
            default_model=os.getenv('SWEBENCH_MODEL'),
            max_workers=int(os.getenv('SWEBENCH_MAX_WORKERS', '5')),
            generation_timeout=int(os.getenv('SWEBENCH_GENERATION_TIMEOUT', '300'))
        )
```

#### Task 4.2: Integrate Environment Config into CLI
**File**: `src/swebench_runner/cli.py`
**Function**: `run()` command
**Changes**:
1. Import GenerationConfig
2. Load config at start of run command
3. Use config values as defaults when not specified via CLI
4. Add tests for environment variable precedence

### Issue 5: Data Format Compatibility

Generated patches format may not match evaluation expectations.

#### Task 5.1: Create PatchFormatter Class
**File**: Create `src/swebench_runner/generation/patch_formatter.py`
**Purpose**: Convert between generation and evaluation formats
**Implementation**:
```python
class PatchFormatter:
    """Formats patches for evaluation compatibility."""

    def format_for_evaluation(self, generation_results: List[Dict]) -> List[Dict]:
        """Convert generation format to evaluation format."""
        # Ensure 'patch' field (not 'prediction')
        # Add required metadata
        # Validate format
```

#### Task 5.2: Update GenerationIntegration Output Format
**File**: `src/swebench_runner/generation_integration.py`
**Function**: Lines 158-172 (save results section)
**Changes**:
1. Use PatchFormatter to ensure correct format
2. Validate output format before saving
3. Add format version to output for future compatibility
4. Test with actual evaluation pipeline

### Issue 6: Progress Display Enhancement

Current progress display is too basic.

#### Task 6.1: Create RichProgressDisplay Class
**File**: Create `src/swebench_runner/generation/progress_display.py`
**Implementation**:
```python
class RichProgressDisplay:
    """Enhanced progress display for generation."""

    def __init__(self, console: Console):
        self.console = console
        self.progress = Progress(...)

    def display_instance_progress(self, instance_id: str, status: str):
        # Show per-instance progress

    def update_cost_estimate(self, current_cost: float):
        # Live cost tracking
```

#### Task 6.2: Integrate Progress Display
**File**: `src/swebench_runner/generation_integration.py`
**Changes**:
1. Replace basic progress with RichProgressDisplay
2. Add callback to BatchProcessor for per-instance updates
3. Show live cost accumulation
4. Add ETA calculation based on completion rate

### Issue 7: Error Recovery

Missing proper error handling and recovery mechanisms.

#### Task 7.1: Create ErrorRecoveryStrategy Class
**File**: Create `src/swebench_runner/generation/error_recovery.py`
**Implementation**:
```python
class ErrorRecoveryStrategy:
    """Strategies for recovering from generation errors."""

    def should_retry(self, error: Exception, attempt: int) -> bool:
        # Determine if error is retryable

    def get_backoff_time(self, error: Exception, attempt: int) -> float:
        # Calculate backoff time

    def modify_request(self, instance: Dict, error: Exception) -> Dict:
        # Modify request based on error (e.g., reduce tokens)
```

#### Task 7.2: Add Timeout Handling
**File**: `src/swebench_runner/generation_integration.py`
**Function**: `generate_patches_for_evaluation()`
**Changes**:
1. Add timeout parameter (default from env var)
2. Wrap provider calls with asyncio.timeout()
3. Handle timeout errors gracefully
4. Add timeout info to error reporting

### Issue 8: Test Coverage

Most tests are skipped, providing false confidence.

#### Task 8.1: Create Integration Test with MockProvider
**File**: `tests/test_generation_integration.py`
**Function**: Replace `test_generate_patches_for_evaluation`
**Implementation**:
1. Remove skip decorator
2. Use actual MockProvider instead of mocks
3. Test full flow from instances to saved patches
4. Verify output format is correct
5. No mocking of core components

#### Task 8.2: Create End-to-End Test
**File**: Create `tests/test_cli_generation_e2e.py`
**Implementation**:
```python
def test_generation_flow_with_mock_provider():
    """Test complete generation flow using MockProvider."""
    # 1. Create test instances
    # 2. Run CLI with MockProvider
    # 3. Verify patches generated
    # 4. Verify format compatible with evaluation
    # 5. No Docker required
```

#### Task 8.3: Add Error Scenario Tests
**File**: `tests/test_generation_error_handling.py`
**Implementation**:
1. Test provider initialization failures
2. Test rate limit handling
3. Test token limit errors
4. Test network failures
5. Test partial batch failures
6. Verify error messages and recovery

### Issue 9: Missing Configuration File Support

No support for .swebench.yaml configuration.

#### Task 9.1: Create ConfigLoader Class
**File**: Create `src/swebench_runner/generation/config_loader.py`
**Implementation**:
```python
class ConfigLoader:
    """Load configuration from .swebench.yaml file."""

    def load_config(self, path: Path = Path('.swebench.yaml')) -> Dict:
        # Load YAML config
        # Validate schema
        # Merge with defaults
```

#### Task 9.2: Integrate Config Loading
**File**: `src/swebench_runner/cli.py`
**Function**: At module level or in run command
**Changes**:
1. Check for .swebench.yaml in current directory
2. Load configuration if found
3. Use as defaults for CLI options
4. Environment variables override config file
5. CLI flags override everything

### Task Execution Order

**Phase 1 - Critical Fixes** (Do these first):
- Task 1.1: Fix BatchProcessor Constructor Call
- Task 2.1: Add Validation to GenerationResult Processing
- Task 5.1: Create PatchFormatter Class
- Task 5.2: Update GenerationIntegration Output Format

**Phase 2 - Core Functionality** (Do these second):
- Task 1.2: Create BatchProcessorAdapter Class
- Task 1.3: Update PatchGenerator to Use PromptBuilder
- Task 3.1: Implement Checkpoint Loading
- Task 3.2: Implement Checkpoint Saving
- Task 7.1: Create ErrorRecoveryStrategy Class
- Task 7.2: Add Timeout Handling

**Phase 3 - Enhanced Features** (Do these third):
- Task 4.1: Add Environment Variable Loading
- Task 4.2: Integrate Environment Config into CLI
- Task 6.1: Create RichProgressDisplay Class
- Task 6.2: Integrate Progress Display
- Task 9.1: Create ConfigLoader Class
- Task 9.2: Integrate Config Loading

**Phase 4 - Testing** (Do these last):
- Task 8.1: Create Integration Test with MockProvider
- Task 8.2: Create End-to-End Test
- Task 8.3: Add Error Scenario Tests
- Task 2.2: Create ValidationStats Class (can be done anytime)

Each task is now small enough for a single agent to complete successfully, with clear boundaries and specific implementation details.
