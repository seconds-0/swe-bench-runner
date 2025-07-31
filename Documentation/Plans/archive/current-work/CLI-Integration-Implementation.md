# Work Plan: CLI Integration Implementation

**Task ID**: FEAT-CLI-Integration-Implementation
**Status**: In Planning
**Priority**: High
**Last Updated**: 2025-01-25

## Problem Statement

The model provider system (Phase 2) has been implemented with all core generation components:
- PatchGenerator for orchestrating patch generation
- BatchProcessor for concurrent processing with checkpoints
- PromptBuilder for creating prompts from instances
- ResponseParser for extracting patches from model responses

However, these components are not integrated with the main SWE-bench runner CLI. The `run` command requires pre-generated patches, and the `generate` command is just a demo placeholder.

## Proposed Solution

Integrate the generation components with the existing CLI to enable seamless patch generation and evaluation in a single command. The implementation will follow the plan in `CLI-Integration-Plan.md` but with adjustments based on the actual codebase state.

## Implementation Checklist

### Phase 1: Core Integration Module

- [ ] Create `generation_integration.py` module
  - [ ] GenerationIntegration class to orchestrate patch generation
  - [ ] Convert instance data to generation-compatible format
  - [ ] Handle dataset loading with full context (including BM25 retrieval)
  - [ ] Save patches in evaluation-compatible JSONL format
  - [ ] Basic cost estimation and warnings

### Phase 2: Enhanced Run Command

- [ ] Modify `cli.py` run command to support generation
  - [ ] Add --provider flag (optional, uses default if not specified)
  - [ ] Add --model flag (optional, overrides provider default)
  - [ ] Add --generate-only flag (skip evaluation)
  - [ ] Add --generation-output flag (save patches to specific file)
  - [ ] Add --max-workers flag (control concurrency, default 5)
  - [ ] Integrate generation when no patches provided
  - [ ] Maintain backward compatibility with existing --patches workflow

### Phase 3: Update Generate Command

- [ ] Replace demo implementation in `cli.py` generate command
  - [ ] Load instance data from datasets using DatasetManager
  - [ ] Use PromptBuilder to create appropriate prompts
  - [ ] Handle single instance generation properly
  - [ ] Save patch in evaluation-compatible format
  - [ ] Show cost information if available

### Phase 4: Helper Classes

- [ ] Create CostEstimator class
  - [ ] Estimate batch costs based on provider pricing
  - [ ] Warn for expensive operations (>$10)
  - [ ] Track actual vs estimated costs

- [ ] Create GenerationFailureHandler
  - [ ] Handle partial batch failures gracefully
  - [ ] Offer to continue with successful patches
  - [ ] Provide clear error reporting

- [ ] Add progress display utilities
  - [ ] Rich progress bars during generation
  - [ ] Show current instance being processed
  - [ ] Display success/failure counts
  - [ ] Show cost accumulation

### Phase 5: Enhanced Context Loading

- [ ] Implement context enhancement for instances
  - [ ] Research BM25 retrieval datasets integration
  - [ ] Load code context from retrieval datasets when available
  - [ ] Handle missing context gracefully
  - [ ] Support different context sizes (13K, 27K, 40K tokens)

### Phase 6: Testing

- [ ] Create unit tests for GenerationIntegration
- [ ] Test CLI command integration
- [ ] Test error handling scenarios
- [ ] Test checkpoint/resume functionality
- [ ] Create end-to-end integration test

## Implementation Details

### GenerationIntegration Module Structure

```python
# src/swebench_runner/generation_integration.py

class GenerationIntegration:
    """Integrates patch generation with evaluation pipeline."""

    def __init__(self, provider: ModelProvider, ...):
        self.generator = PatchGenerator(provider)
        self.batch_processor = BatchProcessor(self.generator)
        self.prompt_builder = PromptBuilder()
        self.cost_estimator = CostEstimator()

    async def generate_patches_for_evaluation(
        self,
        instances: list[dict[str, Any]],
        output_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        max_workers: int = 5,
        progress_callback: Optional[Callable] = None
    ) -> Path:
        """Generate patches and return path to JSONL."""
        # Implementation details...
```

### CLI Integration Points

1. **Run Command Enhancement**:
   - Check if patches provided
   - If not, check for provider flag or default
   - Call GenerationIntegration to generate patches
   - Save to temp location and continue with evaluation

2. **Generate Command Rewrite**:
   - Remove demo code
   - Use real dataset loading
   - Generate actual patch
   - Save in proper format

### Key Technical Decisions

1. **Async/Sync Bridge**: Use existing SyncProviderWrapper for CLI integration
2. **Progress Display**: Use Rich library (already in dependencies) for beautiful progress
3. **Cost Management**: Estimate before starting, track during, report after
4. **Error Recovery**: Use BatchProcessor's checkpoint functionality
5. **Dataset Integration**: Start with basic dataset, add BM25 retrieval in phase 2

## Acceptance Criteria

1. **Basic Flow Works**
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
   - [ ] Memory-efficient for large datasets
   - [ ] Reasonable timeouts with retry logic

4. **User Experience**
   - [ ] Rich progress display during generation
   - [ ] Cost warnings for expensive operations
   - [ ] Clear success/failure reporting
   - [ ] Helpful error messages

## Research Required

### BM25 Retrieval Integration
- [x] Confirmed datasets available: `princeton-nlp/SWE-bench_bm25_13K`, etc.
- [ ] Test loading retrieval data and matching with instances
- [ ] Determine optimal context size based on model

### Cost Calculation
- [ ] Research actual provider pricing APIs
- [ ] Implement accurate cost estimation based on tokens
- [ ] Add cost tracking to BatchResult

## Questions/Uncertainties

**Non-blocking**:
1. Should we cache generated patches between runs? (Assumption: Yes, in .swebench/cache/patches/)
2. Default temperature for generation? (Assumption: 0.0 for consistency)
3. Should failed instances be retried automatically? (Assumption: Yes, up to 2 retries)

**Blocking**:
None - all information needed is available in the codebase.

## Dependencies

- Existing provider system (complete)
- Generation components (complete)
- Dataset loading (complete)
- CLI structure (existing)

## Testing Strategy

1. **Unit Tests**:
   - Mock provider for testing
   - Test cost estimation logic
   - Test error handling paths

2. **Integration Tests**:
   - Test full generation flow with mock provider
   - Test checkpoint/resume
   - Test partial batch handling

3. **End-to-End Test**:
   - Use mock provider
   - Generate patches for small dataset
   - Verify output format
   - Run evaluation on generated patches

## Notes

- Start with Phase 1-3 to get basic functionality working
- Enhanced context (BM25) can be added later without breaking changes
- Focus on user experience - clear progress and error messages
- Leverage existing components rather than reimplementing
