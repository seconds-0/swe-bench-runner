# Core Plans - Phase 1: Minimum Viable Magic

This folder contains the work plans for Phase 1, which focuses on getting the absolute minimum working: a CLI that can execute a single SWE-bench instance and show the result.

## Phase 1 Work Plans (In Order)

1. **[01-MVP-CLI](01-MVP-CLI.md)** - Basic CLI structure with Click
   - Status: ✅ Completed
   - Creates the `swebench` command
   - Implements basic `run --patches` functionality
   - Foundation for everything else

2. **[02-MVP-DockerRun](02-MVP-DockerRun.md)** - Execute single instance in Docker
   - Status: Not Started
   - Depends on: 01-MVP-CLI
   - Connects to Docker and runs evaluation
   - Core execution engine

3. **[03-MVP-BasicOutput](03-MVP-BasicOutput.md)** - Print results to terminal
   - Status: Not Started
   - Depends on: 01-MVP-CLI, 02-MVP-DockerRun
   - Shows pass/fail results
   - Provides summary statistics

## Success Criteria

When Phase 1 is complete, a user should be able to:

```bash
pip install swebench-runner
swebench run --patches my_patches.jsonl
```

And see:
```
Starting evaluation...
Evaluating django__django-12345...
✓ django__django-12345: PASSED

Summary:
Total: 1
Passed: 1 (100.0%)
Failed: 0 (0.0%)
```

## Key Principles

- **Minimal but working** - No fancy features, just execution
- **Single instance only** - No parallelism yet
- **Basic output** - No colors or progress bars
- **Hardcoded defaults** - No configuration needed

## Next Phase

After Phase 1 is complete, Phase 2 (Make It Trustworthy) will add:
- Patch validation
- Exit codes
- Error handling
- HTML reports
- Logging

But first, we need to get basic execution working!