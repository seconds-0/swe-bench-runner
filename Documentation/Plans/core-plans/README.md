# Core Plans - Phase 1: Minimum Viable Magic

This folder contains the work plans for Phase 1, which focuses on getting the absolute minimum working: a CLI that can execute a single SWE-bench instance and show the result.

## Phase 1 Work Plans (In Order)

1. **[01-MVP-CLI](01-MVP-CLI.md)** - Basic CLI structure with Click
   - Status: ✅ Completed
   - Creates the `swebench` command
   - Implements basic `run --patches` functionality
   - Foundation for everything else

2. **[02-MVP-DockerRun](02-MVP-DockerRun.md)** - Execute single instance in Docker
   - Status: ✅ Completed
   - Depends on: 01-MVP-CLI
   - Connects to Docker and runs evaluation
   - Core execution engine

3. **Output Implementation** (Split into two phases)
   - **[02a-MVP-BasicOutput-v1](02a-MVP-BasicOutput-v1.md)** - Minimal output (~150 lines)
     - Status: Not Started
     - Depends on: 01-MVP-CLI, 02-MVP-DockerRun
     - Basic pass/fail display
     - Simple JSON results
     - Correct exit codes

   - **[02b-MVP-BasicOutput-v2](02b-MVP-BasicOutput-v2.md)** - Delightful output (~1200 lines)
     - Status: Not Started
     - Depends on: 02a-MVP-BasicOutput-v1
     - Real-time progress bars
     - Beautiful HTML reports
     - Success celebrations
     - Threading for responsive UI

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
- **Basic output first** - v1 has no colors or progress bars
- **Hardcoded defaults** - No configuration needed
- **Progressive enhancement** - v1 works, v2 delights

## Implementation Strategy

1. **MVP-BasicOutput-v1** gets us to a working tool quickly (~150 lines)
2. **MVP-BasicOutput-v2** makes it delightful without changing core functionality (~1200 lines)

This approach ensures we always have a working tool while iterating on UX.

## Next Phase

After Phase 1 is complete, Phase 2 (Make It Trustworthy) will add:
- Patch validation
- Multiple instance support
- Better error handling
- Resource management
- Resume capabilities

But first, we need to get basic execution working!
