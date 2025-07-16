# SWE-Bench Runner

> Run any subset of SWE-bench with one clear command and fix any issue in minutes, not hours.

⚠️ **Status**: In active development. Not yet ready for use.

## Vision

SWE-bench has become the de-facto benchmark for evaluating code-fixing agents, but getting the harness to run locally is still painful. This tool makes SWE-bench evaluation so simple that users think "Holy shit, this is what I wanted the whole time!"

## Target Experience

```bash
pip install swebench-runner
swebench run --patches my_patches.jsonl
```

That's it. No Docker setup, no config files, no cryptic errors.

## Documentation

- [Product Requirements Document](Documentation/PRD.md) - What we're building and why
- [UX Plan](Documentation/UX_Plan.md) - How users will interact with the tool
- [Architecture](Documentation/Architecture.md) - Technical design and implementation
- [V2 Features](Documentation/V2_Features.md) - Future enhancements

## Development

This project is being developed with AI assistance following the workplan methodology outlined in [CLAUDE.md](CLAUDE.md).

## License

MIT