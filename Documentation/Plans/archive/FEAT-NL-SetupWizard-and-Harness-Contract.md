# Task ID: FEAT-NL-SetupWizard-and-Harness-Contract

## Problem Statement
Running SWE-bench should be a one-line, human-friendly experience. Current friction points:
- Users must know provider models, arch vs namespace, and harness quirks
- Harness images/arch/registry cause confusing failures
- Our runner couples to unstable outputs (evaluation_results/ only)

## Goals
- Natural-language setup and run (wizard + single command)
- Strict, minimal integration contract with the harness
- Actionable preflight and error messages

## Scope
- Wizard: provider discovery, model validation, Docker & registry preflight
- Runner: remove hardcoded namespace/arch; parse canonical harness report first
- One-liners: defaults from .env, flags override

## Design Principles
- Harness is source of truth for images and results
- Runner orchestrates and explains, never guesses
- Progressive disclosure: show options only when needed

## User Flows
1) First-run wizard (NL prompts)
   - Detect Docker, disk, memory
   - Ask provider (openai/anthropic/openrouter)
   - Fetch models via list-models endpoint; offer selection
   - Optional: test one harness instance pull (dry-run) and suggest fixes
   - Save working config to `.env`

2) Run
   - `swebench run -d lite` uses `.env` defaults
   - Advanced users can override provider/model/namespace

## Harness Contract
- Invocation: `python -m swebench.harness.run_evaluation` with only
  - `--dataset_name`, `--predictions_path`, `--max_workers`, `--timeout`, `--run_id`, `--cache_level`
  - Optional: `--namespace` only when user sets `SWEBENCH_DOCKER_NAMESPACE`
- Results parsing order:
  1. Top-level `test.<run_id>.json` report (canonical)
  2. `evaluation_results/*.json` (legacy)
  3. Log tail with path (diagnostics)

## Implementation Plan
- Phase 1 (This PR)
  - Remove hardcoded `--namespace`; add env `SWEBENCH_DOCKER_NAMESPACE` opt-in
  - Add robust fallback parser (done)
  - Improve error text to always include log path
- Phase 2
  - Add `setup` wizard step: provider list-models calls, arch detection, harness preflight
  - Persist `.env` with working defaults
  - Add `SWEBENCH_DISABLE_NAMESPACE` docs; we already now default to no namespace
- Phase 3
  - Natural-language prompts throughout wizard with smart defaults
  - Optional: JSON output for CI pipelines

## Test Plan
- Unit: fallbacks in `docker_run.parse_harness_results`
- E2E: run CLI with mock harness outputs for each fallback path
- Manual: live harness preflight against 1 instance; verify clear guidance on failure

## Risks & Mitigations
- Harness output shape changes → parse report guarded and tolerant
- Registry policy changes → wizard preflight detects and suggests docker login or alt namespace

## Status
In Progress (Phase 1 implemented)
