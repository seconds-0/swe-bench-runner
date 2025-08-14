# Live TUI (Single-Screen) and Non-Interactive Full Reprint

Status: Draft (kept up to date during implementation)

## Goals
- Delightful, calm configuration UX in terminals (TTY): single-screen feel, minimal noise.
- Deterministic, thorough logs for agents/CI (non-interactive): full context printed before/after actions; no ANSI.

## Modes
- TTY (human): `sys.stdin.isatty()` and `sys.stdout.isatty()` and not `--no-input`
- Non-interactive (agent/CI): any of
  - `--no-input` (aliases: `--non-interactive`, `--noninteractive`)
  - `CI=1`, `PYTEST_CURRENT_TEST` set, `TERM=dumb`
  - Not a TTY

## Live TUI (TTY only)
- Gate: `SWEBENCH_TUI_LIVE`
  - Phase 1: opt-in (`SWEBENCH_TUI_LIVE=1`)
  - Phase 2: default ON for TTY (disable with `SWEBENCH_TUI_LIVE=0`)
- Screens to convert (in order):
  1. Options (profiles/keys)
  2. Guided Setup (provider → key → model → dataset → count → preflight → run)
  3. Preflight (progress + auto-fix)
  4. Home (entry)
  5. Help/Docs (read-only)
- Render contract: `render(state) -> Rich renderable`; state-only mutations; Live updates in place.
- Prompts: stop Live → prompt → validate → mutate state → Live update.

## Non-Interactive “Full Reprint”
- Behavior when non-interactive detected:
  - No prompts; fail fast if missing inputs
  - No `.env` writes; no ANSI; no clears
  - Before action: print section header + full snapshot of relevant state
  - Decision line: print chosen action/parameters
  - After action: print confirmation + updated state snapshot
- Snapshots:
  - Options: grouped providers with profiles, Active ✓, Key yes/no
  - Setup: current step and inputs set so far
  - Preflight: step progress and last result
- Opt-in verbosity: `SWEBENCH_NONINTERACTIVE_VERBOSE=1` (forces full reprint in ambiguous cases)

## Agent/CI Subcommands (Implemented)
- `swebench profiles list [--json]`
- `swebench profiles create --provider <p> --profile <name> [--active]`
- `swebench profiles set-active --provider <p> --profile <name>`
- `swebench profiles set-key --provider <p> --profile <name> --key <value>`
- `swebench profiles clear-key --provider <p> --profile <name>`
- `swebench profiles rename --provider <p> --from <old> --to <new>`
- `swebench profiles delete --provider <p> --profile <name>`
- Discoverability: Help/Docs mention; Options footer tip

## Implementation Phases
- [x] Action-first Options (grouped), last-selection memory, minimal reprints
- [x] CI-safe flags: `--non-interactive`, `--noninteractive` alias `--no-input`
- [x] Profiles subcommands (list/create/set-active/set-key/clear-key/rename/delete) with `--json` on list
- [x] GHCR auto-retry + CI `.env` guard
- [ ] Non-interactive full reprint for Options
- [ ] Live TUI for Options (opt-in)
- [ ] Non-interactive full reprint for Setup + Preflight
- [ ] Live TUI for Setup + Preflight (opt-in)
- [ ] Make Live default ON for TTY (with disable flag) after bake-in

## Testing
- Unit:
  - Mode detection (TTY vs non-interactive)
  - Profiles subcommands text/JSON outputs
  - Snapshot renderers (no ANSI in non-interactive)
  - GHCR retry and CI behaviors
- E2E:
  - TTY: Options/Setup/Preflight behave as single-screen (no scroll)
  - Non-TTY: full context printed before/after actions, deterministic strings
- CI:
  - `--no-input` path avoids prompts and `.env` writes

## Risks / Mitigations
- ANSI in non-TTY → Gate Live/clears strictly to TTY; static prints elsewhere
- Prompt interleaving with Live → Centralized input helpers that stop/start Live
- Test brittleness of UI → Test states and textual snapshots, not raw screen diffs
- Maintenance → Keep state small; logic stays in services

## Success Criteria
- TTY: clear, calm single-screen feel across configuration
- Non-interactive: exhaustive, deterministic logs; easy agent automation
