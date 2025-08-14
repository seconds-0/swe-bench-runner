# FEAT-NL-Wizard-and-QuickRun

> Progress snapshot (to keep us honest)

- [x] Interactive entry (swebench home: Quick Run / Guided Setup / Help)
- [x] Guided Setup scaffold (provider → key capture → live model list → dataset → count → run now)
- [x] Secrets stored securely via OS keychain; no plaintext secrets in `.env`
- [x] Minimal harness contract (no hardcoded namespace/arch; opt-in `SWEBENCH_DOCKER_NAMESPACE`)
- [x] Results parsing fallback (report → dir → logs tail + path)
- [ ] Wizard Preflight (image pull + one-instance smoke) with guided fixes
  - [x] Detect and sanitize pasted API keys (whitespace/newlines) before validation
  - [ ] Auto-detect Docker image pulls failing on arm64 due to Docker Hub missing images and seamlessly switch to GHCR
  - [ ] Persist `SWEBENCH_DOCKER_NAMESPACE=ghcr.io/epoch-research` after successful retry
  - [ ] If GHCR requires auth, prompt once to run `docker login ghcr.io` (TTY) or print instructions (non-interactive)
  - [ ] Optional macOS arm64 fallback: offer `DOCKER_DEFAULT_PLATFORM=linux/amd64` emulation for this session
- [ ] Help & Docs pane content
- [ ] Quick Run polish (reuse saved defaults; minimal prompts if missing)
- [ ] Unit/E2E tests for new modules (secrets/model catalog/tui) and preflight
- [ ] Optional JSON/CI outputs and profiles (later)

### Non-Interactive & Agent-Friendly Controls (New)
- Add CLI subcommands for profile/key management (no TUI required):
  - `swebench profiles list [--json]`
  - `swebench profiles create --provider <p> --profile <name> [--active]`
  - `swebench profiles set-active --provider <p> --profile <name>`
  - `swebench profiles set-key --provider <p> --profile <name> --key <value|env>`
  - `swebench profiles clear-key --provider <p> --profile <name>`
  - `swebench profiles rename --provider <p> --from <old> --to <new>`
  - `swebench profiles delete --provider <p> --profile <name>`
- Add flag aliases for CI: `--no-input` (aliases: `--non-interactive`, `--noninteractive`)
  - In non-interactive mode: no prompts, no `.env` writes, no ANSI clears; stable one-line outputs or `--json`
  - TTY-only screen clearing is opt-in via `SWEBENCH_TUI_CLEAR=1` (default off)

## Problem Statement
Running SWE-bench with our runner should be frictionless for a new user:
- No memorizing flags or reading long docs before a first success
- No repeated API key entry
- Clear guidance and fixes when Docker/Harness or provider issues occur

Today the experience is still CLI-first and error surfacing can feel low-level (registry/arch/logs). We need a natural-language terminal UX that guides the user end-to-end, saves working defaults, and runs with a single action the next time.

## Goals
- Provide an interactive, natural-language terminal UI (wizard) that:
  - Detects environment: Python, Docker, disk, network
  - Connects to local or API providers (OpenAI/Anthropic/OpenRouter/HuggingFace Inference API)
  - Validates API keys and lists available models live
  - Helps choose dataset and sample size
  - Runs a preflight (optional one-instance pull+smoke) to avoid later surprises
  - Saves working defaults so future runs are one command
- Offer a “Quick Run” path that uses saved defaults with no prompts
- Present clear, actionable errors with log paths and last lines
- Maintain a minimal, stable integration contract with the official SWE-bench harness

## Non-Goals (for this feature)
- Implement a graphical desktop UI
- Full project profiles/workspaces (nice-to-have later)
- Advanced cost governance/budgets (still show estimates; don’t block)

---

## UX Spec

### Entry point: `swebench` (interactive TTY)
```
────────────────────────────────────────────────────────────────────────
🚀 SWE-bench Runner
────────────────────────────────────────────────────────────────────────
Welcome! What would you like to do?

  1) Quick Run
  2) Guided Setup (recommended)
  3) Help & Docs

Enter number [2]:
```

- Quick Run
  - If defaults exist: shows summary and runs on Enter
  - If not: explains it needs a configured provider → routes to Setup
- Guided Setup
  - Full wizard: provider → key → model → dataset → count → (optional) preflight → run now
- Help & Docs
  - Shows a concise cheatsheet and links

### Guided Setup Flow

1) Welcome
```
────────────────────────────────────────────────────────────────────────
🚀 SWE-bench Runner Setup
────────────────────────────────────────────────────────────────────────
We’ll help you connect a model, pick a dataset, and run a small smoke test.

Press Enter to continue.
```

2) System checks (automatic)
```
🔎 Checking your system...
  • Python: 3.11 ✔
  • Docker: Running ✔
  • Disk space: 45 GB free ✔
  • Network: OK ✔

Press Enter to continue.
```

3) Choose compute source
```
Select a compute source:
  1) Local (Ollama)
  2) Hugging Face Inference API
  3) Anthropic API
  4) OpenAI API
  5) OpenRouter
Enter number [2]:
```

#### Branch A: Local (Ollama)
```
🔧 Local (Ollama)
  • Ollama detected at http://localhost:11434
  • Available models:
      1) llama3.2:1b
      2) llama3.1:8b-instruct
      3) mistral:7b-instruct
Enter number [1] or type a model name to pull:
```

#### Branch B: Hugging Face Inference API
```
🔗 Hugging Face Inference API
Enter your Hugging Face token (starts with hf_): ********** (input hidden)
Validating token... ✔

Fetching model catalog... (this may take a moment)
  • Example picks:
      1) meta-llama/Meta-Llama-3.1-8B-Instruct
      2) microsoft/phi-3-mini-4k-instruct
      3) mistralai/Mistral-7B-Instruct-v0.3
Enter number [1] or type a model id:
```

#### Branch C: Anthropic API
```
🔗 Anthropic API
Enter your Anthropic API key (sk-ant-...): ********** (input hidden)
Validating key... ✔

Available models:
  1) claude-3-haiku-20240307
  2) claude-3-sonnet-20240229
  3) claude-3-opus-20240229
Enter number [1]:
```

#### Branch D: OpenAI / OpenRouter
- Same UX shape as Anthropic/HF: enter key → list models → pick one
- For OpenRouter: key is optional; list models may still work for public catalog

4) Dataset selection
```
Pick a dataset:
  1) lite (quick validation runs)
  2) verified (curated)
  3) full (complete benchmark)
Enter number [1]:
```

5) Sample size
```
Limit the run to N instances?
  • Yes → Enter a number (default 5)
  • No  → Leave blank to run all of the dataset
How many instances? [5]:
```

6) Secrets storage (only for API providers)
```
Save your API key securely so you won’t be prompted again? [Y/n]
  • macOS Keychain / Windows Credential Locker / Linux Secret Service
  • Stored as: service="swebench-runner", account="openai"
```

7) Preflight (optional, recommended on first run)
```
Would you like to run a quick preflight?
  • Tests one harness image pull and a minimal evaluation
  • Catches registry/arch/login issues early
Run preflight now? [Y/n]:
```
- If pull fails (e.g., “denied”):
```
❌ Image pull was denied by registry.
How to fix:
  • Run: docker login ghcr.io
  • Or: use an alternate namespace (Advanced → registry mirror)
  • Or: switch to emulation (x86) if your arch image is restricted
Press Enter to apply a fix or Esc to skip.
```

8) Summary and Run
```
Summary
  • Compute: Anthropic → claude-3-haiku-20240307
  • Dataset: SWE-bench lite
  • Count: 5
  • Harness: Ready

Press Enter to run now, or Ctrl+C to cancel.
```

9) Progress and Results
```
🤖 Generating patches [#####-----] 3/5
⏱  Rate limit/backoff shown inline when applicable
🧪 Running harness evaluation [##--------] 1/5
   Pulling container image (first time can take a while)...

🎯 EVALUATION SUMMARY
  • Selected: 5
  • Evaluated: 5
  • Passed:   2
  • Failed:   3
📁 Results saved to evaluation_results/... and results/<instance_id>/

Press Enter to exit or type “R” to rerun failed only.
```

### Quick Run Flow (`swebench` → 1)
- If defaults exist:
```
Quick Run will use:
  • Provider: Anthropic
  • Model: claude-3-haiku-20240307
  • Dataset: lite
  • Count: 5
Press Enter to run now.
```
- If missing defaults: “Quick Run requires a configured model. Choose Setup first.”

### Help & Docs (`swebench` → 3)
```
Common commands:
  swebench setup      # guided setup
  swebench run        # uses saved defaults
  swebench run -d lite --count 5
  swebench clean --all

Troubleshooting:
  • Image pull denied → docker login ghcr.io
  • Provider 401 → check your key / permissions
  • Harness did not write evaluation_results → see log with path shown in error
```

---

## Secrets Storage
- Secure by default via OS keychain (per-user):
  - macOS: Keychain Access
  - Windows: Credential Manager
  - Linux: Secret Service (Gnome Keyring/KWallet)
- Naming: `service=swebench-runner`, `account={provider}` (e.g., `openai`, `anthropic`)
- Resolution order (runtime):
  1) Environment variable (CI/containers)
  2) OS keychain
  3) .env pointer (e.g., `OPENAI_API_KEY=keyring://swebench-runner/openai`)
  4) Interactive prompt (TTY only)
- Management UX:
  - `swebench secrets set openai` (prompts hidden, validates, stores)
  - `swebench secrets get openai` (reports presence/backend; never prints key)
  - `swebench secrets clear openai`

---

## Provider Model Catalog (Live)
- OpenAI: `GET /v1/models` → list `id`
- Anthropic: `GET /v1/models` → list `id`
- OpenRouter: `GET https://openrouter.ai/api/v1/models` → list `id`
- Hugging Face: list via Inference API/project endpoints as available; otherwise curated list with validation attempt
- UX: search/filter list; allow manual entry with live validation request

---

## Harness Contract
- Invocation (minimal, stable):
  - `--dataset_name`, `--predictions_path`, `--max_workers`, `--timeout`, `--run_id`, `--cache_level`
  - `--namespace` only when user explicitly sets `SWEBENCH_DOCKER_NAMESPACE`
- Results parsing order:
  1) Top-level harness report `test.<run_id>.json`
  2) `evaluation_results/*.json`
  3) Log tail (+ file path)
- Preflight: optional one-instance pull+smoke to verify registry/arch/login before a full run

---

## Configuration
- `.env` holds non-sensitive defaults:
  - `SWEBENCH_PROVIDER`, `SWEBENCH_MODEL`, `SWEBENCH_MAX_WORKERS`, `NO_COLOR`
  - Optional: `SWEBENCH_DOCKER_NAMESPACE` (advanced), `SWEBENCH_ARCH` (informational only)
- Secrets in keychain; `.env` may use keyring pointer for clarity

---

## Implementation Phases

### Phase 1 (MVP Wizard & Contract)
- Add `swebench setup` wizard (Rich-based) with:
  - Provider selection → key capture → live catalog listing → model selection
  - Dataset + count
  - Save defaults and show ready-to-run command
- Remove hardcoded harness namespace / arch guessing  ✅ Done
- Add robust results fallback parsing (report → dir → logs)  ✅ Done
Status: Implemented initial wizard (provider/model/dataset/count, secure secrets, run-now).

### Phase 2 (Preflight & Secrets)
- Wizard preflight: one-instance pull+smoke; diagnose registry/arch/auth and suggest concrete fixes (docker login, mirror, emulation)
- OS keychain integration for saving API keys  ✅ Done
- Secrets management subcommands (`secrets set/get/clear`)  ✅ Done

  Enhancements (auto-fix registry pulls):
  - Detection points: preflight and first failing runtime instance (log tail / stderr)
  - Auto-retry with `--namespace ghcr.io/epoch-research` when 404 pull access denied on `swebench/sweb.eval.arm64.*`
  - Persist `.env` namespace after a successful retry; never overwrite if already set
  - On GHCR 401: prompt to `docker login ghcr.io`; continue gracefully if declined
  - Guardrails: at most one auto-retry per instance, non-interactive prints clear instructions
Status: Preflight pending.

### Phase 3 (Polish)
- Full NL prompts and fewer steps when defaults exist
- Quick Run shortcut command (using saved defaults silently)
- Optional JSON output for CI pipelines
- Profiles (e.g., personal/work keys) [optional]

---

## Test Plan

### Unit
- Model catalog adapters: success/error for OpenAI/Anthropic/OpenRouter
- Keychain wrapper: set/get/clear; env override precedence
- Harness parser: report parsing, dir parsing, logs fallback
- Docker auto-namespace: when stderr/log tail includes 404 pull access denied for `swebench/sweb.eval.arm64.*`, runner retries once with `--namespace ghcr.io/epoch-research` and persists `.env` on success
- GHCR denied path: surfaces actionable `docker login ghcr.io` guidance without looping
- Profiles subcommands: list/create/set-active/set-key/clear-key/rename/delete; `--json` on list; non-interactive behavior (no prompts)

### E2E (Wizard)
- Local (Ollama): detect models, pick dataset/count, run
- HF Inference API: token entry → model list → validation → run
- Anthropic: key entry → model list → run
- Error UX: 401 (auth), 429 (rate limit), registry denied (docker login), offline dataset
- Auto-fix UX: preflight detects missing arm64 images on Docker Hub, switches to GHCR, persists namespace, re-runs successfully
- Options UX: grouped provider/profile list with action-first flows; no provider pre-pick for profile actions

### Live Smoke (Opt-in)
- With Docker running and network available:
  - Preflight: pull and run one instance; verify summary and saved defaults

---

## Risks & Tradeoffs
- Harness output shape may change → tolerant parsers and fallback to log-tail
- Registry policies may differ per user/org → preflight catches early and offers fixes
- Keychain not available on headless Linux → fall back to env vars with clear warning

---

## Examples for Integration Testing

### Example 1: Local (Ollama) Happy Path
1) `swebench setup`
2) Choose `1) Local (Ollama)` → pick `llama3.2:1b`
3) Pick dataset `lite` → count `5` → Run
4) Expect:
   - Generation shown with no rate limit warnings
   - Harness run progress
   - Summary with 5 evaluated and artifacts written

### Example 2: Anthropic Happy Path
1) `swebench setup`
2) Choose `3) Anthropic` → enter valid `sk-ant-...` → list models → select `claude-3-haiku-20240307`
3) Dataset `lite` → count `5` → Run
4) Expect:
   - Key saved in keychain (if accepted)
   - Cost estimate shown (informational)
   - Summary with log artifacts

### Example 3: HF Inference API Invalid Token
1) `swebench setup`
2) Choose `2) Hugging Face` → enter invalid token
3) Expect:
   - Validation fails with 401
   - Re-prompt for token with help link
   - After fix, model list appears

### Example 4: Harness Registry Denied
1) `swebench setup`
2) Preflight enabled
3) Expect:
   - If image denied, show log-tail + path and guidance: `docker login ghcr.io` or set alternate namespace
   - Offer “Retry preflight” after applying fix

### Example 5: Quick Run
1) After a successful setup, run `swebench`
2) Choose `1) Quick Run`
3) Expect:
   - Immediate run using saved defaults
   - No prompts, shows progress then summary

---

## Open Questions
- Should we auto-run preflight silently after Setup unless the user opts out?
- Default sample-size for first Quick Run (5 vs 10)?
- Do we mirror a curated subset of models for providers with very large catalogs for faster listing?

---

## Status
Draft for review. This plan captures UX, secrets strategy, harness contract, and testable examples to drive implementation and integration tests.
