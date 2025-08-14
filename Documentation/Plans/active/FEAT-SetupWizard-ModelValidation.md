# Task ID: FEAT-SetupWizard-ModelValidation

## Problem Statement
Our setup wizard should validate model names for configured providers (OpenAI, Anthropic, OpenRouter) and help users pick valid models. Today, users must consult docs or guess model IDs. This leads to misconfiguration, confusing errors, and wasted time.

## Proposed Solution
Add a provider-agnostic "Model Discovery & Validation" step in the interactive setup wizard that:
- Detects configured providers (API keys present, auth passes)
- Calls each provider's official list-models endpoint to retrieve available models for the user account
- Normalizes and displays models in a selectable list (with basic metadata like context length and pricing when available)
- Validates user-entered model IDs against the fetched list and stores them as defaults
- Caches model lists briefly to avoid rate limits; supports refresh on demand

### Endpoints
- OpenAI: `GET /v1/models`
- Anthropic: `GET /v1/models` (header `anthropic-version` required)
- OpenRouter: `GET https://openrouter.ai/api/v1/models`

### UX
- If provider is configured: show a list of models with search/filter; allow manual entry with inline validation
- If provider is not configured: prompt to configure first or skip
- If listing fails (network/permissions): show actionable error and allow manual model entry with best-effort validation (e.g., known prefixes)

### Data Normalization
- Common fields to surface: `id` (model name), `context_window` (if available), `pricing` (OpenRouter provides), `provider` origin
- Keep raw payload for advanced/hidden details if needed later

### Caching Strategy
- Cache per-provider model list for 10–30 minutes in the runner cache dir
- Provide a "Refresh models" option in the wizard

## Automated Test Plan
- Unit tests per provider adapter:
  - Success path: returns list with expected shape
  - Error paths: 401/403 (auth), 429 (rate limit), network errors – ensure actionable messages
- Integration tests for wizard step:
  - With mock providers returning lists: shows choices and validates selection
  - With failures: allows manual entry, warns properly
- E2E (optional): run wizard with `SWEBENCH_TEST_MODE=true` using doubles for providers

## Components Involved
- `swebench_runner/providers/*` (lightweight HTTP adapter for list models)
- `swebench_runner/cli.py` (setup wizard step)
- `swebench_runner/cache.py` (short-lived cache storage)
- Tests in `tests/unit/` and `tests/e2e/`

## Dependencies
- Existing provider config detection (`ensure_provider_configured`)
- Network abstraction (optional) or direct `requests` usage

## Implementation Checklist
- [ ] Add provider-agnostic `list_models(provider_name)` interface
- [ ] Implement provider-specific calls:
  - [ ] OpenAI: `GET /v1/models`
  - [ ] Anthropic: `GET /v1/models` with version header
  - [ ] OpenRouter: `GET /api/v1/models`
- [ ] Normalize model entries (id, context/pricing if available)
- [ ] Add short-lived caching (10–30 min) with manual refresh
- [ ] Extend setup wizard to:
  - [ ] Detect configured providers
  - [ ] Fetch & display models (search/filter)
  - [ ] Validate manual entry; persist selection
  - [ ] Handle failures with actionable guidance
- [ ] Unit tests for adapters (success + error)
- [ ] Integration tests for wizard step (mock providers)
- [ ] Update docs: provider setup guide and wizard screenshots

## Verification Steps
- Run wizard; select provider; see model list; pick a valid model; confirm stored default
- Enter invalid model; see inline validation error
- Simulate 401/429/network errors; see clear messages and options

## Decision Authority
- Engineering can implement adapter, caching, and wizard UX
- Product/UX to review copy and default model recommendations

## Questions/Uncertainties
- Should we also surface tool/function calling or vision/audio capabilities? (out of scope MVP)
- Rate limit and pagination behaviors for each provider? (handle basic pagination if present)

## Acceptable Tradeoffs
- MVP surfaces only `id`, basic pricing (when available), and context size; deeper capability flags deferred
- Use synchronous HTTP; no async plumbing in V1

## Status
In Progress

## Notes
- Respect provider-specific auth headers; do not log API keys
- Backoff on 429 (simple exponential backoff with cap)
- Timebox model fetch (e.g., 10s per provider) to keep wizard snappy
