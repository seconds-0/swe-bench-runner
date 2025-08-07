# Product Requirements Document (PRD)

## 1. Background & Motivation

SWE-bench has become the de-facto benchmark for evaluating code-fixing agents, but **getting the harness to run locally is still painful**:

* 10–20 GB Docker images must be built or pulled manually.
* Users have to craft JSON configs, understand dataset file formats, and juggle cache directories.
* When something breaks (e.g. repo build failure), the error surface is cryptic and time-consuming to debug.

This friction slows down researchers and practitioners who want to iterate on agent ideas or simply validate a new set of patches.

## 2. Vision Statement

“**Run any subset of SWE-bench with one clear command and fix any issue in minutes, not hours.**”

## 3. Personas & Use-Cases

| Persona | Description | Key Goals |
|---------|-------------|-----------|
| Curious First-Timer | Heard about SWE-bench and wants to try their model’s patches on the Lite split. | *Install & finish first run in <30 min with default settings* |
| Agent Developer | Iterates daily on an agent; needs to re-run 20 failing instances quickly. | *Subset selection, incremental caching, meaningful logs* |
| Research / CI Engineer | Runs nightly full-suite regression in GitHub Actions or a Slurm node. | *Deterministic, non-interactive, exit codes, HTML/JSON reports* |

## 4. Objectives & Success Metrics

### 4.1 Primary Objectives

1. **Zero-to-run time ≤ 30 min** on a fresh macOS laptop for Lite split.
2. **One-liner CLI** with sensible defaults (`swebench run --patches …`).
3. **Flexible targeting**: full, lite, verified, random N%, or regex/glob on `instance_id`.
4. **Self-healing UX**: common failures auto-detected with actionable suggestions.

### 4.2 Success Metrics

| Metric | Target (v1 GA) |
|--------|----------------|
| Time-to-first-green run (Lite) | ≤ 30 min on 8-core / 16 GB / 100 Mbps |
| Mean subsequent run time (Lite) | ≤ 8 min (with cache) |
| CLI adoption (PyPI weekly downloads) | 500+ within 3 months |
| GitHub issues labeled “user error” closed via self-help docs | 90 % |

## 5. Functional Requirements

1. **Installation**
   * `pip install swebench-runner` installs a <1 MB wheel with CLI.
   * First invocation pulls correct runner image (multi-arch) if missing.
2. **Dataset Management**
   * `--dataset lite|verified|full|/path/to/custom.jsonl`.
   * Automatic download & checksum verification from HuggingFace if builtin.
   * Dataset versions pinned to specific release in each CLI version
3. **Subset Selection**
   * `--subset "django/**"`, `--count 50 --sample random-seed`.
4. **Execution Pipeline**
   * Parallel workers default to CPU cores.
   * Progress bar with per-repo status.
   * Graceful cancellation (Ctrl-C once to finish current instance, twice to abort).
   * **Timeouts:**
     - `--timeout-mins N` per instance (default: 30, max: 120)
     - `--global-timeout-hours N` for entire run (default: none)
     - Timeout applies to full instance evaluation (patch + build + test)
     - On timeout: mark as `"status": "timeout"` and continue
   * **Concurrency Management:**
     - Auto-detect Docker daemon limits and adjust workers accordingly
     - Queue instances when at limit rather than failing
     - Clear warning when throttling: "Reduced to 5 workers due to Docker limits"
5. **Caching**
   * Docker layers & dataset files cached under `~/.swebench` (override via env).
   * Smart invalidation when image tag or harness version changes.
6. **Reporting**
   * Always emits `final_report.json` and `report.html` (single-page view).
   * Per-instance logs in `logs/{instance_id}/`.
   * Show last 50 lines of log on failure.
7. **Error Surface & Self-Healing**
   * Detect missing Docker, low disk space, or incompatible platform and print remedy.
   * On repo build/test failure show last 50 log lines + pointer to full log path.
8. **Benchmark Scope (v1)**
   * Must fully support all *SWE-bench* variants shipped by Princeton-NLP: `lite`, `verified`, `full`, and `multimodal`.
   * Code should route all SWE-bench logic through a `plugins/swebench.py` module so that **additional benchmarks can be registered later** without changing the CLI surface.
9. **CI/Headless Mode**
   * `--no-input` suppresses all prompts; fails fast if required values missing.
   * `--json` emits structured output to stdout for parsing.
   * Non-zero exit codes: 1=harness error, 2=Docker missing, 3=network failure, 4=disk full.
10. **Patch Format & Validation**
    * Accept JSONL with schema: `{"instance_id": "...", "patch": "diff --git..."}`.
    * Also support `--patches-dir /path/to/patches/` where each file is named `{instance_id}.patch`.
    * **Patch Requirements:**
      - Format: Unified diff format (`diff -u`) or Git format-patch
      - Encoding: UTF-8 only (reject patches with invalid encoding)
      - Size limit: 5MB per patch (configurable via `--max-patch-size`)
      - Binary files: Rejected by default; `--allow-binary` flag to override
      - **Multimodal Support**: For SWE-bench multimodal dataset:
        * Image files (PNG, JPG) in patches encoded as base64 in diff
        * Automatic conversion between base64 and binary during application
        * Size limit increases to 20MB for multimodal patches
        * Validation ensures image format matches expected types
    * **Validation Behavior:**
      - Pre-flight validation with `swebench validate --patches`
      - Check patch syntax, encoding, and size before execution
      - Warn on suspicious patterns: file deletions >1000 lines, permission changes, binary files
    * **Failed Patch Handling:**
      - When patch fails to apply cleanly: mark as `patch_failed` status in report
      - Continue to next instance (don't halt evaluation)
      - Log exact `git apply` error to `logs/{instance_id}/patch_error.log`
      - Summary in final report: `"status": "patch_failed", "reason": "Hunk #3 FAILED at 245"`
11. **Disk Management**
    * `swebench clean` removes old images, datasets, and logs (with `--dry-run` preview).
    * Pre-flight check warns if <50 GB free space for full dataset.
    * `--cache-dir` flag overrides default `~/.swebench`.
12. **Network Resilience**
    * Automatic retry with exponential backoff for dataset downloads and Docker pulls.
    * `--offline` mode uses only cached resources; fails if cache incomplete.
    * Honor `HTTP_PROXY`/`HTTPS_PROXY` environment variables.
13. **External Service Authentication**
    * **HuggingFace Hub:**
      - Support `--hf-token TOKEN` flag and `HUGGINGFACE_TOKEN` env var
      - Anonymous limit: 10 requests/hour → authenticated: 1000 requests/hour
      - Token used for dataset downloads from HuggingFace
      - Automatic retry with exponential backoff on 429 errors
    * **GitHub API (future):**
      - Reserved `--github-token` for potential repo cloning needs
    * **Security:**
      - Tokens never logged or included in reports
      - Stored in memory only, cleared after use
14. **Flaky Test Handling**
    * **Retry Mechanism:**
      - `--retry-failures N` flag (default: 0, max: 3) for power users only
    * **Flaky Detection:**
      - If instance passes on retry, mark as `"flaky": true` in report
    * **Determinism:**
      - Set fixed random seeds where possible: `PYTHONHASHSEED=0`
      - UTC timezone for all containers
      - Sort test discovery to ensure consistent ordering
15. **Smart Defaults & User Delight**
    * **Auto-detection:**
      - If `predictions.jsonl` exists in current directory, suggest using it
      - If patches file not specified, look for common names
    * **Success Celebration:**
      - ASCII art or motivational message on successful completion
      - First run success: "🎉 Congrats on your first run! Next try: `swebench run --subset 'django/**' --patches ...`"
    * **Progress Simplicity:**
      - Clean progress: "150/300 (50%) - 5 min remaining"
      - Emoji indicators for status: 🟢 passed, 🔴 failed, 🟡 running

## 6. Non-Functional Requirements

- **Deterministic & reproducible**: image digests, dataset SHAs, version-locked dependencies.
- **Platform support**: macOS (arm64/x86), Linux (x86/arm64).
- **Performance**: Completes Lite split on 8-core/16GB/SSD laptop in < 30 min.
- **Security**: Runner container runs as non-root user, read-only bind mounts for repo code, no `--privileged` flag.
- **Resource requirements**: Minimum 8 GB RAM (16 GB recommended), 50 GB free disk for Lite, 200 GB for full.
- **Python compatibility**: CLI supports Python 3.11+ (matches SWE-bench harness requirements).
- **Arm64 behavior**: Native arm64 images preferred; falls back to x86 emulation with performance warning.

## 7. Out-of-Scope (v1)

- Config files (`.swebench.yaml`)
- Dataset version management (`--dataset-version`)
- Log compression and retention policies
- Configurable log levels
- TUI dashboard
- Cloud/SaaS runner
- **Non–SWE-bench benchmarks** (e.g., SWE-rebench, Multi-SWE, BEHAVIOR). *The architecture must allow plug-ins, but UX polishing for those is deferred to v2.*

## 8. Key Design Decisions (resolved)

| Question | Option A | Option B | Decision & Rationale |
|----------|----------|----------|----------------------|
| **Docker image strategy** | Single *monolithic* image (~15 GB) containing Conda env and harness; simpler release pipeline. | *Per-repo* images (hundreds of small layers); faster per-instance startup but heavy registry maintenance. | **V1 = monolithic image** for simplicity and deterministic caching.  CLI will expose optional `prepare-images` sub-command in v1.1 to generate per-repo layers locally if users need speed. |
| **Image registry** | GitHub Container Registry (GHCR) under `swebench/runner`. Free, no pull limits for public images; integrates with GitHub Actions CI. | Docker Hub. More familiar to some users but subject to pull-rate throttling and arm/x86 multi-arch quirks. | **Use GHCR**; fall back to Docker Hub mirror only if corporate networks block GHCR. CLI flag `--registry` can override. |

---

## 9. Assumptions & Dependencies

- Docker available
- Public GHCR registry for images

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Arbitrary code execution** from malicious patches | High - could compromise user system | Run all repo code in Docker with strict security opts; patches applied inside container only; no host filesystem access beyond results dir |
| **Performance variance** across hardware | Medium - "30 min" target may be 2+ hours on slow machines | Clearly document reference hardware specs; provide `--benchmark` command to test user's system speed |
| **Docker Desktop licensing** for enterprise users | Medium - may block adoption | Document alternative: native Docker Engine on Linux, Colima on macOS |
| **Princeton namespace conflict** on GHCR | Low - may need different registry path | Start with our own namespace; coordinate with Princeton team for official handoff |
| **GPL dependencies** in runner image | Low - potential redistribution issues | Audit all dependencies; document licenses in image labels; consider Alpine base to minimize |
| **Large patch env var limits** | Low - Docker has 1MB env var limit | Automatic fallback to bind-mounted patch file when patch > 500KB |

---

## 11. Report Schema Specification

```json
{
  "version": "1.0",
  "run_id": "2025-07-16_14-50-11",
  "dataset": "swe-bench-lite",
  "started_at": "2025-07-16T14:50:11Z",
  "completed_at": "2025-07-16T15:02:44Z",
  "summary": {
    "total": 300,
    "resolved": 186,
    "failed": 89,
    "patch_failed": 20,
    "timeout": 5,
    "error": 0
  },
  "config": {
    "workers": 8,
    "timeout_mins": 30,
    "retry_failures": 1
  },
  "instances": [
    {
      "instance_id": "django__django-12345",
      "status": "resolved",  // resolved|failed|patch_failed|timeout|error
      "duration_seconds": 245,
      "attempts": 1,
      "flaky": false,
      "patch_applied": true,
      "tests_passed": true,
      "error_message": null,
      "log_path": "logs/django__django-12345/"
    }
  ]
}
```

---
*Draft created: <!-- date will be filled automatically by git commit -->*
