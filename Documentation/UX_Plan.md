# UX Plan – SWE-Bench Runner

> Document version 0.1
> Scope: v1 GA release covering all SWE-bench variants

---

## 1. High-level Philosophy

* "One clear command" is the primary mental model.
  *CLI must feel as straightforward as `pytest`.*
* Progressive disclosure – novices see only two flags, power-users can drill into 30+ options via `--help-advanced`.
* Safe by default – runs as non-root, write-only results dir, prompts before large downloads.

---

## 2. Actor Matrix

| Actor | Context | Typical Goal |
|-------|---------|--------------|
| First-timer | Laptop, first install | Evaluate GPT-4 patches on Lite split |
| Agent Dev | Local workstation | Re-run last 20 failed instances quickly |
| CI Engineer | GitHub Actions | Nightly full run with JSON artefact |
| Teaching Assistant | University lab | Run 5 random instances for grading |

---

## 3. Installation / Download Flow

### 3.1 Prerequisites Check

1. User types `pip install swebench-runner` (wheel <1 MB).
2. First CLI invocation triggers *bootstrap* script:
   1. Verify Docker daemon reachable (socket ping)
      *If not*: print ⛔ `Docker not detected. Install Docker Desktop or podman + docker.sock.` Exit 2.
   2. Detect platform (`linux/amd64`, `linux/arm64`).
   3. Confirm free disk ≥ 50 GB (configurable).
      *If not*: `Free disk space (12 GB) below required minimum (50 GB). Free space or run 'swebench clean'.` Exit 4.
3. Prompt:
   ```
   🎁 Welcome to SWE-bench Runner!

First-time setup will download ~14 GB of Docker layers.
This is a one-time download that enables lightning-fast evaluations.

Ready to get started? [Y/n]
   ```
   *`--yes` or CI mode skips prompt.*
4. Pull `ghcr.io/swebench/runner:<cli-version>-<arch>`.
5. Download selected dataset (default Lite) to `~/.swebench/datasets` (16 MB).  Retry ×3 w/ backoff.
6. Create `~/.swebench/config.toml` storing cache paths & last image digest.

### 3.2 Setup Wizard (first-time users)

If Docker not found, offer interactive help:
```
$ swebench setup
Checking prerequisites...
✅ Python 3.11.2
⚠️  Docker not found

Would you like instructions for:
1) macOS (Docker Desktop)
2) Ubuntu/Debian (Docker Engine)
3) Skip

Choice [1-3]: 1

📘 macOS Docker Desktop Setup:
1. Download from https://docker.com/products/docker-desktop
2. Install and start Docker Desktop
3. Wait for whale icon in menu bar
4. Run 'swebench setup' again to verify

Press Enter to open download page...
```

### 3.3 Upgrade Path

* User runs `pip install -U swebench-runner` → new CLI checks cached image tag mismatch → prompts to pull new image.
* `swebench upgrade --image-only` pulls latest runner image without touching CLI.

### 3.4 HuggingFace Authentication

First-time dataset download without token:
```
$ swebench run --patches gpt4_patches.jsonl
📚 Downloading swe-bench-lite from HuggingFace...
⚠️ Using anonymous access (10 downloads/hour limit)
💡 For faster downloads, set HUGGINGFACE_TOKEN or use --hf-token

[If rate limited:]
⛔ HuggingFace rate limit reached (anonymous quota: 10/hour)

Options:
1. Wait 45 minutes for quota reset
2. Get free token at https://huggingface.co/settings/tokens
3. Run: export HUGGINGFACE_TOKEN=hf_xxxxx

Retry with authentication? [Y/n]: Y
Enter HuggingFace token (or press Enter to wait): hf_xxxxx
✅ Authenticated! Downloading dataset...
```

---

## 4. Core Usage Flows

### 4.1 Quick Lite Evaluation (happy path)

```
$ swebench run --patches gpt4_patches.jsonl
```

For first-time users:
```
🎉 First run detected! Let me check your setup...
✅ Docker is running
✅ 75 GB disk space available
🎁 I'll download the runner image and dataset (one-time setup ~14GB)

Ready to start? [Y/n]: Y
```

Smart defaults in action:
```
💡 Found predictions.jsonl in current directory - using it!
📚 Dataset: swe-bench-lite ✓ (cached)
🐳 Runner image ready ✓
🚀 Evaluating 300 instances with 8 workers...

[▇▇▇▇▁] 195/300 (65%) • 🟢 122 passed • 🔴 73 failed • ⏱️ 4m remaining
```
At completion:
```
┌───────────────────────────┐
│   🎆 SUCCESS! 🎆       │
└───────────────────────────┘

🏆 Success Rate: 62.0% (186/300)
⏱️  Total Time: 12m 33s
📁 Results: ./results/latest/

🌐 View in browser:
   file:///path/to/results/latest/report.html

💡 Next steps:
   • Focus on failures: swebench run --rerun-failed ./results/latest
   • Try a subset: swebench run --subset "django/**" --patches ...
   • Share results: swebench share ./results/latest
```

### 4.2 Subsetting examples

| Goal | Command |
|------|---------|
| Only Django instances | `swebench run --patches predictions.jsonl --subset "django/**"` |
| Random 50 from verified | `swebench run -d verified --count 50 --sample random-seed=42` |
| Re-run failures | `swebench run --rerun-failed ./results/latest` |

### 4.3 CI Workflow

```yaml
- uses: actions/checkout@v4
- run: pip install swebench-runner==0.4.0
- run: swebench run --patches patches.jsonl --dataset full --no-input --json > report.json
- uses: actions/upload-artifact@v4
  with:
    name: swebench-report
    path: report.json
```
CLI exits non-zero if harness error; test failures still exit 0 (determined by report).

---

## 5. Configuration Surface

### 5.1 CLI Flags (core)

| Flag | Default | Description |
|------|---------|-------------|
| `--patches FILE` | **required** | JSONL or directory of `.patch` files |
| `--dataset {lite,verified,full,mm}|PATH` | `lite` | Which dataset or custom path |
| `--subset GLOB` | all | Glob/regex on `instance_id` |
| `--count N` | all | Evaluate first `N` instances after filtering |
| `--sample random-seed` | n/a | Random N% selection reproducible by seed |
| `--workers N` | CPU cores | Parallel workers |
| `--cache-dir DIR` | `~/.swebench` | Override cache root |
| `--no-input` | false | CI mode (fail on prompt) |
| `--json` | false | Emit JSON status to stdout |
| `--hf-token TOKEN` | env var | HuggingFace token for dataset downloads |
| `--retry-failures N` | 0 | Retry failed instances N times (max 3) |
| `--timeout-mins N` | 30 | Timeout per instance in minutes |
| `--max-patch-size MB` | 5 | Maximum patch size in megabytes |

### 5.2 Advanced Flags (hidden under `--help-advanced`)

`--registry`, `--image-tag`, `--offline`, `--pull-policy`, `--benchmark`, `--clean` (sub-command).


---

## 6. Expected Error Messages

| Code | Trigger | Example Message | Suggested Action |
|------|---------|-----------------|------------------|
| 2 | Docker not running | `⛔ Docker daemon unreachable at /var/run/docker.sock` | Start Docker Desktop or set `DOCKER_HOST` |
| 3 | Network failure after 3 retries | `⚠️ Unable to download dataset. Check internet or use --offline` | Retry or provide offline cache |
| 4 | Disk space < threshold | `⛔ Only 8 GB free; 50 GB required. Run 'swebench clean'.` | Free space or change cache dir |
| 5 | Invalid patch schema | `⛔ Line 12: missing 'patch' field` | Fix JSONL format |
| 6 | Patch too large for env var | `⛔ Patch 1.2 MB exceeds Docker env limit; use --patches-dir` | Move to directory mode |
| 7 | Unsupported arch image | `⛔ No arm64 image tag v0.4.0 found; falling back to x86 emulation (slow).` | Accept or specify older tag |
| 10 | Docker permission denied | `⛔ Docker permission denied. Run: sudo usermod -aG docker $USER && newgrp docker` | Add user to docker group |
| 11 | Docker Desktop stopped | `⛔ Docker Desktop not running. Start it from Applications and wait for whale icon.` | Start Docker Desktop |
| 13 | OOM during test | `⚠️ Instance django-123 killed (OOM). Increase Docker memory limit to 16GB.` | Adjust Docker resources |
| 14 | Patch conflict | `⚠️ Patch failed to apply cleanly. See logs/django-123/patch.log` | Manual patch fix needed |
| 15 | GHCR blocked | `⛔ Cannot reach ghcr.io. Set --registry docker.io/swebench/runner` | Use alternate registry |
| 16 | Git rate limit | `⚠️ GitHub rate limit hit. Retrying in 60s... (attempt 2/5)` | Wait or use token |
| 17 | Corrupted cache | `⛔ Dataset checksum mismatch. Run: swebench clean --datasets` | Clear bad cache |
| 18 | Stale image | `⚠️ Runner image 6 months old. Run: swebench upgrade` | Update image |
| 19 | No space during run | `⛔ Docker storage full. Clean with: docker system prune` | Free Docker space |
| 20 | Invalid Python | `⛔ Python 3.7 detected; 3.8+ required. Use pyenv or conda.` | Upgrade Python |
| 21 | Container timeout | `⚠️ Instance sklearn-456 timed out after 30 min` | Increase timeout or skip |
| 22 | Invalid instance ID | `⛔ Unknown instance IDs: fake-123, bad-456` | Check dataset for valid IDs |
| 23 | Patch encoding error | `⛔ Patch contains invalid UTF-8 at line 145` | Fix encoding to UTF-8 |
| 24 | Patch too large | `⛔ Patch size 7.2MB exceeds limit (5MB). Use --max-patch-size 10` | Increase limit or split patch |
| 25 | Binary patch | `⛔ Binary files not allowed: assets/logo.png. Use --allow-binary` | Remove binary changes |
| 26 | Patch apply failed | `⚠️ Patch failed: Hunk #3 FAILED at 245. See logs/django-123/patch_error.log` | Fix patch context |
| 27 | HF rate limit | `⚠️ HuggingFace rate limit (10/hour). Set --hf-token or wait 45 min` | Authenticate with token |
| 28 | Instance timeout | `⚠️ Instance django-123 timed out after 30 min. Use --timeout-mins 60` | Increase timeout |
| 29 | Flaky test detected | `⚠️ Instance passed on retry 2/3 (flaky test suspected)` | No action needed |
| 30 | Docker limit reached | `⚠️ Reduced to 5 workers due to Docker container limit (100)` | No action needed |

All errors include links to docs section `docs/troubleshooting#CODE`.

---

## 7. Debugging & Recovery Flows

### 7.1 Interrupted Run Recovery

```
$ swebench run --patches patches.jsonl
[▇▇▁▁▁] 40%  120/300 complete
^C
⚠️ Run interrupted. Resume with: swebench run --resume ./results/2025-07-16_14-50-11

$ swebench run --resume ./results/2025-07-16_14-50-11
📂 Resuming from instance 121/300...
[▁▁▇▇▇] 60%  180/300 complete
```

### 7.2 Debug Specific Instance

```
$ swebench debug django__django-12345
🐳 Starting debug container for django__django-12345...
📁 Repo mounted at: /testbed
📝 Patch available at: /tmp/patch.diff
🔧 Apply patch: git apply /tmp/patch.diff
🧪 Run tests: python -m pytest tests/
💡 Type 'exit' to quit

root@container:/testbed$
```

### 7.3 Pre-flight Validation

```
$ swebench validate --patches bad.jsonl
Validating patches file...
⛔ Line 5: Invalid diff format (missing diff header)
⛔ Line 12: Unknown instance_id 'fake-123'
⚠️ Line 18: Large patch (1.1 MB) may fail; consider --patches-dir
❌ Validation failed: 2 errors, 1 warning
```

### 7.4 Patch Validation Examples

Pre-run validation:
```
$ swebench validate --patches patches.jsonl
Validating 300 patches...
✅ 290 patches valid
⚠️  8 patches with warnings:
    - django-123: Large patch (4.8 MB)
    - flask-456: Contains file deletion (2000+ lines)
⛔  2 patches with errors:
    - numpy-789: Invalid UTF-8 encoding at line 234
    - scipy-012: Binary file change detected (data.pkl)

Run anyway with --force? [y/N]: n
```

Runtime patch failure:
```
🚀 Evaluating 300 instances...
⚠️ django-123: Patch failed to apply
   ↳ Hunk #3 FAILED at line 245
   ↳ Context mismatch suggests outdated patch
   ↳ Full error: logs/django-123/patch_error.log
[▇▇▇▁▁] 55% - continuing with remaining instances...
```

### 7.5 Test Retry Behavior

With retry enabled:
```
$ swebench run --patches patches.jsonl --retry-failures 2
🚀 Evaluating 300 instances...
⚠️ django-123 failed - retrying (attempt 2/3)...
✅ django-123 passed on retry (marked as flaky)
[▇▇▇▇▁] 80% - 15 instances retried, 8 now passing
```

Final report will show:
```json
{
  "instance_id": "django-123",
  "status": "resolved",
  "flaky": true,
  "attempts": 2,
  "duration_seconds": 487
}
```

### 7.6 Timeout Configuration

Per-instance timeout:
```
$ swebench run --patches patches.jsonl --timeout-mins 45
⚙️ Instance timeout: 45 minutes
⚠️ sklearn-789 timed out after 45 min
   ↳ Consider --timeout-mins 60 for ML repositories
```

Global timeout:
```
$ swebench run --patches patches.jsonl --global-timeout-hours 2
⏰ Global timeout: 2 hours (may evaluate ~150-200 instances)
⚠️ Global timeout reached after 2h - completed 187/300
```

### 7.7 Understanding Results

Enhanced report summary:
```
$ swebench summary results/latest
┌─────────────────────────────────────────┐
│ Run Summary: 2025-07-16_14-50-11        │
├─────────────────────────────────────────┤
│ Total:         300                      │
│ Resolved:      186 (62.0%)              │
│ Failed:         89 (29.7%)              │
│ Patch Failed:   20 (6.7%)               │
│ Timeout:         5 (1.7%)               │
│ Flaky Tests:     8 (2.7% of total)      │
├─────────────────────────────────────────┤
│ Runtime:       12m 33s                  │
│ Avg Duration:  2.5s/instance            │
└─────────────────────────────────────────┘

Top failures by category:
- Patch context mismatch: 12 instances
- Test assertion errors: 45 instances
- Import errors: 18 instances
- Timeouts: 5 instances (consider --timeout-mins 45)
```

---

## 8. Self-Help & Maintenance Commands

| Command | Purpose |
|---------|---------|
| `swebench doctor` | Prints env diagnostics, recent log tail, shareable bundle |
| `swebench clean [--images --datasets --logs]` | Deletes cached artefacts with progress & `--dry-run` |
| `swebench open-report PATH` | Open HTML or pretty console viewer |
| `swebench benchmark` | Measures repo-build performance to set expectations |
| `swebench validate --patches FILE` | Pre-check patch format before run |
| `swebench check-env` | Detailed prerequisites verification |
| `swebench status` | Show active evaluation containers |
| `swebench stop [--force]` | Gracefully stop running evaluation |

---

## 9. Environment Variables

Supported environment variables:
- `SWEBENCH_CACHE_DIR`: Override cache location (default: `~/.swebench`)
- `SWEBENCH_WORKERS`: Override worker count (default: CPU cores)
- `HUGGINGFACE_TOKEN`: Authentication for dataset downloads

---

## 10. Common Setup Issues

### 10.1 Docker Permission Errors (Linux)

```
$ swebench run --patches ...
⛔ Got permission denied while trying to connect to Docker daemon

Fix:
$ sudo usermod -aG docker $USER
$ newgrp docker  # or logout/login
```

### 10.2 Docker Resource Limits

Default Docker Desktop settings (2GB RAM) insufficient for many repos:
1. Open Docker Desktop → Settings → Resources
2. Set Memory: 16 GB, CPUs: 4+, Disk: 100 GB
3. Apply & Restart

### 10.3 Corporate Proxy Issues

```
$ export HTTP_PROXY=http://proxy.company.com:8080
$ export HTTPS_PROXY=$HTTP_PROXY
$ export NO_PROXY=localhost,127.0.0.1
$ swebench run --patches ... --registry docker.io/swebench
```

---

## 11. Accessibility & Internationalisation

* All prompts & messages plain UTF-8 text, no emoji required (CLI flag `--no-color` disables ANSI + emoji).
* Minimum terminal width: 80 columns.
* Screen reader friendly: structured output with `--json` flag.
* Localisation table planned for v2 (English only v1).

---

## 12. Future UX Considerations (deferred)

* `swebench ui` Textual-based dashboard for long-running runs.
* Support for other benchmarks via `plugins/<name>.py` once their images published.
* Cloud execution with progress streaming.

---
End of UX plan.
