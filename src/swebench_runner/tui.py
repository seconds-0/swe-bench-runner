from __future__ import annotations

import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from .docker_client import get_docker_client
from .docker_run import _persist_namespace, detect_platform
from .fs_abstraction import get_free_disk_gb
from .model_catalog import (
    ModelCatalogError,
    list_anthropic_models,
    list_openai_models,
    list_openrouter_models,
)
from .network_abstraction import check_general_connectivity, check_ghcr_access
from .secrets_store import (
    _env_var_name,
    _sanitize_secret,
    get_active_profile,
    get_api_key,
    get_key_source,
    get_keyring_backend_label,
    is_keychain_available,
    list_profiles,
    probe_keychain_usable,
    set_active_profile,
    set_api_key,
)

console = Console()
# Optional PTY support for reliable device-code capture
try:  # pragma: no cover - environment specific
    import fcntl  # type: ignore
    import pty  # type: ignore
    import select  # type: ignore
except Exception:  # pragma: no cover - environment specific
    pty = None  # type: ignore
    select = None  # type: ignore
    fcntl = None  # type: ignore
# Tuning knobs for setup/preflight runtime
_PREFLIGHT_TIMEOUT_S = int(os.getenv("SWEBENCH_TUI_PREFLIGHT_TIMEOUT", "60"))
_SETUP_TOTAL_BUDGET_S = int(os.getenv("SWEBENCH_TUI_SETUP_BUDGET", "180"))
_DOCKER_PING_TIMEOUT_S = int(os.getenv("SWEBENCH_TUI_DOCKER_PING_TIMEOUT", "8"))
_KEYCHAIN_PROBE_TIMEOUT_S = int(os.getenv("SWEBENCH_TUI_KEYCHAIN_TIMEOUT", "6"))

# Cached keychain probe result (ok, message)
_KC_PROBE_RESULT: tuple[bool, str] | None = None
def _ping_docker_quick() -> bool:
    """Bounded Docker availability check using docker CLI.

    Avoids potential hangs in docker-py by shelling out to a short 'docker info'.
    """
    try:
        proc = subprocess.run(  # noqa: S603
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=_DOCKER_PING_TIMEOUT_S,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _debug_log(msg: str) -> None:
    if os.getenv("SWEBENCH_DEBUG") == "1":
        try:
            console.print(f"[dim](debug) {msg}[/dim]")
            with open("tui-debug.log", "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        except Exception:
            pass



def _is_interactive() -> bool:
    """Return True if running in an interactive TTY session."""
    try:
        return (
            sys.stdin.isatty()
            and sys.stdout.isatty()
            and os.getenv("TERM", "").lower() != "dumb"
            and os.getenv("SWEBENCH_NO_INPUT") != "1"
        )
    except Exception:
        return False


def _has_ghcr_docker_auth() -> bool:
    """Best-effort check for GHCR auth in Docker config."""
    try:
        docker_config = os.getenv("DOCKER_CONFIG")
        cfg_path = (
            Path(docker_config) / "config.json"
            if docker_config else Path.home() / ".docker" / "config.json"
        )
        if not cfg_path.exists():
            return False
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        auths = data.get("auths", {})
        return any(str(k).strip().lower().startswith("ghcr.io") for k in auths.keys())
    except Exception:
        return False


def _gh_auth_status() -> tuple[bool, bool]:
    """Return (logged_in, has_read_packages_scope) using gh CLI if available."""
    gh_path = shutil.which("gh")
    if not gh_path:
        return False, False
    try:
        # Check login state first
        proc = subprocess.run(  # noqa: S603
            [gh_path, "auth", "status", "-h", "github.com"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        logged_in = proc.returncode == 0
        has_scope = False
        # Robust scope detection via API headers: x-oauth-scopes contains granted scopes
        try:
            hdr = subprocess.run(  # noqa: S603
                [gh_path, "api", "-i", "user", "-H", "Accept: application/vnd.github+json", "-h", "github.com"],
                capture_output=True,
                text=True,
                timeout=8,
            )
            header_text = hdr.stdout or ""
            for line in header_text.splitlines():
                if line.lower().startswith("x-oauth-scopes:"):
                    if "read:packages" in line.lower():
                        has_scope = True
                    break
        except Exception:
            # Fall back to parsing auth status output (less reliable)
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            for line in out.splitlines():
                if "token scopes:" in line.lower() and "read:packages" in line.lower():
                    has_scope = True
                    break
        return logged_in, has_scope
    except Exception:
        return False, False


def _gh_device_login_and_link() -> bool:
    """Run GitHub device login (prints code), then link GHCR to Docker.

    Returns True on success, False otherwise.
    """
    gh_path = shutil.which("gh")
    if not gh_path:
        return False
    try:
        code: str | None = None
        code_re = re.compile(r"\b([A-Z0-9]{4}-[A-Z0-9]{4})\b")
        started_at = time.monotonic()
        # Prefer PTY to ensure interactive output (device code appears reliably)
        if pty and select and fcntl:
            pid, master_fd = pty.fork()  # type: ignore[attr-defined]
            if pid == 0:  # Child process: exec gh device login
                try:
                    os.execvp(gh_path, [gh_path, "auth", "login", "--device", "-s", "read:packages", "-h", "github.com"])  # noqa: S606
                except Exception:
                    os._exit(1)
            # Parent: make PTY nonblocking
            try:
                flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)  # type: ignore[attr-defined]
                fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)  # type: ignore[attr-defined]
            except Exception:
                pass
            # Read PTY output up to 60s to capture code and echo guidance
            while time.monotonic() - started_at < 60:
                try:
                    rlist, _, _ = select.select([master_fd], [], [], 0.5)  # type: ignore[attr-defined]
                except Exception:
                    rlist = []
                if master_fd in rlist:
                    try:
                        chunk = os.read(master_fd, 1024).decode("utf-8", errors="ignore")
                    except Exception:
                        chunk = ""
                    for raw_line in chunk.splitlines():
                        line = raw_line.strip()
                        if line:
                            try:
                                console.print(f"[dim]{line}[/dim]")
                            except Exception:
                                pass
                        m = code_re.search(line)
                        if m:
                            code = m.group(1)
                            break
                    if code:
                        break
            # Leave child running while user authorizes; we'll terminate later
        else:
            # Fallback: pipe mode (less reliable, but works in many cases)
            proc = subprocess.Popen(  # noqa: S603
                [gh_path, "auth", "login", "--device", "-s", "read:packages", "-h", "github.com"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            while time.monotonic() - started_at < 60:
                line = proc.stdout.readline()
                if not line:
                    time.sleep(0.2)
                    continue
                text = line.strip()
                if text:
                    try:
                        console.print(f"[dim]{text}[/dim]")
                    except Exception:
                        pass
                m = code_re.search(line)
                if m:
                    code = m.group(1)
                    break
        if code:
            console.print(f"Copy this code and paste into GitHub device page: [bold]{code}[/bold]")
            try:
                webbrowser.open("https://github.com/login/device")
            except Exception:
                pass
        else:
            # Fallback guidance when code isn't printed (some gh versions/IO modes)
            console.print("[yellow]No device code detected from gh output.[/]")
            console.print("Run this in another terminal to get your code, then approve in the browser:")
            console.print("[dim]gh auth login --device -s read:packages -h github.com[/dim]")
            try:
                webbrowser.open("https://github.com/login/device")
            except Exception:
                pass
            try:
                Prompt.ask("Press Enter after authorizing on the device page", default="")
            except Exception:
                pass
        console.print("Waiting for device authorization to complete (up to 3 minutes)...")
        console.print("[dim]Tip: Approve the request in your browser. This window will continue automatically when finished.[/dim]")
        # Poll for scope rather than waiting on process (more reliable UX)
        deadline = time.monotonic() + 180
        authorized = False
        while time.monotonic() < deadline:
            logged_in, has_scope = _gh_auth_status()
            if logged_in and has_scope:
                authorized = True
                break
            time.sleep(3)
        # Clean up PTY child if we forked one
        if pty and select and fcntl:
            try:
                # Best-effort terminate child process
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
            try:
                os.close(master_fd)
            except Exception:
                pass
        if not authorized:
            return False
        # Link GHCR to Docker using gh token
        who = subprocess.run(  # noqa: S603
            [gh_path, "api", "user", "-q", ".login", "-h", "github.com"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        user = (who.stdout or "").strip() or os.getenv("USER", "gh")
        token = subprocess.run(  # noqa: S603
            [gh_path, "auth", "token", "-h", "github.com"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        subprocess.run(  # noqa: S603
            ["docker", "login", "ghcr.io", "-u", user, "--password-stdin"],
            input=token.stdout,
            text=True,
            timeout=20,
        )
        return _has_ghcr_docker_auth()
    except Exception:
        return False


def _gh_refresh_scope_read_packages() -> bool:
    """Attempt to refresh GitHub CLI token to include read:packages scope."""
    gh_path = shutil.which("gh")
    if not gh_path:
        return False
    try:
        rc = subprocess.run(  # noqa: S603
            [gh_path, "auth", "refresh", "-s", "read:packages", "-h", "github.com"],
            capture_output=True,
            text=True,
            timeout=60,
        ).returncode
        if rc != 0:
            return False
        # Re-check scopes via headers
        logged_in, has_scope = _gh_auth_status()
        return logged_in and has_scope
    except Exception:
        return False

def _env(name: str) -> str | None:
    v = os.getenv(name)
    return v if v and v.strip() else None


def pick_provider(allow_back: bool = False) -> str:
    """Render provider selector and return chosen provider or '__back__'."""
    console.print(Panel.fit("Select a provider", style="bold cyan"))
    table = Table(box=box.SIMPLE)
    table.add_column("#")
    table.add_column("Provider")
    providers = ["openai", "anthropic", "openrouter", "ollama"]
    start_idx = 1
    if allow_back:
        table.add_row("0", "Back")
        start_idx = 1
    for i, p in enumerate(providers, start_idx):
        table.add_row(str(i), p)
    console.print(table)
    while True:
        choice = IntPrompt.ask("Enter number", default=0 if allow_back else 1)
        if allow_back and choice == 0:
            return "__back__"
        if 1 <= choice <= len(providers):
            provider = providers[choice - 1]
            console.print(f"\n[bold]Selected:[/] {provider}\n")
            return provider
        console.print("Invalid choice. Try again.")


def _pick_profile(provider: str, allow_back: bool = False) -> str:
    current = get_active_profile(provider)
    profiles = list_profiles(provider)
    console.print(Panel.fit(f"Profiles for {provider}", style="bold cyan"))
    table = Table(box=box.SIMPLE)
    table.add_column("#")
    table.add_column("Option")
    idx = 1
    options: list[tuple[int, str, str]] = []
    if allow_back:
        options.append((0, "Back", "__back__"))
    if current:
        options.append((idx, f"Use active profile: {current}", current))
        idx += 1
    for p in profiles:
        if p == current:
            continue
        options.append((idx, f"Switch to: {p}", p))
        idx += 1
    options.append((idx, "New profile", "__new__"))
    idx += 1
    for num, _text, _ in options:
        table.add_row(str(num), _text)
    console.print(table)
    choice = IntPrompt.ask("Enter number", default=0 if allow_back else 1)
    for num, _text, value in options:
        if choice == num:
            if value == "__new__":
                name = Prompt.ask("Name this profile (e.g., personal, work)")
                set_active_profile(provider, name)
                return name
            if value == "__back__":
                return "__back__"
            set_active_profile(provider, value)
            return value
    # Fallback: set and use a named profile
    name = Prompt.ask("Name this profile (e.g., personal, work)", default="work")
    set_active_profile(provider, name)
    return name


def _validate_api_key(provider: str, api_key: str | None) -> bool:
    """Validate API key by attempting a lightweight catalog fetch."""
    # OpenRouter key is optional; accept None
    if provider == "openrouter" and not api_key:
        return True
    try:
        with console.status("[bold]Validating API key...[/]", spinner="dots"):
            if provider == "openai":
                list_openai_models(api_key or "")
            elif provider == "anthropic":
                list_anthropic_models(api_key or "")
            elif provider == "openrouter":
                list_openrouter_models(api_key)
            else:
                return True
        return True
    except ModelCatalogError:
        return False


def prompt_api_key(provider: str, allow_back: bool = False) -> str | None:
    """Prompt for API key with validation and optional back navigation."""
    # Offer to use an existing key/profile first
    existing = get_api_key(provider)
    if existing:
        active_prof = get_active_profile(provider) or "(default)"
        source = get_key_source(provider, active_prof)
        console.print(f"Found a saved key for this provider/profile. [dim](profile: {active_prof}, source: {source})[/]")
        use = Confirm.ask("Use the saved key?", default=True)
        if use:
            if _validate_api_key(provider, existing):
                return existing
            console.print("[red]Invalid or unauthorized API key.[/]")
            # proceed to re-enter
    profile = _pick_profile(provider, allow_back=allow_back)
    if profile == "__back__":
        return "__back__"
    if provider == "openai":
        console.print("[dim]Input will be hidden. Paste your key and press Enter.[/]")
        key = _env("OPENAI_API_KEY") or Prompt.ask(
            "Enter OpenAI API key (sk-...) or type '0' to go back",
            password=True,
        )
        if key == "0" and allow_back:
            return "__back__"
        key = _sanitize_secret(key)
        if key:
            try:
                console.print(f"Captured {len(key)} characters.")
            except Exception:
                pass
        if not _validate_api_key(provider, key):
            console.print("[red]Invalid API key. Please try again or go back.[/]")
            return prompt_api_key(provider, allow_back=allow_back)
        if key and Confirm.ask("Save securely to your OS keychain?", default=True):
            ok = set_api_key("openai", key, profile=profile)
            if not ok:
                console.print("[yellow]Failed to store in OS keychain. Will use for this session only.[/]")
        # Ensure availability in current session for selected profile
        try:
            env_name = _env_var_name("openai", profile)
            if env_name:
                os.environ[env_name] = key
            # Also set generic for broader compatibility
            os.environ["OPENAI_API_KEY"] = key
        except Exception:
            pass
        return key
    if provider == "anthropic":
        console.print("[dim]Input will be hidden. Paste your key and press Enter.[/]")
        key = _env("ANTHROPIC_API_KEY") or Prompt.ask(
            "Enter Anthropic API key (sk-ant-...) or type '0' to go back",
            password=True,
        )
        if key == "0" and allow_back:
            return "__back__"
        key = _sanitize_secret(key)
        if key:
            try:
                console.print(f"Captured {len(key)} characters.")
            except Exception:
                pass
        if not _validate_api_key(provider, key):
            console.print("[red]Invalid API key. Please try again or go back.[/]")
            return prompt_api_key(provider, allow_back=allow_back)
        if key and Confirm.ask("Save securely to your OS keychain?", default=True):
            ok = set_api_key("anthropic", key, profile=profile)
            if not ok:
                console.print("[yellow]Failed to store in OS keychain. Will use for this session only.[/]")
        # Ensure availability in current session for selected profile
        try:
            env_name = _env_var_name("anthropic", profile)
            if env_name:
                os.environ[env_name] = key
            os.environ["ANTHROPIC_API_KEY"] = key
        except Exception:
            pass
        return key
    if provider == "openrouter":
        console.print("[dim]Input is optional. Paste your key and press Enter (input hidden only when set).[/]")
        key = _env("OPENROUTER_API_KEY") or Prompt.ask(
            "Enter OpenRouter API key (optional), or '0' to go back",
            default="",
        )
        if key == "0" and allow_back:
            return "__back__"
        key = _sanitize_secret(key) if key else None
        try:
            console.print(f"Captured {len(key) if key else 0} characters.")
        except Exception:
            pass
        if not _validate_api_key(provider, key):
            console.print("[red]Invalid API key. Please try again or go back.[/]")
            return prompt_api_key(provider, allow_back=allow_back)
        if key and Confirm.ask("Save securely to your OS keychain?", default=True):
            ok = set_api_key("openrouter", key, profile=profile)
            if not ok:
                console.print("[yellow]Failed to store in OS keychain. Will use for this session only.[/]")
        # Ensure availability in current session for selected profile
        try:
            env_name = _env_var_name("openrouter", profile)
            if env_name and key:
                os.environ[env_name] = key
            if key:
                os.environ["OPENROUTER_API_KEY"] = key
        except Exception:
            pass
        return key or None
    return None


def fetch_models(provider: str, api_key: str | None) -> list[str]:
    """Fetch models for a provider using the supplied API key (best-effort)."""
    try:
        if provider == "openai":
            return list_openai_models(api_key or "")
        if provider == "anthropic":
            return list_anthropic_models(api_key or "")
        if provider == "openrouter":
            return list_openrouter_models(api_key)
        return []
    except ModelCatalogError as e:
        console.print(f"[red]Failed to fetch models:[/] {e}")
        return []


def pick_model(models: list[str], allow_back: bool = False) -> str:
    """Render model selector and return model id or '__back__'."""
    if not models:
        val = Prompt.ask("Enter model id (no catalog available) or '0' to go back")
        if allow_back and val == "0":
            return "__back__"
        return val

    def render(limit: int | None = 19) -> None:
        console.print(Panel.fit("Select a model", style="bold cyan"))
        table = Table(box=box.SIMPLE, show_edge=False)
        table.add_column("#", style="cyan")
        table.add_column("Model", style="green")
        shown = models if limit is None else models[:limit]
        start_idx = 1
        if allow_back:
            table.add_row("0", "Back")
        for i, m in enumerate(shown, start_idx):
            table.add_row(str(i), m)
        if limit is not None and len(models) > limit:
            table.add_row("20", "SHOW ALL (#)")
        console.print(table)

    all_mode = False
    render(limit=19)
    while True:
        choice = IntPrompt.ask("Enter number", default=0 if allow_back else 1)
        if allow_back and choice == 0:
            return "__back__"
        # SHOW ALL handler is only meaningful before all_mode
        if not all_mode and choice == 20 and len(models) > 19:
            all_mode = True
            render(limit=None)
            continue
        allowed_max = len(models) if all_mode else min(len(models), 19)
        if 1 <= choice <= allowed_max:
            return models[choice - 1]
        # Fallback to manual entry
        val = Prompt.ask("Enter model id or '0' to go back")
        if allow_back and val == "0":
            return "__back__"
        return val


def pick_dataset(allow_back: bool = False) -> str:
    """Render dataset selector and return dataset key or '__back__'."""
    console.print(Panel.fit("Pick dataset", style="bold cyan"))
    options = ["lite", "verified", "full"]
    table = Table(box=box.SIMPLE)
    table.add_column("#")
    table.add_column("Dataset")
    if allow_back:
        table.add_row("0", "Back")
    for i, d in enumerate(options, 1):
        table.add_row(str(i), d)
    console.print(table)
    choice = IntPrompt.ask("Enter number", default=0 if allow_back else 1)
    if allow_back and choice == 0:
        return "__back__"
    return options[choice - 1]


def pick_count(allow_back: bool = False) -> int | None:
    """Optionally limit run count; returns int, '__back__' or None."""
    if Confirm.ask("Do you want to limit the run to N instances?", default=True):
        val = Prompt.ask("How many instances? (enter number) or '0' to go back", default="5")
        if allow_back and val.strip() == "0":
            return "__back__"  # type: ignore[return-value]
        try:
            return int(val)
        except Exception:
            console.print("Please enter a valid integer or '0' to go back")
            return pick_count(allow_back=allow_back)
    return None


def _run_harness_preflight(namespace: str | None = None, timeout_s: int | None = None) -> tuple[bool, str]:
    """Run a minimal harness invocation to validate Docker/registry.

    Returns (ok, message).
    """
    _debug_log(f"preflight: start (ns={namespace or '(auto)'})")
    with tempfile.TemporaryDirectory() as td:
        pred_path = os.path.join(td, "predictions.jsonl")
        payload = {
            "instance_id": "sympy__sympy-20639",
            "model_name_or_path": "wizard-preflight",
            "model_patch": "diff --git a/test.py b/test.py\n--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-old\n+new",
        }
        with open(pred_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

        cmd = [
            sys.executable, "-m", "swebench.harness.run_evaluation",
            "--dataset_name", "SWE-bench_Lite",
            "--predictions_path", pred_path,
            "--max_workers", "1",
            "--run_id", "wizard-preflight",
            "--timeout", "120",
            "--cache_level", "env",
        ]
        if namespace:
            cmd += ["--namespace", namespace]
        try:
            eff_timeout = int(timeout_s if timeout_s is not None else _PREFLIGHT_TIMEOUT_S)
            _debug_log(f"preflight: running subprocess (timeout={eff_timeout}s)")
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=td,
                timeout=eff_timeout,  # keep setup fast
            )
        except Exception as e:
            _debug_log(f"preflight: exception starting harness: {e}")
            return False, f"Failed to start harness: {e}"

        stderr = (proc.stderr or "").lower()
        stdout = (proc.stdout or "")
        _debug_log(f"preflight: completed rc={proc.returncode}")
        if proc.returncode == 0:
            # Heuristic success: no obvious registry denial
            denied_terms = ("denied", "pull access denied")
            if any(t in stderr for t in denied_terms):
                _debug_log("preflight: rc=0 but denial terms detected in stderr")
                return False, stdout + "\n" + proc.stderr
            return True, "Preflight succeeded."
        # Auto-retry with GHCR if Docker Hub arm64 images missing
        hub_404 = ("swebench/sweb.eval.arm64" in stderr) and any(tok in stderr for tok in ["pull access denied", "repository does not exist", "not found"])
        if hub_404 and namespace is None:
            _debug_log("preflight: hub arm64 404 detected; retrying with GHCR")
            retry_cmd = list(cmd) + ["--namespace", "ghcr.io/epoch-research"]
            try:
                retry = subprocess.run(
                    retry_cmd,
                    capture_output=True,
                    text=True,
                    cwd=td,
                    timeout=eff_timeout,
                )
                _debug_log(f"preflight: ghcr retry rc={retry.returncode}")
                if retry.returncode == 0:
                    # Persist namespace only in interactive mode
                    os.environ["SWEBENCH_DOCKER_NAMESPACE"] = "ghcr.io/epoch-research"
                    if os.getenv("SWEBENCH_NO_INPUT") != "1":
                        try:
                            line = "SWEBENCH_DOCKER_NAMESPACE=ghcr.io/epoch-research"
                            existing = ""
                            try:
                                with open(".env", encoding="utf-8") as fr:
                                    existing = fr.read()
                            except Exception:
                                existing = ""
                            if line not in existing:
                                with open(".env", "a", encoding="utf-8") as f:
                                    f.write("\n" + line + "\n")
                        except Exception:
                            pass
                    return True, "Preflight succeeded via GHCR (namespace saved)."
                comb = (retry.stdout or "") + "\n" + (retry.stderr or "")
                if any(tok in comb.lower() for tok in ["denied", "unauthorized", "requires 'docker login'"]):
                    return False, comb + "\nHint: docker login ghcr.io"
                return False, comb
            except Exception as e:  # noqa: BLE001
                _debug_log(f"preflight: exception during ghcr retry: {e}")
                return False, f"Failed retry via GHCR: {e}"
        return False, stdout + "\n" + proc.stderr


def preflight_wizard() -> None:
    """Interactive preflight: resources, network, registry, and GHCR login hints."""
    console.print(Panel.fit("Preflight Check", style="bold cyan"))
    # Greenlight checklist (Python, Docker, Disk, Network, GHCR/arch)
    checks: list[tuple[str, bool, str]] = []
    # Python
    py_ok = sys.version_info >= (3, 10)
    checks.append(("Python", py_ok, f"{platform.python_version()}"))
    # Docker
    docker_ok = False
    try:
        client = get_docker_client()
        client.ping()
        docker_ok = True
    except Exception:
        docker_ok = False
    checks.append(("Docker", docker_ok, "running" if docker_ok else "not available"))
    # Disk space
    try:
        free_gb = get_free_disk_gb(".")
    except Exception:
        free_gb = 0.0
    disk_needed = 50.0
    disk_ok = free_gb >= disk_needed
    checks.append(("Disk", disk_ok, f"{free_gb:.1f} GB free (need {disk_needed:.0f}+)") )
    # Network
    net_ok = True
    try:
        check_general_connectivity()
    except Exception:
        net_ok = False
    checks.append(("Network", net_ok, "OK" if net_ok else "unreachable"))
    # GHCR Access (best-effort)
    ghcr_ok = True
    try:
        check_ghcr_access()
    except Exception:
        ghcr_ok = False
    checks.append(("GHCR", ghcr_ok, "reachable" if ghcr_ok else "blocked/denied"))
    # Arch & Namespace
    try:
        arch = detect_platform()
    except Exception:
        arch = "x86_64"
    ns = os.getenv("SWEBENCH_DOCKER_NAMESPACE")
    if ns == "":
        ns_display = "(local build)"
    elif ns is None:
        ns_display = "(default)"
    else:
        ns_display = ns
    checks.append(("Arch", True, arch))
    checks.append(("Namespace", True, ns_display))
    # Keychain availability
    keychain_ok = is_keychain_available()
    checks.append(("Keychain", keychain_ok, "available" if keychain_ok else "unavailable"))

    # Render checklist
    table = Table(box=box.SIMPLE)
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    for name, ok, info in checks:
        status = "[green]✓[/]" if ok else "[red]✗[/]"
        table.add_row(name, f"{status} {info}")
    console.print(table)

    # Quick hints if something is red
    if not py_ok:
        console.print("[yellow]Hint:[/] Use Python 3.11+ for best results.")
    if not docker_ok:
        if platform.system() == "Darwin":
            console.print("[yellow]Hint:[/] Start Docker Desktop and wait for the whale icon.")
        else:
            console.print("[yellow]Hint:[/] Ensure docker.sock is reachable (systemctl start docker).")
    if not disk_ok:
        console.print("[yellow]Hint:[/] Free up space or run: swebench clean --all")
    if not net_ok:
        console.print("[yellow]Hint:[/] Check your internet connection or proxy settings.")
    if not ghcr_ok:
        console.print("[yellow]Hint:[/] Corporate firewall may block ghcr.io; use a mirror or authenticate.")
    if not keychain_ok:
        if platform.system() == "Darwin":
            console.print("[yellow]Hint:[/] macOS Keychain should be available by default. If this persists, ensure the Python environment has 'keyring' installed and access to Keychain.")
        else:
            console.print("[yellow]Hint:[/] OS keychain unavailable. On Debian/Ubuntu: 'sudo apt install gnome-keyring' then log out/in. On Fedora: 'sudo dnf install gnome-keyring'. Alternatively, use environment variables.")

    # Optional early GHCR login for arm64 users to avoid denied errors later
    try:
        arch = detect_platform()
    except Exception:
        arch = "x86_64"
    if arch == "arm64" and not os.getenv("SWEBENCH_DOCKER_NAMESPACE"):
        # For ARM64, use empty namespace to trigger local builds
        _persist_namespace("")
        console.print("\n[bold cyan]━━━ ARM64 Architecture Detected (Apple Silicon) ━━━[/]")
        console.print("[yellow]ℹ️  SWE-bench will build Docker images locally for ARM64[/]")
        console.print("[dim]   • First build: 30-60+ minutes per repository[/]")
        console.print("[dim]   • Subsequent runs use cached images (fast)[/]")
        console.print("[dim]   • Requires ~120GB free disk space[/]")
        # Note: We don't return here - let preflight continue to run

    # GHCR authentication logic (skip for ARM64 local builds)
    # For ARM64 local builds, pretend we have GHCR auth to skip prompts
    if arch == "arm64" and not os.getenv("SWEBENCH_DOCKER_NAMESPACE"):
        docker_has_ghcr = True  # Skip GHCR auth for local builds
    else:
        # Smarter guidance: detect auth status and scope
        gh_logged_in, gh_has_scope = _gh_auth_status()
        docker_has_ghcr = _has_ghcr_docker_auth()
        # 1) If Docker already authed to GHCR, we're done (no prompts or verbose info)
        if docker_has_ghcr:
            console.print("[green]Docker is already authenticated to ghcr.io.[/]")
        else:
            console.print("You are on arm64. Some images are hosted on GHCR.")
            console.print("[dim]GHCR (GitHub Container Registry) hosts the official SWE-bench images.\n"
                          "Logging in lets Docker pull private or rate-limited images reliably.\n"
                          "We never store your token; Docker's credential store keeps it on your machine.[/]")
        # 2) Try to auto-link GHCR immediately if gh is present (most magical path)
        gh_path = shutil.which("gh")
        if gh_path:
            try:
                console.print("Linking GHCR to your GitHub token...")
                who = subprocess.run(  # noqa: S603
                    [gh_path, "api", "user", "-q", ".login", "-h", "github.com"],
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                user = (who.stdout or "").strip() or os.getenv("USER", "gh")
                token = subprocess.run(  # noqa: S603
                    [gh_path, "auth", "token", "-h", "github.com"],
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                subprocess.run(  # noqa: S603
                    ["docker", "login", "ghcr.io", "-u", user, "--password-stdin"],
                    input=token.stdout,
                    text=True,
                    timeout=20,
                )
            except Exception:
                pass
            docker_has_ghcr = _has_ghcr_docker_auth()
        # 3) If still not authed, handle scopes/logins only as needed
        if not docker_has_ghcr:
            gh_logged_in, gh_has_scope = _gh_auth_status()
            if gh_logged_in and not gh_has_scope:
                console.print("[yellow]GitHub CLI logged in, but token lacks read:packages. Attempting to refresh token scope...[/]")
                with console.status("Refreshing GitHub token scope (read:packages)...", spinner="dots"):
                    refreshed = _gh_refresh_scope_read_packages()
                if refreshed:
                    console.print("[green]Scope updated. Your token now includes read:packages.[/]")
                    # try linking again
                    _gh_device_login_and_link()  # safe no-op if already authed; reuses linking logic
                else:
                    console.print("[yellow]Could not refresh token scope automatically. Falling back to device login...[/]")
                    if _is_interactive():
                        ok_dev = _gh_device_login_and_link()
                        docker_has_ghcr = _has_ghcr_docker_auth()
                        gh_logged_in, gh_has_scope = _gh_auth_status()
                        if ok_dev and gh_logged_in and gh_has_scope and docker_has_ghcr:
                            console.print("[green]Device login completed and GHCR linked successfully.[/]")
                        else:
                            console.print("[yellow]Device login did not complete. You can run 'gh auth login --web -s read:packages' in another terminal.[/]")
            elif gh_logged_in and gh_has_scope:
                # Scope ok but still not authed: one more linking attempt
                _gh_device_login_and_link()
            else:
                # gh missing or not logged in; will offer manual helpers below
                pass
        # Final one-line status for clarity
        gh_logged_in, gh_has_scope = _gh_auth_status()
        docker_has_ghcr = _has_ghcr_docker_auth()
        status_line = (
            f"GitHub: {'logged-in' if gh_logged_in else 'not logged-in'} | "
            f"read:packages: {'yes' if gh_has_scope else 'no/unknown'} | "
            f"Docker GHCR auth: {'yes' if docker_has_ghcr else 'no'}"
        )
        console.print(f"[dim]{status_line}[/dim]")
        # Offer manual helpers only if Docker still lacks GHCR auth
        if _is_interactive() and not docker_has_ghcr:
            try:
                if Confirm.ask("Open GitHub login page in your browser now?", default=False):
                    webbrowser.open("https://github.com/login")
                if Confirm.ask("Open device login page (for codes)?", default=False):
                    webbrowser.open("https://github.com/login/device")
                if Confirm.ask("Open token creation page with read:packages scope?", default=False):
                    webbrowser.open("https://github.com/settings/tokens/new?scopes=read:packages")
            except Exception:
                pass
        gh_path = shutil.which("gh")
        # Only offer interactive login helpers if Docker still lacks GHCR auth
        if gh_path and _is_interactive() and not docker_has_ghcr and Confirm.ask("Login to GitHub (web/device) and auto-link GHCR?", default=True):
            # GitHub CLI web flow with bounded timeouts and clear fallback guidance
            console.print("Starting GitHub web login (read:packages)...")
            try:
                # First check status quickly
                status = subprocess.run(  # noqa: S603
                    ["gh", "auth", "status"],
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                if status.returncode != 0:
                    # Attempt web login (opens browser). Bound to 120s to avoid indefinite waits.
                    # Try web flow first, else fall back to device flow with code printing
                    rc = subprocess.run(  # noqa: S603
                        ["gh", "auth", "login", "--web", "-s", "read:packages"],
                        timeout=120,
                    ).returncode
                    if rc != 0:
                        console.print("[yellow]Web login failed or was cancelled. Falling back to device flow...[/]")
                        if not _gh_device_login_and_link():
                            console.print("[red]Device login failed. You can run 'gh auth login --device -s read:packages' in another terminal.[/]")
            except Exception:
                # Fallback: provide manual instructions and optional token page
                console.print("[yellow]Could not complete GitHub web login automatically.[/]")
                console.print("Run manually in another terminal: [dim]gh auth login --web -s read:packages[/dim]")
                if Confirm.ask("Open the GitHub token page in your browser?", default=False):
                    try:
                        webbrowser.open("https://github.com/settings/tokens")
                    except Exception:
                        pass
            # Link GHCR with the GitHub token (best-effort; bounded timeout)
            console.print("Linking GHCR to your GitHub token...")
            try:
                who = subprocess.run(  # noqa: S603
                    ["gh", "api", "user", "-q", ".login"],
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                user = (who.stdout or "").strip() or os.getenv("USER", "gh")
                token = subprocess.run(  # noqa: S603
                    ["gh", "auth", "token"],
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                subprocess.run(  # noqa: S603
                    ["docker", "login", "ghcr.io", "-u", user, "--password-stdin"],
                    input=token.stdout,
                    text=True,
                    timeout=20,
                )
            except Exception:
                console.print("[yellow]GHCR linking skipped (could not retrieve token automatically).[/]")
        elif not gh_path and not docker_has_ghcr and Confirm.ask("Install GitHub CLI for browser login?", default=False):
            if platform.system() == "Darwin":
                os.system("brew install gh")
            elif platform.system() == "Linux":
                os.system("sudo apt update && sudo apt install -y gh")
            else:
                console.print("Please install GitHub CLI from https://cli.github.com/")
        elif not docker_has_ghcr and Confirm.ask("Authenticate with GHCR now via docker login?", default=False):
            console.print("Launching docker login... (press Ctrl+C to cancel)")
            os.system("docker login ghcr.io")
        elif not docker_has_ghcr:
            # Offer to open PAT page as a fallback (only if not authenticated)
            console.print("You can create a GitHub Personal Access Token (PAT) with read:packages and use it with 'docker login ghcr.io'.")
            if Confirm.ask("Open token page in your browser?", default=False):
                try:
                    webbrowser.open("https://github.com/settings/tokens")
                except Exception:
                    pass
    # Skip preflight prompt if already authenticated
    preflight_failed = False
    if docker_has_ghcr:
        console.print("✅ Docker is authenticated to GHCR. Skipping additional setup prompts.")
        # For ARM64 with local builds, we still want to run preflight but not prompt
        # Animated spinner while preflight runs
        with console.status("[bold]Running preflight...[/]", spinner="dots") as status:  # type: Status
            ok, msg = _run_harness_preflight(timeout_s=_PREFLIGHT_TIMEOUT_S)
        if ok:
            console.print("✅ Preflight succeeded. You're good to go.\n")
            # Also show provider key sources for quick visibility
            try:
                prov_table = Table(box=box.SIMPLE)
                prov_table.add_column("Provider", style="cyan")
                prov_table.add_column("Active Profile")
                prov_table.add_column("Key Source")
                for prov in ["openai", "anthropic", "openrouter", "ollama"]:
                    prof = get_active_profile(prov) or "(default)"
                    source = get_key_source(prov, prof)
                    prov_table.add_row(prov, prof, source)
                console.print(Panel(prov_table, title="Provider Keys", style="green"))
            except Exception:
                pass
            # Let the wizard continue to Summary (don't return)
        else:
            # Preflight failed, show error and continue to offer fixes below
            preflight_failed = True
            console.print("❌ Preflight failed. Details:")
            console.print(Panel.fit(msg[-1200:], title="harness output (tail)", style="red"))
    else:
        if not Confirm.ask("Run a quick preflight to validate Docker & registry?", default=True):
            return

        # Animated spinner while preflight runs
        with console.status("[bold]Running preflight...[/]", spinner="dots") as status:  # type: Status
            ok, msg = _run_harness_preflight(timeout_s=_PREFLIGHT_TIMEOUT_S)
        if ok:
            console.print("✅ Preflight succeeded. You're good to go.\n")
            # Also show provider key sources for quick visibility
            try:
                prov_table = Table(box=box.SIMPLE)
                prov_table.add_column("Provider", style="cyan")
                prov_table.add_column("Active Profile")
                prov_table.add_column("Key Source")
                for prov in ["openai", "anthropic", "openrouter", "ollama"]:
                    prof = get_active_profile(prov) or "(default)"
                    source = get_key_source(prov, prof)
                    prov_table.add_row(prov, prof, source)
                console.print(Panel(prov_table, title="Provider Keys", style="green"))
            except Exception:
                pass
            # Let the wizard continue to Summary (don't return)
        else:
            preflight_failed = True
            # Show error details below

    # Only show error details and offer fixes if preflight failed
    if preflight_failed:
        console.print("❌ Preflight failed. Details:")
        console.print(Panel.fit(msg[-1200:], title="harness output (tail)", style="red"))

        # Offer fixes
        console.print("Possible fixes:")
        console.print("  • Run: docker login ghcr.io (if your registry requires auth)")
        console.print("  • Use an alternate namespace/mirror (Advanced)")
        console.print("  • Switch to a local model or try emulation if arch images are restricted")

        if Confirm.ask("Have you run docker login or want to try a custom namespace now?", default=True):
            if Confirm.ask("Run 'docker login ghcr.io' now?", default=False):
                os.system("docker login ghcr.io")
            ns = Prompt.ask("Enter namespace (e.g., ghcr.io/epoch-research) or leave blank to skip", default="")
            ns = ns.strip() or None
            with console.status("[bold]Retrying preflight...[/]", spinner="dots") as status:  # type: Status
                ok2, msg2 = _run_harness_preflight(namespace=ns, timeout_s=_PREFLIGHT_TIMEOUT_S)
            if ok2:
                console.print("✅ Preflight succeeded with your settings.\n")
                # Persist namespace if provided
                if ns:
                    _write_env_lines([f"SWEBENCH_DOCKER_NAMESPACE={ns}"])
                    console.print(f"Saved SWEBENCH_DOCKER_NAMESPACE to .env ({ns})")
                # Let wizard continue after successful retry
            else:
                console.print("❌ Preflight still failing. Details:")
                console.print(Panel.fit(msg2[-1200:], title="harness output (tail)", style="red"))
                console.print("You can proceed, but runs may fail until registry/arch issues are resolved.\n")
        else:
            console.print("Skipping preflight.\n")


def run_wizard() -> None:
    """Guided setup: provider, key, model, dataset, count, and optional preflight."""
    console.print(Panel.fit("🚀 SWE-bench Runner Setup", style="bold magenta"))

    provider: str | None = None
    api_key: str | None = None
    model: str | None = None
    dataset: str | None = None
    count: int | None = None

    step = 0
    while True:
        if step == 0:
            sel = pick_provider(allow_back=False)
            if sel == "__back__":
                # no-op; cannot go back from first step
                continue
            provider = sel
            step = 1
            continue
        if step == 1:
            key = prompt_api_key(provider, allow_back=True)  # type: ignore[arg-type]
            if key == "__back__":
                step = 0
                continue
            api_key = key
            step = 2
            continue
        if step == 2:
            models = fetch_models(provider, api_key)  # type: ignore[arg-type]
            sel_model = pick_model(models, allow_back=True)
            if sel_model == "__back__":
                step = 1
                continue
            model = sel_model
            step = 3
            continue
        if step == 3:
            ds = pick_dataset(allow_back=True)
            if ds == "__back__":
                step = 2
                continue
            dataset = ds
            step = 4
            continue
        if step == 4:
            c = pick_count(allow_back=True)
            if c == "__back__":  # type: ignore[comparison-overlap]
                step = 3
                continue
            count = c  # type: ignore[assignment]
            # Preflight and summary
            preflight_wizard()

            # Persist to .env (non-sensitive only)
            env_lines = [
                f"SWEBENCH_PROVIDER={provider}",
                f"SWEBENCH_MODEL={model}",
            ]
            _write_env_lines(env_lines)

            console.print(Panel.fit("Summary", style="bold green"))
            cmd = ["swebench", "run", "-d", dataset]
            if count:
                cmd += ["--count", str(count)]
            console.print("Command: " + " ".join(cmd))
            if Confirm.ask("Run now?", default=True):
                console.print("\n[cyan]Starting evaluation...[/cyan]\n")
                # Import and call evaluate function directly instead of subprocess
                from .cli import evaluate
                try:
                    evaluate(
                        patches=None,
                        patches_dir=None,
                        dataset=dataset,
                        count=count,
                        subset=None,
                        timeout=60,  # Default timeout
                        workers=None,  # Auto-detect
                        no_cache=False,
                        no_preflight=False,
                        output_dir=None,
                        provider=None,  # Use environment variables
                        model=None,    # Use environment variables
                        temperature=None,
                        max_tokens=None
                    )
                except SystemExit as e:
                    sys.exit(e.code)
            else:
                console.print("\nYou can run the command above anytime.")
            return


def _render_header(title: str) -> None:
    width = console.size.width if console.size.width else 80
    line = "-" * width
    mid = f"*** {title} ***"
    padded = mid.center(width)
    console.print(line)
    console.print(f"[bold]{padded}[/bold]")
    console.print(line)


def run_auto_setup() -> bool:
    """Run automatic first-time setup with progress UI.

    Returns True if all critical checks pass; False otherwise.
    """
    # Ensure we can update the cached keychain probe result from any branch
    global _KC_PROBE_RESULT  # noqa: PLW0603
    _debug_log(
        "auto-setup: start "
        f"(pref={_PREFLIGHT_TIMEOUT_S}s, setup_budget={_SETUP_TOTAL_BUDGET_S}s, "
        f"docker_ping={_DOCKER_PING_TIMEOUT_S}s, keychain={_KEYCHAIN_PROBE_TIMEOUT_S}s)"
    )
    console.print(Panel.fit("Automatic Setup", style="bold magenta"))
    ok = True
    details: list[str] = []
    debug = os.getenv("SWEBENCH_DEBUG") == "1"
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as progress:
        # Step 1: Docker
        t0_start = time.monotonic()
        # Let users know the expected time up-front
        console.print(
            f"[dim]Expected runtime: ~30–90s (hard cap {_SETUP_TOTAL_BUDGET_S//60} min). You can cancel anytime with Ctrl+C.[/dim]"
        )
        t1 = progress.add_task("Checking Docker daemon...", total=None)
        docker_ok = False
        # Fast, bounded check using docker CLI, then confirm with SDK if quick check passes
        if debug:
            _debug_log("docker quick ping start")
        if _ping_docker_quick():
            try:
                client = get_docker_client()
                client.ping()
                docker_ok = True
                if debug:
                    _debug_log("docker sdk ping ok")
            except Exception:
                docker_ok = False
                if debug:
                    _debug_log("docker sdk ping failed")
        elif debug:
            _debug_log("docker quick ping failed")
        if not docker_ok:
            ok = False
            sys_plat = platform.system()
            if sys_plat == "Darwin":
                details.append("Docker Desktop not running. Start it and wait for the whale icon.")
            else:
                details.append("Docker daemon not reachable. Start service (systemctl start docker).")
        _debug_log(f"auto-setup: docker_ok={docker_ok}")
        progress.update(t1, completed=1)

        # Step 2: Network & GHCR reachability (non-fatal unless GHCR will be required)
        t2 = progress.add_task("Checking network connectivity...", total=None)
        net_ok = True
        try:
            check_general_connectivity()
            if debug:
                _debug_log("general network ok")
        except Exception:
            net_ok = False
            ok = False
            details.append("Network unreachable. Check your internet/proxy settings.")
        ghcr_ok = True
        try:
            check_ghcr_access()
            if debug:
                _debug_log("ghcr reachable")
        except Exception:
            ghcr_ok = False
            details.append("Cannot reach ghcr.io (will prefer Docker Hub if possible).")
        _debug_log(f"auto-setup: net_ok={net_ok} ghcr_ok={ghcr_ok}")
        progress.update(t2, completed=1)

        # Budget check before potentially long probes
        if time.monotonic() - t0_start > _SETUP_TOTAL_BUDGET_S:
            details.append("Setup exceeded time budget; skipped registry probe.")
            _debug_log("auto-setup: time budget exceeded; skipping registry probe")
            # Mark remaining tasks to keep UI consistent
            t3 = progress.add_task("Probing best registry...", total=None)
            progress.update(t3, completed=1)
            t4 = progress.add_task("Checking disk space...", total=None)
            try:
                free_gb = get_free_disk_gb(".")
                if free_gb < 50.0:
                    ok = False
                    details.append(f"Low disk space: {free_gb:.1f}GB free (need 50GB+). Run 'swebench clean --all'.")
            except Exception:
                pass
            else:
                _debug_log(f"auto-setup: free_gb={free_gb:.1f}")
            progress.update(t4, completed=1)
            t5 = progress.add_task("Checking key storage...", total=None)
            kc_ok, kc_msg = probe_keychain_usable()
            _KC_PROBE_RESULT = (kc_ok, kc_msg)
            _debug_log(f"auto-setup: keychain usable={kc_ok} msg={kc_msg[:80] if kc_msg else ''}")
            if not kc_ok:
                sys_plat = platform.system()
                if sys_plat == "Darwin":
                    details.append("Keychain unavailable: run from a GUI Terminal and 'Always Allow'. Using env vars this session.")
                else:
                    details.append("Secret Service unavailable: install and unlock a keyring. Using env vars this session.")
            progress.update(t5, completed=1)
            # Short-circuit summary
            _debug_log(f"auto-setup: end early ok={ok}")
            return ok

        # Step 3: Registry autodetect via harness preflight (sets namespace if needed)
        t3 = progress.add_task("Probing best registry...", total=None)
        preflight_ok = False
        if docker_ok and net_ok:
            if debug:
                _debug_log("preflight start (auto)")
            ok_pf, msg_pf = _run_harness_preflight()
            if ok_pf:
                preflight_ok = True
                if debug:
                    _debug_log("preflight ok")
            else:
                # Try GHCR explicitly first (common for arm64), then Docker Hub
                if debug:
                    _debug_log("preflight ghcr")
                ok_ghcr, msg_ghcr = _run_harness_preflight(namespace="ghcr.io/epoch-research")
                if ok_ghcr:
                    preflight_ok = True
                    _persist_namespace("ghcr.io/epoch-research")
                    if debug:
                        _debug_log("preflight ghcr ok")
                else:
                    if debug:
                        _debug_log("preflight hub")
                    ok_hub, msg_hub = _run_harness_preflight(namespace="docker.io/swebench")
                    if ok_hub:
                        preflight_ok = True
                        _persist_namespace("docker.io/swebench")
                        if debug:
                            _debug_log("preflight hub ok")
                    else:
                        # Leave unset; report tail
                        details.append("Registry probe failed; will build locally or retry during run.")
                        if debug:
                            _debug_log("preflight all failed")
        else:
            details.append("Skipped registry probe due to earlier failures.")
        _debug_log(f"auto-setup: preflight_ok={preflight_ok}")
        progress.update(t3, completed=1)

        # Step 4: Disk space (warn only)
        t4 = progress.add_task("Checking disk space...", total=None)
        try:
            free_gb = get_free_disk_gb(".")
            if free_gb < 50.0:
                ok = False
                details.append(f"Low disk space: {free_gb:.1f}GB free (need 50GB+). Run 'swebench clean --all'.")
        except Exception:
            pass
        else:
            _debug_log(f"auto-setup: free_gb={free_gb:.1f}")
        progress.update(t4, completed=1)

        # Step 5: Key storage (probe; warn on fallback)
        t5 = progress.add_task("Checking key storage...", total=None)
        # Bound the keychain probe with a watchdog to avoid any potential backend hangs
        kc_ok = False
        kc_msg = ""
        def _probe():
            nonlocal kc_ok, kc_msg
            try:
                kc_ok, kc_msg = probe_keychain_usable()
            except Exception as e:  # noqa: BLE001
                kc_ok, kc_msg = False, str(e)

        thread = threading.Thread(target=_probe, daemon=True)
        thread.start()
        thread.join(timeout=_KEYCHAIN_PROBE_TIMEOUT_S)
        if thread.is_alive():
            _debug_log("keychain probe timed out")
            kc_ok, kc_msg = False, "Keychain probe timed out"
        # Cache for later banner use
        _KC_PROBE_RESULT = (kc_ok, kc_msg)
        _debug_log(f"auto-setup: keychain usable={kc_ok} msg={kc_msg[:80] if kc_msg else ''}")
        if not kc_ok:
            sys_plat = platform.system()
            if sys_plat == "Darwin":
                details.append("Keychain unavailable: run from a GUI Terminal and 'Always Allow'. Using env vars this session.")
            else:
                details.append("Secret Service unavailable: install and unlock a keyring. Using env vars this session.")
                # Offer thorough Linux guidance
                if _is_interactive():
                    try:
                        if Confirm.ask("Show Linux Secret Service setup steps?", default=True):
                            guide_lines = [
                                "Ubuntu/Debian:",
                                "  sudo apt update && sudo apt install -y gnome-keyring libsecret-1-0 dbus-user-session",
                                "  Log out and log back in to start a user session that unlocks the keyring.",
                                "  Headless shell (temporary): eval $(gnome-keyring-daemon --start --components=secrets)",
                                "Fedora:",
                                "  sudo dnf install -y gnome-keyring libsecret",
                                "KDE/KWallet:",
                                "  Install kwalletmanager and enable the 'Secret Service' (org.freedesktop.secrets) plugin.",
                                "Verify:",
                                "  python -c 'import keyring; print(keyring.get_keyring())'",
                            ]
                            console.print(Panel.fit("\n".join(guide_lines), title="Linux Key Storage Help", style="cyan"))
                    except Exception:
                        pass
            # Offer to install python-keyring if missing
            if "not installed" in (kc_msg or "").lower():
                if _is_interactive():
                    try:
                        if Confirm.ask("Install python-keyring now?", default=False):
                            with console.status("Installing keyring...", spinner="dots"):
                                proc = subprocess.run(  # noqa: S603
                                    [sys.executable, "-m", "pip", "install", "keyring"],
                                    capture_output=True,
                                    text=True,
                                    timeout=120,
                                )
                            if proc.returncode == 0:
                                # Re-probe
                                kc_ok2, kc_msg2 = probe_keychain_usable()
                                _KC_PROBE_RESULT = (kc_ok2, kc_msg2)
                                if kc_ok2:
                                    console.print("[green]Installed keyring successfully; key storage is now available.[/]")
                                    # Remove previous warning, add success note
                                    details.append("Key storage enabled via python-keyring.")
                                else:
                                    console.print("[yellow]Keyring installed but backend still unavailable. Using env vars this session.[/]")
                            else:
                                console.print("[red]Failed to install keyring. Using env vars this session.[/]")
                    except Exception:
                        console.print("[red]Installation attempt failed. Please run: pip install keyring[/]")
        progress.update(t5, completed=1)

    # Summary
    if ok:
        console.print("[green]✅ Environment ready![/]")
    else:
        console.print("[yellow]⚠️  Setup completed with issues:[/]")
        for d in details:
            console.print(f" - {d}")
    _debug_log(f"auto-setup: end ok={ok} details={len(details)}")
    return ok

def run_home() -> None:
    """Home screen: Quick Run / Guided Setup / Help & Docs.

    Loops so that returning from sub-menus (e.g., Options) brings users back here.
    Uses saved defaults when available.
    """
    while True:
        console.clear()
        # Minimal, bold full-width header
        _render_header("SWE BENCH RUNNER")
        console.print("Welcome! What would you like to do?\n")

        # Early keychain probe (show once per process) to set expectations when secrets backend isn't usable
        if not getattr(run_home, "_kc_banner_shown", False):
            # Allow disabling automatic setup via env (useful to isolate hangs)
            if os.getenv("SWEBENCH_TUI_DISABLE_AUTO") == "1":
                _debug_log("auto-setup: disabled via SWEBENCH_TUI_DISABLE_AUTO=1")
            else:
                # Only run auto-setup in an interactive TTY to avoid blocking in non-interactive shells
                if _is_interactive():
                    run_auto_setup()
                else:
                    _debug_log("auto-setup: skipped (non-interactive session)")
            # Reuse cached result if available to avoid double prompt
            global _KC_PROBE_RESULT  # noqa: PLW0603
            if _KC_PROBE_RESULT is not None:
                kc_ok, kc_msg = _KC_PROBE_RESULT
            else:
                kc_ok, kc_msg = probe_keychain_usable()
            if not kc_ok:
                backend = get_keyring_backend_label()
                reason = (kc_msg or "unavailable")[:120]
                if platform.system() == "Darwin":
                    help_text = (
                        "macOS: run from a GUI Terminal to allow Keychain prompts, then 'Always Allow'.\n"
                        "We will use environment variables for this session."
                    )
                else:
                    help_text = (
                        "Linux: enable Secret Service (e.g., gnome-keyring/KWallet) and log in to unlock it.\n"
                        "We will use environment variables for this session."
                    )
                console.print(Panel.fit(
                    f"Key storage backend: {backend}\n"
                    f"Reason: {reason}\n"
                    f"{help_text}",
                    title="Key Storage",
                    style="yellow"
                ))
                # Offer inline install for python-keyring on macOS when not installed
                try:
                    if (
                        platform.system() == "Darwin"
                        and _is_interactive()
                        and "not installed" in (kc_msg or "").lower()
                    ):
                        if Confirm.ask("Install python-keyring now?", default=True):
                            with console.status("Installing keyring...", spinner="dots"):
                                proc = subprocess.run(  # noqa: S603
                                    [sys.executable, "-m", "pip", "install", "keyring"],
                                    capture_output=True,
                                    text=True,
                                    timeout=180,
                                )
                            if proc.returncode == 0:
                                # Re-probe and update cached result
                                kc_ok2, kc_msg2 = probe_keychain_usable()
                                # Update global cache (assign a local tuple first to avoid global-before-assignment issues)
                                new_probe = (kc_ok2, kc_msg2)
                                try:
                                    globals()["_KC_PROBE_RESULT"] = new_probe
                                except Exception:
                                    pass
                                if kc_ok2:
                                    console.print("[green]Installed keyring successfully; Keychain is now available.[/]")
                                else:
                                    console.print("[yellow]Keyring installed but backend still unavailable. Using env vars this session.[/]")
                            else:
                                console.print("[red]Failed to install keyring. You can run: pip install keyring[/]")
                    # Even if keyring is installed but backend is still unavailable, offer to trigger a Keychain prompt now
                    if platform.system() == "Darwin" and _is_interactive() and not kc_ok:
                        if Confirm.ask("Trigger macOS Keychain prompt now (test set/get)?", default=True):
                            with console.status("Testing Keychain...", spinner="dots"):
                                ok3, msg3 = probe_keychain_usable()
                            try:
                                globals()["_KC_PROBE_RESULT"] = (ok3, msg3)
                            except Exception:
                                pass
                            if ok3:
                                console.print("[green]Keychain is now usable.[/]")
                            else:
                                console.print("[yellow]Keychain still unavailable. We'll use env vars for this session.[/]")
                except Exception:
                    pass
            # Show banner only once per process
            run_home._kc_banner_shown = True

        options = [
            "Quick Run",
            "Guided Setup (recommended)",
            "Options",
            "Help & Docs",
            "Exit",
        ]
        table = Table(box=box.SIMPLE, show_edge=False)
        table.add_column("#", style="cyan")
        table.add_column("Action", style="bold")
        for i, opt in enumerate(options, 1):
            table.add_row(str(i), opt)
        console.print(table)

        # Smart default: if user has config, default to Quick Run (1), otherwise Setup (2)
        has_config = bool(_env("SWEBENCH_PROVIDER") and _env("SWEBENCH_MODEL"))
        default_choice = 1 if has_config else 2
        choice = IntPrompt.ask("Enter number", default=default_choice)

        # Exit should return immediately without showing Help
        if choice == 5:
            return

        if choice == 1:
            # Attempt Quick Run using saved defaults; if missing, route to setup
            provider = _env("SWEBENCH_PROVIDER") or ""
            model = _env("SWEBENCH_MODEL") or ""
            dataset = _env("SWEBENCH_DATASET") or "lite"
            count = _env("SWEBENCH_COUNT")

            if not provider or not model:
                console.print("\n[bold yellow]Quick Run requires a configured provider/model.[/]")
                console.print("Routing to Guided Setup...\n")
                run_wizard()
                continue

            # Summary and confirmation
            summary = Table(box=box.SIMPLE)
            summary.add_column("Setting")
            summary.add_column("Value")
            summary.add_row("Provider", provider)
            summary.add_row("Model", model)
            summary.add_row("Dataset", dataset)
            if count:
                summary.add_row("Count", count)
            console.print(Panel(summary, title="Quick Run", style="green"))
            if not Confirm.ask("Run now?", default=True):
                continue
            cmd = ["swebench", "run", "-d", dataset]
            if count:
                cmd += ["--count", count]
            console.print("\n[cyan]Starting evaluation...[/cyan]\n")
            # Import and call evaluate function directly instead of subprocess
            from .cli import evaluate
            try:
                evaluate(
                    patches=None,
                    patches_dir=None,
                    dataset=dataset,
                    count=int(count) if count else None,
                    subset=None,
                    timeout=60,  # Default timeout
                    workers=None,  # Auto-detect
                    no_cache=False,
                    no_preflight=False,
                    output_dir=None,
                    provider=None,  # Use environment variables
                    model=None,    # Use environment variables
                    temperature=None,
                    max_tokens=None
                )
            except SystemExit as e:
                sys.exit(e.code)

        if choice == 2:
            run_wizard()
            continue

        if choice == 3:
            run_options()
            continue

        # Help & Docs
        if choice == 4:
            console.print(Panel.fit("""
Common commands:
  swebench setup      # guided setup
  swebench run        # uses saved defaults
  swebench run -d lite --count 5
  swebench clean --all

Troubleshooting:
  • Image pull denied → docker login ghcr.io
  • Provider 401 → check your key / permissions
  • No evaluation_results → harness ran but did not produce results, see logs path
""", title="Help & Docs", style="cyan"))
            # Keep users in the TUI after viewing help
            Prompt.ask("Press Enter to return to Home", default="")
            continue
        # Exit
        if choice == 5:
            return


def _write_env_lines(lines: list[str]) -> None:
    """Write environment variables to .env file, updating existing values."""
    try:
        # Parse the new values
        env_updates = {}
        for line in lines:
            if "=" in line:
                key, value = line.split("=", 1)
                env_updates[key] = value

        # Read existing .env file
        env_path = Path(".env")
        existing_lines = []
        if env_path.exists():
            with open(env_path) as f:
                existing_lines = f.readlines()

        # Update or add values
        updated = set()
        new_lines = []
        for line in existing_lines:
            # Skip comment lines and empty lines
            if line.strip().startswith("#") or not line.strip():
                # Skip "Saved by swebench" comments to avoid duplicates
                if "Saved by swebench" not in line:
                    new_lines.append(line)
                continue

            # Check if this line sets one of our variables
            line_updated = False
            for key, value in env_updates.items():
                if line.startswith(f"{key}="):
                    new_lines.append(f"{key}={value}\n")
                    updated.add(key)
                    line_updated = True
                    break

            if not line_updated:
                # Keep the line if it's not one we're updating
                new_lines.append(line)

        # Add any new variables that weren't in the file
        for key, value in env_updates.items():
            if key not in updated:
                new_lines.append(f"{key}={value}\n")

        # Write back the updated file
        with open(env_path, "w") as f:
            f.writelines(new_lines)
        console.print("Saved defaults to [bold].env[/]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not save to .env: {e}[/]")


def run_options() -> None:
    """Options menu: profiles/keys/defaults/docker/preflight.

    Single-screen with Back navigation.
    """
    while True:
        console.print(Panel.fit("Options", style="bold cyan"))
        table = Table(box=box.SIMPLE)
        table.add_column("#")
        table.add_column("Action")
        actions = [
            "Provider & Profile",
            "API Key (set/clear)",
            "Model Catalog Preview",
            "Defaults (dataset/count)",
            "Docker & Registry",
            "Preflight",
        ]
        # Back is standardized as 0
        table.add_row("0", "Back")
        for idx, item in enumerate(actions, start=1):
            table.add_row(str(idx), item)
        console.print(table)
        choice = IntPrompt.ask("Enter number", default=0)
        if choice == 0:
            return
        if choice == 1:
            # Provider & Profile (flattened view)
            from .secrets_store import list_profiles as sec_list_profiles

            console.print(Panel.fit("Manage Provider Profiles", style="bold cyan"))
            items_pp: list[tuple[int, str, str]] = []  # (num, provider, profile)
            counter_pp = 1
            providers_pp = ["openai", "anthropic", "openrouter", "ollama"]
            for prov in providers_pp:
                profs = sec_list_profiles(prov) or ["default"]
                console.print(f"[bold]{prov.upper()}[/]")
                shown_any = False
                active = get_active_profile(prov)
                for prof in profs:
                    shown_any = True
                    tag = " [active]" if prof == active else ""
                    console.print(f"  {counter_pp}. {prof}{tag}")
                    items_pp.append((counter_pp, prov, prof))
                    counter_pp += 1
                if not shown_any:
                    console.print("  - NA")
                console.print("")

            console.print("  0. Back")
            add_choice_pp = counter_pp
            console.print(f"  {add_choice_pp}. Add profile…")

            selection_pp = IntPrompt.ask("Enter number", default=0)
            if selection_pp == 0:
                continue
            if selection_pp == add_choice_pp:
                prov = pick_provider(allow_back=True)
                if prov == "__back__":
                    continue
                name = Prompt.ask("Name this profile (e.g., personal, work)")
                set_active_profile(prov, name)
                console.print(f"Created and set active profile '{name}' for {prov}")
                continue

            match_pp = next(((p, pr) for num, p, pr in items_pp if num == selection_pp), None)
            if not match_pp:
                console.print("Invalid choice.")
                continue
            prov, prof = match_pp
            set_active_profile(prov, prof)
            console.print(f"Set active profile for {prov} → {prof}")
            _write_env_lines([f"SWEBENCH_PROVIDER={prov}"])
        elif choice == 2:
            # API Key Manager (flattened view)
            from .secrets_store import (
                get_key_source,
                has_key_for_profile,
                is_keychain_available,
            )
            from .secrets_store import list_profiles as sec_list_profiles

            keychain_status = "available" if is_keychain_available() else "unavailable"
            console.print(Panel.fit(f"Manage API Keys\n[dim]Keychain: {keychain_status}[/]", style="bold cyan"))
            # Build flattened list: [(num, provider, profile)]
            items: list[tuple[int, str, str]] = []
            counter = 1
            providers = ["openai", "anthropic", "openrouter", "ollama"]
            for prov in providers:
                profs = sec_list_profiles(prov) or ["default"]
                header = f"[bold]{prov.upper()}[/]"
                console.print(header)
                if not profs:
                    console.print("  - NA")
                shown_any = False
                for prof in profs:
                    shown_any = True
                    is_active = (get_active_profile(prov) == prof)
                    active_tag = " [active]" if is_active else ""
                    key_tag = "present" if has_key_for_profile(prov, prof) else "not set"
                    source = get_key_source(prov, prof)
                    console.print(f"  {counter}. {prof}{active_tag} — key: {key_tag} — source: {source}")
                    items.append((counter, prov, prof))
                    counter += 1
                if not shown_any:
                    console.print("  - NA")
                console.print("")

            console.print("  0. Back")
            add_choice = counter
            console.print(f"  {add_choice}. Add profile…")

            selection = IntPrompt.ask("Enter number", default=0)
            if selection == 0:
                continue
            if selection == add_choice:
                prov = pick_provider(allow_back=True)
                if prov == "__back__":
                    continue
                name = Prompt.ask("Name this profile (e.g., personal, work)")
                set_active_profile(prov, name)
                console.print(f"Created and set active profile '{name}' for {prov}")
                continue

            # Find selected item
            match = next(((p, pr) for num, p, pr in items if num == selection), None)
            if not match:
                console.print("Invalid choice.")
                continue
            prov, prof = match
            console.print(Panel.fit(f"{prov.upper()} — {prof}", style="bold green"))
            has_key = has_key_for_profile(prov, prof)
            source = get_key_source(prov, prof)
            console.print(f"Key status: {'present' if has_key else 'not set'} (source: {source})")
            action = Prompt.ask("Action: set / clear / make-active / test / back", default="set")
            if action.startswith("back"):
                continue
            if action == "make-active":
                set_active_profile(prov, prof)
                console.print("Set as active profile.")
                continue
            if action == "test":
                # Lightweight model fetch validation
                key = get_api_key(prov)
                models = fetch_models(prov, key)
                if models:
                    console.print("[green]Key looks valid (models fetched).[/]")
                else:
                    console.print("[yellow]Could not fetch models. Key may be missing/invalid, or provider blocks listing.[/]")
                    console.print("[dim]Tip: Try 'set' or run Preflight for provider diagnostics.[/]")
                continue
            if action == "clear":
                from .secrets_store import clear_api_key
                ok = clear_api_key(prov, prof)
                console.print("Cleared" if ok else "Failed to clear (no keyring?)")
                continue
            # set
            # Temporarily ensure chosen profile is active so prompt uses it
            set_active_profile(prov, prof)
            res = prompt_api_key(prov, allow_back=True)
            if res == "__back__":
                continue
            console.print("Key stored (if provided)")
        elif choice == 3:
            # Model Catalog Preview (flattened provider selector)
            console.print(Panel.fit("Model Catalog Preview", style="bold cyan"))
            providers_mc = ["openai", "anthropic", "openrouter", "ollama"]
            nums: list[tuple[int, str]] = []
            n = 1
            for prov in providers_mc:
                active_prof = get_active_profile(prov) or "(default)"
                key_present = bool(get_api_key(prov)) or prov == "openrouter"
                key_tag = "key: present" if key_present else "key: not set"
                console.print(f"  {n}. {prov.upper()} — active: {active_prof} — {key_tag}")
                nums.append((n, prov))
                n += 1
            console.print("  0. Back")
            sel = IntPrompt.ask("Choose provider", default=0)
            if sel == 0:
                continue
            prov = next((p for (num, p) in nums if num == sel), None)
            if not prov:
                console.print("Invalid choice.")
                continue
            key = get_api_key(prov)
            if not key and prov != "openrouter":
                console.print("[yellow]No API key configured. Manage keys in Options → API Key.[/]")
            models = fetch_models(prov, key)
            if models:
                table = Table(box=box.SIMPLE)
                table.add_column("#")
                table.add_column("Model")
                for i, m in enumerate(models[:19], 1):
                    table.add_row(str(i), m)
                console.print(Panel(table, title=f"Top Models — {prov}", style="green"))
                Prompt.ask("Press Enter to return", default="")
            else:
                console.print("No models available or failed to fetch.")
                Prompt.ask("Press Enter to return", default="")
        elif choice == 4:
            # Defaults (dataset/count)
            ds = pick_dataset(allow_back=True)
            if ds == "__back__":
                continue
            c = pick_count(allow_back=True)
            if c == "__back__":
                continue
            lines = [f"SWEBENCH_DATASET={ds}"]
            if c is not None:
                lines.append(f"SWEBENCH_COUNT={c}")
            _write_env_lines(lines)
        elif choice == 5:
            # Docker & Registry (with autodetect/reset)
            current_ns = os.getenv("SWEBENCH_DOCKER_NAMESPACE") or "(auto)"
            console.print(Panel.fit(f"Docker & Registry\n[dim]Current: {current_ns}[/]", style="bold cyan"))

            table_dr = Table(box=box.SIMPLE)
            table_dr.add_column("#")
            table_dr.add_column("Action")
            actions_dr = [
                "Autodetect best registry",
                "Set namespace manually",
                "Reset to auto",
                "Run Preflight (recommended)",
            ]
            table_dr.add_row("0", "Back")
            for idx, item in enumerate(actions_dr, start=1):
                table_dr.add_row(str(idx), item)
            console.print(table_dr)
            sel_dr = IntPrompt.ask("Enter number", default=4)
            if sel_dr == 0:
                continue
            if sel_dr == 1:
                # Autodetect: probe via preflight helper; it already retries GHCR when Hub ARM64 missing
                ok, msg = _run_harness_preflight()
                if ok:
                    console.print("[green]Registry validated via preflight.[/]")
                else:
                    # If message indicates GHCR 404, try docker.io explicitly
                    console.print("[yellow]Preflight failed. Trying Docker Hub (swebench) namespace...[/]")
                    ok2, msg2 = _run_harness_preflight(namespace="docker.io/swebench")
                    if ok2:
                        console.print("[green]Validated Docker Hub (swebench); saving.[/]")
                        try:
                            with open(".env", "a", encoding="utf-8") as f:
                                f.write("\nSWEBENCH_DOCKER_NAMESPACE=docker.io/swebench\n")
                        except Exception:
                            pass
                    else:
                        console.print("[red]Autodetect failed. Leaving namespace unset (auto).[/]")
                        console.print(Panel.fit((msg2 or msg)[-800:], title="probe output (tail)", style="red"))
                continue
            if sel_dr == 2:
                new_ns = Prompt.ask("Enter namespace (e.g., docker.io/swebench or ghcr.io/epoch-research)", default="")
                if new_ns.strip():
                    _write_env_lines([f"SWEBENCH_DOCKER_NAMESPACE={new_ns.strip()}"])
                continue
            if sel_dr == 3:
                # Reset to auto: remove env var from current session and suggest removing from .env
                try:
                    if "SWEBENCH_DOCKER_NAMESPACE" in os.environ:
                        del os.environ["SWEBENCH_DOCKER_NAMESPACE"]
                    # Also remove from .env if present
                    try:
                        from pathlib import Path
                        env_path = Path(".env")
                        if env_path.exists():
                            content = env_path.read_text(encoding="utf-8")
                            lines = [ln for ln in content.splitlines() if not ln.strip().startswith("SWEBENCH_DOCKER_NAMESPACE=")]
                            env_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                    except Exception:
                        pass
                except Exception:
                    pass
                console.print("Reset namespace to auto.")
                continue
            # Run Preflight
            preflight_wizard()
        elif choice == 6:
            preflight_wizard()
