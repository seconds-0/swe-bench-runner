from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from shutil import disk_usage

import click

from .cache import get_cache_dir


def is_non_interactive() -> bool:
    """Detect non-interactive/CI mode using env and TTY."""
    try:
        return (
            os.getenv("SWEBENCH_NO_INPUT") == "1"
            or not (sys.stdin.isatty() and sys.stdout.isatty())
            or os.getenv("CI") == "1"
            or os.getenv("TERM", "").lower() == "dumb"
        )
    except Exception:
        return True


@click.command("doctor")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def doctor(json_output: bool) -> None:
    """Run environment diagnostics and suggest fixes (no prompts)."""
    info: dict[str, object] = {}
    problems: list[str] = []
    hints: list[str] = []

    # Basics
    info["python"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    info["os"] = platform.platform()
    info["tty"] = bool(sys.stdin.isatty() and sys.stdout.isatty())
    info["ci"] = os.getenv("CI") == "1"
    info["non_interactive"] = is_non_interactive()

    # Disk space
    try:
        _, _, free = disk_usage(str(Path.home()))
        info["disk_free_gb"] = round(free / (1024**3), 1)
        if free < 5 * 1024**3:
            problems.append("Low disk space (<5GB free)")
            hints.append("Free space or run 'swebench clean --all'")
    except Exception as e:
        # Best-effort disk check; record as hint
        hints.append(f"Disk check unavailable: {e}")

    # Cache dir
    try:
        info["cache_dir"] = str(get_cache_dir())
    except Exception as e:
        problems.append(f"Cache dir error: {e}")

    # Docker
    docker_ok = False
    try:
        from .docker_client import get_docker_client

        client = get_docker_client()
        client.ping()
        docker_ok = True
    except Exception as e:  # noqa: S110
        info["docker_error"] = str(e)
        hints.append(
            "Ensure Docker Desktop is running (macOS/Windows) or docker daemon (Linux)"
        )
    info["docker"] = "ok" if docker_ok else "unavailable"

    # Provider/profile/key snapshot
    try:
        from .secrets_store import get_active_profile, get_api_key, list_profiles

        provider = os.getenv("SWEBENCH_PROVIDER")
        info["provider_env"] = provider or "(unset)"
        providers = ["openai", "anthropic", "openrouter", "huggingface"]
        prov_report: dict[str, object] = {}
        for prov in providers:
            profiles = list_profiles(prov)
            active = get_active_profile(prov) or ""
            prov_report[prov] = {
                "profiles": profiles,
                "active": active,
                "has_key_active": bool(get_api_key(prov, profile=active))
                if active
                else False,
            }
        info["profiles"] = prov_report
        if provider:
            active = prov_report.get(provider, {}).get("active")  # type: ignore[assignment]
            has_key = prov_report.get(provider, {}).get("has_key_active")  # type: ignore[assignment]
            if not active:
                hints.append(
                    "Create and set active profile: swebench profiles create "
                    f"--provider {provider} --profile default --active"
                )
            elif not has_key:
                hints.append(
                    "Set API key: swebench profiles set-key "
                    f"--provider {provider} --profile {active} --key <secret>"
                )
    except Exception as e:
        # Keyring may be unavailable in CI; record a hint rather than failing
        hints.append(f"Keyring unavailable: {e}")

    # Apple Silicon hint for namespace
    arch = platform.machine().lower()
    info["arch"] = arch
    if arch in ("arm64", "aarch64") and not os.getenv("SWEBENCH_DOCKER_NAMESPACE"):
        hints.append(
            "On Apple Silicon, we auto-switch to GHCR if Docker Hub image is missing. "
            "If pull is denied, run: docker login ghcr.io"
        )

    output = {"info": info, "problems": problems, "hints": hints}
    if json_output:
        click.echo(json.dumps(output))
        return

    click.echo("\n🩺 Environment Doctor\n")
    click.echo(f"Python: {info['python']}")
    click.echo(f"OS: {info['os']}")
    click.echo(
        "TTY: "
        + ("yes" if info["tty"] else "no")
        + " | CI: "
        + ("yes" if info["ci"] else "no")
        + " | Non-interactive: "
        + ("yes" if info["non_interactive"] else "no")
    )
    if "disk_free_gb" in info:
        click.echo(f"Disk free: {info['disk_free_gb']} GB")
    click.echo(f"Docker: {info['docker']}")
    if not docker_ok and "docker_error" in info:
        click.echo(f"  Error: {info['docker_error']}")
    click.echo(f"Cache dir: {info.get('cache_dir', '(unknown)')}")
    click.echo(f"Provider (env): {info.get('provider_env', '(unset)')}")
    if problems:
        click.echo("\n⚠️  Problems detected:")
        for p in problems:
            click.echo(f"  - {p}")
    if hints:
        click.echo("\n💡 Hints:")
        for h in hints:
            click.echo(f"  - {h}")
    click.echo(
        "\nRun 'swebench setup' for guided setup, or "
        "'swebench run -d lite --count 5' to try now."
    )
