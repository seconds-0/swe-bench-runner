"""Bootstrap UX flow for first-time users."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import click

from . import exit_codes
from .cache import auto_detect_patches_file, is_first_run, mark_first_run_complete


def show_welcome_message() -> None:
    """Show welcome message for first-time users."""
    click.echo("🎁 Welcome to SWE-bench Runner!")
    click.echo()
    click.echo(
        "This tool helps you evaluate code patches against the SWE-bench benchmark."
    )
    click.echo("First-time setup will download necessary components for evaluation.")
    click.echo()


def show_setup_wizard() -> None:
    """Show setup wizard for Docker installation."""
    click.echo("🔧 Setup Wizard")
    click.echo()
    click.echo("SWE-bench Runner requires Docker to run evaluations.")
    click.echo("Would you like installation instructions?")
    click.echo()
    click.echo("1) macOS (Docker Desktop)")
    click.echo("2) Ubuntu/Debian (Docker Engine)")
    click.echo("3) Skip")
    click.echo()

    choice = click.prompt("Choice", type=click.Choice(["1", "2", "3"]), default="3")

    if choice == "1":
        show_macos_instructions()
    elif choice == "2":
        show_linux_instructions()
    else:
        click.echo("Skipping Docker installation instructions.")

    click.echo()


def show_macos_instructions() -> None:
    """Show macOS Docker installation instructions."""
    click.echo()
    click.echo("📘 macOS Docker Desktop Setup:")
    click.echo("1. Download from https://docker.com/products/docker-desktop")
    click.echo("2. Install and start Docker Desktop")
    click.echo("3. Wait for whale icon in menu bar")
    click.echo("4. Run this command again to continue")
    click.echo()

    if click.confirm("Open download page in browser?", default=False):
        import webbrowser
        webbrowser.open("https://docker.com/products/docker-desktop")


def show_linux_instructions() -> None:
    """Show Linux Docker installation instructions."""
    click.echo()
    click.echo("📘 Ubuntu/Debian Docker Engine Setup:")
    click.echo("1. Update package index:")
    click.echo("   sudo apt-get update")
    click.echo("2. Install Docker:")
    click.echo("   sudo apt-get install docker.io")
    click.echo("3. Start Docker service:")
    click.echo("   sudo systemctl start docker")
    click.echo("4. Add your user to docker group:")
    click.echo("   sudo usermod -aG docker $USER")
    click.echo("5. Log out and back in, then run this command again")
    click.echo()


def check_and_prompt_first_run(no_input: bool = False) -> bool:
    """Check if this is first run and show appropriate prompts.

    Args:
        no_input: If True, skip all prompts (CI mode)

    Returns:
        True if this was the first run, False otherwise
    """
    if not is_first_run():
        return False

    if no_input:
        # CI mode - just mark as complete and continue
        mark_first_run_complete()
        return True

    # Interactive mode - show welcome and setup
    show_welcome_message()

    if click.confirm("Ready to get started?", default=True):
        mark_first_run_complete()
        return True
    else:
        click.echo("Setup cancelled. Run again when ready.")
        sys.exit(exit_codes.SUCCESS)


def suggest_patches_file() -> Path | None:
    """Suggest patches file if none provided and one is auto-detected."""
    detected_file = auto_detect_patches_file()
    if detected_file:
        click.echo(f"💡 Found {detected_file.name} in current directory")
        if click.confirm(f"Use {detected_file.name}?", default=True):
            return detected_file
    return None


def show_success_message(instance_id: str, is_first_success: bool = False) -> None:
    """Show success message with next steps."""
    if is_first_success:
        click.echo()
        click.echo("┌───────────────────────────┐")
        click.echo("│   🎉 SUCCESS! 🎉       │")
        click.echo("└───────────────────────────┘")
        click.echo()
        click.echo("🎊 Congrats on your first successful evaluation!")
        click.echo(f"Instance {instance_id} completed successfully.")
        click.echo()
        click.echo("💡 Next steps:")
        click.echo(
            "   • Try multiple instances: swebench run --patches predictions.jsonl"
        )
        click.echo(
            "   • Filter by repository: swebench run --subset 'django/**' --patches ..."
        )
        click.echo("   • Check results: swebench clean (shows cache usage)")
        click.echo()
    else:
        click.echo(f"✅ {instance_id}: Evaluation completed successfully")


def show_docker_setup_help() -> None:
    """Show Docker setup help when Docker is not available."""
    click.echo("⚠️  Docker is not available or not running.")
    click.echo()

    system = platform.system()
    if system == "Darwin":
        click.echo("macOS users:")
        click.echo("• Install Docker Desktop from https://docker.com/products/docker-desktop")
        click.echo("• Start Docker Desktop and wait for the whale icon")
    elif system == "Linux":
        click.echo("Linux users:")
        click.echo("• Install Docker: sudo apt-get install docker.io")
        click.echo("• Start Docker: sudo systemctl start docker")
        click.echo("• Add user to docker group: sudo usermod -aG docker $USER")
    else:
        click.echo("• Install Docker from https://docs.docker.com/get-docker/")

    click.echo()
    click.echo("For more help, run: swebench setup")


def show_resource_warning(free_gb: float) -> None:
    """Show resource warning for insufficient disk space."""
    click.echo(f"⚠️  Warning: Only {free_gb:.1f}GB free disk space available")
    click.echo("   SWE-bench evaluations require substantial disk space:")
    click.echo("   • Minimum: 50GB for lite dataset")
    click.echo("   • Recommended: 120GB+ for full dataset")
    click.echo()
    click.echo("   Free up space or run: swebench clean --all")


def show_memory_warning(mem_gb: float) -> None:
    """Show memory warning for insufficient RAM."""
    click.echo(f"⚠️  Warning: Only {mem_gb:.1f}GB RAM available")
    click.echo("   SWE-bench evaluations require substantial memory:")
    click.echo("   • Minimum: 8GB RAM")
    click.echo("   • Recommended: 16GB+ RAM")
    click.echo()
    click.echo("   Close other applications or increase system memory.")
