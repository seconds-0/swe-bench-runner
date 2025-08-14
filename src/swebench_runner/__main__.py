"""Entry point for ``python -m swebench_runner`` execution.

This ensures that running the package as a module displays
help when no arguments are provided, matching CLI UX tests.
"""

from .cli import cli


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
