"""Shared utility functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config/default.yaml") -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def print_section(title: str, width: int = 60) -> None:
    """Print a formatted section header to stdout.

    Args:
        title: Section title text.
        width: Total width of the header line.
    """
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")
