"""Centralized paths for non-code keynote inputs and generated outputs.

The default data directory is outside the source repository so notebooks and
renderers do not mix scientific assets with importable code. Set
``SBGEA_KEYNOTE_DATA`` to use a different location on another workstation.
"""
from __future__ import annotations

import os
from pathlib import Path


DEFAULT_DATA_ROOT = (
    Path.home() / "Documents" / "Science" / "Programs" / "LFS" / "KEYNOTE"
)


def data_root(*, create: bool = False) -> Path:
    """Return the shared KEYNOTE data root, honoring the environment override."""
    root = Path(os.environ.get("SBGEA_KEYNOTE_DATA", DEFAULT_DATA_ROOT)).expanduser()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def input_path(*parts: str) -> Path:
    """Return a path below ``KEYNOTE/inputs`` without creating it."""
    return data_root() / "inputs" / Path(*parts)


def output_path(*parts: str, create_parent: bool = True) -> Path:
    """Return a path below ``KEYNOTE/outputs`` and create its parent by default."""
    path = data_root(create=True) / "outputs" / Path(*parts)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def output_dir(*parts: str) -> Path:
    """Return and create a directory below ``KEYNOTE/outputs``."""
    path = data_root(create=True) / "outputs" / Path(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path

