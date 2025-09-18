"""Utility helpers shared across CLI entrypoints and training."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

__all__ = [
    "apply_yaml_config_defaults",
    "load_yaml_config",
    "save_scatter",
    "seed_everything",
]


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file and ensure it returns a mapping."""

    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):  # pragma: no cover - defensive guard
        raise ValueError(f"Config file {path} must contain a mapping at the top level")
    return data


def apply_yaml_config_defaults(
    parser: argparse.ArgumentParser, config_path: str | Path
) -> Dict[str, Any]:
    """Inject defaults from ``config_path`` into an ``ArgumentParser``.

    Only keys that match known parser destinations are applied. The filtered
    mapping is returned so callers may log or inspect the resolved defaults.
    """

    config = load_yaml_config(config_path)
    valid = {action.dest for action in parser._actions if action.dest != argparse.SUPPRESS}
    filtered = {key: value for key, value in config.items() if key in valid}
    parser.set_defaults(**filtered)
    return filtered


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - only on CUDA builds
        torch.cuda.manual_seed_all(seed)


def save_scatter(points: torch.Tensor, path: str | Path, lim: float = 8.0) -> None:
    """Persist a 2D scatter plot to ``path`` without leaving open figures."""

    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 4))
    pts = points.detach().cpu().numpy()
    plt.scatter(pts[:, 0], pts[:, 1], s=3)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal", "box")
    plt.tight_layout()
    plt.savefig(Path(path), dpi=150)
    plt.close()
