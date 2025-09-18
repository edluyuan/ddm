"""Utility helpers shared by the toy and CIFAR training scripts."""

from __future__ import annotations

import argparse
from typing import Any, Mapping

import torch
import yaml


def load_yaml_config(path: str) -> dict[str, Any]:
    """Return a dictionary loaded from ``path``.

    Empty files resolve to an empty mapping and we enforce that the YAML
    document contains a mapping at the top level so it can be merged into the
    ``argparse`` defaults used by our CLIs.
    """

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):  # pragma: no cover - defensive guard
        raise ValueError(f"Config file {path} must contain a mapping at the top level")
    return data


def config_overrides_for_parser(
    parser: argparse.ArgumentParser, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Filter ``config`` so that only keys understood by ``parser`` remain."""

    valid = {action.dest for action in parser._actions if action.dest != argparse.SUPPRESS}
    return {key: value for key, value in config.items() if key in valid}


def apply_config_overrides(parser: argparse.ArgumentParser, path: str) -> dict[str, Any]:
    """Load overrides from ``path`` and apply them to ``parser`` defaults."""

    overrides = config_overrides_for_parser(parser, load_yaml_config(path))
    if overrides:
        parser.set_defaults(**overrides)
    return overrides


def maybe_init_wandb(
    enabled: bool,
    *,
    project: str,
    config: Mapping[str, Any] | None = None,
    run_name: str | None = None,
    import_error_message: str | None = None,
    **wandb_kwargs: Any,
):
    """Initialise a Weights & Biases run if ``enabled``.

    The helper encapsulates the optional dependency and provides a consistent
    error message when the user asks for logging but the package is not
    available.
    """

    if not enabled:
        return None
    try:  # pragma: no cover - light wrapper whose behaviour we guard with tests
        import wandb
    except ImportError as exc:  # pragma: no cover - defensive guard
        message = import_error_message or "Weights & Biases is not installed"
        raise RuntimeError(message) from exc

    return wandb.init(
        project=project,
        name=run_name,
        config=dict(config or {}),
        **wandb_kwargs,
    )


def save_scatter(points: torch.Tensor, path: str, lim: float = 8.0) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 4))
    pts = points.detach().cpu().numpy()
    plt.scatter(pts[:, 0], pts[:, 1], s=3)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal", "box")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
