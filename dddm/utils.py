from collections.abc import Iterable, Mapping
from typing import Sequence

import torch


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


def plot_training_curves(
    history: Mapping[str, Sequence[float]] | Mapping[str, list[float]],
    path: str,
    *,
    title: str,
    xlabel: str,
    x_key: str | None = None,
    metrics: Iterable[str] | None = None,
) -> str:
    """Plot training/evaluation dynamics from a metrics history mapping.

    Parameters
    ----------
    history:
        Mapping of metric name to a sequence of values. When ``x_key`` is
        provided, the corresponding entry in ``history`` is used as the x-axis
        coordinates for all plotted metrics.
    path:
        Destination path for the saved plot.
    title / xlabel:
        Text labels for the generated figure.
    x_key:
        Optional key inside ``history`` holding explicit x-axis coordinates
        (e.g. global steps or epochs). When omitted, ``range(1, N + 1)`` is
        used for each metric.
    metrics:
        Optional iterable restricting the plotted metrics. When ``None`` all
        metrics except ``x_key`` are rendered.
    """

    import matplotlib.pyplot as plt

    if metrics is None:
        metrics = [k for k in history.keys() if k != x_key]

    x_values: Sequence[float] | None = None
    if x_key is not None:
        x_values = history.get(x_key)
        if x_values is not None and len(x_values) == 0:
            x_values = None

    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for key in metrics:
        if key == x_key:
            continue
        values = history.get(key)
        if values is None or len(values) == 0:
            continue

        if x_values is None:
            xs: Sequence[float] = range(1, len(values) + 1)
        else:
            if len(x_values) != len(values):
                continue
            xs = x_values

        ax.plot(xs, values, label=key)
        plotted = True

    if not plotted:
        plt.close(fig)
        raise ValueError("No metrics available to plot training dynamics.")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
