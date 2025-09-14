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
