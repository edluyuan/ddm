import torch


class GMM2D(torch.utils.data.IterableDataset):
    """Two-Gaussian mixture dataset used in Section 6.1."""

    def __init__(self, mu1=(3.0, 3.0), mu2=(-3.0, 3.0), sigma: float = 0.5, seed: int = 0):
        super().__init__()
        self.mu1 = torch.tensor(mu1, dtype=torch.float32)
        self.mu2 = torch.tensor(mu2, dtype=torch.float32)
        self.sigma = float(sigma)
        self.rng = torch.Generator().manual_seed(seed)

    def __iter__(self):
        while True:
            which = torch.bernoulli(torch.tensor(0.5)).item()
            mu = self.mu1 if which > 0.5 else self.mu2
            x = torch.randn(2, generator=self.rng) * self.sigma + mu
            yield x


def sample_gmm(
    batch: int,
    mu1=(3.0, 3.0),
    mu2=(-3.0, 3.0),
    sigma: float = 0.5,
    device: str = "cpu",
) -> torch.Tensor:
    b1 = torch.bernoulli(0.5 * torch.ones(batch, device=device))
    mu = torch.stack(
        [torch.tensor(mu1, device=device), torch.tensor(mu2, device=device)], dim=0
    )
    pick = mu[b1.long()]
    return pick + sigma * torch.randn(batch, 2, device=device)
