import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal time embedding used in diffusion transformers."""

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            t = t.view(-1)
        half = self.dim // 2
        device = t.device
        exponent = torch.arange(half, device=device, dtype=t.dtype)
        exponent = -math.log(self.max_period) * exponent / max(half - 1, 1)
        freqs = torch.exp(exponent)
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class TimeFeat(nn.Module):
    """Fourier time features."""

    def __init__(self, n: int = 16) -> None:
        super().__init__()
        self.freq = nn.Parameter(torch.linspace(1.0, n, n), requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        f = self.freq[None, :] * (2.0 * math.pi) * t[:, None]
        return torch.cat([torch.sin(f), torch.cos(f)], dim=-1)


class DDDMMLP(nn.Module):
    r"""Distributional denoiser ``\hat{x}_θ(t, x_t, ξ)``.

    Small MLP that takes ``[x_t (2), ξ (2), time-features]`` and outputs a 2D
    sample ``\hat{x}_0``.
    """

    def __init__(self, time_dim: int = 32, hidden: int = 128) -> None:
        super().__init__()
        self.tfeat = TimeFeat(n=time_dim // 2)
        inp = 2 + 2 + time_dim
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, xt: torch.Tensor, t: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        tf = self.tfeat(t)
        h = torch.cat([xt, xi, tf], dim=-1)
        return self.net(h)


class PatchEmbed(nn.Module):
    """Embed images into a sequence of patch tokens."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 6,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("Image size must be divisible by patch size")
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchUnembed(nn.Module):
    """Map patch tokens back into image space."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        out_channels: int = 3,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.num_patches_per_side = img_size // patch_size
        self.proj = nn.Linear(embed_dim, out_channels * patch_size * patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        if C != self.embed_dim:
            raise ValueError("Unexpected embedding dimension")
        h = self.num_patches_per_side
        if N != h * h:
            raise ValueError("Token count does not match image dimensions")
        x = self.proj(x)
        x = x.view(B, h, h, self.out_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(
            B, self.out_channels, self.img_size, self.img_size
        )
        return x


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class DDDMDiT(nn.Module):
    """Distributional diffusion model with a DiT backbone for images."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 6,
        out_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        time_embed_dim: int = 256,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim)
        )
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.unembed = PatchUnembed(
            img_size=img_size,
            patch_size=patch_size,
            out_channels=out_channels,
            embed_dim=embed_dim,
        )
        self.out_channels = out_channels

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        if xt.shape != xi.shape:
            raise ValueError("xt and xi must have the same shape")
        if t.ndim != 1:
            t = t.view(-1)
        if xt.dim() != 4:
            raise ValueError("Expecting image tensors with shape [B, C, H, W]")
        x = torch.cat([xt, xi], dim=1)
        h = self.patch_embed(x)
        temb = self.time_mlp(self.time_embed(t))
        h = h + temb[:, None, :] + self.pos_embed
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        x0_hat = self.unembed(h)
        return x0_hat
