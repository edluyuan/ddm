import torch


def rbf_mmd2(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Unbiased MMD^2 with RBF kernel, σ fixed."""

    def pdist2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a2 = (a * a).sum(-1).unsqueeze(-1)
        b2 = (b * b).sum(-1).unsqueeze(0)
        return a2 + b2 - 2.0 * (a @ b.T)

    k = lambda d2: torch.exp(-d2 / (2.0 * (sigma**2)))

    n = x.size(0)
    m = y.size(0)
    dxx = pdist2(x, x)
    dyy = pdist2(y, y)
    dxy = pdist2(x, y)

    mask_x = ~torch.eye(n, dtype=torch.bool, device=x.device)
    mask_y = ~torch.eye(m, dtype=torch.bool, device=x.device)
    kxx = k(dxx)[mask_x].mean()
    kyy = k(dyy)[mask_y].mean()
    kxy = k(dxy).mean()
    return kxx + kyy - 2.0 * kxy


import torch
import torch.nn as nn


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=1, fix_sigma=None):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def gaussian_kernel(
            self, source, target, kernel_mul=2.0, kernel_num=1, fix_sigma=None
    ):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        L2_distance = ((total0 - total1) ** 2).sum(2)

        # Check for extreme values in L2_distance
        if torch.isinf(L2_distance).any() or torch.isnan(L2_distance).any():
            print(f"Warning: Extreme values detected in L2_distance")
            print(f"L2_distance contains inf: {torch.isinf(L2_distance).any()}")
            print(f"L2_distance contains nan: {torch.isnan(L2_distance).any()}")
            # Use a fixed sigma when we have extreme values
            fix_sigma = 1.0

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            denominator = n_samples ** 2 - n_samples
            if denominator <= 0:
                bandwidth = torch.tensor(1.0, device=L2_distance.device)
            else:
                bandwidth = torch.sum(L2_distance.data) / denominator
                # Check if bandwidth is problematic
                if torch.isinf(bandwidth) or torch.isnan(bandwidth) or bandwidth <= 0:
                    print(f"Warning: Problematic bandwidth {bandwidth}, using fixed value")
                    bandwidth = torch.tensor(1.0, device=L2_distance.device)
                else:
                    bandwidth = torch.clamp(bandwidth, min=1e-3, max=1e3)

        # Create bandwidth list for multiple kernels
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        kernel_val = []
        for i, bandwidth_temp in enumerate(bandwidth_list):
            # Clamp L2_distance to prevent overflow in exp
            L2_clamped = torch.clamp(L2_distance, min=0.0, max=1e3)
            kernel_temp = torch.exp(-L2_clamped / bandwidth_temp)
            kernel_val.append(kernel_temp)

        result = sum(kernel_val)
        return result

    def forward(self, source, target):
        # Check for extreme values in inputs
        source_extreme = torch.abs(source).max() > 1e6
        target_extreme = torch.abs(target).max() > 1e6

        if source_extreme or target_extreme:
            print(f"Warning: Extreme values detected in inputs!")
            print(f"Source max abs value: {torch.abs(source).max()}")
            print(f"Target max abs value: {torch.abs(target).max()}")
            print("This suggests a problem with your model's sampling process.")

            # Normalize inputs to reasonable range for MMD calculation
            source_norm = torch.clamp(source, min=-50, max=50)
            target_norm = torch.clamp(target, min=-50, max=50)
            print("Clamping values to [-50, 50] for MMD calculation")
            source, target = source_norm, target_norm

        # Ensure inputs are valid
        if source.size(0) == 0 or target.size(0) == 0:
            print("Warning: Empty tensors in MMD inputs")
            return torch.tensor(0.0, device=source.device)

        # Check for NaN or Inf in inputs
        if torch.isnan(source).any() or torch.isnan(target).any():
            print("Warning: NaN detected in MMD inputs")
            return torch.tensor(0.0, device=source.device)

        if torch.isinf(source).any() or torch.isinf(target).any():
            print("Warning: Inf detected in MMD inputs")
            return torch.tensor(0.0, device=source.device)

        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(
            source,
            target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma,
        )

        # Check if kernels contain NaN or Inf
        if torch.isnan(kernels).any() or torch.isinf(kernels).any():
            print("Warning: NaN/Inf detected in kernel computation")
            return torch.tensor(0.0, device=source.device)

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]

        loss = torch.mean(XX + YY - XY - YX)

        # Final check for NaN in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/Inf detected in final MMD loss")
            return torch.tensor(0.0, device=source.device)

        return loss
