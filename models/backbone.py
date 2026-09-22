"""MCDet 3D backbone for spatio-temporal dim target detection."""

from typing import Tuple

import torch
import torch.nn as nn

from .modules_3d import C2PSA3D, C3k2Block3D, ConvBNAct3D, SPPF3D


def _scale_depth(repeats: int, depth_mult: float) -> int:
    """Scale repeat count following compound-scaling behavior.

    Args:
        repeats: Base repeat count.
        depth_mult: Depth multiplier.

    Returns:
        int: Scaled repeat count.
    """

    if repeats <= 1:
        return repeats
    return max(int(round(repeats * depth_mult)), 1)


def _make_divisible(value: int, divisor: int = 8) -> int:
    """Round channels to be divisible by `divisor` for stable CUDA kernels.

    Args:
        value: Raw channel count.
        divisor: Divisor used for channel alignment.

    Returns:
        int: Adjusted channel count.
    """

    return max(divisor, int((value + divisor / 2) // divisor) * divisor)


def _scale_channels(channels: int, width_mult: float, max_channels: int) -> int:
    """Scale channel count with width multiplier and cap to max channels.

    Args:
        channels: Base channel count.
        width_mult: Width multiplier.
        max_channels: Maximum channel cap.

    Returns:
        int: Scaled channel count.
    """

    scaled = _make_divisible(int(channels * width_mult), divisor=8)
    return min(scaled, max_channels)


class MCDetBackbone3D(nn.Module):
    """MCDet backbone producing multi-scale 3D feature maps for neck fusion.

    Args:
        in_channels: Input channel count (e.g. 1 for grayscale sequences).
        base_channels: Base channel multiplier; actual channels are derived via width scaling.
        depth_mult: Depth multiplier controlling block repeat counts.
        width_mult: Width multiplier controlling channel scaling.
        max_channels: Upper bound on channel counts after scaling.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 16,
        depth_mult: float = 1.0,
        width_mult: float = 1.0,
        max_channels: int = 1024,
    ) -> None:
        """Initialize backbone stages and expose P3/P4/P5 output channel metadata."""

        super().__init__()

        c1 = _scale_channels(base_channels * 2, width_mult, max_channels)
        c2 = _scale_channels(base_channels * 4, width_mult, max_channels)
        c3 = _scale_channels(base_channels * 8, width_mult, max_channels)
        c4 = _scale_channels(base_channels * 16, width_mult, max_channels)
        c5 = _scale_channels(base_channels * 32, width_mult, max_channels)

        n2 = _scale_depth(2, depth_mult)
        n_psa = _scale_depth(2, depth_mult)

        self.layer0 = ConvBNAct3D(in_channels, c1, kernel_size=3, stride=(1, 2, 2))
        self.layer1 = ConvBNAct3D(c1, c2, kernel_size=3, stride=(1, 2, 2))
        self.layer2 = C3k2Block3D(c2, c3, n=n2, shortcut=False, expansion=0.25, c3k=False)

        self.layer3 = ConvBNAct3D(c3, c3, kernel_size=3, stride=(1, 2, 2))
        self.layer4 = C3k2Block3D(c3, c4, n=n2, shortcut=False, expansion=0.25, c3k=False)  # P3

        self.layer5 = ConvBNAct3D(c4, c4, kernel_size=3, stride=(1, 2, 2))
        self.layer6 = C3k2Block3D(c4, c4, n=n2, shortcut=True, expansion=0.5, c3k=True)  # P4

        self.layer7 = ConvBNAct3D(c4, c5, kernel_size=3, stride=(1, 2, 2))
        self.layer8 = C3k2Block3D(c5, c5, n=n2, shortcut=True, expansion=0.5, c3k=True)
        self.layer9 = SPPF3D(c5, c5, pool_kernel=5, n=3, shortcut=True)
        self.layer10 = nn.Sequential(*(C2PSA3D(c5, c5, n=1) for _ in range(n_psa)))  # P5

        self.out_channels: Tuple[int, int, int] = (c4, c4, c5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return multi-scale feature maps `(P3, P4, P5)` for neck fusion.

        Args:
            x: Input tensor of shape `(B, C, T, H, W)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: `(P3, P4, P5)` feature maps.
        """

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        p3 = self.layer4(x)

        x = self.layer5(p3)
        p4 = self.layer6(x)

        x = self.layer7(p4)
        x = self.layer8(x)
        x = self.layer9(x)
        p5 = self.layer10(x)

        return p3, p4, p5
