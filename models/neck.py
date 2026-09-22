"""MCDet 3D feature pyramid neck for spatio-temporal detection."""

from typing import Tuple

import torch
import torch.nn as nn

from .modules_3d import C3k2Block3D, ConvBNAct3D, SpatialUpsample3D


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


class MCDetNeck3D(nn.Module):
    """FPN + PAN neck for 3D feature pyramid fusion across P3/P4/P5 scales.

    Args:
        in_channels: Tuple of channel counts from backbone outputs (P3, P4, P5).
        depth_mult: Depth multiplier controlling block repeat counts.
    """

    def __init__(self, in_channels: Tuple[int, int, int], depth_mult: float = 1.0) -> None:
        """Initialize top-down and bottom-up fusion paths for P3/P4/P5."""

        super().__init__()
        p3_ch, p4_ch, p5_ch = in_channels
        n2 = _scale_depth(2, depth_mult)

        self.reduce_p5 = ConvBNAct3D(p5_ch, p4_ch, kernel_size=1)
        self.upsample = SpatialUpsample3D(scale_factor=2)
        self.fuse_p4 = C3k2Block3D(p4_ch + p4_ch, p4_ch, n=n2, shortcut=True, expansion=0.5, c3k=True)

        self.reduce_p4 = ConvBNAct3D(p4_ch, p3_ch, kernel_size=1)
        self.fuse_p3 = C3k2Block3D(p3_ch + p3_ch, p3_ch, n=n2, shortcut=True, expansion=0.5, c3k=True)

        self.down_p3 = ConvBNAct3D(p3_ch, p3_ch, kernel_size=3, stride=(1, 2, 2))
        self.pan_p4 = C3k2Block3D(p3_ch + p4_ch, p4_ch, n=n2, shortcut=True, expansion=0.5, c3k=True)

        self.down_p4 = ConvBNAct3D(p4_ch, p4_ch, kernel_size=3, stride=(1, 2, 2))
        self.pan_p5 = C3k2Block3D(p4_ch + p5_ch, p5_ch, n=1, shortcut=True, expansion=0.5, attn=True)

        self.out_channels: Tuple[int, int, int] = (p3_ch, p4_ch, p5_ch)

    def forward(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse backbone features and return refined `(P3, P4, P5)` tensors.

        Args:
            p3: P3 feature map from the backbone.
            p4: P4 feature map from the backbone.
            p5: P5 feature map from the backbone.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Fused `(P3, P4, P5)` outputs.
        """

        # Top-down FPN fusion.
        p5_up = self.upsample(self.reduce_p5(p5))
        p4_td = self.fuse_p4(torch.cat((p5_up, p4), dim=1))

        p4_up = self.upsample(self.reduce_p4(p4_td))
        p3_out = self.fuse_p3(torch.cat((p4_up, p3), dim=1))

        # Bottom-up PAN refinement.
        p3_down = self.down_p3(p3_out)
        p4_out = self.pan_p4(torch.cat((p3_down, p4_td), dim=1))

        p4_down = self.down_p4(p4_out)
        p5_out = self.pan_p5(torch.cat((p4_down, p5), dim=1))

        return p3_out, p4_out, p5_out
