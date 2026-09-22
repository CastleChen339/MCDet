"""Reusable 3D building blocks for MCDet spatio-temporal detectors."""

from typing import Iterable, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_3tuple(value: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """Convert an integer or tuple-like value to a fixed 3D tuple.

    Args:
        value: Integer or tuple-like kernel/stride value.

    Returns:
        Tuple[int, int, int]: Normalized 3D tuple.
    """

    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, Iterable):
        value = tuple(value)
        if len(value) == 3:
            return value  # type: ignore[return-value]
    raise ValueError("Expected int or tuple/list with length 3 for 3D argument.")


def _autopad_3d(kernel_size: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """Return symmetric padding that preserves tensor size for odd kernels.

    Args:
        kernel_size: Kernel size as int or 3D tuple.

    Returns:
        Tuple[int, int, int]: Symmetric padding for `(t, h, w)`.
    """

    k_t, k_h, k_w = _to_3tuple(kernel_size)
    return (k_t // 2, k_h // 2, k_w // 2)


class ConvBNAct3D(nn.Module):
    """3D convolution block with BatchNorm and SiLU activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size (int or 3-tuple).
        stride: Convolution stride (int or 3-tuple).
        padding: Explicit padding; auto-computed for odd kernels when None.
        groups: Group convolution parameter.
        act: Whether to apply SiLU activation after BatchNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[None, int, Tuple[int, int, int]] = None,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        """Initialize the 3D convolutional block."""

        super().__init__()
        k = _to_3tuple(kernel_size)
        s = _to_3tuple(stride)
        p = _autopad_3d(k) if padding is None else _to_3tuple(padding)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Conv3D + BatchNorm3D + activation.

        Args:
            x: Input tensor of shape `(B, C, T, H, W)`.

        Returns:
            torch.Tensor: Transformed tensor.
        """

        return self.act(self.bn(self.conv(x)))


class Bottleneck3D(nn.Module):
    """Standard residual bottleneck block in 3D space."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
        groups: int = 1,
        kernel_size: Union[int, Tuple[int, int, int]] = (1, 3, 3),
    ) -> None:
        """Initialize a 3D bottleneck with optional residual connection."""

        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvBNAct3D(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = ConvBNAct3D(hidden_channels, out_channels, kernel_size=kernel_size, groups=groups)
        self.use_shortcut = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the bottleneck path and optional residual add.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor after bottleneck and residual add.
        """

        y = self.cv2(self.cv1(x))
        if self.use_shortcut:
            y = y + x
        return y


class C3k2Block3D(nn.Module):
    """3D C3k2 block with C2f-style multi-branch aggregation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        n: Number of inner bottleneck or C3k branches.
        shortcut: Whether to add residual connections in inner bottlenecks.
        expansion: Channel expansion ratio for the hidden path.
        c3k: Use C3k sub-blocks instead of plain bottlenecks when True.
        attn: Use PSA attention sub-blocks instead of bottlenecks when True.
        groups: Group convolution parameter for inner bottlenecks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 2,
        shortcut: bool = True,
        expansion: float = 0.5,
        c3k: bool = False,
        attn: bool = False,
        groups: int = 1,
    ) -> None:
        """Initialize C3k2-style 3D block used by MCDet backbone and neck."""

        super().__init__()
        self.c = int(out_channels * expansion)
        self.cv1 = ConvBNAct3D(in_channels, 2 * self.c, kernel_size=1)
        self.cv2 = ConvBNAct3D((2 + n) * self.c, out_channels, kernel_size=1)

        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck3D(self.c, self.c, shortcut=shortcut, expansion=1.0, groups=groups),
                PSABlock3D(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3kBlock3D(self.c, self.c, n=2, shortcut=shortcut, expansion=1.0, groups=groups)
            if c3k
            else Bottleneck3D(self.c, self.c, shortcut=shortcut, expansion=1.0, groups=groups)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run C2f-style branch expansion, iterative transforms, and fusion.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Fused output tensor.
        """

        # Split channels and iteratively refine the last branch.
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


class C3kBlock3D(nn.Module):
    """3D C3k block with configurable bottleneck kernel shape.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        n: Number of sequential bottleneck repetitions.
        shortcut: Whether inner bottlenecks use residual connections.
        expansion: Channel expansion ratio for hidden path.
        groups: Group convolution parameter.
        kernel_size: Bottleneck convolution kernel shape (int or 3-tuple).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
        groups: int = 1,
        kernel_size: Union[int, Tuple[int, int, int]] = (1, 3, 3),
    ) -> None:
        """Initialize C3k-style two-path bottleneck fusion block in 3D."""

        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvBNAct3D(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = ConvBNAct3D(in_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                Bottleneck3D(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    shortcut=shortcut,
                    expansion=1.0,
                    groups=groups,
                    kernel_size=kernel_size,
                )
                for _ in range(n)
            ]
        )
        self.cv3 = ConvBNAct3D(hidden_channels * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse transformed and bypass branches after repeated bottlenecks.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor with fused branches.
        """

        return self.cv3(torch.cat((self.blocks(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF3D(nn.Module):
    """3D SPPF block using repeated spatial max-pooling over temporal volumes.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        pool_kernel: Spatial kernel size for max-pooling (temporal dim is 1).
        n: Number of successive pooling iterations.
        shortcut: Whether to add a residual connection when in_channels == out_channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_kernel: int = 5,
        n: int = 3,
        shortcut: bool = False,
    ) -> None:
        """Initialize the 3D SPPF module."""

        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = ConvBNAct3D(in_channels, hidden_channels, kernel_size=1, act=False)
        self.pool = nn.MaxPool3d(
            kernel_size=(1, pool_kernel, pool_kernel),
            stride=1,
            padding=(0, pool_kernel // 2, pool_kernel // 2),
        )
        self.cv2 = ConvBNAct3D(hidden_channels * (n + 1), out_channels, kernel_size=1)
        self.n = n
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate multi-scale pooled context and fuse channels.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor after spatial pooling and fusion.
        """

        y = [self.cv1(x)]
        y.extend(self.pool(y[-1]) for _ in range(self.n))
        y = self.cv2(torch.cat(y, dim=1))
        return y + x if self.add else y


class Attention3D(nn.Module):
    """3D multi-head attention block with depthwise positional encoding.

    Args:
        dim: Feature dimension (must be divisible by num_heads).
        num_heads: Number of attention heads.
        attn_ratio: Key dimension ratio relative to head dimension.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention for 3D feature volumes."""

        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = max(int(self.head_dim * attn_ratio), 1)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = ConvBNAct3D(dim, h, kernel_size=1, act=False)
        self.proj = ConvBNAct3D(dim, dim, kernel_size=1, act=False)
        self.pe = ConvBNAct3D(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global 3D attention on flattened `(T*H*W)` tokens.

        Args:
            x: Input tensor of shape `(B, C, T, H, W)`.

        Returns:
            torch.Tensor: Attention-enhanced tensor.
        """

        b, c, t, h, w = x.shape
        n = t * h * w
        # Project and reshape to `(B, heads, tokens, dim)` for attention.
        qkv = self.qkv(x)
        q, k, v = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim],
            dim=2,
        )

        # Scaled dot-product attention across spatiotemporal tokens.
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        y = (v @ attn.transpose(-2, -1)).view(b, c, t, h, w)
        # Add depthwise positional encoding before projection.
        y = y + self.pe(v.reshape(b, c, t, h, w))
        return self.proj(y)


class PSABlock3D(nn.Module):
    """3D PSA block with residual attention and feed-forward paths.

    Args:
        channels: Feature channel count.
        attn_ratio: Key dimension ratio for the attention sub-block.
        num_heads: Number of attention heads.
        shortcut: Whether to add residual connections around attn and ffn.
    """

    def __init__(self, channels: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """Initialize PSA block in 3D."""

        super().__init__()
        self.attn = Attention3D(channels, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(
            ConvBNAct3D(channels, channels * 2, kernel_size=1),
            ConvBNAct3D(channels * 2, channels, kernel_size=1, act=False),
        )
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual attention followed by residual feed-forward refinement.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Refined tensor after attention and FFN.
        """

        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class ChannelSpatialAttention3D(nn.Module):
    """Lightweight 3D attention mixing channel squeeze-excitation and spatial-temporal cues.

    Args:
        channels: Feature channel count.
        reduction: Channel reduction ratio for the squeeze-excitation path.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        """Initialize channel squeeze-excitation and depthwise spatial attention."""

        super().__init__()
        hidden = max(channels // reduction, 8)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1, bias=True)

        self.dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pw = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multiplicative channel and spatial-temporal attention masks.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Reweighted tensor.
        """

        ch = self.avg_pool(x)
        ch = self.fc2(self.act(self.fc1(ch)))
        ch = torch.sigmoid(ch)

        sp = self.bn(self.pw(self.dw(x)))
        sp = torch.sigmoid(sp)

        return x * ch * sp


class C2PSA3D(nn.Module):
    """3D C2PSA-style attention block with channel split and stacked PSA modules.

    Args:
        in_channels: Number of input channels (must equal out_channels).
        out_channels: Number of output channels (must equal in_channels).
        n: Number of stacked PSA attention blocks.
        expansion: Channel expansion ratio for the split path.
    """

    def __init__(self, in_channels: int, out_channels: int, n: int = 1, expansion: float = 0.5) -> None:
        """Initialize C2PSA-style channel split and stacked PSA blocks."""

        super().__init__()
        if in_channels != out_channels:
            raise ValueError("C2PSA3D expects in_channels == out_channels to match C2PSA semantics.")

        self.c = int(in_channels * expansion)
        self.cv1 = ConvBNAct3D(in_channels, 2 * self.c, kernel_size=1)
        self.cv2 = ConvBNAct3D(2 * self.c, in_channels, kernel_size=1)
        num_heads = max(self.c // 64, 1)
        self.m = nn.Sequential(*(PSABlock3D(self.c, attn_ratio=0.5, num_heads=num_heads) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split channels, run stacked PSA blocks, concatenate and fuse.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor after PSA fusion.
        """

        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), dim=1))


class SpatialUpsample3D(nn.Module):
    """Upsample only spatial dimensions while keeping temporal length unchanged.

    Args:
        scale_factor: Spatial upsample factor (default 2).
        mode: Interpolation mode (default 'nearest').
    """

    def __init__(self, scale_factor: int = 2, mode: str = "nearest") -> None:
        """Initialize upsampler for feature pyramid fusion."""

        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolate with `(1, s, s)` scaling to preserve the time axis.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Upsampled tensor.
        """

        return F.interpolate(x, scale_factor=(1, self.scale_factor, self.scale_factor), mode=self.mode)
