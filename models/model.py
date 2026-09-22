"""MCDet 3D detector for dim moving object trajectory detection."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from config import ModelConfig
from utils.decode import decode_predictions, postprocess_detections

from .backbone import MCDetBackbone3D
from .head import PoseHead3D
from .neck import MCDetNeck3D


MODEL_SCALES: Dict[str, Tuple[float, float, int]] = {
    # [depth, width, max_channels]
    "n": (0.50, 0.25, 512),
    "s": (0.50, 0.50, 512),
    "m": (0.50, 0.75, 512),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.25, 512),
}


def _resolve_scale(
    model_cfg: ModelConfig,
    scale: Optional[str] = None,
) -> Tuple[float, float, int]:
    """Resolve preset model scale from args or config, defaulting to `n`.

    Args:
        model_cfg: Model configuration containing the default scale name.
        scale: Optional override scale name.

    Returns:
        Tuple[float, float, int]: Depth multiplier, width multiplier, and max channels.
    """

    scale_name = scale if scale is not None else getattr(model_cfg, "model_size", "n")
    if scale_name not in MODEL_SCALES:
        valid = ", ".join(MODEL_SCALES.keys())
        raise ValueError(f"Unknown model scale '{scale_name}'. Available scales: {valid}.")
    return MODEL_SCALES[scale_name]


class MCDetDetector(nn.Module):
    """End-to-end 3D detector combining MCDet backbone, neck, and pose head.

    Args:
        model_cfg: Model configuration dataclass.
        seq_len: Temporal length of input sequences.
        scale: Preset model scale identifier ('n', 's', 'm', 'l', 'x').
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        seq_len: int = 5,
        scale: Optional[str] = None,
    ) -> None:
        """Initialize model components from configuration.

        Args:
            model_cfg: Model configuration dataclass.
            seq_len: Temporal length of input sequences.
            scale: Optional preset model scale identifier.
        """

        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")
        self.model_cfg = model_cfg
        model_cfg_values = vars(model_cfg)
        depth_mult, width_mult, max_channels = _resolve_scale(model_cfg, scale=scale)

        self.backbone = MCDetBackbone3D(
            in_channels=model_cfg.in_channels,
            base_channels=model_cfg.base_channels,
            depth_mult=depth_mult,
            width_mult=width_mult,
            max_channels=max_channels,
        )
        self.neck = MCDetNeck3D(self.backbone.out_channels, depth_mult=depth_mult)
        self.head = PoseHead3D(
            self.neck.out_channels,
            num_classes=model_cfg.num_classes,
            kpt_shape=(seq_len, 4),
            end2end=model_cfg.end2end,
            one2one_full_gradient=bool(
                model_cfg_values.get("one2one_full_gradient", False)
            ),
            one2one_separate_head=bool(
                model_cfg_values.get("one2one_separate_head", False)
            ),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor] | dict[str, List[torch.Tensor]]:
        """Run a forward pass and return raw multi-scale prediction volumes.

        Args:
            x: Input tensor of shape `(B, C, T, H, W)`.

        Returns:
            List[torch.Tensor] | Dict[str, List[torch.Tensor]]: Raw per-scale outputs.
        """

        if x.ndim != 5:
            raise ValueError("Input tensor must have shape (B, C, T, H, W).")

        # Backbone -> neck -> head for multi-scale temporal features.
        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck(p3, p4, p5)
        return self.head((n3, n4, n5))

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.30,
        point_conf_threshold: float = 0.50,
        point_nms_threshold_px: float = 4.0,
        max_points_per_frame: int = 20,
        image_size: int | Tuple[int, int] = 640,
        max_detections: int = 200,
    ) -> List[Dict[str, object]]:
        """Run inference, decode predictions, and apply 3D NMS post-processing.

        Args:
            x: Input tensor of shape [B, C, T, H, W].
            conf_threshold: Confidence threshold for candidate filtering.
            iou_threshold: IoU threshold for NMS suppression.
            point_conf_threshold: Keypoint visibility confidence threshold.
            point_nms_threshold_px: Same-frame point duplicate suppression threshold.
            max_points_per_frame: Maximum visible points retained per frame.
            image_size: Image size used for pixel-distance point suppression.
            max_detections: Maximum number of retained detections per sample.

        Returns:
            A list of dicts, one per batch sample, containing decoded detections.
        """

        raw_outputs = self.forward(x)
        if isinstance(raw_outputs, dict):
            # Prefer the one2one branch for inference when available.
            raw_outputs = raw_outputs.get("one2one", raw_outputs.get("one2many", []))
        if not isinstance(raw_outputs, list):
            raise TypeError("Model head must return a prediction list for decode.")
        scores, boxes, keypoints, _ = decode_predictions(raw_outputs)
        detections = postprocess_detections(
            scores=scores,
            boxes=boxes,
            keypoints=keypoints,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            point_conf_threshold=point_conf_threshold,
            point_nms_threshold_px=point_nms_threshold_px,
            max_points_per_frame=max_points_per_frame,
            image_size=image_size,
            max_detections=max_detections,
            apply_nms=not self.model_cfg.end2end,
        )
        return detections


def build_model(
    model_cfg: Optional[ModelConfig] = None,
    seq_len: int = 5,
    scale: Optional[str] = None,
) -> MCDetDetector:
    """Factory function to build the 3D detector with default config when needed.

    Args:
        model_cfg: Optional model configuration; defaults to `ModelConfig()`.
        seq_len: Temporal length of input sequences.
        scale: Optional preset model scale identifier.

    Returns:
        MCDetDetector: Instantiated detector module.
    """

    cfg = model_cfg if model_cfg is not None else ModelConfig()
    return MCDetDetector(cfg, seq_len=seq_len, scale=scale)
