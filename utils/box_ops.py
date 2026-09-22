"""3D bounding box geometry utilities for trajectory-volume regression."""

import math
import torch


def cxcyczwhd_to_xyzxyz(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from `(cx, cy, cz, w, h, d)` to `(x1, y1, z1, x2, y2, z2)`.

    Args:
        boxes: Input boxes in center-size format.

    Returns:
        torch.Tensor: Converted boxes in corner format.
    """

    if boxes.numel() == 0:
        return boxes.new_zeros((0, 6))

    cx, cy, cz, w, h, d = boxes.unbind(dim=-1)
    half_w = w * 0.5
    half_h = h * 0.5
    half_d = d * 0.5

    x1 = cx - half_w
    y1 = cy - half_h
    z1 = cz - half_d
    x2 = cx + half_w
    y2 = cy + half_h
    z2 = cz + half_d

    return torch.stack((x1, y1, z1, x2, y2, z2), dim=-1)


def xyzxyz_to_cxcyczwhd(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from `(x1, y1, z1, x2, y2, z2)` to `(cx, cy, cz, w, h, d)`.

    Args:
        boxes: Input boxes in corner format.

    Returns:
        torch.Tensor: Converted boxes in center-size format.
    """

    if boxes.numel() == 0:
        return boxes.new_zeros((0, 6))

    x1, y1, z1, x2, y2, z2 = boxes.unbind(dim=-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    cz = (z1 + z2) * 0.5
    w = (x2 - x1).clamp(min=0.0)
    h = (y2 - y1).clamp(min=0.0)
    d = (z2 - z1).clamp(min=0.0)

    return torch.stack((cx, cy, cz, w, h, d), dim=-1)


def box_iou_3d(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute pairwise IoU between two sets of 3D boxes in `cxcyczwhd` format.

    Args:
        boxes1: First set of boxes.
        boxes2: Second set of boxes.
        eps: Numerical stability epsilon.

    Returns:
        torch.Tensor: IoU matrix of shape `(len(boxes1), len(boxes2))`.
    """

    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    b1 = cxcyczwhd_to_xyzxyz(boxes1)
    b2 = cxcyczwhd_to_xyzxyz(boxes2)

    max_xyz = torch.min(b1[:, None, 3:], b2[None, :, 3:])
    min_xyz = torch.max(b1[:, None, :3], b2[None, :, :3])
    inter_dims = (max_xyz - min_xyz).clamp(min=0.0)
    inter = inter_dims[..., 0] * inter_dims[..., 1] * inter_dims[..., 2]

    vol1_dims = (b1[:, 3:] - b1[:, :3]).clamp(min=0.0)
    vol2_dims = (b2[:, 3:] - b2[:, :3]).clamp(min=0.0)
    vol1 = vol1_dims[:, 0] * vol1_dims[:, 1] * vol1_dims[:, 2]
    vol2 = vol2_dims[:, 0] * vol2_dims[:, 1] * vol2_dims[:, 2]

    union = vol1[:, None] + vol2[None, :] - inter
    return inter / (union + eps)


def box_iou_3d_aligned(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute aligned IoU for corresponding box pairs in `cxcyczwhd` format.

    Args:
        boxes1: First set of boxes.
        boxes2: Second set of boxes (aligned with `boxes1`).
        eps: Numerical stability epsilon.

    Returns:
        torch.Tensor: IoU values for each aligned pair.
    """

    if boxes1.shape != boxes2.shape:
        raise ValueError("Aligned IoU expects boxes1 and boxes2 to have the same shape.")

    if boxes1.numel() == 0:
        return boxes1.new_zeros((0,))

    b1 = cxcyczwhd_to_xyzxyz(boxes1)
    b2 = cxcyczwhd_to_xyzxyz(boxes2)

    max_xyz = torch.min(b1[:, 3:], b2[:, 3:])
    min_xyz = torch.max(b1[:, :3], b2[:, :3])
    inter_dims = (max_xyz - min_xyz).clamp(min=0.0)
    inter = inter_dims[:, 0] * inter_dims[:, 1] * inter_dims[:, 2]

    vol1_dims = (b1[:, 3:] - b1[:, :3]).clamp(min=0.0)
    vol2_dims = (b2[:, 3:] - b2[:, :3]).clamp(min=0.0)
    vol1 = vol1_dims[:, 0] * vol1_dims[:, 1] * vol1_dims[:, 2]
    vol2 = vol2_dims[:, 0] * vol2_dims[:, 1] * vol2_dims[:, 2]

    union = vol1 + vol2 - inter
    return inter / (union + eps)


def box_ciou_3d_aligned(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute aligned 3D CIoU for corresponding box pairs in `cxcyczwhd` format.

    Args:
        boxes1: First set of boxes.
        boxes2: Second set of boxes (aligned with `boxes1`).
        eps: Numerical stability epsilon.

    Returns:
        torch.Tensor: CIoU values for each aligned pair.
    """

    if boxes1.shape != boxes2.shape:
        raise ValueError("Aligned CIoU expects boxes1 and boxes2 to have the same shape.")

    if boxes1.numel() == 0:
        return boxes1.new_zeros((0,))

    iou = box_iou_3d_aligned(boxes1, boxes2, eps=eps)
    b1 = cxcyczwhd_to_xyzxyz(boxes1)
    b2 = cxcyczwhd_to_xyzxyz(boxes2)

    center_dist_sq = ((boxes1[:, :3] - boxes2[:, :3]) ** 2).sum(dim=-1)

    enc_min = torch.min(b1[:, :3], b2[:, :3])
    enc_max = torch.max(b1[:, 3:], b2[:, 3:])
    enc_diag_sq = ((enc_max - enc_min) ** 2).sum(dim=-1).clamp(min=eps)

    w1 = boxes1[:, 3].clamp(min=eps)
    h1 = boxes1[:, 4].clamp(min=eps)
    d1 = boxes1[:, 5].clamp(min=eps)
    w2 = boxes2[:, 3].clamp(min=eps)
    h2 = boxes2[:, 4].clamp(min=eps)
    d2 = boxes2[:, 5].clamp(min=eps)

    v_wh = (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
    v_wd = (torch.atan(w2 / d2) - torch.atan(w1 / d1)) ** 2
    v_hd = (torch.atan(h2 / d2) - torch.atan(h1 / d1)) ** 2
    v = (4.0 / (math.pi**2)) * (v_wh + v_wd + v_hd) / 3.0
    alpha = (v / (1.0 - iou + v + eps)).detach()

    ciou = iou - center_dist_sq / enc_diag_sq - alpha * v
    return ciou.clamp(min=-1.0, max=1.0)


def nms_3d(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.30,
    max_detections: int = 200,
) -> torch.Tensor:
    """Run greedy 3D NMS over trajectory volumes and return kept indices.

    Args:
        boxes: Boxes in `(cx, cy, cz, w, h, d)` format.
        scores: Objectness scores for each box.
        iou_threshold: IoU threshold for suppression.
        max_detections: Maximum number of boxes to keep.

    Returns:
        torch.Tensor: Indices of kept boxes.
    """

    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)

    order = torch.argsort(scores, descending=True)
    keep = []

    while order.numel() > 0 and len(keep) < max_detections:
        # Greedy selection: keep highest score, suppress boxes with high IoU.
        current = order[0].item()
        keep.append(current)

        if order.numel() == 1:
            break

        rest = order[1:]
        ious = box_iou_3d(boxes[current : current + 1], boxes[rest]).squeeze(0)
        order = rest[ious <= iou_threshold]

    return torch.tensor(keep, device=boxes.device, dtype=torch.long)
