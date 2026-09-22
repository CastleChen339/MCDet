"""Shared keypoint utilities used by both loss criterion and evaluation."""

from typing import Dict, Tuple

import torch


def default_keypoints_from_boxes(
    boxes: torch.Tensor,
    num_keypoints: int,
) -> torch.Tensor:
    """Create fallback keypoint sequences from box centres.

    When ground-truth points are unavailable for a trajectory, this function
    synthesises evenly-spaced keypoints along the temporal axis at each box's
    spatial centre.

    Args:
        boxes: Ground-truth boxes in normalized `(cx, cy, cz, w, h, d)` format.
        num_keypoints: Number of keypoints per box.

    Returns:
        torch.Tensor: Flattened keypoints of shape `(M, num_keypoints * 4)`.
    """

    keypoints = boxes.new_zeros((boxes.shape[0], num_keypoints, 4))
    keypoints[:, :, 0] = boxes[:, 0:1]
    keypoints[:, :, 1] = boxes[:, 1:2]
    if num_keypoints > 1:
        t_values = (
            torch.arange(num_keypoints, device=boxes.device, dtype=boxes.dtype)
            / float(num_keypoints)
        )
        keypoints[:, :, 2] = t_values.unsqueeze(0).expand(boxes.shape[0], -1)
    else:
        keypoints[:, :, 2] = boxes[:, 2:3]
    keypoints[:, :, 3] = 1.0
    return keypoints.reshape(boxes.shape[0], -1)


def derive_group_keypoints(
    target: Dict[str, object],
    num_keypoints: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build fixed-length keypoint targets and per-frame validity masks for each box.

    This function maps per-group point annotations (normalized x, y, t) onto a
    fixed-size keypoint grid of length ``num_keypoints``.  When the source
    sequence length ``source_seq_len`` differs from ``num_keypoints`` the frame
    indices are linearly interpolated so that the mapping is lossless.

    Args:
        target: Target dictionary containing points, boxes, and group IDs.
        num_keypoints: Number of keypoints per trajectory.
        device: Device for output tensors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Flattened keypoints and validity mask.
    """

    boxes = target.get("bboxes")
    box_group_ids = target.get("bboxes_group_ids")
    points = target.get("points")
    point_group_ids = target.get("points_group_ids")

    if not isinstance(boxes, torch.Tensor) or boxes.numel() == 0:
        if isinstance(points, torch.Tensor):
            empty = points.new_zeros((0, num_keypoints * 4), device=device)
            return empty, torch.zeros((0, num_keypoints), dtype=torch.bool, device=device)
        empty = torch.zeros((0, num_keypoints * 4), dtype=torch.float32, device=device)
        return empty, torch.zeros((0, num_keypoints), dtype=torch.bool, device=device)

    boxes = boxes.float().to(device)
    seq_keypoints = default_keypoints_from_boxes(boxes, num_keypoints).view(
        boxes.shape[0], num_keypoints, 4
    )
    valid_mask = torch.zeros((boxes.shape[0], num_keypoints), dtype=torch.bool, device=device)
    seq_keypoints[:, :, 3] = 0.0

    if (
        not isinstance(points, torch.Tensor)
        or not isinstance(box_group_ids, torch.Tensor)
        or not isinstance(point_group_ids, torch.Tensor)
    ):
        valid_mask[:] = True
        return seq_keypoints.reshape(boxes.shape[0], -1), valid_mask

    points = points.float().to(device)
    box_group_ids = box_group_ids.to(device)
    point_group_ids = point_group_ids.to(device)
    source_seq_len = target.get("seq_len", num_keypoints)
    if isinstance(source_seq_len, torch.Tensor):
        source_seq_len = int(source_seq_len.item())
    source_seq_len = max(int(source_seq_len), 1)
    source_max_index = max(source_seq_len - 1, 0)

    for box_idx, gid in enumerate(box_group_ids.tolist()):
        mask = point_group_ids == gid
        if not mask.any():
            continue

        group_points = points[mask]
        source_frame_indices = torch.clamp(
            torch.round(group_points[:, 2] * float(source_seq_len)),
            min=0,
            max=float(source_max_index),
        )

        if num_keypoints <= 1 or source_seq_len <= 1:
            frame_indices = torch.zeros_like(source_frame_indices, dtype=torch.long)
        elif num_keypoints == source_seq_len:
            frame_indices = source_frame_indices.long()
        else:
            frame_indices = torch.clamp(
                torch.round(
                    source_frame_indices
                    * float(num_keypoints - 1)
                    / float(source_seq_len - 1)
                ),
                min=0,
                max=float(num_keypoints - 1),
            ).long()

        for frame_idx in range(num_keypoints):
            frame_mask = frame_indices == frame_idx
            if not frame_mask.any():
                continue
            mean_point = group_points[frame_mask].mean(dim=0)
            valid_mask[box_idx, frame_idx] = True
            seq_keypoints[box_idx, frame_idx, 0] = mean_point[0]
            seq_keypoints[box_idx, frame_idx, 1] = mean_point[1]
            seq_keypoints[box_idx, frame_idx, 2] = mean_point[2]
            seq_keypoints[box_idx, frame_idx, 3] = 1.0

    return seq_keypoints.reshape(boxes.shape[0], -1), valid_mask
