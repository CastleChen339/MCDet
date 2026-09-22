"""Point-level matching utilities for MCDet evaluation and visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch


@dataclass
class PointMatchResult:
    """Container for greedy one-to-one point matching results."""

    matches: List[Tuple[int, int, float]]
    unmatched_pred: List[int]
    unmatched_gt: List[int]


def _resolve_hw(image_size: int | Sequence[int]) -> Tuple[float, float]:
    """Return `(height, width)` from an int or a two-value sequence."""

    if isinstance(image_size, int):
        return float(image_size), float(image_size)
    if len(image_size) != 2:
        raise ValueError("image_size must be an int or a (height, width) sequence.")
    return float(image_size[0]), float(image_size[1])


def flatten_pred_keypoints(
    keypoints: torch.Tensor,
    point_conf_threshold: float,
    image_size: int | Sequence[int],
) -> List[Dict[str, float | int]]:
    """Flatten visible predicted keypoints into frame-aware point records.

    Args:
        keypoints: Tensor with shape `(D, K*4)` or `(D, K, 4)`.
        point_conf_threshold: Minimum keypoint visibility confidence.
        image_size: Evaluation image size as int or `(height, width)`.

    Returns:
        A list of records with `det_idx`, `kpt_idx`, `frame`, `x`, `y`, and `score`.
    """

    if keypoints.numel() == 0:
        return []
    if keypoints.ndim == 2:
        if keypoints.shape[-1] % 4 != 0:
            raise ValueError("Flattened keypoints must have channels divisible by 4.")
        keypoints = keypoints.view(keypoints.shape[0], -1, 4)
    if keypoints.ndim != 3 or keypoints.shape[-1] != 4:
        raise ValueError("keypoints must have shape (D, K*4) or (D, K, 4).")

    height, width = _resolve_hw(image_size)
    records: List[Dict[str, float | int]] = []
    keypoints_cpu = keypoints.detach().cpu()

    for det_idx in range(keypoints_cpu.shape[0]):
        for kpt_idx in range(keypoints_cpu.shape[1]):
            x_norm, y_norm, _, score = keypoints_cpu[det_idx, kpt_idx].tolist()
            if score < point_conf_threshold:
                continue
            records.append(
                {
                    "det_idx": det_idx,
                    "kpt_idx": kpt_idx,
                    "frame": kpt_idx,
                    "x": float(x_norm) * width,
                    "y": float(y_norm) * height,
                    "score": float(score),
                }
            )
    return records


def flatten_gt_keypoints(
    keypoints: torch.Tensor,
    valid_mask: torch.Tensor,
    image_size: int | Sequence[int],
) -> List[Dict[str, float | int]]:
    """Flatten valid GT keypoints into frame-aware point records.

    Args:
        keypoints: Tensor with shape `(G, K*4)` or `(G, K, 4)`.
        valid_mask: Boolean tensor with shape `(G, K)`.
        image_size: Evaluation image size as int or `(height, width)`.

    Returns:
        A list of records with `gt_idx`, `kpt_idx`, `frame`, `x`, and `y`.
    """

    if keypoints.numel() == 0:
        return []
    if keypoints.ndim == 2:
        if keypoints.shape[-1] % 4 != 0:
            raise ValueError("Flattened keypoints must have channels divisible by 4.")
        keypoints = keypoints.view(keypoints.shape[0], -1, 4)
    if keypoints.ndim != 3 or keypoints.shape[-1] != 4:
        raise ValueError("keypoints must have shape (G, K*4) or (G, K, 4).")
    if valid_mask.shape != keypoints.shape[:2]:
        raise ValueError("valid_mask must have shape (G, K).")

    height, width = _resolve_hw(image_size)
    records: List[Dict[str, float | int]] = []
    keypoints_cpu = keypoints.detach().cpu()
    valid_cpu = valid_mask.detach().cpu().bool()

    for gt_idx in range(keypoints_cpu.shape[0]):
        for kpt_idx in range(keypoints_cpu.shape[1]):
            if not bool(valid_cpu[gt_idx, kpt_idx]):
                continue
            x_norm, y_norm, _, _ = keypoints_cpu[gt_idx, kpt_idx].tolist()
            records.append(
                {
                    "gt_idx": gt_idx,
                    "kpt_idx": kpt_idx,
                    "frame": kpt_idx,
                    "x": float(x_norm) * width,
                    "y": float(y_norm) * height,
                }
            )
    return records


def match_points_by_distance(
    pred_points: Sequence[Dict[str, float | int]],
    gt_points: Sequence[Dict[str, float | int]],
    distance_threshold_px: float,
) -> PointMatchResult:
    """Greedily match predicted points to nearest same-frame GT points.

    Each predicted point and GT point can be used at most once.  Candidate pairs
    are restricted to the same frame and sorted by Euclidean distance.
    """

    candidates: List[Tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(pred_points):
        pred_frame = int(pred["frame"])
        pred_x = float(pred["x"])
        pred_y = float(pred["y"])
        for gt_idx, gt in enumerate(gt_points):
            if pred_frame != int(gt["frame"]):
                continue
            dx = pred_x - float(gt["x"])
            dy = pred_y - float(gt["y"])
            dist = float((dx * dx + dy * dy) ** 0.5)
            if dist <= distance_threshold_px:
                candidates.append((dist, pred_idx, gt_idx))

    candidates.sort(key=lambda item: item[0])
    used_pred = set()
    used_gt = set()
    matches: List[Tuple[int, int, float]] = []

    for dist, pred_idx, gt_idx in candidates:
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append((pred_idx, gt_idx, dist))

    unmatched_pred = [idx for idx in range(len(pred_points)) if idx not in used_pred]
    unmatched_gt = [idx for idx in range(len(gt_points)) if idx not in used_gt]
    return PointMatchResult(matches, unmatched_pred, unmatched_gt)


def point_match_stats(
    pred_keypoints: torch.Tensor,
    gt_keypoints: torch.Tensor,
    gt_valid_mask: torch.Tensor,
    point_conf_threshold: float,
    distance_threshold_px: float,
    image_size: int | Sequence[int],
) -> Tuple[PointMatchResult, List[Dict[str, float | int]], List[Dict[str, float | int]]]:
    """Flatten predicted/GT keypoints and return point-level match stats."""

    pred_points = flatten_pred_keypoints(
        pred_keypoints,
        point_conf_threshold=point_conf_threshold,
        image_size=image_size,
    )
    gt_points = flatten_gt_keypoints(
        gt_keypoints,
        valid_mask=gt_valid_mask,
        image_size=image_size,
    )
    result = match_points_by_distance(
        pred_points,
        gt_points,
        distance_threshold_px=distance_threshold_px,
    )
    return result, pred_points, gt_points
