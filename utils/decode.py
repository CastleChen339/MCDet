"""Prediction decoding and post-processing for MCDet 3D outputs."""

from typing import Dict, List, Sequence, Tuple

import torch

from .box_ops import nms_3d


_KEYPOINT_BOX_SPAN_SCALE = 1.2


def _resolve_image_hw(image_size: int | Sequence[int] | None) -> Tuple[float, float]:
    """Return `(height, width)` for pixel-distance point suppression."""

    if image_size is None:
        return 640.0, 640.0
    if isinstance(image_size, int):
        return float(image_size), float(image_size)
    if len(image_size) != 2:
        raise ValueError("image_size must be None, an int, or a (height, width) sequence.")
    return float(image_size[0]), float(image_size[1])


def _make_grid(
    t_size: int,
    h_size: int,
    w_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create temporal and spatial grid coordinates for a 3D feature map.

    Args:
        t_size: Temporal size.
        h_size: Height size.
        w_size: Width size.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: `(t, y, x)` meshgrids.
    """

    t = torch.arange(t_size, device=device, dtype=dtype)
    y = torch.arange(h_size, device=device, dtype=dtype)
    x = torch.arange(w_size, device=device, dtype=dtype)
    tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
    return tt, yy, xx


def _resolve_num_keypoints(prediction: torch.Tensor) -> int:
    """Resolve keypoint count for fixed 4D keypoint layout `(x, y, t, v)`.

    Args:
        prediction: Raw prediction tensor of shape `(B, C, T, H, W)`.

    Returns:
        int: Number of keypoints per location.
    """

    if prediction.shape[1] <= 7:
        raise ValueError("Prediction channel count must be greater than 7 (obj + box).")

    kpt_channels = int(prediction.shape[1] - 7)
    if kpt_channels <= 0:
        raise ValueError("Keypoint channel count must be positive.")

    if kpt_channels % 4 != 0:
        raise ValueError(
            f"Cannot infer keypoint count from {kpt_channels} channels. "
            "Expected channels divisible by 4."
        )
    return kpt_channels // 4


def decode_feature_map(prediction: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Decode one raw feature-map output into normalized boxes and keypoints.

    The expected channel layout is:
    `0: objectness, 1..6: (cx, cy, cz, w, h, d), 7..: keypoints`.
    Keypoint coordinates are decoded box-locally so one trajectory hypothesis can
    express points across multiple spatial cells instead of tying every frame
    point to the same feature-grid location.

    Args:
        prediction: Raw prediction tensor of shape [B, C, T, H, W].

    Returns:
        Dict with keys: obj_logits, scores, boxes, keypoints, kpt_shape, shape.
    """

    if prediction.ndim != 5:
        raise ValueError("Prediction must have shape (B, C, T, H, W).")

    b, _, t_size, h_size, w_size = prediction.shape
    num_keypoints = _resolve_num_keypoints(prediction)

    # Flatten spatial-temporal dimensions for per-location outputs.
    obj_logits = prediction[:, 0].reshape(b, -1)
    obj_scores = torch.sigmoid(obj_logits)

    box_raw = prediction[:, 1:7]
    kpt_raw = prediction[:, 7:7 + num_keypoints * 4]
    kpt_raw = kpt_raw.reshape(b, num_keypoints, 4, t_size, h_size, w_size)

    tt, yy, xx = _make_grid(t_size, h_size, w_size, prediction.device, prediction.dtype)
    tt = tt.unsqueeze(0)
    yy = yy.unsqueeze(0)
    xx = xx.unsqueeze(0)

    # Grid centres (xx + 0.5, yy + 0.5, tt + 0.5) for consistency with
    # the SimOTA centre prior in the loss criterion.
    cx = (torch.sigmoid(box_raw[:, 0]) - 0.5 + (xx + 0.5)) / float(w_size)
    cy = (torch.sigmoid(box_raw[:, 1]) - 0.5 + (yy + 0.5)) / float(h_size)
    cz = (torch.sigmoid(box_raw[:, 2]) - 0.5 + (tt + 0.5)) / float(t_size)
    w = torch.sigmoid(box_raw[:, 3]).clamp(min=1e-4, max=1.0)
    h = torch.sigmoid(box_raw[:, 4]).clamp(min=1e-4, max=1.0)
    d = torch.sigmoid(box_raw[:, 5]).clamp(min=1e-4, max=1.0)

    decoded_kpts = kpt_raw.clone()
    box_cx = cx.unsqueeze(1)
    box_cy = cy.unsqueeze(1)
    box_w = w.unsqueeze(1)
    box_h = h.unsqueeze(1)

    kpt_x_offset = (torch.sigmoid(kpt_raw[:, :, 0]) - 0.5) * float(_KEYPOINT_BOX_SPAN_SCALE)
    kpt_y_offset = (torch.sigmoid(kpt_raw[:, :, 1]) - 0.5) * float(_KEYPOINT_BOX_SPAN_SCALE)
    decoded_kpts[:, :, 0] = (box_cx + kpt_x_offset * box_w).clamp(min=0.0, max=1.0)
    decoded_kpts[:, :, 1] = (box_cy + kpt_y_offset * box_h).clamp(min=0.0, max=1.0)

    # Keypoint slots already correspond to frame indices in training and
    # evaluation, so use deterministic temporal coordinates instead of forcing
    # all slots to share the detection grid's temporal cell.
    frame_t = (
        torch.arange(num_keypoints, device=prediction.device, dtype=prediction.dtype)
        / float(max(num_keypoints, 1))
    ).view(1, num_keypoints, 1, 1, 1)
    decoded_kpts[:, :, 2] = frame_t.expand(b, num_keypoints, t_size, h_size, w_size)
    decoded_kpts[:, :, 3] = torch.sigmoid(kpt_raw[:, :, 3])

    boxes = torch.stack((cx, cy, cz, w, h, d), dim=-1).reshape(b, -1, 6)
    keypoints = decoded_kpts.permute(0, 3, 4, 5, 1, 2).reshape(b, -1, num_keypoints * 4)

    return {
        "obj_logits": obj_logits,
        "scores": obj_scores,
        "boxes": boxes,
        "keypoints": keypoints,
        "kpt_shape": torch.tensor([num_keypoints, 4], device=prediction.device, dtype=torch.long),
        "shape": torch.tensor([t_size, h_size, w_size], device=prediction.device, dtype=torch.long),
    }


def decode_predictions(
    predictions: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode and concatenate all scales from the detection head.

    Args:
        predictions: List of raw prediction tensors from each feature scale.

    Returns:
        Tuple of (scores, boxes, keypoints, level_ids) concatenated across scales.
    """

    decoded_per_level = [decode_feature_map(pred) for pred in predictions]

    scores = torch.cat([item["scores"] for item in decoded_per_level], dim=1)
    boxes = torch.cat([item["boxes"] for item in decoded_per_level], dim=1)
    keypoints = torch.cat([item["keypoints"] for item in decoded_per_level], dim=1)

    level_ids_per_scale: List[torch.Tensor] = []
    for level, item in enumerate(decoded_per_level):
        cells = item["scores"].shape[1]
        level_ids_per_scale.append(torch.full((cells,), level, device=scores.device, dtype=torch.long))

    level_ids = torch.cat(level_ids_per_scale, dim=0).unsqueeze(0).expand(scores.shape[0], -1)
    return scores, boxes, keypoints, level_ids


def _point_aware_filter(
    keypoints_seq: torch.Tensor,
    detection_scores: torch.Tensor,
    point_conf_threshold: float,
    point_nms_threshold_px: float,
    max_points_per_frame: int,
    image_size: int | Sequence[int] | None,
) -> torch.Tensor:
    """Suppress duplicate visible points inside each frame slot.

    Points are ranked by `detection_score * keypoint_visibility`.  Suppressed
    points keep their coordinates but have visibility set to zero so downstream
    point-level evaluation naturally ignores them.
    """

    if keypoints_seq.numel() == 0:
        return keypoints_seq

    filtered = keypoints_seq.clone()
    visible = filtered[:, :, 3] >= float(point_conf_threshold)
    if not visible.any():
        return filtered

    height, width = _resolve_image_hw(image_size)
    threshold = max(float(point_nms_threshold_px), 0.0)
    max_per_frame = int(max_points_per_frame)
    point_scores = filtered[:, :, 3] * detection_scores.view(-1, 1)
    keep_mask = torch.zeros_like(visible)

    for frame_idx in range(filtered.shape[1]):
        frame_candidates = visible[:, frame_idx].nonzero(as_tuple=False).squeeze(-1)
        if frame_candidates.numel() == 0:
            continue

        order_scores = point_scores[frame_candidates, frame_idx]
        order = torch.argsort(order_scores, descending=True)
        ordered_candidates = frame_candidates[order]
        kept_indices: List[int] = []

        for det_idx_tensor in ordered_candidates:
            det_idx = int(det_idx_tensor.item())
            if max_per_frame > 0 and len(kept_indices) >= max_per_frame:
                break

            if threshold > 0.0 and kept_indices:
                current_xy = filtered[det_idx, frame_idx, :2]
                kept_xy = filtered[kept_indices, frame_idx, :2]
                dx = (kept_xy[:, 0] - current_xy[0]) * width
                dy = (kept_xy[:, 1] - current_xy[1]) * height
                distances = torch.sqrt(dx * dx + dy * dy)
                if bool((distances <= threshold).any().item()):
                    continue

            kept_indices.append(det_idx)
            keep_mask[det_idx, frame_idx] = True

    filtered[:, :, 3] = torch.where(
        keep_mask,
        filtered[:, :, 3],
        filtered[:, :, 3].new_zeros(()),
    )
    return filtered


def postprocess_detections(
    scores: torch.Tensor,
    boxes: torch.Tensor,
    keypoints: torch.Tensor,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
    point_conf_threshold: float = 0.50,
    point_nms_threshold_px: float = 4.0,
    max_points_per_frame: int = 20,
    image_size: int | Sequence[int] | None = None,
    apply_nms: bool = True,
) -> List[Dict[str, object]]:
    """Threshold predictions and optionally suppress duplicate boxes and points.

    Args:
        scores: Objectness scores of shape [B, N].
        boxes: Decoded 3D boxes of shape [B, N, 6].
        keypoints: Keypoint predictions of shape [B, N, K*4].
        conf_threshold: Minimum objectness score to keep a detection.
        iou_threshold: IoU threshold for NMS suppression.
        max_detections: Maximum detections retained per sample.
        point_conf_threshold: Keypoint visibility confidence threshold.
        point_nms_threshold_px: Same-frame point duplicate suppression threshold.
        max_points_per_frame: Per-frame cap after point-level suppression.
        image_size: Image size used to convert normalized point coordinates to pixels.
        apply_nms: Apply both 3D box NMS and same-frame point NMS.

    Returns:
        A list of dicts, one per batch sample, with filtered detections.
    """

    results: List[Dict[str, object]] = []
    keypoint_channels = int(keypoints.shape[-1])
    if keypoint_channels % 4 != 0:
        raise ValueError(
            f"Expected keypoint channels divisible by 4, got {keypoint_channels}."
        )
    num_keypoints = keypoint_channels // 4

    for batch_idx in range(scores.shape[0]):
        sample_scores = scores[batch_idx]
        sample_boxes = boxes[batch_idx]
        sample_keypoints = keypoints[batch_idx]

        # Confidence filter before top-k selection and NMS.
        keep = sample_scores >= conf_threshold
        if keep.sum() == 0:
            results.append(
                {
                    "scores": sample_scores.new_zeros((0,)),
                    "bboxes": sample_boxes.new_zeros((0, 6)),
                    "keypoints": sample_keypoints.new_zeros((0, sample_keypoints.shape[-1])),
                    "keypoint_scores": sample_scores.new_zeros((0, num_keypoints)),
                    "keypoint_valid_mask": torch.zeros((0, num_keypoints), dtype=torch.bool, device=sample_scores.device),
                    "visible_keypoints": [],
                }
            )
            continue

        sample_scores = sample_scores[keep]
        sample_boxes = sample_boxes[keep]
        sample_keypoints = sample_keypoints[keep]
        sample_keypoints_seq = sample_keypoints.view(-1, num_keypoints, 4)
        sample_point_scores = sample_keypoints_seq[:, :, 3].amax(dim=1)
        sample_ranking_scores = sample_scores * sample_point_scores

        if sample_ranking_scores.numel() > max_detections:
            topk_scores, topk_idx = torch.topk(sample_ranking_scores, k=max_detections)
            sample_scores = sample_scores[topk_idx]
            sample_ranking_scores = topk_scores
            sample_boxes = sample_boxes[topk_idx]
            sample_keypoints = sample_keypoints[topk_idx]

        if apply_nms:
            nms_keep = nms_3d(
                boxes=sample_boxes,
                scores=sample_ranking_scores,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
            )
        else:
            nms_keep = torch.arange(
                sample_ranking_scores.shape[0],
                device=sample_ranking_scores.device,
            )

        final_scores = sample_ranking_scores[nms_keep]
        final_boxes = sample_boxes[nms_keep]
        final_keypoints = sample_keypoints[nms_keep]

        if final_keypoints.numel() == 0:
            keypoint_scores = final_scores.new_zeros((0, num_keypoints))
            keypoint_valid_mask = torch.zeros((0, num_keypoints), dtype=torch.bool, device=final_scores.device)
            visible_keypoints: List[torch.Tensor] = []
        else:
            keypoints_seq = final_keypoints.view(-1, num_keypoints, 4)
            if apply_nms:
                keypoints_seq = _point_aware_filter(
                    keypoints_seq=keypoints_seq,
                    detection_scores=final_scores,
                    point_conf_threshold=point_conf_threshold,
                    point_nms_threshold_px=point_nms_threshold_px,
                    max_points_per_frame=max_points_per_frame,
                    image_size=image_size,
                )
            final_keypoints = keypoints_seq.reshape(-1, num_keypoints * 4)
            keypoint_scores = keypoints_seq[:, :, 3]
            keypoint_valid_mask = keypoint_scores >= float(point_conf_threshold)

            visible_keypoints = [
                keypoints_seq[det_idx][keypoint_valid_mask[det_idx]]
                for det_idx in range(keypoints_seq.shape[0])
            ]

        results.append(
            {
                "scores": final_scores,
                "bboxes": final_boxes,
                "keypoints": final_keypoints,
                "keypoint_scores": keypoint_scores,
                "keypoint_valid_mask": keypoint_valid_mask,
                "visible_keypoints": visible_keypoints,
            }
        )

    return results
