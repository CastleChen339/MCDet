"""Evaluation utilities for 3D trajectory and keypoint detection."""

from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from config import AppConfig
from utils.decode import decode_predictions, postprocess_detections
from utils.keypoint_ops import default_keypoints_from_boxes, derive_group_keypoints
from utils.misc import move_targets_to_device
from utils.point_ops import point_match_stats


def _nearest_gt_point_stats(
    pred_points: Sequence[Dict[str, float | int]],
    gt_points: Sequence[Dict[str, float | int]],
    thresholds: Sequence[float] = (8.0, 16.0, 32.0),
) -> Dict[str, float]:
    """Return same-frame nearest-prediction distance stats for GT points."""

    if len(gt_points) == 0:
        return {
            "gt_count": 0.0,
            "gt_with_same_frame_pred": 0.0,
            "distance_sum": 0.0,
            **{f"within_{int(threshold)}px": 0.0 for threshold in thresholds},
        }

    stats = {
        "gt_count": float(len(gt_points)),
        "gt_with_same_frame_pred": 0.0,
        "distance_sum": 0.0,
        **{f"within_{int(threshold)}px": 0.0 for threshold in thresholds},
    }

    for gt in gt_points:
        gt_frame = int(gt["frame"])
        gt_x = float(gt["x"])
        gt_y = float(gt["y"])
        best_dist: float | None = None

        for pred in pred_points:
            if int(pred["frame"]) != gt_frame:
                continue
            dx = float(pred["x"]) - gt_x
            dy = float(pred["y"]) - gt_y
            dist = float((dx * dx + dy * dy) ** 0.5)
            if best_dist is None or dist < best_dist:
                best_dist = dist

        if best_dist is None:
            continue

        stats["gt_with_same_frame_pred"] += 1.0
        stats["distance_sum"] += best_dist
        for threshold in thresholds:
            if best_dist <= float(threshold):
                stats[f"within_{int(threshold)}px"] += 1.0

    return stats


class Evaluator:
    """Compute detection metrics for spatio-temporal trajectory predictions."""

    def __init__(self, cfg: AppConfig) -> None:
        """Initialize evaluator with confidence and IoU thresholds."""

        self.conf_threshold = cfg.infer.conf_threshold
        self.iou_threshold = cfg.infer.iou_threshold
        self.point_conf_threshold = cfg.infer.point_conf_threshold
        self.point_match_threshold_px = float(getattr(cfg.infer, "point_match_threshold_px", 8.0))
        self.point_nms_threshold_px = float(getattr(cfg.infer, "point_nms_threshold_px", 4.0))
        self.max_points_per_frame = int(getattr(cfg.infer, "max_points_per_frame", 100))
        self.max_detections = cfg.infer.max_detections
        self.image_size = cfg.dataset.crop_size
        self.apply_nms = not cfg.model.end2end

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        print_interval: int = 0,
        log_fn: Optional[Callable[[str], None]] = None,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run validation and return precision/recall/F1 and keypoint error.

        Args:
            model: Model to evaluate.
            dataloader: Validation DataLoader.
            device: Device for inference.
            print_interval: Validation progress logging interval in steps.
            log_fn: Optional logging function used for progress prints.
            epoch: Optional epoch index for progress context.
            total_epochs: Optional total number of epochs for progress context.

        Returns:
            Dict[str, float]: Precision/recall/F1 and keypoint metrics.
        """

        model.eval()
        log = log_fn or print

        total_steps: Optional[int]
        try:
            total_steps = len(dataloader)
        except TypeError:
            total_steps = None

        tp = 0
        fp = 0
        fn = 0
        point_distances: List[float] = []
        keypoint_vis_accuracies: List[float] = []
        total_samples = 0
        total_detections = 0
        total_pred_points = 0
        total_gt_points = 0
        nearest_gt_count = 0.0
        nearest_gt_with_same_frame_pred = 0.0
        nearest_gt_distance_sum = 0.0
        nearest_gt_within_8px = 0.0
        nearest_gt_within_16px = 0.0
        nearest_gt_within_32px = 0.0
        eps = 1e-9

        for step, (images, targets) in enumerate(dataloader, start=1):
            images = images.to(device, non_blocking=True)
            targets = move_targets_to_device(targets, device)

            raw_predictions = model(images)
            if isinstance(raw_predictions, dict):
                raw_predictions = raw_predictions.get("one2one", raw_predictions.get("one2many", []))
            if not isinstance(raw_predictions, Sequence):
                raise TypeError("Model must return a prediction list for evaluation decode.")

            scores, boxes, keypoints, _ = decode_predictions(raw_predictions)
            keypoint_channels = int(keypoints.shape[-1])
            if keypoint_channels % 4 != 0:
                raise ValueError(f"Expected keypoint channels divisible by 4, got {keypoint_channels}.")
            num_keypoints = max(keypoint_channels // 4, 1)

            detections = postprocess_detections(
                scores=scores,
                boxes=boxes,
                keypoints=keypoints,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                point_conf_threshold=self.point_conf_threshold,
                point_nms_threshold_px=self.point_nms_threshold_px,
                max_points_per_frame=self.max_points_per_frame,
                image_size=self.image_size,
                max_detections=self.max_detections,
                apply_nms=self.apply_nms,
            )

            for detection, target in zip(detections, targets):
                pred_boxes = detection["bboxes"]
                pred_kpts = detection["keypoints"]

                gt_boxes = target.get("bboxes")
                if not isinstance(gt_boxes, torch.Tensor):
                    gt_boxes = pred_boxes.new_zeros((0, 6))
                gt_kpts, gt_kpt_valid = derive_group_keypoints(
                    target,
                    num_keypoints=num_keypoints,
                    device=gt_boxes.device,
                )
                gt_kpts = gt_kpts.to(gt_boxes.device)
                gt_kpt_valid = gt_kpt_valid.to(gt_boxes.device)
                if gt_kpts.shape[0] != gt_boxes.shape[0]:
                    gt_kpts = default_keypoints_from_boxes(gt_boxes, num_keypoints)
                    gt_kpt_valid = torch.ones(
                        (gt_boxes.shape[0], num_keypoints),
                        dtype=torch.bool,
                        device=gt_boxes.device,
                    )

                match_result, pred_points, gt_points = point_match_stats(
                    pred_kpts,
                    gt_kpts,
                    gt_kpt_valid,
                    point_conf_threshold=self.point_conf_threshold,
                    distance_threshold_px=self.point_match_threshold_px,
                    image_size=self.image_size,
                )

                tp += len(match_result.matches)
                fp += len(match_result.unmatched_pred)
                fn += len(match_result.unmatched_gt)
                point_distances.extend(dist for _, _, dist in match_result.matches)
                total_samples += 1
                total_detections += int(pred_boxes.shape[0])
                total_pred_points += len(pred_points)
                total_gt_points += len(gt_points)

                nearest_stats = _nearest_gt_point_stats(pred_points, gt_points)
                nearest_gt_count += nearest_stats["gt_count"]
                nearest_gt_with_same_frame_pred += nearest_stats["gt_with_same_frame_pred"]
                nearest_gt_distance_sum += nearest_stats["distance_sum"]
                nearest_gt_within_8px += nearest_stats["within_8px"]
                nearest_gt_within_16px += nearest_stats["within_16px"]
                nearest_gt_within_32px += nearest_stats["within_32px"]

                if len(gt_points) == 0:
                    vis_acc = 1.0 if len(pred_points) == 0 else 0.0
                else:
                    vis_acc = max(
                        0.0,
                        1.0 - abs(len(pred_points) - len(gt_points)) / float(len(gt_points)),
                    )
                keypoint_vis_accuracies.append(vis_acc)

            if print_interval > 0:
                should_log = (step % print_interval == 0)
                if total_steps is not None and step == total_steps:
                    should_log = True

                if should_log:
                    precision = tp / (tp + fp + eps)
                    recall = tp / (tp + fn + eps)
                    f1 = 2.0 * precision * recall / (precision + recall + eps)

                    if epoch is not None and total_epochs is not None:
                        prefix = f"[Val] epoch={epoch}/{total_epochs} "
                    elif epoch is not None:
                        prefix = f"[Val] epoch={epoch} "
                    else:
                        prefix = "[Val] "
                    if total_steps is not None:
                        log(
                            f"{prefix}step={step}/{total_steps} "
                            f"tp={tp} fp={fp} fn={fn} "
                            f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}"
                        )
                    else:
                        log(
                            f"{prefix}step={step} "
                            f"tp={tp} fp={fp} fn={fn} "
                            f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}"
                        )

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        mean_point_dist_px = sum(point_distances) / max(len(point_distances), 1)
        crop_h, crop_w = self.image_size
        mean_kpt_l1 = mean_point_dist_px / max(float(crop_h), float(crop_w), 1.0)
        mean_kpt_vis_acc = sum(keypoint_vis_accuracies) / max(len(keypoint_vis_accuracies), 1)
        mean_detections_per_sample = total_detections / max(total_samples, 1)
        mean_pred_points_per_sample = total_pred_points / max(total_samples, 1)
        mean_gt_points_per_sample = total_gt_points / max(total_samples, 1)
        mean_nearest_gt_point_dist_px = (
            nearest_gt_distance_sum / max(nearest_gt_with_same_frame_pred, 1.0)
        )
        nearest_gt_same_frame_coverage = nearest_gt_with_same_frame_pred / max(nearest_gt_count, 1.0)
        nearest_gt_within_8px = nearest_gt_within_8px / max(nearest_gt_count, 1.0)
        nearest_gt_within_16px = nearest_gt_within_16px / max(nearest_gt_count, 1.0)
        nearest_gt_within_32px = nearest_gt_within_32px / max(nearest_gt_count, 1.0)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "mean_kpt_l1": float(mean_kpt_l1),
            "mean_point_dist_px": float(mean_point_dist_px),
            "mean_kpt_vis_acc": float(mean_kpt_vis_acc),
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "mean_detections_per_sample": float(mean_detections_per_sample),
            "mean_pred_points_per_sample": float(mean_pred_points_per_sample),
            "mean_gt_points_per_sample": float(mean_gt_points_per_sample),
            "mean_nearest_gt_point_dist_px": float(mean_nearest_gt_point_dist_px),
            "nearest_gt_same_frame_coverage": float(nearest_gt_same_frame_coverage),
            "nearest_gt_within_8px": float(nearest_gt_within_8px),
            "nearest_gt_within_16px": float(nearest_gt_within_16px),
            "nearest_gt_within_32px": float(nearest_gt_within_32px),
        }
