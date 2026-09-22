"""MCDet loss for 3D trajectory volume and center keypoint prediction."""

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LossConfig
from utils.box_ops import box_ciou_3d_aligned, box_iou_3d_aligned
from utils.decode import decode_feature_map
from utils.keypoint_ops import default_keypoints_from_boxes, derive_group_keypoints


class TemporalDetectionCriterion(nn.Module):
    """Combined objectness, trajectory volume, and keypoint regression loss.

    Supports single-branch and dual-branch (one2many + one2one) end-to-end
    training modes with configurable VFL/BCE/QFL objectness and CIoU/IoU regression.

    Improvements over baseline:
      - Point-aware SimOTA dynamic label assignment (multi-positive per GT)
      - QFL / VFL-with-quality-min for cold-start mitigation
      - Keypoint-first optimization with auxiliary box regression

    Args:
        loss_cfg: LossConfig dataclass with weighting coefficients and loss type settings.
    """

    def __init__(self, loss_cfg: LossConfig) -> None:
        """Initialize criterion with configurable balancing coefficients.

        Args:
            loss_cfg: LossConfig dataclass with weighting and loss type settings.
        """

        super().__init__()
        self.loss_cfg = loss_cfg
        self.one2one_weight = float(loss_cfg.one2one_weight)
        self.obj_loss_type = str(getattr(loss_cfg, "obj_loss_type", "qfl")).lower()
        if self.obj_loss_type not in {"bce", "varifocal", "qfl"}:
            raise ValueError(
                f"Unsupported obj_loss_type={self.obj_loss_type}. "
                "Expected 'bce', 'varifocal', or 'qfl'."
            )

        self.iou_loss_type = str(getattr(loss_cfg, "iou_loss_type", "ciou")).lower()
        if self.iou_loss_type not in {"iou", "ciou"}:
            raise ValueError(
                f"Unsupported iou_loss_type={self.iou_loss_type}. Expected 'iou' or 'ciou'."
            )

        self.vfl_alpha = float(getattr(loss_cfg, "vfl_alpha", 0.75))
        self.vfl_gamma = float(getattr(loss_cfg, "vfl_gamma", 2.0))
        self.vfl_quality_min = float(getattr(loss_cfg, "vfl_quality_min", 0.1))
        self.qfl_beta = float(getattr(loss_cfg, "qfl_beta", 2.0))
        self.vfl_warmup_epochs = int(getattr(loss_cfg, "vfl_warmup_epochs", 3))
        self.obj_pos_weight = float(getattr(loss_cfg, "obj_pos_weight", 1.0))
        self.obj_neg_weight = float(getattr(loss_cfg, "obj_neg_weight", 0.25))
        self.obj_hard_negative_ratio = max(
            float(getattr(loss_cfg, "obj_hard_negative_ratio", 0.0)),
            0.0,
        )
        self.obj_hard_negative_min = max(
            int(getattr(loss_cfg, "obj_hard_negative_min", 0)),
            0,
        )
        self.simota_topk = int(getattr(loss_cfg, "simota_topk", 10))
        self.simota_max_positives_per_gt = max(
            int(getattr(loss_cfg, "simota_max_positives_per_gt", 0)),
            0,
        )
        self.simota_reg_weight = float(getattr(loss_cfg, "simota_reg_weight", 0.5))
        self.simota_box_weight = float(getattr(loss_cfg, "simota_box_weight", self.simota_reg_weight))
        self.simota_point_weight = float(getattr(loss_cfg, "simota_point_weight", 6.0))
        self.simota_vis_weight = float(getattr(loss_cfg, "simota_vis_weight", 0.2))
        self.point_quality_sigma = max(float(getattr(loss_cfg, "point_quality_sigma", 0.02)), 1e-6)
        self.kpt_time_weight = float(getattr(loss_cfg, "kpt_time_weight", 0.2))
        self.one2one_candidate_radius = max(
            float(getattr(loss_cfg, "one2one_candidate_radius", 2.5)),
            0.5,
        )
        self.one2one_obj_cost_weight = float(
            getattr(loss_cfg, "one2one_obj_cost_weight", 1.0)
        )
        self.one2one_box_cost_weight = float(
            getattr(loss_cfg, "one2one_box_cost_weight", self.simota_box_weight)
        )
        self.one2one_point_cost_weight = float(
            getattr(loss_cfg, "one2one_point_cost_weight", self.simota_point_weight)
        )
        self.one2one_center_cost_weight = float(
            getattr(loss_cfg, "one2one_center_cost_weight", 0.25)
        )
        self.one2one_warmup_epochs = max(
            int(getattr(loss_cfg, "one2one_warmup_epochs", 0)),
            0,
        )
        self.one2one_obj_neg_weight = max(
            float(getattr(loss_cfg, "one2one_obj_neg_weight", self.obj_neg_weight)),
            0.0,
        )

        self.obj_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.reg_loss_fn = nn.SmoothL1Loss(reduction="none")

        # Epoch counter for VFL warmup scheduling (set via set_epoch).
        self.register_buffer("_epoch", torch.tensor(0, dtype=torch.long))

    @property
    def epoch(self) -> int:
        """Return the current training epoch (0-indexed internally)."""
        return int(self._epoch.item())

    def set_epoch(self, value: int) -> None:
        """Update epoch tracker, called by Trainer each epoch."""
        self._epoch.fill_(value)

    # ------------------------------------------------------------------
    #  Loss kernels
    # ------------------------------------------------------------------

    @staticmethod
    def _balanced_objectness_reduce(
        loss: torch.Tensor,
        target_labels: torch.Tensor,
        pos_weight: float,
        neg_weight: float,
        hard_negative_ratio: float = 0.0,
        hard_negative_min: int = 0,
    ) -> torch.Tensor:
        """Reduce objectness loss with separate positive/negative normalizers.

        Dense 3D grids contain many more negatives than positives. A plain mean
        can make the positive objectness gradient nearly disappear, which is
        harmful for a point-recall-first detector.
        """

        pos_mask = target_labels > 0
        neg_mask = ~pos_mask
        zero = loss.sum() * 0.0

        if pos_mask.any():
            pos_loss = loss[pos_mask].mean()
        else:
            pos_loss = zero

        if neg_mask.any():
            neg_losses = loss[neg_mask]
            if hard_negative_ratio > 0.0 or hard_negative_min > 0:
                num_pos = int(pos_mask.sum().item())
                num_hard = max(
                    int(hard_negative_min),
                    int(math.ceil(num_pos * float(hard_negative_ratio))),
                )
                num_hard = min(max(num_hard, 1), int(neg_losses.numel()))
                neg_losses = torch.topk(neg_losses, k=num_hard, largest=True).values
            neg_loss = neg_losses.mean()
        else:
            neg_loss = zero

        return float(pos_weight) * pos_loss + float(neg_weight) * neg_loss

    @staticmethod
    def _varifocal_loss(
        pred_logits: torch.Tensor,
        target_scores: torch.Tensor,
        target_labels: torch.Tensor,
        alpha: float,
        gamma: float,
        pos_weight: float,
        neg_weight: float,
        hard_negative_ratio: float = 0.0,
        hard_negative_min: int = 0,
    ) -> torch.Tensor:
        """Compute Varifocal Loss with IoU-quality targets for positives.

        Args:
            pred_logits: Predicted objectness logits.
            target_scores: IoU-quality targets for positives.
            target_labels: Binary labels indicating positive samples.
            alpha: Varifocal alpha parameter.
            gamma: Varifocal gamma parameter.

        Returns:
            torch.Tensor: Scalar loss value.
        """

        pred_prob = torch.sigmoid(pred_logits)
        weight = alpha * pred_prob.pow(gamma) * (1.0 - target_labels) + target_scores * target_labels
        loss = F.binary_cross_entropy_with_logits(pred_logits, target_scores, reduction="none")
        return TemporalDetectionCriterion._balanced_objectness_reduce(
            loss * weight,
            target_labels,
            pos_weight=pos_weight,
            neg_weight=neg_weight,
            hard_negative_ratio=hard_negative_ratio,
            hard_negative_min=hard_negative_min,
        )

    @staticmethod
    def _quality_focal_loss(
        pred_logits: torch.Tensor,
        target_labels: torch.Tensor,
        target_quality: torch.Tensor,
        beta: float,
        pos_weight: float,
        neg_weight: float,
        hard_negative_ratio: float = 0.0,
        hard_negative_min: int = 0,
    ) -> torch.Tensor:
        """Compute Quality Focal Loss (GFLv1).

        QFL directly predicts the IoU quality score for positive locations
        and uses a focal modulation based on the L1 distance between the
        prediction and the continuous target.  This avoids the cold-start
        issue of VFL because the positive target is always ≥ quality_min
        (clamped before calling) and negatives remain at 0.

        Args:
            pred_logits: Predicted objectness logits  (any shape).
            target_labels: Binary labels (1=positive, 0=negative).
            target_quality: Continuous quality score (IoU) for positives;
                ignored for negatives (internally multiplied by label).
            beta: Focal exponent controlling the down-weighting of
                well-predicted samples.

        Returns:
            torch.Tensor: Scalar loss value.
        """

        pred_prob = pred_logits.sigmoid()
        # Merged target: quality for positives, 0 for negatives.
        combined_target = target_labels * target_quality
        # Focal weight: |pred - target|^beta.
        delta = (pred_prob - combined_target).abs()
        focal_weight = delta.pow(beta)
        loss = F.binary_cross_entropy_with_logits(
            pred_logits, combined_target, reduction="none"
        )
        return TemporalDetectionCriterion._balanced_objectness_reduce(
            loss * focal_weight,
            target_labels,
            pos_weight=pos_weight,
            neg_weight=neg_weight,
            hard_negative_ratio=hard_negative_ratio,
            hard_negative_min=hard_negative_min,
        )

    # ------------------------------------------------------------------
    #  Label assignment
    # ------------------------------------------------------------------

    @staticmethod
    def _select_level(gt_box: torch.Tensor) -> int:
        """Heuristically choose feature level based on normalized target size.

        Used as a fallback when SimOTA produces no candidates for a GT.

        Args:
            gt_box: Ground-truth box in normalized `(cx, cy, cz, w, h, d)` format.

        Returns:
            int: Feature level index.
        """

        target_scale = float(torch.max(gt_box[3:6]).item())
        if target_scale < 0.10:
            return 0
        if target_scale < 0.22:
            return 1
        return 2

    @staticmethod
    def _point_cost_quality(
        pred_keypoints: torch.Tensor,
        gt_keypoints: torch.Tensor,
        gt_valid: torch.Tensor,
        quality_sigma: float,
        vis_weight: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute point-first assignment cost and objectness quality."""

        if pred_keypoints.ndim == 2:
            pred_keypoints = pred_keypoints.view(pred_keypoints.shape[0], -1, 4)
        elif pred_keypoints.ndim != 3:
            raise ValueError("pred_keypoints must have shape (N, K*4) or (N, K, 4).")

        if gt_keypoints.ndim == 1:
            gt_keypoints = gt_keypoints.view(-1, 4)
        elif gt_keypoints.ndim != 2:
            raise ValueError("gt_keypoints must have shape (K*4) or (K, 4).")

        gt_valid = gt_valid.bool()
        if gt_valid.shape[0] != pred_keypoints.shape[1]:
            raise ValueError("gt_valid length must match the number of keypoint slots.")

        if gt_valid.any():
            pred_xy = pred_keypoints[:, gt_valid, :2]
            gt_xy = gt_keypoints[gt_valid, :2].unsqueeze(0)
            spatial_dist = torch.linalg.vector_norm(pred_xy - gt_xy, dim=-1).mean(dim=1)
        else:
            spatial_dist = pred_keypoints.new_zeros((pred_keypoints.shape[0],))

        pred_vis = pred_keypoints[:, :, 3].clamp(min=1e-6, max=1.0 - 1e-6)
        target_vis = gt_valid.float().unsqueeze(0).expand_as(pred_vis)
        vis_cost = F.binary_cross_entropy(pred_vis, target_vis, reduction="none").mean(dim=1)
        point_cost = spatial_dist + float(vis_weight) * vis_cost
        point_quality = torch.exp(-spatial_dist / float(quality_sigma)).clamp(min=0.0, max=1.0)
        return point_cost, point_quality

    @staticmethod
    def _linear_sum_assignment(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solve a rectangular minimum-cost assignment without extra dependencies."""

        if cost.ndim != 2:
            raise ValueError("cost must be a 2D tensor.")

        num_rows, num_cols = int(cost.shape[0]), int(cost.shape[1])
        if num_rows == 0 or num_cols == 0:
            empty = torch.empty((0,), dtype=torch.long, device=cost.device)
            return empty, empty
        if num_rows > num_cols:
            col_ind, row_ind = TemporalDetectionCriterion._linear_sum_assignment(
                cost.transpose(0, 1)
            )
            order = torch.argsort(row_ind)
            return row_ind[order], col_ind[order]

        cpu_cost = cost.detach().to(device="cpu", dtype=torch.float64)
        u = [0.0] * (num_rows + 1)
        v = [0.0] * (num_cols + 1)
        p = [0] * (num_cols + 1)
        way = [0] * (num_cols + 1)

        for row in range(1, num_rows + 1):
            p[0] = row
            min_values = [float("inf")] * (num_cols + 1)
            used = [False] * (num_cols + 1)
            col0 = 0

            while True:
                used[col0] = True
                row0 = p[col0]
                delta = float("inf")
                col1 = 0
                for col in range(1, num_cols + 1):
                    if used[col]:
                        continue
                    current = float(cpu_cost[row0 - 1, col - 1].item()) - u[row0] - v[col]
                    if current < min_values[col]:
                        min_values[col] = current
                        way[col] = col0
                    if min_values[col] < delta:
                        delta = min_values[col]
                        col1 = col

                for col in range(num_cols + 1):
                    if used[col]:
                        u[p[col]] += delta
                        v[col] -= delta
                    else:
                        min_values[col] -= delta
                col0 = col1
                if p[col0] == 0:
                    break

            while True:
                col1 = way[col0]
                p[col0] = p[col1]
                col0 = col1
                if col0 == 0:
                    break

        row_to_col = [-1] * num_rows
        for col in range(1, num_cols + 1):
            if p[col] != 0:
                row_to_col[p[col] - 1] = col - 1

        row_ind = torch.arange(num_rows, dtype=torch.long, device=cost.device)
        col_ind = torch.tensor(row_to_col, dtype=torch.long, device=cost.device)
        return row_ind, col_ind

    @torch.no_grad()
    def _one2one_assign_sample(
        self,
        decoded_levels: List[Dict[str, torch.Tensor]],
        gt_boxes: torch.Tensor,
        gt_keypoints: torch.Tensor,
        gt_keypoint_valid: torch.Tensor,
        batch_idx: int,
    ) -> Tuple[List[int], List[int], List[int], Dict[str, float]]:
        """Assign exactly one unique prediction to each GT with Hungarian matching."""

        device = gt_boxes.device
        dtype = gt_boxes.dtype
        num_gt = int(gt_boxes.shape[0])
        if num_gt == 0:
            return [], [], [], {
                "raw_assignments": 0.0,
                "unique_assignments": 0.0,
                "assignment_conflicts": 0.0,
                "max_positives_per_gt": 0.0,
            }

        candidate_logits: List[torch.Tensor] = []
        candidate_boxes: List[torch.Tensor] = []
        candidate_keypoints: List[torch.Tensor] = []
        candidate_levels: List[torch.Tensor] = []
        candidate_flats: List[torch.Tensor] = []
        candidate_centers: List[torch.Tensor] = []
        candidate_radius_scales: List[torch.Tensor] = []
        allowed_masks: List[torch.Tensor] = []

        for level, decoded in enumerate(decoded_levels):
            shape = decoded["shape"]
            t_size, h_size, w_size = (int(shape[0]), int(shape[1]), int(shape[2]))
            t_idx = torch.arange(t_size, device=device, dtype=dtype)
            y_idx = torch.arange(h_size, device=device, dtype=dtype)
            x_idx = torch.arange(w_size, device=device, dtype=dtype)
            tt, yy, xx = torch.meshgrid(t_idx, y_idx, x_idx, indexing="ij")
            centers = torch.stack(
                (
                    (xx + 0.5) / float(w_size),
                    (yy + 0.5) / float(h_size),
                    (tt + 0.5) / float(t_size),
                ),
                dim=-1,
            ).reshape(-1, 3)
            radius_scale = centers.new_tensor(
                (
                    self.one2one_candidate_radius / float(w_size),
                    self.one2one_candidate_radius / float(h_size),
                    self.one2one_candidate_radius / float(t_size),
                )
            )
            allowed = (
                torch.abs(gt_boxes[:, None, :3] - centers[None, :, :])
                <= radius_scale.view(1, 1, 3)
            ).all(dim=-1)
            union_mask = allowed.any(dim=0)
            if not union_mask.any():
                continue

            flat_indices = union_mask.nonzero(as_tuple=False).squeeze(-1)
            num_candidates = int(flat_indices.numel())
            candidate_logits.append(decoded["obj_logits"][batch_idx, flat_indices])
            candidate_boxes.append(decoded["boxes"][batch_idx, flat_indices])
            candidate_keypoints.append(decoded["keypoints"][batch_idx, flat_indices])
            candidate_levels.append(
                torch.full((num_candidates,), level, dtype=torch.long, device=device)
            )
            candidate_flats.append(flat_indices)
            candidate_centers.append(centers[flat_indices])
            candidate_radius_scales.append(
                radius_scale.view(1, 3).expand(num_candidates, 3)
            )
            allowed_masks.append(allowed[:, flat_indices])

        if not candidate_logits:
            raise RuntimeError("One-to-one assignment could not construct any candidates.")

        logits = torch.cat(candidate_logits, dim=0)
        boxes = torch.cat(candidate_boxes, dim=0)
        keypoints = torch.cat(candidate_keypoints, dim=0)
        levels = torch.cat(candidate_levels, dim=0)
        flats = torch.cat(candidate_flats, dim=0)
        centers = torch.cat(candidate_centers, dim=0)
        radius_scales = torch.cat(candidate_radius_scales, dim=0).clamp(min=1e-6)
        allowed = torch.cat(allowed_masks, dim=1)
        num_candidates = int(logits.shape[0])

        cost_rows: List[torch.Tensor] = []
        for gt_idx in range(num_gt):
            gt_box = gt_boxes[gt_idx]
            expanded_gt_box = gt_box.unsqueeze(0).expand(num_candidates, -1)
            iou = box_iou_3d_aligned(boxes, expanded_gt_box)
            point_cost, _ = self._point_cost_quality(
                keypoints,
                gt_keypoints[gt_idx],
                gt_keypoint_valid[gt_idx],
                quality_sigma=self.point_quality_sigma,
                vis_weight=self.simota_vis_weight,
            )
            center_cost = (
                torch.abs(centers - gt_box[:3].unsqueeze(0)) / radius_scales
            ).mean(dim=1)
            objectness_cost = F.softplus(-logits)
            row_cost = (
                self.one2one_obj_cost_weight * objectness_cost
                + self.one2one_point_cost_weight * point_cost
                + self.one2one_box_cost_weight * (1.0 - iou)
                + self.one2one_center_cost_weight * center_cost
            )
            row_cost = torch.where(
                allowed[gt_idx],
                row_cost,
                row_cost.new_full((), 1e6),
            )
            cost_rows.append(row_cost)

        cost_matrix = torch.stack(cost_rows, dim=0)
        cost_matrix = torch.nan_to_num(cost_matrix, nan=1e6, posinf=1e6, neginf=-1e6)
        gt_indices, candidate_indices = self._linear_sum_assignment(cost_matrix)

        if int(gt_indices.numel()) != num_gt:
            raise RuntimeError(
                f"One-to-one assignment matched {gt_indices.numel()} of {num_gt} GT tracks."
            )
        if bool((cost_matrix[gt_indices, candidate_indices] >= 1e6).any().item()):
            raise RuntimeError("One-to-one assignment selected a candidate outside the center prior.")

        order = torch.argsort(levels[candidate_indices] * (flats.max() + 1) + flats[candidate_indices])
        gt_indices = gt_indices[order]
        candidate_indices = candidate_indices[order]

        stats = {
            "raw_assignments": float(num_gt),
            "unique_assignments": float(candidate_indices.unique().numel()),
            "assignment_conflicts": 0.0,
            "max_positives_per_gt": 1.0,
        }
        return (
            levels[candidate_indices].tolist(),
            flats[candidate_indices].tolist(),
            gt_indices.tolist(),
            stats,
        )

    def _simota_assign_sample(
        self,
        decoded_levels: List[Dict[str, torch.Tensor]],
        gt_boxes: torch.Tensor,
        gt_keypoints: torch.Tensor,
        gt_keypoint_valid: torch.Tensor,
        batch_idx: int,
    ) -> Tuple[List[int], List[int], List[int], Dict[str, float]]:
        """SimOTA dynamic label assignment for one batch sample.

        For each GT track:
        1. Collect candidate predictions within a centre prior (2.5×stride)
           across all feature levels.
        2. Compute pairwise cost = -obj_score + λ·(1 – IoU).
        3. Determine dynamic k from the sum of top-k point-quality scores.
        4. Select the top-k lowest-cost candidates as positives.

        Falls back to the original single-cell heuristic when no candidate
        satisfies the centre prior.

        Args:
            decoded_levels: Per-level decoded predictions (each contains
                'scores', 'boxes', 'shape').
            gt_boxes: Ground-truth boxes for this sample  (num_gt, 6).
            batch_idx: Index of the sample within the batch.

        Returns:
            Tuple containing parallel assignment lists
            `(levels, flat_indices, gt_indices)` and assignment diagnostics.
        """

        selected_assignments: List[Tuple[float, int, int, int]] = []

        device = gt_boxes.device
        dtype = gt_boxes.dtype
        num_levels = len(decoded_levels)

        for gt_idx in range(gt_boxes.shape[0]):
            gt_box = gt_boxes[gt_idx]

            cand_scores_list: List[torch.Tensor] = []
            cand_boxes_list: List[torch.Tensor] = []
            cand_keypoints_list: List[torch.Tensor] = []
            cand_levels_list: List[torch.Tensor] = []
            cand_flats_list: List[torch.Tensor] = []

            for lvl, decoded in enumerate(decoded_levels):
                shape = decoded["shape"]  # [T, H, W]
                T, H, W = int(shape[0]), int(shape[1]), int(shape[2])

                # Build grid centres in normalised coords.
                t_idx = torch.arange(T, device=device, dtype=dtype)
                y_idx = torch.arange(H, device=device, dtype=dtype)
                x_idx = torch.arange(W, device=device, dtype=dtype)
                tt, yy, xx = torch.meshgrid(t_idx, y_idx, x_idx, indexing="ij")

                grid_cx = (xx + 0.5) / float(W)
                grid_cy = (yy + 0.5) / float(H)
                grid_cz = (tt + 0.5) / float(T)

                # Centre prior: 2.5 grid cells.
                radius = 2.5
                mask = (
                    (torch.abs(grid_cx - gt_box[0]) <= radius / float(W))
                    & (torch.abs(grid_cy - gt_box[1]) <= radius / float(H))
                    & (torch.abs(grid_cz - gt_box[2]) <= radius / float(T))
                )

                if not mask.any():
                    continue

                flat_mask = mask.reshape(-1)
                flat_indices = flat_mask.nonzero(as_tuple=False).squeeze(-1)

                cand_scores_list.append(decoded["scores"][batch_idx, flat_indices])
                cand_boxes_list.append(decoded["boxes"][batch_idx, flat_indices])
                cand_keypoints_list.append(decoded["keypoints"][batch_idx, flat_indices])
                cand_levels_list.append(
                    torch.full_like(flat_indices, lvl, dtype=torch.long)
                )
                cand_flats_list.append(flat_indices)

            # Fallback: single-cell heuristic.
            if not cand_scores_list:
                level = self._select_level(gt_box)
                level = min(level, num_levels - 1)
                shape = decoded_levels[level]["shape"]
                T, H, W = int(shape[0]), int(shape[1]), int(shape[2])
                gx = int(torch.clamp(gt_box[0] * W, 0, W - 1).item())
                gy = int(torch.clamp(gt_box[1] * H, 0, H - 1).item())
                gz = int(torch.clamp(gt_box[2] * T, 0, T - 1).item())
                flat_idx = (gz * H + gy) * W + gx
                pred_box = decoded_levels[level]["boxes"][batch_idx, flat_idx]
                pred_kpt = decoded_levels[level]["keypoints"][batch_idx, flat_idx]
                fallback_iou = box_iou_3d_aligned(
                    pred_box.unsqueeze(0),
                    gt_box.unsqueeze(0),
                ).squeeze(0)
                fallback_point_cost, _ = self._point_cost_quality(
                    pred_kpt.unsqueeze(0),
                    gt_keypoints[gt_idx],
                    gt_keypoint_valid[gt_idx],
                    quality_sigma=self.point_quality_sigma,
                    vis_weight=self.simota_vis_weight,
                )
                fallback_cost = (
                    -decoded_levels[level]["scores"][batch_idx, flat_idx]
                    + self.simota_point_weight * fallback_point_cost.squeeze(0)
                    + self.simota_box_weight * (1.0 - fallback_iou)
                )
                selected_assignments.append(
                    (float(fallback_cost.detach().item()), level, flat_idx, gt_idx)
                )
                continue

            # Concatenate candidates across levels.
            cat_scores = torch.cat(cand_scores_list, dim=0)      # (C,)
            cat_boxes = torch.cat(cand_boxes_list, dim=0)        # (C, 6)
            cat_keypoints = torch.cat(cand_keypoints_list, dim=0)  # (C, K*4)
            cat_levels = torch.cat(cand_levels_list, dim=0)
            cat_flats = torch.cat(cand_flats_list, dim=0)
            num_candidates = cat_scores.shape[0]

            # Pairwise IoU between candidate boxes and the GT box.
            gt_expanded = gt_box.unsqueeze(0).expand(num_candidates, -1)
            ious = box_iou_3d_aligned(cat_boxes, gt_expanded)    # (C,)
            point_cost, point_quality = self._point_cost_quality(
                cat_keypoints,
                gt_keypoints[gt_idx],
                gt_keypoint_valid[gt_idx],
                quality_sigma=self.point_quality_sigma,
                vis_weight=self.simota_vis_weight,
            )

            # Cost = –obj_score + λ·(1 – IoU).
            cost = (
                -cat_scores
                + self.simota_point_weight * point_cost
                + self.simota_box_weight * (1.0 - ious)
            )

            # Dynamic k: sum of top-K IoUs, at least 1.
            effective_topk = min(self.simota_topk, num_candidates)
            topk_quality, _ = point_quality.topk(effective_topk)
            dynamic_k = max(1, int(topk_quality.sum().item()))
            dynamic_k = min(dynamic_k, num_candidates)
            if self.simota_max_positives_per_gt > 0:
                dynamic_k = min(dynamic_k, self.simota_max_positives_per_gt)

            # Select top-dynamic_k by lowest cost.
            _, top_indices = cost.topk(dynamic_k, largest=False)

            for idx in top_indices.tolist():
                selected_assignments.append(
                    (
                        float(cost[idx].detach().item()),
                        int(cat_levels[idx].item()),
                        int(cat_flats[idx].item()),
                        gt_idx,
                    )
                )

        # A prediction location can be proposed for multiple nearby GT tracks.
        # Keep only its lowest-cost assignment to avoid contradictory regression
        # targets and inflated positive counts.
        best_by_location: Dict[Tuple[int, int], Tuple[float, int, int, int]] = {}
        for assignment in selected_assignments:
            cost_value, level, flat_idx, _ = assignment
            key = (level, flat_idx)
            previous = best_by_location.get(key)
            if previous is None or cost_value < previous[0]:
                best_by_location[key] = assignment

        resolved = list(best_by_location.values())
        resolved.sort(key=lambda item: (item[1], item[2]))
        gt_counts = [0 for _ in range(int(gt_boxes.shape[0]))]
        for _, _, _, gt_idx in resolved:
            gt_counts[gt_idx] += 1

        stats = {
            "raw_assignments": float(len(selected_assignments)),
            "unique_assignments": float(len(resolved)),
            "assignment_conflicts": float(len(selected_assignments) - len(resolved)),
            "max_positives_per_gt": float(max(gt_counts, default=0)),
        }
        return (
            [item[1] for item in resolved],
            [item[2] for item in resolved],
            [item[3] for item in resolved],
            stats,
        )

    # ------------------------------------------------------------------
    #  Keypoint helpers (delegated to utils.keypoint_ops)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    #  Core loss computation
    # ------------------------------------------------------------------

    def _compute_single_branch_loss(
        self,
        predictions: Sequence[torch.Tensor],
        targets: List[Dict[str, object]],
        assignment: str = "simota",
        obj_neg_weight: float | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute one branch loss using dense or strict one-to-one assignment.

        Args:
            predictions: Raw prediction tensors per feature scale.
            targets: List of ground-truth target dictionaries.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Total loss and loss breakdown.
        """

        if len(predictions) == 0:
            raise ValueError("Prediction list is empty.")
        branch_obj_neg_weight = (
            self.obj_neg_weight
            if obj_neg_weight is None
            else max(float(obj_neg_weight), 0.0)
        )

        # Decode the first level to infer keypoint layout.
        first_level = decode_feature_map(predictions[0])
        keypoint_shape = tuple(int(v) for v in first_level["kpt_shape"].tolist())
        num_keypoints, keypoint_dims = keypoint_shape
        if keypoint_dims != 4:
            raise ValueError(f"Expected keypoint dims 4, got {keypoint_dims}.")

        decoded_levels = [first_level]
        for pred in predictions[1:]:
            decoded_levels.append(decode_feature_map(pred))

        # Initialise label tensors per level.
        obj_labels: List[torch.Tensor] = [
            torch.zeros_like(item["obj_logits"]) for item in decoded_levels
        ]
        obj_targets: List[torch.Tensor] = [
            torch.zeros_like(item["obj_logits"]) for item in decoded_levels
        ]

        pos_pred_boxes: List[torch.Tensor] = []
        pos_gt_boxes: List[torch.Tensor] = []
        pos_pred_kpts: List[torch.Tensor] = []
        pos_gt_kpts: List[torch.Tensor] = []
        pos_gt_kpt_valid: List[torch.Tensor] = []
        raw_assignments = 0.0
        assignment_conflicts = 0.0
        max_positives_per_gt = 0.0

        device = predictions[0].device

        for batch_idx, target in enumerate(targets):
            gt_boxes = target.get("bboxes")
            if not isinstance(gt_boxes, torch.Tensor) or gt_boxes.numel() == 0:
                continue

            gt_boxes = gt_boxes.float().to(device)
            gt_keypoints, gt_keypoint_valid = derive_group_keypoints(
                target,
                num_keypoints=num_keypoints,
                device=device,
            )
            gt_keypoints = gt_keypoints.float()
            if gt_keypoints.shape[0] != gt_boxes.shape[0]:
                gt_keypoints = default_keypoints_from_boxes(gt_boxes, num_keypoints)
                gt_keypoint_valid = torch.ones(
                    (gt_boxes.shape[0], num_keypoints),
                    dtype=torch.bool,
                    device=device,
                )

            if assignment == "simota":
                assign_fn = self._simota_assign_sample
            elif assignment == "one2one":
                assign_fn = self._one2one_assign_sample
            else:
                raise ValueError(f"Unsupported assignment mode: {assignment}")
            pos_levels, pos_flats, pos_gt_idxs, assignment_stats = assign_fn(
                decoded_levels, gt_boxes, gt_keypoints, gt_keypoint_valid, batch_idx
            )
            raw_assignments += assignment_stats["raw_assignments"]
            assignment_conflicts += assignment_stats["assignment_conflicts"]
            max_positives_per_gt = max(
                max_positives_per_gt,
                assignment_stats["max_positives_per_gt"],
            )

            for lvl, flat, gt_idx in zip(pos_levels, pos_flats, pos_gt_idxs):
                gt_box = gt_boxes[gt_idx]
                gt_kpt = gt_keypoints[gt_idx]
                gt_kpt_valid = gt_keypoint_valid[gt_idx]

                pred_box = decoded_levels[lvl]["boxes"][batch_idx, flat]
                pred_kpt = decoded_levels[lvl]["keypoints"][batch_idx, flat]
                _, point_quality = self._point_cost_quality(
                    pred_kpt.unsqueeze(0),
                    gt_kpt,
                    gt_kpt_valid,
                    quality_sigma=self.point_quality_sigma,
                    vis_weight=self.simota_vis_weight,
                )
                quality = point_quality.squeeze(0).detach().clamp(min=0.0, max=1.0)

                obj_labels[lvl][batch_idx, flat] = 1.0

                if self.obj_loss_type == "varifocal":
                    # Clamp quality minimum to mitigate cold-start.
                    quality_clamped = quality.clamp(min=self.vfl_quality_min)
                    obj_targets[lvl][batch_idx, flat] = torch.maximum(
                        obj_targets[lvl][batch_idx, flat], quality_clamped,
                    )
                elif self.obj_loss_type == "qfl":
                    # QFL: target = quality, minimum clamped for cold-start.
                    quality_clamped = quality.clamp(min=self.vfl_quality_min)
                    obj_targets[lvl][batch_idx, flat] = torch.maximum(
                        obj_targets[lvl][batch_idx, flat], quality_clamped,
                    )
                else:  # bce
                    obj_targets[lvl][batch_idx, flat] = 1.0

                pos_pred_boxes.append(pred_box)
                pos_gt_boxes.append(gt_box)
                pos_pred_kpts.append(pred_kpt)
                pos_gt_kpts.append(gt_kpt)
                pos_gt_kpt_valid.append(gt_kpt_valid)

        # ---- Objectness loss ----
        obj_loss = torch.tensor(0.0, device=device)
        for level, decoded in enumerate(decoded_levels):
            if self.obj_loss_type == "varifocal":
                # VFL warmup: use BCE for the first N epochs while quality
                # targets are unreliable (controlled by vfl_warmup_epochs).
                if self.epoch <= self.vfl_warmup_epochs:
                    warmup_loss = F.binary_cross_entropy_with_logits(
                        decoded["obj_logits"],
                        obj_labels[level],
                        reduction="none",
                    )
                    level_obj_loss = self._balanced_objectness_reduce(
                        warmup_loss,
                        obj_labels[level],
                        pos_weight=self.obj_pos_weight,
                        neg_weight=branch_obj_neg_weight,
                        hard_negative_ratio=self.obj_hard_negative_ratio,
                        hard_negative_min=self.obj_hard_negative_min,
                    )
                else:
                    level_obj_loss = self._varifocal_loss(
                        pred_logits=decoded["obj_logits"],
                        target_scores=obj_targets[level],
                        target_labels=obj_labels[level],
                        alpha=self.vfl_alpha,
                        gamma=self.vfl_gamma,
                        pos_weight=self.obj_pos_weight,
                        neg_weight=branch_obj_neg_weight,
                        hard_negative_ratio=self.obj_hard_negative_ratio,
                        hard_negative_min=self.obj_hard_negative_min,
                    )
            elif self.obj_loss_type == "qfl":
                level_obj_loss = self._quality_focal_loss(
                    pred_logits=decoded["obj_logits"],
                    target_labels=obj_labels[level],
                    target_quality=obj_targets[level],
                    beta=self.qfl_beta,
                    pos_weight=self.obj_pos_weight,
                    neg_weight=branch_obj_neg_weight,
                    hard_negative_ratio=self.obj_hard_negative_ratio,
                    hard_negative_min=self.obj_hard_negative_min,
                )
            else:
                bce_loss = F.binary_cross_entropy_with_logits(
                    decoded["obj_logits"],
                    obj_labels[level],
                    reduction="none",
                )
                level_obj_loss = self._balanced_objectness_reduce(
                    bce_loss,
                    obj_labels[level],
                    pos_weight=self.obj_pos_weight,
                    neg_weight=branch_obj_neg_weight,
                    hard_negative_ratio=self.obj_hard_negative_ratio,
                    hard_negative_min=self.obj_hard_negative_min,
                )
            obj_loss = obj_loss + level_obj_loss
        obj_loss = obj_loss / max(len(decoded_levels), 1)

        # ---- Box / Keypoint / IoU loss ----
        if len(pos_pred_boxes) > 0:
            pred_boxes = torch.stack(pos_pred_boxes, dim=0)
            gt_boxes = torch.stack(pos_gt_boxes, dim=0)
            pred_kpts = torch.stack(pos_pred_kpts, dim=0)
            gt_kpts = torch.stack(pos_gt_kpts, dim=0)
            gt_kpt_valid = torch.stack(pos_gt_kpt_valid, dim=0)

            # IoU-modulated box regression:
            # low-IoU predictions receive higher regression weight so the
            # model focuses on poorly-localised boxes.
            if self.iou_loss_type == "ciou":
                iou_metric = box_ciou_3d_aligned(pred_boxes, gt_boxes)
                iou_raw = box_iou_3d_aligned(pred_boxes, gt_boxes)
            else:
                iou_metric = box_iou_3d_aligned(pred_boxes, gt_boxes)
                iou_raw = iou_metric

            iou_loss = 1.0 - iou_metric.mean()

            # IoU-modulated SmoothL1 weight:  (1 – IoU) gives more weight to
            # worse-localised positives.  Clamp so the minimum weight is 0.3.
            iou_weight = (1.0 - iou_raw.detach()).clamp(min=0.3)
            box_loss_per_elem = self.reg_loss_fn(pred_boxes, gt_boxes)  # (N, 6)
            box_loss = (box_loss_per_elem * iou_weight.unsqueeze(-1)).mean()

            # Keypoint loss.
            pred_kpts = pred_kpts.view(-1, num_keypoints, 4)
            gt_kpts = gt_kpts.view(-1, num_keypoints, 4)

            xy_mask = gt_kpt_valid.unsqueeze(-1).expand(-1, -1, 2)
            if xy_mask.any():
                xy_error = torch.abs(pred_kpts[:, :, :2] - gt_kpts[:, :, :2])
                kpt_xy_loss = xy_error[xy_mask].mean()
                t_error = torch.abs(pred_kpts[:, :, 2] - gt_kpts[:, :, 2])
                kpt_t_loss = t_error[gt_kpt_valid].mean()
                kpt_coord_loss = kpt_xy_loss + self.kpt_time_weight * kpt_t_loss
            else:
                kpt_xy_loss = torch.tensor(0.0, device=device)
                kpt_t_loss = torch.tensor(0.0, device=device)
                kpt_coord_loss = torch.tensor(0.0, device=device)

            pred_vis = pred_kpts[:, :, 3].clamp(min=1e-6, max=1.0 - 1e-6)
            gt_vis = gt_kpt_valid.float()
            kpt_vis_loss = F.binary_cross_entropy(pred_vis, gt_vis, reduction="mean")
            kpt_loss = kpt_coord_loss + self.loss_cfg.kpt_vis_weight * kpt_vis_loss

            num_pos = torch.tensor(float(pred_boxes.shape[0]), device=device)
        else:
            zero = torch.tensor(0.0, device=device)
            box_loss = zero
            kpt_loss = zero
            kpt_coord_loss = zero
            kpt_xy_loss = zero
            kpt_t_loss = zero
            kpt_vis_loss = zero
            iou_loss = zero
            num_pos = zero

        total_loss = (
            self.loss_cfg.obj_weight * obj_loss
            + self.loss_cfg.box_weight * box_loss
            + self.loss_cfg.iou_weight * iou_loss
            + self.loss_cfg.kpt_weight * kpt_loss
        )

        loss_dict = {
            "loss_total": total_loss.detach(),
            "loss_obj": obj_loss.detach(),
            "loss_box": box_loss.detach(),
            "loss_iou": iou_loss.detach(),
            "loss_kpt": kpt_loss.detach(),
            "loss_kpt_coord": kpt_coord_loss.detach(),
            "loss_kpt_xy": kpt_xy_loss.detach(),
            "loss_kpt_t": kpt_t_loss.detach(),
            "loss_kpt_vis": kpt_vis_loss.detach(),
            "num_pos": num_pos.detach(),
            "num_assignments_raw": torch.tensor(raw_assignments, device=device),
            "num_assignment_conflicts": torch.tensor(assignment_conflicts, device=device),
            "max_positives_per_gt": torch.tensor(max_positives_per_gt, device=device),
        }
        return total_loss, loss_dict

    def forward(
        self,
        predictions: Sequence[torch.Tensor] | Dict[str, Sequence[torch.Tensor]],
        targets: List[Dict[str, object]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total loss for single-branch or dual-branch end-to-end training.

        Args:
            predictions: Prediction list or dict with one2many/one2one branches.
            targets: List of ground-truth target dictionaries.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Total loss and loss breakdown.
        """

        if isinstance(predictions, dict):
            one2many_predictions = predictions.get("one2many")
            one2one_predictions = predictions.get("one2one")
            if not isinstance(one2many_predictions, Sequence) or not isinstance(
                one2one_predictions, Sequence
            ):
                raise TypeError(
                    "When predictions is a dict, both 'one2many' and 'one2one' "
                    "must be prediction lists."
                )

            main_loss, main_items = self._compute_single_branch_loss(
                one2many_predictions, targets, assignment="simota"
            )
            one2one_assignment = (
                "simota"
                if self.epoch <= self.one2one_warmup_epochs
                else "one2one"
            )
            aux_loss, aux_items = self._compute_single_branch_loss(
                one2one_predictions,
                targets,
                assignment=one2one_assignment,
                obj_neg_weight=self.one2one_obj_neg_weight,
            )
            total_loss = main_loss + self.one2one_weight * aux_loss

            loss_dict = {
                "loss_total": total_loss.detach(),
                "loss_main": main_loss.detach(),
                "loss_aux": aux_loss.detach(),
                "loss_aux_weighted": (self.one2one_weight * aux_loss).detach(),
                "loss_obj": main_items["loss_obj"],
                "loss_box": main_items["loss_box"],
                "loss_iou": main_items["loss_iou"],
                "loss_kpt": main_items["loss_kpt"],
                "loss_kpt_coord": main_items["loss_kpt_coord"],
                "loss_kpt_xy": main_items["loss_kpt_xy"],
                "loss_kpt_t": main_items["loss_kpt_t"],
                "loss_kpt_vis": main_items["loss_kpt_vis"],
                "num_pos": main_items["num_pos"],
                "num_assignments_raw": main_items["num_assignments_raw"],
                "num_assignment_conflicts": main_items["num_assignment_conflicts"],
                "max_positives_per_gt": main_items["max_positives_per_gt"],
                "loss_obj_aux": aux_items["loss_obj"],
                "loss_box_aux": aux_items["loss_box"],
                "loss_iou_aux": aux_items["loss_iou"],
                "loss_kpt_aux": aux_items["loss_kpt"],
                "loss_kpt_coord_aux": aux_items["loss_kpt_coord"],
                "loss_kpt_xy_aux": aux_items["loss_kpt_xy"],
                "loss_kpt_t_aux": aux_items["loss_kpt_t"],
                "loss_kpt_vis_aux": aux_items["loss_kpt_vis"],
                "num_pos_aux": aux_items["num_pos"],
                "num_assignments_raw_aux": aux_items["num_assignments_raw"],
                "num_assignment_conflicts_aux": aux_items["num_assignment_conflicts"],
                "max_positives_per_gt_aux": aux_items["max_positives_per_gt"],
            }
            return total_loss, loss_dict

        if not isinstance(predictions, Sequence):
            raise TypeError(
                "Predictions must be a list of tensors or a dict with "
                "one2many/one2one branches."
            )

        return self._compute_single_branch_loss(predictions, targets)
