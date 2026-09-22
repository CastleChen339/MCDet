"""Central configuration definitions for AstroDim 3D detection project."""

from dataclasses import dataclass, field
from typing import Literal, Tuple


@dataclass
class DatasetConfig:
    """Configuration for AstroDim dataset construction and augmentation.

    Attributes:
        train_root: Path to training dataset root.
        val_root: Path to validation dataset root.
        seq_len: Number of frames per sample sequence.
        image_channels: Number of channels to decode from images.
        crop_size: Spatial crop size `(height, width)`.
        min_group_points: Minimum visible points required to keep a target track.
        batch_size: Batch size for dataloaders.
        num_workers: DataLoader worker count.
        pin_memory: Whether to pin host memory in DataLoader.
        prefetch_factor: Prefetch factor for multi-worker loading.
        persistent_workers: Keep worker processes alive between epochs.
        use_torchvision_io: Prefer torchvision IO backend when available.
    """

    train_root: str = "/home/cjc/datasets/AstroDim_mini/train"
    val_root: str = "/home/cjc/datasets/AstroDim_mini/val"
    seq_len: int = 5
    image_channels: int = 1
    crop_size: Tuple[int, int] = (640, 640)
    min_group_points: int = 2  # The reason why it is not set to 1 here is because it means there is only one point in the trajectory, which is illogical and prone to false alarms
    batch_size: int = 4
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 1
    persistent_workers: bool = True
    use_torchvision_io: bool = True


@dataclass
class ModelConfig:
    """Configuration for MCDet 3D backbone/neck/head.

    Attributes:
        in_channels: Input image channel count.
        base_channels: Base width multiplier for backbone channels.
        num_classes: Number of object classes (objectness uses 1).
        model_size: Compound-scale preset name (`n`, `s`, `m`, `l`, `x`).
        end2end: Enable dual one2many/one2one head training.
        one2one_full_gradient: Allow one2one loss to update shared features.
        one2one_separate_head: Use independent task towers for the one2one head.
    """

    in_channels: int = 1
    base_channels: int = 16
    num_classes: int = 1
    model_size: str = "m"
    end2end: bool = True
    one2one_full_gradient: bool = True
    one2one_separate_head: bool = True


@dataclass
class LossConfig:
    """Configuration for detection loss weighting.

    Attributes:
        obj_weight: Weight for objectness loss.
        box_weight: Auxiliary weight for box regression loss.
        iou_weight: Auxiliary weight for IoU loss.
        kpt_weight: Weight for keypoint regression loss.
        kpt_vis_weight: Weight for keypoint visibility loss.
        kpt_time_weight: Relative weight for temporal keypoint coordinates.
        one2one_weight: Weight for the auxiliary one2one branch loss.
        obj_pos_weight: Positive-location weight for balanced objectness reduction.
        obj_neg_weight: Negative-location weight for balanced objectness reduction.
        obj_hard_negative_ratio: Maximum hard negatives retained per positive.
        obj_hard_negative_min: Minimum hard negatives retained per feature level.
        obj_loss_type: Objectness loss type (`varifocal`, `bce`, or `qfl`).
        iou_loss_type: IoU loss type (`ciou` or `iou`).
        vfl_alpha: Varifocal loss alpha parameter.
        vfl_gamma: Varifocal loss gamma parameter.
        qfl_beta: Quality Focal Loss beta exponent (default 2.0).
        vfl_quality_min: Minimum quality target for VFL/QFL cold-start mitigation.
        simota_topk: SimOTA candidate count upper bound.
        simota_max_positives_per_gt: Maximum assigned locations per GT; 0 disables the cap.
        simota_reg_weight: Legacy SimOTA box regression cost weight.
        simota_box_weight: Box IoU term weight in point-aware SimOTA.
        simota_point_weight: Point distance term weight in point-aware SimOTA.
        simota_vis_weight: Visibility mismatch term weight in point-aware SimOTA.
        point_quality_sigma: Normalized point-distance scale for objectness quality.
        one2one_candidate_radius: Center-prior radius in feature cells for one-to-one matching.
        one2one_obj_cost_weight: Objectness term weight in one-to-one matching.
        one2one_box_cost_weight: Box IoU term weight in one-to-one matching.
        one2one_point_cost_weight: Point term weight in one-to-one matching.
        one2one_center_cost_weight: Grid-center prior term weight in one-to-one matching.
        one2one_warmup_epochs: Initial epochs using dense assignment on the one-to-one head.
        one2one_obj_neg_weight: Negative objectness weight for the one-to-one head.
        vfl_warmup_epochs: Number of initial epochs to use BCE instead of VFL.
    """

    obj_weight: float = 5.0
    box_weight: float = 0.05
    iou_weight: float = 0.10
    kpt_weight: float = 5.0
    kpt_vis_weight: float = 1.0
    kpt_time_weight: float = 0.2
    one2one_weight: float = 0.5
    obj_pos_weight: float = 1.0
    obj_neg_weight: float = 0.5
    obj_hard_negative_ratio: float = 0.0
    obj_hard_negative_min: int = 0
    obj_loss_type: str = "bce"  # options: varifocal, bce, qfl
    iou_loss_type: str = "ciou"  # options: ciou, iou
    vfl_alpha: float = 0.75
    vfl_gamma: float = 2.0
    qfl_beta: float = 2.0
    vfl_quality_min: float = 0.25
    vfl_warmup_epochs: int = 3
    simota_topk: int = 20
    simota_max_positives_per_gt: int = 8
    simota_reg_weight: float = 0.5
    simota_box_weight: float = 0.1
    simota_point_weight: float = 8.0
    simota_vis_weight: float = 0.2
    point_quality_sigma: float = 0.04
    one2one_candidate_radius: float = 2.5
    one2one_obj_cost_weight: float = 1.0
    one2one_box_cost_weight: float = 0.1
    one2one_point_cost_weight: float = 8.0
    one2one_center_cost_weight: float = 0.25
    one2one_warmup_epochs: int = 10
    one2one_obj_neg_weight: float = 1.0


@dataclass
class TrainConfig:
    """Configuration for optimization and training schedule.

    Attributes:
        epochs: Number of training epochs.
        optimizer: Optimizer name (`adamw` or `sgd`).
        lr: Initial learning rate.
        weight_decay: Weight decay for optimizer.
        momentum: SGD momentum value.
        scheduler: LR scheduler type (`cosine` or `none`).
        min_lr: Minimum learning rate for cosine schedule.
        warmup_epochs: Number of initial epochs below the base learning rate.
        warmup_start_factor: Initial warmup learning rate as a fraction of `lr`.
        grad_clip_norm: Gradient norm clipping threshold.
        print_interval: Logging interval in training steps.
        seed: RNG seed for reproducibility.
        device: Device string for training.
    """

    epochs: int = 30
    optimizer: str = "adamw"  # options: adamw, sgd
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    scheduler: Literal["cosine", "none"] = "cosine"
    min_lr: float = 1e-5
    warmup_epochs: int = 1
    warmup_start_factor: float = 0.1
    grad_clip_norm: float = 10.0
    print_interval: int = 50
    seed: int = 42
    device: str = "cuda"


@dataclass
class InferConfig:
    """Configuration for post-processing thresholds during inference.

    Attributes:
        conf_threshold: Confidence threshold for candidate filtering.
        iou_threshold: IoU threshold for 3D NMS suppression.
        point_conf_threshold: Visibility threshold for keypoints.
        point_match_threshold_px: Pixel-distance threshold for point-level TP/FP/FN matching.
        point_nms_threshold_px: Pixel-distance threshold for point-level duplicate suppression.
        max_points_per_frame: Maximum visible points retained per frame after point-level suppression.
        max_detections: Maximum number of detections to keep per sample.
    """

    conf_threshold: float = 0.20
    iou_threshold: float = 0.30
    point_conf_threshold: float = 0.50
    point_match_threshold_px: float = 8.0
    point_nms_threshold_px: float = 8.0
    max_points_per_frame: int = 6
    max_detections: int = 30


@dataclass
class AppConfig:
    """Top-level application configuration object.

    Attributes:
        dataset: Dataset configuration block.
        model: Model configuration block.
        loss: Loss configuration block.
        train: Training configuration block.
        infer: Inference configuration block.
    """

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)


def get_default_config() -> AppConfig:
    """Return a fully initialized project configuration."""

    return AppConfig()
