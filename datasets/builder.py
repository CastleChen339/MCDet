"""Dataset and DataLoader builders for AstroDim dim-moving-object detection."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from config import AppConfig

try:
    from datasets.AstroDim_dataset import AstroDimDataset
    from datasets.AstroDim_transforms import (
        GenerateBoundingBoxes3D,
        Normalize,
        RandomFlip3D,
        RandomResize3D,
        Crop,
        RandomValidCrop,
        RemoveSingletonGroups,
    )
except ImportError as exc:
    raise ImportError(
        "Failed to import AstroDim dataset/transform modules. "
        "Ensure AstroDim_dataset.py and AstroDim_transforms.py are in workspace root."
    ) from exc


def _to_plain_tensor(obj: Any) -> Any:
    """Convert TVTensor-like objects to plain tensors while keeping other types unchanged.

    Args:
        obj: Input object that may wrap a tensor.

    Returns:
        Any: Plain tensor when available, otherwise the original object.
    """

    if hasattr(obj, "to_tensor"):
        return obj.to_tensor()
    return obj


def astrodim_collate_fn(batch: List[Tuple[Any, Dict[str, Any]]]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Collate a batch of `(ImageSequence, target_dict)` into tensors usable by PyTorch.

    The image tensor shape is `(B, C, T, H, W)`. Targets remain a list of dictionaries
    because each sample can contain a different number of trajectories.

    Args:
        batch: List of dataset samples.

    Returns:
        Tuple[torch.Tensor, List[Dict[str, Any]]]: Image batch and target list.
    """

    images: List[torch.Tensor] = []
    targets: List[Dict[str, Any]] = []

    for img_seq, target in batch:
        image_tensor = _to_plain_tensor(img_seq).float()
        images.append(image_tensor)

        converted: Dict[str, Any] = {}
        for key, value in target.items():
            converted[key] = _to_plain_tensor(value)

        if "points" in converted:
            converted["points"] = converted["points"].float()
        if "bboxes" in converted:
            converted["bboxes"] = converted["bboxes"].float()
        if "points_labels" in converted:
            converted["points_labels"] = converted["points_labels"].long()
        if "bboxes_labels" in converted:
            converted["bboxes_labels"] = converted["bboxes_labels"].long()
        if "points_group_ids" in converted:
            converted["points_group_ids"] = converted["points_group_ids"].long()
        if "bboxes_group_ids" in converted:
            converted["bboxes_group_ids"] = converted["bboxes_group_ids"].long()

        targets.append(converted)

    image_batch = torch.stack(images, dim=0)
    return image_batch, targets


def build_train_transforms(cfg: AppConfig) -> v2.Compose:
    """Build the training transform pipeline in the required strict order.

    Args:
        cfg: Application configuration.

    Returns:
        v2.Compose: Training transform pipeline.
    """

    return v2.Compose([
        RandomValidCrop(crop_size=cfg.dataset.crop_size, least_target_num=cfg.dataset.min_group_points),
        RemoveSingletonGroups(min_group_points=cfg.dataset.min_group_points),
        GenerateBoundingBoxes3D(),
        Normalize(),
    ])


def build_eval_transforms(cfg: AppConfig) -> v2.Compose:
    """Build a deterministic evaluation pipeline while preserving transform order.

    Args:
        cfg: Application configuration.

    Returns:
        v2.Compose: Evaluation transform pipeline.
    """

    return v2.Compose([
        RandomValidCrop(crop_size=cfg.dataset.crop_size, least_target_num=cfg.dataset.min_group_points),
        RemoveSingletonGroups(min_group_points=cfg.dataset.min_group_points),
        GenerateBoundingBoxes3D(),
        Normalize(),
    ])


def build_train_dataset(cfg: AppConfig) -> AstroDimDataset:
    """Create the training dataset from config.

    Args:
        cfg: Application configuration.

    Returns:
        AstroDimDataset: Training dataset instance.
    """

    return AstroDimDataset(
        main_folder=cfg.dataset.train_root,
        seq_len=cfg.dataset.seq_len,
        transforms=build_train_transforms(cfg),
        image_channels=cfg.dataset.image_channels,
        use_torchvision_io=cfg.dataset.use_torchvision_io,
        min_group_points=cfg.dataset.min_group_points,
    )


def build_eval_dataset(cfg: AppConfig) -> AstroDimDataset:
    """Create the validation dataset from config.

    Args:
        cfg: Application configuration.

    Returns:
        AstroDimDataset: Evaluation dataset instance.
    """

    return AstroDimDataset(
        main_folder=cfg.dataset.val_root,
        seq_len=cfg.dataset.seq_len,
        transforms=build_eval_transforms(cfg),
        image_channels=cfg.dataset.image_channels,
        use_torchvision_io=cfg.dataset.use_torchvision_io,
        min_group_points=cfg.dataset.min_group_points,
    )


def build_dataloaders(cfg: AppConfig) -> Tuple[DataLoader, DataLoader]:
    """Build training and validation dataloaders with a custom collate function.

    Args:
        cfg: Application configuration.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation dataloaders.
    """

    train_dataset = build_train_dataset(cfg)
    val_dataset = build_eval_dataset(cfg)

    pin_memory = bool(
        cfg.dataset.pin_memory
        and cfg.train.device.startswith("cuda")
        and torch.cuda.is_available()
    )

    shared_loader_kwargs = {
        "batch_size": cfg.dataset.batch_size,
        "num_workers": cfg.dataset.num_workers,
        "pin_memory": pin_memory,
        "collate_fn": astrodim_collate_fn,
    }

    if cfg.dataset.num_workers > 0:
        shared_loader_kwargs["prefetch_factor"] = max(1, int(cfg.dataset.prefetch_factor))
        shared_loader_kwargs["persistent_workers"] = bool(cfg.dataset.persistent_workers)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **shared_loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **shared_loader_kwargs,
    )

    return train_loader, val_loader
