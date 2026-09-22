"""General training/inference helpers for reproducibility and logging."""

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Seed value applied across RNGs.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class AverageMeter:
    """Track running mean statistics for scalar values."""

    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        """Accumulate a scalar value over `n` samples.

        Args:
            value: Scalar value to accumulate.
            n: Number of samples represented by the value.
        """

        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        """Return the running average or zero when no values are observed."""

        if self.count == 0:
            return 0.0
        return self.total / self.count


def move_targets_to_device(targets: List[Dict[str, object]], device: torch.device) -> List[Dict[str, object]]:
    """Move all tensor values in each target dictionary to a device.

    Args:
        targets: List of target dictionaries.
        device: Destination device.

    Returns:
        List[Dict[str, object]]: Targets with tensors moved to the device.
    """

    moved: List[Dict[str, object]] = []
    for target in targets:
        item: Dict[str, object] = {}
        for key, value in target.items():
            if isinstance(value, torch.Tensor):
                item[key] = value.to(device)
            else:
                item[key] = value
        moved.append(item)
    return moved


def format_loss_dict(loss_dict: Dict[str, torch.Tensor]) -> str:
    """Format a loss dictionary into a compact log string.

    Args:
        loss_dict: Mapping of loss names to scalar tensors.

    Returns:
        str: Formatted loss string for logging.
    """

    fields: List[str] = []
    for key, value in loss_dict.items():
        scalar = float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
        fields.append(f"{key}={scalar:.4f}")
    return " ".join(fields)
