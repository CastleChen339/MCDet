"""Training engine for MCDet dim-target detector."""

from __future__ import annotations

import math
import os
from typing import Callable, Dict, Optional

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from config import AppConfig
from utils.misc import AverageMeter, format_loss_dict, move_targets_to_device


class Trainer:
    """Encapsulates optimization loop, checkpointing, and logging.

    Args:
        model: Detector model to train.
        criterion: Loss module computing training objectives.
        optimizer: Optimizer instance for gradient-based updates.
        scheduler: Optional LR scheduler; None disables scheduling.
        cfg: Top-level AppConfig for training hyperparameters.
        device: Target compute device.
        save_dir: Directory for checkpoint persistence.
        log_fn: Optional logging callback; defaults to print.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        cfg: AppConfig,
        device: torch.device,
        save_dir: str,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Initialize trainer with model, criterion, and optimization utilities."""

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device
        self.save_dir = save_dir
        self.log = log_fn or print

        os.makedirs(self.save_dir, exist_ok=True)

    def train_one_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Run one full epoch of training and return averaged loss metrics.

        Args:
            dataloader: Training DataLoader.
            epoch: Current epoch index.

        Returns:
            Dict[str, float]: Averaged loss metrics.
        """

        self.model.train()
        meters: Dict[str, AverageMeter] = {}

        for step, (images, targets) in enumerate(dataloader, start=1):
            images = images.to(self.device, non_blocking=True)
            targets = move_targets_to_device(targets, self.device)

            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.model(images)
            loss, loss_items = self.criterion(predictions, targets)
            loss.backward()

            if self.cfg.train.grad_clip_norm > 0:
                clip_grad_norm_(
                    self.model.parameters(), max_norm=self.cfg.train.grad_clip_norm
                )

            self.optimizer.step()

            batch_size = images.shape[0]
            for key, value in loss_items.items():
                if key not in meters:
                    meters[key] = AverageMeter()
                meters[key].update(float(value.item()), n=batch_size)

            if step % self.cfg.train.print_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self.log(
                    f"[Train] epoch={epoch}/{self.cfg.train.epochs} "
                    f"step={step}/{len(dataloader)} "
                    f"lr={lr:.6f} {format_loss_dict(loss_items)}"
                )

        return {key: meter.avg for key, meter in meters.items()}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        evaluator: Optional[object] = None,
        start_epoch: int = 1,
        best_f1: float = -1.0,
        best_epoch: int = -1,
    ) -> Dict[str, float]:
        """Run end-to-end training with optional validation and checkpointing.

        Args:
            train_loader: Training DataLoader.
            val_loader: Optional validation DataLoader.
            evaluator: Optional evaluator for validation metrics.
            start_epoch: Starting epoch index.
            best_f1: Best validation F1 observed so far.
            best_epoch: Epoch index of best validation F1.

        Returns:
            Dict[str, float]: Summary with best metric values.
        """

        if start_epoch > self.cfg.train.epochs:
            self.log(
                "Configured total epochs already reached "
                f"(resume_epoch={start_epoch - 1}, "
                f"total_epochs={self.cfg.train.epochs})."
            )
            return {"best_f1": float(best_f1), "best_epoch": float(best_epoch)}

        for epoch in range(start_epoch, self.cfg.train.epochs + 1):
            # Update criterion epoch tracker (used for VFL warmup).
            if hasattr(self.criterion, "set_epoch"):
                self.criterion.set_epoch(epoch)

            epoch_lr = float(self.optimizer.param_groups[0]["lr"])
            train_stats = self.train_one_epoch(train_loader, epoch)
            train_stats["lr"] = epoch_lr
            self.log(f"[Train][Epoch {epoch}/{self.cfg.train.epochs}] {train_stats}")

            # The scheduler sets the learning rate for the next epoch.
            if self.scheduler is not None:
                self.scheduler.step()

            val_stats: Dict[str, float] = {}
            is_best = False
            if val_loader is not None and evaluator is not None:
                val_stats = evaluator.evaluate(
                    self.model,
                    val_loader,
                    self.device,
                    print_interval=self.cfg.train.print_interval,
                    log_fn=self.log,
                    epoch=epoch,
                    total_epochs=self.cfg.train.epochs,
                )
                self.log(f"[Val][Epoch {epoch}/{self.cfg.train.epochs}] {val_stats}")

                if val_stats.get("f1", 0.0) > best_f1:
                    best_f1 = val_stats["f1"]
                    best_epoch = epoch
                    is_best = True

            if is_best:
                self._save_checkpoint(
                    os.path.join(self.save_dir, "best.pt"),
                    epoch,
                    best_metric=best_f1,
                    best_epoch=best_epoch,
                )

            self._save_checkpoint(
                os.path.join(self.save_dir, "last.pt"),
                epoch,
                best_metric=best_f1,
                best_epoch=best_epoch,
            )

        return {"best_f1": best_f1, "best_epoch": float(best_epoch)}

    def _save_checkpoint(
        self, path: str, epoch: int, best_metric: float, best_epoch: int
    ) -> None:
        """Persist model and optimizer states for resume or inference.

        Args:
            path: Checkpoint output path.
            epoch: Current epoch index.
            best_metric: Best validation metric value.
            best_epoch: Epoch index for the best metric.
        """

        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "config": self.cfg,
        }
        torch.save(state, path)


def build_optimizer(
    model: torch.nn.Module, cfg: AppConfig
) -> torch.optim.Optimizer:
    """Create optimizer with weight decay only applied to weight parameters.

    Bias terms and normalisation-layer parameters (BatchNorm, LayerNorm, etc.)
    are excluded from weight decay, following standard practice.

    Args:
        model: Model with parameters to optimize.
        cfg: Application configuration.

    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """

    weight_decay = cfg.train.weight_decay
    lr = cfg.train.lr

    # Separate parameters that should receive weight decay.
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Exclude bias and normalisation parameters from weight decay.
        if (
            "bias" in name
            or "bn" in name
            or "norm" in name
            or "BatchNorm" in name
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    name = cfg.train.optimizer.lower()
    if name == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=cfg.train.momentum,
            nesterov=True,
        )

    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: AppConfig,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create an epoch-indexed learning-rate scheduler.

    The returned scheduler initializes the optimizer LR for epoch 1 and is
    stepped after each epoch to prepare the LR for the next epoch.

    Args:
        optimizer: Optimizer instance to schedule.
        cfg: Application configuration.

    Returns:
        Optional[torch.optim.lr_scheduler._LRScheduler]: Scheduler or None.
    """

    epochs = max(int(cfg.train.epochs), 1)
    warmup_epochs = max(int(getattr(cfg.train, "warmup_epochs", 0)), 0)
    warmup_epochs = min(warmup_epochs, max(epochs - 1, 0))
    warmup_start_factor = float(
        getattr(cfg.train, "warmup_start_factor", 0.1)
    )
    if not 0.0 < warmup_start_factor <= 1.0:
        raise ValueError("warmup_start_factor must be in (0, 1].")

    base_lr = float(cfg.train.lr)
    if base_lr <= 0.0:
        raise ValueError("train.lr must be positive.")

    name = cfg.train.scheduler.lower()

    def warmup_factor(epoch: int) -> Optional[float]:
        if warmup_epochs <= 0 or epoch > warmup_epochs:
            return None
        progress = (epoch - 1) / max(warmup_epochs, 1)
        return warmup_start_factor + (1.0 - warmup_start_factor) * progress

    if name == "cosine":
        min_lr = float(cfg.train.min_lr)
        if not 0.0 <= min_lr <= base_lr:
            raise ValueError("min_lr must be in [0, train.lr].")
        min_factor = min_lr / base_lr
        cosine_epochs = epochs - warmup_epochs

        def cosine_factor(step_index: int) -> float:
            epoch = step_index + 1
            factor = warmup_factor(epoch)
            if factor is not None:
                return factor
            cosine_index = max(epoch - warmup_epochs - 1, 0)
            denominator = max(cosine_epochs - 1, 1)
            progress = min(cosine_index / denominator, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        return LambdaLR(optimizer, lr_lambda=cosine_factor)

    if name == "none":
        return None

    raise ValueError(
        f"Unsupported scheduler {cfg.train.scheduler!r}; expected 'cosine' or 'none'."
    )
