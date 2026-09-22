"""Training entrypoint for MCDet dim moving object detector."""

import argparse
import logging
import os
from dataclasses import asdict
from datetime import datetime
from typing import Callable, Optional

import torch

from config import get_default_config
from datasets import build_dataloaders
from engine import Evaluator, Trainer, build_optimizer, build_scheduler
from losses import TemporalDetectionCriterion
from models import build_model
from utils.misc import set_seed



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for runtime-only training options.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Train MCDet detector")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./runs/train_latest_head_m",
        help="Checkpoint output directory",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to training log file (.txt). Defaults to <save-dir>/train_log_<timestamp>.txt",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted training (defaults to <save-dir>/last.pt).",
    )
    parser.add_argument(
        "--resume-weights",
        type=str,
        default=None,
        help="Checkpoint path to resume from. If omitted with --resume, uses <save-dir>/last.pt.",
    )
    parser.add_argument("--model-size", choices=["n", "s", "m", "l", "x"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default=None)
    parser.add_argument("--end2end", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--one2one-full-gradient",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--one2one-separate-head",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def build_logger(save_dir: str, log_file: Optional[str] = None) -> tuple[logging.Logger, str]:
    """Create a logger that writes to both console and a text log file.

    Args:
        save_dir: Directory for training artifacts and logs.
        log_file: Optional explicit log file path.

    Returns:
        Tuple[logging.Logger, str]: Logger instance and resolved log file path.
    """

    os.makedirs(save_dir, exist_ok=True)
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(save_dir, f"train_log_{timestamp}.txt")
    else:
        log_path = log_file
    log_dir = os.path.dirname(os.path.abspath(log_path))
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("mcdet.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_path


def log_training_config(logger: logging.Logger, cfg: object, args: argparse.Namespace) -> None:
    """Log the full training configuration to console and log file.

    Args:
        logger: Training logger instance.
        cfg: Configuration dataclass object.
        args: Parsed runtime arguments.
    """

    def format_value(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        if isinstance(value, (list, tuple)):
            inner = ",".join(format_value(item) for item in value)
            return f"[{inner}]"
        return str(value)

    def format_section(section: dict[str, object]) -> str:
        parts = [f"{key}={format_value(section[key])}" for key in sorted(section.keys())]
        return ", ".join(parts)

    config_dict = asdict(cfg)
    runtime_dict = {
        "save_dir": args.save_dir,
        "log_file": args.log_file,
        "resume": args.resume,
        "resume_weights": args.resume_weights,
        "model_size": args.model_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "scheduler": args.scheduler,
        "end2end": args.end2end,
        "one2one_full_gradient": args.one2one_full_gradient,
        "one2one_separate_head": args.one2one_separate_head,
    }

    logger.info("Training configuration (start):")
    for section_name in ["dataset", "infer", "loss", "model", "train"]:
        if section_name in config_dict:
            logger.info(f"{section_name}: {format_section(config_dict[section_name])}")
    logger.info(f"runtime: {format_section(runtime_dict)}")


def resolve_device(device_name: str, log_fn: Optional[Callable[[str], None]] = None) -> torch.device:
    """Resolve configured device string with graceful CUDA fallback.

    Args:
        device_name: Requested device string (e.g., "cuda", "cpu").
        log_fn: Optional logging callback for fallback messages.

    Returns:
        torch.device: Resolved device instance.
    """

    log = log_fn or print
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        log("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def load_resume_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    ckpt_path: str,
    device: torch.device,
    log_fn: Callable[[str], None],
) -> tuple[int, float, int]:
    """Load training state for resume and return next epoch and best metrics.

    Args:
        model: Model to restore weights into.
        optimizer: Optimizer to restore state into.
        scheduler: Optional scheduler to restore state into.
        ckpt_path: Path to the checkpoint file.
        device: Device mapping for checkpoint loading.
        log_fn: Logging callback for resume warnings.

    Returns:
        Tuple[int, float, int]: Next epoch index, best F1, and best epoch.
    """

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported resume checkpoint format.")

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
        log_fn("Checkpoint does not include optimizer/scheduler state; restarting from epoch 1.")
        return 1, -1.0, -1

    optimizer_state = checkpoint.get("optimizer")
    if isinstance(optimizer_state, dict):
        optimizer.load_state_dict(optimizer_state)
    else:
        log_fn("Optimizer state missing in checkpoint; optimizer will start fresh.")

    scheduler_state = checkpoint.get("scheduler")
    if scheduler is not None and isinstance(scheduler_state, dict):
        scheduler.load_state_dict(scheduler_state)
    elif scheduler is not None:
        log_fn("Scheduler state missing in checkpoint; scheduler will start fresh.")

    resume_epoch = int(checkpoint.get("epoch", 0))
    start_epoch = max(resume_epoch + 1, 1)
    best_f1 = float(checkpoint.get("best_metric", -1.0))
    best_epoch = int(checkpoint.get("best_epoch", resume_epoch if best_f1 >= 0 else -1))
    return start_epoch, best_f1, best_epoch


def main() -> None:
    """Build all components and run the full training + validation loop."""

    args = parse_args()
    cfg = get_default_config()
    if args.model_size is not None:
        cfg.model.model_size = args.model_size
    if args.epochs is not None:
        cfg.train.epochs = max(int(args.epochs), 1)
    if args.lr is not None:
        cfg.train.lr = float(args.lr)
    if args.scheduler is not None:
        cfg.train.scheduler = args.scheduler
    if args.end2end is not None:
        cfg.model.end2end = bool(args.end2end)
    if args.one2one_full_gradient is not None:
        cfg.model.one2one_full_gradient = bool(args.one2one_full_gradient)
    if args.one2one_separate_head is not None:
        cfg.model.one2one_separate_head = bool(args.one2one_separate_head)
    save_dir = args.save_dir

    logger, log_path = build_logger(save_dir, args.log_file)
    logger.info(f"Training log file: {log_path}")
    log_training_config(logger, cfg, args)

    try:
        set_seed(cfg.train.seed)
        device = resolve_device(cfg.train.device, logger.info)

        logger.info("Building dataloaders...")
        train_loader, val_loader = build_dataloaders(cfg)

        logger.info("Building model, loss, and optimizer...")
        model = build_model(cfg.model, seq_len=cfg.dataset.seq_len).to(device)
        criterion = TemporalDetectionCriterion(cfg.loss)
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)

        start_epoch = 1
        best_f1 = -1.0
        best_epoch = -1
        enable_resume = args.resume or args.resume_weights is not None
        if enable_resume:
            resume_path = args.resume_weights or os.path.join(save_dir, "last.pt")
            logger.info(f"Resuming training from checkpoint: {resume_path}")
            start_epoch, best_f1, best_epoch = load_resume_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ckpt_path=resume_path,
                device=device,
                log_fn=logger.info,
            )
            logger.info(
                f"Resume state loaded: next_epoch={start_epoch}, "
                f"best_f1={best_f1:.6f}, best_epoch={best_epoch}"
            )

        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {num_params:,}")

        evaluator = Evaluator(cfg)
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            device=device,
            save_dir=save_dir,
            log_fn=logger.info,
        )

        logger.info("Starting training...")
        summary = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            evaluator=evaluator,
            start_epoch=start_epoch,
            best_f1=best_f1,
            best_epoch=best_epoch,
        )
        logger.info(f"Training complete: {summary}")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        raise
    except Exception:
        logger.exception("Training failed with an exception.")
        raise


if __name__ == "__main__":
    main()
