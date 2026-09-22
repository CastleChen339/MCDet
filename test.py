"""Inference and evaluation entrypoint for trained MCDet models."""

from __future__ import annotations

import argparse
import copy
from typing import Dict

import torch
from torch.utils.data import DataLoader

from config import AppConfig, get_default_config
from datasets.builder import astrodim_collate_fn, build_eval_dataset
from engine import Evaluator
from models import build_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model evaluation and inference.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Evaluate MCDet detector")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--data-root", type=str, required=True, help="Path to evaluation dataset root")
    parser.add_argument("--seq-len", type=int, default=None, help="Input temporal length")
    parser.add_argument("--batch-size", type=int, default=2, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device string")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.30, help="IoU threshold")
    parser.add_argument("--point-conf", type=float, default=0.50, help="Point visibility threshold")
    parser.add_argument("--point-match", type=float, default=None, help="Point match distance threshold in pixels")
    parser.add_argument("--point-nms", type=float, default=None, help="Same-frame point NMS distance threshold in pixels")
    parser.add_argument("--max-points-per-frame", type=int, default=None, help="Maximum predicted points retained per frame")
    parser.add_argument("--show-samples", type=int, default=2, help="Number of samples to print")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    """Resolve runtime device with fallback when CUDA is unavailable.

    Args:
        device_name: Requested device string (e.g., "cuda", "cpu").

    Returns:
        torch.device: Resolved device instance.
    """

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def load_checkpoint_data(ckpt_path: str, device: torch.device) -> Dict[str, object]:
    """Load a checkpoint dictionary before constructing the model.

    Args:
        ckpt_path: Path to checkpoint file.
        device: Device used for checkpoint loading.
    """

    try:
        # The project checkpoints are generated locally and may contain full Python objects
        # (e.g., config dataclasses). Explicitly disable weights_only for compatibility
        # with PyTorch>=2.6 default behavior.
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format.")
    return checkpoint


def load_model_weights(
    model: torch.nn.Module,
    checkpoint: Dict[str, object],
) -> None:
    """Load model weights from a full checkpoint or raw state dictionary."""

    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a valid model state dict.")
    model.load_state_dict(state_dict, strict=True)


def apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Apply evaluation-time argument overrides to default configuration.

    Args:
        cfg: Default configuration to mutate.
        args: Parsed evaluation arguments.

    Returns:
        AppConfig: Updated configuration instance.
    """

    cfg.dataset.val_root = args.data_root
    cfg.dataset.batch_size = args.batch_size
    if args.seq_len is not None:
        cfg.dataset.seq_len = args.seq_len

    cfg.infer.conf_threshold = args.conf
    cfg.infer.iou_threshold = args.iou
    cfg.infer.point_conf_threshold = args.point_conf
    if args.point_match is not None:
        cfg.infer.point_match_threshold_px = args.point_match
    if args.point_nms is not None:
        cfg.infer.point_nms_threshold_px = args.point_nms
    if args.max_points_per_frame is not None:
        cfg.infer.max_points_per_frame = args.max_points_per_frame
    cfg.train.device = args.device
    return cfg


def main() -> None:
    """Run evaluation and print a small set of decoded predictions."""

    args = parse_args()
    checkpoint = load_checkpoint_data(args.weights, resolve_device(args.device))
    checkpoint_cfg = checkpoint.get("config")
    if isinstance(checkpoint_cfg, AppConfig):
        cfg = copy.deepcopy(checkpoint_cfg)
    else:
        cfg = get_default_config()
    cfg = apply_overrides(cfg, args)
    device = resolve_device(cfg.train.device)

    dataset = build_eval_dataset(cfg)

    loader_kwargs = {
        "batch_size": cfg.dataset.batch_size,
        "num_workers": cfg.dataset.num_workers,
        "pin_memory": bool(cfg.dataset.pin_memory and device.type == "cuda"),
        "drop_last": False,
        "collate_fn": astrodim_collate_fn,
    }
    if cfg.dataset.num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(cfg.dataset.prefetch_factor))
        loader_kwargs["persistent_workers"] = bool(cfg.dataset.persistent_workers)

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        **loader_kwargs,
    )

    model = build_model(cfg.model, seq_len=cfg.dataset.seq_len).to(device)
    load_model_weights(model, checkpoint)
    model.eval()

    evaluator = Evaluator(cfg)
    stats = evaluator.evaluate(model, dataloader, device)
    print(f"Evaluation stats: {stats}")

    batch = next(iter(dataloader), None)
    if batch is None:
        print("No samples were found in evaluation dataset.")
        return

    images, _ = batch
    images = images.to(device)

    detections = model.predict(
        images,
        conf_threshold=cfg.infer.conf_threshold,
        iou_threshold=cfg.infer.iou_threshold,
        point_conf_threshold=cfg.infer.point_conf_threshold,
        point_nms_threshold_px=cfg.infer.point_nms_threshold_px,
        max_points_per_frame=cfg.infer.max_points_per_frame,
        image_size=cfg.dataset.crop_size,
        max_detections=cfg.infer.max_detections,
    )

    print("Sample predictions:")
    for idx, det in enumerate(detections[: args.show_samples]):
        print(
            f"sample={idx} "
            f"num_det={det['scores'].shape[0]} "
            f"scores={det['scores'][:5].detach().cpu().tolist()}"
        )
        print(f"  bboxes(first5)={det['bboxes'][:5].detach().cpu().tolist()}")
        print(f"  keypoints(first5)={det['keypoints'][:5].detach().cpu().tolist()}")
        valid_counts = det["keypoint_valid_mask"].sum(dim=1)
        print(f"  valid_point_counts(first5)={valid_counts[:5].detach().cpu().tolist()}")


if __name__ == "__main__":
    main()
