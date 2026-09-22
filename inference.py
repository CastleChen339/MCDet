"""Batch inference and visualization for trained MCDet checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from config import AppConfig, get_default_config
from models import build_model
from utils.point_ops import flatten_pred_keypoints, match_points_by_distance


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
LABEL_TO_ID = {"debris": 0}
VIS_COLORS = {
    "tp": (0, 220, 0),
    "fp": (255, 40, 40),
    "fn": (255, 220, 0),
    "det": (0, 220, 0),
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Run MCDet inference and save visualized results.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("./runs/tiny_baseline/best.pt"),
        help="Path to trained checkpoint.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("./test_samples/testset/testset_1"),
        help="Directory containing images, or images/json subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/inference"),
        help="Directory used to save visualizations and predictions.json.",
    )
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length. Defaults to config value.")
    parser.add_argument("--stride", type=int, default=None, help="Window stride. Defaults to seq-len.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of sequences per inference batch.")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device.")
    parser.add_argument(
        "--end2end",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use one-to-one NMS-free inference (--end2end) or one-to-many inference "
            "with box/point NMS (--no-end2end). In --end2end mode, --iou, "
            "--point-nms, and --max-points-per-frame are ignored. Defaults to "
            "the checkpoint configuration."
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="Detection confidence threshold. Applies to both inference modes.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.30,
        help="3D box NMS IoU threshold. Only applies to --no-end2end; ignored by --end2end.",
    )
    parser.add_argument(
        "--point-conf",
        type=float,
        default=0.60,
        help="Keypoint visibility threshold. Applies to both inference modes.",
    )
    parser.add_argument(
        "--point-match",
        type=float,
        default=None,
        help=(
            "Point match distance threshold in pixels. Only used to evaluate and "
            "visualize sources with JSON ground truth; independent of inference mode."
        ),
    )
    parser.add_argument(
        "--point-nms",
        type=float,
        default=None,
        help=(
            "Same-frame point NMS threshold in pixels. Only applies to --no-end2end; ignored by --end2end."
        ),
    )
    parser.add_argument(
        "--max-points-per-frame",
        type=int,
        default=None,
        help=(
            "Maximum points retained per frame after point NMS. Only applies to --no-end2end; ignored by --end2end."
        ),
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=200,
        help="Maximum detections retained per sequence. Applies to both inference modes.",
    )
    parser.add_argument("--radius", type=int, default=6, help="Circle radius in output images.")
    parser.add_argument("--line-width", type=int, default=2, help="Circle line width in output images.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Square input size used for model inference.",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Disable resizing before inference. Images must already match the model input size.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    """Resolve a requested device and fall back to CPU when needed."""

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def load_checkpoint_data(path: Path, device: torch.device) -> Dict[str, Any]:
    """Load a checkpoint dictionary with compatibility for PyTorch defaults."""

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format.")


def load_model_weights(model: torch.nn.Module, checkpoint: Dict[str, Any]) -> None:
    """Load model weights from a checkpoint dictionary."""

    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a valid model state dict.")
    model.load_state_dict(state_dict, strict=True)


def configure_inference_mode(
    model: torch.nn.Module,
    cfg: AppConfig,
    requested_end2end: Optional[bool],
) -> bool:
    """Select the inference branch after checkpoint weights have been loaded."""

    runtime_end2end = (
        bool(cfg.model.end2end)
        if requested_end2end is None
        else bool(requested_end2end)
    )
    head = getattr(model, "head", None)
    has_one2one_head = head is not None and hasattr(head, "one2one_obj_pred_layers")
    if runtime_end2end and not has_one2one_head:
        raise ValueError(
            "This checkpoint has no one-to-one head and cannot use --end2end. "
            "Use --no-end2end or a checkpoint trained with end2end=True."
        )

    cfg.model.end2end = runtime_end2end
    model.model_cfg.end2end = runtime_end2end
    head.end2end = runtime_end2end
    return runtime_end2end


def resolve_config(checkpoint: Dict[str, Any], args: argparse.Namespace) -> AppConfig:
    """Resolve runtime config from checkpoint metadata and CLI overrides."""

    cfg = checkpoint.get("config")
    if not isinstance(cfg, AppConfig):
        cfg = get_default_config()
    if not hasattr(cfg.infer, "point_match_threshold_px"):
        cfg.infer.point_match_threshold_px = get_default_config().infer.point_match_threshold_px

    if args.seq_len is not None:
        cfg.dataset.seq_len = int(args.seq_len)
    cfg.infer.conf_threshold = float(args.conf)
    cfg.infer.iou_threshold = float(args.iou)
    cfg.infer.point_conf_threshold = float(args.point_conf)
    if args.point_match is not None:
        cfg.infer.point_match_threshold_px = float(args.point_match)
    if args.point_nms is not None:
        cfg.infer.point_nms_threshold_px = float(args.point_nms)
    if args.max_points_per_frame is not None:
        cfg.infer.max_points_per_frame = int(args.max_points_per_frame)
    cfg.infer.max_detections = int(args.max_detections)
    cfg.train.device = args.device
    return cfg


def discover_inputs(source: Path) -> Tuple[List[Path], Optional[Path]]:
    """Find image files and an optional LabelMe-style json directory."""

    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    image_dir = source / "images" if (source / "images").is_dir() else source
    json_dir = source / "json" if (source / "json").is_dir() else None

    image_paths = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        raise ValueError(f"No supported images found in {image_dir}")

    if json_dir is not None:
        missing = [p.name for p in image_paths if not (json_dir / f"{p.stem}.json").is_file()]
        if missing:
            raise FileNotFoundError(
                "JSON annotations are present, but these images have no matching json: "
                + ", ".join(missing[:5])
            )

    return image_paths, json_dir


def make_windows(num_images: int, seq_len: int, stride: int) -> List[Tuple[int, int]]:
    """Create sequence window start/end index pairs."""

    if seq_len <= 0:
        raise ValueError(f"seq-len must be positive, got {seq_len}.")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}.")
    if num_images < seq_len:
        raise ValueError(f"Need at least {seq_len} images, but found {num_images}.")

    return [(start, start + seq_len) for start in range(0, num_images - seq_len + 1, stride)]


def read_frame_for_model(path: Path, channels: int, image_size: int, resize: bool) -> torch.Tensor:
    """Read one frame as CHW float tensor in [0, 1]."""

    mode = "L" if channels == 1 else "RGB"
    with Image.open(path) as image:
        image = image.convert(mode)
        if resize:
            image = image.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(image).copy()

    if channels == 1:
        tensor = torch.from_numpy(arr).unsqueeze(0)
    else:
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor.contiguous().float() / 255.0


def load_sequence_tensor(
    image_paths: Sequence[Path],
    channels: int,
    image_size: int,
    resize: bool,
) -> torch.Tensor:
    """Load a sequence as a tensor with shape C,T,H,W."""

    frames = [
        read_frame_for_model(path, channels=channels, image_size=image_size, resize=resize)
        for path in image_paths
    ]
    return torch.stack(frames, dim=1)


def load_visual_canvases(image_paths: Iterable[Path]) -> Dict[str, Image.Image]:
    """Load RGB visualization canvases keyed by image name."""

    canvases: Dict[str, Image.Image] = {}
    for path in image_paths:
        with Image.open(path) as image:
            canvases[path.name] = image.convert("RGB")
    return canvases


def load_annotation(path: Path) -> Dict[str, Any]:
    """Load one LabelMe-style annotation file."""

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def expand_interval(
    v_min: float,
    v_max: float,
    padding: float,
    min_size: float,
    lower: float,
    upper: float,
) -> Tuple[float, float]:
    """Match the training transform interval expansion for GT boxes."""

    v_min = float(v_min) - padding
    v_max = float(v_max) + padding

    center = 0.5 * (v_min + v_max)
    size = max(v_max - v_min, min_size)
    size = min(size, max(upper - lower, min_size))
    half = 0.5 * size

    v_min = center - half
    v_max = center + half

    if v_min < lower:
        shift = lower - v_min
        v_min += shift
        v_max += shift
    if v_max > upper:
        shift = v_max - upper
        v_min -= shift
        v_max -= shift

    v_min = max(v_min, lower)
    v_max = min(v_max, upper)

    cur_size = v_max - v_min
    if cur_size < min_size:
        deficit = min_size - cur_size
        left_room = v_min - lower
        right_room = upper - v_max
        add_left = min(deficit * 0.5, left_room)
        add_right = min(deficit - add_left, right_room)
        v_min -= add_left
        v_max += add_right

        cur_size = v_max - v_min
        if cur_size < min_size and left_room > add_left:
            v_min -= min(min_size - cur_size, v_min - lower)
        cur_size = v_max - v_min
        if cur_size < min_size and right_room > add_right:
            v_max += min(min_size - cur_size, upper - v_max)

    return v_min, v_max


def build_gt_tracks(
    json_paths: Sequence[Path],
    label_to_id: Dict[str, int],
    min_points: int = 1,
) -> Dict[str, Any]:
    """Build normalized GT trajectory boxes and per-frame point tracks."""

    annotations = [load_annotation(path) for path in json_paths]
    if not annotations:
        return {"boxes": torch.zeros((0, 6), dtype=torch.float32), "tracks": [], "points": []}

    width = int(annotations[0].get("imageWidth", 0))
    height = int(annotations[0].get("imageHeight", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {json_paths[0]}")

    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for frame_idx, ann in enumerate(annotations):
        if int(ann.get("imageWidth", width)) != width or int(ann.get("imageHeight", height)) != height:
            raise ValueError("All annotations in one sequence must have the same image size.")
        for shape in ann.get("shapes", []):
            label = shape.get("label", "")
            points = shape.get("points") or []
            gid = shape.get("group_id")
            if label not in label_to_id or gid is None or not points:
                continue
            x, y = points[0][:2]
            grouped[gid].append(
                {
                    "x": float(x),
                    "y": float(y),
                    "frame": frame_idx,
                    "label": int(label_to_id[label]),
                }
            )

    boxes: List[torch.Tensor] = []
    tracks: List[Dict[str, Any]] = []
    gt_points: List[Dict[str, Any]] = []
    seq_len = len(json_paths)

    for gid, points in grouped.items():
        if len(points) < min_points:
            continue
        xs = [p["x"] for p in points]
        ys = [p["y"] for p in points]
        ts = [p["frame"] for p in points]

        x_min, x_max = expand_interval(min(xs), max(xs), 2.0, 4.0, 0.0, max(width - 1, 0))
        y_min, y_max = expand_interval(min(ys), max(ys), 2.0, 4.0, 0.0, max(height - 1, 0))
        t_min, t_max = expand_interval(min(ts), max(ts), 0.5, 1.0, 0.0, max(seq_len - 1, 0))

        box = torch.tensor(
            [
                ((x_min + x_max) * 0.5) / float(width),
                ((y_min + y_max) * 0.5) / float(height),
                ((t_min + t_max) * 0.5) / float(seq_len),
                (x_max - x_min) / float(width),
                (y_max - y_min) / float(height),
                (t_max - t_min) / float(seq_len),
            ],
            dtype=torch.float32,
        )
        boxes.append(box)
        gt_idx = len(tracks)
        tracks.append({"group_id": gid, "points": points})
        for point in points:
            frame_idx = int(point["frame"])
            gt_points.append(
                {
                    "gt_idx": gt_idx,
                    "group_id": gid,
                    "kpt_idx": frame_idx,
                    "frame": frame_idx,
                    "x": float(point["x"]),
                    "y": float(point["y"]),
                }
            )

    if not boxes:
        box_tensor = torch.zeros((0, 6), dtype=torch.float32)
    else:
        box_tensor = torch.stack(boxes, dim=0)
    return {"boxes": box_tensor, "tracks": tracks, "points": gt_points}


def draw_circle(
    canvas: Image.Image,
    x: float,
    y: float,
    color: Tuple[int, int, int],
    radius: int,
    line_width: int,
) -> None:
    """Draw one outlined circle on a canvas."""

    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    x = max(0.0, min(float(w - 1), float(x)))
    y = max(0.0, min(float(h - 1), float(y)))
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=color,
        width=line_width,
    )


def draw_prediction_label(
    canvas: Image.Image,
    x: float,
    y: float,
    score: float,
    visibility: float,
    color: Tuple[int, int, int],
    radius: int,
) -> None:
    """Draw detection score and keypoint visibility above one predicted point."""

    draw = ImageDraw.Draw(canvas)
    label = f"score: {float(score):.2f}\nvisibility: {float(visibility):.2f}"
    text_bbox = draw.multiline_textbbox(
        (0, 0),
        label,
        spacing=1,
        stroke_width=1,
    )
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    padding = 2

    canvas_width, canvas_height = canvas.size
    label_x = max(
        padding,
        min(float(canvas_width - text_width - padding), float(x) - text_width * 0.5),
    )
    label_y = max(
        padding,
        min(
            float(canvas_height - text_height - padding),
            float(y) - float(radius) - text_height - 2 * padding,
        ),
    )
    draw.rectangle(
        (
            label_x - padding,
            label_y - padding,
            label_x + text_width + padding,
            label_y + text_height + padding,
        ),
        fill=(0, 0, 0),
    )
    draw.text(
        (label_x, label_y),
        label,
        fill=color,
        spacing=1,
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )


def draw_point_record(
    canvases: Dict[str, Image.Image],
    sequence_paths: Sequence[Path],
    point: Dict[str, Any],
    status: str,
    radius: int,
    line_width: int,
    score: Optional[float] = None,
    visibility: Optional[float] = None,
) -> bool:
    """Draw one prediction or GT point record."""

    frame_idx = int(point["frame"])
    if frame_idx < 0 or frame_idx >= len(sequence_paths):
        return False

    canvas = canvases[sequence_paths[frame_idx].name]
    draw_circle(
        canvas,
        float(point["x"]),
        float(point["y"]),
        color=VIS_COLORS[status],
        radius=radius,
        line_width=line_width,
    )
    if score is not None and visibility is not None:
        draw_prediction_label(
            canvas=canvas,
            x=float(point["x"]),
            y=float(point["y"]),
            score=score,
            visibility=visibility,
            color=VIS_COLORS[status],
            radius=radius,
        )
    return True


def tensor_to_list(tensor: torch.Tensor) -> List[Any]:
    """Convert a tensor to nested Python lists for JSON output."""

    return tensor.detach().cpu().tolist()


def run_inference(args: argparse.Namespace) -> Dict[str, Any]:
    """Run inference, visualization, and JSON export."""

    device = resolve_device(args.device)
    checkpoint = load_checkpoint_data(args.weights, device)
    cfg = resolve_config(checkpoint, args)
    seq_len = int(cfg.dataset.seq_len)
    stride = int(args.stride) if args.stride is not None else seq_len

    image_paths, json_dir = discover_inputs(args.source)
    windows = make_windows(len(image_paths), seq_len=seq_len, stride=stride)
    has_gt = json_dir is not None
    resize = not bool(args.no_resize)

    checkpoint_end2end = bool(cfg.model.end2end)
    model = build_model(cfg.model, seq_len=seq_len).to(device)
    load_model_weights(model, checkpoint)
    runtime_end2end = configure_inference_mode(model, cfg, args.end2end)
    model.eval()

    args.output.mkdir(parents=True, exist_ok=True)
    vis_dir = args.output / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    canvases = load_visual_canvases(image_paths)

    summary: Dict[str, Any] = {
        "weights": str(args.weights),
        "source": str(args.source),
        "output": str(args.output),
        "has_ground_truth": has_gt,
        "seq_len": seq_len,
        "stride": stride,
        "checkpoint_end2end": checkpoint_end2end,
        "end2end": runtime_end2end,
        "conf_threshold": cfg.infer.conf_threshold,
        "iou_threshold": cfg.infer.iou_threshold,
        "point_conf_threshold": cfg.infer.point_conf_threshold,
        "point_match_threshold_px": cfg.infer.point_match_threshold_px,
        "point_nms_threshold_px": cfg.infer.point_nms_threshold_px,
        "max_points_per_frame": cfg.infer.max_points_per_frame,
        "windows": [],
        "totals": {"tp": 0, "fp": 0, "fn": 0, "detections": 0, "detected_points": 0},
    }

    with torch.inference_mode():
        for batch_start in range(0, len(windows), int(args.batch_size)):
            batch_windows = windows[batch_start : batch_start + int(args.batch_size)]
            batch_sequences: List[torch.Tensor] = []
            for start, end in batch_windows:
                batch_sequences.append(
                    load_sequence_tensor(
                        image_paths[start:end],
                        channels=int(cfg.dataset.image_channels),
                        image_size=int(args.image_size),
                        resize=resize,
                    )
                )

            images = torch.stack(batch_sequences, dim=0).to(device, non_blocking=True)
            detections = model.predict(
                images,
                conf_threshold=cfg.infer.conf_threshold,
                iou_threshold=cfg.infer.iou_threshold,
                point_conf_threshold=cfg.infer.point_conf_threshold,
                point_nms_threshold_px=cfg.infer.point_nms_threshold_px,
                max_points_per_frame=cfg.infer.max_points_per_frame,
                image_size=(int(args.image_size), int(args.image_size)),
                max_detections=cfg.infer.max_detections,
            )

            for (start, end), detection in zip(batch_windows, detections):
                sequence_paths = image_paths[start:end]
                gt_data = None
                match_result = None
                first_canvas = canvases[sequence_paths[0].name]
                width, height = first_canvas.size
                pred_points = flatten_pred_keypoints(
                    detection["keypoints"],
                    point_conf_threshold=cfg.infer.point_conf_threshold,
                    image_size=(height, width),
                )
                gt_points: List[Dict[str, Any]] = []

                if has_gt and json_dir is not None:
                    json_paths = [json_dir / f"{path.stem}.json" for path in sequence_paths]
                    gt_data = build_gt_tracks(json_paths, LABEL_TO_ID)
                    gt_points = gt_data["points"]
                    match_result = match_points_by_distance(
                        pred_points,
                        gt_points,
                        distance_threshold_px=cfg.infer.point_match_threshold_px,
                    )

                matched_pred = set()
                matched_gt = set()
                point_match_rows: List[Dict[str, Any]] = []
                if match_result is not None:
                    for pred_idx, gt_idx, dist in match_result.matches:
                        matched_pred.add(pred_idx)
                        matched_gt.add(gt_idx)
                        point_match_rows.append(
                            {
                                "pred_point": pred_idx,
                                "gt_point": gt_idx,
                                "distance_px": dist,
                            }
                        )

                for pred_idx, point in enumerate(pred_points):
                    status = "tp" if pred_idx in matched_pred else ("fp" if has_gt else "det")
                    det_idx = int(point["det_idx"])
                    draw_point_record(
                        canvases=canvases,
                        sequence_paths=sequence_paths,
                        point=point,
                        status=status,
                        radius=int(args.radius),
                        line_width=int(args.line_width),
                        score=float(detection["scores"][det_idx].detach().cpu()),
                        visibility=float(point["score"]),
                    )

                for gt_idx, point in enumerate(gt_points):
                    if gt_idx in matched_gt:
                        continue
                    draw_point_record(
                        canvases=canvases,
                        sequence_paths=sequence_paths,
                        point=point,
                        status="fn",
                        radius=int(args.radius),
                        line_width=int(args.line_width),
                    )

                num_det = int(detection["bboxes"].shape[0])
                num_pred_points = len(pred_points)
                num_gt_points = len(gt_points)
                pred_status = [
                    "tp" if idx in matched_pred else ("fp" if has_gt else "det")
                    for idx in range(num_pred_points)
                ]

                window_summary = {
                    "start": start,
                    "end": end,
                    "images": [str(path) for path in sequence_paths],
                    "num_detections": num_det,
                    "num_pred_points": num_pred_points,
                    "num_gt_points": num_gt_points,
                    "point_matches": point_match_rows,
                    "points": [
                        {
                            **point,
                            "status": pred_status[idx],
                        }
                        for idx, point in enumerate(pred_points)
                    ],
                    "detections": [
                        {
                            "score": float(detection["scores"][idx].detach().cpu()),
                            "bbox": tensor_to_list(detection["bboxes"][idx]),
                            "keypoints": tensor_to_list(detection["keypoints"][idx].view(-1, 4)),
                        }
                        for idx in range(num_det)
                    ],
                }
                summary["windows"].append(window_summary)
                summary["totals"]["detections"] += num_det
                summary["totals"]["detected_points"] += num_pred_points
                if match_result is not None:
                    summary["totals"]["tp"] += len(match_result.matches)
                    summary["totals"]["fp"] += len(match_result.unmatched_pred)
                    summary["totals"]["fn"] += len(match_result.unmatched_gt)

    for name, canvas in canvases.items():
        canvas.save(vis_dir / name)

    predictions_path = args.output / "predictions.json"
    with predictions_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Processed {len(image_paths)} images in {len(windows)} sequence window(s).")
    print(f"Ground truth: {'yes' if has_gt else 'no'}")
    print(
        "Inference mode: "
        f"{'end2end one-to-one (NMS-free)' if runtime_end2end else 'one-to-many (NMS)'}"
    )
    print(f"Totals: {summary['totals']}")
    print(f"Visualizations: {vis_dir}")
    print(f"Predictions: {predictions_path}")
    return summary


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
