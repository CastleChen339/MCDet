"""Transform operators for spatiotemporal dim-moving-target detection.

All transforms are implemented as torchvision v2 `Transform` subclasses and
operate on custom `TVTensor` wrappers defined in `_tensors.py`.
"""

import random
import re
import warnings
from collections import Counter
from typing import Any, Dict, Tuple, Union, List
import torch
import torchvision
from torchvision.transforms import v2
import torchvision.transforms.functional as F

try:
    from ._tensors import *
except ImportError:
    from datasets._tensors import *

__all__ = [
    'RandomFlip3D',
    'Resize3D',
    'RandomResize3D',
    'Crop',
    'RandomValidCrop',
    'RemoveSingletonGroups',
    'GenerateBoundingBoxes3D',
    'Normalize'
]


def _parse_torchvision_version(version: str) -> Tuple[int, int]:
    """Parse major/minor from version strings such as "0.20.1+cu121".

    Args:
        version: Torchvision version string.

    Returns:
        Tuple[int, int]: Parsed `(major, minor)` version numbers.
    """
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


_TV_MAJOR, _TV_MINOR = _parse_torchvision_version(torchvision.__version__)
_USE_PRIVATE_V2_HOOKS = (_TV_MAJOR, _TV_MINOR) <= (0, 20)


class CompatTransform(v2.Transform):
    """Version-gated compatibility shim for torchvision v2 hooks.

    - torchvision <= 0.20.x: dispatch through `_get_params` / `_transform`.
    - torchvision >= 0.21.x: dispatch through `make_params` / `transform`.
    """

    if _USE_PRIVATE_V2_HOOKS:
        def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
            """Default public param hook for legacy torchvision versions.

            Args:
                flat_inputs: Flattened transform inputs.

            Returns:
                Dict[str, Any]: Parameter dictionary.
            """
            return {}

        def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
            """Default public transform hook for legacy torchvision versions.

            Args:
                inpt: Input to transform.
                params: Parameters produced by `make_params`.

            Returns:
                Any: Transformed input.
            """
            raise NotImplementedError(
                f"{type(self).__name__} must implement 'transform' when using torchvision <= 0.20.x"
            )

        def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
            return self.make_params(flat_inputs)

        def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
            return self.transform(inpt, params)


class RandomFlip3D(CompatTransform):
    """Randomly flip image sequences and point coordinates.

    Horizontal and vertical flips are sampled independently and applied
    consistently across all frames and associated points.
    """

    _transformed_types = (ImageSequence, Points)

    def __init__(self, p_h=0.5, p_v=0.05):
        """Initialize flip probabilities.

        Args:
            p_h: Probability of horizontal flip.
            p_v: Probability of vertical flip.
        """
        super().__init__()
        self.p_h = p_h
        self.p_v = p_v

    def make_params(self, inpt: Any) -> Dict[str, Any]:
        """Sample whether horizontal/vertical flips should be applied.

        Args:
            inpt: Input to be transformed (unused for random sampling).

        Returns:
            Dict[str, Any]: Flip configuration for this call.
        """
        return {
            "do_h": random.random() < self.p_h,
            "do_v": random.random() < self.p_v,
        }

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """Apply flipping to supported input types.

        Args:
            inpt: Input to transform.
            params: Sampled flip configuration.

        Returns:
            Any: Flipped input when supported.
        """
        if isinstance(inpt, ImageSequence):
            return self._transform_image_sequence(inpt, params)
        elif isinstance(inpt, Points):
            return self._transform_points(inpt, params)
        return inpt

    @staticmethod
    def _transform_image_sequence(img_seq: ImageSequence, params: Dict[str, Any]) -> ImageSequence:
        """Flip all frames in an `ImageSequence` using sampled params.

        Args:
            img_seq: Input image sequence.
            params: Sampled flip configuration.

        Returns:
            ImageSequence: Flipped image sequence.
        """
        seq = img_seq.to_tensor()
        if params["do_h"]:
            seq = F.hflip(seq)
        if params["do_v"]:
            seq = F.vflip(seq)
        return ImageSequence(seq, canvas_size=img_seq.canvas_size, fps=img_seq.fps)

    @staticmethod
    def _transform_points(pts: Points, params: Dict[str, Any]) -> Points:
        """Flip point coordinates with respect to the canvas boundaries.

        Args:
            pts: Input point coordinates.
            params: Sampled flip configuration.

        Returns:
            Points: Flipped point coordinates.
        """
        w, h = pts.canvas_size[:2]
        data = pts.to_tensor().clone()
        x, y, t = data[:, 0], data[:, 1], data[:, 2]
        if params["do_h"]:
            x = w - 1 - x
        if params["do_v"]:
            y = h - 1 - y
        return Points(torch.stack([x, y, t], dim=1), canvas_size=pts.canvas_size)


class Resize3D(CompatTransform):
    """Resize image sequences and rescale point coordinates accordingly."""

    _transformed_types = (ImageSequence, Points)

    def __init__(self, size):
        """Initialize deterministic target size.

        Args:
            size: Target `(height, width)` or scalar for square resize.
        """
        super().__init__()

        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, (tuple, list)):
            if len(size) != 2:
                raise ValueError("If 'size' is a tuple or list, it must contain exactly 2 elements (height, width).")
            if not all(isinstance(s, int) for s in size):
                raise TypeError("Elements of 'size' must be integers.")
            self.size = tuple(size)
        else:
            raise TypeError(f"'size' must be an int, tuple, or list, but got {type(size).__name__}.")

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        """Return fixed resize parameters for the current transform call.

        Args:
            flat_inputs: Flattened transform inputs (unused).

        Returns:
            Dict[str, Any]: Resize parameters for height/width.
        """
        return {'h': self.size[0], 'w': self.size[1]}

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """Resize supported inputs while preserving metadata semantics.

        Args:
            inpt: Input to resize.
            params: Resize parameters containing `h` and `w`.

        Returns:
            Any: Resized input when supported.
        """
        target_h, target_w = params['h'], params['w']

        if isinstance(inpt, ImageSequence):
            data = F.resize(inpt.to_tensor(), [target_h, target_w])
            return ImageSequence(data, canvas_size=(target_w, target_h) + inpt.canvas_size[2:], fps=inpt.fps)

        elif isinstance(inpt, Points):
            w0, h0 = inpt.canvas_size[:2]
            scale_x = target_w / w0
            scale_y = target_h / h0
            d = inpt.to_tensor().clone()
            d[:, 0] *= scale_x
            d[:, 1] *= scale_y
            return Points(d, canvas_size=(target_w, target_h) + inpt.canvas_size[2:])
        return inpt


class RandomResize3D(Resize3D):
    """
    Randomly resize 3D inputs (ImageSequence or Points) to a size sampled
    from given ranges for width and height.

    Args:
        w_range: int or tuple/list of 2 ints specifying min/max width
        h_range: int or tuple/list of 2 ints specifying min/max height.
                 If None, h_range = w_range (square resize)
    """

    def __init__(self, w_range: Union[int, Tuple[int, int], list], h_range: Union[int, Tuple[int, int], list] = None):
        """Initialize random resize ranges.

        Args:
            w_range: Width range as int or `(min_w, max_w)`.
            h_range: Height range as int or `(min_h, max_h)`. If `None`,
                uses `w_range`.
        """
        if h_range is None:
            h_range = w_range

        self.w_range = self._parse_range(w_range, "w_range")
        self.h_range = self._parse_range(h_range, "h_range")
        super().__init__(size=(0, 0))

    @staticmethod
    def _parse_range(r, name):
        """Normalize resize range input to `(min_value, max_value)`.

        Args:
            r: Range value as int or `(min, max)` tuple/list.
            name: Parameter name used for error messages.

        Returns:
            Tuple[int, int]: Normalized `(min, max)` range.
        """
        if isinstance(r, int):
            return r, r
        elif isinstance(r, (tuple, list)):
            if len(r) != 2:
                raise ValueError(f"{name} must have exactly 2 elements (min, max)")
            if not all(isinstance(x, int) for x in r):
                raise TypeError(f"Elements of {name} must be integers")
            return tuple(r)
        else:
            raise TypeError(f"{name} must be int, tuple, or list, got {type(r).__name__}")

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        """Sample a random target size inside configured ranges.

        Args:
            flat_inputs: Flattened transform inputs (unused).

        Returns:
            Dict[str, Any]: Sampled resize parameters.
        """
        target_w = random.randint(self.w_range[0], self.w_range[1])
        target_h = random.randint(self.h_range[0], self.h_range[1])
        return {'h': target_h, 'w': target_w}


class Crop(CompatTransform):
    """Deterministically crop image sequences and point annotations.

    Crops the region defined by top-left corner ``(x0, y0)`` with the given
    width and height. Points outside the crop region are removed, and
    remaining point coordinates are shifted relative to the new origin.
    """

    _transformed_types = (ImageSequence, Points, PointLabels, PointGroupIDs)

    def __init__(self, x0: int, y0: int, crop_size: Tuple[int, int]):
        """Initialize deterministic crop parameters.

        Args:
            x0: X-coordinate of the top-left corner (column index).
            y0: Y-coordinate of the top-left corner (row index).
            crop_size: ``(target_h, target_w)`` in pixels.
        """
        super().__init__()
        self.x0 = x0
        self.y0 = y0
        self.crop_size = crop_size

    def make_params(self, inpt: Any) -> Dict[str, Any]:
        """Build crop region parameters and point keep-mask.

        Args:
            inpt: Transform input list containing ``Points``.

        Returns:
            Dict[str, Any]: Crop parameters with keep-mask.
        """
        pts = None
        if isinstance(inpt, (list, tuple)):
            for item in inpt:
                if isinstance(item, Points):
                    pts = item
                    break

        crop_h, crop_w = self.crop_size

        crop_region = {
            "x0": self.x0,
            "y0": self.y0,
            "w": crop_w,
            "h": crop_h,
        }

        if pts is not None:
            pts_tensor = pts.to_tensor()
            keep_mask = (
                (pts_tensor[:, 0] >= self.x0)
                & (pts_tensor[:, 0] < self.x0 + crop_w)
                & (pts_tensor[:, 1] >= self.y0)
                & (pts_tensor[:, 1] < self.y0 + crop_h)
            )
            crop_region["keep_mask"] = keep_mask
        else:
            crop_region["keep_mask"] = None

        return crop_region

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """Crop supported inputs to the specified region.

        Args:
            inpt: Input to crop.
            params: Crop parameters with keep-mask.

        Returns:
            Any: Cropped input when supported.
        """
        x0, y0, w, h = params["x0"], params["y0"], params["w"], params["h"]
        keep_mask = params["keep_mask"]

        if isinstance(inpt, ImageSequence):
            seq = inpt.to_tensor()
            cropped = F.crop(seq, top=y0, left=x0, height=h, width=w)
            return ImageSequence(
                cropped,
                canvas_size=(w, h) + inpt.canvas_size[2:],
                fps=inpt.fps,
            )

        elif isinstance(inpt, Points):
            if keep_mask is None:
                return inpt
            d = inpt.to_tensor()[keep_mask].clone()
            if d.numel() == 0:
                return Points(d, canvas_size=(w, h) + inpt.canvas_size[2:])
            d[:, 0] -= x0
            d[:, 1] -= y0
            return Points(d, canvas_size=(w, h) + inpt.canvas_size[2:])

        elif isinstance(inpt, PointLabels):
            if keep_mask is None:
                return inpt
            d = inpt.to_tensor()[keep_mask]
            return PointLabels(d, label_map=inpt.name_to_id or inpt.id_to_name)

        elif isinstance(inpt, PointGroupIDs):
            if keep_mask is None:
                return inpt
            d = inpt.to_tensor()[keep_mask]
            return PointGroupIDs(d)

        return inpt


class RandomValidCrop(CompatTransform):
    """
    Randomly crop the image and points such that
    the cropped region contains at least one complete group of points.
    """

    _transformed_types = (ImageSequence, Points, PointLabels, PointGroupIDs)

    def __init__(
            self,
            crop_size: Tuple[int, int],
            least_target_num=3,
            max_tries: int = 50,
            margin: float = 1.0,
    ):
        """
        Args:
            crop_size: (target_h, target_w) in pixels.
            max_tries: maximum attempts to find valid crop.
            margin: extra margin around selected groups.
        """
        super().__init__()
        self.crop_size = crop_size
        self.least_target_num = least_target_num
        self.max_tries = max_tries
        self.margin = margin

    def make_params(self, inpt: Any) -> Dict[str, Any]:
        """Sample a crop region and build the point keep-mask.

        The keep-mask marks points that remain inside the sampled crop.

        Args:
            inpt: Transform input list containing `Points` and `PointGroupIDs`.

        Returns:
            Dict[str, Any]: Crop parameters and keep-mask.
        """
        pts = None
        if isinstance(inpt, (list, tuple)):
            for item in inpt:
                if isinstance(item, Points):
                    pts = item
        pts_tensor = pts.to_tensor()

        crop_region = self.get_valid_crop(inpt)
        keep_mask = (
                (pts_tensor[:, 0] >= crop_region['x0'])
                & (pts_tensor[:, 0] < crop_region['x0'] + crop_region['w'])
                & (pts_tensor[:, 1] >= crop_region['y0'])
                & (pts_tensor[:, 1] < crop_region['y0'] + crop_region['h'])
        )
        crop_region.update({'keep_mask': keep_mask})
        return crop_region

    def get_valid_crop(self, inpt: Any) -> Dict[str, Any]:
        """Find a valid crop region that fully contains at least one group.

        Args:
            inpt: Transform input list containing `Points` and `PointGroupIDs`.

        Returns:
            Dict[str, Any]: Crop parameters including `x0`, `y0`, `w`, `h`.
        """
        pts = None
        group_ids = None

        # Find Points and GroupIDs from transform inputs.
        if isinstance(inpt, (list, tuple)):
            for item in inpt:
                if isinstance(item, Points):
                    pts = item
                elif isinstance(item, PointGroupIDs):
                    group_ids = item

        if pts is None or group_ids is None:
            raise ValueError("RandomValidCrop requires both Points and GroupIDs in input.")

        w, h = pts.canvas_size[:2]
        crop_h, crop_w = self.crop_size

        d = pts.to_tensor()
        gids = group_ids.to_tensor()

        # Compute 2D envelopes for each group based on all its points.
        boxes = {}
        for gid in torch.unique(gids):
            mask = gids == gid
            x, y = d[mask, 0], d[mask, 1]
            if x.shape[-1] < self.least_target_num:
                continue
            boxes[int(gid.item())] = [
                x.min().item(),
                y.min().item(),
                x.max().item(),
                y.max().item(),
            ]

        valid_boxes = []
        for gid, (x_min, y_min, x_max, y_max) in boxes.items():
            box_w, box_h = x_max - x_min, y_max - y_min
            if box_w < crop_w and box_h < crop_h:
                valid_boxes.append((gid, x_min, y_min, x_max, y_max))

        if not valid_boxes:
            warnings.warn("No Valid Box for Random Crop!", UserWarning)
            x0 = max(0, (w - crop_w) // 2)
            y0 = max(0, (h - crop_h) // 2)
            return {"x0": x0, "y0": y0, "w": crop_w, "h": crop_h}

        for _ in range(self.max_tries):
            gid, x_min, y_min, x_max, y_max = random.choice(valid_boxes)
            # Extend candidate envelope with user-defined margin.
            x_min = max(0, x_min - self.margin)
            y_min = max(0, y_min - self.margin)
            x_max = min(w, x_max + self.margin)
            y_max = min(h, y_max + self.margin)

            min_x0 = max(0, x_max - crop_w)
            min_y0 = max(0, y_max - crop_h)
            max_x0 = min(x_min, w - crop_w)
            max_y0 = min(y_min, h - crop_h)

            if max_x0 >= min_x0 and max_y0 >= min_y0:
                x0 = random.uniform(min_x0, max_x0)
                y0 = random.uniform(min_y0, max_y0)
                x0 = int(max(0, min(w - crop_w, x0)))
                y0 = int(max(0, min(h - crop_h, y0)))
                return {"x0": x0, "y0": y0, "w": crop_w, "h": crop_h}

        # Fallback when no random crop could satisfy constraints.
        return {"x0": 0, "y0": 0, "w": crop_w, "h": crop_h}

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """Crop supported targets and filter annotations with keep-mask.

        Args:
            inpt: Input to crop.
            params: Crop parameters with keep-mask.

        Returns:
            Any: Cropped input when supported.
        """
        x0, y0, w, h = params["x0"], params["y0"], params["w"], params["h"]
        keep_mask = params["keep_mask"]

        if isinstance(inpt, ImageSequence):
            seq = inpt.to_tensor()
            cropped = F.crop(seq, top=y0, left=x0, height=h, width=w)
            return ImageSequence(
                cropped,
                canvas_size=(w, h) + inpt.canvas_size[2:],
                fps=inpt.fps,
            )

        elif isinstance(inpt, Points):
            d = inpt.to_tensor()[keep_mask].clone()
            if d.numel() == 0:  # Return an empty yet well-formed point tensor.
                return Points(d, canvas_size=(w, h) + inpt.canvas_size[2:])
            d[:, 0] -= x0
            d[:, 1] -= y0
            return Points(d, canvas_size=(w, h) + inpt.canvas_size[2:])

        elif isinstance(inpt, PointLabels):
            d = inpt.to_tensor()[keep_mask]
            return PointLabels(d, label_map=inpt.name_to_id or inpt.id_to_name)

        elif isinstance(inpt, PointGroupIDs):
            d = inpt.to_tensor()[keep_mask]
            return PointGroupIDs(d)

        return inpt


class RemoveSingletonGroups(CompatTransform):
    """
    Remove samples (points/labels/groupids) whose GroupID has too few visible points.

    Example:
        GroupIDs = [9, 9, 9, 9, 11]
        --> group 11 appears once → removed
        --> keep_mask = [True, True, True, True, False]
    """

    _transformed_types = (Points, PointLabels, PointGroupIDs)

    def __init__(self, min_group_points: int = 1):
        """Initialize the minimum number of visible points required per group."""

        super().__init__()
        self.min_group_points = max(int(min_group_points), 1)

    def make_params(self, inpt: Any) -> Dict[str, Any]:
        """Build a keep-mask that removes under-supported group IDs.

        Args:
            inpt: Transform input list containing `PointGroupIDs`.

        Returns:
            Dict[str, Any]: Keep-mask for valid group IDs.
        """
        group_ids = None
        if isinstance(inpt, (list, tuple)):
            for item in inpt:
                if isinstance(item, PointGroupIDs):
                    group_ids = item
                    break

        if group_ids is None:
            raise ValueError("RemoveSingletonGroups requires GroupIDs in input.")

        gids = group_ids.to_tensor()
        unique_ids, counts = torch.unique(gids, return_counts=True)
        valid_ids = unique_ids[counts >= self.min_group_points]

        keep_mask = torch.isin(gids, valid_ids)
        return {"keep_mask": keep_mask}

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """Apply keep-mask to points, labels, and group IDs.

        Args:
            inpt: Input to filter.
            params: Keep-mask parameters.

        Returns:
            Any: Filtered input when supported.
        """
        keep_mask = params["keep_mask"]

        if isinstance(inpt, Points):
            d = inpt.to_tensor()[keep_mask]
            return Points(d, canvas_size=inpt.canvas_size)

        elif isinstance(inpt, PointLabels):
            d = inpt.to_tensor()[keep_mask]
            return PointLabels(d, label_map=inpt.name_to_id or inpt.id_to_name)

        elif isinstance(inpt, PointGroupIDs):
            d = inpt.to_tensor()[keep_mask]
            return PointGroupIDs(d)

        return inpt


class GenerateBoundingBoxes3D(CompatTransform):
    """
    Compute 3D bounding boxes for each unique group_id.

    For each group_id, computes the (cx, cy, cz, w, h, d) box enclosing
    all points belonging to that group across time (z = frame index).

    Updates:
        - target["bboxes"]
        - target["bboxes_labels"]
        - target["bboxes_group_ids"]

    Example:
        points: (x, y, t)
        group_ids: [1, 1, 2, 2, 2]
        labels: [cat, cat, dog, dog, dog]
        --> boxes per group_id: one for 1 (cat), one for 2 (dog)

    Notes:
        - Applies configurable spatial/temporal padding before box creation.
        - Enforces minimal box size on XY and T axes.
        - Clamps intervals to valid canvas bounds.
    """

    _transformed_types = (BoundingBoxes3D, Points, PointLabels, PointGroupIDs, BoxLabels, BoxGroupIDs)

    def __init__(
            self,
            spatial_padding=2.0,
            temporal_padding=0.5,
            min_size_xy=4.0,
            min_size_t=1.0,
    ):
        """Initialize padding and size constraints for generated 3D boxes.

        Args:
            spatial_padding: Symmetric padding added to x/y intervals.
            temporal_padding: Symmetric padding added to t interval.
            min_size_xy: Minimum width/height after expansion and clamping.
            min_size_t: Minimum depth (time span) after expansion and clamping.
        """
        super().__init__()
        self.spatial_padding = float(spatial_padding)
        self.temporal_padding = float(temporal_padding)
        self.min_size_xy = float(min_size_xy)
        self.min_size_t = float(min_size_t)

    @staticmethod
    def _expand_interval(v_min, v_max, padding, min_size, lower, upper):
        """Expand and regularize a 1D interval inside [lower, upper].

        Steps:
            1) Expand [v_min, v_max] with symmetric padding.
            2) Enforce minimum size around the interval center.
            3) Shift/clamp interval to remain within [lower, upper].
            4) If clamping shrinks size below min_size, recover size when possible.

        Args:
            v_min: Minimum coordinate value.
            v_max: Maximum coordinate value.
            padding: Symmetric padding to apply.
            min_size: Minimum interval size.
            lower: Lower bound for clamping.
            upper: Upper bound for clamping.

        Returns:
            Tuple[float, float]: Expanded `(min, max)` interval.
        """
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
                extra = min(min_size - cur_size, v_min - lower)
                v_min -= extra
            cur_size = v_max - v_min
            if cur_size < min_size and right_room > add_right:
                extra = min(min_size - cur_size, upper - v_max)
                v_max += extra

        return v_min, v_max

    def make_params(self, inpt: Any) -> Dict[str, Any]:
        """Generate one 3D box per group ID and aligned box metadata.

        For each group, this method computes min/max over (x, y, t), expands
        each axis interval with padding and minimum-size constraints, then
        converts to (cx, cy, cz, w, h, d) format.

        Args:
            inpt: Transform input list containing points, labels, and group IDs.

        Returns:
            Dict[str, Any]: Generated boxes, labels, and group IDs.
        """
        points = None
        labels = None
        group_ids = None

        # Find Points / Labels / GroupIDs in transform inputs.
        if isinstance(inpt, (list, tuple)):
            for item in inpt:
                if isinstance(item, Points):
                    points = item
                elif isinstance(item, PointLabels):
                    labels = item
                elif isinstance(item, PointGroupIDs):
                    group_ids = item

        if points is None or group_ids is None or labels is None:
            raise ValueError("GenerateBoundingBoxes3D requires Points, PointLabels, and PointGroupIDs in input.")

        pts = points.to_tensor()  # shape (N, 3): [x, y, t]
        gids = group_ids.to_tensor()
        lbls = labels.to_tensor()
        canvas_w, canvas_h, canvas_t = points.canvas_size

        unique_gids = torch.unique(gids)
        boxes = []
        box_labels = []
        box_group_ids = []

        for gid in unique_gids.tolist():
            mask = (gids == gid)
            pts_grp = pts[mask]
            lbl_grp = lbls[mask]

            if pts_grp.numel() == 0:
                continue

            x_min, y_min, t_min = pts_grp.min(dim=0).values
            x_max, y_max, t_max = pts_grp.max(dim=0).values

            # Expand and clamp each axis interval with configurable constraints.
            x_min, x_max = self._expand_interval(
                x_min, x_max, self.spatial_padding, self.min_size_xy, 0.0, max(canvas_w - 1, 0.0)
            )
            y_min, y_max = self._expand_interval(
                y_min, y_max, self.spatial_padding, self.min_size_xy, 0.0, max(canvas_h - 1, 0.0)
            )
            t_min, t_max = self._expand_interval(
                t_min, t_max, self.temporal_padding, self.min_size_t, 0.0, max(canvas_t - 1, 0.0)
            )

            cx = (x_max + x_min) / 2
            cy = (y_max + y_min) / 2
            cz = (t_max + t_min) / 2
            w = x_max - x_min
            h = y_max - y_min
            d = t_max - t_min

            boxes.append(torch.tensor([cx, cy, cz, w, h, d], dtype=torch.float32))

            # Assign the dominant label of this track/group to the box.
            most_common_label = Counter(lbl_grp.tolist()).most_common(1)[0][0]
            box_labels.append(most_common_label)
            box_group_ids.append(gid)

        if len(boxes) == 0:
            bboxes = BoundingBoxes3D.empty(format="cxcyczwhd", canvas_size=points.canvas_size)
            bboxes_labels = BoxLabels.empty(label_map=labels.name_to_id or labels.id_to_name)
            bboxes_group_ids = BoxGroupIDs.empty(id_type="int")
        else:
            bboxes = BoundingBoxes3D(torch.stack(boxes), format="cxcyczwhd", canvas_size=points.canvas_size)
            bboxes_labels = BoxLabels(torch.tensor(box_labels, dtype=torch.int64),
                                      label_map=labels.name_to_id or labels.id_to_name)
            bboxes_group_ids = BoxGroupIDs(torch.tensor(box_group_ids, dtype=torch.int64))

        return {
            "bboxes": bboxes,
            "bboxes_labels": bboxes_labels,
            "bboxes_group_ids": bboxes_group_ids,
        }

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """Route generated box tensors to their corresponding target fields.

        Args:
            inpt: Input wrapper to transform.
            params: Generated bounding box metadata.

        Returns:
            Any: Updated input wrapper when supported.
        """
        if isinstance(inpt, BoundingBoxes3D):
            return params["bboxes"]
        elif isinstance(inpt, BoxLabels):
            return params["bboxes_labels"]
        elif isinstance(inpt, BoxGroupIDs):
            return params["bboxes_group_ids"]
        return inpt


class Normalize(CompatTransform):
    """
    Normalize ImageSequence, Points, and BoundingBoxes3D based on canvas_size.
    Compatible with torchvision v2 transforms.
    """

    _transformed_types = (ImageSequence, BoundingBoxes3D, Points)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """Normalize supported tensor wrappers according to their semantics.

        Args:
            inpt: Input wrapper to normalize.
            params: Unused parameter dictionary.

        Returns:
            Any: Normalized input wrapper when supported.
        """
        if isinstance(inpt, ImageSequence):
            return self._normalize_image_seq(inpt)
        elif isinstance(inpt, Points):
            return self._normalize_points(inpt)
        elif isinstance(inpt, BoundingBoxes3D):
            return self._normalize_boxes3d(inpt)
        return inpt

    @staticmethod
    def _normalize_image_seq(img_seq):
        """Normalize ImageSequence tensor values to [0, 1].

        Args:
            img_seq: Input image sequence.

        Returns:
            ImageSequence: Normalized image sequence.
        """
        tensor = img_seq.to_tensor().float()
        if tensor.max() > 1:
            tensor = tensor / 255.0
        return ImageSequence(
            tensor,
            canvas_size=img_seq.canvas_size,
            fps=img_seq.fps
        )

    @staticmethod
    def _normalize_points(pts):
        """Normalize Points coordinates using canvas_size.

        Points are expected in `xyt` form.

        Args:
            pts: Input points.

        Returns:
            Points: Normalized points.
        """
        data = pts.to_tensor().clone()
        if pts.canvas_size is None:
            return pts  # Skip normalization when geometry metadata is absent.

        w, h, t = pts.canvas_size
        data[:, 0] /= w
        data[:, 1] /= h
        data[:, 2] /= t

        return Points(data, canvas_size=pts.canvas_size)

    @staticmethod
    def _normalize_boxes3d(boxes):
        """Normalize 3D bounding boxes using canvas_size.

        Args:
            boxes: Input 3D boxes.

        Returns:
            BoundingBoxes3D: Normalized boxes.
        """
        data = boxes.to_tensor().clone()
        if boxes.canvas_size is None:
            return boxes
        w, h, d = boxes.canvas_size

        data[:, 0] /= w
        data[:, 1] /= h
        data[:, 2] /= d
        data[:, 3] /= w
        data[:, 4] /= h
        data[:, 5] /= d

        return boxes.__class__(data, format=boxes.format, canvas_size=boxes.canvas_size)

    def __repr__(self):
        """Return a string representation of normalization settings."""
        return f"{self.__class__.__name__}()"
