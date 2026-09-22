"""Custom tensor wrappers for spatiotemporal point-centric detection.

This module defines `torchvision.tv_tensors.TVTensor` subclasses used by AstroDim-style datasets and transforms. The wrappers keep metadata such as canvas size and label/group mappings alongside raw tensors.
"""

import torch
from typing import Optional, Tuple, Union, List, Literal
import numpy as np
from PIL import Image

from torchvision import tv_tensors

__all__ = [
    'ImageSequence',
    'Points',
    'Labels',
    'GroupIDs',
    'BoundingBoxes3D',
    'BoxLabels',
    'PointLabels',
    'BoxGroupIDs',
    'PointGroupIDs'
]

class ImageSequence(tv_tensors.TVTensor):
    """
    A tensor-like wrapper for a sequence of images.
    Compatible with torchvision v2 transforms.
    """

    _repr_attrs = ("num_frames", "canvas_size", "fps")

    def __new__(
            cls,
            data: Union[torch.Tensor, List[Image.Image]],
            *,
            canvas_size: Optional[Tuple[int, int, int]] = None,
            fps: Optional[float] = None,
    ):
        """Construct an `ImageSequence` tensor.

        Args:
            data: Frame data as a tensor (typically `(C, T, H, W)`) or a list
                of PIL images.
            canvas_size: Optional `(width, height, num_frames)` metadata.
            fps: Optional frame rate metadata.
        """
        if isinstance(data, list):
            # Convert list of PIL images to a single channel-first tensor.
            tensors = []
            for im in data:
                arr = np.array(im)
                if arr.ndim == 2:
                    arr = np.expand_dims(arr, axis=-1)
                tensor = torch.as_tensor(arr).permute(2, 0, 1)  # (C, H, W)
                tensors.append(tensor)
            data = torch.stack(tensors, dim=1)  # (C, T, H, W)
        elif not torch.is_tensor(data):
            data = torch.as_tensor(data)

        return super().__new__(cls, data)

    def __init__(
            self,
            data: Union[torch.Tensor, List[Image.Image]],
            *,
            canvas_size: Optional[Tuple[int, int, int]] = None,
            fps: Optional[float] = None,
    ):
        """Attach metadata to an `ImageSequence` instance.

        Args:
            data: Same input accepted by `__new__`.
            canvas_size: Optional `(width, height, num_frames)` metadata. When omitted, it is inferred from tensor shape.
            fps: Optional frame rate.
        """
        super().__init__()
        _, t, h, w = self.shape[-4:]
        if canvas_size is None:
            canvas_size = (w, h, t)
        self.canvas_size = canvas_size
        self.fps = fps

    @classmethod
    def empty(cls, *, canvas_size: Optional[Tuple[int, int]] = None, fps: Optional[float] = None):
        """Create an empty ImageSequence.

        Args:
            canvas_size: Optional `(width, height, num_frames)` metadata.
            fps: Optional frame rate metadata.

        Returns:
            ImageSequence: Empty sequence tensor.
        """
        empty_tensor = torch.empty((3, 0, 0, 0))
        return cls(empty_tensor, canvas_size=canvas_size, fps=fps)

    def clone(self, *, memory_format: torch.memory_format | None = None):
        """Return a cloned `ImageSequence` with preserved metadata.

        Args:
            memory_format: Optional memory format for the clone.

        Returns:
            ImageSequence: Cloned sequence tensor.
        """
        return ImageSequence(
            self.as_subclass(torch.Tensor).clone(),
            canvas_size=self.canvas_size,
            fps=self.fps,
        )

    def to_tensor(self) -> torch.Tensor:
        """Return the underlying plain `torch.Tensor` view.

        Returns:
            torch.Tensor: Raw tensor data.
        """
        return self.as_subclass(torch.Tensor)

    def __getitem__(self, idx):
        """Index sequence data while preserving wrapper behavior.

        Integer indexing returns a one-frame `ImageSequence`. Other indexing
        modes delegate to `TVTensor` behavior.

        Args:
            idx: Index or slice to apply.

        Returns:
            ImageSequence | torch.Tensor: Indexed sequence data.
        """
        if isinstance(idx, int):
            frame = self[:, idx:idx + 1, :, :]
            return ImageSequence(
                frame.clone(),
                canvas_size=self.canvas_size,
                fps=self.fps,
            )
        return super().__getitem__(idx)

    def __repr__(self):
        """Return a compact debug string with shape and metadata."""
        return (
            f"ImageSequence(shape={tuple(self.shape)}, "
            f"fps={self.fps}, canvas_size={self.canvas_size})"
        )

    def to(self, device: torch.device, dtype: torch.dtype = None, ** kwargs):
        """Move/cast data and keep `ImageSequence` metadata unchanged.

        Args:
            device: Target device.
            dtype: Optional target dtype.
            **kwargs: Additional arguments forwarded to `Tensor.to`.

        Returns:
            ImageSequence: Converted sequence tensor.
        """
        data = self.as_subclass(torch.Tensor).to(device, dtype=dtype, **kwargs)
        return ImageSequence(
            data,
            canvas_size=self.canvas_size,
            fps=self.fps
        )


class Points(tv_tensors.TVTensor):
    """
    A lightweight wrapper for point coordinates in image (or sequence) space.
    Compatible with torchvision v2 transforms.
    """

    _repr_attrs = ("canvas_size",)

    def __new__(
            cls,
            data,
            *,
            canvas_size: Optional[Tuple[int, int, int]] = None,
    ):
        """Construct a spatiotemporal point tensor wrapper.

        Args:
            data: Point coordinates with shape `(N, 3)` as `(x, y, t)`.
            canvas_size: Optional `(width, height, num_frames)` metadata.
        """
        if not torch.is_tensor(data):
            data = torch.as_tensor(data, dtype=torch.float32)

        if data.ndim != 2 or data.shape[-1] != 3:
            raise ValueError("Points expects shape (N, 3) with coordinates in (x, y, t) format.")

        return super().__new__(cls, data)

    def __init__(self, data, *, canvas_size: Optional[Tuple[int, int, int]] = None):
        """Attach canvas metadata to the point tensor."""
        super().__init__()
        self.canvas_size = canvas_size

    @classmethod
    def empty(cls, *, canvas_size: Optional[Tuple[int, int, int]] = None):
        """Create an empty `Points` container with `(x, y, t)` layout.

        Args:
            canvas_size: Optional `(width, height, num_frames)` metadata.

        Returns:
            Points: Empty points container.
        """
        return cls(torch.empty((0, 3)), canvas_size=canvas_size)

    def clone(self, *, memory_format: torch.memory_format | None = None):
        """Return a cloned `Points` object with metadata preserved.

        Args:
            memory_format: Optional memory format for the clone.

        Returns:
            Points: Cloned points object.
        """
        return Points(
            self.as_subclass(torch.Tensor).clone(memory_format=memory_format),
            canvas_size=self.canvas_size
        )

    def to_tensor(self):
        """Expose the raw tensor values for downstream operations.

        Returns:
            torch.Tensor: Raw point tensor.
        """
        return self.as_subclass(torch.Tensor)

    def __repr__(self):
        """Return a concise text representation for debugging."""
        s = f"Points(shape={tuple(self.shape)}, canvas_size={self.canvas_size})"
        return s

    def to(self, device: torch.device, dtype: torch.dtype = None, ** kwargs):
        """Move/cast points and keep coordinate metadata intact.

        Args:
            device: Target device.
            dtype: Optional target dtype.
            **kwargs: Additional arguments forwarded to `Tensor.to`.

        Returns:
            Points: Converted points object.
        """
        data = self.as_subclass(torch.Tensor).to(device, dtype=dtype, **kwargs)
        return Points(
            data,
            canvas_size=self.canvas_size,
        )


class Labels(tv_tensors.TVTensor):
    """
    Wrapper for class labels of detected points or objects.
    Supports bidirectional mapping between label IDs and names.
    Compatible with torchvision v2 transforms.
    """

    _repr_attrs = ("num_labels", "num_classes")

    def __new__(cls, data, *, label_map: Optional[dict] = None):
        """Construct a label tensor with integer dtype normalization."""
        if not torch.is_tensor(data):
            data = torch.as_tensor(data, dtype=torch.int64)
        elif data.dtype != torch.int64:  # Ensure tensor is int64
            data = data.to(torch.int64)
        return super().__new__(cls, data)

    def __init__(self, data, *, label_map: Optional[dict] = None):
        """Attach metadata for label decoding/encoding.

        Args:
            data: Label IDs.
            label_map: Either `{name: id}` or `{id: name}` mapping.
        """
        super().__init__()
        self.num_labels = int(self.numel())

        # Handle label_map: can be {name: id} or {id: name}
        self.name_to_id, self.id_to_name = self._process_label_map(label_map)
        self.num_classes = len(self.name_to_id) if self.name_to_id else None

    @classmethod
    def empty(cls, *, label_map: Optional[dict] = None):
        """Create an empty `Labels` object with optional class mapping."""
        return cls(torch.empty((0,), dtype=torch.int64), label_map=label_map)

    @staticmethod
    def _process_label_map(label_map):
        """Normalize class mapping into both `name->id` and `id->name` forms.

        Args:
            label_map: Label mapping dictionary.

        Returns:
            Tuple[dict, dict]: `(name_to_id, id_to_name)` mappings.
        """
        if label_map is None:
            return {}, {}
        # Detect direction
        if all(isinstance(k, str) and isinstance(v, int) for k, v in label_map.items()):
            name_to_id = label_map
            id_to_name = {v: k for k, v in label_map.items()}
        elif all(isinstance(k, int) and isinstance(v, str) for k, v in label_map.items()):
            id_to_name = label_map
            name_to_id = {v: k for k, v in label_map.items()}
        else:
            raise ValueError(
                "label_map must be either {name: id} or {id: name}"
            )
        return name_to_id, id_to_name

    def clone(self, *, memory_format: torch.memory_format | None = None):
        """Return a cloned `Labels` object with the same mapping metadata.

        Args:
            memory_format: Optional memory format for the clone.

        Returns:
            Labels: Cloned labels object.
        """
        return Labels(
            self.as_subclass(torch.Tensor).clone(memory_format=memory_format),
            label_map=self.name_to_id or self.id_to_name
        )

    def to_tensor(self):
        """Return labels as a plain integer tensor.

        Returns:
            torch.Tensor: Raw label tensor.
        """
        return self.as_subclass(torch.Tensor)

    def decode(self):
        """Decode IDs to human-readable names when mapping is available.

        Returns:
            List[str]: Decoded label names.
        """
        if not self.id_to_name:
            return [int(x.item()) for x in self.flatten()]
        return [self.id_to_name.get(int(x.item()), f"unk_{x.item()}") for x in self.flatten()]

    def encode(self, names: list[str]):
        """Encode class names into a new `Labels` instance of integer IDs.

        Args:
            names: List of class names.

        Returns:
            Labels: Encoded labels instance.
        """
        if not self.name_to_id:
            raise ValueError("Cannot encode names: label_map not provided.")
        ids = [self.name_to_id[n] for n in names]
        return Labels(ids, label_map=self.name_to_id)

    def __repr__(self):
        """Return summary text including class-map availability."""
        class_info = f"num_classes={self.num_classes}" if self.num_classes else "num_classes=?"
        mapping_info = (
            f", label_map(keys)={list(self.name_to_id.keys())[:3]}..."
            if self.name_to_id else ""
        )
        return f"Labels(shape={tuple(self.shape)}, {class_info}{mapping_info})"


class GroupIDs(tv_tensors.TVTensor):
    """
    Wrapper for object or track group identifiers.
    Supports either integer or string-based IDs.
    Compatible with torchvision v2 transforms.
    """

    _repr_attrs = ("num_ids", "id_type")

    def __new__(cls, data):
        """Construct a group-ID tensor for tracking identities.

        String IDs are stored as byte arrays to stay tensor-compatible.
        """
        # Convert to tensor of strings or ints
        if isinstance(data, (list, tuple)) and isinstance(data[0], str):
            # Encode strings as bytes for tensor storage
            data = np.array(data, dtype='S')
            data = torch.from_numpy(data)
            id_type = "str"
        else:
            data = torch.as_tensor(data)
            id_type = "int"
        obj = super().__new__(cls, data)
        obj.id_type = id_type
        return obj

    def __init__(self, data):
        """Attach metadata describing ID count and representation type."""
        super().__init__()
        self.num_ids = int(self.shape[0]) if self.ndim > 0 else 1

    def clone(self, *, memory_format: torch.memory_format | None = None):
        """Return a cloned `GroupIDs` object."""
        return GroupIDs(
            self.as_subclass(torch.Tensor).clone(memory_format=memory_format)
        )

    def to_tensor(self):
        """Return group IDs as a plain tensor."""
        return self.as_subclass(torch.Tensor)

    @classmethod
    def empty(cls, id_type: Literal["int", "str"] = "int"):
        """Create an empty `GroupIDs` container for the chosen ID type.

        Args:
            id_type: Identifier storage type (`int` or `str`).

        Returns:
            GroupIDs: Empty group ID container.
        """
        if id_type == "str":
            data = torch.from_numpy(np.array([], dtype='S'))
        else:
            data = torch.empty((0,), dtype=torch.int64)
        return cls(data)

    def __repr__(self):
        """Return a concise string summary of group-ID metadata."""
        s = f"GroupIDs(shape={tuple(self.shape)}, type={self.id_type})"
        return s


class BoundingBoxes3D(tv_tensors.TVTensor):
    """
    A lightweight wrapper for 3D bounding boxes.

    Supports two formats:
        - "cx cy cz w h d": (center_x, center_y, center_z, width, height, depth)
        - "x y z x y z": (x1, y1, z1, x2, y2, z2)

    Compatible with torchvision v2 transforms.
    """

    _repr_attrs = ("format", "canvas_size")

    def __new__(
            cls,
            data,
            *,
            format: Literal["cxcyczwhd", "xyzxyz"] = "xyzxyz",
            canvas_size: Optional[Tuple[int, int, int]] = None,
    ):
        """Construct a 3D box tensor wrapper.

        Args:
            data: Tensor-like object of shape `(N, 6)`.
            format: Box encoding convention.
            canvas_size: Optional `(width, height, depth)` metadata.
        """
        # data shape: (N, 6)
        if not torch.is_tensor(data):
            data = torch.as_tensor(data, dtype=torch.float32)
        return super().__new__(cls, data)

    def __init__(
            self,
            data,
            *,
            format: Literal["cxcyczwhd", "xyzxyz"] = "xyzxyz",
            canvas_size: Optional[Tuple[int, int, int]] = None,
    ):
        """Attach box format and canvas metadata."""
        super().__init__()
        self.format = format
        self.canvas_size = canvas_size
        self.num_boxes = int(self.shape[0]) if self.ndim > 0 else 0

    @classmethod
    def empty(
            cls,
            *,
            format: Literal["cxcyczwhd", "xyzxyz"] = "xyzxyz",
            canvas_size: Optional[Tuple[int, int, int]] = None,
    ):
        """Create an empty `BoundingBoxes3D` container.

        Args:
            format: Bounding box format.
            canvas_size: Optional `(width, height, depth)` metadata.

        Returns:
            BoundingBoxes3D: Empty box container.
        """
        return cls(torch.empty((0, 6), dtype=torch.float32), format=format, canvas_size=canvas_size)

    def clone(self, *, memory_format: torch.memory_format | None = None):
        """Return a cloned 3D box container with preserved metadata.

        Args:
            memory_format: Optional memory format for the clone.

        Returns:
            BoundingBoxes3D: Cloned box container.
        """
        return BoundingBoxes3D(
            self.as_subclass(torch.Tensor).clone(memory_format=memory_format),
            format=self.format,
            canvas_size=self.canvas_size,
        )

    def to_tensor(self):
        """Return the underlying `(N, 6)` tensor.

        Returns:
            torch.Tensor: Raw box tensor.
        """
        return self.as_subclass(torch.Tensor)

    def __repr__(self):
        """Return debug text with shape, format, and canvas metadata."""
        return (
            f"BoundingBoxes3D(shape={tuple(self.shape)}, "
            f"format={self.format}, canvas_size={self.canvas_size})"
        )

    def to(self, device: torch.device, dtype: torch.dtype = None, ** kwargs):
        """Move/cast box tensor and preserve metadata.

        Args:
            device: Target device.
            dtype: Optional target dtype.
            **kwargs: Additional arguments forwarded to `Tensor.to`.

        Returns:
            BoundingBoxes3D: Converted box container.
        """
        data = self.as_subclass(torch.Tensor).to(device, dtype=dtype, **kwargs)
        return BoundingBoxes3D(
            data,
            format=self.format,
            canvas_size=self.canvas_size,
        )

class BoxLabels(Labels):
    """Labels corresponding to detected bounding boxes."""

    def __repr__(self):
        """Return representation renamed from `Labels` to `BoxLabels`."""
        base = super().__repr__().replace("Labels", "BoxLabels")
        return base


class PointLabels(Labels):
    """Labels corresponding to detected keypoints or points."""

    def __repr__(self):
        """Return representation renamed from `Labels` to `PointLabels`."""
        base = super().__repr__().replace("Labels", "PointLabels")
        return base


class BoxGroupIDs(GroupIDs):
    """Group or track IDs corresponding to bounding boxes."""

    def __repr__(self):
        """Return representation renamed from `GroupIDs` to `BoxGroupIDs`."""
        base = super().__repr__().replace("GroupIDs", "BoxGroupIDs")
        return base


class PointGroupIDs(GroupIDs):
    """Group or track IDs corresponding to points."""

    def __repr__(self):
        """Return representation renamed from `GroupIDs` to `PointGroupIDs`."""
        base = super().__repr__().replace("GroupIDs", "PointGroupIDs")
        return base
