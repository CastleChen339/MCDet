"""Dataset loaders for AstroDim dim-moving-target sequences.

The dataset is organized as multiple subfolders, each containing paired JSON
annotations and PNG frames. Samples are generated as fixed-length frame
sequences with point-level labels and group IDs, then converted to custom
`TVTensor` wrappers for transform compatibility.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from collections import Counter

from torch.utils.data import Dataset, ConcatDataset
try:
    from torchvision.io import ImageReadMode, read_image
except Exception:  # pragma: no cover - torchvision io is optional at runtime
    ImageReadMode = None
    read_image = None

try:
    from ._tensors import *
except ImportError:
    from datasets._tensors import *

LABEL_DICT = {"debris": 0}

class AstroDimDataset:
    """A concatenated dataset over all valid AstroDim sub-datasets.

    This class scans each child directory under `main_folder`, tries to build
    an `AstroDimSubDataset`, and concatenates all successfully loaded subsets.
    """

    def __init__(
        self,
        main_folder,
        seq_len,
        transforms,
        image_channels=3,
        use_torchvision_io=True,
        min_group_points=1,
    ):
        """Initialize and aggregate all available sub-datasets.

        Args:
            main_folder: Root directory containing multiple scene folders.
            seq_len: Number of consecutive frames per sample sequence.
            transforms: Optional callable transform pipeline applied in
                `AstroDimSubDataset.__getitem__`.
            image_channels: Number of channels to read from disk (1 or 3).
            use_torchvision_io: Whether to prefer `torchvision.io.read_image`
                for PNG decoding when available.
            min_group_points: Minimum number of points required for at least
                one group in a valid sequence window.
        """
        if image_channels not in (1, 3):
            raise ValueError("image_channels must be 1 or 3")

        self.datasets = []
        for dataset_folder in sorted(os.listdir(main_folder)):
            full_path = os.path.join(main_folder, dataset_folder)
            if os.path.isdir(full_path):
                try:
                    dataset = AstroDimSubDataset(
                        full_path,
                        seq_len,
                        transforms,
                        image_channels=image_channels,
                        use_torchvision_io=use_torchvision_io,
                        min_group_points=min_group_points,
                    )
                    self.datasets.append(dataset)
                except Exception as e:
                    print(f"Failed to load dataset {dataset_folder}: {e}")

        self.combined_dataset = ConcatDataset(self.datasets)

    def __len__(self):
        """Return the total number of samples across all sub-datasets.

        Returns:
            int: Total number of samples.
        """
        return len(self.combined_dataset)

    def __getitem__(self, idx):
        """Fetch one sample from the concatenated dataset.

        Args:
            idx: Sample index.

        Returns:
            Tuple[ImageSequence, Dict[str, Any]]: Image sequence and target dictionary.
        """
        return self.combined_dataset[idx]


class AstroDimSubDataset(Dataset):
    """Sequence dataset for one AstroDim scene folder.

    Expected folder layout:
        - `images/*.png`
        - `json/*.json`

    Each sample contains a fixed-length image sequence and point-level target
    metadata. Sequences are filtered to keep only windows with at least one
    group appearing at least three times.
    """

    def __init__(
        self,
        data_folder,
        seq_len,
        transforms,
        image_channels=3,
        use_torchvision_io=True,
        min_group_points=1,
    ):
        """Initialize one scene-level dataset.

        Args:
            data_folder: Path to one dataset folder containing `images` and
                `json` subfolders.
            seq_len: Number of consecutive frames per training sample.
            transforms: Optional callable transform pipeline.
            image_channels: Number of channels to read from disk (1 or 3).
            use_torchvision_io: Enable torchvision backend for image decoding.
                Falls back to PIL when torchvision IO is unavailable.
            min_group_points: Minimum number of points required for at least
                one group in a valid sequence window.
        """
        super().__init__()
        if image_channels not in (1, 3):
            raise ValueError("image_channels must be 1 or 3")

        self.dataset_folder = data_folder
        self.seq_len = seq_len
        self.label_dict = LABEL_DICT
        self._transforms = transforms
        self.image_channels = image_channels
        self.use_torchvision_io = bool(use_torchvision_io and read_image is not None)
        self.min_group_points = max(int(min_group_points), 1)

        self.json_folder = os.path.join(self.dataset_folder, 'json')
        self.png_folder = os.path.join(self.dataset_folder, 'images')
        self.json_files = [
            os.path.join(self.json_folder, f)
            for f in sorted(os.listdir(self.json_folder))
            if f.endswith(".json")
        ]
        self.png_files = [
            os.path.join(self.png_folder, f)
            for f in sorted(os.listdir(self.png_folder))
            if f.endswith(".png")
        ]
        assert len(self.json_files) == len(self.png_files), "Mismatch between JSON and PNG files"

        self.frame_annotations = self._load_all_annotations()
        self.valid_items_list = []
        self._filter_valid_item()

    def __len__(self):
        """Return the number of valid sequence windows.

        Returns:
            int: Number of valid windows.
        """
        return len(self.valid_items_list)

    def __getitem__(self, idx):
        """Load one sample and optionally apply transforms.

        Args:
            idx: Sample index within the sub-dataset.

        Returns:
            Tuple `(img_seq, target)` where `img_seq` is an `ImageSequence`
            and `target` is a dictionary of point/box annotations.
        """
        img_seq, target = self.load_item(idx)
        if self._transforms is not None:
            img_seq, target, _ = self._transforms(img_seq, target, self)
        return img_seq, target

    def _filter_valid_item(self):
        """Build the list of valid sequence windows from raw annotations.

        A window is considered valid if at least one `group_id` appears three
        times or more across the `seq_len` frames.
        """
        for idx in range(len(self.json_files) - self.seq_len + 1):
            img_files_seq = self.png_files[idx:idx + self.seq_len]
            target_dict = {"points": [], "labels": [], "group_ids": [], "img_seq": []}

            for i in range(self.seq_len):
                data = self.frame_annotations[idx + i]
                if "img_w" not in target_dict and "img_h" not in target_dict:
                    target_dict["img_w"] = data.get('imageWidth', 0)
                    target_dict["img_h"] = data.get('imageHeight', 0)
                else:
                    if target_dict["img_w"] != data.get('imageWidth', 0) or target_dict["img_h"] != data.get(
                            'imageHeight', 0):
                        raise ValueError(
                            "Inconsistent image dimensions: existing 'img_w' or 'img_h' in target_dict does not match the values provided in data")

                for shape in data.get('shapes', []):
                    label = shape.get("label", "")
                    gid = shape.get("group_id", "")
                    if label not in self.label_dict:
                        continue
                    xy = shape["points"][0]
                    target_dict["points"].append(xy)
                    target_dict["labels"].append(self.label_dict[label])
                    target_dict["group_ids"].append(gid)
                    target_dict["img_seq"].append(i)

            if len(target_dict["group_ids"]) > 0:
                max_count = max(Counter(target_dict["group_ids"]).values())
                if max_count >= self.min_group_points:
                    self.valid_items_list.append((tuple(img_files_seq), target_dict))

    def _load_all_annotations(self):
        """Preload per-frame JSON annotations for fast window assembly.

        Returns:
            List[dict]: Raw JSON annotation dictionaries per frame.
        """
        annotations = []
        for json_path in self.json_files:
            with open(json_path, 'r', encoding='utf-8') as f:
                annotations.append(json.load(f))
        return annotations

    def _read_image(self, image_path):
        """Read one image as CHW uint8 tensor.

        Uses torchvision IO when enabled and available; otherwise falls back
        to PIL decoding.

        Args:
            image_path: Path to the image file.

        Returns:
            torch.Tensor: Image tensor with shape `(C, H, W)`.
        """
        if self.use_torchvision_io:
            mode = ImageReadMode.GRAY if self.image_channels == 1 else ImageReadMode.RGB
            return read_image(image_path, mode=mode)

        with Image.open(image_path) as img:
            if self.image_channels == 1:
                arr = np.array(img.convert('L'))
                tensor = torch.from_numpy(arr).unsqueeze(0)
            else:
                arr = np.array(img.convert('RGB'))
                tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor.contiguous()

    def load_item(self, idx):
        """Load one validated sequence window and convert to custom tensors.

        Args:
            idx: Index into `self.valid_items_list`.

        Returns:
            A tuple `(img_seq, target)` where `target` includes point-level and
            placeholder box-level fields used by downstream transforms.
        """
        img_files_seq, target_dict = self.valid_items_list[idx]
        frames = [self._read_image(image_path) for image_path in img_files_seq]

        img_w = target_dict["img_w"]
        img_h = target_dict["img_h"]
        canvas_size = (img_w, img_h, self.seq_len)

        points = np.array([xy + [t] for xy, t in zip(target_dict["points"], target_dict["img_seq"])])

        labels = PointLabels(
            np.array(target_dict["labels"], dtype=np.int64),
            label_map=self.label_dict
        )
        group_ids = PointGroupIDs(target_dict["group_ids"])

        img_seq = ImageSequence(torch.stack(frames, dim=1), canvas_size=canvas_size)
        pts = Points(points, canvas_size=canvas_size)

        # Target field formats
        target = {
            "points": pts,  # Points[N, 3], columns are (x, y, t)
            "points_labels": labels,  # PointLabels[N]
            "points_group_ids": group_ids,  # PointGroupIDs[N]; has correspondence with bboxes_group_ids: same group_id means the point is inside the box (the box encloses the point)
            "seq_len": self.seq_len,  # int, temporal length T
            "image_files": img_files_seq,  # tuple[str, ...], len == T
            "dataset_root": self.dataset_folder,  # str
            "index": idx,  # int, index inside this sub-dataset
            "bboxes": BoundingBoxes3D.empty(format="cxcyczwhd"),  # BoundingBoxes3D[M, 6], empty here; usually filled by GenerateBoundingBoxes3D
            "bboxes_labels": BoxLabels.empty(label_map=self.label_dict),  # BoxLabels[M]
            "bboxes_group_ids": BoxGroupIDs.empty(id_type="int"),  # BoxGroupIDs[M]; has correspondence with points_group_ids: same group_id means the box encloses the point(s) with that group_id
        }

        return img_seq, target

    def extra_repr(self) -> str:
        """Return an extra text summary for dataset introspection."""
        s = f" dataset_folder: {self.dataset_folder}\n seq_len: {self.seq_len}\n"
        if hasattr(self, "_transforms") and self._transforms is not None:
            s += f" transforms:\n   {repr(self._transforms)}"
        return s
