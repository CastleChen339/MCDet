"""MCDet 3D head for objectness, trajectory box, and center keypoint."""

import copy
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from .modules_3d import ConvBNAct3D


class PoseHead3D(nn.Module):
    """Multi-scale 3D detection head with keypoint regression.

    Produces per-scale prediction volumes containing objectness, 6D trajectory
    boxes, and sequential center keypoints for each spatial-temporal location.

    Args:
        in_channels: Channel counts of (P3, P4, P5) feature inputs.
        num_classes: Reserved for compatibility; objectness-only mode uses 1.
        kpt_shape: Keypoint layout (num_keypoints, keypoint_dims), e.g. (T, 4).
        end2end: Enable dual one2many + one2one heads for end-to-end training.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        num_classes: int = 1,
        kpt_shape: Tuple[int, int] = (5, 4),
        end2end: bool = False,
        one2one_full_gradient: bool = False,
        one2one_separate_head: bool = False,
    ) -> None:
        """Initialize per-scale stem and prediction branches.

        Args:
            in_channels: Channel counts of `(P3, P4, P5)` features.
            num_classes: Reserved for compatibility; objectness-only training uses 1 class.
            kpt_shape: Keypoint layout `(num_keypoints, keypoint_dims)`, e.g. `(T, 4)` for `(x, y, t, v)`.
            end2end: Enable one2many + one2one dual heads for end-to-end training.
            one2one_full_gradient: Allow one2one loss to update shared task features.
            one2one_separate_head: Use independent one2one task towers.
        """

        super().__init__()
        self.num_classes = num_classes
        if len(kpt_shape) != 2:
            raise ValueError("kpt_shape must be a tuple in format (num_keypoints, keypoint_dims).")

        num_keypoints = int(kpt_shape[0])
        keypoint_dims = int(kpt_shape[1])
        if num_keypoints <= 0:
            raise ValueError("num_keypoints must be a positive integer.")
        if keypoint_dims != 4:
            raise ValueError("PoseHead3D expects fixed keypoint dims = 4 for (x, y, t, v).")

        self.kpt_shape = (num_keypoints, keypoint_dims)
        self.nk = self.kpt_shape[0] * self.kpt_shape[1]
        self.output_dims = 1 + 6 + self.nk
        self.end2end = end2end
        self.one2one_full_gradient = bool(one2one_full_gradient)
        self.one2one_separate_head = bool(one2one_separate_head)

        self.shared_stems = nn.ModuleList(
            ConvBNAct3D(ch, ch, kernel_size=3) for ch in in_channels
        )
        self.obj_stems = nn.ModuleList(
            ConvBNAct3D(ch, ch, kernel_size=(1, 3, 3)) for ch in in_channels
        )
        self.box_stems = nn.ModuleList(
            ConvBNAct3D(ch, ch, kernel_size=(1, 3, 3)) for ch in in_channels
        )
        self.kpt_stems = nn.ModuleList(
            ConvBNAct3D(ch, ch, kernel_size=(1, 3, 3)) for ch in in_channels
        )
        self.obj_pred_layers = nn.ModuleList(
            nn.Conv3d(ch, 1, kernel_size=1, stride=1, padding=0)
            for ch in in_channels
        )
        self.box_pred_layers = nn.ModuleList(
            nn.Conv3d(ch, 6, kernel_size=1, stride=1, padding=0)
            for ch in in_channels
        )
        self.kpt_pred_layers = nn.ModuleList(
            nn.Conv3d(ch, self.nk, kernel_size=1, stride=1, padding=0)
            for ch in in_channels
        )

        if end2end:
            if self.one2one_separate_head:
                self.one2one_shared_stems = copy.deepcopy(self.shared_stems)
                self.one2one_obj_stems = copy.deepcopy(self.obj_stems)
                self.one2one_box_stems = copy.deepcopy(self.box_stems)
                self.one2one_kpt_stems = copy.deepcopy(self.kpt_stems)
            self.one2one_obj_pred_layers = copy.deepcopy(self.obj_pred_layers)
            self.one2one_box_pred_layers = copy.deepcopy(self.box_pred_layers)
            self.one2one_kpt_pred_layers = copy.deepcopy(self.kpt_pred_layers)

    @property
    def one2many(self) -> dict[str, nn.ModuleList]:
        """Return the one-to-many branch modules."""

        return dict(
            shared_stems=self.shared_stems,
            obj_stems=self.obj_stems,
            box_stems=self.box_stems,
            kpt_stems=self.kpt_stems,
            obj_pred_layers=self.obj_pred_layers,
            box_pred_layers=self.box_pred_layers,
            kpt_pred_layers=self.kpt_pred_layers,
        )

    @property
    def one2one(self) -> dict[str, nn.ModuleList]:
        """Return the one-to-one prediction modules."""

        return dict(
            obj_pred_layers=self.one2one_obj_pred_layers,
            box_pred_layers=self.one2one_box_pred_layers,
            kpt_pred_layers=self.one2one_kpt_pred_layers,
        )

    @property
    def one2one_towers(self) -> dict[str, nn.ModuleList]:
        """Return independent one-to-one task towers."""

        return dict(
            shared_stems=self.one2one_shared_stems,
            obj_stems=self.one2one_obj_stems,
            box_stems=self.one2one_box_stems,
            kpt_stems=self.one2one_kpt_stems,
        )

    @property
    def end2end(self) -> bool:
        """Check whether end-to-end dual-head mode is active."""

        return (
            getattr(self, "_end2end", False)
            and hasattr(self, "one2one_obj_pred_layers")
        )

    @end2end.setter
    def end2end(self, value: bool) -> None:
        """Set end-to-end dual-head mode state."""

        self._end2end = bool(value)

    def forward_features(
        self,
        features: Sequence[torch.Tensor],
        shared_stems: nn.ModuleList | None = None,
        obj_stems: nn.ModuleList | None = None,
        box_stems: nn.ModuleList | None = None,
        kpt_stems: nn.ModuleList | None = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Build shared task features for all detection scales.

        Args:
            features: Feature maps for each detection scale.

        Returns:
            List of `(objectness, box, keypoint)` task features per scale.
        """

        modules = (shared_stems, obj_stems, box_stems, kpt_stems)
        if any(module_list is None for module_list in modules):
            return []

        outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for feature, shared, obj_stem, box_stem, kpt_stem in zip(
            features,
            shared_stems,
            obj_stems,
            box_stems,
            kpt_stems,
        ):
            shared_feature = shared(feature)
            outputs.append(
                (
                    obj_stem(shared_feature),
                    box_stem(shared_feature),
                    kpt_stem(shared_feature),
                )
            )
        return outputs

    def predict_features(
        self,
        task_features: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        obj_pred_layers: nn.ModuleList | None = None,
        box_pred_layers: nn.ModuleList | None = None,
        kpt_pred_layers: nn.ModuleList | None = None,
    ) -> List[torch.Tensor]:
        """Project task features into raw prediction tensors."""

        modules = (obj_pred_layers, box_pred_layers, kpt_pred_layers)
        if any(module_list is None for module_list in modules):
            return []

        outputs: List[torch.Tensor] = []
        for task_feature, obj_pred, box_pred, kpt_pred in zip(
            task_features,
            obj_pred_layers,
            box_pred_layers,
            kpt_pred_layers,
        ):
            obj_feature, box_feature, kpt_feature = task_feature
            outputs.append(
                torch.cat(
                    (
                        obj_pred(obj_feature),
                        box_pred(box_feature),
                        kpt_pred(kpt_feature),
                    ),
                    dim=1,
                )
            )
        return outputs

    def forward_head(
        self,
        features: Sequence[torch.Tensor],
        shared_stems: nn.ModuleList | None = None,
        obj_stems: nn.ModuleList | None = None,
        box_stems: nn.ModuleList | None = None,
        kpt_stems: nn.ModuleList | None = None,
        obj_pred_layers: nn.ModuleList | None = None,
        box_pred_layers: nn.ModuleList | None = None,
        kpt_pred_layers: nn.ModuleList | None = None,
    ) -> List[torch.Tensor]:
        """Apply task towers and prediction layers for one branch."""

        task_features = self.forward_features(
            features,
            shared_stems=shared_stems,
            obj_stems=obj_stems,
            box_stems=box_stems,
            kpt_stems=kpt_stems,
        )
        return self.predict_features(
            task_features,
            obj_pred_layers=obj_pred_layers,
            box_pred_layers=box_pred_layers,
            kpt_pred_layers=kpt_pred_layers,
        )

    def forward(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor] | dict[str, List[torch.Tensor]]:
        """Apply head branches and return raw prediction tensors.

        Args:
            features: Multi-scale neck features.

        Returns:
            List[torch.Tensor] | Dict[str, List[torch.Tensor]]: Predictions per scale.
        """

        task_features = self.forward_features(
            features,
            shared_stems=self.shared_stems,
            obj_stems=self.obj_stems,
            box_stems=self.box_stems,
            kpt_stems=self.kpt_stems,
        )
        if self.end2end and not self.training:
            if self.one2one_separate_head:
                task_features = self.forward_features(
                    features,
                    **self.one2one_towers,
                )
            return self.predict_features(
                task_features,
                **self.one2one,
            )

        one2many = self.predict_features(
            task_features,
            obj_pred_layers=self.obj_pred_layers,
            box_pred_layers=self.box_pred_layers,
            kpt_pred_layers=self.kpt_pred_layers,
        )
        if self.end2end:
            if self.one2one_separate_head:
                one2one_inputs = features
                if not self.one2one_full_gradient:
                    one2one_inputs = [feature.detach() for feature in features]
                one2one_features = self.forward_features(
                    one2one_inputs,
                    **self.one2one_towers,
                )
            else:
                one2one_features = task_features
                if not self.one2one_full_gradient:
                    one2one_features = [
                        tuple(feature.detach() for feature in scale_features)
                        for scale_features in task_features
                    ]
            one2one = self.predict_features(
                one2one_features,
                **self.one2one,
            )
            if self.training:
                return {"one2many": one2many, "one2one": one2one}
        return one2many

    def split_prediction(self, prediction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split one scale prediction into objectness, box, and keypoint tensors.

        Args:
            prediction: Raw prediction tensor for one scale.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Objectness, box, and keypoint tensors.
        """

        expected = 1 + 6 + self.nk
        if prediction.shape[1] < expected:
            raise ValueError(f"Prediction channels are insufficient: got {prediction.shape[1]}, expected >= {expected}.")

        obj_logit = prediction[:, 0:1]
        box_raw = prediction[:, 1:7]
        kpt_raw = prediction[:, 7: 7 + self.nk]
        return obj_logit, box_raw, kpt_raw
