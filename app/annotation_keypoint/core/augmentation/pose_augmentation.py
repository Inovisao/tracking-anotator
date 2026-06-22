"""Pose-aware augmentation built on top of the bbox augmentation service.

Each keypoint is encoded as a tiny tracer box whose class slot carries a unique
id. Running the existing ``apply_preset`` transforms image + tracer boxes
consistently (flip/rotate/shear/crop/photometric); we then read the transformed
box centers back as the augmented keypoint positions. This reuses every existing
transform with no duplication. Points dropped by a transform become absent (v=0).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from app.annotation.core.augmentation.augmentation_service import apply_preset
from app.annotation.core.augmentation.augmentation_types import AugmentationPreset

# (class_index, keypoints) where keypoints is [[x, y, v], ...] in absolute pixels.
PoseInstance = Tuple[int, List[List[float]]]
_TRACER_SIZE = 2.0


def augment_pose(
    image: np.ndarray,
    instances: List[PoseInstance],
    preset: AugmentationPreset,
) -> List[Tuple[np.ndarray, List[PoseInstance]]]:
    if image is None or preset is None or not preset.enabled:
        return []
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return []

    tracers: List[List[float]] = []
    uid_by_point = {}
    visibility = {}
    for i, (_cls, kps) in enumerate(instances):
        for j, kp in enumerate(kps):
            if kp[2] > 0:
                uid = len(tracers)
                uid_by_point[(i, j)] = uid
                visibility[uid] = int(kp[2])
                tracers.append([uid, kp[0] / width, kp[1] / height, _TRACER_SIZE / width, _TRACER_SIZE / height])

    results: List[Tuple[np.ndarray, List[PoseInstance]]] = []
    for aug_image, aug_boxes in apply_preset(image, tracers, preset):
        ah, aw = aug_image.shape[:2]
        center_by_uid = {int(box[0]): (float(box[1]) * aw, float(box[2]) * ah) for box in aug_boxes}
        aug_instances: List[PoseInstance] = []
        for i, (cls, kps) in enumerate(instances):
            new_kps: List[List[float]] = []
            for j, kp in enumerate(kps):
                uid = uid_by_point.get((i, j))
                if uid is not None and uid in center_by_uid:
                    cx, cy = center_by_uid[uid]
                    new_kps.append([cx, cy, visibility[uid]])
                else:
                    new_kps.append([0.0, 0.0, 0])
            aug_instances.append((cls, new_kps))
        results.append((aug_image, aug_instances))
    return results
