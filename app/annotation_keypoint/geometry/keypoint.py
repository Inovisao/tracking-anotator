from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Visibility flags follow the COCO keypoints convention.
V_ABSENT = 0   # not annotated
V_HIDDEN = 1   # annotated but occluded
V_VISIBLE = 2  # annotated and visible


@dataclass
class KeypointInstance:
    category_id: int
    keypoints: List[List[float]] = field(default_factory=list)  # [[x, y, v], ...] in fixed order
    confidence: float = 1.0
    source: str = "manual"
    internal_id: Optional[int] = None
    track_id: Optional[int] = None

    def num_keypoints(self) -> int:
        return sum(1 for kp in self.keypoints if kp[2] > 0)


def clone_keypoint(inst: KeypointInstance) -> KeypointInstance:
    return KeypointInstance(
        category_id=inst.category_id,
        keypoints=[list(kp) for kp in inst.keypoints],
        confidence=inst.confidence,
        source=inst.source,
        internal_id=inst.internal_id,
        track_id=inst.track_id,
    )


def visible_points(inst: KeypointInstance) -> List[Tuple[float, float]]:
    return [(kp[0], kp[1]) for kp in inst.keypoints if kp[2] > 0]


def keypoints_bbox(inst: KeypointInstance) -> Tuple[float, float, float, float]:
    """Smallest axis-aligned box enclosing every point with v > 0."""
    points = visible_points(inst)
    if not points:
        return 0.0, 0.0, 0.0, 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
    return x_min, y_min, x_max - x_min, y_max - y_min


def instance_area(inst: KeypointInstance) -> float:
    _, _, w, h = keypoints_bbox(inst)
    return float(max(w, 0.0) * max(h, 0.0))


def validate_instance(inst: KeypointInstance) -> bool:
    return inst.num_keypoints() > 0


def nearest_keypoint(inst: KeypointInstance, x: float, y: float, max_dist: float) -> Optional[int]:
    """Index of the closest annotated point within max_dist, or None."""
    best_idx = None
    best_dist = max_dist
    for idx, kp in enumerate(inst.keypoints):
        if kp[2] <= 0:
            continue
        dist = math.hypot(kp[0] - x, kp[1] - y)
        if dist <= best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx
