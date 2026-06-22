"""Shared dependencies for Keypoint mode."""

import math

from app.annotation.shared import *
from app.annotation_keypoint.geometry.keypoint import (
    V_ABSENT,
    V_HIDDEN,
    V_VISIBLE,
    KeypointInstance,
    clone_keypoint,
    instance_area,
    keypoints_bbox,
    nearest_keypoint,
    validate_instance,
    visible_points,
)
