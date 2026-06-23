"""Shared dependencies for the annotation mixins."""

import json
import os
import shutil
import signal
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# NOTE: ultralytics (YOLO) and tracker.byte_tracker (BYTETracker) pull in torch
# (~4s, hundreds of MB). They are imported lazily where actually used
# (core_init.ensure_models_loaded, runtime_state, source_helpers) so modes that
# never load a model — keypoint, classification, OBB browsing — start instantly.
from app.core.session import AnnotationSessionConfig, AnnotationTaskMode
from app.config import (
    CANVAS_PADDING_PX,
    CONF_THRESHOLD,
    DATA_ROOT,
    FALLBACK_TO_ORIGINAL_IF_EMPTY,
    IMAGE_EXTENSIONS,
    IMAGE_LIST_EXTENSIONS,
    MANUAL_IOU_THRESHOLD,
    MAX_SAVED_FRAME_CACHE,
    SAVE_RECTIFIED_FRAMES,
    SHOW_MANUAL_DETECTIONS,
    SHOW_MODEL_DETECTIONS,
    TARGET_CLASSES,
    USE_RECTIFIED_FOR_DETECTION,
    VIDEO_EXTENSIONS,
    WEIGHTS_PATH,
    WINDOW_MARGIN_PX,
    WINDOW_TOP_RESERVED_PX,
)
from app.geometry import (
    bbox_center,
    bbox_iou,
    clip_bbox,
    destination_size,
    order_points,
    parse_frame_number_from_name,
)
from app.models import ByteTrackerArgs, Detection
