"""Lightweight harness to unit-test keypoint mixins without Tk."""

import numpy as np

from app.annotation.core.services.class_service import ClassServiceMixin
from app.annotation_keypoint.detection.frame_pipeline import KPFramePipelineMixin
from app.annotation_keypoint.detection.review_nav import KPReviewNavMixin
from app.annotation_keypoint.detection.selection_edit import KPSelectionEditMixin
from app.annotation_keypoint.infrastructure.persistence.coco_storage import KPCocoStorageMixin
from app.annotation_keypoint.state.runtime_state import KPRuntimeStateMixin
from app.annotation_keypoint.ui.mouse_events import KPMouseEventsMixin


class _Var:
    """Stand-in for tk.StringVar."""

    def __init__(self, value=""):
        self._v = value

    def set(self, value):
        self._v = value

    def get(self):
        return self._v


class FakeKPTool(
    KPMouseEventsMixin,
    KPSelectionEditMixin,
    KPReviewNavMixin,
    KPFramePipelineMixin,
    KPCocoStorageMixin,
    ClassServiceMixin,
    KPRuntimeStateMixin,
):
    """Composes the keypoint logic mixins with UI calls stubbed out."""

    def __init__(self, keypoints=("tl", "tr", "br", "bl")):
        self._initialize_runtime_state()
        self.categories = [{"id": 1, "name": "roi", "keypoints": list(keypoints), "skeleton": []}]
        self.class_to_category_id = {"roi": 1}
        self.target_classes = ["roi"]
        self.manual_class_var = _Var("roi")
        self.info_var = _Var()
        self.current_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.display_scale = 1.0
        self.quit_called = False

    # ── stubbed UI hooks ───────────────────────────────────────────
    def update_display(self, *args, **kwargs):
        pass

    def update_status(self, *args, **kwargs):
        pass

    def update_class_panel(self, *args, **kwargs):
        pass

    def on_quit(self):
        self.quit_called = True

    def canvas_to_image_coords(self, x, y):
        return (x, y)

    def image_to_canvas_coords(self, x, y):
        return (x, y)


class Event:
    def __init__(self, x, y):
        self.x = x
        self.y = y
