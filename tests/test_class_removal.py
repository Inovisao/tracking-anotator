import unittest
from collections import deque
from unittest.mock import patch

import numpy as np

from app.annotation.state.class_config import ClassConfigMixin
from app.models import Detection


def make_detection(category_id: int) -> Detection:
    return Detection(
        original_bbox=np.array([1, 2, 10, 20], dtype=np.float32),
        warp_bbox=None,
        confidence=1.0,
        category_id=category_id,
        track_id=None,
        source="manual",
        internal_id=None,
    )


class ClassRemovalTest(unittest.TestCase):
    class DummyClassConfig(ClassConfigMixin):
        def __init__(self):
            self.target_classes = ["car", "bus"]
            self.class_to_category_id = {"car": 2, "bus": 7}
            self.categories = [
                {"id": 2, "name": "car", "color": "#22c55e", "supercategory": "none"},
                {"id": 7, "name": "bus", "color": "#3b82f6", "supercategory": "none"},
            ]
            self.annotations = [
                {"id": 1, "image_id": 1, "category_id": 2},
                {"id": 2, "image_id": 1, "category_id": 7},
            ]
            self.current_detections = [make_detection(2), make_detection(7)]
            self.manual_detections = [make_detection(7)]
            self.saved_records = [{"detections": [make_detection(2), make_detection(7)]}]
            self.live_snapshot = {
                "detections": [make_detection(7)],
                "manual_detections": [make_detection(2), make_detection(7)],
            }
            self.undo_stack = deque(
                [
                    {
                        "current_detections": [make_detection(7)],
                        "manual_detections": [make_detection(2), make_detection(7)],
                        "selected_detection": ("manual", 0),
                    }
                ],
                maxlen=40,
            )
            self.tracker_id_map = {(2, 101): 1001, (7, 202): 1002}
            self.multiclass_tracker = self.DummyTracker()
            self.max_undo_states = 40
            self.model = None
            self.target_classes_var = None
            self.manual_class_var = None
            self.classes_panel = None
            self.canvas = object()
            self.selected_detection = ("manual", 0)
            self.write_calls = 0
            self.sync_calls = 0
            self.display_calls = 0
            self.status_calls = 0

        def write_annotations(self, *, blocking: bool = False):
            self.write_calls += 1

        def sync_export_metadata(self):
            self.sync_calls += 1

        def update_display(self):
            self.display_calls += 1

        def update_status(self):
            self.status_calls += 1

        class DummyTracker:
            def __init__(self):
                self.reset_calls = 0

            def reset(self):
                self.reset_calls += 1

    def test_remove_class_purges_category_annotations_and_detection_caches(self):
        config = self.DummyClassConfig()


        with patch("app.annotation.state.class_config.messagebox.askyesno", return_value=True):
            config.remove_class("bus")

        self.assertEqual(config.target_classes, ["car"])
        self.assertEqual(config.class_to_category_id, {"car": 1})
        self.assertEqual(config.categories, [{"id": 1, "name": "car", "color": "#22c55e", "supercategory": "none"}])
        self.assertEqual(config.annotations, [{"id": 1, "image_id": 1, "category_id": 1}])
        self.assertEqual([det.category_id for det in config.current_detections], [1])
        self.assertEqual(config.manual_detections, [])
        self.assertEqual([det.category_id for det in config.saved_records[0]["detections"]], [1])
        self.assertEqual(config.live_snapshot["detections"], [])
        self.assertEqual([det.category_id for det in config.live_snapshot["manual_detections"]], [1])
        self.assertEqual([det.category_id for det in config.undo_stack[0]["manual_detections"]], [1])
        self.assertEqual(config.tracker_id_map, {(1, 101): 1001})
        self.assertEqual(config.multiclass_tracker.reset_calls, 1)
        self.assertIsNone(config.undo_stack[0]["selected_detection"])
        self.assertIsNone(config.selected_detection)
        self.assertGreaterEqual(config.write_calls, 1)
        self.assertEqual(config.sync_calls, 1)
        self.assertEqual(config.display_calls, 1)
        self.assertEqual(config.status_calls, 1)

    def test_reordering_classes_remaps_annotations_and_caches(self):
        config = self.DummyClassConfig()

        config.apply_target_classes(["bus", "car"])

        self.assertEqual(config.target_classes, ["bus", "car"])
        self.assertEqual(config.class_to_category_id, {"bus": 1, "car": 2})
        self.assertEqual(
            config.categories,
            [
                {"id": 1, "name": "bus", "color": "#3b82f6", "supercategory": "none"},
                {"id": 2, "name": "car", "color": "#22c55e", "supercategory": "none"},
            ],
        )
        self.assertEqual([ann["category_id"] for ann in config.annotations], [2, 1])
        self.assertEqual([det.category_id for det in config.current_detections], [2, 1])
        self.assertEqual([det.category_id for det in config.manual_detections], [1])
        self.assertEqual([det.category_id for det in config.saved_records[0]["detections"]], [2, 1])
        self.assertEqual(config.tracker_id_map, {(2, 101): 1001, (1, 202): 1002})
        self.assertEqual(config.multiclass_tracker.reset_calls, 1)


if __name__ == "__main__":
    unittest.main()
