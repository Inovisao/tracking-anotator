import tempfile
import unittest
from pathlib import Path

from app.annotation_keypoint.geometry.keypoint import KeypointInstance
from app.annotation_keypoint.detection.selection_edit import KPSelectionEditMixin
from app.annotation_keypoint.infrastructure.persistence.coco_storage import KPCocoStorageMixin
from app.annotation_keypoint.infrastructure.export.yolo_pose_exporter import export_yolo_pose_dataset
from utils.fix_keypoint_coco import fix_payload


class _Store(KPCocoStorageMixin):
    def __init__(self, categories, annotations):
        self.categories = categories
        self.annotations = annotations

    def category_name_by_id(self):
        return {int(c["id"]): c["name"] for c in self.categories}


class EnsureKeypointMetadataTest(unittest.TestCase):
    def test_empty_keypoints_filled_to_match_annotations(self):
        store = _Store(
            categories=[{"id": 1, "name": "obj", "keypoints": [], "skeleton": []}],
            annotations=[{"category_id": 1, "keypoints": [0, 0, 2, 1, 1, 2, 2, 2, 1]}],
        )
        store.ensure_keypoint_metadata()
        self.assertEqual(store.categories[0]["keypoints"], ["point_1", "point_2", "point_3"])

    def test_dataset_errors_flags_inconsistent_counts(self):
        store = _Store(
            categories=[{"id": 1, "name": "doc", "keypoints": ["a", "b", "c", "d"]}],
            annotations=[
                {"category_id": 1, "keypoints": [0] * 12},
                {"category_id": 1, "keypoints": [0] * 15},
            ],
        )
        errors = store.keypoint_dataset_errors()
        self.assertTrue(any("diferentes" in e for e in errors))


class ClosingPointDedupTest(unittest.TestCase):
    def test_trailing_duplicate_of_first_point_is_dropped(self):
        inst = KeypointInstance(
            category_id=1,
            keypoints=[[21, 14, 2], [1015, 23, 2], [1028, 724, 2], [19, 755, 2], [19, 16, 2]],
        )
        KPSelectionEditMixin._drop_duplicate_closing_point(inst)
        self.assertEqual(len(inst.keypoints), 4)


class ExportUniformLengthTest(unittest.TestCase):
    def test_all_lines_match_kpt_shape(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            images = root / "images"
            images.mkdir()
            (images / "a.jpg").write_bytes(b"x")
            (images / "b.jpg").write_bytes(b"x")
            payload = {
                "categories": [{"id": 1, "name": "obj", "keypoints": ["p1", "p2", "p3"]}],
                "images": [
                    {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
                    {"id": 2, "file_name": "b.jpg", "width": 100, "height": 100},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "keypoints": [10, 10, 2]},
                    {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10],
                     "keypoints": [10, 10, 2, 20, 20, 2, 30, 30, 2]},
                ],
            }
            export_yolo_pose_dataset(payload, root / "out", images)
            line_a = (root / "out" / "labels" / "train" / "a.txt").read_text().strip()
            line_b = (root / "out" / "labels" / "train" / "b.txt").read_text().strip()

        # 5 bbox fields + 3 keypoints × 3 values = 14 tokens on every line.
        self.assertEqual(len(line_a.split()), 14)
        self.assertEqual(len(line_b.split()), 14)


class FixScriptTest(unittest.TestCase):
    def test_fix_payload_repairs_document_annotation(self):
        payload = {
            "categories": [{"id": 1, "name": "document", "keypoints": [], "skeleton": []}],
            "annotations": [{
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [19, 14, 1009, 741], "num_keypoints": 5,
                "keypoints": [21, 14, 2, 19, 755, 2, 1028, 724, 2, 1015, 23, 2, 19, 16, 2],
            }],
        }
        fixed = fix_payload(payload)
        ann = fixed["annotations"][0]
        self.assertEqual(ann["num_keypoints"], 4)
        self.assertEqual(len(ann["keypoints"]), 12)
        # Default ordering is TL, TR, BR, BL.
        self.assertEqual(ann["keypoints"], [21.0, 14.0, 2, 1015.0, 23.0, 2, 1028.0, 724.0, 2, 19.0, 755.0, 2])
        self.assertEqual(fixed["categories"][0]["keypoints"],
                         ["top_left", "top_right", "bottom_right", "bottom_left"])
        self.assertEqual(fixed["categories"][0]["skeleton"], [[1, 2], [2, 3], [3, 4], [4, 1]])

    def test_no_sort_preserves_order(self):
        payload = {
            "categories": [{"id": 1, "name": "document", "keypoints": [], "skeleton": []}],
            "annotations": [{
                "id": 1, "image_id": 1, "category_id": 1,
                "keypoints": [21, 14, 2, 19, 755, 2, 1028, 724, 2, 1015, 23, 2],
            }],
        }
        fixed = fix_payload(payload, sort_corners=False)
        self.assertEqual(fixed["annotations"][0]["keypoints"],
                         [21.0, 14.0, 2, 19.0, 755.0, 2, 1028.0, 724.0, 2, 1015.0, 23.0, 2])


if __name__ == "__main__":
    unittest.main()
