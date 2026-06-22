import tempfile
import unittest
from pathlib import Path

from app.annotation_keypoint.infrastructure.export.yolo_pose_exporter import export_yolo_pose_dataset


class YOLOPoseExportTest(unittest.TestCase):
    def test_exports_normalized_pose_labels(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_images = root / "images"
            source_images.mkdir()
            (source_images / "img.jpg").write_bytes(b"fake")
            payload = {
                "categories": [{"id": 2, "name": "peca", "keypoints": ["a", "b", "c", "d"]}],
                "images": [{"id": 1, "file_name": "img.jpg", "width": 100, "height": 100}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 2,
                        "bbox": [10, 20, 40, 40],
                        "keypoints": [10, 20, 2, 30, 40, 2, 0, 0, 0, 50, 60, 1],
                    }
                ],
            }

            summary = export_yolo_pose_dataset(payload, root / "out", source_images)
            label = (root / "out" / "labels" / "train" / "img.txt").read_text(encoding="utf-8").strip()
            data_yaml = (root / "out" / "data.yaml").read_text(encoding="utf-8")

        self.assertEqual(summary["images"], 1)
        self.assertEqual(summary["labels"], 1)
        self.assertEqual(
            label,
            "0 0.300000 0.400000 0.400000 0.400000 "
            "0.100000 0.200000 2 0.300000 0.400000 2 0.000000 0.000000 0 0.500000 0.600000 1",
        )
        self.assertIn("kpt_shape: [4, 3]", data_yaml)

    def test_absent_points_exported_as_zero(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_images = root / "images"
            source_images.mkdir()
            (source_images / "img.jpg").write_bytes(b"fake")
            payload = {
                "categories": [{"id": 1, "name": "obj", "keypoints": ["a", "b"]}],
                "images": [{"id": 1, "file_name": "img.jpg", "width": 200, "height": 200}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 100, 100],
                        "keypoints": [100, 100, 2, 0, 0, 0],
                    }
                ],
            }
            export_yolo_pose_dataset(payload, root / "out", source_images)
            label = (root / "out" / "labels" / "train" / "img.txt").read_text(encoding="utf-8").strip()

        self.assertTrue(label.endswith("0.500000 0.500000 2 0.000000 0.000000 0"))


if __name__ == "__main__":
    unittest.main()
