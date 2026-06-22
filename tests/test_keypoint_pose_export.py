import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.annotation.core.augmentation.augmentation_types import AugEntry, AugmentationPreset
from app.annotation_keypoint.core.augmentation.pose_augmentation import augment_pose
from app.annotation_keypoint.infrastructure.export.yolo_pose_exporter import export_yolo_pose_dataset


def _payload(n_images: int) -> dict:
    images = [{"id": i + 1, "file_name": f"img{i}.jpg", "width": 100, "height": 100} for i in range(n_images)]
    annotations = [
        {"id": i + 1, "image_id": i + 1, "category_id": 1, "bbox": [10, 10, 40, 40],
         "keypoints": [10, 10, 2, 50, 50, 2, 30, 70, 2, 70, 30, 2]}
        for i in range(n_images)
    ]
    return {
        "categories": [{"id": 1, "name": "document", "keypoints": ["tl", "tr", "br", "bl"]}],
        "images": images,
        "annotations": annotations,
    }


def _write_images(source_dir: Path, n: int):
    source_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(source_dir / f"img{i}.jpg"), np.zeros((100, 100, 3), dtype=np.uint8))


class PoseAugmentationTest(unittest.TestCase):
    def test_flip_h_mirrors_keypoint_x(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        preset = AugmentationPreset(
            enabled=True, copies_per_image=1,
            entries=[AugEntry(key="flip_h", enabled=True, params={"prob": 1.0})],
        )
        results = augment_pose(image, [(0, [[10, 20, 2], [30, 40, 2]])], preset)
        self.assertEqual(len(results), 1)
        _aug_img, instances = results[0]
        (cls, kps) = instances[0]
        self.assertEqual(cls, 0)
        self.assertAlmostEqual(kps[0][0], 90.0, delta=2.0)
        self.assertAlmostEqual(kps[0][1], 20.0, delta=2.0)
        self.assertEqual(kps[0][2], 2)


class PoseExportSplitTest(unittest.TestCase):
    def test_split_creates_train_val_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "images"
            _write_images(src, 4)
            export_yolo_pose_dataset(_payload(4), root / "out", src, split_ratios=(0.5, 0.25, 0.25))
            for split in ("train", "val", "test"):
                self.assertTrue((root / "out" / "images" / split).is_dir())
            labels = list((root / "out" / "labels").rglob("*.txt"))
            self.assertEqual(len(labels), 4)

    def test_augmentation_adds_train_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "images"
            _write_images(src, 2)
            preset = AugmentationPreset(
                enabled=True, copies_per_image=2,
                entries=[AugEntry(key="brightness", enabled=True, params={"range_pct": 20.0})],
            )
            report = export_yolo_pose_dataset(_payload(2), root / "out", src, augmentation_preset=preset)
            train_labels = list((root / "out" / "labels" / "train").glob("*.txt"))
            # 2 originals + 2 copies each = 6
            self.assertEqual(len(train_labels), 6)
            self.assertEqual(report["images"], 6)


if __name__ == "__main__":
    unittest.main()
