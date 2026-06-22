import unittest

from app.annotation_keypoint.geometry.keypoint import KeypointInstance
from app.annotation_keypoint.infrastructure.persistence.coco_storage import KPCocoStorageMixin


class _Store(KPCocoStorageMixin):
    def __init__(self):
        self.video_path = None
        self.video_name = "vid"


class KeypointStorageTest(unittest.TestCase):
    def setUp(self):
        self.store = _Store()
        self.inst = KeypointInstance(
            category_id=2,
            keypoints=[[10, 20, 2], [30, 40, 2], [0, 0, 0], [50, 60, 1]],
        )

    def test_instance_to_annotation_builds_coco_keypoint_record(self):
        ann = self.store.instance_to_annotation(self.inst, image_id=7, annotation_id=3)
        self.assertEqual(ann["id"], 3)
        self.assertEqual(ann["image_id"], 7)
        self.assertEqual(ann["category_id"], 2)
        self.assertEqual(ann["bbox"], [10.0, 20.0, 40.0, 40.0])
        self.assertEqual(ann["area"], 1600.0)
        self.assertEqual(ann["num_keypoints"], 3)
        self.assertEqual(ann["keypoints"], [10.0, 20.0, 2, 30.0, 40.0, 2, 0.0, 0.0, 0, 50.0, 60.0, 1])
        self.assertEqual(ann["annotation_type"], "keypoint")

    def test_round_trip_preserves_points_and_class(self):
        ann = self.store.instance_to_annotation(self.inst, image_id=7, annotation_id=3)
        restored = self.store.annotation_to_instance(ann)
        self.assertEqual(restored.category_id, 2)
        self.assertEqual(restored.num_keypoints(), 3)
        self.assertEqual(restored.keypoints, [[10.0, 20.0, 2], [30.0, 40.0, 2], [0.0, 0.0, 0], [50.0, 60.0, 1]])


if __name__ == "__main__":
    unittest.main()
