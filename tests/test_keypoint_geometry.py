import unittest

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
)


def _sample() -> KeypointInstance:
    return KeypointInstance(
        category_id=1,
        keypoints=[[10, 20, V_VISIBLE], [30, 40, V_VISIBLE], [0, 0, V_ABSENT], [50, 60, V_HIDDEN]],
    )


class KeypointGeometryTest(unittest.TestCase):
    def test_num_keypoints_counts_only_annotated(self):
        self.assertEqual(_sample().num_keypoints(), 3)

    def test_bbox_encloses_points_with_visibility_above_zero(self):
        self.assertEqual(keypoints_bbox(_sample()), (10, 20, 40, 40))

    def test_bbox_is_zero_when_no_points_are_annotated(self):
        empty = KeypointInstance(category_id=1, keypoints=[[0, 0, V_ABSENT]])
        self.assertEqual(keypoints_bbox(empty), (0.0, 0.0, 0.0, 0.0))

    def test_instance_area_uses_bbox_dimensions(self):
        self.assertEqual(instance_area(_sample()), 1600.0)

    def test_validate_requires_at_least_one_visible_point(self):
        self.assertTrue(validate_instance(_sample()))
        self.assertFalse(validate_instance(KeypointInstance(category_id=1, keypoints=[[0, 0, V_ABSENT]])))

    def test_nearest_keypoint_ignores_absent_points(self):
        inst = _sample()
        self.assertEqual(nearest_keypoint(inst, 11, 21, 5), 0)
        self.assertIsNone(nearest_keypoint(inst, 1, 1, 5))  # closest is an absent point

    def test_clone_is_a_deep_copy(self):
        inst = _sample()
        copy = clone_keypoint(inst)
        copy.keypoints[0][0] = 999
        self.assertEqual(inst.keypoints[0][0], 10)


if __name__ == "__main__":
    unittest.main()
