import unittest

from app.annotation_keypoint.geometry.keypoint import V_VISIBLE, KeypointInstance
from _keypoint_harness import FakeKPTool


def _instance(points, category_id=1):
    return KeypointInstance(category_id=category_id, keypoints=[list(p) for p in points])


class FindInstanceTest(unittest.TestCase):
    def test_returns_nearest_keypoint_within_radius(self):
        tool = FakeKPTool()
        tool.kp_instances = [_instance([(50, 50, V_VISIBLE)])]
        self.assertEqual(tool.find_instance_at(52, 52), (0, 0))

    def test_falls_back_to_bbox_hit(self):
        tool = FakeKPTool()
        tool.kp_instances = [_instance([(10, 10, V_VISIBLE), (90, 90, V_VISIBLE)])]
        self.assertEqual(tool.find_instance_at(50, 50), (0, None))

    def test_returns_none_when_outside(self):
        tool = FakeKPTool()
        tool.kp_instances = [_instance([(10, 10, V_VISIBLE)])]
        self.assertIsNone(tool.find_instance_at(80, 80))


class RemoveTest(unittest.TestCase):
    def test_removes_point_keeping_instance(self):
        tool = FakeKPTool()
        tool.kp_instances = [_instance([(10, 10, V_VISIBLE), (20, 20, V_VISIBLE)])]
        self.assertTrue(tool.remove_annotation_at(10, 10))
        self.assertEqual(len(tool.kp_instances), 1)
        self.assertEqual(tool.kp_instances[0].keypoints[0][2], 0)  # marked absent
        self.assertEqual(tool.kp_instances[0].num_keypoints(), 1)

    def test_removes_whole_instance_on_last_point(self):
        tool = FakeKPTool()
        tool.kp_instances = [_instance([(10, 10, V_VISIBLE)])]
        self.assertTrue(tool.remove_annotation_at(10, 10))
        self.assertEqual(len(tool.kp_instances), 0)


class UndoTest(unittest.TestCase):
    def test_undo_restores_previous_instances(self):
        tool = FakeKPTool()
        tool.push_undo_state("antes")           # snapshot: empty
        tool.kp_instances = [_instance([(1, 1, V_VISIBLE)])]
        tool.undo_last_action()
        self.assertEqual(len(tool.kp_instances), 0)


class SelectedTest(unittest.TestCase):
    def test_get_and_validate_selected(self):
        tool = FakeKPTool()
        tool.kp_instances = [_instance([(5, 5, V_VISIBLE)])]
        tool.selected_instance = 0
        self.assertIsNotNone(tool.get_selected_detection())
        tool.selected_instance = 9  # out of range
        tool.validate_selected_detection()
        self.assertIsNone(tool.selected_instance)


if __name__ == "__main__":
    unittest.main()
