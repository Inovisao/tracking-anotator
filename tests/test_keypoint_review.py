import unittest

from app.annotation_keypoint.geometry.keypoint import V_VISIBLE, KeypointInstance
from _keypoint_harness import FakeKPTool


class RebuildTest(unittest.TestCase):
    def test_rebuild_from_annotations(self):
        tool = FakeKPTool()
        tool.annotations = [{
            "id": 1, "image_id": 7, "category_id": 1,
            "keypoints": [10, 20, 2, 30, 40, 1], "source": "manual",
        }]
        instances = tool.rebuild_detections_from_annotations(7)
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].keypoints, [[10.0, 20.0, 2], [30.0, 40.0, 1]])

    def test_restore_for_current_frame(self):
        tool = FakeKPTool()
        tool.current_source_type = "video"
        tool.video_name = "vid"
        tool.frame_index = 1
        file_name = tool.current_frame_file_name()
        tool.images = [{"id": 3, "file_name": file_name, "width": 100, "height": 100}]
        tool.annotations = [{"id": 1, "image_id": 3, "category_id": 1, "keypoints": [1, 2, 2]}]
        tool.restore_saved_annotations_for_current_frame()
        self.assertEqual(len(tool.kp_instances), 1)
        self.assertEqual(tool.kp_instances[0].keypoints, [[1.0, 2.0, 2]])


class ResetAndSaveTest(unittest.TestCase):
    def test_reset_frame_instances(self):
        tool = FakeKPTool()
        tool.kp_instances = [KeypointInstance(category_id=1, keypoints=[[1, 1, V_VISIBLE]])]
        tool.selected_instance = 0
        tool.wip_index = 2
        tool.reset_frame_instances()
        self.assertEqual(tool.kp_instances, [])
        self.assertIsNone(tool.wip_instance)
        self.assertIsNone(tool.selected_instance)
        self.assertEqual(tool.wip_index, 0)

    def test_detections_to_save_commits_wip(self):
        tool = FakeKPTool(keypoints=())
        for x, y in [(0, 0), (10, 0), (10, 10)]:
            tool.place_point(x, y, V_VISIBLE)
        saved = tool.detections_to_save()
        self.assertIsNone(tool.wip_instance)        # wip committed
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].num_keypoints(), 3)


if __name__ == "__main__":
    unittest.main()
