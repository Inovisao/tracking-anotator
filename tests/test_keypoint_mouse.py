import unittest

from app.annotation_keypoint.geometry.keypoint import V_ABSENT, V_HIDDEN, V_VISIBLE
from _keypoint_harness import Event, FakeKPTool


class PlacePointTest(unittest.TestCase):
    def test_fixed_fills_in_order_and_autocommits(self):
        tool = FakeKPTool(keypoints=("tl", "tr", "br", "bl"))
        for x, y in [(0, 0), (10, 0), (10, 10), (0, 10)]:
            tool.place_point(x, y, V_VISIBLE)
        self.assertIsNone(tool.wip_instance)  # auto-closed at N
        self.assertEqual(len(tool.kp_instances), 1)
        pts = [(kp[0], kp[1]) for kp in tool.kp_instances[0].keypoints]
        self.assertEqual(pts, [(0, 0), (10, 0), (10, 10), (0, 10)])

    def test_free_appends_without_autoclose(self):
        tool = FakeKPTool(keypoints=())
        for x, y in [(0, 0), (10, 0), (10, 10)]:
            tool.place_point(x, y, V_VISIBLE)
        self.assertIsNotNone(tool.wip_instance)
        self.assertEqual(len(tool.wip_instance.keypoints), 3)
        self.assertEqual(len(tool.kp_instances), 0)


class WipCloseTest(unittest.TestCase):
    def _free_triangle(self):
        tool = FakeKPTool(keypoints=())
        for x, y in [(10, 10), (90, 10), (50, 90)]:
            tool.place_point(x, y, V_VISIBLE)
        return tool

    def test_close_only_near_first_point_free_mode(self):
        tool = self._free_triangle()
        self.assertTrue(tool._wip_should_close(11, 11))    # near first
        self.assertFalse(tool._wip_should_close(80, 80))   # far

    def test_fixed_mode_never_closes_by_first_point(self):
        tool = FakeKPTool(keypoints=("a", "b", "c", "d"))
        for x, y in [(10, 10), (90, 10), (50, 90)]:
            tool.place_point(x, y, V_VISIBLE)
        self.assertFalse(tool._wip_should_close(11, 11))

    def test_closing_click_does_not_add_extra_point(self):
        tool = self._free_triangle()
        tool.on_mouse_down(Event(11, 11))  # click back on first point
        self.assertIsNone(tool.wip_instance)
        self.assertEqual(len(tool.kp_instances), 1)
        self.assertEqual(len(tool.kp_instances[0].keypoints), 3)  # no duplicate 4th


class EscapeTest(unittest.TestCase):
    def test_escape_cancels_wip_without_quitting(self):
        tool = FakeKPTool(keypoints=())
        tool.place_point(10, 10, V_VISIBLE)
        tool.on_escape()
        self.assertIsNone(tool.wip_instance)
        self.assertFalse(tool.quit_called)

    def test_escape_never_quits_when_idle(self):
        tool = FakeKPTool()
        tool.on_escape()
        self.assertFalse(tool.quit_called)

    def test_escape_clears_selection(self):
        tool = FakeKPTool()
        tool.selected_instance = 0
        tool.selected_kp = 1
        tool.on_escape()
        self.assertIsNone(tool.selected_instance)
        self.assertIsNone(tool.selected_kp)
        self.assertFalse(tool.quit_called)


class VisibilityAndUndoTest(unittest.TestCase):
    def test_cycle_visibility(self):
        tool = FakeKPTool()
        self.assertEqual(tool.next_visibility, V_VISIBLE)
        tool.cycle_next_visibility()
        self.assertEqual(tool.next_visibility, V_HIDDEN)
        tool.cycle_next_visibility()
        self.assertEqual(tool.next_visibility, V_ABSENT)
        tool.cycle_next_visibility()
        self.assertEqual(tool.next_visibility, V_VISIBLE)

    def test_skip_point_marks_absent_fixed(self):
        tool = FakeKPTool(keypoints=("a", "b"))
        tool.place_point(5, 5, V_VISIBLE)
        tool.skip_point()
        self.assertIsNone(tool.wip_instance)  # 2 slots filled (1 real + 1 absent) -> committed
        kps = tool.kp_instances[0].keypoints
        self.assertEqual(kps[1][2], V_ABSENT)

    def test_undo_last_point_then_cancel(self):
        tool = FakeKPTool(keypoints=())
        tool.place_point(10, 10, V_VISIBLE)
        tool.place_point(20, 20, V_VISIBLE)
        tool.undo_last_point()
        self.assertEqual(len(tool.wip_instance.keypoints), 1)
        tool.undo_last_point()
        self.assertIsNone(tool.wip_instance)  # cancelled when last point removed


if __name__ == "__main__":
    unittest.main()
