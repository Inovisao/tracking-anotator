import unittest

from app.annotation_keypoint.geometry.keypoint import V_ABSENT, V_VISIBLE, KeypointInstance
from app.annotation_keypoint.ui.display_overlays import KPDisplayOverlaysMixin


class _Canvas:
    def __init__(self):
        self.lines = 0
        self.ovals = 0
        self.texts = 0

    def delete(self, *args):
        pass

    def create_line(self, *args, **kwargs):
        self.lines += 1

    def create_oval(self, *args, **kwargs):
        self.ovals += 1

    def create_text(self, *args, **kwargs):
        self.texts += 1


class _Overlays(KPDisplayOverlaysMixin):
    def __init__(self, categories):
        self.categories = categories
        self.canvas = _Canvas()

    def image_to_canvas_coords(self, x, y):
        return (int(x), int(y))

    def keypoint_names_for_category(self, category_id):
        for cat in self.categories:
            if cat["id"] == category_id:
                return cat.get("keypoints", [])
        return []


def _instance(points):
    return KeypointInstance(category_id=1, keypoints=[list(p) for p in points])


_CATS = [{"id": 1, "name": "doc", "keypoints": [], "skeleton": []}]
_COLORS = {1: "#ffffff"}


class CategorySkeletonTest(unittest.TestCase):
    def test_returns_skeleton_or_empty(self):
        obj = _Overlays([{"id": 1, "name": "doc", "skeleton": [[1, 2]]}])
        self.assertEqual(obj._category_skeleton(1), [[1, 2]])
        self.assertEqual(obj._category_skeleton(99), [])


class OverlayInstanceTest(unittest.TestCase):
    def setUp(self):
        self.obj = _Overlays(_CATS)
        self.inst = _instance([(0, 0, V_VISIBLE), (10, 0, V_VISIBLE),
                               (10, 10, V_VISIBLE), (0, 10, V_VISIBLE)])

    def test_open_chain_has_n_minus_1_edges(self):
        self.obj._overlay_instance(self.inst, _COLORS, selected=False, sel_kp=None, closed=False)
        self.assertEqual(self.obj.canvas.lines, 3)
        self.assertEqual(self.obj.canvas.ovals, 4)

    def test_closed_chain_adds_closing_edge(self):
        self.obj._overlay_instance(self.inst, _COLORS, selected=False, sel_kp=None, closed=True)
        self.assertEqual(self.obj.canvas.lines, 4)

    def test_absent_points_are_skipped(self):
        inst = _instance([(0, 0, V_VISIBLE), (10, 0, V_ABSENT), (10, 10, V_VISIBLE)])
        self.obj._overlay_instance(inst, _COLORS, selected=False, sel_kp=None, closed=True)
        # 2 visible points -> 1 edge, no closing (needs >= 3), 2 dots
        self.assertEqual(self.obj.canvas.lines, 1)
        self.assertEqual(self.obj.canvas.ovals, 2)

    def test_skeleton_used_when_defined(self):
        obj = _Overlays([{"id": 1, "name": "doc", "keypoints": [], "skeleton": [[0, 2]]}])
        obj._overlay_instance(self.inst, _COLORS, selected=False, sel_kp=None, closed=True)
        self.assertEqual(obj.canvas.lines, 1)  # only the declared skeleton link


if __name__ == "__main__":
    unittest.main()
