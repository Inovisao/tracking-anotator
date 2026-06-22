import unittest
from unittest import mock

import numpy as np

from app.annotation_keypoint.geometry.keypoint import V_ABSENT, V_VISIBLE, KeypointInstance
from app.annotation_keypoint.ui.display_overlays import KPDisplayOverlaysMixin


class _Overlays(KPDisplayOverlaysMixin):
    def __init__(self, categories):
        self.categories = categories


def _instance(points):
    return KeypointInstance(category_id=1, keypoints=[list(p) for p in points])


class CategorySkeletonTest(unittest.TestCase):
    def test_returns_skeleton_or_empty(self):
        obj = _Overlays([{"id": 1, "name": "doc", "skeleton": [[1, 2]]}])
        self.assertEqual(obj._category_skeleton(1), [[1, 2]])
        self.assertEqual(obj._category_skeleton(99), [])


class PointChainTest(unittest.TestCase):
    def setUp(self):
        self.obj = _Overlays([{"id": 1, "name": "doc", "skeleton": []}])
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.inst = _instance([(0, 0, V_VISIBLE), (10, 0, V_VISIBLE),
                               (10, 10, V_VISIBLE), (0, 10, V_VISIBLE)])

    def test_open_chain_has_n_minus_1_edges(self):
        with mock.patch("cv2.line") as line:
            self.obj._draw_point_chain(self.frame, self.inst, (0, 255, 0), closed=False)
        self.assertEqual(line.call_count, 3)

    def test_closed_chain_adds_closing_edge(self):
        with mock.patch("cv2.line") as line:
            self.obj._draw_point_chain(self.frame, self.inst, (0, 255, 0), closed=True)
        self.assertEqual(line.call_count, 4)

    def test_absent_points_are_skipped(self):
        inst = _instance([(0, 0, V_VISIBLE), (10, 0, V_ABSENT), (10, 10, V_VISIBLE)])
        with mock.patch("cv2.line") as line:
            self.obj._draw_point_chain(self.frame, inst, (0, 255, 0), closed=True)
        # only 2 visible points -> 1 edge, no closing (needs >= 3)
        self.assertEqual(line.call_count, 1)


if __name__ == "__main__":
    unittest.main()
