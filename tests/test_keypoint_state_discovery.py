import json
import tempfile
import unittest
from pathlib import Path

from app.core.output_state import (
    ANNOTATION_FILE_NAMES,
    find_annotations_path,
    list_output_states_for_sources,
    load_annotation_state,
)

_PAYLOAD = {
    "info": {"task_mode": "keypoint"},
    "categories": [{"id": 1, "name": "document", "keypoints": ["tl", "tr", "br", "bl"], "skeleton": []}],
    "images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10, "video": "src"}],
    "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5],
                     "keypoints": [0, 0, 2, 5, 0, 2, 5, 5, 2, 0, 5, 2], "num_keypoints": 4}],
}


def _make_state(root: Path, source: str = "src") -> Path:
    states = root / "saved_data_states"
    states.mkdir(parents=True)
    payload = dict(_PAYLOAD)
    payload["info"] = {"task_mode": "keypoint", "data_root": source, "video_sources": [source]}
    (states / "annotations_keypoints.coco.json").write_text(json.dumps(payload), encoding="utf-8")
    return states / "annotations_keypoints.coco.json"


class KeypointStateDiscoveryTest(unittest.TestCase):
    def test_filename_registered(self):
        self.assertIn("annotations_keypoints.coco.json", ANNOTATION_FILE_NAMES)

    def test_find_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_state(Path(tmp))
            self.assertEqual(find_annotations_path(path), path)

    def test_find_from_project_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "proj"
            path = _make_state(project)
            self.assertEqual(find_annotations_path(project), path)

    def test_list_output_states_discovers_keypoint_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            _make_state(parent / "proj", source=str(parent / "src"))
            states = list_output_states_for_sources((parent / "src",), parent)
            self.assertTrue(states, "keypoint project was not discovered")

    def test_load_annotation_state_reads_keypoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_state(Path(tmp))
            state = load_annotation_state(path)
            self.assertEqual(state.categories[0]["keypoints"], ["tl", "tr", "br", "bl"])


if __name__ == "__main__":
    unittest.main()
