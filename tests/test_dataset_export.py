import tempfile
import unittest
import json
from pathlib import Path

import cv2
import numpy as np

from app.annotation.core.augmentation.augmentation_types import AugEntry, AugmentationPreset
from app.annotation.detection.persistence import PersistenceMixin
from app.annotation.infrastructure.persistence.export_actions import ExportActionsMixin
from app.annotation.ui.display_canvas import DisplayCanvasMixin
from app.annotation.detection.workflow_actions import WorkflowActionsMixin
from app.annotation.sources.source_helpers import SourceHelpersMixin
from app.config import DATA_ROOT
from app.dataset_export import export_yolo_dataset
from utils.merge_yolo_splits import merge_yolo_splits


class ExportYoloDatasetTest(unittest.TestCase):
    def test_keeps_empty_image_in_train_without_special_split_handling(self):
        payload = {
            "images": [
                {"id": 1, "file_name": "img_001.jpg", "width": 100, "height": 100},
                {"id": 2, "file_name": "img_002.jpg", "width": 100, "height": 100},
                {"id": 3, "file_name": "img_003.jpg", "width": 100, "height": 100},
                {"id": 4, "file_name": "img_004.jpg", "width": 100, "height": 100},
                {"id": 5, "file_name": "img_005.jpg", "width": 100, "height": 100},
                {"id": 6, "file_name": "img_006.jpg", "width": 100, "height": 100},
            ],
            "annotations": [
                {"id": 1, "image_id": 2, "category_id": 1, "bbox": [10, 10, 20, 20]},
                {"id": 2, "image_id": 3, "category_id": 1, "bbox": [15, 15, 20, 20]},
                {"id": 3, "image_id": 5, "category_id": 1, "bbox": [20, 20, 20, 20]},
            ],
            "categories": [{"id": 1, "name": "car"}],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_images_dir = tmp_path / "images"
            dataset_root = tmp_path / "yolo_dataset"
            source_images_dir.mkdir(parents=True, exist_ok=True)

            for image in payload["images"]:
                (source_images_dir / image["file_name"]).write_bytes(b"fake-image")

            report = export_yolo_dataset(
                payload,
                source_images_dir=source_images_dir,
                dataset_root=dataset_root,
                split_ratios=(0.5, 0.25, 0.25),
            )

            self.assertEqual(report["images_per_split"], {"train": 3, "val": 2, "test": 1})
            self.assertEqual(report["empty_images_per_split"], {"train": 1, "val": 1, "test": 1})
            self.assertEqual(
                (dataset_root / "labels" / "train" / "img_001.txt").read_text(encoding="utf-8"),
                "",
            )
            self.assertTrue((dataset_root / "images" / "train" / "img_001.jpg").exists())

    def test_yolo_export_writes_augmented_images_and_labels_in_same_split(self):
        payload = {
            "images": [{"id": 1, "file_name": "img_001.jpg", "width": 100, "height": 100}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 20, 20]}],
            "categories": [{"id": 1, "name": "car"}],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_images_dir = tmp_path / "images"
            dataset_root = tmp_path / "yolo_dataset"
            source_images_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(source_images_dir / "img_001.jpg"), np.zeros((100, 100, 3), dtype=np.uint8))

            preset = AugmentationPreset(
                enabled=True,
                copies_per_image=1,
                entries=[AugEntry(key="flip_h", enabled=True, params={"prob": 1.0})],
            )

            report = export_yolo_dataset(
                payload,
                source_images_dir=source_images_dir,
                dataset_root=dataset_root,
                split_ratios=(1.0, 0.0, 0.0),
                augmentation_preset=preset,
            )

            self.assertTrue((dataset_root / "images" / "train" / "img_001_aug1.jpg").exists())
            aug_label = (dataset_root / "labels" / "train" / "img_001_aug1.txt").read_text(encoding="utf-8")
            self.assertIn("0 0.800000 0.300000 0.200000 0.200000", aug_label)
            self.assertEqual(report["images_per_split"]["train"], 2)
            self.assertEqual(report["labels_per_split"]["train"], 2)

    def test_yolo_export_keeps_all_originals_when_augmentation_has_multiple_copies(self):
        payload = {
            "images": [
                {"id": idx, "file_name": f"img_{idx:03d}.jpg", "width": 100, "height": 100}
                for idx in range(1, 4)
            ],
            "annotations": [
                {"id": idx, "image_id": idx, "category_id": 1, "bbox": [10, 10, 20, 20]}
                for idx in range(1, 4)
            ],
            "categories": [{"id": 1, "name": "car"}],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_images_dir = tmp_path / "images"
            dataset_root = tmp_path / "yolo_dataset"
            source_images_dir.mkdir(parents=True, exist_ok=True)
            for image in payload["images"]:
                cv2.imwrite(
                    str(source_images_dir / image["file_name"]),
                    np.zeros((100, 100, 3), dtype=np.uint8),
                )

            preset = AugmentationPreset(
                enabled=True,
                copies_per_image=2,
                entries=[AugEntry(key="flip_h", enabled=True, params={"prob": 1.0})],
            )

            report = export_yolo_dataset(
                payload,
                source_images_dir=source_images_dir,
                dataset_root=dataset_root,
                split_ratios=(1.0, 0.0, 0.0),
                augmentation_preset=preset,
            )

            for idx in range(1, 4):
                stem = f"img_{idx:03d}"
                self.assertTrue((dataset_root / "images" / "train" / f"{stem}.jpg").exists())
                self.assertTrue((dataset_root / "images" / "train" / f"{stem}_aug1.jpg").exists())
                self.assertTrue((dataset_root / "images" / "train" / f"{stem}_aug2.jpg").exists())
                self.assertTrue((dataset_root / "labels" / "train" / f"{stem}.txt").exists())

            self.assertEqual(report["images_per_split"]["train"], 9)
            self.assertEqual(report["labels_per_split"]["train"], 9)

    def test_preserves_subfolders_for_images_and_labels(self):
        payload = {
            "images": [
                {"id": 1, "file_name": "lote_a/img_001.jpg", "width": 100, "height": 100},
                {"id": 2, "file_name": "lote_b/img_002.jpg", "width": 100, "height": 100},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
            ],
            "categories": [{"id": 1, "name": "car"}],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_images_dir = tmp_path / "images"
            dataset_root = tmp_path / "yolo_dataset"
            for image in payload["images"]:
                image_path = source_images_dir / image["file_name"]
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(b"fake-image")

            export_yolo_dataset(
                payload,
                source_images_dir=source_images_dir,
                dataset_root=dataset_root,
                split_ratios=(1.0, 0.0, 0.0),
            )

            self.assertTrue((dataset_root / "images" / "train" / "lote_a" / "img_001.jpg").exists())
            self.assertTrue((dataset_root / "labels" / "train" / "lote_a" / "img_001.txt").exists())
            self.assertTrue((dataset_root / "images" / "train" / "lote_b" / "img_002.jpg").exists())
            self.assertTrue((dataset_root / "labels" / "train" / "lote_b" / "img_002.txt").exists())


class PersistenceMixinTest(unittest.TestCase):
    def test_build_output_file_name_uses_relative_path_inside_data_root(self):
        class DummyPersistence(PersistenceMixin):
            pass

        persistence = DummyPersistence()
        persistence.data_root = DATA_ROOT
        persistence.current_source_type = "images"
        persistence.current_source_image_path = DATA_ROOT / "lote_a" / "img_001.jpg"
        persistence.video_path = DATA_ROOT
        persistence.video_name = DATA_ROOT.name

        self.assertEqual(
            persistence.build_output_file_name(new_frame=True, existing_file_name=None),
            "lote_a/img_001.jpg",
        )


class WorkflowActionsTest(unittest.TestCase):
    def test_accept_saves_empty_frame(self):
        class DummyWorkflow(WorkflowActionsMixin):
            def __init__(self):
                self.current_frame = np.zeros((16, 16, 3), dtype=np.uint8)
                self.current_detections = []
                self.manual_detections = []
                self.review_idx = None
                self.saved_records = []
                self.store_calls = []
                self.write_calls = 0
                self.load_next_frame_calls = 0

            def store_annotations(self, detections, existing_image_id=None, existing_file_name=None):
                self.store_calls.append(
                    {
                        "detections": list(detections),
                        "existing_image_id": existing_image_id,
                        "existing_file_name": existing_file_name,
                    }
                )
                return 7, "frame_00007.jpg"

            def write_annotations(self, *, blocking: bool = False):
                self.write_calls += 1

            def append_saved_record(self, detections, image_id, file_name):
                self.saved_records.append(
                    {"detections": list(detections), "image_id": image_id, "file_name": file_name}
                )

            def load_next_frame(self):
                self.load_next_frame_calls += 1

            def update_manual_memory_after_accept(self, detections):
                raise AssertionError("Nao deveria atualizar memoria manual sem deteccoes.")

        workflow = DummyWorkflow()

        workflow.on_accept()

        self.assertEqual(len(workflow.store_calls), 1)
        self.assertEqual(workflow.store_calls[0]["detections"], [])
        self.assertEqual(workflow.write_calls, 1)
        self.assertEqual(len(workflow.saved_records), 1)
        self.assertEqual(workflow.saved_records[0]["detections"], [])
        self.assertEqual(workflow.load_next_frame_calls, 1)


class ExportOutputDatasetPathTest(unittest.TestCase):
    def test_external_folder_receives_output_folder_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "outputs" / "output_dataset1"
            destination_root = tmp_path / "exports"
            output_dir.mkdir(parents=True)
            destination_root.mkdir()

            class DummyPersistence(PersistenceMixin):
                pass

            persistence = DummyPersistence()
            persistence.output_dir = output_dir

            self.assertEqual(
                persistence.resolve_export_dataset_path(destination_root),
                destination_root / output_dir.name,
            )

    def test_output_folder_selection_exports_to_sibling(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "output_dataset1"
            output_dir.mkdir()

            class DummyPersistence(PersistenceMixin):
                pass

            persistence = DummyPersistence()
            persistence.output_dir = output_dir

            destination = persistence.resolve_export_dataset_path(output_dir)

            self.assertNotEqual(destination, output_dir.resolve())
            self.assertEqual(destination.parent, output_dir.parent.resolve())
            self.assertTrue(destination.name.startswith(f"{output_dir.name}_export"))

    def test_output_subfolder_selection_exports_to_sibling(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "output_dataset1"
            subfolder = output_dir / "subfolder"
            subfolder.mkdir(parents=True)

            class DummyPersistence(PersistenceMixin):
                pass

            persistence = DummyPersistence()
            persistence.output_dir = output_dir

            destination = persistence.resolve_export_dataset_path(subfolder)

            self.assertNotEqual(destination, output_dir.resolve())
            self.assertFalse(output_dir.resolve() in destination.parents)

    def test_user_export_root_never_points_inside_output_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "output_dataset1"
            output_dir.mkdir()

            class DummyExportActions(ExportActionsMixin):
                pass

            actions = DummyExportActions()
            actions.output_dir = output_dir

            destination = actions.resolve_user_export_root(output_dir.parent, output_dir.name)

            self.assertNotEqual(destination, output_dir.resolve())
            self.assertFalse(output_dir.resolve() in destination.parents)
            self.assertEqual(destination.parent, output_dir.parent.resolve())

    def test_load_export_payload_uses_persisted_state_instead_of_memory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_dir = root / "output_dataset1"
            images_dir = output_dir / "images"
            images_dir.mkdir(parents=True)
            annotations_path = output_dir / "annotations.coco.json"
            persisted_payload = {
                "images": [
                    {"id": 1, "file_name": "img_001.jpg", "width": 10, "height": 10},
                ],
                "annotations": [],
                "categories": [{"id": 1, "name": "car"}],
            }
            annotations_path.write_text(json.dumps(persisted_payload), encoding="utf-8")
            for name in ("img_001.jpg", "img_002.jpg", "img_003.jpg"):
                cv2.imwrite(str(images_dir / name), np.zeros((10, 10, 3), dtype=np.uint8))

            class DummyExportActions(ExportActionsMixin):
                def autosave_current_frame(self, *, reason=""):
                    self.autosave_reason = reason

                def write_annotations(self, *, blocking: bool = False):
                    self.write_called = True
                    self.write_blocking = blocking

            actions = DummyExportActions()
            actions.annotations_path = annotations_path
            actions.output_images_dir = images_dir
            actions.images = [{"id": 99, "file_name": "only_memory.jpg"}]

            payload = actions.load_export_payload_from_state()

            self.assertTrue(actions.write_called)
            self.assertEqual(actions.autosave_reason, "exportar dataset")
            self.assertEqual(len(payload["images"]), 3)
            self.assertEqual(
                sorted(image["file_name"] for image in payload["images"]),
                ["img_001.jpg", "img_002.jpg", "img_003.jpg"],
            )


class ResumeStateTest(unittest.TestCase):
    def test_resume_state_rewinds_one_position_to_reopen_active_image(self):
        class DummySource(SourceHelpersMixin):
            def __init__(self):
                self.annotation_state = {
                    "last_active_source_index": 0,
                    "last_active_frame_index": 21,
                    "last_active_file_name": "img_021.jpg",
                }

            def _find_last_saved_frame(self):
                return 30

        source = DummySource()

        self.assertEqual(source._resolve_resume_frame_index(0), 20)

    def test_resume_state_falls_back_when_source_does_not_match(self):
        class DummySource(SourceHelpersMixin):
            def __init__(self):
                self.annotation_state = {
                    "last_active_source_index": 1,
                    "last_active_frame_index": 21,
                }

            def _find_last_saved_frame(self):
                return 30

        source = DummySource()

        self.assertEqual(source._resolve_resume_frame_index(0), 30)

    def test_resume_image_cursor_uses_persisted_file_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            image_paths = [data_root / f"img_{idx:03d}.jpg" for idx in range(1, 4)]

            class DummySource(SourceHelpersMixin):
                def __init__(self):
                    self.annotation_state = {"last_active_file_name": "img_002.jpg"}
                    self.current_image_paths = image_paths
                    self.data_root = data_root
                    self.video_path = data_root
                    self.frame_index = 0

            source = DummySource()

            self.assertEqual(source._find_resume_image_cursor(), 1)


class PanZoomMathTest(unittest.TestCase):
    def test_clamp_recenters_image_when_smaller_than_viewport(self):
        class DummyCanvas(DisplayCanvasMixin):
            def __init__(self):
                self.zoom_pan_x = 120
                self.zoom_pan_y = -80

        canvas = DummyCanvas()

        canvas.clamp_zoom_pan(200, 150, 320, 240, 60, 45)

        self.assertEqual(canvas.zoom_pan_x, 0)
        self.assertEqual(canvas.zoom_pan_y, 0)

    def test_clamp_limits_pan_when_image_is_larger_than_viewport(self):
        class DummyCanvas(DisplayCanvasMixin):
            def __init__(self):
                self.zoom_pan_x = -999
                self.zoom_pan_y = 999

        canvas = DummyCanvas()

        canvas.clamp_zoom_pan(800, 600, 320, 240, -240, -180)

        self.assertEqual(canvas.zoom_pan_x, -240)
        self.assertEqual(canvas.zoom_pan_y, 180)


class MergeYoloSplitsTest(unittest.TestCase):
    def test_merges_all_splits_into_train_only_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_root = tmp_path / "yolo_dataset"
            output_root = tmp_path / "yolo_dataset_train_only"

            for split in ("train", "val", "test"):
                (input_root / "images" / split).mkdir(parents=True, exist_ok=True)
                (input_root / "labels" / split).mkdir(parents=True, exist_ok=True)

            (input_root / "images" / "train" / "img_train.jpg").write_bytes(b"train-image")
            (input_root / "labels" / "train" / "img_train.txt").write_text(
                "0 0.500000 0.500000 0.200000 0.200000\n",
                encoding="utf-8",
            )
            (input_root / "images" / "val" / "img_val.jpg").write_bytes(b"val-image")
            (input_root / "labels" / "val" / "img_val.txt").write_text("", encoding="utf-8")
            (input_root / "images" / "test" / "img_test.jpg").write_bytes(b"test-image")
            (input_root / "labels" / "test" / "img_test.txt").write_text("", encoding="utf-8")
            (input_root / "data.yaml").write_text(
                "\n".join(
                    [
                        f"path: {input_root}",
                        "train: images/train",
                        "val: images/val",
                        "test: images/test",
                        "",
                        "names:",
                        "  0: car",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = merge_yolo_splits(input_root, output_root)

            self.assertEqual(report["merged_counts"], {"train": 1, "val": 1, "test": 1})
            self.assertEqual(report["total_images"], 3)
            self.assertEqual(report["empty_label_files"], 2)
            self.assertTrue((output_root / "images" / "train" / "img_train.jpg").exists())
            self.assertTrue((output_root / "images" / "train" / "img_val.jpg").exists())
            self.assertTrue((output_root / "images" / "train" / "img_test.jpg").exists())
            self.assertEqual(
                (output_root / "labels" / "train" / "img_val.txt").read_text(encoding="utf-8"),
                "",
            )
            output_yaml = (output_root / "data.yaml").read_text(encoding="utf-8")
            self.assertIn("train: images/train", output_yaml)
            self.assertNotIn("val: images/val", output_yaml)
            self.assertNotIn("test: images/test", output_yaml)


if __name__ == "__main__":
    unittest.main()
