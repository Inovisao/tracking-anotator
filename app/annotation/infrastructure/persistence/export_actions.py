"""Export actions triggered by the UI (COCO, YOLO, full dataset)."""

from datetime import datetime

from app.annotation.shared import *
from app.dataset_export import export_detection_coco_json, export_yolo_dataset, export_yolo_no_split, load_json


class ExportActionsMixin:
    def resolve_user_export_root(self, destination_parent: Path, folder_name: str) -> Path:
        requested = (Path(destination_parent).expanduser() / folder_name).resolve()
        output_dir = self.output_dir.resolve()
        if requested == output_dir or output_dir in requested.parents:
            candidate = output_dir.with_name(f"{output_dir.name}_export")
            if not candidate.exists():
                return candidate
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return candidate.with_name(f"{candidate.name}_{stamp}")
        return requested

    def resolve_export_dataset_path(self, selected_dir: Path) -> Path:
        selected_dir = Path(selected_dir).expanduser()
        output_dir = self.output_dir.resolve()
        selected_resolved = selected_dir.resolve()
        if selected_resolved == output_dir or output_dir in selected_resolved.parents:
            candidate = output_dir.with_name(f"{output_dir.name}_export")
        elif selected_dir.name == self.output_dir.name:
            candidate = selected_dir
        else:
            candidate = selected_dir / self.output_dir.name
        candidate = candidate.resolve()
        if candidate == output_dir or output_dir in candidate.parents:
            candidate = output_dir.with_name(f"{output_dir.name}_export")
        if not candidate.exists():
            return candidate
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return candidate.with_name(f"{candidate.name}_{stamp}")

    def sync_export_metadata(self):
        if self.coco_detection_export_path.exists():
            export_detection_coco_json(self.build_coco_payload(), self.coco_detection_export_path)
        if (self.yolo_dataset_dir / "data.yaml").exists():
            export_yolo_dataset(
                self.build_coco_payload(),
                source_images_dir=self.output_images_dir,
                dataset_root=self.yolo_dataset_dir,
            )

    def on_export_dataset(self):
        self.show_export_screen()

    def load_export_payload_from_state(self) -> dict:
        self.autosave_current_frame(reason="exportar dataset")
        self.write_annotations()
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotation state not found: {self.annotations_path}")
        payload = load_json(self.annotations_path)
        payload = self.reconcile_export_payload_with_state_files(payload)
        if not payload.get("images"):
            raise RuntimeError("No images saved in the current state to export.")
        return payload

    def reconcile_export_payload_with_state_files(self, payload: dict) -> dict:
        payload = dict(payload)
        images = [dict(image) for image in payload.get("images", [])]
        known_files = {str(image.get("file_name", "")).strip() for image in images}
        known_ids = [int(image.get("id", 0) or 0) for image in images]
        next_image_id = max(known_ids, default=0) + 1

        if not self.output_images_dir.exists():
            payload["images"] = images
            return payload

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        for image_path in sorted(self.output_images_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in image_extensions:
                continue
            file_name = image_path.relative_to(self.output_images_dir).as_posix()
            if file_name in known_files:
                continue

            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            height, width = frame.shape[:2]
            images.append(
                {
                    "id": next_image_id,
                    "file_name": file_name,
                    "width": int(width),
                    "height": int(height),
                    "video": "",
                }
            )
            known_files.add(file_name)
            next_image_id += 1

        payload["images"] = images
        return payload

    def _export_yolo_format(self, payload, yolo_root, config, on_progress):
        """Returns (summary_line, exported_part). Override per task to change the YOLO flavor."""
        if config.use_split:
            report = export_yolo_dataset(
                payload,
                source_images_dir=self.output_images_dir,
                dataset_root=yolo_root,
                split_ratios=config.split_ratios,
                augmentation_preset=config.augmentation,
                on_progress=on_progress,
            )
            total_images = sum(report["images_per_split"].values())
            total_labels = sum(report["labels_per_split"].values())
            split_text = " ".join(f"{name}={count}" for name, count in report["images_per_split"].items())
            return f"YOLO: {total_images} imagens, {total_labels} labels", f"YOLO {split_text}"
        report = export_yolo_no_split(
            payload,
            source_images_dir=self.output_images_dir,
            dataset_root=yolo_root,
            augmentation_preset=config.augmentation,
            on_progress=on_progress,
        )
        return (
            f"YOLO: {report['total_images']} imagens, {report['total_labels']} labels",
            f"YOLO all={report['total_images']} imgs",
        )

    def _export_coco_format(self, payload, coco_dir, on_progress):
        """Returns (summary_line, exported_part). Override per task to change the COCO flavor."""
        coco_path = coco_dir / "annotations.coco.json"
        converted = export_detection_coco_json(
            payload,
            coco_path,
            source_images_dir=self.output_images_dir,
            on_progress=on_progress,
        )
        return f"COCO: {len(converted['images'])} imagens", f"COCO imgs={len(converted['images'])}"

    def perform_dataset_export(self, config, cancel_event=None):
        multi_format = len(config.formats) > 1
        export_root = self.resolve_user_export_root(config.destination_parent, config.folder_name)
        try:
            payload = self.load_export_payload_from_state()
            exported_parts: list = []
            yolo_summary = ""
            coco_summary = ""

            n_images = len(payload.get("images", []))
            n_formats = len(config.formats)
            total_steps = n_images * n_formats
            steps_done = [0]

            def _progress(done_in_format: int, offset: int = 0):
                if cancel_event and cancel_event.is_set():
                    raise InterruptedError("Export cancelled by user.")
                steps_done[0] = offset + done_in_format
                if hasattr(self, "update_export_progress"):
                    pct = int(steps_done[0] * 100 / max(total_steps, 1))
                    self._post_to_main(lambda p=pct: self.update_export_progress(p))

            yolo_offset = 0
            coco_offset = n_images if "yolo" in config.formats else 0

            if "yolo" in config.formats:
                yolo_root = export_root / "yolo" if multi_format else export_root
                yolo_summary, part = self._export_yolo_format(
                    payload, yolo_root, config, lambda d, _t: _progress(d, yolo_offset)
                )
                exported_parts.append(part)

            if "coco" in config.formats:
                coco_dir = export_root / "coco" if multi_format else export_root
                coco_summary, part = self._export_coco_format(
                    payload, coco_dir, lambda d, _t: _progress(d, coco_offset)
                )
                exported_parts.append(part)

            summary_lines = [line for line in (yolo_summary, coco_summary) if line]
            message = f"Dataset exportado com sucesso em: {export_root}"
            if summary_lines:
                message += " | " + " | ".join(summary_lines)
            print(f"[INFO] {message}")
            def _on_success(msg=message, root=export_root, parts=exported_parts, cfg=config):
                self.info_var.set(msg)
                if hasattr(self, "set_export_status"):
                    self.set_export_status(root, parts, cfg)
            self._post_to_main(_on_success)
        except InterruptedError:
            print("[INFO] Export cancelled by user.")
        except Exception as exc:  # pylint: disable=broad-except
            message = f"Falha ao exportar dataset: {exc}"
            print(f"[ERRO] {message}")
            def _on_error(msg=message):
                self.info_var.set(msg)
                if hasattr(self, "set_export_error"):
                    self.set_export_error(msg)
            self._post_to_main(_on_error)
