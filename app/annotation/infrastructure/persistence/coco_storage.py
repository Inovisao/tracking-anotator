"""Read/write operations for the COCO payload in memory and on disk."""

from app.annotation.shared import *
from app.annotation.infrastructure.persistence.async_writer import AnnotationsAsyncWriterMixin


class CocoStorageMixin(AnnotationsAsyncWriterMixin):
    # ── Index caches — invalidated whenever self.images or self.annotations changes ──
    _image_index: Optional[Dict[str, dict]] = None          # file_name → image record
    _annotation_index: Optional[Dict[int, List[dict]]] = None  # image_id → annotations

    def _invalidate_storage_cache(self):
        self._image_index = None
        self._annotation_index = None

    def _build_image_index(self) -> Dict[str, dict]:
        idx: Dict[str, dict] = {}
        for image in self.images:
            fn = str(image.get("file_name", ""))
            if fn and fn not in idx:
                idx[fn] = image
        return idx

    def _build_annotation_index(self) -> Dict[int, List[dict]]:
        idx: Dict[int, List[dict]] = {}
        for ann in self.annotations:
            iid = ann.get("image_id")
            if iid is not None:
                idx.setdefault(int(iid), []).append(ann)
        return idx

    def detections_to_save(self) -> List[Detection]:
        return list(self.current_detections) + list(self.manual_detections)

    def current_frame_file_name(self) -> Optional[str]:
        if self.current_frame is None:
            return None
        try:
            return self.build_output_file_name(new_frame=True, existing_file_name=None)
        except Exception:  # pylint: disable=broad-except
            return None

    def find_image_record_by_file_name(self, file_name: str) -> Optional[dict]:
        if self._image_index is None:
            self._image_index = self._build_image_index()
        return self._image_index.get(file_name)

    def _source_image_output_name(self, source_path: Path) -> str:
        try:
            return source_path.resolve().relative_to(self.data_root.resolve()).as_posix()
        except ValueError:
            pass
        if self.video_path is not None and self.video_path.is_dir():
            try:
                return source_path.resolve().relative_to(self.video_path.resolve()).as_posix()
            except ValueError:
                pass
        return source_path.name

    def build_output_file_name(self, new_frame: bool, existing_file_name: Optional[str]) -> str:
        if not new_frame and existing_file_name is not None:
            return existing_file_name
        if self.current_source_type == "images" and self.current_source_image_path is not None:
            return self._source_image_output_name(self.current_source_image_path)
        return f"{self.video_name}_frame_{self.frame_index:05d}.jpg"

    def update_annotation_state(self):
        if self.current_frame is None:
            return
        self.annotation_state = {
            "last_active_file_name": self.current_frame_file_name(),
            "last_active_frame_index": int(self.frame_index),
            "last_active_source_index": int(self.current_video_index),
            "last_active_source": str(self.video_path) if self.video_path else "",
            "last_active_source_type": str(self.current_source_type),
        }

    def build_coco_payload(self) -> dict:
        self.normalize_category_ids()
        self.ensure_category_metadata()
        return {
            "info": {
                "description": "Validacao manual de deteccoes com ROI e homografia",
                "version": "1.0",
                "task_mode": self.task_mode.value,
                "data_root": str(self.data_root),
                "video_sources": [str(v) for v in self.video_files],
                "frames_are_rectified": SAVE_RECTIFIED_FRAMES,
            },
            "licenses": [],
            "categories": self.categories,
            "images": self.images,
            "annotations": self.annotations,
            "annotation_state": getattr(self, "annotation_state", {}),
        }

    def store_annotations(
        self,
        detections: List[Detection],
        existing_image_id: Optional[int] = None,
        existing_file_name: Optional[str] = None,
    ) -> Tuple[int, str]:
        frame_to_save = (
            self.current_rectified_frame
            if SAVE_RECTIFIED_FRAMES and self.current_rectified_frame is not None
            else self.current_frame
        )
        if frame_to_save is None:
            raise RuntimeError("Frame atual ausente para salvamento.")

        height, width = frame_to_save.shape[:2]
        new_frame = existing_image_id is None
        image_id = self.image_id if new_frame else existing_image_id
        file_name = self.build_output_file_name(new_frame, existing_file_name)
        image_path = self.output_images_dir / file_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), frame_to_save):
            raise RuntimeError(f"Falha ao salvar frame em {image_path}")

        self.images = [img for img in self.images if img.get("id") != image_id]
        self.images.append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "video": str(self.video_path) if self.video_path else self.video_name,
        })
        self._invalidate_storage_cache()

        self.annotations = [ann for ann in self.annotations if ann.get("image_id") != image_id]
        for det in detections:
            chosen_bbox = det.warp_bbox if SAVE_RECTIFIED_FRAMES and det.warp_bbox is not None else det.original_bbox
            chosen_bbox = chosen_bbox.astype(np.float32)
            chosen_bbox = clip_bbox(chosen_bbox[0], chosen_bbox[1], chosen_bbox[2], chosen_bbox[3], width, height)
            x1, y1, x2, y2 = chosen_bbox
            w = x2 - x1
            h = y2 - y1
            annotation = {
                "id": self.annotation_id,
                "image_id": image_id,
                "category_id": det.category_id,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "area": float(max(w, 0.0) * max(h, 0.0)),
                "iscrowd": 0,
                "segmentation": [],
                "score": float(det.confidence),
                "source": det.source,
                "video": str(self.video_path) if self.video_path else self.video_name,
            }
            if self.tracking_enabled and det.track_id is not None:
                annotation["track_id"] = int(det.track_id)
            self.annotations.append(annotation)
            self.annotation_id += 1

        if new_frame:
            self.image_id += 1
            self.frames_saved_in_current_video += 1
        return image_id, file_name

    def write_annotations(self, *, blocking: bool = False):
        """Persist the COCO payload.

        The payload is built here, on the caller's (UI) thread, so the snapshot is
        consistent with in-memory state. Only serialization and disk I/O are handed to a
        background writer — those grow linearly with the dataset and would otherwise
        freeze the UI on every frame change. Pass blocking=True to wait for the write
        (shutdown, export), where the file must be complete before moving on.
        """
        self.update_annotation_state()
        data = self.build_coco_payload()
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        if blocking:
            self._flush_annotations(data)
            return
        self._queue_annotations_write(data)

    def backup_annotations_file(self):
        if not self.annotations_path.exists():
            return None
        backup_path = self.annotations_path.with_name(f"{self.annotations_path.name}.bak")
        shutil.copy2(self.annotations_path, backup_path)
        for stale_backup in self.annotations_path.parent.glob(f"{self.annotations_path.name}.bak_*"):
            stale_backup.unlink()
        return backup_path

    def delete_image_annotations(self, image_id: int) -> int:
        removed = sum(1 for ann in self.annotations if ann.get("image_id") == image_id)
        self.annotations = [ann for ann in self.annotations if ann.get("image_id") != image_id]
        self.images = [img for img in self.images if img.get("id") != image_id]
        self._invalidate_storage_cache()
        return removed

    def remove_image_file(self, file_name: str) -> bool:
        image_path = self.output_images_dir / file_name
        if not image_path.exists():
            return False
        image_path.unlink()
        return True

    def remove_exported_dataset_files(self, file_name: str):
        label_name = Path(file_name).with_suffix(".txt")
        for split in ("train", "val", "test"):
            image_path = self.yolo_dataset_dir / "images" / split / file_name
            label_path = self.yolo_dataset_dir / "labels" / split / label_name
            if image_path.exists():
                image_path.unlink()
            if label_path.exists():
                label_path.unlink()
