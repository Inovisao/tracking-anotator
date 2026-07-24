from app.annotation_obb.shared import *
from app.annotation.infrastructure.persistence.async_writer import AnnotationsAsyncWriterMixin


class OBBCocoStorageMixin(AnnotationsAsyncWriterMixin):
    _annotations_log_label = "Anotacoes OBB"

    def detections_to_save(self) -> List[OBBDetection]:
        return list(self.current_obb_detections) + list(self.manual_obb_detections)

    def current_frame_file_name(self) -> Optional[str]:
        if self.current_frame is None:
            return None
        try:
            return self.build_output_file_name(new_frame=True, existing_file_name=None)
        except Exception:  # pylint: disable=broad-except
            return None

    def find_image_record_by_file_name(self, file_name: str) -> Optional[dict]:
        for image in self.images:
            if str(image.get("file_name", "")) == file_name:
                return image
        return None

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
                "description": "OBB oriented annotation with HBB compatibility bbox",
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

    def annotation_to_obb(self, ann: dict) -> Optional[OBBDetection]:
        try:
            category_id = int(ann.get("category_id", 1))
            confidence = float(ann.get("score", 1.0))
            source = str(ann.get("source", "manual"))
            obb = ann.get("obb")
            if isinstance(obb, dict):
                return OBBDetection(
                    cx=float(obb["cx"]),
                    cy=float(obb["cy"]),
                    width=float(obb["width"]),
                    height=float(obb["height"]),
                    angle=float(obb.get("angle", 0.0)),
                    category_id=category_id,
                    confidence=confidence,
                    source=source,
                )
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            return hbb_to_obb(x, y, w, h, category_id=category_id, confidence=confidence, source=source)
        except Exception:  # pylint: disable=broad-except
            return None

    def obb_to_annotation(self, det: OBBDetection, image_id: int, annotation_id: int) -> dict:
        points = obb_to_points(det.cx, det.cy, det.width, det.height, det.angle)
        x, y, w, h = points_to_hbb(points)
        return {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": int(det.category_id),
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": obb_area(det),
            "iscrowd": 0,
            "segmentation": [],
            "score": float(det.confidence),
            "source": det.source,
            "video": str(self.video_path) if self.video_path else self.video_name,
            "annotation_type": "obb",
            "obb": {
                "cx": float(det.cx),
                "cy": float(det.cy),
                "width": float(det.width),
                "height": float(det.height),
                "angle": float(normalize_angle(det.angle)),
            },
        }

    def store_annotations(
        self,
        detections: List[OBBDetection],
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
        image_id = self.image_id if new_frame else int(existing_image_id)
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

        self.annotations = [ann for ann in self.annotations if ann.get("image_id") != image_id]
        for det in detections:
            det = clip_obb_to_image(det, width, height)
            if not validate_obb(det):
                continue
            self.annotations.append(self.obb_to_annotation(det, image_id, self.annotation_id))
            self.annotation_id += 1

        if new_frame:
            self.image_id += 1
            self.frames_saved_in_current_video += 1
        return image_id, file_name

    def write_annotations(self, *, blocking: bool = False):
        """Persist the COCO OBB payload — see AnnotationsAsyncWriterMixin for the rationale."""
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

    def load_existing_annotations(self):
        annotations_path = getattr(self, "annotations_path", None)
        if annotations_path is None or not annotations_path.exists():
            return
        try:
            with open(annotations_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[AVISO] Falha ao ler anotacoes OBB existentes: {exc}")
            return
        self.images = data.get("images", [])
        self.annotations = data.get("annotations", [])
        state = data.get("annotation_state", {})
        self.annotation_state = state if isinstance(state, dict) else {}
        cats = data.get("categories")
        if cats:
            self.categories = cats
            self.class_to_category_id = {}
            self.ensure_category_metadata()
            for cat in self.categories:
                name = str(cat.get("name", "")).strip()
                cid = int(cat.get("id", 0))
                if name and cid > 0:
                    self.class_to_category_id[name] = cid
            if not self.target_classes:
                self.target_classes = [cat["name"] for cat in self.categories if cat.get("name")]
            if self.target_classes_var is not None:
                self.target_classes_var.set(", ".join(self.target_classes))
        self.annotation_id = max((ann.get("id", 0) for ann in self.annotations), default=0) + 1
        self.image_id = max((img.get("id", 0) for img in self.images), default=0) + 1
        print(
            f"[INFO] Anotacoes OBB carregadas. imagens={len(self.images)}, "
            f"anotacoes={len(self.annotations)}, prox_image_id={self.image_id}, "
            f"prox_annotation_id={self.annotation_id}"
        )
