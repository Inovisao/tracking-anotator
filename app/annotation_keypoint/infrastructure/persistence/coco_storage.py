from app.annotation_keypoint.shared import *


class KPCocoStorageMixin:
    def ensure_keypoint_metadata(self):
        for cat in self.categories:
            if not isinstance(cat.get("skeleton"), list):
                cat["skeleton"] = []
            names = cat.get("keypoints")
            if not isinstance(names, list):
                names = []
            # Never leave keypoints empty when annotations exist — derive a
            # consistent N so categories.keypoints matches the annotation data.
            if not names:
                count = self._max_keypoints_for_category(int(cat.get("id", 0)))
                names = [f"point_{i + 1}" for i in range(count)]
            cat["keypoints"] = names

    def _max_keypoints_for_category(self, category_id: int) -> int:
        counts = [
            len(ann.get("keypoints", [])) // 3
            for ann in self.annotations
            if int(ann.get("category_id", -1)) == category_id
        ]
        return max(counts, default=0)

    def keypoint_dataset_errors(self) -> List[str]:
        """Validation issues that would make a pose dataset inconsistent."""
        counts_by_cat: Dict[int, set] = {}
        for ann in self.annotations:
            cid = int(ann.get("category_id", -1))
            counts_by_cat.setdefault(cid, set()).add(len(ann.get("keypoints", [])) // 3)
        name_by_id = self.category_name_by_id()
        errors = []
        for cat in self.categories:
            cid = int(cat.get("id", 0))
            counts = counts_by_cat.get(cid)
            if not counts:
                continue
            declared = len(cat.get("keypoints", []))
            if declared == 0:
                errors.append(f"Classe '{name_by_id.get(cid)}' sem keypoints declarados.")
            if len(counts) > 1:
                errors.append(
                    f"Classe '{name_by_id.get(cid)}' tem instancias com nº de keypoints diferentes: "
                    f"{sorted(counts)}."
                )
        return errors

    def keypoint_names_for_category(self, category_id: int) -> List[str]:
        for cat in self.categories:
            if int(cat.get("id", 0)) == int(category_id):
                return [str(n) for n in cat.get("keypoints", [])]
        return []

    def detections_to_save(self) -> List[KeypointInstance]:
        self.commit_wip_instance()
        return list(self.kp_instances)

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
        self.ensure_keypoint_metadata()
        return {
            "info": {
                "description": "COCO keypoints annotation",
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

    def annotation_to_instance(self, ann: dict) -> Optional[KeypointInstance]:
        try:
            flat = ann.get("keypoints") or []
            kps = [
                [float(flat[i]), float(flat[i + 1]), int(flat[i + 2])]
                for i in range(0, len(flat) - 2, 3)
            ]
            return KeypointInstance(
                category_id=int(ann.get("category_id", 1)),
                keypoints=kps,
                confidence=float(ann.get("score", 1.0)),
                source=str(ann.get("source", "manual")),
            )
        except Exception:  # pylint: disable=broad-except
            return None

    def instance_to_annotation(self, inst: KeypointInstance, image_id: int, annotation_id: int) -> dict:
        x, y, w, h = keypoints_bbox(inst)
        flat: List[float] = []
        for kp in inst.keypoints:
            flat.extend([float(kp[0]), float(kp[1]), int(kp[2])])
        return {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": int(inst.category_id),
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": instance_area(inst),
            "iscrowd": 0,
            "num_keypoints": inst.num_keypoints(),
            "keypoints": flat,
            "score": float(inst.confidence),
            "source": inst.source,
            "video": str(self.video_path) if self.video_path else self.video_name,
            "annotation_type": "keypoint",
        }

    @staticmethod
    def _clip_instance(inst: KeypointInstance, width: int, height: int) -> KeypointInstance:
        for kp in inst.keypoints:
            if kp[2] > 0:
                kp[0] = float(np.clip(kp[0], 0, max(width - 1, 0)))
                kp[1] = float(np.clip(kp[1], 0, max(height - 1, 0)))
        return inst

    def store_annotations(
        self,
        detections: List[KeypointInstance],
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
        for inst in detections:
            inst = self._clip_instance(inst, width, height)
            if not validate_instance(inst):
                continue
            self.annotations.append(self.instance_to_annotation(inst, image_id, self.annotation_id))
            self.annotation_id += 1

        if new_frame:
            self.image_id += 1
            self.frames_saved_in_current_video += 1
        return image_id, file_name

    def write_annotations(self):
        self.update_annotation_state()
        data = self.build_coco_payload()
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.annotations_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Anotacoes keypoint atualizadas em {self.annotations_path}")

    def backup_annotations_file(self):
        if not self.annotations_path.exists():
            return None
        backup_path = self.annotations_path.with_name(f"{self.annotations_path.name}.bak")
        shutil.copy2(self.annotations_path, backup_path)
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

    def load_existing_annotations(self):
        annotations_path = getattr(self, "annotations_path", None)
        if annotations_path is None or not annotations_path.exists():
            return
        try:
            with open(annotations_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[AVISO] Falha ao ler anotacoes keypoint existentes: {exc}")
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
            self.ensure_keypoint_metadata()
            for cat in self.categories:
                name = str(cat.get("name", "")).strip()
                cid = int(cat.get("id", 0))
                if name and cid > 0:
                    self.class_to_category_id[name] = cid
            if not self.target_classes:
                self.target_classes = [cat["name"] for cat in self.categories if cat.get("name")]
            if self.target_classes_var is not None:
                self.target_classes_var.set(", ".join(self.target_classes))
        self._prune_missing_images()
        self.annotation_id = max((ann.get("id", 0) for ann in self.annotations), default=0) + 1
        self.image_id = max((img.get("id", 0) for img in self.images), default=0) + 1
        print(
            f"[INFO] Anotacoes keypoint carregadas. imagens={len(self.images)}, "
            f"anotacoes={len(self.annotations)}, prox_image_id={self.image_id}, "
            f"prox_annotation_id={self.annotation_id}"
        )

    def _prune_missing_images(self):
        """Drop records whose image file no longer exists on disk (broken state)."""
        images_dir = self.output_images_dir
        kept = []
        removed_ids = set()
        for img in self.images:
            file_name = str(img.get("file_name", "")).strip()
            if file_name and (images_dir / file_name).exists():
                kept.append(img)
            else:
                removed_ids.add(img.get("id"))
        if not removed_ids:
            return
        self.images = kept
        self.annotations = [a for a in self.annotations if a.get("image_id") not in removed_ids]
        print(f"[INFO] {len(removed_ids)} imagem(ns) ausente(s) removida(s) do estado keypoint.")
