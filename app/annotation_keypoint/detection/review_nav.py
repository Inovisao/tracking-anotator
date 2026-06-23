from app.annotation_keypoint.shared import *


class KPReviewNavMixin:
    def append_saved_record(self, detections: List[KeypointInstance], image_id: int, file_name: str):
        if self.current_frame is None:
            return
        # Frames live on disk already; keep them out of RAM and load on demand
        # (review navigation) via _load_record_frame — saves ~MBs per saved frame.
        self.saved_records.append({
            "image_id": image_id,
            "file_name": file_name,
            "frame_index": self.frame_index,
            "frame": None,
            "rectified_frame": None,
            "detections": [clone_keypoint(inst) for inst in detections],
        })
        if len(self.saved_records) > MAX_SAVED_FRAME_CACHE:
            self.saved_records.pop(0)
        self.review_idx = None
        self.live_snapshot = None

    def update_saved_record(self, idx: int, detections: List[KeypointInstance], image_id: int, file_name: str):
        if idx < 0 or idx >= len(self.saved_records) or self.current_frame is None:
            return
        self.saved_records[idx] = {
            "image_id": image_id,
            "file_name": file_name,
            "frame_index": self.frame_index,
            "frame": None,
            "rectified_frame": None,
            "detections": [clone_keypoint(inst) for inst in detections],
        }

    def remember_saved_record(self, detections: List[KeypointInstance], image_id: int, file_name: str):
        for idx, record in enumerate(self.saved_records):
            if record.get("image_id") == image_id or record.get("file_name") == file_name:
                self.update_saved_record(idx, detections, image_id, file_name)
                return
        self.append_saved_record(detections, image_id, file_name)

    def _populate_saved_records_from_state(self):
        for img in self.images:
            file_name = str(img.get("file_name", "")).strip()
            if not file_name:
                continue
            self.saved_records.append({
                "image_id": int(img.get("id", 0)),
                "file_name": file_name,
                "frame_index": 0,
                "frame": None,
                "rectified_frame": None,
                "detections": [],
            })

    def rebuild_detections_from_annotations(self, image_id: int) -> List[KeypointInstance]:
        instances = []
        for ann in self.annotations:
            if ann.get("image_id") != image_id:
                continue
            inst = self.annotation_to_instance(ann)
            if inst is not None:
                instances.append(inst)
        return instances

    def _apply_loaded_instances(self, instances: List[KeypointInstance]):
        self.kp_instances = list(instances)
        self.manual_detections = self.kp_instances
        self.current_detections = []
        self.wip_instance = None
        self.wip_index = 0
        self.selected_instance = None
        self.selected_kp = None
        self.selected_detection = None

    def restore_saved_annotations_for_current_frame(self):
        file_name = self.current_frame_file_name()
        if not file_name:
            return
        record = self.find_image_record_by_file_name(file_name)
        if record is None or self.current_frame is None:
            return
        self._apply_loaded_instances(self.rebuild_detections_from_annotations(int(record["id"])))

    def current_review_record(self) -> Optional[dict]:
        if self.review_idx is None or not self.saved_records:
            return None
        if 0 <= self.review_idx < len(self.saved_records):
            return self.saved_records[self.review_idx]
        return None

    def go_to_saved_frame(self, idx: int):
        if not self.saved_records or idx < 0 or idx >= len(self.saved_records):
            print("[INFO] Nenhum frame salvo para revisar.")
            return
        if self.review_idx is None and self.current_frame is not None:
            self.live_snapshot = {
                "frame": self.current_frame.copy(),
                "rectified": self.current_rectified_frame.copy() if self.current_rectified_frame is not None else None,
                "manual": [clone_keypoint(inst) for inst in self.kp_instances],
                "frame_index": self.frame_index,
                "source_image_path": self.current_source_image_path,
            }
        record = self.saved_records[idx]
        frame = self._load_record_frame(record)
        if frame is None:
            print("[INFO] Frame do registro indisponivel.")
            return
        self.review_idx = idx
        self.frame_index = record["frame_index"]
        self.current_frame = frame.copy()
        self.current_rectified_frame = record.get("rectified_frame")
        self._apply_loaded_instances(self.rebuild_detections_from_annotations(record["image_id"]))
        self.update_display(refresh_status=True)

    def _load_record_frame(self, record: dict) -> Optional[np.ndarray]:
        # Load from disk without caching back, so reviewing many frames does not
        # grow RAM — go_to_saved_frame keeps only the single current frame.
        if record.get("frame") is not None:
            return record["frame"]
        file_name = record.get("file_name", "")
        if not file_name:
            return None
        return cv2.imread(str(self.output_images_dir / file_name))

    def on_prev_saved(self):
        self.autosave_current_frame(reason="antes de voltar")
        if not self.saved_records or self.review_idx is None:
            self.load_previous_frame()
            return
        self.go_to_saved_frame(max(0, self.review_idx - 1))

    def on_next_saved(self):
        self.autosave_current_frame(reason="antes de avancar")
        if self.review_idx is None:
            self.load_next_frame()
            return
        next_idx = self.review_idx + 1
        if next_idx < len(self.saved_records):
            self.go_to_saved_frame(next_idx)
        else:
            self.exit_review_mode()

    def load_previous_frame(self):
        if self.current_source_type == "images":
            target_cursor = max(0, self.current_image_cursor - 2)
            self.current_image_cursor = target_cursor
            self.frame_index = target_cursor
            frame = self.read_next_image_frame()
            if frame is not None:
                self.process_current_frame(frame, advance_index=True, render=False)
                self.restore_saved_annotations_for_current_frame()
                self.update_display(refresh_status=True)
            return
        print("[INFO] Fonte atual nao suporta voltar no modo keypoint MVP.")

    def exit_review_mode(self):
        if self.live_snapshot is not None:
            snap = self.live_snapshot
            self.frame_index = snap["frame_index"]
            self.current_frame = snap["frame"]
            self.current_rectified_frame = snap["rectified"]
            self._apply_loaded_instances(snap["manual"])
            self.current_source_image_path = snap.get("source_image_path")
            self.live_snapshot = None
            self.review_idx = None
            self.update_display(refresh_status=True)
            return
        self.review_idx = None
        self.load_next_frame()

    def advance_after_review_accept(self):
        if self.review_idx is None:
            return
        next_idx = self.review_idx + 1
        if next_idx < len(self.saved_records):
            self.go_to_saved_frame(next_idx)
        else:
            self.exit_review_mode()
