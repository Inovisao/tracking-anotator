from app.annotation_keypoint.shared import *


class KPSourceHelpersMixin:
    def _reset_open_source(self):
        if self.cap is not None:
            self.cap.release()

    def _reset_model_tracking_state(self):
        return None

    def _prepare_source_state(self, index: int):
        self.video_path = self.video_files[index]
        self.video_name = self.video_path.stem
        self.current_video_index = index
        self.frame_index = self._resolve_resume_frame_index(index)
        self.frames_saved_in_current_video = self.frame_index
        self.saved_records.clear()
        self.review_idx = None
        self.live_snapshot = None
        self.selected_instance = None
        self.selected_kp = None
        self.selected_detection = None

    def _find_last_saved_frame(self) -> int:
        if self.video_path is None:
            return 0
        saved_for_video = [img for img in self.images if img.get("video") in (str(self.video_path), self.video_name)]
        if not self.is_video_source(self.video_path):
            return len(saved_for_video)
        last_frame_saved = 0
        for img in saved_for_video:
            parsed = parse_frame_number_from_name(img.get("file_name", ""), self.video_name)
            if parsed is not None:
                last_frame_saved = max(last_frame_saved, parsed)
        return last_frame_saved

    def _resolve_resume_frame_index(self, source_index: int) -> int:
        state = getattr(self, "annotation_state", {}) or {}
        try:
            saved_source_index = int(state.get("last_active_source_index", -1))
        except (TypeError, ValueError):
            saved_source_index = -1
        if saved_source_index != source_index:
            return self._find_last_saved_frame()
        try:
            saved_frame_index = int(state.get("last_active_frame_index", 0) or 0)
        except (TypeError, ValueError):
            saved_frame_index = 0
        if saved_frame_index <= 0:
            return self._find_last_saved_frame()
        return max(0, saved_frame_index - 1)

    def _reset_source_runtime_data(self):
        self.roi_points = []
        self.roi_defined = False
        self.roi_capture_mode = False
        self.homography_matrix = None
        self.inverse_homography = None
        self.warp_size = None
        self.roi_polygon = None
        self.dest_points = None
        self.current_rectified_frame = None
        self.reset_frame_instances()

    def _load_first_frame_for_source(self) -> Optional[np.ndarray]:
        if self.video_path is None:
            return None
        if self.is_video_source(self.video_path):
            return self._load_first_frame_from_video()
        return self._load_first_frame_from_images()

    def _load_first_frame_from_video(self) -> Optional[np.ndarray]:
        if self.video_path is None:
            return None
        self.current_source_type = "video"
        self.current_image_paths = []
        self.current_image_cursor = 0
        self.current_source_image_path = None
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            print(f"[ERRO] Falha ao abrir video: {self.video_path}")
            return None
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_rate = int(fps) if fps and fps > 1 else 30
        if self.frame_index > 0:
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(self.frame_index))
            except Exception:  # pylint: disable=broad-except
                pass
        ret, first_frame = self.cap.read()
        if not ret or first_frame is None:
            print(f"[ERRO] Falha ao ler o primeiro frame: {self.video_path}")
            return None
        return first_frame

    def _load_first_frame_from_images(self) -> Optional[np.ndarray]:
        if self.video_path is None:
            return None
        self.current_source_type = "images"
        self.cap = None
        self.current_image_paths = self.build_image_sequence(self.video_path)
        resume_cursor = self._find_resume_image_cursor()
        if resume_cursor is None:
            self.frame_index = self._find_last_saved_frame()
            resume_cursor = self.frame_index
        self.current_image_cursor = resume_cursor
        self.frame_rate = 30
        if not self.current_image_paths:
            print(f"[ERRO] Nenhuma imagem valida encontrada para: {self.video_path}")
            return None
        first_frame = self.read_next_image_frame()
        if first_frame is None:
            print(f"[ERRO] Falha ao ler imagens da fonte: {self.video_path}")
            return None
        return first_frame

    def _find_resume_image_cursor(self) -> Optional[int]:
        state = getattr(self, "annotation_state", {}) or {}
        target_name = str(state.get("last_active_file_name", "") or "")
        if not target_name:
            return self.frame_index
        for idx, image_path in enumerate(self.current_image_paths):
            if self._source_image_output_name(image_path) == target_name:
                return idx
        return None

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
