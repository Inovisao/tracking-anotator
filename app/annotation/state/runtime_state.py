from app.annotation.shared import *


class RuntimeStateMixin:
    def _initialize_runtime_state(self):
        from tracker.byte_tracker import BYTETracker  # deferred: pulls torch (~4s)
        from app.tracking import MultiClassByteTracker
        self.default_tracker_args = ByteTrackerArgs()
        self.frame_rate = 30
        self.bytetracker = BYTETracker(self.default_tracker_args, frame_rate=self.frame_rate)
        self.multiclass_tracker = MultiClassByteTracker(self.default_tracker_args, frame_rate=self.frame_rate)
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_source_type = "video"
        self.current_image_paths: List[Path] = []
        self.current_image_cursor = 0
        self.current_source_image_path: Optional[Path] = None

        self.video_name = ""
        self.video_path: Optional[Path] = None
        self.current_video_index = 0
        self.frame_index = 0
        self.image_id = 1
        self.annotation_id = 1
        self.frames_saved_in_current_video = 0

        self.current_frame: Optional[np.ndarray] = None
        self.current_rectified_frame: Optional[np.ndarray] = None
        self.current_detections: List[Detection] = []
        self.manual_detections: List[Detection] = []
        self.tk_image = None
        self.last_frame_shape: Optional[Tuple[int, int]] = None

        self.annotation_mode = True
        self.remove_mode = False
        self.selection_mode = False
        self.drawing_start: Optional[Tuple[int, int]] = None
        self.drawing_rect_id: Optional[int] = None
        self.canvas_image_id: Optional[int] = None

        self.images: List[dict] = []
        self.annotations: List[dict] = []
        self.annotation_state: dict = {}
        self.homographies: List[dict] = []
        self.roi_points: List[Tuple[int, int]] = []
        self.roi_defined = False
        self.roi_capture_mode = False
        self.homography_matrix: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None
        self.warp_size: Optional[Tuple[int, int]] = None
        self.roi_polygon: Optional[np.ndarray] = None
        self.dest_points: Optional[np.ndarray] = None

        self.manual_track_memory: Dict[int, Dict[str, np.ndarray]] = {}
        self.global_track_counter = 1
        self.track_history: Dict[int, List[dict]] = {}
        self.history_window = 5
        self.recent_tracks: deque = deque(maxlen=self.history_window)
        self.tracker_id_map: Dict[Tuple[int, int], int] = {}
        self.edit_id_mode = False
        self.selected_detection: Optional[Tuple[str, int]] = None
        self.offset_x = 0
        self.offset_y = 0
        self.display_scale = 1.0
        self.zoom_scale: float = 1.0
        self.zoom_pan_x: int = 0
        self.zoom_pan_y: int = 0
        self.pan_mode = False
        self.pan_drag_start: Optional[Tuple[int, int]] = None
        self.pan_start_offset: Tuple[int, int] = (0, 0)
        self.key_mapping_mode = "arrows"
        self.frame_rotation: int = 0  # 0 / 90 / 180 / 270 — visual only

        self.saved_records: List[dict] = []
        self.review_idx: Optional[int] = None
        self.live_snapshot: Optional[dict] = None
        self.max_undo_states = 40
        self.undo_stack: deque = deque(maxlen=self.max_undo_states)
        self.closed = False
