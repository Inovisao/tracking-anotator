from app.annotation_keypoint.shared import *


class _NoopTracker:
    def reset(self):
        return None


class KPRuntimeStateMixin:
    # kp_instances aliases manual_detections so the shared class-service helpers
    # (which mutate manual_detections) stay in sync with the keypoint runtime.
    @property
    def kp_instances(self) -> List[KeypointInstance]:
        return self.manual_detections

    @kp_instances.setter
    def kp_instances(self, value: List[KeypointInstance]):
        self.manual_detections = value

    def _initialize_runtime_state(self):
        self.default_tracker_args = ByteTrackerArgs()
        self.frame_rate = 30
        self.bytetracker = _NoopTracker()
        self.multiclass_tracker = _NoopTracker()

        self.cap = None
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
        self.tk_image = None
        self.last_frame_shape: Optional[Tuple[int, int]] = None
        self.offset_x = 0
        self.offset_y = 0
        self.display_scale = 1.0
        self.zoom_scale = 1.0
        self.zoom_pan_x = 0
        self.zoom_pan_y = 0
        self.pan_mode = False
        self.pan_drag_start: Optional[Tuple[int, int]] = None
        self.pan_start_offset: Tuple[int, int] = (0, 0)
        self.canvas_image_id = None
        self.key_mapping_mode = "arrows"
        self.frame_rotation: int = 0

        self.roi_points: List[Tuple[int, int]] = []
        self.roi_defined = False
        self.roi_capture_mode = False
        self.homography_matrix: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None
        self.warp_size: Optional[Tuple[int, int]] = None
        self.roi_polygon: Optional[np.ndarray] = None
        self.dest_points: Optional[np.ndarray] = None

        self.images: List[dict] = []
        self.annotations: List[dict] = []
        self.annotation_state: dict = {}
        self.homographies: List[dict] = []

        self.max_undo_states = 40
        self.undo_stack: deque = deque(maxlen=self.max_undo_states)

        # Finalized instances + the one currently being placed point by point.
        self.kp_instances: List[KeypointInstance] = []
        self.wip_instance: Optional[KeypointInstance] = None
        self.wip_index = 0
        self.next_visibility = V_VISIBLE
        self.kp_label_font_size = 9  # on-image label size (adjustable with +/-)
        self.cursor_image_pos: Optional[Tuple[int, int]] = None
        # Aliases so the generic class-service helpers can edit instances.
        self.current_detections: List[KeypointInstance] = []
        self.manual_detections = self.kp_instances

        self.annotation_mode = True
        self.remove_mode = False
        self.selection_mode = False
        self.edit_id_mode = False
        self.selected_instance: Optional[int] = None
        self.selected_kp: Optional[int] = None
        self.selected_detection = None
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drawing_start: Optional[Tuple[int, int]] = None
        self.drawing_rect_id: Optional[int] = None

        self.saved_records: List[dict] = []
        self.review_idx: Optional[int] = None
        self.live_snapshot: Optional[dict] = None
        self.closed = False
