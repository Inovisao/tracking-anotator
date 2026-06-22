from app.annotation_keypoint.shared import *


class KPFramePipelineMixin:
    def process_current_frame(self, frame: np.ndarray, advance_index: bool = True, *, render: bool = True):
        if frame is None:
            return
        self.review_idx = None
        self.live_snapshot = None
        self.zoom_scale = 1.0
        self.zoom_pan_x = 0
        self.zoom_pan_y = 0
        self.frame_rotation = 0

        if advance_index:
            self.frame_index += 1
        self.current_frame = frame
        self.current_rectified_frame = self.warp_frame(frame)
        self.reset_frame_instances()
        self.undo_stack = deque(maxlen=self.max_undo_states)
        self.annotation_mode = True
        self.remove_mode = False
        self.selection_mode = False
        self.pan_mode = False
        self.drag_start = None
        self.update_annotation_button()
        self.update_remove_button()
        self.update_selection_button()
        if render:
            self.update_display(refresh_status=True)

    def reset_frame_instances(self):
        self.kp_instances = []
        self.manual_detections = self.kp_instances
        self.current_detections = []
        self.wip_instance = None
        self.wip_index = 0
        self.next_visibility = V_VISIBLE
        self.cursor_image_pos = None
        self.selected_instance = None
        self.selected_kp = None
        self.selected_detection = None

    def load_next_frame(self):
        if self.review_idx is not None:
            return
        self.autosave_current_frame(reason="antes de trocar frame")

        if self.current_source_type == "video":
            if self.cap is None:
                self.finish_current_video()
                return
            ret, frame = self.cap.read()
            if not ret:
                self.finish_current_video()
                return
            self.current_source_image_path = None
        else:
            frame = self.read_next_image_frame()
            if frame is None:
                self.finish_current_video()
                return

        self.process_current_frame(frame, render=False)
        self.restore_saved_annotations_for_current_frame()
        self.update_display(refresh_status=True)

    def run_model(self, original_frame: np.ndarray) -> List[KeypointInstance]:
        # MVP: keypoints are placed manually; no automatic inference.
        return []
