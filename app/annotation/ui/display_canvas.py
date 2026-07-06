from app.annotation.shared import *
from app.annotation.ui.rotation_utils import (
    apply_frame_rotation,
    image_to_rotated,
    rotated_to_image,
    rotated_dims,
)


class DisplayCanvasMixin:
    def image_to_canvas_coords(self, x: float, y: float) -> Tuple[int, int]:
        rotation = getattr(self, "frame_rotation", 0)
        if rotation and self.current_frame is not None:
            orig_h, orig_w = self.current_frame.shape[:2]
            x, y = image_to_rotated(x, y, orig_w, orig_h, rotation)
        cx = int(round(x * self.display_scale + self.offset_x))
        cy = int(round(y * self.display_scale + self.offset_y))
        return cx, cy

    def canvas_to_image_coords(self, canvas_x: int, canvas_y: int) -> Optional[Tuple[int, int]]:
        if self.current_frame is None:
            return None
        orig_h, orig_w = self.current_frame.shape[:2]
        rotation = getattr(self, "frame_rotation", 0)
        rot_w, rot_h = rotated_dims(orig_w, orig_h, rotation)

        x = (canvas_x - self.offset_x) / max(self.display_scale, 1e-9)
        y = (canvas_y - self.offset_y) / max(self.display_scale, 1e-9)
        if x < 0 or y < 0 or x >= rot_w or y >= rot_h:
            return None
        rx = int(np.clip(x, 0, rot_w - 1))
        ry = int(np.clip(y, 0, rot_h - 1))
        if rotation:
            ox, oy = rotated_to_image(rx, ry, orig_w, orig_h, rotation)
            return int(np.clip(ox, 0, orig_w - 1)), int(np.clip(oy, 0, orig_h - 1))
        return rx, ry

    def _canvas_viewport_limits(self) -> Tuple[int, int, int, int]:
        screen_w = self.window.winfo_screenwidth()
        screen_h = self.window.winfo_screenheight()
        available_w = self.canvas_frame.winfo_width() if hasattr(self, "canvas_frame") else 0
        available_h = self.canvas_frame.winfo_height() if hasattr(self, "canvas_frame") else 0
        max_canvas_w = max(320, min(screen_w - WINDOW_MARGIN_PX, available_w or screen_w - WINDOW_MARGIN_PX))
        max_canvas_h = max(240, min(screen_h - WINDOW_TOP_RESERVED_PX, available_h or screen_h - WINDOW_TOP_RESERVED_PX))
        return max_canvas_w, max_canvas_h, screen_w, screen_h

    def clamp_zoom_pan(
        self,
        disp_w: int,
        disp_h: int,
        canvas_w: int,
        canvas_h: int,
        base_offset_x: int,
        base_offset_y: int,
    ):
        if disp_w <= canvas_w:
            self.zoom_pan_x = 0
        else:
            min_pan_x = canvas_w - disp_w - base_offset_x
            max_pan_x = -base_offset_x
            self.zoom_pan_x = int(np.clip(self.zoom_pan_x, min_pan_x, max_pan_x))

        if disp_h <= canvas_h:
            self.zoom_pan_y = 0
        else:
            min_pan_y = canvas_h - disp_h - base_offset_y
            max_pan_y = -base_offset_y
            self.zoom_pan_y = int(np.clip(self.zoom_pan_y, min_pan_y, max_pan_y))

    def update_display(self, *, refresh_status: bool = False):
        if self.current_frame is None:
            return

        annotated = self.current_frame.copy()
        annotated = self._draw_roi_overlay_on_frame(annotated)
        self.validate_selected_detection()

        if SHOW_MODEL_DETECTIONS:
            annotated = self.draw_detections(annotated, self.current_detections, "model")
        if SHOW_MANUAL_DETECTIONS:
            annotated = self.draw_detections(annotated, self.manual_detections, "manual")

        rotation = getattr(self, "frame_rotation", 0)
        if rotation:
            annotated = apply_frame_rotation(annotated, rotation)

        frame_h, frame_w = annotated.shape[:2]
        max_canvas_w, max_canvas_h, screen_w, screen_h = self._canvas_viewport_limits()
        disp_w, disp_h = self._compute_display_size(frame_w, frame_h, max_canvas_w, max_canvas_h)
        if disp_w != frame_w or disp_h != frame_h:
            interp = cv2.INTER_AREA if self.display_scale < 1.0 else cv2.INTER_LINEAR
            annotated = cv2.resize(annotated, (disp_w, disp_h), interpolation=interp)
        self._render_frame_on_canvas(annotated, disp_w, disp_h, max_canvas_w, max_canvas_h, screen_w, screen_h)
        self._draw_roi_overlay_on_canvas()
        self._draw_active_manual_rectangle()

        self.last_frame_shape = (frame_w, frame_h)
        if refresh_status:
            self.update_status()

    def _draw_roi_overlay_on_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self.roi_points:
            return frame
        pts = np.array(self.roi_points, dtype=np.int32)
        is_closed = len(self.roi_points) == 4
        cv2.polylines(frame, [pts], isClosed=is_closed, color=(255, 0, 0), thickness=2)
        for idx, (x, y) in enumerate(self.roi_points):
            xi, yi = int(round(x)), int(round(y))
            cv2.circle(frame, (xi, yi), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(idx + 1), (xi + 4, yi - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return frame

    def _compute_display_size(
        self, frame_w: int, frame_h: int, max_canvas_w: int, max_canvas_h: int
    ) -> Tuple[int, int]:
        fit_scale = min(1.0, max_canvas_w / frame_w, max_canvas_h / frame_h)
        self.display_scale = fit_scale * self.zoom_scale
        disp_w = max(1, int(round(frame_w * self.display_scale)))
        disp_h = max(1, int(round(frame_h * self.display_scale)))
        return disp_w, disp_h

    def _render_frame_on_canvas(
        self,
        frame: np.ndarray,
        disp_w: int,
        disp_h: int,
        max_canvas_w: int,
        max_canvas_h: int,
        screen_w: int,
        screen_h: int,
    ):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        self.tk_image = ImageTk.PhotoImage(image=pil_image, master=self.window)

        self.canvas.delete("all")
        canvas_w = min(max_canvas_w, disp_w + CANVAS_PADDING_PX)
        canvas_h = min(max_canvas_h, disp_h + CANVAS_PADDING_PX)
        base_offset_x = (canvas_w - disp_w) // 2
        base_offset_y = (canvas_h - disp_h) // 2
        self.clamp_zoom_pan(disp_w, disp_h, canvas_w, canvas_h, base_offset_x, base_offset_y)
        self.offset_x = base_offset_x + self.zoom_pan_x
        self.offset_y = base_offset_y + self.zoom_pan_y
        if self.canvas.winfo_width() != canvas_w or self.canvas.winfo_height() != canvas_h:
            self.canvas.config(width=canvas_w, height=canvas_h)
        self.canvas_image_id = self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image)
        self.window.maxsize(screen_w, screen_h)
