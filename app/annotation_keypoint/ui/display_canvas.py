from app.annotation_keypoint.shared import *
from app.annotation.ui.rotation_utils import apply_frame_rotation


class KPDisplayCanvasMixin:
    def update_display(self, *, refresh_status: bool = False):
        if self.current_frame is None:
            return
        # Re-rasterize the base image only when frame/zoom/pan/rotation changes.
        # Annotation edits keep the same signature → only the vector overlay is
        # redrawn, which makes clicking/dragging fluid even on large images.
        sig = (
            id(self.current_frame),
            round(self.zoom_scale, 4),
            self.zoom_pan_x,
            self.zoom_pan_y,
            getattr(self, "frame_rotation", 0),
        )
        if sig != getattr(self, "_last_base_sig", None):
            self._render_base_image()
            self._last_base_sig = sig
        self.validate_selected_detection()
        self.render_overlays()
        if refresh_status:
            self.update_status()

    def _render_base_image(self):
        base = self.current_frame
        rotation = getattr(self, "frame_rotation", 0)
        if rotation:
            base = apply_frame_rotation(base, rotation)
        frame_h, frame_w = base.shape[:2]
        max_w, max_h, screen_w, screen_h = self._canvas_viewport_limits()
        disp_w, disp_h = self._compute_display_size(frame_w, frame_h, max_w, max_h)  # sets display_scale
        self._render_viewport(base, frame_w, frame_h, disp_w, disp_h, max_w, max_h, screen_w, screen_h)
        self._draw_roi_overlay_on_canvas()  # ROI drawn as canvas items, not baked in
        self.last_frame_shape = (frame_w, frame_h)

    def _render_viewport(self, base, frame_w, frame_h, disp_w, disp_h, max_w, max_h, screen_w, screen_h):
        """Rasterize only the visible crop, so zoom cost stays ~canvas-sized."""
        canvas_w = min(max_w, disp_w + CANVAS_PADDING_PX)
        canvas_h = min(max_h, disp_h + CANVAS_PADDING_PX)
        base_off_x = (canvas_w - disp_w) // 2
        base_off_y = (canvas_h - disp_h) // 2
        self.clamp_zoom_pan(disp_w, disp_h, canvas_w, canvas_h, base_off_x, base_off_y)
        self.offset_x = base_off_x + self.zoom_pan_x
        self.offset_y = base_off_y + self.zoom_pan_y

        self.canvas.delete("all")
        scale = max(self.display_scale, 1e-9)
        # Visible window (canvas coords) intersected with the scaled image extent.
        vx0 = max(0, self.offset_x)
        vy0 = max(0, self.offset_y)
        vx1 = min(canvas_w, self.offset_x + disp_w)
        vy1 = min(canvas_h, self.offset_y + disp_h)
        if vx1 > vx0 and vy1 > vy0:
            ox0 = max(0, int(np.floor((vx0 - self.offset_x) / scale)))
            oy0 = max(0, int(np.floor((vy0 - self.offset_y) / scale)))
            ox1 = min(frame_w, int(np.ceil((vx1 - self.offset_x) / scale)))
            oy1 = min(frame_h, int(np.ceil((vy1 - self.offset_y) / scale)))
            crop = base[oy0:oy1, ox0:ox1]
            out_w = max(1, int(round((ox1 - ox0) * scale)))
            out_h = max(1, int(round((oy1 - oy0) * scale)))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            resized = cv2.resize(crop, (out_w, out_h), interpolation=interp)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            place_x = int(round(self.offset_x + ox0 * scale))
            place_y = int(round(self.offset_y + oy0 * scale))
            self.canvas_image_id = self.canvas.create_image(place_x, place_y, anchor=tk.NW, image=self.tk_image)

        if self.canvas.winfo_width() != canvas_w or self.canvas.winfo_height() != canvas_h:
            self.canvas.config(width=canvas_w, height=canvas_h)
        self.window.maxsize(screen_w, screen_h)
