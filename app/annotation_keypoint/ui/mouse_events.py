from app.annotation_keypoint.shared import *
from app.annotation.ui.rotation_utils import rotated_dims, rotated_to_image


class KPMouseEventsMixin:
    # ── wip helpers ───────────────────────────────────────────────
    def start_instance(self, category_id: int):
        names = self.keypoint_names_for_category(category_id)
        keypoints = [[0.0, 0.0, V_ABSENT] for _ in names]
        self.wip_instance = KeypointInstance(category_id=category_id, keypoints=keypoints, source="manual")
        self.wip_index = 0

    def wip_is_fixed(self) -> bool:
        return bool(self.keypoint_names_for_category(self.wip_instance.category_id)) if self.wip_instance else False

    def place_point(self, x: int, y: int, visibility: int):
        if self.wip_instance is None:
            self.start_instance(self.active_category_id())
        self.push_undo_state("adicionar keypoint")
        if self.wip_is_fixed():
            total = len(self.wip_instance.keypoints)
            if self.wip_index >= total:
                return
            self.wip_instance.keypoints[self.wip_index] = [float(x), float(y), int(visibility)]
            self.wip_index += 1
            if self.wip_index >= total:
                self.commit_wip_instance()
        else:
            self.wip_instance.keypoints.append([float(x), float(y), int(visibility)])
            self.wip_index += 1
        self.update_display(refresh_status=True)

    def skip_point(self):
        """Marks the current keypoint as absent (fixed mode only)."""
        if self.wip_instance is None or not self.wip_is_fixed():
            return
        self.place_point(0, 0, V_ABSENT)

    def undo_last_point(self):
        if self.wip_instance is None:
            return
        if self.wip_index <= 0:
            self.cancel_wip_instance()
            return
        self.push_undo_state("remover keypoint do wip")
        self.wip_index -= 1
        if self.wip_is_fixed():
            self.wip_instance.keypoints[self.wip_index] = [0.0, 0.0, V_ABSENT]
        else:
            self.wip_instance.keypoints.pop()
        if self.wip_index <= 0:  # last point removed — drop the empty instance
            self.wip_instance = None
            self.wip_index = 0
        self.update_display(refresh_status=True)

    def on_escape(self):
        """Esc only aborts the current operation — it never closes the application."""
        if self.wip_instance is not None:
            self.cancel_wip_instance()
            self.info_var.set("Instancia cancelada.")
        elif self.selected_instance is not None:
            self.selected_instance = None
            self.selected_kp = None
            self.update_display(refresh_status=True)

    @staticmethod
    def _next_visibility(value: int) -> int:
        order = [V_VISIBLE, V_HIDDEN, V_ABSENT]
        try:
            return order[(order.index(value) + 1) % len(order)]
        except ValueError:
            return V_VISIBLE

    def cycle_next_visibility(self):
        # With a keypoint selected, toggle THAT point's visibility (2<->1) so it
        # never vanishes; otherwise set the visibility for the next point placed.
        if self.get_selected_detection() is not None and self.selected_kp is not None:
            self.toggle_selected_visibility()
            return
        self.next_visibility = self._next_visibility(self.next_visibility)
        self.info_var.set(f"Visibilidade do proximo ponto: {self.next_visibility}")
        self.update_status()

    def toggle_selected_visibility(self):
        """Toggle the selected keypoint between visible (2) and occluded (1)."""
        det = self.get_selected_detection()
        if det is None or self.selected_kp is None or not (0 <= self.selected_kp < len(det.keypoints)):
            self.info_var.set("Selecione um ponto (S + clique) ou use o clique direito.")
            self.update_status()
            return
        kp = det.keypoints[self.selected_kp]
        self.push_undo_state("alterar visibilidade")
        kp[2] = V_HIDDEN if kp[2] == V_VISIBLE else V_VISIBLE
        self.info_var.set("Ponto oculto (v=1)" if kp[2] == V_HIDDEN else "Ponto visivel (v=2)")
        self.update_display(refresh_status=True)

    def change_label_font(self, delta: int):
        """Resize the on-image keypoint labels (clamped 6–28 px)."""
        self.kp_label_font_size = int(min(28, max(6, self.kp_label_font_size + delta)))
        self.info_var.set(f"Fonte dos rotulos: {self.kp_label_font_size}px")
        self.update_display(refresh_status=True)

    def on_right_click(self, event):
        """Right-click a keypoint to toggle visible (2) <-> occluded (1)."""
        if self.current_frame is None:
            return
        coords = self.canvas_to_image_coords(event.x, event.y)
        if coords is None:
            return
        hit = self.find_instance_at(*coords)
        if hit is None or hit[1] is None:
            return
        idx, kp_idx = hit
        kp = self.kp_instances[idx].keypoints[kp_idx]
        self.push_undo_state("alterar visibilidade")
        kp[2] = V_HIDDEN if kp[2] == V_VISIBLE else V_VISIBLE
        self.selected_instance, self.selected_kp = idx, kp_idx
        self.selected_detection = self.get_selected_detection()
        self.info_var.set("Keypoint -> oculto (v=1)" if kp[2] == V_HIDDEN else "Keypoint -> visivel (v=2)")
        self.update_display(refresh_status=True)

    # ── mouse handlers ────────────────────────────────────────────
    def on_mouse_down(self, event):
        if self.current_frame is None:
            return
        if self.pan_mode:
            self.on_pan_start(event)
            return "break"
        img_coords = self.canvas_to_image_coords(event.x, event.y)
        if img_coords is None:
            return
        x, y = img_coords
        if self.roi_capture_mode and not self.roi_defined:
            self.add_roi_point(x, y)
            return
        if self.remove_mode:
            self.remove_annotation_at(x, y)
            return
        if self.selection_mode:
            self.select_detection_at(x, y)
            if self.selected_kp is not None:
                self.drag_start = (x, y)
            return
        if not self.annotation_mode:
            return
        if self.wip_instance is None:
            hit = self.find_instance_at(x, y)
            if hit is not None and hit[1] is not None:
                self.push_undo_state("mover keypoint")
                self.selected_instance, self.selected_kp = hit
                self.selected_detection = self.get_selected_detection()
                self.drag_start = (x, y)
                return
        if self._wip_should_close(x, y):
            self.finish_instance()
            return
        self.place_point(x, y, self.next_visibility)

    def _wip_first_point(self) -> Optional[List[float]]:
        if self.wip_instance is None:
            return None
        for kp in self.wip_instance.keypoints:
            if kp[2] > 0:
                return kp
        return None

    def _wip_should_close(self, x: int, y: int) -> bool:
        """Free-mode (no fixed keypoint list): clicking the first point closes the shape."""
        if self.wip_instance is None or self.wip_is_fixed():
            return False
        placed = [kp for kp in self.wip_instance.keypoints if kp[2] > 0]
        first = self._wip_first_point()
        return len(placed) >= 3 and math.hypot(first[0] - x, first[1] - y) <= self.hit_radius()

    def on_mouse_drag(self, event):
        if self.pan_mode:
            self.on_pan_drag(event)
            return "break"
        if self.drag_start is None or self.selected_kp is None:
            return
        img_coords = self._event_to_image_clamped(event)
        if img_coords is None:
            return
        det = self.get_selected_detection()
        if det is None or not (0 <= self.selected_kp < len(det.keypoints)):
            return
        x, y = img_coords
        kp = det.keypoints[self.selected_kp]
        kp[0], kp[1] = float(x), float(y)
        if kp[2] <= 0:
            kp[2] = V_VISIBLE
        self.update_display(refresh_status=True)

    def on_mouse_up(self, event):
        if self.pan_mode:
            self.on_pan_end(event)
            return "break"
        self.drag_start = None
        self.update_display(refresh_status=True)

    def _event_to_image_clamped(self, event) -> Optional[Tuple[int, int]]:
        coords = self.canvas_to_image_coords(event.x, event.y)
        if coords is not None:
            return coords
        if self.current_frame is None:
            return None
        orig_h, orig_w = self.current_frame.shape[:2]
        rotation = getattr(self, "frame_rotation", 0)
        rot_w, rot_h = rotated_dims(orig_w, orig_h, rotation)
        rx = int(np.clip((event.x - self.offset_x) / max(self.display_scale, 1e-9), 0, rot_w - 1))
        ry = int(np.clip((event.y - self.offset_y) / max(self.display_scale, 1e-9), 0, rot_h - 1))
        if rotation:
            ox, oy = rotated_to_image(rx, ry, orig_w, orig_h, rotation)
            return int(np.clip(ox, 0, orig_w - 1)), int(np.clip(oy, 0, orig_h - 1))
        return rx, ry

    def on_mouse_move(self, event):
        if self.current_frame is None or self.wip_instance is None:
            return
        coords = self.canvas_to_image_coords(event.x, event.y)
        if coords is None or coords == self.cursor_image_pos:
            return  # skip the full re-render when the image-space point is unchanged
        self.cursor_image_pos = coords
        self.update_display()

    # ── pan & zoom ────────────────────────────────────────────────
    def on_pan_start(self, event):
        if self.current_frame is None:
            return
        self.pan_drag_start = (event.x, event.y)
        self.pan_start_offset = (self.zoom_pan_x, self.zoom_pan_y)
        try:
            self.canvas.config(cursor="fleur")
        except Exception:  # pylint: disable=broad-except
            pass

    def on_pan_drag(self, event):
        if self.current_frame is None or self.pan_drag_start is None:
            return
        start_x, start_y = self.pan_drag_start
        start_pan_x, start_pan_y = self.pan_start_offset
        self.zoom_pan_x = start_pan_x + int(event.x - start_x)
        self.zoom_pan_y = start_pan_y + int(event.y - start_y)
        self.update_display()

    def on_pan_end(self, _event):
        self.pan_drag_start = None
        self.update_canvas_cursor()

    def update_canvas_cursor(self):
        cursor = "fleur" if self.pan_mode else "crosshair"
        try:
            self.canvas.config(cursor=cursor)
        except Exception:  # pylint: disable=broad-except
            pass

    def toggle_pan_mode(self):
        self.pan_mode = not self.pan_mode
        if self.pan_mode:
            self.annotation_mode = False
            self.remove_mode = False
            self.selection_mode = False
        self.pan_drag_start = None
        self.update_annotation_button()
        self.update_remove_button()
        self.update_selection_button()
        self.update_pan_button()
        self.update_canvas_cursor()
        self.info_var.set("Pan ON: arraste a imagem para mover." if self.pan_mode else "Pan OFF.")
        self.update_status()

    def on_zoom(self, event):
        if self.current_frame is None:
            return
        event_delta = getattr(event, "delta", 0)
        event_num = getattr(event, "num", None)
        if event_delta != 0:
            factor = 1.1 if event_delta > 0 else 1 / 1.1
        elif event_num == 4:
            factor = 1.1
        elif event_num == 5:
            factor = 1 / 1.1
        else:
            return
        new_zoom = max(0.2, min(8.0, self.zoom_scale * factor))
        if new_zoom == self.zoom_scale:
            return
        old_img_x = (event.x - self.offset_x) / max(self.display_scale, 1e-9)
        old_img_y = (event.y - self.offset_y) / max(self.display_scale, 1e-9)
        frame_h, frame_w = self.current_frame.shape[:2]
        max_canvas_w, max_canvas_h, _, _ = self._canvas_viewport_limits()
        fit_scale = min(1.0, max_canvas_w / frame_w, max_canvas_h / frame_h)
        new_display_scale = fit_scale * new_zoom
        new_disp_w = max(1, int(round(frame_w * new_display_scale)))
        new_disp_h = max(1, int(round(frame_h * new_display_scale)))
        new_canvas_w = min(max_canvas_w, new_disp_w + CANVAS_PADDING_PX)
        new_canvas_h = min(max_canvas_h, new_disp_h + CANVAS_PADDING_PX)
        base_x = (new_canvas_w - new_disp_w) // 2
        base_y = (new_canvas_h - new_disp_h) // 2
        new_offset_x = event.x - old_img_x * new_display_scale
        new_offset_y = event.y - old_img_y * new_display_scale
        self.zoom_scale = new_zoom
        self.zoom_pan_x = int(round(new_offset_x - base_x))
        self.zoom_pan_y = int(round(new_offset_y - base_y))
        self.clamp_zoom_pan(new_disp_w, new_disp_h, new_canvas_w, new_canvas_h, base_x, base_y)
        self.update_display()

    def reset_zoom(self):
        self.zoom_scale = 1.0
        self.zoom_pan_x = 0
        self.zoom_pan_y = 0
        self.update_display()
