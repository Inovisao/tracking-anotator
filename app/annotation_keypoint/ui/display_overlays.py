from app.annotation_keypoint.shared import *

_LABEL_FONT = ("Helvetica", 9)
_SEL_COLOR = "#FFD400"
_WIP_COLOR = "#22D3EE"
_CLOSE_COLOR = "#22C55E"


class KPDisplayOverlaysMixin:
    """Draws keypoints as lightweight Tk canvas items (no image re-rasterization)."""

    def _category_skeleton(self, category_id: int) -> list:
        for cat in self.categories:
            if int(cat.get("id", 0)) == category_id:
                return cat.get("skeleton", []) or []
        return []

    def render_overlays(self):
        self.canvas.delete("kp_overlay")
        color_by_id = self.category_color_by_id()
        for idx, inst in enumerate(self.kp_instances):
            selected = idx == self.selected_instance
            self._overlay_instance(
                inst, color_by_id,
                selected=selected, sel_kp=self.selected_kp if selected else None, closed=True,
            )
        if self.wip_instance is not None:
            self._overlay_instance(self.wip_instance, color_by_id, selected=True, sel_kp=None, closed=False)
            self._overlay_wip_preview(color_by_id)

    def _overlay_instance(self, inst, color_by_id, *, selected, sel_kp, closed):
        color = _SEL_COLOR if selected else color_by_id.get(inst.category_id, "#22c55e")
        names = self.keypoint_names_for_category(inst.category_id)
        canvas_pts = [
            self.image_to_canvas_coords(kp[0], kp[1]) if kp[2] > 0 else None
            for kp in inst.keypoints
        ]
        skeleton = self._category_skeleton(inst.category_id)
        if skeleton:
            for link in skeleton:
                try:
                    a, b = int(link[0]), int(link[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if 0 <= a < len(canvas_pts) and 0 <= b < len(canvas_pts) and canvas_pts[a] and canvas_pts[b]:
                    self.canvas.create_line(*canvas_pts[a], *canvas_pts[b], fill=color, width=2, tags="kp_overlay")
        else:
            chain = [p for p in canvas_pts if p is not None]
            for i in range(len(chain) - 1):
                self.canvas.create_line(*chain[i], *chain[i + 1], fill=color, width=2, tags="kp_overlay")
            if closed and len(chain) >= 3:
                self.canvas.create_line(*chain[-1], *chain[0], fill=color, width=2, tags="kp_overlay")
        for kp_idx, kp in enumerate(inst.keypoints):
            if kp[2] <= 0:
                continue
            cx, cy = canvas_pts[kp_idx]
            r = 6 if kp_idx == sel_kp else 4
            fill = "" if kp[2] == V_HIDDEN else color
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, fill=fill, width=2, tags="kp_overlay")
            label = names[kp_idx] if kp_idx < len(names) else str(kp_idx + 1)
            self.canvas.create_text(cx + 7, cy - 7, text=label, fill=color, anchor="w", font=_LABEL_FONT, tags="kp_overlay")

    def _overlay_wip_preview(self, color_by_id):
        if self.cursor_image_pos is None:
            return
        inst = self.wip_instance
        names = self.keypoint_names_for_category(inst.category_id)
        cur = self.image_to_canvas_coords(*self.cursor_image_pos)
        placed = [kp for kp in inst.keypoints if kp[2] > 0]
        if placed:
            last = self.image_to_canvas_coords(placed[-1][0], placed[-1][1])
            self.canvas.create_line(*last, *cur, fill=_WIP_COLOR, dash=(4, 2), tags="kp_overlay")
        if self._wip_should_close(self.cursor_image_pos[0], self.cursor_image_pos[1]):
            fx, fy = self.image_to_canvas_coords(placed[0][0], placed[0][1])
            self.canvas.create_oval(fx - 9, fy - 9, fx + 9, fy + 9, outline=_CLOSE_COLOR, width=2, tags="kp_overlay")
            self.canvas.create_text(cur[0] + 8, cur[1] + 14, text="fechar", fill=_CLOSE_COLOR, anchor="w",
                                    font=_LABEL_FONT, tags="kp_overlay")
            return
        total = len(names) if names else "?"
        nxt = names[self.wip_index] if self.wip_index < len(names) else str(self.wip_index + 1)
        self.canvas.create_text(cur[0] + 8, cur[1] + 14, text=f"-> {nxt} [{self.wip_index + 1}/{total}]",
                                fill=_WIP_COLOR, anchor="w", font=_LABEL_FONT, tags="kp_overlay")

    def _draw_roi_overlay_on_canvas(self):
        if not self.roi_points:
            return
        shifted = [self.image_to_canvas_coords(x, y) for (x, y) in self.roi_points]
        for i in range(len(shifted) - 1):
            self.canvas.create_line(*shifted[i], *shifted[i + 1], fill="blue", width=2)
        if len(shifted) == 4:
            self.canvas.create_line(*shifted[-1], *shifted[0], fill="blue", width=2)
        for sx, sy in shifted:
            self.canvas.create_oval(sx - 3, sy - 3, sx + 3, sy + 3, fill="red", outline="")
