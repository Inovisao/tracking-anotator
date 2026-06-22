from app.annotation_keypoint.shared import *


class KPDisplayOverlaysMixin:
    @staticmethod
    def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
        color = hex_color.lstrip("#")
        if len(color) != 6:
            return (34, 197, 94)
        r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
        return (b, g, r)

    def draw_kp_instances(self, frame: np.ndarray) -> np.ndarray:
        color_by_id = self.category_color_by_id()
        name_by_id = self.category_name_by_id()
        for idx, inst in enumerate(self.kp_instances):
            selected = idx == self.selected_instance
            self._draw_single_instance(
                frame, inst, color_by_id, name_by_id,
                selected=selected, sel_kp=self.selected_kp if selected else None, closed=True,
            )
        if self.wip_instance is not None:
            self._draw_single_instance(
                frame, self.wip_instance, color_by_id, name_by_id,
                selected=True, sel_kp=None, closed=False,
            )
            self._draw_wip_preview(frame, color_by_id, name_by_id)
        return frame

    def _draw_single_instance(self, frame, inst, color_by_id, name_by_id, *, selected, sel_kp, closed):
        color = self.hex_to_bgr(color_by_id.get(inst.category_id, "#22c55e"))
        if selected:
            color = (0, 255, 255)
        names = self.keypoint_names_for_category(inst.category_id)
        # Follow the annotated points: a skeleton when defined, otherwise the
        # chain of clicked points (closed into a polygon once finalized).
        if self._category_skeleton(inst.category_id):
            self._draw_skeleton(frame, inst, color)
        else:
            self._draw_point_chain(frame, inst, color, closed)
        for kp_idx, kp in enumerate(inst.keypoints):
            if kp[2] <= 0:
                continue
            px, py = int(round(kp[0])), int(round(kp[1]))
            radius = 6 if kp_idx == sel_kp else 4
            if kp[2] == V_HIDDEN:
                cv2.circle(frame, (px, py), radius, color, 1)
            else:
                cv2.circle(frame, (px, py), radius, color, -1)
            label = names[kp_idx] if kp_idx < len(names) else str(kp_idx + 1)
            cv2.putText(frame, label, (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    def _category_skeleton(self, category_id: int) -> list:
        for cat in self.categories:
            if int(cat.get("id", 0)) == category_id:
                return cat.get("skeleton", []) or []
        return []

    def _draw_point_chain(self, frame, inst, color, closed: bool):
        pts = [(int(round(kp[0])), int(round(kp[1]))) for kp in inst.keypoints if kp[2] > 0]
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], color, 1)
        if closed and len(pts) >= 3:
            cv2.line(frame, pts[-1], pts[0], color, 1)

    def _draw_skeleton(self, frame, inst, color):
        for cat in self.categories:
            if int(cat.get("id", 0)) != inst.category_id:
                continue
            for link in cat.get("skeleton", []):
                try:
                    a, b = int(link[0]), int(link[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if 0 <= a < len(inst.keypoints) and 0 <= b < len(inst.keypoints):
                    ka, kb = inst.keypoints[a], inst.keypoints[b]
                    if ka[2] > 0 and kb[2] > 0:
                        cv2.line(frame, (int(ka[0]), int(ka[1])), (int(kb[0]), int(kb[1])), color, 1)
            return

    def _draw_wip_preview(self, frame, color_by_id, name_by_id):
        inst = self.wip_instance
        names = self.keypoint_names_for_category(inst.category_id)
        next_name = names[self.wip_index] if self.wip_index < len(names) else str(self.wip_index + 1)
        if self.cursor_image_pos is None:
            return
        cx, cy = int(self.cursor_image_pos[0]), int(self.cursor_image_pos[1])
        placed = [kp for kp in inst.keypoints if kp[2] > 0]
        closing = self._wip_should_close(cx, cy)
        if placed:
            last = placed[-1]
            cv2.line(frame, (int(last[0]), int(last[1])), (cx, cy), (0, 255, 255), 1)
        if closing:
            first = placed[0]
            cv2.circle(frame, (int(first[0]), int(first[1])), 9, (0, 255, 0), 2)
            cv2.putText(frame, "fechar", (cx + 8, cy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            return
        total = len(names) if names else "?"
        cv2.putText(
            frame, f"-> {next_name} [{self.wip_index + 1}/{total}]",
            (cx + 8, cy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
        )

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

    def _draw_active_manual_rectangle(self):
        return None
