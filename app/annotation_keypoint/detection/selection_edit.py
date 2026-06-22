from app.annotation_keypoint.shared import *


class KPSelectionEditMixin:
    def clone_detection(self, inst: KeypointInstance) -> KeypointInstance:
        return clone_keypoint(inst)

    def hit_radius(self) -> float:
        return 10.0 / max(self.display_scale, 1e-9)

    # ── undo ──────────────────────────────────────────────────────
    def push_undo_state(self, reason: str = ""):
        if self.current_frame is None:
            return
        self.undo_stack.append({
            "reason": reason,
            "manual": [clone_keypoint(inst) for inst in self.kp_instances],
            "wip": clone_keypoint(self.wip_instance) if self.wip_instance is not None else None,
            "wip_index": self.wip_index,
            "selected": self.selected_instance,
        })

    def undo_last_action(self):
        if not self.undo_stack:
            print("[INFO] Nada para desfazer.")
            return
        snapshot = self.undo_stack.pop()
        self.kp_instances = [clone_keypoint(inst) for inst in snapshot["manual"]]
        self.manual_detections = self.kp_instances
        self.wip_instance = clone_keypoint(snapshot["wip"]) if snapshot["wip"] is not None else None
        self.wip_index = snapshot["wip_index"]
        self.selected_instance = snapshot["selected"]
        self.selected_kp = None
        self.selected_detection = self.get_selected_detection()
        print(f"[INFO] Desfeito: {snapshot.get('reason') or 'ultima acao'}.")
        self.update_display(refresh_status=True)

    # ── selection ─────────────────────────────────────────────────
    def validate_selected_detection(self):
        if self.selected_instance is not None and not (0 <= self.selected_instance < len(self.kp_instances)):
            self.selected_instance = None
            self.selected_kp = None
        self.selected_detection = self.get_selected_detection()

    def get_selected_detection(self) -> Optional[KeypointInstance]:
        if self.selected_instance is not None and 0 <= self.selected_instance < len(self.kp_instances):
            return self.kp_instances[self.selected_instance]
        return None

    def find_instance_at(self, x: int, y: int) -> Optional[Tuple[int, Optional[int]]]:
        """Returns (instance_index, keypoint_index) for the closest hit, point first."""
        radius = self.hit_radius()
        best = None
        best_dist = radius
        for idx in range(len(self.kp_instances) - 1, -1, -1):
            kp_idx = nearest_keypoint(self.kp_instances[idx], x, y, radius)
            if kp_idx is not None:
                kp = self.kp_instances[idx].keypoints[kp_idx]
                dist = ((kp[0] - x) ** 2 + (kp[1] - y) ** 2) ** 0.5
                if dist <= best_dist:
                    best_dist = dist
                    best = (idx, kp_idx)
        if best is not None:
            return best
        for idx in range(len(self.kp_instances) - 1, -1, -1):
            bx, by, bw, bh = keypoints_bbox(self.kp_instances[idx])
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return idx, None
        return None

    def select_detection_at(self, x: int, y: int):
        hit = self.find_instance_at(x, y)
        if hit is None:
            self.selected_instance = None
            self.selected_kp = None
        else:
            self.selected_instance, self.selected_kp = hit
            det = self.get_selected_detection()
            class_name = self.category_name_by_id().get(det.category_id) if det else None
            manual_var = getattr(self, "manual_class_var", None)
            if class_name and manual_var is not None:
                manual_var.set(class_name)
            self.update_class_panel()
        self.selected_detection = self.get_selected_detection()
        self.update_display(refresh_status=True)

    def remove_annotation_at(self, x: int, y: int) -> bool:
        hit = self.find_instance_at(x, y)
        if hit is None:
            print("[INFO] Nenhuma instancia encontrada para remover.")
            return False
        idx, kp_idx = hit
        self.push_undo_state("remover keypoint")
        if kp_idx is not None and self.kp_instances[idx].num_keypoints() > 1:
            self.kp_instances[idx].keypoints[kp_idx] = [0.0, 0.0, V_ABSENT]
            print("[INFO] Keypoint marcado como ausente.")
        else:
            del self.kp_instances[idx]
            print("[INFO] Instancia removida.")
        self.manual_detections = self.kp_instances
        self.selected_instance = None
        self.selected_kp = None
        self.selected_detection = None
        self.update_display(refresh_status=True)
        return True

    # ── wip lifecycle ─────────────────────────────────────────────
    def commit_wip_instance(self):
        if self.wip_instance is not None:
            if not self.keypoint_names_for_category(self.wip_instance.category_id):
                self._drop_duplicate_closing_point(self.wip_instance)
            if validate_instance(self.wip_instance):
                self.kp_instances.append(self.wip_instance)
                self.manual_detections = self.kp_instances
        self.wip_instance = None
        self.wip_index = 0

    @staticmethod
    def _drop_duplicate_closing_point(inst: KeypointInstance, tol: float = 3.0):
        """Free-mode: drop a trailing point that coincides with the first one."""
        pts = [kp for kp in inst.keypoints if kp[2] > 0]
        if len(pts) >= 4:
            first, last = pts[0], pts[-1]
            if abs(first[0] - last[0]) <= tol and abs(first[1] - last[1]) <= tol:
                inst.keypoints.remove(last)

    def cancel_wip_instance(self):
        if self.wip_instance is None:
            return
        self.push_undo_state("cancelar instancia")
        self.wip_instance = None
        self.wip_index = 0
        self.update_display(refresh_status=True)

    def finish_instance(self):
        if self.wip_instance is None:
            return
        self.push_undo_state("finalizar instancia")
        self.commit_wip_instance()
        self.update_display(refresh_status=True)

    # ── tracking-less stubs (presentation compatibility) ──────────
    def apply_manual_id_to_selection(self):
        print("[INFO] O modo keypoint nao usa IDs de tracking.")

    def remove_detection_from_runtime_state(self, det):
        _ = det
        return None
