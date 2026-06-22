from app.annotation_keypoint.shared import *
from app.ui.file_manager import reveal_path
from app.ui.theme import COLORS


class KPDisplayStatusMixin:
    @staticmethod
    def _set_var_if_changed(var, value: str):
        if var.get() != value:
            var.set(value)

    @staticmethod
    def _config_if_changed(widget, **kwargs):
        updates = {}
        for key, value in kwargs.items():
            try:
                current = widget.cget(key)
            except tk.TclError:
                current = None
            if str(current) != str(value):
                updates[key] = value
        if updates:
            widget.config(**updates)

    def current_deletable_image_path(self) -> Optional[Path]:
        record = self.current_review_record()
        if record is not None:
            file_name = str(record.get("file_name", "")).strip()
            return self.output_images_dir / file_name if file_name else None
        if self.current_source_type == "images" and self.current_source_image_path is not None:
            return self.current_source_image_path
        return None

    def current_display_file_name(self) -> str:
        record = self.current_review_record()
        if record is not None:
            return str(record.get("file_name", "-"))
        if self.current_source_type == "images" and self.current_source_image_path is not None:
            return self.current_source_image_path.name
        return f"{self.video_name}_frame_{self.frame_index:05d}.jpg"

    def current_open_target_path(self) -> Optional[Path]:
        record = self.current_review_record()
        if record is not None:
            file_name = record.get("file_name")
            if file_name:
                saved_path = self.output_images_dir / str(file_name)
                if saved_path.exists():
                    return saved_path
            return None
        file_name = self.current_frame_file_name()
        if file_name:
            saved_path = self.output_images_dir / file_name
            if saved_path.exists():
                return saved_path
        if self.current_source_type == "images" and self.current_source_image_path is not None:
            if self.current_source_image_path.exists():
                return self.current_source_image_path
        return None

    def build_status_message(self) -> str:
        if self.review_idx is not None and self.saved_records:
            return f"Revisao keypoint {self.review_idx + 1}/{len(self.saved_records)} · {self.video_name}"
        if self.pan_mode:
            mode = "Pan ON"
        elif self.annotation_mode:
            mode = "Keypoint manual ON"
        elif self.remove_mode:
            mode = "Remocao ON"
        elif self.selection_mode:
            mode = "Selecao ON"
        else:
            mode = self.task_mode.label
        return f"{mode} · {self.video_name} · Frame {self.frame_index}"

    def update_status(self):
        self._set_var_if_changed(self.info_var, self.build_status_message())
        self.update_class_panel()
        self.update_status_blocks()
        self.update_image_info()
        self.update_annotation_button()
        self.update_remove_button()
        self.update_selection_button()
        self.update_edit_id_button()
        self.update_pan_button()

    def update_status_blocks(self):
        if not hasattr(self, "status_source_var"):
            return
        idx = self.current_video_index + 1
        total = len(self.video_files)
        frame_info = f"Rev. {self.review_idx + 1}/{len(self.saved_records)}" if self.review_idx is not None else f"Frame {self.frame_index}"
        self._set_var_if_changed(self.status_source_var, f"{self.video_name}  [{idx}/{total}]  ·  {frame_info}")
        self._set_var_if_changed(self.status_roi_var, "ROI OK" if self.roi_defined else f"ROI {len(self.roi_points)}/4 pts")
        self._config_if_changed(self.status_roi_lbl, fg=COLORS["primary"] if self.roi_defined else COLORS["muted"])
        self._set_var_if_changed(self.status_mode_var, "KEYPOINT")
        self._config_if_changed(self.status_mode_lbl, fg=COLORS["accent"])
        active = self.active_class_name() if hasattr(self, "active_class_name") else ""
        n_inst = len(self.kp_instances) + (1 if self.wip_instance is not None else 0)
        self._set_var_if_changed(self.status_class_var, f"● {active}  ·  {n_inst} inst.")
        self._config_if_changed(self.status_class_lbl, fg=COLORS["text"])
        self._set_var_if_changed(self.status_sel_var, "selecionada" if self.selected_instance is not None else "")

    def update_image_info(self):
        self._set_var_if_changed(self.image_name_var, f"Imagem: {self.current_display_file_name()}")
        target = self.current_open_target_path()
        can_open = target is not None or getattr(self, "current_frame", None) is not None
        self._config_if_changed(self.open_folder_button, state=(tk.NORMAL if can_open else tk.DISABLED))
        delete_target = self.current_deletable_image_path()
        self._config_if_changed(self.delete_image_button, state=(tk.NORMAL if delete_target is not None else tk.DISABLED))

    def open_in_file_manager(self, target: Path) -> bool:
        return reveal_path(target)

    def on_open_in_folder(self):
        target = self.current_open_target_path()
        if target is None and self.current_frame is not None:
            saved = self.autosave_current_frame(reason="abrir no folder")
            if saved is not None:
                _image_id, file_name = saved
                target = self.output_images_dir / file_name
        if target is None or not target.exists():
            print("[INFO] Nenhuma imagem de arquivo disponivel neste frame.")
            return
        if not self.open_in_file_manager(target):
            print(f"[AVISO] Nao foi possivel abrir o gerenciador para: {target}")
            return
        print(f"[INFO] Abrindo no gerenciador: {target}")
