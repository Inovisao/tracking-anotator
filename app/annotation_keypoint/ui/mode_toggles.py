from app.annotation.ui.mode_toggles import ModeTogglesMixin


class KPModeTogglesMixin(ModeTogglesMixin):
    def toggle_selection_mode(self):
        super().toggle_selection_mode()
        if not self.selection_mode:
            self.selected_instance = None
            self.selected_kp = None

    def toggle_remove_mode(self):
        super().toggle_remove_mode()
        if self.remove_mode:
            self.selected_instance = None
            self.selected_kp = None

    def toggle_edit_id_mode(self):
        print("[INFO] O modo keypoint nao usa edicao de ID.")
