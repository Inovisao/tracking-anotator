from app.annotation_keypoint.shared import *
from app.annotation.keybinds.keybind_mixin import KeybindMixin


class KPUIControlsMixin(KeybindMixin):
    def _bind_shortcuts(self):
        self.window.bind("<Escape>", lambda event: self._run_shortcut(event, self.on_escape))
        for key in "123456789":
            self.window.bind(key, self.on_class_shortcut)
        # Fixed keypoint instance controls
        self.window.bind("f", lambda e: self._run_shortcut(e, self.finish_instance))
        self.window.bind("x", lambda e: self._run_shortcut(e, self.skip_point))
        self.window.bind("c", lambda e: self._run_shortcut(e, self.cycle_next_visibility))
        self.window.bind("<BackSpace>", lambda e: self._run_shortcut(e, self.undo_last_point))
        self.init_keybind_service()

    @staticmethod
    def _shortcut_is_text_input(event) -> bool:
        return isinstance(getattr(event, "widget", None), (tk.Entry, tk.Text))

    def _run_shortcut(self, event, action):
        if self._shortcut_is_text_input(event):
            return
        action()

    def _build_canvas(self):
        self.canvas = tk.Canvas(self.window, bg="black", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
        self._bind_canvas_events()

    def _bind_canvas_events(self):
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_end)
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<Control-MouseWheel>", self.on_zoom)
        self.canvas.bind("<Command-MouseWheel>", self.on_zoom)
        self.canvas.bind("<Button-4>", self.on_zoom)
        self.canvas.bind("<Button-5>", self.on_zoom)
        self.canvas.bind("<Control-Button-4>", self.on_zoom)
        self.canvas.bind("<Control-Button-5>", self.on_zoom)

    def enable_controls_after_roi(self):
        self.accept_button.config(state=tk.NORMAL)
        self.reject_button.config(state=tk.NORMAL)
        self.annotation_button.config(state=tk.NORMAL)
        self.remove_button.config(state=tk.NORMAL)
        self.selection_button.config(state=tk.NORMAL)
        self.pan_button.config(state=tk.NORMAL)
        self.apply_id_button.config(state=tk.DISABLED)
        self.edit_id_button.config(state=tk.DISABLED)
        self.export_dataset_button.config(state=tk.NORMAL)

    def disable_controls_for_roi(self):
        for name in (
            "accept_button", "reject_button", "annotation_button", "remove_button",
            "selection_button", "apply_id_button", "edit_id_button", "export_dataset_button",
        ):
            button = getattr(self, name, None)
            if button is not None:
                button.config(state=tk.DISABLED)

    def update_pan_button(self):
        if hasattr(self, "pan_button"):
            text = "Mover imagem  ON  (H)" if self.pan_mode else "Mover imagem  OFF  (H)"
            self.pan_button.config(text=text)

    def update_annotation_button(self):
        if hasattr(self, "annotation_button"):
            estado = "ON" if self.annotation_mode else "OFF"
            self._config_if_changed(self.annotation_button, text=f"Modo anotacao {estado} (K)")

    def update_remove_button(self):
        if hasattr(self, "remove_button"):
            estado = "ON" if self.remove_mode else "OFF"
            self._config_if_changed(self.remove_button, text=f"Remover anotacao {estado}")

    def update_selection_button(self):
        if hasattr(self, "selection_button"):
            estado = "ON" if self.selection_mode else "OFF"
            self._config_if_changed(self.selection_button, text=f"Selecionar anotacao {estado} (S)")

    def update_edit_id_button(self):
        if hasattr(self, "edit_id_button"):
            self._config_if_changed(self.edit_id_button, text="Editar ID indisponivel")
