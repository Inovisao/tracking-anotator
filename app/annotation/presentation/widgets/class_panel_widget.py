from app.annotation.shared import *
from app.ui.components import make_btn, make_entry
from app.ui.theme.tokens import COLORS, FONTS, SIZES, SPACING


class ClassPanelWidgetMixin:
    def update_class_panel(self, *, force: bool = False):
        panel = getattr(self, "classes_panel", None)
        if panel is None:
            return
        if getattr(self, "_class_panel_editing", False) and not force:
            return

        self.ensure_category_metadata()
        active_name = self.active_class_name()
        frame_counts: Dict[int, int] = {}
        for det in list(getattr(self, "current_detections", [])) + list(getattr(self, "manual_detections", [])):
            frame_counts[det.category_id] = frame_counts.get(det.category_id, 0) + 1

        # Register up front so the snapshot sees the same ids/colors the rebuild will use —
        # otherwise a class registered only during the rebuild changes the snapshot afterwards
        # and triggers a second rebuild on the next call.
        category_ids = [self.register_category(class_name) for class_name in self.target_classes]
        color_by_id = self.category_color_by_id()
        # Only what the panel actually draws: one tag per target class, with its color.
        # Categories registered by the model outside target_classes never become tags,
        # so they must not invalidate the structure and force a full rebuild.
        structure_snapshot = tuple(
            (class_name, color_by_id.get(category_id, COLORS["primary"]))
            for class_name, category_id in zip(self.target_classes, category_ids)
        )
        dynamic_snapshot = (
            active_name,
            tuple(sorted(frame_counts.items())),
        )
        if (
            not force
            and structure_snapshot == getattr(self, "_class_panel_structure_snapshot", None)
            and dynamic_snapshot == getattr(self, "_class_panel_dynamic_snapshot", None)
        ):
            return

        if not force and structure_snapshot == getattr(self, "_class_panel_structure_snapshot", None):
            self._class_panel_dynamic_snapshot = dynamic_snapshot
            self._update_class_tag_widgets(active_name, frame_counts, color_by_id)
            return

        self._class_panel_structure_snapshot = structure_snapshot
        self._class_panel_dynamic_snapshot = dynamic_snapshot
        self._class_panel_snapshot = (structure_snapshot, dynamic_snapshot)
        self._class_tag_widgets = {}

        for child in panel.winfo_children():
            child.destroy()

        for idx, (class_name, category_id) in enumerate(zip(self.target_classes, category_ids)):
            cat_color = color_by_id.get(category_id, COLORS["primary"])
            is_active = class_name == active_name
            count = frame_counts.get(category_id, 0)
            self._build_class_tag(panel, idx, class_name, cat_color, is_active, count)

        self._build_add_class_button(panel)

    def _update_class_tag_widgets(self, active_name: str, frame_counts: Dict[int, int], color_by_id: Dict[int, str]):
        widgets_by_name = getattr(self, "_class_tag_widgets", {})
        for idx, class_name in enumerate(self.target_classes):
            widgets = widgets_by_name.get(class_name)
            if not widgets:
                continue
            category_id = self.class_to_category_id.get(class_name, 0)
            color = color_by_id.get(category_id, COLORS["primary"])
            is_active = class_name == active_name
            count = frame_counts.get(category_id, 0)
            tag_bg = color if is_active else COLORS["input_bg"]
            tag_fg = COLORS["fg_light"] if is_active else COLORS["text"]
            self._config_if_changed(widgets["tag"], bg=tag_bg)
            self._config_if_changed(widgets["name_btn"], text=f"{idx + 1}  {class_name}  ({count})", bg=tag_bg, fg=tag_fg)

    def _build_class_tag(self, panel, idx: int, class_name: str, color: str, is_active: bool, count: int):
        tag_bg = color if is_active else COLORS["input_bg"]
        tag_fg = COLORS["fg_light"] if is_active else COLORS["text"]

        tag = tk.Frame(
            panel,
            bg=tag_bg,
            highlightbackground=color,
            highlightthickness=1,
            bd=0,
        )
        tag.pack(fill=tk.X, padx=0, pady=(0, SPACING["sm"]))
        tag.columnconfigure(1, weight=1)

        dot = tk.Button(
            tag,
            text="  ",
            bg=color,
            activebackground=color,
            bd=0, relief=tk.FLAT, cursor="hand2",
            highlightthickness=0,
            command=lambda n=class_name: self.cycle_class_color(n),
        )
        dot.grid(row=0, column=0, sticky="nsw", padx=(0, SPACING["xs"]))

        name_btn = tk.Button(
            tag,
            text=f"{idx + 1}  {class_name}  ({count})",
            font=FONTS["tag"],
            padx=SPACING["sm"], pady=SPACING["sm"],
            bd=0, relief=tk.FLAT, cursor="hand2",
            bg=tag_bg, fg=tag_fg,
            activebackground=color, activeforeground=COLORS["fg_light"],
            highlightthickness=0,
            anchor="w",
            command=lambda n=class_name: self.set_active_class(n),
        )
        name_btn.grid(row=0, column=1, sticky="ew")

        if len(self.target_classes) > 1:
            up_btn = tk.Button(
                tag,
                text="↑",
                font=FONTS["tag"],
                padx=SPACING["sm"], pady=SPACING["sm"],
                bd=0, relief=tk.FLAT,
                cursor="hand2" if idx > 0 else "arrow",
                bg=COLORS["neutral"], fg=COLORS["text"],
                activebackground=COLORS["neutral_active"], activeforeground=COLORS["text"],
                disabledforeground=COLORS["muted"],
                highlightthickness=0,
                state=(tk.NORMAL if idx > 0 else tk.DISABLED),
                command=lambda n=class_name: self.move_class(n, -1),
            )
            up_btn.grid(row=0, column=2, sticky="e", padx=(SPACING["sm"], 0))
            down_btn = tk.Button(
                tag,
                text="↓",
                font=FONTS["tag"],
                padx=SPACING["sm"], pady=SPACING["sm"],
                bd=0, relief=tk.FLAT,
                cursor="hand2" if idx < len(self.target_classes) - 1 else "arrow",
                bg=COLORS["neutral"], fg=COLORS["text"],
                activebackground=COLORS["neutral_active"], activeforeground=COLORS["text"],
                disabledforeground=COLORS["muted"],
                highlightthickness=0,
                state=(tk.NORMAL if idx < len(self.target_classes) - 1 else tk.DISABLED),
                command=lambda n=class_name: self.move_class(n, 1),
            )
            down_btn.grid(row=0, column=3, sticky="e", padx=(SPACING["xs"], 0))

        remove_state = tk.NORMAL if len(self.target_classes) > 1 else tk.DISABLED
        remove_btn = tk.Button(
            tag,
            text="Remover",
            font=FONTS["tag"],
            padx=SPACING["sm"], pady=SPACING["sm"],
            bd=0, relief=tk.FLAT,
            cursor="hand2" if len(self.target_classes) > 1 else "arrow",
            bg=COLORS["danger"], fg=COLORS["fg_light"],
            activebackground=COLORS["danger_active"], activeforeground=COLORS["fg_light"],
            disabledforeground=COLORS["disabled_fg"],
            highlightthickness=0,
            state=remove_state,
            command=lambda n=class_name: self.remove_class(n),
        )
        remove_btn.grid(row=0, column=4, sticky="e", padx=(SPACING["sm"], 0))
        self._class_tag_widgets[class_name] = {"tag": tag, "name_btn": name_btn}

    def _build_add_class_button(self, panel):
        make_btn(
            panel, "+ Nova classe",
            lambda: self._show_add_class_entry(panel),
            variant="neutral", anchor="w",
        ).pack(fill=tk.X, pady=(0, SPACING["sm"]))

    def _show_add_class_entry(self, panel):
        if getattr(self, "_class_panel_editing", False):
            return
        self._class_panel_editing = True

        children = panel.winfo_children()
        if children:
            children[-1].destroy()

        entry_var = tk.StringVar()

        wrap = tk.Frame(
            panel,
            bg=COLORS["input_bg"],
            highlightbackground=COLORS["accent"],
            highlightthickness=1, bd=0,
        )
        wrap.pack(fill=tk.X, pady=(0, SPACING["sm"]))

        row = tk.Frame(wrap, bg=COLORS["input_bg"])
        row.pack(fill=tk.X, expand=True)

        entry = make_entry(row, entry_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        make_btn(row, "Adicionar", lambda: confirm(), variant="primary", size="sm").pack(
            side=tk.LEFT, padx=(SPACING["xs"], 0)
        )
        make_btn(row, "Cancelar", lambda: cancel(), variant="neutral", size="sm").pack(
            side=tk.LEFT, padx=(SPACING["xs"], 0)
        )

        entry.focus_set()

        def confirm(_=None):
            self._add_new_class(entry_var.get())

        def cancel(_=None):
            self._class_panel_editing = False
            self._class_panel_snapshot = None
            self.update_class_panel(force=True)

        entry.bind("<Return>", confirm)
        entry.bind("<Escape>", cancel)
