"""Tkinter layout and keyboard bindings for the classification tool."""

from __future__ import annotations

import tkinter as tk

from app.ui.theme import COLORS


class ClassificationUIMixin:
    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        topbar = tk.Frame(self.root, bg=self.colors["panel"], padx=self.spacing["lg"], pady=self.spacing["md"])
        topbar.grid(row=0, column=0, sticky="ew")
        topbar.columnconfigure(0, weight=1)

        tk.Label(
            topbar,
            textvariable=self.image_name_var,
            bg=self.colors["panel"],
            fg=self.colors["text"],
            font=self.fonts["heading"],
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")
        tk.Label(
            topbar,
            textvariable=self.current_class_var,
            bg=self.colors["panel"],
            fg=self.colors["accent"],
            font=self.fonts["label"],
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", pady=(self.spacing["xs"], 0))
        tk.Label(
            topbar,
            textvariable=self.counter_var,
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            font=self.fonts["caption"],
            anchor="e",
        ).grid(row=0, column=1, rowspan=2, sticky="e", padx=(self.spacing["lg"], 0))

        body = tk.Frame(self.root, bg=self.colors["bg"])
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, minsize=300)
        body.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(body, bg="#111827", highlightthickness=0)
        self.canvas.configure(takefocus=1)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._render_image())

        sidebar = tk.Frame(body, bg=self.colors["panel"], padx=self.spacing["lg"], pady=self.spacing["lg"])
        sidebar.grid(row=0, column=1, sticky="nsew")
        sidebar.columnconfigure(0, weight=1)
        self.class_panel = tk.Frame(sidebar, bg=self.colors["panel"])
        self.class_panel.grid(row=0, column=0, sticky="ew")
        self._redraw_class_buttons()

        add_class_panel = tk.Frame(sidebar, bg=self.colors["panel"])
        add_class_panel.grid(row=1, column=0, sticky="ew", pady=(self.spacing["lg"], 0))
        add_class_panel.columnconfigure(0, weight=1)
        tk.Label(
            add_class_panel,
            text="Nova classe",
            bg=self.colors["panel"],
            fg=self.colors["text"],
            font=self.fonts["label"],
            anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, self.spacing["xs"]))
        entry = tk.Entry(
            add_class_panel,
            textvariable=self.new_class_var,
            font=self.fonts["body"],
            bg=self.colors["input_bg"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["accent"],
            bd=self.sizes["input_pad"],
        )
        entry.grid(row=1, column=0, sticky="ew", padx=(0, self.spacing["sm"]))
        entry.bind("<Return>", lambda _event: self.on_add_class())
        self._button(add_class_panel, "Adicionar", self.on_add_class, primary=True).grid(row=1, column=1, sticky="ew")

        self._build_sidebar_controls(sidebar)

        tk.Label(
            sidebar,
            textvariable=self.status_var,
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            font=self.fonts["caption"],
            justify=tk.LEFT,
            anchor="w",
        ).grid(row=3, column=0, sticky="ew", pady=(self.spacing["lg"], 0))

        statusbar = tk.Label(
            self.root,
            textvariable=self.info_var,
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            font=self.fonts["caption"],
            anchor="w",
            padx=self.spacing["lg"],
            pady=self.spacing["sm"],
        )
        statusbar.grid(row=2, column=0, sticky="ew")

    def _build_sidebar_controls(self, sidebar):
        controls = tk.Frame(sidebar, bg=self.colors["panel"])
        controls.grid(row=2, column=0, sticky="ew", pady=(self.spacing["lg"], 0))
        controls.columnconfigure(0, weight=1)
        nav = tk.Frame(controls, bg=self.colors["panel"])
        nav.grid(row=0, column=0, sticky="ew", pady=(0, self.spacing["sm"]))
        nav.columnconfigure(0, weight=1)
        nav.columnconfigure(1, weight=1)
        self._button(nav, "Anterior", self.on_previous_image).grid(row=0, column=0, sticky="ew", padx=(0, self.spacing["xs"]))
        self._button(nav, "Proxima", self.on_skip).grid(row=0, column=1, sticky="ew", padx=(self.spacing["xs"], 0))
        self._button(controls, "Desfazer (backspace)", self.on_undo).grid(row=1, column=0, sticky="ew", pady=(0, self.spacing["sm"]))
        self._button(controls, "Ver em folder", self.on_open_output_folder).grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(0, self.spacing["sm"]),
        )
        self._button(controls, "Remover imagem do dataset", self.on_remove_current_image, danger=True).grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(0, self.spacing["sm"]),
        )
        self._button(controls, "Exportar dataset", self.on_export_dataset, primary=True).grid(
            row=4,
            column=0,
            sticky="ew",
            pady=(0, self.spacing["sm"]),
        )
        self._button(controls, "Sair", self.on_quit, danger=True).grid(row=5, column=0, sticky="ew")

    def _button(self, parent, text: str, command, *, primary: bool = False, danger: bool = False):
        button = tk.Button(
            parent,
            text=text,
            command=command,
            font=self.fonts["button"],
            padx=self.sizes["btn_pad_x"],
            pady=self.sizes["btn_pad_y"],
            bd=0,
            relief=tk.FLAT,
            cursor="hand2",
            highlightthickness=0,
        )
        if danger:
            button.configure(
                bg=self.colors["danger"],
                fg=COLORS["fg_light"],
                activebackground=self.colors["danger"],
                activeforeground=COLORS["fg_light"],
            )
        elif primary:
            button.configure(
                bg=self.colors["primary"],
                fg=COLORS["fg_light"],
                activebackground=self.colors["primary_active"],
                activeforeground=COLORS["fg_light"],
            )
        else:
            button.configure(
                bg=self.colors["neutral"],
                fg=self.colors["text"],
                activebackground=self.colors["neutral_active"],
                activeforeground=self.colors["text"],
            )
        return button

    def _redraw_class_buttons(self):
        """Refresh the class buttons.

        The buttons are persistent widgets: navigating between images only changes the
        counters, so rebuilding the whole panel every time made each image change cost
        proportionally to the number of classes. Widgets are only created/destroyed when
        the class list itself changes.
        """
        counts = self._counts_by_class()
        widgets = getattr(self, "_class_button_widgets", None)
        if widgets is None or self._class_button_names != list(self.classes):
            self._rebuild_class_button_widgets()
            widgets = self._class_button_widgets

        remove_state = tk.NORMAL if len(self.classes) > 1 else tk.DISABLED
        for idx, class_name in enumerate(self.classes):
            entry = widgets.get(class_name)
            if entry is None:
                continue
            label = f"{idx + 1}  {class_name}  ({counts.get(class_name, 0)})"
            if entry["label"] != label:
                entry["button"].config(text=label)
                entry["label"] = label
            if entry["remove_state"] != remove_state:
                entry["remove"].config(state=remove_state)
                entry["remove_state"] = remove_state

    def _rebuild_class_button_widgets(self):
        """Create the per-class rows. Only called when the class list changes."""
        for child in self.class_panel.winfo_children():
            child.destroy()
        self._class_button_widgets = {}
        self._class_button_names = list(self.classes)
        for class_name in self.classes:
            row = tk.Frame(self.class_panel, bg=self.colors["panel"])
            row.pack(fill=tk.X, pady=(0, self.spacing["sm"]))
            row.columnconfigure(0, weight=1)
            button = self._button(
                row,
                "",
                lambda name=class_name: self.on_class_selected(name),
                primary=True,
            )
            button.grid(row=0, column=0, sticky="ew", padx=(0, self.spacing["xs"]))
            remove_btn = self._button(row, "Remover", lambda name=class_name: self.on_remove_class(name), danger=True)
            remove_btn.grid(row=0, column=1, sticky="e")
            self._class_button_widgets[class_name] = {
                "button": button,
                "remove": remove_btn,
                "label": None,
                "remove_state": None,
            }

    def _bind_shortcuts(self):
        for idx in range(1, 10):
            self.root.unbind_all(str(idx))
            self.root.unbind_all(f"<KP_{idx}>")
        for idx, class_name in enumerate(self.classes[:9], start=1):
            self.root.bind_all(str(idx), lambda event, name=class_name: self._run_shortcut(event, lambda: self.on_class_selected(name)))
            self.root.bind_all(f"<KP_{idx}>", lambda event, name=class_name: self._run_shortcut(event, lambda: self.on_class_selected(name)))
        for key in ("<Right>", "<Down>", "d", "D", "s", "S"):
            self.root.bind_all(key, lambda event: self._run_shortcut(event, self.on_skip))
        for key in ("<Left>", "<Up>", "a", "A", "w", "W"):
            self.root.bind_all(key, lambda event: self._run_shortcut(event, self.on_previous_image))
        self.root.bind_all("<space>", lambda event: self._run_shortcut(event, self.on_skip))
        self.root.bind_all("<BackSpace>", lambda event: self._run_shortcut(event, self.on_undo))
        self.root.bind_all("<Escape>", lambda event: self._run_shortcut(event, self.on_quit))

    def _run_shortcut(self, event, action):
        if isinstance(getattr(event, "widget", None), (tk.Entry, tk.Text)):
            return
        action()

    def _focus_navigation_surface(self):
        try:
            self.canvas.focus_set()
        except tk.TclError:
            self.root.focus_set()
