"""Responsive startup wizard used before the annotation screen."""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Callable, List, Optional

from ultralytics import YOLO

from app.classification.dataset import (
    STATE_FILE_NAME as CLASSIFICATION_STATE_FILE_NAME,
    discover_images,
    latest_output_state_for_sources as latest_classification_state_for_sources,
    list_output_states_for_sources as list_classification_states_for_sources,
    load_required_state as load_classification_state,
)
from app.config import DATA_ROOT, LOGO_PATH, WEIGHTS_PATH
from app.core.output_state import (
    ANNOTATION_FILE_NAMES,
    OutputState,
    create_new_output_dir,
    find_annotations_path,
    latest_output_state_for_sources,
    list_output_states_for_sources,
    load_annotation_state,
    output_dir_from_annotations_path,
)
from app.core.session import AnnotationSessionConfig, AnnotationTaskMode, normalize_class_names
from app.core.startup_cache import load_startup_cache, save_startup_cache
from app.sources.discovery import SourceDiscoveryService, SourceSummary
from app.ui.layout.responsive_window import apply_responsive_geometry
from app.ui.layout.scrollable_frame import ScrollableFrame
from app.ui.theme import install_scaled_theme
from app.ui.theme.palette import CLASS_COLORS


def ask_startup_config() -> AnnotationSessionConfig:
    wizard = StartupWizard()
    result = wizard.run()
    if result is None:
        sys.exit(0)
    return result


class StartupWizard:
    """Three-step startup flow: mode, dataset, model/classes."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("InoLabel — Configuração")
        self.ui = install_scaled_theme(self.root)
        self.colors = self.ui["colors"]
        self.fonts = self.ui["fonts"]
        self.spacing = self.ui["spacing"]
        self.sizes = self.ui["sizes"]

        self.root.configure(bg=self.colors["bg"])
        apply_responsive_geometry(self.root, width_ratio=0.92, height_ratio=0.88)
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)

        self.cache = load_startup_cache()
        self.discovery = SourceDiscoveryService()
        self.result: Optional[AnnotationSessionConfig] = None
        cached_mode = self.cache.mode or AnnotationTaskMode.TRACKING
        self.mode_var = tk.StringVar(value=cached_mode.value)
        self.data_root_var = tk.StringVar(value=self._initial_path_text(self.cache.data_root, DATA_ROOT))
        # Lista de caminhos de modelos (ensemble)
        if self.cache.weights_paths:
            self.weights_paths: List[str] = [str(p) for p in self.cache.weights_paths]
        elif Path(WEIGHTS_PATH).exists():
            self.weights_paths = [str(WEIGHTS_PATH)]
        else:
            self.weights_paths = []
        self.model_status_var = tk.StringVar(value="Modelos ainda nao validados.")
        self.output_state_mode_var = tk.StringVar(value="new")
        self.output_state_status_var = tk.StringVar(value="")
        self.classification_move_var = tk.BooleanVar(value=False)
        self.selected_state_path: Optional[Path] = None
        self.loaded_state_categories: tuple[dict, ...] = ()
        self.classes: List[str] = []
        # KEYPOINT mode: class name → comma-separated keypoint names (fixed order)
        self.keypoint_names_by_class: dict[str, str] = {}
        self.class_panel: Optional[tk.Frame] = None
        self.summary: Optional[SourceSummary] = None
        self.output_states: list[OutputState] = []
        # Project location: parent dir + session name chosen by user.
        # States always default to <app root>/state_saved (still editable below);
        # the cached parent is intentionally ignored so new states land at the root.
        from app.config import STATE_SAVED_ROOT
        self.parent_dir_var = tk.StringVar(value=str(STATE_SAVED_ROOT))
        self.session_name_var = tk.StringVar(value="")

        self.page = tk.Frame(self.root, bg=self.colors["bg"])
        self.page.pack(fill=tk.BOTH, expand=True)
        self.root.bind("<Configure>", self._on_resize)
        self.show_mode_screen()

    @staticmethod
    def _initial_path_text(cached_path: Optional[Path], fallback_path: Path) -> str:
        if cached_path is not None:
            return str(cached_path)
        fallback = Path(fallback_path)
        if fallback.exists():
            return str(fallback)
        return ""

    def run(self) -> Optional[AnnotationSessionConfig]:
        self.root.mainloop()
        return self.result

    def cancel(self):
        self.result = None
        self.root.destroy()

    def _clear(self):
        for child in self.page.winfo_children():
            child.destroy()

    def _on_resize(self, _event):
        wrap = max(420, min(self.sizes["content_max_w"], self.root.winfo_width() - (self.spacing["xl"] * 2)))
        for widget in self.root.winfo_children():
            self._update_wraplengths(widget, wrap)

    def _update_wraplengths(self, widget, wrap: int):
        if isinstance(widget, tk.Label) and getattr(widget, "_responsive_wrap", False):
            widget.configure(wraplength=wrap)
        for child in widget.winfo_children():
            self._update_wraplengths(child, wrap)

    def _step_indicator(self, parent, current: int):
        if AnnotationTaskMode(self.mode_var.get()) is AnnotationTaskMode.CLASSIFICATION:
            steps = ["Modo", "Imagens", "Estado", "Classes"]
        else:
            steps = ["Modo", "Dataset", "Estado", "Modelo"]
        row = tk.Frame(parent, bg=self.colors["bg"])
        row.pack(fill=tk.X, pady=(0, self.spacing["lg"]))

        for i, label in enumerate(steps, 1):
            is_done = i < current
            is_active = i == current

            if is_done:
                c_bg, c_fg = self.colors["primary"], self.colors["fg_light"]
                l_fg = self.colors["text"]
            elif is_active:
                c_bg, c_fg = self.colors["accent"], self.colors["text"]
                l_fg = self.colors["text"]
            else:
                c_bg, c_fg = self.colors["neutral"], self.colors["muted"]
                l_fg = self.colors["muted"]

            number = tk.Label(
                row, text="✓" if is_done else str(i),
                font=self.fonts["tag"],
                bg=c_bg, fg=c_fg,
                padx=self.spacing["sm"], pady=max(3, self.spacing["xs"]),
                cursor="hand2",
            )
            number.pack(side=tk.LEFT)

            name = tk.Label(
                row, text=label,
                font=self.fonts["tag"],
                bg=self.colors["bg"], fg=l_fg,
                cursor="hand2",
            )
            name.pack(side=tk.LEFT, padx=(self.spacing["xs"], 0))
            number.bind("<Button-1>", lambda _event, step=i: self._go_to_step(step))
            name.bind("<Button-1>", lambda _event, step=i: self._go_to_step(step))

            if i < len(steps):
                tk.Frame(
                    row, height=2, width=36,
                    bg=self.colors["primary"] if is_done else self.colors["border"],
                ).pack(side=tk.LEFT, padx=self.spacing["sm"])

    def _go_to_step(self, step: int):
        current_mode = AnnotationTaskMode(self.mode_var.get())
        if step == 1:
            self.show_mode_screen()
            return
        if step == 2:
            self.show_dataset_screen()
            return
        if step == 3:
            if current_mode is AnnotationTaskMode.CLASSIFICATION:
                if self.summary is None:
                    self.validate_dataset_and_continue()
                else:
                    self.show_state_screen()
                return
            if self.summary is None:
                self.validate_dataset_and_continue()
            else:
                self.show_state_screen()
            return
        if step == 4:
            if current_mode is AnnotationTaskMode.CLASSIFICATION:
                self.show_model_screen()
                return
            if self.summary is None:
                self.validate_dataset_and_continue()
            else:
                self.show_model_screen()

    def _build_logo_bar(self, parent: tk.Frame) -> None:
        bar = tk.Frame(parent, bg=self.colors["bg"])
        bar.pack(fill=tk.X, padx=self.spacing["xl"], pady=(self.spacing["sm"], 0))
        bar.columnconfigure(1, weight=1)

        if LOGO_PATH.exists():
            try:
                from PIL import Image, ImageTk  # pylint: disable=import-outside-toplevel
                img = Image.open(LOGO_PATH).convert("RGBA")
                img.thumbnail((96, 40), Image.LANCZOS)
                # Blend transparent pixels onto the wizard background
                bg_color = tuple(int(self.colors["bg"][i:i+2], 16) for i in (1, 3, 5))
                bg = Image.new("RGBA", img.size, bg_color + (255,))
                bg.paste(img, mask=img.split()[3])
                photo = ImageTk.PhotoImage(bg.convert("RGB"))
                lbl = tk.Label(bar, image=photo, bg=self.colors["bg"], cursor="arrow")
                lbl.image = photo
                lbl.grid(row=0, column=0, sticky="w", padx=(0, self.spacing["sm"]))
            except Exception:  # pylint: disable=broad-except
                pass

        tk.Label(
            bar, text="InoLabel",
            font=self.fonts["heading"],
            bg=self.colors["bg"],
            fg=self.colors["primary"],
            anchor="w",
        ).grid(row=0, column=1, sticky="w")

        tk.Label(
            bar, text="Inovisão",
            font=self.fonts["caption"],
            bg=self.colors["bg"],
            fg=self.colors["accent"],
            anchor="e",
        ).grid(row=0, column=2, sticky="e")

        tk.Frame(parent, height=2, bg=self.colors["accent"]).pack(
            fill=tk.X, padx=self.spacing["xl"], pady=(self.spacing["xs"], 0)
        )

    def _screen(self, title: str, subtitle: str, *, step: int):
        self._clear()

        self._build_logo_bar(self.page)

        # Step indicator — fixed above the scroll
        step_bar = tk.Frame(self.page, bg=self.colors["bg"])
        step_bar.pack(fill=tk.X, padx=self.spacing["xl"], pady=(self.spacing["lg"], 0))
        self._step_indicator(step_bar, step)

        scroll = ScrollableFrame(self.page, bg=self.colors["bg"])
        scroll.pack(fill=tk.BOTH, expand=True)
        outer = scroll.content
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=0)
        outer.columnconfigure(2, weight=1)

        width = min(self.sizes["content_max_w"], max(self.sizes["content_min_w"], self.root.winfo_width() - self.spacing["2xl"]))
        body = tk.Frame(outer, bg=self.colors["bg"], padx=self.spacing["xl"], pady=self.spacing["xl"], width=width)
        body.grid(row=0, column=1, sticky="n")
        body.columnconfigure(0, weight=1)

        title_label = tk.Label(
            body,
            text=title,
            font=self.fonts["title"],
            bg=self.colors["bg"],
            fg=self.colors["text"],
            anchor="w",
        )
        title_label.grid(row=0, column=0, sticky="ew", pady=(0, self.spacing["sm"]))

        subtitle_label = tk.Label(
            body,
            text=subtitle,
            font=self.fonts["body"],
            bg=self.colors["bg"],
            fg=self.colors["muted"],
            justify=tk.LEFT,
            anchor="w",
        )
        subtitle_label._responsive_wrap = True
        subtitle_label.grid(row=1, column=0, sticky="ew", pady=(0, self.spacing["xl"]))
        return body

    def _button(self, parent, text: str, command: Callable, *, primary: bool = False):
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
        if primary:
            button.configure(
                bg=self.colors["primary"],
                fg=self.colors["fg_light"],
                activebackground=self.colors["primary_active"],
                activeforeground=self.colors["fg_light"],
            )
            button.bind("<Enter>", lambda _e: button.configure(bg=self.colors["primary_active"]))
            button.bind("<Leave>", lambda _e: button.configure(bg=self.colors["primary"]))
        else:
            button.configure(
                bg=self.colors["neutral"],
                fg=self.colors["text"],
                activebackground=self.colors["neutral_active"],
                activeforeground=self.colors["text"],
            )
            button.bind("<Enter>", lambda _e: button.configure(bg=self.colors["neutral_active"]))
            button.bind("<Leave>", lambda _e: button.configure(bg=self.colors["neutral"]))
        return button

    def _footer(self, parent, back: Optional[Callable], next_: Callable, next_text: str = "Continuar"):
        footer = tk.Frame(parent, bg=self.colors["bg"])
        footer.grid(row=99, column=0, sticky="ew", pady=(self.spacing["xl"], 0))
        footer.columnconfigure(0, weight=1)
        if back is not None:
            self._button(footer, "Voltar", back).grid(row=0, column=1, padx=(0, self.spacing["sm"]), sticky="ew")
        self._button(footer, next_text, next_, primary=True).grid(row=0, column=2, sticky="ew")

    def _build_card(self, parent):
        card = tk.Frame(
            parent,
            bg=self.colors["panel"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            bd=0,
            padx=self.spacing["lg"],
            pady=self.spacing["lg"],
        )
        return card

    def show_mode_screen(self):
        body = self._screen(
            "Escolha o fluxo de anotacao",
            "Defina se esta sessao vai manter identidade dos objetos ao longo dos frames ou gerar anotacoes de deteccao padrao.",
            step=1,
        )
        body.rowconfigure(2, weight=1)
        options = tk.Frame(body, bg=self.colors["bg"])
        options.grid(row=2, column=0, sticky="nsew")
        for col in range(3):
            options.columnconfigure(col, weight=1)

        cards = [
            (AnnotationTaskMode.TRACKING, "Mantem IDs por objeto e usa rastreamento multiclass."),
            (AnnotationTaskMode.DETECTION, "Gera caixas independentes, sem IDs de tracking."),
            (AnnotationTaskMode.OBB, "Gera caixas orientadas com angulo para exportacao YOLO OBB."),
            (AnnotationTaskMode.KEYPOINT, "Marca pontos-chave por instancia para exportar YOLO Pose."),
            (AnnotationTaskMode.CLASSIFICATION, "Copia imagens para subpastas ao clicar na classe."),
        ]
        for idx, (mode, description) in enumerate(cards):
            self._mode_card(options, mode, description).grid(
                row=idx // 3, column=idx % 3, sticky="nsew", padx=8, pady=8
            )
        self._footer(body, None, self.show_dataset_screen)

    def _mode_card(self, parent, mode: AnnotationTaskMode, description: str):
        card = self._build_card(parent)
        card.columnconfigure(0, weight=1)
        radio = tk.Radiobutton(
            card,
            text=mode.label,
            value=mode.value,
            variable=self.mode_var,
            font=self.fonts["heading"],
            bg=self.colors["panel"],
            fg=self.colors["text"],
            activebackground=self.colors["panel"],
            selectcolor=self.colors["panel_alt"],
            anchor="w",
            command=lambda: self.mode_var.set(mode.value),
        )
        radio.grid(row=0, column=0, sticky="ew")
        desc = tk.Label(
            card,
            text=description,
            font=self.fonts["caption"],
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            justify=tk.LEFT,
            anchor="w",
        )
        desc._responsive_wrap = True
        desc.grid(row=1, column=0, sticky="ew", pady=(self.spacing["sm"], 0))
        return card

    def show_dataset_screen(self):
        body = self._screen(
            "Importe o dataset que sera anotado",
            "Selecione uma pasta, video, imagem unica ou lista de imagens. A validacao ocorre antes de abrir a tela de anotacao.",
            step=2,
        )
        body.columnconfigure(0, weight=1)
        form = self._build_card(body)
        form.grid(row=2, column=0, sticky="ew")
        form.columnconfigure(0, weight=1)

        tk.Label(
            form,
            text="Fonte de dados",
            bg=self.colors["panel"],
            fg=self.colors["text"],
            font=self.fonts["label"],
            anchor="w",
        ).grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, self.spacing["sm"]))
        entry = self._entry(form, self.data_root_var)
        entry.grid(row=1, column=0, columnspan=3, sticky="ew")

        actions = tk.Frame(form, bg=self.colors["panel"])
        actions.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(self.spacing["sm"], 0))
        actions.columnconfigure(0, weight=1)
        self._button(actions, "Selecionar pasta", self.browse_dataset_folder).grid(
            row=0, column=1, padx=(0, self.spacing["sm"])
        )
        self._button(actions, "Selecionar arquivo", self.browse_dataset_file).grid(row=0, column=2)

        summary_text = self._build_summary_text()
        summary = tk.Label(
            form,
            text=summary_text,
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            font=self.fonts["caption"],
            justify=tk.LEFT,
            anchor="w",
        )
        summary._responsive_wrap = True
        summary.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(self.spacing["md"], 0))
        self._footer(body, self.show_mode_screen, self.validate_dataset_and_continue)

    def _entry(self, parent, variable: tk.StringVar):
        return tk.Entry(
            parent,
            textvariable=variable,
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

    def browse_dataset_folder(self):
        initial = Path(self.data_root_var.get()).expanduser() if self.data_root_var.get().strip() else Path.home()
        from app.config import _EXE_DIR
        path = filedialog.askdirectory(
            title="Selecione a pasta com imagens ou videos",
            initialdir=str(initial if initial.exists() else _EXE_DIR),
            parent=self.root,
        )
        if path:
            self.data_root_var.set(path)
            self.summary = None
            self.show_dataset_screen()

    def browse_dataset_file(self):
        initial = Path(self.data_root_var.get()).expanduser() if self.data_root_var.get().strip() else Path.home()
        from app.config import _EXE_DIR
        path = filedialog.askopenfilename(
            title="Selecione uma imagem, video ou lista",
            initialdir=str(initial.parent if initial.parent.exists() else _EXE_DIR),
            filetypes=[
                ("Fontes suportadas", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp *.tif *.tiff *.txt *.lst"),
                ("Todos os arquivos", "*.*"),
            ],
            parent=self.root,
        )
        if path:
            self.data_root_var.set(path)
            self.summary = None
            self.show_dataset_screen()

    def _build_summary_text(self) -> str:
        if self.summary is None:
            return "Nenhuma fonte validada ainda."
        return (
            f"Fontes encontradas: {self.summary.total} | "
            f"videos: {self.summary.video_count} | "
            f"imagens: {self.summary.image_count} | "
            f"listas: {self.summary.image_list_count}"
        )

    def validate_dataset_and_continue(self):
        raw_path = self.data_root_var.get().strip()
        if not raw_path:
            messagebox.showerror("Dataset invalido", "Selecione uma fonte de dados antes de continuar.")
            return
        data_root = Path(raw_path).expanduser()
        if not data_root.exists():
            messagebox.showerror("Dataset invalido", f"Fonte nao encontrada:\n{data_root}")
            return
        self.summary = self.discovery.summarize(data_root)
        if not self.summary.has_sources:
            messagebox.showerror("Dataset invalido", f"Nenhuma fonte valida encontrada em:\n{data_root}")
            return
        if AnnotationTaskMode(self.mode_var.get()) is AnnotationTaskMode.CLASSIFICATION:
            if not discover_images(data_root):
                messagebox.showerror("Dataset invalido", f"Nenhuma imagem valida encontrada em:\n{data_root}")
                return
            self.show_state_screen()
            return
        self.show_state_screen()

    def show_model_screen(self):
        is_classification = AnnotationTaskMode(self.mode_var.get()) is AnnotationTaskMode.CLASSIFICATION
        body = self._screen(
            "Defina as classes" if is_classification else "Escolha os modelos auxiliares",
            (
                "Crie as classes de destino. O dataset final sera a pasta de saida com uma subpasta para cada classe."
                if is_classification
                else "Adicione um ou mais pesos YOLO (opcional). Com varios modelos as deteccoes sao mescladas via NMS (ensemble). Ajuste as classes iniciais para a sessao."
            ),
            step=4 if is_classification else 4,
        )
        form = self._build_card(body)
        form.grid(row=2, column=0, sticky="ew")
        form.columnconfigure(0, weight=1)

        class_row = 0
        if not is_classification:
            tk.Label(
                form,
                text="Modelos (.pt)",
                bg=self.colors["panel"],
                fg=self.colors["text"],
                font=self.fonts["label"],
                anchor="w",
            ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, self.spacing["sm"]))

            # Lista de modelos
            models_panel = tk.Frame(form, bg=self.colors["panel"])
            models_panel.grid(row=1, column=0, columnspan=3, sticky="ew")
            models_panel.columnconfigure(0, weight=1)
            self._redraw_models(models_panel)

            actions = tk.Frame(form, bg=self.colors["panel"])
            actions.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(self.spacing["sm"], 0))
            actions.columnconfigure(0, weight=1)
            self._button(actions, "Adicionar modelo(s)", self.browse_model).grid(
                row=0, column=1, padx=(0, self.spacing["sm"])
            )
            self._button(actions, "Validar modelos", self.validate_models_for_current_state).grid(row=0, column=2)
            class_row = 3

        tk.Label(
            form,
            text="Classes",
            bg=self.colors["panel"],
            fg=self.colors["text"],
            font=self.fonts["label"],
            anchor="w",
        ).grid(row=class_row, column=0, columnspan=2, sticky="ew", pady=(self.spacing["lg"], self.spacing["sm"]))
        class_panel = tk.Frame(form, bg=self.colors["panel"])
        class_panel.grid(row=class_row + 1, column=0, columnspan=3, sticky="ew")
        self.class_panel = class_panel
        self._redraw_classes(class_panel)

        summary_row = class_row + 2
        if not is_classification:
            model_status = tk.Label(
                form,
                textvariable=self.model_status_var,
                bg=self.colors["panel"],
                fg=self.colors["muted"],
                font=self.fonts["caption"],
                justify=tk.LEFT,
                anchor="w",
            )
            model_status._responsive_wrap = True
            model_status.grid(row=class_row + 2, column=0, columnspan=3, sticky="ew", pady=(self.spacing["lg"], 0))
            summary_row = class_row + 3

        summary = tk.Label(
            form,
            text=f"Modo escolhido: {AnnotationTaskMode(self.mode_var.get()).label}. Dataset validado: {self._build_summary_text()}",
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            font=self.fonts["caption"],
            justify=tk.LEFT,
            anchor="w",
        )
        summary._responsive_wrap = True
        summary.grid(row=summary_row, column=0, columnspan=3, sticky="ew", pady=(self.spacing["md"], 0))
        back = self.show_state_screen
        next_text = "Iniciar classificacao" if is_classification else "Iniciar anotacao"
        self._footer(body, back, self.finish, next_text)

    def _redraw_models(self, panel: tk.Frame):
        for child in panel.winfo_children():
            child.destroy()
        panel.columnconfigure(0, weight=1)

        if not self.weights_paths:
            tk.Label(
                panel,
                text="Nenhum modelo adicionado.",
                bg=self.colors["panel"],
                fg=self.colors["muted"],
                font=self.fonts["caption"],
                anchor="w",
            ).grid(row=0, column=0, sticky="ew")
            return

        for idx, path_str in enumerate(self.weights_paths):
            row_frame = tk.Frame(
                panel,
                bg=self.colors["input_bg"],
                highlightbackground=self.colors["border"],
                highlightthickness=1,
                bd=0,
            )
            row_frame.grid(row=idx, column=0, sticky="ew", pady=(0, self.spacing["xs"]))
            row_frame.columnconfigure(0, weight=1)

            name = Path(path_str).name
            tk.Label(
                row_frame,
                text=name,
                bg=self.colors["input_bg"],
                fg=self.colors["text"],
                font=self.fonts["body"],
                anchor="w",
                padx=self.spacing["sm"],
                pady=self.spacing["xs"],
            ).grid(row=0, column=0, sticky="ew")

            def _remove(i=idx):
                self.weights_paths.pop(i)
                self.show_model_screen()

            tk.Button(
                row_frame,
                text="×",
                bg=self.colors["input_bg"],
                fg=self.colors["danger"],
                font=self.fonts["button"],
                relief=tk.FLAT,
                cursor="hand2",
                command=_remove,
            ).grid(row=0, column=1, padx=(0, self.spacing["xs"]))

    def _current_parent_dir(self) -> Path:
        raw = self.parent_dir_var.get().strip()
        return Path(raw).expanduser() if raw else Path.home()

    def show_state_screen(self):
        is_classification = AnnotationTaskMode(self.mode_var.get()) is AnnotationTaskMode.CLASSIFICATION
        project_sources = self._current_project_sources()
        parent_dir = self._current_parent_dir()
        if is_classification:
            self.output_states = list_classification_states_for_sources(project_sources, parent_dir) if project_sources else []
        else:
            self.output_states = list_output_states_for_sources(project_sources, parent_dir) if project_sources else []
        latest = self.output_states[-1] if self.output_states else None
        if latest is not None and self.output_state_mode_var.get() == "new" and self.selected_state_path is None:
            self.output_state_mode_var.set("resume_latest")
            self.selected_state_path = latest.state_path if is_classification else latest.annotations_path
            self.output_state_status_var.set(f"Ultimo estado encontrado: {latest.label}")
            self._apply_state_template(self.selected_state_path)
        elif latest is None and self.output_state_mode_var.get() in {"resume_latest", "template_latest"}:
            self.output_state_mode_var.set("new")
            self.selected_state_path = None
            self.loaded_state_categories = ()
            self.classes = []
            self.output_state_status_var.set("Nenhum estado salvo encontrado para este projeto.")

        body = self._screen(
            "Escolha o estado de saida",
            (
                f"Continue um output deste projeto, use um {CLASSIFICATION_STATE_FILE_NAME} manualmente, ou crie um dataset novo."
                if is_classification
                else "Continue um output deste projeto, use um annotations.coco.json manualmente, ou crie um output novo isolado."
            ),
            step=3,
        )

        # ── Project location ──────────────────────────────────────────────────
        loc_card = self._build_card(body)
        loc_card.grid(row=2, column=0, sticky="ew", pady=(0, self.spacing["sm"]))
        loc_card.columnconfigure(1, weight=1)

        tk.Label(
            loc_card, text="Pasta pai:",
            font=self.fonts["label"], bg=self.colors["panel"], fg=self.colors["text"], anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=(0, self.spacing["sm"]), pady=(0, self.spacing["xs"]))
        parent_entry = tk.Entry(loc_card, textvariable=self.parent_dir_var, font=self.fonts["body"], **self._entry_opts())
        parent_entry.grid(row=0, column=1, sticky="ew")
        self._button(loc_card, "...", self._browse_parent_dir).grid(row=0, column=2, padx=(self.spacing["xs"], 0))

        tk.Label(
            loc_card, text="Nome do projeto:",
            font=self.fonts["label"], bg=self.colors["panel"], fg=self.colors["text"], anchor="w",
        ).grid(row=1, column=0, sticky="w", padx=(0, self.spacing["sm"]), pady=(self.spacing["xs"], 0))
        tk.Entry(loc_card, textvariable=self.session_name_var, font=self.fonts["body"], **self._entry_opts()).grid(
            row=1, column=1, sticky="ew", columnspan=2, pady=(self.spacing["xs"], 0)
        )
        tk.Label(
            loc_card, text="Obrigatorio ao criar sessao nova ou usar como modelo.",
            font=self.fonts["caption"], bg=self.colors["panel"], fg=self.colors["muted"], anchor="w",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(2, 0))

        form = self._build_card(body)
        form.grid(row=3, column=0, sticky="ew")
        form.columnconfigure(0, weight=1)

        latest_text = latest.label if latest else "Nenhum estado anterior encontrado."
        self._state_option(
            form,
            row=0,
            value="resume_latest",
            title="Continuar ultimo estado",
            description=latest_text,
            enabled=latest is not None,
        )
        self._state_option(
            form,
            row=1,
            value="template_latest",
            title="Usar ultimo estado como modelo",
            description="Carrega classes/configuracoes do ultimo output deste projeto e cria um output novo vazio.",
            enabled=latest is not None,
        )
        self._state_option(
            form,
            row=2,
            value="manual",
            title=(
                f"Escolher {CLASSIFICATION_STATE_FILE_NAME} manualmente"
                if is_classification
                else "Escolher annotations.coco.json manualmente"
            ),
            description="Permite continuar ou usar como modelo qualquer estado salvo.",
            enabled=True,
        )
        self._state_option(
            form,
            row=3,
            value="new",
            title="Criar sessao nova",
            description="Cria uma pasta nova com o nome informado acima, sem misturar anotacoes antigas.",
            enabled=True,
        )

        manual_row = tk.Frame(form, bg=self.colors["panel"])
        manual_row.grid(row=4, column=0, sticky="ew", pady=(self.spacing["md"], 0))
        manual_row.columnconfigure(0, weight=1)
        tk.Label(
            manual_row,
            textvariable=self.output_state_status_var,
            font=self.fonts["caption"],
            bg=self.colors["panel"],
            fg=self.colors["muted"],
            anchor="w",
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="ew", padx=(0, self.spacing["sm"]))
        self._button(manual_row, "Selecionar arquivo", self.browse_annotation_state).grid(row=0, column=1)

        # Inline validation hint for session name
        name_hint_var = tk.StringVar(value="")
        tk.Label(
            loc_card,
            textvariable=name_hint_var,
            font=self.fonts["caption"],
            bg=self.colors["panel"],
            fg=self.colors["accent"],
            anchor="w",
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(2, 0))

        def _needs_session_name() -> bool:
            return self.output_state_mode_var.get() in ("new", "template_latest") or (
                self.output_state_mode_var.get() == "manual"
                and self.selected_state_path is not None
                # For manual, the answer (resume=False) is only known at finish() time,
                # so we allow advancing and validate there
                and False
            )

        def _go_to_model_screen():
            if _needs_session_name() and not self.session_name_var.get().strip():
                name_hint_var.set("⚠ Informe o nome do projeto para continuar.")
                return
            name_hint_var.set("")
            self.show_model_screen()

        # Clear hint whenever mode or name changes
        self.output_state_mode_var.trace_add("write", lambda *_: name_hint_var.set(""))
        self.session_name_var.trace_add("write", lambda *_: name_hint_var.set(""))

        self._footer(body, self.show_dataset_screen, _go_to_model_screen)

    def _state_option(self, parent, *, row: int, value: str, title: str, description: str, enabled: bool):
        option = tk.Frame(parent, bg=self.colors["panel"])
        option.grid(row=row, column=0, sticky="ew", pady=(0, self.spacing["sm"]))
        option.columnconfigure(0, weight=1)
        state = tk.NORMAL if enabled else tk.DISABLED
        radio = tk.Radiobutton(
            option,
            text=title,
            value=value,
            variable=self.output_state_mode_var,
            font=self.fonts["label"],
            bg=self.colors["panel"],
            fg=self.colors["text"],
            activebackground=self.colors["panel"],
            selectcolor=self.colors["panel_alt"],
            anchor="w",
            state=state,
            command=lambda v=value: self._on_state_mode_changed(v),
        )
        radio.grid(row=0, column=0, sticky="ew")
        tk.Label(
            option,
            text=description,
            font=self.fonts["caption"],
            bg=self.colors["panel"],
            fg=self.colors["muted"] if enabled else self.colors["disabled_fg"],
            anchor="w",
            justify=tk.LEFT,
        ).grid(row=1, column=0, sticky="ew", padx=(self.spacing["lg"], 0))

    def _on_state_mode_changed(self, value: str):
        is_classification = AnnotationTaskMode(self.mode_var.get()) is AnnotationTaskMode.CLASSIFICATION
        if value in {"resume_latest", "template_latest"}:
            if is_classification:
                latest = latest_classification_state_for_sources(self._current_project_sources())
                self.selected_state_path = latest.state_path if latest is not None else None
            else:
                latest = latest_output_state_for_sources(self._current_project_sources())
                self.selected_state_path = latest.annotations_path if latest is not None else None
            if latest is not None:
                self.output_state_status_var.set(f"Estado selecionado: {latest.label}")
                self._apply_state_template(self.selected_state_path)
            else:
                self.output_state_mode_var.set("new")
                self.loaded_state_categories = ()
                self.classes = []
                self.output_state_status_var.set("Nenhum estado salvo encontrado para este projeto.")
                self._refresh_classes_panel()
        elif value == "new":
            self.selected_state_path = None
            self.loaded_state_categories = ()
            self.classes = []
            self.output_state_status_var.set("Novo estado sera criado ao iniciar.")
            self._refresh_classes_panel()

    def _entry_opts(self) -> dict:
        return {
            "bg": self.colors["input_bg"],
            "fg": self.colors["text"],
            "insertbackground": self.colors["text"],
            "relief": tk.FLAT,
            "highlightthickness": 1,
            "highlightbackground": self.colors["border"],
            "highlightcolor": self.colors["primary"],
        }

    def _browse_parent_dir(self):
        current = self.parent_dir_var.get().strip()
        initial = current if current and Path(current).exists() else str(Path.home())
        path = filedialog.askdirectory(
            title="Escolha a pasta onde o projeto sera criado",
            initialdir=initial,
            mustexist=True,
            parent=self.root,
        )
        if path:
            self.parent_dir_var.set(path)
            self.show_state_screen()

    def browse_annotation_state(self):
        is_classification = AnnotationTaskMode(self.mode_var.get()) is AnnotationTaskMode.CLASSIFICATION
        initial = self.selected_state_path.parent if self.selected_state_path else Path.cwd()
        path = filedialog.askopenfilename(
            title=f"Selecione {CLASSIFICATION_STATE_FILE_NAME}" if is_classification else "Selecione annotations.coco.json",
            initialdir=str(initial if initial.exists() else Path.cwd()),
            filetypes=[
                (
                    "Estado de classificacao" if is_classification else "COCO annotations",
                    CLASSIFICATION_STATE_FILE_NAME if is_classification else "annotations.coco.json annotations_obb.coco.json annotations_keypoints.coco.json __annotations.coco.json *.json",
                ),
                ("Todos os arquivos", "*.*"),
            ],
            parent=self.root,
        )
        if not path:
            return
        selected_path = Path(path).expanduser()
        if is_classification:
            if not self._is_classification_state_path(selected_path):
                messagebox.showerror(
                    "Estado invalido",
                    f"Selecione um arquivo {CLASSIFICATION_STATE_FILE_NAME} para o modo classificacao.",
                )
                return
        elif self._is_classification_state_path(selected_path):
            messagebox.showerror(
                "Estado invalido",
                "Este e um estado de classificacao. Selecione um annotations.coco.json para deteccao/OBB.",
            )
            return
        elif find_annotations_path(selected_path) is None:
            messagebox.showerror(
                "Estado invalido",
                "Selecione um estado COCO valido: annotations.coco.json, annotations_obb.coco.json, "
                "annotations_keypoints.coco.json ou __annotations.coco.json.",
            )
            return
        if is_classification:
            try:
                state = load_classification_state(selected_path)
            except Exception as exc:  # pylint: disable=broad-except
                messagebox.showerror("Estado invalido", str(exc))
                return
            if not state.classes:
                messagebox.showerror("Estado invalido", "O arquivo nao possui classes para carregar.")
                return
            self.selected_state_path = selected_path
            self.output_state_mode_var.set("manual")
            self.output_state_status_var.set(
                f"Selecionado: {self.selected_state_path} | {len(state.classes)} classes | "
                f"{len(state.records)} imagens classificadas"
            )
            self.classes = list(state.classes)
            self.loaded_state_categories = ()
        else:
            try:
                state = load_annotation_state(selected_path)
            except Exception as exc:  # pylint: disable=broad-except
                messagebox.showerror("Estado invalido", str(exc))
                return
            if not state.class_names:
                messagebox.showerror("Estado invalido", "O arquivo nao possui categorias/classes para carregar.")
                return
            self.selected_state_path = state.annotations_path
            self.output_state_mode_var.set("manual")
            self.output_state_status_var.set(
                f"Selecionado: {state.annotations_path} | {len(state.class_names)} classes | "
                f"{state.image_count} imagens | {state.annotation_count} anotacoes"
            )
            self.classes = list(state.class_names)
            self.loaded_state_categories = state.categories
            self._sync_loaded_categories_to_classes()
        self._refresh_classes_panel()

    def _apply_state_template(self, annotations_path: Path) -> bool:
        current_mode = AnnotationTaskMode(self.mode_var.get())
        if current_mode is AnnotationTaskMode.CLASSIFICATION:
            if not self._is_classification_state_path(annotations_path):
                messagebox.showerror(
                    "Estado invalido",
                    f"O modo classificacao so aceita {CLASSIFICATION_STATE_FILE_NAME}.",
                )
                return False
            try:
                state = load_classification_state(annotations_path)
            except Exception as exc:  # pylint: disable=broad-except
                messagebox.showerror("Estado invalido", str(exc))
                return False
            if not state.classes:
                messagebox.showerror("Estado invalido", "O estado selecionado nao possui classes.")
                return False
            self.classes = list(state.classes)
            self.loaded_state_categories = ()
        else:
            if self._is_classification_state_path(annotations_path):
                messagebox.showerror(
                    "Estado invalido",
                    "Estado de classificacao nao pode ser usado em deteccao/OBB.",
                )
                return False
            try:
                state = load_annotation_state(annotations_path)
            except Exception as exc:  # pylint: disable=broad-except
                messagebox.showerror("Estado invalido", str(exc))
                return False
            if not state.class_names:
                messagebox.showerror("Estado invalido", "O estado selecionado nao possui classes.")
                return False
            self.classes = list(state.class_names)
            self.loaded_state_categories = state.categories
            self._sync_loaded_categories_to_classes()
            if state.task_mode is not None:
                self.mode_var.set(state.task_mode.value)
        self._refresh_classes_panel()
        return True

    @staticmethod
    def _is_classification_state_path(path: Path) -> bool:
        path = Path(path).expanduser()
        if path.is_file():
            return path.name == CLASSIFICATION_STATE_FILE_NAME
        if path.is_dir():
            return (path / CLASSIFICATION_STATE_FILE_NAME).exists()
        return path.name == CLASSIFICATION_STATE_FILE_NAME and path.name not in ANNOTATION_FILE_NAMES

    def _sync_loaded_categories_to_classes(self):
        metadata_by_name = {}
        for cat in self.loaded_state_categories:
            name = str(cat.get("name", "")).strip()
            if name and name not in metadata_by_name:
                metadata_by_name[name] = dict(cat)

        synced = []
        for idx, name in enumerate(normalize_class_names(self.classes)):
            cat = dict(metadata_by_name.get(name, {}))
            cat["id"] = idx + 1
            cat["name"] = name
            cat.setdefault("color", CLASS_COLORS[idx % len(CLASS_COLORS)])
            cat.setdefault("supercategory", "none")
            synced.append(cat)

        self.classes = [cat["name"] for cat in synced]
        self.loaded_state_categories = tuple(synced)

    def _refresh_classes_panel(self):
        panel = getattr(self, "class_panel", None)
        if panel is None:
            return
        try:
            exists = panel.winfo_exists()
        except tk.TclError:
            exists = False
        if exists:
            self._redraw_classes(panel)

    def _state_classes_are_authoritative(self) -> bool:
        return self.output_state_mode_var.get() != "new" and bool(self.loaded_state_categories)

    def _current_project_sources(self) -> tuple[Path, ...]:
        if self.summary is not None and self.summary.sources:
            return tuple(Path(source).expanduser() for source in self.summary.sources)
        raw_path = self.data_root_var.get().strip()
        return (Path(raw_path).expanduser(),) if raw_path else ()

    def browse_model(self):
        initial_dir = Path.home()
        if self.weights_paths:
            candidate = Path(self.weights_paths[-1]).parent
            if candidate.exists():
                initial_dir = candidate
        paths = filedialog.askopenfilenames(
            title="Selecione um ou mais arquivos de pesos (.pt)",
            initialdir=str(initial_dir),
            filetypes=[("Pesos YOLO", "*.pt"), ("Todos os arquivos", "*.*")],
            parent=self.root,
        )
        if paths:
            existing = set(self.weights_paths)
            added = [p for p in paths if p not in existing]
            self.weights_paths.extend(added)
            self.model_status_var.set(
                f"{len(added)} modelo(s) adicionado(s). Valide antes de iniciar."
            )
            self.show_model_screen()

    def validate_models(self, *, import_classes: bool = True, refresh_screen: bool = True) -> bool:
        if not self.weights_paths:
            messagebox.showerror("Modelos invalidos", "Adicione ao menos um arquivo de pesos antes de continuar.")
            return False

        merged_classes: List[str] = []
        failed: List[str] = []
        loaded_names: List[str] = []

        for raw_path in self.weights_paths:
            weights_path = Path(raw_path).expanduser()
            if not weights_path.exists():
                failed.append(f"{weights_path.name}: arquivo nao encontrado")
                continue
            try:
                model = YOLO(str(weights_path))
            except Exception as exc:  # pylint: disable=broad-except
                failed.append(f"{weights_path.name}: {exc}")
                continue
            names = getattr(model, "names", None)
            model_classes = self._model_class_names(names)
            loaded_names.append(weights_path.name)
            for cls in model_classes:
                if cls not in merged_classes:
                    merged_classes.append(cls)

        if failed:
            messagebox.showerror(
                "Modelos invalidos",
                "Falha ao carregar:\n" + "\n".join(f"• {f}" for f in failed),
            )
            return False

        if merged_classes and import_classes:
            self.classes = merged_classes
            self._sync_loaded_categories_to_classes()

        count = len(loaded_names)
        cls_preview = ", ".join(merged_classes[:8]) + ("..." if len(merged_classes) > 8 else "")
        self.model_status_var.set(
            f"{count} modelo(s) validado(s): {', '.join(loaded_names)} | "
            f"classes: {cls_preview}"
        )
        if refresh_screen:
            self.show_model_screen()
        return True

    def validate_models_for_current_state(self) -> bool:
        import_classes = not self._state_classes_are_authoritative()
        ok = self.validate_models(import_classes=import_classes, refresh_screen=True)
        if ok and not import_classes:
            self.model_status_var.set(
                f"{self.model_status_var.get()} | classes preservadas do estado selecionado"
            )
        return ok

    def validate_model(self, *, import_classes: bool = True, refresh_screen: bool = True) -> bool:
        """Alias mantido para compatibilidade interna."""
        return self.validate_models(import_classes=import_classes, refresh_screen=refresh_screen)

    @staticmethod
    def _model_class_names(names) -> List[str]:
        if isinstance(names, dict):
            ordered = [names[key] for key in sorted(names.keys())]
        elif isinstance(names, list):
            ordered = names
        else:
            return []
        return list(normalize_class_names(str(name) for name in ordered))

    def _redraw_classes(self, panel: tk.Frame):
        self._class_editor_open = False
        for child in panel.winfo_children():
            child.destroy()
        panel.columnconfigure(0, weight=1)
        for idx, name in enumerate(self.classes):
            color = CLASS_COLORS[idx % len(CLASS_COLORS)]
            row = tk.Frame(
                panel,
                bg=self.colors["input_bg"],
                highlightbackground=color,
                highlightthickness=1,
                bd=0,
            )
            row.pack(fill=tk.X, pady=(0, self.spacing["sm"]))
            row.columnconfigure(1, weight=1)

            tk.Label(
                row,
                text="",
                bg=color,
                width=2,
            ).grid(row=0, column=0, sticky="nsw")
            tk.Label(
                row,
                text=f"{idx + 1} {name}",
                font=self.fonts["tag"],
                padx=self.spacing["sm"],
                pady=self.spacing["sm"],
                bg=self.colors["input_bg"],
                fg=self.colors["text"],
                anchor="w",
            ).grid(row=0, column=1, sticky="ew")
            if len(self.classes) > 1:
                tk.Button(
                    row,
                    text="↑",
                    font=self.fonts["tag"],
                    padx=self.spacing["sm"],
                    pady=self.spacing["sm"],
                    bd=0,
                    relief=tk.FLAT,
                    cursor="hand2" if idx > 0 else "arrow",
                    bg=self.colors["neutral"],
                    fg=self.colors["text"],
                    activebackground=self.colors["neutral_active"],
                    activeforeground=self.colors["text"],
                    disabledforeground=self.colors["muted"],
                    state=(tk.NORMAL if idx > 0 else tk.DISABLED),
                    command=lambda i=idx: self._move_class(panel, i, -1),
                ).grid(row=0, column=2, sticky="e", padx=(self.spacing["sm"], 0))
                tk.Button(
                    row,
                    text="↓",
                    font=self.fonts["tag"],
                    padx=self.spacing["sm"],
                    pady=self.spacing["sm"],
                    bd=0,
                    relief=tk.FLAT,
                    cursor="hand2" if idx < len(self.classes) - 1 else "arrow",
                    bg=self.colors["neutral"],
                    fg=self.colors["text"],
                    activebackground=self.colors["neutral_active"],
                    activeforeground=self.colors["text"],
                    disabledforeground=self.colors["muted"],
                    state=(tk.NORMAL if idx < len(self.classes) - 1 else tk.DISABLED),
                    command=lambda i=idx: self._move_class(panel, i, 1),
                ).grid(row=0, column=3, sticky="e", padx=(self.spacing["xs"], 0))
            remove_state = tk.NORMAL if len(self.classes) > 1 else tk.DISABLED
            remove_btn = tk.Button(
                row,
                text="Remover",
                font=self.fonts["tag"],
                padx=self.spacing["sm"],
                pady=self.spacing["sm"],
                bd=0,
                relief=tk.FLAT,
                cursor="hand2" if len(self.classes) > 1 else "arrow",
                bg=self.colors["danger"],
                fg=self.colors["fg_light"],
                activebackground=self.colors["danger"],
                activeforeground=self.colors["fg_light"],
                disabledforeground=self.colors["disabled_fg"],
                state=remove_state,
                command=lambda n=name: self._remove_class(panel, n),
            )
            remove_btn.grid(row=0, column=4, sticky="e", padx=(self.spacing["sm"], 0))
            if AnnotationTaskMode(self.mode_var.get()) is AnnotationTaskMode.KEYPOINT:
                self._build_keypoint_entry(row, name)
        self._button(panel, "+ Nova classe", lambda: self._show_class_entry(panel)).pack(
            fill=tk.X,
            pady=self.spacing["xs"],
        )

    DEFAULT_KEYPOINTS = "top_left, top_right, bottom_right, bottom_left"

    def _build_keypoint_entry(self, row: tk.Frame, name: str):
        if name not in self.keypoint_names_by_class:
            for cat in self.loaded_state_categories:
                if str(cat.get("name", "")).strip() == name and cat.get("keypoints"):
                    self.keypoint_names_by_class[name] = ", ".join(str(n) for n in cat["keypoints"])
                    break
            else:
                self.keypoint_names_by_class[name] = self.DEFAULT_KEYPOINTS
        var = tk.StringVar(value=self.keypoint_names_by_class.get(name, ""))
        var.trace_add("write", lambda *_: self.keypoint_names_by_class.__setitem__(name, var.get()))
        tk.Label(
            row, text="Keypoints (em ordem, separados por virgula)  —  ex: top_left, top_right, bottom_right, bottom_left",
            font=self.fonts["caption"], bg=self.colors["input_bg"], fg=self.colors["muted"], anchor="w",
        ).grid(row=1, column=0, columnspan=5, sticky="ew", padx=self.spacing["sm"])
        self._entry(row, var).grid(row=2, column=0, columnspan=5, sticky="ew",
                                   padx=self.spacing["sm"], pady=(0, self.spacing["sm"]))

    def _parse_keypoint_names(self, class_name: str) -> list[str]:
        raw = self.keypoint_names_by_class.get(class_name, "")
        names, seen = [], set()
        for part in raw.split(","):
            item = part.strip()
            if item and item not in seen:
                seen.add(item)
                names.append(item)
        return names

    def _effective_keypoints(self, class_name: str) -> list[str]:
        parsed = self._parse_keypoint_names(class_name)
        if parsed:
            return parsed
        for cat in self.loaded_state_categories:
            if str(cat.get("name", "")).strip() == class_name and cat.get("keypoints"):
                return [str(n) for n in cat["keypoints"]]
        return []

    def _remove_class(self, panel: tk.Frame, name: str):
        if len(self.classes) <= 1:
            messagebox.showwarning("Classes", "Mantenha ao menos uma classe para iniciar a anotacao.")
            return
        self.classes = [item for item in self.classes if item != name]
        self._sync_loaded_categories_to_classes()
        self._redraw_classes(panel)

    def _move_class(self, panel: tk.Frame, index: int, direction: int):
        new_index = index + direction
        if new_index < 0 or new_index >= len(self.classes):
            return
        self.classes[index], self.classes[new_index] = self.classes[new_index], self.classes[index]
        self._sync_loaded_categories_to_classes()
        self._redraw_classes(panel)

    def _show_class_entry(self, panel: tk.Frame):
        if getattr(self, "_class_editor_open", False):
            return
        self._class_editor_open = True

        children = panel.winfo_children()
        if children:
            children[-1].destroy()
        var = tk.StringVar()
        editor = tk.Frame(panel, bg=self.colors["panel"])
        editor.pack(fill=tk.X, pady=self.spacing["xs"])
        editor.columnconfigure(0, weight=1)

        entry = self._entry(editor, var)
        entry.grid(row=0, column=0, sticky="ew", padx=(0, self.spacing["sm"]))
        self._button(editor, "Adicionar", lambda: self._confirm_class(panel, var), primary=True).grid(
            row=0,
            column=1,
            padx=(0, self.spacing["sm"]),
        )
        self._button(editor, "Cancelar", lambda: self._redraw_classes(panel)).grid(row=0, column=2)
        entry.focus_set()
        entry.bind("<Return>", lambda _event: self._confirm_class(panel, var))
        entry.bind("<Escape>", lambda _event: self._redraw_classes(panel))

    def _confirm_class(self, panel: tk.Frame, var: tk.StringVar):
        name = var.get().strip()
        if not name:
            self._redraw_classes(panel)
            return
        if name not in self.classes:
            self.classes.append(name)
            self._sync_loaded_categories_to_classes()
        self._redraw_classes(panel)

    def finish(self):
        raw_data_root = self.data_root_var.get().strip()
        if not raw_data_root:
            messagebox.showerror("Dataset invalido", "Selecione uma fonte de dados antes de iniciar.")
            return
        mode = AnnotationTaskMode(self.mode_var.get())
        if not self.classes:
            messagebox.showerror("Classes invalidas", "Adicione ao menos uma classe antes de iniciar.")
            return
        if mode is AnnotationTaskMode.KEYPOINT:
            missing = [c for c in self.classes if not self._effective_keypoints(c)]
            if missing:
                messagebox.showerror(
                    "Keypoints obrigatorios",
                    "Defina os keypoints (em ordem) para: " + ", ".join(missing)
                    + "\n\nEx: top_left, top_right, bottom_right, bottom_left",
                )
                return
        data_root = Path(raw_data_root).expanduser()
        weights_paths = tuple(Path(p).expanduser() for p in self.weights_paths) if mode is not AnnotationTaskMode.CLASSIFICATION else ()
        if mode is not AnnotationTaskMode.CLASSIFICATION and self.weights_paths and not self.validate_models(import_classes=False, refresh_screen=False):
            return
        state_mode = self.output_state_mode_var.get()
        parent_dir = self._current_parent_dir()
        session_name = self.session_name_var.get().strip()
        output_dir = None
        annotations_path = None
        resume_existing = False
        category_metadata: tuple[dict, ...] = ()

        # Modes that create a new directory require a session name
        needs_new_dir = state_mode in ("new", "template_latest") or (
            state_mode == "manual" and self.selected_state_path is not None
        )
        if needs_new_dir and state_mode != "resume_latest" and state_mode != "manual":
            if not session_name:
                messagebox.showerror("Nome obrigatorio", "Informe um nome para o projeto antes de iniciar.")
                return

        if mode is AnnotationTaskMode.CLASSIFICATION and state_mode == "resume_latest":
            latest = latest_classification_state_for_sources(self._current_project_sources(), parent_dir)
            if latest is None:
                messagebox.showerror("Estado invalido", "Nenhum estado anterior foi encontrado para este projeto.")
                return
            self.selected_state_path = latest.state_path
            output_dir = latest.path
            annotations_path = latest.state_path
            resume_existing = True
        elif mode is AnnotationTaskMode.CLASSIFICATION and state_mode == "template_latest":
            latest = latest_classification_state_for_sources(self._current_project_sources(), parent_dir)
            if latest is None:
                messagebox.showerror("Estado invalido", "Nenhum estado anterior foi encontrado para este projeto.")
                return
            self.selected_state_path = latest.state_path
            if not session_name:
                messagebox.showerror("Nome obrigatorio", "Informe um nome para o projeto antes de iniciar.")
                return
            output_dir = create_new_output_dir(parent_dir, session_name, create_images_dir=False)
        elif mode is AnnotationTaskMode.CLASSIFICATION and state_mode == "manual":
            if self.selected_state_path is None:
                messagebox.showerror("Estado invalido", f"Selecione um {CLASSIFICATION_STATE_FILE_NAME} antes de iniciar.")
                return
            if not self.selected_state_path.exists():
                messagebox.showerror("Estado invalido", f"Arquivo nao encontrado:\n{self.selected_state_path}")
                return
            answer = messagebox.askyesnocancel(
                "Carregar estado",
                "Deseja continuar salvando neste estado?\n\n"
                "Sim: continua o dataset selecionado e carrega classificacoes antigas.\n"
                "Nao: usa apenas classes/configuracoes e cria um dataset novo.",
                parent=self.root,
            )
            if answer is None:
                return
            resume_existing = bool(answer)
            if resume_existing:
                output_dir = output_dir_from_annotations_path(self.selected_state_path)
            else:
                if not session_name:
                    messagebox.showerror("Nome obrigatorio", "Informe um nome para o projeto antes de iniciar.")
                    return
                output_dir = create_new_output_dir(parent_dir, session_name, create_images_dir=False)
            annotations_path = self.selected_state_path if resume_existing else None
        elif mode is AnnotationTaskMode.CLASSIFICATION:
            if not session_name:
                messagebox.showerror("Nome obrigatorio", "Informe um nome para o projeto antes de iniciar.")
                return
            output_dir = create_new_output_dir(parent_dir, session_name, create_images_dir=False)
        elif state_mode == "resume_latest":
            latest = latest_output_state_for_sources(self._current_project_sources(), parent_dir)
            if latest is None:
                messagebox.showerror("Estado invalido", "Nenhum estado anterior foi encontrado para este projeto.")
                return
            self.selected_state_path = latest.annotations_path
            output_dir = latest.path
            annotations_path = latest.annotations_path
            resume_existing = True
        elif state_mode == "template_latest":
            latest = latest_output_state_for_sources(self._current_project_sources(), parent_dir)
            if latest is None:
                messagebox.showerror("Estado invalido", "Nenhum estado anterior foi encontrado para este projeto.")
                return
            self.selected_state_path = latest.annotations_path
            if not session_name:
                messagebox.showerror("Nome obrigatorio", "Informe um nome para o projeto antes de iniciar.")
                return
            output_dir = create_new_output_dir(parent_dir, session_name)
        elif state_mode == "manual":
            if self.selected_state_path is None:
                messagebox.showerror("Estado invalido", "Selecione um annotations.coco.json antes de iniciar.")
                return
            if not self.selected_state_path.exists():
                messagebox.showerror("Estado invalido", f"Arquivo nao encontrado:\n{self.selected_state_path}")
                return
            answer = messagebox.askyesnocancel(
                "Carregar estado",
                "Deseja continuar salvando neste estado?\n\n"
                "Sim: continua o output selecionado e carrega anotacoes antigas.\n"
                "Nao: usa apenas classes/configuracoes e cria um output novo.",
                parent=self.root,
            )
            if answer is None:
                return
            resume_existing = bool(answer)
            if resume_existing:
                output_dir = output_dir_from_annotations_path(self.selected_state_path)
            else:
                if not session_name:
                    messagebox.showerror("Nome obrigatorio", "Informe um nome para o projeto antes de iniciar.")
                    return
                output_dir = create_new_output_dir(parent_dir, session_name)
            annotations_path = self.selected_state_path if resume_existing else None
        else:
            if not session_name:
                messagebox.showerror("Nome obrigatorio", "Informe um nome para o projeto antes de iniciar.")
                return
            output_dir = create_new_output_dir(parent_dir, session_name)

        self._sync_loaded_categories_to_classes()
        category_metadata = self.loaded_state_categories
        if mode is AnnotationTaskMode.KEYPOINT:
            category_metadata = tuple(
                {
                    **cat,
                    "keypoints": self._effective_keypoints(cat.get("name", "")),
                    "skeleton": cat.get("skeleton", []),
                }
                for cat in category_metadata
            )

        try:
            if mode is AnnotationTaskMode.OBB and annotations_path is None:
                annotations_path = output_dir / "saved_data_states" / "annotations_obb.coco.json"
            if mode is AnnotationTaskMode.KEYPOINT and annotations_path is None:
                annotations_path = output_dir / "saved_data_states" / "annotations_keypoints.coco.json"
            self.result = AnnotationSessionConfig(
                mode=mode,
                data_root=data_root,
                weights_paths=weights_paths,
                target_classes=tuple(self.classes),
                output_dir=output_dir,
                annotations_path=annotations_path,
                resume_existing_annotations=resume_existing,
                category_metadata=category_metadata,
                classification_move_files=False,
            )
        except ValueError as exc:
            messagebox.showerror("Configuracao invalida", str(exc))
            return
        save_startup_cache(data_root=data_root, weights_paths=weights_paths, mode=mode, parent_dir=parent_dir)
        self.root.destroy()
