"""Internal dataset export screen."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any, Dict, List

from app.annotation.core.augmentation.augmentation_types import (
    AUGMENTATION_CATALOG,
    AugCatalogItem,
    AugEntry,
    AugParamDef,
    AugmentationPreset,
)
from app.annotation.core.export.export_types import ExportConfig
from app.annotation.core.export.yolo_label_service import build_zero_based_category_mapping
from app.annotation.presentation.export.preview_dialog import open_augmentation_preview_dialog
from app.ui.layout.scrollable_frame import ScrollableFrame
from app.ui.theme import COLORS, FONTS, SIZES, SPACING


class ExportScreenMixin:
    def show_export_screen(self):
        self.autosave_current_frame(reason="abrir exportacao")
        self.export_screen_active = True
        self._clear_main_content()
        self.info_var.set("Configure a exportacao. O output_dataset permanece intacto.")

        self._export_dest_var = tk.StringVar(value=str(self.output_dir.parent))
        self._export_folder_var = tk.StringVar(value=f"{self.output_dir.name}_export")
        self._export_yolo_var = tk.BooleanVar(value=True)
        self._export_coco_var = tk.BooleanVar(value=False)
        self._export_split_var = tk.BooleanVar(value=True)
        self._export_train_var = tk.StringVar(value="0.7")
        self._export_val_var = tk.StringVar(value="0.2")
        self._export_test_var = tk.StringVar(value="0.1")
        self._export_aug_enabled_var = tk.BooleanVar(value=False)
        self._export_copies_var = tk.IntVar(value=1)
        self._export_aug_enabled: Dict[str, tk.BooleanVar] = {}
        self._export_aug_params: Dict[str, Dict[str, tk.Variable]] = {}
        self._export_aug_widgets: List[tk.Widget] = []
        self._export_status_var = tk.StringVar(value="")

        root = tk.Frame(self.main_content_area, bg=COLORS["bg"])
        root.pack(fill=tk.BOTH, expand=True)
        header = tk.Frame(root, bg=COLORS["panel"], highlightbackground=COLORS["border"], highlightthickness=1)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="Exportacao de dataset",
            font=FONTS["heading"],
            bg=COLORS["panel"],
            fg=COLORS["text"],
            anchor="w",
            padx=SPACING["md"],
            pady=SPACING["sm"],
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._export_button(header, "Voltar para anotacao", self.show_annotation_screen).pack(
            side=tk.RIGHT, padx=SPACING["md"], pady=SPACING["sm"]
        )

        scroll = ScrollableFrame(root, bg=COLORS["bg"])
        scroll.pack(fill=tk.BOTH, expand=True)
        content = scroll.content
        self._build_export_destination_section(content)
        self._build_export_format_section(content)
        self._build_export_split_section(content)
        self._build_export_augmentation_section(content)

        actions = tk.Frame(root, bg=COLORS["panel"], highlightbackground=COLORS["border"], highlightthickness=1)
        actions.pack(fill=tk.X, side=tk.BOTTOM)

        # Status + buttons row (always visible)
        bottom_row = tk.Frame(actions, bg=COLORS["panel"])
        bottom_row.pack(fill=tk.X, side=tk.BOTTOM)

        # Progress bar row — packed above bottom_row, hidden until export starts
        self._export_progress_frame = tk.Frame(actions, bg=COLORS["panel"])
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Export.Horizontal.TProgressbar",
            troughcolor=COLORS["border"],
            background=COLORS["primary"],
            thickness=6,
        )
        self._export_progressbar = ttk.Progressbar(
            self._export_progress_frame,
            style="Export.Horizontal.TProgressbar",
            orient=tk.HORIZONTAL,
            mode="determinate",
            maximum=100,
        )
        self._export_progressbar.pack(fill=tk.X, padx=SPACING["md"], pady=(SPACING["xs"], SPACING["xs"]))
        self._export_status_label = tk.Label(
            bottom_row,
            textvariable=self._export_status_var,
            font=FONTS["caption"],
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            justify=tk.LEFT,
            anchor="w",
        )
        self._export_status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=SPACING["md"], pady=SPACING["sm"])
        self._export_confirm_btn = self._export_button(bottom_row, "Exportar", self._confirm_export_screen, primary=True)
        self._export_confirm_btn.pack(side=tk.RIGHT, padx=SPACING["md"], pady=SPACING["sm"])
        self._export_button(bottom_row, "Cancelar", self._cancel_or_close_export).pack(
            side=tk.RIGHT, pady=SPACING["sm"]
        )

    def _build_export_destination_section(self, parent):
        self._export_section(parent, "Destino")
        row = tk.Frame(parent, bg=COLORS["bg"])
        row.pack(fill=tk.X, padx=SPACING["md"], pady=SPACING["xs"])
        tk.Entry(row, textvariable=self._export_dest_var, **self._export_entry_opts()).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        self._export_button(row, "...", self._browse_export_destination, compact=True).pack(
            side=tk.LEFT, padx=(SPACING["xs"], 0)
        )
        self._export_labeled_entry(parent, "Nome da pasta:", self._export_folder_var)

    def _build_export_format_section(self, parent):
        self._export_section(parent, "Formato")
        self._export_check(parent, "YOLO", self._export_yolo_var).pack(
            fill=tk.X, padx=SPACING["md"], pady=SPACING["xs"]
        )
        self._export_check(parent, "COCO (.json)", self._export_coco_var).pack(
            fill=tk.X, padx=SPACING["md"], pady=SPACING["xs"]
        )

    def _build_export_split_section(self, parent):
        self._export_section(parent, "Divisao de dados")
        self._export_check(parent, "Aplicar split train/val/test", self._export_split_var).pack(
            fill=tk.X, padx=SPACING["md"], pady=SPACING["xs"]
        )
        row = tk.Frame(parent, bg=COLORS["bg"])
        row.pack(fill=tk.X, padx=SPACING["md"], pady=SPACING["xs"])
        for label, var in (
            ("Train", self._export_train_var),
            ("Val", self._export_val_var),
            ("Test", self._export_test_var),
        ):
            tk.Label(row, text=label, font=FONTS["label"], bg=COLORS["bg"], fg=COLORS["text"]).pack(side=tk.LEFT)
            tk.Entry(row, textvariable=var, width=7, **self._export_entry_opts()).pack(
                side=tk.LEFT, padx=(SPACING["xs"], SPACING["md"])
            )

    def _build_export_augmentation_section(self, parent):
        self._export_section(parent, "Data Augmentation")
        top = tk.Frame(parent, bg=COLORS["bg"])
        top.pack(fill=tk.X, padx=SPACING["md"], pady=SPACING["xs"])
        self._export_check(
            top,
            "Adicionar data augmentation ao YOLO exportado",
            self._export_aug_enabled_var,
            command=self._toggle_export_aug_fields,
        ).pack(side=tk.LEFT)
        tk.Label(top, text="Copias por imagem:", font=FONTS["label"], bg=COLORS["bg"], fg=COLORS["text"]).pack(
            side=tk.LEFT, padx=(SPACING["lg"], SPACING["xs"])
        )
        copies = tk.Spinbox(top, from_=1, to=5, width=4, textvariable=self._export_copies_var, font=FONTS["body"])
        copies.pack(side=tk.LEFT)
        self._export_aug_widgets.append(copies)

        cards = tk.Frame(parent, bg=COLORS["bg"])
        cards.pack(fill=tk.X, padx=SPACING["md"], pady=SPACING["xs"])
        for item in AUGMENTATION_CATALOG:
            self._build_export_aug_card(cards, item)
        self._toggle_export_aug_fields()

    def _build_export_aug_card(self, parent, item: AugCatalogItem):
        card = tk.Frame(parent, bg=COLORS["panel"], highlightbackground=COLORS["border"], highlightthickness=1)
        card.pack(fill=tk.X, pady=SPACING["xs"])
        header = tk.Frame(card, bg=COLORS["panel"])
        header.pack(fill=tk.X, padx=SPACING["sm"], pady=(SPACING["sm"], 0))
        enabled = self._export_aug_enabled.setdefault(item.key, tk.BooleanVar(value=False))
        chk = tk.Checkbutton(header, variable=enabled, bg=COLORS["panel"], selectcolor=COLORS["input_bg"])
        chk.pack(side=tk.LEFT)
        self._export_aug_widgets.append(chk)
        tk.Label(header, text=item.label, font=FONTS["label"], bg=COLORS["panel"], fg=COLORS["text"]).pack(side=tk.LEFT)
        self._export_button(
            header,
            "Preview",
            lambda aug=item: self._open_export_aug_preview(aug),
            compact=True,
        ).pack(side=tk.RIGHT)
        tk.Label(header, text=item.category, font=FONTS["caption"], bg=COLORS["neutral"], fg=COLORS["text"]).pack(
            side=tk.RIGHT, padx=SPACING["sm"]
        )
        tk.Label(
            card,
            text=item.description,
            font=FONTS["caption"],
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            anchor="w",
        ).pack(fill=tk.X, padx=SPACING["sm"], pady=(0, SPACING["xs"]))
        param_vars = self._export_aug_params.setdefault(item.key, {})
        for param in item.params:
            self._build_export_param_row(card, item.key, param, param_vars)

    def _build_export_param_row(self, parent, _aug_key: str, param: AugParamDef, param_vars: Dict[str, tk.Variable]):
        row = tk.Frame(parent, bg=COLORS["panel"])
        row.pack(fill=tk.X, padx=SPACING["sm"], pady=2)
        tk.Label(row, text=param.key, width=16, font=FONTS["caption"], bg=COLORS["panel"], fg=COLORS["text"]).pack(
            side=tk.LEFT
        )
        if param.kind == "int":
            var = param_vars.setdefault(param.key, tk.IntVar(value=int(param.default)))
            resolution = 1
        else:
            var = param_vars.setdefault(param.key, tk.DoubleVar(value=float(param.default)))
            resolution = 0.01 if param.max <= 1 else 1
        scale = tk.Scale(
            row,
            from_=param.min,
            to=param.max,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            variable=var,
            showvalue=False,
            length=220,
            bg=COLORS["panel"],
            highlightthickness=0,
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entry = tk.Entry(row, textvariable=var, width=7, font=FONTS["caption"])
        entry.pack(side=tk.LEFT, padx=(SPACING["xs"], 0))
        self._export_aug_widgets.extend([scale, entry])

    def _open_export_aug_preview(self, item: AugCatalogItem):
        class_mapping, _ = build_zero_based_category_mapping(self.categories)
        open_augmentation_preview_dialog(
            self.window,
            aug_key=item.key,
            aug_label=item.label,
            params=self._export_aug_params_for(item.key),
            images=self.images,
            annotations=self.annotations,
            image_root=self.output_images_dir,
            class_mapping=class_mapping,
        )

    def _export_aug_params_for(self, key: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name, var in self._export_aug_params.get(key, {}).items():
            params[name] = var.get()
        return params

    def _build_export_preset(self):
        if not self._export_aug_enabled_var.get():
            return None
        entries = [
            AugEntry(key=item.key, enabled=bool(self._export_aug_enabled[item.key].get()), params=self._export_aug_params_for(item.key))
            for item in AUGMENTATION_CATALOG
        ]
        return AugmentationPreset(
            enabled=True,
            copies_per_image=int(max(1, min(5, self._export_copies_var.get()))),
            entries=entries,
        )

    def _confirm_export_screen(self):
        try:
            config = ExportConfig(
                destination_parent=Path(self._export_dest_var.get().strip()),
                folder_name=self._export_folder_var.get().strip(),
                formats=[name for name, var in (("yolo", self._export_yolo_var), ("coco", self._export_coco_var)) if var.get()],
                use_split=bool(self._export_split_var.get()),
                split_ratios=(
                    float(self._export_train_var.get()),
                    float(self._export_val_var.get()),
                    float(self._export_test_var.get()),
                ),
                augmentation=self._build_export_preset(),
            )
        except ValueError:
            self.info_var.set("Split invalido. Use numeros em train/val/test.")
            return
        if not config.destination_parent or not config.folder_name or not config.formats:
            self.info_var.set("Informe destino, pasta e ao menos um formato.")
            return
        self._export_status_var.set("Exportando...")
        self._export_status_label.config(fg=COLORS["muted"])
        self._export_confirm_btn.config(state=tk.DISABLED)
        self._start_export_progress()
        # All Tk updates from the worker go through a queue drained on the main
        # thread — calling Tcl (e.g. window.after) from the worker thread crashes
        # with "Tcl_AsyncDelete: async handler deleted by the wrong thread".
        self._export_main_queue = queue.Queue()
        self._export_running = True
        self._export_cancel_event = threading.Event()
        self.window.after(50, self._drain_export_queue)
        self._export_thread = threading.Thread(
            target=self._run_export_thread,
            args=(config,),
            daemon=True,
        )
        self._export_thread.start()

    def _post_to_main(self, fn):
        """Schedule a callable to run on the main (Tk) thread. Safe from workers."""
        self._export_main_queue.put(fn)

    def _drain_export_queue(self):
        try:
            while True:
                self._export_main_queue.get_nowait()()
        except queue.Empty:
            pass
        except Exception:  # pylint: disable=broad-except
            pass
        if getattr(self, "_export_running", False) or not self._export_main_queue.empty():
            self.window.after(50, self._drain_export_queue)

    def _cancel_or_close_export(self):
        cancel_event = getattr(self, "_export_cancel_event", None)
        if cancel_event is not None and not cancel_event.is_set():
            cancel_event.set()
            self._export_status_var.set("Cancelando...")
        else:
            self.show_annotation_screen()

    def _run_export_thread(self, config):
        cancel_event = getattr(self, "_export_cancel_event", None)
        try:
            self.perform_dataset_export(config, cancel_event=cancel_event)
        except Exception as exc:  # pylint: disable=broad-except
            msg = f"Falha ao exportar: {exc}"
            print(f"[ERRO] {msg}")
            self._post_to_main(lambda m=msg: self._show_export_thread_error(m))
        finally:
            self._post_to_main(self._on_export_thread_done)

    def _show_export_thread_error(self, message: str):
        try:
            self._export_status_var.set(message)
            self._export_status_label.config(fg=COLORS["danger"])
        except Exception:  # pylint: disable=broad-except
            pass

    def _start_export_progress(self):
        self._export_progressbar["value"] = 0
        self._export_progress_frame.pack(fill=tk.X, side=tk.TOP, padx=0, pady=0)

    def update_export_progress(self, percent: int):
        try:
            if hasattr(self, "_export_progressbar"):
                self._export_progressbar["value"] = percent
        except Exception:  # pylint: disable=broad-except
            pass  # widget destroyed before export finished (e.g. user navigated away)

    def _on_export_thread_done(self):
        self._export_running = False  # lets the queue poller stop after draining
        cancelled = getattr(self, "_export_cancel_event", None) and self._export_cancel_event.is_set()
        self._export_cancel_event = None
        if cancelled:
            self.show_annotation_screen()
            return
        try:
            if hasattr(self, "_export_progressbar"):
                self._export_progressbar["value"] = 100
            if hasattr(self, "_export_confirm_btn"):
                self._export_confirm_btn.config(state=tk.NORMAL)
            self.window.after(800, self._hide_export_progress)
        except Exception:  # pylint: disable=broad-except
            pass

    def _hide_export_progress(self):
        try:
            if hasattr(self, "_export_progress_frame"):
                self._export_progress_frame.pack_forget()
        except Exception:  # pylint: disable=broad-except
            pass

    def set_export_status(self, export_root: Path, exported_parts: List[str], config: ExportConfig):
        aug_text = "augmentation: desligado"
        if config.augmentation is not None and config.augmentation.enabled:
            aug_text = f"augmentation: {config.augmentation.copies_per_image} copias/imagem"
        self._export_status_var.set(
            f"Exportado com sucesso: {export_root}\n" + " | ".join(exported_parts + [aug_text])
        )
        self._export_status_label.config(fg=COLORS["primary"])

    def set_export_error(self, message: str):
        self._export_status_var.set(message)
        self._export_status_label.config(fg=COLORS["danger"])

    def _browse_export_destination(self):
        selected = filedialog.askdirectory(title="Escolha a pasta de destino", mustexist=True, parent=self.window)
        if selected:
            self._export_dest_var.set(selected)

    def _toggle_export_aug_fields(self):
        state = tk.NORMAL if self._export_aug_enabled_var.get() else tk.DISABLED
        for widget in self._export_aug_widgets:
            try:
                widget.config(state=state)
            except tk.TclError:
                pass

    def _export_section(self, parent, title: str):
        tk.Label(
            parent,
            text=title.upper(),
            font=FONTS["caption"],
            bg=COLORS["bg"],
            fg=COLORS["muted"],
            anchor="w",
        ).pack(fill=tk.X, padx=SPACING["md"], pady=(SPACING["md"], 2))

    def _export_labeled_entry(self, parent, label: str, var: tk.StringVar):
        tk.Label(parent, text=label, font=FONTS["label"], bg=COLORS["bg"], fg=COLORS["text"], anchor="w").pack(
            fill=tk.X, padx=SPACING["md"], pady=(SPACING["xs"], 0)
        )
        tk.Entry(parent, textvariable=var, **self._export_entry_opts()).pack(
            fill=tk.X, padx=SPACING["md"], pady=SPACING["xs"]
        )

    def _export_check(self, parent, text: str, variable: tk.BooleanVar, command=None):
        return tk.Checkbutton(
            parent,
            text=text,
            variable=variable,
            font=FONTS["body"],
            bg=COLORS["bg"],
            fg=COLORS["text"],
            selectcolor=COLORS["input_bg"],
            activebackground=COLORS["bg"],
            bd=0,
            highlightthickness=0,
            anchor="w",
            command=command,
        )

    def _export_button(self, parent, text: str, command, *, primary: bool = False, compact: bool = False):
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=FONTS["tag"] if compact else FONTS["button"],
            padx=SPACING["sm"] if compact else SIZES["btn_pad_x"],
            pady=SPACING["xs"] if compact else SIZES["btn_pad_y"],
            bd=0,
            relief=tk.FLAT,
            cursor="hand2",
            bg=COLORS["primary"] if primary else COLORS["neutral"],
            fg=COLORS["fg_light"] if primary else COLORS["text"],
            activebackground=COLORS["primary_active"] if primary else COLORS["neutral_active"],
            activeforeground=COLORS["fg_light"] if primary else COLORS["text"],
            highlightthickness=0,
        )

    def _export_entry_opts(self) -> dict:
        return {
            "font": FONTS["body"],
            "bg": COLORS["input_bg"],
            "fg": COLORS["text"],
            "insertbackground": COLORS["text"],
            "relief": tk.FLAT,
            "bd": SIZES["input_pad"],
            "highlightthickness": 1,
            "highlightbackground": COLORS["border"],
            "highlightcolor": COLORS["accent"],
        }
