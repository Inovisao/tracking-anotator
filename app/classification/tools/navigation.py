"""Image navigation and canvas rendering for classification."""

from __future__ import annotations

from pathlib import Path

import tkinter as tk
from PIL import Image, ImageTk


class ClassificationNavigationMixin:
    def _skip_classified_forward(self):
        classified = self._classified_sources()
        while self.current_index < len(self.images) and self.images[self.current_index] in classified:
            self.current_index += 1

    def _load_current_image(self, *, skip_classified: bool = True):
        if skip_classified:
            self._skip_classified_forward()
        if self.current_index >= len(self.images):
            self.current_image = None
            self.canvas.delete("all")
            self.image_name_var.set("Classificacao concluida")
            self.current_class_var.set("Use Exportar dataset para gerar as subpastas por classe.")
            self.counter_var.set(f"{len(self.records)}/{self.source_image_count} imagens")
            self.info_var.set(f"Dataset salvo em: {self.output_dir}")
            self._update_status()
            return

        source_path = self.images[self.current_index]
        image_path = self._display_path_for_source(source_path)
        try:
            self.current_image = self._open_for_display(image_path)
        except Exception as exc:  # pylint: disable=broad-except
            self.info_var.set(f"Falha ao abrir {image_path.name}: {exc}")
            self.current_index += 1
            self._load_current_image()
            return

        self.image_name_var.set(image_path.name)
        current_record = self._record_for_source(source_path)
        if current_record is None:
            self.current_class_var.set("Classe atual: ainda nao selecionada")
        else:
            self.current_class_var.set(f"Classe atual: {current_record.class_name}")
        self.counter_var.set(f"{self.current_index + 1}/{len(self.images)}")
        self.info_var.set(str(image_path))
        self._update_status()
        self._render_image()
        self._focus_navigation_surface()

    def _open_for_display(self, image_path: Path) -> Image.Image:
        """Decode an image for on-screen review only.

        draft() lets the JPEG decoder work at a reduced scale, which is much cheaper than
        decoding at full resolution just to shrink it right after. It is a no-op for
        formats that do not support it, and never touches the file on disk — the
        classification state stores paths, so the originals are untouched.
        """
        image = Image.open(image_path)
        try:
            canvas_w = max(1, self.canvas.winfo_width())
            canvas_h = max(1, self.canvas.winfo_height())
            if canvas_w > 2 and canvas_h > 2:
                image.draft("RGB", (canvas_w, canvas_h))
        except Exception:  # pylint: disable=broad-except
            pass
        return image.convert("RGB")

    def _display_path_for_source(self, source_path: Path) -> Path:
        if Path(source_path).exists():
            return source_path
        for record in reversed(self.records):
            if record.source_path == source_path and record.destination_path.exists():
                return record.destination_path
        return source_path

    def _render_image(self):
        if self.current_image is None:
            return
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        if canvas_w <= 2 or canvas_h <= 2:
            if not self._render_retry_scheduled:
                self._render_retry_scheduled = True
                self.root.after(50, self._retry_render_image)
            return
        self._render_retry_scheduled = False
        target_w = max(1, canvas_w - 24)
        target_h = max(1, canvas_h - 24)
        # The <Configure> binding fires on every canvas event, not just real resizes.
        # Rescaling a full-resolution image with LANCZOS costs 50-230 ms, so skip the
        # work when the same image is already drawn at this exact size.
        signature = (id(self.current_image), target_w, target_h)
        if signature == getattr(self, "_rendered_signature", None):
            return
        image = self.current_image.copy()
        image.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self._photo, anchor=tk.CENTER)
        self._rendered_signature = signature

    def _retry_render_image(self):
        self._render_retry_scheduled = False
        self._render_image()

    def _update_status(self):
        counts = self._counts_by_class()
        lines = [f"{class_name}: {counts.get(class_name, 0)}" for class_name in self.classes]
        pending = max(0, len(self.images) - self.current_index)
        if self.current_index < len(self.images):
            record = self._record_for_source(self.images[self.current_index])
            lines.append(f"Imagem atual: {record.class_name if record else 'sem classe'}")
        lines.append("Acao: salvar associacao no JSON")
        lines.append(f"Pendentes: {pending}")
        lines.append(f"Dataset: {self.output_dir.name}")
        self.status_var.set("\n".join(lines))
        if hasattr(self, "class_panel"):
            self._redraw_class_buttons()

    def on_skip(self):
        if self.current_index >= len(self.images):
            return
        self.current_index += 1
        self._load_current_image()
        self._focus_navigation_surface()

    def on_previous_image(self):
        if self.current_index <= 0:
            return
        self.current_index -= 1
        self._load_current_image(skip_classified=False)
        self._focus_navigation_surface()

    def on_undo(self):
        if not self.undo_stack:
            self.info_var.set("Nada para desfazer nesta sessao.")
            return
        record = self.undo_stack.pop()
        self.records = [item for item in self.records if item != record]
        try:
            self.current_index = self.images.index(record.source_path)
        except ValueError:
            self.current_index = max(0, self.current_index - 1)
        self._save_state()
        self.info_var.set(f"Desfeito: {record.source_path.name}")
        self._load_current_image(skip_classified=False)
        self._focus_navigation_surface()
