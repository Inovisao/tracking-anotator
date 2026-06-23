"""Reusable loading screen shown while a UI page finishes building/loading."""

from __future__ import annotations

import tkinter as tk

from app.ui.theme.tokens import COLORS

_WIDTH = 360
_HEIGHT = 150


class LoadingScreen:
    """Borderless overlay drawn on top while heavy/blocking UI work runs.

    The Tk mainloop is usually not running yet during page construction, so the
    window is painted explicitly via update(); call close() when the page is ready.
    """

    def __init__(self, parent: tk.Misc, message: str = "Carregando..."):
        self._top = tk.Toplevel(parent)
        self._top.overrideredirect(True)
        self._top.configure(bg=COLORS["panel"])
        try:
            self._top.attributes("-topmost", True)
        except tk.TclError:
            pass
        self._center()

        canvas = tk.Canvas(self._top, width=_WIDTH, height=_HEIGHT, bg=COLORS["panel"],
                           highlightthickness=1, bd=0)
        canvas.configure(highlightbackground=COLORS["border"])
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.create_text(_WIDTH // 2, 48, text="InoLabel", fill=COLORS["primary"],
                           font=("Helvetica", 22, "bold"), anchor=tk.CENTER)
        self._message_item = canvas.create_text(
            _WIDTH // 2, 92, text=message, fill=COLORS["muted"],
            font=("Helvetica", 11), anchor=tk.CENTER,
        )
        canvas.create_rectangle(0, _HEIGHT - 6, _WIDTH, _HEIGHT, fill=COLORS["primary"], outline="")
        canvas.create_line(0, _HEIGHT - 6, _WIDTH, _HEIGHT - 6, fill=COLORS["accent"], width=2)
        self._canvas = canvas
        self._paint()

    def _center(self) -> None:
        self._top.update_idletasks()
        sw = self._top.winfo_screenwidth()
        sh = self._top.winfo_screenheight()
        x = (sw - _WIDTH) // 2
        y = (sh - _HEIGHT) // 2
        self._top.geometry(f"{_WIDTH}x{_HEIGHT}+{x}+{y}")

    def _paint(self) -> None:
        try:
            self._top.update()  # force a draw — the mainloop may not be running yet
        except tk.TclError:
            pass

    def update_message(self, message: str) -> None:
        try:
            self._canvas.itemconfig(self._message_item, text=message)
            self._paint()
        except tk.TclError:
            pass

    def close(self) -> None:
        try:
            self._top.destroy()
        except tk.TclError:
            pass
