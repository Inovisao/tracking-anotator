"""Annotation session lifecycle: autosave, shutdown and main loop."""

from app.annotation.shared import *


class LifecycleMixin:
    def autosave_current_frame(self, *, reason: str = "") -> Optional[Tuple[int, str]]:
        if self.current_frame is None or getattr(self, "closed", False):
            return None
        if getattr(self, "_autosaving", False):
            return None
        file_name = self.current_frame_file_name()
        existing = self.find_image_record_by_file_name(file_name) if file_name else None
        existing_id = int(existing["id"]) if existing is not None else None
        existing_file = str(existing["file_name"]) if existing is not None else None
        try:
            self._autosaving = True
            detections = self.detections_to_save()
            image_id, saved_file = self.store_annotations(
                detections,
                existing_image_id=existing_id,
                existing_file_name=existing_file,
            )
            self.write_annotations()
            self.update_manual_memory_after_accept(detections)
            self.remember_saved_record(detections, image_id, saved_file)
            msg = f"Autosave concluido: {saved_file}"
            if reason:
                msg += f" ({reason})"
            print(f"[INFO] {msg}")
            return image_id, saved_file
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[ERRO] Falha no autosave: {exc}")
            return None
        finally:
            self._autosaving = False

    def finish_processing(self, message: str):
        if self.closed:
            self._destroy_window()
            return
        self.autosave_current_frame(reason="shutdown")
        self.closed = True
        key_mapping_dialog = getattr(self, "_key_mapping_dialog", None)
        if key_mapping_dialog is not None:
            try:
                key_mapping_dialog.destroy()
            except Exception:  # pylint: disable=broad-except
                pass
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[AVISO] Falha ao liberar fonte: {exc}")
        for shortcut in ("<Return>", "<space>", "<Escape>"):
            try:
                self.window.unbind(shortcut)
            except Exception:  # pylint: disable=broad-except
                pass
        for button_name in (
            "accept_button",
            "reject_button",
            "quit_button",
            "delete_image_button",
            "pan_button",
            "export_dataset_button",
        ):
            button = getattr(self, button_name, None)
            if button is not None:
                try:
                    button.config(state=tk.DISABLED)
                except Exception:  # pylint: disable=broad-except
                    pass
        try:
            if self.images or self.annotations:
                # Blocking: the backup copies the file right after, so it must be complete.
                self.write_annotations(blocking=True)
                self.backup_annotations_file()
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[ERRO] Falha ao salvar anotacoes no encerramento: {exc}")
        try:
            if self.homographies:
                with open(self.homography_path, "w", encoding="utf-8") as f:
                    json.dump(self.homographies, f, indent=4, ensure_ascii=False)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[ERRO] Falha ao salvar homografias no encerramento: {exc}")
        try:
            self.info_var.set(message)
        except Exception:  # pylint: disable=broad-except
            pass
        self._shutdown_export_thread()
        self.flush_pending_annotations()
        self._destroy_window()

    def _shutdown_export_thread(self):
        """Stop any background export before tearing down Tk.

        A daemon export thread alive during interpreter shutdown is a classic
        cause of 'Tcl_AsyncDelete: async handler deleted by the wrong thread'.
        """
        self._export_running = False
        event = getattr(self, "_export_cancel_event", None)
        if event is not None:
            event.set()
        thread = getattr(self, "_export_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)

    def _destroy_window(self):
        # Release Tk-owned images while the interpreter is still alive, otherwise
        # PhotoImage finalizers run after teardown and abort with Tcl_AsyncDelete.
        pending = getattr(self, "_resize_after_id", None)
        if pending is not None:
            try:
                self.window.after_cancel(pending)
            except Exception:  # pylint: disable=broad-except
                pass
            self._resize_after_id = None
        try:
            if getattr(self, "canvas", None) is not None:
                self.canvas.delete("all")
        except Exception:  # pylint: disable=broad-except
            pass
        self.tk_image = None
        try:
            self.window.quit()
        except Exception:  # pylint: disable=broad-except
            pass
        try:
            self.window.destroy()
        except Exception:  # pylint: disable=broad-except
            pass

    def run(self):
        self.window.mainloop()
