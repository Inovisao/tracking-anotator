from app.annotation_keypoint.shared import *


class KPWorkflowActionsMixin:
    def on_accept(self):
        if self.current_frame is None:
            return
        detections_to_save = self.detections_to_save()
        if self.review_idx is not None and self.saved_records:
            record = self.saved_records[self.review_idx]
            image_id, file_name = self.store_annotations(
                detections_to_save,
                existing_image_id=record.get("image_id"),
                existing_file_name=record.get("file_name"),
            )
            self.update_saved_record(self.review_idx, detections_to_save, image_id, file_name)
            self.write_annotations()
            self.advance_after_review_accept()
            return
        image_id, file_name = self.store_annotations(detections_to_save)
        self.write_annotations()
        self.append_saved_record(detections_to_save, image_id, file_name)
        self.load_next_frame()

    def on_reject(self):
        if self.review_idx is not None:
            self.exit_review_mode()
            return
        self.load_next_frame()

    def on_quit(self):
        self.finish_processing("Processo keypoint encerrado pelo usuario.")

    def on_delete_image(self):
        record = self.current_review_record()
        delete_target = self.current_deletable_image_path()
        if record is None and delete_target is None:
            self.info_var.set("Nenhuma imagem disponivel para exclusao neste momento.")
            return
        if not messagebox.askyesno("Confirmar exclusao", "Deletar imagem e anotacoes associadas?"):
            return
        try:
            if record is not None:
                image_id = int(record.get("image_id") or 0)
                file_name = str(record.get("file_name", "")).strip()
                removed = self.delete_image_annotations(image_id)
                self.remove_image_file(file_name)
                if self.review_idx is not None and 0 <= self.review_idx < len(self.saved_records):
                    self.saved_records.pop(self.review_idx)
                self.write_annotations()
                self.info_var.set(f"Imagem deletada: {file_name} | anotacoes removidas: {removed}")
                self.exit_review_mode()
            else:
                delete_target.unlink(missing_ok=True)
                self.remove_current_image_from_sequence(delete_target)
                self.current_frame = None
                self.load_next_frame()
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[ERRO] Falha ao deletar imagem keypoint: {exc}")

    def rotate_frame_cw(self):
        self.frame_rotation = (getattr(self, "frame_rotation", 0) + 90) % 360
        self.update_display()

    def rotate_frame_ccw(self):
        self.frame_rotation = (getattr(self, "frame_rotation", 0) + 270) % 360
        self.update_display()

    def update_manual_memory_after_accept(self, detections):
        _ = detections
        return None
