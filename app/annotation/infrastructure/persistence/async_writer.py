"""Background writer for the annotations state file.

Serializing the whole COCO payload grows linearly with the dataset, so doing it inline
on every frame change freezes the UI for progressively longer (hundreds of ms once the
project reaches a few thousand annotations). Callers build the payload on the UI thread —
keeping the snapshot consistent with in-memory state — and hand it here for the disk work.

Shared by the detection, OBB and keypoint storage mixins.
"""

import json
import os
import threading
from typing import Optional


class AnnotationsAsyncWriterMixin:
    _annotations_writer_lock = None
    _annotations_writer_wakeup = None
    _annotations_writer_thread = None
    _annotations_pending_payload: Optional[dict] = None
    _annotations_writer_stop: bool = False
    _annotations_log_label: str = "Anotacoes"

    def _flush_annotations(self, data: dict):
        """Serialize and write atomically, so a crash mid-write never truncates the state."""
        tmp_path = self.annotations_path.with_name(f"{self.annotations_path.name}.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp_path, self.annotations_path)
        print(f"[INFO] {self._annotations_log_label} atualizadas em {self.annotations_path}")

    def _ensure_annotations_writer_state(self):
        if getattr(self, "_annotations_writer_lock", None) is None:
            self._annotations_writer_lock = threading.Lock()
            self._annotations_writer_wakeup = threading.Event()
            self._annotations_pending_payload = None
            self._annotations_writer_thread = None
            self._annotations_writer_stop = False

    def _queue_annotations_write(self, data: dict):
        """Hand the payload to the writer thread, collapsing pending writes.

        Only the newest payload matters — if the user advances frames faster than the disk
        keeps up, superseded snapshots are dropped instead of piling up in a queue.
        """
        self._ensure_annotations_writer_state()
        with self._annotations_writer_lock:
            self._annotations_pending_payload = data
            thread = self._annotations_writer_thread
            if thread is None or not thread.is_alive():
                self._annotations_writer_stop = False
                thread = threading.Thread(
                    target=self._annotations_writer_loop,
                    name="annotations-writer",
                    daemon=True,
                )
                self._annotations_writer_thread = thread
                thread.start()
        self._annotations_writer_wakeup.set()

    def _annotations_writer_loop(self):
        while True:
            with self._annotations_writer_lock:
                payload = self._annotations_pending_payload
                self._annotations_pending_payload = None
                should_stop = self._annotations_writer_stop
                if payload is None:
                    if should_stop:
                        return
                    # Clear under the lock, so a payload queued after this point is
                    # guaranteed to re-set the event rather than be missed.
                    self._annotations_writer_wakeup.clear()
            if payload is None:
                self._annotations_writer_wakeup.wait()
                continue
            try:
                self._flush_annotations(payload)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[ERRO] Falha ao gravar anotacoes em background: {exc}")

    def flush_pending_annotations(self, timeout: float = 10.0):
        """Drain any queued write and stop the writer. Called before teardown."""
        if getattr(self, "_annotations_writer_lock", None) is None:
            return
        with self._annotations_writer_lock:
            self._annotations_writer_stop = True
            thread = self._annotations_writer_thread
        self._annotations_writer_wakeup.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        # Whatever the writer did not pick up before stopping is written here, so a
        # pending payload is never silently lost on shutdown.
        with self._annotations_writer_lock:
            payload = self._annotations_pending_payload
            self._annotations_pending_payload = None
        if payload is not None:
            try:
                self._flush_annotations(payload)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[ERRO] Falha ao gravar anotacoes pendentes: {exc}")
