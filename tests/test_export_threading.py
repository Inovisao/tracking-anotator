"""Guards the export threading fix.

The dataset export runs in a worker thread. Touching Tk (e.g. window.after) from
that thread crashes with "Tcl_AsyncDelete: async handler deleted by the wrong
thread". The worker must only enqueue callables via _post_to_main; the main
thread drains them. These tests verify that contract without a real Tk loop.
"""

import queue
import unittest

from app.annotation.presentation.export.export_screen import ExportScreenMixin


class _FakeWindow:
    def __init__(self):
        self.after_calls = 0

    def after(self, _ms, _fn=None):
        self.after_calls += 1


class _Tool(ExportScreenMixin):
    def __init__(self, raises=False):
        self.window = _FakeWindow()
        self._export_main_queue = queue.Queue()
        self._export_running = True
        self._export_cancel_event = None
        self._raises = raises

    def perform_dataset_export(self, config, cancel_event=None):
        if self._raises:
            raise RuntimeError("boom")

    def _show_export_thread_error(self, message):
        pass

    def _on_export_thread_done(self):
        pass


class ExportThreadingTest(unittest.TestCase):
    def test_post_to_main_is_drained_on_main_thread(self):
        tool = _Tool()
        tool._export_running = False
        ran = []
        tool._post_to_main(lambda: ran.append("x"))
        tool._drain_export_queue()
        self.assertEqual(ran, ["x"])

    def test_worker_never_calls_window_after(self):
        tool = _Tool(raises=True)
        tool._run_export_thread(config=None)  # simulates the worker body
        # error + done both enqueued, and no direct Tcl call from the worker
        self.assertEqual(tool._export_main_queue.qsize(), 2)
        self.assertEqual(tool.window.after_calls, 0)

    def test_worker_success_path_enqueues_done(self):
        tool = _Tool(raises=False)
        tool._run_export_thread(config=None)
        self.assertEqual(tool.window.after_calls, 0)
        self.assertGreaterEqual(tool._export_main_queue.qsize(), 1)


if __name__ == "__main__":
    unittest.main()
