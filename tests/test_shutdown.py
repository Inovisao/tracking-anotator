"""Guards the shutdown hardening that prevents Tcl_AsyncDelete at exit.

PhotoImage refs must be released and the window destroyed synchronously while the
Tcl interpreter is alive; any export thread must be stopped before teardown.
"""

import threading
import unittest

from app.annotation.application.lifecycle import LifecycleMixin


class _Window:
    def __init__(self):
        self.quit_calls = 0
        self.destroy_calls = 0

    def quit(self):
        self.quit_calls += 1

    def destroy(self):
        self.destroy_calls += 1


class _Tool(LifecycleMixin):
    def __init__(self):
        self.window = _Window()
        self.tk_image = "photo-ref"
        self.canvas = None


class ShutdownTest(unittest.TestCase):
    def test_destroy_releases_image_and_destroys_synchronously(self):
        tool = _Tool()
        tool._destroy_window()
        self.assertIsNone(tool.tk_image)            # PhotoImage ref dropped
        self.assertEqual(tool.window.quit_calls, 1)
        self.assertEqual(tool.window.destroy_calls, 1)

    def test_shutdown_export_is_safe_without_export(self):
        tool = _Tool()
        tool._shutdown_export_thread()              # no export attrs → must not raise
        self.assertFalse(tool._export_running)

    def test_shutdown_export_signals_and_joins(self):
        tool = _Tool()
        tool._export_cancel_event = threading.Event()
        finished = threading.Thread(target=lambda: None)
        finished.start()
        finished.join()
        tool._export_thread = finished
        tool._shutdown_export_thread()
        self.assertTrue(tool._export_cancel_event.is_set())


if __name__ == "__main__":
    unittest.main()
