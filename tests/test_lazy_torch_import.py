"""Guards the startup optimization: importing the tools/wizard must not pull torch.

torch (~4s, hundreds of MB) is only acceptable when a model is actually loaded.
If someone re-adds a top-level `from ultralytics import ...` / `tracker.byte_tracker`
to a module on the import path, this test fails.
"""

import subprocess
import sys
import unittest


def _torch_loaded_after_import(import_line: str) -> bool:
    code = (
        f"import sys\n{import_line}\n"
        "print('TORCH' if 'torch' in sys.modules else 'NO_TORCH')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True,
    ).stdout
    return "TORCH" in out and "NO_TORCH" not in out


class LazyTorchImportTest(unittest.TestCase):
    def test_keypoint_tool_import_does_not_load_torch(self):
        self.assertFalse(_torch_loaded_after_import(
            "import app.annotation_keypoint.tool"))

    def test_hbb_and_obb_tool_import_does_not_load_torch(self):
        self.assertFalse(_torch_loaded_after_import(
            "import app.annotation.tool, app.annotation_obb.tool"))

    def test_wizard_import_does_not_load_torch(self):
        self.assertFalse(_torch_loaded_after_import(
            "import app.ui.startup.wizard"))


if __name__ == "__main__":
    unittest.main()
