import sys
from pathlib import Path

# When frozen by PyInstaller, assets live in sys._MEIPASS.
# In development, they live two levels above this file (project root).
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys._MEIPASS)
    # Executable-relative root — model.pt and dataset/ must sit next to the .exe
    _EXE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
    _EXE_DIR = BASE_DIR

LOGO_PATH = BASE_DIR / "assets" / "inovisao.png"
DATA_ROOT = _EXE_DIR / "dataset"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
IMAGE_LIST_EXTENSIONS = (".txt", ".lst")
WEIGHTS_PATH = _EXE_DIR / "model.pt"
OUTPUT_DATASET_PREFIX = "output_dataset"  # kept for parsing legacy state dir names
SAVED_STATES_SUBDIR = "saved_data_states"
CONF_THRESHOLD = 0.40
TARGET_CLASSES = []
SAVE_RECTIFIED_FRAMES = False
MANUAL_IOU_THRESHOLD = 0.30
USE_RECTIFIED_FOR_DETECTION = True
FALLBACK_TO_ORIGINAL_IF_EMPTY = True
MAX_SAVED_FRAME_CACHE = 200
SHOW_MODEL_DETECTIONS = True
SHOW_MANUAL_DETECTIONS = True
WINDOW_MARGIN_PX = 80
WINDOW_TOP_RESERVED_PX = 240
CANVAS_PADDING_PX = 80
