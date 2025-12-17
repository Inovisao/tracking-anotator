"""Ferramenta interativa para validar detecoes YOLO e gerar anotacoes COCO."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
from tracker.byte_tracker import BYTETracker
import signal

# ====== CONFIGURACOES ======
VIDEOS_ROOT = Path(__file__).resolve().parent / "videos"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
WEIGHTS_PATH = Path(__file__).resolve().parent / "yolo11l.pt"
OUTPUT_DIR = Path(__file__).resolve().parent / "output_dataset"
OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"
ANNOTATIONS_PATH = OUTPUT_DIR / "annotations.coco.json"
HOMOGRAPHY_PATH = OUTPUT_DIR / "homography.json"
CONF_THRESHOLD = 0.40
TARGET_CLASS = "car"
SAVE_RECTIFIED_FRAMES = False  # True = salva frames warpPerspective; False = salva frames originais
MANUAL_IOU_THRESHOLD = 0.30  # limiar para reutilizar track_id manual entre frames
USE_RECTIFIED_FOR_DETECTION = True  # True = roda YOLO no frame retificado; False = roda no frame original
FALLBACK_TO_ORIGINAL_IF_EMPTY = True  # Se retificado nao tiver deteccoes, tenta original
MAX_SAVED_FRAME_CACHE = 200  # limite de frames guardados para revisao
SHOW_MODEL_DETECTIONS = True  # desenha caixas do modelo na UI
SHOW_MANUAL_DETECTIONS = True  # desenha caixas manuais na UI


@dataclass
class Detection:
    original_bbox: np.ndarray
    warp_bbox: Optional[np.ndarray]
    confidence: float
    category_id: int
    track_id: int
    source: str  # "model" ou "manual"
    internal_id: Optional[int] = None


@dataclass
class ByteTrackerArgs:
    """Argumentos para inicializar o BYTETracker (compatibilidade YOLOX)."""

    track_thresh: float = 0.3
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 1.6
    min_box_area: float = 10
    mot20: bool = False


def order_points(pts: np.ndarray) -> np.ndarray:
    """Ordena pontos para (top-left, top-right, bottom-right, bottom-left)."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def destination_size(ordered_pts: np.ndarray) -> Tuple[int, int]:
    """Calcula largura/altura destino para homografia."""
    (tl, tr, br, bl) = ordered_pts
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    width = int(max(width_a, width_b))
    height = int(max(height_a, height_b))
    return max(width, 1), max(height, 1)


def clip_bbox(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> np.ndarray:
    """Limita bbox aos limites da imagem."""
    x1_c = max(0.0, min(float(width - 1), x1))
    y1_c = max(0.0, min(float(height - 1), y1))
    x2_c = max(0.0, min(float(width - 1), x2))
    y2_c = max(0.0, min(float(height - 1), y2))
    return np.array([x1_c, y1_c, x2_c, y2_c], dtype=np.float32)


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Calcula IoU entre duas bboxes xyxy."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0
    return inter_area / union


def bbox_center(bbox: np.ndarray) -> Tuple[float, float]:
    """Retorna centro (cx, cy) de uma bbox xyxy."""
    x1, y1, x2, y2 = bbox
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def parse_frame_number_from_name(file_name: str, video_stem: str) -> Optional[int]:
    """Extrai numero do frame a partir de {video}_frame_00001.jpg."""
    prefix = f"{video_stem}_frame_"
    if not file_name.startswith(prefix):
        return None
    try:
        number_part = file_name[len(prefix) :].split(".")[0]
        return int(number_part)
    except Exception:  # pylint: disable=broad-except
        return None


class AnnotationTool:
    """Controla a interface de validacao e a geracao do arquivo COCO."""

    def __init__(self):
        if not VIDEOS_ROOT.exists():
            raise FileNotFoundError(f"Pasta de videos nao encontrada: {VIDEOS_ROOT}")
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"Pesos nao encontrados: {WEIGHTS_PATH}")

        self.video_files = sorted(
            [p for p in VIDEOS_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
        )
        if not self.video_files:
            raise FileNotFoundError(f"Nenhum video encontrado em {VIDEOS_ROOT}")

        OUTPUT_DIR.mkdir(exist_ok=True)
        OUTPUT_IMAGES_DIR.mkdir(exist_ok=True)

        self.model = YOLO(str(WEIGHTS_PATH))
        self.default_tracker_args = ByteTrackerArgs()
        self.frame_rate = 30
        self.bytetracker = BYTETracker(self.default_tracker_args, frame_rate=self.frame_rate)
        self.cap: Optional[cv2.VideoCapture] = None

        self.video_name = ""
        self.video_path: Optional[Path] = None
        self.current_video_index = 0
        self.frame_index = 0
        self.image_id = 1
        self.annotation_id = 1
        self.frames_saved_in_current_video = 0

        self.current_frame: Optional[np.ndarray] = None
        self.current_rectified_frame: Optional[np.ndarray] = None
        self.current_detections: List[Detection] = []
        self.manual_detections: List[Detection] = []
        self.tk_image = None
        self.last_frame_shape: Optional[Tuple[int, int]] = None

        self.annotation_mode = True
        self.remove_mode = False
        self.drawing_start: Optional[Tuple[int, int]] = None
        self.drawing_rect_id: Optional[int] = None
        self.canvas_image_id: Optional[int] = None

        self.images: List[dict] = []
        self.annotations: List[dict] = []
        self.homographies: List[dict] = []
        self.categories = [{"id": 1, "name": TARGET_CLASS}]

        self.roi_points: List[Tuple[int, int]] = []
        self.roi_defined = False
        self.homography_matrix: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None
        self.warp_size: Optional[Tuple[int, int]] = None
        self.roi_polygon: Optional[np.ndarray] = None
        self.dest_points: Optional[np.ndarray] = None

        self.manual_track_memory: Dict[int, Dict[str, np.ndarray]] = {}
        self.global_track_counter = 1
        self.track_history: Dict[int, List[dict]] = {}
        self.recent_tracks: List[dict] = []
        self.history_window = 5
        self.tracker_id_map: Dict[int, int] = {}  # internal_byte_id -> global_id
        self.manual_id_var: Optional[tk.StringVar] = None
        self.edit_id_mode = False
        self.selected_detection: Optional[Tuple[str, int]] = None
        self.offset_x = 0
        self.offset_y = 0

        self.saved_records: List[dict] = []
        self.review_idx: Optional[int] = None
        self.live_snapshot: Optional[dict] = None
        self.closed = False

        self.window = tk.Tk()
        self.window.title("Validador de deteccoes")
        self.window.protocol("WM_DELETE_WINDOW", self.on_quit)
        self.manual_id_var = tk.StringVar(value="")

        self.info_var = tk.StringVar(value="Selecione 4 pontos do ROI.")
        self.info_label = tk.Label(self.window, textvariable=self.info_var, font=("Arial", 12))
        self.info_label.pack(pady=10)

        buttons_frame = tk.Frame(self.window)
        buttons_frame.pack(pady=10)

        self.accept_button = tk.Button(
            buttons_frame, text="Validar (Enter)", command=self.on_accept, width=18, state=tk.DISABLED
        )
        self.accept_button.grid(row=0, column=0, padx=5)

        self.reject_button = tk.Button(
            buttons_frame, text="Rejeitar (Espaco)", command=self.on_reject, width=18, state=tk.DISABLED
        )
        self.reject_button.grid(row=0, column=1, padx=5)

        self.quit_button = tk.Button(buttons_frame, text="Sair (Esc)", command=self.on_quit, width=18)
        self.quit_button.grid(row=0, column=2, padx=5)

        self.annotation_button = tk.Button(
            buttons_frame,
            text="Modo anotacao ON (K)",
            command=self.toggle_annotation_mode,
            width=22,
            state=tk.DISABLED,
        )
        self.annotation_button.grid(row=0, column=3, padx=5)

        self.remove_button = tk.Button(
            buttons_frame,
            text="Remover anotacao OFF",
            command=self.toggle_remove_mode,
            width=22,
            state=tk.DISABLED,
        )
        self.remove_button.grid(row=0, column=4, padx=5)

        self.roi_button = tk.Button(
            buttons_frame,
            text="Redefinir ROI (R)",
            command=self.reset_roi,
            width=18,
        )
        self.roi_button.grid(row=0, column=5, padx=5)

        self.prev_button = tk.Button(buttons_frame, text="< Frame anterior", command=self.on_prev_saved, width=18)
        self.prev_button.grid(row=1, column=0, padx=5, pady=(6, 0))

        self.next_button = tk.Button(buttons_frame, text="Proximo frame >", command=self.on_next_saved, width=18)
        self.next_button.grid(row=1, column=1, padx=5, pady=(6, 0))

        self.manual_id_label = tk.Label(buttons_frame, text="ID manual:")
        self.manual_id_label.grid(row=1, column=2, padx=5, pady=(6, 0))
        self.manual_id_entry = tk.Entry(buttons_frame, textvariable=self.manual_id_var, width=10)
        self.manual_id_entry.grid(row=1, column=3, padx=5, pady=(6, 0))
        self.apply_id_button = tk.Button(
            buttons_frame, text="Aplicar ID", command=self.apply_manual_id_to_selection, width=12, state=tk.DISABLED
        )
        self.apply_id_button.grid(row=1, column=4, padx=5, pady=(6, 0))
        self.edit_id_button = tk.Button(
            buttons_frame, text="Editar ID OFF (E)", command=self.toggle_edit_id_mode, width=16, state=tk.DISABLED
        )
        self.edit_id_button.grid(row=1, column=5, padx=5, pady=(6, 0))

        self.window.bind("<Return>", lambda event: self.on_accept())
        self.window.bind("<space>", lambda event: self.on_reject())
        self.window.bind("<Escape>", lambda event: self.on_quit())
        self.window.bind("k", lambda event: self.toggle_annotation_mode())
        self.window.bind("K", lambda event: self.toggle_annotation_mode())
        self.window.bind("r", lambda event: self.reset_roi())
        self.window.bind("R", lambda event: self.reset_roi())
        self.window.bind("e", lambda event: self.toggle_edit_id_mode())
        self.window.bind("E", lambda event: self.toggle_edit_id_mode())
        self.window.bind("<Left>", lambda event: self.on_prev_saved())
        self.window.bind("<Right>", lambda event: self.on_next_saved())

        self.canvas = tk.Canvas(self.window, bg="black", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.load_existing_annotations()
        self.register_signal_handlers()
        self.start_video(self.current_video_index)

    # ===================== CONTROLE DE VIDEOS =====================
    def load_existing_annotations(self):
        """Carrega anotacoes existentes para continuar de onde parou."""
        if not ANNOTATIONS_PATH.exists():
            return
        try:
            with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[AVISO] Falha ao ler anotacoes existentes: {exc}")
            return
        self.images = data.get("images", [])
        self.annotations = data.get("annotations", [])
        cats = data.get("categories")
        if cats:
            self.categories = cats
        max_ann_id = max((ann.get("id", 0) for ann in self.annotations), default=0)
        max_img_id = max((img.get("id", 0) for img in self.images), default=0)
        self.annotation_id = max_ann_id + 1
        self.image_id = max_img_id + 1
        max_track = max((ann.get("track_id", 0) for ann in self.annotations), default=0)
        self.global_track_counter = max(max_track + 1, 1)
        print(
            f"[INFO] Anotacoes carregadas. imagens={len(self.images)}, "
            f"anotacoes={len(self.annotations)}, prox_image_id={self.image_id}, prox_annotation_id={self.annotation_id}"
        )

    def register_signal_handlers(self):
        """Garante que a janela feche se o processo receber SIGINT/SIGTERM."""
        def handler(signum, frame):
            _ = frame  # unused
            print(f"[INFO] Encerrando por sinal {signum}.")
            self.finish_processing("Processo encerrado pelo sistema.")

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except Exception:  # pylint: disable=broad-except
                pass

    def start_video(self, index: int):
        """Abre o video indicado e prepara para anotacao."""
        if index < 0 or index >= len(self.video_files):
            self.finish_processing("Todos os videos processados.")
            return

        if self.cap is not None:
            self.cap.release()

        if hasattr(self.model, "reset"):
            try:
                self.model.reset()
            except Exception:  # pylint: disable=broad-except
                pass
        elif hasattr(self.model, "tracker"):
            try:
                self.model.tracker = None
            except Exception:  # pylint: disable=broad-except
                pass

        self.video_path = self.video_files[index]
        self.video_name = self.video_path.stem
        self.current_video_index = index
        self.frame_index = 0
        self.frames_saved_in_current_video = 0
        self.manual_track_memory.clear()
        self.saved_records.clear()
        self.review_idx = None
        self.live_snapshot = None
        self.recent_tracks.clear()
        self.tracker_id_map.clear()
        self.edit_id_mode = False
        self.selected_detection = None
        last_frame_saved = 0
        saved_for_video = [img for img in self.images if img.get("video") in (str(self.video_path), self.video_name)]
        if saved_for_video:
            for img in saved_for_video:
                fn = img.get("file_name", "")
                parsed = parse_frame_number_from_name(fn, self.video_name)
                if parsed is not None:
                    last_frame_saved = max(last_frame_saved, parsed)
        self.frames_saved_in_current_video = last_frame_saved
        self.frame_index = last_frame_saved

        self.roi_points = []
        self.roi_defined = False
        self.homography_matrix = None
        self.inverse_homography = None
        self.warp_size = None
        self.roi_polygon = None
        self.dest_points = None
        self.current_rectified_frame = None
        self.current_detections = []
        self.manual_detections = []
        self.selected_detection = None

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            print(f"[ERRO] Falha ao abrir video: {self.video_path}")
            self.start_video(index + 1)
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_rate = int(fps) if fps and fps > 1 else 30
        self.bytetracker = BYTETracker(ByteTrackerArgs(), frame_rate=self.frame_rate)
        if self.frame_index > 0:
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(self.frame_index))
            except Exception:  # pylint: disable=broad-except
                pass

        ret, first_frame = self.cap.read()
        if not ret or first_frame is None:
            print(f"[ERRO] Falha ao ler o primeiro frame: {self.video_path}")
            self.start_video(index + 1)
            return

        self.current_frame = first_frame
        self.disable_controls_for_roi()
        self.info_var.set(
            f"[Video {index + 1}/{len(self.video_files)}] {self.video_name}: selecione 4 pontos do ROI."
        )
        self.update_display()

    def finish_current_video(self):
        """Fecha o video atual e segue para o proximo se existir."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        next_index = self.current_video_index + 1
        if next_index < len(self.video_files):
            print(f"[INFO] Video concluido: {self.video_name}. Iniciando proximo.")
            self.start_video(next_index)
        else:
            self.finish_processing("Todos os videos processados.")

    # ===================== ROI & HOMOGRAFIA =====================
    def reset_roi(self):
        """Permite redefinir ROI (desde que o video atual nao tenha frames salvos)."""
        if self.frames_saved_in_current_video > 0:
            print("[AVISO] Nao e possivel redefinir ROI apos salvar frames do video atual.")
            return
        self.roi_points = []
        self.roi_defined = False
        self.homography_matrix = None
        self.inverse_homography = None
        self.warp_size = None
        self.roi_polygon = None
        self.dest_points = None
        self.current_rectified_frame = None
        self.current_detections = []
        self.manual_detections = []
        self.edit_id_mode = False
        self.selected_detection = None
        self.disable_controls_for_roi()
        self.info_var.set("Selecione 4 pontos do ROI.")
        self.update_display()

    def add_roi_point(self, x: int, y: int):
        """Registra ponto clicado para ROI."""
        if len(self.roi_points) >= 4:
            return
        # permite ROI fora das bordas sem clamp
        self.roi_points.append((float(x), float(y)))
        if len(self.roi_points) == 4:
            self.compute_homography()
        self.update_display()

    def compute_homography(self):
        """Calcula homografia a partir dos 4 pontos clicados."""
        if len(self.roi_points) != 4:
            return
        src = order_points(np.array(self.roi_points, dtype=np.float32))
        width, height = destination_size(src)
        dst = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.float32,
        )
        self.homography_matrix = cv2.getPerspectiveTransform(src, dst)
        self.inverse_homography = cv2.getPerspectiveTransform(dst, src)
        self.warp_size = (width, height)
        self.roi_polygon = src
        self.dest_points = dst
        self.roi_defined = True
        self.enable_controls_after_roi()
        self.save_homography_file()
        print(f"[INFO] ROI definida com tamanho retificado {width}x{height}.")
        if self.current_frame is not None:
            self.process_current_frame(self.current_frame)

    def warp_frame(self, frame: np.ndarray) -> np.ndarray:
        """Aplica warpPerspective ao frame atual usando a homografia."""
        if self.homography_matrix is None or self.warp_size is None:
            return frame
        return cv2.warpPerspective(frame, self.homography_matrix, self.warp_size)

    def project_bbox(
        self, bbox: np.ndarray, matrix: Optional[np.ndarray], width: int, height: int
    ) -> np.ndarray:
        """Projeta bbox xyxy usando a matriz de homografia fornecida."""
        if matrix is None:
            return bbox.astype(np.float32)
        pts = np.array(
            [
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
            ],
            dtype=np.float32,
        ).reshape((1, 4, 2))
        projected = cv2.perspectiveTransform(pts, matrix)[0]
        xs = projected[:, 0]
        ys = projected[:, 1]
        return clip_bbox(xs.min(), ys.min(), xs.max(), ys.max(), width, height)

    def is_inside_roi(self, bbox: np.ndarray) -> bool:
        """Verifica se a bbox está majoritariamente dentro do ROI usando pointPolygonTest (robusto)."""
        if self.roi_polygon is None:
            return True
        
        x1, y1, x2, y2 = bbox
        box_pts = np.array(
            [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ],
            dtype=np.float32,
        )
        
        poly = self.roi_polygon.astype(np.float32)

        inside_count = 0
        for (px, py) in box_pts:
            # retorna >0 se dentro, 0 se na borda, <0 se fora
            if cv2.pointPolygonTest(poly, (px, py), False) >= 0:
                inside_count += 1

        # critério robusto: 3 ou 4 vértices da bbox dentro do ROI
        return inside_count >= 3


    def save_homography_file(self):
        """Persiste homografia em disco (lista para multiplos videos)."""
        if self.homography_matrix is None or self.inverse_homography is None or self.dest_points is None:
            return
        if self.video_path is None:
            return
        payload = {
            "video": str(self.video_path),
            "video_name": self.video_name,
            "roi_points": [[int(x), int(y)] for (x, y) in self.roi_points],
            "ordered_roi": self.roi_polygon.tolist() if self.roi_polygon is not None else None,
            "destination_points": self.dest_points.tolist(),
            "warp_size": {"width": self.warp_size[0], "height": self.warp_size[1]} if self.warp_size else None,
            "homography": self.homography_matrix.tolist(),
            "inverse_homography": self.inverse_homography.tolist(),
        }
        self.homographies = [h for h in self.homographies if h.get("video") != str(self.video_path)]
        self.homographies.append(payload)
        with open(HOMOGRAPHY_PATH, "w", encoding="utf-8") as f:
            json.dump(self.homographies, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Homografia salva/atualizada em {HOMOGRAPHY_PATH}")

    # ===================== INTERFACE =====================
    def enable_controls_after_roi(self):
        """Habilita botoes apos definir ROI."""
        self.accept_button.config(state=tk.NORMAL)
        self.reject_button.config(state=tk.NORMAL)
        self.annotation_button.config(state=tk.NORMAL)
        self.remove_button.config(state=tk.NORMAL)
        self.apply_id_button.config(state=tk.NORMAL)
        self.edit_id_button.config(state=tk.NORMAL)
        self.info_var.set(self.build_status_message())

    def disable_controls_for_roi(self):
        """Desabilita botoes enquanto ROI nao for definido."""
        self.accept_button.config(state=tk.DISABLED)
        self.reject_button.config(state=tk.DISABLED)
        self.annotation_button.config(state=tk.DISABLED)
        self.remove_button.config(state=tk.DISABLED)
        self.apply_id_button.config(state=tk.DISABLED)
        self.edit_id_button.config(state=tk.DISABLED)

    def update_display(self):
        """Renderiza o frame com ROI, deteccoes e anotacoes manuais."""
        if self.current_frame is None:
            return

        annotated = self.current_frame.copy()

        if self.roi_points:
            pts = np.array(self.roi_points, dtype=np.int32)
            is_closed = len(self.roi_points) == 4
            cv2.polylines(annotated, [pts], isClosed=is_closed, color=(255, 0, 0), thickness=2)
            for idx, (x, y) in enumerate(self.roi_points):
                xi, yi = int(round(x)), int(round(y))
                cv2.circle(annotated, (xi, yi), 4, (0, 0, 255), -1)
                cv2.putText(
                    annotated,
                    str(idx + 1),
                    (xi + 4, yi - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

        self.validate_selected_detection()

        if SHOW_MODEL_DETECTIONS:
            annotated = self.draw_detections(annotated, self.current_detections, "model")
        if SHOW_MANUAL_DETECTIONS:
            annotated = self.draw_detections(annotated, self.manual_detections, "manual")

        height, width = annotated.shape[:2]
        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        self.tk_image = ImageTk.PhotoImage(image=pil_image)

        self.canvas.delete("all")
        EXPAND = 500
        canvas_w = width + EXPAND
        canvas_h = height + EXPAND
        self.offset_x = (canvas_w - width) // 2
        self.offset_y = (canvas_h - height) // 2
        self.canvas.config(width=canvas_w, height=canvas_h)
        self.canvas_image_id = self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image)

        if self.roi_points:
            shifted = [(x + self.offset_x, y + self.offset_y) for (x, y) in self.roi_points]
            if len(shifted) >= 2:
                for i in range(len(shifted) - 1):
                    self.canvas.create_line(
                        shifted[i][0],
                        shifted[i][1],
                        shifted[i + 1][0],
                        shifted[i + 1][1],
                        fill="blue",
                        width=2,
                    )
                if len(shifted) == 4:
                    self.canvas.create_line(
                        shifted[-1][0],
                        shifted[-1][1],
                        shifted[0][0],
                        shifted[0][1],
                        fill="blue",
                        width=2,
                    )
            for sx, sy in shifted:
                self.canvas.create_oval(sx - 3, sy - 3, sx + 3, sy + 3, fill="red", outline="")

        if self.drawing_rect_id is not None and self.drawing_start is not None:
            x, y = self.drawing_start
            self.drawing_rect_id = self.canvas.create_rectangle(
                x, y, x, y, outline="yellow", width=2, dash=(4, 2)
            )

        self.last_frame_shape = (width, height)
        self.update_status()

    def draw_detections(self, frame: np.ndarray, detections: List[Detection], source_tag: str):
        """Desenha as caixas detectadas no frame visivel."""
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det.original_bbox.astype(int)
            color = (0, 255, 0) if det.source == "model" else (0, 255, 255)
            thickness = 2
            if self.selected_detection == (source_tag, idx):
                color = (0, 0, 255)
                thickness = 3
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"ID {det.track_id}"
            if det.source == "model":
                label += f" {det.confidence * 100:.1f}%"
            else:
                label += " manual"
            if self.selected_detection == (source_tag, idx):
                label = f"[selecionada] {label}"
            cv2.putText(frame, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def build_status_message(self) -> str:
        """Gera mensagem de status para a barra de informacoes."""
        base = (
            f"[{self.current_video_index + 1}/{len(self.video_files)}] {self.video_name} | "
            f"Frame {self.frame_index} | Deteccoes validas (> {CONF_THRESHOLD*100:.0f}%): "
            f"{len(self.current_detections)}"
        )
        if self.review_idx is not None and self.saved_records:
            base = (
                f"[Revisao {self.review_idx + 1}/{len(self.saved_records)}] "
                f"{self.video_name} | Frame {self.frame_index} | Deteccoes: {len(self.current_detections)}"
            )
        if self.last_frame_shape:
            width, height = self.last_frame_shape
            base += f" | Resolucao: {width}x{height}"
        if self.roi_defined and self.warp_size:
            base += f" | ROI retificado: {self.warp_size[0]}x{self.warp_size[1]}"
        else:
            base += f" | ROI: {len(self.roi_points)}/4 pontos"
        base += f" | Modo anotacao: {'ON' if self.annotation_mode else 'OFF'}"
        base += f" | Remover anotacao: {'ON' if self.remove_mode else 'OFF'}"
        base += f" | Editar ID: {'ON' if self.edit_id_mode else 'OFF'}"
        selected = self.get_selected_detection()
        if selected is not None:
            base += f" | Selecionada ID {selected.track_id}"
        if self.manual_detections:
            base += f" | BBoxes manuais: {len(self.manual_detections)}"
        base += f" | Salvando frames {'retificados' if SAVE_RECTIFIED_FRAMES else 'originais'}"
        return base

    def update_status(self):
        """Atualiza o texto de status exibido na interface."""
        self.info_var.set(self.build_status_message())
        self.update_annotation_button()
        self.update_remove_button()
        self.update_edit_id_button()

    def update_annotation_button(self):
        """Atualiza o texto do botao de modo de anotacao."""
        if hasattr(self, "annotation_button"):
            estado = "ON" if self.annotation_mode else "OFF"
            self.annotation_button.config(text=f"Modo anotacao {estado} (K)")

    def update_remove_button(self):
        """Atualiza o texto do botao de remocao."""
        if hasattr(self, "remove_button"):
            estado = "ON" if self.remove_mode else "OFF"
            self.remove_button.config(text=f"Remover anotacao {estado}")

    def update_edit_id_button(self):
        """Atualiza o texto do botao de edicao de ID."""
        if hasattr(self, "edit_id_button"):
            estado = "ON" if self.edit_id_mode else "OFF"
            self.edit_id_button.config(text=f"Editar ID {estado} (E)")

    # ===================== EVENTOS DE MOUSE =====================
    def on_mouse_down(self, event):
        """Inicia ROI ou desenho de anotacao manual."""
        if self.current_frame is None:
            return

        x = event.x - self.offset_x
        y = event.y - self.offset_y

        if not self.roi_defined:
            self.add_roi_point(x, y)
            return

        if self.edit_id_mode:
            self.select_detection_at(x, y)
            return

        if self.remove_mode:
            self.remove_annotation_at(x, y)
            return

        if not self.annotation_mode:
            return

        self.drawing_start = (x, y)
        self.drawing_rect_id = self.canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="yellow",
            width=2,
            dash=(4, 2),
        )

    def on_mouse_drag(self, event):
        """Atualiza a caixa em desenho."""
        if self.remove_mode or not self.annotation_mode or self.drawing_start is None or not self.roi_defined:
            return

        if self.drawing_rect_id is None:
            self.drawing_rect_id = self.canvas.create_rectangle(
                self.drawing_start[0],
                self.drawing_start[1],
                event.x,
                event.y,
                outline="yellow",
                width=2,
                dash=(4, 2),
            )
        self.canvas.coords(
            self.drawing_rect_id,
            self.drawing_start[0] + self.offset_x,
            self.drawing_start[1] + self.offset_y,
            event.x,
            event.y,
        )

    def on_mouse_up(self, event):
        """Finaliza a caixa manual e a armazena para o frame atual."""
        if self.remove_mode or not self.annotation_mode or self.drawing_start is None or not self.roi_defined:
            return

        start_x, start_y = self.drawing_start
        end_x = event.x - self.offset_x
        end_y = event.y - self.offset_y

        if self.drawing_rect_id is not None:
            self.canvas.delete(self.drawing_rect_id)
            self.drawing_rect_id = None

        self.drawing_start = None

        x1, x2 = sorted((start_x, end_x))
        y1, y2 = sorted((start_y, end_y))

        if abs(x2 - x1) < 3 or abs(y2 - y1) < 3:
            return

        bbox = np.array([int(x1), int(y1), int(x2), int(y2)], dtype=np.float32)
        if not self.is_inside_roi(bbox):
            print("[INFO] Caixa manual ignorada pois esta fora do ROI.")
            return

        track_id = self.consume_manual_id_override()
        if track_id is None:
            track_id = self.match_manual_to_history(bbox)
        if track_id is None:
            track_id = self.new_track_id()
        warp_bbox = None
        if self.homography_matrix is not None and self.warp_size is not None:
            warp_bbox = self.project_bbox(bbox, self.homography_matrix, self.warp_size[0], self.warp_size[1])

        manual_det = Detection(
            original_bbox=bbox,
            warp_bbox=warp_bbox,
            confidence=1.0,
            category_id=1,
            track_id=track_id,
            source="manual",
        )
        self.manual_detections.append(manual_det)
        self.track_history.setdefault(track_id, []).append({"frame": self.frame_index, "bbox": bbox.tolist()})
        # garante que anotacao manual tambem alimente o historico recente
        self.recent_tracks.append({"frame": self.frame_index, "tracks": [{"id": track_id, "bbox": bbox.copy()}]})
        if len(self.recent_tracks) > self.history_window:
            self.recent_tracks.pop(0)
        self.update_display()

    def remove_annotation_at(self, x: int, y: int) -> bool:
        """Remove caixa que contenha o ponto (x, y) se existir."""
        for idx in range(len(self.manual_detections) - 1, -1, -1):
            det = self.manual_detections[idx]
            x1, y1, x2, y2 = det.original_bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                del self.manual_detections[idx]
                self.selected_detection = None
                print("[INFO] Caixa manual removida.")
                self.update_display()
                return True

        for idx in range(len(self.current_detections) - 1, -1, -1):
            det = self.current_detections[idx]
            x1, y1, x2, y2 = det.original_bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                del self.current_detections[idx]
                self.selected_detection = None
                print("[INFO] Deteccao removida.")
                self.update_display()
                return True

        print("[INFO] Nenhuma caixa encontrada para remover.")
        return False

    # ===================== BOTOES =====================
    def toggle_annotation_mode(self):
        """Alterna o modo de anotacao manual ativado pelo atalho 'k'."""
        if self.current_frame is None or not self.roi_defined:
            return

        self.annotation_mode = not self.annotation_mode
        if self.annotation_mode and self.edit_id_mode:
            self.edit_id_mode = False
            self.selected_detection = None
        if self.annotation_mode and self.remove_mode:
            self.remove_mode = False

        if not self.annotation_mode:
            if self.drawing_rect_id is not None:
                self.canvas.delete(self.drawing_rect_id)
                self.drawing_rect_id = None
            self.drawing_start = None
        estado_msg = "ativado" if self.annotation_mode else "desativado"
        print(f"[INFO] Modo anotacao manual {estado_msg}. Clique e arraste para desenhar caixas.")
        self.update_status()

    def toggle_remove_mode(self):
        """Alterna o modo de remocao de anotacoes."""
        if self.current_frame is None or not self.roi_defined:
            return

        self.remove_mode = not self.remove_mode
        if self.remove_mode and self.edit_id_mode:
            self.edit_id_mode = False
            self.selected_detection = None
        if self.remove_mode:
            if self.annotation_mode:
                self.annotation_mode = False
            if self.drawing_rect_id is not None:
                self.canvas.delete(self.drawing_rect_id)
                self.drawing_rect_id = None
            self.drawing_start = None
        else:
            if not self.annotation_mode:
                self.annotation_mode = True
        estado_msg = "ativado" if self.remove_mode else "desativado"
        print(f"[INFO] Modo remover anotacao {estado_msg}. Clique sobre uma caixa para remove-la.")
        self.update_status()

    def toggle_edit_id_mode(self):
        """Alterna o modo de selecao/edicao de track_id."""
        if self.current_frame is None or not self.roi_defined:
            return

        self.edit_id_mode = not self.edit_id_mode
        if self.edit_id_mode:
            if self.annotation_mode:
                self.annotation_mode = False
            if self.remove_mode:
                self.remove_mode = False
        else:
            if not self.annotation_mode:
                self.annotation_mode = True
            self.selected_detection = None
        estado_msg = "ativado" if self.edit_id_mode else "desativado"
        print(f"[INFO] Modo editar ID {estado_msg}. Clique em uma caixa para selecionar.")
        self.update_status()

    # ===================== PROCESSAMENTO DE FRAMES =====================
    def process_current_frame(self, frame: np.ndarray):
        """Aplica warp, roda modelo e atualiza tela para o frame fornecido."""
        if frame is None:
            return

        self.review_idx = None
        self.live_snapshot = None

        self.frame_index += 1
        self.current_frame = frame
        self.current_rectified_frame = self.warp_frame(frame)
        self.current_detections = self.run_model(frame)
        self.manual_detections = []
        self.selected_detection = None
        self.annotation_mode = True
        self.remove_mode = False
        self.drawing_start = None
        if self.drawing_rect_id is not None:
            self.canvas.delete(self.drawing_rect_id)
            self.drawing_rect_id = None
        self.update_annotation_button()
        self.update_remove_button()
        self.update_display()

    def load_next_frame(self):
        """Carrega o proximo frame do video e atualiza a tela."""
        if self.review_idx is not None:
            return
        if not self.roi_defined:
            print("[INFO] Defina o ROI antes de processar frames.")
            return

        if self.cap is None:
            self.finish_current_video()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.finish_current_video()
            return

        self.process_current_frame(frame)

    def run_model(self, original_frame: np.ndarray) -> List[Detection]:
        """Executa YOLO (detecao) + ByteTrack para manter IDs estaveis."""
        detections: List[Detection] = []
        if original_frame is None:
            return detections

        img_height, img_width = original_frame.shape[:2]

        # 1) YOLO detecta
        yolo_result = self.model(original_frame, verbose=False)[0]
        names = yolo_result.names

        dets = []
        scores = []
        for box in getattr(yolo_result, "boxes", []):
            conf = float(box.conf)
            cls_id = int(box.cls)
            label = names.get(cls_id, str(cls_id))
            if conf < CONF_THRESHOLD or label != TARGET_CLASS:
                continue
            xyxy = box.xyxy.cpu().numpy()[0]
            xyxy[0::2] = np.clip(xyxy[0::2], 0, img_width - 1)
            xyxy[1::2] = np.clip(xyxy[1::2], 0, img_height - 1)
            dets.append(xyxy)
            scores.append(conf)

        img_info = (img_height, img_width)
        img_size = (img_height, img_width)

        # 2) Se nao ha deteccoes, apenas atualiza tracker
        if not dets:
            empty = np.empty((0, 5), dtype=np.float32)
            self.bytetracker.update(empty, img_info, img_size)
            return detections

        detections_bt = np.concatenate(
            [np.array(dets, dtype=np.float32), np.array(scores, dtype=np.float32).reshape(-1, 1)], axis=1
        )

        # 3) ByteTrack
        tracks = self.bytetracker.update(detections_bt, img_info, img_size)

        # 4) Converter para Detection, aplicar ROI/homografia
        for track in tracks:
            tlbr = track.tlbr
            internal_id = int(track.track_id)
            track_id = self.get_global_id(internal_id)
            score = float(track.score)
            original_box = clip_bbox(tlbr[0], tlbr[1], tlbr[2], tlbr[3], img_width, img_height)
            if not self.is_inside_roi(original_box):
                continue
            warp_box = None
            if self.homography_matrix is not None and self.warp_size is not None:
                warp_box = self.project_bbox(original_box, self.homography_matrix, self.warp_size[0], self.warp_size[1])
            self.track_history.setdefault(track_id, []).append({"frame": self.frame_index, "bbox": original_box.tolist()})
            detections.append(
                Detection(
                    original_bbox=original_box,
                    warp_bbox=warp_box,
                    confidence=score,
                    category_id=1,
                    track_id=track_id,
                    source="model",
                    internal_id=internal_id,
                )
            )

        # atualiza historico recente imediatamente
        frame_tracks = [{"id": det.track_id, "bbox": det.original_bbox.copy()} for det in detections]
        self.recent_tracks.append({"frame": self.frame_index, "tracks": frame_tracks})
        if len(self.recent_tracks) > self.history_window:
            self.recent_tracks.pop(0)

        return detections

    def consume_manual_id_override(self) -> Optional[int]:
        """Consome ID manual digitado pelo usuario (se valido)."""
        raw = self.manual_id_var.get().strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            print("[AVISO] ID manual invalido, informe um numero inteiro.")
            return None
        if value <= 0:
            print("[AVISO] ID manual deve ser > 0.")
            return None
        self.manual_id_var.set("")
        if value >= self.global_track_counter:
            self.global_track_counter = value + 1
        return value

    def new_track_id(self) -> int:
        """Gera novo track_id sequencial e inicia historico."""
        tid = self.global_track_counter
        self.global_track_counter += 1
        self.track_history[tid] = []
        return tid

    def get_global_id(self, internal_id: int) -> int:
        """Mapeia ID interno do ByteTrack para ID global fixo e sequencial."""
        if internal_id not in self.tracker_id_map:
            self.tracker_id_map[internal_id] = self.new_track_id()
        return self.tracker_id_map[internal_id]

    def match_manual_to_history(self, manual_bbox: np.ndarray) -> Optional[int]:
        """Associa anotacao manual priorizando deteccoes do frame atual, depois historico recente."""
        cx1, cy1 = bbox_center(manual_bbox)

        # 1) prioridade absoluta: deteccoes do frame atual
        for det in self.current_detections:
            iou = bbox_iou(manual_bbox, det.original_bbox)
            if iou > 0.4:
                return det.track_id

        # 2) fallback: frames anteriores
        best_id = None
        best_score = 0.0
        for frame_data in reversed(self.recent_tracks):
            for tr in frame_data.get("tracks", []):
                iou = bbox_iou(manual_bbox, tr["bbox"])
                if iou < 0.1:
                    continue
                cx2, cy2 = bbox_center(tr["bbox"])
                dist = np.hypot(cx1 - cx2, cy1 - cy2)
                combined = iou - 0.001 * dist
                if combined > best_score:
                    best_score = combined
                    best_id = tr["id"]

        if best_score > 0.2:
            return best_id

        return None

    # ===================== CONTROLE DE FLUXO =====================
    def on_accept(self):
        """Persistir anotacoes quando o usuario aprova o frame."""
        if self.current_frame is None or not self.roi_defined:
            return
        detections_to_save = list(self.current_detections) + list(self.manual_detections)
        if detections_to_save:
            if self.review_idx is not None and self.saved_records:
                record = self.saved_records[self.review_idx]
                image_id, file_name = self.store_annotations(
                    detections_to_save, existing_image_id=record["image_id"], existing_file_name=record["file_name"]
                )
                self.update_manual_memory_after_accept(detections_to_save)
                self.update_saved_record(self.review_idx, detections_to_save, image_id, file_name)
                self.write_annotations()
                self.advance_after_review_accept()
                return
            image_id, file_name = self.store_annotations(detections_to_save)
            self.write_annotations()
            self.update_manual_memory_after_accept(detections_to_save)
            self.append_saved_record(detections_to_save, image_id, file_name)
        self.load_next_frame()

    def on_reject(self):
        """Ignora o frame atual e avanca para o proximo."""
        if self.review_idx is not None:
            self.exit_review_mode()
            return
        if not self.roi_defined:
            return
        self.load_next_frame()

    def on_quit(self):
        """Encerra o processo de anotacao."""
        self.finish_processing("Processo encerrado pelo usuario.")

    # ===================== ANOTACOES =====================
    def update_manual_memory_after_accept(self, detections: List[Detection]):
        """Atualiza memoria de tracks manuais para reaproveitar ids."""
        for det in detections:
            if det.source != "manual":
                continue
            self.manual_track_memory[det.track_id] = {"bbox": det.original_bbox.copy()}

    @staticmethod
    def clone_detection(det: Detection) -> Detection:
        """Cria uma copia profunda de Detection."""
        return Detection(
            original_bbox=det.original_bbox.copy(),
            warp_bbox=det.warp_bbox.copy() if det.warp_bbox is not None else None,
            confidence=det.confidence,
            category_id=det.category_id,
            track_id=det.track_id,
            source=det.source,
            internal_id=det.internal_id,
        )

    def append_saved_record(self, detections: List[Detection], image_id: int, file_name: str):
        """Guarda frame anotado para revisao posterior."""
        if self.current_frame is None:
            return
        record = {
            "image_id": image_id,
            "file_name": file_name,
            "frame_index": self.frame_index,
            "frame": self.current_frame.copy(),
            "rectified_frame": self.current_rectified_frame.copy() if self.current_rectified_frame is not None else None,
            "detections": [self.clone_detection(det) for det in detections],
        }
        self.saved_records.append(record)
        if len(self.saved_records) > MAX_SAVED_FRAME_CACHE:
            self.saved_records.pop(0)
            if self.review_idx is not None:
                self.review_idx = max(0, self.review_idx - 1)
        self.review_idx = None
        self.live_snapshot = None

    def update_saved_record(self, idx: int, detections: List[Detection], image_id: int, file_name: str):
        """Atualiza um registro salvo apos reanotacao."""
        if idx < 0 or idx >= len(self.saved_records) or self.current_frame is None:
            return
        self.saved_records[idx] = {
            "image_id": image_id,
            "file_name": file_name,
            "frame_index": self.frame_index,
            "frame": self.current_frame.copy(),
            "rectified_frame": self.current_rectified_frame.copy() if self.current_rectified_frame is not None else None,
            "detections": [self.clone_detection(det) for det in detections],
        }

    def rebuild_detections_from_annotations(self, image_id: int, width: int, height: int) -> List[Detection]:
        """Reconstrui deteccoes a partir das anotacoes salvas."""
        dets: List[Detection] = []
        for ann in self.annotations:
            if ann.get("image_id") != image_id:
                continue
            bbox = ann.get("bbox", [0, 0, 0, 0])
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h
            clipped = clip_bbox(x1, y1, x2, y2, width, height)
            dets.append(
                Detection(
                    original_bbox=clipped,
                    warp_bbox=None,
                    confidence=float(ann.get("score", 1.0)),
                    category_id=int(ann.get("category_id", 1)),
                    track_id=int(ann.get("track_id", -1)),
                    source=ann.get("source", "manual"),
                    internal_id=None,
                )
            )
        return dets

    def go_to_saved_frame(self, idx: int):
        """Entra em modo revisao e carrega um frame salvo."""
        if not self.saved_records or idx < 0 or idx >= len(self.saved_records):
            print("[INFO] Nenhum frame salvo para revisar.")
            return
        if self.review_idx is None and self.current_frame is not None:
            self.live_snapshot = {
                "frame": self.current_frame.copy(),
                "rectified": self.current_rectified_frame.copy() if self.current_rectified_frame is not None else None,
                "detections": [self.clone_detection(det) for det in self.current_detections],
                "manual_detections": [self.clone_detection(det) for det in self.manual_detections],
                "frame_index": self.frame_index,
            }
        record = self.saved_records[idx]
        self.review_idx = idx
        self.frame_index = record["frame_index"]
        self.current_frame = record["frame"].copy()
        self.current_rectified_frame = self.warp_frame(self.current_frame)
        dets = self.rebuild_detections_from_annotations(
            record["image_id"], self.current_frame.shape[1], self.current_frame.shape[0]
        )
        self.current_detections = [d for d in dets if d.source == "model"]
        self.manual_detections = [d for d in dets if d.source != "model"]
        self.selected_detection = None
        self.annotation_mode = True
        self.remove_mode = False
        self.update_display()

    def on_prev_saved(self):
        """Navega para frame salvo anterior."""
        if not self.saved_records:
            print("[INFO] Nenhum frame salvo para revisar.")
            return
        if self.review_idx is None:
            self.go_to_saved_frame(len(self.saved_records) - 1)
        else:
            self.go_to_saved_frame(max(0, self.review_idx - 1))

    def on_next_saved(self):
        """Navega para proximo frame salvo; se for o ultimo, sai da revisao."""
        if not self.saved_records:
            print("[INFO] Nenhum frame salvo para revisar.")
            return
        if self.review_idx is None:
            self.go_to_saved_frame(0)
            return
        next_idx = self.review_idx + 1
        if next_idx < len(self.saved_records):
            self.go_to_saved_frame(next_idx)
        else:
            self.exit_review_mode()

    def exit_review_mode(self):
        """Retorna ao fluxo ao vivo apos revisao."""
        if self.live_snapshot is not None:
            snap = self.live_snapshot
            self.frame_index = snap["frame_index"]
            self.current_frame = snap["frame"]
            self.current_rectified_frame = snap["rectified"]
            self.current_detections = snap["detections"]
            self.manual_detections = snap["manual_detections"]
            self.selected_detection = None
            self.live_snapshot = None
            self.review_idx = None
            self.update_display()
            return
        self.review_idx = None
        self.load_next_frame()

    def validate_selected_detection(self):
        """Limpa selecao se o indice estiver invalido."""
        if self.get_selected_detection() is None:
            self.selected_detection = None

    def get_selected_detection(self) -> Optional[Detection]:
        """Retorna a detection selecionada (se existir)."""
        if self.selected_detection is None:
            return None
        source, idx = self.selected_detection
        dets = self.manual_detections if source == "manual" else self.current_detections
        if 0 <= idx < len(dets):
            return dets[idx]
        return None

    @staticmethod
    def bbox_close(stored_bbox, bbox: np.ndarray, tol: float = 1.0) -> bool:
        """Compara bboxes com tolerancia."""
        if stored_bbox is None:
            return False
        arr = np.array(stored_bbox, dtype=np.float32)
        return np.allclose(arr, bbox, atol=tol)

    def find_detection_at(self, x: int, y: int) -> Optional[Tuple[str, int]]:
        """Retorna (source, idx) da bbox que contem o ponto."""
        candidates: List[Tuple[float, str, int]] = []
        for idx, det in enumerate(self.manual_detections):
            x1, y1, x2, y2 = det.original_bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = float(max(x2 - x1, 0.0) * max(y2 - y1, 0.0))
                candidates.append((area, "manual", idx))
        for idx, det in enumerate(self.current_detections):
            x1, y1, x2, y2 = det.original_bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = float(max(x2 - x1, 0.0) * max(y2 - y1, 0.0))
                candidates.append((area, "model", idx))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        _, source, idx = candidates[0]
        return (source, idx)

    def select_detection_at(self, x: int, y: int):
        """Seleciona a bbox clicada para editar o ID."""
        hit = self.find_detection_at(x, y)
        if hit is None:
            self.selected_detection = None
            self.update_display()
            return
        self.selected_detection = hit
        det = self.get_selected_detection()
        if det is not None:
            self.manual_id_var.set(str(det.track_id))
        self.update_display()

    def apply_manual_id_to_selection(self):
        """Aplica o ID digitado a bbox selecionada."""
        if self.selected_detection is None:
            print("[AVISO] Nenhuma caixa selecionada para editar.")
            return
        new_id = self.consume_manual_id_override()
        if new_id is None:
            return
        det = self.get_selected_detection()
        if det is None:
            print("[AVISO] Selecao invalida. Clique novamente na caixa.")
            self.selected_detection = None
            return
        old_id = det.track_id
        if old_id == new_id:
            print("[INFO] ID selecionado ja e o mesmo.")
            return
        det.track_id = new_id
        if det.source == "model" and det.internal_id is not None:
            self.tracker_id_map[det.internal_id] = new_id
        self.update_track_history_for_edit(old_id, new_id, det.original_bbox)
        self.update_recent_tracks_for_edit(old_id, new_id, det.original_bbox)
        if det.source == "manual":
            if old_id in self.manual_track_memory:
                del self.manual_track_memory[old_id]
            self.manual_track_memory[new_id] = {"bbox": det.original_bbox.copy()}
        print(f"[INFO] Track ID atualizado: {old_id} -> {new_id}.")
        self.update_display()

    def update_track_history_for_edit(self, old_id: int, new_id: int, bbox: np.ndarray):
        """Atualiza track_history ao trocar track_id."""
        old_entries = self.track_history.get(old_id, [])
        if old_entries:
            filtered = [
                entry
                for entry in old_entries
                if not (entry.get("frame") == self.frame_index and self.bbox_close(entry.get("bbox"), bbox))
            ]
            if filtered:
                self.track_history[old_id] = filtered
            else:
                self.track_history.pop(old_id, None)
        entries = self.track_history.setdefault(new_id, [])
        exists = any(
            entry.get("frame") == self.frame_index and self.bbox_close(entry.get("bbox"), bbox) for entry in entries
        )
        if not exists:
            entries.append({"frame": self.frame_index, "bbox": bbox.tolist()})

    def update_recent_tracks_for_edit(self, old_id: int, new_id: int, bbox: np.ndarray):
        """Atualiza recent_tracks no frame atual."""
        for frame_data in self.recent_tracks:
            if frame_data.get("frame") != self.frame_index:
                continue
            for tr in frame_data.get("tracks", []):
                if tr.get("id") == old_id and self.bbox_close(tr.get("bbox"), bbox):
                    tr["id"] = new_id

    def advance_after_review_accept(self):
        """Avanca apos aceitar revisao."""
        if self.review_idx is None:
            return
        next_idx = self.review_idx + 1
        if next_idx < len(self.saved_records):
            self.go_to_saved_frame(next_idx)
        else:
            self.exit_review_mode()

    def store_annotations(
        self, detections: List[Detection], existing_image_id: Optional[int] = None, existing_file_name: Optional[str] = None
    ) -> Tuple[int, str]:
        """Adiciona as detecoes aprovadas na estrutura COCO MOT e retorna (image_id, file_name)."""
        frame_to_save = (
            self.current_rectified_frame
            if SAVE_RECTIFIED_FRAMES and self.current_rectified_frame is not None
            else self.current_frame
        )
        if frame_to_save is None:
            raise RuntimeError("Frame atual ausente para salvamento.")

        height, width = frame_to_save.shape[:2]
        new_frame = existing_image_id is None
        image_id = self.image_id if new_frame else existing_image_id
        file_name = (
            f"{self.video_name}_frame_{self.frame_index:05d}.jpg" if new_frame or existing_file_name is None else existing_file_name
        )
        image_path = OUTPUT_IMAGES_DIR / file_name
        if not cv2.imwrite(str(image_path), frame_to_save):
            raise RuntimeError(f"Falha ao salvar frame em {image_path}")

        self.images = [img for img in self.images if img.get("id") != image_id]
        image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "video": str(self.video_path) if self.video_path else self.video_name,
        }
        self.images.append(image_info)

        self.annotations = [ann for ann in self.annotations if ann.get("image_id") != image_id]
        for det in detections:
            chosen_bbox = det.warp_bbox if SAVE_RECTIFIED_FRAMES and det.warp_bbox is not None else det.original_bbox
            chosen_bbox = chosen_bbox.astype(np.float32)
            chosen_bbox = clip_bbox(chosen_bbox[0], chosen_bbox[1], chosen_bbox[2], chosen_bbox[3], width, height)
            x1, y1, x2, y2 = chosen_bbox
            w = x2 - x1
            h = y2 - y1
            annotation = {
                "id": self.annotation_id,
                "image_id": image_id,
                "category_id": det.category_id,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "area": float(max(w, 0.0) * max(h, 0.0)),
                "iscrowd": 0,
                "segmentation": [],
                "score": float(det.confidence),
                "track_id": int(det.track_id),
                "source": det.source,
                "video": str(self.video_path) if self.video_path else self.video_name,
            }
            self.annotations.append(annotation)
            self.annotation_id += 1

        if new_frame:
            self.image_id += 1
            self.frames_saved_in_current_video += 1
        return image_id, file_name

    def write_annotations(self):
        """Grava o arquivo annotations.coco.json com as anotacoes atuais."""
        data = {
            "info": {
                "description": "Validacao manual de deteccoes com ROI e homografia",
                "version": "1.0",
                "video_sources": [str(v) for v in self.video_files],
                "frames_are_rectified": SAVE_RECTIFIED_FRAMES,
            },
            "licenses": [],
            "categories": self.categories,
            "images": self.images,
            "annotations": self.annotations,
        }
        with open(ANNOTATIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Anotacoes atualizadas em {ANNOTATIONS_PATH}")

    def finish_processing(self, message: str):
        """Libera recursos e encerra a interface."""
        if self.closed:
            return
        self.closed = True
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.window.unbind("<Return>")
        self.window.unbind("<space>")
        self.window.unbind("<Escape>")
        self.accept_button.config(state=tk.DISABLED)
        self.reject_button.config(state=tk.DISABLED)
        self.quit_button.config(state=tk.DISABLED)
        if self.images or self.annotations:
            self.write_annotations()
        if self.homographies:
            with open(HOMOGRAPHY_PATH, "w", encoding="utf-8") as f:
                json.dump(self.homographies, f, indent=4, ensure_ascii=False)
        self.info_var.set(message)
        try:
            self.window.after(500, self.window.destroy)
        except Exception:  # pylint: disable=broad-except
            try:
                self.window.destroy()
            except Exception:
                pass

    def run(self):
        """Inicia o loop principal da interface Tkinter."""
        self.window.mainloop()


def main():
    tool = None
    try:
        tool = AnnotationTool()
        tool.run()
        return 0
    except KeyboardInterrupt:
        if tool is not None:
            tool.finish_processing("Processo interrompido.")
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        if tool is not None:
            tool.finish_processing(f"Erro: {exc}")
        print(f"Erro: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
