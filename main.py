"""Ferramenta interativa para validar detecoes YOLO e gerar anotacoes COCO."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO

# ====== CONFIGURACOES ======
VIDEO_PATH = Path(
    "C:/Users/astma/Documents/inovisao/hungria/tracker/videos/POZ3_Thomas_green/1_2017-10-25_17-24-28_POZ3_Thomas_g1.mp4"
)
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "best.pt"
ANNOTATIONS_PATH = Path(__file__).resolve().parent / "annotations.coco.json"
CONF_THRESHOLD = 0.5
TARGET_CLASS = "car"


@dataclass
class Detection:
    bbox_xyxy: np.ndarray
    confidence: float
    category_id: int


class AnnotationTool:
    """Controla a interface de validacao e a geracao do arquivo COCO."""

    def __init__(self):
        if not VIDEO_PATH.exists():
            raise FileNotFoundError(f"Video nao encontrado: {VIDEO_PATH}")
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"Pesos nao encontrados: {WEIGHTS_PATH}")

        self.model = YOLO(str(WEIGHTS_PATH))
        self.cap = cv2.VideoCapture(str(VIDEO_PATH))
        if not self.cap.isOpened():
            raise RuntimeError(f"Falha ao abrir video: {VIDEO_PATH}")

        self.video_name = VIDEO_PATH.stem
        self.frame_index = 0
        self.image_id = 1
        self.annotation_id = 1

        self.current_frame = None
        self.current_detections: List[Detection] = []
        self.tk_image = None

        self.images = []
        self.annotations = []
        self.categories = [{"id": 1, "name": TARGET_CLASS}]

        self.window = tk.Tk()
        self.window.title("Validador de deteccoes")
        self.window.protocol("WM_DELETE_WINDOW", self.on_quit)

        self.info_var = tk.StringVar(value="Carregando...")
        self.info_label = tk.Label(self.window, textvariable=self.info_var, font=("Arial", 12))
        self.info_label.pack(pady=10)

        self.image_label = tk.Label(self.window)
        self.image_label.pack()

        buttons_frame = tk.Frame(self.window)
        buttons_frame.pack(pady=10)

        self.accept_button = tk.Button(buttons_frame, text="Validar (Enter)", command=self.on_accept, width=18)
        self.accept_button.grid(row=0, column=0, padx=5)

        self.reject_button = tk.Button(buttons_frame, text="Rejeitar (Espaco)", command=self.on_reject, width=18)
        self.reject_button.grid(row=0, column=1, padx=5)

        self.quit_button = tk.Button(buttons_frame, text="Sair (Esc)", command=self.on_quit, width=18)
        self.quit_button.grid(row=0, column=2, padx=5)

        self.window.bind("<Return>", lambda event: self.on_accept())
        self.window.bind("<space>", lambda event: self.on_reject())
        self.window.bind("<Escape>", lambda event: self.on_quit())

        self.load_next_frame()

    def load_next_frame(self):
        """Carrega o proximo frame do video e atualiza a tela."""
        ret, frame = self.cap.read()
        if not ret:
            self.finish_processing("Video finalizado.")
            return

        self.frame_index += 1
        self.current_frame = frame
        self.current_detections = self.run_model(frame)
        annotated_frame = self.draw_detections(frame.copy(), self.current_detections)
        self.show_frame(annotated_frame)

    def run_model(self, frame) -> List[Detection]:
        """Executa o modelo YOLO e filtra detecoes da classe alvo."""
        height, width = frame.shape[:2]
        detections: List[Detection] = []
        results = self.model(frame, verbose=False)
        if not results:
            return detections

        result = results[0]
        names = result.names

        for box in result.boxes:
            conf = float(box.conf)
            cls_id = int(box.cls)
            label = names.get(cls_id, str(cls_id))
            if conf < CONF_THRESHOLD or label != TARGET_CLASS:
                continue
            xyxy = box.xyxy.cpu().numpy()[0]
            xyxy[0::2] = np.clip(xyxy[0::2], 0, width - 1)
            xyxy[1::2] = np.clip(xyxy[1::2], 0, height - 1)
            detections.append(Detection(bbox_xyxy=xyxy, confidence=conf, category_id=1))

        return detections

    def draw_detections(self, frame, detections: List[Detection]):
        """Desenha as caixas detectadas no frame visivel."""
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{TARGET_CLASS} {det.confidence * 100:.1f}%"
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

    def show_frame(self, frame):
        """Renderiza o frame no widget Tkinter."""
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        self.tk_image = ImageTk.PhotoImage(image=pil_image)
        self.image_label.configure(image=self.tk_image)
        self.info_var.set(
            f"Frame {self.frame_index} | Deteccoes validas (> {CONF_THRESHOLD*100:.0f}%): {len(self.current_detections)} | "
            f"Resolucao: {width}x{height}"
        )

    def on_accept(self):
        """Persistir anotacoes quando o usuario aprova o frame."""
        if self.current_frame is None:
            return
        if self.current_detections:
            self.store_annotations()
            self.write_annotations()
        self.load_next_frame()

    def on_reject(self):
        """Ignora o frame atual e avanca para o proximo."""
        self.load_next_frame()

    def on_quit(self):
        """Encerra o processo de anotacao."""
        self.finish_processing("Processo encerrado pelo usuario.")

    def store_annotations(self):
        """Adiciona as detecoes aprovadas na estrutura COCO."""
        height, width = self.current_frame.shape[:2]
        image_info = {
            "id": self.image_id,
            "file_name": f"{self.video_name}_frame_{self.frame_index:05d}.jpg",
            "width": width,
            "height": height,
        }
        self.images.append(image_info)

        for det in self.current_detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            w = x2 - x1
            h = y2 - y1
            annotation = {
                "id": self.annotation_id,
                "image_id": self.image_id,
                "category_id": det.category_id,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
                "segmentation": [],
                "score": float(det.confidence),
            }
            self.annotations.append(annotation)
            self.annotation_id += 1

        self.image_id += 1

    def write_annotations(self):
        """Grava o arquivo annotations.coco.json com as anotacoes atuais."""
        data = {
            "info": {
                "description": "Validacao manual de deteccoes",
                "version": "1.0",
                "video_source": str(VIDEO_PATH),
            },
            "licenses": [],
            "categories": self.categories,
            "images": self.images,
            "annotations": self.annotations,
        }
        with open(ANNOTATIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        self.info_var.set(self.info_var.get() + " | Anotacoes salvas.")

    def finish_processing(self, message: str):
        """Libera recursos e encerra a interface."""
        self.cap.release()
        self.window.unbind("<Return>")
        self.window.unbind("<space>")
        self.window.unbind("<Escape>")
        self.accept_button.config(state=tk.DISABLED)
        self.reject_button.config(state=tk.DISABLED)
        self.quit_button.config(state=tk.DISABLED)
        if self.current_frame is not None and self.current_detections:
            self.write_annotations()
        self.info_var.set(message)
        self.window.after(1500, self.window.destroy)

    def run(self):
        """Inicia o loop principal da interface Tkinter."""
        self.window.mainloop()


def main():
    try:
        tool = AnnotationTool()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    tool.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
