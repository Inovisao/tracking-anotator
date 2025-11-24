"""Teste rapido: processa 10 frames do primeiro video encontrado em `videos/` e exibe deteccoes."""

import sys
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO

VIDEOS_ROOT = Path(__file__).resolve().parent / "videos"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
WEIGHTS_PATH = Path(__file__).resolve().parent / "yolo11l.pt"
CONF_THRESHOLD = 0.40
TARGET_CLASS = "car"


def find_first_video() -> Optional[Path]:
    """Retorna o primeiro video encontrado (ordenado) dentro de `videos/`."""
    if not VIDEOS_ROOT.exists():
        return None
    videos = sorted(
        [p for p in VIDEOS_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    )
    return videos[0] if videos else None


def main() -> int:
    video_path = find_first_video()
    if video_path is None:
        print(f"Nenhum video encontrado em {VIDEOS_ROOT}")
        return 1

    if not WEIGHTS_PATH.exists():
        print(f"Pesos nao encontrados: {WEIGHTS_PATH}")
        return 1

    print(f"[INFO] Usando video: {video_path}")
    print(f"[INFO] Usando pesos: {WEIGHTS_PATH}")

    model = YOLO(str(WEIGHTS_PATH))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Falha ao abrir video: {video_path}")
        return 1

    processed = 0
    while processed < 10:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Fim do video antes de atingir 10 frames.")
            break

        results = model.track(frame, persist=True, verbose=False)
        det_count = 0
        track_ids = []
        annotated = frame.copy()
        if results and results[0].boxes is not None:
            names = results[0].names
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
            for box in results[0].boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                label = names.get(cls_id, str(cls_id))
                if label == TARGET_CLASS and conf >= CONF_THRESHOLD:
                    det_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy.cpu().tolist()[0])
                    tid = int(box.id.item()) if box.id is not None else -1
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        f"{label} {conf*100:.1f}% ID:{tid}",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

        processed += 1
        print(f"Frame {processed:02d}: {det_count} detecoes '{TARGET_CLASS}' | track_ids: {track_ids}")
        cv2.imshow("YOLO main_test", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Interrompido pelo usuario (q).")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Processamento concluido ({processed} frames).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
