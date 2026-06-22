from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, List, Optional


def keypoint_payload_errors(payload: dict) -> List[str]:
    """Inconsistencies that would make a pose dataset invalid for training."""
    counts_by_cat = {}
    for ann in payload.get("annotations", []):
        cid = int(ann.get("category_id", -1))
        counts_by_cat.setdefault(cid, set()).add(len(ann.get("keypoints", [])) // 3)
    name_by_id = {int(c.get("id", 0)): c.get("name", "?") for c in payload.get("categories", [])}
    errors = []
    for cat in payload.get("categories", []):
        cid = int(cat.get("id", 0))
        counts = counts_by_cat.get(cid)
        if not counts:
            continue
        if not cat.get("keypoints"):
            errors.append(f"Classe '{name_by_id.get(cid)}' sem keypoints declarados.")
        if len(counts) > 1:
            errors.append(
                f"Classe '{name_by_id.get(cid)}' tem instancias com nº de keypoints diferentes: {sorted(counts)}."
            )
    return errors


def export_coco_keypoints(
    payload: dict,
    output_path: Path,
    source_images_dir: Path,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    output_path = Path(output_path)
    images_dir = output_path.parent / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    images = payload.get("images", [])
    total = len(images)
    for done, image in enumerate(images, start=1):
        file_name = str(image.get("file_name", "")).strip()
        if file_name:
            src = source_images_dir / file_name
            if src.exists():
                dst = images_dir / file_name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        if on_progress:
            on_progress(done, total)
    output_path.write_text(json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8")
    return {"images": len(images)}
