from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2

from app.annotation.core.augmentation.augmentation_types import AugmentationPreset
from app.annotation.core.export.split_service import assign_splits
from app.annotation.core.export.yolo_label_service import build_zero_based_category_mapping
from app.annotation_keypoint.core.augmentation.pose_augmentation import augment_pose

PoseInstance = Tuple[int, List[List[float]]]


def _image_lookup(payload: dict) -> Dict[int, dict]:
    return {int(image.get("id")): image for image in payload.get("images", []) if image.get("id") is not None}


def _norm(value: float, size: int) -> float:
    return max(0.0, min(1.0, float(value) / max(size, 1)))


def _max_keypoints(payload: dict) -> int:
    counts = [len(cat.get("keypoints", [])) for cat in payload.get("categories", [])]
    by_ann = [len(ann.get("keypoints", [])) // 3 for ann in payload.get("annotations", [])]
    return max(counts + by_ann + [0])


def _instances_for_image(annotations: List[dict], class_mapping: Dict[int, int]) -> List[PoseInstance]:
    instances: List[PoseInstance] = []
    for ann in annotations:
        cid = int(ann.get("category_id", -1))
        if cid not in class_mapping:
            continue
        flat = ann.get("keypoints") or []
        kps = [[float(flat[i]), float(flat[i + 1]), int(flat[i + 2])] for i in range(0, len(flat) - 2, 3)]
        instances.append((class_mapping[cid], kps))
    return instances


def _pose_line(class_index: int, kps_abs: List[List[float]], img_w: int, img_h: int, n_kpts: int) -> str:
    visible = [(kp[0], kp[1]) for kp in kps_abs if kp[2] > 0]
    if visible:
        xs = [p[0] for p in visible]
        ys = [p[1] for p in visible]
        x, y, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    else:
        x = y = w = h = 0.0
    values = [
        str(class_index),
        f"{_norm(x + w / 2.0, img_w):.6f}", f"{_norm(y + h / 2.0, img_h):.6f}",
        f"{_norm(w, img_w):.6f}", f"{_norm(h, img_h):.6f}",
    ]
    for idx in range(n_kpts):
        kp = kps_abs[idx] if idx < len(kps_abs) else [0, 0, 0]
        vis = int(kp[2])
        if vis <= 0:
            values.extend(["0.000000", "0.000000", "0"])
        else:
            values.extend([f"{_norm(kp[0], img_w):.6f}", f"{_norm(kp[1], img_h):.6f}", str(vis)])
    return " ".join(values)


def _format_data_yaml(dataset_root: Path, names: Dict[int, str], n_kpts: int, splits: List[str]) -> str:
    lines = [f"path: {dataset_root}"]
    for split in splits:
        lines.append(f"{split}: images/{split}")
    lines += ["", f"kpt_shape: [{n_kpts}, 3]", "", "names:"]
    for class_id, name in names.items():
        lines.append(f"  {class_id}: {name}")
    return "\n".join(lines) + "\n"


def _write_pair(dataset_root: Path, split: str, file_name: str, source: Path, lines: List[str]) -> None:
    image_path = dataset_root / "images" / split / file_name
    label_path = dataset_root / "labels" / split / Path(file_name).with_suffix(".txt")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, image_path)
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_augmented(dataset_root: Path, split: str, file_name: str, source: Path,
                     instances: List[PoseInstance], n_kpts: int, preset: AugmentationPreset) -> int:
    image = cv2.imread(str(source))
    if image is None:
        return 0
    written = 0
    for idx, (aug_image, aug_instances) in enumerate(augment_pose(image, instances, preset)):
        ah, aw = aug_image.shape[:2]
        stem = Path(file_name).stem
        suffix = Path(file_name).suffix
        aug_name = f"{stem}_aug{idx + 1}{suffix}"
        image_path = dataset_root / "images" / split / aug_name
        label_path = dataset_root / "labels" / split / f"{stem}_aug{idx + 1}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), aug_image):
            continue
        lines = [_pose_line(cls, kps, aw, ah, n_kpts) for cls, kps in aug_instances]
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1
    return written


def export_yolo_pose_dataset(
    payload: dict,
    output_dir: Path,
    source_images_dir: Path,
    *,
    split_ratios: Optional[Tuple[float, float, float]] = None,
    augmentation_preset: Optional[AugmentationPreset] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    class_mapping, names = build_zero_based_category_mapping(payload.get("categories", []))
    if not names:
        raise ValueError("Nenhuma categoria valida para exportar YOLO Pose.")
    images = payload.get("images", [])
    annotations_by_image: Dict[int, List[dict]] = {}
    for ann in payload.get("annotations", []):
        annotations_by_image.setdefault(int(ann.get("image_id", -1)), []).append(ann)

    n_kpts = _max_keypoints(payload)
    splits = ["train", "val", "test"] if split_ratios else ["train"]
    assignments = assign_splits(images, split_ratios) if split_ratios else {}
    for split in splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    copied = 0
    labels = 0
    total = len(images)
    for done, image in enumerate(sorted(images, key=lambda im: str(im.get("file_name", ""))), start=1):
        file_name = str(image.get("file_name", "")).strip()
        if not file_name:
            continue
        source = source_images_dir / file_name
        if not source.exists():
            continue
        split = assignments.get(int(image.get("id", -1)), "train") if split_ratios else "train"
        instances = _instances_for_image(annotations_by_image.get(int(image["id"]), []), class_mapping)
        img_w, img_h = int(image.get("width", 1)), int(image.get("height", 1))
        lines = [_pose_line(cls, kps, img_w, img_h, n_kpts) for cls, kps in instances]
        _write_pair(output_dir, split, file_name, source, lines)
        copied += 1
        labels += len(lines)
        if split == "train" and augmentation_preset is not None and augmentation_preset.enabled:
            aug = _write_augmented(output_dir, split, file_name, source, instances, n_kpts, augmentation_preset)
            copied += aug
        if on_progress:
            on_progress(done, total)

    (output_dir / "data.yaml").write_text(
        _format_data_yaml(output_dir, names, n_kpts, splits), encoding="utf-8"
    )
    return {"images": copied, "labels": labels, "names": names, "kpt_shape": n_kpts}
