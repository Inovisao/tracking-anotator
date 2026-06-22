#!/usr/bin/env python3
"""Repara um COCO Keypoints inconsistente (categories.keypoints vazio, ponto de
fechamento duplicado, bbox/num_keypoints desatualizados)."""

import argparse
import json
from pathlib import Path
from typing import List

DOCUMENT_CORNERS = ["top_left", "top_right", "bottom_right", "bottom_left"]
DOCUMENT_SKELETON = [[1, 2], [2, 3], [3, 4], [4, 1]]


def _triples(flat: List[float]) -> List[List[float]]:
    return [[float(flat[i]), float(flat[i + 1]), int(flat[i + 2])] for i in range(0, len(flat) - 2, 3)]


def _drop_duplicate_closing_point(triples: List[List[float]], tol: float = 3.0) -> List[List[float]]:
    visible = [kp for kp in triples if kp[2] > 0]
    if len(visible) >= 4:
        first, last = visible[0], visible[-1]
        if abs(first[0] - last[0]) <= tol and abs(first[1] - last[1]) <= tol:
            triples = [kp for kp in triples if kp is not last]
    return triples


def _sort_document_corners(triples: List[List[float]]) -> List[List[float]]:
    visible = [kp for kp in triples if kp[2] > 0]
    if len(visible) != 4:
        return triples
    top = sorted(visible, key=lambda kp: kp[1])[:2]
    bottom = sorted(visible, key=lambda kp: kp[1])[2:]
    top_left, top_right = sorted(top, key=lambda kp: kp[0])
    bottom_left, bottom_right = sorted(bottom, key=lambda kp: kp[0])
    return [top_left, top_right, bottom_right, bottom_left]


def _bbox(triples: List[List[float]]):
    pts = [(kp[0], kp[1]) for kp in triples if kp[2] > 0]
    if not pts:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def fix_payload(payload: dict, *, sort_corners: bool = True) -> dict:
    counts_by_cat = {}
    # Pass 1: repair annotations.
    for ann in payload.get("annotations", []):
        triples = _drop_duplicate_closing_point(_triples(ann.get("keypoints", [])))
        cid = int(ann.get("category_id", -1))
        if sort_corners:
            triples = _sort_document_corners(triples)
        ann["keypoints"] = [c for kp in triples for c in (kp[0], kp[1], int(kp[2]))]
        ann["num_keypoints"] = sum(1 for kp in triples if kp[2] > 0)
        x, y, w, h = _bbox(triples)
        ann["bbox"] = [x, y, w, h]
        ann["area"] = float(w * h)
        ann["annotation_type"] = "keypoint"
        counts_by_cat.setdefault(cid, set()).add(len(triples))

    # Pass 2: fill empty categories.keypoints to match annotations.
    for cat in payload.get("categories", []):
        cat.setdefault("skeleton", [])
        if cat.get("keypoints"):
            continue
        counts = counts_by_cat.get(int(cat.get("id", -1)), set())
        n = max(counts) if counts else 0
        if n == 4:
            cat["keypoints"] = list(DOCUMENT_CORNERS)
            if not cat["skeleton"]:
                cat["skeleton"] = [list(link) for link in DOCUMENT_SKELETON]
        else:
            cat["keypoints"] = [f"point_{i + 1}" for i in range(n)]
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repara um COCO Keypoints para treino de pose.")
    parser.add_argument("input", type=Path, help="annotations_keypoints.coco.json de entrada")
    parser.add_argument("output", type=Path, nargs="?", help="Saida (padrao: sobrescreve a entrada)")
    parser.add_argument(
        "--no-sort-corners",
        action="store_true",
        help="Nao reordena pontos. Por padrao, instancias de 4 pontos viram TL, TR, BR, BL.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {args.input}")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    fixed = fix_payload(payload, sort_corners=not args.no_sort_corners)
    output = args.output or args.input
    output.write_text(json.dumps(fixed, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] COCO keypoints reparado salvo em {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
