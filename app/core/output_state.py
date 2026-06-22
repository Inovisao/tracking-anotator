"""Output dataset state management.

Each annotation run writes to an isolated directory under ``outputs/``. Existing
states can be resumed or used as a template for classes/configuration.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from app.config import OUTPUT_DATASET_PREFIX, SAVED_STATES_SUBDIR
from app.core.session import AnnotationTaskMode, normalize_class_names

ANNOTATION_FILE_NAMES = (
    "annotations.coco.json",
    "annotations_obb.coco.json",
    "annotations_keypoints.coco.json",
    "__annotations.coco.json",
)
STATE_PATTERN = re.compile(rf"^{re.escape(OUTPUT_DATASET_PREFIX)}(?P<index>\d+)_(?P<stamp>\d{{8}}_\d{{6}})$")
NEW_STATE_PATTERN = re.compile(r"^.+_(?P<day>\d{2})\.(?P<month>\d{2})\.(?P<hour>\d{2})-(?P<minute>\d{2})(?:_\d{3})?$")
TASK_DIR_NAMES = {
    AnnotationTaskMode.DETECTION: "detecção",
    AnnotationTaskMode.TRACKING: "tracking",
    AnnotationTaskMode.OBB: "obb",
    AnnotationTaskMode.CLASSIFICATION: "classificação",
}


@dataclass(frozen=True)
class OutputState:
    """Summary of a persisted annotation state."""

    path: Path
    annotations_path: Path
    index: int
    created_at: Optional[datetime]
    modified_at: Optional[datetime]
    task_mode: Optional[AnnotationTaskMode]
    class_names: tuple[str, ...]
    image_count: int
    annotation_count: int
    source_paths: tuple[Path, ...]

    @property
    def label(self) -> str:
        mode = self.task_mode.label if self.task_mode is not None else "modo desconhecido"
        stamp_source = self.modified_at or self.created_at
        stamp = stamp_source.strftime("%d/%m/%Y %H:%M:%S") if stamp_source else self.path.name
        return (
            f"{self.path.name} | {stamp} | {mode} | "
            f"{len(self.class_names)} classes | {self.image_count} imagens | {self.annotation_count} anotacoes"
        )


@dataclass(frozen=True)
class LoadedAnnotationState:
    """Data read from a COCO annotations file."""

    annotations_path: Path
    output_dir: Path
    task_mode: Optional[AnnotationTaskMode]
    class_names: tuple[str, ...]
    categories: tuple[dict, ...]
    image_count: int
    annotation_count: int
    source_paths: tuple[Path, ...]


def annotations_path_for(output_dir: Path) -> Path:
    """Canonical annotations path for new output states."""
    return Path(output_dir) / SAVED_STATES_SUBDIR / ANNOTATION_FILE_NAMES[0]


def output_dir_from_annotations_path(annotations_path: Path) -> Path:
    """Return the project root given an annotations file path.

    Handles both new-style (inside saved_data_states/) and legacy (root) layouts.
    """
    p = Path(annotations_path)
    if p.parent.name == SAVED_STATES_SUBDIR:
        return p.parent.parent
    return p.parent


def find_annotations_path(path: Path) -> Optional[Path]:
    """Return a supported annotations file from a file or output directory.

    Searches saved_data_states/ first (new layout), then the root (legacy compat).
    """
    path = Path(path).expanduser()
    if path.is_file() and path.name in ANNOTATION_FILE_NAMES:
        return path
    if path.is_dir():
        # new layout: saved_data_states/ subfolder
        for name in ANNOTATION_FILE_NAMES:
            candidate = path / SAVED_STATES_SUBDIR / name
            if candidate.exists():
                return candidate
        # legacy layout: annotations at root
        for name in ANNOTATION_FILE_NAMES:
            candidate = path / name
            if candidate.exists():
                return candidate
    return None


def list_output_states(outputs_dir: Path) -> list[OutputState]:
    """List valid output states inside outputs_dir, ordered from oldest to newest."""

    outputs_dir = Path(outputs_dir).expanduser()
    if not outputs_dir.exists():
        return []
    states = []
    for child in outputs_dir.iterdir():
        if not child.is_dir():
            continue
        annotations_path = find_annotations_path(child)
        if annotations_path is None:
            continue
        state = _build_state(child, annotations_path)
        if state is not None:
            states.append(state)
    return sorted(states, key=lambda s: (s.modified_at or s.created_at or datetime.min, s.path.name))


def latest_output_state(outputs_dir: Path) -> Optional[OutputState]:
    """Return the newest available output state inside outputs_dir."""

    states = list_output_states(outputs_dir)
    return states[-1] if states else None


def list_output_states_for_sources(
    sources: Iterable[Path],
    outputs_dir: Path,
) -> list[OutputState]:
    """List output states in outputs_dir associated with the given source paths."""

    project_sources = _normalize_source_paths(sources)
    if not project_sources:
        return []
    return [s for s in list_output_states(outputs_dir) if _state_matches_sources(s, project_sources)]


def latest_output_state_for_sources(
    sources: Iterable[Path],
    outputs_dir: Path,
) -> Optional[OutputState]:
    """Return the newest output state in outputs_dir for the given source paths."""

    states = list_output_states_for_sources(sources, outputs_dir)
    return states[-1] if states else None


def create_new_output_dir(
    parent_dir: Path,
    session_name: str,
    *,
    create_images_dir: bool = True,
) -> Path:
    """Create a uniquely named project directory under parent_dir.

    Creates saved_data_states/ (and optionally images/) inside it.
    Raises ValueError if session_name is empty.
    """
    session_name = str(session_name).strip()
    if not session_name:
        raise ValueError("Session name cannot be empty.")
    parent_dir = Path(parent_dir).expanduser()
    parent_dir.mkdir(parents=True, exist_ok=True)
    candidate = parent_dir / session_name
    suffix = 1
    while candidate.exists():
        candidate = parent_dir / f"{session_name}_{suffix:03d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    (candidate / SAVED_STATES_SUBDIR).mkdir()
    if create_images_dir:
        (candidate / "images").mkdir()
    return candidate


def load_annotation_state(path: Path) -> LoadedAnnotationState:
    """Load categories/configuration from a supported annotations file or state directory."""

    annotations_path = find_annotations_path(path)
    if annotations_path is None:
        raise FileNotFoundError(f"Arquivo de anotacoes nao encontrado em: {path}")
    try:
        payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError(f"Nao foi possivel ler {annotations_path}: {exc}") from exc

    categories = tuple(dict(cat) for cat in payload.get("categories", []) if str(cat.get("name", "")).strip())
    class_names = normalize_class_names(str(cat.get("name", "")) for cat in categories)
    info = payload.get("info", {}) if isinstance(payload.get("info"), dict) else {}
    mode = _parse_mode(info.get("task_mode"))
    source_paths = _extract_source_paths(info, payload.get("annotation_state", {}))
    return LoadedAnnotationState(
        annotations_path=annotations_path,
        output_dir=output_dir_from_annotations_path(annotations_path),
        task_mode=mode,
        class_names=class_names,
        categories=categories,
        image_count=len(payload.get("images", []) or []),
        annotation_count=len(payload.get("annotations", []) or []),
        source_paths=source_paths,
    )


def _build_state(path: Path, annotations_path: Path) -> Optional[OutputState]:
    try:
        loaded = load_annotation_state(annotations_path)
    except Exception:
        return None
    index, created_at = _parse_state_name(path.name)
    modified_at = _modified_at(annotations_path)
    return OutputState(
        path=path,
        annotations_path=annotations_path,
        index=index,
        created_at=created_at,
        modified_at=modified_at,
        task_mode=loaded.task_mode,
        class_names=loaded.class_names,
        image_count=loaded.image_count,
        annotation_count=loaded.annotation_count,
        source_paths=loaded.source_paths,
    )


def _parse_mode(value) -> Optional[AnnotationTaskMode]:
    if not value:
        return None
    try:
        return AnnotationTaskMode(str(value))
    except ValueError:
        return None


def _parse_state_name(name: str) -> tuple[int, Optional[datetime]]:
    match = STATE_PATTERN.match(name)
    if match:
        try:
            created_at = datetime.strptime(match.group("stamp"), "%Y%m%d_%H%M%S")
        except ValueError:
            created_at = None
        return int(match.group("index")), created_at
    match = NEW_STATE_PATTERN.match(name)
    if match:
        try:
            created_at = datetime(
                datetime.now().year,
                int(match.group("month")),
                int(match.group("day")),
                int(match.group("hour")),
                int(match.group("minute")),
            )
        except ValueError:
            created_at = None
        return 0, created_at
    return 0, None


def _task_dir_name(task_mode: AnnotationTaskMode | str) -> str:
    if isinstance(task_mode, AnnotationTaskMode):
        return TASK_DIR_NAMES.get(task_mode, task_mode.value)
    try:
        mode = AnnotationTaskMode(str(task_mode))
    except ValueError:
        return str(task_mode).strip().lower() or TASK_DIR_NAMES[AnnotationTaskMode.DETECTION]
    return TASK_DIR_NAMES.get(mode, mode.value)


def _modified_at(path: Path) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime)
    except OSError:
        return None


def _extract_source_paths(info: dict, annotation_state) -> tuple[Path, ...]:
    raw_sources = []
    data_root = info.get("data_root")
    if data_root:
        raw_sources.append(data_root)
    raw_sources.extend(info.get("video_sources") or [])
    if isinstance(annotation_state, dict) and annotation_state.get("last_active_source"):
        raw_sources.append(annotation_state["last_active_source"])
    return _normalize_source_paths(raw_sources)


def _normalize_source_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    normalized = []
    seen = set()
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path).expanduser()
        try:
            path = path.resolve()
        except OSError:
            path = path.absolute()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(path)
    return tuple(normalized)


def _state_matches_sources(state: OutputState, project_sources: tuple[Path, ...]) -> bool:
    state_sources = set(state.source_paths)
    if not state_sources:
        return False
    for source in project_sources:
        if source in state_sources:
            return True
        if any(_paths_overlap(source, state_source) for state_source in state_sources):
            return True
    return False


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False
