"""Dataset operations for manual image classification."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from app.config import IMAGE_EXTENSIONS, OUTPUT_DATASET_PREFIX
from app.core.session import normalize_class_names

STATE_FILE_NAME = "classification_state.json"
STATE_PATTERN = re.compile(rf"^{re.escape(OUTPUT_DATASET_PREFIX)}(?P<index>\d+)_(?P<stamp>\d{{8}}_\d{{6}})")
NEW_STATE_PATTERN = re.compile(r"^.+_(?P<day>\d{2})\.(?P<month>\d{2})\.(?P<hour>\d{2}):(?P<minute>\d{2})(?:_\d{3})?$")


@dataclass(frozen=True)
class ClassificationRecord:
    source_path: Path
    destination_path: Path
    class_name: str
    classified_at: str
    operation: str = "copy"


@dataclass(frozen=True)
class ClassificationState:
    classes: tuple[str, ...]
    class_directories: dict[str, str]
    source_root: Path
    records: tuple[ClassificationRecord, ...]

    @property
    def classified_sources(self) -> set[Path]:
        return {record.source_path for record in self.records}


@dataclass(frozen=True)
class ClassificationOutputState:
    path: Path
    state_path: Path
    index: int
    created_at: Optional[datetime]
    modified_at: Optional[datetime]
    class_names: tuple[str, ...]
    image_count: int
    source_root: Path

    @property
    def label(self) -> str:
        stamp_source = self.modified_at or self.created_at
        stamp = stamp_source.strftime("%d/%m/%Y %H:%M:%S") if stamp_source else self.path.name
        return f"{self.path.name} | {stamp} | classificacao | {len(self.class_names)} classes | {self.image_count} imagens"


def sanitize_class_dir_name(name: str) -> str:
    """Return a filesystem-safe directory name for a user-facing class."""

    cleaned = str(name).strip().lower()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9._-]+", "_", cleaned)
    cleaned = cleaned.strip("._-")
    return cleaned or "classe"


def class_directories_for(classes: Iterable[str]) -> dict[str, str]:
    """Map class names to unique safe directory names."""

    directories: dict[str, str] = {}
    used: set[str] = set()
    for class_name in normalize_class_names(classes):
        base = sanitize_class_dir_name(class_name)
        candidate = base
        suffix = 1
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        directories[class_name] = candidate
        used.add(candidate)
    return directories


def prepare_dataset(output_dir: Path, classes: Iterable[str]) -> dict[str, str]:
    """Create one subdirectory per class and return class-to-folder mapping."""

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    directories = class_directories_for(classes)
    for dirname in directories.values():
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)
    return directories


def add_class_directory(output_dir: Path, class_name: str, class_directories: dict[str, str]) -> str:
    """Create a unique class subfolder and update ``class_directories``."""

    clean_name = str(class_name).strip()
    if not clean_name:
        raise ValueError("Nome de classe vazio.")
    if clean_name in class_directories:
        return class_directories[clean_name]

    used = set(class_directories.values())
    base = sanitize_class_dir_name(clean_name)
    candidate = base
    suffix = 1
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1

    (Path(output_dir).expanduser() / candidate).mkdir(parents=True, exist_ok=True)
    class_directories[clean_name] = candidate
    return candidate


def class_directory_path(output_dir: Path, class_name: str, class_directories: dict[str, str]) -> Path | None:
    """Return the filesystem path for a class directory."""

    dirname = class_directories.get(class_name)
    if not dirname:
        return None
    return Path(output_dir).expanduser() / dirname


def class_directory_has_files(output_dir: Path, class_name: str, class_directories: dict[str, str]) -> bool:
    """Return True when the class folder contains any file."""

    path = class_directory_path(output_dir, class_name, class_directories)
    if path is None or not path.exists():
        return False
    return any(candidate.is_file() for candidate in path.rglob("*"))


def remove_class_directory(
    output_dir: Path,
    class_name: str,
    class_directories: dict[str, str],
    *,
    delete_files: bool = False,
    archive_files: bool = False,
) -> Path | None:
    """Remove class mapping and optionally delete its folder from disk."""

    path = class_directory_path(output_dir, class_name, class_directories)
    class_directories.pop(class_name, None)
    if delete_files and path is not None and path.exists():
        shutil.rmtree(path)
    elif archive_files and path is not None and path.exists():
        archive_root = unique_destination_path(Path(output_dir).expanduser() / "_removed" / path.name)
        archive_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(archive_root))
    elif path is not None and path.exists():
        try:
            path.rmdir()
        except OSError:
            pass
    return path


def discover_images(data_root: Path) -> list[Path]:
    """Discover images from a folder, single image, or text list."""

    data_root = Path(data_root).expanduser()
    if data_root.is_file() and data_root.suffix.lower() in IMAGE_EXTENSIONS:
        return [data_root]
    if data_root.is_file() and data_root.suffix.lower() in {".txt", ".lst"}:
        return _read_image_list(data_root)
    if data_root.is_dir():
        return sorted(path for path in data_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    return []


def find_state_path(path: Path) -> Path | None:
    """Return a classification state path from a file or output directory."""

    path = Path(path).expanduser()
    if path.is_file() and path.name == STATE_FILE_NAME:
        return path
    if path.is_dir():
        candidate = path / STATE_FILE_NAME
        if candidate.exists():
            return candidate
    return None


def load_state(state_path: Path) -> ClassificationState | None:
    """Load a classification state file if it exists."""

    state_path = Path(state_path).expanduser()
    if not state_path.exists():
        return None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    task_mode = str(payload.get("task_mode", "")).strip()
    if task_mode and task_mode != "classification":
        raise ValueError(f"Estado nao e de classificacao: {task_mode}")
    if "categories" in payload or "annotations" in payload:
        raise ValueError("Estado COCO nao pode ser usado como estado de classificacao.")
    records = tuple(
        ClassificationRecord(
            source_path=Path(item["source_path"]).expanduser(),
            destination_path=Path(item["destination_path"]).expanduser(),
            class_name=str(item["class_name"]),
            classified_at=str(item.get("classified_at", "")),
            operation=str(item.get("operation", "copy")),
        )
        for item in payload.get("records", [])
        if item.get("source_path") and item.get("destination_path") and item.get("class_name")
    )
    return ClassificationState(
        classes=normalize_class_names(payload.get("classes", [])),
        class_directories=dict(payload.get("class_directories", {})),
        source_root=Path(payload.get("source_root", "")).expanduser(),
        records=records,
    )


def load_required_state(path: Path) -> ClassificationState:
    """Load a classification state from a file or output directory."""

    state_path = find_state_path(path)
    if state_path is None:
        raise FileNotFoundError(f"Arquivo {STATE_FILE_NAME} nao encontrado em: {path}")
    state = load_state(state_path)
    if state is None:
        raise FileNotFoundError(f"Arquivo {STATE_FILE_NAME} nao encontrado em: {path}")
    return state


def list_output_states(outputs_dir: Path) -> list[ClassificationOutputState]:
    """List classification output states ordered from oldest to newest."""

    outputs_dir = Path(outputs_dir).expanduser()
    if not outputs_dir.exists():
        return []
    states = []
    for child in outputs_dir.iterdir():
        if not child.is_dir():
            continue
        state_path = find_state_path(child)
        if state_path is None:
            continue
        try:
            state = load_state(state_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if state is None:
            continue
        index, created_at = _parse_state_name(child.name)
        states.append(
            ClassificationOutputState(
                path=child,
                state_path=state_path,
                index=index,
                created_at=created_at,
                modified_at=_modified_at(state_path),
                class_names=state.classes,
                image_count=len(state.records),
                source_root=state.source_root,
            )
        )
    return sorted(states, key=lambda item: (item.modified_at or item.created_at or datetime.min, item.path.name))


def list_output_states_for_sources(
    sources: Iterable[Path],
    outputs_dir: Path,
) -> list[ClassificationOutputState]:
    """List classification states associated with selected source paths."""

    project_sources = _normalize_paths(sources)
    if not project_sources:
        return []
    return [
        state for state in list_output_states(outputs_dir)
        if any(_paths_overlap(source, state.source_root) for source in project_sources)
    ]


def latest_output_state_for_sources(
    sources: Iterable[Path],
    outputs_dir: Path,
) -> ClassificationOutputState | None:
    states = list_output_states_for_sources(sources, outputs_dir)
    if not states:
        return None
    return states[-1]


def write_state(
    state_path: Path,
    *,
    classes: Iterable[str],
    class_directories: dict[str, str],
    source_root: Path,
    records: Iterable[ClassificationRecord],
):
    """Persist classification progress."""

    payload = {
        "task_mode": "classification",
        "source_root": str(Path(source_root).expanduser()),
        "classes": list(normalize_class_names(classes)),
        "class_directories": dict(class_directories),
        "records": [
            {
                "source_path": str(record.source_path),
                "destination_path": str(record.destination_path),
                "class_name": record.class_name,
                "classified_at": record.classified_at,
                "operation": record.operation,
            }
            for record in records
        ],
    }
    # Written on every classified image; compact output keeps the cost flat as the
    # record list grows. Read back via load_state, never by hand.
    state_path = Path(state_path)
    tmp_path = state_path.with_name(f"{state_path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp_path, state_path)


def classify_image_source(
    image_path: Path,
    *,
    class_name: str,
    output_dir: Path,
    class_directories: dict[str, str],
) -> ClassificationRecord:
    """Record the selected class for an image without touching image files."""

    image_path = Path(image_path).expanduser()
    class_dir = class_directories[class_name]
    destination_path = Path(output_dir).expanduser() / class_dir / image_path.name
    return ClassificationRecord(
        source_path=image_path,
        destination_path=destination_path,
        class_name=class_name,
        classified_at=datetime.now().isoformat(timespec="seconds"),
        operation="state",
    )


def export_classification_dataset(
    *,
    records: Iterable[ClassificationRecord],
    classes: Iterable[str],
    class_directories: dict[str, str],
    dataset_root: Path,
) -> dict[str, object]:
    """Export classified images into class subfolders from the JSON state."""

    dataset_root = Path(dataset_root).expanduser()
    directories = dict(class_directories)
    for class_name, dirname in class_directories_for(classes).items():
        directories.setdefault(class_name, dirname)
    for dirname in directories.values():
        (dataset_root / dirname).mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped: list[str] = []
    exported_by_class = {class_name: 0 for class_name in normalize_class_names(classes)}
    for record in _latest_records_by_source(records):
        if record.class_name not in directories:
            skipped.append(str(record.source_path))
            continue
        source_path = _existing_record_image_path(record)
        if source_path is None:
            skipped.append(str(record.source_path))
            continue

        destination_dir = dataset_root / directories[record.class_name]
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = unique_destination_path(destination_dir / Path(record.source_path).name)
        shutil.copy2(source_path, destination_path)
        copied += 1
        exported_by_class[record.class_name] = exported_by_class.get(record.class_name, 0) + 1

    return {
        "dataset_root": dataset_root,
        "copied": copied,
        "skipped": skipped,
        "by_class": exported_by_class,
    }


def transfer_image_to_class(
    image_path: Path,
    *,
    class_name: str,
    output_dir: Path,
    class_directories: dict[str, str],
    move: bool = False,
) -> ClassificationRecord:
    """Copy or move an image into the selected class directory."""

    image_path = Path(image_path).expanduser()
    class_dir = class_directories[class_name]
    destination_dir = Path(output_dir).expanduser() / class_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = unique_destination_path(destination_dir / image_path.name)
    if move:
        shutil.move(str(image_path), str(destination_path))
    else:
        shutil.copy2(image_path, destination_path)
    return ClassificationRecord(
        source_path=image_path,
        destination_path=destination_path,
        class_name=class_name,
        classified_at=datetime.now().isoformat(timespec="seconds"),
        operation="move" if move else "copy",
    )


def copy_image_to_class(
    image_path: Path,
    *,
    class_name: str,
    output_dir: Path,
    class_directories: dict[str, str],
) -> ClassificationRecord:
    """Copy an image into the selected class directory."""

    return transfer_image_to_class(
        image_path,
        class_name=class_name,
        output_dir=output_dir,
        class_directories=class_directories,
        move=False,
    )


def source_looks_used(image_path: Path, output_dir: Path, class_directories: dict[str, str]) -> bool:
    """Return True when an image name already exists in any class subfolder.

    This is a legacy fallback only. The primary filter uses exact source paths
    persisted in ``classification_state.json``.
    """

    image_path = Path(image_path)
    output_dir = Path(output_dir).expanduser()
    for dirname in class_directories.values():
        class_dir = output_dir / dirname
        if not class_dir.is_dir():
            continue
        if any(candidate.is_file() and _same_original_name(candidate.name, image_path.name) for candidate in class_dir.iterdir()):
            return True
    return False


def unique_destination_path(candidate: Path) -> Path:
    """Return a non-existing path by appending a numeric suffix when needed."""

    candidate = Path(candidate)
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    index = 1
    while True:
        next_candidate = candidate.with_name(f"{stem}__{index:03d}{suffix}")
        if not next_candidate.exists():
            return next_candidate
        index += 1


def _existing_record_image_path(record: ClassificationRecord) -> Path | None:
    source_path = Path(record.source_path).expanduser()
    if source_path.exists():
        return source_path
    destination_path = Path(record.destination_path).expanduser()
    if destination_path.exists():
        return destination_path
    return None


def _latest_records_by_source(records: Iterable[ClassificationRecord]) -> tuple[ClassificationRecord, ...]:
    latest: dict[Path, ClassificationRecord] = {}
    order: list[Path] = []
    for record in records:
        source_path = Path(record.source_path).expanduser()
        if source_path not in latest:
            order.append(source_path)
        latest[source_path] = record
    return tuple(latest[source_path] for source_path in order)


def _same_original_name(candidate_name: str, source_name: str) -> bool:
    source = Path(source_name)
    candidate = Path(candidate_name)
    if candidate_name == source_name:
        return True
    pattern = re.compile(rf"^{re.escape(source.stem)}__\d{{3}}{re.escape(source.suffix)}$")
    return bool(pattern.match(candidate.name))


def _read_image_list(list_path: Path) -> list[Path]:
    base_dir = list_path.parent
    images = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return images


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


def _modified_at(path: Path) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime)
    except OSError:
        return None


def _normalize_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
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


def _paths_overlap(left: Path, right: Path) -> bool:
    if not right:
        return False
    try:
        left = Path(left).expanduser().resolve()
    except OSError:
        left = Path(left).expanduser().absolute()
    try:
        right = Path(right).expanduser().resolve()
    except OSError:
        right = Path(right).expanduser().absolute()
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
