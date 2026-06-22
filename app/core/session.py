"""Session configuration shared by startup screens and annotation runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Tuple

from app.config import CONF_THRESHOLD


class AnnotationTaskMode(str, Enum):
    """Supported annotation workflows."""

    TRACKING = "tracking"
    DETECTION = "detection"
    OBB = "obb"
    KEYPOINT = "keypoint"
    CLASSIFICATION = "classification"

    @property
    def label(self) -> str:
        if self is AnnotationTaskMode.TRACKING:
            return "Tracking"
        if self is AnnotationTaskMode.OBB:
            return "Deteccao orientada (OBB)"
        if self is AnnotationTaskMode.KEYPOINT:
            return "Keypoint detection"
        if self is AnnotationTaskMode.CLASSIFICATION:
            return "Classificacao de imagens"
        return "Deteccao padrao"


@dataclass(frozen=True)
class AnnotationSessionConfig:
    """Immutable configuration chosen before the annotation UI starts."""

    mode: AnnotationTaskMode
    data_root: Path
    weights_paths: Tuple[Path, ...] = ()
    target_classes: Tuple[str, ...] = ()
    weights_path: Optional[Path] = None
    output_dir: Path = Path("output")
    annotations_path: Optional[Path] = None
    resume_existing_annotations: bool = False
    category_metadata: Tuple[dict, ...] = ()
    confidence_threshold: float = CONF_THRESHOLD
    classification_move_files: bool = False

    def __post_init__(self):
        object.__setattr__(self, "data_root", Path(self.data_root).expanduser())
        raw_weights = self.weights_paths
        if isinstance(raw_weights, (str, Path)):
            raw_weights = (Path(raw_weights),)
        if not raw_weights and self.weights_path is not None:
            raw_weights = (self.weights_path,)
        object.__setattr__(
            self,
            "weights_paths",
            tuple(Path(p).expanduser() for p in raw_weights),
        )
        object.__setattr__(self, "weights_path", self.weights_paths[0] if self.weights_paths else None)
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser())
        if self.annotations_path is not None:
            object.__setattr__(self, "annotations_path", Path(self.annotations_path).expanduser())
        object.__setattr__(self, "target_classes", normalize_class_names(self.target_classes))
        object.__setattr__(self, "category_metadata", tuple(dict(cat) for cat in self.category_metadata))
        if not self.target_classes:
            raise ValueError("Informe ao menos uma classe.")
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            raise ValueError("confidence_threshold deve ficar entre 0 e 1.")

    @property
    def tracking_enabled(self) -> bool:
        return self.mode is AnnotationTaskMode.TRACKING


def normalize_class_names(values: Iterable[str]) -> Tuple[str, ...]:
    """Return clean class names preserving order and removing duplicates."""

    cleaned = []
    seen = set()
    for value in values:
        name = str(value).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        cleaned.append(name)
    return tuple(cleaned)
