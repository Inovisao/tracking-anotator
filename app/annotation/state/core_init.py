from app.annotation.shared import *


class CoreInitMixin:
    def __init__(
        self,
        *,
        session_config: Optional[AnnotationSessionConfig] = None,
        data_root: Path = DATA_ROOT,
        weights_path: Path = WEIGHTS_PATH,
        initial_classes=None,
    ):
        if session_config is None:
            raw_classes = initial_classes if initial_classes is not None else TARGET_CLASSES
            session_config = AnnotationSessionConfig(
                mode=AnnotationTaskMode.TRACKING,
                data_root=Path(data_root),
                weights_paths=(Path(weights_path),),
                target_classes=tuple(str(name) for name in raw_classes),
            )

        self.session_config = session_config
        self.task_mode = session_config.mode
        self.tracking_enabled = session_config.tracking_enabled
        self.data_root = session_config.data_root
        self.weights_paths = list(session_config.weights_paths)
        self.output_dir = session_config.output_dir
        self.output_images_dir = self.output_dir / "images"
        self.saved_states_dir = self.output_dir / "saved_data_states"
        self.annotations_path = session_config.annotations_path or (
            self.saved_states_dir / "annotations.coco.json"
        )
        self.coco_detection_export_path = self.output_dir / "annotations_detection.coco.json"
        self.yolo_dataset_dir = self.output_dir / "yolo_dataset"
        self.homography_path = self.saved_states_dir / "homography.json"
        self.conf_threshold = session_config.confidence_threshold
        self._initial_classes = list(session_config.target_classes)
        self._initial_categories = [dict(cat) for cat in session_config.category_metadata]

        self._validate_required_paths()
        self.video_files = self.discover_sources(self.data_root)
        if not self.video_files:
            raise FileNotFoundError(f"Nenhuma fonte valida encontrada em {self.data_root}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.saved_states_dir.mkdir(parents=True, exist_ok=True)

        self._initialize_model_state()
        self._initialize_runtime_state()
        self._build_ui()
        # Cover the gap while the page is still loading (UI build, first frame,
        # and possibly the model/torch) — the main window stays withdrawn here.
        from app.ui.components.loading import LoadingScreen
        loading = LoadingScreen(self.window, "Preparando anotação...")
        try:
            if self.session_config.resume_existing_annotations:
                self.load_existing_annotations()
                self.current_video_index = self._initial_source_index_from_annotation_state()
            self.register_signal_handlers()
            if self.weights_paths:
                loading.update_message("Carregando modelo...")
            self.start_video(self.current_video_index)
        finally:
            loading.close()
        self.window.deiconify()  # show only after the first frame is rendered

    def _validate_required_paths(self):
        if not self.data_root.exists():
            raise FileNotFoundError(f"Origem de dados nao encontrada: {self.data_root}")

    def _initial_source_index_from_annotation_state(self) -> int:
        state = getattr(self, "annotation_state", {}) or {}
        try:
            index = int(state.get("last_active_source_index", 0) or 0)
        except (TypeError, ValueError):
            return 0
        if 0 <= index < len(self.video_files):
            return index
        return 0

    def _initialize_model_state(self):
        # models[i] is None until first inference triggers lazy loading
        self.models: List = [None] * len(self.weights_paths)
        self.model = None  # points to models[0] after loading (legacy compat.)
        self.target_classes = [str(name).strip() for name in self._initial_classes if str(name).strip()]
        self.class_to_category_id: Dict[str, int] = {}
        self.categories: List[dict] = []
        for cat in self._initial_categories:
            name = str(cat.get("name", "")).strip()
            if not name:
                continue
            cid = int(cat.get("id", len(self.categories) + 1) or len(self.categories) + 1)
            cat["id"] = cid
            cat["name"] = name
            self.categories.append(cat)
            self.class_to_category_id[name] = cid
        for cname in self.target_classes:
            self.register_category(cname)
        self.uses_text_prompt = False
        self.apply_target_classes(self.target_classes)

    def ensure_models_loaded(self) -> List:
        """Lazily loads all models and returns the list."""
        from ultralytics import YOLO  # deferred: importing torch costs ~4s
        for i, weights_path in enumerate(self.weights_paths):
            if self.models[i] is not None:
                continue
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights not found: {weights_path}")
            self.models[i] = YOLO(str(weights_path))
        if self.model is None and self.models:
            self.model = self.models[0]
            self.apply_target_classes(self.target_classes)
        return [m for m in self.models if m is not None]

    def ensure_model_loaded(self):
        """Loads and returns the first model (compatibility with legacy code)."""
        return self.ensure_models_loaded()[0]
