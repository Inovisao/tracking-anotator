"""State persistence helpers for the manual classification tool."""

from __future__ import annotations

from pathlib import Path

from app.classification.dataset import ClassificationRecord, load_state, write_state


class _RecordList(list):
    """List of records that invalidates the owner's cache on in-place mutation."""

    def __init__(self, iterable=(), *, owner):
        super().__init__(iterable)
        self._owner = owner

    def _invalidate(self):
        self._owner._classified_sources_cache = None

    def append(self, item):
        super().append(item)
        self._invalidate()

    def extend(self, iterable):
        super().extend(iterable)
        self._invalidate()

    def insert(self, index, item):
        super().insert(index, item)
        self._invalidate()

    def remove(self, item):
        super().remove(item)
        self._invalidate()

    def pop(self, index=-1):
        item = super().pop(index)
        self._invalidate()
        return item

    def clear(self):
        super().clear()
        self._invalidate()


class ClassificationStateMixin:
    def _load_existing_state(self):
        state = load_state(self.state_path)
        if state is None:
            self._save_state()
            return
        if state.classes:
            self.classes = list(state.classes)
        if state.class_directories:
            self.class_directories = dict(state.class_directories)
            for dirname in self.class_directories.values():
                (self.output_dir / dirname).mkdir(parents=True, exist_ok=True)
        self.records = self._latest_records_by_source(state.records)

    def _save_state(self):
        write_state(
            self.state_path,
            classes=self.classes,
            class_directories=self.class_directories,
            source_root=self.data_root,
            records=self.records,
        )

    def _counts_by_class(self) -> dict[str, int]:
        counts = {class_name: 0 for class_name in self.classes}
        for record in self.records:
            counts[record.class_name] = counts.get(record.class_name, 0) + 1
        return counts

    @property
    def records(self) -> list[ClassificationRecord]:
        return self._records

    @records.setter
    def records(self, value):
        # Reassignment always invalidates. In-place appends go through
        # _RecordList below, so the cache can never silently go stale.
        self._records = _RecordList(value, owner=self)
        self._classified_sources_cache = None

    def invalidate_records_cache(self):
        self._classified_sources_cache = None

    def _classified_sources(self) -> set[Path]:
        """Set of already-classified sources, cached between navigations.

        Rebuilt only when records change — it is walked on every image change to skip
        classified entries, and rebuilding it each time made navigation cost grow with
        the number of images already done.
        """
        cache = getattr(self, "_classified_sources_cache", None)
        if cache is None:
            cache = {record.source_path for record in self.records}
            self._classified_sources_cache = cache
        return cache

    @staticmethod
    def _latest_records_by_source(records) -> list[ClassificationRecord]:
        latest: dict[Path, ClassificationRecord] = {}
        order: list[Path] = []
        for record in records:
            source_path = Path(record.source_path).expanduser()
            if source_path not in latest:
                order.append(source_path)
            latest[source_path] = record
        return [latest[source_path] for source_path in order]

    def _record_for_source(self, source_path: Path) -> ClassificationRecord | None:
        for record in reversed(self.records):
            if record.source_path == source_path:
                return record
        return None

    def _remove_previous_classification(self, record: ClassificationRecord):
        self.records = [item for item in self.records if item != record]
        self.undo_stack = [item for item in self.undo_stack if item != record]
