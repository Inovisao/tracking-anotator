from app.annotation.infrastructure.persistence.export_actions import ExportActionsMixin
from app.annotation.presentation.export.export_screen import ExportScreenMixin
from app.annotation_keypoint.infrastructure.export.yolo_pose_exporter import export_yolo_pose_dataset
from app.annotation_keypoint.infrastructure.export.coco_keypoints_exporter import (
    export_coco_keypoints,
    keypoint_payload_errors,
)


class KPExportActionsMixin(ExportScreenMixin, ExportActionsMixin):
    """Reuses the shared export screen; only the format writers are pose-specific."""

    def sync_export_metadata(self):
        return None

    def _export_yolo_format(self, payload, yolo_root, config, on_progress):
        errors = keypoint_payload_errors(payload)
        if errors:
            raise ValueError(" | ".join(errors))
        report = export_yolo_pose_dataset(
            payload,
            yolo_root,
            self.output_images_dir,
            split_ratios=config.split_ratios if config.use_split else None,
            augmentation_preset=config.augmentation,
            on_progress=on_progress,
        )
        return (
            f"YOLO Pose: {report['images']} imagens, {report['labels']} labels",
            f"YOLO Pose imgs={report['images']}",
        )

    def _export_coco_format(self, payload, coco_dir, on_progress):
        errors = keypoint_payload_errors(payload)
        if errors:
            raise ValueError(" | ".join(errors))
        summary = export_coco_keypoints(
            payload, coco_dir / "annotations.coco.json", self.output_images_dir, on_progress
        )
        return f"COCO Keypoints: {summary['images']} imagens", f"COCO imgs={summary['images']}"
