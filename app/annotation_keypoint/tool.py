"""Composition of the keypoint annotation tool."""

from app.annotation.state.core_init import CoreInitMixin
from app.annotation.core.services.class_service import ClassServiceMixin
from app.annotation.sources.source_discovery import SourceDiscoveryMixin
from app.annotation.sources.source_loading import SourceLoadingMixin
from app.annotation.roi.roi_state import ROIStateMixin
from app.annotation.roi.roi_projection import ROIProjectionMixin
from app.annotation.application.lifecycle import LifecycleMixin
from app.annotation.presentation.panels.main_window import MainWindowMixin
from app.annotation.presentation.panels.topbar_panel import TopbarPanelMixin
from app.annotation.presentation.panels.statusbar_panel import StatusbarPanelMixin
from app.annotation.presentation.panels.sidebar_panel import SidebarPanelMixin
from app.annotation.presentation.panels.canvas_panel import CanvasPanelMixin
from app.annotation.presentation.widgets.class_panel_widget import ClassPanelWidgetMixin
from app.annotation.ui.display_canvas import DisplayCanvasMixin

from app.annotation_keypoint.state.runtime_state import KPRuntimeStateMixin
from app.annotation_keypoint.sources.source_helpers import KPSourceHelpersMixin
from app.annotation_keypoint.detection.frame_pipeline import KPFramePipelineMixin
from app.annotation_keypoint.detection.workflow_actions import KPWorkflowActionsMixin
from app.annotation_keypoint.detection.review_nav import KPReviewNavMixin
from app.annotation_keypoint.detection.selection_edit import KPSelectionEditMixin
from app.annotation_keypoint.infrastructure.persistence.coco_storage import KPCocoStorageMixin
from app.annotation_keypoint.infrastructure.persistence.export_actions import KPExportActionsMixin
from app.annotation_keypoint.ui.display_canvas import KPDisplayCanvasMixin
from app.annotation_keypoint.ui.display_overlays import KPDisplayOverlaysMixin
from app.annotation_keypoint.ui.display_status import KPDisplayStatusMixin
from app.annotation_keypoint.ui.mouse_events import KPMouseEventsMixin
from app.annotation_keypoint.ui.mode_toggles import KPModeTogglesMixin
from app.annotation_keypoint.ui.ui_controls import KPUIControlsMixin


class KeypointAnnotationTool(
    CoreInitMixin,
    KPRuntimeStateMixin,
    ClassServiceMixin,
    SourceDiscoveryMixin,
    KPCocoStorageMixin,
    SourceLoadingMixin,
    KPSourceHelpersMixin,
    ROIStateMixin,
    ROIProjectionMixin,
    KPFramePipelineMixin,
    KPWorkflowActionsMixin,
    KPReviewNavMixin,
    KPSelectionEditMixin,
    KPExportActionsMixin,
    LifecycleMixin,
    MainWindowMixin,
    TopbarPanelMixin,
    StatusbarPanelMixin,
    SidebarPanelMixin,
    CanvasPanelMixin,
    ClassPanelWidgetMixin,
    KPUIControlsMixin,
    KPDisplayCanvasMixin,
    KPDisplayOverlaysMixin,
    DisplayCanvasMixin,
    KPDisplayStatusMixin,
    KPMouseEventsMixin,
    KPModeTogglesMixin,
):
    """Keypoint detection tool isolated from the other annotation flows."""
    pass
