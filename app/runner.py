import os
import sys


def _disable_x_input_method():
    """Detach the X input method (ibus) before Tk connects to the display.

    Tk asks the input method to create an XIC for every widget it maps. Against ibus
    that handshake blocks on a synchronous round-trip, costing ~85 ms per widget and
    turning a normal screen into seconds of frozen UI (measured: 2.9 s for a sidebar
    that takes 136 ms without it).

    Only affects how composed characters reach the app; layouts that produce accents
    through XKB (br, us-intl) keep working. Set INOLABEL_KEEP_IM=1 to opt out if you
    depend on ibus composition inside the annotation window.
    """
    if os.environ.get("INOLABEL_KEEP_IM"):
        return
    if not os.environ.get("XMODIFIERS", "").startswith("@im="):
        return
    if "tkinter" in sys.modules:  # too late: Tk already read the setting
        return
    os.environ["XMODIFIERS"] = "@im=none"


def main() -> int:
    _disable_x_input_method()

    from app.ui.startup.splash import show_splash
    show_splash()

    from app.startup_dialog import ask_startup_config
    from app.annotation_tool import AnnotationTool
    from app.core.session import AnnotationTaskMode

    session_config = ask_startup_config()
    if session_config.mode is AnnotationTaskMode.CLASSIFICATION:
        from app.classification.tool import ClassificationTool

        tool_cls = ClassificationTool
    elif session_config.mode is AnnotationTaskMode.OBB:
        from app.annotation_obb.tool import OBBAnnotationTool

        tool_cls = OBBAnnotationTool
    elif session_config.mode is AnnotationTaskMode.KEYPOINT:
        from app.annotation_keypoint.tool import KeypointAnnotationTool

        tool_cls = KeypointAnnotationTool
    else:
        tool_cls = AnnotationTool

    tool = None
    try:
        tool = tool_cls(session_config=session_config)
        tool.run()
        return 0
    except KeyboardInterrupt:
        if tool is not None:
            tool.finish_processing("Processo interrompido.")
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        if tool is not None:
            tool.finish_processing(f"Erro: {exc}")
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
