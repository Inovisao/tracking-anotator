import sys


def main() -> int:
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
