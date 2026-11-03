"""Camera-backed facial-expression scope, independent of session orchestration."""
import unicodedata

_VISUAL_EXPRESSION_SCOPE_MARKERS = (
    "表情",
    "神情",
    "脸部",
    "expression",
    "facialexpression",
)

def _normalized_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith(("P", "Z"))
    )

def is_visual_deictic_expression_request(
    *,
    face_task: str,
    has_user_camera: bool,
) -> bool:
    """Return whether a camera-backed request names the expression range."""

    normalized_task = _normalized_label(face_task)
    return bool(
        has_user_camera
        and any(
            marker in normalized_task
            for marker in _VISUAL_EXPRESSION_SCOPE_MARKERS
        )
    )
