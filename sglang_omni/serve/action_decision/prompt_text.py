"""Plain-text rendering for persona audio recall prompts."""

from collections.abc import Mapping


def quoted_text(text: str) -> str:
    """Keep data on quoted lines so it cannot introduce a bare section marker."""
    return "\n".join("> " + line for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"))


def fields_text(fields: Mapping, indent: str = "") -> str:
    lines = []
    for key, value in sorted(fields.items()):
        label = indent + str(key).replace("\r", " ").replace("\n", " ")
        if isinstance(value, Mapping):
            lines.extend((f"{label}:", fields_text(value, indent + "  ")))
        elif isinstance(value, (list, tuple)):
            lines.append(f"{label}: " + "、".join(str(x).replace("\n", " ").replace("\r", " ") for x in value))
        elif isinstance(value, str) and len(value.splitlines()) > 1:
            lines.extend((f"{label}:", quoted_text(value)))
        else:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def render_observation(fields: Mapping) -> str:
    """Render media descriptions before the verbatim keyboard input."""
    metadata = fields_text({key: value for key, value in fields.items() if key != "typed_text"})
    text = fields.get("typed_text")
    if text is None:
        return metadata
    return metadata + "\n用户原文：\n" + text if metadata else text
