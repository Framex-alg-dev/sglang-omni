"""Explicit structured entity projection for a separately gated quality trial.

Free-form source text and unrecognised structures always remain intact. This
does not select an action or interpret user instructions.
"""

import json
import os


def action_entity_text(text: str) -> str:
    raw = os.environ.get("SGLANG_OMNI_ACTION_ENTITY_OMIT_FIELDS")
    if not raw:
        return text
    fields = json.loads(raw)
    if not isinstance(fields, list) or not all(isinstance(item, str) and item for item in fields):
        raise ValueError("action entity omitted fields must be a JSON array of field names")
    # Physical facts and constraints are never part of the projection trial.
    protected = {"type", "name", "id", "size", "state", "position", "weight",
                 "physical_affordances", "constraints", "allowed_actions",
                 "forbidden_actions", "capabilities", "safety", "posture"}
    if protected.intersection(fields):
        raise ValueError("cannot omit physical facts or action constraints")
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return text
    if not isinstance(data, dict) or not any(key in data for key in fields):
        return text
    # Preserve insertion order, including unknown fields; never recurse into
    # nested facts. Configuration names must be reviewed against real data.
    return json.dumps({key: value for key, value in data.items() if key not in fields}, ensure_ascii=False)
