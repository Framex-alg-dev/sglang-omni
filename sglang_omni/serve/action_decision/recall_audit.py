"""Replayable model input; media stored separately by content hash."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


def recall_variant(request) -> str:
    return "explicit_action_request" if request.requested_action is not None else "standard_recall"


def input_snapshot(request, catalog, system_prompt, user_content) -> dict:
    root = os.environ.get("SGLANG_OMNI_ACTION_REPLAY_ROOT")
    media = []
    for clip in request.media:
        digest = hashlib.sha256(clip.payload).hexdigest()
        item = dict(kind=clip.kind, evidence_role=clip.evidence_role, encoding=clip.encoding,
                    start_ms=clip.start_ms, end_ms=clip.end_ms, sha256=digest, bytes=len(clip.payload))
        if root:
            try:
                folder = Path(root) / "media" / digest[:2]
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / digest
                # Publish only complete media. Concurrent identical requests
                # cannot observe another writer's partially written payload.
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=folder, delete=False) as stream:
                        temporary = Path(stream.name)
                        stream.write(clip.payload)
                    try:
                        os.link(temporary, path)
                    except FileExistsError:
                        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                            raise OSError("replay media hash mismatch")
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
                item["path"] = str(path)
            except OSError as exc:
                item["save_error"] = str(exc)
        media.append(item)
    return dict(
        recall_variant=recall_variant(request),
        requested_action=request.requested_action,
        decision_mode=request.runtime_context.get("decision_mode"),
        character_prompt=request.character_prompt, session_prompt=request.session_prompt,
        user_text=request.text, avatar_current_text=request.avatar_current_text,
        avatar_history_text=request.avatar_history_text,
        runtime_context=request.runtime_context,
        catalog_version=catalog.catalog_version, mapping_hash=catalog.mapping_hash,
        catalog_snapshot=[dict(candidate_id=e.candidate_id,code=e.code,label=e.label,
                               motion_definition=e.motion_definition) for e in catalog.entries],
        system_prompt=system_prompt, user_content=user_content,
        system_sha256=hashlib.sha256(system_prompt.encode()).hexdigest(),
        media=media, media_replay_complete=all("path" in m for m in media),
    )
