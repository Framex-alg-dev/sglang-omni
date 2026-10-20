"""An opt-in encoder domain; never a text/KV cache owner."""
import hashlib
import json
import os


def turn_media_scope(session_id: str, turn_id: str, identity_epoch, input_revision):
    domain = os.getenv("SGLANG_OMNI_SHARED_MEDIA_CACHE_DOMAIN", "").strip()
    if (not domain or type(identity_epoch) is not int or identity_epoch < 0
            or type(input_revision) is not int or input_revision < 0):
        return None
    encoded = json.dumps([domain, session_id, identity_epoch, turn_id, input_revision],
                         separators=(",", ":")).encode()
    return "turn-media-v1:" + hashlib.sha256(encoded).hexdigest()
