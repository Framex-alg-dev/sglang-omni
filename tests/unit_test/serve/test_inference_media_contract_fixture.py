from __future__ import annotations

import hashlib
import json
from pathlib import Path


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "inference_media_contract_v4.json"
)
EXPECTED_SHA256 = "af11cb962b95702e734b6a195fa1df086e8f5a8466014010fade327207c49e5c"


def test_fixture_is_the_frozen_v4_v2_contract() -> None:
    raw = FIXTURE.read_bytes()
    contract = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256
    assert contract["outer_contract_version"] == 4
    assert contract["inner_contract_version"] == 2
    assert contract["evidence_roles"]["audio"] == ["user_audio"]
    assert contract["evidence_roles"]["image"] == ["user_camera", "avatar_state"]
    assert not contract["renderer_submission_receipt"]["send_completion_is_acceptance"]
