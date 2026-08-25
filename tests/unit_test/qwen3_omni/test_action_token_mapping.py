# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy

import pytest

from sglang_omni.models.qwen3_omni.action_token_mapping import (
    allocate_single_token_ids,
    build_mapping_manifest,
    build_runtime_catalog,
    discover_single_token_ids,
    parse_canonical_action_catalog,
    sha256_json,
    validate_mapping_manifest,
)


class FakeTokenizer:
    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        self.vocab = vocab or {
            "AA": 10,
            "AB": 11,
            "AC": 12,
            "AD": 13,
            "AE": 14,
            "AF": 15,
            "A1": 16,
            " bad": 17,
            "<eos>": 99,
        }
        self.reverse = {token_id: token for token, token_id in self.vocab.items()}
        self.all_special_ids = [99]
        self.special_tokens_map = {"eos_token": "<eos>"}
        self.bos_token_id = None
        self.eos_token_id = 99
        self.pad_token_id = 99

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        if text in self.vocab:
            return [self.vocab[text]]
        return [1000 + ord(char) for char in text]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(self.reverse[token_id] for token_id in token_ids)


def _catalog(*action_ids: str) -> dict:
    actions = []
    for index, action_id in enumerate(action_ids):
        group = "基础姿态" if index < 2 else "情绪"
        actions.append(
            {
                "action_id": action_id,
                "source": {"source_label": f"动作{index}"},
                "category_path": ["一级", group, f"动作{index}"],
                "short_definition": f"动作定义{index}",
            }
        )
    return {"schema_version": 1, "actions": actions}


def _generate(
    document: dict,
    tokenizer: FakeTokenizer,
    *,
    previous: dict | None = None,
):
    parsed = parse_canonical_action_catalog(document, add_no_action=False)
    pool = discover_single_token_ids(tokenizer, pattern=r"[A-Z]{2}")
    entries = allocate_single_token_ids(
        parsed.entities,
        pool,
        allocation_salt="test",
        previous_mapping=previous,
    )
    manifest = build_mapping_manifest(
        parsed,
        entries,
        tokenizer=tokenizer,
        model_path="fake-model",
        source_catalog_path="catalog.json",
        source_catalog_sha256=sha256_json(document),
        pattern=r"[A-Z]{2}",
        min_token_id=0,
        max_token_id=None,
        allocation_salt="test",
        pool_size=len(pool),
        category_depth=2,
        add_no_action=False,
    )
    return parsed, manifest


def test_discover_single_token_ids_filters_special_unsafe_and_pattern() -> None:
    pool = discover_single_token_ids(FakeTokenizer(), pattern=r"[A-Z]{2}")

    assert [(item.text, item.token_id) for item in pool] == [
        ("AA", 10),
        ("AB", 11),
        ("AC", 12),
        ("AD", 13),
        ("AE", 14),
        ("AF", 15),
    ]


def test_generate_mapping_is_global_deterministic_and_runtime_ready() -> None:
    tokenizer = FakeTokenizer()
    document = _catalog("act-1", "act-2", "act-3")

    parsed, first = _generate(document, tokenizer)
    _, second = _generate(document, tokenizer)
    runtime = build_runtime_catalog(parsed, first)

    assert first["entries"] == second["entries"]
    active = [entry for entry in first["entries"] if entry["active"]]
    assert len(active) == 5
    assert len({entry["short_id"] for entry in active}) == 5
    assert len({entry["token_id"] for entry in active}) == 5
    assert len(runtime["action_candidates"]) == 2
    assert sum(
        len(category["children"])
        for category in runtime["action_candidates"]
    ) == 3
    assert validate_mapping_manifest(tokenizer, first, parsed=parsed)["valid"]


def test_previous_mapping_preserves_active_and_reserves_retired_tokens() -> None:
    tokenizer = FakeTokenizer()
    old_document = _catalog("act-1", "act-2")
    _, old = _generate(old_document, tokenizer)
    old_by_key = {entry["stable_key"]: entry for entry in old["entries"]}
    new_document = _catalog("act-1", "act-3")

    parsed, new = _generate(new_document, tokenizer, previous=old)
    new_by_key = {entry["stable_key"]: entry for entry in new["entries"]}

    assert new_by_key["action:act-1"]["short_id"] == old_by_key[
        "action:act-1"
    ]["short_id"]
    assert new_by_key["action:act-2"]["active"] is False
    assert new_by_key["action:act-3"]["short_id"] != new_by_key[
        "action:act-2"
    ]["short_id"]
    assert validate_mapping_manifest(tokenizer, new, parsed=parsed)["valid"]


def test_validation_rejects_tokenizer_drift_and_assignment_tampering() -> None:
    tokenizer = FakeTokenizer()
    parsed, manifest = _generate(_catalog("act-1"), tokenizer)
    drifted = FakeTokenizer(
        {
            **tokenizer.vocab,
            "AA": 20,
        }
    )

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_mapping_manifest(drifted, manifest, parsed=parsed)

    tampered = copy.deepcopy(manifest)
    tampered["entries"][0]["short_id"] = tampered["entries"][1]["short_id"]
    with pytest.raises(ValueError, match="validation failed"):
        validate_mapping_manifest(tokenizer, tampered, parsed=parsed)
