"""Catalog output strings must agree with both frozen decoding maps."""

import json

import pytest

from sglang_omni.serve.action_decision.catalog import ActionCatalogRegistry
from tests.unit_test.serve.test_persona_window_prompt import config


@pytest.mark.parametrize("bad_code", ["_x_e", "", None, ["_x", "_d"]])
def test_inconsistent_catalog_output_code_is_rejected(tmp_path, bad_code):
    c = config()
    product = json.loads(open(c.product_catalog_path).read())
    row = next(row for cat in product["categories"] for row in cat["children"]
               if row["candidate_id"] == "260")
    row["output_code"] = bad_code
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(product))
    with pytest.raises(ValueError, match="output_code.*260"):
        ActionCatalogRegistry(mapping_path=c.mapping_path, full_mapping_path=c.full_mapping_path,
                              product_catalog_path=path, agent_policy_path=c.agent_policy_path)


def test_all_catalog_output_codes_match_and_older_exports_remain_readable(tmp_path):
    c = config()
    product = json.loads(open(c.product_catalog_path).read())
    kwargs = dict(mapping_path=c.mapping_path, full_mapping_path=c.full_mapping_path,
                  agent_policy_path=c.agent_policy_path)
    registry = ActionCatalogRegistry(product_catalog_path=c.product_catalog_path, **kwargs)
    rows = [row for cat in product["categories"] for row in cat["children"]]
    assert len(rows) == 116
    for row in rows:
        assert registry._entry(row["candidate_id"]).code == row.pop("output_code")
    path = tmp_path / "legacy_catalog.json"
    path.write_text(json.dumps(product))
    legacy = ActionCatalogRegistry(product_catalog_path=path, **kwargs)
    assert [legacy._entry(row["candidate_id"]) for row in rows] == [
        registry._entry(row["candidate_id"]) for row in rows]
