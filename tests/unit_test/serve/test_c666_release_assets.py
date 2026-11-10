"""Prevent shipping a C666 output codebook with E68-A catalog/scenario codes."""
import hashlib
import json
from pathlib import Path

from sglang_omni.serve.action_decision.catalog import ActionCatalogRegistry
from sglang_omni.serve.action_decision.contracts import ActionDecisionRequest, DecisionChannel
from sglang_omni.serve.action_decision.usage_scenarios import parse_usage_scenarios

RELEASE = Path(__file__).resolve().parents[3] / 'deploy/c666_pro5000_7_20261110'


def test_catalog_output_codes_match_the_complete_c666_mapping():
    assets = RELEASE / 'assets'
    manifest = json.loads((assets / 'provenance.json').read_text())
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((assets / name).read_bytes()).hexdigest() == digest
    mapping = json.loads((assets / 'action_pair_token_map_business111.json').read_text())
    codes = {row['candidate_id']: row['output_text'] for row in mapping['action_entries']}
    catalog = json.loads((assets / 'action_catalog.json').read_text())
    rows = [row for category in catalog['categories'] for row in category['children']]
    assert len(rows) == len(codes) == 111
    assert {row['candidate_id']: row['output_code'] for row in rows} == codes
    assert codes['271'] == '_v_o' and codes['162'] == '_j_k'


def test_c666_scenes_match_both_channels_and_release_identity():
    assets = RELEASE / 'assets'
    registry = ActionCatalogRegistry(
        mapping_path=str(assets / 'action_pair_token_map_current.json'),
        full_mapping_path=str(assets / 'action_pair_token_map_full.json'),
        product_catalog_path=str(assets / 'action_catalog.json'),
        agent_policy_path=str(assets / 'agent_action_policy.json'),
    )
    for channel, expected_count, filename in (
        (DecisionChannel.BODY, 98, 'action_usage_scenarios_v116_r4.json'),
        (DecisionChannel.EXPRESSION, 13, 'action_usage_scenarios_v116.json'),
    ):
        request = ActionDecisionRequest(request_id='release-check',session_id='s',turn_id='t',
            decision_point_id='d',channel=channel,text='你好',character_prompt='自然友好的助手',
            runtime_context=dict(prompt_profile='persona-audio-window-v2',decision_mode='user_request',
                persona_revision='c666-r1',last_dispatched_action=None))
        catalog = registry.resolve(request)
        assert sum(row.kind == 'action' for row in catalog.entries) == expected_count
        assert {row.candidate_id for row in catalog.entries if row.kind != 'action'} == {'000'}
        parse_usage_scenarios((RELEASE / 'usage_scenarios' / filename).read_bytes()).validate_catalog(catalog)
