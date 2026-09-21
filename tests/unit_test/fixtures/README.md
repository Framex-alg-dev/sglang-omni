# Realtime evaluation fixtures

`realtime_action_recall_cases.json` is the fixed oracle for hierarchical
Category recall and final Child selection.  It currently contains 245 unique
Chinese text inputs:

- 150 `catalog_definition_seed` cases provide three concrete-action probes for
  each ordinary category.  They are broad coverage, not evidence of natural
  language quality.
- 75 `changed_category_boundary` cases provide three reviewed natural requests
  for each of the 25 categories whose descriptions are under evaluation.  This
  is the primary group for the category-description before/after comparison.
- 10 `system_route` cases cover reply and silent accompaniment categories.
- 10 `restricted_category_unsupported` cases request an action outside a
  deliberately narrowed Session catalog and expect raw Category Top-1 `00`.

Report the groups independently.  In particular, do not combine the easier
catalog-definition seeds with the reviewed boundary cases and present the
result as one natural-language accuracy number.

Each case separates `acceptable_category_ids`, `acceptable_candidate_ids`, and
`support_status`.  An empty candidate list means that only category routing is
judged.  Multiple acceptable IDs are supported by the schema for actions with
more than one valid catalog membership.

The generator is an authoring and validation aid, not a dynamic oracle:

```bash
python scripts/generate_realtime_action_recall_fixture.py
python -m pytest -q \
  tests/unit_test/qwen3_omni/test_action_recall_fixture.py
```

After any catalog change, review the generated diff before accepting it.  The
quality test intentionally fails when the committed fixture and generator no
longer agree.

For model evaluation, retain the raw Category ranking even when the production
pipeline later applies adaptive Top-1 or an exact-action shortcut.  Recommended
reports include Category Top-1 accuracy, Top-2 recall, confusion matrix, final
candidate accuracy, unsupported precision/recall, adaptive Top-1 coverage, and
Category/Child latency with cold and warm cache samples separated.
