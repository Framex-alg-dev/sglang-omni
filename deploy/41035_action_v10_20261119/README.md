# 41035 Action V10 runtime snapshot

This directory contains the non-secret runtime inputs for the Action service
currently consumed by the 41035 Agent. It intentionally keeps the existing
`/v1/action-recall` interface instead of introducing a shared Reply/Event/Action
contract.

## Runtime identity

- Model ID and version: `v10-boundary-ep1-final-fp8`
- Service role: `action-decision`
- Contract: `action-decision.v1`
- Channels: `body`, `expression`
- Catalog and mapping version: `taxonomy4-global210-review-v1`
- Default port: `18049`

The model weights and bearer token are external deployment inputs and are not
stored in Git. `SGLANG_OMNI_ACTION_DECISION_TOKEN` must be supplied by the
service environment.

## Provenance

On 2026-11-23, the active service source at
`/data/xingmt/action_recall_v10_20261119/sglang_omni` was compared against the
captured release tree: all 506 non-bytecode files matched by relative path and
SHA-256. The `action_decision` source in this branch is copied from that exact
release. The seven files under `assets/` also match the active deployment
byte-for-byte.

The branch additionally preserves the newer Reply and Event code paths. They
must not be replaced with the older shared files from the standalone Action
release.

## Launch

Use `run_service.sh` after providing a Python environment, model directory, GPU
selection, and token. The defaults retain the active model identity and port;
deployment-specific paths can be supplied through environment variables.
