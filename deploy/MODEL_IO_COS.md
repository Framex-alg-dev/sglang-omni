# Omni model I/O archive

Event, Router, Reply and Action enqueue their exact model request and terminal
model output into a bounded in-process queue. A daemon writer serializes them to
a local NVMe spool; no inference coroutine performs filesystem or COS I/O.

Run one uploader sidecar against the shared spool:

```bash
python -m sglang_omni.serve.model_io.uploader \
  --root /var/lib/sglang-omni/model-io \
  --credentials-file /run/secrets/omni-cos.json
```

The credential file is supplied by the deployment secret manager and has this
shape: `{"secret_id":"...","secret_key":"...","token":"optional"}`.
The uploader image must install `model-io-uploader-requirements.in`; the
model-worker image does not need the COS SDK.

The COS layout is intentionally shallow for console lookup:
`omni/model-io/v1/{root_session_id}/YYYY-MM-DD/{stage}/`.
Each invocation is a flat set of `{turn_id}__{request_id}__{invocation_id}.input.json`,
`.output.json`, `.manifest.json`, and optional `.media.*` objects. The local spool keeps the more explicit
`dt=/root=/turn=/request=/invocation=` hierarchy for recovery and pruning.
Payload objects are uploaded first with AES256 server-side encryption and
`manifest.json` is uploaded last as the commit marker. Local `.cos-uploaded.json`
and `.cos-retry.json` files provide idempotency, retention and exponential retry.

COS outages, queue exhaustion and local byte-cap exhaustion fail open for the
model call. CLS receives metadata-only `artifact.archive.*` and
`artifact.upload.*` records; raw prompts, media and outputs stay out of CLS.
