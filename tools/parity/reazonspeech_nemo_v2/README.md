# ReazonSpeech NeMo v2 preflight

This directory contains the model-free approval contract for the pinned
ReazonSpeech NeMo v2 validation route.  The source is fixed to
`reazon-research/reazonspeech-nemo-v2` revision
`33693408be76b7cba9fd4a7546a0a8772430211b`, with the authenticated
2,477,946,880-byte archive SHA-256 recorded in
`license_gate_manifest.json`.  The model license is Apache-2.0 and the
validation contract is `no_upload=true`.

`preflight_gate.py` must pass before either worker may probe a host, create
scratch/evidence directories, sync dependencies, download or load a model, or
invoke Cargo.  It validates the exact generic parity project/lock bytes and a
separate external approval JSON whose signer and canonical scope digest are
bound to those bytes and this manifest.  `operator_approval` deliberately
remains `PENDING_EXTERNAL`; the existing `docs/license-audit.md` row is factual
license evidence only and is not an operator approval.

Model execution, conversion, and network access are intentionally absent from
the gate and its self-test.  Use `uv run --no-cache --no-project --offline
--python 3.12 python preflight_gate.py --self-test` for the model-free test.
