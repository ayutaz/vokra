# Canary-1B approval and preflight gate

`preflight_gate.py` is a standard-library-only, offline gate shared by the
Canary-1B-Flash and Canary-1B-v2 VAST workers and their remote Apple Silicon
verifiers. Each worker must run it before host/resource checks, scratch or
evidence creation, Python environment synchronization, source/model download,
conversion, or Cargo.

The gate binds an owner-supplied external approval JSON to exactly one variant,
the upstream repository and commit, the immutable `.nemo` archive byte count
and SHA-256, and (for v2) the exact `./model_weights.ckpt` member and byte
count. It also binds `attribution_required: true` and the exact NVIDIA
attribution string already stamped by the native converter. The approval also
carries `no_upload: true`, `decision: "APPROVED"`, a non-placeholder signer,
the manifest SHA-256, and a canonical scope digest covering every one of these
facts.
Duplicate JSON keys, missing or extra fields, symlinked evidence, changed
identities, false upload policy, unresolved signers, and stale/tampered scope
digests are rejected.

`owner-example` is used only by the in-process self-test and is rejected by a
normal gate invocation; production approval must carry the real approving
owner's handle.

The recorded upstream weight license is CC-BY-4.0, matching the repository's
primary-source Canary audit rows. This gate does not authorize publication or
an artifact upload. The actual release archive is hashed by the VAST worker
after this approval gate and is never pulled to the maintainer Mac.

Example approval shape (values must be generated from the exact manifest and
variant; the placeholder below is intentionally not accepted):

```json
{
  "schema": "vokra-canary-1b-approval-v1",
  "variant": "canary-1b-flash",
  "model": "canary-1b-flash",
  "upstream_repo": "nvidia/canary-1b-flash",
  "upstream_revision": "2b6e4d2dacb11cc1b1724de31bb48fe68c26c12e",
  "license_spdx": "CC-BY-4.0",
  "archive_filename": "canary-1b-flash.nemo",
  "archive_bytes": 3540715520,
  "archive_sha256": "3887cce1afdd425429cfc5109575a8f2cffeb07c02c503a9faff7612bd74e324",
  "main_checkpoint_member": null,
  "main_checkpoint_bytes": null,
  "attribution_required": true,
  "attribution_text": "This application uses NVIDIA Canary-1B-Flash (multilingual ASR / AST for English, German, Spanish and French). Model weights are licensed under CC-BY 4.0. Copyright (c) NVIDIA. Source: https://huggingface.co/nvidia/canary-1b-flash",
  "manifest_sha256": "<sha256 of license_gate_manifest.json>",
  "no_upload": true,
  "publication": "NO_UPLOAD",
  "decision": "APPROVED",
  "signer": "<owner handle>",
  "scope_sha256": "<sha256 of canonical scope object>"
}
```

Run the hermetic gate test with:

```bash
uv run --no-project --offline --python 3.12 python tools/parity/canary_1b/preflight_gate.py --self-test
```
