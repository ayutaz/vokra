# CosyVoice2 HiFT official reference gates

This directory is the isolated Python project for the official CosyVoice2 HiFT
oracle.  It is intentionally Linux `x86_64` CPU-only and keeps the package
closure to the pinned `torch==2.7.1`, `numpy==2.3.5`, and `scipy==1.16.3` plus
their transitive CPU wheel dependencies.  The explicit `pytorch-cpu` index is
used for torch; numpy and scipy resolve from PyPI.  `triton`, NVIDIA packages,
`librosa`, `soxr`, and `soundfile` are forbidden.

The source checkout must be the clean CosyVoice repository at
`8555549e882236e6541748b1042d95693caa82ba`, with the four source-role blob
identities in `license_gate_manifest.json`.  The model input is the exact
`FunAudioLLM/CosyVoice2-0.5B` `hift.pt` at revision
`eec1ae6c79877dbd9379285cf8789c9e0879293d`; its 83,390,254-byte SHA-256 is
authenticated by both utilities.  The config snapshot is likewise checked by
size, SHA-256, and Git blob SHA-1.

License and publication status is deliberately fail-closed:
`PENDING_REVIEW`, `OWNER_SIGNOFF_REQUIRED`, and `NO_UPLOAD`.  The checked-in
manifest is not authorized for production execution.  The preflight gate
intentionally exits blocked until an owner changes it to the explicit approved
status/signoff pair; do not run the preparer or dumper before that approval.
No model or source checkout is downloaded by these tools.  First stage and
audit the Linux wheels, then have the owner review the candidate and update the
exact license manifest and approval scope.  Only after that review run the
stdlib license gate; only a passing gate permits project sync or model scripts.
All steps below are VAST-only.  The stage/audit commands are stdlib-only and
must run before any project sync; do not invoke model scripts via their shebang
or the project environment before the license gate succeeds:

`VOKRA_PUBLISH_ON_VAST=1` is an execution-location guard for the VAST/Linux
jobs only.  It is not publication authorization: the staged closure and audit
candidate remain `NO_UPLOAD`, and the license gate is still required.

```sh
VOKRA_PUBLISH_ON_VAST=1 uv run --no-project --offline --python 3.12 \
  python tools/parity/cosyvoice2_hift_reference/preflight_linux_closure.py \
  --output /vast/output/linux-wheels

VOKRA_PUBLISH_ON_VAST=1 uv run --no-project --offline --python 3.12 \
  python tools/parity/cosyvoice2_hift_reference/audit_linux_closure.py \
  --artifacts /vast/output/linux-wheels \
  --output /vast/output/linux-closure-candidate.json

# Owner reviews the candidate, updates the exact license manifest, and records
# the matching approval scope. Then run the stdlib-only authorization gate.
uv run --no-project --offline --python 3.12 \
  python tools/parity/cosyvoice2_hift_reference/preflight_gate.py \
  --license-manifest /vast/input/license_gate_manifest.json

# Only after the gate passes:
uv sync --frozen --project tools/parity/cosyvoice2_hift_reference

VOKRA_PUBLISH_ON_VAST=1 uv run --frozen --offline --project tools/parity/cosyvoice2_hift_reference \
  python tools/parity/cosyvoice2_hift_prepare_checkpoint.py \
  --checkpoint /vast/input/hift.pt \
  --manifest /vast/input/tensor_manifest.json \
  --license-manifest /vast/input/license_gate_manifest.json \
  --output /vast/output/hift.safetensors

VOKRA_PUBLISH_ON_VAST=1 uv run --frozen --offline --project tools/parity/cosyvoice2_hift_reference \
  python tools/parity/cosyvoice2_hift_dump_reference.py \
  --source /vast/input/CosyVoice \
  --checkpoint /vast/input/hift.pt \
  --config /vast/input/cosyvoice2.yaml \
  --license-manifest /vast/input/license_gate_manifest.json \
  --output /vast/output/reference
```

The preparer uses `torch.load(weights_only=True)`, requires the caller's exact
328-name/F32 manifest, preserves tensor names/dtypes, and atomically emits a
plain F32 safetensors file.  The dumper imports and instantiates only the
official `HiFTGenerator` and `ConvRNNF0Predictor`, runs a fixed finite mel
through the official forward path, and exclusively claims an absent output
directory.  It writes the exact input `mel.f32`, then `pcm.f32` and `f0.f32`,
and writes `manifest.json` last as the completion marker.  Downstream parity
consumers must read `mel.f32` rather than regenerate the input formula.
Existing outputs are rejected and failed runs clean up the claimed directory.
These commands are single-writer VAST jobs:
the output paths must remain absent for the duration of a run; a directory
without its final manifest is incomplete.

The Linux closure preflight stages only the selected hash/size-bound wheels into
an absent directory.  The archive audit reads those wheels without importing
them, inventories every license/notice and native payload hash, and emits only
an `OWNER_REVIEW_REQUIRED` candidate with `PENDING_PACKAGE_AND_NATIVE_PAYLOAD_REVIEW`
and `NO_UPLOAD`.  It never infers a license or creates approval.  `uv sync` is
forbidden until the reviewed manifest and approval scope pass the stdlib gate.

The stdlib preflight self-test is safe to run locally without syncing the
Linux-only environment:

```sh
uv run --no-project --offline --python 3.12 \
  python tools/parity/cosyvoice2_hift_reference/preflight_gate.py --self-test
```

The preparer and dumper self-tests are model-free, but import the pinned torch
closure and therefore run only after owner authorization on a Linux x86_64 VAST
worker:

```sh
VOKRA_PUBLISH_ON_VAST=1 uv run --frozen --project tools/parity/cosyvoice2_hift_reference \
  python tools/parity/cosyvoice2_hift_prepare_checkpoint.py --self-test
VOKRA_PUBLISH_ON_VAST=1 uv run --frozen --project tools/parity/cosyvoice2_hift_reference \
  python tools/parity/cosyvoice2_hift_dump_reference.py --self-test
```
