# Dia reference dependency approval

## 2026-09-30 Torch security-refresh candidate (blocked)

The dedicated candidate lock now resolves CPU Torch `2.13.0` and
TorchAudio `2.11.0` (including their platform-specific `+cpu` Linux wheels)
from the explicit PyTorch CPU index.  The TorchAudio version is a compatibility
candidate only; the [official TorchAudio installation matrix](https://docs.pytorch.org/audio/main/installation.html)
is the source for the stated PyTorch/TorchAudio compatibility claim.
The exact candidate inputs are:

```text
pyproject.toml  sha256=4dcc396ff3f7387b4b00b32db00ad79fa38f3cf1ef7ad22e3e7f3f8f563be4eb
uv.lock         sha256=06d1f30607934c822c12fdef1db62369f2af0a72372e19d2ba782ffb95583449
```

## 2026-10-05 primary TorchAudio metadata fact (not approval)

The exact official Linux CPU TorchAudio wheel above was read in memory and
verified against its full SHA-256.  Its distribution-owned
`torchaudio-2.11.0+cpu.dist-info/METADATA` member is 6,865 bytes with SHA-256
`e77a84f87ce319a673f35dccb5658112ed7e5c1465e8a10ddc947f0136ed2d7c`.  The
wheel URL is
`https://download-r2.pytorch.org/whl/cpu/torchaudio-2.11.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl`,
and the authenticated wheel identity is 341,338 bytes with SHA-256
`2354248848d06a9ae1e7a12165f800f0dda7df60ecac9fca892322b722b922c0`.
The metadata declares `Name: torchaudio`, `Version: 2.11.0+cpu`, and zero
`Requires-Dist` lines.  This explains the intentionally empty TorchAudio
dependency edge in the lock; it does not establish Torch 2.13 ABI
compatibility, reconcile the authenticated Dia source's 2.6.0 pin, establish
native payload licensing, or provide owner/legal approval.  `PENDING_REVIEW`
and `NO_UPLOAD` remain unchanged.

This is a security-refresh candidate, not a compatibility or parity result.
The authenticated Dia source revision `2811af1c5f476b1f49f4744fabf56cf352be21e5`
declares `torch==2.6.0` and `torchaudio==2.6.0`; it does not declare support
for the candidate Torch 2.13.0/TorchAudio 2.11.0 pair.  No local Torch installation, model load, forward, or
weight acquisition was performed.  A VAST model-free API probe and an
owner-approved real-weight CPU parity run are required before this candidate
can replace the reference environment.

`run-dia-1-6b-validation.sh` therefore exits with
`BLOCKED_UPSTREAM_PINNED_COMPATIBILITY` before any model/input path is
accepted.  The separate `--source-only` inspection path remains available on
VAST for the model-free upstream API probe.

The probe is intentionally explicit about its remote-only boundary.  On the
authenticated Dia checkout and this exact candidate project, VAST may run:

```sh
VOKRA_PUBLISH_ON_VAST=1 uv run --frozen --project /path/to/vokra/tools/parity/dia_1_6b_reference \
  /path/to/vokra/tools/parity/dia_1_6b_reference/upstream_compat_probe.py \
  --source /path/to/dia-2811af1c5f476b1f49f4744fabf56cf352be21e5 \
  --output /dev/shm/dia-upstream-compatibility.json
```

This performs source revision/dependency/signature checks and a source import
probe with offline Hugging Face variables; the probe itself rejects non-Linux,
non-x86_64, or sub-60,000,000-KiB workers before any Torch subprocess starts.
It never constructs a model, downloads weights, or performs a forward pass.
Expected exit status is `2` until the upstream Torch/TorchAudio pins, the
candidate lock, and an exact owner-approved real-weight CPU parity decision are
reconciled.  Its JSON keeps `module_status` separate from
`closure_status`; a module import with missing or incompatible TorchAudio is
reported as `BLOCKED_TORCH_TORCHAUDIO_CLOSURE`, never as a compatibility pass.

This guard is intentionally unconditional: an owner-approved parity record
cannot make the current candidate continue into the later validation stages.
After the VAST probe has recorded source-import/signature and
Torch/TorchAudio compatibility facts, and the owner has made an exact
decision, a separately reviewed change must replace this guard and refresh
the adapter's lock binding before real-weight CPU parity is allowed.

A fresh exact-head VAST dependency/native collector at checkout
`f9e095f7` recorded report SHA-256
`0697b2beb83b6385a9dda3476f7a14bac3ebcedfb4ff1727d30b3629863a9881` and
preparation SHA-256
`f54032397a24f5a701290e6928a3a6123aefff7308d28452c9492edd02b595f3`:
27 active packages, exact closure, 147 publisher-license file records with raw
bytes captured, 164 native files, and no collection failures.  It generated
unsigned owner-scope SHA-256
`651a186c62c4d381f2bec6400b54acf0c3bb7f9ed4afa12ca30193d9f79d3214`, with
status `PENDING_OWNER_REVIEW` and publication `NO_UPLOAD`.

The historical 2.6.0 26-package VAST audit and owner scope below are
superseded for this candidate.  They must not be rehashed, relabeled, or used
for approval.  The protected license manifest remains bound to that historical
closure, so publication and execution stay fail-closed until owner/legal
reviews the fresh exact-head evidence.

The Dia reference worker keeps the generated dependency audit fail-closed:
`BLOCKED_UNREVIEWED_TRANSITIVE` and `NO_UPLOAD` remain factual properties of
the pinned `uv.lock` closure.  An owner decision is supplied separately by an
external approval record and is never generated by this tree.

The VAST validation runner must first validate an external
`owner-review-scope.json` and approval with `dependency_approval.py`.  It then
passes only the resulting approval SHA-256, exact scope SHA-256, expected
checkout HEAD, `VALIDATED`, and `NO_UPLOAD` values to
`dia_1_6b_dump_reference.py`.  The dumper records that exact binding in
`manifest.json`; `dia_1_6b_validate_evidence.py` reopens the external files and
verifies the binding again.  Missing, stale, duplicated, path-overlapping,
placeholder-signed, or upload-enabled records fail closed before model or
checkpoint access.

Self-tests exercise a synthetic approval success and tampered binding
failures only.  They do not load model weights or run inference.

## Historical 2.6.0 dependency closure review (superseded)

The VAST evidence report is bound by SHA-256
`8ce645073916f8f1148ed572b476653fc602a2236fbeee90183b0b01b1c57cd8` and was
collected at checkout `87da78dc7709075d9dc23b797fc978b9c678c777`. It covers
the exact Linux x86_64 Python 3.12 closure: 26 packages, 159 native/bundled
files, and 50 publisher license entries, with no collection failure. The
external owner scope file is independently bound by SHA-256
`0d51a44f9494313d01b6cb0314d35215c48c311c0c071d8516a10617a2f224e3`.

The checked-in `license_gate_manifest.json` records each row's captured
license label and native/bundled counts. The review is factual only:
NumPy's zlib/NCSA notices, Torch's NOTICE and native third-party payloads,
setuptools' vendored license set, and MPL components remain visible for
owner/legal review. The setuptools vendor set visibly includes LGPL-3.0 and
MPL/other notice material. Dia source and weight licensing were not assessed because
source/model acquisition was explicitly `NONE`.

The historical `license_gate.py --self-test` command belongs to the old
2.6.0 manifest and is not a candidate verification.  The 2.13.0 candidate
must instead pass the lock check and the model-free dependency/owner-scope
self-tests; the protected manifest gate is expected to reject the stale
2.6.0 evidence until a fresh VAST collector and owner/legal review replace
it.  Publication stays `NO_UPLOAD`, and the historical report and scope are
not valid evidence for the 2.13.0 candidate.
