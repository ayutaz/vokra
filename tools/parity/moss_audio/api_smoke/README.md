# MOSS-Audio model-free API smoke

This directory is a VAST-only, model-free compatibility probe for the pinned
OpenMOSS source revision and the 4B/8B metadata snapshots. The project lock is
Python 3.12-only and pins the repository security-floor route
`transformers==5.10.4` and the complete official-reference dependency closure;
it is deliberately separate from the historical MOSS project lock, which is
retained for its existing license/closure gates. The official reference path
uses this patched project only after the model-free smoke evidence passes;
model compatibility and numerical parity still require the real VAST run.

The probe downloads no checkpoint shards. It authenticates the source checkout,
the complete metadata-only snapshot, and the locked package artifacts, then
imports the official configuration, model, and processor classes. It constructs
only the configuration and processor; model instantiation and checkpoint
loading are explicitly not performed. An incompatible API produces
`BLOCKED_INCOMPATIBLE_API` evidence and never becomes a compatibility PASS.
The metadata gate verifies the hash-authenticated config plus selected nested
`language_config` structural axes (including variant-specific hidden and
intermediate sizes), along with `model_type=moss_audio` and
`architectures=["MossAudioModel"]`; legitimate additional nested config keys
are retained, while root-level topology fields are rejected.

The model-free phase is executable before owner evidence is available:

```text
scripts/publish/vast-ai/run-moss-audio-api-smoke.sh --model-free --variant all \
  --expected-head <HEAD>
```

It stages only the exact source and metadata files, imports the official
configuration/model/processor classes, constructs config and processor objects,
and records `PASS_MODEL_FREE` while source/model/operator approvals remain
`PENDING_OWNER_APPROVAL`. This evidence is an API compatibility inspection,
not a parity result, and cannot satisfy the approval-bound full worker.
Source, metadata, and API incompatibilities are emitted as atomic structured
`BLOCKED_INCOMPATIBLE_API` evidence without overwriting an existing output;
infrastructure and explicit input precondition failures still stop normally.

Before the approval-bound full worker can install its project environment, it runs the
stdlib-only `--closure-only` preflight. That gate requires an owner approval
bound to the exact Vokra HEAD, project/lock bytes, package identities, and
source/4B/8B license-review evidence. The normal VAST result writes a digest
bound evidence file and summary. The metadata/reference evidence is transferred
VAST-to-Apple directly; only small logs are recovered locally. There is no
upload path.

Examples (on the approved VAST host):

```text
scripts/publish/vast-ai/run-moss-audio-api-smoke.sh --self-test
scripts/publish/vast-ai/run-moss-audio-api-smoke.sh --closure-only \
  --variant all --approval-evidence /path/approval.json --expected-head <HEAD>
scripts/publish/vast-ai/run-moss-audio-api-smoke.sh --variant all \
  --approval-evidence /path/approval.json --expected-head <HEAD>
```

The pinned identities in `api_smoke.py` must be regenerated from authenticated
primary-source evidence if upstream metadata changes; no topology or parity
bound is inferred by this smoke.

## Torch/TorchAudio security-floor assessment

As of 2026-09-29, Dependabot alerts #400 and #401 require `torch>=2.13.0`.
The official TorchAudio installation guidance states that the 2.11.0 line
uses the stable ABI and supports PyTorch 2.11 and all later versions,
including 2.13. The official CPU index therefore supports this compatible
security-floor pair, which is now pinned here as `torch==2.13.0` /
`torchaudio==2.11.0` (resolved Linux CPU distributions are
`2.13.0+cpu` / `2.11.0+cpu`). A Torch-only bump would still be invalid; the
TorchAudio ABI floor is updated together with the Torch security floor.

The model-free import/API smoke remains required before any real-weight
validation, and no model or checkpoint is acquired by this project. Primary
sources for the pair and ABI rule:

- <https://download.pytorch.org/whl/cpu/torch/>
- <https://download.pytorch.org/whl/cpu/torchaudio/>
- <https://docs.pytorch.org/audio/main/installation.html>
