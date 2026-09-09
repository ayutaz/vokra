# CLAP HTSAT-fused reference contract

This directory is a VAST-only, reference-side lock for
`laion/clap-htsat-fused` at revision
`365dea6ef167def6676140ed93bbc43f84dabb28`. It does not contain weights and
does not enable the native CLAP binder. The model license has the explicit
2026-07-30 yousan commercial sign-off recorded for the CLAP row in
`docs/license-audit.md`; that sign-off is limited to the model license and
does not approve Python dependencies, native payloads, execution, parity, or
publication. No upload is permitted.

The pinned environment is Python 3.12 on Linux x86_64 with CPU-only Torch
(`torch==2.7.1` from the explicit PyTorch CPU index) and
`transformers==5.10.4`. The lock is a dependency record only; it is not a
permission to synchronize or acquire a checkpoint locally.
The locked Transformers wheel is SHA-256
`8c5b99b141b53619435a76629b0284f04d27ff46d788b463fc0ecb23b8ff130e`.

The Rust `vokra-convert --model clap` route is also fail-closed while this
evidence is incomplete. It returns an `INSPECTION_ONLY` usage error before
reading the input or touching the output, so arbitrary safetensors cannot be
published with CLAP provenance. A GGUF conversion requires the owner-reviewed
VAST checkpoint manifest, native preprocessing/forward contract, and
dependency/license closure described below.

The official processor must authenticate this exact HTSAT audio contract on
the disposable VAST host: 48 kHz, 10 seconds/480,000 samples, 1024 FFT and
window, 480 hop, 513 frequency bins, 64 feature size, 1,000 frames,
50–14,000 Hz, right-side `repeatpad`, and `truncation="fusion"` with no
attention mask. The inspector records the complete returned feature-extractor
contract and fails closed on drift; it does not reimplement the mel path.

The inspector also authenticates the pinned `config.json` topology before it
records any tensors: CLAP projection width 512, HTSAT audio tower
(`spec_size=256`, patch embedding width 96, depths `[2,2,6,2]`, heads
`[4,8,16,32]`, fusion enabled), and the 12-layer 768-wide RoBERTa-style text
tower. These are config facts, not invented per-tensor shapes. Every observed
state-dict entry is then recorded verbatim as a name, role, shape, and dtype;
the only accepted roles are `audio_tower`, `text_tower`, `audio_projection`,
`text_projection`, and the two scalar entries `logit_scale_a` and
`logit_scale_t`. Unknown names, missing roles, malformed shapes, or
non-scalar contrastive temperatures abort the inspector.

The state-dict inspector assigns every official tensor to the observed
`audio_tower`, `text_tower`, `audio_projection`, or `text_projection` role
(plus exactly `logit_scale_a` and `logit_scale_t` contrastive scalar entries),
and records exact shape/dtype/name rows.
Unknown names or a missing required role fail closed. This is an authenticated
inspection artifact, not a runtime implementation or a numerical parity
result. A VAST run must use:

```sh
UV_CACHE_DIR=/private/tmp/vokra-clap-uv-cache \
  uv run --no-sync --project tools/parity/clap \
  python tools/parity/clap_dump_reference.py --self-test
UV_CACHE_DIR=/private/tmp/vokra-clap-uv-cache \
  uv run --no-sync --project tools/parity/clap \
  python tools/parity/clap/license_gate.py --self-test
UV_CACHE_DIR=/private/tmp/vokra-clap-uv-cache \
  uv run --no-project --offline --python 3.12 python \
  tools/parity/clap_source_only_reference.py --self-test
UV_CACHE_DIR=/private/tmp/vokra-clap-uv-cache \
  uv run --no-project --offline --python 3.12 python \
  tools/parity/clap_expected_manifest.py --self-test
```

Before any owner-approved real-weight inspection, the VAST worker supports an
independent model-free closure:

```sh
VOKRA_PUBLISH_ON_VAST=1 \
  scripts/publish/vast-ai/run-clap-htsat-fused-validation.sh \
  --model-free --expected-head <clean-40-hex> \
  --work-dir /tmp/vokra-clap-model-free
```

This path resolves only `config.json` and `preprocessor_config.json` from the
pinned revision. `HfApi.model_info` (using the public `card_data` property),
`list_repo_files`, and an expanded `list_repo_tree` first authenticate the
resolved commit and record CardData license metadata separately from any
repository `LICENSE` file. The tree packet binds each selected metadata file's
remote size/blob or LFS identity to its materialized local bytes. The
allowlisted snapshot is copied into regular files before the audit, so cache
symlinks cannot satisfy the metadata gate. It then runs the frozen
Transformers config/feature-extractor API and records lock/dependency/license
facts. Dependency license evidence remains explicitly `PENDING`; it is not an
approval. If an installed distribution has no bundled license file, the
collector may fetch only its exact, hash-and-size-pinned PyPI sdist. The
bounded in-memory archive inspection hashes only safe LICENSE/COPYING/NOTICE
members and writes no archive payloads. Missing PEP-639
`License-Expression` is an owner-review flag when legacy metadata, classifiers,
or license bytes exist; it is not treated as a factual collection failure.
Dependency/native-payload disposition remains `PENDING_OWNER_REVIEW` and
publication remains `NO_UPLOAD`. Its evidence explicitly says
`weights=NOT_ACQUIRED`, `model_load=NOT_PERFORMED`, and
`publication=NO_UPLOAD`. The real-weight path remains approval-gated and
cannot be authorized by model-free evidence.

The exact model-free VAST result is tracked as an owner-review candidate in
`owner_review_candidate.json`. Its `owner_review_candidate.py` validator uses
only the Python standard library and binds the three immutable VAST SHA-256
values (model-free audit, dependency inventory, and summary), the pinned
upstream revision, and the fail-closed disposition. The candidate's own
canonical payload digest is checked as well, so schema, evidence hashes, or
status changes are rejected. It deliberately contains no approval digest:
`candidate_status=PENDING_OWNER_REVIEW`, `runtime_status=BLOCKED`,
`weights=NOT_ACQUIRED`, `model_load=NOT_PERFORMED`, and `publication=NO_UPLOAD`
remain required. Validating this file is evidence packaging only; it does
not approve a real-weight inspection, native execution, parity, or upload.
The normal validator invocation prints the pending state and exits 2 by
design, so a pending candidate cannot be mistaken for approval by a shell
pipeline.

Run the dependency-free candidate self-test offline with:

```sh
UV_CACHE_DIR=/private/tmp/vokra-clap-candidate-uv-cache \
  uv run --no-project --offline --python 3.12 python \
  tools/parity/clap/owner_review_candidate.py --self-test
```

The same model-free worker emits `dependency-license-inventory.json`. It
enumerates every non-virtual package in the single frozen Linux x86_64 uv
resolution (`platform_machine == 'x86_64'` and `sys_platform == 'linux'`)
with exact source and artifact hashes, then records installed METADATA license
fields/classifiers, bundled LICENSE/COPYING/NOTICE bytes and hashes, and native
payload files and hashes. RECORD entries such as console scripts and manpages
are non-target metadata and are ignored; only LICENSE/COPYING/NOTICE and native
payload candidates are resolved, prefix-checked, and hashed. A candidate that
escapes the installed prefix remains an explicit fail-closed finding.
Missing, multiple, or unknown entries remain explicit fail-closed findings;
the dependency audit status stays `PENDING_VAST_AUDIT` and never becomes
`COMPLETE` from this inventory alone.

The raw release preprocessor JSON is validated against the complete
`PREPROCESSOR_CONTRACT`. Transformers 5.10.4 does not serialize
`processor_class` through `ClapFeatureExtractor.to_dict()`, so the serializer
round-trip contract intentionally excludes only that key; the raw
`processor_class=ClapProcessor` is separately bound to the official
`ClapProcessor` source fact in the evidence.

The model-free audit emits an independent `source_contract`, but it remains
`PENDING_VAST_WHEEL_BINDING` until the exact locked Transformers 5.10.4 wheel
(`8c5b...`) is downloaded and its archive members, `RECORD`, and installed
files agree. Only then may it become `AUTHENTICATED_LOCKED_WHEEL_SOURCE`.
The bound surface includes the official `ClapFeatureExtractor` entrypoints,
`ClapProcessor`/`ProcessorMixin`, and the `RobertaTokenizer` path; model HF
repository/revision and Transformers wheel identity are separate fields.

The model-free worker then runs a source-only reference stage against the
materialized config/preprocessor/tokenizer files. It uses deterministic PCM
and fixed text and emits F32 audio features plus IDs/masks atomically. It never
loads `ClapModel` weights or runs a forward. A separate meta-device stage
constructs `ClapModel(config)` only to emit
`SOURCE_DERIVED_EXPECTED_MANIFEST`; this must never be confused with an
observed checkpoint state-dict manifest. Both stages fail closed.

Only after owner approval and a clean disposable VAST checkout may the normal
reference command be run with a pinned local snapshot and an output directory.
The resulting metadata remains `INSPECTION_ONLY` until independent review and
real CPU/Metal parity are completed.
