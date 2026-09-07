# CLAP HTSAT-fused reference contract

This directory is a VAST-only, reference-side lock for
`laion/clap-htsat-fused` at revision
`365dea6ef167def6676140ed93bbc43f84dabb28`. It does not contain weights and
does not enable the native CLAP binder. The source/model license remains
`OWNER_REVIEW_PENDING`; the reported HF `apache-2.0` metadata is evidence, not
an owner approval. No upload is permitted.

The pinned environment is Python 3.12 on Linux x86_64 with CPU-only Torch
(`torch==2.7.1` from the explicit PyTorch CPU index) and
`transformers==5.10.4`. The lock is a dependency record only; it is not a
permission to synchronize or acquire a checkpoint locally.
The locked Transformers wheel is SHA-256
`8c5b99b141b53619435a76629b0284f04d27ff46d788b463fc0ecb23b8ff130e`.

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
approval. Its evidence explicitly says
`weights=NOT_ACQUIRED`, `model_load=NOT_PERFORMED`, and
`publication=NO_UPLOAD`. The real-weight path remains approval-gated and
cannot be authorized by model-free evidence.

Only after owner approval and a clean disposable VAST checkout may the normal
reference command be run with a pinned local snapshot and an output directory.
The resulting metadata remains `INSPECTION_ONLY` until independent review and
real CPU/Metal parity are completed.
