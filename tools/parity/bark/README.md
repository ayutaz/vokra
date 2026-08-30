# Bark official-reference parity

This directory is a VAST-only independent oracle for the public `vokra/bark`
and `vokra/bark-small` GGUFs. It imports the released `BarkModel` directly from
locked Transformers 5.5.0 and loads the exact immutable Suno checkpoint. It
does not import Vokra or reproduce Vokra's Rust graph in Python.

The no-upload worker is:

```sh
bash scripts/publish/vast-ai/run-bark-validation.sh
```

Before any checkpoint is acquired, an authorized VAST run performs the
model-free dependency audit. It compares the installed distribution multiset
with the exact `uv.lock`, records package license metadata and publisher
LICENSE/NOTICE hashes, streams hashes for bundled native files, and records
ELF `DT_NEEDED` entries after a four-byte magic check. The audit never imports
Bark or model code and never invokes Cargo. The standalone wrapper is:

```sh
bash scripts/publish/vast-ai/audit-bark-dependencies.sh --output /absolute/path/audit.json
```

The Python environment must already have been synchronized by a separately
authorized, named VAST job; the wrapper uses `uv run --frozen --no-sync` and
does not sync. It fetches only the exact `LICENSE` paths for the pinned
`suno/bark-small` and `suno/bark` revisions, retaining URL and byte/hash
evidence. The existing contract has no pinned Bark source-code LICENSE
revision, so the audit reports that factual blocker and leaves the tracked
gate manifest unresolved.

The operational order is intentionally non-circular:

1. A separately authorized named VAST job performs the frozen Python sync.
2. `audit-bark-dependencies.sh` runs model-free against that environment.
3. The factual audit is reviewed and any required owner approval is obtained;
   the current source-revision blocker and unresolved manifest rows stay open.
4. Only after that approval can the production worker pass its pre-sync gate,
   run its own frozen sync and audit, and then acquire checkpoints or invoke
   Cargo.

It generates four greedy semantic tokens from fixed caller-visible text token
IDs, runs the official coarse/fine schedule, and records the final frame-major
eight-codebook packet plus official 24 kHz PCM. The Rust gate first compares
generated codes exactly, then decodes the official packet independently and
applies the standard FP32 `max_abs <= 0.01` ceiling to PCM. This separation
distinguishes LM/schedule drift from codec drift.

The worker has no upload or push option. Pull its `logs/` and `reference/`
directories before destroying the VAST instance. Metal execution is a separate
remote Apple Silicon run using the same GGUF and reference; do not execute real
Bark models on the maintainer Mac.
