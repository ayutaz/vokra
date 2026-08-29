# tools/parity/melodyflow_t24_30secs

Offline sidecar for **facebook/melodyflow-t24-30secs** (Meta AudioCraft
MelodyFlow T24 30secs, CC-BY-NC-4.0, 4,088,594,620-byte fixed bundle = 1 B flow-matching
DiT transformer + 48 kHz RVQ codec + T5-base text encoder) — bridges the
upstream torch-pickle bundle to the flat safetensors the Rust converter
(`crates/vokra-convert/src/models/melodyflow_t24_30secs.rs`) consumes.

Sibling of:

- `../magnet_small_10secs/prepare_checkpoint.py` (1,076,848,566-byte,
  300 M / 10 sec
  — non-autoregressive masked-LM decoding, entirely different sampler)
- `../magnet_medium_30secs/prepare_checkpoint.py` (3,913,673,878-byte,
  1.5B / 30 sec
  — non-autoregressive masked-LM decoding, same-scale sibling in the
  Meta music-gen catalog)
- a future `../jasco_400m_chords_drums/` (not yet written — only the
  wave-D tickets `docs/tickets/coverage-audit-2026-08-03/wave-d/jasco-*.md`
  exist today; ~1.6 GB, joint audio-symbolic conditioning — same op
  family (flow-matching) but different conditioning stack from
  MelodyFlow's dual text + audio prefix for editing)
- `../musicgen_medium_prepare_checkpoint.py` (11.4 GB, vast.ai required
  — AR-over-EnCodec sibling family, entirely different decoder loop
  from MelodyFlow flow-matching)

## What this directory contains

- `prepare_checkpoint.py` — the VAST-only torch-pickle → flat safetensors
  bridge. Loads only the exact authenticated `state_dict.bin` under
  `--input-dir` only via `torch.load(..., weights_only=True)`; failures
  stop without an unsafe retry,
  dedupes tied tensors (data_ptr collision → clone + audit trail),
  strips non-float training scaffold (`.num_batches_tracked` /
  `.total_ops` / `.total_params`), rejects unexpected non-float
  dtypes loudly (FR-EX-08). Emits `<output>` + `<output>.sha256` +
  `<output>.shared_pairs.json`. See the script's module docstring
  for the honest write-up.
- `pyproject.toml` + `uv.lock` — exact Python 3.12 CPU-only reference
  closure, including the pinned `huggingface-hub==1.27.0` acquisition
  closure, with lock/package/license digests and a fail-closed audit. The
  worker allows only `README.md`, `compression_state_dict.bin`, and the
  exact `state_dict.bin` payload.
- `.python-version` — `3.12`.

## VAST-only workflow

The aggregate checkpoint exceeds the repository's 2 GB threshold. Do not
download it, create a model work directory, run `uv sync`, load pickle,
or run conversion on the maintainer Mac. On an authorized Linux x86_64
VAST worker, first run `run-melodyflow-t24-30secs-inspection.sh` or
`run-melodyflow-t24-30secs-validation.sh`; each invokes the stdlib-only
`audiocraft_safe_gate.py` with `uv run --no-project` before any future
sync/download/work step. The fixed model revision is
`77bcfce24371bf29a06152c72169162c6f2791de`; the authenticated source is
`facebook/MelodyFlow@9d0d223e9a63bbb8c20b9f57c5afcb4de297e6da`.
Both workers intentionally exit 2 until external owner clearance is supplied,
and perform no acquisition.

Publication is a separate permission and is currently prohibited by the
worker (`NO_UPLOAD`). The weight is CC-BY-NC-4.0 / Research-only and the
owner's license sign-off, training-data review, and publication gate must
be handled independently.

## What the script does NOT do

- **Runtime forward**. This is a converter-side bridge — the
  `flow_editing_inversion` + `t24_transformer` runtime ops are a
  follow-up wave (FR-OP-86 anchor). Loud-partial per RMVPE /
  Charsiu / MOSS-Audio-Tokenizer / MioCodec / sibling MAGNeT
  precedent. The core DiT forward can reuse `vokra_ops::flow_sampler`
  from M3-05 for the ODE integrator, but the editing-specific
  inversion path and the 48 kHz RVQ codec bundle need explicit
  binder ADR judgement — the phase task explicitly punts DiT sampler
  forward to a future wave.
- **License override**. The default `cc-by-nc-4.0` SPDX resolves to
  `LicenseClass::NonCommercial` (T4 fail-closed). A caller who trained
  on a different corpus (or holds the weight under a distinct SPDX id)
  overrides at the outer `--license <spdx>` boundary in `vokra-cli
  convert`.
- **Real-weight parity**. This land is converter code only. A future
  wave (once §3.1 sign-off is granted) will add a
  `parity_melodyflow.rs` test that dumps upstream reference outputs
  and byte-compares the first ODE step / mel frame. Same loud-partial
  defer pattern as RMVPE / DeepFilterNet3 / Charsiu / sibling MAGNeT.

## Owner critical path (post-land)

- **License/source gate**: the weight is CC-BY-NC-4.0 / Research-only,
  but this sidecar remains blocked until the owner confirms the
  versioned primary-source evidence and the fixed source/HF revisions.
- **training-data audit** (medium-high risk): Meta MusicGen family
  shares training corpus with Suno / Udio litigation cloud, and the
  MelodyFlow **editing** use-case (existing audio rewritten under a
  new text prompt) is a direct target of the copyright-infringement
  argument in those suits. Legal review before any separately authorized
  distribution (higher
  scrutiny than text-to-music sibling releases).
- **runtime binder ADR** (FR-OP-86): decide whether MelodyFlow's
  editing-specific ODE inversion path and the 48 kHz RVQ codec bundle
  get first-class op paths or stay as a loud-partial defer. Owner
  judgement. The `vokra_ops::flow_sampler` from M3-05 is reusable for
  the core DiT forward; the incremental scope is the inversion path +
  the codec bundle.
- **VAST-only operational gate**: the aggregate artefact is above the
  repository threshold. Conversion and validation remain blocked until
  an authenticated fixed revision, dependency clearance, and an
  authorized VAST worker are available.
- **parity**: no real-weight reference or runtime parity has run; this
  remains an explicit blocker after the source and license gates clear.
