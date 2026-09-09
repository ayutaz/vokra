# tools/parity/magnet_medium_30secs

Offline sidecar for **facebook/magnet-medium-30secs** (Meta AudioCraft
MAGNeT Medium 30secs, CC-BY-NC-4.0, 3,913,673,878-byte fixed bundle;
1.5B params
non-autoregressive masked-LM transformer + bundled EnCodec 32 kHz codec
+ T5-base text encoder) — bridges the upstream torch-pickle bundle to
the flat safetensors the Rust converter
(`crates/vokra-convert/src/models/magnet_medium_30secs.rs`) consumes.

Sibling of:

- `../magnet_small_10secs/prepare_checkpoint.py` (1,076,848,566-byte fixed
  bundle, 300 M / 10 sec
  — same non-autoregressive masked-LM decoding op path, narrower
  hidden width, shorter span)
- `../musicgen_medium_prepare_checkpoint.py` (11.4 GB, vast.ai
  required — AR-over-EnCodec sibling family, entirely different
  decoder loop from MAGNeT non-autoregressive masked-LM)
- `../firered_asr_llm_l/prepare_checkpoint.py` (16.6 GB, vast.ai
  required)

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
VAST worker, first run `run-magnet-medium-30secs-inspection.sh` or
`run-magnet-medium-30secs-validation.sh`; each invokes the stdlib-only
`audiocraft_safe_gate.py` with `uv run --no-project` before any future
sync/download/work step. The fixed model revision is
`2559c5978450f62782cf1d17826d384fb93fb64b`; the authenticated source is
`facebookresearch/audiocraft@905371a779f608169353fe6ad42bb5fc10c5c9a8`.
Both workers intentionally exit 2 until external owner clearance is supplied,
and perform no acquisition.

Publication is a separate permission and is currently prohibited by the
worker (`NO_UPLOAD`). The weight is CC-BY-NC-4.0 / Research-only and the
owner's license sign-off, training-data review, and publication gate must
be handled independently.

## What the script does NOT do

- **Runtime forward**. This is a converter-side bridge — the
  `magnet_masked_decode` + `span_masking_scheduler` runtime ops are
  a follow-up wave (FR-OP-85 anchor). Loud-partial per RMVPE /
  Charsiu / MOSS-Audio-Tokenizer / MioCodec / sibling
  `magnet_small_10secs` precedent.
- **License override**. The default `cc-by-nc-4.0` SPDX resolves to
  `LicenseClass::NonCommercial` (T4 fail-closed). A caller who trained
  on a different corpus (or holds the weight under a distinct SPDX id)
  overrides at the outer `--license <spdx>` boundary in `vokra-cli
  convert`.
- **Real-weight parity**. This land is converter code only. A future
  wave (once §3.1 sign-off is granted) will add a `parity_magnet.rs`
  test that dumps upstream reference outputs and byte-compares the
  first token / mel frame. Same loud-partial defer pattern as
  RMVPE / DeepFilterNet3 / Charsiu / sibling `magnet_small_10secs`.

## Owner critical path (post-land)

- **License/source gate**: the weight is CC-BY-NC-4.0 / Research-only,
  but this sidecar remains blocked until the owner confirms the
  versioned primary-source evidence and the fixed source/HF revisions.
- **training-data audit** (medium-high risk): Meta MusicGen family
  shares training corpus with Suno / Udio litigation cloud. Legal
  review before any separately authorized distribution (same posture as
  sibling small); this staging worker remains `NO_UPLOAD`.
- **runtime binder ADR** (FR-OP-85): decide whether MAGNeT masked-LM
  parallel decoding gets a first-class op path or stays as a
  loud-partial defer. Owner judgement. The ADR outcome applies to
  BOTH small and medium variants — the op path is shared, only
  hparams differ.
- **parity**: no real-weight reference or runtime parity has run; this
  remains an explicit blocker after the source and license gates clear.
