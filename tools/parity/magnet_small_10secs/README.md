# tools/parity/magnet_small_10secs

Offline sidecar for **facebook/magnet-small-10secs** (Meta AudioCraft
MAGNeT Small 10secs, CC-BY-NC-4.0, 1,076,848,566-byte fixed bundle;
300M params non-autoregressive
masked-LM transformer + bundled EnCodec 32 kHz codec + T5-base text
encoder) — bridges the upstream torch-pickle bundle to the flat
safetensors the Rust converter
(`crates/vokra-convert/src/models/magnet_small_10secs.rs`) consumes.

Sibling of:

- `../firered_asr_llm_l/prepare_checkpoint.py` (16.6 GB, vast.ai required)
- `../higgs_audio_v3_tts_4b/prepare_checkpoint.py` (4B params, vast.ai required)
- `../musicgen_medium_prepare_checkpoint.py` (11.4 GB, vast.ai required)

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

The fixed bundle is 1,076,848,566 bytes (below the repository's 2 GB
threshold), but this route remains VAST-only by explicit workflow policy. Do not
download it, create a model work directory, run `uv sync`, load pickle,
or run conversion on the maintainer Mac. On an authorized Linux x86_64
VAST worker, first run `run-magnet-small-10secs-inspection.sh` or
`run-magnet-small-10secs-validation.sh`; each invokes the stdlib-only
`audiocraft_safe_gate.py` with `uv run --no-project` before any future
sync/download/work step. The fixed model revision is
`2c9084771bd2e83c5c7e36303e24550da30da8e0`; the authenticated source is
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
  Charsiu / MOSS-Audio-Tokenizer / MioCodec precedent.
- **License override**. The default `cc-by-nc-4.0` SPDX resolves to
  `LicenseClass::NonCommercial` (T4 fail-closed). A caller who trained
  on a different corpus (or holds the weight under a distinct SPDX id)
  overrides at the outer `--license <spdx>` boundary in `vokra-cli
  convert`.
- **Real-weight parity**. This land is converter code only. A future
  wave (once §3.1 sign-off is granted) will add a `parity_magnet.rs`
  test that dumps upstream reference outputs and byte-compares the
  first token / mel frame. Same loud-partial defer pattern as
  RMVPE / DeepFilterNet3 / Charsiu.

## Owner critical path (post-land)

- **License/source gate**: the weight is CC-BY-NC-4.0 / Research-only,
  but this sidecar remains blocked until the owner confirms the
  versioned primary-source evidence and the fixed source/HF revisions.
- **training-data audit** (medium-high risk): Meta MusicGen family
  shares training corpus with Suno / Udio litigation cloud. Legal
  review before any separately authorized distribution; this staging worker
  remains `NO_UPLOAD`.
- **runtime binder ADR** (FR-OP-85): decide whether MAGNeT masked-LM
  parallel decoding gets a first-class op path or stays as a
  loud-partial defer. Owner judgement.
- **parity**: no real-weight reference or runtime parity has run; this
  remains an explicit blocker after the source and license gates clear.
