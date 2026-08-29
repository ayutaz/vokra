# OmniASR-CTC-1B independent reference

This is a VAST-only reference environment.  `pyproject.toml` is intentionally
separate from the general parity tool project because the official Omnilingual
ASR release requires fairseq2 `0.5.2`.  The committed `uv.lock` must be
generated from this file on the pinned Linux VAST image before the validation
worker can run; a missing lock is a hard failure, never a fallback.

The dumper imports the official `omnilingual_asr.datasets.utils.audio` and
fairseq2 `Wav2Vec2AsrModel` implementations from these fixed source commits:

- `facebookresearch/omnilingual-asr@a7fb36017a46eee8953f76bd628c174d51aefeef`
- `facebookresearch/fairseq2@8ae890e1b4d3e36307d0ba5fb695f0fc4815ecca`

It emits only raw little-endian `f32`/`u32` artifacts and a strict JSON
manifest.  It does not emit or compare transcript text.  The worker refuses
to proceed until the lock and fixed raw source-file SHA-256 values are
recorded and reviewed.  The authenticated source boundary includes the
Omnilingual card/audio/config files, fairseq2 wav2vec2 factory/frontend/
feature-extractor/position-encoder and ASR model/factory, plus the upstream
Transformer encoder/layer/attention, normalization, and projection modules.
Every listed path is checked against a fixed raw SHA-256 pin; a missing path,
extra path, or byte drift is a hard failure rather than permission to record
whatever bytes a worker happens to find.  The Apple consumer verifies the
immutable packet and manifest digest only; it does not resolve Python
dependencies or fetch upstream sources.  The manifest pre-registers exact
`frontend_atol`, `encoder_atol`, and `logits_atol` values of `0.01`, plus exact
greedy token IDs; all three numeric surfaces are measured by the Rust test.

The authenticated GGUF/reference bundle is a remote artifact.  Keep it on
VAST and transfer it directly to an authenticated Apple/Scaleway worker (or
another explicitly authorized remote path); never pull the bundle to the
maintainer Mac.  Only small logs and JSON manifests may be returned locally.
