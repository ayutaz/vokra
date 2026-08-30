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

## 2026-08-30 authenticated CPU lock

The pinned official implementation completed the real-weight CPU parity gate
on VAST. The prepared checkpoint SHA-256 is
`cda8d7dd7cad2a0361b6946c42342b85ef7b0a8d672b99631dc75b4c3123dbc5`, the
GGUF SHA-256 is
`abf6f0ee8c028e7c79955f68d841d4445fa9664a2e87ff26c80a37a3b4a3561e`, and
the reference-manifest SHA-256 is
`7a37e36e56c90370390c741bac421e211834116f14c4d0305a84f8b87552dd1b`.
The 49-frame gate recorded maximum absolute errors of `2.918243408e-4`
(frontend), `1.520276070e-3` (encoder), and `1.450002193e-3` (logits); all
five emitted token IDs matched exactly. These are independent official VAST
CPU results, not a self-authored mirror result.

Apple CPU repetition and Metal execution remain pending and must run on the
authenticated Apple/Scaleway worker. The packet remains remote-only at
`/root/scratchpad/apple-transfer-568dc192` on VAST (4.9 GB / 33 files), to be
transferred directly VAST-to-Scaleway and destroyed from VAST after the
evidence handoff. No Hugging Face upload was performed.
