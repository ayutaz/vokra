# YuE xcodec-mini validation staging

This is a dedicated Python 3.12 reference environment for the official
m-a-p/xcodec_mini_infer token decoder and Vocos 0.1.0 ISTFT decoder.
The lock contains 45 exact package rows (the reachable Vocos/encodec/CPU-Torch
plus official RVQ utility import closure); dependency review remains pending.
The CPU torch source is the official PyTorch CPU index.

Fixed identities:

- Public historical GGUF: vokra/yue-xcodec-mini@83c14a67ed792a0d5b3b61fff8ae35a04c6da8fa,
  yue-xcodec-mini.gguf, 1,810,001,760 bytes,
  SHA-256 60e21aa5335646080102196454d7ffad5e012467d6f5eb9b776bf07d666b02bc,
  manifest SHA-256 cc0a5e9a5a6f1cfbd93b1869bbcb70744814bd8c855d173949abbf6b6cc08f15.
- Upstream source: m-a-p/xcodec_mini_infer@fe781a67815ab47b4a3a5fce1e8d0a692da7e4e5.
- Codec checkpoint: final_ckpt/ckpt_00360000.pth, SHA-256
  c8c379ea2d3cbde1c8ba1b9717975220e79ba3f556bb161766fd5e4585dcd59c.
- Vocos decoder checkpoint: decoders/decoder_151000.pth, SHA-256
  8af97a29d3483f9d4a3755992837501bd7d6caa1a69382ed16e64039e0ea0998.
- Semantic checkpoint is recorded as a public composite identity but is not
  executed by this oracle.

The oracle imports the authenticated upstream quantization implementation and
Vocos. It generates five deterministic frames and writes exactly
manifest.json, codes.u32le, features.f32le, backbone.f32le, and
waveform.f32le. Pickle loading is weights_only=True only; no unrestricted
fallback exists.

The gate runs before host/tooling/scratch/cache/sync/network/Cargo work. It is
currently expected to exit 2 because every dependency/component review is
pending and the authenticated HF tree has no source LICENSE object. The Linux
x86_64 torch CPU wheel had no size in the official lock response; its exact
175,833,687-byte Content-Length was obtained by an HTTPS HEAD request to the
official `download-r2.pytorch.org` artifact on 2026-08-29 and is now bound in
the lock/package scope. The six source-role bytes/SHA/git-blob identities and
the README card identity are fixed; they are not license evidence for the
absent code LICENSE. The RepCodec PCM-encode path is an independently
non-executed MIT/CC-BY-NC boundary; this staging executes token decode plus
Vocos 151k only once the token-decode closure is separately approved.

The VAST worker is no-upload and emits portable quoted Apple arguments.
Apple validation requires explicit GGUF/reference hashes, an exact regular
reference set, Darwin arm64/Metal, and separate strict CPU/Metal evidence.
Numeric results remain MEASURED_NOT_GATED with bounds unset. Do not run
model acquisition, sync, Cargo, VAST, Apple, conversion, or upload locally.
