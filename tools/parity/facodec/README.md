# NaturalSpeech 3 FACodec V2 real-checkpoint parity

This isolated Python 3.12 project generates an independent reference from the
official Amphion implementation at commit
`26f6883110181f1dbfe95c70a7c7dbaf4de5f42a`. It imports that checkout directly;
it does not import Vokra or mirror the forward pass.

Run only through `scripts/publish/vast-ai/run-facodec-parity.sh`. The worker
pins and verifies the official encoder/decoder files and the public Vokra GGUF,
then produces deterministic PCM, encoder/prosody diagnostics, exact six-stream
codes, the 256-value speaker embedding, and decoded PCM. No upload or publish
operation exists in the worker.

The Rust consumer is gated by `VOKRA_FACODEC_GGUF` and
`VOKRA_FACODEC_PARITY_DIR`. Codes must match exactly. Speaker embedding and
waveform comparisons use the repository FP32 ceiling `max |delta| <= 0.01`;
failures must be investigated rather than widening the bound.

The VAST run proves native CPU parity and Apple-target compilation. Real Metal
execution requires a separate remote Apple Silicon run with
`VOKRA_FACODEC_BACKEND=metal`; do not run real checkpoints on the maintainer
Mac.
