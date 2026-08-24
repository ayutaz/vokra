# FocalCodec real-checkpoint parity

This isolated Python 3.12 uv project generates independent reference fixtures
for the three public non-causal FocalCodec checkpoints supported by Vokra.  It
imports the official `lucadellalib/focalcodec` implementation at commit
`912b7f2c0cd43d54a8aed296bbcc925dec7d4ea3`; it does not import Vokra or mirror
the forward pass.

The dumper fails closed unless the installed package has matching PEP 610 Git
provenance and the selected HF `model.safetensors` matches both its immutable
revision and the audited SHA-256 embedded in `dump_reference.py`.

Run on VAST from the repository root:

```sh
uv sync --python 3.12 --project tools/parity/focalcodec --locked
uv run --python 3.12 --project tools/parity/focalcodec \
  tools/parity/focalcodec/dump_reference.py \
  --variant 50hz \
  --output /root/focalcodec-reference/50hz
```

Repeat with `25hz` and `12_5hz`.  Each fixture contains deterministic PCM,
official WavLM features, normalized compressor latents, exact BSQ tokens and
codes, decompressor features, decoder PCM, and a manifest recording source,
checkpoint and execution-environment provenance.  Only the small fixtures are
committed; safetensors and GGUF checkpoints remain on VAST.

Run the Rust consumer with the matching Vokra GGUF:

```sh
VOKRA_FOCALCODEC_GGUF=/root/models/focalcodec-50hz.gguf \
VOKRA_FOCALCODEC_PARITY_DIR=/root/focalcodec-reference/50hz \
  CARGO_BUILD_JOBS=1 cargo test -p vokra-models \
  --test parity_focalcodec_real -- --nocapture
```

The encoder token sequence must be exactly equal.  The decoded waveform uses
the project FP32 reference bound `max |delta| <= 0.01`; a failure must be
investigated rather than widening the bound.
