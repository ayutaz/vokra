# X-Codec2 official parity oracle

> **Publication boundary (2026-10-04): `NO_UPLOAD`.** The primary-source
> bytes and production-gate packet required for a replacement artifact are
> still pending in the open draft PR #152 candidate (`b0add994b4d0350c338e60cb29b8165d8145fe32`).
> That candidate is not merged and must not be described as released. The
> audited GGUF hash below identifies an input only; it does not authorize
> conversion, publication, or a fresh parity/completion claim.

`dump_reference.py` imports the official `xcodec2==0.1.5` PyPI package,
verifies the installed decoder source and the audited public GGUF SHA-256,
restores the official `CodecDecoderVocos` modules, and calls their FSQ plus
decoder forward. It does not import Vokra and does not mirror the Rust
equations.

The public GGUF is 3,291,064,672 bytes, so dependency installation, reference
generation, and Rust consumer execution belong on VAST. The committed fixture
contains only codes and the official output:

```bash
cd tools/parity/xcodec2
uv sync --frozen --python 3.12
uv run --frozen --python 3.12 python dump_reference.py \
  --gguf /path/to/vokra-xcodec2/model.gguf \
  --codes /path/to/codes.u32le \
  --output /path/to/reference
```

The oracle pins `xcodec2==0.1.5`, `vector-quantize-pytorch==1.17.8`,
`torchtune==0.3.1`, the official decoder source hashes, and public GGUF
SHA-256 `7ab4b94006068226b0741930081f7e149316e045511c1cddb94769e7f598698e`.
