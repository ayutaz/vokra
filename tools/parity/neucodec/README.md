# NeuCodec official decoder parity

This isolated Python 3.12 uv project regenerates the independent official
token-to-waveform fixtures for both public Vokra NeuCodec GGUFs. It is not
part of the Rust runtime and does not alter the zero-dependency `Cargo.lock`.

The dumper verifies the official `neuphonic/neucodec` source commit, the exact
public GGUF SHA-256, `torchtune==0.3.1`, the SHA-256 of torchtune's official
RoPE implementation file, and `vector-quantize-pytorch==1.17.8`. It restores
the GGUF decoder weights into the official `CodecDecoderVocos` modules and
calls their FSQ and forward methods. The base GGUF's older normalized tensor
namespace and the distill GGUF's pass-through namespace are handled as two
explicit, fail-closed layouts.

Run on VAST from the repository root. The base artifact is larger than 2 GB
and must not be processed on the maintainer Mac.

```sh
uv sync --python 3.12 --project tools/parity/neucodec --frozen
uv run --python 3.12 --project tools/parity/neucodec --frozen \
  python tools/parity/neucodec/dump_reference.py \
  --source /path/to/neuphonic-neucodec \
  --gguf /path/to/model.gguf \
  --codes crates/vokra-models/tests/fixtures/neucodec/codes.u32le \
  --output /tmp/neucodec-reference
```
