# FunCodec official decoder parity

This isolated Python 3.12 project generates a token-to-waveform reference from
the exact official Alibaba DAMO FunCodec source and checkpoint. It is offline
tooling only and does not enter the zero-dependency Rust runtime.

The dumper verifies the source commit and hashes of the official quantizer,
SEANet, convolution and LSTM files, plus the immutable upstream `config.yaml`
and `model.pth`. It instantiates the official `CostumeQuantizer` and
`SEANetDecoder`, loads their checkpoint state, and calls their `decode` and
`forward` methods. It does not import Vokra or mirror the model equations.

Run this on VAST, not on the maintainer Mac:

```sh
uv sync --project tools/parity/funcodec --frozen --python 3.12
uv run --project tools/parity/funcodec --frozen --python 3.12 \
  python tools/parity/funcodec/dump_reference.py \
  --source /path/to/modelscope-FunCodec \
  --checkpoint /path/to/model.pth \
  --config /path/to/config.yaml \
  --output /path/to/reference
```

The corresponding Rust comparison consumes `codes.u32le` and
`decoded_pcm.f32` from that output through
`VOKRA_FUNCODEC_REFERENCE_DIR`. The first real CPU run and Apple Metal run
remain required evidence; an unset environment variable is a documented skip,
not a pass.
