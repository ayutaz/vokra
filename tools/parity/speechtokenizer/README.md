# SpeechTokenizer official decoder parity

This isolated Python 3.12 project generates a token-to-waveform reference from
the exact official Fudan/OpenMOSS SpeechTokenizer source and checkpoint. It is
offline tooling only and does not enter the zero-dependency Rust runtime.

The dumper verifies the source commit and every inference-import file, plus the
immutable upstream `config.json` and `SpeechTokenizer.pt`. It instantiates the
official `SpeechTokenizer`, strictly loads the checkpoint and calls its public
`decode` method. It does not import Vokra or mirror the model equations.

Run this on VAST, not on the maintainer Mac:

```sh
uv sync --project tools/parity/speechtokenizer --frozen --python 3.12
uv run --project tools/parity/speechtokenizer --frozen --python 3.12 \
  python tools/parity/speechtokenizer/dump_reference.py \
  --source /path/to/SpeechTokenizer \
  --checkpoint /path/to/SpeechTokenizer.pt \
  --config /path/to/config.json \
  --output /path/to/reference
```

The Rust comparison consumes `codes.u32le` and `decoded_pcm.f32` through
`VOKRA_SPEECHTOKENIZER_REFERENCE_DIR`. The first real CPU run and Apple Metal
run remain required evidence; an unset variable is a documented skip, not a
pass.
