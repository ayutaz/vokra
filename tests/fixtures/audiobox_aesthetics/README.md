# Audiobox Aesthetics official parity fixture

The numerical fixture is generated only from Meta's official
`AesMultiOutput` package at source revision
`2618e9d451b456e9328b39495b5e6234678aa550` and the immutable
`facebook/audiobox-aesthetics` snapshot revision
`9b1dd8e5df9af7216e836a98974fe3b82c56ded6`.

No numerical files are committed yet. Generate them on VAST after cloning the
official source at the exact revision and downloading the exact HF snapshot:

```text
uv run --project tools/parity --python 3.12 python \
  tools/parity/audiobox_aesthetics_dump_reference.py \
  --source-tree /path/to/audiobox-aesthetics \
  --checkpoint-dir /path/to/facebook-audiobox-aesthetics-snapshot \
  --wav /path/to/mono-pcm16-16khz.wav \
  --output-dir /path/to/audiobox-reference
```

The dumper aborts if the official package cannot be imported; it has no local
WavLM mirror. Forward hooks record the official stem, projection, encoder
input, all 12 encoder layers, normalized per-axis embeddings, raw chunk scores,
inverse-transformed chunk scores and final duration-weighted CE/CU/PC/PQ
scores. `manifest.json` records revisions, source/checkpoint/input hashes,
versions, shapes and target transforms.

Run the Rust consumer with `VOKRA_AUDIOBOX_GGUF`,
`VOKRA_AUDIOBOX_REFERENCE_DIR` and optionally
`VOKRA_AUDIOBOX_METAL_PARITY=1`. The reference maximum-absolute-error gate is
the project FP32 default `0.01`, registered before the first Vokra execution;
it must not be widened in response to a failure without diagnosing the
worst element and the responsible stage.
