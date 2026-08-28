# FocalCodec official reference fixtures

These fixtures were generated on VAST with
`tools/parity/focalcodec/dump_reference.py`.  The oracle imports the official
`lucadellalib/focalcodec` package at Git commit
`912b7f2c0cd43d54a8aed296bbcc925dec7d4ea3`; it never imports Vokra or mirrors
the Rust forward pass.  Every directory's `manifest.json` records the pinned
upstream checkpoint revision, checkpoint/config SHA-256, tensor shapes, file
hashes, and execution environment.

The real Vokra GGUFs are intentionally not committed.  The fixed public files
used for the 2026-08-25 CPU run were:

| Variant | Vokra HF revision | File | SHA-256 |
| --- | --- | --- | --- |
| 50 Hz | `f9b5504c2e4fd7c4545e4b1a1344968b54f81813` | `focalcodec-50hz.gguf` | `3d19613193fe8cd4f3725209fa83c278e33d8b1e96fde43594b6c4328cf18d93` |
| 25 Hz | `346b834d7399b5276419c57683cef235b2c84e0f` | `model.gguf` | `1b11f8deb5fb0447b3f3b6a8cbdacbdb43e2aeb02604aff93bbfe1c8c4c57be6` |
| 12.5 Hz | `213e11c0105a71d6ea3f0883ab7e1f7509cf4ce2` | `model.gguf` | `d17de845cd25ec434d05df56e6befca0c992cb3c072d1d76c97285371c39e4cb` |

`parity_focalcodec_real` requires exact BSQ token equality and uses the project
FP32 maximum absolute waveform bound `0.01`.  The measured CPU results were:

| Variant | Tokens | Samples | Max absolute error | RMSE |
| --- | ---: | ---: | ---: | ---: |
| 50 Hz | 9 | 2880 | `1.169741154e-6` | `2.094736810e-7` |
| 25 Hz | 5 | 3200 | `1.396238804e-5` | `2.263951977e-6` |
| 12.5 Hz | 3 | 3840 | `6.938353181e-7` | `9.222271064e-8` |

Example VAST invocation:

```sh
VOKRA_FOCALCODEC_GGUF=/root/models/focalcodec-50hz.gguf \
VOKRA_FOCALCODEC_PARITY_DIR="$PWD/crates/vokra-models/tests/fixtures/focalcodec/50hz" \
  CARGO_BUILD_JOBS=1 cargo test -p vokra-models \
  --test parity_focalcodec_real -- --nocapture
```

For Metal, add `VOKRA_FOCALCODEC_BACKEND=metal` and run the same test on an
Apple-silicon host built with the crate's Metal feature.  No backend path may
silently fall back to CPU.
