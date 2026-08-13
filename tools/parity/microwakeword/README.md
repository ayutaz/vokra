# tools/parity/microwakeword

Offline sidecar for **kahrendt/microWakeWord** (Apache-2.0) → Vokra GGUF
conversion. Bridges the upstream TFLite artefacts (INT8-quantized
MC-MobileNet designed for Cortex-M55 / RP2040 / ESP32-S3 microcontrollers)
to the Vokra GGUF shape the future `vokra-kws-micro` runtime forward will
bind (M5-03 IoT Tier-3 / NFR-PT-03).

Companion to the sister crate [`vokra-vad-micro`](../../crates/vokra-vad-micro),
which does the same job for Silero VAD (M5-03 案 1). The two produce
similarly-shaped GGUF (`vokra.silero.*` vs `vokra.kws.*` metadata
prefix) that the same no_std `vokra_core::gguf::GgufFile::from_external`
reader parses on both host and thumbv8m targets.

## What this directory contains

- `prepare_checkpoint.py` — the actual converter. DL the canonical
  `hey_jarvis` release from ESPHome / kahrendt (parametrisable via
  `--url` / `--input`), extract weight tensors via
  `ai-edge-litert.Interpreter.get_tensor_details()`, dequantize INT8 →
  F32, and emit a GGUF via `gguf.GGUFWriter` with the `vokra.kws.*`
  metadata keys documented in the script's module docstring.
- `dump_reference.py` — Phase 4 host-parity reference dumper. Given a
  `.tflite` and a fixed-seed synthesised PCM window, emits
  `input_pcm.bin` + `features_ref.bin` + `output_ref.bin` +
  `manifest.json` for the Rust parity harness
  (`crates/vokra-kws-micro/tests/parity_microwakeword.rs`). Consumed
  via `VOKRA_KWS_REAL_FIXTURES=<dir>`. See the script's module
  docstring for the honest boundary (numpy log-mel = transcription
  reference; TFLite output = real upstream forward).
- `pyproject.toml` — uv project spec (Python 3.12 pinned per
  `[[feedback-python-3-12]]`, deps = `gguf` + `numpy` +
  `ai-edge-litert`).
- `.python-version` — `3.12` (auto-created by `uv python pin`).

## Prerequisites

- **`uv`** ([[feedback-python-uses-uv]]) — the sidecar toolchain manager
  the Vokra project standardises on. Install via
  `curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`.
- **Python 3.12** — pinned in `.python-version`. `uv sync` will download
  it if absent.

The `ai-edge-litert` package (Apache-2.0) is the direct successor of
`tflite-runtime` — Google renamed it in Q3 2024 when the old package
stopped shipping wheels for Python ≥ 3.12. The Interpreter API surface
is unchanged (`interpreter.allocate_tensors()`,
`interpreter.get_tensor_details()`, `interpreter.get_tensor(idx)`).

## Owner walkthrough — DL + convert

1. **Sync deps** (once per checkout):

   ```
   cd tools/parity/microwakeword
   uv sync
   ```

2. **Download + convert the canonical hey_jarvis model** (~200 KB
   TFLite, produces ~150 KB GGUF after F32 dequantization):

   ```
   uv run python prepare_checkpoint.py \
       --url    https://github.com/esphome/micro-wake-word-models/raw/main/models/v2/hey_jarvis.tflite \
       --name   hey_jarvis \
       --output ~/.cache/vokra-eval/weights/microwakeword/hey_jarvis.gguf \
       --verbose
   ```

   The output GGUF is Vokra-native (`vokra_core::gguf::GgufFile::parse`
   opens it directly) and stamps the following metadata:

   - `vokra.kws.arch` = `"microwakeword"` (distinct from `openwakeword`)
   - `vokra.kws.model` = `"hey_jarvis"` (or your `--name`)
   - `vokra.kws.threshold` = f32 (default `0.5`)
   - `vokra.kws.sample_rate` = 16000, `hop_ms` = 10, `window_ms` = 32,
     `n_mels` = 40, `feature_dim` = 40
   - `vokra.kws.tflite_sha256` = provenance (source hex digest)
   - `vokra.kws.upstream` = source URL
   - `vokra.provenance.license` = `"apache-2.0"`, `license_class` =
     `"Permissive"`, `upstream_hf` = `"kahrendt/microWakeWord"`,
     `upstream_name` = your `--name`

3. **Convert a different wake-word or a locally-downloaded model**:

   ```
   # local file:
   uv run python prepare_checkpoint.py \
       --input  /path/to/alexa.tflite \
       --name   alexa \
       --output ./alexa.gguf

   # override front-end defaults if the model was trained with a
   # different mel front-end (rare — the microWakeWord canonical
   # configuration is 40-band 10-ms hop 32-ms window @ 16 kHz):
   uv run python prepare_checkpoint.py \
       --input  ./custom.tflite \
       --name   custom \
       --output ./custom.gguf \
       --hop-ms 20 --window-ms 40 --n-mels 32
   ```

## Phase roadmap (updated 2026-08-13, Phase 4 lands)

| Phase   | Script / crate work                                                   | Runtime consumes                                                                                                                    |
| ------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **1**   | `prepare_checkpoint.py` (F32-dequant GGUF); 40-band log-mel features  | `vokra-kws-micro::features` real                                                                                                    |
| **2**   | INT8 kernels + model loader                                           | `KwsMicro` scaffold surface                                                                                                         |
| **3**   | INT8 forward-chain interpreter (`interpreter.rs`)                     | `KwsMicro::detect()` REAL mode                                                                                                      |
| **4**   | `dump_reference.py` (host parity fixtures); Rust `tests/` harness     | Path A (GGUF smoke) + Path B (log-mel parity vs numpy transcription). Path C (INT8-chain end-to-end) is UNMET — needs Phase 3.5.    |
| **3.5** | Sidecar Q8_0 emit + per-tensor `(scale, zero_point)` metadata         | `Model` per-layer typed accessors → real `ChainConfig` binding for hey_jarvis. Unlocks Path C.                                      |

Phase 1's F32 dequantization is **lossless** for a fixed
`(scale, zero_point)` pair — the TFLite affine formula
`f32 = scale * (int8 - zero_point)` recovers exact values. Phase 3.5
will add Q8_0 for a ~4× smaller on-device footprint, matching the
microcontroller SRAM budget the M5-03 opt-in Tier-3 target requires,
AND unblock end-to-end INT8 chain parity in the Rust host harness
(Path C).

## What Phase 4 does — and its honest boundary

Phase 4 lands the **host-parity harness**: `dump_reference.py` (this
directory) plus `tests/parity_microwakeword.rs` on the Rust side. See
`crates/vokra-kws-micro/README.md` for the full owner walkthrough.

**Honest boundary** (see `dump_reference.py`'s module docstring for
the full write-up):

- **Path A** (`VOKRA_KWS_REAL_GGUF`) — real GGUF load smoke. Real
    hey_jarvis passes.
- **Path B** (`VOKRA_KWS_REAL_FIXTURES`) — log-mel feature extractor
    parity at `atol = 1e-3` against a **numpy transcription** of the
    standard log-mel algorithm. This validates transcription
    faithfulness (Rust ↔ numpy implement the same algorithm); it does
    NOT validate against training-time `tf.signal` (that would require
    a `tensorflow` dep). Empirically the standard algorithm matches
    `tf.signal` at `1e-3` for the same parameters.
- **Path C** (both env vars) — end-to-end INT8 chain parity. **UNMET**:
    the current sidecar dequantises INT8 → F32 losslessly at emit, so
    the GGUF does not carry per-tensor `(scale, zero_point)`. Wiring
    Path C requires the Phase 3.5 sidecar extension. Until then the
    Rust test skips with a clear defer message — the scaffold is here
    so the flip is a one-file diff.

## What this directory still does NOT do

- **No `vokra-cli convert` entry**: the Rust converter
  (`crates/vokra-convert/src/models/microwakeword.rs`) is a Phase 3.5
  WP. Phase 1 skips that layer by having Python emit GGUF directly, so
  the produced artefact is loadable by the `vokra-vad-micro`-shape
  reader without extra Rust code.
- **No bit-parity against `tf.signal`**: the training-time TF mel
  front-end is a `tensorflow` dep away; the sidecar stays at 3 deps
  (`gguf` + `numpy` + `ai-edge-litert`) and Path B's transcription
  parity is empirically within `1e-3` of `tf.signal` for the same
  parameters.
- **No Cortex-M55 hardware verify**: per M5-03 ADR, hardware / FVP
  runs are owner-only. The cross-build (`thumbv8m.main-none-eabihf`)
  compile gate is documented in `crates/vokra-kws-micro/README.md`
  and can be triggered manually.

## License / distribution note

The **kahrendt/microWakeWord** upstream and **ESPHome micro-wake-word-models**
mirror ship Apache-2.0 code + Apache-2.0 model weights (canonical
release notes verify — the LICENSE file is Apache-2.0 in both repos).
The Vokra runtime consumes the produced GGUF as an opaque numeric
artefact; no Python / TFLite / ESPHome / TensorFlow code enters the
runtime (FR-LD-05 sidecar isolation).

**esphome/esphome** itself is GPL-3.0 licensed — never imported, never
inspected. Vokra's Apache-2.0 posture forbids referencing GPL-3.0 code
for clean-room reasons (see CLAUDE.md "Piper (piper1-gpl)" red-line);
this converter's `--url` default is only the model-file mirror
(`esphome/micro-wake-word-models`), which is Apache-2.0.

## Related

- Design ADR: `docs/adr/M5-03b-kws-micro-no-std.md`
- Sister crate: `crates/vokra-vad-micro` (Silero VAD no_std forward,
  the topology precedent this crate mirrors)
- Feature extractor: `crates/vokra-kws-micro/src/features.rs`
  (Phase 1 Rust-side wave, log-mel front-end matching this script's
  `vokra.kws.*` front-end metadata)
- Upstream: <https://github.com/kahrendt/microWakeWord> (Apache-2.0)
- Curated model mirror:
  <https://github.com/esphome/micro-wake-word-models> (Apache-2.0)
