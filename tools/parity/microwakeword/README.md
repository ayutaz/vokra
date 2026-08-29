# tools/parity/microwakeword

> **Current status (2026-08-29): BLOCKED.** The dedicated Python 3.12 lock
> is present and records the complete 17-package closure. Exact source/model
> Git identities are recorded below, but license policy review remains open:
> ai-edge-litert's precompiled-wheel notices, certifi/MPL, NumPy's composite
> bundled notices, protobuf precompiled-wheel metadata, PyYAML
> native-extension notices, tqdm/MPL, typing-extensions/PSF, and the
> ml-dtypes wheel's Eigen/MPL notice. The VAST worker therefore exits before environment sync
> or acquisition; the target's byte SHA-256 is also pending that acquisition:
> `scripts/publish/vast-ai/run-microwakeword-validation.sh`.
>
> The dedicated lock SHA-256 is
> `43e17e20616bc06072424abadaaed520244673db2f964a29ea2472e22e72afbe`;
> its 17-row package/dependency digest is
> `3250cac13ab9f8cf0a67ffc1f590988afa8cac3b346edf52d0e03924ec08ef06`,
> and its version-keyed license digest is
> `2bcae92a909b92617e1ddc96a7cf4704a6c9305dcd94651584da4b68c49a7906`.

License evidence is version-keyed in `microwakeword_inspect.py`. Each row was
checked against the exact PyPI JSON record, preferring `license_expression`,
then `license`, then the exact wheel METADATA/classifiers when needed. Thus
ai-edge-litert is Apache-2.0 but its precompiled TFLite runtime wheel still
requires bundled-notice review; charset-normalizer is MIT, idna is BSD-3-Clause, NumPy records its exact
`BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` expression,
typing-extensions is PSF-2.0, and urllib3 is MIT. The ml-dtypes description /
license section says Apache-2.0 but also requires review of the Eigen/MPL-2.0
notice shipped by precompiled wheels. Protobuf's exact metadata also remains
subject to precompiled-wheel notice review. No source archive was fetched; a
resolved lock is not license approval, and policy-sensitive rows remain
blocked.

For auditability, every non-first-party lock row has its exact package/version
and primary PyPI JSON URL in `LICENSE_ROWS`: `license_expression` supplies
idna, NumPy, typing-extensions, and urllib3; `license` supplies ai-edge-litert,
backports-strenum, certifi, charset-normalizer, flatbuffers, protobuf, PyYAML,
requests, and tqdm; and exact wheel METADATA/classifiers supply colorama and
gguf. The ml-dtypes row uses the exact PyPI description/license section and
records its precompiled-wheel Eigen/MPL-2.0 notice. No license is inferred
from a lock or from an unversioned project page.

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

- `prepare_checkpoint.py` — a future converter design. It would extract
  tensors with `ai-edge-litert.Interpreter.get_tensor_details()`, dequantize
  INT8 → F32, and emit a GGUF via `gguf.GGUFWriter`; it is not a local
  acquisition or execution procedure while this gate is blocked.
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

- **`uv`** ([[feedback-python-uses-uv]]) and **Python 3.12** are the sidecar
  toolchain. The project is isolated from the parent workspace and its exact
  lock is checked by `microwakeword_inspect.py`.
- Conversion, reference generation, and any artifact acquisition are
  VAST-only. The local Mac path is deliberately terminally blocked; do not
  install, sync, acquire, or convert from this directory.

The `ai-edge-litert` package reports Apache-2.0 in exact PyPI metadata and is
the direct successor of `tflite-runtime` — Google renamed it in Q3 2024 when
the old package stopped shipping wheels for Python ≥ 3.12. Its precompiled
TFLite runtime wheel still requires bundled-notice review, so this fact does
not clear the gate. The future Interpreter API surface is expected to remain
(`interpreter.allocate_tensors()`, `interpreter.get_tensor_details()`,
`interpreter.get_tensor(idx)`).

## Historical conversion notes (not an execution procedure)

Earlier drafts described local dependency sync, raw GitHub model retrieval,
and direct `prepare_checkpoint.py` conversion. Those notes are retained only
to explain the intended future sidecar shape. They are not runnable guidance:
all conversion/reference execution and all source/model acquisition must be
performed by an owner-approved VAST workflow after the dependency and license
gate clears. The current worker has no acquisition or conversion path.

The future GGUF is expected to carry `vokra.kws.arch = "microwakeword"`,
`vokra.kws.model = "hey_jarvis"`, the 16 kHz / 40-band front-end metadata,
the acquired artifact's byte SHA-256, and the immutable upstream identity.
Those fields are design targets, not a conversion or parity result.

## Authenticated upstream identities

The following identities were observed from the upstream Git repositories on
2026-08-29. The values labelled `Git blob` are Git object IDs, not file
SHA-256 digests. The actual model-byte SHA-256 is intentionally unset until a
VAST acquisition records it.

| Role | Repository revision | Path | Git blob | Size |
| --- | --- | --- | --- | ---: |
| Source license | `kahrendt/microWakeWord@4665173cd35f1cff9a61e06fc427f124766c488e` | `LICENSE` | `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` | 11357 |
| Source inference | same | `microwakeword/inference.py` | `ec0634376accb8e7832205c117149f4acb3e6cf0` | — |
| Source network | same | `microwakeword/mixednet.py` | `75cbb9fa950fa4135a0e3a4171b9fba84c4b989c` | — |
| Source streaming | same | `microwakeword/layers/stream.py` | `37b77702c8ee8038c4e6e91979560e264e7555c1` | — |
| Source spectrogram | same | `microwakeword/audio/spectrograms.py` | `5adb585ab3a650dfd17728a0e200a143d41c23f7` | — |
| Source metadata | same | `pyproject.toml` | `e2156f94b8a2bc4821cccd72492889016e40b532` | — |
| Model license | `esphome/micro-wake-word-models@05b65922cc433c9df13e98e32a7fe520758c837e` | `LICENSE` | `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` | 11357 |
| Target model | same | `models/v2/hey_jarvis.tflite` | `0075302434cc72a460ced0b8f6c09c69214e5cf0` | 52272 |
| Target metadata | same | `models/v2/hey_jarvis.json` | `e6733fe13852f04a5a3ae83e0d39b5726aee62cc` | 388 |

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
  (a future `crates/vokra-convert/src/models/microwakeword.rs`, not yet
  written) is a Phase 3.5 WP. Phase 1 skips that layer by having Python emit GGUF directly, so
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
mirror each have the authenticated `LICENSE` blob recorded above; the
repository policy review identifies both as Apache-2.0. This is provenance
evidence, not permission to acquire or distribute the model from the local
Mac.
The Vokra runtime consumes the produced GGUF as an opaque numeric
artefact; no Python / TFLite / ESPHome / TensorFlow code enters the
runtime (FR-LD-05 sidecar isolation).

**esphome/esphome** itself is GPL-3.0 licensed — never imported, never
inspected. Vokra's Apache-2.0 posture forbids referencing GPL-3.0 code
for clean-room reasons (see CLAUDE.md "Piper (piper1-gpl)" red-line). The
future converter must use only the authenticated model mirror above.

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
