# tools/parity/microwakeword

> **Current status (2026-09-01): BLOCKED.** The dedicated Python 3.12 lock
> contains only the first-party preparer (zero external dependencies). The
> stdlib raw FlatBuffer manifest carries authenticated constant bytes; the
> preparer emits Q8_0/F32/I32 GGUF without an interpreter or NumPy. The VAST
> normal production invocation still exits before acquisition because the
> target byte SHA-256, reviewed topology authority, and parity evidence are
> pending. The explicitly separate `--inspect-only` mode is VAST-gated and
> records evidence only:
> `scripts/publish/vast-ai/run-microwakeword-validation.sh`.
>
> The dedicated lock SHA-256 is
> `984703d5bafdd6c88006bd381095961d42ef684d269d66194edbeda1fddf8dc2`;
> its one-row package digest is
> `d9b806830227b4fdbdbe59ea5a20b529bfae40f6aa70e239b44a6238fabd5ad7`,
> and its first-party license digest is
> `4ee7351311d5d0bf69758093e88be7b4146fefdcbc80e026662bbdf58032272c`.

License evidence is version-keyed in `microwakeword_inspect.py`; the only
locked package is the first-party preparer. This removes the former
transitive-wheel license gate. Artifact provenance, reviewed topology, and
real parity remain independent fail-closed gates.

> **2026-08-30 historical boundary:** The earlier 2026-08-29 audit snapshot
> does not authorize local acquisition, conversion, model execution, or an
> implied parity result. The current 2026-09-01 status above adds the
> dependency-free preparer and VAST-only inspection boundary. Apply the current
> repository AGENTS/skill gates and consult the M5 ledger before changing this
> status.

For auditability, `LICENSE_ROWS` records the first-party package identity. No
third-party license is inferred from a lock or from an unversioned project
page.

Offline sidecar for **kahrendt/microWakeWord** (Apache-2.0) → Vokra GGUF
conversion. Bridges the upstream TFLite artefacts (INT8-quantized
MC-MobileNet designed for Cortex-M55 / RP2040 / ESP32-S3 microcontrollers)
to the Vokra GGUF shape the `vokra-kws-micro` forward scaffold can validate
through its explicitly untrusted typed-topology seam (M5-03 IoT Tier-3 /
NFR-PT-03). Production binding remains closed until an owner-reviewed VAST
artifact and parity evidence set the compiled topology authority.

Companion to the sister crate [`vokra-vad-micro`](../../crates/vokra-vad-micro),
which does the same job for Silero VAD (M5-03 案 1). The two produce
similarly-shaped GGUF (`vokra.silero.*` vs `vokra.kws.*` metadata
prefix) that the same no_std `vokra_core::gguf::GgufFile::from_external`
reader parses on both host and thumbv8m targets.

## What this directory contains

- `prepare_checkpoint.py` — the authenticated conversion design. It consumes
  raw producer `data_hex` constants and emits
  direct GGUF Q8_0 source-byte carriers plus exact dense GGUF I32 carriers for
  affine bias tensors. Production conversion additionally
  refuses a topology whose `canonical_identity` is unset; the raw producer
  emits only a canonical evidence digest until VAST review closes that gate.
  It requires an independently hashed VAST tensor manifest proving which
  FlatBuffer buffers are persistent constants (`complete: true`, exact tensor
  index/name, buffer index/size, and source dtype); it is not a local
  acquisition or execution procedure while this gate is blocked.
- `../microwakeword_tensor_manifest.py` — dependency-free raw TFLite
  FlatBuffer producer. On VAST it authenticates `TFL3`, the single subgraph,
  tensor/buffer ownership, exact shapes and byte hashes, then publishes a
  complete no-clobber manifest. It never uses interpreter tensor success to
  classify constants.
- `dump_reference.py` — Phase 4 host-parity reference dumper. Given a
  `.tflite` and a fixed-seed synthesised PCM window, emits
  `input_pcm.bin` + `features_ref.bin` + `output_ref.bin` +
  `manifest.json` for the Rust parity harness
  (`crates/vokra-kws-micro/tests/parity_microwakeword.rs`). Consumed
  via `VOKRA_KWS_REAL_FIXTURES=<dir>`. It is a separate VAST-only reference
  path and is not included in this production lock; see the script's module
  docstring for its honest boundary (numerical transcription/reference
  tooling is not a production conversion dependency).
- `pyproject.toml` — dependency-free uv project spec (Python 3.12).
- `.python-version` — `3.12` (auto-created by `uv python pin`).

## Prerequisites

- **`uv`** ([[feedback-python-uses-uv]]) and **Python 3.12** are the sidecar
  toolchain. The project is isolated from the parent workspace and its exact
  lock is checked by `microwakeword_inspect.py`.
- Conversion, reference generation, and any artifact acquisition are
  VAST-only. The local Mac path is deliberately terminally blocked; do not
  install, sync, acquire, or convert from this directory.

The VAST worker's `VOKRA_INSPECT_ONLY=1 --inspect-only` mode is the only
acquisition path. It uses the fixed repository revision/path, verifies the
recorded model and license Git blobs plus sizes, hashes the downloaded model,
and emits raw-manifest/canonical-topology evidence into the existing
`MICROWAKEWORD_INSPECTION_DIR`. The directory receives the complete tensor
manifest, fixed companion JSON, LICENSE, and a summary containing their
SHA-256 values; the model `.tflite` remains temporary and is cleaned up. It
performs no GGUF conversion, inference, Cargo build, or upload. Production
remains closed until that evidence is reviewed and the compiled
topology/artifact authorities are set.

The production preparer intentionally has no interpreter or numerical Python
dependency. Reference generation in `dump_reference.py` remains a separate
VAST-only research path and is not part of the production conversion lock.

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
| **1**   | `prepare_checkpoint.py` (direct Q8_0 GGUF); 40-band log-mel features  | `vokra-kws-micro::features` real                                                                                                    |
| **2**   | INT8 kernels + model loader                                           | `KwsMicro` scaffold surface                                                                                                         |
| **3**   | INT8 forward-chain interpreter (`interpreter.rs`)                     | Untrusted/synthetic `Model::bind_untrusted_topology` path; production authenticated artifact binding remains pending                  |
| **4**   | `dump_reference.py` (host parity fixtures); Rust `tests/` harness     | Path A/B require authenticated VAST artefacts; Path C (INT8-chain end-to-end) remains unmet                                        |
| **3.5** | Sidecar Q8_0 carrier + per-tensor `(scale, zero_point)` metadata (implemented); dense I32 bias preservation (implemented) | Raw TFLite topology producer and typed binder are synthetic-tested; real artifact bind and parity remain pending |

The sidecar's source-byte carrier preserves exact INT8 values; its F32 view is
**lossless** for a fixed
`(scale, zero_point)` pair — the TFLite affine formula
`f32 = scale * (int8 - zero_point)` recovers exact values. Phase 3.5 now
provides the ~4× smaller Q8_0 carrier, matching the microcontroller SRAM
budget the M5-03 opt-in Tier-3 target requires. It does not by itself
establish topology binding or end-to-end parity.

## What Phase 4 does — and its honest boundary

Phase 4 lands the **host-parity harness**: `dump_reference.py` (this
directory) plus `tests/parity_microwakeword.rs` on the Rust side. See
`crates/vokra-kws-micro/README.md` for the full owner walkthrough.

**Honest boundary** (see `dump_reference.py`'s module docstring for
the full write-up):

- **Path A** (`VOKRA_KWS_REAL_GGUF`) — real GGUF load smoke once the
    authenticated hey_jarvis GGUF is produced by the approved VAST workflow.
- **Path B** (`VOKRA_KWS_REAL_FIXTURES`) — log-mel feature extractor
    parity at `atol = 1e-3` against a **numpy transcription** of the
    standard log-mel algorithm. This validates transcription
    faithfulness (Rust ↔ numpy implement the same algorithm); it does
    NOT validate against training-time `tf.signal` (that would require
    a `tensorflow` dep). Empirically the standard algorithm matches
    `tf.signal` at `1e-3` for the same parameters.
- **Path C** (both env vars) — end-to-end INT8 chain parity. **UNMET**:
  the Q8_0/I32 carriers and affine metadata are implemented, but authenticated
    model topology and independent reference fixtures remain pending the
    VAST-only acquisition/license gate. The Rust test therefore still skips
    with a clear defer message.

## What this directory still does NOT do

- **No `vokra-cli convert` entry**: the Rust converter
  (a future `crates/vokra-convert/src/models/microwakeword.rs`, not yet
  written) is a Phase 3.5 WP. Phase 1 skips that layer by having Python emit GGUF directly, so
  the produced artefact is loadable by the `vokra-vad-micro`-shape
  reader without extra Rust code.
- **No bit-parity against `tf.signal`**: the training-time TF mel
  front-end is a `tensorflow` dependency away. The production sidecar stays
  dependency-free; the separate VAST-only reference path and Path B's
  transcription parity are not production conversion evidence.
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
