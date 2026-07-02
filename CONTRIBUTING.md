# Contributing to Vokra

Thank you for considering a contribution. Vokra is developed fully in the
open, and every change — including changes by the maintainer — goes through
the same pull-request and CI pipeline described below.

The design documents under `docs/` (currently Japanese, Draft status) are
the source of truth for requirements and scope; requirement IDs (BR / FR /
NFR) referenced below are defined there.

## 1. Pull requests and branch protection

- `main` is protected: **direct pushes are not allowed**. Every change is
  made through a pull request and must pass CI before merging.
- Link each PR to the issue / work package (WP) it implements, so that the
  change is traceable to requirement IDs.
- The project currently runs with a single maintainer, so the required
  approving-review count may be 0 at this stage — the CI gates below are
  the non-negotiable blocker, not review count.

## 2. CI required checks

As of M0 (v0.1 spike) every PR must pass **6 required checks**:

| Check | What it runs |
|---|---|
| `build` | `cargo build --release` on Linux / macOS / Windows |
| `test` | `cargo test --workspace` on Linux / macOS / Windows |
| `fmt` | `cargo fmt --all -- --check` |
| `clippy` | `cargo clippy --all-targets -- -D warnings` |
| `parity` | `cargo test -p vokra-parity` — numerical parity harness against reference implementations (`tests/parity/`) |
| `license` | `cargo deny check licenses advisories bans` + `cargo audit` + `scripts/check-forbidden-symbols.sh` |

Run the same commands locally before pushing; the CI configuration is
`.github/workflows/ci.yml`.

Planned extension (M1 and later, completing the NFR-MT-07 set): a
performance-regression gate (5% threshold) and execution checks for code
examples in documentation will be added as required checks, and nightly
audio-quality threshold violations will block or revert the offending PR.

## 3. Dependency license policy

- **Allowed**: Apache-2.0, MIT, BSD-family licenses only.
- **Forbidden**: GPL and LGPL in any form — Vokra targets Unity / Godot and
  other proprietary embedding scenarios where (L)GPL is not acceptable.
- **MPL-2.0** (e.g. symphonia): limited use only, after evaluating the
  file-level copyleft implications case by case (see
  [docs/license-audit.md](docs/license-audit.md)).
- `cargo-deny` runs in CI and is a **PR blocker**: a PR that introduces a
  GPL/LGPL dependency cannot merge (NFR-LC-04).
- Keep new dependencies minimal and justified; prefer std / existing
  workspace code over adding a crate.

## 4. Adding support for a new model

A PR that adds model support must:

1. **Update [docs/license-audit.md](docs/license-audit.md)** in the same PR
   (license of code *and* weights, commercial usability, training-data
   provenance).
2. Respect the model-zoo policy: weights under **CC-BY-NC / CC-BY-NC-SA or
   with unclear training-data rights are excluded from the official model
   zoo** and may only be exercised behind an explicit research flag
   (engine support without weight distribution).
3. For TTS / VC models, go through the
   [docs/legal-compliance.md](docs/legal-compliance.md) checklist
   (EU AI Act Article 50 / California SB 942: AudioSeal watermarking ON by
   default, C2PA manifest support, disclosure requirements).
4. Update [NOTICE](NOTICE) when the addition carries attribution or
   distribution-relevant terms.

## 5. Design red lines

The following are fixed design decisions. PRs that cross them will be
declined regardless of implementation quality:

- **No ONNX graph loading in the runtime.** ONNX models are handled
  exclusively by the offline conversion tool; the runtime must stay free of
  onnxruntime / onnx / protobuf dependencies (FR-LD-05).
- **No onnxruntime in the piper-plus inference path.** The MB-iSTFT-VITS2
  inference stack is natively reimplemented in Rust (maintainer decision,
  2026-07-02); only the G2P text preprocessing is reused from piper-plus
  for the time being.
- **No eSpeak-NG** (GPL-3.0) in the core. G2P comes from piper-plus's own
  MIT implementation or IPA-dictionary-based approaches.
- **No NNAPI backend** (deprecated by Google as of Android 15).
- **No soxr / rubberband** (GPL). Resampling is a native implementation
  based on the speexdsp (BSD) resampler design.

## 6. Finding something to work on

Issues labeled **`good first issue`** are curated to be self-contained
entry points with clear acceptance criteria. If you want to take a larger
work package, comment on the corresponding WP issue first so scope can be
agreed before you invest time.
