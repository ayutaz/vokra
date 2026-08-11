//! SBV2 (Style-Bert-VITS2 v2) **real**-checkpoint parity (Task 28).
//!
//! Sibling of `parity_sbv2_synthetic.rs` (Task 27, which exercises
//! `SbV2Model::synthetic_for_test()`'s procedurally-generated weights — no
//! trained checkpoint, no reference to diff against). This file is the
//! other half: it reads a **real** Style-Bert-VITS2 v2 checkpoint (converted
//! to GGUF) plus a Python-generated reference dump, and — when both are
//! present — diffs a real Rust forward pass against them using
//! `sbv2::parity::tolerance_for` (Task 26).
//!
//! # Gating (fabricated pass 禁止)
//!
//! The one `#[test]` below is `#[ignore]`d: `cargo test` skips it by
//! default (this file's committed state — no fixture exists in this repo
//! yet), and Task 28's own DoD is compile-only (`docs/superpowers/plans/
//! 2026-07-26-sbv2-v2.md` Task 28 Step 1/Step 4: "ignored、compile 通過が本
//! task の DoD" / "1 test ignored"). Fixtures land with **Task 34** (`tests/
//! fixtures/sbv2/README.md` + `*.gguf.sha256` placeholders + this file's
//! manifest) and **Task 30** (`tools/parity/sbv2_dump_reference.py`, which
//! populates `reference_dump/*.bin`) — run with `cargo test -p vokra-models
//! --test parity_sbv2_real -- --ignored` once both have landed.
//!
//! Once opted in via `--ignored`, a missing fixture is a **loud panic**
//! naming exactly what is missing and how to produce it — never a silent
//! skip-and-pass (FR-EX-08; the `#[ignore]` attribute itself is the one and
//! only "not run" gate, matching the sibling `sbv2_gguf_loader.rs`'s
//! `from_gguf_loads_real_sbv2_weights` real-fixture test's own convention).
//!
//! **WP-06 update (Wave 0 Task 6, 2026-08-11)**: the fixtures named above
//! have since landed for real (`tests/fixtures/sbv2/*.gguf` +
//! `reference_dump.manifest.json` + `reference_dump/*.bin`, all committed),
//! so the `#[ignore]` attribute this section describes has been REMOVED —
//! the test now runs by default under plain `cargo test`, same as any
//! other test in this file. The historical "gated behind --ignored" text
//! above is preserved (append-never-delete, Kokoro `PROSODY_F0_ATOL`
//! precedent) since it documents why the test was originally written
//! ignored; it no longer describes the current state.
//! `require_fixture`'s loud-panic behavior on a genuinely missing fixture
//! is unchanged — see [`StageResult`] below for how a NUMERIC parity miss
//! (fixture present, diff exceeds atol) is now reported instead: every
//! stage's outcome is collected into ONE aggregated report and asserted
//! once at the end of the test, rather than each stage panicking the
//! whole run on the first miss.
//!
//! # The manifest contract (this file's schema — Task 34 must match)
//!
//! `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §10 specifies that
//! Task 34 commits `tests/fixtures/sbv2/reference_dump.manifest.json` but
//! does not pin its exact shape beyond "dump 対象 tensor 名 + shape の期待
//! 値の JSON". This file defines that shape, since Task 34 has not landed
//! yet and something must be authoritative first:
//!
//! ```text
//! {
//!   "generator_version": "1.0",
//!   "checkpoint": {
//!     "sbv2_main": "sbv2-v2-multilingual-base.gguf",
//!     "bert_ja": "deberta-v2-large-japanese-char-wwm.gguf",
//!     "bert_en": "deberta-v3-large.gguf"
//!   },
//!   "request": {
//!     "text": "...",
//!     "language": "JA",
//!     "speaker_id": 0,
//!     "style_vec": [0.0, 0.0, "..."],
//!     "speed": 1.0,
//!     "noise_scale": 0.667,
//!     "noise_scale_w": 0.8,
//!     "seed": 42
//!   },
//!   "phonemize_fixture": {                                // Task 7 addition
//!     "phoneme_ids":     {"path": "reference_dump/phoneme_ids.bin",
//!                         "count": "T_text", "dtype": "uint16"},
//!     "tones":           {"path": "reference_dump/tones.bin",
//!                         "count": "T_text", "dtype": "uint8"},
//!     // M6 refactor (2026-08-06): `word_boundaries` is retained (the G2P
//!     // stage still emits it — see `g2p.rs`'s M6 note), but is no longer
//!     // consumed by `SbV2TextEncoder::forward`. The reader below leaves
//!     // this field in the manifest for schema stability but does not read
//!     // the side file if it is present-but-unused.
//!     "word_boundaries": {"path": "reference_dump/word_boundaries.bin",
//!                         "count": "T_text", "dtype": "uint8"}
//!   },
//!   "tensors": [
//!     {"name": "phoneme_embed",   "path": "reference_dump/phoneme_embed.bin",   "shape": ["T_text", 192]},
//!     {"name": "text_hidden",     "path": "reference_dump/text_hidden.bin",     "shape": ["T_text", 192]},
//!     {"name": "bert_hidden_ja",  "path": "reference_dump/bert_hidden_ja.bin",  "shape": ["T_bert", 1024]},
//!     {"name": "bert_hidden_en",  "path": "reference_dump/bert_hidden_en.bin",  "shape": ["T_bert", 1024]},
//!     {"name": "bert_bridge_out", "path": "reference_dump/bert_bridge_out.bin", "shape": ["T_text", 192]},
//!     {"name": "speaker_embed",   "path": "reference_dump/speaker_embed.bin",   "shape": [1, 512]},
//!     {"name": "style_projected", "path": "reference_dump/style_projected.bin", "shape": [1, 192]},
//!     {"name": "sdp_sample",      "path": "reference_dump/sdp_sample.bin",      "shape": ["T_text"]},
//!     {"name": "mel_hidden",      "path": "reference_dump/mel_hidden.bin",      "shape": ["T_mel", 192]},
//!     {"name": "z_latent",        "path": "reference_dump/z_latent.bin",        "shape": ["T_mel", 192]},
//!     {"name": "waveform",        "path": "reference_dump/waveform.bin",        "shape": [1, "samples"]}
//!   ]
//! }
//! ```
//!
//! `phonemize_fixture` (Task 7) is the fixture-bypass G2P input this test
//! uses to reproduce the reference forward pass's exact phoneme
//! ids/tones/word_boundaries without needing a real 8-language piper-plus
//! G2P available in-workspace (see the "The G2P bypass" section below for
//! why). Each entry is one raw-bytes side file `T_text` elements long,
//! typed by `dtype` (`uint16` LE for `phoneme_ids`, `uint8` for the two
//! others). The Rust reader dispatches on `dtype`.
//!
//! * `checkpoint.*` values are **bare filenames**, siblings of this
//!   manifest directly inside `tests/fixtures/sbv2/` — matching Task 34's
//!   own `Files:` list verbatim (`sbv2-v2-multilingual-base.gguf` /
//!   `deberta-v2-large-japanese-char-wwm.gguf` / `deberta-v3-large.gguf`),
//!   which is also what their `*.gguf.sha256` sidecar placeholders name.
//!   This test reads them from the manifest rather than hard-coding them,
//!   so Task 34 is free to point at a differently-named real checkpoint
//!   without a Rust-side edit.
//! * `tensors[].path` values are relative to that **same** directory but
//!   already include the `reference_dump/` sub-directory prefix (Task 30's
//!   dumper writes there) — do not re-prepend it.
//! * `tensors[].shape` elements must all be JSON integers (the `"T_text"` /
//!   `"samples"` placeholders above are illustrative only — a real manifest
//!   substitutes the actual dumped dimensions, e.g. `[1, 24000]`).
//! * `request` is **not** in the design doc's §10 sketch — it is added
//!   here because [`SbV2SynthRequest`] (the only input
//!   [`SbV2Model::synthesize`] accepts) has no "raw phoneme ids" entry
//!   point; reproducing the Python dumper's exact forward pass needs its
//!   `text`/`language`/`speaker_id`/`style_vec`/`speed`/`noise_scale`/
//!   `noise_scale_w`/`seed` verbatim. `speed` must be strictly positive
//!   (`SbV2Model::synthesize`'s own precondition).
//!
//! # Scope: full 11-tensor manifest diff (WP-01, 2026-08-09)
//!
//! WP-01 wires the harness through
//! [`SbV2Model::synthesize_with_intermediates`] +
//! [`SbV2Intermediates::to_dumper_map`] (both landed in Wave-4
//! INTERMEDIATE-ACCESSORS), so the harness now diffs every manifest tensor
//! it can — not just `waveform` — against its `reference_dump/<name>.bin`
//! fixture, using `sbv2::parity::tolerance_for` for the per-tensor bound
//! (`sbv2::parity::atol_calibration_for` records whether that bound is
//! measured, estimated pre-fixture, or an `UnmeasuredDefault` pass-through
//! to `ATOL_DEFAULT`).
//!
//! The waveform assertion mechanism is unchanged (tolerance-based length
//! band + max_abs_diff + RMS on the overlapping prefix — see the
//! `PR27-WAVEFORM-TOLERANCE` block below). Intermediate tensors use a
//! simpler per-tensor `max_abs_diff` gate — their shapes are pinned by the
//! manifest schema, so a length mismatch there is a real bug, not a
//! discrete-step ±1 flip like the waveform's `CEIL(exp(logw))` boundary.
//!
//! Every step emits a `[parity_sbv2_real] <name>: max |Δ| = X <= atol Y
//! (status)` summary line to stderr — mirroring the Kokoro parity CI's
//! per-tensor summary format, so a CI viewer can see all 11 rows even when
//! all 11 pass.
//!
//! # The G2P bypass: `from_gguf_with_phonemizer` + `PhonemizeFixture` (Task 7)
//!
//! [`SbV2Model::from_gguf`]'s own doc ("G2P is not loaded here") is explicit:
//! the 3-file loader signature (`main` + `bert_ja` + `bert_en`) has no
//! piper-plus G2P GGUF, so the model it returns carries an internal
//! `UnwiredPhonemizer` that turns **every** `synthesize` call into
//! `Err(VokraError::NotImplemented(_))` at the G2P step — deliberately, so a
//! `from_gguf`-loaded model never *silently* emits wrong-but-plausible audio
//! (FR-EX-08). A real, working G2P
//! (`SbV2Phonemizer::from_piper_g2p`) needs a `vokra_piper_plus::Phonemizer`
//! implementation with real 8-language coverage, which — per
//! `vokra-piper-plus/src/phonemizer.rs`'s own doc — lives **out of** this
//! zero-dependency root workspace (`integrations/vokra-piper-g2p`, M1-01-A);
//! `crates/vokra-models` cannot depend on it (NFR-DS-02).
//!
//! Task 7 resolves this specifically for parity testing (not for
//! production) via two additions in
//! `crates/vokra-models/src/sbv2/g2p.rs` +
//! `crates/vokra-models/src/sbv2/mod.rs`:
//! [`SbV2Phonemizer::from_fixture`] (a pre-computed
//! `(language, text) -> PhonemizeResult` lookup) and
//! [`SbV2Model::from_gguf_with_phonemizer`] (a sibling of
//! [`SbV2Model::from_gguf`] that swaps the internal `UnwiredPhonemizer`
//! placeholder for a caller-supplied [`SbV2Phonemizer`]). This test uses
//! them by reading `phonemize_fixture.*` from the manifest and its three
//! typed side files (`phoneme_ids.bin` `uint16`, `tones.bin` `uint8`,
//! `word_boundaries.bin` `uint8`), constructing a single-entry
//! [`PhonemizeFixture`] keyed on the manifest's own
//! `(request.language, request.text)` pair, wrapping that in an
//! `SbV2Phonemizer::from_fixture`, and passing it to
//! `SbV2Model::from_gguf_with_phonemizer` — reproducing the exact G2P
//! output the Python reference dumper fed the reference forward pass, so
//! `SbV2Model::synthesize` can then run end-to-end without an in-workspace
//! 8-language G2P. A `synthesize` `Err` under this wiring is a real
//! parity failure and always panics (there is no longer a documented
//! `NotImplemented` outcome to log-and-pass through).
//!
//! The fixture-bypass is deliberately **not** a production G2P — a
//! `(language, text)` pair absent from the fixture is a loud
//! `VokraError::InvalidArgument`, never a silent fall-through to a
//! different path — so this construction path validates nothing outside
//! the one manifest-declared test sentence per run. Extending coverage
//! means populating the fixture with more `(language, text)` entries and
//! adding manifest side files for each, not adding a fall-through.
//!
//! # WP-24: UTMOS quality gate (env-gated tail check)
//!
//! Once the waveform max-|Δ| check above passes, an *optional* UTMOS
//! quality gate runs — an absolute UTMOS-delta assertion between the Rust
//! forward pass's waveform and the reference-dumped waveform, gated at
//! [`vokra_models::sbv2::UTMOS_ATOL`] (see [`vokra_models::sbv2::parity`]'s
//! module doc for the honest-atol derivation). The waveform check pins
//! numerical parity (bit-for-bit-ish agreement at the tolerance floor);
//! the UTMOS check pins **perceptual** parity (a real regression could in
//! principle slip below the raw sample delta while degrading the
//! MOS-predicted quality noticeably — that is exactly what NFR-QL-02 asks
//! the scorer to catch).
//!
//! The gate is **fail-closed opt-in** (mirrors `parity-deepfilternet3-real.yml`
//! and `parity-utmos.yml`'s discipline): the environment variable
//! `VOKRA_SBV2_UTMOS_ENABLE=1` **plus** `VOKRA_SBV2_UTMOS_GGUF=<path to a
//! `vokra.utmos.*` GGUF>` are both required to opt in. When `ENABLE` is
//! unset, the whole leg silently skips (a clean, deliberate skip — never a
//! fabricated pass, FR-EX-08); when `ENABLE=1` but `GGUF` is missing/
//! unreadable/synthesized/wrong-shape, the test panics loudly (a broken
//! opt-in must never look like a skip). See [`utmos_gate`] for the
//! implementation and [`UtmosGateSettings`] for the opt-in resolution.
//!
//! Sample-rate handling: UTMOS22-strong runs at 16 kHz, SBV2 outputs
//! 44.1 kHz. The UTMOS metric refuses silent resampling (FR-EX-08), so
//! this leg resamples **both** waveforms explicitly via
//! [`vokra_ops::resample`] before scoring — the Rust and reference
//! waveforms take the identical resampler path, so any downsample-induced
//! rounding cancels between the two scores (an absolute delta is
//! symmetric).
//!
//! CI wiring lives in `.github/workflows/parity-sbv2-real.yml` (the
//! "UTMOS quality gate" step conditionally sets both env vars from
//! `vars.VOKRA_SBV2_UTMOS_ENABLE` + a converted UTMOS GGUF the same
//! workflow produces before running this test).

use std::path::{Path, PathBuf};

use vokra_core::VokraError;
use vokra_core::gguf::GgufFile;
use vokra_core::ir::graph::{MelAttrs, MelInterp, MelNorm, MelScale, StftAttrs};
use vokra_core::json::{self, JsonValue};
use vokra_eval::metrics::utmos::Utmos;
use vokra_models::sbv2::{
    AtolCalibration, Language, MEL_LOSS_ATOL, PhonemizeFixture, PhonemizeResult, RngMode,
    SbV2Intermediates, SbV2Model, SbV2Phonemizer, SbV2SynthRequest, UTMOS_ATOL,
    atol_calibration_for, tolerance_for,
};
use vokra_ops::{mel_filterbank, stft};

// SBV2 v2 JP-Extra base target: 44.1 kHz output (see
// `crates/vokra-models/src/sbv2/decoder.rs`'s module doc, `sample_rate` field
// there is pinned to `44_100`), with the mel-loss front-end using the same
// n_fft / hop / n_mels VITS-family checkpoints train against.
// `n_fft=2048, hop=512, n_mels=128` matches the litagin/Style-Bert-VITS2 v2
// upstream config the parity CI (`.github/workflows/parity-sbv2-real.yml`)
// downloads (`filter_length=2048`, `hop_length=512`, `n_mel_channels=128`).
// Sample rate, n_fft, and hop are also what the SbV2 config side-car
// (`vokra.sbv2.sample_rate` / `vokra.sbv2.decoder.upsample_rates` product =
// hop) exposes to the loader — this front-end matches, so a real-fixture run
// diffs against the same mel band structure the reference dumper computes.
const SBV2_MEL_SR: u32 = 44_100;
const SBV2_MEL_N_FFT: usize = 2048;
const SBV2_MEL_HOP: usize = 512;
const SBV2_MEL_N_MELS: usize = 128;

/// Repo-root-relative real-fixture directory for SBV2 parity
/// (`tests/fixtures/sbv2/`, sibling of the existing `tests/fixtures/audio/`
/// Whisper/Voxtral convention). `CARGO_MANIFEST_DIR` is
/// `<repo>/crates/vokra-models` — `cargo test` sets a test binary's working
/// directory to the crate root, not the invocation directory, so every
/// repo-root fixture path in this workspace's parity tests is built this
/// way (`parity_whisper.rs`, `parity_kokoro.rs`, `parity_voxtral.rs`,
/// `parity_csm.rs`, `parity_moshi.rs`) rather than as a bare relative
/// literal.
fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("sbv2")
}

/// Loud, actionable precondition check (FR-EX-08): once a caller has opted
/// in via `--ignored`, a missing fixture panics — it never silently
/// skips-and-passes. `what` names the fixture's role in the panic message.
fn require_fixture(path: &Path, what: &str) {
    assert!(
        path.exists(),
        "[parity_sbv2_real] MISSING fixture: {} ({what}). Populate `tests/fixtures/sbv2/` \
         per Task 34's README.md (the real SBV2 v2 + DeBERTa v2/v3 checkpoints, converted \
         with `vokra-cli convert`) plus `reference_dump.manifest.json` and \
         `reference_dump/*.bin` (Task 30's `tools/parity/sbv2_dump_reference.py`), then \
         re-run with `cargo test -p vokra-models --test parity_sbv2_real -- --ignored`. This \
         is a clean gated precondition failure, not a numeric-parity regression.",
        path.display(),
    );
}

/// Reads `path` as a flat little-endian `f32` array (Task 30's dumper
/// format — the same raw-`f32`-bytes convention `parity_moshi.rs`'s
/// `read_f32` and `parity_csm.rs` use for their own `reference_dump/*.bin`
/// siblings).
fn read_f32_bin(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(
        bytes.len() % 4,
        0,
        "{}: byte length {} is not a multiple of 4 (not f32-aligned)",
        path.display(),
        bytes.len(),
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Task 7: reads `path` as a flat little-endian `u16` array (dumper's
/// `phoneme_ids.bin` format — `PHONEMIZE_FIXTURE_SCHEMA["phoneme_ids"].dtype
/// == "uint16"` in `tools/parity/sbv2_dump_reference.py`). Panics on a
/// non-`u16`-aligned length so a truncated/corrupt fixture is a loud
/// failure rather than a silent short read.
fn read_u16_bin(path: &Path) -> Vec<u16> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(
        bytes.len() % 2,
        0,
        "{}: byte length {} is not a multiple of 2 (not u16-aligned)",
        path.display(),
        bytes.len(),
    );
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Task 7: reads `path` as a flat `u8` array (dumper's `tones.bin` /
/// `word_boundaries.bin` format — `PHONEMIZE_FIXTURE_SCHEMA[*].dtype ==
/// "uint8"`). No endian conversion; every byte is one element.
fn read_u8_bin(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Largest absolute per-element difference between two equal-length slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .fold(0.0f32, |worst, (x, y)| worst.max((x - y).abs()))
}

/// Root-mean-square difference between two slices, computed over the
/// **overlapping prefix** `[0, min(a.len(), b.len()))`. Waveform-signal
/// comparison after `max_abs_diff` — the max-diff is the "worst frame"
/// (dominated by transient boundaries), and RMS is the "average energy"
/// (more forgiving to a single ceiling-flip artifact but very sensitive
/// to sustained divergence).
///
/// Returns `0.0` if either slice is empty (defensive — the caller has
/// already asserted non-empty length above).
fn rms_diff_over_prefix(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let sum_sq: f64 = a
        .iter()
        .take(n)
        .zip(b.iter().take(n))
        .map(|(x, y)| {
            let d = (*x as f64) - (*y as f64);
            d * d
        })
        .sum();
    (sum_sq / n as f64).sqrt() as f32
}

/// Looks up `key` in JSON object `v`, panicking with `ctx` on a missing key
/// (an opted-in fixture set is expected to be *complete* — a hole in it is
/// a hard failure, not a default-and-continue).
fn json_get<'v>(v: &'v JsonValue, key: &str, ctx: &str) -> &'v JsonValue {
    v.get(key)
        .unwrap_or_else(|| panic!("{ctx}: missing JSON key `{key}`"))
}

/// Reads `v[key]` as a JSON string.
fn json_str<'v>(v: &'v JsonValue, key: &str, ctx: &str) -> &'v str {
    json_get(v, key, ctx)
        .as_str()
        .unwrap_or_else(|| panic!("{ctx}: `{key}` is not a JSON string"))
}

/// Reads `v[key]` as a non-negative JSON integer.
fn json_u64(v: &JsonValue, key: &str, ctx: &str) -> u64 {
    json_get(v, key, ctx)
        .as_u64()
        .unwrap_or_else(|| panic!("{ctx}: `{key}` is not a non-negative integer"))
}

/// Reads `v[key]` as a non-negative JSON integer narrowed to `u32`.
fn json_u32(v: &JsonValue, key: &str, ctx: &str) -> u32 {
    let n = json_u64(v, key, ctx);
    u32::try_from(n).unwrap_or_else(|_| panic!("{ctx}: `{key}` = {n} does not fit in u32"))
}

/// Reads `v[key]` as a JSON number (`Int` or `Float`), narrowed to `f32`.
fn json_f32(v: &JsonValue, key: &str, ctx: &str) -> f32 {
    match json_get(v, key, ctx) {
        JsonValue::Int(i) => *i as f32,
        JsonValue::Float(f) => *f as f32,
        other => panic!("{ctx}: `{key}` is not a JSON number: {other:?}"),
    }
}

/// Reads `v[key]` as a JSON array of numbers, narrowed element-wise to `f32`.
fn json_f32_array(v: &JsonValue, key: &str, ctx: &str) -> Vec<f32> {
    json_get(v, key, ctx)
        .as_array()
        .unwrap_or_else(|| panic!("{ctx}: `{key}` is not a JSON array"))
        .iter()
        .map(|item| match item {
            JsonValue::Int(i) => *i as f32,
            JsonValue::Float(f) => *f as f32,
            other => panic!("{ctx}: `{key}` array element is not a JSON number: {other:?}"),
        })
        .collect()
}

/// Looks up the `tensors[]` entry named `name` in `manifest`, returning its
/// `(path, shape)` pair — `path` relative to [`fixtures_dir`] (already
/// carrying the `reference_dump/` prefix, per the module doc's manifest
/// schema), `shape` as a flat `u64` dimension list.
fn find_tensor<'v>(manifest: &'v JsonValue, name: &str, ctx: &str) -> (&'v str, Vec<u64>) {
    let tensors = json_get(manifest, "tensors", ctx)
        .as_array()
        .unwrap_or_else(|| panic!("{ctx}: `tensors` is not a JSON array"));
    let entry = tensors
        .iter()
        .find(|t| json_str(t, "name", ctx) == name)
        .unwrap_or_else(|| {
            panic!(
                "{ctx}: `tensors` has no entry named `{name}` — see this file's module doc \
                 for the required manifest schema"
            )
        });
    let path = json_str(entry, "path", ctx);
    let shape = json_get(entry, "shape", ctx)
        .as_array()
        .unwrap_or_else(|| panic!("{ctx}: tensors[name={name}].shape is not a JSON array"))
        .iter()
        .map(|elem| {
            elem.as_u64().unwrap_or_else(|| {
                panic!("{ctx}: tensors[name={name}].shape element is not a non-negative integer")
            })
        })
        .collect();
    (path, shape)
}

/// Task 7: reads the manifest's `phonemize_fixture` block plus the three
/// side files it names, and builds a single-entry [`PhonemizeFixture`]
/// keyed on `(request.language, request.text)` — the `(language, text)`
/// pair the Python reference dumper's forward pass used.
///
/// The returned fixture is exactly what a subsequent
/// [`SbV2Model::from_gguf_with_phonemizer`] call needs to reproduce the
/// reference G2P output byte-for-byte: [`SbV2Phonemizer::from_fixture`]
/// wraps it, and [`SbV2Model::synthesize`] looks up the same
/// `(language, text)` pair internally. Any inconsistency in the fixture
/// (dtype in the manifest disagrees with what this reader knows how to
/// parse, or the three side files have different lengths) is a loud panic
/// — the fixture must be internally consistent before the numeric-parity
/// assertion is trustworthy (FR-EX-08).
fn phonemize_fixture_from_manifest(
    manifest: &JsonValue,
    request: &SbV2SynthRequest,
    dir: &Path,
    ctx: &str,
) -> PhonemizeFixture {
    let block = json_get(manifest, "phonemize_fixture", ctx);
    let entry_ctx = format!("{ctx}: phonemize_fixture");

    // Small local helpers: pull `{path, dtype}` from one named entry.
    let entry_path_dtype = |name: &str| -> (PathBuf, String) {
        let entry = json_get(block, name, &entry_ctx);
        let rel = json_str(entry, "path", &entry_ctx);
        let dtype = json_str(entry, "dtype", &entry_ctx);
        (dir.join(rel), dtype.to_string())
    };

    // Read the three side files, dispatching on dtype so a manifest
    // schema drift (e.g. a later dumper widening to u32) is caught here
    // rather than silently mis-parsed. Only the exact dtypes
    // `PHONEMIZE_FIXTURE_SCHEMA` declares are accepted; anything else
    // panics with the offending name+dtype.
    let (pids_path, pids_dtype) = entry_path_dtype("phoneme_ids");
    require_fixture(&pids_path, "phonemize_fixture.phoneme_ids (Task 30 dump)");
    let phoneme_ids: Vec<u16> = match pids_dtype.as_str() {
        "uint16" => read_u16_bin(&pids_path),
        other => panic!(
            "{entry_ctx}: phoneme_ids dtype {other:?} is not supported by this reader (expected \
             \"uint16\" per PHONEMIZE_FIXTURE_SCHEMA in tools/parity/sbv2_dump_reference.py)"
        ),
    };

    let (tones_path, tones_dtype) = entry_path_dtype("tones");
    require_fixture(&tones_path, "phonemize_fixture.tones (Task 30 dump)");
    let tones: Vec<u8> = match tones_dtype.as_str() {
        "uint8" => read_u8_bin(&tones_path),
        other => panic!(
            "{entry_ctx}: tones dtype {other:?} is not supported by this reader (expected \
             \"uint8\" per PHONEMIZE_FIXTURE_SCHEMA in tools/parity/sbv2_dump_reference.py)"
        ),
    };

    let (wb_path, wb_dtype) = entry_path_dtype("word_boundaries");
    require_fixture(&wb_path, "phonemize_fixture.word_boundaries (Task 30 dump)");
    let word_boundaries: Vec<bool> = match wb_dtype.as_str() {
        "uint8" => read_u8_bin(&wb_path).into_iter().map(|b| b != 0).collect(),
        other => panic!(
            "{entry_ctx}: word_boundaries dtype {other:?} is not supported by this reader \
             (expected \"uint8\" per PHONEMIZE_FIXTURE_SCHEMA in \
             tools/parity/sbv2_dump_reference.py)"
        ),
    };

    // Cross-file consistency: the three side files describe the SAME
    // G2P output, so their lengths must all equal T_text.
    assert_eq!(
        tones.len(),
        phoneme_ids.len(),
        "{entry_ctx}: tones.len() ({}) != phoneme_ids.len() ({}) — the three side files must \
         describe the same G2P output for the same input text",
        tones.len(),
        phoneme_ids.len(),
    );
    assert_eq!(
        word_boundaries.len(),
        phoneme_ids.len(),
        "{entry_ctx}: word_boundaries.len() ({}) != phoneme_ids.len() ({}) — the three side \
         files must describe the same G2P output for the same input text",
        word_boundaries.len(),
        phoneme_ids.len(),
    );

    let result = PhonemizeResult {
        phoneme_ids,
        tones,
        word_boundaries,
        // The Python dumper passes `text` unmodified as its BERT input text
        // (the `PhonemizeResult::bert_input_text` contract on the Rust side
        // is also "the original input text, passed through" — see g2p.rs's
        // struct doc). Reproduce that here.
        bert_input_text: request.text.clone(),
    };

    let mut fixture = PhonemizeFixture::new();
    fixture.insert(request.language, request.text.clone(), result);
    fixture
}

/// Builds the [`SbV2SynthRequest`] the Python dumper's reference forward
/// pass used, from the manifest's `request` object (see the module doc's
/// schema).
///
/// M6 refactor (2026-08-06): `"ZH"` is now accepted for
/// `request.language` because [`Language`] gained a `ZH` variant to match
/// the real SBV2 v2 checkpoint's `enc_p.language_emb.weight [3, d_model]`
/// row 2. Note that under the fixture-only bypass this test uses, ZH is
/// reachable in the text encoder but will loud-fail at the BERT
/// tokenizer step (see `SbV2Model::synthesize`'s ZH note); a real ZH
/// parity run requires the pending ZH BERT + G2P plumbing.
///
/// FORWARD POINTER (WP-21 doc sweep, 2026-08-10): the "pending ZH BERT +
/// G2P plumbing" above is now concretely scoped by the 2026-08-09 owner
/// decisions — ZH BERT = `hfl/chinese-roberta-wwm-ext-large` (Apache-2.0,
/// standard `BertForMaskedLM`, NOT DeBERTa) and ZH G2P = piper-plus reuse
/// via `integrations/vokra-piper-g2p`. WP-17 landed the WordPiece
/// tokenizer that pairs with the ZH BERT encoder; the remaining
/// `BertBaseEncoder` + `SbV2Phonemizer::from_piper_g2p` ZH-wiring gap-fill
/// is a later WP, plus owner CI fixture regeneration (WP-20) and owner
/// §3.1 license sign-off before any HF publish.
fn request_from_manifest(manifest: &JsonValue, ctx: &str) -> SbV2SynthRequest {
    let request = json_get(manifest, "request", ctx);
    let language = match json_str(request, "language", ctx) {
        "JA" => Language::JA,
        "EN" => Language::EN,
        // WP-18: fixture manifests can declare ZH so future ZH reference
        // dumps can flow through this same loader; note that
        // `SbV2Model::synthesize` currently fail-closes on Language::ZH
        // until the ZH BERT WP lands (WP-19+).
        "ZH" => Language::ZH,
        other => {
            panic!("{ctx}: request.language must be \"JA\", \"EN\", or \"ZH\", got {other:?}")
        }
    };
    SbV2SynthRequest {
        text: json_str(request, "text", ctx).to_string(),
        language,
        speaker_id: json_u32(request, "speaker_id", ctx),
        // Blocker 3: the Python reference dumper for the real ckpt
        // currently exercises the deterministic zero-shot default —
        // `None` here forwards that intent to
        // `SbV2Model::synthesize`'s step 5, which then uses the
        // all-zero `[d_speaker]` external default (matching
        // `SynthesisRequest::speaker_embedding`'s "None = zero" doc).
        // Extending the manifest schema to carry an explicit reference
        // 512-d embedding is a follow-up: the ckpt loader itself is
        // what must land first (`sbv2.text_encoder.spk_emb_linear.*`
        // Rename + `SbV2Model::from_gguf` binding), and that is scope
        // for the Blocker 3 converter-side wave.
        speaker_embedding: None,
        style_vec: json_f32_array(request, "style_vec", ctx),
        speed: json_f32(request, "speed", ctx),
        noise_scale: json_f32(request, "noise_scale", ctx),
        noise_scale_w: json_f32(request, "noise_scale_w", ctx),
        seed: json_u64(request, "seed", ctx),
        // Real-parity test: the Python reference dumper must use
        // PhiloxRNGEngine.h (via `tools/parity/torch_philox_dump.py`'s
        // shared port) so its noise buffer byte-matches Vokra's
        // `TorchRandnStream`. Default = torch parity, which is exactly
        // that path.
        rng_mode: RngMode::default(),
    }
}

/// Renders one [`AtolCalibration`] variant to a short marker string used
/// in the per-tensor stderr summary the parity harness emits. `waveform`
/// is `[Measured]`, the 4 pre-fixture bounds are `[EstimatedPreFixture]`,
/// and the 6 pass-throughs are `[UnmeasuredDefault(ATOL_DEFAULT)]` — a
/// CI viewer sees the calibration status alongside each `max |Δ|` row.
fn calibration_marker(name: &str) -> &'static str {
    match atol_calibration_for(name) {
        Some(AtolCalibration::Measured) => "[Measured]",
        Some(AtolCalibration::EstimatedPreFixture) => "[EstimatedPreFixture]",
        Some(AtolCalibration::UnmeasuredDefault) => "[UnmeasuredDefault(ATOL_DEFAULT)]",
        None => "[UNPINNED]",
    }
}

/// WP-06 (Wave 0 Task 6, 2026-08-11): one row of the all-stage aggregated
/// parity report this test builds. Every named parity gate — the ~10
/// [`SbV2Intermediates::to_dumper_map`] tensors (via
/// [`diff_intermediates_against_manifest`]), and the waveform length-band,
/// max|Δ|/RMS checks, and mel-loss aggregator (all three in the caller,
/// [`parity_sbv2_real_waveform_matches_reference_dump`]) — becomes exactly
/// one `StageResult`, collected into a single `Vec` and asserted ONCE at
/// the very end of the test instead of via N separate early-exit
/// `assert!`s.
///
/// This matters because a real-fixture run's first purpose is
/// *diagnostic*: which of the ~13 pipeline stages diverge, and by how
/// much? A test that panics on stage 3 of 13 hides stages 4-13 from the
/// very same run that could have reported them, forcing an
/// iterate-fix-rerun cycle to discover each subsequent failure one at a
/// time. See `docs/handoff/sbv2-parity-baseline-2026-08-11.md` for the
/// first real trace this produced.
///
/// A `StageResult` is always constructed — never a panic path — even a
/// "structural" failure (missing fixture file, shape mismatch, non-f32-
/// aligned payload) becomes a `FAIL` row with `max_abs_diff` / `rms_diff`
/// = `NaN` and [`Self::note`] set, explaining why no numeric diff was
/// possible, rather than aborting the whole harness before later stages
/// run. `NaN` is a deliberate sentinel: `NaN <= atol` is always `false`
/// (IEEE 754), so `passed` is correctly `false` without a special case,
/// and the downstream `tools/parity/sbv2_atol_summary_from_log.py` parser
/// gracefully warns-and-skips a `max_diff` it cannot parse as a float
/// rather than mis-recording a fabricated number `sbv2_atol_updater.py`
/// might otherwise propose as a real bound.
struct StageResult {
    stage: &'static str,
    atol_expected: f32,
    max_abs_diff: f32,
    rms_diff: f32,
    passed: bool,
    /// Calibration marker (`[Measured]` / `[EstimatedPreFixture]` / ...)
    /// from [`calibration_marker`], or `"[N/A]"` for a pseudo-stage that
    /// has no manifest tensor of its own (currently only
    /// `waveform_length_band`).
    marker: &'static str,
    /// `Some(reason)` only for a structural failure (see the type doc) —
    /// `None` means the stage got as far as computing a real numeric
    /// diff, whether or not it passed its atol.
    note: Option<String>,
}

impl StageResult {
    /// Gates only on `max_abs_diff <= atol_expected` — the convention
    /// every intermediate tensor, the waveform length-band check, and
    /// mel_loss use (module doc: "Intermediate tensors use a simpler
    /// per-tensor `max_abs_diff` gate"). `rms_diff` is still recorded for
    /// the summary line but does not affect `passed`.
    fn numeric(
        stage: &'static str,
        max_abs_diff: f32,
        rms_diff: f32,
        atol_expected: f32,
        marker: &'static str,
    ) -> Self {
        Self {
            stage,
            atol_expected,
            max_abs_diff,
            rms_diff,
            passed: max_abs_diff <= atol_expected,
            marker,
            note: None,
        }
    }

    /// Gates on BOTH `max_abs_diff <= atol_expected` AND `rms_diff <=
    /// atol_expected` — the convention the pre-WP-06 waveform block used
    /// (two separate `assert!`s against the same `tolerance_for("waveform")`
    /// ceiling). Only the `"waveform"` stage uses this constructor.
    fn numeric_dual_gate(
        stage: &'static str,
        max_abs_diff: f32,
        rms_diff: f32,
        atol_expected: f32,
        marker: &'static str,
    ) -> Self {
        Self {
            stage,
            atol_expected,
            max_abs_diff,
            rms_diff,
            passed: max_abs_diff <= atol_expected && rms_diff <= atol_expected,
            marker,
            note: None,
        }
    }

    /// A stage that could not even be diffed — see the type doc's `NaN`
    /// sentinel rationale. Always `passed == false`.
    fn structural_failure(
        stage: &'static str,
        atol_expected: f32,
        marker: &'static str,
        reason: String,
    ) -> Self {
        Self {
            stage,
            atol_expected,
            max_abs_diff: f32::NAN,
            rms_diff: f32::NAN,
            passed: false,
            marker,
            note: Some(reason),
        }
    }

    /// Emits both the CI-parser-stable `[parity_sbv2_real] <name>: ...`
    /// row (format UNCHANGED from the pre-WP-06 shape — see
    /// `tools/parity/sbv2_atol_summary_from_log.py`'s `ROW_RE`, which
    /// `sbv2_atol_updater.py` consumes downstream) and a second,
    /// human-readable `[PASS/FAIL] stage — ...` summary line carrying the
    /// `rms_diff` the machine-parseable row does not.
    fn emit(&self) {
        let verdict = if self.passed { "PASS" } else { "FAIL" };
        let note_suffix = self
            .note
            .as_ref()
            .map(|n| format!(" NOTE: {n}"))
            .unwrap_or_default();
        eprintln!(
            "[parity_sbv2_real] {}: max |Δ| = {:.6e} atol {} verdict {verdict} {}{note_suffix}",
            self.stage, self.max_abs_diff, self.atol_expected, self.marker,
        );
        eprintln!(
            "  [{verdict}] {} — max_abs={:.6e} atol={:.6e} rms={:.6e}",
            self.stage, self.max_abs_diff, self.atol_expected, self.rms_diff,
        );
    }
}

/// Non-panicking twin of [`find_tensor`] for
/// [`diff_intermediates_against_manifest`]'s aggregated loop: `Ok` mirrors
/// `find_tensor`'s return exactly; `Err(reason)` describes what went
/// wrong so the caller can record a [`StageResult::structural_failure`]
/// and move on to the NEXT stage instead of aborting the whole harness.
/// The harness-wide preconditions (manifest itself missing, checkpoint
/// GGUFs missing) stay on [`find_tensor`] / [`require_fixture`]'s
/// hard-panic path — only the PER-TENSOR lookups inside the aggregated
/// loop below need a fallible variant, so one drifted manifest entry
/// doesn't hide the other ~9 tensors' results.
fn try_find_tensor<'v>(manifest: &'v JsonValue, name: &str) -> Result<(&'v str, Vec<u64>), String> {
    let tensors = manifest
        .get("tensors")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| "manifest `tensors` is missing or not a JSON array".to_string())?;
    let entry = tensors
        .iter()
        .find(|t| t.get("name").and_then(JsonValue::as_str) == Some(name))
        .ok_or_else(|| format!("no `tensors[]` entry named `{name}` in the manifest"))?;
    let path = entry
        .get("path")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| format!("tensors[name={name}] has no string `path`"))?;
    let shape_arr = entry
        .get("shape")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| format!("tensors[name={name}].shape is not a JSON array"))?;
    let mut shape = Vec::with_capacity(shape_arr.len());
    for elem in shape_arr {
        let v = elem.as_u64().ok_or_else(|| {
            format!("tensors[name={name}].shape element is not a non-negative integer")
        })?;
        shape.push(v);
    }
    Ok((path, shape))
}

/// WP-01 (2026-08-09) / WP-06 (2026-08-11, all-stage aggregation): iterates
/// every intermediate tensor from [`SbV2Intermediates::to_dumper_map`] and
/// diffs it against the corresponding `reference_dump/<name>.bin` fixture
/// named in the manifest, returning ONE [`StageResult`] per tensor rather
/// than panicking.
///
/// Contract (unchanged from WP-01 — only the control flow changed, per
/// [`StageResult`]'s doc):
/// - Every dumper-map entry SHOULD have a matching manifest `tensors[]`
///   entry; a miss is now recorded as a [`StageResult::structural_failure`]
///   row rather than a panic, so the OTHER tensors still get diffed in
///   the same run (the harness expects the two to stay in sync — drift
///   is a real fixture bug, not a soft-skip; it is simply no longer a
///   whole-run-aborting one).
/// - Every fixture's `.bin` byte length SHOULD match the manifest's
///   declared shape product AND the intermediate's own element count. A
///   mismatch is likewise a structural-failure row, not a panic.
/// - The per-element `max_abs_diff` MUST be `<= tolerance_for(name)` for
///   `passed` to be `true` — this numeric verdict is unchanged from
///   WP-01; only WHEN the caller learns about a `false` verdict changed
///   (deferred to the caller's single end-of-run assert instead of an
///   in-function panic).
///
/// Emits `[parity_sbv2_real] <name>: max |Δ| = ... atol ... verdict ...`
/// to stderr for each tensor (format UNCHANGED — CI-parser-stable), PLUS
/// a second `[PASS/FAIL] <name> — max_abs=... atol=... rms=...` line per
/// [`StageResult::emit`].
///
/// The `waveform` diff stays outside this loop — its length is
/// discretely dependent on SDP durations and needs the tolerance-based
/// length-band + max-diff + RMS-on-overlap gate (see the caller, which
/// appends its own [`StageResult`]s to this function's return value).
/// Every OTHER manifest tensor's shape is pinned exactly by the manifest
/// schema.
fn diff_intermediates_against_manifest(
    manifest: &JsonValue,
    intermediates: &SbV2Intermediates,
    dir: &Path,
) -> Vec<StageResult> {
    let mut results: Vec<StageResult> = Vec::new();
    for (name, rust_bytes) in intermediates.to_dumper_map() {
        let marker = calibration_marker(name);
        let atol = tolerance_for(name);

        let (rel, shape) = match try_find_tensor(manifest, name) {
            Ok(v) => v,
            Err(reason) => {
                let row = StageResult::structural_failure(name, atol, marker, reason);
                row.emit();
                results.push(row);
                continue;
            }
        };
        let ref_path = dir.join(rel);
        if !ref_path.exists() {
            let row = StageResult::structural_failure(
                name,
                atol,
                marker,
                format!(
                    "missing fixture: {} (tensors[name={name}].path, Task 30 dump)",
                    ref_path.display()
                ),
            );
            row.emit();
            results.push(row);
            continue;
        }
        let bytes = match std::fs::read(&ref_path) {
            Ok(b) => b,
            Err(e) => {
                let row = StageResult::structural_failure(
                    name,
                    atol,
                    marker,
                    format!("{}: {e}", ref_path.display()),
                );
                row.emit();
                results.push(row);
                continue;
            }
        };
        if bytes.len() % 4 != 0 {
            let row = StageResult::structural_failure(
                name,
                atol,
                marker,
                format!(
                    "{}: byte length {} is not a multiple of 4 (not f32-aligned)",
                    ref_path.display(),
                    bytes.len()
                ),
            );
            row.emit();
            results.push(row);
            continue;
        }
        let reference: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let declared_len: u64 = shape.iter().product();
        if reference.len() as u64 != declared_len {
            let row = StageResult::structural_failure(
                name,
                atol,
                marker,
                format!(
                    "{}: byte length ({} f32 elements) disagrees with the manifest's \
                     declared `shape` {shape:?} — Task 34/30's dumper produced an \
                     inconsistent fixture",
                    ref_path.display(),
                    reference.len()
                ),
            );
            row.emit();
            results.push(row);
            continue;
        }

        // `to_dumper_map`'s `Vec<u8>` payload is little-endian f32 (its
        // `f32_bytes` helper). A misalignment here is an internal
        // Wave-4 INTERMEDIATE-ACCESSORS contract violation, not a
        // fixture problem — still recorded as a structural failure
        // rather than a panic so sibling tensors keep getting checked.
        if rust_bytes.len() % 4 != 0 {
            let row = StageResult::structural_failure(
                name,
                atol,
                marker,
                format!(
                    "SbV2Intermediates::to_dumper_map(`{name}`) payload {} bytes is not \
                     f32-aligned — Wave-4 INTERMEDIATE-ACCESSORS contract violation",
                    rust_bytes.len()
                ),
            );
            row.emit();
            results.push(row);
            continue;
        }
        let rust: Vec<f32> = rust_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if rust.len() != reference.len() {
            let row = StageResult::structural_failure(
                name,
                atol,
                marker,
                format!(
                    "Rust `{name}` len {} != reference len {} — shape mismatch is a real \
                     pipeline bug, not a numeric drift (the manifest fixes each \
                     intermediate's shape exactly; unlike waveform, no discrete-step ±1 \
                     flip applies)",
                    rust.len(),
                    reference.len()
                ),
            );
            row.emit();
            results.push(row);
            continue;
        }

        let diff = max_abs_diff(&rust, &reference);
        let rms = rms_diff_over_prefix(&rust, &reference);
        let row = StageResult::numeric(name, diff, rms, atol, marker);
        row.emit();
        results.push(row);
    }
    results
}

/// WP-24 UTMOS opt-in env vars, parsed once via [`UtmosGateSettings::resolve`]
/// so the fail-closed decision is centralized (loud when only one of the
/// two is set — "opted in but no scorer" is not a skip).
///
/// - `VOKRA_SBV2_UTMOS_ENABLE`: `"1"` opts in. Any other value (including
///   unset / `"0"` / `"false"`) skips the leg entirely.
/// - `VOKRA_SBV2_UTMOS_GGUF`: filesystem path to a `vokra.utmos.*` GGUF
///   (see `parity-utmos.yml` for the CI-side conversion recipe, or
///   `tests/parity/utmos/README.md` for the local recipe). Required when
///   `ENABLE=1`; unset with `ENABLE=1` is a loud panic (broken opt-in).
///
/// Environment variable name for opting in to UTMOS quality gate (WP-24).
pub const ENV_UTMOS_ENABLE: &str = "VOKRA_SBV2_UTMOS_ENABLE";

/// Environment variable name for the path to the UTMOS22-strong GGUF (WP-24).
pub const ENV_UTMOS_GGUF: &str = "VOKRA_SBV2_UTMOS_GGUF";

/// Resolved WP-24 UTMOS opt-in state; see [`UtmosGateSettings::resolve`].
pub enum UtmosGateSettings {
    /// The gate is off (env var unset or explicitly `"0"` / `"false"`).
    /// `utmos_gate` will emit a skip diagnostic and return without
    /// scoring.
    Disabled,
    /// The gate is on and the GGUF path is populated. `utmos_gate` will
    /// panic loudly on any downstream failure — the caller explicitly
    /// opted in, so a broken opt-in is a hard failure.
    Enabled {
        /// Path to the UTMOS GGUF file.
        gguf_path: PathBuf,
    },
}

impl UtmosGateSettings {
    /// Reads both env vars and dispatches:
    /// - `ENABLE` unset / not `"1"` → [`Self::Disabled`] (silent skip);
    /// - `ENABLE=1` **and** `GGUF` set to a non-empty string →
    ///   [`Self::Enabled`];
    /// - `ENABLE=1` **without** `GGUF` (or with an empty string) → loud
    ///   panic (a broken opt-in must never masquerade as a skip,
    ///   FR-EX-08).
    pub fn resolve() -> Self {
        let enabled = std::env::var(ENV_UTMOS_ENABLE)
            .ok()
            .is_some_and(|v| v == "1");
        if !enabled {
            return Self::Disabled;
        }
        let gguf_path = std::env::var_os(ENV_UTMOS_GGUF).unwrap_or_else(|| {
            panic!(
                "{ENV_UTMOS_ENABLE}=1 but {ENV_UTMOS_GGUF} is unset — the WP-24 UTMOS quality \
                 gate needs a `vokra.utmos.*` GGUF path to score against. A broken opt-in must \
                 not look like a skip (FR-EX-08). Convert one with `parity-utmos.yml`'s recipe \
                 (or the local one in `tests/parity/utmos/README.md`) and re-run this test \
                 with `{ENV_UTMOS_GGUF}=<path>` set."
            )
        });
        let path = PathBuf::from(&gguf_path);
        assert!(
            !path.as_os_str().is_empty(),
            "{ENV_UTMOS_ENABLE}=1 but {ENV_UTMOS_GGUF} is empty — see the panic message above"
        );
        Self::Enabled { gguf_path: path }
    }
}

/// WP-24: runs the tail-position UTMOS quality gate over the Rust and
/// reference waveforms once the numerical waveform parity has already
/// passed. Fail-closed opt-in: see [`UtmosGateSettings`] and the module
/// doc's "WP-24: UTMOS quality gate" section.
///
/// Panics on any downstream failure (missing/unreadable/synthesized GGUF,
/// resample failure, scorer error, delta over [`UTMOS_ATOL`]) — the
/// caller explicitly opted in, so every failure is a real regression, not
/// a skip.
fn utmos_gate(rust_wave: &[f32], reference_wave: &[f32], sbv2_sample_rate: u32) {
    let settings = UtmosGateSettings::resolve();
    let UtmosGateSettings::Enabled { gguf_path } = settings else {
        eprintln!(
            "[parity_sbv2_real] UTMOS quality gate: SKIPPED ({ENV_UTMOS_ENABLE} not set to \"1\"). \
             This is an FR-EX-08 explicit skip, not a fabricated pass. Set \
             `{ENV_UTMOS_ENABLE}=1 {ENV_UTMOS_GGUF}=<path/to/utmos.gguf>` to opt in — see the \
             module doc's \"WP-24: UTMOS quality gate\" section for the recipe."
        );
        return;
    };

    // Cross-input sanity: both waveforms must be the same length here (the
    // caller has just asserted that above, but re-checking makes the
    // panic message point at the right root cause if a future edit moves
    // things around).
    assert_eq!(
        rust_wave.len(),
        reference_wave.len(),
        "[parity_sbv2_real] UTMOS gate: rust/reference length mismatch ({} vs {}) — the \
         waveform-parity assertion above should have caught this first",
        rust_wave.len(),
        reference_wave.len(),
    );

    // Load the scorer. Refusing synthesized weights (fabricated-pass ban,
    // NFR-QL-04) mirrors `parity_utmos.rs`'s `native_score_for_parity`.
    let scorer = Utmos::from_path(&gguf_path).unwrap_or_else(|e| {
        panic!(
            "[parity_sbv2_real] UTMOS gate: failed to load `vokra.utmos.*` GGUF from {}: {e}",
            gguf_path.display()
        )
    });
    assert!(
        !scorer.is_synthesized(),
        "[parity_sbv2_real] UTMOS gate: refusing to score against a synthesized-weight \
         UTMOS GGUF ({}) — a synthetic scorer cannot honestly measure perceptual parity \
         against a real waveform (NFR-QL-04). Point {ENV_UTMOS_GGUF} at a GGUF converted from \
         the real upstream UTMOS22-strong checkpoint (see `parity-utmos.yml`'s recipe).",
        gguf_path.display(),
    );

    // UTMOS22-strong is 16 kHz; SBV2 is 44.1 kHz. The metric refuses
    // silent resampling, so downsample both waveforms explicitly (same
    // resampler path for both = the rounding cancels in the absolute
    // delta comparison below). Equal rates are a bit-exact no-op inside
    // `vokra_ops::resample`, so this is safe even in the (unusual) case
    // where SBV2 and the UTMOS GGUF happen to share a rate.
    let target_sr = scorer.config().sample_rate;
    let rust_at_target = if target_sr == sbv2_sample_rate {
        rust_wave.to_vec()
    } else {
        vokra_ops::resample(
            rust_wave,
            sbv2_sample_rate,
            target_sr,
            vokra_ops::resample::DEFAULT_QUALITY,
        )
        .unwrap_or_else(|e| {
            panic!(
                "[parity_sbv2_real] UTMOS gate: resample of rust_wave from {sbv2_sample_rate} \
                 Hz to {target_sr} Hz failed: {e}"
            )
        })
    };
    let reference_at_target = if target_sr == sbv2_sample_rate {
        reference_wave.to_vec()
    } else {
        vokra_ops::resample(
            reference_wave,
            sbv2_sample_rate,
            target_sr,
            vokra_ops::resample::DEFAULT_QUALITY,
        )
        .unwrap_or_else(|e| {
            panic!(
                "[parity_sbv2_real] UTMOS gate: resample of reference_wave from \
                 {sbv2_sample_rate} Hz to {target_sr} Hz failed: {e}"
            )
        })
    };

    let score_rust = scorer
        .score(&rust_at_target, target_sr)
        .unwrap_or_else(|e| panic!("[parity_sbv2_real] UTMOS gate: rust waveform scoring: {e}"));
    let score_reference = scorer
        .score(&reference_at_target, target_sr)
        .unwrap_or_else(|e| {
            panic!("[parity_sbv2_real] UTMOS gate: reference waveform scoring: {e}")
        });

    let delta = (score_rust - score_reference).abs();
    assert!(
        delta <= UTMOS_ATOL,
        "[parity_sbv2_real] UTMOS gate FAILED: |utmos(rust) - utmos(reference)| = {delta:.6e} \
         > UTMOS_ATOL {UTMOS_ATOL:.6e} (rust = {score_rust}, reference = {score_reference}, \
         resampled from {sbv2_sample_rate} Hz to {target_sr} Hz for scoring). If the divergence \
         is an architectural bound — not a real regression — record a per-fixture honest atol \
         with rationale in `sbv2::parity`'s module doc (Kokoro `PROSODY_F0_ATOL` precedent), \
         never widen `UTMOS_ATOL` itself to hunt a green."
    );
    eprintln!(
        "[parity_sbv2_real] UTMOS quality gate OK: |Δ| = {delta:.6e} <= {UTMOS_ATOL:.6e} \
         (rust = {score_rust}, reference = {score_reference}, resampled {sbv2_sample_rate} Hz \
         → {target_sr} Hz)"
    );
}

/// Real-checkpoint SBV2 full-manifest + UTMOS parity, gated on the Task 34 + Task 30
/// fixture set (`tests/fixtures/sbv2/`). See the module doc for: the
/// manifest schema this reads (including the Task 7 `phonemize_fixture`
/// block that bypasses the missing in-workspace 8-language G2P), the
/// WP-01 (2026-08-09) full-manifest iteration built on
/// `SbV2Model::synthesize_with_intermediates` +
/// `SbV2Intermediates::to_dumper_map`, and how the G2P bypass wires
/// `SbV2Model::from_gguf_with_phonemizer` +
/// `SbV2Phonemizer::from_fixture` together.
///
/// WP-06 (Wave 0 Task 6, 2026-08-11): no longer `#[ignore]`d — the
/// `tests/fixtures/sbv2/` fixture set landed for real, so this test now
/// runs under plain `cargo test` like any other. Every parity stage
/// (intermediates + waveform + mel_loss) is collected into one
/// `Vec<StageResult>` and asserted ONCE at the end, so a single run
/// reports every stage's PASS/FAIL — see [`StageResult`]'s doc.
#[test]
fn parity_sbv2_real_waveform_matches_reference_dump() {
    let dir = fixtures_dir();

    let manifest_path = dir.join("reference_dump.manifest.json");
    require_fixture(&manifest_path, "reference_dump.manifest.json (Task 34)");
    let manifest_bytes = std::fs::read(&manifest_path)
        .unwrap_or_else(|e| panic!("{}: {e}", manifest_path.display()));
    let manifest = json::parse(&manifest_bytes)
        .unwrap_or_else(|e| panic!("{}: JSON parse error: {e}", manifest_path.display()));
    let ctx = manifest_path.display().to_string();

    let checkpoint = json_get(&manifest, "checkpoint", &ctx);
    let main_path = dir.join(json_str(checkpoint, "sbv2_main", &ctx));
    let bert_ja_path = dir.join(json_str(checkpoint, "bert_ja", &ctx));
    let bert_en_path = dir.join(json_str(checkpoint, "bert_en", &ctx));
    require_fixture(
        &main_path,
        "checkpoint.sbv2_main (Task 34, converted via Task 25)",
    );
    require_fixture(
        &bert_ja_path,
        "checkpoint.bert_ja (Task 34, converted via vokra-bert)",
    );
    require_fixture(
        &bert_en_path,
        "checkpoint.bert_en (Task 34, converted via vokra-bert)",
    );

    let main =
        GgufFile::open(&main_path).unwrap_or_else(|e| panic!("{}: {e}", main_path.display()));
    let bert_ja =
        GgufFile::open(&bert_ja_path).unwrap_or_else(|e| panic!("{}: {e}", bert_ja_path.display()));
    let bert_en =
        GgufFile::open(&bert_en_path).unwrap_or_else(|e| panic!("{}: {e}", bert_en_path.display()));

    let req = request_from_manifest(&manifest, &ctx);

    // Task 7: build the fixture G2P from the manifest's phonemize_fixture
    // block (three typed side files) + assemble it into an
    // SbV2Phonemizer::from_fixture; hand it to
    // SbV2Model::from_gguf_with_phonemizer so `synthesize` below actually
    // runs (rather than the UnwiredPhonemizer's FR-EX-08 loud refusal
    // SbV2Model::from_gguf installs by default).
    let fixture = phonemize_fixture_from_manifest(&manifest, &req, &dir, &ctx);
    let phonemizer = SbV2Phonemizer::from_fixture(fixture);
    let model = SbV2Model::from_gguf_with_phonemizer(&main, &bert_ja, &bert_en, phonemizer)
        .unwrap_or_else(|e| panic!("SbV2Model::from_gguf_with_phonemizer: {e}"));

    // Blocker 3 close-out (2026-08-10): the SBV2 v2 base ckpt
    // (`litagin/Style-Bert-VITS2-2.0-base-JP-Extra` and its
    // downstream fine-tunes) has **no per-speaker embedding table** —
    // speaker conditioning enters through `enc_p.encoder.spk_emb_linear`
    // which the Blocker 1 converter mapping table renames to
    // `sbv2.text_encoder.spk_emb_linear.{weight,bias}` and the
    // `SbV2Model::from_gguf` speaker section (mod.rs
    // `---- speaker ----`) binds as an `ExternalSpeakerProjection`.
    //
    // Assert-loudly here that the projection actually bound at load
    // time. A converter regression that silently drops the pair (or
    // renames it under a different tensor path) would otherwise only
    // surface as a subtle waveform drift downstream — and even then
    // only when a caller supplies a non-`None` speaker_embedding on the
    // request. FR-EX-08 prefers a loud, named-cause failure at load
    // time, and `model.speaker_projection()` (see that method's doc)
    // gives us exactly that observability.
    //
    // The current manifest sets `speaker_embedding: None` (see
    // `request_from_manifest`'s "Blocker 3" comment above), so the
    // projection is not exercised on the forward pass — but that is
    // fine for this assertion: we only need to confirm the tensors
    // bound, not that they influenced the waveform. Extending the
    // manifest schema to carry an explicit reference 512-d embedding
    // (which would exercise the projection on the forward pass) is
    // a follow-up.
    assert!(
        model.speaker_projection().is_some(),
        "{ctx}: expected the SBV2 v2 real-ckpt loader to bind an \
         ExternalSpeakerProjection from `sbv2.text_encoder.spk_emb_linear.\
         {{weight,bias}}` — the tensors are absent (or the converter \
         renamed them under a different path). Check that Blocker 1's \
         classify_tensor rename table still emits \
         `sbv2.text_encoder.spk_emb_linear.{{weight,bias}}` for upstream \
         `enc_p.encoder.spk_emb_linear.{{weight,bias}}`, and that the \
         checkpoint fixture at `{}` is a real Style-Bert-VITS2 v2 base \
         ckpt (fine-tunes with `emb_g` populated but no `spk_emb_linear` \
         would also fail this assertion — the two paths co-exist per the \
         `speaker` module doc).",
        main_path.display(),
    );

    // WP-01 (2026-08-09): `synthesize_with_intermediates` returns the
    // final PCM AND the per-stage tensor snapshots the Python dumper
    // records — driving the whole 11-tensor manifest diff off a SINGLE
    // forward pass (rather than one per tensor). See
    // `SbV2Intermediates::to_dumper_map` for the emit order and the
    // per-language BERT bucket skip convention.
    let (audio, intermediates) = model
        .synthesize_with_intermediates(&req)
        .unwrap_or_else(|e| panic!("SbV2Model::synthesize_with_intermediates: {e}"));

    // WP-06 (Wave 0 Task 6, 2026-08-11): every stage from here on is
    // collected into ONE `Vec<StageResult>` and asserted ONCE at the very
    // end of this test, instead of via N separate early-exit `assert!`s —
    // see `StageResult`'s doc for why. `results` starts with the ~10
    // intermediate-tensor rows; the waveform + mel_loss checks below
    // append their own rows to the SAME vec before the single final
    // assert further down.
    let mut results: Vec<StageResult> =
        diff_intermediates_against_manifest(&manifest, &intermediates, &dir);

    let (waveform_rel, waveform_shape) = find_tensor(&manifest, "waveform", &ctx);
    let waveform_path = dir.join(waveform_rel);
    require_fixture(&waveform_path, "tensors[name=waveform].path (Task 30 dump)");
    let reference = read_f32_bin(&waveform_path);
    let declared_len: u64 = waveform_shape.iter().product();
    assert_eq!(
        reference.len() as u64,
        declared_len,
        "{}: byte length ({} f32 elements) disagrees with the manifest's declared \
         `shape` {waveform_shape:?} — Task 34/30's dumper produced an inconsistent \
         fixture",
        waveform_path.display(),
        reference.len(),
    );

    // Length is a **tolerance-based** check, not `assert_eq!` (PR27-WAVEFORM-
    // TOLERANCE audit gap, 2026-08-08). SbV2SDP's terminal
    // `duration = ceil(exp(logw)).max(1) as i32` (duration.rs:1074) is a
    // discrete step: any residual ε near an integer boundary flips ±1 frame
    // → ±hop_length samples in the final waveform. For an 8-phoneme "テスト"
    // input at hop_length=512, a worst-case ±1-frame-per-phoneme flip
    // produces ±4096 samples out of ~27000 = ~15% length variance PURELY
    // from the discrete step, before any transcendental precision drift
    // enters the picture. Kokoro precedent: `PROSODY_F0_ATOL = 0.05` is an
    // architectural bound (F0_proj Conv1d 256→1 amplifies BiLstm1d
    // accumulator delta ~9×), NOT a CI-green loosening. The 10% length band
    // + tolerance-on-overlap contract below is the same posture applied to
    // the SBV2 waveform.
    //
    // See docs/adr/sbv2-libm-strategy.md §2.2 for the transcendental
    // amplification catalog (CEIL infinite, RQS sqrt unbounded, coupling
    // exp exponential, Box-Muller ~1 ULP per pair) and §4.4 for the
    // "MUST be tolerance-based" ruling this test implements.
    //
    // WP-06: this used to be a standalone `assert!` (early-exit on a
    // length-band miss, before max_abs_diff/RMS/mel_loss even ran). It is
    // now a `StageResult` like every other stage — `(len_ratio - 1.0).abs()
    // <= LENGTH_BAND_FRACTION` is the exact same boundary as the old
    // `(1.0-frac..=1.0+frac).contains(&len_ratio)` range check, just
    // expressed as a distance-from-1.0 so it fits the `max_abs_diff <=
    // atol_expected` shape every other row uses.
    const LENGTH_BAND_FRACTION: f32 = 0.10; // ±10% ≈ ±1 CEIL flip / phoneme worst-case
    let rust_len = audio.samples.len() as f32;
    let ref_len = reference.len() as f32;
    let len_ratio = rust_len / ref_len;
    let length_band_row = StageResult::numeric(
        "waveform_length_band",
        (len_ratio - 1.0).abs(),
        0.0, // no separate RMS concept for a length-ratio check
        LENGTH_BAND_FRACTION,
        "[N/A]",
    );
    length_band_row.emit();
    eprintln!(
        "[parity_sbv2_real] waveform_length_band detail: rust={} samples ref={} samples \
         (ratio {len_ratio:.4}). SbV2SDP's terminal `ceil()` accepts ±1 CEIL flip per \
         phoneme as an architectural bound (see docs/adr/sbv2-libm-strategy.md §2.2); a \
         miss this large usually means duration-computation drift, wrong `hop_length`, \
         or wrong `manifest.request` reproduction. If SBV2-BUG4 (text_encoder emits \
         hidden ~35× too large) is still un-landed, expect large-value durations to \
         make this fire.",
        audio.samples.len(),
        reference.len(),
    );
    results.push(length_band_row);

    // Signal comparison: max |Δ| on the overlapping prefix. Truncating to
    // `min(rust.len, ref.len)` is honest because up to `hop_length` trailing
    // samples in the longer buffer are ≤ 1 phoneme worth of silence padding
    // (the CEIL step producing 1 extra frame of a zero-input decoder
    // pushes on ~hop_length samples of near-silence), not audible signal
    // delta that a per-sample comparison would meaningfully measure.
    //
    // WP-06: max_abs_diff AND RMS-on-overlap (below) used to be two
    // separate early-exit `assert!`s against the same
    // `tolerance_for("waveform")` ceiling. They are now ONE
    // `StageResult::numeric_dual_gate` row — `passed` requires BOTH
    // `<= atol`, exactly like the two assertions it replaces.
    let overlap_len = audio.samples.len().min(reference.len());
    let rust_prefix = &audio.samples[..overlap_len];
    let ref_prefix = &reference[..overlap_len];
    let atol = tolerance_for("waveform");
    let diff = max_abs_diff(rust_prefix, ref_prefix);

    // RMS-on-overlap as a second signal-quality gate. Same tolerance
    // ceiling (Kokoro-precedent 6.84e-3 × 1.5 ≈ 0.01 default), but RMS
    // catches sustained low-level divergence that max_abs_diff can miss
    // (e.g. every sample off by 0.005 = max_abs_diff = 0.005 well under
    // atol=0.01, but RMS = 0.005 also well under — the two agree at low
    // divergence, and RMS catches high-fraction low-magnitude drift that
    // max_abs_diff would miss).
    let rms = rms_diff_over_prefix(rust_prefix, ref_prefix);
    let waveform_row =
        StageResult::numeric_dual_gate("waveform", diff, rms, atol, calibration_marker("waveform"));
    waveform_row.emit();
    if waveform_row.passed {
        eprintln!(
            "[parity_sbv2_real] waveform parity OK: rust={} samples ref={} samples \
             (ratio {:.4}, band ±{}%), overlap {} samples: max |Δ| = {diff:.3e}, \
             RMS |Δ| = {rms:.3e} <= atol {atol}",
            audio.samples.len(),
            reference.len(),
            len_ratio,
            LENGTH_BAND_FRACTION * 100.0,
            overlap_len,
        );
    }
    results.push(waveform_row);

    // WP-04: mel_loss aggregator — a tighter architectural bound than raw
    // waveform atol. `log(|X|^2 · mel_fb)` compresses the same
    // ~130M-transcendental cross-platform delta by two-three orders of
    // magnitude (ADR `docs/adr/sbv2-libm-strategy.md` §2.2), so a real
    // parity regression that stays under the (loose) waveform floor still
    // shows up here.
    //
    // WP-06: an error from `mel_loss_rms` itself (e.g. a degenerate
    // waveform too short for one STFT frame) is now a structural-failure
    // row rather than an immediate panic, so it does not hide whatever
    // the intermediate-tensor and waveform rows above already reported.
    let mel_loss_row = match mel_loss_rms(
        &audio.samples,
        &reference,
        SBV2_MEL_SR,
        SBV2_MEL_N_FFT,
        SBV2_MEL_HOP,
        SBV2_MEL_N_MELS,
    ) {
        Ok(mel_loss) => {
            let row = StageResult::numeric(
                "mel_loss",
                mel_loss as f32,
                0.0, // mel_loss is itself an RMS aggregate; no secondary metric
                MEL_LOSS_ATOL,
                calibration_marker("mel_loss"),
            );
            row.emit();
            if row.passed {
                eprintln!(
                    "[parity_sbv2_real] mel_loss parity OK: rms = {mel_loss:.3e} <= \
                     atol {MEL_LOSS_ATOL} (n_fft={SBV2_MEL_N_FFT}, hop={SBV2_MEL_HOP}, \
                     n_mels={SBV2_MEL_N_MELS}, sr={SBV2_MEL_SR})"
                );
            }
            row
        }
        Err(e) => {
            let row = StageResult::structural_failure(
                "mel_loss",
                MEL_LOSS_ATOL,
                calibration_marker("mel_loss"),
                format!("mel_loss_rms(rust, reference): {e}"),
            );
            row.emit();
            row
        }
    };
    results.push(mel_loss_row);

    // WP-06: the ONE end-of-run assertion every stage above feeds. Lists
    // every failing stage at once (numeric misses AND structural
    // failures), mirroring `diff_intermediates_against_manifest`'s
    // pre-WP-06 per-function panic message but now spanning the WHOLE
    // pipeline (intermediates + waveform + mel_loss) instead of just the
    // ~10 intermediate tensors.
    let failed: Vec<&StageResult> = results.iter().filter(|r| !r.passed).collect();
    assert!(
        failed.is_empty(),
        "{} of {} stages exceeded per-tensor atol (or failed structurally) — see the \
         per-stage [parity_sbv2_real] / [PASS/FAIL] lines above for the full report. \
         Failing stages: {}. See `docs/adr/sbv2-parity-atol.md` §5 for the \
         empirical-measurement cycle and `tools/parity/sbv2_atol_updater.py` (WP-02, \
         2026-08-09) for the atol proposal from the workflow's atol-summary artifact.",
        failed.len(),
        results.len(),
        failed
            .iter()
            .map(|r| {
                let extra = r
                    .note
                    .as_deref()
                    .map(|n| format!(" [{n}]"))
                    .unwrap_or_default();
                format!(
                    "{} (max |Δ| = {:.3e} > atol {}{extra})",
                    r.stage, r.max_abs_diff, r.atol_expected
                )
            })
            .collect::<Vec<_>>()
            .join(", "),
    );

    // WP-24: tail-position UTMOS quality gate. Fail-closed opt-in — see
    // the module doc's "WP-24: UTMOS quality gate" section for the
    // fixture recipe and CI wiring, and `utmos_gate` for the panic /
    // skip semantics. Runs AFTER every aggregated stage above has
    // passed (the final `assert!` above returns normally only when
    // `failed` is empty) so a failing UTMOS score always points at a
    // real perceptual regression rather than an incidental sample-level
    // noise floor drift.
    utmos_gate(&audio.samples, &reference, audio.sample_rate);
}

/// Aggregator mel-loss (WP-04, ADR `docs/adr/sbv2-libm-strategy.md` §2.2):
/// computes mel-spectrograms of `a` and `b` and returns the RMS of the
/// log-mel-magnitude difference over their shared frame overlap.
///
/// Pipeline (matches every `vokra_ops::mel_filterbank` consumer in this
/// workspace — see `crates/vokra-models/src/whisper/mel.rs` for the
/// canonical STFT → |X|² → mel-filterbank → log chain):
///   1. `vokra_ops::stft(_, StftAttrs::new(n_fft, hop))` on each input;
///   2. `Spectrogram::power()` → `[frames, n_freqs]` (element-wise
///      `re² + im²`);
///   3. `MelFilterbank::apply(power, frames)` → `[frames, n_mels]`;
///   4. clamp-with-epsilon-then-ln, per-bin subtract, RMS over the shared
///      `min(frames_a, frames_b)` frames × `n_mels` bins.
///
/// The `MelAttrs` this builds mirror librosa/torchaudio defaults for the
/// SBV2 v2 base: `Slaney` scale, `Slaney` norm, `Hz`-linear ramp, `fmin=0`,
/// `fmax=sr/2` (the frontend spec the reference dumper uses; a per-model
/// override would flow through the manifest's `phonemize_fixture` sibling
/// once a follow-up widens the API — WP-04 keeps the front-end constant so
/// this parity's mel bound matches what the CI dumper computes).
///
/// # Length handling
///
/// Mirrors `tolerance_for("waveform")`'s implicit contract at the
/// waveform level (`max_abs_diff` currently asserts equal lengths, so a
/// length mismatch surfaces at that check first): if the two waveforms
/// produce a differing number of STFT frames, only the leading
/// `min(frames_a, frames_b)` frames are compared, matching the sibling
/// `vokra_eval::MelLoss::loss` shared-overlap convention.
///
/// # Errors
///
/// Returns [`VokraError::InvalidArgument`] if either input is too short to
/// produce a single STFT frame (i.e. `signal.len() + n_fft < n_fft` under
/// centered padding), or if `vokra_ops::stft`'s own attrs validation
/// rejects the constructed [`StftAttrs`]. Never silently returns `0.0` on
/// a degenerate input — FR-EX-08.
fn mel_loss_rms(
    a: &[f32],
    b: &[f32],
    sample_rate: u32,
    n_fft: usize,
    hop: usize,
    n_mels: usize,
) -> vokra_core::Result<f64> {
    if n_mels == 0 {
        return Err(VokraError::InvalidArgument(
            "mel_loss_rms: n_mels must be non-zero".to_owned(),
        ));
    }
    let stft_attrs = StftAttrs::new(n_fft, hop);
    let mel_attrs = MelAttrs {
        norm: MelNorm::Slaney,
        scale: MelScale::Slaney,
        interp: MelInterp::Hz,
        fmin: 0.0,
        fmax: Some(sample_rate as f32 / 2.0),
        ..MelAttrs::new(sample_rate, n_fft, n_mels)
    };
    let fb = mel_filterbank(&mel_attrs);

    let spec_a = stft(a, &stft_attrs)?;
    let spec_b = stft(b, &stft_attrs)?;

    let common = spec_a.frames.min(spec_b.frames);
    if common == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "mel_loss_rms: no shared STFT frames (spec_a.frames={}, spec_b.frames={}, \
             a.len()={}, b.len()={}, n_fft={n_fft}, hop={hop}); inputs are too short",
            spec_a.frames,
            spec_b.frames,
            a.len(),
            b.len()
        )));
    }

    // Compute power then project through mel filterbank. `MelFilterbank::apply`
    // expects `[frames, n_freqs]`; `Spectrogram::power()` returns exactly that
    // shape. Only the leading `common` frames matter — save the trailing work.
    let n_freqs = spec_a.bins; // == n_fft/2 + 1 (real_input default)
    let power_a: Vec<f32> = spec_a.re[..common * n_freqs]
        .iter()
        .zip(&spec_a.im[..common * n_freqs])
        .map(|(r, i)| r * r + i * i)
        .collect();
    let power_b: Vec<f32> = spec_b.re[..common * n_freqs]
        .iter()
        .zip(&spec_b.im[..common * n_freqs])
        .map(|(r, i)| r * r + i * i)
        .collect();
    let mel_a = fb.apply(&power_a, common);
    let mel_b = fb.apply(&power_b, common);

    // Log-mel + RMS. `1e-10` matches the sibling `vokra_eval::MelLoss` epsilon
    // (log-domain floor keeps `.ln()` finite on true-zero bands from the
    // filterbank cutoff).
    let eps: f32 = 1e-10;
    let mut sum_sq: f64 = 0.0;
    let mut count: usize = 0;
    for t in 0..common {
        for m in 0..n_mels {
            let la = mel_a[t * n_mels + m].max(eps).ln();
            let lb = mel_b[t * n_mels + m].max(eps).ln();
            let d = (la - lb) as f64;
            sum_sq += d * d;
            count += 1;
        }
    }
    Ok((sum_sq / count as f64).sqrt())
}

// ---------------------------------------------------------------------------
// WP-04 TDD RED-turned-GREEN: the two unit tests below assert
// `mel_loss_rms`'s core contract (identical inputs → 0.0, differing inputs →
// > 0.0). Written BEFORE the implementation above; without `mel_loss_rms`
// this test binary refuses to compile — the concrete RED signal — so
// landing them together with the implementation is the honest GREEN.
// ---------------------------------------------------------------------------

/// Synthetic mono sinusoid at `freq_hz` for `duration_s` seconds sampled at
/// `sr` Hz — the fixture the two unit tests below build inputs from.
fn sinusoid(freq_hz: f32, duration_s: f32, sr: u32) -> Vec<f32> {
    let n = (duration_s * sr as f32) as usize;
    let two_pi = 2.0 * std::f32::consts::PI;
    (0..n)
        .map(|t| (two_pi * freq_hz * t as f32 / sr as f32).sin())
        .collect()
}

#[test]
fn mel_loss_rms_is_zero_on_identical_inputs() {
    // Log-mel RMS of a waveform against itself is 0 by construction (every
    // per-bin delta is exactly 0). Uses an f64 epsilon rather than a strict
    // `== 0.0` — the RMS accumulator is f64, so intermediate `sum_sq / count`
    // divisions can produce a bit-flipped 0 depending on ordering.
    let a = sinusoid(440.0, 0.25, SBV2_MEL_SR);
    let loss = mel_loss_rms(
        &a,
        &a,
        SBV2_MEL_SR,
        SBV2_MEL_N_FFT,
        SBV2_MEL_HOP,
        SBV2_MEL_N_MELS,
    )
    .expect("mel_loss_rms on non-degenerate inputs");
    assert!(
        loss < 1e-6,
        "mel_loss_rms(a, a) = {loss}, expected ~0 (log-mel of a waveform against \
         itself is 0 by construction)"
    );
}

#[test]
fn mel_loss_rms_is_positive_on_differing_inputs() {
    // Two sinusoids at different frequencies produce mel-spectrograms with
    // different band-energy distributions; the log-mel RMS therefore must
    // be strictly positive. This is the "detect a real difference" half of
    // WP-04's mel-loss guard — if this ever returns 0, the aggregator has
    // silently collapsed (e.g. an accidental `mel_b = mel_a`).
    let a = sinusoid(440.0, 0.25, SBV2_MEL_SR);
    let b = sinusoid(880.0, 0.25, SBV2_MEL_SR);
    let loss = mel_loss_rms(
        &a,
        &b,
        SBV2_MEL_SR,
        SBV2_MEL_N_FFT,
        SBV2_MEL_HOP,
        SBV2_MEL_N_MELS,
    )
    .expect("mel_loss_rms on non-degenerate inputs");
    assert!(
        loss > 1e-3,
        "mel_loss_rms(a, b) = {loss} but the inputs are different frequencies \
         (440 Hz vs 880 Hz); a near-zero result means the aggregator has \
         silently collapsed"
    );
}

#[test]
fn mel_loss_rms_rejects_degenerate_inputs_loudly() {
    // FR-EX-08: n_mels==0 is a documented error path — never a silent 0.0.
    // (An empty / too-short signal is NOT a degenerate input under centered
    // STFT: `pad_for_analysis` reflects `n_fft/2` samples on each end, so
    // even `signal.len()==0` produces exactly one frame of pure padding
    // and mel_loss_rms of an empty input against itself is a legitimate
    // 0.0 — that mirrors `vokra_ops::stft`'s own semantics and does NOT
    // violate FR-EX-08 because the returned number is arithmetically
    // correct, not silently wrong.)
    let a = sinusoid(440.0, 0.05, SBV2_MEL_SR);
    let err_zero_mels =
        mel_loss_rms(&a, &a, SBV2_MEL_SR, SBV2_MEL_N_FFT, SBV2_MEL_HOP, 0).expect_err("n_mels=0");
    assert!(
        matches!(err_zero_mels, VokraError::InvalidArgument(_)),
        "expected InvalidArgument on n_mels=0, got {err_zero_mels:?}"
    );
    // A downstream `stft` error (from a n_fft=0 attrs, which our aggregator
    // does NOT construct itself but a caller could conceivably plumb) also
    // propagates via `?` — proven by asking for n_fft=0 directly. This is
    // the FR-EX-08 loud-propagation half.
    let err_zero_nfft =
        mel_loss_rms(&a, &a, SBV2_MEL_SR, 0, SBV2_MEL_HOP, SBV2_MEL_N_MELS).expect_err("n_fft=0");
    assert!(
        matches!(err_zero_nfft, VokraError::InvalidArgument(_)),
        "expected InvalidArgument from stft on n_fft=0, got {err_zero_nfft:?}"
    );
}

#[test]
fn mel_loss_rms_handles_length_mismatch_via_shared_overlap() {
    // Mirrors `tolerance_for("waveform")`'s spirit at the frame level:
    // rather than error on a length mismatch, mel_loss_rms compares the
    // leading `min(frames_a, frames_b)` frames.
    //
    // The tail frames of the shorter input use `reflect` padding for
    // samples past its own end, while the longer input uses its real
    // samples in the same positions — so the two spectrograms will differ
    // on the last frame or two, and this test does NOT expect a strict 0.
    // What it DOES prove is (a) the call succeeds without an error /
    // panic and (b) the return value is a finite non-negative real (the
    // shared-overlap machinery ran instead of the shape-mismatch error a
    // strict-equal-length aggregator would raise).
    let a = sinusoid(440.0, 0.5, SBV2_MEL_SR);
    let b = &a[..a.len() / 2];
    let loss = mel_loss_rms(
        &a,
        b,
        SBV2_MEL_SR,
        SBV2_MEL_N_FFT,
        SBV2_MEL_HOP,
        SBV2_MEL_N_MELS,
    )
    .expect("mel_loss_rms on length-mismatched inputs must not error");
    assert!(
        loss.is_finite() && loss >= 0.0,
        "shared-overlap mel_loss must be a finite non-negative real, got {loss}"
    );
}

// --- unit tests for the tolerance-based helpers (PR27-WAVEFORM-TOLERANCE) ---
//
// These are runnable via plain `cargo test -p vokra-models --test
// parity_sbv2_real` (no `--ignored` flag), so they always fire in CI
// even when the real Task 30 fixture is absent. They validate the
// helpers the ignored real-parity harness above depends on — a
// regression in `max_abs_diff` or `rms_diff_over_prefix` semantics
// would surface here loudly instead of silently corrupting the
// real-parity waveform-tolerance calc when a fixture finally lands.
#[cfg(test)]
mod tolerance_helpers {
    use super::{max_abs_diff, rms_diff_over_prefix};

    #[test]
    fn max_abs_diff_zero_when_equal() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0, 3.0];
        assert_eq!(max_abs_diff(&a, &b), 0.0);
    }

    #[test]
    fn max_abs_diff_reports_worst_frame() {
        // Delta pattern: 0.1, 0.0, 5.0, 0.1 → max = 5.0.
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.1, 2.0, 8.0, 4.1];
        let diff = max_abs_diff(&a, &b);
        assert!((diff - 5.0).abs() < 1e-6, "max = {diff}");
    }

    #[test]
    fn rms_diff_over_prefix_zero_when_equal() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(rms_diff_over_prefix(&a, &b), 0.0);
    }

    #[test]
    fn rms_diff_over_prefix_matches_hand_calc() {
        // Delta: 0.1, 0.2, 0.3 → sum_sq = 0.01 + 0.04 + 0.09 = 0.14
        // → mean_sq = 0.14 / 3 ≈ 0.04667 → sqrt ≈ 0.21602
        let a = [1.0, 2.0, 3.0];
        let b = [1.1, 2.2, 3.3];
        let rms = rms_diff_over_prefix(&a, &b);
        let expected = (0.14_f64 / 3.0).sqrt() as f32;
        assert!(
            (rms - expected).abs() < 1e-6,
            "rms = {rms}, expected {expected}"
        );
    }

    #[test]
    fn rms_diff_over_prefix_empty_slice_returns_zero() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        assert_eq!(rms_diff_over_prefix(&a, &b), 0.0);
    }

    #[test]
    fn rms_diff_over_prefix_uses_shorter_length() {
        // Longer slice's trailing samples are IGNORED — this is the
        // "truncate to overlap" semantic the waveform tolerance-check
        // relies on to accept ±1-CEIL-flip-per-phoneme length drift
        // without penalising the audible-signal-comparison prefix.
        let a = [1.0, 2.0, 3.0]; // shorter
        let b = [1.0, 2.0, 3.0, 999.0, 999.0]; // longer, trailing garbage
        assert_eq!(
            rms_diff_over_prefix(&a, &b),
            0.0,
            "trailing samples in the longer slice must NOT contribute to RMS"
        );
        assert_eq!(
            rms_diff_over_prefix(&b, &a),
            0.0,
            "min-length semantic must be symmetric — swapping argument order \
             cannot change the truncation prefix"
        );
    }
}

// --- unit tests for UTMOS opt-in settings (WP-24) ---
//
// T19: pins the UtmosGateSettings::resolve() env-var resolution mechanism.
// Without VOKRA_SBV2_UTMOS_ENABLE=1, the gate MUST skip; with it, the
// gate MUST require VOKRA_SBV2_UTMOS_GGUF to point at a real (non-
// synthesized) UTMOS22-strong GGUF. Neither path should silently pass.
//
// NOTE: These tests modify global env vars and can interfere with each
// other if run in parallel. Run with `--test-threads=1` to ensure
// isolation:
//   cargo test -p vokra-models --test parity_sbv2_real utmos_gate_settings -- --test-threads=1
#[cfg(test)]
#[allow(unsafe_code)]
mod utmos_gate_settings {
    use std::panic;
    use std::path::PathBuf;

    /// T19 snapshot: pins that when VOKRA_SBV2_UTMOS_ENABLE is unset,
    /// resolve() returns Disabled (no panic, silent skip).
    #[test]
    fn utmos_gate_settings_disabled_when_enable_unset() {
        // Save-and-restore so this test doesn't clobber a run's actual settings
        let saved_enable = std::env::var(crate::ENV_UTMOS_ENABLE).ok();
        let saved_gguf = std::env::var_os(crate::ENV_UTMOS_GGUF);

        // SAFETY: This test runs in isolation; no concurrent env access.
        unsafe {
            // Clear both
            std::env::remove_var(crate::ENV_UTMOS_ENABLE);
            std::env::remove_var(crate::ENV_UTMOS_GGUF);

            // Expected: UtmosGateSettings::resolve() returns Disabled variant
            let settings = crate::UtmosGateSettings::resolve();
            assert!(
                matches!(settings, crate::UtmosGateSettings::Disabled),
                "utmos_gate must skip (Disabled) when VOKRA_SBV2_UTMOS_ENABLE is unset"
            );

            // Restore
            if let Some(v) = saved_enable {
                std::env::set_var(crate::ENV_UTMOS_ENABLE, v);
            }
            if let Some(v) = saved_gguf {
                std::env::set_var(crate::ENV_UTMOS_GGUF, v);
            } else {
                std::env::remove_var(crate::ENV_UTMOS_GGUF);
            }
        }
    }

    /// T19 snapshot: pins that when VOKRA_SBV2_UTMOS_ENABLE=1 but
    /// VOKRA_SBV2_UTMOS_GGUF is unset, resolve() panics loudly with an
    /// FR-EX-08 explicit error (never a silent skip).
    #[test]
    fn utmos_gate_settings_panics_when_enable_set_but_gguf_unset() {
        // Save-and-restore
        let saved_enable = std::env::var(crate::ENV_UTMOS_ENABLE).ok();
        let saved_gguf = std::env::var_os(crate::ENV_UTMOS_GGUF);

        // SAFETY: This test runs in isolation; no concurrent env access.
        unsafe {
            // Set enable to "1", clear gguf
            std::env::set_var(crate::ENV_UTMOS_ENABLE, "1");
            std::env::remove_var(crate::ENV_UTMOS_GGUF);

            // Expected: resolve() panics with a clear message containing
            // both env var names and "FR-EX-08" guidance.
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                crate::UtmosGateSettings::resolve()
            }));

            assert!(
                result.is_err(),
                "utmos_gate must panic when ENABLE=1 but GGUF is unset \
                 (broken opt-in is never a skip, FR-EX-08)"
            );

            // Restore
            if let Some(v) = saved_enable {
                std::env::set_var(crate::ENV_UTMOS_ENABLE, v);
            } else {
                std::env::remove_var(crate::ENV_UTMOS_ENABLE);
            }
            if let Some(v) = saved_gguf {
                std::env::set_var(crate::ENV_UTMOS_GGUF, v);
            } else {
                std::env::remove_var(crate::ENV_UTMOS_GGUF);
            }
        }
    }

    /// T19 snapshot: pins that when both VOKRA_SBV2_UTMOS_ENABLE=1 and
    /// VOKRA_SBV2_UTMOS_GGUF are set to valid values, resolve() returns
    /// Enabled variant with the correct PathBuf.
    #[test]
    fn utmos_gate_settings_enabled_when_both_set() {
        // Save-and-restore
        let saved_enable = std::env::var(crate::ENV_UTMOS_ENABLE).ok();
        let saved_gguf = std::env::var_os(crate::ENV_UTMOS_GGUF);

        // SAFETY: This test runs in isolation; no concurrent env access.
        unsafe {
            // Set both to valid values (no need for file to exist in this unit test)
            let test_path = "/tmp/test_utmos.gguf";
            std::env::set_var(crate::ENV_UTMOS_ENABLE, "1");
            std::env::set_var(crate::ENV_UTMOS_GGUF, test_path);

            // Expected: resolve() returns Enabled with the correct path
            let settings = crate::UtmosGateSettings::resolve();
            match settings {
                crate::UtmosGateSettings::Enabled { gguf_path } => {
                    assert_eq!(
                        gguf_path,
                        PathBuf::from(test_path),
                        "Enabled variant must contain the exact path from VOKRA_SBV2_UTMOS_GGUF"
                    );
                }
                crate::UtmosGateSettings::Disabled => {
                    panic!("expected Enabled variant, got Disabled");
                }
            }

            // Restore
            if let Some(v) = saved_enable {
                std::env::set_var(crate::ENV_UTMOS_ENABLE, v);
            } else {
                std::env::remove_var(crate::ENV_UTMOS_ENABLE);
            }
            if let Some(v) = saved_gguf {
                std::env::set_var(crate::ENV_UTMOS_GGUF, v);
            } else {
                std::env::remove_var(crate::ENV_UTMOS_GGUF);
            }
        }
    }
}
