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

use std::path::{Path, PathBuf};

use vokra_core::VokraError;
use vokra_core::gguf::GgufFile;
use vokra_core::ir::graph::{MelAttrs, MelInterp, MelNorm, MelScale, StftAttrs};
use vokra_core::json::{self, JsonValue};
use vokra_models::sbv2::{
    AtolCalibration, Language, MEL_LOSS_ATOL, PhonemizeFixture, PhonemizeResult, RngMode,
    SbV2Intermediates, SbV2Model, SbV2Phonemizer, SbV2SynthRequest, atol_calibration_for,
    tolerance_for,
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
fn request_from_manifest(manifest: &JsonValue, ctx: &str) -> SbV2SynthRequest {
    let request = json_get(manifest, "request", ctx);
    let language = match json_str(request, "language", ctx) {
        "JA" => Language::JA,
        "EN" => Language::EN,
        "ZH" => Language::ZH,
        other => panic!("{ctx}: request.language must be \"JA\", \"EN\", or \"ZH\", got {other:?}"),
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

/// WP-01 (2026-08-09): iterates every intermediate tensor from
/// [`SbV2Intermediates::to_dumper_map`] and diffs it against the
/// corresponding `reference_dump/<name>.bin` fixture named in the
/// manifest.
///
/// Contract:
/// - Every dumper-map entry MUST have a matching manifest `tensors[]`
///   entry (finding it goes through [`find_tensor`], which panics
///   loudly on a miss — the harness expects the two to stay in sync,
///   drift is a real fixture bug, not a soft-skip).
/// - Every fixture's `.bin` byte length MUST match the manifest's
///   declared shape product AND the intermediate's own element count.
///   Both mismatches panic loudly.
/// - The per-element `max_abs_diff` MUST be `<= tolerance_for(name)`.
///   Emits `[parity_sbv2_real] <name>: max |Δ| = ... <= atol ... <status>`
///   to stderr for each tensor, so a CI viewer sees the full 11-row
///   report even when every row passes (mirrors the Kokoro parity CI's
///   per-tensor summary format).
///
/// The `waveform` diff stays outside this loop — its length is
/// discretely dependent on SDP durations and needs the tolerance-based
/// length-band + max-diff + RMS-on-overlap gate (see the caller).
/// Every OTHER manifest tensor's shape is pinned exactly by the
/// manifest schema.
fn diff_intermediates_against_manifest(
    manifest: &JsonValue,
    intermediates: &SbV2Intermediates,
    dir: &Path,
    ctx: &str,
) {
    // WP-02 (2026-08-09): collect every tensor's max|Δ| BEFORE asserting.
    // The pre-WP-02 shape (assert! in-loop, eprintln! post-assert) meant a
    // failing tensor never emitted its diagnostic line, so
    // `.github/workflows/parity-sbv2-real.yml`'s downstream atol-summary
    // step could only see PASSING tensor rows — an owner reviewing the
    // artifact would have to guess whether "no line for bert_hidden_ja"
    // meant "not run" or "failed silently". The two-phase shape here
    // emits ALL 10 rows first (mirroring the Kokoro parity CI's per-tensor
    // summary), then asserts once at the end so a failing tensor is loud
    // AND still leaves the workflow parser a full row set to feed into
    // `tools/parity/sbv2_atol_updater.py`.
    struct Row {
        name: &'static str,
        max_diff: f32,
        atol: f32,
        marker: &'static str,
        verdict: &'static str,
    }
    let mut rows: Vec<Row> = Vec::new();
    for (name, rust_bytes) in intermediates.to_dumper_map() {
        let (rel, shape) = find_tensor(manifest, name, ctx);
        let ref_path = dir.join(rel);
        require_fixture(
            &ref_path,
            &format!("tensors[name={name}].path (Task 30 dump)"),
        );
        let reference = read_f32_bin(&ref_path);
        let declared_len: u64 = shape.iter().product();
        assert_eq!(
            reference.len() as u64,
            declared_len,
            "{}: byte length ({} f32 elements) disagrees with the manifest's declared \
             `shape` {shape:?} — Task 34/30's dumper produced an inconsistent fixture",
            ref_path.display(),
            reference.len(),
        );

        // `to_dumper_map`'s `Vec<u8>` payload is little-endian f32 (its
        // `f32_bytes` helper). Rebuild the f32 slice here to diff against
        // the reference. A length mismatch is a real bug — the Rust
        // pipeline emitted a different tensor shape than the Python
        // dumper — so panic loudly, not soft-skip (FR-EX-08). This kind
        // of failure is a fixture / pipeline inconsistency that cannot
        // be papered over by a bigger atol, so it stays in-loop as a
        // hard early exit (unlike the numeric max|Δ| gate below, which
        // the two-phase pattern defers to the end).
        assert_eq!(
            rust_bytes.len() % 4,
            0,
            "SbV2Intermediates::to_dumper_map(`{name}`) payload {} bytes is not \
             f32-aligned — Wave-4 INTERMEDIATE-ACCESSORS contract violation",
            rust_bytes.len(),
        );
        let rust: Vec<f32> = rust_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(
            rust.len(),
            reference.len(),
            "{}: Rust `{name}` len {} != reference len {} — shape mismatch is a real \
             pipeline bug, not a numeric drift (the manifest fixes each intermediate's \
             shape exactly; unlike waveform, no discrete-step ±1 flip applies)",
            ref_path.display(),
            rust.len(),
            reference.len(),
        );

        let atol = tolerance_for(name);
        let diff = max_abs_diff(&rust, &reference);
        let marker = calibration_marker(name);
        let verdict = if diff <= atol { "PASS" } else { "FAIL" };
        // Emit the diagnostic line BEFORE deferring the assertion so the
        // workflow's downstream atol-summary parser (regex against
        // `[parity_sbv2_real] ...`) sees a row for every tensor even
        // when one fails. Format is stable: workflow parser depends on
        // this exact prefix.
        eprintln!(
            "[parity_sbv2_real] {name}: max |Δ| = {diff:.6e} atol {atol} verdict {verdict} \
             {marker}"
        );
        rows.push(Row {
            name,
            max_diff: diff,
            atol,
            marker,
            verdict,
        });
    }

    // Two-phase gate: fail LOUDLY once at the end, naming EVERY failing
    // tensor at once. Under the pre-WP-02 in-loop assert, an owner
    // reviewing a red CI run saw one failure and had to iterate. Now the
    // first CI cycle produces the complete "here are all 3 tensors that
    // exceeded their atol" list, so the empirical-measurement cycle can
    // batch-propose in one pass.
    let failures: Vec<&Row> = rows.iter().filter(|r| r.verdict == "FAIL").collect();
    if !failures.is_empty() {
        let names: Vec<String> = failures
            .iter()
            .map(|r| {
                format!(
                    "{} (max |Δ| = {:.3e} > atol {} {})",
                    r.name, r.max_diff, r.atol, r.marker
                )
            })
            .collect();
        panic!(
            "[parity_sbv2_real] {} of {} intermediate tensors exceeded their atol: {}. \
             See `docs/adr/sbv2-parity-atol.md` §5 for the empirical-measurement cycle \
             and `tools/parity/sbv2_atol_updater.py` (WP-02, 2026-08-09) for the atol \
             proposal from the workflow's atol-summary artifact.",
            failures.len(),
            rows.len(),
            names.join(", "),
        );
    }
}

/// Real-checkpoint SBV2 full-manifest parity, gated on the Task 34 + Task 30
/// fixture set (`tests/fixtures/sbv2/`). See the module doc for: the
/// manifest schema this reads (including the Task 7 `phonemize_fixture`
/// block that bypasses the missing in-workspace 8-language G2P), the
/// WP-01 (2026-08-09) full-manifest iteration built on
/// `SbV2Model::synthesize_with_intermediates` +
/// `SbV2Intermediates::to_dumper_map`, and how the G2P bypass wires
/// `SbV2Model::from_gguf_with_phonemizer` +
/// `SbV2Phonemizer::from_fixture` together.
#[test]
#[ignore = "Task 34 real fixture: tests/fixtures/sbv2/{reference_dump.manifest.json,*.gguf,reference_dump/*.bin}"]
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

    // WP-01 (2026-08-09): `synthesize_with_intermediates` returns the
    // final PCM AND the per-stage tensor snapshots the Python dumper
    // records — driving the whole 11-tensor manifest diff off a SINGLE
    // forward pass (rather than one per tensor). See
    // `SbV2Intermediates::to_dumper_map` for the emit order and the
    // per-language BERT bucket skip convention.
    let (audio, intermediates) = model
        .synthesize_with_intermediates(&req)
        .unwrap_or_else(|e| panic!("SbV2Model::synthesize_with_intermediates: {e}"));

    // Diff every intermediate tensor against its manifest fixture.
    // The waveform assertion below stays separate — its length is
    // discretely dependent on SDP durations and needs the
    // tolerance-based length-band + RMS-on-overlap gate (see the
    // PR27-WAVEFORM-TOLERANCE block).
    diff_intermediates_against_manifest(&manifest, &intermediates, &dir, &ctx);

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
    const LENGTH_BAND_FRACTION: f32 = 0.10; // ±10% ≈ ±1 CEIL flip / phoneme worst-case
    let rust_len = audio.samples.len() as f32;
    let ref_len = reference.len() as f32;
    let len_ratio = rust_len / ref_len;
    assert!(
        (1.0 - LENGTH_BAND_FRACTION..=1.0 + LENGTH_BAND_FRACTION).contains(&len_ratio),
        "waveform length outside the ±{}% band: Rust `synthesize` produced {} \
         samples, the reference dump has {} (ratio = {:.4}). SbV2SDP's terminal \
         `ceil()` accepts ±1 CEIL flip per phoneme as an architectural bound \
         (see docs/adr/sbv2-libm-strategy.md §2.2), but a delta of this size \
         indicates either duration-computation drift, wrong `hop_length`, or \
         wrong `manifest.request` reproduction. If SBV2-BUG4 (text_encoder \
         emits hidden ~35× too large) is still un-landed, expect large-value \
         durations to make this fire — that is the intended loud-failure.",
        LENGTH_BAND_FRACTION * 100.0,
        audio.samples.len(),
        reference.len(),
        len_ratio,
    );

    // Signal comparison: max |Δ| on the overlapping prefix. Truncating to
    // `min(rust.len, ref.len)` is honest because up to `hop_length` trailing
    // samples in the longer buffer are ≤ 1 phoneme worth of silence padding
    // (the CEIL step producing 1 extra frame of a zero-input decoder
    // pushes on ~hop_length samples of near-silence), not audible signal
    // delta that a per-sample comparison would meaningfully measure.
    let overlap_len = audio.samples.len().min(reference.len());
    let rust_prefix = &audio.samples[..overlap_len];
    let ref_prefix = &reference[..overlap_len];
    let atol = tolerance_for("waveform");
    let diff = max_abs_diff(rust_prefix, ref_prefix);
    assert!(
        diff <= atol,
        "waveform max |Δ| = {diff} exceeds atol {atol} (sbv2::parity::tolerance_for(\"waveform\")) \
         over the overlapping prefix [0..{}] samples (rust={} ref={})",
        overlap_len,
        audio.samples.len(),
        reference.len(),
    );

    // RMS-on-overlap as a second signal-quality gate. Same tolerance
    // ceiling (Kokoro-precedent 6.84e-3 × 1.5 ≈ 0.01 default), but RMS
    // catches sustained low-level divergence that max_abs_diff can miss
    // (e.g. every sample off by 0.005 = max_abs_diff = 0.005 well under
    // atol=0.01, but RMS = 0.005 also well under — the two agree at low
    // divergence, and RMS catches high-fraction low-magnitude drift that
    // max_abs_diff would miss).
    let rms = rms_diff_over_prefix(rust_prefix, ref_prefix);
    assert!(
        rms <= atol,
        "waveform RMS |Δ| = {rms} exceeds atol {atol} on overlapping prefix \
         [0..{}] samples — max_abs_diff was {:.3e} which passed, but RMS \
         indicates sustained low-level divergence across the whole prefix. \
         See docs/adr/sbv2-libm-strategy.md §2 for the residual model.",
        overlap_len,
        diff,
    );

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

    // WP-04: mel_loss aggregator — a tighter architectural bound than raw
    // waveform atol. `log(|X|^2 · mel_fb)` compresses the same
    // ~130M-transcendental cross-platform delta by two-three orders of
    // magnitude (ADR `docs/adr/sbv2-libm-strategy.md` §2.2), so a real
    // parity regression that stays under the (loose) waveform floor still
    // shows up here. Run AFTER the waveform check so both sides get to
    // emit their per-metric diagnostic on a partial pass.
    let mel_loss = mel_loss_rms(
        &audio.samples,
        &reference,
        SBV2_MEL_SR,
        SBV2_MEL_N_FFT,
        SBV2_MEL_HOP,
        SBV2_MEL_N_MELS,
    )
    .unwrap_or_else(|e| panic!("mel_loss_rms(rust, reference): {e}"));
    assert!(
        mel_loss <= MEL_LOSS_ATOL as f64,
        "mel_loss RMS = {mel_loss} exceeds MEL_LOSS_ATOL {} \
         (sbv2::parity::MEL_LOSS_ATOL, WP-04 EstimatedPreFixture — see docstring \
         for derivation; a genuine regression means widening this needs an ADR + \
         updated derivation, not silent loosening)",
        MEL_LOSS_ATOL
    );
    eprintln!(
        "[parity_sbv2_real] mel_loss parity OK: rms = {mel_loss:.3e} <= \
         atol {} (n_fft={SBV2_MEL_N_FFT}, hop={SBV2_MEL_HOP}, n_mels={SBV2_MEL_N_MELS}, \
         sr={SBV2_MEL_SR})",
        MEL_LOSS_ATOL,
    );
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
