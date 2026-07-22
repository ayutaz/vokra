//! UTMOS stage-by-stage parity vs the **upstream** reference (M5-15 T19).
//!
//! # What makes this an honest reference
//!
//! The fixtures this test consumes are produced by
//! `tools/parity/utmos_dump_reference.py`, which **imports the real upstream
//! implementation** — the `sarulab-speech/UTMOS-demo` sources at a pinned,
//! sha256-verified revision, running on the real `fairseq` `Wav2Vec2Model` at
//! commit `d03f4e77`. Nothing in the reference path is re-implemented here.
//!
//! That discipline is the direct lesson of Kokoro (`92dbc92`): a reference
//! produced by a *mirror* of the model under test agrees with it by
//! construction, so parity goes green while the audio is wrong. A dumper that
//! cannot import upstream aborts loudly rather than substituting a mirror.
//!
//! # Why stage-by-stage and not just the score
//!
//! A single scalar cannot localize a fault: a transposed `ln1`/`ln2` mapping,
//! a mis-folded weight-norm, or a backwards LSTM direction all show up as
//! "the number is wrong". Comparing at each upstream hook point turns a
//! failure into a named stage.
//!
//! # Honest tolerance (M5-15 T39, NFR-QL-02 / `feedback-honest-parity-atol`)
//!
//! The pinned reference environment is torch 1.11.0, which has **no
//! macOS-arm64 wheel**, so the reference on this machine necessarily runs on
//! a different torch. The band therefore includes an env delta and is *not* a
//! pure float-association band. The dumper records the reference's own re-run
//! spread in `manifest.json` (`rerun_band`); the per-stage tolerances below
//! are derived from measurement and recorded with their rationale — never
//! widened to chase a green.
//!
//! # Skip semantics (fabricated-pass ban, NFR-QL-04)
//!
//! - `VOKRA_UTMOS_GGUF` **and** `VOKRA_UTMOS_REFDIR` unset → clean skip;
//! - set but unreadable/malformed → loud panic (a broken opt-in must never
//!   look like a skip).

use std::path::{Path, PathBuf};

use vokra_eval::metrics::utmos::Utmos;

const ENV_GGUF: &str = "VOKRA_UTMOS_GGUF";
const ENV_REFDIR: &str = "VOKRA_UTMOS_REFDIR";

/// Per-stage absolute tolerances = **measured worst case × 2**.
///
/// # Derivation (2026-07-20, M1 iMac / arm64 NEON, this repo's own run)
///
/// The reference's own re-run band is exactly `0.0` (upstream CPU inference is
/// deterministic — `manifest.json:rerun_band`), so the band being bounded here
/// is not re-run noise: it is the *cross-implementation* delta between torch
/// 2.8.0's kernels and Vokra's own, plus the torch-version env delta forced by
/// the pinned torch 1.11.0 having no macOS-arm64 wheel (M5-15 T38/T39).
///
/// Measured `max |Δ|` on the committed `ref-clip.wav` (99 frames), with the
/// bound set at ×2 per `feedback-honest-parity-atol`:
///
/// | stage            | measured  | bound   |
/// |------------------|-----------|---------|
/// | `conv_out`       | 1.378e-7  | 3e-7    |
/// | `feature_ln`     | 2.384e-6  | 5e-6    |
/// | `feat_proj`      | 7.391e-6  | 1.5e-5  |
/// | `pos_conv`       | 1.621e-5  | 4e-5    |
/// | `enc_in_ln`      | 3.759e-6  | 8e-6    |
/// | `enc_block_last` | 1.311e-6  | 3e-6    |
/// | `blstm_out`      | 4.172e-7  | 1e-6    |
/// | `head_out`       | 7.153e-7  | 2e-6    |
/// | `score`          | 1.192e-7  | 3e-7    |
///
/// These are float-association noise (1e-7 … 1e-5). A genuine port defect —
/// a swapped `ln1`/`ln2` mapping, a mis-folded weight-norm, a backwards LSTM
/// direction — lands at 1e-1 … 1e0, four to seven orders of magnitude above
/// the bound, so ×2 loses no diagnostic power.
///
/// **ISA caveat (Kokoro precedent, memory `project-kokoro-avx2-parity`):**
/// these numbers are calibrated on arm64/NEON. Kokoro showed that a different
/// CPU class can shift a parity delta *deterministically* (its AVX2 leg sat at
/// 4.34e-2 where AVX-512 sat at 1.58e-2). If this suite is ever run on x86 and
/// a stage exceeds its bound while the *score* stays at ~1e-7, that is an
/// ISA re-derivation, not a regression — re-measure on that ISA and record a
/// second calibrated row here. Do **not** widen these to cover both.
const STAGE_ATOL: &[(&str, f32)] = &[
    // Conv stack: 7 im2col-GEMM layers + GELU + the layer-0 GroupNorm.
    ("conv_out", 3.0e-7),
    ("feature_ln", 5.0e-6),
    ("feat_proj", 1.5e-5),
    // The weight-norm fold happens offline in the converter, so this stage
    // is also the check on that arithmetic.
    ("pos_conv", 4.0e-5),
    ("enc_in_ln", 8.0e-6),
    // 12 transformer blocks; post-norm keeps the accumulation bounded.
    ("enc_block_last", 3.0e-6),
    ("blstm_out", 1.0e-6),
    ("head_out", 2.0e-6),
];

/// Final-score tolerance = measured `1.192e-7` × ~2.5. The head mean-pools 99
/// frames and applies `×2 + 3`, so per-frame error averages down rather than
/// accumulating.
const SCORE_ATOL: f64 = 3.0e-7;

fn atol(stage: &str) -> f32 {
    STAGE_ATOL
        .iter()
        .find(|(n, _)| *n == stage)
        .map(|(_, a)| *a)
        .unwrap_or_else(|| panic!("no tolerance recorded for stage `{stage}`"))
}

/// Reads a flat little-endian f32 dump.
fn read_bin(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        panic!("reading reference tap {path:?}: {e} — a broken opt-in must not look like a skip")
    });
    assert!(
        bytes.len() % 4 == 0,
        "tap {path:?} is {} bytes, not a whole number of f32",
        bytes.len()
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Max |Δ| between two equal-length buffers, panicking on a length mismatch
/// (a shape disagreement is a real failure, not something to truncate past).
fn max_abs_delta(stage: &str, native: &[f32], reference: &[f32]) -> f32 {
    assert_eq!(
        native.len(),
        reference.len(),
        "stage `{stage}`: native has {} elements, reference {} — shapes disagree",
        native.len(),
        reference.len()
    );
    native
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn compare(stage: &str, native: &[f32], reference: &[f32], failures: &mut Vec<String>) {
    let d = max_abs_delta(stage, native, reference);
    let tol = atol(stage);
    let verdict = if d <= tol { "PASS" } else { "FAIL" };
    println!("  {stage:16} max|Δ| = {d:.6e}   atol = {tol:.1e}   {verdict}");
    if d > tol {
        failures.push(format!("{stage}: max|Δ| {d:.6e} > atol {tol:.1e}"));
    }
}

#[test]
fn parity_utmos_stages_vs_upstream() {
    let (Some(gguf), Some(refdir)) = (std::env::var_os(ENV_GGUF), std::env::var_os(ENV_REFDIR))
    else {
        eprintln!(
            "skipping parity_utmos_stages_vs_upstream: set {ENV_GGUF} (a converted \
             wav2vec2_regression.v1 GGUF) and {ENV_REFDIR} (the output dir of \
             tools/parity/utmos_dump_reference.py). Neither the checkpoint nor the reference \
             is committed — the weights are owner-gated (docs/license-audit.md) and a \
             fabricated fixture is banned (NFR-QL-04)."
        );
        return;
    };

    // From here the opt-in is explicit: everything is loud.
    let refdir = PathBuf::from(refdir);
    let manifest_path = refdir.join("manifest.json");
    let manifest_bytes =
        std::fs::read(&manifest_path).unwrap_or_else(|e| panic!("reading {manifest_path:?}: {e}"));
    let manifest = vokra_core::json::parse(&manifest_bytes)
        .unwrap_or_else(|e| panic!("malformed {manifest_path:?}: {e:?}"));

    let expected_score = manifest
        .get("score")
        .and_then(|v| match v {
            vokra_core::json::JsonValue::Float(f) => Some(*f),
            vokra_core::json::JsonValue::Int(i) => Some(*i as f64),
            _ => None,
        })
        .expect("manifest has a numeric `score`");
    let rerun_band = manifest
        .get("rerun_band")
        .and_then(|v| match v {
            vokra_core::json::JsonValue::Float(f) => Some(*f),
            vokra_core::json::JsonValue::Int(i) => Some(*i as f64),
            _ => None,
        })
        .expect("manifest has a numeric `rerun_band`");
    let clip_name = manifest
        .get("clip")
        .and_then(|v| v.as_str())
        .expect("manifest has `clip`")
        .to_owned();

    // The clip lives next to this repo's fixtures, not in the reference dir.
    let clip_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("parity")
        .join("utmos")
        .join(&clip_name);
    let wav = vokra_eval::wav::read_wav(&clip_path)
        .unwrap_or_else(|e| panic!("reading clip {clip_path:?}: {e}"));

    let m = Utmos::from_path(&gguf).unwrap_or_else(|e| panic!("loading UTMOS GGUF {gguf:?}: {e}"));
    assert!(
        !m.is_synthesized(),
        "refusing to compare synthesized weights against a real upstream reference (NFR-QL-04)"
    );
    let (score, taps) = m
        .score_with_taps(&wav.samples, wav.sample_rate)
        .unwrap_or_else(|e| panic!("native scoring failed: {e}"));

    println!("UTMOS stage parity ({} frames):", taps.t);
    let mut failures = Vec::new();
    compare(
        "conv_out",
        &taps.conv_out,
        &read_bin(&refdir.join("conv_out.bin")),
        &mut failures,
    );
    compare(
        "feature_ln",
        &taps.feature_ln,
        &read_bin(&refdir.join("feature_ln.bin")),
        &mut failures,
    );
    compare(
        "feat_proj",
        &taps.feat_proj,
        &read_bin(&refdir.join("feat_proj.bin")),
        &mut failures,
    );
    compare(
        "pos_conv",
        &taps.pos_conv,
        &read_bin(&refdir.join("pos_conv.bin")),
        &mut failures,
    );
    compare(
        "enc_in_ln",
        &taps.enc_in_ln,
        &read_bin(&refdir.join("enc_in_ln.bin")),
        &mut failures,
    );

    // The upstream encoder taps are `[t_padded, 1, d]` (fairseq pads the time
    // axis to a multiple of `required_seq_len_multiple = 2` and masks the pad
    // out of attention, then strips it). Compare only the real frames — the
    // padded row is discarded upstream too.
    let last = taps.enc_blocks.len() - 1;
    let reference_last = read_bin(&refdir.join(format!("enc_block_{last}.bin")));
    let d = taps.enc_blocks[last].len() / taps.t;
    compare(
        "enc_block_last",
        &taps.enc_blocks[last],
        &reference_last[..taps.t * d],
        &mut failures,
    );

    compare(
        "blstm_out",
        &taps.blstm_out,
        &read_bin(&refdir.join("blstm_out.bin")),
        &mut failures,
    );
    compare(
        "head_out",
        &taps.head_out,
        &read_bin(&refdir.join("head_out.bin")),
        &mut failures,
    );

    let score_delta = (score - expected_score).abs();
    println!(
        "  {:16} native = {score:.6}  reference = {expected_score:.6}  |Δ| = {score_delta:.6e}  \
         atol = {SCORE_ATOL:.1e}  (reference re-run band {rerun_band:.3e})",
        "score"
    );
    if score_delta > SCORE_ATOL {
        failures.push(format!(
            "score: |Δ| {score_delta:.6e} > atol {SCORE_ATOL:.1e} (native {score}, reference \
             {expected_score})"
        ));
    }

    assert!(
        failures.is_empty(),
        "UTMOS parity FAILED at {} stage(s):\n  {}\n\nDo NOT widen a tolerance to make this \
         green. Localize the stage, fix the port, and only if the divergence is an architectural \
         bound record a per-stage atol with its derivation (Kokoro PROSODY_F0_ATOL precedent).",
        failures.len(),
        failures.join("\n  ")
    );
}
