//! UTMOS parity vs the upstream reference — **flip-the-switch harness**
//! (M4-18 T09; NFR-QL-01/NFR-QL-04).
//!
//! # Status: weight-deferred (kickoff gate = NO-GO-defer)
//!
//! The M4-18 kickoff gate deferred the real UTMOS weights + license to a
//! v1.0.x patch (ADR `docs/adr/M4-18-utmos-gate.md`, gitignore-local), so
//! **no reference fixture is committed** — committing one would require
//! fabricating an expected score, which is exactly what NFR-QL-04 bans.
//! What lands now is the harness: fixture format, honest-atol plumbing,
//! the synthesized-weight refusal gate, and a cleanly-skipping gated test.
//!
//! # Flip recipe (owner, once T02 delivers the weights)
//!
//! 1. Generate the reference score offline against the pinned upstream
//!    SaruLab UTMOS22 implementation and commit
//!    `tests/parity/utmos/score.json` + the input clip (see
//!    `tests/parity/utmos/README.md` for the exact format and the
//!    honest-atol rule: reference reproduction error band × 1.5–2, never a
//!    CI-green-hunting constant — memory `feedback-honest-parity-atol`).
//! 2. Convert the checkpoint to a `vokra.utmos.*` GGUF (T05 converter,
//!    deferred with the weights).
//! 3. `VOKRA_UTMOS_GGUF=path/to/utmos.gguf cargo test -p vokra-eval --test
//!    parity_utmos` — the gated test stops skipping automatically.
//!
//! # Skip semantics (fabricated pass ban)
//!
//! - `VOKRA_UTMOS_GGUF` unset → the gated test **skips** with an explicit
//!   eprintln (the CI posture while the weights are deferred; the
//!   `parity-utmos.yml` workflow annotates the skip, it never synthesizes a
//!   pass).
//! - env set but fixture / clip missing or malformed → **loud panic** (the
//!   owner explicitly opted in; a broken opt-in must not look like a skip).
//! - synthesized weights → **refused** ([`native_score_for_parity`]): an
//!   in-memory synthesized scorer can never be compared against a real
//!   reference. (A GGUF *written from* synthesized values still loads as
//!   non-synthesized — the flag tracks construction provenance; fixture
//!   provenance + the README recipe carry the honesty obligation for real
//!   references.)

use std::path::PathBuf;

use vokra_core::json::{self, JsonValue};
use vokra_core::{Result, VokraError};
use vokra_eval::metrics::utmos::{ConvActivation, HeadPool, TransformerNorm, Utmos, UtmosConfig};

/// Env var pointing at the converted `vokra.utmos.*` GGUF (flip-the-switch).
const ENV_GGUF: &str = "VOKRA_UTMOS_GGUF";

/// Repo-root fixture dir (`tests/parity/utmos/`), following the
/// `parity_whisper.rs` convention.
fn fixtures_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = <repo>/crates/vokra-eval.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("parity")
        .join("utmos")
}

/// Parsed `score.json` fixture sidecar (format: `tests/parity/utmos/README.md`).
#[derive(Debug, PartialEq)]
struct UtmosScoreFixture {
    /// Input clip filename, relative to the fixture dir (mono WAV).
    clip: String,
    /// The clip's sample rate — must equal both the WAV header and the
    /// GGUF's `vokra.utmos.sample_rate` (mismatch is loud, FR-EX-08).
    sample_rate: u32,
    /// Upstream reference score for the clip.
    expected_score: f64,
    /// Honest absolute tolerance: reference reproduction error band
    /// × 1.5–2, recorded at generation time (README).
    atol: f64,
    /// Non-empty provenance: upstream repo + commit, generation date, and
    /// the measured reproduction band the atol was derived from.
    provenance: String,
}

/// Parses and validates the fixture JSON. Every violation is a loud
/// `Err(String)` naming the field — a malformed opt-in fixture must never
/// downgrade to a skip.
fn parse_fixture(bytes: &[u8]) -> std::result::Result<UtmosScoreFixture, String> {
    let root = json::parse(bytes).map_err(|e| format!("malformed JSON: {e:?}"))?;
    let field = |key: &str| {
        root.get(key)
            .ok_or_else(|| format!("missing required field `{key}`"))
    };
    let as_f64 = |v: &JsonValue, key: &str| match v {
        JsonValue::Int(i) => Ok(*i as f64),
        JsonValue::Float(f) => Ok(*f),
        _ => Err(format!("field `{key}` is not a number")),
    };

    let clip = field("clip")?
        .as_str()
        .ok_or("field `clip` is not a string")?
        .to_owned();
    if clip.is_empty() {
        return Err("field `clip` is empty".into());
    }
    let sample_rate_wide = field("sample_rate")?
        .as_u64()
        .ok_or("field `sample_rate` is not a non-negative integer")?;
    let sample_rate = u32::try_from(sample_rate_wide)
        .map_err(|_| format!("field `sample_rate` = {sample_rate_wide} does not fit in u32"))?;
    if sample_rate == 0 {
        return Err("field `sample_rate` must be > 0".into());
    }
    let expected_score = as_f64(field("expected_score")?, "expected_score")?;
    if !expected_score.is_finite() {
        return Err("field `expected_score` must be finite".into());
    }
    let atol = as_f64(field("atol")?, "atol")?;
    if !(atol.is_finite() && atol > 0.0) {
        return Err(format!(
            "field `atol` must be a finite positive number, got {atol} — a non-positive atol \
             hides every mismatch (and the value itself must be the honest reproduction-band \
             derivation from the README, never a green-hunting constant)"
        ));
    }
    let provenance = field("provenance")?
        .as_str()
        .ok_or("field `provenance` is not a string")?
        .to_owned();
    if provenance.trim().is_empty() {
        return Err(
            "field `provenance` is empty — the honest-atol rule requires upstream commit + \
             generation recipe + measured reproduction band"
                .into(),
        );
    }
    Ok(UtmosScoreFixture {
        clip,
        sample_rate,
        expected_score,
        atol,
        provenance,
    })
}

/// The comparison core: `|native - expected| <= atol`, with a diagnostic
/// message carrying all three numbers on failure.
fn parity_verdict(native: f64, expected: f64, atol: f64) -> std::result::Result<(), String> {
    if !native.is_finite() || !expected.is_finite() {
        return Err(format!(
            "scores must be finite: native = {native}, reference = {expected}"
        ));
    }
    let delta = (native - expected).abs();
    if delta <= atol {
        Ok(())
    } else {
        Err(format!(
            "|Δ| = {delta:.6e} > atol {atol:.6e} (native = {native}, reference = {expected})"
        ))
    }
}

/// Scores `clip` for a parity comparison, **refusing synthesized weights**:
/// comparing a seed-random skeleton against a real upstream reference could
/// only ever "pass" by fabrication (NFR-QL-04).
fn native_score_for_parity(m: &Utmos, clip: &[f32], sample_rate: u32) -> Result<f64> {
    if m.is_synthesized() {
        return Err(VokraError::InvalidArgument(
            "utmos parity: refusing to compare synthesized (seed-random) weights against a real \
             upstream reference — such a comparison could only pass by fabrication (NFR-QL-04); \
             load a GGUF converted from the real checkpoint instead"
                .to_owned(),
        ));
    }
    m.score(clip, sample_rate)
}

// ---------------------------------------------------------------------------
// Harness-logic unit tests (run everywhere, no weights involved)
// ---------------------------------------------------------------------------

/// A structurally-valid fixture body used by the parser tests. Field values
/// are placeholders for the parser only — no parity meaning.
fn valid_fixture_json() -> Vec<u8> {
    br#"{
        "clip": "ref-clip.wav",
        "sample_rate": 16000,
        "expected_score": 3.25,
        "atol": 0.02,
        "provenance": "sarulab-speech/UTMOS22 @ <commit>, generated 2026-XX-XX, reproduction band 0.01 x2"
    }"#
    .to_vec()
}

#[test]
fn fixture_parser_accepts_the_documented_shape() {
    let f = parse_fixture(&valid_fixture_json()).expect("valid fixture parses");
    assert_eq!(f.clip, "ref-clip.wav");
    assert_eq!(f.sample_rate, 16_000);
    assert_eq!(f.expected_score, 3.25);
    assert_eq!(f.atol, 0.02);
    assert!(f.provenance.contains("UTMOS22"));
}

#[test]
fn fixture_parser_names_missing_or_invalid_fields() {
    // Missing clip.
    let err = parse_fixture(
        br#"{"sample_rate":16000,"expected_score":3.0,"atol":0.02,"provenance":"x"}"#,
    )
    .expect_err("missing clip");
    assert!(err.contains("clip"), "got: {err}");

    // Non-positive atol can hide any mismatch — reject loudly.
    let err = parse_fixture(
        br#"{"clip":"a.wav","sample_rate":16000,"expected_score":3.0,"atol":0.0,"provenance":"x"}"#,
    )
    .expect_err("zero atol");
    assert!(err.contains("atol"), "got: {err}");
    let err = parse_fixture(
        br#"{"clip":"a.wav","sample_rate":16000,"expected_score":3.0,"atol":-0.1,"provenance":"x"}"#,
    )
    .expect_err("negative atol");
    assert!(err.contains("atol"), "got: {err}");

    // Empty provenance defeats the honest-atol traceability rule.
    let err = parse_fixture(
        br#"{"clip":"a.wav","sample_rate":16000,"expected_score":3.0,"atol":0.02,"provenance":""}"#,
    )
    .expect_err("empty provenance");
    assert!(err.contains("provenance"), "got: {err}");

    // Malformed JSON.
    let err = parse_fixture(b"{not json").expect_err("malformed JSON");
    assert!(err.contains("JSON"), "got: {err}");
}

#[test]
fn parity_verdict_accepts_within_atol_and_reports_mismatch() {
    parity_verdict(3.25, 3.26, 0.02).expect("within atol");
    parity_verdict(3.25, 3.25, 0.02).expect("exact match");
    // Boundary: |Δ| == atol passes (dyadic values keep it exact).
    parity_verdict(3.0, 3.0625, 0.0625).expect("at the boundary");

    let err = parity_verdict(3.25, 3.40, 0.02).expect_err("outside atol");
    assert!(err.contains("atol"), "got: {err}");
    assert!(err.contains("3.4"), "message carries the reference: {err}");

    let err = parity_verdict(f64::NAN, 3.0, 0.02).expect_err("non-finite native");
    assert!(err.contains("finite"), "got: {err}");
}

#[test]
fn synthesized_weights_are_refused_for_parity() {
    let config = UtmosConfig {
        sample_rate: 16_000,
        conv_channels: vec![4, 6],
        conv_kernels: vec![5, 3],
        conv_strides: vec![3, 2],
        conv_activation: ConvActivation::Gelu,
        n_layer: 1,
        n_head: 2,
        hidden_dim: 6,
        ffn_dim: 12,
        norm: TransformerNorm::Post,
        ln_eps: 1e-5,
        head_dims: vec![4, 1],
        head_pool: HeadPool::MeanAfter,
        head_scale: 1.0,
        head_offset: 3.0,
    };
    let m = Utmos::synthesized(config, 7).expect("synthesized skeleton");
    let clip: Vec<f32> = (0..64)
        .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 16_000.0).sin())
        .collect();
    // The plain scorer runs fine over synthesized weights …
    m.score(&clip, 16_000).expect("plain scoring works");
    // … but the parity path refuses them (fabricated pass ban).
    let err = native_score_for_parity(&m, &clip, 16_000).expect_err("must refuse");
    assert!(matches!(err, VokraError::InvalidArgument(_)), "got: {err}");
    assert!(
        err.to_string().contains("synthesized"),
        "the refusal must say why: {err}"
    );
}

// ---------------------------------------------------------------------------
// The gated parity test (skips until the owner flips the switch)
// ---------------------------------------------------------------------------

#[test]
fn parity_utmos_vs_reference() {
    let Some(gguf_path) = std::env::var_os(ENV_GGUF) else {
        eprintln!(
            "skipping parity_utmos_vs_reference: {ENV_GGUF} unset — the M4-18 kickoff gate \
             deferred the UTMOS weights (v1.0.x patch); once the owner lands T02 + the T05 \
             converter, set {ENV_GGUF} to the converted GGUF and commit \
             tests/parity/utmos/score.json (see that dir's README.md)"
        );
        return;
    };

    // From here on the owner explicitly opted in: everything is loud.
    let dir = fixtures_dir();
    let fixture_path = dir.join("score.json");
    let bytes = std::fs::read(&fixture_path).unwrap_or_else(|e| {
        panic!(
            "{ENV_GGUF} is set but the fixture {fixture_path:?} is unreadable ({e}) — a broken \
             opt-in must not look like a skip"
        )
    });
    let fixture = parse_fixture(&bytes).unwrap_or_else(|e| panic!("fixture {fixture_path:?}: {e}"));

    let clip_path = dir.join(&fixture.clip);
    let wav = vokra_eval::wav::read_wav(&clip_path)
        .unwrap_or_else(|e| panic!("reading clip {clip_path:?}: {e}"));
    assert_eq!(
        wav.sample_rate, fixture.sample_rate,
        "fixture sample_rate vs WAV header mismatch (no silent resample, FR-EX-08)"
    );

    let m = Utmos::from_path(&gguf_path)
        .unwrap_or_else(|e| panic!("loading UTMOS GGUF {gguf_path:?}: {e}"));
    let native = native_score_for_parity(&m, &wav.samples, wav.sample_rate)
        .unwrap_or_else(|e| panic!("native scoring failed: {e}"));

    parity_verdict(native, fixture.expected_score, fixture.atol).unwrap_or_else(|msg| {
        panic!(
            "UTMOS parity FAILED: {msg}\nprovenance: {}\n(if the divergence is an architectural \
             bound, record a per-fixture honest atol with rationale — Kokoro PROSODY_F0_ATOL \
             precedent — never widen it to hunt a green)",
            fixture.provenance
        )
    });
}
