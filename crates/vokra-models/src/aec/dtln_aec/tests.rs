//! Unit tests for the DTLN-AEC runtime binder.
//!
//! Every test builds a minimal valid GGUF in memory (no fixture files
//! on disk); the fenced surfaces are: constant handshake with the
//! converter, arch-tag mismatch refusal, missing-chunk refusal,
//! loud-partial UnsupportedOp message shape, and FR-EX-08 length
//! mismatch precedence.

use super::*;
use vokra_core::VokraError;
use vokra_core::gguf::{GgufBuilder, GgufFile, chunks};

/// The runtime binder's arch / name / category constants must exactly
/// match the converter's. A drift would land as a load-time
/// "wrong arch" refusal on every real checkpoint — the tests pin the
/// handshake here so a converter-side rename gets caught at the
/// converter's own unit tests + the runtime's own unit tests + the
/// combined smoke on the shared arch string.
#[test]
fn arch_and_provenance_constants_match_converter() {
    assert_eq!(ARCH, "dtln_aec");
    assert_eq!(DEFAULT_NAME, "dtln-aec");
    assert_eq!(CATEGORY, "aec");
    assert_eq!(N_FFT, 512);
    assert_eq!(BLOCK_LEN, 512);
    assert_eq!(HOP, 128);
    assert_eq!(SAMPLE_RATE, 16_000);
    assert_eq!(F_BINS, 257);
    assert_eq!(KEY_VARIANT_LSTM_UNITS, "vokra.dtln_aec.lstm_units");
    // The primary source URLs must include the two identifiers cited
    // in the loud-partial message shape.
    assert!(PRIMARY_SOURCE_GITHUB.contains("github.com/breizhn/DTLN-aec"));
    assert!(PRIMARY_SOURCE_ARXIV.contains("arxiv.org/abs/2010.15754"));
    assert!(PRIMARY_SOURCE_PAPER.contains("INTERSPEECH 2021"));
}

/// Builds a valid GGUF in memory with the given arch tag and (optional)
/// stamped `lstm_units` chunk. Returns the fully-parsed
/// [`GgufFile`] the loader consumes. Uses `add_u32` because
/// `GgufBuilder` does not expose `add_u64` today — the reader-side
/// `as_u64()` losslessly widens U32.
fn build_gguf(arch: Option<&str>, lstm_units: Option<u32>) -> GgufFile {
    let mut b = GgufBuilder::new();
    if let Some(a) = arch {
        b.add_string(chunks::KEY_MODEL_ARCH, a);
    }
    b.add_string(chunks::KEY_MODEL_NAME, DEFAULT_NAME);
    if let Some(u) = lstm_units {
        b.add_u32(KEY_VARIANT_LSTM_UNITS, u);
    }
    let bytes = b.to_bytes().expect("build gguf");
    GgufFile::parse(bytes).expect("parse gguf")
}

/// Missing `vokra.model.arch` chunk must fail loudly at load time, not
/// silently default to the DTLN-AEC arch.
#[test]
fn from_gguf_rejects_missing_arch_key_loudly() {
    let gguf = build_gguf(None, Some(128));
    let err = DtlnAec::from_gguf(&gguf).expect_err("expected ModelLoad on missing arch");
    match err {
        VokraError::ModelLoad(msg) => {
            assert!(
                msg.contains("vokra.model.arch"),
                "error must name the missing chunk (got: {msg})"
            );
        }
        other => panic!("expected VokraError::ModelLoad, got {other:?}"),
    }
}

/// A wrong arch tag (e.g. the sibling `nkf_aec`) must fail loudly at
/// load time with a clear "wrong arch" message — the arch tag is the
/// first fence against a mis-fed GGUF (silently accepting would try to
/// interpret NKF-AEC's Kalman tensors as DTLN-AEC's LSTM tensors, which
/// is a wrong-topology bug not a wrong-shape bug so downstream tensor-
/// name lookups would produce noisy "missing tensor" errors rather
/// than the clean root cause).
#[test]
fn from_gguf_rejects_wrong_arch_loudly() {
    let gguf = build_gguf(Some("nkf_aec"), Some(128));
    let err = DtlnAec::from_gguf(&gguf).expect_err("expected ModelLoad on wrong arch");
    match err {
        VokraError::ModelLoad(msg) => {
            assert!(
                msg.contains("nkf_aec") && msg.contains("dtln_aec"),
                "error must cite BOTH the observed arch and the expected arch \
                 (got: {msg})"
            );
        }
        other => panic!("expected VokraError::ModelLoad, got {other:?}"),
    }
}

/// A stamped `lstm_units` that doesn't match any known upstream release
/// width (128 / 256 / 512) must fail loudly — silent-default to
/// Units128 would leak into tensor-shape validation errors much later
/// in the pipeline (fail-closed per FR-EX-08).
#[test]
fn from_gguf_rejects_unknown_lstm_units_loudly() {
    let gguf = build_gguf(Some("dtln_aec"), Some(384)); // not 128/256/512
    let err = DtlnAec::from_gguf(&gguf).expect_err("expected ModelLoad on unknown units");
    match err {
        VokraError::ModelLoad(msg) => {
            assert!(
                msg.contains("384") && msg.contains("128") && msg.contains("512"),
                "error must cite the stamped value AND the known-good range (got: {msg})"
            );
        }
        other => panic!("expected VokraError::ModelLoad, got {other:?}"),
    }
}

/// Missing `vokra.dtln_aec.lstm_units` chunk must fail loudly — this
/// is the fail-closed guard against a stale GGUF produced by a
/// pre-Wave-6 converter that predates chunk stamping (silently
/// defaulting to Units128 would misroute a real Units512 checkpoint
/// on the loud-partial arm and land noisy dim errors much later).
#[test]
fn from_gguf_rejects_missing_lstm_units_chunk_loudly() {
    let gguf = build_gguf(Some("dtln_aec"), None);
    let err = DtlnAec::from_gguf(&gguf).expect_err("expected ModelLoad on missing chunk");
    match err {
        VokraError::ModelLoad(msg) => {
            assert!(
                msg.contains("vokra.dtln_aec.lstm_units"),
                "error must name the missing chunk (got: {msg})"
            );
        }
        other => panic!("expected VokraError::ModelLoad, got {other:?}"),
    }
}

/// Loud-partial pinning: on a valid GGUF with a real variant width,
/// `process(mic, farend)` must return
/// [`VokraError::UnsupportedOp`] naming (i) the generic LSTM primitive
/// gap in `vokra_ops`, (ii) the four wiring pieces still owed, and
/// (iii) the primary source URLs (canary_qwen precedent). This is the
/// core loud-partial contract per CLAUDE.md 教訓 (a).
#[test]
fn process_returns_unsupported_op_naming_lstm_primitive_gap_and_primary_source() {
    let gguf = build_gguf(Some("dtln_aec"), Some(128));
    let engine = DtlnAec::from_gguf(&gguf).expect("bind gguf");
    let mic = vec![0.0f32; 512];
    let farend = vec![0.0f32; 512];
    let err = engine
        .process(&mic, &farend)
        .expect_err("loud-partial UnsupportedOp expected");
    match err {
        VokraError::UnsupportedOp(msg) => {
            // (i) primitive gap named
            assert!(
                msg.contains("generic") && msg.contains("LSTM"),
                "message must name the generic LSTM primitive gap (got: {msg})"
            );
            // (ii) wiring pieces named — at least one of the four
            assert!(
                msg.contains("LstmCell") && msg.contains("STFT-domain"),
                "message must name the primitive + at least the STFT stage \
                 (got: {msg})"
            );
            assert!(
                msg.contains("time-domain") && msg.contains("AecEngine"),
                "message must name the time-domain stage + the AecEngine \
                 trait pieces still owed (got: {msg})"
            );
            // (iii) primary source URLs cited verbatim
            assert!(
                msg.contains("github.com/breizhn/DTLN-aec"),
                "message must cite the primary source repo (got: {msg})"
            );
            assert!(
                msg.contains("arxiv.org/abs/2010.15754"),
                "message must cite the arXiv paper (got: {msg})"
            );
            assert!(
                msg.contains("INTERSPEECH 2021"),
                "message must cite the paper venue (got: {msg})"
            );
            // "loud-partial" self-label per the canary_qwen precedent
            assert!(
                msg.contains("loud-partial"),
                "message must self-label as loud-partial (got: {msg})"
            );
            // FR-EX-08 posture named
            assert!(
                msg.contains("FR-EX-08"),
                "message must cite FR-EX-08 (got: {msg})"
            );
        }
        other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
    }
}

/// FR-EX-08 precedence: `mic.len() != farend.len()` must fire an
/// [`VokraError::InvalidArgument`] BEFORE the loud-partial
/// UnsupportedOp arm. This is a concrete inputs → wrong-output
/// scenario: a caller pushing `mic.len()=512, farend.len()=256`
/// should get "length mismatch" (a fixable-on-caller-side error),
/// not "generic LSTM primitive missing" (a fixable-on-vokra-side
/// error) — mixing the two would hide the real caller bug.
#[test]
fn process_rejects_length_mismatch_before_unsupported_op() {
    let gguf = build_gguf(Some("dtln_aec"), Some(128));
    let engine = DtlnAec::from_gguf(&gguf).expect("bind gguf");
    let mic = vec![0.0f32; 512];
    let farend = vec![0.0f32; 256]; // deliberate mismatch
    let err = engine
        .process(&mic, &farend)
        .expect_err("expected InvalidArgument on length mismatch");
    match err {
        VokraError::InvalidArgument(msg) => {
            assert!(
                msg.contains("512") && msg.contains("256"),
                "error must cite both lengths (got: {msg})"
            );
            assert!(
                msg.contains("mic.len()") && msg.contains("farend.len()"),
                "error must name both slots (got: {msg})"
            );
            assert!(
                msg.contains("FR-EX-08"),
                "error must cite FR-EX-08 posture (got: {msg})"
            );
        }
        other => panic!(
            "expected VokraError::InvalidArgument (length mismatch), got {other:?} — \
             the FR-EX-08 gate must fire BEFORE the loud-partial UnsupportedOp"
        ),
    }
}

/// Empty PCM must fire an [`VokraError::InvalidArgument`] loudly (both
/// slots — mic OR farend — trigger the guard separately so callers get
/// the specific slot named).
#[test]
fn process_rejects_empty_pcm_loudly() {
    let gguf = build_gguf(Some("dtln_aec"), Some(128));
    let engine = DtlnAec::from_gguf(&gguf).expect("bind gguf");
    let empty = Vec::<f32>::new();
    let full = vec![0.0f32; 512];

    // Empty mic
    let err_mic = engine
        .process(&empty, &full)
        .expect_err("expected InvalidArgument on empty mic");
    match err_mic {
        VokraError::InvalidArgument(msg) => {
            assert!(msg.contains("mic"), "error must name `mic` (got: {msg})");
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }

    // Empty farend
    let err_far = engine
        .process(&full, &empty)
        .expect_err("expected InvalidArgument on empty farend");
    match err_far {
        VokraError::InvalidArgument(msg) => {
            assert!(
                msg.contains("farend"),
                "error must name `farend` (got: {msg})"
            );
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

/// All three known variants (`Units128` / `Units256` / `Units512`)
/// round-trip through the loader: a GGUF stamped with each width
/// binds to the matching [`DtlnVariant`] on the config. This is the
/// counterpart of the converter's `variant_roundtrip_pins_all_three_widths`
/// test on the loader side (encoder writes `lstm_units`, loader reads
/// it back and re-derives the variant).
#[test]
fn variant_roundtrip_pins_all_three_widths_at_load_time() {
    for (units, expected_variant) in [
        (128u32, DtlnVariant::Units128),
        (256u32, DtlnVariant::Units256),
        (512u32, DtlnVariant::Units512),
    ] {
        let gguf = build_gguf(Some("dtln_aec"), Some(units));
        let engine = DtlnAec::from_gguf(&gguf).expect("bind gguf");
        assert_eq!(engine.config().variant, expected_variant);
        assert_eq!(engine.config().variant.lstm_units(), units as usize);
        // Fixed dims must always come out of the upstream_default
        // path regardless of variant.
        assert_eq!(engine.config().n_fft, N_FFT);
        assert_eq!(engine.config().hop, HOP);
        assert_eq!(engine.config().sample_rate, SAMPLE_RATE);
    }
}

/// [`DtlnAecConfig::validate`] rejects zero-valued fields loudly.
/// Since callers cannot construct arbitrary configs today (the struct
/// is `#[non_exhaustive]` and only `upstream_default` produces one),
/// this test exercises the trait method directly to pin the validation
/// contract for the future variant-taking API surface.
#[test]
fn config_validate_rejects_zero_fields_loudly() {
    // Mutate through a builder-style clone (all fields are `pub`).
    let mut cfg = DtlnAecConfig::upstream_default(DtlnVariant::Units128);
    cfg.n_fft = 0;
    let err = cfg.validate().expect_err("n_fft=0 must fail");
    match err {
        VokraError::InvalidArgument(msg) => {
            assert!(msg.contains("n_fft"), "must name n_fft (got: {msg})");
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }

    let mut cfg = DtlnAecConfig::upstream_default(DtlnVariant::Units128);
    cfg.hop = 0;
    let err = cfg.validate().expect_err("hop=0 must fail");
    match err {
        VokraError::InvalidArgument(msg) => {
            assert!(msg.contains("hop"), "must name hop (got: {msg})");
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }

    // block_len > n_fft is loud
    let mut cfg = DtlnAecConfig::upstream_default(DtlnVariant::Units128);
    cfg.block_len = cfg.n_fft + 1;
    let err = cfg.validate().expect_err("block_len > n_fft must fail");
    match err {
        VokraError::InvalidArgument(msg) => {
            assert!(
                msg.contains("block_len") && msg.contains("n_fft"),
                "must name both (got: {msg})"
            );
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}
