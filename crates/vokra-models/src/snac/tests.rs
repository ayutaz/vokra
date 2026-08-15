//! Tests for the SNAC runtime binder — round-trip on the variant
//! discriminator, negative-space round-trip on the loud-partial gates.
//!
//! # What "round-trip" means here
//!
//! The task spec asks for a "round-trip" unit test. On real PCM this
//! would be `encode(decode(codes)) == pcm` (up to codec quantization
//! error), but the encoder / decoder body primitives do not exist in
//! `vokra-ops` today (see the module doc + [`Snac::encode`] /
//! [`Snac::decode`] rustdoc). Fabricating a real-PCM round-trip would
//! violate CLAUDE.md 教訓 (a) ("loud-partial は fake-complete より
//! honest").
//!
//! The round-trip semantics we *can* honestly test here are:
//!
//! 1. **Variant round-trip**: `from_gguf` accepts both `"24khz"` and
//!    `"44khz"` tag values, and every stamped variant produces the
//!    correct per-variant config axes.
//! 2. **Loud-error negative-space round-trip**: every stated blocker
//!    (missing arch / wrong arch / missing variant / unknown variant /
//!    unsupported forward surface) fires at its documented surface
//!    point, in the documented error variant. A silent stub swap
//!    (e.g. someone replacing the loud gate with a `Vec::new()`
//!    return) would break these tests immediately.

use super::*;
use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile};

// ---------------------------------------------------------------------------
// Fixture helpers — hand-assembled GGUFs (bypass the converter for isolation;
// the converter e2e lives in `crates/vokra-convert/src/models/snac.rs::tests`).
// ---------------------------------------------------------------------------

/// Builds a minimal SNAC GGUF carrying the arch tag + variant tag +
/// provenance stamp — the same three chunks every real converter output
/// carries. `weight_license_class` is written under
/// `vokra.provenance.weight_license` (or omitted if `None`).
fn snac_gguf_for(variant_tag: &str, weight_license_class: Option<LicenseClass>) -> GgufFile {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, &format!("snac-{variant_tag}"));
    b.add_string(KEY_SNAC_VARIANT, variant_tag);
    if let Some(cls) = weight_license_class {
        b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
    }
    // Also stamp a defensive extra tensor so a downstream reader that
    // walks tensors on a real-weight GGUF does not accidentally
    // short-circuit on an empty-file heuristic. Value is deliberately
    // arbitrary — no primitive today consumes it.
    b.add_tensor(
        "encoder.block.0.block.0.weight",
        GgmlType::F32,
        vec![2, 3],
        vec![0u8; 2 * 3 * 4],
    )
    .expect("add_tensor");
    GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
}

// ---------------------------------------------------------------------------
// Variant round-trip — every SnacVariant round-trips through from_gguf
// ---------------------------------------------------------------------------

#[test]
fn from_gguf_reads_hz24_variant() {
    let file = snac_gguf_for("24khz", Some(LicenseClass::Permissive));
    let snac = Snac::from_gguf(&file).expect("Hz24 GGUF must bind");
    assert_eq!(snac.variant(), SnacVariant::Hz24);
    let cfg = snac.config();
    assert_eq!(cfg.variant, SnacVariant::Hz24);
    assert_eq!(cfg.sample_rate, 24_000);
    assert_eq!(cfg.n_stages, 3);
    // Full slice includes the trailing 0 slot (honest — a caller
    // iterating past `n_stages` must see the unpopulated marker, never
    // a fabricated stride).
    assert_eq!(cfg.vq_strides, [4, 2, 1, 0]);
    // Active slice trims to the 3 real Hz24 stages.
    assert_eq!(cfg.active_vq_strides(), &[4, 2, 1][..]);
    assert_eq!(snac.weight_license(), LicenseClass::Permissive);
}

#[test]
fn from_gguf_reads_hz44_variant() {
    let file = snac_gguf_for("44khz", Some(LicenseClass::Permissive));
    let snac = Snac::from_gguf(&file).expect("Hz44 GGUF must bind");
    assert_eq!(snac.variant(), SnacVariant::Hz44);
    let cfg = snac.config();
    assert_eq!(cfg.variant, SnacVariant::Hz44);
    assert_eq!(cfg.sample_rate, 44_100);
    assert_eq!(cfg.n_stages, 4);
    assert_eq!(cfg.vq_strides, [8, 4, 2, 1]);
    assert_eq!(cfg.active_vq_strides(), &[8, 4, 2, 1][..]);
    assert_eq!(snac.weight_license(), LicenseClass::Permissive);
}

#[test]
fn from_gguf_defaults_weight_license_to_unknown_when_missing() {
    // A GGUF missing `vokra.provenance.weight_license` reads back as
    // `Unknown` (fail-closed at the compliance gate). Never a silent
    // Permissive default.
    let file = snac_gguf_for("24khz", None);
    let snac = Snac::from_gguf(&file).expect("missing provenance must still bind");
    assert_eq!(snac.weight_license(), LicenseClass::Unknown);
}

// ---------------------------------------------------------------------------
// Loud-error round-trip — arch / variant validation (FR-EX-08)
// ---------------------------------------------------------------------------

#[test]
fn from_gguf_rejects_wrong_arch() {
    // A DAC / Mimi / WavTokenizer GGUF handed to the SNAC binder by
    // mistake must fail loud with a specific message rather than
    // silently mis-binding.
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, "dac");
    b.add_string(KEY_SNAC_VARIANT, "24khz");
    let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
    let err = Snac::from_gguf(&file).expect_err("wrong arch must be rejected");
    match err {
        VokraError::ModelLoad(m) => {
            assert!(
                m.contains("`dac`") && m.contains("`snac`"),
                "message must name both the got and expected arch tags, got `{m}`"
            );
        }
        other => panic!("expected VokraError::ModelLoad, got {other:?}"),
    }
}

#[test]
fn from_gguf_rejects_missing_arch() {
    // A GGUF with no `vokra.model.arch` at all — a converter that
    // forgot to stamp it must be caught here, not surface as a
    // downstream "missing tensor".
    let mut b = GgufBuilder::new();
    b.add_string(KEY_SNAC_VARIANT, "24khz");
    let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
    let err = Snac::from_gguf(&file).expect_err("missing arch must be rejected");
    match err {
        VokraError::ModelLoad(m) => {
            assert!(
                m.contains("vokra.model.arch"),
                "message must name the missing arch key, got `{m}`"
            );
        }
        other => panic!("expected VokraError::ModelLoad, got {other:?}"),
    }
}

#[test]
fn from_gguf_rejects_missing_variant() {
    // Correct arch but missing `vokra.snac.variant` — a partially-
    // stamped GGUF must be caught here, not silently defaulted to
    // Hz24 (which would corrupt every downstream code-rate).
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
    let err = Snac::from_gguf(&file).expect_err("missing variant must be rejected");
    match err {
        VokraError::ModelLoad(m) => {
            assert!(
                m.contains(KEY_SNAC_VARIANT),
                "message must name the missing variant key, got `{m}`"
            );
            // Both accepted tag values MUST appear in the hint so the
            // reader can pick the correct one without cross-referencing
            // rustdoc.
            assert!(m.contains(VARIANT_TAG_HZ24), "hint missing Hz24 tag: `{m}`");
            assert!(m.contains(VARIANT_TAG_HZ44), "hint missing Hz44 tag: `{m}`");
        }
        other => panic!("expected VokraError::ModelLoad, got {other:?}"),
    }
}

#[test]
fn from_gguf_rejects_unknown_variant_tag() {
    // A rogue converter or a future 3rd variant this runtime does not
    // dispatch on — never a silent default.
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(KEY_SNAC_VARIANT, "16khz"); // not a real SNAC variant
    let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
    let err = Snac::from_gguf(&file).expect_err("unknown variant must be rejected");
    match err {
        VokraError::ModelLoad(m) => {
            assert!(
                m.contains("`16khz`"),
                "message must echo the bad tag, got `{m}`"
            );
            assert!(
                m.contains(VARIANT_TAG_HZ24) && m.contains(VARIANT_TAG_HZ44),
                "message must list the accepted tags, got `{m}`"
            );
        }
        other => panic!("expected VokraError::ModelLoad, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Loud-partial round-trip — encode / decode / decode_codes_to_features
// each fire at their documented surface point with the documented variant.
// A silent stub swap (replacing the loud gate with a `Vec::new()` return)
// would break these tests immediately.
// ---------------------------------------------------------------------------

#[test]
fn encode_returns_unsupported_op_with_primitive_gap_message() {
    let file = snac_gguf_for("24khz", Some(LicenseClass::Permissive));
    let snac = Snac::from_gguf(&file).unwrap();
    // Give the encode path a legitimate-shape PCM buffer so the loud-
    // partial gate is what fires, not the sample-rate mismatch guard.
    let pcm = vec![0.0f32; 24_000]; // 1 s of silence at Hz24
    let err = snac
        .encode(&pcm, 24_000)
        .expect_err("encode must loud-partial");
    match err {
        VokraError::UnsupportedOp(m) => {
            // Grep-style substring assert — a silent stub swap would drop
            // these substrings, failing the test loudly.
            assert!(
                m.contains("encoder Conv1D"),
                "message must name the encoder Conv1D gap, got `{m}`"
            );
            assert!(
                m.contains("VectorQuantize.forward"),
                "message must name the VectorQuantize gap, got `{m}`"
            );
            // Variant-specific rate list must be present (Hz24 = [2,4,8,8]).
            assert!(
                m.contains("[2, 4, 8, 8]"),
                "message must cite the Hz24 encoder_rates, got `{m}`"
            );

            // --- Anti-rot guard: an earlier revision claimed "none of the
            // --- required primitives are in `vokra-ops` today", which is
            // --- false for the convolution — a conv1d kernel exists and is
            // --- reachable through the Compute seam. The negative assertion
            // --- keeps that phrasing from rotting back in.
            assert!(
                !m.contains("none of the required primitives"),
                "stale claim — `vokra_backend_cpu::kernels::conv1d_f32` exists and is \
                 reachable through `Compute::conv1d_f32`, got `{m}`"
            );
            assert!(
                m.contains("NOT missing, do not re-report") && m.contains("conv1d_f32"),
                "message must name the convolution primitive that already exists so the \
                 reader wires the seam instead of writing one, got `{m}`"
            );
        }
        other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
    }
}

#[test]
fn encode_rejects_sample_rate_mismatch_before_loud_partial() {
    // A sample-rate mismatch fires as InvalidArgument BEFORE the
    // encoder loud-partial gate — a caller passing the wrong SR sees
    // the SR error (which they can fix by resampling), not the deeper
    // "primitive missing" error (which they can't fix at all).
    let file = snac_gguf_for("24khz", Some(LicenseClass::Permissive));
    let snac = Snac::from_gguf(&file).unwrap();
    let pcm = vec![0.0f32; 24_000];
    let err = snac
        .encode(&pcm, 48_000)
        .expect_err("wrong SR must be rejected");
    match err {
        VokraError::InvalidArgument(m) => {
            assert!(
                m.contains("48000") && m.contains("24000"),
                "message must name both the got and expected SR, got `{m}`"
            );
        }
        other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
    }
}

#[test]
fn decode_returns_unsupported_op_with_primitive_gap_message() {
    let file = snac_gguf_for("44khz", Some(LicenseClass::Permissive));
    let snac = Snac::from_gguf(&file).unwrap();
    // Give decode a legitimate outer shape (4 stages for Hz44) so the
    // loud-partial gate fires, not the stage-count guard.
    let codes: Vec<Vec<u32>> = vec![vec![0u32; 1], vec![0u32; 2], vec![0u32; 4], vec![0u32; 8]];
    let err = snac.decode(&codes).expect_err("decode must loud-partial");
    match err {
        VokraError::UnsupportedOp(m) => {
            assert!(
                m.contains("decoder Conv1D"),
                "message must name the decoder Conv1D gap, got `{m}`"
            );
            assert!(
                m.contains("feature→PCM"),
                "message must call out the terminal PCM synthesis gap, got `{m}`"
            );
            // Variant-specific rate list must be present (Hz44 = [8,8,3,2]).
            assert!(
                m.contains("[8, 8, 3, 2]"),
                "message must cite the Hz44 decoder_rates, got `{m}`"
            );
            // Must forward the reader to the intermediate-features seam.
            assert!(
                m.contains("decode_codes_to_features"),
                "message must forward the reader to the intermediate seam, got `{m}`"
            );

            // --- Anti-rot guard (mirror of the `beat_this` / `squim` guards).
            //
            // An earlier revision listed "Snake activation on every decoder
            // block" among the MISSING pieces. `vokra_ops::snake_activation_f32`
            // and `vokra_ops::snake_beta_f32` are landed public primitives
            // (Metal-covered via `HotOp::SnakeActivation`), so that entry sent
            // the next reader off to write an activation that already exists.
            // Asserting it is ABSENT is the load-bearing half — omission alone
            // is not enforceable.
            assert!(
                !m.contains("(b) Snake activation"),
                "stale claim — `vokra_ops::snake_activation_f32` is a landed, \
                 Metal-covered primitive, got `{m}`"
            );
            assert!(
                m.contains("NOT missing, do not re-report"),
                "message must positively disclaim the resolved blockers rather \
                 than merely omitting them, got `{m}`"
            );
            assert!(
                m.contains("snake_activation_f32") && m.contains("conv1d_f32"),
                "message must name the primitives that already exist so the reader \
                 does not rewrite them, got `{m}`"
            );
        }
        other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
    }
}

#[test]
fn decode_rejects_wrong_stage_count_before_loud_partial() {
    // A caller passing 3 stages to a Hz44 binder (which needs 4) sees
    // the shape error, not the loud-partial gate — the shape error is
    // actionable ("pad the stage list"), the loud-partial is not.
    let file = snac_gguf_for("44khz", Some(LicenseClass::Permissive));
    let snac = Snac::from_gguf(&file).unwrap();
    let codes: Vec<Vec<u32>> = vec![vec![0u32; 1], vec![0u32; 2], vec![0u32; 4]]; // 3, not 4
    let err = snac
        .decode(&codes)
        .expect_err("wrong stage count must be rejected");
    match err {
        VokraError::InvalidArgument(m) => {
            assert!(
                m.contains("3") && m.contains("4"),
                "message must name both the got and expected stage counts, got `{m}`"
            );
        }
        other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
    }
}

#[test]
fn decode_codes_to_features_reports_derived_tensor_gap() {
    let file = snac_gguf_for("24khz", Some(LicenseClass::Permissive));
    let snac = Snac::from_gguf(&file).unwrap();
    let codes: Vec<Vec<u32>> = vec![vec![0u32; 1], vec![0u32; 2], vec![0u32; 4]];
    let err = snac
        .decode_codes_to_features(&codes)
        .expect_err("intermediate features must loud-partial");
    // ModelLoad (not UnsupportedOp) because the primitive itself exists
    // in vokra-ops — the block is a converter-side missing tensor.
    match err {
        VokraError::ModelLoad(m) => {
            assert!(
                m.contains("vokra.snac.codebook_tables"),
                "message must name the derived codebook-tables tensor, got `{m}`"
            );
            assert!(
                m.contains("out_proj_weight") && m.contains("out_proj_bias"),
                "message must name the derived out_proj tensors, got `{m}`"
            );
            assert!(
                m.contains("weight_norm") || m.contains("weight-norm"),
                "message must call out the weight-norm folding step, got `{m}`"
            );
            assert!(
                m.contains("vokra_ops::SnacDecoder"),
                "message must forward the reader to the existing primitive, got `{m}`"
            );
        }
        other => panic!("expected VokraError::ModelLoad, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// SnacVariant / SnacConfig direct property tests
// ---------------------------------------------------------------------------

#[test]
fn variant_tag_round_trips() {
    for v in [SnacVariant::Hz24, SnacVariant::Hz44] {
        assert_eq!(SnacVariant::from_tag(v.tag()), Some(v));
    }
    assert_eq!(SnacVariant::from_tag("16khz"), None);
    assert_eq!(SnacVariant::from_tag(""), None);
    assert_eq!(SnacVariant::from_tag("24"), None);
}

#[test]
fn variant_tags_are_distinct() {
    // Copy-paste guard — every SnacVariant maps to a distinct tag
    // (mirror of the converter's `every_variant_has_distinct_stamps`
    // test). A tuple that ties two variants to the same string would
    // silently collapse the runtime dispatch.
    let variants = [SnacVariant::Hz24, SnacVariant::Hz44];
    for i in 0..variants.len() {
        for j in (i + 1)..variants.len() {
            let a = variants[i];
            let b = variants[j];
            assert_ne!(a.tag(), b.tag(), "tags must differ ({a:?} vs {b:?})");
        }
    }
}

#[test]
fn config_active_slice_never_contains_zero_stride() {
    // A caller iterating `active_vq_strides()` MUST never see the
    // trailing 0 slot — that slot exists to make `[u32; 4]` honestly
    // represent "unpopulated" on Hz24, but consuming it as a real
    // stride would divide the base frame rate by zero.
    for v in [SnacVariant::Hz24, SnacVariant::Hz44] {
        let cfg = SnacConfig::for_variant(v);
        for (i, &s) in cfg.active_vq_strides().iter().enumerate() {
            assert!(
                s > 0,
                "variant {v:?} active stride[{i}] = 0 (would divide the base \
                 frame rate by zero)"
            );
        }
    }
}

#[test]
fn config_stride_axes_match_upstream_config_json() {
    // Primary-source pin — the axes must match what the converter
    // transcribed from the upstream `config.json`. A silent drift in
    // either the converter or this binder would fail here.
    let hz24 = SnacConfig::for_variant(SnacVariant::Hz24);
    assert_eq!(hz24.sample_rate, 24_000);
    assert_eq!(hz24.active_vq_strides(), &[4, 2, 1][..]);

    let hz44 = SnacConfig::for_variant(SnacVariant::Hz44);
    assert_eq!(hz44.sample_rate, 44_100);
    assert_eq!(hz44.active_vq_strides(), &[8, 4, 2, 1][..]);
}
