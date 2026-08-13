//! MAGNeT runtime shell (SCAFFOLD) tests — structural / config
//! round-trip / FR-EX-08 / loud-partial forward. Mirror of the
//! `dnsmos_p808_p835::tests` pattern (RMVPE / openwakeword precedent).

use super::*;
use vokra_core::VokraError;
use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile, chunks};

/// Adds a zero-filled F32 tensor of the given dims (weight-catalogue
/// stand-in — the shell only reads names + shapes).
fn add_zero(b: &mut GgufBuilder, name: &str, dims: &[u64]) {
    let n: u64 = dims.iter().product();
    b.add_tensor(
        name,
        GgmlType::F32,
        dims.to_vec(),
        vec![0u8; (n * 4) as usize],
    )
    .expect("add tensor");
}

/// Builds a synthetic MAGNeT GGUF with the caller's arch tag and a
/// full `vokra.magnet.*` config chunk group. Small vs medium is
/// selected by [`MagnetVariant`]; the hparams passed are transcribed
/// from `github.com/facebookresearch/audiocraft` reference configs but
/// scaled DOWN in the tests (num_layers=2 / hidden_size=32 /
/// num_heads=4) so a synthetic zero-weight tensor stays cheap. What we
/// pin here is the shape of the metadata schema + the FR-EX-08
/// contract, not the upstream numbers themselves.
fn build_tiny_gguf(variant: MagnetVariant) -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, variant.arch());
    b.add_string(chunks::KEY_MODEL_NAME, variant.short());
    b.add_u32(KEY_MAGNET_NUM_LAYERS, 2);
    b.add_u32(KEY_MAGNET_HIDDEN_SIZE, 32);
    b.add_u32(KEY_MAGNET_NUM_HEADS, 4);
    // seq_len picks a tiny value per-variant so both are structurally
    // distinguishable (small gets 8 frames, medium gets 24 — the same
    // 1:3 ratio the upstream 500 vs 1500 uses).
    b.add_u32(
        KEY_MAGNET_SEQ_LEN,
        match variant {
            MagnetVariant::Small10secs => 8,
            MagnetVariant::Medium30secs => 24,
        },
    );
    b.add_u32(KEY_MAGNET_NUM_CODEBOOKS, 4);
    b.add_u32(KEY_MAGNET_CODEBOOK_SIZE, 2048);
    b.add_u32(KEY_MAGNET_MASK_TOKEN_ID, 2048); // upstream = codebook_size
    b.add_f32(KEY_MAGNET_TOP_P, 0.9);
    b.add_f32(KEY_MAGNET_CFG_COEF, 3.0);
    b.add_u32(KEY_MAGNET_NUM_STEPS, 20);
    // A single nominal weight tensor so the non-empty catalogue check
    // does not fire. The runtime forward does not read it (loud-partial).
    add_zero(
        &mut b,
        "transformer.layers.0.self_attn.q_proj.weight",
        &[32, 32],
    );
    b.to_bytes().expect("serialise tiny MAGNeT GGUF")
}

/// Config round-trip: small variant loads, every field is preserved.
#[test]
fn from_gguf_round_trips_small_variant_config() {
    let bytes = build_tiny_gguf(MagnetVariant::Small10secs);
    let gguf = GgufFile::parse(bytes).unwrap();
    let engine = MagnetEngine::from_gguf(&gguf).expect("small variant must load");
    let cfg = engine.config();
    assert_eq!(cfg.variant, MagnetVariant::Small10secs);
    assert_eq!(cfg.num_layers, 2);
    assert_eq!(cfg.hidden_size, 32);
    assert_eq!(cfg.num_heads, 4);
    assert_eq!(cfg.seq_len, 8);
    assert_eq!(cfg.num_codebooks, 4);
    assert_eq!(cfg.codebook_size, 2048);
    assert_eq!(cfg.mask_token_id, 2048);
    assert!((cfg.top_p - 0.9).abs() < 1e-6);
    assert!((cfg.cfg_coef - 3.0).abs() < 1e-6);
    assert_eq!(cfg.num_steps, 20);
    assert_eq!(engine.weights().len(), 1);
    assert_eq!(
        engine.weights()[0].name,
        "transformer.layers.0.self_attn.q_proj.weight"
    );
}

/// Config round-trip: medium variant loads with the wider seq_len +
/// the correct variant tag so the runtime dispatch cannot silently
/// share the small-variant hparams.
#[test]
fn from_gguf_round_trips_medium_variant_config() {
    let bytes = build_tiny_gguf(MagnetVariant::Medium30secs);
    let gguf = GgufFile::parse(bytes).unwrap();
    let engine = MagnetEngine::from_gguf(&gguf).expect("medium variant must load");
    let cfg = engine.config();
    assert_eq!(cfg.variant, MagnetVariant::Medium30secs);
    assert_eq!(cfg.seq_len, 24, "medium must carry its own 3x seq_len");
}

/// A wrong-arch GGUF (silently sharing with a sibling music-gen family
/// like `musicgen_small`) is a loud [`VokraError::ModelLoad`] naming
/// both the seen and expected tags — the FR-EX-08 wall the audit ticket
/// records for MAGNeT specifically.
#[test]
fn from_gguf_rejects_musicgen_arch_tag() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, "musicgen_small");
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = MagnetEngine::from_gguf(&gguf).expect_err("musicgen arch must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(msg.contains("musicgen_small"), "must name seen arch: {msg}");
    assert!(
        msg.contains(ARCH_SMALL),
        "must name expected small arch: {msg}"
    );
    assert!(
        msg.contains(ARCH_MEDIUM),
        "must name expected medium arch: {msg}"
    );
}

/// Missing `vokra.magnet.num_layers` is a loud [`VokraError::ModelLoad`]
/// (the current BF16 pass-through converter does NOT emit these keys —
/// the message explicitly names the "extend the converter" recipe).
#[test]
fn from_gguf_rejects_missing_config_metadata() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH_SMALL);
    // Deliberately omit every `vokra.magnet.*` key.
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = MagnetEngine::from_gguf(&gguf).expect_err("missing config must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(
        msg.contains(KEY_MAGNET_NUM_LAYERS),
        "error must name a missing key: {msg}"
    );
    assert!(
        msg.contains("converter"),
        "error must direct owner to extend the converter: {msg}"
    );
}

/// A GGUF carrying zero weight tensors is a loud
/// [`VokraError::ModelLoad`] (the future forward would otherwise
/// decode against no weights — silent-partial forbidden per FR-EX-08).
#[test]
fn from_gguf_rejects_zero_weight_tensors() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH_SMALL);
    b.add_u32(KEY_MAGNET_NUM_LAYERS, 2);
    b.add_u32(KEY_MAGNET_HIDDEN_SIZE, 32);
    b.add_u32(KEY_MAGNET_NUM_HEADS, 4);
    b.add_u32(KEY_MAGNET_SEQ_LEN, 8);
    b.add_u32(KEY_MAGNET_NUM_CODEBOOKS, 4);
    b.add_u32(KEY_MAGNET_CODEBOOK_SIZE, 2048);
    b.add_u32(KEY_MAGNET_MASK_TOKEN_ID, 2048);
    b.add_f32(KEY_MAGNET_TOP_P, 0.9);
    b.add_f32(KEY_MAGNET_CFG_COEF, 3.0);
    b.add_u32(KEY_MAGNET_NUM_STEPS, 20);
    // No add_zero call — deliberately empty tensor list.
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = MagnetEngine::from_gguf(&gguf).expect_err("zero weight tensors must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(
        msg.contains("zero weight tensors"),
        "error must name the emptiness offense: {msg}"
    );
    assert!(msg.contains("FR-EX-08"), "error must cite FR-EX-08: {msg}");
}

/// A config carrying `hidden_size` not divisible by `num_heads` is a
/// loud [`VokraError::ModelLoad`] (silent floor would leave a fractional
/// head_dim = wrong-shape matmul downstream).
#[test]
fn from_gguf_rejects_non_divisible_head_shape() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH_SMALL);
    b.add_u32(KEY_MAGNET_NUM_LAYERS, 2);
    b.add_u32(KEY_MAGNET_HIDDEN_SIZE, 33); // NOT divisible by 4
    b.add_u32(KEY_MAGNET_NUM_HEADS, 4);
    b.add_u32(KEY_MAGNET_SEQ_LEN, 8);
    b.add_u32(KEY_MAGNET_NUM_CODEBOOKS, 4);
    b.add_u32(KEY_MAGNET_CODEBOOK_SIZE, 2048);
    b.add_u32(KEY_MAGNET_MASK_TOKEN_ID, 2048);
    b.add_f32(KEY_MAGNET_TOP_P, 0.9);
    b.add_f32(KEY_MAGNET_CFG_COEF, 3.0);
    b.add_u32(KEY_MAGNET_NUM_STEPS, 20);
    add_zero(&mut b, "w", &[4, 4]);
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = MagnetEngine::from_gguf(&gguf).expect_err("bad head shape must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(
        msg.contains("33") && msg.contains("4"),
        "must name shapes: {msg}"
    );
}

/// Loud-partial contract (RMVPE / DNSMOS / openwakeword precedent):
/// [`MagnetEngine::forward`] on a valid config returns
/// [`VokraError::UnsupportedOp`] naming the ADR + the two `vokra-ops`
/// primitives that need to land. No silent fabricated `Vec<u32>`.
#[test]
fn forward_returns_loud_partial_until_ops_and_adr_land() {
    let bytes = build_tiny_gguf(MagnetVariant::Small10secs);
    let gguf = GgufFile::parse(bytes).unwrap();
    let engine = MagnetEngine::from_gguf(&gguf).unwrap();
    // Non-empty conditioning + valid sampling args so the guardrails
    // pass and we reach the loud-partial.
    let text = vec![0.0f32; 32];
    let err = engine
        .forward(&text, 20, 1.0, 0.9, 3.0)
        .expect_err("forward must fire the loud-partial (FR-EX-08)");
    let msg = match err {
        VokraError::UnsupportedOp(m) => m,
        other => panic!("expected UnsupportedOp, got {other:?}"),
    };
    // The message must name the ADR + both ops so an owner knows
    // exactly where to flip the switch.
    assert!(
        msg.contains("docs/adr/M5-magnet-masked-ar-op.md"),
        "loud-partial must name the ADR: {msg}"
    );
    assert!(
        msg.contains("magnet_masked_decode"),
        "loud-partial must name the driver op: {msg}"
    );
    assert!(
        msg.contains("span_masking_scheduler"),
        "loud-partial must name the scheduler op: {msg}"
    );
    assert!(
        msg.contains("FR-OP-85"),
        "loud-partial must name the FR-OP-85 anchor: {msg}"
    );
    assert!(
        msg.contains("Proposed"),
        "loud-partial must clarify the ADR is not yet ratified: {msg}"
    );
    assert!(
        msg.contains("SCAFFOLD"),
        "loud-partial must self-identify as a scaffold: {msg}"
    );
}

/// Argument validation runs BEFORE the loud-partial (`num_steps = 0`
/// case) — a caller feeding bad sampling args gets a targeted
/// `InvalidArgument`, not a confusing `UnsupportedOp`. Pin the gate
/// order (FR-EX-08 order of operations).
#[test]
fn forward_rejects_zero_num_steps_before_loud_partial() {
    let bytes = build_tiny_gguf(MagnetVariant::Small10secs);
    let gguf = GgufFile::parse(bytes).unwrap();
    let engine = MagnetEngine::from_gguf(&gguf).unwrap();
    let text = vec![0.0f32; 32];
    let err = engine
        .forward(&text, 0, 1.0, 0.9, 3.0)
        .expect_err("zero num_steps must be an InvalidArgument, not UnsupportedOp");
    let msg = match err {
        VokraError::InvalidArgument(m) => m,
        other => panic!(
            "expected InvalidArgument (arg guard fires before loud-partial), \
             got {other:?}"
        ),
    };
    assert!(msg.contains("num_steps"), "must name the bad arg: {msg}");
}

/// Empty text conditioning is also caught by the argument guard
/// before the loud-partial — MAGNeT is text-to-music and silently
/// zero-filling the conditioning would misrepresent the run.
#[test]
fn forward_rejects_empty_text_conditioning() {
    let bytes = build_tiny_gguf(MagnetVariant::Small10secs);
    let gguf = GgufFile::parse(bytes).unwrap();
    let engine = MagnetEngine::from_gguf(&gguf).unwrap();
    let err = engine
        .forward(&[], 20, 1.0, 0.9, 3.0)
        .expect_err("empty conditioning must be an InvalidArgument");
    let msg = match err {
        VokraError::InvalidArgument(m) => m,
        other => panic!("expected InvalidArgument, got {other:?}"),
    };
    assert!(
        msg.contains("text_conditioning"),
        "must name the bad arg: {msg}"
    );
}

/// The two arch constants are pinned so a future rename cannot silently
/// drift the runtime dispatch away from the converter's stamp. Mirror
/// of the converter-side `arch_and_name_are_stable_constants_*` test.
#[test]
fn arch_constants_are_stable_and_distinct() {
    assert_eq!(ARCH_SMALL, "magnet_small_10secs");
    assert_eq!(ARCH_MEDIUM, "magnet_medium_30secs");
    assert_ne!(ARCH_SMALL, ARCH_MEDIUM);
    // Distinct from sibling music-gen families (structural — a silent
    // rename to `musicgen` would break the FR-EX-08 dispatch boundary).
    assert_ne!(ARCH_SMALL, "musicgen");
    assert_ne!(ARCH_MEDIUM, "musicgen_medium");
}

/// The `MagnetVariant::arch()` mapping must match the top-level `ARCH_*`
/// constants — otherwise a converter that stamps one and a runtime that
/// checks the other would silently diverge.
#[test]
fn variant_arch_mapping_matches_top_level_consts() {
    assert_eq!(MagnetVariant::Small10secs.arch(), ARCH_SMALL);
    assert_eq!(MagnetVariant::Medium30secs.arch(), ARCH_MEDIUM);
}
