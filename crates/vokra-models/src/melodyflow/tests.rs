//! MelodyFlow T24 30secs runtime shell (SCAFFOLD) tests — structural
//! / config round-trip / FR-EX-08 / loud-partial forward. Mirror of the
//! `magnet::tests` pattern (RMVPE / DNSMOS / openwakeword precedent).

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

/// Builds a synthetic MelodyFlow GGUF with the [`ARCH`] tag and a
/// full `vokra.melodyflow.*` config chunk group. Hparams are
/// transcribed from the T24 30secs release shape but scaled DOWN in
/// the tests (num_layers=2 / hidden_size=32 / num_heads=4) so a
/// synthetic zero-weight tensor stays cheap. What we pin here is the
/// shape of the metadata schema + the FR-EX-08 contract, not the
/// upstream numbers themselves.
fn build_tiny_gguf() -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, "melodyflow_t24_30secs");
    b.add_u32(KEY_MELODYFLOW_NUM_LAYERS, 2);
    b.add_u32(KEY_MELODYFLOW_HIDDEN_SIZE, 32);
    b.add_u32(KEY_MELODYFLOW_NUM_HEADS, 4);
    // num_timesteps deliberately picks a value distinct from
    // num_layers so a code path that confuses the two would fire a
    // round-trip mismatch (T24 in the release name means
    // num_timesteps=24, NOT num_layers=24 — see the module docstring
    // and the ADR §D-3).
    b.add_u32(KEY_MELODYFLOW_NUM_TIMESTEPS, 24);
    b.add_u32(KEY_MELODYFLOW_LATENT_DIM, 128);
    b.add_u32(KEY_MELODYFLOW_NUM_CODEBOOKS, 4);
    b.add_u32(KEY_MELODYFLOW_CODEBOOK_SIZE, 2048);
    // 48 kHz sample rate / hop = 1920 → 25 fps codec frame rate. Small
    // representative value; the shell does not depend on the upstream
    // hop being byte-exact here.
    b.add_u32(KEY_MELODYFLOW_CODEC_FRAME_RATE_HZ, 25);
    b.add_u32(KEY_MELODYFLOW_MAX_DURATION_SECS, 30);
    b.add_u32(KEY_MELODYFLOW_SAMPLE_RATE_HZ, 48_000);
    b.add_u32(KEY_MELODYFLOW_TEXT_PREFIX_LEN, 64);
    b.add_f32(KEY_MELODYFLOW_CFG_SCALE, 4.0);
    // A single nominal weight tensor so the non-empty catalogue check
    // does not fire. The runtime forward does not read it
    // (loud-partial).
    add_zero(&mut b, "transformer.blocks.0.attn.qkv.weight", &[32, 32]);
    b.to_bytes().expect("serialise tiny MelodyFlow GGUF")
}

/// Config round-trip: T24 30secs variant loads, every field is preserved.
#[test]
fn from_gguf_round_trips_t24_30secs_config() {
    let bytes = build_tiny_gguf();
    let gguf = GgufFile::parse(bytes).unwrap();
    let engine = MelodyFlowEngine::from_gguf(&gguf).expect("T24 30secs must load");
    let cfg = engine.config();
    assert_eq!(cfg.num_layers, 2);
    assert_eq!(cfg.hidden_size, 32);
    assert_eq!(cfg.num_heads, 4);
    assert_eq!(cfg.num_timesteps, 24, "T24 = 24 solver steps by default");
    assert_eq!(cfg.latent_dim, 128);
    assert_eq!(cfg.num_codebooks, 4);
    assert_eq!(cfg.codebook_size, 2048);
    assert_eq!(cfg.codec_frame_rate_hz, 25);
    assert_eq!(cfg.max_duration_secs, 30);
    assert_eq!(cfg.sample_rate_hz, 48_000);
    assert_eq!(cfg.text_prefix_len, 64);
    assert!((cfg.cfg_scale - 4.0).abs() < 1e-6);
    // Derived accessor: 25 frames/sec × 30 sec = 750 frames.
    assert_eq!(cfg.max_seq_len(), 750);
    assert_eq!(engine.weights().len(), 1);
    assert_eq!(
        engine.weights()[0].name,
        "transformer.blocks.0.attn.qkv.weight"
    );
}

/// A wrong-arch GGUF (silently sharing with a sibling music-gen family
/// like `magnet_small_10secs` — same T4 tier + music category but
/// entirely different decoder loop) is a loud [`VokraError::ModelLoad`]
/// naming both the seen and expected tags — the FR-EX-08 wall the
/// audit ticket records for MelodyFlow specifically.
#[test]
fn from_gguf_rejects_sibling_music_gen_arch_tag() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, "magnet_small_10secs");
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = MelodyFlowEngine::from_gguf(&gguf).expect_err("sibling arch must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(
        msg.contains("magnet_small_10secs"),
        "must name seen arch: {msg}"
    );
    assert!(msg.contains(ARCH), "must name expected arch: {msg}");
    // Sanity: the message must explicitly mention decoder-loop mismatch
    // so a future reader knows silent-share is a semantic hazard, not
    // just a naming mismatch.
    assert!(
        msg.contains("decoder loop"),
        "message must call out the decoder-loop mismatch hazard: {msg}"
    );
}

/// Missing `vokra.melodyflow.num_layers` is a loud
/// [`VokraError::ModelLoad`] (the current BF16 pass-through converter
/// does NOT emit these keys — the message explicitly names the
/// "extend the converter" recipe).
#[test]
fn from_gguf_rejects_missing_config_metadata() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    // Deliberately omit every `vokra.melodyflow.*` key.
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = MelodyFlowEngine::from_gguf(&gguf).expect_err("missing config must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(
        msg.contains(KEY_MELODYFLOW_NUM_LAYERS),
        "error must name a missing key: {msg}"
    );
    assert!(
        msg.contains("converter"),
        "error must direct owner to extend the converter: {msg}"
    );
}

/// A GGUF carrying zero weight tensors is a loud
/// [`VokraError::ModelLoad`] (the future forward would otherwise
/// integrate against no weights — silent-partial forbidden per
/// FR-EX-08).
#[test]
fn from_gguf_rejects_zero_weight_tensors() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_u32(KEY_MELODYFLOW_NUM_LAYERS, 2);
    b.add_u32(KEY_MELODYFLOW_HIDDEN_SIZE, 32);
    b.add_u32(KEY_MELODYFLOW_NUM_HEADS, 4);
    b.add_u32(KEY_MELODYFLOW_NUM_TIMESTEPS, 24);
    b.add_u32(KEY_MELODYFLOW_LATENT_DIM, 128);
    b.add_u32(KEY_MELODYFLOW_NUM_CODEBOOKS, 4);
    b.add_u32(KEY_MELODYFLOW_CODEBOOK_SIZE, 2048);
    b.add_u32(KEY_MELODYFLOW_CODEC_FRAME_RATE_HZ, 25);
    b.add_u32(KEY_MELODYFLOW_MAX_DURATION_SECS, 30);
    b.add_u32(KEY_MELODYFLOW_SAMPLE_RATE_HZ, 48_000);
    b.add_u32(KEY_MELODYFLOW_TEXT_PREFIX_LEN, 64);
    b.add_f32(KEY_MELODYFLOW_CFG_SCALE, 4.0);
    // No add_zero call — deliberately empty tensor list.
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = MelodyFlowEngine::from_gguf(&gguf).expect_err("zero weight tensors must be loud");
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
/// loud [`VokraError::ModelLoad`] (silent floor would leave a
/// fractional head_dim = wrong-shape matmul downstream).
#[test]
fn from_gguf_rejects_non_divisible_head_shape() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_u32(KEY_MELODYFLOW_NUM_LAYERS, 2);
    b.add_u32(KEY_MELODYFLOW_HIDDEN_SIZE, 33); // NOT divisible by 4
    b.add_u32(KEY_MELODYFLOW_NUM_HEADS, 4);
    b.add_u32(KEY_MELODYFLOW_NUM_TIMESTEPS, 24);
    b.add_u32(KEY_MELODYFLOW_LATENT_DIM, 128);
    b.add_u32(KEY_MELODYFLOW_NUM_CODEBOOKS, 4);
    b.add_u32(KEY_MELODYFLOW_CODEBOOK_SIZE, 2048);
    b.add_u32(KEY_MELODYFLOW_CODEC_FRAME_RATE_HZ, 25);
    b.add_u32(KEY_MELODYFLOW_MAX_DURATION_SECS, 30);
    b.add_u32(KEY_MELODYFLOW_SAMPLE_RATE_HZ, 48_000);
    b.add_u32(KEY_MELODYFLOW_TEXT_PREFIX_LEN, 64);
    b.add_f32(KEY_MELODYFLOW_CFG_SCALE, 4.0);
    add_zero(&mut b, "w", &[4, 4]);
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = MelodyFlowEngine::from_gguf(&gguf).expect_err("bad head shape must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(
        msg.contains("33") && msg.contains("4"),
        "must name shapes: {msg}"
    );
}

/// Loud-partial contract (RMVPE / DNSMOS / openwakeword / MAGNeT
/// precedent): [`MelodyFlowEngine::forward`] on a valid config
/// returns [`VokraError::UnsupportedOp`] naming the ADR + the two
/// `vokra-ops` primitives that need to land + the reused
/// `flow_sample` seam. No silent fabricated `Vec<f32>`.
#[test]
fn forward_returns_loud_partial_until_ops_and_adr_land() {
    let bytes = build_tiny_gguf();
    let gguf = GgufFile::parse(bytes).unwrap();
    let engine = MelodyFlowEngine::from_gguf(&gguf).unwrap();
    // Non-empty conditioning + valid sampling args so the guardrails
    // pass and we reach the loud-partial. Text-only (generation) path.
    let text = vec![0.0f32; 64 * 32];
    let err = engine
        .forward(&text, None, 24, 4.0)
        .expect_err("forward must fire the loud-partial (FR-EX-08)");
    let msg = match err {
        VokraError::UnsupportedOp(m) => m,
        other => panic!("expected UnsupportedOp, got {other:?}"),
    };
    // The message must name the ADR + both ops + the reused
    // `flow_sample` seam + the codec integration path so an owner
    // knows exactly where to flip the switch.
    assert!(
        msg.contains("docs/adr/M5-melodyflow-dit-sampler.md"),
        "loud-partial must name the ADR: {msg}"
    );
    assert!(
        msg.contains("flow_editing_inversion"),
        "loud-partial must name the reverse-ODE driver op: {msg}"
    );
    assert!(
        msg.contains("t24_transformer"),
        "loud-partial must name the DiT block op: {msg}"
    );
    assert!(
        msg.contains("flow_sampler"),
        "loud-partial must name the reused M3-05 seam: {msg}"
    );
    assert!(
        msg.contains("FR-OP-86"),
        "loud-partial must name the FR-OP-86 anchor: {msg}"
    );
    assert!(
        msg.contains("Proposed"),
        "loud-partial must clarify the ADR is not yet ratified: {msg}"
    );
    assert!(
        msg.contains("SCAFFOLD"),
        "loud-partial must self-identify as a scaffold: {msg}"
    );
    // Text-only path advertises "generation" in the use-case tag so an
    // owner reading the error knows they weren't blocked on the
    // editing-specific inversion.
    assert!(
        msg.contains("generation"),
        "text-only forward must tag the generation use-case: {msg}"
    );
}

/// Editing path (with melody conditioning) fires the SAME loud-partial
/// but tags the "editing" use-case so an owner reading the error
/// knows the reverse-ODE inversion is on the critical path.
#[test]
fn forward_editing_path_tags_editing_use_case_in_loud_partial() {
    let bytes = build_tiny_gguf();
    let gguf = GgufFile::parse(bytes).unwrap();
    let engine = MelodyFlowEngine::from_gguf(&gguf).unwrap();
    let text = vec![0.0f32; 64 * 32];
    // Non-empty melody latent so the guard passes and we reach the
    // loud-partial. Content is opaque to the shell (no real weight).
    let melody = vec![0.0f32; 750 * 128]; // seq_len × latent_dim = 750 × 128
    let err = engine
        .forward(&text, Some(&melody), 24, 4.0)
        .expect_err("editing forward must fire the loud-partial");
    let msg = match err {
        VokraError::UnsupportedOp(m) => m,
        other => panic!("expected UnsupportedOp, got {other:?}"),
    };
    assert!(
        msg.contains("editing"),
        "editing forward must tag the editing use-case: {msg}"
    );
    assert!(
        msg.contains("flow_editing_inversion"),
        "editing forward must still name the reverse-ODE driver op: {msg}"
    );
}

/// Argument validation runs BEFORE the loud-partial (`num_solver_steps
/// = 0` case) — a caller feeding bad sampling args gets a targeted
/// `InvalidArgument`, not a confusing `UnsupportedOp`. Pin the gate
/// order (FR-EX-08 order of operations).
#[test]
fn forward_rejects_zero_num_solver_steps_before_loud_partial() {
    let bytes = build_tiny_gguf();
    let gguf = GgufFile::parse(bytes).unwrap();
    let engine = MelodyFlowEngine::from_gguf(&gguf).unwrap();
    let text = vec![0.0f32; 64 * 32];
    let err = engine
        .forward(&text, None, 0, 4.0)
        .expect_err("zero num_solver_steps must be an InvalidArgument, not UnsupportedOp");
    let msg = match err {
        VokraError::InvalidArgument(m) => m,
        other => panic!(
            "expected InvalidArgument (arg guard fires before loud-partial), \
             got {other:?}"
        ),
    };
    assert!(
        msg.contains("num_solver_steps"),
        "must name the bad arg: {msg}"
    );
}

/// Empty text conditioning is also caught by the argument guard
/// before the loud-partial — MelodyFlow is text-conditioned and
/// silently zero-filling the conditioning would misrepresent the run.
#[test]
fn forward_rejects_empty_text_conditioning() {
    let bytes = build_tiny_gguf();
    let gguf = GgufFile::parse(bytes).unwrap();
    let engine = MelodyFlowEngine::from_gguf(&gguf).unwrap();
    let err = engine
        .forward(&[], None, 24, 4.0)
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

/// `Some(&[])` for melody conditioning is ambiguous with `None` — the
/// gate rejects it loudly so a caller cannot accidentally trigger the
/// editing path with a zero-length latent (FR-EX-08).
#[test]
fn forward_rejects_empty_melody_conditioning_over_some() {
    let bytes = build_tiny_gguf();
    let gguf = GgufFile::parse(bytes).unwrap();
    let engine = MelodyFlowEngine::from_gguf(&gguf).unwrap();
    let text = vec![0.0f32; 64 * 32];
    let err = engine
        .forward(&text, Some(&[]), 24, 4.0)
        .expect_err("Some(&[]) melody must be an InvalidArgument");
    let msg = match err {
        VokraError::InvalidArgument(m) => m,
        other => panic!("expected InvalidArgument, got {other:?}"),
    };
    assert!(
        msg.contains("melody_conditioning"),
        "must name the bad arg: {msg}"
    );
    assert!(
        msg.contains("None"),
        "must suggest None for text-only: {msg}"
    );
}

/// Non-finite `cfg_scale` is caught by the argument guard before the
/// loud-partial — silently clamping / defaulting is forbidden per
/// FR-EX-08.
#[test]
fn forward_rejects_non_finite_cfg_scale() {
    let bytes = build_tiny_gguf();
    let gguf = GgufFile::parse(bytes).unwrap();
    let engine = MelodyFlowEngine::from_gguf(&gguf).unwrap();
    let text = vec![0.0f32; 64 * 32];
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err = engine
            .forward(&text, None, 24, bad)
            .expect_err("non-finite cfg_scale must be an InvalidArgument");
        let msg = match err {
            VokraError::InvalidArgument(m) => m,
            other => panic!("expected InvalidArgument for bad={bad}, got {other:?}"),
        };
        assert!(msg.contains("cfg_scale"), "must name the bad arg: {msg}");
    }
}

/// The arch constant is pinned so a future rename cannot silently
/// drift the runtime dispatch away from the converter's stamp. Mirror
/// of the converter-side `arch_and_name_are_stable_constants_*` test.
#[test]
fn arch_constant_is_stable_and_distinct_from_music_gen_family() {
    assert_eq!(ARCH, "melodyflow_t24_30secs");
    // Distinct from sibling music-gen families (structural — a silent
    // rename to `magnet_small_10secs` / `musicgen` / `jasco_*` would
    // break the FR-EX-08 dispatch boundary).
    assert_ne!(ARCH, "magnet_small_10secs");
    assert_ne!(ARCH, "magnet_medium_30secs");
    assert_ne!(ARCH, "musicgen");
    assert_ne!(ARCH, "musicgen_small");
    assert_ne!(ARCH, "musicgen_medium");
    assert_ne!(ARCH, "musicgen_large");
    assert_ne!(ARCH, "audiogen_medium");
    assert_ne!(ARCH, "jasco_400m_chords_drums");
    assert_ne!(ARCH, "audioldm2");
    assert_ne!(ARCH, "stable_audio_open_small");
    assert_ne!(ARCH, "ace_step");
    assert_ne!(ARCH, "bs_roformer");
}

/// The T24 / 30secs marker embedded in the arch constant must survive
/// rename cycles — a future sibling `melodyflow-t12-30secs` or
/// `melodyflow-t48-30secs` cannot silently collide with the T24
/// variant (mirror of the converter-side
/// `upstream_slug_stays_under_facebook_melodyflow_family` test).
#[test]
fn arch_constant_pins_t24_30secs_marker() {
    assert!(
        ARCH.contains("t24"),
        "T24 timestep marker must survive rename cycles: {ARCH}"
    );
    assert!(
        ARCH.contains("30secs"),
        "30secs horizon marker must survive rename cycles: {ARCH}"
    );
}

/// `MelodyFlowConfig::max_seq_len` mirrors the codec_frame_rate_hz *
/// max_duration_secs product. Pin this because downstream RVQ latent
/// allocation depends on it, and a silent u32 overflow (unlikely at
/// realistic scales but the saturating multiply is defensive) would
/// silently truncate the shape.
#[test]
fn max_seq_len_multiplies_frame_rate_by_duration() {
    let bytes = build_tiny_gguf();
    let gguf = GgufFile::parse(bytes).unwrap();
    let engine = MelodyFlowEngine::from_gguf(&gguf).unwrap();
    let cfg = engine.config();
    assert_eq!(
        cfg.max_seq_len(),
        u64::from(cfg.codec_frame_rate_hz) * u64::from(cfg.max_duration_secs),
        "max_seq_len must equal frame_rate * duration",
    );
    // Concrete number for the tiny fixture: 25 fps × 30 s = 750 frames.
    assert_eq!(cfg.max_seq_len(), 750);
}
