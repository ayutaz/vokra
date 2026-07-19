//! Real-checkpoint Voxtral audio-tower + projector parity (M4-residual
//! cc-05 / cc-07 — the 32-layer transformer stack + ×4 frame-stack adapter).
//!
//! # Gate posture
//!
//! Gated on TWO env vars — unset → clean skip with a diagnostic, never a
//! fabricated pass:
//!
//! - `VOKRA_VOXTRAL_GGUF` — a converted real Voxtral GGUF (frame_stack_mlp
//!   adapter chunk, real hparams incl. `text_decoder.head_dim`);
//! - `VOKRA_VOXTRAL_REF_DIR` — a directory of upstream reference dumps
//!   produced by the offline venv script
//!   (`~/.cache/vokra-eval/out/p1-voxtral-asr/dump_voxtral_tower_reference.py`),
//!   which runs the REAL `transformers.models.voxtral` `VoxtralEncoder` +
//!   `VoxtralMultiModalProjector` (fp32, eager attention) on the jfk-30s mel
//!   and dumps little-endian f32 taps:
//!
//!   | file | shape |
//!   |---|---|
//!   | `input_mel.f32.bin` | `[n_mels, 2·n_ctx]` (the exact mel both sides consume) |
//!   | `conv_stem_pos.f32.bin` | `[n_ctx, d]` hidden before block 0 |
//!   | `after_layer_0.f32.bin` | `[n_ctx, d]` |
//!   | `after_layer_15.f32.bin` | `[n_ctx, d]` |
//!   | `encoder_final.f32.bin` | `[n_ctx, d]` (post final LayerNorm) |
//!   | `soft_prefix.f32.bin` | `[n_ctx / frame_stack, d_text]` (post-projector) |
//!
//! # Honest atol derivation (2026-07-19 measured run, M1 iMac)
//!
//! The GGUF stores the upstream BF16 weights **verbatim** (BF16 → f32 widening
//! is exact) and `embed_positions` stays F32, so weight representation is
//! bit-identical to upstream; the reference tower runs in fp32 (bf16 weights
//! widened exactly the same way). Residual deltas therefore measure
//! **accumulation-order differences only** (GEMM loop order, erf-GELU libm
//! vs torch), which grow with depth across the 32 pre-norm blocks. Measured
//! max |Δ| on the real checkpoint + jfk-30s mel (bf16-fs GGUF, 2026-07-19):
//!
//! | tap | measured max \|Δ\| | atol (≈ measured × 2) |
//! |---|---|---|
//! | conv_stem_pos  | 6.914e-6 | 1.5e-5 |
//! | after_layer_0  | 6.795e-6 | 1.5e-5 |
//! | after_layer_15 | 2.480e-5 | 5e-5   |
//! | encoder_final  | 6.454e-4 | 1.5e-3 |
//! | soft_prefix    | 2.617e-5 | 6e-5   |
//! | own_log_mel    | 2.086e-5 | 5e-5   |
//!
//! `encoder_final` is the largest in absolute terms because the final-LN
//! output spans ±19 (large magnitudes → larger absolute deltas); the
//! projector re-normalizes into ±2.8 which is why `soft_prefix` tightens
//! again. Each `ATOL_*` below is the measured max |Δ| rounded up ×~2 (the
//! project's honest-parity convention — bound derived from a measured run,
//! never tuned to force a pass; see `feedback-honest-parity-atol`).

use std::path::{Path, PathBuf};

use vokra_core::BackendKind;
use vokra_models::voxtral::{AudioAdapter, VoxtralConfig, audio_encoder};

/// Conv stem + positional add (before block 0). Dominated by the f32 conv /
/// GELU accumulation-order delta on bit-identical weights.
/// Measured 6.914e-6 (2026-07-19).
const ATOL_CONV_STEM: f32 = 1.5e-5;
/// After transformer block 0. Measured 6.795e-6.
const ATOL_LAYER_0: f32 = 1.5e-5;
/// After transformer block 15 (mid-stack) — accumulation-order drift grows
/// with depth. Measured 2.480e-5.
const ATOL_LAYER_15: f32 = 5e-5;
/// Final hidden (post final LayerNorm, 32 blocks deep; values span ±19 so
/// absolute deltas are largest here). Measured 6.454e-4.
const ATOL_FINAL: f32 = 1.5e-3;
/// Post-projector soft prefix (values re-normalized into ±2.8).
/// Measured 2.617e-5.
const ATOL_SOFT_PREFIX: f32 = 6e-5;
/// Vokra's own log-mel front-end vs the upstream `WhisperFeatureExtractor`
/// mel (informational isolation of front-end vs tower error).
/// Measured 2.086e-5.
const ATOL_MEL: f32 = 5e-5;

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key).map(PathBuf::from)
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(bytes.len() % 4, 0, "{path:?}: not a whole number of f32");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn max_abs_delta(got: &[f32], expected: &[f32], ctx: &str) -> f32 {
    assert_eq!(
        got.len(),
        expected.len(),
        "{ctx}: length mismatch {} != {}",
        got.len(),
        expected.len()
    );
    let mut max = 0.0f32;
    for (g, e) in got.iter().zip(expected) {
        assert!(g.is_finite(), "{ctx}: non-finite runtime value");
        let d = (g - e).abs();
        if d > max {
            max = d;
        }
    }
    max
}

fn check(tap: &str, got: &[f32], expected: &[f32], atol: f32) {
    let d = max_abs_delta(got, expected, tap);
    eprintln!("[voxtral_tower_parity] {tap}: max |Δ| = {d:.3e} (atol {atol:.1e})");
    assert!(
        d <= atol,
        "{tap}: max |Δ| {d:.3e} exceeds atol {atol:.1e} — do NOT loosen this bound to pass; \
         localize the divergence (the per-layer taps bracket it)"
    );
}

/// Real 32-layer tower + ×4 frame-stack projector vs the upstream reference
/// dumps, tap by tap. The mel input comes FROM the dump so both sides
/// consume identical features (front-end parity is isolated in
/// [`own_mel_matches_upstream_feature_extractor`]).
#[test]
fn tower_and_projector_match_upstream_reference_taps() {
    let (Some(gguf), Some(ref_dir)) = (
        env_path("VOKRA_VOXTRAL_GGUF"),
        env_path("VOKRA_VOXTRAL_REF_DIR"),
    ) else {
        eprintln!(
            "[voxtral_tower_parity] SKIP: set VOKRA_VOXTRAL_GGUF (converted real GGUF) and \
             VOKRA_VOXTRAL_REF_DIR (upstream tap dumps) to enable."
        );
        return;
    };

    let file = vokra_mmap::open_gguf(&gguf).expect("mmap-parse the Voxtral GGUF");
    let cfg = VoxtralConfig::from_gguf(&file).expect("read vokra.voxtral.* hparams");
    let (d, n_ctx, n_mels) = (cfg.audio.hidden_dim, cfg.audio.n_ctx, cfg.audio.n_mels);
    eprintln!(
        "[voxtral_tower_parity] audio hparams: n_layer={} d={d} n_head={} n_ctx={n_ctx} \
         n_mels={n_mels} ffn={}",
        cfg.audio.n_layer, cfg.audio.n_head, cfg.audio.ffn_dim
    );

    let encoder =
        vokra_models::voxtral::AudioEncoder::load(&file, &cfg).expect("bind the 32-layer tower");
    assert_eq!(encoder.n_layer(), cfg.audio.n_layer, "full stack bound");
    let adapter = AudioAdapter::from_gguf(&file).expect("bind the projector");
    assert!(
        adapter.is_active(),
        "real GGUF must carry an active adapter"
    );

    // The exact mel the upstream reference consumed.
    let mel = read_f32(&ref_dir.join("input_mel.f32.bin"));
    let n_frames = 2 * n_ctx;
    assert_eq!(mel.len(), n_mels * n_frames, "input_mel shape");

    let compute = vokra_models::compute::Compute::for_backend(
        BackendKind::Cpu,
        vokra_models::voxtral::VOXTRAL_HOT_OPS,
    )
    .expect("cpu compute");
    let (out, taps) =
        audio_encoder::forward_with_layer_taps(&compute, &cfg, &encoder, &mel, n_frames, &[0, 15])
            .expect("tower forward");

    check(
        "conv_stem_pos",
        &taps.pre_blocks,
        &read_f32(&ref_dir.join("conv_stem_pos.f32.bin")),
        ATOL_CONV_STEM,
    );
    check(
        "after_layer_0",
        &taps.after_layer[0].1,
        &read_f32(&ref_dir.join("after_layer_0.f32.bin")),
        ATOL_LAYER_0,
    );
    check(
        "after_layer_15",
        &taps.after_layer[1].1,
        &read_f32(&ref_dir.join("after_layer_15.f32.bin")),
        ATOL_LAYER_15,
    );
    check(
        "encoder_final",
        &out.hidden,
        &read_f32(&ref_dir.join("encoder_final.f32.bin")),
        ATOL_FINAL,
    );

    // ×frame_stack projector → soft prefix.
    let prefix = adapter
        .apply(&compute, &out.hidden, out.n_ctx, out.hidden_dim)
        .expect("projector forward");
    let expected_prefix = read_f32(&ref_dir.join("soft_prefix.f32.bin"));
    check("soft_prefix", &prefix, &expected_prefix, ATOL_SOFT_PREFIX);
    // Shape pin: [n_ctx / 4, d_text] on the shipping mini.
    assert_eq!(
        prefix.len() % cfg.text.hidden_dim,
        0,
        "soft prefix rows must be d_text-wide"
    );
    eprintln!(
        "[voxtral_tower_parity] soft prefix: {} rows × {} (from {} encoder positions)",
        prefix.len() / cfg.text.hidden_dim,
        cfg.text.hidden_dim,
        out.n_ctx
    );
}

/// Vokra's own log-mel front-end on the dumped PCM vs the upstream
/// `WhisperFeatureExtractor` mel — isolates front-end drift from tower
/// drift (the tap test above deliberately feeds both towers the SAME mel).
#[test]
fn own_mel_matches_upstream_feature_extractor() {
    let (Some(gguf), Some(ref_dir)) = (
        env_path("VOKRA_VOXTRAL_GGUF"),
        env_path("VOKRA_VOXTRAL_REF_DIR"),
    ) else {
        eprintln!("[voxtral_tower_parity] SKIP own-mel check: env vars unset.");
        return;
    };
    let pcm_path = ref_dir.join("jfk_pcm.f32.bin");
    if !pcm_path.is_file() {
        eprintln!("[voxtral_tower_parity] SKIP own-mel check: {pcm_path:?} absent.");
        return;
    }
    let file = vokra_mmap::open_gguf(&gguf).expect("mmap-parse the Voxtral GGUF");
    let cfg = VoxtralConfig::from_gguf(&file).expect("hparams");
    let pcm = read_f32(&pcm_path);
    let mel = vokra_models::whisper::mel::log_mel(&pcm, cfg.audio.n_mels);
    let upstream = read_f32(&ref_dir.join("input_mel.f32.bin"));
    check("own_log_mel", &mel, &upstream, ATOL_MEL);
}
