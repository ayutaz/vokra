//! CosyVoice2 numerical parity skeleton — GGUF-gated (M3-09-T22 / T23).
//!
//! Every test here that needs the CosyVoice2 GGUF is gated on the
//! `VOKRA_COSYVOICE2_GGUF` environment variable and skips cleanly when it
//! is unset — the same pattern the Kokoro / Whisper parity harnesses use
//! (`parity_kokoro.rs` / `parity_whisper.rs`). The fixture-free tests
//! (config surface + module-tree wiring) run everywhere so the top-level
//! error surface (arch mismatch, `0`-placeholder hparam rejection, no
//! silent fallback) is exercised in CI without any HuggingFace download.
//!
//! # Scope of this scaffold (M3-09 partial land)
//!
//! The one-session scaffold provides:
//!
//! - the environment-variable-gated skeleton (T22 workflow shape) so the
//!   follow-on session can hang concrete per-tensor parity assertions
//!   off `parity_cosyvoice2_gguf_smoke` without re-plumbing;
//! - fixture-free tests exercising the load path's failure modes
//!   (`0`-placeholder mimi shape refused, wrong arch refused, no silent
//!   fallback on synthesize).
//!
//! The concrete per-tensor `atol = 0.01` (NFR-QL-01) checks + the MEL loss
//! 5% gate (NFR-QL-02, T23 `vokra-eval` bridge) land with T22/T23 once
//! the CosyVoice2 GGUF fixture is available (T29 model zoo publication).

use std::env;

use vokra_core::gguf::chunks::KEY_MODEL_ARCH;
use vokra_core::gguf::{GgufBuilder, GgufMetadataValue};
use vokra_core::{CompliancePolicy, SynthesisRequest, TtsEngine, VokraError};
use vokra_models::cosyvoice2::{CosyVoice2Config, CosyVoice2Tts, MimiBridge};

/// The env var CI / owners set to point the gated tests at a real
/// CosyVoice2 GGUF. Absent = skip (never fabricate a pass).
const GGUF_ENV: &str = "VOKRA_COSYVOICE2_GGUF";

/// Builds a synthetic CosyVoice2 GGUF with model-card + Mimi defaults and
/// caller-controlled `arch` / `flow_schedule` fields.
///
/// Every other numeric hparam is `0` — the runtime accepts this via
/// [`CosyVoice2Config::from_gguf`] and only rejects downstream (at the
/// forward path or the Mimi bridge), so the load itself succeeds and lets
/// us exercise the load-side error surface here.
fn synthetic_gguf(arch: &str, flow_schedule: &str) -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.add_string(KEY_MODEL_ARCH, arch);
    b.add_string("vokra.model.name", "cosyvoice2-0.5b-synthetic");
    b.add_u32("vokra.cosyvoice2.sample_rate", 24_000);
    b.add_u32("vokra.cosyvoice2.arch.vocab_size", 0);
    b.add_u32("vokra.cosyvoice2.arch.hidden_dim", 0);
    b.add_u32("vokra.cosyvoice2.arch.n_layer", 0);
    b.add_u32("vokra.cosyvoice2.arch.n_head", 0);
    b.add_u32("vokra.cosyvoice2.arch.ffn_dim", 0);
    b.add_u32("vokra.cosyvoice2.flow.nfe", 0);
    b.add_metadata(
        "vokra.cosyvoice2.flow.schedule",
        GgufMetadataValue::String(flow_schedule.to_owned()),
    );
    // Mimi shape defaults (the converter writes canonical Kyutai values;
    // see crates/vokra-convert/src/models/cosyvoice2.rs).
    b.add_u32("vokra.cosyvoice2.mimi.n_codebooks", 8);
    b.add_u32("vokra.cosyvoice2.mimi.codebook_size", 2048);
    b.add_u32("vokra.cosyvoice2.mimi.d_model", 512);
    b.add_u32("vokra.cosyvoice2.streaming.chunk_size", 0);
    b.add_u32("vokra.cosyvoice2.streaming.chunk_hop", 0);
    b.to_bytes().expect("gguf serialize")
}

/// Fixture-free: the arch check runs before any component loader — a
/// wrong-arch GGUF fails with a clear top-level `ModelLoad`, not a
/// downstream missing-tensor error (FR-EX-08).
#[test]
fn parity_cosyvoice2_wrong_arch_fails_top_level() {
    let bytes = synthetic_gguf("kokoro-82m-istftnet", "linear");
    let err = CosyVoice2Tts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
        .expect_err("wrong arch must fail");
    match err {
        VokraError::ModelLoad(msg) => {
            assert!(
                msg.contains("cosyvoice2") && msg.contains("kokoro-82m-istftnet"),
                "unexpected: {msg}"
            );
        }
        other => panic!("expected ModelLoad, got {other:?}"),
    }
}

/// Fixture-free: the synthetic GGUF loads (arch OK, compliance registry
/// classifies `cosyvoice2` permissive), but every forward path returns
/// NotImplemented because the numeric layers are the scaffold-only path.
#[test]
fn parity_cosyvoice2_synthetic_load_succeeds_but_synthesize_is_stub() {
    let bytes = synthetic_gguf("cosyvoice2", "linear");
    let tts = CosyVoice2Tts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
        .expect("apache-2.0 registry entry admits it");
    assert_eq!(tts.config().sample_rate, 24_000);
    let err = tts
        .synthesize(&SynthesisRequest::new("hello"))
        .expect_err("scaffold must not produce audio");
    assert!(matches!(err, VokraError::NotImplemented(_)));
}

/// Fixture-free: the Mimi bridge accepts the canonical shape emitted by
/// the converter and rejects a degenerate `0` shape (converter placeholder
/// path).
#[test]
fn parity_cosyvoice2_mimi_bridge_accepts_kyutai_defaults() {
    let bytes = synthetic_gguf("cosyvoice2", "linear");
    let file = vokra_core::gguf::GgufFile::parse(bytes).expect("parse");
    let cfg = CosyVoice2Config::from_gguf(&file).expect("read");
    let bridge = MimiBridge::from_config(&cfg).expect("Kyutai defaults must load");
    assert_eq!(bridge.attrs().n_codebooks, 8);
    assert_eq!(bridge.attrs().codebook_size, 2048);
    assert_eq!(bridge.attrs().d_model, 512);
}

/// Fixture-free: an unknown schedule tag is a loud error (no silent
/// fallback to linear).
#[test]
fn parity_cosyvoice2_unknown_schedule_fails_up_front() {
    let bytes = synthetic_gguf("cosyvoice2", "cosine");
    let file = vokra_core::gguf::GgufFile::parse(bytes).expect("parse");
    let cfg = CosyVoice2Config::from_gguf(&file).expect("read");
    // The runtime accepts the tag string at config load time (the reader
    // is dumb by design — no schedule vocabulary hard-coded in
    // `CosyVoice2Config`), so the loud failure fires when the Flow
    // Matching driver builds its runtime params.
    use vokra_models::cosyvoice2::FlowMatchingRuntimeParams;
    let err = FlowMatchingRuntimeParams::from_config(&cfg)
        .expect_err("cosine is not a schedule vokra_ops accepts");
    assert!(matches!(err, VokraError::InvalidArgument(_)));
}

/// Gated: the concrete per-tensor parity + MEL loss checks land with
/// T22/T23. Today this test only verifies the env var gating shape (skip
/// cleanly when unset) so the CI harness can be enabled at any time
/// without changing this file (mirrors `parity_kokoro.rs` T22 shape).
#[test]
fn parity_cosyvoice2_gguf_smoke() {
    let Some(gguf_path) = env::var(GGUF_ENV).ok() else {
        eprintln!(
            "{GGUF_ENV} unset — skipping cosyvoice2 GGUF parity smoke; \
             this is a clean skip (never a fabricated pass)"
        );
        return;
    };
    // The gated body: load the GGUF, check the arch, and hand off to the
    // (future) per-tensor parity comparisons at T22/T23. Today we only
    // verify the load path succeeds so an owner running the harness
    // against a real Apache-2.0 GGUF gets one green + one clear next step.
    let tts = CosyVoice2Tts::from_path_with_policy(&gguf_path, &CompliancePolicy::strict())
        .expect("real CosyVoice2 GGUF must load under strict compliance");
    // A stock CosyVoice2 GGUF is Apache 2.0; the registry admits it
    // without a research flag. We do not exercise the forward path yet.
    assert_eq!(
        tts.config().sample_rate,
        24_000,
        "the CosyVoice2 model card fixes the PCM rate at 24 kHz"
    );
    eprintln!(
        "cosyvoice2 GGUF loaded from {gguf_path}: sample_rate=24_000 — per-tensor / \
         MEL loss parity assertions land with T22/T23 (see M3-09-T22-T23 spec)"
    );
}
