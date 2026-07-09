//! CosyVoice2 native TTS / S2S — module scaffold (M3-09).
//!
//! Native re-implementation of the CosyVoice2 inference core (text tokenizer,
//! LLM backbone, Flow Matching CFM, Mimi codec, chunk-aware streaming) in the
//! whisper.cpp style: the model *definition* lives in Rust; only the upstream
//! **safetensors** checkpoint (Apache 2.0 code + weight, official
//! `iic/CosyVoice2-0.5B` on HuggingFace, converted offline to GGUF by
//! `vokra-convert`, T03) is consumed at runtime. No ONNX at runtime
//! (FR-LD-05, permanent constraint; CLAUDE.md design judgement 4).
//!
//! # Scope of this scaffold (M3-09 session partial land)
//!
//! One CC session cannot cover all 28 M3-09 CC tickets (14 h of code work).
//! This scaffold delivers the **module tree + config surface + engine wiring
//! contracts** so the follow-up sessions can land the numeric forward paths
//! against the same public surface without re-plumbing the top-level types:
//!
//! - [`CosyVoice2Config`] — `vokra.cosyvoice2.*` metadata surface (T04);
//! - [`CosyVoice2Tts`] — the [`TtsEngine`] handle carrying the loaded arch,
//!   the compliance / research-flag gate wiring (T01/T25) and the
//!   watermark / disclosure config (T17/T18);
//! - the `text_encoder`, `flow_matching`, `mimi_bridge` submodules — stubs
//!   returning [`VokraError::NotImplemented`] with a clear next-step message
//!   until T07 / T10 / T13 land.
//!
//! Every stub surfaces an explicit [`VokraError`] (never a silent fallback,
//! FR-EX-08) so a caller who wires a session against this scaffold today
//! does *not* silently receive a degraded output — they get a loud error
//! naming the ticket that would satisfy it.
//!
//! # Dependencies (all landed as Wave 2/3 in the M3 batch)
//!
//! - `vokra_ops::flow_sample` (M3-05, FR-EX-10 — runtime function, NOT a
//!   graph op; sampler axes are configurable per invocation);
//! - `vokra_ops::mimi_rvq_decode` / [`MimiDecoder`] (M3-06, RVQ decode +
//!   Mimi CC-BY 4.0 attribution recorded in NOTICE / license-audit.md);
//! - `vokra_ops::length_conditioning` (M3-08, mode A `UserSpecified` /
//!   mode B `RefLinear`);
//! - `vokra_ops::ProsodyControl` + [`ApplyProsody`] (M3-17, unified prosody
//!   control message; the CosyVoice2 adapter folds pitch/speed/pause into
//!   the model's native instruction-string surface — T17-follow-on).
//!
//! # Streaming (NFR-PF-07)
//!
//! chunk-aware streaming with `[time, stream, codebook]` paged KV cache
//! (M3-03) lands with T14 / T15 / T16. The type surface for that path is
//! *reserved* here (module `flow_matching` docs `chunk_size` / `chunk_hop`)
//! but the concrete state machine is deferred — see T14's docstring for the
//! next-session contract.
//!
//! # Compliance
//!
//! CosyVoice2 is **Apache 2.0 code + weight** (CLAUDE.md モデル表, verified in
//! docs/license-audit.md — appended by the T26 owner ticket). The runtime
//! reads the license class through
//! [`vokra_core::check_weight_license`] on load; the M2-13 compliance gate
//! rejects a CC-BY-NC provenance (F5-TTS / Fish-Speech) even if it were
//! mislabelled as CosyVoice2 (T25 regression test lives beside T22 parity
//! CI once the GGUF fixture is available).
//!
//! # AudioSeal watermark / C2PA manifest (T17 / T18)
//!
//! [`WatermarkConfig::default()`] preserves the FR-CP-01/02 design intent
//! (AudioSeal + C2PA + SilentCipher = ON), and
//! [`WatermarkConfig::backend_status()`] returns
//! [`WatermarkBackendStatus::Deferred`] — no embedding backend was
//! implemented (M1-07 client drop 2026-07-04). CosyVoice2 must NOT lie
//! about watermarking: **the deployer-side disclosure MUST**
//! (`docs/legal-compliance.md` §1.4) still applies regardless of config
//! flags, and the loader surfaces the deferred-backend notice through the
//! same session-level hook the piper-plus / Kokoro loaders use (wired at
//! T17 follow-on when the CosyVoice2 GGUF is available).

pub(crate) mod config;
pub(crate) mod flow_matching;
pub(crate) mod mimi_bridge;
pub(crate) mod text_encoder;

use std::path::Path;

use vokra_core::gguf::GgufFile;
use vokra_core::{
    BackendKind, CompliancePolicy, Result, SynthesisRequest, SynthesizedAudio, TtsEngine,
    VokraError, WatermarkConfig, check_weight_license,
};
use vokra_ops::{ApplyProsody, ProsodyControl};

pub use config::CosyVoice2Config;
pub use flow_matching::{ChunkAwareCfm, FlowMatchingRuntimeParams};
pub use mimi_bridge::MimiBridge;
pub use text_encoder::TextEncoderStub;

/// `vokra.model.arch` a CosyVoice2 GGUF must carry.
///
/// Written by `vokra-convert::models::cosyvoice2::ARCH` (T03); kept in sync
/// with the runtime constant here. The registry
/// (`vokra_core::compliance::license_class::registry_lookup`) already knows
/// this id as `LicenseClass::Permissive` (Apache 2.0 code + weight —
/// docs/license-audit.md), so a stock CosyVoice2 GGUF passes the
/// [`check_weight_license`] gate without a research flag.
const EXPECTED_ARCH: &str = "cosyvoice2";

/// The backend hot ops the CosyVoice2 native model dispatches through the
/// [`crate::compute::Compute`] seam.
///
/// Populated by follow-on tickets (T19 CUDA seam / T20 Metal seam). Today the
/// list is deliberately **empty** so a caller pointing at a Metal or CUDA
/// backend does not falsely believe the forward is GPU-accelerated: with an
/// empty hot-op set, `Compute::for_backend` currently trivially accepts every
/// backend, but every forward-path stub returns
/// [`VokraError::NotImplemented`] before the seam is consulted (FR-EX-08 —
/// no silent fallback). The list will grow when T19/T20 wire the LLM GEMM
/// path.
#[allow(dead_code)] // consumed by T19/T20 follow-on
pub(crate) const COSYVOICE2_HOT_OPS: &[crate::compute::HotOp] = &[];

/// A loaded CosyVoice2 model — engine handle.
///
/// The struct is intentionally light: it carries the resolved config, the
/// selected backend, and the watermark / prosody control state. The heavy
/// numeric state (text encoder / LLM backbone / Flow Matching / Mimi decoder)
/// lands in follow-on tickets and hangs off private fields added at that
/// time. A stub constructor
/// ([`CosyVoice2Tts::from_gguf_with_policy`]) still validates the arch, runs
/// the compliance gate, and reads the config so the engine's error surface
/// (bad arch → [`VokraError::ModelLoad`], mismatched hparams →
/// [`VokraError::InvalidArgument`]) is exercised today.
#[derive(Debug)]
pub struct CosyVoice2Tts {
    /// The resolved GGUF metadata (arch / vocab / streaming / flow / mimi
    /// hyperparameters — T04 chunk design).
    config: CosyVoice2Config,
    /// Selected compute backend (default [`BackendKind::Cpu`], overridable
    /// via [`CosyVoice2Tts::with_backend`]; the numeric path lands with
    /// T19/T20).
    backend_kind: BackendKind,
    /// Watermark / disclosure knobs. Defaults to design intent — AudioSeal +
    /// C2PA + SilentCipher = ON. Embedding backend is deferred (T17 doc),
    /// deployer-side disclosure MUST still applies
    /// (docs/legal-compliance.md §1.4).
    watermark: WatermarkConfig,
}

impl CosyVoice2Tts {
    /// Loads a CosyVoice2 GGUF from disk with the fail-closed
    /// [`CompliancePolicy::strict`] gate.
    ///
    /// # Errors
    ///
    /// Propagates GGUF parse errors, arch mismatch, and any
    /// compliance-gate refusal (a CC-BY-NC provenance without a research
    /// flag is rejected — [`VokraError::ResearchLicenseRequired`]).
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_path_with_policy(path, &CompliancePolicy::strict())
    }

    /// Loads a CosyVoice2 GGUF from disk under an explicit `policy`.
    pub fn from_path_with_policy(
        path: impl AsRef<Path>,
        policy: &CompliancePolicy,
    ) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).map_err(VokraError::Io)?;
        Self::from_gguf_with_policy(&bytes, policy)
    }

    /// Loads a CosyVoice2 GGUF from raw bytes under an explicit `policy`.
    ///
    /// The `vokra.model.arch` is checked first, so a non-CosyVoice2 (or
    /// wrong architecture) GGUF fails with a clear
    /// [`VokraError::ModelLoad`] rather than a confusing missing-tensor
    /// error deep in a component loader (the pattern piper-plus and
    /// Kokoro established). Then the shared weight-license gate
    /// ([`check_weight_license`], FR-CP-03) runs on the container before
    /// any weight tensor is bound — a non-commercial or unknown weight
    /// license without a research flag is refused, never silently loaded.
    /// CosyVoice2 is Apache 2.0 code + weight, so a stock (unlabelled)
    /// CosyVoice2 GGUF classifies permissive (built-in registry, arch
    /// `cosyvoice2`) and passes.
    pub fn from_gguf_with_policy(bytes: &[u8], policy: &CompliancePolicy) -> Result<Self> {
        let file = GgufFile::parse(bytes.to_vec())
            .map_err(|e| VokraError::ModelLoad(format!("cosyvoice2 GGUF: {e}")))?;
        let arch = file
            .get(vokra_core::gguf::chunks::KEY_MODEL_ARCH)
            .and_then(|v| v.as_str());
        if arch != Some(EXPECTED_ARCH) {
            return Err(VokraError::ModelLoad(format!(
                "not a CosyVoice2 GGUF: vokra.model.arch = {arch:?}, expected \
                 `{EXPECTED_ARCH}`"
            )));
        }
        check_weight_license(&file, policy)?;
        let config = CosyVoice2Config::from_gguf(&file)?;
        Ok(Self {
            config,
            backend_kind: BackendKind::Cpu,
            watermark: WatermarkConfig::default(),
        })
    }

    /// Selects the backend the synthesis hot path runs on (default
    /// [`BackendKind::Cpu`]; wired at T19/T20).
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend_kind = backend;
        self
    }

    /// Overrides the watermark configuration (opt-out surface for
    /// FR-CP-01 AudioSeal — see [`WatermarkConfig::audioseal_opted_out`]).
    ///
    /// Note: the embedding backend is deferred (M1-07 client drop
    /// 2026-07-04); toggling flags here **does not** cause audio to be
    /// watermarked (see [`WatermarkConfig::backend_status`]). The
    /// deployer-side disclosure MUST still applies (EU AI Act Article 50,
    /// docs/legal-compliance.md §1.4).
    #[must_use]
    pub fn with_watermark(mut self, watermark: WatermarkConfig) -> Self {
        self.watermark = watermark;
        self
    }

    /// The resolved CosyVoice2 configuration (arch + streaming + flow /
    /// mimi hyperparameters).
    #[must_use]
    pub fn config(&self) -> &CosyVoice2Config {
        &self.config
    }

    /// The current backend selection.
    #[must_use]
    pub fn backend_kind(&self) -> BackendKind {
        self.backend_kind
    }

    /// The current watermark configuration.
    #[must_use]
    pub fn watermark(&self) -> &WatermarkConfig {
        &self.watermark
    }
}

impl TtsEngine for CosyVoice2Tts {
    /// Text → PCM adapter (T14/T15 chunk-aware streaming pipeline lands the
    /// concrete numeric path).
    ///
    /// Until the LLM backbone (T07/T08), Flow Matching CFM (T10/T11), and
    /// Mimi bridge (T13) are wired end-to-end, this returns
    /// [`VokraError::NotImplemented`] with a clear next-step message —
    /// never a silent zero-fill fallback (FR-EX-08).
    fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesizedAudio> {
        // Reference the request text so the intent is documented in-source;
        // the future path consumes this through
        // [`text_encoder::TextEncoderStub::encode`] once a real GGUF binds
        // the tokenizer.
        let _ = request.text.as_str();
        Err(VokraError::NotImplemented(
            "CosyVoice2 TtsEngine::synthesize needs the T07/T08 LLM backbone, T10/T11 \
             Flow Matching CFM, T13 Mimi decoder and T14/T15 chunk-aware streaming pipeline; \
             this session lands the scaffold only",
        ))
    }
}

/// [`ApplyProsody`] adapter for CosyVoice2 (M3-17 unified prosody control /
/// T17 follow-on).
///
/// # Contract
///
/// - **Identity is passthrough.** An identity [`ProsodyControl`] leaves
///   `ctx` untouched (M3-17 contract).
/// - **Instruction folding.** `pitch_shift` / `speed_scale` / `pause_ms`
///   are folded into `ctx.instruction` as a compact natural-language
///   instruction string when either the caller's `ctx.instruction` is
///   `None` or empty — the actual textual template is fixed by the
///   T17-follow-on session against the upstream CosyVoice2 instruction
///   prompt (ハルシネーション厳禁: this scaffold does not invent the
///   template). Today the adapter is a **passthrough** by contract; it
///   validates the axes and preserves the caller's `ctx.instruction`
///   without folding, so no invented instruction text leaks into the
///   output.
///
/// # Rationale for the passthrough
///
/// M3-17 landed the API surface (trait + struct) but not the model
/// adapter — that is deliberately deferred to M3-09 (this WP). Because
/// the CosyVoice2 numeric forward is itself a scaffold in this session,
/// wiring the instruction template today would require inventing text
/// that the model would never actually consume — a hallucination the
/// project bans (CLAUDE.md「ハルシネーション厳禁」). The trait is
/// implemented so the type surface is stable; the folding is a strictly
/// additive change in the follow-on session.
impl ApplyProsody for CosyVoice2Tts {
    fn apply(&self, ctx: &mut ProsodyControl) {
        // Passthrough — per M3-17 trait contract when identity, and
        // T17-follow-on lands the non-identity instruction template
        // folding. Today we preserve the caller's `ctx` verbatim so no
        // invented text (CLAUDE.md hallucination ban) enters the pipeline.
        // Callers must run `ctx.validate()` before `apply` — M3-17 trait
        // rustdoc — because `apply` has no `Result` return channel.
        let _ = ctx;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::chunks::KEY_MODEL_ARCH;
    use vokra_core::gguf::{GgufBuilder, GgufMetadataValue};

    fn minimal_gguf_bytes(arch: &str) -> Vec<u8> {
        let mut b = GgufBuilder::new();
        b.add_string(KEY_MODEL_ARCH, arch);
        b.add_string("vokra.model.name", "cosyvoice2-0.5b");
        // config keys — see config::from_gguf; degenerate but structurally
        // valid so the scaffold exercises the load path.
        b.add_u32(config::KEY_SAMPLE_RATE, 24_000);
        b.add_u32(config::KEY_VOCAB_SIZE, 0);
        b.add_u32(config::KEY_HIDDEN_DIM, 0);
        b.add_u32(config::KEY_N_LAYER, 0);
        b.add_u32(config::KEY_N_HEAD, 0);
        b.add_u32(config::KEY_FFN_DIM, 0);
        b.add_u32(config::KEY_FLOW_NFE, 0);
        b.add_u32(config::KEY_MIMI_N_CODEBOOKS, 0);
        b.add_u32(config::KEY_MIMI_CODEBOOK_SIZE, 0);
        b.add_u32(config::KEY_MIMI_D_MODEL, 0);
        b.add_u32(config::KEY_STREAMING_CHUNK_SIZE, 0);
        b.add_u32(config::KEY_STREAMING_CHUNK_HOP, 0);
        b.add_metadata(
            config::KEY_FLOW_SCHEDULE,
            GgufMetadataValue::String("linear".to_owned()),
        );
        b.to_bytes().expect("gguf serialize")
    }

    #[test]
    fn arch_mismatch_fails_loudly() {
        // A wrong-arch GGUF must fail at the arch check — not deep inside a
        // component loader (FR-EX-08).
        let bytes = minimal_gguf_bytes("kokoro-82m-istftnet");
        let err = CosyVoice2Tts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect_err("wrong arch must fail");
        match err {
            VokraError::ModelLoad(msg) => assert!(
                msg.contains(EXPECTED_ARCH) && msg.contains("kokoro-82m-istftnet"),
                "unexpected message: {msg}"
            ),
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn correct_arch_loads_scaffold_config() {
        // The registry classifies `cosyvoice2` permissive (Apache 2.0), so
        // the strict policy admits it.
        let bytes = minimal_gguf_bytes(EXPECTED_ARCH);
        let tts = CosyVoice2Tts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect("apache-2.0 registry entry admits it");
        assert_eq!(tts.config().sample_rate, 24_000);
        assert_eq!(tts.backend_kind(), BackendKind::Cpu);
        assert!(tts.watermark().any_enabled());
    }

    #[test]
    fn synthesize_is_not_implemented_never_silent() {
        // No silent zero-fill fallback (FR-EX-08).
        let bytes = minimal_gguf_bytes(EXPECTED_ARCH);
        let tts = CosyVoice2Tts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect("load");
        let err = tts
            .synthesize(&SynthesisRequest::new("hello world"))
            .expect_err("scaffold must not produce audio");
        assert!(matches!(err, VokraError::NotImplemented(_)));
    }

    #[test]
    fn apply_prosody_identity_is_passthrough() {
        // M3-17 trait contract: identity control leaves ctx untouched.
        let bytes = minimal_gguf_bytes(EXPECTED_ARCH);
        let tts = CosyVoice2Tts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect("load");
        let mut ctx = ProsodyControl::identity();
        tts.apply(&mut ctx);
        assert!(ctx.is_identity(), "identity in → identity out");
    }

    #[test]
    fn apply_prosody_non_identity_is_currently_preserved() {
        // T17-follow-on will fold pitch/speed/pause into ctx.instruction;
        // today the scaffold preserves the caller's ctx verbatim (see the
        // impl rustdoc for the honest-negative rationale).
        let bytes = minimal_gguf_bytes(EXPECTED_ARCH);
        let tts = CosyVoice2Tts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect("load");
        let mut ctx = ProsodyControl::default()
            .with_speed_scale(1.25)
            .with_pause_ms(200);
        let before = ctx.clone();
        tts.apply(&mut ctx);
        assert_eq!(
            ctx, before,
            "T17-follow-on lands the folding; today passthrough"
        );
    }
}
