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

pub(crate) mod chunk_pipeline;
pub(crate) mod config;
pub(crate) mod flow_matching;
// Public so integration tests can reach the parity harness
// (`vokra_models::cosyvoice2::llm::parity`). The internal-oracle path
// through the `pub use` list below remains the primary surface; the
// module handle is exposed only for `parity::forward_matches_step_by_step`
// / `parity::assert_vs_hf_reference` — moving those to a top-level
// re-export would drift as the parity API grows.
pub mod llm;
pub(crate) mod mimi_bridge;
pub(crate) mod text_encoder;

use std::path::Path;

use vokra_core::gguf::GgufFile;
use vokra_core::{
    BackendKind, CompliancePolicy, Result, SynthesisRequest, SynthesizedAudio, TtsEngine,
    VokraError, WatermarkConfig, check_weight_license,
};
use vokra_ops::{ApplyProsody, ProsodyControl};

pub use chunk_pipeline::{ChunkAwareStreamingPipeline, PipelineChunk, PipelineOutput};
pub use config::CosyVoice2Config;
pub use flow_matching::{ChunkAwareCfm, ChunkContinuation, FlowMatchingRuntimeParams};
pub use llm::{
    DEFAULT_RMS_NORM_EPS, DEFAULT_ROPE_BASE_QWEN2, LlmBackbone, LlmBackboneConfig, LlmBackboneStep,
};
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
    /// LLM backbone (M3-09-T07/T08 body). Decoder-only Mistral-style
    /// transformer whose output token stream drives the Flow Matching CFM.
    ///
    /// `None` when the GGUF carries the 0-placeholder shape config the
    /// scaffold converter emits — the LLM backbone refuses to bind a
    /// synthesized fixture on zero dims (FR-EX-08 — the shape-only
    /// converter path is not a silent-fallback path). A caller who wires
    /// a real synthesized fixture receives a `Some(LlmBackbone)` and can
    /// exercise the full Mistral forward via
    /// [`CosyVoice2Tts::llm`] → [`llm::LlmBackbone::forward`].
    ///
    /// The LLM config is read from the same GGUF as the top-level config
    /// (`vokra.cosyvoice2.arch.*` LLM-side keys), so the two are always
    /// consistent — a mismatch is impossible by construction.
    llm: Option<llm::LlmBackbone>,
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
        // Bind the LLM backbone off the same GGUF. `from_gguf` binds **real
        // weights** when the GGUF carries the backbone tensors, else a
        // synthesized fixture against the metadata shape.
        //
        // Exactly one binding failure is tolerated: a GGUF whose LLM dims are
        // all the converter's 0 sentinel (a pre-hparam-fix conversion). Such a
        // container must stay loadable so it can be inspected and re-converted,
        // so the LLM handle is surfaced as `None` and
        // [`CosyVoice2Tts::synthesize`] names that as the reason.
        //
        // The condition is read off the *config*, not off the error variant.
        // Keying it on `InvalidArgument` (as this did until the 2026-07-19
        // audit, cc-28) also swallowed genuinely malformed GGUFs — wrong-typed
        // metadata keys and non-GQA-well-formed dims raise the same variant —
        // so a broken container reported a successful load and only failed
        // later, misattributed. Everything except the sentinel now propagates
        // (FR-EX-08); real tensor-binding problems were and remain `ModelLoad`.
        let llm_cfg = llm::LlmBackboneConfig::from_gguf(&file, &config)?;
        let llm = if llm_cfg.is_placeholder_shape() {
            None
        } else {
            Some(llm::LlmBackbone::from_gguf(&file, &config)?)
        };
        Ok(Self {
            config,
            llm,
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

    /// Access to the LLM backbone (M3-09-T07/T08 body).
    ///
    /// `None` when the GGUF carries 0-placeholder dims (the pre-hparam-fix
    /// converter path — re-convert with `--config` to populate them).
    /// Real dims → `Some(LlmBackbone)`: **real weights** when the GGUF
    /// carries the backbone tensors (`LlmWeights::from_gguf`), else the
    /// seed-deterministic synthesized fixture (metadata-only test GGUFs).
    #[must_use]
    pub fn llm(&self) -> Option<&llm::LlmBackbone> {
        self.llm.as_ref()
    }

    /// Runs the chunk-aware streaming pipeline with caller-supplied
    /// velocity and code closures (M3-09-T12/T13/T14 injection point).
    ///
    /// This is the **internal-oracle testable path** for the CosyVoice2
    /// engine — the real LLM velocity closure (T07/T08) and Mimi
    /// codebook binding (T13 real-checkpoint) will replace the caller's
    /// injections once the upstream inspection (T02) fills in the
    /// tensor names. Until then, tests use an identity Mimi decoder and
    /// deterministic velocity/code closures to exercise the plumbing
    /// without inventing upstream tensor names (CLAUDE.md「ハルシネー
    /// ション厳禁」).
    ///
    /// # Arguments
    ///
    /// - `length_input` — M3-08 length_conditioning input (mode A / B).
    /// - `initial_state` — Flow Matching starting state for the first
    ///   chunk. Shape is preserved across all chunks (FR-EX-08).
    /// - `velocity_fn` — the caller-supplied velocity closure.
    /// - `code_fn` — the caller-supplied "state → codes" mapper.
    ///
    /// The Mimi bridge is constructed **with the M3-06 identity
    /// decoder fixture** — the T13 follow-on replaces this with a real
    /// codebook binding when the CosyVoice2 GGUF is fully populated.
    ///
    /// # Errors
    ///
    /// Propagates every downstream error verbatim.
    pub fn synthesize_with_pipeline<V, C>(
        &self,
        length_input: vokra_core::ir::graph::LengthConditioningAttrs,
        initial_state: &vokra_ops::FlowSamplerState,
        velocity_fn: V,
        code_fn: C,
    ) -> Result<chunk_pipeline::PipelineOutput>
    where
        V: FnMut(
            &vokra_ops::FlowSamplerState,
            f32,
            vokra_ops::ForwardPass,
            &flow_matching::ChunkContinuation<'_>,
        ) -> Result<vokra_ops::FlowSamplerState>,
        C: FnMut(&vokra_ops::FlowSamplerState, usize, usize) -> Result<Vec<u32>>,
    {
        let cfm = flow_matching::ChunkAwareCfm::new(self.config.clone())?;
        let bridge = mimi_bridge::MimiBridge::with_identity_decoder(&self.config)?;
        let pipeline =
            chunk_pipeline::ChunkAwareStreamingPipeline::new(&self.config, &cfm, &bridge)?;
        pipeline.synthesize(length_input, initial_state, velocity_fn, code_fn)
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
    ///
    /// # Chain wiring (M3-09 partial land)
    ///
    /// The module tree is chained today — a follow-on session composes
    /// text → [`TextEncoderStub::encode`] → [`llm::LlmBackbone::forward`]
    /// → [`ChunkAwareCfm::run_chunks`] → [`MimiBridge::decode_chunk`] by
    /// filling in each stage's numeric path. The top-level `synthesize`
    /// short-circuits with NotImplemented because the tokenizer (T06),
    /// LLM weight binding (T07), and forward pass (T08) are all deferred.
    /// The `synthesize_with_pipeline` entry point below exposes the
    /// injected-closure oracle path for internal-oracle tests today.
    fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesizedAudio> {
        // Reference the LLM backbone handle so the engine's chain owner
        // is visible in-source (documented dependency, not consumed
        // today).
        let _ = self.llm.as_ref().map(|l| l.config());
        let _ = request.text.as_str();
        if self.llm.is_none() {
            // Name the actual blocker instead of letting this GGUF fall through
            // to the generic scaffold message: the container loaded, but it
            // carries no usable LLM hparams, and re-converting is the fix.
            return Err(VokraError::NotImplemented(
                "CosyVoice2 TtsEngine::synthesize: this GGUF carries 0-placeholder \
                 LLM dims (a pre-hparam-fix conversion), so no backbone is bound. \
                 Re-convert with `vokra-cli convert --model cosyvoice2 --config \
                 <upstream config.json>` — note that CosyVoice2-0.5B's top-level \
                 config.json is a stub; the real one is CosyVoice-BlankEN/config.json",
            ));
        }
        Err(VokraError::NotImplemented(
            "CosyVoice2 TtsEngine::synthesize needs T06 tokenizer, T07/T08 LLM \
             backbone, T10/T11 Flow Matching CFM, T13 Mimi decoder and T14/T15 \
             chunk-aware streaming pipeline; this session lands the scaffold + \
             chain wiring only (call synthesize_with_pipeline for the injected-\
             closure oracle path)",
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

    /// A 0-placeholder GGUF is the one LLM-binding failure the engine
    /// deliberately tolerates (the container must stay loadable), and it must
    /// keep loading — with the LLM handle honestly absent.
    #[test]
    fn placeholder_shape_gguf_loads_with_absent_llm() {
        let bytes = minimal_gguf_bytes(EXPECTED_ARCH);
        let tts = CosyVoice2Tts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect("placeholder GGUF must still load");
        assert!(tts.llm().is_none(), "0-placeholder dims → no LLM backbone");
    }

    /// Regression for the 2026-07-19 audit (cc-28): the engine used to key its
    /// tolerance on the *error variant* (`InvalidArgument` → `None`), so a GGUF
    /// that was genuinely malformed rather than merely old was swallowed just
    /// the same and reported a successful load. Non-zero but non-GQA dims are
    /// exactly that case — `n_head = 7` does not divide `hidden_dim = 512` — and
    /// must now fail loudly (FR-EX-08).
    #[test]
    fn malformed_llm_dims_fail_loudly_instead_of_binding_none() {
        let mut b = GgufBuilder::new();
        b.add_string(KEY_MODEL_ARCH, EXPECTED_ARCH);
        b.add_string("vokra.model.name", "cosyvoice2-0.5b");
        b.add_u32(config::KEY_SAMPLE_RATE, 24_000);
        // Non-zero (so not the placeholder sentinel) but not GQA-well-formed.
        b.add_u32(config::KEY_VOCAB_SIZE, 1024);
        b.add_u32(config::KEY_HIDDEN_DIM, 512);
        b.add_u32(config::KEY_N_LAYER, 2);
        b.add_u32(config::KEY_N_HEAD, 7);
        b.add_u32(config::KEY_FFN_DIM, 1024);
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
        let bytes = b.to_bytes().expect("gguf serialize");

        let err = CosyVoice2Tts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect_err("malformed LLM dims must not load as `llm = None`");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "unexpected error variant: {err:?}"
        );
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
        let VokraError::NotImplemented(msg) = err else {
            panic!("unexpected error variant: {err:?}");
        };
        // This fixture carries 0-placeholder LLM dims, so the message must name
        // *that* blocker and its fix, not the generic scaffold text. Asserting
        // only the variant let the branch added for the 2026-07-19 audit
        // (cc-28) be deleted with every test still green.
        assert!(
            msg.contains("0-placeholder") && msg.contains("--config"),
            "placeholder GGUF must name its own blocker: {msg}"
        );
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

    // ---- Pipeline integration through CosyVoice2Tts --------------------

    fn nondegenerate_gguf_bytes() -> Vec<u8> {
        // Same fixture as minimal_gguf_bytes but with sane mimi_* +
        // streaming_* values so the pipeline can actually run.
        let mut b = GgufBuilder::new();
        b.add_string(KEY_MODEL_ARCH, "cosyvoice2");
        b.add_string("vokra.model.name", "cosyvoice2-0.5b");
        b.add_u32(config::KEY_SAMPLE_RATE, 24_000);
        b.add_u32(config::KEY_VOCAB_SIZE, 32);
        b.add_u32(config::KEY_HIDDEN_DIM, 16);
        b.add_u32(config::KEY_N_LAYER, 2);
        b.add_u32(config::KEY_N_HEAD, 2);
        b.add_u32(config::KEY_FFN_DIM, 32);
        b.add_u32(config::KEY_FLOW_NFE, 2);
        b.add_u32(config::KEY_MIMI_N_CODEBOOKS, 2);
        b.add_u32(config::KEY_MIMI_CODEBOOK_SIZE, 8);
        b.add_u32(config::KEY_MIMI_D_MODEL, 4);
        b.add_u32(config::KEY_STREAMING_CHUNK_SIZE, 4);
        b.add_u32(config::KEY_STREAMING_CHUNK_HOP, 4);
        b.add_metadata(
            config::KEY_FLOW_SCHEDULE,
            GgufMetadataValue::String("linear".to_owned()),
        );
        b.to_bytes().expect("gguf serialize")
    }

    #[test]
    fn synthesize_with_pipeline_end_to_end_smoke() {
        // Full pipeline run through the engine handle: length_conditioning
        // → run_chunks → identity MimiDecoder → PipelineOutput.
        //
        // Uses a zero-velocity closure so each chunk's terminal is the
        // chunk's initial state (predictable). The code closure emits
        // constant 1s → identity decoder produces a predictable feature
        // buffer. This is the internal-oracle path — no real safetensors
        // checkpoint invoked (CLAUDE.md hallucination ban).
        let bytes = nondegenerate_gguf_bytes();
        let tts = CosyVoice2Tts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect("load");
        let length_input =
            vokra_core::ir::graph::LengthConditioningAttrs::user_specified_frames(6.0);
        let x0 = vokra_ops::FlowSamplerState::new(vec![1], vec![0.0]).unwrap();
        let out = tts
            .synthesize_with_pipeline(
                length_input,
                &x0,
                |s, _t, _p, _c| {
                    Ok(vokra_ops::FlowSamplerState {
                        shape: s.shape.clone(),
                        data: vec![0.0; s.data.len()],
                    })
                },
                |_s, chunk_frames, n_cb| Ok(vec![1u32; chunk_frames * n_cb]),
            )
            .expect("pipeline succeeds");
        assert_eq!(out.target_frames, 6);
        assert_eq!(out.chunks.len(), 2, "6 frames / 4 chunk_size → 2 chunks");
        assert_eq!(out.chunks[0].chunk_frames, 4);
        assert_eq!(out.chunks[1].chunk_frames, 2);
        // Every feature must be finite (the "no NaN" invariant).
        for c in &out.chunks {
            for &v in &c.features {
                assert!(v.is_finite(), "feature must be finite");
            }
        }
    }

    #[test]
    fn synthesize_with_pipeline_propagates_synthesize_stub_rationale() {
        // FR-EX-08: the top-level TtsEngine::synthesize returns
        // NotImplemented (real LLM path unwired), but
        // synthesize_with_pipeline succeeds because it accepts injected
        // closures. This mirrors the T14 streaming pipeline promotion
        // pattern: today's testable oracle path does not depend on
        // upstream tensor names.
        let bytes = nondegenerate_gguf_bytes();
        let tts = CosyVoice2Tts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect("load");
        // Native synthesize() still stub.
        let err = tts
            .synthesize(&SynthesisRequest::new("hello"))
            .expect_err("no real LLM path yet");
        assert!(matches!(err, VokraError::NotImplemented(_)));
        // Pipeline path succeeds with injected closures.
        let length_input =
            vokra_core::ir::graph::LengthConditioningAttrs::user_specified_frames(4.0);
        let x0 = vokra_ops::FlowSamplerState::new(vec![1], vec![0.0]).unwrap();
        let out = tts
            .synthesize_with_pipeline(
                length_input,
                &x0,
                |s, _t, _p, _c| {
                    Ok(vokra_ops::FlowSamplerState {
                        shape: s.shape.clone(),
                        data: vec![0.0; s.data.len()],
                    })
                },
                |_s, chunk_frames, n_cb| Ok(vec![0u32; chunk_frames * n_cb]),
            )
            .expect("pipeline succeeds");
        assert_eq!(out.target_frames, 4);
        assert_eq!(out.chunks.len(), 1);
    }
}
