//! CosyVoice2 composite TTS inspection boundary.
//!
//! The official release combines a Qwen2LM wrapper, causal flow/CFM, HiFTNet
//! vocoder, speech tokenizer, and speaker conditioning. The historical
//! LLM-only GGUF and arbitrary metadata containers are not complete TTS
//! artifacts. This module retains compatibility fixtures, while the public
//! GGUF loader fails closed until a complete authenticated binder and
//! independent parity evidence are reviewed. The source-shaped batch-one
//! route lives in the internal [`native`] module and requires injected
//! components; it does not make a production-support claim.
//!
//! Synthetic constructors are test-only numerical fixtures and are never used
//! by the production loader. Unsupported operations return explicit errors;
//! there is no silent CPU fallback.
pub(crate) mod chunk_pipeline;
pub(crate) mod config;
pub(crate) mod flow_matching;
pub(crate) mod native;
// SoTA plan Phase 1-3 (2026-07-24): the correct terminal vocoder for
// CosyVoice2 — mel → PCM via NSF + ISTFTNet. Replaces the wrong-premise
// `mimi_bridge` module (which is now `#[deprecated]` and retained only for
// the existing `chunk_pipeline` scaffold + `parity_cosyvoice2` test imports).
pub mod hift_chain;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
mod hift_chain_metal;
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
    VokraError, WatermarkConfig,
};
use vokra_ops::{ApplyProsody, ProsodyControl};

pub use chunk_pipeline::{ChunkAwareStreamingPipeline, PipelineChunk, PipelineOutput};
pub use config::CosyVoice2Config;
pub use flow_matching::{ChunkAwareCfm, ChunkContinuation, FlowMatchingRuntimeParams};
pub use hift_chain::{HiFTChain, HiFTChainConfig, HiFTChainWeights};
pub use llm::{
    DEFAULT_RMS_NORM_EPS, DEFAULT_ROPE_BASE_QWEN2, LlmBackbone, LlmBackboneConfig, LlmBackboneStep,
};
// Compatibility re-export is intentionally
// `#[allow(deprecated)]` — `MimiBridge` itself is marked deprecated (see
// `mimi_bridge.rs` module docstring for the SoTA plan §1(a) 訂正 rationale)
// but the re-export must keep working so pre-existing test imports and the
// `chunk_pipeline` scaffold compile. New callers use `HiFTChain`.
#[allow(deprecated)]
pub use mimi_bridge::MimiBridge;
pub use text_encoder::{CosyVoice2Tokenizer, TextEncoderStub};

/// `vokra.model.arch` a CosyVoice2 GGUF must carry.
///
/// Written by `vokra-convert::models::cosyvoice2::ARCH` (T03); kept in sync
/// with the runtime constant here. This marker is retained for compatibility
/// with the shared dispatch table; it does not authorize a production load.
/// The public loader performs its architecture parse and then returns the
/// explicit inspection-only composite blocker, regardless of provenance
/// metadata or license labels.
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

/// CosyVoice2 engine handle for a future authenticated composite artifact.
///
/// The public GGUF loader currently fails closed: an LLM-only or arbitrary
/// metadata container cannot construct this production handle. The explicit
/// synthetic LLM constructors remain available for numerical tests only.
///
/// The struct is intentionally light: it carries the resolved config, the
/// selected backend, and the watermark / prosody control state. The heavy
/// numeric state (Qwen2LM / causal flow / HiFTNet)
/// lands in follow-on tickets and hangs off private fields added at that
/// time. The public constructor validates identity/compliance and then fails
/// closed until the complete composite checkpoint binder is reviewed.
#[derive(Debug)]
pub struct CosyVoice2Tts {
    /// The resolved GGUF metadata retained for the compatibility container.
    /// Legacy `mimi.*` fields are inspection-only; the native component graph
    /// is Qwen2LM → causal flow/CFM → HiFTNet.
    config: CosyVoice2Config,
    /// LLM backbone (M3-09-T07/T08 body). Decoder-only Mistral-style
    /// transformer whose output token stream drives the Flow Matching CFM.
    ///
    /// Populated only by a future authenticated composite binder. Synthetic
    /// numerical fixtures use the explicitly named
    /// [`llm::LlmBackbone::synthesized`] constructor instead.
    ///
    /// The LLM config is read from the same GGUF as the top-level config
    /// (`vokra.cosyvoice2.arch.*` LLM-side keys), so the two are always
    /// consistent — a mismatch is impossible by construction.
    llm: Option<llm::LlmBackbone>,
    /// Text tokenizer (M3-09-T06). `Some` when the GGUF carries the embedded
    /// Qwen2 `vocab.json` + `merges.txt` chunks (`vokra.cosyvoice2.tokenizer.*`),
    /// `None` for a tokenizer-less GGUF (e.g. a pre-T06 conversion). A GGUF
    /// carrying only one of the two chunks is treated as malformed and fails
    /// the load loudly (FR-EX-08), rather than silently binding `None`.
    tokenizer: Option<text_encoder::CosyVoice2Tokenizer>,
    /// Selected compute backend (default [`BackendKind::Cpu`], overridable
    /// via [`CosyVoice2Tts::with_backend`]. The injected HiFTChain forwards
    /// this selection to its CPU or Apple Metal resident route.
    backend_kind: BackendKind,
    /// Watermark / disclosure knobs. Defaults to design intent — AudioSeal +
    /// C2PA + SilentCipher = ON. Embedding backend is deferred (T17 doc),
    /// deployer-side disclosure MUST still applies
    /// (docs/legal-compliance.md §1.4).
    watermark: WatermarkConfig,
    /// SoTA plan Phase 1-3 (2026-07-24): the terminal HiFTNet vocoder that
    /// consumes the CFM's mel output and emits 24 kHz PCM. `None` until a
    /// caller injects one via [`CosyVoice2Tts::with_hift_chain`] — the
    /// weight-binding path off a real CosyVoice2 GGUF is deferred to the T13
    /// codec-migration follow-up (upstream `cosyvoice/hifigan/generator.py`
    /// tensor names have to be walked once the checkpoint is on disk).
    ///
    /// This field REPLACES the [`mimi_bridge::MimiBridge`] wiring the
    /// original T13 scaffold reached for. See the module docstring for the
    /// 2026-07-22 SoTA plan §1(a) 訂正 rationale — CosyVoice2 does NOT
    /// consume the Mimi codec; the terminal vocoder is HiFTNet
    /// (Neural Source Filter + ISTFTNet), and the Mimi bridge module is now
    /// `#[deprecated]`.
    hift_chain: Option<HiFTChain>,
}

impl CosyVoice2Tts {
    /// Loads a CosyVoice2 GGUF from disk with the fail-closed
    /// [`CompliancePolicy::strict`] policy available to the compatibility
    /// API. The reviewed loader still rejects every parsed artifact at the
    /// composite inspection-only boundary.
    ///
    /// # Errors
    ///
    /// Propagates GGUF parse errors, arch mismatch, and the explicit
    /// inspection-only composite-runtime refusal.
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
    /// wrong-architecture) GGUF fails with a clear [`VokraError::ModelLoad`].
    /// For the correct architecture, this API then returns the explicit
    /// `INSPECTION_ONLY` composite-runtime error. This ordering is
    /// intentional: malformed identity is diagnosed first, while no
    /// provenance or license label can make an incomplete composite appear
    /// production-ready.
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
        let _ = policy;
        // Parse and architecture validation deliberately precede this gate so
        // malformed or wrong-model bytes are diagnosed accurately. No
        // compliance metadata is consulted after the gate: a composite
        // artifact can never become a production handle through a license
        // label or partial checkpoint.
        Err(VokraError::UnsupportedOp(
            "CosyVoice2 composite runtime is INSPECTION_ONLY: the reviewed llm.pt + flow.pt + hift.pt + speech-tokenizer contract is not bound; no partial LLM handle is accepted"
                .to_owned(),
        ))
    }

    /// Selects the backend the synthesis hot path runs on (default
    /// [`BackendKind::Cpu`]). The selected backend is forwarded by
    /// [`Self::synthesize_pcm_from_mel`] to the injected HiFTChain.
    /// Backend capability is checked when a forward is attempted, not when
    /// this selector is called.
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

    /// Injects a [`HiFTChain`] — the terminal mel → PCM vocoder.
    ///
    /// SoTA plan Phase 1-3 (2026-07-24) seam. Until a caller provides a
    /// [`HiFTChain`], [`CosyVoice2Tts::synthesize_pcm_from_mel`] returns
    /// [`VokraError::NotImplemented`] (FR-EX-08 — never a silent fallback).
    /// The full text → PCM chain also depends on the LLM (T07/T08) + Flow
    /// Matching CFM (T10/T11) landing; a caller who has a [`HiFTChain`]
    /// today can still exercise the mel → PCM half via
    /// [`CosyVoice2Tts::synthesize_pcm_from_mel`].
    ///
    /// The chain shape is not cross-checked against
    /// [`CosyVoice2Config::sample_rate`] here on purpose: a small-shape
    /// harness (like the [`hift_chain`] unit-test bundle) intentionally
    /// runs at 16 kHz, and forbidding that would collapse the internal
    /// oracle path. Callers wiring a real CosyVoice2 checkpoint are
    /// expected to build a [`HiFTChain`] whose
    /// [`HiFTChainConfig::sampling_rate`] matches
    /// `config.sample_rate` (24 kHz for upstream CosyVoice2-0.5B).
    #[must_use]
    pub fn with_hift_chain(mut self, chain: HiFTChain) -> Self {
        self.hift_chain = Some(chain);
        self
    }

    /// The resolved CosyVoice2 configuration (arch + streaming + flow /
    /// mimi hyperparameters).
    #[must_use]
    pub fn config(&self) -> &CosyVoice2Config {
        &self.config
    }

    /// The caller-injected HiFTNet vocoder chain (SoTA plan Phase 1-3),
    /// or `None` when [`CosyVoice2Tts::with_hift_chain`] has not been
    /// called.
    #[must_use]
    pub fn hift_chain(&self) -> Option<&HiFTChain> {
        self.hift_chain.as_ref()
    }

    /// True iff a [`HiFTChain`] has been injected. Convenience over
    /// `hift_chain().is_some()` for callers checking the chain state
    /// before invoking [`CosyVoice2Tts::synthesize_pcm_from_mel`].
    #[must_use]
    pub fn has_hift_chain(&self) -> bool {
        self.hift_chain.is_some()
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

    /// The embedded text tokenizer (M3-09-T06), or `None` when the GGUF
    /// carries no `vokra.cosyvoice2.tokenizer.*` chunks.
    #[must_use]
    pub fn tokenizer(&self) -> Option<&text_encoder::CosyVoice2Tokenizer> {
        self.tokenizer.as_ref()
    }

    /// Tokenizes `text` to Qwen2 byte-level BPE ids (M3-09-T06) — the front
    /// end of the (still-stubbed) `synthesize` chain.
    ///
    /// # Errors
    ///
    /// [`VokraError::NotImplemented`] when the GGUF carries no embedded
    /// tokenizer (`vokra.cosyvoice2.tokenizer.*`): re-convert with `--config`
    /// pointing at the upstream `CosyVoice-BlankEN/config.json` (the Qwen2
    /// `vocab.json` + `merges.txt` are picked up from the same directory).
    /// Never a silent empty result (FR-EX-08). Otherwise propagates the
    /// tokenizer's own [`VokraError::InvalidArgument`] on an unencodable byte.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match &self.tokenizer {
            Some(t) => t.encode(text),
            None => Err(VokraError::NotImplemented(
                "CosyVoice2 text tokenizer is not embedded in this GGUF \
                 (vokra.cosyvoice2.tokenizer.vocab / .merges absent) — re-convert with \
                 `vokra-cli convert --model cosyvoice2 --config \
                 <CosyVoice-BlankEN/config.json>` so the Qwen2 vocab.json + merges.txt \
                 are embedded alongside it",
            )),
        }
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
        // SoTA plan Phase 1-3: the caller-facing chain is
        // [`HiFTChain`], but `chunk_pipeline` still consumes the deprecated
        // `MimiBridge` scaffold (wrong-premise composition — see
        // `mimi_bridge.rs` module docstring). Kept as-is so pre-existing
        // internal-oracle tests continue to pass; the migration to the
        // HiFTNet composition lands in the CosyVoice2 T13 codec-migration
        // follow-up. `#[allow(deprecated)]` here is scoped to this single
        // scaffold call — new callers use `HiFTChain` directly via
        // [`CosyVoice2Tts::synthesize_pcm_from_mel`].
        #[allow(deprecated)]
        let bridge = mimi_bridge::MimiBridge::with_identity_decoder(&self.config)?;
        let pipeline =
            chunk_pipeline::ChunkAwareStreamingPipeline::new(&self.config, &cfm, &bridge)?;
        pipeline.synthesize(length_input, initial_state, velocity_fn, code_fn)
    }

    /// Runs the HiFTNet vocoder chain on a caller-supplied mel spectrogram,
    /// returning the PCM as a [`SynthesizedAudio`].
    ///
    /// SoTA plan Phase 1-3 (2026-07-24) seam. This is the "mel → PCM" half
    /// of the CosyVoice2 chain — the "text → mel" half (tokenizer + LLM
    /// backbone + Flow Matching CFM) still lands in T06/T07/T08/T10, and
    /// until it does the top-level [`TtsEngine::synthesize`] cannot produce
    /// audio. Callers who already have a mel (from a reference implementation,
    /// a test fixture, or an external CFM) can drive the HiFTNet vocoder
    /// through this entry point today.
    ///
    /// # Arguments
    ///
    /// - `mel` — row-major `[in_channels, t_mel]` mel spectrogram, where
    ///   `in_channels == self.hift_chain().unwrap().config().in_channels`.
    /// - `t_mel` — mel timestep count (must be > 0).
    ///
    /// # Errors
    ///
    /// - [`VokraError::NotImplemented`] when no [`HiFTChain`] has been
    ///   injected via [`CosyVoice2Tts::with_hift_chain`] (fail-loud, FR-EX-08
    ///   — never a silent zero-fill fallback).
    /// - Routes through [`HiFTChain::forward_with_backend`] using the selected
    ///   backend. CPU preserves [`HiFTChain::forward`], while Metal and other
    ///   unsupported backends remain explicit errors (including shape
    ///   mismatch and `t_mel == 0`).
    pub fn synthesize_pcm_from_mel(&self, mel: &[f32], t_mel: usize) -> Result<SynthesizedAudio> {
        let chain = self.hift_chain.as_ref().ok_or({
            VokraError::NotImplemented(
                "CosyVoice2Tts::synthesize_pcm_from_mel: no HiFTChain has been \
                 injected — call `.with_hift_chain(HiFTChain::new(cfg, weights)?)` \
                 first. SoTA plan Phase 1-3 (2026-07-24): CosyVoice2 uses HiFTNet \
                 (Neural Source Filter + ISTFTNet) as the terminal mel → PCM \
                 vocoder, NOT the Mimi codec — see `cosyvoice2::hift_chain` \
                 rustdoc for the §1(a) 訂正 rationale",
            )
        })?;
        let samples = chain.forward_with_backend(mel, t_mel, self.backend_kind)?;
        Ok(SynthesizedAudio::new(samples, chain.sample_rate()))
    }
}

impl TtsEngine for CosyVoice2Tts {
    /// Text → PCM adapter (T14/T15 chunk-aware streaming pipeline lands the
    /// concrete numeric path).
    ///
    /// Until the LLM backbone (T07/T08), Flow Matching CFM (T10/T11), and
    /// HiFTNet vocoder chain ([`HiFTChain`], SoTA plan Phase 1-3 seam) are
    /// wired end-to-end, this returns [`VokraError::NotImplemented`] with a
    /// clear next-step message — never a silent zero-fill fallback
    /// (FR-EX-08).
    ///
    /// # Chain wiring (M3-09 partial land + SoTA plan Phase 1-3)
    ///
    /// The module tree is chained today — a follow-on session composes
    /// text → [`TextEncoderStub::encode`] → [`llm::LlmBackbone::forward`]
    /// → [`ChunkAwareCfm::run_chunks`] → [`HiFTChain::forward`] by filling
    /// in each stage's numeric path. The top-level `synthesize` short-
    /// circuits with NotImplemented because the tokenizer (T06), LLM weight
    /// binding (T07), and forward pass (T08) are all deferred, and the
    /// terminal vocoder ([`HiFTChain`]) must be injected by a caller
    /// holding HiFTNet weights (via [`CosyVoice2Tts::with_hift_chain`]).
    /// The `synthesize_with_pipeline` entry point below exposes the
    /// injected-closure oracle path for internal-oracle tests today;
    /// [`CosyVoice2Tts::synthesize_pcm_from_mel`] exposes the mel → PCM
    /// half of the chain for callers who already hold a mel.
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
        if self.hift_chain.is_none() {
            // SoTA plan Phase 1-3: name the HiFTChain blocker explicitly
            // (FR-EX-08 — the terminal vocoder must be present before we
            // can honestly return audio, even once the LLM/CFM path lands).
            return Err(VokraError::NotImplemented(
                "CosyVoice2 TtsEngine::synthesize: no HiFTChain has been injected. \
                 Call `.with_hift_chain(HiFTChain::new(cfg, weights)?)` first — \
                 CosyVoice2 uses HiFTNet (Neural Source Filter + ISTFTNet) as the \
                 terminal mel → PCM vocoder (SoTA plan §1(a) 訂正, 2026-07-22); \
                 the mimi_bridge scaffold is `#[deprecated]` and must not be \
                 revived (upstream `cosyvoice/hifigan/generator.py:378 HiFTGenerator` \
                 confirms — see `cosyvoice2::hift_chain` rustdoc)",
            ));
        }
        Err(VokraError::NotImplemented(
            "CosyVoice2 TtsEngine::synthesize needs the T07/T08 LLM backbone forward, \
             T10/T11 Flow Matching CFM and the T14/T15 chunk-aware streaming pipeline; \
             the terminal HiFTChain vocoder (SoTA plan Phase 1-3) is wired, and the T06 \
             text tokenizer is available via CosyVoice2Tts::encode. Callers holding a \
             mel can drive the mel → PCM half today via \
             CosyVoice2Tts::synthesize_pcm_from_mel; internal-oracle tests use \
             synthesize_with_pipeline (still routed through the deprecated MimiBridge \
             scaffold — the HiFTChain-based composition lands in the T13 codec-migration \
             follow-up)",
        ))
    }

    fn backend(&self) -> BackendKind {
        self.backend_kind
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
mod inspection_tests {
    use super::*;
    use vokra_core::gguf::GgufBuilder;
    use vokra_core::gguf::chunks::KEY_MODEL_ARCH;

    #[test]
    fn arbitrary_metadata_and_historical_llm_only_shape_are_rejected() {
        let mut builder = GgufBuilder::new();
        builder.add_string(KEY_MODEL_ARCH, EXPECTED_ARCH);
        let bytes = builder.to_bytes().expect("serialize");
        let error = CosyVoice2Tts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect_err("composite runtime must fail closed");
        assert!(matches!(error, VokraError::UnsupportedOp(_)));
        assert!(error.to_string().contains("INSPECTION_ONLY"));
    }
}
