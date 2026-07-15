//! `MoshiEngine` — the [`S2sDuplexEngine`] + [`S2sEngine`] implementation
//! wrapping the Moshi pipeline (M4-06-T19): frame generation (T12) +
//! shared Mimi codec ends (T04〜T08 = consume, ADR M4-06 §D1-(b)) + AEC
//! posture (T21) + attribution surface (T23).
//!
//! # Text contract — the inverse of CSM (ADR M4-06 §D5)
//!
//! Moshi **generates its own reply** (the inner monologue is its
//! transcript); [`DialogRequest::reply_text`] must therefore be **empty**
//! — a caller-supplied reply is a loud [`VokraError::InvalidArgument`]
//! pointing at CSM (which has the opposite contract). `S2s::dialog
//! (samples)` — whose default request carries an empty reply — flows
//! through unchanged, and [`DialogTurn::text`] returns the decoded
//! monologue.
//!
//! # Loading (FR-EX-08 posture)
//!
//! `from_gguf_with_policy` mirrors the CSM/CosyVoice2 gate order: arch
//! check → M2-13 weight-license gate (Moshi = CC-BY 4.0 →
//! `AttributionRequired`, commercially allowed, **no** research flag) →
//! config read → weight binding. The **LM weights bind for real** (the
//! T02 manifest pinned the upstream tensor names — `MoshiBackboneWeights
//! ::from_gguf`); the **Mimi ends stay on the synthesized bridge** until
//! the shared module's T29 binding lands (`MimiEncoder::from_gguf` is an
//! honest `NotImplemented` — M4-05 posture, documented there). The
//! attribution surface resolves at load
//! ([`vokra_core::resolve_attribution`]) so every deployer face (Rust /
//! C ABI / CLI banner) reads one source.

use std::sync::Arc;

use vokra_core::{
    AttributionInfo, BackendKind, CompliancePolicy, DialogRequest, DialogTurn, DuplexSessionConfig,
    Result, S2sDuplexEngine, S2sDuplexHandle, S2sEngine, SynthesizedAudio, VokraError,
    WatermarkConfig, check_weight_license, gguf::GgufFile, resolve_attribution,
};
use vokra_ops::aec::AecAttrs;
use vokra_ops::mimi_rvq::MimiRvqAttrs;

use super::config::MoshiConfig;
use super::duplex::MoshiDuplexSession;
use super::frame::MoshiModel;
use super::tokenizer::{FixtureMoshiTokenizer, GgufMoshiTokenizer, MoshiTextTokenizer};
use crate::csm::aec_front::AecFront;
use crate::csm::audio::CsmAudioDecodeChain;
use crate::csm::{EchoPath, pad_to_whole_frames};
use crate::mimi::{MimiEncoder, MimiNeuralConfig, MimiNeuralDecoder};

/// One session's assembled input front: the optional canceller pair
/// plus the construction-time warnings (T21 posture check output).
type DuplexFront = (
    Option<(AecFront, vokra_core::stream::AecRefWriter)>,
    Vec<String>,
);

/// The stored AEC construction recipe: each duplex session builds a
/// **fresh** canceller + reference queue from it (per-session echo path;
/// contrast the turn-based CSM engine's single shared front).
#[derive(Debug, Clone, Copy)]
struct AecRecipe {
    attrs: AecAttrs,
    queue_capacity_samples: usize,
}

/// The Moshi engine (module docs).
pub struct MoshiEngine {
    model: MoshiModel,
    encoder: MimiEncoder,
    chain: CsmAudioDecodeChain,
    tokenizer: Arc<dyn MoshiTextTokenizer>,
    mimi_config: MimiNeuralConfig,
    aec: Option<AecRecipe>,
    echo_path: EchoPath,
    watermark: WatermarkConfig,
    attribution: Option<AttributionInfo>,
}

impl std::fmt::Debug for MoshiEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoshiEngine")
            .field("config", self.model.config())
            .field("echo_path", &self.echo_path)
            .field("aec_wired", &self.aec.is_some())
            .field("attribution", &self.attribution.is_some())
            .finish()
    }
}

impl MoshiEngine {
    /// Assembles an engine from explicit components, validating the codec
    /// seams: Moshi duplex is symmetric (`dep_q` own codes decoded,
    /// `n_user_streams` mic codes encoded) over **one** Mimi codec, so
    /// the quantizer width must equal both (`CheckpointInfo.get_mimi`:
    /// `num_codebooks = max(dep_q, n_q − dep_q)` with both sides 8 on the
    /// 7B model — ADR M4-06 §D1-(b)).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the mismatched seam.
    pub fn new(
        model: MoshiModel,
        encoder: MimiEncoder,
        chain: CsmAudioDecodeChain,
        tokenizer: Arc<dyn MoshiTextTokenizer>,
        mimi_config: MimiNeuralConfig,
    ) -> Result<Self> {
        let cfg = model.config();
        if cfg.dep_q != cfg.n_user_streams() {
            return Err(VokraError::InvalidArgument(format!(
                "moshi engine: dep_q {} != user streams {} — the shared Mimi codec \
                 serves both directions at one width (loaders.py get_mimi)",
                cfg.dep_q,
                cfg.n_user_streams()
            )));
        }
        if encoder.config().quantizer.n_q != cfg.n_user_streams() {
            return Err(VokraError::InvalidArgument(format!(
                "moshi engine: mimi encoder quantizes {} codebooks but the model \
                 consumes {} user streams",
                encoder.config().quantizer.n_q,
                cfg.n_user_streams()
            )));
        }
        if encoder.config().quantizer.bins != cfg.audio_card {
            return Err(VokraError::InvalidArgument(format!(
                "moshi engine: mimi codebook bins {} != audio card {} — upstream \
                 ties them (`_lm_kwargs[\"card\"] = _quantizer_kwargs[\"bins\"]`, \
                 loaders.py); a mismatch would emit codes the LM rejects (FR-EX-08 \
                 at construction, not mid-stream)",
                encoder.config().quantizer.bins,
                cfg.audio_card
            )));
        }
        if chain.attrs().n_codebooks != cfg.dep_q {
            return Err(VokraError::InvalidArgument(format!(
                "moshi engine: decode chain has {} codebooks but the depformer \
                 generates {}",
                chain.attrs().n_codebooks,
                cfg.dep_q
            )));
        }
        if tokenizer.vocab_size() != cfg.text_card {
            return Err(VokraError::InvalidArgument(format!(
                "moshi engine: tokenizer vocab {} != config text_card {}",
                tokenizer.vocab_size(),
                cfg.text_card
            )));
        }
        Ok(Self {
            model,
            encoder,
            chain,
            tokenizer,
            mimi_config,
            aec: None,
            echo_path: EchoPath::AecRequired,
            watermark: WatermarkConfig::default(),
            attribution: None,
        })
    }

    /// A fully synthesized fixture engine (tiny dims, deterministic): the
    /// duplex pipeline end to end without any real weight. The fixture
    /// Mimi quantizer is reshaped so both codec ends match the tiny Moshi
    /// stream split (`n_q = dep_q = n_user`, `bins = audio_card`).
    ///
    /// # Errors
    ///
    /// Propagates component construction errors.
    pub fn synthesized_fixture(seed: u64) -> Result<Self> {
        Self::synthesized_with_config(MoshiConfig::tiny_for_tests(), seed)
    }

    /// [`Self::synthesized_fixture`] with an explicit (tiny) config.
    ///
    /// # Errors
    ///
    /// Propagates component construction errors.
    pub fn synthesized_with_config(cfg: MoshiConfig, seed: u64) -> Result<Self> {
        let mut mimi_cfg = MimiNeuralConfig::tiny_for_tests();
        mimi_cfg.quantizer.n_q = cfg.dep_q;
        mimi_cfg.quantizer.bins = cfg.audio_card;
        mimi_cfg.validate()?;
        let model = MoshiModel::synthesized(cfg.clone(), seed)?;
        let encoder = MimiEncoder::synthesized(&mimi_cfg, seed ^ 0x5EED_5EED)?;
        let neural = MimiNeuralDecoder::synthesized(&mimi_cfg, seed ^ 0xDEC0_DEC0, true)?;
        let attrs = MimiRvqAttrs {
            n_codebooks: mimi_cfg.quantizer.n_q,
            codebook_size: mimi_cfg.quantizer.bins,
            d_model: mimi_cfg.quantizer.dimension,
        };
        let chain = CsmAudioDecodeChain::new(encoder.tables().to_vec(), attrs, neural)?;
        let tokenizer = Arc::new(FixtureMoshiTokenizer::new(cfg.text_card)?);
        Self::new(model, encoder, chain, tokenizer, mimi_cfg)
    }

    /// Loads a Moshi GGUF from raw bytes under `policy` (M2-13 gate; the
    /// CC-BY 4.0 `AttributionRequired` class passes commercially and the
    /// attribution surface resolves here — module docs).
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] on a wrong arch / missing tensor /
    /// missing tokenizer blob; compliance-gate refusals verbatim;
    /// [`VokraError::InvalidArgument`] on a `0`-placeholder shape config.
    pub fn from_gguf_with_policy(bytes: &[u8], policy: &CompliancePolicy) -> Result<Self> {
        let file = GgufFile::parse(bytes.to_vec())
            .map_err(|e| VokraError::ModelLoad(format!("moshi GGUF: {e}")))?;
        let arch = file
            .get(vokra_core::gguf::chunks::KEY_MODEL_ARCH)
            .and_then(|v| v.as_str());
        if arch != Some(super::EXPECTED_ARCH) {
            return Err(VokraError::ModelLoad(format!(
                "not a Moshi GGUF: vokra.model.arch = {arch:?}, expected `{}`",
                super::EXPECTED_ARCH
            )));
        }
        check_weight_license(&file, policy)?;
        let attribution = resolve_attribution(&file);
        let cfg = MoshiConfig::from_gguf(&file)?;
        cfg.validate_for_forward()?;
        let mimi_cfg = MimiNeuralConfig::from_gguf(&file)?;
        mimi_cfg.validate()?;
        // LM weights bind for real (T02 manifest names); the Mimi ends are
        // the documented synthesized bridge until the shared module's T29
        // binding (module docs).
        let backbone = super::backbone::MoshiBackboneWeights::from_gguf(&file, &cfg)?;
        let depth = super::depth::MoshiDepthWeights::from_gguf(&file, &cfg)?;
        let model = MoshiModel::new(cfg.clone(), backbone, depth)?;
        let encoder = MimiEncoder::synthesized(
            &mimi_cfg,
            super::backbone::MOSHI_FROM_GGUF_DEFAULT_SEED ^ 0x5EED,
        )?;
        let neural = MimiNeuralDecoder::synthesized(
            &mimi_cfg,
            super::backbone::MOSHI_FROM_GGUF_DEFAULT_SEED ^ 0xDEC0,
            true,
        )?;
        let attrs = MimiRvqAttrs {
            n_codebooks: mimi_cfg.quantizer.n_q,
            codebook_size: mimi_cfg.quantizer.bins,
            d_model: mimi_cfg.quantizer.dimension,
        };
        let chain = CsmAudioDecodeChain::new(encoder.tables().to_vec(), attrs, neural)?;
        let tokenizer: Arc<dyn MoshiTextTokenizer> =
            Arc::new(GgufMoshiTokenizer::from_gguf(&file, cfg.text_card)?);
        let mut engine = Self::new(model, encoder, chain, tokenizer, mimi_cfg)?;
        engine.attribution = attribution;
        Ok(engine)
    }

    /// Loads from a file path with the fail-closed strict policy.
    ///
    /// # Errors
    ///
    /// See [`Self::from_gguf_with_policy`].
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).map_err(VokraError::Io)?;
        Self::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
    }

    /// Wires the AEC recipe: each duplex session builds a fresh canceller
    /// plus a time-tagged far-end queue from it (M4-03 consumer contract).
    /// The attrs' sample rate must match the Mimi PCM rate.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a rate mismatch (loud — a
    /// wrong-rate canceller would silently mis-align the echo).
    pub fn with_aec(mut self, attrs: &AecAttrs, queue_capacity_samples: usize) -> Result<Self> {
        if attrs.sample_rate != self.mimi_config.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "moshi engine: AEC sample rate {} != Mimi PCM rate {} (one clock — \
                 FR-EX-08)",
                attrs.sample_rate, self.mimi_config.sample_rate
            )));
        }
        // Validate the attrs eagerly (fail at wiring, not first session).
        let _probe = AecFront::new(attrs, queue_capacity_samples)?;
        self.aec = Some(AecRecipe {
            attrs: *attrs,
            queue_capacity_samples,
        });
        Ok(self)
    }

    /// Selects the batch-`dialog` echo posture (default
    /// [`EchoPath::AecRequired`]; the recorded-input bypass is an
    /// explicit opt-in — `csm::aec_front` rules). Duplex sessions take
    /// the equivalent switch per session via
    /// [`DuplexSessionConfig::with_aec_disabled_explicitly`].
    #[must_use]
    pub fn with_echo_path(mut self, path: EchoPath) -> Self {
        self.echo_path = path;
        self
    }

    /// Routes the LM + codec hot ops through `backend` (explicit; an
    /// unsupported op on the selected backend is a loud error — FR-EX-08,
    /// no silent CPU fallback).
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.model = self.model.with_backend(backend);
        self.encoder = self.encoder.with_backend(backend);
        self
    }

    /// Overrides the watermark configuration (default **ON**; the
    /// embedding backend stays Deferred — deployer-side disclosure is a
    /// MUST, docs/legal-compliance.md §1.4).
    #[must_use]
    pub fn with_watermark(mut self, watermark: WatermarkConfig) -> Self {
        self.watermark = watermark;
        self
    }

    /// The current watermark configuration.
    #[must_use]
    pub fn watermark(&self) -> &WatermarkConfig {
        &self.watermark
    }

    /// The FR-MD-09 attribution surface: `Some` for attribution-required
    /// weights (GGUF-resolved at load — chunk text or registry fallback),
    /// `None` for permissive weights and synthesized fixtures.
    #[must_use]
    pub fn attribution(&self) -> Option<&AttributionInfo> {
        self.attribution.as_ref()
    }

    /// Overrides the attribution info (fixture flows / loader plumbing).
    #[must_use]
    pub fn with_attribution(mut self, attribution: AttributionInfo) -> Self {
        self.attribution = Some(attribution);
        self
    }

    /// The resolved LM config.
    #[must_use]
    pub fn config(&self) -> &MoshiConfig {
        self.model.config()
    }

    /// The resolved Mimi config (PCM rates).
    #[must_use]
    pub fn mimi_config(&self) -> &MimiNeuralConfig {
        &self.mimi_config
    }

    pub(super) fn model(&self) -> &MoshiModel {
        &self.model
    }

    pub(super) fn encoder(&self) -> &MimiEncoder {
        &self.encoder
    }

    pub(super) fn chain(&self) -> &CsmAudioDecodeChain {
        &self.chain
    }

    pub(super) fn tokenizer(&self) -> &dyn MoshiTextTokenizer {
        self.tokenizer.as_ref()
    }

    /// The T21 AEC posture check shared by both session shapes:
    ///
    /// - default (`aec_disabled_explicitly == false`): the engine **must**
    ///   have an AEC recipe ([`Self::with_aec`]) — otherwise a loud error
    ///   naming both fixes;
    /// - explicit opt-out: no canceller, but a warning is recorded on the
    ///   session (and echoed to stderr) — never silent (FR-EX-08).
    fn duplex_front_for(&self, config: &DuplexSessionConfig) -> Result<DuplexFront> {
        let mut warnings = Vec::new();
        let aec = if config.aec_disabled_explicitly {
            let w = "moshi duplex: echo cancellation EXPLICITLY DISABLED — AEC 無しの \
                     Moshi/CSM は自己エコーで即崩壊 (CLAUDE.md レビュアー C 指摘 #3); \
                     only recorded-file / loopback-free input is safe on this session"
                .to_owned();
            eprintln!("vokra: WARNING {w}");
            warnings.push(w);
            None
        } else {
            let Some(recipe) = &self.aec else {
                return Err(VokraError::InvalidArgument(
                    "moshi duplex: AEC is required but no canceller is wired — either \
                     construct the engine with MoshiEngine::with_aec (interactive \
                     default, FR-OP-60) or opt in per session with \
                     DuplexSessionConfig::with_aec_disabled_explicitly for \
                     recorded-file input (no silent skip — FR-EX-08)"
                        .into(),
                ));
            };
            Some(AecFront::new(&recipe.attrs, recipe.queue_capacity_samples)?)
        };
        Ok((aec, warnings))
    }

    /// Opens an **owning** duplex session (the facade / C ABI shape —
    /// `'static`, movable across threads). See [`Self::duplex_front_for`]
    /// for the AEC posture rules.
    ///
    /// # Errors
    ///
    /// The posture error above; session construction errors verbatim.
    pub fn open_duplex_session(
        self: &Arc<Self>,
        config: &DuplexSessionConfig,
    ) -> Result<MoshiDuplexSession> {
        let (aec, warnings) = self.duplex_front_for(config)?;
        MoshiDuplexSession::new(Arc::clone(self), config, aec, warnings)
    }

    /// Opens a **borrowed** duplex session (the internal batch-`dialog`
    /// shape — same pipeline body, engine lifetime-bound).
    ///
    /// # Errors
    ///
    /// See [`Self::open_duplex_session`].
    pub fn open_duplex_session_borrowed(
        &self,
        config: &DuplexSessionConfig,
    ) -> Result<MoshiDuplexSession<&'_ MoshiEngine>> {
        let (aec, warnings) = self.duplex_front_for(config)?;
        MoshiDuplexSession::new(self, config, aec, warnings)
    }
}

impl S2sDuplexEngine for MoshiEngine {
    fn open_duplex(
        self: Arc<Self>,
        config: &DuplexSessionConfig,
    ) -> Result<Box<dyn S2sDuplexHandle + Send>> {
        Ok(Box::new(self.open_duplex_session(config)?))
    }
}

impl S2sEngine for MoshiEngine {
    /// One turn = the duplex pipeline driven over the whole input
    /// utterance (mic frames pushed in order, model frames pulled as they
    /// emit — the run_inference.py batch loop, which stops at the end of
    /// the audio). [`DialogTurn::text`] carries the decoded inner
    /// monologue.
    fn dialog(self: &MoshiEngine, request: &DialogRequest) -> Result<DialogTurn> {
        if !request.reply_text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "moshi dialog: reply_text must be empty — Moshi GENERATES its own \
                 reply (inner monologue, FR-MD-09); a caller-supplied reply is the \
                 CSM contract (M4-05). Use S2s::dialog(samples) or clear reply_text"
                    .into(),
            ));
        }
        let Some(input) = &request.input_audio else {
            return Err(VokraError::InvalidArgument(
                "moshi dialog: input_audio is required — a duplex model converses \
                 over audio; there is no text-prompted mode (FR-EX-08)"
                    .into(),
            ));
        };
        let hop = self.encoder.frame_hop()?;
        if input.is_empty() || input.len() % hop != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "moshi dialog: input length {} is not a positive multiple of the \
                 frame hop {hop} — buffer whole frames or pad explicitly \
                 (pad_to_whole_frames); no silent zero-pad (FR-EX-08)",
                input.len()
            )));
        }
        let _ = pad_to_whole_frames; // rustdoc-linked helper (same contract as CSM)
        let mut cfg = DuplexSessionConfig::new().with_seed(request.seed);
        if request.deterministic {
            cfg = cfg.deterministic();
        }
        if matches!(self.echo_path, EchoPath::BypassRecordedInput) {
            cfg = cfg.with_aec_disabled_explicitly();
        }
        // Borrowed-engine session (the same pipeline body the trait path
        // owns through Arc — duplex.rs generic parameter).
        let mut session = self.open_duplex_session_borrowed(&cfg)?;
        let n_frames = input.len() / hop;
        let max_frames = request.max_frames.unwrap_or(n_frames).min(n_frames);
        let mut pcm = Vec::with_capacity(max_frames * hop);
        for f in 0..max_frames {
            session.push_mic_frame(&input[f * hop..(f + 1) * hop])?;
            while let Some(frame) = session.pull_model_frame()? {
                pcm.extend_from_slice(&frame);
            }
        }
        let text = session.monologue_text()?;
        Ok(DialogTurn::new(
            text,
            Some(SynthesizedAudio::new(pcm, self.mimi_config.sample_rate)),
        ))
    }
}
