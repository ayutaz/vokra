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
//! ::from_gguf`).
//!
//! ## Mimi codec ends — three postures, no silent downgrade
//!
//! 1. **The model GGUF itself carries the neural chain** (`mimi.enc.*`
//!    tensors present): both codec ends bind those weights for real; any
//!    missing / mis-shaped tensor is a loud load error — never a fall
//!    back to the synthesized bridge.
//! 2. **A standalone Mimi side-car is attached**
//!    ([`MoshiEngine::with_mimi_gguf`], the `vokra-cli run --mimi` flag):
//!    the real kyutai codec from `vokra-cli convert --model mimi` binds,
//!    clipped to `dep_q` codebooks (upstream `get_mimi`
//!    `set_num_codebooks`). The caller asked for the real codec, so
//!    **every** failure (wrong arch, license refusal, hparam mismatch
//!    against the model's `vokra.mimi.*` contract, missing tensor) is a
//!    hard error — the engine never silently keeps the synthesized
//!    bridge.
//! 3. **Neither** → the documented **synthesized bridge** (deterministic
//!    seed-derived codec weights): the pipeline is numerically end-to-end
//!    but its PCM carries no real audio semantics.
//!    [`MoshiEngine::mimi_is_synthesized`] surfaces which posture is
//!    live.
//!
//! The attribution surface resolves at load
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
use crate::codec::MimiCodecGguf;
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

/// How the temporal-backbone blocks bind at load (M4 cc-06).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WeightResidency {
    /// Every weight widened to resident f32 up front (~30 GiB at the
    /// full-7B shape) — the in-memory bytes path.
    Resident,
    /// Temporal blocks stay in the GGUF mapping and widen one layer at a
    /// time per forward (bit-identical values; bounded footprint) — the
    /// `from_path` mmap path.
    MappedLazy,
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
    /// Every weight binds fully **resident** (temporal blocks widened to
    /// f32 up front — ~30 GiB at the full-7B shape). For bounded-memory
    /// loading of large checkpoints from disk use [`Self::from_path`]
    /// (mmap + per-layer materialization).
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] on a wrong arch / missing tensor /
    /// missing tokenizer blob; compliance-gate refusals verbatim;
    /// [`VokraError::InvalidArgument`] on a `0`-placeholder shape config.
    pub fn from_gguf_with_policy(bytes: &[u8], policy: &CompliancePolicy) -> Result<Self> {
        let file = GgufFile::parse(bytes.to_vec())
            .map_err(|e| VokraError::ModelLoad(format!("moshi GGUF: {e}")))?;
        Self::from_gguf_file(file, policy, WeightResidency::Resident)
    }

    /// Loads from a file path with the fail-closed strict policy, through
    /// a true **mmap** ([`vokra_mmap::open_gguf`]) with **mapped-lazy
    /// temporal blocks** — the bounded-memory path that fits the full-7B
    /// `kyutai/moshiko-pytorch-bf16` GGUF on a 16 GB machine:
    ///
    /// - the file is mapped read-only, never copied (the old
    ///   `fs::read` + `parse(to_vec)` double copy is gone);
    /// - the head + depformer weights widen to resident f32 (read every
    ///   step); the 32 temporal blocks — ~86% of the model — stay in the
    ///   mapping and widen **one layer at a time** during each forward
    ///   ([`super::backbone::MappedTemporalBlocks`]), with bit-identical
    ///   f32 values to the resident binding. The trade is per-step decode
    ///   bandwidth for a bounded resident footprint.
    ///
    /// On targets without a real mmap (Emscripten / non-unix-non-windows)
    /// this fails with the mapper's explicit `Unsupported` error — load
    /// through [`Self::from_gguf_with_policy`] there instead (no silent
    /// buffered fallback: the memory contract would silently differ,
    /// FR-EX-08).
    ///
    /// # Errors
    ///
    /// See [`Self::from_gguf_with_policy`]; plus mapping errors verbatim.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_path_with_policy(path, &CompliancePolicy::strict())
    }

    /// [`Self::from_path`] under an explicit compliance policy.
    ///
    /// # Errors
    ///
    /// See [`Self::from_path`].
    pub fn from_path_with_policy(
        path: impl AsRef<std::path::Path>,
        policy: &CompliancePolicy,
    ) -> Result<Self> {
        // `GgufError::Io` (missing / unreadable path) converts to
        // `VokraError::Io` through the shared From impl — the same error
        // class the old `fs::read` path surfaced (no error-type regression
        // for missing-file callers); parse errors become `ModelLoad`.
        let file = vokra_mmap::open_gguf(path.as_ref()).map_err(VokraError::from)?;
        Self::from_gguf_file(file, policy, WeightResidency::MappedLazy)
    }

    /// The shared load body: gate order (arch → M2-13 license →
    /// attribution → config) then weight binding per `residency`.
    fn from_gguf_file(
        file: GgufFile,
        policy: &CompliancePolicy,
        residency: WeightResidency,
    ) -> Result<Self> {
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
        // binding (module docs). The Arc keeps the byte source (owned
        // buffer or mapping) alive; only the MappedLazy block store
        // retains a clone past this function.
        let file = Arc::new(file);
        let depth = super::depth::MoshiDepthWeights::from_gguf(&file, &cfg)?;
        let model = match residency {
            WeightResidency::Resident => {
                let backbone = super::backbone::MoshiBackboneWeights::from_gguf(&file, &cfg)?;
                MoshiModel::new(cfg.clone(), backbone, depth)?
            }
            WeightResidency::MappedLazy => {
                let head = super::backbone::MoshiBackboneWeights::head_from_gguf(&file, &cfg)?;
                let mapped = super::backbone::MappedTemporalBlocks::bind(Arc::clone(&file), &cfg)?;
                let backbone =
                    super::backbone::MoshiBackbone::new_mapped(cfg.clone(), head, mapped)?;
                let depth_t = super::depth::MoshiDepthTransformer::new(cfg.clone(), depth)?;
                MoshiModel::from_parts(backbone, depth_t)?
            }
        };
        let (encoder, chain) = if file.tensor_info("mimi.enc.init").is_some() {
            // Posture 1 (module docs): the model GGUF itself carries the
            // Mimi neural chain — bind it for real. Any inconsistency is
            // loud; there is NO fall back to the synthesized bridge
            // (FR-EX-08 — a half-present codec would decode plausibly
            // wrong PCM).
            let (enc, chain, bound_cfg) =
                Self::bind_real_mimi(&file, &cfg, &mimi_cfg, BackendKind::Cpu)?;
            debug_assert_eq!(bound_cfg, mimi_cfg, "same file ⇒ same mimi config");
            (enc, chain)
        } else {
            // Posture 3: the documented synthesized bridge (module docs).
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
            (encoder, chain)
        };
        let tokenizer: Arc<dyn MoshiTextTokenizer> =
            Arc::new(GgufMoshiTokenizer::from_gguf(&file, cfg.text_card)?);
        let mut engine = Self::new(model, encoder, chain, tokenizer, mimi_cfg)?;
        engine.attribution = attribution;
        Ok(engine)
    }

    /// Binds the real Mimi codec ends (encoder + decode chain) from
    /// `file`, clipped to the model's `dep_q` streams — the shared body of
    /// posture 1 (chain embedded in the model GGUF) and posture 2 (the
    /// [`Self::with_mimi_gguf`] side-car). See the module docs.
    ///
    /// # Clipping (upstream `loaders.py` `get_mimi`)
    ///
    /// The standalone codec carries the full RVQ depth (32 codebooks);
    /// Moshi consumes `dep_q` (8 on the 7B) per direction —
    /// `set_num_codebooks` truncates the residual chain **from the
    /// front** (semantic codebook 0 + the first `dep_q − 1` acoustic
    /// codebooks), which is exactly what binding under a config with
    /// `quantizer.n_q = dep_q` reads (`mimi.enc.cb{0..dep_q}` + the first
    /// `dep_q` effective tables).
    ///
    /// # Decode-path selection (presence-driven — no hidden flag)
    ///
    /// `mimi.dec.feature_proj` present → the raw-table path (the chain
    /// shares the encoder's own codebooks at the quantizer width); absent
    /// → the effective-table path (the converter-derived pre-projected
    /// `vokra.mimi.codebook_tables`, required then). Mirrors
    /// [`MimiNeuralDecoder::from_gguf`]'s own branch.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a codec too shallow for `dep_q`
    /// or on clipped hparams that disagree with the model's
    /// `vokra.mimi.*` contract (`expected_mimi`); [`VokraError::ModelLoad`]
    /// on any missing / mis-shaped tensor (FR-EX-08 — all loud, never a
    /// synthesized fall back).
    fn bind_real_mimi(
        file: &GgufFile,
        cfg: &MoshiConfig,
        expected_mimi: &MimiNeuralConfig,
        backend: BackendKind,
    ) -> Result<(MimiEncoder, CsmAudioDecodeChain, MimiNeuralConfig)> {
        let side = MimiNeuralConfig::from_gguf(file)?;
        side.validate()?;
        let dep_q = cfg.dep_q;
        if side.quantizer.n_q < dep_q {
            return Err(VokraError::InvalidArgument(format!(
                "moshi real-Mimi bind: the codec carries {} codebooks but the model \
                 consumes {dep_q} streams per direction — a shallower RVQ cannot \
                 serve this LM (upstream get_mimi clips the 32-deep chain to dep_q; \
                 FR-EX-08, refusing)",
                side.quantizer.n_q
            )));
        }
        let mut clipped = side;
        clipped.quantizer.n_q = dep_q;
        clipped.validate()?;
        if &clipped != expected_mimi {
            return Err(VokraError::InvalidArgument(format!(
                "moshi real-Mimi bind: codec hparams (clipped to n_q = {dep_q}) do \
                 not match the model GGUF's vokra.mimi.* contract — the LM was \
                 trained against exactly those rates/shapes, so a mismatched codec \
                 would emit plausible-but-wrong codes (FR-EX-08, refusing).\n  \
                 codec = {clipped:?}\n  model = {expected_mimi:?}"
            )));
        }
        let encoder = MimiEncoder::from_gguf(file, &clipped)?.with_backend(backend);
        let neural = MimiNeuralDecoder::from_gguf(file, &clipped)?;
        let feature_dim = neural.expected_feature_dim();
        let attrs = MimiRvqAttrs {
            n_codebooks: dep_q,
            codebook_size: clipped.quantizer.bins,
            d_model: feature_dim,
        };
        let tables = if file.tensor_info("mimi.dec.feature_proj").is_some() {
            // Raw-table path: the chain shares the encoder's codebooks
            // (already exactly `dep_q` of them — the clipped binding).
            encoder.tables().to_vec()
        } else {
            // Effective-table path: the converter-derived pre-projected
            // tables are REQUIRED (their absence next to a projection-less
            // decoder would leave codes → features unbridgeable).
            let codec = MimiCodecGguf::from_gguf(file).map_err(|e| {
                VokraError::ModelLoad(format!(
                    "moshi real-Mimi bind: the decoder has no feature projection \
                     (effective-table path), but the derived codebook tables did \
                     not bind: {e}"
                ))
            })?;
            if codec.attrs.codebook_size != attrs.codebook_size
                || codec.attrs.d_model != feature_dim
                || codec.tables.len() < dep_q
            {
                return Err(VokraError::ModelLoad(format!(
                    "moshi real-Mimi bind: derived codebook tables are \
                     [{} × {} × {}] but the model needs at least {dep_q} tables of \
                     [{} × {feature_dim}] (FR-EX-08)",
                    codec.tables.len(),
                    codec.attrs.codebook_size,
                    codec.attrs.d_model,
                    attrs.codebook_size,
                )));
            }
            codec.tables[..dep_q].to_vec()
        };
        let chain = CsmAudioDecodeChain::new(tables, attrs, neural)?.with_backend(backend);
        Ok((encoder, chain, clipped))
    }

    /// Attaches a **real Mimi codec side-car** (posture 2, module docs): a
    /// standalone Mimi GGUF produced by `vokra-cli convert --model mimi`
    /// from the kyutai tokenizer checkpoint replaces the synthesized codec
    /// bridge on **both** ends (mic encode + model-frame decode), clipped
    /// to `dep_q` codebooks ([`Self::bind_real_mimi`]).
    ///
    /// Runs under the fail-closed strict compliance policy; use
    /// [`Self::with_mimi_gguf_with_policy`] to supply another. The side-car
    /// is CC-BY 4.0 (`AttributionRequired`) — its attribution attaches to
    /// the engine when the engine has none yet (a synthesized-fixture
    /// engine); a model-GGUF attribution already covering Mimi is never
    /// overwritten.
    ///
    /// # Errors
    ///
    /// The caller asked for the real codec, so every failure is a hard
    /// error (wrong `vokra.model.arch`, license refusal, hparam mismatch,
    /// missing tensor) — the engine never silently keeps the synthesized
    /// bridge (FR-EX-08; module docs).
    pub fn with_mimi_gguf(self, path: impl AsRef<std::path::Path>) -> Result<Self> {
        self.with_mimi_gguf_with_policy(path, &CompliancePolicy::strict())
    }

    /// [`Self::with_mimi_gguf`] under an explicit compliance policy.
    ///
    /// # Errors
    ///
    /// See [`Self::with_mimi_gguf`].
    pub fn with_mimi_gguf_with_policy(
        self,
        path: impl AsRef<std::path::Path>,
        policy: &CompliancePolicy,
    ) -> Result<Self> {
        let file = vokra_mmap::open_gguf(path.as_ref()).map_err(VokraError::from)?;
        let arch = file
            .get(vokra_core::gguf::chunks::KEY_MODEL_ARCH)
            .and_then(|v| v.as_str());
        if arch != Some("mimi") {
            return Err(VokraError::ModelLoad(format!(
                "moshi Mimi side-car: not a standalone Mimi codec GGUF — \
                 vokra.model.arch = {arch:?}, expected `mimi` (produce one with \
                 `vokra-cli convert --model mimi --input <tokenizer safetensors> \
                 --output mimi.gguf`)"
            )));
        }
        check_weight_license(&file, policy)?;
        let sidecar_attribution = resolve_attribution(&file);
        let backend = self.chain.backend();
        let (encoder, chain, mimi_cfg) =
            Self::bind_real_mimi(&file, self.model.config(), &self.mimi_config, backend)?;
        let Self {
            model,
            tokenizer,
            aec,
            echo_path,
            watermark,
            attribution,
            ..
        } = self;
        // Rebuild through `new` so every codec seam re-validates; the
        // session knobs (AEC recipe / echo posture / watermark /
        // attribution) carry over — the swap must not reset them.
        let mut engine = Self::new(model, encoder, chain, tokenizer, mimi_cfg)?;
        engine.aec = aec;
        engine.echo_path = echo_path;
        engine.watermark = watermark;
        engine.attribution = attribution.or(sidecar_attribution);
        Ok(engine)
    }

    /// `true` while the Mimi codec ends ride the **synthesized bridge**
    /// (posture 3, module docs): the duplex pipeline is numerically end to
    /// end but its PCM carries no real audio semantics. `false` once the
    /// real codec is bound (embedded chain or [`Self::with_mimi_gguf`]).
    #[must_use]
    pub fn mimi_is_synthesized(&self) -> bool {
        self.encoder.is_synthesized()
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

    /// Routes the LM + **both** Mimi codec ends (encode via `self.encoder`
    /// AND decode via `self.chain`'s neural decoder) through `backend`
    /// (explicit; an unsupported op on the selected backend is a loud error
    /// — FR-EX-08, no silent CPU fallback). The chain's RVQ codebook lookup
    /// stays CPU (cheap array indexing — [`CsmAudioDecodeChain::with_backend`]).
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.model = self.model.with_backend(backend);
        self.encoder = self.encoder.with_backend(backend);
        self.chain = self.chain.with_backend(backend);
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

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Real-Mimi side-car bind (cc-16) — tiny structural side-car GGUFs
    // built through the same pack helpers the mimi round-trip tests pin.
    // -----------------------------------------------------------------

    /// Tiny side-car recipe (defaults = a well-formed raw-table side-car
    /// one codebook DEEPER than `dep_q`, so the clip path always runs).
    struct SidecarSpec {
        /// `quantizer.n_q` of the side-car (`None` → `dep_q + 1`).
        n_q: Option<usize>,
        /// `true` → decoder carries `mimi.dec.feature_proj` (raw-table
        /// path); `false` → effective-table path (+ derived tables).
        with_feature_proj: bool,
        seed: u64,
        /// The provenance stamp (class + raw license string). The default
        /// is the real side-car's CC-BY 4.0; a research-only class proves
        /// the gate wiring. (An **unstamped** `mimi`-arch GGUF resolves
        /// through the registry to the same CC-BY 4.0 — the arch tag is a
        /// known model id — so absence alone cannot exercise fail-closed.)
        license: (vokra_core::LicenseClass, &'static str),
        arch: &'static str,
        /// Override the PCM rate (hparam-mismatch fixture).
        sample_rate: Option<u32>,
        /// Drop every `mimi.dec.*` tensor (missing-tensor fixture).
        skip_decoder: bool,
    }

    impl Default for SidecarSpec {
        fn default() -> Self {
            Self {
                n_q: None,
                with_feature_proj: true,
                seed: 0xC0DE,
                license: (vokra_core::LicenseClass::AttributionRequired, "CC-BY-4.0"),
                arch: "mimi",
                sample_rate: None,
                skip_decoder: false,
            }
        }
    }

    /// Writes the side-car GGUF to a unique temp file; the caller removes
    /// it. Returns the path plus the (un-clipped) side-car config.
    fn build_sidecar(name: &str, spec: &SidecarSpec) -> (std::path::PathBuf, MimiNeuralConfig) {
        use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};
        let moshi_cfg = MoshiConfig::tiny_for_tests();
        let mut mimi_cfg = MimiNeuralConfig::tiny_for_tests();
        mimi_cfg.quantizer.n_q = spec.n_q.unwrap_or(moshi_cfg.dep_q + 1);
        mimi_cfg.quantizer.bins = moshi_cfg.audio_card;
        if let Some(sr) = spec.sample_rate {
            mimi_cfg.sample_rate = sr;
        }
        mimi_cfg.validate().expect("side-car config validates");
        let enc = MimiEncoder::synthesized(&mimi_cfg, spec.seed).expect("side-car encoder");
        let dec =
            MimiNeuralDecoder::synthesized(&mimi_cfg, spec.seed ^ 0x77, spec.with_feature_proj)
                .expect("side-car decoder");
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, spec.arch);
        mimi_cfg.write_gguf_metadata(&mut b);
        let (class, license) = spec.license;
        vokra_core::stamp_provenance(
            &mut b,
            class,
            license,
            Some("mimi"),
            Some("engine-sidecar-test-fixture"),
        );
        crate::mimi::encoder::pack_encoder_structural(&mut b, &enc);
        if !spec.skip_decoder {
            crate::mimi::decoder::pack_decoder_structural(&mut b, &dec);
            if !spec.with_feature_proj {
                // Effective-table trio + derived tables (the standalone
                // converter's shape: [n_q, bins, seanet.dimension]).
                let (n_q, bins, dim) = (
                    mimi_cfg.quantizer.n_q,
                    mimi_cfg.quantizer.bins,
                    mimi_cfg.seanet.dimension,
                );
                b.add_u32("vokra.mimi.n_codebooks", n_q as u32);
                b.add_u32("vokra.mimi.codebook_size", bins as u32);
                b.add_u32("vokra.mimi.d_model", dim as u32);
                let mut rng = vokra_core::rng::SplitMix64::new(spec.seed ^ 0xEFF);
                let bytes: Vec<u8> = (0..n_q * bins * dim)
                    .map(|_| rng.next_unit_f32() * 0.5 - 0.25)
                    .flat_map(f32::to_le_bytes)
                    .collect();
                b.add_tensor(
                    "vokra.mimi.codebook_tables",
                    GgmlType::F32,
                    vec![n_q as u64, bins as u64, dim as u64],
                    bytes,
                )
                .expect("tables tensor");
            }
        }
        let path = std::env::temp_dir().join(format!(
            "vokra-moshi-sidecar-{}-{name}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, b.to_bytes().expect("serialize")).expect("write side-car");
        (path, mimi_cfg)
    }

    /// One deterministic dialog turn over `n` whole frames.
    fn turn(engine: &MoshiEngine, n_frames: usize) -> DialogTurn {
        let hop = engine.mimi_config().frame_hop_samples().expect("hop");
        let input: Vec<f32> = (0..hop * n_frames)
            .map(|i| ((i as f32) * 0.05).sin() * 0.3)
            .collect();
        engine
            .dialog(
                &DialogRequest::new("")
                    .with_input_audio(input)
                    .deterministic(),
            )
            .expect("dialog")
    }

    #[test]
    fn with_mimi_gguf_swaps_both_codec_ends_and_clips_to_dep_q() {
        let (path, side_cfg) = build_sidecar("raw-bind", &SidecarSpec::default());
        let engine = MoshiEngine::synthesized_fixture(55)
            .expect("fixture engine")
            .with_echo_path(EchoPath::BypassRecordedInput);
        assert!(engine.mimi_is_synthesized(), "fixture starts synthesized");
        let dep_q = engine.config().dep_q;
        assert_eq!(side_cfg.quantizer.n_q, dep_q + 1, "side-car is deeper");
        let before = turn(&engine, 4);

        let engine = engine.with_mimi_gguf(&path).expect("side-car binds");
        let _ = std::fs::remove_file(&path);
        assert!(!engine.mimi_is_synthesized(), "real codec is live");
        assert_eq!(
            engine.mimi_config().quantizer.n_q,
            dep_q,
            "codec clipped to dep_q (upstream set_num_codebooks)"
        );
        // Echo posture carried over: the recorded-input bypass still lets
        // `dialog` run without an AEC recipe.
        let after = turn(&engine, 4);
        let (a, b) = (before.audio.expect("pcm"), after.audio.expect("pcm"));
        assert_eq!(a.samples.len(), b.samples.len());
        assert!(b.samples.iter().all(|v| v.is_finite()));
        assert_ne!(
            a.samples, b.samples,
            "swapping the codec bridge must change the decoded PCM"
        );
        // The side-car attribution attached (the fixture engine had none).
        let attribution = engine
            .attribution()
            .expect("CC-BY 4.0 side-car attribution");
        assert!(
            attribution.text.contains("mimi"),
            "attribution names the codec: {}",
            attribution.text
        );
    }

    #[test]
    fn with_mimi_gguf_binds_the_effective_table_path() {
        let (path, _) = build_sidecar(
            "effective-bind",
            &SidecarSpec {
                with_feature_proj: false,
                ..SidecarSpec::default()
            },
        );
        let engine = MoshiEngine::synthesized_fixture(7)
            .expect("fixture engine")
            .with_echo_path(EchoPath::BypassRecordedInput)
            .with_mimi_gguf(&path)
            .expect("effective-table side-car binds");
        let _ = std::fs::remove_file(&path);
        assert!(!engine.mimi_is_synthesized());
        let audio = turn(&engine, 3).audio.expect("pcm");
        assert!(audio.samples.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn with_mimi_gguf_refuses_a_codec_shallower_than_dep_q() {
        let dep_q = MoshiConfig::tiny_for_tests().dep_q;
        let (path, _) = build_sidecar(
            "shallow",
            &SidecarSpec {
                n_q: Some(dep_q - 1),
                ..SidecarSpec::default()
            },
        );
        let err = MoshiEngine::synthesized_fixture(1)
            .expect("fixture engine")
            .with_mimi_gguf(&path)
            .expect_err("shallow codec must refuse");
        let _ = std::fs::remove_file(&path);
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err}");
        assert!(err.to_string().contains("codebooks"), "{err}");
    }

    #[test]
    fn with_mimi_gguf_refuses_mismatched_codec_hparams() {
        // Same structural geometry, different PCM rate: the LM's
        // vokra.mimi.* contract (16 kHz tiny fixture) must win — a
        // mismatched codec is refused, not resampled silently.
        let (path, _) = build_sidecar(
            "hparam-mismatch",
            &SidecarSpec {
                sample_rate: Some(32_000),
                ..SidecarSpec::default()
            },
        );
        let err = MoshiEngine::synthesized_fixture(2)
            .expect("fixture engine")
            .with_mimi_gguf(&path)
            .expect_err("hparam mismatch must refuse");
        let _ = std::fs::remove_file(&path);
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err}");
        assert!(err.to_string().contains("vokra.mimi.* contract"), "{err}");
    }

    #[test]
    fn with_mimi_gguf_missing_decoder_tensor_is_loud() {
        let (path, _) = build_sidecar(
            "missing-dec",
            &SidecarSpec {
                skip_decoder: true,
                ..SidecarSpec::default()
            },
        );
        let err = MoshiEngine::synthesized_fixture(3)
            .expect("fixture engine")
            .with_mimi_gguf(&path)
            .expect_err("half a codec must refuse");
        let _ = std::fs::remove_file(&path);
        assert!(matches!(err, VokraError::ModelLoad(_)), "{err}");
        assert!(err.to_string().contains("mimi.dec."), "{err}");
    }

    #[test]
    fn with_mimi_gguf_rejects_a_non_mimi_arch() {
        let (path, _) = build_sidecar(
            "wrong-arch",
            &SidecarSpec {
                arch: "dac",
                ..SidecarSpec::default()
            },
        );
        let err = MoshiEngine::synthesized_fixture(4)
            .expect("fixture engine")
            .with_mimi_gguf(&path)
            .expect_err("non-mimi side-car must refuse");
        let _ = std::fs::remove_file(&path);
        assert!(matches!(err, VokraError::ModelLoad(_)), "{err}");
        assert!(err.to_string().contains("expected `mimi`"), "{err}");
    }

    #[test]
    fn with_mimi_gguf_research_only_weight_fails_closed_under_strict() {
        // Proves the M2-13 gate actually runs on the side-car: a
        // non-commercial-stamped codec refuses under the strict policy.
        let (path, _) = build_sidecar(
            "nc-license",
            &SidecarSpec {
                license: (vokra_core::LicenseClass::NonCommercial, "CC-BY-NC-4.0"),
                ..SidecarSpec::default()
            },
        );
        let err = MoshiEngine::synthesized_fixture(5)
            .expect("fixture engine")
            .with_mimi_gguf(&path)
            .expect_err("research-only weight must fail closed");
        let _ = std::fs::remove_file(&path);
        assert!(
            matches!(err, VokraError::ResearchLicenseRequired { .. }),
            "{err}"
        );
    }

    #[test]
    fn with_backend_routes_the_decode_chain_not_only_lm_and_encoder() {
        // #12 regression: the rustdoc promises the codec (encode AND decode)
        // follows with_backend, but before the fix `self.chain` (the Mimi
        // neural decoder) stayed on CPU while `model` + `encoder` moved. All
        // three must route now — otherwise a Metal/CUDA deployment would
        // silently decode PCM on the CPU (defeating the GPU selection, and
        // hiding a coverage gap that FR-EX-08 requires be loud).
        let engine = MoshiEngine::synthesized_fixture(3).expect("fixture engine");
        assert_eq!(
            engine.chain().backend(),
            BackendKind::Cpu,
            "assembled engine's decode chain is CPU"
        );
        let engine = engine.with_backend(BackendKind::Cuda);
        assert_eq!(
            engine.chain().backend(),
            BackendKind::Cuda,
            "with_backend must route self.chain (the decode half of the codec)"
        );
    }
}
