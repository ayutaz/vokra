//! Source-shaped VoxCPM generation primitives.

use vokra_core::{Result, VokraError};

use crate::compute::Compute;
use crate::strict_checkpoint::load_tensor;
use crate::voxcpm2::{
    LocalDit, LocalEncoder, MiniCpm4BlockWeights, MiniCpm4Config, MiniCpm4KvCache, MiniCpm4Stack,
    MiniCpm4StackWeights, UnifiedCfm,
};
use vokra_core::gguf::GgufFile;

/// VoxCPM emits two 64-wide feature rows for every generated LM step.
pub const FEATURE_PATCHES_PER_STEP: usize = 2;
const FEATURE_PATCH_WIDTH: usize = FEATURE_PATCHES_PER_STEP * 64;

#[allow(dead_code)] // Used only by the staged GGUF generation path.
const VOXCPM_HIDDEN: usize = 1_024;
#[allow(dead_code)] // Used only by the staged GGUF generation path.
const VOXCPM_FFN: usize = 4_096;
#[allow(dead_code)] // Used only by the staged GGUF generation path.
const VOXCPM_KV: usize = 128;
#[allow(dead_code)] // Used only by the staged GGUF generation path.
const VOXCPM_VOCAB: usize = 73_448;

/// Source-shaped feature-generation loop. VoxCPM pre-fills the text prompt
/// into its base LM and then autoregressively emits a 2×64 feature patch per
/// step. The callbacks own authenticated LM/DiT weights; this type fixes the
/// source protocol and refuses malformed learned outputs.
#[derive(Debug, Clone, Copy)]
pub struct FeatureGenerationLoop {
    /// Maximum number of feature patches to emit.
    pub max_steps: usize,
    /// Minimum number of patches before the learned stop decision applies.
    pub min_steps: usize,
    /// Base/residual MiniCPM hidden width (0.5B: 1024).
    pub hidden_dim: usize,
    /// Continuous feature width emitted per codebook (0.5B: 64).
    pub feature_dim: usize,
}

/// The two streams assembled before the causal LMs are prefetched.
///
/// `text_embeddings_raw` and `audio_features` are parallel row buffers. A false
/// `audio_mask` row is a text row and is scaled by the caller's effective
/// `scale_emb` (the fixed 0.5B non-µP path passes `1.0`); a true row is
/// replaced by `encode_audio(audio_features[row])`, where each feature row is
/// the complete `[P=2,D=64]` patch (128 values), not one 64-wide codebook.
/// Keeping both buffers explicit prevents an audio feature from accidentally being fed through the
/// text embedding path.  The residual stream is the source's elementwise
/// `enc_outputs + audio_mask * feat_embed` operation.
#[derive(Debug, Clone)]
pub struct PrefillState {
    base_rows: Vec<f32>,
    audio_embeddings: Vec<f32>,
    audio_mask: Vec<bool>,
    rows: usize,
    hidden_dim: usize,
}

impl PrefillState {
    /// Number of text/audio rows in the prefill packet.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Hidden width of each prefill row.
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Base-LM input rows in row-major `[rows, hidden_dim]` layout.
    #[must_use]
    pub fn base_rows(&self) -> &[f32] {
        &self.base_rows
    }

    /// Apply the source's masked prefill FSQ to raw base-LM outputs.  Text
    /// rows remain raw while audio rows alone take the learned FSQ route.
    pub fn encoded_outputs(
        &self,
        raw_outputs: &[f32],
        quantizer: &ScalarQuantizer,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        if raw_outputs.len() != self.base_rows.len() || raw_outputs.iter().any(|x| !x.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "voxcpm prefill output shape/finiteness mismatch".to_owned(),
            ));
        }
        let mut encoded = raw_outputs.to_vec();
        for row in 0..self.rows {
            if self.audio_mask[row] {
                let start = row * self.hidden_dim;
                let fsq = quantizer.apply(&raw_outputs[start..start + self.hidden_dim], compute)?;
                encoded[start..start + self.hidden_dim].copy_from_slice(&fsq);
            }
        }
        Ok(encoded)
    }

    /// Build residual-LM prefill from already masked `enc_outputs`.  This is
    /// the source operation `enc_outputs + audio_mask * feat_embed`; it must
    /// happen after the causal base prefill and masked FSQ.
    pub fn residual_rows_from_encoded(&self, enc_outputs: &[f32]) -> Result<Vec<f32>> {
        if enc_outputs.len() != self.base_rows.len() || enc_outputs.iter().any(|x| !x.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "voxcpm residual prefill output shape/finiteness mismatch".to_owned(),
            ));
        }
        let mut residual = enc_outputs.to_vec();
        for row in 0..self.rows {
            if self.audio_mask[row] {
                for channel in 0..self.hidden_dim {
                    residual[row * self.hidden_dim + channel] +=
                        self.audio_embeddings[row * self.hidden_dim + channel];
                }
            }
        }
        Ok(residual)
    }
}

/// Persistent base/residual causal-LM state for the source feature loop.
/// Both stacks retain their own KV cache across feature patches; rebuilding a
/// cache for every patch would change the conditioning protocol and memory
/// position used by RoPE.
#[derive(Debug, Clone)]
pub struct CausalLanguageState {
    base: MiniCpm4Stack,
    residual: MiniCpm4Stack,
    base_cache: MiniCpm4KvCache,
    residual_cache: MiniCpm4KvCache,
    lm_hidden: Vec<f32>,
    residual_hidden: Vec<f32>,
}

/// A validated source-layout projection.  `MiniCpm4Linear` transposes the
/// `[out,in]` checkpoint matrix once at construction, so the selected Compute
/// backend is used for every hot projection without per-step transposes.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Bound only when the complete staged composite is authorized.
pub(crate) struct FeatureProjection {
    linear: crate::voxcpm2::MiniCpm4Linear,
}

impl FeatureProjection {
    #[allow(dead_code)] // Staged projection constructor awaits composite authorization.
    pub(crate) fn from_source(
        weight: Vec<f32>,
        bias: Vec<f32>,
        out_features: usize,
        in_features: usize,
    ) -> Result<Self> {
        Ok(Self {
            linear: crate::voxcpm2::MiniCpm4Linear::from_source(
                weight,
                Some(bias),
                out_features,
                in_features,
            )?,
        })
    }

    #[allow(dead_code)] // Staged projection is dormant until its composite route is authorized.
    pub(crate) fn apply(&self, input: &[f32], rows: usize, compute: &Compute) -> Result<Vec<f32>> {
        let mut output = vec![0.0; rows * self.linear.out_features()];
        self.linear.apply(compute, input, rows, &mut output)?;
        Ok(output)
    }
}

/// Source-shaped staged VoxCPM learned path.  This is crate-private on
/// purpose: it binds exact tensor names and axes for VAST-prepared inputs but
/// does not assert the missing complete composite manifest, so public
/// production loaders remain fail-closed.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Staged runtime awaits the authenticated complete composite manifest.
pub(crate) struct StagedGenerationRuntime {
    language: CausalLanguageState,
    embedding: Vec<f32>,
    enc_to_lm: FeatureProjection,
    lm_to_dit: FeatureProjection,
    res_to_dit: FeatureProjection,
    quantizer: ScalarQuantizer,
    stop: LearnedStopController,
}

/// Caller-owned deterministic flow draws. One draw is consumed for each
/// generated LM step and contains the channel-major `[64,2]` initial state.
/// No RNG, temperature default, or hidden fallback is kept by the runtime.
#[derive(Debug, Clone)]
pub struct VoxCpm2FlowDraws {
    noises: Vec<Vec<f32>>,
}

impl VoxCpm2FlowDraws {
    /// Construct caller-owned finite draws for exactly `required_steps` steps.
    pub fn new(noises: Vec<Vec<f32>>, required_steps: usize) -> Result<Self> {
        if required_steps == 0
            || noises.len() != required_steps
            || noises.iter().any(|noise| {
                noise.len() != FEATURE_PATCH_WIDTH || noise.iter().any(|x| !x.is_finite())
            })
        {
            return Err(VokraError::InvalidArgument(
                "voxcpm flow draws require exactly one finite channel-major [64,2] row per max step".to_owned(),
            ));
        }
        Ok(Self { noises })
    }

    #[allow(dead_code)] // Consumed only by the dormant staged batch-one route.
    fn get(&self, step: usize) -> Result<&[f32]> {
        self.noises.get(step).map(Vec::as_slice).ok_or_else(|| {
            VokraError::InvalidArgument("voxcpm flow draw packet is exhausted".to_owned())
        })
    }

    /// Number of validated draws available.
    #[must_use]
    pub fn len(&self) -> usize {
        self.noises.len()
    }

    /// Whether no flow draws are available.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.noises.is_empty()
    }
}

#[allow(dead_code)] // All methods bind the dormant staged composite route.
impl StagedGenerationRuntime {
    pub(crate) fn from_staged_gguf(file: &GgufFile) -> Result<Self> {
        let base = load_causal_stack(file, "base_lm", 24)?;
        let residual = load_causal_stack(file, "residual_lm", 6)?;
        let language = CausalLanguageState::from_stacks(base, residual)?;
        let embedding = load_tensor(
            file,
            "voxcpm2",
            "base_lm.embed_tokens.weight",
            &[VOXCPM_VOCAB, VOXCPM_HIDDEN],
        )?;
        Ok(Self {
            language,
            embedding,
            enc_to_lm: FeatureProjection::from_source(
                load_tensor(
                    file,
                    "voxcpm2",
                    "enc_to_lm_proj.weight",
                    &[VOXCPM_HIDDEN, VOXCPM_HIDDEN],
                )?,
                load_tensor(file, "voxcpm2", "enc_to_lm_proj.bias", &[VOXCPM_HIDDEN])?,
                VOXCPM_HIDDEN,
                VOXCPM_HIDDEN,
            )?,
            lm_to_dit: FeatureProjection::from_source(
                load_tensor(
                    file,
                    "voxcpm2",
                    "lm_to_dit_proj.weight",
                    &[VOXCPM_HIDDEN, VOXCPM_HIDDEN],
                )?,
                load_tensor(file, "voxcpm2", "lm_to_dit_proj.bias", &[VOXCPM_HIDDEN])?,
                VOXCPM_HIDDEN,
                VOXCPM_HIDDEN,
            )?,
            res_to_dit: FeatureProjection::from_source(
                load_tensor(
                    file,
                    "voxcpm2",
                    "res_to_dit_proj.weight",
                    &[VOXCPM_HIDDEN, VOXCPM_HIDDEN],
                )?,
                load_tensor(file, "voxcpm2", "res_to_dit_proj.bias", &[VOXCPM_HIDDEN])?,
                VOXCPM_HIDDEN,
                VOXCPM_HIDDEN,
            )?,
            quantizer: ScalarQuantizer::from_staged_gguf(file)?,
            stop: LearnedStopController::from_staged_gguf(file)?,
        })
    }

    /// Raw token embedding lookup. Callers pass this result to
    /// [`FeatureGenerationLoop::assemble_prefill`], which applies the
    /// effective source embedding scale. For the fixed 0.5B checkpoint
    /// `use_mup=false`, so that scale is `1.0`.
    pub(crate) fn embed_tokens_raw(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(VokraError::InvalidArgument(
                "voxcpm staged embedding lookup requires tokens".to_owned(),
            ));
        }
        let mut rows = vec![0.0; tokens.len() * VOXCPM_HIDDEN];
        for (row, &token) in tokens.iter().enumerate() {
            let token = usize::try_from(token).map_err(|_| {
                VokraError::InvalidArgument("voxcpm token id conversion overflow".to_owned())
            })?;
            if token >= VOXCPM_VOCAB {
                return Err(VokraError::InvalidArgument(
                    "voxcpm token id exceeds the authenticated vocabulary".to_owned(),
                ));
            }
            for channel in 0..VOXCPM_HIDDEN {
                rows[row * VOXCPM_HIDDEN + channel] =
                    self.embedding[token * VOXCPM_HIDDEN + channel];
            }
        }
        Ok(rows)
    }

    pub(crate) fn enc_to_lm(
        &self,
        encoder_rows: &[f32],
        rows: usize,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        self.enc_to_lm.apply(encoder_rows, rows, compute)
    }

    pub(crate) fn dit_condition(
        &self,
        lm_hidden: &[f32],
        residual_hidden: &[f32],
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        if lm_hidden.len() != VOXCPM_HIDDEN || residual_hidden.len() != VOXCPM_HIDDEN {
            return Err(VokraError::InvalidArgument(
                "voxcpm DiT condition hidden rows must be [1024]".to_owned(),
            ));
        }
        let lm = self.lm_to_dit.apply(lm_hidden, 1, compute)?;
        let residual = self.res_to_dit.apply(residual_hidden, 1, compute)?;
        Ok(lm
            .into_iter()
            .zip(residual)
            .map(|(left, right)| left + right)
            .collect())
    }

    pub(crate) fn quantize(&self, hidden: &[f32], compute: &Compute) -> Result<Vec<f32>> {
        self.quantizer.apply(hidden, compute)
    }

    pub(crate) fn should_stop(&self, hidden: &[f32], compute: &Compute) -> Result<bool> {
        self.stop.should_stop(hidden, compute)
    }

    pub(crate) fn language_mut(&mut self) -> &mut CausalLanguageState {
        &mut self.language
    }

    pub(crate) fn prefill(&mut self, rows: &PrefillState, compute: &Compute) -> Result<()> {
        self.language.prefill(rows, &self.quantizer, compute)
    }

    pub(crate) fn lm_hidden(&self) -> &[f32] {
        self.language.lm_hidden()
    }

    pub(crate) fn residual_hidden(&self) -> &[f32] {
        self.language.residual_hidden()
    }

    /// Run the complete staged batch=1 feature protocol. The caller supplies
    /// the already-authenticated prefill state and flow draws; the source CFG
    /// negative half repeats the current prefix and uses zero mu. This method
    /// never fabricates tokenizer rows or noise. Public production construction remains blocked by the
    /// complete composite manifest gate.
    #[allow(clippy::too_many_arguments)] // Batch-one wiring has one argument per authenticated component.
    pub(crate) fn generate_batch1(
        &mut self,
        loop_: &FeatureGenerationLoop,
        seed_prefix: &[f32],
        local_encoder: &LocalEncoder,
        local_dit: &LocalDit,
        flow: &UnifiedCfm,
        draws: &VoxCpm2FlowDraws,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        if seed_prefix.len() != FEATURE_PATCH_WIDTH
            || seed_prefix.iter().any(|value| !value.is_finite())
            || loop_.feature_dim != 64
            || draws.len() != loop_.max_steps
            || flow.sway_coefficient() != 1.0
            || flow.cfg_scale() != 2.0
        {
            return Err(VokraError::InvalidArgument(
                "voxcpm batch-1 flow inputs have invalid shape or draws".to_owned(),
            ));
        }
        let local = local_encoder;
        let dit = local_dit;
        let quantizer = &self.quantizer;
        let stop = &self.stop;
        let lm_to_dit = &self.lm_to_dit;
        let res_to_dit = &self.res_to_dit;
        loop_.generate_with_language(
            seed_prefix,
            &mut self.language,
            quantizer,
            compute,
            |step, lm_hidden, residual_hidden, prefix| {
                let draw = draws.get(step)?;
                let condition = cfg_condition(prefix)?;
                let mut positive_mu = vec![0.0; VOXCPM_HIDDEN];
                let lm = lm_to_dit.apply(lm_hidden, 1, compute)?;
                let residual = res_to_dit.apply(residual_hidden, 1, compute)?;
                for (dst, (left, right)) in positive_mu.iter_mut().zip(lm.into_iter().zip(residual))
                {
                    *dst = left + right;
                }
                let positive = |time: f32, state: &[f32]| {
                    dit.forward(state, 2, &condition, 2, time, 0.0, &positive_mu, compute)
                };
                // UnifiedCFM split-CFG repeats the current dynamic prefix in
                // both halves and zeros only mu in the negative half. The
                // condition is rebuilt for every generated patch.
                let zero_mu = cfg_negative_mu();
                let negative = |time: f32, state: &[f32]| {
                    dit.forward(state, 2, &condition, 2, time, 0.0, &zero_mu, compute)
                };
                let channel_major = flow.integrate(draw, positive, negative)?;
                channel_major_to_row_major(&channel_major, 64, 2)
            },
            |patch| local.forward(patch, 1, 2, compute),
            |hidden| stop.should_stop(hidden, compute),
        )
    }
}

impl CausalLanguageState {
    #[allow(dead_code)] // Used for atomic rollback by the dormant staged route.
    fn snapshot(&self) -> LanguageStateSnapshot {
        LanguageStateSnapshot {
            base_cache: self.base_cache.checkpoint(),
            residual_cache: self.residual_cache.checkpoint(),
            lm_hidden: self.lm_hidden.clone(),
            residual_hidden: self.residual_hidden.clone(),
        }
    }

    #[allow(dead_code)] // Used for atomic rollback by the dormant staged route.
    fn restore_snapshot(&mut self, snapshot: &LanguageStateSnapshot) {
        self.base_cache.restore(&snapshot.base_cache);
        self.residual_cache.restore(&snapshot.residual_cache);
        self.lm_hidden.copy_from_slice(&snapshot.lm_hidden);
        self.residual_hidden
            .copy_from_slice(&snapshot.residual_hidden);
    }

    /// Attach the source's 24-layer base and 6-layer residual stacks.
    pub fn from_stacks(base: MiniCpm4Stack, residual: MiniCpm4Stack) -> Result<Self> {
        if base.layer_count() != 24
            || residual.layer_count() != 6
            || base.config().hidden_dim() != 1_024
            || residual.config().hidden_dim() != 1_024
        {
            return Err(VokraError::InvalidArgument(
                "voxcpm causal stacks must be 24/6 layers at hidden width 1024".to_owned(),
            ));
        }
        Ok(Self {
            base_cache: MiniCpm4KvCache::new(&base),
            residual_cache: MiniCpm4KvCache::new(&residual),
            base,
            residual,
            lm_hidden: vec![0.0; 1_024],
            residual_hidden: vec![0.0; 1_024],
        })
    }

    /// Prefill both persistent caches.  The residual rows are derived from
    /// the base-LM output and audio embedding mask, matching the upstream
    /// `enc_outputs + audio_mask * feat_embed` expression.
    pub fn prefill(
        &mut self,
        rows: &PrefillState,
        quantizer: &ScalarQuantizer,
        compute: &Compute,
    ) -> Result<()> {
        // Build both caches off to the side.  Existing state is not touched
        // until both full prefills and their final hidden rows succeed.
        let mut base_cache = MiniCpm4KvCache::new(&self.base);
        let base_outputs = self.base.forward_cached(
            rows.base_rows(),
            rows.rows(),
            true,
            &mut base_cache,
            compute,
        )?;
        let enc_outputs = rows.encoded_outputs(&base_outputs, quantizer, compute)?;
        let residual_rows = rows.residual_rows_from_encoded(&enc_outputs)?;
        let mut residual_cache = MiniCpm4KvCache::new(&self.residual);
        let residual_outputs = self.residual.forward_cached(
            &residual_rows,
            rows.rows(),
            true,
            &mut residual_cache,
            compute,
        )?;
        if base_outputs.len() < 1_024 || residual_outputs.len() < 1_024 {
            return Err(VokraError::ModelLoad(
                "voxcpm causal prefill returned no final hidden row".to_owned(),
            ));
        }
        self.lm_hidden
            .copy_from_slice(&enc_outputs[enc_outputs.len() - 1_024..]);
        self.residual_hidden
            .copy_from_slice(&residual_outputs[residual_outputs.len() - 1_024..]);
        self.base_cache = base_cache;
        self.residual_cache = residual_cache;
        Ok(())
    }

    /// Advance both LM caches for one generated patch.  The base state is
    /// advanced by `curr_embed`; the residual state receives
    /// `FSQ(lm_hidden) + curr_embed` as an elementwise sum.
    pub fn step(
        &mut self,
        curr_embed: &[f32],
        quantizer: &ScalarQuantizer,
        compute: &Compute,
    ) -> Result<()> {
        if curr_embed.len() != 1_024 || curr_embed.iter().any(|x| !x.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "voxcpm LM step embedding must be finite [1024]".to_owned(),
            ));
        }
        let base_checkpoint = self.base_cache.checkpoint();
        let residual_checkpoint = self.residual_cache.checkpoint();
        let lm_hidden = match self
            .base
            .forward_step(curr_embed, &mut self.base_cache, compute)
        {
            Ok(hidden) => hidden,
            Err(error) => {
                self.base_cache.restore(&base_checkpoint);
                self.residual_cache.restore(&residual_checkpoint);
                return Err(error);
            }
        };
        let fsq_hidden = match quantizer.apply(&lm_hidden, compute) {
            Ok(hidden) => hidden,
            Err(error) => {
                self.base_cache.restore(&base_checkpoint);
                self.residual_cache.restore(&residual_checkpoint);
                return Err(error);
            }
        };
        let residual_input: Vec<f32> = fsq_hidden
            .iter()
            .zip(curr_embed)
            .map(|(fsq, current)| fsq + current)
            .collect();
        let residual_hidden =
            match self
                .residual
                .forward_step(&residual_input, &mut self.residual_cache, compute)
            {
                Ok(hidden) => hidden,
                Err(error) => {
                    self.base_cache.restore(&base_checkpoint);
                    self.residual_cache.restore(&residual_checkpoint);
                    return Err(error);
                }
            };
        self.lm_hidden.copy_from_slice(&fsq_hidden);
        self.residual_hidden.copy_from_slice(&residual_hidden);
        Ok(())
    }

    #[must_use]
    /// Current base-LM hidden row.
    pub fn lm_hidden(&self) -> &[f32] {
        &self.lm_hidden
    }

    #[must_use]
    /// Current residual-LM hidden row.
    pub fn residual_hidden(&self) -> &[f32] {
        &self.residual_hidden
    }

    #[must_use]
    /// Number of cached positions in the base LM.
    pub fn base_cache_positions(&self) -> usize {
        self.base_cache.positions()
    }

    #[must_use]
    /// Number of cached positions in the residual LM.
    pub fn residual_cache_positions(&self) -> usize {
        self.residual_cache.positions()
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // Snapshot exists only for dormant staged-route rollback.
struct LanguageStateSnapshot {
    base_cache: crate::voxcpm2::minicpm4::CacheCheckpoint,
    residual_cache: crate::voxcpm2::minicpm4::CacheCheckpoint,
    lm_hidden: Vec<f32>,
    residual_hidden: Vec<f32>,
}

impl FeatureGenerationLoop {
    /// Construct a validated feature-generation loop.
    pub fn new(
        max_steps: usize,
        min_steps: usize,
        hidden_dim: usize,
        feature_dim: usize,
    ) -> Result<Self> {
        if max_steps == 0 || min_steps > max_steps || hidden_dim == 0 || feature_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm feature loop has invalid step/latent dimensions".to_owned(),
            ));
        }
        Ok(Self {
            max_steps,
            min_steps,
            hidden_dim,
            feature_dim,
        })
    }

    /// Assemble the exact text/audio prefill rows used by VoxCPM.
    ///
    /// Both input arrays have `rows` rows.  Text rows contain a hidden-sized
    /// token embedding and audio rows contain a
    /// `FEATURE_PATCHES_PER_STEP * feature_dim`-sized feature patch. Only rows selected by `audio_mask` call the supplied learned
    /// encoder.  The returned buffers are ready for the persistent base and
    /// residual [`MiniCpm4KvCache`] prefill calls; this helper itself does not
    /// run a language model.
    pub fn assemble_prefill<E>(
        &self,
        text_embeddings_raw: &[f32],
        audio_features: &[f32],
        audio_mask: &[bool],
        scale_emb: f32,
        mut encode_audio: E,
    ) -> Result<PrefillState>
    where
        E: FnMut(&[f32]) -> Result<Vec<f32>>,
    {
        if !scale_emb.is_finite()
            || text_embeddings_raw.len() % self.hidden_dim != 0
            || audio_features.len() % (self.feature_dim * FEATURE_PATCHES_PER_STEP) != 0
            || text_embeddings_raw.len() / self.hidden_dim
                != audio_features.len() / (self.feature_dim * FEATURE_PATCHES_PER_STEP)
            || audio_mask.len() != text_embeddings_raw.len() / self.hidden_dim
        {
            return Err(VokraError::InvalidArgument(
                "voxcpm prefill row/scale dimensions mismatch".to_owned(),
            ));
        }
        let rows = audio_mask.len();
        if rows == 0
            || text_embeddings_raw.iter().any(|x| !x.is_finite())
            || audio_features.iter().any(|x| !x.is_finite())
        {
            return Err(VokraError::InvalidArgument(
                "voxcpm prefill inputs must be finite and non-empty".to_owned(),
            ));
        }
        let mut base_rows = vec![0.0; rows * self.hidden_dim];
        let mut audio_embeddings = vec![0.0; rows * self.hidden_dim];
        for row in 0..rows {
            let text = &text_embeddings_raw[row * self.hidden_dim..(row + 1) * self.hidden_dim];
            let feature = &audio_features[row * self.feature_dim * FEATURE_PATCHES_PER_STEP
                ..(row + 1) * self.feature_dim * FEATURE_PATCHES_PER_STEP];
            let base = if audio_mask[row] {
                let encoded = encode_audio(feature)?;
                if encoded.len() != self.hidden_dim || encoded.iter().any(|x| !x.is_finite()) {
                    return Err(VokraError::InvalidArgument(
                        "voxcpm audio prefill encoder returned an invalid hidden row".to_owned(),
                    ));
                }
                encoded
            } else {
                text.iter().map(|x| x * scale_emb).collect()
            };
            base_rows[row * self.hidden_dim..(row + 1) * self.hidden_dim].copy_from_slice(&base);
            if audio_mask[row] {
                for channel in 0..self.hidden_dim {
                    audio_embeddings[row * self.hidden_dim + channel] = base[channel];
                }
            }
        }
        Ok(PrefillState {
            base_rows,
            audio_embeddings,
            audio_mask: audio_mask.to_vec(),
            rows,
            hidden_dim: self.hidden_dim,
        })
    }

    /// Generate feature patches and return them row-major `[steps, 2*64]`.
    /// `seed` is `[lm_hidden(1024), residual_hidden(1024), prefix_feat(2,64)]`
    /// after the text prompt has been prefixed into the base LM. `cfm` is the
    /// local DiT/UnifiedCFM patch sampler, `local_encoder` plus `base_lm`
    /// advance the base-LM KV state, `quantize` is learned FSQ, `residual`
    /// advances the residual-LM KV state, and `stop` applies
    /// `SiLU(stop_proj(lm_hidden)).argmax` before either state is updated.
    #[allow(clippy::too_many_arguments)] // The source protocol exposes one callback per graph stage.
    pub fn generate<C, E, B, Q, R, S>(
        &self,
        seed: &[f32],
        mut cfm: C,
        mut local_encoder: E,
        mut base_lm: B,
        mut quantize: Q,
        mut residual: R,
        mut stop: S,
    ) -> Result<Vec<f32>>
    where
        C: FnMut(usize, &[f32], &[f32], &[f32]) -> Result<Vec<f32>>,
        E: FnMut(&[f32]) -> Result<Vec<f32>>,
        B: FnMut(&[f32]) -> Result<Vec<f32>>,
        Q: FnMut(&[f32]) -> Result<Vec<f32>>,
        R: FnMut(&[f32]) -> Result<Vec<f32>>,
        S: FnMut(&[f32]) -> Result<bool>,
    {
        let prefix_len = self.feature_dim * FEATURE_PATCHES_PER_STEP;
        let state_len = self.hidden_dim * 2 + prefix_len;
        if seed.len() != state_len || seed.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "voxcpm feature loop seed shape/finiteness mismatch".to_owned(),
            ));
        }
        let mut lm_hidden = seed[..self.hidden_dim].to_vec();
        let mut residual_hidden = seed[self.hidden_dim..self.hidden_dim * 2].to_vec();
        let mut prefix = seed[self.hidden_dim * 2..].to_vec();
        let mut patches = Vec::with_capacity(self.max_steps * prefix_len);
        for step in 0..self.max_steps {
            let patch = cfm(step, &lm_hidden, &residual_hidden, &prefix)?;
            if patch.len() != prefix_len || patch.iter().any(|value| !value.is_finite()) {
                return Err(VokraError::InvalidArgument(
                    "voxcpm feature loop CFM patch must be finite and have shape [2, latent_dim]"
                        .to_owned(),
                ));
            }
            // The patch belongs to the output even if its stop prediction is
            // true.  The official loop appends first, then evaluates the
            // previous LM hidden with the 0-based `i > min_len` condition.
            patches.extend_from_slice(&patch);
            prefix.copy_from_slice(&patch);
            let curr_embed = local_encoder(&patch)?;
            if curr_embed.len() != self.hidden_dim
                || curr_embed.iter().any(|value| !value.is_finite())
            {
                return Err(VokraError::InvalidArgument(
                    "voxcpm feature loop local encoder output is empty or non-finite".to_owned(),
                ));
            }
            if step > self.min_steps && stop(&lm_hidden)? {
                break;
            }
            lm_hidden = base_lm(&curr_embed)?;
            if lm_hidden.len() != self.hidden_dim
                || lm_hidden.iter().any(|value| !value.is_finite())
            {
                return Err(VokraError::InvalidArgument(
                    "voxcpm feature loop base LM state shape/finiteness mismatch".to_owned(),
                ));
            }
            let fsq_hidden = quantize(&lm_hidden)?;
            if fsq_hidden.len() != self.hidden_dim
                || fsq_hidden.iter().any(|value| !value.is_finite())
            {
                return Err(VokraError::InvalidArgument(
                    "voxcpm feature loop FSQ hidden shape/finiteness mismatch".to_owned(),
                ));
            }
            // The next DiT/stop iteration observes the source's
            // FSQ-transformed base hidden, while the base KV cache retains
            // the raw transformer output internally.
            lm_hidden = fsq_hidden.clone();
            // VoxCPM feeds the residual LM the elementwise sum, not a
            // concatenation.  This is important for the residual stack's
            // hidden width (1024).
            let residual_input: Vec<f32> = fsq_hidden
                .iter()
                .zip(&curr_embed)
                .map(|(fsq, current)| fsq + current)
                .collect();
            residual_hidden = residual(&residual_input)?;
            if residual_hidden.len() != self.hidden_dim
                || residual_hidden.iter().any(|value| !value.is_finite())
            {
                return Err(VokraError::InvalidArgument(
                    "voxcpm feature loop residual LM state shape/finiteness mismatch".to_owned(),
                ));
            }
        }
        Ok(patches)
    }

    /// Run the source feature protocol against persistent base/residual
    /// caches. The caller owns the CFM draw/conditioning callback and local
    /// encoder; this method supplies the exact ordering and commits each LM
    /// step atomically through [`CausalLanguageState::step`]. On any later
    /// error, cache lengths and both hidden states are restored to entry.
    #[allow(dead_code)] // Enabled only by the dormant staged batch-one route.
    #[allow(clippy::too_many_arguments)] // Persistent source state and graph callbacks are distinct inputs.
    pub(crate) fn generate_with_language<C, E, S>(
        &self,
        seed_prefix: &[f32],
        state: &mut CausalLanguageState,
        quantizer: &ScalarQuantizer,
        compute: &Compute,
        mut cfm: C,
        mut local_encoder: E,
        mut stop: S,
    ) -> Result<Vec<f32>>
    where
        C: FnMut(usize, &[f32], &[f32], &[f32]) -> Result<Vec<f32>>,
        E: FnMut(&[f32]) -> Result<Vec<f32>>,
        S: FnMut(&[f32]) -> Result<bool>,
    {
        let prefix_len = self.feature_dim * FEATURE_PATCHES_PER_STEP;
        if seed_prefix.len() != prefix_len || seed_prefix.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "voxcpm staged feature prefix must be finite [2,64]".to_owned(),
            ));
        }
        if state.lm_hidden.len() != self.hidden_dim
            || state.residual_hidden.len() != self.hidden_dim
        {
            return Err(VokraError::InvalidArgument(
                "voxcpm staged language state hidden width mismatch".to_owned(),
            ));
        }
        let snapshot = state.snapshot();
        let result = (|| {
            let mut prefix = seed_prefix.to_vec();
            let mut patches = Vec::with_capacity(self.max_steps * prefix_len);
            for step in 0..self.max_steps {
                let patch = cfm(step, state.lm_hidden(), state.residual_hidden(), &prefix)?;
                if patch.len() != prefix_len || patch.iter().any(|value| !value.is_finite()) {
                    return Err(VokraError::InvalidArgument(
                        "voxcpm staged CFM patch must be finite [2,64]".to_owned(),
                    ));
                }
                patches.extend_from_slice(&patch);
                prefix.copy_from_slice(&patch);
                let curr_embed = local_encoder(&patch)?;
                if curr_embed.len() != self.hidden_dim
                    || curr_embed.iter().any(|value| !value.is_finite())
                {
                    return Err(VokraError::InvalidArgument(
                        "voxcpm staged local encoder row must be finite [1024]".to_owned(),
                    ));
                }
                if step > self.min_steps && stop(state.lm_hidden())? {
                    break;
                }
                state.step(&curr_embed, quantizer, compute)?;
            }
            Ok(patches)
        })();
        if result.is_err() {
            state.restore_snapshot(&snapshot);
        }
        result
    }

    /// Rearrange generated `[B,T,P,D]` patches to source channel-major
    /// `[B,D,T*P]`, removing the first and last feature token from each
    /// sequence.  The endpoint removal is explicit because it is part of the
    /// VoxCPM waveform contract, not an incidental slice in a caller.
    pub fn patches_to_latent(
        patches: &[f32],
        batch: usize,
        steps: usize,
        patches_per_step: usize,
        feature_dim: usize,
    ) -> Result<Vec<f32>> {
        if batch == 0 || steps == 0 || patches_per_step == 0 || feature_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm latent rearrange dimensions must be non-zero".to_owned(),
            ));
        }
        let total_tokens = steps.checked_mul(patches_per_step).ok_or_else(|| {
            VokraError::InvalidArgument("voxcpm latent token overflow".to_owned())
        })?;
        if total_tokens <= 2
            || patches.len()
                != batch
                    .checked_mul(total_tokens)
                    .and_then(|v| v.checked_mul(feature_dim))
                    .ok_or_else(|| {
                        VokraError::InvalidArgument("voxcpm latent buffer overflow".to_owned())
                    })?
            || patches.iter().any(|x| !x.is_finite())
        {
            return Err(VokraError::InvalidArgument(
                "voxcpm latent patches have invalid shape or values".to_owned(),
            ));
        }
        let kept = total_tokens - 2;
        let mut output = vec![0.0; batch * feature_dim * kept];
        for b in 0..batch {
            for token in 1..(total_tokens - 1) {
                let t = token / patches_per_step;
                let p = token % patches_per_step;
                let source = (b * total_tokens + t * patches_per_step + p) * feature_dim;
                let target_token = token - 1;
                for channel in 0..feature_dim {
                    output[(b * feature_dim + channel) * kept + target_token] =
                        patches[source + channel];
                }
            }
        }
        Ok(output)
    }
}

/// Scalar quantisation used between the base LM and acoustic path.
#[derive(Debug, Clone)]
pub struct ScalarQuantizer {
    /// Input LM width (0.5B: 1024).
    pub input_dim: usize,
    /// Number of projected channels (0.5B: 256).
    pub channels: usize,
    /// Symmetric source scale (0.5B: 9).
    pub scale: f32,
    /// Source-layout input projection weights `[channels, input_dim]`.
    pub in_weight: Vec<f32>,
    /// Source-layout input projection bias `[channels]`.
    pub in_bias: Vec<f32>,
    /// Source-layout output projection weights `[input_dim, channels]`.
    pub out_weight: Vec<f32>,
    /// Source-layout output projection bias `[input_dim]`.
    pub out_bias: Vec<f32>,
    in_weight_t: Vec<f32>,
    out_weight_t: Vec<f32>,
}

impl ScalarQuantizer {
    /// Construct the learned in/out projections from authenticated tensors.
    pub fn from_weights(
        input_dim: usize,
        channels: usize,
        scale: f32,
        in_weight: Vec<f32>,
        in_bias: Vec<f32>,
        out_weight: Vec<f32>,
        out_bias: Vec<f32>,
    ) -> Result<Self> {
        if input_dim == 0
            || channels == 0
            || !scale.is_finite()
            || scale <= 0.0
            || in_weight.len() != channels * input_dim
            || in_bias.len() != channels
            || out_weight.len() != input_dim * channels
            || out_bias.len() != input_dim
            || in_weight
                .iter()
                .chain(&in_bias)
                .chain(&out_weight)
                .chain(&out_bias)
                .any(|value| !value.is_finite())
        {
            return Err(VokraError::InvalidArgument(
                "voxcpm scalar quantizer projection shape/axis mismatch".to_owned(),
            ));
        }
        let in_weight_t = transpose_weight(&in_weight, channels, input_dim);
        let out_weight_t = transpose_weight(&out_weight, input_dim, channels);
        Ok(Self {
            input_dim,
            channels,
            scale,
            in_weight,
            in_bias,
            out_weight,
            out_bias,
            in_weight_t,
            out_weight_t,
        })
    }

    /// Load only the source-shaped FSQ tensors from a VAST-staged GGUF.
    /// This crate-private entry point is not production authentication:
    /// callers must first obtain the complete immutable composite manifest.
    #[allow(dead_code)] // Staged FSQ loading awaits complete composite authorization.
    pub(crate) fn from_staged_gguf(file: &GgufFile) -> Result<Self> {
        Self::from_weights(
            1_024,
            256,
            9.0,
            load_tensor(file, "voxcpm2", "fsq_layer.in_proj.weight", &[256, 1_024])?,
            load_tensor(file, "voxcpm2", "fsq_layer.in_proj.bias", &[256])?,
            load_tensor(file, "voxcpm2", "fsq_layer.out_proj.weight", &[1_024, 256])?,
            load_tensor(file, "voxcpm2", "fsq_layer.out_proj.bias", &[1_024])?,
        )
    }

    /// Apply learned projection, tanh and the source `round(x*scale)/scale`
    /// FSQ rule to one hidden vector.
    pub fn quantize(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.input_dim || input.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "voxcpm scalar quantizer input shape/finiteness mismatch".to_owned(),
            ));
        }
        let mut projected = vec![0.0; self.channels];
        for (channel, value) in projected.iter_mut().enumerate() {
            let row = &self.in_weight[channel * self.input_dim..(channel + 1) * self.input_dim];
            *value = (row.iter().zip(input).map(|(w, x)| w * x).sum::<f32>()
                + self.in_bias[channel])
                .tanh();
            *value = (*value * self.scale).round() / self.scale;
        }
        Ok(projected)
    }

    /// Apply the complete learned FSQ route through the selected backend:
    /// `Linear(1024→256) → tanh → round(x*9)/9 → Linear(256→1024)`.
    ///
    /// The small scalar rounding step is intentionally performed between the
    /// two backend-dispatched learned projections; it is not a replacement
    /// for either projection and cannot silently route a Metal model through
    /// host-side matrix multiplication.
    pub fn apply(&self, input: &[f32], compute: &Compute) -> Result<Vec<f32>> {
        if input.len() != self.input_dim || input.iter().any(|x| !x.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "voxcpm scalar quantizer input shape/finiteness mismatch".to_owned(),
            ));
        }
        let mut projected = vec![0.0; self.channels];
        compute.gemm_f32(
            1,
            self.channels,
            self.input_dim,
            input,
            &self.in_weight_t,
            Some(&self.in_bias),
            &mut projected,
        )?;
        let mut activated = vec![0.0; self.channels];
        compute.tanh_f32(&projected, &mut activated)?;
        for value in &mut activated {
            *value = (*value * self.scale).round() / self.scale;
        }
        let mut output = vec![0.0; self.input_dim];
        compute.gemm_f32(
            1,
            self.input_dim,
            self.channels,
            &activated,
            &self.out_weight_t,
            Some(&self.out_bias),
            &mut output,
        )?;
        if output.iter().any(|x| !x.is_finite()) {
            return Err(VokraError::ModelLoad(
                "voxcpm scalar quantizer produced non-finite hidden values".to_owned(),
            ));
        }
        Ok(output)
    }

    /// Apply the learned output projection to an FSQ vector.
    pub fn dequantize(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.channels
            || input
                .iter()
                .any(|value| !value.is_finite() || *value < -1.0 || *value > 1.0)
        {
            return Err(VokraError::InvalidArgument(
                "voxcpm scalar quantizer dequantized vector is out of range".to_owned(),
            ));
        }
        let mut output = vec![0.0; self.input_dim];
        for (row, value) in output.iter_mut().enumerate() {
            let weights = &self.out_weight[row * self.channels..(row + 1) * self.channels];
            *value =
                weights.iter().zip(input).map(|(w, x)| w * x).sum::<f32>() + self.out_bias[row];
        }
        Ok(output)
    }
}

fn transpose_weight(weight: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            transposed[col * rows + row] = weight[row * cols + col];
        }
    }
    transposed
}

#[allow(dead_code)] // Staged stack loading awaits complete composite authorization.
fn load_causal_stack(file: &GgufFile, prefix: &str, layers: usize) -> Result<MiniCpm4Stack> {
    let base = MiniCpm4Config::voxcpm_0_5b()?;
    let config = MiniCpm4Config::new_with_original_max_positions(
        VOXCPM_HIDDEN,
        VOXCPM_FFN,
        layers,
        base.n_heads(),
        base.n_kv_heads(),
        base.max_positions(),
        base.original_max_positions(),
        base.rope_theta(),
        base.rms_norm_eps(),
        false,
        base.rope_short_factor().to_vec(),
        base.rope_long_factor().to_vec(),
    )?;
    let mut blocks = Vec::with_capacity(layers);
    for layer in 0..layers {
        let stem = format!("{prefix}.layers.{layer}");
        let tensor = |suffix: &str, shape: &[usize]| {
            load_tensor(file, "voxcpm2", &format!("{stem}.{suffix}"), shape)
        };
        blocks.push(MiniCpm4BlockWeights::from_source(
            &config,
            tensor("input_layernorm.weight", &[VOXCPM_HIDDEN])?,
            tensor("post_attention_layernorm.weight", &[VOXCPM_HIDDEN])?,
            tensor("self_attn.q_proj.weight", &[VOXCPM_HIDDEN, VOXCPM_HIDDEN])?,
            tensor("self_attn.k_proj.weight", &[VOXCPM_KV, VOXCPM_HIDDEN])?,
            tensor("self_attn.v_proj.weight", &[VOXCPM_KV, VOXCPM_HIDDEN])?,
            tensor("self_attn.o_proj.weight", &[VOXCPM_HIDDEN, VOXCPM_HIDDEN])?,
            tensor("mlp.gate_proj.weight", &[VOXCPM_FFN, VOXCPM_HIDDEN])?,
            tensor("mlp.up_proj.weight", &[VOXCPM_FFN, VOXCPM_HIDDEN])?,
            tensor("mlp.down_proj.weight", &[VOXCPM_HIDDEN, VOXCPM_FFN])?,
        )?);
    }
    MiniCpm4Stack::new(
        config,
        MiniCpm4StackWeights::from_source(
            blocks,
            load_tensor(
                file,
                "voxcpm2",
                &format!("{prefix}.norm.weight"),
                &[VOXCPM_HIDDEN],
            )?,
        ),
    )
}

/// Deterministic Euler flow used by UnifiedCFM.
#[derive(Debug, Clone, Copy)]
pub struct EulerFlow {
    /// Number of integration steps.
    pub steps: usize,
    /// Sway coefficient from UnifiedCFM's timestep transform.
    pub sway_coefficient: f32,
    /// Classifier-free guidance scale.
    pub cfg_scale: f32,
}

impl EulerFlow {
    /// Construct a valid flow schedule.
    pub fn new(steps: usize) -> Result<Self> {
        if steps == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm Euler flow requires at least one step".to_owned(),
            ));
        }
        Ok(Self {
            steps,
            sway_coefficient: 0.0,
            cfg_scale: 1.0,
        })
    }

    /// Construct the source CFG/sway schedule explicitly.
    pub fn with_schedule(steps: usize, sway_coefficient: f32, cfg_scale: f32) -> Result<Self> {
        if !sway_coefficient.is_finite() || !cfg_scale.is_finite() {
            return Err(VokraError::InvalidArgument(
                "voxcpm Euler schedule must be finite".to_owned(),
            ));
        }
        let mut flow = Self::new(steps)?;
        flow.sway_coefficient = sway_coefficient;
        flow.cfg_scale = cfg_scale;
        Ok(flow)
    }

    fn sway(&self, t: f32) -> f32 {
        t + self.sway_coefficient * ((core::f32::consts::FRAC_PI_2 * t).cos() - 1.0 + t)
    }

    fn t_span(&self) -> Vec<f32> {
        (0..=self.steps)
            .map(|index| self.sway(1.0 - index as f32 / self.steps as f32))
            .collect()
    }

    /// Integrate UnifiedCFM's reverse `t=1→0` Euler schedule.
    pub fn integrate<F>(&self, mut state: Vec<f32>, mut velocity: F) -> Result<Vec<f32>>
    where
        F: FnMut(f32, &[f32]) -> Result<Vec<f32>>,
    {
        if state.is_empty() || state.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "voxcpm Euler flow state must be finite and non-empty".to_owned(),
            ));
        }
        let t_span = self.t_span();
        for index in 1..=self.steps {
            let dt = t_span[index - 1] - t_span[index];
            let update = velocity(t_span[index - 1], &state)?;
            if update.len() != state.len() || update.iter().any(|value| !value.is_finite()) {
                return Err(VokraError::InvalidArgument(
                    "voxcpm Euler velocity shape/finiteness mismatch".to_owned(),
                ));
            }
            for (value, derivative) in state.iter_mut().zip(update) {
                *value -= dt * derivative;
            }
        }
        Ok(state)
    }

    /// CFG form of the UnifiedCFM Euler schedule. The initial
    /// `max(1, int((N+1)*0.04))` velocity estimates are zeroed by source
    /// design. The positive/negative pair uses UnifiedCFM's optimized scale.
    pub fn integrate_cfg<P, N>(
        &self,
        mut state: Vec<f32>,
        mut positive: P,
        mut negative: N,
    ) -> Result<Vec<f32>>
    where
        P: FnMut(f32, &[f32]) -> Result<Vec<f32>>,
        N: FnMut(f32, &[f32]) -> Result<Vec<f32>>,
    {
        if state.is_empty() || state.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "voxcpm Euler CFG state must be finite and non-empty".to_owned(),
            ));
        }
        let t_span = self.t_span();
        let zero_steps = ((self.steps + 1) * 4 / 100).max(1);
        for index in 1..=self.steps {
            let dt = t_span[index - 1] - t_span[index];
            // UnifiedCFM leaves the zero-star prefix untouched and does not
            // invoke either estimator during those steps.
            if index <= zero_steps {
                continue;
            }
            let plus = positive(t_span[index - 1], &state)?;
            let minus = negative(t_span[index - 1], &state)?;
            if plus.len() != state.len()
                || minus.len() != state.len()
                || plus.iter().chain(&minus).any(|value| !value.is_finite())
            {
                return Err(VokraError::InvalidArgument(
                    "voxcpm Euler CFG velocity shape/finiteness mismatch".to_owned(),
                ));
            }
            let dot = plus.iter().zip(&minus).map(|(p, n)| p * n).sum::<f32>();
            let norm = minus.iter().map(|value| value * value).sum::<f32>();
            let optimized_scale = dot / (norm + 1e-8);
            for ((value, pos), neg) in state.iter_mut().zip(plus).zip(minus) {
                let scaled_neg = neg * optimized_scale;
                *value -= dt * (scaled_neg + self.cfg_scale * (pos - scaled_neg));
            }
        }
        Ok(state)
    }
}

/// Two-class stop projection argmax controller.
#[derive(Debug, Clone, Copy)]
pub struct StopController {
    /// Index of the stop class.
    pub stop_class: usize,
}

impl StopController {
    /// Return whether `logits` selects the stop class.
    pub fn should_stop(&self, logits: &[f32]) -> Result<bool> {
        if logits.is_empty()
            || self.stop_class >= logits.len()
            || logits.iter().any(|value| !value.is_finite())
        {
            return Err(VokraError::InvalidArgument(
                "voxcpm stop logits are empty, non-finite, or missing stop class".to_owned(),
            ));
        }
        let best = logits
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .expect("non-empty logits checked above");
        Ok(best == self.stop_class)
    }
}

/// Learned stop path used by the feature loop.  The projection and head are
/// deliberately separate because the source applies SiLU between them:
/// `stop_head(SiLU(stop_proj(lm_hidden))).argmax()`.
#[derive(Debug, Clone)]
pub struct LearnedStopController {
    hidden_dim: usize,
    stop_proj_bias: Vec<f32>,
    stop_proj_weight_t: Vec<f32>,
    stop_head_weight_t: Vec<f32>,
    stop_class: usize,
}

impl LearnedStopController {
    /// Construct the learned two-class stop predictor from source-layout weights.
    pub fn from_weights(
        hidden_dim: usize,
        stop_proj_weight: Vec<f32>,
        stop_proj_bias: Vec<f32>,
        stop_head_weight: Vec<f32>,
        stop_class: usize,
    ) -> Result<Self> {
        if hidden_dim == 0
            || stop_class >= 2
            || stop_proj_weight.len() != hidden_dim * hidden_dim
            || stop_proj_bias.len() != hidden_dim
            || stop_head_weight.len() != 2 * hidden_dim
            || stop_proj_weight
                .iter()
                .chain(&stop_proj_bias)
                .chain(&stop_head_weight)
                .any(|x| !x.is_finite())
        {
            return Err(VokraError::InvalidArgument(
                "voxcpm stop path has invalid learned tensor shapes or values".to_owned(),
            ));
        }
        let stop_proj_weight_t = transpose_weight(&stop_proj_weight, hidden_dim, hidden_dim);
        let stop_head_weight_t = transpose_weight(&stop_head_weight, 2, hidden_dim);
        Ok(Self {
            hidden_dim,
            stop_proj_bias,
            stop_proj_weight_t,
            stop_head_weight_t,
            stop_class,
        })
    }

    /// Load the source-shaped stop projection and parameter head from a
    /// VAST-staged GGUF.  The public production composite binder remains
    /// fail-closed until its complete manifest is authenticated.
    #[allow(dead_code)] // Staged stop weights await complete composite authorization.
    pub(crate) fn from_staged_gguf(file: &GgufFile) -> Result<Self> {
        Self::from_weights(
            1_024,
            load_tensor(file, "voxcpm2", "stop_proj.weight", &[1_024, 1_024])?,
            load_tensor(file, "voxcpm2", "stop_proj.bias", &[1_024])?,
            load_tensor(file, "voxcpm2", "stop_head.weight", &[2, 1_024])?,
            1,
        )
    }

    /// Predict whether the supplied hidden row selects the source stop class.
    pub fn should_stop(&self, hidden: &[f32], compute: &Compute) -> Result<bool> {
        if hidden.len() != self.hidden_dim || hidden.iter().any(|x| !x.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "voxcpm stop hidden row shape/finiteness mismatch".to_owned(),
            ));
        }
        let mut projected = vec![0.0; self.hidden_dim];
        compute.gemm_f32(
            1,
            self.hidden_dim,
            self.hidden_dim,
            hidden,
            &self.stop_proj_weight_t,
            Some(&self.stop_proj_bias),
            &mut projected,
        )?;
        let mut activated = vec![0.0; self.hidden_dim];
        compute.silu_f32(&projected, &mut activated)?;
        let mut logits = vec![0.0; 2];
        compute.gemm_f32(
            1,
            2,
            self.hidden_dim,
            &activated,
            &self.stop_head_weight_t,
            None,
            &mut logits,
        )?;
        StopController {
            stop_class: self.stop_class,
        }
        .should_stop(&logits)
    }
}

#[allow(dead_code)] // Used only by the dormant staged CFG path and its layout tests.
fn row_major_to_channel_major(
    input: &[f32],
    positions: usize,
    channels: usize,
) -> Result<Vec<f32>> {
    if positions == 0 || channels == 0 || input.len() != positions * channels {
        return Err(VokraError::InvalidArgument(
            "voxcpm feature layout requires [positions, channels] input".to_owned(),
        ));
    }
    let mut output = vec![0.0; input.len()];
    for position in 0..positions {
        for channel in 0..channels {
            output[channel * positions + position] = input[position * channels + channel];
        }
    }
    Ok(output)
}

#[allow(dead_code)] // Used only by the dormant staged CFG path.
fn cfg_condition(prefix: &[f32]) -> Result<Vec<f32>> {
    row_major_to_channel_major(prefix, FEATURE_PATCHES_PER_STEP, 64)
}

#[allow(dead_code)] // Used only by the dormant staged CFG path.
fn cfg_negative_mu() -> Vec<f32> {
    vec![0.0; VOXCPM_HIDDEN]
}

#[allow(dead_code)] // Used only by the dormant staged CFG path and its layout tests.
fn channel_major_to_row_major(
    input: &[f32],
    channels: usize,
    positions: usize,
) -> Result<Vec<f32>> {
    if positions == 0 || channels == 0 || input.len() != positions * channels {
        return Err(VokraError::InvalidArgument(
            "voxcpm feature layout requires [channels, positions] input".to_owned(),
        ));
    }
    let mut output = vec![0.0; input.len()];
    for channel in 0..channels {
        for position in 0..positions {
            output[position * channels + channel] = input[channel * positions + position];
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[test]
    fn batch1_flow_draws_are_caller_owned_and_layout_explicit() {
        let mut channel_major = vec![0.0f32; FEATURE_PATCH_WIDTH];
        channel_major[0] = 1.0;
        channel_major[1] = 2.0;
        let draws = VoxCpm2FlowDraws::new(vec![channel_major], 1).unwrap();
        assert_eq!(draws.len(), 1);
        assert_eq!(
            channel_major_to_row_major(draws.get(0).unwrap(), 64, 2).unwrap()[..4],
            [1.0, 0.0, 2.0, 0.0]
        );
        assert!(VoxCpm2FlowDraws::new(vec![vec![0.0; FEATURE_PATCH_WIDTH]], 2).is_err());
    }

    #[test]
    fn cfg_negative_reuses_each_dynamic_prefix_and_zeroes_only_mu() {
        let first = vec![0.0f32; FEATURE_PATCH_WIDTH];
        let mut second = first.clone();
        second[0] = 7.0;
        second[65] = -3.0;
        let first_negative = cfg_condition(&first).unwrap();
        let second_negative = cfg_condition(&second).unwrap();
        assert_ne!(first_negative, second_negative);
        assert_eq!(second_negative[0], 7.0);
        assert_eq!(second_negative[1], 0.0);
        assert_eq!(second_negative[2], 0.0);
        assert_eq!(second_negative[3], -3.0);
        assert!(cfg_negative_mu().iter().all(|value| *value == 0.0));
    }

    #[test]
    fn scalar_quantizer_projects_fsq_and_rejects_nonfinite() {
        let quantizer = ScalarQuantizer::from_weights(
            3,
            3,
            9.0,
            vec![1.0; 9],
            vec![0.0; 3],
            vec![1.0; 9],
            vec![0.0; 3],
        )
        .unwrap();
        assert_eq!(quantizer.quantize(&[0.0, 0.0, 0.0]).unwrap(), vec![0.0; 3]);
        assert!(quantizer.quantize(&[f32::NAN, 0.0, 0.0]).is_err());
        assert!(quantizer.dequantize(&[2.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn euler_schedule_is_deterministic_and_shape_checked() {
        let flow = EulerFlow::new(2).unwrap();
        assert_eq!(
            flow.integrate(vec![0.0], |_time, _state| Ok(vec![2.0]))
                .unwrap(),
            vec![-2.0]
        );
        assert!(
            flow.integrate(vec![0.0], |_time, _state| Ok(vec![]))
                .is_err()
        );
    }

    #[test]
    fn stop_controller_uses_argmax() {
        let controller = StopController { stop_class: 1 };
        assert!(controller.should_stop(&[0.0, 1.0]).unwrap());
        assert!(!controller.should_stop(&[2.0, 1.0]).unwrap());
        assert!(controller.should_stop(&[f32::INFINITY, 0.0]).is_err());
    }

    #[test]
    fn feature_loop_pins_patch_order_and_minimum_stop() {
        let loop_ = FeatureGenerationLoop::new(3, 0, 2, 2).unwrap();
        let calls = Cell::new(0);
        let result = loop_
            .generate(
                &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                |_step, _lm, _residual, _prefix| {
                    calls.set(calls.get() + 1);
                    Ok(vec![calls.get() as f32; 4])
                },
                |_patch| Ok(vec![1.0, 2.0]),
                |_embed| Ok(vec![1.0, 2.0]),
                |_hidden| Ok(vec![0.0, 0.0]),
                |_input| Ok(vec![0.0, 0.0]),
                |_lm| Ok(calls.get() >= 2),
            )
            .unwrap();
        assert_eq!(result, vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]);
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn feature_loop_updates_prefix_and_sums_residual_input() {
        let loop_ = FeatureGenerationLoop::new(3, 3, 2, 1).unwrap();
        let mut seen_prefix = Vec::new();
        let mut residual_input = Vec::new();
        let patches = loop_
            .generate(
                &[0.0, 0.0, 0.0, 0.0],
                |step, _lm, _residual, prefix| {
                    seen_prefix.push(prefix[0]);
                    Ok(vec![step as f32 + 1.0, step as f32 + 1.0])
                },
                |_patch| Ok(vec![3.0, 4.0]),
                |_embed| Ok(vec![1.0, 2.0]),
                |_hidden| Ok(vec![10.0, 20.0]),
                |input| {
                    residual_input.extend_from_slice(input);
                    Ok(vec![5.0, 6.0])
                },
                |_hidden| Ok(false),
            )
            .unwrap();
        assert_eq!(seen_prefix, vec![0.0, 1.0, 2.0]);
        assert_eq!(residual_input, vec![11.0, 22.0, 11.0, 22.0]);
        assert_eq!(patches, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn patches_to_latent_is_channel_major_and_drops_endpoints() {
        // B=1, T=2, P=2, D=2; token values are [token, channel].
        let patches = vec![
            0.0, 10.0, 1.0, 11.0, // first step
            2.0, 12.0, 3.0, 13.0, // last step
        ];
        assert_eq!(
            FeatureGenerationLoop::patches_to_latent(&patches, 1, 2, 2, 2).unwrap(),
            vec![1.0, 2.0, 11.0, 12.0]
        );
    }

    #[test]
    fn euler_cfg_skips_zero_star_estimators_and_uses_current_time() {
        let flow = EulerFlow::with_schedule(10, 0.0, 1.0).unwrap();
        let mut plus_calls = 0;
        let mut first_time = None;
        let mut minus_calls = 0;
        flow.integrate_cfg(
            vec![1.0],
            |time, _| {
                plus_calls += 1;
                first_time.get_or_insert(time);
                Ok(vec![0.0])
            },
            |_time, _| {
                minus_calls += 1;
                Ok(vec![0.0])
            },
        )
        .unwrap();
        assert_eq!(plus_calls, 9);
        assert_eq!(minus_calls, 9);
        assert_eq!(first_time, Some(0.9));
    }

    #[test]
    fn prefill_audio_mask_is_applied_only_after_base_outputs() {
        let loop_ = FeatureGenerationLoop::new(1, 0, 2, 1).unwrap();
        let state = loop_
            .assemble_prefill(
                &[1.0, 2.0, 3.0, 4.0],
                &[7.0, 8.0, 9.0, 10.0],
                &[false, true],
                2.0,
                |feature| Ok(vec![feature[0], feature[0] + 1.0]),
            )
            .unwrap();
        assert_eq!(state.base_rows(), &[2.0, 4.0, 9.0, 10.0]);
        let quantizer = ScalarQuantizer::from_weights(
            2,
            2,
            9.0,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0; 2],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0; 2],
        )
        .unwrap();
        let encoded = state
            .encoded_outputs(&[10.0, 20.0, 30.0, 40.0], &quantizer, &Compute::cpu())
            .unwrap();
        assert_eq!(encoded, vec![10.0, 20.0, 1.0, 1.0]);
        assert_eq!(
            state.residual_rows_from_encoded(&encoded).unwrap(),
            vec![10.0, 20.0, 10.0, 11.0]
        );
    }
}
