//! **CC-side runtime shell (SCAFFOLD)** for Meta AudioCraft
//! **MelodyFlow T24 30secs** (facebook/melodyflow-t24-30secs,
//! CC-BY-NC-4.0) — post-audit CC-gap 2026-08-13 Wave D remaining WF8.
//!
//! # SCAFFOLD status (RMVPE / DNSMOS / openwakeword / MAGNeT precedent)
//!
//! This module is deliberately **not** a working MelodyFlow runtime.
//! It ships:
//!
//! - [`MelodyFlowConfig`] — hparams deserialised from GGUF metadata
//!   under `vokra.melodyflow.*` (num_layers, hidden_size, num_heads,
//!   num_timesteps, latent_dim, num_codebooks, codebook_size,
//!   codec_frame_rate_hz, max_duration_secs, sample_rate_hz,
//!   text_prefix_len, cfg_scale).
//! - [`MelodyFlowWeightEntry`] — one bound weight tensor entry
//!   (name-only reference; the shell does NOT preload weights, same
//!   posture as [`MagnetWeightEntry`](crate::magnet::MagnetWeightEntry)
//!   and `DnsmosBundle`).
//! - [`MelodyFlowEngine::from_gguf`] — validates arch tag against
//!   [`ARCH`], deserialises config, catalogues the weight tensor
//!   names.
//! - [`MelodyFlowEngine::forward`] — returns
//!   [`VokraError::UnsupportedOp`] naming (a) the ADR to ratify
//!   (`docs/adr/M5-melodyflow-dit-sampler.md`, Status: **Proposed**),
//!   (b) the two `vokra-ops` primitives that need to land
//!   (`flow_editing_inversion` + `t24_transformer` — the FR-OP-86
//!   anchor), (c) the reused `vokra_ops::flow_sampler::flow_sample`
//!   seam from M3-05 (regeneration ODE integrator — already exists),
//!   and (d) the codec decode integration owner-only path (see the
//!   ADR §D-5). The bundled 48 kHz RVQ codec's `--allow-noncommercial`
//!   distribution gating (FR-OP-32) is inherited from the
//!   `MelodyflowT2430secsReport` license class in the converter.
//!
//! There is no synthesised DiT forward, no fabricated `Vec<f32>` PCM
//! output, no silent CPU stub. [`MelodyFlowEngine::forward`] is
//! **loud-partial** — see [`crate::dnsmos_p808_p835::Dnsmos::score_p808`]
//! and [`crate::f0::rmvpe`] and [`crate::magnet::MagnetEngine::forward`]
//! for the same posture.
//!
//! # Why this shape (design summary — see the ADR for the full rationale)
//!
//! MelodyFlow (Le Lan et al. 2024 arXiv:2407.03648) is a
//! **flow-matching / DiT** music model whose primary use-case is
//! **text-conditioned music editing**: an existing 48 kHz audio clip
//! is inverted through a rectified-flow ODE (noise ← audio) and then
//! regenerated forward under a new text prompt. The two runtime-side
//! primitives it needs beyond what already exists in `vokra-ops`
//! today —
//! [`flow_editing_inversion`](../../docs/adr/M5-melodyflow-dit-sampler.md#D-2)
//! (reverse-ODE walk that maps ground-truth audio latent → noise
//! latent under the source text) and
//! [`t24_transformer`](../../docs/adr/M5-melodyflow-dit-sampler.md#D-3)
//! (the 24-layer DiT block stack with timestep-conditioned adaLN and
//! dual text + audio prefix cross-attention) — are runtime functions
//! under [`vokra_ops`], NOT `OpKind` variants, following the M3-05
//! `flow_sampler` / M3-06 `mimi_rvq` / M4-04 `dac_rvq` / MAGNeT
//! `magnet_masked_decode` / openwakeword classifier precedent
//! (`docs/adr/M3-05-flow-sampler.md` §D1,
//! `docs/adr/M3-06-mimi-rvq.md` §D-b,
//! `docs/adr/M5-magnet-masked-ar-op.md` §D-1). Landing them today
//! would also violate ADR M4-20 §D-1 (trigger-backed subset rule —
//! the C ABI freeze IF-01 = M5-13 is coming; no live consumer today).
//!
//! The **regeneration ODE integrator** is *already* covered by
//! [`vokra_ops::flow_sampler::flow_sample`](../../vokra_ops/flow_sampler/fn.flow_sample.html)
//! from M3-05 — `Schedule::Linear` + `OdeSolver::Euler` +
//! `CfgMode::DualForward` matches Le Lan et al. 2024 Algorithm 1
//! verbatim, so no new op is needed for the forward walk. Only the
//! reverse (editing inversion) walk requires the new
//! `flow_editing_inversion` driver — the ADR §D-2 records the
//! reasoning for a distinct entry point over a `Schedule::Reversed`
//! variant.
//!
//! # Codec decode is owner-driven
//!
//! MelodyFlow bundles a **48 kHz RVQ codec** decoder. That codec is
//! CC-BY-NC-4.0 (FR-OP-32 distribution restriction — see the ADR
//! §D-5 for the Option A / Option B rationale). The codec decode step
//! is deliberately outside this module — a future integration will
//! either (Option A) land a new `melodyflow_rvq_48khz` engine op
//! (keeps T4 = research-only) or (Option B) substitute a permissively
//! -licensed 48 kHz codec (requires retraining evidence — see the ADR
//! §D-5). Neither is CC-scope today.
//!
//! # Downstream vocoder handoff (BigVGAN / Vocos / …)
//!
//! Consumers who want to skip the bundled RVQ codec and feed the DiT
//! output through a general-purpose neural vocoder (BigVGAN / Vocos /
//! HiFi-GAN) can wire the vocoder downstream — this shell does not
//! prescribe or block that path. See the ADR §D-6 for the owner
//! integration record and `vokra-models::codec::bigvgan` for the
//! Vokra-side BigVGAN op-family.
//!
//! # Distinct arch tag (FR-EX-08)
//!
//! One arch tag today: [`ARCH`] = `"melodyflow_t24_30secs"`.
//! Silent-sharing an arch tag with any sibling music-gen family —
//! MAGNeT (`magnet_small_10secs` / `magnet_medium_30secs`,
//! non-autoregressive masked-LM decoding), MusicGen family
//! (`musicgen_{small,medium,large}` / `audiogen_medium`,
//! AR-over-EnCodec), JASCO (`jasco_400m_chords_drums`, flow-matching
//! with joint audio-symbolic chord/drum conditioning stack rather
//! than dual text + audio prefix), AudioLDM2 (score-based latent
//! diffusion U-Net), Stable Audio Open (DiT + audio VAE with
//! different conditioning), ACE-Step, YuE bundle, or BS-RoFormer
//! (music-source separation) — is a FR-EX-08 violation because the
//! runtime dispatch would mis-route to a family with a different
//! decoder loop.

use std::sync::Arc;

use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{Result, VokraError};

#[cfg(test)]
mod tests;

// -----------------------------------------------------------------------------
// arch tag — mirror of the converter module
// (crates/vokra-convert/src/models/melodyflow_t24_30secs.rs `pub const ARCH`).
// Duplicated here (same convention every fsmn_vad / silero_vad / dnsmos /
// openwakeword / magnet binder uses) so the runtime crate does not need a
// cross-crate dependency edge onto the converter.
// -----------------------------------------------------------------------------

/// `vokra.model.arch` value for MelodyFlow T24 30secs GGUFs
/// (~1 B params, 30-second music **editing** horizon at 48 kHz).
/// Mirror of `vokra-convert::models::melodyflow_t24_30secs::ARCH`.
pub const ARCH: &str = "melodyflow_t24_30secs";

// -----------------------------------------------------------------------------
// vokra.melodyflow.* metadata keys — proposed schema for the future converter
// extension. The current converter (`convert_melodyflow_t24_30secs_file`)
// does NOT emit these keys yet — it is a BF16 pass-through skeleton only.
// `MelodyFlowEngine::from_gguf` therefore fails loudly (FR-EX-08) on a GGUF
// that does not carry them — no silent default (Le Lan et al. 2024 §4.2
// sweeps num_timesteps + cfg_scale at inference; a silently-defaulted
// config would misrepresent the run).
// -----------------------------------------------------------------------------

/// GGUF metadata key: DiT transformer depth (u32). Le Lan et al.
/// 2024 T24 release = 24.
pub const KEY_MELODYFLOW_NUM_LAYERS: &str = "vokra.melodyflow.num_layers";
/// GGUF metadata key: DiT transformer hidden width (u32).
pub const KEY_MELODYFLOW_HIDDEN_SIZE: &str = "vokra.melodyflow.hidden_size";
/// GGUF metadata key: DiT transformer attention head count (u32).
pub const KEY_MELODYFLOW_NUM_HEADS: &str = "vokra.melodyflow.num_heads";
/// GGUF metadata key: default number of ODE solver steps (u32). The
/// `T24` in the release name is `num_timesteps=24` — Le Lan et al.
/// 2024 §4.2 recommended NFE for editing. Kept as an explicit attribute
/// so runtime callers can sweep it without re-converting (per ADR §D-1).
pub const KEY_MELODYFLOW_NUM_TIMESTEPS: &str = "vokra.melodyflow.num_timesteps";
/// GGUF metadata key: dimensionality of the RVQ latent the DiT
/// backbone operates on (u32). Distinct from `hidden_size` — `latent_dim`
/// is the input / output width of the DiT block stack (in_proj /
/// out_proj), `hidden_size` is the internal DiT hidden width.
pub const KEY_MELODYFLOW_LATENT_DIM: &str = "vokra.melodyflow.latent_dim";
/// GGUF metadata key: number of RVQ codebooks in the bundled 48 kHz
/// codec (u32). Kept as an attribute here even though the codec decode
/// is owner-driven (see ADR §D-5) so the shell can spot-check the
/// weight catalogue against the config without a codec dispatch.
pub const KEY_MELODYFLOW_NUM_CODEBOOKS: &str = "vokra.melodyflow.num_codebooks";
/// GGUF metadata key: vocabulary size per codebook (u32).
pub const KEY_MELODYFLOW_CODEBOOK_SIZE: &str = "vokra.melodyflow.codebook_size";
/// GGUF metadata key: bundled 48 kHz RVQ codec frame rate (u32,
/// frames/s). `sample_rate_hz / hop_length` — a 48 kHz codec running at
/// 25 Hz (~ hop 1920) would carry `codec_frame_rate_hz = 25`. Kept
/// explicit so the shell can compute `seq_len = codec_frame_rate_hz *
/// max_duration_secs` without hard-coding a hop.
pub const KEY_MELODYFLOW_CODEC_FRAME_RATE_HZ: &str = "vokra.melodyflow.codec_frame_rate_hz";
/// GGUF metadata key: maximum generation horizon in seconds (u32). The
/// T24 30secs release = 30.
pub const KEY_MELODYFLOW_MAX_DURATION_SECS: &str = "vokra.melodyflow.max_duration_secs";
/// GGUF metadata key: sample rate of the PCM output the codec targets
/// (u32, Hz). MelodyFlow T24 30secs = 48000.
pub const KEY_MELODYFLOW_SAMPLE_RATE_HZ: &str = "vokra.melodyflow.sample_rate_hz";
/// GGUF metadata key: T5-base text conditioning prefix length in
/// tokens (u32). Kept as an attribute so a future variant with a
/// different text encoder (T5-large / mT5 / …) can override without
/// a converter change.
pub const KEY_MELODYFLOW_TEXT_PREFIX_LEN: &str = "vokra.melodyflow.text_prefix_len";
/// GGUF metadata key: default classifier-free-guidance coefficient
/// (f32). Le Lan et al. 2024 §4.2 typical 4.0. Kept as an explicit
/// attribute so runtime callers can sweep it without re-converting
/// (per ADR §D-1).
pub const KEY_MELODYFLOW_CFG_SCALE: &str = "vokra.melodyflow.cfg_scale";

/// MelodyFlow runtime configuration (transcribed verbatim from
/// `vokra.melodyflow.*` at load time; every field is required and
/// validated loudly per FR-EX-08).
#[derive(Debug, Clone, PartialEq)]
pub struct MelodyFlowConfig {
    /// DiT transformer depth (Le Lan et al. 2024 T24 release = 24).
    pub num_layers: u32,
    /// DiT transformer hidden width.
    pub hidden_size: u32,
    /// DiT transformer attention head count.
    pub num_heads: u32,
    /// Default number of ODE solver steps.
    pub num_timesteps: u32,
    /// Dimensionality of the RVQ latent the DiT operates on.
    pub latent_dim: u32,
    /// Number of RVQ codebooks in the bundled 48 kHz codec.
    pub num_codebooks: u32,
    /// Vocabulary size per codebook.
    pub codebook_size: u32,
    /// Bundled 48 kHz RVQ codec frame rate (frames per second).
    pub codec_frame_rate_hz: u32,
    /// Maximum generation horizon in seconds.
    pub max_duration_secs: u32,
    /// Sample rate of the PCM output (Hz).
    pub sample_rate_hz: u32,
    /// T5-base text conditioning prefix length in tokens.
    pub text_prefix_len: u32,
    /// Default classifier-free-guidance coefficient.
    pub cfg_scale: f32,
}

impl MelodyFlowConfig {
    /// Validates the config loudly (FR-EX-08 — no silent zero /
    /// silent default). Called at the end of [`Self::from_gguf`].
    pub fn validate(&self) -> Result<()> {
        if self.num_layers == 0 {
            return Err(VokraError::ModelLoad(format!(
                "melodyflow: `{KEY_MELODYFLOW_NUM_LAYERS}` = 0 — DiT \
                 transformer must have at least one layer (upstream T24 \
                 release = 24)"
            )));
        }
        if self.num_heads == 0 {
            return Err(VokraError::ModelLoad(format!(
                "melodyflow: `{KEY_MELODYFLOW_NUM_HEADS}` = 0 — must be ≥ 1"
            )));
        }
        if self.hidden_size == 0 || self.hidden_size % self.num_heads != 0 {
            return Err(VokraError::ModelLoad(format!(
                "melodyflow: `{KEY_MELODYFLOW_HIDDEN_SIZE}` = {} not divisible \
                 by `{KEY_MELODYFLOW_NUM_HEADS}` = {} (head_dim must be an integer)",
                self.hidden_size, self.num_heads,
            )));
        }
        if self.num_timesteps == 0 {
            return Err(VokraError::ModelLoad(format!(
                "melodyflow: `{KEY_MELODYFLOW_NUM_TIMESTEPS}` = 0 — ODE \
                 integration requires ≥ 1 step (upstream T24 release = 24; \
                 Le Lan et al. 2024 §4.2 typical range = 5..=50)"
            )));
        }
        if self.latent_dim == 0 {
            return Err(VokraError::ModelLoad(format!(
                "melodyflow: `{KEY_MELODYFLOW_LATENT_DIM}` = 0 — the DiT \
                 in_proj / out_proj target dim must be ≥ 1 (upstream RVQ \
                 latent width, see the bundled 48 kHz codec)"
            )));
        }
        if self.num_codebooks == 0 {
            return Err(VokraError::ModelLoad(format!(
                "melodyflow: `{KEY_MELODYFLOW_NUM_CODEBOOKS}` = 0 — the \
                 bundled 48 kHz codec must have ≥ 1 RVQ codebook"
            )));
        }
        if self.codebook_size == 0 {
            return Err(VokraError::ModelLoad(format!(
                "melodyflow: `{KEY_MELODYFLOW_CODEBOOK_SIZE}` = 0 — vocab \
                 per codebook must be ≥ 1"
            )));
        }
        if self.codec_frame_rate_hz == 0 {
            return Err(VokraError::ModelLoad(format!(
                "melodyflow: `{KEY_MELODYFLOW_CODEC_FRAME_RATE_HZ}` = 0 — \
                 the bundled 48 kHz codec must emit at least one frame per \
                 second (sample_rate_hz / hop_length)"
            )));
        }
        if self.max_duration_secs == 0 {
            return Err(VokraError::ModelLoad(format!(
                "melodyflow: `{KEY_MELODYFLOW_MAX_DURATION_SECS}` = 0 — \
                 generation horizon must be ≥ 1 s (upstream T24 30secs = 30)"
            )));
        }
        if self.sample_rate_hz == 0 {
            return Err(VokraError::ModelLoad(format!(
                "melodyflow: `{KEY_MELODYFLOW_SAMPLE_RATE_HZ}` = 0 — PCM \
                 output sample rate must be ≥ 1 (upstream T24 = 48000)"
            )));
        }
        if self.text_prefix_len == 0 {
            return Err(VokraError::ModelLoad(format!(
                "melodyflow: `{KEY_MELODYFLOW_TEXT_PREFIX_LEN}` = 0 — text \
                 conditioning prefix must be ≥ 1 token (T5-base default \
                 sequence length is model-specific but non-zero)"
            )));
        }
        if !self.cfg_scale.is_finite() {
            return Err(VokraError::ModelLoad(format!(
                "melodyflow: `{KEY_MELODYFLOW_CFG_SCALE}` = {} — CFG \
                 coefficient must be finite (Le Lan et al. 2024 §4.2 \
                 typical 4.0)",
                self.cfg_scale,
            )));
        }
        Ok(())
    }

    /// Computes the RVQ latent sequence length in codec frames for
    /// the model's configured maximum horizon (`codec_frame_rate_hz *
    /// max_duration_secs`). Uses saturating multiply to avoid overflow
    /// on pathological configs — the validator ensures both operands
    /// are non-zero.
    pub fn max_seq_len(&self) -> u64 {
        u64::from(self.codec_frame_rate_hz).saturating_mul(u64::from(self.max_duration_secs))
    }

    /// Reads config from a parsed GGUF's `vokra.melodyflow.*` chunk
    /// group. The arch tag drives variant dispatch (single variant
    /// today, matches [`ARCH`]); every other key is deserialised
    /// loudly (FR-EX-08 — missing key = hard error, no silent
    /// default).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // Arch dispatch first so a wrong-family GGUF fails with a
        // clear message rather than the downstream "missing metadata".
        let arch = gguf
            .get(chunks::KEY_MODEL_ARCH)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "melodyflow: GGUF is missing `{}` (converter did not stamp it)",
                    chunks::KEY_MODEL_ARCH,
                ))
            })?
            .to_owned();
        if arch != ARCH {
            return Err(VokraError::ModelLoad(format!(
                "melodyflow: GGUF arch is `{arch}`, expected `{ARCH}` — \
                 silently loading a sibling music-gen arch (`magnet_small_10secs` / \
                 `magnet_medium_30secs` / `musicgen_small` / `musicgen_medium` / \
                 `musicgen_large` / `audiogen_medium` / `jasco_400m_chords_drums` / \
                 `audioldm2` / `stable_audio_open_small`) would mis-route the \
                 runtime dispatch to a family with a different decoder loop \
                 (masked-LM vs AR vs different conditioning stack — FR-EX-08)"
            )));
        }

        let num_layers = require_u32(gguf, KEY_MELODYFLOW_NUM_LAYERS)?;
        let hidden_size = require_u32(gguf, KEY_MELODYFLOW_HIDDEN_SIZE)?;
        let num_heads = require_u32(gguf, KEY_MELODYFLOW_NUM_HEADS)?;
        let num_timesteps = require_u32(gguf, KEY_MELODYFLOW_NUM_TIMESTEPS)?;
        let latent_dim = require_u32(gguf, KEY_MELODYFLOW_LATENT_DIM)?;
        let num_codebooks = require_u32(gguf, KEY_MELODYFLOW_NUM_CODEBOOKS)?;
        let codebook_size = require_u32(gguf, KEY_MELODYFLOW_CODEBOOK_SIZE)?;
        let codec_frame_rate_hz = require_u32(gguf, KEY_MELODYFLOW_CODEC_FRAME_RATE_HZ)?;
        let max_duration_secs = require_u32(gguf, KEY_MELODYFLOW_MAX_DURATION_SECS)?;
        let sample_rate_hz = require_u32(gguf, KEY_MELODYFLOW_SAMPLE_RATE_HZ)?;
        let text_prefix_len = require_u32(gguf, KEY_MELODYFLOW_TEXT_PREFIX_LEN)?;
        let cfg_scale = require_f32(gguf, KEY_MELODYFLOW_CFG_SCALE)?;

        let cfg = Self {
            num_layers,
            hidden_size,
            num_heads,
            num_timesteps,
            latent_dim,
            num_codebooks,
            codebook_size,
            codec_frame_rate_hz,
            max_duration_secs,
            sample_rate_hz,
            text_prefix_len,
            cfg_scale,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

fn require_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
    let raw = gguf.get(key).and_then(|v| v.as_u64()).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "melodyflow GGUF missing required u32 metadata `{key}` — the \
             converter does not yet emit `vokra.melodyflow.*` config (BF16 \
             pass-through skeleton only). Extend the converter to stamp \
             this key before loading the GGUF into `MelodyFlowEngine::from_gguf`."
        ))
    })?;
    u32::try_from(raw).map_err(|_| {
        VokraError::ModelLoad(format!(
            "melodyflow GGUF metadata `{key}` = {raw} does not fit in u32"
        ))
    })
}

fn require_f32(gguf: &GgufFile, key: &str) -> Result<f32> {
    let value = gguf.get(key).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "melodyflow GGUF missing required f32 metadata `{key}`"
        ))
    })?;
    // GGUF f32 metadata comes back as GgufMetadataValue::F32 — use the
    // typed accessor rather than as_u64 / as_str (loud FR-EX-08 on
    // wrong-type). Sibling magnet::require_f32 mirrors this.
    if let GgufMetadataValue::F32(v) = value {
        Ok(*v)
    } else {
        Err(VokraError::ModelLoad(format!(
            "melodyflow GGUF metadata `{key}` is not f32 (got {:?})",
            value.value_type()
        )))
    }
}

/// One bound MelodyFlow weight tensor entry (name-only reference; the
/// shell does not preload weights). Same posture as
/// [`crate::magnet::MagnetWeightEntry`],
/// [`crate::dnsmos_p808_p835`], and
/// [`crate::f0::rmvpe`] — the follow-up wave that lights up the
/// runtime forward decides the caching shape based on ADR ratification
/// (`docs/adr/M5-melodyflow-dit-sampler.md`).
#[derive(Debug, Clone)]
pub struct MelodyFlowWeightEntry {
    /// The GGUF tensor name (verbatim upstream HF safetensors key —
    /// see the converter module docstring).
    pub name: String,
    /// Element count for a spot-check the weight is non-empty; a zero
    /// element count would be a converter bug that the loud-partial
    /// forward could not detect on its own.
    pub num_elements: u64,
}

/// MelodyFlow runtime session (SCAFFOLD — see the module docstring).
/// Holds the validated [`MelodyFlowConfig`] and a name-only catalogue
/// of the GGUF's weight tensors. Every inference method is a
/// **loud-partial** [`VokraError::UnsupportedOp`] until the ADR
/// (`docs/adr/M5-melodyflow-dit-sampler.md`) is ratified and the two
/// `vokra-ops` primitives (`flow_editing_inversion` +
/// `t24_transformer`) land.
#[derive(Debug, Clone)]
pub struct MelodyFlowEngine {
    cfg: MelodyFlowConfig,
    weights: Arc<Vec<MelodyFlowWeightEntry>>,
}

impl MelodyFlowEngine {
    /// Binds the model shell from a parsed GGUF (FR-LD-01). Validates
    /// the arch tag against [`ARCH`], deserialises the config,
    /// catalogues weight tensors. A GGUF with no weight tensors is a
    /// hard error (silent-partial forbidden per FR-EX-08 — the future
    /// forward would otherwise integrate against a zero-weight DiT).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let cfg = MelodyFlowConfig::from_gguf(gguf)?;

        let weights: Vec<MelodyFlowWeightEntry> = gguf
            .tensors()
            .iter()
            .map(|t| MelodyFlowWeightEntry {
                name: t.name.clone(),
                num_elements: t.dimensions.iter().product(),
            })
            .collect();

        if weights.is_empty() {
            return Err(VokraError::ModelLoad(
                "melodyflow t24-30secs: GGUF carries zero weight tensors — the \
                 runtime forward would integrate against no weights (FR-EX-08 \
                 forbids silent-partial). Re-run `vokra-cli convert --model \
                 melodyflow-t24` against a real Meta AudioCraft checkpoint."
                    .to_owned(),
            ));
        }

        Ok(Self {
            cfg,
            weights: Arc::new(weights),
        })
    }

    /// Opens and binds the model shell from a GGUF file on disk.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// Returns the validated config.
    pub fn config(&self) -> &MelodyFlowConfig {
        &self.cfg
    }

    /// Returns the (name-only) weight tensor catalogue.
    pub fn weights(&self) -> &[MelodyFlowWeightEntry] {
        &self.weights
    }

    /// **SCAFFOLD — loud-partial.** Runs the MelodyFlow flow-matching
    /// forward integration (text → RVQ latent) over `text_conditioning`,
    /// optionally with `melody_conditioning` for the editing use-case,
    /// returning the final RVQ latent tensor.
    ///
    /// # Signature
    ///
    /// - `text_conditioning` — flattened T5-base text feature vector
    ///   (`text_prefix_len * hidden_size` floats).
    /// - `melody_conditioning` — optional audio latent for the editing
    ///   use-case (source audio RVQ latent, `seq_len * latent_dim`
    ///   floats). `None` for pure text-to-music generation; `Some(_)`
    ///   triggers the reverse-ODE inversion path that requires
    ///   `flow_editing_inversion`.
    /// - `num_solver_steps` — ODE integration step count (typical 24
    ///   for T24; sweep-friendly at runtime per ADR §D-1).
    /// - `cfg_scale` — classifier-free-guidance coefficient.
    ///
    /// # Loud-partial contract (RMVPE / DNSMOS / openwakeword / MAGNeT precedent)
    ///
    /// The current implementation returns
    /// [`VokraError::UnsupportedOp`] naming (a) the ADR to ratify
    /// (`docs/adr/M5-melodyflow-dit-sampler.md`, Status: **Proposed**),
    /// (b) the two `vokra-ops` primitives that need to land
    /// (`flow_editing_inversion` + `t24_transformer` — the FR-OP-86
    /// anchor), (c) the reused `flow_sample` seam from M3-05 (already
    /// exists), and (d) the codec decode integration owner-only path
    /// (ADR §D-5). No fabricated latent tensor, no synthesised DiT
    /// forward — following the precedent set by
    /// [`crate::magnet::MagnetEngine::forward`],
    /// [`crate::dnsmos_p808_p835::Dnsmos::score_p808`], and
    /// [`crate::f0::rmvpe`].
    ///
    /// The arguments are still validated loudly (FR-EX-08 — a caller
    /// with `num_solver_steps = 0` gets an `InvalidArgument`, not a
    /// silent fall-through to `UnsupportedOp`).
    pub fn forward(
        &self,
        text_conditioning: &[f32],
        melody_conditioning: Option<&[f32]>,
        num_solver_steps: usize,
        cfg_scale: f32,
    ) -> Result<Vec<f32>> {
        // Argument validation runs BEFORE the loud-partial so the
        // caller cannot confuse "bad args → wrong output" with
        // "loud-partial gate → no output at all".
        if text_conditioning.is_empty() {
            return Err(VokraError::InvalidArgument(
                "melodyflow.forward: `text_conditioning` is empty — MelodyFlow \
                 is a text-conditioned music model and requires a T5-base \
                 conditioning feature vector (upstream `MelodyFlow.generate` \
                 rejects empty conditioning; silent zero-fill would misrepresent \
                 the run)"
                    .to_owned(),
            ));
        }
        if let Some(mel) = melody_conditioning {
            if mel.is_empty() {
                return Err(VokraError::InvalidArgument(
                    "melodyflow.forward: `melody_conditioning` was Some(&[]) \
                     — an empty conditioning slice is ambiguous between \
                     'no editing (use None)' and 'zero-length audio latent' \
                     (silent misinterpretation forbidden per FR-EX-08). Pass \
                     `None` for pure text-to-music generation."
                        .to_owned(),
                ));
            }
        }
        if num_solver_steps == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "melodyflow.forward: `num_solver_steps` = 0 — ODE integration \
                 requires ≥ 1 step (config default = {}; upstream T24 = 24; \
                 Le Lan et al. 2024 §4.2 typical range = 5..=50)",
                self.cfg.num_timesteps,
            )));
        }
        if !cfg_scale.is_finite() {
            return Err(VokraError::InvalidArgument(format!(
                "melodyflow.forward: `cfg_scale` = {cfg_scale} — must be \
                 finite (config default = {})",
                self.cfg.cfg_scale,
            )));
        }

        Err(dit_sampler_loud_partial(melody_conditioning.is_some()))
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned
/// by [`MelodyFlowEngine::forward`] until the ADR is ratified and the
/// two `vokra-ops` primitives (`flow_editing_inversion` +
/// `t24_transformer`) land.
///
/// The message names the ADR + both ops + the reused `flow_sample`
/// seam from M3-05 + the codec integration owner path so an owner (or
/// a future CC wave) reading this error knows exactly where to flip
/// the switch — no fabricated `Vec<f32>` ever appears (FR-EX-08).
fn dit_sampler_loud_partial(is_editing: bool) -> VokraError {
    let use_case = if is_editing {
        "editing (source audio latent + target text)"
    } else {
        "generation (text-only)"
    };
    VokraError::UnsupportedOp(format!(
        "melodyflow t24-30secs: runtime forward is a SCAFFOLD for the {use_case} \
         path — the model shell (config + weight catalogue) exists in \
         `vokra-models::melodyflow`, but the runtime primitives that drive the \
         DiT flow-matching integrator (`flow_editing_inversion` + \
         `t24_transformer` — the FR-OP-86 anchor) are deferred to a follow-up \
         wave. The regeneration ODE integrator can reuse \
         `vokra_ops::flow_sampler::flow_sample` from M3-05 unchanged (Schedule \
         Linear + OdeSolver Euler + CfgMode DualForward matches Le Lan et al. \
         2024 Algorithm 1), but the reverse-ODE editing inversion driver + the \
         DiT block stack itself are the two new ops. See \
         `docs/adr/M5-melodyflow-dit-sampler.md` (Status: **Proposed**) for the \
         proposed signatures and owner ratification queue; codec decode (the \
         bundled 48 kHz RVQ — FR-OP-32 CC-BY-NC-4.0 distribution restriction) \
         is a separate integration handled per that ADR §D-5. Real weight \
         testing + ADR ratification are prerequisites for the switch flip — no \
         fabricated latents will ever be returned (FR-EX-08)."
    ))
}
