//! **CC-side runtime shell (SCAFFOLD)** for Meta AudioCraft **MAGNeT**
//! (facebook/magnet-small-10secs + facebook/magnet-medium-30secs,
//! CC-BY-NC-4.0) — post-audit CC-gap 2026-08-13 Wave D.
//!
//! # SCAFFOLD status (RMVPE / DNSMOS / openwakeword precedent)
//!
//! This module is deliberately **not** a working MAGNeT runtime. It
//! ships:
//!
//! - `MagnetConfig` — hparams deserialised from GGUF metadata under
//!   `vokra.magnet.*` (num_layers / hidden_size / num_heads / seq_len /
//!   num_codebooks / codebook_size / mask_token_id / top_p / cfg_coef /
//!   num_steps).
//! - `MagnetVariant::{Small10secs, Medium30secs}` — the arch dispatch
//!   tag mirror of the two converter modules (crates/vokra-convert/src/
//!   models/magnet_{small_10secs,medium_30secs}.rs).
//! - `MagnetEngine::from_gguf(&GgufFile)` — validates arch tag,
//!   deserialises config, catalogues the weight tensor names.
//! - `MagnetEngine::forward(...)` — returns
//!   [`VokraError::UnsupportedOp`] naming (a) the ADR to ratify
//!   (`docs/adr/M5-magnet-masked-ar-op.md`, Status: **Proposed**),
//!   (b) the two `vokra-ops` primitives that need to land
//!   (`magnet_masked_decode` + `span_masking_scheduler` — the FR-OP-85
//!   anchor), and (c) the codec decode integration owner-only path
//!   (see the ADR §D-5).
//!
//! There is no synthesised transformer forward, no fabricated
//! `Vec<u32>` output, no silent CPU stub. `MagnetEngine::forward` is
//! **loud-partial** — see `vokra-models::dnsmos_p808_p835::Dnsmos::score_p808`
//! and `vokra-models::f0::rmvpe` for the same posture.
//!
//! # Why this shape (design summary — see the ADR for the full rationale)
//!
//! MAGNeT (Ziv et al. 2024 arXiv:2401.04577) is a
//! **non-autoregressive** music-generation model that decodes RVQ
//! codebook tokens in parallel through iterative masked-LM denoising
//! with a confidence-based span masking schedule. The two runtime-side
//! primitives it needs — `magnet_masked_decode` (parallel masked
//! sampling with a confidence-based masking schedule) and
//! `span_masking_scheduler` (schedule that decides how many positions
//! to unmask in which step) — are runtime functions under
//! [`vokra_ops`], NOT `OpKind` variants, following the M3-05
//! `flow_sampler` / M3-06 `mimi_rvq` / M4-04 `dac_rvq` /
//! openwakeword classifier precedent (`docs/adr/M3-05-flow-sampler.md`
//! §D1 / `docs/adr/M3-06-mimi-rvq.md` §D-b). Landing them today would
//! also violate ADR M4-20 §D-1 (trigger-backed subset rule — the C ABI
//! freeze IF-01 = M5-13 is coming; no live consumer today).
//!
//! # Codec decode is owner-driven
//!
//! MAGNeT bundles a 32 kHz EnCodec decoder. EnCodec pretrained weights
//! are CC-BY-NC-4.0 (permanent FR-OP-32 distribution restriction — see
//! ADR M3-06 §D-2). The codec decode step is deliberately outside this
//! module — a future integration will either (Option A) reuse the
//! existing `vokra_ops::encodec_rvq` engine op against the bundled
//! EnCodec weights (keeps T4 = research-only) or (Option B) substitute
//! a permissively-licensed codec at 32 kHz (requires retraining
//! evidence — see the ADR §D-5). Neither is CC-scope today.
//!
//! # Distinct arch tag (FR-EX-08)
//!
//! Two arch tags:
//!
//! - [`ARCH_SMALL`] = `"magnet_small_10secs"` (500 M params, 10-second
//!   horizon)
//! - [`ARCH_MEDIUM`] = `"magnet_medium_30secs"` (1.5 B params, 30-second
//!   horizon — same non-AR masked-LM op path, wider hidden / more
//!   layers)
//!
//! Silent-sharing an arch tag with a sibling music-gen family
//! (`musicgen_{small,medium,large}` / `audiogen_medium` /
//! `jasco_400m_chords_drums` / `audioldm2` / `stable_audio_open_small` /
//! `ace_step` / `bs_roformer`) is an FR-EX-08 violation because the
//! runtime dispatch would mis-route to a token-by-token AR loop with no
//! `mask` token semantics.

use std::sync::Arc;

use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{Result, VokraError};

#[cfg(test)]
mod tests;

// -----------------------------------------------------------------------------
// arch tags — mirror of the two converter modules
// (crates/vokra-convert/src/models/magnet_{small_10secs,medium_30secs}.rs
// `pub const ARCH`). Duplicated here (same convention every fsmn_vad /
// silero_vad / dnsmos / openwakeword binder uses) so the runtime crate does
// not need a cross-crate dependency edge onto the converter.
// -----------------------------------------------------------------------------

/// `vokra.model.arch` value for MAGNeT Small 10 secs GGUFs
/// (500 M params, 10-second music generation horizon). Mirror of
/// `vokra-convert::models::magnet_small_10secs::ARCH`.
pub const ARCH_SMALL: &str = "magnet_small_10secs";

/// `vokra.model.arch` value for MAGNeT Medium 30 secs GGUFs
/// (1.5 B params, 30-second music generation horizon — same non-AR
/// masked-LM op path as [`ARCH_SMALL`], different hparam set).
/// Mirror of `vokra-convert::models::magnet_medium_30secs::ARCH`.
pub const ARCH_MEDIUM: &str = "magnet_medium_30secs";

// -----------------------------------------------------------------------------
// vokra.magnet.* metadata keys — proposed schema for the future converter
// extension. The current converter (`convert_magnet_small_10secs_file` /
// `convert_magnet_medium_30secs_file`) does NOT emit these keys yet — it is
// a BF16 pass-through skeleton only. `MagnetEngine::from_gguf` therefore
// fails loudly (FR-EX-08) on a GGUF that does not carry them — no silent
// default (Ziv et al. 2024 §5 sweeps num_steps + top_p + cfg_coef at
// inference; a silently-defaulted config would misrepresent the run).
// -----------------------------------------------------------------------------

/// GGUF metadata key: transformer LM depth (u32).
pub const KEY_MAGNET_NUM_LAYERS: &str = "vokra.magnet.num_layers";
/// GGUF metadata key: transformer LM hidden width (u32).
pub const KEY_MAGNET_HIDDEN_SIZE: &str = "vokra.magnet.hidden_size";
/// GGUF metadata key: transformer LM attention head count (u32).
pub const KEY_MAGNET_NUM_HEADS: &str = "vokra.magnet.num_heads";
/// GGUF metadata key: sequence length in codec frames (u32) —
/// `sample_rate_hz * generation_seconds / codec_hop`. For MAGNeT small
/// (10 s @ 50 Hz EnCodec) = 500; medium (30 s @ 50 Hz) = 1500.
pub const KEY_MAGNET_SEQ_LEN: &str = "vokra.magnet.seq_len";
/// GGUF metadata key: number of RVQ codebooks in the codec stream
/// (u32, typically 4 for MAGNeT).
pub const KEY_MAGNET_NUM_CODEBOOKS: &str = "vokra.magnet.num_codebooks";
/// GGUF metadata key: vocabulary size per codebook (u32, typically
/// 2048).
pub const KEY_MAGNET_CODEBOOK_SIZE: &str = "vokra.magnet.codebook_size";
/// GGUF metadata key: MASK sentinel token id (u32). Upstream MAGNeT
/// uses `codebook_size` as the mask sentinel; kept as an explicit
/// attribute so a future non-Meta variant can override it (no silent
/// default per FR-EX-08).
pub const KEY_MAGNET_MASK_TOKEN_ID: &str = "vokra.magnet.mask_token_id";
/// GGUF metadata key: default nucleus sampling probability (f32).
pub const KEY_MAGNET_TOP_P: &str = "vokra.magnet.top_p";
/// GGUF metadata key: default classifier-free-guidance coefficient (f32).
pub const KEY_MAGNET_CFG_COEF: &str = "vokra.magnet.cfg_coef";
/// GGUF metadata key: default number of masked-LM decoding steps
/// (u32). Ziv et al. 2024 §5 typical 20.
pub const KEY_MAGNET_NUM_STEPS: &str = "vokra.magnet.num_steps";

/// The two MAGNeT variants CC currently supports (mirror of the two
/// converter modules). Kept as an enum rather than a slug string so
/// `match` arms are exhaustive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MagnetVariant {
    /// facebook/magnet-small-10secs — 500 M params, 10-second horizon.
    Small10secs,
    /// facebook/magnet-medium-30secs — 1.5 B params, 30-second horizon.
    Medium30secs,
}

impl MagnetVariant {
    /// Canonical arch tag string (matches [`ARCH_SMALL`] /
    /// [`ARCH_MEDIUM`]).
    pub const fn arch(&self) -> &'static str {
        match self {
            Self::Small10secs => ARCH_SMALL,
            Self::Medium30secs => ARCH_MEDIUM,
        }
    }

    /// Canonical short display name for logs / errors.
    pub const fn short(&self) -> &'static str {
        match self {
            Self::Small10secs => "magnet-small-10secs",
            Self::Medium30secs => "magnet-medium-30secs",
        }
    }
}

/// MAGNeT runtime configuration (transcribed verbatim from
/// `vokra.magnet.*` at load time; every field is required and
/// validated loudly per FR-EX-08).
#[derive(Debug, Clone, PartialEq)]
pub struct MagnetConfig {
    /// Which variant this GGUF represents (from `vokra.model.arch`).
    pub variant: MagnetVariant,
    /// Transformer LM depth.
    pub num_layers: u32,
    /// Transformer LM hidden width.
    pub hidden_size: u32,
    /// Transformer LM attention head count.
    pub num_heads: u32,
    /// Sequence length in codec frames (`sample_rate_hz *
    /// generation_seconds / codec_hop`).
    pub seq_len: u32,
    /// Number of RVQ codebooks in the codec stream.
    pub num_codebooks: u32,
    /// Vocabulary size per codebook.
    pub codebook_size: u32,
    /// MASK sentinel token id.
    pub mask_token_id: u32,
    /// Default nucleus sampling probability.
    pub top_p: f32,
    /// Default classifier-free-guidance coefficient.
    pub cfg_coef: f32,
    /// Default number of masked-LM decoding steps.
    pub num_steps: u32,
}

impl MagnetConfig {
    /// Validates the config loudly (FR-EX-08 — no silent zero /
    /// silent default). Called at the end of [`Self::from_gguf`].
    pub fn validate(&self) -> Result<()> {
        if self.num_layers == 0 {
            return Err(VokraError::ModelLoad(format!(
                "magnet: `{KEY_MAGNET_NUM_LAYERS}` = 0 — transformer LM must have \
                 at least one layer (upstream small = 24, medium = 48)"
            )));
        }
        if self.hidden_size == 0 || self.hidden_size % self.num_heads != 0 {
            return Err(VokraError::ModelLoad(format!(
                "magnet: `{KEY_MAGNET_HIDDEN_SIZE}` = {} not divisible by \
                 `{KEY_MAGNET_NUM_HEADS}` = {} (head_dim must be an integer)",
                self.hidden_size, self.num_heads,
            )));
        }
        if self.num_heads == 0 {
            return Err(VokraError::ModelLoad(format!(
                "magnet: `{KEY_MAGNET_NUM_HEADS}` = 0 — must be ≥ 1"
            )));
        }
        if self.seq_len == 0 {
            return Err(VokraError::ModelLoad(format!(
                "magnet: `{KEY_MAGNET_SEQ_LEN}` = 0 — generation horizon must \
                 be ≥ 1 codec frame (upstream small = 500 @ 50 Hz for 10 s, \
                 medium = 1500 for 30 s)"
            )));
        }
        if self.num_codebooks == 0 {
            return Err(VokraError::ModelLoad(format!(
                "magnet: `{KEY_MAGNET_NUM_CODEBOOKS}` = 0 — RVQ stream must \
                 have ≥ 1 codebook (upstream = 4)"
            )));
        }
        if self.codebook_size == 0 {
            return Err(VokraError::ModelLoad(format!(
                "magnet: `{KEY_MAGNET_CODEBOOK_SIZE}` = 0 — vocab per \
                 codebook must be ≥ 1 (upstream = 2048)"
            )));
        }
        if !self.top_p.is_finite() || !(0.0..=1.0).contains(&self.top_p) {
            return Err(VokraError::ModelLoad(format!(
                "magnet: `{KEY_MAGNET_TOP_P}` = {} — nucleus sampling \
                 probability must be finite ∈ [0.0, 1.0] (upstream typical 0.9)",
                self.top_p,
            )));
        }
        if !self.cfg_coef.is_finite() {
            return Err(VokraError::ModelLoad(format!(
                "magnet: `{KEY_MAGNET_CFG_COEF}` = {} — CFG coefficient must \
                 be finite (upstream typical 3.0–10.0)",
                self.cfg_coef,
            )));
        }
        if self.num_steps == 0 {
            return Err(VokraError::ModelLoad(format!(
                "magnet: `{KEY_MAGNET_NUM_STEPS}` = 0 — masked-LM decoding \
                 requires ≥ 1 step (upstream typical 20)"
            )));
        }
        Ok(())
    }

    /// Reads config from a parsed GGUF's `vokra.magnet.*` chunk group.
    /// The arch tag drives variant dispatch; every other key is
    /// deserialised loudly (FR-EX-08 — missing key = hard error, no
    /// silent default).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // Arch dispatch first so a wrong-family GGUF fails with a
        // clear message rather than the downstream "missing metadata".
        let arch = gguf
            .get(chunks::KEY_MODEL_ARCH)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "magnet: GGUF is missing `{}` (converter did not stamp it)",
                    chunks::KEY_MODEL_ARCH,
                ))
            })?
            .to_owned();
        let variant = match arch.as_str() {
            ARCH_SMALL => MagnetVariant::Small10secs,
            ARCH_MEDIUM => MagnetVariant::Medium30secs,
            other => {
                return Err(VokraError::ModelLoad(format!(
                    "magnet: GGUF arch is `{other}`, expected `{ARCH_SMALL}` or \
                     `{ARCH_MEDIUM}` — silently loading a sibling music-gen arch \
                     (`musicgen_small` / `musicgen_medium` / `musicgen_large` / \
                     `audiogen_medium` / `jasco_400m_chords_drums` / `audioldm2` \
                     / `stable_audio_open_small`) would mis-route the runtime \
                     dispatch to a token-by-token AR loop with no `mask` token \
                     semantics (FR-EX-08)"
                )));
            }
        };

        let num_layers = require_u32(gguf, KEY_MAGNET_NUM_LAYERS)?;
        let hidden_size = require_u32(gguf, KEY_MAGNET_HIDDEN_SIZE)?;
        let num_heads = require_u32(gguf, KEY_MAGNET_NUM_HEADS)?;
        let seq_len = require_u32(gguf, KEY_MAGNET_SEQ_LEN)?;
        let num_codebooks = require_u32(gguf, KEY_MAGNET_NUM_CODEBOOKS)?;
        let codebook_size = require_u32(gguf, KEY_MAGNET_CODEBOOK_SIZE)?;
        let mask_token_id = require_u32(gguf, KEY_MAGNET_MASK_TOKEN_ID)?;
        let top_p = require_f32(gguf, KEY_MAGNET_TOP_P)?;
        let cfg_coef = require_f32(gguf, KEY_MAGNET_CFG_COEF)?;
        let num_steps = require_u32(gguf, KEY_MAGNET_NUM_STEPS)?;

        let cfg = Self {
            variant,
            num_layers,
            hidden_size,
            num_heads,
            seq_len,
            num_codebooks,
            codebook_size,
            mask_token_id,
            top_p,
            cfg_coef,
            num_steps,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

fn require_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
    let raw = gguf.get(key).and_then(|v| v.as_u64()).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "magnet GGUF missing required u32 metadata `{key}` — the converter \
             does not yet emit `vokra.magnet.*` config (BF16 pass-through \
             skeleton only). Extend the converter to stamp this key before \
             loading the GGUF into `MagnetEngine::from_gguf`."
        ))
    })?;
    u32::try_from(raw).map_err(|_| {
        VokraError::ModelLoad(format!(
            "magnet GGUF metadata `{key}` = {raw} does not fit in u32"
        ))
    })
}

fn require_f32(gguf: &GgufFile, key: &str) -> Result<f32> {
    let value = gguf.get(key).ok_or_else(|| {
        VokraError::ModelLoad(format!("magnet GGUF missing required f32 metadata `{key}`"))
    })?;
    // GGUF f32 metadata comes back as GgufMetadataValue::F32 — use the
    // typed accessor rather than as_u64 / as_str (loud FR-EX-08 on
    // wrong-type).
    if let GgufMetadataValue::F32(v) = value {
        Ok(*v)
    } else {
        Err(VokraError::ModelLoad(format!(
            "magnet GGUF metadata `{key}` is not f32 (got {:?})",
            value.value_type()
        )))
    }
}

/// One bound MAGNeT weight tensor entry (name-only reference; the shell
/// does not preload weights). Same posture as
/// `dnsmos_p808_p835::DnsmosBundle` and
/// `f0::rmvpe::RmvpeWeights` — the follow-up wave that lights up the
/// runtime forward decides the caching shape based on ADR
/// ratification (`docs/adr/M5-magnet-masked-ar-op.md` §D-4).
#[derive(Debug, Clone)]
pub struct MagnetWeightEntry {
    /// The GGUF tensor name (verbatim upstream HF safetensors key —
    /// see the converter module docstring).
    pub name: String,
    /// Element count for a spot-check the weight is non-empty; a zero
    /// element count would be a converter bug that the loud-partial
    /// forward could not detect on its own.
    pub num_elements: u64,
}

/// MAGNeT runtime session (SCAFFOLD — see the module docstring). Holds
/// the validated [`MagnetConfig`] and a name-only catalogue of the
/// GGUF's weight tensors. Every inference method is a **loud-partial**
/// [`VokraError::UnsupportedOp`] until the ADR
/// (`docs/adr/M5-magnet-masked-ar-op.md`) is ratified and the two
/// `vokra-ops` primitives (`magnet_masked_decode` +
/// `span_masking_scheduler`) land.
#[derive(Debug, Clone)]
pub struct MagnetEngine {
    cfg: MagnetConfig,
    weights: Arc<Vec<MagnetWeightEntry>>,
}

impl MagnetEngine {
    /// Binds the model shell from a parsed GGUF (FR-LD-01). Validates
    /// the arch tag against [`ARCH_SMALL`] / [`ARCH_MEDIUM`],
    /// deserialises the config, catalogues weight tensors. A GGUF with
    /// no weight tensors is a hard error (silent-partial forbidden per
    /// FR-EX-08 — the future forward would otherwise decode against a
    /// zero-weight LM).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let cfg = MagnetConfig::from_gguf(gguf)?;

        let weights: Vec<MagnetWeightEntry> = gguf
            .tensors()
            .iter()
            .map(|t| MagnetWeightEntry {
                name: t.name.clone(),
                num_elements: t.dimensions.iter().product(),
            })
            .collect();

        if weights.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "magnet {}: GGUF carries zero weight tensors — the runtime \
                 forward would decode against no weights (FR-EX-08 forbids \
                 silent-partial). Re-run `vokra-cli convert --model magnet-{}` \
                 against a real Meta AudioCraft checkpoint.",
                cfg.variant.short(),
                if cfg.variant == MagnetVariant::Small10secs {
                    "small-10secs"
                } else {
                    "medium-30secs"
                },
            )));
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
    pub fn config(&self) -> &MagnetConfig {
        &self.cfg
    }

    /// Returns the (name-only) weight tensor catalogue.
    pub fn weights(&self) -> &[MagnetWeightEntry] {
        &self.weights
    }

    /// **SCAFFOLD — loud-partial.** Runs the MAGNeT non-autoregressive
    /// masked-LM decoding loop over `text_conditioning`, returning the
    /// final RVQ codebook token stream (`Vec<u32>` of length
    /// `seq_len * num_codebooks`).
    ///
    /// # Loud-partial contract (RMVPE / DNSMOS / openwakeword precedent)
    ///
    /// The current implementation returns
    /// [`VokraError::UnsupportedOp`] naming (a) the ADR to ratify
    /// (`docs/adr/M5-magnet-masked-ar-op.md`, Status: **Proposed**),
    /// (b) the two `vokra-ops` primitives that need to land
    /// (`magnet_masked_decode` + `span_masking_scheduler` — the
    /// FR-OP-85 anchor), and (c) the codec decode integration
    /// owner-only path (ADR §D-5). No fabricated codebook stream, no
    /// synthesised transformer forward — following the precedent set
    /// by [`crate::dnsmos_p808_p835::Dnsmos::score_p808`] and
    /// [`crate::f0::rmvpe`].
    ///
    /// The arguments are still validated loudly (FR-EX-08 — a caller
    /// with bad `num_steps` = 0 gets an `InvalidArgument`, not a
    /// silent fall-through to `UnsupportedOp`).
    pub fn forward(
        &self,
        text_conditioning: &[f32],
        num_steps: usize,
        temperature: f32,
        top_p: f32,
        cfg_coef: f32,
    ) -> Result<Vec<u32>> {
        // Argument validation runs BEFORE the loud-partial so the
        // caller cannot confuse "bad args → wrong tokens" with
        // "loud-partial gate → no output at all".
        if text_conditioning.is_empty() {
            return Err(VokraError::InvalidArgument(
                "magnet.forward: `text_conditioning` is empty — MAGNeT is a \
                 text-to-music model and requires a T5 conditioning feature \
                 vector (upstream `MagnetLMModel.generate` rejects empty \
                 conditioning; silent zero-fill would misrepresent the run)"
                    .to_owned(),
            ));
        }
        if num_steps == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "magnet.forward: `num_steps` = 0 — masked-LM decoding requires \
                 ≥ 1 step (config default = {}; upstream typical = 20)",
                self.cfg.num_steps,
            )));
        }
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "magnet.forward: `temperature` = {temperature} — must be finite \
                 ∈ [0.0, ∞) (a negative temperature would flip the softmax sign; \
                 silent clamp forbidden per FR-EX-08)"
            )));
        }
        if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
            return Err(VokraError::InvalidArgument(format!(
                "magnet.forward: `top_p` = {top_p} — must be finite ∈ [0.0, 1.0] \
                 (config default = {})",
                self.cfg.top_p,
            )));
        }
        if !cfg_coef.is_finite() {
            return Err(VokraError::InvalidArgument(format!(
                "magnet.forward: `cfg_coef` = {cfg_coef} — must be finite \
                 (config default = {})",
                self.cfg.cfg_coef,
            )));
        }

        Err(masked_decode_loud_partial(self.cfg.variant))
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned
/// by [`MagnetEngine::forward`] until the ADR is ratified and the two
/// `vokra-ops` primitives (`magnet_masked_decode` +
/// `span_masking_scheduler`) land.
///
/// The message names the ADR + both ops + the follow-up wave contract
/// so an owner (or a future CC wave) reading this error knows exactly
/// where to flip the switch — no fabricated `Vec<u32>` ever appears
/// (FR-EX-08).
fn masked_decode_loud_partial(variant: MagnetVariant) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "magnet {short}: runtime forward is a SCAFFOLD — the model shell \
         (config + weight catalogue) exists in `vokra-models::magnet`, but \
         the runtime primitives that drive the non-autoregressive masked-LM \
         decoding loop (`magnet_masked_decode` + `span_masking_scheduler` — \
         the FR-OP-85 anchor) are deferred to a follow-up wave. See \
         `docs/adr/M5-magnet-masked-ar-op.md` (Status: **Proposed**) for the \
         proposed signatures and the owner ratification queue; codec decode \
         (the bundled 32 kHz EnCodec — FR-OP-32 CC-BY-NC-4.0 distribution \
         restriction) is a separate integration handled per that ADR §D-5. \
         Real weight testing + ADR ratification are prerequisites for the \
         switch flip — no fabricated tokens will ever be returned (FR-EX-08).",
        short = variant.short(),
    ))
}
