//! Continuous VAE encoder / decoder primitives (SoTA plan Phase 4 TTS
//! foundation).
//!
//! # What this module is
//!
//! A shared scaffold for the **continuous-latent** audio VAE families that
//! newer text-to-speech models pair with a diffusion / flow-matching feature
//! generator instead of a discrete RVQ codebook lookup:
//!
//! - **VoxCPM-0.5B** (OpenBMB, Apache-2.0) — the first consumer. Encoder
//!   downsamples PCM to a `[T', latent_dim]` continuous stream at
//!   `patch_size` per LM step (Phase 4 primary target).
//! - **VibeVoice** (long-form multi-speaker TTS, planned) — expected to
//!   share this primitive.
//!
//! The op is a **runtime function**, not an [`vokra_core::OpKind`] variant —
//! same posture as [`crate::flow_sampler`] / [`crate::mimi_rvq`] /
//! [`crate::dac_rvq`] / [`crate::qwen3_tts_codec`] (FR-OP-30 / FR-EX-10 / ADR
//! M3-06 §D-b). Embedding the shape / stride / activation choices in an
//! `OpKind` variant would force a model re-conversion each time a caller
//! swaps a strides list or an activation kind, which is precisely the
//! configuration point sibling models change most often.
//!
//! # Distinct from every existing codec op in this crate (why a new module)
//!
//! - [`crate::mimi_rvq`] / [`crate::dac_rvq`] / [`crate::encodec_rvq`] /
//!   [`crate::snac_decode`] — **discrete RVQ / hierarchical RVQ**: the LM
//!   emits per-quantizer *integer indices*, and decode is
//!   `codebook_lookup + out_proj + sum`. There is no scalar continuous
//!   latent path in that family.
//! - [`crate::fsq_codec`] (`wavtokenizer_vq`, `xcodec2_fsq`) — **finite
//!   scalar quantization**: the encode/decode is a factorized *discrete*
//!   index → grid → codeword bridge; the LM still emits a single integer
//!   per frame. The `scalar_quantization_scale` axis VoxCPM carries is a
//!   *quantization bottleneck inside the LM's own hidden stream* (a
//!   [`ScalarQuantizationLayer`](https://github.com/OpenBMB/VoxCPM/blob/main/src/voxcpm/modules/layers/__init__.py)
//!   projection with an FSQ constraint) — orthogonal to the VAE.
//! - [`crate::hifigan`] / [`crate::hiftnet`] / [`crate::bigvgan_generator`]
//!   — **vocoder-style upsamplers**: consume a mel spectrogram (or
//!   compatible feature) and emit PCM. They do not carry an encoder half
//!   and they do not consume the continuous VAE latent stream directly.
//!
//! The continuous VAE is a first-class new op family because it introduces
//! **factorized `mu` + `logvar` heads on the encoder** and a decoder that
//! runs at the LM step rate, not the codec frame rate — a topology no
//! existing op in this crate expresses.
//!
//! # Shape contract
//!
//! For a VoxCPM-flavour VAE:
//!
//! - **Encoder input** (`encode`): `[T]` mono PCM at the encoder sample
//!   rate (`sample_rate_hz`), or `[1, 1, T]` when a caller wants an
//!   explicit `[B=1, C=1, T]` shape (both shapes produce the same
//!   `[latent_dim, T']` output — the encoder unwraps a trailing 1-D
//!   accidentally passed by callers who mirror upstream's PyTorch
//!   `[B, 1, T]`).
//! - **Encoder output** (`encode`): `[T', latent_dim]` row-major `f32`
//!   continuous mean vectors (the encoder's `mu` head). `T' =
//!   ceil(T / hop_length)` where `hop_length = product(encoder_rates)`.
//!   Downstream callers (VoxCPM: [`crate::flow_sampler`] + upstream local
//!   DiT) treat these as flow-matching conditioning `mu` vectors — they
//!   never round-trip them through a discrete index.
//! - **Decoder input** (`decode`): `[T', latent_dim]` continuous latents.
//! - **Decoder output** (`decode`): `[T_out]` mono PCM at the decoder
//!   sample rate (`out_sample_rate_hz`).
//!
//! Byte layout matches [`crate::mimi_rvq`] / [`crate::dac_rvq`]
//! (row-major, contiguous frames) so a caller can hand the encoder output
//! straight to the shared [`crate::flow_sampler`] state without an extra
//! transpose.
//!
//! # No silent fallback (FR-EX-08)
//!
//! Zero-length inputs, mis-shaped weight stores, and every allocation
//! failure surface as [`VokraError::InvalidArgument`] rather than a silent
//! clamp / zero fill. Weight-store binding is a follow-up wave: today
//! [`ContinuousVaeDecoder::decode`] / [`ContinuousVaeEncoder::encode`]
//! return [`VokraError::NotImplemented`] naming the blocker (the
//! neural-chain forward path — Snake + causal Conv1d + weight-norm folded
//! residual units — is transcribed from upstream `audio_vae_v2.py` and
//! ports in the T29-equivalent follow-up wave).
//!
//! # No SIMD / no unsafe
//!
//! Everything in the scaffold is safe Rust; the SIMD hot path lands with
//! the neural chain in the follow-up wave (the encoder / decoder blocks
//! reuse the same Snake / weight-norm Conv1d kernels the M3-07 HiFi-GAN
//! generator already ships in this crate).

use vokra_core::{Result, VokraError};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Architectural hparams shared by every continuous-VAE encoder / decoder
/// in this family.
///
/// Transcription convention mirrors the codec configs
/// ([`crate::mimi_rvq::MimiRvqAttrs`] / [`crate::dac_rvq::DacRvqAttrs`]):
/// every axis is a **primary-source constant** the caller supplies at
/// construction time; the op never invents an axis from tensor shapes and
/// fails loudly (FR-EX-08) whenever the two disagree.
///
/// For VoxCPM-0.5B the canonical fill lives in
/// [`ContinuousVaeConfig::voxcpm_0_5b`] and is transcribed from the
/// upstream `openbmb/VoxCPM/src/voxcpm/modules/audiovae/audio_vae_v2.py`
/// `AudioVAEConfig` defaults (fetched 2026-07-24 — CLAUDE.md
/// 「ハルシネーション厳禁」).
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuousVaeConfig {
    /// Encoder input PCM sample rate (Hz). VoxCPM-0.5B: `16_000`.
    pub sample_rate_hz: u32,
    /// Decoder output PCM sample rate (Hz). VoxCPM-0.5B: `48_000` (the
    /// upstream AudioVAE decoder upsamples during synthesis).
    pub out_sample_rate_hz: u32,
    /// Encoder base channel count (upstream `encoder_dim`).
    /// VoxCPM-0.5B: `128`.
    pub encoder_dim: u32,
    /// Encoder stride list (`encoder_rates`). Product =
    /// [`Self::hop_length`]. VoxCPM-0.5B: `[2, 5, 8, 8]` (hop 640 =
    /// `16_000 / 25 Hz` frames).
    pub encoder_rates: Vec<u32>,
    /// Latent dimension emitted per encoder step (upstream `latent_dim`).
    /// This is the width of the continuous `mu` vector the decoder /
    /// downstream flow-matching sampler consume. VoxCPM-0.5B: `64`.
    pub latent_dim: u32,
    /// Decoder base channel count (upstream `decoder_dim`).
    /// VoxCPM-0.5B: `2048`.
    pub decoder_dim: u32,
    /// Decoder stride list (upstream `decoder_rates`). Product =
    /// [`Self::decode_hop_length`]. VoxCPM-0.5B: `[8, 6, 5, 2, 2, 2]`
    /// (decode hop 1920 = `48_000 / 25 Hz` frames).
    pub decoder_rates: Vec<u32>,
    /// Whether the causal Conv1d / weight-norm layers use depthwise
    /// separation. VoxCPM-0.5B: `true`.
    pub depthwise: bool,
    /// Whether the decoder blocks are augmented with a per-frame noise
    /// projection. VoxCPM-0.5B: `false`.
    pub use_noise_block: bool,
}

impl ContinuousVaeConfig {
    /// Canonical VoxCPM-0.5B `AudioVAE V2` config.
    ///
    /// Primary source: `openbmb/VoxCPM/src/voxcpm/modules/audiovae/audio_vae_v2.py`
    /// `class AudioVAEConfig(BaseModel)` block (fetched 2026-07-24) —
    /// every field on that Pydantic model is transcribed verbatim.
    ///
    /// Note: upstream carries additional sample-rate-conditioning fields
    /// (`sr_bin_boundaries`, `cond_type`, `cond_dim`, `cond_out_layer`)
    /// that are training-side switches; the runtime honours them via
    /// weight-tensor shapes at load time (T29-equivalent), not through
    /// this config axis today.
    #[must_use]
    pub fn voxcpm_0_5b() -> Self {
        Self {
            sample_rate_hz: 16_000,
            out_sample_rate_hz: 48_000,
            encoder_dim: 128,
            encoder_rates: vec![2, 5, 8, 8],
            latent_dim: 64,
            decoder_dim: 2048,
            decoder_rates: vec![8, 6, 5, 2, 2, 2],
            depthwise: true,
            use_noise_block: false,
        }
    }

    /// A miniature well-formed config for shape / stability tests.
    ///
    /// Every ratio / structural axis mirrors the real config; only
    /// magnitudes shrink so synthesized-weight builds fit in KB.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            sample_rate_hz: 16_000,
            out_sample_rate_hz: 48_000,
            encoder_dim: 8,
            encoder_rates: vec![2, 2],
            latent_dim: 4,
            decoder_dim: 32,
            decoder_rates: vec![2, 2, 2],
            depthwise: false,
            use_noise_block: false,
        }
    }

    /// Total encoder temporal downsampling factor
    /// (`product(encoder_rates)`). PCM samples per encoded frame.
    ///
    /// Returns `None` when the product overflows `u32` — an obvious
    /// misconfiguration (a well-formed release keeps the product well
    /// under a million).
    #[must_use]
    pub fn hop_length(&self) -> Option<u32> {
        self.encoder_rates
            .iter()
            .try_fold(1u32, |acc, r| acc.checked_mul(*r))
    }

    /// Total decoder temporal upsampling factor
    /// (`product(decoder_rates)`). Output PCM samples per decoded frame.
    ///
    /// Returns `None` when the product overflows `u32`.
    #[must_use]
    pub fn decode_hop_length(&self) -> Option<u32> {
        self.decoder_rates
            .iter()
            .try_fold(1u32, |acc, r| acc.checked_mul(*r))
    }

    /// Encoded frame rate (Hz) — `sample_rate_hz / hop_length`.
    /// VoxCPM-0.5B: `25 Hz`.
    ///
    /// Returns `None` when [`Self::hop_length`] does or when
    /// `hop_length` is zero.
    #[must_use]
    pub fn frame_rate_hz(&self) -> Option<f32> {
        let hop = self.hop_length()?;
        if hop == 0 {
            return None;
        }
        Some(self.sample_rate_hz as f32 / hop as f32)
    }

    /// Rejects zeroed / ill-formed configs at binding time (FR-EX-08 —
    /// never a silent zero-fill).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate_for_forward(&self) -> Result<()> {
        if self.sample_rate_hz == 0 || self.out_sample_rate_hz == 0 {
            return Err(VokraError::InvalidArgument(
                "vae_continuous config: sample_rate_hz / out_sample_rate_hz must be > 0".to_owned(),
            ));
        }
        if self.encoder_dim == 0 || self.decoder_dim == 0 || self.latent_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "vae_continuous config: encoder_dim / decoder_dim / latent_dim must be > 0"
                    .to_owned(),
            ));
        }
        if self.encoder_rates.is_empty() {
            return Err(VokraError::InvalidArgument(
                "vae_continuous config: encoder_rates must not be empty".to_owned(),
            ));
        }
        if self.decoder_rates.is_empty() {
            return Err(VokraError::InvalidArgument(
                "vae_continuous config: decoder_rates must not be empty".to_owned(),
            ));
        }
        for (i, r) in self.encoder_rates.iter().enumerate() {
            if *r == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "vae_continuous config: encoder_rates[{i}] must be > 0"
                )));
            }
        }
        for (i, r) in self.decoder_rates.iter().enumerate() {
            if *r == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "vae_continuous config: decoder_rates[{i}] must be > 0"
                )));
            }
        }
        // Product-overflow / frame-rate sanity — a well-formed release
        // keeps every product well under a million.
        self.hop_length().ok_or_else(|| {
            VokraError::InvalidArgument(
                "vae_continuous config: encoder_rates product overflows u32".to_owned(),
            )
        })?;
        self.decode_hop_length().ok_or_else(|| {
            VokraError::InvalidArgument(
                "vae_continuous config: decoder_rates product overflows u32".to_owned(),
            )
        })?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Weight-store scaffolds
// ---------------------------------------------------------------------------

/// Continuous VAE encoder weights (scaffold — real binding is a
/// follow-up wave).
///
/// The upstream `CausalEncoder` (audio_vae_v2.py L125-159) is a chain of
/// weight-normalized causal Conv1d layers grouped into `CausalEncoderBlock`
/// stages (each stage: three dilated residual units + a strided Conv1d
/// downsample), terminated by two 1x1 `WNConv1d` heads (`fc_mu` and
/// `fc_logvar`). The scaffold carries the *aggregate* tensor bundles a
/// real binding will populate; per-block indexing is a follow-up wave
/// concern (kept intentionally out of this seam to avoid pinning a name
/// contract before the T29 tensor-name manifest is ratified).
#[derive(Debug, Clone)]
pub struct ContinuousVaeEncoderWeights {
    /// Flat concatenation of every stage's residual + downsample tensor
    /// bytes, in upstream traversal order. `is_synthesized = true` for a
    /// deterministic fixture; a real binding walks the same slot with
    /// the upstream `encoder.block.*.block.*` naming (T29-equivalent).
    pub blocks: Vec<f32>,
    /// `[latent_dim, enc_out_channels]` for the `fc_mu` head.
    pub fc_mu_weight: Vec<f32>,
    /// `[latent_dim]` bias for `fc_mu`.
    pub fc_mu_bias: Vec<f32>,
    /// `[latent_dim, enc_out_channels]` for the `fc_logvar` head.
    pub fc_logvar_weight: Vec<f32>,
    /// `[latent_dim]` bias for `fc_logvar`.
    pub fc_logvar_bias: Vec<f32>,
    /// `true` when built by [`Self::synthesized`] — never a real upstream
    /// checkpoint.
    pub is_synthesized: bool,
}

impl ContinuousVaeEncoderWeights {
    /// Deterministic zero-initialized fixture (shape scaffold only).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `config.validate_for_forward`
    /// fails.
    pub fn synthesized(config: &ContinuousVaeConfig) -> Result<Self> {
        config.validate_for_forward()?;
        // Encoder terminal channel width doubles per stage (upstream
        // `d_model *= 2` in `CausalEncoder.__init__`).
        let enc_out = (config.encoder_dim as usize) * (1 << config.encoder_rates.len());
        let latent = config.latent_dim as usize;
        Ok(Self {
            blocks: Vec::new(),
            fc_mu_weight: vec![0.0; latent * enc_out],
            fc_mu_bias: vec![0.0; latent],
            fc_logvar_weight: vec![0.0; latent * enc_out],
            fc_logvar_bias: vec![0.0; latent],
            is_synthesized: true,
        })
    }
}

/// Continuous VAE decoder weights (scaffold — real binding is a
/// follow-up wave).
///
/// Upstream `CausalDecoder` (audio_vae_v2.py L270-356) is a chain of
/// `CausalDecoderBlock`s (each: Snake activation + weight-normalized
/// `CausalTransposeConv1d` upsample + three dilated residual units),
/// terminated by a `Snake1d + WNCausalConv1d(d_out=1) + Tanh` head. The
/// scaffold mirrors `ContinuousVaeEncoderWeights` — an aggregate byte
/// bundle plus the final-conv head. Real binding is T29-equivalent.
#[derive(Debug, Clone)]
pub struct ContinuousVaeDecoderWeights {
    /// Flat concatenation of every decoder block's residual + upsample
    /// tensor bytes, in upstream traversal order.
    pub blocks: Vec<f32>,
    /// `[1, dec_terminal_channels]` for the final `WNCausalConv1d(d_out=1)`
    /// head weight.
    pub final_conv_weight: Vec<f32>,
    /// `[1]` bias for the final `WNCausalConv1d(d_out=1)` head.
    pub final_conv_bias: Vec<f32>,
    /// `true` when built by [`Self::synthesized`] — never a real upstream
    /// checkpoint.
    pub is_synthesized: bool,
}

impl ContinuousVaeDecoderWeights {
    /// Deterministic zero-initialized fixture (shape scaffold only).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `config.validate_for_forward`
    /// fails.
    pub fn synthesized(config: &ContinuousVaeConfig) -> Result<Self> {
        config.validate_for_forward()?;
        // Decoder terminal channel width halves per upsample stage
        // (upstream `output_dim = channels // 2 ** (i + 1)` in
        // `CausalDecoder.__init__`).
        let mut terminal = config.decoder_dim as usize;
        for _ in 0..config.decoder_rates.len() {
            terminal /= 2;
        }
        let terminal = terminal.max(1);
        Ok(Self {
            blocks: Vec::new(),
            final_conv_weight: vec![0.0; terminal],
            final_conv_bias: vec![0.0; 1],
            is_synthesized: true,
        })
    }
}

// ---------------------------------------------------------------------------
// Engine handles
// ---------------------------------------------------------------------------

/// Encoder handle carrying config + weights.
///
/// [`Self::encode`] is the primary PCM → `[T', latent_dim]` entry point.
/// Until real weights are bound and the causal-Conv1d + Snake + residual
/// stack is wired end-to-end (T29-equivalent follow-up wave), it returns
/// [`VokraError::NotImplemented`] naming the blocker — never a silent
/// zero-fill (FR-EX-08).
#[derive(Debug, Clone)]
pub struct ContinuousVaeEncoder {
    cfg: ContinuousVaeConfig,
    weights: ContinuousVaeEncoderWeights,
}

impl ContinuousVaeEncoder {
    /// Assemble the encoder from `cfg` and `weights`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] from `cfg.validate_for_forward`
    /// or a scaffold shape mismatch.
    pub fn new(cfg: ContinuousVaeConfig, weights: ContinuousVaeEncoderWeights) -> Result<Self> {
        cfg.validate_for_forward()?;
        let enc_out = (cfg.encoder_dim as usize) * (1 << cfg.encoder_rates.len());
        let latent = cfg.latent_dim as usize;
        if weights.fc_mu_weight.len() != latent * enc_out {
            return Err(VokraError::InvalidArgument(format!(
                "vae_continuous encoder: fc_mu_weight.len()={} != latent_dim * enc_out = {}",
                weights.fc_mu_weight.len(),
                latent * enc_out,
            )));
        }
        if weights.fc_logvar_weight.len() != latent * enc_out {
            return Err(VokraError::InvalidArgument(format!(
                "vae_continuous encoder: fc_logvar_weight.len()={} != latent_dim * enc_out = {}",
                weights.fc_logvar_weight.len(),
                latent * enc_out,
            )));
        }
        if weights.fc_mu_bias.len() != latent {
            return Err(VokraError::InvalidArgument(format!(
                "vae_continuous encoder: fc_mu_bias.len()={} != latent_dim={}",
                weights.fc_mu_bias.len(),
                latent,
            )));
        }
        if weights.fc_logvar_bias.len() != latent {
            return Err(VokraError::InvalidArgument(format!(
                "vae_continuous encoder: fc_logvar_bias.len()={} != latent_dim={}",
                weights.fc_logvar_bias.len(),
                latent,
            )));
        }
        Ok(Self { cfg, weights })
    }

    /// The resolved config.
    #[must_use]
    pub fn config(&self) -> &ContinuousVaeConfig {
        &self.cfg
    }

    /// `true` iff the underlying weight store is synthesized.
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        self.weights.is_synthesized
    }

    /// Encode `[T]` mono PCM to `[T', latent_dim]` `mu` vectors
    /// (row-major, contiguous frames).
    ///
    /// Zero-length input is rejected loudly (FR-EX-08). Real
    /// forward-pass binding lives in the T29-equivalent follow-up wave —
    /// the neural-chain forward path (causal Conv1d + Snake + weight-norm
    /// folded residual units) is transcribed from upstream
    /// `audio_vae_v2.py` and ports there.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on zero-length input.
    /// - [`VokraError::NotImplemented`] otherwise (real forward not yet
    ///   bound — FR-EX-08).
    pub fn encode(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "vae_continuous encode: pcm slice is empty".to_owned(),
            ));
        }
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "vae_continuous encode: this encoder holds synthesized weights \
                 (deterministic scaffold fixture). Bind a real continuous-VAE checkpoint \
                 (e.g. VoxCPM-0.5B AudioVAE V2, apache-2.0) before invoking encode.",
            ));
        }
        Err(VokraError::NotImplemented(
            "vae_continuous encode: real weights are bound but the causal-Conv1d + Snake + \
             weight-norm folded residual encoder forward path has not landed yet. Follow-up \
             wave (T29-equivalent): port the upstream `CausalEncoder` (audio_vae_v2.py \
             L125-159) verbatim — Snake activation, WNCausalConv1d stem, three-dilated \
             residual units per stage, per-stage strided downsample, terminal `fc_mu` head.",
        ))
    }
}

/// Decoder handle carrying config + weights.
///
/// [`Self::decode`] is the primary `[T', latent_dim]` → `[T_out]` PCM
/// entry point. Until real weights are bound and the causal-Conv1d +
/// Snake + residual + transpose-conv upsample stack is wired end-to-end
/// (T29-equivalent follow-up wave), it returns
/// [`VokraError::NotImplemented`] naming the blocker.
#[derive(Debug, Clone)]
pub struct ContinuousVaeDecoder {
    cfg: ContinuousVaeConfig,
    weights: ContinuousVaeDecoderWeights,
}

impl ContinuousVaeDecoder {
    /// Assemble the decoder from `cfg` and `weights`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] from `cfg.validate_for_forward`
    /// or a scaffold shape mismatch.
    pub fn new(cfg: ContinuousVaeConfig, weights: ContinuousVaeDecoderWeights) -> Result<Self> {
        cfg.validate_for_forward()?;
        let mut terminal = cfg.decoder_dim as usize;
        for _ in 0..cfg.decoder_rates.len() {
            terminal /= 2;
        }
        let terminal = terminal.max(1);
        if weights.final_conv_weight.len() != terminal {
            return Err(VokraError::InvalidArgument(format!(
                "vae_continuous decoder: final_conv_weight.len()={} != terminal_channels={}",
                weights.final_conv_weight.len(),
                terminal,
            )));
        }
        if weights.final_conv_bias.len() != 1 {
            return Err(VokraError::InvalidArgument(format!(
                "vae_continuous decoder: final_conv_bias.len()={} != 1",
                weights.final_conv_bias.len(),
            )));
        }
        Ok(Self { cfg, weights })
    }

    /// The resolved config.
    #[must_use]
    pub fn config(&self) -> &ContinuousVaeConfig {
        &self.cfg
    }

    /// `true` iff the underlying weight store is synthesized.
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        self.weights.is_synthesized
    }

    /// Decode `[T', latent_dim]` continuous latents to `[T_out]` mono PCM.
    ///
    /// Zero-length input / mis-aligned length are rejected loudly
    /// (FR-EX-08). Real forward-pass binding lives in the T29-equivalent
    /// follow-up wave.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on zero-length input or a
    ///   length that is not a whole multiple of `latent_dim`.
    /// - [`VokraError::NotImplemented`] otherwise.
    pub fn decode(&self, latents: &[f32]) -> Result<Vec<f32>> {
        if latents.is_empty() {
            return Err(VokraError::InvalidArgument(
                "vae_continuous decode: latents slice is empty".to_owned(),
            ));
        }
        let latent = self.cfg.latent_dim as usize;
        if latents.len() % latent != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "vae_continuous decode: latents.len()={} is not a whole multiple of \
                 latent_dim={}",
                latents.len(),
                latent,
            )));
        }
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "vae_continuous decode: this decoder holds synthesized weights \
                 (deterministic scaffold fixture). Bind a real continuous-VAE checkpoint \
                 (e.g. VoxCPM-0.5B AudioVAE V2, apache-2.0) before invoking decode.",
            ));
        }
        Err(VokraError::NotImplemented(
            "vae_continuous decode: real weights are bound but the causal-Conv1d + Snake + \
             transpose-Conv1d upsample + residual decoder forward path has not landed yet. \
             Follow-up wave (T29-equivalent): port the upstream `CausalDecoder` \
             (audio_vae_v2.py L270-356) verbatim — WNCausalConv1d stem, per-stage \
             `Snake + WNCausalTransposeConv1d` upsample, three-dilated residual units per \
             stage, terminal `Snake1d + WNCausalConv1d(d_out=1) + Tanh` head, optional \
             `SampleRateConditionLayer` scale/bias.",
        ))
    }
}

// ---------------------------------------------------------------------------
// Free-standing runtime entry point (op-only re-export style)
// ---------------------------------------------------------------------------

/// Convenience free function mirroring
/// [`crate::mimi_rvq::mimi_rvq_decode`] / [`crate::dac_rvq::dac_rvq_decode`]
/// re-export style. Constructs a fresh [`ContinuousVaeDecoder`] and
/// invokes it. For repeated calls prefer holding a
/// [`ContinuousVaeDecoder`] and calling [`ContinuousVaeDecoder::decode`]
/// directly — the config + weight validation runs only at construction
/// time.
///
/// # Errors
///
/// See [`ContinuousVaeDecoder::new`] and [`ContinuousVaeDecoder::decode`].
pub fn continuous_vae_decode(
    cfg: ContinuousVaeConfig,
    weights: ContinuousVaeDecoderWeights,
    latents: &[f32],
) -> Result<Vec<f32>> {
    ContinuousVaeDecoder::new(cfg, weights)?.decode(latents)
}

/// Convenience free function mirroring [`continuous_vae_decode`] for the
/// encoder half. Constructs a fresh [`ContinuousVaeEncoder`] and invokes
/// it. For repeated calls prefer holding a [`ContinuousVaeEncoder`].
///
/// # Errors
///
/// See [`ContinuousVaeEncoder::new`] and [`ContinuousVaeEncoder::encode`].
pub fn continuous_vae_encode(
    cfg: ContinuousVaeConfig,
    weights: ContinuousVaeEncoderWeights,
    pcm: &[f32],
) -> Result<Vec<f32>> {
    ContinuousVaeEncoder::new(cfg, weights)?.encode(pcm)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voxcpm_config_matches_primary_source() {
        // Transcribed verbatim from
        // openbmb/VoxCPM/src/voxcpm/modules/audiovae/audio_vae_v2.py
        // `class AudioVAEConfig(BaseModel)` (fetched 2026-07-24).
        let c = ContinuousVaeConfig::voxcpm_0_5b();
        assert_eq!(c.sample_rate_hz, 16_000);
        assert_eq!(c.out_sample_rate_hz, 48_000);
        assert_eq!(c.encoder_dim, 128);
        assert_eq!(c.encoder_rates, vec![2, 5, 8, 8]);
        assert_eq!(c.latent_dim, 64);
        assert_eq!(c.decoder_dim, 2048);
        assert_eq!(c.decoder_rates, vec![8, 6, 5, 2, 2, 2]);
        assert!(c.depthwise);
        assert!(!c.use_noise_block);
    }

    #[test]
    fn voxcpm_hop_length_matches_expected_25hz_frame_rate() {
        let c = ContinuousVaeConfig::voxcpm_0_5b();
        // encoder_rates product = 2*5*8*8 = 640
        assert_eq!(c.hop_length(), Some(640));
        // decoder_rates product = 8*6*5*2*2*2 = 1920
        assert_eq!(c.decode_hop_length(), Some(1920));
        // encoder frame rate = 16_000 / 640 = 25.0 Hz
        assert!((c.frame_rate_hz().unwrap() - 25.0).abs() < 1e-4);
    }

    #[test]
    fn tiny_config_is_well_formed() {
        let c = ContinuousVaeConfig::tiny_for_tests();
        assert!(c.validate_for_forward().is_ok());
        assert_eq!(c.hop_length(), Some(4));
        assert_eq!(c.decode_hop_length(), Some(8));
    }

    #[test]
    fn zero_axes_rejected_loudly() {
        let mut c = ContinuousVaeConfig::tiny_for_tests();
        c.sample_rate_hz = 0;
        assert!(c.validate_for_forward().is_err());

        let mut c = ContinuousVaeConfig::tiny_for_tests();
        c.encoder_rates.clear();
        assert!(c.validate_for_forward().is_err());

        let mut c = ContinuousVaeConfig::tiny_for_tests();
        c.encoder_rates.push(0);
        assert!(c.validate_for_forward().is_err());

        let mut c = ContinuousVaeConfig::tiny_for_tests();
        c.decoder_rates = vec![0];
        assert!(c.validate_for_forward().is_err());

        let mut c = ContinuousVaeConfig::tiny_for_tests();
        c.latent_dim = 0;
        assert!(c.validate_for_forward().is_err());
    }

    #[test]
    fn synthesized_encoder_shape_matches_config() {
        let cfg = ContinuousVaeConfig::tiny_for_tests();
        let w = ContinuousVaeEncoderWeights::synthesized(&cfg).unwrap();
        let enc_out = (cfg.encoder_dim as usize) * (1 << cfg.encoder_rates.len());
        let latent = cfg.latent_dim as usize;
        assert_eq!(w.fc_mu_weight.len(), latent * enc_out);
        assert_eq!(w.fc_logvar_weight.len(), latent * enc_out);
        assert_eq!(w.fc_mu_bias.len(), latent);
        assert_eq!(w.fc_logvar_bias.len(), latent);
        assert!(w.is_synthesized);
    }

    #[test]
    fn synthesized_decoder_shape_matches_config() {
        let cfg = ContinuousVaeConfig::tiny_for_tests();
        let w = ContinuousVaeDecoderWeights::synthesized(&cfg).unwrap();
        let mut terminal = cfg.decoder_dim as usize;
        for _ in 0..cfg.decoder_rates.len() {
            terminal /= 2;
        }
        assert_eq!(w.final_conv_weight.len(), terminal.max(1));
        assert_eq!(w.final_conv_bias.len(), 1);
        assert!(w.is_synthesized);
    }

    #[test]
    fn encoder_new_rejects_mismatched_shapes() {
        let cfg = ContinuousVaeConfig::tiny_for_tests();
        let mut w = ContinuousVaeEncoderWeights::synthesized(&cfg).unwrap();
        w.fc_mu_weight.pop();
        assert!(ContinuousVaeEncoder::new(cfg.clone(), w).is_err());

        let mut w = ContinuousVaeEncoderWeights::synthesized(&cfg).unwrap();
        w.fc_logvar_bias.pop();
        assert!(ContinuousVaeEncoder::new(cfg, w).is_err());
    }

    #[test]
    fn decoder_new_rejects_mismatched_shapes() {
        let cfg = ContinuousVaeConfig::tiny_for_tests();
        let mut w = ContinuousVaeDecoderWeights::synthesized(&cfg).unwrap();
        w.final_conv_weight.push(0.0);
        assert!(ContinuousVaeDecoder::new(cfg, w).is_err());
    }

    #[test]
    fn encoder_encode_synthesized_is_not_implemented_and_says_why() {
        let cfg = ContinuousVaeConfig::tiny_for_tests();
        let w = ContinuousVaeEncoderWeights::synthesized(&cfg).unwrap();
        let enc = ContinuousVaeEncoder::new(cfg, w).unwrap();
        let err = enc.encode(&[0.1_f32; 128]).expect_err("synth encode");
        assert!(
            matches!(err, VokraError::NotImplemented(_)),
            "encode on synth weights must be NotImplemented, got {err:?}"
        );
    }

    #[test]
    fn decoder_decode_synthesized_is_not_implemented_and_says_why() {
        let cfg = ContinuousVaeConfig::tiny_for_tests();
        let w = ContinuousVaeDecoderWeights::synthesized(&cfg).unwrap();
        let dec = ContinuousVaeDecoder::new(cfg.clone(), w).unwrap();
        let latents = vec![0.0_f32; (cfg.latent_dim as usize) * 3];
        let err = dec.decode(&latents).expect_err("synth decode");
        assert!(matches!(err, VokraError::NotImplemented(_)));
    }

    #[test]
    fn encoder_encode_rejects_empty_pcm() {
        let cfg = ContinuousVaeConfig::tiny_for_tests();
        let w = ContinuousVaeEncoderWeights::synthesized(&cfg).unwrap();
        let enc = ContinuousVaeEncoder::new(cfg, w).unwrap();
        let err = enc.encode(&[]).expect_err("empty pcm");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn decoder_decode_rejects_misaligned_length() {
        let cfg = ContinuousVaeConfig::tiny_for_tests();
        let w = ContinuousVaeDecoderWeights::synthesized(&cfg).unwrap();
        let dec = ContinuousVaeDecoder::new(cfg.clone(), w).unwrap();
        // latent_dim = 4; 7 elements is not a whole multiple.
        let err = dec.decode(&[0.0_f32; 7]).expect_err("misaligned");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn free_functions_mirror_handle_methods() {
        let cfg = ContinuousVaeConfig::tiny_for_tests();
        let w_dec = ContinuousVaeDecoderWeights::synthesized(&cfg).unwrap();
        let w_enc = ContinuousVaeEncoderWeights::synthesized(&cfg).unwrap();
        let err_dec = continuous_vae_decode(cfg.clone(), w_dec, &[0.0_f32; 4])
            .expect_err("synth decode via free fn");
        assert!(matches!(err_dec, VokraError::NotImplemented(_)));

        let err_enc = continuous_vae_encode(cfg, w_enc, &[0.0_f32; 128])
            .expect_err("synth encode via free fn");
        assert!(matches!(err_enc, VokraError::NotImplemented(_)));
    }
}
