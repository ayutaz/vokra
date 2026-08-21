//! BigVGAN generator (SoTA plan Phase 3, TTS `bigvgan_generator` primitive).
//!
//! Upstream: <https://github.com/NVIDIA/BigVGAN> (MIT, Copyright (c) 2024
//! NVIDIA CORPORATION). Reference files / line ranges cited in this
//! module:
//!
//! - `bigvgan.py` — `BigVGAN.__init__` L206-322, `BigVGAN.forward` L324-354,
//!   `AMPBlock1.__init__` L23-133, `AMPBlock1.forward` L135-145.
//! - `activations.py` — `Snake` L7-59, `SnakeBeta` L62-114.
//! - `alias_free_activation/torch/{act,filter,resample}.py` — adapted by
//!   NVIDIA from `junjun3518/alias-free-torch` (Apache-2.0), with the sinc
//!   and low-pass construction adapted from `adefossez/julius` (MIT).
//!
//! This Rust port changes storage, error handling, tensor layout, and scalar
//! execution. The applicable third-party texts are retained in
//! `THIRD_PARTY_LICENSES/` and the project `NOTICE`.
//!
//! # Op contract
//!
//! Given
//!
//! - `mel` — a `[in_channels, t_mel]` row-major slice of FP32 mel spectrogram
//!   values;
//! - `weights` — a [`BigVGanWeights`] bundle carrying every conv1d /
//!   transposed_conv1d / AMPBlock1 parameter plus the terminal Snake /
//!   SnakeBeta activation;
//! - `cfg` — [`BigVGanConfig`] shape metadata (upsample rates + kernels, MRF
//!   kernels + dilations, activation kind, `snake_logscale`,
//!   `use_tanh_at_final`, `use_bias_at_final`);
//!
//! [`BigVGanGenerator::forward`] returns a `[n_samples]` mono `Vec<f32>`
//! waveform bounded to `[-1, 1]` by the terminal `tanh` (or `clamp` when
//! `use_tanh_at_final` is false). `n_samples = t_mel *
//! prod(upsample_rates)`.
//!
//! The forward stack matches upstream `BigVGAN.forward` verbatim
//! (`bigvgan.py:324-354`):
//!
//! 1. `conv_pre` (Conv1d k=7, pad=3): `[in_channels] → [upsample_initial_channel]`
//!    (upstream L212);
//! 2. per stage `i ∈ 0..num_upsamples`:
//!    - `ConvTranspose1d(2^i → 2^(i+1), k=upsample_kernel_sizes[i],
//!       s=upsample_rates[i], pad=(k-u)/2)` (upstream L235-245);
//!    - MRF: for every `j ∈ 0..num_kernels`, compute AMPBlock1 branch, average
//!      the branch outputs (upstream L328-340: `xs = xs + branch` accumulator,
//!      `x = xs / num_kernels`);
//! 3. `activation_post` (Snake or SnakeBeta on the last stage's output
//!    channel count) (upstream L263-275);
//! 4. `conv_post` (Conv1d k=7, pad=3): `[last_ch] → [1]` (upstream L281-283);
//! 5. `tanh` when `use_tanh_at_final`, else `clamp(-1, 1)` (upstream
//!    L349-354).
//!
//! # Snake activation reuse
//!
//! Snake (upstream `activations.py:7-59`) is reused verbatim from
//! [`crate::hiftnet::Snake`]. SnakeBeta (upstream `activations.py:62-114`)
//! is a modified variant with an additional per-channel `beta` parameter
//! that controls magnitude while `alpha` controls frequency; because
//! HiFTNet only needs Snake, SnakeBeta lives here.
//!
//! # AMPBlock1 vs HiFTNet ResBlock
//!
//! AMPBlock1 (upstream `bigvgan.py:23-146`) is structurally the same as
//! HiFTNet's `ResBlock` (`hiftnet.rs`): per-branch `activation → dilated
//! Conv1d → activation → Conv1d → residual`, with `dilations = (1, 3, 5)`
//! by default. The difference is the activation kind: HiFTNet is fixed on
//! Snake with `alpha_logscale = false`; BigVGAN allows either Snake or
//! SnakeBeta and honours `h.snake_logscale` (usually `true` in released
//! configs — alphas are then log-scale and exponentiated at forward time).
//! We spell out our own `AmpBlock1` so callers do not accidentally couple
//! to HiFTNet's fixed-activation ResBlock.
//!
//! # Anti-aliased activation
//!
//! Upstream wraps every `Snake` / `SnakeBeta` call with an `Activation1d`
//! module that inserts a polyphase `UpSample1d → activation → DownSample1d`
//! chain (`alias_free_activation/torch/act.py`, cited from `bigvgan.py:87`
//! and `bigvgan.py:277`). That chain is what makes BigVGAN "anti-aliased".
//! [`AliasFreeActivation`] implements that wrapper directly from the stored
//! upstream Kaiser filters. It reproduces the reference's replicate padding,
//! grouped stride-2 transposed convolution, asymmetric crop, activation, and
//! grouped stride-2 low-pass convolution. The input and output time axes are
//! identical; the periodic nonlinearity runs at twice the time resolution.
//!
//! The task description hint "Anti-aliased upsampling uses low-pass filter
//! after each ConvTranspose" does *not* match upstream (upstream wraps the
//! activation, not the transposed conv). Following the primary source
//! verbatim, we do **not** insert a low-pass filter after each
//! `ConvTranspose1d` — that would be a hallucinated deviation. See
//! CLAUDE.md "ハルシネーション厳禁".
//!
//! # Zero third-party deps
//!
//! Every op is native FP32 scalar; no SIMD, no `unsafe`, no matmul crate
//! dependency (matches NFR-DS-02 / M0 zero-dep invariant).

use vokra_core::{Result, VokraError};

use crate::hiftnet::Snake;

// ---------------------------------------------------------------------------
// SnakeBeta activation (upstream `activations.py:62-114`)
// ---------------------------------------------------------------------------

/// Per-channel `SnakeBeta` activation with separate learnable `alpha`
/// (frequency) and `beta` (magnitude) vectors. Closed form
/// (upstream `activations.py:105-113`):
///
/// ```text
/// alpha_eff = exp(alpha) if alpha_logscale else alpha
/// beta_eff  = exp(beta)  if alpha_logscale else beta
/// y = x + (1 / (beta_eff + eps)) * sin(x * alpha_eff)^2
/// ```
///
/// `eps = 1e-9` matches upstream's `no_div_by_zero = 0.000000001`
/// (`activations.py:97`).
///
/// Upstream initialises `alpha` and `beta` to zeros when
/// `alpha_logscale = true` (log-scale identity) and to ones otherwise
/// (`activations.py:84-91`); at load time our converter is expected to have
/// already applied that convention, so we consume the values as-is.
#[derive(Debug, Clone)]
pub struct SnakeBeta {
    alpha: Vec<f32>,
    beta: Vec<f32>,
    alpha_logscale: bool,
    no_div_by_zero: f32,
}

impl SnakeBeta {
    /// Construct a `SnakeBeta` from per-channel `alpha` and `beta` vectors.
    ///
    /// - `alpha_logscale = true` interprets each entry as `log α` / `log β`
    ///   (upstream default when `h.snake_logscale` is set);
    /// - `alpha_logscale = false` uses each value directly.
    ///
    /// Fails loudly if either vector is empty or the two lengths disagree —
    /// upstream ties both to `in_features` so a mismatch here is a converter
    /// bug that would otherwise silently truncate the activation.
    pub fn new(alpha: Vec<f32>, beta: Vec<f32>, alpha_logscale: bool) -> Result<Self> {
        if alpha.is_empty() {
            return Err(VokraError::InvalidArgument(
                "SnakeBeta: alpha vector must not be empty".to_owned(),
            ));
        }
        if beta.len() != alpha.len() {
            return Err(VokraError::InvalidArgument(format!(
                "SnakeBeta: alpha length {} != beta length {}",
                alpha.len(),
                beta.len(),
            )));
        }
        Ok(Self {
            alpha,
            beta,
            alpha_logscale,
            no_div_by_zero: 1e-9,
        })
    }

    /// Number of channels this activation covers (== `alpha.len() == beta.len()`).
    pub fn channels(&self) -> usize {
        self.alpha.len()
    }

    /// Apply the activation in place to a `[channels, time]` row-major
    /// tensor. Fails loudly on shape drift so a caller does not silently
    /// scramble mid-`resblocks` state.
    pub fn forward_in_place(&self, x: &mut [f32], channels: usize, time: usize) -> Result<()> {
        if self.alpha.len() != channels {
            return Err(VokraError::InvalidArgument(format!(
                "SnakeBeta: alpha length {} != channels {channels}",
                self.alpha.len()
            )));
        }
        if x.len() != channels * time {
            return Err(VokraError::InvalidArgument(format!(
                "SnakeBeta forward: input length {} != channels * time = {}",
                x.len(),
                channels * time
            )));
        }
        for (c, (&alpha_raw, &beta_raw)) in self.alpha.iter().zip(self.beta.iter()).enumerate() {
            let alpha = if self.alpha_logscale {
                alpha_raw.exp()
            } else {
                alpha_raw
            };
            let beta = if self.alpha_logscale {
                beta_raw.exp()
            } else {
                beta_raw
            };
            let inv_beta = 1.0 / (beta + self.no_div_by_zero);
            let row_offset = c * time;
            for slot in x[row_offset..row_offset + time].iter_mut() {
                let s = (*slot * alpha).sin();
                *slot += inv_beta * s * s;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Activation kind (config side) + runtime-instantiated wrapper
// ---------------------------------------------------------------------------

/// Which periodic activation family each AMPBlock1 (and the terminal
/// `activation_post`) uses. Mirrors upstream `h.activation` — the two
/// values shipped by the released BigVGAN configs are `"snake"` and
/// `"snakebeta"` (`bigvgan.py:95-115` + `bigvgan.py:263-275`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnakeKind {
    /// `activations.Snake` (upstream `activations.py:7-59`).
    Snake,
    /// `activations.SnakeBeta` (upstream `activations.py:62-114`).
    SnakeBeta,
}

/// Runtime activation — either an already-built [`Snake`] or [`SnakeBeta`].
/// Kept internal so the AMPBlock1 forward can call one code path.
#[derive(Debug, Clone)]
enum AmpActivation {
    Snake(Snake),
    SnakeBeta(SnakeBeta),
}

impl AmpActivation {
    fn channels(&self) -> usize {
        match self {
            Self::Snake(s) => s.channels(),
            Self::SnakeBeta(sb) => sb.channels(),
        }
    }

    fn forward_in_place(&self, x: &mut [f32], channels: usize, time: usize) -> Result<()> {
        match self {
            Self::Snake(s) => s.forward_in_place(x, channels, time),
            Self::SnakeBeta(sb) => sb.forward_in_place(x, channels, time),
        }
    }
}

// ---------------------------------------------------------------------------
// Alias-free Activation1d (upstream `alias_free_activation/torch/`)
// ---------------------------------------------------------------------------

/// The two per-activation Kaiser-window filters stored by upstream
/// `Activation1d`. Released BigVGAN checkpoints use ratio 2 and 12 taps for
/// both sides; the values are buffers in the real state dict and are bound
/// rather than regenerated from remembered constants.
#[derive(Debug, Clone)]
pub struct AliasFreeActivationWeights {
    /// `upsample.filter`, shape `[1, 1, kernel]`.
    pub upsample_filter: Vec<f32>,
    /// `downsample.lowpass.filter`, shape `[1, 1, kernel]`.
    pub downsample_filter: Vec<f32>,
}

#[derive(Debug, Clone)]
struct AliasFreeActivation {
    activation: AmpActivation,
    weights: AliasFreeActivationWeights,
}

impl AliasFreeActivation {
    const RATIO: usize = 2;

    fn new(activation: AmpActivation, weights: AliasFreeActivationWeights) -> Result<Self> {
        for (name, filter) in [
            ("upsample.filter", &weights.upsample_filter),
            ("downsample.lowpass.filter", &weights.downsample_filter),
        ] {
            if filter.len() < Self::RATIO || filter.len() % 2 != 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "AliasFreeActivation {name}: expected a non-empty even kernel >= {}, got {} taps",
                    Self::RATIO,
                    filter.len()
                )));
            }
            if !filter.iter().all(|value| value.is_finite()) {
                return Err(VokraError::InvalidArgument(format!(
                    "AliasFreeActivation {name}: filter contains a non-finite value"
                )));
            }
        }
        Ok(Self {
            activation,
            weights,
        })
    }

    fn forward_in_place(&self, x: &mut [f32], channels: usize, time: usize) -> Result<()> {
        if x.len() != channels * time {
            return Err(VokraError::InvalidArgument(format!(
                "AliasFreeActivation forward: input length {} != channels * time = {}",
                x.len(),
                channels * time
            )));
        }
        let mut upsampled = alias_free_upsample(
            x,
            channels,
            time,
            Self::RATIO,
            &self.weights.upsample_filter,
        )?;
        self.activation
            .forward_in_place(&mut upsampled, channels, time * Self::RATIO)?;
        let downsampled = alias_free_downsample(
            &upsampled,
            channels,
            time * Self::RATIO,
            Self::RATIO,
            &self.weights.downsample_filter,
        )?;
        debug_assert_eq!(downsampled.len(), x.len());
        x.copy_from_slice(&downsampled);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AMPBlock1 (upstream `bigvgan.py:23-146`)
// ---------------------------------------------------------------------------

/// Weights for a single [`AmpBlock1`] — one weight/bias per conv (2 per
/// dilation) and one `alpha` (+ optional `beta` for SnakeBeta) per
/// activation (2 per dilation, so `num_layers = 2 * dilations.len()`
/// per upstream `bigvgan.py:78-80`).
#[derive(Debug, Clone)]
pub struct AmpBlock1Weights {
    /// Row-major `[channels, channels, kernel]` per branch — `convs1[i]`
    /// uses `dilations[i]` for its stride-1 dilated convolution
    /// (upstream `bigvgan.py:41-53`).
    pub convs1_w: Vec<Vec<f32>>,
    /// `[channels]` bias per `convs1[i]`.
    pub convs1_b: Vec<Vec<f32>>,
    /// Row-major `[channels, channels, kernel]` per branch — `convs2[i]`
    /// is always `dilation=1` upstream (`bigvgan.py:58-70`).
    pub convs2_w: Vec<Vec<f32>>,
    /// `[channels]` bias per `convs2[i]`.
    pub convs2_b: Vec<Vec<f32>>,
    /// One `[channels]` Snake/SnakeBeta `alpha` per branch first
    /// activation (`activations[0::2]`, upstream `bigvgan.py:135-140`).
    /// Length must equal `dilations.len()`.
    pub activations1_alpha: Vec<Vec<f32>>,
    /// One `[channels]` `alpha` per branch second activation
    /// (`activations[1::2]`). Length must equal `dilations.len()`.
    pub activations2_alpha: Vec<Vec<f32>>,
    /// Per-branch first-activation `beta` — only populated when the block
    /// was configured with [`SnakeKind::SnakeBeta`]. `None` for Snake.
    pub activations1_beta: Option<Vec<Vec<f32>>>,
    /// Per-branch second-activation `beta` — only populated when the block
    /// was configured with [`SnakeKind::SnakeBeta`]. `None` for Snake.
    pub activations2_beta: Option<Vec<Vec<f32>>>,
    /// Per-branch first-activation alias-free filters. Length must equal
    /// `dilations.len()`; each entry binds the upstream buffers for
    /// `activations[0::2]`.
    pub activations1_filters: Vec<AliasFreeActivationWeights>,
    /// Per-branch second-activation alias-free filters for
    /// `activations[1::2]`.
    pub activations2_filters: Vec<AliasFreeActivationWeights>,
}

/// AMPBlock1 (upstream `bigvgan.py:23-146`). The BigVGAN default, matched
/// by every released variant (`bigvgan-v2-*` and the v1 checkpoints).
///
/// The per-branch call chain (upstream `bigvgan.py:135-145`):
///
/// ```text
/// for c1, c2, a1, a2 in zip(convs1, convs2, acts1, acts2):
///     xt = a1(x)
///     xt = c1(xt)      # dilated conv, stride=1
///     xt = a2(xt)
///     xt = c2(xt)      # dilation=1 conv, stride=1
///     x  = xt + x       # residual accumulate
/// ```
///
/// `AmpBlock2` (upstream `bigvgan.py:150-224`) drops the `c2` step and
/// halves the activation count; that variant is not in any released
/// checkpoint we currently target, so this file only carries AMPBlock1
/// and defers AMPBlock2 to a follow-up Wave if a consumer ever needs it.
#[derive(Debug, Clone)]
pub struct AmpBlock1 {
    channels: u32,
    kernel_size: u32,
    dilations: Vec<u32>,
    weights: AmpBlock1Weights,
    activations1: Vec<AliasFreeActivation>,
    activations2: Vec<AliasFreeActivation>,
}

impl AmpBlock1 {
    /// Build an `AmpBlock1` from its shape metadata + weight bundle.
    /// Fails loudly on any shape disagreement.
    pub fn new(
        channels: u32,
        kernel_size: u32,
        dilations: Vec<u32>,
        activation: SnakeKind,
        alpha_logscale: bool,
        weights: AmpBlock1Weights,
    ) -> Result<Self> {
        let n_branches = dilations.len();
        if n_branches == 0 {
            return Err(VokraError::InvalidArgument(
                "AmpBlock1: dilations must not be empty".to_owned(),
            ));
        }
        if channels == 0 || kernel_size == 0 {
            return Err(VokraError::InvalidArgument(
                "AmpBlock1: channels and kernel_size must be > 0".to_owned(),
            ));
        }
        for (name, v) in [
            ("convs1_w", weights.convs1_w.len()),
            ("convs1_b", weights.convs1_b.len()),
            ("convs2_w", weights.convs2_w.len()),
            ("convs2_b", weights.convs2_b.len()),
            ("activations1_alpha", weights.activations1_alpha.len()),
            ("activations2_alpha", weights.activations2_alpha.len()),
            ("activations1_filters", weights.activations1_filters.len()),
            ("activations2_filters", weights.activations2_filters.len()),
        ] {
            if v != n_branches {
                return Err(VokraError::InvalidArgument(format!(
                    "AmpBlock1: {name} has {v} entries but dilations has {n_branches}"
                )));
            }
        }
        // Beta shapes: required for SnakeBeta, forbidden for Snake.
        match activation {
            SnakeKind::Snake => {
                if weights.activations1_beta.is_some() || weights.activations2_beta.is_some() {
                    return Err(VokraError::InvalidArgument(
                        "AmpBlock1: activations{1,2}_beta must be None when \
                         activation == Snake"
                            .to_owned(),
                    ));
                }
            }
            SnakeKind::SnakeBeta => {
                let (b1, b2) = match (&weights.activations1_beta, &weights.activations2_beta) {
                    (Some(b1), Some(b2)) => (b1, b2),
                    _ => {
                        return Err(VokraError::InvalidArgument(
                            "AmpBlock1: activations{1,2}_beta must be Some when \
                             activation == SnakeBeta"
                                .to_owned(),
                        ));
                    }
                };
                if b1.len() != n_branches || b2.len() != n_branches {
                    return Err(VokraError::InvalidArgument(format!(
                        "AmpBlock1 SnakeBeta beta: expected {n_branches} entries \
                         each, got {} + {}",
                        b1.len(),
                        b2.len()
                    )));
                }
            }
        }
        let ch = channels as usize;
        let k = kernel_size as usize;
        let expected_w = ch * ch * k;
        for i in 0..n_branches {
            if weights.convs1_w[i].len() != expected_w {
                return Err(VokraError::InvalidArgument(format!(
                    "AmpBlock1 convs1_w[{i}]: expected length {expected_w} \
                     ({ch}*{ch}*{k}), got {}",
                    weights.convs1_w[i].len(),
                )));
            }
            if weights.convs2_w[i].len() != expected_w {
                return Err(VokraError::InvalidArgument(format!(
                    "AmpBlock1 convs2_w[{i}]: expected length {expected_w} \
                     ({ch}*{ch}*{k}), got {}",
                    weights.convs2_w[i].len(),
                )));
            }
            if weights.convs1_b[i].len() != ch {
                return Err(VokraError::InvalidArgument(format!(
                    "AmpBlock1 convs1_b[{i}]: expected length {ch}, got {}",
                    weights.convs1_b[i].len(),
                )));
            }
            if weights.convs2_b[i].len() != ch {
                return Err(VokraError::InvalidArgument(format!(
                    "AmpBlock1 convs2_b[{i}]: expected length {ch}, got {}",
                    weights.convs2_b[i].len(),
                )));
            }
        }
        let mut activations1 = Vec::with_capacity(n_branches);
        let mut activations2 = Vec::with_capacity(n_branches);
        for i in 0..n_branches {
            let a1 = match activation {
                SnakeKind::Snake => AmpActivation::Snake(Snake::new(
                    weights.activations1_alpha[i].clone(),
                    alpha_logscale,
                )?),
                SnakeKind::SnakeBeta => {
                    let beta = weights.activations1_beta.as_ref().unwrap()[i].clone();
                    AmpActivation::SnakeBeta(SnakeBeta::new(
                        weights.activations1_alpha[i].clone(),
                        beta,
                        alpha_logscale,
                    )?)
                }
            };
            let a2 = match activation {
                SnakeKind::Snake => AmpActivation::Snake(Snake::new(
                    weights.activations2_alpha[i].clone(),
                    alpha_logscale,
                )?),
                SnakeKind::SnakeBeta => {
                    let beta = weights.activations2_beta.as_ref().unwrap()[i].clone();
                    AmpActivation::SnakeBeta(SnakeBeta::new(
                        weights.activations2_alpha[i].clone(),
                        beta,
                        alpha_logscale,
                    )?)
                }
            };
            if a1.channels() != ch {
                return Err(VokraError::InvalidArgument(format!(
                    "AmpBlock1 activations1[{i}]: expected {ch} channels, got {}",
                    a1.channels()
                )));
            }
            if a2.channels() != ch {
                return Err(VokraError::InvalidArgument(format!(
                    "AmpBlock1 activations2[{i}]: expected {ch} channels, got {}",
                    a2.channels()
                )));
            }
            activations1.push(AliasFreeActivation::new(
                a1,
                weights.activations1_filters[i].clone(),
            )?);
            activations2.push(AliasFreeActivation::new(
                a2,
                weights.activations2_filters[i].clone(),
            )?);
        }
        Ok(Self {
            channels,
            kernel_size,
            dilations,
            weights,
            activations1,
            activations2,
        })
    }

    /// Forward pass. Reproduces upstream `AMPBlock1.forward`
    /// (`bigvgan.py:135-145`). Mutates `x` in place across every branch
    /// (residual accumulation); the caller supplies a `[channels, t]`
    /// row-major buffer.
    pub fn forward_in_place(&self, x: &mut [f32], t: usize) -> Result<()> {
        let ch = self.channels as usize;
        let k = self.kernel_size as usize;
        if x.len() != ch * t {
            return Err(VokraError::InvalidArgument(format!(
                "AmpBlock1 forward: input length {} != channels * t = {}",
                x.len(),
                ch * t
            )));
        }
        for (idx, &dilation) in self.dilations.iter().enumerate() {
            let d = dilation as usize;
            let mut xt = x.to_vec();
            self.activations1[idx].forward_in_place(&mut xt, ch, t)?;
            let pad1 = get_padding(k, d);
            xt = conv1d_dilated_same_padding(
                &xt,
                ch,
                ch,
                k,
                d,
                pad1,
                t,
                &self.weights.convs1_w[idx],
                &self.weights.convs1_b[idx],
            )?;
            self.activations2[idx].forward_in_place(&mut xt, ch, t)?;
            let pad2 = get_padding(k, 1);
            xt = conv1d_dilated_same_padding(
                &xt,
                ch,
                ch,
                k,
                1,
                pad2,
                t,
                &self.weights.convs2_w[idx],
                &self.weights.convs2_b[idx],
            )?;
            for (dst, &delta) in x.iter_mut().zip(xt.iter()) {
                *dst += delta;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BigVGanConfig / BigVGanWeights / BigVGanGenerator
// ---------------------------------------------------------------------------

/// Hyperparameters for [`BigVGanGenerator`]. Defaults mirror the released
/// `bigvgan_v2_24khz_100band_256x` checkpoint's `config.json` (upstream
/// [Hugging Face](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x));
/// callers deserialising a different variant override the fields their
/// checkpoint disagrees with.
///
/// Fields ordered as they appear in upstream `bigvgan.py:206-322`.
#[derive(Debug, Clone)]
pub struct BigVGanConfig {
    /// Mel bins on the input (upstream `h.num_mels`, `bigvgan.py:212`).
    pub in_channels: u32,
    /// Initial upsample-side channel count (upstream
    /// `h.upsample_initial_channel`, `bigvgan.py:212`).
    pub upsample_initial_channel: u32,
    /// Per-stage stride (upstream `h.upsample_rates`, `bigvgan.py:234`).
    pub upsample_rates: Vec<u32>,
    /// Per-stage kernel size (upstream `h.upsample_kernel_sizes`,
    /// `bigvgan.py:234`). Length must equal `upsample_rates`.
    pub upsample_kernel_sizes: Vec<u32>,
    /// MRF branch kernel sizes (upstream `h.resblock_kernel_sizes`,
    /// `bigvgan.py:250`).
    pub resblock_kernel_sizes: Vec<u32>,
    /// MRF branch dilation lists (upstream `h.resblock_dilation_sizes`,
    /// `bigvgan.py:250`). Length must equal `resblock_kernel_sizes`.
    pub resblock_dilation_sizes: Vec<Vec<u32>>,
    /// Which activation family each AMPBlock1 and the terminal
    /// `activation_post` use (upstream `h.activation`).
    pub activation: SnakeKind,
    /// Whether `alpha` (and `beta` for SnakeBeta) are stored in log-scale
    /// (upstream `h.snake_logscale`, default `true` in released configs).
    pub snake_logscale: bool,
    /// Whether the terminal `conv_post` uses a bias term (upstream
    /// `h.use_bias_at_final`, default `true`, `bigvgan.py:278-283`).
    pub use_bias_at_final: bool,
    /// Whether the final activation is `tanh` (`true`, upstream default)
    /// or `clamp(-1, 1)` (`false`) (upstream `h.use_tanh_at_final`,
    /// `bigvgan.py:289 + L349-354`).
    pub use_tanh_at_final: bool,
}

impl Default for BigVGanConfig {
    fn default() -> Self {
        Self {
            in_channels: 100,
            upsample_initial_channel: 1536,
            upsample_rates: vec![4, 4, 2, 2, 2, 2],
            upsample_kernel_sizes: vec![8, 8, 4, 4, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            activation: SnakeKind::SnakeBeta,
            snake_logscale: true,
            use_bias_at_final: false,
            use_tanh_at_final: false,
        }
    }
}

impl BigVGanConfig {
    /// Number of upsample stages (== `upsample_rates.len()`).
    pub fn num_upsamples(&self) -> usize {
        self.upsample_rates.len()
    }

    /// Number of MRF kernels per stage (== `resblock_kernel_sizes.len()`).
    pub fn num_kernels(&self) -> usize {
        self.resblock_kernel_sizes.len()
    }

    /// Total upsample factor (`prod(upsample_rates)`). Output waveform length is
    /// `t_mel * total_upsample_factor()`.
    pub fn total_upsample_factor(&self) -> u32 {
        self.upsample_rates.iter().product::<u32>()
    }

    /// Feature channels on stage `i`'s output side —
    /// `upsample_initial_channel / 2^(i+1)` (upstream L245).
    /// Stage 0 halves, stage 1 quarters, etc.
    pub fn output_channels_at(&self, stage: usize) -> u32 {
        self.upsample_initial_channel >> (stage as u32 + 1)
    }
}

/// Learned parameters for [`BigVGanGenerator`]. Layout notes:
///
/// - `conv_pre_w`: row-major `[upsample_initial_channel, in_channels, 7]`
///   (upstream L212).
/// - `ups_w[i]`: row-major `[in_ch_i, out_ch_i, upsample_kernel_sizes[i]]`
///   (PyTorch `nn.ConvTranspose1d` layout with in-channels leading, upstream
///   L235-245). `in_ch_i = upsample_initial_channel >> i`; `out_ch_i =
///   upsample_initial_channel >> (i+1)`.
/// - `amp_blocks[i * num_kernels + j]`: AMPBlock1 weights for stage `i`,
///   MRF branch `j` (upstream L254-260). `channels = out_ch_i`,
///   `kernel = resblock_kernel_sizes[j]`, `dilations =
///   resblock_dilation_sizes[j]`.
/// - `activation_post_alpha`: `[out_ch_{n-1}]` alpha for the terminal
///   activation (upstream L263-275).
/// - `activation_post_beta`: `[out_ch_{n-1}]` beta — required for
///   [`SnakeKind::SnakeBeta`], forbidden for [`SnakeKind::Snake`].
/// - `activation_post_filter`: the terminal activation's stored alias-free
///   upsample/downsample filter buffers.
/// - `conv_post_w`: row-major `[1, out_ch_{n-1}, 7]` (upstream L281-283).
/// - `conv_post_b`: `[1]` when `use_bias_at_final = true`, otherwise
///   `None` (upstream `bias=self.use_bias_at_final`).
#[derive(Debug, Clone)]
pub struct BigVGanWeights {
    /// Row-major `[upsample_initial_channel, in_channels, 7]`.
    pub conv_pre_w: Vec<f32>,
    /// `[upsample_initial_channel]` bias for `conv_pre`.
    pub conv_pre_b: Vec<f32>,
    /// Per-stage upsample ConvTranspose1d weights.
    pub ups_w: Vec<Vec<f32>>,
    /// Per-stage upsample ConvTranspose1d biases.
    pub ups_b: Vec<Vec<f32>>,
    /// Row-major `num_upsamples * num_kernels` AMPBlock1 weights.
    pub amp_blocks: Vec<AmpBlock1Weights>,
    /// `[out_ch_{n-1}]` alpha for the terminal activation.
    pub activation_post_alpha: Vec<f32>,
    /// Optional `[out_ch_{n-1}]` beta for the terminal activation —
    /// required for SnakeBeta, forbidden for Snake.
    pub activation_post_beta: Option<Vec<f32>>,
    /// Alias-free filter buffers for the terminal activation.
    pub activation_post_filter: AliasFreeActivationWeights,
    /// Row-major `[1, out_ch_{n-1}, 7]` post-conv weight.
    pub conv_post_w: Vec<f32>,
    /// `[1]` post-conv bias — `None` iff `cfg.use_bias_at_final == false`
    /// (upstream `bigvgan.py:281-283`).
    pub conv_post_b: Option<Vec<f32>>,
}

/// BigVGAN neural vocoder generator — the full anti-aliased upsample chain
/// (see module docstring for the exact call sequence).
#[derive(Debug, Clone)]
pub struct BigVGanGenerator {
    cfg: BigVGanConfig,
    weights: BigVGanWeights,
    /// One AMPBlock1 per `(stage, kernel)` slot — laid out row-major so
    /// `amp_blocks[i * num_kernels + j]` gives the block for upsample
    /// stage `i` and MRF branch `j`.
    amp_blocks: Vec<AmpBlock1>,
    /// Terminal `activation_post` — Snake or SnakeBeta on the last stage's
    /// output channel count.
    activation_post: AliasFreeActivation,
}

impl BigVGanGenerator {
    /// Build a `BigVGanGenerator` from its config + weights bundle. Every
    /// shape is checked upfront so a mismatch surfaces at build time
    /// rather than mid-forward.
    pub fn new(cfg: BigVGanConfig, weights: BigVGanWeights) -> Result<Self> {
        // ---- Config-shape invariants -----------------------------------
        let n_ups = cfg.num_upsamples();
        let n_kernels = cfg.num_kernels();
        if n_ups == 0 {
            return Err(VokraError::InvalidArgument(
                "BigVGanGenerator: upsample_rates must not be empty".to_owned(),
            ));
        }
        if cfg.upsample_kernel_sizes.len() != n_ups {
            return Err(VokraError::InvalidArgument(format!(
                "BigVGanGenerator: upsample_kernel_sizes length {} != \
                 upsample_rates length {n_ups}",
                cfg.upsample_kernel_sizes.len()
            )));
        }
        if n_kernels == 0 {
            return Err(VokraError::InvalidArgument(
                "BigVGanGenerator: resblock_kernel_sizes must not be empty".to_owned(),
            ));
        }
        if cfg.resblock_dilation_sizes.len() != n_kernels {
            return Err(VokraError::InvalidArgument(format!(
                "BigVGanGenerator: resblock_dilation_sizes length {} != \
                 resblock_kernel_sizes length {n_kernels}",
                cfg.resblock_dilation_sizes.len()
            )));
        }
        if cfg.in_channels == 0 || cfg.upsample_initial_channel == 0 {
            return Err(VokraError::InvalidArgument(
                "BigVGanGenerator: in_channels and upsample_initial_channel must be > 0".to_owned(),
            ));
        }

        // ---- conv_pre ---------------------------------------------------
        let bc = cfg.upsample_initial_channel as usize;
        let inc = cfg.in_channels as usize;
        let expected_conv_pre_w = bc * inc * 7;
        if weights.conv_pre_w.len() != expected_conv_pre_w {
            return Err(VokraError::InvalidArgument(format!(
                "BigVGanGenerator conv_pre_w: expected length {expected_conv_pre_w} \
                 ({bc}*{inc}*7), got {}",
                weights.conv_pre_w.len()
            )));
        }
        if weights.conv_pre_b.len() != bc {
            return Err(VokraError::InvalidArgument(format!(
                "BigVGanGenerator conv_pre_b: expected length {bc}, got {}",
                weights.conv_pre_b.len()
            )));
        }

        // ---- ups ---------------------------------------------------------
        if weights.ups_w.len() != n_ups || weights.ups_b.len() != n_ups {
            return Err(VokraError::InvalidArgument(format!(
                "BigVGanGenerator ups: expected {n_ups} weight and bias sets, \
                 got {} weights / {} biases",
                weights.ups_w.len(),
                weights.ups_b.len()
            )));
        }
        for i in 0..n_ups {
            let in_ch = (cfg.upsample_initial_channel >> (i as u32)) as usize;
            let out_ch = (cfg.upsample_initial_channel >> (i as u32 + 1)) as usize;
            if out_ch == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "BigVGanGenerator ups[{i}]: derived out_ch is 0 — \
                     upsample_initial_channel ({}) shifted right by {} is 0",
                    cfg.upsample_initial_channel,
                    i as u32 + 1
                )));
            }
            let k = cfg.upsample_kernel_sizes[i] as usize;
            let expected = in_ch * out_ch * k;
            if weights.ups_w[i].len() != expected {
                return Err(VokraError::InvalidArgument(format!(
                    "BigVGanGenerator ups_w[{i}]: expected length {expected} \
                     ({in_ch}*{out_ch}*{k}), got {}",
                    weights.ups_w[i].len()
                )));
            }
            if weights.ups_b[i].len() != out_ch {
                return Err(VokraError::InvalidArgument(format!(
                    "BigVGanGenerator ups_b[{i}]: expected length {out_ch}, got {}",
                    weights.ups_b[i].len()
                )));
            }
            let stride = cfg.upsample_rates[i] as usize;
            if k < stride {
                return Err(VokraError::InvalidArgument(format!(
                    "BigVGanGenerator ups[{i}]: kernel {k} < stride {stride} \
                     (upstream `padding = (k-u)//2` requires k >= u)"
                )));
            }
        }

        // ---- amp_blocks --------------------------------------------------
        let expected_amp = n_ups * n_kernels;
        if weights.amp_blocks.len() != expected_amp {
            return Err(VokraError::InvalidArgument(format!(
                "BigVGanGenerator amp_blocks: expected {expected_amp} \
                 (num_upsamples * num_kernels), got {}",
                weights.amp_blocks.len()
            )));
        }
        let mut amp_blocks = Vec::with_capacity(expected_amp);
        for i in 0..n_ups {
            let ch = cfg.output_channels_at(i);
            for (j, (&kern, dils)) in cfg
                .resblock_kernel_sizes
                .iter()
                .zip(cfg.resblock_dilation_sizes.iter())
                .enumerate()
            {
                let idx = i * n_kernels + j;
                let block = AmpBlock1::new(
                    ch,
                    kern,
                    dils.clone(),
                    cfg.activation,
                    cfg.snake_logscale,
                    weights.amp_blocks[idx].clone(),
                )?;
                amp_blocks.push(block);
            }
        }

        // ---- activation_post + conv_post --------------------------------
        let last_ch = cfg.output_channels_at(n_ups - 1) as usize;
        if last_ch == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "BigVGanGenerator: derived last-stage channel count is 0 — \
                 upsample_initial_channel ({}) shifted right by {} is 0",
                cfg.upsample_initial_channel, n_ups
            )));
        }
        if weights.activation_post_alpha.len() != last_ch {
            return Err(VokraError::InvalidArgument(format!(
                "BigVGanGenerator activation_post_alpha: expected length {last_ch}, got {}",
                weights.activation_post_alpha.len()
            )));
        }
        let activation_post_inner = match cfg.activation {
            SnakeKind::Snake => {
                if weights.activation_post_beta.is_some() {
                    return Err(VokraError::InvalidArgument(
                        "BigVGanGenerator: activation_post_beta must be None when \
                         activation == Snake"
                            .to_owned(),
                    ));
                }
                AmpActivation::Snake(Snake::new(
                    weights.activation_post_alpha.clone(),
                    cfg.snake_logscale,
                )?)
            }
            SnakeKind::SnakeBeta => {
                let beta = weights.activation_post_beta.as_ref().ok_or_else(|| {
                    VokraError::InvalidArgument(
                        "BigVGanGenerator: activation_post_beta must be Some when \
                         activation == SnakeBeta"
                            .to_owned(),
                    )
                })?;
                if beta.len() != last_ch {
                    return Err(VokraError::InvalidArgument(format!(
                        "BigVGanGenerator activation_post_beta: expected length \
                         {last_ch}, got {}",
                        beta.len()
                    )));
                }
                AmpActivation::SnakeBeta(SnakeBeta::new(
                    weights.activation_post_alpha.clone(),
                    beta.clone(),
                    cfg.snake_logscale,
                )?)
            }
        };
        let activation_post = AliasFreeActivation::new(
            activation_post_inner,
            weights.activation_post_filter.clone(),
        )?;
        // Layout is `[1, last_ch, 7]` — `1 * last_ch * 7 = last_ch * 7`.
        let expected_post_w = last_ch * 7;
        if weights.conv_post_w.len() != expected_post_w {
            return Err(VokraError::InvalidArgument(format!(
                "BigVGanGenerator conv_post_w: expected length {expected_post_w} \
                 (1*{last_ch}*7), got {}",
                weights.conv_post_w.len()
            )));
        }
        match (cfg.use_bias_at_final, &weights.conv_post_b) {
            (true, Some(b)) => {
                if b.len() != 1 {
                    return Err(VokraError::InvalidArgument(format!(
                        "BigVGanGenerator conv_post_b: expected length 1, got {}",
                        b.len()
                    )));
                }
            }
            (true, None) => {
                return Err(VokraError::InvalidArgument(
                    "BigVGanGenerator: conv_post_b must be Some when \
                     use_bias_at_final = true"
                        .to_owned(),
                ));
            }
            (false, Some(_)) => {
                return Err(VokraError::InvalidArgument(
                    "BigVGanGenerator: conv_post_b must be None when \
                     use_bias_at_final = false"
                        .to_owned(),
                ));
            }
            (false, None) => {}
        }

        Ok(Self {
            cfg,
            weights,
            amp_blocks,
            activation_post,
        })
    }

    /// Immutable access to the config this generator was built with.
    pub fn config(&self) -> &BigVGanConfig {
        &self.cfg
    }

    /// Forward pass. Reproduces upstream `BigVGAN.forward`
    /// (`bigvgan.py:324-354`) including every alias-free activation wrapper.
    ///
    /// `mel` is row-major `[in_channels, t_mel]`. Output length is
    /// `t_mel * total_upsample_factor()`; the value range is `[-1, 1]`
    /// after the terminal `tanh` (or clamp).
    pub fn forward(&self, mel: &[f32], t_mel: usize) -> Result<Vec<f32>> {
        if t_mel == 0 {
            return Err(VokraError::InvalidArgument(
                "BigVGanGenerator forward: t_mel must be > 0".to_owned(),
            ));
        }
        let inc = self.cfg.in_channels as usize;
        let bc = self.cfg.upsample_initial_channel as usize;
        if mel.len() != inc * t_mel {
            return Err(VokraError::InvalidArgument(format!(
                "BigVGanGenerator forward: mel length {} != in_channels * t_mel = {}",
                mel.len(),
                inc * t_mel
            )));
        }

        // ---- 1. conv_pre (Conv1d k=7, pad=3) -----------------------------
        let mut x = conv1d_same_padding(
            mel,
            inc,
            bc,
            7,
            3,
            t_mel,
            &self.weights.conv_pre_w,
            &self.weights.conv_pre_b,
        );
        let mut t_cur = t_mel;

        let n_ups = self.cfg.num_upsamples();
        let n_kernels = self.cfg.num_kernels();

        // ---- 2. per-stage upsample + MRF averaging -----------------------
        for i in 0..n_ups {
            let in_ch = (self.cfg.upsample_initial_channel >> (i as u32)) as usize;
            let out_ch = (self.cfg.upsample_initial_channel >> (i as u32 + 1)) as usize;
            let k = self.cfg.upsample_kernel_sizes[i] as usize;
            let stride = self.cfg.upsample_rates[i] as usize;
            let padding = (k - stride) / 2;
            x = conv_transpose1d(
                &x,
                in_ch,
                out_ch,
                k,
                stride,
                padding,
                t_cur,
                &self.weights.ups_w[i],
                &self.weights.ups_b[i],
            )?;
            t_cur *= stride;

            // MRF averaging: for each MRF branch, apply the AMP block on a
            // fresh copy of x, then average branch outputs (upstream
            // `bigvgan.py:334-340`).
            let branch_len = out_ch * t_cur;
            let mut xs = vec![0.0f32; branch_len];
            for j in 0..n_kernels {
                let mut branch = x.clone();
                self.amp_blocks[i * n_kernels + j].forward_in_place(&mut branch, t_cur)?;
                for (dst, &delta) in xs.iter_mut().zip(branch.iter()) {
                    *dst += delta;
                }
            }
            let inv_n = 1.0 / n_kernels as f32;
            for slot in xs.iter_mut() {
                *slot *= inv_n;
            }
            x = xs;
        }

        // ---- 3. activation_post ------------------------------------------
        let last_ch = self.cfg.output_channels_at(n_ups - 1) as usize;
        self.activation_post
            .forward_in_place(&mut x, last_ch, t_cur)?;

        // ---- 4. conv_post (Conv1d k=7, pad=3) -----------------------------
        let bias = self.weights.conv_post_b.as_deref().unwrap_or(&[0.0f32; 1]);
        let out_1ch =
            conv1d_same_padding(&x, last_ch, 1, 7, 3, t_cur, &self.weights.conv_post_w, bias);
        // out_1ch is [1, t_cur] row-major = length t_cur.

        // ---- 5. tanh (or clamp(-1, 1)) ------------------------------------
        let mut y = out_1ch;
        if self.cfg.use_tanh_at_final {
            for slot in y.iter_mut() {
                *slot = slot.tanh();
            }
        } else {
            for slot in y.iter_mut() {
                *slot = slot.clamp(-1.0, 1.0);
            }
        }
        Ok(y)
    }
}

// ---------------------------------------------------------------------------
// Private helpers (duplicated from hiftnet.rs — see module docstring
// "Zero third-party deps"; a shared conv helper crate is deferred until a
// third consumer materialises).
// ---------------------------------------------------------------------------

/// Reference-equivalent `UpSample1d`: replicate-pad, grouped transposed
/// convolution scaled by `ratio`, then crop the reference's asymmetric
/// `pad_left` / `pad_right` interval.
fn alias_free_upsample(
    input: &[f32],
    channels: usize,
    time: usize,
    ratio: usize,
    filter: &[f32],
) -> Result<Vec<f32>> {
    if ratio == 0 || time == 0 || filter.len() < ratio || input.len() != channels * time {
        return Err(VokraError::InvalidArgument(format!(
            "alias_free_upsample: invalid shape/input (channels={channels}, time={time}, ratio={ratio}, taps={}, input={})",
            filter.len(),
            input.len()
        )));
    }
    let pad = filter.len() / ratio - 1;
    let padded_time = time + 2 * pad;
    let core_time = (padded_time - 1) * ratio + filter.len();
    let crop_left = pad * ratio + (filter.len() - ratio) / 2;
    let crop_right = pad * ratio + (filter.len() - ratio + 1) / 2;
    let output_time = core_time
        .checked_sub(crop_left + crop_right)
        .ok_or_else(|| {
            VokraError::InvalidArgument(
                "alias_free_upsample: crop exceeds transposed-convolution output".to_owned(),
            )
        })?;
    if output_time != time * ratio {
        return Err(VokraError::InvalidArgument(format!(
            "alias_free_upsample: derived output time {output_time} != time * ratio {}",
            time * ratio
        )));
    }

    let mut output = vec![0.0f32; channels * output_time];
    for channel in 0..channels {
        let input_row = channel * time;
        let output_row = channel * output_time;
        for padded_index in 0..padded_time {
            let source = padded_index.saturating_sub(pad).min(time - 1);
            let value = input[input_row + source] * ratio as f32;
            for (tap, &coefficient) in filter.iter().enumerate() {
                let core_index = padded_index * ratio + tap;
                if core_index >= crop_left && core_index < core_time - crop_right {
                    output[output_row + core_index - crop_left] += value * coefficient;
                }
            }
        }
    }
    Ok(output)
}

/// Reference-equivalent `DownSample1d`: asymmetric replicate padding then a
/// grouped low-pass convolution with stride `ratio`.
fn alias_free_downsample(
    input: &[f32],
    channels: usize,
    time: usize,
    ratio: usize,
    filter: &[f32],
) -> Result<Vec<f32>> {
    if ratio == 0 || time == 0 || filter.is_empty() || input.len() != channels * time {
        return Err(VokraError::InvalidArgument(format!(
            "alias_free_downsample: invalid shape/input (channels={channels}, time={time}, ratio={ratio}, taps={}, input={})",
            filter.len(),
            input.len()
        )));
    }
    let even = usize::from(filter.len() % 2 == 0);
    let pad_left = filter.len() / 2 - even;
    let pad_right = filter.len() / 2;
    let padded_time = time + pad_left + pad_right;
    if padded_time < filter.len() {
        return Err(VokraError::InvalidArgument(
            "alias_free_downsample: filter exceeds padded input".to_owned(),
        ));
    }
    let output_time = (padded_time - filter.len()) / ratio + 1;
    let mut output = vec![0.0f32; channels * output_time];
    for channel in 0..channels {
        let input_row = channel * time;
        let output_row = channel * output_time;
        for output_index in 0..output_time {
            let mut sum = 0.0f32;
            for (tap, &coefficient) in filter.iter().enumerate() {
                let padded_index = output_index * ratio + tap;
                let source = padded_index.saturating_sub(pad_left).min(time - 1);
                sum += input[input_row + source] * coefficient;
            }
            output[output_row + output_index] = sum;
        }
    }
    Ok(output)
}

/// Same-padded 1-D convolution.
///
/// Matches upstream PyTorch `Conv1d(in_ch, out_ch, kernel, stride=1,
/// padding=k/2)` with the standard same-length output. `input` is row-major
/// `[in_ch, t]`, `weight` is row-major `[out_ch, in_ch, kernel]`, `bias` is
/// `[out_ch]`. Output is row-major `[out_ch, t]`.
#[allow(clippy::too_many_arguments)]
fn conv1d_same_padding(
    input: &[f32],
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    padding: usize,
    t: usize,
    weight: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0f32; out_ch * t];
    let t_i = t as isize;
    let pad_i = padding as isize;
    for (oc, &b) in bias.iter().enumerate() {
        let row_offset = oc * t;
        let w_offset = oc * in_ch * kernel;
        for ti in 0..t {
            let mut acc = b;
            for ic in 0..in_ch {
                let x_row = ic * t;
                let w_row = w_offset + ic * kernel;
                for k in 0..kernel {
                    let src = ti as isize + k as isize - pad_i;
                    if src < 0 || src >= t_i {
                        continue;
                    }
                    acc += input[x_row + src as usize] * weight[w_row + k];
                }
            }
            output[row_offset + ti] = acc;
        }
    }
    output
}

/// Dilated same-padded 1-D convolution — same interface as
/// [`conv1d_same_padding`] plus an explicit `dilation`.
#[allow(clippy::too_many_arguments)]
fn conv1d_dilated_same_padding(
    input: &[f32],
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    dilation: usize,
    padding: usize,
    t: usize,
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>> {
    if input.len() != in_ch * t {
        return Err(VokraError::InvalidArgument(format!(
            "conv1d_dilated_same_padding: input length {} != in_ch * t = {}",
            input.len(),
            in_ch * t
        )));
    }
    if weight.len() != out_ch * in_ch * kernel {
        return Err(VokraError::InvalidArgument(format!(
            "conv1d_dilated_same_padding: weight length {} != out_ch * in_ch * kernel = {}",
            weight.len(),
            out_ch * in_ch * kernel
        )));
    }
    if bias.len() != out_ch {
        return Err(VokraError::InvalidArgument(format!(
            "conv1d_dilated_same_padding: bias length {} != out_ch = {out_ch}",
            bias.len()
        )));
    }
    if dilation == 0 || kernel == 0 {
        return Err(VokraError::InvalidArgument(
            "conv1d_dilated_same_padding: dilation and kernel must be > 0".to_owned(),
        ));
    }
    let mut output = vec![0.0f32; out_ch * t];
    let t_i = t as isize;
    let pad_i = padding as isize;
    let d_i = dilation as isize;
    for (oc, &b) in bias.iter().enumerate() {
        let row_offset = oc * t;
        let w_offset = oc * in_ch * kernel;
        for ti in 0..t {
            let mut acc = b;
            for ic in 0..in_ch {
                let x_row = ic * t;
                let w_row = w_offset + ic * kernel;
                for k in 0..kernel {
                    let src = ti as isize + k as isize * d_i - pad_i;
                    if src < 0 || src >= t_i {
                        continue;
                    }
                    acc += input[x_row + src as usize] * weight[w_row + k];
                }
            }
            output[row_offset + ti] = acc;
        }
    }
    Ok(output)
}

/// Transposed 1-D convolution.
///
/// `weight` layout is PyTorch's `nn.ConvTranspose1d` `[in_ch, out_ch,
/// kernel]`. Output length: `t_out = (t_in - 1) * stride + kernel - 2 *
/// padding`.
#[allow(clippy::too_many_arguments)]
fn conv_transpose1d(
    input: &[f32],
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    t_in: usize,
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>> {
    if input.len() != in_ch * t_in {
        return Err(VokraError::InvalidArgument(format!(
            "conv_transpose1d: input length {} != in_ch * t_in = {}",
            input.len(),
            in_ch * t_in
        )));
    }
    if weight.len() != in_ch * out_ch * kernel {
        return Err(VokraError::InvalidArgument(format!(
            "conv_transpose1d: weight length {} != in_ch * out_ch * kernel = {}",
            weight.len(),
            in_ch * out_ch * kernel
        )));
    }
    if bias.len() != out_ch {
        return Err(VokraError::InvalidArgument(format!(
            "conv_transpose1d: bias length {} != out_ch = {out_ch}",
            bias.len()
        )));
    }
    if stride == 0 {
        return Err(VokraError::InvalidArgument(
            "conv_transpose1d: stride must be > 0".to_owned(),
        ));
    }
    if t_in == 0 {
        return Err(VokraError::InvalidArgument(
            "conv_transpose1d: t_in must be > 0".to_owned(),
        ));
    }
    let core = (t_in - 1) * stride + kernel;
    if 2 * padding > core {
        return Err(VokraError::InvalidArgument(format!(
            "conv_transpose1d: 2*padding ({}) exceeds (t_in-1)*stride + kernel \
             ({core})",
            2 * padding
        )));
    }
    let t_out = core - 2 * padding;

    let mut output = vec![0.0f32; out_ch * t_out];
    for (oc, &b) in bias.iter().enumerate() {
        let row = oc * t_out;
        for slot in output[row..row + t_out].iter_mut() {
            *slot = b;
        }
    }
    for ic in 0..in_ch {
        let in_row = ic * t_in;
        for ti in 0..t_in {
            let x = input[in_row + ti];
            for oc in 0..out_ch {
                let w_off = ic * out_ch * kernel + oc * kernel;
                let out_row = oc * t_out;
                for k in 0..kernel {
                    let dst = (ti * stride + k) as isize - padding as isize;
                    if dst < 0 || dst >= t_out as isize {
                        continue;
                    }
                    output[out_row + dst as usize] += x * weight[w_off + k];
                }
            }
        }
    }
    Ok(output)
}

/// Upstream `get_padding(kernel_size, dilation)` returns
/// `(kernel_size * dilation - dilation) // 2 = dilation * (kernel_size - 1) / 2`
/// (upstream `utils.py:get_padding`, cited from `bigvgan.py:47`).
#[inline]
fn get_padding(kernel: usize, dilation: usize) -> usize {
    dilation * (kernel - 1) / 2
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_alias_filter() -> AliasFreeActivationWeights {
        AliasFreeActivationWeights {
            upsample_filter: vec![0.5, 0.5],
            downsample_filter: vec![0.5, 0.5],
        }
    }

    // ---- SnakeBeta ---------------------------------------------------

    #[test]
    fn snake_beta_new_rejects_empty_alpha() {
        let err = SnakeBeta::new(vec![], vec![], false).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("alpha vector must not be empty")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn snake_beta_new_rejects_length_mismatch() {
        let err = SnakeBeta::new(vec![1.0, 2.0], vec![1.0], false).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("alpha length 2 != beta length 1")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn snake_beta_channels_matches_alpha_len() {
        let sb = SnakeBeta::new(vec![1.0, 2.0, 3.0], vec![1.0, 1.0, 1.0], false).unwrap();
        assert_eq!(sb.channels(), 3);
    }

    #[test]
    fn snake_beta_forward_linear_scale_matches_reference() {
        // With alpha_logscale=false, alpha=1.0, beta=1.0, eps=1e-9:
        //   y = x + (1/(1+1e-9)) * sin(x)^2 ≈ x + sin(x)^2
        let sb = SnakeBeta::new(vec![1.0], vec![1.0], false).unwrap();
        let mut x = vec![0.5, -0.5, 1.5, -1.5];
        sb.forward_in_place(&mut x, 1, 4).unwrap();
        let expected = [
            0.5 + (0.5f32.sin()).powi(2) / (1.0 + 1e-9),
            -0.5 + ((-0.5f32).sin()).powi(2) / (1.0 + 1e-9),
            1.5 + (1.5f32.sin()).powi(2) / (1.0 + 1e-9),
            -1.5 + ((-1.5f32).sin()).powi(2) / (1.0 + 1e-9),
        ];
        for (a, b) in x.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "got {a}, expected {b}");
        }
    }

    #[test]
    fn snake_beta_forward_logscale_exponentiates_params() {
        // With alpha_logscale=true, stored alpha=0.0 → alpha_eff = exp(0)=1.0.
        // Same for beta. Result should match the linear case above.
        let sb = SnakeBeta::new(vec![0.0], vec![0.0], true).unwrap();
        let mut x = vec![0.7];
        sb.forward_in_place(&mut x, 1, 1).unwrap();
        let expected = 0.7 + (0.7f32.sin()).powi(2) / (1.0 + 1e-9);
        assert!(
            (x[0] - expected).abs() < 1e-6,
            "got {}, expected {expected}",
            x[0]
        );
    }

    #[test]
    fn snake_beta_forward_rejects_channel_mismatch() {
        let sb = SnakeBeta::new(vec![1.0, 2.0], vec![1.0, 1.0], false).unwrap();
        let err = sb.forward_in_place(&mut [0.0; 3], 3, 1).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("channels")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn snake_beta_forward_rejects_length_mismatch() {
        let sb = SnakeBeta::new(vec![1.0, 2.0], vec![1.0, 1.0], false).unwrap();
        let err = sb.forward_in_place(&mut [0.0; 5], 2, 3).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("input length")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn alias_free_activation_matches_upstream_bigvgan_fixture() {
        let fixture = include_str!("../../../tools/parity/fixtures/bigvgan_alias_free.csv");
        let mut lines = fixture.lines();
        let filter_row: Vec<&str> = lines.next().expect("filter row").split(',').collect();
        assert_eq!(filter_row[0], "filter");
        let filter: Vec<f32> = filter_row[1..]
            .iter()
            .map(|value| value.parse::<f32>().expect("filter f32"))
            .collect();
        assert_eq!(filter.len(), 12);

        let mut alpha = Vec::new();
        let mut beta = Vec::new();
        let mut input = Vec::new();
        let mut expected = Vec::new();
        for (channel, line) in lines.enumerate() {
            let fields: Vec<&str> = line.split(',').collect();
            assert_eq!(fields.len(), 18);
            assert_eq!(fields[0], "channel");
            assert_eq!(fields[1].parse::<usize>().unwrap(), channel);
            alpha.push(fields[2].parse::<f32>().unwrap());
            beta.push(fields[3].parse::<f32>().unwrap());
            input.extend(
                fields[4..11]
                    .iter()
                    .map(|value| value.parse::<f32>().unwrap()),
            );
            expected.extend(
                fields[11..18]
                    .iter()
                    .map(|value| value.parse::<f32>().unwrap()),
            );
        }
        let channels = alpha.len();
        assert_eq!(channels, 2);
        let activation =
            AmpActivation::SnakeBeta(SnakeBeta::new(alpha, beta, true).expect("fixture SnakeBeta"));
        let alias_free = AliasFreeActivation::new(
            activation,
            AliasFreeActivationWeights {
                upsample_filter: filter.clone(),
                downsample_filter: filter,
            },
        )
        .expect("fixture alias-free activation");
        alias_free
            .forward_in_place(&mut input, channels, 7)
            .expect("alias-free forward");
        let max_abs = input
            .iter()
            .zip(expected.iter())
            .map(|(actual, reference)| (actual - reference).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 3e-6,
            "BigVGAN upstream Activation1d max |Δ| {max_abs:e} exceeds 3e-6"
        );
    }

    // ---- AmpBlock1 ---------------------------------------------------

    fn snake_amp_weights(ch: usize, k: usize, n_branches: usize) -> AmpBlock1Weights {
        let expected_w = ch * ch * k;
        AmpBlock1Weights {
            convs1_w: vec![vec![0.0f32; expected_w]; n_branches],
            convs1_b: vec![vec![0.0f32; ch]; n_branches],
            convs2_w: vec![vec![0.0f32; expected_w]; n_branches],
            convs2_b: vec![vec![0.0f32; ch]; n_branches],
            activations1_alpha: vec![vec![1.0f32; ch]; n_branches],
            activations2_alpha: vec![vec![1.0f32; ch]; n_branches],
            activations1_beta: None,
            activations2_beta: None,
            activations1_filters: vec![test_alias_filter(); n_branches],
            activations2_filters: vec![test_alias_filter(); n_branches],
        }
    }

    fn snakebeta_amp_weights(ch: usize, k: usize, n_branches: usize) -> AmpBlock1Weights {
        let mut w = snake_amp_weights(ch, k, n_branches);
        w.activations1_beta = Some(vec![vec![1.0f32; ch]; n_branches]);
        w.activations2_beta = Some(vec![vec![1.0f32; ch]; n_branches]);
        w
    }

    #[test]
    fn amp_block1_snake_new_ok_and_forward_preserves_shape() {
        let w = snake_amp_weights(4, 3, 3);
        let blk = AmpBlock1::new(4, 3, vec![1, 3, 5], SnakeKind::Snake, false, w).unwrap();
        let mut x = vec![0.1f32; 4 * 8];
        let before = x.clone();
        blk.forward_in_place(&mut x, 8).unwrap();
        // With zero conv weights, `xt = a(bias=0*x + 0) = a(0)`, then next
        // conv with zero weight+bias produces 0, so residual should leave x
        // unchanged. This exercises the shape flow end-to-end.
        assert_eq!(x, before);
    }

    #[test]
    fn amp_block1_snakebeta_new_ok() {
        let w = snakebeta_amp_weights(4, 3, 3);
        let blk = AmpBlock1::new(4, 3, vec![1, 3, 5], SnakeKind::SnakeBeta, true, w).unwrap();
        let mut x = vec![0.0f32; 4 * 6];
        blk.forward_in_place(&mut x, 6).unwrap();
        assert_eq!(x.len(), 24);
    }

    #[test]
    fn amp_block1_rejects_empty_dilations() {
        let w = snake_amp_weights(4, 3, 1);
        // Pass empty dilations but n_branches weights — expected to fail on
        // the dilations check first.
        let err = AmpBlock1::new(4, 3, vec![], SnakeKind::Snake, false, w).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("dilations must not be empty")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn amp_block1_rejects_zero_channels() {
        let w = snake_amp_weights(4, 3, 3);
        let err = AmpBlock1::new(0, 3, vec![1, 3, 5], SnakeKind::Snake, false, w).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("channels and kernel_size must be > 0")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn amp_block1_rejects_conv_weight_length_mismatch() {
        let mut w = snake_amp_weights(4, 3, 3);
        w.convs1_w[1] = vec![0.0; 10]; // wrong length
        let err = AmpBlock1::new(4, 3, vec![1, 3, 5], SnakeKind::Snake, false, w).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("convs1_w[1]")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn amp_block1_snake_forbids_beta() {
        let mut w = snake_amp_weights(4, 3, 3);
        w.activations1_beta = Some(vec![vec![1.0; 4]; 3]);
        let err = AmpBlock1::new(4, 3, vec![1, 3, 5], SnakeKind::Snake, false, w).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("must be None when")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn amp_block1_snakebeta_requires_beta() {
        let w = snake_amp_weights(4, 3, 3); // No beta populated
        let err = AmpBlock1::new(4, 3, vec![1, 3, 5], SnakeKind::SnakeBeta, false, w).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("must be Some when")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn amp_block1_forward_rejects_input_length_mismatch() {
        let w = snake_amp_weights(4, 3, 3);
        let blk = AmpBlock1::new(4, 3, vec![1, 3, 5], SnakeKind::Snake, false, w).unwrap();
        let mut x = vec![0.0f32; 4 * 8 - 1];
        let err = blk.forward_in_place(&mut x, 8).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("input length")),
            "unexpected error: {err:?}"
        );
    }

    // ---- BigVGanConfig -----------------------------------------------

    #[test]
    fn bigvgan_config_defaults_match_v2_24k_100band_256x() {
        let cfg = BigVGanConfig::default();
        assert_eq!(cfg.in_channels, 100);
        assert_eq!(cfg.upsample_initial_channel, 1536);
        assert_eq!(cfg.upsample_rates, vec![4, 4, 2, 2, 2, 2]);
        assert_eq!(cfg.upsample_kernel_sizes, vec![8, 8, 4, 4, 4, 4]);
        assert_eq!(cfg.resblock_kernel_sizes, vec![3, 7, 11]);
        assert_eq!(cfg.activation, SnakeKind::SnakeBeta);
        assert!(cfg.snake_logscale);
        assert!(!cfg.use_bias_at_final);
        assert!(!cfg.use_tanh_at_final);
        assert_eq!(cfg.total_upsample_factor(), 4 * 4 * 2 * 2 * 2 * 2);
        assert_eq!(cfg.total_upsample_factor(), 256);
        assert_eq!(cfg.num_upsamples(), 6);
        assert_eq!(cfg.num_kernels(), 3);
        // Stage 0 halves, stage 1 quarters, ..., stage 5 → 1536 / 64 = 24.
        assert_eq!(cfg.output_channels_at(0), 768);
        assert_eq!(cfg.output_channels_at(5), 24);
    }

    // ---- BigVGanGenerator: mini synthetic config for shape flow -------

    /// Build a mini `(cfg, weights)` bundle for shape-flow testing. Two
    /// upsample stages, one MRF branch each, small channel counts so the
    /// scalar loops finish in ms.
    fn mini_bundle(
        activation: SnakeKind,
        logscale: bool,
        tanh: bool,
        bias: bool,
    ) -> (BigVGanConfig, BigVGanWeights) {
        let cfg = BigVGanConfig {
            in_channels: 4,
            upsample_initial_channel: 8, // stage 0 out=4, stage 1 out=2
            upsample_rates: vec![2, 2],
            upsample_kernel_sizes: vec![4, 4],
            resblock_kernel_sizes: vec![3],
            resblock_dilation_sizes: vec![vec![1, 3]],
            activation,
            snake_logscale: logscale,
            use_bias_at_final: bias,
            use_tanh_at_final: tanh,
        };
        let bc = cfg.upsample_initial_channel as usize;
        let inc = cfg.in_channels as usize;
        let weights = BigVGanWeights {
            conv_pre_w: vec![0.0f32; bc * inc * 7],
            conv_pre_b: vec![0.0f32; bc],
            ups_w: (0..cfg.num_upsamples())
                .map(|i| {
                    let in_ch = (cfg.upsample_initial_channel >> (i as u32)) as usize;
                    let out_ch = (cfg.upsample_initial_channel >> (i as u32 + 1)) as usize;
                    let k = cfg.upsample_kernel_sizes[i] as usize;
                    vec![0.0f32; in_ch * out_ch * k]
                })
                .collect(),
            ups_b: (0..cfg.num_upsamples())
                .map(|i| {
                    let out_ch = (cfg.upsample_initial_channel >> (i as u32 + 1)) as usize;
                    vec![0.0f32; out_ch]
                })
                .collect(),
            amp_blocks: {
                let mut v = Vec::new();
                for i in 0..cfg.num_upsamples() {
                    let ch = cfg.output_channels_at(i) as usize;
                    for j in 0..cfg.num_kernels() {
                        let k = cfg.resblock_kernel_sizes[j] as usize;
                        let n_branches = cfg.resblock_dilation_sizes[j].len();
                        let mut w = snake_amp_weights(ch, k, n_branches);
                        if matches!(activation, SnakeKind::SnakeBeta) {
                            w.activations1_beta = Some(vec![vec![1.0f32; ch]; n_branches]);
                            w.activations2_beta = Some(vec![vec![1.0f32; ch]; n_branches]);
                        }
                        v.push(w);
                    }
                }
                v
            },
            activation_post_alpha: vec![
                1.0f32;
                cfg.output_channels_at(cfg.num_upsamples() - 1) as usize
            ],
            activation_post_beta: if matches!(activation, SnakeKind::SnakeBeta) {
                Some(vec![
                    1.0f32;
                    cfg.output_channels_at(cfg.num_upsamples() - 1) as usize
                ])
            } else {
                None
            },
            activation_post_filter: test_alias_filter(),
            conv_post_w: vec![
                0.0f32;
                (cfg.output_channels_at(cfg.num_upsamples() - 1) as usize) * 7
            ],
            conv_post_b: if bias { Some(vec![0.0f32]) } else { None },
        };
        (cfg, weights)
    }

    #[test]
    fn bigvgan_generator_new_snake_ok() {
        let (cfg, w) = mini_bundle(SnakeKind::Snake, false, true, true);
        let vg = BigVGanGenerator::new(cfg, w).unwrap();
        assert_eq!(vg.config().num_upsamples(), 2);
    }

    #[test]
    fn bigvgan_generator_new_snakebeta_ok() {
        let (cfg, w) = mini_bundle(SnakeKind::SnakeBeta, true, true, true);
        let _vg = BigVGanGenerator::new(cfg, w).unwrap();
    }

    #[test]
    fn bigvgan_generator_forward_output_length_matches_total_upsample() {
        let (cfg, w) = mini_bundle(SnakeKind::Snake, false, true, true);
        let t_mel = 5;
        let n_out = t_mel * cfg.total_upsample_factor() as usize;
        let vg = BigVGanGenerator::new(cfg, w).unwrap();
        let mel = vec![0.1f32; 4 * t_mel];
        let y = vg.forward(&mel, t_mel).unwrap();
        assert_eq!(y.len(), n_out);
        // All-zero weights + bias produce all-zero pre-tanh signal → tanh(0)=0.
        assert!(y.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn bigvgan_generator_forward_output_bounded_by_tanh() {
        // Non-trivial (all-ones) input + non-trivial random-ish weights: the
        // terminal `tanh` must map the output into `[-1, 1]`. Mathematically
        // `tanh: R → (-1, 1)` is open, but f32 arithmetic saturates at
        // exactly ±1.0 for large enough inputs (tanh(9.02) ≈ 0.99999997 in
        // f32 rounds to 1.0), so the inclusive bound is what the runtime
        // actually guarantees.
        let (cfg, mut w) = mini_bundle(SnakeKind::Snake, false, true, true);
        w.conv_post_w = vec![100.0f32; w.conv_post_w.len()];
        w.conv_post_b = Some(vec![50.0f32]);
        w.conv_pre_w = vec![1.0f32; w.conv_pre_w.len()];
        let vg = BigVGanGenerator::new(cfg, w).unwrap();
        let mel = vec![1.0f32; 4 * 3];
        let y = vg.forward(&mel, 3).unwrap();
        assert!(!y.is_empty());
        for &v in &y {
            assert!((-1.0..=1.0).contains(&v), "unbounded output: {v}");
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    #[test]
    fn bigvgan_generator_forward_clamp_when_tanh_disabled() {
        // Terminal `clamp(-1, 1)` bounds output into [-1, 1] inclusive.
        let (cfg, mut w) = mini_bundle(SnakeKind::Snake, false, false, true);
        // Force a huge pre-clamp value.
        w.conv_post_w = vec![1000.0f32; w.conv_post_w.len()];
        w.conv_post_b = Some(vec![500.0f32]);
        w.conv_pre_w = vec![1.0f32; w.conv_pre_w.len()];
        let vg = BigVGanGenerator::new(cfg, w).unwrap();
        let mel = vec![1.0f32; 4 * 3];
        let y = vg.forward(&mel, 3).unwrap();
        for &v in &y {
            assert!((-1.0..=1.0).contains(&v), "clamp violated: {v}");
        }
    }

    #[test]
    fn bigvgan_generator_forward_rejects_zero_t_mel() {
        let (cfg, w) = mini_bundle(SnakeKind::Snake, false, true, true);
        let vg = BigVGanGenerator::new(cfg, w).unwrap();
        let err = vg.forward(&[], 0).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("t_mel must be > 0")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn bigvgan_generator_forward_rejects_mel_length_mismatch() {
        let (cfg, w) = mini_bundle(SnakeKind::Snake, false, true, true);
        let vg = BigVGanGenerator::new(cfg, w).unwrap();
        let err = vg.forward(&[0.0; 3], 5).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("mel length")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn bigvgan_new_rejects_upsample_rates_kernel_length_mismatch() {
        let (mut cfg, w) = mini_bundle(SnakeKind::Snake, false, true, true);
        cfg.upsample_kernel_sizes = vec![4]; // len 1 vs upsample_rates len 2
        let err = BigVGanGenerator::new(cfg, w).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("upsample_kernel_sizes")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn bigvgan_new_rejects_resblock_dilation_length_mismatch() {
        let (mut cfg, w) = mini_bundle(SnakeKind::Snake, false, true, true);
        cfg.resblock_dilation_sizes = vec![]; // len 0 vs resblock_kernel_sizes len 1
        let err = BigVGanGenerator::new(cfg, w).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("resblock_dilation_sizes")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn bigvgan_new_rejects_empty_upsample_rates() {
        let (mut cfg, mut w) = mini_bundle(SnakeKind::Snake, false, true, true);
        cfg.upsample_rates = vec![];
        cfg.upsample_kernel_sizes = vec![];
        w.ups_w.clear();
        w.ups_b.clear();
        w.amp_blocks.clear();
        let err = BigVGanGenerator::new(cfg, w).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("upsample_rates must not be empty")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn bigvgan_new_rejects_zero_in_channels() {
        let (mut cfg, w) = mini_bundle(SnakeKind::Snake, false, true, true);
        cfg.in_channels = 0;
        let err = BigVGanGenerator::new(cfg, w).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("in_channels and upsample_initial_channel must be > 0")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn bigvgan_new_rejects_conv_pre_weight_shape() {
        let (cfg, mut w) = mini_bundle(SnakeKind::Snake, false, true, true);
        w.conv_pre_w = vec![0.0; 10]; // wrong length
        let err = BigVGanGenerator::new(cfg, w).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("conv_pre_w")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn bigvgan_new_rejects_kernel_smaller_than_stride() {
        let (mut cfg, mut w) = mini_bundle(SnakeKind::Snake, false, true, true);
        cfg.upsample_kernel_sizes[0] = 1; // kernel < stride (2)
        let bc = cfg.upsample_initial_channel as usize;
        let in_ch = bc;
        let out_ch = bc / 2;
        // Layout is `[in_ch, out_ch, kernel]` with kernel=1.
        w.ups_w[0] = vec![0.0; in_ch * out_ch];
        let err = BigVGanGenerator::new(cfg, w).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("kernel") && m.contains("< stride")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn bigvgan_new_rejects_amp_block_count_mismatch() {
        let (cfg, mut w) = mini_bundle(SnakeKind::Snake, false, true, true);
        w.amp_blocks.pop(); // Now 1 fewer block than expected
        let err = BigVGanGenerator::new(cfg, w).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("amp_blocks: expected")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn bigvgan_new_snake_forbids_activation_post_beta() {
        let (cfg, mut w) = mini_bundle(SnakeKind::Snake, false, true, true);
        w.activation_post_beta = Some(vec![1.0; 2]);
        let err = BigVGanGenerator::new(cfg, w).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("activation_post_beta must be None")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn bigvgan_new_snakebeta_requires_activation_post_beta() {
        let (cfg, mut w) = mini_bundle(SnakeKind::SnakeBeta, true, true, true);
        w.activation_post_beta = None;
        let err = BigVGanGenerator::new(cfg, w).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("activation_post_beta must be Some")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn bigvgan_new_bias_at_final_true_requires_conv_post_b() {
        let (cfg, mut w) = mini_bundle(SnakeKind::Snake, false, true, true);
        w.conv_post_b = None;
        let err = BigVGanGenerator::new(cfg, w).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("conv_post_b must be Some")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn bigvgan_new_bias_at_final_false_forbids_conv_post_b() {
        let (cfg, mut w) = mini_bundle(SnakeKind::Snake, false, true, false);
        w.conv_post_b = Some(vec![0.0]); // Should have been None
        let err = BigVGanGenerator::new(cfg, w).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("conv_post_b must be None")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn bigvgan_new_rejects_zero_last_stage_channels() {
        // 4 stages on 8 channels → out at stage 3 = 8 >> 4 = 0.
        let cfg = BigVGanConfig {
            in_channels: 4,
            upsample_initial_channel: 8,
            upsample_rates: vec![2, 2, 2, 2],
            upsample_kernel_sizes: vec![4, 4, 4, 4],
            resblock_kernel_sizes: vec![3],
            resblock_dilation_sizes: vec![vec![1]],
            activation: SnakeKind::Snake,
            snake_logscale: false,
            use_bias_at_final: true,
            use_tanh_at_final: true,
        };
        // Populate any weights — the shape check should fire before conv checks.
        let bc = cfg.upsample_initial_channel as usize;
        let inc = cfg.in_channels as usize;
        let w = BigVGanWeights {
            conv_pre_w: vec![0.0; bc * inc * 7],
            conv_pre_b: vec![0.0; bc],
            ups_w: (0..cfg.num_upsamples())
                .map(|i| {
                    let in_ch = (cfg.upsample_initial_channel >> (i as u32)) as usize;
                    let out_ch = (cfg.upsample_initial_channel >> (i as u32 + 1)) as usize;
                    let k = cfg.upsample_kernel_sizes[i] as usize;
                    vec![0.0f32; in_ch * out_ch * k]
                })
                .collect(),
            ups_b: (0..cfg.num_upsamples())
                .map(|i| {
                    let out_ch = (cfg.upsample_initial_channel >> (i as u32 + 1)) as usize;
                    vec![0.0f32; out_ch]
                })
                .collect(),
            amp_blocks: vec![],
            activation_post_alpha: vec![],
            activation_post_beta: None,
            activation_post_filter: test_alias_filter(),
            conv_post_w: vec![],
            conv_post_b: Some(vec![0.0]),
        };
        let err = BigVGanGenerator::new(cfg, w).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("out_ch is 0")),
            "unexpected error: {err:?}"
        );
    }
}
