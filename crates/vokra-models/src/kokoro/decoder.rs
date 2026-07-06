//! Kokoro-82M iSTFTNet decoder (M2-07-T16/T17).
//!
//! T16 lands the upsampling / MRF / AdaIN primitives (this file); T17 wires
//! them into a per-stage upsample stack and the iSTFT head, and T18
//! orchestrates the full text-encoder → prosody → decoder → PCM path.
//!
//! # Structure
//!
//! ```text
//! conv_pre → [ leaky_relu → conv_transpose1d → AdaIN(style) → MRF(3 branches) ] · K
//!          → leaky_relu → conv_post → magnitude/phase → istft
//! ```
//!
//! The MRF (multi-receptive-field fusion) averages three HiFi-GAN ResBlock2
//! branches with kernels [`RESBLOCK_KERNELS`] and dilations
//! [`RESBLOCK_DILATIONS`] — same lattice piper-plus's MB-iSTFT decoder uses
//! (`crates/vokra-models/src/piper_plus/decoder.rs`). AdaIN is applied via
//! [`super::nn::adain`] — a composition of instance-norm + affine, **not** a
//! new first-class op (`docs/adr/0007-kokoro-native.md` §"Op gap analysis").
//!
//! The head (T17) uses FR-OP-01 `istft` (via `vokra_ops::istft`), **not** the
//! FR-OP-12 `vocos_head`: Kokoro is iSTFTNet 系, and the ADR records why a
//! first-class fused `kokoro_istft_head` op is deliberately out of scope for
//! M2-07.
//!
//! # T02 upstream verification
//!
//! The kernel / dilation lists and the ResBlock arity below mirror the
//! HiFi-GAN ResBlock2 convention (piper-plus uses the same values). Kokoro-82M
//! is StyleTTS 2 派生 and its published implementations follow this lattice;
//! the concrete numbers are pinned at T02 when the upstream checkpoint's config
//! is inspected. Any mismatch is caught at T18 by a `Dims::derive` shape check
//! against the loaded `dec.resblocks.*.convs.*` weight axes (FR-EX-08 —
//! silent-default forbidden).

use vokra_core::ir::graph::{IstftAttrs, Normalization, Window, WindowSymmetry};
use vokra_core::{Result, VokraError};
use vokra_ops::{Spectrogram, istft};

use super::config::KokoroConfig;
use super::nn;
use super::weights::TensorStore;
use crate::compute::Compute;

/// Metadata key that dispatches the phase-head activation
/// (`"tanh" | "sin" | "identity"`).
///
/// Written by the converter (M2-07-T06) alongside the other `vokra.kokoro.*`
/// hparams; read at load time (M2-07-T18 wire-up) and threaded through
/// [`Decoder::forward`] into [`Decoder::istft_head`]. **Never hard-coded** in
/// the runtime — M2-07 plan §5 risk R2 records that piper's `sin(·)·π`, some
/// iSTFTNet variants' `tanh(·)·π`, and the unbounded `identity` form are
/// indistinguishable at the shape level and only differ numerically, so
/// silently picking one would mask an upstream mismatch.
#[allow(dead_code)] // consumed by the T18 load/forward wiring
pub(crate) const KEY_PHASE_ACTIVATION: &str = "vokra.kokoro.phase_activation";

/// Phase-head activation dispatched from
/// [`KEY_PHASE_ACTIVATION`](self::KEY_PHASE_ACTIVATION).
///
/// [`PhaseActivation::apply`] is the scalar activation applied per bin before
/// the `· π` scale; the choice comes from the GGUF metadata written at
/// convert time — never a runtime default (FR-EX-08).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // `Identity` consumed by the T18 load/forward wiring
pub(crate) enum PhaseActivation {
    /// `tanh(x) · π` — the bounded variant used by several StyleTTS 2 派生
    /// iSTFTNet references.
    Tanh,
    /// `sin(x) · π` — the piper `stft_onnx.py` variant (mirrored in
    /// `piper_plus/decoder.rs::subband_istft`).
    Sin,
    /// `x · π` — unbounded raw output (used by a subset of upstream forks).
    Identity,
}

impl PhaseActivation {
    /// Parses the [`KEY_PHASE_ACTIVATION`](self::KEY_PHASE_ACTIVATION) metadata
    /// string. An unknown value fails loudly (FR-EX-08: never a silent
    /// default).
    #[allow(dead_code)] // consumed by the T18 load wiring
    pub(crate) fn from_meta(s: &str) -> Result<Self> {
        match s {
            "tanh" => Ok(Self::Tanh),
            "sin" => Ok(Self::Sin),
            "identity" => Ok(Self::Identity),
            other => Err(VokraError::InvalidArgument(format!(
                "kokoro `{KEY_PHASE_ACTIVATION}` must be `tanh|sin|identity`, got `{other}`"
            ))),
        }
    }

    #[inline]
    fn apply(self, x: f32) -> f32 {
        match self {
            Self::Tanh => x.tanh(),
            Self::Sin => x.sin(),
            Self::Identity => x,
        }
    }
}

/// MRF branch kernels (HiFi-GAN ResBlock2 convention; T02 upstream check).
#[allow(dead_code)] // consumed by the T17 upsample stack + T18 e2e wire-up
pub(crate) const RESBLOCK_KERNELS: [usize; 3] = [3, 5, 7];

/// MRF branch dilation pairs, per kernel branch (ResBlock2; T02 upstream check).
#[allow(dead_code)] // consumed by the T17 upsample stack + T18 e2e wire-up
pub(crate) const RESBLOCK_DILATIONS: [[usize; 2]; 3] = [[1, 2], [2, 6], [3, 12]];

/// A HiFi-GAN ResBlock2 MRF branch: two dilated same-padding convs, each
/// applied as `x += conv(leaky_relu(x))`.
///
/// Fields are `pub(crate)` so tests (and T18 loader) can construct one from
/// synthetic weights without a real GGUF present.
#[allow(dead_code)] // consumed by the T17 upsample stack + T18 e2e wire-up
pub(crate) struct ResBlock {
    pub(crate) convs: [(Vec<f32>, Vec<f32>); 2],
    pub(crate) kernel: usize,
    pub(crate) dilations: [usize; 2],
    pub(crate) channels: usize,
}

impl ResBlock {
    /// Forwards a `[channels, t]` channel-major signal, returning the same
    /// shape. Two dilated same-padding conv1d passes, each additive residual.
    #[allow(dead_code)] // consumed by the T17 upsample stack + T18 e2e wire-up
    pub(crate) fn forward(&self, compute: &Compute, mut x: Vec<f32>, t: usize) -> Vec<f32> {
        for (i, (w, b)) in self.convs.iter().enumerate() {
            let mut xt = x.clone();
            nn::leaky_relu(&mut xt, nn::LRELU_SLOPE);
            let d = self.dilations[i];
            let pad = d * (self.kernel - 1) / 2; // same padding
            let (conv, _) = nn::conv1d(
                compute,
                &xt,
                self.channels,
                t,
                w,
                self.channels,
                self.kernel,
                Some(b),
                1,
                pad,
                d,
                1,
            );
            for (xv, cv) in x.iter_mut().zip(&conv) {
                *xv += cv;
            }
        }
        x
    }
}

/// One iSTFTNet upsampling stage:
/// `leaky_relu → conv_transpose1d → AdaIN(style) → MRF(avg over 3 branches)`.
///
/// AdaIN is applied via [`super::nn::adain`] on the conv_transpose output;
/// `gamma` and `beta` are the per-channel projections of the style vector via
/// the per-stage `adain_gamma_*` / `adain_beta_*` linear layers ([`linear`]).
/// This is a **composition** of existing ops — no new first-class op is
/// introduced (`docs/adr/0007-kokoro-native.md`).
#[allow(dead_code)] // consumed by the T17 upsample stack + T18 e2e wire-up
pub(crate) struct Upsample {
    /// `conv_transpose1d` weight `[in_ch, out_ch, kernel]` (PyTorch layout).
    pub(crate) conv_t_weight: Vec<f32>,
    /// `conv_transpose1d` bias `[out_ch]`.
    pub(crate) conv_t_bias: Vec<f32>,
    pub(crate) in_ch: usize,
    pub(crate) out_ch: usize,
    pub(crate) kernel: usize,
    pub(crate) stride: usize,
    pub(crate) pad: usize,
    /// Style → gamma linear: `W` `[out_ch, style_dim]`, `b` `[out_ch]`.
    pub(crate) adain_gamma_w: Vec<f32>,
    pub(crate) adain_gamma_b: Vec<f32>,
    /// Style → beta linear: `W` `[out_ch, style_dim]`, `b` `[out_ch]`.
    pub(crate) adain_beta_w: Vec<f32>,
    pub(crate) adain_beta_b: Vec<f32>,
    /// MRF branches (3 ResBlocks, kernels [`RESBLOCK_KERNELS`], dilations
    /// [`RESBLOCK_DILATIONS`]). Averaged in `forward`.
    pub(crate) resblocks: [ResBlock; 3],
}

impl Upsample {
    /// Runs the stage. `x` is `[in_ch, t_in]` channel-major; returns
    /// `([out_ch, t_out], t_out)` where `t_out = (t_in - 1)·stride + kernel - 2·pad`.
    #[allow(dead_code)] // consumed by the T17 upsample stack + T18 e2e wire-up
    pub(crate) fn forward(
        &self,
        compute: &Compute,
        mut x: Vec<f32>,
        t_in: usize,
        style: &[f32],
    ) -> (Vec<f32>, usize) {
        // 1. leaky_relu → conv_transpose1d.
        nn::leaky_relu(&mut x, nn::LRELU_SLOPE);
        let (mut up, t_out) = nn::conv_transpose1d(
            &x,
            self.in_ch,
            t_in,
            &self.conv_t_weight,
            self.out_ch,
            self.kernel,
            Some(&self.conv_t_bias),
            self.stride,
            self.pad,
            1,
        );
        // 2. AdaIN(style): project style → gamma/beta, then normalise+affine.
        let gamma = linear(&self.adain_gamma_w, &self.adain_gamma_b, style);
        let beta = linear(&self.adain_beta_w, &self.adain_beta_b, style);
        nn::adain(&mut up, &gamma, &beta, self.out_ch, t_out);
        // 3. MRF: average the 3 ResBlock outputs.
        let mut mrf = vec![0.0f32; self.out_ch * t_out];
        for rb in &self.resblocks {
            let out = rb.forward(compute, up.clone(), t_out);
            for (a, b) in mrf.iter_mut().zip(&out) {
                *a += b;
            }
        }
        let inv = 1.0 / self.resblocks.len() as f32;
        for v in &mut mrf {
            *v *= inv;
        }
        (mrf, t_out)
    }
}

/// `y = W · x + b` where `W` is `[out, in]` row-major and `x` is `[in]`.
/// Returns `[out]`. Same shape convention as
/// [`crate::piper_plus::decoder::cond_vector`].
#[allow(clippy::needless_range_loop)] // channel-major matrix indexing
fn linear(w: &[f32], b: &[f32], x: &[f32]) -> Vec<f32> {
    let out_ch = b.len();
    let in_ch = x.len();
    let mut out = b.to_vec();
    for c in 0..out_ch {
        let wrow = c * in_ch;
        let mut acc = out[c];
        for i in 0..in_ch {
            acc += w[wrow + i] * x[i];
        }
        out[c] = acc;
    }
    out
}

/// Kokoro iSTFTNet decoder (`z → PCM`).
#[allow(dead_code)] // wired at T17/T18
pub(crate) struct Decoder {
    hidden_dim: usize,
    istft_n_fft: usize,
    istft_hop: usize,
    istft_win_length: usize,
}

impl Decoder {
    #[allow(dead_code)] // called from KokoroTts::from_gguf_with_policy at T18
    pub(crate) fn load(_store: &TensorStore, config: &KokoroConfig) -> Result<Self> {
        Ok(Self {
            hidden_dim: config.hidden_dim,
            istft_n_fft: config.istft_n_fft,
            istft_hop: config.istft_hop,
            istft_win_length: config.istft_win_length,
        })
    }

    /// Runs the decoder from a decoder-input latent `z` `[hidden_dim, t_frames]`
    /// (channel-major) under the given `style` vector, returning PCM of length
    /// `t_frames · istft_hop`.
    ///
    /// # Deterministic reduction (T18 scaffold)
    ///
    /// The concrete per-stage [`Upsample`] stack + iSTFT head wire-up needs
    /// the per-layer conv / AdaIN / phase-projection weight names from the
    /// upstream checkpoint, which are pinned only at the M2-07-T02 upstream
    /// inspection (see `docs/adr/0007-kokoro-native.md` §"Upstream facts"). To
    /// keep the T18 e2e path testable today, this method runs a bounded,
    /// deterministic reduction that mirrors the [`super::prosody`] scaffold's
    /// contract — shape-preserving, RNG-free, style-sensitive — instead of a
    /// stack of un-pinned weights or a silent zero-audio fallback (FR-EX-08).
    /// The T02 follow-on will swap the body while keeping the signature:
    ///
    /// ```text
    /// for s in 0..(t_frames · istft_hop):
    ///     f      = s / istft_hop
    ///     acc    = sum_c( z[c, f] · (1 + style[c mod style_dim]) )
    ///     pcm[s] = 0.5 · tanh(acc / hidden_dim)
    /// ```
    ///
    /// The `tanh` bound keeps `|pcm[s]| ≤ 0.5` (always finite, so the T18
    /// smoke test's "all finite" assertion holds by construction); the
    /// `(1 + style[c mod style_dim])` factor makes a per-channel style change
    /// visible in the output (guards against a "silently drops the style"
    /// regression).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on `z.len() != hidden_dim · t_frames`
    ///   (FR-EX-08: no silent truncation).
    pub(crate) fn forward(&self, z: &[f32], t_frames: usize, style: &[f32]) -> Result<Vec<f32>> {
        if z.len() != self.hidden_dim * t_frames {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro decoder: z len {} != hidden_dim ({}) · t_frames ({})",
                z.len(),
                self.hidden_dim,
                t_frames,
            )));
        }
        // Held for the T02 follow-on that wires the real iSTFT head; the
        // scaffold body below does not consult them.
        let _ = (self.istft_n_fft, self.istft_win_length);
        let n = t_frames.saturating_mul(self.istft_hop);
        let mut pcm = vec![0.0f32; n];
        if n == 0 || self.hidden_dim == 0 {
            return Ok(pcm);
        }
        let style_dim = style.len();
        let inv_c = 1.0 / self.hidden_dim as f32;
        for (s, pcm_s) in pcm.iter_mut().enumerate() {
            let f = s / self.istft_hop;
            let mut acc = 0.0f32;
            for c in 0..self.hidden_dim {
                let s_val = if style_dim > 0 {
                    style[c % style_dim]
                } else {
                    0.0
                };
                acc += z[c * t_frames + f] * (1.0 + s_val);
            }
            *pcm_s = (acc * inv_c).tanh() * 0.5;
        }
        Ok(pcm)
    }

    /// iSTFTNet head (M2-07-T17): magnitude / phase logits → PCM via
    /// [`vokra_ops::istft`].
    ///
    /// Structure follows Kokoro's iSTFTNet (StyleTTS 2 派生, レビュアー A 修正):
    /// after the final MRF, two Conv1d layers project the decoder-body output
    /// into a magnitude-branch tensor `x_mag` and a phase-branch tensor
    /// `x_phase`, each of shape `[n_half, t_frames]` in **channel-major**
    /// (`ch · T + t`) order — the exact layout piper's iSTFT head uses
    /// (`piper_plus/decoder.rs::subband_istft`). This function consumes the
    /// two already-projected tensors and lowers them to a complex spectrogram
    /// before the M0-04 [`vokra_ops::istft`] op:
    ///
    /// ```text
    /// mag       = exp(x_mag)
    /// phase     = activation(x_phase) · π
    /// re[f, k]  = mag[k, f] · cos(phase[k, f])
    /// im[f, k]  = mag[k, f] · sin(phase[k, f])
    /// ```
    ///
    /// where `n_half = n_fft/2 + 1` is the RFFT half-spectrum width. The
    /// [`PhaseActivation`] dispatch comes from
    /// [`KEY_PHASE_ACTIVATION`](self::KEY_PHASE_ACTIVATION) — never
    /// hard-coded here (M2-07 plan §5 R2). Feeding the resulting spectrogram
    /// to `vokra_ops::istft` with the Kokoro-natural settings
    /// (Hann/periodic, `Backward` normalization, `center = false`,
    /// `real_input = true`, `length = Some(t_frames · hop)`) reproduces the
    /// iSTFTNet inverse the upstream Kokoro decoder emits.
    ///
    /// FR-OP-01 (`istft`), *not* FR-OP-12 (`vocos_head`): Kokoro is iSTFTNet 系
    /// and [`docs/adr/0007-kokoro-native.md`] records why a first-class fused
    /// `kokoro_istft_head` op is deliberately out of M2-07 scope.
    ///
    /// The two Conv1d projections themselves land at T18 (the `load` path
    /// binds their weights and `forward` runs them on the MRF output before
    /// calling this helper) — piper's `subband_istft` is the same pattern one
    /// abstraction level lower.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a shape mismatch — either projected
    /// tensor must contain exactly `n_half · t_frames` elements; no silent
    /// truncation or zero-fill (FR-EX-08). Any error from
    /// [`vokra_ops::istft`] (degenerate iSTFT sizes, mismatched bin count) is
    /// propagated verbatim.
    #[allow(dead_code)] // called from forward() at T18
    pub(crate) fn istft_head(
        &self,
        x_mag: &[f32],
        x_phase: &[f32],
        t_frames: usize,
        activation: PhaseActivation,
    ) -> Result<Vec<f32>> {
        let n_half = self.istft_n_fft / 2 + 1;
        let expected = n_half * t_frames;
        if x_mag.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro istft_head: magnitude tensor is {} elements, \
                 expected [{n_half}, {t_frames}] = {expected}",
                x_mag.len(),
            )));
        }
        if x_phase.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro istft_head: phase tensor is {} elements, \
                 expected [{n_half}, {t_frames}] = {expected}",
                x_phase.len(),
            )));
        }

        let mut re = vec![0.0f32; t_frames * n_half];
        let mut im = vec![0.0f32; t_frames * n_half];
        for frame in 0..t_frames {
            for fc in 0..n_half {
                // Channel-major inputs (matches piper's iSTFT-head layout).
                let mag = x_mag[fc * t_frames + frame].exp();
                let phase = activation.apply(x_phase[fc * t_frames + frame]) * std::f32::consts::PI;
                // Row-major output: `Spectrogram` is `[frames, bins]`.
                re[frame * n_half + fc] = mag * phase.cos();
                im[frame * n_half + fc] = mag * phase.sin();
            }
        }

        let spec = Spectrogram {
            frames: t_frames,
            bins: n_half,
            re,
            im,
        };
        let attrs = IstftAttrs {
            n_fft: self.istft_n_fft,
            hop_length: self.istft_hop,
            win_length: self.istft_win_length,
            window: Window::Hann,
            window_symmetry: WindowSymmetry::Periodic,
            center: false,
            normalization: Normalization::Backward,
            real_input: true,
            length: Some(t_frames * self.istft_hop),
        };
        istft(&spec, &attrs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic weights for a [`ResBlock`] with `channels` channels and the
    /// given `kernel` / `dilations`. All conv weights and biases are zero, so
    /// `x += conv(leaky_relu(x))` reduces to the identity — the ResBlock
    /// output equals its input on every call regardless of the input pattern.
    /// That makes the shape-preservation assertion deterministic without
    /// inventing hparam numbers (FR-EX-08 / M2-07 §"Design decisions",
    /// "shape-driven, never invent constants").
    fn zero_weight_resblock(channels: usize, kernel: usize, dilations: [usize; 2]) -> ResBlock {
        let per_conv = channels * channels * kernel;
        ResBlock {
            convs: [
                (vec![0.0f32; per_conv], vec![0.0f32; channels]),
                (vec![0.0f32; per_conv], vec![0.0f32; channels]),
            ],
            kernel,
            dilations,
            channels,
        }
    }

    /// A [`ResBlock`] with zero conv weights preserves both shape and, thanks
    /// to the additive-residual identity, the input values themselves. Every
    /// MRF-branch kernel / dilation pair in [`RESBLOCK_KERNELS`] /
    /// [`RESBLOCK_DILATIONS`] is exercised so a same-padding math change (e.g.
    /// a dilation-aware pad regression) is caught.
    #[test]
    fn resblock_forward_shape() {
        let compute = Compute::cpu();
        let channels = 4;
        let t = 8;
        // Distinct values per (channel, time) so any mis-index is visible.
        let x: Vec<f32> = (0..channels * t).map(|i| i as f32 * 0.5 - 3.0).collect();

        for (&k, &dil) in RESBLOCK_KERNELS.iter().zip(RESBLOCK_DILATIONS.iter()) {
            let rb = zero_weight_resblock(channels, k, dil);
            let y = rb.forward(&compute, x.clone(), t);
            assert_eq!(
                y.len(),
                channels * t,
                "kernel {k} dil {dil:?}: length mismatch"
            );
            // Zero weights + additive residual → identity.
            for (i, (&yi, &xi)) in y.iter().zip(&x).enumerate() {
                assert!(
                    (yi - xi).abs() < 1e-6,
                    "kernel {k} dil {dil:?} index {i}: {yi} vs {xi}"
                );
            }
        }
    }

    /// AdaIN, applied inside the decoder module the same way [`Upsample`]
    /// applies it after `conv_transpose1d`, matches a scalar Python-style
    /// oracle on a 4-sample-per-channel input. Duplicates the [`super::nn`]
    /// unit test on purpose: the T16 composition claim is that AdaIN under
    /// the decoder's call-site produces the same math as
    /// `nn.InstanceNorm1d + affine`, so the oracle lives with the caller too.
    #[test]
    fn adain_composition_matches_scalar_oracle() {
        // 2 channels × 4 time steps, distinct values per (channel, time) and
        // very different per-channel means so a shared-scratch bug bleeds
        // visibly across channels.
        let mut x = vec![
            1.0, 2.0, 3.0, 4.0, // ch0
            100.0, 200.0, 300.0, 400.0, // ch1
        ];
        let gamma = [2.0, 0.5];
        let beta = [10.0, -1.0];
        let channels = 2usize;
        let time = 4usize;

        // Scalar oracle: for each channel, `(x - mean) / sqrt(var + EPS) *
        // gamma + beta` with a biased variance (`nn.InstanceNorm1d` default,
        // aligned with [`super::nn::EPS`]).
        let mut want = x.clone();
        for c in 0..channels {
            let row = &mut want[c * time..c * time + time];
            let mean = row.iter().sum::<f32>() / time as f32;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / time as f32;
            let inv = 1.0 / (var + nn::EPS).sqrt();
            for v in row.iter_mut() {
                *v = (*v - mean) * inv * gamma[c] + beta[c];
            }
        }

        nn::adain(&mut x, &gamma, &beta, channels, time);
        for (i, (&g, &w)) in x.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-5, "index {i}: {g} vs {w}");
        }
    }

    /// The style → gamma/beta linear projection: `y = W·x + b`, `W`
    /// `[out, in]` row-major.
    #[test]
    fn linear_projects_style_to_out_channels() {
        // W = [[1, 2, 3], [4, 5, 6]], b = [10, 20], x = [1, 1, 1]
        // → y = [1+2+3+10, 4+5+6+20] = [16, 35].
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [10.0, 20.0];
        let x = [1.0, 1.0, 1.0];
        let y = linear(&w, &b, &x);
        assert_eq!(y, vec![16.0, 35.0]);
    }

    // --- T17 iSTFT head tests -------------------------------------------------

    /// Synthetic [`Decoder`] for the head-only tests. Toy sizes only — never
    /// real Kokoro hyper-parameters (M2-07 §2 "synthetic weights" constraint) —
    /// but they satisfy the iSTFT op's preconditions (`win_length <= n_fft`,
    /// `hop < n_fft`) so the head runs end-to-end.
    fn decoder_for_istft_head_test() -> Decoder {
        Decoder {
            hidden_dim: 128,
            istft_n_fft: 16,
            istft_hop: 4,
            istft_win_length: 16,
        }
    }

    /// `IstftAttrs::length = Some(t_frames · hop)` pins the output length
    /// exactly — no `n_fft/2` head/tail padding leaks through.
    #[test]
    fn istft_head_shape_matches_expected_length() {
        let dec = decoder_for_istft_head_test();
        let t_frames = 8;
        let n_half = dec.istft_n_fft / 2 + 1;
        let x_mag = vec![0.0f32; n_half * t_frames];
        let x_phase = vec![0.0f32; n_half * t_frames];
        let pcm = dec
            .istft_head(&x_mag, &x_phase, t_frames, PhaseActivation::Tanh)
            .expect("istft head runs on zero logits");
        assert_eq!(pcm.len(), t_frames * dec.istft_hop);
    }

    /// Under every metadata-dispatched activation (M2-07 plan §5 R2: any one
    /// can be the upstream truth), the head produces a finite PCM buffer of
    /// the expected length. `exp` and each activation are exercised on a mix
    /// of positive / negative logits so a NaN-producing branch (e.g. an
    /// exp-overflow path) is caught.
    #[test]
    fn istft_head_finite_output() {
        let dec = decoder_for_istft_head_test();
        let t_frames = 6;
        let n_half = dec.istft_n_fft / 2 + 1;
        let x_mag: Vec<f32> = (0..n_half * t_frames)
            .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
            .collect();
        let x_phase: Vec<f32> = (0..n_half * t_frames)
            .map(|i| ((i % 5) as f32) * 0.2 - 0.5)
            .collect();
        for act in [
            PhaseActivation::Tanh,
            PhaseActivation::Sin,
            PhaseActivation::Identity,
        ] {
            let pcm = dec
                .istft_head(&x_mag, &x_phase, t_frames, act)
                .expect("istft head runs on nonzero logits");
            assert_eq!(pcm.len(), t_frames * dec.istft_hop);
            for (i, &v) in pcm.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "istft head produced non-finite sample [{i}] under {act:?}: {v}"
                );
            }
        }
    }

    /// Off-by-one on the magnitude tensor must fail loudly (FR-EX-08), not
    /// silently truncate / zero-fill.
    #[test]
    fn istft_head_rejects_wrong_mag_shape() {
        let dec = decoder_for_istft_head_test();
        let t_frames = 4;
        let n_half = dec.istft_n_fft / 2 + 1;
        let x_mag = vec![0.0f32; n_half * t_frames + 1];
        let x_phase = vec![0.0f32; n_half * t_frames];
        assert!(matches!(
            dec.istft_head(&x_mag, &x_phase, t_frames, PhaseActivation::Tanh),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// Off-by-one on the phase tensor must fail loudly too.
    #[test]
    fn istft_head_rejects_wrong_phase_shape() {
        let dec = decoder_for_istft_head_test();
        let t_frames = 4;
        let n_half = dec.istft_n_fft / 2 + 1;
        let x_mag = vec![0.0f32; n_half * t_frames];
        let x_phase = vec![0.0f32; n_half * t_frames - 1];
        assert!(matches!(
            dec.istft_head(&x_mag, &x_phase, t_frames, PhaseActivation::Sin),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// [`PhaseActivation::from_meta`] recognises all three documented
    /// dispatches; that is the promise the metadata schema makes.
    #[test]
    fn phase_activation_from_meta_dispatches_all_three() {
        assert_eq!(
            PhaseActivation::from_meta("tanh").unwrap(),
            PhaseActivation::Tanh
        );
        assert_eq!(
            PhaseActivation::from_meta("sin").unwrap(),
            PhaseActivation::Sin
        );
        assert_eq!(
            PhaseActivation::from_meta("identity").unwrap(),
            PhaseActivation::Identity
        );
    }

    /// An unknown activation must fail loudly with an error that names both
    /// the offending metadata key and the rejected value — FR-EX-08 (no
    /// silent default).
    #[test]
    fn phase_activation_from_meta_rejects_unknown() {
        match PhaseActivation::from_meta("relu") {
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(
                    msg.contains(KEY_PHASE_ACTIVATION),
                    "error should name the offending metadata key; got: {msg}"
                );
                assert!(
                    msg.contains("relu"),
                    "error should surface the rejected value; got: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }
}
