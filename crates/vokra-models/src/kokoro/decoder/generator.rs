//! Kokoro-82M iSTFTNet generator (M2-07-T15 — companion to `decoder.rs`).
//!
//! The generator is the HiFi-GAN-style back-end of the Kokoro decoder: it
//! consumes the last decode block's output (`[decode_final_out, t]`
//! channel-major), a style vector, and the F0 curve, and produces the
//! `(x_mag, x_phase)` pair the iSTFT head lowers to PCM.
//!
//! # Architecture (upstream)
//!
//! ```text
//! Inputs
//!   x          [decode_final_out (=512), t]
//!   style      [style_dim (=128)]
//!   f0_curve   [t]                             (per-frame F0)
//!
//! Source path (NSF-style harmonic generator)
//!   f0 upsampled to full audio rate → SineGen (9 harmonics) →
//!   Linear[9 → 1] → tanh → source_output[T_full]
//!   STFT(source_output, n_fft, hop) → mag+phase concat = source_spec[2·n_half, T_stft]
//!
//! Main path (main resolution pyramid)
//!   x0 = x                                     [decode_final_out, t]
//!   ---- stage 0 ----
//!   x0 = LeakyReLU(x0)
//!   x0 = ups.0(x0)                             [gen_mid,   t · stride_0]
//!   src0 = noise_convs.0(source_spec)          [gen_mid,   t · stride_0]
//!   res0 = noise_res.0(src0, s)                [gen_mid,   t · stride_0]
//!   x0 = x0 + res0
//!   x0 = MRF(resblocks[0..3], x0, s)           # avg of 3 branches
//!   ---- stage 1 ----
//!   x0 = LeakyReLU(x0)
//!   x0 = ups.1(x0)                             [gen_final, t · stride_0 · stride_1]
//!   src1 = noise_convs.1(source_spec)          [gen_final, t · stride_0 · stride_1]
//!   res1 = noise_res.1(src1, s)                [gen_final, t · stride_0 · stride_1]
//!   x0 = x0 + res1
//!   x0 = MRF(resblocks[3..6], x0, s)
//!   ---- head ----
//!   x0 = LeakyReLU(x0)
//!   xh = conv_post(x0)                         [2·n_half, T_gen]
//!   x_mag   = xh[0..n_half]
//!   x_phase = xh[n_half..2·n_half]
//!   → iSTFT (decoder.rs) → PCM
//! ```
//!
//! `AmpResBlock` is the HiFi-GAN AMP ResBlock (BigVGAN's per-block Snake
//! activation): 3 dilated Conv1d pairs with dilations [1, 3, 5], each with a
//! preceding AdaIN(x, s) + Snake(alpha) block.
//!
//! # M2-07-T15 pragmatic simplifications
//!
//! * F0-driven harmonic source: for the M2-07-T15 landing the source path is
//!   simplified to a **zero-fill** of the `[2·n_half, T_stft]` spec buffer.
//!   The `noise_convs` weights are still loaded strictly (FR-EX-08) so the
//!   loader catches an upstream rename; forward feeds zeros through them,
//!   producing a well-defined but silent noise/source contribution. T17
//!   parity work wires the SineGen + STFT properly.
//! * Time axes match the ups strides exactly (no output_padding trickery):
//!   `t_after_ups = (t_in - 1) · stride + kernel - 2 · pad`. This mirrors
//!   the piper-plus decoder's ConvTranspose1d contract; padding values are
//!   derived from `(kernel - stride) / 2` at load.

use vokra_core::{Result, VokraError};

// `generator` lives at `kokoro/decoder/generator.rs`; `super` is the decoder
// module (its public helpers are `pub(super)` from decoder.rs), and
// `super::super` is the kokoro module (nn, weights, config).
use super::super::config::KokoroConfig;
use super::super::nn;
use super::super::weights::TensorStore;
use super::{AdaIN1d, WeightNormedConv1d, WeightNormedConvTranspose1d};
use crate::compute::Compute;

/// LeakyReLU slope used throughout the generator — HiFi-GAN default (0.1).
const GEN_LRELU_SLOPE: f32 = 0.1;

/// Snake activation `x + (1/(α+eps)) · sin²(αx)` used inside every generator
/// resblock (BigVGAN AMP style; confirmed by the per-block
/// `alpha1.j / alpha2.j` tensors in the manifest).
fn snake(x: &mut [f32], alpha: &[f32], channels: usize, time: usize) {
    nn::snake_activation(x, alpha, channels, time);
}

/// One HiFi-GAN AMP ResBlock (BigVGAN style): three dilated Conv1d pairs
/// [(1, 3, 5)] with per-sub-block AdaIN + Snake activation.
///
/// Forward on `[channels, t]` channel-major (returns same shape):
///
/// ```text
/// for j in 0..3:
///     xj = adain1[j](x, s)
///     xj = snake(xj, alpha1[j])
///     xj = convs1[j](xj)       # dilated by [1, 3, 5][j]
///     xj = adain2[j](xj, s)
///     xj = snake(xj, alpha2[j])
///     xj = convs2[j](xj)
///     x  = x + xj
/// ```
///
/// Weights are bound at load-time (FR-EX-08); tensor names follow the
/// manifest verbatim (`{prefix}.convs1.{j}.weight_g` etc).
pub(super) struct AmpResBlock {
    channels: usize,
    /// `[j]` = (convs1[j], convs2[j], adain1[j], adain2[j], alpha1[j], alpha2[j])
    subs: [AmpResBlockSub; 3],
}

struct AmpResBlockSub {
    convs1: WeightNormedConv1d,
    convs2: WeightNormedConv1d,
    adain1: AdaIN1d,
    adain2: AdaIN1d,
    alpha1: Vec<f32>,
    alpha2: Vec<f32>,
}

impl AmpResBlock {
    /// Loads a single [`AmpResBlock`] under `prefix` (e.g.
    /// `"decoder.module.generator.resblocks.0"`) with the given `channels`
    /// (per-block width) and `kernel` (3 / 7 / 11 for the three MRF branches).
    ///
    /// Per-sub-block tensor names (per the manifest at
    /// `data/upstream_tensors_v1_0.tsv`):
    ///
    /// * `{prefix}.convs1.{j}.weight_g / weight_v / bias`  (dilation 1,3,5)
    /// * `{prefix}.convs2.{j}.weight_g / weight_v / bias`  (dilation 1)
    /// * `{prefix}.adain1.{j}.fc.weight / .bias`           AdaIN over `channels`
    /// * `{prefix}.adain2.{j}.fc.weight / .bias`           AdaIN over `channels`
    /// * `{prefix}.alpha1.{j}`                             `[1, channels, 1]`
    /// * `{prefix}.alpha2.{j}`                             `[1, channels, 1]`
    pub(super) fn load(
        store: &TensorStore,
        prefix: &str,
        channels: usize,
        kernel: usize,
        style_dim: usize,
    ) -> Result<Self> {
        // BigVGAN AMP ResBlock uses dilations (1, 3, 5) on convs1; convs2 uses
        // dilation 1 (matching HiFi-GAN's ResBlock1). Padding is `same` so
        // `pad = (kernel - 1) · dilation / 2`.
        const DILATIONS: [usize; 3] = [1, 3, 5];
        let mut subs: [Option<AmpResBlockSub>; 3] = [None, None, None];
        for j in 0..3 {
            let d = DILATIONS[j];
            let pad1 = (kernel - 1) * d / 2;
            let pad2 = (kernel - 1) / 2;
            let convs1 = WeightNormedConv1d::load(
                store,
                &format!("{prefix}.convs1.{j}"),
                channels,
                channels,
                kernel,
                /* stride */ 1,
                pad1,
                d,
                /* groups */ 1,
                /* has_bias */ true,
            )?;
            let convs2 = WeightNormedConv1d::load(
                store,
                &format!("{prefix}.convs2.{j}"),
                channels,
                channels,
                kernel,
                1,
                pad2,
                /* dilation */ 1,
                1,
                true,
            )?;
            let adain1 =
                AdaIN1d::load(store, &format!("{prefix}.adain1.{j}"), channels, style_dim)?;
            let adain2 =
                AdaIN1d::load(store, &format!("{prefix}.adain2.{j}"), channels, style_dim)?;
            let alpha1 = store.tensor_shaped(&format!("{prefix}.alpha1.{j}"), &[1, channels, 1])?;
            let alpha2 = store.tensor_shaped(&format!("{prefix}.alpha2.{j}"), &[1, channels, 1])?;
            subs[j] = Some(AmpResBlockSub {
                convs1,
                convs2,
                adain1,
                adain2,
                alpha1,
                alpha2,
            });
        }
        // Move Options into a plain array (each is Some by construction).
        let [Some(s0), Some(s1), Some(s2)] = subs else {
            unreachable!("AmpResBlock::load: all subs populated");
        };
        Ok(Self {
            channels,
            subs: [s0, s1, s2],
        })
    }

    /// In-place forward pass on `[channels, t]` channel-major `x`.
    pub(super) fn forward(&self, compute: &Compute, x: &mut [f32], t: usize, style: &[f32]) {
        self.forward_with_dump(compute, x, t, style, None);
    }

    /// Same as [`forward`] with per-sub-block dumps when `dump_prefix` is set
    /// (T17-fixup #1 bisection). Emits `<prefix>_sub_<j>_{adain1,snake1,c1,adain2,snake2,c2,acc}`.
    pub(super) fn forward_with_dump(
        &self,
        compute: &Compute,
        x: &mut [f32],
        t: usize,
        style: &[f32],
        dump_prefix: Option<&str>,
    ) {
        for (j, sub) in self.subs.iter().enumerate() {
            let mut xj = x.to_vec();
            // adain1 + snake + convs1
            sub.adain1.apply(&mut xj, t, style);
            if let Some(p) = dump_prefix {
                super::maybe_dump_stage(&format!("{p}_sub_{j}_adain1"), &xj);
            }
            snake(&mut xj, &sub.alpha1, self.channels, t);
            if let Some(p) = dump_prefix {
                super::maybe_dump_stage(&format!("{p}_sub_{j}_snake1"), &xj);
            }
            let (xj, t_after_c1) = sub.convs1.forward(compute, &xj, t);
            debug_assert_eq!(t_after_c1, t, "AMP convs1 must be same-padding");
            if let Some(p) = dump_prefix {
                super::maybe_dump_stage(&format!("{p}_sub_{j}_c1"), &xj);
            }
            // adain2 + snake + convs2
            let mut xj = xj;
            sub.adain2.apply(&mut xj, t, style);
            if let Some(p) = dump_prefix {
                super::maybe_dump_stage(&format!("{p}_sub_{j}_adain2"), &xj);
            }
            snake(&mut xj, &sub.alpha2, self.channels, t);
            if let Some(p) = dump_prefix {
                super::maybe_dump_stage(&format!("{p}_sub_{j}_snake2"), &xj);
            }
            let (xj, t_after_c2) = sub.convs2.forward(compute, &xj, t);
            debug_assert_eq!(t_after_c2, t, "AMP convs2 must be same-padding");
            if let Some(p) = dump_prefix {
                super::maybe_dump_stage(&format!("{p}_sub_{j}_c2"), &xj);
            }
            // Additive residual
            for (a, b) in x.iter_mut().zip(&xj) {
                *a += b;
            }
            if let Some(p) = dump_prefix {
                super::maybe_dump_stage(&format!("{p}_sub_{j}_acc"), x);
            }
        }
    }
}

/// Plain (non-weight-normed) 1-D convolution — used only by
/// `generator.noise_convs`, which the upstream stores as `nn.Conv1d(...)`
/// without a `weight_norm` wrapper.
struct PlainConv1d {
    weight: Vec<f32>, // `[out_ch, in_ch, kernel]` row-major
    bias: Vec<f32>,   // `[out_ch]`
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
}

impl PlainConv1d {
    fn load(
        store: &TensorStore,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        pad: usize,
    ) -> Result<Self> {
        let weight = store.tensor_shaped(&format!("{prefix}.weight"), &[out_ch, in_ch, kernel])?;
        let bias = store.tensor_shaped(&format!("{prefix}.bias"), &[out_ch])?;
        Ok(Self {
            weight,
            bias,
            in_ch,
            out_ch,
            kernel,
            stride,
            pad,
        })
    }

    fn forward(&self, compute: &Compute, x: &[f32], in_len: usize) -> (Vec<f32>, usize) {
        nn::conv1d(
            compute,
            x,
            self.in_ch,
            in_len,
            &self.weight,
            self.out_ch,
            self.kernel,
            Some(&self.bias),
            self.stride,
            self.pad,
            /* dilation */ 1,
            /* groups */ 1,
        )
    }
}

/// Kokoro iSTFTNet generator sub-module.
pub(super) struct Generator {
    /// Number of upsampling stages (=2 for real Kokoro).
    n_stages: usize,
    /// Input channel count (== `decode_final_out` = 512 for real).
    #[allow(dead_code)] // exposed for future generator-only tests / T17 parity
    in_ch: usize,
    /// Generator resblock channel widths per stage (real: [256, 128]).
    stage_channels: Vec<usize>,
    /// Number of MRF branches per stage (=3, the three kernels [3, 7, 11]).
    #[allow(dead_code)] // exposed for future generator-only tests / T17 parity
    n_kernels: usize,
    /// The n_fft used by the iSTFT head — read from config, echoed here so
    /// the noise convs know the source-spec channel count.
    #[allow(dead_code)] // reserved for T17 SineGen wiring
    n_half: usize,
    /// `2·n_half` — the mag+phase channel count of the source spec / conv_post
    /// output.
    source_ch: usize,

    // ---- Sub-modules -------------------------------------------------------
    /// Upsampling ConvTranspose1d layers, one per stage.
    ups: Vec<WeightNormedConvTranspose1d>,
    /// Noise-source convs (one per stage), plain Conv1d without weight_norm.
    noise_convs: Vec<PlainConv1d>,
    /// Noise-source resblocks (one per stage, AMP block).
    noise_res: Vec<AmpResBlock>,
    /// Main-path resblocks: `n_stages · n_kernels` blocks, indexed as
    /// `stage · n_kernels + kernel_idx` (row-major).
    resblocks: Vec<AmpResBlock>,
    /// Final projection to `(mag, phase)`: WeightNormedConv1d(gen_final → 2·n_half, k=7).
    conv_post: WeightNormedConv1d,
    /// `SourceModuleHnNSF` harmonic mixer: `Linear(harm+1 → 1)`. Loaded strictly
    /// (FR-EX-08) but the forward path does not yet feed real F0 into it (see
    /// module doc).
    #[allow(dead_code)] // consumed by the T17 SineGen wiring
    m_source_weight: Vec<f32>,
    #[allow(dead_code)] // consumed by the T17 SineGen wiring
    m_source_bias: Vec<f32>,
}

impl Generator {
    /// Loads the generator sub-module from `store`. `in_ch` is the input
    /// channel count (== `decode_final_out` in `DecoderReal`), `style_dim` is
    /// the AdaIN style width; both are derived from the tensor shapes at the
    /// caller.
    pub(super) fn load(store: &TensorStore, in_ch: usize, style_dim: usize) -> Result<Self> {
        let g = "decoder.module.generator";

        // ---- m_source (Linear harmonic mixer) ---------------------------
        // Shape: weight [1, harm+1], bias [1].
        let m_shape = store.shape(&format!("{g}.m_source.l_linear.weight"))?;
        if m_shape.len() != 2 || m_shape[0] != 1 {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro generator: m_source.l_linear.weight shape {m_shape:?} \
                 must be [1, harm+1]",
            )));
        }
        let m_source_weight =
            store.tensor_shaped(&format!("{g}.m_source.l_linear.weight"), &m_shape)?;
        let m_source_bias = store.tensor_shaped(&format!("{g}.m_source.l_linear.bias"), &[1])?;

        // ---- conv_post (gen_final → 2·n_half, k=7) ----------------------
        // Shape: weight_v [2·n_half, gen_final, 7].
        let cp_shape = store.shape(&format!("{g}.conv_post.weight_v"))?;
        if cp_shape.len() != 3 {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro generator: conv_post.weight_v shape {cp_shape:?} \
                 must be [2·n_half, gen_final, kernel]",
            )));
        }
        let source_ch = cp_shape[0]; // = 2·n_half
        if source_ch % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro generator: conv_post out_ch ({source_ch}) must be even (mag+phase)",
            )));
        }
        let n_half = source_ch / 2;
        let gen_final = cp_shape[1];
        let conv_post_kernel = cp_shape[2];
        let conv_post_pad = (conv_post_kernel - 1) / 2;
        let conv_post = WeightNormedConv1d::load(
            store,
            &format!("{g}.conv_post"),
            gen_final,
            source_ch,
            conv_post_kernel,
            /* stride */ 1,
            conv_post_pad,
            /* dilation */ 1,
            /* groups */ 1,
            /* has_bias */ true,
        )?;

        // ---- ups.i (ConvTranspose1d) ------------------------------------
        // Determine the number of stages by probing ups.0, ups.1, ... until
        // shape returns Err. For real Kokoro that's 2 (ups.0 and ups.1).
        let mut ups = Vec::new();
        let mut stage_channels = Vec::new();
        let mut cur_in = in_ch;
        for i in 0..8 {
            // Bounded probe (safety cap — real Kokoro has 2 ups stages).
            let name = format!("{g}.ups.{i}.weight_v");
            let Ok(shape) = store.shape(&name) else {
                break;
            };
            if shape.len() != 3 {
                return Err(VokraError::InvalidArgument(format!(
                    "kokoro generator: {name} shape {shape:?} must be [in_ch, out_ch, kernel]",
                )));
            }
            if shape[0] != cur_in {
                return Err(VokraError::InvalidArgument(format!(
                    "kokoro generator: ups.{i}.weight_v in_ch {} != previous stage out_ch ({cur_in})",
                    shape[0],
                )));
            }
            let out_ch = shape[1];
            let kernel = shape[2];
            // Stride is `kernel / 2` in the reference (upsample_kernel = 2·stride).
            // We infer stride from `kernel / 2` since the tensor shape carries only kernel.
            let stride = kernel / 2;
            if stride == 0 || 2 * stride != kernel {
                return Err(VokraError::InvalidArgument(format!(
                    "kokoro generator: ups.{i} kernel ({kernel}) must be 2·stride \
                     (StyleTTS 2 iSTFTNet convention)",
                )));
            }
            let pad = (kernel - stride) / 2;
            let up = WeightNormedConvTranspose1d::load(
                store,
                &format!("{g}.ups.{i}"),
                cur_in,
                out_ch,
                kernel,
                stride,
                pad,
                /* groups */ 1,
                /* has_bias */ true,
            )?;
            ups.push(up);
            stage_channels.push(out_ch);
            cur_in = out_ch;
        }
        if ups.is_empty() {
            return Err(VokraError::InvalidArgument(
                "kokoro generator: no ups.i tensors found (expected at least ups.0)".to_owned(),
            ));
        }
        let n_stages = ups.len();
        if stage_channels[n_stages - 1] != gen_final {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro generator: last ups out_ch ({}) != conv_post in_ch ({gen_final})",
                stage_channels[n_stages - 1],
            )));
        }

        // ---- noise_convs.i (plain Conv1d, 2·n_half → stage_channels[i]) --
        // The upstream sets kernel for stage i to `stride_downstream · 2` where
        // `stride_downstream = ∏ strides[j] for j > i`. For the last stage,
        // kernel = 1.
        //
        // We infer kernel from the tensor shape directly (no reconstruction needed).
        let mut noise_convs = Vec::with_capacity(n_stages);
        for (i, &stage_ch) in stage_channels.iter().enumerate().take(n_stages) {
            let name = format!("{g}.noise_convs.{i}.weight");
            let shape = store.shape(&name)?;
            if shape.len() != 3 {
                return Err(VokraError::InvalidArgument(format!(
                    "kokoro generator: {name} shape {shape:?} must be [out_ch, in_ch, kernel]",
                )));
            }
            if shape[0] != stage_ch {
                return Err(VokraError::InvalidArgument(format!(
                    "kokoro generator: noise_convs.{i} out_ch {} != stage_channels[{i}] ({stage_ch})",
                    shape[0],
                )));
            }
            if shape[1] != source_ch {
                return Err(VokraError::InvalidArgument(format!(
                    "kokoro generator: noise_convs.{i} in_ch {} != source_ch ({source_ch})",
                    shape[1],
                )));
            }
            let kernel = shape[2];
            let stride = if kernel > 1 { kernel.div_ceil(2) } else { 1 };
            let pad = if kernel > 1 { stride.div_ceil(2) } else { 0 };
            let nc = PlainConv1d::load(
                store,
                &format!("{g}.noise_convs.{i}"),
                source_ch,
                stage_ch,
                kernel,
                stride,
                pad,
            )?;
            noise_convs.push(nc);
        }

        // ---- noise_res.i (AmpResBlock at stage_channels[i]) -------------
        //
        // Kernel: the upstream noise_res follows the last MRF resblock's kernel
        // for the same stage. To be robust, we derive the kernel from the
        // loaded `convs1.0.weight_v` shape.
        let mut noise_res = Vec::with_capacity(n_stages);
        for (i, &stage_ch) in stage_channels.iter().enumerate().take(n_stages) {
            let probe = format!("{g}.noise_res.{i}.convs1.0.weight_v");
            let shape = store.shape(&probe)?;
            if shape.len() != 3 {
                return Err(VokraError::InvalidArgument(format!(
                    "kokoro generator: {probe} shape {shape:?} must be [ch, ch, kernel]",
                )));
            }
            let kernel = shape[2];
            let ares = AmpResBlock::load(
                store,
                &format!("{g}.noise_res.{i}"),
                stage_ch,
                kernel,
                style_dim,
            )?;
            noise_res.push(ares);
        }

        // ---- resblocks (MRF, n_stages · n_kernels blocks) ---------------
        // Determine n_kernels by probing resblocks.0, .1, .2, ... Each stage
        // has the same number of kernels; total blocks = n_stages · n_kernels.
        // For robustness, we derive per-block kernel from tensor shapes.
        let mut resblocks = Vec::new();
        for k in 0..32 {
            // Bounded probe (real Kokoro has 6 resblocks = 2 stages · 3 kernels).
            let probe = format!("{g}.resblocks.{k}.convs1.0.weight_v");
            let Ok(shape) = store.shape(&probe) else {
                break;
            };
            if shape.len() != 3 {
                return Err(VokraError::InvalidArgument(format!(
                    "kokoro generator: {probe} shape {shape:?} must be [ch, ch, kernel]",
                )));
            }
            let ch = shape[0];
            let kernel = shape[2];
            // Assign this resblock to its stage: sequentially, filling per stage.
            let ares =
                AmpResBlock::load(store, &format!("{g}.resblocks.{k}"), ch, kernel, style_dim)?;
            resblocks.push(ares);
        }
        if resblocks.is_empty() {
            return Err(VokraError::InvalidArgument(
                "kokoro generator: no resblocks.i tensors found".to_owned(),
            ));
        }
        let total = resblocks.len();
        if total % n_stages != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro generator: {total} resblocks not divisible by n_stages ({n_stages})",
            )));
        }
        let n_kernels = total / n_stages;

        Ok(Self {
            n_stages,
            in_ch,
            stage_channels,
            n_kernels,
            n_half,
            source_ch,
            ups,
            noise_convs,
            noise_res,
            resblocks,
            conv_post,
            m_source_weight,
            m_source_bias,
        })
    }

    /// Forward pass. `x` is `[in_ch, t]` channel-major (the last decode
    /// block's output); `style` is `[style_dim]`; `f0_curve` is `[t]`
    /// (per-frame F0 — currently used only to derive the source-spec time
    /// axis; see module-doc for the M2-07-T15 simplification).
    ///
    /// Returns `(x_mag, x_phase, t_gen)` where each spectrum tensor is
    /// `[n_half, t_gen]` channel-major.
    pub(super) fn forward(
        &self,
        compute: &Compute,
        x: &[f32],
        t_in: usize,
        style: &[f32],
        _f0_curve: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, usize)> {
        // Track current x / t / current channel count across stages.
        let mut cur = x.to_vec();
        let mut cur_len = t_in;
        let mut cur_ch = self.in_ch;

        for stage in 0..self.n_stages {
            // 1. LeakyReLU → ups
            nn::leaky_relu(&mut cur, GEN_LRELU_SLOPE);
            super::maybe_dump_stage(&format!("gen_stage_{stage}_pre_ups"), &cur);
            let (up, t_up) = self.ups[stage].forward(&cur, cur_len);
            let stage_ch = self.stage_channels[stage];
            debug_assert_eq!(up.len(), stage_ch * t_up);
            super::maybe_dump_stage(&format!("gen_stage_{stage}_ups"), &up);

            // 2. Noise / source contribution.
            //    Source spec = zeros (simplification; see module-doc). Shape
            //    is `[source_ch, t_up]` so the noise_convs kernel + stride
            //    line up regardless of the upstream STFT frame count.
            let source_spec = vec![0.0f32; self.source_ch * t_up];
            let (noise_x, t_noise) = self.noise_convs[stage].forward(compute, &source_spec, t_up);
            // The plain Conv1d + zero-fill produces a bias-only signal; that's
            // fine for a well-defined forward. Nearest-neighbor pad/crop to
            // t_up so the residual add operates on matching time axes.
            let noise_aligned = if t_noise == t_up {
                noise_x
            } else {
                nearest_align(&noise_x, stage_ch, t_noise, t_up)
            };
            super::maybe_dump_stage(&format!("gen_stage_{stage}_noise_pre_res"), &noise_aligned);
            // Noise resblock (in-place additive residual over 3 sub-blocks).
            let mut noise_aligned_mut = noise_aligned;
            // Dump per-sub-block intermediates for the noise_res AmpResBlock
            // when VOKRA_KOKORO_PARITY_DUMP is set (T17-fixup #1 bisection).
            let dump_prefix = format!("gen_stage_{stage}_noise_res");
            self.noise_res[stage].forward_with_dump(
                compute,
                &mut noise_aligned_mut,
                t_up,
                style,
                Some(&dump_prefix),
            );
            super::maybe_dump_stage(
                &format!("gen_stage_{stage}_noise_post_res"),
                &noise_aligned_mut,
            );

            // 3. Main + noise fusion.
            let mut fused = up;
            for (a, b) in fused.iter_mut().zip(&noise_aligned_mut) {
                *a += b;
            }
            super::maybe_dump_stage(&format!("gen_stage_{stage}_fused"), &fused);

            // 4. MRF: average over `n_kernels` main resblocks for this stage.
            let mut mrf = vec![0.0f32; stage_ch * t_up];
            for k in 0..self.n_kernels {
                let rb_idx = stage * self.n_kernels + k;
                let mut branch = fused.clone();
                self.resblocks[rb_idx].forward(compute, &mut branch, t_up, style);
                super::maybe_dump_stage(&format!("gen_stage_{stage}_rb_{k}"), &branch);
                for (a, b) in mrf.iter_mut().zip(&branch) {
                    *a += b;
                }
            }
            let inv = 1.0 / self.n_kernels as f32;
            for v in &mut mrf {
                *v *= inv;
            }
            super::maybe_dump_stage(&format!("gen_stage_{stage}_mrf"), &mrf);

            cur = mrf;
            cur_len = t_up;
            cur_ch = stage_ch;
        }
        debug_assert_eq!(cur_ch, self.stage_channels[self.n_stages - 1]);

        // 5. Head: LeakyReLU → conv_post → split (mag, phase)
        nn::leaky_relu(&mut cur, GEN_LRELU_SLOPE);
        super::maybe_dump_stage("gen_pre_conv_post", &cur);
        let (post, t_gen) = self.conv_post.forward(compute, &cur, cur_len);
        debug_assert_eq!(post.len(), self.source_ch * t_gen);
        super::maybe_dump_stage("gen_conv_post", &post);
        let n_half = self.source_ch / 2;
        let x_mag = post[..n_half * t_gen].to_vec();
        let x_phase = post[n_half * t_gen..].to_vec();
        Ok((x_mag, x_phase, t_gen))
    }
}

/// Nearest-neighbor align a `[channels, t_in]` buffer to `[channels, t_out]`.
fn nearest_align(x: &[f32], channels: usize, t_in: usize, t_out: usize) -> Vec<f32> {
    if t_in == t_out {
        return x.to_vec();
    }
    if t_in == 0 {
        return vec![0.0f32; channels * t_out];
    }
    let mut out = vec![0.0f32; channels * t_out];
    for c in 0..channels {
        for t in 0..t_out {
            let src = ((t as u64 * t_in as u64) / t_out as u64) as usize;
            let src = src.min(t_in - 1);
            out[c * t_out + t] = x[c * t_in + src];
        }
    }
    out
}

// KokoroConfig is imported by decoder.rs and not required here directly, but we
// re-export the alias so mod.rs sees no surprise ordering when it consumes
// generator via `decoder::forward_full`.
#[allow(dead_code)]
fn _module_link_check(_config: &KokoroConfig) {}
