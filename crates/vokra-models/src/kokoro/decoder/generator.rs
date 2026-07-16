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
//!   ---- stage 1 (last) ----
//!   x0 = LeakyReLU(x0)
//!   x0 = ups.1(x0)                             [gen_final, t · stride_0 · stride_1]
//!   x0 = ReflectionPad1d((1, 0))(x0)           [gen_final, t · ∏strides + 1]
//!   src1 = noise_convs.1(source_spec)          [gen_final, t · ∏strides + 1]
//!   res1 = noise_res.1(src1, s)                [gen_final, t · ∏strides + 1]
//!   x0 = x0 + res1
//!   x0 = MRF(resblocks[3..6], x0, s)
//!   ---- head ----
//!   x0 = LeakyReLU(x0)  # slope 0.01 (PyTorch default — istftnet.py:321)
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
//! The full source path (SineGen + `l_linear` + tanh + STFT) is implemented
//! here since the 2026-07-16 P1 fidelity fix, replacing the M2-07-T15
//! zero-fill placeholder that left the vocoder without pitch excitation
//! (a root cause of the round-trip WER 1.0 finding). The SineGen's two RNG
//! draws are deterministically substituted (phase offsets → 0, dither →
//! shared reproducible generator) — see [`Generator::forward`]
//! §Determinism.

use vokra_core::{Result, VokraError};

// `generator` lives at `kokoro/decoder/generator.rs`; `super` is the decoder
// module (its public helpers are `pub(super)` from decoder.rs), and
// `super::super` is the kokoro module (nn, weights, config).
use super::super::config::KokoroConfig;
use super::super::nn;
use super::super::weights::TensorStore;
use super::{AdaIN1d, WeightNormedConv1d, WeightNormedConvTranspose1d};
use crate::compute::Compute;

/// LeakyReLU slope used in the generator's upsample loop — upstream
/// `istftnet.py:307` `F.leaky_relu(x, negative_slope=0.1)`.
const GEN_LRELU_SLOPE: f32 = 0.1;

/// LeakyReLU slope of the generator HEAD (pre-`conv_post`) — upstream
/// `istftnet.py:321` `F.leaky_relu(x)` uses PyTorch's DEFAULT
/// `negative_slope = 0.01`, not the loop's 0.1. The pre-fix reuse of 0.1
/// here was part of the P1 decoder divergence (2026-07-16 real-weight eval).
const GEN_HEAD_LRELU_SLOPE: f32 = 0.01;

/// SineGen sine amplitude — upstream `istftnet.py:262-265`
/// `SourceModuleHnNSF(..., sine_amp=0.1)` default.
const SINE_AMP: f32 = 0.1;

/// SineGen voiced/unvoiced threshold in Hz — upstream `istftnet.py:265`
/// `voiced_threshod=10`.
const VOICED_THRESHOLD: f32 = 10.0;

/// SineGen additive-noise std in VOICED regions — upstream
/// `SineGen(..., noise_std=0.003)` default (`istftnet.py:123-128`); the
/// UNVOICED amplitude is `sine_amp / 3` (`istftnet.py:204`).
const NOISE_STD: f32 = 0.003;

/// SplitMix64 — the deterministic counter-based generator behind
/// [`deterministic_gauss`]. Public-domain constants (Steele et al.).
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic standard-normal draw for SineGen's additive dither.
///
/// Upstream `SineGen.forward` injects `noise_amp · torch.randn_like(...)`
/// (`istftnet.py:204-208`) — the dither is *designed excitation* (in
/// unvoiced regions it is the ONLY source energy, amplitude `sine_amp/3`),
/// not an optional artifact, so zeroing it would mute fricatives. But
/// torch's RNG is irreproducible from Rust, so this runtime replaces the
/// draw with a **counter-based SplitMix64 + Box–Muller** normal that the
/// parity reference dumper mirrors exactly
/// (`tools/parity/dump_kokoro_reference.py` patches `torch.randn_like` to
/// the same generator). Same N(0, 1) statistics, reproducible bits on both
/// sides. `m` is the flat row-major index into the `[t_full, n_harm]`
/// dither tensor.
///
/// f64 math throughout, narrowed once — `ln` / `cos` are within 1 ULP
/// across libms, far below the noise amplitude itself.
fn deterministic_gauss(m: u64) -> f32 {
    let x = splitmix64(2 * m);
    let y = splitmix64(2 * m + 1);
    // u1 ∈ (0, 1] (never 0 → ln is finite); u2 ∈ [0, 1).
    let u1 = ((x >> 11) as f64 + 1.0) / 9_007_199_254_740_992.0; // 2^53
    let u2 = (y >> 11) as f64 / 9_007_199_254_740_992.0;
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    z as f32
}

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
    in_ch: usize,
    /// Generator resblock channel widths per stage (real: [256, 128]).
    stage_channels: Vec<usize>,
    /// Number of MRF branches per stage (=3, the three kernels [3, 7, 11]).
    n_kernels: usize,
    /// The n_fft/2+1 bin count of the source-spec STFT / iSTFT head —
    /// derived from `conv_post.weight_v` axis 0 and cross-checked against
    /// the GGUF-carried `istft.n_fft` at forward time.
    #[allow(dead_code)] // cross-check duplicate of source_ch / 2
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
    /// `SourceModuleHnNSF` harmonic mixer: `Linear(harm+1 → 1)` — consumed by
    /// [`Generator::harmonic_source_spec`] (`istftnet.py:238,251`). The
    /// weight length pins the harmonic count (9 for Kokoro-82M).
    m_source_weight: Vec<f32>,
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
                /* output_padding */ 0,
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
    /// block's output); `style` is `[style_dim]`; `f0_curve` is `[t]` — the
    /// prosody `F0Ntrain` contour at the SAME time axis as `x` (upstream
    /// `Decoder.forward` passes `F0_curve` verbatim: `istftnet.py:420`).
    ///
    /// `sample_rate` / `istft_n_fft` / `istft_hop` / `istft_win_length`
    /// parametrize the NSF harmonic-source path: the upstream `Generator`
    /// hardcodes `sampling_rate=24000` (`istftnet.py:262-264`) and shares one
    /// `TorchSTFT(gen_istft_n_fft, gen_istft_hop_size)` between the source
    /// analysis (`self.stft.transform`) and the head inverse
    /// (`istftnet.py:293-297`); here both come from the GGUF-carried config.
    ///
    /// Returns `(x_mag, x_phase, t_gen)` where each spectrum tensor is
    /// `[n_half, t_gen]` channel-major and `t_gen = t · ∏strides + 1` (the
    /// `+1` is the last-stage `ReflectionPad1d((1, 0))` — `istftnet.py:292` +
    /// `311-312`).
    ///
    /// # Determinism (SineGen RNG)
    ///
    /// Upstream `SineGen` draws two RNG tensors per forward, irreproducible
    /// run-to-run even upstream-to-upstream (the 2026-07-16 eval measured
    /// oracle-vs-its-own-ONNX waveform `max |Δ| = 0.44` from exactly this):
    ///
    /// * `torch.rand` initial phase offsets for harmonics ≥ 2
    ///   (`istftnet.py:150-152`) — **pinned to zero** here (a per-utterance
    ///   random rotation of the upper harmonics; zero is a valid draw);
    /// * `noise_amp · torch.randn_like` additive dither
    ///   (`istftnet.py:204-208`) — **replaced by a shared deterministic
    ///   normal generator** ([`deterministic_gauss`]). The dither is
    ///   designed excitation (sole source energy in unvoiced regions), so
    ///   it is kept at the upstream amplitude, just with reproducible bits.
    ///
    /// The parity reference dumper applies the SAME two substitutions
    /// (see `tools/parity/dump_kokoro_reference.py`), so fixtures compare
    /// deterministic-to-deterministic. Documented, bounded substitution —
    /// not a silent fallback.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward(
        &self,
        compute: &Compute,
        x: &[f32],
        t_in: usize,
        style: &[f32],
        f0_curve: &[f32],
        sample_rate: u32,
        istft_n_fft: usize,
        istft_hop: usize,
        istft_win_length: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, usize)> {
        if f0_curve.len() != t_in {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro generator: f0_curve len {} != generator input frames ({t_in}) — \
                 upstream feeds F0_curve at the decode.3-output rate (istftnet.py:420)",
                f0_curve.len(),
            )));
        }
        let n_half = self.source_ch / 2;
        if istft_n_fft / 2 + 1 != n_half {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro generator: istft n_fft ({istft_n_fft}) / 2 + 1 != conv_post n_half ({n_half})"
            )));
        }

        // ---- NSF harmonic source (upstream istftnet.py:299-305) ------------
        // f0_upsamp (nearest, scale = ∏ups_strides · hop) → SineGen →
        // l_linear + tanh → STFT → har = [mag ; angle] [2·n_half, t_stft].
        let upsample_scale: usize =
            self.ups.iter().map(|u| u.stride).product::<usize>() * istft_hop;
        let har = self.harmonic_source_spec(
            f0_curve,
            t_in,
            upsample_scale,
            sample_rate,
            istft_n_fft,
            istft_hop,
            istft_win_length,
        )?;
        let t_stft = har.len() / self.source_ch;
        super::maybe_dump_stage("gen_har_spec", &har);

        // Track current x / t / current channel count across stages.
        let mut cur = x.to_vec();
        let mut cur_len = t_in;
        let mut cur_ch = self.in_ch;

        for stage in 0..self.n_stages {
            // 1. LeakyReLU(0.1) → source conv → source resblock → ups.
            nn::leaky_relu(&mut cur, GEN_LRELU_SLOPE);
            super::maybe_dump_stage(&format!("gen_stage_{stage}_pre_ups"), &cur);

            // 2. Source contribution from the shared harmonic spec
            //    (x_source = noise_res[i](noise_convs[i](har), s) —
            //    istftnet.py:308-309).
            let stage_ch = self.stage_channels[stage];
            let (mut src, t_src) = self.noise_convs[stage].forward(compute, &har, t_stft);
            super::maybe_dump_stage(&format!("gen_stage_{stage}_noise_pre_res"), &src);
            let dump_prefix = format!("gen_stage_{stage}_noise_res");
            self.noise_res[stage].forward_with_dump(
                compute,
                &mut src,
                t_src,
                style,
                Some(&dump_prefix),
            );
            super::maybe_dump_stage(&format!("gen_stage_{stage}_noise_post_res"), &src);

            // 3. Main-path upsample; the LAST stage reflection-pads one frame
            //    on the left (ReflectionPad1d((1, 0)) — istftnet.py:292 +
            //    311-312) so its length lands on `∏strides · t + 1`, exactly
            //    the source STFT frame count.
            let (up, t_up) = self.ups[stage].forward(&cur, cur_len);
            debug_assert_eq!(up.len(), stage_ch * t_up);
            let (up, t_up) = if stage == self.n_stages - 1 {
                (reflection_pad_left1(&up, stage_ch, t_up), t_up + 1)
            } else {
                (up, t_up)
            };
            super::maybe_dump_stage(&format!("gen_stage_{stage}_ups"), &up);

            // Time axes must agree EXACTLY — the upstream shapes do
            // (stage i: (t·∏strides/hop-rate) vs the strided source conv),
            // and a mismatch here means the checkpoint deviates from the
            // Kokoro-82M architecture. No silent re-alignment (FR-EX-08).
            if t_src != t_up {
                return Err(VokraError::InvalidArgument(format!(
                    "kokoro generator: stage {stage} source frames ({t_src}) != \
                     main-path frames ({t_up}); the noise_convs stride/kernel do not \
                     match the ups stride product for this checkpoint"
                )));
            }

            // 4. Main + source fusion (x = x + x_source — istftnet.py:313).
            let mut fused = up;
            for (a, b) in fused.iter_mut().zip(&src) {
                *a += b;
            }
            super::maybe_dump_stage(&format!("gen_stage_{stage}_fused"), &fused);

            // 5. MRF: average over `n_kernels` main resblocks for this stage.
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

        // 6. Head: LeakyReLU(0.01 — PyTorch default, istftnet.py:321) →
        //    conv_post → split (mag, phase).
        nn::leaky_relu(&mut cur, GEN_HEAD_LRELU_SLOPE);
        super::maybe_dump_stage("gen_pre_conv_post", &cur);
        let (post, t_gen) = self.conv_post.forward(compute, &cur, cur_len);
        debug_assert_eq!(post.len(), self.source_ch * t_gen);
        super::maybe_dump_stage("gen_conv_post", &post);
        let x_mag = post[..n_half * t_gen].to_vec();
        let x_phase = post[n_half * t_gen..].to_vec();
        Ok((x_mag, x_phase, t_gen))
    }

    /// Builds the `[2·n_half, t_stft]` channel-major harmonic-source spectrum
    /// `har = cat([|STFT(har_source)|, angle(STFT(har_source))])`
    /// (upstream `istftnet.py:299-305`).
    ///
    /// Pipeline (all deterministic — see [`Generator::forward`] §Determinism):
    ///
    /// 1. `f0_full = nn.Upsample(scale_factor=upsample_scale)(f0)` — nearest.
    /// 2. `SineGen._f02sine` (istftnet.py:142-158): per harmonic `k ∈ 1..=H`,
    ///    `rad = (f0_full·k / sample_rate) % 1` → linear-downsample by
    ///    `1/upsample_scale` → `phase = cumsum·2π` → `·upsample_scale` →
    ///    linear-upsample by `upsample_scale` → `sin`.
    /// 3. `sine_waves = sines · sine_amp · uv + noise_amp · gauss` with
    ///    `uv = (f0_full > 10)` and
    ///    `noise_amp = uv·noise_std + (1−uv)·sine_amp/3`
    ///    (istftnet.py:196-208; gauss = shared deterministic generator).
    /// 4. `har_source = tanh(l_linear(sine_waves))` (istftnet.py:251).
    /// 5. `TorchSTFT.transform` = `torch.stft(center=True, reflect)` →
    ///    magnitude + angle (istftnet.py:89-94).
    #[allow(clippy::too_many_arguments)]
    fn harmonic_source_spec(
        &self,
        f0_curve: &[f32],
        t_in: usize,
        upsample_scale: usize,
        sample_rate: u32,
        istft_n_fft: usize,
        istft_hop: usize,
        istft_win_length: usize,
    ) -> Result<Vec<f32>> {
        use vokra_core::ir::graph::{Normalization, PadMode, StftAttrs, Window, WindowSymmetry};

        let t_full = t_in * upsample_scale;
        let inv_sr = 1.0f32 / sample_rate as f32;

        // 1. Nearest upsample f0 → [t_full] (nn.Upsample default mode).
        let mut f0_full = vec![0.0f32; t_full];
        for (j, dst) in f0_full.iter_mut().enumerate() {
            *dst = f0_curve[j / upsample_scale];
        }

        // 2-3. Per-harmonic sine synthesis, mixed through l_linear + tanh on
        // the fly (avoids a [t_full × n_harm] transpose buffer): first build
        // each harmonic's sine row, gate by uv · sine_amp, and accumulate
        // `w[k] · sine` into the mix (k-ascending, matching torch's GEMV
        // reduction order at k=9); bias is added AFTER the sum — torch's
        // `F.linear` computes `x @ Wᵀ + b`, so the bias joins last.
        let mut mix = vec![0.0f32; t_full];
        let mut rad_ds = vec![0.0f32; t_in];
        let mut phase = vec![0.0f32; t_in];
        for (k, &w_k) in self.m_source_weight.iter().enumerate() {
            let harmonic = (k + 1) as f32;
            // rad[j] = (f0_full[j] · harmonic / sr) % 1 — torch `%` keeps the
            // result non-negative for a positive divisor (rem_euclid).
            // Linear-downsample by 1/upsample_scale (F.interpolate linear,
            // align_corners=False): src x = (j+0.5)·scale − 0.5, clamped ≥ 0,
            // lerped in torch's `λ0·v[x0] + λ1·v[x1]` form (the CPU
            // `compute_source_index_and_lambda` kernel). The algebraically
            // equal `v0 + w·(v1−v0)` rounds differently in f32, and the phase
            // argument below is large enough (10³–10⁴ rad) that a 1-ULP input
            // drift becomes ~1e-3 after `sin` — measured on the fixture
            // input: λ-form ≈ torch, delta-form drifts `har_source` to 8e-4.
            for (j, dst) in rad_ds.iter_mut().enumerate() {
                let x = (j as f32 + 0.5) * upsample_scale as f32 - 0.5;
                let x = if x < 0.0 { 0.0 } else { x };
                let x0 = x as usize;
                let x1 = (x0 + 1).min(t_full - 1);
                let lambda1 = x - x0 as f32;
                let lambda0 = 1.0 - lambda1;
                let r0 = (f0_full[x0] * harmonic * inv_sr).rem_euclid(1.0);
                let r1 = (f0_full[x1] * harmonic * inv_sr).rem_euclid(1.0);
                *dst = lambda0 * r0 + lambda1 * r1;
            }
            // phase = cumsum(rad_ds) · 2π (sequential f32, mirroring torch
            // CPU cumsum), then ·upsample_scale before the linear upsample —
            // three separate f32 rounding steps, matching upstream's
            // `cumsum(rad) * 2 * π` then `phase · upsample_scale` tensor ops.
            let mut acc = 0.0f32;
            for (p, &r) in phase.iter_mut().zip(rad_ds.iter()) {
                acc += r;
                *p = acc * 2.0 * std::f32::consts::PI * upsample_scale as f32;
            }
            // Linear-upsample by upsample_scale (λ-form, as above) → sin →
            // gate + dither → mix. Element-wise chain mirrors upstream
            // `istftnet.py:196-208`: `sine_waves = _f02sine(fn) · sine_amp`
            // then `· uv + noise` with
            // `noise = (uv·noise_std + (1−uv)·sine_amp/3) · randn` — the
            // randn replaced by the shared deterministic generator
            // ([`deterministic_gauss`], flat index `i·n_harm + k` over the
            // `[t_full, n_harm]` dither tensor).
            let inv_scale = 1.0f32 / upsample_scale as f32;
            let n_harm = self.m_source_weight.len() as u64;
            for (i, m) in mix.iter_mut().enumerate() {
                let x = (i as f32 + 0.5) * inv_scale - 0.5;
                let x = if x < 0.0 { 0.0 } else { x };
                let x0 = x as usize;
                let x1 = (x0 + 1).min(t_in - 1);
                let lambda1 = x - x0 as f32;
                let lambda0 = 1.0 - lambda1;
                let p = lambda0 * phase[x0] + lambda1 * phase[x1];
                let uv = if f0_full[i] > VOICED_THRESHOLD {
                    1.0
                } else {
                    0.0
                };
                let noise_amp = uv * NOISE_STD + (1.0 - uv) * (SINE_AMP / 3.0);
                let g = deterministic_gauss(i as u64 * n_harm + k as u64);
                *m += w_k * (p.sin() * SINE_AMP * uv + noise_amp * g);
            }
        }
        let bias0 = self.m_source_bias[0];
        let mut har_source = mix;
        for v in har_source.iter_mut() {
            *v = (*v + bias0).tanh();
        }
        super::maybe_dump_stage("gen_har_source", &har_source);

        // 5. STFT (torch.stft defaults: center=True, reflect padding, no
        // normalization, onesided) → [mag ; angle] channel-major.
        let attrs = StftAttrs {
            n_fft: istft_n_fft,
            hop_length: istft_hop,
            win_length: istft_win_length,
            window: Window::Hann,
            window_symmetry: WindowSymmetry::Periodic,
            center: true,
            pad_mode: PadMode::Reflect,
            normalization: Normalization::Backward,
            causal: false,
            real_input: true,
        };
        let spec = vokra_ops::stft(&har_source, &attrs)?;
        let n_half = self.source_ch / 2;
        debug_assert_eq!(spec.bins, n_half);
        let t_stft = spec.frames;
        let mut har = vec![0.0f32; self.source_ch * t_stft];
        for f in 0..t_stft {
            for b in 0..n_half {
                let re = spec.re[f * n_half + b];
                let im = spec.im[f * n_half + b];
                har[b * t_stft + f] = re.hypot(im);
                har[(n_half + b) * t_stft + f] = im.atan2(re);
            }
        }
        Ok(har)
    }
}

/// `nn.ReflectionPad1d((1, 0))` on a `[channels, t]` channel-major buffer:
/// one reflected frame on the left (`out[0] = x[1]`, `out[i] = x[i-1]`) —
/// upstream `istftnet.py:292`.
fn reflection_pad_left1(x: &[f32], channels: usize, t: usize) -> Vec<f32> {
    debug_assert!(t >= 2, "reflection pad needs at least 2 frames");
    let t_out = t + 1;
    let mut out = vec![0.0f32; channels * t_out];
    for c in 0..channels {
        let src = &x[c * t..(c + 1) * t];
        let dst = &mut out[c * t_out..(c + 1) * t_out];
        dst[0] = src[1];
        dst[1..].copy_from_slice(src);
    }
    out
}

// KokoroConfig is imported by decoder.rs and not required here directly, but we
// re-export the alias so mod.rs sees no surprise ordering when it consumes
// generator via `decoder::forward_full`.
#[allow(dead_code)]
fn _module_link_check(_config: &KokoroConfig) {}
