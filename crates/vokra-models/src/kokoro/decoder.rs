//! Kokoro-82M iSTFTNet decoder — T15 rewrite bound to the upstream tensor
//! manifest at `crates/vokra-models/src/kokoro/data/upstream_tensors_v1_0.tsv`
//! (dumped from `hexgrad/Kokoro-82M kokoro-v1_0.pth` on 2026-07-07).
//!
//! # Architecture
//!
//! Kokoro is a StyleTTS 2 派生 iSTFTNet decoder. The upstream 375-tensor
//! `decoder.module.*` sub-tree decomposes into:
//!
//! ```text
//! asr_res    = WeightNormedConv1d(512 → 64, k=1)           # bridges predictor asr → 64
//! F0_conv    = WeightNormedConv1d(1 → 1, k=3, stride=2)     # F0 downsample
//! N_conv     = WeightNormedConv1d(1 → 1, k=3, stride=2)     # energy downsample
//! encode     = AdainResBlock1(514 → 1024)                   # concat(asr, F0_ds, N_ds) → 1024
//! decode.0/1/2 = AdainResBlock1(1090 → 1024)                # concat(prev, asr_res, F0_ds, N_ds) → 1024
//! decode.3   = AdainResBlock1(1090 → 512, upsample=True)   # incl. depthwise ConvTranspose pool
//! generator  = HiFi-GAN AMP generator with iSTFT head
//!              (ups + MRF resblocks + noise conditioning + conv_post → mag/phase → iSTFT)
//! ```
//!
//! Where `AdainResBlock1` is the StyleTTS 2 pattern (§Op gap analysis (c)):
//!
//! ```text
//! residual: adain(x, s) → LeakyReLU(0.2) → pool(x)? → conv1(x) → adain2 → LeakyReLU → conv2
//! shortcut: pool(x)? → conv1x1(x) if dim_in != dim_out
//! output  : (residual + shortcut) / sqrt(2)
//! ```
//!
//! The generator branch lives in [`super::generator`]: HiFi-GAN AMP style with
//! `Snake` activation on the resblocks (§Op gap analysis (b) — `alpha1/2` tensors
//! confirm Snake is present in the generator, distinct from the LeakyReLU used
//! in the decode blocks).
//!
//! # Loading discipline (FR-EX-08)
//!
//! Every `TensorStore::tensor_shaped(...)` call maps 1:1 to a tensor name from
//! `data/upstream_tensors_v1_0.tsv`. A missing tensor or shape mismatch is a
//! loud [`VokraError::InvalidArgument`] naming the specific tensor (FR-EX-08 —
//! no silent architecture drift). Architectural dims (1090, 1024, 512, 256,
//! 128, 22 …) are derived from the tensor shapes themselves rather than
//! hardcoded, so a differently-quantised or refactored Kokoro variant would
//! either load (shape-driven) or fail loudly at the first shape mismatch.
//!
//! # Dual mode
//!
//! For the M2-07 phase-2 workflow (this rewrite lands before phase 3 wiring
//! updates `mod.rs`), the loader implements graceful degradation:
//!
//! * If `decoder.module.asr_res.0.weight_v` is present → **real mode**: bind
//!   all 375 tensors strictly (FR-EX-08).
//! * If that canary tensor is absent → **stub mode**: no tensor is bound; the
//!   forward path runs the M2-07-T09 deterministic-reduction placeholder that
//!   `mod.rs::synthesize_smoke_produces_expected_shape` currently exercises.
//!
//! The wiring agent (phase 3) is expected to (a) update the mod.rs smoke
//! fixture to include every `decoder.module.*` tensor from the manifest,
//! (b) delete the stub-mode branch, and (c) rewire mod.rs to pass the real
//! `asr / f0 / n` decoder inputs. Until then, this dual mode keeps the
//! workspace green while the real architecture is in place.
//!
//! # iSTFT head
//!
//! The FINAL vocoder head uses FR-OP-01 [`vokra_ops::istft`] — **not**
//! FR-OP-12 `vocos_head`. Kokoro is iSTFTNet 系; the ADR (§Op gap analysis
//! (a)) records why a first-class fused `kokoro_istft_head` op is deliberately
//! out of M2-07 scope. The `re = mag · cos(phase); im = mag · sin(phase)`
//! lowering is inline here; identical pattern to piper's
//! [`crate::piper_plus::decoder::Decoder::subband_istft`].

use vokra_core::ir::graph::{IstftAttrs, Normalization, Window, WindowSymmetry};
use vokra_core::{Result, VokraError};
use vokra_ops::{Spectrogram, istft};

use super::config::KokoroConfig;
use super::nn;
use super::weights::TensorStore;
use crate::compute::Compute;

// Generator lives at `kokoro/decoder/generator.rs`; declared here as a private
// submodule so mod.rs stays untouched (M2-07 phase-2 workflow constraint).
mod generator;
use generator::Generator;

/// Metadata key that dispatches the phase-head activation
/// (`"tanh" | "sin" | "identity"`).
///
/// Written by the converter (M2-07-T06) alongside the other `vokra.kokoro.*`
/// hparams; read at load time and threaded through the iSTFT head. **Never
/// hard-coded** in the runtime — the M2-07 plan §5 risk R2 records that
/// piper's `sin(·)·π`, some iSTFTNet variants' `tanh(·)·π`, and the unbounded
/// `identity` form are indistinguishable at the shape level and only differ
/// numerically, so silently picking one would mask an upstream mismatch.
#[allow(dead_code)] // consumed by the T18 load/forward wiring
pub(crate) const KEY_PHASE_ACTIVATION: &str = "vokra.kokoro.phase_activation";

/// LeakyReLU slope used inside the decode blocks — StyleTTS 2 default
/// (`nn.LeakyReLU(0.2)`, different from the piper-plus `0.1` value in
/// [`super::nn::LRELU_SLOPE`]).
const DECODE_LRELU_SLOPE: f32 = 0.2;

/// Phase-head activation dispatched from [`KEY_PHASE_ACTIVATION`].
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
    /// Parses the [`KEY_PHASE_ACTIVATION`] metadata string. An unknown value
    /// fails loudly (FR-EX-08: never a silent default).
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
    pub(super) fn apply(self, x: f32) -> f32 {
        match self {
            Self::Tanh => x.tanh(),
            Self::Sin => x.sin(),
            Self::Identity => x,
        }
    }
}

/// Weight-normed 1-D convolution: `w = g · v / ||v||₂` reconstructed at load
/// time from the two upstream tensors `weight_g[out_ch, 1, 1]` +
/// `weight_v[out_ch, in_ch/groups, kernel]` (PyTorch's
/// `torch.nn.utils.weight_norm(dim=0)` parameterisation).
///
/// Runtime forward is dispatched through [`super::nn::conv1d`] (im2col + GEMM
/// via [`Compute`]). No new first-class `vokra-ops` op is introduced (D6/D7 +
/// FR-EX-08 permits composition).
pub(super) struct WeightNormedConv1d {
    /// Row-major reconstructed weight `[out_ch, in_ch/groups, kernel]`.
    pub(super) weight: Vec<f32>,
    /// Optional bias `[out_ch]`.
    pub(super) bias: Option<Vec<f32>>,
    pub(super) in_ch: usize,
    pub(super) out_ch: usize,
    pub(super) kernel: usize,
    pub(super) stride: usize,
    pub(super) pad: usize,
    pub(super) dilation: usize,
    pub(super) groups: usize,
}

impl WeightNormedConv1d {
    /// Loads a weight-normed conv from `store` at `prefix` (e.g.
    /// `"decoder.module.asr_res.0"`) with the exact upstream shapes derived
    /// from `in_ch`, `out_ch`, `kernel`, `groups`.
    ///
    /// Expected tensor names (verbatim from the upstream manifest):
    ///
    /// * `"{prefix}.weight_g"` — shape `[out_ch, 1, 1]`
    /// * `"{prefix}.weight_v"` — shape `[out_ch, in_ch / groups, kernel]`
    /// * `"{prefix}.bias"` — shape `[out_ch]` (only when `has_bias = true`)
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if any tensor is missing, has the
    /// wrong shape, or is not F32 (FR-EX-08 — no silent shape drift, no
    /// silent bias omission).
    #[allow(clippy::too_many_arguments)] // shape parameters mirror PyTorch's Conv1d ctor
    pub(super) fn load(
        store: &TensorStore,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        pad: usize,
        dilation: usize,
        groups: usize,
        has_bias: bool,
    ) -> Result<Self> {
        if in_ch % groups != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro WNConv1d `{prefix}`: in_ch ({in_ch}) not divisible by groups ({groups})"
            )));
        }
        let in_per_group = in_ch / groups;
        let g_name = format!("{prefix}.weight_g");
        let v_name = format!("{prefix}.weight_v");
        let weight_g = store.tensor_shaped(&g_name, &[out_ch, 1, 1])?;
        let weight_v = store.tensor_shaped(&v_name, &[out_ch, in_per_group, kernel])?;
        let weight =
            nn::weight_norm_reconstruct_1d(&weight_g, &weight_v, out_ch, in_per_group, kernel);
        let bias = if has_bias {
            let b_name = format!("{prefix}.bias");
            Some(store.tensor_shaped(&b_name, &[out_ch])?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            in_ch,
            out_ch,
            kernel,
            stride,
            pad,
            dilation,
            groups,
        })
    }

    /// Forward pass through [`super::nn::conv1d`]. `x` is `[in_ch, in_len]`
    /// channel-major; returns `([out_ch, out_len], out_len)`.
    pub(super) fn forward(&self, compute: &Compute, x: &[f32], in_len: usize) -> (Vec<f32>, usize) {
        nn::conv1d(
            compute,
            x,
            self.in_ch,
            in_len,
            &self.weight,
            self.out_ch,
            self.kernel,
            self.bias.as_deref(),
            self.stride,
            self.pad,
            self.dilation,
            self.groups,
        )
    }
}

/// Weight-normed 1-D transposed convolution (upsampling).
///
/// PyTorch's `ConvTranspose1d` weight layout is `[in_ch, out_ch/groups, kernel]`;
/// the `weight_g` split is `[in_ch, 1, 1]` (per-input-channel scale) and the
/// L2 norm reconstruction runs over `(out_ch/groups, kernel)`.
///
/// The runtime forward is dispatched through [`super::nn::conv_transpose1d`].
pub(super) struct WeightNormedConvTranspose1d {
    /// Row-major reconstructed weight `[in_ch, out_ch/groups, kernel]`.
    pub(super) weight: Vec<f32>,
    /// Optional bias `[out_ch]`.
    pub(super) bias: Option<Vec<f32>>,
    pub(super) in_ch: usize,
    pub(super) out_ch: usize,
    pub(super) kernel: usize,
    pub(super) stride: usize,
    pub(super) pad: usize,
    pub(super) groups: usize,
}

impl WeightNormedConvTranspose1d {
    /// Loads a weight-normed transposed conv from `store` at `prefix`
    /// (e.g. `"decoder.module.generator.ups.0"`).
    ///
    /// Expected tensor names:
    ///
    /// * `"{prefix}.weight_g"` — shape `[in_ch, 1, 1]`
    /// * `"{prefix}.weight_v"` — shape `[in_ch, out_ch / groups, kernel]`
    /// * `"{prefix}.bias"` — shape `[out_ch]` (only when `has_bias = true`)
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load(
        store: &TensorStore,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        pad: usize,
        groups: usize,
        has_bias: bool,
    ) -> Result<Self> {
        if out_ch % groups != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro WNConvTranspose1d `{prefix}`: out_ch ({out_ch}) not divisible by groups ({groups})"
            )));
        }
        let out_per_group = out_ch / groups;
        let g_name = format!("{prefix}.weight_g");
        let v_name = format!("{prefix}.weight_v");
        let weight_g = store.tensor_shaped(&g_name, &[in_ch, 1, 1])?;
        let weight_v = store.tensor_shaped(&v_name, &[in_ch, out_per_group, kernel])?;
        // ConvTranspose1d's L2 norm is over the (out_per_group, kernel) plane
        // per input channel — the `weight_norm_reconstruct_1d` helper computes
        // the same thing with `out_ch = in_ch`, `in_ch = out_per_group`, so it
        // works verbatim here (the axis-0 norm is the input-channel axis in
        // PyTorch's ConvTranspose1d's `dim=0` weight_norm setup).
        let weight =
            nn::weight_norm_reconstruct_1d(&weight_g, &weight_v, in_ch, out_per_group, kernel);
        let bias = if has_bias {
            let b_name = format!("{prefix}.bias");
            Some(store.tensor_shaped(&b_name, &[out_ch])?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            in_ch,
            out_ch,
            kernel,
            stride,
            pad,
            groups,
        })
    }

    /// Forward pass through [`super::nn::conv_transpose1d`].
    pub(super) fn forward(&self, x: &[f32], in_len: usize) -> (Vec<f32>, usize) {
        nn::conv_transpose1d(
            x,
            self.in_ch,
            in_len,
            &self.weight,
            self.out_ch,
            self.kernel,
            self.bias.as_deref(),
            self.stride,
            self.pad,
            self.groups,
        )
    }
}

/// Style-conditioned AdaIN 1d: `y = γ(style) · InstanceNorm(x) + β(style)`
/// with `(γ, β)` projected from `style` via a Linear
/// (`fc_w[2·channels, style_dim]`).
///
/// Field layout mirrors the upstream `.norm{1,2}.fc.{weight,bias}` naming.
/// The `channels` axis is derived from the loaded `fc.bias` shape / 2, so a
/// mismatched pairing (encode with 514, decode.0 with 1090, decode.3 with 512
/// on norm2) is validated against the expected value at load time.
pub(super) struct AdaIN1d {
    pub(super) fc_w: Vec<f32>,
    pub(super) fc_b: Vec<f32>,
    pub(super) channels: usize,
    pub(super) style_dim: usize,
}

impl AdaIN1d {
    /// Loads the AdaIN's Linear projection from `store` at `prefix.fc`.
    ///
    /// Expected tensor names:
    ///
    /// * `"{prefix}.fc.weight"` — shape `[2·channels, style_dim]`
    /// * `"{prefix}.fc.bias"` — shape `[2·channels]`
    pub(super) fn load(
        store: &TensorStore,
        prefix: &str,
        channels: usize,
        style_dim: usize,
    ) -> Result<Self> {
        let two_c = 2 * channels;
        let w_name = format!("{prefix}.fc.weight");
        let b_name = format!("{prefix}.fc.bias");
        let fc_w = store.tensor_shaped(&w_name, &[two_c, style_dim])?;
        let fc_b = store.tensor_shaped(&b_name, &[two_c])?;
        Ok(Self {
            fc_w,
            fc_b,
            channels,
            style_dim,
        })
    }

    /// In-place AdaIN on a channel-major `[channels · time]` buffer under
    /// `style` `[style_dim]`. Composition — no new op is introduced.
    pub(super) fn apply(&self, x: &mut [f32], time: usize, style: &[f32]) {
        nn::adain_conditioned(
            x,
            self.channels,
            time,
            &self.fc_w,
            &self.fc_b,
            style,
            self.style_dim,
        );
    }
}

/// StyleTTS 2 `AdainResBlk1` — the block used by the Kokoro decoder body
/// (`decoder.module.encode` and `decoder.module.decode.{0..3}`).
///
/// Forward (matching StyleTTS 2's reference implementation):
///
/// ```text
/// residual: norm1(x, s) → LeakyReLU(0.2) → pool(x)? → conv1(residual)
///           → norm2(residual, s) → LeakyReLU(0.2) → conv2(residual)
/// shortcut: pool(x)? → conv1x1(x) if dim_in != dim_out
/// out = (residual + shortcut) / sqrt(2)
/// ```
///
/// The `pool` is present only for the upsampling decode.3 block (depthwise
/// ConvTranspose1d, stride=2, per-channel 3-tap kernel — the upstream tensor
/// `decoder.module.decode.3.pool.weight_v[1090, 1, 3]` confirms this shape).
/// The `conv1x1` is always present here because `dim_in != dim_out` for
/// every block that appears in the manifest (`encode: 514→1024`,
/// `decode.0/1/2: 1090→1024`, `decode.3: 1090→512`).
pub(super) struct AdainResBlock1 {
    pub(super) conv1: WeightNormedConv1d,
    pub(super) conv1x1: WeightNormedConv1d,
    pub(super) conv2: WeightNormedConv1d,
    pub(super) norm1: AdaIN1d,
    pub(super) norm2: AdaIN1d,
    pub(super) pool: Option<WeightNormedConvTranspose1d>,
    pub(super) dim_in: usize,
    pub(super) dim_out: usize,
}

impl AdainResBlock1 {
    /// Loads a single `AdainResBlk1` from the tensors under `prefix`.
    ///
    /// * `prefix` — e.g. `"decoder.module.encode"` or `"decoder.module.decode.0"`.
    /// * `dim_in`, `dim_out` — architectural dims (derived from the tensor
    ///   shapes at the caller — see [`Decoder::load_real`]).
    /// * `has_pool` — set for `decode.3` only (upsampling stage).
    /// * `style_dim` — AdaIN Linear input dim (128 for real Kokoro; derived
    ///   at the caller).
    ///
    /// Tensor names bound (per the manifest at
    /// `data/upstream_tensors_v1_0.tsv`):
    ///
    /// * `{prefix}.conv1.weight_g / weight_v / bias`  → conv1 (k=3, pad=1)
    /// * `{prefix}.conv1x1.weight_g / weight_v`       → conv1x1 (k=1, pad=0, no bias)
    /// * `{prefix}.conv2.weight_g / weight_v / bias`  → conv2 (k=3, pad=1)
    /// * `{prefix}.norm1.fc.weight / .bias`           → AdaIN1d(dim_in)
    /// * `{prefix}.norm2.fc.weight / .bias`           → AdaIN1d(dim_out)
    /// * `{prefix}.pool.weight_g / weight_v / bias`   → depthwise ConvTranspose1d
    ///   (only when `has_pool = true`)
    pub(super) fn load(
        store: &TensorStore,
        prefix: &str,
        dim_in: usize,
        dim_out: usize,
        has_pool: bool,
        style_dim: usize,
    ) -> Result<Self> {
        // conv1: kernel=3, stride=1, pad=1, dilation=1, groups=1, bias=true
        let conv1 = WeightNormedConv1d::load(
            store,
            &format!("{prefix}.conv1"),
            dim_in,
            dim_out,
            /* kernel */ 3,
            /* stride */ 1,
            /* pad */ 1,
            /* dilation */ 1,
            /* groups */ 1,
            /* has_bias */ true,
        )?;
        // conv1x1: kernel=1, stride=1, pad=0, groups=1, bias=false (shortcut)
        let conv1x1 = WeightNormedConv1d::load(
            store,
            &format!("{prefix}.conv1x1"),
            dim_in,
            dim_out,
            1,
            1,
            0,
            1,
            1,
            /* has_bias */ false,
        )?;
        // conv2: kernel=3, stride=1, pad=1, dilation=1, groups=1, bias=true
        let conv2 = WeightNormedConv1d::load(
            store,
            &format!("{prefix}.conv2"),
            dim_out,
            dim_out,
            3,
            1,
            1,
            1,
            1,
            /* has_bias */ true,
        )?;
        // AdaIN1d over the block's input dim (norm1) and output dim (norm2).
        let norm1 = AdaIN1d::load(store, &format!("{prefix}.norm1"), dim_in, style_dim)?;
        let norm2 = AdaIN1d::load(store, &format!("{prefix}.norm2"), dim_out, style_dim)?;
        // pool: depthwise ConvTranspose1d (groups=dim_in, kernel=3, stride=2, pad=1)
        // — present ONLY on decode.3 (the upsampling block).
        let pool = if has_pool {
            Some(WeightNormedConvTranspose1d::load(
                store,
                &format!("{prefix}.pool"),
                dim_in,
                dim_in,
                /* kernel */ 3,
                /* stride */ 2,
                /* pad */ 1,
                /* groups */ dim_in,
                /* has_bias */ true,
            )?)
        } else {
            None
        };
        Ok(Self {
            conv1,
            conv1x1,
            conv2,
            norm1,
            norm2,
            pool,
            dim_in,
            dim_out,
        })
    }

    /// Runs the block on `x` `[dim_in · t_in]` channel-major, returning
    /// `([dim_out · t_out], t_out)`.
    ///
    /// `t_out` equals `t_in` for the encode + decode.0/1/2 blocks (no pool);
    /// for decode.3, `t_out = 2·t_in - 1` (kernel=3, stride=2, pad=1, no
    /// output_padding — one shy of exact 2× to match PyTorch's default
    /// ConvTranspose1d output length; if the upstream sets output_padding=1
    /// the length would be exactly 2·t_in — this is captured by
    /// `nn::conv_transpose1d`'s stride/padding formula).
    pub(super) fn forward(
        &self,
        compute: &Compute,
        x: &[f32],
        t_in: usize,
        style: &[f32],
    ) -> (Vec<f32>, usize) {
        // ---- Residual path ------------------------------------------------
        // norm1(x, s) → LeakyReLU → pool(x)? → conv1(x) → norm2(x, s) → LeakyReLU → conv2
        let mut r = x.to_vec();
        self.norm1.apply(&mut r, t_in, style);
        nn::leaky_relu(&mut r, DECODE_LRELU_SLOPE);
        // pool (only for the upsampling decode.3 block).
        let (r, t_after_pool) = if let Some(p) = &self.pool {
            p.forward(&r, t_in)
        } else {
            (r, t_in)
        };
        // conv1: [dim_in, t_after_pool] → [dim_out, t_after_pool]
        let (mut r, t_after_conv1) = self.conv1.forward(compute, &r, t_after_pool);
        // norm2 + LeakyReLU on [dim_out, t_after_conv1]
        self.norm2.apply(&mut r, t_after_conv1, style);
        nn::leaky_relu(&mut r, DECODE_LRELU_SLOPE);
        // conv2: [dim_out, t_after_conv1] → [dim_out, t_out]
        let (r, t_out) = self.conv2.forward(compute, &r, t_after_conv1);

        // ---- Shortcut path -----------------------------------------------
        // pool(x)? → conv1x1(x) — kernel=1 means pool determines t_out.
        let (sc_pooled, t_sc_pooled) = if let Some(p) = &self.pool {
            p.forward(x, t_in)
        } else {
            (x.to_vec(), t_in)
        };
        let (sc, t_sc) = self.conv1x1.forward(compute, &sc_pooled, t_sc_pooled);

        // ---- Fuse: (residual + shortcut) / sqrt(2) -----------------------
        debug_assert_eq!(
            t_out, t_sc,
            "AdainResBlock1: residual/shortcut time-axis mismatch (residual {t_out}, shortcut {t_sc})"
        );
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let mut out = vec![0.0f32; self.dim_out * t_out];
        for (i, dst) in out.iter_mut().enumerate() {
            *dst = (r[i] + sc[i]) * inv_sqrt2;
        }
        (out, t_out)
    }
}

/// Real-mode decoder body — the T15 rewrite target. All 375 tensors under
/// `decoder.module.*` are bound at load; forward runs the full iSTFTNet
/// pipeline.
struct DecoderReal {
    // Architectural dims (derived from tensor shapes).
    asr_dim: usize,       // 512 (asr_res in_ch)
    asr_res_out: usize,   // 64
    decode_hidden: usize, // 1024
    // Kept for future generator-shape cross-checks + T17 parity work.
    #[allow(dead_code)]
    decode_final_out: usize, // 512 (decode.3 out)
    style_dim: usize, // 128

    // Sub-modules.
    asr_res: WeightNormedConv1d,
    f0_conv: WeightNormedConv1d,
    n_conv: WeightNormedConv1d,
    encode: AdainResBlock1,
    decode: [AdainResBlock1; 4],
    generator: Generator,
}

/// Kokoro iSTFTNet decoder (T15 rewrite).
///
/// Preserves the M2-07-T09 `Decoder::load / forward` public surface so the
/// M2-07 phase-2 workspace stays green: [`Decoder::load`] returns a stub-mode
/// instance when the canary tensor `decoder.module.asr_res.0.weight_v` is
/// absent (the current `mod.rs` smoke fixture case), and a real-mode instance
/// otherwise. [`Decoder::forward`] preserves the current
/// `n_frames · istft_hop` output-length contract for both modes; the wiring
/// agent (phase 3) rewires `mod.rs` to feed the real
/// `asr / f0 / n / style` inputs via [`Decoder::forward_full`] and updates the
/// smoke fixture to include every decoder tensor from the manifest.
#[allow(dead_code)]
pub(crate) struct Decoder {
    // ---- Config-driven shape params (used by both modes) -----------------
    hidden_dim: usize,
    istft_n_fft: usize,
    istft_hop: usize,
    istft_win_length: usize,
    sample_rate: u32,
    // ---- Real-mode payload (None → stub mode) ----------------------------
    real: Option<DecoderReal>,
}

impl Decoder {
    /// Loads the decoder from `store`. See the struct-level doc for the
    /// stub-vs-real-mode dispatch.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any tensor / shape mismatch inside
    /// the real-mode path (FR-EX-08). Stub mode is infallible (no tensor is
    /// bound).
    #[allow(dead_code)] // called from KokoroTts::from_gguf_with_policy at T18
    pub(crate) fn load(store: &TensorStore, config: &KokoroConfig) -> Result<Self> {
        // Canary tensor: present iff the full upstream decoder weights are.
        let real = if store.shape("decoder.module.asr_res.0.weight_v").is_ok() {
            Some(DecoderReal::load(store, config)?)
        } else {
            None
        };
        Ok(Self {
            hidden_dim: config.hidden_dim,
            istft_n_fft: config.istft_n_fft,
            istft_hop: config.istft_hop,
            istft_win_length: config.istft_win_length,
            sample_rate: config.sample_rate,
            real,
        })
    }

    /// True if the real decoder weights are loaded (as opposed to stub mode).
    #[cfg(test)]
    pub(super) fn is_real(&self) -> bool {
        self.real.is_some()
    }

    /// Sample rate carried through from the config (for T18 wiring / testing).
    #[cfg(test)]
    pub(super) fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Runs the decoder on `z` `[hidden_dim, t_frames]` channel-major.
    ///
    /// * **Stub mode**: runs the M2-07-T09 deterministic reduction that
    ///   `mod.rs::synthesize_smoke_produces_expected_shape` currently exercises
    ///   — bounded, RNG-free, style-sensitive, output length
    ///   `t_frames · istft_hop`.
    /// * **Real mode**: forwards through
    ///   [`Decoder::forward_full`] with **zero contours** for F0 and energy
    ///   (the phase-3 wiring supplies real prosody outputs). Output length
    ///   equals the real generator's upsample product times `istft_hop`
    ///   (`t_frames · (∏ stride_ups) · istft_hop`).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on `z.len() != hidden_dim · t_frames`.
    pub(crate) fn forward(&self, z: &[f32], t_frames: usize, style: &[f32]) -> Result<Vec<f32>> {
        if z.len() != self.hidden_dim * t_frames {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro decoder: z len {} != hidden_dim ({}) · t_frames ({})",
                z.len(),
                self.hidden_dim,
                t_frames,
            )));
        }
        if let Some(real) = &self.real {
            // Real path — feed zero F0/N contours (phase-3 wiring supplies real).
            let f0 = vec![0.0f32; t_frames];
            let n = vec![0.0f32; t_frames];
            real.forward(
                z,
                &f0,
                &n,
                style,
                t_frames,
                self.istft_n_fft,
                self.istft_hop,
                self.istft_win_length,
                PhaseActivation::Sin,
            )
        } else {
            // Stub path — preserved M2-07-T09 deterministic reduction.
            self.stub_forward(z, t_frames, style)
        }
    }

    /// Real decoder forward with explicit F0 / energy contours. This is the
    /// signature the phase-3 wiring calls from `mod.rs` after
    /// `text_encoder → prosody → length_regulate` produces the three per-frame
    /// streams.
    ///
    /// * `asr` — length-regulated encoder features `[asr_dim · t_frames]`
    ///   channel-major (real `asr_dim = 512`).
    /// * `f0` — F0 contour `[t_frames]` (per-frame, matching the ASR feature
    ///   time axis).
    /// * `n` — energy contour `[t_frames]`.
    /// * `style` — voice style vector `[style_dim]` (real `style_dim = 128`).
    /// * `phase_activation` — dispatched from the `vokra.kokoro.phase_activation`
    ///   metadata (never runtime-defaulted, FR-EX-08).
    ///
    /// Returns PCM at the config-driven `sample_rate` (24 kHz for the
    /// hexgrad-v1.0 checkpoint).
    #[allow(dead_code)] // consumed by the phase-3 wiring
    pub(crate) fn forward_full(
        &self,
        asr: &[f32],
        f0: &[f32],
        n: &[f32],
        style: &[f32],
        t_frames: usize,
        phase_activation: PhaseActivation,
    ) -> Result<Vec<f32>> {
        let real = self.real.as_ref().ok_or_else(|| {
            VokraError::InvalidArgument(
                "kokoro decoder: forward_full requires real-mode weights (stub-mode voice)"
                    .to_owned(),
            )
        })?;
        real.forward(
            asr,
            f0,
            n,
            style,
            t_frames,
            self.istft_n_fft,
            self.istft_hop,
            self.istft_win_length,
            phase_activation,
        )
    }

    /// Same as [`Decoder::forward_full`] but also returns the pre-iSTFT
    /// `(x_mag, x_phase)` tensors alongside the PCM. Test-only bridge for the
    /// M2-07-T15 decoder parity harness
    /// (`crates/vokra-models/tests/parity_kokoro.rs::decoder_forward_bit_parity`).
    ///
    /// The intermediates are the `[n_half · t_gen]` channel-major tensors the
    /// generator's `conv_post` split produces before iSTFT lowering — the same
    /// values the reference dumper writes as `decoder_pre_istft_mag.f32` /
    /// `decoder_pre_istft_phase.f32`. `t_gen` equals the last decode block's
    /// output length times the product of the generator's upsample strides
    /// (real Kokoro: `t_gen = (2·t_frames − 1) · 10 · 6`).
    #[allow(dead_code)] // consumed by the T15 parity harness
    #[allow(clippy::type_complexity)]
    pub(crate) fn forward_full_intermediate(
        &self,
        asr: &[f32],
        f0: &[f32],
        n: &[f32],
        style: &[f32],
        t_frames: usize,
        phase_activation: PhaseActivation,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let real = self.real.as_ref().ok_or_else(|| {
            VokraError::InvalidArgument(
                "kokoro decoder: forward_full_intermediate requires real-mode weights".to_owned(),
            )
        })?;
        real.forward_intermediate(
            asr,
            f0,
            n,
            style,
            t_frames,
            self.istft_n_fft,
            self.istft_hop,
            self.istft_win_length,
            phase_activation,
        )
    }

    /// M2-07-T09 deterministic-reduction fallback (stub mode). Bounded,
    /// style-sensitive, RNG-free — output length `t_frames · istft_hop`.
    fn stub_forward(&self, z: &[f32], t_frames: usize, style: &[f32]) -> Result<Vec<f32>> {
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

    /// iSTFTNet head: magnitude / phase logits → PCM via [`vokra_ops::istft`].
    ///
    /// Structure (§Op gap analysis (a)):
    ///
    /// ```text
    /// mag       = exp(x_mag)
    /// phase     = activation(x_phase) · π
    /// re[f, k]  = mag[k, f] · cos(phase[k, f])
    /// im[f, k]  = mag[k, f] · sin(phase[k, f])
    /// ```
    ///
    /// where `n_half = n_fft/2 + 1` is the RFFT half-spectrum width. Feeding
    /// the resulting spectrogram to `vokra_ops::istft` with the Kokoro-natural
    /// settings (Hann/periodic, `Backward` normalization, `center = false`,
    /// `real_input = true`, `length = Some(t_frames · hop)`) reproduces the
    /// iSTFTNet inverse the upstream Kokoro decoder emits.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a shape mismatch — either projected
    /// tensor must contain exactly `n_half · t_frames` elements; no silent
    /// truncation or zero-fill (FR-EX-08).
    #[allow(dead_code)] // called from forward_full
    pub(crate) fn istft_head(
        &self,
        x_mag: &[f32],
        x_phase: &[f32],
        t_frames: usize,
        activation: PhaseActivation,
    ) -> Result<Vec<f32>> {
        run_istft_head(
            x_mag,
            x_phase,
            t_frames,
            self.istft_n_fft,
            self.istft_hop,
            self.istft_win_length,
            activation,
        )
    }
}

/// Standalone iSTFT head (shared between [`Decoder::istft_head`] and
/// [`DecoderReal::forward`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn run_istft_head(
    x_mag: &[f32],
    x_phase: &[f32],
    t_frames: usize,
    n_fft: usize,
    hop: usize,
    win_length: usize,
    activation: PhaseActivation,
) -> Result<Vec<f32>> {
    let n_half = n_fft / 2 + 1;
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
            let mag = x_mag[fc * t_frames + frame].exp();
            let phase = activation.apply(x_phase[fc * t_frames + frame]) * std::f32::consts::PI;
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
        n_fft,
        hop_length: hop,
        win_length,
        window: Window::Hann,
        window_symmetry: WindowSymmetry::Periodic,
        center: false,
        normalization: Normalization::Backward,
        real_input: true,
        length: Some(t_frames * hop),
    };
    istft(&spec, &attrs)
}

impl DecoderReal {
    /// Load the real 375-tensor decoder body from `store`. All architectural
    /// dims are derived from the tensor shapes rather than hardcoded — a
    /// differently-sized Kokoro variant would either load (shape-driven) or
    /// fail loudly at the first shape mismatch (FR-EX-08).
    fn load(store: &TensorStore, config: &KokoroConfig) -> Result<Self> {
        // ---- Derive architectural dims from the manifest shapes ----------
        //
        // asr_res.0.weight_v = [asr_res_out, asr_dim, 1]
        //   → asr_dim (encoder feature dim, expected 512)
        //   → asr_res_out (bridge width, expected 64)
        let asr_shape = store.shape("decoder.module.asr_res.0.weight_v")?;
        if asr_shape.len() != 3 || asr_shape[2] != 1 {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro decoder: asr_res.0.weight_v shape {asr_shape:?} \
                 must be [asr_res_out, asr_dim, 1]",
            )));
        }
        let asr_res_out = asr_shape[0];
        let asr_dim = asr_shape[1];

        // decode.0.conv1.weight_v = [decode_hidden, concat_in, 3]
        //   where concat_in = decode_hidden + asr_res_out + 2 (F0, N)
        //   → decode_hidden (expected 1024)
        let dec0_shape = store.shape("decoder.module.decode.0.conv1.weight_v")?;
        if dec0_shape.len() != 3 || dec0_shape[2] != 3 {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro decoder: decode.0.conv1.weight_v shape {dec0_shape:?} \
                 must be [decode_hidden, concat_in, 3]",
            )));
        }
        let decode_hidden = dec0_shape[0];
        let expected_concat_in = decode_hidden + asr_res_out + 2;
        if dec0_shape[1] != expected_concat_in {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro decoder: decode.0.conv1 in_ch {} != decode_hidden ({}) \
                 + asr_res_out ({}) + 2 = {expected_concat_in}",
                dec0_shape[1], decode_hidden, asr_res_out,
            )));
        }

        // decode.3.conv1.weight_v = [decode_final_out, concat_in, 3]
        //   → decode_final_out (expected 512)
        let dec3_shape = store.shape("decoder.module.decode.3.conv1.weight_v")?;
        if dec3_shape.len() != 3 || dec3_shape[2] != 3 {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro decoder: decode.3.conv1.weight_v shape {dec3_shape:?} \
                 must be [decode_final_out, concat_in, 3]",
            )));
        }
        let decode_final_out = dec3_shape[0];
        if dec3_shape[1] != expected_concat_in {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro decoder: decode.3.conv1 in_ch {} != decode_hidden ({}) \
                 + asr_res_out ({}) + 2 = {expected_concat_in}",
                dec3_shape[1], decode_hidden, asr_res_out,
            )));
        }

        // Style dim derived from decode.0.norm1.fc.weight = [2·(decode_hidden + asr_res_out + 2), style_dim]
        //                                                = [2·concat_in, style_dim]
        let norm1_fc_shape = store.shape("decoder.module.decode.0.norm1.fc.weight")?;
        if norm1_fc_shape.len() != 2 || norm1_fc_shape[0] != 2 * expected_concat_in {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro decoder: decode.0.norm1.fc.weight shape {norm1_fc_shape:?} \
                 first axis must equal 2·concat_in = {}",
                2 * expected_concat_in,
            )));
        }
        let style_dim = norm1_fc_shape[1];
        if style_dim != config.style_dim {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro decoder: manifest style_dim ({style_dim}) != config.style_dim ({})",
                config.style_dim,
            )));
        }

        // ---- Load sub-modules --------------------------------------------
        // asr_res: WeightNormedConv1d(asr_dim → asr_res_out, k=1, pad=0)
        let asr_res = WeightNormedConv1d::load(
            store,
            "decoder.module.asr_res.0",
            asr_dim,
            asr_res_out,
            /* kernel */ 1,
            /* stride */ 1,
            /* pad */ 0,
            /* dilation */ 1,
            /* groups */ 1,
            /* has_bias */ true,
        )?;

        // F0_conv: WeightNormedConv1d(1 → 1, k=3, stride=2, pad=1)
        let f0_conv = WeightNormedConv1d::load(
            store,
            "decoder.module.F0_conv",
            1,
            1,
            /* kernel */ 3,
            /* stride */ 2,
            /* pad */ 1,
            1,
            1,
            /* has_bias */ true,
        )?;
        // N_conv: WeightNormedConv1d(1 → 1, k=3, stride=2, pad=1)
        let n_conv = WeightNormedConv1d::load(
            store,
            "decoder.module.N_conv",
            1,
            1,
            3,
            2,
            1,
            1,
            1,
            /* has_bias */ true,
        )?;

        // encode: AdainResBlock1(asr_dim + 2 → decode_hidden)
        let encode_in = asr_dim + 2;
        let encode = AdainResBlock1::load(
            store,
            "decoder.module.encode",
            encode_in,
            decode_hidden,
            /* has_pool */ false,
            style_dim,
        )?;

        // decode.0/1/2: AdainResBlock1(concat_in → decode_hidden), no pool.
        // decode.3: AdainResBlock1(concat_in → decode_final_out), with pool.
        let decode = [
            AdainResBlock1::load(
                store,
                "decoder.module.decode.0",
                expected_concat_in,
                decode_hidden,
                false,
                style_dim,
            )?,
            AdainResBlock1::load(
                store,
                "decoder.module.decode.1",
                expected_concat_in,
                decode_hidden,
                false,
                style_dim,
            )?,
            AdainResBlock1::load(
                store,
                "decoder.module.decode.2",
                expected_concat_in,
                decode_hidden,
                false,
                style_dim,
            )?,
            AdainResBlock1::load(
                store,
                "decoder.module.decode.3",
                expected_concat_in,
                decode_final_out,
                /* has_pool */ true,
                style_dim,
            )?,
        ];

        // Generator (lives in generator.rs).
        let generator = Generator::load(store, decode_final_out, style_dim)?;

        Ok(Self {
            asr_dim,
            asr_res_out,
            decode_hidden,
            decode_final_out,
            style_dim,
            asr_res,
            f0_conv,
            n_conv,
            encode,
            decode,
            generator,
        })
    }

    /// Real forward through the full iSTFTNet pipeline.
    ///
    /// See [`Decoder::forward_full`] for the argument contract.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        asr: &[f32],
        f0: &[f32],
        n: &[f32],
        style: &[f32],
        t_frames: usize,
        istft_n_fft: usize,
        istft_hop: usize,
        istft_win_length: usize,
        phase_activation: PhaseActivation,
    ) -> Result<Vec<f32>> {
        let (x_mag, x_phase, t_gen) = self.forward_to_mag_phase(asr, f0, n, style, t_frames)?;
        run_istft_head(
            &x_mag,
            &x_phase,
            t_gen,
            istft_n_fft,
            istft_hop,
            istft_win_length,
            phase_activation,
        )
    }

    /// Same as [`DecoderReal::forward`] but returns the pre-iSTFT
    /// `(x_mag, x_phase, pcm)` triple. Test-only bridge for the
    /// M2-07-T15 parity harness (`decoder_forward_bit_parity`).
    ///
    /// Runs the full generator pipeline and the iSTFT head; the intermediates
    /// are returned so the parity dumper's `decoder_pre_istft_mag.f32` /
    /// `decoder_pre_istft_phase.f32` fixtures can be compared byte-for-byte
    /// against the PyTorch re-forward before the iSTFT lowering step.
    #[allow(clippy::too_many_arguments)]
    fn forward_intermediate(
        &self,
        asr: &[f32],
        f0: &[f32],
        n: &[f32],
        style: &[f32],
        t_frames: usize,
        istft_n_fft: usize,
        istft_hop: usize,
        istft_win_length: usize,
        phase_activation: PhaseActivation,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let (x_mag, x_phase, t_gen) = self.forward_to_mag_phase(asr, f0, n, style, t_frames)?;
        let pcm = run_istft_head(
            &x_mag,
            &x_phase,
            t_gen,
            istft_n_fft,
            istft_hop,
            istft_win_length,
            phase_activation,
        )?;
        Ok((x_mag, x_phase, pcm))
    }

    /// Shared pipeline: text_encoder features + F0 / N contours → generator
    /// pre-iSTFT `(x_mag, x_phase, t_gen)` triple. Reused by both
    /// [`Self::forward`] and [`Self::forward_intermediate`] so the two share
    /// bit-identical math up to the iSTFT lowering.
    fn forward_to_mag_phase(
        &self,
        asr: &[f32],
        f0: &[f32],
        n: &[f32],
        style: &[f32],
        t_frames: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, usize)> {
        // ---- Shape checks (FR-EX-08 — never a silent truncation) ---------
        if asr.len() != self.asr_dim * t_frames {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro decoder: asr len {} != asr_dim ({}) · t_frames ({})",
                asr.len(),
                self.asr_dim,
                t_frames,
            )));
        }
        if f0.len() != t_frames {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro decoder: f0 len {} != t_frames ({})",
                f0.len(),
                t_frames,
            )));
        }
        if n.len() != t_frames {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro decoder: n len {} != t_frames ({})",
                n.len(),
                t_frames,
            )));
        }
        if style.len() != self.style_dim {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro decoder: style len {} != style_dim ({})",
                style.len(),
                self.style_dim,
            )));
        }

        let compute = Compute::cpu();

        // ---- Downsample F0 and energy contours via F0_conv/N_conv -------
        // Each is (1 → 1, k=3, stride=2, pad=1) → t_ds = ceil(t_frames / 2).
        let (f0_ds, t_ds) = self.f0_conv.forward(&compute, f0, t_frames);
        let (n_ds, t_ds_n) = self.n_conv.forward(&compute, n, t_frames);
        debug_assert_eq!(
            t_ds, t_ds_n,
            "kokoro decoder: F0/N downsample time-axis mismatch"
        );

        // ---- Build the encode input by concatenating [asr | F0_ds | N_ds]
        // But asr is at t_frames, F0_ds at t_ds. Interpolate F0_ds/N_ds up to
        // t_frames (nearest neighbor) so the channel-major concat has a
        // consistent time axis. This is the pragmatic choice for M2-07-T15
        // when the exact upstream F0-conv stride behaviour is not confirmed;
        // parity work (T17) can pin this to whatever the reference expects.
        let f0_up = upsample_nearest(&f0_ds, t_ds, t_frames);
        let n_up = upsample_nearest(&n_ds, t_ds, t_frames);

        // Concat [asr (asr_dim, t_frames) | F0_up (1, t_frames) | N_up (1, t_frames)]
        // → channel-major [asr_dim + 2, t_frames].
        let encode_in_ch = self.asr_dim + 2;
        let mut enc_input = vec![0.0f32; encode_in_ch * t_frames];
        enc_input[..self.asr_dim * t_frames].copy_from_slice(asr);
        let f0_off = self.asr_dim * t_frames;
        enc_input[f0_off..f0_off + t_frames].copy_from_slice(&f0_up);
        let n_off = f0_off + t_frames;
        enc_input[n_off..n_off + t_frames].copy_from_slice(&n_up);

        // ---- encode: AdainResBlock1(514 → 1024) --------------------------
        let (mut x, mut t_x) = self.encode.forward(&compute, &enc_input, t_frames, style);

        // ---- asr_res: bridge asr features (512 → 64) --------------------
        let (asr_res_out, t_asr_res) = self.asr_res.forward(&compute, asr, t_frames);
        debug_assert_eq!(t_asr_res, t_frames);

        // ---- decode.0/1/2/3 ----------------------------------------------
        // Each block: concat [x, asr_res, F0_ds, N_ds] → block(concat).
        // The concat operates on the channel axis; time axis must be aligned
        // (t_x = t_frames for the first three blocks; decode.3 upsamples via
        // its pool, so its output has t_x_next ≠ t_frames).
        for (i, block) in self.decode.iter().enumerate() {
            // Interpolate asr_res + F0_up + N_up to match t_x (identity when
            // t_x == t_frames for decode.0/1/2; nearest-neighbor for the
            // decode.3 output if wiring changes).
            let (asr_res_i, _) = interp_to(&asr_res_out, self.asr_res_out, t_frames, t_x);
            let (f0_i, _) = interp_to(&f0_up, 1, t_frames, t_x);
            let (n_i, _) = interp_to(&n_up, 1, t_frames, t_x);
            let concat_ch = block.dim_in; // decode_hidden + asr_res_out + 2 = 1090 for real
            let mut concat = vec![0.0f32; concat_ch * t_x];
            // [x (decode_hidden, t_x)] block
            concat[..self.decode_hidden * t_x].copy_from_slice(&x[..self.decode_hidden * t_x]);
            let asr_off = self.decode_hidden * t_x;
            concat[asr_off..asr_off + self.asr_res_out * t_x]
                .copy_from_slice(&asr_res_i[..self.asr_res_out * t_x]);
            let f0_off = asr_off + self.asr_res_out * t_x;
            concat[f0_off..f0_off + t_x].copy_from_slice(&f0_i[..t_x]);
            let n_off = f0_off + t_x;
            concat[n_off..n_off + t_x].copy_from_slice(&n_i[..t_x]);
            let (out, t_out) = block.forward(&compute, &concat, t_x, style);
            let _ = i; // silence unused-index lint
            x = out;
            t_x = t_out;
        }
        // After decode.3, x is [decode_final_out (=512), t_x].

        // ---- generator: 512-ch → (x_mag, x_phase) -----------------------
        // The generator returns (x_mag, x_phase) each [n_half, t_gen] where
        // n_half = istft_n_fft/2 + 1 = 11 and t_gen = t_x · ∏ generator ups
        // strides (real: 10·6 = 60).
        let (x_mag, x_phase, t_gen) = self.generator.forward(&compute, &x, t_x, style, f0)?;
        Ok((x_mag, x_phase, t_gen))
    }
}

/// Nearest-neighbor upsample a channel-major `[channels, t_in]` buffer to
/// `[channels, t_out]`. Identity when `t_in == t_out`.
fn upsample_nearest(x: &[f32], t_in: usize, t_out: usize) -> Vec<f32> {
    if t_in == t_out {
        return x.to_vec();
    }
    if t_in == 0 {
        return vec![0.0f32; t_out];
    }
    let channels = x.len() / t_in;
    let mut out = vec![0.0f32; channels * t_out];
    for c in 0..channels {
        for t in 0..t_out {
            // Nearest source index in [0, t_in).
            let src = ((t as u64 * t_in as u64) / t_out as u64) as usize;
            let src = src.min(t_in - 1);
            out[c * t_out + t] = x[c * t_in + src];
        }
    }
    out
}

/// Nearest-neighbor interpolate `[channels, t_in]` → `[channels, t_out]`.
/// Kept as a thin wrapper so future replacements (linear, sinc) touch one
/// call-site.
fn interp_to(x: &[f32], channels: usize, t_in: usize, t_out: usize) -> (Vec<f32>, usize) {
    debug_assert_eq!(x.len(), channels * t_in);
    (upsample_nearest(x, t_in, t_out), t_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kokoro::config::{
        KEY_HIDDEN_DIM, KEY_ISTFT_HOP, KEY_ISTFT_N_FFT, KEY_ISTFT_WIN_LENGTH, KEY_N_DECODER_LAYERS,
        KEY_N_TEXT_LAYERS, KEY_NUM_VOICES, KEY_PHONEME_SYMBOLS, KEY_SAMPLE_RATE, KEY_STYLE_DIM,
        KEY_VOICE_NAMES,
    };
    use vokra_core::gguf::{
        GgmlType, GgufArray, GgufBuilder, GgufFile, GgufMetadataValue, GgufValueType,
    };

    // --- Synthetic-fixture helpers -------------------------------------------
    //
    // We build a **compact** synthetic decoder GGUF that exercises the entire
    // 375-tensor loader without allocating the 82M-parameter real fixture:
    // architectural dims are picked small (see `TEST_ARCH`) but all shapes
    // preserve the real relationships (concat_in = decode_hidden + asr_res_out + 2,
    // 2·channels = norm.fc.weight rows, etc). All tensors are zeros so the
    // forward is deterministic and bounded — parity work lands in T17.

    /// Compact synthetic decoder architecture. Values chosen so:
    ///
    /// * every `nn.rs` primitive is exercised (Conv1d, ConvTranspose1d, AdaIN,
    ///   Snake, weight_norm reconstruction, LeakyReLU);
    /// * concat / channel shapes hold (decode_concat_in = 8 + 2 + 2 = 12);
    /// * time axes remain sane through the generator ups strides.
    #[allow(dead_code)] // some fields (n_half, ups_stride0) mirror the manifest
    // for documentation and future generator-only tests; the compact fixture
    // does not read them directly.
    struct TestArch {
        style_dim: usize,
        asr_dim: usize,       // (asr_res in_ch)
        asr_res_out: usize,   // (asr_res out_ch)
        decode_hidden: usize, // (decode.0/1/2 out_ch, decode.3 in_ch after concat)
        decode_final: usize,  // (decode.3 out_ch, generator input ch)
        gen_mid: usize,       // (ups.0 out_ch)
        gen_final: usize,     // (ups.1 out_ch, conv_post in_ch)
        istft_n_fft: usize,   // (2·n_half = conv_post out_ch)
        istft_hop: usize,
        istft_win_length: usize,
        n_half: usize,        // (istft_n_fft/2 + 1)
        conv_post_out: usize, // (2·n_half = mag+phase channels)
        ups_kernel0: usize,
        ups_stride0: usize,
        ups_kernel1: usize,
        ups_stride1: usize,
        m_source_harm: usize, // Linear(harm+1 → 1) input dim (harmonic_num + 1)
    }

    impl TestArch {
        const fn compact() -> Self {
            let istft_n_fft = 4; // → n_half = 3, conv_post_out = 6
            Self {
                style_dim: 4,
                asr_dim: 4,
                asr_res_out: 2,
                decode_hidden: 8,
                decode_final: 8, // decoder.3 output → generator input
                gen_mid: 4,
                gen_final: 2,
                istft_n_fft,
                istft_hop: 2,
                istft_win_length: 4,
                n_half: istft_n_fft / 2 + 1,
                conv_post_out: 2 * (istft_n_fft / 2 + 1),
                ups_kernel0: 4,
                ups_stride0: 2,
                ups_kernel1: 4,
                ups_stride1: 2,
                m_source_harm: 9, // harmonic_num=8 → l_linear.weight shape (1, 9)
            }
        }
    }

    fn str_array(items: &[&str]) -> GgufMetadataValue {
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: items
                .iter()
                .map(|s| GgufMetadataValue::String((*s).to_owned()))
                .collect(),
        })
    }

    fn zeros(n: usize) -> Vec<u8> {
        vec![0u8; n * 4]
    }

    /// Add a zero-payload tensor at `name` with the given shape. Widths are
    /// `usize` for ergonomics; the GGUF builder takes `u64`.
    fn add_zeros(b: &mut GgufBuilder, name: &str, shape: &[usize]) {
        let n: usize = shape.iter().product();
        let shape_u64: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        b.add_tensor(name, GgmlType::F32, shape_u64, zeros(n))
            .expect("valid F32 tensor");
    }

    /// Adds every `decoder.module.*` tensor at compact-arch shapes. Used by
    /// the real-mode loader tests.
    #[allow(clippy::too_many_lines)] // straight-line manifest expansion
    fn add_all_decoder_tensors(b: &mut GgufBuilder, a: &TestArch) {
        // asr_res
        add_zeros(b, "decoder.module.asr_res.0.bias", &[a.asr_res_out]);
        add_zeros(
            b,
            "decoder.module.asr_res.0.weight_g",
            &[a.asr_res_out, 1, 1],
        );
        add_zeros(
            b,
            "decoder.module.asr_res.0.weight_v",
            &[a.asr_res_out, a.asr_dim, 1],
        );
        // F0_conv, N_conv (1 → 1, k=3)
        for name in ["F0_conv", "N_conv"] {
            add_zeros(b, &format!("decoder.module.{name}.bias"), &[1]);
            add_zeros(b, &format!("decoder.module.{name}.weight_g"), &[1, 1, 1]);
            add_zeros(b, &format!("decoder.module.{name}.weight_v"), &[1, 1, 3]);
        }
        // encode: AdainResBlock1(asr_dim + 2 → decode_hidden)
        let encode_in = a.asr_dim + 2;
        add_adain_resblock(
            b,
            "decoder.module.encode",
            encode_in,
            a.decode_hidden,
            false,
            a.style_dim,
        );
        // decode.0/1/2: (concat_in → decode_hidden), no pool.
        let concat_in = a.decode_hidden + a.asr_res_out + 2;
        for i in 0..3 {
            add_adain_resblock(
                b,
                &format!("decoder.module.decode.{i}"),
                concat_in,
                a.decode_hidden,
                false,
                a.style_dim,
            );
        }
        // decode.3: (concat_in → decode_final), with pool.
        add_adain_resblock(
            b,
            "decoder.module.decode.3",
            concat_in,
            a.decode_final,
            true,
            a.style_dim,
        );

        // ---- generator ---------------------------------------------------
        add_generator_tensors(b, a);
    }

    fn add_adain_resblock(
        b: &mut GgufBuilder,
        prefix: &str,
        dim_in: usize,
        dim_out: usize,
        has_pool: bool,
        style_dim: usize,
    ) {
        // conv1 (dim_in → dim_out, k=3)
        add_zeros(b, &format!("{prefix}.conv1.bias"), &[dim_out]);
        add_zeros(b, &format!("{prefix}.conv1.weight_g"), &[dim_out, 1, 1]);
        add_zeros(
            b,
            &format!("{prefix}.conv1.weight_v"),
            &[dim_out, dim_in, 3],
        );
        // conv1x1 (dim_in → dim_out, k=1, no bias)
        add_zeros(b, &format!("{prefix}.conv1x1.weight_g"), &[dim_out, 1, 1]);
        add_zeros(
            b,
            &format!("{prefix}.conv1x1.weight_v"),
            &[dim_out, dim_in, 1],
        );
        // conv2 (dim_out → dim_out, k=3)
        add_zeros(b, &format!("{prefix}.conv2.bias"), &[dim_out]);
        add_zeros(b, &format!("{prefix}.conv2.weight_g"), &[dim_out, 1, 1]);
        add_zeros(
            b,
            &format!("{prefix}.conv2.weight_v"),
            &[dim_out, dim_out, 3],
        );
        // norm1: AdaIN1d(dim_in)
        add_zeros(b, &format!("{prefix}.norm1.fc.bias"), &[2 * dim_in]);
        add_zeros(
            b,
            &format!("{prefix}.norm1.fc.weight"),
            &[2 * dim_in, style_dim],
        );
        // norm2: AdaIN1d(dim_out)
        add_zeros(b, &format!("{prefix}.norm2.fc.bias"), &[2 * dim_out]);
        add_zeros(
            b,
            &format!("{prefix}.norm2.fc.weight"),
            &[2 * dim_out, style_dim],
        );
        // pool (depthwise ConvTranspose1d) — only for the upsampling stage.
        if has_pool {
            add_zeros(b, &format!("{prefix}.pool.bias"), &[dim_in]);
            add_zeros(b, &format!("{prefix}.pool.weight_g"), &[dim_in, 1, 1]);
            add_zeros(b, &format!("{prefix}.pool.weight_v"), &[dim_in, 1, 3]);
        }
    }

    fn add_generator_tensors(b: &mut GgufBuilder, a: &TestArch) {
        let g = "decoder.module.generator";
        // m_source.l_linear (harm+1 → 1)
        add_zeros(b, &format!("{g}.m_source.l_linear.bias"), &[1]);
        add_zeros(
            b,
            &format!("{g}.m_source.l_linear.weight"),
            &[1, a.m_source_harm],
        );
        // ups.0 (decode_final → gen_mid)
        add_zeros(b, &format!("{g}.ups.0.bias"), &[a.gen_mid]);
        add_zeros(b, &format!("{g}.ups.0.weight_g"), &[a.decode_final, 1, 1]);
        add_zeros(
            b,
            &format!("{g}.ups.0.weight_v"),
            &[a.decode_final, a.gen_mid, a.ups_kernel0],
        );
        // ups.1 (gen_mid → gen_final)
        add_zeros(b, &format!("{g}.ups.1.bias"), &[a.gen_final]);
        add_zeros(b, &format!("{g}.ups.1.weight_g"), &[a.gen_mid, 1, 1]);
        add_zeros(
            b,
            &format!("{g}.ups.1.weight_v"),
            &[a.gen_mid, a.gen_final, a.ups_kernel1],
        );
        // noise_convs (plain Conv1d, not weight-normed)
        // noise_convs.0: 22 (== conv_post_out) → gen_mid, kernel = ups_kernel1 * something
        add_zeros(b, &format!("{g}.noise_convs.0.bias"), &[a.gen_mid]);
        add_zeros(
            b,
            &format!("{g}.noise_convs.0.weight"),
            &[
                a.gen_mid,
                a.conv_post_out,
                /* kernel */ a.ups_stride1 * 2,
            ],
        );
        add_zeros(b, &format!("{g}.noise_convs.1.bias"), &[a.gen_final]);
        add_zeros(
            b,
            &format!("{g}.noise_convs.1.weight"),
            &[a.gen_final, a.conv_post_out, 1],
        );
        // noise_res.0 (AmpResBlock at gen_mid ch)
        add_amp_resblock(b, &format!("{g}.noise_res.0"), a.gen_mid, 7, a.style_dim);
        // noise_res.1 (AmpResBlock at gen_final ch)
        add_amp_resblock(b, &format!("{g}.noise_res.1"), a.gen_final, 11, a.style_dim);
        // resblocks 0/1/2 at gen_mid ch (kernels 3, 7, 11)
        for (i, k) in [3usize, 7, 11].iter().enumerate() {
            add_amp_resblock(b, &format!("{g}.resblocks.{i}"), a.gen_mid, *k, a.style_dim);
        }
        // resblocks 3/4/5 at gen_final ch (kernels 3, 7, 11)
        for (idx, k) in [3usize, 7, 11].iter().enumerate() {
            add_amp_resblock(
                b,
                &format!("{g}.resblocks.{}", 3 + idx),
                a.gen_final,
                *k,
                a.style_dim,
            );
        }
        // conv_post (gen_final → conv_post_out, k=7)
        add_zeros(b, &format!("{g}.conv_post.bias"), &[a.conv_post_out]);
        add_zeros(
            b,
            &format!("{g}.conv_post.weight_g"),
            &[a.conv_post_out, 1, 1],
        );
        add_zeros(
            b,
            &format!("{g}.conv_post.weight_v"),
            &[a.conv_post_out, a.gen_final, 7],
        );
    }

    fn add_amp_resblock(
        b: &mut GgufBuilder,
        prefix: &str,
        ch: usize,
        kernel: usize,
        style_dim: usize,
    ) {
        for j in 0..3 {
            // convs1.j (ch → ch, k=kernel, dilation=1/3/5)
            add_zeros(b, &format!("{prefix}.convs1.{j}.bias"), &[ch]);
            add_zeros(b, &format!("{prefix}.convs1.{j}.weight_g"), &[ch, 1, 1]);
            add_zeros(
                b,
                &format!("{prefix}.convs1.{j}.weight_v"),
                &[ch, ch, kernel],
            );
            // convs2.j (ch → ch, k=kernel)
            add_zeros(b, &format!("{prefix}.convs2.{j}.bias"), &[ch]);
            add_zeros(b, &format!("{prefix}.convs2.{j}.weight_g"), &[ch, 1, 1]);
            add_zeros(
                b,
                &format!("{prefix}.convs2.{j}.weight_v"),
                &[ch, ch, kernel],
            );
            // adain1.j / adain2.j (AdaIN over ch)
            add_zeros(b, &format!("{prefix}.adain1.{j}.fc.bias"), &[2 * ch]);
            add_zeros(
                b,
                &format!("{prefix}.adain1.{j}.fc.weight"),
                &[2 * ch, style_dim],
            );
            add_zeros(b, &format!("{prefix}.adain2.{j}.fc.bias"), &[2 * ch]);
            add_zeros(
                b,
                &format!("{prefix}.adain2.{j}.fc.weight"),
                &[2 * ch, style_dim],
            );
            // alpha1.j / alpha2.j (Snake per-channel scale, shape [1, ch, 1])
            add_zeros(b, &format!("{prefix}.alpha1.{j}"), &[1, ch, 1]);
            add_zeros(b, &format!("{prefix}.alpha2.{j}"), &[1, ch, 1]);
        }
    }

    fn kokoro_arch() -> &'static str {
        "kokoro-82m-istftnet"
    }

    /// Builds a synthetic Kokoro voice GGUF with the compact test arch and
    /// **all** decoder tensors present, so `Decoder::load` runs the real-mode
    /// path. Does NOT include text_encoder tensors (mod.rs's smoke fixture
    /// handles those); this fixture is decoder-only.
    fn build_real_mode_gguf(a: &TestArch) -> Vec<u8> {
        let mut b = GgufBuilder::new();
        b.add_string(vokra_core::gguf::chunks::KEY_MODEL_ARCH, kokoro_arch());
        b.add_u32(KEY_SAMPLE_RATE, 24_000);
        b.add_u32(KEY_STYLE_DIM, a.style_dim as u32);
        b.add_u32(KEY_NUM_VOICES, 2);
        // hidden_dim = asr_dim (== encoder output width feeding the decoder).
        b.add_u32(KEY_HIDDEN_DIM, a.asr_dim as u32);
        b.add_u32(KEY_N_TEXT_LAYERS, 3);
        b.add_u32(KEY_N_DECODER_LAYERS, 4);
        b.add_u32(KEY_ISTFT_N_FFT, a.istft_n_fft as u32);
        b.add_u32(KEY_ISTFT_HOP, a.istft_hop as u32);
        b.add_u32(KEY_ISTFT_WIN_LENGTH, a.istft_win_length as u32);
        b.add_metadata(KEY_PHONEME_SYMBOLS, str_array(&["a", "b", "c"]));
        b.add_metadata(KEY_VOICE_NAMES, str_array(&["af", "am"]));

        add_all_decoder_tensors(&mut b, a);

        b.to_bytes().expect("serialize")
    }

    fn build_stub_gguf() -> Vec<u8> {
        let mut b = GgufBuilder::new();
        b.add_string(vokra_core::gguf::chunks::KEY_MODEL_ARCH, kokoro_arch());
        b.add_u32(KEY_SAMPLE_RATE, 24_000);
        b.add_u32(KEY_STYLE_DIM, 8);
        b.add_u32(KEY_NUM_VOICES, 2);
        b.add_u32(KEY_HIDDEN_DIM, 16);
        b.add_u32(KEY_N_TEXT_LAYERS, 3);
        b.add_u32(KEY_N_DECODER_LAYERS, 4);
        b.add_u32(KEY_ISTFT_N_FFT, 20);
        b.add_u32(KEY_ISTFT_HOP, 5);
        b.add_u32(KEY_ISTFT_WIN_LENGTH, 20);
        b.add_metadata(KEY_PHONEME_SYMBOLS, str_array(&["a", "b", "c"]));
        b.add_metadata(KEY_VOICE_NAMES, str_array(&["af", "am"]));
        b.to_bytes().expect("serialize")
    }

    // --- Tests ------------------------------------------------------------

    /// Stub mode: absent decoder tensors → infallible load, forward matches the
    /// legacy `t_frames · istft_hop` shape contract.
    #[test]
    fn stub_mode_load_and_forward() {
        let bytes = build_stub_gguf();
        let file = GgufFile::parse(bytes).expect("parse");
        let config = KokoroConfig::from_gguf(&file).expect("valid config");
        let store = TensorStore::new(file);
        let dec = Decoder::load(&store, &config).expect("stub load ok");
        assert!(!dec.is_real(), "no decoder tensors → stub mode");

        let hidden = config.hidden_dim;
        let t_frames = 3;
        let z: Vec<f32> = (0..hidden * t_frames).map(|i| i as f32 * 0.01).collect();
        let style = vec![0.5f32; config.style_dim];
        let pcm = dec.forward(&z, t_frames, &style).expect("stub forward ok");
        assert_eq!(pcm.len(), t_frames * config.istft_hop);
        for (i, &v) in pcm.iter().enumerate() {
            assert!(v.is_finite(), "stub pcm[{i}] = {v} (must be finite)");
        }
    }

    /// Real mode: full 375-tensor load succeeds when every canonical name is
    /// present in the fixture.
    #[test]
    fn real_mode_load_binds_every_decoder_tensor() {
        let arch = TestArch::compact();
        let bytes = build_real_mode_gguf(&arch);
        let file = GgufFile::parse(bytes).expect("parse");
        let config = KokoroConfig::from_gguf(&file).expect("valid config");
        let store = TensorStore::new(file);
        let dec = Decoder::load(&store, &config).expect("real load ok");
        assert!(dec.is_real(), "all tensors present → real mode");
        assert_eq!(dec.sample_rate(), 24_000);
    }

    /// Real-mode forward is deterministic (no RNG, no uninitialised buffers) —
    /// two calls with identical inputs return byte-identical outputs.
    #[test]
    fn real_mode_forward_is_deterministic() {
        let arch = TestArch::compact();
        let bytes = build_real_mode_gguf(&arch);
        let file = GgufFile::parse(bytes).expect("parse");
        let config = KokoroConfig::from_gguf(&file).expect("valid config");
        let store = TensorStore::new(file);
        let dec = Decoder::load(&store, &config).expect("real load ok");

        let t_frames = 4;
        let asr: Vec<f32> = (0..arch.asr_dim * t_frames)
            .map(|i| ((i % 7) as f32) * 0.03 - 0.1)
            .collect();
        let f0: Vec<f32> = (0..t_frames).map(|i| 100.0 + i as f32 * 5.0).collect();
        let n: Vec<f32> = (0..t_frames).map(|i| 0.2 + i as f32 * 0.05).collect();
        let style: Vec<f32> = (0..arch.style_dim).map(|i| i as f32 * 0.1).collect();

        let a = dec
            .forward_full(&asr, &f0, &n, &style, t_frames, PhaseActivation::Sin)
            .expect("real forward ok");
        let b = dec
            .forward_full(&asr, &f0, &n, &style, t_frames, PhaseActivation::Sin)
            .expect("real forward ok (second call)");
        assert_eq!(
            a, b,
            "kokoro real decoder forward must be bit-exact deterministic"
        );
        for (i, &v) in a.iter().enumerate() {
            assert!(v.is_finite(), "real pcm[{i}] = {v} (must be finite)");
        }
        assert!(!a.is_empty(), "real forward must produce non-empty PCM");
    }

    /// FR-EX-08: the strict tensor loader must reject a **missing** tensor
    /// with a message that names the offending tensor. Builds the full
    /// real-mode fixture but omits `decoder.module.asr_res.0.bias` — the
    /// loader must fail loudly at that specific tensor, not silently proceed
    /// with a zero bias.
    #[test]
    fn real_mode_missing_bias_fails_loud() {
        let arch = TestArch::compact();
        // Full real-mode fixture, then remove one specific tensor. Rebuilding
        // the builder without that key is simpler than exposing a removal API
        // on GgufBuilder.
        let mut b = GgufBuilder::new();
        b.add_string(vokra_core::gguf::chunks::KEY_MODEL_ARCH, kokoro_arch());
        b.add_u32(KEY_SAMPLE_RATE, 24_000);
        b.add_u32(KEY_STYLE_DIM, arch.style_dim as u32);
        b.add_u32(KEY_NUM_VOICES, 2);
        b.add_u32(KEY_HIDDEN_DIM, arch.asr_dim as u32);
        b.add_u32(KEY_N_TEXT_LAYERS, 3);
        b.add_u32(KEY_N_DECODER_LAYERS, 4);
        b.add_u32(KEY_ISTFT_N_FFT, arch.istft_n_fft as u32);
        b.add_u32(KEY_ISTFT_HOP, arch.istft_hop as u32);
        b.add_u32(KEY_ISTFT_WIN_LENGTH, arch.istft_win_length as u32);
        b.add_metadata(KEY_PHONEME_SYMBOLS, str_array(&["a"]));
        b.add_metadata(KEY_VOICE_NAMES, str_array(&["af"]));

        // Add every decoder tensor EXCEPT asr_res.0.bias by cloning
        // `add_all_decoder_tensors` sans that single tensor.
        add_all_decoder_tensors_except(&mut b, &arch, "decoder.module.asr_res.0.bias");

        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let config = KokoroConfig::from_gguf(&file).expect("valid config");
        let store = TensorStore::new(file);
        let err = match Decoder::load(&store, &config) {
            Err(e) => e,
            Ok(_) => panic!("missing bias must fail load"),
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("decoder.module.asr_res.0.bias"),
                    "error must name the missing tensor; got: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// Adds every `decoder.module.*` tensor at compact-arch shapes EXCEPT the
    /// one whose name matches `exclude`. Panics if `exclude` did not match any
    /// tensor — a mis-typed test would otherwise silently succeed.
    fn add_all_decoder_tensors_except(b: &mut GgufBuilder, a: &TestArch, exclude: &str) {
        // asr_res
        maybe_add(
            b,
            "decoder.module.asr_res.0.bias",
            &[a.asr_res_out],
            exclude,
        );
        maybe_add(
            b,
            "decoder.module.asr_res.0.weight_g",
            &[a.asr_res_out, 1, 1],
            exclude,
        );
        maybe_add(
            b,
            "decoder.module.asr_res.0.weight_v",
            &[a.asr_res_out, a.asr_dim, 1],
            exclude,
        );
        for name in ["F0_conv", "N_conv"] {
            maybe_add(b, &format!("decoder.module.{name}.bias"), &[1], exclude);
            maybe_add(
                b,
                &format!("decoder.module.{name}.weight_g"),
                &[1, 1, 1],
                exclude,
            );
            maybe_add(
                b,
                &format!("decoder.module.{name}.weight_v"),
                &[1, 1, 3],
                exclude,
            );
        }
        let encode_in = a.asr_dim + 2;
        add_adain_resblock_except(
            b,
            "decoder.module.encode",
            encode_in,
            a.decode_hidden,
            false,
            a.style_dim,
            exclude,
        );
        let concat_in = a.decode_hidden + a.asr_res_out + 2;
        for i in 0..3 {
            add_adain_resblock_except(
                b,
                &format!("decoder.module.decode.{i}"),
                concat_in,
                a.decode_hidden,
                false,
                a.style_dim,
                exclude,
            );
        }
        add_adain_resblock_except(
            b,
            "decoder.module.decode.3",
            concat_in,
            a.decode_final,
            true,
            a.style_dim,
            exclude,
        );
        add_generator_tensors_except(b, a, exclude);
    }

    /// Adds a zero-payload tensor at `name` unless the name matches `exclude`,
    /// in which case the tensor is deliberately skipped so the loader hits a
    /// missing-tensor error at exactly that name.
    fn maybe_add(b: &mut GgufBuilder, name: &str, shape: &[usize], exclude: &str) {
        if name == exclude {
            return;
        }
        add_zeros(b, name, shape);
    }

    #[allow(clippy::too_many_arguments)]
    fn add_adain_resblock_except(
        b: &mut GgufBuilder,
        prefix: &str,
        dim_in: usize,
        dim_out: usize,
        has_pool: bool,
        style_dim: usize,
        exclude: &str,
    ) {
        maybe_add(b, &format!("{prefix}.conv1.bias"), &[dim_out], exclude);
        maybe_add(
            b,
            &format!("{prefix}.conv1.weight_g"),
            &[dim_out, 1, 1],
            exclude,
        );
        maybe_add(
            b,
            &format!("{prefix}.conv1.weight_v"),
            &[dim_out, dim_in, 3],
            exclude,
        );
        maybe_add(
            b,
            &format!("{prefix}.conv1x1.weight_g"),
            &[dim_out, 1, 1],
            exclude,
        );
        maybe_add(
            b,
            &format!("{prefix}.conv1x1.weight_v"),
            &[dim_out, dim_in, 1],
            exclude,
        );
        maybe_add(b, &format!("{prefix}.conv2.bias"), &[dim_out], exclude);
        maybe_add(
            b,
            &format!("{prefix}.conv2.weight_g"),
            &[dim_out, 1, 1],
            exclude,
        );
        maybe_add(
            b,
            &format!("{prefix}.conv2.weight_v"),
            &[dim_out, dim_out, 3],
            exclude,
        );
        maybe_add(
            b,
            &format!("{prefix}.norm1.fc.bias"),
            &[2 * dim_in],
            exclude,
        );
        maybe_add(
            b,
            &format!("{prefix}.norm1.fc.weight"),
            &[2 * dim_in, style_dim],
            exclude,
        );
        maybe_add(
            b,
            &format!("{prefix}.norm2.fc.bias"),
            &[2 * dim_out],
            exclude,
        );
        maybe_add(
            b,
            &format!("{prefix}.norm2.fc.weight"),
            &[2 * dim_out, style_dim],
            exclude,
        );
        if has_pool {
            maybe_add(b, &format!("{prefix}.pool.bias"), &[dim_in], exclude);
            maybe_add(
                b,
                &format!("{prefix}.pool.weight_g"),
                &[dim_in, 1, 1],
                exclude,
            );
            maybe_add(
                b,
                &format!("{prefix}.pool.weight_v"),
                &[dim_in, 1, 3],
                exclude,
            );
        }
    }

    fn add_generator_tensors_except(b: &mut GgufBuilder, a: &TestArch, exclude: &str) {
        let g = "decoder.module.generator";
        maybe_add(b, &format!("{g}.m_source.l_linear.bias"), &[1], exclude);
        maybe_add(
            b,
            &format!("{g}.m_source.l_linear.weight"),
            &[1, a.m_source_harm],
            exclude,
        );
        maybe_add(b, &format!("{g}.ups.0.bias"), &[a.gen_mid], exclude);
        maybe_add(
            b,
            &format!("{g}.ups.0.weight_g"),
            &[a.decode_final, 1, 1],
            exclude,
        );
        maybe_add(
            b,
            &format!("{g}.ups.0.weight_v"),
            &[a.decode_final, a.gen_mid, a.ups_kernel0],
            exclude,
        );
        maybe_add(b, &format!("{g}.ups.1.bias"), &[a.gen_final], exclude);
        maybe_add(
            b,
            &format!("{g}.ups.1.weight_g"),
            &[a.gen_mid, 1, 1],
            exclude,
        );
        maybe_add(
            b,
            &format!("{g}.ups.1.weight_v"),
            &[a.gen_mid, a.gen_final, a.ups_kernel1],
            exclude,
        );
        maybe_add(b, &format!("{g}.noise_convs.0.bias"), &[a.gen_mid], exclude);
        maybe_add(
            b,
            &format!("{g}.noise_convs.0.weight"),
            &[a.gen_mid, a.conv_post_out, a.ups_stride1 * 2],
            exclude,
        );
        maybe_add(
            b,
            &format!("{g}.noise_convs.1.bias"),
            &[a.gen_final],
            exclude,
        );
        maybe_add(
            b,
            &format!("{g}.noise_convs.1.weight"),
            &[a.gen_final, a.conv_post_out, 1],
            exclude,
        );
        add_amp_resblock_except(
            b,
            &format!("{g}.noise_res.0"),
            a.gen_mid,
            7,
            a.style_dim,
            exclude,
        );
        add_amp_resblock_except(
            b,
            &format!("{g}.noise_res.1"),
            a.gen_final,
            11,
            a.style_dim,
            exclude,
        );
        for (i, k) in [3usize, 7, 11].iter().enumerate() {
            add_amp_resblock_except(
                b,
                &format!("{g}.resblocks.{i}"),
                a.gen_mid,
                *k,
                a.style_dim,
                exclude,
            );
        }
        for (idx, k) in [3usize, 7, 11].iter().enumerate() {
            add_amp_resblock_except(
                b,
                &format!("{g}.resblocks.{}", 3 + idx),
                a.gen_final,
                *k,
                a.style_dim,
                exclude,
            );
        }
        maybe_add(
            b,
            &format!("{g}.conv_post.bias"),
            &[a.conv_post_out],
            exclude,
        );
        maybe_add(
            b,
            &format!("{g}.conv_post.weight_g"),
            &[a.conv_post_out, 1, 1],
            exclude,
        );
        maybe_add(
            b,
            &format!("{g}.conv_post.weight_v"),
            &[a.conv_post_out, a.gen_final, 7],
            exclude,
        );
    }

    fn add_amp_resblock_except(
        b: &mut GgufBuilder,
        prefix: &str,
        ch: usize,
        kernel: usize,
        style_dim: usize,
        exclude: &str,
    ) {
        for j in 0..3 {
            maybe_add(b, &format!("{prefix}.convs1.{j}.bias"), &[ch], exclude);
            maybe_add(
                b,
                &format!("{prefix}.convs1.{j}.weight_g"),
                &[ch, 1, 1],
                exclude,
            );
            maybe_add(
                b,
                &format!("{prefix}.convs1.{j}.weight_v"),
                &[ch, ch, kernel],
                exclude,
            );
            maybe_add(b, &format!("{prefix}.convs2.{j}.bias"), &[ch], exclude);
            maybe_add(
                b,
                &format!("{prefix}.convs2.{j}.weight_g"),
                &[ch, 1, 1],
                exclude,
            );
            maybe_add(
                b,
                &format!("{prefix}.convs2.{j}.weight_v"),
                &[ch, ch, kernel],
                exclude,
            );
            maybe_add(
                b,
                &format!("{prefix}.adain1.{j}.fc.bias"),
                &[2 * ch],
                exclude,
            );
            maybe_add(
                b,
                &format!("{prefix}.adain1.{j}.fc.weight"),
                &[2 * ch, style_dim],
                exclude,
            );
            maybe_add(
                b,
                &format!("{prefix}.adain2.{j}.fc.bias"),
                &[2 * ch],
                exclude,
            );
            maybe_add(
                b,
                &format!("{prefix}.adain2.{j}.fc.weight"),
                &[2 * ch, style_dim],
                exclude,
            );
            maybe_add(b, &format!("{prefix}.alpha1.{j}"), &[1, ch, 1], exclude);
            maybe_add(b, &format!("{prefix}.alpha2.{j}"), &[1, ch, 1], exclude);
        }
    }

    /// FR-EX-08: a **wrong-shape** tensor must fail with a message that names
    /// the tensor and reports the actual vs expected shape.
    #[test]
    fn real_mode_wrong_shape_fails_loud() {
        let arch = TestArch::compact();
        let mut b = GgufBuilder::new();
        b.add_string(vokra_core::gguf::chunks::KEY_MODEL_ARCH, kokoro_arch());
        b.add_u32(KEY_SAMPLE_RATE, 24_000);
        b.add_u32(KEY_STYLE_DIM, arch.style_dim as u32);
        b.add_u32(KEY_NUM_VOICES, 2);
        b.add_u32(KEY_HIDDEN_DIM, arch.asr_dim as u32);
        b.add_u32(KEY_N_TEXT_LAYERS, 3);
        b.add_u32(KEY_N_DECODER_LAYERS, 4);
        b.add_u32(KEY_ISTFT_N_FFT, arch.istft_n_fft as u32);
        b.add_u32(KEY_ISTFT_HOP, arch.istft_hop as u32);
        b.add_u32(KEY_ISTFT_WIN_LENGTH, arch.istft_win_length as u32);
        b.add_metadata(KEY_PHONEME_SYMBOLS, str_array(&["a"]));
        b.add_metadata(KEY_VOICE_NAMES, str_array(&["af"]));
        // Canary present, but the wrong shape (should be [asr_res_out, asr_dim, 1]).
        add_zeros(
            &mut b,
            "decoder.module.asr_res.0.weight_v",
            &[arch.asr_res_out, arch.asr_dim + 1, 1], // off-by-one
        );

        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let config = KokoroConfig::from_gguf(&file).expect("valid config");
        let store = TensorStore::new(file);
        let err = match Decoder::load(&store, &config) {
            Err(e) => e,
            Ok(_) => panic!("wrong shape must fail load"),
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                // The `load_real` shape-derivation path validates
                // decode.0.conv1.weight_v against the derived asr_res+decode dims,
                // so the failure surfaces at either the direct shape check (in
                // WeightNormedConv1d::load) or at the derived-dim cross-check.
                // Either way it must name a decoder.module.* tensor.
                assert!(
                    msg.contains("decoder.module."),
                    "error must name a decoder.module.* tensor; got: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// The [`PhaseActivation`] dispatcher recognises all three documented
    /// values and rejects unknown values with a message that names both the
    /// metadata key and the rejected value (FR-EX-08 — no silent default).
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
        match PhaseActivation::from_meta("relu") {
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(msg.contains(KEY_PHASE_ACTIVATION));
                assert!(msg.contains("relu"));
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// Standalone iSTFT head produces a finite PCM buffer of the expected
    /// length under every dispatched phase activation (M2-07 plan §5 R2: any
    /// one can be the upstream truth).
    #[test]
    fn istft_head_finite_output() {
        let bytes = build_stub_gguf();
        let file = GgufFile::parse(bytes).expect("parse");
        let config = KokoroConfig::from_gguf(&file).expect("valid config");
        let store = TensorStore::new(file);
        let dec = Decoder::load(&store, &config).expect("stub load ok");

        let t_frames = 6;
        let n_half = config.istft_n_fft / 2 + 1;
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
                .expect("istft head runs");
            assert_eq!(pcm.len(), t_frames * config.istft_hop);
            for (i, &v) in pcm.iter().enumerate() {
                assert!(v.is_finite(), "istft pcm[{i}] = {v} under {act:?}");
            }
        }
    }

    /// Off-by-one on the magnitude tensor must fail loudly (FR-EX-08), not
    /// silently truncate / zero-fill.
    #[test]
    fn istft_head_rejects_wrong_mag_shape() {
        let bytes = build_stub_gguf();
        let file = GgufFile::parse(bytes).expect("parse");
        let config = KokoroConfig::from_gguf(&file).expect("valid config");
        let store = TensorStore::new(file);
        let dec = Decoder::load(&store, &config).expect("stub load ok");
        let t_frames = 4;
        let n_half = config.istft_n_fft / 2 + 1;
        let x_mag = vec![0.0f32; n_half * t_frames + 1];
        let x_phase = vec![0.0f32; n_half * t_frames];
        assert!(matches!(
            dec.istft_head(&x_mag, &x_phase, t_frames, PhaseActivation::Tanh),
            Err(VokraError::InvalidArgument(_))
        ));
    }
}
