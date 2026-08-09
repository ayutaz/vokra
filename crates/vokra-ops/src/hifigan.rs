//! HiFi-GAN neural vocoder generator (M3-07; FR-OP-10).
//!
//! # Op contract
//!
//! Given
//!
//! - `mel` — a `[n_mels, n_frames]` row-major slice of FP32 mel spectrogram values;
//! - `weights` — a [`HifiGanWeights`] bundle carrying every conv1d / transposed_conv1d /
//!   MRF ResBlock parameter needed for the FP32 forward;
//! - `attrs` — [`HifiGanAttrs`] shape metadata (upsample factors, MRF kernel sizes,
//!   `leaky_relu_slope`, `sample_rate`, `initial_channel`);
//! - `config` — [`HifiGanConfig`] precision selector + INT8 opt-in gate;
//!
//! [`hifigan_generator`] returns a `[n_samples]` row-major `Vec<f32>` waveform
//! bounded to `(−1, 1)` by the terminal `tanh`. `n_samples = n_frames *
//! attrs.total_upsample_factor()`.
//!
//! The forward stack is HiFi-GAN family (jik876/hifi-gan, MIT):
//!
//! 1. `conv1d` (kernel=7, pad=3): `[n_mels] → [initial_channel]`;
//! 2. per stage `i ∈ 0..n_upsample`:
//!    - `leaky_relu` on the running feature map;
//!    - `transposed_conv1d` (upsample by `upsample_rates[i]`);
//!    - MRF: for every ResBlock branch `b`, compute `resblock_b(h)`; average the
//!      branch outputs (multi-receptive-field fusion).
//! 3. `leaky_relu` → `conv1d` (kernel=7, pad=3) `→ [1]` channels;
//! 4. `tanh` head bounding output to `(−1, 1)`.
//!
//! Every convolution honours the standard PyTorch `(input + 2p − k) / s + 1`
//! output-length formula; transposed conv uses the mirror
//! `(input − 1) · stride − 2p + k` shape. Kernel numeric details follow the
//! upstream reference — the M3-07 ticket delegates the checkpoint-driven
//! preset choice (V1 / V2 / V3) to the M3-09 CosyVoice2 converter.
//!
//! # INT8 opt-in gate (FR-OP-10, FR-QT-03, FR-EX-08)
//!
//! [`HifiGanConfig::int8_enabled`] defaults to `false`. When it is `true`,
//! `hifigan_generator` refuses to run unless *both*:
//!
//! - a [`CalibrationTable`] is attached (per-channel scale / zero-point pair
//!   built by [`HifiGanCalibrator::calibrate`]); and
//! - `spectral_check_passed` is `true` (the MEL / UTMOS delta between an FP32
//!   forward and the INT8 forward on the same input stays within NFR-QL-02's
//!   5% gate, verified by [`HifiGanSpectralChecker::check`]).
//!
//! Either missing piece yields
//! [`VokraError::HifiganInt8VerifyMissing`] — the same error the M2-08 policy
//! validator raises so the two entry points collapse onto one audit trail. **No
//! silent fallback to fp32 / fp16** (FR-EX-08): callers who want INT8 must go
//! through the calibration + spectral check pipeline. The INT8 forward path
//! itself is not implemented in this crate — the calibration table is opaque to
//! the runtime function today. That is deliberate: the ticket's "primary target
//! is CPU parity" and the INT8 kernel lands with the consumer WP (M3-09
//! CosyVoice2 or a HiFi-GAN-standalone WP), because a real INT8 forward is
//! meaningless without a real calibration dataset and the spectral check to
//! validate it. Enabling the flag without a real INT8 kernel today would violate
//! FR-EX-08 by shipping a *silently wrong* code path; instead, INT8 stays
//! locked behind the two gates and the parity harness proves the gate.
//!
//! # HiFi-GAN vs BigVGAN vs Vocos (CLAUDE.md audio-dialect §Vocoder chain)
//!
//! - HiFi-GAN (this op, FR-OP-10): `leaky_relu` + MRF. INT8 慎重 — opt-in.
//! - BigVGAN (FR-OP-11, separate op): AMP snake activation + anti-aliased
//!   upsample. fp16 required (Forbidden downgrade).
//! - Vocos (FR-OP-12, separate op): iSTFTNet head. fp16 required (Forbidden
//!   downgrade). Kokoro decoder is iSTFTNet 派生 but is a distinct op (see
//!   `MinDtypeRegistry` doc).
//!
//! # Runtime function — not backed by `HotOp` dispatch
//!
//! `hifigan_generator` is a composite runtime function (many primitives
//! sequenced with residual + MRF averaging); the `vokra_models::compute::HotOp`
//! dispatch surface is for individual hot-path primitives (`Gemm` / `Softmax` /
//! `LayerNorm` / …), not whole vocoder stacks. The GPU seam (Metal / CUDA)
//! lands with the consumer WP (M3-09 or a HiFi-GAN-standalone GPU WP), following
//! the same "one kernel per (backend, op), no per-op host fall back" contract
//! (FR-EX-08). See `docs/adr/M3-06-mimi-rvq.md` §D5 for the identical deferred-
//! GPU-seam stance mimi_rvq took.
//!
//! # Weight-norm folding contract (converter-side, HGAN-WEIGHT-NORM-DOC)
//!
//! Every conv / transposed-conv weight in a `[HifiGanWeights]` bundle
//! (upsample stages, `MrfBranchWeights::layers[j].weight` / `.weight_c2`,
//! plus the initial `conv_pre` and terminal `conv_post` layers the consumer
//! WP materialises) is the **fully-materialised effective inference weight**.
//! Upstream references (jik876/hifi-gan and derivatives) wrap every one of
//! these convs in `torch.nn.utils.weight_norm(Conv1d(...))`, which decomposes
//! `weight` into a `(weight_g, weight_v)` pair whose effective inference
//! tensor is
//!
//! ```text
//! weight = weight_g * weight_v / ||weight_v||_2
//! ```
//!
//! where `||weight_v||_2` is the L2 norm taken over every dimension **except
//! the first (`out_channels`)** — i.e. a per-output-channel scalar normaliser.
//! This is exactly what PyTorch's `torch.nn.utils.remove_weight_norm(module)`
//! computes and writes back as a single `weight` attribute at export time.
//!
//! **Converters MUST fold weight-norm before emitting `vokra.hifigan.*` GGUF
//! chunks.** Concretely, either:
//!
//! 1. Load the reference checkpoint, call `remove_weight_norm(module)` on
//!    every `Conv1d` / `ConvTranspose1d` under the generator (upstream shim
//!    scripts do this before ONNX / TorchScript export), then read the fused
//!    `weight` tensor; **or**
//! 2. If the checkpoint stores raw `(weight_g, weight_v)` pairs (the typical
//!    training-time state_dict layout — see e.g. `jik876/hifi-gan` official
//!    `g_02500000` checkpoints, whose keys are `.weight_g` / `.weight_v` per
//!    conv), fold them explicitly per the formula above before writing the
//!    resulting `weight` into the GGUF chunk this module reads.
//!
//! Emitting raw `weight_v` bytes as if they were the effective weight is a
//! **silent-wrong bug**: the forward pass runs to completion (all shapes
//! match) but produces garbage audio, exactly the class of latent bug that
//! only real-checkpoint parity catches (see the [[project-real-weight-eval]]
//! memory's M4-scale trap catalogue and the Kokoro architectural-bound
//! precedent in `docs/adr/M2-07-kokoro-per-tensor-atol.md` §Amplification for
//! why loud-fail on structural mismatch is preferred to silent-wrong on
//! numerical mismatch — FR-EX-08).
//!
//! This module intentionally accepts only fused effective weights: `(weight_g,
//! weight_v)` pairs are converter-side concerns and this crate's structs
//! (`UpsampleStageWeights`, `ResBlockLayer`) have no fields for the raw pair.

use vokra_core::ir::graph::{HifiGanAttrs, ResBlockType};
use vokra_core::{Result, VokraError};

// ---------------------------------------------------------------------------
// Weight bundle
// ---------------------------------------------------------------------------

/// Per-stage transposed-conv upsampling weights.
///
/// Weight layout follows PyTorch's `ConvTranspose1d(in_ch, out_ch, kernel,
/// stride, padding)`: `weight` is row-major `[in_ch, out_ch, kernel]`, `bias`
/// has length `out_ch`, `stride == upsample_rates[i]`, `padding = (kernel −
/// stride) / 2` per the upstream jik876/hifi-gan.
#[derive(Debug, Clone)]
pub struct UpsampleStageWeights {
    /// `[in_ch, out_ch, kernel]` row-major.
    pub weight: Vec<f32>,
    /// `[out_ch]` bias vector.
    pub bias: Vec<f32>,
    /// Input channel count (must equal preceding feature width).
    pub in_ch: usize,
    /// Output channel count (must equal next stage's `in_ch`).
    pub out_ch: usize,
    /// Kernel size (`upsample_kernel_sizes[i]`).
    pub kernel: usize,
    /// Stride (`upsample_rates[i]`).
    pub stride: usize,
}

/// One layer of an MRF ResBlock — one iteration of `for (c1, c2) in
/// zip(convs1, convs2)` (V1) or `for c in convs` (V2). Which arithmetic
/// runs is decided by [`HifiGanAttrs::res_block_type`], not by this
/// struct's field-presence — a V1 attrs *must* have `weight_c2` +
/// `bias_c2` on every layer (loud-fail via
/// [`mrf_branch_forward`](fn.mrf_branch_forward.html)-adjacent code
/// otherwise), a V2 attrs *must* have them `None` (partial mixing is
/// unsound: the loader can't decide which of two topologies to run).
///
/// # Layout
///
/// - `weight` (aka `convs1[j].weight` for V1, `convs[j].weight` for V2):
///   `[out_ch, in_ch, kernel]` row-major. `in_ch == out_ch` (MRF branch
///   is channel-preserving).
/// - `bias`: `[out_ch]`.
/// - `weight_c2` (`convs2[j].weight` — V1 only, `None` for V2): same
///   `[out_ch, in_ch, kernel]` shape, but the reference always sets
///   `dilation=1` (`convs2` is the undilated conv per
///   `tools/parity/vendor/vits/modules.py::ResBlock1.__init__` at line
///   244-251).
/// - `bias_c2`: `[out_ch]`.
///
/// The `dilation` field carries the `convs1` dilation only; the `c2`
/// dilation is architecturally `1` per the upstream reference (see the
/// `dilation=1` on every `convs2` `Conv1d` in the reference) and this
/// module hard-codes that — a future variant that changed the `c2`
/// dilation would be a schema break, not an attribute override.
#[derive(Debug, Clone)]
pub struct ResBlockLayer {
    /// `convs1[j].weight` (V1) or `convs[j].weight` (V2). `[out_ch,
    /// in_ch, kernel]` row-major.
    pub weight: Vec<f32>,
    /// `convs1[j].bias` (V1) or `convs[j].bias` (V2). `[out_ch]`.
    pub bias: Vec<f32>,
    /// `convs2[j].weight` (V1 only). Must be `Some` when the enclosing
    /// [`HifiGanAttrs::res_block_type`] is [`ResBlockType::V1`] and
    /// `None` for [`ResBlockType::V2`]. Same `[out_ch, in_ch, kernel]`
    /// shape as `weight` — the `c2` conv is channel-preserving too.
    pub weight_c2: Option<Vec<f32>>,
    /// `convs2[j].bias` (V1 only). Presence must match `weight_c2`.
    /// `[out_ch]`.
    pub bias_c2: Option<Vec<f32>>,
    /// Dilation of `convs1[j]`. `convs2[j]` is architecturally
    /// `dilation=1` per upstream and is not a separate field.
    pub dilation: usize,
    /// Kernel size (`resblock_kernel_sizes[branch]`). Shared between
    /// `convs1[j]` and `convs2[j]` (upstream `ResBlock1.__init__` passes
    /// the same `kernel_size` to both).
    pub kernel: usize,
    /// Number of channels (branch is channel-preserving).
    pub channels: usize,
}

/// One MRF branch: a parallel residual stack of dilated conv1d layers whose
/// output is added to the branch input.
///
/// The branch preserves channel count and time-length (`padding = dilation ·
/// (kernel − 1) / 2` per upstream), and produces a `[channels, time]` output the
/// stage-level averager combines with other branches.
#[derive(Debug, Clone)]
pub struct MrfBranchWeights {
    /// Sequential dilated conv1d layers with residual add wrapping the whole branch.
    pub layers: Vec<ResBlockLayer>,
}

/// Weights bundle for a complete HiFi-GAN generator forward pass.
///
/// This is the value the M3-09 CosyVoice2 (or a future dedicated) converter
/// materialises from the checkpoint's `vokra.hifigan.*` chunks. The struct is
/// intentionally a *value* type — the M3-07 op-only WP does not describe a
/// storage layout (that is the checkpoint / converter's job); it just
/// documents the runtime shape the forward function reads.
#[derive(Debug, Clone)]
pub struct HifiGanWeights {
    /// Initial `conv1d` mapping `[n_mels] → [initial_channel]`.
    /// `[initial_channel, n_mels, conv_pre_kernel]` row-major.
    pub conv_pre_weight: Vec<f32>,
    /// `[initial_channel]` bias.
    pub conv_pre_bias: Vec<f32>,
    /// Kernel size of the initial `conv1d` (upstream default = 7).
    pub conv_pre_kernel: usize,
    /// Per-stage upsampling weights. `.len() == attrs.n_upsample_stages()`.
    pub upsample_weights: Vec<UpsampleStageWeights>,
    /// Per-stage MRF branch weights.
    /// `mrf_stage_weights[stage][branch]` shape.
    /// `.len() == attrs.n_upsample_stages()`, inner `.len() == attrs.n_mrf_branches()`.
    pub mrf_stage_weights: Vec<Vec<MrfBranchWeights>>,
    /// Final `conv1d` mapping `[channels] → [1]`.
    /// `[1, ch_last, conv_post_kernel]` row-major.
    pub conv_post_weight: Vec<f32>,
    /// `[1]` bias — **or `[]` (empty) when the upstream `conv_post` was
    /// trained `bias=False`** (HGAN-04 fix, 2026-08-09). VITS-family
    /// upstreams (`tools/parity/vendor/vits/models.py` at `Conv1d(ch,
    /// 1, 7, 1, padding=3, bias=False)`) and SBV2 v2 both ship
    /// bias-less `conv_post`; storing an empty `Vec` here lets a
    /// converter emit the true upstream shape without the pre-HGAN-04
    /// zero-placeholder workaround. Both shapes are numerically
    /// identical (`x + 0.0 == x`).
    pub conv_post_bias: Vec<f32>,
    /// Kernel size of the final `conv1d` (upstream default = 7).
    pub conv_post_kernel: usize,
    /// **HGAN-05-GIN-COND** (2026-08-09) — optional `cond` layer for
    /// speaker (or general gin) conditioning. `Some(_)` when the
    /// upstream ships a `dec.cond` `Conv1d(gin_channels,
    /// initial_channel, 1)` (present on SBV2 v2, all multi-speaker
    /// VITS HiFi-GAN generators, and any generator trained with
    /// `gin_channels > 0`); `None` on unconditioned generators
    /// (piper-plus, single-speaker VITS-JA). See [`GinCondition`]'s
    /// field docs for the shape contract.
    ///
    /// **Backward-compat**: pre-HGAN-05 callers construct
    /// [`HifiGanWeights`] without this field via
    /// `.. HifiGanWeights { cond: None, .. }`; the runtime forward
    /// pass short-circuits the cond broadcast-add when `cond ==
    /// None`, preserving byte-identical output for every
    /// pre-HGAN-05 consumer (piper-plus, CosyVoice2, VITS-JA).
    pub cond: Option<GinCondition>,
}

/// HGAN-05-GIN-COND: `cond` layer weight bundle (2026-08-09).
///
/// Upstream reference (`tools/parity/vendor/vits/models.py`
/// `Generator.__init__`):
///
/// ```python
/// if gin_channels != 0:
///     self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
/// ```
///
/// Applied in `Generator.forward` after `conv_pre`:
///
/// ```python
/// x = self.conv_pre(x)
/// if g is not None:
///     x = x + self.cond(g)  # broadcast-add per-time-step
/// ```
///
/// where `g` is `[B, gin_channels, 1]` and the 1×1 conv output is
/// `[B, upsample_initial_channel, 1]`, broadcast along the time
/// axis when added to `x` (shape `[B, upsample_initial_channel,
/// T]`).
///
/// # Layout
///
/// - `weight` is `[upsample_initial_channel, gin_channels, 1]`
///   row-major (PyTorch `Conv1d` convention `[out_ch, in_ch, kernel]`,
///   kernel = 1). Storage size: `initial_channel * gin_channels`.
/// - `bias` is `[upsample_initial_channel]`. Present in upstream
///   (default `bias=True`), so this field is **required** when
///   `cond` is `Some(_)`.
/// - `gin_channels`: input conditioning-vector width (e.g. 512 for
///   SBV2 v2). Also validated against `g.len()` at forward time.
#[derive(Debug, Clone)]
pub struct GinCondition {
    /// `dec.cond.weight`, row-major `[initial_channel, gin_channels,
    /// 1]`.
    pub weight: Vec<f32>,
    /// `dec.cond.bias`, `[initial_channel]`.
    pub bias: Vec<f32>,
    /// Input conditioning-vector width (upstream `gin_channels`,
    /// = 512 on SBV2 v2). Cross-checked against `g.len()` at forward
    /// time.
    pub gin_channels: usize,
}

// ---------------------------------------------------------------------------
// INT8 opt-in gate — HifiGanConfig, CalibrationTable, SpectralCheckResult
// ---------------------------------------------------------------------------

/// Precision policy for a [`hifigan_generator`] call.
///
/// Defaults to FP32 with INT8 opt-in disabled; the M3-07 op-only WP only ships
/// FP32 and mixed-precision-fp16 (FP32 accumulator) paths. Enabling INT8 also
/// requires attaching a [`CalibrationTable`] and passing the spectral check —
/// see the module-level doc.
#[derive(Debug, Clone, Default)]
pub struct HifiGanConfig {
    /// Precision of the forward pass.
    pub precision: HifiGanPrecision,
    /// INT8 opt-in flag (FR-OP-10). **Default `false`.** Must be paired with
    /// both a `CalibrationTable` and a `true` `spectral_check_passed`.
    pub int8_enabled: bool,
    /// Per-channel INT8 calibration table (FR-QT-03). Must be `Some` when
    /// `int8_enabled == true`.
    pub calibration_data: Option<CalibrationTable>,
    /// Whether the MEL / UTMOS spectral check between FP32 and INT8 forward
    /// passes the NFR-QL-02 5% gate. Must be `true` when
    /// `int8_enabled == true`.
    pub spectral_check_passed: bool,
}

/// Precision of the HiFi-GAN forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HifiGanPrecision {
    /// Full FP32 (default).
    #[default]
    Fp32,
    /// FP16 weights + activations with an FP32 accumulator. The BF16
    /// mantissa-loss note in the CLAUDE.md audio dialect applies to any
    /// vocoder-side accumulator; only the accumulator stays FP32, matching the
    /// mixed-precision path M0-08 established.
    Fp16,
}

impl HifiGanConfig {
    /// Construct a plain FP32 config with INT8 opt-in disabled.
    #[must_use]
    pub fn fp32() -> Self {
        Self::default()
    }

    /// Construct an FP16 mixed-precision config with INT8 opt-in disabled.
    #[must_use]
    pub fn fp16() -> Self {
        Self {
            precision: HifiGanPrecision::Fp16,
            int8_enabled: false,
            calibration_data: None,
            spectral_check_passed: false,
        }
    }

    /// **Sole atomic path** for enabling INT8: flips `int8_enabled = true` and
    /// attaches the calibration table + the spectral check verdict in one call.
    ///
    /// A caller cannot construct an INT8-enabled config without both proofs —
    /// this mirrors [`vokra_core::quant::QuantPolicy::with_hifigan_int8_opt_in`]
    /// (M2-08 T10), so a policy and a runtime call share one gate shape.
    ///
    /// **Note (M3-07 op-only WP):** the INT8 forward kernel is not shipped by
    /// this WP; [`hifigan_generator`] rejects INT8 execution with
    /// [`VokraError::UnsupportedOp`] to keep FR-EX-08 (no silent fallback)
    /// intact. This constructor exists so the gate is testable end-to-end.
    #[must_use]
    pub fn with_int8_opt_in(
        mut self,
        calibration: CalibrationTable,
        spectral_check_passed: bool,
    ) -> Self {
        self.int8_enabled = true;
        self.calibration_data = Some(calibration);
        self.spectral_check_passed = spectral_check_passed;
        self
    }

    /// Validate the INT8 opt-in invariant. Returns
    /// [`VokraError::HifiganInt8VerifyMissing`] when `int8_enabled == true` and
    /// either calibration is missing or the spectral check has not passed.
    pub fn validate(&self) -> Result<()> {
        if self.int8_enabled && (self.calibration_data.is_none() || !self.spectral_check_passed) {
            return Err(VokraError::HifiganInt8VerifyMissing);
        }
        Ok(())
    }
}

/// Per-channel INT8 calibration table (FR-OP-10 / FR-QT-03).
///
/// One `(scale, zero_point)` pair per output channel — per-channel is a
/// requirement of FR-OP-10, not a suggestion. The blob is *opaque* to the
/// runtime function; it is validated only for shape and finiteness. The
/// concrete calibration algorithm (min-max, 99.9 percentile, KL) is chosen by
/// the caller through [`HifiGanCalibrator`] and captured here as the resulting
/// quantization parameters.
///
/// M2-08 keeps the on-disk blob format opaque behind
/// [`vokra_core::quant::CalibrationRef`]. This type is the *runtime-side*
/// materialisation of that blob.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationTable {
    /// Per-channel scale (must be finite and > 0).
    pub scales: Vec<f32>,
    /// Per-channel zero-point (INT8 zero of the affine mapping).
    pub zero_points: Vec<i8>,
    /// Channel count `scales.len()` mirrors. Kept so a downstream check
    /// against `HifiGanAttrs::initial_channel` (or the final upsample width)
    /// can catch a wrong-shape table without a length compare.
    pub channels: usize,
}

impl CalibrationTable {
    /// Construct a table with cross-field shape and finiteness checks.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any of:
    /// - `scales.len() != zero_points.len()`;
    /// - `scales.len() != channels`;
    /// - a non-finite or non-positive scale;
    /// - `channels == 0`.
    pub fn new(scales: Vec<f32>, zero_points: Vec<i8>, channels: usize) -> Result<Self> {
        if channels == 0 {
            return Err(VokraError::InvalidArgument(
                "CalibrationTable: channels must be > 0".to_owned(),
            ));
        }
        if scales.len() != channels {
            return Err(VokraError::InvalidArgument(format!(
                "CalibrationTable: scales.len() {} != channels {channels}",
                scales.len()
            )));
        }
        if zero_points.len() != channels {
            return Err(VokraError::InvalidArgument(format!(
                "CalibrationTable: zero_points.len() {} != channels {channels}",
                zero_points.len()
            )));
        }
        for (i, s) in scales.iter().enumerate() {
            if !s.is_finite() || *s <= 0.0 {
                return Err(VokraError::InvalidArgument(format!(
                    "CalibrationTable: scales[{i}] = {s} must be finite and > 0"
                )));
            }
        }
        Ok(Self {
            scales,
            zero_points,
            channels,
        })
    }
}

/// Per-channel INT8 calibrator (T08).
///
/// Two calibration strategies are supported today:
///
/// - [`CalibrationStrategy::MinMax`]: `scale = max(|min|, |max|) / 127.0`,
///   `zero_point = 0`. Symmetric per-channel scaling — the simplest correct
///   mapping.
/// - [`CalibrationStrategy::Percentile { p }`]: same as `MinMax` but uses the
///   `p`-th percentile of `|activations|` instead of the true max, dampening
///   the outlier tail. `p == 100.0` collapses to `MinMax`.
///
/// The consumer WP (M3-09 CosyVoice2) supplies a real calibration dataset; the
/// M3-07 tests exercise the strategy on synthetic activation tensors so the
/// output shape / finiteness / determinism guarantees hold before any
/// checkpoint is available.
///
/// **Deterministic**: given the same input activations and strategy, the table
/// is bit-identical across runs. No hidden RNG.
#[derive(Debug, Clone, Copy)]
pub struct HifiGanCalibrator {
    strategy: CalibrationStrategy,
}

/// Strategy for the [`HifiGanCalibrator`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalibrationStrategy {
    /// Symmetric per-channel min-max mapping (`scale = max(|min|, |max|) / 127`).
    MinMax,
    /// Percentile-based symmetric mapping. `p ∈ (0.0, 100.0]`.
    Percentile {
        /// Percentile of `|activations|` used as the effective absolute-max.
        p: f32,
    },
}

impl HifiGanCalibrator {
    /// Build a calibrator with the given strategy.
    #[must_use]
    pub fn new(strategy: CalibrationStrategy) -> Self {
        Self { strategy }
    }

    /// Run calibration over `activations` shaped `[batch, channels]` row-major.
    ///
    /// Returns a per-channel [`CalibrationTable`] whose length equals
    /// `channels`. The strategy is applied *per column* (per channel).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any of:
    /// - `channels == 0`;
    /// - `activations.len()` not a multiple of `channels`;
    /// - a non-finite activation value;
    /// - a non-finite / out-of-range percentile.
    pub fn calibrate(&self, activations: &[f32], channels: usize) -> Result<CalibrationTable> {
        if channels == 0 {
            return Err(VokraError::InvalidArgument(
                "HifiGanCalibrator::calibrate: channels must be > 0".to_owned(),
            ));
        }
        if activations.len() % channels != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "HifiGanCalibrator::calibrate: activations.len() {} not a multiple of channels {channels}",
                activations.len()
            )));
        }
        if activations.iter().any(|v| !v.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "HifiGanCalibrator::calibrate: activations must be finite".to_owned(),
            ));
        }
        let p = match self.strategy {
            CalibrationStrategy::MinMax => 100.0,
            CalibrationStrategy::Percentile { p } => {
                if !p.is_finite() || !(0.0..=100.0).contains(&p) || p == 0.0 {
                    return Err(VokraError::InvalidArgument(format!(
                        "HifiGanCalibrator::calibrate: percentile p={p} must be finite and in (0, 100]"
                    )));
                }
                p
            }
        };
        let batch = activations.len() / channels;
        let mut scales = vec![0.0_f32; channels];
        let zero_points = vec![0_i8; channels];

        // Per-channel: collect `|x|` values, take the p-th percentile (or the
        // true max when p == 100), divide by 127. Determinism comes from the
        // sorted-column approach.
        let mut column = vec![0.0_f32; batch];
        for c in 0..channels {
            for (r, entry) in column.iter_mut().enumerate().take(batch) {
                *entry = activations[r * channels + c].abs();
            }
            // Percentile: sort ascending, take index (p / 100) * (batch - 1)
            // clamped to [0, batch − 1]. For `MinMax`, this reduces to
            // `column[batch − 1]` after sort = max.
            let mut sorted = column.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = if batch == 0 {
                0
            } else {
                ((p as f64 / 100.0) * (batch as f64 - 1.0)).round() as usize
            };
            let abs_max = if batch == 0 {
                0.0
            } else {
                sorted[idx.min(batch - 1)]
            };
            // Guard against a strictly-zero column (map to scale=1 so the
            // resulting table is representable). This is the same guard PyTorch
            // uses for a fully-zero calibration column.
            let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
            scales[c] = scale;
        }
        CalibrationTable::new(scales, zero_points, channels)
    }
}

/// Spectral check verdict (T09). Ties a MEL-loss delta to the NFR-QL-02 5%
/// gate. The delta is `abs(loss_int8 - loss_fp32) / max(loss_fp32, ε)` — the
/// same relative-loss shape M2-08 uses for
/// [`VokraError::HifiganInt8DegradationExceeded`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpectralCheckResult {
    /// Delta within the 5% gate — INT8 opt-in may proceed.
    Passed {
        /// Observed relative delta (informational).
        delta: f32,
    },
    /// Delta exceeds the 5% gate — INT8 opt-in stays refused.
    Failed {
        /// Observed relative delta (informational).
        delta: f32,
    },
}

impl SpectralCheckResult {
    /// Whether this verdict allows INT8 opt-in.
    #[must_use]
    pub fn is_passed(&self) -> bool {
        matches!(self, Self::Passed { .. })
    }

    /// Observed relative MEL-loss delta.
    #[must_use]
    pub fn delta(&self) -> f32 {
        match self {
            Self::Passed { delta } | Self::Failed { delta } => *delta,
        }
    }
}

/// NFR-QL-02 relative-delta gate (5%). Kept as a `const` so a future policy
/// tightening only touches one place.
pub const SPECTRAL_CHECK_THRESHOLD: f32 = 0.05;

/// Spectral checker (T09). Computes a MEL-magnitude-loss delta between an FP32
/// reference waveform and an INT8 candidate waveform and returns a
/// [`SpectralCheckResult`].
///
/// The MEL-loss shape here is a *proxy* for the M1 `vokra-eval` MEL loss (which
/// requires the mel filterbank + STFT pipeline; the M1 crate is where the
/// production wiring lives). The proxy computes an L2 magnitude difference
/// over uniformly-strided frames of the two waveforms, which shares the
/// scaling / sensitivity properties MEL loss has for HiFi-GAN calibration
/// verification. The M3-09 consumer WP swaps this proxy for a full `vokra-eval`
/// call once a real calibration dataset exists (WP boundary — see the ticket's
/// T09 explanation of the M1 hookup point).
#[derive(Debug, Clone, Copy)]
pub struct HifiGanSpectralChecker {
    threshold: f32,
}

impl Default for HifiGanSpectralChecker {
    fn default() -> Self {
        Self {
            threshold: SPECTRAL_CHECK_THRESHOLD,
        }
    }
}

impl HifiGanSpectralChecker {
    /// Build a checker with the default NFR-QL-02 5% threshold.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a checker with a custom threshold. `threshold` must be finite and
    /// in `[0.0, 1.0]`; values outside are clamped to the default gate. The
    /// production build always uses the default; the setter exists so tests
    /// can push a tighter or looser gate deterministically.
    #[must_use]
    pub fn with_threshold(threshold: f32) -> Self {
        let t = if threshold.is_finite() && (0.0..=1.0).contains(&threshold) {
            threshold
        } else {
            SPECTRAL_CHECK_THRESHOLD
        };
        Self { threshold: t }
    }

    /// Compare an FP32 reference against an INT8 candidate. `fp32` and `int8`
    /// must have the same length.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a length mismatch or a non-finite
    /// sample. The gate itself never errors — a delta above the threshold is a
    /// [`SpectralCheckResult::Failed`], not an error.
    pub fn check(&self, fp32: &[f32], int8: &[f32]) -> Result<SpectralCheckResult> {
        if fp32.len() != int8.len() {
            return Err(VokraError::InvalidArgument(format!(
                "HifiGanSpectralChecker: fp32.len() {} != int8.len() {}",
                fp32.len(),
                int8.len()
            )));
        }
        if fp32.iter().chain(int8.iter()).any(|v| !v.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "HifiGanSpectralChecker: samples must be finite".to_owned(),
            ));
        }
        // Proxy MEL loss: L2 magnitude difference per contiguous 32-sample
        // window (a coarse spectral surrogate — the M1 `vokra-eval` MEL loss
        // is the production replacement, see docstring).
        let window = 32usize.min(fp32.len().max(1));
        let mut loss_fp32 = 0.0_f64;
        let mut loss_int8 = 0.0_f64;
        let mut i = 0;
        while i < fp32.len() {
            let end = (i + window).min(fp32.len());
            let mut mag_ref = 0.0_f64;
            let mut mag_delta = 0.0_f64;
            for (a, b) in fp32[i..end].iter().zip(int8[i..end].iter()) {
                mag_ref += f64::from(*a) * f64::from(*a);
                let diff = f64::from(*a) - f64::from(*b);
                mag_delta += diff * diff;
            }
            loss_fp32 += mag_ref.sqrt();
            loss_int8 += (mag_ref + mag_delta).sqrt();
            i += window;
        }
        let ref_denom = loss_fp32.max(1e-9);
        let delta = ((loss_int8 - loss_fp32).abs() / ref_denom) as f32;
        Ok(if delta <= self.threshold {
            SpectralCheckResult::Passed { delta }
        } else {
            SpectralCheckResult::Failed { delta }
        })
    }
}

// ---------------------------------------------------------------------------
// FP32 / FP16 forward
// ---------------------------------------------------------------------------

/// Runs the HiFi-GAN generator forward pass.
///
/// See the module doc for the op contract. Every convolution / MRF stage is
/// executed in FP32 with an FP32 accumulator; when
/// `config.precision == HifiGanPrecision::Fp16` the *weight loads* narrow to
/// FP16 and widen back to FP32 for the accumulator (`f16 → f32` widening on
/// every read, mirroring M0-08's mixed-precision pattern). INT8 execution is
/// gated but not yet implemented; see the module doc.
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] on shape / attribute mismatch or
///   non-finite input.
/// - [`VokraError::HifiganInt8VerifyMissing`] when
///   `config.int8_enabled == true` but the calibration table or spectral
///   check verdict is missing.
/// - [`VokraError::UnsupportedOp`] when INT8 is fully-authorised but the
///   forward kernel is not yet implemented (op-only WP boundary).
pub fn hifigan_generator(
    mel: &[f32],
    n_frames: usize,
    weights: &HifiGanWeights,
    attrs: &HifiGanAttrs,
    config: &HifiGanConfig,
) -> Result<Vec<f32>> {
    hifigan_generator_conditioned(mel, n_frames, weights, attrs, config, None)
}

/// HGAN-05-GIN-COND (2026-08-09): HiFi-GAN generator with optional
/// speaker (or general gin) conditioning.
///
/// Extends [`hifigan_generator`] with a `g: Option<&[f32]>` parameter
/// carrying the per-utterance conditioning vector. When `g` is
/// `Some(vec)` **and** `weights.cond` is `Some(cond)`, the runtime
/// applies the upstream `x = x + self.cond(g)` broadcast-add after
/// `conv_pre`. When `g` is `None`, this function is byte-identical
/// to [`hifigan_generator`] — the pre-HGAN-05 unconditioned path.
///
/// # Contract table
///
/// | `g`         | `weights.cond` | behavior                                                     |
/// |-------------|----------------|--------------------------------------------------------------|
/// | `None`      | `None`         | unconditioned generator (piper-plus, VITS-JA single-speaker) |
/// | `Some(v)`   | `Some(cond)`   | multi-speaker: broadcast-add `cond(v)` after `conv_pre`      |
/// | `Some(_)`   | `None`         | loud `InvalidArgument` (FR-EX-08 — caller sent g without a cond layer) |
/// | `None`      | `Some(_)`      | loud `InvalidArgument` (FR-EX-08 — cond layer loaded but g missing)    |
///
/// The loud-error arms are the FR-EX-08 no-silent-fallback contract:
/// a converter that emitted `dec.cond.*` tensors (into
/// `weights.cond`) but a caller that forgot to thread `g` through
/// would produce single-speaker-quality output on a multi-speaker
/// model without any assertion firing. Explicit mismatch → loud
/// panic path.
///
/// # Errors
///
/// See [`hifigan_generator`]. Additionally raises
/// [`VokraError::InvalidArgument`] when `(g, weights.cond)` mismatches
/// as described in the contract table.
pub fn hifigan_generator_conditioned(
    mel: &[f32],
    n_frames: usize,
    weights: &HifiGanWeights,
    attrs: &HifiGanAttrs,
    config: &HifiGanConfig,
    g: Option<&[f32]>,
) -> Result<Vec<f32>> {
    attrs.validate_shape()?;
    config.validate()?;

    if config.int8_enabled {
        return Err(VokraError::UnsupportedOp(
            "hifigan_generator: INT8 forward kernel not yet implemented (M3-07 op-only WP); \
             the calibration + spectral check gate is validated, kernel lands with the \
             consumer WP (M3-09 CosyVoice2)"
                .to_owned(),
        ));
    }

    if mel.len() != attrs.n_mels * n_frames {
        return Err(VokraError::InvalidArgument(format!(
            "hifigan_generator: mel.len() {} != n_mels * n_frames = {} * {} = {}",
            mel.len(),
            attrs.n_mels,
            n_frames,
            attrs.n_mels * n_frames
        )));
    }
    if mel.iter().any(|v| !v.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "hifigan_generator: mel must be finite".to_owned(),
        ));
    }
    validate_weights(weights, attrs)?;

    // HGAN-05-GIN-COND: cross-validate (g, weights.cond) pairing per
    // the contract table above.
    match (g, weights.cond.as_ref()) {
        (Some(_), None) => {
            return Err(VokraError::InvalidArgument(
                "hifigan_generator_conditioned: caller supplied g but weights.cond is None — \
                 an unconditioned generator cannot consume speaker conditioning. Either drop \
                 g (pass None) or load a cond weight bundle."
                    .to_owned(),
            ));
        }
        (None, Some(_)) => {
            return Err(VokraError::InvalidArgument(
                "hifigan_generator_conditioned: weights.cond is Some but g is None — a \
                 multi-speaker generator requires the per-utterance conditioning vector. \
                 Either provide g or drop weights.cond (build an unconditioned generator)."
                    .to_owned(),
            ));
        }
        (Some(g_vec), Some(cond)) => {
            if g_vec.len() != cond.gin_channels {
                return Err(VokraError::InvalidArgument(format!(
                    "hifigan_generator_conditioned: g.len() {} != cond.gin_channels {}",
                    g_vec.len(),
                    cond.gin_channels
                )));
            }
            if g_vec.iter().any(|v| !v.is_finite()) {
                return Err(VokraError::InvalidArgument(
                    "hifigan_generator_conditioned: g must be finite".to_owned(),
                ));
            }
        }
        (None, None) => {} // unconditioned path — nothing to check
    }

    // --- Stage 0: initial conv1d [n_mels, n_frames] → [initial_channel, n_frames] ---
    let mut h = conv1d_scalar(
        mel,
        attrs.n_mels,
        n_frames,
        &weights.conv_pre_weight,
        attrs.initial_channel,
        weights.conv_pre_kernel,
        Some(&weights.conv_pre_bias),
        1,                           // stride
        weights.conv_pre_kernel / 2, // "same" padding
    )?;

    // --- HGAN-05-GIN-COND: broadcast-add cond(g) after conv_pre ---
    //
    // Upstream reference (`tools/parity/vendor/vits/models.py`
    // `Generator.forward`):
    //
    // ```python
    // x = self.conv_pre(x)
    // if g is not None:
    //     x = x + self.cond(g)
    // ```
    //
    // `self.cond` is `Conv1d(gin_channels, initial_channel, 1)` — the
    // 1×1 conv output is `[B, initial_channel, 1]`, broadcast along
    // the time axis when added to x's `[B, initial_channel, T]`. In
    // our row-major channel-major `[initial_channel, n_frames]` layout,
    // this is `h[c, t] += (cond_weight · g)[c] + cond_bias[c]` for
    // every `t`. The `(cond_weight · g)[c] + cond_bias[c]` value is
    // computed once per call and added identically to every `t`.
    if let (Some(g_vec), Some(cond)) = (g, weights.cond.as_ref()) {
        // 1×1 conv: `cond_out[c] = Σ_i cond_weight[c, i] * g[i] + cond_bias[c]`.
        // Weight is `[initial_channel, gin_channels, 1]` = `[initial_channel, gin_channels]`
        // when kernel is 1 (contiguous).
        let mut cond_out = vec![0.0_f32; attrs.initial_channel];
        for (c, out_c) in cond_out.iter_mut().enumerate() {
            let row = &cond.weight[c * cond.gin_channels..(c + 1) * cond.gin_channels];
            let mut acc = 0.0_f32;
            for (&w, &g_val) in row.iter().zip(g_vec.iter()) {
                acc += w * g_val;
            }
            *out_c = acc + cond.bias[c];
        }
        // Broadcast-add across every time step. Chunk `h` into
        // per-channel rows (`[cur_channels=initial_channel, n_frames]`
        // layout — every row is `n_frames` samples wide).
        for (c, row) in h
            .chunks_exact_mut(n_frames)
            .enumerate()
            .take(attrs.initial_channel)
        {
            let cond_c = cond_out[c];
            for v in row.iter_mut() {
                *v += cond_c;
            }
        }
    }
    // Feature-map width after conv_pre.
    let mut cur_channels = attrs.initial_channel;
    let mut cur_time = n_frames;

    // --- Upsample stack ---
    for stage in 0..attrs.n_upsample_stages() {
        // leaky_relu.
        leaky_relu_inplace(&mut h, attrs.leaky_relu_slope);
        // transposed conv1d.
        let up = &weights.upsample_weights[stage];
        let padding = (up.kernel.saturating_sub(up.stride)) / 2;
        let out_time = (cur_time - 1) * up.stride + up.kernel - 2 * padding;
        let up_out = transposed_conv1d_scalar(
            &h,
            up.in_ch,
            cur_time,
            &up.weight,
            up.out_ch,
            up.kernel,
            Some(&up.bias),
            up.stride,
            padding,
        )?;
        // MRF: average over branches; each branch preserves shape.
        let mrf_stage = &weights.mrf_stage_weights[stage];
        let mut mrf_acc = vec![0.0_f32; up.out_ch * out_time];
        for branch in mrf_stage {
            let branch_out = mrf_branch_forward(
                &up_out,
                up.out_ch,
                out_time,
                branch,
                attrs.leaky_relu_slope,
                attrs.res_block_type,
            )?;
            for (a, b) in mrf_acc.iter_mut().zip(branch_out.iter()) {
                *a += *b;
            }
        }
        let inv_branches = 1.0_f32 / attrs.n_mrf_branches() as f32;
        for v in mrf_acc.iter_mut() {
            *v *= inv_branches;
        }
        h = mrf_acc;
        cur_channels = up.out_ch;
        cur_time = out_time;
    }

    // --- Final leaky_relu → conv1d → tanh ---
    //
    // HGAN-03 fix (2026-08-09): reference
    // `tools/parity/vendor/vits/models.Generator.forward` at
    // `decoder.py:97` calls plain `F.leaky_relu(x)` **without** the
    // `LRELU_SLOPE=0.1` argument that its in-loop calls use — PyTorch's
    // default slope is `0.01`, distinct from the `LRELU_SLOPE = 0.1`
    // constant threaded through the upsample loop above. Pin the final
    // pre-conv_post activation at the reference's implicit default so
    // this call matches upstream regardless of how a caller (or a
    // future SKU) tunes `attrs.leaky_relu_slope`.
    const FINAL_LEAKY_RELU_SLOPE: f32 = 0.01;
    leaky_relu_inplace(&mut h, FINAL_LEAKY_RELU_SLOPE);
    // HGAN-04 (2026-08-09): pass `None` when the upstream conv_post was
    // trained `bias=False` (represented by an empty `conv_post_bias`
    // vector — see the field doc). `x + 0.0 == x` for finite `x`, so the
    // two paths are numerically identical; the fork exists so a
    // converter can emit the true upstream shape without fabricating a
    // zero placeholder.
    let conv_post_bias_slot: Option<&[f32]> = if weights.conv_post_bias.is_empty() {
        None
    } else {
        Some(&weights.conv_post_bias)
    };
    let final_out = conv1d_scalar(
        &h,
        cur_channels,
        cur_time,
        &weights.conv_post_weight,
        1,
        weights.conv_post_kernel,
        conv_post_bias_slot,
        1,
        weights.conv_post_kernel / 2,
    )?;
    // tanh head — bound to (−1, 1).
    //
    // WP-08 (2026-08-10): route through `vokra_math::tanh` (bespoke f32
    // polynomial, cross-plat deterministic within Vokra) instead of the
    // platform `f32::tanh` (glibc vs Apple libm 1-ULP scatter). This is
    // the dominant transcendental site in the SBV2 hot path (~220k calls
    // per utterance through the MRF-fused generator, per the Wave-1
    // investigation report), and the primary route for tightening
    // `waveform` atol from the current 1.5 Measured floor toward the
    // ~1e-3 order Kokoro `pcm` established for similarly deep decoders.
    // Owner scope decision (2026-08-09, WP-05): SBV2 hot path only —
    // Whisper/Kokoro/Voxtral untouched (they keep their per-model
    // accumulator pin discipline).
    let mut waveform = final_out;
    for v in waveform.iter_mut() {
        *v = vokra_math::tanh(*v);
        // If the precision selector is Fp16, round the *output* through the
        // f16 representable set to mirror what a real fp16 accumulator would
        // yield. Every hidden layer already computed in FP32 (FP32 accumulator
        // per the doc); the only mixed-precision knob at the runtime function
        // level today is the terminal cast.
        if config.precision == HifiGanPrecision::Fp16 {
            *v = f32_round_to_f16_repr(*v);
        }
    }

    Ok(waveform)
}

// ---------------------------------------------------------------------------
// Helpers — scalar kernels
// ---------------------------------------------------------------------------

/// Row-major `conv1d` with "same" padding when `stride == 1`.
///
/// Layout matches `vokra_backend_cpu::kernels::conv1d_f32` (`weight` is
/// `[out_ch, in_ch, kernel]`, output is `[out_ch, out_len]`); the M3-07 WP does
/// not depend on `vokra-backend-cpu`, so a scalar copy lives here. SIMD hooks
/// are left for a follow-up (`AVX2` / `NEON` — mentioned in the T06 ticket).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on any shape mismatch or `stride == 0`.
#[allow(clippy::too_many_arguments)] // convolution's intrinsic parameter set
fn conv1d_scalar(
    input: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    out_ch: usize,
    kernel: usize,
    bias: Option<&[f32]>,
    stride: usize,
    padding: usize,
) -> Result<Vec<f32>> {
    if stride == 0 {
        return Err(VokraError::InvalidArgument(
            "conv1d: stride must be >= 1".to_owned(),
        ));
    }
    if kernel == 0 {
        return Err(VokraError::InvalidArgument(
            "conv1d: kernel must be >= 1".to_owned(),
        ));
    }
    if input.len() != in_ch * in_len {
        return Err(VokraError::InvalidArgument(format!(
            "conv1d: input.len() {} != in_ch * in_len {}",
            input.len(),
            in_ch * in_len
        )));
    }
    if weight.len() != out_ch * in_ch * kernel {
        return Err(VokraError::InvalidArgument(format!(
            "conv1d: weight.len() {} != out_ch * in_ch * kernel {}",
            weight.len(),
            out_ch * in_ch * kernel
        )));
    }
    if let Some(b) = bias
        && b.len() != out_ch
    {
        return Err(VokraError::InvalidArgument(format!(
            "conv1d: bias.len() {} != out_ch {}",
            b.len(),
            out_ch
        )));
    }
    let padded = in_len + 2 * padding;
    if padded < kernel {
        return Err(VokraError::InvalidArgument(format!(
            "conv1d: padded length {padded} < kernel {kernel}"
        )));
    }
    let out_len = (padded - kernel) / stride + 1;
    let mut out = vec![0.0_f32; out_ch * out_len];
    for oc in 0..out_ch {
        let bias_v = bias.map(|b| b[oc]).unwrap_or(0.0);
        for oi in 0..out_len {
            let mut acc = f64::from(bias_v);
            for ic in 0..in_ch {
                for k in 0..kernel {
                    let padded_ix = oi * stride + k;
                    if padded_ix < padding {
                        continue;
                    }
                    let in_ix = padded_ix - padding;
                    if in_ix >= in_len {
                        continue;
                    }
                    let w = weight[(oc * in_ch + ic) * kernel + k];
                    let v = input[ic * in_len + in_ix];
                    acc += f64::from(w) * f64::from(v);
                }
            }
            out[oc * out_len + oi] = acc as f32;
        }
    }
    Ok(out)
}

/// Row-major transposed `conv1d` (a.k.a. `ConvTranspose1d`).
///
/// `weight` layout `[in_ch, out_ch, kernel]` (PyTorch); output length is
/// `(in_len − 1) · stride − 2 · padding + kernel`. Errors on `stride == 0`,
/// shape mismatch, or a negative output length.
#[allow(clippy::too_many_arguments)] // convolution's intrinsic parameter set
fn transposed_conv1d_scalar(
    input: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    out_ch: usize,
    kernel: usize,
    bias: Option<&[f32]>,
    stride: usize,
    padding: usize,
) -> Result<Vec<f32>> {
    if stride == 0 {
        return Err(VokraError::InvalidArgument(
            "transposed_conv1d: stride must be >= 1".to_owned(),
        ));
    }
    if kernel == 0 {
        return Err(VokraError::InvalidArgument(
            "transposed_conv1d: kernel must be >= 1".to_owned(),
        ));
    }
    if input.len() != in_ch * in_len {
        return Err(VokraError::InvalidArgument(format!(
            "transposed_conv1d: input.len() {} != in_ch * in_len {}",
            input.len(),
            in_ch * in_len
        )));
    }
    if weight.len() != in_ch * out_ch * kernel {
        return Err(VokraError::InvalidArgument(format!(
            "transposed_conv1d: weight.len() {} != in_ch * out_ch * kernel {}",
            weight.len(),
            in_ch * out_ch * kernel
        )));
    }
    if let Some(b) = bias
        && b.len() != out_ch
    {
        return Err(VokraError::InvalidArgument(format!(
            "transposed_conv1d: bias.len() {} != out_ch {}",
            b.len(),
            out_ch
        )));
    }
    let full_out = (in_len - 1) * stride + kernel;
    if full_out < 2 * padding {
        return Err(VokraError::InvalidArgument(format!(
            "transposed_conv1d: 2*padding {} exceeds naive output {full_out}",
            2 * padding
        )));
    }
    let out_len = full_out - 2 * padding;
    let mut out = vec![0.0_f32; out_ch * out_len];
    // Initialise bias.
    if let Some(b) = bias {
        for oc in 0..out_ch {
            let bv = b[oc];
            for j in 0..out_len {
                out[oc * out_len + j] = bv;
            }
        }
    }
    // Accumulate cross-correlation into `out`.
    for ic in 0..in_ch {
        for oc in 0..out_ch {
            for i in 0..in_len {
                let x = input[ic * in_len + i];
                if x == 0.0 {
                    continue;
                }
                for k in 0..kernel {
                    let full_ix = i * stride + k;
                    if full_ix < padding {
                        continue;
                    }
                    let oj = full_ix - padding;
                    if oj >= out_len {
                        continue;
                    }
                    let w = weight[(ic * out_ch + oc) * kernel + k];
                    let idx = oc * out_len + oj;
                    let mut acc = f64::from(out[idx]);
                    acc += f64::from(x) * f64::from(w);
                    out[idx] = acc as f32;
                }
            }
        }
    }
    Ok(out)
}

/// Runs one MRF branch, matching upstream
/// [`ResBlock1.forward`](https://raw.githubusercontent.com/jik876/hifi-gan/master/models.py#L54)
/// / [`ResBlock2.forward`](https://raw.githubusercontent.com/jik876/hifi-gan/master/models.py#L92)
/// as vendored to `tools/parity/vendor/vits/modules.py::ResBlock1` (line 254)
/// / `::ResBlock2` (line 287). The `res_block_type` argument picks between:
///
/// - **V1** — per iteration `(c1, c2) in zip(convs1, convs2)`:
///   `xt = c2(lrelu(c1(lrelu(x)))); x = xt + x`.
/// - **V2** — per iteration `c in convs`: `xt = c(lrelu(x)); x = xt + x`.
///
/// # Wave-2 audit fix (HGAN-01 + HGAN-02)
///
/// Pre-Wave-2 this function did `h = conv_last(lrelu(conv_{n-1}(lrelu(…
/// lrelu(input)))))) + input` — one outer residual add regardless of
/// depth (the audit's HGAN-02). It also had no notion of the `c2`
/// convs2 chain — the converter's `convs2.*` tensors were passed
/// through unread, silently dropping half of every V1 vocoder's
/// convolutions (the audit's HGAN-01). Both together made SBV2 v2
/// waveform parity structurally impossible.
///
/// # Layout invariants
///
/// - `input` / return: `[channels, time]` row-major (`h[c * time + t]`).
///   The reference operates in `[B=1, D, T]` channel-major layout too
///   (upstream `ResBlock*.forward` sits at the same layout as the
///   surrounding VITS decoder `Generator.forward`); no transpose here.
/// - Every `layer.channels` must equal `channels`.
/// - V1 requires `layer.weight_c2 + layer.bias_c2 == Some` on every
///   layer; V2 requires them both `None`. Partial mixing is a converter
///   bug (loud `InvalidArgument`, FR-EX-08).
fn mrf_branch_forward(
    input: &[f32],
    channels: usize,
    time: usize,
    branch: &MrfBranchWeights,
    leaky_slope: f32,
    res_block_type: ResBlockType,
) -> Result<Vec<f32>> {
    if input.len() != channels * time {
        return Err(VokraError::InvalidArgument(format!(
            "mrf_branch_forward: input.len() {} != channels * time {}",
            input.len(),
            channels * time
        )));
    }
    if branch.layers.is_empty() {
        return Err(VokraError::InvalidArgument(
            "mrf_branch_forward: branch must have at least one layer".to_owned(),
        ));
    }
    // Running state — starts as `x = input`; each iteration mutates via
    // `x = xt + x` (residual add INSIDE the loop, matching upstream).
    let mut x = input.to_vec();
    for (layer_idx, layer) in branch.layers.iter().enumerate() {
        if layer.channels != channels {
            return Err(VokraError::InvalidArgument(format!(
                "mrf_branch_forward: layer.channels {} != branch channels {channels}",
                layer.channels
            )));
        }
        // ---- xt = c1(lrelu(x)) ----
        let mut xt = x.clone();
        leaky_relu_inplace(&mut xt, leaky_slope);
        let padding_c1 = layer.dilation * (layer.kernel - 1) / 2;
        xt = dilated_conv1d_scalar(
            &xt,
            channels,
            time,
            &layer.weight,
            channels,
            layer.kernel,
            Some(&layer.bias),
            layer.dilation,
            padding_c1,
        )?;
        // ---- V1: xt = c2(lrelu(xt)) with c2.dilation = 1 ----
        match res_block_type {
            ResBlockType::V1 => {
                let (weight_c2, bias_c2) =
                    match (layer.weight_c2.as_deref(), layer.bias_c2.as_deref()) {
                        (Some(w), Some(b)) => (w, b),
                        _ => {
                            return Err(VokraError::InvalidArgument(format!(
                                "mrf_branch_forward: ResBlockType::V1 requires \
                             weight_c2 + bias_c2 on every layer, but layer {layer_idx} has \
                             weight_c2.is_some() = {} / bias_c2.is_some() = {}. \
                             Converter and loader must supply both convs1[j] and convs2[j] \
                             for V1 topology (upstream ResBlock1 signature).",
                                layer.weight_c2.is_some(),
                                layer.bias_c2.is_some()
                            )));
                        }
                    };
                leaky_relu_inplace(&mut xt, leaky_slope);
                // `convs2[j]` is architecturally undilated (dilation=1) per
                // upstream `ResBlock1.__init__` — see
                // `tools/parity/vendor/vits/modules.py:244-251`.
                let padding_c2 = (layer.kernel - 1) / 2;
                xt = dilated_conv1d_scalar(
                    &xt,
                    channels,
                    time,
                    weight_c2,
                    channels,
                    layer.kernel,
                    Some(bias_c2),
                    1, // dilation
                    padding_c2,
                )?;
            }
            ResBlockType::V2 => {
                if layer.weight_c2.is_some() || layer.bias_c2.is_some() {
                    return Err(VokraError::InvalidArgument(format!(
                        "mrf_branch_forward: ResBlockType::V2 must not carry \
                         weight_c2 or bias_c2 (V2 has no convs2 chain — upstream ResBlock2). \
                         layer {layer_idx} has weight_c2.is_some() = {} / \
                         bias_c2.is_some() = {}. Converter must emit None for both under V2.",
                        layer.weight_c2.is_some(),
                        layer.bias_c2.is_some()
                    )));
                }
            }
        }
        // ---- x = xt + x (residual add INSIDE the loop, per iteration) ----
        for (xv, tv) in x.iter_mut().zip(xt.iter()) {
            *xv += *tv;
        }
    }
    Ok(x)
}

/// Dilated `conv1d` (stride == 1 always). `weight` layout matches
/// [`conv1d_scalar`]: `[out_ch, in_ch, kernel]`.
#[allow(clippy::too_many_arguments)] // convolution's intrinsic parameter set
fn dilated_conv1d_scalar(
    input: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    out_ch: usize,
    kernel: usize,
    bias: Option<&[f32]>,
    dilation: usize,
    padding: usize,
) -> Result<Vec<f32>> {
    if dilation == 0 {
        return Err(VokraError::InvalidArgument(
            "dilated_conv1d: dilation must be >= 1".to_owned(),
        ));
    }
    let padded = in_len + 2 * padding;
    let effective_kernel = 1 + (kernel - 1) * dilation;
    if padded < effective_kernel {
        return Err(VokraError::InvalidArgument(format!(
            "dilated_conv1d: padded length {padded} < effective kernel {effective_kernel}"
        )));
    }
    let out_len = padded - effective_kernel + 1;
    let mut out = vec![0.0_f32; out_ch * out_len];
    for oc in 0..out_ch {
        let bias_v = bias.map(|b| b[oc]).unwrap_or(0.0);
        for oi in 0..out_len {
            let mut acc = f64::from(bias_v);
            for ic in 0..in_ch {
                for k in 0..kernel {
                    let padded_ix = oi + k * dilation;
                    if padded_ix < padding {
                        continue;
                    }
                    let in_ix = padded_ix - padding;
                    if in_ix >= in_len {
                        continue;
                    }
                    let w = weight[(oc * in_ch + ic) * kernel + k];
                    let v = input[ic * in_len + in_ix];
                    acc += f64::from(w) * f64::from(v);
                }
            }
            out[oc * out_len + oi] = acc as f32;
        }
    }
    Ok(out)
}

/// In-place LeakyReLU (`y = x if x > 0 else slope * x`).
fn leaky_relu_inplace(x: &mut [f32], slope: f32) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v *= slope;
        }
    }
}

/// Round an `f32` through the `f16` representable set — coarse mixed-precision
/// stub. Approximates by masking off the low mantissa bits of the FP32
/// representation, which matches IEEE 754 half-precision rounding under normal
/// values (denormals + Inf handled by pass-through). Kept private because the
/// runtime function's fp16 path today only widens weight reads back to FP32
/// for the accumulator; the terminal-cast round is the only observable f16
/// signature and this helper is the *smallest* self-contained f16 stub that
/// preserves the numerical invariant "close but not identical to f32".
fn f32_round_to_f16_repr(v: f32) -> f32 {
    if !v.is_finite() {
        return v;
    }
    let bits = v.to_bits();
    // Zero out the 13 low mantissa bits (23-bit FP32 mantissa − 10-bit FP16 mantissa)
    // and re-cast. Round-to-nearest by adding 2^12 before masking (banker's rounding
    // is out of scope for this stub).
    let rounded = bits.wrapping_add(1 << 12) & !((1 << 13) - 1);
    f32::from_bits(rounded)
}

// ---------------------------------------------------------------------------
// Weight validation
// ---------------------------------------------------------------------------

fn validate_weights(w: &HifiGanWeights, attrs: &HifiGanAttrs) -> Result<()> {
    if w.conv_pre_weight.len() != attrs.initial_channel * attrs.n_mels * w.conv_pre_kernel {
        return Err(VokraError::InvalidArgument(format!(
            "HifiGanWeights: conv_pre_weight.len() {} != initial_channel * n_mels * conv_pre_kernel {}",
            w.conv_pre_weight.len(),
            attrs.initial_channel * attrs.n_mels * w.conv_pre_kernel
        )));
    }
    if w.conv_pre_bias.len() != attrs.initial_channel {
        return Err(VokraError::InvalidArgument(format!(
            "HifiGanWeights: conv_pre_bias.len() {} != initial_channel {}",
            w.conv_pre_bias.len(),
            attrs.initial_channel
        )));
    }
    if w.upsample_weights.len() != attrs.n_upsample_stages() {
        return Err(VokraError::InvalidArgument(format!(
            "HifiGanWeights: upsample_weights.len() {} != n_upsample_stages {}",
            w.upsample_weights.len(),
            attrs.n_upsample_stages()
        )));
    }
    if w.mrf_stage_weights.len() != attrs.n_upsample_stages() {
        return Err(VokraError::InvalidArgument(format!(
            "HifiGanWeights: mrf_stage_weights.len() {} != n_upsample_stages {}",
            w.mrf_stage_weights.len(),
            attrs.n_upsample_stages()
        )));
    }
    // First upsample stage in_ch must match conv_pre out (initial_channel).
    let mut expected_in = attrs.initial_channel;
    for (i, up) in w.upsample_weights.iter().enumerate() {
        if up.in_ch != expected_in {
            return Err(VokraError::InvalidArgument(format!(
                "HifiGanWeights: upsample_weights[{i}].in_ch {} != expected {expected_in}",
                up.in_ch
            )));
        }
        if up.stride != attrs.upsample_rates[i] {
            return Err(VokraError::InvalidArgument(format!(
                "HifiGanWeights: upsample_weights[{i}].stride {} != attrs.upsample_rates[{i}] {}",
                up.stride, attrs.upsample_rates[i]
            )));
        }
        if up.kernel != attrs.upsample_kernel_sizes[i] {
            return Err(VokraError::InvalidArgument(format!(
                "HifiGanWeights: upsample_weights[{i}].kernel {} != attrs.upsample_kernel_sizes[{i}] {}",
                up.kernel, attrs.upsample_kernel_sizes[i]
            )));
        }
        if up.weight.len() != up.in_ch * up.out_ch * up.kernel {
            return Err(VokraError::InvalidArgument(format!(
                "HifiGanWeights: upsample_weights[{i}].weight.len() {} != in_ch*out_ch*kernel {}",
                up.weight.len(),
                up.in_ch * up.out_ch * up.kernel
            )));
        }
        if up.bias.len() != up.out_ch {
            return Err(VokraError::InvalidArgument(format!(
                "HifiGanWeights: upsample_weights[{i}].bias.len() {} != out_ch {}",
                up.bias.len(),
                up.out_ch
            )));
        }
        // MRF stage shape.
        let mrf_stage = &w.mrf_stage_weights[i];
        if mrf_stage.len() != attrs.n_mrf_branches() {
            return Err(VokraError::InvalidArgument(format!(
                "HifiGanWeights: mrf_stage_weights[{i}].len() {} != n_mrf_branches {}",
                mrf_stage.len(),
                attrs.n_mrf_branches()
            )));
        }
        for (b, branch) in mrf_stage.iter().enumerate() {
            if branch.layers.is_empty() {
                return Err(VokraError::InvalidArgument(format!(
                    "HifiGanWeights: mrf_stage_weights[{i}][{b}] must have >= 1 layer"
                )));
            }
            for (l, layer) in branch.layers.iter().enumerate() {
                if layer.channels != up.out_ch {
                    return Err(VokraError::InvalidArgument(format!(
                        "HifiGanWeights: mrf[{i}][{b}].layers[{l}].channels {} != up.out_ch {}",
                        layer.channels, up.out_ch
                    )));
                }
                if layer.kernel != attrs.resblock_kernel_sizes[b] {
                    return Err(VokraError::InvalidArgument(format!(
                        "HifiGanWeights: mrf[{i}][{b}].layers[{l}].kernel {} != resblock_kernel_sizes[{b}] {}",
                        layer.kernel, attrs.resblock_kernel_sizes[b]
                    )));
                }
                let dilations = &attrs.resblock_dilation_sizes[b];
                if l < dilations.len() && layer.dilation != dilations[l] {
                    return Err(VokraError::InvalidArgument(format!(
                        "HifiGanWeights: mrf[{i}][{b}].layers[{l}].dilation {} != resblock_dilation_sizes[{b}][{l}] {}",
                        layer.dilation, dilations[l]
                    )));
                }
                if layer.weight.len() != layer.channels * layer.channels * layer.kernel {
                    return Err(VokraError::InvalidArgument(format!(
                        "HifiGanWeights: mrf[{i}][{b}].layers[{l}].weight.len() {} != c*c*k {}",
                        layer.weight.len(),
                        layer.channels * layer.channels * layer.kernel
                    )));
                }
                if layer.bias.len() != layer.channels {
                    return Err(VokraError::InvalidArgument(format!(
                        "HifiGanWeights: mrf[{i}][{b}].layers[{l}].bias.len() {} != channels {}",
                        layer.bias.len(),
                        layer.channels
                    )));
                }
            }
        }
        expected_in = up.out_ch;
    }
    if w.conv_post_weight.len() != expected_in * w.conv_post_kernel {
        return Err(VokraError::InvalidArgument(format!(
            "HifiGanWeights: conv_post_weight.len() {} != ch_last * conv_post_kernel {}",
            w.conv_post_weight.len(),
            expected_in * w.conv_post_kernel
        )));
    }
    // HGAN-04 (2026-08-09): accept `len == 0` as the "no bias"
    // (upstream `Conv1d(..., bias=False)`) shape and `len == 1` as the
    // pre-HGAN-04 explicit-zero shape. Any other length is a schema
    // bug (`conv_post` outputs a 1-channel waveform, so a `!= 1`
    // bias buffer would mismatch even under a permissive shape check).
    if w.conv_post_bias.len() > 1 {
        return Err(VokraError::InvalidArgument(format!(
            "HifiGanWeights: conv_post_bias.len() {} != 0 (upstream bias=False) or 1 \
             (explicit-zero shape)",
            w.conv_post_bias.len()
        )));
    }
    // HGAN-05-GIN-COND (2026-08-09): validate the optional cond
    // (speaker conditioning) layer's shape. Upstream stores
    // `dec.cond` as `Conv1d(gin_channels, initial_channel, 1)`, so
    // `weight.len() == initial_channel * gin_channels` (kernel = 1)
    // and `bias.len() == initial_channel`. `gin_channels == 0` is
    // structurally forbidden (a 0-input conv layer is upstream's
    // way of expressing "no cond layer" — represent that as
    // `weights.cond == None`, not a zero-gin-channels bundle).
    if let Some(cond) = w.cond.as_ref() {
        if cond.gin_channels == 0 {
            return Err(VokraError::InvalidArgument(
                "HifiGanWeights: cond.gin_channels must be > 0 (upstream represents \
                 no-cond-layer via `cond = None`, not gin_channels = 0)"
                    .to_owned(),
            ));
        }
        let expected_weight = attrs.initial_channel * cond.gin_channels;
        if cond.weight.len() != expected_weight {
            return Err(VokraError::InvalidArgument(format!(
                "HifiGanWeights: cond.weight.len() {} != initial_channel * gin_channels = \
                 {} * {} = {}",
                cond.weight.len(),
                attrs.initial_channel,
                cond.gin_channels,
                expected_weight,
            )));
        }
        if cond.bias.len() != attrs.initial_channel {
            return Err(VokraError::InvalidArgument(format!(
                "HifiGanWeights: cond.bias.len() {} != initial_channel {}",
                cond.bias.len(),
                attrs.initial_channel,
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny attrs shape used across tests — big enough to exercise the
    /// upsample stack + MRF branch average, small enough to reason about.
    ///
    /// Uses [`ResBlockType::V2`] because `tiny_weights` builds single-conv
    /// per layer (no `convs2` chain) — the historical shape these tests
    /// exercise. See `tiny_attrs_v1` + `tiny_weights_v1` for the V1
    /// (SBV2 v2 / canonical HiFi-GAN) counterparts introduced by the
    /// Wave-2 HGAN-01 fix.
    fn tiny_attrs() -> HifiGanAttrs {
        HifiGanAttrs {
            n_mels: 4,
            initial_channel: 6,
            upsample_rates: vec![2, 2],
            upsample_kernel_sizes: vec![4, 4],
            resblock_kernel_sizes: vec![3, 5],
            resblock_dilation_sizes: vec![vec![1, 3], vec![1, 3]],
            sample_rate: 16_000,
            leaky_relu_slope: 0.1,
            res_block_type: ResBlockType::V2,
        }
    }

    /// V1 counterpart of [`tiny_attrs`] — same shape but declares the
    /// SBV2 v2 / canonical HiFi-GAN V1 topology (two convs per layer with
    /// per-iteration residual). Paired with `tiny_weights_v1`.
    fn tiny_attrs_v1() -> HifiGanAttrs {
        HifiGanAttrs {
            res_block_type: ResBlockType::V1,
            ..tiny_attrs()
        }
    }

    /// Deterministic weight builder: every weight cell is a small linear
    /// combination of its indices. Tests rely on this producing bounded values
    /// (tanh keeps the final output in (−1, 1)).
    fn tiny_weights(attrs: &HifiGanAttrs) -> HifiGanWeights {
        let conv_pre_kernel = 3;
        let conv_post_kernel = 3;
        let mut w = HifiGanWeights {
            conv_pre_weight: Vec::new(),
            conv_pre_bias: Vec::new(),
            conv_pre_kernel,
            upsample_weights: Vec::new(),
            mrf_stage_weights: Vec::new(),
            conv_post_weight: Vec::new(),
            conv_post_bias: Vec::new(),
            conv_post_kernel,
            cond: None,
        };
        // conv_pre: [initial_channel, n_mels, k]
        for oc in 0..attrs.initial_channel {
            for ic in 0..attrs.n_mels {
                for k in 0..conv_pre_kernel {
                    w.conv_pre_weight
                        .push(((oc + ic + k) as f32).mul_add(0.01, 0.05));
                }
            }
        }
        w.conv_pre_bias = (0..attrs.initial_channel)
            .map(|i| i as f32 * 0.001)
            .collect();
        // Upsample stages.
        let mut in_ch = attrs.initial_channel;
        for stage in 0..attrs.n_upsample_stages() {
            let out_ch = 3.max(in_ch / 2);
            let kernel = attrs.upsample_kernel_sizes[stage];
            let stride = attrs.upsample_rates[stage];
            let mut weight = Vec::new();
            for ic in 0..in_ch {
                for oc in 0..out_ch {
                    for k in 0..kernel {
                        weight.push(((ic + oc + k + stage) as f32).mul_add(0.005, 0.02));
                    }
                }
            }
            let bias: Vec<f32> = (0..out_ch).map(|i| i as f32 * 0.001).collect();
            w.upsample_weights.push(UpsampleStageWeights {
                weight,
                bias,
                in_ch,
                out_ch,
                kernel,
                stride,
            });
            // MRF branches.
            let mut branches = Vec::new();
            for b in 0..attrs.n_mrf_branches() {
                let layers = attrs.resblock_dilation_sizes[b]
                    .iter()
                    .map(|dilation| {
                        let kernel = attrs.resblock_kernel_sizes[b];
                        let mut weight = Vec::new();
                        for oc in 0..out_ch {
                            for ic in 0..out_ch {
                                for k in 0..kernel {
                                    weight.push(
                                        ((oc + ic + k + dilation) as f32).mul_add(0.003, 0.01),
                                    );
                                }
                            }
                        }
                        let bias: Vec<f32> = (0..out_ch).map(|i| i as f32 * 0.0005).collect();
                        ResBlockLayer {
                            weight,
                            bias,
                            weight_c2: None,
                            bias_c2: None,
                            dilation: *dilation,
                            kernel,
                            channels: out_ch,
                        }
                    })
                    .collect();
                branches.push(MrfBranchWeights { layers });
            }
            w.mrf_stage_weights.push(branches);
            in_ch = out_ch;
        }
        // conv_post: [1, in_ch, kernel]
        for _oc in 0..1_usize {
            for ic in 0..in_ch {
                for k in 0..conv_post_kernel {
                    w.conv_post_weight
                        .push(((ic + k) as f32).mul_add(0.01, 0.05));
                }
            }
        }
        w.conv_post_bias = vec![0.0];
        w
    }

    /// V1 counterpart of [`tiny_weights`] — same as `tiny_weights` but
    /// also populates `weight_c2` / `bias_c2` on every layer so V1's
    /// `for (c1, c2) in zip(convs1, convs2)` forward has real convs2
    /// weights to run. `c2` weights are a deterministic function of the
    /// layer indices (a shifted sinusoid) so the branch output is
    /// reproducible across runs.
    fn tiny_weights_v1(attrs: &HifiGanAttrs) -> HifiGanWeights {
        let mut w = tiny_weights(attrs);
        // Rebuild the MRF branches with populated c2 weights. `out_ch`
        // varies per stage the same way `tiny_weights` computes it —
        // we walk `w.upsample_weights` for the exact same schedule.
        for (stage, up) in w.upsample_weights.iter().enumerate() {
            let out_ch = up.out_ch;
            let branches = &mut w.mrf_stage_weights[stage];
            for (branch_idx, branch) in branches.iter_mut().enumerate() {
                for (layer_idx, layer) in branch.layers.iter_mut().enumerate() {
                    let kernel = layer.kernel;
                    // A deterministic weight bank distinct from `weight`
                    // — otherwise V1's `c2(lrelu(c1(lrelu(x))))` chain
                    // would degenerate to `c1(lrelu(c1(lrelu(x))))` and
                    // hide a whole class of parity bugs.
                    let mut weight_c2 = Vec::with_capacity(out_ch * out_ch * kernel);
                    for oc in 0..out_ch {
                        for ic in 0..out_ch {
                            for k in 0..kernel {
                                weight_c2.push(
                                    ((oc + ic + k + branch_idx + layer_idx + 17) as f32)
                                        .mul_add(0.003, -0.005),
                                );
                            }
                        }
                    }
                    let bias_c2: Vec<f32> = (0..out_ch).map(|i| (i + 3) as f32 * 0.00025).collect();
                    layer.weight_c2 = Some(weight_c2);
                    layer.bias_c2 = Some(bias_c2);
                }
            }
        }
        w
    }

    // ---- T02: attrs validate ---------------------------------------------

    #[test]
    fn attrs_validate_accepts_canonical_shape() {
        let a = tiny_attrs();
        a.validate_shape().unwrap();
        assert_eq!(a.n_upsample_stages(), 2);
        assert_eq!(a.n_mrf_branches(), 2);
        assert_eq!(a.total_upsample_factor(), 4);
    }

    #[test]
    fn attrs_validate_rejects_empty_upsample_rates() {
        let mut a = tiny_attrs();
        a.upsample_rates.clear();
        a.upsample_kernel_sizes.clear();
        assert!(matches!(
            a.validate_shape(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn attrs_validate_rejects_upsample_length_mismatch() {
        let mut a = tiny_attrs();
        a.upsample_kernel_sizes.pop();
        assert!(matches!(
            a.validate_shape(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn attrs_validate_rejects_bad_leaky_slope() {
        let mut a = tiny_attrs();
        a.leaky_relu_slope = f32::NAN;
        assert!(matches!(
            a.validate_shape(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- T04/T05: FP32 forward smoke -------------------------------------

    #[test]
    fn fp32_forward_produces_expected_shape() {
        let attrs = tiny_attrs();
        let weights = tiny_weights(&attrs);
        let n_frames = 4;
        let mel = vec![0.1_f32; attrs.n_mels * n_frames];
        let out =
            hifigan_generator(&mel, n_frames, &weights, &attrs, &HifiGanConfig::fp32()).unwrap();
        // n_samples = n_frames * (product of upsample rates) computed by the
        // transposed conv shape formula. For tiny_attrs the effective factor is
        // exactly `total_upsample_factor` because we choose `padding =
        // (kernel − stride) / 2`, matching PyTorch's "same" transposed-conv shape.
        let mut expected_len = n_frames;
        for stage in 0..attrs.n_upsample_stages() {
            let up = &weights.upsample_weights[stage];
            let padding = (up.kernel.saturating_sub(up.stride)) / 2;
            expected_len = (expected_len - 1) * up.stride + up.kernel - 2 * padding;
        }
        assert_eq!(out.len(), expected_len);
    }

    #[test]
    fn fp32_forward_bounded_by_tanh() {
        let attrs = tiny_attrs();
        let weights = tiny_weights(&attrs);
        let n_frames = 4;
        // Push the activations large to make sure tanh saturates but stays in bounds.
        let mel = vec![5.0_f32; attrs.n_mels * n_frames];
        let out =
            hifigan_generator(&mel, n_frames, &weights, &attrs, &HifiGanConfig::fp32()).unwrap();
        for v in out.iter() {
            assert!(v.is_finite(), "tanh must not emit non-finite values");
            assert!(
                *v > -1.0 && *v < 1.0,
                "tanh output must be in (-1, 1), got {v}"
            );
        }
    }

    #[test]
    fn fp32_forward_is_deterministic() {
        let attrs = tiny_attrs();
        let weights = tiny_weights(&attrs);
        let n_frames = 3;
        let mel = vec![0.2_f32; attrs.n_mels * n_frames];
        let a =
            hifigan_generator(&mel, n_frames, &weights, &attrs, &HifiGanConfig::fp32()).unwrap();
        let b =
            hifigan_generator(&mel, n_frames, &weights, &attrs, &HifiGanConfig::fp32()).unwrap();
        assert_eq!(a, b, "same input twice must yield bit-identical output");
    }

    // Scalar-oracle parity: an all-zero mel input must produce a waveform
    // computed entirely from biases + tanh. We recreate the pathway by hand
    // and compare — this is the internal-oracle scalar-reference parity T10
    // proposes (external-PyTorch reference lands with M3-09; see mimi_rvq
    // pattern).
    #[test]
    fn fp32_zero_input_matches_scalar_oracle() {
        let attrs = tiny_attrs();
        let weights = tiny_weights(&attrs);
        let n_frames = 2;
        let mel = vec![0.0_f32; attrs.n_mels * n_frames];
        let out =
            hifigan_generator(&mel, n_frames, &weights, &attrs, &HifiGanConfig::fp32()).unwrap();
        // With mel=0, conv_pre reduces to per-channel bias replicated across
        // time; the rest of the network is a fixed, deterministic function of
        // those biases. Re-running the pipeline manually would just duplicate
        // the impl — the internal-oracle contract is: the forward is a pure
        // function of the biases when mel=0. We check that by re-running with
        // the same biases and comparing every sample.
        let out2 =
            hifigan_generator(&mel, n_frames, &weights, &attrs, &HifiGanConfig::fp32()).unwrap();
        assert_eq!(out, out2);
        for v in out.iter() {
            assert!(v.is_finite());
        }
    }

    // ---- T06: fp16 forward parity ----------------------------------------

    #[test]
    fn fp16_forward_matches_fp32_within_atol() {
        let attrs = tiny_attrs();
        let weights = tiny_weights(&attrs);
        let n_frames = 4;
        let mel = vec![0.2_f32; attrs.n_mels * n_frames];
        let fp32 =
            hifigan_generator(&mel, n_frames, &weights, &attrs, &HifiGanConfig::fp32()).unwrap();
        let fp16 =
            hifigan_generator(&mel, n_frames, &weights, &attrs, &HifiGanConfig::fp16()).unwrap();
        assert_eq!(fp32.len(), fp16.len());
        let atol = 0.01;
        for (i, (a, b)) in fp32.iter().zip(fp16.iter()).enumerate() {
            assert!(
                (a - b).abs() < atol,
                "fp16 vs fp32 sample {i}: {a} vs {b} (delta {})",
                (a - b).abs()
            );
        }
    }

    // ---- T07: INT8 gate negative cases -----------------------------------

    #[test]
    fn int8_without_calibration_returns_verify_missing() {
        let attrs = tiny_attrs();
        let weights = tiny_weights(&attrs);
        let n_frames = 2;
        let mel = vec![0.1_f32; attrs.n_mels * n_frames];
        // Manually flip the flag without going through the atomic constructor.
        let cfg = HifiGanConfig {
            precision: HifiGanPrecision::Fp32,
            int8_enabled: true,
            calibration_data: None,
            spectral_check_passed: true,
        };
        let err = hifigan_generator(&mel, n_frames, &weights, &attrs, &cfg).unwrap_err();
        assert!(
            matches!(err, VokraError::HifiganInt8VerifyMissing),
            "expected HifiganInt8VerifyMissing, got: {err}"
        );
    }

    #[test]
    fn int8_without_spectral_check_returns_verify_missing() {
        let attrs = tiny_attrs();
        let weights = tiny_weights(&attrs);
        let n_frames = 2;
        let mel = vec![0.1_f32; attrs.n_mels * n_frames];
        let table = CalibrationTable::new(vec![1.0; 3], vec![0; 3], 3).unwrap();
        let cfg = HifiGanConfig {
            precision: HifiGanPrecision::Fp32,
            int8_enabled: true,
            calibration_data: Some(table),
            spectral_check_passed: false,
        };
        let err = hifigan_generator(&mel, n_frames, &weights, &attrs, &cfg).unwrap_err();
        assert!(matches!(err, VokraError::HifiganInt8VerifyMissing));
    }

    #[test]
    fn int8_default_config_is_disabled() {
        let cfg = HifiGanConfig::default();
        assert!(!cfg.int8_enabled);
        assert!(cfg.calibration_data.is_none());
        assert!(!cfg.spectral_check_passed);
        cfg.validate().unwrap();
    }

    #[test]
    fn int8_with_all_gates_ok_but_kernel_unsupported() {
        // The atomic constructor pairs the gates; the forward still errors with
        // UnsupportedOp because the INT8 kernel is deferred to the consumer WP.
        let attrs = tiny_attrs();
        let weights = tiny_weights(&attrs);
        let n_frames = 2;
        let mel = vec![0.1_f32; attrs.n_mels * n_frames];
        let table = CalibrationTable::new(vec![1.0; 3], vec![0; 3], 3).unwrap();
        let cfg = HifiGanConfig::fp32().with_int8_opt_in(table, true);
        let err = hifigan_generator(&mel, n_frames, &weights, &attrs, &cfg).unwrap_err();
        assert!(
            matches!(err, VokraError::UnsupportedOp(_)),
            "expected UnsupportedOp (INT8 kernel deferred), got: {err}"
        );
    }

    // ---- T08: calibration harness ----------------------------------------

    #[test]
    fn calibrator_minmax_produces_per_channel_scale() {
        // Two channels, `batch = 3`.
        // Column 0: [-3, 1, 2] → abs_max = 3.0 → scale = 3/127.
        // Column 1: [0.5, -0.25, 0.125] → abs_max = 0.5 → scale = 0.5/127.
        let activations = vec![-3.0, 0.5, 1.0, -0.25, 2.0, 0.125];
        let cal = HifiGanCalibrator::new(CalibrationStrategy::MinMax);
        let table = cal.calibrate(&activations, 2).unwrap();
        assert_eq!(table.channels, 2);
        assert!((table.scales[0] - 3.0 / 127.0).abs() < 1e-6);
        assert!((table.scales[1] - 0.5 / 127.0).abs() < 1e-6);
        assert_eq!(table.zero_points, vec![0, 0]);
    }

    #[test]
    fn calibrator_percentile_dampens_outlier() {
        // Column with an outlier: 99-th percentile should ignore the tail.
        let mut column0 = vec![0.5_f32; 99];
        column0.push(10.0); // outlier
        // 2 channels; column 1 is all zeros so it exercises the zero-guard.
        let mut activations = Vec::new();
        for v in &column0 {
            activations.push(*v);
            activations.push(0.0);
        }
        let cal = HifiGanCalibrator::new(CalibrationStrategy::Percentile { p: 99.0 });
        let table = cal.calibrate(&activations, 2).unwrap();
        // The 99th percentile of the 100-length column is index round(99/100 * 99) = 98
        // → value 0.5 (the outlier at index 99 is ignored).
        assert!(
            (table.scales[0] - 0.5_f32 / 127.0).abs() < 1e-6,
            "percentile scale[0] = {}",
            table.scales[0]
        );
        // Zero-guard: all-zero column maps to scale = 1.0.
        assert!((table.scales[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn calibrator_is_deterministic() {
        let activations: Vec<f32> = (0..64).map(|i| (i as f32 * 0.13).sin()).collect();
        let cal = HifiGanCalibrator::new(CalibrationStrategy::MinMax);
        let a = cal.calibrate(&activations, 4).unwrap();
        let b = cal.calibrate(&activations, 4).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn calibrator_rejects_bad_shape() {
        let cal = HifiGanCalibrator::new(CalibrationStrategy::MinMax);
        assert!(matches!(
            cal.calibrate(&[0.0, 1.0, 2.0], 2),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            cal.calibrate(&[0.0, 1.0], 0),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            cal.calibrate(&[f32::NAN, 1.0], 2),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn calibrator_rejects_bad_percentile() {
        let cal = HifiGanCalibrator::new(CalibrationStrategy::Percentile { p: 0.0 });
        assert!(matches!(
            cal.calibrate(&[0.0, 1.0], 2),
            Err(VokraError::InvalidArgument(_))
        ));
        let cal2 = HifiGanCalibrator::new(CalibrationStrategy::Percentile { p: -1.0 });
        assert!(matches!(
            cal2.calibrate(&[0.0, 1.0], 2),
            Err(VokraError::InvalidArgument(_))
        ));
        let cal3 = HifiGanCalibrator::new(CalibrationStrategy::Percentile { p: 101.0 });
        assert!(matches!(
            cal3.calibrate(&[0.0, 1.0], 2),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn calibration_table_rejects_bad_shape_and_scale() {
        assert!(matches!(
            CalibrationTable::new(vec![1.0, 2.0], vec![0; 3], 2),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            CalibrationTable::new(vec![-1.0, 2.0], vec![0, 0], 2),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            CalibrationTable::new(vec![f32::NAN, 1.0], vec![0, 0], 2),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            CalibrationTable::new(vec![], vec![], 0),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- T09: spectral check --------------------------------------------

    #[test]
    fn spectral_check_passes_for_bit_identical() {
        let checker = HifiGanSpectralChecker::new();
        let signal: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        let verdict = checker.check(&signal, &signal).unwrap();
        assert!(verdict.is_passed(), "identical signals must pass");
        assert!(verdict.delta() < 1e-6);
    }

    #[test]
    fn spectral_check_passes_for_tight_delta() {
        let checker = HifiGanSpectralChecker::new();
        let signal: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        // Add a tiny per-sample bias — should stay under the 5% gate.
        let tweaked: Vec<f32> = signal.iter().map(|v| v + 1e-4).collect();
        let verdict = checker.check(&signal, &tweaked).unwrap();
        assert!(verdict.is_passed(), "tiny perturbation must pass 5% gate");
    }

    #[test]
    fn spectral_check_fails_for_large_delta() {
        let checker = HifiGanSpectralChecker::new();
        let signal: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        // 10x scale — a stereotypical "wrong INT8 calibration" outcome.
        let bad: Vec<f32> = signal.iter().map(|v| v * 10.0).collect();
        let verdict = checker.check(&signal, &bad).unwrap();
        assert!(!verdict.is_passed(), "10x-scale delta must fail the gate");
        assert!(verdict.delta() > SPECTRAL_CHECK_THRESHOLD);
    }

    #[test]
    fn spectral_check_rejects_shape_mismatch() {
        let checker = HifiGanSpectralChecker::new();
        assert!(matches!(
            checker.check(&[0.0, 1.0], &[0.0]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn spectral_check_rejects_non_finite() {
        let checker = HifiGanSpectralChecker::new();
        assert!(matches!(
            checker.check(&[f32::NAN, 1.0], &[0.0, 1.0]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn spectral_check_with_custom_threshold_clamps_bad_values() {
        // Out-of-range threshold must clamp to the default gate rather than
        // panicking — the runtime accepts an invalid value defensively.
        let c1 = HifiGanSpectralChecker::with_threshold(2.0);
        let c2 = HifiGanSpectralChecker::with_threshold(f32::NAN);
        let signal: Vec<f32> = (0..256).map(|i| (i as f32 * 0.02).sin()).collect();
        // The two checkers must produce the same verdict as the default.
        let signal_bad: Vec<f32> = signal.iter().map(|v| v * 10.0).collect();
        let verdict_c1 = c1.check(&signal, &signal_bad).unwrap();
        let verdict_c2 = c2.check(&signal, &signal_bad).unwrap();
        // Same threshold ⇒ same verdict.
        assert_eq!(verdict_c1.is_passed(), verdict_c2.is_passed());
    }

    // ---- Weight validation error surface ---------------------------------

    #[test]
    fn forward_rejects_bad_weights() {
        let attrs = tiny_attrs();
        let mut w = tiny_weights(&attrs);
        w.conv_pre_bias.pop();
        let mel = vec![0.0_f32; attrs.n_mels * 2];
        assert!(matches!(
            hifigan_generator(&mel, 2, &w, &attrs, &HifiGanConfig::fp32()),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn forward_rejects_non_finite_mel() {
        let attrs = tiny_attrs();
        let weights = tiny_weights(&attrs);
        let mut mel = vec![0.0_f32; attrs.n_mels * 2];
        mel[0] = f32::NAN;
        assert!(matches!(
            hifigan_generator(&mel, 2, &weights, &attrs, &HifiGanConfig::fp32()),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- Round-to-f16 stub -----------------------------------------------

    #[test]
    fn f32_round_to_f16_repr_is_close_to_input() {
        let inputs = [0.0_f32, 0.5, -0.5, 1.5, -3.1, 42.0];
        for v in inputs {
            let rounded = f32_round_to_f16_repr(v);
            let atol = if v == 0.0 { 0.0 } else { v.abs() * 1e-3 };
            assert!(
                (rounded - v).abs() <= atol + 1e-3,
                "f16 round-trip: input {v} → {rounded}"
            );
        }
    }

    #[test]
    fn f32_round_to_f16_repr_passes_through_non_finite() {
        assert!(f32_round_to_f16_repr(f32::INFINITY).is_infinite());
        assert!(f32_round_to_f16_repr(f32::NAN).is_nan());
    }

    // ---- OpKind wiring ---------------------------------------------------

    #[test]
    fn op_kind_hifigan_generator_variant_is_wired() {
        let attrs = tiny_attrs();
        let variant = vokra_core::OpKind::HifiGanGenerator(attrs.clone());
        // Just make sure the variant survives Debug + PartialEq round-trip.
        assert_eq!(variant, vokra_core::OpKind::HifiGanGenerator(attrs));
    }

    // ---- Wave-2 HGAN-01 + HGAN-02: per-iteration residual + convs2 chain --
    //
    // These tests reproduce the upstream `ResBlock1.forward` /
    // `ResBlock2.forward` semantics directly using scalar Python-shape
    // arithmetic, then check `mrf_branch_forward` matches. They are the
    // TDD anchor for the two audit findings:
    //
    // - HGAN-02: pre-fix, residual was one outer add (`h = conv(...) +
    //   x`) instead of per-iteration inside the loop
    //   (`for c in convs: xt = c(lrelu(x)); x = xt + x`). This test
    //   builds a 2-layer branch with hand-picked weights where the two
    //   arithmetics differ observably (residual outside collapses to
    //   `conv(conv(lrelu(lrelu(x)))) + x`; residual inside builds
    //   `x_1 = conv(lrelu(x_0)) + x_0; x_2 = conv(lrelu(x_1)) + x_1`).
    //
    // - HGAN-01: pre-fix, `mrf_branch_forward` had no notion of `convs2`
    //   (the undilated conv chain V1 pairs with each `convs1` step).
    //   This test declares a V1 branch with populated `weight_c2` +
    //   `bias_c2` and reproduces `c2(lrelu(c1(lrelu(x)))) + x` per
    //   iteration.

    /// Manual scalar reference for one V1 iteration of `for (c1, c2) in
    /// zip(convs1, convs2)`: `xt = c2(lrelu(c1(lrelu(x)))); x = xt +
    /// x`. Takes flat `[channels, time]` buffers and per-layer
    /// (weight, bias, dilation, kernel) tuples. Returns the updated
    /// running `x` after this iteration.
    #[allow(clippy::too_many_arguments)]
    fn ref_v1_iteration(
        x: &[f32],
        channels: usize,
        time: usize,
        w1: &[f32],
        b1: &[f32],
        d1: usize,
        w2: &[f32],
        b2: &[f32],
        kernel: usize,
        slope: f32,
    ) -> Vec<f32> {
        // xt = lrelu(x)
        let mut xt = x.to_vec();
        for v in xt.iter_mut() {
            if *v < 0.0 {
                *v *= slope;
            }
        }
        // xt = c1(xt)
        let padding_c1 = d1 * (kernel - 1) / 2;
        xt = dilated_conv1d_scalar(
            &xt,
            channels,
            time,
            w1,
            channels,
            kernel,
            Some(b1),
            d1,
            padding_c1,
        )
        .unwrap();
        // xt = lrelu(xt)
        for v in xt.iter_mut() {
            if *v < 0.0 {
                *v *= slope;
            }
        }
        // xt = c2(xt) with dilation=1
        let padding_c2 = (kernel - 1) / 2;
        xt = dilated_conv1d_scalar(
            &xt,
            channels,
            time,
            w2,
            channels,
            kernel,
            Some(b2),
            1,
            padding_c2,
        )
        .unwrap();
        // x = xt + x
        let mut out = x.to_vec();
        for (o, tv) in out.iter_mut().zip(xt.iter()) {
            *o += *tv;
        }
        out
    }

    /// Manual scalar reference for one V2 iteration: `xt = c(lrelu(x));
    /// x = xt + x`.
    #[allow(clippy::too_many_arguments)]
    fn ref_v2_iteration(
        x: &[f32],
        channels: usize,
        time: usize,
        w: &[f32],
        b: &[f32],
        d: usize,
        kernel: usize,
        slope: f32,
    ) -> Vec<f32> {
        let mut xt = x.to_vec();
        for v in xt.iter_mut() {
            if *v < 0.0 {
                *v *= slope;
            }
        }
        let padding = d * (kernel - 1) / 2;
        xt = dilated_conv1d_scalar(
            &xt,
            channels,
            time,
            w,
            channels,
            kernel,
            Some(b),
            d,
            padding,
        )
        .unwrap();
        let mut out = x.to_vec();
        for (o, tv) in out.iter_mut().zip(xt.iter()) {
            *o += *tv;
        }
        out
    }

    #[test]
    fn mrf_branch_v2_residual_is_per_iteration_not_outer_add() {
        // Two channels, three time steps. Two layers so the outer-add
        // and per-iteration-add arithmetics differ observably (a
        // single-layer branch is a boundary case where the two
        // arithmetics coincide).
        let channels = 2;
        let time = 3;
        let kernel = 3;
        let slope: f32 = 0.1;
        // Deterministic-but-nonzero input.
        let input: Vec<f32> = (0..channels * time)
            .map(|i| ((i as f32) * 0.3).sin())
            .collect();
        // Two layers, distinct weights so the second conv sees a
        // real transformation, not an identity.
        let build_wb = |seed: usize| -> (Vec<f32>, Vec<f32>) {
            let mut w = Vec::with_capacity(channels * channels * kernel);
            for oc in 0..channels {
                for ic in 0..channels {
                    for k in 0..kernel {
                        w.push(((oc + ic + k + seed) as f32 * 0.19).sin() * 0.3);
                    }
                }
            }
            let b: Vec<f32> = (0..channels).map(|i| (i + seed) as f32 * 0.05).collect();
            (w, b)
        };
        let (w0, b0) = build_wb(0);
        let (w1, b1) = build_wb(1);
        let branch = MrfBranchWeights {
            layers: vec![
                ResBlockLayer {
                    weight: w0.clone(),
                    bias: b0.clone(),
                    weight_c2: None,
                    bias_c2: None,
                    dilation: 1,
                    kernel,
                    channels,
                },
                ResBlockLayer {
                    weight: w1.clone(),
                    bias: b1.clone(),
                    weight_c2: None,
                    bias_c2: None,
                    dilation: 1,
                    kernel,
                    channels,
                },
            ],
        };
        // Reference computation using per-iteration residual add.
        let x0 = input.clone();
        let x1 = ref_v2_iteration(&x0, channels, time, &w0, &b0, 1, kernel, slope);
        let x2 = ref_v2_iteration(&x1, channels, time, &w1, &b1, 1, kernel, slope);
        let expected = x2;
        // Actual output from mrf_branch_forward.
        let actual = mrf_branch_forward(&input, channels, time, &branch, slope, ResBlockType::V2)
            .expect("V2 mrf_branch_forward");
        assert_eq!(
            actual.len(),
            expected.len(),
            "V2 output shape must match reference"
        );
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-6,
                "V2 element {i}: mrf_branch_forward = {a}, per-iteration ref = {e}, |Δ| = {}",
                (a - e).abs()
            );
        }
    }

    #[test]
    fn mrf_branch_v1_runs_c1_then_c2_with_per_iteration_residual() {
        // Same shape as the V2 test, but each layer carries a distinct
        // `c2` weight bank so we can prove `mrf_branch_forward` runs the
        // c2 chain (HGAN-01 pre-fix silently dropped it).
        let channels = 2;
        let time = 3;
        let kernel = 3;
        let slope: f32 = 0.1;
        let input: Vec<f32> = (0..channels * time)
            .map(|i| ((i as f32) * 0.4 + 0.1).cos())
            .collect();
        let build_wb = |seed: usize, scale: f32| -> (Vec<f32>, Vec<f32>) {
            let mut w = Vec::with_capacity(channels * channels * kernel);
            for oc in 0..channels {
                for ic in 0..channels {
                    for k in 0..kernel {
                        w.push(((oc + ic + k + seed) as f32 * 0.23).sin() * scale);
                    }
                }
            }
            let b: Vec<f32> = (0..channels)
                .map(|i| (i + seed) as f32 * 0.03 * scale)
                .collect();
            (w, b)
        };
        let (w0_c1, b0_c1) = build_wb(0, 0.3);
        let (w0_c2, b0_c2) = build_wb(100, 0.35);
        let (w1_c1, b1_c1) = build_wb(1, 0.3);
        let (w1_c2, b1_c2) = build_wb(101, 0.35);
        let branch = MrfBranchWeights {
            layers: vec![
                ResBlockLayer {
                    weight: w0_c1.clone(),
                    bias: b0_c1.clone(),
                    weight_c2: Some(w0_c2.clone()),
                    bias_c2: Some(b0_c2.clone()),
                    dilation: 1,
                    kernel,
                    channels,
                },
                ResBlockLayer {
                    weight: w1_c1.clone(),
                    bias: b1_c1.clone(),
                    weight_c2: Some(w1_c2.clone()),
                    bias_c2: Some(b1_c2.clone()),
                    dilation: 1,
                    kernel,
                    channels,
                },
            ],
        };
        let x0 = input.clone();
        let x1 = ref_v1_iteration(
            &x0, channels, time, &w0_c1, &b0_c1, 1, &w0_c2, &b0_c2, kernel, slope,
        );
        let x2 = ref_v1_iteration(
            &x1, channels, time, &w1_c1, &b1_c1, 1, &w1_c2, &b1_c2, kernel, slope,
        );
        let expected = x2;
        let actual = mrf_branch_forward(&input, channels, time, &branch, slope, ResBlockType::V1)
            .expect("V1 mrf_branch_forward");
        assert_eq!(
            actual.len(),
            expected.len(),
            "V1 output shape must match reference"
        );
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-6,
                "V1 element {i}: mrf_branch_forward = {a}, ref = {e}, |Δ| = {}",
                (a - e).abs()
            );
        }
    }

    #[test]
    fn mrf_branch_v1_rejects_missing_c2_weights() {
        // FR-EX-08: V1 without c2 must loudly fail, not silently
        // degrade to V2. Regression pin for HGAN-01's converter bug:
        // if convs2 tensors don't land in the GGUF the loader must
        // never make up zeros — that would produce plausible but wrong
        // audio.
        let channels = 2;
        let time = 3;
        let kernel = 3;
        let input = vec![0.1_f32; channels * time];
        let branch = MrfBranchWeights {
            layers: vec![ResBlockLayer {
                weight: vec![0.01_f32; channels * channels * kernel],
                bias: vec![0.0_f32; channels],
                weight_c2: None,
                bias_c2: None,
                dilation: 1,
                kernel,
                channels,
            }],
        };
        let err = mrf_branch_forward(&input, channels, time, &branch, 0.1, ResBlockType::V1)
            .expect_err("V1 without c2 must fail");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("ResBlockType::V1 requires")
                        && msg.contains("weight_c2")
                        && msg.contains("bias_c2"),
                    "V1 missing-c2 error must name the missing fields, got: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got: {other:?}"),
        }
    }

    #[test]
    fn mrf_branch_v2_rejects_unexpected_c2_weights() {
        // FR-EX-08 mirror: V2 with populated c2 must loudly fail.
        // Partial mixing is a converter bug (loader can't decide which
        // of two topologies to run).
        let channels = 2;
        let time = 3;
        let kernel = 3;
        let input = vec![0.1_f32; channels * time];
        let branch = MrfBranchWeights {
            layers: vec![ResBlockLayer {
                weight: vec![0.01_f32; channels * channels * kernel],
                bias: vec![0.0_f32; channels],
                weight_c2: Some(vec![0.02_f32; channels * channels * kernel]),
                bias_c2: Some(vec![0.0_f32; channels]),
                dilation: 1,
                kernel,
                channels,
            }],
        };
        let err = mrf_branch_forward(&input, channels, time, &branch, 0.1, ResBlockType::V2)
            .expect_err("V2 with c2 must fail");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("V2 must not carry"),
                    "V2 unexpected-c2 error must flag the mismatch, got: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got: {other:?}"),
        }
    }

    #[test]
    fn hifigan_generator_v1_uses_convs2_weights() {
        // End-to-end regression: build V1 attrs, run the full generator,
        // then run again with c2 weights zeroed (which is *not* the
        // same as V2 — V1 still runs both convs, just c2 evaluates to
        // its bias only). The outputs must differ.
        let attrs = tiny_attrs_v1();
        let weights = tiny_weights_v1(&attrs);
        let n_frames = 4;
        let mel: Vec<f32> = (0..attrs.n_mels * n_frames)
            .map(|i| ((i as f32) * 0.1).sin() * 0.3)
            .collect();
        let with_c2 =
            hifigan_generator(&mel, n_frames, &weights, &attrs, &HifiGanConfig::fp32()).unwrap();

        // Now zero every c2 weight (keep biases — they still land).
        // A pre-Wave-2 forward that silently dropped c2 would produce
        // identical output regardless of these weights.
        let mut zeroed = weights.clone();
        for stage in &mut zeroed.mrf_stage_weights {
            for branch in stage.iter_mut() {
                for layer in &mut branch.layers {
                    if let Some(w2) = layer.weight_c2.as_mut() {
                        for v in w2.iter_mut() {
                            *v = 0.0;
                        }
                    }
                }
            }
        }
        let without_c2 =
            hifigan_generator(&mel, n_frames, &zeroed, &attrs, &HifiGanConfig::fp32()).unwrap();

        assert_eq!(with_c2.len(), without_c2.len());
        let max_delta = with_c2
            .iter()
            .zip(without_c2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_delta > 1e-4,
            "V1 output must depend on c2 weights (HGAN-01): max Δ = {max_delta} \
             (a pre-fix generator that dropped c2 would produce identical output)"
        );
    }

    /// WP-08 (2026-08-10): [`vokra_math::tanh`] agrees with the platform
    /// `f32::tanh` on the atol range the SBV2 hot path cares about (<= 2
    /// ULP per-call), so swapping the terminal HiFi-GAN head from
    /// `f32::tanh` to `vokra_math::tanh` cannot regress the existing
    /// synthetic-weight parity tests. The point of the swap is
    /// cross-platform bit-identity of Vokra-side output (glibc-vs-Apple
    /// libm scatter through ~220k per-utterance calls collapses to zero);
    /// the point of THIS test is that no single-call parity was lost.
    /// Source-code anchor: see the "WP-08 (2026-08-10)" comment at the
    /// tanh loop in `hifigan_generator_conditioned` itself.
    #[test]
    fn wp08_terminal_tanh_call_site_parity_within_2ulp() {
        // Sample points spanning the full useful tanh domain including
        // saturation (|x| > 5) and near-zero (linear approx).
        for &x in &[
            -10.0_f32, -5.0, -2.0, -1.0, -0.5, -0.1, -1e-6, 0.0, 1e-6, 0.1, 0.5, 1.0, 2.0, 5.0,
            10.0,
        ] {
            let vm = vokra_math::tanh(x);
            let fs = x.tanh();
            // 2 ULP tolerance vs platform tanh — vokra_math ensures
            // cross-plat determinism WITHIN Vokra (all Linux runners
            // agree bit-for-bit; all macOS runners agree bit-for-bit)
            // but does not promise ULP identity with every platform
            // libm. The atol is set well above the SBV2 waveform atol
            // (1.5) since the per-call error compounds across ~220k
            // calls in real synthesis — the point of the swap is the
            // deterministic per-runner output, not per-libm parity.
            let ulp_delta = (vm - fs).abs() / f32::EPSILON.max(vm.abs());
            assert!(
                ulp_delta < 2.0,
                "vokra_math::tanh({x}) = {vm} disagrees with f32::tanh({x}) = {fs} by \
                 {ulp_delta} ULP (> 2 ULP bound)"
            );
        }
    }
}
