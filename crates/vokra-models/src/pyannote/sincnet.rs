//! SincNet frontend primitive for `pyannote/segmentation-3.0`.
//!
//! Native Vokra reimplementation of the SincNet block described in
//! **PyanNet** (Bredin, CNRS, MIT). The upstream Python lives at
//!
//! - <https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/models/blocks/sincnet.py>
//!   (fetched 2026-07-30, MIT LICENSE Copyright (c) 2019- CNRS by Hervé
//!   Bredin — cited verbatim by the runtime binder in `mod.rs`).
//! - Learnable sinc filter synthesis:
//!   <https://github.com/asteroid-team/asteroid-filterbanks/blob/master/asteroid_filterbanks/param_sinc_fb.py>
//!   (MIT LICENSE) — the algorithm is re-implemented here as pure Rust,
//!   the crate is **not** vendored (NFR-DS-02 zero-dep invariant).
//!
//! # Forward path (transcribed from `sincnet.py:47-119`)
//!
//! ```text
//! waveform : (batch, 1, num_samples)      # 16 kHz mono PCM
//!   -> wav_norm1d = InstanceNorm1d(1, affine=True)
//!   -> conv1d[0] = Encoder(ParamSincFB(n_filters=80, kernel_size=251,
//!                    stride=10, sample_rate=16000,
//!                    min_low_hz=50, min_band_hz=50))    # no bias
//!   -> abs()                                            # SincNet-only
//!   -> pool1d[0] = MaxPool1d(kernel=3, stride=3)
//!   -> norm1d[0] = InstanceNorm1d(80, affine=True)
//!   -> LeakyReLU(0.01)
//!   -> conv1d[1] = Conv1d(80, 60, kernel=5, stride=1)   # with bias
//!   -> pool1d[1] = MaxPool1d(kernel=3, stride=3)
//!   -> norm1d[1] = InstanceNorm1d(60, affine=True)
//!   -> LeakyReLU(0.01)
//!   -> conv1d[2] = Conv1d(60, 60, kernel=5, stride=1)   # with bias
//!   -> pool1d[2] = MaxPool1d(kernel=3, stride=3)
//!   -> norm1d[2] = InstanceNorm1d(60, affine=True)
//!   -> LeakyReLU(0.01)
//! => features : (batch, 60, num_frames)
//! ```
//!
//! `num_frames` follows the six-layer `multi_conv_num_frames` recurrence
//! from `pyannote/audio/utils/receptive_field.py:56-69` (kernels
//! `[251,3,5,3,5,3]`, strides `[stride,3,1,3,1,3]`, no padding, no
//! dilation). At `stride=10` and `num_samples=160000` the pin-tested
//! recurrence yields **589** frames — this is the value the downstream
//! BiLSTM stack allocates against.
//!
//! # Learnable parameters
//!
//! - `low_hz_` : `(n_filters / 2, 1)` = `(40, 1)` — raw low-cutoff
//!   parameter per filter, mapped as
//!   `low = min_low_hz + |low_hz_|`.
//! - `band_hz_` : `(n_filters / 2, 1)` = `(40, 1)` — raw bandwidth
//!   parameter per filter, mapped as
//!   `high = clamp(low + min_band_hz + |band_hz_|, min_low_hz, sr/2)`.
//! - `wav_norm1d.{weight,bias}` : `(1,)` each — affine after the raw
//!   waveform InstanceNorm.
//! - `norm1d[k].{weight,bias}` for `k ∈ {0,1,2}` : `(80,) / (60,) /
//!   (60,)` — affine after each post-conv InstanceNorm.
//! - `conv1d[k].{weight,bias}` for `k ∈ {1,2}` : Conv1d weight
//!   `(60, 80, 5)` then `(60, 60, 5)`, bias `(60,)`, `(60,)`.
//!
//! `sincnet.conv1d.0.filterbank.{window_, n_}` are **buffers** in the
//! upstream state_dict (not learnable). Vokra treats them as
//! compile-time constants derived from `(kernel_size, sample_rate)` —
//! any real GGUF that ships them is accepted (dequant-checked in the
//! runtime binder), and the compile-time constants are used at forward
//! time (they are numerically identical to the buffers).
//!
//! # Numerics and reduction ordering (parity)
//!
//! The convolutions dispatch through the [`crate::compute::Compute`]
//! CPU seam ([`Compute::conv1d_f32`]), so the multiply-accumulate order
//! is the shared im2col + GEMM order (same as every other Vokra CPU
//! conv1d). The Hamming window, sinc synthesis, InstanceNorm mean /
//! variance, and max-pool reductions use a plain scalar `for` loop —
//! LLVM auto-vectorises them on both AArch64 and x86-64.
//!
//! # Zero-dep invariant (NFR-DS-02)
//!
//! Every operation is in-repo: `sin` / `cos` / `abs` / `clamp` from
//! `std`, `conv1d_f32` from the CPU backend, no crates.io addition.
//! `asteroid_filterbanks` is **not** a dependency; its sinc-synthesis
//! algorithm is transcribed here with author credit and MIT licence
//! preservation in the module comment.

use vokra_core::{Result, VokraError};

use crate::compute::Compute;

use super::PyanNetWeights;

// ---------------------------------------------------------------------------
// Primary-source constants (`sincnet.py` + `PyanNet.SINCNET_DEFAULTS`
// + `param_sinc_fb.py`)
// ---------------------------------------------------------------------------

/// Number of learnable sinc bandpass filters (`SincNet.__init__` L53:
/// `n_filters=80`).
pub const N_FILTERS_SINC: usize = 80;
/// Sinc filter kernel size in samples (`SincNet.__init__` L53:
/// `kernel_size=251`).
pub const KERNEL_SIZE_SINC: usize = 251;
/// Half-kernel span (`KERNEL_SIZE_SINC // 2` = 125). This is the number
/// of taps that feed both the Hamming half-window and the negative-side
/// time-vector buffer (`param_sinc_fb.py` L60-68).
pub const HALF_KERNEL_SINC: usize = KERNEL_SIZE_SINC / 2;
/// Minimum low-cutoff frequency in Hz (`SincNet.__init__` L54:
/// `min_low_hz=50`). Added to `|low_hz_|` to guarantee the filter never
/// dips below 50 Hz.
pub const MIN_LOW_HZ: f32 = 50.0;
/// Minimum bandwidth in Hz (`SincNet.__init__` L54: `min_band_hz=50`).
/// Ensures the high-cutoff always sits at least 50 Hz above the low.
pub const MIN_BAND_HZ: f32 = 50.0;
/// Only 16 kHz PCM is supported by SincNet (`sincnet.py` L47 raises
/// `NotImplementedError` for any other sample rate).
pub const SAMPLE_RATE_SINCNET: u32 = 16_000;
/// LeakyReLU slope after every post-pool InstanceNorm. PyTorch's
/// `F.leaky_relu` default; the upstream `sincnet.py:115` call passes no
/// explicit slope.
pub const LEAKY_RELU_SLOPE: f32 = 0.01;
/// InstanceNorm epsilon (`nn.InstanceNorm1d` PyTorch default 1e-5).
pub const INSTANCE_NORM_EPS: f32 = 1e-5;
/// MaxPool kernel size — used for all three of the sincnet.py pools.
pub const POOL_KERNEL: usize = 3;
/// MaxPool stride — matches `pool_kernel` (non-overlapping).
pub const POOL_STRIDE: usize = 3;
/// Second Conv1d input channel count (matches `n_filters_sinc`).
pub const CONV1_IN_CH: usize = N_FILTERS_SINC;
/// Second Conv1d output channel count (`sincnet.py:77 nn.Conv1d(80, 60,
/// 5)`).
pub const CONV1_OUT_CH: usize = 60;
/// Second / third Conv1d kernel size.
pub const CONV_KERNEL_LATER: usize = 5;
/// Third Conv1d input channel count (matches `conv1_out_ch`).
pub const CONV2_IN_CH: usize = CONV1_OUT_CH;
/// Third Conv1d output channel count (`sincnet.py:81 nn.Conv1d(60, 60,
/// 5)`) and the final SincNet output feature dim consumed by the
/// downstream BiLSTM.
pub const CONV2_OUT_CH: usize = 60;

// ---------------------------------------------------------------------------
// Compile-time buffers (Hamming half-window + time vector)
// ---------------------------------------------------------------------------

/// Builds the Hamming half-window `np.hamming(kernel_size)[:half_kernel]`
/// used by `param_sinc_fb.py:63` to weight the sinc filters.
///
/// Formula: `0.54 - 0.46 · cos(2π · n / (kernel - 1))` for
/// `n ∈ [0, half_kernel)`. The half-window is symmetric about the
/// centre; the second half is implicit by mirroring in the sinc
/// synthesis loop.
fn hamming_half_window() -> [f32; HALF_KERNEL_SINC] {
    let mut w = [0.0f32; HALF_KERNEL_SINC];
    let denom = (KERNEL_SIZE_SINC - 1) as f32;
    let two_pi = 2.0 * std::f32::consts::PI;
    for (n, slot) in w.iter_mut().enumerate() {
        *slot = 0.54 - 0.46 * (two_pi * (n as f32) / denom).cos();
    }
    w
}

/// Builds the negative-time vector `2π · arange(-half_kernel, 0) /
/// sample_rate` used by `param_sinc_fb.py:64-68` to seed the sinc
/// filter's phase.
fn negative_time_vector() -> [f32; HALF_KERNEL_SINC] {
    let mut n = [0.0f32; HALF_KERNEL_SINC];
    let two_pi_over_sr = 2.0 * std::f32::consts::PI / (SAMPLE_RATE_SINCNET as f32);
    for (i, slot) in n.iter_mut().enumerate() {
        // arange(-125, 0) → i in 0..125 corresponds to sample -125 + i,
        // i.e. -(half_kernel - i).
        let sample = -(HALF_KERNEL_SINC as f32) + (i as f32);
        *slot = two_pi_over_sr * sample;
    }
    n
}

// ---------------------------------------------------------------------------
// Frame-count arithmetic (six-layer `multi_conv_num_frames` recurrence)
// ---------------------------------------------------------------------------

/// Per-layer output-length formula from
/// `pyannote/audio/utils/receptive_field.py:34-49 conv1d_num_frames`:
///
/// `out = 1 + floor((in + 2·padding - dilation·(kernel - 1) - 1) / stride)`
///
/// SincNet uses no padding and no dilation; the recurrence collapses to
/// `out = 1 + floor((in - (kernel - 1) - 1) / stride)` (equivalently
/// `1 + (in - kernel) / stride`). Returns `0` when the input is shorter
/// than `kernel`, matching the upstream Python which would raise a
/// negative-dim error the pipeline never reaches.
fn conv1d_out_len(in_len: usize, kernel: usize, stride: usize) -> usize {
    if in_len < kernel {
        return 0;
    }
    1 + (in_len - kernel) / stride
}

/// Computes `SincNet.num_frames(num_samples)` for the pyannote/segmentation-3.0
/// six-layer default stack — same numeric recurrence
/// `pyannote/audio/utils/receptive_field.py:56-69 multi_conv_num_frames`
/// runs at import time.
///
/// Layer sequence at `sincnet_stride = 10`:
/// `kernel = [251, 3, 5, 3, 5, 3]`,
/// `stride = [10,  3, 1, 3, 1, 3]`.
///
/// Pin-tested values (see `num_frames_matches_primary_source_recurrence`
/// in tests below):
///
/// | input  | after L0 | L1   | L2   | L3   | L4   | L5   |
/// |--------|----------|------|------|------|------|------|
/// | 160000 | 15975    | 5325 | 5321 | 1773 | 1769 | 589  |
/// | 16000  | 1575     | 525  | 521  | 173  | 169  | 56   |
pub fn num_frames(num_samples: usize, sincnet_stride: usize) -> usize {
    let l0 = conv1d_out_len(num_samples, KERNEL_SIZE_SINC, sincnet_stride);
    let l1 = conv1d_out_len(l0, POOL_KERNEL, POOL_STRIDE);
    let l2 = conv1d_out_len(l1, CONV_KERNEL_LATER, 1);
    let l3 = conv1d_out_len(l2, POOL_KERNEL, POOL_STRIDE);
    let l4 = conv1d_out_len(l3, CONV_KERNEL_LATER, 1);
    conv1d_out_len(l4, POOL_KERNEL, POOL_STRIDE)
}

// ---------------------------------------------------------------------------
// SincNet primitive struct
// ---------------------------------------------------------------------------

/// SincNet frontend — the learnable-sinc + 2×Conv1d stack.
///
/// Constructed from a bound [`PyanNetWeights`] via
/// [`SincNet::from_weights`]; the constructor performs every shape check
/// upfront so a mis-shaped state_dict fails loudly at load
/// (`VokraError::ModelLoad` naming FR-EX-08).
///
/// Forward inputs are `(1, 1, num_samples)` at 16 kHz mono; the output
/// is `(1, 60, num_frames(num_samples, stride))` in `[channels, time]`
/// row-major (matching `crate::compute::Compute::conv1d_f32`'s output
/// convention).
#[derive(Debug)]
pub struct SincNet {
    // Learned sinc filter parameters, `(n_filters / 2, 1)` each.
    low_hz: Vec<f32>,
    band_hz: Vec<f32>,
    // Post-conv InstanceNorm affine (weight, bias) per level.
    wav_norm_scale: Vec<f32>, // len 1
    wav_norm_bias: Vec<f32>,  // len 1
    norm0_scale: Vec<f32>,    // len 80
    norm0_bias: Vec<f32>,     // len 80
    norm1_scale: Vec<f32>,    // len 60
    norm1_bias: Vec<f32>,     // len 60
    norm2_scale: Vec<f32>,    // len 60
    norm2_bias: Vec<f32>,     // len 60
    // Conv1d weights (row-major `[out_ch, in_ch, kernel]`) + bias.
    conv1_weight: Vec<f32>, // (60, 80, 5)
    conv1_bias: Vec<f32>,   // (60,)
    conv2_weight: Vec<f32>, // (60, 60, 5)
    conv2_bias: Vec<f32>,   // (60,)
    sincnet_stride: usize,
}

impl SincNet {
    /// Binds a SincNet from a [`PyanNetWeights`] manifest — every
    /// tensor is looked up by its upstream `state_dict` name and
    /// shape-checked. `sincnet_stride` comes from
    /// [`super::PyanNetConfig::sincnet_stride`].
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] with an FR-EX-08 tag if any tensor is
    /// missing, mis-shaped, or has the wrong element count. Every
    /// message names the offending tensor path so the caller can trace
    /// back to the converter output.
    pub fn from_weights(w: &PyanNetWeights, sincnet_stride: usize) -> Result<Self> {
        // Sinc filter learnable parameters
        // (`sincnet.conv1d.0.filterbank.{low_hz_,band_hz_}`, each
        // `(n_filters / 2, 1)`).
        let n_sinc_learnable = N_FILTERS_SINC / 2;
        let low_hz = bind_tensor(
            w,
            "sincnet.conv1d.0.filterbank.low_hz_",
            &[n_sinc_learnable, 1],
        )?;
        let band_hz = bind_tensor(
            w,
            "sincnet.conv1d.0.filterbank.band_hz_",
            &[n_sinc_learnable, 1],
        )?;

        // wav_norm1d affine (sincnet.py:60 `nn.InstanceNorm1d(1,
        // affine=True)`).
        let wav_norm_scale = bind_tensor(w, "sincnet.wav_norm1d.weight", &[1])?;
        let wav_norm_bias = bind_tensor(w, "sincnet.wav_norm1d.bias", &[1])?;

        // norm1d[0..2] affine (sincnet.py:75, 79, 83).
        let norm0_scale = bind_tensor(w, "sincnet.norm1d.0.weight", &[N_FILTERS_SINC])?;
        let norm0_bias = bind_tensor(w, "sincnet.norm1d.0.bias", &[N_FILTERS_SINC])?;
        let norm1_scale = bind_tensor(w, "sincnet.norm1d.1.weight", &[CONV1_OUT_CH])?;
        let norm1_bias = bind_tensor(w, "sincnet.norm1d.1.bias", &[CONV1_OUT_CH])?;
        let norm2_scale = bind_tensor(w, "sincnet.norm1d.2.weight", &[CONV2_OUT_CH])?;
        let norm2_bias = bind_tensor(w, "sincnet.norm1d.2.bias", &[CONV2_OUT_CH])?;

        // Conv1d[1..2] (sincnet.py:77, 81).
        let conv1_weight = bind_tensor(
            w,
            "sincnet.conv1d.1.weight",
            &[CONV1_OUT_CH, CONV1_IN_CH, CONV_KERNEL_LATER],
        )?;
        let conv1_bias = bind_tensor(w, "sincnet.conv1d.1.bias", &[CONV1_OUT_CH])?;
        let conv2_weight = bind_tensor(
            w,
            "sincnet.conv1d.2.weight",
            &[CONV2_OUT_CH, CONV2_IN_CH, CONV_KERNEL_LATER],
        )?;
        let conv2_bias = bind_tensor(w, "sincnet.conv1d.2.bias", &[CONV2_OUT_CH])?;

        Ok(Self {
            low_hz,
            band_hz,
            wav_norm_scale,
            wav_norm_bias,
            norm0_scale,
            norm0_bias,
            norm1_scale,
            norm1_bias,
            norm2_scale,
            norm2_bias,
            conv1_weight,
            conv1_bias,
            conv2_weight,
            conv2_bias,
            sincnet_stride,
        })
    }

    /// Runs the SincNet forward on a mono 16 kHz PCM buffer and returns
    /// the `[60, num_frames]` row-major feature slab (batch dim is
    /// squeezed out — a single-utterance forward is the only shape the
    /// pipeline calls).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] if `sample_rate ≠ 16000` — mirror
    ///   of `sincnet.py:47 raise NotImplementedError` (FR-EX-08).
    /// - [`VokraError::UnsupportedOp`] if the input is shorter than the
    ///   sinc kernel (251 samples).
    pub fn forward(&self, pcm: &[f32], sample_rate: u32) -> Result<SincNetOutput> {
        if sample_rate != SAMPLE_RATE_SINCNET {
            return Err(VokraError::UnsupportedOp(format!(
                "pyannote-segmentation SincNet: sample_rate={sample_rate}, only 16000 is \
                 supported (FR-EX-08, mirror of upstream sincnet.py:47 `raise \
                 NotImplementedError`)"
            )));
        }
        if pcm.len() < KERNEL_SIZE_SINC {
            return Err(VokraError::UnsupportedOp(format!(
                "pyannote-segmentation SincNet: input length {} is shorter than the sinc \
                 kernel {KERNEL_SIZE_SINC} — no output frames producible (FR-EX-08)",
                pcm.len()
            )));
        }
        let cpu = Compute::cpu();

        // Step 1: `wav_norm1d = InstanceNorm1d(1, affine=True)` on the
        // raw waveform. Single-channel case collapses to `x_norm =
        // (x - mean(x)) / sqrt(var(x) + eps)`, then `y = γ·x_norm + β`
        // with `γ = wav_norm_scale[0]`, `β = wav_norm_bias[0]`.
        let mut wav = pcm.to_vec();
        instance_norm_1_channel(&mut wav, self.wav_norm_scale[0], self.wav_norm_bias[0]);

        // Step 2: sinc bandpass Conv1d — synthesise the (80, 1, 251)
        // filter kernel then dispatch through Compute::conv1d_f32. No
        // bias in the upstream filterbank Encoder.
        let filters = self.synthesise_sinc_filters();
        let l0_len = conv1d_out_len(wav.len(), KERNEL_SIZE_SINC, self.sincnet_stride);
        let mut l0 = vec![0.0f32; N_FILTERS_SINC * l0_len];
        cpu.conv1d_f32(
            &wav,
            1,
            wav.len(),
            &filters,
            N_FILTERS_SINC,
            KERNEL_SIZE_SINC,
            None,
            self.sincnet_stride,
            0,
            &mut l0,
        )?;

        // Step 3: `torch.abs(outputs)` — SincNet-only half-wave rectify
        // (sincnet.py:112-113 `if c == 0: outputs = torch.abs(outputs)`).
        for v in l0.iter_mut() {
            *v = v.abs();
        }

        // Step 4: `MaxPool1d(3, stride=3)` per channel.
        let l1_len = conv1d_out_len(l0_len, POOL_KERNEL, POOL_STRIDE);
        let mut l1 = vec![0.0f32; N_FILTERS_SINC * l1_len];
        max_pool_1d_channelwise(&l0, N_FILTERS_SINC, l0_len, &mut l1, l1_len);

        // Step 5: `InstanceNorm1d(80, affine=True)` per channel across
        // the time axis, then LeakyReLU(0.01).
        instance_norm_and_leaky(
            &mut l1,
            N_FILTERS_SINC,
            l1_len,
            &self.norm0_scale,
            &self.norm0_bias,
        );

        // Step 6: Conv1d(80, 60, kernel=5, stride=1) — with bias.
        let l2_len = conv1d_out_len(l1_len, CONV_KERNEL_LATER, 1);
        let mut l2 = vec![0.0f32; CONV1_OUT_CH * l2_len];
        cpu.conv1d_f32(
            &l1,
            N_FILTERS_SINC,
            l1_len,
            &self.conv1_weight,
            CONV1_OUT_CH,
            CONV_KERNEL_LATER,
            Some(&self.conv1_bias),
            1,
            0,
            &mut l2,
        )?;

        // Step 7: `MaxPool1d(3, stride=3)` per channel.
        let l3_len = conv1d_out_len(l2_len, POOL_KERNEL, POOL_STRIDE);
        let mut l3 = vec![0.0f32; CONV1_OUT_CH * l3_len];
        max_pool_1d_channelwise(&l2, CONV1_OUT_CH, l2_len, &mut l3, l3_len);

        // Step 8: `InstanceNorm1d(60, affine=True)` + LeakyReLU(0.01).
        instance_norm_and_leaky(
            &mut l3,
            CONV1_OUT_CH,
            l3_len,
            &self.norm1_scale,
            &self.norm1_bias,
        );

        // Step 9: Conv1d(60, 60, kernel=5, stride=1) — with bias.
        let l4_len = conv1d_out_len(l3_len, CONV_KERNEL_LATER, 1);
        let mut l4 = vec![0.0f32; CONV2_OUT_CH * l4_len];
        cpu.conv1d_f32(
            &l3,
            CONV1_OUT_CH,
            l3_len,
            &self.conv2_weight,
            CONV2_OUT_CH,
            CONV_KERNEL_LATER,
            Some(&self.conv2_bias),
            1,
            0,
            &mut l4,
        )?;

        // Step 10: `MaxPool1d(3, stride=3)` per channel.
        let l5_len = conv1d_out_len(l4_len, POOL_KERNEL, POOL_STRIDE);
        let mut l5 = vec![0.0f32; CONV2_OUT_CH * l5_len];
        max_pool_1d_channelwise(&l4, CONV2_OUT_CH, l4_len, &mut l5, l5_len);

        // Step 11: `InstanceNorm1d(60, affine=True)` + LeakyReLU(0.01).
        instance_norm_and_leaky(
            &mut l5,
            CONV2_OUT_CH,
            l5_len,
            &self.norm2_scale,
            &self.norm2_bias,
        );

        // `l5` is `[60, num_frames]` row-major, `num_frames = l5_len`
        // = `num_frames(pcm.len(), stride)`. Cross-check the algebraic
        // identity to catch any drift in the recurrence table.
        debug_assert_eq!(
            l5_len,
            num_frames(pcm.len(), self.sincnet_stride),
            "sincnet forward emitted {l5_len} frames but num_frames() said {}",
            num_frames(pcm.len(), self.sincnet_stride)
        );

        Ok(SincNetOutput {
            features: l5,
            num_frames: l5_len,
            num_channels: CONV2_OUT_CH,
        })
    }

    /// Synthesises the `(80, 1, 251)` sinc bandpass filter bank from
    /// the learned `low_hz_` / `band_hz_` parameters (per-call —
    /// filters are input-independent but weight-dependent, and at
    /// inference the weights are fixed so this is deterministic).
    ///
    /// Algorithm from `param_sinc_fb.py:82-108`:
    ///
    /// ```text
    /// low  = min_low_hz + |low_hz_|
    /// high = clamp(low + min_band_hz + |band_hz_|, min_low_hz, sr/2)
    /// bp(t) = (sin(2π·high·t) - sin(2π·low·t)) / (π·t) · hamming(t)
    /// filter[i]      = bp(negative half) · 2·high / sr
    ///                  concat centre = 2·(high - low) / sr
    ///                  concat mirror(bp)
    /// ```
    ///
    /// The `2·high / sr` normalisation is folded into every tap; the
    /// centre tap avoids the `1/t → ∞` singularity by taking the
    /// analytic limit `2·(high - low)`.
    ///
    /// The output layout matches Vokra's Conv1d convention:
    /// `[out_ch=80, in_ch=1, kernel=251]` row-major, so the returned
    /// `Vec<f32>` has length `80 · 1 · 251 = 20080`.
    fn synthesise_sinc_filters(&self) -> Vec<f32> {
        let hamming = hamming_half_window();
        let n_vec = negative_time_vector();
        let n_learnable = N_FILTERS_SINC / 2;

        let mut filters = vec![0.0f32; N_FILTERS_SINC * KERNEL_SIZE_SINC];
        let sr_half = (SAMPLE_RATE_SINCNET as f32) * 0.5;

        for i in 0..n_learnable {
            // `min_low_hz + |low_hz_|` — the raw parameter is
            // unsigned via `torch.abs`; clamp guards against pathological
            // `band_hz_` values that would push high past sr/2.
            let low = MIN_LOW_HZ + self.low_hz[i].abs();
            let high = (low + MIN_BAND_HZ + self.band_hz[i].abs()).clamp(MIN_LOW_HZ, sr_half);
            let low_row = 2.0 * low; // absorbed into the sinc: sin(2π·f·t)/π·t
            let high_row = 2.0 * high;

            // The filter is symmetric about the centre tap. We fill
            // the negative half using `sin(2π·high·t) - sin(2π·low·t)`
            // divided by `π·t` (the `π` is absorbed into `n_vec`, and
            // the `1/t` becomes `1/n[k]` since `n_vec[k] = 2π·k/sr`
            // where `k = -125 + i`). Follow the upstream sequence
            // exactly:
            //
            //     band_pass_left  = (sin(high·n) - sin(low·n)) / (n/2) · window
            //     band_pass_right = mirror(band_pass_left)
            //     centre = 2·(high - low)
            //     filter = concat(left, [centre], right) / (2·high)
            //
            // (Upstream L92-105.) But the outer `2·high` is *not*
            // divided per filter in the asteroid implementation — it
            // is a per-filter row-normalisation. See below.

            let filter_offset = i * KERNEL_SIZE_SINC;
            let mirror_row_offset = (i + n_learnable) * KERNEL_SIZE_SINC;

            // Left half `t < 0`.
            for k in 0..HALF_KERNEL_SINC {
                let t = n_vec[k]; // negative for k in [0, half)
                let s_high = (high_row * t).sin();
                let s_low = (low_row * t).sin();
                // 1 / t (= π·t in the original but absorbed into n_vec).
                // Guard against a divide-by-zero even though t ≠ 0 for
                // k < half_kernel.
                let denom = if t.abs() < f32::EPSILON { 1.0 } else { t };
                let tap = (s_high - s_low) / denom * hamming[k];
                let row_norm = 1.0 / (2.0 * high);
                filters[filter_offset + k] = tap * row_norm;
                // Mirror (right half) at the second `n_learnable`
                // filter row — upstream `torch.flip(band_pass_left,
                // dims=[1])` reverses the same coefficients for the
                // second half.
                filters[mirror_row_offset + (KERNEL_SIZE_SINC - 1 - k)] = tap * row_norm;
            }
            // Centre tap `t = 0` — analytic limit `2·(high - low)`,
            // then row-normalised.
            let centre = 2.0 * (high - low) / (2.0 * high);
            filters[filter_offset + HALF_KERNEL_SINC] = centre;
            filters[mirror_row_offset + HALF_KERNEL_SINC] = centre;

            // Also fill the right half of the first `n_learnable` rows
            // (mirror of the same left half) so the first-half filters
            // are full symmetric bandpasses.
            for k in 0..HALF_KERNEL_SINC {
                filters[filter_offset + KERNEL_SIZE_SINC - 1 - k] = filters[filter_offset + k];
            }
            // Fill the left half of the mirror rows too.
            for k in 0..HALF_KERNEL_SINC {
                filters[mirror_row_offset + k] =
                    filters[mirror_row_offset + KERNEL_SIZE_SINC - 1 - k];
            }
        }

        filters
    }

    /// Number of learnable filter parameters (n_filters / 2 = 40).
    pub fn n_learnable_filters(&self) -> usize {
        self.low_hz.len()
    }

    /// SincNet stride (`vokra.pyannote.sincnet.stride`, default 10).
    pub fn stride(&self) -> usize {
        self.sincnet_stride
    }
}

/// SincNet forward output — `[60, num_frames]` row-major, plus the
/// axis descriptors the downstream BiLSTM stack allocates against.
#[derive(Debug)]
pub struct SincNetOutput {
    /// Row-major `[num_channels · num_frames]` payload.
    pub features: Vec<f32>,
    /// Second-axis extent (matches
    /// [`crate::pyannote::sincnet::num_frames`]).
    pub num_frames: usize,
    /// First-axis extent (always [`CONV2_OUT_CH`] = 60 for
    /// pyannote/segmentation-3.0).
    pub num_channels: usize,
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Binds one tensor from [`PyanNetWeights`] with an exact shape check.
/// Every tensor referenced by [`SincNet::from_weights`] passes through
/// here; a mismatched shape is a loud [`VokraError::ModelLoad`] naming
/// the offending tensor path (FR-EX-08).
fn bind_tensor(w: &PyanNetWeights, name: &str, expect: &[usize]) -> Result<Vec<f32>> {
    let (dims, payload) = w.tensor(name).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "pyannote-segmentation SincNet: required tensor `{name}` is missing from the \
             GGUF (FR-EX-08). Expected shape {expect:?}."
        ))
    })?;
    if dims != expect {
        return Err(VokraError::ModelLoad(format!(
            "pyannote-segmentation SincNet: tensor `{name}` has shape {dims:?}, expected \
             {expect:?} (FR-EX-08)"
        )));
    }
    Ok(payload.to_vec())
}

/// InstanceNorm1d for the single-channel raw-waveform case: normalise
/// the whole buffer, then apply the affine `(γ, β)`.
fn instance_norm_1_channel(x: &mut [f32], gamma: f32, beta: f32) {
    if x.is_empty() {
        return;
    }
    let n = x.len() as f64;
    let mut mean64 = 0.0f64;
    for &v in x.iter() {
        mean64 += v as f64;
    }
    mean64 /= n;
    let mut var64 = 0.0f64;
    for &v in x.iter() {
        let d = v as f64 - mean64;
        var64 += d * d;
    }
    var64 /= n;
    let inv = 1.0 / ((var64 as f32) + INSTANCE_NORM_EPS).sqrt();
    let mean = mean64 as f32;
    for v in x.iter_mut() {
        *v = (*v - mean) * inv * gamma + beta;
    }
}

/// InstanceNorm1d + LeakyReLU in one channel-major pass. `x` is
/// `[channels · time]` row-major with `channel` outer. Each channel is
/// normalised independently across the time axis, then affine-applied
/// with `(γ, β)` and passed through `F.leaky_relu(0.01)`.
fn instance_norm_and_leaky(
    x: &mut [f32],
    channels: usize,
    time: usize,
    gamma: &[f32],
    beta: &[f32],
) {
    debug_assert_eq!(x.len(), channels * time, "instance_norm_and_leaky: x len");
    debug_assert_eq!(gamma.len(), channels, "instance_norm_and_leaky: gamma len");
    debug_assert_eq!(beta.len(), channels, "instance_norm_and_leaky: beta len");
    if time == 0 {
        return;
    }
    let inv_t = 1.0 / (time as f64);
    for c in 0..channels {
        let row = &mut x[c * time..c * time + time];
        let mut mean64 = 0.0f64;
        for &v in row.iter() {
            mean64 += v as f64;
        }
        mean64 *= inv_t;
        let mean = mean64 as f32;
        let mut var64 = 0.0f64;
        for &v in row.iter() {
            let d = v as f64 - mean64;
            var64 += d * d;
        }
        var64 *= inv_t;
        let inv = 1.0 / ((var64 as f32) + INSTANCE_NORM_EPS).sqrt();
        let g = gamma[c];
        let b = beta[c];
        for v in row.iter_mut() {
            let y = (*v - mean) * inv * g + b;
            *v = if y < 0.0 { y * LEAKY_RELU_SLOPE } else { y };
        }
    }
}

/// MaxPool1d(kernel=3, stride=3) per channel on a `[channels, in_time]`
/// row-major buffer. The output is `[channels, out_time]`. `out_time`
/// must equal `conv1d_out_len(in_time, POOL_KERNEL, POOL_STRIDE)`.
fn max_pool_1d_channelwise(
    src: &[f32],
    channels: usize,
    in_time: usize,
    dst: &mut [f32],
    out_time: usize,
) {
    debug_assert_eq!(src.len(), channels * in_time);
    debug_assert_eq!(dst.len(), channels * out_time);
    debug_assert_eq!(
        out_time,
        conv1d_out_len(in_time, POOL_KERNEL, POOL_STRIDE),
        "max_pool_1d_channelwise: out_time / recurrence mismatch"
    );
    for c in 0..channels {
        let src_row = &src[c * in_time..c * in_time + in_time];
        let dst_row = &mut dst[c * out_time..c * out_time + out_time];
        // `enumerate` + `iter_mut` silences `needless_range_loop` while
        // preserving the `t` index needed to compute the src window
        // offset.
        for (t, dst_t) in dst_row.iter_mut().enumerate() {
            let start = t * POOL_STRIDE;
            let end = start + POOL_KERNEL;
            debug_assert!(end <= in_time);
            let window = &src_row[start..end];
            let mut m = window[0];
            for &v in &window[1..] {
                if v > m {
                    m = v;
                }
            }
            *dst_t = m;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyannote::{DEFAULT_SINCNET_STRIDE, PyanNetConfig, PyanNetWeights};
    use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile};

    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-pyannote-sincnet-{}-{}-{}.gguf",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        p
    }

    /// Builds a minimal synthetic PyanNet GGUF whose tensor set covers
    /// every SincNet-required shape. Weights are deterministic
    /// f32-per-element so tests can reason about the forward output
    /// without importing a Python numeric fixture.
    ///
    /// Returns the GGUF path — caller cleans up.
    fn synthetic_sincnet_gguf() -> std::path::PathBuf {
        use crate::pyannote::{
            DEFAULT_LINEAR_HIDDEN_SIZE, DEFAULT_LINEAR_NUM_LAYERS, DEFAULT_LSTM_BIDIRECTIONAL,
            DEFAULT_LSTM_HIDDEN_SIZE, DEFAULT_LSTM_MONOLITHIC, DEFAULT_LSTM_NUM_LAYERS,
            DEFAULT_NUM_POWERSET_CLASSES, DEFAULT_SAMPLE_RATE, GGUF_KEY_LINEAR_HIDDEN_SIZE,
            GGUF_KEY_LINEAR_NUM_LAYERS, GGUF_KEY_LSTM_BIDIRECTIONAL, GGUF_KEY_LSTM_HIDDEN_SIZE,
            GGUF_KEY_LSTM_MONOLITHIC, GGUF_KEY_LSTM_NUM_LAYERS, GGUF_KEY_NUM_POWERSET_CLASSES,
            GGUF_KEY_SAMPLE_RATE, GGUF_KEY_SINCNET_STRIDE,
        };
        let mut b = GgufBuilder::new();
        b.add_u32(GGUF_KEY_SAMPLE_RATE, DEFAULT_SAMPLE_RATE);
        b.add_u32(GGUF_KEY_SINCNET_STRIDE, DEFAULT_SINCNET_STRIDE);
        b.add_u32(GGUF_KEY_LSTM_HIDDEN_SIZE, DEFAULT_LSTM_HIDDEN_SIZE);
        b.add_u32(GGUF_KEY_LSTM_NUM_LAYERS, DEFAULT_LSTM_NUM_LAYERS);
        b.add_bool(GGUF_KEY_LSTM_BIDIRECTIONAL, DEFAULT_LSTM_BIDIRECTIONAL);
        b.add_bool(GGUF_KEY_LSTM_MONOLITHIC, DEFAULT_LSTM_MONOLITHIC);
        b.add_u32(GGUF_KEY_LINEAR_HIDDEN_SIZE, DEFAULT_LINEAR_HIDDEN_SIZE);
        b.add_u32(GGUF_KEY_LINEAR_NUM_LAYERS, DEFAULT_LINEAR_NUM_LAYERS);
        b.add_u32(GGUF_KEY_NUM_POWERSET_CLASSES, DEFAULT_NUM_POWERSET_CLASSES);

        let n_learnable = (N_FILTERS_SINC / 2) as u64;
        // (a) Learnable sinc filter parameters — small positive scalars
        // so `abs()` and `clamp()` do not chop them. `low_hz_[i] = 100·i`
        // gives filters at 50 + 100·i Hz per row; band_hz_ = 100 keeps
        // each filter ~200 Hz wide.
        let low_hz_bytes: Vec<u8> = (0..(n_learnable) as usize)
            .flat_map(|i| ((i as f32) * 100.0).to_le_bytes())
            .collect();
        b.add_tensor(
            "sincnet.conv1d.0.filterbank.low_hz_",
            GgmlType::F32,
            vec![n_learnable, 1],
            low_hz_bytes,
        )
        .unwrap();
        let band_hz_bytes: Vec<u8> = (0..(n_learnable) as usize)
            .flat_map(|_| 100.0f32.to_le_bytes())
            .collect();
        b.add_tensor(
            "sincnet.conv1d.0.filterbank.band_hz_",
            GgmlType::F32,
            vec![n_learnable, 1],
            band_hz_bytes,
        )
        .unwrap();

        // (b) InstanceNorm affine (identity-ish).
        let scalar_bytes =
            |c: usize, v: f32| -> Vec<u8> { (0..c).flat_map(|_| v.to_le_bytes()).collect() };
        b.add_tensor(
            "sincnet.wav_norm1d.weight",
            GgmlType::F32,
            vec![1],
            scalar_bytes(1, 1.0),
        )
        .unwrap();
        b.add_tensor(
            "sincnet.wav_norm1d.bias",
            GgmlType::F32,
            vec![1],
            scalar_bytes(1, 0.0),
        )
        .unwrap();
        for (name, c) in [
            ("sincnet.norm1d.0.weight", N_FILTERS_SINC),
            ("sincnet.norm1d.1.weight", CONV1_OUT_CH),
            ("sincnet.norm1d.2.weight", CONV2_OUT_CH),
        ] {
            b.add_tensor(name, GgmlType::F32, vec![c as u64], scalar_bytes(c, 1.0))
                .unwrap();
        }
        for (name, c) in [
            ("sincnet.norm1d.0.bias", N_FILTERS_SINC),
            ("sincnet.norm1d.1.bias", CONV1_OUT_CH),
            ("sincnet.norm1d.2.bias", CONV2_OUT_CH),
        ] {
            b.add_tensor(name, GgmlType::F32, vec![c as u64], scalar_bytes(c, 0.0))
                .unwrap();
        }

        // (c) Conv1d weights + bias.
        let conv1_w: Vec<u8> = (0..(CONV1_OUT_CH * CONV1_IN_CH * CONV_KERNEL_LATER))
            .flat_map(|i| ((i as f32) * 0.0001).to_le_bytes())
            .collect();
        b.add_tensor(
            "sincnet.conv1d.1.weight",
            GgmlType::F32,
            vec![
                CONV1_OUT_CH as u64,
                CONV1_IN_CH as u64,
                CONV_KERNEL_LATER as u64,
            ],
            conv1_w,
        )
        .unwrap();
        b.add_tensor(
            "sincnet.conv1d.1.bias",
            GgmlType::F32,
            vec![CONV1_OUT_CH as u64],
            scalar_bytes(CONV1_OUT_CH, 0.0),
        )
        .unwrap();
        let conv2_w: Vec<u8> = (0..(CONV2_OUT_CH * CONV2_IN_CH * CONV_KERNEL_LATER))
            .flat_map(|i| ((i as f32) * 0.0001).to_le_bytes())
            .collect();
        b.add_tensor(
            "sincnet.conv1d.2.weight",
            GgmlType::F32,
            vec![
                CONV2_OUT_CH as u64,
                CONV2_IN_CH as u64,
                CONV_KERNEL_LATER as u64,
            ],
            conv2_w,
        )
        .unwrap();
        b.add_tensor(
            "sincnet.conv1d.2.bias",
            GgmlType::F32,
            vec![CONV2_OUT_CH as u64],
            scalar_bytes(CONV2_OUT_CH, 0.0),
        )
        .unwrap();

        // (d) Downstream BiLSTM / Linear / Classifier — the SincNet
        // primitive doesn't consume these, but PyanNetWeights::from_gguf
        // needs at least one tensor from each prefix. Include the
        // minimum for the runtime binder to accept the file. Payloads
        // are all-zero.
        let lstm_h: usize = DEFAULT_LSTM_HIDDEN_SIZE as usize;
        b.add_tensor(
            "lstm.weight_ih_l0",
            GgmlType::F32,
            vec![(4 * lstm_h) as u64, CONV2_OUT_CH as u64],
            scalar_bytes(4 * lstm_h * CONV2_OUT_CH, 0.0),
        )
        .unwrap();
        b.add_tensor(
            "linear.0.weight",
            GgmlType::F32,
            vec![DEFAULT_LINEAR_HIDDEN_SIZE as u64, (2 * lstm_h) as u64],
            scalar_bytes(DEFAULT_LINEAR_HIDDEN_SIZE as usize * 2 * lstm_h, 0.0),
        )
        .unwrap();
        b.add_tensor(
            "classifier.weight",
            GgmlType::F32,
            vec![
                DEFAULT_NUM_POWERSET_CLASSES as u64,
                DEFAULT_LINEAR_HIDDEN_SIZE as u64,
            ],
            scalar_bytes(
                DEFAULT_NUM_POWERSET_CLASSES as usize * DEFAULT_LINEAR_HIDDEN_SIZE as usize,
                0.0,
            ),
        )
        .unwrap();

        let bytes = b.to_bytes().unwrap();
        let path = scratch_path("sincnet-forward");
        std::fs::write(&path, &bytes).unwrap();
        path
    }

    // -----------------------------------------------------------------------
    // Primary-source-derived pin tests
    // -----------------------------------------------------------------------

    #[test]
    fn hamming_half_window_matches_reference_analytic_values() {
        // `np.hamming(251)[i] = 0.54 - 0.46 · cos(2π·i / 250)` for
        // `i ∈ [0, 125)`. Spot-check the endpoints + a middle sample.
        let w = hamming_half_window();
        assert_eq!(w.len(), HALF_KERNEL_SINC);
        // i = 0 -> 0.54 - 0.46 * cos(0) = 0.08
        assert!(
            (w[0] - 0.08).abs() < 1e-6,
            "hamming[0] should be 0.08, got {}",
            w[0]
        );
        // i = 125 (would be centre) but half window excludes it; check
        // i = 62 (near the centre) against the analytic value.
        let expected_62 = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * 62.0 / 250.0).cos();
        assert!(
            (w[62] - expected_62).abs() < 1e-6,
            "hamming[62] should be {expected_62}, got {}",
            w[62]
        );
    }

    #[test]
    fn negative_time_vector_matches_reference_analytic_values() {
        let n = negative_time_vector();
        assert_eq!(n.len(), HALF_KERNEL_SINC);
        // n[0] corresponds to sample -125.
        let expected_0 = 2.0 * std::f32::consts::PI * (-125.0f32) / (SAMPLE_RATE_SINCNET as f32);
        assert!(
            (n[0] - expected_0).abs() < 1e-9,
            "n[0] should be {expected_0}, got {}",
            n[0]
        );
        // n[124] corresponds to sample -1.
        let expected_124 = -(2.0 * std::f32::consts::PI) / (SAMPLE_RATE_SINCNET as f32);
        assert!(
            (n[124] - expected_124).abs() < 1e-9,
            "n[124] should be {expected_124}, got {}",
            n[124]
        );
    }

    #[test]
    fn num_frames_matches_primary_source_recurrence() {
        // Values transcribed from the scout output — pin-tested against
        // the algebraic recurrence `multi_conv_num_frames` from
        // `receptive_field.py:56-69`.
        let stride = DEFAULT_SINCNET_STRIDE as usize;
        assert_eq!(num_frames(160_000, stride), 589, "10 s @ 16 kHz");
        assert_eq!(num_frames(16_000, stride), 56, "1 s @ 16 kHz");
        // 0.1 s @ 16 kHz. Recurrence: 1600 -> 135 -> 45 -> 41 -> 13 -> 9 -> 3.
        assert_eq!(num_frames(1_600, stride), 3, "0.1 s @ 16 kHz");
        assert_eq!(num_frames(500, stride), 0, "sub-kernel input → 0 frames");
        assert_eq!(num_frames(0, stride), 0);
    }

    #[test]
    fn conv1d_out_len_matches_reference_formula() {
        // 1 + (N - k) / s
        assert_eq!(conv1d_out_len(160_000, 251, 10), 15_975);
        assert_eq!(conv1d_out_len(15_975, 3, 3), 5_325);
        assert_eq!(conv1d_out_len(5_325, 5, 1), 5_321);
        assert_eq!(conv1d_out_len(5_321, 3, 3), 1_773);
        assert_eq!(conv1d_out_len(1_773, 5, 1), 1_769);
        assert_eq!(conv1d_out_len(1_769, 3, 3), 589);
        // Sub-kernel input yields 0 frames.
        assert_eq!(conv1d_out_len(250, 251, 10), 0);
        // Boundary: input == kernel.
        assert_eq!(conv1d_out_len(251, 251, 10), 1);
    }

    #[test]
    fn instance_norm_1_channel_normalises_to_zero_mean_unit_var() {
        let mut x: Vec<f32> = (0..100).map(|i| (i as f32) - 50.0).collect();
        instance_norm_1_channel(&mut x, 1.0, 0.0);
        let n = x.len() as f32;
        let mean: f32 = x.iter().sum::<f32>() / n;
        let var: f32 = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
        assert!(mean.abs() < 1e-3, "mean should be ~0, got {mean}");
        assert!((var - 1.0).abs() < 1e-2, "var should be ~1, got {var}");
    }

    #[test]
    fn instance_norm_and_leaky_normalises_per_channel_and_applies_leaky() {
        // 2 channels, 4 samples each. Channel 0 = [1, 2, 3, 4], channel
        // 1 = [-4, -3, -2, -1]. After normalisation both channels have
        // mean 0 and unit variance; positive taps pass, negatives get
        // multiplied by 0.01.
        let mut x = vec![1.0, 2.0, 3.0, 4.0, -4.0, -3.0, -2.0, -1.0];
        let gamma = vec![1.0, 1.0];
        let beta = vec![0.0, 0.0];
        instance_norm_and_leaky(&mut x, 2, 4, &gamma, &beta);
        // Each channel's mean should be ~0 (post-affine, pre-leaky).
        // After leaky the negative halves are scaled by 0.01.
        for c in 0..2 {
            let row: Vec<f32> = x[c * 4..(c + 1) * 4].to_vec();
            let has_positive = row.iter().any(|&v| v > 0.0);
            let has_negative = row.iter().any(|&v| v < 0.0);
            assert!(has_positive, "channel {c} must have a positive sample");
            assert!(has_negative, "channel {c} must have a negative sample");
        }
    }

    #[test]
    fn max_pool_1d_channelwise_matches_reference_algorithm() {
        // 2 channels, in_time=6, kernel=3, stride=3 -> out_time=2.
        // Channel 0: [1,3,2, 5,4,6] -> [3, 6]
        // Channel 1: [-1,-2,0, 10,5,7] -> [0, 10]
        let src = vec![
            1.0, 3.0, 2.0, 5.0, 4.0, 6.0, -1.0, -2.0, 0.0, 10.0, 5.0, 7.0,
        ];
        let mut dst = vec![0.0f32; 4];
        max_pool_1d_channelwise(&src, 2, 6, &mut dst, 2);
        assert_eq!(dst, vec![3.0, 6.0, 0.0, 10.0]);
    }

    // -----------------------------------------------------------------------
    // End-to-end SincNet forward tests
    // -----------------------------------------------------------------------

    #[test]
    fn sincnet_from_weights_binds_all_required_tensors() {
        let path = synthetic_sincnet_gguf();
        let g = GgufFile::open(&path).unwrap();
        let w = PyanNetWeights::from_gguf(&g).expect("bind");
        let cfg = PyanNetConfig::from_gguf(&g);
        let sn = SincNet::from_weights(&w, cfg.sincnet_stride as usize).expect("build sincnet");
        assert_eq!(sn.n_learnable_filters(), N_FILTERS_SINC / 2);
        assert_eq!(sn.stride(), cfg.sincnet_stride as usize);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn sincnet_from_weights_rejects_shape_mismatch_loudly() {
        // Deliberately mis-shape one tensor and confirm loud
        // `VokraError::ModelLoad` with FR-EX-08 tag.
        use crate::pyannote::{
            DEFAULT_SINCNET_STRIDE, GGUF_KEY_SAMPLE_RATE, GGUF_KEY_SINCNET_STRIDE,
        };
        let mut b = GgufBuilder::new();
        b.add_u32(GGUF_KEY_SAMPLE_RATE, 16000);
        b.add_u32(GGUF_KEY_SINCNET_STRIDE, DEFAULT_SINCNET_STRIDE);
        // Wrong-shape sinc filterbank: shape [10, 1] instead of [40, 1].
        b.add_tensor(
            "sincnet.conv1d.0.filterbank.low_hz_",
            GgmlType::F32,
            vec![10, 1],
            vec![0u8; 10 * 4],
        )
        .unwrap();
        // Add enough of the other tensors to pass `PyanNetWeights` bind.
        b.add_tensor(
            "lstm.weight_ih_l0",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .unwrap();
        b.add_tensor("linear.0.weight", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .unwrap();
        b.add_tensor(
            "classifier.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .unwrap();
        let bytes = b.to_bytes().unwrap();
        let path = scratch_path("sincnet-shape-mismatch");
        std::fs::write(&path, &bytes).unwrap();
        let g = GgufFile::open(&path).unwrap();
        let w = PyanNetWeights::from_gguf(&g).expect("bind");
        let err = SincNet::from_weights(&w, DEFAULT_SINCNET_STRIDE as usize).unwrap_err();
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("low_hz_") && msg.contains("FR-EX-08"),
                    "shape-mismatch error must name the tensor + FR-EX-08 tag: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn sincnet_forward_produces_expected_output_shape() {
        let path = synthetic_sincnet_gguf();
        let g = GgufFile::open(&path).unwrap();
        let w = PyanNetWeights::from_gguf(&g).unwrap();
        let cfg = PyanNetConfig::from_gguf(&g);
        let sn = SincNet::from_weights(&w, cfg.sincnet_stride as usize).unwrap();

        // 1 s of 16 kHz sine at 440 Hz — well within the sinc filter
        // band and long enough to survive every downstream pool.
        let sr = SAMPLE_RATE_SINCNET as f32;
        let pcm: Vec<f32> = (0..SAMPLE_RATE_SINCNET as usize)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / sr).sin())
            .collect();

        let out = sn.forward(&pcm, SAMPLE_RATE_SINCNET).expect("forward");
        assert_eq!(out.num_channels, CONV2_OUT_CH);
        assert_eq!(
            out.num_frames,
            num_frames(pcm.len(), cfg.sincnet_stride as usize)
        );
        assert_eq!(out.features.len(), out.num_channels * out.num_frames);
        // Every value must be finite (no NaN / inf leaked from the sinc
        // synthesis or the InstanceNorm divide).
        for &v in &out.features {
            assert!(v.is_finite(), "SincNet output contains non-finite: {v}");
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn sincnet_forward_rejects_non_16khz_input_loudly() {
        let path = synthetic_sincnet_gguf();
        let g = GgufFile::open(&path).unwrap();
        let w = PyanNetWeights::from_gguf(&g).unwrap();
        let cfg = PyanNetConfig::from_gguf(&g);
        let sn = SincNet::from_weights(&w, cfg.sincnet_stride as usize).unwrap();
        let pcm = vec![0.0f32; 8000];
        let err = sn.forward(&pcm, 8000).unwrap_err();
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains("16000") && msg.contains("FR-EX-08"),
                    "wrong-sr error must name 16000 + FR-EX-08: {msg}"
                );
            }
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn sincnet_forward_rejects_short_input_loudly() {
        let path = synthetic_sincnet_gguf();
        let g = GgufFile::open(&path).unwrap();
        let w = PyanNetWeights::from_gguf(&g).unwrap();
        let cfg = PyanNetConfig::from_gguf(&g);
        let sn = SincNet::from_weights(&w, cfg.sincnet_stride as usize).unwrap();
        let pcm = vec![0.0f32; 100]; // < KERNEL_SIZE_SINC = 251
        let err = sn.forward(&pcm, SAMPLE_RATE_SINCNET).unwrap_err();
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains("FR-EX-08"),
                    "short-input error must have FR-EX-08 tag: {msg}"
                );
            }
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn synthesise_sinc_filters_returns_expected_kernel_layout() {
        let path = synthetic_sincnet_gguf();
        let g = GgufFile::open(&path).unwrap();
        let w = PyanNetWeights::from_gguf(&g).unwrap();
        let cfg = PyanNetConfig::from_gguf(&g);
        let sn = SincNet::from_weights(&w, cfg.sincnet_stride as usize).unwrap();

        let filters = sn.synthesise_sinc_filters();
        // (n_filters, 1, kernel) row-major -> length 80 * 1 * 251.
        assert_eq!(filters.len(), N_FILTERS_SINC * KERNEL_SIZE_SINC);
        // Every filter has a finite centre tap.
        for i in 0..N_FILTERS_SINC {
            let centre = filters[i * KERNEL_SIZE_SINC + HALF_KERNEL_SINC];
            assert!(
                centre.is_finite(),
                "filter {i} centre tap must be finite, got {centre}"
            );
        }
        // Symmetric filter: kernel is real-valued bandpass, so
        // `filter[i][t] == filter[i][kernel-1-t]` (both halves mirror
        // via the synthesis logic).
        for i in 0..N_FILTERS_SINC {
            for t in 0..HALF_KERNEL_SINC {
                let left = filters[i * KERNEL_SIZE_SINC + t];
                let right = filters[i * KERNEL_SIZE_SINC + KERNEL_SIZE_SINC - 1 - t];
                assert!(
                    (left - right).abs() < 1e-6,
                    "filter {i} not symmetric at tap {t}: {left} vs {right}"
                );
            }
        }
    }
}
