//! `denoise` — speech enhancement (M4-20 (c), FR-OP-61): the native
//! DeepFilterNet3 denoiser (CLAUDE.md「denoise — DeepFilterNet (MIT) /
//! GTCRN / RNNoise (BSD) 統合」). whisper.cpp pattern — no ONNX / PyTorch at
//! runtime (NFR-DS-02).
//!
//! # Runtime function, not an `OpKind` variant (ADR M4-20 §D-5)
//!
//! Like `agc` / `hpf` / `loudness_norm`, `denoise` is a first-class API
//! function, not an `OpKind` variant / `dispatch.rs` arm (a graph-side call →
//! `UnsupportedOp` default, FR-EX-08).
//!
//! # Topology (upstream Rikorose/DeepFilterNet — transcribed, not invented)
//!
//! This is a faithful Rust port of the published DeepFilterNet3 inference
//! graph. Formula sources are cited per stage; "libDF" = the upstream
//! `libDF/src/lib.rs` DSP crate, "df/…" = the upstream Python package.
//!
//! 1. **Streaming STFT** (libDF `frame_analysis`, lib.rs L356-394): per
//!    `hop`-sample chunk, `[prev hop, cur hop] × vorbis_window → RFFT ×
//!    wnorm` with `wnorm = 2·hop/n_fft²` applied in analysis only. The
//!    window is the Vorbis window `sin(π/2·sin²(π(i+0.5)/n_fft))` (lib.rs
//!    L127-132), which satisfies the Princen-Bradley condition, so
//!    synthesis applies the same window again and overlap-adds to unity.
//! 2. **ERB features** (libDF `feat_erb`, lib.rs L206-212): per-band mean
//!    power over the libDF ERB partition (`erb_fb`, lib.rs L68-100) →
//!    `10·log10(x + 1e-10)` → exponential mean-norm
//!    (`band_mean_norm_erb`, lib.rs L244-251: `s ← x(1−α) + sα; x ←
//!    (x−s)/40`, state init `linspace(-60, -90)`).
//! 3. **Complex spec features** (libDF `feat_cplx` / `band_unit_norm`,
//!    lib.rs L253-259): the low `df_bins` complex bins unit-normed by an
//!    exponential magnitude estimate (`s ← |x|(1−α) + sα; x ← x/√s`, state
//!    init `linspace(0.001, 0.0001)`).
//! 4. **DfNet** (df/deepfilternet3.py): conv encoder over ERB features + a
//!    dedicated complex-feature conv branch, squeezed GRUs
//!    (grouped-linear in → GRU → grouped-linear out, df/modules.py
//!    `SqueezedGRU_S`), an ERB mask decoder (transposed-conv U-Net with
//!    pathway convs, sigmoid gains expanded through `mask.erb_inv_fb`),
//!    and a deep-filter decoder producing order-`df_order` complex FIR
//!    coefficients for the low `df_bins` bins with `df_lookahead` frames
//!    of lookahead (df/multiframe.py `DF`).
//! 5. **Streaming iSTFT** (libDF `frame_synthesis`, lib.rs L396-427) and
//!    the `n_fft − hop` delay trim of `df/enhance.py` L229-248.
//!
//! Numeric parity against the real DeepFilterNet3 checkpoint is exercised
//! stage-by-stage in `tests/parity_denoise_dfn3.rs` (env-gated on the real
//! GGUF + reference taps) and at primitive level in
//! `tests/parity_denoise_primitives.rs` (committed torch fixtures).
//!
//! # License
//!
//! Upstream Rikorose/DeepFilterNet is dual MIT / Apache-2.0 (code AND the
//! released DeepFilterNet3 checkpoint distributed in-repo under
//! `models/DeepFilterNet3.zip`) — see `docs/license-audit.md`.

use std::collections::BTreeMap;

use vokra_core::rng::SplitMix64;
use vokra_core::{Result, VokraError};

use crate::fft::{Complex32, RealFftPlan};
use crate::stft::Spectrogram;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// DeepFilterNet3 architecture / frontend configuration.
///
/// [`DeepFilterNetConfig::deep_filter_net3`] carries the published
/// DeepFilterNet3 hyper-parameters (the `[df]` / `[deepfilternet]` sections
/// of the released `config.ini`); every field is validated against the
/// checkpoint tensor shapes at load.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeepFilterNetConfig {
    /// STFT size (`fft_size`).
    pub n_fft: usize,
    /// STFT hop (`hop_size`). libDF requires `2·hop ≤ n_fft`.
    pub hop: usize,
    /// Sample rate (Hz).
    pub sample_rate: u32,
    /// Number of ERB bands (`nb_erb`).
    pub n_erb: usize,
    /// Number of low-frequency bins the deep-filter stage refines (`nb_df`).
    pub df_bins: usize,
    /// Deep-filter order (FIR taps over past + current + lookahead frames).
    pub df_order: usize,
    /// Minimum number of FFT bins per ERB band (`min_nb_erb_freqs`,
    /// libDF `erb_fb` redistribution).
    pub min_nb_erb_freqs: usize,
    /// Feature lookahead in frames (`conv_lookahead`): features are shifted
    /// this many frames into the future relative to the spectrum
    /// (`DfNet.pad_feat`, df/deepfilternet3.py L357-361).
    pub conv_lookahead: usize,
    /// Deep-filter lookahead in frames (`df_lookahead`): the FIR window for
    /// output frame `t` covers `t − (df_order − 1 − df_lookahead) ..= t +
    /// df_lookahead` (df/multiframe.py `MultiFrameModule.__init__`).
    pub df_lookahead: usize,
    /// Conv channel width (`conv_ch`).
    pub conv_ch: usize,
    /// Embedding GRU hidden size (`emb_hidden_dim`).
    pub emb_hidden: usize,
    /// Deep-filter GRU hidden size (`df_hidden_dim`).
    pub df_hidden: usize,
    /// Groups of the encoder `df_fc_emb` grouped linear
    /// (`enc_linear_groups`).
    pub enc_linear_groups: usize,
    /// Groups of every other grouped linear (`linear_groups`).
    pub linear_groups: usize,
    /// Groups of the df_gru squeeze linears. Not in `config.ini` — it is
    /// the `SqueezedGRU_S` signature default 8 (df/modules.py L712), which
    /// `DfDecoder.__init__` does not override.
    pub df_gru_linear_groups: usize,
    /// Total embedding-GRU depth (`emb_num_layers`): 1 encoder layer +
    /// `emb_num_layers − 1` ERB-decoder layers (df/deepfilternet3.py L216).
    pub emb_num_layers: usize,
    /// Deep-filter GRU depth (`df_num_layers`).
    pub df_num_layers: usize,
    /// Local-SNR head lower bound in dB (`lsnr_min`).
    pub lsnr_min: f32,
    /// Local-SNR head upper bound in dB (`lsnr_max`).
    pub lsnr_max: f32,
    /// Exponential-norm decay `α` for the ERB / unit norms. Upstream
    /// `get_norm_alpha` (df/utils.py L111-127): `round(exp(−hop/sr/τ), 3)`
    /// with `norm_tau = 1` → `0.99` for the DFN3 frontend.
    pub norm_alpha: f32,
}

impl DeepFilterNetConfig {
    /// DeepFilterNet3 published hyper-parameters (released `config.ini`).
    pub fn deep_filter_net3() -> Self {
        Self {
            n_fft: 960,
            hop: 480,
            sample_rate: 48000,
            n_erb: 32,
            df_bins: 96,
            df_order: 5,
            min_nb_erb_freqs: 2,
            conv_lookahead: 2,
            df_lookahead: 2,
            conv_ch: 64,
            emb_hidden: 256,
            df_hidden: 256,
            enc_linear_groups: 32,
            linear_groups: 16,
            df_gru_linear_groups: 8,
            emb_num_layers: 3,
            df_num_layers: 2,
            lsnr_min: -15.0,
            lsnr_max: 35.0,
            norm_alpha: 0.99,
        }
    }

    /// Number of one-sided (RFFT) frequency bins, `n_fft / 2 + 1`.
    pub fn n_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Flattened embedding width `conv_ch · n_erb / 4` (two stride-2 conv
    /// stages over the ERB axis).
    pub fn emb_dim(&self) -> usize {
        self.conv_ch * self.n_erb / 4
    }

    /// The flattened df-branch width `conv_ch · df_bins / 2` (one stride-2
    /// stage over the complex-feature axis).
    fn cemb_dim(&self) -> usize {
        self.conv_ch * self.df_bins / 2
    }

    /// Per-frame deep-filter output width `df_bins · df_order · 2`.
    fn df_out_dim(&self) -> usize {
        self.df_bins * self.df_order * 2
    }

    /// The libDF ERB filter bank: FFT-bin width of each of the `n_erb`
    /// bands (libDF `erb_fb`, lib.rs L68-100 — cumulative
    /// `round(erb2freq/freq_width)` boundaries with a `min_nb_erb_freqs`
    /// redistribution; the final band absorbs the `n_fft/2 + 1`-th bin).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] when the partition cannot cover
    /// `n_bins` bins (config inconsistent).
    pub fn erb_widths(&self) -> Result<Vec<usize>> {
        let nyq = self.sample_rate as f32 / 2.0;
        let freq_width = self.sample_rate as f32 / self.n_fft as f32;
        let erb_low = freq2erb(0.0);
        let erb_high = freq2erb(nyq);
        let nb = self.n_erb;
        let mut widths = vec![0usize; nb];
        let step = (erb_high - erb_low) / nb as f32;
        let min_nb = self.min_nb_erb_freqs as i64;
        let mut prev_freq: i64 = 0;
        let mut freq_over: i64 = 0;
        for i in 1..=nb {
            let f = erb2freq(erb_low + i as f32 * step);
            let fb = (f / freq_width).round() as i64;
            let mut nb_freqs = fb - prev_freq - freq_over;
            if nb_freqs < min_nb {
                freq_over = min_nb - nb_freqs;
                nb_freqs = min_nb;
            } else {
                freq_over = 0;
            }
            widths[i - 1] = nb_freqs as usize;
            prev_freq = fb;
        }
        // One-sided spectrum has n_fft/2 + 1 bins: widen the last band by
        // one, then clip any overshoot (libDF lib.rs L93-97).
        widths[nb - 1] += 1;
        let sum: usize = widths.iter().sum();
        let n_bins = self.n_bins();
        if sum < n_bins {
            return Err(VokraError::InvalidArgument(format!(
                "denoise: ERB partition covers {sum} of {n_bins} bins (config inconsistent)"
            )));
        }
        let too_large = sum - n_bins;
        if too_large > 0 {
            if widths[nb - 1] <= too_large {
                return Err(VokraError::InvalidArgument(
                    "denoise: ERB partition overshoot exceeds the last band".into(),
                ));
            }
            widths[nb - 1] -= too_large;
        }
        debug_assert_eq!(widths.iter().sum::<usize>(), n_bins);
        Ok(widths)
    }

    fn validate(&self) -> Result<()> {
        let bad = |msg: String| Err(VokraError::InvalidArgument(format!("denoise: {msg}")));
        if self.n_fft == 0 || self.hop == 0 || self.sample_rate == 0 {
            return bad("n_fft / hop / sample_rate must be > 0".into());
        }
        if self.hop * 2 > self.n_fft {
            // libDF DFState::new asserts hop_size * 2 <= fft_size.
            return bad(format!(
                "hop {} must satisfy 2·hop ≤ n_fft {}",
                self.hop, self.n_fft
            ));
        }
        if self.n_erb == 0 || self.n_erb % 8 != 0 {
            // Upstream asserts nb_erb % 8 == 0 (DfNet / ErbDecoder).
            return bad(format!(
                "n_erb {} must be a positive multiple of 8",
                self.n_erb
            ));
        }
        if self.df_bins < 2 || self.df_bins % 2 != 0 || self.df_bins > self.n_bins() {
            return bad(format!(
                "df_bins {} must be even and ≤ n_bins {}",
                self.df_bins,
                self.n_bins()
            ));
        }
        if self.df_order == 0 || self.df_lookahead >= self.df_order {
            return bad(format!(
                "df_order {} must be > df_lookahead {}",
                self.df_order, self.df_lookahead
            ));
        }
        if self.min_nb_erb_freqs == 0
            || self.conv_ch == 0
            || self.emb_hidden == 0
            || self.df_hidden == 0
        {
            return bad("min_nb_erb_freqs / conv_ch / emb_hidden / df_hidden must be > 0".into());
        }
        if self.emb_num_layers < 2 || self.df_num_layers == 0 {
            return bad(format!(
                "emb_num_layers {} must be ≥ 2 and df_num_layers {} ≥ 1",
                self.emb_num_layers, self.df_num_layers
            ));
        }
        let div = |what: &str, n: usize, g: usize| -> Result<()> {
            if g == 0 || n % g != 0 {
                Err(VokraError::InvalidArgument(format!(
                    "denoise: {what} {n} must be divisible by its group count {g}"
                )))
            } else {
                Ok(())
            }
        };
        div("emb_dim", self.emb_dim(), self.linear_groups)?;
        div("emb_hidden", self.emb_hidden, self.linear_groups)?;
        div("df_hidden", self.df_hidden, self.linear_groups)?;
        div("df_out_dim", self.df_out_dim(), self.linear_groups)?;
        div("emb_dim", self.emb_dim(), self.enc_linear_groups)?;
        div("cemb_dim", self.cemb_dim(), self.enc_linear_groups)?;
        div("emb_dim", self.emb_dim(), self.df_gru_linear_groups)?;
        div("df_hidden", self.df_hidden, self.df_gru_linear_groups)?;
        if !(self.norm_alpha > 0.0 && self.norm_alpha < 1.0) {
            return bad(format!("norm_alpha {} must be in (0, 1)", self.norm_alpha));
        }
        if self.lsnr_max <= self.lsnr_min || !(self.lsnr_max - self.lsnr_min).is_finite() {
            return bad(format!(
                "lsnr_max {} must exceed lsnr_min {}",
                self.lsnr_max, self.lsnr_min
            ));
        }
        Ok(())
    }
}

/// libDF `freq2erb` (lib.rs L42-44).
fn freq2erb(freq_hz: f32) -> f32 {
    9.265 * (freq_hz / (24.7 * 9.265)).ln_1p()
}

/// libDF `erb2freq` (lib.rs L45-47).
fn erb2freq(n_erb: f32) -> f32 {
    24.7 * 9.265 * ((n_erb / 9.265).exp() - 1.0)
}

/// The Vorbis window `sin(π/2 · sin²(π(i+0.5)/n_fft))`, computed in f64 like
/// libDF (lib.rs L126-132).
fn vorbis_window(n_fft: usize) -> Vec<f32> {
    let pi = std::f64::consts::PI;
    let half = (n_fft / 2) as f64;
    (0..n_fft)
        .map(|i| {
            let s = (0.5 * pi * (i as f64 + 0.5) / half).sin();
            (0.5 * pi * s * s).sin() as f32
        })
        .collect()
}

/// ndarray-style linspace (libDF transforms.rs erb_norm/unit_norm inits).
fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f32;
    (0..n).map(|i| start + i as f32 * step).collect()
}

// ---------------------------------------------------------------------------
// NN primitives (weights are checkpoint tensors, layouts cited per struct)
// ---------------------------------------------------------------------------

/// `[C, T, F]` activation tensor (channel-major, row-major within a channel).
#[derive(Debug, Clone)]
struct Act3 {
    ch: usize,
    t: usize,
    f: usize,
    data: Vec<f32>,
}

impl Act3 {
    fn zeros(ch: usize, t: usize, f: usize) -> Self {
        Self {
            ch,
            t,
            f,
            data: vec![0.0; ch * t * f],
        }
    }

    #[inline]
    fn idx(&self, c: usize, t: usize, f: usize) -> usize {
        (c * self.t + t) * self.f + f
    }

    #[inline]
    fn at(&self, c: usize, t: usize, f: usize) -> f32 {
        self.data[self.idx(c, t, f)]
    }
}

/// Post-conv per-channel affine (eval-mode `BatchNorm2d` folded at load:
/// `scale = γ/√(σ² + 1e-5)`, `shift = β − μ·scale`) + activation.
#[derive(Debug, Clone)]
enum Act {
    Relu,
    Sigmoid,
}

/// Grouped `Conv2d` over `[C, T, F]` with causal time padding `kt − 1`
/// (`Conv2dNormAct`'s `ConstantPad2d((0, 0, kt−1, 0))`, df/modules.py
/// L36-48) and symmetric frequency padding `fpad`.
///
/// Weight layout: `[out_ch, in_ch/groups, kt, kf]` (torch `Conv2d.weight`).
#[derive(Debug, Clone)]
struct Conv2d {
    w: Vec<f32>,
    in_ch: usize,
    out_ch: usize,
    groups: usize,
    kt: usize,
    kf: usize,
    fstride: usize,
    fpad: usize,
}

impl Conv2d {
    fn out_f(&self, f_in: usize) -> usize {
        (f_in + 2 * self.fpad - self.kf) / self.fstride + 1
    }

    fn forward(&self, x: &Act3) -> Act3 {
        debug_assert_eq!(x.ch, self.in_ch);
        let f_out = self.out_f(x.f);
        let mut y = Act3::zeros(self.out_ch, x.t, f_out);
        let in_g = self.in_ch / self.groups;
        let out_g = self.out_ch / self.groups;
        let w_per_out = in_g * self.kt * self.kf;
        for o in 0..self.out_ch {
            let g = o / out_g;
            let w_base = o * w_per_out;
            for t in 0..x.t {
                for fo in 0..f_out {
                    let mut acc = 0.0f32;
                    for igl in 0..in_g {
                        let ic = g * in_g + igl;
                        for dt in 0..self.kt {
                            // Causal pad: kt−1 zero frames before t=0.
                            let Some(ts) = (t + dt).checked_sub(self.kt - 1) else {
                                continue;
                            };
                            for df in 0..self.kf {
                                let fs = fo * self.fstride + df;
                                let Some(fs) = fs.checked_sub(self.fpad) else {
                                    continue;
                                };
                                if fs >= x.f {
                                    continue;
                                }
                                let wi = w_base + (igl * self.kt + dt) * self.kf + df;
                                acc += self.w[wi] * x.at(ic, ts, fs);
                            }
                        }
                    }
                    let yi = y.idx(o, t, fo);
                    y.data[yi] = acc;
                }
            }
        }
        y
    }
}

/// Grouped `ConvTranspose2d` with time kernel 1 (frequency upsampling only —
/// `ConvTranspose2dNormAct` with `convt_kernel = (1, kf)`, df/modules.py
/// L75-126: `padding = (0, kf/2)`, `output_padding = (0, kf/2)`).
///
/// Weight layout: `[in_ch, out_ch/groups, 1, kf]` (torch
/// `ConvTranspose2d.weight`).
#[derive(Debug, Clone)]
struct ConvT2dF {
    w: Vec<f32>,
    in_ch: usize,
    out_ch: usize,
    groups: usize,
    kf: usize,
    fstride: usize,
    fpad: usize,
    out_pad: usize,
}

impl ConvT2dF {
    fn out_f(&self, f_in: usize) -> usize {
        (f_in - 1) * self.fstride + self.kf + self.out_pad - 2 * self.fpad
    }

    fn forward(&self, x: &Act3) -> Act3 {
        debug_assert_eq!(x.ch, self.in_ch);
        let f_out = self.out_f(x.f);
        let mut y = Act3::zeros(self.out_ch, x.t, f_out);
        let in_g = self.in_ch / self.groups;
        let out_g = self.out_ch / self.groups;
        for ic in 0..self.in_ch {
            let g = ic / in_g;
            for og in 0..out_g {
                let o = g * out_g + og;
                let w_base = (ic * out_g + og) * self.kf;
                for t in 0..x.t {
                    for fi in 0..x.f {
                        let xv = x.at(ic, t, fi);
                        for k in 0..self.kf {
                            let fo = fi * self.fstride + k;
                            let Some(fo) = fo.checked_sub(self.fpad) else {
                                continue;
                            };
                            if fo >= f_out {
                                continue;
                            }
                            let yi = y.idx(o, t, fo);
                            y.data[yi] += self.w[w_base + k] * xv;
                        }
                    }
                }
            }
        }
        y
    }
}

#[derive(Debug, Clone)]
enum ConvUnit {
    Std(Conv2d),
    Transposed(ConvT2dF),
}

/// One upstream `Conv2dNormAct` / `ConvTranspose2dNormAct` block: (transposed)
/// conv → optional 1×1 pointwise (the `separable=True` tail) → folded
/// BatchNorm → activation.
#[derive(Debug, Clone)]
struct ConvBlock {
    conv: ConvUnit,
    /// `[out_ch, mid_ch]` pointwise weight (torch `Conv2d(out, out, 1)`).
    pointwise: Option<Vec<f32>>,
    bn_scale: Vec<f32>,
    bn_shift: Vec<f32>,
    act: Act,
}

impl ConvBlock {
    fn forward(&self, x: &Act3) -> Act3 {
        let mid = match &self.conv {
            ConvUnit::Std(c) => c.forward(x),
            ConvUnit::Transposed(c) => c.forward(x),
        };
        let mut y = match &self.pointwise {
            Some(w) => {
                // 1×1 conv: per (t, f) channel matmul.
                let mut y = Act3::zeros(mid.ch, mid.t, mid.f);
                let plane = mid.t * mid.f;
                for o in 0..mid.ch {
                    let w_row = &w[o * mid.ch..(o + 1) * mid.ch];
                    let y_plane = &mut y.data[o * plane..(o + 1) * plane];
                    for (ic, &wv) in w_row.iter().enumerate() {
                        let x_plane = &mid.data[ic * plane..(ic + 1) * plane];
                        for (yv, &xv) in y_plane.iter_mut().zip(x_plane) {
                            *yv += wv * xv;
                        }
                    }
                }
                y
            }
            None => mid,
        };
        let plane = y.t * y.f;
        for c in 0..y.ch {
            let (s, b) = (self.bn_scale[c], self.bn_shift[c]);
            for v in &mut y.data[c * plane..(c + 1) * plane] {
                let z = *v * s + b;
                *v = match self.act {
                    Act::Relu => z.max(0.0),
                    Act::Sigmoid => sigmoid(z),
                };
            }
        }
        y
    }
}

/// `GroupedLinearEinsum` (df/modules.py L741-776): weight `[G, I/G, H/G]`,
/// group `g` maps input slice `[g·I/G, (g+1)·I/G)` to output slice
/// `[g·H/G, (g+1)·H/G)`.
#[derive(Debug, Clone)]
struct GroupedLinear {
    w: Vec<f32>,
    groups: usize,
    in_dim: usize,
    out_dim: usize,
}

impl GroupedLinear {
    fn forward_frame(&self, x: &[f32], out: &mut [f32]) {
        let ig = self.in_dim / self.groups;
        let og = self.out_dim / self.groups;
        for g in 0..self.groups {
            let xg = &x[g * ig..(g + 1) * ig];
            let outg = &mut out[g * og..(g + 1) * og];
            let wg = &self.w[g * ig * og..(g + 1) * ig * og];
            outg.fill(0.0);
            for (i, &xv) in xg.iter().enumerate() {
                let w_row = &wg[i * og..(i + 1) * og];
                for (o, &wv) in outg.iter_mut().zip(w_row) {
                    *o += xv * wv;
                }
            }
        }
    }

    /// `[T, in] → [T, out]`.
    fn forward_seq(&self, x: &[f32], t_len: usize) -> Vec<f32> {
        let mut out = vec![0.0; t_len * self.out_dim];
        for t in 0..t_len {
            self.forward_frame(
                &x[t * self.in_dim..(t + 1) * self.in_dim],
                &mut out[t * self.out_dim..(t + 1) * self.out_dim],
            );
        }
        out
    }
}

/// One `nn.GRU` layer with the PyTorch gate semantics (gate chunk order
/// `[r, z, n]`; `h ← (1−z)∘n + z∘h`), zero initial state.
///
/// Weight layouts: `weight_ih_l{k}` `[3H, I]`, `weight_hh_l{k}` `[3H, H]`,
/// biases `[3H]`. Scalar loops — correctness over speed (the BiLstm1d
/// precedent from Kokoro).
#[derive(Debug, Clone)]
struct GruLayer {
    w_ih: Vec<f32>,
    w_hh: Vec<f32>,
    b_ih: Vec<f32>,
    b_hh: Vec<f32>,
    input: usize,
    hidden: usize,
}

impl GruLayer {
    /// `[T, input] → [T, hidden]`.
    fn forward_seq(&self, x: &[f32], t_len: usize) -> Vec<f32> {
        let h_dim = self.hidden;
        let mut h = vec![0.0f32; h_dim];
        let mut out = vec![0.0f32; t_len * h_dim];
        let mut xg = vec![0.0f32; 3 * h_dim];
        let mut hg = vec![0.0f32; 3 * h_dim];
        for t in 0..t_len {
            let xt = &x[t * self.input..(t + 1) * self.input];
            for (row, (xv, bi)) in xg.iter_mut().zip(self.b_ih.iter()).enumerate() {
                let w_row = &self.w_ih[row * self.input..(row + 1) * self.input];
                let mut acc = *bi;
                for (&wv, &iv) in w_row.iter().zip(xt) {
                    acc += wv * iv;
                }
                *xv = acc;
            }
            for (row, (hv, bh)) in hg.iter_mut().zip(self.b_hh.iter()).enumerate() {
                let w_row = &self.w_hh[row * h_dim..(row + 1) * h_dim];
                let mut acc = *bh;
                for (&wv, &sv) in w_row.iter().zip(h.iter()) {
                    acc += wv * sv;
                }
                *hv = acc;
            }
            let ht = &mut out[t * h_dim..(t + 1) * h_dim];
            for j in 0..h_dim {
                let r = sigmoid(xg[j] + hg[j]);
                let z = sigmoid(xg[h_dim + j] + hg[h_dim + j]);
                let n = (xg[2 * h_dim + j] + r * hg[2 * h_dim + j]).tanh();
                let hv = (1.0 - z) * n + z * h[j];
                ht[j] = hv;
            }
            h.copy_from_slice(ht);
        }
        out
    }
}

/// `SqueezedGRU_S` (df/modules.py L702-738): `relu(grouped_linear_in) → GRU
/// stack → relu(grouped_linear_out)` (the skip op is `None` for every DFN3
/// instance; the df decoder's `df_skip` is a separate module).
#[derive(Debug, Clone)]
struct SqueezedGru {
    linear_in: GroupedLinear,
    layers: Vec<GruLayer>,
    linear_out: Option<GroupedLinear>,
}

impl SqueezedGru {
    /// `[T, in] → [T, out]`.
    fn forward(&self, x: &[f32], t_len: usize) -> Vec<f32> {
        let mut cur = self.linear_in.forward_seq(x, t_len);
        for v in &mut cur {
            *v = v.max(0.0);
        }
        for layer in &self.layers {
            cur = layer.forward_seq(&cur, t_len);
        }
        if let Some(lo) = &self.linear_out {
            cur = lo.forward_seq(&cur, t_len);
            for v in &mut cur {
                *v = v.max(0.0);
            }
        }
        cur
    }
}

/// `nn.Linear`: weight `[out, in]`, bias `[out]` (`out` is the caller's
/// output-slice length).
#[derive(Debug, Clone)]
struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,
    in_dim: usize,
}

impl Linear {
    fn forward_frame(&self, x: &[f32], out: &mut [f32]) {
        for (o, (ov, bv)) in out.iter_mut().zip(self.b.iter()).enumerate() {
            let w_row = &self.w[o * self.in_dim..(o + 1) * self.in_dim];
            let mut acc = *bv;
            for (&wv, &xv) in w_row.iter().zip(x) {
                acc += wv * xv;
            }
            *ov = acc;
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Tensor manifest (single source of truth for converter + loader + synth)
// ---------------------------------------------------------------------------

/// One expected checkpoint tensor: upstream name + torch shape (outermost
/// first).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// Upstream state-dict key (kept verbatim in the GGUF).
    pub name: String,
    /// Torch shape, outermost dimension first.
    pub shape: Vec<usize>,
}

fn spec(name: &str, shape: &[usize]) -> TensorSpec {
    TensorSpec {
        name: name.to_string(),
        shape: shape.to_vec(),
    }
}

/// The full inference tensor manifest of the DeepFilterNet3 checkpoint for
/// `cfg` (115 tensors at the published config). Shapes are the torch
/// `state_dict` shapes; the converter validates the checkpoint against this
/// list exactly and the loader validates every GGUF tensor against it.
pub fn denoise_tensor_manifest(cfg: &DeepFilterNetConfig) -> Vec<TensorSpec> {
    let ch = cfg.conv_ch;
    let emb = cfg.emb_dim();
    let eh = cfg.emb_hidden;
    let dh = cfg.df_hidden;
    let lg = cfg.linear_groups;
    let dg = cfg.df_gru_linear_groups;
    let out_ch_df = cfg.df_order * 2;
    let mut v = Vec::with_capacity(120);
    let bn = |v: &mut Vec<TensorSpec>, prefix: &str, ch: usize| {
        v.push(spec(&format!("{prefix}.weight"), &[ch]));
        v.push(spec(&format!("{prefix}.bias"), &[ch]));
        v.push(spec(&format!("{prefix}.running_mean"), &[ch]));
        v.push(spec(&format!("{prefix}.running_var"), &[ch]));
    };
    let gru = |v: &mut Vec<TensorSpec>, prefix: &str, layers: usize, hidden: usize| {
        for l in 0..layers {
            v.push(spec(
                &format!("{prefix}.weight_ih_l{l}"),
                &[3 * hidden, hidden],
            ));
            v.push(spec(
                &format!("{prefix}.weight_hh_l{l}"),
                &[3 * hidden, hidden],
            ));
            v.push(spec(&format!("{prefix}.bias_ih_l{l}"), &[3 * hidden]));
            v.push(spec(&format!("{prefix}.bias_hh_l{l}"), &[3 * hidden]));
        }
    };
    // --- encoder (df/deepfilternet3.py Encoder.__init__) ---
    // erb_conv0: full 3×3 conv (in 1 → gcd(1, ch) = 1 group → not separable).
    v.push(spec("enc.erb_conv0.1.weight", &[ch, 1, 3, 3]));
    bn(&mut v, "enc.erb_conv0.2", ch);
    for name in ["enc.erb_conv1", "enc.erb_conv2", "enc.erb_conv3"] {
        // Depthwise (1,3) + pointwise + BN.
        v.push(spec(&format!("{name}.0.weight"), &[ch, 1, 1, 3]));
        v.push(spec(&format!("{name}.1.weight"), &[ch, ch, 1, 1]));
        bn(&mut v, &format!("{name}.2"), ch);
    }
    // df_conv0: groups = gcd(2, ch) = 2 conv (3,3) + pointwise + BN.
    v.push(spec("enc.df_conv0.1.weight", &[ch, 1, 3, 3]));
    v.push(spec("enc.df_conv0.2.weight", &[ch, ch, 1, 1]));
    bn(&mut v, "enc.df_conv0.3", ch);
    // df_conv1: depthwise (1,3) stride 2 + pointwise + BN.
    v.push(spec("enc.df_conv1.0.weight", &[ch, 1, 1, 3]));
    v.push(spec("enc.df_conv1.1.weight", &[ch, ch, 1, 1]));
    bn(&mut v, "enc.df_conv1.2", ch);
    let elg = cfg.enc_linear_groups;
    v.push(spec(
        "enc.df_fc_emb.0.weight",
        &[elg, cfg.cemb_dim() / elg, emb / elg],
    ));
    v.push(spec(
        "enc.emb_gru.linear_in.0.weight",
        &[lg, emb / lg, eh / lg],
    ));
    gru(&mut v, "enc.emb_gru.gru", 1, eh);
    v.push(spec(
        "enc.emb_gru.linear_out.0.weight",
        &[lg, eh / lg, emb / lg],
    ));
    v.push(spec("enc.lsnr_fc.0.weight", &[1, emb]));
    v.push(spec("enc.lsnr_fc.0.bias", &[1]));
    // --- ERB decoder (ErbDecoder.__init__) ---
    v.push(spec(
        "erb_dec.emb_gru.linear_in.0.weight",
        &[lg, emb / lg, eh / lg],
    ));
    gru(&mut v, "erb_dec.emb_gru.gru", cfg.emb_num_layers - 1, eh);
    v.push(spec(
        "erb_dec.emb_gru.linear_out.0.weight",
        &[lg, eh / lg, emb / lg],
    ));
    for name in [
        "erb_dec.conv3p",
        "erb_dec.conv2p",
        "erb_dec.conv1p",
        "erb_dec.conv0p",
    ] {
        // Depthwise 1×1 pathway conv (groups = gcd(ch, ch) = ch survives the
        // max(kernel)==1 separable=False downgrade — weight [ch, 1, 1, 1]).
        v.push(spec(&format!("{name}.0.weight"), &[ch, 1, 1, 1]));
        bn(&mut v, &format!("{name}.1"), ch);
    }
    // convt3: regular depthwise (1,3) + pointwise + BN.
    v.push(spec("erb_dec.convt3.0.weight", &[ch, 1, 1, 3]));
    v.push(spec("erb_dec.convt3.1.weight", &[ch, ch, 1, 1]));
    bn(&mut v, "erb_dec.convt3.2", ch);
    for name in ["erb_dec.convt2", "erb_dec.convt1"] {
        // Depthwise transposed (1,3) fstride 2 + pointwise + BN.
        v.push(spec(&format!("{name}.0.weight"), &[ch, 1, 1, 3]));
        v.push(spec(&format!("{name}.1.weight"), &[ch, ch, 1, 1]));
        bn(&mut v, &format!("{name}.2"), ch);
    }
    // conv0_out: full (1,3) conv → BN → sigmoid.
    v.push(spec("erb_dec.conv0_out.0.weight", &[1, ch, 1, 3]));
    bn(&mut v, "erb_dec.conv0_out.1", 1);
    // --- mask expansion ---
    v.push(spec("mask.erb_inv_fb", &[cfg.n_erb, cfg.n_bins()]));
    // --- DF decoder (DfDecoder.__init__) ---
    // df_convp: groups = gcd(ch, order·2) conv (kt=5, kf=1) + pointwise + BN.
    let pg = gcd(ch, out_ch_df);
    v.push(spec(
        "df_dec.df_convp.1.weight",
        &[out_ch_df, ch / pg, 5, 1],
    ));
    v.push(spec(
        "df_dec.df_convp.2.weight",
        &[out_ch_df, out_ch_df, 1, 1],
    ));
    bn(&mut v, "df_dec.df_convp.3", out_ch_df);
    v.push(spec(
        "df_dec.df_gru.linear_in.0.weight",
        &[dg, emb / dg, dh / dg],
    ));
    gru(&mut v, "df_dec.df_gru.gru", cfg.df_num_layers, dh);
    v.push(spec("df_dec.df_skip.weight", &[lg, emb / lg, dh / lg]));
    v.push(spec(
        "df_dec.df_out.0.weight",
        &[lg, dh / lg, cfg.df_out_dim() / lg],
    ));
    v
}

/// Checkpoint tensors that are intentionally NOT converted (each one dead in
/// the DFN3 inference graph):
///
/// * `erb_fb` — registered buffer, never read by `DfNet.forward` (the ERB
///   features arrive precomputed from the libDF frontend; the expansion
///   uses `mask.erb_inv_fb`).
/// * `df_dec.df_fc_a.0.{weight,bias}` — the legacy DF-alpha head;
///   `DfDecoder.forward` (df/deepfilternet3.py L323-331) never calls it.
///
/// (`*.num_batches_tracked` BatchNorm training counters are dropped one step
/// earlier, in `tools/parity/dfn3_prepare_checkpoint.py`.)
pub fn denoise_skipped_checkpoint_tensors() -> &'static [&'static str] {
    &["erb_fb", "df_dec.df_fc_a.0.weight", "df_dec.df_fc_a.0.bias"]
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Deterministic synthetic tensor set for the full real topology
/// (SplitMix64-seeded small values). Not a trained network — round-trip /
/// shape tests only; numeric parity runs against the real checkpoint.
pub fn denoise_synthesized_tensors(
    cfg: &DeepFilterNetConfig,
    seed: u64,
) -> Vec<(TensorSpec, Vec<f32>)> {
    let mut rng = SplitMix64::new(seed);
    denoise_tensor_manifest(cfg)
        .into_iter()
        .map(|s| {
            let n: usize = s.shape.iter().product();
            let data: Vec<f32> = if s.name.ends_with("running_var") {
                // Keep BatchNorm variances positive.
                (0..n).map(|_| 0.5 + rng.next_unit_f32() * 0.5).collect()
            } else {
                (0..n).map(|_| (rng.next_unit_f32() - 0.5) * 0.2).collect()
            };
            (s, data)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// A bound DeepFilterNet3 denoiser: config + the full conv/GRU network +
/// the ERB frontend tables.
#[derive(Debug, Clone)]
pub struct DenoiseModel {
    cfg: DeepFilterNetConfig,
    widths: Vec<usize>,
    window: Vec<f32>,
    wnorm: f32,
    // Encoder.
    erb_conv0: ConvBlock,
    erb_conv1: ConvBlock,
    erb_conv2: ConvBlock,
    erb_conv3: ConvBlock,
    df_conv0: ConvBlock,
    df_conv1: ConvBlock,
    df_fc_emb: GroupedLinear,
    enc_emb_gru: SqueezedGru,
    lsnr_fc: Linear,
    // ERB decoder.
    dec_emb_gru: SqueezedGru,
    conv3p: ConvBlock,
    convt3: ConvBlock,
    conv2p: ConvBlock,
    convt2: ConvBlock,
    conv1p: ConvBlock,
    convt1: ConvBlock,
    conv0p: ConvBlock,
    conv0_out: ConvBlock,
    // DF decoder.
    df_convp: ConvBlock,
    df_gru: SqueezedGru,
    df_skip: GroupedLinear,
    df_out: GroupedLinear,
    /// `[n_erb, n_bins]` mask expansion matrix (checkpoint buffer).
    erb_inv_fb: Vec<f32>,
}

/// Per-stage diagnostic taps of one [`DenoiseModel::enhance_with_taps`] run
/// (layouts documented per field; `T` = number of STFT frames of the padded
/// input). Exists for the stage-parity harness — not a stable API.
#[derive(Debug, Clone)]
pub struct DenoiseTaps {
    /// STFT frame count `T` of the padded input.
    pub frames: usize,
    /// `[T, n_bins]` raw analysis spectrum, real parts (frontend stage 1).
    pub spec_re: Vec<f32>,
    /// `[T, n_bins]` raw analysis spectrum, imaginary parts.
    pub spec_im: Vec<f32>,
    /// `[T, n_erb]` normalized ERB features (pre-lookahead-shift).
    pub feat_erb: Vec<f32>,
    /// `[T, df_bins]` unit-normed complex features, real parts.
    pub feat_spec_re: Vec<f32>,
    /// `[T, df_bins]` unit-normed complex features, imaginary parts.
    pub feat_spec_im: Vec<f32>,
    /// `[conv_ch, T, n_erb]` encoder erb_conv0 output.
    pub e0: Vec<f32>,
    /// `[conv_ch, T, n_erb/2]` encoder erb_conv1 output.
    pub e1: Vec<f32>,
    /// `[conv_ch, T, n_erb/4]` encoder erb_conv2 output.
    pub e2: Vec<f32>,
    /// `[conv_ch, T, n_erb/4]` encoder erb_conv3 output.
    pub e3: Vec<f32>,
    /// `[conv_ch, T, df_bins]` encoder df_conv0 output.
    pub c0: Vec<f32>,
    /// `[T, emb_dim]` df branch embedding (df_fc_emb output).
    pub cemb: Vec<f32>,
    /// `[T, emb_dim]` combined embedding-GRU input.
    pub emb_in: Vec<f32>,
    /// `[T, emb_dim]` encoder embedding (emb_gru output).
    pub emb: Vec<f32>,
    /// `[T]` local SNR estimate (dB).
    pub lsnr: Vec<f32>,
    /// `[T, n_erb]` ERB mask (sigmoid gains).
    pub m: Vec<f32>,
    /// `[T, df_hidden]` df GRU output after the grouped-linear skip.
    pub df_gru_out: Vec<f32>,
    /// `[T, df_bins, df_order·2]` DF coefficients ((re, im) pairs, order-major
    /// within the last axis).
    pub coefs: Vec<f32>,
    /// `[T, n_bins]` final enhanced spectrum, real parts.
    pub spec_e_re: Vec<f32>,
    /// `[T, n_bins]` final enhanced spectrum, imaginary parts.
    pub spec_e_im: Vec<f32>,
}

/// Manifest-checked tensor consumption during [`DenoiseModel::from_tensors`].
struct TensorLoader<'a> {
    manifest: &'a [TensorSpec],
    tensors: BTreeMap<String, Vec<f32>>,
}

impl TensorLoader<'_> {
    fn take(&mut self, name: &str) -> Result<Vec<f32>> {
        let s = self
            .manifest
            .iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("internal: `{name}` missing from manifest"));
        let data = self
            .tensors
            .remove(name)
            .ok_or_else(|| VokraError::ModelLoad(format!("denoise: missing tensor `{name}`")))?;
        let want: usize = s.shape.iter().product();
        if data.len() != want {
            return Err(VokraError::ModelLoad(format!(
                "denoise: tensor `{name}` has {} elements, expected {want} (shape {:?})",
                data.len(),
                s.shape
            )));
        }
        Ok(data)
    }

    /// Eval-mode BatchNorm2d fold (torch batch_norm_cpu_inference):
    /// `invstd = 1/√(σ² + eps); scale = invstd·γ; shift = β − μ·scale`.
    fn bn(&mut self, prefix: &str, n: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let gamma = self.take(&format!("{prefix}.weight"))?;
        let beta = self.take(&format!("{prefix}.bias"))?;
        let mean = self.take(&format!("{prefix}.running_mean"))?;
        let var = self.take(&format!("{prefix}.running_var"))?;
        const EPS: f32 = 1e-5;
        let mut scale = vec![0.0f32; n];
        let mut shift = vec![0.0f32; n];
        for i in 0..n {
            let invstd = 1.0 / (var[i] + EPS).sqrt();
            scale[i] = invstd * gamma[i];
            shift[i] = beta[i] - mean[i] * scale[i];
        }
        Ok((scale, shift))
    }

    /// Depthwise (1, 3) conv + pointwise + BN + ReLU (`Conv2dNormAct`
    /// separable with `conv_kernel = (1, 3)`).
    fn sep13(&mut self, name: &str, ch: usize, fstride: usize) -> Result<ConvBlock> {
        let dw = self.take(&format!("{name}.0.weight"))?;
        let pw = self.take(&format!("{name}.1.weight"))?;
        let (s, b) = self.bn(&format!("{name}.2"), ch)?;
        Ok(ConvBlock {
            conv: ConvUnit::Std(Conv2d {
                w: dw,
                in_ch: ch,
                out_ch: ch,
                groups: ch,
                kt: 1,
                kf: 3,
                fstride,
                fpad: 1,
            }),
            pointwise: Some(pw),
            bn_scale: s,
            bn_shift: b,
            act: Act::Relu,
        })
    }

    /// Depthwise 1×1 pathway conv + BN + ReLU (`Conv2dNormAct` with
    /// `kernel_size = 1`: groups = gcd survives, the pointwise tail does
    /// not — df/modules.py L49-67).
    fn pathway(&mut self, name: &str, ch: usize) -> Result<ConvBlock> {
        let w = self.take(&format!("{name}.0.weight"))?;
        let (s, b) = self.bn(&format!("{name}.1"), ch)?;
        Ok(ConvBlock {
            conv: ConvUnit::Std(Conv2d {
                w,
                in_ch: ch,
                out_ch: ch,
                groups: ch,
                kt: 1,
                kf: 1,
                fstride: 1,
                fpad: 0,
            }),
            pointwise: None,
            bn_scale: s,
            bn_shift: b,
            act: Act::Relu,
        })
    }

    /// Depthwise transposed (1, 3) fstride-2 conv + pointwise + BN + ReLU
    /// (`ConvTranspose2dNormAct`).
    fn tconv(&mut self, name: &str, ch: usize) -> Result<ConvBlock> {
        let dw = self.take(&format!("{name}.0.weight"))?;
        let pw = self.take(&format!("{name}.1.weight"))?;
        let (s, b) = self.bn(&format!("{name}.2"), ch)?;
        Ok(ConvBlock {
            conv: ConvUnit::Transposed(ConvT2dF {
                w: dw,
                in_ch: ch,
                out_ch: ch,
                groups: ch,
                kf: 3,
                fstride: 2,
                fpad: 1,
                out_pad: 1,
            }),
            pointwise: Some(pw),
            bn_scale: s,
            bn_shift: b,
            act: Act::Relu,
        })
    }

    fn glin(
        &mut self,
        name: &str,
        groups: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<GroupedLinear> {
        Ok(GroupedLinear {
            w: self.take(name)?,
            groups,
            in_dim,
            out_dim,
        })
    }

    fn gru_stack(&mut self, prefix: &str, layers: usize, hidden: usize) -> Result<Vec<GruLayer>> {
        (0..layers)
            .map(|l| {
                Ok(GruLayer {
                    w_ih: self.take(&format!("{prefix}.weight_ih_l{l}"))?,
                    w_hh: self.take(&format!("{prefix}.weight_hh_l{l}"))?,
                    b_ih: self.take(&format!("{prefix}.bias_ih_l{l}"))?,
                    b_hh: self.take(&format!("{prefix}.bias_hh_l{l}"))?,
                    input: hidden,
                    hidden,
                })
            })
            .collect()
    }
}

impl DenoiseModel {
    /// Binds a model from a config + a full named tensor map (upstream
    /// state-dict names). Consumes the map; every manifest tensor must be
    /// present with the exact element count and no unknown tensor may
    /// remain (hard error — never a silent skip, FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] for a bad config;
    /// [`VokraError::ModelLoad`] for missing / mis-sized / unknown tensors.
    pub fn from_tensors(
        cfg: DeepFilterNetConfig,
        tensors: BTreeMap<String, Vec<f32>>,
    ) -> Result<Self> {
        cfg.validate()?;
        let manifest = denoise_tensor_manifest(&cfg);
        let mut ld = TensorLoader {
            manifest: &manifest,
            tensors,
        };
        let ch = cfg.conv_ch;

        // --- encoder ---
        let erb_conv0 = {
            let w = ld.take("enc.erb_conv0.1.weight")?;
            let (s, b) = ld.bn("enc.erb_conv0.2", ch)?;
            ConvBlock {
                conv: ConvUnit::Std(Conv2d {
                    w,
                    in_ch: 1,
                    out_ch: ch,
                    groups: 1,
                    kt: 3,
                    kf: 3,
                    fstride: 1,
                    fpad: 1,
                }),
                pointwise: None,
                bn_scale: s,
                bn_shift: b,
                act: Act::Relu,
            }
        };
        let erb_conv1 = ld.sep13("enc.erb_conv1", ch, 2)?;
        let erb_conv2 = ld.sep13("enc.erb_conv2", ch, 2)?;
        let erb_conv3 = ld.sep13("enc.erb_conv3", ch, 1)?;
        let df_conv0 = {
            let w = ld.take("enc.df_conv0.1.weight")?;
            let pw = ld.take("enc.df_conv0.2.weight")?;
            let (s, b) = ld.bn("enc.df_conv0.3", ch)?;
            ConvBlock {
                conv: ConvUnit::Std(Conv2d {
                    w,
                    in_ch: 2,
                    out_ch: ch,
                    groups: 2,
                    kt: 3,
                    kf: 3,
                    fstride: 1,
                    fpad: 1,
                }),
                pointwise: Some(pw),
                bn_scale: s,
                bn_shift: b,
                act: Act::Relu,
            }
        };
        let df_conv1 = ld.sep13("enc.df_conv1", ch, 2)?;
        let emb = cfg.emb_dim();
        let df_fc_emb = ld.glin(
            "enc.df_fc_emb.0.weight",
            cfg.enc_linear_groups,
            cfg.cemb_dim(),
            emb,
        )?;
        let enc_emb_gru = SqueezedGru {
            linear_in: ld.glin(
                "enc.emb_gru.linear_in.0.weight",
                cfg.linear_groups,
                emb,
                cfg.emb_hidden,
            )?,
            layers: ld.gru_stack("enc.emb_gru.gru", 1, cfg.emb_hidden)?,
            linear_out: Some(ld.glin(
                "enc.emb_gru.linear_out.0.weight",
                cfg.linear_groups,
                cfg.emb_hidden,
                emb,
            )?),
        };
        let lsnr_fc = Linear {
            w: ld.take("enc.lsnr_fc.0.weight")?,
            b: ld.take("enc.lsnr_fc.0.bias")?,
            in_dim: emb,
        };

        // --- ERB decoder ---
        let dec_emb_gru = SqueezedGru {
            linear_in: ld.glin(
                "erb_dec.emb_gru.linear_in.0.weight",
                cfg.linear_groups,
                emb,
                cfg.emb_hidden,
            )?,
            layers: ld.gru_stack(
                "erb_dec.emb_gru.gru",
                cfg.emb_num_layers - 1,
                cfg.emb_hidden,
            )?,
            linear_out: Some(ld.glin(
                "erb_dec.emb_gru.linear_out.0.weight",
                cfg.linear_groups,
                cfg.emb_hidden,
                emb,
            )?),
        };
        let conv3p = ld.pathway("erb_dec.conv3p", ch)?;
        let conv2p = ld.pathway("erb_dec.conv2p", ch)?;
        let conv1p = ld.pathway("erb_dec.conv1p", ch)?;
        let conv0p = ld.pathway("erb_dec.conv0p", ch)?;
        let convt3 = ld.sep13("erb_dec.convt3", ch, 1)?;
        let convt2 = ld.tconv("erb_dec.convt2", ch)?;
        let convt1 = ld.tconv("erb_dec.convt1", ch)?;
        let conv0_out = {
            let w = ld.take("erb_dec.conv0_out.0.weight")?;
            let (s, b) = ld.bn("erb_dec.conv0_out.1", 1)?;
            ConvBlock {
                conv: ConvUnit::Std(Conv2d {
                    w,
                    in_ch: ch,
                    out_ch: 1,
                    groups: 1,
                    kt: 1,
                    kf: 3,
                    fstride: 1,
                    fpad: 1,
                }),
                pointwise: None,
                bn_scale: s,
                bn_shift: b,
                act: Act::Sigmoid,
            }
        };
        let erb_inv_fb = ld.take("mask.erb_inv_fb")?;

        // --- DF decoder ---
        let out_ch_df = cfg.df_order * 2;
        let df_convp = {
            let dw = ld.take("df_dec.df_convp.1.weight")?;
            let pw = ld.take("df_dec.df_convp.2.weight")?;
            let (s, b) = ld.bn("df_dec.df_convp.3", out_ch_df)?;
            ConvBlock {
                conv: ConvUnit::Std(Conv2d {
                    w: dw,
                    in_ch: ch,
                    out_ch: out_ch_df,
                    groups: gcd(ch, out_ch_df),
                    kt: 5,
                    kf: 1,
                    fstride: 1,
                    fpad: 0,
                }),
                pointwise: Some(pw),
                bn_scale: s,
                bn_shift: b,
                act: Act::Relu,
            }
        };
        let df_gru = SqueezedGru {
            linear_in: ld.glin(
                "df_dec.df_gru.linear_in.0.weight",
                cfg.df_gru_linear_groups,
                emb,
                cfg.df_hidden,
            )?,
            layers: ld.gru_stack("df_dec.df_gru.gru", cfg.df_num_layers, cfg.df_hidden)?,
            linear_out: None,
        };
        let df_skip = ld.glin(
            "df_dec.df_skip.weight",
            cfg.linear_groups,
            emb,
            cfg.df_hidden,
        )?;
        let df_out = ld.glin(
            "df_dec.df_out.0.weight",
            cfg.linear_groups,
            cfg.df_hidden,
            cfg.df_out_dim(),
        )?;

        if let Some((name, _)) = ld.tensors.into_iter().next() {
            return Err(VokraError::ModelLoad(format!(
                "denoise: unknown tensor `{name}` (checkpoint layout drift?)"
            )));
        }

        Ok(Self {
            widths: cfg.erb_widths()?,
            window: vorbis_window(cfg.n_fft),
            // libDF lib.rs L133: 1 / (fft² / (2·hop)), computed in f32.
            wnorm: 1.0 / (cfg.n_fft.pow(2) as f32 / (2 * cfg.hop) as f32),
            cfg,
            erb_conv0,
            erb_conv1,
            erb_conv2,
            erb_conv3,
            df_conv0,
            df_conv1,
            df_fc_emb,
            enc_emb_gru,
            lsnr_fc,
            dec_emb_gru,
            conv3p,
            convt3,
            conv2p,
            convt2,
            conv1p,
            convt1,
            conv0p,
            conv0_out,
            df_convp,
            df_gru,
            df_skip,
            df_out,
            erb_inv_fb,
        })
    }

    /// The model config.
    pub fn config(&self) -> &DeepFilterNetConfig {
        &self.cfg
    }

    /// Enhances `noisy`, returning a signal of the same length (the
    /// `df/enhance.py enhance(pad=True)` contract: right-pad by `n_fft`,
    /// run the pipeline, trim the `n_fft − hop` STFT/ISTFT delay).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] for an empty or non-finite input.
    pub fn enhance(&self, noisy: &[f32]) -> Result<Vec<f32>> {
        Ok(self.enhance_impl(noisy, false)?.0)
    }

    /// [`DenoiseModel::enhance`] + per-stage diagnostic taps (parity
    /// harness surface).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] for an empty or non-finite input.
    pub fn enhance_with_taps(&self, noisy: &[f32]) -> Result<(Vec<f32>, DenoiseTaps)> {
        let (out, taps) = self.enhance_impl(noisy, true)?;
        Ok((out, taps.expect("taps requested")))
    }

    fn enhance_impl(
        &self,
        noisy: &[f32],
        want_taps: bool,
    ) -> Result<(Vec<f32>, Option<DenoiseTaps>)> {
        if noisy.is_empty() {
            return Err(VokraError::InvalidArgument("denoise: empty input".into()));
        }
        if noisy.iter().any(|s| !s.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "denoise: input has a non-finite sample".into(),
            ));
        }
        let cfg = &self.cfg;
        // enhance() pad=True: right-pad n_fft zeros (df/enhance.py L229-232).
        let mut padded = Vec::with_capacity(noisy.len() + cfg.n_fft);
        padded.extend_from_slice(noisy);
        padded.resize(noisy.len() + cfg.n_fft, 0.0);

        let spec = self.analysis(&padded);
        let (spec_e, taps) = self.forward_spec(&spec, want_taps);
        let full = self.synthesis(&spec_e);
        // Delay trim (df/enhance.py L241-248): d = n_fft − hop.
        let d = cfg.n_fft - cfg.hop;
        let out = full[d..d + noisy.len()].to_vec();
        Ok((out, taps))
    }

    // -- frontend ----------------------------------------------------------

    /// libDF streaming analysis over `hop`-sample chunks (`frame_analysis`,
    /// lib.rs L356-394): frame `i` windows `[(i−1)·hop, (i+1)·hop)` (zeros
    /// before the signal start; the final partial chunk is zero-padded, as
    /// in transforms.rs `stft` L177-183).
    fn analysis(&self, x: &[f32]) -> Spectrogram {
        let cfg = &self.cfg;
        let (n_fft, hop) = (cfg.n_fft, cfg.hop);
        let n_bins = cfg.n_bins();
        let frames = x.len().div_ceil(hop);
        let plan = RealFftPlan::new(n_fft);
        let mut mem = vec![0.0f32; n_fft - hop];
        let mut buf = vec![0.0f32; n_fft];
        let mut re = vec![0.0f32; frames * n_bins];
        let mut im = vec![0.0f32; frames * n_bins];
        let chunk_at = |i: usize| -> f32 { if i < x.len() { x[i] } else { 0.0 } };
        for t in 0..frames {
            let start = t * hop;
            for (b, (&m, &w)) in buf[..n_fft - hop]
                .iter_mut()
                .zip(mem.iter().zip(&self.window[..n_fft - hop]))
            {
                *b = m * w;
            }
            for i in 0..hop {
                buf[n_fft - hop + i] = chunk_at(start + i) * self.window[n_fft - hop + i];
            }
            // Update analysis memory with the RAW chunk (lib.rs L377-384).
            let split = mem.len() - hop;
            if split > 0 {
                mem.rotate_left(hop);
            }
            for i in 0..hop {
                mem[split + i] = chunk_at(start + i);
            }
            let spec = plan.forward(&buf);
            for (k, c) in spec.iter().enumerate() {
                // wnorm applied in analysis only (lib.rs L389-393).
                re[t * n_bins + k] = c.re * self.wnorm;
                im[t * n_bins + k] = c.im * self.wnorm;
            }
        }
        Spectrogram {
            frames,
            bins: n_bins,
            re,
            im,
        }
    }

    /// libDF streaming synthesis (`frame_synthesis`, lib.rs L396-427):
    /// unnormalized inverse RFFT × window, overlap-add through the
    /// synthesis memory. (The upstream realfft `ComplexToReal` ignores the
    /// imaginary parts of the DC / Nyquist bins — verified empirically —
    /// exactly like [`RealFftPlan::inverse`]'s hermitian reconstruction.)
    fn synthesis(&self, spec: &Spectrogram) -> Vec<f32> {
        let cfg = &self.cfg;
        let (n_fft, hop) = (cfg.n_fft, cfg.hop);
        let plan = RealFftPlan::new(n_fft);
        let mut mem = vec![0.0f32; n_fft - hop];
        let mut out = vec![0.0f32; spec.frames * hop];
        let mut row = vec![Complex32::ZERO; spec.bins];
        let scale = n_fft as f32; // undo RealFftPlan::inverse's 1/n (realfft c2r is unnormalized)
        for t in 0..spec.frames {
            for (k, c) in row.iter_mut().enumerate() {
                *c = Complex32 {
                    re: spec.re[t * spec.bins + k],
                    im: spec.im[t * spec.bins + k],
                };
            }
            let mut x = plan.inverse(&row);
            for (v, &w) in x.iter_mut().zip(&self.window) {
                *v *= scale * w;
            }
            let chunk = &mut out[t * hop..(t + 1) * hop];
            for ((o, &xv), &m) in chunk.iter_mut().zip(&x[..hop]).zip(mem.iter()) {
                *o = xv + m;
            }
            let split = mem.len() - hop;
            if split > 0 {
                mem.rotate_left(hop);
            }
            let x_second = &x[hop..];
            for (m, &xv) in mem[..split].iter_mut().zip(&x_second[..split]) {
                *m += xv; // overlap-add tail
            }
            for (m, &xv) in mem[split..].iter_mut().zip(&x_second[split..]) {
                *m = xv; // override the freshly shifted region
            }
        }
        out
    }

    /// ERB feature: per-band mean power → `10·log10(· + 1e-10)` →
    /// exponential mean-norm `/40` (libDF `feat_erb` L206-212 +
    /// `band_mean_norm_erb` L244-251, state init linspace(−60, −90)).
    fn feat_erb(&self, spec: &Spectrogram) -> Vec<f32> {
        let cfg = &self.cfg;
        let n_erb = cfg.n_erb;
        let alpha = cfg.norm_alpha;
        let mut state = linspace(-60.0, -90.0, n_erb);
        let mut out = vec![0.0f32; spec.frames * n_erb];
        for t in 0..spec.frames {
            let row = &mut out[t * n_erb..(t + 1) * n_erb];
            let mut bin = 0usize;
            for (b, &width) in self.widths.iter().enumerate() {
                let k = 1.0 / width as f32;
                let mut acc = 0.0f32;
                for j in bin..bin + width {
                    let (re, im) = (spec.re[t * spec.bins + j], spec.im[t * spec.bins + j]);
                    // compute_band_corr (lib.rs L280-295): per-bin ×k accumulate.
                    acc += (re * re + im * im) * k;
                }
                bin += width;
                row[b] = (acc + 1e-10).log10() * 10.0;
            }
            for (x, s) in row.iter_mut().zip(state.iter_mut()) {
                *s = *x * (1.0 - alpha) + *s * alpha;
                *x -= *s;
                *x /= 40.0;
            }
        }
        out
    }

    /// Unit-normed complex features over the low `df_bins` bins (libDF
    /// `feat_cplx` L214-217 + `band_unit_norm` L253-259, state init
    /// linspace(0.001, 0.0001)).
    fn feat_spec(&self, spec: &Spectrogram) -> (Vec<f32>, Vec<f32>) {
        let cfg = &self.cfg;
        let nb = cfg.df_bins;
        let alpha = cfg.norm_alpha;
        let mut state = linspace(0.001, 0.0001, nb);
        let mut re = vec![0.0f32; spec.frames * nb];
        let mut im = vec![0.0f32; spec.frames * nb];
        for t in 0..spec.frames {
            for (f, s) in state.iter_mut().enumerate() {
                let (r, i) = (spec.re[t * spec.bins + f], spec.im[t * spec.bins + f]);
                // Complex32::norm() is hypot in upstream num_complex.
                let mag = r.hypot(i);
                *s = mag * (1.0 - alpha) + *s * alpha;
                let d = s.sqrt();
                re[t * nb + f] = r / d;
                im[t * nb + f] = i / d;
            }
        }
        (re, im)
    }

    // -- network -----------------------------------------------------------

    /// Full DfNet forward over an analysis spectrogram (df/deepfilternet3.py
    /// `DfNet.forward` L389-456).
    fn forward_spec(
        &self,
        spec: &Spectrogram,
        want_taps: bool,
    ) -> (Spectrogram, Option<DenoiseTaps>) {
        let cfg = &self.cfg;
        let t_len = spec.frames;
        let n_erb = cfg.n_erb;
        let nb_df = cfg.df_bins;
        let emb_dim = cfg.emb_dim();
        let ch = cfg.conv_ch;

        let feat_erb = self.feat_erb(spec);
        let (fs_re, fs_im) = self.feat_spec(spec);

        // pad_feat: shift features `conv_lookahead` frames into the future
        // (crop the first rows, zero-pad the tail — deepfilternet3.py
        // L357-361 / L409-410).
        let la = cfg.conv_lookahead;
        let shift2 = |x: &[f32], width: usize| -> Vec<f32> {
            let mut y = vec![0.0f32; t_len * width];
            if t_len > la {
                y[..(t_len - la) * width].copy_from_slice(&x[la * width..]);
            }
            y
        };
        let fe = shift2(&feat_erb, n_erb);
        let fsr = shift2(&fs_re, nb_df);
        let fsi = shift2(&fs_im, nb_df);

        // Encoder.
        let mut x_erb = Act3::zeros(1, t_len, n_erb);
        x_erb.data.copy_from_slice(&fe);
        let mut x_spec = Act3::zeros(2, t_len, nb_df);
        x_spec.data[..t_len * nb_df].copy_from_slice(&fsr);
        x_spec.data[t_len * nb_df..].copy_from_slice(&fsi);
        let e0 = self.erb_conv0.forward(&x_erb);
        let e1 = self.erb_conv1.forward(&e0);
        let e2 = self.erb_conv2.forward(&e1);
        let e3 = self.erb_conv3.forward(&e2);
        let c0 = self.df_conv0.forward(&x_spec);
        let c1 = self.df_conv1.forward(&c0);
        // cemb = df_fc_emb(flatten(c1, permute(0,2,3,1))): x[t, f·ch + c].
        let cemb_in = flatten_tfc(&c1);
        let mut cemb = self.df_fc_emb.forward_seq(&cemb_in, t_len);
        for v in &mut cemb {
            *v = v.max(0.0);
        }
        // emb_in = flatten(e3) + cemb (Add combine).
        let mut emb_in = flatten_tfc(&e3);
        for (a, &b) in emb_in.iter_mut().zip(&cemb) {
            *a += b;
        }
        let emb = self.enc_emb_gru.forward(&emb_in, t_len);
        let mut lsnr = vec![0.0f32; t_len];
        let mut one = [0.0f32; 1];
        for (t, l) in lsnr.iter_mut().enumerate() {
            self.lsnr_fc
                .forward_frame(&emb[t * emb_dim..(t + 1) * emb_dim], &mut one);
            *l = sigmoid(one[0]) * (cfg.lsnr_max - cfg.lsnr_min) + cfg.lsnr_min;
        }

        // ERB decoder (ErbDecoder.forward L245-254).
        let emb2 = self.dec_emb_gru.forward(&emb, t_len);
        let f8 = n_erb / 4;
        let mut emb2r = Act3::zeros(ch, t_len, f8);
        for t in 0..t_len {
            for f in 0..f8 {
                for c in 0..ch {
                    let i = emb2r.idx(c, t, f);
                    emb2r.data[i] = emb2[t * emb_dim + f * ch + c];
                }
            }
        }
        let mut d3 = self.conv3p.forward(&e3);
        add_assign(&mut d3.data, &emb2r.data);
        let d3 = self.convt3.forward(&d3);
        let mut d2 = self.conv2p.forward(&e2);
        add_assign(&mut d2.data, &d3.data);
        let d2 = self.convt2.forward(&d2);
        let mut d1 = self.conv1p.forward(&e1);
        add_assign(&mut d1.data, &d2.data);
        let d1 = self.convt1.forward(&d1);
        let mut d0 = self.conv0p.forward(&e0);
        add_assign(&mut d0.data, &d1.data);
        let m = self.conv0_out.forward(&d0); // [1, T, n_erb], sigmoid

        // DF decoder (DfDecoder.forward L323-331).
        let mut c = self.df_gru.forward(&emb, t_len);
        let dh = cfg.df_hidden;
        {
            let mut skip = vec![0.0f32; dh];
            for t in 0..t_len {
                self.df_skip
                    .forward_frame(&emb[t * emb_dim..(t + 1) * emb_dim], &mut skip);
                for (cv, &sv) in c[t * dh..(t + 1) * dh].iter_mut().zip(&skip) {
                    *cv += sv;
                }
            }
        }
        let c0p = self.df_convp.forward(&c0); // [order·2, T, nb_df]
        let od2 = cfg.df_order * 2;
        let out_dim = cfg.df_out_dim();
        let mut coefs = vec![0.0f32; t_len * out_dim];
        {
            let mut frame = vec![0.0f32; out_dim];
            for t in 0..t_len {
                self.df_out
                    .forward_frame(&c[t * dh..(t + 1) * dh], &mut frame);
                let row = &mut coefs[t * out_dim..(t + 1) * out_dim];
                for (f, chunk) in row.chunks_exact_mut(od2).enumerate() {
                    for (o, v) in chunk.iter_mut().enumerate() {
                        // tanh(df_out) + df_convp pathway (channels-last).
                        *v = frame[f * od2 + o].tanh() + c0p.at(o, t, f);
                    }
                }
            }
        }

        // Output stage: ERB mask on all bins, deep filter overwrite of the
        // low `df_bins` (DfNet.forward L426-443 — DF runs on the RAW spec).
        let n_bins = spec.bins;
        let mut out = Spectrogram {
            frames: t_len,
            bins: n_bins,
            re: vec![0.0f32; t_len * n_bins],
            im: vec![0.0f32; t_len * n_bins],
        };
        // spec_m = spec · (m @ erb_inv_fb).
        let mut gains = vec![0.0f32; n_bins];
        for t in 0..t_len {
            gains.fill(0.0);
            for b in 0..n_erb {
                let mv = m.data[t * n_erb + b];
                let fb_row = &self.erb_inv_fb[b * n_bins..(b + 1) * n_bins];
                for (g, &fv) in gains.iter_mut().zip(fb_row) {
                    *g += mv * fv;
                }
            }
            let row = t * n_bins..(t + 1) * n_bins;
            for (((or, oi), (&sr, &si)), &g) in out.re[row.clone()]
                .iter_mut()
                .zip(&mut out.im[row.clone()])
                .zip(spec.re[row.clone()].iter().zip(&spec.im[row]))
                .zip(&gains)
            {
                *or = sr * g;
                *oi = si * g;
            }
        }
        // Deep filtering: out[t, f<nb_df] = Σ_o spec[t − (O−1−la) + o, f] ·
        // coef[t, f, o] (complex; zero outside [0, T)) — df/multiframe.py
        // `DF.forward` + `spec_unfold` with lookahead.
        let back = cfg.df_order - 1 - cfg.df_lookahead;
        for t in 0..t_len {
            for f in 0..nb_df {
                let mut acc_re = 0.0f32;
                let mut acc_im = 0.0f32;
                for o in 0..cfg.df_order {
                    let Some(src_t) = (t + o).checked_sub(back) else {
                        continue;
                    };
                    if src_t >= t_len {
                        continue;
                    }
                    let (sr, si) = (spec.re[src_t * n_bins + f], spec.im[src_t * n_bins + f]);
                    let ci = t * out_dim + f * od2 + o * 2;
                    let (cr, cim) = (coefs[ci], coefs[ci + 1]);
                    acc_re += sr * cr - si * cim;
                    acc_im += sr * cim + si * cr;
                }
                out.re[t * n_bins + f] = acc_re;
                out.im[t * n_bins + f] = acc_im;
            }
        }

        let taps = want_taps.then(|| DenoiseTaps {
            frames: t_len,
            spec_re: spec.re.clone(),
            spec_im: spec.im.clone(),
            feat_erb,
            feat_spec_re: fs_re,
            feat_spec_im: fs_im,
            e0: e0.data.clone(),
            e1: e1.data.clone(),
            e2: e2.data.clone(),
            e3: e3.data.clone(),
            c0: c0.data.clone(),
            cemb,
            emb_in,
            emb,
            lsnr: lsnr.clone(),
            m: m.data.clone(),
            df_gru_out: c,
            coefs,
            spec_e_re: out.re.clone(),
            spec_e_im: out.im.clone(),
        });
        (out, taps)
    }
}

/// Flattens `[C, T, F]` to `[T, F·C]` with `x[t, f·C + c]` ordering (the
/// upstream `permute(0, 2, 3, 1).flatten(2)`).
fn flatten_tfc(x: &Act3) -> Vec<f32> {
    let mut y = vec![0.0f32; x.t * x.f * x.ch];
    for c in 0..x.ch {
        for t in 0..x.t {
            for f in 0..x.f {
                y[t * (x.f * x.ch) + f * x.ch + c] = x.at(c, t, f);
            }
        }
    }
    y
}

fn add_assign(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for (av, &bv) in a.iter_mut().zip(b) {
        *av += bv;
    }
}

/// One-shot [`DenoiseModel::enhance`] convenience.
///
/// # Errors
///
/// Propagates [`DenoiseModel::enhance`] errors.
pub fn denoise(noisy: &[f32], model: &DenoiseModel) -> Result<Vec<f32>> {
    model.enhance(noisy)
}

// ---- GGUF binding (M4-20 T12/T17): `vokra.denoise.*` ----------------------
//
// Config keys are u32 / f32 under the `vokra.denoise.*` namespace; every
// neural tensor keeps its upstream state-dict name verbatim (F32, dims
// stored innermost-first per the GGUF convention). The converter
// (`vokra-cli convert --model denoise`) writes this layout from the
// prepared safetensors (tools/parity/dfn3_prepare_checkpoint.py).

const KEY_N_FFT: &str = "vokra.denoise.n_fft";
const KEY_HOP: &str = "vokra.denoise.hop";
const KEY_SAMPLE_RATE: &str = "vokra.denoise.sample_rate";
const KEY_N_ERB: &str = "vokra.denoise.n_erb";
const KEY_DF_BINS: &str = "vokra.denoise.df_bins";
const KEY_DF_ORDER: &str = "vokra.denoise.df_order";
const KEY_MIN_NB_ERB_FREQS: &str = "vokra.denoise.min_nb_erb_freqs";
const KEY_CONV_LOOKAHEAD: &str = "vokra.denoise.conv_lookahead";
const KEY_DF_LOOKAHEAD: &str = "vokra.denoise.df_lookahead";
const KEY_CONV_CH: &str = "vokra.denoise.conv_ch";
const KEY_EMB_HIDDEN: &str = "vokra.denoise.emb_hidden_dim";
const KEY_DF_HIDDEN: &str = "vokra.denoise.df_hidden_dim";
const KEY_ENC_LINEAR_GROUPS: &str = "vokra.denoise.enc_linear_groups";
const KEY_LINEAR_GROUPS: &str = "vokra.denoise.linear_groups";
const KEY_DF_GRU_LINEAR_GROUPS: &str = "vokra.denoise.df_gru_linear_groups";
const KEY_EMB_NUM_LAYERS: &str = "vokra.denoise.emb_num_layers";
const KEY_DF_NUM_LAYERS: &str = "vokra.denoise.df_num_layers";
const KEY_LSNR_MIN: &str = "vokra.denoise.lsnr_min";
const KEY_LSNR_MAX: &str = "vokra.denoise.lsnr_max";
const KEY_NORM_ALPHA: &str = "vokra.denoise.norm_alpha";

impl DeepFilterNetConfig {
    /// Reads the `vokra.denoise.*` config keys from a parsed GGUF.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] for a missing / mis-typed key;
    /// [`VokraError::InvalidArgument`] for an inconsistent config.
    pub fn from_gguf(gguf: &vokra_core::gguf::GgufFile) -> Result<Self> {
        let u = |key: &str| -> Result<usize> {
            gguf.get(key)
                .and_then(|v| v.as_u64())
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| VokraError::ModelLoad(format!("denoise gguf: missing/bad `{key}`")))
        };
        let f = |key: &str| -> Result<f32> {
            gguf.get(key)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .ok_or_else(|| VokraError::ModelLoad(format!("denoise gguf: missing/bad `{key}`")))
        };
        let cfg = Self {
            n_fft: u(KEY_N_FFT)?,
            hop: u(KEY_HOP)?,
            sample_rate: u(KEY_SAMPLE_RATE)? as u32,
            n_erb: u(KEY_N_ERB)?,
            df_bins: u(KEY_DF_BINS)?,
            df_order: u(KEY_DF_ORDER)?,
            min_nb_erb_freqs: u(KEY_MIN_NB_ERB_FREQS)?,
            conv_lookahead: u(KEY_CONV_LOOKAHEAD)?,
            df_lookahead: u(KEY_DF_LOOKAHEAD)?,
            conv_ch: u(KEY_CONV_CH)?,
            emb_hidden: u(KEY_EMB_HIDDEN)?,
            df_hidden: u(KEY_DF_HIDDEN)?,
            enc_linear_groups: u(KEY_ENC_LINEAR_GROUPS)?,
            linear_groups: u(KEY_LINEAR_GROUPS)?,
            df_gru_linear_groups: u(KEY_DF_GRU_LINEAR_GROUPS)?,
            emb_num_layers: u(KEY_EMB_NUM_LAYERS)?,
            df_num_layers: u(KEY_DF_NUM_LAYERS)?,
            lsnr_min: f(KEY_LSNR_MIN)?,
            lsnr_max: f(KEY_LSNR_MAX)?,
            norm_alpha: f(KEY_NORM_ALPHA)?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Writes the `vokra.denoise.*` config keys (converter side).
    pub fn write_gguf_metadata(&self, b: &mut vokra_core::gguf::GgufBuilder) {
        b.add_string("vokra.model.arch", "denoise");
        b.add_u32(KEY_N_FFT, self.n_fft as u32);
        b.add_u32(KEY_HOP, self.hop as u32);
        b.add_u32(KEY_SAMPLE_RATE, self.sample_rate);
        b.add_u32(KEY_N_ERB, self.n_erb as u32);
        b.add_u32(KEY_DF_BINS, self.df_bins as u32);
        b.add_u32(KEY_DF_ORDER, self.df_order as u32);
        b.add_u32(KEY_MIN_NB_ERB_FREQS, self.min_nb_erb_freqs as u32);
        b.add_u32(KEY_CONV_LOOKAHEAD, self.conv_lookahead as u32);
        b.add_u32(KEY_DF_LOOKAHEAD, self.df_lookahead as u32);
        b.add_u32(KEY_CONV_CH, self.conv_ch as u32);
        b.add_u32(KEY_EMB_HIDDEN, self.emb_hidden as u32);
        b.add_u32(KEY_DF_HIDDEN, self.df_hidden as u32);
        b.add_u32(KEY_ENC_LINEAR_GROUPS, self.enc_linear_groups as u32);
        b.add_u32(KEY_LINEAR_GROUPS, self.linear_groups as u32);
        b.add_u32(KEY_DF_GRU_LINEAR_GROUPS, self.df_gru_linear_groups as u32);
        b.add_u32(KEY_EMB_NUM_LAYERS, self.emb_num_layers as u32);
        b.add_u32(KEY_DF_NUM_LAYERS, self.df_num_layers as u32);
        b.add_f32(KEY_LSNR_MIN, self.lsnr_min);
        b.add_f32(KEY_LSNR_MAX, self.lsnr_max);
        b.add_f32(KEY_NORM_ALPHA, self.norm_alpha);
    }
}

impl DenoiseModel {
    /// Binds a denoiser from a parsed GGUF (`vokra.denoise.*` config +
    /// upstream-named tensors). Every manifest tensor must be present with
    /// the exact size; unknown tensors are a hard error.
    ///
    /// # Errors
    ///
    /// Propagates [`DeepFilterNetConfig::from_gguf`] /
    /// [`DenoiseModel::from_tensors`] errors.
    pub fn from_gguf(gguf: &vokra_core::gguf::GgufFile) -> Result<Self> {
        let cfg = DeepFilterNetConfig::from_gguf(gguf)?;
        let mut tensors = BTreeMap::new();
        for info in gguf.tensors() {
            tensors.insert(info.name.clone(), gguf.tensor_f32(&info.name)?);
        }
        Self::from_tensors(cfg, tensors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small real-topology config for CPU-cheap tests (all upstream
    /// divisibility constraints hold: n_erb % 8, group divisions, 2·hop ≤
    /// n_fft).
    fn small_cfg() -> DeepFilterNetConfig {
        DeepFilterNetConfig {
            n_fft: 64,
            hop: 32,
            sample_rate: 16000,
            n_erb: 8,
            df_bins: 12,
            df_order: 3,
            min_nb_erb_freqs: 1,
            conv_lookahead: 1,
            df_lookahead: 1,
            conv_ch: 8,
            emb_hidden: 16,
            df_hidden: 16,
            enc_linear_groups: 4,
            linear_groups: 4,
            df_gru_linear_groups: 2,
            emb_num_layers: 3,
            df_num_layers: 2,
            lsnr_min: -15.0,
            lsnr_max: 35.0,
            norm_alpha: 0.99,
        }
    }

    fn small_model(seed: u64) -> DenoiseModel {
        let cfg = small_cfg();
        let tensors: BTreeMap<String, Vec<f32>> = denoise_synthesized_tensors(&cfg, seed)
            .into_iter()
            .map(|(s, d)| (s.name, d))
            .collect();
        DenoiseModel::from_tensors(cfg, tensors).unwrap()
    }

    #[test]
    fn deep_filter_net3_defaults_construct() {
        // The published DFN3 config + synthesized tensors binds (full real
        // topology at the published dims; construct-only — forward parity is
        // the env-gated real-weight suite).
        let cfg = DeepFilterNetConfig::deep_filter_net3();
        assert_eq!(cfg.n_bins(), 481);
        assert_eq!(cfg.emb_dim(), 512);
        let manifest = denoise_tensor_manifest(&cfg);
        assert_eq!(
            manifest.len(),
            115,
            "DFN3 inference manifest is 115 tensors"
        );
        let tensors: BTreeMap<String, Vec<f32>> = denoise_synthesized_tensors(&cfg, 1)
            .into_iter()
            .map(|(s, d)| (s.name, d))
            .collect();
        assert!(DenoiseModel::from_tensors(cfg, tensors).is_ok());
    }

    #[test]
    fn dfn3_manifest_matches_checkpoint_inventory() {
        // Spot-check manifest shapes against the released checkpoint
        // inventory (ckpt-tensors.tsv of the sha256 49c52edc… release).
        let cfg = DeepFilterNetConfig::deep_filter_net3();
        let manifest = denoise_tensor_manifest(&cfg);
        let find = |n: &str| -> &TensorSpec {
            manifest
                .iter()
                .find(|s| s.name == n)
                .unwrap_or_else(|| panic!("missing {n}"))
        };
        assert_eq!(find("enc.erb_conv0.1.weight").shape, vec![64, 1, 3, 3]);
        assert_eq!(find("enc.df_fc_emb.0.weight").shape, vec![32, 96, 16]);
        assert_eq!(find("enc.emb_gru.gru.weight_ih_l0").shape, vec![768, 256]);
        assert_eq!(find("erb_dec.conv3p.0.weight").shape, vec![64, 1, 1, 1]);
        assert_eq!(find("erb_dec.convt2.0.weight").shape, vec![64, 1, 1, 3]);
        assert_eq!(find("erb_dec.conv0_out.0.weight").shape, vec![1, 64, 1, 3]);
        assert_eq!(find("mask.erb_inv_fb").shape, vec![32, 481]);
        assert_eq!(find("df_dec.df_convp.1.weight").shape, vec![10, 32, 5, 1]);
        assert_eq!(
            find("df_dec.df_gru.linear_in.0.weight").shape,
            vec![8, 64, 32]
        );
        assert_eq!(find("df_dec.df_skip.weight").shape, vec![16, 32, 16]);
        assert_eq!(find("df_dec.df_out.0.weight").shape, vec![16, 16, 60]);
        // The total parameter count over the manifest matches the trainable
        // + used-buffer inventory: 133 ckpt tensors − 15 num_batches_tracked
        // − erb_fb − df_fc_a.{weight,bias} = 115.
        let params: usize = manifest
            .iter()
            .map(|s| s.shape.iter().product::<usize>())
            .sum();
        // 2,167,954 total F32 params − erb_fb (481·32) − df_fc_a (257).
        assert_eq!(params, 2_167_954 - 481 * 32 - 257);
    }

    #[test]
    fn erb_widths_match_libdf_published_partition() {
        // The exact libDF erb_fb output for the DFN3 config (widths as
        // reported by df_state.erb_widths() of the released package).
        let cfg = DeepFilterNetConfig::deep_filter_net3();
        let widths = cfg.erb_widths().unwrap();
        assert_eq!(
            widths,
            vec![
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7, 7, 8, 10, 12, 13, 15, 18, 20, 24,
                28, 31, 37, 42, 50, 56, 67
            ]
        );
        assert_eq!(widths.iter().sum::<usize>(), 481);
    }

    #[test]
    fn vorbis_window_matches_upstream_form() {
        // w[i] = sin(π/2·sin²(π(i+0.5)/N)) — checked against values observed
        // from the upstream libDF synthesis path (DC-impulse probe) and the
        // Princen-Bradley condition w²[i] + w²[i+N/2] = 1.
        let w = vorbis_window(960);
        assert!((w[0] - 4.2054917e-06).abs() < 1e-12);
        for i in 0..480 {
            let pb = w[i] * w[i] + w[i + 480] * w[i + 480];
            assert!((pb - 1.0).abs() < 1e-6, "PB violated at {i}: {pb}");
        }
    }

    #[test]
    fn analysis_synthesis_roundtrip_is_identity_after_delay() {
        // The streaming STFT→iSTFT chain is unity up to the n_fft − hop
        // algorithmic delay (the first synthesized chunk is half-windowed —
        // exactly what the enhance() delay trim removes).
        let model = small_model(3);
        let cfg = model.config();
        let n = 4096;
        let x: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.05).sin() * 0.5 + (i as f32 * 0.013).cos() * 0.2)
            .collect();
        let mut padded = x.clone();
        padded.resize(n + cfg.n_fft, 0.0);
        let spec = model.analysis(&padded);
        let out = model.synthesis(&spec);
        let d = cfg.n_fft - cfg.hop;
        let max_delta = x
            .iter()
            .zip(&out[d..d + n])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_delta < 1e-5, "roundtrip max delta {max_delta}");
    }

    #[test]
    fn feat_erb_norm_recurrence_matches_hand_computation() {
        // band_mean_norm_erb: s ← x(1−α) + sα; x ← (x−s)/40, state init
        // linspace(−60, −90).
        let model = small_model(1);
        let cfg = model.config();
        let n = cfg.hop * 3;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).sin()).collect();
        let spec = model.analysis(&x);
        let feats = model.feat_erb(&spec);
        // Recompute frame 0 band 0 by hand.
        let width = model.widths[0];
        let mut acc = 0.0f32;
        let k = 1.0 / width as f32;
        for j in 0..width {
            acc += (spec.re[j] * spec.re[j] + spec.im[j] * spec.im[j]) * k;
        }
        let db = (acc + 1e-10).log10() * 10.0;
        let s0 = db * (1.0 - cfg.norm_alpha) + (-60.0) * cfg.norm_alpha;
        let want = (db - s0) / 40.0;
        assert!((feats[0] - want).abs() < 1e-6);
    }

    #[test]
    fn feat_spec_unit_norm_matches_hand_computation() {
        let model = small_model(1);
        let cfg = model.config();
        let x: Vec<f32> = (0..cfg.hop * 2).map(|i| (i as f32 * 0.21).cos()).collect();
        let spec = model.analysis(&x);
        let (re, im) = model.feat_spec(&spec);
        // Frame 0, bin 1 (state init linspace(0.001, 0.0001, nb)).
        let nb = cfg.df_bins;
        let step = (0.0001 - 0.001) / (nb - 1) as f32;
        let init = 0.001 + step;
        let (r, i) = (spec.re[1], spec.im[1]);
        let s = r.hypot(i) * (1.0 - cfg.norm_alpha) + init * cfg.norm_alpha;
        assert!((re[1] - r / s.sqrt()).abs() < 1e-6);
        assert!((im[1] - i / s.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn enhance_returns_signal_of_input_length() {
        let model = small_model(7);
        let n = 3000; // deliberately not a hop multiple
        let noisy: Vec<f32> = (0..n)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 16000.0).sin())
            .collect();
        let out = denoise(&noisy, &model).unwrap();
        assert_eq!(out.len(), n, "enhanced length must equal input length");
        assert!(out.iter().all(|s| s.is_finite()), "output must be finite");
    }

    #[test]
    fn empty_and_nonfinite_inputs_are_rejected() {
        let model = small_model(1);
        assert!(model.enhance(&[]).is_err());
        assert!(model.enhance(&[0.1, f32::NAN, 0.2]).is_err());
    }

    #[test]
    fn missing_and_unknown_tensors_are_rejected() {
        let cfg = small_cfg();
        // Missing tensor → ModelLoad error naming it.
        let mut tensors: BTreeMap<String, Vec<f32>> = denoise_synthesized_tensors(&cfg, 5)
            .into_iter()
            .map(|(s, d)| (s.name, d))
            .collect();
        tensors.remove("df_dec.df_out.0.weight");
        let err = DenoiseModel::from_tensors(cfg, tensors).unwrap_err();
        assert!(err.to_string().contains("df_dec.df_out.0.weight"), "{err}");
        // Unknown tensor → hard error (layout drift alarm).
        let mut tensors: BTreeMap<String, Vec<f32>> = denoise_synthesized_tensors(&cfg, 5)
            .into_iter()
            .map(|(s, d)| (s.name, d))
            .collect();
        tensors.insert("enc.mystery.weight".into(), vec![0.0; 4]);
        let err = DenoiseModel::from_tensors(cfg, tensors).unwrap_err();
        assert!(err.to_string().contains("enc.mystery.weight"), "{err}");
        // Mis-sized tensor → error naming it and the expected shape.
        let mut tensors: BTreeMap<String, Vec<f32>> = denoise_synthesized_tensors(&cfg, 5)
            .into_iter()
            .map(|(s, d)| (s.name, d))
            .collect();
        tensors.get_mut("mask.erb_inv_fb").unwrap().pop();
        let err = DenoiseModel::from_tensors(cfg, tensors).unwrap_err();
        assert!(err.to_string().contains("mask.erb_inv_fb"), "{err}");
    }

    #[test]
    fn config_validation_rejects_inconsistent_configs() {
        let mut cfg = small_cfg();
        cfg.n_erb = 12; // not a multiple of 8
        assert!(cfg.validate().is_err());
        let mut cfg = small_cfg();
        cfg.hop = 48; // 2·hop > n_fft
        assert!(cfg.validate().is_err());
        let mut cfg = small_cfg();
        cfg.df_lookahead = cfg.df_order; // lookahead must be < order
        assert!(cfg.validate().is_err());
        let mut cfg = small_cfg();
        cfg.linear_groups = 5; // does not divide emb_dim
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn gguf_round_trip_binds_and_reproduces_enhance() {
        // Synthesized tensors → GGUF bytes → parse → from_gguf → enhance
        // must reproduce the original model bit-for-bit.
        let cfg = small_cfg();
        let mut b = vokra_core::gguf::GgufBuilder::new();
        cfg.write_gguf_metadata(&mut b);
        for (s, data) in denoise_synthesized_tensors(&cfg, 42) {
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            let mut dims: Vec<u64> = s.shape.iter().map(|&d| d as u64).collect();
            dims.reverse(); // GGUF stores innermost-first
            b.add_tensor(&s.name, vokra_core::gguf::GgmlType::F32, dims, bytes)
                .unwrap();
        }
        let gguf = vokra_core::gguf::GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let bound = DenoiseModel::from_gguf(&gguf).unwrap();
        assert_eq!(bound.config(), &cfg);

        let reference = small_model(42);
        let noisy: Vec<f32> = (0..2048).map(|i| 0.2 * (i as f32 * 0.05).sin()).collect();
        let a = reference.enhance(&noisy).unwrap();
        let b2 = bound.enhance(&noisy).unwrap();
        assert_eq!(a, b2, "round-tripped model must reproduce enhance exactly");
    }

    #[test]
    fn from_gguf_rejects_missing_keys() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "denoise");
        let gguf = vokra_core::gguf::GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(DenoiseModel::from_gguf(&gguf).is_err());
    }

    #[test]
    fn zero_network_gates_the_spectrum_to_silence() {
        // Force the ERB mask to ~0 (conv0_out BN shift → sigmoid ≈ 0) and the
        // DF coefficients to 0 (df_out weight = 0, df_convp pathway = 0):
        // bins < df_bins are replaced by a zero FIR and the rest are gated by
        // a ~0 mask → output ≈ silence. Proves both output paths gate.
        let cfg = small_cfg();
        let mut tensors: BTreeMap<String, Vec<f32>> = denoise_synthesized_tensors(&cfg, 3)
            .into_iter()
            .map(|(s, d)| (s.name, d))
            .collect();
        tensors
            .get_mut("erb_dec.conv0_out.1.weight")
            .unwrap()
            .fill(0.0);
        tensors
            .get_mut("erb_dec.conv0_out.1.bias")
            .unwrap()
            .fill(-50.0);
        for key in [
            "df_dec.df_out.0.weight",
            "df_dec.df_convp.1.weight",
            "df_dec.df_convp.2.weight",
            "df_dec.df_convp.3.weight",
            "df_dec.df_convp.3.bias",
        ] {
            tensors.get_mut(key).unwrap().fill(0.0);
        }
        let model = DenoiseModel::from_tensors(cfg, tensors).unwrap();
        let noisy: Vec<f32> = (0..2048).map(|i| 0.5 * (i as f32 * 0.1).sin()).collect();
        let out = model.enhance(&noisy).unwrap();
        let peak = out.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(peak < 1e-3, "zero network must silence output, peak {peak}");
    }

    #[test]
    fn taps_expose_consistent_shapes() {
        let model = small_model(9);
        let cfg = *model.config();
        let n = cfg.hop * 4;
        let noisy: Vec<f32> = (0..n).map(|i| (i as f32 * 0.17).sin() * 0.4).collect();
        let (out, taps) = model.enhance_with_taps(&noisy).unwrap();
        assert_eq!(out.len(), n);
        let t = taps.frames;
        assert_eq!(t, (n + cfg.n_fft).div_ceil(cfg.hop));
        assert_eq!(taps.feat_erb.len(), t * cfg.n_erb);
        assert_eq!(taps.feat_spec_re.len(), t * cfg.df_bins);
        assert_eq!(taps.e0.len(), cfg.conv_ch * t * cfg.n_erb);
        assert_eq!(taps.e1.len(), cfg.conv_ch * t * cfg.n_erb / 2);
        assert_eq!(taps.e2.len(), cfg.conv_ch * t * cfg.n_erb / 4);
        assert_eq!(taps.e3.len(), cfg.conv_ch * t * cfg.n_erb / 4);
        assert_eq!(taps.c0.len(), cfg.conv_ch * t * cfg.df_bins);
        assert_eq!(taps.emb.len(), t * cfg.emb_dim());
        assert_eq!(taps.lsnr.len(), t);
        assert_eq!(taps.m.len(), t * cfg.n_erb);
        assert_eq!(taps.coefs.len(), t * cfg.df_bins * cfg.df_order * 2);
        assert_eq!(taps.spec_e_re.len(), t * cfg.n_bins());
        // lsnr is bounded by construction.
        assert!(
            taps.lsnr
                .iter()
                .all(|&v| v >= cfg.lsnr_min && v <= cfg.lsnr_max)
        );
    }

    #[test]
    fn grouped_linear_matches_block_diagonal_matmul() {
        // groups=2, in=4, out=6: y[g·3+h] = Σ_i x[g·2+i]·W[g,i,h].
        let gl = GroupedLinear {
            w: (0..12).map(|i| i as f32 * 0.1).collect(),
            groups: 2,
            in_dim: 4,
            out_dim: 6,
        };
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut y = [0.0f32; 6];
        gl.forward_frame(&x, &mut y);
        // Group 0: W rows [0.0,0.1,0.2] (i=0), [0.3,0.4,0.5] (i=1).
        let want0 = [
            1.0 * 0.0 + 2.0 * 0.3,
            1.0 * 0.1 + 2.0 * 0.4,
            1.0 * 0.2 + 2.0 * 0.5,
        ];
        // Group 1: rows [0.6,0.7,0.8], [0.9,1.0,1.1].
        let want1 = [
            3.0 * 0.6 + 4.0 * 0.9,
            3.0 * 0.7 + 4.0 * 1.0,
            3.0 * 0.8 + 4.0 * 1.1,
        ];
        for (got, want) in y.iter().zip(want0.iter().chain(want1.iter())) {
            assert!((got - want).abs() < 1e-6);
        }
    }

    #[test]
    fn gru_layer_matches_hand_computed_cell() {
        // 1-dim GRU, 1 step, hand-computed with the PyTorch gate order
        // [r, z, n]: h = (1−z)·n (h0 = 0).
        let g = GruLayer {
            w_ih: vec![0.5, -0.3, 0.8],
            w_hh: vec![0.2, 0.4, -0.6],
            b_ih: vec![0.1, 0.0, -0.2],
            b_hh: vec![0.0, 0.1, 0.3],
            input: 1,
            hidden: 1,
        };
        let x = [1.0f32];
        let out = g.forward_seq(&x, 1);
        let r = sigmoid(0.5 * 1.0 + 0.1 + 0.0);
        let z = sigmoid(-0.3 * 1.0 + 0.0 + 0.1);
        let n = (0.8 * 1.0 - 0.2 + r * 0.3).tanh();
        let want = (1.0 - z) * n;
        assert!((out[0] - want).abs() < 1e-6, "{} vs {want}", out[0]);
    }

    #[test]
    fn depthwise_1x1_pathway_conv_is_per_channel_scale() {
        // conv3p-style depthwise 1×1: y[c] = relu(bn(x[c]·w[c])).
        let conv = ConvBlock {
            conv: ConvUnit::Std(Conv2d {
                w: vec![2.0, -1.0],
                in_ch: 2,
                out_ch: 2,
                groups: 2,
                kt: 1,
                kf: 1,
                fstride: 1,
                fpad: 0,
            }),
            pointwise: None,
            bn_scale: vec![1.0, 1.0],
            bn_shift: vec![0.0, 0.0],
            act: Act::Relu,
        };
        let mut x = Act3::zeros(2, 1, 2);
        x.data.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let y = conv.forward(&x);
        assert_eq!(y.data, vec![2.0, 4.0, 0.0, 0.0]); // ch1: −3, −4 → relu 0
    }

    #[test]
    fn transposed_conv_doubles_the_frequency_axis() {
        // Depthwise ConvTranspose (1,3) fstride 2, fpad 1, out_pad 1 on a
        // width-2 input → width 4; kernel [1, 2, 3] on x = [a, b]:
        // full stride-2 scatter minus the pad-1 crop.
        let conv = ConvT2dF {
            w: vec![1.0, 2.0, 3.0],
            in_ch: 1,
            out_ch: 1,
            groups: 1,
            kf: 3,
            fstride: 2,
            fpad: 1,
            out_pad: 1,
        };
        let mut x = Act3::zeros(1, 1, 2);
        x.data.copy_from_slice(&[1.0, 10.0]);
        let y = conv.forward(&x);
        assert_eq!(y.f, 4);
        // Scatter: x0 → out[-1..=1] gets [1,2,3]@[0−1,1−1,2−1]=[−1,0,1];
        // x1 → positions [1,2,3] with [1,2,3].
        assert_eq!(y.data, vec![2.0, 3.0 + 10.0, 20.0, 30.0]);
    }
}

/// Primitive parity against the REAL upstream `df.modules` classes
/// (`Conv2dNormAct` / `ConvTranspose2dNormAct` / `GroupedLinearEinsum` /
/// `SqueezedGRU_S`): committed fixtures generated by
/// `tools/parity/dfn3_primitives_fixture.py` (torch 2.1.2 CPU, seed
/// 20260717) under `tests/parity/dfn3/`. Every structural conv variant of
/// the DFN3 graph is covered. Measured max |Δ| per case (2026-07-17, M1):
/// conv_full33 / conv_grp2 6.0e-8, conv_sep13 / convt_sep13 / sgru 3.0e-8,
/// conv_path / conv_kt51 / glin exact 0.0; bound 1e-5 (~170x headroom for
/// cross-machine libm variance).
#[cfg(test)]
mod primitive_parity {
    use super::*;

    const ATOL: f32 = 1e-5;

    fn fixture_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("tests")
            .join("parity")
            .join("dfn3")
    }

    fn rd(name: &str) -> Vec<f32> {
        let path = fixture_dir().join(format!("{name}.f32"));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        assert_eq!(bytes.len() % 4, 0);
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Folds the dumped BatchNorm stats exactly like the model loader.
    fn bn(case: &str, idx: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
        let gamma = rd(&format!("{case}.{idx}.weight"));
        let beta = rd(&format!("{case}.{idx}.bias"));
        let mean = rd(&format!("{case}.{idx}.running_mean"));
        let var = rd(&format!("{case}.{idx}.running_var"));
        const EPS: f32 = 1e-5;
        let mut scale = vec![0.0f32; n];
        let mut shift = vec![0.0f32; n];
        for i in 0..n {
            let invstd = 1.0 / (var[i] + EPS).sqrt();
            scale[i] = invstd * gamma[i];
            shift[i] = beta[i] - mean[i] * scale[i];
        }
        (scale, shift)
    }

    fn act3(case: &str, name: &str, ch: usize, t: usize, f: usize) -> Act3 {
        let data = rd(&format!("{case}.{name}"));
        assert_eq!(data.len(), ch * t * f, "{case}.{name}");
        Act3 { ch, t, f, data }
    }

    fn assert_close(case: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{case}: length");
        let d = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(d <= ATOL, "{case}: max |Δ| {d:.3e} exceeds {ATOL:.1e}");
    }

    #[test]
    fn conv_full33_matches_upstream_module() {
        // erb_conv0 shape: full (3,3) conv (gcd(1,4)=1 → not separable).
        let (s, b) = bn("conv_full33", 2, 4);
        let block = ConvBlock {
            conv: ConvUnit::Std(Conv2d {
                w: rd("conv_full33.1.weight"),
                in_ch: 1,
                out_ch: 4,
                groups: 1,
                kt: 3,
                kf: 3,
                fstride: 1,
                fpad: 1,
            }),
            pointwise: None,
            bn_scale: s,
            bn_shift: b,
            act: Act::Relu,
        };
        let y = block.forward(&act3("conv_full33", "x", 1, 6, 8));
        assert_close("conv_full33", &y.data, &rd("conv_full33.y"));
    }

    #[test]
    fn conv_sep13_stride2_matches_upstream_module() {
        // erb_conv1 shape: depthwise (1,3) fstride 2 + pointwise.
        let (s, b) = bn("conv_sep13", 2, 4);
        let block = ConvBlock {
            conv: ConvUnit::Std(Conv2d {
                w: rd("conv_sep13.0.weight"),
                in_ch: 4,
                out_ch: 4,
                groups: 4,
                kt: 1,
                kf: 3,
                fstride: 2,
                fpad: 1,
            }),
            pointwise: Some(rd("conv_sep13.1.weight")),
            bn_scale: s,
            bn_shift: b,
            act: Act::Relu,
        };
        let y = block.forward(&act3("conv_sep13", "x", 4, 6, 8));
        assert_eq!(y.f, 4);
        assert_close("conv_sep13", &y.data, &rd("conv_sep13.y"));
    }

    #[test]
    fn conv_grouped_input_matches_upstream_module() {
        // df_conv0 shape: (3,3) groups=gcd(2,4)=2 + pointwise.
        let (s, b) = bn("conv_grp2", 3, 4);
        let block = ConvBlock {
            conv: ConvUnit::Std(Conv2d {
                w: rd("conv_grp2.1.weight"),
                in_ch: 2,
                out_ch: 4,
                groups: 2,
                kt: 3,
                kf: 3,
                fstride: 1,
                fpad: 1,
            }),
            pointwise: Some(rd("conv_grp2.2.weight")),
            bn_scale: s,
            bn_shift: b,
            act: Act::Relu,
        };
        let y = block.forward(&act3("conv_grp2", "x", 2, 6, 8));
        assert_close("conv_grp2", &y.data, &rd("conv_grp2.y"));
    }

    #[test]
    fn pathway_1x1_matches_upstream_module() {
        // conv3p shape: depthwise 1×1 (groups survive max(kernel)==1).
        let (s, b) = bn("conv_path", 1, 4);
        let block = ConvBlock {
            conv: ConvUnit::Std(Conv2d {
                w: rd("conv_path.0.weight"),
                in_ch: 4,
                out_ch: 4,
                groups: 4,
                kt: 1,
                kf: 1,
                fstride: 1,
                fpad: 0,
            }),
            pointwise: None,
            bn_scale: s,
            bn_shift: b,
            act: Act::Relu,
        };
        let y = block.forward(&act3("conv_path", "x", 4, 6, 8));
        assert_close("conv_path", &y.data, &rd("conv_path.y"));
    }

    #[test]
    fn conv_kt51_matches_upstream_module() {
        // df_convp shape: (5,1) time kernel, groups=gcd(4,2)=2 + pointwise.
        let (s, b) = bn("conv_kt51", 3, 2);
        let block = ConvBlock {
            conv: ConvUnit::Std(Conv2d {
                w: rd("conv_kt51.1.weight"),
                in_ch: 4,
                out_ch: 2,
                groups: 2,
                kt: 5,
                kf: 1,
                fstride: 1,
                fpad: 0,
            }),
            pointwise: Some(rd("conv_kt51.2.weight")),
            bn_scale: s,
            bn_shift: b,
            act: Act::Relu,
        };
        let y = block.forward(&act3("conv_kt51", "x", 4, 6, 8));
        assert_close("conv_kt51", &y.data, &rd("conv_kt51.y"));
    }

    #[test]
    fn transposed_sep13_matches_upstream_module() {
        // convt2 shape: depthwise transposed (1,3) fstride 2 + pointwise.
        let (s, b) = bn("convt_sep13", 2, 4);
        let block = ConvBlock {
            conv: ConvUnit::Transposed(ConvT2dF {
                w: rd("convt_sep13.0.weight"),
                in_ch: 4,
                out_ch: 4,
                groups: 4,
                kf: 3,
                fstride: 2,
                fpad: 1,
                out_pad: 1,
            }),
            pointwise: Some(rd("convt_sep13.1.weight")),
            bn_scale: s,
            bn_shift: b,
            act: Act::Relu,
        };
        let y = block.forward(&act3("convt_sep13", "x", 4, 6, 8));
        assert_eq!(y.f, 16);
        assert_close("convt_sep13", &y.data, &rd("convt_sep13.y"));
    }

    #[test]
    fn grouped_linear_matches_upstream_module() {
        let gl = GroupedLinear {
            w: rd("glin.weight"),
            groups: 2,
            in_dim: 6,
            out_dim: 4,
        };
        let x = rd("glin.x");
        let got = gl.forward_seq(&x, 6);
        assert_close("glin", &got, &rd("glin.y"));
    }

    #[test]
    fn squeezed_gru_matches_upstream_module() {
        // erb_dec.emb_gru shape: grouped in → 2-layer GRU → grouped out,
        // ReLU after both linears.
        let g = |name: &str, input: usize| GruLayer {
            w_ih: rd(&format!("sgru.gru.weight_ih_l{name}")),
            w_hh: rd(&format!("sgru.gru.weight_hh_l{name}")),
            b_ih: rd(&format!("sgru.gru.bias_ih_l{name}")),
            b_hh: rd(&format!("sgru.gru.bias_hh_l{name}")),
            input,
            hidden: 8,
        };
        let sgru = SqueezedGru {
            linear_in: GroupedLinear {
                w: rd("sgru.linear_in.0.weight"),
                groups: 2,
                in_dim: 6,
                out_dim: 8,
            },
            layers: vec![g("0", 8), g("1", 8)],
            linear_out: Some(GroupedLinear {
                w: rd("sgru.linear_out.0.weight"),
                groups: 2,
                in_dim: 8,
                out_dim: 6,
            }),
        };
        let x = rd("sgru.x");
        let got = sgru.forward(&x, 6);
        assert_close("sgru", &got, &rd("sgru.y"));
    }
}
