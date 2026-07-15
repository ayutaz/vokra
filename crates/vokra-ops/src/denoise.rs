//! `denoise` — speech enhancement (M4-20 (c), FR-OP-61): a native
//! DeepFilterNet-topology denoiser (CLAUDE.md「denoise — DeepFilterNet (MIT) /
//! GTCRN / RNNoise (BSD) 統合」). whisper.cpp pattern — no ONNX / PyTorch at
//! runtime (NFR-DS-02).
//!
//! # Runtime function, not an `OpKind` variant (ADR M4-20 §D-5)
//!
//! Like `agc` / `hpf` / `loudness_norm`, `denoise` is a first-class API
//! function, not an `OpKind` variant / `dispatch.rs` arm (a graph-side call →
//! `UnsupportedOp` default, FR-EX-08).
//!
//! # Topology (upstream Rikorose/DeepFilterNet — not invented)
//!
//! The DeepFilterNet enhancement chain, transcribed in structure (ADR M4-20
//! §D-5):
//!
//! 1. **STFT** the noisy signal → complex spectrogram `[T, F]`.
//! 2. **ERB features**: pool `|X|²` into `n_erb` ERB-scale bands
//!    (`erb(f) = 21.4·log10(1 + 0.00437·f)`), take `log`.
//! 3. **Encoder** → embedding, **ERB-gain decoder** → per-band real gains in
//!    `[0, 1]` (sigmoid), applied to every bin of the band.
//! 4. **Deep-filter (DF) stage**: for the low `df_bins` bins, a per-bin complex
//!    FIR of order `df_order` over the current + past frames (the DeepFilterNet
//!    "deep filtering" refinement).
//! 5. **iSTFT** → enhanced signal.
//!
//! # Honest scope (ADR M4-20 §D-5, ticket T11/T12/T17)
//!
//! The signal chain (STFT / ERB pooling / gain application / DF complex filter
//! / iSTFT) is faithful. The **neural predictor** ([`DenoiseWeights`]) is a
//! shape-correct per-frame linear scaffold: it makes the forward run end-to-end
//! and shapes right with **synthetic** weights, but the real DeepFilterNet
//! encoder is a conv+GRU stack whose weights load from GGUF (T12). The
//! **numeric** enhancement parity against the real DeepFilterNet checkpoint is
//! the owner leg (T17, GGUF-driven) — it is NOT faked here with a synthetic
//! pass. What this module proves in CI: the topology runs and returns an
//! enhanced waveform of the correct length.

use vokra_core::rng::SplitMix64;
use vokra_core::{Result, VokraError};

use crate::attrs::{IstftAttrs, Normalization, PadMode, StftAttrs, Window, WindowSymmetry};
use crate::istft::istft;
use crate::stft::{Spectrogram, stft};

/// DeepFilterNet architecture / STFT configuration.
///
/// [`DeepFilterNetConfig::deep_filter_net3`] carries the DeepFilterNet3
/// published defaults (48 kHz, `n_fft = 960`, `hop = 480`, `n_erb = 32`,
/// `df_bins = 96`, `df_order = 5`); the concrete layer widths of the real
/// encoder are checkpoint-driven (T12) — `hidden` here sizes the scaffold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeepFilterNetConfig {
    /// STFT size.
    pub n_fft: usize,
    /// STFT hop.
    pub hop: usize,
    /// Sample rate (Hz).
    pub sample_rate: u32,
    /// Number of ERB bands.
    pub n_erb: usize,
    /// Encoder embedding width (scaffold).
    pub hidden: usize,
    /// Number of low-frequency bins the deep-filter stage refines.
    pub df_bins: usize,
    /// Deep-filter order (taps over current + past frames).
    pub df_order: usize,
}

impl DeepFilterNetConfig {
    /// DeepFilterNet3 published defaults (Rikorose/DeepFilterNet).
    pub fn deep_filter_net3() -> Self {
        Self {
            n_fft: 960,
            hop: 480,
            sample_rate: 48000,
            n_erb: 32,
            hidden: 256,
            df_bins: 96,
            df_order: 5,
        }
    }

    /// Number of one-sided (RFFT) frequency bins, `n_fft / 2 + 1`.
    pub fn n_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    fn validate(&self) -> Result<()> {
        if self.n_fft == 0 || self.hop == 0 || self.sample_rate == 0 {
            return Err(VokraError::InvalidArgument(
                "denoise: n_fft / hop / sample_rate must be > 0".into(),
            ));
        }
        if self.n_erb == 0 || self.hidden == 0 || self.df_order == 0 {
            return Err(VokraError::InvalidArgument(
                "denoise: n_erb / hidden / df_order must be > 0".into(),
            ));
        }
        if self.df_bins > self.n_bins() {
            return Err(VokraError::InvalidArgument(format!(
                "denoise: df_bins {} exceeds n_bins {}",
                self.df_bins,
                self.n_bins()
            )));
        }
        Ok(())
    }
}

/// Per-frame linear layer `[in, out]` (row-major weight) + bias `[out]`.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseLayer {
    /// Row-major `[in_dim, out_dim]`.
    pub weight: Vec<f32>,
    /// `[out_dim]`.
    pub bias: Vec<f32>,
    /// Input width.
    pub in_dim: usize,
    /// Output width.
    pub out_dim: usize,
}

impl DenseLayer {
    fn check(&self, in_dim: usize, out_dim: usize, name: &str) -> Result<()> {
        if self.in_dim != in_dim || self.out_dim != out_dim {
            return Err(VokraError::InvalidArgument(format!(
                "denoise: {name} shape [{}, {}] != expected [{in_dim}, {out_dim}]",
                self.in_dim, self.out_dim
            )));
        }
        if self.weight.len() != in_dim * out_dim || self.bias.len() != out_dim {
            return Err(VokraError::InvalidArgument(format!(
                "denoise: {name} buffer sizes inconsistent with dims"
            )));
        }
        Ok(())
    }

    /// `y = x @ weight + bias` for one frame (`x` length `in_dim`).
    fn forward_frame(&self, x: &[f32], out: &mut [f32]) {
        for (o, out_o) in out.iter_mut().enumerate().take(self.out_dim) {
            let mut acc = self.bias[o];
            for (i, &xi) in x.iter().enumerate().take(self.in_dim) {
                acc += xi * self.weight[i * self.out_dim + o];
            }
            *out_o = acc;
        }
    }
}

/// The DeepFilterNet neural predictor (scaffold): an encoder, an ERB-gain
/// decoder, and a deep-filter (complex-coefficient) decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct DenoiseWeights {
    /// `n_erb → hidden` (tanh).
    pub encoder: DenseLayer,
    /// `hidden → n_erb` (sigmoid gains).
    pub erb_decoder: DenseLayer,
    /// `hidden → df_bins·df_order·2` (tanh complex FIR coefficients).
    pub df_decoder: DenseLayer,
}

impl DenoiseWeights {
    /// Deterministic synthetic weights for shape-sanity / round-trip tests
    /// (SplitMix64-seeded, small values). Not a trained network — the real
    /// DeepFilterNet weights load from GGUF (T12); numeric parity is owner
    /// (T17).
    pub fn synthesized(cfg: &DeepFilterNetConfig, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let mut layer = |in_dim: usize, out_dim: usize| {
            let weight = (0..in_dim * out_dim)
                .map(|_| (rng.next_unit_f32() - 0.5) * 0.2)
                .collect();
            let bias = (0..out_dim)
                .map(|_| (rng.next_unit_f32() - 0.5) * 0.1)
                .collect();
            DenseLayer {
                weight,
                bias,
                in_dim,
                out_dim,
            }
        };
        let df_out = cfg.df_bins * cfg.df_order * 2;
        Self {
            encoder: layer(cfg.n_erb, cfg.hidden),
            erb_decoder: layer(cfg.hidden, cfg.n_erb),
            df_decoder: layer(cfg.hidden, df_out),
        }
    }

    fn validate(&self, cfg: &DeepFilterNetConfig) -> Result<()> {
        self.encoder.check(cfg.n_erb, cfg.hidden, "encoder")?;
        self.erb_decoder
            .check(cfg.hidden, cfg.n_erb, "erb_decoder")?;
        self.df_decoder
            .check(cfg.hidden, cfg.df_bins * cfg.df_order * 2, "df_decoder")?;
        Ok(())
    }
}

/// A bound DeepFilterNet denoiser: config + weights + the derived ERB
/// bin→band map.
#[derive(Debug, Clone)]
pub struct DenoiseModel {
    cfg: DeepFilterNetConfig,
    weights: DenoiseWeights,
    /// ERB band index for each of the `n_bins` FFT bins.
    band_of_bin: Vec<usize>,
    /// Number of FFT bins in each ERB band.
    band_count: Vec<usize>,
}

impl DenoiseModel {
    /// Binds a model from a config + weights (validates shapes).
    pub fn new(cfg: DeepFilterNetConfig, weights: DenoiseWeights) -> Result<Self> {
        cfg.validate()?;
        weights.validate(&cfg)?;
        let (band_of_bin, band_count) = erb_band_map(&cfg);
        Ok(Self {
            cfg,
            weights,
            band_of_bin,
            band_count,
        })
    }

    /// The model config.
    pub fn config(&self) -> &DeepFilterNetConfig {
        &self.cfg
    }

    /// Enhances `noisy`, returning a signal of the same length.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] for an empty or non-finite input.
    pub fn forward(&self, noisy: &[f32]) -> Result<Vec<f32>> {
        if noisy.is_empty() {
            return Err(VokraError::InvalidArgument("denoise: empty input".into()));
        }
        if noisy.iter().any(|s| !s.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "denoise: input has a non-finite sample".into(),
            ));
        }
        let cfg = &self.cfg;
        let stft_attrs = self.stft_attrs();
        let mut spec = stft(noisy, &stft_attrs)?;
        let (t, f) = (spec.frames, spec.bins);
        debug_assert_eq!(f, cfg.n_bins());

        // (2) ERB features: log band energy [T, n_erb].
        let power = spec.power();
        let mut erb_feat = vec![0.0f32; t * cfg.n_erb];
        for ti in 0..t {
            for bin in 0..f {
                erb_feat[ti * cfg.n_erb + self.band_of_bin[bin]] += power[ti * f + bin];
            }
            for b in 0..cfg.n_erb {
                let cnt = self.band_count[b].max(1) as f32;
                let e = erb_feat[ti * cfg.n_erb + b] / cnt;
                erb_feat[ti * cfg.n_erb + b] = (e + 1e-10).ln();
            }
        }

        // (3) Encoder → ERB gains, then apply per band to every bin.
        let mut hidden = vec![0.0f32; cfg.hidden];
        let mut erb_gain = vec![0.0f32; cfg.n_erb];
        let df_out = cfg.df_bins * cfg.df_order * 2;
        let mut df_coef = vec![0.0f32; t * df_out];
        for ti in 0..t {
            self.weights
                .encoder
                .forward_frame(&erb_feat[ti * cfg.n_erb..(ti + 1) * cfg.n_erb], &mut hidden);
            for h in &mut hidden {
                *h = h.tanh();
            }
            self.weights
                .erb_decoder
                .forward_frame(&hidden, &mut erb_gain);
            for g in &mut erb_gain {
                *g = sigmoid(*g);
            }
            // Apply ERB gains to every bin of the frame.
            for bin in 0..f {
                let g = erb_gain[self.band_of_bin[bin]];
                spec.re[ti * f + bin] *= g;
                spec.im[ti * f + bin] *= g;
            }
            // (4) Deep-filter coefficients for this frame.
            self.weights
                .df_decoder
                .forward_frame(&hidden, &mut df_coef[ti * df_out..(ti + 1) * df_out]);
            for c in &mut df_coef[ti * df_out..(ti + 1) * df_out] {
                *c = c.tanh();
            }
        }

        // (4) Deep filtering on the low `df_bins` bins: per-bin complex FIR of
        // order `df_order` over the current + past `df_order-1` frames of the
        // ERB-gained spectrum. Reads a snapshot so taps use the pre-DF values.
        if cfg.df_bins > 0 {
            let re0 = spec.re.clone();
            let im0 = spec.im.clone();
            for ti in 0..t {
                for j in 0..cfg.df_bins {
                    let mut acc_re = 0.0f32;
                    let mut acc_im = 0.0f32;
                    for k in 0..cfg.df_order {
                        if ti < k {
                            break;
                        }
                        let src = (ti - k) * f + j;
                        let ci = (ti * df_out) + (j * cfg.df_order + k) * 2;
                        let (cr, cc) = (df_coef[ci], df_coef[ci + 1]);
                        // (cr + i·cc) · (re + i·im).
                        acc_re += cr * re0[src] - cc * im0[src];
                        acc_im += cr * im0[src] + cc * re0[src];
                    }
                    spec.re[ti * f + j] = acc_re;
                    spec.im[ti * f + j] = acc_im;
                }
            }
        }

        // (5) iSTFT back to the time domain (trim to the input length).
        let istft_attrs = IstftAttrs {
            n_fft: cfg.n_fft,
            hop_length: cfg.hop,
            win_length: cfg.n_fft,
            window: Window::Hann,
            window_symmetry: WindowSymmetry::Periodic,
            center: true,
            normalization: Normalization::Backward,
            real_input: true,
            length: Some(noisy.len()),
        };
        let out = istft(&spec_owned(&spec), &istft_attrs)?;
        Ok(out)
    }

    fn stft_attrs(&self) -> StftAttrs {
        StftAttrs {
            n_fft: self.cfg.n_fft,
            hop_length: self.cfg.hop,
            win_length: self.cfg.n_fft,
            window: Window::Hann,
            window_symmetry: WindowSymmetry::Periodic,
            center: true,
            pad_mode: PadMode::Reflect,
            normalization: Normalization::Backward,
            causal: false,
            real_input: true,
        }
    }
}

/// Rebuilds an owned [`Spectrogram`] (istft takes `&Spectrogram`; we already
/// hold one, so this is just a borrow shim kept explicit for clarity).
fn spec_owned(spec: &Spectrogram) -> Spectrogram {
    Spectrogram {
        frames: spec.frames,
        bins: spec.bins,
        re: spec.re.clone(),
        im: spec.im.clone(),
    }
}

/// One-shot [`DenoiseModel::forward`] convenience.
///
/// # Errors
///
/// Propagates [`DenoiseModel::forward`] errors.
pub fn denoise(noisy: &[f32], model: &DenoiseModel) -> Result<Vec<f32>> {
    model.forward(noisy)
}

// ---- GGUF binding (M4-20 T12): `vokra.denoise.*` --------------------------
//
// Config keys are u32; every neural tensor is a flat F32 blob keyed under the
// `vokra.denoise.*` namespace (reshape is config-driven, so no dim ambiguity).
// The converter (`vokra-cli convert --model denoise`, T12) writes this layout
// from a prepared DeepFilterNet checkpoint (real .ckpt parsing is owner). The
// round-trip test writes it from synthesized weights and reads it back.

const KEY_N_FFT: &str = "vokra.denoise.n_fft";
const KEY_HOP: &str = "vokra.denoise.hop";
const KEY_SAMPLE_RATE: &str = "vokra.denoise.sample_rate";
const KEY_N_ERB: &str = "vokra.denoise.n_erb";
const KEY_HIDDEN: &str = "vokra.denoise.hidden";
const KEY_DF_BINS: &str = "vokra.denoise.df_bins";
const KEY_DF_ORDER: &str = "vokra.denoise.df_order";

const T_ENC_W: &str = "vokra.denoise.encoder.weight";
const T_ENC_B: &str = "vokra.denoise.encoder.bias";
const T_ERB_W: &str = "vokra.denoise.erb_decoder.weight";
const T_ERB_B: &str = "vokra.denoise.erb_decoder.bias";
const T_DF_W: &str = "vokra.denoise.df_decoder.weight";
const T_DF_B: &str = "vokra.denoise.df_decoder.bias";

impl DeepFilterNetConfig {
    /// Reads the `vokra.denoise.*` config keys from a parsed GGUF.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] for a missing / non-u32 key.
    pub fn from_gguf(gguf: &vokra_core::gguf::GgufFile) -> Result<Self> {
        let u = |key: &str| -> Result<usize> {
            gguf.get(key)
                .and_then(|v| v.as_u64())
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| VokraError::ModelLoad(format!("denoise gguf: missing/bad `{key}`")))
        };
        let cfg = Self {
            n_fft: u(KEY_N_FFT)?,
            hop: u(KEY_HOP)?,
            sample_rate: u(KEY_SAMPLE_RATE)? as u32,
            n_erb: u(KEY_N_ERB)?,
            hidden: u(KEY_HIDDEN)?,
            df_bins: u(KEY_DF_BINS)?,
            df_order: u(KEY_DF_ORDER)?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn write_metadata(&self, b: &mut vokra_core::gguf::GgufBuilder) {
        b.add_u32(KEY_N_FFT, self.n_fft as u32);
        b.add_u32(KEY_HOP, self.hop as u32);
        b.add_u32(KEY_SAMPLE_RATE, self.sample_rate);
        b.add_u32(KEY_N_ERB, self.n_erb as u32);
        b.add_u32(KEY_HIDDEN, self.hidden as u32);
        b.add_u32(KEY_DF_BINS, self.df_bins as u32);
        b.add_u32(KEY_DF_ORDER, self.df_order as u32);
    }
}

impl DenoiseWeights {
    /// Reads the `vokra.denoise.*` neural tensors from a parsed GGUF and
    /// validates them against `cfg`.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] for a missing tensor; [`VokraError::InvalidArgument`]
    /// for a shape mismatch against `cfg`.
    pub fn from_gguf(gguf: &vokra_core::gguf::GgufFile, cfg: &DeepFilterNetConfig) -> Result<Self> {
        let t = |name: &str| -> Result<Vec<f32>> { Ok(gguf.tensor_f32(name)?) };
        let dense = |w: Vec<f32>, bias: Vec<f32>, in_dim: usize, out_dim: usize| DenseLayer {
            weight: w,
            bias,
            in_dim,
            out_dim,
        };
        let df_out = cfg.df_bins * cfg.df_order * 2;
        let w = Self {
            encoder: dense(t(T_ENC_W)?, t(T_ENC_B)?, cfg.n_erb, cfg.hidden),
            erb_decoder: dense(t(T_ERB_W)?, t(T_ERB_B)?, cfg.hidden, cfg.n_erb),
            df_decoder: dense(t(T_DF_W)?, t(T_DF_B)?, cfg.hidden, df_out),
        };
        w.validate(cfg)?;
        Ok(w)
    }

    fn write_tensors(&self, b: &mut vokra_core::gguf::GgufBuilder) -> Result<()> {
        use vokra_core::gguf::GgmlType;
        let mut add = |name: &str, data: &[f32]| -> Result<()> {
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            b.add_tensor(name, GgmlType::F32, vec![data.len() as u64], bytes)?;
            Ok(())
        };
        add(T_ENC_W, &self.encoder.weight)?;
        add(T_ENC_B, &self.encoder.bias)?;
        add(T_ERB_W, &self.erb_decoder.weight)?;
        add(T_ERB_B, &self.erb_decoder.bias)?;
        add(T_DF_W, &self.df_decoder.weight)?;
        add(T_DF_B, &self.df_decoder.bias)?;
        Ok(())
    }
}

impl DenoiseModel {
    /// Binds a denoiser from a parsed GGUF (`vokra.denoise.*` config +
    /// tensors).
    ///
    /// # Errors
    ///
    /// Propagates [`DeepFilterNetConfig::from_gguf`] / [`DenoiseWeights::from_gguf`].
    pub fn from_gguf(gguf: &vokra_core::gguf::GgufFile) -> Result<Self> {
        let cfg = DeepFilterNetConfig::from_gguf(gguf)?;
        let weights = DenoiseWeights::from_gguf(gguf, &cfg)?;
        Self::new(cfg, weights)
    }

    /// Serializes this model to a `vokra.denoise.*` GGUF byte buffer (the
    /// converter's core; the CLI parses a real DeepFilterNet checkpoint into a
    /// [`DenoiseModel`] first — owner-side — then calls this).
    ///
    /// # Errors
    ///
    /// [`VokraError`] wrapping any GGUF write error.
    pub fn to_gguf_bytes(&self) -> Result<Vec<u8>> {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "denoise");
        self.cfg.write_metadata(&mut b);
        self.weights.write_tensors(&mut b)?;
        Ok(b.to_bytes()?)
    }
}

/// Maps each of the `n_bins` FFT bins to an ERB band (linear spacing on the
/// ERB scale between 0 and the Nyquist), returning `(band_of_bin, band_count)`.
fn erb_band_map(cfg: &DeepFilterNetConfig) -> (Vec<usize>, Vec<usize>) {
    let n_bins = cfg.n_bins();
    let nyquist = cfg.sample_rate as f32 * 0.5;
    let erb_max = hz_to_erb(nyquist).max(1e-6);
    let mut band_of_bin = vec![0usize; n_bins];
    let mut band_count = vec![0usize; cfg.n_erb];
    for (bin, slot) in band_of_bin.iter_mut().enumerate() {
        let hz = bin as f32 / (n_bins - 1).max(1) as f32 * nyquist;
        let e = hz_to_erb(hz);
        let mut band = ((e / erb_max) * cfg.n_erb as f32) as usize;
        if band >= cfg.n_erb {
            band = cfg.n_erb - 1;
        }
        *slot = band;
        band_count[band] += 1;
    }
    (band_of_bin, band_count)
}

/// ERB-scale mapping `erb(f) = 21.4·log10(1 + 0.00437·f)` (Glasberg & Moore).
fn hz_to_erb(hz: f32) -> f32 {
    21.4 * (1.0 + 0.00437 * hz).log10()
}

/// Logistic sigmoid.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> DeepFilterNetConfig {
        DeepFilterNetConfig {
            n_fft: 64,
            hop: 16,
            sample_rate: 16000,
            n_erb: 8,
            hidden: 16,
            df_bins: 6,
            df_order: 3,
        }
    }

    #[test]
    fn deep_filter_net3_defaults_construct() {
        // The published DeepFilterNet3 config + synthetic weights binds.
        let cfg = DeepFilterNetConfig::deep_filter_net3();
        let w = DenoiseWeights::synthesized(&cfg, 1);
        assert!(DenoiseModel::new(cfg, w).is_ok());
        assert_eq!(cfg.n_bins(), 481);
    }

    #[test]
    fn forward_returns_enhanced_signal_of_input_length() {
        // Shape sanity: synthetic weights → forward → enhanced waveform of the
        // SAME length as the noisy input, all finite (T11 completion bar).
        let cfg = small_cfg();
        let model = DenoiseModel::new(cfg, DenoiseWeights::synthesized(&cfg, 7)).unwrap();
        let n = 4096;
        let noisy: Vec<f32> = (0..n)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 16000.0).sin())
            .collect();
        let out = denoise(&noisy, &model).unwrap();
        assert_eq!(out.len(), n, "enhanced length must equal input length");
        assert!(out.iter().all(|s| s.is_finite()), "output must be finite");
    }

    #[test]
    fn erb_band_map_covers_all_bins_monotonically() {
        let cfg = small_cfg();
        let (band_of_bin, band_count) = erb_band_map(&cfg);
        assert_eq!(band_of_bin.len(), cfg.n_bins());
        assert_eq!(band_count.iter().sum::<usize>(), cfg.n_bins());
        // ERB scale is monotone in Hz, so bin→band must be non-decreasing.
        for w in band_of_bin.windows(2) {
            assert!(
                w[1] >= w[0],
                "bin→band must be non-decreasing: {band_of_bin:?}"
            );
        }
        assert!(band_of_bin.iter().all(|&b| b < cfg.n_erb));
    }

    #[test]
    fn shape_mismatched_weights_are_rejected() {
        let cfg = small_cfg();
        let mut w = DenoiseWeights::synthesized(&cfg, 1);
        w.encoder.out_dim += 1; // corrupt the encoder shape
        assert!(matches!(
            DenoiseModel::new(cfg, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn empty_and_nonfinite_inputs_are_rejected() {
        let cfg = small_cfg();
        let model = DenoiseModel::new(cfg, DenoiseWeights::synthesized(&cfg, 1)).unwrap();
        assert!(model.forward(&[]).is_err());
        assert!(model.forward(&[0.1, f32::NAN, 0.2]).is_err());
    }

    #[test]
    fn gguf_round_trip_binds_and_reproduces_forward() {
        // T12: synthesized weights → GGUF bytes → parse → from_gguf → forward
        // must reproduce the original model's output bit-for-bit (same weights,
        // same config). Proves the `vokra.denoise.*` write/read binding.
        let cfg = small_cfg();
        let model = DenoiseModel::new(cfg, DenoiseWeights::synthesized(&cfg, 42)).unwrap();
        let bytes = model.to_gguf_bytes().unwrap();

        let gguf = vokra_core::gguf::GgufFile::parse(bytes).unwrap();
        let bound = DenoiseModel::from_gguf(&gguf).unwrap();
        assert_eq!(bound.config(), &cfg, "config must round-trip");
        assert_eq!(bound.weights, model.weights, "weights must round-trip");

        let noisy: Vec<f32> = (0..2048).map(|i| 0.2 * (i as f32 * 0.05).sin()).collect();
        let a = model.forward(&noisy).unwrap();
        let b = bound.forward(&noisy).unwrap();
        assert_eq!(
            a, b,
            "round-tripped model must reproduce the forward exactly"
        );
    }

    #[test]
    fn from_gguf_rejects_missing_keys() {
        // A GGUF with the arch marker but no denoise config keys is a load error
        // (never a silent default).
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "denoise");
        let gguf = vokra_core::gguf::GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(DenoiseModel::from_gguf(&gguf).is_err());
    }

    #[test]
    fn zero_gain_network_silences_output() {
        // A network forced to output ~0 ERB gains (huge negative decoder bias →
        // sigmoid → 0) and zero DF coefficients must drive the enhanced signal
        // toward silence — proves the gains actually gate the spectrum.
        let cfg = small_cfg();
        let mut w = DenoiseWeights::synthesized(&cfg, 3);
        for b in &mut w.erb_decoder.bias {
            *b = -50.0; // sigmoid(-50) ≈ 0
        }
        w.erb_decoder.weight.iter_mut().for_each(|v| *v = 0.0);
        w.df_decoder.weight.iter_mut().for_each(|v| *v = 0.0);
        w.df_decoder.bias.iter_mut().for_each(|v| *v = 0.0);
        let model = DenoiseModel::new(cfg, w).unwrap();
        let noisy: Vec<f32> = (0..2048).map(|i| 0.5 * (i as f32 * 0.1).sin()).collect();
        let out = denoise(&noisy, &model).unwrap();
        let peak = out.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(
            peak < 1e-3,
            "zero-gain network must silence output, peak {peak}"
        );
    }
}
