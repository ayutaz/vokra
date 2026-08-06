//! NKF-AEC (`fjiang9/NKF-AEC`, MIT repo LICENSE + BSD-3-Clause source
//! file — Yang et al. ICASSP 2023, arXiv:2207.11388) — runtime binder
//! for the `nkf_aec` converter arch (2026-08-05).
//!
//! # Upstream primary source
//!
//! Every dim, tensor name, hparam, and algorithm step below is
//! transcribed verbatim from `github.com/fjiang9/NKF-AEC/blob/main/
//! src/nkf.py` (Tencent, BSD-3-Clause). The runtime forward is a
//! whisper.cpp-style native re-implementation — the upstream Python
//! defines the reference, this file consumes only the flattened
//! `state_dict` (via the `tools/parity/nkf_aec_prepare_checkpoint.py`
//! offline sidecar + the `vokra-convert nkf-aec` GGUF emitter).
//!
//! # Runtime layout
//!
//! ```text
//! mic PCM (16 kHz mono f32) ──┐
//!                             ├─► rolling frame pair (hop=256 samples)
//! farend PCM (16 kHz mono f32)┘
//!   -> STFT (n_fft=1024, hop=256, win=1024, Hann, center)
//!      -> mic spectrogram Y[F=513, T]  (complex)
//!      -> ref spectrogram X[F=513, T]  (complex)
//!   -> per-bin adaptive Kalman filter loop, ONE forward-pass per new
//!      frame `t`:
//!        (a) build ref tap window `xt[k, :] = X[k, t-L+1..=t]`  (L=4)
//!        (b) if ‖xt‖₁ < 1e-5: skip (dead-air short-circuit; echo_hat
//!            stays 0, filter state carries)
//!        (c) dh = h_posterior - h_prior
//!            h_prior = h_posterior
//!        (d) e[k] = Y[k, t] - <xt[k, :], h_prior[k, :]>
//!        (e) feat[k, :] = [xt[k, :], e[k], dh[k, :]]  # 2L+1 = 9 complex
//!            kg[k, :] = KgNet(feat[k, :])            # L complex
//!        (f) h_posterior[k, :] = h_prior[k, :] + kg[k, :] * e[k]
//!            echo_hat[k, t] = <xt[k, :], h_posterior[k, :]>
//!   -> E[k, t] = Y[k, t] - echo_hat[k, t]  (cleaned STFT)
//!   -> iSTFT (matching Hann/COLA, `center = true`) -> cleaned PCM
//! ```
//!
//! # `KgNet` (upstream `nkf.py::KGNet`)
//!
//! Two-block `ComplexDense → ComplexPReLU → ComplexGRU →
//! ComplexDense → ComplexPReLU → ComplexDense` with **fixed** widths:
//!
//! - `fc_in`  — [`ComplexDense`]`(in = 2L+1 = 9, out = fc_dim = 18)`
//! - `prelu_1` — [`ComplexPReLU`]
//! - `complex_gru` — [`ComplexGru`]`(input_size = fc_dim = 18,
//!                                  hidden_size = rnn_dim = 18,
//!                                  layers = 1)`
//! - `fc_out.0` — [`ComplexDense`]`(in = rnn_dim = 18, out = fc_dim = 18)`
//! - `prelu_2` — [`ComplexPReLU`]
//! - `fc_out.2` — [`ComplexDense`]`(in = fc_dim = 18, out = L = 4)`
//!
//! Every `Complex*` primitive is a pair of real primitives (see the
//! `ComplexGru` / `ComplexDense` / `ComplexPReLU` doc comments) — 22
//! real tensors in total.
//!
//! # `vokra.nkf_aec.*` chunk group (converter contract)
//!
//! The BF16-passthrough converter (`crates/vokra-convert/src/models/
//! nkf_aec.rs`) does NOT emit a `vokra.nkf_aec.*` chunk group today
//! (see its module doc: "hparams are recovered from tensor shapes"). All
//! dims (`F`, `L`, `fc_dim`, `rnn_dim`, `sample_rate`) are pinned by
//! upstream (`nkf.py`) and cross-verified against the loaded tensors'
//! `dimensions` slots at [`NkfAecWeights::from_gguf`] time. A future
//! release that ships a different `L` / `fc_dim` variant would emit
//! this chunk group; every load-bearing dim is transcribed here in one
//! place so the loader validation stays honest until then.
//!
//! # FR-EX-08 posture
//!
//! - **Sample rate**: the shared complex weights were fit at 16 kHz;
//!   [`NkfAec::open_stream`] refuses any other rate loudly.
//! - **Mic / far-end length mismatch**: [`NkfAecStream::push_paired`]
//!   refuses `mic.len() != farend.len()` loudly — the two streams are
//!   strictly sample-aligned in AEC (silent trim / repeat is a
//!   correctness bug, not a convenience).
//! - **Tensor dim gates**: every one of the 22 tensors is checked
//!   against the exact `[out, in]` (linear) / `[3H, D]` (GRU) row-major
//!   layout the upstream state_dict emits (openWakeWord precedent —
//!   defense against a Python bridge that silently writes the
//!   transpose).
//! - **Non-CPU backend**: [`NkfAecStream::push_paired`] runs entirely
//!   on the CPU; requesting a different backend at the session level is
//!   a load-time `UnsupportedOp` because no backend arm is wired yet.

use std::sync::Arc;

use vokra_core::Complex32;
use vokra_core::engines::{AecEngine, AecStreamHandle};
use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::ir::graph::{Normalization, Window, WindowSymmetry};
use vokra_core::{Result, VokraError};
use vokra_ops::fft::RealFftPlan;
use vokra_ops::window::window;

#[cfg(test)]
mod tests;

// ---- arch / provenance constants ----------------------------------------
//
// Mirror of `vokra-convert::models::nkf_aec::{ARCH, NAME, CATEGORY}` —
// kept as duplicated `pub const` so the runtime binder does not add a
// cross-crate dependency edge onto the converter (the sibling
// fsmn_vad / openwakeword convention).

/// Expected `vokra.model.arch` value written by
/// `vokra-convert --model nkf-aec`.
pub const ARCH: &str = "nkf_aec";

/// Default `vokra.model.name` value written by the converter.
pub const DEFAULT_NAME: &str = "nkf-aec";

/// `vokra.model.category` — AEC family.
pub const CATEGORY: &str = "aec";

// ---- upstream-pinned dims (nkf.py::NKF / KGNet — hardcoded release axes)
//
// Every one of these is a fixed constant in upstream `nkf.py`; a variant
// checkpoint that trains a different width would need its own arch
// tag (silently changing any of these here would misforward every
// per-bin Kalman step, so they are pinned rather than loaded).

/// Number of Kalman filter taps per bin (`nkf.py::NKF.__init__` L=4).
pub const L: usize = 4;

/// Hidden width shared by every `ComplexDense` intermediate + the
/// `ComplexGRU` (`KGNet.__init__` fc_dim = rnn_dim = 18).
pub const H: usize = 18;

/// Fully-connected width consumed by the first / middle
/// `ComplexDense` (`KGNet.__init__` fc_dim=18).
pub const FC_DIM: usize = H;

/// GRU hidden width (`KGNet.__init__` rnn_dim=18).
pub const RNN_DIM: usize = H;

/// Number of `ComplexGru` layers (`KGNet.__init__` rnn_layers=1).
pub const RNN_LAYERS: usize = 1;

/// KGNet input feature width (upstream `input_feature` shape
/// `2*L + 1 = 9` complex values per bin — see the module docs step (e)).
pub const KGNET_IN: usize = 2 * L + 1;

/// STFT FFT size (`nkf.py::NKF.__init__` n_fft=1024).
pub const N_FFT: usize = 1024;

/// STFT hop size (`nkf.py::NKF.__init__` hop_length=256).
pub const HOP: usize = 256;

/// STFT window length (`nkf.py::NKF.__init__` win_length=1024).
pub const WIN_LENGTH: usize = N_FFT;

/// Number of RFFT bins (`n_fft/2 + 1`).
pub const F_BINS: usize = N_FFT / 2 + 1;

/// PCM sample rate the release was trained at (the paper / demo assets
/// use 16 kHz).
pub const SAMPLE_RATE: u32 = 16_000;

/// Dead-air short-circuit threshold (`nkf.py::NKF.forward`
/// `if xt.abs().mean() < 1e-5: continue`).
pub const XT_ZERO_THRESHOLD: f32 = 1e-5;

// ---- tensor name conventions --------------------------------------------
//
// Verbatim upstream torch `state_dict` keys (see `nkf.py`) — the
// converter emits every float tensor pass-through and the loader binds
// against these exact identifiers. Documented as `pub const` strings
// so the tests + adjacent modules can reference them without stringly
// duplication.

/// `KGNet.fc_in.0.linear_real.weight` (shape `[H, 2L+1] = [18, 9]`).
pub const T_FC_IN_LR_W: &str = "kg_net.fc_in.0.linear_real.weight";
/// `KGNet.fc_in.0.linear_real.bias` (shape `[H] = [18]`).
pub const T_FC_IN_LR_B: &str = "kg_net.fc_in.0.linear_real.bias";
/// `KGNet.fc_in.0.linear_imag.weight` (shape `[H, 2L+1] = [18, 9]`).
pub const T_FC_IN_LI_W: &str = "kg_net.fc_in.0.linear_imag.weight";
/// `KGNet.fc_in.0.linear_imag.bias` (shape `[H] = [18]`).
pub const T_FC_IN_LI_B: &str = "kg_net.fc_in.0.linear_imag.bias";
/// `KGNet.fc_in.1.prelu.weight` (shape `[1]`).
pub const T_FC_IN_PRELU: &str = "kg_net.fc_in.1.prelu.weight";

/// `complex_gru.gru_r.weight_ih_l0` (shape `[3*H, H] = [54, 18]`).
pub const T_GRU_R_W_IH: &str = "kg_net.complex_gru.gru_r.weight_ih_l0";
/// `complex_gru.gru_r.weight_hh_l0` (shape `[3*H, H] = [54, 18]`).
pub const T_GRU_R_W_HH: &str = "kg_net.complex_gru.gru_r.weight_hh_l0";
/// `complex_gru.gru_r.bias_ih_l0` (shape `[3*H] = [54]`).
pub const T_GRU_R_B_IH: &str = "kg_net.complex_gru.gru_r.bias_ih_l0";
/// `complex_gru.gru_r.bias_hh_l0` (shape `[3*H] = [54]`).
pub const T_GRU_R_B_HH: &str = "kg_net.complex_gru.gru_r.bias_hh_l0";
/// `complex_gru.gru_i.weight_ih_l0` (shape `[3*H, H] = [54, 18]`).
pub const T_GRU_I_W_IH: &str = "kg_net.complex_gru.gru_i.weight_ih_l0";
/// `complex_gru.gru_i.weight_hh_l0` (shape `[3*H, H] = [54, 18]`).
pub const T_GRU_I_W_HH: &str = "kg_net.complex_gru.gru_i.weight_hh_l0";
/// `complex_gru.gru_i.bias_ih_l0` (shape `[3*H] = [54]`).
pub const T_GRU_I_B_IH: &str = "kg_net.complex_gru.gru_i.bias_ih_l0";
/// `complex_gru.gru_i.bias_hh_l0` (shape `[3*H] = [54]`).
pub const T_GRU_I_B_HH: &str = "kg_net.complex_gru.gru_i.bias_hh_l0";

/// `KGNet.fc_out.0.linear_real.weight` (shape `[H, H] = [18, 18]`).
pub const T_FC_OUT0_LR_W: &str = "kg_net.fc_out.0.linear_real.weight";
/// `KGNet.fc_out.0.linear_real.bias` (shape `[H] = [18]`).
pub const T_FC_OUT0_LR_B: &str = "kg_net.fc_out.0.linear_real.bias";
/// `KGNet.fc_out.0.linear_imag.weight` (shape `[H, H] = [18, 18]`).
pub const T_FC_OUT0_LI_W: &str = "kg_net.fc_out.0.linear_imag.weight";
/// `KGNet.fc_out.0.linear_imag.bias` (shape `[H] = [18]`).
pub const T_FC_OUT0_LI_B: &str = "kg_net.fc_out.0.linear_imag.bias";
/// `KGNet.fc_out.1.prelu.weight` (shape `[1]`).
pub const T_FC_OUT_PRELU: &str = "kg_net.fc_out.1.prelu.weight";
/// `KGNet.fc_out.2.linear_real.weight` (shape `[L, H] = [4, 18]`).
pub const T_FC_OUT2_LR_W: &str = "kg_net.fc_out.2.linear_real.weight";
/// `KGNet.fc_out.2.linear_real.bias` (shape `[L] = [4]`).
pub const T_FC_OUT2_LR_B: &str = "kg_net.fc_out.2.linear_real.bias";
/// `KGNet.fc_out.2.linear_imag.weight` (shape `[L, H] = [4, 18]`).
pub const T_FC_OUT2_LI_W: &str = "kg_net.fc_out.2.linear_imag.weight";
/// `KGNet.fc_out.2.linear_imag.bias` (shape `[L] = [4]`).
pub const T_FC_OUT2_LI_B: &str = "kg_net.fc_out.2.linear_imag.bias";

// ---- config -------------------------------------------------------------

/// NKF-AEC runtime config (fully-pinned by upstream `nkf.py` today; the
/// struct is `#[non_exhaustive]` so a future variant checkpoint can
/// carry a differently-widthed hparam without breaking downstream
/// callers).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct NkfAecConfig {
    /// Number of Kalman filter taps per bin.
    pub l: usize,
    /// Fully-connected width consumed by the first / middle
    /// `ComplexDense`.
    pub fc_dim: usize,
    /// `ComplexGru` hidden width.
    pub rnn_dim: usize,
    /// STFT FFT size.
    pub n_fft: usize,
    /// STFT hop size.
    pub hop: usize,
    /// STFT window length.
    pub win_length: usize,
    /// PCM sample rate the model was trained at.
    pub sample_rate: u32,
}

impl NkfAecConfig {
    /// The upstream release config (`nkf.py` constants — L=4, fc_dim=18,
    /// rnn_dim=18, n_fft=1024, hop=256, win=1024, 16 kHz).
    pub fn upstream_default() -> Self {
        Self {
            l: L,
            fc_dim: FC_DIM,
            rnn_dim: RNN_DIM,
            n_fft: N_FFT,
            hop: HOP,
            win_length: WIN_LENGTH,
            sample_rate: SAMPLE_RATE,
        }
    }

    /// Number of RFFT bins (`n_fft/2 + 1`).
    #[inline]
    pub fn f_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// KGNet input feature width (`2*L + 1`).
    #[inline]
    pub fn kgnet_in(&self) -> usize {
        2 * self.l + 1
    }

    /// Validates the config loudly (FR-EX-08). Every field must be
    /// non-zero and `win_length <= n_fft`.
    pub fn validate(&self) -> Result<()> {
        for (label, v) in [
            ("l", self.l),
            ("fc_dim", self.fc_dim),
            ("rnn_dim", self.rnn_dim),
            ("n_fft", self.n_fft),
            ("hop", self.hop),
            ("win_length", self.win_length),
            ("sample_rate", self.sample_rate as usize),
        ] {
            if v == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "nkf-aec config: `{label}` must be > 0 (got 0)"
                )));
            }
        }
        if self.win_length > self.n_fft {
            return Err(VokraError::InvalidArgument(format!(
                "nkf-aec config: win_length ({}) > n_fft ({}) — invalid framing",
                self.win_length, self.n_fft
            )));
        }
        Ok(())
    }
}

// ---- weight bundles ------------------------------------------------------

/// One real `Linear` layer's parameters (`weight` row-major `[out, in]`,
/// `bias` `[out]`).
#[derive(Debug, Clone)]
struct LinearWeights {
    weight: Vec<f32>,
    bias: Vec<f32>,
    out_dim: usize,
    in_dim: usize,
}

impl LinearWeights {
    /// `y = weight · x + bias`. `x.len()` must equal `in_dim`; the
    /// caller has already checked.
    #[inline]
    fn forward(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_dim);
        debug_assert_eq!(out.len(), self.out_dim);
        for (o, out_slot) in out.iter_mut().enumerate() {
            let row = &self.weight[o * self.in_dim..(o + 1) * self.in_dim];
            let mut acc = self.bias[o];
            for (i, &xv) in x.iter().enumerate() {
                acc += row[i] * xv;
            }
            *out_slot = acc;
        }
    }
}

/// One real `nn.GRU` layer's parameters (upstream torch layout).
///
/// PyTorch `nn.GRU` stores the three gates (`r`, `z`, `n`) concatenated
/// along dim 0 of `weight_ih`, `weight_hh`, `bias_ih`, `bias_hh` — so
/// `weight_ih_l0.shape = [3*H, D]`, `bias_ih_l0.shape = [3*H]`. This
/// bundle keeps the concatenated layout verbatim and slices at forward
/// time.
#[derive(Debug, Clone)]
struct GruWeights {
    weight_ih: Vec<f32>, // [3*H, D]  row-major
    weight_hh: Vec<f32>, // [3*H, H]  row-major
    bias_ih: Vec<f32>,   // [3*H]
    bias_hh: Vec<f32>,   // [3*H]
    hidden: usize,
    input_size: usize,
}

impl GruWeights {
    /// Single-step forward for one nn.GRU cell (PyTorch semantics):
    /// ```text
    /// r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
    /// z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
    /// n = tanh   (W_in x + b_in + r * (W_hn h + b_hn))
    /// h' = (1 - z) * n + z * h
    /// ```
    /// Writes the new hidden into `h_out`. `x` len == `input_size`,
    /// `h_in` / `h_out` len == `hidden`.
    fn step(&self, x: &[f32], h_in: &[f32], h_out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.input_size);
        debug_assert_eq!(h_in.len(), self.hidden);
        debug_assert_eq!(h_out.len(), self.hidden);
        let h = self.hidden;

        // Slice `[3*H]` bias / `[3*H, D]` weight rows into per-gate views.
        // Layout matches upstream torch: rows 0..H = reset (r),
        // H..2H = update (z), 2H..3H = new (n).
        let (b_ir, rest) = self.bias_ih.split_at(h);
        let (b_iz, b_in) = rest.split_at(h);
        let (b_hr, rest) = self.bias_hh.split_at(h);
        let (b_hz, b_hn) = rest.split_at(h);
        let (w_ir, rest) = self.weight_ih.split_at(h * self.input_size);
        let (w_iz, w_in) = rest.split_at(h * self.input_size);
        let (w_hr, rest) = self.weight_hh.split_at(h * h);
        let (w_hz, w_hn) = rest.split_at(h * h);

        // r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
        // z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
        // pre_n_ih = W_in x + b_in
        // pre_n_hh = W_hn h + b_hn
        // n = tanh(pre_n_ih + r * pre_n_hh)
        // h' = (1 - z) * n + z * h
        for o in 0..h {
            let row_ir = &w_ir[o * self.input_size..(o + 1) * self.input_size];
            let row_iz = &w_iz[o * self.input_size..(o + 1) * self.input_size];
            let row_in = &w_in[o * self.input_size..(o + 1) * self.input_size];
            let row_hr = &w_hr[o * h..(o + 1) * h];
            let row_hz = &w_hz[o * h..(o + 1) * h];
            let row_hn = &w_hn[o * h..(o + 1) * h];

            let mut a_ir = b_ir[o];
            let mut a_iz = b_iz[o];
            let mut a_in = b_in[o];
            let mut a_hr = b_hr[o];
            let mut a_hz = b_hz[o];
            let mut a_hn = b_hn[o];
            for i in 0..self.input_size {
                a_ir += row_ir[i] * x[i];
                a_iz += row_iz[i] * x[i];
                a_in += row_in[i] * x[i];
            }
            for i in 0..h {
                a_hr += row_hr[i] * h_in[i];
                a_hz += row_hz[i] * h_in[i];
                a_hn += row_hn[i] * h_in[i];
            }
            let r = sigmoid(a_ir + a_hr);
            let z = sigmoid(a_iz + a_hz);
            let n = (a_in + r * a_hn).tanh();
            h_out[o] = (1.0 - z) * n + z * h_in[o];
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Full weight bundle bound from a Vokra GGUF.
#[derive(Debug)]
pub struct NkfAecWeights {
    // fc_in (ComplexDense)
    fc_in_lr: LinearWeights,
    fc_in_li: LinearWeights,
    fc_in_prelu: f32, // shared PReLU coefficient (upstream stores as `[1]`)
    // complex_gru (ComplexGRU = 2 real nn.GRU)
    gru_r: GruWeights,
    gru_i: GruWeights,
    // fc_out (ComplexDense → ComplexPReLU → ComplexDense)
    fc_out0_lr: LinearWeights,
    fc_out0_li: LinearWeights,
    fc_out_prelu: f32,
    fc_out2_lr: LinearWeights,
    fc_out2_li: LinearWeights,
}

impl NkfAecWeights {
    /// Binds every one of the 22 KGNet tensors from a Vokra GGUF,
    /// checking each tensor's `dimensions` against the exact row-major
    /// layout upstream emits (openWakeWord dim-order-assertion
    /// precedent).
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] naming the offending tensor on any
    /// missing / wrong-shape / wrong-dtype tensor.
    pub fn from_gguf(gguf: &GgufFile, cfg: &NkfAecConfig) -> Result<Self> {
        cfg.validate()?;
        let l = cfg.l;
        let h = cfg.rnn_dim; // == fc_dim (upstream ties them)
        let ki = cfg.kgnet_in();

        let fc_in_lr = load_linear(gguf, T_FC_IN_LR_W, T_FC_IN_LR_B, h, ki)?;
        let fc_in_li = load_linear(gguf, T_FC_IN_LI_W, T_FC_IN_LI_B, h, ki)?;
        let fc_in_prelu = load_prelu(gguf, T_FC_IN_PRELU)?;

        let gru_r = load_gru(
            gguf,
            T_GRU_R_W_IH,
            T_GRU_R_W_HH,
            T_GRU_R_B_IH,
            T_GRU_R_B_HH,
            h,
            cfg.fc_dim,
        )?;
        let gru_i = load_gru(
            gguf,
            T_GRU_I_W_IH,
            T_GRU_I_W_HH,
            T_GRU_I_B_IH,
            T_GRU_I_B_HH,
            h,
            cfg.fc_dim,
        )?;

        let fc_out0_lr = load_linear(gguf, T_FC_OUT0_LR_W, T_FC_OUT0_LR_B, h, h)?;
        let fc_out0_li = load_linear(gguf, T_FC_OUT0_LI_W, T_FC_OUT0_LI_B, h, h)?;
        let fc_out_prelu = load_prelu(gguf, T_FC_OUT_PRELU)?;
        let fc_out2_lr = load_linear(gguf, T_FC_OUT2_LR_W, T_FC_OUT2_LR_B, l, h)?;
        let fc_out2_li = load_linear(gguf, T_FC_OUT2_LI_W, T_FC_OUT2_LI_B, l, h)?;

        Ok(Self {
            fc_in_lr,
            fc_in_li,
            fc_in_prelu,
            gru_r,
            gru_i,
            fc_out0_lr,
            fc_out0_li,
            fc_out_prelu,
            fc_out2_lr,
            fc_out2_li,
        })
    }

    /// Constructs a valid but all-zero weight bundle for structural
    /// tests. With zero weights every KGNet call emits `kg = 0`, so
    /// `h_posterior == h_prior == 0` for every step — the Kalman filter
    /// is stuck at zero and the residual is `E = Y - 0 = Y = mic`
    /// verbatim. Callers that need a non-trivial forward wire in real
    /// weights (either via [`Self::from_gguf`] or a hand-built bundle
    /// through the `#[cfg(test)]` builder in this module's tests).
    #[cfg(test)]
    pub(crate) fn zeros(cfg: &NkfAecConfig) -> Self {
        let h = cfg.rnn_dim;
        let l = cfg.l;
        let ki = cfg.kgnet_in();
        let lin = |out_dim, in_dim| LinearWeights {
            weight: vec![0.0; out_dim * in_dim],
            bias: vec![0.0; out_dim],
            out_dim,
            in_dim,
        };
        let gru = |input_size| GruWeights {
            weight_ih: vec![0.0; 3 * h * input_size],
            weight_hh: vec![0.0; 3 * h * h],
            bias_ih: vec![0.0; 3 * h],
            bias_hh: vec![0.0; 3 * h],
            hidden: h,
            input_size,
        };
        Self {
            fc_in_lr: lin(h, ki),
            fc_in_li: lin(h, ki),
            fc_in_prelu: 0.0,
            gru_r: gru(cfg.fc_dim),
            gru_i: gru(cfg.fc_dim),
            fc_out0_lr: lin(h, h),
            fc_out0_li: lin(h, h),
            fc_out_prelu: 0.0,
            fc_out2_lr: lin(l, h),
            fc_out2_li: lin(l, h),
        }
    }
}

fn load_linear(
    gguf: &GgufFile,
    weight_name: &str,
    bias_name: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<LinearWeights> {
    let weight = load_f32(gguf, weight_name, out_dim * in_dim)?;
    let bias = load_f32(gguf, bias_name, out_dim)?;
    assert_dims(gguf, weight_name, &[out_dim as u64, in_dim as u64])?;
    assert_dims(gguf, bias_name, &[out_dim as u64])?;
    Ok(LinearWeights {
        weight,
        bias,
        out_dim,
        in_dim,
    })
}

fn load_gru(
    gguf: &GgufFile,
    w_ih: &str,
    w_hh: &str,
    b_ih: &str,
    b_hh: &str,
    hidden: usize,
    input_size: usize,
) -> Result<GruWeights> {
    let weight_ih = load_f32(gguf, w_ih, 3 * hidden * input_size)?;
    let weight_hh = load_f32(gguf, w_hh, 3 * hidden * hidden)?;
    let bias_ih = load_f32(gguf, b_ih, 3 * hidden)?;
    let bias_hh = load_f32(gguf, b_hh, 3 * hidden)?;
    assert_dims(gguf, w_ih, &[(3 * hidden) as u64, input_size as u64])?;
    assert_dims(gguf, w_hh, &[(3 * hidden) as u64, hidden as u64])?;
    assert_dims(gguf, b_ih, &[(3 * hidden) as u64])?;
    assert_dims(gguf, b_hh, &[(3 * hidden) as u64])?;
    Ok(GruWeights {
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        hidden,
        input_size,
    })
}

fn load_prelu(gguf: &GgufFile, name: &str) -> Result<f32> {
    let v = load_f32(gguf, name, 1)?;
    assert_dims(gguf, name, &[1])?;
    Ok(v[0])
}

fn load_f32(gguf: &GgufFile, name: &str, expect: usize) -> Result<Vec<f32>> {
    let v = gguf
        .tensor_f32(name)
        .map_err(|e| VokraError::ModelLoad(format!("nkf-aec: tensor `{name}` load failed: {e}")))?;
    if v.len() != expect {
        return Err(VokraError::ModelLoad(format!(
            "nkf-aec: tensor `{name}` has {} elements, expected {expect}",
            v.len()
        )));
    }
    Ok(v)
}

fn assert_dims(gguf: &GgufFile, name: &str, expected: &[u64]) -> Result<()> {
    let info = gguf.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "nkf-aec: tensor `{name}` info unavailable after load — GGUF invariant broken",
        ))
    })?;
    if info.dimensions.as_slice() != expected {
        return Err(VokraError::ModelLoad(format!(
            "nkf-aec: tensor `{name}` dims {:?}, expected {:?} (upstream nkf.py \
             row-major layout — a Python bridge that silently writes the transpose \
             is a load-time bug, not a runtime one)",
            info.dimensions, expected
        )));
    }
    Ok(())
}

// ---- KGNet forward -------------------------------------------------------

/// Applies upstream `ComplexDense` semantics (`nkf.py:23-33`,
/// `ComplexDense.forward`): **two independent real `nn.Linear`s applied
/// separately to the real and imaginary parts, with NO cross-terms**.
///
/// ```text
/// # nkf.py::ComplexDense.forward, verbatim:
/// return torch.complex(self.linear_real(x.real), self.linear_imag(x.imag))
/// ```
///
/// That is: `y_re = W_r · x_re + b_r` and `y_im = W_i · x_im + b_i`,
/// where `W_r`/`W_i` are the two independent real weight matrices
/// (`linear_real.weight` / `linear_imag.weight`) and `b_r`/`b_i` are
/// their biases. The output real part depends **only** on the input
/// real part; likewise for imaginary. This is the "split-real" complex
/// linear parametrisation — not a true complex-valued multiply
/// (`(W_r + i·W_i)(x_re + i·x_im) = (W_r·x_re − W_i·x_im) + i(W_r·x_im
/// + W_i·x_re)`), which would introduce cross-terms and reuse one set
/// of weights across both output components. Upstream's `ComplexGRU`
/// (`nkf.py:41-58`) DOES use complex-multiply cross-terms when
/// combining the four real GRU outputs into a complex result; see
/// [`complex_gru_step`] for that formula. This function does not.
fn complex_dense_forward(
    lr: &LinearWeights,
    li: &LinearWeights,
    x_re: &[f32],
    x_im: &[f32],
    y_re: &mut [f32],
    y_im: &mut [f32],
) {
    lr.forward(x_re, y_re);
    li.forward(x_im, y_im);
}

/// Applies `ComplexPReLU`: independent real PReLU on the real and
/// imaginary parts — mirror of `nkf.py::ComplexPReLU.forward` which
/// does exactly `complex(prelu(x.real), prelu(x.imag))`. The shared
/// scalar coefficient is applied to negative values as
/// `y = max(0, x) + coeff * min(0, x)`.
fn complex_prelu_apply(coeff: f32, buf: &mut [f32]) {
    for v in buf.iter_mut() {
        if *v < 0.0 {
            *v *= coeff;
        }
    }
}

/// Applies the ComplexGRU step. Consumes one complex input row
/// (`x_re`, `x_im`, both `[fc_dim]`) plus the four hidden states
/// (`h_rr`, `h_ir`, `h_ri`, `h_ii`, each `[rnn_dim]`); writes the
/// updated hidden state IN-PLACE and returns the complex output
/// (`out_re`, `out_im`, both `[rnn_dim]`).
///
/// Upstream `nkf.py::ComplexGRU.forward` runs four independent real
/// `nn.GRU` cell steps:
///
/// ```text
/// Frr, h_rr = gru_r(x.real, h_rr)
/// Fir, h_ir = gru_r(x.imag, h_ir)    # same gru_r weights, different hidden
/// Fri, h_ri = gru_i(x.real, h_ri)
/// Fii, h_ii = gru_i(x.imag, h_ii)    # same gru_i weights, different hidden
/// y = complex(Frr - Fii, Fri + Fir)
/// ```
///
/// The two `_ir` / `_ii` hidden states are independent history tracks
/// for the imag-input passes; they carry across steps just like the
/// `_rr` / `_ri` states.
#[allow(clippy::too_many_arguments)]
fn complex_gru_step(
    gru_r: &GruWeights,
    gru_i: &GruWeights,
    x_re: &[f32],
    x_im: &[f32],
    h_rr: &mut [f32],
    h_ir: &mut [f32],
    h_ri: &mut [f32],
    h_ii: &mut [f32],
    out_re: &mut [f32],
    out_im: &mut [f32],
    scratch: &mut [f32],
) {
    let hh = gru_r.hidden;
    let (h_rr_new, rest) = scratch.split_at_mut(hh);
    let (h_ir_new, rest) = rest.split_at_mut(hh);
    let (h_ri_new, h_ii_new) = rest.split_at_mut(hh);

    gru_r.step(x_re, h_rr, h_rr_new);
    gru_r.step(x_im, h_ir, h_ir_new);
    gru_i.step(x_re, h_ri, h_ri_new);
    gru_i.step(x_im, h_ii, h_ii_new);

    // Frr = h_rr_new; Fir = h_ir_new; Fri = h_ri_new; Fii = h_ii_new
    // y = complex(Frr - Fii, Fri + Fir)
    for k in 0..hh {
        out_re[k] = h_rr_new[k] - h_ii_new[k];
        out_im[k] = h_ri_new[k] + h_ir_new[k];
    }

    h_rr.copy_from_slice(h_rr_new);
    h_ir.copy_from_slice(h_ir_new);
    h_ri.copy_from_slice(h_ri_new);
    h_ii.copy_from_slice(h_ii_new);
}

/// Full `KGNet.forward` for a single bin. Consumes the `2L+1` complex
/// input features and returns `L` complex Kalman gain values in
/// `kg_re` / `kg_im`. Mutates the four `[rnn_dim]` GRU hidden buffers
/// in place. `scratch` must be large enough for every intermediate
/// (`4 * fc_dim + 4 * rnn_dim` values); the session owns it and reuses
/// across bins / steps.
#[allow(clippy::too_many_arguments)]
fn kgnet_step(
    w: &NkfAecWeights,
    feat_re: &[f32],
    feat_im: &[f32],
    h_rr: &mut [f32],
    h_ir: &mut [f32],
    h_ri: &mut [f32],
    h_ii: &mut [f32],
    kg_re: &mut [f32],
    kg_im: &mut [f32],
    scratch: &mut KgnetScratch,
) {
    // fc_in
    complex_dense_forward(
        &w.fc_in_lr,
        &w.fc_in_li,
        feat_re,
        feat_im,
        &mut scratch.fc_in_out_re,
        &mut scratch.fc_in_out_im,
    );
    complex_prelu_apply(w.fc_in_prelu, &mut scratch.fc_in_out_re);
    complex_prelu_apply(w.fc_in_prelu, &mut scratch.fc_in_out_im);

    // complex_gru
    complex_gru_step(
        &w.gru_r,
        &w.gru_i,
        &scratch.fc_in_out_re,
        &scratch.fc_in_out_im,
        h_rr,
        h_ir,
        h_ri,
        h_ii,
        &mut scratch.gru_out_re,
        &mut scratch.gru_out_im,
        &mut scratch.gru_scratch,
    );

    // fc_out.0
    complex_dense_forward(
        &w.fc_out0_lr,
        &w.fc_out0_li,
        &scratch.gru_out_re,
        &scratch.gru_out_im,
        &mut scratch.fc_out0_re,
        &mut scratch.fc_out0_im,
    );
    complex_prelu_apply(w.fc_out_prelu, &mut scratch.fc_out0_re);
    complex_prelu_apply(w.fc_out_prelu, &mut scratch.fc_out0_im);

    // fc_out.2
    complex_dense_forward(
        &w.fc_out2_lr,
        &w.fc_out2_li,
        &scratch.fc_out0_re,
        &scratch.fc_out0_im,
        kg_re,
        kg_im,
    );
}

/// Per-bin scratch buffers reused across steps (allocated once per
/// session, per bin — the recurrence is bin-parallel but scalar per
/// step, so one shared scratch suffices).
pub(crate) struct KgnetScratch {
    fc_in_out_re: Vec<f32>,
    fc_in_out_im: Vec<f32>,
    gru_out_re: Vec<f32>,
    gru_out_im: Vec<f32>,
    fc_out0_re: Vec<f32>,
    fc_out0_im: Vec<f32>,
    // 4 * rnn_dim — one contiguous block for the ComplexGRU step
    // (`h_rr_new`, `h_ir_new`, `h_ri_new`, `h_ii_new`).
    gru_scratch: Vec<f32>,
}

impl KgnetScratch {
    fn new(cfg: &NkfAecConfig) -> Self {
        Self {
            fc_in_out_re: vec![0.0; cfg.fc_dim],
            fc_in_out_im: vec![0.0; cfg.fc_dim],
            gru_out_re: vec![0.0; cfg.rnn_dim],
            gru_out_im: vec![0.0; cfg.rnn_dim],
            fc_out0_re: vec![0.0; cfg.fc_dim],
            fc_out0_im: vec![0.0; cfg.fc_dim],
            gru_scratch: vec![0.0; 4 * cfg.rnn_dim],
        }
    }
}

// ---- session ------------------------------------------------------------

/// NKF-AEC model — immutable shareable weight bundle plus the config it
/// was bound against. Hand out per-open recurrent streams via the
/// [`AecEngine`] trait.
#[derive(Debug)]
pub struct NkfAec {
    cfg: NkfAecConfig,
    weights: Arc<NkfAecWeights>,
}

impl NkfAec {
    /// Binds the model from a parsed GGUF (FR-LD-01). The arch tag is
    /// verified first so a mis-fed GGUF (fsmn-vad / openwakeword / ...)
    /// fails with a clear "wrong arch" message instead of a downstream
    /// "missing tensor".
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        match gguf.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "nkf-aec: GGUF arch is `{other}`, expected `{ARCH}`"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "nkf-aec: GGUF is missing `vokra.model.arch` (converter did not stamp it)"
                        .to_owned(),
                ));
            }
        }
        let cfg = NkfAecConfig::upstream_default();
        let weights = NkfAecWeights::from_gguf(gguf, &cfg)?;
        Ok(Self {
            cfg,
            weights: Arc::new(weights),
        })
    }

    /// Opens and binds the model from a GGUF file on disk.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// Constructs a session bound to caller-supplied weights (test-only
    /// convenience — real deploys go through [`Self::from_gguf`]).
    #[cfg(test)]
    pub(crate) fn from_parts(cfg: NkfAecConfig, weights: NkfAecWeights) -> Self {
        Self {
            cfg,
            weights: Arc::new(weights),
        }
    }

    /// The bound checkpoint's config.
    pub fn config(&self) -> &NkfAecConfig {
        &self.cfg
    }
}

impl AecEngine for NkfAec {
    fn open_stream(&self, sample_rate: u32) -> Result<Box<dyn AecStreamHandle + Send>> {
        if sample_rate != self.cfg.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "nkf-aec: open_stream sample_rate {sample_rate} != model rate {} (upstream \
                 nkf.py was trained at 16 kHz — no silent resample, FR-EX-08)",
                self.cfg.sample_rate
            )));
        }
        Ok(Box::new(NkfAecStream::new(
            self.cfg.clone(),
            Arc::clone(&self.weights),
        )))
    }
}

/// Stateful NKF-AEC stream: consume paired mic + far-end PCM, emit
/// echo-cancelled PCM.
///
/// Every recurrent buffer (per-bin Kalman taps `W`, per-bin GRU hidden
/// vectors `h_rr`/`h_ir`/`h_ri`/`h_ii`, iSTFT overlap-add tail, pending
/// PCM residues) is owned here and hidden from callers (FR-LD-06).
///
/// # Streaming semantics (`center=False`) — deviation from upstream
///
/// Upstream `nkf.py` runs the whole utterance under
/// `torch.stft(center=True)` + one-shot iSTFT — a batch operation. This
/// binder instead advances the STFT **per new frame** under `center=False`
/// semantics (see [`Self::drain`]), which is the streaming-safe shape:
/// each new frame's Kalman step fires **exactly once** (fixing the
/// re-STFT / re-step latent in a block-based drain that recomputes the
/// pending buffer's STFT every push), and no hop-sized push can be
/// dropped no matter how the caller chunks the input.
///
/// The trade-off is a small bit-nonequivalence vs the upstream
/// whole-utterance path — the head/tail of the utterance receive
/// different reflection padding under `center=True` vs `center=False`,
/// and Hann OLA takes `n_fft/hop` frames to reach unity gain (~4 frames
/// = 1024 samples of "warmup" distortion at the head). The upstream
/// parity harness (`crates/vokra-models/tests/parity_nkf_aec.rs`)
/// bounds this at `atol=1e-3`; whole-utterance vs hop-chunked
/// consistency is bit-identical past the warmup (see
/// [`tests::whole_utterance_equals_hop_chunked_stream_within_tolerance`]).
pub struct NkfAecStream {
    pub(crate) cfg: NkfAecConfig,
    pub(crate) weights: Arc<NkfAecWeights>,
    // ---- per-bin recurrent state ----------------------------------------
    //
    // Row-major `[F_bins, L]` complex — one Kalman filter tap vector per
    // frequency bin. Upstream initialises both `h_prior` and
    // `h_posterior` to zero.
    pub(crate) h_prior: Vec<Complex32>,
    pub(crate) h_posterior: Vec<Complex32>,
    // Row-major `[F_bins, rnn_dim]` real — four independent GRU hidden
    // tracks per bin (see `complex_gru_step` docstring).
    pub(crate) h_rr: Vec<f32>,
    pub(crate) h_ir: Vec<f32>,
    pub(crate) h_ri: Vec<f32>,
    pub(crate) h_ii: Vec<f32>,
    // Rolling history of the last `L` far-end STFT frames, row-major
    // `[L, F_bins]` complex. Element `[tap, k]` holds `X[k]` at absolute
    // frame index `frames_processed - L + tap` (older toward index 0,
    // newer toward `L - 1`). At session start every entry is zero — the
    // upstream `torch.cat([zeros, x[:t+1]], dim=-1)` warmup path (first
    // `L - 1` steps see zero-padded taps).
    pub(crate) x_history_re: Vec<f32>,
    pub(crate) x_history_im: Vec<f32>,
    // ---- pending PCM residues ------------------------------------------
    //
    // Callers can push arbitrarily-sized `mic` / `farend` slices; the
    // Kalman step only fires when we have enough samples for the next
    // whole `n_fft` window (per-frame; not per-block). Everything before
    // the next unprocessed frame's absolute start position is trimmed
    // once the frame is consumed (see `drain`).
    pub(crate) pending_mic: Vec<f32>,
    pub(crate) pending_farend: Vec<f32>,
    // Absolute sample index of `pending_mic[0]` / `pending_farend[0]`
    // (samples the caller has pushed but that have not yet been trimmed
    // from the local buffer). Advances monotonically.
    pub(crate) buffer_offset_abs: usize,
    // Total Kalman frames processed since [`Self::reset`] (monotonic;
    // exactly one Kalman step per new STFT frame — this is the
    // invariant that fixes Defect 2). A drain advances this by
    // `available_frames - frames_processed` per call, never
    // re-processing an earlier frame.
    pub(crate) frames_processed: usize,
    // Persistent overlap-add output ring in the absolute time domain.
    // `ola_output[i]` accumulates `E`-frame contributions for absolute
    // sample `ola_start_abs + i`; `ola_wss[i]` is the sum of squared
    // synthesis windows at that position (NOLA normaliser). Both grow
    // as new frames add to the tail and shrink as emitted samples are
    // dropped from the head.
    pub(crate) ola_output: Vec<f32>,
    pub(crate) ola_wss: Vec<f32>,
    // Absolute sample index of `ola_output[0]` / `ola_wss[0]`.
    pub(crate) ola_start_abs: usize,
    // Total samples returned to caller (absolute — monotonic). Emitted
    // samples are `ola_output[emitted_abs - ola_start_abs..
    // committed_abs - ola_start_abs]` divided by `ola_wss` under the
    // NOLA guard.
    pub(crate) emitted_abs: usize,
    // ---- cached DSP plans ----------------------------------------------
    //
    // Reused across every frame (rebuilding per-frame is measurable at
    // hop = 256 / n_fft = 1024).
    pub(crate) real_plan: RealFftPlan,
    // Analysis + synthesis window (Hann periodic, length `n_fft`). Both
    // share the same window sample by sample under upstream's default
    // (`nkf.py`: same `torch.hann_window(1024)` for analysis and
    // synthesis). Kept as separate buffers so a future asymmetric
    // variant does not fork the accumulation loop.
    pub(crate) analysis_window: Vec<f32>,
    pub(crate) synthesis_window: Vec<f32>,
    // The `1 / n_fft` inverse-FFT scaling for the `Backward` convention
    // (matches upstream `torch.istft` with default norm); `RealFftPlan::
    // inverse` includes this factor internally, so we do not multiply
    // by anything at emission time.
    // ---- scratch --------------------------------------------------------
    pub(crate) scratch: KgnetScratch,
    // One frame's `E` buffer (mic − echo_hat), reused across frames.
    pub(crate) frame_e_re: Vec<f32>,
    pub(crate) frame_e_im: Vec<f32>,
}

impl NkfAecStream {
    fn new(cfg: NkfAecConfig, weights: Arc<NkfAecWeights>) -> Self {
        let f = cfg.f_bins();
        let l = cfg.l;
        let rnn = cfg.rnn_dim;
        let scratch = KgnetScratch::new(&cfg);
        let real_plan = RealFftPlan::new(cfg.n_fft);
        let analysis_window = build_window(&cfg);
        let synthesis_window = analysis_window.clone();
        Self {
            h_prior: vec![Complex32::ZERO; f * l],
            h_posterior: vec![Complex32::ZERO; f * l],
            h_rr: vec![0.0; f * rnn],
            h_ir: vec![0.0; f * rnn],
            h_ri: vec![0.0; f * rnn],
            h_ii: vec![0.0; f * rnn],
            x_history_re: vec![0.0; l * f],
            x_history_im: vec![0.0; l * f],
            pending_mic: Vec::new(),
            pending_farend: Vec::new(),
            buffer_offset_abs: 0,
            frames_processed: 0,
            ola_output: Vec::new(),
            ola_wss: Vec::new(),
            ola_start_abs: 0,
            emitted_abs: 0,
            real_plan,
            analysis_window,
            synthesis_window,
            scratch,
            frame_e_re: vec![0.0; f],
            frame_e_im: vec![0.0; f],
            cfg,
            weights,
        }
    }

    /// Runs one KGNet + Kalman update step across every frequency bin
    /// for one **new** frame, given the current mic frame's spectrum
    /// (`y_re`/`y_im`) and the rolling `[L, F_bins]` far-end history in
    /// `self.x_history_*` (oldest at tap 0, newest at tap `L - 1`).
    /// Writes the residual `E = Y − echo_hat_posterior` into
    /// `self.frame_e_re`/`frame_e_im` — the caller then inverts + OLAs.
    ///
    /// # Kalman advancement
    ///
    /// Exactly ONE Kalman step per call — `h_prior`/`h_posterior`/GRU
    /// hidden all advance by one time step for every bin. The drain
    /// invokes this exactly `available_frames − frames_processed` times
    /// per call, never on a re-STFT of a prior frame (fixes Defect 2).
    ///
    /// # Dead-air short-circuit (nkf.py:127-131)
    ///
    /// If `mean(|xt|) < XT_ZERO_THRESHOLD` for a bin, upstream
    /// `continue`s: `echo_hat[k] = 0`, `E[k] = Y[k]`, and **NO** state
    /// mutation (`h_prior` / `h_posterior` / GRU hidden unchanged).
    /// This binder mirrors that verbatim (see the `continue` below).
    // The Kalman recurrence walks parallel `[Complex32; MAX_L]` stack
    // buffers by index (`xt`, `dh`) alongside independent Vec slices
    // (`h_prior`, `h_posterior`, `feat_re`) — an enumerate-based zip
    // would rebuild the parallel indexing into an opaque tuple that
    // obscures the paper's per-tap recurrence steps. Keep the index-
    // based loops so the code reads as the paper does.
    #[allow(clippy::needless_range_loop)]
    fn step_frame(&mut self, y_re: &[f32], y_im: &[f32]) {
        let f_bins = self.cfg.f_bins();
        let l = self.cfg.l;
        let rnn = self.cfg.rnn_dim;
        let kgnet_in = self.cfg.kgnet_in();

        debug_assert_eq!(y_re.len(), f_bins);
        debug_assert_eq!(y_im.len(), f_bins);
        debug_assert_eq!(self.frame_e_re.len(), f_bins);
        debug_assert_eq!(self.frame_e_im.len(), f_bins);
        debug_assert_eq!(self.x_history_re.len(), l * f_bins);
        debug_assert_eq!(self.x_history_im.len(), l * f_bins);

        // Per-frame scratch: the KGNet input feature buffer (2L+1
        // complex) reused for every bin.
        let mut feat_re = vec![0.0f32; kgnet_in];
        let mut feat_im = vec![0.0f32; kgnet_in];
        // Per-bin KGNet output (`kg`, `[L]` complex).
        let mut kg_re = vec![0.0f32; l];
        let mut kg_im = vec![0.0f32; l];

        for k in 0..f_bins {
            // (a) Build `xt[k, :]` — the last `L` X frames at bin `k`,
            // oldest at tap 0, newest at tap `L - 1`. Warmup zero-taps
            // (upstream `torch.cat([zeros, x[:t+1]], dim=-1)`) live in
            // the still-zero rows at the head of `x_history_*` (see
            // [`Self::reset`] / [`Self::new`]).
            let mut xt = [Complex32::ZERO; MAX_L];
            for tap in 0..l {
                let idx = tap * f_bins + k;
                xt[tap] = Complex32::new(self.x_history_re[idx], self.x_history_im[idx]);
            }

            // (b) Dead-air short-circuit: `if xt.abs().mean() < 1e-5:
            // continue` (nkf.py:127). E[k] = Y[k], NO state mutation.
            let mut abs_sum = 0.0f32;
            for tap in 0..l {
                abs_sum += (xt[tap].re * xt[tap].re + xt[tap].im * xt[tap].im).sqrt();
            }
            let abs_mean = abs_sum / (l as f32);
            let y = Complex32::new(y_re[k], y_im[k]);
            if abs_mean < XT_ZERO_THRESHOLD {
                self.frame_e_re[k] = y.re;
                self.frame_e_im[k] = y.im;
                continue;
            }

            // (c) dh = h_posterior - h_prior; then h_prior <- h_posterior.
            let mut dh = [Complex32::ZERO; MAX_L];
            for tap in 0..l {
                let base = k * l + tap;
                dh[tap] = self.h_posterior[base] - self.h_prior[base];
                self.h_prior[base] = self.h_posterior[base];
            }

            // (d) e = Y - <xt, h_prior>
            let mut echo_prior = Complex32::ZERO;
            for tap in 0..l {
                echo_prior = echo_prior + xt[tap] * self.h_prior[k * l + tap];
            }
            let e = y - echo_prior;

            // (e) input_feature = [xt (L complex), e (1 complex), dh (L complex)]
            //     kg = KgNet(input_feature)
            for tap in 0..l {
                feat_re[tap] = xt[tap].re;
                feat_im[tap] = xt[tap].im;
            }
            feat_re[l] = e.re;
            feat_im[l] = e.im;
            for tap in 0..l {
                feat_re[l + 1 + tap] = dh[tap].re;
                feat_im[l + 1 + tap] = dh[tap].im;
            }

            let h_rr_slice = &mut self.h_rr[k * rnn..(k + 1) * rnn];
            let h_ir_slice = &mut self.h_ir[k * rnn..(k + 1) * rnn];
            let h_ri_slice = &mut self.h_ri[k * rnn..(k + 1) * rnn];
            let h_ii_slice = &mut self.h_ii[k * rnn..(k + 1) * rnn];
            kgnet_step(
                &self.weights,
                &feat_re,
                &feat_im,
                h_rr_slice,
                h_ir_slice,
                h_ri_slice,
                h_ii_slice,
                &mut kg_re,
                &mut kg_im,
                &mut self.scratch,
            );

            // (f) h_posterior = h_prior + kg * e; echo_hat = <xt, h_posterior>
            let mut echo_post = Complex32::ZERO;
            for tap in 0..l {
                let base = k * l + tap;
                let kg = Complex32::new(kg_re[tap], kg_im[tap]);
                self.h_posterior[base] = self.h_prior[base] + kg * e;
                echo_post = echo_post + xt[tap] * self.h_posterior[base];
            }

            // Committed cleaned bin = Y - echo_hat_posterior.
            let e_out = y - echo_post;
            self.frame_e_re[k] = e_out.re;
            self.frame_e_im[k] = e_out.im;
        }
    }

    /// Streaming drain — new-frame-only, per-frame OLA.
    ///
    /// # Algorithm (per-frame, `center=False`)
    ///
    /// 1. `available_frames = ((buffer_offset_abs + pending_len) >=
    ///    n_fft) ? ((total - n_fft) / hop + 1) : 0`.
    /// 2. If `available_frames <= frames_processed` → return
    ///    `Vec::new()` (no new work — idempotent under repeated
    ///    zero-length pushes and monotonically non-decreasing).
    /// 3. For each new frame `f in frames_processed..available_frames`:
    ///    (a) window mic + far-end samples at absolute start `f * hop`
    ///    with the cached Hann analysis window;
    ///    (b) `RealFftPlan::forward` on each → `Y[f]` and `X[f]`;
    ///    (c) shift `x_history` left by one row and put `X[f]` at the
    ///    newest tap (`L - 1`);
    ///    (d) `step_frame(Y[f])` — advances Kalman exactly once
    ///    (Defect 2 fix invariant: STFT and Kalman are 1-to-1);
    ///    (e) `RealFftPlan::inverse(E[f])` (includes `1/n_fft`), apply
    ///    synthesis window, overlap-add into `ola_output` at
    ///    `f * hop - ola_start_abs`; accumulate `synth_window²`
    ///    into `ola_wss` at the same offset (persistent OLA ring;
    ///    Defect 1 fix — no per-block clearing).
    /// 4. `committed_abs = available_frames * hop` — every sample
    ///    `< committed_abs` is finalised because no future frame will
    ///    contribute to it under `center=False` (frames start at
    ///    `f * hop`).
    /// 5. Emit `ola_output[emitted_abs - ola_start_abs .. committed_abs
    ///    - ola_start_abs]`, dividing by `ola_wss` under the NOLA guard
    ///    (`> NOLA_EPS`), else 0.0. `emitted_abs = committed_abs`.
    /// 6. Trim `pending_mic` / `pending_farend` so their new front is
    ///    at absolute `frames_processed * hop` (the next unprocessed
    ///    frame's start); advance `buffer_offset_abs` to match.
    /// 7. Drop the finalised head of `ola_output` / `ola_wss` up to
    ///    `committed_abs`; advance `ola_start_abs = committed_abs`.
    ///
    /// # Deviation from upstream `nkf.py` (`center=True`)
    ///
    /// Upstream runs one whole-utterance `torch.stft(center=True)` +
    /// one iSTFT. This binder uses `center=False` streaming so no push
    /// pattern can drop or duplicate a Kalman step (the previous
    /// block-based drain re-STFT'd the pending buffer every push,
    /// running the Kalman `available_frames` times per push — Defect 2).
    /// The head warmup (`n_fft - hop` samples) has partial Hann
    /// coverage under `center=False`; the module-level docstring
    /// documents this trade-off and the parity harness bounds
    /// upstream-vs-Vokra divergence at `atol=1e-3`.
    // The drain walks parallel index-based buffers (mic + far-end
    // pending frames, ola_output + ola_wss, x_history_re + x_history_im
    // per row) that read as the algorithm above — an enumerate-based
    // zip would obscure the frame-by-frame recurrence.
    #[allow(clippy::needless_range_loop)]
    fn drain(&mut self) -> Result<Vec<f32>> {
        let n = self.cfg.n_fft;
        let hop = self.cfg.hop;
        let f_bins = self.cfg.f_bins();
        let l = self.cfg.l;

        // (1) Frames available given the current pending buffer + prior
        // offset. Monotonically non-decreasing across drain calls
        // (frames_processed never rewinds; buffer_offset_abs is
        // strictly the base coordinate of pending_mic).
        let total_abs = self.buffer_offset_abs + self.pending_mic.len();
        let available_frames = if total_abs >= n {
            (total_abs - n) / hop + 1
        } else {
            0
        };

        // (2) Idempotent: no new frames ⇒ nothing to emit. The buffer
        // stays intact for the next push (a caller that swallows the
        // return and retries with more samples must see identical state).
        if available_frames <= self.frames_processed {
            return Ok(Vec::new());
        }

        // Grow OLA ring to accommodate every new frame's tail. The
        // last new frame extends up to `(available_frames - 1) * hop +
        // n_fft`; the ring must reach that absolute sample.
        let required_ola_end_abs = (available_frames - 1) * hop + n;
        let required_local_end = required_ola_end_abs - self.ola_start_abs;
        if self.ola_output.len() < required_local_end {
            self.ola_output.resize(required_local_end, 0.0);
            self.ola_wss.resize(required_local_end, 0.0);
        }

        // Scratch reused across every new frame in this drain call.
        let mut mic_frame = vec![0.0f32; n];
        let mut far_frame = vec![0.0f32; n];

        // (3) Process each new frame — STFT ⇒ Kalman step ⇒ iSTFT ⇒ OLA.
        // Each new frame advances the Kalman state exactly once.
        for f in self.frames_processed..available_frames {
            let start_abs = f * hop;
            let local_start = start_abs - self.buffer_offset_abs;
            debug_assert!(local_start + n <= self.pending_mic.len());
            debug_assert!(local_start + n <= self.pending_farend.len());

            // (3a) Window + (3b) RFFT the mic and far-end frames.
            // Backward normalisation ⇒ forward transform has scale 1.
            for i in 0..n {
                mic_frame[i] = self.pending_mic[local_start + i] * self.analysis_window[i];
                far_frame[i] = self.pending_farend[local_start + i] * self.analysis_window[i];
            }
            let y_spec_bins = self.real_plan.forward(&mic_frame);
            let x_spec_bins = self.real_plan.forward(&far_frame);
            debug_assert_eq!(y_spec_bins.len(), f_bins);
            debug_assert_eq!(x_spec_bins.len(), f_bins);

            // (3c) Shift x_history left by one row (drop oldest), put
            // X[f] at the newest tap (`L - 1`). Rows are `[tap * F_bins,
            // (tap + 1) * F_bins)`; shift is `[tap+1 → tap]`.
            for tap in 0..(l - 1) {
                let dst = tap * f_bins;
                let src = (tap + 1) * f_bins;
                for k in 0..f_bins {
                    self.x_history_re[dst + k] = self.x_history_re[src + k];
                    self.x_history_im[dst + k] = self.x_history_im[src + k];
                }
            }
            let newest = (l - 1) * f_bins;
            for k in 0..f_bins {
                self.x_history_re[newest + k] = x_spec_bins[k].re;
                self.x_history_im[newest + k] = x_spec_bins[k].im;
            }

            // (3d) Split Y bins into planar buffers so `step_frame` can
            // consume a familiar `[f_bins]` slice. Kalman advances once.
            let y_re: Vec<f32> = y_spec_bins.iter().map(|c| c.re).collect();
            let y_im: Vec<f32> = y_spec_bins.iter().map(|c| c.im).collect();
            self.step_frame(&y_re, &y_im);

            // (3e) Inverse-FFT E[f] (includes `1/n_fft` under `Backward`
            // convention — `RealFftPlan::inverse`); apply the synthesis
            // window and overlap-add into the ring at the frame's
            // absolute position.
            let e_bins: Vec<Complex32> = self
                .frame_e_re
                .iter()
                .zip(&self.frame_e_im)
                .map(|(&r, &i)| Complex32::new(r, i))
                .collect();
            let inv_frame = self.real_plan.inverse(&e_bins);
            debug_assert_eq!(inv_frame.len(), n);

            let ola_off = start_abs - self.ola_start_abs;
            for i in 0..n {
                let w = self.synthesis_window[i];
                self.ola_output[ola_off + i] += inv_frame[i] * w;
                self.ola_wss[ola_off + i] += w * w;
            }

            self.frames_processed = f + 1;
        }

        // (4)(5) Commit + emit. Samples `< committed_abs` are complete
        // (no future frame will contribute — `center=False`).
        let committed_abs = available_frames * hop;
        debug_assert!(committed_abs >= self.emitted_abs);
        let out_len = committed_abs - self.emitted_abs;
        let emit_start = self.emitted_abs - self.ola_start_abs;
        let emit_end = committed_abs - self.ola_start_abs;
        let mut out = Vec::with_capacity(out_len);
        for i in emit_start..emit_end {
            let w = self.ola_wss[i];
            if w > NOLA_EPS {
                out.push(self.ola_output[i] / w);
            } else {
                // Warmup / NOLA-violating region — leave 0.0. Under
                // Hann periodic + hop = n_fft/4, only sample index 0 in
                // each fresh frame's contribution has `window[0] = 0`;
                // interior samples reach unity gain past `n_fft - hop`.
                out.push(0.0);
            }
        }
        self.emitted_abs = committed_abs;

        // (6) Trim pending buffers so their new front sits at absolute
        // `frames_processed * hop` (the next unprocessed frame's start).
        // Keeps just enough tail for the next drain to consume the next
        // full window.
        let new_offset = self.frames_processed * hop;
        let drop_local = new_offset - self.buffer_offset_abs;
        if drop_local > 0 {
            self.pending_mic.drain(..drop_local);
            self.pending_farend.drain(..drop_local);
            self.buffer_offset_abs = new_offset;
        }

        // (7) Drop the finalised OLA head. Every sample `< committed_abs`
        // has been emitted; future frames (`f >= available_frames`)
        // start at `available_frames * hop = committed_abs`, so we can
        // never accumulate into a position `< committed_abs` again.
        let ola_drop_local = committed_abs - self.ola_start_abs;
        if ola_drop_local > 0 {
            self.ola_output.drain(..ola_drop_local);
            self.ola_wss.drain(..ola_drop_local);
            self.ola_start_abs = committed_abs;
        }

        Ok(out)
    }
}

/// NOLA (non-zero window-overlap-add) floor. Samples whose accumulated
/// `synth_window²` sum is below this are treated as unreconstructable
/// (division by ~0 would emit NaN/Inf); leave them at zero instead.
/// Matches `vokra_ops::istft::NOLA_EPS` numerically so a whole-utterance
/// vs streaming comparison hits the same boundary.
const NOLA_EPS: f32 = 1e-8;

/// Builds the length-`n_fft` Hann periodic analysis / synthesis window
/// upstream `nkf.py` uses (`torch.hann_window(1024)` — periodic by
/// default). `win_length < n_fft` is centered and zero-padded to
/// `n_fft`, matching [`vokra_ops::stft`] semantics.
fn build_window(cfg: &NkfAecConfig) -> Vec<f32> {
    let w = window(Window::Hann, cfg.win_length, WindowSymmetry::Periodic);
    if cfg.win_length == cfg.n_fft {
        return w;
    }
    let mut full = vec![0.0f32; cfg.n_fft];
    let offset = (cfg.n_fft - cfg.win_length) / 2;
    full[offset..offset + cfg.win_length].copy_from_slice(&w);
    full
}

// The `Normalization::Backward` convention is upstream's default: the
// forward RFFT is unscaled and the inverse carries `1/n_fft`.
// `RealFftPlan::inverse` already applies that factor, so the drain
// does not multiply by anything at emission time. This anchor exists
// so a future variant that trains under `Forward` / `Ortho` normalises
// the branch here explicitly instead of silently mis-scaling.
#[allow(dead_code)]
const UPSTREAM_NORMALIZATION: Normalization = Normalization::Backward;

impl AecStreamHandle for NkfAecStream {
    fn push_paired(&mut self, mic: &[f32], farend: &[f32]) -> Result<Vec<f32>> {
        // Sample alignment is a load-bearing invariant of AEC; any
        // silent drop / repeat is a correctness bug. Refuse loudly
        // (FR-EX-08) before appending — a caller that swallows the
        // error in a retry loop must not grow either pending buffer.
        if mic.len() != farend.len() {
            return Err(VokraError::InvalidArgument(format!(
                "nkf-aec: push_paired mic.len ({}) != farend.len ({}) — the two \
                 streams must be sample-aligned (silent trim / repeat is a \
                 correctness bug, not a convenience — FR-EX-08)",
                mic.len(),
                farend.len()
            )));
        }
        self.pending_mic.extend_from_slice(mic);
        self.pending_farend.extend_from_slice(farend);
        self.drain()
    }

    fn reset(&mut self) {
        for v in &mut self.h_prior {
            *v = Complex32::ZERO;
        }
        for v in &mut self.h_posterior {
            *v = Complex32::ZERO;
        }
        for v in [
            &mut self.h_rr,
            &mut self.h_ir,
            &mut self.h_ri,
            &mut self.h_ii,
        ] {
            for x in v.iter_mut() {
                *x = 0.0;
            }
        }
        for v in [&mut self.x_history_re, &mut self.x_history_im] {
            for x in v.iter_mut() {
                *x = 0.0;
            }
        }
        for v in [&mut self.frame_e_re, &mut self.frame_e_im] {
            for x in v.iter_mut() {
                *x = 0.0;
            }
        }
        self.pending_mic.clear();
        self.pending_farend.clear();
        self.buffer_offset_abs = 0;
        self.frames_processed = 0;
        self.ola_output.clear();
        self.ola_wss.clear();
        self.ola_start_abs = 0;
        self.emitted_abs = 0;
    }
}

/// Upper bound on `L` used to stack `[Complex32; MAX_L]` on-stack in
/// the hot path — upstream pins L=4 and there is no known variant with
/// a wider tap length; this keeps the frame-step allocation-free while
/// leaving headroom for a future 6- or 8-tap release without changing
/// the code shape.
const MAX_L: usize = 8;
