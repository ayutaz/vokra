//! Neural-net primitives and layers used by the UTMOS
//! `wav2vec2_regression.v1` stack (M5-15 T15/T16/T17).
//!
//! # Why these live here and not in `vokra-backend-cpu`
//!
//! `vokra-backend-cpu::kernels` is an **op-primitive** crate (GEMM, conv1d,
//! layer-norm, softmax, activations) whose entries are the units the models'
//! Compute seam dispatches per `(backend, op)`. The three additions below are
//! a different shape of thing:
//!
//! - [`group_norm_f32`] is a genuine missing primitive, but it has exactly one
//!   consumer (the UTMOS feature encoder) and no GPU counterpart is planned;
//! - [`grouped_conv1d_f32`] delegates to the shared backend composition of
//!   `kernels::conv1d_f32`; Charsiu's positional convolution now consumes the
//!   same checked implementation;
//! - [`BiLstm`] is a whole recurrent *layer*, not an op — putting a composite
//!   layer into the kernel crate would be its first, and would invite the
//!   Kokoro `BiLstm1d` to be moved there too.
//!
//! ADR `docs/adr/M5-15-utmos.md` §(a) records the decision (M5-15 §G-1's two
//! choices: extract an lstm kernel into `vokra-backend-cpu`, or scalar-loop in
//! `vokra-eval`). The second was taken, with the Kokoro precedent as support:
//! M2 T17-fixup #2 rewrote `BiLstm1d`'s scalar loop into three GEMV forms and
//! **every one regressed parity**, so the pinned scalar loop stayed. `eval` is
//! an offline/CI path with no RTF requirement, so there is nothing to trade
//! away by staying scalar in the recurrent step.
//!
//! **Kokoro is not touched.** The M5-15 red-line forbids moving
//! `vokra_models::kokoro::nn::BiLstm1d` (it is behind the banned
//! `vokra-eval → vokra-models` edge anyway, and relocating it would perturb a
//! parity surface that is pinned by measurement).
//!
//! # Numerical posture
//!
//! Each primitive mirrors the *upstream* evaluation order it is reproducing so
//! the parity band stays a pure float-association band rather than an
//! algorithmic difference:
//!
//! - [`group_norm_f32`] accumulates mean/variance in `f64` and uses the
//!   biased (`1/n`) variance, matching `torch.nn.functional.group_norm`;
//! - [`BiLstm`] projects the whole input sequence with one
//!   `kernels::gemm_f32` (what `torch.nn.LSTM` does too) and keeps the
//!   recurrent step scalar.
//!
//! # No silent fallback (FR-EX-08)
//!
//! Every shape disagreement is a loud [`VokraError::InvalidArgument`] naming
//! the offending extent; nothing is padded, truncated or zero-filled.

use vokra_backend_cpu::kernels;
use vokra_core::{Result, VokraError};

/// Row-wise GroupNorm over a **channel-major** `[channels, len]` buffer.
///
/// Normalizes each of the `groups` contiguous channel blocks over the joint
/// `(channels_per_group × len)` extent, then applies the per-channel affine:
///
/// ```text
/// mean_g = Σ x / n          var_g = Σ (x - mean_g)² / n      (biased, 1/n)
/// y[c, t] = (x[c, t] - mean_g) / sqrt(var_g + eps) * gamma[c] + beta[c]
/// ```
///
/// This is `torch.nn.functional.group_norm`'s definition. The UTMOS feature
/// encoder's only use is `Fp32GroupNorm(512, 512, affine=True)` — i.e.
/// `groups == channels`, one group per channel — which fairseq applies to conv
/// layer 0 alone (`fairseq/models/wav2vec/wav2vec2.py`
/// `ConvFeatureExtractionModel::block`, `is_group_norm = mode == "default" and
/// i == 0`, pinned at `d03f4e77`). The general `groups` form is implemented
/// anyway so the config, not the code, decides.
///
/// Mean and variance accumulate in `f64`: a 512-channel group over a
/// multi-second clip sums tens of thousands of terms, where an `f32`
/// accumulator's drift would show up in the parity band as an artefact of
/// *our* summation rather than of the reference.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] when `channels`/`len`/`groups` are zero,
/// when `channels % groups != 0`, when `input`/`out` are not exactly
/// `channels * len`, when the affine slices are not `channels` long, or when
/// `eps` is not finite and positive.
#[allow(clippy::too_many_arguments)] // norm's intrinsic parameter set
pub fn group_norm_f32(
    input: &[f32],
    out: &mut [f32],
    channels: usize,
    len: usize,
    groups: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Result<()> {
    let fail = |what: String| Err(VokraError::InvalidArgument(format!("group_norm: {what}")));
    if channels == 0 || len == 0 || groups == 0 {
        return fail(format!(
            "channels/len/groups must all be > 0 (got {channels} / {len} / {groups})"
        ));
    }
    if channels % groups != 0 {
        return fail(format!(
            "channels {channels} is not divisible by groups {groups}"
        ));
    }
    let expected = channels * len;
    if input.len() != expected || out.len() != expected {
        return fail(format!(
            "buffer length mismatch: input {} / out {}, expected {expected} (= {channels} × {len})",
            input.len(),
            out.len()
        ));
    }
    if gamma.len() != channels || beta.len() != channels {
        return fail(format!(
            "affine length mismatch: gamma {} / beta {}, expected {channels}",
            gamma.len(),
            beta.len()
        ));
    }
    if !(eps.is_finite() && eps > 0.0) {
        return fail(format!("eps must be finite and > 0, got {eps}"));
    }

    let per_group = channels / groups;
    let n = (per_group * len) as f64;
    for g in 0..groups {
        let c0 = g * per_group;
        let block = &input[c0 * len..(c0 + per_group) * len];
        let mut sum = 0.0f64;
        for &v in block {
            sum += f64::from(v);
        }
        let mean = sum / n;
        let mut var = 0.0f64;
        for &v in block {
            let d = f64::from(v) - mean;
            var += d * d;
        }
        var /= n;
        // `eps` is added inside the square root, as PyTorch does.
        let inv_std = 1.0 / (var + f64::from(eps)).sqrt();
        for c in c0..c0 + per_group {
            let (gam, bet) = (f64::from(gamma[c]), f64::from(beta[c]));
            for t in 0..len {
                let i = c * len + t;
                out[i] = ((f64::from(input[i]) - mean) * inv_std * gam + bet) as f32;
            }
        }
    }
    Ok(())
}

/// Grouped 1-D convolution, composed from `groups` calls to the first-party
/// [`kernels::conv1d_f32`] (M5-15 T16).
///
/// Layout matches `kernels::conv1d_f32`: `input` is `in_ch × in_len`
/// row-major, `weight` is `out_ch × (in_ch / groups) × kernel` row-major
/// (PyTorch's grouped-conv weight layout), optional `bias` has length
/// `out_ch`, and `out` is `out_ch × out_len` with
/// `out_len = (in_len + 2·padding − kernel) / stride + 1`.
///
/// `groups == 1` degenerates to a single [`kernels::conv1d_f32`] call and is
/// **bit-identical** to calling it directly (a property test pins this).
///
/// UTMOS's only use is the wav2vec2 positional convolution
/// (`Conv1d(768, 768, kernel_size=128, groups=16, padding=64)` — `make_conv_pos`
/// at fairseq `d03f4e77`). Its weight-norm parameterization is folded
/// **offline in the converter** (the DAC `out_proj` precedent), so what
/// arrives here is already a plain dense kernel.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on a zero/indivisible `groups`, a weight or
/// bias length that disagrees with the declared extents, or a too-short padded
/// input; plus anything [`kernels::conv1d_f32`] itself rejects.
#[allow(clippy::too_many_arguments)] // convolution's intrinsic parameter set
pub fn grouped_conv1d_f32(
    input: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    out_ch: usize,
    kernel: usize,
    bias: Option<&[f32]>,
    stride: usize,
    padding: usize,
    groups: usize,
    out: &mut [f32],
) -> Result<()> {
    kernels::grouped_conv1d_f32(
        input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, groups, out,
    )
}

/// One direction's `torch.nn.LSTM` parameter set, in PyTorch's storage layout.
///
/// `w_ih` is `[4·hidden, input]`, `w_hh` is `[4·hidden, hidden]`, and both
/// biases are `4·hidden` long. The gate blocks are stacked in PyTorch's
/// **i, f, g, o** order (input, forget, cell, output) — the order the
/// checkpoint ships, so nothing is permuted on load.
#[derive(Debug, Clone)]
pub struct LstmDirection {
    /// Input-to-hidden weight, `[4·hidden, input]` row-major.
    pub w_ih: Vec<f32>,
    /// Hidden-to-hidden weight, `[4·hidden, hidden]` row-major.
    pub w_hh: Vec<f32>,
    /// Input-to-hidden bias, `4·hidden`.
    pub b_ih: Vec<f32>,
    /// Hidden-to-hidden bias, `4·hidden`.
    pub b_hh: Vec<f32>,
}

impl LstmDirection {
    /// Validates the four buffers against `input` / `hidden`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the first buffer whose length
    /// disagrees with the declared extents.
    pub fn validate(&self, input: usize, hidden: usize, which: &str) -> Result<()> {
        let g = 4 * hidden;
        let check = |name: &str, got: usize, want: usize| -> Result<()> {
            if got == want {
                Ok(())
            } else {
                Err(VokraError::InvalidArgument(format!(
                    "bi_lstm: {which}.{name} length {got} != {want}"
                )))
            }
        };
        check("w_ih", self.w_ih.len(), g * input)?;
        check("w_hh", self.w_hh.len(), g * hidden)?;
        check("b_ih", self.b_ih.len(), g)?;
        check("b_hh", self.b_hh.len(), g)
    }
}

/// A single-layer **bidirectional** LSTM, matching
/// `torch.nn.LSTM(input, hidden, num_layers=1, batch_first=True,
/// bidirectional=True)` for one sequence (batch size 1).
///
/// Output is the per-step concatenation `[h_forward | h_backward]`, width
/// `2·hidden` — PyTorch's `output` tensor, which is what UTMOS's
/// `LDConditioner` consumes (`decoder_output, (h, c) = self.decoder_rnn(...)`).
/// The final `(h, c)` are not returned: upstream discards them.
///
/// Initial state is zero, as PyTorch's is when `hx` is omitted.
#[derive(Debug, Clone)]
pub struct BiLstm {
    /// Input width.
    pub input: usize,
    /// Per-direction hidden width (output width is `2 · hidden`).
    pub hidden: usize,
    /// Forward-direction parameters (`*_l0`).
    pub forward: LstmDirection,
    /// Backward-direction parameters (`*_l0_reverse`).
    pub backward: LstmDirection,
}

impl BiLstm {
    /// Validates both directions' parameter shapes.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a zero extent or a mis-sized buffer.
    pub fn validate(&self) -> Result<()> {
        if self.input == 0 || self.hidden == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "bi_lstm: input and hidden must be > 0 (got {} / {})",
                self.input, self.hidden
            )));
        }
        self.forward.validate(self.input, self.hidden, "forward")?;
        self.backward.validate(self.input, self.hidden, "backward")
    }

    /// Runs the layer over a frame-major `[t, input]` sequence, returning
    /// `[t, 2·hidden]`.
    ///
    /// The input-to-hidden term is computed for **all** timesteps with a
    /// single [`kernels::gemm_f32`] (`X @ W_ih^T`) — the same factoring
    /// `torch.nn.LSTM` uses — and only the recurrent term is stepped, scalar,
    /// per timestep (ADR `M5-15-utmos.md` §(a)).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a mis-shaped parameter set, an empty
    /// sequence, or an `x` whose length is not `t · input`.
    pub fn forward(&self, x: &[f32], t: usize) -> Result<Vec<f32>> {
        self.validate()?;
        if t == 0 {
            return Err(VokraError::InvalidArgument(
                "bi_lstm: empty sequence (t = 0) — an empty output is announced, never silently \
                 produced (FR-EX-08)"
                    .to_owned(),
            ));
        }
        if x.len() != t * self.input {
            return Err(VokraError::InvalidArgument(format!(
                "bi_lstm: input length {} != t × input = {t} × {} = {}",
                x.len(),
                self.input,
                t * self.input
            )));
        }
        let h = self.hidden;
        let out_w = 2 * h;
        let mut out = vec![0.0f32; t * out_w];

        for (dir, is_reverse) in [(&self.forward, false), (&self.backward, true)] {
            // Whole-sequence input projection: [t, input] @ [input, 4h].
            let mut gates = vec![0.0f32; t * 4 * h];
            let w_ih_t = transpose(&dir.w_ih, 4 * h, self.input);
            kernels::gemm_f32(
                t,
                4 * h,
                self.input,
                x,
                &w_ih_t,
                Some(&dir.b_ih),
                &mut gates,
            )?;

            let mut h_prev = vec![0.0f32; h];
            let mut c_prev = vec![0.0f32; h];
            let mut rec = vec![0.0f32; 4 * h];
            for step in 0..t {
                let tt = if is_reverse { t - 1 - step } else { step };
                // Recurrent term: W_hh [4h, h] @ h_prev [h] + b_hh. The
                // checkpoint's storage order is already `[4h, h]`, so this is
                // `gemv_f32`'s native layout — no transpose, no copy.
                kernels::gemv_f32(4 * h, h, &dir.w_hh, &h_prev, Some(&dir.b_hh), &mut rec)?;
                let g = &gates[tt * 4 * h..(tt + 1) * 4 * h];
                let base = if is_reverse { h } else { 0 };
                for j in 0..h {
                    let i_g = sigmoid(g[j] + rec[j]);
                    let f_g = sigmoid(g[h + j] + rec[h + j]);
                    let c_g = tanh(g[2 * h + j] + rec[2 * h + j]);
                    let o_g = sigmoid(g[3 * h + j] + rec[3 * h + j]);
                    let c_new = f_g * c_prev[j] + i_g * c_g;
                    let h_new = o_g * tanh(c_new);
                    c_prev[j] = c_new;
                    h_prev[j] = h_new;
                    out[tt * out_w + base + j] = h_new;
                }
            }
        }
        Ok(out)
    }
}

/// `[rows, cols]` row-major → `[cols, rows]` row-major.
fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = m[r * cols + c];
        }
    }
    t
}

/// Logistic sigmoid, computed in `f64` to match the reference's `expf`
/// accuracy on the gate pre-activations (the gates saturate, so an `f32`
/// `exp` there is the largest single contributor to a step-to-step drift).
fn sigmoid(x: f32) -> f32 {
    (1.0 / (1.0 + (-f64::from(x)).exp())) as f32
}

/// Hyperbolic tangent in `f64`, for the same reason as [`sigmoid`].
fn tanh(x: f32) -> f32 {
    f64::from(x).tanh() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- group_norm --------------------------------------------------------

    #[test]
    fn group_norm_one_group_per_channel_matches_hand_computation() {
        // channels = 2, len = 4, groups = 2 → each channel normalized over
        // its own 4 samples (the UTMOS `Fp32GroupNorm(C, C)` shape).
        // ch0 = [1, 2, 3, 4]: mean 2.5, biased var = (2.25+0.25+0.25+2.25)/4
        //                     = 1.25, std = sqrt(1.25 + 0) …
        let input = [1.0f32, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0];
        let gamma = [1.0f32, 2.0];
        let beta = [0.0f32, -1.0];
        let eps = 1e-5f32;
        let mut out = [0.0f32; 8];
        group_norm_f32(&input, &mut out, 2, 4, 2, &gamma, &beta, eps).unwrap();

        let inv = 1.0 / (1.25f64 + f64::from(eps)).sqrt();
        for (i, &x) in [1.0f64, 2.0, 3.0, 4.0].iter().enumerate() {
            let want = ((x - 2.5) * inv) as f32;
            assert!(
                (out[i] - want).abs() < 1e-6,
                "ch0[{i}]: got {}, want {want}",
                out[i]
            );
        }
        // ch1 is constant → variance 0 → every value maps to beta (0 * gamma
        // + beta). This is the branch where a missing eps would divide by 0.
        for (i, &v) in out.iter().enumerate().skip(4) {
            assert!(
                (v - (-1.0)).abs() < 1e-6,
                "ch1[{i}] constant channel must collapse to beta, got {v}"
            );
        }
    }

    #[test]
    fn group_norm_single_group_normalizes_jointly() {
        // groups = 1 pools both channels: mean over all 4 values = 2.5.
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        group_norm_f32(&input, &mut out, 2, 2, 1, &[1.0, 1.0], &[0.0, 0.0], 1e-5).unwrap();
        let inv = 1.0 / (1.25f64 + 1e-5).sqrt();
        for (i, &x) in [1.0f64, 2.0, 3.0, 4.0].iter().enumerate() {
            assert!((out[i] - ((x - 2.5) * inv) as f32).abs() < 1e-6, "i={i}");
        }
        // Output is zero-mean / unit-variance (up to eps) — the definition.
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        assert!(
            mean.abs() < 1e-6,
            "normalized mean should be ~0, got {mean}"
        );
    }

    #[test]
    fn group_norm_rejects_bad_shapes_loudly() {
        let x = [0.0f32; 8];
        let mut o = [0.0f32; 8];
        // channels not divisible by groups.
        assert!(group_norm_f32(&x, &mut o, 4, 2, 3, &[1.0; 4], &[0.0; 4], 1e-5).is_err());
        // buffer length mismatch.
        assert!(group_norm_f32(&x, &mut o, 4, 3, 2, &[1.0; 4], &[0.0; 4], 1e-5).is_err());
        // affine length mismatch.
        assert!(group_norm_f32(&x, &mut o, 4, 2, 2, &[1.0; 3], &[0.0; 4], 1e-5).is_err());
        // zero extents.
        assert!(group_norm_f32(&[], &mut [], 0, 2, 1, &[], &[], 1e-5).is_err());
        // non-positive eps would divide by zero on a constant channel.
        assert!(group_norm_f32(&x, &mut o, 4, 2, 2, &[1.0; 4], &[0.0; 4], 0.0).is_err());
    }

    // ---- grouped_conv1d ----------------------------------------------------

    /// Deterministic pseudo-random fill in `[-1, 1)` — no RNG dep (NFR-DS-02).
    fn ramp(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / (1u32 << 23) as f32 - 1.0
            })
            .collect()
    }

    #[test]
    fn grouped_conv_with_one_group_is_bit_identical_to_plain_conv1d() {
        // The composition oracle: groups = 1 must not perturb a single bit
        // relative to the kernel it is composed from.
        let (in_ch, in_len, out_ch, k, stride, pad) = (3usize, 17usize, 5usize, 4usize, 2usize, 1);
        let x = ramp(in_ch * in_len, 7);
        let w = ramp(out_ch * in_ch * k, 11);
        let b = ramp(out_ch, 13);
        let out_len = (in_len + 2 * pad - k) / stride + 1;

        let mut a = vec![0.0f32; out_ch * out_len];
        kernels::conv1d_f32(
            &x,
            in_ch,
            in_len,
            &w,
            out_ch,
            k,
            Some(&b),
            stride,
            pad,
            &mut a,
        )
        .unwrap();

        let mut g = vec![0.0f32; out_ch * out_len];
        grouped_conv1d_f32(
            &x,
            in_ch,
            in_len,
            &w,
            out_ch,
            k,
            Some(&b),
            stride,
            pad,
            1,
            &mut g,
        )
        .unwrap();

        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "groups=1 must degenerate to conv1d_f32 bit-for-bit"
        );
    }

    #[test]
    fn grouped_conv_channels_only_see_their_own_group() {
        // groups = 2 over 4 in / 4 out channels: zeroing group 0's input must
        // leave group 1's outputs untouched (and vice versa). This is the
        // property that separates a real grouped conv from a dense one.
        let (in_ch, in_len, out_ch, k, groups) = (4usize, 9usize, 4usize, 3usize, 2usize);
        let w = ramp(out_ch * (in_ch / groups) * k, 5);
        let out_len = in_len - k + 1;
        let x = ramp(in_ch * in_len, 3);

        let mut full = vec![0.0f32; out_ch * out_len];
        grouped_conv1d_f32(
            &x, in_ch, in_len, &w, out_ch, k, None, 1, 0, groups, &mut full,
        )
        .unwrap();

        let mut x0 = x.clone();
        x0[..2 * in_len].fill(0.0); // zero group 0's two input channels
        let mut zeroed = vec![0.0f32; out_ch * out_len];
        grouped_conv1d_f32(
            &x0,
            in_ch,
            in_len,
            &w,
            out_ch,
            k,
            None,
            1,
            0,
            groups,
            &mut zeroed,
        )
        .unwrap();

        // Group 1's outputs (rows 2..4) are unchanged …
        assert_eq!(
            full[2 * out_len..],
            zeroed[2 * out_len..],
            "group 1 must not depend on group 0's input"
        );
        // … and group 0's outputs did change (otherwise the test proves
        // nothing about the wiring).
        assert_ne!(
            full[..2 * out_len],
            zeroed[..2 * out_len],
            "group 0's outputs must depend on group 0's input"
        );
    }

    #[test]
    fn grouped_conv_rejects_bad_shapes_loudly() {
        let x = ramp(8, 1);
        let w = ramp(8, 2);
        let mut o = vec![0.0f32; 8];
        // groups = 0.
        assert!(grouped_conv1d_f32(&x, 4, 2, &w, 4, 1, None, 1, 0, 0, &mut o).is_err());
        // in_ch not divisible by groups.
        assert!(grouped_conv1d_f32(&x, 3, 2, &w, 4, 1, None, 1, 0, 2, &mut o).is_err());
        // weight length disagrees with the declared extents.
        assert!(grouped_conv1d_f32(&x, 4, 2, &w[..3], 4, 1, None, 1, 0, 2, &mut o).is_err());
        // padded input shorter than the kernel.
        let mut small = vec![0.0f32; 1];
        assert!(grouped_conv1d_f32(&x[..4], 4, 1, &w, 4, 9, None, 1, 0, 1, &mut small).is_err());
    }

    // ---- BiLstm ------------------------------------------------------------

    fn lstm_dir(input: usize, hidden: usize, seed: u32) -> LstmDirection {
        LstmDirection {
            w_ih: ramp(4 * hidden * input, seed),
            w_hh: ramp(4 * hidden * hidden, seed ^ 0x5555),
            b_ih: ramp(4 * hidden, seed ^ 0xAAAA),
            b_hh: ramp(4 * hidden, seed ^ 0x1357),
        }
    }

    #[test]
    fn bilstm_single_step_matches_the_gate_equations() {
        // t = 1, hidden = 1, input = 1: with zero initial state the recurrent
        // term is just b_hh, so the whole step is hand-computable and pins
        // the i/f/g/o gate ORDER — the one thing a shape test cannot catch.
        let (input, hidden) = (1usize, 1usize);
        let dir = LstmDirection {
            // rows are [i, f, g, o] × hidden
            w_ih: vec![0.5, -0.25, 1.5, 0.75],
            w_hh: vec![0.0, 0.0, 0.0, 0.0],
            b_ih: vec![0.1, 0.2, -0.3, 0.4],
            b_hh: vec![0.0, 0.0, 0.0, 0.0],
        };
        let lstm = BiLstm {
            input,
            hidden,
            forward: dir.clone(),
            backward: dir,
        };
        let x = [2.0f32];
        let out = lstm.forward(&x, 1).unwrap();
        assert_eq!(out.len(), 2, "one step × 2·hidden");

        let sig = |v: f64| 1.0 / (1.0 + (-v).exp());
        let i_g = sig(0.5 * 2.0 + 0.1);
        let f_g = sig(-0.25 * 2.0 + 0.2);
        let c_g = (1.5f64 * 2.0 - 0.3).tanh();
        let o_g = sig(0.75 * 2.0 + 0.4);
        let c = f_g * 0.0 + i_g * c_g;
        let want = (o_g * c.tanh()) as f32;
        assert!(
            (out[0] - want).abs() < 1e-6,
            "forward h: got {}, want {want}",
            out[0]
        );
        // With identical parameters both directions see the same single
        // frame, so the concatenated halves must agree.
        assert!((out[1] - want).abs() < 1e-6, "backward h: got {}", out[1]);
    }

    #[test]
    fn bilstm_backward_direction_reads_the_sequence_in_reverse() {
        // Feed a sequence and its reverse through the same parameters: the
        // backward half of run A at step t must equal the forward half of
        // run B at step (t-1-t). This is the property that catches a
        // direction that silently runs forward.
        let (input, hidden, t) = (3usize, 4usize, 5usize);
        let f = lstm_dir(input, hidden, 21);
        let lstm = BiLstm {
            input,
            hidden,
            forward: f.clone(),
            backward: f,
        };
        let x = ramp(t * input, 33);
        let mut x_rev = vec![0.0f32; x.len()];
        for i in 0..t {
            x_rev[i * input..(i + 1) * input]
                .copy_from_slice(&x[(t - 1 - i) * input..(t - i) * input]);
        }
        let a = lstm.forward(&x, t).unwrap();
        let b = lstm.forward(&x_rev, t).unwrap();
        let w = 2 * hidden;
        for i in 0..t {
            let back_a = &a[i * w + hidden..(i + 1) * w];
            let fwd_b = &b[(t - 1 - i) * w..(t - 1 - i) * w + hidden];
            for (j, (p, q)) in back_a.iter().zip(fwd_b).enumerate() {
                assert!(
                    (p - q).abs() < 1e-5,
                    "t={i} j={j}: backward {p} vs mirrored forward {q}"
                );
            }
        }
    }

    #[test]
    fn bilstm_is_deterministic_and_shaped() {
        let (input, hidden, t) = (6usize, 5usize, 7usize);
        let lstm = BiLstm {
            input,
            hidden,
            forward: lstm_dir(input, hidden, 2),
            backward: lstm_dir(input, hidden, 9),
        };
        let x = ramp(t * input, 44);
        let a = lstm.forward(&x, t).unwrap();
        let b = lstm.forward(&x, t).unwrap();
        assert_eq!(a.len(), t * 2 * hidden);
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "reruns must be bit-identical"
        );
        assert!(a.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn bilstm_rejects_bad_shapes_loudly() {
        let (input, hidden) = (3usize, 2usize);
        let good = lstm_dir(input, hidden, 5);
        let lstm = BiLstm {
            input,
            hidden,
            forward: good.clone(),
            backward: good.clone(),
        };
        // Empty sequence.
        assert!(lstm.forward(&[], 0).is_err());
        // x length disagrees with t.
        assert!(lstm.forward(&ramp(input * 2, 1), 3).is_err());
        // Mis-sized parameter buffer.
        let mut bad = good.clone();
        bad.w_hh.pop();
        let broken = BiLstm {
            input,
            hidden,
            forward: bad,
            backward: good,
        };
        let err = broken.forward(&ramp(input, 1), 1).expect_err("mis-sized");
        assert!(format!("{err}").contains("w_hh"), "must name it: {err}");
    }
}
