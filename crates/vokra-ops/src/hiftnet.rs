//! HiFTNet vocoder (SoTA plan Phase 1-2).
//!
//! HiFTNet is "Neural Source Filter + ISTFTNet" (upstream CosyVoice's
//! `cosyvoice/hifigan/generator.py:378`, docstring quoted verbatim). The
//! stack is:
//!
//! 1. `F0Predictor` — mel → F0 sequence (this file, Wave 2).
//! 2. `SourceModuleHnNSF` — F0 → source signal (see [`crate::nsf`], Wave 1).
//! 3. `HiFTGenerator` chain — upsample + source fusion + resblocks +
//!    magnitude/phase → iSTFT (Wave 3, forthcoming).
//! 4. Parity harness — synthesized-weight shape/determinism pin (Wave 4).
//!
//! Multiple published models feed the same layer (CosyVoice2 / CosyVoice3
//! / Chatterbox family), so this lives in `vokra-ops` rather than a
//! per-model module.

use vokra_core::{Result, VokraError};

// ---------------------------------------------------------------------------
// F0Predictor (upstream ConvRNNF0Predictor — no RNN despite the name)
// ---------------------------------------------------------------------------

/// Hyperparameters for [`F0Predictor`] — verbatim from upstream
/// `ConvRNNF0Predictor.__init__` (`cosyvoice/hifigan/f0_predictor.py`).
///
/// The upstream defaults `num_class=1, in_channels=80, cond_channels=512`
/// with a **5-layer** Conv1d + ELU stack are the CosyVoice2 shape. The
/// public defaults here mirror them, so a caller wiring in synthesized
/// weights only needs to supply the weight tensors.
#[derive(Debug, Clone, Copy)]
pub struct F0PredictorConfig {
    /// Number of F0 output classes upstream `nn.Linear` produces.
    /// Kept at 1 (regression head) — upstream `torch.abs(...).squeeze(-1)`
    /// requires exactly 1 output channel.
    pub num_class: u32,
    /// Mel channels on the input (upstream `in_channels`, default 80).
    pub in_channels: u32,
    /// Conditioning channels inside the Conv1d stack (upstream
    /// `cond_channels`, default 512).
    pub cond_channels: u32,
    /// Conv kernel size for every layer of the stack (upstream fixed at
    /// `kernel_size=3, padding=1`; kept configurable so a symmetry test
    /// can exercise a smaller kernel, but 3 is the only shape upstream
    /// ships).
    pub kernel_size: u32,
    /// Number of Conv1d + ELU pairs in the stack (upstream ships 5).
    pub num_layers: u32,
}

impl Default for F0PredictorConfig {
    fn default() -> Self {
        Self {
            num_class: 1,
            in_channels: 80,
            cond_channels: 512,
            kernel_size: 3,
            num_layers: 5,
        }
    }
}

/// Learned parameters for [`F0Predictor`]. `conv_weights[l]` is layer `l`'s
/// Conv1d weight in row-major `[out_ch, in_ch_l, kernel]` layout, with
/// `in_ch_0 = in_channels` and `in_ch_l = cond_channels` for `l >= 1`;
/// `conv_biases[l]` is length `cond_channels`. `linear_w` is row-major
/// `[num_class, cond_channels]` (the final `nn.Linear(cond_channels,
/// num_class)`), and `linear_b` is length `num_class`.
#[derive(Debug, Clone)]
pub struct F0PredictorWeights {
    /// One row-major `[cond_channels, prev_channels, kernel]` weight per
    /// layer; `prev_channels` is `in_channels` on layer 0 and
    /// `cond_channels` afterwards.
    pub conv_weights: Vec<Vec<f32>>,
    /// One `[cond_channels]` bias per layer.
    pub conv_biases: Vec<Vec<f32>>,
    /// Row-major `[num_class, cond_channels]` linear head weight.
    pub linear_w: Vec<f32>,
    /// `[num_class]` linear head bias.
    pub linear_b: Vec<f32>,
}

/// The `ConvRNNF0Predictor` port — despite the upstream class name, this
/// is a pure Conv1d + ELU stack followed by a Linear head, no RNN. See
/// [`Self::forward`] for the exact call sequence.
#[derive(Debug, Clone)]
pub struct F0Predictor {
    cfg: F0PredictorConfig,
    weights: F0PredictorWeights,
}

impl F0Predictor {
    /// Build an `F0Predictor` from its config and weights. Fails loudly on
    /// any shape mismatch — the caller supplies weight tensors, and a
    /// silent shape drift would produce garbage F0 at forward time.
    pub fn new(cfg: F0PredictorConfig, weights: F0PredictorWeights) -> Result<Self> {
        let n = cfg.num_layers as usize;
        if n == 0 {
            return Err(VokraError::InvalidArgument(
                "F0Predictor num_layers must be >= 1".to_owned(),
            ));
        }
        if cfg.num_class == 0 {
            return Err(VokraError::InvalidArgument(
                "F0Predictor num_class must be >= 1".to_owned(),
            ));
        }
        if cfg.in_channels == 0 || cfg.cond_channels == 0 || cfg.kernel_size == 0 {
            return Err(VokraError::InvalidArgument(
                "F0Predictor channels / kernel_size must be > 0".to_owned(),
            ));
        }
        if weights.conv_weights.len() != n || weights.conv_biases.len() != n {
            return Err(VokraError::InvalidArgument(format!(
                "F0Predictor: expected {n} conv weights + {n} conv biases, got \
                 {} weights + {} biases",
                weights.conv_weights.len(),
                weights.conv_biases.len(),
            )));
        }
        let out_ch = cfg.cond_channels as usize;
        let k = cfg.kernel_size as usize;
        for (l, w) in weights.conv_weights.iter().enumerate() {
            let in_ch = if l == 0 {
                cfg.in_channels as usize
            } else {
                out_ch
            };
            let expected = out_ch * in_ch * k;
            if w.len() != expected {
                return Err(VokraError::InvalidArgument(format!(
                    "F0Predictor conv layer {l}: expected weight length \
                     {expected} ({out_ch}*{in_ch}*{k}), got {}",
                    w.len(),
                )));
            }
            let b_len = weights.conv_biases[l].len();
            if b_len != out_ch {
                return Err(VokraError::InvalidArgument(format!(
                    "F0Predictor conv layer {l}: expected bias length \
                     {out_ch}, got {b_len}"
                )));
            }
        }
        let expected_lw = (cfg.num_class as usize) * out_ch;
        if weights.linear_w.len() != expected_lw {
            return Err(VokraError::InvalidArgument(format!(
                "F0Predictor linear_w: expected length {expected_lw} \
                 ({}*{out_ch}), got {}",
                cfg.num_class,
                weights.linear_w.len(),
            )));
        }
        if weights.linear_b.len() != cfg.num_class as usize {
            return Err(VokraError::InvalidArgument(format!(
                "F0Predictor linear_b: expected length {}, got {}",
                cfg.num_class,
                weights.linear_b.len(),
            )));
        }
        // The upstream `torch.abs(...).squeeze(-1)` requires `num_class = 1`.
        // We enforce it here rather than in `forward` so a bad config fails
        // at construction time.
        if cfg.num_class != 1 {
            return Err(VokraError::InvalidArgument(format!(
                "F0Predictor num_class must be 1 (upstream `abs().squeeze(-1)` \
                 semantics); got {}",
                cfg.num_class
            )));
        }
        Ok(Self { cfg, weights })
    }

    /// Immutable access to the config this predictor was built with.
    pub fn config(&self) -> &F0PredictorConfig {
        &self.cfg
    }

    /// Forward pass. Reproduces upstream `ConvRNNF0Predictor.forward`
    /// (`cosyvoice/hifigan/f0_predictor.py`):
    ///
    /// ```text
    /// x = self.condnet(x)              # 5 × (Conv1d(k=3, pad=1) + ELU)
    /// x = x.transpose(1, 2)             # [B, cond, T] → [B, T, cond]
    /// return torch.abs(self.classifier(x).squeeze(-1))
    /// ```
    ///
    /// `mel` is row-major `[in_channels, t]`. Returns `[t]` (F0 in the
    /// unit upstream's classifier was trained against — typically Hz-like
    /// after appropriate weight training; the port makes no independent
    /// claim about the unit).
    pub fn forward(&self, mel: &[f32], t: usize) -> Result<Vec<f32>> {
        if t == 0 {
            return Err(VokraError::InvalidArgument(
                "F0Predictor forward: t must be > 0".to_owned(),
            ));
        }
        let in_ch = self.cfg.in_channels as usize;
        let cond_ch = self.cfg.cond_channels as usize;
        let k = self.cfg.kernel_size as usize;
        let padding = k / 2; // upstream `padding = (k-1)/2 + 1` collapses to k/2 for odd k = 3
        if mel.len() != in_ch * t {
            return Err(VokraError::InvalidArgument(format!(
                "F0Predictor forward: mel length {} != in_channels * t = {}",
                mel.len(),
                in_ch * t
            )));
        }

        // ---- 5-layer Conv1d + ELU stack ----------------------------------
        let mut x = mel.to_vec(); // [in_ch, t] to start
        let mut current_in = in_ch;
        for l in 0..self.cfg.num_layers as usize {
            let w = &self.weights.conv_weights[l];
            let b = &self.weights.conv_biases[l];
            x = conv1d_same_padding(&x, current_in, cond_ch, k, padding, t, w, b);
            for v in x.iter_mut() {
                *v = elu(*v);
            }
            current_in = cond_ch;
        }
        // x is [cond_ch, t] row-major.

        // ---- transpose(1, 2) → [t, cond_ch] ------------------------------
        let mut x_t = vec![0.0f32; t * cond_ch];
        for ti in 0..t {
            for c in 0..cond_ch {
                x_t[ti * cond_ch + c] = x[c * t + ti];
            }
        }

        // ---- Linear(cond_ch, 1) → [t, 1] then abs + squeeze(-1) → [t] ----
        let mut f0 = vec![0.0f32; t];
        let lw = &self.weights.linear_w; // [1, cond_ch]
        let lb = self.weights.linear_b[0]; // num_class = 1, verified above
        for (ti, out) in f0.iter_mut().enumerate() {
            let mut acc = lb;
            for c in 0..cond_ch {
                acc += x_t[ti * cond_ch + c] * lw[c];
            }
            *out = acc.abs();
        }
        Ok(f0)
    }
}

// ---------------------------------------------------------------------------
// Primitive helpers (kept local to hiftnet.rs; shared with the coming Wave 3
// generator chain rather than promoted to a public op until a second caller
// materialises).
// ---------------------------------------------------------------------------

/// ELU activation: `x` for `x > 0`, `exp(x) - 1` otherwise. Upstream uses
/// `nn.ELU()` with the default `alpha = 1.0`, so no scale parameter.
#[inline]
fn elu(x: f32) -> f32 {
    if x > 0.0 { x } else { x.exp() - 1.0 }
}

/// Same-padded 1-D convolution.
///
/// - `input`: row-major `[in_ch, t]`
/// - `weight`: row-major `[out_ch, in_ch, kernel]`
/// - `bias`: `[out_ch]`
/// - Output: row-major `[out_ch, t]` (same length as `t` via zero padding)
///
/// This is the naive `O(out_ch × in_ch × kernel × t)` loop — HiFTNet's
/// convs are small (kernel = 3, in/out ≤ 512), so the arithmetic budget
/// per generator step is modest; a matmul refactor is deferred until the
/// full pipeline profiles as a bottleneck.
// The 8-arg surface matches the underlying nn primitive faithfully. A
// struct-of-args would trade one lint for one more layer of indirection
// with no readability gain — the callers spell every arg in the same order
// upstream does.
#[allow(clippy::too_many_arguments)]
fn conv1d_same_padding(
    input: &[f32],
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    padding: usize,
    t: usize,
    weight: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0f32; out_ch * t];
    let t_i = t as isize;
    let pad_i = padding as isize;
    for (oc, &b) in bias.iter().enumerate() {
        let row_offset = oc * t;
        let w_offset = oc * in_ch * kernel;
        for ti in 0..t {
            let mut acc = b;
            for ic in 0..in_ch {
                let x_row = ic * t;
                let w_row = w_offset + ic * kernel;
                for k in 0..kernel {
                    let src = ti as isize + k as isize - pad_i;
                    if src < 0 || src >= t_i {
                        continue;
                    }
                    acc += input[x_row + src as usize] * weight[w_row + k];
                }
            }
            output[row_offset + ti] = acc;
        }
    }
    output
}

// ---------------------------------------------------------------------------
// Snake activation (upstream `cosyvoice.transformer.activation.Snake`)
// ---------------------------------------------------------------------------

/// Per-channel Snake activation with a learnable `alpha` (upstream default
/// `alpha=1.0`, initialised as a length-`in_features` parameter). The
/// closed-form is `snake(x) = x + (1/(alpha + eps)) * sin(x*alpha)^2`,
/// with `eps = 1e-9` to match upstream's `no_div_by_zero`. When
/// `alpha_logscale` is true the stored parameter is exponentiated before
/// use (upstream `alpha = torch.exp(alpha)`).
#[derive(Debug, Clone)]
pub struct Snake {
    alpha: Vec<f32>,
    alpha_logscale: bool,
    no_div_by_zero: f32,
}

impl Snake {
    /// Construct a `Snake` from a per-channel alpha vector.
    ///
    /// `alpha_logscale = true` interprets each entry as `log α`;
    /// `alpha_logscale = false` uses the value directly. Upstream ships
    /// with `alpha_logscale = false` for the ResBlock activations
    /// (`Snake(channels, alpha_logscale=False)` in `HiFTGenerator`).
    pub fn new(alpha: Vec<f32>, alpha_logscale: bool) -> Result<Self> {
        if alpha.is_empty() {
            return Err(VokraError::InvalidArgument(
                "Snake: alpha vector must not be empty".to_owned(),
            ));
        }
        Ok(Self {
            alpha,
            alpha_logscale,
            no_div_by_zero: 1e-9,
        })
    }

    /// Number of channels this activation covers (== `alpha.len()`).
    pub fn channels(&self) -> usize {
        self.alpha.len()
    }

    /// Apply the activation in place to a `[channels, time]` row-major
    /// tensor. Faster than a functional variant because upstream calls it
    /// mid-ResBlock and never keeps the pre-activation around.
    pub fn forward_in_place(&self, x: &mut [f32], channels: usize, time: usize) -> Result<()> {
        if self.alpha.len() != channels {
            return Err(VokraError::InvalidArgument(format!(
                "Snake: alpha length {} != channels {channels}",
                self.alpha.len()
            )));
        }
        if x.len() != channels * time {
            return Err(VokraError::InvalidArgument(format!(
                "Snake forward: input length {} != channels * time = {}",
                x.len(),
                channels * time
            )));
        }
        for (c, &alpha_raw) in self.alpha.iter().enumerate() {
            let alpha = if self.alpha_logscale {
                alpha_raw.exp()
            } else {
                alpha_raw
            };
            let inv_alpha = 1.0 / (alpha + self.no_div_by_zero);
            let row_offset = c * time;
            for slot in x[row_offset..row_offset + time].iter_mut() {
                let s = (*slot * alpha).sin();
                *slot += inv_alpha * s * s;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ConvTranspose1d (upstream `nn.ConvTranspose1d`, weight layout
// [in_ch, out_ch, kernel])
// ---------------------------------------------------------------------------

/// Row-major transposed 1-D convolution (aka fractional-stride convolution).
///
/// - `input`: `[in_ch, t_in]`
/// - `weight`: `[in_ch, out_ch, kernel]` — the PyTorch `nn.ConvTranspose1d`
///   layout (in-channels leading).
/// - `bias`: `[out_ch]` (broadcast along time)
/// - Output: `[out_ch, t_out]` with
///   `t_out = (t_in - 1) * stride + kernel - 2 * padding`.
///
/// Upstream's `HiFTGenerator.ups` all use `padding = (k - u) // 2`
/// (`k = kernel`, `u = stride`), which turns the output length into
/// `t_in * stride`. This helper takes `padding` explicitly so the caller
/// stays in control of the shape and this file makes no independent
/// assumption about it.
///
/// Naive `O(in × out × kernel × t_in)` scatter — every convolution in this
/// vocoder is small (kernel ≤ 16, in/out ≤ 512), so the arithmetic budget
/// per synthesis step is modest; a matmul refactor is deferred.
// Wave 3b (HiFTGenerator chain) is the intended consumer. Wave 3a lands
// this primitive on its own so its arithmetic can be reviewed and
// property-tested in isolation before the chain plugs it in — dropping
// the lint here keeps the intent visible.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn conv_transpose1d(
    input: &[f32],
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    t_in: usize,
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>> {
    if input.len() != in_ch * t_in {
        return Err(VokraError::InvalidArgument(format!(
            "conv_transpose1d: input length {} != in_ch * t_in = {}",
            input.len(),
            in_ch * t_in
        )));
    }
    if weight.len() != in_ch * out_ch * kernel {
        return Err(VokraError::InvalidArgument(format!(
            "conv_transpose1d: weight length {} != in_ch * out_ch * kernel = {}",
            weight.len(),
            in_ch * out_ch * kernel
        )));
    }
    if bias.len() != out_ch {
        return Err(VokraError::InvalidArgument(format!(
            "conv_transpose1d: bias length {} != out_ch = {out_ch}",
            bias.len()
        )));
    }
    if stride == 0 {
        return Err(VokraError::InvalidArgument(
            "conv_transpose1d: stride must be > 0".to_owned(),
        ));
    }
    if t_in == 0 {
        return Err(VokraError::InvalidArgument(
            "conv_transpose1d: t_in must be > 0".to_owned(),
        ));
    }
    // t_out = (t_in - 1) * stride + kernel - 2*padding. Guard against
    // underflow: kernel must dominate 2*padding.
    let core = (t_in - 1) * stride + kernel;
    if 2 * padding > core {
        return Err(VokraError::InvalidArgument(format!(
            "conv_transpose1d: 2*padding ({}) exceeds (t_in-1)*stride + kernel \
             ({core})",
            2 * padding
        )));
    }
    let t_out = core - 2 * padding;

    // Initialise with bias broadcast over time.
    let mut output = vec![0.0f32; out_ch * t_out];
    for (oc, &b) in bias.iter().enumerate() {
        let row = oc * t_out;
        for slot in output[row..row + t_out].iter_mut() {
            *slot = b;
        }
    }

    // Scatter: for each input position, contribute a `kernel`-long stripe
    // to every output channel, offset by `ti * stride - padding + k`.
    for ic in 0..in_ch {
        let in_row = ic * t_in;
        for ti in 0..t_in {
            let x = input[in_row + ti];
            for oc in 0..out_ch {
                let w_off = ic * out_ch * kernel + oc * kernel;
                let out_row = oc * t_out;
                for k in 0..kernel {
                    let dst = (ti * stride + k) as isize - padding as isize;
                    if dst < 0 || dst >= t_out as isize {
                        continue;
                    }
                    output[out_row + dst as usize] += x * weight[w_off + k];
                }
            }
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small synthesized F0Predictor for shape / determinism tests.
    /// Every weight is set to a fixed pattern that keeps outputs bounded
    /// and lets us reason about the sign of the pre-abs classifier value.
    fn small_predictor() -> F0Predictor {
        let cfg = F0PredictorConfig {
            num_class: 1,
            in_channels: 4,
            cond_channels: 8,
            kernel_size: 3,
            num_layers: 3,
        };
        let mut conv_weights = Vec::new();
        let mut conv_biases = Vec::new();
        // Layer 0: [8, 4, 3]
        conv_weights.push(vec![0.01f32; 8 * 4 * 3]);
        conv_biases.push(vec![0.0f32; 8]);
        // Layers 1-2: [8, 8, 3]
        for _ in 1..3 {
            conv_weights.push(vec![0.01f32; 8 * 8 * 3]);
            conv_biases.push(vec![0.0f32; 8]);
        }
        let weights = F0PredictorWeights {
            conv_weights,
            conv_biases,
            linear_w: vec![0.1f32; 8],
            linear_b: vec![0.0f32; 1],
        };
        F0Predictor::new(cfg, weights).expect("small predictor must build")
    }

    #[test]
    fn f0_predictor_forward_output_shape_matches_time_length() {
        let p = small_predictor();
        let t = 16;
        // mel = [4, 16] row-major, filled with a ramp so the output has some
        // structure (not identically zero).
        let mel: Vec<f32> = (0..(4 * t)).map(|i| (i as f32) * 0.01).collect();
        let out = p.forward(&mel, t).unwrap();
        assert_eq!(out.len(), t);
        assert!(out.iter().all(|v| v.is_finite()));
        // The abs head means every output is ≥ 0.
        assert!(out.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn f0_predictor_forward_zero_mel_produces_finite_output() {
        let p = small_predictor();
        let t = 8;
        let mel = vec![0.0f32; 4 * t];
        let out = p.forward(&mel, t).unwrap();
        assert_eq!(out.len(), t);
        assert!(out.iter().all(|v| v.is_finite()));
        // Zero mel + zero bias + linear head with zero bias → pre-abs
        // linear value is 0 (ELU(0) = 0), so |0| = 0.
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn f0_predictor_forward_deterministic_on_same_input() {
        let p = small_predictor();
        let t = 12;
        let mel: Vec<f32> = (0..(4 * t))
            .map(|i| ((i % 7) as f32) * 0.03 - 0.05)
            .collect();
        let a = p.forward(&mel, t).unwrap();
        let b = p.forward(&mel, t).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn f0_predictor_forward_rejects_wrong_input_length() {
        let p = small_predictor();
        let mel = vec![0.0f32; 4 * 16 - 1]; // one short
        let err = p.forward(&mel, 16).unwrap_err();
        assert!(err.to_string().contains("mel length"), "{err}");
    }

    #[test]
    fn f0_predictor_forward_rejects_zero_t() {
        let p = small_predictor();
        let err = p.forward(&[], 0).unwrap_err();
        assert!(err.to_string().contains("t must be > 0"), "{err}");
    }

    #[test]
    fn f0_predictor_new_rejects_wrong_weight_shape() {
        let cfg = F0PredictorConfig {
            num_class: 1,
            in_channels: 4,
            cond_channels: 8,
            kernel_size: 3,
            num_layers: 2,
        };
        // Layer 0 has wrong length (should be 8*4*3 = 96).
        let weights = F0PredictorWeights {
            conv_weights: vec![vec![0.0f32; 10], vec![0.0f32; 8 * 8 * 3]],
            conv_biases: vec![vec![0.0f32; 8], vec![0.0f32; 8]],
            linear_w: vec![0.0f32; 8],
            linear_b: vec![0.0f32; 1],
        };
        let err = F0Predictor::new(cfg, weights).unwrap_err();
        assert!(err.to_string().contains("conv layer 0"), "{err}");
    }

    #[test]
    fn f0_predictor_new_rejects_num_class_not_one() {
        let cfg = F0PredictorConfig {
            num_class: 2, // not 1
            in_channels: 4,
            cond_channels: 8,
            kernel_size: 3,
            num_layers: 1,
        };
        let weights = F0PredictorWeights {
            conv_weights: vec![vec![0.0f32; 8 * 4 * 3]],
            conv_biases: vec![vec![0.0f32; 8]],
            linear_w: vec![0.0f32; 2 * 8],
            linear_b: vec![0.0f32; 2],
        };
        let err = F0Predictor::new(cfg, weights).unwrap_err();
        assert!(err.to_string().contains("num_class must be 1"), "{err}");
    }

    #[test]
    fn f0_predictor_new_rejects_zero_layers() {
        let cfg = F0PredictorConfig {
            num_class: 1,
            in_channels: 4,
            cond_channels: 8,
            kernel_size: 3,
            num_layers: 0,
        };
        let weights = F0PredictorWeights {
            conv_weights: vec![],
            conv_biases: vec![],
            linear_w: vec![0.0f32; 8],
            linear_b: vec![0.0f32; 1],
        };
        let err = F0Predictor::new(cfg, weights).unwrap_err();
        assert!(err.to_string().contains("num_layers must be >= 1"), "{err}");
    }

    #[test]
    fn elu_activation_matches_reference() {
        // Non-negative branch is identity.
        assert_eq!(elu(0.0), 0.0);
        assert_eq!(elu(1.5), 1.5);
        // Negative branch is exp(x) - 1.
        assert!((elu(-1.0) - ((-1.0f32).exp() - 1.0)).abs() < 1e-6);
        // The negative branch is bounded above by 0.
        assert!(elu(-10.0) <= 0.0);
    }

    #[test]
    fn conv1d_same_padding_preserves_length_and_biases_baseline() {
        // in_ch=1, out_ch=1, kernel=3, padding=1, t=5.
        // weight = [1, 1, 1] (sum of neighbours), bias = 2.
        // input = [1, 2, 3, 4, 5].
        // out[i] = input[i-1] + input[i] + input[i+1] + 2 (zero-padded ends).
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![1.0f32; 3];
        let bias = vec![2.0f32];
        let out = conv1d_same_padding(&input, 1, 1, 3, 1, 5, &weight, &bias);
        // Expected: [0+1+2+2, 1+2+3+2, 2+3+4+2, 3+4+5+2, 4+5+0+2]
        //         = [5, 8, 11, 14, 11]
        assert_eq!(out, vec![5.0, 8.0, 11.0, 14.0, 11.0]);
    }

    #[test]
    fn conv1d_same_padding_multi_channel_output_shape() {
        let input = vec![0.5f32; 2 * 4]; // in_ch=2, t=4
        let weight = vec![0.1f32; 3 * 2 * 3]; // [out_ch=3, in_ch=2, k=3]
        let bias = vec![0.0f32; 3];
        let out = conv1d_same_padding(&input, 2, 3, 3, 1, 4, &weight, &bias);
        assert_eq!(out.len(), 3 * 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn snake_zero_input_is_zero_under_default_alpha() {
        // snake(0) = 0 + (1/α) * sin(0)^2 = 0. Deterministic identity at
        // the origin regardless of α (finite α).
        let snake = Snake::new(vec![1.0f32; 4], false).unwrap();
        let mut x = vec![0.0f32; 4 * 8];
        snake.forward_in_place(&mut x, 4, 8).unwrap();
        assert!(x.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn snake_alpha_one_matches_closed_form() {
        // With α = 1, snake(x) = x + sin(x)^2. Verify at a few sample
        // points using f32::sin (the same primitive the port uses, so
        // the check is a self-consistency assertion — a change to the
        // implementation would still trip it).
        let snake = Snake::new(vec![1.0f32], false).unwrap();
        let inputs = [-2.0f32, -0.5, 0.0, 0.5, 1.7];
        for x0 in inputs {
            let mut x = vec![x0];
            snake.forward_in_place(&mut x, 1, 1).unwrap();
            let s = x0.sin();
            let expected = x0 + s * s / (1.0 + 1e-9);
            assert!(
                (x[0] - expected).abs() < 1e-6,
                "snake({x0}) = {} but expected {}",
                x[0],
                expected
            );
        }
    }

    #[test]
    fn snake_alpha_logscale_exponentiates() {
        // Store log(α) = 0 → α_effective = exp(0) = 1. Same as
        // `alpha_logscale=False, alpha=1`.
        let with_log = Snake::new(vec![0.0f32], true).unwrap();
        let without_log = Snake::new(vec![1.0f32], false).unwrap();
        let mut a = vec![0.7f32];
        let mut b = vec![0.7f32];
        with_log.forward_in_place(&mut a, 1, 1).unwrap();
        without_log.forward_in_place(&mut b, 1, 1).unwrap();
        assert!((a[0] - b[0]).abs() < 1e-6, "{} vs {}", a[0], b[0]);
    }

    #[test]
    fn snake_rejects_wrong_alpha_length() {
        let snake = Snake::new(vec![1.0f32; 4], false).unwrap();
        let mut x = vec![0.0f32; 3 * 8]; // 3 channels, alpha has 4
        let err = snake.forward_in_place(&mut x, 3, 8).unwrap_err();
        assert!(err.to_string().contains("alpha length"), "{err}");
    }

    #[test]
    fn snake_rejects_empty_alpha_at_construction() {
        let err = Snake::new(vec![], false).unwrap_err();
        assert!(err.to_string().contains("must not be empty"), "{err}");
    }

    #[test]
    fn snake_rejects_wrong_input_length_at_forward() {
        let snake = Snake::new(vec![1.0f32; 2], false).unwrap();
        let mut x = vec![0.0f32; 2 * 8 - 1]; // one short
        let err = snake.forward_in_place(&mut x, 2, 8).unwrap_err();
        assert!(err.to_string().contains("input length"), "{err}");
    }

    #[test]
    fn conv_transpose1d_length_formula_matches_upstream() {
        // t_in = 4, stride = 2, kernel = 4, padding = 1
        // t_out = (4-1)*2 + 4 - 2*1 = 6 + 4 - 2 = 8
        let input = vec![0.5f32; 4];
        let weight = vec![0.0f32; 4];
        let bias = vec![0.0f32; 1];
        let out = conv_transpose1d(&input, 1, 1, 4, 2, 1, 4, &weight, &bias).unwrap();
        assert_eq!(out.len(), 8, "output length");
    }

    #[test]
    fn conv_transpose1d_upstream_hiftgen_ups_shape_x2() {
        // Upstream `HiFTGenerator.ups[0]`: kernel=16, stride=8, padding=4.
        // t_out = (t_in - 1) * 8 + 16 - 8 = t_in * 8.
        let t_in = 5;
        let input = vec![0.1f32; 3 * t_in]; // in_ch = 3
        let weight = vec![0.01f32; 3 * 2 * 16]; // [in=3, out=2, k=16]
        let bias = vec![0.0f32; 2];
        let out = conv_transpose1d(&input, 3, 2, 16, 8, 4, t_in, &weight, &bias).unwrap();
        // Upstream contract: t_out = t_in * stride = 40 for these knobs.
        assert_eq!(out.len(), 2 * (t_in * 8));
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn conv_transpose1d_zero_input_produces_bias_only_output() {
        // Zero input + non-zero bias → output should equal bias broadcast.
        let input = vec![0.0f32; 2 * 4];
        let weight = vec![0.5f32; 2 * 3 * 5]; // [in=2, out=3, k=5]
        let bias = vec![1.5f32, -0.5, 0.25];
        let stride = 1;
        let padding = 2; // (5 - 1) / 2 = 2 → same-length
        let out = conv_transpose1d(&input, 2, 3, 5, stride, padding, 4, &weight, &bias).unwrap();
        // t_out = (4-1)*1 + 5 - 4 = 4.
        assert_eq!(out.len(), 3 * 4);
        for (oc, &b) in bias.iter().enumerate() {
            for t_idx in 0..4 {
                assert_eq!(out[oc * 4 + t_idx], b, "channel {oc} t={t_idx}");
            }
        }
    }

    #[test]
    fn conv_transpose1d_rejects_shape_mismatches() {
        let input = vec![0.0f32; 2 * 4];
        let weight = vec![0.0f32; 2 * 2 * 3];
        let bias = vec![0.0f32; 2];
        // Wrong weight length triggers a loud error.
        assert!(conv_transpose1d(&input, 2, 2, 3, 1, 1, 4, &weight[..5], &bias).is_err());
        // Wrong bias length triggers a loud error.
        assert!(conv_transpose1d(&input, 2, 2, 3, 1, 1, 4, &weight, &bias[..1]).is_err());
        // stride = 0 triggers a loud error.
        assert!(conv_transpose1d(&input, 2, 2, 3, 0, 1, 4, &weight, &bias).is_err());
        // t_in = 0 triggers a loud error.
        assert!(conv_transpose1d(&[], 2, 2, 3, 1, 1, 0, &weight, &bias).is_err());
    }
}
