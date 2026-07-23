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
}
