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

use vokra_core::ir::graph::{IstftAttrs, StftAttrs};
use vokra_core::{Result, VokraError};

use crate::nsf::{
    NsfEntropy, SineGenConfig, SourceModuleHnNSF, SourceModuleHnNSFConfig, SourceModuleHnNSFWeights,
};
use crate::{Spectrogram, istft, stft};

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
// ResBlock (upstream L41-93, dilated Conv1d pair + Snake + residual)
// ---------------------------------------------------------------------------

/// Hyperparameters for [`ResBlock`] — verbatim from upstream `ResBlock.__init__`.
///
/// Upstream defaults are `channels=512, kernel_size=3, dilations=[1, 3, 5]`.
/// Each `dilations[i]` produces one branch (a `Conv1d(dilation=d) → Snake →
/// Conv1d(dilation=1) → Snake → residual` sub-block); with the default list
/// that is 3 branches.
#[derive(Debug, Clone)]
pub struct ResBlockConfig {
    /// Feature channels — the layer preserves the channel count end-to-end.
    pub channels: u32,
    /// Conv kernel size (`kernel_size` upstream, default 3).
    pub kernel_size: u32,
    /// Per-branch dilation for `convs1` (`convs2` is always dilation=1
    /// upstream). Length = number of branches.
    pub dilations: Vec<u32>,
}

/// Learned parameters for [`ResBlock`] — one weight/bias per Conv1d (2 per
/// dilation) and one Snake `alpha` per activation (2 per dilation).
#[derive(Debug, Clone)]
pub struct ResBlockWeights {
    /// One row-major `[channels, channels, kernel]` weight per branch —
    /// `convs1[i]` uses `dilations[i]` for its stride-1 dilated convolution.
    pub convs1_w: Vec<Vec<f32>>,
    /// One `[channels]` bias per `convs1[i]`.
    pub convs1_b: Vec<Vec<f32>>,
    /// One row-major `[channels, channels, kernel]` weight per branch —
    /// `convs2[i]` is always dilation=1 upstream (`get_padding(k, 1)`).
    pub convs2_w: Vec<Vec<f32>>,
    /// One `[channels]` bias per `convs2[i]`.
    pub convs2_b: Vec<Vec<f32>>,
    /// One `[channels]` Snake alpha per `activations1[i]`.
    pub activations1_alpha: Vec<Vec<f32>>,
    /// One `[channels]` Snake alpha per `activations2[i]`.
    pub activations2_alpha: Vec<Vec<f32>>,
}

/// ResBlock (upstream L41-93). Sequential branches of
/// `Snake → dilated Conv1d → Snake → Conv1d → residual`; the branch
/// results accumulate via the residual connection so a single call
/// mutates its input in place across every branch.
#[derive(Debug, Clone)]
pub struct ResBlock {
    cfg: ResBlockConfig,
    weights: ResBlockWeights,
    activations1: Vec<Snake>,
    activations2: Vec<Snake>,
}

impl ResBlock {
    /// Build a `ResBlock` from its config and weights. Fails loudly on
    /// any shape disagreement — an inconsistent branch count between
    /// `convs1`, `convs2`, and the two activation vectors would silently
    /// truncate the forward loop otherwise.
    pub fn new(cfg: ResBlockConfig, weights: ResBlockWeights) -> Result<Self> {
        let n_branches = cfg.dilations.len();
        if n_branches == 0 {
            return Err(VokraError::InvalidArgument(
                "ResBlock: dilations must not be empty".to_owned(),
            ));
        }
        if cfg.channels == 0 || cfg.kernel_size == 0 {
            return Err(VokraError::InvalidArgument(
                "ResBlock: channels and kernel_size must be > 0".to_owned(),
            ));
        }
        for (name, v) in [
            ("convs1_w", weights.convs1_w.len()),
            ("convs1_b", weights.convs1_b.len()),
            ("convs2_w", weights.convs2_w.len()),
            ("convs2_b", weights.convs2_b.len()),
            ("activations1_alpha", weights.activations1_alpha.len()),
            ("activations2_alpha", weights.activations2_alpha.len()),
        ] {
            if v != n_branches {
                return Err(VokraError::InvalidArgument(format!(
                    "ResBlock: {name} has {v} entries but dilations has {n_branches}"
                )));
            }
        }
        let ch = cfg.channels as usize;
        let k = cfg.kernel_size as usize;
        let expected_w = ch * ch * k;
        for i in 0..n_branches {
            if weights.convs1_w[i].len() != expected_w {
                return Err(VokraError::InvalidArgument(format!(
                    "ResBlock convs1_w[{i}]: expected length {expected_w} \
                     ({ch}*{ch}*{k}), got {}",
                    weights.convs1_w[i].len(),
                )));
            }
            if weights.convs2_w[i].len() != expected_w {
                return Err(VokraError::InvalidArgument(format!(
                    "ResBlock convs2_w[{i}]: expected length {expected_w} \
                     ({ch}*{ch}*{k}), got {}",
                    weights.convs2_w[i].len(),
                )));
            }
            if weights.convs1_b[i].len() != ch {
                return Err(VokraError::InvalidArgument(format!(
                    "ResBlock convs1_b[{i}]: expected length {ch}, got {}",
                    weights.convs1_b[i].len(),
                )));
            }
            if weights.convs2_b[i].len() != ch {
                return Err(VokraError::InvalidArgument(format!(
                    "ResBlock convs2_b[{i}]: expected length {ch}, got {}",
                    weights.convs2_b[i].len(),
                )));
            }
        }
        // Snake activations — upstream `Snake(channels, alpha_logscale=False)`
        // twice per branch. `alpha_logscale=False` is upstream's default for
        // ResBlock activations.
        let mut activations1 = Vec::with_capacity(n_branches);
        let mut activations2 = Vec::with_capacity(n_branches);
        for i in 0..n_branches {
            activations1.push(Snake::new(weights.activations1_alpha[i].clone(), false)?);
            activations2.push(Snake::new(weights.activations2_alpha[i].clone(), false)?);
            if activations1[i].channels() != ch {
                return Err(VokraError::InvalidArgument(format!(
                    "ResBlock activations1_alpha[{i}]: expected {ch} channels, got {}",
                    activations1[i].channels()
                )));
            }
            if activations2[i].channels() != ch {
                return Err(VokraError::InvalidArgument(format!(
                    "ResBlock activations2_alpha[{i}]: expected {ch} channels, got {}",
                    activations2[i].channels()
                )));
            }
        }
        Ok(Self {
            cfg,
            weights,
            activations1,
            activations2,
        })
    }

    /// Immutable access to the [`ResBlockConfig`] this block was built with.
    pub fn config(&self) -> &ResBlockConfig {
        &self.cfg
    }

    /// Forward pass. Reproduces upstream `ResBlock.forward`
    /// (`generator.py:88-93`):
    ///
    /// ```text
    /// for idx in range(len(self.convs1)):
    ///     xt = self.activations1[idx](x)
    ///     xt = self.convs1[idx](xt)
    ///     xt = self.activations2[idx](xt)
    ///     xt = self.convs2[idx](xt)
    ///     x = xt + x
    /// return x
    /// ```
    ///
    /// `x` is a `[channels, t]` row-major buffer; forward mutates it in
    /// place so an outer caller (the HiFTGenerator chain) does not have to
    /// juggle allocations.
    pub fn forward_in_place(&self, x: &mut [f32], t: usize) -> Result<()> {
        let ch = self.cfg.channels as usize;
        let k = self.cfg.kernel_size as usize;
        if x.len() != ch * t {
            return Err(VokraError::InvalidArgument(format!(
                "ResBlock forward: input length {} != channels * t = {}",
                x.len(),
                ch * t
            )));
        }
        for (idx, &dilation) in self.cfg.dilations.iter().enumerate() {
            let d = dilation as usize;
            // xt = activations1[idx](x)
            let mut xt = x.to_vec();
            self.activations1[idx].forward_in_place(&mut xt, ch, t)?;
            // xt = convs1[idx](xt) — dilated same-padding
            let pad1 = get_padding(k, d);
            xt = conv1d_dilated_same_padding(
                &xt,
                ch,
                ch,
                k,
                d,
                pad1,
                t,
                &self.weights.convs1_w[idx],
                &self.weights.convs1_b[idx],
            )?;
            // xt = activations2[idx](xt)
            self.activations2[idx].forward_in_place(&mut xt, ch, t)?;
            // xt = convs2[idx](xt) — dilation=1
            let pad2 = get_padding(k, 1);
            xt = conv1d_dilated_same_padding(
                &xt,
                ch,
                ch,
                k,
                1,
                pad2,
                t,
                &self.weights.convs2_w[idx],
                &self.weights.convs2_b[idx],
            )?;
            // x = xt + x
            for (dst, &delta) in x.iter_mut().zip(xt.iter()) {
                *dst += delta;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Dilated same-padding Conv1d helper (upstream `get_padding(k, d)` mirror)
// ---------------------------------------------------------------------------

/// Upstream `get_padding(kernel_size, dilation)` returns
/// `(kernel_size * dilation - dilation) // 2 = dilation * (kernel_size - 1) / 2`.
/// The formula matches PyTorch's same-length dilated convolution.
#[inline]
fn get_padding(kernel: usize, dilation: usize) -> usize {
    dilation * (kernel - 1) / 2
}

/// Dilated same-padding 1-D convolution.
///
/// Same interface as [`conv1d_same_padding`] plus an explicit `dilation`.
/// A separate helper (rather than a `dilation=1` default on the original)
/// keeps the F0 predictor's inner loop free of the extra `d_i` multiply on
/// paths that will never use it.
#[allow(clippy::too_many_arguments)]
fn conv1d_dilated_same_padding(
    input: &[f32],
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    dilation: usize,
    padding: usize,
    t: usize,
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>> {
    if input.len() != in_ch * t {
        return Err(VokraError::InvalidArgument(format!(
            "conv1d_dilated_same_padding: input length {} != in_ch * t = {}",
            input.len(),
            in_ch * t
        )));
    }
    if weight.len() != out_ch * in_ch * kernel {
        return Err(VokraError::InvalidArgument(format!(
            "conv1d_dilated_same_padding: weight length {} != out_ch * in_ch * kernel = {}",
            weight.len(),
            out_ch * in_ch * kernel
        )));
    }
    if bias.len() != out_ch {
        return Err(VokraError::InvalidArgument(format!(
            "conv1d_dilated_same_padding: bias length {} != out_ch = {out_ch}",
            bias.len()
        )));
    }
    if dilation == 0 || kernel == 0 {
        return Err(VokraError::InvalidArgument(
            "conv1d_dilated_same_padding: dilation and kernel must be > 0".to_owned(),
        ));
    }
    let mut output = vec![0.0f32; out_ch * t];
    let t_i = t as isize;
    let pad_i = padding as isize;
    let d_i = dilation as isize;
    for (oc, &b) in bias.iter().enumerate() {
        let row_offset = oc * t;
        let w_offset = oc * in_ch * kernel;
        for ti in 0..t {
            let mut acc = b;
            for ic in 0..in_ch {
                let x_row = ic * t;
                let w_row = w_offset + ic * kernel;
                for k in 0..kernel {
                    let src = ti as isize + k as isize * d_i - pad_i;
                    if src < 0 || src >= t_i {
                        continue;
                    }
                    acc += input[x_row + src as usize] * weight[w_row + k];
                }
            }
            output[row_offset + ti] = acc;
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// HiFTGenerator (upstream L378-490: NSF + ISTFTNet)
// ---------------------------------------------------------------------------

/// Hyperparameters for [`HiFTGenerator`]. Defaults mirror the upstream
/// CosyVoice HiFTNet __init__ signature (`generator.py:378-395`) so a caller
/// only needs to override the values their voice checkpoint disagrees with.
///
/// Upstream ships CosyVoice2 at `sampling_rate = 22050`, mel `in_channels =
/// 80`, `upsample_rates = [8, 8]`, `istft = {n_fft: 16, hop_len: 4}` which
/// gives `total_upsample_factor = 8 * 8 * 4 = 256`. Every model that feeds
/// HiFTNet uses those knobs today; the vector fields stay `Vec` rather than
/// arrays so a future 24-kHz variant (which upstream flags via
/// `sinegen_type='2'`) can differ without a struct migration.
#[derive(Debug, Clone)]
pub struct HiFTGeneratorConfig {
    /// Mel channels on the input (upstream `in_channels`, default 80).
    pub in_channels: u32,
    /// Base conv channels; every upsample stage `i` uses
    /// `base_channels / 2^i` on its input side and `base_channels / 2^(i+1)`
    /// on the output side (upstream `base_channels`, default 512).
    pub base_channels: u32,
    /// Number of harmonics on the source (upstream `nb_harmonics`, default 8;
    /// forwarded to `SourceModuleHnNSF`'s `harmonic_num`).
    pub nb_harmonics: u32,
    /// Audio sampling rate in Hz (upstream `sampling_rate`, default 22050).
    pub sampling_rate: u32,
    /// SineGen `sine_amp` (upstream `nsf_alpha`, default 0.1).
    pub nsf_alpha: f32,
    /// SineGen `noise_std` (upstream `nsf_sigma`, default 0.003).
    pub nsf_sigma: f32,
    /// SineGen `voiced_threshold` (upstream `nsf_voiced_threshold`, default 10).
    pub nsf_voiced_threshold: f32,
    /// Per-stage stride (upstream `upsample_rates`, default `[8, 8]`).
    pub upsample_rates: Vec<u32>,
    /// Per-stage kernel size (upstream `upsample_kernel_sizes`, default `[16, 16]`).
    /// Must have the same length as `upsample_rates`.
    pub upsample_kernel_sizes: Vec<u32>,
    /// iSTFT window size (upstream `istft_params["n_fft"]`, default 16).
    pub istft_n_fft: u32,
    /// iSTFT hop length (upstream `istft_params["hop_len"]`, default 4).
    pub istft_hop_len: u32,
    /// MRF (multi-receptive-field) branch kernel sizes (upstream
    /// `resblock_kernel_sizes`, default `[3, 7, 11]`).
    pub resblock_kernel_sizes: Vec<u32>,
    /// MRF branch dilation lists (upstream `resblock_dilation_sizes`, default
    /// `[[1,3,5], [1,3,5], [1,3,5]]`). Length must match
    /// `resblock_kernel_sizes`.
    pub resblock_dilation_sizes: Vec<Vec<u32>>,
    /// Source-side ResBlock branch kernels (upstream
    /// `source_resblock_kernel_sizes`, default `[7, 11]`).
    pub source_resblock_kernel_sizes: Vec<u32>,
    /// Source-side ResBlock dilations (upstream
    /// `source_resblock_dilation_sizes`, default `[[1,3,5], [1,3,5]]`).
    pub source_resblock_dilation_sizes: Vec<Vec<u32>>,
    /// Leaky-ReLU negative slope (upstream `lrelu_slope`, default 0.1).
    pub lrelu_slope: f32,
    /// Terminal clamp on the produced waveform (upstream `audio_limit`,
    /// default 0.99).
    pub audio_limit: f32,
}

impl Default for HiFTGeneratorConfig {
    fn default() -> Self {
        Self {
            in_channels: 80,
            base_channels: 512,
            nb_harmonics: 8,
            sampling_rate: 22050,
            nsf_alpha: 0.1,
            nsf_sigma: 0.003,
            nsf_voiced_threshold: 10.0,
            upsample_rates: vec![8, 8],
            upsample_kernel_sizes: vec![16, 16],
            istft_n_fft: 16,
            istft_hop_len: 4,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            source_resblock_kernel_sizes: vec![7, 11],
            source_resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5]],
            lrelu_slope: 0.1,
            audio_limit: 0.99,
        }
    }
}

impl HiFTGeneratorConfig {
    /// Number of upsample stages (== `upsample_rates.len()`).
    pub fn num_upsamples(&self) -> usize {
        self.upsample_rates.len()
    }

    /// Number of MRF kernels per stage (== `resblock_kernel_sizes.len()`).
    /// Every upsample stage runs this many parallel `ResBlock` branches that
    /// are averaged into a single feature map (upstream's `xs / num_kernels`).
    pub fn num_kernels(&self) -> usize {
        self.resblock_kernel_sizes.len()
    }

    /// Total upsample factor applied to the F0 signal before the source
    /// module — `prod(upsample_rates) * istft_hop_len`. With upstream
    /// defaults this is `8 * 8 * 4 = 256`.
    pub fn total_upsample_factor(&self) -> u32 {
        self.upsample_rates.iter().product::<u32>() * self.istft_hop_len
    }

    /// Feature channels the layer produces on stage `i`'s output side —
    /// `base_channels / 2^(i+1)`. Stage 0 halves, stage 1 quarters, and so
    /// on. Upstream ships 2 stages so the final channel count is
    /// `base_channels / 4 = 128`.
    pub fn output_channels_at(&self, stage: usize) -> u32 {
        self.base_channels >> (stage as u32 + 1)
    }
}

/// Learned parameters for [`HiFTGenerator`].
///
/// Every conv weight is row-major:
/// - `conv_pre_w`: `[base_channels, in_channels, 7]`
/// - `ups_w[i]`: `[in_ch_i, out_ch_i, upsample_kernel_sizes[i]]` (PyTorch
///   `nn.ConvTranspose1d` layout with in-channels leading — same as
///   [`conv_transpose1d`] takes).
/// - `source_downs_w[i]`: `[out_ch_i, istft_n_fft + 2, kernel_i]` (regular
///   Conv1d). `kernel_i` and stride are chosen by upstream's decision:
///   `u == 1` gives `k = 1, stride = 1`; otherwise `k = 2u, stride = u,
///   padding = u/2`.
/// - `conv_post_w`: `[istft_n_fft + 2, output_channels_at(num_ups - 1), 7]`
///
/// `f0_predictor_weights` is the standalone F0 predictor (Wave 2).
/// `m_source_linear_w` / `m_source_linear_b` is upstream
/// `SourceModuleHnNSF.l_linear` (Linear(nb_harmonics + 1, 1)).
///
/// `resblock_weights` has length `num_upsamples * num_kernels` — laid out
/// row-major (upstream `resblocks[i * num_kernels + j]`).
/// `source_resblock_weights` has length `num_upsamples`.
#[derive(Debug, Clone)]
pub struct HiFTGeneratorWeights {
    /// Row-major `[base_channels, in_channels, 7]` — the initial mel
    /// projection.
    pub conv_pre_w: Vec<f32>,
    /// `[base_channels]` bias for `conv_pre`.
    pub conv_pre_b: Vec<f32>,
    /// Per-stage upsample ConvTranspose1d weights, row-major
    /// `[in_ch_i, out_ch_i, k_i]`. Length must equal `num_upsamples`.
    pub ups_w: Vec<Vec<f32>>,
    /// Per-stage ups bias, `[out_ch_i]`.
    pub ups_b: Vec<Vec<f32>>,
    /// Per-stage source-down Conv1d weight, `[out_ch_i, n_fft+2, k]`.
    pub source_downs_w: Vec<Vec<f32>>,
    /// Per-stage source-down Conv1d bias, `[out_ch_i]`.
    pub source_downs_b: Vec<Vec<f32>>,
    /// Per-stage source ResBlock weights.
    pub source_resblock_weights: Vec<ResBlockWeights>,
    /// Row-major `num_upsamples * num_kernels` ResBlock weights.
    pub resblock_weights: Vec<ResBlockWeights>,
    /// Row-major `[n_fft+2, output_channels_at(num_ups - 1), 7]` post-conv.
    pub conv_post_w: Vec<f32>,
    /// `[n_fft+2]` post-conv bias.
    pub conv_post_b: Vec<f32>,
    /// Linear head for the source module: `[nb_harmonics + 1]`.
    pub m_source_linear_w: Vec<f32>,
    /// Scalar bias for the source module linear head.
    pub m_source_linear_b: f32,
    /// Weights for the standalone F0 predictor (Wave 2).
    pub f0_predictor_weights: F0PredictorWeights,
}

/// HiFTNet generator — the full "Neural Source Filter + ISTFTNet" stack.
/// See [`Self::forward`] for the top-level call sequence and [`Self::decode`]
/// for the fusion/upsample chain (upstream `generator.py:467-506`).
#[derive(Debug, Clone)]
pub struct HiFTGenerator {
    cfg: HiFTGeneratorConfig,
    f0_predictor: F0Predictor,
    m_source: SourceModuleHnNSF,
    conv_pre_w: Vec<f32>,
    conv_pre_b: Vec<f32>,
    ups_w: Vec<Vec<f32>>,
    ups_b: Vec<Vec<f32>>,
    source_downs_w: Vec<Vec<f32>>,
    source_downs_b: Vec<Vec<f32>>,
    source_downs_kernel: Vec<u32>,
    source_downs_stride: Vec<u32>,
    source_downs_padding: Vec<u32>,
    source_resblocks: Vec<ResBlock>,
    resblocks: Vec<ResBlock>,
    conv_post_w: Vec<f32>,
    conv_post_b: Vec<f32>,
}

impl HiFTGenerator {
    /// Build a `HiFTGenerator` from its config + weights bundle. Every
    /// shape is checked upfront so a mismatch surfaces at build time rather
    /// than mid-forward. F0Predictor and SourceModuleHnNSF ownership pass
    /// into this struct — after construction only the generator is exposed.
    pub fn new(cfg: HiFTGeneratorConfig, weights: HiFTGeneratorWeights) -> Result<Self> {
        // ---- Config-shape invariants -----------------------------------
        let n_ups = cfg.num_upsamples();
        let n_kernels = cfg.num_kernels();
        if n_ups == 0 {
            return Err(VokraError::InvalidArgument(
                "HiFTGenerator: upsample_rates must not be empty".to_owned(),
            ));
        }
        if cfg.upsample_kernel_sizes.len() != n_ups {
            return Err(VokraError::InvalidArgument(format!(
                "HiFTGenerator: upsample_kernel_sizes length {} != \
                 upsample_rates length {n_ups}",
                cfg.upsample_kernel_sizes.len()
            )));
        }
        if n_kernels == 0 {
            return Err(VokraError::InvalidArgument(
                "HiFTGenerator: resblock_kernel_sizes must not be empty".to_owned(),
            ));
        }
        if cfg.resblock_dilation_sizes.len() != n_kernels {
            return Err(VokraError::InvalidArgument(format!(
                "HiFTGenerator: resblock_dilation_sizes length {} != \
                 resblock_kernel_sizes length {n_kernels}",
                cfg.resblock_dilation_sizes.len()
            )));
        }
        if cfg.source_resblock_kernel_sizes.len() != n_ups {
            return Err(VokraError::InvalidArgument(format!(
                "HiFTGenerator: source_resblock_kernel_sizes length {} != \
                 num_upsamples {n_ups}",
                cfg.source_resblock_kernel_sizes.len()
            )));
        }
        if cfg.source_resblock_dilation_sizes.len() != n_ups {
            return Err(VokraError::InvalidArgument(format!(
                "HiFTGenerator: source_resblock_dilation_sizes length {} != \
                 num_upsamples {n_ups}",
                cfg.source_resblock_dilation_sizes.len()
            )));
        }
        if cfg.istft_n_fft == 0 || cfg.istft_hop_len == 0 {
            return Err(VokraError::InvalidArgument(
                "HiFTGenerator: istft_n_fft and istft_hop_len must be > 0".to_owned(),
            ));
        }

        // ---- Sub-modules -----------------------------------------------
        let f0_predictor = F0Predictor::new(
            F0PredictorConfig {
                num_class: 1,
                in_channels: cfg.in_channels,
                cond_channels: cfg.base_channels,
                kernel_size: 3,
                num_layers: 5,
            },
            weights.f0_predictor_weights,
        )?;

        let m_source = SourceModuleHnNSF::new(
            SourceModuleHnNSFConfig {
                sine_gen: SineGenConfig {
                    samp_rate: cfg.sampling_rate,
                    harmonic_num: cfg.nb_harmonics,
                    sine_amp: cfg.nsf_alpha,
                    noise_std: cfg.nsf_sigma,
                    voiced_threshold: cfg.nsf_voiced_threshold,
                },
            },
            SourceModuleHnNSFWeights {
                linear_w: weights.m_source_linear_w,
                linear_b: weights.m_source_linear_b,
            },
        )?;

        // ---- conv_pre --------------------------------------------------
        let bc = cfg.base_channels as usize;
        let inc = cfg.in_channels as usize;
        let expected_conv_pre_w = bc * inc * 7;
        if weights.conv_pre_w.len() != expected_conv_pre_w {
            return Err(VokraError::InvalidArgument(format!(
                "HiFTGenerator conv_pre_w: expected length {expected_conv_pre_w} \
                 ({bc}*{inc}*7), got {}",
                weights.conv_pre_w.len()
            )));
        }
        if weights.conv_pre_b.len() != bc {
            return Err(VokraError::InvalidArgument(format!(
                "HiFTGenerator conv_pre_b: expected length {bc}, got {}",
                weights.conv_pre_b.len()
            )));
        }

        // ---- ups + shape derivations -----------------------------------
        if weights.ups_w.len() != n_ups || weights.ups_b.len() != n_ups {
            return Err(VokraError::InvalidArgument(format!(
                "HiFTGenerator ups: expected {n_ups} weight and bias sets, \
                 got {} weights / {} biases",
                weights.ups_w.len(),
                weights.ups_b.len()
            )));
        }
        for i in 0..n_ups {
            let in_ch = (cfg.base_channels >> (i as u32)) as usize;
            let out_ch = (cfg.base_channels >> (i as u32 + 1)) as usize;
            let k = cfg.upsample_kernel_sizes[i] as usize;
            let expected = in_ch * out_ch * k;
            if weights.ups_w[i].len() != expected {
                return Err(VokraError::InvalidArgument(format!(
                    "HiFTGenerator ups_w[{i}]: expected length {expected} \
                     ({in_ch}*{out_ch}*{k}), got {}",
                    weights.ups_w[i].len()
                )));
            }
            if weights.ups_b[i].len() != out_ch {
                return Err(VokraError::InvalidArgument(format!(
                    "HiFTGenerator ups_b[{i}]: expected length {out_ch}, got {}",
                    weights.ups_b[i].len()
                )));
            }
            let stride = cfg.upsample_rates[i] as usize;
            if k < stride {
                return Err(VokraError::InvalidArgument(format!(
                    "HiFTGenerator ups[{i}]: kernel {k} < stride {stride} \
                     (upstream `padding = (k-u)//2` requires k >= u)"
                )));
            }
        }

        // ---- source_downs (upstream `downsample_rates` + `downsample_cum_rates`)
        // downsample_rates = [1] + upsample_rates[::-1][:-1]
        // → e.g. upsample [8, 8] gives [1, 8]
        // downsample_cum_rates reversed gives the per-stage `u`.
        let mut downsample_rates: Vec<u32> = Vec::with_capacity(n_ups);
        downsample_rates.push(1);
        for i in (0..n_ups - 1).rev() {
            downsample_rates.push(cfg.upsample_rates[i]);
        }
        let mut downsample_cum: Vec<u32> = Vec::with_capacity(n_ups);
        let mut acc: u32 = 1;
        for &r in &downsample_rates {
            acc = acc.saturating_mul(r);
            downsample_cum.push(acc);
        }
        let downsample_us: Vec<u32> = downsample_cum.iter().rev().copied().collect();

        let (mut source_downs_kernel, mut source_downs_stride, mut source_downs_padding) = (
            Vec::with_capacity(n_ups),
            Vec::with_capacity(n_ups),
            Vec::with_capacity(n_ups),
        );
        if weights.source_downs_w.len() != n_ups || weights.source_downs_b.len() != n_ups {
            return Err(VokraError::InvalidArgument(format!(
                "HiFTGenerator source_downs: expected {n_ups} weight+bias, \
                 got {}+{}",
                weights.source_downs_w.len(),
                weights.source_downs_b.len()
            )));
        }
        let n_fft_plus_2 = cfg.istft_n_fft as usize + 2;
        for (i, &u) in downsample_us.iter().enumerate() {
            let out_ch = cfg.output_channels_at(i) as usize;
            let (k, stride, padding) = if u == 1 {
                (1u32, 1u32, 0u32)
            } else {
                (u * 2, u, u / 2)
            };
            source_downs_kernel.push(k);
            source_downs_stride.push(stride);
            source_downs_padding.push(padding);
            let expected = out_ch * n_fft_plus_2 * (k as usize);
            if weights.source_downs_w[i].len() != expected {
                return Err(VokraError::InvalidArgument(format!(
                    "HiFTGenerator source_downs_w[{i}]: expected length \
                     {expected} ({out_ch}*{n_fft_plus_2}*{k}), got {}",
                    weights.source_downs_w[i].len()
                )));
            }
            if weights.source_downs_b[i].len() != out_ch {
                return Err(VokraError::InvalidArgument(format!(
                    "HiFTGenerator source_downs_b[{i}]: expected length \
                     {out_ch}, got {}",
                    weights.source_downs_b[i].len()
                )));
            }
        }

        // ---- source_resblocks -----------------------------------------
        if weights.source_resblock_weights.len() != n_ups {
            return Err(VokraError::InvalidArgument(format!(
                "HiFTGenerator source_resblock_weights: expected {n_ups}, got {}",
                weights.source_resblock_weights.len()
            )));
        }
        let mut source_resblocks = Vec::with_capacity(n_ups);
        for (i, rbw) in weights.source_resblock_weights.into_iter().enumerate() {
            let ch = cfg.output_channels_at(i);
            let cfg_i = ResBlockConfig {
                channels: ch,
                kernel_size: cfg.source_resblock_kernel_sizes[i],
                dilations: cfg.source_resblock_dilation_sizes[i].clone(),
            };
            source_resblocks.push(ResBlock::new(cfg_i, rbw)?);
        }

        // ---- resblocks -------------------------------------------------
        let n_rbs = n_ups * n_kernels;
        if weights.resblock_weights.len() != n_rbs {
            return Err(VokraError::InvalidArgument(format!(
                "HiFTGenerator resblock_weights: expected {n_rbs} \
                 (num_ups * num_kernels = {n_ups} * {n_kernels}), got {}",
                weights.resblock_weights.len()
            )));
        }
        let mut resblocks = Vec::with_capacity(n_rbs);
        for (idx, rbw) in weights.resblock_weights.into_iter().enumerate() {
            let i = idx / n_kernels;
            let j = idx % n_kernels;
            let ch = cfg.output_channels_at(i);
            let cfg_ij = ResBlockConfig {
                channels: ch,
                kernel_size: cfg.resblock_kernel_sizes[j],
                dilations: cfg.resblock_dilation_sizes[j].clone(),
            };
            resblocks.push(ResBlock::new(cfg_ij, rbw)?);
        }

        // ---- conv_post -------------------------------------------------
        let final_ch = cfg.output_channels_at(n_ups - 1) as usize;
        let expected_conv_post_w = n_fft_plus_2 * final_ch * 7;
        if weights.conv_post_w.len() != expected_conv_post_w {
            return Err(VokraError::InvalidArgument(format!(
                "HiFTGenerator conv_post_w: expected length {expected_conv_post_w} \
                 ({n_fft_plus_2}*{final_ch}*7), got {}",
                weights.conv_post_w.len()
            )));
        }
        if weights.conv_post_b.len() != n_fft_plus_2 {
            return Err(VokraError::InvalidArgument(format!(
                "HiFTGenerator conv_post_b: expected length {n_fft_plus_2}, got {}",
                weights.conv_post_b.len()
            )));
        }

        Ok(Self {
            cfg,
            f0_predictor,
            m_source,
            conv_pre_w: weights.conv_pre_w,
            conv_pre_b: weights.conv_pre_b,
            ups_w: weights.ups_w,
            ups_b: weights.ups_b,
            source_downs_w: weights.source_downs_w,
            source_downs_b: weights.source_downs_b,
            source_downs_kernel,
            source_downs_stride,
            source_downs_padding,
            source_resblocks,
            resblocks,
            conv_post_w: weights.conv_post_w,
            conv_post_b: weights.conv_post_b,
        })
    }

    /// Immutable access to the config this generator was built with.
    pub fn config(&self) -> &HiFTGeneratorConfig {
        &self.cfg
    }

    /// Forward pass. Reproduces upstream `HiFTGenerator.forward`
    /// (`generator.py:497-506`):
    ///
    /// ```text
    /// speech_feat = batch['speech_feat'].transpose(1, 2).to(device)
    /// f0 = self.f0_predictor(speech_feat)
    /// s = self.f0_upsamp(f0[:, None]).transpose(1, 2)   # bs, n, t
    /// s, _, _ = self.m_source(s)
    /// s = s.transpose(1, 2)
    /// generated_speech = self.decode(x=speech_feat, s=s)
    /// ```
    ///
    /// `mel` is row-major `[in_channels, t_mel]`. Returns the reconstructed
    /// waveform as a `Vec<f32>` of length `(t_current - 1) * istft_hop_len`,
    /// where `t_current = t_mel * prod(upsample_rates) + 1` (the final
    /// `nn.ReflectionPad1d((1, 0))` contributes the trailing `+ 1`). Under
    /// upstream defaults `(prod(upsample_rates), istft_hop_len) = (64, 4)`,
    /// so the audio length equals `t_source = t_mel * total_upsample_factor()`.
    ///
    /// The source module is driven with [`NsfEntropy::Deterministic`] —
    /// upstream sets the noise draw's amplitude to `sine_amp / 3` but the
    /// port only offers deterministic entropy for now, so the noise branch
    /// stays at exactly zero rather than being re-derived from a stream.
    /// Upstream also returns `(generated_speech, f0)`; the Vokra port
    /// returns only the audio — a caller who needs the F0 sequence can call
    /// [`Self::f0_predictor_forward`] on the same mel input.
    pub fn forward(&self, mel: &[f32], t_mel: usize) -> Result<Vec<f32>> {
        let in_ch = self.cfg.in_channels as usize;
        if t_mel == 0 {
            return Err(VokraError::InvalidArgument(
                "HiFTGenerator forward: t_mel must be > 0".to_owned(),
            ));
        }
        if mel.len() != in_ch * t_mel {
            return Err(VokraError::InvalidArgument(format!(
                "HiFTGenerator forward: mel length {} != in_channels * t_mel = {}",
                mel.len(),
                in_ch * t_mel
            )));
        }

        // f0 = self.f0_predictor(speech_feat)      → [t_mel]
        let f0 = self.f0_predictor.forward(mel, t_mel)?;

        // s = self.f0_upsamp(f0[:, None]).transpose(1, 2)   → [1, t_source] flat = [t_source]
        let factor = self.cfg.total_upsample_factor() as usize;
        let s_upsampled = upsample_nearest_row_major(&f0, 1, t_mel, factor);
        let t_source = t_mel * factor;

        // s, _, _ = self.m_source(s) — [t_source]
        let src_out = self
            .m_source
            .forward(&s_upsampled, NsfEntropy::Deterministic)?;
        if src_out.sine_merge.len() != t_source {
            return Err(VokraError::InvalidArgument(format!(
                "HiFTGenerator forward: source module returned {} samples, expected {t_source}",
                src_out.sine_merge.len()
            )));
        }

        // generated_speech = self.decode(x=speech_feat, s=s)
        self.decode(mel, &src_out.sine_merge, t_mel, t_source)
    }

    /// Convenience: run only the F0 predictor on `mel`. Kept as a thin
    /// wrapper so a caller who wants both the audio (via [`Self::forward`])
    /// and the F0 sequence (upstream returns both from `forward`) does not
    /// have to hold a separate F0 predictor handle.
    pub fn f0_predictor_forward(&self, mel: &[f32], t_mel: usize) -> Result<Vec<f32>> {
        self.f0_predictor.forward(mel, t_mel)
    }

    /// Decode step — upstream `HiFTGenerator.decode` (`generator.py:467-490`):
    ///
    /// ```text
    /// s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
    /// s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)   # [B, 2F, T]
    ///
    /// x = self.conv_pre(x)
    /// for i in range(self.num_upsamples):
    ///     x = F.leaky_relu(x, self.lrelu_slope)
    ///     x = self.ups[i](x)
    ///     if i == self.num_upsamples - 1: x = self.reflection_pad(x)
    ///
    ///     si = self.source_downs[i](s_stft)
    ///     si = self.source_resblocks[i](si)
    ///     x = x + si
    ///
    ///     xs = None
    ///     for j in range(self.num_kernels):
    ///         xs = self.resblocks[i * self.num_kernels + j](x) if xs is None \
    ///              else xs + self.resblocks[i * self.num_kernels + j](x)
    ///     x = xs / self.num_kernels
    ///
    /// x = F.leaky_relu(x)          # default negative_slope = 0.01
    /// x = self.conv_post(x)
    /// magnitude = torch.exp(x[:, :F, :])
    /// phase    = torch.sin(x[:, F:, :])
    /// x = self._istft(magnitude, phase)     # clips magnitude at 1e2 first
    /// x = torch.clamp(x, -self.audio_limit, self.audio_limit)
    /// ```
    ///
    /// `x` is row-major `[in_channels, t_mel]` (the mel front end);
    /// `s` is `[t_source]` (the upsampled source signal). `t_source` is
    /// carried in explicitly so we fail loudly on any caller mistake
    /// instead of silently trusting `s.len()`.
    fn decode(&self, x: &[f32], s: &[f32], t_mel: usize, t_source: usize) -> Result<Vec<f32>> {
        let in_ch = self.cfg.in_channels as usize;
        let base_ch = self.cfg.base_channels as usize;
        let n_fft = self.cfg.istft_n_fft as usize;
        let hop_len = self.cfg.istft_hop_len as usize;
        let f_bins = n_fft / 2 + 1; // half-spectrum bin count
        let two_f = n_fft + 2; // stacked real + imag row count
        let num_ups = self.cfg.num_upsamples();
        let num_kernels = self.cfg.num_kernels();
        let lrelu_slope = self.cfg.lrelu_slope;

        if x.len() != in_ch * t_mel {
            return Err(VokraError::InvalidArgument(format!(
                "HiFTGenerator decode: x length {} != in_channels * t_mel = {}",
                x.len(),
                in_ch * t_mel
            )));
        }
        if s.len() != t_source {
            return Err(VokraError::InvalidArgument(format!(
                "HiFTGenerator decode: s length {} != t_source = {t_source}",
                s.len()
            )));
        }

        // ---- s_stft = concat([Re, Im], dim=1) : [n_fft+2, t_stft] ---------
        //
        // Upstream `_stft` calls `torch.stft(x, n_fft, hop_len, win_length =
        // n_fft, window = hann_periodic, return_complex=True)` — `center`
        // defaults to `True`, `normalized` to `False` (== backward norm).
        // Vokra `StftAttrs::new(n_fft, hop)` picks the same knobs, so the
        // transform is bit-for-bit equivalent to the upstream call.
        //
        // Vokra `Spectrogram` is row-major `[frames, bins]`; upstream stacks
        // the half-spectrum along the channel (row) axis and keeps time on
        // the column axis. Transpose so rows are `(Re bins, Im bins)` and
        // cols are frames.
        let stft_attrs = StftAttrs::new(n_fft, hop_len);
        let spec = stft(s, &stft_attrs)?;
        let t_stft = spec.frames;
        if spec.bins != f_bins {
            return Err(VokraError::InvalidArgument(format!(
                "HiFTGenerator decode: STFT produced {} bins, expected n_fft/2+1 = {f_bins}",
                spec.bins
            )));
        }
        let mut s_stft = vec![0.0f32; two_f * t_stft];
        for f in 0..f_bins {
            for tt in 0..t_stft {
                s_stft[f * t_stft + tt] = spec.re[tt * f_bins + f];
                s_stft[(f_bins + f) * t_stft + tt] = spec.im[tt * f_bins + f];
            }
        }

        // ---- x = conv_pre(x)  → [base_ch, t_mel] --------------------------
        //
        // Upstream `Conv1d(in_channels, base_channels, 7, 1, padding=3)` is
        // a same-length convolution — kernel = 7, stride = 1, padding = 3.
        let mut x_current = conv1d_same_padding(
            x,
            in_ch,
            base_ch,
            7,
            3,
            t_mel,
            &self.conv_pre_w,
            &self.conv_pre_b,
        );
        let mut t_current = t_mel;

        for i in 0..num_ups {
            // x = F.leaky_relu(x, self.lrelu_slope)
            for v in x_current.iter_mut() {
                *v = leaky_relu(*v, lrelu_slope);
            }

            // x = self.ups[i](x)
            //
            // Upstream `ConvTranspose1d(base_ch/2^i, base_ch/2^(i+1), k=u_k[i],
            // stride=u[i], padding=(k-u)//2)` produces
            // `t_out = (t_in - 1) * stride + k - 2 * padding = t_in * stride`
            // for the fixed `padding = (k - u) // 2` upstream ships. The
            // helper is faithful to that formula but takes `padding`
            // explicitly, so we derive it here rather than folding it in.
            let stride = self.cfg.upsample_rates[i] as usize;
            let kernel = self.cfg.upsample_kernel_sizes[i] as usize;
            let padding = (kernel - stride) / 2;
            let in_stage = base_ch >> i;
            let out_stage = base_ch >> (i + 1);
            x_current = conv_transpose1d(
                &x_current,
                in_stage,
                out_stage,
                kernel,
                stride,
                padding,
                t_current,
                &self.ups_w[i],
                &self.ups_b[i],
            )?;
            t_current *= stride;

            // Terminal reflection pad — upstream `nn.ReflectionPad1d((1, 0))`
            // only fires on the last stage (`if i == self.num_upsamples - 1`).
            if i == num_ups - 1 {
                x_current = reflection_pad_1d_left(&x_current, out_stage, t_current, 1)?;
                t_current += 1;
            }

            // ---- fusion: si = source_downs[i](s_stft) → source_resblocks[i]
            let k_src = self.source_downs_kernel[i] as usize;
            let stride_src = self.source_downs_stride[i] as usize;
            let padding_src = self.source_downs_padding[i] as usize;
            let mut si = conv1d_strided_no_dilation(
                &s_stft,
                two_f,
                out_stage,
                k_src,
                stride_src,
                padding_src,
                t_stft,
                &self.source_downs_w[i],
                &self.source_downs_b[i],
            )?;
            let t_si = si.len() / out_stage;
            if t_si != t_current {
                // Upstream's downsample_rates / downsample_cum_rates choice
                // guarantees `t_si == t_current` at every stage — failing
                // here means the config's upsample_rates and istft params
                // are inconsistent, and we surface it loudly rather than
                // silently truncating one side (FR-EX-08).
                return Err(VokraError::InvalidArgument(format!(
                    "HiFTGenerator decode: source_downs[{i}] produced t_si = {t_si} but \
                     stage t_current = {t_current} (upstream contract mismatch — check \
                     upsample_rates × istft_hop_len against t_mel)"
                )));
            }
            self.source_resblocks[i].forward_in_place(&mut si, t_si)?;

            // x = x + si  (both [out_stage, t_current])
            if x_current.len() != si.len() {
                return Err(VokraError::InvalidArgument(format!(
                    "HiFTGenerator decode: fusion length mismatch — x has {} samples, si has {}",
                    x_current.len(),
                    si.len()
                )));
            }
            for (dst, &delta) in x_current.iter_mut().zip(si.iter()) {
                *dst += delta;
            }

            // ---- MRF: xs = sum_j resblocks[i*num_kernels + j](x); x = xs / num_kernels
            let mut xs_accum = vec![0.0f32; out_stage * t_current];
            for j in 0..num_kernels {
                let mut xt = x_current.clone();
                self.resblocks[i * num_kernels + j].forward_in_place(&mut xt, t_current)?;
                for (acc, &v) in xs_accum.iter_mut().zip(xt.iter()) {
                    *acc += v;
                }
            }
            let inv_n = 1.0 / (num_kernels as f32);
            for v in xs_accum.iter_mut() {
                *v *= inv_n;
            }
            x_current = xs_accum;
        }

        // ---- final F.leaky_relu (default negative_slope = 0.01) ----------
        for v in x_current.iter_mut() {
            *v = leaky_relu(*v, 0.01);
        }

        // ---- conv_post: [final_ch, t_current] → [n_fft+2, t_current] -----
        let final_ch = base_ch >> num_ups;
        let x_post = conv1d_same_padding(
            &x_current,
            final_ch,
            two_f,
            7,
            3,
            t_current,
            &self.conv_post_w,
            &self.conv_post_b,
        );

        // ---- magnitude / phase / complex reassembly ----------------------
        //
        // Upstream (`generator.py:485-489` + `_istft` L459-465):
        //
        //   magnitude = exp(x[:, :F, :])
        //   phase     = sin(x[:, F:, :])
        //   magnitude = clip(magnitude, max=1e2)
        //   real = magnitude * cos(phase)
        //   img  = magnitude * sin(phase)
        //
        // The clip is applied AFTER exp (so a large exponent that overflows
        // to +inf in f32 still collapses to 1e2 by `min(inf, 100) = 100`
        // under IEEE-754 rules — we preserve that exact order).
        //
        // Layout: `x_post` is row-major `[n_fft+2, t_current]` with real
        // bins in rows [0, F) and imag bins in rows [F, 2F). Vokra
        // `Spectrogram` is row-major `[frames, bins]`, so we transpose from
        // `[F, t_current]` to `[t_current, F]` while filling in real/imag.
        let mut re_out = vec![0.0f32; t_current * f_bins];
        let mut im_out = vec![0.0f32; t_current * f_bins];
        for f in 0..f_bins {
            let row_mag = f * t_current;
            let row_pha = (f_bins + f) * t_current;
            for tt in 0..t_current {
                let magnitude = x_post[row_mag + tt].exp().min(1e2);
                let phase = x_post[row_pha + tt].sin();
                let dst = tt * f_bins + f;
                re_out[dst] = magnitude * phase.cos();
                im_out[dst] = magnitude * phase.sin();
            }
        }
        let spec_out = Spectrogram {
            frames: t_current,
            bins: f_bins,
            re: re_out,
            im: im_out,
        };

        // ---- iSTFT + audio_limit clamp -----------------------------------
        //
        // `IstftAttrs::new(n_fft, hop_len)` inherits the same window / norm
        // / center defaults as the forward analysis, so this pairs
        // bit-exactly with the earlier `stft` call above (M0-04-T12 COLA
        // path).
        let istft_attrs = IstftAttrs::new(n_fft, hop_len);
        let mut audio = istft(&spec_out, &istft_attrs)?;
        let limit = self.cfg.audio_limit;
        for v in audio.iter_mut() {
            *v = v.clamp(-limit, limit);
        }
        Ok(audio)
    }
}

// ---------------------------------------------------------------------------
// Private helpers for HiFTGenerator's forward + decode chain
// ---------------------------------------------------------------------------

/// Leaky ReLU: `x` if `x > 0`, otherwise `x * slope`.
///
/// Upstream reaches for this twice — once in each upsample stage with the
/// configured `lrelu_slope` (default 0.1), and once immediately before
/// `conv_post` where the call is written `F.leaky_relu(x)` (PyTorch's
/// functional default is `negative_slope = 0.01`).
#[inline]
fn leaky_relu(x: f32, slope: f32) -> f32 {
    if x > 0.0 { x } else { x * slope }
}

/// Nearest-neighbour upsampling by an integer `factor` along the time axis
/// of a row-major `[ch, t_in]` tensor.
///
/// Upstream `nn.Upsample(scale_factor=factor, mode='nearest')` on a 1-D
/// signal `[a, b, c]` with `factor = 3` produces `[a, a, a, b, b, b, c, c, c]`.
/// The port mirrors that — `output[i] = input[i / factor]` per channel.
fn upsample_nearest_row_major(input: &[f32], ch: usize, t_in: usize, factor: usize) -> Vec<f32> {
    if factor == 0 || t_in == 0 {
        return Vec::new();
    }
    let t_out = t_in * factor;
    let mut output = vec![0.0f32; ch * t_out];
    for c in 0..ch {
        let src = c * t_in;
        let dst = c * t_out;
        for i in 0..t_out {
            output[dst + i] = input[src + i / factor];
        }
    }
    output
}

/// Left-side reflection padding on a row-major `[ch, t]` tensor.
///
/// Upstream `nn.ReflectionPad1d((pad_left, 0))` reflects the input past its
/// left boundary, *excluding* the boundary sample itself — for
/// `pad_left = 1` and channel data `[a, b, c, ...]` the result is
/// `[b, a, b, c, ...]` (PyTorch docs, "Pads the input tensor using the
/// reflection of the input boundary."). The upstream `HiFTGenerator` uses
/// `pad_left = 1` on the last upsample stage only; larger paddings are
/// supported here for defensive completeness but each fires the same
/// mirror formula.
fn reflection_pad_1d_left(input: &[f32], ch: usize, t: usize, pad_left: usize) -> Result<Vec<f32>> {
    if pad_left == 0 {
        return Ok(input.to_vec());
    }
    if input.len() != ch * t {
        return Err(VokraError::InvalidArgument(format!(
            "reflection_pad_1d_left: input length {} != ch * t = {}",
            input.len(),
            ch * t
        )));
    }
    if t <= pad_left {
        return Err(VokraError::InvalidArgument(format!(
            "reflection_pad_1d_left: t ({t}) must exceed pad_left ({pad_left}) — reflection \
             needs an interior sample at index pad_left"
        )));
    }
    let t_out = t + pad_left;
    let mut output = vec![0.0f32; ch * t_out];
    for c in 0..ch {
        let src = c * t;
        let dst = c * t_out;
        // Reflected prefix: output[i] = input[pad_left - i] for i in 0..pad_left.
        for i in 0..pad_left {
            output[dst + i] = input[src + pad_left - i];
        }
        // Original samples shifted right by pad_left.
        for j in 0..t {
            output[dst + pad_left + j] = input[src + j];
        }
    }
    Ok(output)
}

/// Strided 1-D convolution with explicit `padding` and no dilation.
///
/// Output length: `t_out = (t_in + 2 * padding - kernel) / stride + 1` —
/// the standard PyTorch `nn.Conv1d` formula. Used by `HiFTGenerator.decode`
/// to downsample the concatenated STFT source stream so its time axis
/// meets each upsample stage's `t_current`; the existing
/// [`conv1d_same_padding`] helper only covers stride = 1, so a dedicated
/// strided path avoids folding an unused `stride == 1` fast-path onto
/// every same-padded convolution in this file.
///
/// Naive `O(out_ch × in_ch × kernel × t_out)` loop; every source-side
/// convolution in HiFTNet is small (`in_ch = n_fft + 2 = 18`,
/// `out_ch ≤ 256`, `kernel ≤ 16`), so the arithmetic budget per synthesis
/// step stays modest.
// The 9-arg surface matches the underlying nn primitive faithfully — see
// the same rationale on [`conv1d_same_padding`] and [`conv_transpose1d`].
#[allow(clippy::too_many_arguments)]
fn conv1d_strided_no_dilation(
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
    if stride == 0 || kernel == 0 {
        return Err(VokraError::InvalidArgument(
            "conv1d_strided_no_dilation: stride and kernel must be > 0".to_owned(),
        ));
    }
    if input.len() != in_ch * t_in {
        return Err(VokraError::InvalidArgument(format!(
            "conv1d_strided_no_dilation: input length {} != in_ch * t_in = {}",
            input.len(),
            in_ch * t_in
        )));
    }
    if weight.len() != out_ch * in_ch * kernel {
        return Err(VokraError::InvalidArgument(format!(
            "conv1d_strided_no_dilation: weight length {} != out_ch * in_ch * kernel = {}",
            weight.len(),
            out_ch * in_ch * kernel
        )));
    }
    if bias.len() != out_ch {
        return Err(VokraError::InvalidArgument(format!(
            "conv1d_strided_no_dilation: bias length {} != out_ch = {out_ch}",
            bias.len()
        )));
    }
    if t_in + 2 * padding < kernel {
        return Err(VokraError::InvalidArgument(format!(
            "conv1d_strided_no_dilation: t_in ({t_in}) + 2*padding ({padding}) < kernel \
             ({kernel})"
        )));
    }
    let t_out = (t_in + 2 * padding - kernel) / stride + 1;
    let mut output = vec![0.0f32; out_ch * t_out];
    let t_i = t_in as isize;
    let pad_i = padding as isize;
    for (oc, &b) in bias.iter().enumerate() {
        let row = oc * t_out;
        let w_off = oc * in_ch * kernel;
        for to in 0..t_out {
            let mut acc = b;
            for ic in 0..in_ch {
                let x_row = ic * t_in;
                let w_row = w_off + ic * kernel;
                for k in 0..kernel {
                    let src = (to * stride) as isize + k as isize - pad_i;
                    if src < 0 || src >= t_i {
                        continue;
                    }
                    acc += input[x_row + src as usize] * weight[w_row + k];
                }
            }
            output[row + to] = acc;
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

    #[test]
    fn get_padding_formula_matches_upstream() {
        // Upstream `get_padding(k, d) = (k*d - d) / 2 = d * (k - 1) / 2`.
        assert_eq!(get_padding(3, 1), 1);
        assert_eq!(get_padding(3, 3), 3);
        assert_eq!(get_padding(3, 5), 5);
        assert_eq!(get_padding(7, 1), 3);
        assert_eq!(get_padding(11, 1), 5);
    }

    #[test]
    fn dilated_conv_dilation_one_matches_undilated() {
        // With dilation=1 the dilated helper must produce the same output
        // as the plain-dilation conv1d_same_padding.
        let input: Vec<f32> = (0..(2 * 6)).map(|i| (i as f32) * 0.1).collect();
        let weight: Vec<f32> = (0..(3 * 2 * 3)).map(|i| ((i % 5) as f32) * 0.02).collect();
        let bias = vec![0.1f32, -0.2, 0.05];
        let a = conv1d_same_padding(&input, 2, 3, 3, 1, 6, &weight, &bias);
        let b = conv1d_dilated_same_padding(&input, 2, 3, 3, 1, 1, 6, &weight, &bias).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn dilated_conv_dilation_greater_than_one_broadens_receptive_field() {
        // With dilation=3 the same-padding formula gives padding = 3.
        // Zero input + non-zero bias → output equals bias.
        let input = vec![0.0f32; 4 * 8];
        let weight = vec![0.0f32; 4 * 4 * 3];
        let bias = vec![0.5f32, -0.5, 0.25, 1.0];
        let pad = get_padding(3, 3);
        let out = conv1d_dilated_same_padding(&input, 4, 4, 3, 3, pad, 8, &weight, &bias).unwrap();
        assert_eq!(out.len(), 4 * 8);
        for (oc, &b) in bias.iter().enumerate() {
            for t_idx in 0..8 {
                assert_eq!(out[oc * 8 + t_idx], b);
            }
        }
    }

    // Build a small ResBlock with zero weights + zero alphas. Under Snake
    // with `alpha = 0`, `snake(x) = x + (1/(0+1e-9)) * sin(0)^2 = x`, so
    // the whole branch collapses to `x = x + 0 = x`; the block acts as
    // the identity. Handy for shape / determinism tests.
    fn zero_res_block(channels: usize, kernel_size: usize, dilations: Vec<u32>) -> ResBlock {
        let n = dilations.len();
        let cfg = ResBlockConfig {
            channels: channels as u32,
            kernel_size: kernel_size as u32,
            dilations,
        };
        let weights = ResBlockWeights {
            convs1_w: vec![vec![0.0f32; channels * channels * kernel_size]; n],
            convs1_b: vec![vec![0.0f32; channels]; n],
            convs2_w: vec![vec![0.0f32; channels * channels * kernel_size]; n],
            convs2_b: vec![vec![0.0f32; channels]; n],
            activations1_alpha: vec![vec![0.0f32; channels]; n],
            activations2_alpha: vec![vec![0.0f32; channels]; n],
        };
        ResBlock::new(cfg, weights).unwrap()
    }

    #[test]
    fn res_block_zero_weights_is_identity() {
        // With every conv weight = 0 the residual carries the input
        // unchanged; Snake with alpha = 0 is identity by the closed form
        // above.
        let rb = zero_res_block(4, 3, vec![1, 3, 5]);
        let t = 6;
        let mut x: Vec<f32> = (0..(4 * t)).map(|i| (i as f32) * 0.1).collect();
        let x0 = x.clone();
        rb.forward_in_place(&mut x, t).unwrap();
        for (a, b) in x.iter().zip(x0.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "identity broken at diff = {}",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn res_block_forward_preserves_shape_and_finiteness() {
        let rb = zero_res_block(8, 3, vec![1, 3, 5]);
        let t = 16;
        let mut x = vec![0.0f32; 8 * t];
        rb.forward_in_place(&mut x, t).unwrap();
        assert_eq!(x.len(), 8 * t);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn res_block_forward_rejects_wrong_input_length() {
        let rb = zero_res_block(4, 3, vec![1, 3]);
        let mut x = vec![0.0f32; 4 * 6 - 1]; // one short
        let err = rb.forward_in_place(&mut x, 6).unwrap_err();
        assert!(err.to_string().contains("input length"), "{err}");
    }

    #[test]
    fn res_block_new_rejects_empty_dilations() {
        let cfg = ResBlockConfig {
            channels: 4,
            kernel_size: 3,
            dilations: vec![],
        };
        let weights = ResBlockWeights {
            convs1_w: vec![],
            convs1_b: vec![],
            convs2_w: vec![],
            convs2_b: vec![],
            activations1_alpha: vec![],
            activations2_alpha: vec![],
        };
        let err = ResBlock::new(cfg, weights).unwrap_err();
        assert!(
            err.to_string().contains("dilations must not be empty"),
            "{err}"
        );
    }

    #[test]
    fn res_block_new_rejects_branch_count_mismatch() {
        let cfg = ResBlockConfig {
            channels: 4,
            kernel_size: 3,
            dilations: vec![1, 3, 5],
        };
        // convs1_w has 2 entries but dilations has 3.
        let weights = ResBlockWeights {
            convs1_w: vec![vec![0.0f32; 4 * 4 * 3]; 2],
            convs1_b: vec![vec![0.0f32; 4]; 3],
            convs2_w: vec![vec![0.0f32; 4 * 4 * 3]; 3],
            convs2_b: vec![vec![0.0f32; 4]; 3],
            activations1_alpha: vec![vec![0.0f32; 4]; 3],
            activations2_alpha: vec![vec![0.0f32; 4]; 3],
        };
        let err = ResBlock::new(cfg, weights).unwrap_err();
        assert!(err.to_string().contains("convs1_w has 2 entries"), "{err}");
    }

    #[test]
    fn res_block_new_rejects_wrong_conv_weight_shape() {
        let cfg = ResBlockConfig {
            channels: 4,
            kernel_size: 3,
            dilations: vec![1],
        };
        let weights = ResBlockWeights {
            convs1_w: vec![vec![0.0f32; 10]], // wrong length
            convs1_b: vec![vec![0.0f32; 4]],
            convs2_w: vec![vec![0.0f32; 4 * 4 * 3]],
            convs2_b: vec![vec![0.0f32; 4]],
            activations1_alpha: vec![vec![0.0f32; 4]],
            activations2_alpha: vec![vec![0.0f32; 4]],
        };
        let err = ResBlock::new(cfg, weights).unwrap_err();
        assert!(err.to_string().contains("convs1_w[0]"), "{err}");
    }

    // -----------------------------------------------------------------------
    // Helpers landed in Wave 3c-2 (private, but exposed to tests via the
    // module boundary).
    // -----------------------------------------------------------------------

    #[test]
    fn leaky_relu_matches_reference_on_both_branches() {
        // Positive branch is identity.
        assert_eq!(leaky_relu(0.0, 0.1), 0.0);
        assert_eq!(leaky_relu(2.5, 0.1), 2.5);
        // Negative branch scales by slope.
        assert!((leaky_relu(-3.0, 0.1) - (-0.3)).abs() < 1e-6);
        // Default 0.01 slope (upstream `F.leaky_relu(x)` — the final
        // pre-conv_post call in decode).
        assert!((leaky_relu(-5.0, 0.01) - (-0.05)).abs() < 1e-6);
    }

    #[test]
    fn upsample_nearest_row_major_repeats_each_sample() {
        // Upstream `nn.Upsample(scale_factor=3, mode="nearest")` on `[a, b]`
        // yields `[a, a, a, b, b, b]`.
        let input = [1.0f32, 2.0];
        let out = upsample_nearest_row_major(&input, 1, 2, 3);
        assert_eq!(out, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn upsample_nearest_row_major_preserves_channels() {
        // Two channels, each of length 3, upsampled 4x → per-channel repeat
        // stays inside its own row (row-major layout `[ch, t]`).
        let input = vec![
            10.0, 20.0, 30.0, // channel 0
            -1.0, -2.0, -3.0, // channel 1
        ];
        let out = upsample_nearest_row_major(&input, 2, 3, 4);
        assert_eq!(out.len(), 2 * 3 * 4);
        // Channel 0
        assert_eq!(
            &out[0..12],
            &[
                10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 30.0
            ][..]
        );
        // Channel 1
        assert_eq!(
            &out[12..24],
            &[
                -1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -2.0, -2.0, -3.0, -3.0, -3.0, -3.0
            ][..]
        );
    }

    #[test]
    fn upsample_nearest_row_major_factor_one_is_identity() {
        let input = vec![0.5f32, -0.5, 0.25, -0.25];
        let out = upsample_nearest_row_major(&input, 1, 4, 1);
        assert_eq!(out, input);
    }

    #[test]
    fn reflection_pad_1d_left_pad_one_mirrors_index_one() {
        // Upstream `nn.ReflectionPad1d((1, 0))` on `[a, b, c, d]` yields
        // `[b, a, b, c, d]` (see torch docs example).
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let out = reflection_pad_1d_left(&input, 1, 4, 1).unwrap();
        assert_eq!(out, vec![2.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn reflection_pad_1d_left_pad_zero_is_identity() {
        // A no-op guard: pad_left = 0 must return the input verbatim, so
        // callers can supply it unconditionally without a branch of their own.
        let input = vec![7.0f32, 8.0, 9.0];
        let out = reflection_pad_1d_left(&input, 1, 3, 0).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn reflection_pad_1d_left_multichannel_preserves_channel_isolation() {
        // Row-major `[ch, t]`: reflection per channel must not leak into
        // the neighbouring row.
        let input = vec![
            10.0, 20.0, 30.0, // channel 0
            -1.0, -2.0, -3.0, // channel 1
        ];
        let out = reflection_pad_1d_left(&input, 2, 3, 1).unwrap();
        assert_eq!(out.len(), 2 * 4);
        assert_eq!(&out[0..4], &[20.0, 10.0, 20.0, 30.0][..]);
        assert_eq!(&out[4..8], &[-2.0, -1.0, -2.0, -3.0][..]);
    }

    #[test]
    fn reflection_pad_1d_left_rejects_pad_at_least_t() {
        // pad_left = 1 needs input[1], so t = 1 is insufficient — the
        // reflection would read out of bounds.
        let input = [5.0f32];
        let err = reflection_pad_1d_left(&input, 1, 1, 1).unwrap_err();
        assert!(err.to_string().contains("must exceed pad_left"), "{err}");
    }

    #[test]
    fn conv1d_strided_length_formula_matches_upstream_hift_source_downs() {
        // Upstream HiFTGenerator source_downs stage 0 with u = 8:
        // k = 16, stride = 8, padding = 4. `t_stft = t_mel * 64 + 1`; for
        // t_mel = 1 → t_stft = 65, t_out = (65 + 8 - 16)/8 + 1 = 8. That
        // matches the ups[0] output length `t_mel * upsample_rates[0] = 8`.
        let n_fft_plus_2 = 18;
        let out_ch = 4;
        let t_stft = 65;
        let input = vec![0.1f32; n_fft_plus_2 * t_stft];
        let weight = vec![0.01f32; out_ch * n_fft_plus_2 * 16];
        let bias = vec![0.0f32; out_ch];
        let out = conv1d_strided_no_dilation(
            &input,
            n_fft_plus_2,
            out_ch,
            16,
            8,
            4,
            t_stft,
            &weight,
            &bias,
        )
        .unwrap();
        assert_eq!(out.len(), out_ch * 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn conv1d_strided_stride_one_kernel_one_matches_source_downs_last_stage() {
        // Upstream last stage with u = 1: k = 1, stride = 1, pad = 0.
        // t_out == t_in.
        let n_fft_plus_2 = 10;
        let out_ch = 2;
        let t_stft = 9;
        // Weight is [out_ch, n_fft_plus_2, k=1] — a plain per-channel mix
        // (kernel = 1 makes the third dimension trivial, so we drop the
        // literal `* 1` from the allocation to keep clippy quiet).
        let weight = vec![1.0f32; out_ch * n_fft_plus_2];
        let bias = vec![0.5f32; out_ch];
        let input: Vec<f32> = (0..n_fft_plus_2 * t_stft)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let out = conv1d_strided_no_dilation(
            &input,
            n_fft_plus_2,
            out_ch,
            1,
            1,
            0,
            t_stft,
            &weight,
            &bias,
        )
        .unwrap();
        assert_eq!(out.len(), out_ch * t_stft);
    }

    #[test]
    fn conv1d_strided_rejects_shape_mismatches() {
        let input = vec![0.0f32; 4 * 6];
        let weight = vec![0.0f32; 2 * 4 * 3];
        let bias = vec![0.0f32; 2];
        // Wrong weight length triggers a loud error.
        assert!(conv1d_strided_no_dilation(&input, 4, 2, 3, 1, 1, 6, &weight[..5], &bias).is_err());
        // Wrong bias length triggers a loud error.
        assert!(conv1d_strided_no_dilation(&input, 4, 2, 3, 1, 1, 6, &weight, &bias[..1]).is_err());
        // Zero stride triggers a loud error.
        assert!(conv1d_strided_no_dilation(&input, 4, 2, 3, 0, 1, 6, &weight, &bias).is_err());
        // Zero kernel triggers a loud error.
        assert!(conv1d_strided_no_dilation(&input, 4, 2, 0, 1, 1, 6, &weight, &bias).is_err());
        // Wrong input length triggers a loud error.
        assert!(conv1d_strided_no_dilation(&input[..5], 4, 2, 3, 1, 1, 6, &weight, &bias).is_err());
    }

    // -----------------------------------------------------------------------
    // HiFTGenerator forward + decode — end-to-end shape / determinism pins.
    // -----------------------------------------------------------------------

    /// Build a small synthesized `HiFTGenerator` whose config still satisfies
    /// the upstream `t_si == t_current` construction contract:
    /// `n_fft = 8, hop_len = 2, upsample_rates = [2, 2]`, giving
    /// `total_upsample_factor = 2 * 2 * 2 = 8` and
    /// `t_stft = t_mel * 4 + 1 = t_current_final`.
    ///
    /// Every weight is 0, so the pipeline reduces to
    /// `x_post = 0 ⇒ magnitude = min(exp(0), 100) = 1, phase = sin(0) = 0`
    /// on every bin — bounded, finite, deterministic, and (because
    /// `audio_limit = 0.99`) explicitly clamped, so this fixture also
    /// exercises the terminal saturation branch.
    fn small_hift_generator() -> HiFTGenerator {
        let cfg = HiFTGeneratorConfig {
            in_channels: 4,
            base_channels: 8,
            nb_harmonics: 2,
            sampling_rate: 16000,
            nsf_alpha: 0.1,
            nsf_sigma: 0.003,
            nsf_voiced_threshold: 10.0,
            upsample_rates: vec![2, 2],
            upsample_kernel_sizes: vec![4, 4],
            istft_n_fft: 8,
            istft_hop_len: 2,
            resblock_kernel_sizes: vec![3],
            resblock_dilation_sizes: vec![vec![1]],
            source_resblock_kernel_sizes: vec![3, 3],
            source_resblock_dilation_sizes: vec![vec![1], vec![1]],
            lrelu_slope: 0.1,
            audio_limit: 0.99,
        };

        // F0Predictor (cond_channels = base_channels = 8, num_layers = 5).
        let mut f0_conv_weights = vec![vec![0.0f32; 8 * 4 * 3]]; // layer 0: [8, 4, 3]
        for _ in 1..5 {
            f0_conv_weights.push(vec![0.0f32; 8 * 8 * 3]); // layers 1-4: [8, 8, 3]
        }
        let f0_weights = F0PredictorWeights {
            conv_weights: f0_conv_weights,
            conv_biases: vec![vec![0.0f32; 8]; 5],
            linear_w: vec![0.0f32; 8],
            linear_b: vec![0.0f32; 1],
        };

        let ups_w = vec![
            vec![0.0f32; 8 * 4 * 4], // ups[0]: [in=8, out=4, k=4]
            vec![0.0f32; 4 * 2 * 4], // ups[1]: [in=4, out=2, k=4]
        ];
        let ups_b = vec![vec![0.0f32; 4], vec![0.0f32; 2]];

        // downsample_us for upsample_rates=[2, 2] resolves to [2, 1]:
        //   downsample_rates = [1, 2], cum = [1, 2], reversed = [2, 1].
        // Stage 0 uses u = 2 → k = 4, stride = 2, pad = 1; stage 1 uses u = 1
        // → k = 1, stride = 1, pad = 0.
        let n_fft_plus_2 = 10;
        let source_downs_w = vec![
            vec![0.0f32; 4 * n_fft_plus_2 * 4], // stage 0: [out=4, in=10, k=4]
            vec![0.0f32; 2 * n_fft_plus_2],     // stage 1: [out=2, in=10, k=1] (kernel = 1 elided)
        ];
        let source_downs_b = vec![vec![0.0f32; 4], vec![0.0f32; 2]];

        let make_res_zero = |ch: usize, k: usize, n_branches: usize| ResBlockWeights {
            convs1_w: vec![vec![0.0f32; ch * ch * k]; n_branches],
            convs1_b: vec![vec![0.0f32; ch]; n_branches],
            convs2_w: vec![vec![0.0f32; ch * ch * k]; n_branches],
            convs2_b: vec![vec![0.0f32; ch]; n_branches],
            activations1_alpha: vec![vec![0.0f32; ch]; n_branches],
            activations2_alpha: vec![vec![0.0f32; ch]; n_branches],
        };

        // source_resblocks: one per stage.
        let source_resblock_weights = vec![
            make_res_zero(4, 3, 1), // stage 0: channels = 4
            make_res_zero(2, 3, 1), // stage 1: channels = 2
        ];
        // resblocks: row-major [num_ups * num_kernels], num_kernels = 1.
        let resblock_weights = vec![
            make_res_zero(4, 3, 1), // resblocks[0]: stage 0, kernel 0
            make_res_zero(2, 3, 1), // resblocks[1]: stage 1, kernel 0
        ];

        let weights = HiFTGeneratorWeights {
            conv_pre_w: vec![0.0f32; 8 * 4 * 7],
            conv_pre_b: vec![0.0f32; 8],
            ups_w,
            ups_b,
            source_downs_w,
            source_downs_b,
            source_resblock_weights,
            resblock_weights,
            conv_post_w: vec![0.0f32; n_fft_plus_2 * 2 * 7],
            conv_post_b: vec![0.0f32; n_fft_plus_2],
            m_source_linear_w: vec![0.0f32; 3], // nb_harmonics + 1 = 3
            m_source_linear_b: 0.0,
            f0_predictor_weights: f0_weights,
        };

        HiFTGenerator::new(cfg, weights).expect("small hift generator must build")
    }

    #[test]
    fn hift_generator_forward_output_length_matches_upstream_contract() {
        // `t_current_final = t_mel * prod(upsample_rates) + 1 = t_mel * 4 + 1`,
        // and the istft output length is `(t_current - 1) * hop_len =
        // (t_mel * 4) * 2 = t_mel * 8 = t_source`. Under upstream defaults
        // that equals `t_mel * total_upsample_factor()`, i.e. the source
        // signal length.
        let g = small_hift_generator();
        for t_mel in [1usize, 2, 3, 5] {
            let mel = vec![0.0f32; 4 * t_mel];
            let audio = g.forward(&mel, t_mel).unwrap();
            assert_eq!(
                audio.len(),
                t_mel * g.cfg.total_upsample_factor() as usize,
                "t_mel = {t_mel}"
            );
        }
    }

    #[test]
    fn hift_generator_forward_is_deterministic_on_same_input() {
        // NsfEntropy::Deterministic + identical input ⇒ identical output.
        let g = small_hift_generator();
        let t_mel = 4;
        let mel: Vec<f32> = (0..(4 * t_mel))
            .map(|i| ((i % 7) as f32) * 0.03 - 0.05)
            .collect();
        let a = g.forward(&mel, t_mel).unwrap();
        let b = g.forward(&mel, t_mel).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn hift_generator_forward_output_is_finite_and_bounded_by_audio_limit() {
        let g = small_hift_generator();
        let t_mel = 3;
        let mel: Vec<f32> = (0..(4 * t_mel)).map(|i| (i as f32) * 0.01).collect();
        let audio = g.forward(&mel, t_mel).unwrap();
        let limit = g.cfg.audio_limit;
        for (k, &v) in audio.iter().enumerate() {
            assert!(v.is_finite(), "non-finite sample at {k}: {v}");
            assert!(
                v.abs() <= limit + 1e-6,
                "sample {k} = {v} exceeds audio_limit = {limit}"
            );
        }
    }

    #[test]
    fn hift_generator_forward_rejects_wrong_mel_shape() {
        let g = small_hift_generator();
        let mel = vec![0.0f32; 4 * 4 - 1]; // one short
        let err = g.forward(&mel, 4).unwrap_err();
        assert!(err.to_string().contains("mel length"), "{err}");
    }

    #[test]
    fn hift_generator_forward_rejects_zero_t_mel() {
        let g = small_hift_generator();
        let err = g.forward(&[], 0).unwrap_err();
        assert!(err.to_string().contains("t_mel must be > 0"), "{err}");
    }

    #[test]
    fn hift_generator_f0_predictor_forward_matches_direct_call() {
        // The `f0_predictor_forward` convenience wrapper must return the
        // same sequence the internal F0Predictor produces on the same mel.
        let g = small_hift_generator();
        let t_mel = 6;
        let mel: Vec<f32> = (0..(4 * t_mel)).map(|i| (i as f32) * 0.02 - 0.1).collect();
        let via_wrapper = g.f0_predictor_forward(&mel, t_mel).unwrap();
        let via_direct = g.f0_predictor.forward(&mel, t_mel).unwrap();
        assert_eq!(via_wrapper, via_direct);
    }
}
