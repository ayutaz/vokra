//! Native FunASR FSMN-VAD encoder primitive.
//!
//! This is the released `funasr/fsmn-vad` topology at model revision
//! `df20e6b30c653645fa4ff125cacfcabd1020a669`, not the earlier synthetic
//! two-class FFN scaffold. The primary source is FunASR
//! `funasr/models/fsmn_vad_streaming/encoder.py`: two input affines, four
//! `linear -> causal depthwise memory -> affine -> ReLU` blocks, two output
//! affines, and a 248-pdf softmax. Runtime dependencies remain first-party.

use vokra_core::{Result, VokraError};

/// Exact encoder geometry carried by the official checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsmnEncoderConfig {
    /// Number of FSMN blocks.
    pub n_blocks: usize,
    /// LFR-stacked input width.
    pub input_dim: usize,
    /// First input-affine width.
    pub input_affine_dim: usize,
    /// Per-block input/output width.
    pub linear_dim: usize,
    /// FSMN projection and depthwise-memory width.
    pub proj_dim: usize,
    /// Number of taps in the causal left-memory convolution.
    pub lorder: usize,
    /// Future-memory taps. The streaming runtime currently requires zero.
    pub rorder: usize,
    /// Dilation between left-memory taps.
    pub lstride: usize,
    /// Dilation between right-memory taps. Canonical streaming value is zero.
    pub rstride: usize,
    /// First output-affine width.
    pub output_affine_dim: usize,
    /// Number of posterior pdfs (248; pdf 0 is silence).
    pub output_dim: usize,
}

impl FsmnEncoderConfig {
    /// Geometry from the pinned official `config.yaml`.
    pub fn upstream_default() -> Self {
        Self {
            n_blocks: 4,
            input_dim: 400,
            input_affine_dim: 140,
            linear_dim: 250,
            proj_dim: 128,
            lorder: 20,
            rorder: 0,
            lstride: 1,
            rstride: 0,
            output_affine_dim: 140,
            output_dim: 248,
        }
    }

    /// Number of projected frames retained per block between chunks.
    pub fn left_history_frames(&self) -> usize {
        self.lorder.saturating_sub(1) * self.lstride
    }

    /// Validates the implemented streaming contract.
    pub fn validate(&self) -> Result<()> {
        for (label, value) in [
            ("n_blocks", self.n_blocks),
            ("input_dim", self.input_dim),
            ("input_affine_dim", self.input_affine_dim),
            ("linear_dim", self.linear_dim),
            ("proj_dim", self.proj_dim),
            ("lorder", self.lorder),
            ("lstride", self.lstride),
            ("output_affine_dim", self.output_affine_dim),
            ("output_dim", self.output_dim),
        ] {
            if value == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "fsmn_vad: {label} must be > 0 (got {value})"
                )));
            }
        }
        if self.rorder != 0 || self.rstride != 0 {
            return Err(VokraError::UnsupportedOp(format!(
                "fsmn_vad streaming supports the released causal contract only: \
                 rorder=0/rstride=0, got {}/{}; no silent future-context truncation",
                self.rorder, self.rstride
            )));
        }
        Ok(())
    }
}

/// Weights for one official FunASR `BasicBlock`.
#[derive(Debug, Clone)]
pub struct FsmnBlockWeights {
    /// Bias-free projection `[proj_dim, linear_dim]`.
    pub linear_weight: Vec<f32>,
    /// Depthwise left-memory weights `[proj_dim, lorder]`.
    pub memory_weight: Vec<f32>,
    /// Affine expansion `[linear_dim, proj_dim]`.
    pub affine_weight: Vec<f32>,
    /// Affine bias `[linear_dim]`.
    pub affine_bias: Vec<f32>,
}

impl FsmnBlockWeights {
    /// Validates all element counts against `cfg`.
    pub fn validate(&self, cfg: &FsmnEncoderConfig) -> Result<()> {
        check_len(
            "block.linear_weight",
            self.linear_weight.len(),
            cfg.proj_dim * cfg.linear_dim,
        )?;
        check_len(
            "block.memory_weight",
            self.memory_weight.len(),
            cfg.proj_dim * cfg.lorder,
        )?;
        check_len(
            "block.affine_weight",
            self.affine_weight.len(),
            cfg.linear_dim * cfg.proj_dim,
        )?;
        check_len("block.affine_bias", self.affine_bias.len(), cfg.linear_dim)
    }
}

/// Complete official FSMN encoder weights.
#[derive(Debug, Clone)]
pub struct FsmnVadWeights {
    /// `[input_affine_dim, input_dim]`.
    pub in_linear1_weight: Vec<f32>,
    /// `[input_affine_dim]`.
    pub in_linear1_bias: Vec<f32>,
    /// `[linear_dim, input_affine_dim]`.
    pub in_linear2_weight: Vec<f32>,
    /// `[linear_dim]`.
    pub in_linear2_bias: Vec<f32>,
    /// Four FSMN blocks for the canonical release.
    pub blocks: Vec<FsmnBlockWeights>,
    /// `[output_affine_dim, linear_dim]`.
    pub out_linear1_weight: Vec<f32>,
    /// `[output_affine_dim]`.
    pub out_linear1_bias: Vec<f32>,
    /// `[output_dim, output_affine_dim]`.
    pub out_linear2_weight: Vec<f32>,
    /// `[output_dim]`.
    pub out_linear2_bias: Vec<f32>,
}

impl FsmnVadWeights {
    /// Validates the exact released tensor geometry.
    pub fn validate(&self, cfg: &FsmnEncoderConfig) -> Result<()> {
        cfg.validate()?;
        if self.blocks.len() != cfg.n_blocks {
            return Err(VokraError::InvalidArgument(format!(
                "fsmn_vad: {} blocks, expected {}",
                self.blocks.len(),
                cfg.n_blocks
            )));
        }
        check_len(
            "in_linear1_weight",
            self.in_linear1_weight.len(),
            cfg.input_affine_dim * cfg.input_dim,
        )?;
        check_len(
            "in_linear1_bias",
            self.in_linear1_bias.len(),
            cfg.input_affine_dim,
        )?;
        check_len(
            "in_linear2_weight",
            self.in_linear2_weight.len(),
            cfg.linear_dim * cfg.input_affine_dim,
        )?;
        check_len(
            "in_linear2_bias",
            self.in_linear2_bias.len(),
            cfg.linear_dim,
        )?;
        check_len(
            "out_linear1_weight",
            self.out_linear1_weight.len(),
            cfg.output_affine_dim * cfg.linear_dim,
        )?;
        check_len(
            "out_linear1_bias",
            self.out_linear1_bias.len(),
            cfg.output_affine_dim,
        )?;
        check_len(
            "out_linear2_weight",
            self.out_linear2_weight.len(),
            cfg.output_dim * cfg.output_affine_dim,
        )?;
        check_len(
            "out_linear2_bias",
            self.out_linear2_bias.len(),
            cfg.output_dim,
        )?;
        for (index, block) in self.blocks.iter().enumerate() {
            block.validate(cfg).map_err(|error| {
                VokraError::InvalidArgument(format!("fsmn_vad block {index}: {error}"))
            })?;
        }
        Ok(())
    }
}

fn check_len(label: &str, got: usize, expected: usize) -> Result<()> {
    if got != expected {
        return Err(VokraError::InvalidArgument(format!(
            "fsmn_vad: {label} has {got} elements, expected {expected}"
        )));
    }
    Ok(())
}

/// Per-stream causal projected-history buffers.
#[derive(Debug, Clone)]
pub struct FsmnStreamState {
    per_block_history: Vec<Vec<f32>>,
    proj_dim: usize,
    history_frames: usize,
    n_blocks: usize,
}

impl FsmnStreamState {
    /// Allocates a zero state matching the released causal geometry.
    pub fn zeros(cfg: &FsmnEncoderConfig) -> Result<Self> {
        cfg.validate()?;
        let history_frames = cfg.left_history_frames();
        Ok(Self {
            per_block_history: vec![vec![0.0; history_frames * cfg.proj_dim]; cfg.n_blocks],
            proj_dim: cfg.proj_dim,
            history_frames,
            n_blocks: cfg.n_blocks,
        })
    }

    /// Clears every retained projected frame.
    pub fn reset(&mut self) {
        for history in &mut self.per_block_history {
            history.fill(0.0);
        }
    }

    /// Returns whether the state is completely zero.
    pub fn is_zero(&self) -> bool {
        self.per_block_history
            .iter()
            .all(|history| history.iter().all(|value| *value == 0.0))
    }

    /// Checks state/config geometry before a forward.
    pub fn matches(&self, cfg: &FsmnEncoderConfig) -> bool {
        self.proj_dim == cfg.proj_dim
            && self.history_frames == cfg.left_history_frames()
            && self.n_blocks == cfg.n_blocks
            && self.per_block_history.len() == cfg.n_blocks
            && self
                .per_block_history
                .iter()
                .all(|history| history.len() == self.history_frames * self.proj_dim)
    }
}

/// Backend seam for the learned projections and causal depthwise memory in
/// the released FSMN-VAD encoder.
///
/// Frontend feature extraction, ReLU, softmax, residual addition and stream
/// history bookkeeping remain host control/DSP. Implementations must execute
/// the learned matrix and depthwise-convolution work on their declared backend;
/// returning an explicit error is required when a shape or primitive is
/// unsupported.
pub trait FsmnBackendOps {
    /// Applies a row-wise linear projection with an optional learned bias.
    fn linear(
        &mut self,
        input: &[f32],
        rows: usize,
        input_dim: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
        output_dim: usize,
    ) -> Result<Vec<f32>>;

    /// Applies the learned causal depthwise memory and residual connection,
    /// updating the per-block projected history.
    fn causal_memory(
        &mut self,
        projected: &[f32],
        frames: usize,
        cfg: &FsmnEncoderConfig,
        weights: &[f32],
        history: &mut Vec<f32>,
    ) -> Result<Vec<f32>>;
}

struct ScalarFsmnOps;

impl FsmnBackendOps for ScalarFsmnOps {
    fn linear(
        &mut self,
        input: &[f32],
        rows: usize,
        input_dim: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
        output_dim: usize,
    ) -> Result<Vec<f32>> {
        let mut output = linear(input, rows, input_dim, weight, output_dim);
        if let Some(bias) = bias {
            debug_assert_eq!(bias.len(), output_dim);
            for row in 0..rows {
                for column in 0..output_dim {
                    output[row * output_dim + column] += bias[column];
                }
            }
        }
        Ok(output)
    }

    fn causal_memory(
        &mut self,
        projected: &[f32],
        frames: usize,
        cfg: &FsmnEncoderConfig,
        weights: &[f32],
        history: &mut Vec<f32>,
    ) -> Result<Vec<f32>> {
        Ok(causal_memory(projected, frames, cfg, weights, history))
    }
}

/// Runs the exact released encoder and returns `[frames, output_dim]` logits.
pub fn fsmn_vad_forward(
    cfg: &FsmnEncoderConfig,
    weights: &FsmnVadWeights,
    input_features: &[f32],
    state: &mut FsmnStreamState,
) -> Result<Vec<f32>> {
    fsmn_vad_forward_with_ops(cfg, weights, input_features, state, &mut ScalarFsmnOps)
}

/// Runs the exact released encoder through an injected learned-op backend.
///
/// [`fsmn_vad_forward`] remains the scalar CPU oracle and calls this function
/// with the built-in adapter. Device backends can inject GEMV and grouped
/// Conv1D without duplicating the model topology or stream-state semantics.
pub fn fsmn_vad_forward_with_ops<O: FsmnBackendOps>(
    cfg: &FsmnEncoderConfig,
    weights: &FsmnVadWeights,
    input_features: &[f32],
    state: &mut FsmnStreamState,
    ops: &mut O,
) -> Result<Vec<f32>> {
    weights.validate(cfg)?;
    if !state.matches(cfg) {
        return Err(VokraError::InvalidArgument(
            "fsmn_vad: stream state does not match encoder config".to_owned(),
        ));
    }
    if input_features.is_empty() || input_features.len() % cfg.input_dim != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "fsmn_vad: feature length {} must be a non-zero multiple of input_dim {}",
            input_features.len(),
            cfg.input_dim
        )));
    }
    let frames = input_features.len() / cfg.input_dim;

    let input_affine = affine_with_ops(
        ops,
        input_features,
        frames,
        cfg.input_dim,
        &weights.in_linear1_weight,
        Some(&weights.in_linear1_bias),
        cfg.input_affine_dim,
    )?;
    let mut hidden = affine_with_ops(
        ops,
        &input_affine,
        frames,
        cfg.input_affine_dim,
        &weights.in_linear2_weight,
        Some(&weights.in_linear2_bias),
        cfg.linear_dim,
    )?;
    hidden.iter_mut().for_each(|value| *value = value.max(0.0));

    for (block_index, block) in weights.blocks.iter().enumerate() {
        let projected = ops.linear(
            &hidden,
            frames,
            cfg.linear_dim,
            &block.linear_weight,
            None,
            cfg.proj_dim,
        )?;
        let memory = ops.causal_memory(
            &projected,
            frames,
            cfg,
            &block.memory_weight,
            &mut state.per_block_history[block_index],
        )?;
        hidden = affine_with_ops(
            ops,
            &memory,
            frames,
            cfg.proj_dim,
            &block.affine_weight,
            Some(&block.affine_bias),
            cfg.linear_dim,
        )?;
        hidden.iter_mut().for_each(|value| *value = value.max(0.0));
    }

    let output_affine = affine_with_ops(
        ops,
        &hidden,
        frames,
        cfg.linear_dim,
        &weights.out_linear1_weight,
        Some(&weights.out_linear1_bias),
        cfg.output_affine_dim,
    )?;
    affine_with_ops(
        ops,
        &output_affine,
        frames,
        cfg.output_affine_dim,
        &weights.out_linear2_weight,
        Some(&weights.out_linear2_bias),
        cfg.output_dim,
    )
}

fn causal_memory(
    projected: &[f32],
    frames: usize,
    cfg: &FsmnEncoderConfig,
    weights: &[f32],
    history: &mut Vec<f32>,
) -> Vec<f32> {
    let history_frames = cfg.left_history_frames();
    let mut combined = Vec::with_capacity((history_frames + frames) * cfg.proj_dim);
    combined.extend_from_slice(history);
    combined.extend_from_slice(projected);

    let mut output = projected.to_vec();
    for frame in 0..frames {
        for channel in 0..cfg.proj_dim {
            let mut sum = 0.0f32;
            for tap in 0..cfg.lorder {
                let source_frame = frame + tap * cfg.lstride;
                sum += combined[source_frame * cfg.proj_dim + channel]
                    * weights[channel * cfg.lorder + tap];
            }
            output[frame * cfg.proj_dim + channel] += sum;
        }
    }

    let keep = history_frames * cfg.proj_dim;
    history.clear();
    if keep > 0 {
        history.extend_from_slice(&combined[combined.len() - keep..]);
    }
    output
}

fn affine_with_ops<O: FsmnBackendOps>(
    ops: &mut O,
    input: &[f32],
    rows: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
) -> Result<Vec<f32>> {
    ops.linear(input, rows, input_dim, weight, bias, output_dim)
}

fn linear(
    input: &[f32],
    rows: usize,
    input_dim: usize,
    weight: &[f32],
    output_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; rows * output_dim];
    for row in 0..rows {
        for out in 0..output_dim {
            let mut sum = 0.0f32;
            for inner in 0..input_dim {
                sum += input[row * input_dim + inner] * weight[out * input_dim + inner];
            }
            output[row * output_dim + out] = sum;
        }
    }
    output
}

/// Stable row-wise softmax used by the released 248-pdf head.
pub fn softmax_last_axis(logits: &[f32], width: usize) -> Vec<f32> {
    if width == 0 || logits.len() % width != 0 {
        return Vec::new();
    }
    let mut output = vec![0.0f32; logits.len()];
    for (source, destination) in logits
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
    {
        let maximum = source.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (dst, &value) in destination.iter_mut().zip(source) {
            *dst = (value - maximum).exp();
            sum += *dst;
        }
        if sum > 0.0 && sum.is_finite() {
            for value in destination {
                *value /= sum;
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_weights(cfg: &FsmnEncoderConfig) -> FsmnVadWeights {
        FsmnVadWeights {
            in_linear1_weight: vec![0.0; cfg.input_affine_dim * cfg.input_dim],
            in_linear1_bias: vec![0.0; cfg.input_affine_dim],
            in_linear2_weight: vec![0.0; cfg.linear_dim * cfg.input_affine_dim],
            in_linear2_bias: vec![0.0; cfg.linear_dim],
            blocks: (0..cfg.n_blocks)
                .map(|_| FsmnBlockWeights {
                    linear_weight: vec![0.0; cfg.proj_dim * cfg.linear_dim],
                    memory_weight: vec![0.0; cfg.proj_dim * cfg.lorder],
                    affine_weight: vec![0.0; cfg.linear_dim * cfg.proj_dim],
                    affine_bias: vec![0.0; cfg.linear_dim],
                })
                .collect(),
            out_linear1_weight: vec![0.0; cfg.output_affine_dim * cfg.linear_dim],
            out_linear1_bias: vec![0.0; cfg.output_affine_dim],
            out_linear2_weight: vec![0.0; cfg.output_dim * cfg.output_affine_dim],
            out_linear2_bias: vec![0.0; cfg.output_dim],
        }
    }

    #[test]
    fn upstream_default_is_the_released_geometry() {
        let cfg = FsmnEncoderConfig::upstream_default();
        cfg.validate().unwrap();
        assert_eq!(
            (
                cfg.input_dim,
                cfg.input_affine_dim,
                cfg.linear_dim,
                cfg.proj_dim,
                cfg.output_affine_dim,
                cfg.output_dim,
            ),
            (400, 140, 250, 128, 140, 248)
        );
        assert_eq!(cfg.left_history_frames(), 19);
    }

    #[test]
    fn zero_graph_reduces_to_terminal_bias() {
        let mut cfg = FsmnEncoderConfig::upstream_default();
        cfg.n_blocks = 1;
        cfg.input_dim = 3;
        cfg.input_affine_dim = 2;
        cfg.linear_dim = 2;
        cfg.proj_dim = 2;
        cfg.lorder = 2;
        cfg.output_affine_dim = 2;
        cfg.output_dim = 3;
        let mut weights = zero_weights(&cfg);
        weights.out_linear2_bias = vec![0.5, -0.25, 0.125];
        let mut state = FsmnStreamState::zeros(&cfg).unwrap();
        let logits = fsmn_vad_forward(&cfg, &weights, &[1.0; 12], &mut state).unwrap();
        for row in logits.chunks_exact(3) {
            assert_eq!(row, [0.5, -0.25, 0.125]);
        }
    }

    #[test]
    fn state_carry_matches_one_chunk() {
        let mut cfg = FsmnEncoderConfig::upstream_default();
        cfg.n_blocks = 1;
        cfg.input_dim = 1;
        cfg.input_affine_dim = 1;
        cfg.linear_dim = 1;
        cfg.proj_dim = 1;
        cfg.lorder = 3;
        cfg.output_affine_dim = 1;
        cfg.output_dim = 2;
        let mut weights = zero_weights(&cfg);
        weights.in_linear1_weight[0] = 1.0;
        weights.in_linear2_weight[0] = 1.0;
        weights.blocks[0].linear_weight[0] = 1.0;
        weights.blocks[0]
            .memory_weight
            .copy_from_slice(&[0.2, 0.3, 0.4]);
        weights.blocks[0].affine_weight[0] = 1.0;
        weights.out_linear1_weight[0] = 1.0;
        weights.out_linear2_weight[0] = 1.0;
        weights.out_linear2_weight[1] = -1.0;

        let input = [1.0, 2.0, 3.0, 4.0];
        let mut whole_state = FsmnStreamState::zeros(&cfg).unwrap();
        let whole = fsmn_vad_forward(&cfg, &weights, &input, &mut whole_state).unwrap();

        let mut split_state = FsmnStreamState::zeros(&cfg).unwrap();
        let mut split = fsmn_vad_forward(&cfg, &weights, &input[..2], &mut split_state).unwrap();
        split.extend(fsmn_vad_forward(&cfg, &weights, &input[2..], &mut split_state).unwrap());
        assert_eq!(whole, split);
    }

    #[test]
    fn noncausal_config_refuses_without_truncating_future_context() {
        let mut cfg = FsmnEncoderConfig::upstream_default();
        cfg.rorder = 1;
        assert!(matches!(cfg.validate(), Err(VokraError::UnsupportedOp(_))));
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let probabilities = softmax_last_axis(&[1000.0, 999.0, -1000.0, 0.0], 2);
        for row in probabilities.chunks_exact(2) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }
}
