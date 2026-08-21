//! FireRedVAD streaming DFSMN primitive.
//!
//! This is a native transcription of `FireRedTeam/FireRedVAD` commit
//! `c30ec49e8cc69642b0ee65362eba11b9d11c6e54`, specifically
//! `fireredvad/core/detect_model.py` and the official
//! `fireredvad_stream_vad_with_cache.onnx` graph.  The primitive consumes
//! already-normalized 80-bin Kaldi fbank rows; PCM framing and checkpoint
//! CMVN remain model-level concerns.

use vokra_core::{Result, VokraError};

/// Fixed streaming DFSMN geometry carried by the official VAD checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FireredVadDfsmnConfig {
    /// Input feature width.
    pub input_dim: usize,
    /// ReLU affine width.
    pub hidden_dim: usize,
    /// Projected memory-channel width.
    pub projection_dim: usize,
    /// Number of causal memory stages, including the input stage.
    pub n_blocks: usize,
    /// Number of taps in each depthwise memory filter.
    pub memory_order: usize,
    /// Frame spacing between memory taps.
    pub memory_stride: usize,
    /// Sigmoid-head width.
    pub output_dim: usize,
}

impl FireredVadDfsmnConfig {
    /// Official `Stream-VAD` geometry.
    #[must_use]
    pub const fn official() -> Self {
        Self {
            input_dim: 80,
            hidden_dim: 256,
            projection_dim: 128,
            n_blocks: 8,
            memory_order: 20,
            memory_stride: 1,
            output_dim: 1,
        }
    }

    /// Validates non-zero geometry and the single-output speech head.
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("input_dim", self.input_dim),
            ("hidden_dim", self.hidden_dim),
            ("projection_dim", self.projection_dim),
            ("n_blocks", self.n_blocks),
            ("memory_order", self.memory_order),
            ("memory_stride", self.memory_stride),
            ("output_dim", self.output_dim),
        ] {
            if value == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "firered-vad DFSMN: {name} must be > 0"
                )));
            }
        }
        if self.output_dim != 1 {
            return Err(VokraError::InvalidArgument(format!(
                "firered-vad DFSMN: output_dim={}, expected one sigmoid speech head",
                self.output_dim
            )));
        }
        Ok(())
    }

    #[must_use]
    /// Number of projected history frames carried per memory stage.
    pub const fn cache_frames(&self) -> usize {
        (self.memory_order - 1) * self.memory_stride
    }
}

/// One residual DFSMN block's affine parameters.
#[derive(Debug, Clone)]
pub struct FireredVadDfsmnBlockWeights {
    /// `[projection_dim, hidden_dim]` in input-by-output order.
    pub fc1_weight: Vec<f32>,
    /// Hidden affine bias.
    pub fc1_bias: Vec<f32>,
    /// `[hidden_dim, projection_dim]` in input-by-output order.
    pub fc2_weight: Vec<f32>,
}

/// Complete official Stream-VAD DFSMN parameter bundle.
#[derive(Debug, Clone)]
pub struct FireredVadDfsmnWeights {
    /// `[input_dim, hidden_dim]`, input-by-output.
    pub input_fc1_weight: Vec<f32>,
    /// Input hidden-affine bias.
    pub input_fc1_bias: Vec<f32>,
    /// `[hidden_dim, projection_dim]`, input-by-output.
    pub input_fc2_weight: Vec<f32>,
    /// Input projection bias.
    pub input_fc2_bias: Vec<f32>,
    /// One `[projection_dim, memory_order]` depthwise kernel per FSMN stage.
    pub memory_weights: Vec<Vec<f32>>,
    /// `n_blocks - 1` residual DFSMN blocks.
    pub blocks: Vec<FireredVadDfsmnBlockWeights>,
    /// `[projection_dim, hidden_dim]`, input-by-output.
    pub dnn_weight: Vec<f32>,
    /// Final hidden-affine bias.
    pub dnn_bias: Vec<f32>,
    /// `[hidden_dim, 1]`, input-by-output.
    pub output_weight: Vec<f32>,
    /// Sigmoid-head bias.
    pub output_bias: Vec<f32>,
}

impl FireredVadDfsmnWeights {
    /// Checks every flat tensor length against `cfg`.
    pub fn validate(&self, cfg: &FireredVadDfsmnConfig) -> Result<()> {
        cfg.validate()?;
        let require = |name: &str, got: usize, expected: usize| -> Result<()> {
            if got != expected {
                return Err(VokraError::InvalidArgument(format!(
                    "firered-vad DFSMN: {name} has {got} elements, expected {expected}"
                )));
            }
            Ok(())
        };
        require(
            "input_fc1_weight",
            self.input_fc1_weight.len(),
            cfg.input_dim * cfg.hidden_dim,
        )?;
        require("input_fc1_bias", self.input_fc1_bias.len(), cfg.hidden_dim)?;
        require(
            "input_fc2_weight",
            self.input_fc2_weight.len(),
            cfg.hidden_dim * cfg.projection_dim,
        )?;
        require(
            "input_fc2_bias",
            self.input_fc2_bias.len(),
            cfg.projection_dim,
        )?;
        require("memory_weights", self.memory_weights.len(), cfg.n_blocks)?;
        require("blocks", self.blocks.len(), cfg.n_blocks - 1)?;
        for (index, memory) in self.memory_weights.iter().enumerate() {
            require(
                &format!("memory_weights[{index}]"),
                memory.len(),
                cfg.projection_dim * cfg.memory_order,
            )?;
        }
        for (index, block) in self.blocks.iter().enumerate() {
            require(
                &format!("blocks[{index}].fc1_weight"),
                block.fc1_weight.len(),
                cfg.projection_dim * cfg.hidden_dim,
            )?;
            require(
                &format!("blocks[{index}].fc1_bias"),
                block.fc1_bias.len(),
                cfg.hidden_dim,
            )?;
            require(
                &format!("blocks[{index}].fc2_weight"),
                block.fc2_weight.len(),
                cfg.hidden_dim * cfg.projection_dim,
            )?;
        }
        require(
            "dnn_weight",
            self.dnn_weight.len(),
            cfg.projection_dim * cfg.hidden_dim,
        )?;
        require("dnn_bias", self.dnn_bias.len(), cfg.hidden_dim)?;
        require(
            "output_weight",
            self.output_weight.len(),
            cfg.hidden_dim * cfg.output_dim,
        )?;
        require("output_bias", self.output_bias.len(), cfg.output_dim)
    }
}

/// Per-stream causal memory.  Each row is `[cache_frames, projection_dim]`.
#[derive(Debug, Clone)]
pub struct FireredVadDfsmnState {
    histories: Vec<Vec<f32>>,
    projection_dim: usize,
    cache_frames: usize,
}

impl FireredVadDfsmnState {
    /// Allocates zero causal histories for a fresh stream.
    pub fn zeros(cfg: &FireredVadDfsmnConfig) -> Result<Self> {
        cfg.validate()?;
        let cache_frames = cfg.cache_frames();
        Ok(Self {
            histories: vec![vec![0.0; cache_frames * cfg.projection_dim]; cfg.n_blocks],
            projection_dim: cfg.projection_dim,
            cache_frames,
        })
    }

    /// Returns whether this state was allocated for `cfg`.
    #[must_use]
    pub fn matches(&self, cfg: &FireredVadDfsmnConfig) -> bool {
        self.projection_dim == cfg.projection_dim
            && self.cache_frames == cfg.cache_frames()
            && self.histories.len() == cfg.n_blocks
            && self
                .histories
                .iter()
                .all(|history| history.len() == self.cache_frames * self.projection_dim)
    }

    /// Clears every causal history in place.
    pub fn reset(&mut self) {
        for history in &mut self.histories {
            history.fill(0.0);
        }
    }
}

/// Runs the official streaming DFSMN on normalized fbank rows.
pub fn firered_vad_dfsmn_forward(
    cfg: &FireredVadDfsmnConfig,
    weights: &FireredVadDfsmnWeights,
    features: &[f32],
    state: &mut FireredVadDfsmnState,
) -> Result<Vec<f32>> {
    weights.validate(cfg)?;
    if !state.matches(cfg) {
        return Err(VokraError::InvalidArgument(
            "firered-vad DFSMN state does not match the model geometry".to_owned(),
        ));
    }
    if features.is_empty() || features.len() % cfg.input_dim != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "firered-vad DFSMN features have {} elements, expected a non-empty multiple of {}",
            features.len(),
            cfg.input_dim
        )));
    }
    let frames = features.len() / cfg.input_dim;
    let mut hidden = affine_input_output(
        features,
        frames,
        cfg.input_dim,
        &weights.input_fc1_weight,
        &weights.input_fc1_bias,
        cfg.hidden_dim,
    );
    relu_in_place(&mut hidden);
    let mut projected = affine_input_output(
        &hidden,
        frames,
        cfg.hidden_dim,
        &weights.input_fc2_weight,
        &weights.input_fc2_bias,
        cfg.projection_dim,
    );
    relu_in_place(&mut projected);
    projected = memory_forward(
        &projected,
        frames,
        cfg,
        &weights.memory_weights[0],
        &mut state.histories[0],
    );

    let projection_zero_bias = vec![0.0; cfg.projection_dim];
    for (index, block) in weights.blocks.iter().enumerate() {
        let residual = projected.clone();
        hidden = affine_input_output(
            &projected,
            frames,
            cfg.projection_dim,
            &block.fc1_weight,
            &block.fc1_bias,
            cfg.hidden_dim,
        );
        relu_in_place(&mut hidden);
        projected = affine_input_output(
            &hidden,
            frames,
            cfg.hidden_dim,
            &block.fc2_weight,
            &projection_zero_bias,
            cfg.projection_dim,
        );
        projected = memory_forward(
            &projected,
            frames,
            cfg,
            &weights.memory_weights[index + 1],
            &mut state.histories[index + 1],
        );
        for (value, skip) in projected.iter_mut().zip(residual) {
            *value += skip;
        }
    }

    hidden = affine_input_output(
        &projected,
        frames,
        cfg.projection_dim,
        &weights.dnn_weight,
        &weights.dnn_bias,
        cfg.hidden_dim,
    );
    relu_in_place(&mut hidden);
    let logits = affine_input_output(
        &hidden,
        frames,
        cfg.hidden_dim,
        &weights.output_weight,
        &weights.output_bias,
        cfg.output_dim,
    );
    Ok(logits
        .into_iter()
        .map(|value| 1.0 / (1.0 + (-value).exp()))
        .collect())
}

fn memory_forward(
    input: &[f32],
    frames: usize,
    cfg: &FireredVadDfsmnConfig,
    weight: &[f32],
    history: &mut Vec<f32>,
) -> Vec<f32> {
    let cache_frames = cfg.cache_frames();
    let mut combined = Vec::with_capacity((cache_frames + frames) * cfg.projection_dim);
    combined.extend_from_slice(history);
    combined.extend_from_slice(input);
    let mut output = input.to_vec();
    for frame in 0..frames {
        for channel in 0..cfg.projection_dim {
            let mut memory = 0.0f32;
            for tap in 0..cfg.memory_order {
                let source_frame = frame + tap * cfg.memory_stride;
                memory += weight[channel * cfg.memory_order + tap]
                    * combined[source_frame * cfg.projection_dim + channel];
            }
            output[frame * cfg.projection_dim + channel] += memory;
        }
    }
    let keep_start = combined.len() - cache_frames * cfg.projection_dim;
    history.clear();
    history.extend_from_slice(&combined[keep_start..]);
    output
}

fn affine_input_output(
    input: &[f32],
    rows: usize,
    input_dim: usize,
    weight: &[f32],
    bias: &[f32],
    output_dim: usize,
) -> Vec<f32> {
    debug_assert_eq!(input.len(), rows * input_dim);
    debug_assert_eq!(weight.len(), input_dim * output_dim);
    debug_assert_eq!(bias.len(), output_dim);
    let mut output = vec![0.0; rows * output_dim];
    for row in 0..rows {
        for out in 0..output_dim {
            let mut sum = bias[out];
            for inner in 0..input_dim {
                sum += input[row * input_dim + inner] * weight[inner * output_dim + out];
            }
            output[row * output_dim + out] = sum;
        }
    }
    output
}

fn relu_in_place(values: &mut [f32]) {
    for value in values {
        *value = value.max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> (FireredVadDfsmnConfig, FireredVadDfsmnWeights) {
        let cfg = FireredVadDfsmnConfig {
            input_dim: 2,
            hidden_dim: 2,
            projection_dim: 2,
            n_blocks: 2,
            memory_order: 2,
            memory_stride: 1,
            output_dim: 1,
        };
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let weights = FireredVadDfsmnWeights {
            input_fc1_weight: identity.clone(),
            input_fc1_bias: vec![0.0; 2],
            input_fc2_weight: identity.clone(),
            input_fc2_bias: vec![0.0; 2],
            memory_weights: vec![vec![0.0; 4]; 2],
            blocks: vec![FireredVadDfsmnBlockWeights {
                fc1_weight: vec![0.0; 4],
                fc1_bias: vec![0.0; 2],
                fc2_weight: vec![0.0; 4],
            }],
            dnn_weight: identity,
            dnn_bias: vec![0.0; 2],
            output_weight: vec![1.0, 1.0],
            output_bias: vec![0.0],
        };
        (cfg, weights)
    }

    #[test]
    fn zero_memory_residual_path_is_exact() {
        let (cfg, weights) = tiny();
        let mut state = FireredVadDfsmnState::zeros(&cfg).unwrap();
        let output = firered_vad_dfsmn_forward(&cfg, &weights, &[1.0, 2.0], &mut state).unwrap();
        let expected = 1.0 / (1.0 + (-3.0f32).exp());
        assert!((output[0] - expected).abs() < 1e-7);
    }

    #[test]
    fn chunked_forward_matches_batch() {
        let (cfg, mut weights) = tiny();
        weights.memory_weights[0] = vec![0.25, 0.5, 0.25, 0.5];
        let features = [1.0, 0.5, 0.25, 1.5, 0.75, 0.2];
        let mut batch_state = FireredVadDfsmnState::zeros(&cfg).unwrap();
        let batch = firered_vad_dfsmn_forward(&cfg, &weights, &features, &mut batch_state).unwrap();
        let mut stream_state = FireredVadDfsmnState::zeros(&cfg).unwrap();
        let mut streamed = Vec::new();
        for row in features.chunks_exact(2) {
            streamed
                .extend(firered_vad_dfsmn_forward(&cfg, &weights, row, &mut stream_state).unwrap());
        }
        assert_eq!(batch, streamed);
    }
}
