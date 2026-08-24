//! Native TEN-VAD v1.0 runtime.
//!
//! The model is pinned to `TEN-framework/ten-vad` commit
//! `8e96899ba05a8e8c0e883ec7417e7a144bd9dec0` (`v1.0-ONNX`). The offline
//! sidecar validates the released ONNX and canonicalizes all 19 float
//! initializers; this binder rejects every other manifest. Runtime inference
//! is first-party Rust: the LPCNet-derived DSP frontend and the exact ONNX
//! separable-conv/two-LSTM graph live in `vokra_ops::ten_vad`.
//!
//! License: the upstream release adds non-compete and application-only terms
//! to Apache-2.0, so canonical conversions fail closed as redistribution
//! forbidden. The modified LPCNet frontend separately preserves the upstream
//! BSD-2-Clause and BSD-3-Clause `NOTICES` obligations.

use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_core::engines::{VadEngine, VadStreamHandle};
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};
use vokra_ops::ten_vad::{
    CONV_CHANNELS, HIDDEN_DIM, LSTM0_INPUT, LstmWeights, SeparableConvWeights, TenVadBackendOps,
    TenVadFrontend, TenVadNetworkState, TenVadNetworkWeights, network_forward,
    network_forward_with_ops,
};

use crate::compute::{Compute, HotOp};

const TEN_VAD_HOT_OPS: &[HotOp] = &[
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
    HotOp::Gemv,
    HotOp::Gemm,
];

/// GGUF architecture discriminator.
pub const ARCH: &str = "ten_vad";
/// Canonical model name.
pub const NAME: &str = "ten-vad-v1.0";
/// Shared runtime category.
pub const CATEGORY: &str = "vad-kws";
/// Pinned primary-source repository.
pub const UPSTREAM_URL: &str = "github.com/TEN-framework/ten-vad";
/// SPDX-style identifier for Agora's restricted TEN-VAD license.
pub const DEFAULT_LICENSE_SPDX: &str = "LicenseRef-Agora-TEN-VAD-Open-Source-License-2025";
/// SPDX expression for the LPCNet-derived frontend code.
pub const FRONTEND_LICENSE_SPDX: &str = "bsd-2-clause AND bsd-3-clause";
/// Pinned upstream commit for the v1.0 ONNX release.
pub const REVISION: &str = "8e96899ba05a8e8c0e883ec7417e7a144bd9dec0";
/// SHA-256 of the pinned upstream ONNX file.
pub const ONNX_SHA256: &str = "e10b98a0cab1c98e847fbdda14cb3d45a38336d47535a3f63a0fb6c4e0f4cdf4";
/// Exact number of float initializers in the pinned graph.
pub const TENSOR_COUNT: usize = 19;

/// GGUF metadata key for the model category.
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
/// GGUF metadata key for the upstream source URL.
pub const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";
/// GGUF metadata key for the required PCM sample rate.
pub const KEY_SAMPLE_RATE: &str = "vokra.ten_vad.sample_rate";
/// GGUF metadata key for the streaming hop size.
pub const KEY_HOP_SIZE: &str = "vokra.ten_vad.hop_size";
/// GGUF metadata key for features per frame.
pub const KEY_N_FEATURES: &str = "vokra.ten_vad.n_features";
/// GGUF metadata key for frontend context frames.
pub const KEY_CONTEXT_FRAMES: &str = "vokra.ten_vad.context_frames";
/// GGUF metadata key for recurrent hidden width.
pub const KEY_HIDDEN_DIM: &str = "vokra.ten_vad.hidden_dim";
/// GGUF metadata key for recurrent layer count.
pub const KEY_N_LAYERS: &str = "vokra.ten_vad.n_layers";
/// GGUF metadata key for the pinned upstream revision.
pub const KEY_REVISION: &str = "vokra.ten_vad.revision";
/// GGUF metadata key for the pinned source-model digest.
pub const KEY_ONNX_SHA256: &str = "vokra.ten_vad.onnx_sha256";
/// GGUF metadata key for the frontend license expression.
pub const KEY_FRONTEND_LICENSE: &str = "vokra.ten_vad.frontend_license_spdx";

/// Fixed topology of the official v1.0 release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TenVadConfig {
    /// Required mono PCM sample rate.
    pub sample_rate: u32,
    /// Samples consumed per inference step.
    pub hop_size: usize,
    /// Features produced per frontend frame.
    pub n_features: usize,
    /// Consecutive feature frames passed to the graph.
    pub context_frames: usize,
    /// Hidden width of each LSTM layer.
    pub hidden_dim: usize,
    /// Number of stacked LSTM layers.
    pub n_layers: usize,
}

impl TenVadConfig {
    /// Returns the immutable topology of the pinned v1.0 graph.
    #[must_use]
    pub const fn upstream_default() -> Self {
        Self {
            sample_rate: 16_000,
            hop_size: 256,
            n_features: 41,
            context_frames: 3,
            hidden_dim: 64,
            n_layers: 2,
        }
    }

    fn from_gguf(file: &GgufFile) -> Result<Self> {
        let expected = Self::upstream_default();
        let actual = Self {
            sample_rate: metadata_u32(file, KEY_SAMPLE_RATE)?,
            hop_size: metadata_usize(file, KEY_HOP_SIZE)?,
            n_features: metadata_usize(file, KEY_N_FEATURES)?,
            context_frames: metadata_usize(file, KEY_CONTEXT_FRAMES)?,
            hidden_dim: metadata_usize(file, KEY_HIDDEN_DIM)?,
            n_layers: metadata_usize(file, KEY_N_LAYERS)?,
        };
        if actual != expected {
            return Err(VokraError::ModelLoad(format!(
                "ten_vad topology is {actual:?}, expected pinned v1.0 topology {expected:?}"
            )));
        }
        Ok(actual)
    }
}

/// Strictly bound canonical TEN-VAD weights.
#[derive(Debug)]
pub struct TenVadWeights {
    tensors: Vec<(String, Vec<usize>)>,
    network: TenVadNetworkWeights,
}

impl TenVadWeights {
    /// Strictly binds all 19 canonical tensors from `file`.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        if file.tensors().len() != TENSOR_COUNT {
            return Err(VokraError::ModelLoad(format!(
                "ten_vad GGUF has {} tensors, expected exactly {TENSOR_COUNT}",
                file.tensors().len()
            )));
        }
        let tensors = file
            .tensors()
            .iter()
            .map(|info| {
                (
                    info.name.clone(),
                    info.dimensions
                        .iter()
                        .map(|&dimension| dimension as usize)
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let conv = |prefix: &str,
                    depthwise: &[usize],
                    pointwise: &[usize]|
         -> Result<SeparableConvWeights> {
            Ok(SeparableConvWeights {
                depthwise: tensor(file, &format!("{prefix}.depthwise.weight"), depthwise)?,
                pointwise: tensor(file, &format!("{prefix}.pointwise.weight"), pointwise)?,
                bias: tensor(file, &format!("{prefix}.pointwise.bias"), &[16])?,
            })
        };
        let lstm = |prefix: &str, input_size: usize| -> Result<LstmWeights> {
            Ok(LstmWeights {
                weight_ih: tensor(file, &format!("{prefix}.weight_ih"), &[1, 256, input_size])?,
                weight_hh: tensor(file, &format!("{prefix}.weight_hh"), &[1, 256, HIDDEN_DIM])?,
                bias: tensor(file, &format!("{prefix}.bias"), &[1, 512])?,
                input_size,
            })
        };
        let dense1_bias = tensor(file, "ten_vad.dense1.bias", &[1])?[0];
        let network = TenVadNetworkWeights {
            conv0: conv("ten_vad.conv0", &[1, 1, 3, 3], &[16, 1, 1, 1])?,
            conv1: conv("ten_vad.conv1", &[16, 1, 1, 3], &[16, 16, 1, 1])?,
            conv2: conv("ten_vad.conv2", &[16, 1, 1, 3], &[16, 16, 1, 1])?,
            lstm0: lstm("ten_vad.lstm0", LSTM0_INPUT)?,
            lstm1: lstm("ten_vad.lstm1", HIDDEN_DIM)?,
            dense0_weight: tensor(file, "ten_vad.dense0.weight", &[128, 32])?,
            dense0_bias: tensor(file, "ten_vad.dense0.bias", &[32])?,
            dense1_weight: tensor(file, "ten_vad.dense1.weight", &[32, 1])?,
            dense1_bias,
        };
        network
            .validate()
            .map_err(|error| VokraError::ModelLoad(error.to_string()))?;
        Ok(Self { tensors, network })
    }

    #[must_use]
    /// Returns the bound tensor count.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    #[must_use]
    /// Returns zero because the pinned manifest is strictly F32.
    pub fn bf16_count(&self) -> usize {
        0
    }

    /// Iterates over canonical tensor names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|(name, _)| name.as_str())
    }

    #[must_use]
    /// Returns whether the canonical manifest contains `name`.
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.iter().any(|(candidate, _)| candidate == name)
    }

    /// Returns the shape of `name`, or a loud model-load error when absent.
    pub fn require_tensor(&self, name: &str) -> Result<&[usize]> {
        self.tensors
            .iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, dimensions)| dimensions.as_slice())
            .ok_or_else(|| VokraError::ModelLoad(format!("ten_vad missing tensor `{name}`")))
    }

    #[must_use]
    /// Returns canonical tensor names and shapes.
    pub fn tensors(&self) -> &[(String, Vec<usize>)] {
        &self.tensors
    }
}

/// Native TEN-VAD v1.0 model handle.
#[derive(Debug)]
pub struct TenVad {
    weights: Arc<TenVadWeights>,
    config: TenVadConfig,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl TenVad {
    /// Strictly binds a parsed canonical TEN-VAD GGUF.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        require_string(file, chunks::KEY_MODEL_ARCH, ARCH)?;
        require_string(file, KEY_REVISION, REVISION)?;
        require_string(file, KEY_ONNX_SHA256, ONNX_SHA256)?;
        require_string(file, KEY_FRONTEND_LICENSE, FRONTEND_LICENSE_SPDX)?;
        let config = TenVadConfig::from_gguf(file)?;
        let weights = TenVadWeights::from_gguf(file)?;
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(GgufMetadataValue::as_str)
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);
        Ok(Self {
            weights: Arc::new(weights),
            config,
            weight_license,
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and binds a canonical TEN-VAD GGUF from disk.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    #[must_use]
    /// Returns the pinned runtime configuration.
    pub const fn config(&self) -> &TenVadConfig {
        &self.config
    }

    #[must_use]
    /// Returns the bound canonical weights.
    pub fn weights(&self) -> &TenVadWeights {
        &self.weights
    }

    #[must_use]
    /// Returns the bound tensor count.
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    #[must_use]
    /// Returns zero because the pinned manifest is strictly F32.
    pub fn bf16_count(&self) -> usize {
        0
    }

    #[must_use]
    /// Returns the fail-closed weight license class stamped in the GGUF.
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Selects the backend for every learned convolution, recurrent
    /// projection and dense layer. The LPCNet frontend remains host DSP.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Returns the selected learned-op backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Runs the exact neural graph on one normalized `3 x 41` feature context.
    pub fn predict_features(
        &self,
        features: &[f32],
        state: &mut TenVadNetworkState,
    ) -> Result<f32> {
        ten_vad_forward_dispatch(self.backend, features, &self.weights.network, state)
    }

    /// Runs a fresh frontend/network state on one 256-sample PCM frame.
    pub fn frame_probability(&self, frame: &[f32]) -> Result<f32> {
        let mut frontend = TenVadFrontend::new();
        let features = frontend.process_frame(frame)?;
        self.predict_features(features, &mut TenVadNetworkState::default())
    }
}

fn ten_vad_forward_dispatch(
    backend: BackendKind,
    features: &[f32],
    weights: &TenVadNetworkWeights,
    state: &mut TenVadNetworkState,
) -> Result<f32> {
    if backend == BackendKind::Cpu {
        return network_forward(features, weights, state);
    }
    let compute = Compute::for_backend(backend, TEN_VAD_HOT_OPS)?;
    network_forward_with_ops(
        features,
        weights,
        state,
        &mut ComputeTenVadOps { compute: &compute },
    )
}

struct ComputeTenVadOps<'a> {
    compute: &'a Compute,
}

impl TenVadBackendOps for ComputeTenVadOps<'_> {
    fn conv_stack(&mut self, features: &[f32], weights: &TenVadNetworkWeights) -> Result<Vec<f32>> {
        let mut conv0_depthwise = vec![0.0f32; 39];
        self.compute.conv1d_f32(
            features,
            3,
            41,
            &weights.conv0.depthwise,
            1,
            3,
            None,
            1,
            0,
            &mut conv0_depthwise,
        )?;
        let mut conv0 = vec![0.0f32; CONV_CHANNELS * 39];
        self.compute.conv1d_f32(
            &conv0_depthwise,
            1,
            39,
            &weights.conv0.pointwise,
            CONV_CHANNELS,
            1,
            Some(&weights.conv0.bias),
            1,
            0,
            &mut conv0,
        )?;
        conv0.iter_mut().for_each(|value| *value = value.max(0.0));

        let mut pooled = vec![0.0f32; CONV_CHANNELS * 19];
        for channel in 0..CONV_CHANNELS {
            for output in 0..19 {
                let start = channel * 39 + output * 2;
                pooled[channel * 19 + output] =
                    conv0[start].max(conv0[start + 1]).max(conv0[start + 2]);
            }
        }

        let mut conv1_depthwise = vec![0.0f32; CONV_CHANNELS * 10];
        self.compute.grouped_conv1d_f32(
            &pooled,
            CONV_CHANNELS,
            19,
            &weights.conv1.depthwise,
            CONV_CHANNELS,
            3,
            None,
            2,
            1,
            CONV_CHANNELS,
            &mut conv1_depthwise,
        )?;
        let mut conv1 = vec![0.0f32; CONV_CHANNELS * 10];
        self.compute.conv1d_f32(
            &conv1_depthwise,
            CONV_CHANNELS,
            10,
            &weights.conv1.pointwise,
            CONV_CHANNELS,
            1,
            Some(&weights.conv1.bias),
            1,
            0,
            &mut conv1,
        )?;
        conv1.iter_mut().for_each(|value| *value = value.max(0.0));

        // The final upstream depthwise convolution uses right-only padding by
        // one sample. Materialize that layout explicitly, then call the
        // ordinary valid grouped kernel; symmetric padding would shift it.
        let mut conv1_right_padded = vec![0.0f32; CONV_CHANNELS * 11];
        for channel in 0..CONV_CHANNELS {
            conv1_right_padded[channel * 11..channel * 11 + 10]
                .copy_from_slice(&conv1[channel * 10..(channel + 1) * 10]);
        }
        let mut conv2_depthwise = vec![0.0f32; CONV_CHANNELS * 5];
        self.compute.grouped_conv1d_f32(
            &conv1_right_padded,
            CONV_CHANNELS,
            11,
            &weights.conv2.depthwise,
            CONV_CHANNELS,
            3,
            None,
            2,
            0,
            CONV_CHANNELS,
            &mut conv2_depthwise,
        )?;
        let mut conv2 = vec![0.0f32; CONV_CHANNELS * 5];
        self.compute.conv1d_f32(
            &conv2_depthwise,
            CONV_CHANNELS,
            5,
            &weights.conv2.pointwise,
            CONV_CHANNELS,
            1,
            Some(&weights.conv2.bias),
            1,
            0,
            &mut conv2,
        )?;
        conv2.iter_mut().for_each(|value| *value = value.max(0.0));

        let mut lstm_input = vec![0.0f32; LSTM0_INPUT];
        for time in 0..5 {
            for channel in 0..CONV_CHANNELS {
                lstm_input[time * CONV_CHANNELS + channel] = conv2[channel * 5 + time];
            }
        }
        Ok(lstm_input)
    }

    fn lstm_step(
        &mut self,
        input: &[f32],
        weights: &LstmWeights,
        hidden: &mut [f32],
        cell: &mut [f32],
    ) -> Result<Vec<f32>> {
        let previous_hidden = hidden.to_vec();
        let previous_cell = cell.to_vec();
        let gate_dim = 4 * HIDDEN_DIM;
        let mut combined_bias = vec![0.0f32; gate_dim];
        for (row, bias) in combined_bias.iter_mut().enumerate() {
            *bias = weights.bias[row] + weights.bias[gate_dim + row];
        }
        let mut gates = vec![0.0f32; gate_dim];
        self.compute.gemv_f32(
            gate_dim,
            weights.input_size,
            &weights.weight_ih,
            input,
            Some(&combined_bias),
            &mut gates,
        )?;
        let mut recurrent = vec![0.0f32; gate_dim];
        self.compute.gemv_f32(
            gate_dim,
            HIDDEN_DIM,
            &weights.weight_hh,
            &previous_hidden,
            None,
            &mut recurrent,
        )?;
        for (gate, recurrent) in gates.iter_mut().zip(recurrent) {
            *gate += recurrent;
        }
        for unit in 0..HIDDEN_DIM {
            let input_gate = ten_vad_sigmoid(gates[unit]);
            let output_gate = ten_vad_sigmoid(gates[HIDDEN_DIM + unit]);
            let forget_gate = ten_vad_sigmoid(gates[2 * HIDDEN_DIM + unit]);
            let cell_gate = gates[3 * HIDDEN_DIM + unit].tanh();
            cell[unit] = forget_gate * previous_cell[unit] + input_gate * cell_gate;
            hidden[unit] = output_gate * cell[unit].tanh();
        }
        Ok(hidden.to_vec())
    }

    fn dense_head(
        &mut self,
        hidden1: &[f32],
        hidden0: &[f32],
        weights: &TenVadNetworkWeights,
    ) -> Result<f32> {
        let mut dense_input = Vec::with_capacity(HIDDEN_DIM * 2);
        dense_input.extend_from_slice(hidden1);
        dense_input.extend_from_slice(hidden0);
        let mut dense = vec![0.0f32; 32];
        self.compute.gemm_f32(
            1,
            32,
            HIDDEN_DIM * 2,
            &dense_input,
            &weights.dense0_weight,
            Some(&weights.dense0_bias),
            &mut dense,
        )?;
        dense.iter_mut().for_each(|value| *value = value.max(0.0));
        let bias = [weights.dense1_bias];
        let mut logit = [0.0f32];
        self.compute.gemm_f32(
            1,
            1,
            32,
            &dense,
            &weights.dense1_weight,
            Some(&bias),
            &mut logit,
        )?;
        Ok(ten_vad_sigmoid(logit[0]))
    }
}

#[inline]
fn ten_vad_sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

impl VadEngine for TenVad {
    fn open_stream(&self) -> Box<dyn VadStreamHandle + Send> {
        Box::new(TenVadStream {
            config: self.config,
            weights: Arc::clone(&self.weights),
            pending_pcm: Vec::new(),
            frontend: TenVadFrontend::new(),
            network_state: TenVadNetworkState::default(),
            backend: self.backend,
        })
    }
}

/// Stateful chunk-invariant TEN-VAD stream.
pub struct TenVadStream {
    config: TenVadConfig,
    weights: Arc<TenVadWeights>,
    pending_pcm: Vec<f32>,
    frontend: TenVadFrontend,
    network_state: TenVadNetworkState,
    backend: BackendKind,
}

impl TenVadStream {
    #[must_use]
    /// Number of unconsumed PCM samples buffered for the next hop.
    pub fn pending_samples(&self) -> usize {
        self.pending_pcm.len()
    }

    #[must_use]
    /// Returns the pinned runtime configuration.
    pub const fn config(&self) -> &TenVadConfig {
        &self.config
    }
}

impl VadStreamHandle for TenVadStream {
    fn push_pcm(&mut self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        if sample_rate != self.config.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "ten_vad: sample rate {sample_rate} does not match required {} Hz; resample upstream",
                self.config.sample_rate
            )));
        }
        if pcm.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "ten_vad: PCM contains a non-finite sample".to_owned(),
            ));
        }
        self.pending_pcm.extend_from_slice(pcm);
        let compute = if self.backend == BackendKind::Cpu {
            None
        } else {
            Some(Compute::for_backend(self.backend, TEN_VAD_HOT_OPS)?)
        };
        let mut probabilities = Vec::new();
        let mut consumed = 0usize;
        while self.pending_pcm.len() - consumed >= self.config.hop_size {
            let frame = &self.pending_pcm[consumed..consumed + self.config.hop_size];
            let features = self.frontend.process_frame(frame)?;
            probabilities.push(match &compute {
                None => network_forward(features, &self.weights.network, &mut self.network_state)?,
                Some(compute) => network_forward_with_ops(
                    features,
                    &self.weights.network,
                    &mut self.network_state,
                    &mut ComputeTenVadOps { compute },
                )?,
            });
            consumed += self.config.hop_size;
        }
        if consumed > 0 {
            self.pending_pcm.drain(..consumed);
        }
        Ok(probabilities)
    }

    fn reset(&mut self) {
        self.pending_pcm.clear();
        self.frontend.reset();
        self.network_state.reset();
    }
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| VokraError::ModelLoad(format!("ten_vad GGUF missing string `{key}`")))?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "ten_vad GGUF `{key}` is `{actual}`, expected `{expected}`"
        )));
    }
    Ok(())
}

fn metadata_u32(file: &GgufFile, key: &str) -> Result<u32> {
    let value = file
        .get(key)
        .and_then(GgufMetadataValue::as_u64)
        .ok_or_else(|| VokraError::ModelLoad(format!("ten_vad GGUF missing unsigned `{key}`")))?;
    u32::try_from(value)
        .map_err(|_| VokraError::ModelLoad(format!("ten_vad GGUF `{key}` does not fit u32")))
}

fn metadata_usize(file: &GgufFile, key: &str) -> Result<usize> {
    usize::try_from(metadata_u32(file, key)?)
        .map_err(|_| VokraError::ModelLoad(format!("ten_vad GGUF `{key}` does not fit usize")))
}

fn tensor(file: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("ten_vad missing tensor `{name}`")))?;
    let actual = info
        .dimensions
        .iter()
        .map(|&dimension| dimension as usize)
        .collect::<Vec<_>>();
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "ten_vad tensor `{name}` has shape {actual:?}, expected {expected:?}"
        )));
    }
    file.tensor_f32(name)
        .map_err(|error| VokraError::ModelLoad(format!("ten_vad tensor `{name}` decode: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    fn zero_file() -> GgufFile {
        let mut builder = GgufBuilder::new();
        builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        builder.add_string(chunks::KEY_MODEL_NAME, NAME);
        builder.add_string(KEY_MODEL_CATEGORY, CATEGORY);
        builder.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);
        builder.add_string(KEY_REVISION, REVISION);
        builder.add_string(KEY_ONNX_SHA256, ONNX_SHA256);
        builder.add_string(KEY_FRONTEND_LICENSE, FRONTEND_LICENSE_SPDX);
        builder.add_u32(KEY_SAMPLE_RATE, 16_000);
        builder.add_u32(KEY_HOP_SIZE, 256);
        builder.add_u32(KEY_N_FEATURES, 41);
        builder.add_u32(KEY_CONTEXT_FRAMES, 3);
        builder.add_u32(KEY_HIDDEN_DIM, 64);
        builder.add_u32(KEY_N_LAYERS, 2);
        vokra_core::stamp_provenance(
            &mut builder,
            LicenseClass::RedistributionForbidden,
            DEFAULT_LICENSE_SPDX,
            Some(NAME),
            Some("synthetic TEN-VAD unit fixture"),
        );
        for (name, shape) in [
            ("ten_vad.conv0.depthwise.weight", &[1, 1, 3, 3][..]),
            ("ten_vad.conv0.pointwise.weight", &[16, 1, 1, 1]),
            ("ten_vad.conv0.pointwise.bias", &[16]),
            ("ten_vad.conv1.depthwise.weight", &[16, 1, 1, 3]),
            ("ten_vad.conv1.pointwise.weight", &[16, 16, 1, 1]),
            ("ten_vad.conv1.pointwise.bias", &[16]),
            ("ten_vad.conv2.depthwise.weight", &[16, 1, 1, 3]),
            ("ten_vad.conv2.pointwise.weight", &[16, 16, 1, 1]),
            ("ten_vad.conv2.pointwise.bias", &[16]),
            ("ten_vad.lstm0.weight_ih", &[1, 256, 80]),
            ("ten_vad.lstm0.weight_hh", &[1, 256, 64]),
            ("ten_vad.lstm0.bias", &[1, 512]),
            ("ten_vad.lstm1.weight_ih", &[1, 256, 64]),
            ("ten_vad.lstm1.weight_hh", &[1, 256, 64]),
            ("ten_vad.lstm1.bias", &[1, 512]),
            ("ten_vad.dense0.weight", &[128, 32]),
            ("ten_vad.dense0.bias", &[32]),
            ("ten_vad.dense1.weight", &[32, 1]),
            ("ten_vad.dense1.bias", &[1]),
        ] {
            let elements = shape.iter().product::<usize>();
            builder
                .add_tensor(
                    name,
                    GgmlType::F32,
                    shape.iter().map(|&dimension| dimension as u64).collect(),
                    vec![0; elements * 4],
                )
                .unwrap();
        }
        GgufFile::parse(builder.to_bytes().unwrap()).unwrap()
    }

    #[test]
    fn strict_binder_and_stream_are_chunk_invariant() {
        let model = TenVad::from_gguf(&zero_file()).unwrap();
        assert_eq!(model.backend(), BackendKind::Cpu);
        assert_eq!(
            TenVad::from_gguf(&zero_file())
                .unwrap()
                .with_backend(BackendKind::Metal)
                .backend(),
            BackendKind::Metal
        );
        assert_eq!(model.tensor_count(), TENSOR_COUNT);
        assert_eq!(
            model.weight_license(),
            LicenseClass::RedistributionForbidden
        );

        let pcm = vec![0.0f32; 512];
        let mut whole = model.open_stream();
        let expected = whole.push_pcm(&pcm, 16_000).unwrap();
        assert_eq!(expected, vec![0.5, 0.5]);

        let mut chunked = model.open_stream();
        let mut actual = Vec::new();
        for chunk in pcm.chunks(173) {
            actual.extend(chunked.push_pcm(chunk, 16_000).unwrap());
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn stream_rejects_sample_rate_and_reset_clears_pending_pcm() {
        let model = TenVad::from_gguf(&zero_file()).unwrap();
        let mut stream = model.open_stream();
        let error = stream.push_pcm(&[0.0; 32], 48_000).unwrap_err();
        assert!(error.to_string().contains("resample upstream"));
        assert!(stream.push_pcm(&[0.0; 100], 16_000).unwrap().is_empty());
        stream.reset();
        assert!(stream.push_pcm(&[0.0; 156], 16_000).unwrap().is_empty());
    }
}
