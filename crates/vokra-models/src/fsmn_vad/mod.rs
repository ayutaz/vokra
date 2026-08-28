//! Native binding for the released FunASR FSMN-VAD checkpoint.
//!
//! The canonical model is `funasr/fsmn-vad` revision
//! `df20e6b30c653645fa4ff125cacfcabd1020a669`, mirrored from ModelScope
//! `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`.  Its frontend is 16 kHz
//! Kaldi fbank with a Hamming window, five-frame LFR stacking, and the affine
//! transform stored in `am.mvn`.  The encoder has a 248-pdf head; pdf 0 is
//! silence, so the public VAD score is `1 - p(pdf=0)`.

use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, GgufValueType, chunks};
use vokra_core::{Result, VokraError};
use vokra_ops::{
    FsmnBackendOps, FsmnBlockWeights, FsmnEncoderConfig, FsmnStreamState, FsmnVadWeights,
    KaldiFbankOpts, KaldiFbankWindow, fsmn_vad_forward, fsmn_vad_forward_with_ops,
    kaldi_fbank_with_window, softmax_last_axis,
};

use crate::compute::{Compute, HotOp};

#[cfg(test)]
mod tests;

/// Canonical GGUF architecture tag.
pub const ARCH: &str = "fsmn-vad";
/// Canonical Vokra model name.
pub const DEFAULT_NAME: &str = "fsmn-vad-zh-cn-16k-common";
/// Runtime engine category.
pub const CATEGORY: &str = "vad";
/// Official Hugging Face mirror.
pub const UPSTREAM_HF: &str = "funasr/fsmn-vad";
/// Original ModelScope model identifier.
pub const UPSTREAM_MODELSCOPE: &str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch";
/// Pinned model revision.
pub const UPSTREAM_REVISION: &str = "df20e6b30c653645fa4ff125cacfcabd1020a669";
/// SHA-256 of the pinned `model.pt`.
pub const MODEL_SHA256: &str = "b3be75be477f0780277f3bae0fe489f48718f585f3a6e45d7dd1fbb1a4255fc5";
/// SHA-256 of the pinned `am.mvn`.
pub const CMVN_SHA256: &str = "df189fd5f4352df84a0fd464eeab4e450a5e645665d6b38f13c832492261a739";
/// SHA-256 of the pinned `config.yaml`.
pub const CONFIG_SHA256: &str = "486861ca26ddb79081663b6179cb204c6bfae71c52f04aafc48a9e9d8dde1e93";

/// Every learned primitive in the released FSMN encoder. CMVN, fbank, ReLU,
/// residual addition, softmax and stream-history bookkeeping remain host
/// preprocessing/control flow.
const FSMN_VAD_HOT_OPS: &[HotOp] = &[HotOp::Gemv, HotOp::GroupedConv1d];

/// Model-category metadata key.
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
/// Hugging Face provenance key.
pub const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
/// ModelScope provenance key.
pub const KEY_PROVENANCE_UPSTREAM_MODELSCOPE: &str = "vokra.provenance.upstream_modelscope";
/// Upstream revision provenance key.
pub const KEY_PROVENANCE_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
/// Checkpoint-hash metadata key.
pub const KEY_CHECKPOINT_SHA256: &str = "vokra.fsmn_vad.checkpoint_sha256";
/// CMVN-hash metadata key.
pub const KEY_CMVN_SHA256: &str = "vokra.fsmn_vad.cmvn_sha256";
/// Config-hash metadata key.
pub const KEY_CONFIG_SHA256: &str = "vokra.fsmn_vad.config_sha256";

/// Encoder block-count key.
pub const KEY_N_BLOCKS: &str = "vokra.fsmn_vad.n_blocks";
/// LFR input-width key.
pub const KEY_INPUT_DIM: &str = "vokra.fsmn_vad.input_dim";
/// First input-affine width key.
pub const KEY_INPUT_AFFINE_DIM: &str = "vokra.fsmn_vad.input_affine_dim";
/// FSMN block width key.
pub const KEY_LINEAR_DIM: &str = "vokra.fsmn_vad.linear_dim";
/// FSMN projection-width key.
pub const KEY_PROJ_DIM: &str = "vokra.fsmn_vad.proj_dim";
/// Left-memory order key.
pub const KEY_LORDER: &str = "vokra.fsmn_vad.lorder";
/// Right-memory order key.
pub const KEY_RORDER: &str = "vokra.fsmn_vad.rorder";
/// Left-memory stride key.
pub const KEY_LSTRIDE: &str = "vokra.fsmn_vad.lstride";
/// Right-memory stride key.
pub const KEY_RSTRIDE: &str = "vokra.fsmn_vad.rstride";
/// First output-affine width key.
pub const KEY_OUTPUT_AFFINE_DIM: &str = "vokra.fsmn_vad.output_affine_dim";
/// Posterior-pdf count key.
pub const KEY_OUTPUT_DIM: &str = "vokra.fsmn_vad.output_dim";
/// Raw fbank-width key.
pub const KEY_N_MELS: &str = "vokra.fsmn_vad.n_mels";
/// LFR window-width key.
pub const KEY_LFR_M: &str = "vokra.fsmn_vad.lfr_m";
/// LFR stride key.
pub const KEY_LFR_N: &str = "vokra.fsmn_vad.lfr_n";
/// Required sample-rate key.
pub const KEY_SAMPLE_RATE: &str = "vokra.fsmn_vad.sample_rate";
/// `am.mvn` AddShift vector key.
pub const KEY_CMVN_ADD_SHIFT: &str = "vokra.fsmn_vad.cmvn_add_shift";
/// `am.mvn` Rescale vector key.
pub const KEY_CMVN_RESCALE: &str = "vokra.fsmn_vad.cmvn_rescale";

/// First input-affine weight name.
pub const TENSOR_IN_LINEAR1_WEIGHT: &str = "encoder.in_linear1.linear.weight";
/// First input-affine bias name.
pub const TENSOR_IN_LINEAR1_BIAS: &str = "encoder.in_linear1.linear.bias";
/// Second input-affine weight name.
pub const TENSOR_IN_LINEAR2_WEIGHT: &str = "encoder.in_linear2.linear.weight";
/// Second input-affine bias name.
pub const TENSOR_IN_LINEAR2_BIAS: &str = "encoder.in_linear2.linear.bias";
/// First output-affine weight name.
pub const TENSOR_OUT_LINEAR1_WEIGHT: &str = "encoder.out_linear1.linear.weight";
/// First output-affine bias name.
pub const TENSOR_OUT_LINEAR1_BIAS: &str = "encoder.out_linear1.linear.bias";
/// Posterior-head weight name.
pub const TENSOR_OUT_LINEAR2_WEIGHT: &str = "encoder.out_linear2.linear.weight";
/// Posterior-head bias name.
pub const TENSOR_OUT_LINEAR2_BIAS: &str = "encoder.out_linear2.linear.bias";

/// Returns one block's projection tensor name.
pub fn tensor_block_linear_weight(index: usize) -> String {
    format!("encoder.fsmn.{index}.linear.linear.weight")
}

/// Returns one block's causal-memory tensor name.
pub fn tensor_block_memory_weight(index: usize) -> String {
    format!("encoder.fsmn.{index}.fsmn_block.conv_left.weight")
}

/// Returns one block's expansion tensor name.
pub fn tensor_block_affine_weight(index: usize) -> String {
    format!("encoder.fsmn.{index}.affine.linear.weight")
}

/// Returns one block's expansion-bias tensor name.
pub fn tensor_block_affine_bias(index: usize) -> String {
    format!("encoder.fsmn.{index}.affine.linear.bias")
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Complete frontend and encoder geometry read from GGUF.
pub struct FsmnVadConfig {
    /// Encoder geometry.
    pub encoder: FsmnEncoderConfig,
    /// Raw Kaldi fbank width.
    pub n_mels: u32,
    /// Adjacent raw frames stacked per LFR row.
    pub lfr_m: u32,
    /// Raw-frame stride between LFR rows.
    pub lfr_n: u32,
    /// Required PCM sample rate.
    pub sample_rate: u32,
}

impl FsmnVadConfig {
    /// Returns the pinned release geometry.
    pub fn upstream_default() -> Self {
        Self {
            encoder: FsmnEncoderConfig::upstream_default(),
            n_mels: 80,
            lfr_m: 5,
            lfr_n: 1,
            sample_rate: 16_000,
        }
    }

    /// Validates non-zero axes and the LFR input-width invariant.
    pub fn validate(&self) -> Result<()> {
        self.encoder.validate()?;
        for (label, value) in [
            ("n_mels", self.n_mels),
            ("lfr_m", self.lfr_m),
            ("lfr_n", self.lfr_n),
            ("sample_rate", self.sample_rate),
        ] {
            if value == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "fsmn-vad config: {label} must be > 0"
                )));
            }
        }
        let expected = self.lfr_m as usize * self.n_mels as usize;
        if self.encoder.input_dim != expected {
            return Err(VokraError::InvalidArgument(format!(
                "fsmn-vad config: input_dim {} != lfr_m {} * n_mels {} = {expected}",
                self.encoder.input_dim, self.lfr_m, self.n_mels
            )));
        }
        Ok(())
    }

    /// Reads every required geometry value from GGUF metadata.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let get = |key: &str| -> Result<usize> {
            let value = gguf
                .get(key)
                .and_then(|value| value.as_u64())
                .ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "fsmn-vad GGUF missing required u32 metadata `{key}`"
                    ))
                })?;
            usize::try_from(value).map_err(|_| {
                VokraError::ModelLoad(format!("fsmn-vad metadata `{key}` is too large"))
            })
        };
        let cfg = Self {
            encoder: FsmnEncoderConfig {
                n_blocks: get(KEY_N_BLOCKS)?,
                input_dim: get(KEY_INPUT_DIM)?,
                input_affine_dim: get(KEY_INPUT_AFFINE_DIM)?,
                linear_dim: get(KEY_LINEAR_DIM)?,
                proj_dim: get(KEY_PROJ_DIM)?,
                lorder: get(KEY_LORDER)?,
                rorder: get(KEY_RORDER)?,
                lstride: get(KEY_LSTRIDE)?,
                rstride: get(KEY_RSTRIDE)?,
                output_affine_dim: get(KEY_OUTPUT_AFFINE_DIM)?,
                output_dim: get(KEY_OUTPUT_DIM)?,
            },
            n_mels: u32::try_from(get(KEY_N_MELS)?).map_err(|_| {
                VokraError::ModelLoad(format!("fsmn-vad metadata `{KEY_N_MELS}` is too large"))
            })?,
            lfr_m: u32::try_from(get(KEY_LFR_M)?).map_err(|_| {
                VokraError::ModelLoad(format!("fsmn-vad metadata `{KEY_LFR_M}` is too large"))
            })?,
            lfr_n: u32::try_from(get(KEY_LFR_N)?).map_err(|_| {
                VokraError::ModelLoad(format!("fsmn-vad metadata `{KEY_LFR_N}` is too large"))
            })?,
            sample_rate: u32::try_from(get(KEY_SAMPLE_RATE)?).map_err(|_| {
                VokraError::ModelLoad(format!(
                    "fsmn-vad metadata `{KEY_SAMPLE_RATE}` is too large"
                ))
            })?,
        };
        cfg.validate()
            .map_err(|error| VokraError::ModelLoad(error.to_string()))?;
        Ok(cfg)
    }
}

fn require_string(gguf: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = gguf.get(key).and_then(|value| value.as_str());
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "fsmn-vad: metadata `{key}` is {actual:?}, expected `{expected}`"
        )));
    }
    Ok(())
}

fn read_f32_array(gguf: &GgufFile, key: &str, expected: usize) -> Result<Vec<f32>> {
    let array = gguf
        .get(key)
        .and_then(|value| value.as_array())
        .ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "fsmn-vad GGUF missing required Array<F32> metadata `{key}`"
            ))
        })?;
    if array.element_type != GgufValueType::F32 || array.values.len() != expected {
        return Err(VokraError::ModelLoad(format!(
            "fsmn-vad metadata `{key}` must be Array<F32>[{expected}]"
        )));
    }
    array
        .values
        .iter()
        .enumerate()
        .map(|(index, value)| match value {
            GgufMetadataValue::F32(value) if value.is_finite() => Ok(*value),
            _ => Err(VokraError::ModelLoad(format!(
                "fsmn-vad metadata `{key}[{index}]` is not finite F32"
            ))),
        })
        .collect()
}

fn fsmn_vad_fbank_opts(sample_rate: u32, num_mel_bins: usize) -> KaldiFbankOpts {
    KaldiFbankOpts {
        sample_rate,
        num_mel_bins,
        frame_length: sample_rate as usize * 25 / 1000,
        frame_shift: sample_rate as usize * 10 / 1000,
        remove_dc_offset: true,
        preemph_coeff: 0.97,
        low_freq: 20.0,
        high_freq: 0.0,
        use_power: true,
        use_log: true,
        subtract_mean: false,
        round_to_power_of_two: true,
    }
}

#[derive(Debug)]
/// Immutable, shareable FSMN-VAD model bound from canonical GGUF.
pub struct FsmnVadV1 {
    cfg: FsmnVadConfig,
    weights: Arc<FsmnVadWeights>,
    cmvn_add_shift: Arc<Vec<f32>>,
    cmvn_rescale: Arc<Vec<f32>>,
    backend: BackendKind,
}

impl FsmnVadV1 {
    /// Binds a GGUF after validating identity, tensors, and CMVN metadata.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        require_string(gguf, chunks::KEY_MODEL_ARCH, ARCH)?;
        require_string(gguf, KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF)?;
        require_string(
            gguf,
            KEY_PROVENANCE_UPSTREAM_MODELSCOPE,
            UPSTREAM_MODELSCOPE,
        )?;
        require_string(gguf, KEY_PROVENANCE_UPSTREAM_REVISION, UPSTREAM_REVISION)?;
        require_string(gguf, KEY_CHECKPOINT_SHA256, MODEL_SHA256)?;
        require_string(gguf, KEY_CMVN_SHA256, CMVN_SHA256)?;
        require_string(gguf, KEY_CONFIG_SHA256, CONFIG_SHA256)?;
        let cfg = FsmnVadConfig::from_gguf(gguf)?;

        let load = |name: &str, shape: &[u64]| -> Result<Vec<f32>> {
            let info = gguf.tensor_info(name).ok_or_else(|| {
                VokraError::ModelLoad(format!("fsmn-vad: missing tensor `{name}`"))
            })?;
            if info.dimensions != shape {
                return Err(VokraError::ModelLoad(format!(
                    "fsmn-vad: tensor `{name}` has shape {:?}, expected {shape:?}",
                    info.dimensions
                )));
            }
            gguf.tensor_f32(name).map_err(|error| {
                VokraError::ModelLoad(format!("fsmn-vad: tensor `{name}` load failed: {error}"))
            })
        };
        let e = &cfg.encoder;
        let mut blocks = Vec::with_capacity(e.n_blocks);
        for index in 0..e.n_blocks {
            blocks.push(FsmnBlockWeights {
                linear_weight: load(
                    &tensor_block_linear_weight(index),
                    &[e.proj_dim as u64, e.linear_dim as u64],
                )?,
                memory_weight: load(
                    &tensor_block_memory_weight(index),
                    &[e.proj_dim as u64, 1, e.lorder as u64, 1],
                )?,
                affine_weight: load(
                    &tensor_block_affine_weight(index),
                    &[e.linear_dim as u64, e.proj_dim as u64],
                )?,
                affine_bias: load(&tensor_block_affine_bias(index), &[e.linear_dim as u64])?,
            });
        }
        let weights = FsmnVadWeights {
            in_linear1_weight: load(
                TENSOR_IN_LINEAR1_WEIGHT,
                &[e.input_affine_dim as u64, e.input_dim as u64],
            )?,
            in_linear1_bias: load(TENSOR_IN_LINEAR1_BIAS, &[e.input_affine_dim as u64])?,
            in_linear2_weight: load(
                TENSOR_IN_LINEAR2_WEIGHT,
                &[e.linear_dim as u64, e.input_affine_dim as u64],
            )?,
            in_linear2_bias: load(TENSOR_IN_LINEAR2_BIAS, &[e.linear_dim as u64])?,
            blocks,
            out_linear1_weight: load(
                TENSOR_OUT_LINEAR1_WEIGHT,
                &[e.output_affine_dim as u64, e.linear_dim as u64],
            )?,
            out_linear1_bias: load(TENSOR_OUT_LINEAR1_BIAS, &[e.output_affine_dim as u64])?,
            out_linear2_weight: load(
                TENSOR_OUT_LINEAR2_WEIGHT,
                &[e.output_dim as u64, e.output_affine_dim as u64],
            )?,
            out_linear2_bias: load(TENSOR_OUT_LINEAR2_BIAS, &[e.output_dim as u64])?,
        };
        weights
            .validate(e)
            .map_err(|error| VokraError::ModelLoad(error.to_string()))?;
        let cmvn_add_shift = read_f32_array(gguf, KEY_CMVN_ADD_SHIFT, e.input_dim)?;
        let cmvn_rescale = read_f32_array(gguf, KEY_CMVN_RESCALE, e.input_dim)?;
        for (index, value) in cmvn_rescale.iter().enumerate() {
            if *value <= 0.0 {
                return Err(VokraError::ModelLoad(format!(
                    "fsmn-vad: `{KEY_CMVN_RESCALE}[{index}]` must be positive"
                )));
            }
        }
        Ok(Self {
            cfg,
            weights: Arc::new(weights),
            cmvn_add_shift: Arc::new(cmvn_add_shift),
            cmvn_rescale: Arc::new(cmvn_rescale),
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and binds a canonical GGUF file.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Returns the bound model geometry.
    pub fn config(&self) -> &FsmnVadConfig {
        &self.cfg
    }

    /// Selects the backend for every learned projection and causal memory
    /// convolution. Unsupported/unavailable backends fail explicitly when the
    /// first feature chunk runs; they are never replaced with CPU execution.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Returns the selected learned-op backend.
    #[must_use]
    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Runs a fresh-state network forward on normalized LFR features.
    pub fn forward_features(&self, features: &[f32]) -> Result<Vec<f32>> {
        let mut state = FsmnStreamState::zeros(&self.cfg.encoder)?;
        let logits = fsmn_forward_dispatch(
            self.backend,
            &self.cfg.encoder,
            &self.weights,
            features,
            &mut state,
        )?;
        Ok(softmax_last_axis(&logits, self.cfg.encoder.output_dim))
    }
}

impl vokra_core::engines::VadEngine for FsmnVadV1 {
    fn open_stream(&self) -> Box<dyn vokra_core::engines::VadStreamHandle + Send> {
        Box::new(FsmnVadStream::new(
            self.cfg.clone(),
            Arc::clone(&self.weights),
            Arc::clone(&self.cmvn_add_shift),
            Arc::clone(&self.cmvn_rescale),
            self.backend,
        ))
    }
}

/// Stateful PCM frontend and causal-FSMN stream.
pub struct FsmnVadStream {
    cfg: FsmnVadConfig,
    weights: Arc<FsmnVadWeights>,
    state: FsmnStreamState,
    cmvn_add_shift: Arc<Vec<f32>>,
    cmvn_rescale: Arc<Vec<f32>>,
    fbank_opts: KaldiFbankOpts,
    pending_pcm: Vec<f32>,
    pending_frames: Vec<f32>,
    lfr_initialized: bool,
    backend: BackendKind,
}

impl FsmnVadStream {
    fn new(
        cfg: FsmnVadConfig,
        weights: Arc<FsmnVadWeights>,
        cmvn_add_shift: Arc<Vec<f32>>,
        cmvn_rescale: Arc<Vec<f32>>,
        backend: BackendKind,
    ) -> Self {
        let state = FsmnStreamState::zeros(&cfg.encoder).expect("validated FSMN config");
        let fbank_opts = fsmn_vad_fbank_opts(cfg.sample_rate, cfg.n_mels as usize);
        Self {
            cfg,
            weights,
            state,
            cmvn_add_shift,
            cmvn_rescale,
            fbank_opts,
            pending_pcm: Vec::new(),
            pending_frames: Vec::new(),
            lfr_initialized: false,
            backend,
        }
    }

    /// Runs normalized LFR features and returns `1 - p(silence)` per row.
    pub fn push_features(&mut self, features: &[f32]) -> Result<Vec<f32>> {
        let logits = fsmn_forward_dispatch(
            self.backend,
            &self.cfg.encoder,
            &self.weights,
            features,
            &mut self.state,
        )?;
        let width = self.cfg.encoder.output_dim;
        let probabilities = softmax_last_axis(&logits, width);
        Ok(probabilities
            .chunks_exact(width)
            .map(|row| 1.0 - row[0])
            .collect())
    }

    fn drain_pcm_into_frames(&mut self) -> Result<()> {
        let frame_length = self.fbank_opts.frame_length;
        if self.pending_pcm.len() < frame_length {
            return Ok(());
        }
        // FunASR's WavFrontend converts normalized PCM back to the Kaldi
        // 16-bit amplitude domain before calling torchaudio's fbank.
        let scaled = self
            .pending_pcm
            .iter()
            .map(|sample| sample * 32768.0)
            .collect::<Vec<_>>();
        let (features, frames) =
            kaldi_fbank_with_window(&scaled, &self.fbank_opts, KaldiFbankWindow::Hamming)?;
        if frames == 0 {
            return Ok(());
        }
        let n_mels = self.cfg.n_mels as usize;
        if !self.lfr_initialized {
            let left_padding = (self.cfg.lfr_m as usize - 1) / 2;
            for _ in 0..left_padding {
                self.pending_frames.extend_from_slice(&features[..n_mels]);
            }
            self.lfr_initialized = true;
        }
        self.pending_frames.extend_from_slice(&features);
        let consumed = frames
            .checked_mul(self.fbank_opts.frame_shift)
            .ok_or_else(|| VokraError::InvalidArgument("fsmn-vad PCM overflow".to_owned()))?;
        self.pending_pcm.drain(..consumed);
        Ok(())
    }

    fn drain_frames_into_lfr(&mut self) -> Vec<f32> {
        let n_mels = self.cfg.n_mels as usize;
        let lfr_m = self.cfg.lfr_m as usize;
        let lfr_n = self.cfg.lfr_n as usize;
        let mut output = Vec::new();
        while self.pending_frames.len() / n_mels >= lfr_m {
            output.extend_from_slice(&self.pending_frames[..lfr_m * n_mels]);
            self.pending_frames.drain(..lfr_n * n_mels);
        }
        output
    }

    fn apply_cmvn(&self, features: &mut [f32]) {
        let width = self.cfg.encoder.input_dim;
        for row in features.chunks_exact_mut(width) {
            for ((value, shift), scale) in row
                .iter_mut()
                .zip(self.cmvn_add_shift.iter())
                .zip(self.cmvn_rescale.iter())
            {
                *value = (*value + shift) * scale;
            }
        }
    }
}

fn fsmn_forward_dispatch(
    backend: BackendKind,
    cfg: &FsmnEncoderConfig,
    weights: &FsmnVadWeights,
    features: &[f32],
    state: &mut FsmnStreamState,
) -> Result<Vec<f32>> {
    if backend == BackendKind::Cpu {
        return fsmn_vad_forward(cfg, weights, features, state);
    }
    let compute = Compute::for_backend(backend, FSMN_VAD_HOT_OPS)?;
    fsmn_vad_forward_with_ops(
        cfg,
        weights,
        features,
        state,
        &mut ComputeFsmnOps { compute: &compute },
    )
}

struct ComputeFsmnOps<'a> {
    compute: &'a Compute,
}

impl FsmnBackendOps for ComputeFsmnOps<'_> {
    fn linear(
        &mut self,
        input: &[f32],
        rows: usize,
        input_dim: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
        output_dim: usize,
    ) -> Result<Vec<f32>> {
        let input_len = rows.checked_mul(input_dim).ok_or_else(|| {
            VokraError::InvalidArgument("fsmn-vad linear input extent overflow".to_owned())
        })?;
        let weight_len = output_dim.checked_mul(input_dim).ok_or_else(|| {
            VokraError::InvalidArgument("fsmn-vad linear weight extent overflow".to_owned())
        })?;
        if input.len() != input_len
            || weight.len() != weight_len
            || bias.is_some_and(|values| values.len() != output_dim)
        {
            return Err(VokraError::InvalidArgument(format!(
                "fsmn-vad linear shape mismatch: input={} expected={input_len}, weight={} expected={weight_len}, bias={} expected=0-or-{output_dim}",
                input.len(),
                weight.len(),
                bias.map_or(0, <[f32]>::len)
            )));
        }
        let output_len = rows.checked_mul(output_dim).ok_or_else(|| {
            VokraError::InvalidArgument("fsmn-vad linear output extent overflow".to_owned())
        })?;
        let mut output = vec![0.0f32; output_len];
        for row in 0..rows {
            self.compute.gemv_f32(
                output_dim,
                input_dim,
                weight,
                &input[row * input_dim..(row + 1) * input_dim],
                bias,
                &mut output[row * output_dim..(row + 1) * output_dim],
            )?;
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
        if cfg.lstride != 1 {
            return Err(VokraError::UnsupportedOp(format!(
                "fsmn-vad Metal causal memory requires the released lstride=1 contract, got {}; no CPU fallback is performed",
                cfg.lstride
            )));
        }
        let history_frames = cfg.left_history_frames();
        let projected_len = frames.checked_mul(cfg.proj_dim).ok_or_else(|| {
            VokraError::InvalidArgument("fsmn-vad projected extent overflow".to_owned())
        })?;
        let history_len = history_frames.checked_mul(cfg.proj_dim).ok_or_else(|| {
            VokraError::InvalidArgument("fsmn-vad history extent overflow".to_owned())
        })?;
        let weight_len = cfg.proj_dim.checked_mul(cfg.lorder).ok_or_else(|| {
            VokraError::InvalidArgument("fsmn-vad memory-weight extent overflow".to_owned())
        })?;
        if projected.len() != projected_len
            || history.len() != history_len
            || weights.len() != weight_len
        {
            return Err(VokraError::InvalidArgument(format!(
                "fsmn-vad causal-memory shape mismatch: projected={} expected={projected_len}, history={} expected={history_len}, weights={} expected={weight_len}",
                projected.len(),
                history.len(),
                weights.len()
            )));
        }

        let in_len = history_frames + frames;
        let mut combined = Vec::with_capacity((history_frames + frames) * cfg.proj_dim);
        combined.extend_from_slice(history);
        combined.extend_from_slice(projected);

        // Grouped Conv1D consumes [channel, time], while the public FSMN op
        // keeps [time, channel] to match the checkpoint/reference fixture.
        let mut channel_major = vec![0.0f32; cfg.proj_dim * in_len];
        for channel in 0..cfg.proj_dim {
            for frame in 0..in_len {
                channel_major[channel * in_len + frame] = combined[frame * cfg.proj_dim + channel];
            }
        }
        let mut convolved = vec![0.0f32; cfg.proj_dim * frames];
        self.compute.grouped_conv1d_f32(
            &channel_major,
            cfg.proj_dim,
            in_len,
            weights,
            cfg.proj_dim,
            cfg.lorder,
            None,
            1,
            0,
            cfg.proj_dim,
            &mut convolved,
        )?;

        let mut output = projected.to_vec();
        for frame in 0..frames {
            for channel in 0..cfg.proj_dim {
                output[frame * cfg.proj_dim + channel] += convolved[channel * frames + frame];
            }
        }

        history.clear();
        if history_len > 0 {
            history.extend_from_slice(&combined[combined.len() - history_len..]);
        }
        Ok(output)
    }
}

impl vokra_core::engines::VadStreamHandle for FsmnVadStream {
    fn push_pcm(&mut self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        if sample_rate != self.cfg.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "fsmn-vad: sample rate mismatch: got {sample_rate}, expected {}",
                self.cfg.sample_rate
            )));
        }
        self.pending_pcm.extend_from_slice(pcm);
        self.drain_pcm_into_frames()?;
        let mut features = self.drain_frames_into_lfr();
        if features.is_empty() {
            return Ok(Vec::new());
        }
        self.apply_cmvn(&mut features);
        self.push_features(&features)
    }

    fn reset(&mut self) {
        self.state.reset();
        self.pending_pcm.clear();
        self.pending_frames.clear();
        self.lfr_initialized = false;
    }
}
