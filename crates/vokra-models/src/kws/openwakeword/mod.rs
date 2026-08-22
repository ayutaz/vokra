//! openWakeWord (`dscripka/openWakeWord`, Apache-2.0 code) — runtime
//! binder for the `openwakeword_op` converter arch (2026-08-05).
//!
//! # Native v0.5.1 runtime layout
//!
//! ```text
//! PCM (16 kHz mono f32)
//!   -> learned 512-sample real/imag DFT Conv1d (stride 160)
//!   -> 257x32 learned mel projection + dB clipping + `/10 + 2`
//!   -> rolling 76x32 melspec buffer
//!   -> official 20-convolution speech embedding CNN
//!   -> shared 96-d embedding
//!   -> rolling 16x96 embedding window
//!   -> per-wake-word execution-order DNN + final sigmoid
//!   -> per-wake-word probability ∈ [0, 1]
//! ```
//!
//! The native path is selected by
//! `classifier_format = "dnn-relu-sigmoid-v1"`. Older synthetic GGUFs
//! without that additive key retain the two-layer classifier-only
//! compatibility API and its loud `UnsupportedOp` streaming behavior.
//!
//! # `vokra.openwakeword.*` chunk group
//!
//! - `vokra.openwakeword.n_wakewords` (u32): number of per-wake-word MLP
//!   classifiers bound in this GGUF.
//! - `vokra.openwakeword.embedding_dim` (u32): shared embedding width
//!   (96 in the reference release).
//! - `vokra.openwakeword.window_frames` (u32): rolling melspec window
//!   the embedding extractor consumes (76 = ~775 ms at 16 kHz).
//! - `vokra.openwakeword.mel_bins` (u32): melspec width per frame (32
//!   in the reference release).
//! - `vokra.openwakeword.sample_rate` (u32): PCM sample rate (16 000).
//! - `vokra.openwakeword.hop_samples` (u32): analysis hop between
//!   melspec frames (160 = 10 ms at 16 kHz).
//! - `vokra.openwakeword.wakeword_names` (`Array<String>` of length
//!   `n_wakewords`): human-readable per-wake-word names in the order
//!   the classifier weights are indexed.
//!
//! The seven original keys remain required and validated loudly at load time.
//! Native artifacts additionally require `classifier_format`,
//! `classifier_input_frames`, `classifier_layer_counts`, and
//! `predict_chunk_samples`; the fixed v0.5.1 axes are refused if changed.
//! This prevents a differently-shaped fork from silently entering the
//! transcribed topology.
//!
//! [`OpenwakewordConfig::from_gguf`] errors with
//! [`VokraError::ModelLoad`] on any absent key, and
//! [`OpenwakewordConfig::validate`] then refuses a `0` sentinel on every
//! numeric hparam.
//!
//! # Tensors (NOT part of the chunk group above)
//!
//! - `openwakeword.classifier.{i}.linear.{j}.{weight,bias}`:
//!   native execution-order dense layers.
//! - `openwakeword.classifier.{i}.linear{1,2}.{weight,bias}`:
//!   legacy classifier-only compatibility tensors. All are read through
//!   `GgufFile::tensor_f32`, so F32 / F16 / BF16 all bind (BF16 widens
//!   losslessly at load).
//! - `openwakeword.melspec.{dft_real,dft_imag,mel}` and
//!   `openwakeword.embedding.conv.{0..19}.*`: required native frontend.
//!
//! These names carry **no `vokra.` prefix** — they are tensor names, not
//! metadata keys, and the canonical spellings are the
//! [`tensor_classifier_linear1_weight`] family below. Until 2026-08-15
//! this section listed them inside the chunk group with a
//! `vokra.openwakeword.` prefix they have never had, which is a plausible
//! way for a converter author reading these docs to emit tensor names the
//! binder cannot find.
//!
//! # Producer
//!
//! `vokra-cli convert --model openwakeword-op --config <config.json>`
//! (`crates/vokra-convert/src/models/openwakeword_op.rs`). The `--config`
//! side-car is required because `wakeword_names` cannot be derived from
//! the tensors. The v0.5.1 preparation script derives the DNN topology from
//! ONNX graph order and includes the learned frontend weights. The
//! two halves are held together by
//! `crates/vokra-models/tests/openwakeword_convert_bind.rs`, which runs
//! that converter into this binder — added after a 2026-08-15 audit found
//! the converter had never stamped this chunk group at all, so no GGUF it
//! produced could load here.
//!
//! # Wake-word threshold
//!
//! The engine returns raw sigmoid probabilities; the caller thresholds
//! (typically at `0.5`). The CLI `run --model kws.gguf` prints only
//! `(wakeword, prob)` pairs whose probability exceeds `0.5` by default,
//! so no threshold is baked into the engine (upstream tunes per-wake-word).

use std::sync::Arc;

use vokra_core::engines::KwsEngine;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, GgufValueType};
use vokra_core::{Result, VokraError};
use vokra_ops::{
    OpenwakewordClassifierWeights, OpenwakewordConv2dWeights, OpenwakewordDenseWeights,
    OpenwakewordDnnClassifierWeights, OpenwakewordEmbeddingWeights, OpenwakewordMelspecWeights,
    openwakeword_classifier_forward, openwakeword_dnn_classifier_forward,
    openwakeword_melspectrogram,
};

use crate::compute::Compute;

#[cfg(test)]
mod tests;

// ---- arch / provenance constants ----------------------------------------
//
// Mirror of `vokra-convert::models::openwakeword_op::{ARCH, NAME, CATEGORY,
// UPSTREAM_HF}` — kept as duplicated `pub const` so the runtime binder does
// not add a cross-crate dependency edge onto the converter (the sibling
// fsmn_vad / silero_vad convention).

/// Expected `vokra.model.arch` value written by
/// `vokra-convert --model openwakeword-op`.
pub const ARCH: &str = "openwakeword_op";

/// Default `vokra.model.name` value written by the op converter.
pub const DEFAULT_NAME: &str = "openwakeword_op";

/// `vokra.model.category` — VAD/KWS family. Sibling of `silero-vad` /
/// `fsmn-vad`.
pub const CATEGORY: &str = "vad-kws";

// ---- vokra.openwakeword.* metadata keys ---------------------------------

/// GGUF metadata key: number of per-wake-word classifiers (u32).
pub const KEY_N_WAKEWORDS: &str = "vokra.openwakeword.n_wakewords";
/// GGUF metadata key: shared embedding width (u32; upstream = 96).
pub const KEY_EMBEDDING_DIM: &str = "vokra.openwakeword.embedding_dim";
/// GGUF metadata key: rolling melspec window (u32; upstream = 76 frames
/// = ~775 ms at 16 kHz / 10 ms hop).
pub const KEY_WINDOW_FRAMES: &str = "vokra.openwakeword.window_frames";
/// GGUF metadata key: mel-bin count per frame (u32; upstream = 32).
pub const KEY_MEL_BINS: &str = "vokra.openwakeword.mel_bins";
/// GGUF metadata key: PCM sample rate the checkpoint expects (u32 Hz;
/// upstream = 16 000).
pub const KEY_SAMPLE_RATE: &str = "vokra.openwakeword.sample_rate";
/// GGUF metadata key: analysis hop between melspec frames (u32 samples;
/// upstream = 160 = 10 ms at 16 kHz).
pub const KEY_HOP_SAMPLES: &str = "vokra.openwakeword.hop_samples";
/// GGUF metadata key: per-wake-word names (`Array<String>`).
pub const KEY_WAKEWORD_NAMES: &str = "vokra.openwakeword.wakeword_names";
/// GGUF metadata key: classifier tensor topology identifier (string).
pub const KEY_CLASSIFIER_FORMAT: &str = "vokra.openwakeword.classifier_format";
/// GGUF metadata key: rolling embedding frames consumed per prediction (u32).
pub const KEY_CLASSIFIER_INPUT_FRAMES: &str = "vokra.openwakeword.classifier_input_frames";
/// GGUF metadata key: dense-layer count for each wake-word (`Array<U32>`).
pub const KEY_CLASSIFIER_LAYER_COUNTS: &str = "vokra.openwakeword.classifier_layer_counts";
/// GGUF metadata key: PCM samples consumed per prediction (u32).
pub const KEY_PREDICT_CHUNK_SAMPLES: &str = "vokra.openwakeword.predict_chunk_samples";

const CLASSIFIER_FORMAT_DNN: &str = "dnn-relu-sigmoid-v1";
const CLASSIFIER_FORMAT_LEGACY: &str = "legacy-two-layer-v1";

/// Formats a per-wake-word classifier tensor name for the first linear
/// layer weight (row-major `[hidden_dim, embedding_dim]`).
pub fn tensor_classifier_linear1_weight(idx: usize) -> String {
    format!("openwakeword.classifier.{idx}.linear1.weight")
}
/// Formats a per-wake-word classifier tensor name for the first linear
/// layer bias (`[hidden_dim]`).
pub fn tensor_classifier_linear1_bias(idx: usize) -> String {
    format!("openwakeword.classifier.{idx}.linear1.bias")
}
/// Formats a per-wake-word classifier tensor name for the output linear
/// layer weight (row-major `[1, hidden_dim]`).
pub fn tensor_classifier_linear2_weight(idx: usize) -> String {
    format!("openwakeword.classifier.{idx}.linear2.weight")
}
/// Formats a per-wake-word classifier tensor name for the output linear
/// layer bias (`[1]`).
pub fn tensor_classifier_linear2_bias(idx: usize) -> String {
    format!("openwakeword.classifier.{idx}.linear2.bias")
}

/// Formats an execution-order DNN weight tensor name.
pub fn tensor_classifier_dnn_weight(idx: usize, layer: usize) -> String {
    format!("openwakeword.classifier.{idx}.linear.{layer}.weight")
}

/// Formats an execution-order DNN bias tensor name.
pub fn tensor_classifier_dnn_bias(idx: usize, layer: usize) -> String {
    format!("openwakeword.classifier.{idx}.linear.{layer}.bias")
}

/// openWakeWord runtime config (transcribed verbatim from
/// `vokra.openwakeword.*` at load time).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenwakewordConfig {
    /// Number of per-wake-word classifiers.
    pub n_wakewords: usize,
    /// Shared embedding width (96 for the reference release).
    pub embedding_dim: usize,
    /// Rolling melspec window the embedding extractor consumes.
    pub window_frames: usize,
    /// Mel-bin count per frame.
    pub mel_bins: usize,
    /// PCM sample rate the checkpoint expects (Hz).
    pub sample_rate: u32,
    /// Analysis hop between melspec frames (samples).
    pub hop_samples: usize,
    /// Per-wake-word names, one per classifier, in weight-index order.
    pub wakeword_names: Vec<String>,
    /// Classifier tensor topology identifier.
    pub classifier_format: String,
    /// Embedding frames flattened into a native DNN head.
    pub classifier_input_frames: usize,
    /// Dense-layer count for each wake-word.
    pub classifier_layer_counts: Vec<usize>,
    /// PCM samples consumed per emitted prediction.
    pub predict_chunk_samples: usize,
}

impl OpenwakewordConfig {
    /// Validates the config loudly (FR-EX-08). `0`-sentinels are refused
    /// on every hparam, and `wakeword_names.len()` must equal
    /// `n_wakewords`.
    pub fn validate(&self) -> Result<()> {
        for (label, v) in [
            ("n_wakewords", self.n_wakewords as u64),
            ("embedding_dim", self.embedding_dim as u64),
            ("window_frames", self.window_frames as u64),
            ("mel_bins", self.mel_bins as u64),
            ("sample_rate", u64::from(self.sample_rate)),
            ("hop_samples", self.hop_samples as u64),
        ] {
            if v == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "openwakeword config: {label} must be > 0 (got 0 — the GGUF's \
                     vokra.openwakeword.* chunk is missing or malformed)"
                )));
            }
        }
        if self.wakeword_names.len() != self.n_wakewords {
            return Err(VokraError::ModelLoad(format!(
                "openwakeword config: wakeword_names has {} entries, expected \
                 n_wakewords={}",
                self.wakeword_names.len(),
                self.n_wakewords
            )));
        }
        if self.classifier_format != CLASSIFIER_FORMAT_LEGACY
            && self.classifier_format != CLASSIFIER_FORMAT_DNN
        {
            return Err(VokraError::ModelLoad(format!(
                "openwakeword config: unsupported classifier_format `{}`",
                self.classifier_format
            )));
        }
        if self.classifier_input_frames == 0 || self.predict_chunk_samples == 0 {
            return Err(VokraError::InvalidArgument(
                "openwakeword config: classifier_input_frames and predict_chunk_samples must be > 0"
                    .to_owned(),
            ));
        }
        if self.classifier_layer_counts.len() != self.n_wakewords
            || self.classifier_layer_counts.contains(&0)
        {
            return Err(VokraError::ModelLoad(format!(
                "openwakeword config: classifier_layer_counts {:?} does not describe {}/non-empty classifiers",
                self.classifier_layer_counts, self.n_wakewords
            )));
        }
        if self.classifier_format == CLASSIFIER_FORMAT_DNN
            && (self.embedding_dim != 96
                || self.window_frames != 76
                || self.mel_bins != 32
                || self.sample_rate != 16_000
                || self.hop_samples != 160
                || self.classifier_input_frames != 16
                || self.predict_chunk_samples != 1_280)
        {
            return Err(VokraError::ModelLoad(format!(
                "openwakeword native v0.5.1 topology requires embedding/window/mel/rate/hop/input/chunk = 96/76/32/16000/160/16/1280, got {}/{}/{}/{}/{}/{}/{}",
                self.embedding_dim,
                self.window_frames,
                self.mel_bins,
                self.sample_rate,
                self.hop_samples,
                self.classifier_input_frames,
                self.predict_chunk_samples
            )));
        }
        Ok(())
    }

    /// Reads config from `vokra.openwakeword.*` metadata in a parsed
    /// GGUF.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let get_u32 = |k: &str| -> Result<u32> {
            let v = gguf.get(k).and_then(|v| v.as_u64()).ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "openwakeword GGUF missing required u32 metadata `{k}`"
                ))
            })?;
            u32::try_from(v).map_err(|_| {
                VokraError::ModelLoad(format!(
                    "openwakeword GGUF metadata `{k}` = {v} does not fit in u32"
                ))
            })
        };
        let n_wakewords = get_u32(KEY_N_WAKEWORDS)? as usize;
        let embedding_dim = get_u32(KEY_EMBEDDING_DIM)? as usize;
        let window_frames = get_u32(KEY_WINDOW_FRAMES)? as usize;
        let mel_bins = get_u32(KEY_MEL_BINS)? as usize;
        let sample_rate = get_u32(KEY_SAMPLE_RATE)?;
        let hop_samples = get_u32(KEY_HOP_SAMPLES)? as usize;
        let wakeword_names = read_string_array(gguf, KEY_WAKEWORD_NAMES)?;
        let classifier_format = gguf
            .get(KEY_CLASSIFIER_FORMAT)
            .and_then(|value| value.as_str())
            .unwrap_or(CLASSIFIER_FORMAT_LEGACY)
            .to_owned();
        let classifier_input_frames = gguf
            .get(KEY_CLASSIFIER_INPUT_FRAMES)
            .and_then(|value| value.as_u64())
            .unwrap_or(1) as usize;
        let predict_chunk_samples = gguf
            .get(KEY_PREDICT_CHUNK_SAMPLES)
            .and_then(|value| value.as_u64())
            .unwrap_or(1_280) as usize;
        let classifier_layer_counts = if classifier_format == CLASSIFIER_FORMAT_DNN {
            read_u32_array(gguf, KEY_CLASSIFIER_LAYER_COUNTS)?
                .into_iter()
                .map(|value| value as usize)
                .collect()
        } else {
            vec![2; n_wakewords]
        };

        let cfg = Self {
            n_wakewords,
            embedding_dim,
            window_frames,
            mel_bins,
            sample_rate,
            hop_samples,
            wakeword_names,
            classifier_format,
            classifier_input_frames,
            classifier_layer_counts,
            predict_chunk_samples,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

/// Reads a required `Array<String>` metadata chunk into an owned
/// `Vec<String>`, enforcing element-type (FR-EX-08 — refuse the load
/// rather than silently coerce).
fn read_string_array(gguf: &GgufFile, key: &str) -> Result<Vec<String>> {
    let value = gguf.get(key).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "openwakeword GGUF missing required Array<String> metadata `{key}`"
        ))
    })?;
    let arr = value.as_array().ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "openwakeword GGUF metadata `{key}` is not an array (expected Array<String>)"
        ))
    })?;
    if arr.element_type != GgufValueType::String {
        return Err(VokraError::ModelLoad(format!(
            "openwakeword GGUF metadata `{key}` has element_type {:?}, expected String",
            arr.element_type
        )));
    }
    let mut out = Vec::with_capacity(arr.values.len());
    for (i, v) in arr.values.iter().enumerate() {
        match v {
            GgufMetadataValue::String(s) => out.push(s.clone()),
            other => {
                return Err(VokraError::ModelLoad(format!(
                    "openwakeword GGUF metadata `{key}[{i}]` is not String (got {:?})",
                    other.value_type()
                )));
            }
        }
    }
    Ok(out)
}

fn read_u32_array(gguf: &GgufFile, key: &str) -> Result<Vec<u32>> {
    let value = gguf.get(key).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "openwakeword GGUF missing required Array<U32> metadata `{key}`"
        ))
    })?;
    let array = value.as_array().ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "openwakeword GGUF metadata `{key}` is not an array"
        ))
    })?;
    if array.element_type != GgufValueType::U32 {
        return Err(VokraError::ModelLoad(format!(
            "openwakeword GGUF metadata `{key}` has element_type {:?}, expected U32",
            array.element_type
        )));
    }
    array
        .values
        .iter()
        .enumerate()
        .map(|(index, value)| match value {
            GgufMetadataValue::U32(value) => Ok(*value),
            other => Err(VokraError::ModelLoad(format!(
                "openwakeword GGUF metadata `{key}[{index}]` is not U32 (got {:?})",
                other.value_type()
            ))),
        })
        .collect()
}

fn model_load_from_invalid(error: VokraError) -> VokraError {
    match error {
        VokraError::InvalidArgument(message) => VokraError::ModelLoad(message),
        other => other,
    }
}

fn native_embedding_forward(
    weights: &OpenwakewordEmbeddingWeights,
    melspec: &[f32],
) -> Result<Vec<f32>> {
    weights.validate()?;
    if melspec.len() != 76 * 32 {
        return Err(VokraError::InvalidArgument(format!(
            "openwakeword embedding input has {} elements, expected {}",
            melspec.len(),
            76 * 32
        )));
    }
    type EmbeddingLayerLayout = (usize, usize, Option<(usize, usize)>);
    const LAYOUT: [EmbeddingLayerLayout; 20] = [
        (0, 1, None),
        (0, 1, None),
        (0, 0, Some((2, 2))),
        (0, 1, None),
        (0, 0, None),
        (0, 1, None),
        (0, 0, Some((1, 2))),
        (0, 1, None),
        (0, 0, None),
        (0, 1, None),
        (0, 0, Some((2, 2))),
        (0, 1, None),
        (0, 0, None),
        (0, 1, None),
        (0, 0, Some((1, 2))),
        (0, 1, None),
        (0, 0, None),
        (0, 1, None),
        (0, 0, Some((2, 2))),
        (0, 0, None),
    ];
    let compute = Compute::cpu();
    let mut value = melspec.to_vec();
    let (mut height, mut width) = (76usize, 32usize);
    for (index, (conv, (pad_h, pad_w, pool))) in weights.convs.iter().zip(LAYOUT).enumerate() {
        let padded_h = height + 2 * pad_h;
        let padded_w = width + 2 * pad_w;
        let out_h = padded_h - conv.kernel_h + 1;
        let out_w = padded_w - conv.kernel_w + 1;
        let columns = out_h * out_w;
        let patch = conv.in_channels * conv.kernel_h * conv.kernel_w;
        let mut im2col = vec![0.0; patch * columns];
        for input_channel in 0..conv.in_channels {
            for kernel_y in 0..conv.kernel_h {
                for kernel_x in 0..conv.kernel_w {
                    let row = (input_channel * conv.kernel_h + kernel_y) * conv.kernel_w + kernel_x;
                    for out_y in 0..out_h {
                        let source_y = out_y + kernel_y;
                        if source_y < pad_h || source_y - pad_h >= height {
                            continue;
                        }
                        for out_x in 0..out_w {
                            let source_x = out_x + kernel_x;
                            if source_x < pad_w || source_x - pad_w >= width {
                                continue;
                            }
                            im2col[row * columns + out_y * out_w + out_x] =
                                value[(input_channel * height + source_y - pad_h) * width
                                    + source_x
                                    - pad_w];
                        }
                    }
                }
            }
        }
        let mut next = vec![0.0; conv.out_channels * columns];
        compute.gemm_f32(
            conv.out_channels,
            columns,
            patch,
            &conv.weight,
            &im2col,
            None,
            &mut next,
        )?;
        if let Some(bias) = &conv.bias {
            for (channel, plane) in next.chunks_exact_mut(columns).enumerate() {
                for cell in plane {
                    *cell += bias[channel];
                }
            }
        }
        value = next;
        height = out_h;
        width = out_w;
        if index != 19 {
            for cell in &mut value {
                let leaky = if *cell >= 0.0 { *cell } else { *cell * 0.2 };
                *cell = leaky.max(-0.4);
            }
        }
        if let Some((pool_h, pool_w)) = pool {
            let pooled_h = height / pool_h;
            let pooled_w = width / pool_w;
            let mut pooled = vec![f32::NEG_INFINITY; conv.out_channels * pooled_h * pooled_w];
            for channel in 0..conv.out_channels {
                for out_y in 0..pooled_h {
                    for out_x in 0..pooled_w {
                        let mut maximum = f32::NEG_INFINITY;
                        for kernel_y in 0..pool_h {
                            for kernel_x in 0..pool_w {
                                maximum = maximum.max(
                                    value[(channel * height + out_y * pool_h + kernel_y) * width
                                        + out_x * pool_w
                                        + kernel_x],
                                );
                            }
                        }
                        pooled[(channel * pooled_h + out_y) * pooled_w + out_x] = maximum;
                    }
                }
            }
            value = pooled;
            height = pooled_h;
            width = pooled_w;
        }
    }
    if height != 1 || width != 1 || value.len() != 96 {
        return Err(VokraError::InvalidArgument(format!(
            "openwakeword embedding topology ended at [96,{height},{width}]"
        )));
    }
    Ok(value)
}

/// Bound per-wake-word classifier bundle: one
/// [`OpenwakewordClassifierWeights`] per wake-word (name + weights).
#[derive(Debug, Clone)]
pub struct BoundClassifier {
    /// Wake-word display name.
    pub name: String,
    /// Classifier MLP weights.
    pub weights: OpenwakewordClassifierWeights,
}

/// Bound official variable-depth DNN head.
#[derive(Debug, Clone)]
pub struct BoundDnnClassifier {
    /// Wake-word display name.
    pub name: String,
    /// Execution-order DNN weights.
    pub weights: OpenwakewordDnnClassifierWeights,
}

#[derive(Debug, Clone)]
struct NativeOpenwakewordWeights {
    melspec: OpenwakewordMelspecWeights,
    embedding: OpenwakewordEmbeddingWeights,
}

/// Legacy classifier-only embedding extractor compatibility facade.
///
/// The upstream openWakeWord embedding is produced by the frozen Google
/// `speech_embedding` TFLite (Apache-2.0), whose weight tensors are not
/// primary-source-transcribable to Vokra's GGUF layout without the
/// owner-provisioned bundle (mirror of the RMVPE `extract_real` posture
/// per `crates/vokra-models/src/f0/rmvpe.rs`). The extractor therefore
/// only holds a **capability flag**; when
/// [`Self::has_real_embedding_weights`] is `false`,
/// [`Self::forward`] returns
/// [`VokraError::UnsupportedOp`] with owner-flip instructions.
///
/// Native DNN artifacts use the private bound CNN path directly. This public
/// type remains so older callers and classifier-only fixtures retain their
/// explicit loud-partial contract.
#[derive(Debug, Clone)]
pub struct EmbeddingExtractor {
    /// Capability indicator retained for source compatibility.
    pub has_real_embedding_weights: bool,
    /// Emit width (== `OpenwakewordConfig::embedding_dim`, cached for
    /// the forward's dimension check).
    pub embedding_dim: usize,
}

impl EmbeddingExtractor {
    /// Runs the embedding extractor forward on the rolling melspec
    /// window (`[window_frames, mel_bins]`, row-major), returning the
    /// `embedding_dim`-wide vector.
    ///
    /// # Errors
    ///
    /// [`VokraError::UnsupportedOp`] until [`Self::has_real_embedding_weights`]
    /// flips to `true` (owner-provisioned bundle wired). The message
    /// names the env-gate and the parity script for direct owner
    /// action — no fabricated `0.0` on the missing weight (FR-EX-08).
    pub fn forward(&self, melspec_window: &[f32]) -> Result<Vec<f32>> {
        let _ = melspec_window; // Consumed once the real forward binds.
        if !self.has_real_embedding_weights {
            return Err(VokraError::UnsupportedOp(
                "openwakeword embedding extractor: real weight binding is a follow-up \
                 wave (owner-provisioned Google speech_embedding bundle). Set \
                 VOKRA_OPENWAKEWORD_REAL_GGUF and follow the recipe in \
                 crates/vokra-models/tests/parity_openwakeword.rs to flip the switch. \
                 Until then this is a loud partial — no silent fabricated 0.0 \
                 probability (FR-EX-08)."
                    .to_owned(),
            ));
        }
        // The real forward binds here once the owner-provisioned bundle
        // is wired; today the flag never flips, so this arm is
        // unreachable — keep it explicit rather than silently returning
        // an all-zero vector.
        Err(VokraError::UnsupportedOp(
            "openwakeword embedding real forward: skeleton reached — a future wave \
             wires the real Google speech_embedding kernel here (FR-EX-08 honest \
             pending)"
                .to_owned(),
        ))
    }
}

/// openWakeWord session — an immutable shareable weight bundle plus the
/// config it was bound against.
#[derive(Debug)]
pub struct OpenwakewordSession {
    cfg: OpenwakewordConfig,
    classifiers: Arc<Vec<BoundClassifier>>,
    dnn_classifiers: Arc<Vec<BoundDnnClassifier>>,
    embedding: EmbeddingExtractor,
    native_weights: Option<Arc<NativeOpenwakewordWeights>>,
    /// Rolling native melspectrogram buffer.
    melspec_buffer: Vec<f32>,
    /// PCM not yet forming a complete 1280-sample prediction chunk.
    pending_pcm: Vec<f32>,
    raw_context: Vec<f32>,
    embedding_buffer: Vec<f32>,
    predictions_emitted: usize,
}

impl OpenwakewordSession {
    /// Binds the model from a parsed GGUF (FR-LD-01).
    ///
    /// Returns [`VokraError::ModelLoad`] if any required
    /// `vokra.openwakeword.*` chunk is missing, any documented tensor is
    /// absent, or any tensor has the wrong shape / dtype (FR-EX-08 —
    /// no silent reshape).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // Verify the arch tag first so a fsmn-vad / silero-vad GGUF
        // handed to us by mistake fails with a clear message instead of
        // a downstream "missing tensor".
        match gguf
            .get(vokra_core::gguf::chunks::KEY_MODEL_ARCH)
            .and_then(|v| v.as_str())
        {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "openwakeword: GGUF arch is `{other}`, expected `{ARCH}`"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "openwakeword: GGUF is missing `vokra.model.arch` (converter did not \
                     stamp it)"
                        .to_owned(),
                ));
            }
        }

        let cfg = OpenwakewordConfig::from_gguf(gguf)?;
        if cfg.classifier_format == CLASSIFIER_FORMAT_DNN {
            return Self::from_native_gguf(gguf, cfg);
        }

        let mut classifiers = Vec::with_capacity(cfg.n_wakewords);
        for i in 0..cfg.n_wakewords {
            let l1_name = tensor_classifier_linear1_weight(i);
            let l1_bias_name = tensor_classifier_linear1_bias(i);
            let l2_name = tensor_classifier_linear2_weight(i);
            let l2_bias_name = tensor_classifier_linear2_bias(i);

            let linear1_weight = gguf.tensor_f32(&l1_name).map_err(|e| {
                VokraError::ModelLoad(format!("openwakeword: tensor `{l1_name}` load failed: {e}"))
            })?;
            let linear1_bias = gguf.tensor_f32(&l1_bias_name).map_err(|e| {
                VokraError::ModelLoad(format!(
                    "openwakeword: tensor `{l1_bias_name}` load failed: {e}"
                ))
            })?;
            let linear2_weight = gguf.tensor_f32(&l2_name).map_err(|e| {
                VokraError::ModelLoad(format!("openwakeword: tensor `{l2_name}` load failed: {e}"))
            })?;
            let linear2_bias = gguf.tensor_f32(&l2_bias_name).map_err(|e| {
                VokraError::ModelLoad(format!(
                    "openwakeword: tensor `{l2_bias_name}` load failed: {e}"
                ))
            })?;

            if linear1_bias.is_empty() {
                return Err(VokraError::ModelLoad(format!(
                    "openwakeword: tensor `{l1_bias_name}` has zero elements — \
                     hidden_dim must be > 0"
                )));
            }
            let hidden_dim = linear1_bias.len();
            let expected_l1 = hidden_dim * cfg.embedding_dim;
            if linear1_weight.len() != expected_l1 {
                return Err(VokraError::ModelLoad(format!(
                    "openwakeword: tensor `{l1_name}` has {} elements, expected {} \
                     (hidden_dim={} * embedding_dim={})",
                    linear1_weight.len(),
                    expected_l1,
                    hidden_dim,
                    cfg.embedding_dim
                )));
            }

            // Dim-order assertion (defense against silent misforward from
            // a Python bridge that writes the transpose): the docstring
            // in `OpenwakewordClassifierWeights::linear1_weight` pins
            // the layout as row-major `[hidden_dim, embedding_dim]`. A
            // bridge that emits `[embedding_dim, hidden_dim]` would pass
            // the product check above but silently misclassify.
            let expected_l1_dims: [u64; 2] = [hidden_dim as u64, cfg.embedding_dim as u64];
            let l1_info = gguf.tensor_info(&l1_name).ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "openwakeword: tensor `{l1_name}` info unavailable after successful load — \
                     GGUF invariant broken"
                ))
            })?;
            if l1_info.dimensions.as_slice() != expected_l1_dims {
                return Err(VokraError::ModelLoad(format!(
                    "openwakeword: tensor `{l1_name}` dims {:?} — expected [hidden_dim={}, \
                     embedding_dim={}] row-major (see docstring on \
                     OpenwakewordClassifierWeights::linear1_weight)",
                    l1_info.dimensions, hidden_dim, cfg.embedding_dim
                )));
            }
            let expected_l2_dims: [u64; 2] = [1, hidden_dim as u64];
            let l2_info = gguf.tensor_info(&l2_name).ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "openwakeword: tensor `{l2_name}` info unavailable after successful load — \
                     GGUF invariant broken"
                ))
            })?;
            if l2_info.dimensions.as_slice() != expected_l2_dims {
                return Err(VokraError::ModelLoad(format!(
                    "openwakeword: tensor `{l2_name}` dims {:?} — expected [1, hidden_dim={}] \
                     row-major (single binary output class per wake-word)",
                    l2_info.dimensions, hidden_dim
                )));
            }

            let weights = OpenwakewordClassifierWeights {
                embedding_dim: cfg.embedding_dim,
                hidden_dim,
                linear1_weight,
                linear1_bias,
                linear2_weight,
                linear2_bias,
            };
            // Re-validate through the op-side validator so a misbound
            // classifier fails at load time, not first inference.
            weights.validate().map_err(|e| match e {
                VokraError::InvalidArgument(msg) => {
                    VokraError::ModelLoad(format!("openwakeword classifier {i}: {msg}"))
                }
                other => other,
            })?;

            classifiers.push(BoundClassifier {
                name: cfg.wakeword_names[i].clone(),
                weights,
            });
        }

        let embedding = EmbeddingExtractor {
            // Legacy classifier-only GGUFs have no canonical native
            // frontend tensor group and retain the loud-partial contract.
            has_real_embedding_weights: false,
            embedding_dim: cfg.embedding_dim,
        };

        Ok(Self {
            cfg,
            classifiers: Arc::new(classifiers),
            dnn_classifiers: Arc::new(Vec::new()),
            embedding,
            native_weights: None,
            melspec_buffer: Vec::new(),
            pending_pcm: Vec::new(),
            raw_context: Vec::new(),
            embedding_buffer: Vec::new(),
            predictions_emitted: 0,
        })
    }

    fn from_native_gguf(gguf: &GgufFile, cfg: OpenwakewordConfig) -> Result<Self> {
        let bind = |name: &str, expected: &[u64]| -> Result<Vec<f32>> {
            let info = gguf.tensor_info(name).ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "openwakeword: required native tensor `{name}` is missing"
                ))
            })?;
            if info.dimensions.as_slice() != expected {
                return Err(VokraError::ModelLoad(format!(
                    "openwakeword: tensor `{name}` dims {:?}, expected {expected:?}",
                    info.dimensions
                )));
            }
            gguf.tensor_f32(name).map_err(|error| {
                VokraError::ModelLoad(format!(
                    "openwakeword: tensor `{name}` load failed: {error}"
                ))
            })
        };

        let melspec = OpenwakewordMelspecWeights {
            dft_real: bind("openwakeword.melspec.dft_real", &[257, 512])?,
            dft_imag: bind("openwakeword.melspec.dft_imag", &[257, 512])?,
            mel: bind("openwakeword.melspec.mel", &[257, 32])?,
        };
        melspec.validate().map_err(model_load_from_invalid)?;

        const CONVS: [(usize, usize, usize, usize); 20] = [
            (1, 24, 3, 3),
            (24, 24, 1, 3),
            (24, 24, 3, 1),
            (24, 48, 1, 3),
            (48, 48, 3, 1),
            (48, 48, 1, 3),
            (48, 48, 3, 1),
            (48, 72, 1, 3),
            (72, 72, 3, 1),
            (72, 72, 1, 3),
            (72, 72, 3, 1),
            (72, 96, 1, 3),
            (96, 96, 3, 1),
            (96, 96, 1, 3),
            (96, 96, 3, 1),
            (96, 96, 1, 3),
            (96, 96, 3, 1),
            (96, 96, 1, 3),
            (96, 96, 3, 1),
            (96, 96, 3, 1),
        ];
        let mut convs = Vec::with_capacity(CONVS.len());
        for (index, (input, output, kh, kw)) in CONVS.into_iter().enumerate() {
            let weight_name = format!("openwakeword.embedding.conv.{index}.weight");
            let weight = bind(
                &weight_name,
                &[output as u64, input as u64, kh as u64, kw as u64],
            )?;
            let bias_name = format!("openwakeword.embedding.conv.{index}.bias");
            let bias = if index == 19 {
                if gguf.tensor_info(&bias_name).is_some() {
                    return Err(VokraError::ModelLoad(format!(
                        "openwakeword: final embedding convolution must not carry `{bias_name}`"
                    )));
                }
                None
            } else {
                Some(bind(&bias_name, &[output as u64])?)
            };
            convs.push(OpenwakewordConv2dWeights {
                in_channels: input,
                out_channels: output,
                kernel_h: kh,
                kernel_w: kw,
                weight,
                bias,
            });
        }
        let embedding_weights = OpenwakewordEmbeddingWeights { convs };
        embedding_weights
            .validate()
            .map_err(model_load_from_invalid)?;

        let mut dnn_classifiers = Vec::with_capacity(cfg.n_wakewords);
        for (classifier, &layer_count) in cfg.classifier_layer_counts.iter().enumerate() {
            let mut input_dim = cfg.classifier_input_frames * cfg.embedding_dim;
            let mut layers = Vec::with_capacity(layer_count);
            for layer in 0..layer_count {
                let weight_name = tensor_classifier_dnn_weight(classifier, layer);
                let info = gguf.tensor_info(&weight_name).ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "openwakeword: required DNN tensor `{weight_name}` is missing"
                    ))
                })?;
                if info.dimensions.len() != 2
                    || info.dimensions[1] != input_dim as u64
                    || info.dimensions[0] == 0
                {
                    return Err(VokraError::ModelLoad(format!(
                        "openwakeword: tensor `{weight_name}` dims {:?}, expected [out, {input_dim}]",
                        info.dimensions
                    )));
                }
                let output_dim = info.dimensions[0] as usize;
                let bias_name = tensor_classifier_dnn_bias(classifier, layer);
                layers.push(OpenwakewordDenseWeights {
                    input_dim,
                    output_dim,
                    weight: bind(&weight_name, &[output_dim as u64, input_dim as u64])?,
                    bias: bind(&bias_name, &[output_dim as u64])?,
                });
                input_dim = output_dim;
            }
            let weights = OpenwakewordDnnClassifierWeights {
                input_frames: cfg.classifier_input_frames,
                embedding_dim: cfg.embedding_dim,
                layers,
            };
            weights.validate().map_err(model_load_from_invalid)?;
            dnn_classifiers.push(BoundDnnClassifier {
                name: cfg.wakeword_names[classifier].clone(),
                weights,
            });
        }

        let embedding = EmbeddingExtractor {
            has_real_embedding_weights: true,
            embedding_dim: cfg.embedding_dim,
        };
        let melspec_buffer = vec![1.0; cfg.window_frames * cfg.mel_bins];
        let embedding_buffer = vec![0.0; cfg.classifier_input_frames * cfg.embedding_dim];
        Ok(Self {
            cfg,
            classifiers: Arc::new(Vec::new()),
            dnn_classifiers: Arc::new(dnn_classifiers),
            embedding,
            native_weights: Some(Arc::new(NativeOpenwakewordWeights {
                melspec,
                embedding: embedding_weights,
            })),
            melspec_buffer,
            pending_pcm: Vec::new(),
            raw_context: Vec::new(),
            embedding_buffer,
            predictions_emitted: 0,
        })
    }

    /// Opens and binds the model from a GGUF file on disk.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// Returns the checkpoint's config.
    pub fn config(&self) -> &OpenwakewordConfig {
        &self.cfg
    }

    /// Returns the bound per-wake-word classifiers.
    pub fn classifiers(&self) -> &[BoundClassifier] {
        &self.classifiers
    }

    /// Returns official execution-order DNN classifiers, if this is a
    /// native v0.5.1 artifact.
    pub fn dnn_classifiers(&self) -> &[BoundDnnClassifier] {
        &self.dnn_classifiers
    }
}

impl KwsEngine for OpenwakewordSession {
    fn wakeword_names(&self) -> &[String] {
        &self.cfg.wakeword_names
    }

    fn push_pcm16k(&mut self, samples: &[f32]) -> Result<Vec<(String, f32)>> {
        // Sample-rate invariant: the mel front-end is fit against the
        // checkpoint's sample rate. A caller who pushes 8 kHz PCM into
        // a 16 kHz checkpoint would get silently misclassified
        // wake-words — refuse loudly (FR-EX-08).
        if self.cfg.sample_rate != 16_000 {
            return Err(VokraError::InvalidArgument(format!(
                "openwakeword: engine bound at {} Hz — push_pcm16k accepts 16 kHz only \
                 (resample upstream, or open a stream on the matching rate)",
                self.cfg.sample_rate
            )));
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "openwakeword: input PCM contains a non-finite sample".to_owned(),
            ));
        }

        let Some(native) = self.native_weights.clone() else {
            let embedding = vec![0.0f32; self.cfg.embedding_dim];
            self.embedding.forward(&embedding)?;
            unreachable!(
                "embedding.forward must return UnsupportedOp when the real \
                          bundle is unbound (FR-EX-08 honest pending)"
            );
        };
        self.pending_pcm.extend_from_slice(samples);
        let mut output = Vec::new();
        while self.pending_pcm.len() >= self.cfg.predict_chunk_samples {
            let chunk = self
                .pending_pcm
                .drain(..self.cfg.predict_chunk_samples)
                .collect::<Vec<_>>();
            let mut pcm16 = Vec::with_capacity(self.raw_context.len() + chunk.len());
            pcm16.extend_from_slice(&self.raw_context);
            pcm16.extend(
                chunk
                    .iter()
                    .map(|sample| (sample * 32_768.0).round().clamp(-32_768.0, 32_767.0)),
            );

            let mel = openwakeword_melspectrogram(&native.melspec, &pcm16)?;
            self.melspec_buffer.extend_from_slice(&mel);
            let mel_capacity = self.cfg.window_frames * self.cfg.mel_bins;
            if self.melspec_buffer.len() > mel_capacity {
                let excess = self.melspec_buffer.len() - mel_capacity;
                self.melspec_buffer.drain(..excess);
            }
            let embedding = native_embedding_forward(&native.embedding, &self.melspec_buffer)?;
            self.embedding_buffer.extend_from_slice(&embedding);
            let embedding_capacity = self.cfg.classifier_input_frames * self.cfg.embedding_dim;
            if self.embedding_buffer.len() > embedding_capacity {
                let excess = self.embedding_buffer.len() - embedding_capacity;
                self.embedding_buffer.drain(..excess);
            }

            for classifier in self.dnn_classifiers.iter() {
                let mut probability = openwakeword_dnn_classifier_forward(
                    &classifier.weights,
                    &self.embedding_buffer,
                )?;
                if self.predictions_emitted < 5 {
                    probability = 0.0;
                }
                output.push((classifier.name.clone(), probability));
            }
            self.predictions_emitted += 1;
            self.raw_context.clear();
            self.raw_context
                .extend_from_slice(&pcm16[pcm16.len().saturating_sub(3 * self.cfg.hop_samples)..]);
        }
        Ok(output)
    }
}

/// Runs one classifier per wake-word against a single embedding vector
/// and returns `(name, probability)` pairs in wake-word index order.
///
/// This is the same code path [`OpenwakewordSession::push_pcm16k`] will
/// call once the embedding extractor lights up; exposed here so unit
/// tests + downstream code that already holds a real embedding (e.g.
/// through an external Python interop) can exercise the classifier
/// half without waiting on the loud-partial extractor.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] if `embedding.len()` differs from
/// any classifier's `embedding_dim` (all classifiers in one session
/// share the same width by construction, so this is normally a load-
/// time invariant — the check here catches a hand-constructed session).
pub fn classify_embedding(
    classifiers: &[BoundClassifier],
    embedding: &[f32],
) -> Result<Vec<(String, f32)>> {
    let mut out = Vec::with_capacity(classifiers.len());
    for bc in classifiers {
        let p = openwakeword_classifier_forward(&bc.weights, embedding)?;
        out.push((bc.name.clone(), p));
    }
    Ok(out)
}
