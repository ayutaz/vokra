//! Native Audiobox Aesthetics quality scorer for CPU and Metal.
//!
//! The runtime binds the immutable `facebook/audiobox-aesthetics` WavLM Base
//! checkpoint and its four independent CE / CU / PC / PQ projection heads.
//! Every learned hot operation runs through [`Compute`]. WavLM's relative
//! position bucket lookup and GRU-style gate remain host-side tensor-layout
//! glue; selecting an unsupported backend still fails when [`Compute`] is
//! created and never falls back to CPU.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue, GgufValueType, chunks};
use vokra_core::{LicenseClass, Result, VokraError};
use vokra_ops::{
    ConvLayerAttrs, ConvLayerWeights, Norm, WaveformFrontendAttrs, WaveformFrontendWeights,
};

use crate::align::charsiu::{
    CharsiuConfig, CharsiuFeatureProjection, CharsiuPosConv,
    feature_projection_forward_with_compute, layer_norm_with_compute_inplace,
    linear_forward_with_compute, positional_conv_forward_with_compute,
};
use crate::compute::{Compute, HotOp};
use crate::wav2vec2_ctc::{fold_weight_norm_dim2, waveform_frontend_with_compute};

/// Converter/runtime architecture handshake.
pub const ARCH: &str = "audiobox-aesthetics";
/// Canonical public GGUF name.
pub const NAME: &str = "audiobox-aesthetics";
/// Upstream Hugging Face repository.
pub const UPSTREAM_HF: &str = "facebook/audiobox-aesthetics";
/// Immutable upstream checkpoint revision.
pub const CHECKPOINT_REVISION: &str = "9b1dd8e5df9af7216e836a98974fe3b82c56ded6";
/// Immutable source revision used to transcribe the forward.
pub const SOURCE_REVISION: &str = "2618e9d451b456e9328b39495b5e6234678aa550";
/// Ordered upstream output axes.
pub const AXES: [&str; 4] = ["CE", "CU", "PC", "PQ"];

/// Required mono PCM sample rate.
pub const SAMPLE_RATE: u32 = 16_000;
const WINDOW_SAMPLES: usize = 160_000;
const HOP_SAMPLES: usize = 160_000;
const FEATURE_DIM: usize = 512;
const HIDDEN_SIZE: usize = 768;
const FFN_DIM: usize = 3_072;
const N_LAYER: usize = 12;
const N_HEAD: usize = 12;
const HEAD_DIM: usize = HIDDEN_SIZE / N_HEAD;
const POS_CONV_KERNEL: usize = 128;
const POS_CONV_GROUPS: usize = 16;
const NUM_BUCKETS: usize = 320;
const MAX_DISTANCE: usize = 800;
const NTH_LAYER: usize = 13;
const PROJ_NUM_LAYER: usize = 5;
const OUTPUT_DIM: usize = 1;
const LAYER_NORM_EPS: f32 = 1.0e-5;
const TENSOR_COUNT: usize = 324;

const CATEGORY: &str = "classification";
const TARGET_MEANS: [f32; 4] = [5.06865, 5.73633, 3.18591, 6.57505];
const TARGET_STDS: [f32; 4] = [1.93029, 1.75669, 1.86637, 1.51466];
const L2_EPS: f32 = 1.0e-12;
const CONV_DIM: [usize; 7] = [FEATURE_DIM; 7];
const CONV_KERNEL: [usize; 7] = [10, 3, 3, 3, 3, 2, 2];
const CONV_STRIDE: [usize; 7] = [5, 2, 2, 2, 2, 2, 2];

const KEY_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const PREFIX: &str = "vokra.audiobox_aesthetics";
const KEY_CHECKPOINT_REVISION: &str = "vokra.audiobox_aesthetics.checkpoint_revision";
const KEY_SOURCE_REVISION: &str = "vokra.audiobox_aesthetics.source_revision";
const KEY_SAMPLE_RATE: &str = "vokra.audiobox_aesthetics.sample_rate";
const KEY_WINDOW_SAMPLES: &str = "vokra.audiobox_aesthetics.window_samples";
const KEY_HOP_SAMPLES: &str = "vokra.audiobox_aesthetics.hop_samples";
const KEY_FEATURE_DIM: &str = "vokra.audiobox_aesthetics.feature_dim";
const KEY_HIDDEN_SIZE: &str = "vokra.audiobox_aesthetics.hidden_size";
const KEY_FFN_DIM: &str = "vokra.audiobox_aesthetics.ffn_dim";
const KEY_N_LAYER: &str = "vokra.audiobox_aesthetics.n_layer";
const KEY_N_HEAD: &str = "vokra.audiobox_aesthetics.n_head";
const KEY_POS_CONV_KERNEL: &str = "vokra.audiobox_aesthetics.pos_conv_kernel";
const KEY_POS_CONV_GROUPS: &str = "vokra.audiobox_aesthetics.pos_conv_groups";
const KEY_NUM_BUCKETS: &str = "vokra.audiobox_aesthetics.num_buckets";
const KEY_MAX_DISTANCE: &str = "vokra.audiobox_aesthetics.max_distance";
const KEY_NTH_LAYER: &str = "vokra.audiobox_aesthetics.nth_layer";
const KEY_PROJ_NUM_LAYER: &str = "vokra.audiobox_aesthetics.proj_num_layer";
const KEY_OUTPUT_DIM: &str = "vokra.audiobox_aesthetics.output_dim";
const KEY_NORMALIZE_EMBED: &str = "vokra.audiobox_aesthetics.normalize_embed";
const KEY_WEIGHTED_LAYER_SUM: &str = "vokra.audiobox_aesthetics.use_weighted_layer_sum";
const KEY_LAYER_NORM_EPS: &str = "vokra.audiobox_aesthetics.layer_norm_eps";
const KEY_AXES: &str = "vokra.audiobox_aesthetics.axes";
const KEY_TARGET_MEANS: &str = "vokra.audiobox_aesthetics.target_means";
const KEY_TARGET_STDS: &str = "vokra.audiobox_aesthetics.target_stds";

const CONTRACT_KEYS: &[&str] = &[
    KEY_CHECKPOINT_REVISION,
    KEY_SOURCE_REVISION,
    KEY_SAMPLE_RATE,
    KEY_WINDOW_SAMPLES,
    KEY_HOP_SAMPLES,
    KEY_FEATURE_DIM,
    KEY_HIDDEN_SIZE,
    KEY_FFN_DIM,
    KEY_N_LAYER,
    KEY_N_HEAD,
    KEY_POS_CONV_KERNEL,
    KEY_POS_CONV_GROUPS,
    KEY_NUM_BUCKETS,
    KEY_MAX_DISTANCE,
    KEY_NTH_LAYER,
    KEY_PROJ_NUM_LAYER,
    KEY_OUTPUT_DIM,
    KEY_NORMALIZE_EMBED,
    KEY_WEIGHTED_LAYER_SUM,
    KEY_LAYER_NORM_EPS,
    KEY_AXES,
    KEY_TARGET_MEANS,
    KEY_TARGET_STDS,
];

/// Complete learned-op registry for both CPU and Metal.
pub const AUDIOBOX_AESTHETICS_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Gelu,
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
];

/// Fixed topology resolved from the audited checkpoint contract.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioboxAestheticsConfig {
    /// Required input sample rate.
    pub sample_rate: u32,
    /// Non-overlapping upstream inference-window length in PCM samples.
    pub window_samples: usize,
    /// Upstream window hop in PCM samples.
    pub hop_samples: usize,
    /// WavLM residual width.
    pub hidden_size: usize,
    /// WavLM encoder block count.
    pub n_layer: usize,
    /// WavLM attention-head count.
    pub n_head: usize,
    /// WavLM feed-forward inner width.
    pub intermediate_size: usize,
    /// LayerNorm epsilon used by the WavLM encoder and projection heads.
    pub layer_norm_eps: f32,
}

impl Default for AudioboxAestheticsConfig {
    fn default() -> Self {
        Self {
            sample_rate: SAMPLE_RATE,
            window_samples: WINDOW_SAMPLES,
            hop_samples: HOP_SAMPLES,
            hidden_size: HIDDEN_SIZE,
            n_layer: N_LAYER,
            n_head: N_HEAD,
            intermediate_size: FFN_DIM,
            layer_norm_eps: LAYER_NORM_EPS,
        }
    }
}

/// Four inverse-normalized Audiobox quality ratings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioboxScores {
    values: [f32; 4],
}

impl AudioboxScores {
    /// Returns values in the official CE / CU / PC / PQ order.
    #[must_use]
    pub const fn as_array(self) -> [f32; 4] {
        self.values
    }

    /// Content Enjoyment score.
    #[must_use]
    pub const fn content_enjoyment(self) -> f32 {
        self.values[0]
    }

    /// Content Usefulness score.
    #[must_use]
    pub const fn content_usefulness(self) -> f32 {
        self.values[1]
    }

    /// Production Complexity score.
    #[must_use]
    pub const fn production_complexity(self) -> f32 {
        self.values[2]
    }

    /// Production Quality score.
    #[must_use]
    pub const fn production_quality(self) -> f32 {
        self.values[3]
    }
}

#[derive(Debug, Clone)]
struct WavLmBlock {
    q_w: Vec<f32>,
    q_b: Vec<f32>,
    k_w: Vec<f32>,
    k_b: Vec<f32>,
    v_w: Vec<f32>,
    v_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    grep_w: Vec<f32>,
    grep_b: Vec<f32>,
    grep_a: Vec<f32>,
    self_attn_norm_gamma: Vec<f32>,
    self_attn_norm_beta: Vec<f32>,
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: Vec<f32>,
    final_norm_gamma: Vec<f32>,
    final_norm_beta: Vec<f32>,
}

#[derive(Debug, Clone)]
struct ProjectionBlock {
    weight: Vec<f32>,
    bias: Vec<f32>,
    norm_gamma: Vec<f32>,
    norm_beta: Vec<f32>,
}

#[derive(Debug, Clone)]
struct AxisHead {
    layer_weights: Vec<f32>,
    blocks: Vec<ProjectionBlock>,
    output_weight: Vec<f32>,
    output_bias: Vec<f32>,
}

#[derive(Debug, Clone)]
struct AudioboxWeights {
    stem_attrs: WaveformFrontendAttrs,
    stem: WaveformFrontendWeights,
    feature_projection: CharsiuFeatureProjection,
    position: CharsiuPosConv,
    encoder_norm_gamma: Vec<f32>,
    encoder_norm_beta: Vec<f32>,
    relative_attention_bias: Vec<f32>,
    blocks: Vec<WavLmBlock>,
    heads: Vec<AxisHead>,
}

/// Strict native Audiobox Aesthetics engine.
#[derive(Debug, Clone)]
pub struct AudioboxAesthetics {
    config: AudioboxAestheticsConfig,
    weights: AudioboxWeights,
    backend: BackendKind,
    legacy_metadata_repaired: bool,
}

impl AudioboxAesthetics {
    /// Opens and binds the canonical public GGUF.
    pub fn from_gguf(path: impl AsRef<Path>) -> Result<Self> {
        let file = GgufFile::open(path)?;
        Self::from_file(&file)
    }

    /// Strictly binds an already-open GGUF.
    pub fn from_file(file: &GgufFile) -> Result<Self> {
        require_string(file, chunks::KEY_MODEL_ARCH, ARCH)?;
        require_string(file, chunks::KEY_MODEL_NAME, NAME)?;
        require_string(file, KEY_CATEGORY, CATEGORY)?;
        require_string(file, KEY_UPSTREAM_HF, UPSTREAM_HF)?;
        let license = vokra_core::resolve_license_class(file);
        if license.class != LicenseClass::AttributionRequired {
            return Err(VokraError::ModelLoad(format!(
                "{ARCH}: weight license resolves to {}, expected AttributionRequired for CC-BY-4.0",
                license.class.as_str()
            )));
        }

        validate_manifest(file)?;
        let present = CONTRACT_KEYS
            .iter()
            .filter(|key| file.get(key).is_some())
            .count();
        let legacy_metadata_repaired = match present {
            0 => true,
            count if count == CONTRACT_KEYS.len() => {
                validate_contract_metadata(file)?;
                false
            }
            count => {
                return Err(VokraError::ModelLoad(format!(
                    "{ARCH}: partial `{PREFIX}.*` metadata ({count}/{} keys); refusing topology repair",
                    CONTRACT_KEYS.len()
                )));
            }
        };

        let stem_attrs = WaveformFrontendAttrs {
            in_channels: 1,
            layers: CONV_DIM
                .iter()
                .zip(CONV_KERNEL)
                .zip(CONV_STRIDE)
                .map(|((&out_channels, kernel), stride)| ConvLayerAttrs {
                    out_channels,
                    kernel,
                    stride,
                })
                .collect(),
            norm: Norm::GroupFirstOnly,
            conv_bias: false,
        };
        let mut stem_layers = Vec::with_capacity(CONV_DIM.len());
        let mut in_channels = 1;
        for layer in 0..CONV_DIM.len() {
            stem_layers.push(ConvLayerWeights {
                conv_w: tensor(
                    file,
                    &format!("wavlm_model.feature_extractor.conv_layers.{layer}.0.weight"),
                    &[CONV_DIM[layer], in_channels, CONV_KERNEL[layer]],
                )?,
                conv_b: Vec::new(),
                norm_gamma: (layer == 0)
                    .then(|| {
                        tensor(
                            file,
                            "wavlm_model.feature_extractor.conv_layers.0.2.weight",
                            &[FEATURE_DIM],
                        )
                    })
                    .transpose()?,
                norm_beta: (layer == 0)
                    .then(|| {
                        tensor(
                            file,
                            "wavlm_model.feature_extractor.conv_layers.0.2.bias",
                            &[FEATURE_DIM],
                        )
                    })
                    .transpose()?,
            });
            in_channels = CONV_DIM[layer];
        }
        let stem = WaveformFrontendWeights {
            layers: stem_layers,
        };
        stem.validate(&stem_attrs)?;

        let feature_projection = CharsiuFeatureProjection {
            norm_gamma: Some(tensor(
                file,
                "wavlm_model.layer_norm.weight",
                &[FEATURE_DIM],
            )?),
            norm_beta: Some(tensor(file, "wavlm_model.layer_norm.bias", &[FEATURE_DIM])?),
            linear_w: tensor(
                file,
                "wavlm_model.post_extract_proj.weight",
                &[HIDDEN_SIZE, FEATURE_DIM],
            )?,
            linear_b: tensor(file, "wavlm_model.post_extract_proj.bias", &[HIDDEN_SIZE])?,
        };
        let position_v = tensor(
            file,
            "wavlm_model.encoder.pos_conv.0.weight_v",
            &[HIDDEN_SIZE, HIDDEN_SIZE / POS_CONV_GROUPS, POS_CONV_KERNEL],
        )?;
        let position_g = tensor(
            file,
            "wavlm_model.encoder.pos_conv.0.weight_g",
            &[1, 1, POS_CONV_KERNEL],
        )?;
        let position = CharsiuPosConv {
            weight: fold_weight_norm_dim2(
                &position_g,
                &position_v,
                HIDDEN_SIZE,
                HIDDEN_SIZE / POS_CONV_GROUPS,
                POS_CONV_KERNEL,
            )?,
            bias: tensor(file, "wavlm_model.encoder.pos_conv.0.bias", &[HIDDEN_SIZE])?,
        };
        let encoder_norm_gamma = tensor(
            file,
            "wavlm_model.encoder.layer_norm.weight",
            &[HIDDEN_SIZE],
        )?;
        let encoder_norm_beta =
            tensor(file, "wavlm_model.encoder.layer_norm.bias", &[HIDDEN_SIZE])?;
        let relative_attention_bias = tensor(
            file,
            "wavlm_model.encoder.layers.0.self_attn.relative_attention_bias.weight",
            &[NUM_BUCKETS, N_HEAD],
        )?;

        let mut blocks = Vec::with_capacity(N_LAYER);
        for layer in 0..N_LAYER {
            blocks.push(bind_block(file, layer)?);
        }
        let heads = AXES
            .iter()
            .map(|axis| bind_axis_head(file, axis))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            config: AudioboxAestheticsConfig::default(),
            weights: AudioboxWeights {
                stem_attrs,
                stem,
                feature_projection,
                position,
                encoder_norm_gamma,
                encoder_norm_beta,
                relative_attention_bias,
                blocks,
                heads,
            },
            backend: BackendKind::Cpu,
            legacy_metadata_repaired,
        })
    }

    /// Selects the backend used for every learned hot operation.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Returns the selected compute backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Returns the strict fixed topology.
    #[must_use]
    pub const fn config(&self) -> &AudioboxAestheticsConfig {
        &self.config
    }

    /// Whether the exact historical public GGUF lacked the topology group and
    /// was repaired only after its complete 324-tensor manifest matched.
    #[must_use]
    pub const fn legacy_metadata_repaired(&self) -> bool {
        self.legacy_metadata_repaired
    }

    /// Scores arbitrary-length mono PCM at exactly 16 kHz.
    pub fn score_pcm(&self, pcm: &[f32], sample_rate: u32) -> Result<AudioboxScores> {
        validate_pcm(pcm, sample_rate)?;
        let compute = Compute::for_backend(self.backend, AUDIOBOX_AESTHETICS_HOT_OPS)?;
        let mut weighted = [0.0f32; 4];
        let mut total_weight = 0.0f32;
        for chunk in pcm.chunks(HOP_SAMPLES) {
            let valid_samples = chunk.len().min(WINDOW_SAMPLES);
            let mut window = vec![0.0f32; WINDOW_SAMPLES];
            window[..valid_samples].copy_from_slice(&chunk[..valid_samples]);
            let raw = self.score_window(&window, valid_samples, &compute)?;
            let weight = valid_samples as f32 / WINDOW_SAMPLES as f32;
            for axis in 0..4 {
                weighted[axis] += raw[axis] * weight;
            }
            total_weight += weight;
        }
        if total_weight == 0.0 {
            return Err(VokraError::InvalidArgument(
                "audiobox-aesthetics: PCM is empty".to_owned(),
            ));
        }
        let mut values = [0.0f32; 4];
        for axis in 0..4 {
            values[axis] = weighted[axis] / total_weight;
        }
        reject_non_finite("Audiobox final scores", &values)?;
        Ok(AudioboxScores { values })
    }

    fn score_window(
        &self,
        window: &[f32],
        valid_samples: usize,
        compute: &Compute,
    ) -> Result<[f32; 4]> {
        let features = waveform_frontend_with_compute(
            window,
            &self.weights.stem_attrs,
            &self.weights.stem,
            compute,
        )?;
        let frames = features.len() / FEATURE_DIM;
        if frames == 0 {
            return Err(VokraError::InvalidArgument(
                "audiobox-aesthetics: waveform stem produced no frames".to_owned(),
            ));
        }
        let valid_frames = collapsed_valid_frames(valid_samples, frames, WINDOW_SAMPLES);
        let mut hidden = feature_projection_forward_with_compute(
            &features,
            frames,
            FEATURE_DIM,
            &self.weights.feature_projection,
            HIDDEN_SIZE,
            true,
            LAYER_NORM_EPS,
            compute,
        )?;
        for value in &mut hidden[valid_frames * HIDDEN_SIZE..] {
            *value = 0.0;
        }

        let position = positional_conv_forward_with_compute(
            &hidden,
            frames,
            &encoder_config(),
            &self.weights.position,
            compute,
        )?;
        for (value, positional) in hidden.iter_mut().zip(position) {
            *value += positional;
        }
        layer_norm_with_compute_inplace(
            &mut hidden,
            frames,
            HIDDEN_SIZE,
            &self.weights.encoder_norm_gamma,
            &self.weights.encoder_norm_beta,
            LAYER_NORM_EPS,
            compute,
        )?;

        let mut normalized_layer_weights = Vec::with_capacity(4);
        let mut accumulated = (0..4)
            .map(|_| vec![0.0f32; hidden.len()])
            .collect::<Vec<_>>();
        for (axis, head) in self.weights.heads.iter().enumerate() {
            let mut weights = vec![0.0f32; NTH_LAYER];
            compute.softmax_f32(&head.layer_weights, &mut weights, 1, NTH_LAYER)?;
            add_scaled(&mut accumulated[axis], &hidden, weights[0]);
            normalized_layer_weights.push(weights);
        }

        let raw_relative_bias = build_relative_bias(
            frames,
            &self.weights.relative_attention_bias,
            NUM_BUCKETS,
            MAX_DISTANCE,
        );
        for (layer, block) in self.weights.blocks.iter().enumerate() {
            wavlm_block_forward(
                &mut hidden,
                frames,
                valid_frames,
                block,
                &raw_relative_bias,
                compute,
            )?;
            for axis in 0..4 {
                add_scaled(
                    &mut accumulated[axis],
                    &hidden,
                    normalized_layer_weights[axis][layer + 1],
                );
            }
        }

        let mut scores = [0.0f32; 4];
        for axis in 0..4 {
            let mut embedding = mean_valid_frames(&accumulated[axis], valid_frames);
            l2_normalize(&mut embedding);
            let raw = projection_head_forward(&embedding, &self.weights.heads[axis], compute)?;
            scores[axis] = raw * TARGET_STDS[axis] + TARGET_MEANS[axis];
        }
        reject_non_finite("Audiobox window scores", &scores)?;
        Ok(scores)
    }
}

fn bind_block(file: &GgufFile, layer: usize) -> Result<WavLmBlock> {
    let prefix = format!("wavlm_model.encoder.layers.{layer}");
    let attention = format!("{prefix}.self_attn");
    Ok(WavLmBlock {
        q_w: tensor(
            file,
            &format!("{attention}.q_proj.weight"),
            &[HIDDEN_SIZE, HIDDEN_SIZE],
        )?,
        q_b: tensor(file, &format!("{attention}.q_proj.bias"), &[HIDDEN_SIZE])?,
        k_w: tensor(
            file,
            &format!("{attention}.k_proj.weight"),
            &[HIDDEN_SIZE, HIDDEN_SIZE],
        )?,
        k_b: tensor(file, &format!("{attention}.k_proj.bias"), &[HIDDEN_SIZE])?,
        v_w: tensor(
            file,
            &format!("{attention}.v_proj.weight"),
            &[HIDDEN_SIZE, HIDDEN_SIZE],
        )?,
        v_b: tensor(file, &format!("{attention}.v_proj.bias"), &[HIDDEN_SIZE])?,
        out_w: tensor(
            file,
            &format!("{attention}.out_proj.weight"),
            &[HIDDEN_SIZE, HIDDEN_SIZE],
        )?,
        out_b: tensor(file, &format!("{attention}.out_proj.bias"), &[HIDDEN_SIZE])?,
        grep_w: tensor(
            file,
            &format!("{attention}.grep_linear.weight"),
            &[8, HEAD_DIM],
        )?,
        grep_b: tensor(file, &format!("{attention}.grep_linear.bias"), &[8])?,
        grep_a: tensor(file, &format!("{attention}.grep_a"), &[1, N_HEAD, 1, 1])?,
        self_attn_norm_gamma: tensor(
            file,
            &format!("{prefix}.self_attn_layer_norm.weight"),
            &[HIDDEN_SIZE],
        )?,
        self_attn_norm_beta: tensor(
            file,
            &format!("{prefix}.self_attn_layer_norm.bias"),
            &[HIDDEN_SIZE],
        )?,
        fc1_w: tensor(
            file,
            &format!("{prefix}.fc1.weight"),
            &[FFN_DIM, HIDDEN_SIZE],
        )?,
        fc1_b: tensor(file, &format!("{prefix}.fc1.bias"), &[FFN_DIM])?,
        fc2_w: tensor(
            file,
            &format!("{prefix}.fc2.weight"),
            &[HIDDEN_SIZE, FFN_DIM],
        )?,
        fc2_b: tensor(file, &format!("{prefix}.fc2.bias"), &[HIDDEN_SIZE])?,
        final_norm_gamma: tensor(
            file,
            &format!("{prefix}.final_layer_norm.weight"),
            &[HIDDEN_SIZE],
        )?,
        final_norm_beta: tensor(
            file,
            &format!("{prefix}.final_layer_norm.bias"),
            &[HIDDEN_SIZE],
        )?,
    })
}

fn bind_axis_head(file: &GgufFile, axis: &str) -> Result<AxisHead> {
    let mut blocks = Vec::with_capacity(PROJ_NUM_LAYER - 1);
    for (linear_index, norm_index) in [(0, 1), (3, 4), (6, 7), (9, 10)] {
        blocks.push(ProjectionBlock {
            weight: tensor(
                file,
                &format!("proj_layer.{axis}.{linear_index}.weight"),
                &[HIDDEN_SIZE, HIDDEN_SIZE],
            )?,
            bias: tensor(
                file,
                &format!("proj_layer.{axis}.{linear_index}.bias"),
                &[HIDDEN_SIZE],
            )?,
            norm_gamma: tensor(
                file,
                &format!("proj_layer.{axis}.{norm_index}.weight"),
                &[HIDDEN_SIZE],
            )?,
            norm_beta: tensor(
                file,
                &format!("proj_layer.{axis}.{norm_index}.bias"),
                &[HIDDEN_SIZE],
            )?,
        });
    }
    Ok(AxisHead {
        layer_weights: tensor(file, &format!("layer_weights.{axis}"), &[NTH_LAYER])?,
        blocks,
        output_weight: tensor(
            file,
            &format!("proj_layer.{axis}.12.weight"),
            &[OUTPUT_DIM, HIDDEN_SIZE],
        )?,
        output_bias: tensor(file, &format!("proj_layer.{axis}.12.bias"), &[OUTPUT_DIM])?,
    })
}

fn wavlm_block_forward(
    hidden: &mut [f32],
    frames: usize,
    valid_frames: usize,
    block: &WavLmBlock,
    raw_relative_bias: &[f32],
    compute: &Compute,
) -> Result<()> {
    let q = linear_forward_with_compute(
        hidden,
        frames,
        HIDDEN_SIZE,
        &block.q_w,
        &block.q_b,
        HIDDEN_SIZE,
        compute,
    )?;
    let k = linear_forward_with_compute(
        hidden,
        frames,
        HIDDEN_SIZE,
        &block.k_w,
        &block.k_b,
        HIDDEN_SIZE,
        compute,
    )?;
    let v = linear_forward_with_compute(
        hidden,
        frames,
        HIDDEN_SIZE,
        &block.v_w,
        &block.v_b,
        HIDDEN_SIZE,
        compute,
    )?;
    let attention = gated_relative_attention(
        hidden,
        &q,
        &k,
        &v,
        frames,
        valid_frames,
        block,
        raw_relative_bias,
        compute,
    )?;
    let projected = linear_forward_with_compute(
        &attention,
        frames,
        HIDDEN_SIZE,
        &block.out_w,
        &block.out_b,
        HIDDEN_SIZE,
        compute,
    )?;
    for (value, residual) in hidden.iter_mut().zip(projected) {
        *value += residual;
    }
    layer_norm_with_compute_inplace(
        hidden,
        frames,
        HIDDEN_SIZE,
        &block.self_attn_norm_gamma,
        &block.self_attn_norm_beta,
        LAYER_NORM_EPS,
        compute,
    )?;

    let residual = hidden.to_vec();
    let intermediate = linear_forward_with_compute(
        hidden,
        frames,
        HIDDEN_SIZE,
        &block.fc1_w,
        &block.fc1_b,
        FFN_DIM,
        compute,
    )?;
    let mut activated = vec![0.0f32; intermediate.len()];
    compute.gelu_f32(&intermediate, &mut activated)?;
    let output = linear_forward_with_compute(
        &activated,
        frames,
        FFN_DIM,
        &block.fc2_w,
        &block.fc2_b,
        HIDDEN_SIZE,
        compute,
    )?;
    for ((value, residual), output) in hidden.iter_mut().zip(residual).zip(output) {
        *value = residual + output;
    }
    layer_norm_with_compute_inplace(
        hidden,
        frames,
        HIDDEN_SIZE,
        &block.final_norm_gamma,
        &block.final_norm_beta,
        LAYER_NORM_EPS,
        compute,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gated_relative_attention(
    raw_query: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    frames: usize,
    valid_frames: usize,
    block: &WavLmBlock,
    raw_relative_bias: &[f32],
    compute: &Compute,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; frames * HIDDEN_SIZE];
    let mut q_head = vec![0.0f32; frames * HEAD_DIM];
    let mut k_head_t = vec![0.0f32; HEAD_DIM * frames];
    let mut v_head = vec![0.0f32; frames * HEAD_DIM];
    let mut scores = vec![0.0f32; frames * frames];
    let mut probabilities = vec![0.0f32; frames * frames];
    let mut head_output = vec![0.0f32; frames * HEAD_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    for head in 0..N_HEAD {
        for frame in 0..frames {
            let source = frame * HIDDEN_SIZE + head * HEAD_DIM;
            let destination = frame * HEAD_DIM;
            q_head[destination..destination + HEAD_DIM]
                .copy_from_slice(&q[source..source + HEAD_DIM]);
            v_head[destination..destination + HEAD_DIM]
                .copy_from_slice(&v[source..source + HEAD_DIM]);
            for dim in 0..HEAD_DIM {
                k_head_t[dim * frames + frame] = k[source + dim];
            }
        }
        compute.gemm_f32(
            frames,
            frames,
            HEAD_DIM,
            &q_head,
            &k_head_t,
            None,
            &mut scores,
        )?;
        for query in 0..frames {
            let gate = gru_relative_gate(raw_query, query, head, block);
            let row = &mut scores[query * frames..(query + 1) * frames];
            for key in 0..frames {
                row[key] = if key < valid_frames {
                    row[key] * scale
                        + gate * raw_relative_bias[(head * frames + query) * frames + key]
                } else {
                    f32::NEG_INFINITY
                };
            }
        }
        compute.softmax_f32(&scores, &mut probabilities, frames, frames)?;
        compute.gemm_f32(
            frames,
            HEAD_DIM,
            frames,
            &probabilities,
            &v_head,
            None,
            &mut head_output,
        )?;
        for frame in 0..frames {
            let source = frame * HEAD_DIM;
            let destination = frame * HIDDEN_SIZE + head * HEAD_DIM;
            output[destination..destination + HEAD_DIM]
                .copy_from_slice(&head_output[source..source + HEAD_DIM]);
        }
    }
    Ok(output)
}

fn gru_relative_gate(raw_query: &[f32], frame: usize, head: usize, block: &WavLmBlock) -> f32 {
    let input = &raw_query
        [frame * HIDDEN_SIZE + head * HEAD_DIM..frame * HIDDEN_SIZE + (head + 1) * HEAD_DIM];
    let mut sums = [0.0f32; 2];
    for output in 0..8 {
        let mut value = block.grep_b[output];
        for (dim, input_value) in input.iter().copied().enumerate() {
            value += input_value * block.grep_w[output * HEAD_DIM + dim];
        }
        sums[output / 4] += value;
    }
    let gate_a = sigmoid(sums[0]);
    let gate_b = sigmoid(sums[1]);
    gate_a * (gate_b * block.grep_a[head] - 1.0) + 2.0
}

fn build_relative_bias(
    frames: usize,
    embedding: &[f32],
    num_buckets: usize,
    max_distance: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; N_HEAD * frames * frames];
    for head in 0..N_HEAD {
        for query in 0..frames {
            for key in 0..frames {
                let relative = key as isize - query as isize;
                let bucket = relative_position_bucket(relative, num_buckets, max_distance);
                output[(head * frames + query) * frames + key] = embedding[bucket * N_HEAD + head];
            }
        }
    }
    output
}

fn relative_position_bucket(
    relative_position: isize,
    num_buckets: usize,
    max_distance: usize,
) -> usize {
    let half = num_buckets / 2;
    let direction = usize::from(relative_position > 0) * half;
    let distance = relative_position.unsigned_abs();
    let max_exact = half / 2;
    let bucket = if distance < max_exact {
        distance
    } else {
        let ratio = distance as f64 / max_exact as f64;
        let span = half - max_exact;
        let large = max_exact
            + (ratio.ln() / (max_distance as f64 / max_exact as f64).ln() * span as f64) as usize;
        large.min(half - 1)
    };
    direction + bucket
}

fn projection_head_forward(embedding: &[f32], head: &AxisHead, compute: &Compute) -> Result<f32> {
    let mut hidden = embedding.to_vec();
    for block in &head.blocks {
        hidden = linear_forward_with_compute(
            &hidden,
            1,
            HIDDEN_SIZE,
            &block.weight,
            &block.bias,
            HIDDEN_SIZE,
            compute,
        )?;
        layer_norm_with_compute_inplace(
            &mut hidden,
            1,
            HIDDEN_SIZE,
            &block.norm_gamma,
            &block.norm_beta,
            LAYER_NORM_EPS,
            compute,
        )?;
        let mut activated = vec![0.0f32; HIDDEN_SIZE];
        compute.gelu_f32(&hidden, &mut activated)?;
        hidden = activated;
    }
    let output = linear_forward_with_compute(
        &hidden,
        1,
        HIDDEN_SIZE,
        &head.output_weight,
        &head.output_bias,
        OUTPUT_DIM,
        compute,
    )?;
    Ok(output[0])
}

fn encoder_config() -> CharsiuConfig {
    CharsiuConfig {
        hidden_size: HIDDEN_SIZE,
        n_layer: N_LAYER,
        n_head: N_HEAD,
        ffn_dim: FFN_DIM,
        vocab_size: 0,
        silence_id: 0,
        pad_id: 0,
        sample_rate: SAMPLE_RATE,
        frame_shift_sec: 0.02,
        layer_norm_eps: LAYER_NORM_EPS,
        pos_conv_kernel: POS_CONV_KERNEL,
        pos_conv_groups: POS_CONV_GROUPS,
        silence_threshold: 0,
        feature_projection_has_layer_norm: true,
        stem_conv_bias: false,
    }
}

fn add_scaled(output: &mut [f32], input: &[f32], scale: f32) {
    for (output, input) in output.iter_mut().zip(input) {
        *output += *input * scale;
    }
}

fn mean_valid_frames(values: &[f32], valid_frames: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; HIDDEN_SIZE];
    for frame in 0..valid_frames {
        for dim in 0..HIDDEN_SIZE {
            output[dim] += values[frame * HIDDEN_SIZE + dim];
        }
    }
    let inverse = 1.0 / valid_frames.max(1) as f32;
    for value in &mut output {
        *value *= inverse;
    }
    output
}

fn l2_normalize(values: &mut [f32]) {
    let norm = values
        .iter()
        .map(|value| *value * *value)
        .sum::<f32>()
        .sqrt()
        .max(L2_EPS);
    for value in values {
        *value /= norm;
    }
}

fn collapsed_valid_frames(valid_samples: usize, frames: usize, source_samples: usize) -> usize {
    let truncated = source_samples - source_samples % frames;
    let samples_per_frame = truncated / frames;
    valid_samples
        .min(truncated)
        .div_ceil(samples_per_frame)
        .clamp(1, frames)
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn validate_pcm(pcm: &[f32], sample_rate: u32) -> Result<()> {
    if sample_rate != SAMPLE_RATE {
        return Err(VokraError::InvalidArgument(format!(
            "audiobox-aesthetics: expected {SAMPLE_RATE} Hz mono PCM, got {sample_rate} Hz; resample explicitly before scoring"
        )));
    }
    if pcm.is_empty() {
        return Err(VokraError::InvalidArgument(
            "audiobox-aesthetics: PCM is empty".to_owned(),
        ));
    }
    if let Some((index, value)) = pcm
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(format!(
            "audiobox-aesthetics: non-finite PCM value {value} at index {index}"
        )));
    }
    Ok(())
}

fn reject_non_finite(label: &str, values: &[f32]) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(format!(
            "{ARCH}: {label} contains non-finite value {value} at index {index}"
        )));
    }
    Ok(())
}

fn tensor(file: &GgufFile, name: &str, dims: &[usize]) -> Result<Vec<f32>> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("{ARCH}: missing tensor `{name}`")))?;
    let expected = dims.iter().map(|&dim| dim as u64).collect::<Vec<_>>();
    if info.dimensions != expected {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: tensor `{name}` has shape {:?}, expected {expected:?}",
            info.dimensions
        )));
    }
    file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!("{ARCH}: reading tensor `{name}` failed: {error}"))
    })
}

fn expected_manifest() -> BTreeMap<String, Vec<u64>> {
    let mut expected = BTreeMap::new();
    for axis in AXES {
        expected.insert(format!("layer_weights.{axis}"), vec![NTH_LAYER as u64]);
        for index in [0, 3, 6, 9] {
            expected.insert(
                format!("proj_layer.{axis}.{index}.weight"),
                vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            );
            expected.insert(
                format!("proj_layer.{axis}.{index}.bias"),
                vec![HIDDEN_SIZE as u64],
            );
        }
        for index in [1, 4, 7, 10] {
            expected.insert(
                format!("proj_layer.{axis}.{index}.weight"),
                vec![HIDDEN_SIZE as u64],
            );
            expected.insert(
                format!("proj_layer.{axis}.{index}.bias"),
                vec![HIDDEN_SIZE as u64],
            );
        }
        expected.insert(
            format!("proj_layer.{axis}.12.weight"),
            vec![OUTPUT_DIM as u64, HIDDEN_SIZE as u64],
        );
        expected.insert(
            format!("proj_layer.{axis}.12.bias"),
            vec![OUTPUT_DIM as u64],
        );
    }
    expected.insert(
        "wavlm_model.encoder.layer_norm.weight".to_owned(),
        vec![HIDDEN_SIZE as u64],
    );
    expected.insert(
        "wavlm_model.encoder.layer_norm.bias".to_owned(),
        vec![HIDDEN_SIZE as u64],
    );
    for layer in 0..N_LAYER {
        let prefix = format!("wavlm_model.encoder.layers.{layer}");
        for (suffix, shape) in [
            ("fc1.weight", vec![FFN_DIM as u64, HIDDEN_SIZE as u64]),
            ("fc1.bias", vec![FFN_DIM as u64]),
            ("fc2.weight", vec![HIDDEN_SIZE as u64, FFN_DIM as u64]),
            ("fc2.bias", vec![HIDDEN_SIZE as u64]),
            ("final_layer_norm.weight", vec![HIDDEN_SIZE as u64]),
            ("final_layer_norm.bias", vec![HIDDEN_SIZE as u64]),
            ("self_attn.grep_a", vec![1, N_HEAD as u64, 1, 1]),
            ("self_attn.grep_linear.weight", vec![8, HEAD_DIM as u64]),
            ("self_attn.grep_linear.bias", vec![8]),
            (
                "self_attn.k_proj.weight",
                vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            ),
            ("self_attn.k_proj.bias", vec![HIDDEN_SIZE as u64]),
            (
                "self_attn.out_proj.weight",
                vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            ),
            ("self_attn.out_proj.bias", vec![HIDDEN_SIZE as u64]),
            (
                "self_attn.q_proj.weight",
                vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            ),
            ("self_attn.q_proj.bias", vec![HIDDEN_SIZE as u64]),
            (
                "self_attn.v_proj.weight",
                vec![HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
            ),
            ("self_attn.v_proj.bias", vec![HIDDEN_SIZE as u64]),
            ("self_attn_layer_norm.weight", vec![HIDDEN_SIZE as u64]),
            ("self_attn_layer_norm.bias", vec![HIDDEN_SIZE as u64]),
        ] {
            expected.insert(format!("{prefix}.{suffix}"), shape);
        }
    }
    expected.insert(
        "wavlm_model.encoder.layers.0.self_attn.relative_attention_bias.weight".to_owned(),
        vec![NUM_BUCKETS as u64, N_HEAD as u64],
    );
    expected.insert(
        "wavlm_model.encoder.pos_conv.0.weight_g".to_owned(),
        vec![1, 1, POS_CONV_KERNEL as u64],
    );
    expected.insert(
        "wavlm_model.encoder.pos_conv.0.weight_v".to_owned(),
        vec![
            HIDDEN_SIZE as u64,
            (HIDDEN_SIZE / POS_CONV_GROUPS) as u64,
            POS_CONV_KERNEL as u64,
        ],
    );
    expected.insert(
        "wavlm_model.encoder.pos_conv.0.bias".to_owned(),
        vec![HIDDEN_SIZE as u64],
    );
    for (layer, kernel) in CONV_KERNEL.into_iter().enumerate() {
        let input = if layer == 0 { 1 } else { FEATURE_DIM as u64 };
        expected.insert(
            format!("wavlm_model.feature_extractor.conv_layers.{layer}.0.weight"),
            vec![FEATURE_DIM as u64, input, kernel as u64],
        );
    }
    for suffix in ["weight", "bias"] {
        expected.insert(
            format!("wavlm_model.feature_extractor.conv_layers.0.2.{suffix}"),
            vec![FEATURE_DIM as u64],
        );
        expected.insert(
            format!("wavlm_model.layer_norm.{suffix}"),
            vec![FEATURE_DIM as u64],
        );
    }
    expected.insert("wavlm_model.mask_emb".to_owned(), vec![HIDDEN_SIZE as u64]);
    expected.insert(
        "wavlm_model.post_extract_proj.weight".to_owned(),
        vec![HIDDEN_SIZE as u64, FEATURE_DIM as u64],
    );
    expected.insert(
        "wavlm_model.post_extract_proj.bias".to_owned(),
        vec![HIDDEN_SIZE as u64],
    );
    expected
}

fn validate_manifest(file: &GgufFile) -> Result<()> {
    let expected = expected_manifest();
    if expected.len() != TENSOR_COUNT || file.tensors().len() != TENSOR_COUNT {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: checkpoint has {} tensors, expected exactly {TENSOR_COUNT}",
            file.tensors().len()
        )));
    }
    let mut seen = BTreeSet::new();
    for info in file.tensors() {
        let expected_shape = expected.get(&info.name).ok_or_else(|| {
            VokraError::ModelLoad(format!("{ARCH}: unexpected tensor `{}`", info.name))
        })?;
        if &info.dimensions != expected_shape {
            return Err(VokraError::ModelLoad(format!(
                "{ARCH}: tensor `{}` has shape {:?}, expected {:?}",
                info.name, info.dimensions, expected_shape
            )));
        }
        if info.dtype != GgmlType::F32 {
            return Err(VokraError::ModelLoad(format!(
                "{ARCH}: tensor `{}` is {:?}, expected F32 for checkpoint {CHECKPOINT_REVISION}",
                info.name, info.dtype
            )));
        }
        seen.insert(info.name.as_str());
    }
    if let Some(missing) = expected.keys().find(|name| !seen.contains(name.as_str())) {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: missing tensor `{missing}`"
        )));
    }
    Ok(())
}

fn validate_contract_metadata(file: &GgufFile) -> Result<()> {
    require_string(file, KEY_CHECKPOINT_REVISION, CHECKPOINT_REVISION)?;
    require_string(file, KEY_SOURCE_REVISION, SOURCE_REVISION)?;
    for (key, value) in [
        (KEY_SAMPLE_RATE, SAMPLE_RATE),
        (KEY_WINDOW_SAMPLES, WINDOW_SAMPLES as u32),
        (KEY_HOP_SAMPLES, HOP_SAMPLES as u32),
        (KEY_FEATURE_DIM, FEATURE_DIM as u32),
        (KEY_HIDDEN_SIZE, HIDDEN_SIZE as u32),
        (KEY_FFN_DIM, FFN_DIM as u32),
        (KEY_N_LAYER, N_LAYER as u32),
        (KEY_N_HEAD, N_HEAD as u32),
        (KEY_POS_CONV_KERNEL, POS_CONV_KERNEL as u32),
        (KEY_POS_CONV_GROUPS, POS_CONV_GROUPS as u32),
        (KEY_NUM_BUCKETS, NUM_BUCKETS as u32),
        (KEY_MAX_DISTANCE, MAX_DISTANCE as u32),
        (KEY_NTH_LAYER, NTH_LAYER as u32),
        (KEY_PROJ_NUM_LAYER, PROJ_NUM_LAYER as u32),
        (KEY_OUTPUT_DIM, OUTPUT_DIM as u32),
    ] {
        require_u32(file, key, value)?;
    }
    require_bool(file, KEY_NORMALIZE_EMBED, true)?;
    require_bool(file, KEY_WEIGHTED_LAYER_SUM, true)?;
    require_f32(file, KEY_LAYER_NORM_EPS, LAYER_NORM_EPS)?;
    require_string_array(file, KEY_AXES, &AXES)?;
    require_f32_array(file, KEY_TARGET_MEANS, &TARGET_MEANS)?;
    require_f32_array(file, KEY_TARGET_STDS, &TARGET_STDS)?;
    Ok(())
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_str);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: metadata `{key}` is {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn require_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_u64);
    if actual != Some(u64::from(expected)) {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: metadata `{key}` is {actual:?}, expected {expected}"
        )));
    }
    Ok(())
}

fn require_bool(file: &GgufFile, key: &str, expected: bool) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_bool);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: metadata `{key}` is {actual:?}, expected {expected}"
        )));
    }
    Ok(())
}

fn require_f32(file: &GgufFile, key: &str, expected: f32) -> Result<()> {
    let actual = match file.get(key) {
        Some(GgufMetadataValue::F32(value)) => Some(*value),
        _ => None,
    };
    if actual.map(f32::to_bits) != Some(expected.to_bits()) {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: metadata `{key}` is {actual:?}, expected {expected}"
        )));
    }
    Ok(())
}

fn require_string_array(file: &GgufFile, key: &str, expected: &[&str]) -> Result<()> {
    let Some(array) = file.get(key).and_then(GgufMetadataValue::as_array) else {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: missing string array `{key}`"
        )));
    };
    let actual = array
        .values
        .iter()
        .map(GgufMetadataValue::as_str)
        .collect::<Option<Vec<_>>>();
    if array.element_type != GgufValueType::String || actual.as_deref() != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: metadata `{key}` does not match ordered axes {expected:?}"
        )));
    }
    Ok(())
}

fn require_f32_array(file: &GgufFile, key: &str, expected: &[f32]) -> Result<()> {
    let Some(array) = file.get(key).and_then(GgufMetadataValue::as_array) else {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: missing F32 array `{key}`"
        )));
    };
    let actual = array
        .values
        .iter()
        .map(|value| match value {
            GgufMetadataValue::F32(value) => Some(*value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>();
    let matches = actual.as_deref().is_some_and(|values| {
        values.len() == expected.len()
            && values
                .iter()
                .zip(expected)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
    });
    if array.element_type != GgufValueType::F32 || !matches {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: metadata `{key}` does not match the pinned target transform"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_and_axes_match_the_pinned_checkpoint() {
        let manifest = expected_manifest();
        assert_eq!(manifest.len(), TENSOR_COUNT);
        assert_eq!(AXES, ["CE", "CU", "PC", "PQ"]);
        assert!(!manifest.keys().any(|name| name.contains("BALANCED")));
    }

    #[test]
    fn relative_position_buckets_match_wavlm_boundaries() {
        assert_eq!(relative_position_bucket(0, 320, 800), 0);
        assert_eq!(relative_position_bucket(79, 320, 800), 239);
        assert_eq!(relative_position_bucket(-79, 320, 800), 79);
        assert_eq!(relative_position_bucket(80, 320, 800), 240);
        assert_eq!(relative_position_bucket(-80, 320, 800), 80);
        assert_eq!(relative_position_bucket(800, 320, 800), 319);
        assert_eq!(relative_position_bucket(-800, 320, 800), 159);
    }

    #[test]
    fn padding_mask_collapse_matches_upstream_grouping() {
        assert_eq!(collapsed_valid_frames(1, 499, 160_000), 1);
        assert_eq!(collapsed_valid_frames(320, 499, 160_000), 1);
        assert_eq!(collapsed_valid_frames(321, 499, 160_000), 2);
        assert_eq!(collapsed_valid_frames(159_680, 499, 160_000), 499);
        assert_eq!(collapsed_valid_frames(160_000, 499, 160_000), 499);
    }

    #[test]
    fn wrong_sample_rate_is_an_explicit_error() {
        let error = validate_pcm(&[0.0], 48_000).unwrap_err().to_string();
        assert!(error.contains("expected 16000 Hz"));
        assert!(error.contains("resample explicitly"));
    }
}
