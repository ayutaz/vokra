//! Strict native Wav2Vec2 CTC runtime for the public `vokra/*wav2vec2*` GGUFs.
//!
//! The loader is intentionally variant-closed.  It selects a verified topology
//! from `vokra.provenance.model_id`, checks the complete public-artifact tensor
//! count and every inference tensor shape, and only then decodes weights.  Three
//! early public artifacts were stamped with the base-960h metadata despite
//! carrying large-model tensors.  Those exact model ids may use the documented
//! base-stamp repair; every other metadata mismatch is a hard load error.
//!
//! CPU and Metal use the same imperative [`Compute`] graph.  Convolution,
//! grouped positional convolution, GEMM, softmax, LayerNorm and GELU are all
//! dispatched through the selected backend.  Host work is limited to tensor
//! layout glue, waveform statistics, CTC folding and tokenizer string assembly.

use std::path::Path;

use vokra_core::backend::BackendKind;
use vokra_core::engines::AsrEngine;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::tasks::Transcription;
use vokra_core::{Result, VokraError};
use vokra_ops::{
    ConvLayerAttrs, ConvLayerWeights, Norm, WaveformFrontendAttrs, WaveformFrontendWeights,
    ctc_decode_greedy,
};

use crate::align::charsiu::{
    CharsiuBlock, CharsiuConfig, CharsiuFeatureProjection, CharsiuHead, CharsiuPosConv,
    feature_projection_forward_with_compute, layer_norm_with_compute_inplace,
    linear_forward_with_compute, positional_conv_forward_with_compute,
    transformer_block_forward_with_valid_keys_and_compute,
};
use crate::compute::{Compute, HotOp};

/// `vokra.model.arch` emitted by the Wav2Vec2 family converter.
pub const ARCH: &str = "wav2vec2_ctc";

/// Complete learned-op registry for the CPU/Metal route.
pub const WAV2VEC2_CTC_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Gelu,
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
];

const SAMPLE_RATE: u32 = 16_000;
const NORMALIZATION_EPS: f32 = 1e-7;
const LAYER_NORM_EPS: f32 = 1e-5;
const FEATURE_DIM: usize = 512;
const POS_KERNEL: usize = 128;
const POS_GROUPS: usize = 16;
const BLANK_ID: usize = 0;
const CONV_DIM: [usize; 7] = [512; 7];
const CONV_KERNEL: [usize; 7] = [10, 3, 3, 3, 3, 2, 2];
const CONV_STRIDE: [usize; 7] = [5, 2, 2, 2, 2, 2, 2];

const KEY_MODEL_ID: &str = "vokra.provenance.model_id";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_HIDDEN: &str = "vokra.wav2vec2_ctc.hidden_size";
const KEY_LAYERS: &str = "vokra.wav2vec2_ctc.n_layer";
const KEY_HEADS: &str = "vokra.wav2vec2_ctc.n_head";
const KEY_FFN: &str = "vokra.wav2vec2_ctc.intermediate_size";
const KEY_VOCAB: &str = "vokra.wav2vec2_ctc.vocab_size";
const KEY_EPS: &str = "vokra.wav2vec2_ctc.layer_norm_eps";
const KEY_FEAT_NORM: &str = "vokra.wav2vec2_ctc.feat_extract_norm";
const KEY_STABLE: &str = "vokra.wav2vec2_ctc.do_stable_layer_norm";
const KEY_ACT: &str = "vokra.wav2vec2_ctc.hidden_act";
const KEY_POS_KERNEL: &str = "vokra.wav2vec2_ctc.num_conv_pos_embeddings";
const KEY_POS_GROUPS: &str = "vokra.wav2vec2_ctc.num_conv_pos_embedding_groups";
const KEY_HAS_HEAD: &str = "vokra.wav2vec2_ctc.has_ctc_head";
const KEY_STEM_LAYERS: &str = "vokra.wav2vec2_ctc.num_feat_extract_layers";
const KEY_CONV_DIM: &str = "vokra.wav2vec2_ctc.conv_dim";
const KEY_CONV_KERNEL: &str = "vokra.wav2vec2_ctc.conv_kernel";
const KEY_CONV_STRIDE: &str = "vokra.wav2vec2_ctc.conv_stride";

const HUBERT_KEY_HIDDEN: &str = "vokra.hubert.hidden_size";
const HUBERT_KEY_LAYERS: &str = "vokra.hubert.n_layer";
const HUBERT_KEY_HEADS: &str = "vokra.hubert.n_head";
const HUBERT_KEY_FFN: &str = "vokra.hubert.intermediate_size";
const HUBERT_KEY_VOCAB: &str = "vokra.hubert.vocab_size";
const HUBERT_KEY_EPS: &str = "vokra.hubert.layer_norm_eps";
const HUBERT_KEY_FEAT_NORM: &str = "vokra.hubert.feat_extract_norm";
const HUBERT_KEY_STABLE: &str = "vokra.hubert.do_stable_layer_norm";
const HUBERT_KEY_ACT: &str = "vokra.hubert.hidden_act";
const HUBERT_KEY_POS_KERNEL: &str = "vokra.hubert.num_conv_pos_embeddings";
const HUBERT_KEY_POS_GROUPS: &str = "vokra.hubert.num_conv_pos_embedding_groups";
const HUBERT_KEY_HAS_HEAD: &str = "vokra.hubert.has_ctc_head";
const HUBERT_KEY_STEM_LAYERS: &str = "vokra.hubert.num_feat_extract_layers";
const HUBERT_KEY_CONV_DIM: &str = "vokra.hubert.conv_dim";
const HUBERT_KEY_CONV_KERNEL: &str = "vokra.hubert.conv_kernel";
const HUBERT_KEY_CONV_STRIDE: &str = "vokra.hubert.conv_stride";

const ENGLISH_VOCAB: &[u8] = include_bytes!("../../resources/wav2vec2_ctc/english.json");
const ARABIC_VOCAB: &[u8] = include_bytes!("../../resources/wav2vec2_ctc/arabic.json");
const ESPEAK_VOCAB: &[u8] = include_bytes!("../../resources/wav2vec2_ctc/espeak.json");
const JAPANESE_VOCAB: &[u8] = include_bytes!("../../resources/wav2vec2_ctc/japanese.json");
const CHINESE_VOCAB: &[u8] = include_bytes!("../../resources/wav2vec2_ctc/chinese_zh_cn.json");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LegacyMetadata {
    None,
    Base960hStamp,
}

#[derive(Debug, Clone, Copy)]
struct VariantSpec {
    model_id: &'static str,
    upstream_hf: &'static str,
    hidden: usize,
    layers: usize,
    heads: usize,
    ffn: usize,
    vocab: usize,
    stable: bool,
    stem_norm: Norm,
    stem_bias: bool,
    has_head: bool,
    tensor_count: usize,
    vocab_json: Option<&'static [u8]>,
    legacy: LegacyMetadata,
}

const VARIANTS: &[VariantSpec] = &[
    VariantSpec {
        model_id: "wav2vec2-base-960h",
        upstream_hf: "facebook/wav2vec2-base-960h",
        hidden: 768,
        layers: 12,
        heads: 12,
        ffn: 3072,
        vocab: 32,
        stable: false,
        stem_norm: Norm::GroupFirstOnly,
        stem_bias: false,
        has_head: true,
        tensor_count: 212,
        vocab_json: Some(ENGLISH_VOCAB),
        legacy: LegacyMetadata::None,
    },
    VariantSpec {
        model_id: "wav2vec2-large-960h-lv60-self",
        upstream_hf: "facebook/wav2vec2-large-960h-lv60-self",
        hidden: 1024,
        layers: 24,
        heads: 16,
        ffn: 4096,
        vocab: 32,
        stable: true,
        stem_norm: Norm::LayerAll,
        stem_bias: true,
        has_head: true,
        tensor_count: 423,
        vocab_json: Some(ENGLISH_VOCAB),
        legacy: LegacyMetadata::Base960hStamp,
    },
    VariantSpec {
        model_id: "wav2vec2-large-xlsr-53",
        upstream_hf: "facebook/wav2vec2-large-xlsr-53",
        hidden: 1024,
        layers: 24,
        heads: 16,
        ffn: 4096,
        vocab: 32,
        stable: true,
        stem_norm: Norm::LayerAll,
        stem_bias: true,
        has_head: false,
        tensor_count: 429,
        vocab_json: None,
        legacy: LegacyMetadata::None,
    },
    VariantSpec {
        model_id: "wav2vec2-large-xlsr-53-arabic",
        upstream_hf: "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
        hidden: 1024,
        layers: 24,
        heads: 16,
        ffn: 4096,
        vocab: 51,
        stable: true,
        stem_norm: Norm::LayerAll,
        stem_bias: true,
        has_head: true,
        tensor_count: 424,
        vocab_json: Some(ARABIC_VOCAB),
        legacy: LegacyMetadata::Base960hStamp,
    },
    VariantSpec {
        model_id: "wav2vec2-large-xlsr-53-chinese-zh-cn",
        upstream_hf: "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        hidden: 1024,
        layers: 24,
        heads: 16,
        ffn: 4096,
        vocab: 3503,
        stable: true,
        stem_norm: Norm::LayerAll,
        stem_bias: true,
        has_head: true,
        tensor_count: 424,
        vocab_json: Some(CHINESE_VOCAB),
        legacy: LegacyMetadata::None,
    },
    VariantSpec {
        model_id: "wav2vec2-large-xlsr-53-japanese",
        upstream_hf: "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
        hidden: 1024,
        layers: 24,
        heads: 16,
        ffn: 4096,
        vocab: 2341,
        stable: true,
        stem_norm: Norm::LayerAll,
        stem_bias: true,
        has_head: true,
        tensor_count: 424,
        vocab_json: Some(JAPANESE_VOCAB),
        legacy: LegacyMetadata::None,
    },
    VariantSpec {
        model_id: "wav2vec2-xlsr-53-espeak-cv-ft",
        upstream_hf: "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
        hidden: 1024,
        layers: 24,
        heads: 16,
        ffn: 4096,
        vocab: 392,
        stable: true,
        stem_norm: Norm::LayerAll,
        stem_bias: true,
        has_head: true,
        tensor_count: 424,
        vocab_json: Some(ESPEAK_VOCAB),
        legacy: LegacyMetadata::Base960hStamp,
    },
];

const HUBERT_LARGE_LS960_SPEC: VariantSpec = VariantSpec {
    model_id: "hubert-large-ls960",
    upstream_hf: "facebook/hubert-large-ls960-ft",
    hidden: 1024,
    layers: 24,
    heads: 16,
    ffn: 4096,
    vocab: 32,
    stable: true,
    stem_norm: Norm::LayerAll,
    stem_bias: true,
    has_head: true,
    tensor_count: 424,
    vocab_json: Some(ENGLISH_VOCAB),
    legacy: LegacyMetadata::None,
};

/// Resolved, shape-checked Wav2Vec2 inference configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Wav2Vec2CtcConfig {
    /// Stable public artifact id (`vokra.provenance.model_id`).
    pub model_id: String,
    /// Transformer width.
    pub hidden_size: usize,
    /// Transformer block count.
    pub n_layer: usize,
    /// Attention head count.
    pub n_head: usize,
    /// Feed-forward inner width.
    pub intermediate_size: usize,
    /// CTC head width, or the upstream sentinel for an encoder-only model.
    pub vocab_size: usize,
    /// Whether the encoder uses pre-norm stable blocks.
    pub do_stable_layer_norm: bool,
    /// Whether a real CTC projection is present.
    pub has_ctc_head: bool,
    /// Native input sample rate.
    pub sample_rate: u32,
}

#[derive(Debug, Clone)]
struct Wav2Vec2Weights {
    stem_attrs: WaveformFrontendAttrs,
    stem: WaveformFrontendWeights,
    projection: CharsiuFeatureProjection,
    position: CharsiuPosConv,
    encoder_norm_gamma: Vec<f32>,
    encoder_norm_beta: Vec<f32>,
    blocks: Vec<CharsiuBlock>,
    head: Option<CharsiuHead>,
}

/// Native Wav2Vec2 encoder plus optional CTC head.
#[derive(Debug, Clone)]
pub struct Wav2Vec2Ctc {
    config: Wav2Vec2CtcConfig,
    weights: Wav2Vec2Weights,
    vocab: Option<Vec<String>>,
    backend: BackendKind,
    legacy_metadata_repaired: bool,
}

impl Wav2Vec2Ctc {
    /// Opens and strictly binds one GGUF.
    pub fn from_gguf(path: impl AsRef<Path>) -> Result<Self> {
        let file = GgufFile::open(path)?;
        Self::from_file(&file)
    }

    /// Strictly binds an already-open GGUF.
    pub fn from_file(file: &GgufFile) -> Result<Self> {
        require_string(file, chunks::KEY_MODEL_ARCH, ARCH)?;
        let model_id = meta_string(file, KEY_MODEL_ID)?;
        if model_id == "mms-1b-all" {
            return Err(VokraError::UnsupportedOp(
                "wav2vec2_ctc: public `vokra/mms-1b-all-base` is an 8.9 MB adapter-only artifact, not the 1B backbone. Its stamped 1024/24 axes and permissive license are also incompatible with the verified facebook/mms-1b-all 1280/48 CC-BY-NC-4.0 contract. Bind is refused until a full, correctly licensed backbone+adapter GGUF is converted and published through the gated workflow."
                    .to_owned(),
            ));
        }
        let spec = VARIANTS
            .iter()
            .find(|variant| variant.model_id == model_id)
            .copied()
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "wav2vec2_ctc: unsupported model id {model_id:?}; supported public ids are {}",
                    VARIANTS
                        .iter()
                        .map(|variant| variant.model_id)
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })?;

        let legacy_metadata_repaired = match spec.legacy {
            LegacyMetadata::None => {
                validate_metadata(file, &spec)?;
                false
            }
            LegacyMetadata::Base960hStamp if metadata_matches_spec(file, &spec) => false,
            LegacyMetadata::Base960hStamp if metadata_matches_base_stamp(file) => true,
            LegacyMetadata::Base960hStamp => {
                return Err(VokraError::ModelLoad(format!(
                    "wav2vec2_ctc: {model_id} metadata matches neither its verified topology nor the exact legacy base-960h stamp; refusing an unrecognised repair"
                )));
            }
        };
        if file.tensors().len() != spec.tensor_count {
            return Err(VokraError::ModelLoad(format!(
                "wav2vec2_ctc: {model_id} has {} tensors, expected exactly {} for the audited public GGUF",
                file.tensors().len(),
                spec.tensor_count
            )));
        }

        Self::bind_audited(file, &spec, "wav2vec2", legacy_metadata_repaired)
    }

    /// Binds the audited public HuBERT-Large checkpoint through the
    /// shared waveform/Transformer implementation while preserving its
    /// distinct arch contract.
    pub(crate) fn from_hubert_file(file: &GgufFile) -> Result<Self> {
        require_string(file, chunks::KEY_MODEL_ARCH, "hubert")?;
        require_string(file, KEY_MODEL_ID, HUBERT_LARGE_LS960_SPEC.model_id)?;
        require_string(
            file,
            chunks::KEY_MODEL_NAME,
            HUBERT_LARGE_LS960_SPEC.model_id,
        )?;
        require_string(file, KEY_UPSTREAM_HF, HUBERT_LARGE_LS960_SPEC.upstream_hf)?;
        let has_hparams = [
            HUBERT_KEY_HIDDEN,
            HUBERT_KEY_LAYERS,
            HUBERT_KEY_HEADS,
            HUBERT_KEY_FFN,
            HUBERT_KEY_VOCAB,
            HUBERT_KEY_EPS,
            HUBERT_KEY_FEAT_NORM,
            HUBERT_KEY_STABLE,
            HUBERT_KEY_ACT,
            HUBERT_KEY_POS_KERNEL,
            HUBERT_KEY_POS_GROUPS,
            HUBERT_KEY_HAS_HEAD,
            HUBERT_KEY_STEM_LAYERS,
            HUBERT_KEY_CONV_DIM,
            HUBERT_KEY_CONV_KERNEL,
            HUBERT_KEY_CONV_STRIDE,
        ]
        .iter()
        .any(|key| file.get(key).is_some());
        if has_hparams && !hubert_metadata_matches_spec(file) {
            return Err(VokraError::ModelLoad(
                "hubert: topology metadata is partial or does not match the verified 1024/24/16/4096 HuBERT-Large-LS960 contract"
                    .to_owned(),
            ));
        }
        if file.tensors().len() != HUBERT_LARGE_LS960_SPEC.tensor_count {
            return Err(VokraError::ModelLoad(format!(
                "hubert: {} has {} tensors, expected exactly {} for the audited public GGUF",
                HUBERT_LARGE_LS960_SPEC.model_id,
                file.tensors().len(),
                HUBERT_LARGE_LS960_SPEC.tensor_count
            )));
        }
        Self::bind_audited(file, &HUBERT_LARGE_LS960_SPEC, "hubert", false)
    }

    fn bind_audited(
        file: &GgufFile,
        spec: &VariantSpec,
        family_prefix: &str,
        legacy_metadata_repaired: bool,
    ) -> Result<Self> {
        let stem_attrs = stem_attrs(spec);
        let mut stem_layers = Vec::with_capacity(CONV_DIM.len());
        let mut in_channels = 1usize;
        for layer in 0..CONV_DIM.len() {
            let prefix = format!("{family_prefix}.feature_extractor.conv_layers.{layer}");
            let norm =
                if spec.stem_norm.has_group_norm(layer) || spec.stem_norm.has_layer_norm(layer) {
                    (
                        Some(tensor(
                            file,
                            &format!("{prefix}.layer_norm.weight"),
                            &[512],
                        )?),
                        Some(tensor(file, &format!("{prefix}.layer_norm.bias"), &[512])?),
                    )
                } else {
                    (None, None)
                };
            stem_layers.push(ConvLayerWeights {
                conv_w: tensor(
                    file,
                    &format!("{prefix}.conv.weight"),
                    &[CONV_DIM[layer], in_channels, CONV_KERNEL[layer]],
                )?,
                conv_b: if spec.stem_bias {
                    tensor(file, &format!("{prefix}.conv.bias"), &[CONV_DIM[layer]])?
                } else {
                    Vec::new()
                },
                norm_gamma: norm.0,
                norm_beta: norm.1,
            });
            in_channels = CONV_DIM[layer];
        }
        let stem = WaveformFrontendWeights {
            layers: stem_layers,
        };
        stem.validate(&stem_attrs)?;

        let projection = CharsiuFeatureProjection {
            norm_gamma: Some(tensor(
                file,
                &format!("{family_prefix}.feature_projection.layer_norm.weight"),
                &[FEATURE_DIM],
            )?),
            norm_beta: Some(tensor(
                file,
                &format!("{family_prefix}.feature_projection.layer_norm.bias"),
                &[FEATURE_DIM],
            )?),
            linear_w: tensor(
                file,
                &format!("{family_prefix}.feature_projection.projection.weight"),
                &[spec.hidden, FEATURE_DIM],
            )?,
            linear_b: tensor(
                file,
                &format!("{family_prefix}.feature_projection.projection.bias"),
                &[spec.hidden],
            )?,
        };
        let position_v = tensor(
            file,
            &format!("{family_prefix}.encoder.pos_conv_embed.conv.weight_v"),
            &[spec.hidden, spec.hidden / POS_GROUPS, POS_KERNEL],
        )?;
        let position_g = tensor(
            file,
            &format!("{family_prefix}.encoder.pos_conv_embed.conv.weight_g"),
            &[1, 1, POS_KERNEL],
        )?;
        let position = CharsiuPosConv {
            weight: fold_weight_norm_dim2(
                &position_g,
                &position_v,
                spec.hidden,
                spec.hidden / POS_GROUPS,
                POS_KERNEL,
            )?,
            bias: tensor(
                file,
                &format!("{family_prefix}.encoder.pos_conv_embed.conv.bias"),
                &[spec.hidden],
            )?,
        };
        let encoder_norm_gamma = tensor(
            file,
            &format!("{family_prefix}.encoder.layer_norm.weight"),
            &[spec.hidden],
        )?;
        let encoder_norm_beta = tensor(
            file,
            &format!("{family_prefix}.encoder.layer_norm.bias"),
            &[spec.hidden],
        )?;
        let mut blocks = Vec::with_capacity(spec.layers);
        for layer in 0..spec.layers {
            let prefix = format!("{family_prefix}.encoder.layers.{layer}");
            blocks.push(bind_family_block(file, &prefix, spec.hidden, spec.ffn)?);
        }
        let head = if spec.has_head {
            Some(CharsiuHead {
                weight: tensor(file, "lm_head.weight", &[spec.vocab, spec.hidden])?,
                bias: tensor(file, "lm_head.bias", &[spec.vocab])?,
            })
        } else {
            if file.tensor_info("lm_head.weight").is_some()
                || file.tensor_info("lm_head.bias").is_some()
            {
                return Err(VokraError::ModelLoad(format!(
                    "wav2vec2_ctc: {} is encoder-only but carries an unexpected lm_head",
                    spec.model_id
                )));
            }
            None
        };
        let vocab = spec
            .vocab_json
            .map(|json| parse_vocab(json, spec.vocab))
            .transpose()?;

        Ok(Self {
            config: Wav2Vec2CtcConfig {
                model_id: spec.model_id.to_owned(),
                hidden_size: spec.hidden,
                n_layer: spec.layers,
                n_head: spec.heads,
                intermediate_size: spec.ffn,
                vocab_size: spec.vocab,
                do_stable_layer_norm: spec.stable,
                has_ctc_head: spec.has_head,
                sample_rate: SAMPLE_RATE,
            },
            weights: Wav2Vec2Weights {
                stem_attrs,
                stem,
                projection,
                position,
                encoder_norm_gamma,
                encoder_norm_beta,
                blocks,
                head,
            },
            vocab,
            backend: BackendKind::Cpu,
            legacy_metadata_repaired,
        })
    }

    /// Selects the backend used by all learned operations.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Resolved configuration.
    pub fn config(&self) -> &Wav2Vec2CtcConfig {
        &self.config
    }

    /// Whether the exact audited legacy base stamp was repaired at bind time.
    pub fn legacy_metadata_repaired(&self) -> bool {
        self.legacy_metadata_repaired
    }

    /// Backend selected for inference.
    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Runs the raw-waveform frontend and Transformer encoder.
    ///
    /// Returns time-major `[frames, hidden_size]` features and `frames`.
    pub fn encode_features(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        validate_pcm(pcm)?;
        let compute = Compute::for_backend(self.backend, WAV2VEC2_CTC_HOT_OPS)?;
        self.encode_with_compute(pcm, &compute)
    }

    /// Runs encoder plus CTC projection and returns `[frames, vocab]` logits.
    pub fn logits(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        validate_pcm(pcm)?;
        let head = self.weights.head.as_ref().ok_or_else(|| {
            VokraError::UnsupportedOp(format!(
                "wav2vec2_ctc: {} is an encoder-only checkpoint and has no CTC head; call encode_features instead",
                self.config.model_id
            ))
        })?;
        let compute = Compute::for_backend(self.backend, WAV2VEC2_CTC_HOT_OPS)?;
        let (features, frames) = self.encode_with_compute(pcm, &compute)?;
        let logits = linear_forward_with_compute(
            &features,
            frames,
            self.config.hidden_size,
            &head.weight,
            &head.bias,
            self.config.vocab_size,
            &compute,
        )?;
        reject_non_finite("logits", &logits)?;
        Ok((logits, frames))
    }

    /// Runs greedy CTC blank/repeat folding.
    pub fn transcribe_tokens(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        let (logits, frames) = self.logits(pcm)?;
        ctc_decode_greedy(&logits, frames, self.config.vocab_size, BLANK_ID)
    }

    /// Runs the complete native PCM → CTC text path.
    pub fn transcribe_text(&self, pcm: &[f32]) -> Result<String> {
        let tokens = self.transcribe_tokens(pcm)?;
        let vocab = self.vocab.as_ref().ok_or_else(|| {
            VokraError::UnsupportedOp(format!(
                "wav2vec2_ctc: {} has no CTC vocabulary because it is encoder-only",
                self.config.model_id
            ))
        })?;
        decode_tokens(&tokens, vocab)
    }

    fn encode_with_compute(&self, pcm: &[f32], compute: &Compute) -> Result<(Vec<f32>, usize)> {
        let normalized = zero_mean_unit_var(pcm);
        let features = waveform_frontend_with_compute(
            &normalized,
            &self.weights.stem_attrs,
            &self.weights.stem,
            compute,
        )?;
        let frames = features.len() / FEATURE_DIM;
        let cfg = shared_encoder_config(&self.config, self.weights.stem_attrs.conv_bias);
        let mut hidden = feature_projection_forward_with_compute(
            &features,
            frames,
            FEATURE_DIM,
            &self.weights.projection,
            self.config.hidden_size,
            true,
            LAYER_NORM_EPS,
            compute,
        )?;
        let position = positional_conv_forward_with_compute(
            &hidden,
            frames,
            &cfg,
            &self.weights.position,
            compute,
        )?;
        for (value, positional) in hidden.iter_mut().zip(position) {
            *value += positional;
        }

        if self.config.do_stable_layer_norm {
            for block in &self.weights.blocks {
                stable_transformer_block(&mut hidden, frames, &cfg, block, compute)?;
            }
            layer_norm_with_compute_inplace(
                &mut hidden,
                frames,
                self.config.hidden_size,
                &self.weights.encoder_norm_gamma,
                &self.weights.encoder_norm_beta,
                LAYER_NORM_EPS,
                compute,
            )?;
        } else {
            layer_norm_with_compute_inplace(
                &mut hidden,
                frames,
                self.config.hidden_size,
                &self.weights.encoder_norm_gamma,
                &self.weights.encoder_norm_beta,
                LAYER_NORM_EPS,
                compute,
            )?;
            for block in &self.weights.blocks {
                transformer_block_forward_with_valid_keys_and_compute(
                    &mut hidden,
                    frames,
                    frames,
                    &cfg,
                    block,
                    compute,
                )?;
            }
        }
        reject_non_finite("encoder output", &hidden)?;
        Ok((hidden, frames))
    }
}

impl AsrEngine for Wav2Vec2Ctc {
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        Ok(Transcription::new(self.transcribe_text(pcm)?))
    }

    fn backend(&self) -> BackendKind {
        self.backend
    }
}

fn shared_encoder_config(config: &Wav2Vec2CtcConfig, stem_bias: bool) -> CharsiuConfig {
    CharsiuConfig {
        hidden_size: config.hidden_size,
        n_layer: config.n_layer,
        n_head: config.n_head,
        ffn_dim: config.intermediate_size,
        vocab_size: config.vocab_size,
        silence_id: 0,
        pad_id: 0,
        sample_rate: SAMPLE_RATE,
        frame_shift_sec: 0.02,
        layer_norm_eps: LAYER_NORM_EPS,
        pos_conv_kernel: POS_KERNEL,
        pos_conv_groups: POS_GROUPS,
        silence_threshold: 1,
        feature_projection_has_layer_norm: true,
        stem_conv_bias: stem_bias,
    }
}

pub(crate) fn bind_family_block(
    file: &GgufFile,
    prefix: &str,
    h: usize,
    ffn: usize,
) -> Result<CharsiuBlock> {
    Ok(CharsiuBlock {
        attn_norm_gamma: tensor(file, &format!("{prefix}.layer_norm.weight"), &[h])?,
        attn_norm_beta: tensor(file, &format!("{prefix}.layer_norm.bias"), &[h])?,
        q_w: tensor(file, &format!("{prefix}.attention.q_proj.weight"), &[h, h])?,
        q_b: tensor(file, &format!("{prefix}.attention.q_proj.bias"), &[h])?,
        k_w: tensor(file, &format!("{prefix}.attention.k_proj.weight"), &[h, h])?,
        k_b: tensor(file, &format!("{prefix}.attention.k_proj.bias"), &[h])?,
        v_w: tensor(file, &format!("{prefix}.attention.v_proj.weight"), &[h, h])?,
        v_b: tensor(file, &format!("{prefix}.attention.v_proj.bias"), &[h])?,
        o_w: tensor(
            file,
            &format!("{prefix}.attention.out_proj.weight"),
            &[h, h],
        )?,
        o_b: tensor(file, &format!("{prefix}.attention.out_proj.bias"), &[h])?,
        ffn_norm_gamma: tensor(file, &format!("{prefix}.final_layer_norm.weight"), &[h])?,
        ffn_norm_beta: tensor(file, &format!("{prefix}.final_layer_norm.bias"), &[h])?,
        fc1_w: tensor(
            file,
            &format!("{prefix}.feed_forward.intermediate_dense.weight"),
            &[ffn, h],
        )?,
        fc1_b: tensor(
            file,
            &format!("{prefix}.feed_forward.intermediate_dense.bias"),
            &[ffn],
        )?,
        fc2_w: tensor(
            file,
            &format!("{prefix}.feed_forward.output_dense.weight"),
            &[h, ffn],
        )?,
        fc2_b: tensor(
            file,
            &format!("{prefix}.feed_forward.output_dense.bias"),
            &[h],
        )?,
    })
}

/// Wav2Vec2 stable/pre-LayerNorm block:
/// `x += attention(LN(x)); x += FFN(LN(x))`.
fn stable_transformer_block(
    hidden: &mut [f32],
    frames: usize,
    cfg: &CharsiuConfig,
    block: &CharsiuBlock,
    compute: &Compute,
) -> Result<()> {
    let h = cfg.hidden_size;
    let heads = cfg.n_head;
    let head_dim = h / heads;
    let mut normalized = hidden.to_vec();
    layer_norm_with_compute_inplace(
        &mut normalized,
        frames,
        h,
        &block.attn_norm_gamma,
        &block.attn_norm_beta,
        cfg.layer_norm_eps,
        compute,
    )?;
    let q =
        linear_forward_with_compute(&normalized, frames, h, &block.q_w, &block.q_b, h, compute)?;
    let k =
        linear_forward_with_compute(&normalized, frames, h, &block.k_w, &block.k_b, h, compute)?;
    let v =
        linear_forward_with_compute(&normalized, frames, h, &block.v_w, &block.v_b, h, compute)?;
    let attention = multi_head_attention(&q, &k, &v, frames, heads, head_dim, compute)?;
    let projected =
        linear_forward_with_compute(&attention, frames, h, &block.o_w, &block.o_b, h, compute)?;
    for (value, residual) in hidden.iter_mut().zip(projected) {
        *value += residual;
    }

    normalized.copy_from_slice(hidden);
    layer_norm_with_compute_inplace(
        &mut normalized,
        frames,
        h,
        &block.ffn_norm_gamma,
        &block.ffn_norm_beta,
        cfg.layer_norm_eps,
        compute,
    )?;
    let intermediate = linear_forward_with_compute(
        &normalized,
        frames,
        h,
        &block.fc1_w,
        &block.fc1_b,
        cfg.ffn_dim,
        compute,
    )?;
    let mut activated = vec![0.0f32; intermediate.len()];
    compute.gelu_f32(&intermediate, &mut activated)?;
    let output = linear_forward_with_compute(
        &activated,
        frames,
        cfg.ffn_dim,
        &block.fc2_w,
        &block.fc2_b,
        h,
        compute,
    )?;
    for (value, residual) in hidden.iter_mut().zip(output) {
        *value += residual;
    }
    Ok(())
}

fn multi_head_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    frames: usize,
    heads: usize,
    head_dim: usize,
    compute: &Compute,
) -> Result<Vec<f32>> {
    let hidden = heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; frames * hidden];
    let mut q_head = vec![0.0f32; frames * head_dim];
    let mut k_head_t = vec![0.0f32; head_dim * frames];
    let mut v_head = vec![0.0f32; frames * head_dim];
    let mut scores = vec![0.0f32; frames * frames];
    let mut probabilities = vec![0.0f32; frames * frames];
    let mut head_output = vec![0.0f32; frames * head_dim];
    for head in 0..heads {
        for frame in 0..frames {
            let source = frame * hidden + head * head_dim;
            let destination = frame * head_dim;
            q_head[destination..destination + head_dim]
                .copy_from_slice(&q[source..source + head_dim]);
            v_head[destination..destination + head_dim]
                .copy_from_slice(&v[source..source + head_dim]);
            for dim in 0..head_dim {
                k_head_t[dim * frames + frame] = k[source + dim];
            }
        }
        compute.gemm_f32(
            frames,
            frames,
            head_dim,
            &q_head,
            &k_head_t,
            None,
            &mut scores,
        )?;
        for score in &mut scores {
            *score *= scale;
        }
        compute.softmax_f32(&scores, &mut probabilities, frames, frames)?;
        compute.gemm_f32(
            frames,
            head_dim,
            frames,
            &probabilities,
            &v_head,
            None,
            &mut head_output,
        )?;
        for frame in 0..frames {
            let source = frame * head_dim;
            let destination = frame * hidden + head * head_dim;
            output[destination..destination + head_dim]
                .copy_from_slice(&head_output[source..source + head_dim]);
        }
    }
    Ok(output)
}

fn stem_attrs(spec: &VariantSpec) -> WaveformFrontendAttrs {
    WaveformFrontendAttrs {
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
        norm: spec.stem_norm,
        conv_bias: spec.stem_bias,
    }
}

pub(crate) fn waveform_frontend_with_compute(
    waveform: &[f32],
    attrs: &WaveformFrontendAttrs,
    weights: &WaveformFrontendWeights,
    compute: &Compute,
) -> Result<Vec<f32>> {
    weights.validate(attrs)?;
    if waveform.len() % attrs.in_channels != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "wav2vec2_ctc: waveform length {} is not divisible by {} channel(s)",
            waveform.len(),
            attrs.in_channels
        )));
    }
    let mut time = waveform.len() / attrs.in_channels;
    let _ = attrs.predict_t_out(time)?;
    let mut current = waveform.to_vec();
    let mut in_channels = attrs.in_channels;
    for (index, (layer, layer_weights)) in attrs.layers.iter().zip(&weights.layers).enumerate() {
        let output_time = (time - layer.kernel) / layer.stride + 1;
        let mut convolution = vec![0.0f32; layer.out_channels * output_time];
        compute.conv1d_f32(
            &current,
            in_channels,
            time,
            &layer_weights.conv_w,
            layer.out_channels,
            layer.kernel,
            attrs.conv_bias.then_some(layer_weights.conv_b.as_slice()),
            layer.stride,
            0,
            &mut convolution,
        )?;
        if attrs.norm.has_group_norm(index) {
            group_norm_each_channel(
                &mut convolution,
                layer.out_channels,
                output_time,
                layer_weights.norm_gamma.as_ref().expect("validated gamma"),
                layer_weights.norm_beta.as_ref().expect("validated beta"),
            );
        } else if attrs.norm.has_layer_norm(index) {
            let mut frame_major =
                transpose_channel_to_frame(&convolution, layer.out_channels, output_time);
            layer_norm_with_compute_inplace(
                &mut frame_major,
                output_time,
                layer.out_channels,
                layer_weights.norm_gamma.as_ref().expect("validated gamma"),
                layer_weights.norm_beta.as_ref().expect("validated beta"),
                LAYER_NORM_EPS,
                compute,
            )?;
            convolution = transpose_frame_to_channel(&frame_major, output_time, layer.out_channels);
        }
        let mut activated = vec![0.0f32; convolution.len()];
        compute.gelu_f32(&convolution, &mut activated)?;
        current = activated;
        time = output_time;
        in_channels = layer.out_channels;
    }
    Ok(transpose_channel_to_frame(&current, in_channels, time))
}

fn group_norm_each_channel(
    values: &mut [f32],
    channels: usize,
    time: usize,
    gamma: &[f32],
    beta: &[f32],
) {
    for channel in 0..channels {
        let row = &mut values[channel * time..(channel + 1) * time];
        let mean = row.iter().copied().sum::<f32>() / time as f32;
        let variance = row
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .sum::<f32>()
            / time as f32;
        let inverse = 1.0 / (variance + LAYER_NORM_EPS).sqrt();
        for value in row {
            *value = (*value - mean) * inverse * gamma[channel] + beta[channel];
        }
    }
}

pub(crate) fn transpose_channel_to_frame(values: &[f32], channels: usize, time: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; values.len()];
    for channel in 0..channels {
        for frame in 0..time {
            output[frame * channels + channel] = values[channel * time + frame];
        }
    }
    output
}

pub(crate) fn transpose_frame_to_channel(values: &[f32], time: usize, channels: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; values.len()];
    for frame in 0..time {
        for channel in 0..channels {
            output[channel * time + frame] = values[frame * channels + channel];
        }
    }
    output
}

pub(crate) fn zero_mean_unit_var(pcm: &[f32]) -> Vec<f32> {
    let mean = pcm.iter().copied().sum::<f32>() / pcm.len() as f32;
    let variance = pcm
        .iter()
        .map(|&value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f32>()
        / pcm.len() as f32;
    let scale = 1.0 / (variance + NORMALIZATION_EPS).sqrt();
    pcm.iter().map(|&value| (value - mean) * scale).collect()
}

pub(crate) fn validate_pcm(pcm: &[f32]) -> Result<()> {
    if pcm.is_empty() {
        return Err(VokraError::InvalidArgument(
            "wav2vec2_ctc: pcm slice is empty".to_owned(),
        ));
    }
    if let Some((index, value)) = pcm
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(format!(
            "wav2vec2_ctc: non-finite PCM value {value} at index {index}"
        )));
    }
    Ok(())
}

pub(crate) fn reject_non_finite(label: &str, values: &[f32]) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(format!(
            "wav2vec2_ctc: {label} contains non-finite value {value} at index {index}"
        )));
    }
    Ok(())
}

pub(crate) fn fold_weight_norm_dim2(
    g: &[f32],
    v: &[f32],
    out: usize,
    input_per_group: usize,
    kernel: usize,
) -> Result<Vec<f32>> {
    let mut weight = vec![0.0f32; v.len()];
    for (tap, gain) in g.iter().copied().enumerate().take(kernel) {
        let mut squared = 0.0f64;
        for output in 0..out {
            for input in 0..input_per_group {
                let index = (output * input_per_group + input) * kernel + tap;
                let value = f64::from(v[index]);
                squared += value * value;
            }
        }
        let norm = squared.sqrt();
        if norm == 0.0 {
            return Err(VokraError::ModelLoad(format!(
                "wav2vec2_ctc: positional weight_v tap {tap} has zero norm"
            )));
        }
        let scale = (f64::from(gain) / norm) as f32;
        for output in 0..out {
            for input in 0..input_per_group {
                let index = (output * input_per_group + input) * kernel + tap;
                weight[index] = v[index] * scale;
            }
        }
    }
    Ok(weight)
}

pub(crate) fn parse_vocab(json: &[u8], expected: usize) -> Result<Vec<String>> {
    let root = vokra_core::json::parse(json).map_err(|error| {
        VokraError::ModelLoad(format!("wav2vec2_ctc: embedded vocabulary JSON: {error}"))
    })?;
    let entries = root.as_object().ok_or_else(|| {
        VokraError::ModelLoad("wav2vec2_ctc: embedded vocabulary is not an object".to_owned())
    })?;
    let mut vocab = vec![None; expected];
    for (token, id) in entries {
        let id = id.as_u64().ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "wav2vec2_ctc: embedded vocabulary id for {token:?} is not u64"
            ))
        })? as usize;
        if id >= expected {
            return Err(VokraError::ModelLoad(format!(
                "wav2vec2_ctc: embedded vocabulary id {id} for {token:?} >= {expected}"
            )));
        }
        if vocab[id].replace(token.clone()).is_some() {
            return Err(VokraError::ModelLoad(format!(
                "wav2vec2_ctc: embedded vocabulary duplicates id {id}"
            )));
        }
    }
    vocab
        .into_iter()
        .enumerate()
        .map(|(id, token)| {
            token.ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "wav2vec2_ctc: embedded vocabulary is missing id {id}"
                ))
            })
        })
        .collect()
}

pub(crate) fn decode_tokens(tokens: &[u32], vocab: &[String]) -> Result<String> {
    let mut text = String::new();
    for &token in tokens {
        let piece = vocab.get(token as usize).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "wav2vec2_ctc: decoded token {token} is outside vocabulary {}",
                vocab.len()
            ))
        })?;
        match piece.as_str() {
            "<pad>" | "<s>" | "</s>" => {}
            "|" => text.push(' '),
            other => text.push_str(other),
        }
    }
    Ok(text.split_whitespace().collect::<Vec<_>>().join(" "))
}

pub(crate) fn tensor(file: &GgufFile, name: &str, dims: &[usize]) -> Result<Vec<f32>> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("wav2vec2_ctc: missing tensor `{name}`")))?;
    let expected = dims.iter().map(|&dim| dim as u64).collect::<Vec<_>>();
    if info.dimensions != expected {
        return Err(VokraError::ModelLoad(format!(
            "wav2vec2_ctc: tensor `{name}` has dims {:?}, expected {expected:?}",
            info.dimensions
        )));
    }
    file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!("wav2vec2_ctc: reading tensor `{name}`: {error}"))
    })
}

fn validate_metadata(file: &GgufFile, spec: &VariantSpec) -> Result<()> {
    if metadata_matches_spec(file, spec) {
        return Ok(());
    }
    Err(VokraError::ModelLoad(format!(
        "wav2vec2_ctc: metadata for {} does not match its verified {}/{}/{}/{} topology",
        spec.model_id, spec.hidden, spec.layers, spec.heads, spec.ffn
    )))
}

fn metadata_matches_spec(file: &GgufFile, spec: &VariantSpec) -> bool {
    meta_is_string(file, chunks::KEY_MODEL_NAME, spec.model_id)
        && meta_is_string(file, KEY_UPSTREAM_HF, spec.upstream_hf)
        && meta_is_u64(file, KEY_HIDDEN, spec.hidden as u64)
        && meta_is_u64(file, KEY_LAYERS, spec.layers as u64)
        && meta_is_u64(file, KEY_HEADS, spec.heads as u64)
        && meta_is_u64(file, KEY_FFN, spec.ffn as u64)
        && meta_is_u64(file, KEY_VOCAB, spec.vocab as u64)
        && meta_is_f32(file, KEY_EPS, LAYER_NORM_EPS)
        && meta_is_string(
            file,
            KEY_FEAT_NORM,
            if spec.stem_norm == Norm::LayerAll {
                "layer"
            } else {
                "group"
            },
        )
        && meta_is_bool(file, KEY_STABLE, spec.stable)
        && meta_is_string(file, KEY_ACT, "gelu")
        && meta_is_u64(file, KEY_POS_KERNEL, POS_KERNEL as u64)
        && meta_is_u64(file, KEY_POS_GROUPS, POS_GROUPS as u64)
        && meta_is_bool(file, KEY_HAS_HEAD, spec.has_head)
        && meta_is_u64(file, KEY_STEM_LAYERS, CONV_DIM.len() as u64)
        && meta_is_u32_array(file, KEY_CONV_DIM, &CONV_DIM)
        && meta_is_u32_array(file, KEY_CONV_KERNEL, &CONV_KERNEL)
        && meta_is_u32_array(file, KEY_CONV_STRIDE, &CONV_STRIDE)
}

fn metadata_matches_base_stamp(file: &GgufFile) -> bool {
    meta_is_string(file, chunks::KEY_MODEL_NAME, "wav2vec2-base-960h")
        && meta_is_string(file, KEY_UPSTREAM_HF, "facebook/wav2vec2-base-960h")
        && meta_is_u64(file, KEY_HIDDEN, 768)
        && meta_is_u64(file, KEY_LAYERS, 12)
        && meta_is_u64(file, KEY_HEADS, 12)
        && meta_is_u64(file, KEY_FFN, 3072)
        && meta_is_u64(file, KEY_VOCAB, 32)
        && meta_is_f32(file, KEY_EPS, LAYER_NORM_EPS)
        && meta_is_string(file, KEY_FEAT_NORM, "group")
        && meta_is_bool(file, KEY_STABLE, false)
        && meta_is_string(file, KEY_ACT, "gelu")
        && meta_is_u64(file, KEY_POS_KERNEL, 128)
        && meta_is_u64(file, KEY_POS_GROUPS, 16)
        && meta_is_bool(file, KEY_HAS_HEAD, true)
        && meta_is_u64(file, KEY_STEM_LAYERS, 7)
        && meta_is_u32_array(file, KEY_CONV_DIM, &CONV_DIM)
        && meta_is_u32_array(file, KEY_CONV_KERNEL, &CONV_KERNEL)
        && meta_is_u32_array(file, KEY_CONV_STRIDE, &CONV_STRIDE)
}

fn hubert_metadata_matches_spec(file: &GgufFile) -> bool {
    meta_is_u64(file, HUBERT_KEY_HIDDEN, 1024)
        && meta_is_u64(file, HUBERT_KEY_LAYERS, 24)
        && meta_is_u64(file, HUBERT_KEY_HEADS, 16)
        && meta_is_u64(file, HUBERT_KEY_FFN, 4096)
        && meta_is_u64(file, HUBERT_KEY_VOCAB, 32)
        && meta_is_f32(file, HUBERT_KEY_EPS, LAYER_NORM_EPS)
        && meta_is_string(file, HUBERT_KEY_FEAT_NORM, "layer")
        && meta_is_bool(file, HUBERT_KEY_STABLE, true)
        && meta_is_string(file, HUBERT_KEY_ACT, "gelu")
        && meta_is_u64(file, HUBERT_KEY_POS_KERNEL, POS_KERNEL as u64)
        && meta_is_u64(file, HUBERT_KEY_POS_GROUPS, POS_GROUPS as u64)
        && meta_is_bool(file, HUBERT_KEY_HAS_HEAD, true)
        && meta_is_u64(file, HUBERT_KEY_STEM_LAYERS, CONV_DIM.len() as u64)
        && meta_is_u32_array(file, HUBERT_KEY_CONV_DIM, &CONV_DIM)
        && meta_is_u32_array(file, HUBERT_KEY_CONV_KERNEL, &CONV_KERNEL)
        && meta_is_u32_array(file, HUBERT_KEY_CONV_STRIDE, &CONV_STRIDE)
}

fn meta_string<'a>(file: &'a GgufFile, key: &str) -> Result<&'a str> {
    file.get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| VokraError::ModelLoad(format!("wav2vec2_ctc: missing string `{key}`")))
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = meta_string(file, key)?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "wav2vec2_ctc: `{key}` is {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn meta_is_string(file: &GgufFile, key: &str, expected: &str) -> bool {
    file.get(key).and_then(GgufMetadataValue::as_str) == Some(expected)
}

fn meta_is_u64(file: &GgufFile, key: &str, expected: u64) -> bool {
    file.get(key).and_then(GgufMetadataValue::as_u64) == Some(expected)
}

fn meta_is_bool(file: &GgufFile, key: &str, expected: bool) -> bool {
    file.get(key).and_then(GgufMetadataValue::as_bool) == Some(expected)
}

fn meta_is_f32(file: &GgufFile, key: &str, expected: f32) -> bool {
    matches!(file.get(key), Some(GgufMetadataValue::F32(value)) if value.to_bits() == expected.to_bits())
}

fn meta_is_u32_array(file: &GgufFile, key: &str, expected: &[usize]) -> bool {
    let Some(array) = file.get(key).and_then(GgufMetadataValue::as_array) else {
        return false;
    };
    array.values.len() == expected.len()
        && array
            .values
            .iter()
            .zip(expected)
            .all(|(value, &expected)| value.as_u64() == Some(expected as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_vocabularies_are_dense_and_pinned() {
        for spec in VARIANTS.iter().filter(|spec| spec.has_head) {
            let vocab = parse_vocab(spec.vocab_json.unwrap(), spec.vocab).unwrap();
            assert_eq!(vocab.len(), spec.vocab, "{}", spec.model_id);
            assert_eq!(vocab[BLANK_ID], "<pad>", "{}", spec.model_id);
        }
    }

    #[test]
    fn weight_norm_fold_is_dim2() {
        let g = [2.0, 3.0];
        let v = [3.0, 0.0, 4.0, 4.0];
        let folded = fold_weight_norm_dim2(&g, &v, 2, 1, 2).unwrap();
        assert!((folded[0] - 1.2).abs() < 1e-6);
        assert_eq!(folded[1], 0.0);
        assert!((folded[2] - 1.6).abs() < 1e-6);
        assert!((folded[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn decoder_replaces_word_delimiter_and_skips_boundaries() {
        let vocab = vec![
            "<pad>".to_owned(),
            "<s>".to_owned(),
            "</s>".to_owned(),
            "<unk>".to_owned(),
            "|".to_owned(),
            "H".to_owned(),
            "I".to_owned(),
        ];
        assert_eq!(decode_tokens(&[1, 5, 6, 4, 5, 2], &vocab).unwrap(), "HI H");
    }

    #[test]
    fn all_public_variants_have_unique_ids() {
        let mut ids = std::collections::BTreeSet::new();
        for spec in VARIANTS {
            assert!(ids.insert(spec.model_id));
            assert_eq!(spec.hidden % spec.heads, 0);
            assert_eq!(spec.hidden % POS_GROUPS, 0);
        }
    }
}
