//! Native Audio Spectrogram Transformer (AST) AudioSet classifier.
//!
//! This module binds the published `vokra/ast-finetuned-audioset` GGUF by its
//! complete tensor name/shape manifest and implements the official Hugging
//! Face AST feature extractor plus the DeiT-derived encoder and 527-way head.
//! CPU and Apple Metal use the same validated topology through [`Compute`].
//! Selecting an unsupported backend fails before inference; there is no
//! implicit CPU fallback.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{CompliancePolicy, LicenseClass, Result, VokraError, check_weight_license};
use vokra_ops::kaldi_fbank::{KaldiFbankOpts, KaldiFbankWindow, kaldi_fbank_with_window};
use vokra_ops::vit::{
    GeluKind, PatchEmbedWeights, PosEmbedPolicy, ViTAttnWeights, ViTAttrs, ViTBackendOps,
    ViTBlockWeights, ViTEncoder, ViTMlpWeights, ViTWeights,
};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec, load_tensor};

/// Public GGUF architecture tag.
pub const ARCH: &str = "ast";
/// Public GGUF model name.
pub const NAME: &str = "ast-finetuned-audioset";
/// Fixed upstream repository recorded by the public artifact.
pub const UPSTREAM_HF: &str = "MIT/ast-finetuned-audioset-10-10-0.4593";
/// Input sample rate of the official feature extractor.
pub const SAMPLE_RATE: u32 = 16_000;
/// Fixed feature-frame extent consumed by the released position table.
pub const MAX_LENGTH: usize = 1_024;
/// AudioSet output cardinality of this checkpoint.
pub const NUM_LABELS: usize = 527;

const HIDDEN: usize = 768;
const LAYERS: usize = 12;
const HEADS: usize = 12;
const INTERMEDIATE: usize = 3_072;
const PATCH: usize = 16;
const STRIDE: usize = 10;
const NUM_MELS: usize = 128;
const PREFIX_TOKENS: usize = 2;
const SEQUENCE_LENGTH: usize = 1_214;
const LAYER_NORM_EPS: f32 = 1.0e-12;
const NORMALIZE_MEAN: f32 = -4.267_739_3;
const NORMALIZE_STD: f32 = 4.568_997_4;
const TENSOR_PREFIX: &str = "audio_spectrogram_transformer.";

/// Complete learned-op set used by the AST forward.
pub const AST_HOT_OPS: &[HotOp] = &[HotOp::Gemm, HotOp::Softmax, HotOp::LayerNorm, HotOp::Gelu];

const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: "ast-finetuned-audioset",
    arch: ARCH,
    model_name: NAME,
    model_name_alias: None,
    tensor_count: 203,
    manifest_sha256: [
        0xcd, 0x67, 0x8a, 0x35, 0x77, 0xfa, 0x41, 0xe5, 0x05, 0x2a, 0xd8, 0xb5, 0x9d, 0x33, 0xea,
        0xf4, 0x5a, 0x86, 0xa3, 0x9e, 0x60, 0x1c, 0x94, 0x2a, 0xe0, 0x55, 0x1a, 0x23, 0x55, 0xe6,
        0x4a, 0x29,
    ],
};

#[derive(Debug, Clone)]
struct AstHead {
    ln_gamma: Vec<f32>,
    ln_beta: Vec<f32>,
    dense_w: Vec<f32>,
    dense_b: Vec<f32>,
}

/// Strict real-weight AST AudioSet classifier.
#[derive(Debug, Clone)]
pub struct AstAudioSet {
    encoder: ViTEncoder,
    head: AstHead,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl AstAudioSet {
    /// Bind the audited public GGUF under the strict runtime licence policy.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        Self::from_gguf_with_policy(file, &CompliancePolicy::strict())
    }

    /// Policy-selectable twin of [`Self::from_gguf`].
    pub fn from_gguf_with_policy(file: &GgufFile, policy: &CompliancePolicy) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        require_string(file, "vokra.model.category", "classification")?;
        require_string(file, chunks::KEY_PROVENANCE_MODEL_ID, NAME)?;
        require_string(file, "vokra.provenance.upstream_hf", UPSTREAM_HF)?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "bsd-3-clause")?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        )?;
        check_weight_license(file, policy)?;

        let embeddings = format!("{TENSOR_PREFIX}embeddings.");
        let cls = load_tensor(
            file,
            NAME,
            &format!("{embeddings}cls_token"),
            &[1, 1, HIDDEN],
        )?;
        let distillation = load_tensor(
            file,
            NAME,
            &format!("{embeddings}distillation_token"),
            &[1, 1, HIDDEN],
        )?;
        let mut prepended_tokens = Vec::with_capacity(PREFIX_TOKENS * HIDDEN);
        prepended_tokens.extend_from_slice(&cls);
        prepended_tokens.extend_from_slice(&distillation);
        let pos_embed = load_tensor(
            file,
            NAME,
            &format!("{embeddings}position_embeddings"),
            &[1, SEQUENCE_LENGTH, HIDDEN],
        )?;

        let projection = format!("{embeddings}patch_embeddings.projection.");
        let patch_embed = PatchEmbedWeights {
            proj_w: load_tensor(
                file,
                NAME,
                &format!("{projection}weight"),
                &[HIDDEN, 1, PATCH, PATCH],
            )?,
            proj_b: Some(load_tensor(
                file,
                NAME,
                &format!("{projection}bias"),
                &[HIDDEN],
            )?),
        };

        let mut blocks = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            let base = format!("{TENSOR_PREFIX}encoder.layer.{layer}.");
            let ln1 = format!("{base}layernorm_before.");
            let ln2 = format!("{base}layernorm_after.");
            let attention = format!("{base}attention.attention.");
            let attention_out = format!("{base}attention.output.dense.");
            let mlp_in = format!("{base}intermediate.dense.");
            let mlp_out = format!("{base}output.dense.");
            blocks.push(ViTBlockWeights {
                ln1_gamma: load_tensor(file, NAME, &format!("{ln1}weight"), &[HIDDEN])?,
                ln1_beta: load_tensor(file, NAME, &format!("{ln1}bias"), &[HIDDEN])?,
                attn: ViTAttnWeights {
                    wq: load_tensor(
                        file,
                        NAME,
                        &format!("{attention}query.weight"),
                        &[HIDDEN, HIDDEN],
                    )?,
                    bq: Some(load_tensor(
                        file,
                        NAME,
                        &format!("{attention}query.bias"),
                        &[HIDDEN],
                    )?),
                    wk: load_tensor(
                        file,
                        NAME,
                        &format!("{attention}key.weight"),
                        &[HIDDEN, HIDDEN],
                    )?,
                    bk: Some(load_tensor(
                        file,
                        NAME,
                        &format!("{attention}key.bias"),
                        &[HIDDEN],
                    )?),
                    wv: load_tensor(
                        file,
                        NAME,
                        &format!("{attention}value.weight"),
                        &[HIDDEN, HIDDEN],
                    )?,
                    bv: Some(load_tensor(
                        file,
                        NAME,
                        &format!("{attention}value.bias"),
                        &[HIDDEN],
                    )?),
                    wo: load_tensor(
                        file,
                        NAME,
                        &format!("{attention_out}weight"),
                        &[HIDDEN, HIDDEN],
                    )?,
                    bo: Some(load_tensor(
                        file,
                        NAME,
                        &format!("{attention_out}bias"),
                        &[HIDDEN],
                    )?),
                },
                ln2_gamma: load_tensor(file, NAME, &format!("{ln2}weight"), &[HIDDEN])?,
                ln2_beta: load_tensor(file, NAME, &format!("{ln2}bias"), &[HIDDEN])?,
                mlp: ViTMlpWeights {
                    w1: load_tensor(
                        file,
                        NAME,
                        &format!("{mlp_in}weight"),
                        &[INTERMEDIATE, HIDDEN],
                    )?,
                    b1: Some(load_tensor(
                        file,
                        NAME,
                        &format!("{mlp_in}bias"),
                        &[INTERMEDIATE],
                    )?),
                    w2: load_tensor(
                        file,
                        NAME,
                        &format!("{mlp_out}weight"),
                        &[HIDDEN, INTERMEDIATE],
                    )?,
                    b2: Some(load_tensor(
                        file,
                        NAME,
                        &format!("{mlp_out}bias"),
                        &[HIDDEN],
                    )?),
                },
            });
        }

        let final_ln = format!("{TENSOR_PREFIX}layernorm.");
        let attrs = ViTAttrs {
            embed_dim: HIDDEN,
            depth: LAYERS,
            n_heads: HEADS,
            mlp_ratio: INTERMEDIATE as f32 / HIDDEN as f32,
            patch_h: PATCH,
            patch_w: PATCH,
            stride_h: STRIDE,
            stride_w: STRIDE,
            n_prepended_tokens: PREFIX_TOKENS,
            layer_norm_eps: LAYER_NORM_EPS,
            gelu: GeluKind::Erf,
            pos_embed_policy: PosEmbedPolicy::RequireExact,
        };
        let encoder = ViTEncoder::new(
            attrs,
            ViTWeights {
                patch_embed,
                prepended_tokens,
                pos_embed,
                blocks,
                final_ln_gamma: load_tensor(file, NAME, &format!("{final_ln}weight"), &[HIDDEN])?,
                final_ln_beta: load_tensor(file, NAME, &format!("{final_ln}bias"), &[HIDDEN])?,
            },
        )?;

        let head = AstHead {
            ln_gamma: load_tensor(file, NAME, "classifier.layernorm.weight", &[HIDDEN])?,
            ln_beta: load_tensor(file, NAME, "classifier.layernorm.bias", &[HIDDEN])?,
            dense_w: load_tensor(file, NAME, "classifier.dense.weight", &[NUM_LABELS, HIDDEN])?,
            dense_b: load_tensor(file, NAME, "classifier.dense.bias", &[NUM_LABELS])?,
        };

        Ok(Self {
            encoder,
            head,
            weight_license: checkpoint.weight_license(),
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and binds an official GGUF.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Selects one backend for the complete classifier graph.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Selected inference backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Stamped public-artifact licence class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Runs the official feature extractor and returns 527 raw AudioSet logits.
    pub fn classify_pcm(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        if sample_rate != SAMPLE_RATE {
            return Err(VokraError::InvalidArgument(format!(
                "AST expects {SAMPLE_RATE} Hz mono PCM, got {sample_rate} Hz; resample explicitly before inference"
            )));
        }
        let compute = match self.backend {
            BackendKind::Cpu | BackendKind::Metal => {
                Compute::for_backend(self.backend, AST_HOT_OPS)?
            }
            other => {
                return Err(VokraError::UnsupportedOp(format!(
                    "AST supports only CPU and Apple Metal in this runtime, got {other:?}; no CPU fallback was used"
                )));
            }
        };
        let backend = ComputeViTBackend { compute: &compute };
        let frame_major = extract_features(pcm, sample_rate)?;
        let mel = transpose_features(&frame_major);
        let (hidden, grid) = self
            .encoder
            .forward_with_backend(&mel, NUM_MELS, MAX_LENGTH, &backend)?;
        if grid.n_tokens(PREFIX_TOKENS) != SEQUENCE_LENGTH {
            return Err(VokraError::InvalidArgument(format!(
                "AST patch grid produced {} tokens, expected {SEQUENCE_LENGTH}",
                grid.n_tokens(PREFIX_TOKENS)
            )));
        }

        let mut pooled = vec![0.0f32; HIDDEN];
        for index in 0..HIDDEN {
            pooled[index] = (hidden[index] + hidden[HIDDEN + index]) * 0.5;
        }
        let mut normed = vec![0.0f32; HIDDEN];
        backend.layer_norm_f32(
            &pooled,
            &mut normed,
            1,
            HIDDEN,
            &self.head.ln_gamma,
            &self.head.ln_beta,
            LAYER_NORM_EPS,
        )?;
        let mut logits = vec![0.0f32; NUM_LABELS];
        backend.linear_f32(
            &normed,
            &self.head.dense_w,
            Some(&self.head.dense_b),
            1,
            HIDDEN,
            NUM_LABELS,
            &mut logits,
        )?;
        Ok(logits)
    }
}

/// Official AST frontend: TorchAudio Kaldi fbank, right pad/truncate to 1024,
/// AudioSet normalization, then transpose `[frames, mels] -> [mels, frames]`.
/// Extracts the official normalized AST input matrix as row-major
/// `[MAX_LENGTH, 128]`. This is the exact tensor accepted by the upstream
/// `ASTForAudioClassification` forward before its patch-convolution transpose.
pub fn extract_features(pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
    if sample_rate != SAMPLE_RATE {
        return Err(VokraError::InvalidArgument(format!(
            "AST feature extractor expects {SAMPLE_RATE} Hz mono PCM, got {sample_rate} Hz"
        )));
    }
    let opts = KaldiFbankOpts::ast_audioset();
    let (features, frames) = kaldi_fbank_with_window(pcm, &opts, KaldiFbankWindow::Hanning)?;
    let kept = frames.min(MAX_LENGTH);
    let mut frame_major = vec![0.0f32; MAX_LENGTH * NUM_MELS];
    frame_major[..kept * NUM_MELS].copy_from_slice(&features[..kept * NUM_MELS]);
    let denom = NORMALIZE_STD * 2.0;
    for value in &mut frame_major {
        *value = (*value - NORMALIZE_MEAN) / denom;
    }
    Ok(frame_major)
}

fn transpose_features(frame_major: &[f32]) -> Vec<f32> {
    debug_assert_eq!(frame_major.len(), MAX_LENGTH * NUM_MELS);
    let mut mel_major = vec![0.0f32; frame_major.len()];
    for frame in 0..MAX_LENGTH {
        for mel in 0..NUM_MELS {
            mel_major[mel * MAX_LENGTH + frame] = frame_major[frame * NUM_MELS + mel];
        }
    }
    mel_major
}

struct ComputeViTBackend<'a> {
    compute: &'a Compute,
}

impl ViTBackendOps for ComputeViTBackend<'_> {
    fn linear_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let mut transposed = vec![0.0f32; weight.len()];
        for out in 0..out_dim {
            for inner in 0..in_dim {
                transposed[inner * out_dim + out] = weight[out * in_dim + inner];
            }
        }
        self.compute
            .gemm_f32(rows, out_dim, in_dim, input, &transposed, bias, output)
    }

    fn matmul_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        left: &[f32],
        right: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        self.compute.gemm_f32(m, n, k, left, right, None, output)
    }

    fn softmax_f32(
        &self,
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        self.compute.softmax_f32(input, output, rows, cols)
    }

    fn layer_norm_f32(
        &self,
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<()> {
        self.compute
            .layer_norm_f32(input, output, rows, cols, gamma, beta, eps)
    }

    fn gelu_f32(&self, kind: GeluKind, input: &[f32], output: &mut [f32]) -> Result<()> {
        if kind != GeluKind::Erf {
            return Err(VokraError::UnsupportedOp(
                "AST Compute backend supports the official exact-erf GELU only".to_owned(),
            ));
        }
        self.compute.gelu_f32(input, output)
    }
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_str);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "{NAME}: metadata `{key}`={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_geometry_is_exact() {
        assert_eq!((NUM_MELS - PATCH) / STRIDE + 1, 12);
        assert_eq!((MAX_LENGTH - PATCH) / STRIDE + 1, 101);
        assert_eq!(12 * 101 + PREFIX_TOKENS, SEQUENCE_LENGTH);
        assert_eq!(SPEC.tensor_count, 203);
    }

    #[test]
    fn hot_ops_are_cpu_and_metal_complete() {
        Compute::for_backend(BackendKind::Cpu, AST_HOT_OPS)
            .expect("CPU covers the complete AST graph");
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        match Compute::for_backend(BackendKind::Metal, AST_HOT_OPS) {
            Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(error) => panic!("AST has a Metal coverage gap: {error}"),
        }
    }

    #[test]
    fn frontend_pads_then_normalizes_like_transformers() {
        let pcm = vec![0.0f32; 400];
        let features = extract_features(&pcm, SAMPLE_RATE).expect("one exact frame");
        assert_eq!(features.len(), NUM_MELS * MAX_LENGTH);
        let normalized_padding = (0.0 - NORMALIZE_MEAN) / (NORMALIZE_STD * 2.0);
        assert!((features[NUM_MELS] - normalized_padding).abs() <= f32::EPSILON);
    }
}
