//! Native Parler-TTS Mini English / Multilingual runtime contract.
//!
//! Both public Vokra GGUFs are self-contained composites: FLAN-T5-large,
//! Parler's nine-codebook causal decoder, the learned prompt embedding and a
//! 44.1 kHz DAC decoder live in one file. This module accepts only the two
//! audited complete tensor manifests. It does not infer a variant from loose
//! metadata or silently fetch the DAC that older converter documentation
//! described as external.
//!
//! The English release carries nine separate LM heads and the older
//! weight-normalized Descript DAC layout; multilingual v1.1 carries one fused
//! LM head and the plain-convolution Transformers DAC layout. Those are
//! authenticated release differences, not
//! runtime heuristics. The shared text/LM path and both embedded DAC layouts
//! execute end to end on CPU or Metal; an uncovered backend fails the complete
//! op set before resident weights are decoded.

mod codec;
mod generation;

pub use codec::ParlerSynthesis;
pub use generation::{ParlerGeneratedCodes, ParlerGenerationConfig};

use std::path::Path;
use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue, GgufTensorInfo, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::audiocraft_lm::{AudioCraftLmConfig, AudioCraftLmDecoder};
use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};
use crate::t5_encoder::{FLAN_T5_LARGE_CONFIG, T5Encoder};

use self::codec::EmbeddedDac;

/// GGUF architecture shared by both releases.
pub const ARCH: &str = "parler_tts";
/// Model-zoo category.
pub const CATEGORY: &str = "tts";
/// Released DAC sample rate.
pub const SAMPLE_RATE: u32 = 44_100;
/// Parler decoder codebooks.
pub const NUM_CODEBOOKS: usize = 9;
/// Valid DAC entries in each codebook.
pub const CODEBOOK_SIZE: usize = 1_024;
/// Decoder logits per codebook, including reserved control-token rows.
pub const DECODER_VOCAB_SIZE: usize = 1_088;
/// Official decoder start/BOS token.
pub const BOS_TOKEN_ID: u32 = 1_025;
/// Official EOS and delay-padding token.
pub const PAD_EOS_TOKEN_ID: u32 = 1_024;

const DECODER_CONFIG: AudioCraftLmConfig = AudioCraftLmConfig {
    d_model: 1_024,
    num_layers: 24,
    n_heads: 16,
    ffn_dim: 4_096,
    vocab_size: DECODER_VOCAB_SIZE,
    num_codebooks: NUM_CODEBOOKS,
};

/// Learned operations required by the text encoder and Parler decoder, before
/// the embedded DAC slice is added. Unsupported backends fail this complete
/// set before any large tensor is decoded.
pub const PARLER_LM_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::RmsNorm,
    HotOp::GeluNew,
    HotOp::LayerNorm,
    HotOp::Gelu,
];

/// Complete learned-op set for end-to-end Parler code generation and embedded
/// DAC waveform synthesis.
pub const PARLER_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::RmsNorm,
    HotOp::GeluNew,
    HotOp::LayerNorm,
    HotOp::Gelu,
    HotOp::DacRvq,
    HotOp::Conv1d,
    HotOp::SnakeActivation,
];

const KEY_VARIANT: &str = "vokra.parler.variant";
const KEY_PROMPT_VOCAB: &str = "vokra.parler.vocab_size";
const KEY_PROMPT_CROSS_ATTENTION: &str = "vokra.parler.prompt_cross_attention";
const KEY_TEXT_VOCAB: &str = "vokra.parler.text_encoder.vocab_size";
const KEY_TEXT_D_MODEL: &str = "vokra.parler.text_encoder.d_model";
const KEY_TEXT_D_FF: &str = "vokra.parler.text_encoder.d_ff";
const KEY_TEXT_LAYERS: &str = "vokra.parler.text_encoder.num_layers";
const KEY_TEXT_HEADS: &str = "vokra.parler.text_encoder.num_heads";
const KEY_DECODER_VOCAB: &str = "vokra.parler.decoder.vocab_size";
const KEY_DECODER_HIDDEN: &str = "vokra.parler.decoder.hidden_size";
const KEY_DECODER_FFN: &str = "vokra.parler.decoder.ffn_dim";
const KEY_DECODER_LAYERS: &str = "vokra.parler.decoder.num_hidden_layers";
const KEY_DECODER_HEADS: &str = "vokra.parler.decoder.num_attention_heads";
const KEY_DECODER_KV_HEADS: &str = "vokra.parler.decoder.num_key_value_heads";
const KEY_DECODER_CODEBOOKS: &str = "vokra.parler.decoder.num_codebooks";
const KEY_CODEC_SAMPLE_RATE: &str = "vokra.parler.audio_encoder.sampling_rate";
const KEY_CODEC_CODEBOOK_SIZE: &str = "vokra.parler.audio_encoder.codebook_size";

/// Authenticated public Parler release.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParlerVariant {
    /// `parler-tts/parler-tts-mini-v1`, English prompt vocabulary and
    /// separate LM heads.
    MiniV1English,
    /// `parler-tts/parler-tts-mini-multilingual-v1.1`, 90,714-entry prompt
    /// vocabulary and fused LM heads.
    MiniMultilingualV11,
}

impl ParlerVariant {
    /// Canonical Vokra model name stamped in the public GGUF.
    pub const fn model_name(self) -> &'static str {
        match self {
            Self::MiniV1English => "parler-tts-mini-v1",
            Self::MiniMultilingualV11 => "parler-tts-mini-multilingual-v1.1",
        }
    }

    /// Exact release variant metadata.
    pub const fn variant_tag(self) -> &'static str {
        match self {
            Self::MiniV1English => "mini-v1-en",
            Self::MiniMultilingualV11 => "mini-multilingual",
        }
    }

    /// Immutable upstream Hugging Face repository.
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::MiniV1English => "parler-tts/parler-tts-mini-v1",
            Self::MiniMultilingualV11 => "parler-tts/parler-tts-mini-multilingual-v1.1",
        }
    }

    /// Audited public `vokra/*` revision containing the GGUF.
    pub const fn public_gguf_revision(self) -> &'static str {
        match self {
            Self::MiniV1English => "cb02a124c8d125231b396a293608f2488ae2e4d2",
            Self::MiniMultilingualV11 => "6f0f56788f06e6d514e0fab8530663b8af8b1fe2",
        }
    }

    /// Learned prompt-embedding vocabulary size.
    pub const fn prompt_vocab_size(self) -> usize {
        match self {
            Self::MiniV1English => 32_128,
            Self::MiniMultilingualV11 => 90_714,
        }
    }

    /// Whether `decoder.lm_heads.weight` is fused across codebooks.
    pub const fn fused_lm_heads(self) -> bool {
        matches!(self, Self::MiniMultilingualV11)
    }

    /// Complete public tensor count, including the embedded DAC.
    pub const fn tensor_count(self) -> usize {
        match self {
            Self::MiniV1English => 926,
            Self::MiniMultilingualV11 => 840,
        }
    }

    /// SHA-256 of the complete sorted tensor-name/shape manifest.
    pub const fn tensor_manifest_sha256(self) -> &'static str {
        match self {
            Self::MiniV1English => {
                "62bf9fd0a48215c0376deb81771bf7cc8da76133a5e5e84caa49ee6506c49a17"
            }
            Self::MiniMultilingualV11 => {
                "8b6f25596f945988f48aa23509f18b1273162192ef6bfa47019b2eeacc2432e4"
            }
        }
    }

    const fn from_tensor_count(count: usize) -> Option<Self> {
        match count {
            926 => Some(Self::MiniV1English),
            840 => Some(Self::MiniMultilingualV11),
            _ => None,
        }
    }

    const fn spec(self) -> StrictCheckpointSpec {
        match self {
            Self::MiniV1English => StrictCheckpointSpec {
                label: "parler/mini-v1-en",
                arch: ARCH,
                model_name: "parler-tts-mini-v1",
                model_name_alias: None,
                tensor_count: 926,
                manifest_sha256: [
                    0x62, 0xbf, 0x9f, 0xd0, 0xa4, 0x82, 0x15, 0xc0, 0x37, 0x6d, 0xeb, 0x81, 0x77,
                    0x1b, 0xf7, 0xcc, 0x8d, 0xa7, 0x61, 0x33, 0xa5, 0xe5, 0xe8, 0x4c, 0xaa, 0x49,
                    0xee, 0x65, 0x06, 0xc4, 0x9a, 0x17,
                ],
            },
            Self::MiniMultilingualV11 => StrictCheckpointSpec {
                label: "parler/mini-multilingual-v1.1",
                arch: ARCH,
                model_name: "parler-tts-mini-multilingual-v1.1",
                model_name_alias: None,
                tensor_count: 840,
                manifest_sha256: [
                    0x8b, 0x6f, 0x25, 0x59, 0x6f, 0x94, 0x59, 0x88, 0xf4, 0x8a, 0xa2, 0x35, 0x09,
                    0xf1, 0x8b, 0x12, 0x73, 0x16, 0x21, 0x92, 0xef, 0x6b, 0xfa, 0x47, 0x01, 0x9b,
                    0x2e, 0xea, 0xcc, 0x24, 0x32, 0xe4,
                ],
            },
        }
    }
}

/// Strictly authenticated end-to-end model from one public Parler GGUF.
#[derive(Debug)]
pub struct ParlerModel {
    file: Arc<GgufFile>,
    variant: ParlerVariant,
    backend: BackendKind,
    weight_license: LicenseClass,
    prompt_embedding: GgufTensorInfo,
    text_encoder: T5Encoder,
    decoder: AudioCraftLmDecoder,
    codec: EmbeddedDac,
}

impl ParlerModel {
    /// Opens a public Parler GGUF through mmap on CPU.
    pub fn open_mapped(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_mapped_with_backend(path, BackendKind::Cpu)
    }

    /// Opens and authenticates a public Parler GGUF for an explicit backend.
    pub fn open_mapped_with_backend(path: impl AsRef<Path>, backend: BackendKind) -> Result<Self> {
        let file = vokra_mmap::open_gguf(path.as_ref()).map_err(VokraError::from)?;
        Self::from_gguf_mapped_with_backend(Arc::new(file), backend)
    }

    /// Authenticates and binds an already mmap-backed public GGUF on CPU.
    pub fn from_gguf_mapped(file: Arc<GgufFile>) -> Result<Self> {
        Self::from_gguf_mapped_with_backend(file, BackendKind::Cpu)
    }

    /// Authenticates the complete release before decoding any resident text
    /// weights or constructing the mapped autoregressive decoder.
    pub fn from_gguf_mapped_with_backend(
        file: Arc<GgufFile>,
        backend: BackendKind,
    ) -> Result<Self> {
        let variant = ParlerVariant::from_tensor_count(file.tensors().len()).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "parler: tensor count {} matches neither Mini v1 English (926) nor Mini Multilingual v1.1 (840)",
                file.tensors().len()
            ))
        })?;
        let checkpoint = StrictCheckpoint::bind(&file, variant.spec())?;
        validate_metadata(&file, variant)?;
        validate_f32_descriptors(&file, variant)?;
        if checkpoint.weight_license() != LicenseClass::Permissive {
            return Err(VokraError::ModelLoad(format!(
                "parler/{}: Apache-2.0 public checkpoint must classify as permissive, got {:?}",
                variant.variant_tag(),
                checkpoint.weight_license()
            )));
        }
        let _ = Compute::for_backend(backend, PARLER_HOT_OPS)?;

        let prompt_embedding = exact_f32_info(
            &file,
            "embed_prompts.weight",
            &[variant.prompt_vocab_size(), DECODER_CONFIG.d_model],
            variant,
        )?;
        let text_encoder = T5Encoder::from_gguf(&file, "text_encoder", FLAN_T5_LARGE_CONFIG)?
            .with_backend(backend);
        let decoder = AudioCraftLmDecoder::bind_transformers_parler(
            Arc::clone(&file),
            DECODER_CONFIG,
            backend,
            variant.fused_lm_heads(),
        )?;
        let codec = EmbeddedDac::bind(&file, variant, backend)?;

        debug_assert_eq!(checkpoint.model_name(), variant.model_name());
        debug_assert_eq!(checkpoint.tensor_count(), variant.tensor_count());
        Ok(Self {
            file,
            variant,
            backend,
            weight_license: checkpoint.weight_license(),
            prompt_embedding,
            text_encoder,
            decoder,
            codec,
        })
    }

    /// Authenticated release.
    pub const fn variant(&self) -> ParlerVariant {
        self.variant
    }

    /// Selected CPU or Metal backend.
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Fail-closed weight-license class.
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Encodes description-token IDs through the embedded FLAN-T5-large.
    /// Tokenization remains an explicit caller boundary because neither
    /// public GGUF contains a tokenizer vocabulary/model.
    pub fn encode_description(
        &self,
        token_ids: &[u32],
        attention_mask: Option<&[bool]>,
    ) -> Result<Vec<f32>> {
        self.text_encoder.encode_tokens(token_ids, attention_mask)
    }

    /// Looks up prompt-token IDs in Parler's learned direct embedding table.
    /// This is intentionally distinct from [`Self::encode_description`].
    pub fn embed_prompt_tokens(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(VokraError::InvalidArgument(
                "parler prompt requires at least one explicit token ID".to_owned(),
            ));
        }
        let vocab = self.variant.prompt_vocab_size();
        let d = DECODER_CONFIG.d_model;
        let bytes = self.file.tensor_bytes(&self.prompt_embedding);
        let row_bytes = d.checked_mul(4).ok_or_else(|| {
            VokraError::InvalidArgument("parler prompt row byte size overflow".to_owned())
        })?;
        let mut output = Vec::with_capacity(token_ids.len().saturating_mul(d));
        for (position, &token) in token_ids.iter().enumerate() {
            let token = token as usize;
            if token >= vocab {
                return Err(VokraError::InvalidArgument(format!(
                    "parler prompt token_ids[{position}]={token} is outside 0..{vocab}"
                )));
            }
            let start = token.checked_mul(row_bytes).ok_or_else(|| {
                VokraError::InvalidArgument("parler prompt row offset overflow".to_owned())
            })?;
            for chunk in bytes[start..start + row_bytes].chunks_exact(4) {
                output.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        Ok(output)
    }

    pub(super) const fn decoder(&self) -> &AudioCraftLmDecoder {
        &self.decoder
    }

    pub(super) const fn codec(&self) -> &EmbeddedDac {
        &self.codec
    }
}

fn validate_metadata(file: &GgufFile, variant: ParlerVariant) -> Result<()> {
    let label = format!("parler/{}", variant.variant_tag());
    for (key, expected) in [
        (chunks::KEY_MODEL_NAME, variant.model_name()),
        ("vokra.model.category", CATEGORY),
        (KEY_VARIANT, variant.variant_tag()),
        (chunks::KEY_PROVENANCE_MODEL_ID, variant.model_name()),
        (chunks::KEY_PROVENANCE_SOURCE, variant.upstream_hf()),
        ("vokra.provenance.upstream_hf", variant.upstream_hf()),
        (chunks::KEY_PROVENANCE_LICENSE, "apache-2.0"),
        (
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        ),
    ] {
        require_string(file, key, expected, &label)?;
    }
    for (key, expected) in [
        (KEY_PROMPT_VOCAB, variant.prompt_vocab_size()),
        (KEY_TEXT_VOCAB, FLAN_T5_LARGE_CONFIG.vocab_size),
        (KEY_TEXT_D_MODEL, FLAN_T5_LARGE_CONFIG.d_model),
        (KEY_TEXT_D_FF, FLAN_T5_LARGE_CONFIG.d_ff),
        (KEY_TEXT_LAYERS, FLAN_T5_LARGE_CONFIG.num_layers),
        (KEY_TEXT_HEADS, FLAN_T5_LARGE_CONFIG.num_heads),
        (KEY_DECODER_VOCAB, DECODER_CONFIG.vocab_size),
        (KEY_DECODER_HIDDEN, DECODER_CONFIG.d_model),
        (KEY_DECODER_FFN, DECODER_CONFIG.ffn_dim),
        (KEY_DECODER_LAYERS, DECODER_CONFIG.num_layers),
        (KEY_DECODER_HEADS, DECODER_CONFIG.n_heads),
        (KEY_DECODER_KV_HEADS, DECODER_CONFIG.n_heads),
        (KEY_DECODER_CODEBOOKS, DECODER_CONFIG.num_codebooks),
        (KEY_CODEC_SAMPLE_RATE, SAMPLE_RATE as usize),
        (KEY_CODEC_CODEBOOK_SIZE, CODEBOOK_SIZE),
    ] {
        require_u64(file, key, expected as u64, &label)?;
    }
    let prompt_cross_attention = file
        .get(KEY_PROMPT_CROSS_ATTENTION)
        .and_then(GgufMetadataValue::as_bool)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "{label}: missing/non-boolean `{KEY_PROMPT_CROSS_ATTENTION}`"
            ))
        })?;
    if prompt_cross_attention {
        return Err(VokraError::ModelLoad(format!(
            "{label}: `{KEY_PROMPT_CROSS_ATTENTION}`=true is unsupported by these pinned releases; expected false"
        )));
    }
    Ok(())
}

fn validate_f32_descriptors(file: &GgufFile, variant: ParlerVariant) -> Result<()> {
    for tensor in file.tensors() {
        if tensor.dtype != GgmlType::F32 {
            return Err(VokraError::ModelLoad(format!(
                "parler/{}: tensor `{}` is {:?}; the authenticated public mmap contract is entirely F32",
                variant.variant_tag(),
                tensor.name,
                tensor.dtype
            )));
        }
    }
    Ok(())
}

fn exact_f32_info(
    file: &GgufFile,
    name: &str,
    expected: &[usize],
    variant: ParlerVariant,
) -> Result<GgufTensorInfo> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "parler/{}: required tensor `{name}` is missing",
            variant.variant_tag()
        ))
    })?;
    let actual: Vec<usize> = info.dimensions.iter().map(|&axis| axis as usize).collect();
    if actual != expected || info.dtype != GgmlType::F32 {
        return Err(VokraError::ModelLoad(format!(
            "parler/{}: tensor `{name}` is {:?} {actual:?}, expected F32 {expected:?}",
            variant.variant_tag(),
            info.dtype
        )));
    }
    Ok(info.clone())
}

fn require_string(file: &GgufFile, key: &str, expected: &str, label: &str) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| VokraError::ModelLoad(format!("{label}: missing/non-string `{key}`")))?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "{label}: `{key}`={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn require_u64(file: &GgufFile, key: &str, expected: u64, label: &str) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(GgufMetadataValue::as_u64)
        .ok_or_else(|| VokraError::ModelLoad(format!("{label}: missing/non-integer `{key}`")))?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "{label}: `{key}`={actual}, expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_contracts_pin_distinct_public_manifests() {
        let english = ParlerVariant::MiniV1English;
        let multilingual = ParlerVariant::MiniMultilingualV11;
        assert_eq!(english.tensor_count(), 926);
        assert_eq!(multilingual.tensor_count(), 840);
        assert_ne!(
            english.tensor_manifest_sha256(),
            multilingual.tensor_manifest_sha256()
        );
        assert!(!english.fused_lm_heads());
        assert!(multilingual.fused_lm_heads());
        assert_eq!(ParlerVariant::from_tensor_count(926), Some(english));
        assert_eq!(ParlerVariant::from_tensor_count(840), Some(multilingual));
        assert_eq!(ParlerVariant::from_tensor_count(841), None);
    }

    #[test]
    fn control_tokens_are_reserved_outside_the_dac_codebook() {
        assert_eq!(PAD_EOS_TOKEN_ID as usize, CODEBOOK_SIZE);
        assert_eq!(BOS_TOKEN_ID, PAD_EOS_TOKEN_ID + 1);
        assert!((BOS_TOKEN_ID as usize) < DECODER_VOCAB_SIZE);
    }

    #[test]
    fn lm_hot_ops_cover_both_subgraphs() {
        for required in [
            HotOp::Gemm,
            HotOp::Gemv,
            HotOp::Softmax,
            HotOp::RmsNorm,
            HotOp::GeluNew,
            HotOp::LayerNorm,
            HotOp::Gelu,
        ] {
            assert!(PARLER_LM_HOT_OPS.contains(&required));
        }
        for required in [HotOp::DacRvq, HotOp::Conv1d, HotOp::SnakeActivation] {
            assert!(PARLER_HOT_OPS.contains(&required));
        }
    }
}
