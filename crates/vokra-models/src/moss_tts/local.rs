//! Strict mapped checkpoint binding for MOSS-TTS Local Transformer v1.5.
//!
//! The public GGUF is about 8.7 GB and combines a Qwen3 global decoder with
//! one GPT-2-style local decoder. This module authenticates the complete
//! 438-tensor release and keeps every payload mmap-backed; construction never
//! widens the checkpoint into a second resident copy.

use std::path::Path;
use std::sync::Arc;

use vokra_core::gguf::{GgufFile, GgufMetadataValue, GgufTensorInfo, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::mapped_weights::{MappedModel, mapped_info};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

use super::delay::{DelayMappedDescriptors, DelayTopology, QwenTensorLayout};
use super::{ARCH, CATEGORY};

pub(super) const LABEL: &str = "moss_tts/local";
pub const NAME: &str = "moss-tts-local-transformer-v1.5";
pub const UPSTREAM_HF: &str = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5";
pub const UPSTREAM_REVISION: &str = "be7766a6735b98bd793f7c79fb720b4d0f5d13b8";
pub const AUDIO_TOKENIZER_UPSTREAM_HF: &str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2";

pub(super) const LOCAL_TOPOLOGY: DelayTopology = DelayTopology {
    hidden_dim: 2_560,
    ffn_dim: 9_728,
    num_layers: 36,
    num_q_heads: 32,
    num_kv_heads: 8,
    head_dim: 128,
    text_vocab_size: 151_936,
    num_audio_codebooks: 12,
    // Local embeddings and heads omit the pad row. Token 1024 contributes an
    // exact zero vector at input and is never a learned output class.
    audio_vocab_with_pad: 1_024,
    audio_pad_token_id: 1_024,
    zero_audio_pad_embedding: true,
    max_position_embeddings: 32_768,
    rms_norm_eps: 1.0e-6,
    rope_base: 1_000_000.0,
};

pub(super) const LOCAL_NUM_HEADS: usize = 32;
pub(super) const LOCAL_HEAD_DIM: usize = 80;
pub(super) const LOCAL_FFN_DIM: usize = 9_728;
pub(super) const LOCAL_LAYER_NORM_EPS: f32 = 1.0e-6;
pub(super) const LOCAL_ROPE_BASE: f32 = 1_000_000.0;
pub(super) const LOCAL_CACHE_CAPACITY: usize = 13;

const TENSOR_COUNT: usize = 438;
const MANIFEST_SHA256: [u8; 32] = [
    0x9b, 0x4a, 0xa2, 0xec, 0xca, 0xe9, 0x17, 0x46, 0x47, 0xaa, 0xb3, 0x81, 0xeb, 0x2e, 0x30, 0x74,
    0x1f, 0x31, 0x22, 0x7b, 0xc6, 0x04, 0x46, 0x1b, 0xa9, 0x9c, 0xcb, 0x30, 0xdf, 0x71, 0xc3, 0x5d,
];
const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: ARCH,
    model_name: NAME,
    model_name_alias: None,
    tensor_count: TENSOR_COUNT,
    manifest_sha256: MANIFEST_SHA256,
};
pub(super) const MAPPED: MappedModel = MappedModel {
    name: LABEL,
    resident_entry: "MossTtsLocalCheckpoint::open_mapped",
};

const SOURCE: &str = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 (moss_tts_local, Qwen3-2.5B backbone, apache-2.0)";
const CONFIG_SHA256: &str = "826f81f163b1b557ad13f83c4f35008f4fee5a6cb6311b4316ff3dbb25149411";
const CONFIGURATION_SOURCE_SHA256: &str =
    "ab6debcb92032cb9dc91ae80aed77dbadd2e59848208baef2b062bd6def3f3be";
const MODELING_SOURCE_SHA256: &str =
    "b0a66211943ae580b087f3e71495fea2f455701a4f6c29b6d3562218f7668c5f";
const PROCESSING_SOURCE_SHA256: &str =
    "3fc5616b1ec3408162b7d859a7696725a40525313b20f9b31a06ee55c93bd7ad";
const GPT2_DECODER_SOURCE_SHA256: &str =
    "f2e877104669f1e6c7cd34680f0da1a8a159e032123ee56b660b63929b6c8989";
const QWEN3_DECODER_SOURCE_SHA256: &str =
    "100163bd7ecf31a59bafacc0b032ace9339edc992a3eb4cc80662502e04e46f0";
const PROCESSOR_CONFIG_SHA256: &str =
    "db574bfebad009e05193196a63a4eeecd353eeca177ccfff28b9379d595d88b7";

const CORRECTED_KEYS: &[&str] = &[
    "vokra.provenance.upstream_revision",
    "vokra.moss_tts.config_sha256",
    "vokra.moss_tts.configuration_source_sha256",
    "vokra.moss_tts.modeling_source_sha256",
    "vokra.moss_tts.processing_source_sha256",
    "vokra.moss_tts.qwen3_decoder_source_sha256",
    "vokra.moss_tts.gpt2_decoder_source_sha256",
    "vokra.moss_tts.processor_config_sha256",
    "vokra.moss_tts.llm.position_embedding_type",
    "vokra.moss_tts.llm.max_position_embeddings",
    "vokra.moss_tts.local_transformer_layers",
    "vokra.moss_tts.local_transformer.hidden_dim",
    "vokra.moss_tts.local_transformer.ffn_dim",
    "vokra.moss_tts.local_transformer.n_head",
    "vokra.moss_tts.local_transformer.head_dim",
    "vokra.moss_tts.local_transformer.position_embedding_type",
    "vokra.moss_tts.local_transformer.rope_base",
    "vokra.moss_tts.local_transformer.layer_norm_eps",
    "vokra.moss_tts.local_transformer.activation",
    "vokra.moss_tts.local_text_head_mode",
    "vokra.moss_tts.local_transformer.use_static_kv_cache",
    "vokra.moss_tts.pad_token_id",
    "vokra.moss_tts.im_start_token_id",
    "vokra.moss_tts.im_end_token_id",
    "vokra.moss_tts.audio_start_token_id",
    "vokra.moss_tts.audio_end_token_id",
    "vokra.moss_tts.audio_user_slot_token_id",
    "vokra.moss_tts.audio_assistant_slot_token_id",
    "vokra.moss_tts.audio_assistant_gen_slot_token_id",
    "vokra.moss_tts.audio_pad_token_id",
    "vokra.moss_tts.audio_tokenizer_upstream_hf",
];

/// Mmap-backed proof of the exact public Local Transformer release.
#[derive(Clone)]
pub struct MossTtsLocalCheckpoint {
    checkpoint: StrictCheckpoint,
    requires_metadata_repair: bool,
    mapped: Arc<LocalMappedDescriptors>,
}

impl std::fmt::Debug for MossTtsLocalCheckpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MossTtsLocalCheckpoint")
            .field("tensor_count", &self.checkpoint.tensor_count())
            .field("weight_license", &self.checkpoint.weight_license())
            .field("requires_metadata_repair", &self.requires_metadata_repair)
            .finish()
    }
}

impl MossTtsLocalCheckpoint {
    /// Opens through the true mmap loader and validates the complete header.
    pub fn open_mapped(path: impl AsRef<Path>) -> Result<Self> {
        let file = vokra_mmap::open_gguf(path.as_ref()).map_err(VokraError::from)?;
        Self::from_gguf_mapped(Arc::new(file))
    }

    /// Strictly binds an already mmap-backed GGUF.
    pub fn from_gguf_mapped(file: Arc<GgufFile>) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(&file, SPEC)?;
        if checkpoint.weight_license() != LicenseClass::Permissive {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: weight license {:?}, expected permissive Apache-2.0",
                checkpoint.weight_license()
            )));
        }
        let requires_metadata_repair = validate_metadata(&file)?;
        let mapped = Arc::new(LocalMappedDescriptors::bind(file)?);
        Ok(Self {
            checkpoint,
            requires_metadata_repair,
            mapped,
        })
    }

    /// Returns the checkpoint weight-license class.
    pub const fn weight_license(&self) -> LicenseClass {
        self.checkpoint.weight_license()
    }

    /// Returns the authenticated checkpoint tensor count.
    pub const fn tensor_count(&self) -> usize {
        self.checkpoint.tensor_count()
    }

    /// Reports whether legacy public metadata was repaired during binding.
    pub const fn requires_metadata_repair(&self) -> bool {
        self.requires_metadata_repair
    }

    pub(super) fn mapped(&self) -> &LocalMappedDescriptors {
        &self.mapped
    }
}

pub(super) struct LocalMappedDescriptors {
    qwen: DelayMappedDescriptors,
    extras: Vec<GgufTensorInfo>,
}

impl std::fmt::Debug for LocalMappedDescriptors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalMappedDescriptors")
            .field("qwen_tensor_count", &LOCAL_TOPOLOGY.tensor_count())
            .field("local_tensor_count", &self.extras.len())
            .finish()
    }
}

impl LocalMappedDescriptors {
    fn bind(file: Arc<GgufFile>) -> Result<Self> {
        let qwen = DelayMappedDescriptors::bind_with_layout(
            Arc::clone(&file),
            LOCAL_TOPOLOGY,
            MAPPED,
            QwenTensorLayout::Local,
        )?;
        let contract = local_tensor_contract();
        let mut extras = Vec::with_capacity(contract.len());
        for (name, elements) in contract {
            extras.push(mapped_info(&file, name, elements, MAPPED)?);
        }
        debug_assert_eq!(LOCAL_TOPOLOGY.tensor_count() + extras.len(), TENSOR_COUNT);
        Ok(Self { qwen, extras })
    }

    pub(super) const fn qwen(&self) -> &DelayMappedDescriptors {
        &self.qwen
    }

    pub(super) fn local_text_head(&self) -> &GgufTensorInfo {
        &self.extras[0]
    }

    pub(super) fn local_block(&self) -> LocalBlockDescriptors<'_> {
        LocalBlockDescriptors {
            qkv_bias: &self.extras[1],
            qkv_weight: &self.extras[2],
            projection_bias: &self.extras[3],
            projection_weight: &self.extras[4],
            norm1_bias: &self.extras[5],
            norm1_weight: &self.extras[6],
            norm2_bias: &self.extras[7],
            norm2_weight: &self.extras[8],
            ffn_in_bias: &self.extras[9],
            ffn_in_weight: &self.extras[10],
            ffn_out_bias: &self.extras[11],
            ffn_out_weight: &self.extras[12],
            final_norm_bias: &self.extras[13],
            final_norm_weight: &self.extras[14],
        }
    }
}

pub(super) struct LocalBlockDescriptors<'a> {
    pub(super) qkv_bias: &'a GgufTensorInfo,
    pub(super) qkv_weight: &'a GgufTensorInfo,
    pub(super) projection_bias: &'a GgufTensorInfo,
    pub(super) projection_weight: &'a GgufTensorInfo,
    pub(super) norm1_bias: &'a GgufTensorInfo,
    pub(super) norm1_weight: &'a GgufTensorInfo,
    pub(super) norm2_bias: &'a GgufTensorInfo,
    pub(super) norm2_weight: &'a GgufTensorInfo,
    pub(super) ffn_in_bias: &'a GgufTensorInfo,
    pub(super) ffn_in_weight: &'a GgufTensorInfo,
    pub(super) ffn_out_bias: &'a GgufTensorInfo,
    pub(super) ffn_out_weight: &'a GgufTensorInfo,
    pub(super) final_norm_bias: &'a GgufTensorInfo,
    pub(super) final_norm_weight: &'a GgufTensorInfo,
}

fn local_tensor_contract() -> Vec<(&'static str, usize)> {
    let hidden = LOCAL_TOPOLOGY.hidden_dim;
    vec![
        ("local_text_lm_head.weight", 2 * hidden),
        ("local_transformer.h.0.attn.c_attn.bias", 3 * hidden),
        (
            "local_transformer.h.0.attn.c_attn.weight",
            3 * hidden * hidden,
        ),
        ("local_transformer.h.0.attn.c_proj.bias", hidden),
        ("local_transformer.h.0.attn.c_proj.weight", hidden * hidden),
        ("local_transformer.h.0.ln_1.bias", hidden),
        ("local_transformer.h.0.ln_1.weight", hidden),
        ("local_transformer.h.0.ln_2.bias", hidden),
        ("local_transformer.h.0.ln_2.weight", hidden),
        ("local_transformer.h.0.mlp.fc_in.bias", LOCAL_FFN_DIM),
        (
            "local_transformer.h.0.mlp.fc_in.weight",
            LOCAL_FFN_DIM * hidden,
        ),
        ("local_transformer.h.0.mlp.fc_out.bias", hidden),
        (
            "local_transformer.h.0.mlp.fc_out.weight",
            hidden * LOCAL_FFN_DIM,
        ),
        ("local_transformer.ln_f.bias", hidden),
        ("local_transformer.ln_f.weight", hidden),
    ]
}

fn validate_metadata(file: &GgufFile) -> Result<bool> {
    for (key, expected) in [
        (chunks::KEY_MODEL_ARCH, ARCH),
        ("vokra.model.category", CATEGORY),
        (chunks::KEY_PROVENANCE_MODEL_ID, NAME),
        (chunks::KEY_PROVENANCE_SOURCE, SOURCE),
        (chunks::KEY_PROVENANCE_LICENSE, "apache-2.0"),
        (
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        ),
        ("vokra.provenance.upstream_hf", UPSTREAM_HF),
        ("vokra.moss_tts.variant", "local"),
        ("vokra.moss_tts.llm.family", "qwen3"),
    ] {
        require_string(file, key, expected)?;
    }
    for (key, expected) in [
        ("vokra.moss_tts.n_vq", 12),
        ("vokra.moss_tts.audio_vocab_size", 1_024),
        ("vokra.moss_tts.sample_rate", 48_000),
        ("vokra.moss_tts.llm.hidden_dim", 2_560),
        ("vokra.moss_tts.llm.ffn_dim", 9_728),
        ("vokra.moss_tts.llm.n_layer", 36),
        ("vokra.moss_tts.llm.n_head", 32),
        ("vokra.moss_tts.llm.n_head_kv", 8),
        ("vokra.moss_tts.llm.head_dim", 128),
        ("vokra.moss_tts.llm.vocab_size", 151_936),
    ] {
        require_u32(file, key, expected)?;
    }
    require_f32(file, "vokra.moss_tts.llm.rope_base", 1_000_000.0)?;
    require_f32(file, "vokra.moss_tts.llm.rms_norm_eps", 1.0e-6)?;

    let present = CORRECTED_KEYS
        .iter()
        .filter(|key| file.get(key).is_some())
        .count();
    if present == 0 {
        return Ok(true);
    }
    if present != CORRECTED_KEYS.len() {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: corrected metadata is partial ({present}/{} keys); accept either the exact historical header or one complete fixed-revision contract",
            CORRECTED_KEYS.len()
        )));
    }

    for (key, expected) in [
        ("vokra.provenance.upstream_revision", UPSTREAM_REVISION),
        ("vokra.moss_tts.config_sha256", CONFIG_SHA256),
        (
            "vokra.moss_tts.configuration_source_sha256",
            CONFIGURATION_SOURCE_SHA256,
        ),
        (
            "vokra.moss_tts.modeling_source_sha256",
            MODELING_SOURCE_SHA256,
        ),
        (
            "vokra.moss_tts.processing_source_sha256",
            PROCESSING_SOURCE_SHA256,
        ),
        (
            "vokra.moss_tts.qwen3_decoder_source_sha256",
            QWEN3_DECODER_SOURCE_SHA256,
        ),
        (
            "vokra.moss_tts.gpt2_decoder_source_sha256",
            GPT2_DECODER_SOURCE_SHA256,
        ),
        (
            "vokra.moss_tts.processor_config_sha256",
            PROCESSOR_CONFIG_SHA256,
        ),
        ("vokra.moss_tts.llm.position_embedding_type", "rope"),
        (
            "vokra.moss_tts.local_transformer.position_embedding_type",
            "rope",
        ),
        ("vokra.moss_tts.local_transformer.activation", "silu"),
        ("vokra.moss_tts.local_text_head_mode", "binary"),
        (
            "vokra.moss_tts.audio_tokenizer_upstream_hf",
            AUDIO_TOKENIZER_UPSTREAM_HF,
        ),
    ] {
        require_string(file, key, expected)?;
    }
    for (key, expected) in [
        ("vokra.moss_tts.llm.max_position_embeddings", 32_768),
        ("vokra.moss_tts.local_transformer_layers", 1),
        ("vokra.moss_tts.local_transformer.hidden_dim", 2_560),
        ("vokra.moss_tts.local_transformer.ffn_dim", 9_728),
        ("vokra.moss_tts.local_transformer.n_head", 32),
        ("vokra.moss_tts.local_transformer.head_dim", 80),
        ("vokra.moss_tts.pad_token_id", 151_643),
        ("vokra.moss_tts.im_start_token_id", 151_644),
        ("vokra.moss_tts.im_end_token_id", 151_645),
        ("vokra.moss_tts.audio_start_token_id", 151_669),
        ("vokra.moss_tts.audio_end_token_id", 151_670),
        ("vokra.moss_tts.audio_user_slot_token_id", 151_654),
        ("vokra.moss_tts.audio_assistant_slot_token_id", 151_656),
        ("vokra.moss_tts.audio_assistant_gen_slot_token_id", 151_656),
        ("vokra.moss_tts.audio_pad_token_id", 1_024),
    ] {
        require_u32(file, key, expected)?;
    }
    require_f32(
        file,
        "vokra.moss_tts.local_transformer.rope_base",
        LOCAL_ROPE_BASE,
    )?;
    require_f32(
        file,
        "vokra.moss_tts.local_transformer.layer_norm_eps",
        LOCAL_LAYER_NORM_EPS,
    )?;
    require_bool(
        file,
        "vokra.moss_tts.local_transformer.use_static_kv_cache",
        true,
    )?;
    Ok(false)
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_str);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: metadata `{key}`={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn require_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    let actual = match file.get(key) {
        Some(GgufMetadataValue::U32(value)) => Some(*value),
        _ => None,
    };
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: metadata `{key}`={actual:?}, expected UINT32 {expected}"
        )));
    }
    Ok(())
}

fn require_f32(file: &GgufFile, key: &str, expected: f32) -> Result<()> {
    let actual = match file.get(key) {
        Some(GgufMetadataValue::F32(value)) => Some(*value),
        _ => None,
    };
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: metadata `{key}`={actual:?}, expected FLOAT32 {expected}"
        )));
    }
    Ok(())
}

fn require_bool(file: &GgufFile, key: &str, expected: bool) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_bool);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: metadata `{key}`={actual:?}, expected BOOL {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn local_manifest_contract_covers_every_tensor_once() {
        let mut contract = tensor_contract_with_layout(LOCAL_TOPOLOGY, QwenTensorLayout::Local);
        contract.extend(
            local_tensor_contract()
                .into_iter()
                .map(|(name, elements)| (name.to_owned(), elements)),
        );
        assert_eq!(contract.len(), TENSOR_COUNT);
        let names = contract
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(names.len(), TENSOR_COUNT);
        assert!(names.contains("transformer.layers.35.self_attn.q_proj.weight"));
        assert!(names.contains("local_transformer.h.0.attn.c_attn.weight"));
        assert!(names.contains("local_text_lm_head.weight"));
        assert!(!names.contains("language_model.embed_tokens.weight"));
    }

    #[test]
    fn local_pad_is_explicit_zero_not_an_embedding_row() {
        assert_eq!(LOCAL_TOPOLOGY.audio_vocab_with_pad, 1_024);
        assert_eq!(LOCAL_TOPOLOGY.audio_pad_token_id, 1_024);
        assert!(LOCAL_TOPOLOGY.accepts_audio_token(1_024));
        assert_eq!(LOCAL_TOPOLOGY.audio_embedding_row(1_024), None);
        assert!(!LOCAL_TOPOLOGY.accepts_audio_token(1_025));
    }

    #[test]
    fn fixed_source_identity_and_local_axes_are_complete() {
        assert_eq!(UPSTREAM_REVISION.len(), 40);
        for digest in [
            CONFIG_SHA256,
            CONFIGURATION_SOURCE_SHA256,
            MODELING_SOURCE_SHA256,
            PROCESSING_SOURCE_SHA256,
            GPT2_DECODER_SOURCE_SHA256,
            QWEN3_DECODER_SOURCE_SHA256,
            PROCESSOR_CONFIG_SHA256,
        ] {
            assert_eq!(digest.len(), 64);
            assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
        }
        assert_eq!(LOCAL_TOPOLOGY.q_dim(), 4_096);
        assert_eq!(LOCAL_TOPOLOGY.kv_dim(), 1_024);
        assert_eq!(LOCAL_NUM_HEADS * LOCAL_HEAD_DIM, LOCAL_TOPOLOGY.hidden_dim);
        assert_eq!(LOCAL_CACHE_CAPACITY, LOCAL_TOPOLOGY.num_audio_codebooks + 1);
    }
}
