//! Strict mapped checkpoint binding for the MOSS-TTS Delay releases.
//!
//! Base and v1.5 have the same Qwen3-8B tensor topology, but remain separate
//! release identities.  The public BF16 GGUF is about 17 GB, so this binder
//! deliberately retains an mmap-backed [`GgufFile`] and validates every
//! tensor descriptor without widening any payload to owned `f32`.

use std::path::Path;
use std::sync::Arc;

use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue, GgufTensorInfo, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::mapped_weights::{MappedModel, mapped_info};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

use super::{ARCH, CATEGORY};

const LABEL: &str = "moss_tts/delay";
const TENSOR_COUNT: usize = 463;
pub(super) const HIDDEN_DIM: usize = 4_096;
pub(super) const FFN_DIM: usize = 12_288;
pub(super) const NUM_LAYERS: usize = 36;
pub(super) const NUM_Q_HEADS: usize = 32;
pub(super) const NUM_KV_HEADS: usize = 8;
pub(super) const HEAD_DIM: usize = 128;
pub(super) const Q_DIM: usize = NUM_Q_HEADS * HEAD_DIM;
pub(super) const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM;
pub(super) const TEXT_VOCAB_SIZE: usize = 155_648;
pub(super) const NUM_AUDIO_CODEBOOKS: usize = 32;
pub(super) const AUDIO_VOCAB_WITH_PAD: usize = 1_025;
pub(super) const MAX_POSITION_EMBEDDINGS: usize = 40_960;
pub(super) const RMS_NORM_EPS: f32 = 1.0e-6;
pub(super) const ROPE_BASE: f32 = 1_000_000.0;

/// Shape-only contract shared by the separate Delay-class checkpoints.
///
/// Base/v1.5 and VoiceGenerator execute the same official algorithm, but a
/// manifest must select one complete topology before any tensor is bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct DelayTopology {
    pub(super) hidden_dim: usize,
    pub(super) ffn_dim: usize,
    pub(super) num_layers: usize,
    pub(super) num_q_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) text_vocab_size: usize,
    pub(super) num_audio_codebooks: usize,
    pub(super) audio_vocab_with_pad: usize,
    pub(super) audio_pad_token_id: usize,
    pub(super) zero_audio_pad_embedding: bool,
    pub(super) max_position_embeddings: usize,
    pub(super) rms_norm_eps: f32,
    pub(super) rope_base: f32,
}

impl DelayTopology {
    pub(super) const fn q_dim(self) -> usize {
        self.num_q_heads * self.head_dim
    }

    pub(super) const fn kv_dim(self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    pub(super) const fn tensor_count(self) -> usize {
        3 + 2 * self.num_audio_codebooks + 11 * self.num_layers
    }

    pub(super) const fn input_columns(self) -> usize {
        1 + self.num_audio_codebooks
    }

    pub(super) const fn accepts_audio_token(self, token: usize) -> bool {
        token < self.audio_vocab_with_pad
            || (self.zero_audio_pad_embedding && token == self.audio_pad_token_id)
    }

    pub(super) const fn audio_embedding_row(self, token: usize) -> Option<usize> {
        if self.zero_audio_pad_embedding && token == self.audio_pad_token_id {
            None
        } else {
            Some(token)
        }
    }
}

pub(super) const DELAY_TOPOLOGY: DelayTopology = DelayTopology {
    hidden_dim: HIDDEN_DIM,
    ffn_dim: FFN_DIM,
    num_layers: NUM_LAYERS,
    num_q_heads: NUM_Q_HEADS,
    num_kv_heads: NUM_KV_HEADS,
    head_dim: HEAD_DIM,
    text_vocab_size: TEXT_VOCAB_SIZE,
    num_audio_codebooks: NUM_AUDIO_CODEBOOKS,
    audio_vocab_with_pad: AUDIO_VOCAB_WITH_PAD,
    audio_pad_token_id: 1_024,
    zero_audio_pad_embedding: false,
    max_position_embeddings: MAX_POSITION_EMBEDDINGS,
    rms_norm_eps: RMS_NORM_EPS,
    rope_base: ROPE_BASE,
};

const MANIFEST_SHA256: [u8; 32] = [
    0x5a, 0x35, 0x78, 0xfb, 0x9e, 0x57, 0x04, 0xbf, 0x80, 0xa1, 0x27, 0x15, 0xe8, 0xc9, 0x44, 0xb2,
    0x34, 0x74, 0x57, 0xb5, 0x7c, 0x9a, 0x4d, 0x6c, 0x05, 0x5d, 0xc5, 0xb3, 0xb2, 0x2e, 0x3d, 0x51,
];

pub(super) const MAPPED: MappedModel = MappedModel {
    name: LABEL,
    // `delay_mapped_info` intercepts non-dense types with the MOSS-specific
    // no-fallback error before the shared helper reaches this field.
    resident_entry: "MossTtsDelayCheckpoint::open_mapped",
};

/// Authenticated public MOSS-TTS Delay release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MossTtsDelayRelease {
    /// `OpenMOSS-Team/MOSS-TTS`.
    Base,
    /// `OpenMOSS-Team/MOSS-TTS-v1.5`.
    V1_5,
}

impl MossTtsDelayRelease {
    /// Canonical Vokra model identity.
    pub const fn model_name(self) -> &'static str {
        match self {
            Self::Base => "moss-tts",
            Self::V1_5 => "moss-tts-v1.5",
        }
    }

    /// Official Hugging Face source repository.
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Base => "OpenMOSS-Team/MOSS-TTS",
            Self::V1_5 => "OpenMOSS-Team/MOSS-TTS-v1.5",
        }
    }

    /// Fixed upstream source revision used for the runtime audit.
    pub const fn upstream_revision(self) -> &'static str {
        match self {
            Self::Base => "b6b0229853ff63c68fa6aeceb380d8c016f55daf",
            Self::V1_5 => "cdd3b911b1585e3f2dbc7775ef10f9926f58850a",
        }
    }

    /// SHA-256 of the official `config.json` at [`Self::upstream_revision`].
    pub const fn config_sha256(self) -> &'static str {
        // The two official releases intentionally carry the same config.
        "214fc997d98f51ab57925a5939afc6280e76044198b664221622e70d098ed06e"
    }

    const fn source(self) -> &'static str {
        match self {
            Self::Base => "OpenMOSS-Team/MOSS-TTS (moss_tts_delay, Qwen3-8B backbone, apache-2.0)",
            Self::V1_5 => {
                "OpenMOSS-Team/MOSS-TTS-v1.5 (moss_tts_delay, Qwen3-8B backbone, apache-2.0)"
            }
        }
    }

    const fn spec(self) -> StrictCheckpointSpec {
        StrictCheckpointSpec {
            label: LABEL,
            arch: ARCH,
            model_name: self.model_name(),
            model_name_alias: None,
            tensor_count: TENSOR_COUNT,
            manifest_sha256: MANIFEST_SHA256,
        }
    }

    fn detect(file: &GgufFile) -> Result<Self> {
        let name = file
            .get(chunks::KEY_MODEL_NAME)
            .and_then(GgufMetadataValue::as_str)
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "{LABEL}: missing string metadata `{}`",
                    chunks::KEY_MODEL_NAME
                ))
            })?;
        match name {
            "moss-tts" => Ok(Self::Base),
            "moss-tts-v1.5" => Ok(Self::V1_5),
            other => Err(VokraError::ModelLoad(format!(
                "{LABEL}: unsupported `{}`={other:?}; expected \"moss-tts\" or \"moss-tts-v1.5\" (Nano, Local and VoiceGenerator are distinct contracts)",
                chunks::KEY_MODEL_NAME
            ))),
        }
    }
}

/// Mmap-backed proof that a public Base/v1.5 Delay GGUF has the exact release
/// identity, complete 463-tensor manifest and dense payload types.
///
/// Construction does not decode or widen tensor payloads. The future native
/// generation graph can therefore materialize one layer/head chunk at a time
/// without ever creating a second 17 GB resident copy.
#[derive(Clone)]
pub struct MossTtsDelayCheckpoint {
    release: MossTtsDelayRelease,
    checkpoint: StrictCheckpoint,
    mapped: Arc<DelayMappedDescriptors>,
}

impl std::fmt::Debug for MossTtsDelayCheckpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MossTtsDelayCheckpoint")
            .field("release", &self.release)
            .field("tensor_count", &self.mapped.infos.len())
            .field("weight_license", &self.checkpoint.weight_license())
            .finish()
    }
}

impl MossTtsDelayCheckpoint {
    /// Opens the GGUF through the true mmap loader and performs a header-only
    /// strict bind. Tensor data remains lazily paged by the operating system.
    pub fn open_mapped(path: impl AsRef<Path>) -> Result<Self> {
        let file = vokra_mmap::open_gguf(path.as_ref()).map_err(VokraError::from)?;
        Self::from_gguf_mapped(Arc::new(file))
    }

    /// Strictly binds an already mmap-backed GGUF.
    ///
    /// Passing a buffered [`GgufFile::open`] inside the `Arc` is valid but
    /// defeats the bounded-memory purpose; callers should use
    /// [`Self::open_mapped`] or `vokra_mmap::open_gguf`.
    pub fn from_gguf_mapped(file: Arc<GgufFile>) -> Result<Self> {
        let release = MossTtsDelayRelease::detect(&file)?;
        let checkpoint = StrictCheckpoint::bind(&file, release.spec())?;
        if checkpoint.weight_license() != LicenseClass::Permissive {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: weight license {:?}, expected permissive Apache-2.0",
                checkpoint.weight_license()
            )));
        }
        validate_metadata(&file, release)?;
        let mapped = Arc::new(DelayMappedDescriptors::bind(file, DELAY_TOPOLOGY, MAPPED)?);
        Ok(Self {
            release,
            checkpoint,
            mapped,
        })
    }

    /// Authenticated Base/v1.5 identity.
    pub const fn release(&self) -> MossTtsDelayRelease {
        self.release
    }

    /// Canonical stamped model name.
    pub fn model_name(&self) -> &str {
        self.checkpoint.model_name()
    }

    /// Fail-closed stamped weight-license class.
    pub const fn weight_license(&self) -> LicenseClass {
        self.checkpoint.weight_license()
    }

    /// Complete release-manifest tensor count.
    pub const fn tensor_count(&self) -> usize {
        self.checkpoint.tensor_count()
    }

    pub(super) fn mapped(&self) -> &DelayMappedDescriptors {
        &self.mapped
    }
}

pub(super) struct DelayMappedDescriptors {
    file: Arc<GgufFile>,
    infos: Vec<GgufTensorInfo>,
    topology: DelayTopology,
    mapped: MappedModel,
}

impl std::fmt::Debug for DelayMappedDescriptors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DelayMappedDescriptors")
            .field("tensor_count", &self.infos.len())
            .field("topology", &self.topology)
            .finish()
    }
}

impl DelayMappedDescriptors {
    pub(super) fn bind(
        file: Arc<GgufFile>,
        topology: DelayTopology,
        mapped: MappedModel,
    ) -> Result<Self> {
        Self::bind_with_layout(file, topology, mapped, QwenTensorLayout::Delay)
    }

    pub(super) fn bind_with_layout(
        file: Arc<GgufFile>,
        topology: DelayTopology,
        mapped: MappedModel,
        layout: QwenTensorLayout,
    ) -> Result<Self> {
        let contract = tensor_contract_with_layout(topology, layout);
        debug_assert_eq!(contract.len(), topology.tensor_count());
        let mut infos = Vec::with_capacity(contract.len());
        for (name, elements) in contract {
            infos.push(delay_mapped_info(&file, &name, elements, mapped)?);
        }
        Ok(Self {
            file,
            infos,
            topology,
            mapped,
        })
    }

    pub(super) fn file(&self) -> &GgufFile {
        &self.file
    }

    pub(super) fn info(&self, index: usize) -> &GgufTensorInfo {
        &self.infos[index]
    }

    pub(super) const fn topology(&self) -> DelayTopology {
        self.topology
    }

    pub(super) const fn mapped_model(&self) -> MappedModel {
        self.mapped
    }

    pub(super) fn text_embedding(&self) -> &GgufTensorInfo {
        self.info(0)
    }

    pub(super) fn audio_embedding(&self, codebook: usize) -> &GgufTensorInfo {
        debug_assert!(codebook < self.topology.num_audio_codebooks);
        self.info(1 + codebook)
    }

    pub(super) fn layer(&self, layer: usize) -> DelayLayerDescriptors<'_> {
        debug_assert!(layer < self.topology.num_layers);
        const LAYER_WIDTH: usize = 11;
        let start = 1 + self.topology.num_audio_codebooks + layer * LAYER_WIDTH;
        DelayLayerDescriptors {
            input_norm: self.info(start),
            q: self.info(start + 1),
            q_norm: self.info(start + 2),
            k: self.info(start + 3),
            k_norm: self.info(start + 4),
            v: self.info(start + 5),
            o: self.info(start + 6),
            ffn_norm: self.info(start + 7),
            gate: self.info(start + 8),
            up: self.info(start + 9),
            down: self.info(start + 10),
        }
    }

    pub(super) fn final_norm(&self) -> &GgufTensorInfo {
        self.info(1 + self.topology.num_audio_codebooks + self.topology.num_layers * 11)
    }

    pub(super) fn head(&self, head: usize) -> &GgufTensorInfo {
        debug_assert!(head <= self.topology.num_audio_codebooks);
        self.info(1 + self.topology.num_audio_codebooks + self.topology.num_layers * 11 + 1 + head)
    }
}

pub(super) struct DelayLayerDescriptors<'a> {
    pub(super) input_norm: &'a GgufTensorInfo,
    pub(super) q: &'a GgufTensorInfo,
    pub(super) q_norm: &'a GgufTensorInfo,
    pub(super) k: &'a GgufTensorInfo,
    pub(super) k_norm: &'a GgufTensorInfo,
    pub(super) v: &'a GgufTensorInfo,
    pub(super) o: &'a GgufTensorInfo,
    pub(super) ffn_norm: &'a GgufTensorInfo,
    pub(super) gate: &'a GgufTensorInfo,
    pub(super) up: &'a GgufTensorInfo,
    pub(super) down: &'a GgufTensorInfo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum QwenTensorLayout {
    Delay,
    Local,
}

fn delay_mapped_info(
    file: &GgufFile,
    name: &str,
    elements: usize,
    mapped: MappedModel,
) -> Result<GgufTensorInfo> {
    if let Some(info) = file.tensor_info(name)
        && !matches!(info.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
    {
        return Err(VokraError::ModelLoad(format!(
            "{}: tensor `{name}` uses unsupported mapped dtype {:?}; the bounded-memory runtime accepts dense F32, F16 or BF16 only and has no silent resident/CPU fallback",
            mapped.name, info.dtype
        )));
    }
    mapped_info(file, name, elements, mapped)
}

pub(super) fn tensor_contract(topology: DelayTopology) -> Vec<(String, usize)> {
    tensor_contract_with_layout(topology, QwenTensorLayout::Delay)
}

pub(super) fn tensor_contract_with_layout(
    topology: DelayTopology,
    layout: QwenTensorLayout,
) -> Vec<(String, usize)> {
    let q_dim = topology.q_dim();
    let kv_dim = topology.kv_dim();
    let mut tensors = Vec::with_capacity(topology.tensor_count());
    let text_embedding = match layout {
        QwenTensorLayout::Delay => "language_model.embed_tokens.weight",
        QwenTensorLayout::Local => "transformer.embed_tokens.weight",
    };
    tensors.push((
        text_embedding.to_owned(),
        topology.text_vocab_size * topology.hidden_dim,
    ));
    for codebook in 0..topology.num_audio_codebooks {
        let name = match layout {
            QwenTensorLayout::Delay => format!("emb_ext.{codebook}.weight"),
            QwenTensorLayout::Local => format!("audio_embeddings.{codebook}.weight"),
        };
        tensors.push((name, topology.audio_vocab_with_pad * topology.hidden_dim));
    }
    for layer in 0..topology.num_layers {
        let prefix = match layout {
            QwenTensorLayout::Delay => format!("language_model.layers.{layer}"),
            QwenTensorLayout::Local => format!("transformer.layers.{layer}"),
        };
        tensors.extend([
            (
                format!("{prefix}.input_layernorm.weight"),
                topology.hidden_dim,
            ),
            (
                format!("{prefix}.self_attn.q_proj.weight"),
                q_dim * topology.hidden_dim,
            ),
            (
                format!("{prefix}.self_attn.q_norm.weight"),
                topology.head_dim,
            ),
            (
                format!("{prefix}.self_attn.k_proj.weight"),
                kv_dim * topology.hidden_dim,
            ),
            (
                format!("{prefix}.self_attn.k_norm.weight"),
                topology.head_dim,
            ),
            (
                format!("{prefix}.self_attn.v_proj.weight"),
                kv_dim * topology.hidden_dim,
            ),
            (
                format!("{prefix}.self_attn.o_proj.weight"),
                topology.hidden_dim * q_dim,
            ),
            (
                format!("{prefix}.post_attention_layernorm.weight"),
                topology.hidden_dim,
            ),
            (
                format!("{prefix}.mlp.gate_proj.weight"),
                topology.ffn_dim * topology.hidden_dim,
            ),
            (
                format!("{prefix}.mlp.up_proj.weight"),
                topology.ffn_dim * topology.hidden_dim,
            ),
            (
                format!("{prefix}.mlp.down_proj.weight"),
                topology.hidden_dim * topology.ffn_dim,
            ),
        ]);
    }
    let final_norm = match layout {
        QwenTensorLayout::Delay => "language_model.norm.weight",
        QwenTensorLayout::Local => "transformer.norm.weight",
    };
    tensors.push((final_norm.to_owned(), topology.hidden_dim));
    let text_head = match layout {
        QwenTensorLayout::Delay => "lm_heads.0.weight".to_owned(),
        QwenTensorLayout::Local => "text_lm_head.weight".to_owned(),
    };
    tensors.push((text_head, topology.text_vocab_size * topology.hidden_dim));
    for codebook in 0..topology.num_audio_codebooks {
        let name = match layout {
            QwenTensorLayout::Delay => format!("lm_heads.{}.weight", codebook + 1),
            QwenTensorLayout::Local => format!("audio_lm_heads.{codebook}.weight"),
        };
        tensors.push((name, topology.audio_vocab_with_pad * topology.hidden_dim));
    }
    tensors
}

fn validate_metadata(file: &GgufFile, release: MossTtsDelayRelease) -> Result<()> {
    require_string(file, chunks::KEY_MODEL_ARCH, ARCH)?;
    require_string(file, "vokra.model.category", CATEGORY)?;
    require_string(file, chunks::KEY_PROVENANCE_MODEL_ID, release.model_name())?;
    require_string(file, chunks::KEY_PROVENANCE_SOURCE, release.source())?;
    require_string(file, chunks::KEY_PROVENANCE_LICENSE, "apache-2.0")?;
    require_string(
        file,
        chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
        LicenseClass::Permissive.as_str(),
    )?;
    require_string(file, "vokra.provenance.upstream_hf", release.upstream_hf())?;
    require_string(file, "vokra.moss_tts.variant", "delay")?;
    require_string(file, "vokra.moss_tts.llm.family", "qwen3")?;
    for (key, expected) in [
        ("vokra.moss_tts.n_vq", NUM_AUDIO_CODEBOOKS as u32),
        ("vokra.moss_tts.audio_vocab_size", 1_024),
        ("vokra.moss_tts.sample_rate", 24_000),
        ("vokra.moss_tts.llm.hidden_dim", HIDDEN_DIM as u32),
        ("vokra.moss_tts.llm.ffn_dim", FFN_DIM as u32),
        ("vokra.moss_tts.llm.n_layer", NUM_LAYERS as u32),
        ("vokra.moss_tts.llm.n_head", NUM_Q_HEADS as u32),
        ("vokra.moss_tts.llm.n_head_kv", NUM_KV_HEADS as u32),
        ("vokra.moss_tts.llm.head_dim", HEAD_DIM as u32),
        ("vokra.moss_tts.llm.vocab_size", TEXT_VOCAB_SIZE as u32),
    ] {
        require_u32(file, key, expected)?;
    }
    require_f32(file, "vokra.moss_tts.llm.rope_base", ROPE_BASE)?;
    require_f32(file, "vokra.moss_tts.llm.rms_norm_eps", RMS_NORM_EPS)?;
    Ok(())
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn release_identities_are_distinct_but_share_the_audited_config() {
        assert_ne!(
            MossTtsDelayRelease::Base.model_name(),
            MossTtsDelayRelease::V1_5.model_name()
        );
        assert_ne!(
            MossTtsDelayRelease::Base.upstream_hf(),
            MossTtsDelayRelease::V1_5.upstream_hf()
        );
        assert_ne!(
            MossTtsDelayRelease::Base.upstream_revision(),
            MossTtsDelayRelease::V1_5.upstream_revision()
        );
        assert_eq!(
            MossTtsDelayRelease::Base.config_sha256(),
            MossTtsDelayRelease::V1_5.config_sha256()
        );
    }

    #[test]
    fn mapped_contract_covers_every_manifest_tensor_once() {
        let contract = tensor_contract(DELAY_TOPOLOGY);
        assert_eq!(contract.len(), TENSOR_COUNT);
        let names: BTreeSet<&str> = contract.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names.len(), TENSOR_COUNT);
        assert_eq!(
            contract.first(),
            Some(&(
                "language_model.embed_tokens.weight".to_owned(),
                TEXT_VOCAB_SIZE * HIDDEN_DIM
            ))
        );
        assert_eq!(
            contract.last(),
            Some(&(
                "lm_heads.32.weight".to_owned(),
                AUDIO_VOCAB_WITH_PAD * HIDDEN_DIM
            ))
        );
        assert!(names.contains("language_model.layers.0.self_attn.q_norm.weight"));
        assert!(names.contains("language_model.layers.35.mlp.down_proj.weight"));
        assert!(names.contains("language_model.norm.weight"));
    }
}
