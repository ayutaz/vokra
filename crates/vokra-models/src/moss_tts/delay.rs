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
const HIDDEN_DIM: usize = 4_096;
const FFN_DIM: usize = 12_288;
const NUM_LAYERS: usize = 36;
const NUM_Q_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM;
const TEXT_VOCAB_SIZE: usize = 155_648;
const NUM_AUDIO_CODEBOOKS: usize = 32;
const AUDIO_VOCAB_WITH_PAD: usize = 1_025;

const MANIFEST_SHA256: [u8; 32] = [
    0x5a, 0x35, 0x78, 0xfb, 0x9e, 0x57, 0x04, 0xbf, 0x80, 0xa1, 0x27, 0x15, 0xe8, 0xc9, 0x44, 0xb2,
    0x34, 0x74, 0x57, 0xb5, 0x7c, 0x9a, 0x4d, 0x6c, 0x05, 0x5d, 0xc5, 0xb3, 0xb2, 0x2e, 0x3d, 0x51,
];

const MAPPED: MappedModel = MappedModel {
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
        let mapped = Arc::new(DelayMappedDescriptors::bind(file)?);
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
}

impl std::fmt::Debug for DelayMappedDescriptors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DelayMappedDescriptors")
            .field("tensor_count", &self.infos.len())
            .finish()
    }
}

impl DelayMappedDescriptors {
    fn bind(file: Arc<GgufFile>) -> Result<Self> {
        let contract = tensor_contract();
        debug_assert_eq!(contract.len(), TENSOR_COUNT);
        let mut infos = Vec::with_capacity(contract.len());
        for (name, elements) in contract {
            infos.push(delay_mapped_info(&file, &name, elements)?);
        }
        Ok(Self { file, infos })
    }

    pub(super) fn file(&self) -> &GgufFile {
        &self.file
    }

    pub(super) fn info(&self, index: usize) -> &GgufTensorInfo {
        &self.infos[index]
    }
}

fn delay_mapped_info(file: &GgufFile, name: &str, elements: usize) -> Result<GgufTensorInfo> {
    if let Some(info) = file.tensor_info(name)
        && !matches!(info.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
    {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: tensor `{name}` uses unsupported mapped dtype {:?}; the Base/v1.5 bounded-memory runtime accepts dense F32, F16 or BF16 only and has no silent resident/CPU fallback",
            info.dtype
        )));
    }
    mapped_info(file, name, elements, MAPPED)
}

fn tensor_contract() -> Vec<(String, usize)> {
    let mut tensors = Vec::with_capacity(TENSOR_COUNT);
    tensors.push((
        "language_model.embed_tokens.weight".to_owned(),
        TEXT_VOCAB_SIZE * HIDDEN_DIM,
    ));
    for codebook in 0..NUM_AUDIO_CODEBOOKS {
        tensors.push((
            format!("emb_ext.{codebook}.weight"),
            AUDIO_VOCAB_WITH_PAD * HIDDEN_DIM,
        ));
    }
    for layer in 0..NUM_LAYERS {
        let prefix = format!("language_model.layers.{layer}");
        tensors.extend([
            (format!("{prefix}.input_layernorm.weight"), HIDDEN_DIM),
            (
                format!("{prefix}.self_attn.q_proj.weight"),
                NUM_Q_HEADS * HEAD_DIM * HIDDEN_DIM,
            ),
            (format!("{prefix}.self_attn.q_norm.weight"), HEAD_DIM),
            (
                format!("{prefix}.self_attn.k_proj.weight"),
                KV_DIM * HIDDEN_DIM,
            ),
            (format!("{prefix}.self_attn.k_norm.weight"), HEAD_DIM),
            (
                format!("{prefix}.self_attn.v_proj.weight"),
                KV_DIM * HIDDEN_DIM,
            ),
            (
                format!("{prefix}.self_attn.o_proj.weight"),
                HIDDEN_DIM * NUM_Q_HEADS * HEAD_DIM,
            ),
            (
                format!("{prefix}.post_attention_layernorm.weight"),
                HIDDEN_DIM,
            ),
            (
                format!("{prefix}.mlp.gate_proj.weight"),
                FFN_DIM * HIDDEN_DIM,
            ),
            (format!("{prefix}.mlp.up_proj.weight"), FFN_DIM * HIDDEN_DIM),
            (
                format!("{prefix}.mlp.down_proj.weight"),
                HIDDEN_DIM * FFN_DIM,
            ),
        ]);
    }
    tensors.push(("language_model.norm.weight".to_owned(), HIDDEN_DIM));
    tensors.push(("lm_heads.0.weight".to_owned(), TEXT_VOCAB_SIZE * HIDDEN_DIM));
    for head in 1..=NUM_AUDIO_CODEBOOKS {
        tensors.push((
            format!("lm_heads.{head}.weight"),
            AUDIO_VOCAB_WITH_PAD * HIDDEN_DIM,
        ));
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
    require_f32(file, "vokra.moss_tts.llm.rope_base", 1_000_000.0)?;
    require_f32(file, "vokra.moss_tts.llm.rms_norm_eps", 1.0e-6)?;
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
        let contract = tensor_contract();
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
