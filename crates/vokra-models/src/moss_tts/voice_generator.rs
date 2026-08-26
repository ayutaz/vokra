//! Strict mapped checkpoint binding for MOSS-VoiceGenerator.
//!
//! The upstream class reuses Delay generation, but this release has a
//! Qwen3-1.7B / 16-codebook tensor contract. The public Vokra GGUF contains
//! those correct 343 tensors under a stale 8B/32-codebook header. That one
//! historical header is accepted only behind its complete manifest and is
//! surfaced as a metadata-repair requirement; arbitrary family inference is
//! never allowed.

use std::path::Path;
use std::sync::Arc;

use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::mapped_weights::MappedModel;
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

use super::delay::{DelayMappedDescriptors, DelayTopology};
use super::{ARCH, CATEGORY};

const LABEL: &str = "moss_tts/voice_generator";
const NAME: &str = super::VOICE_GENERATOR_NAME;
const LEGACY_NAME: &str = "moss-tts";
const UPSTREAM_HF: &str = "OpenMOSS-Team/MOSS-VoiceGenerator";
const UPSTREAM_REVISION: &str = "97521ec2b6f3ec5026ac1f5751f8fc302d82c2d4";
const CONFIG_SHA256: &str = "5b6ccfbf309a5844c130d09c9b5fa8b9eef55db27f1b7072695483b6f5524685";
const MODELING_SOURCE_SHA256: &str =
    "666d7320f93ce6b1c1f6ed4dba6fd4b9520a082a90fa7a17211efd83247d28a0";
const PROCESSING_SOURCE_SHA256: &str =
    "16dda5233f9f752518d07a6b780d6555945b48547fba0b4e7faf6eb2c4ed0038";
const SOURCE: &str =
    "OpenMOSS-Team/MOSS-VoiceGenerator (moss_tts_delay, Qwen3-1.7B backbone, apache-2.0)";
const LEGACY_SOURCE: &str = "OpenMOSS-Team/MOSS-VoiceGenerator";

const TENSOR_COUNT: usize = 343;
const MANIFEST_SHA256: [u8; 32] = [
    0x9a, 0x87, 0x4f, 0x01, 0x0e, 0x6e, 0xb9, 0xf4, 0x3f, 0xb7, 0x48, 0x9c, 0xc2, 0xfc, 0xd2, 0x4c,
    0xc7, 0x65, 0x5b, 0x2a, 0x99, 0x5f, 0x61, 0x01, 0x28, 0xaa, 0x8d, 0xac, 0x48, 0x41, 0xdd, 0xd5,
];

pub(super) const VOICE_TOPOLOGY: DelayTopology = DelayTopology {
    hidden_dim: 2_048,
    ffn_dim: 6_144,
    num_layers: 28,
    num_q_heads: 16,
    num_kv_heads: 8,
    head_dim: 128,
    text_vocab_size: 155_648,
    num_audio_codebooks: 16,
    audio_vocab_with_pad: 1_025,
    max_position_embeddings: 40_960,
    rms_norm_eps: 1.0e-6,
    rope_base: 1_000_000.0,
};

const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: ARCH,
    model_name: NAME,
    model_name_alias: Some(LEGACY_NAME),
    tensor_count: TENSOR_COUNT,
    manifest_sha256: MANIFEST_SHA256,
};

const MAPPED: MappedModel = MappedModel {
    name: LABEL,
    resident_entry: "MossVoiceGeneratorCheckpoint::open_mapped",
};

/// Mmap-backed proof of the exact MOSS-VoiceGenerator checkpoint contract.
#[derive(Clone)]
pub struct MossVoiceGeneratorCheckpoint {
    checkpoint: StrictCheckpoint,
    mapped: Arc<DelayMappedDescriptors>,
    requires_metadata_repair: bool,
}

impl std::fmt::Debug for MossVoiceGeneratorCheckpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MossVoiceGeneratorCheckpoint")
            .field("stamped_model_name", &self.checkpoint.model_name())
            .field("tensor_count", &self.checkpoint.tensor_count())
            .field("topology", &self.mapped.topology())
            .field("weight_license", &self.checkpoint.weight_license())
            .field("requires_metadata_repair", &self.requires_metadata_repair)
            .finish()
    }
}

impl MossVoiceGeneratorCheckpoint {
    /// Opens the GGUF through the true mmap loader and validates its header,
    /// complete name/shape manifest and every dense tensor descriptor.
    pub fn open_mapped(path: impl AsRef<Path>) -> Result<Self> {
        let file = vokra_mmap::open_gguf(path.as_ref()).map_err(VokraError::from)?;
        Self::from_gguf_mapped(Arc::new(file))
    }

    /// Strictly binds an already mmap-backed GGUF without widening payloads.
    pub fn from_gguf_mapped(file: Arc<GgufFile>) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(&file, SPEC)?;
        if checkpoint.weight_license() != LicenseClass::Permissive {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: weight license {:?}, expected permissive Apache-2.0",
                checkpoint.weight_license()
            )));
        }
        let requires_metadata_repair = match checkpoint.model_name() {
            NAME => {
                validate_corrected_metadata(&file)?;
                false
            }
            LEGACY_NAME => {
                validate_exact_legacy_metadata(&file)?;
                true
            }
            _ => unreachable!("StrictCheckpoint accepted only canonical or legacy name"),
        };
        let mapped = Arc::new(DelayMappedDescriptors::bind(file, VOICE_TOPOLOGY, MAPPED)?);
        Ok(Self {
            checkpoint,
            mapped,
            requires_metadata_repair,
        })
    }

    /// Canonical release identity, independent of the historical header.
    pub const fn model_name(&self) -> &'static str {
        NAME
    }

    /// Actual `vokra.model.name` found in the bound file.
    pub fn stamped_model_name(&self) -> &str {
        self.checkpoint.model_name()
    }

    /// Fixed official source repository.
    pub const fn upstream_hf(&self) -> &'static str {
        UPSTREAM_HF
    }

    /// Fixed source revision used by the runtime contract.
    pub const fn upstream_revision(&self) -> &'static str {
        UPSTREAM_REVISION
    }

    /// Complete release-manifest tensor count.
    pub const fn tensor_count(&self) -> usize {
        self.checkpoint.tensor_count()
    }

    /// Fail-closed stamped weight-license class.
    pub const fn weight_license(&self) -> LicenseClass {
        self.checkpoint.weight_license()
    }

    /// Whether this is the exact public GGUF with the stale 8B header.
    pub const fn requires_metadata_repair(&self) -> bool {
        self.requires_metadata_repair
    }

    pub(super) fn mapped(&self) -> &DelayMappedDescriptors {
        &self.mapped
    }
}

fn validate_corrected_metadata(file: &GgufFile) -> Result<()> {
    require_common_identity(file, SOURCE, UPSTREAM_HF)?;
    require_string(file, "vokra.moss_tts.variant", "voice_generator")?;
    require_topology(file, VOICE_TOPOLOGY)?;
    for (key, expected) in [
        ("vokra.provenance.upstream_revision", UPSTREAM_REVISION),
        ("vokra.moss_tts.config_sha256", CONFIG_SHA256),
        (
            "vokra.moss_tts.modeling_source_sha256",
            MODELING_SOURCE_SHA256,
        ),
        (
            "vokra.moss_tts.processing_source_sha256",
            PROCESSING_SOURCE_SHA256,
        ),
        ("vokra.moss_tts.llm.position_embedding_type", "rope"),
        (
            "vokra.moss_tts.audio_tokenizer_upstream_hf",
            "OpenMOSS-Team/MOSS-Audio-Tokenizer",
        ),
    ] {
        require_string(file, key, expected)?;
    }
    for (key, expected) in [
        ("vokra.moss_tts.llm.max_position_embeddings", 40_960),
        ("vokra.moss_tts.pad_token_id", 151_643),
        ("vokra.moss_tts.im_start_token_id", 151_644),
        ("vokra.moss_tts.im_end_token_id", 151_645),
        ("vokra.moss_tts.audio_start_token_id", 151_652),
        ("vokra.moss_tts.audio_end_token_id", 151_653),
        ("vokra.moss_tts.audio_user_slot_token_id", 151_654),
        ("vokra.moss_tts.audio_assistant_gen_slot_token_id", 151_656),
        (
            "vokra.moss_tts.audio_assistant_delay_slot_token_id",
            151_662,
        ),
        ("vokra.moss_tts.audio_pad_token_id", 1_024),
    ] {
        require_u32(file, key, expected)?;
    }
    Ok(())
}

fn validate_exact_legacy_metadata(file: &GgufFile) -> Result<()> {
    require_common_identity(file, LEGACY_SOURCE, "OpenMOSS-Team/MOSS-TTS")?;
    require_string(file, "vokra.moss_tts.variant", "delay")?;
    require_topology(
        file,
        DelayTopology {
            hidden_dim: 4_096,
            ffn_dim: 12_288,
            num_layers: 36,
            num_q_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            text_vocab_size: 155_648,
            num_audio_codebooks: 32,
            audio_vocab_with_pad: 1_025,
            max_position_embeddings: 40_960,
            rms_norm_eps: 1.0e-6,
            rope_base: 1_000_000.0,
        },
    )
}

fn require_common_identity(file: &GgufFile, source: &str, upstream_hf: &str) -> Result<()> {
    require_string(file, chunks::KEY_MODEL_ARCH, ARCH)?;
    require_string(file, "vokra.model.category", CATEGORY)?;
    require_string(file, chunks::KEY_PROVENANCE_MODEL_ID, NAME)?;
    require_string(file, chunks::KEY_PROVENANCE_SOURCE, source)?;
    require_string(file, chunks::KEY_PROVENANCE_LICENSE, "apache-2.0")?;
    require_string(
        file,
        chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
        LicenseClass::Permissive.as_str(),
    )?;
    require_string(file, "vokra.provenance.upstream_hf", upstream_hf)?;
    require_string(file, "vokra.moss_tts.llm.family", "qwen3")
}

fn require_topology(file: &GgufFile, topology: DelayTopology) -> Result<()> {
    for (key, expected) in [
        ("vokra.moss_tts.n_vq", topology.num_audio_codebooks as u32),
        (
            "vokra.moss_tts.audio_vocab_size",
            (topology.audio_vocab_with_pad - 1) as u32,
        ),
        ("vokra.moss_tts.sample_rate", 24_000),
        ("vokra.moss_tts.llm.hidden_dim", topology.hidden_dim as u32),
        ("vokra.moss_tts.llm.ffn_dim", topology.ffn_dim as u32),
        ("vokra.moss_tts.llm.n_layer", topology.num_layers as u32),
        ("vokra.moss_tts.llm.n_head", topology.num_q_heads as u32),
        ("vokra.moss_tts.llm.n_head_kv", topology.num_kv_heads as u32),
        ("vokra.moss_tts.llm.head_dim", topology.head_dim as u32),
        (
            "vokra.moss_tts.llm.vocab_size",
            topology.text_vocab_size as u32,
        ),
    ] {
        require_u32(file, key, expected)?;
    }
    require_f32(file, "vokra.moss_tts.llm.rope_base", topology.rope_base)?;
    require_f32(
        file,
        "vokra.moss_tts.llm.rms_norm_eps",
        topology.rms_norm_eps,
    )
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
    use crate::moss_tts::delay::tensor_contract;

    #[test]
    fn voice_manifest_contract_is_exact_and_distinct_from_delay() {
        assert_eq!(VOICE_TOPOLOGY.tensor_count(), TENSOR_COUNT);
        let contract = tensor_contract(VOICE_TOPOLOGY);
        assert_eq!(contract.len(), TENSOR_COUNT);
        let names: BTreeSet<&str> = contract.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names.len(), TENSOR_COUNT);
        assert_eq!(
            contract.first(),
            Some(&(
                "language_model.embed_tokens.weight".to_owned(),
                155_648 * 2_048
            ))
        );
        assert_eq!(
            contract.last(),
            Some(&("lm_heads.16.weight".to_owned(), 1_025 * 2_048))
        );
        assert!(names.contains("language_model.layers.27.mlp.down_proj.weight"));
        assert!(!names.contains("language_model.layers.28.input_layernorm.weight"));
    }

    #[test]
    fn fixed_source_identity_is_complete() {
        assert_eq!(UPSTREAM_REVISION.len(), 40);
        assert_eq!(CONFIG_SHA256.len(), 64);
        assert_eq!(MODELING_SOURCE_SHA256.len(), 64);
        assert_eq!(PROCESSING_SOURCE_SHA256.len(), 64);
        assert_eq!(MANIFEST_SHA256.len(), 32);
    }
}
