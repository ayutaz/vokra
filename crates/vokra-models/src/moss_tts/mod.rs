//! Native OpenMOSS MOSS-TTS family runtime.
//!
//! The first public Vokra GGUF predates the corrected RoPE/provenance
//! metadata. It is accepted only behind the exact 194-tensor name/shape
//! manifest and is surfaced through [`MossTtsNano::requires_metadata_repair`].
//! Base/v1.5 use a separate strict 463-tensor mapped Delay contract; Local and
//! VoiceGenerator are never inferred from the shared arch tag.

mod delay;
mod delay_transformer;
mod generation;
mod transformer;
mod weights;

use std::path::Path;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::moss_audio_tokenizer::{
    MossAudioTokenizer, MossAudioTokenizerVariant, MossDecodedAudio,
};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

pub use self::delay::{MossTtsDelayCheckpoint, MossTtsDelayRelease};
pub use self::delay_transformer::{
    MOSS_TTS_DELAY_HOT_OPS, MossTtsDelay, MossTtsDelayGeneration, MossTtsDelayGenerationOptions,
    MossTtsDelayLogits,
};
pub use self::generation::MossTtsGeneratedCodes;
use self::weights::NanoWeights;

/// GGUF architecture shared by the separately authenticated MOSS-TTS family.
pub const ARCH: &str = "moss_tts";
/// Canonical public Nano identity.
pub const NAME: &str = "moss-tts-nano-100m";
/// Model-zoo category.
pub const CATEGORY: &str = "tts";
/// Official upstream repository.
pub const UPSTREAM_HF: &str = "OpenMOSS-Team/MOSS-TTS-Nano-100M";
/// Exact upstream source/weight revision used by the corrected converter.
pub const UPSTREAM_REVISION: &str = "44502f80dbf9743528fa921cc544d662c685ebec";
/// Required native 48 kHz stereo decoder companion.
pub const AUDIO_TOKENIZER_UPSTREAM_HF: &str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano";
/// Number of residual audio-token channels emitted for each frame.
pub const NUM_AUDIO_CODEBOOKS: usize = 16;
/// Audio vocabulary size per codebook.
pub const AUDIO_CODEBOOK_SIZE: usize = 1_024;
/// Text vocabulary size.
pub const TEXT_VOCAB_SIZE: usize = 16_384;
/// Maximum global context length from the official config.
pub const MAX_POSITION_EMBEDDINGS: usize = 32_768;
/// Audio pad sentinel used in the 17-column prompt matrix.
pub const AUDIO_PAD_TOKEN_ID: u32 = 1_024;
/// Generated-frame text slot.
pub const AUDIO_ASSISTANT_SLOT_TOKEN_ID: u32 = 9;
/// Generation stop token selected by the local text head.
pub const AUDIO_END_TOKEN_ID: u32 = 7;

/// Learned reductions used by both global and local Nano transformers.
pub const MOSS_TTS_NANO_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::GeluNew,
];

/// Generated Nano codes and their authenticated codec decode.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct MossTtsSynthesis {
    /// Greedy frame-major language-model output.
    pub generated: MossTtsGeneratedCodes,
    /// Native MOSS Audio Tokenizer Nano 48 kHz stereo decode.
    pub audio: MossDecodedAudio,
}

const LABEL: &str = "moss_tts/nano";
const TENSOR_COUNT: usize = 194;
const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: ARCH,
    model_name: NAME,
    model_name_alias: None,
    tensor_count: TENSOR_COUNT,
    manifest_sha256: [
        0x12, 0x5c, 0x07, 0x4b, 0x9c, 0xd4, 0x23, 0x7e, 0x0d, 0x05, 0x1f, 0x1c, 0xc8, 0x6f, 0x84,
        0xe8, 0x15, 0x0a, 0x0d, 0x85, 0x4f, 0x4b, 0xee, 0xa9, 0x77, 0xc6, 0x7d, 0x8f, 0x35, 0xe3,
        0x9e, 0xa5,
    ],
};

const CHECKPOINT_SHA256: &str = "24003f2f11ac8a2cbf70514db2d8f1c02fb451aa6b3c0bffc9da09f31cd7caa5";
const CONFIG_SHA256: &str = "ba36b08c80d4ae0805a2bab32b6ac90ec0d1815d01d3854ba42811db1d5bde99";
const SOURCE: &str = "OpenMOSS-Team/MOSS-TTS-Nano-100M (moss_tts_nano, GPT-2 backbone, apache-2.0)";

/// Strictly authenticated MOSS-TTS Nano checkpoint and owned decoded weights.
#[derive(Debug, Clone)]
pub struct MossTtsNano {
    backend: BackendKind,
    weight_license: LicenseClass,
    requires_metadata_repair: bool,
    weights: NanoWeights,
}

impl MossTtsNano {
    /// Opens, authenticates and decodes the exact public Nano GGUF.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Authenticates the complete manifest and all release metadata.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        if checkpoint.weight_license() != LicenseClass::Permissive {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: weight license {:?}, expected permissive Apache-2.0",
                checkpoint.weight_license()
            )));
        }
        let requires_metadata_repair = validate_metadata(file)?;
        Ok(Self {
            backend: BackendKind::Cpu,
            weight_license: checkpoint.weight_license(),
            requires_metadata_repair,
            weights: NanoWeights::bind(file)?,
        })
    }

    /// Binds the exact artifact and preflights every learned operation on one
    /// selected backend. Unsupported backends return an explicit error.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        let _ = Compute::for_backend(backend, MOSS_TTS_NANO_HOT_OPS)?;
        Ok(Self::from_gguf(file)?.with_backend(backend))
    }

    /// Selects one backend for the complete learned graph.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Selected execution backend.
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Fail-closed artifact license class.
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Whether the exact historical public GGUF needs corrected metadata.
    pub const fn requires_metadata_repair(&self) -> bool {
        self.requires_metadata_repair
    }

    /// Greedily generates frame-major 16-codebook tokens from an explicit
    /// upstream-compatible `[rows, 17]` prompt matrix. Column zero is a text
    /// token; columns 1..16 are audio codes or [`AUDIO_PAD_TOKEN_ID`]. The
    /// public GGUF does not bundle `tokenizer.model`, so this API never
    /// invents a raw-text tokenizer or prompt template.
    pub fn generate_codes(
        &self,
        prompt_rows: &[u32],
        max_new_frames: usize,
    ) -> Result<MossTtsGeneratedCodes> {
        let compute = Compute::for_backend(self.backend, MOSS_TTS_NANO_HOT_OPS)?;
        generation::generate_codes(&compute, &self.weights, prompt_rows, max_new_frames)
    }

    /// Runs the complete explicit-companion path from a `[rows,17]` prompt
    /// matrix to 48 kHz stereo PCM. The codec must be the authenticated Nano
    /// release and must use the same backend; Full substitution and a hidden
    /// CPU codec fallback are both rejected.
    pub fn synthesize_prompt_rows(
        &self,
        codec: &MossAudioTokenizer,
        prompt_rows: &[u32],
        max_new_frames: usize,
    ) -> Result<MossTtsSynthesis> {
        if codec.variant() != MossAudioTokenizerVariant::Nano {
            return Err(VokraError::UnsupportedOp(format!(
                "{LABEL}: synthesis requires the exact MOSS Audio Tokenizer Nano companion; got {:?}",
                codec.variant()
            )));
        }
        if codec.backend() != self.backend {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: LLM backend {:?} does not match codec backend {:?}; the composed graph must select one backend and never hide a CPU fallback",
                self.backend,
                codec.backend()
            )));
        }
        let generated = self.generate_codes(prompt_rows, max_new_frames)?;
        let audio = codec.decode_frame_major(
            &generated.codes,
            generated.frames,
            generated.num_codebooks,
        )?;
        Ok(MossTtsSynthesis { generated, audio })
    }
}

fn validate_metadata(file: &GgufFile) -> Result<bool> {
    require_string(file, "vokra.model.category", CATEGORY)?;
    require_string(file, chunks::KEY_PROVENANCE_MODEL_ID, NAME)?;
    require_string(file, chunks::KEY_PROVENANCE_SOURCE, SOURCE)?;
    require_string(file, chunks::KEY_PROVENANCE_LICENSE, "apache-2.0")?;
    require_string(
        file,
        chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
        LicenseClass::Permissive.as_str(),
    )?;
    require_string(file, "vokra.provenance.upstream_hf", UPSTREAM_HF)?;
    require_string(file, "vokra.moss_tts.variant", "nano")?;
    require_string(file, "vokra.moss_tts.llm.family", "gpt2")?;
    for (key, expected) in [
        ("vokra.moss_tts.n_vq", 16),
        ("vokra.moss_tts.audio_vocab_size", 1_024),
        ("vokra.moss_tts.sample_rate", 48_000),
        ("vokra.moss_tts.llm.hidden_dim", 768),
        ("vokra.moss_tts.llm.ffn_dim", 3_072),
        ("vokra.moss_tts.llm.n_layer", 12),
        ("vokra.moss_tts.llm.n_head", 12),
        ("vokra.moss_tts.llm.n_head_kv", 12),
        ("vokra.moss_tts.llm.head_dim", 64),
        ("vokra.moss_tts.llm.vocab_size", 16_384),
    ] {
        require_u32(file, key, expected)?;
    }
    require_f32(file, "vokra.moss_tts.llm.rms_norm_eps", 0.0)?;

    const CORRECTED_KEYS: &[&str] = &[
        "vokra.provenance.upstream_revision",
        "vokra.provenance.checkpoint_sha256",
        "vokra.moss_tts.config_sha256",
        "vokra.moss_tts.llm.position_embedding_type",
        "vokra.moss_tts.llm.layer_norm_eps",
        "vokra.moss_tts.llm.max_position_embeddings",
        "vokra.moss_tts.local_transformer_layers",
        "vokra.moss_tts.pad_token_id",
        "vokra.moss_tts.im_start_token_id",
        "vokra.moss_tts.im_end_token_id",
        "vokra.moss_tts.audio_start_token_id",
        "vokra.moss_tts.audio_end_token_id",
        "vokra.moss_tts.audio_user_slot_token_id",
        "vokra.moss_tts.audio_assistant_slot_token_id",
        "vokra.moss_tts.audio_pad_token_id",
        "vokra.moss_tts.audio_tokenizer_upstream_hf",
    ];
    let corrected_count = CORRECTED_KEYS
        .iter()
        .filter(|&&key| file.get(key).is_some())
        .count();
    if corrected_count == 0 {
        require_f32(file, "vokra.moss_tts.llm.rope_base", 0.0)?;
        return Ok(true);
    }
    if corrected_count != CORRECTED_KEYS.len() {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: corrected Nano metadata is partial ({corrected_count}/{} keys); refusing a mixed legacy contract",
            CORRECTED_KEYS.len()
        )));
    }

    require_f32(file, "vokra.moss_tts.llm.rope_base", 10_000.0)?;
    require_string(
        file,
        "vokra.provenance.upstream_revision",
        UPSTREAM_REVISION,
    )?;
    require_string(
        file,
        "vokra.provenance.checkpoint_sha256",
        CHECKPOINT_SHA256,
    )?;
    require_string(file, "vokra.moss_tts.config_sha256", CONFIG_SHA256)?;
    require_string(file, "vokra.moss_tts.llm.position_embedding_type", "rope")?;
    require_f32(file, "vokra.moss_tts.llm.layer_norm_eps", 1.0e-5)?;
    for (key, expected) in [
        ("vokra.moss_tts.llm.max_position_embeddings", 32_768),
        ("vokra.moss_tts.local_transformer_layers", 1),
        ("vokra.moss_tts.pad_token_id", 3),
        ("vokra.moss_tts.im_start_token_id", 4),
        ("vokra.moss_tts.im_end_token_id", 5),
        ("vokra.moss_tts.audio_start_token_id", 6),
        ("vokra.moss_tts.audio_end_token_id", 7),
        ("vokra.moss_tts.audio_user_slot_token_id", 8),
        ("vokra.moss_tts.audio_assistant_slot_token_id", 9),
        ("vokra.moss_tts.audio_pad_token_id", 1_024),
    ] {
        require_u32(file, key, expected)?;
    }
    require_string(
        file,
        "vokra.moss_tts.audio_tokenizer_upstream_hf",
        AUDIO_TOKENIZER_UPSTREAM_HF,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nano_backend_contract_covers_cpu_and_metal() {
        Compute::for_backend(BackendKind::Cpu, MOSS_TTS_NANO_HOT_OPS)
            .expect("CPU covers the MOSS-TTS Nano learned graph");
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        match Compute::for_backend(BackendKind::Metal, MOSS_TTS_NANO_HOT_OPS) {
            Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(error) => panic!("MOSS-TTS Nano has a Metal coverage gap: {error}"),
        }
    }
}
