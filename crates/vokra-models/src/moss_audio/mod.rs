//! Strict MOSS-Audio 4B/8B Instruct checkpoint binding.
//!
//! OpenMOSS publishes both models as a custom `moss_audio` audio-LLM: a
//! 32-layer Whisper-style audio tower, four GatedMLP adapters (one primary and
//! three DeepStack injections), and a 36-layer Qwen3 language model. The two
//! variants differ only in the text hidden/FFN widths.
//!
//! The first Vokra publications predated that architecture audit and carry a
//! historical `moss_tts` arch stamp plus placeholder TTS topology metadata.
//! This module does not trust or reinterpret those axes. It admits only the
//! exact public 901-tensor name/shape manifest for the matching model name and
//! upstream repository, then derives the immutable official topology from the
//! authenticated variant. Newly converted artifacts must carry the dedicated
//! `moss_audio` arch and the complete fixed-revision metadata group.

use std::path::Path;
use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{FrontendPolicy, FrontendSpec, LicenseClass, Result, VokraError};

use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec, verify_tensor_manifest};

mod audio_encoder;
mod frontend;
mod text_decoder;
mod tokenizer;
mod weights;

use audio_encoder::MossAudioEncoderRuntime;
pub use audio_encoder::{MOSS_AUDIO_ENCODER_HOT_OPS, MossAudioEmbeddings};
use text_decoder::MossAudioTextRuntime;
pub use text_decoder::{
    MOSS_AUDIO_HOT_OPS, MOSS_AUDIO_TEXT_HOT_OPS, MossAudioGenerationOptions, MossAudioTokenOutput,
};
pub use tokenizer::{
    AUDIO_END_TOKEN_ID, AUDIO_START_TOKEN_ID, AUDIO_TOKEN_ID, BASE_VOCAB_SIZE, DEFAULT_USER_PROMPT,
    END_OF_TEXT_TOKEN_ID, IM_END_TOKEN_ID, IM_START_TOKEN_ID, MossAudioTextTokenizer,
};
use weights::MossAudioMappedDescriptors;

/// Dedicated architecture emitted by corrected conversions.
pub const EXPECTED_ARCH: &str = "moss_audio";
/// Narrowly authenticated arch on the already-published legacy GGUFs.
pub const LEGACY_PUBLIC_ARCH: &str = "moss_tts";
/// Required input waveform rate.
pub const SAMPLE_RATE: u32 = 16_000;

const LABEL: &str = "moss_audio";
const CATEGORY: &str = "s2s";
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_LEGACY_VARIANT: &str = "vokra.moss_tts.variant";
const PREFIX: &str = "vokra.moss_audio.";

const KEY_VARIANT: &str = "vokra.moss_audio.variant";
const KEY_SOURCE_REVISION: &str = "vokra.moss_audio.source_revision";
const KEY_SOURCE_CODE_REVISION: &str = "vokra.moss_audio.source_code_revision";
const KEY_CONFIG_SHA256: &str = "vokra.moss_audio.config_sha256";
const KEY_CONFIGURATION_SOURCE_SHA256: &str = "vokra.moss_audio.configuration_source_sha256";
const KEY_MODELING_SOURCE_SHA256: &str = "vokra.moss_audio.modeling_source_sha256";
const KEY_PROCESSING_SOURCE_SHA256: &str = "vokra.moss_audio.processing_source_sha256";
const KEY_TENSOR_MANIFEST_SHA256: &str = "vokra.moss_audio.tensor_manifest_sha256";

const SOURCE_CODE_REVISION: &str = "5cbb1d823937cd5b5de3d8fa4d3a7253ebd3b883";
const CONFIGURATION_SOURCE_SHA256: &str =
    "e597dca441ff7fb58a5ec43186fafdfce19f31dada4955b4910059baa5d52ebd";
const MODELING_SOURCE_SHA256: &str =
    "a52513e518c68a0ba7c636a1ab0e12f7755ceebd0ae033235dc5e2551bfcbf9c";
const PROCESSING_SOURCE_SHA256: &str =
    "05fb788cbdc6482eded8d70f7d2f524bc0cdca47d001acab5661c11f02cc6fe6";

const SPEC_4B: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: EXPECTED_ARCH,
    model_name: "moss-audio-4b-instruct",
    model_name_alias: None,
    tensor_count: 901,
    manifest_sha256: [
        0x4d, 0xb8, 0xbf, 0xa2, 0xa5, 0x4b, 0x75, 0x41, 0xdc, 0x09, 0x2b, 0x73, 0x91, 0x97, 0x71,
        0xfd, 0xef, 0xa9, 0x52, 0xea, 0x1b, 0x05, 0x4c, 0xe1, 0x08, 0x45, 0xe9, 0xd2, 0xbc, 0xd6,
        0xfa, 0xdc,
    ],
};

const SPEC_8B: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: EXPECTED_ARCH,
    model_name: "moss-audio-8b-instruct",
    model_name_alias: None,
    tensor_count: 901,
    manifest_sha256: [
        0x76, 0xc1, 0x27, 0x5d, 0xab, 0xd9, 0xa3, 0xba, 0xf0, 0x18, 0x9f, 0x5f, 0xc3, 0x35, 0xa6,
        0xc1, 0x92, 0xc4, 0x72, 0xe9, 0x6b, 0xc3, 0x63, 0xcc, 0x3a, 0x64, 0xad, 0x2d, 0x37, 0xa5,
        0xf8, 0x3a,
    ],
};

/// One of the two official MOSS-Audio Instruct releases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MossAudioVariant {
    /// `OpenMOSS-Team/MOSS-Audio-4B-Instruct`.
    B4Instruct,
    /// `OpenMOSS-Team/MOSS-Audio-8B-Instruct`.
    B8Instruct,
}

impl MossAudioVariant {
    /// Exact `vokra.model.name` admitted by the binder.
    #[must_use]
    pub const fn model_name(self) -> &'static str {
        match self {
            Self::B4Instruct => "moss-audio-4b-instruct",
            Self::B8Instruct => "moss-audio-8b-instruct",
        }
    }

    /// Dedicated release discriminator.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::B4Instruct => "4b_instruct",
            Self::B8Instruct => "8b_instruct",
        }
    }

    /// Immutable upstream Hugging Face repository.
    #[must_use]
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::B4Instruct => "OpenMOSS-Team/MOSS-Audio-4B-Instruct",
            Self::B8Instruct => "OpenMOSS-Team/MOSS-Audio-8B-Instruct",
        }
    }

    /// Immutable upstream snapshot revision.
    #[must_use]
    pub const fn upstream_revision(self) -> &'static str {
        match self {
            Self::B4Instruct => "6907a499dc0e87cc77c8ae0fe23fd0eb5476a02d",
            Self::B8Instruct => "6521a39181b47a18f2d9f4b3acfb5bca7b76b57f",
        }
    }

    /// SHA-256 of the pinned upstream `config.json`.
    #[must_use]
    pub const fn config_sha256(self) -> &'static str {
        match self {
            Self::B4Instruct => "e528a941446f4443f1b9fede12ea484e58a79d494c28d21ef1e73b5148abfbfa",
            Self::B8Instruct => "535154c2a5bcbd0e18e2f92bcf370ac74b530eec97ad4fd9317993ba0a316536",
        }
    }

    /// Canonical lowercase SHA-256 of the complete sorted tensor manifest.
    #[must_use]
    pub const fn manifest_sha256(self) -> &'static str {
        match self {
            Self::B4Instruct => "4db8bfa2a54b7541dc092b73919771fdefa952ea1b054ce10845e9d2bcd6fadc",
            Self::B8Instruct => "76c1275dabd9a3baf0189f5fc335a6c192c472e96bc363cc3a64ad2d37a5f83a",
        }
    }

    const fn spec(self) -> StrictCheckpointSpec {
        match self {
            Self::B4Instruct => SPEC_4B,
            Self::B8Instruct => SPEC_8B,
        }
    }

    /// Immutable official topology for this release.
    #[must_use]
    pub const fn config(self) -> MossAudioConfig {
        let (hidden_size, ffn_dim) = match self {
            Self::B4Instruct => (2_560, 9_728),
            Self::B8Instruct => (4_096, 12_288),
        };
        MossAudioConfig {
            audio: MossAudioEncoderConfig {
                d_model: 1_280,
                output_dim: 1_280,
                n_mels: 128,
                n_layer: 32,
                n_head: 20,
                ffn_dim: 5_120,
                downsample_rate: 8,
                downsample_hidden_size: 480,
                attention_window_size: 100,
                max_source_positions: 1_500,
                n_window: 200,
                conv_chunksize: 64,
                layer_norm_eps: 1.0e-5,
                deepstack_layer_indexes: [8, 16, 24],
            },
            adapter_hidden_size: 8_192,
            deepstack_num_inject_layers: 3,
            text: MossAudioTextConfig {
                hidden_size,
                ffn_dim,
                n_layer: 36,
                n_head: 32,
                n_kv_head: 8,
                head_dim: 128,
                max_position_embeddings: 40_960,
                vocab_size: 151_936,
                rope_theta: 1_000_000.0,
                rms_norm_eps: 1.0e-6,
                tie_word_embeddings: false,
                attention_bias: false,
            },
            tokens: MossAudioTokenConfig {
                audio: 151_654,
                audio_start: 151_669,
                audio_end: 151_670,
                bos: 151_643,
                eos: 151_645,
            },
        }
    }
}

/// Whisper-style audio encoder topology.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MossAudioEncoderConfig {
    pub d_model: u32,
    pub output_dim: u32,
    pub n_mels: u32,
    pub n_layer: u32,
    pub n_head: u32,
    pub ffn_dim: u32,
    pub downsample_rate: u32,
    pub downsample_hidden_size: u32,
    pub attention_window_size: u32,
    pub max_source_positions: u32,
    pub n_window: u32,
    pub conv_chunksize: u32,
    pub layer_norm_eps: f32,
    pub deepstack_layer_indexes: [u32; 3],
}

/// Qwen3 decoder topology.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MossAudioTextConfig {
    pub hidden_size: u32,
    pub ffn_dim: u32,
    pub n_layer: u32,
    pub n_head: u32,
    pub n_kv_head: u32,
    pub head_dim: u32,
    pub max_position_embeddings: u32,
    pub vocab_size: u32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    pub attention_bias: bool,
}

/// Special token IDs consumed by the official processor/model pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MossAudioTokenConfig {
    pub audio: u32,
    pub audio_start: u32,
    pub audio_end: u32,
    pub bos: u32,
    pub eos: u32,
}

/// Full immutable model topology.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MossAudioConfig {
    pub audio: MossAudioEncoderConfig,
    pub adapter_hidden_size: u32,
    pub deepstack_num_inject_layers: u32,
    pub text: MossAudioTextConfig,
    pub tokens: MossAudioTokenConfig,
}

/// Header-only proof that a GGUF is one exact MOSS-Audio release.
#[derive(Clone)]
pub struct MossAudioCheckpoint {
    model_name: String,
    variant: MossAudioVariant,
    config: MossAudioConfig,
    weight_license: LicenseClass,
    legacy_public_metadata: bool,
    mapped: Option<Arc<MossAudioMappedDescriptors>>,
    tokenizer: Option<Arc<MossAudioTextTokenizer>>,
}

impl std::fmt::Debug for MossAudioCheckpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MossAudioCheckpoint")
            .field("variant", &self.variant)
            .field("tensor_count", &self.tensor_count())
            .field("weight_license", &self.weight_license)
            .field("legacy_public_metadata", &self.legacy_public_metadata)
            .field("mapped", &self.mapped.is_some())
            .field("tokenizer", &self.tokenizer.is_some())
            .finish()
    }
}

impl MossAudioCheckpoint {
    /// Opens through the true mmap loader and validates all dense descriptors
    /// without decoding tensor payloads.
    pub fn open_mapped(path: impl AsRef<Path>) -> Result<Self> {
        let file = vokra_mmap::open_gguf(path.as_ref()).map_err(VokraError::from)?;
        Self::from_gguf_mapped(Arc::new(file))
    }

    /// Strictly binds an already mmap-backed GGUF.
    pub fn from_gguf_mapped(file: Arc<GgufFile>) -> Result<Self> {
        let mut checkpoint = Self::from_gguf(&file)?;
        let tokenizer = if checkpoint.legacy_public_metadata {
            None
        } else {
            Some(Arc::new(MossAudioTextTokenizer::from_gguf(
                &file,
                checkpoint.variant,
            )?))
        };
        let mapped = MossAudioMappedDescriptors::bind(file, checkpoint.config)?;
        checkpoint.mapped = Some(Arc::new(mapped));
        checkpoint.tokenizer = tokenizer;
        Ok(checkpoint)
    }

    /// Validates identity, provenance, topology and complete tensor manifest.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let model_name = required_string(file, chunks::KEY_MODEL_NAME)?;
        let variant = match model_name {
            "moss-audio-4b-instruct" => MossAudioVariant::B4Instruct,
            "moss-audio-8b-instruct" => MossAudioVariant::B8Instruct,
            other => {
                return Err(VokraError::ModelLoad(format!(
                    "{LABEL}: unsupported `{}`={other:?}; expected {:?} or {:?}",
                    chunks::KEY_MODEL_NAME,
                    MossAudioVariant::B4Instruct.model_name(),
                    MossAudioVariant::B8Instruct.model_name()
                )));
            }
        };
        require_string_value(file, KEY_MODEL_CATEGORY, CATEGORY)?;
        require_string_value(file, KEY_UPSTREAM_HF, variant.upstream_hf())?;

        let arch = required_string(file, chunks::KEY_MODEL_ARCH)?;
        let legacy_public_metadata = match arch {
            EXPECTED_ARCH => {
                let strict = StrictCheckpoint::bind(file, variant.spec())?;
                require_permissive(strict.weight_license())?;
                validate_canonical_metadata(file, variant)?;
                false
            }
            LEGACY_PUBLIC_ARCH => {
                require_string_value(file, KEY_LEGACY_VARIANT, variant.tag())?;
                if let Some((key, _)) = file
                    .metadata()
                    .iter()
                    .find(|(key, _)| key.starts_with(PREFIX))
                {
                    return Err(VokraError::ModelLoad(format!(
                        "{LABEL}: historical `{LEGACY_PUBLIC_ARCH}` artifact contains partial corrected metadata `{key}`; refusing a mixed legacy/canonical contract"
                    )));
                }
                verify_tensor_manifest(
                    file,
                    LABEL,
                    variant.spec().tensor_count,
                    variant.spec().manifest_sha256,
                    variant.model_name(),
                )?;
                require_permissive(weight_license(file))?;
                true
            }
            other => {
                return Err(VokraError::ModelLoad(format!(
                    "{LABEL}: unsupported `{}`={other:?}; expected `{EXPECTED_ARCH}` or the exact-manifest legacy `{LEGACY_PUBLIC_ARCH}` publication",
                    chunks::KEY_MODEL_ARCH
                )));
            }
        };

        Ok(Self {
            model_name: model_name.to_owned(),
            variant,
            config: variant.config(),
            weight_license: LicenseClass::Permissive,
            legacy_public_metadata,
            mapped: None,
            tokenizer: None,
        })
    }

    #[must_use]
    pub const fn variant(&self) -> MossAudioVariant {
        self.variant
    }

    #[must_use]
    pub const fn config(&self) -> &MossAudioConfig {
        &self.config
    }

    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        901
    }

    /// True only for the exact already-published GGUF with the wrong arch.
    #[must_use]
    pub const fn legacy_public_metadata(&self) -> bool {
        self.legacy_public_metadata
    }

    #[must_use]
    pub const fn is_mapped(&self) -> bool {
        self.mapped.is_some()
    }

    /// Whether this executable checkpoint carries all six authenticated text
    /// tokenizer/chat/generation/processor sidecars.
    #[must_use]
    pub const fn has_text_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }

    /// Loud boundary on the checkpoint descriptor handle.
    ///
    /// Executable inference lives on [`MossAudio`], which requires an explicit
    /// CPU or Metal backend. Keeping this method non-executing prevents a
    /// backend-less checkpoint from selecting CPU implicitly.
    pub fn respond(&self, pcm: &[f32], sample_rate: u32) -> Result<String> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "moss_audio respond: PCM input is empty".to_owned(),
            ));
        }
        if sample_rate != SAMPLE_RATE {
            return Err(VokraError::InvalidArgument(format!(
                "moss_audio respond: sample_rate={sample_rate}, expected {SAMPLE_RATE} Hz"
            )));
        }
        Err(VokraError::UnsupportedOp(format!(
            "moss_audio checkpoint respond: {} is a descriptor handle without a selected backend; construct MossAudio::from_checkpoint(checkpoint, BackendKind::Cpu or BackendKind::Metal) explicitly",
            self.variant.model_name()
        )))
    }

    pub(super) fn mapped(&self) -> Result<&MossAudioMappedDescriptors> {
        self.mapped.as_deref().ok_or_else(|| {
            VokraError::ModelLoad(
                "moss_audio: executable inference requires MossAudioCheckpoint::open_mapped or from_gguf_mapped; from_gguf performs descriptor-only validation"
                    .to_owned(),
            )
        })
    }

    fn text_tokenizer(&self) -> Result<&MossAudioTextTokenizer> {
        self.tokenizer.as_deref().ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "moss_audio tokenizer: {} has no authenticated embedded tokenizer/chat/generation/processor sidecars; historical GGUFs remain token-level only and must be re-converted from the fixed upstream revision for string generation",
                self.variant.model_name()
            ))
        })
    }
}

/// Executable MOSS-Audio audio tower, adapters and Qwen3 decoder.
///
/// Every route keeps the multi-gigabyte checkpoint mapped and widens one layer
/// at a time. Corrected GGUFs expose authenticated string generation; exact
/// historical publications without sidecars remain token-level only.
#[derive(Clone)]
pub struct MossAudio {
    checkpoint: MossAudioCheckpoint,
    backend: BackendKind,
    encoder: Arc<MossAudioEncoderRuntime>,
    text: Arc<MossAudioTextRuntime>,
}

/// Decoded MOSS-Audio response plus the exact generated token sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MossAudioResponse {
    text: String,
    token_ids: Vec<u32>,
}

impl MossAudioResponse {
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    #[must_use]
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    #[must_use]
    pub fn into_text(self) -> String {
        self.text
    }
}

impl std::fmt::Debug for MossAudio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MossAudio")
            .field("variant", &self.checkpoint.variant())
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

impl MossAudio {
    /// Opens an exact dense GGUF through mmap and preflights every learned
    /// audio/adapter op on the selected CPU or Metal backend.
    pub fn open_mapped(path: impl AsRef<Path>, backend: BackendKind) -> Result<Self> {
        Self::from_checkpoint(MossAudioCheckpoint::open_mapped(path)?, backend)
    }

    /// Builds the executable audio side from a mapped strict checkpoint.
    pub fn from_checkpoint(checkpoint: MossAudioCheckpoint, backend: BackendKind) -> Result<Self> {
        checkpoint.mapped()?;
        let _ = crate::compute::Compute::for_backend(backend, MOSS_AUDIO_HOT_OPS)?;
        Ok(Self {
            checkpoint,
            backend,
            encoder: Arc::new(MossAudioEncoderRuntime::default()),
            text: Arc::new(MossAudioTextRuntime::default()),
        })
    }

    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    #[must_use]
    pub const fn checkpoint(&self) -> &MossAudioCheckpoint {
        &self.checkpoint
    }

    /// Returns the authenticated fixed-revision text tokenizer.
    ///
    /// Historical public GGUFs intentionally return an error because they do
    /// not embed the six required sidecars; callers may still use their
    /// explicit token-level route.
    pub fn tokenizer(&self) -> Result<&MossAudioTextTokenizer> {
        self.checkpoint.text_tokenizer()
    }

    /// Runs the exact frontend, 3-stage convolutional stem, 32 Whisper-style
    /// layers, final LayerNorm and four GatedMLP projections. The result
    /// contains primary text-width audio embeddings plus all three DeepStack
    /// tensors needed by the Qwen3 decoder.
    pub fn encode_audio(&self, pcm: &[f32], sample_rate: u32) -> Result<MossAudioEmbeddings> {
        validate_audio_input(pcm, sample_rate)?;
        audio_encoder::encode(&self.checkpoint, self.backend, &self.encoder, pcm)
    }

    /// Runs audio encoding plus the complete 36-layer Qwen3 decoder from a
    /// caller-supplied official prompt-token sequence. The sequence must carry
    /// exactly one `tokens.audio` placeholder for every encoded audio row.
    /// This token-level entry remains available for the exact historical
    /// publications that predate embedded tokenizer sidecars.
    pub fn generate_tokens(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        prompt_tokens: &[u32],
        options: &MossAudioGenerationOptions,
    ) -> Result<MossAudioTokenOutput> {
        validate_audio_input(pcm, sample_rate)?;
        let audio = audio_encoder::encode(&self.checkpoint, self.backend, &self.encoder, pcm)?;
        text_decoder::generate(
            &self.checkpoint,
            self.backend,
            &self.text,
            prompt_tokens,
            &audio,
            options,
        )
    }

    /// Runs the official example prompt (`"Describe this audio."`) with the
    /// default deterministic generation controls.
    pub fn respond(&self, pcm: &[f32], sample_rate: u32) -> Result<String> {
        self.respond_with_prompt(
            pcm,
            sample_rate,
            DEFAULT_USER_PROMPT,
            &MossAudioGenerationOptions::default(),
        )
        .map(MossAudioResponse::into_text)
    }

    /// Runs one official single-audio ChatML request with caller-supplied text.
    ///
    /// Corrected conversions authenticate all six fixed-revision sidecars.
    /// The legacy public GGUFs fail before audio execution because using a
    /// host tokenizer would make the checkpoint mutable and unverifiable.
    pub fn respond_with_prompt(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        prompt: &str,
        options: &MossAudioGenerationOptions,
    ) -> Result<MossAudioResponse> {
        validate_audio_input(pcm, sample_rate)?;
        let tokenizer = self.checkpoint.text_tokenizer()?;
        let audio = audio_encoder::encode(&self.checkpoint, self.backend, &self.encoder, pcm)?;
        let prompt_tokens = tokenizer.prompt_ids(audio.frames(), prompt)?;
        let output = text_decoder::generate(
            &self.checkpoint,
            self.backend,
            &self.text,
            &prompt_tokens,
            &audio,
            options,
        )?;
        let token_ids = output.into_token_ids();
        let text = tokenizer.decode_generated_ids(&token_ids)?;
        Ok(MossAudioResponse { text, token_ids })
    }
}

fn validate_audio_input(pcm: &[f32], sample_rate: u32) -> Result<()> {
    if pcm.is_empty() {
        return Err(VokraError::InvalidArgument(
            "moss_audio: PCM input is empty".to_owned(),
        ));
    }
    if sample_rate != SAMPLE_RATE {
        return Err(VokraError::InvalidArgument(format!(
            "moss_audio: sample_rate={sample_rate}, expected {SAMPLE_RATE} Hz"
        )));
    }
    Ok(())
}

fn validate_canonical_metadata(file: &GgufFile, variant: MossAudioVariant) -> Result<()> {
    require_string_value(file, KEY_UPSTREAM_REVISION, variant.upstream_revision())?;
    require_string_value(file, KEY_VARIANT, variant.tag())?;
    require_string_value(file, KEY_SOURCE_REVISION, variant.upstream_revision())?;
    require_string_value(file, KEY_SOURCE_CODE_REVISION, SOURCE_CODE_REVISION)?;
    require_string_value(file, KEY_CONFIG_SHA256, variant.config_sha256())?;
    require_string_value(
        file,
        KEY_CONFIGURATION_SOURCE_SHA256,
        CONFIGURATION_SOURCE_SHA256,
    )?;
    require_string_value(file, KEY_MODELING_SOURCE_SHA256, MODELING_SOURCE_SHA256)?;
    require_string_value(file, KEY_PROCESSING_SOURCE_SHA256, PROCESSING_SOURCE_SHA256)?;
    require_string_value(file, KEY_TENSOR_MANIFEST_SHA256, variant.manifest_sha256())?;

    let config = variant.config();
    for (key, expected) in [
        ("vokra.moss_audio.audio.d_model", config.audio.d_model),
        ("vokra.moss_audio.audio.output_dim", config.audio.output_dim),
        ("vokra.moss_audio.audio.n_mels", config.audio.n_mels),
        ("vokra.moss_audio.audio.n_layer", config.audio.n_layer),
        ("vokra.moss_audio.audio.n_head", config.audio.n_head),
        ("vokra.moss_audio.audio.ffn_dim", config.audio.ffn_dim),
        (
            "vokra.moss_audio.audio.downsample_rate",
            config.audio.downsample_rate,
        ),
        (
            "vokra.moss_audio.audio.downsample_hidden_size",
            config.audio.downsample_hidden_size,
        ),
        (
            "vokra.moss_audio.audio.attention_window_size",
            config.audio.attention_window_size,
        ),
        (
            "vokra.moss_audio.audio.max_source_positions",
            config.audio.max_source_positions,
        ),
        ("vokra.moss_audio.audio.n_window", config.audio.n_window),
        (
            "vokra.moss_audio.audio.conv_chunksize",
            config.audio.conv_chunksize,
        ),
        (
            "vokra.moss_audio.adapter_hidden_size",
            config.adapter_hidden_size,
        ),
        (
            "vokra.moss_audio.deepstack_num_inject_layers",
            config.deepstack_num_inject_layers,
        ),
        ("vokra.moss_audio.text.hidden_size", config.text.hidden_size),
        ("vokra.moss_audio.text.ffn_dim", config.text.ffn_dim),
        ("vokra.moss_audio.text.n_layer", config.text.n_layer),
        ("vokra.moss_audio.text.n_head", config.text.n_head),
        ("vokra.moss_audio.text.n_head_kv", config.text.n_kv_head),
        ("vokra.moss_audio.text.head_dim", config.text.head_dim),
        (
            "vokra.moss_audio.text.max_position_embeddings",
            config.text.max_position_embeddings,
        ),
        ("vokra.moss_audio.text.vocab_size", config.text.vocab_size),
        ("vokra.moss_audio.token.audio", config.tokens.audio),
        (
            "vokra.moss_audio.token.audio_start",
            config.tokens.audio_start,
        ),
        ("vokra.moss_audio.token.audio_end", config.tokens.audio_end),
        ("vokra.moss_audio.token.bos", config.tokens.bos),
        ("vokra.moss_audio.token.eos", config.tokens.eos),
    ] {
        require_u32_value(file, key, expected)?;
    }
    require_f32_value(
        file,
        "vokra.moss_audio.audio.layer_norm_eps",
        config.audio.layer_norm_eps,
    )?;
    require_string_value(file, "vokra.moss_audio.audio.activation", "gelu")?;
    require_u32_array_value(
        file,
        "vokra.moss_audio.audio.deepstack_layer_indexes",
        &config.audio.deepstack_layer_indexes,
    )?;
    require_f32_value(
        file,
        "vokra.moss_audio.text.rope_theta",
        config.text.rope_theta,
    )?;
    require_f32_value(
        file,
        "vokra.moss_audio.text.rms_norm_eps",
        config.text.rms_norm_eps,
    )?;
    require_bool_value(
        file,
        "vokra.moss_audio.text.tie_word_embeddings",
        config.text.tie_word_embeddings,
    )?;
    require_bool_value(
        file,
        "vokra.moss_audio.text.attention_bias",
        config.text.attention_bias,
    )?;

    FrontendSpec::from_gguf(file)?
        .check_against(&frontend::runtime_frontend_spec(), FrontendPolicy::Fail)
}

fn weight_license(file: &GgufFile) -> LicenseClass {
    file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
        .and_then(GgufMetadataValue::as_str)
        .and_then(LicenseClass::from_class_str)
        .unwrap_or(LicenseClass::Unknown)
}

fn require_permissive(actual: LicenseClass) -> Result<()> {
    if actual != LicenseClass::Permissive {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{}` must classify as `permissive` for the pinned Apache-2.0 release, got {actual:?}",
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE
        )));
    }
    Ok(())
}

fn required_string<'a>(file: &'a GgufFile, key: &str) -> Result<&'a str> {
    file.get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| VokraError::ModelLoad(format!("{LABEL}: missing/non-string `{key}`")))
}

fn require_string_value(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = required_string(file, key)?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}`={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn require_u32_value(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    match file.get(key) {
        Some(GgufMetadataValue::U32(actual)) if *actual == expected => Ok(()),
        actual => Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}`={actual:?}, expected U32({expected})"
        ))),
    }
}

fn require_f32_value(file: &GgufFile, key: &str, expected: f32) -> Result<()> {
    match file.get(key) {
        Some(GgufMetadataValue::F32(actual)) if actual.to_bits() == expected.to_bits() => Ok(()),
        actual => Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}`={actual:?}, expected F32({expected:?})"
        ))),
    }
}

fn require_bool_value(file: &GgufFile, key: &str, expected: bool) -> Result<()> {
    match file.get(key) {
        Some(GgufMetadataValue::Bool(actual)) if *actual == expected => Ok(()),
        actual => Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}`={actual:?}, expected Bool({expected})"
        ))),
    }
}

fn require_u32_array_value(file: &GgufFile, key: &str, expected: &[u32]) -> Result<()> {
    let actual = match file.get(key) {
        Some(GgufMetadataValue::Array(array)) => array
            .values
            .iter()
            .map(|value| match value {
                GgufMetadataValue::U32(value) => Ok(*value),
                other => Err(VokraError::ModelLoad(format!(
                    "{LABEL}: `{key}` contains {other:?}, expected only U32 elements"
                ))),
            })
            .collect::<Result<Vec<_>>>()?,
        actual => {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: `{key}`={actual:?}, expected U32 array {expected:?}"
            )));
        }
    };
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}`={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::strict_checkpoint::sha256_bytes;

    fn manifest_sha256(manifest: &BTreeMap<String, Vec<u64>>) -> [u8; 32] {
        let mut canonical = Vec::new();
        for (name, shape) in manifest {
            canonical.extend_from_slice(name.as_bytes());
            canonical.push(0);
            canonical.extend_from_slice(&(shape.len() as u64).to_le_bytes());
            for dimension in shape {
                canonical.extend_from_slice(&dimension.to_le_bytes());
            }
        }
        sha256_bytes(&canonical)
    }

    #[test]
    fn generated_manifests_match_range_audited_public_headers() {
        for variant in [MossAudioVariant::B4Instruct, MossAudioVariant::B8Instruct] {
            let manifest = weights::tensor_contract(variant.config())
                .into_iter()
                .collect::<BTreeMap<_, _>>();
            assert_eq!(manifest.len(), 901);
            assert_eq!(manifest_sha256(&manifest), variant.spec().manifest_sha256);
        }
    }

    #[test]
    fn variant_axes_pin_audio_and_qwen3_widths() {
        let b4 = MossAudioVariant::B4Instruct.config();
        let b8 = MossAudioVariant::B8Instruct.config();
        assert_eq!(b4.audio, b8.audio);
        assert_eq!(b4.text.hidden_size, 2_560);
        assert_eq!(b4.text.ffn_dim, 9_728);
        assert_eq!(b8.text.hidden_size, 4_096);
        assert_eq!(b8.text.ffn_dim, 12_288);
        assert_eq!(b4.audio.deepstack_layer_indexes, [8, 16, 24]);
    }

    #[test]
    fn fixed_revisions_are_full_lowercase_commits() {
        for revision in [
            SOURCE_CODE_REVISION,
            MossAudioVariant::B4Instruct.upstream_revision(),
            MossAudioVariant::B8Instruct.upstream_revision(),
        ] {
            assert_eq!(revision.len(), 40);
            assert!(revision.bytes().all(|byte| byte.is_ascii_hexdigit()));
            assert_eq!(revision, revision.to_ascii_lowercase());
        }
    }
}
