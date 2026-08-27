//! Native Microsoft SpeechT5 TTS checkpoint contract.
//!
//! This module binds two immutable identities with the same authenticated
//! 393-F32-tensor manifest: the complete release produced by the strict
//! `vokra-convert` path, and the exact historical `vokra/speecht5-tts` GGUF.
//! The historical file has an exact 23-key metadata contract but omitted the
//! tokenizer and newer runtime axes. Only after its complete tensor manifest
//! and all 23 legacy fields match do we supply the audited official config and
//! 79-piece SentencePiece CHAR vocabulary plus two Hugging Face added tokens
//! from compiled constants. Unknown or partially upgraded artifacts still
//! fail before tensor payload decoding.
//!
//! The cheap checkpoint handle lets callers audit identity and text
//! tokenization without widening all ~585 MB of F32 payload. [`SpeechT5Tts`]
//! is the separate complete CPU/Metal text-to-mel runtime and can attach the
//! strict SpeechT5 HiFi-GAN companion for waveform synthesis.

use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

mod tokenizer;
pub use tokenizer::SpeechT5Tokenizer;
mod forward;
mod weights;
pub use forward::{SPEECHT5_HOT_OPS, SpeechT5GenerationOptions, SpeechT5Mel, SpeechT5Tts};

/// Architecture tag written by the strict converter.
pub const EXPECTED_ARCH: &str = "speecht5";
/// Canonical model name written by the strict converter.
pub const MODEL_NAME: &str = "speecht5-tts";
/// Pinned Microsoft checkpoint revision.
pub const SOURCE_REVISION: &str = "30fcde30f19b87502b8435427b5f5068e401d5f6";
/// Pinned source `pytorch_model.bin` content hash.
pub const SOURCE_WEIGHT_SHA256: &str =
    "d60d28067349ef66b50d8cd643ae56b6d6b8f27def929bc4ef6fcad907954190";
/// Canonical 393-tensor name/shape manifest hash.
pub const TENSOR_MANIFEST_SHA256: &str =
    "fd6a1323b4994781daf6b657e690cca1e741ee2f7810fab03d0d22bf62301e04";
/// Revision carrying the exact historical public Vokra GGUF.
pub const LEGACY_PUBLIC_REVISION: &str = "43cf6592038616d116a98fde4764d827ece59033";
/// Exact byte length of the historical public Vokra GGUF.
pub const LEGACY_PUBLIC_GGUF_BYTES: u64 = 585_382_432;
/// SHA-256 of the historical public Vokra GGUF hosted at
/// [`LEGACY_PUBLIC_REVISION`].
pub const LEGACY_PUBLIC_GGUF_SHA256: &str =
    "f26019f5e2f7106d834b0b1fd4f66286839e000350caad169388467452c8dde0";

/// Hidden width of the released SpeechT5 transformer.
pub const HIDDEN_SIZE: usize = 768;
/// Number of text-encoder transformer layers.
pub const ENCODER_LAYERS: usize = 12;
/// Number of speech-decoder transformer layers.
pub const DECODER_LAYERS: usize = 6;
/// Number of attention heads in each encoder layer.
pub const ENCODER_ATTENTION_HEADS: usize = 12;
/// Number of attention heads in each decoder layer.
pub const DECODER_ATTENTION_HEADS: usize = 12;
/// Intermediate width of each encoder feed-forward network.
pub const ENCODER_FFN_DIM: usize = 3_072;
/// Intermediate width of each decoder feed-forward network.
pub const DECODER_FFN_DIM: usize = 3_072;
/// Size of the authenticated text-token vocabulary.
pub const VOCAB_SIZE: usize = 81;
/// Number of mel-frequency bins emitted per spectrogram frame.
pub const NUM_MEL_BINS: usize = 80;
/// Number of mel frames predicted by one decoder step.
pub const REDUCTION_FACTOR: usize = 2;
/// Hidden width of each speech-decoder prenet layer.
pub const SPEECH_DECODER_PRENET_UNITS: usize = 256;
/// Number of speech-decoder prenet layers.
pub const SPEECH_DECODER_PRENET_LAYERS: usize = 2;
/// Channel width of the convolutional speech postnet.
pub const SPEECH_DECODER_POSTNET_UNITS: usize = 256;
/// Number of convolutional speech-postnet layers.
pub const SPEECH_DECODER_POSTNET_LAYERS: usize = 5;
/// Kernel width used by every speech-postnet convolution.
pub const SPEECH_DECODER_POSTNET_KERNEL: usize = 5;
/// Width of the external speaker embedding expected by the model.
pub const SPEAKER_EMBEDDING_DIM: usize = 512;
/// Maximum supported input-token sequence length.
pub const MAX_TEXT_POSITIONS: usize = 600;
/// Maximum supported speech-decoder position count.
pub const MAX_SPEECH_POSITIONS: usize = 1_876;
/// Encoder relative-position bucket limit.
pub const ENCODER_MAX_RELATIVE_POSITION: usize = 160;
/// Padding token identifier in the released tokenizer.
pub const PAD_TOKEN_ID: u32 = 1;
/// End-of-sequence token identifier in the released tokenizer.
pub const EOS_TOKEN_ID: u32 = 2;

const LABEL: &str = "SpeechT5-TTS";
const TENSOR_COUNT: usize = 393;
const MANIFEST_SHA256: [u8; 32] = [
    0xfd, 0x6a, 0x13, 0x23, 0xb4, 0x99, 0x47, 0x81, 0xda, 0xf6, 0xb6, 0x57, 0xe6, 0x90, 0xcc, 0xa1,
    0xe7, 0x41, 0xee, 0x2f, 0x78, 0x10, 0xfa, 0xb0, 0x3d, 0x0d, 0x22, 0xbf, 0x62, 0x30, 0x1e, 0x04,
];
const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: EXPECTED_ARCH,
    model_name: MODEL_NAME,
    model_name_alias: None,
    tensor_count: TENSOR_COUNT,
    manifest_sha256: MANIFEST_SHA256,
};

const PREFIX: &str = "vokra.speecht5";
const KEY_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_SOURCE_REVISION: &str = "vokra.speecht5.source_revision";
const KEY_SOURCE_WEIGHT_SHA256: &str = "vokra.speecht5.source_weight_sha256";
const KEY_TENSOR_MANIFEST_SHA256: &str = "vokra.speecht5.tensor_manifest_sha256";
const LEGACY_PUBLIC_METADATA_COUNT: usize = 23;

/// Immutable topology/generation snapshot consumed by the native forward.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechT5Config {
    /// Transformer hidden width.
    pub hidden_size: usize,
    /// Number of encoder layers.
    pub encoder_layers: usize,
    /// Number of decoder layers.
    pub decoder_layers: usize,
    /// Attention heads per encoder layer.
    pub encoder_attention_heads: usize,
    /// Attention heads per decoder layer.
    pub decoder_attention_heads: usize,
    /// Encoder feed-forward intermediate width.
    pub encoder_ffn_dim: usize,
    /// Decoder feed-forward intermediate width.
    pub decoder_ffn_dim: usize,
    /// Text-token vocabulary size.
    pub vocab_size: usize,
    /// Mel bins per output frame.
    pub num_mel_bins: usize,
    /// Mel frames predicted per decoder step.
    pub reduction_factor: usize,
    /// Speech-prenet hidden width.
    pub speech_decoder_prenet_units: usize,
    /// Number of speech-prenet layers.
    pub speech_decoder_prenet_layers: usize,
    /// Dropout probability authenticated for the speech prenet.
    pub speech_decoder_prenet_dropout: f32,
    /// Speech-postnet channel width.
    pub speech_decoder_postnet_units: usize,
    /// Number of speech-postnet layers.
    pub speech_decoder_postnet_layers: usize,
    /// Speech-postnet convolution kernel width.
    pub speech_decoder_postnet_kernel: usize,
    /// Dropout probability authenticated for the speech postnet.
    pub speech_decoder_postnet_dropout: f32,
    /// Required speaker-embedding width.
    pub speaker_embedding_dim: usize,
    /// Maximum text-token positions.
    pub max_text_positions: usize,
    /// Maximum decoder speech positions.
    pub max_speech_positions: usize,
    /// Maximum relative position represented by the encoder bias.
    pub encoder_max_relative_position: usize,
    /// Layer-normalization epsilon.
    pub layer_norm_eps: f32,
    /// Padding token identifier.
    pub pad_token_id: u32,
    /// End-of-sequence token identifier.
    pub eos_token_id: u32,
    /// Multiplier used to cap generated decoder steps from input length.
    pub generation_maxlen_ratio: f32,
    /// Stop-probability threshold used during autoregressive decoding.
    pub generation_stop_threshold: f32,
}

impl SpeechT5Config {
    /// Returns the exact configuration of the pinned public release.
    #[must_use]
    pub fn official() -> Self {
        Self {
            hidden_size: HIDDEN_SIZE,
            encoder_layers: ENCODER_LAYERS,
            decoder_layers: DECODER_LAYERS,
            encoder_attention_heads: ENCODER_ATTENTION_HEADS,
            decoder_attention_heads: DECODER_ATTENTION_HEADS,
            encoder_ffn_dim: ENCODER_FFN_DIM,
            decoder_ffn_dim: DECODER_FFN_DIM,
            vocab_size: VOCAB_SIZE,
            num_mel_bins: NUM_MEL_BINS,
            reduction_factor: REDUCTION_FACTOR,
            speech_decoder_prenet_units: SPEECH_DECODER_PRENET_UNITS,
            speech_decoder_prenet_layers: SPEECH_DECODER_PRENET_LAYERS,
            speech_decoder_prenet_dropout: 0.5,
            speech_decoder_postnet_units: SPEECH_DECODER_POSTNET_UNITS,
            speech_decoder_postnet_layers: SPEECH_DECODER_POSTNET_LAYERS,
            speech_decoder_postnet_kernel: SPEECH_DECODER_POSTNET_KERNEL,
            speech_decoder_postnet_dropout: 0.5,
            speaker_embedding_dim: SPEAKER_EMBEDDING_DIM,
            max_text_positions: MAX_TEXT_POSITIONS,
            max_speech_positions: MAX_SPEECH_POSITIONS,
            encoder_max_relative_position: ENCODER_MAX_RELATIVE_POSITION,
            layer_norm_eps: 1.0e-5,
            pad_token_id: PAD_TOKEN_ID,
            eos_token_id: EOS_TOKEN_ID,
            generation_maxlen_ratio: 20.0,
            generation_stop_threshold: 0.5,
        }
    }

    fn from_gguf(file: &GgufFile) -> Result<Self> {
        required_string(file, KEY_CATEGORY, "tts")?;
        required_string(file, KEY_UPSTREAM_HF, "microsoft/speecht5_tts")?;
        required_string(file, KEY_SOURCE_REVISION, SOURCE_REVISION)?;
        required_string(file, KEY_SOURCE_WEIGHT_SHA256, SOURCE_WEIGHT_SHA256)?;
        required_string(file, KEY_TENSOR_MANIFEST_SHA256, TENSOR_MANIFEST_SHA256)?;
        required_string(file, chunks::KEY_PROVENANCE_LICENSE, "mit")?;

        let config = Self::official();
        for (suffix, expected) in [
            ("hidden_size", config.hidden_size as u32),
            ("encoder_layers", config.encoder_layers as u32),
            ("decoder_layers", config.decoder_layers as u32),
            (
                "encoder_attention_heads",
                config.encoder_attention_heads as u32,
            ),
            (
                "decoder_attention_heads",
                config.decoder_attention_heads as u32,
            ),
            ("encoder_ffn_dim", config.encoder_ffn_dim as u32),
            ("decoder_ffn_dim", config.decoder_ffn_dim as u32),
            ("vocab_size", config.vocab_size as u32),
            ("num_mel_bins", config.num_mel_bins as u32),
            ("reduction_factor", config.reduction_factor as u32),
            (
                "speech_decoder_prenet_units",
                config.speech_decoder_prenet_units as u32,
            ),
            (
                "speech_decoder_prenet_layers",
                config.speech_decoder_prenet_layers as u32,
            ),
            (
                "speech_decoder_postnet_units",
                config.speech_decoder_postnet_units as u32,
            ),
            (
                "speech_decoder_postnet_layers",
                config.speech_decoder_postnet_layers as u32,
            ),
            (
                "speech_decoder_postnet_kernel",
                config.speech_decoder_postnet_kernel as u32,
            ),
            ("speaker_embedding_dim", config.speaker_embedding_dim as u32),
            ("max_text_positions", config.max_text_positions as u32),
            ("max_speech_positions", config.max_speech_positions as u32),
            (
                "encoder_max_relative_position",
                config.encoder_max_relative_position as u32,
            ),
            ("pad_token_id", config.pad_token_id),
            ("eos_token_id", config.eos_token_id),
            ("excluded_batch_norm_counters", 5),
        ] {
            required_u32(file, &format!("{PREFIX}.{suffix}"), expected)?;
        }
        for (suffix, expected) in [
            (
                "speech_decoder_prenet_dropout",
                config.speech_decoder_prenet_dropout,
            ),
            (
                "speech_decoder_postnet_dropout",
                config.speech_decoder_postnet_dropout,
            ),
            ("layer_norm_eps", config.layer_norm_eps),
            ("generation.maxlen_ratio", config.generation_maxlen_ratio),
            (
                "generation.stop_threshold",
                config.generation_stop_threshold,
            ),
        ] {
            required_f32(file, &format!("{PREFIX}.{suffix}"), expected)?;
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.encoder_layers == 0
            || self.decoder_layers == 0
            || self.encoder_attention_heads == 0
            || self.decoder_attention_heads == 0
            || !self
                .hidden_size
                .is_multiple_of(self.encoder_attention_heads)
            || !self
                .hidden_size
                .is_multiple_of(self.decoder_attention_heads)
            || self.encoder_ffn_dim == 0
            || self.decoder_ffn_dim == 0
            || self.vocab_size == 0
            || self.num_mel_bins == 0
            || self.reduction_factor == 0
            || self.speech_decoder_prenet_layers == 0
            || self.speech_decoder_postnet_layers == 0
            || self.speech_decoder_postnet_kernel == 0
            || self.speaker_embedding_dim == 0
            || self.max_text_positions == 0
            || self.max_speech_positions == 0
            || self.encoder_max_relative_position == 0
            || !self.layer_norm_eps.is_finite()
            || self.layer_norm_eps <= 0.0
            || !(0.0..1.0).contains(&self.generation_stop_threshold)
            || !self.generation_maxlen_ratio.is_finite()
            || self.generation_maxlen_ratio <= 0.0
        {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: invalid runtime topology {self:?}"
            )));
        }
        Ok(())
    }
}

/// Cheap handle proving that all SpeechT5 identity, topology and tokenizer
/// gates match the one pinned executable release.
#[derive(Debug, Clone)]
pub struct SpeechT5Checkpoint {
    config: SpeechT5Config,
    tokenizer: SpeechT5Tokenizer,
    weight_license: LicenseClass,
}

impl SpeechT5Checkpoint {
    /// Authenticates and binds a SpeechT5 GGUF checkpoint and tokenizer.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let strict = StrictCheckpoint::bind(file, SPEC)?;
        if strict.weight_license() != LicenseClass::Permissive {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: weight license {:?}, expected permissive MIT",
                strict.weight_license()
            )));
        }
        let modern_contract = file.get(KEY_SOURCE_REVISION).is_some()
            || file.get(tokenizer::KEY_MODEL_SHA256).is_some();
        let (config, tokenizer) = if modern_contract {
            (
                SpeechT5Config::from_gguf(file)?,
                SpeechT5Tokenizer::from_gguf(file)?,
            )
        } else {
            validate_legacy_public_metadata(file)?;
            let config = SpeechT5Config::official();
            config.validate()?;
            (config, SpeechT5Tokenizer::official()?)
        };
        Ok(Self {
            config,
            tokenizer,
            weight_license: strict.weight_license(),
        })
    }

    /// Returns the authenticated model configuration.
    #[must_use]
    pub const fn config(&self) -> &SpeechT5Config {
        &self.config
    }

    /// Returns the embedded text tokenizer.
    #[must_use]
    pub const fn tokenizer(&self) -> &SpeechT5Tokenizer {
        &self.tokenizer
    }

    /// Returns the stamped weight-license class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Returns the exact number of tensors required by the release contract.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        TENSOR_COUNT
    }

    /// Encodes normalized input text with the authenticated tokenizer.
    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text)
    }
}

/// Validates the complete 23-key header of the one historical public GGUF.
///
/// The caller has already passed [`SPEC`], so tensor count, names, shapes,
/// architecture, model name, and weight-license class are exact. Requiring
/// every remaining legacy key plus the exact metadata count leaves no room
/// for an unknown or partially upgraded contract to inherit modern defaults.
fn validate_legacy_public_metadata(file: &GgufFile) -> Result<()> {
    if file.metadata().len() != LEGACY_PUBLIC_METADATA_COUNT {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: tokenizer-less metadata count {} != exact historical public count {LEGACY_PUBLIC_METADATA_COUNT} (vokra/speecht5-tts@{LEGACY_PUBLIC_REVISION}, sha256={LEGACY_PUBLIC_GGUF_SHA256})",
            file.metadata().len()
        )));
    }
    for (key, expected) in [
        (chunks::KEY_MODEL_ARCH, EXPECTED_ARCH),
        (chunks::KEY_MODEL_NAME, MODEL_NAME),
        (KEY_CATEGORY, "tts"),
        (chunks::KEY_PROVENANCE_LICENSE, "mit"),
        (chunks::KEY_PROVENANCE_MODEL_ID, MODEL_NAME),
        (chunks::KEY_PROVENANCE_SOURCE, "microsoft/speecht5_tts"),
        (KEY_UPSTREAM_HF, "microsoft/speecht5_tts"),
        (
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        ),
    ] {
        required_string(file, key, expected)?;
    }
    required_nonempty_string(file, chunks::KEY_SCHEMA_PRODUCER)?;
    required_u32(file, chunks::KEY_SCHEMA_VERSION, 1)?;
    for (suffix, expected) in [
        ("hidden_size", HIDDEN_SIZE as u32),
        ("encoder_layers", ENCODER_LAYERS as u32),
        ("decoder_layers", DECODER_LAYERS as u32),
        ("encoder_attention_heads", ENCODER_ATTENTION_HEADS as u32),
        ("decoder_attention_heads", DECODER_ATTENTION_HEADS as u32),
        ("encoder_ffn_dim", ENCODER_FFN_DIM as u32),
        ("decoder_ffn_dim", DECODER_FFN_DIM as u32),
        ("vocab_size", VOCAB_SIZE as u32),
        ("num_mel_bins", NUM_MEL_BINS as u32),
        ("reduction_factor", REDUCTION_FACTOR as u32),
        ("speaker_embedding_dim", SPEAKER_EMBEDDING_DIM as u32),
        (
            "speech_decoder_postnet_layers",
            SPEECH_DECODER_POSTNET_LAYERS as u32,
        ),
        (
            "speech_decoder_prenet_units",
            SPEECH_DECODER_PRENET_UNITS as u32,
        ),
    ] {
        required_u32(file, &format!("{PREFIX}.{suffix}"), expected)?;
    }
    Ok(())
}

fn required_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| VokraError::ModelLoad(format!("{LABEL}: missing/non-string `{key}`")))?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}`={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn required_nonempty_string(file: &GgufFile, key: &str) -> Result<()> {
    if file
        .get(key)
        .and_then(GgufMetadataValue::as_str)
        .is_none_or(|value| value.trim().is_empty())
    {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: missing/empty `{key}`"
        )));
    }
    Ok(())
}

fn required_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    let actual = match file.get(key) {
        Some(GgufMetadataValue::U32(value)) => *value,
        Some(other) => {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: `{key}` must be u32, found {other:?}"
            )));
        }
        None => return Err(VokraError::ModelLoad(format!("{LABEL}: missing `{key}`"))),
    };
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}`={actual}, expected {expected}"
        )));
    }
    Ok(())
}

fn required_f32(file: &GgufFile, key: &str, expected: f32) -> Result<()> {
    let actual = match file.get(key) {
        Some(GgufMetadataValue::F32(value)) => *value,
        Some(other) => {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: `{key}` must be f32, found {other:?}"
            )));
        }
        None => return Err(VokraError::ModelLoad(format!("{LABEL}: missing `{key}`"))),
    };
    if actual.to_bits() != expected.to_bits() {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}`={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufBuilder, GgufFile};

    fn config_only_file() -> GgufFile {
        let config = SpeechT5Config::official();
        let mut builder = GgufBuilder::new();
        builder.add_string(KEY_CATEGORY, "tts");
        builder.add_string(KEY_UPSTREAM_HF, "microsoft/speecht5_tts");
        builder.add_string(KEY_SOURCE_REVISION, SOURCE_REVISION);
        builder.add_string(KEY_SOURCE_WEIGHT_SHA256, SOURCE_WEIGHT_SHA256);
        builder.add_string(KEY_TENSOR_MANIFEST_SHA256, TENSOR_MANIFEST_SHA256);
        builder.add_string(chunks::KEY_PROVENANCE_LICENSE, "mit");
        for (suffix, value) in [
            ("hidden_size", config.hidden_size as u32),
            ("encoder_layers", config.encoder_layers as u32),
            ("decoder_layers", config.decoder_layers as u32),
            (
                "encoder_attention_heads",
                config.encoder_attention_heads as u32,
            ),
            (
                "decoder_attention_heads",
                config.decoder_attention_heads as u32,
            ),
            ("encoder_ffn_dim", config.encoder_ffn_dim as u32),
            ("decoder_ffn_dim", config.decoder_ffn_dim as u32),
            ("vocab_size", config.vocab_size as u32),
            ("num_mel_bins", config.num_mel_bins as u32),
            ("reduction_factor", config.reduction_factor as u32),
            (
                "speech_decoder_prenet_units",
                config.speech_decoder_prenet_units as u32,
            ),
            (
                "speech_decoder_prenet_layers",
                config.speech_decoder_prenet_layers as u32,
            ),
            (
                "speech_decoder_postnet_units",
                config.speech_decoder_postnet_units as u32,
            ),
            (
                "speech_decoder_postnet_layers",
                config.speech_decoder_postnet_layers as u32,
            ),
            (
                "speech_decoder_postnet_kernel",
                config.speech_decoder_postnet_kernel as u32,
            ),
            ("speaker_embedding_dim", config.speaker_embedding_dim as u32),
            ("max_text_positions", config.max_text_positions as u32),
            ("max_speech_positions", config.max_speech_positions as u32),
            (
                "encoder_max_relative_position",
                config.encoder_max_relative_position as u32,
            ),
            ("pad_token_id", config.pad_token_id),
            ("eos_token_id", config.eos_token_id),
            ("excluded_batch_norm_counters", 5),
        ] {
            builder.add_u32(&format!("{PREFIX}.{suffix}"), value);
        }
        for (suffix, value) in [
            (
                "speech_decoder_prenet_dropout",
                config.speech_decoder_prenet_dropout,
            ),
            (
                "speech_decoder_postnet_dropout",
                config.speech_decoder_postnet_dropout,
            ),
            ("layer_norm_eps", config.layer_norm_eps),
            ("generation.maxlen_ratio", config.generation_maxlen_ratio),
            (
                "generation.stop_threshold",
                config.generation_stop_threshold,
            ),
        ] {
            builder.add_f32(&format!("{PREFIX}.{suffix}"), value);
        }
        GgufFile::parse(builder.to_bytes().unwrap()).unwrap()
    }

    fn legacy_public_metadata_only_file(extra: bool) -> GgufFile {
        let config = SpeechT5Config::official();
        let mut builder = GgufBuilder::new();
        builder.add_string(chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        builder.add_string(chunks::KEY_MODEL_NAME, MODEL_NAME);
        builder.add_string(KEY_CATEGORY, "tts");
        builder.add_string(chunks::KEY_PROVENANCE_LICENSE, "mit");
        builder.add_string(chunks::KEY_PROVENANCE_MODEL_ID, MODEL_NAME);
        builder.add_string(chunks::KEY_PROVENANCE_SOURCE, "microsoft/speecht5_tts");
        builder.add_string(KEY_UPSTREAM_HF, "microsoft/speecht5_tts");
        builder.add_string(
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        );
        for (suffix, value) in [
            ("hidden_size", config.hidden_size as u32),
            ("encoder_layers", config.encoder_layers as u32),
            ("decoder_layers", config.decoder_layers as u32),
            (
                "encoder_attention_heads",
                config.encoder_attention_heads as u32,
            ),
            (
                "decoder_attention_heads",
                config.decoder_attention_heads as u32,
            ),
            ("encoder_ffn_dim", config.encoder_ffn_dim as u32),
            ("decoder_ffn_dim", config.decoder_ffn_dim as u32),
            ("vocab_size", config.vocab_size as u32),
            ("num_mel_bins", config.num_mel_bins as u32),
            ("reduction_factor", config.reduction_factor as u32),
            ("speaker_embedding_dim", config.speaker_embedding_dim as u32),
            (
                "speech_decoder_postnet_layers",
                config.speech_decoder_postnet_layers as u32,
            ),
            (
                "speech_decoder_prenet_units",
                config.speech_decoder_prenet_units as u32,
            ),
        ] {
            builder.add_u32(&format!("{PREFIX}.{suffix}"), value);
        }
        if extra {
            builder.add_string("vokra.speecht5.unknown_future_key", "reject");
        }
        GgufFile::parse(builder.to_bytes().unwrap()).unwrap()
    }

    #[test]
    fn official_config_round_trips_typed_metadata() {
        let file = config_only_file();
        assert_eq!(
            SpeechT5Config::from_gguf(&file).unwrap(),
            SpeechT5Config::official()
        );
    }

    #[test]
    fn missing_axis_fails_closed() {
        let file = GgufFile::parse(GgufBuilder::new().to_bytes().unwrap()).unwrap();
        let error = SpeechT5Config::from_gguf(&file).unwrap_err();
        assert!(error.to_string().contains(KEY_CATEGORY));
    }

    #[test]
    fn exact_legacy_public_metadata_inherits_only_audited_constants() {
        let file = legacy_public_metadata_only_file(false);
        assert_eq!(file.metadata().len(), LEGACY_PUBLIC_METADATA_COUNT);
        validate_legacy_public_metadata(&file).unwrap();
        let tokenizer = SpeechT5Tokenizer::official().unwrap();
        assert_eq!(
            tokenizer.encode("Hello world").unwrap(),
            vec![4, 35, 5, 15, 15, 8, 4, 20, 8, 13, 15, 14, 2]
        );
    }

    #[test]
    fn tokenizerless_contract_with_any_extra_metadata_fails_closed() {
        let file = legacy_public_metadata_only_file(true);
        let error = validate_legacy_public_metadata(&file).unwrap_err();
        assert!(error.to_string().contains("metadata count"));
        assert!(error.to_string().contains(LEGACY_PUBLIC_REVISION));
    }
}
