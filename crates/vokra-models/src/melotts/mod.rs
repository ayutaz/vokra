//! Native runtime binding for the five official MyShell MeloTTS releases.
//!
//! The public Vokra GGUFs preserve the upstream MeloTTS tensor names.  This
//! module pins their complete 1,051-entry name/shape ledgers before any tensor
//! payload is decoded.  The English and Chinese artifacts predate a converter
//! metadata correction; those two legacy tuples are accepted only after the
//! complete official manifest and release identity have matched.

use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec, require_tensor_shape};

mod decoder;
mod duration;
mod flow;
mod model;
mod text_encoder;

pub use decoder::{MELOTTS_DECODER_HOT_OPS, MeloDecoder};
pub use duration::{MELOTTS_DURATION_HOT_OPS, MeloDurationModel};
pub use flow::{MELOTTS_FLOW_HOT_OPS, MeloFlowModel};
pub use model::{MeloSynthesisOptions, MeloSynthesisOutput, MeloTts};
pub use text_encoder::{MELOTTS_TEXT_HOT_OPS, MeloTextEncoder, MeloTextFeatures, MeloTextOutput};

const LABEL: &str = "melotts";
const ARCH: &str = "melotts";
const CATEGORY: &str = "tts";
const TENSOR_COUNT: usize = 1_051;

const SAMPLE_RATE: u32 = 44_100;
const N_FFT: u32 = 2_048;
const HOP_LENGTH: u32 = 512;
const N_SPEAKERS_CAPACITY: u32 = 256;
const INTER_CHANNELS: u32 = 192;
const HIDDEN_CHANNELS: u32 = 192;
const FILTER_CHANNELS: u32 = 768;
const N_HEADS: u32 = 2;
const N_LAYERS: u32 = 6;
const N_LAYERS_TRANS_FLOW: u32 = 3;
const GIN_CHANNELS: u32 = 256;
const UPSAMPLE_INITIAL_CHANNEL: u32 = 512;
const UPSAMPLE_TOTAL: u32 = 512;

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_VARIANT: &str = "vokra.melotts.variant";
const KEY_SAMPLE_RATE: &str = "vokra.melotts.sample_rate";
const KEY_N_FFT: &str = "vokra.melotts.n_fft";
const KEY_HOP_LENGTH: &str = "vokra.melotts.hop_length";
const KEY_N_SPEAKERS_CAPACITY: &str = "vokra.melotts.n_speakers_capacity";
const KEY_N_SPEAKERS_ACTIVE: &str = "vokra.melotts.n_speakers_active";
const KEY_INTER_CHANNELS: &str = "vokra.melotts.inter_channels";
const KEY_HIDDEN_CHANNELS: &str = "vokra.melotts.hidden_channels";
const KEY_FILTER_CHANNELS: &str = "vokra.melotts.filter_channels";
const KEY_N_HEADS: &str = "vokra.melotts.n_heads";
const KEY_N_LAYERS: &str = "vokra.melotts.n_layers";
const KEY_N_LAYERS_TRANS_FLOW: &str = "vokra.melotts.n_layers_trans_flow";
const KEY_GIN_CHANNELS: &str = "vokra.melotts.gin_channels";
const KEY_UPSAMPLE_INITIAL_CHANNEL: &str = "vokra.melotts.upsample_initial_channel";
const KEY_UPSAMPLE_TOTAL: &str = "vokra.melotts.upsample_total";
const KEY_N_SYMBOLS: &str = "vokra.melotts.n_symbols";
const KEY_NUM_TONES: &str = "vokra.melotts.num_tones";
const KEY_NUM_LANGUAGES: &str = "vokra.melotts.num_languages";

const COMMON_MANIFEST: [u8; 32] = [
    0x83, 0x88, 0x12, 0x87, 0xfb, 0xc9, 0x8e, 0x65, 0x83, 0xa9, 0x28, 0x48, 0xf5, 0xa7, 0x7c, 0x70,
    0xc1, 0x56, 0xe9, 0x3f, 0x00, 0x1f, 0xf8, 0x2e, 0xbf, 0x34, 0xbe, 0xf2, 0x0a, 0xf4, 0x15, 0x44,
];
const CHINESE_MANIFEST: [u8; 32] = [
    0x41, 0xf1, 0x75, 0x13, 0x4e, 0xd9, 0x44, 0xe9, 0x58, 0xe7, 0x2c, 0xe6, 0x51, 0xc3, 0x4d, 0xc1,
    0x1f, 0xe8, 0xea, 0x59, 0x3f, 0xc5, 0x1b, 0x7b, 0xc5, 0xe9, 0xbf, 0x8f, 0x54, 0x92, 0x5a, 0x09,
];

/// One of the five official language checkpoints published by Vokra.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeloVariant {
    /// `vokra/melotts-english`.
    English,
    /// `vokra/melotts-chinese`.
    Chinese,
    /// `vokra/melotts-korean`.
    Korean,
    /// `vokra/melotts-spanish`.
    Spanish,
    /// `vokra/melotts-japanese`.
    Japanese,
}

impl MeloVariant {
    /// Returns the canonical Vokra model name stamped in the GGUF.
    #[must_use]
    pub const fn model_name(self) -> &'static str {
        match self {
            Self::English => "melotts-english",
            Self::Chinese => "melotts-chinese",
            Self::Korean => "melotts-korean",
            Self::Spanish => "melotts-spanish",
            Self::Japanese => "melotts-japanese",
        }
    }

    /// Returns the short variant tag stamped in the GGUF.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::English => "english",
            Self::Chinese => "chinese",
            Self::Korean => "korean",
            Self::Spanish => "spanish",
            Self::Japanese => "japanese",
        }
    }

    /// Returns the pinned upstream Hugging Face repository.
    #[must_use]
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::English => "myshell-ai/MeloTTS-English",
            Self::Chinese => "myshell-ai/MeloTTS-Chinese",
            Self::Korean => "myshell-ai/MeloTTS-Korean",
            Self::Spanish => "myshell-ai/MeloTTS-Spanish",
            Self::Japanese => "myshell-ai/MeloTTS-Japanese",
        }
    }

    /// Returns the official symbol-vocabulary size.
    #[must_use]
    pub const fn n_symbols(self) -> u32 {
        match self {
            Self::Chinese => 112,
            Self::English | Self::Korean | Self::Spanish | Self::Japanese => 219,
        }
    }

    /// Returns the official tone-vocabulary size.
    #[must_use]
    pub const fn num_tones(self) -> u32 {
        match self {
            Self::Chinese => 11,
            Self::English | Self::Korean | Self::Spanish | Self::Japanese => 16,
        }
    }

    /// Returns the official language-embedding vocabulary size.
    #[must_use]
    pub const fn num_languages(self) -> u32 {
        match self {
            Self::Chinese => 4,
            Self::English | Self::Korean | Self::Spanish | Self::Japanese => 10,
        }
    }

    /// Returns the number of entries exposed by the release's `spk2id` map.
    #[must_use]
    pub const fn n_speakers_active(self) -> u32 {
        match self {
            Self::English => 5,
            Self::Chinese | Self::Korean | Self::Spanish | Self::Japanese => 1,
        }
    }

    const fn manifest_sha256(self) -> [u8; 32] {
        match self {
            Self::Chinese => CHINESE_MANIFEST,
            Self::English | Self::Korean | Self::Spanish | Self::Japanese => COMMON_MANIFEST,
        }
    }

    fn parse(tag: &str) -> Result<Self> {
        match tag {
            "english" => Ok(Self::English),
            "chinese" => Ok(Self::Chinese),
            "korean" => Ok(Self::Korean),
            "spanish" => Ok(Self::Spanish),
            "japanese" => Ok(Self::Japanese),
            _ => Err(VokraError::ModelLoad(format!(
                "{LABEL}: unsupported `{KEY_VARIANT}`={tag:?}"
            ))),
        }
    }

    const fn spec(self) -> StrictCheckpointSpec {
        StrictCheckpointSpec {
            label: LABEL,
            arch: ARCH,
            model_name: self.model_name(),
            model_name_alias: None,
            tensor_count: TENSOR_COUNT,
            manifest_sha256: self.manifest_sha256(),
        }
    }
}

/// Architecture configuration validated against the official release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeloConfig {
    /// Language release selected by the GGUF identity.
    pub variant: MeloVariant,
    /// Output PCM sampling rate.
    pub sample_rate: u32,
    /// Spectral FFT size used during training.
    pub n_fft: u32,
    /// Decoder upsampling factor and PCM hop length.
    pub hop_length: u32,
    /// Speaker-table capacity.
    pub n_speakers_capacity: u32,
    /// Number of active release speaker IDs.
    pub n_speakers_active: u32,
    /// Latent channel count.
    pub inter_channels: u32,
    /// Text hidden channel count.
    pub hidden_channels: u32,
    /// Transformer feed-forward channel count.
    pub filter_channels: u32,
    /// Transformer attention head count.
    pub n_heads: u32,
    /// Text-encoder layer count.
    pub n_layers: u32,
    /// Transformer coupling-flow layer count.
    pub n_layers_trans_flow: u32,
    /// Global speaker-conditioning width.
    pub gin_channels: u32,
    /// HiFi-GAN initial channel count.
    pub upsample_initial_channel: u32,
    /// Product of all HiFi-GAN upsample rates.
    pub upsample_total: u32,
    /// Symbol-vocabulary size.
    pub n_symbols: u32,
    /// Tone-vocabulary size.
    pub num_tones: u32,
    /// Language-embedding vocabulary size.
    pub num_languages: u32,
}

impl MeloConfig {
    const fn official(variant: MeloVariant) -> Self {
        Self {
            variant,
            sample_rate: SAMPLE_RATE,
            n_fft: N_FFT,
            hop_length: HOP_LENGTH,
            n_speakers_capacity: N_SPEAKERS_CAPACITY,
            n_speakers_active: variant.n_speakers_active(),
            inter_channels: INTER_CHANNELS,
            hidden_channels: HIDDEN_CHANNELS,
            filter_channels: FILTER_CHANNELS,
            n_heads: N_HEADS,
            n_layers: N_LAYERS,
            n_layers_trans_flow: N_LAYERS_TRANS_FLOW,
            gin_channels: GIN_CHANNELS,
            upsample_initial_channel: UPSAMPLE_INITIAL_CHANNEL,
            upsample_total: UPSAMPLE_TOTAL,
            n_symbols: variant.n_symbols(),
            num_tones: variant.num_tones(),
            num_languages: variant.num_languages(),
        }
    }
}

/// Strict handle proving that a GGUF is one of the five published releases.
#[derive(Debug, Clone)]
pub struct MeloTtsCheckpoint {
    checkpoint: StrictCheckpoint,
    config: MeloConfig,
    corrected_legacy_metadata: bool,
}

impl MeloTtsCheckpoint {
    /// Validates release identity, provenance, all metadata and all 1,051
    /// tensor names/shapes without decoding tensor payloads.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let variant = MeloVariant::parse(required_string(file, KEY_VARIANT)?)?;
        let checkpoint = StrictCheckpoint::bind(file, variant.spec())?;
        require_string(file, KEY_MODEL_CATEGORY, CATEGORY)?;
        require_string(file, KEY_UPSTREAM_HF, variant.upstream_hf())?;
        if checkpoint.weight_license() != LicenseClass::Permissive {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: `{}` must be `permissive` for the official MIT checkpoint, got {:?}",
                chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
                checkpoint.weight_license()
            )));
        }

        for (key, expected) in [
            (KEY_SAMPLE_RATE, SAMPLE_RATE),
            (KEY_N_FFT, N_FFT),
            (KEY_HOP_LENGTH, HOP_LENGTH),
            (KEY_N_SPEAKERS_CAPACITY, N_SPEAKERS_CAPACITY),
            (KEY_N_SPEAKERS_ACTIVE, variant.n_speakers_active()),
            (KEY_INTER_CHANNELS, INTER_CHANNELS),
            (KEY_HIDDEN_CHANNELS, HIDDEN_CHANNELS),
            (KEY_FILTER_CHANNELS, FILTER_CHANNELS),
            (KEY_N_HEADS, N_HEADS),
            (KEY_N_LAYERS, N_LAYERS),
            (KEY_N_LAYERS_TRANS_FLOW, N_LAYERS_TRANS_FLOW),
            (KEY_GIN_CHANNELS, GIN_CHANNELS),
            (KEY_UPSAMPLE_INITIAL_CHANNEL, UPSAMPLE_INITIAL_CHANNEL),
            (KEY_UPSAMPLE_TOTAL, UPSAMPLE_TOTAL),
        ] {
            require_u32(file, key, expected)?;
        }

        let stamped_axes = (
            required_u32(file, KEY_N_SYMBOLS)?,
            required_u32(file, KEY_NUM_TONES)?,
            required_u32(file, KEY_NUM_LANGUAGES)?,
        );
        let official_axes = (
            variant.n_symbols(),
            variant.num_tones(),
            variant.num_languages(),
        );
        let legacy_axes = match variant {
            MeloVariant::English => Some((178, 0, 1)),
            MeloVariant::Chinese => Some((112, 11, 1)),
            MeloVariant::Korean | MeloVariant::Spanish | MeloVariant::Japanese => None,
        };
        let corrected_legacy_metadata = stamped_axes != official_axes;
        if corrected_legacy_metadata && Some(stamped_axes) != legacy_axes {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: release axes {stamped_axes:?} are neither official {official_axes:?} nor the pinned legacy tuple {legacy_axes:?} for {:?}",
                variant
            )));
        }

        require_tensor_shape(
            file,
            LABEL,
            "enc_p.emb.weight",
            &[variant.n_symbols() as usize, HIDDEN_CHANNELS as usize],
        )?;
        require_tensor_shape(
            file,
            LABEL,
            "enc_p.tone_emb.weight",
            &[variant.num_tones() as usize, HIDDEN_CHANNELS as usize],
        )?;
        require_tensor_shape(
            file,
            LABEL,
            "enc_p.language_emb.weight",
            &[variant.num_languages() as usize, HIDDEN_CHANNELS as usize],
        )?;
        require_tensor_shape(
            file,
            LABEL,
            "emb_g.weight",
            &[N_SPEAKERS_CAPACITY as usize, GIN_CHANNELS as usize],
        )?;
        require_tensor_shape(
            file,
            LABEL,
            "dec.conv_pre.weight",
            &[
                UPSAMPLE_INITIAL_CHANNEL as usize,
                INTER_CHANNELS as usize,
                7,
            ],
        )?;

        Ok(Self {
            checkpoint,
            config: MeloConfig::official(variant),
            corrected_legacy_metadata,
        })
    }

    /// Returns the official configuration, including corrected language axes.
    #[must_use]
    pub const fn config(&self) -> MeloConfig {
        self.config
    }

    /// Returns the selected language release.
    #[must_use]
    pub const fn variant(&self) -> MeloVariant {
        self.config.variant
    }

    /// Reports whether the exact known public legacy axis tuple was corrected.
    #[must_use]
    pub const fn corrected_legacy_metadata(&self) -> bool {
        self.corrected_legacy_metadata
    }

    /// Returns the pinned model name.
    #[must_use]
    pub fn model_name(&self) -> &str {
        self.checkpoint.model_name()
    }

    /// Returns the fail-closed stamped weight-license class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.checkpoint.weight_license()
    }

    /// Returns the complete manifest tensor count.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        self.checkpoint.tensor_count()
    }

    /// Decodes the real embedding, BERT projection and six-layer relative
    /// Transformer tensors into the native CPU/Metal text encoder.
    pub fn load_text_encoder(&self, file: &GgufFile) -> Result<MeloTextEncoder> {
        MeloTextEncoder::from_gguf(file, self.config)
    }

    /// Decodes the stochastic and deterministic duration predictors.
    pub fn load_duration_model(&self, file: &GgufFile) -> Result<MeloDurationModel> {
        MeloDurationModel::from_gguf(file)
    }

    /// Decodes the four VITS2 Transformer coupling-flow blocks.
    pub fn load_flow_model(&self, file: &GgufFile) -> Result<MeloFlowModel> {
        MeloFlowModel::from_gguf(file)
    }

    /// Decodes and folds the speaker-conditioned HiFi-GAN generator.
    pub fn load_decoder(&self, file: &GgufFile) -> Result<MeloDecoder> {
        MeloDecoder::from_gguf(file)
    }

    /// Loads the complete low-level acoustic synthesis stack.
    pub fn load_model(&self, file: &GgufFile) -> Result<MeloTts> {
        MeloTts::from_checkpoint(self, file)
    }
}

fn required_string<'a>(file: &'a GgufFile, key: &str) -> Result<&'a str> {
    file.get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| VokraError::ModelLoad(format!("{LABEL}: missing/non-string `{key}`")))
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = required_string(file, key)?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}`={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn required_u32(file: &GgufFile, key: &str) -> Result<u32> {
    let actual = file
        .get(key)
        .and_then(GgufMetadataValue::as_u64)
        .ok_or_else(|| VokraError::ModelLoad(format!("{LABEL}: missing/non-u32 `{key}`")))?;
    u32::try_from(actual)
        .map_err(|_| VokraError::ModelLoad(format!("{LABEL}: `{key}`={actual} exceeds u32")))
}

fn require_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    let actual = required_u32(file, key)?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}`={actual}, expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use vokra_core::backend::BackendKind;
    use vokra_core::rng::GaussianSplitMix64;

    #[test]
    fn official_variant_axes_match_released_tensor_tables() {
        for (variant, symbols, tones, languages, speakers) in [
            (MeloVariant::English, 219, 16, 10, 5),
            (MeloVariant::Chinese, 112, 11, 4, 1),
            (MeloVariant::Korean, 219, 16, 10, 1),
            (MeloVariant::Spanish, 219, 16, 10, 1),
            (MeloVariant::Japanese, 219, 16, 10, 1),
        ] {
            let config = MeloConfig::official(variant);
            assert_eq!(config.n_symbols, symbols);
            assert_eq!(config.num_tones, tones);
            assert_eq!(config.num_languages, languages);
            assert_eq!(config.n_speakers_active, speakers);
            assert_eq!(config.upsample_total, config.hop_length);
        }
    }

    #[test]
    fn variant_identity_is_one_to_one() {
        for variant in [
            MeloVariant::English,
            MeloVariant::Chinese,
            MeloVariant::Korean,
            MeloVariant::Spanish,
            MeloVariant::Japanese,
        ] {
            assert_eq!(MeloVariant::parse(variant.tag()).unwrap(), variant);
            assert!(variant.model_name().starts_with("melotts-"));
            assert!(variant.upstream_hf().starts_with("myshell-ai/MeloTTS-"));
        }
        assert!(MeloVariant::parse("unknown").is_err());
    }

    #[test]
    fn chinese_has_a_distinct_manifest_shape_contract() {
        assert_ne!(CHINESE_MANIFEST, COMMON_MANIFEST);
        assert_eq!(MeloVariant::English.manifest_sha256(), COMMON_MANIFEST);
        assert_eq!(MeloVariant::Chinese.manifest_sha256(), CHINESE_MANIFEST);
    }

    #[test]
    #[ignore = "requires VOKRA_MELOTTS_GGUF pointing to a released full GGUF"]
    fn released_gguf_binds_loads_and_runs_native_core() {
        let path = std::env::var("VOKRA_MELOTTS_GGUF").expect("VOKRA_MELOTTS_GGUF");
        let file = GgufFile::open(path).expect("open released MeloTTS GGUF");
        let checkpoint = MeloTtsCheckpoint::from_gguf(&file).expect("strict bind");
        let encoder = checkpoint
            .load_text_encoder(&file)
            .expect("load enc_p tensors");
        let bert = vec![0.0; 1_024];
        let ja_bert = vec![0.0; 768];
        let output = encoder
            .encode(
                MeloTextFeatures {
                    phoneme_ids: &[0],
                    tones: &[0],
                    language_ids: &[0],
                    bert: &bert,
                    ja_bert: &ja_bert,
                    speaker_id: 0,
                },
                BackendKind::Cpu,
            )
            .expect("real text forward");
        assert_eq!(output.sequence_len, 1);
        assert_eq!(output.hidden.len(), HIDDEN_CHANNELS as usize);
        assert_eq!(output.mean.len(), INTER_CHANNELS as usize);
        assert_eq!(output.log_scale.len(), INTER_CHANNELS as usize);
        assert!(output.hidden.iter().all(|value| value.is_finite()));
        assert!(output.mean.iter().all(|value| value.is_finite()));
        assert!(output.log_scale.iter().all(|value| value.is_finite()));

        let duration = checkpoint
            .load_duration_model(&file)
            .expect("load duration tensors");
        let mut rng = GaussianSplitMix64::new(17);
        let durations = duration
            .predict(
                &output.hidden,
                output.sequence_len,
                &output.speaker_conditioning,
                0.0,
                0.0,
                1.0,
                &mut rng,
                BackendKind::Cpu,
            )
            .expect("real deterministic duration forward");
        assert_eq!(durations.len(), output.sequence_len);
        assert!(durations.iter().all(|duration| *duration > 0));

        let flow = checkpoint
            .load_flow_model(&file)
            .expect("load flow tensors");
        let latent = vec![0.0; INTER_CHANNELS as usize];
        let decoder_latent = flow
            .inverse(&latent, 1, &output.speaker_conditioning, BackendKind::Cpu)
            .expect("real latent flow forward");
        assert_eq!(decoder_latent.len(), INTER_CHANNELS as usize);
        assert!(decoder_latent.iter().all(|value| value.is_finite()));

        let decoder = checkpoint
            .load_decoder(&file)
            .expect("load decoder tensors");
        let pcm = decoder
            .decode(
                &decoder_latent,
                1,
                &output.speaker_conditioning,
                BackendKind::Cpu,
            )
            .expect("real HiFi-GAN decoder forward");
        assert_eq!(pcm.len(), HOP_LENGTH as usize);
        assert!(pcm.iter().all(|value| value.is_finite()));
        assert!(pcm.iter().all(|value| (-1.0..=1.0).contains(value)));

        drop(encoder);
        drop(duration);
        drop(flow);
        drop(decoder);

        let model = checkpoint.load_model(&file).expect("load complete model");
        let mut rng = GaussianSplitMix64::new(17);
        let synthesis = model
            .synthesize(
                MeloTextFeatures {
                    phoneme_ids: &[0],
                    tones: &[0],
                    language_ids: &[0],
                    bert: &bert,
                    ja_bert: &ja_bert,
                    speaker_id: 0,
                },
                MeloSynthesisOptions {
                    sdp_ratio: 0.0,
                    noise_scale: 0.0,
                    noise_scale_w: 0.0,
                    length_scale: 0.1,
                    max_frames: 512,
                },
                &mut rng,
                BackendKind::Cpu,
            )
            .expect("real end-to-end acoustic forward");
        assert_eq!(synthesis.sample_rate, SAMPLE_RATE);
        assert_eq!(
            synthesis.pcm.len(),
            synthesis.frame_count * HOP_LENGTH as usize
        );
        assert!(synthesis.pcm.iter().all(|value| value.is_finite()));
    }
}
