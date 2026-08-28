//! Native Moonshine Tiny/Base raw-waveform ASR.
//!
//! The implementation follows the pinned Hugging Face Moonshine architecture:
//! three valid Conv1D layers over 16 kHz PCM, pre-norm GELU encoder blocks,
//! pre-norm causal/self + cross-attention SwiGLU decoder blocks, tied output
//! embeddings and greedy BPE decoding.  It deliberately supports CPU only
//! until the composed attention path is backend-routed; selecting another
//! backend is an explicit error, never a silent fallback.

mod forward;
mod tokenizer;
mod weights;

use vokra_core::engines::AsrEngine;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::tasks::Transcription;
use vokra_core::{BackendKind, LicenseClass, Result, VokraError};

use tokenizer::MoonshineTokenizer;
use weights::MoonshineWeights;

/// GGUF architecture discriminator shared by Tiny and Base.
pub const ARCH: &str = "moonshine";
/// Required raw PCM sample rate.
pub const MOONSHINE_SAMPLE_RATE: u32 = 16_000;
/// Canonical Tiny model-name tag.
pub const NAME_TAG_TINY: &str = "moonshine-tiny";
/// Canonical Base model-name tag.
pub const NAME_TAG_BASE: &str = "moonshine-base";

/// Supported official Moonshine release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoonshineVariant {
    /// 27M-parameter, six-layer release.
    Tiny,
    /// 61.5M-parameter, eight-layer release.
    Base,
}

impl MoonshineVariant {
    /// Returns the GGUF model-name tag.
    pub const fn as_name(self) -> &'static str {
        match self {
            Self::Tiny => NAME_TAG_TINY,
            Self::Base => NAME_TAG_BASE,
        }
    }

    /// Returns the canonical Hugging Face repository id.
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Tiny => "moonshine-ai/moonshine-tiny",
            Self::Base => "moonshine-ai/moonshine-base",
        }
    }

    /// Original repository id stamped into the already-published Vokra GGUFs
    /// before the upstream project moved to the `moonshine-ai` organization.
    /// Hugging Face redirects these ids to the canonical repositories; the
    /// loader accepts exactly this one historical alias per variant so the
    /// public artifacts remain runnable without weakening provenance checks
    /// for any unrelated source.
    pub const fn legacy_upstream_hf(self) -> &'static str {
        match self {
            Self::Tiny => "UsefulSensors/moonshine-tiny",
            Self::Base => "UsefulSensors/moonshine-base",
        }
    }

    /// Returns the pinned checkpoint revision.
    pub const fn revision(self) -> &'static str {
        match self {
            Self::Tiny => "390624ed33d594443aa4aa221f5b9f283b545b5a",
            Self::Base => "7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf",
        }
    }

    /// Returns the SHA-256 of the pinned official `model.safetensors`.
    pub const fn checkpoint_sha256(self) -> &'static str {
        match self {
            Self::Tiny => "867cd2215804859c55aa972d740bd5002be149b4e7526328c895d2408848c736",
            Self::Base => "e020c79d0a979a7ec099f718ff1cd2f19e92aead230d69654bca5975a8e1b862",
        }
    }

    /// Returns the SHA-256 of the shared pinned `tokenizer.json`.
    pub const fn tokenizer_sha256(self) -> &'static str {
        "6579793438bc4fbafffacf699169ff53e3769c5a0a0f5e71cdee8853e8130deb"
    }

    /// Returns the official partial-RoPE fraction for this release.
    pub const fn partial_rotary_factor(self) -> f32 {
        match self {
            Self::Tiny => 0.9,
            Self::Base => 0.62,
        }
    }

    /// Parses a canonical GGUF model-name tag.
    pub fn from_name(value: &str) -> Option<Self> {
        match value {
            NAME_TAG_TINY => Some(Self::Tiny),
            NAME_TAG_BASE => Some(Self::Base),
            _ => None,
        }
    }
}

/// Shape and generation axes for an official Moonshine variant.
#[derive(Debug, Clone, PartialEq)]
pub struct MoonshineConfig {
    /// Official release variant.
    pub variant: MoonshineVariant,
    /// Transformer residual width.
    pub hidden_size: usize,
    /// MLP inner width.
    pub intermediate_size: usize,
    /// Encoder block count.
    pub encoder_layers: usize,
    /// Decoder block count.
    pub decoder_layers: usize,
    /// Self/cross-attention head count.
    pub attention_heads: usize,
    /// Per-head prefix width receiving RoPE.
    pub rotary_dim: usize,
    /// RoPE frequency base.
    pub rope_theta: f32,
    /// Maximum decoder sequence length.
    pub max_positions: usize,
    /// Tied decoder vocabulary size.
    pub vocab_size: usize,
    /// First greedy-decoder token.
    pub decoder_start_token_id: u32,
    /// End-of-transcript token.
    pub eos_token_id: u32,
    /// Raw PCM sample rate.
    pub sample_rate: u32,
}

impl MoonshineConfig {
    /// Returns the pinned Tiny topology.
    pub const fn tiny() -> Self {
        Self {
            variant: MoonshineVariant::Tiny,
            hidden_size: 288,
            intermediate_size: 1_152,
            encoder_layers: 6,
            decoder_layers: 6,
            attention_heads: 8,
            rotary_dim: 32,
            rope_theta: 10_000.0,
            max_positions: 194,
            vocab_size: 32_768,
            decoder_start_token_id: 1,
            eos_token_id: 2,
            sample_rate: MOONSHINE_SAMPLE_RATE,
        }
    }

    /// Returns the pinned Base topology.
    pub const fn base() -> Self {
        Self {
            variant: MoonshineVariant::Base,
            hidden_size: 416,
            intermediate_size: 1_664,
            encoder_layers: 8,
            decoder_layers: 8,
            attention_heads: 8,
            rotary_dim: 32,
            rope_theta: 10_000.0,
            max_positions: 194,
            vocab_size: 32_768,
            decoder_start_token_id: 1,
            eos_token_id: 2,
            sample_rate: MOONSHINE_SAMPLE_RATE,
        }
    }

    /// Returns the topology matching `variant`.
    pub const fn for_variant(variant: MoonshineVariant) -> Self {
        match variant {
            MoonshineVariant::Tiny => Self::tiny(),
            MoonshineVariant::Base => Self::base(),
        }
    }

    /// Validates divisibility and non-zero topology invariants.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.encoder_layers == 0
            || self.decoder_layers == 0
            || self.attention_heads == 0
            || self.vocab_size == 0
        {
            return Err(VokraError::ModelLoad(
                "moonshine: zero-valued topology axis".into(),
            ));
        }
        if self.hidden_size % self.attention_heads != 0 {
            return Err(VokraError::ModelLoad(format!(
                "moonshine: hidden size {} is not divisible by {} attention heads",
                self.hidden_size, self.attention_heads
            )));
        }
        let head_dim = self.hidden_size / self.attention_heads;
        if self.rotary_dim == 0 || self.rotary_dim > head_dim || self.rotary_dim % 2 != 0 {
            return Err(VokraError::ModelLoad(format!(
                "moonshine: rotary dim {} is invalid for head dim {head_dim}",
                self.rotary_dim
            )));
        }
        Ok(())
    }
}

/// Bound Moonshine weights, tokenizer and execution backend.
#[derive(Debug, Clone)]
pub struct Moonshine {
    config: MoonshineConfig,
    weights: MoonshineWeights,
    tokenizer: MoonshineTokenizer,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl Moonshine {
    /// Strictly loads topology, all weights, and tokenizer from GGUF.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let arch = required_string(file, chunks::KEY_MODEL_ARCH)?;
        if arch != ARCH {
            return Err(VokraError::ModelLoad(format!(
                "moonshine: GGUF arch is `{arch}`, expected `{ARCH}`"
            )));
        }
        let name = required_string(file, chunks::KEY_MODEL_NAME)?;
        let variant = MoonshineVariant::from_name(name).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "moonshine: model name `{name}` is not `{NAME_TAG_TINY}` or `{NAME_TAG_BASE}`"
            ))
        })?;
        let config = MoonshineConfig::for_variant(variant);
        config.validate()?;
        require_one_of_strings(
            file,
            "vokra.provenance.upstream_hf",
            &[variant.upstream_hf(), variant.legacy_upstream_hf()],
        )?;
        require_string(file, "vokra.moonshine.revision", variant.revision())?;
        require_string(
            file,
            "vokra.moonshine.checkpoint_sha256",
            variant.checkpoint_sha256(),
        )?;
        require_string(
            file,
            "vokra.moonshine.tokenizer_sha256",
            variant.tokenizer_sha256(),
        )?;
        require_u32(file, "vokra.moonshine.sample_rate", config.sample_rate)?;
        require_u32(
            file,
            "vokra.moonshine.hidden_size",
            config.hidden_size as u32,
        )?;
        require_u32(
            file,
            "vokra.moonshine.intermediate_size",
            config.intermediate_size as u32,
        )?;
        require_u32(
            file,
            "vokra.moonshine.encoder_layers",
            config.encoder_layers as u32,
        )?;
        require_u32(
            file,
            "vokra.moonshine.decoder_layers",
            config.decoder_layers as u32,
        )?;
        require_u32(file, "vokra.moonshine.encoder_heads", 8)?;
        require_u32(file, "vokra.moonshine.decoder_heads", 8)?;
        require_u32(file, "vokra.moonshine.vocab_size", config.vocab_size as u32)?;
        require_u32(
            file,
            "vokra.moonshine.max_positions",
            config.max_positions as u32,
        )?;
        require_u32(
            file,
            "vokra.moonshine.decoder_start_token_id",
            config.decoder_start_token_id,
        )?;
        require_u32(file, "vokra.moonshine.eos_token_id", config.eos_token_id)?;
        require_f32(file, "vokra.moonshine.rope_theta", config.rope_theta)?;
        require_f32(
            file,
            "vokra.moonshine.partial_rotary_factor",
            variant.partial_rotary_factor(),
        )?;
        require_string(file, "vokra.moonshine.encoder_activation", "gelu")?;
        require_string(file, "vokra.moonshine.decoder_activation", "silu-swiglu")?;
        let weights = MoonshineWeights::load(file, &config)?;
        let tokenizer = MoonshineTokenizer::from_gguf(file, config.vocab_size)?;
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|value| value.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);
        Ok(Self {
            config,
            weights,
            tokenizer,
            weight_license,
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and strictly loads a Moonshine GGUF from disk.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    #[must_use]
    /// Selects the execution backend. Every Moonshine projection, composed
    /// attention matrix product, softmax, normalization, activation and Conv1D
    /// is dispatched through the model's declared hot-op set; a backend that
    /// cannot cover that complete set fails loudly at inference.
    pub const fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    #[must_use]
    /// Returns the validated topology.
    pub const fn config(&self) -> &MoonshineConfig {
        &self.config
    }

    #[must_use]
    /// Returns the official release variant.
    pub const fn variant(&self) -> MoonshineVariant {
        self.config.variant
    }

    #[must_use]
    /// Returns the provenance weight-license class.
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Transcribes non-empty mono 16 kHz PCM with greedy decoding.
    pub fn transcribe(&self, pcm: &[f32]) -> Result<String> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "moonshine: PCM input is empty".into(),
            ));
        }
        if pcm.len() < 895 {
            return Err(VokraError::InvalidArgument(format!(
                "moonshine: PCM input has {} samples; at least 895 are required by the valid Conv1D stack",
                pcm.len()
            )));
        }
        let ids = forward::generate(&self.weights, &self.config, self.backend, pcm)?;
        self.tokenizer.decode(&ids)
    }
}

impl AsrEngine for Moonshine {
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        Ok(Transcription::new(Moonshine::transcribe(self, pcm)?))
    }

    fn backend(&self) -> BackendKind {
        self.backend
    }
}

fn required_string<'a>(file: &'a GgufFile, key: &str) -> Result<&'a str> {
    file.get(key)
        .and_then(|value| value.as_str())
        .ok_or_else(|| VokraError::ModelLoad(format!("moonshine: missing string metadata `{key}`")))
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = required_string(file, key)?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "moonshine: metadata `{key}` is `{actual}`, expected `{expected}`"
        )));
    }
    Ok(())
}

fn require_one_of_strings(file: &GgufFile, key: &str, expected: &[&str]) -> Result<()> {
    let actual = required_string(file, key)?;
    if !expected.contains(&actual) {
        return Err(VokraError::ModelLoad(format!(
            "moonshine: metadata `{key}` is `{actual}`, expected one of {expected:?}"
        )));
    }
    Ok(())
}

fn require_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(|value| value.as_u64())
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| VokraError::ModelLoad(format!("moonshine: missing u32 metadata `{key}`")))?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "moonshine: metadata `{key}` is {actual}, expected {expected}"
        )));
    }
    Ok(())
}

fn require_f32(file: &GgufFile, key: &str, expected: f32) -> Result<()> {
    let actual = match file.get(key) {
        Some(GgufMetadataValue::F32(value)) => *value,
        _ => {
            return Err(VokraError::ModelLoad(format!(
                "moonshine: missing f32 metadata `{key}`"
            )));
        }
    };
    if actual.to_bits() != expected.to_bits() {
        return Err(VokraError::ModelLoad(format!(
            "moonshine: metadata `{key}` is {actual}, expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json_f32_array(root: &vokra_core::json::JsonValue, key: &str) -> Vec<f32> {
        root.get(key)
            .and_then(|value| value.as_array())
            .unwrap_or_else(|| panic!("missing array `{key}`"))
            .iter()
            .map(|value| match value {
                vokra_core::json::JsonValue::Float(value) => *value as f32,
                vokra_core::json::JsonValue::Int(value) => *value as f32,
                other => panic!("`{key}` contains non-number {other:?}"),
            })
            .collect()
    }

    fn max_abs(left: &[f32], right: &[f32]) -> f32 {
        assert_eq!(left.len(), right.len());
        left.iter()
            .zip(right)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn official_configs_are_well_formed() {
        for config in [MoonshineConfig::tiny(), MoonshineConfig::base()] {
            config.validate().unwrap();
            assert_eq!(config.attention_heads, 8);
            assert_eq!(config.rotary_dim, 32);
            assert_eq!(config.max_positions, 194);
        }
    }

    #[test]
    fn variants_use_canonical_hf_ids() {
        assert_eq!(
            MoonshineVariant::Tiny.upstream_hf(),
            "moonshine-ai/moonshine-tiny"
        );
        assert_eq!(
            MoonshineVariant::Base.upstream_hf(),
            "moonshine-ai/moonshine-base"
        );
        assert_eq!(
            MoonshineVariant::Tiny.legacy_upstream_hf(),
            "UsefulSensors/moonshine-tiny"
        );
        assert_eq!(
            MoonshineVariant::Base.legacy_upstream_hf(),
            "UsefulSensors/moonshine-base"
        );
    }

    /// Environment-gated independent parity against Transformers.  Kept
    /// ignored because the official GGUF is not committed to this repository.
    #[test]
    #[ignore = "set VOKRA_MOONSHINE_GGUF and VOKRA_MOONSHINE_REFERENCE"]
    fn official_tiny_encoder_decoder_logit_parity() {
        let gguf_path = std::env::var("VOKRA_MOONSHINE_GGUF").expect("GGUF path");
        let reference_path = std::env::var("VOKRA_MOONSHINE_REFERENCE").expect("reference path");
        let file = GgufFile::open(gguf_path).unwrap();
        let model = Moonshine::from_gguf(&file).unwrap();
        assert_eq!(model.variant(), MoonshineVariant::Tiny);
        let reference_bytes = std::fs::read(reference_path).unwrap();
        let reference = vokra_core::json::parse(&reference_bytes).unwrap();
        let pcm = json_f32_array(&reference, "pcm");
        let ids = reference
            .get("decoder_ids")
            .and_then(|value| value.as_array())
            .expect("decoder ids")
            .iter()
            .map(|value| value.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let compute =
            crate::compute::Compute::for_backend(BackendKind::Cpu, forward::HOT_OPS).unwrap();
        let (encoder, encoder_rows) =
            forward::encode(&model.weights, &model.config, &compute, &pcm).unwrap();
        let decoder = forward::decode(
            &model.weights,
            &model.config,
            &compute,
            &ids,
            &encoder,
            encoder_rows,
        )
        .unwrap();
        let last = &decoder[(ids.len() - 1) * model.config.hidden_size..];
        let mut logits = vec![0.0; model.config.vocab_size];
        compute
            .gemv_f32(
                model.config.vocab_size,
                model.config.hidden_size,
                &model.weights.embedding,
                last,
                None,
                &mut logits,
            )
            .unwrap();
        let encoder_error = max_abs(&encoder, &json_f32_array(&reference, "encoder"));
        let decoder_error = max_abs(&decoder, &json_f32_array(&reference, "decoder"));
        let logit_error = max_abs(&logits, &json_f32_array(&reference, "last_logits"));
        eprintln!(
            "moonshine parity max_abs: encoder={encoder_error:e} decoder={decoder_error:e} logits={logit_error:e}"
        );
        assert!(encoder_error <= 2e-4, "encoder max_abs={encoder_error:e}");
        assert!(decoder_error <= 2e-4, "decoder max_abs={decoder_error:e}");
        assert!(logit_error <= 5e-4, "logit max_abs={logit_error:e}");

        let expected_ids = reference
            .get("generated_ids")
            .and_then(|value| value.as_array())
            .expect("generated ids")
            .iter()
            .map(|value| value.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        assert_eq!(
            expected_ids.first(),
            Some(&model.config.decoder_start_token_id)
        );
        let generated = forward::generate_with_limit(
            &model.weights,
            &model.config,
            BackendKind::Cpu,
            &pcm,
            expected_ids.len(),
        )
        .unwrap();
        let mut expected_tokens = expected_ids[1..].to_vec();
        if expected_tokens.last() == Some(&model.config.eos_token_id) {
            expected_tokens.pop();
        }
        assert_eq!(generated, expected_tokens);
        let expected_text = reference
            .get("generated_text")
            .and_then(|value| value.as_str())
            .expect("generated text");
        assert_eq!(model.tokenizer.decode(&generated).unwrap(), expected_text);
    }
}
