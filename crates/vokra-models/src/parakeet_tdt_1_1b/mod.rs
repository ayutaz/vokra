//! Native NVIDIA Parakeet-TDT-1.1B ASR runtime.
//!
//! The immutable `nvidia/parakeet-tdt-1.1b` NeMo release is a 42-layer,
//! 1,024-wide FastConformer with an 80-bin frontend and a two-layer,
//! 640-wide TDT prediction network. The public Vokra GGUF preserves the
//! original 1,667 F32 tensors under their NeMo names. Binding authenticates
//! the complete sorted name/shape manifest before decoding any tensor.
//!
//! The model uses a 1,024-piece SentencePiece Unigram vocabulary followed by
//! a blank at id 1,024. Runtime decoding consumes the pinned plaintext
//! `tokenizer.vocab`; it never loads SentencePiece protobuf or Python code.
//! The original config has no EOS token, so decoding terminates only by the
//! TDT frame pointer and step bound rather than by a fabricated sentinel.
//!
//! CPU and Metal share [`crate::parakeet::PARAKEET_HOT_OPS`]. A requested
//! backend is selected explicitly and unsupported backend operations return
//! an error through the shared compute layer; there is no CPU fallback.

use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{AsrEngine, BackendKind, LicenseClass, Result, Transcription, VokraError};
use vokra_ops::{RnntAttrs, RnntDecoderKind, RnntHypothesis, rnnt_decode};

use crate::parakeet::{ParakeetAsr, ParakeetConfig};
use crate::strict_checkpoint::verify_tensor_manifest;

/// Runtime GGUF architecture tag.
pub const ARCH: &str = "parakeet-tdt-1_1b";
/// Canonical Vokra model name and Hugging Face repository slug.
pub const NAME: &str = "parakeet-tdt-1.1b";
/// Model-zoo category.
pub const CATEGORY: &str = "asr";
/// Metadata key used by the converter for the category.
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
/// Metadata key used by the converter for the upstream repository.
pub const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
/// Immutable upstream repository.
pub const UPSTREAM_HF: &str = "nvidia/parakeet-tdt-1.1b";
/// Primary Hugging Face source retained for compatibility with the original
/// decoder-only binder API.
pub const PRIMARY_SOURCE_HF: &str = "huggingface.co/nvidia/parakeet-tdt-1.1b";
/// Primary NeMo implementation source retained for compatibility.
pub const PRIMARY_SOURCE_CODE: &str = "github.com/NVIDIA/NeMo";
/// Released weight license.
pub const DEFAULT_LICENSE: &str = "cc-by-4.0";
/// Official audio boundary.
pub const PARAKEET_TDT_1_1B_SAMPLE_RATE: u32 = 16_000;
/// Official duration bins in head-output order.
pub const PARAKEET_TDT_1_1B_DURATIONS: [u32; 5] = [0, 1, 2, 3, 4];
/// Backwards-compatible name retained for callers of the earlier decoder-only
/// API. The now-verified 1.1B release uses the same duration bins.
pub const PARAKEET_TDT_0_6B_V3_REFERENCE_DURATIONS: [u32; 5] = PARAKEET_TDT_1_1B_DURATIONS;
/// NeMo greedy decoder's zero-duration emission cap.
pub const NEMO_DEFAULT_MAX_SYMBOLS_PER_STEP: usize = 10;

const LABEL: &str = "Parakeet-TDT-1.1B";
const TENSOR_COUNT: usize = 1667;
const MANIFEST_SHA256: [u8; 32] = [
    0x98, 0x80, 0x16, 0xb3, 0xf7, 0xf7, 0x56, 0x2d, 0x9f, 0xd1, 0xf1, 0x79, 0xb6, 0x78, 0x4c, 0x6f,
    0xe6, 0xd2, 0xfd, 0xf0, 0xac, 0xdb, 0xf3, 0x18, 0x4e, 0x44, 0x28, 0x68, 0x7c, 0xa1, 0x39, 0xf5,
];
const SOURCE_REVISION: &str = "53276c6469d1f17a1352e30c4d11be3d0d7e9575";
const SOURCE_NEMO_SHA256: &str = "9c563d52bdffeacbac0c5b894fdea9be82fea3a6bd8bb8018ff57888e2b5d988";
const TOKENIZER_VOCAB_SHA256: &str =
    "dc8f48909c2d3a0374f45b7478226d26a7de16bbc5334448a8e989f4538384d1";
const KEY_TOKENIZER_VOCAB_SHA256: &str = "vokra.parakeet_tdt_1_1b.tokenizer.vocab_sha256";
const CONFIG_U32: &[(&str, u32)] = &[
    ("vokra.parakeet_tdt_1_1b.sample_rate", 16_000),
    ("vokra.parakeet_tdt_1_1b.frontend.n_fft", 512),
    ("vokra.parakeet_tdt_1_1b.frontend.hop_length", 160),
    ("vokra.parakeet_tdt_1_1b.frontend.win_length", 400),
    ("vokra.parakeet_tdt_1_1b.frontend.n_mels", 80),
    ("vokra.parakeet_tdt_1_1b.encoder.n_layer", 42),
    ("vokra.parakeet_tdt_1_1b.encoder.d_model", 1024),
    ("vokra.parakeet_tdt_1_1b.encoder.n_head", 8),
    ("vokra.parakeet_tdt_1_1b.encoder.n_head_kv", 8),
    ("vokra.parakeet_tdt_1_1b.encoder.ffn_dim", 4096),
    ("vokra.parakeet_tdt_1_1b.encoder.conv_kernel_size", 9),
    ("vokra.parakeet_tdt_1_1b.encoder.subsampling_factor", 8),
    ("vokra.parakeet_tdt_1_1b.encoder.subsampling_kernel", 3),
    ("vokra.parakeet_tdt_1_1b.encoder.subsampling_stride", 2),
    ("vokra.parakeet_tdt_1_1b.encoder.subsampling_channels", 256),
    (
        "vokra.parakeet_tdt_1_1b.encoder.max_position_embeddings",
        5000,
    ),
    ("vokra.parakeet_tdt_1_1b.encoder.use_bias", 1),
    ("vokra.parakeet_tdt_1_1b.encoder.scale_input", 0),
    ("vokra.parakeet_tdt_1_1b.decoder.n_layer", 2),
    ("vokra.parakeet_tdt_1_1b.decoder.d_model", 640),
    ("vokra.parakeet_tdt_1_1b.joint.vocab_size", 1025),
    ("vokra.parakeet_tdt_1_1b.joint.blank_token_id", 1024),
    ("vokra.parakeet_tdt_1_1b.joint.pad_token_id", 1024),
    ("vokra.parakeet_tdt_1_1b.joint.n_durations", 5),
    ("vokra.parakeet_tdt_1_1b.joint.max_symbols_per_step", 10),
];

/// Axes required by the public decoder-only TDT helper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TdtDecodeParams {
    /// Encoder frame count.
    pub num_timesteps: usize,
    /// Vocabulary width excluding the blank.
    pub vocab_size: usize,
    /// Blank id in the inclusive `0..=vocab_size` head.
    pub blank_id: u32,
    /// Duration bins in head-output order.
    pub duration_bins: Vec<u32>,
    /// Maximum zero-duration emissions at one frame.
    pub max_symbols_per_step: usize,
}

impl TdtDecodeParams {
    /// Uses NeMo's blank-at-tail and maximum-symbol defaults.
    #[must_use]
    pub fn nemo_defaults(num_timesteps: usize, vocab_size: usize, duration_bins: Vec<u32>) -> Self {
        Self {
            num_timesteps,
            vocab_size,
            blank_id: vocab_size as u32,
            duration_bins,
            max_symbols_per_step: NEMO_DEFAULT_MAX_SYMBOLS_PER_STEP,
        }
    }

    /// Float count in one materialized joint frame.
    #[must_use]
    pub fn joint_frame_stride(&self) -> usize {
        self.vocab_size + 1 + self.duration_bins.len()
    }

    /// Total expected materialized joint-buffer length.
    #[must_use]
    pub fn expected_joint_len(&self) -> Option<usize> {
        self.num_timesteps.checked_mul(self.joint_frame_stride())
    }
}

/// Cheap authenticated tensor-manifest view retained for diagnostics and API
/// compatibility. Payload decoding is owned by [`ParakeetTdt11b`].
#[derive(Debug, Clone)]
pub struct ParakeetTdt11bWeights {
    tensors: Vec<(String, Vec<usize>)>,
}

impl ParakeetTdt11bWeights {
    /// Authenticates and captures the exact public tensor manifest without
    /// reading the 4.28 GB payload.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        require_identity(file)?;
        verify_tensor_manifest(file, LABEL, TENSOR_COUNT, MANIFEST_SHA256, NAME)?;
        validate_runtime_metadata(file)?;
        let tensors = file
            .tensors()
            .iter()
            .map(|tensor| {
                (
                    tensor.name.clone(),
                    tensor
                        .dimensions
                        .iter()
                        .map(|&dimension| dimension as usize)
                        .collect(),
                )
            })
            .collect();
        Ok(Self { tensors })
    }

    /// Exact number of authenticated tensors.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Tensor names in file order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|(name, _)| name.as_str())
    }

    /// Dimensions of one authenticated tensor.
    #[must_use]
    pub fn tensor_dims(&self, name: &str) -> Option<&[usize]> {
        self.tensors
            .iter()
            .find(|(tensor_name, _)| tensor_name == name)
            .map(|(_, dimensions)| dimensions.as_slice())
    }

    /// Whether the exact tensor is present.
    #[must_use]
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor_dims(name).is_some()
    }

    /// Requires a tensor and reports the missing name verbatim.
    pub fn require_tensor(&self, name: &str) -> Result<&[usize]> {
        self.tensor_dims(name).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "{LABEL}: authenticated manifest unexpectedly lacks `{name}`"
            ))
        })
    }
}

/// Complete native Parakeet-TDT-1.1B ASR engine.
#[derive(Debug, Clone)]
pub struct ParakeetTdt11b {
    inner: ParakeetAsr,
    weights: ParakeetTdt11bWeights,
}

impl ParakeetTdt11b {
    /// Binds a GGUF. Token-id transcription is available immediately; text
    /// transcription requires either embedded tokenizer bytes or the sidecar
    /// constructor below.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        Self::from_gguf_with_tokenizer_vocab(file, None)
    }

    /// Binds a GGUF plus the byte-exact official `tokenizer.vocab` sidecar.
    /// Both the tensor manifest and tokenizer SHA-256 fail closed.
    pub fn from_gguf_with_tokenizer_vocab(
        file: &GgufFile,
        tokenizer_vocab: Option<&[u8]>,
    ) -> Result<Self> {
        let weights = ParakeetTdt11bWeights::from_gguf(file)?;
        let inner = ParakeetAsr::from_tdt_1_1b_gguf(file, tokenizer_vocab)?;
        Ok(Self { inner, weights })
    }

    /// Selects CPU or Metal explicitly.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.inner = self.inner.with_backend(backend);
        self
    }

    /// Selected backend.
    #[must_use]
    pub fn backend(&self) -> BackendKind {
        self.inner.backend()
    }

    /// Verified official configuration.
    #[must_use]
    pub fn config(&self) -> &ParakeetConfig {
        self.inner.config()
    }

    /// Whether a verified tokenizer is available for text rendering.
    #[must_use]
    pub fn has_tokenizer(&self) -> bool {
        self.inner.has_tokenizer()
    }

    /// Stamped weight-license class, fail-closed to `Unknown` when absent.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.inner.weight_license()
    }

    /// Authenticated manifest view.
    #[must_use]
    pub fn weights(&self) -> &ParakeetTdt11bWeights {
        &self.weights
    }

    /// Exact public tensor count.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Runs the log-mel, Conv2D subsampler and 42-layer encoder.
    pub fn encode_pcm(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        self.inner.encode_pcm(pcm)
    }

    /// Runs one prediction-network + TDT joint step.
    pub fn tdt_head_step(&self, encoder_hidden: &[f32], token_id: u32) -> Result<Vec<f32>> {
        self.inner.tdt_head_step(encoder_hidden, token_id)
    }

    /// Transcribes PCM to repeated, non-blank TDT token ids.
    pub fn transcribe(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        self.inner.transcribe(pcm)
    }

    /// Decodes a caller-materialized joint buffer with the shared TDT op.
    pub fn decode_tdt(
        &self,
        joint_logprobs: &[f32],
        params: &TdtDecodeParams,
    ) -> Result<RnntHypothesis> {
        let attrs = RnntAttrs {
            num_timesteps: params.num_timesteps,
            vocab_size: params.vocab_size,
            blank_id: params.blank_id,
            max_symbols_per_step: params.max_symbols_per_step,
            kind: RnntDecoderKind::Tdt {
                duration_bins: params.duration_bins.clone(),
            },
        };
        rnnt_decode(joint_logprobs, &attrs)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "Parakeet-TDT-1.1B: TDT decoder returned no hypothesis".to_owned(),
                )
            })
    }
}

impl AsrEngine for ParakeetTdt11b {
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        <ParakeetAsr as AsrEngine>::transcribe(&self.inner, pcm)
    }

    fn backend(&self) -> BackendKind {
        self.inner.backend()
    }
}

fn validate_runtime_metadata(file: &GgufFile) -> Result<()> {
    const KEY_REVISION: &str = "vokra.parakeet_tdt_1_1b.source_revision";
    const KEY_SOURCE_SHA: &str = "vokra.parakeet_tdt_1_1b.source_nemo_sha256";
    const KEY_PREEMPHASIS: &str = "vokra.parakeet_tdt_1_1b.frontend.preemphasis";
    const KEY_ACTIVATION: &str = "vokra.parakeet_tdt_1_1b.joint.activation";
    const DURATION_PREFIX: &str = "vokra.parakeet_tdt_1_1b.joint.duration.";
    let duration_keys = PARAKEET_TDT_1_1B_DURATIONS
        .iter()
        .enumerate()
        .map(|(index, &duration)| (format!("{DURATION_PREFIX}{index}"), duration))
        .collect::<Vec<_>>();
    let total = CONFIG_U32.len() + duration_keys.len() + 4;
    let present = CONFIG_U32
        .iter()
        .filter(|(key, _)| file.get(key).is_some())
        .count()
        + duration_keys
            .iter()
            .filter(|(key, _)| file.get(key).is_some())
            .count()
        + [
            KEY_REVISION,
            KEY_SOURCE_SHA,
            KEY_PREEMPHASIS,
            KEY_ACTIVATION,
        ]
        .iter()
        .filter(|key| file.get(key).is_some())
        .count();
    if present == 0 {
        // Narrow legacy exception: identity plus the immutable complete
        // manifest above authenticate the already-published artifact.
    } else if present != total {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: partial `vokra.parakeet_tdt_1_1b.*` metadata group ({present}/{total} keys); reconvert from the pinned release"
        )));
    } else {
        for &(key, expected) in CONFIG_U32 {
            require_u32(file, key, expected)?;
        }
        for (key, expected) in &duration_keys {
            require_u32(file, key, *expected)?;
        }
        require_string(file, KEY_REVISION, SOURCE_REVISION)?;
        require_string(file, KEY_SOURCE_SHA, SOURCE_NEMO_SHA256)?;
        require_string(file, KEY_ACTIVATION, "relu")?;
        match file.get(KEY_PREEMPHASIS) {
            Some(GgufMetadataValue::F32(value)) if value.to_bits() == 0.97f32.to_bits() => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "{LABEL}: `{KEY_PREEMPHASIS}` must be f32 0.97, found {other:?}"
                )));
            }
            None => unreachable!("complete metadata count checked above"),
        }
    }

    let tokenizer_blob = file
        .get(crate::parakeet::tokenizer::KEY_SENTENCEPIECE_VOCAB)
        .is_some();
    let tokenizer_hash = file.get(KEY_TOKENIZER_VOCAB_SHA256).is_some();
    match (tokenizer_blob, tokenizer_hash) {
        (false, false) => Ok(()),
        (true, true) => require_string(file, KEY_TOKENIZER_VOCAB_SHA256, TOKENIZER_VOCAB_SHA256),
        _ => Err(VokraError::ModelLoad(format!(
            "{LABEL}: embedded tokenizer metadata must contain both `{}` and `{KEY_TOKENIZER_VOCAB_SHA256}`",
            crate::parakeet::tokenizer::KEY_SENTENCEPIECE_VOCAB
        ))),
    }
}

fn require_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    match file.get(key) {
        Some(GgufMetadataValue::U32(value)) if *value == expected => Ok(()),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}` must be u32 {expected}, found {other:?}"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "{LABEL}: missing required metadata `{key}`"
        ))),
    }
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    match file.get(key) {
        Some(GgufMetadataValue::String(value)) if value == expected => Ok(()),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}` must be string {expected:?}, found {other:?}"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "{LABEL}: missing required metadata `{key}`"
        ))),
    }
}

fn require_identity(file: &GgufFile) -> Result<()> {
    let arch = file
        .get(chunks::KEY_MODEL_ARCH)
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "{LABEL}: missing/non-string `{}`",
                chunks::KEY_MODEL_ARCH
            ))
        })?;
    if arch != ARCH {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: unsupported `{}`={arch:?}; expected {ARCH:?}. Parakeet-TDT-0.6B and Parakeet-CTC use different checkpoint contracts",
            chunks::KEY_MODEL_ARCH
        )));
    }
    let name = file
        .get(chunks::KEY_MODEL_NAME)
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "{LABEL}: missing/non-string `{}`",
                chunks::KEY_MODEL_NAME
            ))
        })?;
    if name != NAME {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: unsupported `{}`={name:?}; expected {NAME:?}",
            chunks::KEY_MODEL_NAME
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufArray, GgufBuilder, GgufValueType};

    fn metadata_file(
        tokenizer_blob: bool,
        tokenizer_hash: Option<&str>,
    ) -> vokra_core::gguf::GgufFile {
        let mut builder = GgufBuilder::new();
        if tokenizer_blob {
            builder.add_metadata(
                crate::parakeet::tokenizer::KEY_SENTENCEPIECE_VOCAB,
                GgufMetadataValue::Array(GgufArray {
                    element_type: GgufValueType::U8,
                    values: vec![GgufMetadataValue::U8(b'x')],
                }),
            );
        }
        if let Some(value) = tokenizer_hash {
            builder.add_string(KEY_TOKENIZER_VOCAB_SHA256, value);
        }
        vokra_core::gguf::GgufFile::parse(builder.to_bytes().expect("serialize metadata"))
            .expect("parse metadata")
    }

    #[test]
    fn official_config_has_no_fabricated_eos() {
        let config = ParakeetConfig::parakeet_tdt_1_1b();
        assert_eq!(config.encoder.n_layer, 42);
        assert_eq!(config.encoder.in_dim, 80);
        assert_eq!(config.decoder.d_model, 640);
        assert_eq!(config.joint.vocab_size, 1025);
        assert_eq!(config.joint.blank_token_id, 1024);
        assert_eq!(config.joint.eos_token_id, None);
        assert_eq!(config.joint.durations, PARAKEET_TDT_1_1B_DURATIONS);
        config.validate_for_forward().expect("official config");
    }

    #[test]
    fn decoder_axes_use_blank_at_tail() {
        let params = TdtDecodeParams::nemo_defaults(3, 1024, PARAKEET_TDT_1_1B_DURATIONS.to_vec());
        assert_eq!(params.blank_id, 1024);
        assert_eq!(params.joint_frame_stride(), 1030);
        assert_eq!(params.expected_joint_len(), Some(3090));
    }

    #[test]
    fn embedded_tokenizer_metadata_is_an_atomic_authenticated_pair() {
        validate_runtime_metadata(&metadata_file(false, None)).expect("legacy metadata");
        validate_runtime_metadata(&metadata_file(true, Some(TOKENIZER_VOCAB_SHA256)))
            .expect("complete tokenizer pair");
        assert!(validate_runtime_metadata(&metadata_file(true, None)).is_err());
        assert!(
            validate_runtime_metadata(&metadata_file(false, Some(TOKENIZER_VOCAB_SHA256))).is_err()
        );
        assert!(validate_runtime_metadata(&metadata_file(true, Some("wrong"))).is_err());
    }
}
