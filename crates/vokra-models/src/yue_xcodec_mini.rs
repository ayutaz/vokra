//! Native YuE xcodec-mini token decoder for Mac CPU and Metal.
//!
//! The exact public `vokra/yue-xcodec-mini` GGUF is a composite checkpoint:
//! SoundStream/XCodec weights, a HuBERT semantic encoder, and the same 151k
//! Vocos decoder shipped by `m-a-p/YuE-upsampler`.  YuE's released generation
//! path does not run HuBERT while decoding.  It sums the selected residual-VQ
//! rows (`SoundStream::get_embed`) and sends the resulting 1024-channel feature
//! frames through the Vocos head to obtain 44.1 kHz PCM.
//!
//! This module implements that complete token-to-waveform path.  PCM-to-token
//! encoding remains a separate, explicit unsupported operation until the
//! acoustic encoder, HuBERT frontend, and semantic fusion have all been bound;
//! there is no silent CPU or simplified-codec fallback.

use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};
use vokra_ops::{CodebookTable, MimiRvqAttrs, VocosWeights, vocos_decode};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{
    StrictCheckpoint, StrictCheckpointSpec, load_tensor, require_tensor_shape,
};
use crate::vocos::decode_weights_with_compute;
use crate::yue_upsampler::{HOP_LENGTH, attrs as vocos_attrs, load_prefixed_weights};

/// Public runtime architecture tag.
pub const ARCH: &str = "yue_xcodec_mini";
/// Public model identity.
pub const NAME: &str = "yue-xcodec-mini";
/// Model-zoo category recorded by the public artifact.
pub const CATEGORY: &str = "codec";
/// YuE bundle discriminator recorded by the public artifact.
pub const VARIANT: &str = "xcodec_mini";
/// Upstream Hugging Face repository pinned by provenance metadata.
pub const UPSTREAM_HF: &str = "m-a-p/xcodec_mini_infer";
/// Immutable upstream source revision authenticated for conversion.
pub const UPSTREAM_REVISION: &str = "fe781a67815ab47b4a3a5fce1e8d0a692da7e4e5";
/// Revision of the exact currently published Vokra artifact.
pub const PUBLIC_REVISION: &str = "83c14a67ed792a0d5b3b61fff8ae35a04c6da8fa";
/// Byte length of the exact currently published GGUF.
pub const PUBLIC_GGUF_BYTES: u64 = 1_810_001_760;
/// SHA-256 of the exact currently published GGUF.
pub const PUBLIC_GGUF_SHA256: &str =
    "60e21aa5335646080102196454d7ffad5e012467d6f5eb9b776bf07d666b02bc";
/// SHA-256 of the sorted public tensor name/shape manifest.
pub const MANIFEST_SHA256: &str =
    "cc0a5e9a5a6f1cfbd93b1869bbcb70744814bd8c855d173949abbf6b6cc08f15";
/// Tensor count required by the strict public-artifact binder.
pub const TENSOR_COUNT: usize = 2_145;
/// Number of residual-VQ codebooks consumed for every frame.
pub const CODEBOOKS: usize = 12;
/// Number of entries in each residual-VQ codebook.
pub const CODEBOOK_SIZE: usize = 1_024;
/// Width of each reconstructed acoustic feature frame.
pub const FEATURE_DIM: usize = 1_024;
/// Sample rate whose hop geometry defines the released token stream.
pub const TOKEN_SAMPLE_RATE: u32 = 16_000;
/// Number of token-domain samples represented by one frame.
pub const TOKEN_HOP_LENGTH: usize = 320;
/// Fixed released token frame rate.
pub const TOKEN_FRAME_RATE: u32 = 50;
/// Sample rate emitted by the embedded Vocos decoder.
pub const OUTPUT_SAMPLE_RATE: u32 = 44_100;
/// Codec source checkpoint path relative to the pinned upstream repository.
pub const CODEC_CHECKPOINT_FILE: &str = "final_ckpt/ckpt_00360000.pth";
/// Authenticated byte length of the codec source checkpoint.
pub const CODEC_CHECKPOINT_BYTES: u64 = 1_360_444_883;
/// Authenticated SHA-256 of the codec source checkpoint.
pub const CODEC_CHECKPOINT_SHA256: &str =
    "c8c379ea2d3cbde1c8ba1b9717975220e79ba3f556bb161766fd5e4585dcd59c";
/// Semantic source checkpoint path relative to the pinned upstream repository.
pub const SEMANTIC_CHECKPOINT_FILE: &str = "semantic_ckpts/hf_1_325000/pytorch_model.bin";
/// Authenticated byte length of the semantic source checkpoint.
pub const SEMANTIC_CHECKPOINT_BYTES: u64 = 377_555_286;
/// Authenticated SHA-256 of the semantic source checkpoint.
pub const SEMANTIC_CHECKPOINT_SHA256: &str =
    "c5ddbd7fa2468483cb9b2aa53117813471543dd278e65870333a56c54305f527";
/// Vocos source checkpoint path relative to the pinned upstream repository.
pub const DECODER_CHECKPOINT_FILE: &str = "decoders/decoder_151000.pth";
/// Authenticated byte length of the Vocos source checkpoint.
pub const DECODER_CHECKPOINT_BYTES: u64 = 72_610_550;
/// Authenticated SHA-256 of the Vocos source checkpoint.
pub const DECODER_CHECKPOINT_SHA256: &str =
    "8af97a29d3483f9d4a3755992837501bd7d6caa1a69382ed16e64039e0ea0998";

const LABEL: &str = "yue-xcodec-mini";
const KEY_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_VARIANT: &str = "vokra.yue_bundle.variant";
const KEY_UPSTREAM_REVISION: &str = "vokra.yue_xcodec_mini.upstream_revision";
const KEY_CODEC_CHECKPOINT_FILE: &str = "vokra.yue_xcodec_mini.codec_checkpoint_file";
const KEY_CODEC_CHECKPOINT_BYTES: &str = "vokra.yue_xcodec_mini.codec_checkpoint_bytes";
const KEY_CODEC_CHECKPOINT_SHA256: &str = "vokra.yue_xcodec_mini.codec_checkpoint_sha256";
const KEY_SEMANTIC_CHECKPOINT_FILE: &str = "vokra.yue_xcodec_mini.semantic_checkpoint_file";
const KEY_SEMANTIC_CHECKPOINT_BYTES: &str = "vokra.yue_xcodec_mini.semantic_checkpoint_bytes";
const KEY_SEMANTIC_CHECKPOINT_SHA256: &str = "vokra.yue_xcodec_mini.semantic_checkpoint_sha256";
const KEY_DECODER_CHECKPOINT_FILE: &str = "vokra.yue_xcodec_mini.decoder_checkpoint_file";
const KEY_DECODER_CHECKPOINT_BYTES: &str = "vokra.yue_xcodec_mini.decoder_checkpoint_bytes";
const KEY_DECODER_CHECKPOINT_SHA256: &str = "vokra.yue_xcodec_mini.decoder_checkpoint_sha256";
const KEY_SAMPLE_RATE: &str = "vokra.yue_xcodec_mini.sample_rate";
const KEY_HOP_LENGTH: &str = "vokra.yue_xcodec_mini.hop_length";
const KEY_FRAME_RATE: &str = "vokra.yue_xcodec_mini.frame_rate";
const KEY_CODEBOOKS: &str = "vokra.yue_xcodec_mini.codebooks";
const KEY_CODEBOOK_SIZE: &str = "vokra.yue_xcodec_mini.codebook_size";
const KEY_FEATURE_DIM: &str = "vokra.yue_xcodec_mini.feature_dim";
const KEY_OUTPUT_SAMPLE_RATE: &str = "vokra.yue_xcodec_mini.output_sample_rate";

const ADDITIVE_KEYS: &[&str] = &[
    KEY_UPSTREAM_REVISION,
    KEY_CODEC_CHECKPOINT_FILE,
    KEY_CODEC_CHECKPOINT_BYTES,
    KEY_CODEC_CHECKPOINT_SHA256,
    KEY_SEMANTIC_CHECKPOINT_FILE,
    KEY_SEMANTIC_CHECKPOINT_BYTES,
    KEY_SEMANTIC_CHECKPOINT_SHA256,
    KEY_DECODER_CHECKPOINT_FILE,
    KEY_DECODER_CHECKPOINT_BYTES,
    KEY_DECODER_CHECKPOINT_SHA256,
    KEY_SAMPLE_RATE,
    KEY_HOP_LENGTH,
    KEY_FRAME_RATE,
    KEY_CODEBOOKS,
    KEY_CODEBOOK_SIZE,
    KEY_FEATURE_DIM,
    KEY_OUTPUT_SAMPLE_RATE,
];

const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: ARCH,
    model_name: NAME,
    model_name_alias: None,
    tensor_count: TENSOR_COUNT,
    manifest_sha256: [
        0xcc, 0x0a, 0x5e, 0x9a, 0x5a, 0x6f, 0x1c, 0xfb, 0xd9, 0x3b, 0x18, 0x69, 0xbb, 0xcb, 0x70,
        0x74, 0x48, 0x14, 0xbd, 0x8c, 0x85, 0x5d, 0x17, 0x39, 0x49, 0xab, 0xbf, 0x6b, 0x6c, 0xc0,
        0x8f, 0x15,
    ],
};

/// Every trained reduction in the released code-to-44.1-kHz path.
pub const YUE_XCODEC_MINI_HOT_OPS: &[HotOp] = &[
    HotOp::MimiRvq,
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
    HotOp::LayerNorm,
    HotOp::Gelu,
];

/// Output of the released YuE token upsampling route.
#[derive(Debug, Clone, PartialEq)]
pub struct YueXcodecMiniSynthesis {
    /// Mono PCM at [`OUTPUT_SAMPLE_RATE`].
    pub samples: Vec<f32>,
    /// Output sample rate; fixed at 44.1 kHz for this released route.
    pub sample_rate: u32,
    /// Number of input 50 Hz token frames.
    pub frames: usize,
    /// Number of residual-VQ codebooks consumed per frame.
    pub codebooks: usize,
}

/// Strictly bound public YuE xcodec-mini composite checkpoint.
#[derive(Debug, Clone)]
pub struct YueXcodecMini {
    tables: Arc<Vec<CodebookTable>>,
    vocos: Arc<VocosWeights>,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl YueXcodecMini {
    /// Binds the exact current public artifact and its complete 2,145-tensor
    /// manifest.  Optimizer tensors accidentally preserved by the historical
    /// converter remain part of identity but are never decoded or executed.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        require_string(file, chunks::KEY_PROVENANCE_MODEL_ID, NAME)?;
        require_string(file, KEY_CATEGORY, CATEGORY)?;
        require_string(file, KEY_VARIANT, VARIANT)?;
        require_string(file, KEY_UPSTREAM_HF, UPSTREAM_HF)?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "apache-2.0")?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        )?;
        require_nonempty_string(file, chunks::KEY_PROVENANCE_SOURCE)?;
        validate_additive_contract(file)?;
        validate_f32_manifest(file)?;

        let mut tables = Vec::with_capacity(CODEBOOKS);
        for codebook in 0..CODEBOOKS {
            let name = format!("codec.codec_model.quantizer.vq.layers.{codebook}._codebook.embed");
            tables.push(CodebookTable::new(
                CODEBOOK_SIZE,
                FEATURE_DIM,
                load_tensor(file, LABEL, &name, &[CODEBOOK_SIZE, FEATURE_DIM])?,
            )?);
        }
        let vocos = load_prefixed_weights(file, "decoder.", LABEL)?;
        vocos.validate(&vocos_attrs()).map_err(|error| {
            VokraError::ModelLoad(format!(
                "{LABEL}: embedded Vocos validation failed: {error}"
            ))
        })?;
        Ok(Self {
            tables: Arc::new(tables),
            vocos: Arc::new(vocos),
            weight_license: checkpoint.weight_license(),
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and strictly binds a YuE xcodec-mini GGUF from disk.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Binds and preflights every hot operation before any inference starts.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        Compute::for_backend(backend, YUE_XCODEC_MINI_HOT_OPS)?;
        Ok(Self::from_gguf(file)?.with_backend(backend))
    }

    /// Selects the backend used by subsequent feature and waveform decoding.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Returns the selected execution backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Returns the fail-closed weight license class read from the GGUF.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Returns the fixed 50 Hz input token frame rate.
    #[must_use]
    pub const fn token_frame_rate(&self) -> u32 {
        TOKEN_FRAME_RATE
    }

    /// Returns the fixed 44.1 kHz output waveform rate.
    #[must_use]
    pub const fn output_sample_rate(&self) -> u32 {
        OUTPUT_SAMPLE_RATE
    }

    /// Reconstructs the exact channel-major feature tensor produced by the
    /// upstream `SoundStream::get_embed` route.
    pub fn features_from_codes(&self, codes: &[u32], frames: usize) -> Result<Vec<f32>> {
        validate_codes(codes, frames)?;
        let compute = Compute::for_backend(self.backend, YUE_XCODEC_MINI_HOT_OPS)?;
        let attrs = MimiRvqAttrs {
            n_codebooks: CODEBOOKS,
            codebook_size: CODEBOOK_SIZE,
            d_model: FEATURE_DIM,
        };
        let time_major = compute.mimi_rvq_f32(codes, frames, self.tables.as_slice(), &attrs)?;
        Ok(time_to_channel_major(&time_major, frames, FEATURE_DIM))
    }

    /// Decodes frame-major `[frames, 12]` YuE codes to 44.1 kHz mono PCM.
    pub fn decode_codes_44khz(
        &self,
        codes: &[u32],
        frames: usize,
    ) -> Result<YueXcodecMiniSynthesis> {
        let features = self.features_from_codes(codes, frames)?;
        let attrs = vocos_attrs();
        let samples = if self.backend == BackendKind::Cpu {
            vocos_decode(&features, frames, None, &self.vocos, &attrs)?
        } else {
            let compute = Compute::for_backend(self.backend, YUE_XCODEC_MINI_HOT_OPS)?;
            decode_weights_with_compute(&features, frames, &self.vocos, &attrs, &compute)?
        };
        let expected = frames.checked_mul(HOP_LENGTH).ok_or_else(|| {
            VokraError::InvalidArgument(format!("{LABEL}: output extent overflows usize"))
        })?;
        if samples.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: embedded Vocos emitted {} samples for {frames} frames, expected {expected}",
                samples.len()
            )));
        }
        Ok(YueXcodecMiniSynthesis {
            samples,
            sample_rate: OUTPUT_SAMPLE_RATE,
            frames,
            codebooks: CODEBOOKS,
        })
    }

    /// Explicit boundary for the not-yet-landed HuBERT/acoustic encode route.
    pub fn encode_pcm(&self, _samples: &[f32], _sample_rate: u32) -> Result<Vec<u32>> {
        Err(VokraError::UnsupportedOp(
            "yue-xcodec-mini PCM encode is not implemented: it requires the released DAC acoustic encoder, HuBERT-base frontend, RepCodec semantic encoder, fusion projection, and residual-VQ search. The token-to-44.1-kHz CPU/Metal path is available through decode_codes_44khz; Vokra never substitutes a simpler codec or silently falls back to CPU."
                .to_owned(),
        ))
    }
}

fn validate_codes(codes: &[u32], frames: usize) -> Result<()> {
    if frames == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: frames must be positive"
        )));
    }
    let expected = frames.checked_mul(CODEBOOKS).ok_or_else(|| {
        VokraError::InvalidArgument(format!("{LABEL}: frame/codebook extent overflows usize"))
    })?;
    if codes.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: codes length {} != frames * {CODEBOOKS} ({expected})",
            codes.len()
        )));
    }
    if let Some((index, code)) = codes
        .iter()
        .copied()
        .enumerate()
        .find(|(_, code)| (*code as usize) >= CODEBOOK_SIZE)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: codes[{index}]={code} is outside 0..{CODEBOOK_SIZE}"
        )));
    }
    Ok(())
}

fn time_to_channel_major(input: &[f32], frames: usize, channels: usize) -> Vec<f32> {
    debug_assert_eq!(input.len(), frames * channels);
    let mut output = vec![0.0; input.len()];
    for frame in 0..frames {
        for channel in 0..channels {
            output[channel * frames + frame] = input[frame * channels + channel];
        }
    }
    output
}

fn validate_f32_manifest(file: &GgufFile) -> Result<()> {
    if let Some(tensor) = file
        .tensors()
        .iter()
        .find(|tensor| tensor.dtype != GgmlType::F32)
    {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: tensor `{}` is {:?}, but the pinned public manifest is entirely F32",
            tensor.name, tensor.dtype
        )));
    }
    for codebook in 0..CODEBOOKS {
        require_tensor_shape(
            file,
            LABEL,
            &format!("codec.codec_model.quantizer.vq.layers.{codebook}._codebook.embed"),
            &[CODEBOOK_SIZE, FEATURE_DIM],
        )?;
    }
    Ok(())
}

fn validate_additive_contract(file: &GgufFile) -> Result<()> {
    let present = ADDITIVE_KEYS
        .iter()
        .filter(|&&key| file.get(key).is_some())
        .count();
    if present == 0 {
        return Ok(());
    }
    if present != ADDITIVE_KEYS.len() {
        let missing: Vec<_> = ADDITIVE_KEYS
            .iter()
            .copied()
            .filter(|key| file.get(key).is_none())
            .collect();
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: partial `vokra.yue_xcodec_mini.*` contract ({present}/{} keys); missing={missing:?}",
            ADDITIVE_KEYS.len()
        )));
    }
    for (key, expected) in [
        (KEY_UPSTREAM_REVISION, UPSTREAM_REVISION),
        (KEY_CODEC_CHECKPOINT_FILE, CODEC_CHECKPOINT_FILE),
        (KEY_CODEC_CHECKPOINT_SHA256, CODEC_CHECKPOINT_SHA256),
        (KEY_SEMANTIC_CHECKPOINT_FILE, SEMANTIC_CHECKPOINT_FILE),
        (KEY_SEMANTIC_CHECKPOINT_SHA256, SEMANTIC_CHECKPOINT_SHA256),
        (KEY_DECODER_CHECKPOINT_FILE, DECODER_CHECKPOINT_FILE),
        (KEY_DECODER_CHECKPOINT_SHA256, DECODER_CHECKPOINT_SHA256),
    ] {
        require_string(file, key, expected)?;
    }
    for (key, expected) in [
        (KEY_CODEC_CHECKPOINT_BYTES, CODEC_CHECKPOINT_BYTES),
        (KEY_SEMANTIC_CHECKPOINT_BYTES, SEMANTIC_CHECKPOINT_BYTES),
        (KEY_DECODER_CHECKPOINT_BYTES, DECODER_CHECKPOINT_BYTES),
        (KEY_SAMPLE_RATE, u64::from(TOKEN_SAMPLE_RATE)),
        (KEY_HOP_LENGTH, TOKEN_HOP_LENGTH as u64),
        (KEY_FRAME_RATE, u64::from(TOKEN_FRAME_RATE)),
        (KEY_CODEBOOKS, CODEBOOKS as u64),
        (KEY_CODEBOOK_SIZE, CODEBOOK_SIZE as u64),
        (KEY_FEATURE_DIM, FEATURE_DIM as u64),
        (KEY_OUTPUT_SAMPLE_RATE, u64::from(OUTPUT_SAMPLE_RATE)),
    ] {
        require_u64(file, key, expected)?;
    }
    Ok(())
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_str);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: metadata `{key}` = {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn require_nonempty_string(file: &GgufFile, key: &str) -> Result<()> {
    if file
        .get(key)
        .and_then(GgufMetadataValue::as_str)
        .is_none_or(|value| value.trim().is_empty())
    {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: metadata `{key}` is missing or empty"
        )));
    }
    Ok(())
}

fn require_u64(file: &GgufFile, key: &str, expected: u64) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_u64);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: metadata `{key}` = {actual:?}, expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn read_f32(path: &Path) -> Vec<f32> {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|error| panic!("read YuE xcodec fixture {}: {error}", path.display()));
        assert_eq!(bytes.len() % 4, 0, "fixture must be little-endian f32");
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn read_u32(path: &Path) -> Vec<u32> {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|error| panic!("read YuE xcodec fixture {}: {error}", path.display()));
        assert_eq!(bytes.len() % 4, 0, "fixture must be little-endian u32");
        bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn measure(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        assert!(!actual.is_empty(), "{label} must not be empty");
        assert!(
            actual.iter().all(|value| value.is_finite())
                && expected.iter().all(|value| value.is_finite()),
            "{label} contains non-finite values"
        );
        let mut max_abs = 0.0f64;
        let mut sum_abs = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut dot = 0.0f64;
        let mut actual_sq = 0.0f64;
        let mut expected_sq = 0.0f64;
        let mut worst = 0usize;
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let actual = f64::from(actual);
            let expected = f64::from(expected);
            let delta = (actual - expected).abs();
            if delta > max_abs {
                max_abs = delta;
                worst = index;
            }
            sum_abs += delta;
            sum_sq += delta * delta;
            dot += actual * expected;
            actual_sq += actual * actual;
            expected_sq += expected * expected;
        }
        let count = actual.len() as f64;
        assert!(
            actual_sq.is_finite() && actual_sq > 0.0,
            "{label} actual L2 norm must be positive and finite"
        );
        assert!(
            expected_sq.is_finite() && expected_sq > 0.0,
            "{label} expected L2 norm must be positive and finite"
        );
        assert!(max_abs.is_finite() && sum_abs.is_finite() && sum_sq.is_finite());
        let mean_abs = sum_abs / count;
        let rms = (sum_sq / count).sqrt();
        let cosine = dot / (actual_sq.sqrt() * expected_sq.sqrt());
        assert!(
            mean_abs.is_finite() && rms.is_finite() && cosine.is_finite(),
            "{label} produced a non-finite measurement"
        );
        eprintln!(
            "YUE_XCODEC_MINI_MEASUREMENT label={label} values={} max_abs={max_abs:.9e} \
             worst_index={worst} mean_abs={:.9e} rms={:.9e} cosine={cosine:.12}",
            actual.len(),
            mean_abs,
            rms,
        );
    }

    #[test]
    fn measurement_rejects_nonfinite_and_zero_norm_inputs() {
        for (actual, expected) in [
            (vec![f32::NAN, 1.0], vec![1.0, 1.0]),
            (vec![f32::INFINITY, 1.0], vec![1.0, 1.0]),
            (vec![1.0, 1.0], vec![f32::NEG_INFINITY, 1.0]),
            (vec![0.0, 0.0], vec![1.0, 1.0]),
            (vec![1.0, 1.0], vec![0.0, 0.0]),
        ] {
            assert!(
                std::panic::catch_unwind(|| measure("invalid_measurement", &actual, &expected))
                    .is_err(),
                "invalid measurement input reached the measurement marker"
            );
        }
    }

    fn real_case() -> (GgufFile, Vec<u32>, usize, Vec<f32>, Vec<f32>) {
        let gguf = std::env::var_os("VOKRA_YUE_XCODEC_MINI_GGUF")
            .expect("VOKRA_YUE_XCODEC_MINI_GGUF must point at the strict public GGUF");
        let reference = std::env::var_os("VOKRA_YUE_XCODEC_MINI_REFERENCE_DIR")
            .expect("VOKRA_YUE_XCODEC_MINI_REFERENCE_DIR must point at the official dump");
        let reference = std::path::PathBuf::from(reference);
        let manifest = std::fs::read_to_string(reference.join("manifest.json"))
            .expect("read YuE xcodec-mini reference manifest");
        assert!(manifest.contains("\"format\": \"vokra-yue-xcodec-mini-reference-v2\""));
        assert!(manifest.contains("\"pickle_load_policy\": \"weights_only=True_required\""));
        assert!(manifest.contains(&format!("\"upstream_revision\": \"{UPSTREAM_REVISION}\"")));
        let file = GgufFile::open(gguf).expect("open real YuE xcodec-mini GGUF");
        let codes = read_u32(&reference.join("codes.u32le"));
        assert_eq!(codes.len() % CODEBOOKS, 0);
        let frames = codes.len() / CODEBOOKS;
        let features = read_f32(&reference.join("features.f32le"));
        let waveform = read_f32(&reference.join("waveform.f32le"));
        assert_eq!(features.len(), frames * FEATURE_DIM);
        assert_eq!(waveform.len(), frames * HOP_LENGTH);
        (file, codes, frames, features, waveform)
    }

    #[test]
    fn released_geometry_is_320_hop_50_hz_not_640_hop_25_hz() {
        assert_eq!([8usize, 5, 4, 2].into_iter().product::<usize>(), 320);
        assert_eq!(TOKEN_HOP_LENGTH, 320);
        assert_eq!(TOKEN_SAMPLE_RATE as usize / TOKEN_HOP_LENGTH, 50);
        assert_eq!(TOKEN_FRAME_RATE, 50);
        assert_eq!(CODEBOOKS, 12);
    }

    #[test]
    fn time_to_channel_major_transposes_without_reordering_axes() {
        assert_eq!(
            time_to_channel_major(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 2, 3),
            [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
        );
    }

    #[test]
    fn code_contract_is_fail_loud() {
        let short = validate_codes(&[0; CODEBOOKS - 1], 1).unwrap_err();
        assert!(short.to_string().contains("codes length"));
        let range = validate_codes(&[CODEBOOK_SIZE as u32; CODEBOOKS], 1).unwrap_err();
        assert!(range.to_string().contains("outside"));
        let empty = validate_codes(&[], 0).unwrap_err();
        assert!(empty.to_string().contains("positive"));
    }

    #[test]
    fn hot_ops_include_rvq_and_every_embedded_vocos_reduction() {
        assert!(YUE_XCODEC_MINI_HOT_OPS.contains(&HotOp::MimiRvq));
        for op in crate::yue_upsampler::YUE_UPSAMPLER_HOT_OPS {
            assert!(
                YUE_XCODEC_MINI_HOT_OPS.contains(op),
                "missing embedded Vocos op {op:?}"
            );
        }
    }

    #[test]
    fn learned_hot_ops_are_cpu_and_metal_complete() {
        Compute::for_backend(BackendKind::Cpu, YUE_XCODEC_MINI_HOT_OPS)
            .expect("CPU covers every YuE xcodec-mini learned reduction");
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        match Compute::for_backend(BackendKind::Metal, YUE_XCODEC_MINI_HOT_OPS) {
            Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(error) => panic!("YuE xcodec-mini has a Metal coverage gap: {error}"),
        }
    }

    #[test]
    #[ignore = "requires VAST public GGUF and fixed official xcodec/Vocos fixture"]
    fn measure_real_cpu_against_official_xcodec_and_vocos() {
        let (file, codes, frames, expected_features, expected_waveform) = real_case();
        let model = YueXcodecMini::from_gguf_with_backend(&file, BackendKind::Cpu)
            .expect("strict real YuE xcodec-mini CPU bind");
        let features = model
            .features_from_codes(&codes, frames)
            .expect("real CPU RVQ decode");
        measure("cpu_features_vs_official", &features, &expected_features);
        let audio = model
            .decode_codes_44khz(&codes, frames)
            .expect("real CPU waveform decode");
        measure(
            "cpu_waveform_vs_official",
            &audio.samples,
            &expected_waveform,
        );
        eprintln!(
            "YUE_XCODEC_MINI_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED"
        );
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon, public GGUF and fixed official xcodec/Vocos fixture"]
    fn measure_real_metal_against_cpu_and_official_xcodec_and_vocos() {
        vokra_backend_metal::vokra_metal_probe()
            .expect("YuE xcodec-mini Metal validation requires a real Metal device");
        let (file, codes, frames, expected_features, expected_waveform) = real_case();
        let cpu_model = YueXcodecMini::from_gguf_with_backend(&file, BackendKind::Cpu)
            .expect("strict real YuE xcodec-mini CPU bind");
        let metal_model = YueXcodecMini::from_gguf_with_backend(&file, BackendKind::Metal)
            .expect("strict real YuE xcodec-mini Metal bind");
        let cpu_features = cpu_model
            .features_from_codes(&codes, frames)
            .expect("real CPU RVQ decode");
        let metal_features = metal_model
            .features_from_codes(&codes, frames)
            .expect("real Metal RVQ decode");
        measure("metal_features_vs_cpu", &metal_features, &cpu_features);
        measure(
            "metal_features_vs_official",
            &metal_features,
            &expected_features,
        );
        let cpu = cpu_model
            .decode_codes_44khz(&codes, frames)
            .expect("real CPU waveform decode");
        let metal = metal_model
            .decode_codes_44khz(&codes, frames)
            .expect("real Metal waveform decode");
        measure("metal_waveform_vs_cpu", &metal.samples, &cpu.samples);
        measure(
            "metal_waveform_vs_official",
            &metal.samples,
            &expected_waveform,
        );
        eprintln!(
            "YUE_XCODEC_MINI_MEASUREMENT_ONLY backend=metal numeric_bounds=UNSET verdict=MEASURED_NOT_GATED"
        );
    }
}
