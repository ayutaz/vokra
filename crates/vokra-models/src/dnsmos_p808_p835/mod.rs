//! Native Microsoft DNSMOS P.808 + P.835 quality scoring.
//!
//! This module executes the exact two audited ONNX graphs without an ONNX
//! runtime. P.808 uses the official librosa-compatible 321-point log-mel
//! frontend and five-layer CNN; P.835 uses its released learned STFT kernels
//! and seven-layer CNN. Both heads end in their exact dense stacks, and P.835
//! applies the non-personalized polynomial calibration from
//! `dnsmos_local.py` per 9.01-second window before averaging.
//!
//! Every convolution is tiled into the selected backend's GEMM and every
//! dense layer uses the same GEMM route. Mel/STFT construction, activations,
//! pooling, chunking, and calibration are deterministic host DSP/glue. Backend
//! coverage is preflighted before PCM is processed; Metal never falls back to
//! CPU inference.

mod nn;
#[cfg(test)]
mod tests;
mod weights;

use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_core::engines::{MosScore, MosScorerEngine};
use vokra_core::gguf::{GgufFile, GgufMetadataValue, GgufValueType, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

pub use self::weights::DnsmosWeights;

/// GGUF architecture tag for the DNSMOS bundle.
pub const ARCH: &str = "dnsmos";
/// Canonical public model identifier.
pub const NAME: &str = "dnsmos-p808-p835";
/// Model catalogue category.
pub const CATEGORY: &str = "eval";
/// Audited Microsoft source directory.
pub const UPSTREAM_URL: &str = "https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS";
const SOURCE_REVISION: &str = "591184a9fcb2cbdec02520fed81a32bbbf9d73ff";
const P808_ONNX_SHA256: &str = "9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91";
const P835_ONNX_SHA256: &str = "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd";
const SOURCE_PY_SHA256: &str = "1ab566afe006daab32ac7073296a5d0ef99f8b82f91c7266f3ccf26113d7a28b";
const SOURCE_LICENSE_SHA256: &str =
    "d6239afa918961b465b07bf7411cbe34ff6685854f58553db7966f4881a0211f";
const PUBLIC_HF: &str = "vokra/dnsmos-p808-p835";
const PUBLIC_REVISION: &str = "39293917b4fccf66b149c0734140427f29f5ff84";
const PUBLIC_GGUF_SHA256: &str = "b13c264f26a83b92d27f4385332e69e426f3301d2e48de7732c2aa9355650b2d";
const MANIFEST_SHA256: &str = "d6d13fd5191d399736c8c1558d9dbbc51718a377190836a640a1992dbf404847";

/// Number of learned tensors in the complete public bundle.
pub const TENSOR_COUNT: usize = 38;
/// Required input sample rate in hertz.
pub const SAMPLE_RATE: u32 = 16_000;
/// Backward-compatible name from the original binder.
pub const EXPECTED_SAMPLE_RATE: u32 = SAMPLE_RATE;
/// Samples in one 9.01-second DNSMOS window.
pub const INPUT_LENGTH_SAMPLES: usize = 144_160;
/// P.808 log-mel frame count per window.
pub const P808_FRAMES: usize = 900;
/// P.808 odd FFT size used by the official librosa frontend.
pub const P808_N_FFT: usize = 321;
/// P.808 frontend hop in samples.
pub const P808_HOP: usize = 160;
/// P.808 mel-band count.
pub const P808_N_MELS: usize = 120;
pub(super) const P808_CNN_CHANNELS: usize = 64;
/// P.835 learned-STFT frame count per window.
pub const P835_FRAMES: usize = 900;
/// P.835 learned-STFT window in samples.
pub const P835_WINDOW: usize = 320;
/// P.835 learned-STFT hop in samples.
pub const P835_HOP: usize = 160;
/// P.835 one-sided learned-STFT bin count.
pub const P835_BINS: usize = 161;

/// GGUF key carrying the model category.
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";
/// GGUF key carrying the ordered P.808/P.835 bundle inventory.
pub const KEY_DNSMOS_BUNDLE: &str = "vokra.dnsmos.bundle";
/// GGUF key carrying the required input sample rate.
pub const KEY_DNSMOS_SAMPLE_RATE: &str = "vokra.dnsmos.sample_rate";
/// GGUF key carrying the P.808 upstream checkpoint filename.
pub const KEY_DNSMOS_P808_CKPT: &str = "vokra.dnsmos.p808.checkpoint";
/// GGUF key carrying the P.835 upstream checkpoint filename.
pub const KEY_DNSMOS_P835_CKPT: &str = "vokra.dnsmos.p835.checkpoint";
/// Legacy reserved key from the original loud-partial binder.
///
/// The native runtime audits the immutable ONNX graph directly and does not
/// require a separately tokenized topology array, but the public constant is
/// retained for source compatibility.
pub const KEY_DNSMOS_P808_TOPOLOGY: &str = "vokra.dnsmos.p808.topology";
/// P.835 counterpart to [`KEY_DNSMOS_P808_TOPOLOGY`], retained for source
/// compatibility with callers that inspected the former reserved schema.
pub const KEY_DNSMOS_P835_TOPOLOGY: &str = "vokra.dnsmos.p835.topology";
const KEY_SOURCE_REVISION: &str = "vokra.dnsmos.source_revision";
const KEY_P808_ONNX_SHA256: &str = "vokra.dnsmos.p808.onnx_sha256";
const KEY_P835_ONNX_SHA256: &str = "vokra.dnsmos.p835.onnx_sha256";
const KEY_SOURCE_PY_SHA256: &str = "vokra.dnsmos.source_py_sha256";
const KEY_SOURCE_LICENSE_SHA256: &str = "vokra.dnsmos.source_license_sha256";
const KEY_PUBLIC_HF: &str = "vokra.dnsmos.public_hf";
const KEY_PUBLIC_REVISION: &str = "vokra.dnsmos.public_revision";
const KEY_PUBLIC_GGUF_SHA256: &str = "vokra.dnsmos.public_gguf_sha256";
const KEY_MANIFEST_SHA256: &str = "vokra.dnsmos.manifest_sha256";
const KEY_INPUT_LENGTH: &str = "vokra.dnsmos.input_length";
const KEY_P808_FRAMES: &str = "vokra.dnsmos.p808.frames";
const KEY_P808_N_FFT: &str = "vokra.dnsmos.p808.n_fft";
const KEY_P808_HOP: &str = "vokra.dnsmos.p808.hop";
const KEY_P808_N_MELS: &str = "vokra.dnsmos.p808.n_mels";
const KEY_P835_FRAMES: &str = "vokra.dnsmos.p835.frames";
const KEY_P835_WINDOW: &str = "vokra.dnsmos.p835.window";
const KEY_P835_HOP: &str = "vokra.dnsmos.p835.hop";
const KEY_P835_BINS: &str = "vokra.dnsmos.p835.bins";

pub(super) const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: "dnsmos",
    arch: ARCH,
    model_name: NAME,
    model_name_alias: None,
    tensor_count: TENSOR_COUNT,
    manifest_sha256: [
        0xd6, 0xd1, 0x3f, 0xd5, 0x19, 0x1d, 0x39, 0x97, 0x36, 0xc8, 0xc1, 0x55, 0x8d, 0x9d, 0xbb,
        0xc5, 0x17, 0x18, 0xa3, 0x77, 0x19, 0x08, 0x36, 0xa6, 0x40, 0xa1, 0x99, 0x2d, 0xbf, 0x40,
        0x48, 0x47,
    ],
};

/// Complete learned reduction set for CPU and Apple Metal.
pub const DNSMOS_HOT_OPS: &[HotOp] = &[HotOp::Gemm];

/// One released predictor inside the DNSMOS bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DnsmosSubmodel {
    /// P.808 overall MOS predictor.
    P808,
    /// P.835 SIG/BAK/OVRL predictor.
    P835,
}

impl DnsmosSubmodel {
    /// Returns the canonical metadata/logging name.
    pub const fn short(&self) -> &'static str {
        match self {
            Self::P808 => "p808",
            Self::P835 => "p835",
        }
    }

    /// Returns the strict GGUF tensor prefix.
    pub const fn tensor_prefix(&self) -> &'static str {
        match self {
            Self::P808 => "p808.",
            Self::P835 => "p835.",
        }
    }
}

/// Parsed public DNSMOS metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DnsmosConfig {
    /// Ordered variant inventory.
    pub bundle: Vec<String>,
    /// Required PCM sample rate in hertz.
    pub sample_rate: u32,
    /// Whether the inventory contains P.808.
    pub has_p808: bool,
    /// Whether the inventory contains P.835.
    pub has_p835: bool,
}

impl DnsmosConfig {
    /// Validates the public configuration surface used by the original
    /// binder. Partial inventories remain representable here for source
    /// compatibility; [`Dnsmos::from_gguf`] separately requires the complete
    /// audited public bundle before executing weights.
    pub fn validate(&self) -> Result<()> {
        if self.bundle.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "dnsmos: `{KEY_DNSMOS_BUNDLE}` is empty — expected at least one of `p808` / `p835`"
            )));
        }
        if self.sample_rate != SAMPLE_RATE {
            return Err(VokraError::ModelLoad(format!(
                "dnsmos: `{KEY_DNSMOS_SAMPLE_RATE}` = {}, expected {SAMPLE_RATE}",
                self.sample_rate
            )));
        }
        for variant in &self.bundle {
            if variant != "p808" && variant != "p835" {
                return Err(VokraError::ModelLoad(format!(
                    "dnsmos: `{KEY_DNSMOS_BUNDLE}` contains unknown variant `{variant}`"
                )));
            }
        }
        Ok(())
    }

    /// Reads and validates the original public `vokra.dnsmos.*` config group.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let bundle = read_string_array(file, KEY_DNSMOS_BUNDLE)?;
        let sample_rate = required_u64(file, KEY_DNSMOS_SAMPLE_RATE)?;
        let sample_rate = u32::try_from(sample_rate).map_err(|_| {
            VokraError::ModelLoad(format!(
                "dnsmos: `{KEY_DNSMOS_SAMPLE_RATE}`={sample_rate} does not fit in u32"
            ))
        })?;
        let cfg = Self {
            has_p808: bundle.iter().any(|variant| variant == "p808"),
            has_p835: bundle.iter().any(|variant| variant == "p835"),
            bundle,
            sample_rate,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

/// Diagnostic tensor inventory for one submodel.
#[derive(Debug, Clone)]
pub struct DnsmosBundle {
    /// Predictor represented by this entry.
    pub variant: DnsmosSubmodel,
    /// Exact tensor names under the predictor prefix.
    pub tensor_names: Vec<String>,
}

/// Strict native DNSMOS P.808 + P.835 scorer.
#[derive(Debug, Clone)]
pub struct Dnsmos {
    cfg: DnsmosConfig,
    bundles: Arc<Vec<DnsmosBundle>>,
    weights: Arc<DnsmosWeights>,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl Dnsmos {
    /// Strictly binds the exact public 38-tensor release.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        require_string(file, chunks::KEY_PROVENANCE_MODEL_ID, NAME)?;
        require_string(file, KEY_MODEL_CATEGORY, CATEGORY)?;
        require_string(file, KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL)?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "mit")?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        )?;
        let cfg = DnsmosConfig::from_gguf(file)?;
        if cfg.bundle != ["p808", "p835"] {
            return Err(VokraError::ModelLoad(format!(
                "dnsmos: `{KEY_DNSMOS_BUNDLE}`={:?}, expected the complete public bundle [\"p808\", \"p835\"]",
                cfg.bundle
            )));
        }
        require_string(file, KEY_DNSMOS_P808_CKPT, "model_v8.onnx")?;
        require_string(file, KEY_DNSMOS_P835_CKPT, "sig_bak_ovr.onnx")?;
        validate_additive_contract(file)?;
        let bundles = [DnsmosSubmodel::P808, DnsmosSubmodel::P835]
            .into_iter()
            .map(|variant| DnsmosBundle {
                variant,
                tensor_names: file
                    .tensors()
                    .iter()
                    .filter(|tensor| tensor.name.starts_with(variant.tensor_prefix()))
                    .map(|tensor| tensor.name.clone())
                    .collect(),
            })
            .collect();
        let weights = DnsmosWeights::bind(file)?;
        Ok(Self {
            cfg,
            bundles: Arc::new(bundles),
            weights: Arc::new(weights),
            weight_license: checkpoint.weight_license(),
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and strictly binds a GGUF from disk on the CPU backend.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Strictly binds a GGUF and preflights the selected backend.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        Compute::for_backend(backend, DNSMOS_HOT_OPS)?;
        Ok(Self::from_gguf(file)?.with_backend(backend))
    }

    #[must_use]
    /// Selects a backend; availability is checked before scoring begins.
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    #[must_use]
    /// Returns the selected backend.
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    #[must_use]
    /// Returns the required PCM sample rate.
    pub const fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    #[must_use]
    /// Returns the checkpoint license class.
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    #[must_use]
    /// Returns the immutable learned-tensor count.
    pub const fn tensor_count(&self) -> usize {
        TENSOR_COUNT
    }

    /// Returns the parsed metadata contract.
    pub fn config(&self) -> &DnsmosConfig {
        &self.cfg
    }

    /// Returns the per-predictor tensor inventories.
    pub fn bundles(&self) -> &[DnsmosBundle] {
        &self.bundles
    }

    /// Test fixture model with the official topology and zero weights.
    pub fn synthesized() -> Self {
        Self {
            cfg: DnsmosConfig {
                bundle: vec!["p808".to_owned(), "p835".to_owned()],
                sample_rate: SAMPLE_RATE,
                has_p808: true,
                has_p835: true,
            },
            bundles: Arc::new(vec![
                DnsmosBundle {
                    variant: DnsmosSubmodel::P808,
                    tensor_names: Vec::new(),
                },
                DnsmosBundle {
                    variant: DnsmosSubmodel::P835,
                    tensor_names: Vec::new(),
                },
            ]),
            weights: Arc::new(DnsmosWeights::synthesized()),
            weight_license: LicenseClass::Unknown,
            backend: BackendKind::Cpu,
        }
    }

    /// Scores P.808 overall MOS for 16 kHz mono PCM.
    pub fn score_p808(&self, pcm16k: &[f32]) -> Result<f32> {
        let compute = Compute::for_backend(self.backend, DNSMOS_HOT_OPS)?;
        nn::score(&compute, &self.weights, pcm16k, true, false)?
            .p808
            .ok_or_else(|| VokraError::InvalidArgument("dnsmos: missing P.808 output".to_owned()))
    }

    /// Scores P.835 `(SIG, BAK, OVRL)` for 16 kHz mono PCM.
    pub fn score_p835(&self, pcm16k: &[f32]) -> Result<(f32, f32, f32)> {
        let compute = Compute::for_backend(self.backend, DNSMOS_HOT_OPS)?;
        nn::score(&compute, &self.weights, pcm16k, false, true)?
            .p835
            .ok_or_else(|| VokraError::InvalidArgument("dnsmos: missing P.835 output".to_owned()))
    }

    /// Scores both released predictors in one chunking pass.
    pub fn score_all(&self, pcm16k: &[f32]) -> Result<MosScore> {
        let compute = Compute::for_backend(self.backend, DNSMOS_HOT_OPS)?;
        let score = nn::score(&compute, &self.weights, pcm16k, true, true)?;
        let (sig, bak, ovrl) = score.p835.ok_or_else(|| {
            VokraError::InvalidArgument("dnsmos: missing P.835 output".to_owned())
        })?;
        Ok(MosScore {
            p808: score.p808,
            sig: Some(sig),
            bak: Some(bak),
            ovrl: Some(ovrl),
        })
    }
}

impl MosScorerEngine for Dnsmos {
    fn variants(&self) -> &[&'static str] {
        &["p808", "p835"]
    }

    fn score(&self, pcm16k: &[f32]) -> Result<MosScore> {
        self.score_all(pcm16k)
    }
}

fn validate_additive_contract(file: &GgufFile) -> Result<()> {
    // The historical public artifact predates the richer pins. Its exact
    // complete manifest plus generic provenance is the compatibility proof.
    if file.get(KEY_SOURCE_REVISION).is_none() {
        return Ok(());
    }
    for (key, expected) in [
        (KEY_SOURCE_REVISION, SOURCE_REVISION),
        (KEY_P808_ONNX_SHA256, P808_ONNX_SHA256),
        (KEY_P835_ONNX_SHA256, P835_ONNX_SHA256),
        (KEY_SOURCE_PY_SHA256, SOURCE_PY_SHA256),
        (KEY_SOURCE_LICENSE_SHA256, SOURCE_LICENSE_SHA256),
        (KEY_PUBLIC_HF, PUBLIC_HF),
        (KEY_PUBLIC_REVISION, PUBLIC_REVISION),
        (KEY_PUBLIC_GGUF_SHA256, PUBLIC_GGUF_SHA256),
        (KEY_MANIFEST_SHA256, MANIFEST_SHA256),
    ] {
        require_string(file, key, expected)?;
    }
    for (key, expected) in [
        (KEY_INPUT_LENGTH, INPUT_LENGTH_SAMPLES as u64),
        (KEY_P808_FRAMES, P808_FRAMES as u64),
        (KEY_P808_N_FFT, P808_N_FFT as u64),
        (KEY_P808_HOP, P808_HOP as u64),
        (KEY_P808_N_MELS, P808_N_MELS as u64),
        (KEY_P835_FRAMES, P835_FRAMES as u64),
        (KEY_P835_WINDOW, P835_WINDOW as u64),
        (KEY_P835_HOP, P835_HOP as u64),
        (KEY_P835_BINS, P835_BINS as u64),
    ] {
        if required_u64(file, key)? != expected {
            return Err(VokraError::ModelLoad(format!(
                "dnsmos: metadata `{key}` does not match the audited topology value {expected}"
            )));
        }
    }
    Ok(())
}

fn read_string_array(file: &GgufFile, key: &str) -> Result<Vec<String>> {
    let value = file
        .get(key)
        .ok_or_else(|| VokraError::ModelLoad(format!("dnsmos: missing `{key}`")))?;
    let array = value
        .as_array()
        .ok_or_else(|| VokraError::ModelLoad(format!("dnsmos: `{key}` is not an array")))?;
    if array.element_type != GgufValueType::String {
        return Err(VokraError::ModelLoad(format!(
            "dnsmos: `{key}` is not Array<String>"
        )));
    }
    array
        .values
        .iter()
        .map(|value| match value {
            GgufMetadataValue::String(value) => Ok(value.clone()),
            _ => Err(VokraError::ModelLoad(format!(
                "dnsmos: `{key}` contains a non-string value"
            ))),
        })
        .collect()
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_str);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "dnsmos: metadata `{key}`={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn required_u64(file: &GgufFile, key: &str) -> Result<u64> {
    file.get(key)
        .and_then(GgufMetadataValue::as_u64)
        .ok_or_else(|| VokraError::ModelLoad(format!("dnsmos: missing/non-integer `{key}`")))
}
