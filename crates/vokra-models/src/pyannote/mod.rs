//! **pyannote/segmentation-3.0** (Bredin, CNRS, MIT) — PyanNet
//! voice-activity-detection / speaker-segmentation backbone.
//!
//! # Primary source
//!
//! - Upstream reference:
//!   <https://github.com/pyannote/pyannote-audio/blob/3.0.0/pyannote/audio/models/segmentation/PyanNet.py>
//!   (MIT, exact source tag pinned below).
//! - Weight license: **MIT** (HF cardData primary source 2026-07-30,
//!   `docs/license-audit.md` §3.1 row 263 yousan ☑ Commercial).
//! - `gated: auto` is access control only (HF UI accept で誰でも DL 可、
//!   追加 license 条項なし); the weight-side gate accept is an owner
//!   task, not a runtime blocker.
//!
//! # Architecture (transcribed from PyanNet.py primary source)
//!
//! ```text
//! waveforms (batch, channel=1, samples)  # 16 kHz mono PCM
//!   -> SincNet frontend
//!      - stride=10 (SINCNET_DEFAULTS)
//!      - sample_rate=16000
//!      - output: (batch, 60, num_frames)
//!   -> rearrange "batch feature frame -> batch frame feature"
//!   -> LSTM (monolithic=True, release config override)
//!      - nn.LSTM(input_size=60, hidden_size=128, num_layers=4,
//!                bidirectional=True, batch_first=True)
//!      - output: (batch, num_frames, 256)  # 2 * 128 bidirectional
//!   -> Linear stack (LINEAR_DEFAULTS, num_layers=2, hidden_size=128)
//!      - Linear(256, 128) + leaky_relu
//!      - Linear(128, 128) + leaky_relu
//!   -> Classifier
//!      - Linear(128, num_powerset_classes)
//!      - num_powerset_classes = 7 for segmentation-3.0
//!   -> Activation (Softmax for powerset multiclass)
//! ```
//!
//! Powerset multiclass encoding (7 classes for segmentation-3.0):
//! **class 0 = silence, 1 = spk A, 2 = spk B, 3 = spk C, 4 = A+B overlap,
//! 5 = A+C overlap, 6 = B+C overlap** (3 speakers × 2 overlap slots).
//!
//! # Runtime contract
//!
//! [`PyanNet::open`] first binds the exact 54-F32-tensor public manifest,
//! enforces its owner-signed MIT provenance, and validates one of two
//! all-or-nothing metadata layouts. New files carry the immutable release and
//! public-artifact identity group. The historical public GGUF is accepted
//! only when its complete name/shape manifest and every old metadata value
//! match; its incorrect `lstm.num_layers=2` stamp is then repaired to the four
//! layers independently proven by `lstm.*_l0..l3{,_reverse}`. Partial or
//! foreign contracts fail before weight decode.
//!
//! SincNet, four-layer bidirectional LSTM, two projection layers, classifier,
//! and softmax execute by default. Every learned Conv1D, GEMV/GEMM, and
//! softmax operation uses one selected [`Compute`] backend. Only CPU and Metal
//! are accepted; unsupported or unavailable backends return an explicit error
//! and never fall back to CPU.
//!
//! # No ONNX (permanent)
//!
//! pyannote is distributed as torch `.bin` (pickle) + `config.yaml`;
//! this runtime **never** touches ONNX (FR-LD-05). The `.bin` →
//! safetensors bridge lives in `tools/parity/bin_to_safetensors.py`
//! (an offline side-car tool, not part of the runtime).

use std::path::Path;

use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue, chunks};
use vokra_core::{BackendKind, CompliancePolicy, LicenseClass, VokraError, check_weight_license};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

pub mod sincnet;
use sincnet::SincNet;

// ---------------------------------------------------------------------------
// GGUF metadata keys — mirror of
// `crates/vokra-convert/src/models/pyannote_segmentation.rs::KEY_*`.
// Two copies of the string constant is deliberate: the converter owns
// the writer contract, this runtime owns the reader contract, and a
// drift in either direction would rot silently across the crate boundary
// (a compile-time check would need to pull vokra-convert into vokra-
// models's dep graph which the workspace pins forbid).
// ---------------------------------------------------------------------------

/// The `vokra.model.arch` value a PyanNet (pyannote/segmentation-3.0)
/// GGUF must carry.
///
/// Mirror of
/// `crates/vokra-convert/src/models/pyannote_segmentation.rs::ARCH` —
/// same deliberate two-copies convention as the `GGUF_KEY_*` block below.
///
/// Deliberately **not** `pyannote-speaker-diarization`
/// (`…/pyannote_speaker_diarization_3_1.rs::ARCH`): that GGUF is a
/// weightless *pipeline orchestrator* over this VAD backbone plus a
/// WeSpeaker embedding backbone plus a clusterer. It carries clustering
/// thresholds and sub-model references, no `sincnet.*` / `lstm.*` tensors
/// at all — binding it here would refuse on the empty-manifest gate with
/// a confusing "carries no PyanNet tensor" message instead of the honest
/// "you handed me a pipeline, not a backbone".
pub const EXPECTED_ARCH: &str = "pyannote-segmentation";
/// Canonical `vokra.model.name` for the released backbone.
pub const NAME: &str = "pyannote-segmentation-3.0";
/// Canonical model-zoo category.
pub const CATEGORY: &str = "vad";
/// Immutable official upstream repository.
pub const UPSTREAM_HF: &str = "pyannote/segmentation-3.0";
/// Immutable official upstream model revision.
pub const UPSTREAM_REVISION: &str = "e66f3d3b9eb0873085418a7b813d3b369bf160bb";
/// Exact pyannote.audio source version used by the release.
pub const PYANNOTE_AUDIO_VERSION: &str = "3.0.0";
/// Peeled official pyannote.audio 3.0.0 tag revision.
pub const PYANNOTE_AUDIO_REVISION: &str = "795b92ab265888c58d160f90ae4d91b7bcc6aa2c";
/// Immutable historical public Vokra repository.
pub const PUBLIC_HF: &str = "vokra/pyannote-segmentation-3.0";
/// Immutable historical public Vokra revision.
pub const PUBLIC_REVISION: &str = "50bf4e510e0c689668384aec0f866f02e0fcaea8";
/// Historical public GGUF filename.
pub const PUBLIC_FILE: &str = "pyannote-seg.gguf";
/// Historical public GGUF byte size.
pub const PUBLIC_BYTES: u32 = 5_898_272;
/// Historical public GGUF SHA-256.
pub const PUBLIC_SHA256: &str = "22ff05fddf19e69c8d9aac8daa6d99014e6718bcd8d8c527d26da677d00c63f1";
/// Canonical raw SPDX spelling.
pub const DEFAULT_LICENSE: &str = "mit";

/// `vokra.pyannote.sample_rate` — input sample rate the SincNet was
/// tuned for (upstream PyanNet default 16000).
pub const GGUF_KEY_SAMPLE_RATE: &str = "vokra.pyannote.sample_rate";
/// `vokra.pyannote.sincnet.stride` — SincNet stride (upstream
/// SINCNET_DEFAULTS default 10).
pub const GGUF_KEY_SINCNET_STRIDE: &str = "vokra.pyannote.sincnet.stride";
/// `vokra.pyannote.lstm.hidden_size` — BiLSTM hidden dim (upstream
/// LSTM_DEFAULTS default 128).
pub const GGUF_KEY_LSTM_HIDDEN_SIZE: &str = "vokra.pyannote.lstm.hidden_size";
/// `vokra.pyannote.lstm.num_layers` — BiLSTM layer count (upstream
/// LSTM_DEFAULTS default 2).
pub const GGUF_KEY_LSTM_NUM_LAYERS: &str = "vokra.pyannote.lstm.num_layers";
/// `vokra.pyannote.lstm.bidirectional` — BiLSTM directionality
/// (upstream LSTM_DEFAULTS default true).
pub const GGUF_KEY_LSTM_BIDIRECTIONAL: &str = "vokra.pyannote.lstm.bidirectional";
/// `vokra.pyannote.lstm.monolithic` — single multi-layer nn.LSTM vs
/// stacked mono-layer LSTMs (upstream LSTM_DEFAULTS default true).
pub const GGUF_KEY_LSTM_MONOLITHIC: &str = "vokra.pyannote.lstm.monolithic";
/// `vokra.pyannote.linear.hidden_size` — Linear stack hidden dim
/// (upstream LINEAR_DEFAULTS default 128).
pub const GGUF_KEY_LINEAR_HIDDEN_SIZE: &str = "vokra.pyannote.linear.hidden_size";
/// `vokra.pyannote.linear.num_layers` — Linear stack layer count
/// (upstream LINEAR_DEFAULTS default 2).
pub const GGUF_KEY_LINEAR_NUM_LAYERS: &str = "vokra.pyannote.linear.num_layers";
/// `vokra.pyannote.num_powerset_classes` — output class count of the
/// terminal classifier (3 speakers × 2 overlap = 7 for
/// segmentation-3.0).
pub const GGUF_KEY_NUM_POWERSET_CLASSES: &str = "vokra.pyannote.num_powerset_classes";
/// Immutable upstream model revision within the PyanNet contract.
pub const GGUF_KEY_UPSTREAM_REVISION: &str = "vokra.pyannote.upstream_revision";
/// pyannote.audio source version within the PyanNet contract.
pub const GGUF_KEY_PYANNOTE_AUDIO_VERSION: &str = "vokra.pyannote.pyannote_audio_version";
/// Exact pyannote.audio source revision within the PyanNet contract.
pub const GGUF_KEY_PYANNOTE_AUDIO_REVISION: &str = "vokra.pyannote.pyannote_audio_revision";
/// Sorted name/shape manifest SHA-256 within the PyanNet contract.
pub const GGUF_KEY_MANIFEST_SHA256: &str = "vokra.pyannote.tensor_manifest_sha256";
/// Historical public artifact repository within the PyanNet contract.
pub const GGUF_KEY_PUBLIC_HF: &str = "vokra.pyannote.public_hf";
/// Historical public artifact revision within the PyanNet contract.
pub const GGUF_KEY_PUBLIC_REVISION: &str = "vokra.pyannote.public_revision";
/// Historical public filename within the PyanNet contract.
pub const GGUF_KEY_PUBLIC_FILE: &str = "vokra.pyannote.public_file";
/// Historical public artifact byte size within the PyanNet contract.
pub const GGUF_KEY_PUBLIC_BYTES: &str = "vokra.pyannote.public_bytes";
/// Historical public artifact SHA-256 within the PyanNet contract.
pub const GGUF_KEY_PUBLIC_SHA256: &str = "vokra.pyannote.public_sha256";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_PROVENANCE_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const LEGACY_SOURCE: &str = "pyannote/segmentation-3.0";
const CANONICAL_SOURCE: &str = concat!(
    "pyannote/segmentation-3.0@",
    "e66f3d3b9eb0873085418a7b813d3b369bf160bb",
    " exact 54-F32-tensor inference manifest"
);

/// SHA-256 over the sorted canonical `(tensor name, dimensions)` encoding.
pub const MANIFEST_SHA256: &str =
    "a1c783d4df253742ad5e0e796402310930f52b1a80597420f79a6eba830670d8";
const MANIFEST_SHA256_BYTES: [u8; 32] = [
    0xa1, 0xc7, 0x83, 0xd4, 0xdf, 0x25, 0x37, 0x42, 0xad, 0x5e, 0x0e, 0x79, 0x64, 0x02, 0x31, 0x09,
    0x30, 0xf5, 0x2b, 0x1a, 0x80, 0x59, 0x74, 0x20, 0xf7, 0x9a, 0x6e, 0xba, 0x83, 0x06, 0x70, 0xd8,
];
const TENSOR_COUNT: usize = 54;
const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: NAME,
    arch: EXPECTED_ARCH,
    model_name: NAME,
    model_name_alias: None,
    tensor_count: TENSOR_COUNT,
    manifest_sha256: MANIFEST_SHA256_BYTES,
};

// Primary-source constants transcribed from PyanNet.py (SINCNET_DEFAULTS
// + LSTM_DEFAULTS + LINEAR_DEFAULTS, fetched 2026-07-30 — CLAUDE.md
// 「ハルシネーション厳禁」).
/// PyanNet default sample rate.
pub const DEFAULT_SAMPLE_RATE: u32 = 16000;
/// SincNet default stride.
pub const DEFAULT_SINCNET_STRIDE: u32 = 10;
/// BiLSTM default hidden dim.
pub const DEFAULT_LSTM_HIDDEN_SIZE: u32 = 128;
/// Released checkpoint BiLSTM layer count. The class default is two, but the
/// exact segmentation-3.0 config and state dict override it to four.
pub const DEFAULT_LSTM_NUM_LAYERS: u32 = 4;
/// BiLSTM default directionality.
pub const DEFAULT_LSTM_BIDIRECTIONAL: bool = true;
/// BiLSTM default monolithic flag (single multi-layer nn.LSTM).
pub const DEFAULT_LSTM_MONOLITHIC: bool = true;
/// Linear stack default hidden dim.
pub const DEFAULT_LINEAR_HIDDEN_SIZE: u32 = 128;
/// Linear stack default layer count.
pub const DEFAULT_LINEAR_NUM_LAYERS: u32 = 2;
/// Segmentation-3.0 powerset class count (3 speakers × 2 overlap = 7).
pub const DEFAULT_NUM_POWERSET_CLASSES: u32 = 7;
/// SincNet output feature dim (fixed by the primary-source layout: the
/// first sinc conv1d + 2 conv1d+bn+maxpool blocks emit 60 features per
/// frame, wired verbatim into `nn.LSTM(60, ...)` in PyanNet.py L96).
pub const SINCNET_OUTPUT_FEATURES: u32 = 60;

/// Complete learned-op set for the PyanNet backend route.
pub const PYANNOTE_HOT_OPS: &[HotOp] = &[HotOp::Gemm, HotOp::Gemv, HotOp::Softmax, HotOp::Conv1d];

// ---------------------------------------------------------------------------
// PyanNetConfig — the (sample_rate / sincnet_stride / lstm.* / linear.*
// / num_powerset_classes) hparams
// ---------------------------------------------------------------------------

/// PyanNet hyperparameters as they ride the `vokra.pyannote.*` chunk
/// group.
///
/// [`from_gguf`](Self::from_gguf) is the loose projection used by component
/// tests and tooling. The public [`PyanNet::from_gguf`] release binder does
/// not use its per-key fallbacks: it validates the complete topology and
/// immutable identity group first, including the historical public file's
/// narrowly-scoped two-to-four-layer metadata repair. All numeric axes are
/// `u32` in the GGUF; boolean flags are `bool`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyanNetConfig {
    /// Input sample rate (default 16000, PyanNet fixed default).
    pub sample_rate: u32,
    /// SincNet stride (default 10, SINCNET_DEFAULTS).
    pub sincnet_stride: u32,
    /// BiLSTM hidden dim (default 128, LSTM_DEFAULTS).
    pub lstm_hidden_size: u32,
    /// Released BiLSTM layer count (4; the class default of 2 is overridden
    /// by the segmentation-3.0 config and state dict).
    pub lstm_num_layers: u32,
    /// BiLSTM directionality (default true, LSTM_DEFAULTS).
    pub lstm_bidirectional: bool,
    /// BiLSTM monolithic flag (default true, LSTM_DEFAULTS).
    pub lstm_monolithic: bool,
    /// Linear stack hidden dim (default 128, LINEAR_DEFAULTS).
    pub linear_hidden_size: u32,
    /// Linear stack layer count (default 2, LINEAR_DEFAULTS).
    pub linear_num_layers: u32,
    /// Terminal classifier powerset class count (default 7 for
    /// segmentation-3.0 = 3 speakers × 2 overlap).
    pub num_powerset_classes: u32,
}

impl Default for PyanNetConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            sincnet_stride: DEFAULT_SINCNET_STRIDE,
            lstm_hidden_size: DEFAULT_LSTM_HIDDEN_SIZE,
            lstm_num_layers: DEFAULT_LSTM_NUM_LAYERS,
            lstm_bidirectional: DEFAULT_LSTM_BIDIRECTIONAL,
            lstm_monolithic: DEFAULT_LSTM_MONOLITHIC,
            linear_hidden_size: DEFAULT_LINEAR_HIDDEN_SIZE,
            linear_num_layers: DEFAULT_LINEAR_NUM_LAYERS,
            num_powerset_classes: DEFAULT_NUM_POWERSET_CLASSES,
        }
    }
}

impl PyanNetConfig {
    /// Loosely projects the `vokra.pyannote.*` chunk group, falling back to
    /// the released [`Default`] constants per absent key.
    ///
    /// Production callers should use [`PyanNet::from_gguf`], whose strict
    /// all-or-nothing metadata contract rejects missing or mixed keys.
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let default = Self::default();
        Self {
            sample_rate: gguf
                .get(GGUF_KEY_SAMPLE_RATE)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.sample_rate),
            sincnet_stride: gguf
                .get(GGUF_KEY_SINCNET_STRIDE)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.sincnet_stride),
            lstm_hidden_size: gguf
                .get(GGUF_KEY_LSTM_HIDDEN_SIZE)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.lstm_hidden_size),
            lstm_num_layers: gguf
                .get(GGUF_KEY_LSTM_NUM_LAYERS)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.lstm_num_layers),
            lstm_bidirectional: gguf
                .get(GGUF_KEY_LSTM_BIDIRECTIONAL)
                .and_then(|v| v.as_bool())
                .unwrap_or(default.lstm_bidirectional),
            lstm_monolithic: gguf
                .get(GGUF_KEY_LSTM_MONOLITHIC)
                .and_then(|v| v.as_bool())
                .unwrap_or(default.lstm_monolithic),
            linear_hidden_size: gguf
                .get(GGUF_KEY_LINEAR_HIDDEN_SIZE)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.linear_hidden_size),
            linear_num_layers: gguf
                .get(GGUF_KEY_LINEAR_NUM_LAYERS)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.linear_num_layers),
            num_powerset_classes: gguf
                .get(GGUF_KEY_NUM_POWERSET_CLASSES)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.num_powerset_classes),
        }
    }
}

const IDENTITY_KEYS: &[&str] = &[
    GGUF_KEY_UPSTREAM_REVISION,
    GGUF_KEY_PYANNOTE_AUDIO_VERSION,
    GGUF_KEY_PYANNOTE_AUDIO_REVISION,
    GGUF_KEY_MANIFEST_SHA256,
    GGUF_KEY_PUBLIC_HF,
    GGUF_KEY_PUBLIC_REVISION,
    GGUF_KEY_PUBLIC_FILE,
    GGUF_KEY_PUBLIC_BYTES,
    GGUF_KEY_PUBLIC_SHA256,
];

/// Validates the all-or-nothing release metadata and returns the effective
/// config plus whether the immutable historical two-layer metadata stamp was
/// repaired. The tensor manifest is already strict-bound before this runs.
fn validate_release_metadata(gguf: &GgufFile) -> Result<(PyanNetConfig, bool), VokraError> {
    require_string(gguf, KEY_MODEL_CATEGORY, CATEGORY)?;
    require_string(gguf, chunks::KEY_PROVENANCE_MODEL_ID, NAME)?;
    let raw_license = required_string(gguf, chunks::KEY_PROVENANCE_LICENSE)?;
    if !raw_license.eq_ignore_ascii_case(DEFAULT_LICENSE) {
        return Err(metadata_error(
            chunks::KEY_PROVENANCE_LICENSE,
            raw_license,
            DEFAULT_LICENSE,
        ));
    }
    let class = required_string(gguf, chunks::KEY_PROVENANCE_WEIGHT_LICENSE)?;
    if LicenseClass::from_class_str(class) != Some(LicenseClass::Permissive) {
        return Err(metadata_error(
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            class,
            LicenseClass::Permissive.as_str(),
        ));
    }

    let source = required_string(gguf, chunks::KEY_PROVENANCE_SOURCE)?;
    let identity_count = count_present(gguf, IDENTITY_KEYS);
    match identity_count {
        0 => {
            if source != LEGACY_SOURCE {
                return Err(metadata_error(
                    chunks::KEY_PROVENANCE_SOURCE,
                    source,
                    LEGACY_SOURCE,
                ));
            }
            if gguf.get(KEY_PROVENANCE_UPSTREAM_HF).is_some()
                || gguf.get(KEY_PROVENANCE_UPSTREAM_REVISION).is_some()
            {
                return Err(VokraError::ModelLoad(
                    "pyannote-segmentation: historical metadata repair requires both generic upstream identity keys to be absent; refusing a mixed contract"
                        .to_owned(),
                ));
            }
            validate_topology(gguf, 2)?;
            Ok((PyanNetConfig::default(), true))
        }
        count if count == IDENTITY_KEYS.len() => {
            if source != CANONICAL_SOURCE {
                return Err(metadata_error(
                    chunks::KEY_PROVENANCE_SOURCE,
                    source,
                    CANONICAL_SOURCE,
                ));
            }
            require_string(gguf, KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF)?;
            require_string(gguf, KEY_PROVENANCE_UPSTREAM_REVISION, UPSTREAM_REVISION)?;
            validate_topology(gguf, DEFAULT_LSTM_NUM_LAYERS)?;
            require_string(gguf, GGUF_KEY_UPSTREAM_REVISION, UPSTREAM_REVISION)?;
            require_string(
                gguf,
                GGUF_KEY_PYANNOTE_AUDIO_VERSION,
                PYANNOTE_AUDIO_VERSION,
            )?;
            require_string(
                gguf,
                GGUF_KEY_PYANNOTE_AUDIO_REVISION,
                PYANNOTE_AUDIO_REVISION,
            )?;
            require_string(gguf, GGUF_KEY_MANIFEST_SHA256, MANIFEST_SHA256)?;
            require_string(gguf, GGUF_KEY_PUBLIC_HF, PUBLIC_HF)?;
            require_string(gguf, GGUF_KEY_PUBLIC_REVISION, PUBLIC_REVISION)?;
            require_string(gguf, GGUF_KEY_PUBLIC_FILE, PUBLIC_FILE)?;
            require_u64(gguf, GGUF_KEY_PUBLIC_BYTES, u64::from(PUBLIC_BYTES))?;
            require_string(gguf, GGUF_KEY_PUBLIC_SHA256, PUBLIC_SHA256)?;
            Ok((PyanNetConfig::default(), false))
        }
        _ => Err(VokraError::ModelLoad(format!(
            "pyannote-segmentation: partial immutable metadata {identity_count}/{} keys; refusing topology repair",
            IDENTITY_KEYS.len()
        ))),
    }
}

fn validate_topology(gguf: &GgufFile, stamped_lstm_layers: u32) -> Result<(), VokraError> {
    require_u64(gguf, GGUF_KEY_SAMPLE_RATE, u64::from(DEFAULT_SAMPLE_RATE))?;
    require_u64(
        gguf,
        GGUF_KEY_SINCNET_STRIDE,
        u64::from(DEFAULT_SINCNET_STRIDE),
    )?;
    require_u64(
        gguf,
        GGUF_KEY_LSTM_HIDDEN_SIZE,
        u64::from(DEFAULT_LSTM_HIDDEN_SIZE),
    )?;
    require_u64(
        gguf,
        GGUF_KEY_LSTM_NUM_LAYERS,
        u64::from(stamped_lstm_layers),
    )?;
    require_bool(
        gguf,
        GGUF_KEY_LSTM_BIDIRECTIONAL,
        DEFAULT_LSTM_BIDIRECTIONAL,
    )?;
    require_bool(gguf, GGUF_KEY_LSTM_MONOLITHIC, DEFAULT_LSTM_MONOLITHIC)?;
    require_u64(
        gguf,
        GGUF_KEY_LINEAR_HIDDEN_SIZE,
        u64::from(DEFAULT_LINEAR_HIDDEN_SIZE),
    )?;
    require_u64(
        gguf,
        GGUF_KEY_LINEAR_NUM_LAYERS,
        u64::from(DEFAULT_LINEAR_NUM_LAYERS),
    )?;
    require_u64(
        gguf,
        GGUF_KEY_NUM_POWERSET_CLASSES,
        u64::from(DEFAULT_NUM_POWERSET_CLASSES),
    )
}

fn validate_canonical_dtypes(gguf: &GgufFile) -> Result<(), VokraError> {
    for tensor in gguf.tensors() {
        if tensor.dtype != GgmlType::F32 {
            return Err(VokraError::ModelLoad(format!(
                "pyannote-segmentation: tensor {:?} has {:?}, expected canonical F32",
                tensor.name, tensor.dtype
            )));
        }
    }
    Ok(())
}

fn count_present(gguf: &GgufFile, keys: &[&str]) -> usize {
    keys.iter().filter(|key| gguf.get(key).is_some()).count()
}

fn required_string<'a>(gguf: &'a GgufFile, key: &str) -> Result<&'a str, VokraError> {
    gguf.get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!("pyannote-segmentation: missing/non-string `{key}`"))
        })
}

fn require_string(gguf: &GgufFile, key: &str, expected: &str) -> Result<(), VokraError> {
    let actual = required_string(gguf, key)?;
    if actual != expected {
        return Err(metadata_error(key, actual, expected));
    }
    Ok(())
}

fn require_u64(gguf: &GgufFile, key: &str, expected: u64) -> Result<(), VokraError> {
    let actual = gguf
        .get(key)
        .and_then(GgufMetadataValue::as_u64)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!("pyannote-segmentation: missing/non-u32 `{key}`"))
        })?;
    if actual != expected {
        return Err(metadata_error(key, actual, expected));
    }
    Ok(())
}

fn require_bool(gguf: &GgufFile, key: &str, expected: bool) -> Result<(), VokraError> {
    let actual = gguf
        .get(key)
        .and_then(GgufMetadataValue::as_bool)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!("pyannote-segmentation: missing/non-bool `{key}`"))
        })?;
    if actual != expected {
        return Err(metadata_error(key, actual, expected));
    }
    Ok(())
}

fn metadata_error(
    key: &str,
    actual: impl std::fmt::Debug,
    expected: impl std::fmt::Debug,
) -> VokraError {
    VokraError::ModelLoad(format!(
        "pyannote-segmentation: `{key}` is {actual:?}, expected {expected:?}"
    ))
}

// ---------------------------------------------------------------------------
// PyanNetWeights — real weight-tensor binding with loud-error on missing
// ---------------------------------------------------------------------------

/// The upstream PyanNet state_dict tensor-name prefixes the component weight
/// binder scans for. The public [`PyanNet`] loader first verifies the exact
/// 54-tensor release manifest, so this looser seam exists only for internal
/// primitive tests and diagnostics.
///
/// Sourced from the upstream PyanNet.py class definition: SincNet
/// module (`sincnet.*`), monolithic BiLSTM (`lstm.*`), Linear stack
/// (`linear.*` as an `nn.ModuleList`), and the terminal classifier
/// (`classifier.*`).
const REQUIRED_TENSOR_PREFIXES: &[&str] = &[
    "sincnet.",    // SincNet frontend (learnable sinc conv + conv stack)
    "lstm.",       // Monolithic BiLSTM (`nn.LSTM` — weight_ih_l0, weight_hh_l0, etc.)
    "linear.",     // Linear stack (`nn.ModuleList` of Linear layers)
    "classifier.", // Terminal classifier (Linear(128, num_powerset_classes))
];

/// Rejects a GGUF whose `vokra.model.arch` is absent or is not
/// [`EXPECTED_ARCH`].
///
/// A *loud* validation step (FR-EX-08) — see
/// [`PyanNetWeights::from_gguf`].
pub(crate) fn verify_arch(gguf: &GgufFile) -> Result<(), VokraError> {
    match gguf
        .get(vokra_core::gguf::chunks::KEY_MODEL_ARCH)
        .and_then(|v| v.as_str())
    {
        Some(a) if a == EXPECTED_ARCH => Ok(()),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "pyannote-segmentation: GGUF arch is `{other}`, expected `{EXPECTED_ARCH}` (was \
             this GGUF produced by `vokra-cli convert --model pyannote-segmentation`? Note \
             that `pyannote-speaker-diarization` is a *pipeline orchestrator* over this VAD \
             backbone — it carries clustering thresholds and sub-model references, not \
             `sincnet.*` / `lstm.*` weights — and that sibling VAD arches `silero-vad`, \
             `fsmn-vad`, `firered-vad` are entirely different topologies. This binder \
             matches on bare upstream state_dict prefixes, so a foreign checkpoint with an \
             `lstm.*` tensor would clear the non-emptiness gate and bind a partial, \
             meaningless model (FR-EX-08 — no silent partial load)."
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "pyannote-segmentation: GGUF is missing `{}` — this is not a Vokra-native \
             pyannote/segmentation-3.0 GGUF (was it produced by `vokra-cli convert --model \
             pyannote-segmentation`?)",
            vokra_core::gguf::chunks::KEY_MODEL_ARCH,
        ))),
    }
}

/// Weight tensors bound from a PyanNet GGUF.
///
/// Each field carries the flattened f32 payload of a tensor read from the GGUF
/// by its upstream `state_dict` name. The public release binder admits only
/// the canonical F32 manifest; F16/BF16 widening remains available here for
/// isolated component fixtures and does not relax that release contract.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud*
/// verification step. A GGUF that carries no PyanNet-typical tensor is
/// rejected with [`VokraError::ModelLoad`] naming the required prefixes
/// (FR-EX-08). A tensor whose payload cannot be dequantized to f32 (or
/// which has an unexpected non-float dtype) is likewise refused.
#[derive(Debug)]
pub struct PyanNetWeights {
    /// Tensors indexed by upstream `state_dict` name.
    ///
    /// Each entry is `(name, dims, f32 payload)`. Dims match the
    /// upstream torch shape order (row-major); the f32 payload is
    /// dequantized on load so downstream kernels see a uniform dtype
    /// regardless of the checkpoint's F32 / F16 / BF16 provenance.
    tensors: Vec<(String, Vec<usize>, Vec<f32>)>,
}

impl PyanNetWeights {
    /// Scans `gguf` for all recognized PyanNet `state_dict` tensors and
    /// dequantizes each to f32. Refuses to bind if no tensor matches
    /// any `REQUIRED_TENSOR_PREFIXES` entry (FR-EX-08).
    ///
    /// # Arch verification (FR-EX-08)
    ///
    /// `vokra.model.arch` is checked against [`EXPECTED_ARCH`] **before**
    /// any tensor is scanned. The binder matches on bare upstream
    /// `state_dict` prefixes (`sincnet.` / `lstm.` / `linear.` /
    /// `classifier.`) — generic enough that a foreign checkpoint could
    /// satisfy the non-emptiness gate on `lstm.*` alone and bind a
    /// partial, meaningless model.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent or
    ///   is not [`EXPECTED_ARCH`].
    /// - [`VokraError::ModelLoad`] when the GGUF carries no
    ///   PyanNet-typical tensor. The error message names every prefix
    ///   the binder tried so the caller can validate the checkpoint's
    ///   flattening convention.
    /// - [`VokraError::ModelLoad`] when a matched tensor has an
    ///   unsupported dtype (only F32 / F16 / BF16 are accepted at this
    ///   seam — K-quants are rejected loudly).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, VokraError> {
        // 1. Arch check — always first so a mis-typed model handed here
        //    fails with a specific message instead of a downstream
        //    "carries no PyanNet-typical tensor".
        verify_arch(gguf)?;

        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();

        for info in gguf.tensors() {
            let name = info.name.as_str();
            if !REQUIRED_TENSOR_PREFIXES.iter().any(|p| name.starts_with(p)) {
                continue;
            }
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            let payload = dequant_to_f32(gguf, info)?;
            tensors.push((name.to_owned(), dims, payload));
        }

        if tensors.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "pyannote-segmentation: GGUF carries no tensor matching any of the upstream \
                 PyanNet prefixes {REQUIRED_TENSOR_PREFIXES:?}; refusing to bind an all-zero \
                 forward (FR-EX-08)"
            )));
        }

        Ok(Self { tensors })
    }

    /// Number of PyanNet-typical tensors bound from the GGUF. Purely a
    /// diagnostic accessor — the tests and the follow-up SincNet /
    /// BiLSTM kernel wave use it to size their expectations.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Looks up the f32 payload + dims of a bound tensor by its
    /// upstream `state_dict` name. Returns `None` if the tensor is not
    /// among the loaded set (either the GGUF omits it or its name is
    /// not among the recognized prefixes).
    pub fn tensor(&self, name: &str) -> Option<(&[usize], &[f32])> {
        self.tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, d, p)| (d.as_slice(), p.as_slice()))
    }

    /// Supplementary component shape gate for the four core PyanNet tensors.
    /// The public release loader already rejects every complete-manifest drift
    /// before decode; this helper preserves focused diagnostics for smaller
    /// primitive fixtures.
    ///
    /// # Sentinel-gated strict mode
    ///
    /// - Presence of the SincNet filterbank tensor
    ///   (`sincnet.conv1d.0.filterbank.low_hz_`) is the **sentinel**
    ///   for "this GGUF claims to be a real PyanNet-3.0 checkpoint".
    /// - When the sentinel is present, **all four** core tensors must
    ///   be present with the primary-source shapes derived from
    ///   `config`. A missing core tensor is a loud
    ///   [`VokraError::ModelLoad`] naming the absent tensor path.
    /// - When the sentinel is absent, the fixture is treated as
    ///   illustrative (binder / plumbing smoke test) and missing
    ///   tensors pass through silently — the downstream forward will
    ///   still loud-fail via [`sincnet::SincNet::from_weights`] if a caller
    ///   tries to execute it.
    /// - **Present-but-mis-shaped tensors are always rejected loudly**,
    ///   regardless of sentinel presence — an obviously-wrong shape is
    ///   a silent-fake risk that must not survive the load path.
    ///
    /// # Expected shapes (primary source: PyanNet.py + PyTorch layouts)
    ///
    /// | tensor                                    | shape                            |
    /// |-------------------------------------------|----------------------------------|
    /// | `sincnet.conv1d.0.filterbank.low_hz_`     | `[N_FILTERS_SINC/2, 1]` = `[40, 1]` |
    /// | `lstm.weight_ih_l0`                       | `[4·H, SincNet.out]` = `[512, 60]` |
    /// | `linear.0.weight`                         | `[linear_h, 2·H]` = `[128, 256]`   |
    /// | `classifier.weight`                       | `[n_classes, linear_h]` = `[7, 128]` |
    ///
    /// Every axis is derived from `config` so a future PyanNet variant
    /// with a different `hidden_size` / `num_powerset_classes` / etc.
    /// still gets a correct load-time gate.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] tagged `FR-EX-08` when any present
    /// core tensor has a shape drift, or when the sentinel is present
    /// and another core tensor is missing. Every message names the
    /// offending tensor path + the expected shape so the caller can
    /// trace back to the converter output.
    pub fn verify_core_shapes(&self, config: &PyanNetConfig) -> Result<(), VokraError> {
        // Primary-source constants (mirror of `sincnet.rs`):
        //   N_FILTERS_SINC = 80 → learnable rows = 40 (torch.abs is applied
        //     to only the first half, and the second half is the mirror,
        //     so the on-disk tensor is `(n_filters/2, 1)`).
        //   SincNet.out = 60 (CONV2_OUT_CH — the SincNet final channel dim
        //     that feeds the downstream monolithic BiLSTM as `nn.LSTM(60, H)`).
        const N_SINC_LEARNABLE: usize = 40;
        const SINCNET_FEATURES: usize = 60;

        let lstm_h = config.lstm_hidden_size as usize;
        // PyTorch `nn.LSTM` layout: `weight_ih_l0` is `(4·H, input_size)`
        // (gates concatenated i|f|g|o — see `torch/nn/modules/rnn.py`).
        let gates_dim = 4 * lstm_h;
        // Bidirectional LSTMs concatenate forward + reverse outputs, so
        // the downstream Linear stack sees `2·H` per timestep.
        let lstm_out_dim = if config.lstm_bidirectional {
            2 * lstm_h
        } else {
            lstm_h
        };
        let linear_h = config.linear_hidden_size as usize;
        let n_classes = config.num_powerset_classes as usize;

        // The four "core" tensors that a real PyanNet-3.0 checkpoint
        // MUST carry with these exact shapes.
        let expectations: [(&str, Vec<usize>); 4] = [
            (
                "sincnet.conv1d.0.filterbank.low_hz_",
                vec![N_SINC_LEARNABLE, 1],
            ),
            ("lstm.weight_ih_l0", vec![gates_dim, SINCNET_FEATURES]),
            ("linear.0.weight", vec![linear_h, lstm_out_dim]),
            ("classifier.weight", vec![n_classes, linear_h]),
        ];

        // Sentinel: presence of the SincNet filterbank marks the
        // fixture as "real GGUF topology" and enables the strict
        // co-presence gate on the rest of the core tensors. Illustrative
        // fixtures (no filterbank) get a permissive pass-through — the
        // downstream `SincNet::from_weights` still loud-fails on any
        // real forward attempt, so silent-fake is not possible either
        // way (FR-EX-08).
        let sentinel = "sincnet.conv1d.0.filterbank.low_hz_";
        let has_sentinel = self.tensor(sentinel).is_some();

        for (name, expect) in &expectations {
            match self.tensor(name) {
                Some((dims, _)) if dims != expect.as_slice() => {
                    return Err(VokraError::ModelLoad(format!(
                        "pyannote-segmentation: tensor `{name}` has shape {dims:?}, \
                         expected {expect:?} at load time (FR-EX-08 shape-validation-\
                         load-time). Primary source: PyanNet.py (CNRS, MIT). Fix the \
                         converter or checkpoint."
                    )));
                }
                Some(_) => {
                    // Present and correctly shaped.
                }
                None if has_sentinel => {
                    return Err(VokraError::ModelLoad(format!(
                        "pyannote-segmentation: GGUF carries the SincNet filterbank sentinel \
                         (`{sentinel}`) but is missing the core tensor `{name}` (expected \
                         shape {expect:?}) — a real PyanNet-3.0 checkpoint must carry all \
                         four core tensors (FR-EX-08 shape-validation-load-time)."
                    )));
                }
                None => {
                    // Illustrative fixture (no sentinel) — the downstream
                    // forward will loud-fail via SincNet::from_weights if the
                    // caller ever tries to execute it. Skip.
                }
            }
        }
        Ok(())
    }
}

/// Widens a GGUF tensor payload to a flat `Vec<f32>`. Supports F32,
/// F16, and BF16 (the three PyanNet checkpoint dtypes the converter
/// admits). Every other dtype is a loud [`VokraError::ModelLoad`]
/// (FR-EX-08). Mirror of `crates/vokra-models/src/f0/rmvpe.rs
/// dequant_to_f32` — deliberate copy since crate-boundary sharing would
/// pull the pyannote binder into vokra-core's public API for an
/// internal 30-line helper.
fn dequant_to_f32(
    gguf: &GgufFile,
    info: &vokra_core::gguf::GgufTensorInfo,
) -> Result<Vec<f32>, VokraError> {
    let bytes = gguf.tensor_data(&info.name).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "pyannote-segmentation: no data slice for tensor `{}`",
            info.name
        ))
    })?;
    let elems: usize = info.dimensions.iter().map(|&d| d as usize).product();

    match info.dtype {
        GgmlType::F32 => {
            if bytes.len() != elems * 4 {
                return Err(VokraError::ModelLoad(format!(
                    "pyannote-segmentation: tensor `{}` F32 byte count {} != elems {} * 4",
                    info.name,
                    bytes.len(),
                    elems
                )));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        GgmlType::F16 => {
            if bytes.len() != elems * 2 {
                return Err(VokraError::ModelLoad(format!(
                    "pyannote-segmentation: tensor `{}` F16 byte count {} != elems {} * 2",
                    info.name,
                    bytes.len(),
                    elems
                )));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        }
        GgmlType::BF16 => {
            if bytes.len() != elems * 2 {
                return Err(VokraError::ModelLoad(format!(
                    "pyannote-segmentation: tensor `{}` BF16 byte count {} != elems {} * 2",
                    info.name,
                    bytes.len(),
                    elems
                )));
            }
            // BF16 = top 16 bits of an f32 — `bits << 16` widens
            // losslessly.
            Ok(bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                .collect())
        }
        other => Err(VokraError::ModelLoad(format!(
            "pyannote-segmentation: tensor `{}` has unsupported dtype {other:?} \
             (only F32 / F16 / BF16 are accepted at this seam — FR-EX-08)",
            info.name
        ))),
    }
}

/// Widens an IEEE-754 half-precision f16 bit pattern to f32. Same
/// implementation as `crates/vokra-models/src/f0/rmvpe.rs`.
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1F);
    let mant = u32::from(h & 0x3FF);
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = 1i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let e32 = (127 - 15 + e) as u32;
            sign | (e32 << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        sign | (0xFF << 23) | (mant << 13)
    } else {
        let e32 = exp + (127 - 15);
        sign | (e32 << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn validate_backend(backend: BackendKind) -> Result<(), VokraError> {
    match backend {
        BackendKind::Cpu | BackendKind::Metal => Ok(()),
        other => Err(VokraError::UnsupportedOp(format!(
            "pyannote-segmentation: backend {other:?} is unsupported; the complete release is implemented for Mac CPU and Metal only (FR-EX-08, no CPU fallback)"
        ))),
    }
}

// ---------------------------------------------------------------------------
// PyanNet — the public engine handle
// ---------------------------------------------------------------------------

/// PyanNet segmentation model — the pyannote/segmentation-3.0 backbone
/// (VAD + speaker segmentation, MIT).
///
/// Load with [`from_gguf`](Self::from_gguf) / [`open`](Self::open),
/// then call [`segment`](Self::segment) on a PCM buffer to obtain a
/// per-frame powerset multiclass stream. See the module doc for the
/// current implementation-status matrix and the FR-EX-08 loud-error
/// contract on the SincNet + BiLSTM + Linear forward.
#[derive(Debug)]
pub struct PyanNet {
    checkpoint: StrictCheckpoint,
    config: PyanNetConfig,
    weights: PyanNetWeights,
    backend: BackendKind,
    legacy_metadata_repaired: bool,
}

impl PyanNet {
    /// Loads a PyanNet model from a GGUF file on disk.
    ///
    /// The GGUF must:
    ///
    /// 1. Be openable by the standard GGUF reader — errors surface as
    ///    [`VokraError::Io`] / [`VokraError::ModelLoad`].
    /// 2. Carry a `vokra.model.arch` equal to [`EXPECTED_ARCH`] — checked
    ///    by [`PyanNetWeights::from_gguf`] before any tensor is scanned
    ///    (FR-EX-08).
    /// 3. Match the exact 54-tensor sorted name/shape manifest and F32 dtype.
    /// 4. Carry either the complete new immutable metadata group or the exact
    ///    historical public group whose two-layer stamp is safely repaired.
    /// 5. Pass the strict MIT compliance gate.
    pub fn from_gguf(path: &Path) -> Result<Self, VokraError> {
        let gguf = GgufFile::open(path)?;
        let checkpoint = StrictCheckpoint::bind(&gguf, SPEC)?;
        validate_canonical_dtypes(&gguf)?;
        let license = check_weight_license(&gguf, &CompliancePolicy::strict())?;
        if license.class != LicenseClass::Permissive
            || checkpoint.weight_license() != LicenseClass::Permissive
        {
            return Err(VokraError::ModelLoad(format!(
                "pyannote-segmentation: weight license resolves to {}, expected Permissive for the owner-signed MIT release",
                license.class.as_str()
            )));
        }
        let (config, legacy_metadata_repaired) = validate_release_metadata(&gguf)?;
        let weights = PyanNetWeights::from_gguf(&gguf)?;
        weights.verify_core_shapes(&config)?;
        Ok(Self {
            checkpoint,
            config,
            weights,
            backend: BackendKind::Cpu,
            legacy_metadata_repaired,
        })
    }

    /// Strictly opens a release and preflights complete backend coverage.
    pub fn from_gguf_with_backend(path: &Path, backend: BackendKind) -> Result<Self, VokraError> {
        validate_backend(backend)?;
        Compute::for_backend(backend, PYANNOTE_HOT_OPS)?;
        Ok(Self::from_gguf(path)?.with_backend(backend))
    }

    /// Convenience alias for [`from_gguf`](Self::from_gguf).
    pub fn open(path: impl AsRef<Path>) -> Result<Self, VokraError> {
        Self::from_gguf(path.as_ref())
    }

    /// The strictly validated released hyperparameter set.
    pub fn config(&self) -> &PyanNetConfig {
        &self.config
    }

    /// Canonical model identity proven by the strict manifest binder.
    #[must_use]
    pub fn model_name(&self) -> &str {
        self.checkpoint.model_name()
    }

    /// Owner-signed license class proven at load time.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.checkpoint.weight_license()
    }

    /// Whether the exact historical public artifact's incorrect two-layer
    /// metadata stamp was repaired from its four-layer tensor manifest.
    #[must_use]
    pub const fn legacy_metadata_repaired(&self) -> bool {
        self.legacy_metadata_repaired
    }

    /// Selects the backend for every learned SincNet, BiLSTM, linear and
    /// classifier operation. Unsupported selections fail when inference is
    /// called; use [`Self::from_gguf_with_backend`] for eager preflight.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Returns the selected backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Computes the number of output frames for a given number of
    /// input PCM samples.
    ///
    /// The recurrence is the primary-source `multi_conv_num_frames`
    /// from `pyannote/audio/utils/receptive_field.py:56-69`, transcribed
    /// verbatim into [`sincnet::num_frames`]. The 2026-07-30 Wave 2
    /// stub `num_samples / sincnet_stride` was a placeholder — it
    /// overshot by ~28× because it counted the first SincNet layer's
    /// stride-only output while ignoring the three subsequent
    /// MaxPool1d(3, stride=3) contractions. Wave 3 replaces the stub
    /// with the real recurrence; callers that memoised the old value
    /// against a fixture will need to re-record.
    ///
    /// # Pin-tested values (see [`sincnet::num_frames`])
    ///
    /// | input PCM (samples) | frames (stride=10) |
    /// |---------------------|--------------------|
    /// | 160 000 (10 s)      | 589                |
    /// | 16 000  (1 s)       | 56                 |
    /// | 1 600   (0.1 s)     | 3                  |
    /// | < 251               | 0                  |
    pub fn num_frames(&self, num_samples: usize) -> usize {
        sincnet::num_frames(num_samples, self.config.sincnet_stride as usize)
    }

    /// Deprecated compatibility spelling. Execution is now default-on and the
    /// environment variable is ignored.
    #[deprecated(note = "PyanNet execution is default-on; this env variable is ignored")]
    pub const ENV_ENABLE_FORWARD: &'static str = "VOKRA_PYANNET_ENABLE_FORWARD";

    /// Segments a mono-channel 16-kHz PCM buffer into per-frame
    /// powerset multiclass **probabilities** (post-softmax).
    ///
    /// # Returns
    ///
    /// On success, `Vec<Vec<f32>>` of length
    /// `num_frames(pcm.len())`, each entry a length-`num_powerset_classes`
    /// row of softmax probabilities that sum to ~1. The row layout
    /// matches upstream `PyanNet.forward` (`activation = Softmax(dim=-1)`
    /// as re-derived from `LogSoftmax(dim=-1)` for the powerset multi-
    /// class problem — see [`decode_powerset`] for the frame → active-
    /// speaker mapping).
    ///
    /// # Errors
    ///
    /// Unsupported/unavailable backends and invalid PCM fail explicitly.
    pub fn segment(&self, pcm: &[f32]) -> Result<Vec<Vec<f32>>, VokraError> {
        self.segment_real(pcm, self.config.sample_rate)
    }

    /// Real SincNet + BiLSTM + Linear + Classifier + Softmax forward
    /// shared by the public method and the staged parity harness.
    pub(crate) fn segment_real(
        &self,
        pcm: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<Vec<f32>>, VokraError> {
        // Step 1: SincNet frontend — real, tested primitive.
        let stride = self.config.sincnet_stride as usize;
        let sn = SincNet::from_weights(&self.weights, stride)?;
        validate_backend(self.backend)?;
        let compute = Compute::for_backend(self.backend, PYANNOTE_HOT_OPS)?;
        let sinc_out = sn.forward_with_compute(pcm, sample_rate, &compute)?;
        // sinc_out.features is [num_channels · num_frames] channel-major
        // (row-major with channel outer); the downstream BiLSTM expects
        // row-major [num_frames · num_channels] (batch_first=True). Do
        // the rearrange once so the LSTM sees `[t, feature]`.
        let (t, c) = (sinc_out.num_frames, sinc_out.num_channels);
        let mut lstm_in = vec![0.0f32; t * c];
        for ch in 0..c {
            for f in 0..t {
                lstm_in[f * c + ch] = sinc_out.features[ch * t + f];
            }
        }

        // Step 2: released four-layer monolithic bidirectional LSTM.
        let bilstm = bilstm::MonoLithicBiLstmStack::from_weights(
            &self.weights,
            c,                                     // input_dim = 60
            self.config.lstm_hidden_size as usize, // hidden_dim = 128
            self.config.lstm_num_layers as usize,  // num_layers = 4
        )?;
        let lstm_out = bilstm.forward_with_compute(&lstm_in, t, &compute)?;
        // lstm_out is [t · (2 · hidden_dim)] row-major.
        let lstm_out_dim = 2 * self.config.lstm_hidden_size as usize;

        // Step 3: Linear stack — 2 layers of Linear(in → 128) + LeakyReLU.
        let linear = linear_stack::LinearStack::from_weights(
            &self.weights,
            lstm_out_dim,                            // linear.0.in = 256
            self.config.linear_hidden_size as usize, // linear.*.out = 128
            self.config.linear_num_layers as usize,  // 2
        )?;
        let linear_out = linear.forward_with_compute(&lstm_out, t, &compute)?;
        // linear_out is [t · linear_hidden_size].

        // Step 4: Classifier + Softmax.
        let classifier = classifier::Classifier::from_weights(
            &self.weights,
            self.config.linear_hidden_size as usize,
            self.config.num_powerset_classes as usize,
        )?;
        let probs = classifier.forward_with_compute(&linear_out, t, &compute)?;
        // Reshape to Vec<Vec<f32>> per frame.
        let n_classes = self.config.num_powerset_classes as usize;
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(t);
        for f in 0..t {
            out.push(probs[f * n_classes..(f + 1) * n_classes].to_vec());
        }
        Ok(out)
    }

    /// Segments a mono-channel 16-kHz PCM buffer and decodes the
    /// per-frame powerset probabilities into per-frame active-speaker
    /// sets (multi-label decode, per [`decode_powerset`]).
    ///
    /// Composition of [`Self::segment`] + [`decode_powerset`].
    pub fn segment_powerset(&self, pcm: &[f32]) -> Result<Vec<SpeakerActivity>, VokraError> {
        let probs = self.segment(pcm)?;
        let stride = self.config.sincnet_stride as usize;
        Ok(decode_powerset(
            &probs,
            self.config.num_powerset_classes as usize,
            self.config.sample_rate,
            stride,
        ))
    }

    /// Sample-rate-explicit sibling used by in-crate tests.
    #[allow(dead_code)]
    pub(crate) fn segment_powerset_real(
        &self,
        pcm: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<SpeakerActivity>, VokraError> {
        let probs = self.segment_real(pcm, sample_rate)?;
        let stride = self.config.sincnet_stride as usize;
        Ok(decode_powerset(
            &probs,
            self.config.num_powerset_classes as usize,
            self.config.sample_rate,
            stride,
        ))
    }
}

// ---------------------------------------------------------------------------
// Powerset decoder
// ---------------------------------------------------------------------------

/// Per-frame active-speaker set — the argmax-decoded powerset row
/// looked up in [`POWERSET_MAPPING_3SPK_2OVERLAP`].
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerActivity {
    /// Zero-based frame index within the utterance.
    pub frame_idx: usize,
    /// Frame midpoint in seconds since utterance start.
    /// `t = (frame_idx + 0.5) · sincnet_stride / sample_rate` — an
    /// approximation of the centre of the sinc-conv receptive field.
    pub time_s: f32,
    /// Zero-based indices of the speakers active at this frame
    /// (`{}` for silence, `{0}` / `{1}` / `{2}` for single-speaker,
    /// `{0,1}` / `{0,2}` / `{1,2}` for pairwise overlap).
    pub active_speakers: Vec<usize>,
}

/// Powerset multi-class → multi-label mapping matrix for
/// `pyannote/segmentation-3.0` (`num_classes=3, max_set_size=2`).
///
/// **7 rows × 3 columns**, transcribed verbatim from
/// `pyannote/audio/utils/powerset.py:69-108`. Row `i` is a length-3
/// binary vector marking which of the 3 speakers is active in the
/// powerset class `i`:
///
/// | row | powerset class | active |
/// |-----|----------------|--------|
/// | 0   | silence        | `{}`   |
/// | 1   | speaker A only | `{0}`  |
/// | 2   | speaker B only | `{1}`  |
/// | 3   | speaker C only | `{2}`  |
/// | 4   | A + B overlap  | `{0,1}`|
/// | 5   | A + C overlap  | `{0,2}`|
/// | 6   | B + C overlap  | `{1,2}`|
///
/// The row order matches the nested-loop iteration in the upstream
/// `Powerset._build_mapping()` (set_size then combinations) so a
/// powerset argmax → speaker set lookup requires no recomputation.
pub const POWERSET_MAPPING_3SPK_2OVERLAP: [[u8; 3]; 7] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
];

/// Hard powerset → multi-label decode.
///
/// For each frame, take the argmax of the powerset probability row
/// (7 classes), look up the mapping matrix, and materialise the
/// corresponding active-speaker set. The mapping is transcribed from
/// `powerset.py:132-140 to_multilabel(soft=False)` — Wave 3 lands the
/// hard variant only (deterministic, no numerical stability worry).
///
/// # Panics
///
/// Does not panic — an out-of-bounds argmax is impossible given
/// [`POWERSET_MAPPING_3SPK_2OVERLAP`] has 7 rows and the powerset row
/// is required to have exactly `num_powerset_classes` columns. Every
/// row shorter than `num_powerset_classes` maps to silence (index 0
/// default), an intentional loud-safe behaviour.
pub fn decode_powerset(
    per_frame_probs: &[Vec<f32>],
    num_powerset_classes: usize,
    sample_rate: u32,
    sincnet_stride: usize,
) -> Vec<SpeakerActivity> {
    // Vokra only ships the pyannote-3.0 default (7-class powerset over
    // 3 speakers × 2 overlap); guard against any drift so a
    // future-compat mapping table is added deliberately.
    debug_assert_eq!(
        num_powerset_classes,
        POWERSET_MAPPING_3SPK_2OVERLAP.len(),
        "decode_powerset: only the 7-class pyannote-3.0 mapping is available; \
         to add another powerset shape, extend POWERSET_MAPPING_* and the mapping table"
    );

    let frame_seconds = sincnet_stride as f32 / sample_rate as f32;
    let mut out = Vec::with_capacity(per_frame_probs.len());
    for (f, row) in per_frame_probs.iter().enumerate() {
        // Argmax (first-occurrence semantics per `torch.argmax`).
        let mut max_idx = 0usize;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &p) in row.iter().enumerate() {
            if p > max_val {
                max_val = p;
                max_idx = i;
            }
        }
        // Map the argmax to the active-speaker set.
        let mapping_row = if max_idx < POWERSET_MAPPING_3SPK_2OVERLAP.len() {
            &POWERSET_MAPPING_3SPK_2OVERLAP[max_idx]
        } else {
            &POWERSET_MAPPING_3SPK_2OVERLAP[0] // silence
        };
        let active_speakers: Vec<usize> = mapping_row
            .iter()
            .enumerate()
            .filter_map(|(spk, &on)| if on == 1 { Some(spk) } else { None })
            .collect();
        out.push(SpeakerActivity {
            frame_idx: f,
            time_s: (f as f32 + 0.5) * frame_seconds,
            active_speakers,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Private BiLSTM + Linear + Classifier submodules
// ---------------------------------------------------------------------------

mod bilstm;
mod classifier;
mod linear_stack;

// ---------------------------------------------------------------------------
// Wave 4 pyannote diarization pipeline submodules (2026-07-30 ultracode
// integration). Each landed as a parallel worktree; the integrator wires
// them into this mod.rs while preserving Wave 3's full SincNet + forward
// wire. Design 判断 8: pyannote is speaker diarization, NOT voice
// cloning → stays in main `ayutaz/vokra`.
// ---------------------------------------------------------------------------

/// NIST RTTM (Rich Transcription Time Marked) output writer. Consumes a
/// sorted / merged [`rttm::DiarizationSegment`] list and produces one
/// SPEAKER line per turn per the pyannote-core `Annotation.write_rttm()`
/// contract (verbatim from primary source, MIT Copyright (c) 2020 CNRS).
pub mod rttm;

/// Exact native speaker-diarization 3.1 pipeline composing PyanNet,
/// WeSpeaker, centroid agglomerative clustering, discrete speaker-count
/// reconstruction and [`rttm`] output. CPU and Metal selections cover both
/// learned models without a silent per-operation fallback.
pub mod diarization;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-pyannote-runtime-{}-{}-{}.gguf",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        p
    }

    /// A GGUF with just the required prefix tensors — a synthetic
    /// checkpoint sized so the runtime binder's non-emptiness check
    /// passes but whose shapes are placeholders (not the real
    /// upstream layout). Enough for `from_gguf` smoke tests only —
    /// callers that need the real forward must use
    /// [`synthetic_full_pyannet_gguf`].
    fn synthetic_pyannet_gguf() -> Vec<u8> {
        let mut b = GgufBuilder::new();
        // Arch stamp — `PyanNetWeights::from_gguf` gates on it before any
        // tensor scan (FR-EX-08), so every fixture that expects to reach
        // the tensor manifest must carry it.
        b.add_string(vokra_core::gguf::chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        // Metadata chunks — the converter writes these, and the runtime
        // reads them via `PyanNetConfig::from_gguf`. Using the default
        // constants here lets us pin the fallback path AND the
        // read-back path in one round-trip.
        b.add_u32(GGUF_KEY_SAMPLE_RATE, DEFAULT_SAMPLE_RATE);
        b.add_u32(GGUF_KEY_SINCNET_STRIDE, DEFAULT_SINCNET_STRIDE);
        b.add_u32(GGUF_KEY_LSTM_HIDDEN_SIZE, DEFAULT_LSTM_HIDDEN_SIZE);
        b.add_u32(GGUF_KEY_LSTM_NUM_LAYERS, DEFAULT_LSTM_NUM_LAYERS);
        b.add_bool(GGUF_KEY_LSTM_BIDIRECTIONAL, DEFAULT_LSTM_BIDIRECTIONAL);
        b.add_bool(GGUF_KEY_LSTM_MONOLITHIC, DEFAULT_LSTM_MONOLITHIC);
        b.add_u32(GGUF_KEY_LINEAR_HIDDEN_SIZE, DEFAULT_LINEAR_HIDDEN_SIZE);
        b.add_u32(GGUF_KEY_LINEAR_NUM_LAYERS, DEFAULT_LINEAR_NUM_LAYERS);
        b.add_u32(GGUF_KEY_NUM_POWERSET_CLASSES, DEFAULT_NUM_POWERSET_CLASSES);

        // One tensor per required prefix — enough to satisfy
        // `PyanNetWeights::from_gguf` non-emptiness gate. Payloads are
        // small F32 vectors; the shapes are illustrative, not the
        // upstream shapes (Wave 3 will introduce shape assertions
        // against real dims).
        let tensor_specs: [(&str, &[u64]); 4] = [
            ("sincnet.conv1d.0.weight", &[8, 1, 251]),
            ("lstm.weight_ih_l0", &[512, 60]),
            ("linear.0.weight", &[128, 256]),
            ("classifier.weight", &[7, 128]),
        ];
        for (name, shape) in tensor_specs {
            let elems: u64 = shape.iter().product();
            let bytes: Vec<u8> = (0..elems as usize)
                .flat_map(|i| (i as f32 * 0.001).to_le_bytes())
                .collect();
            b.add_tensor(name, GgmlType::F32, shape.to_vec(), bytes)
                .expect("add_tensor");
        }

        b.to_bytes().expect("gguf serialize")
    }

    #[derive(Clone, Copy)]
    enum TestMetadata {
        Canonical,
        Legacy,
        PartialCanonical,
    }

    fn stamp_test_metadata(b: &mut GgufBuilder, metadata: TestMetadata) {
        b.add_string(vokra_core::gguf::chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        b.add_string(vokra_core::gguf::chunks::KEY_MODEL_NAME, NAME);
        b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
        b.add_u32(GGUF_KEY_SAMPLE_RATE, DEFAULT_SAMPLE_RATE);
        b.add_u32(GGUF_KEY_SINCNET_STRIDE, DEFAULT_SINCNET_STRIDE);
        b.add_u32(GGUF_KEY_LSTM_HIDDEN_SIZE, DEFAULT_LSTM_HIDDEN_SIZE);
        b.add_u32(
            GGUF_KEY_LSTM_NUM_LAYERS,
            match metadata {
                TestMetadata::Legacy => 2,
                TestMetadata::Canonical | TestMetadata::PartialCanonical => DEFAULT_LSTM_NUM_LAYERS,
            },
        );
        b.add_bool(GGUF_KEY_LSTM_BIDIRECTIONAL, DEFAULT_LSTM_BIDIRECTIONAL);
        b.add_bool(GGUF_KEY_LSTM_MONOLITHIC, DEFAULT_LSTM_MONOLITHIC);
        b.add_u32(GGUF_KEY_LINEAR_HIDDEN_SIZE, DEFAULT_LINEAR_HIDDEN_SIZE);
        b.add_u32(GGUF_KEY_LINEAR_NUM_LAYERS, DEFAULT_LINEAR_NUM_LAYERS);
        b.add_u32(GGUF_KEY_NUM_POWERSET_CLASSES, DEFAULT_NUM_POWERSET_CLASSES);

        let source = match metadata {
            TestMetadata::Legacy => LEGACY_SOURCE,
            TestMetadata::Canonical | TestMetadata::PartialCanonical => CANONICAL_SOURCE,
        };
        vokra_core::stamp_provenance(
            b,
            LicenseClass::Permissive,
            DEFAULT_LICENSE,
            Some(NAME),
            Some(source),
        );

        if !matches!(metadata, TestMetadata::Legacy) {
            b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
            b.add_string(KEY_PROVENANCE_UPSTREAM_REVISION, UPSTREAM_REVISION);
            b.add_string(GGUF_KEY_UPSTREAM_REVISION, UPSTREAM_REVISION);
            b.add_string(GGUF_KEY_PYANNOTE_AUDIO_VERSION, PYANNOTE_AUDIO_VERSION);
            b.add_string(GGUF_KEY_PYANNOTE_AUDIO_REVISION, PYANNOTE_AUDIO_REVISION);
            b.add_string(GGUF_KEY_MANIFEST_SHA256, MANIFEST_SHA256);
            b.add_string(GGUF_KEY_PUBLIC_HF, PUBLIC_HF);
            b.add_string(GGUF_KEY_PUBLIC_REVISION, PUBLIC_REVISION);
            b.add_string(GGUF_KEY_PUBLIC_FILE, PUBLIC_FILE);
            b.add_u32(GGUF_KEY_PUBLIC_BYTES, PUBLIC_BYTES);
            if matches!(metadata, TestMetadata::Canonical) {
                b.add_string(GGUF_KEY_PUBLIC_SHA256, PUBLIC_SHA256);
            }
        }
    }

    /// A GGUF with the exact 54-tensor released PyanNet manifest: SincNet
    /// learned parameters and persistent buffers, four-layer bidirectional
    /// LSTM (32 tensors), two projection layers, and the classifier. Payloads
    /// are deterministic synthetic f32 values, so this proves serialization,
    /// binding, routing and output invariants rather than upstream numerical
    /// parity.
    ///
    /// This fixture is enough to drive [`PyanNet::segment`] end to end. Real
    /// numeric parity requires the official pyannote.audio reference and the
    /// immutable public checkpoint on VAST.
    pub(crate) fn synthetic_full_pyannet_gguf() -> Vec<u8> {
        synthetic_full_pyannet_gguf_with_metadata(TestMetadata::Canonical)
    }

    fn synthetic_full_pyannet_gguf_with_metadata(metadata: TestMetadata) -> Vec<u8> {
        use crate::pyannote::sincnet::{
            CONV_KERNEL_LATER, CONV1_IN_CH, CONV1_OUT_CH, CONV2_IN_CH, CONV2_OUT_CH, N_FILTERS_SINC,
        };
        let mut b = GgufBuilder::new();
        stamp_test_metadata(&mut b, metadata);

        // Helper closures to keep the tensor list readable.
        let add_scalar = |b: &mut GgufBuilder, name: &str, shape: Vec<u64>, val: f32| {
            let elems: u64 = shape.iter().product();
            let bytes: Vec<u8> = (0..elems as usize)
                .flat_map(|_| val.to_le_bytes())
                .collect();
            b.add_tensor(name, GgmlType::F32, shape, bytes).unwrap();
        };
        let add_ramp = |b: &mut GgufBuilder, name: &str, shape: Vec<u64>, scale: f32| {
            let elems: u64 = shape.iter().product();
            let bytes: Vec<u8> = (0..elems as usize)
                .flat_map(|i| ((i as f32) * scale).to_le_bytes())
                .collect();
            b.add_tensor(name, GgmlType::F32, shape, bytes).unwrap();
        };

        // --- SincNet learnable filters ---
        let n_learn = (N_FILTERS_SINC / 2) as u64;
        add_ramp(
            &mut b,
            "sincnet.conv1d.0.filterbank.low_hz_",
            vec![n_learn, 1],
            100.0,
        );
        add_scalar(
            &mut b,
            "sincnet.conv1d.0.filterbank.band_hz_",
            vec![n_learn, 1],
            100.0,
        );
        add_scalar(&mut b, "sincnet.conv1d.0.filterbank.n_", vec![1, 125], 0.0);
        add_scalar(
            &mut b,
            "sincnet.conv1d.0.filterbank.window_",
            vec![125],
            0.0,
        );
        // --- SincNet affine (identity: γ=1, β=0) ---
        add_scalar(&mut b, "sincnet.wav_norm1d.weight", vec![1], 1.0);
        add_scalar(&mut b, "sincnet.wav_norm1d.bias", vec![1], 0.0);
        for (name, c) in [
            ("sincnet.norm1d.0.weight", N_FILTERS_SINC),
            ("sincnet.norm1d.1.weight", CONV1_OUT_CH),
            ("sincnet.norm1d.2.weight", CONV2_OUT_CH),
        ] {
            add_scalar(&mut b, name, vec![c as u64], 1.0);
        }
        for (name, c) in [
            ("sincnet.norm1d.0.bias", N_FILTERS_SINC),
            ("sincnet.norm1d.1.bias", CONV1_OUT_CH),
            ("sincnet.norm1d.2.bias", CONV2_OUT_CH),
        ] {
            add_scalar(&mut b, name, vec![c as u64], 0.0);
        }
        // --- SincNet Conv1d ---
        add_ramp(
            &mut b,
            "sincnet.conv1d.1.weight",
            vec![
                CONV1_OUT_CH as u64,
                CONV1_IN_CH as u64,
                CONV_KERNEL_LATER as u64,
            ],
            0.0001,
        );
        add_scalar(
            &mut b,
            "sincnet.conv1d.1.bias",
            vec![CONV1_OUT_CH as u64],
            0.0,
        );
        add_ramp(
            &mut b,
            "sincnet.conv1d.2.weight",
            vec![
                CONV2_OUT_CH as u64,
                CONV2_IN_CH as u64,
                CONV_KERNEL_LATER as u64,
            ],
            0.0001,
        );
        add_scalar(
            &mut b,
            "sincnet.conv1d.2.bias",
            vec![CONV2_OUT_CH as u64],
            0.0,
        );

        // --- Monolithic 4-layer BiLSTM ---
        let h = DEFAULT_LSTM_HIDDEN_SIZE as usize; // 128
        let g = 4 * h; // 512
        let layer_in_dims = [CONV2_OUT_CH, 2 * h, 2 * h, 2 * h];
        for (k, &in_dim) in layer_in_dims.iter().enumerate() {
            for suffix in ["", "_reverse"] {
                add_scalar(
                    &mut b,
                    &format!("lstm.weight_ih_l{k}{suffix}"),
                    vec![g as u64, in_dim as u64],
                    0.001,
                );
                add_scalar(
                    &mut b,
                    &format!("lstm.weight_hh_l{k}{suffix}"),
                    vec![g as u64, h as u64],
                    0.001,
                );
                add_scalar(
                    &mut b,
                    &format!("lstm.bias_ih_l{k}{suffix}"),
                    vec![g as u64],
                    0.0,
                );
                add_scalar(
                    &mut b,
                    &format!("lstm.bias_hh_l{k}{suffix}"),
                    vec![g as u64],
                    0.0,
                );
            }
        }

        // --- Linear stack (2 layers) ---
        // layer 0: (128, 256), layer 1: (128, 128)
        let lin_hidden = DEFAULT_LINEAR_HIDDEN_SIZE as u64;
        add_scalar(
            &mut b,
            "linear.0.weight",
            vec![lin_hidden, (2 * h) as u64],
            0.001,
        );
        add_scalar(&mut b, "linear.0.bias", vec![lin_hidden], 0.0);
        add_scalar(
            &mut b,
            "linear.1.weight",
            vec![lin_hidden, lin_hidden],
            0.001,
        );
        add_scalar(&mut b, "linear.1.bias", vec![lin_hidden], 0.0);

        // --- Classifier ---
        add_scalar(
            &mut b,
            "classifier.weight",
            vec![DEFAULT_NUM_POWERSET_CLASSES as u64, lin_hidden],
            0.001,
        );
        add_scalar(
            &mut b,
            "classifier.bias",
            vec![DEFAULT_NUM_POWERSET_CLASSES as u64],
            0.0,
        );

        b.to_bytes().expect("gguf serialize (full)")
    }

    #[test]
    fn config_default_matches_primary_source() {
        let c = PyanNetConfig::default();
        assert_eq!(c.sample_rate, 16000);
        assert_eq!(c.sincnet_stride, 10);
        assert_eq!(c.lstm_hidden_size, 128);
        assert_eq!(c.lstm_num_layers, 4);
        assert!(c.lstm_bidirectional);
        assert!(c.lstm_monolithic);
        assert_eq!(c.linear_hidden_size, 128);
        assert_eq!(c.linear_num_layers, 2);
        assert_eq!(c.num_powerset_classes, 7);
    }

    #[test]
    fn config_from_gguf_round_trips_the_converter_chunk_group() {
        let bytes = synthetic_pyannet_gguf();
        let path = scratch_path("config-roundtrip");
        std::fs::write(&path, &bytes).unwrap();
        let g = GgufFile::open(&path).unwrap();

        let c = PyanNetConfig::from_gguf(&g);
        assert_eq!(c, PyanNetConfig::default(), "chunk round-trip");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn config_from_gguf_falls_back_to_defaults_when_chunk_absent() {
        // A GGUF with no `vokra.pyannote.*` chunks at all — the
        // fallback path must yield the primary-source Default.
        let mut b = GgufBuilder::new();
        // Non-empty tensors so the file is a valid GGUF; the runtime
        // config parser must NOT depend on tensor presence.
        b.add_tensor(
            "sincnet.conv1d.0.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .unwrap();
        let bytes = b.to_bytes().unwrap();
        let path = scratch_path("config-fallback");
        std::fs::write(&path, &bytes).unwrap();
        let g = GgufFile::open(&path).unwrap();

        let c = PyanNetConfig::from_gguf(&g);
        assert_eq!(c, PyanNetConfig::default(), "fallback to Default");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn weights_from_gguf_rejects_gguf_without_arch_stamp() {
        // FR-EX-08: an unstamped GGUF is not a Vokra-native PyanNet
        // artifact. Without this gate a foreign checkpoint carrying an
        // `lstm.*` tensor would clear the non-emptiness gate and bind a
        // partial, meaningless model.
        let mut b = GgufBuilder::new();
        b.add_tensor(
            "lstm.weight_ih_l0",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .unwrap();
        let path = scratch_path("no-arch");
        std::fs::write(&path, b.to_bytes().unwrap()).unwrap();
        let g = GgufFile::open(&path).unwrap();

        let err = PyanNetWeights::from_gguf(&g).unwrap_err();
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains(vokra_core::gguf::chunks::KEY_MODEL_ARCH),
                    "message must name the missing key: {msg}"
                );
                assert!(
                    msg.contains(EXPECTED_ARCH),
                    "message must name the expected arch: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn weights_from_gguf_rejects_foreign_arch_naming_expected_and_actual() {
        // `pyannote-speaker-diarization` is the most plausible mis-route:
        // same upstream org, same `vokra.pyannote.*`-adjacent naming, but
        // a weightless pipeline orchestrator rather than this backbone.
        let mut b = GgufBuilder::new();
        b.add_string(
            vokra_core::gguf::chunks::KEY_MODEL_ARCH,
            "pyannote-speaker-diarization",
        );
        b.add_tensor(
            "lstm.weight_ih_l0",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .unwrap();
        let path = scratch_path("foreign-arch");
        std::fs::write(&path, b.to_bytes().unwrap()).unwrap();
        let g = GgufFile::open(&path).unwrap();

        let err = PyanNetWeights::from_gguf(&g).unwrap_err();
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("pyannote-speaker-diarization"),
                    "message must name the actual arch: {msg}"
                );
                assert!(
                    msg.contains(EXPECTED_ARCH),
                    "message must name the expected arch: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn weights_from_gguf_binds_all_recognized_prefixes() {
        let bytes = synthetic_pyannet_gguf();
        let path = scratch_path("weights-bind");
        std::fs::write(&path, &bytes).unwrap();
        let g = GgufFile::open(&path).unwrap();

        let w = PyanNetWeights::from_gguf(&g).expect("bind");
        // 4 tensors written = 4 recognized (all match the prefixes).
        assert_eq!(w.tensor_count(), 4, "every prefix must be bound");
        assert!(w.tensor("sincnet.conv1d.0.weight").is_some());
        assert!(w.tensor("lstm.weight_ih_l0").is_some());
        assert!(w.tensor("linear.0.weight").is_some());
        assert!(w.tensor("classifier.weight").is_some());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn weights_from_gguf_refuses_empty_manifest_loudly() {
        // A correctly-stamped GGUF with a tensor whose name matches none
        // of the required prefixes — the binder must refuse loudly
        // (FR-EX-08). The arch stamp keeps this test on the *tensor*
        // gate rather than short-circuiting at the arch gate.
        let mut b = GgufBuilder::new();
        b.add_string(vokra_core::gguf::chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        b.add_tensor(
            "some_unrelated_name.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .unwrap();
        let bytes = b.to_bytes().unwrap();
        let path = scratch_path("weights-refuse");
        std::fs::write(&path, &bytes).unwrap();
        let g = GgufFile::open(&path).unwrap();

        let err = PyanNetWeights::from_gguf(&g).unwrap_err();
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("sincnet.") && msg.contains("FR-EX-08"),
                    "error must name the required prefix + FR-EX-08 tag: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn pyannet_from_gguf_loads_and_config_is_real() {
        let bytes = synthetic_full_pyannet_gguf();
        let path = scratch_path("engine-load");
        std::fs::write(&path, &bytes).unwrap();

        let p = PyanNet::from_gguf(&path).expect("load");
        assert_eq!(p.config(), &PyanNetConfig::default());
        assert_eq!(p.model_name(), NAME);
        assert_eq!(p.weight_license(), LicenseClass::Permissive);
        assert!(!p.legacy_metadata_repaired());

        // Receptive-field arithmetic is real via the primary-source
        // `multi_conv_num_frames` recurrence (Wave 3 landing —
        // supersedes the Wave 2 `num_samples / stride` stub which
        // over-counted by ~28×). Values pin-tested in the sibling
        // `sincnet::tests::num_frames_matches_primary_source_recurrence`.
        assert_eq!(p.num_frames(160_000), 589, "10 s @ 16 kHz");
        assert_eq!(p.num_frames(16_000), 56, "1 s @ 16 kHz");
        assert_eq!(p.num_frames(0), 0);
        assert_eq!(p.num_frames(9), 0, "sub-kernel input yields 0 frames");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn historical_public_metadata_is_narrowly_repaired_to_four_layers() {
        let bytes = synthetic_full_pyannet_gguf_with_metadata(TestMetadata::Legacy);
        let path = scratch_path("legacy-metadata-repair");
        std::fs::write(&path, &bytes).unwrap();

        let p = PyanNet::from_gguf(&path).expect("load");
        assert_eq!(p.config().lstm_num_layers, 4);
        assert!(p.legacy_metadata_repaired());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn partial_immutable_metadata_is_rejected() {
        let bytes = synthetic_full_pyannet_gguf_with_metadata(TestMetadata::PartialCanonical);
        let path = scratch_path("partial-metadata");
        std::fs::write(&path, &bytes).unwrap();

        let err = PyanNet::from_gguf(&path).unwrap_err();
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("partial immutable metadata") && msg.contains("8/9"),
                    "error must name the incomplete identity group: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn segment_is_default_on_and_produces_probability_matrix() {
        // The public method executes directly: there is no hidden environment
        // opt-in. Synthetic values prove routing and probability invariants;
        // the VAST suite supplies independent official numeric parity.
        let bytes = synthetic_full_pyannet_gguf();
        let path = scratch_path("segment-default-on");
        std::fs::write(&path, &bytes).unwrap();

        let p = PyanNet::from_gguf(&path).expect("load");
        // 0.1 s of 16 kHz sine at 440 Hz.
        let sr = DEFAULT_SAMPLE_RATE as f32;
        let pcm: Vec<f32> = (0..1_600)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / sr).sin())
            .collect();
        let out = p.segment(&pcm).expect("default-on segment forward");
        // Frame count matches the SincNet recurrence.
        assert_eq!(out.len(), p.num_frames(pcm.len()));
        for row in &out {
            assert_eq!(row.len(), DEFAULT_NUM_POWERSET_CLASSES as usize);
            let sum: f32 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "softmax row must sum to ~1, got {sum}"
            );
            for &v in row {
                assert!((0.0..=1.0).contains(&v), "prob out of range: {v}");
                assert!(v.is_finite(), "non-finite prob: {v}");
            }
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn segment_powerset_emits_speaker_activity_per_frame() {
        let bytes = synthetic_full_pyannet_gguf();
        let path = scratch_path("segment-powerset");
        std::fs::write(&path, &bytes).unwrap();

        let p = PyanNet::from_gguf(&path).expect("load");
        let sr = DEFAULT_SAMPLE_RATE as f32;
        let pcm: Vec<f32> = (0..1_600)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / sr).sin())
            .collect();

        let activity = p.segment_powerset(&pcm).expect("segment_powerset");

        assert_eq!(activity.len(), p.num_frames(pcm.len()));
        for a in &activity {
            // Every active-speaker set is a subset of {0, 1, 2} of
            // size ≤ 2 (powerset with num_classes=3, max_set_size=2).
            assert!(a.active_speakers.len() <= 2);
            for &spk in &a.active_speakers {
                assert!(spk < 3);
            }
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn unsupported_backend_is_explicit_and_never_falls_back() {
        let bytes = synthetic_full_pyannet_gguf();
        let path = scratch_path("unsupported-backend");
        std::fs::write(&path, &bytes).unwrap();

        let err = PyanNet::from_gguf_with_backend(&path, BackendKind::Vulkan).unwrap_err();
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains("Vulkan") && msg.contains("no CPU fallback"),
                    "backend refusal must name the selection and no-fallback rule: {msg}"
                );
            }
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn decode_powerset_maps_argmax_row_to_active_speakers() {
        // Argmax at index 4 -> {0, 1} overlap per POWERSET_MAPPING.
        // Argmax at index 6 -> {1, 2} overlap.
        // Argmax at index 0 -> silence.
        let probs = vec![
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], // A+B
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], // B+C
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // silence
        ];
        let out = decode_powerset(&probs, 7, 16_000, 10);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].active_speakers, vec![0, 1]);
        assert_eq!(out[1].active_speakers, vec![1, 2]);
        assert_eq!(out[2].active_speakers, Vec::<usize>::new());
        // Time in seconds: frame_idx=0 -> 0.5·(10/16000) = 0.0003125 s.
        assert!((out[0].time_s - 0.0003125).abs() < 1e-6);
        // Frame index preserved.
        assert_eq!(out[2].frame_idx, 2);
    }

    #[test]
    fn powerset_mapping_matches_primary_source_transcription() {
        // Verbatim transcription from `powerset.py:69-108` — pin-test
        // the 7-row 3-column mapping matrix. Any regression means a
        // decoder that maps to the wrong speaker set.
        assert_eq!(
            POWERSET_MAPPING_3SPK_2OVERLAP,
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
            ]
        );
    }

    #[test]
    fn sincnet_output_features_constant_matches_primary_source() {
        // PyanNet.py L96: `self.lstm = nn.LSTM(60, **multi_layer_lstm)`
        // — the LSTM input dim is 60 because SincNet emits 60 features.
        // Any drift in this constant would break the future SincNet
        // primitive's shape contract with the BiLSTM.
        assert_eq!(SINCNET_OUTPUT_FEATURES, 60);
    }

    // -----------------------------------------------------------------------
    // Load-time shape gate — FQ-05 coverage
    // -----------------------------------------------------------------------
    //
    // These tests exercise the [`PyanNetWeights::verify_core_shapes`]
    // component gate used by the SincNet-forward-time shape assertions. The
    // public [`PyanNet::from_gguf`] binder is stronger: it pins the complete
    // 54-tensor manifest before decoding any payload.
    //
    // Coverage plan (matches the FQ-05 gap description — four core
    // tensors that a real PyanNet-3.0 checkpoint MUST carry):
    //   * sincnet.conv1d.0.filterbank.low_hz_  (SincNet learnable sinc)
    //   * lstm.weight_ih_l0                     (monolithic BiLSTM)
    //   * linear.0.weight                       (Linear stack)
    //   * classifier.weight                     (terminal classifier)

    /// Builds a 4-core-tensor GGUF at the primary-source PyanNet-3.0
    /// shapes; optionally overrides (or drops) one tensor to synthesise
    /// a shape-drifted / incomplete real fixture for the load-time gate
    /// tests. Pass `None` for `override_shape` to drop the tensor.
    fn pyannet_gguf_with_core_override(
        override_key: &str,
        override_shape: Option<Vec<u64>>,
    ) -> Vec<u8> {
        let mut b = GgufBuilder::new();
        // Arch stamp — these fixtures must reach the *shape* gates, so
        // they cannot short-circuit at the arch gate (FR-EX-08).
        b.add_string(vokra_core::gguf::chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        b.add_u32(GGUF_KEY_SAMPLE_RATE, DEFAULT_SAMPLE_RATE);
        b.add_u32(GGUF_KEY_SINCNET_STRIDE, DEFAULT_SINCNET_STRIDE);
        b.add_u32(GGUF_KEY_LSTM_HIDDEN_SIZE, DEFAULT_LSTM_HIDDEN_SIZE);
        b.add_u32(GGUF_KEY_LSTM_NUM_LAYERS, DEFAULT_LSTM_NUM_LAYERS);
        b.add_bool(GGUF_KEY_LSTM_BIDIRECTIONAL, DEFAULT_LSTM_BIDIRECTIONAL);
        b.add_bool(GGUF_KEY_LSTM_MONOLITHIC, DEFAULT_LSTM_MONOLITHIC);
        b.add_u32(GGUF_KEY_LINEAR_HIDDEN_SIZE, DEFAULT_LINEAR_HIDDEN_SIZE);
        b.add_u32(GGUF_KEY_LINEAR_NUM_LAYERS, DEFAULT_LINEAR_NUM_LAYERS);
        b.add_u32(GGUF_KEY_NUM_POWERSET_CLASSES, DEFAULT_NUM_POWERSET_CLASSES);

        // Primary-source PyanNet-3.0 core tensor shapes (see
        // PyanNetWeights::verify_core_shapes doc for the derivations).
        let core: [(&str, Vec<u64>); 4] = [
            ("sincnet.conv1d.0.filterbank.low_hz_", vec![40, 1]),
            ("lstm.weight_ih_l0", vec![512, 60]),
            ("linear.0.weight", vec![128, 256]),
            ("classifier.weight", vec![7, 128]),
        ];

        for (name, shape) in &core {
            let (final_shape, drop) = if *name == override_key {
                match &override_shape {
                    Some(s) => (s.clone(), false),
                    None => (shape.clone(), true),
                }
            } else {
                (shape.clone(), false)
            };
            if drop {
                continue;
            }
            let elems: u64 = final_shape.iter().product();
            let bytes: Vec<u8> = (0..elems as usize)
                .flat_map(|i| (i as f32 * 0.001).to_le_bytes())
                .collect();
            b.add_tensor(name, GgmlType::F32, final_shape, bytes)
                .expect("add_tensor");
        }
        b.to_bytes().expect("gguf serialize")
    }

    fn verify_core_override(
        override_key: &str,
        override_shape: Option<Vec<u64>>,
    ) -> Result<(), VokraError> {
        let gguf = GgufFile::parse(pyannet_gguf_with_core_override(
            override_key,
            override_shape,
        ))?;
        let weights = PyanNetWeights::from_gguf(&gguf)?;
        let config = PyanNetConfig::from_gguf(&gguf);
        weights.verify_core_shapes(&config)
    }

    #[test]
    fn pyannet_from_gguf_rejects_wrong_filterbank_shape_at_load_time() {
        // Real-GGUF sentinel present (`sincnet.conv1d.0.filterbank.low_hz_`)
        // at the WRONG shape [10, 1] instead of the primary-source [40, 1].
        // The component gate must reject this before a primitive can execute.
        let err = verify_core_override("sincnet.conv1d.0.filterbank.low_hz_", Some(vec![10, 1]));
        match err.unwrap_err() {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("sincnet.conv1d.0.filterbank.low_hz_") && msg.contains("FR-EX-08"),
                    "load-time gate must name the offending tensor + FR-EX-08: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn pyannet_from_gguf_rejects_wrong_lstm_shape_at_load_time() {
        // lstm.weight_ih_l0 must be [4·H, SincNet.out] = [512, 60] per
        // PyTorch nn.LSTM's `(gates * hidden, input)` layout. A drifted
        // [64, 60] fixture must be caught at load time.
        let err = verify_core_override("lstm.weight_ih_l0", Some(vec![64, 60]));
        match err.unwrap_err() {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("lstm.weight_ih_l0") && msg.contains("FR-EX-08"),
                    "load-time gate must name lstm.weight_ih_l0 + FR-EX-08: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn pyannet_from_gguf_rejects_wrong_linear_shape_at_load_time() {
        // linear.0.weight must be [linear_h, 2·H] = [128, 256] per the
        // primary-source Linear stack (bidirectional BiLSTM output is
        // concatenated → 2·H channels feed the first Linear). A drifted
        // [64, 256] fixture must be caught at load time.
        let err = verify_core_override("linear.0.weight", Some(vec![64, 256]));
        match err.unwrap_err() {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("linear.0.weight") && msg.contains("FR-EX-08"),
                    "load-time gate must name linear.0.weight + FR-EX-08: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn pyannet_from_gguf_rejects_wrong_classifier_shape_at_load_time() {
        // classifier.weight must be [n_powerset_classes, linear_h] =
        // [7, 128] for pyannote/segmentation-3.0. A drifted [3, 128]
        // (e.g. a `speaker-diarization` variant leaked into a segmentation
        // GGUF) must be caught at load time so decode_powerset does not
        // hit an argmax over a wrong-cardinality row.
        let err = verify_core_override("classifier.weight", Some(vec![3, 128]));
        match err.unwrap_err() {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("classifier.weight") && msg.contains("FR-EX-08"),
                    "load-time gate must name classifier.weight + FR-EX-08: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn pyannet_from_gguf_rejects_incomplete_real_manifest_loudly() {
        // Sentinel present (`sincnet.conv1d.0.filterbank.low_hz_` at
        // the correct [40, 1] shape) but the LSTM core tensor is
        // dropped entirely. The sentinel-gated strict mode must name
        // both the missing tensor AND the sentinel that triggered the
        // co-presence gate (so the caller knows *why* the check fired).
        let err = verify_core_override("lstm.weight_ih_l0", None);
        match err.unwrap_err() {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("lstm.weight_ih_l0")
                        && msg.contains("sincnet.conv1d.0.filterbank.low_hz_")
                        && msg.contains("FR-EX-08"),
                    "co-presence error must name missing tensor + sentinel + FR-EX-08: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn verify_core_shapes_permissive_when_sentinel_absent() {
        // Illustrative fixture — no filterbank sentinel, all other core
        // tensor shapes CORRECT. verify_core_shapes must return Ok so
        // the binder / plumbing smoke-test surface (mod.rs's
        // `synthetic_pyannet_gguf` and diarization.rs's
        // `local_synthetic_pyannet_gguf`) keeps working. Silent-fake is
        // still impossible because the downstream forward loud-fails
        // via SincNet::from_weights.
        let bytes = synthetic_pyannet_gguf();
        let path = scratch_path("permissive-no-sentinel");
        std::fs::write(&path, &bytes).unwrap();
        let g = GgufFile::open(&path).unwrap();
        let w = PyanNetWeights::from_gguf(&g).expect("bind");
        let cfg = PyanNetConfig::from_gguf(&g);

        // Pin the illustrative-fixture invariant so a future change to
        // synthetic_pyannet_gguf does not silently promote the fixture
        // to real-GGUF status without co-updating the load-time gate
        // contract.
        assert!(
            w.tensor("sincnet.conv1d.0.filterbank.low_hz_").is_none(),
            "synthetic_pyannet_gguf must remain illustrative (no sentinel)"
        );

        w.verify_core_shapes(&cfg)
            .expect("permissive pass-through when sentinel is absent");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn verify_core_shapes_rejects_wrong_shape_even_without_sentinel() {
        // Present-but-mis-shaped is ALWAYS rejected loudly, regardless
        // of sentinel presence — a wrong-shape core tensor is a
        // silent-fake risk even in an "illustrative" fixture. This is
        // the belt-and-braces half of the sentinel gate.
        let mut b = GgufBuilder::new();
        // Arch stamp — required to reach `verify_core_shapes` at all.
        b.add_string(vokra_core::gguf::chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        b.add_u32(GGUF_KEY_SAMPLE_RATE, DEFAULT_SAMPLE_RATE);
        b.add_u32(GGUF_KEY_SINCNET_STRIDE, DEFAULT_SINCNET_STRIDE);
        b.add_u32(GGUF_KEY_LSTM_HIDDEN_SIZE, DEFAULT_LSTM_HIDDEN_SIZE);
        b.add_u32(GGUF_KEY_LSTM_NUM_LAYERS, DEFAULT_LSTM_NUM_LAYERS);
        b.add_bool(GGUF_KEY_LSTM_BIDIRECTIONAL, DEFAULT_LSTM_BIDIRECTIONAL);
        b.add_bool(GGUF_KEY_LSTM_MONOLITHIC, DEFAULT_LSTM_MONOLITHIC);
        b.add_u32(GGUF_KEY_LINEAR_HIDDEN_SIZE, DEFAULT_LINEAR_HIDDEN_SIZE);
        b.add_u32(GGUF_KEY_LINEAR_NUM_LAYERS, DEFAULT_LINEAR_NUM_LAYERS);
        b.add_u32(GGUF_KEY_NUM_POWERSET_CLASSES, DEFAULT_NUM_POWERSET_CLASSES);
        // No filterbank sentinel (illustrative). But linear.0.weight
        // present with the WRONG shape.
        let bad_elems = 64usize * 256;
        b.add_tensor(
            "linear.0.weight",
            GgmlType::F32,
            vec![64, 256],
            vec![0u8; bad_elems * 4],
        )
        .unwrap();
        let bytes = b.to_bytes().unwrap();
        let path = scratch_path("wrong-shape-no-sentinel");
        std::fs::write(&path, &bytes).unwrap();
        let g = GgufFile::open(&path).unwrap();
        let w = PyanNetWeights::from_gguf(&g).expect("bind");
        let cfg = PyanNetConfig::from_gguf(&g);

        let err = w.verify_core_shapes(&cfg).unwrap_err();
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("linear.0.weight") && msg.contains("FR-EX-08"),
                    "present-but-mis-shaped rejection must name tensor + FR-EX-08: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }

        std::fs::remove_file(&path).ok();
    }
}
