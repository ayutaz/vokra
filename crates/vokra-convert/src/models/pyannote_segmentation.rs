//! **pyannote/segmentation-3.0** (Bredin, CNRS, MIT): safetensors → GGUF
//! conversion (VAD / speaker-segmentation tier, 2026-07-30).
//!
//! Input: an offline `.bin` → safetensors flattening of the immutable
//! upstream `pyannote/segmentation-3.0` checkpoint. Output: a GGUF carrying
//! the exact 54-tensor F32 inference manifest under the upstream state-dict
//! names, plus a `vokra.pyannote.*` contract consumed by the native PyanNet
//! binder in `crates/vokra-models/src/pyannote/`. Foreign, incomplete, or
//! dtype-drifted checkpoints fail closed instead of producing a plausible
//! but unrunnable artifact.
//!
//! # HF / licence / category
//!
//! - Upstream HF: `pyannote/segmentation-3.0` (recorded under
//!   `vokra.provenance.upstream_hf`).
//! - SPDX: **`mit`** (`LicenseClass::Permissive`). CC 直接照合
//!   2026-07-30, authenticated HF API `api/models/pyannote/segmentation-3.0`
//!   = `license: mit, gated: auto` — `gated: auto` は access control
//!   のみで追加条項なし。`docs/license-audit.md` §3.1 row 263 で
//!   2026-07-30 yousan sign-off。
//! - Model category: `vad` (recorded under `vokra.model.category`).
//!   segmentation-3.0 は voice-activity-detection / speaker-segmentation
//!   backbone — `pyannote/speaker-diarization-3.1` pipeline はこの
//!   segmentation backbone + speaker embedding + clustering から成る
//!   (pipeline は Vokra native re-imp、CAM++ 経路と組み合わせる)。
//!
//! # PyanNet architecture (primary source: MIT LICENSE)
//!
//! Source: `github.com/pyannote/pyannote-audio/blob/3.0.0/pyannote/`
//! `audio/models/segmentation/PyanNet.py` (MIT). The class default is two
//! recurrent layers, while this exact release overrides it to four. Both the
//! preserved model config and the `lstm.*_l0..l3{,_reverse}` tensor manifest
//! independently pin that release-specific override.
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
//!      - num_powerset_classes = 7 for segmentation-3.0 (3 speakers × 2 overlap)
//!   -> Activation (Softmax for powerset multiclass)
//! ```
//!
//! # Wiring status
//!
//! Strict release converter. Exactly 54 F32 tensors with the immutable public
//! shape manifest are accepted and copied byte-for-byte. The
//! `vokra.pyannote.*` group records the release topology, upstream source tag,
//! public Vokra artifact identity, and manifest digest so the runtime can
//! distinguish this checkpoint from superficially similar PyanNet variants.
//!
//! # Runtime binder / real forward
//!
//! The runtime contains the native SincNet, four-layer bidirectional LSTM,
//! two-layer projection, and powerset classifier. Execution is default-on for
//! CPU and Metal after this exact manifest binds; unsupported backends fail
//! explicitly. Independent upstream probability parity remains a separate
//! VAST gate and is not implied by the synthetic converter tests.
//!
//! # No ONNX (permanent)
//!
//! pyannote is distributed as torch `.bin` (pickle) + `config.yaml`;
//! this converter **never** touches ONNX (FR-LD-05). The `.bin` →
//! safetensors bridge lives in `tools/parity/bin_to_safetensors.py`
//! (an offline side-car tool, not part of the runtime).

use std::collections::BTreeMap;
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for pyannote-segmentation GGUFs. Distinct arch
/// tag because PyanNet is the first `category = "vad"` binder in the
/// converter tree — silently sharing an arch tag would misroute the
/// runtime dispatch (a speaker-encoder or ASR backbone would try to
/// interpret the 7-class powerset segmentation head).
pub const ARCH: &str = "pyannote-segmentation";

/// `vokra.model.name` value written for the canonical
/// segmentation-3.0 GGUF.
pub const NAME: &str = "pyannote-segmentation-3.0";

/// `vokra.model.category` value — the first `"vad"` in the converter
/// tree. Consumed by the model-card generator + zoo manifest tier
/// gate so a VAD / segmentation backbone is not accidentally
/// advertised as an ASR / TTS / speaker release. Distinct from the
/// `silero-vad` category which occupies the same slot for its
/// end-to-end voice-activity-detection SKU; both are `vad` because
/// downstream consumers care about the semantic role (VAD vs. ASR
/// vs. TTS vs. speaker embedding vs. F0), not the specific
/// implementation family.
pub const CATEGORY: &str = "vad";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf`. Preserves upstream casing.
pub const UPSTREAM_HF: &str = "pyannote/segmentation-3.0";

/// Immutable upstream model revision reported by the official HF API.
pub const UPSTREAM_REVISION: &str = "e66f3d3b9eb0873085418a7b813d3b369bf160bb";
/// Official pyannote.audio source tag used to train and load this release.
pub const PYANNOTE_AUDIO_VERSION: &str = "3.0.0";
/// Peeled official `3.0.0` source tag revision.
pub const PYANNOTE_AUDIO_REVISION: &str = "795b92ab265888c58d160f90ae4d91b7bcc6aa2c";
/// Exact historical public Vokra artifact repository.
pub const PUBLIC_HF: &str = "vokra/pyannote-segmentation-3.0";
/// Immutable public Vokra artifact revision.
pub const PUBLIC_REVISION: &str = "50bf4e510e0c689668384aec0f866f02e0fcaea8";
/// Exact public GGUF filename.
pub const PUBLIC_FILE: &str = "pyannote-seg.gguf";
/// Exact public GGUF byte size.
pub const PUBLIC_BYTES: u32 = 5_898_272;
/// Exact public GGUF SHA-256.
pub const PUBLIC_SHA256: &str = "22ff05fddf19e69c8d9aac8daa6d99014e6718bcd8d8c527d26da677d00c63f1";
/// Complete sorted `(tensor name, shape)` manifest SHA-256.
pub const MANIFEST_SHA256: &str =
    "a1c783d4df253742ad5e0e796402310930f52b1a80597420f79a6eba830670d8";
/// Exact inference tensor count.
pub const TENSOR_COUNT: usize = 54;

/// Canonical weight license SPDX (`mit`). A matching explicit value is
/// accepted for CLI compatibility; conflicting overrides fail closed.
pub const DEFAULT_LICENSE: &str = "mit";

/// Ad-hoc metadata key for the model category. Same key that
/// `wespeaker` / `rmvpe` / `emotion2vec` use (they share the same
/// `vokra.model.category` chunk namespace).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

// GGUF metadata keys for the PyanNet hparam chunk group. Kept in sync
// with the future runtime consumer `vokra_models::pyannote::{
// PyanNetConfig::from_gguf, GGUF_KEY_SAMPLE_RATE, ...}`.
pub(crate) const KEY_SAMPLE_RATE: &str = "vokra.pyannote.sample_rate";
pub(crate) const KEY_SINCNET_STRIDE: &str = "vokra.pyannote.sincnet.stride";
pub(crate) const KEY_LSTM_HIDDEN_SIZE: &str = "vokra.pyannote.lstm.hidden_size";
pub(crate) const KEY_LSTM_NUM_LAYERS: &str = "vokra.pyannote.lstm.num_layers";
pub(crate) const KEY_LSTM_BIDIRECTIONAL: &str = "vokra.pyannote.lstm.bidirectional";
pub(crate) const KEY_LSTM_MONOLITHIC: &str = "vokra.pyannote.lstm.monolithic";
pub(crate) const KEY_LINEAR_HIDDEN_SIZE: &str = "vokra.pyannote.linear.hidden_size";
pub(crate) const KEY_LINEAR_NUM_LAYERS: &str = "vokra.pyannote.linear.num_layers";
pub(crate) const KEY_NUM_POWERSET_CLASSES: &str = "vokra.pyannote.num_powerset_classes";
pub(crate) const KEY_UPSTREAM_REVISION: &str = "vokra.pyannote.upstream_revision";
pub(crate) const KEY_PYANNOTE_AUDIO_VERSION: &str = "vokra.pyannote.pyannote_audio_version";
pub(crate) const KEY_PYANNOTE_AUDIO_REVISION: &str = "vokra.pyannote.pyannote_audio_revision";
pub(crate) const KEY_MANIFEST_SHA256: &str = "vokra.pyannote.tensor_manifest_sha256";
pub(crate) const KEY_PUBLIC_HF: &str = "vokra.pyannote.public_hf";
pub(crate) const KEY_PUBLIC_REVISION: &str = "vokra.pyannote.public_revision";
pub(crate) const KEY_PUBLIC_FILE: &str = "vokra.pyannote.public_file";
pub(crate) const KEY_PUBLIC_BYTES: &str = "vokra.pyannote.public_bytes";
pub(crate) const KEY_PUBLIC_SHA256: &str = "vokra.pyannote.public_sha256";

// Canonical release hparams. Most values equal the PyanNet 3.0.0 class
// defaults; `lstm.num_layers = 4` is the exact checkpoint config override and
// is independently proven by the l0..l3 state-dict tensors. The seven output
// classes are the released three-speaker powerset encoding. These values are
// an identity contract, not permissive fallbacks for hand-crafted files.
pub const DEFAULT_SAMPLE_RATE: u32 = 16000;
pub const DEFAULT_SINCNET_STRIDE: u32 = 10;
pub const DEFAULT_LSTM_HIDDEN_SIZE: u32 = 128;
pub const DEFAULT_LSTM_NUM_LAYERS: u32 = 4;
pub const DEFAULT_LSTM_BIDIRECTIONAL: bool = true;
pub const DEFAULT_LSTM_MONOLITHIC: bool = true;
pub const DEFAULT_LINEAR_HIDDEN_SIZE: u32 = 128;
pub const DEFAULT_LINEAR_NUM_LAYERS: u32 = 2;
pub const DEFAULT_NUM_POWERSET_CLASSES: u32 = 7;

/// Outcome of a pyannote-segmentation conversion.
///
/// A successful conversion always reports the exact 54 F32 tensors. The two
/// legacy counters remain for API compatibility and are always zero.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PyannoteSegmentationReport {
    /// Total tensors seen in the validated safetensors header.
    pub read: usize,
    /// Canonical F32 tensors written verbatim.
    pub written: usize,
    /// Always zero after a successful strict conversion.
    pub skipped_non_float: usize,
    /// Always zero because the canonical checkpoint is F32-only.
    pub bf16_passthrough: usize,
}

/// Reads a safetensors checkpoint at `input` and writes a
/// pyannote-segmentation GGUF to `output`.
///
/// The exact 54-tensor F32 manifest is emitted verbatim under its upstream
/// state-dict names; the `vokra.provenance.*` +
/// `vokra.model.*` + `vokra.pyannote.*` chunk groups pin the upstream
/// repo, weight license, model category and PyanNet hparams so the
/// runtime binder (`crates/vokra-models/src/pyannote/`) can bring the
/// graph up without a side-car config lookup. That binder has landed:
/// `PyanNetConfig::from_gguf`, `PyanNetWeights::from_gguf` and
/// `PyanNet::from_gguf` all read these chunks for real.
///
/// `license` may be absent or match `DEFAULT_LICENSE` (`"mit"`). Any
/// conflicting value is rejected before the input is parsed.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing `output`;
/// [`ConvertError::Parse`] for malformed or non-canonical safetensors input;
/// [`ConvertError::Usage`] for a conflicting license override.
pub fn convert_pyannote_segmentation_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<PyannoteSegmentationReport, ConvertError> {
    if let Some(value) = license
        && !value.eq_ignore_ascii_case(DEFAULT_LICENSE)
    {
        return Err(ConvertError::Usage(format!(
            "pyannote-segmentation: canonical {UPSTREAM_HF}@{UPSTREAM_REVISION} has pinned MIT weights; refusing conflicting --license {value:?}"
        )));
    }

    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;
    validate_manifest(&st)?;

    let mut b = GgufBuilder::new();
    stamp_metadata(&mut b);

    let mut report = PyannoteSegmentationReport {
        read: st.tensors().len(),
        ..PyannoteSegmentationReport::default()
    };
    for t in st.tensors() {
        b.add_tensor(
            &t.name,
            t.dtype,
            t.shape.clone(),
            st.tensor_bytes(t).to_vec(),
        )?;
        report.written += 1;
    }

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Parse(e.to_string()))?;
    std::fs::write(output, out_bytes).map_err(ConvertError::Io)?;
    Ok(report)
}

fn stamp_metadata(b: &mut GgufBuilder) {
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // Exact release contract: the class default has two recurrent layers,
    // but segmentation-3.0 config + state dict both pin four.
    b.add_u32(KEY_SAMPLE_RATE, DEFAULT_SAMPLE_RATE);
    b.add_u32(KEY_SINCNET_STRIDE, DEFAULT_SINCNET_STRIDE);
    b.add_u32(KEY_LSTM_HIDDEN_SIZE, DEFAULT_LSTM_HIDDEN_SIZE);
    b.add_u32(KEY_LSTM_NUM_LAYERS, DEFAULT_LSTM_NUM_LAYERS);
    b.add_bool(KEY_LSTM_BIDIRECTIONAL, DEFAULT_LSTM_BIDIRECTIONAL);
    b.add_bool(KEY_LSTM_MONOLITHIC, DEFAULT_LSTM_MONOLITHIC);
    b.add_u32(KEY_LINEAR_HIDDEN_SIZE, DEFAULT_LINEAR_HIDDEN_SIZE);
    b.add_u32(KEY_LINEAR_NUM_LAYERS, DEFAULT_LINEAR_NUM_LAYERS);
    b.add_u32(KEY_NUM_POWERSET_CLASSES, DEFAULT_NUM_POWERSET_CLASSES);
    b.add_string(KEY_UPSTREAM_REVISION, UPSTREAM_REVISION);
    b.add_string(KEY_PYANNOTE_AUDIO_VERSION, PYANNOTE_AUDIO_VERSION);
    b.add_string(KEY_PYANNOTE_AUDIO_REVISION, PYANNOTE_AUDIO_REVISION);
    b.add_string(KEY_MANIFEST_SHA256, MANIFEST_SHA256);
    b.add_string(KEY_PUBLIC_HF, PUBLIC_HF);
    b.add_string(KEY_PUBLIC_REVISION, PUBLIC_REVISION);
    b.add_string(KEY_PUBLIC_FILE, PUBLIC_FILE);
    b.add_u32(KEY_PUBLIC_BYTES, PUBLIC_BYTES);
    b.add_string(KEY_PUBLIC_SHA256, PUBLIC_SHA256);

    // Self-describing redistribution: pyannote ships MIT end-to-end
    // (upstream source LICENSE and official HF cardData both say MIT).
    vokra_core::stamp_provenance(
        b,
        LicenseClass::Permissive,
        DEFAULT_LICENSE,
        Some(NAME),
        Some(&format!(
            "{UPSTREAM_HF}@{UPSTREAM_REVISION} exact {TENSOR_COUNT}-F32-tensor inference manifest"
        )),
    );
    b.add_string("vokra.provenance.upstream_hf", UPSTREAM_HF);
    b.add_string("vokra.provenance.upstream_revision", UPSTREAM_REVISION);
}

fn validate_manifest(st: &SafetensorsFile) -> Result<(), ConvertError> {
    let observed = st
        .tensors()
        .iter()
        .map(|tensor| (tensor.name.clone(), (tensor.dtype, tensor.shape.clone())))
        .collect::<BTreeMap<_, _>>();
    validate_observed_manifest(&observed)
}

fn validate_observed_manifest(
    observed: &BTreeMap<String, (GgmlType, Vec<u64>)>,
) -> Result<(), ConvertError> {
    let expected = expected_manifest();
    if observed.len() != TENSOR_COUNT {
        return Err(ConvertError::Parse(format!(
            "pyannote-segmentation: checkpoint has {} tensors, expected exactly {TENSOR_COUNT}",
            observed.len()
        )));
    }
    for (name, (dtype, shape)) in observed {
        let wanted = expected.get(name).ok_or_else(|| {
            ConvertError::Parse(format!(
                "pyannote-segmentation: unexpected tensor {name:?}; refusing strict conversion"
            ))
        })?;
        if *dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "pyannote-segmentation: tensor {name:?} has {dtype:?}, expected canonical F32"
            )));
        }
        if shape != wanted {
            return Err(ConvertError::Parse(format!(
                "pyannote-segmentation: tensor {name:?} shape {shape:?}, expected {wanted:?}"
            )));
        }
    }
    for name in expected.keys() {
        if !observed.contains_key(name) {
            return Err(ConvertError::Parse(format!(
                "pyannote-segmentation: required tensor {name:?} is missing"
            )));
        }
    }
    Ok(())
}

pub(crate) fn expected_manifest() -> BTreeMap<String, Vec<u64>> {
    let mut tensors = BTreeMap::new();
    tensors.insert("classifier.bias".to_owned(), vec![7]);
    tensors.insert("classifier.weight".to_owned(), vec![7, 128]);
    tensors.insert("linear.0.bias".to_owned(), vec![128]);
    tensors.insert("linear.0.weight".to_owned(), vec![128, 256]);
    tensors.insert("linear.1.bias".to_owned(), vec![128]);
    tensors.insert("linear.1.weight".to_owned(), vec![128, 128]);

    for layer in 0..DEFAULT_LSTM_NUM_LAYERS {
        let input = if layer == 0 { 60 } else { 256 };
        for suffix in ["", "_reverse"] {
            tensors.insert(format!("lstm.bias_hh_l{layer}{suffix}"), vec![512]);
            tensors.insert(format!("lstm.bias_ih_l{layer}{suffix}"), vec![512]);
            tensors.insert(format!("lstm.weight_hh_l{layer}{suffix}"), vec![512, 128]);
            tensors.insert(format!("lstm.weight_ih_l{layer}{suffix}"), vec![512, input]);
        }
    }

    for (name, shape) in [
        ("sincnet.conv1d.0.filterbank.band_hz_", vec![40, 1]),
        ("sincnet.conv1d.0.filterbank.low_hz_", vec![40, 1]),
        ("sincnet.conv1d.0.filterbank.n_", vec![1, 125]),
        ("sincnet.conv1d.0.filterbank.window_", vec![125]),
        ("sincnet.conv1d.1.bias", vec![60]),
        ("sincnet.conv1d.1.weight", vec![60, 80, 5]),
        ("sincnet.conv1d.2.bias", vec![60]),
        ("sincnet.conv1d.2.weight", vec![60, 60, 5]),
        ("sincnet.norm1d.0.bias", vec![80]),
        ("sincnet.norm1d.0.weight", vec![80]),
        ("sincnet.norm1d.1.bias", vec![60]),
        ("sincnet.norm1d.1.weight", vec![60]),
        ("sincnet.norm1d.2.bias", vec![60]),
        ("sincnet.norm1d.2.weight", vec![60]),
        ("sincnet.wav_norm1d.bias", vec![1]),
        ("sincnet.wav_norm1d.weight", vec![1]),
    ] {
        tensors.insert(name.to_owned(), shape);
    }
    debug_assert_eq!(tensors.len(), TENSOR_COUNT);
    tensors
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Per-process, per-test scratch path in the system temp dir
    /// (rmvpe / emotion2vec pattern — no external `tempfile` dep,
    /// preserving zero-dep NFR-DS-02). The nanosecond suffix
    /// separates the tests in this module so a parallel `cargo test`
    /// cannot clobber files across them.
    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-pyannote-seg-{}-{}-{}.bin",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        p
    }

    /// Builds the old loose one-tensor BF16 probe so the regression test can
    /// prove that the strict release converter now rejects it.
    fn synthetic_bf16_safetensors() -> (Vec<u8>, Vec<u8>) {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");
        let header =
            r#"{"sincnet.conv1d.0.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&bf16);
        (buf, bf16)
    }

    /// Builds an exact-shape, zero-valued copy of the 54-tensor release
    /// manifest. The values are synthetic, so this exercises serialization
    /// and identity gates only; independent numerical parity uses the real
    /// public checkpoint on VAST.
    fn synthetic_canonical_safetensors() -> Vec<u8> {
        let manifest = expected_manifest();
        let mut entries = Vec::with_capacity(manifest.len());
        let mut data_len = 0usize;
        for (name, shape) in manifest {
            let elements = shape.iter().copied().product::<u64>() as usize;
            let byte_len = elements.checked_mul(4).expect("synthetic tensor bytes");
            let end = data_len
                .checked_add(byte_len)
                .expect("synthetic checkpoint bytes");
            let dims = shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            entries.push(format!(
                "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{dims}],\"data_offsets\":[{data_len},{end}]}}"
            ));
            data_len = end;
        }
        let header = format!("{{{}}}", entries.join(","));
        let mut bytes = Vec::with_capacity(8 + header.len() + data_len);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + data_len, 0);
        bytes
    }

    #[test]
    fn canonical_checkpoint_roundtrips_all_identity_metadata() {
        let input = scratch_path("canonical-in");
        let output = scratch_path("canonical-out");
        std::fs::write(&input, synthetic_canonical_safetensors())
            .expect("write canonical safetensors");

        let report = convert_pyannote_segmentation_file(&input, &output, Some("MIT"))
            .expect("convert canonical manifest");
        assert_eq!(report.read, TENSOR_COUNT);
        assert_eq!(report.written, TENSOR_COUNT);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0);

        let file = GgufFile::parse(std::fs::read(&output).expect("read output"))
            .expect("parse output GGUF");
        assert_eq!(file.tensors().len(), TENSOR_COUNT);
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH)
                .and_then(|value| value.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(KEY_LSTM_NUM_LAYERS)
                .and_then(|value| value.as_u64()),
            Some(4)
        );
        assert_eq!(
            file.get(KEY_MANIFEST_SHA256)
                .and_then(|value| value.as_str()),
            Some(MANIFEST_SHA256)
        );
        assert_eq!(
            file.get(KEY_UPSTREAM_REVISION)
                .and_then(|value| value.as_str()),
            Some(UPSTREAM_REVISION)
        );
        assert_eq!(
            file.get(KEY_PYANNOTE_AUDIO_REVISION)
                .and_then(|value| value.as_str()),
            Some(PYANNOTE_AUDIO_REVISION)
        );
        assert_eq!(
            file.get(KEY_PUBLIC_REVISION)
                .and_then(|value| value.as_str()),
            Some(PUBLIC_REVISION)
        );
        assert_eq!(
            file.get(KEY_PUBLIC_BYTES).and_then(|value| value.as_u64()),
            Some(PUBLIC_BYTES as u64)
        );
        assert_eq!(
            file.get(KEY_PUBLIC_SHA256).and_then(|value| value.as_str()),
            Some(PUBLIC_SHA256)
        );
        assert_eq!(
            file.get("vokra.provenance.weight_license")
                .and_then(|value| value.as_str()),
            Some("permissive")
        );
        assert_eq!(
            file.get("vokra.provenance.license")
                .and_then(|value| value.as_str()),
            Some(DEFAULT_LICENSE)
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// A loose one-tensor BF16 probe used to convert successfully. The exact
    /// release is 54 F32 tensors, so the same input must now fail closed.
    #[test]
    fn loose_bf16_probe_fails_closed() {
        let (input_bytes, _) = synthetic_bf16_safetensors();
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let error = convert_pyannote_segmentation_file(&input, &output, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("1 tensors"));
        assert!(error.contains("expected exactly 54"));

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// The canonical release cannot be silently restamped under another SPDX.
    #[test]
    fn conflicting_license_override_fails_before_input_parse() {
        let (input_bytes, _) = synthetic_bf16_safetensors();
        let input = scratch_path("license-in");
        let output = scratch_path("license-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let error = convert_pyannote_segmentation_file(&input, &output, Some("apache-2.0"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("refusing conflicting --license"));

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// An empty header is a truncated/foreign checkpoint, never a model.
    #[test]
    fn zero_tensor_input_fails_closed() {
        let header = "{}";
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());

        let input = scratch_path("zero-in");
        let output = scratch_path("zero-out");
        std::fs::write(&input, &buf).expect("write empty safetensors");

        let error = convert_pyannote_segmentation_file(&input, &output, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("0 tensors"));

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Release constants must not silently drift from the pinned source,
    /// config, and public state-dict manifest.
    #[test]
    #[allow(clippy::assertions_on_constants)] // Compile-time drift guards are intentional.
    fn primary_source_constants_do_not_drift() {
        // From SINCNET_DEFAULTS in PyanNet.py (verified 2026-07-30):
        assert_eq!(DEFAULT_SINCNET_STRIDE, 10);
        // Hidden size is the class default; four layers are the release
        // config override independently visible as l0..l3 tensors.
        assert_eq!(DEFAULT_LSTM_HIDDEN_SIZE, 128);
        assert_eq!(DEFAULT_LSTM_NUM_LAYERS, 4);
        const { assert!(DEFAULT_LSTM_BIDIRECTIONAL) };
        const { assert!(DEFAULT_LSTM_MONOLITHIC) };
        // From LINEAR_DEFAULTS:
        assert_eq!(DEFAULT_LINEAR_HIDDEN_SIZE, 128);
        assert_eq!(DEFAULT_LINEAR_NUM_LAYERS, 2);
        // segmentation-3.0 specifically: three speakers, maximum overlap two,
        // giving seven powerset classes.
        assert_eq!(DEFAULT_NUM_POWERSET_CLASSES, 7);
        // Sample rate is fixed at 16 kHz by PyanNet default:
        assert_eq!(DEFAULT_SAMPLE_RATE, 16000);
    }

    fn observed_manifest() -> BTreeMap<String, (GgmlType, Vec<u64>)> {
        expected_manifest()
            .into_iter()
            .map(|(name, shape)| (name, (GgmlType::F32, shape)))
            .collect()
    }

    #[test]
    fn canonical_manifest_matches_the_public_header() {
        let manifest = expected_manifest();
        assert_eq!(manifest.len(), 54);
        assert_eq!(manifest["lstm.weight_ih_l0"], vec![512, 60]);
        assert_eq!(manifest["lstm.weight_ih_l3_reverse"], vec![512, 256]);
        assert_eq!(manifest["sincnet.conv1d.1.weight"], vec![60, 80, 5]);
        assert_eq!(manifest["classifier.weight"], vec![7, 128]);
        validate_observed_manifest(&observed_manifest()).unwrap();
    }

    #[test]
    fn missing_extra_wrong_shape_and_dtype_fail_closed() {
        let mut missing = observed_manifest();
        missing.remove("classifier.bias");
        assert!(
            validate_observed_manifest(&missing)
                .unwrap_err()
                .to_string()
                .contains("53 tensors")
        );

        let mut extra = observed_manifest();
        extra.remove("classifier.bias");
        extra.insert("fabricated.weight".to_owned(), (GgmlType::F32, vec![7]));
        assert!(
            validate_observed_manifest(&extra)
                .unwrap_err()
                .to_string()
                .contains("unexpected tensor")
        );

        let mut wrong_shape = observed_manifest();
        wrong_shape.get_mut("classifier.weight").unwrap().1 = vec![2, 128];
        assert!(
            validate_observed_manifest(&wrong_shape)
                .unwrap_err()
                .to_string()
                .contains("shape")
        );

        let mut wrong_dtype = observed_manifest();
        wrong_dtype.get_mut("classifier.weight").unwrap().0 = GgmlType::BF16;
        assert!(
            validate_observed_manifest(&wrong_dtype)
                .unwrap_err()
                .to_string()
                .contains("expected canonical F32")
        );
    }
}
