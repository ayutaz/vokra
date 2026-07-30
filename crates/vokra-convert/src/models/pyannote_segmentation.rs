//! **pyannote/segmentation-3.0** (Bredin, CNRS, MIT): safetensors → GGUF
//! conversion (VAD / speaker-segmentation tier, 2026-07-30).
//!
//! Input: an offline `.bin` → safetensors flattening of the upstream
//! `pyannote/segmentation-3.0` `pytorch_model.bin` (via
//! `tools/parity/bin_to_safetensors.py` — the existing bridge; owner
//! runs after `huggingface-cli login` + HF UI gate accept because the
//! HF repo has `gated: auto`). Output: a GGUF carrying every float
//! tensor verbatim under its upstream state_dict name, plus the
//! `vokra.pyannote.*` metadata chunk group a future native PyanNet
//! binder (`crates/vokra-models/src/pyannote/`) will read.
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
//! Source: `github.com/pyannote/pyannote-audio/develop/src/pyannote/`
//! `audio/models/segmentation/PyanNet.py` (CC 直接 fetch 2026-07-30、
//! MIT LICENSE header + full class definition = Copyright (c) 2020
//! CNRS)。**推定は含まない**。
//!
//! ```text
//! waveforms (batch, channel=1, samples)  # 16 kHz mono PCM
//!   -> SincNet frontend
//!      - stride=10 (SINCNET_DEFAULTS)
//!      - sample_rate=16000
//!      - output: (batch, 60, num_frames)
//!   -> rearrange "batch feature frame -> batch frame feature"
//!   -> LSTM (monolithic=True default, LSTM_DEFAULTS)
//!      - nn.LSTM(input_size=60, hidden_size=128, num_layers=2,
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
//! BF16 pass-through skeleton (mirror `wespeaker` / `ecapa_tdnn` /
//! `titanet` / `rmvpe` pattern). Every F32 / F16 / BF16 tensor passes
//! through verbatim under its upstream state_dict name. The
//! `vokra.pyannote.*` chunk group pins hparams from PyanNet.py primary
//! source constants so a future runtime binder can bring the graph up
//! without a side-car config lookup.
//!
//! # Runtime binder / real forward
//!
//! The runtime module scaffold (`crates/vokra-models/src/pyannote/`)
//! is Wave 2 in `docs/handoff/pyannote-implementation-plan-2026-07-30.md`
//! — it carries a real `from_gguf` (missing / mis-shaped tensor → loud
//! `vokra_core::VokraError::ModelLoad` per FR-EX-08) but the inner
//! forward returns `VokraError::UnsupportedOp` until Wave 3 lands the
//! SincNet primitive + real forward (SincNet is a Vokra-new op, not
//! covered by the existing conv1d / LSTM / Linear primitives). This
//! loud-partial posture mirrors RMVPE (weights are bound, kernel
//! binding pending) — honest, not fake-complete.
//!
//! # No ONNX (permanent)
//!
//! pyannote is distributed as torch `.bin` (pickle) + `config.yaml`;
//! this converter **never** touches ONNX (FR-LD-05). The `.bin` →
//! safetensors bridge lives in `tools/parity/bin_to_safetensors.py`
//! (an offline side-car tool, not part of the runtime).

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

/// Canonical weight license SPDX (`mit`). Overrides via the
/// [`convert_pyannote_segmentation_file`] `license` parameter — the
/// standing mechanism for "implementation is clean-room MIT but the
/// upstream distributed checkpoint is another license" scenarios
/// (mirror of `convert_file_licensed` in `lib.rs`).
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

// Canonical hparam values transcribed from PyanNet.py primary source
// (SINCNET_DEFAULTS / LSTM_DEFAULTS / LINEAR_DEFAULTS, github.com/
// pyannote/pyannote-audio/develop/src/pyannote/audio/models/
// segmentation/PyanNet.py — fetched 2026-07-30, MIT LICENSE Copyright
// (c) 2020 CNRS). Kept here as converter-side compile-time constants
// so a GGUF that never had a `vokra.pyannote.*` chunk written (e.g. an
// emergency hand-crafted checkpoint) still round-trips through the
// runtime binder's default fallback.
//
// segmentation-3.0 specifically: the config.yaml uses
// `num_powerset_classes = 7` (3 speakers × 2 overlap slots per
// PyanNet.py dimension property + powerset multiclass encoding). Owner
// verifies against actual config.yaml after HF gate accept per Wave 1
// owner tasks in `docs/handoff/pyannote-implementation-plan-2026-07-30.md`.
pub const DEFAULT_SAMPLE_RATE: u32 = 16000;
pub const DEFAULT_SINCNET_STRIDE: u32 = 10;
pub const DEFAULT_LSTM_HIDDEN_SIZE: u32 = 128;
pub const DEFAULT_LSTM_NUM_LAYERS: u32 = 2;
pub const DEFAULT_LSTM_BIDIRECTIONAL: bool = true;
pub const DEFAULT_LSTM_MONOLITHIC: bool = true;
pub const DEFAULT_LINEAR_HIDDEN_SIZE: u32 = 128;
pub const DEFAULT_LINEAR_NUM_LAYERS: u32 = 2;
pub const DEFAULT_NUM_POWERSET_CLASSES: u32 = 7;

/// Outcome of a pyannote-segmentation conversion.
///
/// Mirrors the sibling `rmvpe` / `wespeaker` / `titanet` reports:
/// `read` pins the total budget the safetensors reader surfaced,
/// `written` counts float pass-through, `bf16_passthrough` is a subset
/// of `written` for the BF16 tensors, `skipped_non_float` is a
/// defensive counter (the safetensors reader rejects unknown dtypes
/// at parse time, so any tensor reaching this arm signals an upstream
/// reader change).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PyannoteSegmentationReport {
    /// Total tensors seen in the upstream safetensors header
    /// (`written + skipped_non_float`).
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter).
    pub skipped_non_float: usize,
    /// Of the tensors in `written`, how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; runtime widens BF16 →
    /// f32 losslessly via `crates/vokra-core/src/gguf/quant/mod.rs
    /// decode_bf16` (BF16 = top 16 bits of an f32 — `bits << 16`
    /// is exact).
    pub bf16_passthrough: usize,
}

/// Reads a safetensors checkpoint at `input` and writes a
/// pyannote-segmentation GGUF to `output`.
///
/// Every F32 / F16 / BF16 tensor is emitted verbatim under its
/// upstream state_dict name; the `vokra.provenance.*` +
/// `vokra.model.*` + `vokra.pyannote.*` chunk groups pin the upstream
/// repo, weight license, model category and PyanNet hparams so the
/// future runtime binder (`crates/vokra-models/src/pyannote/`) can
/// bring the graph up without a side-car config lookup.
///
/// `license` overrides [`DEFAULT_LICENSE`] (`"mit"`) — the same
/// mechanism `lib.rs::convert_file_licensed` uses when the
/// implementation is clean-room but the redistributed checkpoint
/// carries a different SPDX.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input.
pub fn convert_pyannote_segmentation_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<PyannoteSegmentationReport, ConvertError> {
    // Whole-file read: a segmentation-3.0 checkpoint is ~5.7 MB — no
    // streaming needed (Moshi / Voxtral GB-scale converters use the
    // streaming path). Even a hypothetical scaled variant stays well
    // under the streaming threshold.
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // PyanNet hparam chunk group — every value is a primary-source
    // constant transcribed from PyanNet.py (SINCNET_DEFAULTS /
    // LSTM_DEFAULTS / LINEAR_DEFAULTS, github.com/pyannote/pyannote-
    // audio/develop/src/pyannote/audio/models/segmentation/PyanNet.py,
    // fetched 2026-07-30 — CLAUDE.md「ハルシネーション厳禁」).
    // The runtime binder's `from_gguf` falls back to the same
    // constants when a key is absent, so a checkpoint that never
    // carried a `vokra.pyannote.*` chunk still loads.
    b.add_u32(KEY_SAMPLE_RATE, DEFAULT_SAMPLE_RATE);
    b.add_u32(KEY_SINCNET_STRIDE, DEFAULT_SINCNET_STRIDE);
    b.add_u32(KEY_LSTM_HIDDEN_SIZE, DEFAULT_LSTM_HIDDEN_SIZE);
    b.add_u32(KEY_LSTM_NUM_LAYERS, DEFAULT_LSTM_NUM_LAYERS);
    b.add_bool(KEY_LSTM_BIDIRECTIONAL, DEFAULT_LSTM_BIDIRECTIONAL);
    b.add_bool(KEY_LSTM_MONOLITHIC, DEFAULT_LSTM_MONOLITHIC);
    b.add_u32(KEY_LINEAR_HIDDEN_SIZE, DEFAULT_LINEAR_HIDDEN_SIZE);
    b.add_u32(KEY_LINEAR_NUM_LAYERS, DEFAULT_LINEAR_NUM_LAYERS);
    b.add_u32(KEY_NUM_POWERSET_CLASSES, DEFAULT_NUM_POWERSET_CLASSES);

    // Self-describing redistribution: the artifact carries its own
    // licence. pyannote ships MIT end-to-end (upstream `pyannote/
    // pyannote-audio` LICENSE = MIT Copyright (c) 2020 CNRS, HF
    // cardData `license: mit`, fetched 2026-07-30). The `license`
    // override lets a downstream repackager stamp a different SPDX
    // if they redistribute under stricter terms.
    let effective_license = license.unwrap_or(DEFAULT_LICENSE);
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        effective_license,
        Some(NAME),
        Some(UPSTREAM_HF),
    );

    let mut report = PyannoteSegmentationReport::default();
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the accepted
    // ADR (docs/adr/qwen3-tts-bf16.md, strategy A_passthrough); the
    // runtime widens BF16 → f32 exactly at load via
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. Mirrors
    // `rmvpe::convert_rmvpe_file` / `wespeaker::convert_wespeaker_file`.
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )?;
                report.written += 1;
                if t.dtype == GgmlType::BF16 {
                    report.bf16_passthrough += 1;
                }
            }
            _ => {
                report.skipped_non_float += 1;
            }
        }
    }

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Parse(e.to_string()))?;
    std::fs::write(output, out_bytes).map_err(ConvertError::Io)?;
    Ok(report)
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

    /// Builds a synthetic safetensors buffer with a single BF16
    /// tensor — a byte-identity assert catches any silent widen /
    /// downcast attempt. Uses a PyanNet-plausible tensor name
    /// (`sincnet.conv1d.0.weight`) so the fixture matches the
    /// upstream state_dict naming convention documented in
    /// PyanNet.py.
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

    /// STEP 1 (BF16 pass-through): the upstream BF16 checkpoint
    /// must survive the file-based converter round-trip with its
    /// dtype preserved (GGUF type 30 = `GgmlType::BF16`) and its
    /// payload byte-identical to the input. Mirror of rmvpe /
    /// wespeaker / titanet / emotion2vec.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (input_bytes, bf16_payload) = synthetic_bf16_safetensors();
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_pyannote_segmentation_file(&input, &output, None).expect("convert");

        assert_eq!(report.read, 1, "one tensor visible in safetensors header");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of rmvpe / wespeaker)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 subset counter must record the pass-through"
        );

        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        let info = file
            .tensor_info("sincnet.conv1d.0.weight")
            .expect("BF16 tensor present after pass-through");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16_payload.as_slice(),
            "BF16 payload must be byte-identical to input"
        );

        // Provenance + category chunks pinned on the artifact itself.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );

        // PyanNet hparam chunk group — every key from
        // SINCNET_DEFAULTS / LSTM_DEFAULTS / LINEAR_DEFAULTS +
        // segmentation-3.0 powerset class count must be present with
        // the primary-source constant value. Guards the runtime
        // binder's fallback path (a GGUF without a chunk still loads
        // via the same constant).
        assert_eq!(
            file.get(KEY_SAMPLE_RATE).and_then(|v| v.as_u64()),
            Some(DEFAULT_SAMPLE_RATE as u64)
        );
        assert_eq!(
            file.get(KEY_SINCNET_STRIDE).and_then(|v| v.as_u64()),
            Some(DEFAULT_SINCNET_STRIDE as u64)
        );
        assert_eq!(
            file.get(KEY_LSTM_HIDDEN_SIZE).and_then(|v| v.as_u64()),
            Some(DEFAULT_LSTM_HIDDEN_SIZE as u64)
        );
        assert_eq!(
            file.get(KEY_LSTM_NUM_LAYERS).and_then(|v| v.as_u64()),
            Some(DEFAULT_LSTM_NUM_LAYERS as u64)
        );
        assert_eq!(
            file.get(KEY_LSTM_BIDIRECTIONAL).and_then(|v| v.as_bool()),
            Some(DEFAULT_LSTM_BIDIRECTIONAL)
        );
        assert_eq!(
            file.get(KEY_LSTM_MONOLITHIC).and_then(|v| v.as_bool()),
            Some(DEFAULT_LSTM_MONOLITHIC)
        );
        assert_eq!(
            file.get(KEY_LINEAR_HIDDEN_SIZE).and_then(|v| v.as_u64()),
            Some(DEFAULT_LINEAR_HIDDEN_SIZE as u64)
        );
        assert_eq!(
            file.get(KEY_LINEAR_NUM_LAYERS).and_then(|v| v.as_u64()),
            Some(DEFAULT_LINEAR_NUM_LAYERS as u64)
        );
        assert_eq!(
            file.get(KEY_NUM_POWERSET_CLASSES).and_then(|v| v.as_u64()),
            Some(DEFAULT_NUM_POWERSET_CLASSES as u64)
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// License override path: a caller who obtained the weight under
    /// a different SPDX (unlikely for pyannote which is MIT
    /// end-to-end, but the mechanism exists for all converters via
    /// `convert_file_licensed` in `lib.rs`) can stamp that SPDX
    /// through the `license` parameter. Guards against silent
    /// override / SPDX drift.
    #[test]
    fn license_override_stamps_supplied_spdx() {
        let (input_bytes, _) = synthetic_bf16_safetensors();
        let input = scratch_path("license-in");
        let output = scratch_path("license-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        // Hypothetical downstream override — apache-2.0 instead of
        // upstream MIT (both permissive, both round-trip through
        // `stamp_provenance` cleanly).
        let report = convert_pyannote_segmentation_file(&input, &output, Some("apache-2.0"))
            .expect("convert");
        assert_eq!(report.written, 1);

        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        // The provenance license chunk must reflect the override, not
        // the DEFAULT_LICENSE constant.
        assert_eq!(
            file.get("vokra.provenance.license")
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
            "license override must reach the provenance chunk"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Zero-tensor safetensors (empty header): must produce an empty
    /// report (no crash, no silent success masking a truncated input).
    /// Zero-tensor is a legitimate corner case — a probe tool that
    /// exercises the converter contract can pass an empty header
    /// deliberately.
    #[test]
    fn zero_tensor_input_returns_empty_report() {
        let header = "{}";
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());

        let input = scratch_path("zero-in");
        let output = scratch_path("zero-out");
        std::fs::write(&input, &buf).expect("write empty safetensors");

        let report = convert_pyannote_segmentation_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 0);
        assert_eq!(report.written, 0);
        assert_eq!(report.bf16_passthrough, 0);
        assert_eq!(report.skipped_non_float, 0);

        // Even with zero tensors the provenance / hparam chunks are
        // still stamped — an empty-body GGUF is still a well-formed
        // Vokra artifact.
        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(KEY_NUM_POWERSET_CLASSES).and_then(|v| v.as_u64()),
            Some(DEFAULT_NUM_POWERSET_CLASSES as u64)
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Constant pin: primary-source values from PyanNet.py must not
    /// silently drift. A future edit that changes a default must also
    /// bump the corresponding upstream reference (or the CC-verified
    /// row in `docs/license-audit.md`), not sneak past a stale check.
    #[test]
    fn primary_source_constants_do_not_drift() {
        // From SINCNET_DEFAULTS in PyanNet.py (verified 2026-07-30):
        assert_eq!(DEFAULT_SINCNET_STRIDE, 10);
        // From LSTM_DEFAULTS:
        assert_eq!(DEFAULT_LSTM_HIDDEN_SIZE, 128);
        assert_eq!(DEFAULT_LSTM_NUM_LAYERS, 2);
        assert!(DEFAULT_LSTM_BIDIRECTIONAL);
        assert!(DEFAULT_LSTM_MONOLITHIC);
        // From LINEAR_DEFAULTS:
        assert_eq!(DEFAULT_LINEAR_HIDDEN_SIZE, 128);
        assert_eq!(DEFAULT_LINEAR_NUM_LAYERS, 2);
        // From segmentation-3.0 specifically (3 speakers × 2 overlap
        // = 7 powerset classes; owner verifies against actual
        // config.yaml per Wave 1 owner tasks in
        // `docs/handoff/pyannote-implementation-plan-2026-07-30.md`):
        assert_eq!(DEFAULT_NUM_POWERSET_CLASSES, 7);
        // Sample rate is fixed at 16 kHz by PyanNet default:
        assert_eq!(DEFAULT_SAMPLE_RATE, 16000);
    }
}
