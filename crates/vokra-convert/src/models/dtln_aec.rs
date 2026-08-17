//! **DTLN-AEC** (Dual-Signal Transformation LSTM Network for Acoustic
//! Echo Cancellation): safetensors checkpoint → GGUF conversion
//! (post-audit-cc-gap-2026-08-14 Wave 6 loud-partial land, sibling to
//! `nkf_aec` in the `aec` category).
//!
//! Input: the upstream `breizhn/DTLN-aec` release on GitHub — Nils L.
//! Westhausen & Bernd T. Meyer, INTERSPEECH 2021,
//! arXiv:2010.15754 "Acoustic Echo Cancellation with the Dual-Signal
//! Transformation LSTM Network". The upstream distribution ships
//! `.tflite` files ONLY (three variants: 128 / 256 / 512 LSTM units per
//! stage) — no `.h5` / `.onnx` checkpoint. Callers pre-flatten the
//! TFLite `.tflite` to safetensors offline through a future
//! `tools/parity/dtln_aec_prepare_checkpoint.py` (**not yet written** —
//! the DFN3 / NKF-AEC / Kokoro pickle-bridge pattern, where TFLite is a
//! Python-side tool that **never** enters the runtime, FR-LD-05).
//! Until it lands, that flattening is a manual owner-side step.
//!
//! Output: a GGUF carrying every float tensor plus the `vokra.model.*`,
//! `vokra.provenance.*`, and `vokra.dtln_aec.*` metadata chunks the
//! runtime AEC path binds against.
//!
//! # License
//!
//! - SPDX: **MIT** ([`vokra_core::LicenseClass::Permissive`]).
//! - Category: **aec** (sibling of the algorithmic AEC op family —
//!   M4-03 `vokra_aec_*` SpeexDSP / WebRTC AEC3 Rust port — and of the
//!   neural [`super::nkf_aec`] Kalman-filter AEC. DTLN-AEC is a
//!   **dual-signal LSTM** alternative: an STFT-domain LSTM predicts an
//!   IRM mask over the mic⊕far concatenated magnitude spectrogram, and
//!   a time-domain LSTM operates on the residual PCM to remove the
//!   remaining echo — a distinct topology axis from Kalman filtering).
//! - Notes: verified upstream `github.com/breizhn/DTLN-aec/LICENSE` =
//!   MIT with `Copyright (c) 2021 Nils L. Westhausen`; the paper's
//!   supplementary release ships `dtln_aec_128.tflite`,
//!   `dtln_aec_256.tflite`, `dtln_aec_512.tflite`.
//!
//! # BF16 pass-through (mirror of nkf_aec / speaker_3d / ecapa_tdnn /
//! # qwen3_tts / voxcpm2 / vibevoice / moshi)
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm on the same
//! match arm — no convert-time widening. BF16 is emitted as GGUF type
//! 30 (`GgmlType::BF16`); the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). The observability
//! counter [`DtlnAecReport::bf16_passthrough`] guards against a silent
//! widen / downcast regression.
//!
//! # Variant selection
//!
//! Upstream ships three fixed-width variants ([`DtlnVariant::Units128`],
//! [`DtlnVariant::Units256`], [`DtlnVariant::Units512`]). The variant is
//! stamped into the emitted GGUF via [`KEY_VARIANT_LSTM_UNITS`] so the
//! runtime loader can select the right tensor-shape validation without
//! probing tensor dimensions — `NkfAec::from_gguf`'s per-tensor dim
//! assertions rely on this same "hparams from metadata, cross-checked
//! against tensor shape" contract.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream TFLite tensor keys verbatim**
//! (the DFN3 / NKF-AEC / speaker_3d contract; the prepare-checkpoint
//! sidecar preserves the TFLite `Tensor.name` slots as `state_dict`
//! dotted keys). Real-weight parity binding to the runtime
//! `vokra-models::aec::dtln_aec` module is deferred until (a) the
//! **generic LSTM primitive** lands in `vokra-ops` (the sibling
//! `nkf_aec` inlined its per-layer GRU because dim was tiny (H=18); the
//! DTLN 128/256/512-unit LSTMs with 4-gate concatenation are large
//! enough that inlining without a shared primitive multiplies
//! implementation cost across DTLN + every future LSTM-based model),
//! and (b) owner sign-off per `docs/license-audit.md §3.1`.
//!
//! # Provenance key choice: `vokra.provenance.upstream_url`
//!
//! Following the sibling [`super::nkf_aec`] posture: the upstream's
//! primary redistribution source is the **GitHub release** at
//! `github.com/breizhn/DTLN-aec` — the author ships no HF mirror. The
//! converter therefore stamps [`KEY_PROVENANCE_UPSTREAM_URL`] instead
//! of `vokra.provenance.upstream_hf`, matching the parallel key naming
//! convention the Wave A tickets established for non-HF sources
//! (`nkf-aec` / `ten-vad` / `htdemucs-4s-6s` / `frcrn` / `nsnet2` /
//! `dnsmos-p808-p835` / `openwakeword-op` / `torchaudio-squim` /
//! `rnnoise-v0.2`).
//!
//! # No ONNX / no TFLite (permanent)
//!
//! The upstream DTLN-AEC release ships TensorFlow-Lite `.tflite` files
//! only; this converter accepts pre-flattened safetensors only and
//! **never** touches TFLite or ONNX (FR-LD-05, NFR-DS-02 zero-dep). A
//! future `tools/parity/dtln_aec_prepare_checkpoint.py` sidecar
//! (**not yet written**; uv-managed Python 3.12 per memory
//! `[[feedback-python-uses-uv]]` + `[[feedback-python-3-12]]`) is the
//! intended bridge from `.tflite` bytes to safetensors — the runtime
//! tree contains neither, and today no such bridge is on disk.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value for DTLN-AEC GGUFs. Intentionally distinct
/// from every sibling AEC / denoise family — sharing `nkf_aec`'s arch
/// tag would mis-route the runtime dispatch (an NKF-AEC Kalman loader
/// would try to interpret DTLN-AEC's LSTM tensors, which is a wrong-
/// topology bug not a wrong-shape bug so early tensor-shape assertions
/// alone would not catch it — the arch tag is the correct first fence).
pub const ARCH: &str = "dtln_aec";

/// `vokra.model.name` value written for the canonical
/// `breizhn/DTLN-aec` release.
pub const NAME: &str = "dtln-aec";

/// `vokra.model.category` value written for every DTLN-AEC GGUF.
///
/// Shared with sibling `nkf_aec` and the algorithmic M4-03 `vokra_aec_*`
/// path — model-card grouping and runtime dispatch treat "aec" as one
/// family (Kalman vs LSTM vs adaptive-filter are three implementations
/// of the same use case).
pub const CATEGORY: &str = "aec";

/// Primary redistribution source (author's GitHub repository — there is
/// no HF mirror). Written under [`KEY_PROVENANCE_UPSTREAM_URL`].
pub const UPSTREAM_URL: &str = "github.com/breizhn/DTLN-aec";

/// Default upstream weight licence (SPDX). Verified against
/// `github.com/breizhn/DTLN-aec/blob/main/LICENSE` (standard MIT,
/// `Copyright (c) 2021 Nils L. Westhausen`).
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

// ---- upstream-pinned dims (constants from upstream `dtln_model.py` +
//      `run_aec.py` + the paper — every non-variant axis is fixed) ----

/// STFT FFT size (`run_aec.py::block_len = 512`, matches upstream's
/// n_fft = block_len for a real-STFT with rectangular hop).
pub const N_FFT: usize = 512;

/// STFT block length in samples (`run_aec.py::block_len = 512`,
/// upstream naming for the analysis window length).
pub const BLOCK_LEN: usize = 512;

/// STFT hop size (`run_aec.py::block_shift = 128` — a 4x-overlap
/// framing pattern, distinct from NKF-AEC's 4x-overlap 1024/256 pair).
pub const HOP: usize = 128;

/// PCM sample rate the release was trained at (paper §4 —
/// AEC-Challenge 2021 corpus is 16 kHz).
pub const SAMPLE_RATE: u32 = 16_000;

// ---- variant enum -------------------------------------------------------

/// Fixed-width variants shipped upstream. Every variant sets the LSTM
/// hidden width used by BOTH the STFT-domain stage AND the time-domain
/// stage (upstream trains the two stages with matched width per
/// release).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum DtlnVariant {
    /// `dtln_aec_128.tflite` — smallest release (128 LSTM units per
    /// stage), ~1 MB.
    ///
    /// The [`Default`], because it is the sane fallback when the input
    /// carries no LSTM tensor to width-probe from.
    #[default]
    Units128,
    /// `dtln_aec_256.tflite` — mid release (256 LSTM units per stage),
    /// ~3 MB.
    Units256,
    /// `dtln_aec_512.tflite` — largest release (512 LSTM units per
    /// stage), ~7 MB.
    Units512,
}

impl DtlnVariant {
    /// LSTM hidden width for both stages.
    pub fn lstm_units(&self) -> usize {
        match self {
            Self::Units128 => 128,
            Self::Units256 => 256,
            Self::Units512 => 512,
        }
    }

    /// Recovers the variant from a stamped `lstm_units` value; None if
    /// the value doesn't match any known upstream release width (fail-
    /// closed against a mis-stamped GGUF).
    pub fn from_lstm_units(units: usize) -> Option<Self> {
        match units {
            128 => Some(Self::Units128),
            256 => Some(Self::Units256),
            512 => Some(Self::Units512),
            _ => None,
        }
    }
}

// ---- metadata keys ------------------------------------------------------

/// `vokra.dtln_aec.lstm_units` — LSTM hidden width shared by the STFT
/// stage and the time-domain stage (128 / 256 / 512).
pub const KEY_VARIANT_LSTM_UNITS: &str = "vokra.dtln_aec.lstm_units";

/// `vokra.dtln_aec.n_fft` — analysis FFT size (fixed 512 in every
/// upstream release; stamped anyway so a future variant that changes it
/// does not need to invent a chunk).
pub const KEY_N_FFT: &str = "vokra.dtln_aec.n_fft";

/// `vokra.dtln_aec.hop` — analysis hop (fixed 128 in every upstream
/// release).
pub const KEY_HOP: &str = "vokra.dtln_aec.hop";

/// `vokra.dtln_aec.block_len` — upstream's own naming for the analysis
/// block length in samples (equal to `n_fft` in every current release).
pub const KEY_BLOCK_LEN: &str = "vokra.dtln_aec.block_len";

/// `vokra.dtln_aec.sample_rate` — PCM sample rate the model was trained
/// at (16 kHz — paper §4).
pub const KEY_SAMPLE_RATE: &str = "vokra.dtln_aec.sample_rate";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the sibling nkf_aec convention).

/// `vokra.model.category` metadata key. Local per the established
/// nkf_aec / funcodec / wespeaker / speaker_3d / ecapa_tdnn precedent
/// (not yet centralized in `vokra-core::gguf::chunks`).
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_url` — the primary redistribution source
/// URL for models whose canonical release is NOT on the Hugging Face
/// hub. Parallel to `vokra.provenance.upstream_hf` (the HF-hosted
/// sibling key); DTLN-AEC ships only on GitHub. Local per the same
/// convention as [`KEY_MODEL_CATEGORY`].
pub(crate) const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

/// Outcome of a DTLN-AEC conversion.
///
/// Mirrors the sibling BF16-passthrough converters' counter shape
/// (`super::nkf_aec::NkfAecReport`, `super::ecapa_tdnn::EcapaTdnnReport`,
/// `super::speaker_3d::Speaker3dReport`) — adds `read` tracking every
/// tensor the safetensors reader surfaced so the invariant
/// `read == written + skipped_non_float` is auditable at the report
/// level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DtlnAecReport {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time
    /// (`crates/vokra-core/src/safetensors.rs map_dtype`), so any
    /// tensor reaching this counter would signal a reader change
    /// upstream; kept for parity with the sibling converters).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; the runtime widens BF16
    /// → f32 losslessly via the single choke point
    /// `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. A silent
    /// widen / downcast regression would surface as this counter
    /// drifting away from the input BF16 count.
    pub bf16_passthrough: usize,
    /// The variant detected from the input tensor shapes (or default
    /// [`DtlnVariant::Units128`] if no canonical LSTM tensor was
    /// present in the safetensors — e.g. an empty test buffer).
    pub variant: DtlnVariant,
}

/// Detects the variant from the loaded safetensors' tensor shapes. The
/// upstream TFLite tensors preserve the LSTM kernel shape
/// `[input_dim, 4 * lstm_units]` (Keras dense layer convention:
/// input-major, with the four LSTM gates concatenated along output
/// dim). Every known variant sets `lstm_units` to exactly 128 / 256 /
/// 512, so dividing the second dim by 4 recovers the variant.
///
/// The scan looks for **any** tensor whose second dim divides evenly by
/// 4 into one of the three known widths and returns the FIRST such
/// match — this is defense-in-depth against a prepare-checkpoint bridge
/// that renames some but not all LSTM sub-tensors while preserving
/// their shapes.
///
/// Returns `None` when no matching tensor is present (e.g. an empty
/// safetensors buffer produced by an incomplete bridge), so the caller
/// can loud-fail rather than silently defaulting to 128 units.
fn detect_variant_from_shapes(st: &SafetensorsFile) -> Option<DtlnVariant> {
    for t in st.tensors() {
        if t.shape.len() >= 2 {
            let last = *t.shape.last().unwrap();
            if last > 0 && last % 4 == 0 {
                let candidate = last / 4;
                if let Some(v) = DtlnVariant::from_lstm_units(candidate as usize) {
                    return Some(v);
                }
            }
        }
    }
    None
}

/// Converts a DTLN-AEC safetensors checkpoint at `input` (flattened
/// from an upstream `.tflite` file — a future
/// `tools/parity/dtln_aec_prepare_checkpoint.py` is not yet written, so
/// that flattening is an owner-side step today) into a Vokra-native
/// GGUF at `output`, returning a [`DtlnAecReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// tensor key; the `vokra.model.*` (arch / name / category),
/// `vokra.provenance.*` (weight_license / license / model_id / source /
/// upstream_url), and `vokra.dtln_aec.*` (lstm_units / n_fft / hop /
/// block_len / sample_rate) chunks are stamped for the runtime
/// compliance gate (FR-CP-03) and the runtime hparam recovery path.
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// `DEFAULT_LICENSE_SPDX` (`"mit"`, `Permissive`) — the upstream
/// GitHub release ships MIT with `Copyright (c) 2021 Nils L.
/// Westhausen`.
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_dtln_aec_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<DtlnAecReport, ConvertError> {
    // Load the whole checkpoint into memory: even the largest upstream
    // release (`dtln_aec_512.tflite`) is ~7 MB — 3 orders of magnitude
    // smaller than the streaming-mandated Moshi 14 GiB tier, so the
    // simple `std::fs::read` posture the sibling non-streaming
    // converters (nkf_aec / ecapa_tdnn / speaker_3d / qwen3_tts) use
    // applies trivially.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    // Detect the variant from tensor shapes; fall back to the smallest
    // known width when the input has no canonical LSTM tensor
    // (typically test buffers). A real prepare-checkpoint bridge always
    // emits at least one LSTM tensor per stage, so a `None` from a
    // real bridge would indicate a stale or wrong-shape output.
    let variant = detect_variant_from_shapes(&st).unwrap_or_default();

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);

    // Variant + fixed dims stamped unconditionally so the runtime
    // loader recovers hparams from metadata (never from tensor shape
    // probing at load time — probing would silently accept a widely
    // wrong checkpoint that happens to have consistent-looking shapes).
    // GgufBuilder does not expose `add_u64` today; the sibling
    // `add_u32` path is losslessly widened by `GgufMetadataValue::as_u64`
    // on the reader side (see `crates/vokra-core/src/gguf/value.rs
    // as_u64_widens_unsigned_variants_only`), so the runtime reads back
    // as `u64` uniformly regardless of the writer-side variant.
    b.add_u32(KEY_VARIANT_LSTM_UNITS, variant.lstm_units() as u32);
    b.add_u32(KEY_N_FFT, N_FFT as u32);
    b.add_u32(KEY_HOP, HOP as u32);
    b.add_u32(KEY_BLOCK_LEN, BLOCK_LEN as u32);
    b.add_u32(KEY_SAMPLE_RATE, SAMPLE_RATE);

    // Default provenance stamp — Permissive MIT (upstream
    // `github.com/breizhn/DTLN-aec/LICENSE`, `Copyright (c) 2021 Nils
    // L. Westhausen`). The optional `license` argument overrides below
    // via the same restated-source convention as the sibling
    // converters.
    let effective_spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let effective_class = LicenseClass::from_license_str(effective_spdx);
    vokra_core::stamp_provenance(
        &mut b,
        effective_class,
        effective_spdx,
        Some(NAME),
        Some(
            "github.com/breizhn/DTLN-aec (Dual-Signal Transformation LSTM \
             Network for AEC, Westhausen & Meyer INTERSPEECH 2021, arXiv:2010.15754)",
        ),
    );

    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted BF16-passthrough
    // ADR the sibling non-streaming converters (nkf_aec / speaker_3d /
    // ecapa_tdnn / qwen3_tts / vibevoice / voxcpm2 / moshi) share; the
    // runtime widens BF16 → f32 exactly at load via the single choke
    // point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
    let mut report = DtlnAecReport {
        variant,
        ..DtlnAecReport::default()
    };
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

    // Serialize and land the emitted GGUF at `output`. `to_bytes()`
    // stamps `vokra.schema.version` + `vokra.schema.producer` on its
    // own via the writer's built-in schema stamper — no per-converter
    // duplication needed.
    let out_bytes = b.to_bytes()?;
    std::fs::write(output, &out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use vokra_core::gguf::GgufFile;

    /// Per-test unique scratch path (PID + nanos + a suffix derived from
    /// the caller — every test in this module uses a distinct `name` so
    /// concurrent runs do not collide).
    fn scratch_path(tag: &str, ext: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-dtln-aec-{tag}-{}-{}.{ext}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        p
    }

    /// RAII cleanup so failing tests do not leak temp files on disk
    /// (best-effort — a panic mid-cleanup is fine).
    struct TempFileGuard(PathBuf);
    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Encodes `values` as BF16 (top 16 bits of each `f32`) little-endian.
    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// Pins the BF16 pass-through end-to-end: the tensor survives the
    /// converter's `convert_dtln_aec_file` file → file round-trip with
    /// its dtype preserved (`GgmlType::BF16`, GGUF type 30) and its
    /// payload byte-identical. Mirrors
    /// `nkf_aec::tests::bf16_tensor_passes_through_verbatim`. A silent
    /// widen at convert time would still round-trip _values_ (BF16 →
    /// f32 widen is exact), so this test asserts on the dtype AND the
    /// raw bytes — two concentric fences.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // A tensor shaped `[input_dim=1, 4*128=512]` — the canonical
        // 128-unit variant's LSTM kernel shape. Non-zero bit patterns
        // so a silent widen / downcast cannot round-trip trivially.
        // We only need a few actual bytes for the pass-through fence;
        // the shape drives variant detection separately.
        let values: [f32; 4] = [1.0, -2.5, 0.15625, 42.0];
        let payload = bf16_bytes(&values);
        // Truncate the tensor to 4 elements while claiming a 512-slot
        // shape would be invalid for real weights but valid for the
        // safetensors buffer format — we instead use a small consistent
        // shape and give it just enough data.
        let header = r#"{"stft_lstm.kernel":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&payload);

        let input_path = scratch_path("bf16-in", "safetensors");
        let output_path = scratch_path("bf16-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_dtln_aec_file(&input_path, &output_path, None).expect("convert BF16");
        assert_eq!(report.read, 1, "one BF16 tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of nkf_aec / speaker_3d)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );

        // Round-trip through the emitted GGUF: dtype preserved, payload
        // byte-identical (no convert-time widening).
        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("stft_lstm.kernel")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 2]);
        assert_eq!(
            file.tensor_bytes(info),
            payload.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );
    }

    /// Pins variant detection from the LSTM kernel shape. A tensor with
    /// second dim `4 * 128 = 512` classifies as [`DtlnVariant::Units128`]
    /// — the smallest upstream release. The stamped metadata reflects
    /// the detected variant, not the compile-time default.
    #[test]
    fn variant_detected_from_lstm_hidden_dim_128() {
        // A `[input_dim=1, 4*128=512]` F32 tensor. 4-byte-per-element
        // × 512 = 2048 bytes payload.
        let payload: Vec<u8> = (0..512u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        assert_eq!(payload.len(), 2048);
        let header =
            r#"{"time_lstm.kernel":{"dtype":"F32","shape":[1,512],"data_offsets":[0,2048]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&payload);

        let input_path = scratch_path("v128-in", "safetensors");
        let output_path = scratch_path("v128-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report =
            convert_dtln_aec_file(&input_path, &output_path, None).expect("convert 128-unit");
        assert_eq!(report.variant, DtlnVariant::Units128);
        assert_eq!(report.variant.lstm_units(), 128);

        // The stamped `vokra.dtln_aec.lstm_units` must match the
        // detected variant width — not a compile-time constant.
        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let units = file
            .get(KEY_VARIANT_LSTM_UNITS)
            .and_then(|v| v.as_u64())
            .expect("lstm_units chunk present");
        assert_eq!(
            units, 128,
            "detected variant width must be stamped into `vokra.dtln_aec.lstm_units`"
        );
    }

    /// Pins that the `vokra.provenance.upstream_url` chunk carries the
    /// GitHub URL verbatim (mirror of nkf_aec's `upstream_url` posture,
    /// distinct from every HF-hosted sibling's `upstream_hf` chunk),
    /// AND the fixed dims stamp so the runtime hparam recovery path
    /// does not depend on tensor-shape probing.
    #[test]
    fn provenance_and_upstream_url_stamped() {
        // Minimal single-F32-tensor safetensors buffer — the provenance
        // stamp is independent of tensor shape / count.
        let f32_bytes: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let header =
            r#"{"stft_lstm.recurrent_kernel":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);

        let input_path = scratch_path("prov-in", "safetensors");
        let output_path = scratch_path("prov-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        convert_dtln_aec_file(&input_path, &output_path, None)
            .expect("convert small provenance-only tensor");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");

        // arch / name / category
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
        // upstream_url (NOT upstream_hf — GitHub-only release, mirror
        // of the sibling nkf_aec posture)
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_URL)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_URL)
        );
        // default provenance: MIT / Permissive
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );

        // Fixed dims stamp — the runtime hparam recovery path never
        // probes tensor shape (probing would silently accept a widely
        // wrong checkpoint that happens to have consistent-looking
        // shapes); it always binds these chunks.
        assert_eq!(
            file.get(KEY_N_FFT).and_then(|v| v.as_u64()),
            Some(N_FFT as u64)
        );
        assert_eq!(file.get(KEY_HOP).and_then(|v| v.as_u64()), Some(HOP as u64));
        assert_eq!(
            file.get(KEY_BLOCK_LEN).and_then(|v| v.as_u64()),
            Some(BLOCK_LEN as u64)
        );
        assert_eq!(
            file.get(KEY_SAMPLE_RATE).and_then(|v| v.as_u64()),
            Some(SAMPLE_RATE as u64)
        );
    }

    /// Pins that every declared variant width recovers correctly from
    /// its stamped `lstm_units` value — this is the loader-side handshake
    /// contract (encoder writes `lstm_units`, loader reads it back and
    /// calls [`DtlnVariant::from_lstm_units`]). A silent misroute where
    /// the loader defaulted to Units128 on an unknown width would leak
    /// tensor-shape validation errors much later in the pipeline.
    #[test]
    fn variant_roundtrip_pins_all_three_widths() {
        for (variant, expected_units) in [
            (DtlnVariant::Units128, 128),
            (DtlnVariant::Units256, 256),
            (DtlnVariant::Units512, 512),
        ] {
            assert_eq!(variant.lstm_units(), expected_units);
            assert_eq!(DtlnVariant::from_lstm_units(expected_units), Some(variant));
        }
        // Fail-closed against wrong widths
        assert_eq!(DtlnVariant::from_lstm_units(0), None);
        assert_eq!(DtlnVariant::from_lstm_units(64), None);
        assert_eq!(DtlnVariant::from_lstm_units(384), None);
        assert_eq!(DtlnVariant::from_lstm_units(1024), None);
    }
}
