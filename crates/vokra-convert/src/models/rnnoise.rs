//! **Xiph RNNoise v0.2**: safetensors checkpoint → GGUF conversion
//! (coverage-audit 2026-08-03 Wave A ticket).
//!
//! Input: a **prepared** safetensors flattened from the upstream
//! `weights_blob_9.bin` release asset (Xiph's real-time noise reduction,
//! `github.com/xiph/rnnoise/releases/tag/v0.2`). Output: a GGUF carrying
//! every float tensor plus the `vokra.provenance.*` / `vokra.model.*` /
//! `vokra.schema.*` metadata chunks the future native
//! `vokra-models::rnnoise::*` implementation will read.
//!
//! # Why a prep step
//!
//! Xiph ships the model as a compact C-array blob (`weights_blob_9.bin` /
//! `rnn_data.c`) that a Python side-car (`tools/parity/rnnoise_prepare_checkpoint.py`)
//! walks with `numpy.frombuffer` and re-emits as a safetensors file — that
//! way the converter itself sees only the safetensors surface every other
//! Vokra converter reads (mirror of the DAC / Denoise / CSM path where the
//! upstream release is torch-pickle and is flattened before entering
//! `vokra-convert`; **no Python enters the runtime** — NFR-DS-02). The
//! prep script lives under `tools/parity/` and is invoked by the operator
//! on their box, never by the converter.
//!
//! # Model class
//!
//! RNNoise is Xiph's real-time speech denoiser (Valin 2018,
//! `arXiv:1709.08243`): 22-band Bark filterbank features feed a compact
//! GRU stack (`input_dense` 42→24 → `vad_gru` / `noise_gru` /
//! `denoise_gru` → `denoise_output` 96→22, plus a `vad_output` 24→1
//! auxiliary head). The topology is intentionally distinct from
//! DeepFilterNet3 (which drives Vokra's existing `Denoise` ModelKind and
//! stamps `vokra.denoise.*` with a DFN3-shaped hparam chunk): DFN3 is a
//! complex-Conv + ERB-band + deep-filtering architecture, RNNoise is a
//! tiny GRU stack over Bark bands, and silently sharing the arch tag
//! would misroute the runtime dispatch. The category is `denoise`
//! (shared audio-dialect §Speech Enhancement op family — see
//! `crates/vokra-eval/data/zoo/manifest.txt` line 316-322 for the zoo
//! excluded-record row that already recognises RNNoise as a
//! `family = rnnoise` denoise alternative).
//!
//! # License
//!
//! Both code and weights ship **BSD-3-Clause** end-to-end
//! (`github.com/xiph/rnnoise/blob/main/COPYING` — the standard
//! three-clause BSD text with `Copyright (c) 2017-2024, Mozilla /
//! Xiph.Org Foundation / Jean-Marc Valin`). BSD-3-Clause is a
//! `Permissive` license class in `crates/vokra-core/src/compliance/license_class.rs`
//! (the `"bsd"` token in `PERMISSIVE_TOKENS` — same commercial verdict
//! as `apache-2.0` / `mit`, no runtime-side attribution obligation
//! beyond the standard-BSD retention-of-notice clause every downstream
//! satisfies by shipping the LICENSE alongside the GGUF).
//!
//! # BF16 posture
//!
//! Every F32 / F16 / BF16 tensor passes through **verbatim** as the
//! matching GGUF type (BF16 emits type 30 = `GgmlType::BF16`, no
//! convert-time widening — the runtime widens BF16 → f32 losslessly at
//! load via the single choke point `crates/vokra-core/src/gguf/quant/
//! mod.rs decode_bf16`). Mirror of `qwen3_tts` / `vibevoice` /
//! `voxcpm2` / `neucodec` / `emotion2vec` — the landed sibling posture
//! that keeps the CI cache footprint at the smallest tensor payload
//! while preserving the exact upstream bit pattern.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim** —
//! whatever `tools/parity/rnnoise_prepare_checkpoint.py` emits from
//! `weights_blob_9.bin`. Real-weight parity binding is a follow-up wave
//! gated on Xiph reference-C-forward parity + license §3.1 sign-off
//! (`docs/license-audit.md`); this converter passes every float tensor
//! through unchanged so a future `RnnoiseWeights::from_gguf` can walk
//! the same names.
//!
//! # No ONNX (permanent)
//!
//! RNNoise is distributed as a C-array blob + `librnnoise` (a C library);
//! this converter **never** touches ONNX (FR-LD-05). The upstream C
//! forward will be re-implemented natively when a
//! `crates/vokra-models/src/rnnoise/` lands (whisper.cpp 型 self
//! re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for RNNoise GGUFs. Distinct from every sibling
/// arch tag because RNNoise's tiny GRU + 22-band Bark topology is
/// unrelated to DeepFilterNet3's complex-Conv + ERB deep-filtering
/// topology (both `denoise` category but structurally incompatible) —
/// silently sharing the `denoise` arch tag would mis-route the runtime
/// dispatch (a DFN3 loader would try to interpret RNNoise's `input_dense`
/// as `enc_conv0`).
pub const ARCH: &str = "rnnoise";

/// `vokra.model.name` value written for the canonical RNNoise v0.2 GGUF.
/// Matches the `huggingface.co/vokra/rnnoise-v0.2` publish slug and the
/// `as_arg` return value in `lib.rs` so the CLI / model-card / publish
/// pipe all agree on a single identifier.
pub const NAME: &str = "rnnoise-v0.2";

/// `vokra.model.category` value — the second `denoise` model in the
/// converter tree after `Denoise` (DeepFilterNet3). Consumed by the
/// model-card generator + zoo manifest tier gate so a downstream picks
/// the correct decode path (`vokra-eval/data/zoo/manifest.txt` already
/// carries the `family = rnnoise` excluded-record row).
pub const CATEGORY: &str = "denoise";

/// `vokra.provenance.upstream_url` value — the GitHub Release asset
/// where the `weights_blob_9.bin` blob ships from. RNNoise is not
/// distributed on Hugging Face, so the standard
/// `vokra.provenance.upstream_hf` slot does not apply; the sibling
/// converters (emotion2vec / neucodec / …) all stamp `upstream_hf`
/// because they ship from HF, and a URL-shaped stamp would misrepresent
/// their serving location. Kept as an ad-hoc converter-side key
/// (namespaced under `vokra.provenance.*` so a future `chunks::KEY_*`
/// alias can absorb it without breaking existing GGUFs).
pub const UPSTREAM_URL: &str = "https://github.com/xiph/rnnoise/releases/tag/v0.2";

/// Canonical weight license SPDX (`bsd-3-clause`). Overrides via the
/// [`convert_rnnoise_file`] `license` parameter — the standing
/// mechanism for "implementation is clean-room BSD-3 but the upstream
/// redistributed checkpoint is another SPDX" scenarios (mirror of
/// `convert_file_licensed` in `lib.rs`). Lowercase per SPDX convention
/// and matching the `PERMISSIVE_TOKENS` lookup path in
/// `LicenseClass::from_license_str` (which lower-cases before matching).
pub const DEFAULT_LICENSE: &str = "bsd-3-clause";

/// Ad-hoc metadata key for the model category. Kept as a converter-side
/// constant (not a `chunks::KEY_*` alias) mirroring the emotion2vec /
/// neucodec convention until a sibling `category` consumer lands in
/// `vokra-core`.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Ad-hoc metadata key for the upstream URL. Kept converter-side (no
/// `chunks::KEY_*` alias yet) because Vokra's other converters all ship
/// from HF and use `vokra.provenance.upstream_hf`; RNNoise is the first
/// converter with a raw URL provenance slot. Namespaced under
/// `vokra.provenance.*` so a future promotion to a `chunks::KEY_*`
/// alias is byte-compatible with existing GGUFs.
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

/// Outcome of an RNNoise conversion.
///
/// All counters are additive and default to zero — a zero-tensor
/// checkpoint returns `RnnoiseReport::default()` and the caller remains
/// responsible for surfacing the "no float tensors" loud note (mirror of
/// the neucodec / emotion2vec / qwen3_tts `Report` pattern with the
/// `read` counter pinning the total tensor budget the safetensors reader
/// surfaced).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RnnoiseReport {
    /// Total tensors seen in the upstream safetensors header (the sum of
    /// `written + skipped_non_float`). Pins the budget so a truncated
    /// header cannot silently drop tensors without the caller noticing.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 all go through
    /// the same byte-copy path — the BF16 pass-through landed 2026-07-25).
    pub written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time; anything
    /// that reaches this arm is a quantized dtype the runtime is not
    /// expected to consume). RNNoise's on-disk blob is int8 quantized,
    /// so the prep script widens to F32 before writing the safetensors
    /// input to this converter; a real prep-script run should always
    /// leave this counter at 0.
    pub skipped_non_float: usize,
    /// Of the tensors in `written`, how many were BF16 (subset counter).
    /// Emits GGUF type 30 verbatim; runtime widens BF16 → f32 losslessly
    /// via the single choke point `crates/vokra-core/src/gguf/quant/mod.rs
    /// decode_bf16` (BF16 = top 16 bits of an f32 — `bits << 16` is exact).
    pub bf16_passthrough: usize,
}

/// Reads a safetensors checkpoint at `input` and writes an RNNoise v0.2
/// GGUF to `output`.
///
/// Every F32 / F16 / BF16 tensor is emitted verbatim under its upstream
/// name; the `vokra.provenance.*` and `vokra.model.*` chunk groups pin
/// the upstream URL, weight license, and model category so the zoo
/// manifest and model-card generator can gate on the artifact alone (no
/// side-car lookup). `vokra.schema.*` is written unconditionally by the
/// GGUF writer.
///
/// `license` overrides [`DEFAULT_LICENSE`] (`"bsd-3-clause"`) — the same
/// mechanism `lib.rs::convert_file_licensed` uses when the implementation
/// is clean-room but the redistributed checkpoint carries a different
/// SPDX.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_rnnoise_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<RnnoiseReport, ConvertError> {
    // Whole-file read: RNNoise v0.2 ships as a ~90 KB weight blob (order
    // of KB, not MB) — no need for the streaming path the Moshi 15 GB /
    // Voxtral 8.7 GB converters run.
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    // Ad-hoc URL stamp — RNNoise ships from a GitHub Release, not HF, so
    // the standard `KEY_PROVENANCE_UPSTREAM_HF` slot (used by neucodec /
    // emotion2vec / …) would misrepresent the serving location. Both
    // slots live under `vokra.provenance.*`, so a downstream that looks
    // up either can find its answer without a converter-specific decoder.
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);

    // Self-describing redistribution: the artifact carries its own
    // licence. RNNoise ships BSD-3-Clause end-to-end (COPYING at
    // `github.com/xiph/rnnoise/blob/main/COPYING` — the standard
    // three-clause BSD text). The `license` override lets a downstream
    // repackager stamp a different SPDX if they redistribute under
    // stricter terms (the same knob `convert_file_licensed` exposes in
    // `lib.rs`).
    let effective_license = license.unwrap_or(DEFAULT_LICENSE);
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        effective_license,
        Some(NAME),
        Some(UPSTREAM_URL),
    );

    let mut report = RnnoiseReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (docs/adr/qwen3-tts-bf16.md, strategy A_passthrough); the runtime
    // widens BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. Mirrors
    // `neucodec::convert` / `emotion2vec::convert` / `qwen3_tts::convert`.
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
    /// (emotion2vec test pattern — no external `tempfile` dep, preserving
    /// zero-dep NFR-DS-02). The nanosecond suffix separates the two tests
    /// in this module so a parallel `cargo test` run cannot clobber files
    /// across them.
    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-rnnoise-{}-{}-{}.bin",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        p
    }

    /// Builds a synthetic safetensors buffer with a single BF16 tensor.
    ///
    /// The payload is chosen from a known set of non-zero BF16 bit
    /// patterns (`1.0`, `-2.5`, `0.15625`, `3.5`, `-0.5`, `42.0`) so a
    /// byte-identity assert catches any silent widen / downcast attempt
    /// — a zeroed payload would round-trip trivially through F32 / F16
    /// widen and defeat the pin.
    fn synthetic_bf16_safetensors() -> (Vec<u8>, Vec<u8>) {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");
        // Tensor name modelled on the RNNoise topology `input_dense`
        // layer (42→24 kernel) documented in
        // `github.com/xiph/rnnoise/blob/main/src/denoise.c` — the shape
        // here is a stand-in `[2, 3]` for the synthetic pass-through
        // pin; the real prep-script tensor names are the follow-up.
        let header =
            r#"{"input_dense.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&bf16);
        (buf, bf16)
    }

    /// Builds a synthetic safetensors buffer with one F32 tensor
    /// (`shape=[2,3]`, 24 B) followed by one F16 tensor (`shape=[1,4]`,
    /// 8 B). The offsets are chosen so the tensors are contiguous in the
    /// data region.
    fn synthetic_f32_and_f16_safetensors() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        // F32 payload: 6 non-zero floats so a silent widen would flip a
        // fence rather than trivially round-trip a zero buffer.
        let f32_vals: [f32; 6] = [1.0, -2.0, 3.5, -0.25, 100.0, 0.001];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 24, "6 elements × 4 bytes F32 payload");
        // F16 payload: 4 half-floats with known non-zero bit patterns.
        let f16_patterns: [u16; 4] = [0x3C00, 0xC000, 0x4200, 0x0001];
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|p| p.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 8, "4 elements × 2 bytes F16 payload");
        // Header declares F32 first, then F16 in the data region. Tensor
        // names track the RNNoise topology (`vad_gru` / `denoise_output`
        // documented in `src/denoise.c`); shapes are synthetic stand-ins
        // for the pass-through pin.
        let header = r#"{"vad_gru.kernel":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"denoise_output.bias":{"dtype":"F16","shape":[1,4],"data_offsets":[24,32]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&f32_bytes);
        buf.extend_from_slice(&f16_bytes);
        (buf, f32_bytes, f16_bytes)
    }

    /// BF16 pass-through: the upstream BF16 checkpoint must survive the
    /// file-based converter round-trip with its dtype preserved (GGUF
    /// type 30 = `GgmlType::BF16`) and its payload byte-identical to the
    /// input. Mirrors neucodec / emotion2vec / qwen3_tts / vibevoice /
    /// voxcpm2 / moshi / voxtral.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (input_bytes, bf16_payload) = synthetic_bf16_safetensors();
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_rnnoise_file(&input, &output, None).expect("convert");

        // Counters: single BF16 tensor read + written + BF16 subset.
        assert_eq!(report.read, 1, "one tensor visible in safetensors header");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of neucodec)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 subset counter must record the pass-through"
        );

        // Round-trip: dtype preserved, payload byte-identical (no silent widen).
        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        let info = file
            .tensor_info("input_dense.weight")
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
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "category chunk pins the denoise family (sibling of DeepFilterNet3)"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_URL)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_URL),
            "upstream URL chunk pins the GitHub Release the blob ships from"
        );
        // Schema stamp is written unconditionally by the GGUF writer.
        assert!(
            file.get(chunks::KEY_SCHEMA_VERSION).is_some(),
            "vokra.schema.version must be stamped"
        );
        assert!(
            file.get(chunks::KEY_SCHEMA_PRODUCER).is_some(),
            "vokra.schema.producer must be stamped"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// F32 + F16 pass-through: two float tensors of distinct dtypes in
    /// the same input must both reach the pass-through arm without
    /// collapsing into a single dtype branch, and the BF16 counter must
    /// remain 0 (default). Guards against a naive `if bf16 { ... } else`
    /// refactor.
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        let (input_bytes, f32_payload, f16_payload) = synthetic_f32_and_f16_safetensors();
        let input = scratch_path("f32f16-in");
        let output = scratch_path("f32f16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_rnnoise_file(&input, &output, None).expect("convert");

        assert_eq!(report.read, 2, "two tensors visible in header");
        assert_eq!(report.written, 2, "both F32 and F16 must pass through");
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32+F16-only input must leave the BF16 subset counter at the Default 0"
        );

        // Both tensors survive the round-trip with their upstream names
        // and dtypes preserved.
        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        let f32_info = file
            .tensor_info("vad_gru.kernel")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(f32_info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(f32_info), f32_payload.as_slice());

        let f16_info = file
            .tensor_info("denoise_output.bias")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(f16_info.dimensions, vec![1, 4]);
        assert_eq!(file.tensor_bytes(f16_info), f16_payload.as_slice());

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// License override: a caller with an SPDX id distinct from the
    /// default (`bsd-3-clause`) must land on the artifact's licence
    /// stamp; the license class is re-derived from the override string
    /// (mirror of the `convert_file_licensed` pattern in `lib.rs`). Uses
    /// `mit` (still `Permissive`) so the class stays the same for the
    /// pin — a class-changing override is tested end-to-end at the
    /// `convert_file_licensed` boundary in the integration test.
    #[test]
    fn license_override_lands_on_the_artifact() {
        let (input_bytes, _) = synthetic_bf16_safetensors();
        let input = scratch_path("lic-in");
        let output = scratch_path("lic-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let _ = convert_rnnoise_file(&input, &output, Some("mit")).expect("convert with license");

        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
            "override MUST land on the raw licence slot"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
