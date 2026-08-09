//! **FRCRN** (Frequency Recurrent Convolutional Recurrent Network):
//! safetensors checkpoint → GGUF conversion (coverage-audit wave-a,
//! 2026-08-03).
//!
//! Input: the upstream FRCRN release from `alibabasglab/FRCRN`
//! (`github.com/alibabasglab/FRCRN`), also distributed inside the
//! ClearerVoice-Studio pipeline (`github.com/modelscope/ClearerVoice-Studio`).
//! Output: a GGUF carrying every float tensor plus the `vokra.provenance.*`
//! / `vokra.model.*` / `vokra.schema.*` metadata chunks a future native
//! `vokra-models::frcrn::*` implementation will read.
//!
//! # Model class
//!
//! FRCRN is a monaural speech-enhancement model that boosts feature
//! representation via **frequency recurrence** on top of a Complex-valued
//! U-Net + frequency-recurrent LSTM (Zhao et al., ICASSP 2022,
//! `arXiv:2206.07293` "FRCRN: Boosting Feature Representation Using
//! Frequency Recurrence for Monaural Speech Enhancement"). Category is
//! `"denoise"` — same tier as DeepFilterNet3
//! (`crates/vokra-convert/src/models/denoise.rs`), but the topology is
//! completely distinct (Complex U-Net + freq-recurrent LSTM vs the DFN3
//! ERB / DF-net stack) so the arch tag is deliberately not aliased with
//! `"denoise"` (silently sharing the DFN3 tag would misroute the runtime
//! dispatch — the future `vokra-models::frcrn::from_gguf` reads a
//! completely different tensor manifest than DeepFilterNet3).
//!
//! # License
//!
//! Both code and weights ship **Apache-2.0** end-to-end
//! (`github.com/alibabasglab/FRCRN/blob/main/LICENSE` and the
//! ClearerVoice-Studio umbrella `github.com/modelscope/ClearerVoice-Studio`
//! `LICENSE`, both Apache-2.0, verified from the upstream repos'
//! primary sources per the ticket's "Publish gate" `redistributable`
//! entry, `docs/tickets/coverage-audit-2026-08-03/wave-a/frcrn.md`).
//! Apache-2.0 is a `Permissive` license class — no runtime-side
//! attribution obligation (unlike NVIDIA's CC-BY 4.0 Parakeet-CTC or
//! Canary which stamp FR-MD-09 attribution text).
//!
//! # BF16 pass-through
//!
//! Every F32 / F16 / BF16 tensor passes through **verbatim** as the
//! matching GGUF type (BF16 emits type 30 = `GgmlType::BF16`, no
//! convert-time widening — the runtime widens BF16 → f32 losslessly at
//! load via the single choke point `crates/vokra-core/src/gguf/quant/
//! mod.rs decode_bf16`). Mirror of `qwen3_tts` / `vibevoice` /
//! `voxcpm2` / `wespeaker` / `emotion2vec` — the landed sibling posture
//! that keeps the CI cache footprint at the smallest tensor payload
//! while preserving the exact upstream bit pattern.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM /
//! VibeVoice / WeSpeaker / emotion2vec contract). Real-weight parity
//! binding is a follow-up wave gated on the upstream tensor-name
//! manifest fetch + license §3.1 sign-off (`docs/license-audit.md`);
//! this converter passes every float tensor through unchanged so a
//! future `FrcrnWeights::from_gguf` can walk the same names.
//!
//! # Prep script bridge
//!
//! The FRCRN upstream release ships torch `.pt` (or `.pth`) checkpoints
//! (see `alibabasglab/FRCRN` + the ClearerVoice-Studio ModelScope hub
//! download logic), so a downstream user runs the sidecar
//! `tools/parity/frcrn_prepare_checkpoint.py` (uv-managed, Python 3.12)
//! to flatten the pickle to safetensors before invoking this converter —
//! the same posture the DFN3 / DAC / Kokoro / UTMOS / SBV2 converters
//! use. The runtime never sees Python / torch (FR-LD-05).
//!
//! # No ONNX (permanent)
//!
//! FRCRN is distributed as torch pickles + a Python pipeline; this
//! converter **never** touches ONNX (FR-LD-05); the enhancement pipeline
//! is re-implemented natively in a future
//! `crates/vokra-models/src/frcrn/` module (whisper.cpp 型 self
//! re-implementation, CLAUDE.md 設計判断 4).
//!
//! # Wiring status
//!
//! Fully wired: `ModelKind::Frcrn` + `from_arg("frcrn" | …)` +
//! `as_arg() == "frcrn"` + `convert_file` dispatch arm all land with this
//! module (mirror of the ecapa_tdnn / wespeaker / speaker_3d /
//! emotion2vec wiring landed 2026-07-25).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for FRCRN GGUFs. Distinct from every sibling arch
/// tag — silently aliasing `"denoise"` (which DeepFilterNet3 owns) would
/// misroute the runtime dispatch (an ERB / DF-net loader would try to
/// interpret a Complex U-Net + freq-recurrent LSTM checkpoint).
pub const ARCH: &str = "frcrn";

/// `vokra.model.name` value written for the canonical FRCRN GGUF.
pub const NAME: &str = "frcrn";

/// `vokra.model.category` value — the second `"denoise"` model in the
/// converter tree (after DeepFilterNet3). Consumed by the model-card
/// generator + zoo manifest tier gate.
pub const CATEGORY: &str = "denoise";

/// Ad-hoc metadata key for the model category. Kept as a converter-side
/// constant (not a `chunks::KEY_*` alias) until a sibling `category`
/// consumer lands in `vokra-core` — mirror of the wespeaker /
/// speaker_3d / emotion2vec local constant.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Upstream repository slug (`org/name`) recorded under
/// `vokra.provenance.upstream_hf` so a downstream consumer can trace the
/// artifact back to its serving location. The value points at the
/// original author's GitHub repo — the ClearerVoice-Studio mirror is
/// noted in the free-text `vokra.provenance.source` stamp.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
pub const UPSTREAM_HF: &str = "alibabasglab/FRCRN";

/// Canonical weight license SPDX (`apache-2.0`). Overrides via the
/// [`convert_frcrn_file`] `license` parameter — the standing mechanism
/// for "implementation is clean-room MIT but the upstream distributed
/// checkpoint is another license" scenarios (mirror of
/// `convert_file_licensed` in `lib.rs` and the `license` arg on
/// `convert_wespeaker_file` / `convert_emotion2vec_file`).
pub const DEFAULT_LICENSE: &str = "apache-2.0";

/// Outcome of an FRCRN conversion.
///
/// All counters are additive and default to zero — a zero-tensor
/// checkpoint returns `FrcrnReport::default()` and the caller remains
/// responsible for surfacing the "no float tensors" loud note (mirror
/// of the qwen3_tts / vibevoice / voxcpm2 / wespeaker / emotion2vec
/// `Report` pattern). `read == written + skipped_non_float` is an
/// invariant preserved by [`convert_frcrn_file`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct FrcrnReport {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 all go through
    /// the same byte-copy path since the BF16 pass-through landed
    /// 2026-07-25).
    pub written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time; anything
    /// that reaches this arm signals a reader change upstream).
    pub skipped_non_float: usize,
    /// Of the tensors in `written`, how many were BF16 (subset counter).
    /// Emits GGUF type 30 verbatim; runtime widens BF16 → f32 losslessly
    /// via the single choke point `crates/vokra-core/src/gguf/quant/mod.rs
    /// decode_bf16` (BF16 = top 16 bits of an f32 — `bits << 16` is exact).
    pub bf16_passthrough: usize,
}

/// Reads a safetensors checkpoint at `input` and writes an FRCRN GGUF to
/// `output`.
///
/// Every F32 / F16 / BF16 tensor is emitted verbatim under its upstream
/// name; the `vokra.provenance.*` + `vokra.model.*` chunk groups pin the
/// upstream slug, weight license, and model category so the zoo
/// manifest + model-card generator can gate on the artifact alone (no
/// side-car lookup). `vokra.schema.*` is written unconditionally by the
/// GGUF writer.
///
/// `license` overrides `DEFAULT_LICENSE` (`"apache-2.0"`) — the same
/// mechanism `lib.rs::convert_file_licensed` uses when the implementation
/// is clean-room but the redistributed checkpoint carries a different
/// SPDX.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_frcrn_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<FrcrnReport, ConvertError> {
    // Whole-file read: FRCRN ships as a single ~30 MB `.pt` (per the
    // ticket size estimate), which the prep script flattens into an
    // equally small `.safetensors` — well under the streaming threshold
    // the Moshi 15 GB / Voxtral 8.7 GB converters run. Any future
    // 500M+ FRCRN sibling would swap this call for
    // `SafetensorsFileReader::open` + `GgufStreamWriter::begin` per the
    // moshi.rs / qwen3_tts.rs ADR.
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = apache-2.0 (upstream `alibabasglab/FRCRN`
    // `LICENSE` + the ClearerVoice-Studio umbrella `LICENSE`, both
    // Apache-2.0, primary-source verified per the wave-a ticket).
    // `license` overrides for callers who obtained the weight under a
    // different SPDX (see `convert_file_licensed` in `lib.rs`).
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some(
            "alibabasglab/FRCRN (Complex U-Net + frequency-recurrent LSTM, \
             DNS Challenge 2022, apache-2.0; also distributed via \
             modelscope/ClearerVoice-Studio)",
        ),
    );

    let mut report = FrcrnReport::default();
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (docs/adr/qwen3-tts-bf16.md, strategy A_passthrough); the runtime
    // widens BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. Mirrors
    // `qwen3_tts::convert` / `vibevoice::convert` / `voxcpm2::convert` /
    // `wespeaker::convert` / `emotion2vec::convert`.
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )
                .map_err(|e| ConvertError::Gguf(e.to_string()))?;
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
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, out_bytes).map_err(ConvertError::Io)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Per-process, per-test scratch path in the system temp dir (moshi
    /// / emotion2vec / wespeaker test pattern — no external `tempfile`
    /// dep, preserving zero-dep NFR-DS-02). The nanosecond suffix
    /// separates parallel `cargo test` runs so they cannot clobber each
    /// other's files.
    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-frcrn-{}-{}-{}.bin",
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
    /// patterns so a byte-identity assert catches any silent widen /
    /// downcast attempt — the raw zeroed payload would round-trip
    /// trivially through F32 / F16 widen and defeat the pin (mirror of
    /// emotion2vec's fixture).
    fn synthetic_bf16_safetensors() -> (Vec<u8>, Vec<u8>) {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");
        let header = r#"{"encoder.complex_conv0.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&bf16);
        (buf, bf16)
    }

    /// Builds a synthetic safetensors buffer with one F32 tensor
    /// (`shape=[2,3]`, 24 B) followed by one F16 tensor
    /// (`shape=[1,4]`, 8 B). The offsets are chosen so the tensors are
    /// contiguous in the data region — mirror of emotion2vec's fixture.
    fn synthetic_f32_and_f16_safetensors() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let f32_vals: [f32; 6] = [1.0, -2.0, 3.5, -0.25, 100.0, 0.001];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 24, "6 elements × 4 bytes F32 payload");
        let f16_patterns: [u16; 4] = [0x3C00, 0xC000, 0x4200, 0x0001];
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|p| p.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 8, "4 elements × 2 bytes F16 payload");
        let header = r#"{"encoder.complex_conv0.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"decoder.complex_conv0.weight":{"dtype":"F16","shape":[1,4],"data_offsets":[24,32]}}"#;
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
    /// input. Mirror of the emotion2vec / wespeaker equivalent.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (input_bytes, bf16_payload) = synthetic_bf16_safetensors();
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_frcrn_file(&input, &output, None).expect("convert");

        // Counters: single BF16 tensor read + written + BF16 subset.
        assert_eq!(report.read, 1, "one tensor visible in safetensors header");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of emotion2vec)"
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
            .tensor_info("encoder.complex_conv0.weight")
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
            "category chunk pins the second `denoise` model in the tree"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF),
            "upstream slug pins traceability back to alibabasglab/FRCRN"
        );
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
    /// remain 0. Guards against a naive `if bf16 { … } else` refactor.
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        let (input_bytes, f32_payload, f16_payload) = synthetic_f32_and_f16_safetensors();
        let input = scratch_path("f32f16-in");
        let output = scratch_path("f32f16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_frcrn_file(&input, &output, None).expect("convert");

        assert_eq!(report.read, 2, "two tensors visible in header");
        assert_eq!(report.written, 2, "both F32 and F16 must pass through");
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32+F16-only input must leave the BF16 subset counter at Default 0"
        );

        // Both tensors survive the round-trip with their upstream names
        // and dtypes preserved.
        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        let f32_info = file
            .tensor_info("encoder.complex_conv0.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(f32_info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(f32_info), f32_payload.as_slice());

        let f16_info = file
            .tensor_info("decoder.complex_conv0.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(f16_info.dimensions, vec![1, 4]);
        assert_eq!(file.tensor_bytes(f16_info), f16_payload.as_slice());

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// License override: the caller-supplied SPDX must replace the
    /// default `apache-2.0` stamp on the artifact (mirror of the
    /// wespeaker / emotion2vec test — proves the standing
    /// `convert_file_licensed` override reaches this arm).
    #[test]
    fn license_override_replaces_default_stamp() {
        let (input_bytes, _) = synthetic_bf16_safetensors();
        let input = scratch_path("license-in");
        let output = scratch_path("license-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        // Override with `mit` (a Permissive alternative to apache-2.0)
        // — the SPDX must land in the license stamp and the class must
        // re-derive to Permissive.
        convert_frcrn_file(&input, &output, Some("mit")).expect("convert with license override");

        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
            "override SPDX must land in vokra.provenance.license"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "MIT still resolves to Permissive"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
