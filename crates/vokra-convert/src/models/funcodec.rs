//! **FunCodec** (ModelScope, SoundStream/EnCodec-family): safetensors →
//! GGUF conversion (SoTA plan Phase 5, 2026-07-25).
//!
//! Input: an upstream `alibaba-damo/audio_codec-encodec-zh_en-general-
//! 16k-nq32ds320-pytorch` safetensors checkpoint (**mit** — permissive,
//! commercial-OK per `docs/license-audit.md` §3.1 sign-off queue).
//! Output: a GGUF carrying every float tensor plus the `vokra.provenance.*`
//! and `vokra.model.*` metadata chunks. The upstream release is a
//! ModelScope FunCodec — a SoundStream / EnCodec-family neural audio
//! codec with 32 residual codebooks and a 320× down-sampling ratio at
//! 16 kHz (name suffix `nq32ds320`).
//!
//! # Category — codec
//!
//! Category tag `codec` is written under `vokra.model.category` so the
//! runtime dispatcher / model-zoo tooling can distinguish codec GGUFs
//! from LM / TTS / ASR ones without inferring from tensor names. This
//! mirrors the family-marker string every other converter carries under
//! `vokra.<family>.model_family` — a category axis one level up.
//!
//! # BF16 posture — pass-through, mirror of qwen3-tts / vibevoice / voxcpm2
//!
//! F32 / F16 / BF16 tensors pass through **verbatim** under their
//! upstream safetensors names (GGUF types 0 / 1 / 30 respectively). No
//! convert-time widening — the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 = top 16
//! bits of an f32 — `bits << 16` is exact). Non-float dtypes (i.e.
//! anything the safetensors reader accepts that is not F32 / F16 / BF16
//! today) land on the `skipped_non_float` counter.
//!
//! # License override — outer `--license <spdx>` boundary
//!
//! `convert_funcodec_file(_, _, license)` accepts an SPDX override
//! (mirror of the `convert_file_licensed` / `convert_file --license
//! <spdx>` boundary in `crates/vokra-convert/src/lib.rs:1533-1550`). When
//! supplied, it overwrites `vokra.provenance.weight_license`,
//! `vokra.provenance.license` and `vokra.provenance.source` in place —
//! the model-id / model-name stamps are preserved. Absent, the
//! upstream MIT verdict rides through.
//!
//! # Native runtime and real-weight parity
//!
//! `vokra-models::funcodec` now authenticates the complete released
//! 230-tensor name/shape manifest and executes the official residual-VQ plus
//! SEANet decoder on CPU or Metal. This converter still preserves every
//! F32 / F16 / BF16 tensor under its upstream name; the runtime's complete
//! manifest and individual inference-shape checks are the authoritative
//! **FR-EX-08** bind gate. Independent official-reference generation lives in
//! `tools/parity/funcodec/dump_reference.py`; its first real VAST CPU and
//! Apple Metal measurements remain required before recording a numerical
//! parity pass.
//!
//! # No ONNX (permanent)
//!
//! FunCodec ships as a ModelScope PyTorch checkpoint; the converter
//! **never** touches ONNX (FR-LD-05). The token-to-waveform pipeline is
//! re-implemented natively in `crates/vokra-models/src/funcodec.rs`.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for FunCodec GGUFs.
pub(crate) const ARCH: &str = "funcodec";
/// `vokra.model.name` value for the canonical
/// `audio_codec-encodec-zh_en-general-16k-nq32ds320-pytorch` release.
pub(crate) const NAME: &str = "funcodec-encodec-zh-en-16k-nq32-ds320";
/// Upstream Hugging Face repository path — provenance breadcrumb.
pub(crate) const UPSTREAM_HF: &str =
    "alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds320-pytorch";
/// Upstream declared weight license (SPDX id, lower-case per the
/// `docs/license-audit.md` §3.1 sign-off table).
pub(crate) const DEFAULT_LICENSE: &str = "mit";
/// Category axis — one level up from `vokra.<family>.model_family`,
/// distinguishing codec GGUFs from LM / TTS / ASR ones.
pub(crate) const CATEGORY: &str = "codec";

/// `vokra.model.category` — categorical family tag written by codec
/// converters so the dispatcher / model-zoo tooling can filter GGUFs by
/// role. Kept as a converter-local constant while there is only one
/// codec-family emitter using it; the day a second lands, this moves to
/// `vokra-core::gguf::chunks`.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
/// `vokra.provenance.upstream_hf` — provenance breadcrumb for the
/// upstream Hugging Face path (extra to `chunks::KEY_PROVENANCE_MODEL_ID`
/// so downstream tooling can distinguish "who published this" from
/// "what did we call it internally"). Converter-local for the same
/// reason as [`KEY_MODEL_CATEGORY`].
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a FunCodec conversion.
///
/// Mirrors the counter shape used by `qwen3_tts::Qwen3TtsReport` /
/// `vibevoice::VibeVoiceReport` / `voxcpm2::VoxCpm2Report`: `written`
/// (F32/F16/BF16 pass-through), `skipped_non_float` (defensive
/// non-float land), `bf16_passthrough` (subset counter for the BF16
/// pass-through arm added 2026-07-25). Adds `read` (total tensor
/// entries observed on the input side) so callers can assert
/// `read == written + skipped_non_float` end-to-end without walking
/// the safetensors reader themselves.
#[derive(Debug, Default)]
pub struct FuncodecReport {
    /// Total tensor entries observed on the safetensors input side —
    /// every entry the reader hands to the walk, regardless of dtype.
    /// The `read == written + skipped_non_float` invariant lets callers
    /// prove no tensor was double-counted or dropped.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 — all three go
    /// through the same byte-copy path since the BF16 pass-through land
    /// 2026-07-25, mirror of `qwen3-tts` / `vibevoice` / `voxcpm2`).
    pub written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time; anything
    /// that reaches this arm is a dtype the runtime is not expected to
    /// consume as float weights).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; runtime widens BF16 → f32
    /// losslessly via the single choke point
    /// `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 = top
    /// 16 bits of an f32 — `bits << 16` is exact).
    pub bf16_passthrough: usize,
}

/// Convert a FunCodec safetensors checkpoint at `input` into a Vokra
/// GGUF at `output`. When `license` is `Some(spdx)`, the SPDX id
/// overrides the upstream MIT stamp — mirror of the
/// `convert_file --license <spdx>` boundary in `lib.rs`.
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// safetensors name. Non-float dtypes land on the [`FuncodecReport::skipped_non_float`]
/// counter (defensive — the safetensors reader rejects unknown dtypes
/// at parse time today). BF16 tensors additionally increment
/// [`FuncodecReport::bf16_passthrough`] as an observability subset
/// counter: the runtime widens BF16 → f32 losslessly at load via
/// `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 = top
/// 16 bits of an f32 — `bits << 16` is exact).
pub fn convert_funcodec_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<FuncodecReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Self-describing redistribution: the artifact carries its own licence.
    // FunCodec ships MIT (docs/license-audit.md §3.1 sign-off queue —
    // the SoTA plan Phase 5 add). MIT is a `Permissive` license class
    // — same commercial verdict as apache-2.0 (no runtime-side
    // attribution obligation), just a different SPDX string. The
    // outer `--license <spdx>` override below rewrites the trio in
    // place if the caller supplies one.
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        DEFAULT_LICENSE,
        Some(NAME),
        Some(UPSTREAM_HF),
    );

    let mut report = FuncodecReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the shared ADR
    // (docs/adr/qwen3-tts-bf16.md, strategy A_passthrough); the runtime
    // widens BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. Mirrors
    // `qwen3_tts::convert` / `vibevoice::convert` / `voxcpm2::convert`.
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

    // Override the stamped licence when the caller supplies the
    // distribution source's SPDX id (add_string overwrites the key in
    // place, so the model_id / source stamps stamped above are
    // preserved — only the licence and its class change). Mirror of
    // the `convert_file --license <spdx>` boundary in
    // `crates/vokra-convert/src/lib.rs:1533-1550`.
    if let Some(lic) = license {
        let class = LicenseClass::from_license_str(lic);
        b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, class.as_str());
        b.add_string(chunks::KEY_PROVENANCE_LICENSE, lic);
        // The built-in `source` string names the converter's default
        // licence (`funcodec-... (MIT)` via the model_id); once the
        // licence is overridden that parenthetical would contradict
        // it, so restate the source neutrally.
        b.add_string(
            chunks::KEY_PROVENANCE_SOURCE,
            &format!("upstream distribution source (licence {lic} per source)"),
        );
    }

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, &out_bytes)?;

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Produces a single-tensor safetensors byte buffer with a caller-
    /// supplied dtype tag, shape and raw payload. Mirrors the private
    /// `safetensors_one_bf16` helper in `models::qwen3_tts::tests` and
    /// generalizes it across dtypes.
    fn safetensors_one(dtype: &str, shape: &[u64], payload: &[u8]) -> Vec<u8> {
        let shape_str = shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"codec.tensor":{{"dtype":"{dtype}","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    /// Two-tensor safetensors buffer: F32 then F16, contiguous data
    /// region. Names disambiguate the read-order used to assert the
    /// `written == 2` counter.
    fn safetensors_f32_then_f16() -> Vec<u8> {
        let f32_bytes: Vec<u8> = [1.0_f32, -2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(f32_bytes.len(), 8);
        let f16_bytes: Vec<u8> = [0x3C00_u16, 0x4000, 0x4200]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(f16_bytes.len(), 6);
        let header = format!(
            r#"{{"codec.a_f32":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"codec.b_f16":{{"dtype":"F16","shape":[3],"data_offsets":[{},{}]}}}}"#,
            f32_bytes.len(),
            f32_bytes.len(),
            f32_bytes.len() + f16_bytes.len(),
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&f32_bytes);
        out.extend_from_slice(&f16_bytes);
        out
    }

    /// Per-test temp path pair. PID + label keeps the pair unique when
    /// two tests race in the same binary.
    fn temp_pair(label: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let mut input = std::env::temp_dir();
        input.push(format!(
            "vokra-funcodec-{label}-{}-in.safetensors",
            std::process::id()
        ));
        let mut output = std::env::temp_dir();
        output.push(format!(
            "vokra-funcodec-{label}-{}-out.gguf",
            std::process::id()
        ));
        (input, output)
    }

    /// Distinctive BF16 bit patterns (top-16 bits of the IEEE-754 f32
    /// encodings of 1.0, -2.5, 0.15625, 3.5, -0.5, 42.0). A silent
    /// widen-to-f32 or byte-swap would flip the payload-bytes assertion.
    fn distinctive_bf16_payload() -> Vec<u8> {
        [1.0_f32, -2.5, 0.15625, 3.5, -0.5, 42.0]
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// RED-phase pin (2026-07-25): the BF16 pass-through arm surfaces
    /// under the file-based entry point. Fixture writes a
    /// distinctive-payload BF16 safetensors, calls
    /// `convert_funcodec_file`, re-parses the emitted GGUF, and asserts
    /// dtype (GgmlType::BF16 = type 30, no convert-time widening) plus
    /// byte-identical payload.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let payload = distinctive_bf16_payload();
        assert_eq!(payload.len(), 12, "6 elements × 2 B BF16 payload");
        let blob = safetensors_one("BF16", &[2, 3], &payload);

        let (input, output) = temp_pair("bf16");
        std::fs::write(&input, &blob).expect("write input safetensors");

        let report = convert_funcodec_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 1, "one tensor observed on input");
        assert_eq!(report.written, 1, "BF16 must reach the pass-through arm");
        assert_eq!(report.skipped_non_float, 0, "BF16 is float — no skip");
        assert_eq!(report.bf16_passthrough, 1, "BF16 subset counter must fire");

        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");
        let info = file
            .tensor_info("codec.tensor")
            .expect("emitted GGUF must carry the tensor");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — GGUF dtype must remain BF16 (type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            payload.as_slice(),
            "BF16 payload must round-trip byte-for-byte (no silent widen)",
        );

        // Provenance stamp: upstream MIT rides through when `license` is None.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE),
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "vokra.model.category must be `codec`",
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF),
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// RED-phase pin (2026-07-25): the F32 + F16 legs of the union
    /// match arm surface too, and the BF16 subset counter stays at 0
    /// when no BF16 tensor is present. Guards against a regression that
    /// would count every write as BF16 (or drop F16 from the
    /// pass-through arm).
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        let blob = safetensors_f32_then_f16();
        let (input, output) = temp_pair("f32-and-f16");
        std::fs::write(&input, &blob).expect("write input safetensors");

        let report = convert_funcodec_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 2, "two tensors observed on input");
        assert_eq!(report.written, 2, "both F32 and F16 must pass through");
        assert_eq!(report.skipped_non_float, 0, "no non-float tensors here");
        assert_eq!(
            report.bf16_passthrough, 0,
            "no BF16 tensor — subset counter stays at Default 0",
        );

        // Round-trip: both tensors survive with their upstream names + dtypes.
        let out_bytes = std::fs::read(&output).expect("read emitted GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse emitted GGUF");
        let a = file.tensor_info("codec.a_f32").expect("F32 tensor present");
        assert_eq!(a.dtype, GgmlType::F32);
        assert_eq!(a.dimensions, vec![1, 2]);
        let b = file.tensor_info("codec.b_f16").expect("F16 tensor present");
        assert_eq!(b.dtype, GgmlType::F16);
        assert_eq!(b.dimensions, vec![3]);

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
