//! **Kimi-Audio-7B-Instruct**: safetensors checkpoint → Vokra GGUF
//! conversion (TDD skeleton, 2026-07-25).
//!
//! # Upstream
//!
//! - HF: `moonshotai/Kimi-Audio-7B-Instruct`
//! - License (SPDX): `mit`
//! - Category: `s2s` (end-to-end speech-to-speech dialogue)
//! - Notes: Chinese-first S2S dialogue, MoonshotAI, ~13 M h pretrain,
//!   BigVGAN vocoder terminal step.
//!
//! # Conversion posture
//!
//! Every F32 / F16 / BF16 tensor passes through **verbatim** under its
//! upstream safetensors name. BF16 stays GGUF type 30 (`GgmlType::BF16`)
//! — no convert-time widening — mirroring `moshi::convert_streaming`,
//! `qwen3_tts::convert`, `vibevoice::convert`, `voxcpm2::convert`. The
//! runtime widens BF16 → f32 losslessly at load via the single choke
//! point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`
//! (BF16 = top 16 bits of an f32 — `bits << 16` is exact).
//!
//! # License override
//!
//! Default `mit` — the upstream model card declares `license: mit`
//! (fetched 2026-07-25 — CLAUDE.md「ハルシネーション厳禁」). Callers
//! shipping from a mirror that publishes a different SPDX id override
//! at the `license` parameter of [`convert_kimi_audio_file`]; the
//! license *class* is re-derived from the raw SPDX string via
//! [`vokra_core::LicenseClass::from_license_str`].
//!
//! # No ONNX (permanent)
//!
//! Kimi-Audio is distributed as safetensors + a Python pipeline; this
//! converter **never** touches ONNX (FR-LD-05).
//!
//! # Scope
//!
//! Real-weight parity + tensor-name binding are follow-ups gated on the
//! owner license sign-off in `docs/license-audit.md` §3.1; this
//! skeleton lands the qwen3_tts / vibevoice / voxcpm2 pass-through
//! pattern so a future `KimiAudioWeights::from_gguf` can walk the same
//! names.

// The module is a TDD skeleton: [`convert_kimi_audio_file`] and the
// transcribed constants are exercised by this file's `#[cfg(test)]`
// suite but not yet wired into `crate::lib.rs`'s dispatch table
// (`convert_file`) or a `vokra-cli` subcommand. The lib re-export is a
// deliberate follow-up per the task scope (`git add
// kimi_audio.rs + mod.rs` only), so silence the whole-crate dead_code
// warning locally rather than fanning attributes across every item.
#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value written for Kimi-Audio GGUFs. Distinct from
/// every sibling arch tag so the runtime cannot mis-route dispatch.
pub(crate) const ARCH: &str = "kimi_audio";

/// `vokra.model.name` value for the canonical
/// `moonshotai/Kimi-Audio-7B-Instruct` release.
pub(crate) const NAME: &str = "kimi-audio-7b-instruct";

/// `vokra.model.category` value — end-to-end speech-to-speech dialogue.
pub(crate) const CATEGORY: &str = "s2s";

/// `vokra.provenance.upstream_hf` value — the canonical upstream HF id.
pub(crate) const UPSTREAM_HF: &str = "moonshotai/Kimi-Audio-7B-Instruct";

/// Default license SPDX for the upstream release (`license: mit`).
pub(crate) const DEFAULT_LICENSE: &str = "mit";

// Raw metadata keys stamped by this converter. `vokra.model.category`
// and `vokra.provenance.upstream_hf` are Kimi-Audio-scope keys today
// (they do not yet live in `vokra-core::gguf::chunks` as first-class
// constants), so declare them here at module visibility so a future
// crate-level promotion is a single-file diff.
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a Kimi-Audio conversion. Mirrors the observability shape
/// of the qwen3-tts / moshi / vibevoice / voxcpm2 reports (a BF16-aware
/// pass-through counter alongside the classic float / non-float split)
/// with an additional `read` field so the caller can sanity-check
/// `written + skipped_non_float == read`.
#[derive(Debug, Default)]
pub struct KimiAudioReport {
    /// Total number of tensors read from the input safetensors.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive — the safetensors reader
    /// only exposes F32 / F16 / BF16 today; any tensor here signals a
    /// reader-side dtype extension upstream).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Emits GGUF type 30 verbatim; the runtime
    /// widens BF16 → f32 losslessly at load via
    /// `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
    pub bf16_passthrough: usize,
}

/// Reads a safetensors checkpoint from `input`, builds a Vokra GGUF,
/// and writes it to `output`. Returns a [`KimiAudioReport`] describing
/// how the tensors were classified.
///
/// If `license` is `Some`, the provenance stamp uses that SPDX string
/// (overriding the default `mit`); the license class is re-derived
/// from the SPDX via [`vokra_core::LicenseClass::from_license_str`].
///
/// # Errors
///
/// - [`ConvertError::Io`] — reading `input` or writing `output` failed.
/// - [`ConvertError::Parse`] — the safetensors header is malformed.
/// - [`ConvertError::Gguf`] — the GGUF writer refused a tensor.
pub fn convert_kimi_audio_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<KimiAudioReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Self-describing redistribution: the artifact carries its own
    // licence rather than relying on a consumer running Vokra's
    // registry resolver. `mit` is the upstream model-card default;
    // the override lets a mirror ship its own SPDX id and the
    // license class is re-derived from that string
    // (`LicenseClass::from_license_str`).
    let (license_spdx, license_class) = match license {
        Some(spdx) if !spdx.is_empty() => (spdx, LicenseClass::from_license_str(spdx)),
        _ => (DEFAULT_LICENSE, LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        license_class,
        license_spdx,
        Some(NAME),
        Some(UPSTREAM_HF),
    );

    let mut report = KimiAudioReport::default();
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (docs/adr/qwen3-tts-bf16.md, strategy A_passthrough); the
    // runtime widens BF16 → f32 exactly at load via the single choke
    // point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
    // Mirrors `qwen3_tts::convert` / `vibevoice::convert` /
    // `voxcpm2::convert`.
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

    let out_bytes = b.to_bytes()?;
    std::fs::write(output, out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};
    use vokra_core::gguf::GgufFile;

    /// Unique temp path per test (process id + monotonic nanoseconds +
    /// caller-supplied tag). Prevents cross-test collisions when the
    /// suite runs in parallel.
    fn temp_path(tag: &str, ext: &str) -> PathBuf {
        let ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-kimi-audio-{tag}-{}-{ns}.{ext}",
            std::process::id()
        ));
        p
    }

    /// Encode a slice of `f32` values as their BF16 bit patterns
    /// (top 16 bits, `bits << 16` truncation) as little-endian bytes.
    fn bf16_bytes_from_f32(vs: &[f32]) -> Vec<u8> {
        vs.iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// Assembles a safetensors buffer with a single BF16 tensor.
    fn safetensors_one_bf16(name: &str, shape: &[u64], payload: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        assert_eq!(payload.len(), elems as usize * 2, "shape × 2 BF16 bytes");
        let shape_str = shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"BF16","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    /// Assembles a safetensors buffer with one F32 tensor followed by
    /// one F16 tensor (payload concatenated in header-declared order).
    fn safetensors_f32_then_f16(
        f32_name: &str,
        f32_shape: &[u64],
        f32_payload: &[u8],
        f16_name: &str,
        f16_shape: &[u64],
        f16_payload: &[u8],
    ) -> Vec<u8> {
        let f32_elems: u64 = f32_shape.iter().product();
        assert_eq!(f32_payload.len(), f32_elems as usize * 4);
        let f16_elems: u64 = f16_shape.iter().product();
        assert_eq!(f16_payload.len(), f16_elems as usize * 2);
        let f32_shape_s = f32_shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let f16_shape_s = f16_shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let f32_end = f32_payload.len();
        let f16_end = f32_end + f16_payload.len();
        let header = format!(
            r#"{{"{f32_name}":{{"dtype":"F32","shape":[{f32_shape_s}],"data_offsets":[0,{f32_end}]}},"{f16_name}":{{"dtype":"F16","shape":[{f16_shape_s}],"data_offsets":[{f32_end},{f16_end}]}}}}"#
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(f32_payload);
        out.extend_from_slice(f16_payload);
        out
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Non-trivial bit patterns so a silent widen / downcast trips
        // the byte-identity assert (a zeroed BF16 payload would
        // round-trip trivially through a hidden F32/F16 widen too).
        let bf16 = bf16_bytes_from_f32(&[1.0, -2.5, 0.15625, 3.5, -0.5, 42.0]);
        let st_bytes = safetensors_one_bf16("model.embed_tokens.weight", &[2, 3], &bf16);
        let input = temp_path("bf16-in", "safetensors");
        let output = temp_path("bf16-out", "gguf");
        std::fs::write(&input, &st_bytes).expect("write input safetensors");

        let report = convert_kimi_audio_file(&input, &output, None).expect("kimi_audio convert");

        assert_eq!(report.read, 1, "one input tensor was read");
        assert_eq!(report.written, 1, "the BF16 tensor is written");
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);

        let gguf_bytes = std::fs::read(&output).expect("read output GGUF");
        let file = GgufFile::parse(gguf_bytes).expect("parse output GGUF");

        let info = file
            .tensor_info("model.embed_tokens.weight")
            .expect("BF16 tensor present in output GGUF");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "BF16 must stay BF16 (no convert-time widening)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input"
        );

        // Model / provenance / category chunks round-trip.
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
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn f32_and_f16_tensors_pass_through() {
        let f32_values: [f32; 6] = [1.0, -2.0, 3.5, 0.25, -0.5, 42.0];
        let f32_payload: Vec<u8> = f32_values.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Three F16 bit patterns: 1.0 (0x3C00), 2.0 (0x4000), 3.0 (0x4200).
        let f16_payload: Vec<u8> = [0x3C00u16, 0x4000, 0x4200]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let st_bytes = safetensors_f32_then_f16(
            "a.f32.weight",
            &[2, 3],
            &f32_payload,
            "b.f16.weight",
            &[3],
            &f16_payload,
        );

        let input = temp_path("f32f16-in", "safetensors");
        let output = temp_path("f32f16-out", "gguf");
        std::fs::write(&input, &st_bytes).expect("write input safetensors");

        let report = convert_kimi_audio_file(&input, &output, None).expect("kimi_audio convert");

        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2, "both F32 and F16 tensors pass through");
        assert_eq!(report.bf16_passthrough, 0, "no BF16 tensor in this fixture");
        assert_eq!(report.skipped_non_float, 0);

        let gguf_bytes = std::fs::read(&output).expect("read output GGUF");
        let file = GgufFile::parse(gguf_bytes).expect("parse output GGUF");

        let f32_info = file
            .tensor_info("a.f32.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(f32_info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(f32_info), f32_payload.as_slice());

        let f16_info = file
            .tensor_info("b.f16.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16);
        assert_eq!(f16_info.dimensions, vec![3]);
        assert_eq!(file.tensor_bytes(f16_info), f16_payload.as_slice());

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
