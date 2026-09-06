//! **BigVGAN** (`nvidia/bigvgan_*` family): safetensors → GGUF
//! conversion (SoTA plan Phase D2-D5, 2026-07-30).
//!
//! Input: an upstream `nvidia/bigvgan_v2_*` or `nvidia/bigvgan_base_*`
//! release. The upstream repos ship torch-pickle
//! (`bigvgan_generator.pt` alongside `config.json`); callers
//! pre-flatten to safetensors offline via
//! `tools/parity/bigvgan_prepare_checkpoint.py` (the DAC / DFN3 /
//! Parakeet-CTC pattern — upstream release form is torch pickle,
//! converter refuses to touch pickle because that would require
//! embedding a Python interpreter and re-breaking the NFR-DS-02
//! zero-dep posture). Output: a GGUF carrying every float tensor
//! verbatim under its upstream safetensors name, plus the
//! `vokra.provenance.*` / `vokra.model.*` metadata chunks a future
//! native BigVGAN loader will read.
//!
//! # Provenance
//!
//! - **HF paths** (four variants share this single converter):
//!   - `nvidia/bigvgan_v2_22khz_80band_256x` (D2)
//!   - `nvidia/bigvgan_v2_44khz_128band_512x` (D3)
//!   - `nvidia/bigvgan_v2_24khz_100band_256x` (D4)
//!   - `nvidia/bigvgan_base_24khz_100band` (D5, v1 base)
//! - **License (SPDX)**: `mit` — standard MIT (CLAUDE.md 2026-07-22
//!   訂正: `github.com/NVIDIA/BigVGAN/LICENSE` is standard MIT
//!   `Copyright (c) 2024 NVIDIA CORPORATION`, HF `nvidia/bigvgan_v2_*`
//!   / `nvidia/bigvgan_base_*` all carry `license: mit` on the
//!   cardData front-matter; verified 2026-07-30 via HF API — CLAUDE.md
//!   「ハルシネーション厳禁」). Redistribution + commercial use OK.
//! - **Category**: `vocoder` — mel spectrogram → PCM waveform generator.
//!
//! # BigVGAN vs HiFi-GAN
//!
//! The `vokra_ops::bigvgan_generator` op skeleton (SoTA plan Phase 3)
//! already carries the runtime forward primitive: conv_pre + per-stage
//! (transposed_conv1d + MRF of AMPBlock1) + activation_post (Snake or
//! SnakeBeta) + conv_post + tanh / clamp. **Distinct from HiFi-GAN**
//! (leaky_relu vs snake / snakebeta activation, presence of alias-free
//! activation wrappers), so silently sharing an arch tag would
//! mis-route runtime dispatch. See `crates/vokra-convert/src/models/
//! hifigan_vocoder.rs` for the sibling HiFi-GAN converter.
//!
//! # Variant identity
//!
//! All four variants (D2-D5) share the same BigVGAN arch (AMPBlock1 +
//! Snake/SnakeBeta + transposed-conv upsample); they differ only in
//! shape hparams (sample rate, num_mels, upsample_rates, MRF kernels).
//! The [`BigVGanVariant`] discriminator tags the emitted GGUF under
//! `vokra.bigvgan.variant` so the runtime can pick the correct
//! shape-checked config bundle; every hparam is left as a
//! shape-derived value read at `vokra-models` bind time (FR-EX-08
//! authoritative-gate) — this converter is the byte-parallel
//! pass-through side of the SoTA plan Phase D contract.
//!
//! # BF16 pass-through (mirror of wespeaker / ecapa_tdnn / voxcpm2)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm —
//! no convert-time widening. BF16 stays GGUF type 30 (`GgmlType::BF16`);
//! the runtime widens BF16 → f32 losslessly at load via the single
//! choke point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`
//! (BF16 = top 16 bits of an f32 — `bits << 16` is exact).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (`conv_pre.weight`, `ups.{i}.0.weight`,
//! `resblocks.{i*3+j}.convs1.{k}.weight`, `activation_post.alpha` /
//! `activation_post.beta`, `conv_post.weight`, biases; upstream
//! `bigvgan.py` L212-L322 defines the module tree). The strict runtime
//! binder, stored alias-free filter buffers, and real-weight upstream parity
//! are recorded in `docs/handoff/runtime-gap-execution-plan-2026-08-21.md`.
//!
//! # No ONNX (permanent)
//!
//! NVIDIA ships PyTorch checkpoints; this converter **never** touches
//! ONNX (FR-LD-05); the pipeline is re-implemented natively in
//! `crates/vokra-models/src/bigvgan/` when the vocoder lands
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};
use vokra_ops::bigvgan_generator::tensor_manifest_for_variant;

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for BigVGAN GGUFs.
pub const ARCH: &str = "bigvgan";

/// `vokra.model.category` value written for every BigVGAN GGUF.
pub const CATEGORY: &str = "vocoder";

/// Default upstream weight licence (SPDX).
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

/// `vokra.bigvgan.variant` — the variant discriminator key.
const KEY_BIGVGAN_VARIANT: &str = "vokra.bigvgan.variant";
/// Raw string keys not covered by `crate::gguf::chunks`.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

pub use vokra_ops::bigvgan_generator::BigVGanVariant;

fn source_description(variant: BigVGanVariant) -> &'static str {
    match variant {
        BigVGanVariant::V2_22khz80Band256x => {
            "nvidia/bigvgan_v2_22khz_80band_256x (BigVGAN v2 vocoder, MIT)"
        }
        BigVGanVariant::V2_44khz128Band512x => {
            "nvidia/bigvgan_v2_44khz_128band_512x (BigVGAN v2 vocoder, MIT)"
        }
        BigVGanVariant::V2_24khz100Band256x => {
            "nvidia/bigvgan_v2_24khz_100band_256x (BigVGAN v2 vocoder, MIT)"
        }
        BigVGanVariant::BaseV1_24khz100Band => {
            "nvidia/bigvgan_base_24khz_100band (BigVGAN v1 base vocoder, MIT)"
        }
    }
}

/// Outcome of a BigVGAN conversion.
///
/// Mirrors the sibling BF16-pass-through converters' counter shape
/// (`super::wespeaker::WespeakerReport`,
/// `super::ecapa_tdnn::EcapaTdnnReport`) adapted to the
/// file-oriented `convert_bigvgan_file` surface.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BigVGanReport {
    /// Total tensors surfaced by the safetensors reader (before any
    /// dispatch to the pass-through / skipped arm).
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16 at parse time, so a
    /// non-zero here would signal a reader change upstream).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]).
    pub bf16_passthrough: usize,
}

fn validate_descriptor_manifest(
    actual_descriptors: &[(&str, &[u64], GgmlType)],
    variant: BigVGanVariant,
) -> Result<(), ConvertError> {
    use std::collections::BTreeMap;

    let manifest = tensor_manifest_for_variant(variant);
    let expected: BTreeMap<&str, &[u64]> = manifest
        .iter()
        .map(|spec| (spec.name.as_str(), spec.shape.as_slice()))
        .collect();
    let actual: BTreeMap<&str, (&[u64], GgmlType)> = actual_descriptors
        .iter()
        .map(|(name, shape, dtype)| (*name, (*shape, *dtype)))
        .collect();

    let missing: Vec<&str> = expected
        .keys()
        .filter(|name| !actual.contains_key(**name))
        .copied()
        .take(4)
        .collect();
    let extra: Vec<&str> = actual
        .keys()
        .filter(|name| !expected.contains_key(**name))
        .copied()
        .take(4)
        .collect();
    if !missing.is_empty() || !extra.is_empty() || expected.len() != actual.len() {
        return Err(ConvertError::Parse(format!(
            "BigVGAN {variant:?} tensor manifest mismatch (expected {}, found {}); missing={missing:?}, extra={extra:?}",
            expected.len(),
            actual_descriptors.len(),
        )));
    }

    for (name, shape, dtype) in actual_descriptors {
        let expected_shape = expected
            .get(name)
            .expect("manifest cardinality checked above");
        if *shape != *expected_shape {
            return Err(ConvertError::Parse(format!(
                "BigVGAN tensor `{}` shape {:?}, expected {:?}",
                name, shape, expected_shape
            )));
        }
        if !matches!(dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
            return Err(ConvertError::Parse(format!(
                "BigVGAN tensor `{}` has unsupported dtype {:?}; expected F32, F16, or BF16",
                name, dtype
            )));
        }
    }
    Ok(())
}

fn validate_safetensors_manifest(
    st: &SafetensorsFile,
    variant: BigVGanVariant,
) -> Result<(), ConvertError> {
    let descriptors: Vec<(&str, &[u64], GgmlType)> = st
        .tensors()
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor.shape.as_slice(), tensor.dtype))
        .collect();
    validate_descriptor_manifest(&descriptors, variant)?;
    for tensor in st.tensors() {
        if let Some(index) = first_non_finite_index(st, tensor) {
            return Err(ConvertError::Parse(format!(
                "BigVGAN tensor `{}` contains a non-finite value at index {index}",
                tensor.name
            )));
        }
    }
    Ok(())
}

fn first_non_finite_index(
    st: &SafetensorsFile,
    tensor: &vokra_core::safetensors::SafeTensorInfo,
) -> Option<usize> {
    let bytes = st.tensor_bytes(tensor);
    match tensor.dtype {
        GgmlType::F32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_bits(u32::from_le_bytes(chunk.try_into().unwrap())))
            .position(|value| !value.is_finite()),
        GgmlType::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| {
                vokra_core::gguf::quant::f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]))
            })
            .position(|value| !value.is_finite()),
        GgmlType::BF16 => bytes
            .chunks_exact(2)
            .map(|chunk| f32::from_bits(u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16))
            .position(|value| !value.is_finite()),
        _ => None,
    }
}

fn stamp_metadata(b: &mut GgufBuilder, variant: BigVGanVariant, license: Option<&str>) {
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_BIGVGAN_VARIANT, variant.tag());

    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        b,
        class,
        &spdx,
        Some(variant.name()),
        Some(source_description(variant)),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, variant.upstream_hf());
}

fn append_passthrough_tensor(
    b: &mut GgufBuilder,
    st: &SafetensorsFile,
    tensor: &vokra_core::safetensors::SafeTensorInfo,
) -> Result<(), ConvertError> {
    b.add_tensor(
        &tensor.name,
        tensor.dtype,
        tensor.shape.clone(),
        st.tensor_bytes(tensor).to_vec(),
    )
    .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    Ok(())
}

/// Converts a `nvidia/bigvgan_*` safetensors checkpoint at `input`
/// into a Vokra-native GGUF at `output`, tagging the emitted GGUF as
/// the supplied [`BigVGanVariant`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// name; the `vokra.model.*` (arch / name / category) + `vokra.
/// provenance.*` (weight_license / license / model_id / source /
/// upstream_hf) + `vokra.bigvgan.variant` chunks are stamped for the
/// runtime compliance gate (FR-CP-03) and shape-checked config
/// dispatch.
///
/// `license` optionally overrides the stamped weight license (raw
/// SPDX string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// `DEFAULT_LICENSE_SPDX` (`"mit"`, `Permissive`) — the upstream HF
/// releases all ship MIT (verified 2026-07-30 via HF API cardData;
/// GitHub NVIDIA/BigVGAN LICENSE is also standard MIT).
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure; [`ConvertError::Parse`]
/// on a malformed safetensors input.
pub fn convert_bigvgan_file(
    input: &Path,
    output: &Path,
    variant: BigVGanVariant,
    license: Option<&str>,
) -> Result<BigVGanReport, ConvertError> {
    // BigVGAN v2 generators range 112 MB (base 24kHz 100-band, 14M
    // params) to 500+ MB (v2 44kHz 128-band 512x, ~112M params) —
    // still 2 orders of magnitude smaller than the streaming-mandated
    // Moshi 14 GiB tier, so the simple `std::fs::read` posture the
    // sibling non-streaming BF16 pass-through converters use applies.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;
    // Validate the complete descriptor and value contract before constructing
    // metadata or writing any output. A malformed checkpoint must never leave
    // behind a seemingly valid partial GGUF.
    validate_safetensors_manifest(&st, variant)?;

    let mut b = GgufBuilder::new();
    stamp_metadata(&mut b, variant, license);

    let mut report = BigVGanReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted ADR; the runtime
    // widens BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                append_passthrough_tensor(&mut b, &st, t)?;
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
    std::fs::write(output, out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufFile};

    /// Builds a single-BF16-tensor safetensors buffer with a
    /// caller-supplied raw payload.
    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        let expected = elems as usize * 2;
        assert_eq!(
            bf16_bytes.len(),
            expected,
            "test fixture: payload len must match shape × 2 BF16"
        );
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"BF16","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            bf16_bytes.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(bf16_bytes);
        out
    }

    /// Builds an F32 tensor safetensors buffer.
    fn safetensors_one_f32(name: &str, shape: &[u64], f32_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        let expected = elems as usize * 4;
        assert_eq!(f32_bytes.len(), expected);
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"F32","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            f32_bytes.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(f32_bytes);
        out
    }

    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-bigvgan-{kind}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, bytes).expect("write temp file");
        p
    }

    #[test]
    fn shared_descriptor_manifest_covers_all_four_variants() {
        let variants = [
            BigVGanVariant::V2_22khz80Band256x,
            BigVGanVariant::V2_44khz128Band512x,
            BigVGanVariant::V2_24khz100Band256x,
            BigVGanVariant::BaseV1_24khz100Band,
        ];
        for variant in variants {
            let manifest = tensor_manifest_for_variant(variant);
            assert!(!manifest.is_empty());
            assert_eq!(manifest[0].name, "conv_pre.weight");
            assert_eq!(manifest[0].shape[2], 7);
            assert!(manifest.iter().any(|spec| spec.name == "conv_post.weight"));
            let descriptors: Vec<_> = manifest
                .iter()
                .map(|spec| (spec.name.as_str(), spec.shape.as_slice(), GgmlType::F32))
                .collect();
            validate_descriptor_manifest(&descriptors, variant).expect("shared manifest validates");
        }
    }

    fn complete_descriptors(variant: BigVGanVariant) -> Vec<(String, Vec<u64>, GgmlType)> {
        tensor_manifest_for_variant(variant)
            .into_iter()
            .map(|spec| (spec.name, spec.shape, GgmlType::F32))
            .collect()
    }

    fn descriptor_refs(
        descriptors: &[(String, Vec<u64>, GgmlType)],
    ) -> Vec<(&str, &[u64], GgmlType)> {
        descriptors
            .iter()
            .map(|(name, shape, dtype)| (name.as_str(), shape.as_slice(), *dtype))
            .collect()
    }

    #[test]
    fn descriptor_manifest_rejects_missing_tensor() {
        let variant = BigVGanVariant::V2_24khz100Band256x;
        let mut descriptors = complete_descriptors(variant);
        descriptors.pop();
        let error = validate_descriptor_manifest(&descriptor_refs(&descriptors), variant)
            .expect_err("missing tensor must fail");
        assert!(error.to_string().contains("manifest mismatch"));
        assert!(error.to_string().contains("conv_post"));
    }

    #[test]
    fn descriptor_manifest_rejects_extra_tensor() {
        let variant = BigVGanVariant::BaseV1_24khz100Band;
        let mut descriptors = complete_descriptors(variant);
        descriptors.push(("rogue.weight".to_owned(), vec![1], GgmlType::F32));
        let error = validate_descriptor_manifest(&descriptor_refs(&descriptors), variant)
            .expect_err("extra tensor must fail");
        assert!(error.to_string().contains("rogue.weight"));
    }

    #[test]
    fn descriptor_manifest_rejects_wrong_shape() {
        let variant = BigVGanVariant::V2_22khz80Band256x;
        let mut descriptors = complete_descriptors(variant);
        descriptors[0].1[0] += 1;
        let error = validate_descriptor_manifest(&descriptor_refs(&descriptors), variant)
            .expect_err("wrong shape must fail");
        assert!(error.to_string().contains("conv_pre.weight"));
        assert!(error.to_string().contains("shape"));
    }

    #[test]
    fn descriptor_manifest_rejects_unsupported_dtype() {
        let variant = BigVGanVariant::V2_44khz128Band512x;
        let mut descriptors = complete_descriptors(variant);
        descriptors[0].2 = GgmlType::I8;
        let error = validate_descriptor_manifest(&descriptor_refs(&descriptors), variant)
            .expect_err("unsupported dtype must fail");
        assert!(error.to_string().contains("unsupported dtype"));
    }

    #[test]
    fn metadata_stamping_covers_all_variants_and_license_override() {
        let variants = [
            BigVGanVariant::V2_22khz80Band256x,
            BigVGanVariant::V2_44khz128Band512x,
            BigVGanVariant::V2_24khz100Band256x,
            BigVGanVariant::BaseV1_24khz100Band,
        ];
        for variant in variants {
            let mut builder = GgufBuilder::new();
            stamp_metadata(&mut builder, variant, Some("apache-2.0"));
            let file = GgufFile::parse(builder.to_bytes().expect("metadata-only GGUF"))
                .expect("metadata-only GGUF parses");
            assert_eq!(
                file.get(KEY_BIGVGAN_VARIANT).and_then(|v| v.as_str()),
                Some(variant.tag())
            );
            assert_eq!(
                file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
                Some(variant.name())
            );
            assert_eq!(
                file.get(KEY_PROVENANCE_UPSTREAM_HF)
                    .and_then(|v| v.as_str()),
                Some(variant.upstream_hf())
            );
            assert_eq!(
                file.get(chunks::KEY_PROVENANCE_LICENSE)
                    .and_then(|v| v.as_str()),
                Some("apache-2.0")
            );
        }
    }

    #[test]
    fn bf16_passthrough_helper_preserves_wire_bytes() {
        let values = [0x3f80_u16, 0xc020_u16];
        let payload: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input = safetensors_one_bf16("conv_pre.bias", &[2], &payload);
        let st = SafetensorsFile::parse(input).expect("BF16 descriptor parses");
        let mut builder = GgufBuilder::new();
        append_passthrough_tensor(&mut builder, &st, &st.tensors()[0]).expect("append");
        let file = GgufFile::parse(builder.to_bytes().expect("GGUF")).expect("GGUF parses");
        let info = file.tensor_info("conv_pre.bias").expect("tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), payload.as_slice());
    }

    #[test]
    fn finite_value_scan_rejects_nan_without_widening_payload() {
        let payload = f32::NAN.to_le_bytes();
        let input = safetensors_one_f32("conv_pre.bias", &[1], &payload);
        let st = SafetensorsFile::parse(input).expect("F32 descriptor parses");
        assert_eq!(first_non_finite_index(&st, &st.tensors()[0]), Some(0));
    }

    #[test]
    fn conversion_rejects_partial_checkpoint_before_writing_output() {
        let input = write_temp(
            "partial-in",
            &safetensors_one_f32("conv_pre.bias", &[1], &1.0f32.to_le_bytes()),
        );
        let output = write_temp("partial-out", b"sentinel");
        let error =
            convert_bigvgan_file(&input, &output, BigVGanVariant::V2_24khz100Band256x, None)
                .expect_err("partial checkpoint must be rejected");
        assert!(error.to_string().contains("manifest mismatch"));
        assert_eq!(
            std::fs::read(&output).expect("sentinel remains"),
            b"sentinel"
        );
        std::fs::remove_file(input).ok();
        std::fs::remove_file(output).ok();
    }
}
