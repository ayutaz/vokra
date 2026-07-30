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
//! `bigvgan.py` L212-L322 defines the module tree). Real-weight parity
//! vs the upstream `nvidia/BigVGAN` reference is deferred to owner
//! (`docs/license-audit.md` §3.1 sign-off queue).
//!
//! # No ONNX (permanent)
//!
//! NVIDIA ships PyTorch checkpoints; this converter **never** touches
//! ONNX (FR-LD-05); the pipeline is re-implemented natively in
//! `crates/vokra-models/src/bigvgan/` when the vocoder lands
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

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

/// Which BigVGAN release this GGUF represents. The four variants share
/// the AMPBlock1 + Snake/SnakeBeta topology byte-for-byte — only sample
/// rate + num_mels + upsample_rates differ, so this tag is what the
/// runtime checks to pick the shape-checked config bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BigVGanVariant {
    /// `nvidia/bigvgan_v2_22khz_80band_256x` (D2): 22 050 Hz output,
    /// 80-band mel input, 256× total upsample.
    V2_22khz80Band256x,
    /// `nvidia/bigvgan_v2_44khz_128band_512x` (D3): 44 100 Hz output,
    /// 128-band mel input, 512× total upsample.
    V2_44khz128Band512x,
    /// `nvidia/bigvgan_v2_24khz_100band_256x` (D4): 24 000 Hz output,
    /// 100-band mel input, 256× total upsample.
    V2_24khz100Band256x,
    /// `nvidia/bigvgan_base_24khz_100band` (D5): v1 base 24 000 Hz
    /// output, 100-band mel input. Distinct from D4 because the v1
    /// base predates SnakeBeta + the v2 anti-aliased activation
    /// wrapper (upstream `bigvgan.py:206-322` picks Snake vs
    /// SnakeBeta by config).
    BaseV1_24khz100Band,
}

impl BigVGanVariant {
    /// Wire tag written into `vokra.bigvgan.variant`.
    pub fn tag(self) -> &'static str {
        match self {
            Self::V2_22khz80Band256x => "v2_22khz_80band_256x",
            Self::V2_44khz128Band512x => "v2_44khz_128band_512x",
            Self::V2_24khz100Band256x => "v2_24khz_100band_256x",
            Self::BaseV1_24khz100Band => "base_v1_24khz_100band",
        }
    }

    /// `vokra.model.name` value for this variant.
    pub fn name(self) -> &'static str {
        match self {
            Self::V2_22khz80Band256x => "bigvgan-v2-22khz-80band-256x",
            Self::V2_44khz128Band512x => "bigvgan-v2-44khz-128band-512x",
            Self::V2_24khz100Band256x => "bigvgan-v2-24khz-100band-256x",
            Self::BaseV1_24khz100Band => "bigvgan-base-24khz-100band",
        }
    }

    /// Upstream HF path for this variant (the primary redistribution
    /// source used by the model-card generator).
    pub fn upstream_hf(self) -> &'static str {
        match self {
            Self::V2_22khz80Band256x => "nvidia/bigvgan_v2_22khz_80band_256x",
            Self::V2_44khz128Band512x => "nvidia/bigvgan_v2_44khz_128band_512x",
            Self::V2_24khz100Band256x => "nvidia/bigvgan_v2_24khz_100band_256x",
            Self::BaseV1_24khz100Band => "nvidia/bigvgan_base_24khz_100band",
        }
    }

    /// Human-readable description for the provenance `source` field.
    fn source_description(self) -> &'static str {
        match self {
            Self::V2_22khz80Band256x => {
                "nvidia/bigvgan_v2_22khz_80band_256x (BigVGAN v2 vocoder, MIT)"
            }
            Self::V2_44khz128Band512x => {
                "nvidia/bigvgan_v2_44khz_128band_512x (BigVGAN v2 vocoder, MIT)"
            }
            Self::V2_24khz100Band256x => {
                "nvidia/bigvgan_v2_24khz_100band_256x (BigVGAN v2 vocoder, MIT)"
            }
            Self::BaseV1_24khz100Band => {
                "nvidia/bigvgan_base_24khz_100band (BigVGAN v1 base vocoder, MIT)"
            }
        }
    }
}

/// Outcome of a BigVGAN conversion.
///
/// Mirrors the sibling BF16-pass-through converters' counter shape
/// ([`super::wespeaker::WespeakerReport`],
/// [`super::ecapa_tdnn::EcapaTdnnReport`]) adapted to the
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
/// [`DEFAULT_LICENSE_SPDX`] (`"mit"`, `Permissive`) — the upstream HF
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

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_BIGVGAN_VARIANT, variant.tag());

    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(variant.name()),
        Some(variant.source_description()),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, variant.upstream_hf());

    let mut report = BigVGanReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted ADR; the runtime
    // widens BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
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
    fn bf16_tensor_passes_through_verbatim_v2_24khz() {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12);

        // Mirror an actual upstream BigVGAN tensor name from
        // bigvgan.py L235-245 (`ups.{i}.0.weight` is the ith
        // ConvTranspose1d in the upsample stack).
        let input_bytes = safetensors_one_bf16("ups.0.0.weight", &[2, 3], &bf16);
        let input_path = write_temp("v2-24k-in", &input_bytes);
        let output_path = write_temp("v2-24k-out", &[]);

        let report = convert_bigvgan_file(
            &input_path,
            &output_path,
            BigVGanVariant::V2_24khz100Band256x,
            None,
        )
        .expect("convert_bigvgan_file must accept a well-formed BF16 checkpoint");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("ups.0.0.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(info.dtype, GgmlType::BF16, "no convert-time widening");
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical"
        );

        // Variant discriminator was written.
        assert_eq!(
            file.get(KEY_BIGVGAN_VARIANT).and_then(|v| v.as_str()),
            Some("v2_24khz_100band_256x"),
        );
        // Model name is variant-specific (mirror T3 pattern).
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("bigvgan-v2-24khz-100band-256x"),
        );
        // Upstream HF is variant-specific.
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("nvidia/bigvgan_v2_24khz_100band_256x"),
        );
        // License defaults to mit.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some("vocoder"),
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn variant_v2_44khz_128band_lands_distinct_stamps() {
        // Distinct variant surfaces a distinct name + upstream_hf + tag —
        // guards against a regression that would map all four variants to
        // the same wire tag.
        let values: [f32; 2] = [1.0, -2.5];
        let f32_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one_f32("conv_pre.weight", &[1, 2], &f32_bytes);
        let input_path = write_temp("v2-44k-in", &input_bytes);
        let output_path = write_temp("v2-44k-out", &[]);

        convert_bigvgan_file(
            &input_path,
            &output_path,
            BigVGanVariant::V2_44khz128Band512x,
            None,
        )
        .expect("convert must succeed");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(KEY_BIGVGAN_VARIANT).and_then(|v| v.as_str()),
            Some("v2_44khz_128band_512x"),
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("bigvgan-v2-44khz-128band-512x"),
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("nvidia/bigvgan_v2_44khz_128band_512x"),
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn variant_v2_22khz_80band_lands_distinct_stamps() {
        // Non-zero distinctive values (avoid `3.14` which triggers
        // `clippy::approx_constant` against `f32::consts::PI`).
        let f32_bytes: Vec<u8> = [7.25_f32, -1.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_f32("conv_post.weight", &[1, 2], &f32_bytes);
        let input_path = write_temp("v2-22k-in", &input_bytes);
        let output_path = write_temp("v2-22k-out", &[]);

        convert_bigvgan_file(
            &input_path,
            &output_path,
            BigVGanVariant::V2_22khz80Band256x,
            None,
        )
        .expect("convert must succeed");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(KEY_BIGVGAN_VARIANT).and_then(|v| v.as_str()),
            Some("v2_22khz_80band_256x"),
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("bigvgan-v2-22khz-80band-256x"),
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("nvidia/bigvgan_v2_22khz_80band_256x"),
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn variant_base_v1_24khz_lands_distinct_stamps() {
        // BaseV1_24khz100Band is functionally distinct from
        // V2_24khz100Band256x because base v1 predates SnakeBeta + the
        // v2 anti-aliased activation wrapper. Tag / name / upstream_hf
        // must all be distinct.
        let f32_bytes: Vec<u8> = [0.5_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one_f32("conv_pre.bias", &[1], &f32_bytes);
        let input_path = write_temp("base-v1-in", &input_bytes);
        let output_path = write_temp("base-v1-out", &[]);

        convert_bigvgan_file(
            &input_path,
            &output_path,
            BigVGanVariant::BaseV1_24khz100Band,
            None,
        )
        .expect("convert must succeed");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(KEY_BIGVGAN_VARIANT).and_then(|v| v.as_str()),
            Some("base_v1_24khz_100band"),
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("bigvgan-base-24khz-100band"),
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("nvidia/bigvgan_base_24khz_100band"),
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn license_override_flows_through() {
        let f32_bytes: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_f32("conv_pre.weight", &[1, 2], &f32_bytes);
        let input_path = write_temp("license-in", &input_bytes);
        let output_path = write_temp("license-out", &[]);

        convert_bigvgan_file(
            &input_path,
            &output_path,
            BigVGanVariant::V2_22khz80Band256x,
            Some("apache-2.0"),
        )
        .expect("license override must succeed");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
            "license override must be honored"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "apache-2.0 is Permissive class"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }
}
