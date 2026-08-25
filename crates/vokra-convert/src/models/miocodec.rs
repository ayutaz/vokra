//! Exact MioCodec 25 Hz / 44.1 kHz v2 converter.
//!
//! The pinned upstream release contains one 528,105,436-byte F32 safetensors
//! checkpoint with 350 tensors. This converter accepts exactly that audited
//! name/shape/dtype topology and writes the tensor payloads verbatim. Missing,
//! extra, renamed, reshaped, or non-F32 tensors are errors; an arbitrary float
//! checkpoint can no longer acquire the `miocodec` arch tag.
//!
//! The topology is independently transcribed from MioCodec source revision
//! [`SOURCE_REVISION`] and `config.yaml` at Hugging Face revision
//! [`UPSTREAM_REVISION`]. The native runtime consumes the stamped topology and
//! the same full manifest. The historical public Vokra GGUF predates these
//! keys, so runtime compatibility for it is limited to its exact 350-tensor
//! header and fixed public revision rather than treating absent metadata as a
//! general default.
//!
//! MioCodec code and weights are MIT. The model ships safetensors directly;
//! neither conversion nor runtime needs PyTorch pickle, ONNX, protobuf, or an
//! external codec library. The core route is neutral token decoding. Voice-
//! conversion orchestration is deliberately not exposed by this converter.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for MioCodec GGUFs — kept intentionally distinct
/// from every sibling codec so the runtime dispatch cannot silently
/// route a MioCodec artifact through a DAC / Mimi / RVQ / FSQ /
/// SoundStream / focal-modulation decoder.
pub const ARCH: &str = "miocodec";

/// `vokra.model.name` value for the canonical
/// `Aratako/MioCodec-25Hz-44.1kHz-v2` release. Matches the publish
/// repo slug spelling (`vokra/miocodec-25hz-44khz-v2` — HF repo naming
/// = dashes only, lowercase, dots stripped from `44.1` → `44khz`).
pub const NAME: &str = "miocodec-25hz-44khz-v2";

/// `vokra.model.category` key — codec bucket for the artifact.
///
/// Kept as a local constant rather than a `vokra_core::gguf::chunks::*`
/// re-export because it is not yet part of the shared
/// `vokra_core::gguf::chunks` surface (mirrors the bicodec / neucodec /
/// focalcodec convention).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Category string written into [`KEY_MODEL_CATEGORY`].
pub const MODEL_CATEGORY: &str = "codec";

/// `vokra.provenance.upstream_hf` key — HF repo path of the upstream
/// weight.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Value written under [`KEY_PROVENANCE_UPSTREAM_HF`]. Preserves the
/// upstream capitalization + dot in `44.1kHz` — the HF repo slug is
/// case-sensitive and the primary source for the model-card generator.
pub const UPSTREAM_HF: &str = "Aratako/MioCodec-25Hz-44.1kHz-v2";

/// Immutable upstream Hugging Face revision containing the audited checkpoint.
pub const UPSTREAM_REVISION: &str = "67faba34153fe74e6665991c432a7327e23c5c1c";
/// Immutable source revision used to transcribe the native forward.
pub const SOURCE_REVISION: &str = "77473544375d57e96cbdfd5d7d257e8f280fa8e3";
/// SHA-256 of the pinned 528,105,436-byte `model.safetensors`.
pub const MODEL_SHA256: &str = "8e319ef2231bad184f17cb73fd5a21b685c25c6c1622ef33ed9271187e81cd4a";
/// SHA-256 of the pinned 2,705-byte `config.yaml`.
pub const CONFIG_SHA256: &str = "bfabffffaaa5709b8dc69585111ee3d53c1b0609c23d293cd1b4903eafa5bec1";

/// Default weight-license SPDX. Verified 2026-08-04 via HF cardData
/// API primary source (`api/models/Aratako/MioCodec-25Hz-44.1kHz-v2`
/// → `license: mit`). A non-MIT override is rejected because it would relabel
/// the pinned checkpoint.
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

/// Exact number of tensors in the pinned upstream checkpoint and public GGUF.
pub const TENSOR_COUNT: usize = 350;
pub const SAMPLE_RATE: u32 = 44_100;
pub const N_FFT: u32 = 392;
pub const HOP_LENGTH: u32 = 98;
pub const CONTENT_DIM: u32 = 768;
pub const GLOBAL_DIM: u32 = 128;
pub const WAVE_DIM: u32 = 512;
pub const CODE_DIM: u32 = 5;
pub const VOCAB_SIZE: u32 = 12_800;

/// Advisory source note stamped alongside the license into the
/// `vokra.provenance.source` chunk.
const PROVENANCE_SOURCE_NOTE: &str = "Aratako/MioCodec-25Hz-44.1kHz-v2 (JA-focused 25 Hz / 44.1 kHz \
     multilingual speech codec, MIT end-to-end; base \
     Aratako/MioCodec-25Hz-24kHz)";

pub const KEY_UPSTREAM_REVISION: &str = "vokra.miocodec.upstream_revision";
pub const KEY_SOURCE_REVISION: &str = "vokra.miocodec.source_revision";
pub const KEY_MODEL_SHA256: &str = "vokra.miocodec.model_sha256";
pub const KEY_CONFIG_SHA256: &str = "vokra.miocodec.config_sha256";
pub const KEY_SAMPLE_RATE: &str = "vokra.miocodec.sample_rate";
pub const KEY_N_FFT: &str = "vokra.miocodec.n_fft";
pub const KEY_HOP_LENGTH: &str = "vokra.miocodec.hop_length";
pub const KEY_CONTENT_DIM: &str = "vokra.miocodec.content_dim";
pub const KEY_GLOBAL_DIM: &str = "vokra.miocodec.global_dim";
pub const KEY_WAVE_DIM: &str = "vokra.miocodec.wave_dim";
pub const KEY_CODE_DIM: &str = "vokra.miocodec.code_dim";
pub const KEY_VOCAB_SIZE: &str = "vokra.miocodec.vocab_size";
pub const KEY_FSQ_LEVELS: &str = "vokra.miocodec.fsq_levels";
pub const KEY_WAVE_UPSAMPLE_FACTORS: &str = "vokra.miocodec.wave_upsample_factors";
pub const KEY_WAVE_UPSAMPLE_KERNELS: &str = "vokra.miocodec.wave_upsample_kernels";
pub const KEY_ISTFT_PADDING: &str = "vokra.miocodec.istft_padding";
pub const KEY_DECODE_ONLY: &str = "vokra.miocodec.decode_only";

/// Outcome of a MioCodec conversion.
///
/// The four fields preserve the converter's existing public report shape.
/// Exact v2 conversion always returns `350 / 350 / 0 / 0`; any other manifest
/// fails before the output file is written.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MioCodecReport {
    /// Total upstream tensors observed in the safetensors input.
    pub read: usize,
    /// F32 tensors written verbatim.
    pub written: usize,
    /// Retained for API compatibility; strict conversion leaves it zero.
    pub skipped_non_float: usize,
    /// Retained for API compatibility; the pinned release is F32 only.
    pub bf16_passthrough: usize,
}

/// Convert an `Aratako/MioCodec-25Hz-44.1kHz-v2` safetensors
/// checkpoint into a Vokra GGUF.
///
/// `input` is the pinned upstream `model.safetensors` path; the emitted GGUF
/// is written to `output`. `license` may be omitted or spell MIT
/// case-insensitively. Every other value is rejected.
///
/// # Errors
///
/// - I/O reading `input` or writing `output` propagates as
///   [`ConvertError::Io`].
/// - Safetensors parse failure propagates as [`ConvertError::Parse`].
/// - GGUF serialization failure propagates as the `From<GgufError>`
///   impl on `ConvertError`.
pub fn convert_miocodec_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<MioCodecReport, ConvertError> {
    let license = require_official_license(license)?;
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;
    validate_input_manifest(&st)?;

    let mut b = GgufBuilder::new();
    stamp_contract(&mut b, license);

    for t in st.tensors() {
        b.add_tensor(
            &t.name,
            t.dtype,
            t.shape.clone(),
            st.tensor_bytes(t).to_vec(),
        )?;
    }

    let out_bytes = b.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(MioCodecReport {
        read: TENSOR_COUNT,
        written: TENSOR_COUNT,
        skipped_non_float: 0,
        bf16_passthrough: 0,
    })
}

fn require_official_license(license: Option<&str>) -> Result<&'static str, ConvertError> {
    let requested = license.unwrap_or(DEFAULT_LICENSE_SPDX).trim();
    if !requested.eq_ignore_ascii_case(DEFAULT_LICENSE_SPDX) {
        return Err(parse_error(format!(
            "license override `{requested}` conflicts with the pinned official MIT checkpoint"
        )));
    }
    Ok(DEFAULT_LICENSE_SPDX)
}

fn stamp_contract(builder: &mut GgufBuilder, spdx: &str) {
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string(KEY_MODEL_CATEGORY, MODEL_CATEGORY);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    builder.add_string(KEY_UPSTREAM_REVISION, UPSTREAM_REVISION);
    builder.add_string(KEY_SOURCE_REVISION, SOURCE_REVISION);
    builder.add_string(KEY_MODEL_SHA256, MODEL_SHA256);
    builder.add_string(KEY_CONFIG_SHA256, CONFIG_SHA256);
    builder.add_u32(KEY_SAMPLE_RATE, SAMPLE_RATE);
    builder.add_u32(KEY_N_FFT, N_FFT);
    builder.add_u32(KEY_HOP_LENGTH, HOP_LENGTH);
    builder.add_u32(KEY_CONTENT_DIM, CONTENT_DIM);
    builder.add_u32(KEY_GLOBAL_DIM, GLOBAL_DIM);
    builder.add_u32(KEY_WAVE_DIM, WAVE_DIM);
    builder.add_u32(KEY_CODE_DIM, CODE_DIM);
    builder.add_u32(KEY_VOCAB_SIZE, VOCAB_SIZE);
    add_u32_array(builder, KEY_FSQ_LEVELS, &[8, 8, 8, 5, 5]);
    add_u32_array(builder, KEY_WAVE_UPSAMPLE_FACTORS, &[3, 3]);
    add_u32_array(builder, KEY_WAVE_UPSAMPLE_KERNELS, &[9, 9]);
    builder.add_string(KEY_ISTFT_PADDING, "same");
    builder.add_bool(KEY_DECODE_ONLY, true);

    vokra_core::stamp_provenance(
        builder,
        LicenseClass::from_license_str(spdx),
        spdx,
        Some(NAME),
        Some(PROVENANCE_SOURCE_NOTE),
    );
}

fn add_u32_array(builder: &mut GgufBuilder, key: &str, values: &[u32]) {
    builder.add_metadata(
        key,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: values.iter().copied().map(GgufMetadataValue::U32).collect(),
        }),
    );
}

fn validate_input_manifest(st: &SafetensorsFile) -> Result<(), ConvertError> {
    let expected = expected_manifest();
    if expected.len() != TENSOR_COUNT || st.tensors().len() != TENSOR_COUNT {
        return Err(parse_error(format!(
            "checkpoint has {} tensors, expected exactly {TENSOR_COUNT} at revision {UPSTREAM_REVISION}",
            st.tensors().len()
        )));
    }
    let mut seen = BTreeSet::new();
    for tensor in st.tensors() {
        let shape = expected
            .get(&tensor.name)
            .ok_or_else(|| parse_error(format!("unexpected tensor `{}`", tensor.name)))?;
        if &tensor.shape != shape {
            return Err(parse_error(format!(
                "tensor `{}` has shape {:?}, expected {shape:?}",
                tensor.name, tensor.shape
            )));
        }
        if tensor.dtype != GgmlType::F32 {
            return Err(parse_error(format!(
                "tensor `{}` is {:?}, expected F32 at revision {UPSTREAM_REVISION}",
                tensor.name, tensor.dtype
            )));
        }
        seen.insert(tensor.name.as_str());
    }
    if let Some(missing) = expected.keys().find(|name| !seen.contains(name.as_str())) {
        return Err(parse_error(format!(
            "checkpoint is missing tensor `{missing}`"
        )));
    }
    Ok(())
}

fn parse_error(message: impl Into<String>) -> ConvertError {
    ConvertError::Parse(format!("miocodec: {}", message.into()))
}

/// Exact 350-tensor manifest generated from the fixed v2 topology.
fn expected_manifest() -> BTreeMap<String, Vec<u64>> {
    let mut out = BTreeMap::new();

    insert(&mut out, "conv_downsample.bias", &[768]);
    insert(&mut out, "conv_downsample.weight", &[768, 768, 2]);
    add_global_encoder(&mut out);
    insert(&mut out, "istft_head.out.bias", &[394]);
    insert(&mut out, "istft_head.out.weight", &[394, 512]);
    add_affine_transformer(&mut out, "local_encoder", 6, 768, 2048, false);
    insert(&mut out, "local_quantizer.proj_in.bias", &[5]);
    insert(&mut out, "local_quantizer.proj_in.weight", &[5, 768]);
    insert(&mut out, "local_quantizer.proj_out.bias", &[768]);
    insert(&mut out, "local_quantizer.proj_out.weight", &[768, 5]);
    insert(&mut out, "wave_conv_upsample.bias", &[512]);
    insert(&mut out, "wave_conv_upsample.weight", &[512, 512, 2]);
    add_adaln_transformer(&mut out);
    add_resnet_stack(&mut out, "wave_post_net", 2, 512);
    add_affine_transformer(&mut out, "wave_prenet", 6, 768, 2048, true);
    add_resnet_stack(&mut out, "wave_prior_net", 2, 512);
    add_wave_upsampler(&mut out);

    debug_assert_eq!(out.len(), TENSOR_COUNT);
    out
}

fn add_global_encoder(out: &mut BTreeMap<String, Vec<u64>>) {
    for layer in 0..4 {
        let prefix = format!("global_encoder.backbone.convnext.{layer}");
        insert(out, format!("{prefix}.dwconv.bias"), &[384]);
        insert(out, format!("{prefix}.dwconv.weight"), &[384, 1, 7]);
        insert(out, format!("{prefix}.gamma"), &[384]);
        insert(out, format!("{prefix}.norm.bias"), &[384]);
        insert(out, format!("{prefix}.norm.weight"), &[384]);
        insert(out, format!("{prefix}.pwconv1.bias"), &[1152]);
        insert(out, format!("{prefix}.pwconv1.weight"), &[1152, 384]);
        insert(out, format!("{prefix}.pwconv2.bias"), &[384]);
        insert(out, format!("{prefix}.pwconv2.weight"), &[384, 1152]);
    }
    insert(out, "global_encoder.backbone.embed.bias", &[384]);
    insert(out, "global_encoder.backbone.embed.weight", &[384, 768, 7]);
    insert(out, "global_encoder.backbone.final_layer_norm.bias", &[384]);
    insert(
        out,
        "global_encoder.backbone.final_layer_norm.weight",
        &[384],
    );
    insert(out, "global_encoder.backbone.norm.bias", &[384]);
    insert(out, "global_encoder.backbone.norm.weight", &[384]);
    insert(out, "global_encoder.pooling.attn.0.bias", &[128]);
    insert(out, "global_encoder.pooling.attn.0.weight", &[128, 384, 1]);
    insert(out, "global_encoder.pooling.attn.2.bias", &[384]);
    insert(out, "global_encoder.pooling.attn.2.weight", &[384, 128, 1]);
    insert(out, "global_encoder.pooling.norm.bias", &[128]);
    insert(out, "global_encoder.pooling.norm.weight", &[128]);
    insert(out, "global_encoder.pooling.proj.bias", &[128]);
    insert(out, "global_encoder.pooling.proj.weight", &[128, 768]);
}

fn add_affine_transformer(
    out: &mut BTreeMap<String, Vec<u64>>,
    root: &str,
    layers: usize,
    dim: u64,
    hidden: u64,
    output_projection: bool,
) {
    for layer in 0..layers {
        let prefix = format!("{root}.layers.{layer}");
        add_attention(out, &prefix, dim);
        insert(out, format!("{prefix}.attention_norm.bias"), &[dim]);
        insert(out, format!("{prefix}.attention_norm.weight"), &[dim]);
        add_feed_forward(out, &prefix, dim, hidden);
        insert(out, format!("{prefix}.ffn_norm.bias"), &[dim]);
        insert(out, format!("{prefix}.ffn_norm.weight"), &[dim]);
    }
    insert(out, format!("{root}.norm.bias"), &[dim]);
    insert(out, format!("{root}.norm.weight"), &[dim]);
    if output_projection {
        insert(out, format!("{root}.output_proj.bias"), &[512]);
        insert(out, format!("{root}.output_proj.weight"), &[512, dim]);
    }
}

fn add_adaln_transformer(out: &mut BTreeMap<String, Vec<u64>>) {
    for layer in 0..8 {
        let prefix = format!("wave_decoder.layers.{layer}");
        add_attention(out, &prefix, 512);
        insert(
            out,
            format!("{prefix}.attention_norm.condition_proj.1.bias"),
            &[1536],
        );
        insert(
            out,
            format!("{prefix}.attention_norm.condition_proj.1.weight"),
            &[1536, 128],
        );
        add_feed_forward(out, &prefix, 512, 1536);
        insert(
            out,
            format!("{prefix}.ffn_norm.condition_proj.1.bias"),
            &[1536],
        );
        insert(
            out,
            format!("{prefix}.ffn_norm.condition_proj.1.weight"),
            &[1536, 128],
        );
    }
    insert(out, "wave_decoder.norm.condition_proj.1.bias", &[1024]);
    insert(
        out,
        "wave_decoder.norm.condition_proj.1.weight",
        &[1024, 128],
    );
}

fn add_attention(out: &mut BTreeMap<String, Vec<u64>>, prefix: &str, dim: u64) {
    for projection in ["wk", "wo", "wq", "wv"] {
        insert(
            out,
            format!("{prefix}.attention.{projection}.weight"),
            &[dim, dim],
        );
    }
}

fn add_feed_forward(out: &mut BTreeMap<String, Vec<u64>>, prefix: &str, dim: u64, hidden: u64) {
    insert(
        out,
        format!("{prefix}.feed_forward.w1.weight"),
        &[hidden, dim],
    );
    insert(
        out,
        format!("{prefix}.feed_forward.w2.weight"),
        &[dim, hidden],
    );
    insert(
        out,
        format!("{prefix}.feed_forward.w3.weight"),
        &[hidden, dim],
    );
}

fn add_resnet_stack(
    out: &mut BTreeMap<String, Vec<u64>>,
    root: &str,
    blocks: usize,
    channels: u64,
) {
    for block in 0..blocks {
        let prefix = format!("{root}.blocks.{block}");
        for conv in ["conv1", "conv2"] {
            insert(out, format!("{prefix}.{conv}.bias"), &[channels]);
            insert(
                out,
                format!("{prefix}.{conv}.weight"),
                &[channels, channels, 3],
            );
        }
        for norm in ["norm1", "norm2"] {
            insert(out, format!("{prefix}.{norm}.bias"), &[channels]);
            insert(out, format!("{prefix}.{norm}.weight"), &[channels]);
        }
    }
}

fn add_wave_upsampler(out: &mut BTreeMap<String, Vec<u64>>) {
    insert(out, "wave_upsampler.out_proj.bias", &[512]);
    insert(out, "wave_upsampler.out_proj.weight", &[512, 128]);
    insert(out, "wave_upsampler.out_snake.alpha", &[512]);
    insert(out, "wave_upsampler.out_snake.beta", &[512]);

    for (stage, channels, input) in [(0, 256, 512), (1, 128, 256)] {
        let resnet = format!("wave_upsampler.resnet_blocks.{stage}");
        for conv in ["conv1", "conv2"] {
            insert(out, format!("{resnet}.{conv}.bias"), &[channels]);
            insert(
                out,
                format!("{resnet}.{conv}.weight"),
                &[channels, channels, 3],
            );
        }
        for norm in ["norm1", "norm2"] {
            insert(out, format!("{resnet}.{norm}.bias"), &[channels]);
            insert(out, format!("{resnet}.{norm}.weight"), &[channels]);
        }
        insert(
            out,
            format!("wave_upsampler.snake_activations.{stage}.alpha"),
            &[channels],
        );
        insert(
            out,
            format!("wave_upsampler.snake_activations.{stage}.beta"),
            &[channels],
        );
        let upsample = format!("wave_upsampler.upsample_layers.{stage}");
        insert(out, format!("{upsample}.bias"), &[channels]);
        insert(
            out,
            format!("{upsample}.parametrizations.weight.original0"),
            &[input, 1, 1],
        );
        insert(
            out,
            format!("{upsample}.parametrizations.weight.original1"),
            &[input, channels, 9],
        );
    }
}

fn insert(out: &mut BTreeMap<String, Vec<u64>>, name: impl Into<String>, shape: &[u64]) {
    let name = name.into();
    assert!(out.insert(name.clone(), shape.to_vec()).is_none(), "{name}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Builds a one-tensor safetensors buffer for fail-closed manifest tests.
    fn safetensors_one_f32(name: &str, shape: &[u64], f32_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        assert_eq!(
            f32_bytes.len(),
            elems as usize * 4,
            "test fixture: payload len must match shape × 4 F32"
        );
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

    #[test]
    fn exact_manifest_matches_public_header_contract() {
        let manifest = expected_manifest();
        assert_eq!(manifest.len(), TENSOR_COUNT);
        assert_eq!(manifest["local_quantizer.proj_out.weight"], vec![768, 5]);
        assert_eq!(
            manifest["wave_decoder.layers.7.ffn_norm.condition_proj.1.weight"],
            vec![1536, 128]
        );
        assert_eq!(
            manifest["wave_upsampler.upsample_layers.1.parametrizations.weight.original1"],
            vec![256, 128, 9]
        );
        assert_eq!(manifest["istft_head.out.weight"], vec![394, 512]);
    }

    #[test]
    fn incomplete_checkpoint_is_rejected_before_writing() {
        let payload = vec![0_u8; 768 * 4];
        let bytes = safetensors_one_f32("conv_downsample.bias", &[768], &payload);
        let file = SafetensorsFile::parse(bytes).expect("synthetic safetensors");
        let error = validate_input_manifest(&file).expect_err("must reject 1/350 tensors");
        assert!(format!("{error}").contains("expected exactly 350"));
    }

    #[test]
    fn pinned_checkpoint_cannot_be_relabelled() {
        assert_eq!(require_official_license(None).unwrap(), "mit");
        assert_eq!(require_official_license(Some(" MIT ")).unwrap(), "mit");
        let error = require_official_license(Some("apache-2.0")).unwrap_err();
        assert!(format!("{error}").contains("conflicts with the pinned official MIT"));
    }

    #[test]
    fn contract_stamps_revisions_hashes_and_topology() {
        let mut builder = GgufBuilder::new();
        stamp_contract(&mut builder, DEFAULT_LICENSE_SPDX);
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
        assert_eq!(
            file.get(KEY_UPSTREAM_REVISION).and_then(|v| v.as_str()),
            Some(UPSTREAM_REVISION)
        );
        assert_eq!(
            file.get(KEY_SOURCE_REVISION).and_then(|v| v.as_str()),
            Some(SOURCE_REVISION)
        );
        assert_eq!(
            file.get(KEY_MODEL_SHA256).and_then(|v| v.as_str()),
            Some(MODEL_SHA256)
        );
        assert_eq!(
            file.get(KEY_CONFIG_SHA256).and_then(|v| v.as_str()),
            Some(CONFIG_SHA256)
        );
        assert_eq!(
            file.get(KEY_SAMPLE_RATE).and_then(|v| v.as_u64()),
            Some(u64::from(SAMPLE_RATE))
        );
        assert_eq!(
            file.get(KEY_DECODE_ONLY).and_then(|v| v.as_bool()),
            Some(true)
        );
    }
}
