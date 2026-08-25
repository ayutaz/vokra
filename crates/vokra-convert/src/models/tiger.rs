//! Exact converter for the two official JusperLee TIGER checkpoints.
//!
//! TIGER-DnR and TIGER-speech share the tiger_separator runtime
//! architecture, but not a tensor topology: DnR wraps three independent
//! 57-band cores (2,304 tensors), while speech contains one 67-band core
//! (838 tensors). Missing, extra, renamed, reshaped, or non-F32 tensors are
//! rejected before an output file is written.
//!
//! The manifests and frontend parameters are transcribed from immutable
//! upstream revisions. Official weights are Apache-2.0; current source code
//! is MIT. No ONNX, protobuf, or runtime dependency is involved.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    FrontendSpec, GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "tiger_separator";
pub const CATEGORY: &str = "enhancement";
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";
pub const SOURCE_LICENSE_SPDX: &str = "mit";
pub const SOURCE_REVISION: &str = "9f18d4a10a7137e1ce8052cfb62215179f1287b6";
pub const SOURCE_LICENSE_SHA256: &str =
    "edc64d62aa021be7612337d2ced140375f52e4fd064b2f9cf6e656913d01bfa6";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
pub const KEY_TIGER_VARIANT: &str = "vokra.tiger.variant";
pub const KEY_UPSTREAM_REVISION: &str = "vokra.tiger.upstream_revision";
pub const KEY_SOURCE_REVISION: &str = "vokra.tiger.source_revision";
pub const KEY_SOURCE_FILE_SHA256: &str = "vokra.tiger.source_file_sha256";
pub const KEY_SOURCE_LICENSE: &str = "vokra.tiger.source_license";
pub const KEY_SOURCE_LICENSE_SHA256: &str = "vokra.tiger.source_license_sha256";
pub const KEY_MODEL_SHA256: &str = "vokra.tiger.model_sha256";
pub const KEY_CONFIG_SHA256: &str = "vokra.tiger.config_sha256";
pub const KEY_PUBLIC_REVISION: &str = "vokra.tiger.public_revision";
pub const KEY_PUBLIC_MODEL_SHA256: &str = "vokra.tiger.public_model_sha256";
pub const KEY_MANIFEST_SHA256: &str = "vokra.tiger.manifest_sha256";
pub const KEY_SAMPLE_RATE: &str = "vokra.tiger.sample_rate";
pub const KEY_N_FFT: &str = "vokra.tiger.n_fft";
pub const KEY_HOP_LENGTH: &str = "vokra.tiger.hop_length";
pub const KEY_FEATURE_CHANNELS: &str = "vokra.tiger.feature_channels";
pub const KEY_INTERNAL_CHANNELS: &str = "vokra.tiger.internal_channels";
pub const KEY_NUM_BLOCKS: &str = "vokra.tiger.num_blocks";
pub const KEY_NUM_SOURCES: &str = "vokra.tiger.num_sources";
pub const KEY_UPSAMPLING_DEPTH: &str = "vokra.tiger.upsampling_depth";
pub const KEY_ATTENTION_HEADS: &str = "vokra.tiger.attention_heads";
pub const KEY_ATTENTION_HIDDEN_CHANNELS: &str = "vokra.tiger.attention_hidden_channels";
pub const KEY_ATTENTION_KERNEL_SIZE: &str = "vokra.tiger.attention_kernel_size";
pub const KEY_ATTENTION_STRIDE: &str = "vokra.tiger.attention_stride";
pub const KEY_BAND_WIDTHS: &str = "vokra.tiger.band_widths";
pub const KEY_STFT_CENTER: &str = "vokra.tiger.stft_center";
pub const KEY_STFT_NORMALIZED: &str = "vokra.tiger.stft_normalized";
pub const KEY_STFT_ONESIDED: &str = "vokra.tiger.stft_onesided";
pub const KEY_HANN_PERIODIC: &str = "vokra.tiger.hann_periodic";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TigerVariant {
    Dnr,
    Speech,
}

impl TigerVariant {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Dnr => "tiger-dnr",
            Self::Speech => "tiger-speech",
        }
    }

    pub const fn tag(self) -> &'static str {
        match self {
            Self::Dnr => "dnr",
            Self::Speech => "speech",
        }
    }

    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Dnr => "JusperLee/TIGER-DnR",
            Self::Speech => "JusperLee/TIGER-speech",
        }
    }

    pub const fn upstream_revision(self) -> &'static str {
        match self {
            Self::Dnr => "b7a59560bbca10febbcd46fb01600f868e587f57",
            Self::Speech => "f0340340b2d9bbf72074edf8c076dcab59a10ba2",
        }
    }

    pub const fn model_sha256(self) -> &'static str {
        match self {
            Self::Dnr => "dd1c696e72f6adea0085ef1af640882a8260519ad666422835e387a5b4abdd2a",
            Self::Speech => "7e5fac7a9083c94b3a00c524f323188d4dd19ef09a54c29d1fec12ac114922db",
        }
    }

    pub const fn config_sha256(self) -> &'static str {
        match self {
            Self::Dnr => "ba9d2f833bf2f3a5855a35d0ccd11c786f6b92f1a482d84404bc4673edb29b54",
            Self::Speech => "1643c4e30cb97bc67024965aae13d631d44efdd304d8379cfd92143791017946",
        }
    }

    pub const fn source_file_sha256(self) -> &'static str {
        match self {
            Self::Dnr => "89605593bdfc05669e70f2b8647514077197f9870d32b5dd745913f6e03b50e0",
            Self::Speech => "a90ec403c5c024a1c6722a5143e0bd37bb642edec0e1506787ea212a65b287fe",
        }
    }

    pub const fn public_revision(self) -> &'static str {
        match self {
            Self::Dnr => "8c8c78888684ecc8eef6beca3434c7ec9247bb70",
            Self::Speech => "e50793924eaae3897cee01f7f7791d14c296c7ed",
        }
    }

    pub const fn public_model_sha256(self) -> &'static str {
        match self {
            Self::Dnr => "8737e4993efefbfec57ed7a0924503d626d07e410f456ff5693402852784017f",
            Self::Speech => "1fc11c3476bb6938410935e4f1877dcc2fb82005bf4ec0503dc01c013c29e562",
        }
    }

    pub const fn manifest_sha256(self) -> &'static str {
        match self {
            Self::Dnr => "f1daf2c510ef2c272711963a940e1dad74b795a1f04b2b1a524e00c61d307c02",
            Self::Speech => "dd0f9c0f252c9df0498d1e4c516df9ec1bf1230b64b6fbeec2147525cb711ee1",
        }
    }

    pub const fn tensor_count(self) -> usize {
        match self {
            Self::Dnr => 2_304,
            Self::Speech => 838,
        }
    }

    pub const fn sample_rate(self) -> u32 {
        match self {
            Self::Dnr => 44_100,
            Self::Speech => 16_000,
        }
    }

    pub const fn n_fft(self) -> u32 {
        match self {
            Self::Dnr => 2_048,
            Self::Speech => 640,
        }
    }

    pub const fn hop_length(self) -> u32 {
        match self {
            Self::Dnr => 512,
            Self::Speech => 160,
        }
    }

    pub const fn feature_channels(self) -> u32 {
        match self {
            Self::Dnr => 132,
            Self::Speech => 128,
        }
    }

    pub const fn num_sources(self) -> u32 {
        match self {
            Self::Dnr => 3,
            Self::Speech => 2,
        }
    }

    pub fn band_widths(self) -> Vec<u32> {
        let mut widths = Vec::new();
        match self {
            Self::Dnr => {
                widths.extend(std::iter::repeat_n(2, 20));
                widths.extend(std::iter::repeat_n(4, 10));
                widths.extend(std::iter::repeat_n(11, 8));
                widths.extend(std::iter::repeat_n(23, 8));
                widths.extend(std::iter::repeat_n(46, 8));
                widths.extend(std::iter::repeat_n(92, 2));
                widths.push(121);
            }
            Self::Speech => {
                widths.extend(std::iter::repeat_n(1, 40));
                widths.extend(std::iter::repeat_n(4, 10));
                widths.extend(std::iter::repeat_n(10, 8));
                widths.extend(std::iter::repeat_n(20, 8));
                widths.push(1);
            }
        }
        widths
    }

    pub const fn source_description(self) -> &'static str {
        match self {
            Self::Dnr => {
                "JusperLee/TIGER-DnR (DnR dialog/effect/music separation; Apache-2.0 weights, MIT code)"
            }
            Self::Speech => {
                "JusperLee/TIGER-speech (two-speaker separation; Apache-2.0 weights, MIT code)"
            }
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TigerReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
    pub variant: Option<TigerVariant>,
}

pub fn convert_tiger_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    variant: TigerVariant,
) -> Result<TigerReport, ConvertError> {
    let spdx = require_official_license(license)?;
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;
    validate_input_manifest(&st, variant)?;

    let mut builder = GgufBuilder::new();
    stamp_contract(&mut builder, variant, spdx);
    for tensor in st.tensors() {
        builder.add_tensor(
            &tensor.name,
            tensor.dtype,
            tensor.shape.clone(),
            st.tensor_bytes(tensor).to_vec(),
        )?;
    }
    std::fs::write(output, builder.to_bytes()?)?;
    Ok(TigerReport {
        read: variant.tensor_count(),
        written: variant.tensor_count(),
        skipped_non_float: 0,
        bf16_passthrough: 0,
        variant: Some(variant),
    })
}

fn require_official_license(license: Option<&str>) -> Result<&'static str, ConvertError> {
    let requested = license.unwrap_or(DEFAULT_LICENSE_SPDX).trim();
    if !requested.eq_ignore_ascii_case(DEFAULT_LICENSE_SPDX) {
        return Err(parse_error(format!(
            "license override {requested} conflicts with the pinned official Apache-2.0 checkpoint"
        )));
    }
    Ok(DEFAULT_LICENSE_SPDX)
}

fn stamp_contract(builder: &mut GgufBuilder, variant: TigerVariant, spdx: &str) {
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, variant.name());
    builder.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_HF, variant.upstream_hf());
    builder.add_string(KEY_TIGER_VARIANT, variant.tag());
    builder.add_string(KEY_UPSTREAM_REVISION, variant.upstream_revision());
    builder.add_string(KEY_SOURCE_REVISION, SOURCE_REVISION);
    builder.add_string(KEY_SOURCE_FILE_SHA256, variant.source_file_sha256());
    builder.add_string(KEY_SOURCE_LICENSE, SOURCE_LICENSE_SPDX);
    builder.add_string(KEY_SOURCE_LICENSE_SHA256, SOURCE_LICENSE_SHA256);
    builder.add_string(KEY_MODEL_SHA256, variant.model_sha256());
    builder.add_string(KEY_CONFIG_SHA256, variant.config_sha256());
    builder.add_string(KEY_PUBLIC_REVISION, variant.public_revision());
    builder.add_string(KEY_PUBLIC_MODEL_SHA256, variant.public_model_sha256());
    builder.add_string(KEY_MANIFEST_SHA256, variant.manifest_sha256());
    builder.add_u32(KEY_SAMPLE_RATE, variant.sample_rate());
    builder.add_u32(KEY_N_FFT, variant.n_fft());
    builder.add_u32(KEY_HOP_LENGTH, variant.hop_length());
    builder.add_u32(KEY_FEATURE_CHANNELS, variant.feature_channels());
    builder.add_u32(KEY_INTERNAL_CHANNELS, 256);
    builder.add_u32(KEY_NUM_BLOCKS, 8);
    builder.add_u32(KEY_NUM_SOURCES, variant.num_sources());
    builder.add_u32(KEY_UPSAMPLING_DEPTH, 5);
    builder.add_u32(KEY_ATTENTION_HEADS, 4);
    builder.add_u32(KEY_ATTENTION_HIDDEN_CHANNELS, 4);
    builder.add_u32(KEY_ATTENTION_KERNEL_SIZE, 8);
    builder.add_u32(KEY_ATTENTION_STRIDE, 1);
    add_u32_array(builder, KEY_BAND_WIDTHS, &variant.band_widths());
    builder.add_bool(KEY_STFT_CENTER, true);
    builder.add_bool(KEY_STFT_NORMALIZED, false);
    builder.add_bool(KEY_STFT_ONESIDED, true);
    builder.add_bool(KEY_HANN_PERIODIC, true);

    FrontendSpec {
        n_fft: variant.n_fft(),
        hop: variant.hop_length(),
        win_length: variant.n_fft(),
        window_type: "hann".to_owned(),
        mel_norm: "none".to_owned(),
        htk_mode: false,
        fmin: 0.0,
        fmax: variant.sample_rate() as f32 / 2.0,
        n_mels: 0,
        pad_mode: "reflect".to_owned(),
        dc_offset_removal: false,
        pre_emphasis: 0.0,
        sample_rate: variant.sample_rate(),
    }
    .write_into(builder);

    vokra_core::stamp_provenance(
        builder,
        LicenseClass::Permissive,
        spdx,
        Some(variant.name()),
        Some(variant.source_description()),
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

fn validate_input_manifest(
    st: &SafetensorsFile,
    variant: TigerVariant,
) -> Result<(), ConvertError> {
    let expected = expected_manifest(variant);
    if expected.len() != variant.tensor_count() || st.tensors().len() != variant.tensor_count() {
        return Err(parse_error(format!(
            "{} checkpoint has {} tensors, expected exactly {} at revision {}",
            variant.name(),
            st.tensors().len(),
            variant.tensor_count(),
            variant.upstream_revision()
        )));
    }
    let mut seen = BTreeSet::new();
    for tensor in st.tensors() {
        let shape = expected
            .get(&tensor.name)
            .ok_or_else(|| parse_error(format!("unexpected tensor {}", tensor.name)))?;
        if &tensor.shape != shape {
            return Err(parse_error(format!(
                "tensor {} has shape {:?}, expected {shape:?}",
                tensor.name, tensor.shape
            )));
        }
        if tensor.dtype != GgmlType::F32 {
            return Err(parse_error(format!(
                "tensor {} is {:?}, expected F32 at revision {}",
                tensor.name,
                tensor.dtype,
                variant.upstream_revision()
            )));
        }
        seen.insert(tensor.name.as_str());
    }
    if let Some(missing) = expected.keys().find(|name| !seen.contains(name.as_str())) {
        return Err(parse_error(format!(
            "checkpoint is missing tensor {missing}"
        )));
    }
    Ok(())
}

fn parse_error(message: impl Into<String>) -> ConvertError {
    ConvertError::Parse(format!("tiger: {}", message.into()))
}

/// Full official name/shape manifest for one TIGER release.
pub(crate) fn expected_manifest(variant: TigerVariant) -> BTreeMap<String, Vec<u64>> {
    let mut out = BTreeMap::new();
    let widths = variant.band_widths();
    match variant {
        TigerVariant::Dnr => {
            for root in ["dialog.", "effect.", "music."] {
                add_core(&mut out, root, &widths, 132, 3);
            }
        }
        TigerVariant::Speech => add_core(&mut out, "", &widths, 128, 2),
    }
    debug_assert_eq!(out.len(), variant.tensor_count());
    out
}

fn add_core(
    out: &mut BTreeMap<String, Vec<u64>>,
    root: &str,
    widths: &[u32],
    channels: u32,
    sources: u32,
) {
    for (band, &width) in widths.iter().enumerate() {
        let complex = u64::from(2 * width);
        insert(out, format!("{root}BN.{band}.0.weight"), &[complex]);
        insert(out, format!("{root}BN.{band}.0.bias"), &[complex]);
        insert(
            out,
            format!("{root}BN.{band}.1.weight"),
            &[u64::from(channels), complex, 1],
        );
        insert(
            out,
            format!("{root}BN.{band}.1.bias"),
            &[u64::from(channels)],
        );
        let mask_channels = u64::from(4 * sources * width);
        insert(out, format!("{root}mask.{band}.0.weight"), &[1]);
        insert(
            out,
            format!("{root}mask.{band}.1.weight"),
            &[mask_channels, u64::from(channels / sources), 1],
        );
        insert(out, format!("{root}mask.{band}.1.bias"), &[mask_channels]);
    }

    insert(
        out,
        format!("{root}separator.concat_block.0.weight"),
        &[u64::from(channels), 1, 1, 1],
    );
    insert(
        out,
        format!("{root}separator.concat_block.0.bias"),
        &[u64::from(channels)],
    );
    insert(out, format!("{root}separator.concat_block.1.weight"), &[1]);
    for path in ["freq_path", "frame_path"] {
        let prefix = format!("{root}separator.{path}");
        add_uconv(out, &format!("{prefix}.0"), channels);
        add_attention(out, &format!("{prefix}.1"), channels);
        insert(
            out,
            format!("{prefix}.2.gamma"),
            &[1, u64::from(channels), 1, 1],
        );
        insert(
            out,
            format!("{prefix}.2.beta"),
            &[1, u64::from(channels), 1, 1],
        );
    }
}

fn add_uconv(out: &mut BTreeMap<String, Vec<u64>>, prefix: &str, channels: u32) {
    insert(
        out,
        format!("{prefix}.proj_1x1.conv.weight"),
        &[256, u64::from(channels), 1],
    );
    insert(out, format!("{prefix}.proj_1x1.conv.bias"), &[256]);
    insert(out, format!("{prefix}.proj_1x1.norm.weight"), &[256]);
    insert(out, format!("{prefix}.proj_1x1.norm.bias"), &[256]);
    insert(out, format!("{prefix}.proj_1x1.act.weight"), &[1]);
    for stage in 0..5 {
        let stage = format!("{prefix}.spp_dw.{stage}");
        insert(out, format!("{stage}.conv.weight"), &[256, 1, 5]);
        insert(out, format!("{stage}.conv.bias"), &[256]);
        insert(out, format!("{stage}.norm.weight"), &[256]);
        insert(out, format!("{stage}.norm.bias"), &[256]);
    }
    for stage in 0..5 {
        add_injection(out, &format!("{prefix}.loc_glo_fus.{stage}"), 1);
    }
    insert(
        out,
        format!("{prefix}.globalatt.fc1.conv.weight"),
        &[256, 256, 1],
    );
    insert(out, format!("{prefix}.globalatt.fc1.norm.weight"), &[256]);
    insert(out, format!("{prefix}.globalatt.fc1.norm.bias"), &[256]);
    insert(
        out,
        format!("{prefix}.globalatt.dwconv.weight"),
        &[256, 1, 5],
    );
    insert(out, format!("{prefix}.globalatt.dwconv.bias"), &[256]);
    insert(
        out,
        format!("{prefix}.globalatt.fc2.conv.weight"),
        &[256, 256, 1],
    );
    insert(out, format!("{prefix}.globalatt.fc2.norm.weight"), &[256]);
    insert(out, format!("{prefix}.globalatt.fc2.norm.bias"), &[256]);
    for stage in 0..4 {
        add_injection(out, &format!("{prefix}.last_layer.{stage}"), 5);
    }
    insert(
        out,
        format!("{prefix}.res_conv.weight"),
        &[u64::from(channels), 256, 1],
    );
    insert(
        out,
        format!("{prefix}.res_conv.bias"),
        &[u64::from(channels)],
    );
}

fn add_injection(out: &mut BTreeMap<String, Vec<u64>>, prefix: &str, kernel: u64) {
    for branch in ["local_embedding", "global_embedding", "global_act"] {
        let branch = format!("{prefix}.{branch}");
        insert(out, format!("{branch}.conv.weight"), &[256, 1, kernel]);
        insert(out, format!("{branch}.norm.weight"), &[256]);
        insert(out, format!("{branch}.norm.bias"), &[256]);
    }
}

fn add_attention(out: &mut BTreeMap<String, Vec<u64>>, prefix: &str, channels: u32) {
    let value_channels = channels / 4;
    for head in 0..4 {
        for (projection, output) in [("Queries", 4), ("Keys", 4), ("Values", value_channels)] {
            add_attention_projection(
                out,
                &format!("{prefix}.{projection}.{head}"),
                channels,
                output,
            );
        }
    }
    add_attention_projection(
        out,
        &format!("{prefix}.attn_concat_proj"),
        channels,
        channels,
    );
}

fn add_attention_projection(
    out: &mut BTreeMap<String, Vec<u64>>,
    prefix: &str,
    input: u32,
    output: u32,
) {
    insert(
        out,
        format!("{prefix}.conv.weight"),
        &[u64::from(output), u64::from(input), 1, 1],
    );
    insert(out, format!("{prefix}.conv.bias"), &[u64::from(output)]);
    insert(out, format!("{prefix}.act.weight"), &[1]);
    insert(
        out,
        format!("{prefix}.norm.gamma"),
        &[1, u64::from(output), 1, 1],
    );
    insert(
        out,
        format!("{prefix}.norm.beta"),
        &[1, u64::from(output), 1, 1],
    );
}

fn insert(out: &mut BTreeMap<String, Vec<u64>>, name: impl Into<String>, shape: &[u64]) {
    let name = name.into();
    assert!(out.insert(name.clone(), shape.to_vec()).is_none(), "{name}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_variant_manifests_match_published_counts_and_band_sums() {
        for variant in [TigerVariant::Dnr, TigerVariant::Speech] {
            let manifest = expected_manifest(variant);
            assert_eq!(manifest.len(), variant.tensor_count());
            assert_eq!(
                variant.band_widths().iter().sum::<u32>(),
                variant.n_fft() / 2 + 1
            );
        }
        let dnr = expected_manifest(TigerVariant::Dnr);
        assert_eq!(dnr["dialog.BN.56.1.weight"], vec![132, 242, 1]);
        assert_eq!(dnr["music.mask.56.1.weight"], vec![1452, 44, 1]);
        assert_eq!(
            dnr["effect.separator.frame_path.1.Values.3.conv.weight"],
            vec![33, 132, 1, 1]
        );
        let speech = expected_manifest(TigerVariant::Speech);
        assert_eq!(speech["BN.66.1.weight"], vec![128, 2, 1]);
        assert_eq!(speech["mask.66.1.weight"], vec![8, 64, 1]);
        assert_eq!(
            speech["separator.freq_path.1.Values.3.conv.weight"],
            vec![32, 128, 1, 1]
        );
    }

    #[test]
    fn conflicting_license_override_is_rejected() {
        assert_eq!(require_official_license(None).unwrap(), "apache-2.0");
        assert_eq!(
            require_official_license(Some("Apache-2.0")).unwrap(),
            "apache-2.0"
        );
        let error = require_official_license(Some("mit")).unwrap_err();
        assert!(error.to_string().contains("conflicts"));
    }

    #[test]
    fn variant_pins_are_distinct_and_immutable() {
        assert_ne!(
            TigerVariant::Dnr.upstream_revision(),
            TigerVariant::Speech.upstream_revision()
        );
        assert_ne!(
            TigerVariant::Dnr.model_sha256(),
            TigerVariant::Speech.model_sha256()
        );
        assert_ne!(
            TigerVariant::Dnr.manifest_sha256(),
            TigerVariant::Speech.manifest_sha256()
        );
        assert_eq!(SOURCE_LICENSE_SPDX, "mit");
        assert_eq!(DEFAULT_LICENSE_SPDX, "apache-2.0");
    }
}
