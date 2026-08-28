//! Exact converter for `JacobLinCool/MP-SENet-DNS`.
//!
//! The public checkpoint is the original-author `g_best_dns` generator
//! repackaged by `JacobLinCool/MPSENet` through `PyTorchModelHubMixin`. The
//! immutable release contains 247 F32 tensors. Missing, extra, renamed,
//! reshaped, or non-F32 tensors are rejected before an output file is written.
//!
//! The reference package intentionally instantiated PyTorch
//! `MultiheadAttention` without `batch_first=true`. That axis behaviour is
//! part of the released checkpoint contract; the converter records it rather
//! than silently applying the later upstream recommendation.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{FrontendSpec, GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "mp_senet";
pub const NAME: &str = "mp-senet-dns";
pub const CATEGORY: &str = "denoise";
pub const UPSTREAM_HF: &str = "JacobLinCool/MP-SENet-DNS";
pub const UPSTREAM_REVISION: &str = "8b78493f536df1aa53bd3bcbb2f620f705e8589c";
pub const REFERENCE_SOURCE: &str = "JacobLinCool/MPSENet";
/// Exact package revision whose public entry point defines bounded segment
/// processing, including the short-tail join and short-input padding policy.
pub const REFERENCE_REVISION: &str = "958141ca51703c5b1e0c30362ab5b1c8b0e49957";
/// Initial package revision used to publish the audited Hugging Face weights.
pub const PUBLICATION_REVISION: &str = "a65c76f340a0c8a885fbbf1893d5ec0ea009d718";
pub const OFFICIAL_SOURCE: &str = "yxlu-0102/MP-SENet";
pub const OFFICIAL_SOURCE_REVISION: &str = "89932cfe90d1dacb8e170e4a331d762462c21792";
pub const DEFAULT_LICENSE_SPDX: &str = "mit";
pub const SOURCE_LICENSE_SPDX: &str = "mit";
pub const MODEL_SHA256: &str = "74912046c8b352d78ca4056c9624d7256ac4d7eac45ce015822a7f2282749cdc";
pub const CONFIG_SHA256: &str = "0c5973617000142390726f8dad98a5b6b1429b4ef1a94da25f3bc009f86a3365";
pub const REFERENCE_MODEL_SHA256: &str =
    "e629e2858836489a598f9b325aa3abfc2a2360c72fc676d45c458c17efcaa7e8";
pub const PUBLICATION_MODEL_SHA256: &str =
    "63d0ddc067e87b5ebe556e60a89fa4384f5fba51fed37b6cb477abfaa19cb208";
pub const REFERENCE_TRANSFORMER_SHA256: &str =
    "44fb17b9a604f861304fd72517bfea73508393ca0ef00b58aaab6083c012ef0b";
pub const REFERENCE_LICENSE_SHA256: &str =
    "df6322ce3ca3c70a0845c4a384432a9af50e7d70886d316741e2f47b5ae01f34";
pub const OFFICIAL_LICENSE_SHA256: &str =
    "858f31052a5df6bcec94b015607bfade5a7cc6e950f7a9822aa4da3cc6f62fca";
pub const PUBLIC_REVISION: &str = "6017b7d70cf779c03f2fe061b56aa475e870d739";
pub const PUBLIC_MODEL_SHA256: &str =
    "26eec4a59c0eb8d31ea5115b3cb7d890f5b3745703ef0f0974b4e08c58e8da95";
pub const MANIFEST_SHA256: &str =
    "84f05f3ca25e7c8f56e217d57458ea63dd7a0516cad0aeae3e6a1880c3bfd8fe";

pub const TENSOR_COUNT: usize = 247;
pub const SAMPLE_RATE: u32 = 16_000;
pub const N_FFT: u32 = 400;
pub const HOP_LENGTH: u32 = 100;
pub const WIN_LENGTH: u32 = 400;
pub const DENSE_CHANNELS: u32 = 64;
pub const TS_BLOCKS: u32 = 4;
pub const ATTENTION_HEADS: u32 = 4;
pub const GRU_HIDDEN: u32 = 128;
pub const SEGMENT_SIZE: u32 = 32_000;
pub const COMPRESS_FACTOR: f32 = 0.3;
pub const MASK_BETA: f32 = 2.0;

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
pub const KEY_UPSTREAM_REVISION: &str = "vokra.mp_senet.upstream_revision";
pub const KEY_REFERENCE_SOURCE: &str = "vokra.mp_senet.reference_source";
pub const KEY_REFERENCE_REVISION: &str = "vokra.mp_senet.reference_revision";
pub const KEY_PUBLICATION_REVISION: &str = "vokra.mp_senet.publication_revision";
pub const KEY_OFFICIAL_SOURCE: &str = "vokra.mp_senet.official_source";
pub const KEY_OFFICIAL_SOURCE_REVISION: &str = "vokra.mp_senet.official_source_revision";
pub const KEY_REFERENCE_MODEL_SHA256: &str = "vokra.mp_senet.reference_model_sha256";
pub const KEY_PUBLICATION_MODEL_SHA256: &str = "vokra.mp_senet.publication_model_sha256";
pub const KEY_REFERENCE_TRANSFORMER_SHA256: &str = "vokra.mp_senet.reference_transformer_sha256";
pub const KEY_SOURCE_LICENSE: &str = "vokra.mp_senet.source_license";
pub const KEY_REFERENCE_LICENSE_SHA256: &str = "vokra.mp_senet.reference_license_sha256";
pub const KEY_OFFICIAL_LICENSE_SHA256: &str = "vokra.mp_senet.official_license_sha256";
pub const KEY_MODEL_SHA256: &str = "vokra.mp_senet.model_sha256";
pub const KEY_CONFIG_SHA256: &str = "vokra.mp_senet.config_sha256";
pub const KEY_PUBLIC_REVISION: &str = "vokra.mp_senet.public_revision";
pub const KEY_PUBLIC_MODEL_SHA256: &str = "vokra.mp_senet.public_model_sha256";
pub const KEY_MANIFEST_SHA256: &str = "vokra.mp_senet.manifest_sha256";
pub const KEY_SAMPLE_RATE: &str = "vokra.mp_senet.sample_rate";
pub const KEY_N_FFT: &str = "vokra.mp_senet.n_fft";
pub const KEY_HOP_LENGTH: &str = "vokra.mp_senet.hop_length";
pub const KEY_WIN_LENGTH: &str = "vokra.mp_senet.win_length";
pub const KEY_COMPRESS_FACTOR: &str = "vokra.mp_senet.compress_factor";
pub const KEY_MASK_BETA: &str = "vokra.mp_senet.mask_beta";
pub const KEY_DENSE_CHANNELS: &str = "vokra.mp_senet.dense_channels";
pub const KEY_TS_BLOCKS: &str = "vokra.mp_senet.ts_blocks";
pub const KEY_ATTENTION_HEADS: &str = "vokra.mp_senet.attention_heads";
pub const KEY_GRU_HIDDEN: &str = "vokra.mp_senet.gru_hidden";
pub const KEY_SEGMENT_SIZE: &str = "vokra.mp_senet.segment_size";
pub const KEY_ATTENTION_BATCH_FIRST: &str = "vokra.mp_senet.attention_batch_first";
pub const KEY_INSTANCE_NORM_EPS: &str = "vokra.mp_senet.instance_norm_eps";
pub const KEY_LAYER_NORM_EPS: &str = "vokra.mp_senet.layer_norm_eps";
pub const KEY_STFT_CENTER: &str = "vokra.mp_senet.stft_center";
pub const KEY_STFT_NORMALIZED: &str = "vokra.mp_senet.stft_normalized";
pub const KEY_STFT_ONESIDED: &str = "vokra.mp_senet.stft_onesided";
pub const KEY_HANN_PERIODIC: &str = "vokra.mp_senet.hann_periodic";
pub const KEY_MAGNITUDE_EPS: &str = "vokra.mp_senet.magnitude_eps";
pub const KEY_PHASE_IMAG_EPS: &str = "vokra.mp_senet.phase_imag_eps";
pub const KEY_PHASE_REAL_EPS: &str = "vokra.mp_senet.phase_real_eps";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MpSenetReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_mp_senet_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<MpSenetReport, ConvertError> {
    let spdx = require_official_license(license)?;
    let bytes = std::fs::read(input)?;
    let tensors = SafetensorsFile::parse(bytes)?;
    validate_input_manifest(&tensors)?;

    let mut builder = GgufBuilder::new();
    stamp_contract(&mut builder, spdx);
    for tensor in tensors.tensors() {
        builder.add_tensor(
            &tensor.name,
            tensor.dtype,
            tensor.shape.clone(),
            tensors.tensor_bytes(tensor).to_vec(),
        )?;
    }
    std::fs::write(output, builder.to_bytes()?)?;
    Ok(MpSenetReport {
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
            "license override {requested} conflicts with the pinned MIT checkpoint"
        )));
    }
    Ok(DEFAULT_LICENSE_SPDX)
}

fn stamp_contract(builder: &mut GgufBuilder, spdx: &str) {
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    builder.add_string(KEY_UPSTREAM_REVISION, UPSTREAM_REVISION);
    builder.add_string(KEY_REFERENCE_SOURCE, REFERENCE_SOURCE);
    builder.add_string(KEY_REFERENCE_REVISION, REFERENCE_REVISION);
    builder.add_string(KEY_PUBLICATION_REVISION, PUBLICATION_REVISION);
    builder.add_string(KEY_OFFICIAL_SOURCE, OFFICIAL_SOURCE);
    builder.add_string(KEY_OFFICIAL_SOURCE_REVISION, OFFICIAL_SOURCE_REVISION);
    builder.add_string(KEY_REFERENCE_MODEL_SHA256, REFERENCE_MODEL_SHA256);
    builder.add_string(KEY_PUBLICATION_MODEL_SHA256, PUBLICATION_MODEL_SHA256);
    builder.add_string(
        KEY_REFERENCE_TRANSFORMER_SHA256,
        REFERENCE_TRANSFORMER_SHA256,
    );
    builder.add_string(KEY_SOURCE_LICENSE, SOURCE_LICENSE_SPDX);
    builder.add_string(KEY_REFERENCE_LICENSE_SHA256, REFERENCE_LICENSE_SHA256);
    builder.add_string(KEY_OFFICIAL_LICENSE_SHA256, OFFICIAL_LICENSE_SHA256);
    builder.add_string(KEY_MODEL_SHA256, MODEL_SHA256);
    builder.add_string(KEY_CONFIG_SHA256, CONFIG_SHA256);
    builder.add_string(KEY_PUBLIC_REVISION, PUBLIC_REVISION);
    builder.add_string(KEY_PUBLIC_MODEL_SHA256, PUBLIC_MODEL_SHA256);
    builder.add_string(KEY_MANIFEST_SHA256, MANIFEST_SHA256);
    builder.add_u32(KEY_SAMPLE_RATE, SAMPLE_RATE);
    builder.add_u32(KEY_N_FFT, N_FFT);
    builder.add_u32(KEY_HOP_LENGTH, HOP_LENGTH);
    builder.add_u32(KEY_WIN_LENGTH, WIN_LENGTH);
    builder.add_f32(KEY_COMPRESS_FACTOR, COMPRESS_FACTOR);
    builder.add_f32(KEY_MASK_BETA, MASK_BETA);
    builder.add_u32(KEY_DENSE_CHANNELS, DENSE_CHANNELS);
    builder.add_u32(KEY_TS_BLOCKS, TS_BLOCKS);
    builder.add_u32(KEY_ATTENTION_HEADS, ATTENTION_HEADS);
    builder.add_u32(KEY_GRU_HIDDEN, GRU_HIDDEN);
    builder.add_u32(KEY_SEGMENT_SIZE, SEGMENT_SIZE);
    builder.add_bool(KEY_ATTENTION_BATCH_FIRST, false);
    builder.add_f32(KEY_INSTANCE_NORM_EPS, 1.0e-5);
    builder.add_f32(KEY_LAYER_NORM_EPS, 1.0e-5);
    builder.add_bool(KEY_STFT_CENTER, true);
    builder.add_bool(KEY_STFT_NORMALIZED, false);
    builder.add_bool(KEY_STFT_ONESIDED, true);
    builder.add_bool(KEY_HANN_PERIODIC, true);
    builder.add_f32(KEY_MAGNITUDE_EPS, 1.0e-9);
    builder.add_f32(KEY_PHASE_IMAG_EPS, 1.0e-10);
    builder.add_f32(KEY_PHASE_REAL_EPS, 1.0e-5);

    FrontendSpec {
        n_fft: N_FFT,
        hop: HOP_LENGTH,
        win_length: WIN_LENGTH,
        window_type: "hann".to_owned(),
        mel_norm: "none".to_owned(),
        htk_mode: false,
        fmin: 0.0,
        fmax: SAMPLE_RATE as f32 / 2.0,
        n_mels: 0,
        pad_mode: "reflect".to_owned(),
        dc_offset_removal: false,
        pre_emphasis: 0.0,
        sample_rate: SAMPLE_RATE,
    }
    .write_into(builder);

    vokra_core::stamp_provenance(
        builder,
        LicenseClass::Permissive,
        spdx,
        Some(NAME),
        Some(
            "JacobLinCool/MP-SENet-DNS (original-author g_best_dns generator; MIT weight and source declarations)",
        ),
    );
}

fn validate_input_manifest(tensors: &SafetensorsFile) -> Result<(), ConvertError> {
    let expected = expected_manifest();
    if expected.len() != TENSOR_COUNT || tensors.tensors().len() != TENSOR_COUNT {
        return Err(parse_error(format!(
            "checkpoint has {} tensors, expected exactly {TENSOR_COUNT} at revision {UPSTREAM_REVISION}",
            tensors.tensors().len()
        )));
    }
    let mut seen = BTreeSet::new();
    for tensor in tensors.tensors() {
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
                "tensor {} is {:?}, expected F32 at revision {UPSTREAM_REVISION}",
                tensor.name, tensor.dtype
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
    ConvertError::Parse(format!("mp_senet: {}", message.into()))
}

pub(crate) fn expected_manifest() -> BTreeMap<String, Vec<u64>> {
    let mut out = BTreeMap::new();
    for block in 0..TS_BLOCKS {
        for axis in ["time_transformer", "freq_transformer"] {
            add_transformer(&mut out, &format!("TSTransformer.{block}.{axis}"));
        }
    }
    add_dense_block(&mut out, "dense_encoder.dense_block");
    add_conv_norm_prelu(&mut out, "dense_encoder.dense_conv_1", 2, 64, 1, 1);
    add_conv_norm_prelu(&mut out, "dense_encoder.dense_conv_2", 64, 64, 1, 3);

    add_dense_block(&mut out, "mask_decoder.dense_block");
    add_decoder_stem(&mut out, "mask_decoder.mask_conv");
    insert(&mut out, "mask_decoder.mask_conv.3.bias", &[1]);
    insert(&mut out, "mask_decoder.mask_conv.3.weight", &[1, 64, 1, 2]);
    insert(&mut out, "mask_decoder.lsigmoid.slope", &[201, 1]);

    add_dense_block(&mut out, "phase_decoder.dense_block");
    add_decoder_stem(&mut out, "phase_decoder.phase_conv");
    for branch in ["phase_conv_i", "phase_conv_r"] {
        insert(&mut out, format!("phase_decoder.{branch}.bias"), &[1]);
        insert(
            &mut out,
            format!("phase_decoder.{branch}.weight"),
            &[1, 64, 1, 2],
        );
    }
    debug_assert_eq!(out.len(), TENSOR_COUNT);
    out
}

fn add_transformer(out: &mut BTreeMap<String, Vec<u64>>, prefix: &str) {
    insert(out, format!("{prefix}.attention.in_proj_bias"), &[192]);
    insert(
        out,
        format!("{prefix}.attention.in_proj_weight"),
        &[192, 64],
    );
    insert(out, format!("{prefix}.attention.out_proj.bias"), &[64]);
    insert(
        out,
        format!("{prefix}.attention.out_proj.weight"),
        &[64, 64],
    );
    for suffix in ["", "_reverse"] {
        insert(out, format!("{prefix}.ffn.gru.bias_hh_l0{suffix}"), &[384]);
        insert(out, format!("{prefix}.ffn.gru.bias_ih_l0{suffix}"), &[384]);
        insert(
            out,
            format!("{prefix}.ffn.gru.weight_hh_l0{suffix}"),
            &[384, 128],
        );
        insert(
            out,
            format!("{prefix}.ffn.gru.weight_ih_l0{suffix}"),
            &[384, 64],
        );
    }
    insert(out, format!("{prefix}.ffn.linear.bias"), &[64]);
    insert(out, format!("{prefix}.ffn.linear.weight"), &[64, 256]);
    for norm in 1..=3 {
        insert(out, format!("{prefix}.norm{norm}.bias"), &[64]);
        insert(out, format!("{prefix}.norm{norm}.weight"), &[64]);
    }
}

fn add_dense_block(out: &mut BTreeMap<String, Vec<u64>>, prefix: &str) {
    for layer in 0..4 {
        let root = format!("{prefix}.dense_block.{layer}");
        insert(out, format!("{root}.1.bias"), &[64]);
        insert(
            out,
            format!("{root}.1.weight"),
            &[64, 64 * (layer + 1), 2, 3],
        );
        insert(out, format!("{root}.2.bias"), &[64]);
        insert(out, format!("{root}.2.weight"), &[64]);
        insert(out, format!("{root}.3.weight"), &[64]);
    }
}

fn add_conv_norm_prelu(
    out: &mut BTreeMap<String, Vec<u64>>,
    prefix: &str,
    input: u64,
    output: u64,
    kh: u64,
    kw: u64,
) {
    insert(out, format!("{prefix}.0.bias"), &[output]);
    insert(out, format!("{prefix}.0.weight"), &[output, input, kh, kw]);
    insert(out, format!("{prefix}.1.bias"), &[output]);
    insert(out, format!("{prefix}.1.weight"), &[output]);
    insert(out, format!("{prefix}.2.weight"), &[output]);
}

fn add_decoder_stem(out: &mut BTreeMap<String, Vec<u64>>, prefix: &str) {
    insert(out, format!("{prefix}.0.conv.bias"), &[128]);
    insert(out, format!("{prefix}.0.conv.weight"), &[128, 64, 1, 3]);
    insert(out, format!("{prefix}.1.bias"), &[64]);
    insert(out, format!("{prefix}.1.weight"), &[64]);
    insert(out, format!("{prefix}.2.weight"), &[64]);
}

fn insert(out: &mut BTreeMap<String, Vec<u64>>, name: impl Into<String>, shape: &[u64]) {
    let name = name.into();
    assert!(out.insert(name.clone(), shape.to_vec()).is_none(), "{name}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    #[test]
    fn exact_manifest_matches_the_public_checkpoint() {
        let manifest = expected_manifest();
        assert_eq!(manifest.len(), TENSOR_COUNT);
        assert_eq!(
            manifest["TSTransformer.3.freq_transformer.ffn.gru.weight_hh_l0_reverse"],
            vec![384, 128]
        );
        assert_eq!(
            manifest["dense_encoder.dense_block.dense_block.3.1.weight"],
            vec![64, 256, 2, 3]
        );
        assert_eq!(
            manifest["mask_decoder.mask_conv.0.conv.weight"],
            vec![128, 64, 1, 3]
        );
        assert_eq!(
            manifest["phase_decoder.phase_conv_r.weight"],
            vec![1, 64, 1, 2]
        );
    }

    #[test]
    fn conflicting_license_override_is_rejected() {
        assert_eq!(require_official_license(None).unwrap(), "mit");
        assert_eq!(require_official_license(Some("MIT")).unwrap(), "mit");
        let error = require_official_license(Some("apache-2.0")).unwrap_err();
        assert!(error.to_string().contains("conflicts"));
    }

    #[test]
    fn additive_contract_records_the_axis_bug_and_all_pins() {
        let mut builder = GgufBuilder::new();
        stamp_contract(&mut builder, "mit");
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert_eq!(
            file.get(KEY_UPSTREAM_REVISION)
                .and_then(|value| value.as_str()),
            Some(UPSTREAM_REVISION)
        );
        assert_eq!(
            file.get(KEY_REFERENCE_REVISION)
                .and_then(|value| value.as_str()),
            Some(REFERENCE_REVISION)
        );
        assert_eq!(
            file.get(KEY_PUBLICATION_REVISION)
                .and_then(|value| value.as_str()),
            Some(PUBLICATION_REVISION)
        );
        assert_eq!(
            file.get(KEY_PUBLICATION_MODEL_SHA256)
                .and_then(|value| value.as_str()),
            Some(PUBLICATION_MODEL_SHA256)
        );
        assert_eq!(
            file.get(KEY_MANIFEST_SHA256)
                .and_then(|value| value.as_str()),
            Some(MANIFEST_SHA256)
        );
        assert_eq!(
            file.get(KEY_ATTENTION_BATCH_FIRST)
                .and_then(|value| value.as_bool()),
            Some(false)
        );
        assert_eq!(
            file.get(KEY_COMPRESS_FACTOR)
                .and_then(|value| value.as_f64()),
            Some(f64::from(COMPRESS_FACTOR))
        );
    }
}
