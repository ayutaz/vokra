//! Exact AudioSeal 0.2 four-checkpoint bundle converter.
//!
//! The official `facebook/audioseal` repository publishes two generators and
//! two detectors: non-causal `base` checkpoints and causal `streaming`
//! checkpoints. The offline preparation script flattens those four PyTorch
//! state dicts into one safetensors file under the prefixes
//! `generator_{base,streaming}` / `detector_{base,streaming}`. This converter
//! accepts exactly the audited 310-tensor F32 manifest and writes it verbatim
//! to GGUF. Missing, extra, renamed, reshaped, or non-F32 tensors are errors;
//! an arbitrary float checkpoint can no longer acquire the AudioSeal arch tag.
//!
//! The fixed topology is transcribed from AudioSeal source revision
//! [`SOURCE_REVISION`], using the four weight files at Hugging Face revision
//! [`CHECKPOINT_REVISION`]. The native runtime consumes these metadata keys and
//! the same complete manifest. The already-published historical GGUF predates
//! the topology keys; runtime compatibility for it is deliberately isolated
//! behind an exact full-header contract rather than a general "missing metadata
//! means defaults" rule.
//!
//! # License and execution boundary
//!
//! AudioSeal code and weights are MIT. This converter never loads ONNX or a
//! PyTorch pickle: pickle deserialization is confined to the offline VAST-only
//! preparation script, and the Rust converter sees safetensors only. Runtime
//! embedding/detection is explicit model use; enabling AudioSeal automatically
//! for every TTS output remains a separate compliance-policy decision.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for the four-checkpoint AudioSeal bundle.
pub const ARCH: &str = "audioseal_real_weight";
/// Canonical Vokra model name.
pub const NAME: &str = "audioseal_real_weight";
/// First-class audio-watermark model category.
pub const CATEGORY: &str = "watermark";
/// Official upstream Hugging Face repository.
pub const UPSTREAM_HF: &str = "facebook/audioseal";
/// Immutable revision containing all four audited checkpoint files.
pub const CHECKPOINT_REVISION: &str = "3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba";
/// Immutable AudioSeal source revision used to transcribe the forward.
pub const SOURCE_REVISION: &str = "e63a8a0e5cdf7bb797159c92ba15961557fe9bd2";
/// MIT weight license.
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

/// Number of tensors in the exact four-checkpoint bundle.
pub const TENSOR_COUNT: usize = 310;
/// AudioSeal checkpoint sample rate.
pub const SAMPLE_RATE: u32 = 16_000;
/// Embedded message width.
pub const NBITS: u32 = 16;
/// SEANet latent width.
pub const DIMENSION: u32 = 128;
/// SEANet base channel width.
pub const N_FILTERS: u32 = 32;
/// Product of `[8, 5, 4, 2]`.
pub const HOP_LENGTH: u32 = 320;

pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
pub const KEY_CHECKPOINT_REVISION: &str = "vokra.audioseal.checkpoint_revision";
pub const KEY_SOURCE_REVISION: &str = "vokra.audioseal.source_revision";
pub const KEY_SAMPLE_RATE: &str = "vokra.audioseal.sample_rate";
pub const KEY_NBITS: &str = "vokra.audioseal.nbits";
pub const KEY_CHANNELS: &str = "vokra.audioseal.channels";
pub const KEY_DIMENSION: &str = "vokra.audioseal.dimension";
pub const KEY_N_FILTERS: &str = "vokra.audioseal.n_filters";
pub const KEY_N_RESIDUAL_LAYERS: &str = "vokra.audioseal.n_residual_layers";
pub const KEY_RATIOS: &str = "vokra.audioseal.ratios";
pub const KEY_ACTIVATION: &str = "vokra.audioseal.activation";
pub const KEY_COMPRESS: &str = "vokra.audioseal.compress";
pub const KEY_DILATION_BASE: &str = "vokra.audioseal.dilation_base";
pub const KEY_KERNEL_SIZE: &str = "vokra.audioseal.kernel_size";
pub const KEY_LAST_KERNEL_SIZE: &str = "vokra.audioseal.last_kernel_size";
pub const KEY_RESIDUAL_KERNEL_SIZE: &str = "vokra.audioseal.residual_kernel_size";
pub const KEY_LSTM_LAYERS: &str = "vokra.audioseal.lstm_layers";
pub const KEY_NORM: &str = "vokra.audioseal.norm";
pub const KEY_PAD_MODE: &str = "vokra.audioseal.pad_mode";
pub const KEY_TRUE_SKIP: &str = "vokra.audioseal.true_skip";
pub const KEY_BASE_CAUSAL: &str = "vokra.audioseal.base_causal";
pub const KEY_STREAMING_CAUSAL: &str = "vokra.audioseal.streaming_causal";
pub const KEY_DETECTOR_OUTPUT_DIM: &str = "vokra.audioseal.detector_output_dim";
pub const KEY_HOP_LENGTH: &str = "vokra.audioseal.hop_length";
pub const KEY_NORMALIZER: &str = "vokra.audioseal.normalizer";

/// Outcome of an exact AudioSeal conversion.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AudiosealRealWeightReport {
    /// Tensor entries observed after the exact manifest gate.
    pub read: usize,
    /// F32 tensors written verbatim.
    pub written: usize,
    /// Retained for API compatibility; exact conversion always leaves it zero.
    pub skipped_non_float: usize,
    /// Retained for API compatibility; the audited checkpoints are F32.
    pub bf16_passthrough: usize,
}

/// Convert an exact four-checkpoint AudioSeal safetensors bundle to GGUF.
///
/// The input is normally produced by
/// `tools/parity/audioseal_prepare_checkpoint.py` on VAST. No tensor is written
/// until all 310 descriptors have passed the strict name/shape/dtype gate.
pub fn convert_audioseal_real_weight_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<AudiosealRealWeightReport, ConvertError> {
    let license = require_official_license(license)?;
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;
    validate_input_manifest(&st)?;

    let mut builder = GgufBuilder::new();
    stamp_contract(&mut builder, license);

    for tensor in st.tensors() {
        builder.add_tensor(
            &tensor.name,
            tensor.dtype,
            tensor.shape.clone(),
            st.tensor_bytes(tensor).to_vec(),
        )?;
    }

    std::fs::write(output, builder.to_bytes()?)?;
    Ok(AudiosealRealWeightReport {
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
            "license override `{requested}` conflicts with the pinned official MIT checkpoints"
        )));
    }
    Ok(DEFAULT_LICENSE_SPDX)
}

fn stamp_contract(builder: &mut GgufBuilder, spdx: &str) {
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    builder.add_string(KEY_CHECKPOINT_REVISION, CHECKPOINT_REVISION);
    builder.add_string(KEY_SOURCE_REVISION, SOURCE_REVISION);
    builder.add_u32(KEY_SAMPLE_RATE, SAMPLE_RATE);
    builder.add_u32(KEY_NBITS, NBITS);
    builder.add_u32(KEY_CHANNELS, 1);
    builder.add_u32(KEY_DIMENSION, DIMENSION);
    builder.add_u32(KEY_N_FILTERS, N_FILTERS);
    builder.add_u32(KEY_N_RESIDUAL_LAYERS, 1);
    add_u32_array(builder, KEY_RATIOS, &[8, 5, 4, 2]);
    builder.add_string(KEY_ACTIVATION, "elu");
    builder.add_u32(KEY_COMPRESS, 2);
    builder.add_u32(KEY_DILATION_BASE, 2);
    builder.add_u32(KEY_KERNEL_SIZE, 7);
    builder.add_u32(KEY_LAST_KERNEL_SIZE, 7);
    builder.add_u32(KEY_RESIDUAL_KERNEL_SIZE, 3);
    builder.add_u32(KEY_LSTM_LAYERS, 2);
    builder.add_string(KEY_NORM, "weight_norm");
    builder.add_string(KEY_PAD_MODE, "constant");
    builder.add_bool(KEY_TRUE_SKIP, true);
    builder.add_bool(KEY_BASE_CAUSAL, false);
    builder.add_bool(KEY_STREAMING_CAUSAL, true);
    builder.add_u32(KEY_DETECTOR_OUTPUT_DIM, 32);
    builder.add_u32(KEY_HOP_LENGTH, HOP_LENGTH);
    builder.add_bool(KEY_NORMALIZER, false);

    let class = LicenseClass::from_license_str(spdx);
    vokra_core::stamp_provenance(
        builder,
        class,
        spdx,
        Some(NAME),
        Some(
            "facebook/audioseal generator_base + detector_base + generator_streaming + \
             detector_streaming, 16-bit audio watermark, MIT; explicit native runtime \
             binding, automatic TTS watermark policy remains separately configurable",
        ),
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
            "checkpoint has {} tensors, expected exactly {TENSOR_COUNT} \
             (four official base/streaming generator/detector checkpoints)",
            st.tensors().len()
        )));
    }
    let mut seen = BTreeSet::new();
    for tensor in st.tensors() {
        let shape = expected
            .get(&tensor.name)
            .ok_or_else(|| parse_error(format!("unexpected AudioSeal tensor `{}`", tensor.name)))?;
        if &tensor.shape != shape {
            return Err(parse_error(format!(
                "AudioSeal tensor `{}` has shape {:?}, expected {shape:?}",
                tensor.name, tensor.shape
            )));
        }
        if tensor.dtype != GgmlType::F32 {
            return Err(parse_error(format!(
                "AudioSeal tensor `{}` is {:?}, expected F32 at checkpoint revision {CHECKPOINT_REVISION}",
                tensor.name, tensor.dtype
            )));
        }
        seen.insert(tensor.name.as_str());
    }
    if let Some(missing) = expected.keys().find(|name| !seen.contains(name.as_str())) {
        return Err(parse_error(format!(
            "AudioSeal checkpoint is missing tensor `{missing}`"
        )));
    }
    Ok(())
}

fn parse_error(message: impl Into<String>) -> ConvertError {
    ConvertError::Parse(format!("audioseal-real-weight: {}", message.into()))
}

#[derive(Clone, Copy)]
enum WeightNames {
    Legacy,
    Parametrized,
}

/// Exact public/source manifest. Kept generated from the fixed topology so
/// channel arithmetic remains reviewable while still requiring every concrete
/// upstream name.
fn expected_manifest() -> BTreeMap<String, Vec<u64>> {
    let mut out = BTreeMap::new();
    add_detector(&mut out, "detector_base", WeightNames::Legacy);
    add_detector(&mut out, "detector_streaming", WeightNames::Parametrized);
    add_generator(&mut out, "generator_base");
    add_generator(&mut out, "generator_streaming");
    debug_assert_eq!(out.len(), TENSOR_COUNT);
    out
}

fn add_generator(out: &mut BTreeMap<String, Vec<u64>>, root: &str) {
    add_decoder(out, &format!("{root}.decoder"), WeightNames::Legacy);
    add_encoder(out, &format!("{root}.encoder"), WeightNames::Legacy);
    insert(
        out,
        format!("{root}.msg_processor.msg_processor.weight"),
        &[32, 128],
    );
}

fn add_detector(out: &mut BTreeMap<String, Vec<u64>>, root: &str, names: WeightNames) {
    let encoder = format!("{root}.detector.0");
    add_encoder(out, &encoder, names);
    insert(
        out,
        format!("{encoder}.reverse_convolution.weight"),
        &[128, 32, 320],
    );
    insert(out, format!("{encoder}.reverse_convolution.bias"), &[32]);
    insert(out, format!("{root}.detector.1.weight"), &[18, 32, 1]);
    insert(out, format!("{root}.detector.1.bias"), &[18]);
}

fn add_encoder(out: &mut BTreeMap<String, Vec<u64>>, prefix: &str, names: WeightNames) {
    add_conv(out, &format!("{prefix}.model.0.conv.conv"), 32, 1, 7, names);
    for (residual, down, channels, next, kernel) in [
        (1, 3, 32, 64, 4),
        (4, 6, 64, 128, 8),
        (7, 9, 128, 256, 10),
        (10, 12, 256, 512, 16),
    ] {
        add_residual(out, prefix, residual, channels, names);
        add_conv(
            out,
            &format!("{prefix}.model.{down}.conv.conv"),
            next,
            channels,
            kernel,
            names,
        );
    }
    add_lstm(out, &format!("{prefix}.model.13.lstm"));
    add_conv(
        out,
        &format!("{prefix}.model.15.conv.conv"),
        128,
        512,
        7,
        names,
    );
}

fn add_decoder(out: &mut BTreeMap<String, Vec<u64>>, prefix: &str, names: WeightNames) {
    add_conv(
        out,
        &format!("{prefix}.model.0.conv.conv"),
        512,
        128,
        7,
        names,
    );
    add_lstm(out, &format!("{prefix}.model.1.lstm"));
    for (transpose, residual, channels, next, kernel) in [
        (3, 4, 512, 256, 16),
        (6, 7, 256, 128, 10),
        (9, 10, 128, 64, 8),
        (12, 13, 64, 32, 4),
    ] {
        add_conv_transpose(
            out,
            &format!("{prefix}.model.{transpose}.convtr.convtr"),
            channels,
            next,
            kernel,
            names,
        );
        add_residual(out, prefix, residual, next, names);
    }
    add_conv(
        out,
        &format!("{prefix}.model.15.conv.conv"),
        1,
        32,
        7,
        names,
    );
}

fn add_residual(
    out: &mut BTreeMap<String, Vec<u64>>,
    prefix: &str,
    index: usize,
    channels: u64,
    names: WeightNames,
) {
    let hidden = channels / 2;
    add_conv(
        out,
        &format!("{prefix}.model.{index}.block.1.conv.conv"),
        hidden,
        channels,
        3,
        names,
    );
    add_conv(
        out,
        &format!("{prefix}.model.{index}.block.3.conv.conv"),
        channels,
        hidden,
        1,
        names,
    );
}

fn add_conv(
    out: &mut BTreeMap<String, Vec<u64>>,
    prefix: &str,
    output: u64,
    input: u64,
    kernel: u64,
    names: WeightNames,
) {
    add_weight_norm(out, prefix, output, &[output, input, kernel], names);
    insert(out, format!("{prefix}.bias"), &[output]);
}

fn add_conv_transpose(
    out: &mut BTreeMap<String, Vec<u64>>,
    prefix: &str,
    input: u64,
    output: u64,
    kernel: u64,
    names: WeightNames,
) {
    add_weight_norm(out, prefix, input, &[input, output, kernel], names);
    insert(out, format!("{prefix}.bias"), &[output]);
}

fn add_weight_norm(
    out: &mut BTreeMap<String, Vec<u64>>,
    prefix: &str,
    primary: u64,
    weight_shape: &[u64],
    names: WeightNames,
) {
    let (g, v) = match names {
        WeightNames::Legacy => ("weight_g", "weight_v"),
        WeightNames::Parametrized => (
            "parametrizations.weight.original0",
            "parametrizations.weight.original1",
        ),
    };
    insert(out, format!("{prefix}.{g}"), &[primary, 1, 1]);
    insert(out, format!("{prefix}.{v}"), weight_shape);
}

fn add_lstm(out: &mut BTreeMap<String, Vec<u64>>, prefix: &str) {
    for layer in 0..2 {
        insert(out, format!("{prefix}.weight_ih_l{layer}"), &[2048, 512]);
        insert(out, format!("{prefix}.weight_hh_l{layer}"), &[2048, 512]);
        insert(out, format!("{prefix}.bias_ih_l{layer}"), &[2048]);
        insert(out, format!("{prefix}.bias_hh_l{layer}"), &[2048]);
    }
}

fn insert(out: &mut BTreeMap<String, Vec<u64>>, name: String, shape: &[u64]) {
    assert!(out.insert(name.clone(), shape.to_vec()).is_none(), "{name}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_manifest_has_four_complete_variants() {
        let manifest = expected_manifest();
        assert_eq!(manifest.len(), TENSOR_COUNT);
        let counts = [
            ("generator_base.", 101),
            ("generator_streaming.", 101),
            ("detector_base.", 54),
            ("detector_streaming.", 54),
        ];
        for (prefix, expected) in counts {
            assert_eq!(
                manifest
                    .keys()
                    .filter(|name| name.starts_with(prefix))
                    .count(),
                expected,
                "{prefix}"
            );
        }
    }

    #[test]
    fn manifest_pins_public_weight_norm_spellings_and_shapes() {
        let manifest = expected_manifest();
        assert_eq!(
            manifest["generator_base.decoder.model.3.convtr.convtr.weight_v"],
            vec![512, 256, 16]
        );
        assert_eq!(
            manifest["detector_base.detector.0.reverse_convolution.weight"],
            vec![128, 32, 320]
        );
        assert_eq!(
            manifest["detector_streaming.detector.0.model.12.conv.conv.parametrizations.weight.original1"],
            vec![512, 256, 16]
        );
        assert!(
            !manifest.contains_key("detector_streaming.detector.0.model.12.conv.conv.weight_v")
        );
        assert_eq!(
            manifest["generator_streaming.msg_processor.msg_processor.weight"],
            vec![32, 128]
        );
    }

    #[test]
    fn incomplete_checkpoint_is_rejected_before_writing() {
        let payload = 0.0f32.to_le_bytes();
        let header = r#"{"generator_base.msg_processor.msg_processor.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&payload);
        let file = SafetensorsFile::parse(bytes).expect("synthetic safetensors");
        let error = validate_input_manifest(&file).expect_err("must reject 1/310 tensors");
        assert!(format!("{error}").contains("expected exactly 310"));
    }

    #[test]
    fn license_override_cannot_relabel_the_pinned_weights() {
        assert_eq!(require_official_license(None).unwrap(), "mit");
        assert_eq!(require_official_license(Some(" MIT ")).unwrap(), "mit");
        let error = require_official_license(Some("apache-2.0")).unwrap_err();
        assert!(format!("{error}").contains("conflicts with the pinned official MIT"));
    }
}
