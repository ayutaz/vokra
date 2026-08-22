//! Canonical `pipecat-ai/smart-turn-v2` checkpoint → GGUF.
//!
//! The release is a raw-waveform Wav2Vec2-base encoder, attention pooling,
//! and a `768 → 256 → 64 → 1` endpoint classifier.  This converter accepts
//! only the pinned 223-tensor F32 manifest.  It folds the parametrized
//! positional-convolution weight norm, consumes the eval-only masking vector
//! without emitting it, and rejects every missing, extra, renamed, or reshaped
//! tensor.  No ONNX graph or Python dependency enters the runtime.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "smart_turn";
pub const NAME: &str = "smart-turn-v2";
pub const CATEGORY: &str = "vad";
pub const UPSTREAM_HF: &str = "pipecat-ai/smart-turn-v2";
pub const DEFAULT_LICENSE_SPDX: &str = "bsd-2-clause";

pub const REVISION: &str = "3267e96b50db03fe030b9869eb35f849a5eea1fa";
pub const CHECKPOINT_SHA256: &str =
    "0c4429a3f55d42d055e08903eb961f6ec4021c9e35d489007f3dc4981b6b028b";
pub const CONFIG_SHA256: &str = "31aa20aebdee3f961077a9482f909efce4d46199aabd848def1c4d9456e2c716";
pub const PREPROCESSOR_CONFIG_SHA256: &str =
    "617bd0950f8cc9ac4062e8c73a7be60305ca5790a243df55fa6f44fb671b55b1";
pub const REFERENCE_REVISION: &str = "c560a748b4213ca8db6f43a5d165d91aaa124a52";

const HIDDEN: usize = 768;
const FEATURE_DIM: usize = 512;
const FFN: usize = 3072;
const N_LAYER: usize = 12;
const N_HEAD: usize = 12;
const POS_KERNEL: usize = 128;
const POS_GROUPS: usize = 16;
const SAMPLE_RATE: u32 = 16_000;
const MAX_INPUT_SAMPLES: u32 = 16_000 * 16;
const SOURCE_TENSOR_COUNT: usize = 223;

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SmartTurnReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_smart_turn_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<SmartTurnReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;
    if st.tensors().len() != SOURCE_TENSOR_COUNT {
        return Err(ConvertError::Parse(format!(
            "smart-turn-v2: source manifest has {} tensors, expected exactly {SOURCE_TENSOR_COUNT}",
            st.tensors().len()
        )));
    }
    let mut consumed = vec![false; st.tensors().len()];

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    b.add_string("vokra.smart_turn.revision", REVISION);
    b.add_string("vokra.smart_turn.checkpoint_sha256", CHECKPOINT_SHA256);
    b.add_string("vokra.smart_turn.config_sha256", CONFIG_SHA256);
    b.add_string(
        "vokra.smart_turn.preprocessor_config_sha256",
        PREPROCESSOR_CONFIG_SHA256,
    );
    b.add_string("vokra.smart_turn.reference_revision", REFERENCE_REVISION);
    b.add_u32("vokra.smart_turn.sample_rate", SAMPLE_RATE);
    b.add_u32("vokra.smart_turn.max_input_samples", MAX_INPUT_SAMPLES);
    b.add_f32("vokra.smart_turn.max_segment_seconds", 16.0);
    b.add_u32("vokra.smart_turn.hidden_size", HIDDEN as u32);
    b.add_u32("vokra.smart_turn.feature_dim", FEATURE_DIM as u32);
    b.add_u32("vokra.smart_turn.ffn_dim", FFN as u32);
    b.add_u32("vokra.smart_turn.n_layer", N_LAYER as u32);
    b.add_u32("vokra.smart_turn.n_head", N_HEAD as u32);
    b.add_u32("vokra.smart_turn.pos_conv_kernel", POS_KERNEL as u32);
    b.add_u32("vokra.smart_turn.pos_conv_groups", POS_GROUPS as u32);
    b.add_f32("vokra.smart_turn.layer_norm_eps", 1e-5);
    b.add_f32("vokra.smart_turn.normalization_eps", 1e-7);
    b.add_f32("vokra.smart_turn.completion_threshold", 0.5);

    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some("pipecat-ai/smart-turn-v2 Wav2Vec2 endpoint classifier"),
    );

    let mut written = 0usize;
    let mut emit = |name: &str, dims: &[usize], data: &[f32]| -> Result<(), ConvertError> {
        b.add_tensor(
            name,
            GgmlType::F32,
            dims.iter().map(|&d| d as u64).collect(),
            data.iter().flat_map(|x| x.to_le_bytes()).collect(),
        )?;
        written += 1;
        Ok(())
    };
    let mut take = |name: &str, dims: &[usize]| -> Result<Vec<f32>, ConvertError> {
        let (idx, info) = st
            .tensors()
            .iter()
            .enumerate()
            .find(|(_, t)| t.name == name)
            .ok_or_else(|| {
                ConvertError::Parse(format!("smart-turn-v2: missing tensor `{name}`"))
            })?;
        if consumed[idx] {
            return Err(ConvertError::Parse(format!(
                "smart-turn-v2: tensor `{name}` consumed twice"
            )));
        }
        let expected: Vec<u64> = dims.iter().map(|&d| d as u64).collect();
        if info.shape != expected {
            return Err(ConvertError::Parse(format!(
                "smart-turn-v2: tensor `{name}` has shape {:?}, expected {expected:?}",
                info.shape
            )));
        }
        if info.dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "smart-turn-v2: tensor `{name}` must be F32, got {:?}",
                info.dtype
            )));
        }
        consumed[idx] = true;
        st.tensor_f32(name)
            .map_err(|e| ConvertError::Parse(format!("smart-turn-v2: reading `{name}`: {e}")))
    };

    let kernels = [10usize, 3, 3, 3, 3, 2, 2];
    let mut in_channels = 1usize;
    for (i, &kernel) in kernels.iter().enumerate() {
        let name = format!("wav2vec2.feature_extractor.conv_layers.{i}.conv.weight");
        let data = take(&name, &[FEATURE_DIM, in_channels, kernel])?;
        emit(&name, &[FEATURE_DIM, in_channels, kernel], &data)?;
        if i == 0 {
            for suffix in ["weight", "bias"] {
                let name = format!("wav2vec2.feature_extractor.conv_layers.0.layer_norm.{suffix}");
                let data = take(&name, &[FEATURE_DIM])?;
                emit(&name, &[FEATURE_DIM], &data)?;
            }
        }
        in_channels = FEATURE_DIM;
    }

    for (name, dims) in [
        (
            "wav2vec2.feature_projection.layer_norm.weight",
            vec![FEATURE_DIM],
        ),
        (
            "wav2vec2.feature_projection.layer_norm.bias",
            vec![FEATURE_DIM],
        ),
        (
            "wav2vec2.feature_projection.projection.weight",
            vec![HIDDEN, FEATURE_DIM],
        ),
        ("wav2vec2.feature_projection.projection.bias", vec![HIDDEN]),
    ] {
        let data = take(name, &dims)?;
        emit(name, &dims, &data)?;
    }

    let g_name = "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0";
    let v_name = "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1";
    let g = take(g_name, &[1, 1, POS_KERNEL])?;
    let v = take(v_name, &[HIDDEN, HIDDEN / POS_GROUPS, POS_KERNEL])?;
    let folded = fold_weight_norm_dim2(&g, &v)?;
    emit(
        "smart_turn.pos_conv.weight",
        &[HIDDEN, HIDDEN / POS_GROUPS, POS_KERNEL],
        &folded,
    )?;
    let pos_bias = take("wav2vec2.encoder.pos_conv_embed.conv.bias", &[HIDDEN])?;
    emit("smart_turn.pos_conv.bias", &[HIDDEN], &pos_bias)?;

    for suffix in ["weight", "bias"] {
        let name = format!("wav2vec2.encoder.layer_norm.{suffix}");
        let data = take(&name, &[HIDDEN])?;
        emit(&name, &[HIDDEN], &data)?;
    }
    for i in 0..N_LAYER {
        let p = format!("wav2vec2.encoder.layers.{i}");
        for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            for suffix in ["weight", "bias"] {
                let name = format!("{p}.attention.{projection}.{suffix}");
                let dims = if suffix == "weight" {
                    vec![HIDDEN, HIDDEN]
                } else {
                    vec![HIDDEN]
                };
                let data = take(&name, &dims)?;
                emit(&name, &dims, &data)?;
            }
        }
        for norm in ["layer_norm", "final_layer_norm"] {
            for suffix in ["weight", "bias"] {
                let name = format!("{p}.{norm}.{suffix}");
                let data = take(&name, &[HIDDEN])?;
                emit(&name, &[HIDDEN], &data)?;
            }
        }
        for (dense, dims) in [
            ("intermediate_dense", [FFN, HIDDEN]),
            ("output_dense", [HIDDEN, FFN]),
        ] {
            let weight = format!("{p}.feed_forward.{dense}.weight");
            let data = take(&weight, &dims)?;
            emit(&weight, &dims, &data)?;
            let bias = format!("{p}.feed_forward.{dense}.bias");
            let data = take(&bias, &[dims[0]])?;
            emit(&bias, &[dims[0]], &data)?;
        }
    }

    for (name, dims) in [
        ("pool_attention.0.weight", vec![256, HIDDEN]),
        ("pool_attention.0.bias", vec![256]),
        ("pool_attention.2.weight", vec![1, 256]),
        ("pool_attention.2.bias", vec![1]),
        ("classifier.0.weight", vec![256, HIDDEN]),
        ("classifier.0.bias", vec![256]),
        ("classifier.1.weight", vec![256]),
        ("classifier.1.bias", vec![256]),
        ("classifier.4.weight", vec![64, 256]),
        ("classifier.4.bias", vec![64]),
        ("classifier.6.weight", vec![1, 64]),
        ("classifier.6.bias", vec![1]),
    ] {
        let data = take(name, &dims)?;
        emit(name, &dims, &data)?;
    }

    // Wav2Vec2Model uses this vector only for training-time SpecAugment.
    let _ = take("wav2vec2.masked_spec_embed", &[HIDDEN])?;
    let leftovers: Vec<&str> = consumed
        .iter()
        .enumerate()
        .filter(|&(_, used)| !*used)
        .map(|(i, _)| st.tensors()[i].name.as_str())
        .collect();
    if !leftovers.is_empty() {
        return Err(ConvertError::Parse(format!(
            "smart-turn-v2: {} unrecognized tensor(s); refusing a partial conversion: {:?}",
            leftovers.len(),
            &leftovers[..leftovers.len().min(8)]
        )));
    }

    std::fs::write(
        output,
        b.to_bytes()
            .map_err(|e| ConvertError::Gguf(e.to_string()))?,
    )?;
    Ok(SmartTurnReport {
        read: consumed.len(),
        written,
        skipped_non_float: 0,
        bf16_passthrough: 0,
    })
}

fn fold_weight_norm_dim2(g: &[f32], v: &[f32]) -> Result<Vec<f32>, ConvertError> {
    let mut weight = vec![0.0f32; HIDDEN * (HIDDEN / POS_GROUPS) * POS_KERNEL];
    for k in 0..POS_KERNEL {
        let mut squared = 0.0f64;
        for o in 0..HIDDEN {
            for i in 0..HIDDEN / POS_GROUPS {
                let value = f64::from(v[(o * (HIDDEN / POS_GROUPS) + i) * POS_KERNEL + k]);
                squared += value * value;
            }
        }
        let norm = squared.sqrt();
        if norm == 0.0 {
            return Err(ConvertError::Parse(format!(
                "smart-turn-v2: positional weight tap {k} has zero norm"
            )));
        }
        let scale = (f64::from(g[k]) / norm) as f32;
        for o in 0..HIDDEN {
            for i in 0..HIDDEN / POS_GROUPS {
                let idx = (o * (HIDDEN / POS_GROUPS) + i) * POS_KERNEL + k;
                weight[idx] = v[idx] * scale;
            }
        }
    }
    Ok(weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_contract_is_pinned() {
        assert_eq!(SOURCE_TENSOR_COUNT, 223);
        assert_eq!(MAX_INPUT_SAMPLES, 256_000);
        assert_eq!(HIDDEN / POS_GROUPS, 48);
        assert_eq!(REVISION.len(), 40);
        assert_eq!(CHECKPOINT_SHA256.len(), 64);
    }
}
