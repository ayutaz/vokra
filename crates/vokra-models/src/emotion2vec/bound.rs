use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};
use vokra_ops::{
    ConvLayerAttrs, ConvLayerWeights, Norm, WaveformFrontendAttrs, WaveformFrontendWeights,
};

use crate::align::charsiu::{CharsiuBlock, CharsiuFeatureProjection, CharsiuPosConv};
use crate::strict_checkpoint::load_tensor;

use super::{
    CATEGORY, CHECKPOINT_FILE, CHECKPOINT_SHA256, CONTEXT_LAYERS, CONV_DIM, CONV_KERNEL,
    CONV_STRIDE, EMOTION_CLASS_LABELS, EXTRA_TOKENS, FEATURE_DIM, FFN, GLOBAL_LAYERS, HEADS,
    HIDDEN, LAYER_NORM_EPS, NAME, NUM_CLASSES, POSITION_GROUPS, POSITION_KERNEL, POSITION_LAYERS,
    PREFIX, SAMPLE_RATE, UPSTREAM_HF, UPSTREAM_REVISION,
};

const KEY_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_CHECKPOINT_FILE: &str = "vokra.provenance.checkpoint_file";
const KEY_CHECKPOINT_SHA256: &str = "vokra.provenance.checkpoint_sha256";
const CANONICAL_SOURCE: &str = concat!(
    "emotion2vec/emotion2vec_plus_large@",
    "6c303ba987b86b93193de93e34bb2b077a6bedc4",
    "/model.pt sha256:",
    "be501a01f26fcdc7663a062dff86af839afbaef7c4de32f5e42d7e1ad2784da4"
);

const KEY_SAMPLE_RATE: &str = "vokra.emotion2vec.sample_rate";
const KEY_EMBED_DIM: &str = "vokra.emotion2vec.embed_dim";
const KEY_DEPTH: &str = "vokra.emotion2vec.depth";
const KEY_PRENET_DEPTH: &str = "vokra.emotion2vec.prenet_depth";
const KEY_NUM_HEADS: &str = "vokra.emotion2vec.num_heads";
const KEY_MLP_DIM: &str = "vokra.emotion2vec.mlp_dim";
const KEY_NUM_EXTRA_TOKENS: &str = "vokra.emotion2vec.num_extra_tokens";
const KEY_NUM_CLASSES: &str = "vokra.emotion2vec.num_classes";
const KEY_CONV_POS_DEPTH: &str = "vokra.emotion2vec.conv_pos_depth";
const KEY_CONV_POS_KERNEL: &str = "vokra.emotion2vec.conv_pos_kernel";
const KEY_CONV_POS_GROUPS: &str = "vokra.emotion2vec.conv_pos_groups";
const KEY_LAYER_NORM_EPS: &str = "vokra.emotion2vec.layer_norm_eps";
const KEY_NORMALIZE: &str = "vokra.emotion2vec.normalize";
const KEY_CONV_DIM: &str = "vokra.emotion2vec.conv_dim";
const KEY_CONV_KERNEL: &str = "vokra.emotion2vec.conv_kernel";
const KEY_CONV_STRIDE: &str = "vokra.emotion2vec.conv_stride";
const KEY_CLASS_LABELS: &str = "vokra.emotion2vec.class_labels";

const PROVENANCE_KEYS: &[&str] = &[
    KEY_UPSTREAM_HF,
    KEY_UPSTREAM_REVISION,
    KEY_CHECKPOINT_FILE,
    KEY_CHECKPOINT_SHA256,
];

pub(super) const CONTRACT_KEYS: &[&str] = &[
    KEY_SAMPLE_RATE,
    KEY_EMBED_DIM,
    KEY_DEPTH,
    KEY_PRENET_DEPTH,
    KEY_NUM_HEADS,
    KEY_MLP_DIM,
    KEY_NUM_EXTRA_TOKENS,
    KEY_NUM_CLASSES,
    KEY_CONV_POS_DEPTH,
    KEY_CONV_POS_KERNEL,
    KEY_CONV_POS_GROUPS,
    KEY_LAYER_NORM_EPS,
    KEY_NORMALIZE,
    KEY_CONV_DIM,
    KEY_CONV_KERNEL,
    KEY_CONV_STRIDE,
    KEY_CLASS_LABELS,
];

#[derive(Debug)]
pub(super) struct Emotion2VecWeights {
    pub(super) stem_attrs: WaveformFrontendAttrs,
    pub(super) stem: WaveformFrontendWeights,
    pub(super) projection: CharsiuFeatureProjection,
    pub(super) position: Vec<CharsiuPosConv>,
    pub(super) extra_tokens: Vec<f32>,
    pub(super) alibi_scale: Vec<f32>,
    pub(super) context_norm_gamma: Vec<f32>,
    pub(super) context_norm_beta: Vec<f32>,
    pub(super) context_blocks: Vec<CharsiuBlock>,
    pub(super) global_blocks: Vec<CharsiuBlock>,
    pub(super) head_weight: Vec<f32>,
    pub(super) head_bias: Vec<f32>,
}

impl Emotion2VecWeights {
    pub(super) fn bind(file: &GgufFile) -> Result<Self> {
        let audio = "d2v_model.modality_encoders.AUDIO";
        let stem_attrs = WaveformFrontendAttrs {
            in_channels: 1,
            layers: CONV_DIM
                .iter()
                .zip(CONV_KERNEL)
                .zip(CONV_STRIDE)
                .map(|((&out_channels, kernel), stride)| ConvLayerAttrs {
                    out_channels,
                    kernel,
                    stride,
                })
                .collect(),
            norm: Norm::LayerAll,
            conv_bias: false,
        };
        let mut stem_layers = Vec::with_capacity(CONV_DIM.len());
        let mut in_channels = 1usize;
        for layer in 0..CONV_DIM.len() {
            let prefix = format!("{audio}.local_encoder.conv_layers.{layer}");
            stem_layers.push(ConvLayerWeights {
                conv_w: tensor(
                    file,
                    &format!("{prefix}.0.weight"),
                    &[CONV_DIM[layer], in_channels, CONV_KERNEL[layer]],
                )?,
                conv_b: Vec::new(),
                norm_gamma: Some(tensor(
                    file,
                    &format!("{prefix}.2.1.weight"),
                    &[FEATURE_DIM],
                )?),
                norm_beta: Some(tensor(file, &format!("{prefix}.2.1.bias"), &[FEATURE_DIM])?),
            });
            in_channels = CONV_DIM[layer];
        }
        let stem = WaveformFrontendWeights {
            layers: stem_layers,
        };
        stem.validate(&stem_attrs)?;

        let projection = CharsiuFeatureProjection {
            norm_gamma: Some(tensor(
                file,
                &format!("{audio}.project_features.1.weight"),
                &[FEATURE_DIM],
            )?),
            norm_beta: Some(tensor(
                file,
                &format!("{audio}.project_features.1.bias"),
                &[FEATURE_DIM],
            )?),
            linear_w: tensor(
                file,
                &format!("{audio}.project_features.2.weight"),
                &[HIDDEN, FEATURE_DIM],
            )?,
            linear_b: tensor(file, &format!("{audio}.project_features.2.bias"), &[HIDDEN])?,
        };

        let mut position = Vec::with_capacity(POSITION_LAYERS);
        for layer in 1..=POSITION_LAYERS {
            let prefix = format!("{audio}.relative_positional_encoder.{layer}.0");
            position.push(CharsiuPosConv {
                weight: tensor(
                    file,
                    &format!("{prefix}.weight"),
                    &[HIDDEN, HIDDEN / POSITION_GROUPS, POSITION_KERNEL],
                )?,
                bias: tensor(file, &format!("{prefix}.bias"), &[HIDDEN])?,
            });
        }

        let mut context_blocks = Vec::with_capacity(CONTEXT_LAYERS);
        for layer in 0..CONTEXT_LAYERS {
            context_blocks.push(bind_block(
                file,
                &format!("{audio}.context_encoder.blocks.{layer}"),
            )?);
        }
        let mut global_blocks = Vec::with_capacity(GLOBAL_LAYERS);
        for layer in 0..GLOBAL_LAYERS {
            global_blocks.push(bind_block(file, &format!("d2v_model.blocks.{layer}"))?);
        }

        Ok(Self {
            stem_attrs,
            stem,
            projection,
            position,
            extra_tokens: tensor(
                file,
                &format!("{audio}.extra_tokens"),
                &[1, EXTRA_TOKENS, HIDDEN],
            )?,
            alibi_scale: tensor(file, &format!("{audio}.alibi_scale"), &[HEADS])?,
            context_norm_gamma: tensor(
                file,
                &format!("{audio}.context_encoder.norm.weight"),
                &[HIDDEN],
            )?,
            context_norm_beta: tensor(
                file,
                &format!("{audio}.context_encoder.norm.bias"),
                &[HIDDEN],
            )?,
            context_blocks,
            global_blocks,
            head_weight: tensor(file, "proj.weight", &[NUM_CLASSES, HIDDEN])?,
            head_bias: tensor(file, "proj.bias", &[NUM_CLASSES])?,
        })
    }
}

fn bind_block(file: &GgufFile, prefix: &str) -> Result<CharsiuBlock> {
    let qkv_weight = tensor(
        file,
        &format!("{prefix}.attn.qkv.weight"),
        &[HIDDEN * 3, HIDDEN],
    )?;
    let qkv_bias = tensor(file, &format!("{prefix}.attn.qkv.bias"), &[HIDDEN * 3])?;
    let matrix = HIDDEN * HIDDEN;
    Ok(CharsiuBlock {
        attn_norm_gamma: tensor(file, &format!("{prefix}.norm1.weight"), &[HIDDEN])?,
        attn_norm_beta: tensor(file, &format!("{prefix}.norm1.bias"), &[HIDDEN])?,
        q_w: qkv_weight[..matrix].to_vec(),
        q_b: qkv_bias[..HIDDEN].to_vec(),
        k_w: qkv_weight[matrix..matrix * 2].to_vec(),
        k_b: qkv_bias[HIDDEN..HIDDEN * 2].to_vec(),
        v_w: qkv_weight[matrix * 2..].to_vec(),
        v_b: qkv_bias[HIDDEN * 2..].to_vec(),
        o_w: tensor(
            file,
            &format!("{prefix}.attn.proj.weight"),
            &[HIDDEN, HIDDEN],
        )?,
        o_b: tensor(file, &format!("{prefix}.attn.proj.bias"), &[HIDDEN])?,
        ffn_norm_gamma: tensor(file, &format!("{prefix}.norm2.weight"), &[HIDDEN])?,
        ffn_norm_beta: tensor(file, &format!("{prefix}.norm2.bias"), &[HIDDEN])?,
        fc1_w: tensor(file, &format!("{prefix}.mlp.fc1.weight"), &[FFN, HIDDEN])?,
        fc1_b: tensor(file, &format!("{prefix}.mlp.fc1.bias"), &[FFN])?,
        fc2_w: tensor(file, &format!("{prefix}.mlp.fc2.weight"), &[HIDDEN, FFN])?,
        fc2_b: tensor(file, &format!("{prefix}.mlp.fc2.bias"), &[HIDDEN])?,
    })
}

fn tensor(file: &GgufFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    load_tensor(file, "emotion2vec-plus-large", name, shape)
}

/// Validates exact base provenance and the all-or-nothing additive groups.
/// Returns `true` only for the exact historical public GGUF that predates them.
pub(super) fn validate_metadata(file: &GgufFile) -> Result<bool> {
    require_string(file, KEY_CATEGORY, CATEGORY)?;
    require_string(file, chunks::KEY_PROVENANCE_MODEL_ID, NAME)?;
    let license = required_string(file, chunks::KEY_PROVENANCE_LICENSE)?;
    if !license.eq_ignore_ascii_case("mit") {
        return Err(metadata_error(
            chunks::KEY_PROVENANCE_LICENSE,
            license,
            "mit",
        ));
    }
    let class = required_string(file, chunks::KEY_PROVENANCE_WEIGHT_LICENSE)?;
    if LicenseClass::from_class_str(class) != Some(LicenseClass::Permissive) {
        return Err(metadata_error(
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            class,
            "permissive",
        ));
    }
    let source = required_string(file, chunks::KEY_PROVENANCE_SOURCE)?;
    if source != UPSTREAM_HF && source != CANONICAL_SOURCE {
        return Err(VokraError::ModelLoad(format!(
            "emotion2vec: unsupported `{}`={source:?}; expected historical {UPSTREAM_HF:?} or canonical {CANONICAL_SOURCE:?}",
            chunks::KEY_PROVENANCE_SOURCE
        )));
    }

    let provenance_count = count_present(file, PROVENANCE_KEYS);
    match provenance_count {
        0 => {}
        count if count == PROVENANCE_KEYS.len() => {
            require_string(file, KEY_UPSTREAM_HF, UPSTREAM_HF)?;
            require_string(file, KEY_UPSTREAM_REVISION, UPSTREAM_REVISION)?;
            require_string(file, KEY_CHECKPOINT_FILE, CHECKPOINT_FILE)?;
            require_string(file, KEY_CHECKPOINT_SHA256, CHECKPOINT_SHA256)?;
        }
        count => {
            return Err(VokraError::ModelLoad(format!(
                "emotion2vec: partial immutable provenance group ({count}/{} keys)",
                PROVENANCE_KEYS.len()
            )));
        }
    }

    let contract_count = count_present(file, CONTRACT_KEYS);
    match contract_count {
        0 => Ok(true),
        count if count == CONTRACT_KEYS.len() => {
            validate_contract(file)?;
            Ok(false)
        }
        count => Err(VokraError::ModelLoad(format!(
            "emotion2vec: partial `{PREFIX}.*` metadata ({count}/{} keys); refusing topology repair",
            CONTRACT_KEYS.len()
        ))),
    }
}

fn validate_contract(file: &GgufFile) -> Result<()> {
    require_u64(file, KEY_SAMPLE_RATE, u64::from(SAMPLE_RATE))?;
    require_u64(file, KEY_EMBED_DIM, HIDDEN as u64)?;
    require_u64(file, KEY_DEPTH, GLOBAL_LAYERS as u64)?;
    require_u64(file, KEY_PRENET_DEPTH, CONTEXT_LAYERS as u64)?;
    require_u64(file, KEY_NUM_HEADS, HEADS as u64)?;
    require_u64(file, KEY_MLP_DIM, FFN as u64)?;
    require_u64(file, KEY_NUM_EXTRA_TOKENS, EXTRA_TOKENS as u64)?;
    require_u64(file, KEY_NUM_CLASSES, NUM_CLASSES as u64)?;
    require_u64(file, KEY_CONV_POS_DEPTH, POSITION_LAYERS as u64)?;
    require_u64(file, KEY_CONV_POS_KERNEL, POSITION_KERNEL as u64)?;
    require_u64(file, KEY_CONV_POS_GROUPS, POSITION_GROUPS as u64)?;
    require_f64(file, KEY_LAYER_NORM_EPS, f64::from(LAYER_NORM_EPS))?;
    require_bool(file, KEY_NORMALIZE, true)?;
    require_u32_array(file, KEY_CONV_DIM, &CONV_DIM)?;
    require_u32_array(file, KEY_CONV_KERNEL, &CONV_KERNEL)?;
    require_u32_array(file, KEY_CONV_STRIDE, &CONV_STRIDE)?;
    require_string_array(file, KEY_CLASS_LABELS, &EMOTION_CLASS_LABELS)?;
    Ok(())
}

fn count_present(file: &GgufFile, keys: &[&str]) -> usize {
    keys.iter().filter(|key| file.get(key).is_some()).count()
}

fn required_string<'a>(file: &'a GgufFile, key: &str) -> Result<&'a str> {
    file.get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| VokraError::ModelLoad(format!("emotion2vec: missing/non-string `{key}`")))
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = required_string(file, key)?;
    if actual != expected {
        return Err(metadata_error(key, actual, expected));
    }
    Ok(())
}

fn require_u64(file: &GgufFile, key: &str, expected: u64) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(GgufMetadataValue::as_u64)
        .ok_or_else(|| VokraError::ModelLoad(format!("emotion2vec: missing/non-u32 `{key}`")))?;
    if actual != expected {
        return Err(metadata_error(key, actual, expected));
    }
    Ok(())
}

fn require_f64(file: &GgufFile, key: &str, expected: f64) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(GgufMetadataValue::as_f64)
        .ok_or_else(|| VokraError::ModelLoad(format!("emotion2vec: missing/non-f32 `{key}`")))?;
    if actual.to_bits() != expected.to_bits() {
        return Err(metadata_error(key, actual, expected));
    }
    Ok(())
}

fn require_bool(file: &GgufFile, key: &str, expected: bool) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(GgufMetadataValue::as_bool)
        .ok_or_else(|| VokraError::ModelLoad(format!("emotion2vec: missing/non-bool `{key}`")))?;
    if actual != expected {
        return Err(metadata_error(key, actual, expected));
    }
    Ok(())
}

fn require_u32_array(file: &GgufFile, key: &str, expected: &[usize]) -> Result<()> {
    let array = file
        .get(key)
        .and_then(GgufMetadataValue::as_array)
        .ok_or_else(|| VokraError::ModelLoad(format!("emotion2vec: missing/non-array `{key}`")))?;
    if array.values.len() != expected.len() {
        return Err(metadata_error(key, array.values.len(), expected.len()));
    }
    for (index, (actual, expected)) in array.values.iter().zip(expected).enumerate() {
        let actual = actual.as_u64().ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "emotion2vec: `{key}` element {index} is not an unsigned integer"
            ))
        })?;
        if actual != *expected as u64 {
            return Err(VokraError::ModelLoad(format!(
                "emotion2vec: `{key}` element {index}={actual}, expected {expected}"
            )));
        }
    }
    Ok(())
}

fn require_string_array(file: &GgufFile, key: &str, expected: &[&str]) -> Result<()> {
    let array = file
        .get(key)
        .and_then(GgufMetadataValue::as_array)
        .ok_or_else(|| VokraError::ModelLoad(format!("emotion2vec: missing/non-array `{key}`")))?;
    if array.values.len() != expected.len() {
        return Err(metadata_error(key, array.values.len(), expected.len()));
    }
    for (index, (actual, expected)) in array.values.iter().zip(expected).enumerate() {
        let actual = actual.as_str().ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "emotion2vec: `{key}` element {index} is not a string"
            ))
        })?;
        if actual != *expected {
            return Err(VokraError::ModelLoad(format!(
                "emotion2vec: `{key}` element {index}={actual:?}, expected {expected:?}"
            )));
        }
    }
    Ok(())
}

fn metadata_error(
    key: &str,
    actual: impl std::fmt::Debug,
    expected: impl std::fmt::Debug,
) -> VokraError {
    VokraError::ModelLoad(format!(
        "emotion2vec: unsupported `{key}`={actual:?}; expected {expected:?}"
    ))
}
