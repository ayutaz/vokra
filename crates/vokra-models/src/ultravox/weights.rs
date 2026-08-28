//! Exact mmap descriptors for the public Ultravox audio-only checkpoint.

use std::sync::Arc;

use vokra_core::Result;
use vokra_core::gguf::{GgufFile, GgufTensorInfo};

use crate::mapped_weights::{MappedModel, mapped_info};

use super::UltravoxAudioConfig;

const MAPPED: MappedModel = MappedModel {
    name: "ultravox",
    resident_entry: "UltravoxAudioTower::open_mapped after re-converting the official dense BF16 checkpoint",
};

pub(super) struct UltravoxMappedDescriptors {
    file: Arc<GgufFile>,
    config: UltravoxAudioConfig,
    pub(super) stem: StemDescriptors,
    pub(super) layers: Vec<AudioLayerDescriptors>,
    pub(super) projector: ProjectorDescriptors,
}

impl UltravoxMappedDescriptors {
    pub(super) fn bind(file: Arc<GgufFile>, config: UltravoxAudioConfig) -> Result<Self> {
        let d = config.hidden_size;
        let ff = config.ffn_dim;
        let stem = StemDescriptors {
            conv1_weight: exact_info(&file, "audio_tower.conv1.weight", &[d, config.n_mels, 3])?,
            conv1_bias: exact_info(&file, "audio_tower.conv1.bias", &[d])?,
            conv2_weight: exact_info(&file, "audio_tower.conv2.weight", &[d, d, 3])?,
            conv2_bias: exact_info(&file, "audio_tower.conv2.bias", &[d])?,
            positions: exact_info(
                &file,
                "audio_tower.embed_positions.weight",
                &[config.max_mel_frames / 2, d],
            )?,
            final_norm_weight: exact_info(&file, "audio_tower.layer_norm.weight", &[d])?,
            final_norm_bias: exact_info(&file, "audio_tower.layer_norm.bias", &[d])?,
        };

        let mut layers = Vec::with_capacity(config.n_layer);
        for layer in 0..config.n_layer {
            let prefix = format!("audio_tower.layers.{layer}");
            layers.push(AudioLayerDescriptors {
                self_attn_norm_weight: exact_info(
                    &file,
                    &format!("{prefix}.self_attn_layer_norm.weight"),
                    &[d],
                )?,
                self_attn_norm_bias: exact_info(
                    &file,
                    &format!("{prefix}.self_attn_layer_norm.bias"),
                    &[d],
                )?,
                q: linear(&file, &format!("{prefix}.self_attn.q_proj"), d, d, true)?,
                k: linear(&file, &format!("{prefix}.self_attn.k_proj"), d, d, false)?,
                v: linear(&file, &format!("{prefix}.self_attn.v_proj"), d, d, true)?,
                out: linear(&file, &format!("{prefix}.self_attn.out_proj"), d, d, true)?,
                final_norm_weight: exact_info(
                    &file,
                    &format!("{prefix}.final_layer_norm.weight"),
                    &[d],
                )?,
                final_norm_bias: exact_info(
                    &file,
                    &format!("{prefix}.final_layer_norm.bias"),
                    &[d],
                )?,
                fc1: linear(&file, &format!("{prefix}.fc1"), d, ff, true)?,
                fc2: linear(&file, &format!("{prefix}.fc2"), ff, d, true)?,
            });
        }

        let projector = ProjectorDescriptors {
            norm_pre: exact_info(
                &file,
                "multi_modal_projector.ln_pre.weight",
                &[config.stacked_size],
            )?,
            linear_1: exact_info(
                &file,
                "multi_modal_projector.linear_1.weight",
                &[config.projector_packed_size, config.stacked_size],
            )?,
            norm_mid: exact_info(
                &file,
                "multi_modal_projector.ln_mid.weight",
                &[config.text_hidden_size],
            )?,
            linear_2: exact_info(
                &file,
                "multi_modal_projector.linear_2.weight",
                &[config.text_hidden_size, config.text_hidden_size],
            )?,
        };

        Ok(Self {
            file,
            config,
            stem,
            layers,
            projector,
        })
    }

    pub(super) fn file(&self) -> &GgufFile {
        &self.file
    }

    pub(super) const fn config(&self) -> UltravoxAudioConfig {
        self.config
    }

    pub(super) const fn mapped_model(&self) -> MappedModel {
        MAPPED
    }
}

pub(super) struct StemDescriptors {
    pub(super) conv1_weight: GgufTensorInfo,
    pub(super) conv1_bias: GgufTensorInfo,
    pub(super) conv2_weight: GgufTensorInfo,
    pub(super) conv2_bias: GgufTensorInfo,
    pub(super) positions: GgufTensorInfo,
    pub(super) final_norm_weight: GgufTensorInfo,
    pub(super) final_norm_bias: GgufTensorInfo,
}

pub(super) struct LinearDescriptors {
    pub(super) weight: GgufTensorInfo,
    pub(super) bias: Option<GgufTensorInfo>,
}

pub(super) struct AudioLayerDescriptors {
    pub(super) self_attn_norm_weight: GgufTensorInfo,
    pub(super) self_attn_norm_bias: GgufTensorInfo,
    pub(super) q: LinearDescriptors,
    pub(super) k: LinearDescriptors,
    pub(super) v: LinearDescriptors,
    pub(super) out: LinearDescriptors,
    pub(super) final_norm_weight: GgufTensorInfo,
    pub(super) final_norm_bias: GgufTensorInfo,
    pub(super) fc1: LinearDescriptors,
    pub(super) fc2: LinearDescriptors,
}

pub(super) struct ProjectorDescriptors {
    pub(super) norm_pre: GgufTensorInfo,
    pub(super) linear_1: GgufTensorInfo,
    pub(super) norm_mid: GgufTensorInfo,
    pub(super) linear_2: GgufTensorInfo,
}

fn linear(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    bias: bool,
) -> Result<LinearDescriptors> {
    Ok(LinearDescriptors {
        weight: exact_info(file, &format!("{prefix}.weight"), &[output, input])?,
        bias: bias
            .then(|| exact_info(file, &format!("{prefix}.bias"), &[output]))
            .transpose()?,
    })
}

fn exact_info(file: &GgufFile, name: &str, shape: &[usize]) -> Result<GgufTensorInfo> {
    let elements = shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .expect("fixed Ultravox dimensions fit usize");
    let info = mapped_info(file, name, elements, MAPPED)?;
    let actual: Vec<usize> = info
        .dimensions
        .iter()
        .map(|&dimension| dimension as usize)
        .collect();
    if actual != shape {
        return Err(vokra_core::VokraError::ModelLoad(format!(
            "ultravox: tensor `{name}` shape {actual:?}, expected {shape:?}"
        )));
    }
    Ok(info)
}

#[cfg(test)]
pub(super) fn tensor_contract(config: UltravoxAudioConfig) -> Vec<(String, Vec<u64>)> {
    let d = config.hidden_size as u64;
    let ff = config.ffn_dim as u64;
    let mut tensors = Vec::with_capacity(7 + config.n_layer * 15 + 4);
    tensors.extend([
        ("audio_tower.conv1.bias".to_owned(), vec![d]),
        (
            "audio_tower.conv1.weight".to_owned(),
            vec![d, config.n_mels as u64, 3],
        ),
        ("audio_tower.conv2.bias".to_owned(), vec![d]),
        ("audio_tower.conv2.weight".to_owned(), vec![d, d, 3]),
        (
            "audio_tower.embed_positions.weight".to_owned(),
            vec![(config.max_mel_frames / 2) as u64, d],
        ),
        ("audio_tower.layer_norm.bias".to_owned(), vec![d]),
        ("audio_tower.layer_norm.weight".to_owned(), vec![d]),
    ]);
    for layer in 0..config.n_layer {
        let prefix = format!("audio_tower.layers.{layer}");
        tensors.extend([
            (format!("{prefix}.fc1.bias"), vec![ff]),
            (format!("{prefix}.fc1.weight"), vec![ff, d]),
            (format!("{prefix}.fc2.bias"), vec![d]),
            (format!("{prefix}.fc2.weight"), vec![d, ff]),
            (format!("{prefix}.final_layer_norm.bias"), vec![d]),
            (format!("{prefix}.final_layer_norm.weight"), vec![d]),
            (format!("{prefix}.self_attn.k_proj.weight"), vec![d, d]),
            (format!("{prefix}.self_attn.out_proj.bias"), vec![d]),
            (format!("{prefix}.self_attn.out_proj.weight"), vec![d, d]),
            (format!("{prefix}.self_attn.q_proj.bias"), vec![d]),
            (format!("{prefix}.self_attn.q_proj.weight"), vec![d, d]),
            (format!("{prefix}.self_attn.v_proj.bias"), vec![d]),
            (format!("{prefix}.self_attn.v_proj.weight"), vec![d, d]),
            (format!("{prefix}.self_attn_layer_norm.bias"), vec![d]),
            (format!("{prefix}.self_attn_layer_norm.weight"), vec![d]),
        ]);
    }
    tensors.extend([
        (
            "multi_modal_projector.linear_1.weight".to_owned(),
            vec![
                config.projector_packed_size as u64,
                config.stacked_size as u64,
            ],
        ),
        (
            "multi_modal_projector.linear_2.weight".to_owned(),
            vec![
                config.text_hidden_size as u64,
                config.text_hidden_size as u64,
            ],
        ),
        (
            "multi_modal_projector.ln_mid.weight".to_owned(),
            vec![config.text_hidden_size as u64],
        ),
        (
            "multi_modal_projector.ln_pre.weight".to_owned(),
            vec![config.stacked_size as u64],
        ),
    ]);
    tensors
}
