//! Structured mmap descriptors for the exact Qwen3-ASR release manifests.
//!
//! Binding is header-only: every name, shape and dense payload type is checked
//! before execution, while tensor bytes remain in the original `GgufFile`
//! mapping. Runtime stages can then widen one convolution or Transformer layer
//! at a time instead of materialising a multi-gigabyte checkpoint.

use std::sync::Arc;

use vokra_core::Result;
use vokra_core::gguf::{GgufFile, GgufTensorInfo};

use crate::mapped_weights::{MappedModel, mapped_info};

use super::{Qwen3AsrConfig, Qwen3AsrTextConfig};

const LABEL: &str = "qwen3_asr";
const AUDIO_FIXED_WIDTH: usize = 7;
const AUDIO_LAYER_WIDTH: usize = 16;
const AUDIO_POST_WIDTH: usize = 6;
const TEXT_FIXED_WIDTH: usize = 2;
const TEXT_LAYER_WIDTH: usize = 11;
const TEXT_POST_WIDTH: usize = 1;

const MAPPED: MappedModel = MappedModel {
    name: LABEL,
    resident_entry: "Qwen3AsrCheckpoint::open_mapped after re-converting the official dense BF16 checkpoint",
};

pub(super) struct Qwen3AsrMappedDescriptors {
    file: Arc<GgufFile>,
    infos: Vec<GgufTensorInfo>,
    config: Qwen3AsrConfig,
}

impl std::fmt::Debug for Qwen3AsrMappedDescriptors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3AsrMappedDescriptors")
            .field("tensor_count", &self.infos.len())
            .field("config", &self.config)
            .finish()
    }
}

impl Qwen3AsrMappedDescriptors {
    pub(super) fn bind(file: Arc<GgufFile>, config: Qwen3AsrConfig) -> Result<Self> {
        let contract = tensor_contract(config);
        let mut infos = Vec::with_capacity(contract.len());
        for (name, shape) in contract {
            let elements = shape
                .iter()
                .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
                .and_then(|count| usize::try_from(count).ok())
                .expect("strict fixed Qwen3-ASR tensor shapes fit usize");
            infos.push(mapped_info(&file, &name, elements, MAPPED)?);
        }
        Ok(Self {
            file,
            infos,
            config,
        })
    }

    pub(super) fn file(&self) -> &GgufFile {
        &self.file
    }

    fn info(&self, index: usize) -> &GgufTensorInfo {
        &self.infos[index]
    }

    pub(super) const fn config(&self) -> Qwen3AsrConfig {
        self.config
    }

    pub(super) fn convolution(&self, index: usize) -> Qwen3AsrConvDescriptors<'_> {
        debug_assert!((1..=3).contains(&index));
        let start = (index - 1) * 2;
        Qwen3AsrConvDescriptors {
            weight: self.info(start),
            bias: self.info(start + 1),
        }
    }

    pub(super) fn conv_out(&self) -> &GgufTensorInfo {
        self.info(6)
    }

    pub(super) fn audio_layer(&self, layer: usize) -> Qwen3AsrAudioLayerDescriptors<'_> {
        debug_assert!(layer < self.config.audio.n_layer as usize);
        let start = AUDIO_FIXED_WIDTH + layer * AUDIO_LAYER_WIDTH;
        Qwen3AsrAudioLayerDescriptors {
            self_attn_norm_weight: self.info(start),
            self_attn_norm_bias: self.info(start + 1),
            q_weight: self.info(start + 2),
            q_bias: self.info(start + 3),
            k_weight: self.info(start + 4),
            k_bias: self.info(start + 5),
            v_weight: self.info(start + 6),
            v_bias: self.info(start + 7),
            out_weight: self.info(start + 8),
            out_bias: self.info(start + 9),
            final_norm_weight: self.info(start + 10),
            final_norm_bias: self.info(start + 11),
            fc1_weight: self.info(start + 12),
            fc1_bias: self.info(start + 13),
            fc2_weight: self.info(start + 14),
            fc2_bias: self.info(start + 15),
        }
    }

    pub(super) fn audio_post(&self) -> Qwen3AsrAudioPostDescriptors<'_> {
        let start = AUDIO_FIXED_WIDTH + self.config.audio.n_layer as usize * AUDIO_LAYER_WIDTH;
        Qwen3AsrAudioPostDescriptors {
            norm_weight: self.info(start),
            norm_bias: self.info(start + 1),
            proj1_weight: self.info(start + 2),
            proj1_bias: self.info(start + 3),
            proj2_weight: self.info(start + 4),
            proj2_bias: self.info(start + 5),
        }
    }

    fn text_start(&self) -> usize {
        AUDIO_FIXED_WIDTH
            + self.config.audio.n_layer as usize * AUDIO_LAYER_WIDTH
            + AUDIO_POST_WIDTH
    }

    pub(super) fn text_embedding(&self) -> &GgufTensorInfo {
        self.info(self.text_start())
    }

    pub(super) fn text_head(&self) -> &GgufTensorInfo {
        self.info(self.text_start() + 1)
    }

    pub(super) fn text_layer(&self, layer: usize) -> Qwen3AsrTextLayerDescriptors<'_> {
        debug_assert!(layer < self.config.text.n_layer as usize);
        let start = self.text_start() + TEXT_FIXED_WIDTH + layer * TEXT_LAYER_WIDTH;
        Qwen3AsrTextLayerDescriptors {
            input_norm: self.info(start),
            q: self.info(start + 1),
            q_norm: self.info(start + 2),
            k: self.info(start + 3),
            k_norm: self.info(start + 4),
            v: self.info(start + 5),
            o: self.info(start + 6),
            ffn_norm: self.info(start + 7),
            gate: self.info(start + 8),
            up: self.info(start + 9),
            down: self.info(start + 10),
        }
    }

    pub(super) fn text_final_norm(&self) -> &GgufTensorInfo {
        let index = self.text_start()
            + TEXT_FIXED_WIDTH
            + self.config.text.n_layer as usize * TEXT_LAYER_WIDTH;
        self.info(index)
    }
}

pub(super) struct Qwen3AsrConvDescriptors<'a> {
    pub(super) weight: &'a GgufTensorInfo,
    pub(super) bias: &'a GgufTensorInfo,
}

pub(super) struct Qwen3AsrAudioLayerDescriptors<'a> {
    pub(super) self_attn_norm_weight: &'a GgufTensorInfo,
    pub(super) self_attn_norm_bias: &'a GgufTensorInfo,
    pub(super) q_weight: &'a GgufTensorInfo,
    pub(super) q_bias: &'a GgufTensorInfo,
    pub(super) k_weight: &'a GgufTensorInfo,
    pub(super) k_bias: &'a GgufTensorInfo,
    pub(super) v_weight: &'a GgufTensorInfo,
    pub(super) v_bias: &'a GgufTensorInfo,
    pub(super) out_weight: &'a GgufTensorInfo,
    pub(super) out_bias: &'a GgufTensorInfo,
    pub(super) final_norm_weight: &'a GgufTensorInfo,
    pub(super) final_norm_bias: &'a GgufTensorInfo,
    pub(super) fc1_weight: &'a GgufTensorInfo,
    pub(super) fc1_bias: &'a GgufTensorInfo,
    pub(super) fc2_weight: &'a GgufTensorInfo,
    pub(super) fc2_bias: &'a GgufTensorInfo,
}

pub(super) struct Qwen3AsrAudioPostDescriptors<'a> {
    pub(super) norm_weight: &'a GgufTensorInfo,
    pub(super) norm_bias: &'a GgufTensorInfo,
    pub(super) proj1_weight: &'a GgufTensorInfo,
    pub(super) proj1_bias: &'a GgufTensorInfo,
    pub(super) proj2_weight: &'a GgufTensorInfo,
    pub(super) proj2_bias: &'a GgufTensorInfo,
}

pub(super) struct Qwen3AsrTextLayerDescriptors<'a> {
    pub(super) input_norm: &'a GgufTensorInfo,
    pub(super) q: &'a GgufTensorInfo,
    pub(super) q_norm: &'a GgufTensorInfo,
    pub(super) k: &'a GgufTensorInfo,
    pub(super) k_norm: &'a GgufTensorInfo,
    pub(super) v: &'a GgufTensorInfo,
    pub(super) o: &'a GgufTensorInfo,
    pub(super) ffn_norm: &'a GgufTensorInfo,
    pub(super) gate: &'a GgufTensorInfo,
    pub(super) up: &'a GgufTensorInfo,
    pub(super) down: &'a GgufTensorInfo,
}

pub(super) fn tensor_contract(config: Qwen3AsrConfig) -> Vec<(String, Vec<u64>)> {
    let audio = config.audio;
    let text = config.text;
    let conv = u64::from(audio.downsample_hidden_size);
    let audio_dim = u64::from(audio.d_model);
    let audio_ffn = u64::from(audio.ffn_dim);
    let hidden = u64::from(text.hidden_size);
    let q_width = u64::from(text.n_head) * u64::from(text.head_dim);
    let kv_width = u64::from(text.n_kv_head) * u64::from(text.head_dim);
    let text_ffn = u64::from(text.ffn_dim);
    let vocab = u64::from(text.vocab_size);
    let expected = AUDIO_FIXED_WIDTH
        + audio.n_layer as usize * AUDIO_LAYER_WIDTH
        + AUDIO_POST_WIDTH
        + TEXT_FIXED_WIDTH
        + text.n_layer as usize * TEXT_LAYER_WIDTH
        + TEXT_POST_WIDTH;
    let mut tensors = Vec::with_capacity(expected);

    for index in 1..=3 {
        let input_channels = if index == 1 { 1 } else { conv };
        tensors.push((
            format!("thinker.audio_tower.conv2d{index}.weight"),
            vec![conv, input_channels, 3, 3],
        ));
        tensors.push((
            format!("thinker.audio_tower.conv2d{index}.bias"),
            vec![conv],
        ));
    }
    tensors.push((
        "thinker.audio_tower.conv_out.weight".to_owned(),
        vec![audio_dim, 7_680],
    ));

    for layer in 0..audio.n_layer {
        let prefix = format!("thinker.audio_tower.layers.{layer}");
        tensors.extend([
            (
                format!("{prefix}.self_attn_layer_norm.weight"),
                vec![audio_dim],
            ),
            (
                format!("{prefix}.self_attn_layer_norm.bias"),
                vec![audio_dim],
            ),
            (
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![audio_dim, audio_dim],
            ),
            (format!("{prefix}.self_attn.q_proj.bias"), vec![audio_dim]),
            (
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![audio_dim, audio_dim],
            ),
            (format!("{prefix}.self_attn.k_proj.bias"), vec![audio_dim]),
            (
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![audio_dim, audio_dim],
            ),
            (format!("{prefix}.self_attn.v_proj.bias"), vec![audio_dim]),
            (
                format!("{prefix}.self_attn.out_proj.weight"),
                vec![audio_dim, audio_dim],
            ),
            (format!("{prefix}.self_attn.out_proj.bias"), vec![audio_dim]),
            (format!("{prefix}.final_layer_norm.weight"), vec![audio_dim]),
            (format!("{prefix}.final_layer_norm.bias"), vec![audio_dim]),
            (format!("{prefix}.fc1.weight"), vec![audio_ffn, audio_dim]),
            (format!("{prefix}.fc1.bias"), vec![audio_ffn]),
            (format!("{prefix}.fc2.weight"), vec![audio_dim, audio_ffn]),
            (format!("{prefix}.fc2.bias"), vec![audio_dim]),
        ]);
    }

    tensors.extend([
        (
            "thinker.audio_tower.ln_post.weight".to_owned(),
            vec![audio_dim],
        ),
        (
            "thinker.audio_tower.ln_post.bias".to_owned(),
            vec![audio_dim],
        ),
        (
            "thinker.audio_tower.proj1.weight".to_owned(),
            vec![audio_dim, audio_dim],
        ),
        ("thinker.audio_tower.proj1.bias".to_owned(), vec![audio_dim]),
        (
            "thinker.audio_tower.proj2.weight".to_owned(),
            vec![u64::from(audio.output_dim), audio_dim],
        ),
        (
            "thinker.audio_tower.proj2.bias".to_owned(),
            vec![u64::from(audio.output_dim)],
        ),
        (
            "thinker.model.embed_tokens.weight".to_owned(),
            vec![vocab, hidden],
        ),
        ("thinker.lm_head.weight".to_owned(), vec![vocab, hidden]),
    ]);

    append_text_layers(&mut tensors, text, hidden, q_width, kv_width, text_ffn);
    tensors.push(("thinker.model.norm.weight".to_owned(), vec![hidden]));
    debug_assert_eq!(tensors.len(), expected);
    tensors
}

fn append_text_layers(
    tensors: &mut Vec<(String, Vec<u64>)>,
    text: Qwen3AsrTextConfig,
    hidden: u64,
    q_width: u64,
    kv_width: u64,
    text_ffn: u64,
) {
    for layer in 0..text.n_layer {
        let prefix = format!("thinker.model.layers.{layer}");
        tensors.extend([
            (format!("{prefix}.input_layernorm.weight"), vec![hidden]),
            (
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![q_width, hidden],
            ),
            (
                format!("{prefix}.self_attn.q_norm.weight"),
                vec![u64::from(text.head_dim)],
            ),
            (
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![kv_width, hidden],
            ),
            (
                format!("{prefix}.self_attn.k_norm.weight"),
                vec![u64::from(text.head_dim)],
            ),
            (
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![kv_width, hidden],
            ),
            (
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![hidden, q_width],
            ),
            (
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![hidden],
            ),
            (
                format!("{prefix}.mlp.gate_proj.weight"),
                vec![text_ffn, hidden],
            ),
            (
                format!("{prefix}.mlp.up_proj.weight"),
                vec![text_ffn, hidden],
            ),
            (
                format!("{prefix}.mlp.down_proj.weight"),
                vec![hidden, text_ffn],
            ),
        ]);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::qwen3_asr::Qwen3AsrVariant;

    #[test]
    fn contracts_cover_every_release_tensor_once() {
        for variant in [Qwen3AsrVariant::B06, Qwen3AsrVariant::B17] {
            let contract = tensor_contract(variant.config());
            assert_eq!(contract.len(), variant.tensor_count());
            let names: BTreeSet<&str> = contract.iter().map(|(name, _)| name.as_str()).collect();
            assert_eq!(names.len(), contract.len());
            assert!(names.contains("thinker.audio_tower.conv2d3.weight"));
            assert!(names.contains("thinker.model.layers.27.self_attn.q_norm.weight"));
            assert!(names.contains("thinker.model.norm.weight"));
        }
    }
}
