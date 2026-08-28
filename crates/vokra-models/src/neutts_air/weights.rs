//! Structured mmap descriptors for the exact public NeuTTS Air GGUF.
//!
//! The 1.49 GB BF16 payload stays in the original mapping.  Runtime execution
//! widens one Qwen2 layer at a time into reusable scratch storage, rather than
//! retaining a second f32 copy of the complete checkpoint.

use std::sync::Arc;

use vokra_core::Result;
use vokra_core::gguf::{GgufFile, GgufTensorInfo};

use crate::mapped_weights::{MappedModel, mapped_info};

use super::NeuTtsAirConfig;

const FIXED_WIDTH: usize = 2;
const LAYER_WIDTH: usize = 12;
const POST_WIDTH: usize = 1;

const MAPPED: MappedModel = MappedModel {
    name: "neutts_air",
    resident_entry: "NeuTtsAir::open_mapped with a dense F32/F16/BF16 Vokra checkpoint",
};

pub(super) struct NeuTtsAirMappedDescriptors {
    file: Arc<GgufFile>,
    infos: Vec<GgufTensorInfo>,
    config: NeuTtsAirConfig,
}

impl std::fmt::Debug for NeuTtsAirMappedDescriptors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuTtsAirMappedDescriptors")
            .field("tensor_count", &self.infos.len())
            .field("config", &self.config)
            .finish()
    }
}

impl NeuTtsAirMappedDescriptors {
    pub(super) fn bind(file: Arc<GgufFile>, config: NeuTtsAirConfig) -> Result<Self> {
        let contract = tensor_contract(config);
        let mut infos = Vec::with_capacity(contract.len());
        for (name, shape) in contract {
            let elements = shape
                .iter()
                .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
                .and_then(|count| usize::try_from(count).ok())
                .expect("fixed NeuTTS Air tensor shapes fit usize");
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

    pub(super) const fn config(&self) -> NeuTtsAirConfig {
        self.config
    }

    pub(super) const fn mapped_model(&self) -> MappedModel {
        MAPPED
    }

    fn info(&self, index: usize) -> &GgufTensorInfo {
        &self.infos[index]
    }

    pub(super) fn embedding(&self) -> &GgufTensorInfo {
        self.info(0)
    }

    pub(super) fn head(&self) -> &GgufTensorInfo {
        self.info(1)
    }

    pub(super) fn layer(&self, layer: usize) -> NeuTtsAirLayerDescriptors<'_> {
        debug_assert!(layer < self.config.n_layer as usize);
        let start = FIXED_WIDTH + layer * LAYER_WIDTH;
        NeuTtsAirLayerDescriptors {
            input_norm: self.info(start),
            q_weight: self.info(start + 1),
            q_bias: self.info(start + 2),
            k_weight: self.info(start + 3),
            k_bias: self.info(start + 4),
            v_weight: self.info(start + 5),
            v_bias: self.info(start + 6),
            o_weight: self.info(start + 7),
            ffn_norm: self.info(start + 8),
            gate: self.info(start + 9),
            up: self.info(start + 10),
            down: self.info(start + 11),
        }
    }

    pub(super) fn final_norm(&self) -> &GgufTensorInfo {
        self.info(FIXED_WIDTH + self.config.n_layer as usize * LAYER_WIDTH)
    }
}

pub(super) struct NeuTtsAirLayerDescriptors<'a> {
    pub(super) input_norm: &'a GgufTensorInfo,
    pub(super) q_weight: &'a GgufTensorInfo,
    pub(super) q_bias: &'a GgufTensorInfo,
    pub(super) k_weight: &'a GgufTensorInfo,
    pub(super) k_bias: &'a GgufTensorInfo,
    pub(super) v_weight: &'a GgufTensorInfo,
    pub(super) v_bias: &'a GgufTensorInfo,
    pub(super) o_weight: &'a GgufTensorInfo,
    pub(super) ffn_norm: &'a GgufTensorInfo,
    pub(super) gate: &'a GgufTensorInfo,
    pub(super) up: &'a GgufTensorInfo,
    pub(super) down: &'a GgufTensorInfo,
}

pub(super) fn tensor_contract(config: NeuTtsAirConfig) -> Vec<(String, Vec<u64>)> {
    let hidden = u64::from(config.hidden_size);
    let q_width = u64::from(config.n_head) * u64::from(config.head_dim);
    let kv_width = u64::from(config.n_kv_head) * u64::from(config.head_dim);
    let ffn = u64::from(config.ffn_dim);
    let vocab = u64::from(config.vocab_size);
    let expected = FIXED_WIDTH + config.n_layer as usize * LAYER_WIDTH + POST_WIDTH;
    let mut tensors = Vec::with_capacity(expected);

    tensors.extend([
        ("model.embed_tokens.weight".to_owned(), vec![vocab, hidden]),
        ("lm_head.weight".to_owned(), vec![vocab, hidden]),
    ]);
    for layer in 0..config.n_layer {
        let prefix = format!("model.layers.{layer}");
        tensors.extend([
            (format!("{prefix}.input_layernorm.weight"), vec![hidden]),
            (
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![q_width, hidden],
            ),
            (format!("{prefix}.self_attn.q_proj.bias"), vec![q_width]),
            (
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![kv_width, hidden],
            ),
            (format!("{prefix}.self_attn.k_proj.bias"), vec![kv_width]),
            (
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![kv_width, hidden],
            ),
            (format!("{prefix}.self_attn.v_proj.bias"), vec![kv_width]),
            (
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![hidden, q_width],
            ),
            (
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![hidden],
            ),
            (format!("{prefix}.mlp.gate_proj.weight"), vec![ffn, hidden]),
            (format!("{prefix}.mlp.up_proj.weight"), vec![ffn, hidden]),
            (format!("{prefix}.mlp.down_proj.weight"), vec![hidden, ffn]),
        ]);
    }
    tensors.push(("model.norm.weight".to_owned(), vec![hidden]));
    debug_assert_eq!(tensors.len(), expected);
    tensors
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn public_contract_covers_every_tensor_once() {
        let contract = tensor_contract(NeuTtsAirConfig::OFFICIAL);
        assert_eq!(contract.len(), 291);
        let names: BTreeSet<&str> = contract.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names.len(), contract.len());
        assert!(names.contains("lm_head.weight"));
        assert!(names.contains("model.layers.23.self_attn.v_proj.bias"));
        assert!(names.contains("model.norm.weight"));
    }
}
