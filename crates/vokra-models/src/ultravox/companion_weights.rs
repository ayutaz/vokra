//! Exact mmap descriptors for the separately acquired Llama 3.2 companion.
//!
//! The admitted BF16 checkpoint is larger than 2 GB, so binding must remain
//! header-only. Tensor payloads stay in the original mapping; the decoder
//! materializes one layer at a time in a follow-up execution module.

use std::sync::Arc;

use vokra_core::gguf::{GgmlType, GgufFile, GgufTensorInfo};
use vokra_core::{Result, VokraError};

use crate::mapped_weights::{MappedModel, mapped_info};

use super::companion::UltravoxLlamaConfig;

const MAPPED: MappedModel = MappedModel {
    name: "ultravox_llama_companion",
    resident_entry: "UltravoxLlamaCompanion::open_mapped after re-converting the exact dense BF16 snapshot",
};

pub(super) struct UltravoxLlamaMappedDescriptors {
    file: Arc<GgufFile>,
    infos: Vec<GgufTensorInfo>,
    config: UltravoxLlamaConfig,
}

impl std::fmt::Debug for UltravoxLlamaMappedDescriptors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UltravoxLlamaMappedDescriptors")
            .field("mapped_file_tensors", &self.file.tensors().len())
            .field("descriptor_count", &self.infos.len())
            .field("config", &self.config)
            .finish()
    }
}

impl UltravoxLlamaMappedDescriptors {
    pub(super) fn bind(file: Arc<GgufFile>, config: UltravoxLlamaConfig) -> Result<Self> {
        let contract = tensor_contract(config);
        let mut infos = Vec::with_capacity(contract.len());
        for (name, shape) in contract {
            let elements = shape
                .iter()
                .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
                .and_then(|count| usize::try_from(count).ok())
                .expect("fixed Llama-3.2-1B tensor shapes fit usize");
            let info = mapped_info(&file, &name, elements, MAPPED)?;
            if info.dtype != GgmlType::BF16 {
                return Err(VokraError::ModelLoad(format!(
                    "ultravox_llama_companion: tensor `{name}` has {:?}, expected canonical BF16",
                    info.dtype
                )));
            }
            infos.push(info);
        }
        Ok(Self {
            file,
            infos,
            config,
        })
    }

    pub(super) const fn config(&self) -> UltravoxLlamaConfig {
        self.config
    }

    pub(super) fn descriptor_count(&self) -> usize {
        debug_assert_eq!(self.file.tensors().len(), self.infos.len());
        self.infos.len()
    }
}

pub(super) fn tensor_contract(config: UltravoxLlamaConfig) -> Vec<(String, Vec<u64>)> {
    let hidden = u64::from(config.hidden_size);
    let ffn = u64::from(config.ffn_dim);
    let q_width = u64::from(config.n_head) * u64::from(config.head_dim);
    let kv_width = u64::from(config.n_kv_head) * u64::from(config.head_dim);
    let vocab = u64::from(config.vocab_size);
    let expected = 2 + config.n_layer as usize * 9;
    let mut tensors = Vec::with_capacity(expected);
    tensors.push(("model.embed_tokens.weight".to_owned(), vec![vocab, hidden]));
    for layer in 0..config.n_layer {
        let prefix = format!("model.layers.{layer}");
        tensors.extend([
            (format!("{prefix}.input_layernorm.weight"), vec![hidden]),
            (
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![q_width, hidden],
            ),
            (
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![kv_width, hidden],
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
    fn exact_tied_embedding_contract_has_no_lm_head() {
        let contract = tensor_contract(UltravoxLlamaConfig::OFFICIAL);
        assert_eq!(contract.len(), 146);
        let names: BTreeSet<&str> = contract.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names.len(), contract.len());
        assert!(names.contains("model.embed_tokens.weight"));
        assert!(names.contains("model.layers.15.mlp.down_proj.weight"));
        assert!(names.contains("model.norm.weight"));
        assert!(!names.contains("lm_head.weight"));
    }
}
