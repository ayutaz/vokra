//! Structured mmap descriptors for the Qwen3-TTS main generation graph.
//!
//! Binding keeps every dense tensor in the original GGUF mapping. Execution
//! widens one Transformer layer or one embedding/head slice at a time, which
//! avoids a second multi-gigabyte resident copy on the maintainer Mac.

use std::sync::Arc;

use vokra_core::Result;
use vokra_core::gguf::{GgufFile, GgufTensorInfo};

use crate::mapped_weights::{MappedModel, mapped_info};

use super::Qwen3TtsConfig;

const LABEL: &str = "qwen3_tts";
const TALKER_FIXED_WIDTH: usize = 8;
const LAYER_WIDTH: usize = 11;

const MAPPED: MappedModel = MappedModel {
    name: LABEL,
    resident_entry: "Qwen3TtsCheckpoint::open_mapped after re-converting the pinned dense checkpoint",
};

/// Dense descriptors needed by the talker and code predictor.
///
/// Base-only speaker-encoder tensors are already authenticated by the strict
/// release manifest in `bound.rs`; they are intentionally outside this main
/// generation descriptor until the reference-audio frontend is executed.
pub(super) struct Qwen3TtsMappedDescriptors {
    file: Arc<GgufFile>,
    infos: Vec<GgufTensorInfo>,
    config: Qwen3TtsConfig,
}

impl std::fmt::Debug for Qwen3TtsMappedDescriptors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3TtsMappedDescriptors")
            .field("generation_tensor_count", &self.infos.len())
            .field("config", &self.config)
            .finish()
    }
}

impl Qwen3TtsMappedDescriptors {
    pub(super) fn bind(file: Arc<GgufFile>, config: Qwen3TtsConfig) -> Result<Self> {
        let contract = generation_contract(&config);
        let mut infos = Vec::with_capacity(contract.len());
        for (name, elements) in contract {
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

    pub(super) fn info(&self, index: usize) -> &GgufTensorInfo {
        &self.infos[index]
    }

    pub(super) fn config(&self) -> &Qwen3TtsConfig {
        &self.config
    }

    pub(super) const fn mapped_model(&self) -> MappedModel {
        MAPPED
    }

    pub(super) fn text_embedding(&self) -> &GgufTensorInfo {
        self.info(0)
    }

    pub(super) fn talker_codec_embedding(&self) -> &GgufTensorInfo {
        self.info(1)
    }

    pub(super) fn talker_final_norm(&self) -> &GgufTensorInfo {
        self.info(2)
    }

    pub(super) fn text_projection_fc1_weight(&self) -> &GgufTensorInfo {
        self.info(3)
    }

    pub(super) fn text_projection_fc1_bias(&self) -> &GgufTensorInfo {
        self.info(4)
    }

    pub(super) fn text_projection_fc2_weight(&self) -> &GgufTensorInfo {
        self.info(5)
    }

    pub(super) fn text_projection_fc2_bias(&self) -> &GgufTensorInfo {
        self.info(6)
    }

    pub(super) fn talker_codec_head(&self) -> &GgufTensorInfo {
        self.info(7)
    }

    pub(super) fn talker_layer(&self, layer: usize) -> DecoderLayerDescriptors<'_> {
        debug_assert!(layer < self.config.talker.n_layer as usize);
        layer_descriptors(&self.infos, TALKER_FIXED_WIDTH + layer * LAYER_WIDTH)
    }

    fn code_predictor_start(&self) -> usize {
        TALKER_FIXED_WIDTH + self.config.talker.n_layer as usize * LAYER_WIDTH
    }

    pub(super) fn code_predictor_final_norm(&self) -> &GgufTensorInfo {
        self.info(self.code_predictor_start())
    }

    pub(super) fn code_predictor_embedding(&self, group: usize) -> &GgufTensorInfo {
        let groups = self.config.code_predictor.num_code_groups as usize - 1;
        debug_assert!(group < groups);
        self.info(self.code_predictor_start() + 1 + group)
    }

    pub(super) fn code_predictor_head(&self, group: usize) -> &GgufTensorInfo {
        let groups = self.config.code_predictor.num_code_groups as usize - 1;
        debug_assert!(group < groups);
        self.info(self.code_predictor_start() + 1 + groups + group)
    }

    fn code_predictor_projection_start(&self) -> usize {
        let groups = self.config.code_predictor.num_code_groups as usize - 1;
        self.code_predictor_start() + 1 + groups * 2
    }

    pub(super) fn code_predictor_projection(&self) -> Option<(&GgufTensorInfo, &GgufTensorInfo)> {
        (self.config.talker.hidden_dim != self.config.code_predictor.hidden_dim).then(|| {
            let start = self.code_predictor_projection_start();
            (self.info(start), self.info(start + 1))
        })
    }

    fn code_predictor_layers_start(&self) -> usize {
        self.code_predictor_projection_start()
            + usize::from(self.config.talker.hidden_dim != self.config.code_predictor.hidden_dim)
                * 2
    }

    pub(super) fn code_predictor_layer(&self, layer: usize) -> DecoderLayerDescriptors<'_> {
        debug_assert!(layer < self.config.code_predictor.n_layer as usize);
        layer_descriptors(
            &self.infos,
            self.code_predictor_layers_start() + layer * LAYER_WIDTH,
        )
    }
}

pub(super) struct DecoderLayerDescriptors<'a> {
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

fn layer_descriptors(infos: &[GgufTensorInfo], start: usize) -> DecoderLayerDescriptors<'_> {
    DecoderLayerDescriptors {
        input_norm: &infos[start],
        q: &infos[start + 1],
        q_norm: &infos[start + 2],
        k: &infos[start + 3],
        k_norm: &infos[start + 4],
        v: &infos[start + 5],
        o: &infos[start + 6],
        ffn_norm: &infos[start + 7],
        gate: &infos[start + 8],
        up: &infos[start + 9],
        down: &infos[start + 10],
    }
}

pub(super) fn generation_contract(config: &Qwen3TtsConfig) -> Vec<(String, usize)> {
    let talker = &config.talker;
    let predictor = &config.code_predictor;
    let hidden = talker.hidden_dim as usize;
    let text_hidden = talker.text_hidden_size as usize;
    let predictor_hidden = predictor.hidden_dim as usize;
    let groups = predictor.num_code_groups as usize - 1;
    let projection = usize::from(hidden != predictor_hidden) * 2;
    let expected = TALKER_FIXED_WIDTH
        + talker.n_layer as usize * LAYER_WIDTH
        + 1
        + groups * 2
        + projection
        + predictor.n_layer as usize * LAYER_WIDTH;
    let mut tensors = Vec::with_capacity(expected);
    tensors.extend([
        (
            "talker.model.text_embedding.weight".to_owned(),
            talker.text_vocab_size as usize * text_hidden,
        ),
        (
            "talker.model.codec_embedding.weight".to_owned(),
            talker.vocab_size as usize * hidden,
        ),
        ("talker.model.norm.weight".to_owned(), hidden),
        (
            "talker.text_projection.linear_fc1.weight".to_owned(),
            text_hidden * text_hidden,
        ),
        (
            "talker.text_projection.linear_fc1.bias".to_owned(),
            text_hidden,
        ),
        (
            "talker.text_projection.linear_fc2.weight".to_owned(),
            hidden * text_hidden,
        ),
        ("talker.text_projection.linear_fc2.bias".to_owned(), hidden),
        (
            "talker.codec_head.weight".to_owned(),
            talker.vocab_size as usize * hidden,
        ),
    ]);
    append_layers(&mut tensors, "talker.model", talker);

    tensors.push((
        "talker.code_predictor.model.norm.weight".to_owned(),
        predictor_hidden,
    ));
    for group in 0..groups {
        tensors.push((
            format!("talker.code_predictor.model.codec_embedding.{group}.weight"),
            predictor.vocab_size as usize * hidden,
        ));
    }
    for group in 0..groups {
        tensors.push((
            format!("talker.code_predictor.lm_head.{group}.weight"),
            predictor.vocab_size as usize * predictor_hidden,
        ));
    }
    if hidden != predictor_hidden {
        tensors.extend([
            (
                "talker.code_predictor.small_to_mtp_projection.weight".to_owned(),
                predictor_hidden * hidden,
            ),
            (
                "talker.code_predictor.small_to_mtp_projection.bias".to_owned(),
                predictor_hidden,
            ),
        ]);
    }
    append_layers(&mut tensors, "talker.code_predictor.model", predictor);
    debug_assert_eq!(tensors.len(), expected);
    tensors
}

fn append_layers<C: LayerConfig>(tensors: &mut Vec<(String, usize)>, prefix: &str, config: &C) {
    let hidden = config.hidden();
    let q = config.q_width();
    let kv = config.kv_width();
    let head = config.head_dim();
    let ffn = config.ffn();
    for layer in 0..config.layers() {
        let prefix = format!("{prefix}.layers.{layer}");
        tensors.extend([
            (format!("{prefix}.input_layernorm.weight"), hidden),
            (format!("{prefix}.self_attn.q_proj.weight"), q * hidden),
            (format!("{prefix}.self_attn.q_norm.weight"), head),
            (format!("{prefix}.self_attn.k_proj.weight"), kv * hidden),
            (format!("{prefix}.self_attn.k_norm.weight"), head),
            (format!("{prefix}.self_attn.v_proj.weight"), kv * hidden),
            (format!("{prefix}.self_attn.o_proj.weight"), hidden * q),
            (format!("{prefix}.post_attention_layernorm.weight"), hidden),
            (format!("{prefix}.mlp.gate_proj.weight"), ffn * hidden),
            (format!("{prefix}.mlp.up_proj.weight"), ffn * hidden),
            (format!("{prefix}.mlp.down_proj.weight"), hidden * ffn),
        ]);
    }
}

trait LayerConfig {
    fn hidden(&self) -> usize;
    fn layers(&self) -> usize;
    fn q_width(&self) -> usize;
    fn kv_width(&self) -> usize;
    fn head_dim(&self) -> usize;
    fn ffn(&self) -> usize;
}

impl LayerConfig for super::Qwen3TtsTalkerConfig {
    fn hidden(&self) -> usize {
        self.hidden_dim as usize
    }
    fn layers(&self) -> usize {
        self.n_layer as usize
    }
    fn q_width(&self) -> usize {
        self.n_head as usize * self.head_dim as usize
    }
    fn kv_width(&self) -> usize {
        self.n_head_kv as usize * self.head_dim as usize
    }
    fn head_dim(&self) -> usize {
        self.head_dim as usize
    }
    fn ffn(&self) -> usize {
        self.ffn_dim as usize
    }
}

impl LayerConfig for super::Qwen3TtsCodePredictorConfig {
    fn hidden(&self) -> usize {
        self.hidden_dim as usize
    }
    fn layers(&self) -> usize {
        self.n_layer as usize
    }
    fn q_width(&self) -> usize {
        self.n_head as usize * self.head_dim as usize
    }
    fn kv_width(&self) -> usize {
        self.n_head_kv as usize * self.head_dim as usize
    }
    fn head_dim(&self) -> usize {
        self.head_dim as usize
    }
    fn ffn(&self) -> usize {
        self.ffn_dim as usize
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn generation_contract_covers_every_non_speaker_release_tensor() {
        for (config, expected) in [
            (Qwen3TtsConfig::qwen3_tts_0_6b_base(), 402),
            (Qwen3TtsConfig::qwen3_tts_1_7b_base(), 404),
        ] {
            let contract = generation_contract(&config);
            assert_eq!(contract.len(), expected);
            let names: BTreeSet<&str> = contract.iter().map(|(name, _)| name.as_str()).collect();
            assert_eq!(names.len(), contract.len());
            assert!(names.contains("talker.model.layers.27.self_attn.q_norm.weight"));
            assert!(names.contains("talker.code_predictor.lm_head.14.weight"));
            assert!(names.contains("talker.code_predictor.model.layers.4.mlp.down_proj.weight"));
        }
    }
}
