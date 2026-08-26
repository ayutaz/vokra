//! Structured mmap descriptors for the exact MOSS-Audio release manifests.

use std::sync::Arc;

use vokra_core::gguf::{GgmlType, GgufFile, GgufTensorInfo};
use vokra_core::{Result, VokraError};

use super::MossAudioConfig;

const LABEL: &str = "moss_audio";
const AUDIO_FIXED_WIDTH: usize = 8;
const AUDIO_LAYER_WIDTH: usize = 15;
const AUDIO_POST_WIDTH: usize = 2;
const ADAPTER_COUNT: usize = 4;
const ADAPTER_WIDTH: usize = 3;
const TEXT_FIXED_WIDTH: usize = 1;
const TEXT_LAYER_WIDTH: usize = 11;

pub(super) struct MossAudioMappedDescriptors {
    file: Arc<GgufFile>,
    infos: Vec<GgufTensorInfo>,
    config: MossAudioConfig,
}

impl std::fmt::Debug for MossAudioMappedDescriptors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MossAudioMappedDescriptors")
            .field("tensor_count", &self.infos.len())
            .field("config", &self.config)
            .finish()
    }
}

impl MossAudioMappedDescriptors {
    pub(super) fn bind(file: Arc<GgufFile>, config: MossAudioConfig) -> Result<Self> {
        let contract = tensor_contract(config);
        let mut infos = Vec::with_capacity(contract.len());
        for (name, shape) in contract {
            infos.push(dense_info(&file, &name, &shape)?);
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

    pub(super) const fn config(&self) -> MossAudioConfig {
        self.config
    }

    fn info(&self, index: usize) -> &GgufTensorInfo {
        &self.infos[index]
    }

    pub(super) fn convolution(&self, index: usize) -> MossAudioAffineDescriptors<'_> {
        debug_assert!((1..=3).contains(&index));
        let start = (index - 1) * 2;
        MossAudioAffineDescriptors {
            weight: self.info(start),
            bias: Some(self.info(start + 1)),
        }
    }

    pub(super) fn stem_projection(&self) -> MossAudioAffineDescriptors<'_> {
        MossAudioAffineDescriptors {
            weight: self.info(6),
            bias: Some(self.info(7)),
        }
    }

    pub(super) fn audio_layer(&self, layer: usize) -> MossAudioLayerDescriptors<'_> {
        debug_assert!(layer < self.config.audio.n_layer as usize);
        let start = AUDIO_FIXED_WIDTH + layer * AUDIO_LAYER_WIDTH;
        MossAudioLayerDescriptors {
            self_attn_norm_weight: self.info(start),
            self_attn_norm_bias: self.info(start + 1),
            q_weight: self.info(start + 2),
            q_bias: self.info(start + 3),
            k_weight: self.info(start + 4),
            v_weight: self.info(start + 5),
            v_bias: self.info(start + 6),
            out_weight: self.info(start + 7),
            out_bias: self.info(start + 8),
            final_norm_weight: self.info(start + 9),
            final_norm_bias: self.info(start + 10),
            fc1_weight: self.info(start + 11),
            fc1_bias: self.info(start + 12),
            fc2_weight: self.info(start + 13),
            fc2_bias: self.info(start + 14),
        }
    }

    fn audio_post_start(&self) -> usize {
        AUDIO_FIXED_WIDTH + self.config.audio.n_layer as usize * AUDIO_LAYER_WIDTH
    }

    pub(super) fn audio_post_norm(&self) -> MossAudioNormDescriptors<'_> {
        let start = self.audio_post_start();
        MossAudioNormDescriptors {
            weight: self.info(start),
            bias: self.info(start + 1),
        }
    }

    fn adapter_start(&self) -> usize {
        self.audio_post_start() + AUDIO_POST_WIDTH
    }

    pub(super) fn adapter(&self, index: usize) -> MossAudioAdapterDescriptors<'_> {
        debug_assert!(index < ADAPTER_COUNT);
        let start = self.adapter_start() + index * ADAPTER_WIDTH;
        MossAudioAdapterDescriptors {
            gate: self.info(start),
            up: self.info(start + 1),
            down: self.info(start + 2),
        }
    }

    fn text_start(&self) -> usize {
        self.adapter_start() + ADAPTER_COUNT * ADAPTER_WIDTH
    }

    pub(super) fn text_embedding(&self) -> &GgufTensorInfo {
        self.info(self.text_start())
    }

    pub(super) fn text_layer(&self, layer: usize) -> MossAudioTextLayerDescriptors<'_> {
        debug_assert!(layer < self.config.text.n_layer as usize);
        let start = self.text_start() + TEXT_FIXED_WIDTH + layer * TEXT_LAYER_WIDTH;
        MossAudioTextLayerDescriptors {
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

    pub(super) fn text_head(&self) -> &GgufTensorInfo {
        let index = self.text_start()
            + TEXT_FIXED_WIDTH
            + self.config.text.n_layer as usize * TEXT_LAYER_WIDTH
            + 1;
        self.info(index)
    }
}

pub(super) struct MossAudioAffineDescriptors<'a> {
    pub(super) weight: &'a GgufTensorInfo,
    pub(super) bias: Option<&'a GgufTensorInfo>,
}

pub(super) struct MossAudioNormDescriptors<'a> {
    pub(super) weight: &'a GgufTensorInfo,
    pub(super) bias: &'a GgufTensorInfo,
}

pub(super) struct MossAudioLayerDescriptors<'a> {
    pub(super) self_attn_norm_weight: &'a GgufTensorInfo,
    pub(super) self_attn_norm_bias: &'a GgufTensorInfo,
    pub(super) q_weight: &'a GgufTensorInfo,
    pub(super) q_bias: &'a GgufTensorInfo,
    pub(super) k_weight: &'a GgufTensorInfo,
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

pub(super) struct MossAudioAdapterDescriptors<'a> {
    pub(super) gate: &'a GgufTensorInfo,
    pub(super) up: &'a GgufTensorInfo,
    pub(super) down: &'a GgufTensorInfo,
}

pub(super) struct MossAudioTextLayerDescriptors<'a> {
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

fn dense_info(file: &GgufFile, name: &str, expected_shape: &[u64]) -> Result<GgufTensorInfo> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!("{LABEL}: required tensor `{name}` is missing"))
    })?;
    if info.dimensions != expected_shape {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: tensor `{name}` shape {:?}, expected {expected_shape:?}",
            info.dimensions
        )));
    }
    match info.dtype {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => Ok(info.clone()),
        other => Err(VokraError::ModelLoad(format!(
            "{LABEL}: tensor `{name}` uses unsupported dtype {other:?}; MOSS-Audio's bounded-memory runtime supports only the official dense F32/F16/BF16 checkpoints. Quantized execution is not implemented and will not silently select a resident or CPU fallback"
        ))),
    }
}

pub(super) fn tensor_contract(config: MossAudioConfig) -> Vec<(String, Vec<u64>)> {
    let text_hidden = u64::from(config.text.hidden_size);
    let text_ffn = u64::from(config.text.ffn_dim);
    let audio_hidden = u64::from(config.audio.d_model);
    let audio_ffn = u64::from(config.audio.ffn_dim);
    let adapter_hidden = u64::from(config.adapter_hidden_size);
    let query_width = u64::from(config.text.n_head) * u64::from(config.text.head_dim);
    let key_value_width = u64::from(config.text.n_kv_head) * u64::from(config.text.head_dim);
    let mut tensors = Vec::with_capacity(901);
    let mut insert = |name: String, shape: &[u64]| tensors.push((name, shape.to_vec()));

    insert("audio_encoder.conv1.weight".into(), &[480, 1, 3, 3]);
    insert("audio_encoder.conv1.bias".into(), &[480]);
    for index in 2..=3 {
        insert(
            format!("audio_encoder.conv{index}.weight"),
            &[480, 480, 3, 3],
        );
        insert(format!("audio_encoder.conv{index}.bias"), &[480]);
    }
    insert(
        "audio_encoder.stem_proj.weight".into(),
        &[audio_hidden, 7_680],
    );
    insert("audio_encoder.stem_proj.bias".into(), &[audio_hidden]);
    for layer in 0..config.audio.n_layer {
        let prefix = format!("audio_encoder.layers.{layer}");
        insert(
            format!("{prefix}.self_attn_layer_norm.weight"),
            &[audio_hidden],
        );
        insert(
            format!("{prefix}.self_attn_layer_norm.bias"),
            &[audio_hidden],
        );
        insert(
            format!("{prefix}.self_attn.q_proj.weight"),
            &[audio_hidden, audio_hidden],
        );
        insert(format!("{prefix}.self_attn.q_proj.bias"), &[audio_hidden]);
        insert(
            format!("{prefix}.self_attn.k_proj.weight"),
            &[audio_hidden, audio_hidden],
        );
        insert(
            format!("{prefix}.self_attn.v_proj.weight"),
            &[audio_hidden, audio_hidden],
        );
        insert(format!("{prefix}.self_attn.v_proj.bias"), &[audio_hidden]);
        insert(
            format!("{prefix}.self_attn.out_proj.weight"),
            &[audio_hidden, audio_hidden],
        );
        insert(format!("{prefix}.self_attn.out_proj.bias"), &[audio_hidden]);
        insert(format!("{prefix}.final_layer_norm.weight"), &[audio_hidden]);
        insert(format!("{prefix}.final_layer_norm.bias"), &[audio_hidden]);
        insert(format!("{prefix}.fc1.weight"), &[audio_ffn, audio_hidden]);
        insert(format!("{prefix}.fc1.bias"), &[audio_ffn]);
        insert(format!("{prefix}.fc2.weight"), &[audio_hidden, audio_ffn]);
        insert(format!("{prefix}.fc2.bias"), &[audio_hidden]);
    }
    insert("audio_encoder.layer_norm.weight".into(), &[audio_hidden]);
    insert("audio_encoder.layer_norm.bias".into(), &[audio_hidden]);

    for prefix in std::iter::once("audio_adapter".to_owned())
        .chain((0..3).map(|index| format!("deepstack_audio_merger_list.{index}")))
    {
        insert(
            format!("{prefix}.gate_proj.weight"),
            &[adapter_hidden, audio_hidden],
        );
        insert(
            format!("{prefix}.up_proj.weight"),
            &[adapter_hidden, audio_hidden],
        );
        insert(
            format!("{prefix}.down_proj.weight"),
            &[text_hidden, adapter_hidden],
        );
    }

    insert(
        "language_model.embed_tokens.weight".into(),
        &[u64::from(config.text.vocab_size), text_hidden],
    );
    for layer in 0..config.text.n_layer {
        let prefix = format!("language_model.layers.{layer}");
        insert(format!("{prefix}.input_layernorm.weight"), &[text_hidden]);
        insert(
            format!("{prefix}.self_attn.q_proj.weight"),
            &[query_width, text_hidden],
        );
        insert(
            format!("{prefix}.self_attn.q_norm.weight"),
            &[u64::from(config.text.head_dim)],
        );
        insert(
            format!("{prefix}.self_attn.k_proj.weight"),
            &[key_value_width, text_hidden],
        );
        insert(
            format!("{prefix}.self_attn.k_norm.weight"),
            &[u64::from(config.text.head_dim)],
        );
        insert(
            format!("{prefix}.self_attn.v_proj.weight"),
            &[key_value_width, text_hidden],
        );
        insert(
            format!("{prefix}.self_attn.o_proj.weight"),
            &[text_hidden, query_width],
        );
        insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            &[text_hidden],
        );
        insert(
            format!("{prefix}.mlp.gate_proj.weight"),
            &[text_ffn, text_hidden],
        );
        insert(
            format!("{prefix}.mlp.up_proj.weight"),
            &[text_ffn, text_hidden],
        );
        insert(
            format!("{prefix}.mlp.down_proj.weight"),
            &[text_hidden, text_ffn],
        );
    }
    insert("language_model.norm.weight".into(), &[text_hidden]);
    insert(
        "lm_head.weight".into(),
        &[u64::from(config.text.vocab_size), text_hidden],
    );
    debug_assert_eq!(tensors.len(), 901);
    tensors
}
