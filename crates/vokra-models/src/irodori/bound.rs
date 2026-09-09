//! Strict real-checkpoint binding and the native Irodori text block.
//!
//! The historical public Vokra artifact is a partial, 637-tensor diagnostic
//! checkpoint. This module validates that name/shape tree without eagerly
//! decoding two gigabytes of weights, and decodes one text-encoder block on
//! demand for diagnostics; it is not an end-to-end TTS binder.

use std::collections::BTreeMap;

use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use super::{
    EXPECTED_ARCH, IrodoriConfig, IrodoriDitConfig, IrodoriDurationPredictorConfig,
    IrodoriSpeakerEncoderConfig, IrodoriTextEncoderConfig,
};

const MODEL_NAME: &str = "irodori-tts-500m-v3";
const EXPECTED_TENSOR_COUNT: usize = 637;

/// A strictly validated Irodori-TTS-500M-v3 checkpoint.
///
/// The handle stores only resolved metadata. Callers retain the mmap-backed
/// [`GgufFile`] and decode the block they need, avoiding an eager multi-GB F32
/// copy before rectified-flow sampling begins.
#[derive(Debug, Clone)]
pub struct IrodoriCheckpoint {
    config: IrodoriConfig,
    model_name: String,
    weight_license: LicenseClass,
    tensor_count: usize,
}

impl IrodoriCheckpoint {
    /// Validates the architecture, all 30 topology axes, and the exact
    /// historical 637-tensor diagnostic manifest. This does not authenticate
    /// the separate tokenizer, reference encoder, or Semantic-DACVAE codec.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let arch = required_string(file, chunks::KEY_MODEL_ARCH)?;
        if arch != EXPECTED_ARCH {
            return Err(VokraError::ModelLoad(format!(
                "irodori: unsupported `{}`={arch:?}; expected {EXPECTED_ARCH:?}",
                chunks::KEY_MODEL_ARCH
            )));
        }
        let model_name = required_string(file, chunks::KEY_MODEL_NAME)?.to_owned();
        if model_name != MODEL_NAME {
            return Err(VokraError::ModelLoad(format!(
                "irodori: unsupported `{}`={model_name:?}; the strict manifest is pinned to {MODEL_NAME:?}",
                chunks::KEY_MODEL_NAME
            )));
        }

        let config = config_from_gguf(file)?;
        config.validate_for_forward().map_err(|error| {
            VokraError::ModelLoad(format!(
                "irodori: stamped topology is not forward-safe: {error}"
            ))
        })?;
        let canonical = IrodoriConfig::irodori_500m_v3();
        if config != canonical {
            return Err(VokraError::ModelLoad(format!(
                "irodori: stamped topology does not match the audited v3 release; found {config:?}, expected {canonical:?}"
            )));
        }
        if file.tensors().len() != EXPECTED_TENSOR_COUNT {
            return Err(VokraError::ModelLoad(format!(
                "irodori: tensor count {}, expected {EXPECTED_TENSOR_COUNT} for the official v3 release",
                file.tensors().len()
            )));
        }
        validate_manifest(file, &config)?;

        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(GgufMetadataValue::as_str)
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        Ok(Self {
            config,
            model_name,
            weight_license,
            tensor_count: file.tensors().len(),
        })
    }

    /// Returns the topology parsed from the GGUF metadata.
    #[must_use]
    pub fn config(&self) -> &IrodoriConfig {
        &self.config
    }

    /// Returns the exact checkpoint variant admitted by this binder.
    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Returns the fail-closed weight-license class stamped in the GGUF.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Returns the number of tensors checked against the official manifest.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    /// Decodes one real text-encoder block after rechecking its names/shapes.
    pub fn load_text_block(
        &self,
        file: &GgufFile,
        layer: usize,
    ) -> Result<IrodoriTextBlockWeights> {
        let cfg = &self.config.text;
        if layer >= cfg.n_layer as usize {
            return Err(VokraError::InvalidArgument(format!(
                "irodori: text layer {layer} out of range 0..{}",
                cfg.n_layer
            )));
        }
        load_text_block(file, layer, cfg)
    }

    /// Primary text-to-PCM entry point for a bound real checkpoint.
    ///
    /// Strict loading and individual text blocks are real. End-to-end PCM
    /// remains loud until tokenizer sidecars, the remaining encoders/DiT
    /// loop, and the separately distributed Semantic-DACVAE are bound.
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "irodori synthesize: text is empty".to_owned(),
            ));
        }
        Err(VokraError::NotImplemented(
            "irodori synthesize: PARTIAL_RUNTIME only — a text-encoder diagnostic block is available, but end-to-end PCM still requires authenticated tokenizer/reference components, duration and RF-DiT sampling, and the distinct Semantic-DACVAE-Japanese-32dim decoder.",
        ))
    }
}

/// Real weights for one Irodori text-encoder block.
///
/// Matrices retain the upstream PyTorch `[out, in]` row-major layout.
#[derive(Debug, Clone)]
pub struct IrodoriTextBlockWeights {
    /// Pre-attention RMSNorm scale.
    pub attention_norm: Vec<f32>,
    /// Bias-free query projection.
    pub wq: Vec<f32>,
    /// Bias-free key projection.
    pub wk: Vec<f32>,
    /// Bias-free value projection.
    pub wv: Vec<f32>,
    /// Per-head query RMSNorm scales, flattened from `[heads, head_dim]`.
    pub q_norm: Vec<f32>,
    /// Per-head key RMSNorm scales, flattened from `[heads, head_dim]`.
    pub k_norm: Vec<f32>,
    /// Sigmoid attention-output gate projection.
    pub gate: Vec<f32>,
    /// Bias-free attention output projection.
    pub wo: Vec<f32>,
    /// Pre-MLP RMSNorm scale.
    pub mlp_norm: Vec<f32>,
    /// SwiGLU gate projection.
    pub w1: Vec<f32>,
    /// SwiGLU output projection.
    pub w2: Vec<f32>,
    /// SwiGLU value projection.
    pub w3: Vec<f32>,
}

fn load_text_block(
    file: &GgufFile,
    layer: usize,
    cfg: &IrodoriTextEncoderConfig,
) -> Result<IrodoriTextBlockWeights> {
    let d = cfg.dim as usize;
    let h = cfg.n_head as usize;
    let hd = d / h;
    let f = cfg.ffn_inner_dim() as usize;
    let prefix = format!("text_encoder.blocks.{layer}");
    Ok(IrodoriTextBlockWeights {
        attention_norm: tensor(file, &format!("{prefix}.attention_norm.weight"), &[d])?,
        wq: tensor(file, &format!("{prefix}.attention.wq.weight"), &[d, d])?,
        wk: tensor(file, &format!("{prefix}.attention.wk.weight"), &[d, d])?,
        wv: tensor(file, &format!("{prefix}.attention.wv.weight"), &[d, d])?,
        q_norm: tensor(file, &format!("{prefix}.attention.q_norm.weight"), &[h, hd])?,
        k_norm: tensor(file, &format!("{prefix}.attention.k_norm.weight"), &[h, hd])?,
        gate: tensor(file, &format!("{prefix}.attention.gate.weight"), &[d, d])?,
        wo: tensor(file, &format!("{prefix}.attention.wo.weight"), &[d, d])?,
        mlp_norm: tensor(file, &format!("{prefix}.mlp_norm.weight"), &[d])?,
        w1: tensor(file, &format!("{prefix}.mlp.w1.weight"), &[f, d])?,
        w2: tensor(file, &format!("{prefix}.mlp.w2.weight"), &[d, f])?,
        w3: tensor(file, &format!("{prefix}.mlp.w3.weight"), &[f, d])?,
    })
}

/// Runs one real Irodori text-encoder layer for batch size one.
///
/// `key_mask` uses the official PyTorch boolean convention (`true` means the
/// key is visible). Attention is non-causal and RoPE rotates adjacent pairs.
pub fn irodori_text_block_forward(
    cfg: &IrodoriTextEncoderConfig,
    weights: &IrodoriTextBlockWeights,
    hidden: &[f32],
    key_mask: &[bool],
) -> Result<Vec<f32>> {
    let d = cfg.dim as usize;
    let heads = cfg.n_head as usize;
    let Some(head_dim) = cfg.head_dim().map(|value| value as usize) else {
        return Err(VokraError::InvalidArgument(
            "irodori text block: n_head must divide dim".to_owned(),
        ));
    };
    let seq = key_mask.len();
    if seq == 0 || hidden.len() != seq * d || head_dim % 2 != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "irodori text block: hidden len {} must equal non-zero mask len {seq} * dim {d}, with an even head_dim",
            hidden.len()
        )));
    }
    check_text_block_weight_lengths(cfg, weights)?;

    let normed = rms_norm_rows(hidden, &weights.attention_norm, seq, d, 1.0e-5);
    let mut q = linear_rows(&normed, &weights.wq, seq, d, d);
    let mut k = linear_rows(&normed, &weights.wk, seq, d, d);
    let v = linear_rows(&normed, &weights.wv, seq, d, d);
    let gate = linear_rows(&normed, &weights.gate, seq, d, d);
    rms_norm_heads(&mut q, &weights.q_norm, seq, heads, head_dim, 1.0e-5);
    rms_norm_heads(&mut k, &weights.k_norm, seq, heads, head_dim, 1.0e-5);
    apply_adjacent_rope(&mut q, seq, heads, head_dim, 10_000.0);
    apply_adjacent_rope(&mut k, seq, heads, head_dim, 10_000.0);

    let mut attended = vec![0.0f32; seq * d];
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![f32::NEG_INFINITY; seq];
    for t in 0..seq {
        for head in 0..heads {
            let qoff = (t * heads + head) * head_dim;
            let mut max_score = f32::NEG_INFINITY;
            for s in 0..seq {
                if !key_mask[s] {
                    scores[s] = f32::NEG_INFINITY;
                    continue;
                }
                let koff = (s * heads + head) * head_dim;
                let mut dot = 0.0;
                for i in 0..head_dim {
                    dot += q[qoff + i] * k[koff + i];
                }
                scores[s] = dot * scale;
                max_score = max_score.max(scores[s]);
            }
            if !max_score.is_finite() {
                continue;
            }
            let mut denom = 0.0;
            for s in 0..seq {
                if key_mask[s] {
                    scores[s] = (scores[s] - max_score).exp();
                    denom += scores[s];
                }
            }
            for s in 0..seq {
                if !key_mask[s] {
                    continue;
                }
                let probability = scores[s] / denom;
                let voff = (s * heads + head) * head_dim;
                for i in 0..head_dim {
                    attended[qoff + i] += probability * v[voff + i];
                }
            }
        }
    }
    for (value, gate) in attended.iter_mut().zip(gate) {
        *value *= 1.0 / (1.0 + (-gate).exp());
    }
    let attention = linear_rows(&attended, &weights.wo, seq, d, d);
    let mut residual: Vec<f32> = hidden
        .iter()
        .zip(attention)
        .map(|(&left, right)| left + right)
        .collect();

    let mlp_input = rms_norm_rows(&residual, &weights.mlp_norm, seq, d, 1.0e-5);
    let f = cfg.ffn_inner_dim() as usize;
    let w1 = linear_rows(&mlp_input, &weights.w1, seq, d, f);
    let w3 = linear_rows(&mlp_input, &weights.w3, seq, d, f);
    let activated: Vec<f32> = w1
        .iter()
        .zip(w3)
        .map(|(&gate, value)| (gate / (1.0 + (-gate).exp())) * value)
        .collect();
    let mlp = linear_rows(&activated, &weights.w2, seq, f, d);
    for (value, add) in residual.iter_mut().zip(mlp) {
        *value += add;
    }
    Ok(residual)
}

fn check_text_block_weight_lengths(
    cfg: &IrodoriTextEncoderConfig,
    weights: &IrodoriTextBlockWeights,
) -> Result<()> {
    let d = cfg.dim as usize;
    let f = cfg.ffn_inner_dim() as usize;
    for (name, got, want) in [
        ("attention_norm", weights.attention_norm.len(), d),
        ("wq", weights.wq.len(), d * d),
        ("wk", weights.wk.len(), d * d),
        ("wv", weights.wv.len(), d * d),
        ("q_norm", weights.q_norm.len(), d),
        ("k_norm", weights.k_norm.len(), d),
        ("gate", weights.gate.len(), d * d),
        ("wo", weights.wo.len(), d * d),
        ("mlp_norm", weights.mlp_norm.len(), d),
        ("w1", weights.w1.len(), f * d),
        ("w2", weights.w2.len(), d * f),
        ("w3", weights.w3.len(), f * d),
    ] {
        if got != want {
            return Err(VokraError::InvalidArgument(format!(
                "irodori text block: `{name}` has {got} values, expected {want}"
            )));
        }
    }
    Ok(())
}

fn linear_rows(x: &[f32], weight: &[f32], rows: usize, input: usize, output: usize) -> Vec<f32> {
    let mut y = vec![0.0; rows * output];
    for row in 0..rows {
        for out in 0..output {
            let mut sum = 0.0;
            for inner in 0..input {
                sum += x[row * input + inner] * weight[out * input + inner];
            }
            y[row * output + out] = sum;
        }
    }
    y
}

fn rms_norm_rows(x: &[f32], weight: &[f32], rows: usize, width: usize, eps: f32) -> Vec<f32> {
    let mut output = vec![0.0; x.len()];
    for row in 0..rows {
        let offset = row * width;
        let variance = x[offset..offset + width]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / width as f32;
        let inverse = 1.0 / (variance + eps).sqrt();
        for index in 0..width {
            output[offset + index] = x[offset + index] * inverse * weight[index];
        }
    }
    output
}

fn rms_norm_heads(
    x: &mut [f32],
    weight: &[f32],
    seq: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) {
    for token in 0..seq {
        for head in 0..heads {
            let offset = (token * heads + head) * head_dim;
            let variance = x[offset..offset + head_dim]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / head_dim as f32;
            let inverse = 1.0 / (variance + eps).sqrt();
            for index in 0..head_dim {
                x[offset + index] = x[offset + index] * inverse * weight[head * head_dim + index];
            }
        }
    }
}

fn apply_adjacent_rope(x: &mut [f32], seq: usize, heads: usize, head_dim: usize, base: f32) {
    for token in 0..seq {
        for head in 0..heads {
            let offset = (token * heads + head) * head_dim;
            for pair in 0..head_dim / 2 {
                let inverse_frequency = 1.0 / base.powf((2 * pair) as f32 / head_dim as f32);
                let angle = token as f32 * inverse_frequency;
                let (sin, cos) = angle.sin_cos();
                let left = x[offset + 2 * pair];
                let right = x[offset + 2 * pair + 1];
                x[offset + 2 * pair] = left * cos - right * sin;
                x[offset + 2 * pair + 1] = right * cos + left * sin;
            }
        }
    }
}

fn tensor(file: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!("irodori: required tensor `{name}` is missing"))
    })?;
    let actual: Vec<usize> = info
        .dimensions
        .iter()
        .map(|&value| value as usize)
        .collect();
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "irodori: tensor `{name}` shape {actual:?}, expected {expected:?}"
        )));
    }
    file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!("irodori: tensor `{name}` decode failed: {error}"))
    })
}

fn config_from_gguf(file: &GgufFile) -> Result<IrodoriConfig> {
    let family = required_string(file, "vokra.irodori.model_family")?;
    if family != EXPECTED_ARCH {
        return Err(VokraError::ModelLoad(format!(
            "irodori: `vokra.irodori.model_family`={family:?}, expected {EXPECTED_ARCH:?}"
        )));
    }
    Ok(IrodoriConfig {
        dit: IrodoriDitConfig {
            latent_dim: required_u32(file, "vokra.irodori.dit.latent_dim")?,
            latent_patch_size: required_u32(file, "vokra.irodori.dit.latent_patch_size")?,
            model_dim: required_u32(file, "vokra.irodori.dit.model_dim")?,
            num_layers: required_u32(file, "vokra.irodori.dit.num_layers")?,
            num_heads: required_u32(file, "vokra.irodori.dit.num_heads")?,
            mlp_ratio: required_f32(file, "vokra.irodori.dit.mlp_ratio")?,
            timestep_embed_dim: required_u32(file, "vokra.irodori.dit.timestep_embed_dim")?,
            adaln_rank: required_u32(file, "vokra.irodori.dit.adaln_rank")?,
            norm_eps: required_f32(file, "vokra.irodori.dit.norm_eps")?,
            dropout: required_f32(file, "vokra.irodori.dit.dropout")?,
        },
        text: IrodoriTextEncoderConfig {
            vocab_size: required_u32(file, "vokra.irodori.text.vocab_size")?,
            dim: required_u32(file, "vokra.irodori.text.dim")?,
            n_layer: required_u32(file, "vokra.irodori.text.n_layer")?,
            n_head: required_u32(file, "vokra.irodori.text.n_head")?,
            mlp_ratio: required_f32(file, "vokra.irodori.text.mlp_ratio")?,
            add_bos: required_bool(file, "vokra.irodori.text.add_bos")?,
        },
        speaker: IrodoriSpeakerEncoderConfig {
            dim: required_u32(file, "vokra.irodori.speaker.dim")?,
            n_layer: required_u32(file, "vokra.irodori.speaker.n_layer")?,
            n_head: required_u32(file, "vokra.irodori.speaker.n_head")?,
            mlp_ratio: required_f32(file, "vokra.irodori.speaker.mlp_ratio")?,
            patch_size: required_u32(file, "vokra.irodori.speaker.patch_size")?,
        },
        duration: IrodoriDurationPredictorConfig {
            enabled: required_bool(file, "vokra.irodori.duration.enabled")?,
            aux_dim: required_u32(file, "vokra.irodori.duration.aux_dim")?,
            hidden_dim: required_u32(file, "vokra.irodori.duration.hidden_dim")?,
            n_layer: required_u32(file, "vokra.irodori.duration.n_layer")?,
            n_head: required_u32(file, "vokra.irodori.duration.n_head")?,
            dropout: required_f32(file, "vokra.irodori.duration.dropout")?,
            architecture: required_string(file, "vokra.irodori.duration.architecture")?.to_owned(),
            token_init_frames: required_f32(file, "vokra.irodori.duration.token_init_frames")?,
            speaker_fusion: required_string(file, "vokra.irodori.duration.speaker_fusion")?
                .to_owned(),
        },
        sample_rate: required_u32(file, "vokra.irodori.sample_rate_hz")?,
        text_tokenizer_repo: required_string(file, "vokra.irodori.text_tokenizer_repo")?.to_owned(),
    })
}

fn required_string<'a>(file: &'a GgufFile, key: &str) -> Result<&'a str> {
    file.get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!("irodori: missing/non-string metadata `{key}`"))
        })
}

fn required_u32(file: &GgufFile, key: &str) -> Result<u32> {
    match file.get(key) {
        Some(GgufMetadataValue::U32(value)) => Ok(*value),
        _ => Err(VokraError::ModelLoad(format!(
            "irodori: missing/non-u32 metadata `{key}`"
        ))),
    }
}

fn required_f32(file: &GgufFile, key: &str) -> Result<f32> {
    match file.get(key) {
        Some(GgufMetadataValue::F32(value)) => Ok(*value),
        _ => Err(VokraError::ModelLoad(format!(
            "irodori: missing/non-f32 metadata `{key}`"
        ))),
    }
}

fn required_bool(file: &GgufFile, key: &str) -> Result<bool> {
    match file.get(key) {
        Some(GgufMetadataValue::Bool(value)) => Ok(*value),
        _ => Err(VokraError::ModelLoad(format!(
            "irodori: missing/non-bool metadata `{key}`"
        ))),
    }
}

fn validate_manifest(file: &GgufFile, cfg: &IrodoriConfig) -> Result<()> {
    let expected = expected_manifest(cfg);
    let actual: BTreeMap<String, Vec<usize>> = file
        .tensors()
        .iter()
        .map(|tensor| {
            (
                tensor.name.clone(),
                tensor
                    .dimensions
                    .iter()
                    .map(|&value| value as usize)
                    .collect(),
            )
        })
        .collect();
    if actual != expected {
        let missing: Vec<&String> = expected
            .keys()
            .filter(|name| !actual.contains_key(*name))
            .take(4)
            .collect();
        let extra: Vec<&String> = actual
            .keys()
            .filter(|name| !expected.contains_key(*name))
            .take(4)
            .collect();
        let wrong: Vec<(&String, &Vec<usize>, &Vec<usize>)> = expected
            .iter()
            .filter_map(|(name, expected_shape)| {
                actual
                    .get(name)
                    .filter(|actual_shape| *actual_shape != expected_shape)
                    .map(|actual_shape| (name, actual_shape, expected_shape))
            })
            .take(4)
            .collect();
        return Err(VokraError::ModelLoad(format!(
            "irodori: tensor manifest mismatch (expected {}, found {}); missing={missing:?}, extra={extra:?}, wrong_shape={wrong:?}",
            expected.len(),
            actual.len()
        )));
    }
    Ok(())
}

fn expected_manifest(cfg: &IrodoriConfig) -> BTreeMap<String, Vec<usize>> {
    let mut output = BTreeMap::new();
    add_dit_manifest(&mut output, &cfg.dit, &cfg.text, &cfg.speaker);
    add_encoder_manifest(
        &mut output,
        "text_encoder",
        cfg.text.dim as usize,
        cfg.text.n_layer as usize,
        cfg.text.n_head as usize,
        cfg.text.ffn_inner_dim() as usize,
    );
    output.insert(
        "text_encoder.text_embedding.weight".into(),
        vec![cfg.text.vocab_size as usize, cfg.text.dim as usize],
    );
    add_encoder_manifest(
        &mut output,
        "speaker_encoder",
        cfg.speaker.dim as usize,
        cfg.speaker.n_layer as usize,
        cfg.speaker.n_head as usize,
        cfg.speaker.ffn_inner_dim() as usize,
    );
    output.insert(
        "speaker_encoder.in_proj.weight".into(),
        vec![
            cfg.speaker.dim as usize,
            cfg.dit.patched_latent_dim() as usize,
        ],
    );
    output.insert(
        "speaker_encoder.in_proj.bias".into(),
        vec![cfg.speaker.dim as usize],
    );
    add_duration_manifest(
        &mut output,
        &cfg.duration,
        cfg.text.dim as usize,
        cfg.speaker.dim as usize,
    );
    let d = cfg.dit.model_dim as usize;
    let latent = cfg.dit.patched_latent_dim() as usize;
    output.insert(
        "cond_module.0.weight".into(),
        vec![d, cfg.dit.timestep_embed_dim as usize],
    );
    output.insert("cond_module.2.weight".into(), vec![d, d]);
    output.insert("cond_module.4.weight".into(), vec![3 * d, d]);
    output.insert("in_proj.weight".into(), vec![d, latent]);
    output.insert("in_proj.bias".into(), vec![d]);
    output.insert("out_norm.weight".into(), vec![d]);
    output.insert("out_proj.weight".into(), vec![latent, d]);
    output.insert("out_proj.bias".into(), vec![latent]);
    output.insert("speaker_norm.weight".into(), vec![cfg.speaker.dim as usize]);
    output.insert("text_norm.weight".into(), vec![cfg.text.dim as usize]);
    output
}

fn add_dit_manifest(
    output: &mut BTreeMap<String, Vec<usize>>,
    cfg: &IrodoriDitConfig,
    text: &IrodoriTextEncoderConfig,
    speaker: &IrodoriSpeakerEncoderConfig,
) {
    let d = cfg.model_dim as usize;
    let heads = cfg.num_heads as usize;
    let head_dim = d / heads;
    let f = cfg.ffn_inner_dim() as usize;
    let rank = cfg.adaln_rank as usize;
    for layer in 0..cfg.num_layers as usize {
        let prefix = format!("blocks.{layer}");
        for (suffix, shape) in [
            ("attention.gate.weight", vec![d, d]),
            ("attention.k_norm.weight", vec![heads, head_dim]),
            ("attention.q_norm.weight", vec![heads, head_dim]),
            ("attention.wk.weight", vec![d, d]),
            ("attention.wk_speaker.weight", vec![d, speaker.dim as usize]),
            ("attention.wk_text.weight", vec![d, text.dim as usize]),
            ("attention.wo.weight", vec![d, d]),
            ("attention.wq.weight", vec![d, d]),
            ("attention.wv.weight", vec![d, d]),
            ("attention.wv_speaker.weight", vec![d, speaker.dim as usize]),
            ("attention.wv_text.weight", vec![d, text.dim as usize]),
            ("mlp.w1.weight", vec![f, d]),
            ("mlp.w2.weight", vec![d, f]),
            ("mlp.w3.weight", vec![f, d]),
        ] {
            output.insert(format!("{prefix}.{suffix}"), shape);
        }
        for adaln in ["attention_adaln", "mlp_adaln"] {
            for axis in ["shift", "scale", "gate"] {
                output.insert(
                    format!("{prefix}.{adaln}.{axis}_down.weight"),
                    vec![rank, d],
                );
                output.insert(format!("{prefix}.{adaln}.{axis}_up.weight"), vec![d, rank]);
                output.insert(format!("{prefix}.{adaln}.{axis}_up.bias"), vec![d]);
            }
        }
    }
}

fn add_encoder_manifest(
    output: &mut BTreeMap<String, Vec<usize>>,
    prefix: &str,
    dim: usize,
    layers: usize,
    heads: usize,
    ffn: usize,
) {
    let head_dim = dim / heads;
    for layer in 0..layers {
        let block = format!("{prefix}.blocks.{layer}");
        for (suffix, shape) in [
            ("attention.gate.weight", vec![dim, dim]),
            ("attention.k_norm.weight", vec![heads, head_dim]),
            ("attention.q_norm.weight", vec![heads, head_dim]),
            ("attention.wk.weight", vec![dim, dim]),
            ("attention.wo.weight", vec![dim, dim]),
            ("attention.wq.weight", vec![dim, dim]),
            ("attention.wv.weight", vec![dim, dim]),
            ("attention_norm.weight", vec![dim]),
            ("mlp.w1.weight", vec![ffn, dim]),
            ("mlp.w2.weight", vec![dim, ffn]),
            ("mlp.w3.weight", vec![ffn, dim]),
            ("mlp_norm.weight", vec![dim]),
        ] {
            output.insert(format!("{block}.{suffix}"), shape);
        }
    }
}

fn add_duration_manifest(
    output: &mut BTreeMap<String, Vec<usize>>,
    cfg: &IrodoriDurationPredictorConfig,
    text_dim: usize,
    speaker_dim: usize,
) {
    let d = cfg.hidden_dim as usize;
    output.insert("duration_predictor.null_speaker".into(), vec![speaker_dim]);
    output.insert(
        "duration_predictor.token_input_proj.weight".into(),
        vec![d, text_dim],
    );
    output.insert("duration_predictor.token_input_proj.bias".into(), vec![d]);
    for layer in 0..cfg.n_layer as usize {
        let prefix = format!("duration_predictor.token_blocks.{layer}");
        for (suffix, shape) in [
            ("mlp.w1.weight", vec![d, d]),
            ("mlp.w2.weight", vec![d, d]),
            ("mlp.w3.weight", vec![d, d]),
            ("modulation.bias", vec![3 * d]),
            ("modulation.weight", vec![3 * d, speaker_dim]),
            ("norm.weight", vec![d]),
        ] {
            output.insert(format!("{prefix}.{suffix}"), shape);
        }
    }
    output.insert("duration_predictor.token_out_norm.weight".into(), vec![d]);
    output.insert(
        "duration_predictor.token_out_proj.weight".into(),
        vec![1, d],
    );
    output.insert("duration_predictor.token_out_proj.bias".into(), vec![1]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_manifest_has_all_official_tensors() {
        let manifest = expected_manifest(&IrodoriConfig::irodori_500m_v3());
        assert_eq!(manifest.len(), EXPECTED_TENSOR_COUNT);
        assert_eq!(manifest["blocks.0.attention.wk_text.weight"], [1280, 512]);
        assert_eq!(manifest["text_encoder.blocks.0.mlp.w1.weight"], [1331, 512]);
        assert_eq!(
            manifest["speaker_encoder.blocks.7.mlp.w2.weight"],
            [768, 1996]
        );
        assert_eq!(
            manifest["duration_predictor.token_blocks.2.modulation.weight"],
            [3072, 768]
        );
    }

    #[test]
    fn text_block_rejects_missing_weights() {
        let cfg = IrodoriTextEncoderConfig::tiny_for_tests();
        let empty = IrodoriTextBlockWeights {
            attention_norm: Vec::new(),
            wq: Vec::new(),
            wk: Vec::new(),
            wv: Vec::new(),
            q_norm: Vec::new(),
            k_norm: Vec::new(),
            gate: Vec::new(),
            wo: Vec::new(),
            mlp_norm: Vec::new(),
            w1: Vec::new(),
            w2: Vec::new(),
            w3: Vec::new(),
        };
        let error = irodori_text_block_forward(&cfg, &empty, &[0.0; 16], &[true])
            .expect_err("empty weights must fail");
        assert!(error.to_string().contains("attention_norm"));
    }
}
