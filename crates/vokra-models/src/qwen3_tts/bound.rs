//! Strict real-checkpoint binding and the native Qwen3-TTS decoder block.
//!
//! The older types in the parent module predate inspection of the official
//! safetensors manifest and remain only as deterministic shape fixtures.  In
//! particular, the real checkpoint has bias-free attention with per-head
//! Q/K RMSNorm, one talker codec embedding, and fifteen code-predictor
//! embedding/head pairs.  This module is the production contract: it checks
//! all 478 tensors in the official 0.6B/1.7B layout without eagerly decoding
//! the complete checkpoint, and decodes one transformer block on demand.

use std::collections::BTreeMap;

use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use super::{Qwen3TtsCodePredictorConfig, Qwen3TtsConfig, Qwen3TtsTalkerConfig};

const KEY_SAMPLE_RATE: &str = "vokra.qwen3_tts.sample_rate";
const KEY_SPEAKER_EMBED_DIM: &str = "vokra.qwen3_tts.speaker_embed_dim";
const KEY_TALKER_HIDDEN_DIM: &str = "vokra.qwen3_tts.talker.hidden_dim";
const KEY_TALKER_N_LAYER: &str = "vokra.qwen3_tts.talker.n_layer";
const KEY_TALKER_N_HEAD: &str = "vokra.qwen3_tts.talker.n_head";
const KEY_TALKER_N_HEAD_KV: &str = "vokra.qwen3_tts.talker.n_head_kv";
const KEY_TALKER_HEAD_DIM: &str = "vokra.qwen3_tts.talker.head_dim";
const KEY_TALKER_FFN_DIM: &str = "vokra.qwen3_tts.talker.ffn_dim";
const KEY_TALKER_VOCAB_SIZE: &str = "vokra.qwen3_tts.talker.vocab_size";
const KEY_TALKER_TEXT_VOCAB_SIZE: &str = "vokra.qwen3_tts.talker.text_vocab_size";
const KEY_TALKER_MAX_POSITIONS: &str = "vokra.qwen3_tts.talker.max_position_embeddings";
const KEY_TALKER_ROPE_BASE: &str = "vokra.qwen3_tts.talker.rope_base";
const KEY_TALKER_RMS_NORM_EPS: &str = "vokra.qwen3_tts.talker.rms_norm_eps";
const KEY_TALKER_POS_ID_PER_SEC: &str = "vokra.qwen3_tts.talker.position_id_per_seconds";
const KEY_TALKER_NUM_CODE_GROUPS: &str = "vokra.qwen3_tts.talker.num_code_groups";
const KEY_TALKER_TEXT_HIDDEN_SIZE: &str = "vokra.qwen3_tts.talker.text_hidden_size";
const KEY_CP_HIDDEN_DIM: &str = "vokra.qwen3_tts.code_predictor.hidden_dim";
const KEY_CP_N_LAYER: &str = "vokra.qwen3_tts.code_predictor.n_layer";
const KEY_CP_N_HEAD: &str = "vokra.qwen3_tts.code_predictor.n_head";
const KEY_CP_N_HEAD_KV: &str = "vokra.qwen3_tts.code_predictor.n_head_kv";
const KEY_CP_HEAD_DIM: &str = "vokra.qwen3_tts.code_predictor.head_dim";
const KEY_CP_FFN_DIM: &str = "vokra.qwen3_tts.code_predictor.ffn_dim";
const KEY_CP_VOCAB_SIZE: &str = "vokra.qwen3_tts.code_predictor.vocab_size";
const KEY_CP_ROPE_BASE: &str = "vokra.qwen3_tts.code_predictor.rope_base";
const KEY_CP_RMS_NORM_EPS: &str = "vokra.qwen3_tts.code_predictor.rms_norm_eps";
const KEY_CP_NUM_CODE_GROUPS: &str = "vokra.qwen3_tts.code_predictor.num_code_groups";

/// A strictly validated Qwen3-TTS main-model checkpoint.
///
/// This handle intentionally stores only the manifest and resolved config.
/// Callers retain the [`GgufFile`] (preferably mmap-backed) and pass it to the
/// block loaders.  Eagerly widening the complete BF16 checkpoint would add a
/// multi-gigabyte duplicate before generation even begins.
#[derive(Debug, Clone)]
pub struct Qwen3TtsCheckpoint {
    config: Qwen3TtsConfig,
    model_name: String,
    weight_license: LicenseClass,
    tensor_count: usize,
}

impl Qwen3TtsCheckpoint {
    /// Validates the arch, every topology metadata key, and the exact official
    /// 478-tensor name/shape manifest.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let arch = required_string(file, chunks::KEY_MODEL_ARCH)?;
        if arch != super::EXPECTED_ARCH {
            return Err(VokraError::ModelLoad(format!(
                "qwen3_tts: unsupported `{}`={arch:?}; expected {:?}",
                chunks::KEY_MODEL_ARCH,
                super::EXPECTED_ARCH
            )));
        }
        let model_name = required_string(file, chunks::KEY_MODEL_NAME)?.to_owned();
        if model_name != "qwen3-tts-12hz-0.6b-base" {
            return Err(VokraError::ModelLoad(format!(
                "qwen3_tts: unsupported `{}`={model_name:?}; this strict manifest is pinned to the official 12Hz 0.6B-Base release. The 1.7B code predictor adds a small-to-MTP projection and widens its embedding input axis, so it must not be admitted under the 0.6B tensor contract",
                chunks::KEY_MODEL_NAME
            )));
        }

        let config = config_from_gguf(file)?;
        config.validate_for_forward().map_err(|error| {
            VokraError::ModelLoad(format!(
                "qwen3_tts: stamped topology is not forward-safe: {error}"
            ))
        })?;
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

    #[must_use]
    /// Returns the topology parsed from the GGUF metadata.
    pub fn config(&self) -> &Qwen3TtsConfig {
        &self.config
    }

    #[must_use]
    /// Returns the exact checkpoint variant name admitted by the binder.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    #[must_use]
    /// Returns the fail-closed weight-license class stamped in the GGUF.
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    #[must_use]
    /// Returns the number of tensors validated against the strict manifest.
    pub const fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    /// Primary text-to-PCM entry point for a bound real checkpoint.
    ///
    /// Loading and individual talker/code-predictor blocks are real.  Full
    /// synthesis remains loud until the Qwen2 BPE sidecars, autoregressive
    /// generation loop, and separate 12-Hz neural speech-tokenizer decoder
    /// are bound.  The RVQ table fold alone is not a waveform decoder.
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "qwen3_tts synthesize: text is empty".to_owned(),
            ));
        }
        Err(VokraError::NotImplemented(
            "qwen3_tts synthesize: the real 478-tensor main checkpoint is bound and the native talker/code-predictor decoder block is available, but end-to-end PCM still requires three independently gated pieces: embedded Qwen2 BPE vocab+merges, the multi-codebook autoregressive generation loop, and the separate Qwen3-TTS-Tokenizer-12Hz neural decoder (682 MB sibling artifact). vokra_ops::qwen3_tts_codec only folds RVQ tables to codec features and is not substituted for that neural waveform decoder.",
        ))
    }

    /// Decodes one talker layer after rechecking its names and shapes.
    pub fn load_talker_block(
        &self,
        file: &GgufFile,
        layer: usize,
    ) -> Result<Qwen3TtsBoundBlockWeights> {
        if layer >= self.config.talker.n_layer as usize {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts: talker layer {layer} out of range 0..{}",
                self.config.talker.n_layer
            )));
        }
        load_block(
            file,
            &format!("talker.model.layers.{layer}"),
            &self.config.talker,
        )
    }

    /// Decodes one code-predictor layer after rechecking its names and shapes.
    pub fn load_code_predictor_block(
        &self,
        file: &GgufFile,
        layer: usize,
    ) -> Result<Qwen3TtsBoundBlockWeights> {
        if layer >= self.config.code_predictor.n_layer as usize {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts: code-predictor layer {layer} out of range 0..{}",
                self.config.code_predictor.n_layer
            )));
        }
        load_block(
            file,
            &format!("talker.code_predictor.model.layers.{layer}"),
            &self.config.code_predictor,
        )
    }
}

trait BlockConfig {
    fn hidden_dim(&self) -> usize;
    fn n_head(&self) -> usize;
    fn n_head_kv(&self) -> usize;
    fn head_dim(&self) -> usize;
    fn ffn_dim(&self) -> usize;
    fn rope_base(&self) -> f32;
    fn rms_norm_eps(&self) -> f32;
}

impl BlockConfig for Qwen3TtsTalkerConfig {
    fn hidden_dim(&self) -> usize {
        self.hidden_dim as usize
    }
    fn n_head(&self) -> usize {
        self.n_head as usize
    }
    fn n_head_kv(&self) -> usize {
        self.n_head_kv as usize
    }
    fn head_dim(&self) -> usize {
        self.head_dim as usize
    }
    fn ffn_dim(&self) -> usize {
        self.ffn_dim as usize
    }
    fn rope_base(&self) -> f32 {
        self.rope_base
    }
    fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }
}

impl BlockConfig for Qwen3TtsCodePredictorConfig {
    fn hidden_dim(&self) -> usize {
        self.hidden_dim as usize
    }
    fn n_head(&self) -> usize {
        self.n_head as usize
    }
    fn n_head_kv(&self) -> usize {
        self.n_head_kv as usize
    }
    fn head_dim(&self) -> usize {
        self.head_dim as usize
    }
    fn ffn_dim(&self) -> usize {
        self.ffn_dim as usize
    }
    fn rope_base(&self) -> f32 {
        self.rope_base
    }
    fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }
}

/// Real Qwen3-TTS decoder-layer weights.  Projection matrices retain the
/// upstream `[out, in]` row-major layout.
#[derive(Debug, Clone)]
pub struct Qwen3TtsBoundBlockWeights {
    /// Pre-attention RMSNorm scale.
    pub input_layernorm: Vec<f32>,
    /// Bias-free query projection in upstream `[out, in]` layout.
    pub q_proj: Vec<f32>,
    /// Per-head query RMSNorm scale.
    pub q_norm: Vec<f32>,
    /// Bias-free key projection in upstream `[out, in]` layout.
    pub k_proj: Vec<f32>,
    /// Per-head key RMSNorm scale.
    pub k_norm: Vec<f32>,
    /// Bias-free value projection in upstream `[out, in]` layout.
    pub v_proj: Vec<f32>,
    /// Bias-free attention output projection in upstream `[out, in]` layout.
    pub o_proj: Vec<f32>,
    /// Pre-MLP RMSNorm scale.
    pub post_attention_layernorm: Vec<f32>,
    /// SwiGLU gate projection in upstream `[out, in]` layout.
    pub gate_proj: Vec<f32>,
    /// SwiGLU value projection in upstream `[out, in]` layout.
    pub up_proj: Vec<f32>,
    /// SwiGLU output projection in upstream `[out, in]` layout.
    pub down_proj: Vec<f32>,
}

fn load_block<C: BlockConfig>(
    file: &GgufFile,
    prefix: &str,
    cfg: &C,
) -> Result<Qwen3TtsBoundBlockWeights> {
    let d = cfg.hidden_dim();
    let q = cfg.n_head() * cfg.head_dim();
    let kv = cfg.n_head_kv() * cfg.head_dim();
    let h = cfg.head_dim();
    let f = cfg.ffn_dim();
    Ok(Qwen3TtsBoundBlockWeights {
        input_layernorm: tensor(file, &format!("{prefix}.input_layernorm.weight"), &[d])?,
        q_proj: tensor(file, &format!("{prefix}.self_attn.q_proj.weight"), &[q, d])?,
        q_norm: tensor(file, &format!("{prefix}.self_attn.q_norm.weight"), &[h])?,
        k_proj: tensor(file, &format!("{prefix}.self_attn.k_proj.weight"), &[kv, d])?,
        k_norm: tensor(file, &format!("{prefix}.self_attn.k_norm.weight"), &[h])?,
        v_proj: tensor(file, &format!("{prefix}.self_attn.v_proj.weight"), &[kv, d])?,
        o_proj: tensor(file, &format!("{prefix}.self_attn.o_proj.weight"), &[d, q])?,
        post_attention_layernorm: tensor(
            file,
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[d],
        )?,
        gate_proj: tensor(file, &format!("{prefix}.mlp.gate_proj.weight"), &[f, d])?,
        up_proj: tensor(file, &format!("{prefix}.mlp.up_proj.weight"), &[f, d])?,
        down_proj: tensor(file, &format!("{prefix}.mlp.down_proj.weight"), &[d, f])?,
    })
}

fn tensor(file: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!("qwen3_tts: required tensor `{name}` is missing"))
    })?;
    let actual: Vec<usize> = info.dimensions.iter().map(|&x| x as usize).collect();
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "qwen3_tts: tensor `{name}` shape {actual:?}, expected {expected:?}"
        )));
    }
    file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!("qwen3_tts: tensor `{name}` decode failed: {error}"))
    })
}

/// Runs one real talker decoder layer for batch size one.
///
/// `positions` is `[temporal, height, width]` per token and exercises the
/// official interleaved mRoPE section layout `[24, 20, 20]` scaled to the
/// current head width.  The released model uses head_dim=128, for which the
/// sections sum to 64 exactly.
pub fn qwen3_tts_talker_block_forward(
    cfg: &Qwen3TtsTalkerConfig,
    weights: &Qwen3TtsBoundBlockWeights,
    hidden: &[f32],
    positions: &[[u32; 3]],
) -> Result<Vec<f32>> {
    let sections = scaled_mrope_sections(cfg.head_dim as usize)?;
    block_forward(
        cfg,
        weights,
        hidden,
        positions.len(),
        Rope::Interleaved {
            positions,
            sections,
        },
    )
}

/// Runs one real code-predictor decoder layer for batch size one.
pub fn qwen3_tts_code_predictor_block_forward(
    cfg: &Qwen3TtsCodePredictorConfig,
    weights: &Qwen3TtsBoundBlockWeights,
    hidden: &[f32],
    positions: &[u32],
) -> Result<Vec<f32>> {
    block_forward(
        cfg,
        weights,
        hidden,
        positions.len(),
        Rope::Standard(positions),
    )
}

enum Rope<'a> {
    Standard(&'a [u32]),
    Interleaved {
        positions: &'a [[u32; 3]],
        sections: [usize; 3],
    },
}

fn block_forward<C: BlockConfig>(
    cfg: &C,
    weights: &Qwen3TtsBoundBlockWeights,
    hidden: &[f32],
    seq: usize,
    rope: Rope<'_>,
) -> Result<Vec<f32>> {
    let d = cfg.hidden_dim();
    let nh = cfg.n_head();
    let nkv = cfg.n_head_kv();
    let hd = cfg.head_dim();
    let ffn = cfg.ffn_dim();
    if seq == 0 || hidden.len() != seq * d {
        return Err(VokraError::InvalidArgument(format!(
            "qwen3_tts block: hidden len {} must equal non-zero seq {seq} * hidden {d}",
            hidden.len()
        )));
    }
    if nh == 0 || nkv == 0 || nh % nkv != 0 || hd % 2 != 0 {
        return Err(VokraError::InvalidArgument(
            "qwen3_tts block: invalid GQA/RoPE axes".to_owned(),
        ));
    }
    check_block_weight_lengths(cfg, weights)?;

    let normed = rms_norm_rows(hidden, &weights.input_layernorm, seq, d, cfg.rms_norm_eps());
    let mut q = linear_rows(&normed, &weights.q_proj, seq, d, nh * hd);
    let mut k = linear_rows(&normed, &weights.k_proj, seq, d, nkv * hd);
    let v = linear_rows(&normed, &weights.v_proj, seq, d, nkv * hd);
    rms_norm_heads(&mut q, &weights.q_norm, seq, nh, hd, cfg.rms_norm_eps());
    rms_norm_heads(&mut k, &weights.k_norm, seq, nkv, hd, cfg.rms_norm_eps());
    apply_rope(&mut q, seq, nh, hd, cfg.rope_base(), &rope);
    apply_rope(&mut k, seq, nkv, hd, cfg.rope_base(), &rope);

    let groups = nh / nkv;
    let mut attended = vec![0.0f32; seq * nh * hd];
    let scale = 1.0 / (hd as f32).sqrt();
    let mut scores = vec![0.0f32; seq];
    for t in 0..seq {
        for qh in 0..nh {
            let kvh = qh / groups;
            let qoff = (t * nh + qh) * hd;
            let mut max_score = f32::NEG_INFINITY;
            for (s, score_slot) in scores.iter_mut().enumerate().take(t + 1) {
                let koff = (s * nkv + kvh) * hd;
                let mut dot = 0.0;
                for j in 0..hd {
                    dot += q[qoff + j] * k[koff + j];
                }
                let score = dot * scale;
                *score_slot = score;
                max_score = max_score.max(score);
            }
            let mut denom = 0.0;
            for score in scores.iter_mut().take(t + 1) {
                *score = (*score - max_score).exp();
                denom += *score;
            }
            let out_off = (t * nh + qh) * hd;
            for (s, &score) in scores.iter().enumerate().take(t + 1) {
                let probability = score / denom;
                let voff = (s * nkv + kvh) * hd;
                for j in 0..hd {
                    attended[out_off + j] += probability * v[voff + j];
                }
            }
        }
    }

    let attn_out = linear_rows(&attended, &weights.o_proj, seq, nh * hd, d);
    let mut residual: Vec<f32> = hidden
        .iter()
        .zip(attn_out.iter())
        .map(|(a, b)| a + b)
        .collect();
    let ffn_input = rms_norm_rows(
        &residual,
        &weights.post_attention_layernorm,
        seq,
        d,
        cfg.rms_norm_eps(),
    );
    let gate = linear_rows(&ffn_input, &weights.gate_proj, seq, d, ffn);
    let up = linear_rows(&ffn_input, &weights.up_proj, seq, d, ffn);
    let activated: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect();
    let down = linear_rows(&activated, &weights.down_proj, seq, ffn, d);
    for (dst, add) in residual.iter_mut().zip(down) {
        *dst += add;
    }
    Ok(residual)
}

fn check_block_weight_lengths<C: BlockConfig>(
    cfg: &C,
    w: &Qwen3TtsBoundBlockWeights,
) -> Result<()> {
    let d = cfg.hidden_dim();
    let q = cfg.n_head() * cfg.head_dim();
    let kv = cfg.n_head_kv() * cfg.head_dim();
    let h = cfg.head_dim();
    let f = cfg.ffn_dim();
    for (name, got, expected) in [
        ("input_layernorm", w.input_layernorm.len(), d),
        ("q_proj", w.q_proj.len(), q * d),
        ("q_norm", w.q_norm.len(), h),
        ("k_proj", w.k_proj.len(), kv * d),
        ("k_norm", w.k_norm.len(), h),
        ("v_proj", w.v_proj.len(), kv * d),
        ("o_proj", w.o_proj.len(), d * q),
        (
            "post_attention_layernorm",
            w.post_attention_layernorm.len(),
            d,
        ),
        ("gate_proj", w.gate_proj.len(), f * d),
        ("up_proj", w.up_proj.len(), f * d),
        ("down_proj", w.down_proj.len(), d * f),
    ] {
        if got != expected {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts block: {name}.len()={got}, expected {expected}"
            )));
        }
    }
    Ok(())
}

fn linear_rows(x: &[f32], weight: &[f32], rows: usize, input: usize, output: usize) -> Vec<f32> {
    let mut y = vec![0.0; rows * output];
    for r in 0..rows {
        for o in 0..output {
            let mut sum = 0.0;
            let woff = o * input;
            let xoff = r * input;
            for i in 0..input {
                sum += x[xoff + i] * weight[woff + i];
            }
            y[r * output + o] = sum;
        }
    }
    y
}

fn rms_norm_rows(x: &[f32], weight: &[f32], rows: usize, width: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    for r in 0..rows {
        let off = r * width;
        let variance = x[off..off + width].iter().map(|v| v * v).sum::<f32>() / width as f32;
        let inv = 1.0 / (variance + eps).sqrt();
        for i in 0..width {
            y[off + i] = x[off + i] * inv * weight[i];
        }
    }
    y
}

fn rms_norm_heads(x: &mut [f32], weight: &[f32], seq: usize, heads: usize, hd: usize, eps: f32) {
    for t in 0..seq {
        for head in 0..heads {
            let off = (t * heads + head) * hd;
            let variance = x[off..off + hd].iter().map(|v| v * v).sum::<f32>() / hd as f32;
            let inv = 1.0 / (variance + eps).sqrt();
            for i in 0..hd {
                x[off + i] = x[off + i] * inv * weight[i];
            }
        }
    }
}

fn apply_rope(x: &mut [f32], seq: usize, heads: usize, hd: usize, base: f32, rope: &Rope<'_>) {
    let half = hd / 2;
    for t in 0..seq {
        for head in 0..heads {
            let off = (t * heads + head) * hd;
            for i in 0..half {
                let position = match rope {
                    Rope::Standard(positions) => positions[t],
                    Rope::Interleaved {
                        positions,
                        sections,
                    } => {
                        let modality = interleaved_modality(i, *sections);
                        positions[t][modality]
                    }
                } as f32;
                let inv_freq = 1.0 / base.powf((2 * i) as f32 / hd as f32);
                let angle = position * inv_freq;
                let (sin, cos) = angle.sin_cos();
                let a = x[off + i];
                let b = x[off + half + i];
                x[off + i] = a * cos - b * sin;
                x[off + half + i] = b * cos + a * sin;
            }
        }
    }
}

fn interleaved_modality(index: usize, sections: [usize; 3]) -> usize {
    let interleaved = sections[1] * 3;
    if index < interleaved { index % 3 } else { 0 }
}

fn scaled_mrope_sections(head_dim: usize) -> Result<[usize; 3]> {
    if head_dim % 128 != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "qwen3_tts talker: head_dim {head_dim} cannot scale official mRoPE sections [24,20,20] exactly"
        )));
    }
    let scale = head_dim / 128;
    Ok([24 * scale, 20 * scale, 20 * scale])
}

fn config_from_gguf(file: &GgufFile) -> Result<Qwen3TtsConfig> {
    Ok(Qwen3TtsConfig {
        sample_rate: required_u32(file, KEY_SAMPLE_RATE)?,
        speaker_embed_dim: required_u32(file, KEY_SPEAKER_EMBED_DIM)?,
        talker: Qwen3TtsTalkerConfig {
            hidden_dim: required_u32(file, KEY_TALKER_HIDDEN_DIM)?,
            n_layer: required_u32(file, KEY_TALKER_N_LAYER)?,
            n_head: required_u32(file, KEY_TALKER_N_HEAD)?,
            n_head_kv: required_u32(file, KEY_TALKER_N_HEAD_KV)?,
            head_dim: required_u32(file, KEY_TALKER_HEAD_DIM)?,
            ffn_dim: required_u32(file, KEY_TALKER_FFN_DIM)?,
            vocab_size: required_u32(file, KEY_TALKER_VOCAB_SIZE)?,
            text_vocab_size: required_u32(file, KEY_TALKER_TEXT_VOCAB_SIZE)?,
            max_position_embeddings: required_u32(file, KEY_TALKER_MAX_POSITIONS)?,
            rope_base: required_f32(file, KEY_TALKER_ROPE_BASE)?,
            rms_norm_eps: required_f32(file, KEY_TALKER_RMS_NORM_EPS)?,
            position_id_per_seconds: required_u32(file, KEY_TALKER_POS_ID_PER_SEC)?,
            num_code_groups: required_u32(file, KEY_TALKER_NUM_CODE_GROUPS)?,
            text_hidden_size: required_u32(file, KEY_TALKER_TEXT_HIDDEN_SIZE)?,
        },
        code_predictor: Qwen3TtsCodePredictorConfig {
            hidden_dim: required_u32(file, KEY_CP_HIDDEN_DIM)?,
            n_layer: required_u32(file, KEY_CP_N_LAYER)?,
            n_head: required_u32(file, KEY_CP_N_HEAD)?,
            n_head_kv: required_u32(file, KEY_CP_N_HEAD_KV)?,
            head_dim: required_u32(file, KEY_CP_HEAD_DIM)?,
            ffn_dim: required_u32(file, KEY_CP_FFN_DIM)?,
            vocab_size: required_u32(file, KEY_CP_VOCAB_SIZE)?,
            rope_base: required_f32(file, KEY_CP_ROPE_BASE)?,
            rms_norm_eps: required_f32(file, KEY_CP_RMS_NORM_EPS)?,
            num_code_groups: required_u32(file, KEY_CP_NUM_CODE_GROUPS)?,
        },
    })
}

fn required_string<'a>(file: &'a GgufFile, key: &str) -> Result<&'a str> {
    file.get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!("qwen3_tts: missing/non-string metadata `{key}`"))
        })
}

fn required_u32(file: &GgufFile, key: &str) -> Result<u32> {
    match file.get(key) {
        Some(GgufMetadataValue::U32(value)) => Ok(*value),
        _ => Err(VokraError::ModelLoad(format!(
            "qwen3_tts: missing/non-u32 metadata `{key}`"
        ))),
    }
}

fn required_f32(file: &GgufFile, key: &str) -> Result<f32> {
    match file.get(key) {
        Some(GgufMetadataValue::F32(value)) => Ok(*value),
        _ => Err(VokraError::ModelLoad(format!(
            "qwen3_tts: missing/non-f32 metadata `{key}`"
        ))),
    }
}

fn validate_manifest(file: &GgufFile, cfg: &Qwen3TtsConfig) -> Result<()> {
    let expected = expected_manifest(cfg);
    let actual: BTreeMap<String, Vec<usize>> = file
        .tensors()
        .iter()
        .map(|tensor| {
            (
                tensor.name.clone(),
                tensor.dimensions.iter().map(|&x| x as usize).collect(),
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
            .filter_map(|(name, shape)| {
                actual
                    .get(name)
                    .filter(|actual| *actual != shape)
                    .map(|actual| (name, actual, shape))
            })
            .take(4)
            .collect();
        return Err(VokraError::ModelLoad(format!(
            "qwen3_tts: tensor manifest mismatch (expected {}, found {}); missing={missing:?}, extra={extra:?}, wrong_shape={wrong:?}",
            expected.len(),
            actual.len()
        )));
    }
    Ok(())
}

fn expected_manifest(cfg: &Qwen3TtsConfig) -> BTreeMap<String, Vec<usize>> {
    let mut out = BTreeMap::new();
    add_speaker_manifest(&mut out, cfg.speaker_embed_dim as usize);
    add_stack_manifest(
        &mut out,
        "talker.model",
        &cfg.talker,
        cfg.talker.n_layer as usize,
    );
    let t = &cfg.talker;
    out.insert(
        "talker.model.text_embedding.weight".into(),
        vec![t.text_vocab_size as usize, t.text_hidden_size as usize],
    );
    out.insert(
        "talker.model.codec_embedding.weight".into(),
        vec![t.vocab_size as usize, t.hidden_dim as usize],
    );
    out.insert(
        "talker.model.norm.weight".into(),
        vec![t.hidden_dim as usize],
    );
    out.insert(
        "talker.text_projection.linear_fc1.weight".into(),
        vec![t.text_hidden_size as usize, t.text_hidden_size as usize],
    );
    out.insert(
        "talker.text_projection.linear_fc1.bias".into(),
        vec![t.text_hidden_size as usize],
    );
    out.insert(
        "talker.text_projection.linear_fc2.weight".into(),
        vec![t.hidden_dim as usize, t.text_hidden_size as usize],
    );
    out.insert(
        "talker.text_projection.linear_fc2.bias".into(),
        vec![t.hidden_dim as usize],
    );
    out.insert(
        "talker.codec_head.weight".into(),
        vec![t.vocab_size as usize, t.hidden_dim as usize],
    );

    let cp = &cfg.code_predictor;
    add_stack_manifest(
        &mut out,
        "talker.code_predictor.model",
        cp,
        cp.n_layer as usize,
    );
    out.insert(
        "talker.code_predictor.model.norm.weight".into(),
        vec![cp.hidden_dim as usize],
    );
    for group in 0..cp.num_code_groups.saturating_sub(1) as usize {
        out.insert(
            format!("talker.code_predictor.model.codec_embedding.{group}.weight"),
            vec![cp.vocab_size as usize, cp.hidden_dim as usize],
        );
        out.insert(
            format!("talker.code_predictor.lm_head.{group}.weight"),
            vec![cp.vocab_size as usize, cp.hidden_dim as usize],
        );
    }
    out
}

fn add_stack_manifest<C: BlockConfig>(
    out: &mut BTreeMap<String, Vec<usize>>,
    prefix: &str,
    cfg: &C,
    layers: usize,
) {
    let d = cfg.hidden_dim();
    let q = cfg.n_head() * cfg.head_dim();
    let kv = cfg.n_head_kv() * cfg.head_dim();
    let h = cfg.head_dim();
    let f = cfg.ffn_dim();
    for layer in 0..layers {
        let p = format!("{prefix}.layers.{layer}");
        for (suffix, shape) in [
            ("input_layernorm.weight", vec![d]),
            ("self_attn.q_proj.weight", vec![q, d]),
            ("self_attn.q_norm.weight", vec![h]),
            ("self_attn.k_proj.weight", vec![kv, d]),
            ("self_attn.k_norm.weight", vec![h]),
            ("self_attn.v_proj.weight", vec![kv, d]),
            ("self_attn.o_proj.weight", vec![d, q]),
            ("post_attention_layernorm.weight", vec![d]),
            ("mlp.gate_proj.weight", vec![f, d]),
            ("mlp.up_proj.weight", vec![f, d]),
            ("mlp.down_proj.weight", vec![d, f]),
        ] {
            out.insert(format!("{p}.{suffix}"), shape);
        }
    }
}

fn add_speaker_manifest(out: &mut BTreeMap<String, Vec<usize>>, embedding_dim: usize) {
    out.insert(
        "speaker_encoder.blocks.0.conv.weight".into(),
        vec![512, 128, 5],
    );
    out.insert("speaker_encoder.blocks.0.conv.bias".into(), vec![512]);
    for block in 1..=3 {
        out.insert(
            format!("speaker_encoder.blocks.{block}.tdnn1.conv.weight"),
            vec![512, 512, 1],
        );
        out.insert(
            format!("speaker_encoder.blocks.{block}.tdnn1.conv.bias"),
            vec![512],
        );
        for sub in 0..7 {
            out.insert(
                format!("speaker_encoder.blocks.{block}.res2net_block.blocks.{sub}.conv.weight"),
                vec![64, 64, 3],
            );
            out.insert(
                format!("speaker_encoder.blocks.{block}.res2net_block.blocks.{sub}.conv.bias"),
                vec![64],
            );
        }
        out.insert(
            format!("speaker_encoder.blocks.{block}.tdnn2.conv.weight"),
            vec![512, 512, 1],
        );
        out.insert(
            format!("speaker_encoder.blocks.{block}.tdnn2.conv.bias"),
            vec![512],
        );
        out.insert(
            format!("speaker_encoder.blocks.{block}.se_block.conv1.weight"),
            vec![128, 512, 1],
        );
        out.insert(
            format!("speaker_encoder.blocks.{block}.se_block.conv1.bias"),
            vec![128],
        );
        out.insert(
            format!("speaker_encoder.blocks.{block}.se_block.conv2.weight"),
            vec![512, 128, 1],
        );
        out.insert(
            format!("speaker_encoder.blocks.{block}.se_block.conv2.bias"),
            vec![512],
        );
    }
    for (name, shape) in [
        ("speaker_encoder.mfa.conv.weight", vec![1536, 1536, 1]),
        ("speaker_encoder.mfa.conv.bias", vec![1536]),
        ("speaker_encoder.asp.tdnn.conv.weight", vec![128, 4608, 1]),
        ("speaker_encoder.asp.tdnn.conv.bias", vec![128]),
        ("speaker_encoder.asp.conv.weight", vec![1536, 128, 1]),
        ("speaker_encoder.asp.conv.bias", vec![1536]),
        ("speaker_encoder.fc.weight", vec![embedding_dim, 3072, 1]),
        ("speaker_encoder.fc.bias", vec![embedding_dim]),
    ] {
        out.insert(name.into(), shape);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_manifest_has_all_478_official_tensors() {
        let manifest = expected_manifest(&Qwen3TtsConfig::qwen3_tts_0_6b_base());
        assert_eq!(manifest.len(), 478);
        assert_eq!(
            manifest["talker.model.text_embedding.weight"],
            [151_936, 2048]
        );
        assert_eq!(
            manifest["talker.model.layers.0.self_attn.q_proj.weight"],
            [2048, 1024]
        );
        assert_eq!(
            manifest["talker.code_predictor.lm_head.14.weight"],
            [2048, 1024]
        );
        assert!(!manifest.contains_key("talker.model.layers.0.self_attn.q_proj.bias"));
    }

    #[test]
    fn interleaved_mrope_assigns_official_modalities() {
        let sections = scaled_mrope_sections(128).unwrap();
        assert_eq!(sections, [24, 20, 20]);
        assert_eq!(interleaved_modality(0, sections), 0);
        assert_eq!(interleaved_modality(1, sections), 1);
        assert_eq!(interleaved_modality(2, sections), 2);
        assert_eq!(interleaved_modality(59, sections), 2);
        assert_eq!(interleaved_modality(60, sections), 0);
        assert_eq!(interleaved_modality(63, sections), 0);
    }

    #[test]
    fn block_forward_rejects_wrong_weight_shape() {
        let cfg = Qwen3TtsTalkerConfig::qwen3_tts_0_6b_base();
        let weights = Qwen3TtsBoundBlockWeights {
            input_layernorm: Vec::new(),
            q_proj: Vec::new(),
            q_norm: Vec::new(),
            k_proj: Vec::new(),
            k_norm: Vec::new(),
            v_proj: Vec::new(),
            o_proj: Vec::new(),
            post_attention_layernorm: Vec::new(),
            gate_proj: Vec::new(),
            up_proj: Vec::new(),
            down_proj: Vec::new(),
        };
        let err = qwen3_tts_talker_block_forward(&cfg, &weights, &vec![0.0; 1024], &[[0, 0, 0]])
            .expect_err("empty weights must fail");
        assert!(err.to_string().contains("input_layernorm"));
    }
}
