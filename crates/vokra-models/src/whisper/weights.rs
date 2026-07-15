//! Whisper weight binding: GGUF tensors → typed weight structs.
//!
//! # Tensor naming contract (M0-06-T04, shared with M0-03)
//!
//! GGUF tensor names are the **upstream Hugging Face names verbatim** (the
//! converter's [`gguf_tensor_name`] is the identity map). This module looks up
//! weights by those names:
//!
//! - encoder: `model.encoder.conv{1,2}.{weight,bias}`,
//!   `model.encoder.embed_positions.weight`,
//!   `model.encoder.layers.{i}.{self_attn.{q,k,v,out}_proj,self_attn_layer_norm,
//!   fc1,fc2,final_layer_norm}.{weight,bias}`, `model.encoder.layer_norm.*`;
//! - decoder: `model.decoder.embed_tokens.weight`,
//!   `model.decoder.embed_positions.weight`, per-layer `self_attn`,
//!   `encoder_attn`, `fc1`/`fc2` and the final `model.decoder.layer_norm.*`.
//!
//! Whisper's `k_proj` has **no bias** (both self- and cross-attention); the
//! logits head is **tied** to `model.decoder.embed_tokens.weight` (no separate
//! projection tensor). Source: openai/whisper `whisper/model.py`, HF
//! `WhisperModel`.
//!
//! # Owned f32, not a borrowed mmap view (deviation from M0-06-T05)
//!
//! `vokra-models` forbids `unsafe` (workspace lint), so a raw `&[u8]` GGUF
//! payload cannot be reinterpreted as `&[f32]` here. Weights are therefore
//! decoded into owned `Vec<f32>` through the single canonical
//! [`GgufFile::tensor_f32`] path (M1-02), which handles dense `F32` / `F16`
//! **and** dequantizes K-quants (`Q4_K` / `Q5_K` / `Q6_K`) — so a quantized
//! Whisper GGUF loads with no changes here. This also lets `nn.Linear` weights
//! be pre-transposed once from the HF `[out, in]` layout to the `[in, out]`
//! layout the row-major GEMM consumes directly. True lazy mmap (FR-LD-01 /
//! NFR-PF-11) needs an `unsafe`-allowed crate and is a documented follow-up;
//! note the GGUF reader itself already buffers the whole file (see
//! `vokra_core::gguf::reader`).

use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use super::config::WhisperConfig;

/// A HF `nn.Linear` decoded for direct row-major GEMM.
///
/// `w_t` is the transpose of the stored `[out, in]` weight, i.e. `[in, out]`
/// row-major, so `y[t, o] = bias[o] + sum_i x[t, i] * w_t[i, o]`.
pub(crate) struct Linear {
    /// Transposed weight `[in_features, out_features]`, row-major.
    pub(crate) w_t: Vec<f32>,
    /// Input width.
    pub(crate) in_features: usize,
    /// Output width.
    pub(crate) out_features: usize,
    /// Optional per-output bias (`None` for Whisper `k_proj`).
    pub(crate) bias: Option<Vec<f32>>,
}

/// A `nn.LayerNorm` (affine): `weight` = γ, `bias` = β, both width `d`.
pub(crate) struct LayerNorm {
    /// Scale γ, length `d`.
    pub(crate) gamma: Vec<f32>,
    /// Shift β, length `d`.
    pub(crate) beta: Vec<f32>,
}

/// A multi-head attention block's four projections.
pub(crate) struct Attention {
    /// Query projection (has bias).
    pub(crate) q: Linear,
    /// Key projection (**no bias** in Whisper).
    pub(crate) k: Linear,
    /// Value projection (has bias).
    pub(crate) v: Linear,
    /// Output projection (has bias).
    pub(crate) out: Linear,
}

/// One encoder block (pre-norm self-attention + pre-norm MLP).
pub(crate) struct EncoderLayer {
    /// LayerNorm before self-attention.
    pub(crate) attn_ln: LayerNorm,
    /// Bidirectional self-attention.
    pub(crate) attn: Attention,
    /// LayerNorm before the MLP.
    pub(crate) mlp_ln: LayerNorm,
    /// MLP up-projection `d → ffn_dim`.
    pub(crate) fc1: Linear,
    /// MLP down-projection `ffn_dim → d`.
    pub(crate) fc2: Linear,
}

/// One decoder block (causal self-attention → cross-attention → MLP).
pub(crate) struct DecoderLayer {
    /// LayerNorm before causal self-attention.
    pub(crate) self_ln: LayerNorm,
    /// Causal self-attention.
    pub(crate) self_attn: Attention,
    /// LayerNorm before cross-attention.
    pub(crate) cross_ln: LayerNorm,
    /// Cross-attention over the encoder output.
    pub(crate) cross_attn: Attention,
    /// LayerNorm before the MLP.
    pub(crate) mlp_ln: LayerNorm,
    /// MLP up-projection.
    pub(crate) fc1: Linear,
    /// MLP down-projection.
    pub(crate) fc2: Linear,
}

/// Encoder weights.
pub(crate) struct EncoderWeights {
    /// conv1 weight `[d_model, n_mels, 3]`.
    pub(crate) conv1_w: Vec<f32>,
    /// conv1 bias `[d_model]`.
    pub(crate) conv1_b: Vec<f32>,
    /// conv2 weight `[d_model, d_model, 3]`.
    pub(crate) conv2_w: Vec<f32>,
    /// conv2 bias `[d_model]`.
    pub(crate) conv2_b: Vec<f32>,
    /// Sinusoidal positional embedding `[n_audio_ctx, d_model]`.
    pub(crate) pos_emb: Vec<f32>,
    /// Encoder blocks.
    pub(crate) layers: Vec<EncoderLayer>,
    /// Final encoder LayerNorm.
    pub(crate) ln_post: LayerNorm,
}

/// Decoder weights.
pub(crate) struct DecoderWeights {
    /// Token embedding `[n_vocab, d_model]` (also the tied logits projection).
    pub(crate) token_emb: Vec<f32>,
    /// Learned positional embedding `[n_text_ctx, d_model]`.
    pub(crate) pos_emb: Vec<f32>,
    /// Decoder blocks.
    pub(crate) layers: Vec<DecoderLayer>,
    /// Final decoder LayerNorm.
    pub(crate) ln_post: LayerNorm,
}

/// All Whisper weights, bound from a GGUF under the [`WhisperConfig`] shapes.
pub struct WhisperWeights {
    pub(crate) encoder: EncoderWeights,
    pub(crate) decoder: DecoderWeights,
}

impl WhisperWeights {
    /// Binds every weight from `file`, validating each tensor's presence,
    /// dtype and shape against `cfg`.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] naming the offending tensor if it is missing,
    /// has an unsupported dtype, or has an unexpected shape.
    pub fn load(file: &GgufFile, cfg: &WhisperConfig) -> Result<Self> {
        let d = cfg.d_model;
        let ff = cfg.ffn_dim;

        // ---- encoder ----
        let conv1_w = tensor(file, "model.encoder.conv1.weight", &[d, cfg.n_mels, 3])?;
        let conv1_b = tensor(file, "model.encoder.conv1.bias", &[d])?;
        let conv2_w = tensor(file, "model.encoder.conv2.weight", &[d, d, 3])?;
        let conv2_b = tensor(file, "model.encoder.conv2.bias", &[d])?;
        let enc_pos = tensor(
            file,
            "model.encoder.embed_positions.weight",
            &[cfg.n_audio_ctx, d],
        )?;

        let mut enc_layers = Vec::with_capacity(cfg.n_audio_layer);
        for i in 0..cfg.n_audio_layer {
            let p = format!("model.encoder.layers.{i}");
            enc_layers.push(EncoderLayer {
                attn_ln: layer_norm(file, &format!("{p}.self_attn_layer_norm"), d)?,
                attn: attention(file, &format!("{p}.self_attn"), d)?,
                mlp_ln: layer_norm(file, &format!("{p}.final_layer_norm"), d)?,
                fc1: linear(file, &format!("{p}.fc1"), d, ff, true)?,
                fc2: linear(file, &format!("{p}.fc2"), ff, d, true)?,
            });
        }
        let enc_ln_post = layer_norm(file, "model.encoder.layer_norm", d)?;

        // ---- decoder ----
        let token_emb = tensor(file, "model.decoder.embed_tokens.weight", &[cfg.n_vocab, d])?;
        let dec_pos = tensor(
            file,
            "model.decoder.embed_positions.weight",
            &[cfg.n_text_ctx, d],
        )?;

        let mut dec_layers = Vec::with_capacity(cfg.n_text_layer);
        for i in 0..cfg.n_text_layer {
            let p = format!("model.decoder.layers.{i}");
            dec_layers.push(DecoderLayer {
                self_ln: layer_norm(file, &format!("{p}.self_attn_layer_norm"), d)?,
                self_attn: attention(file, &format!("{p}.self_attn"), d)?,
                cross_ln: layer_norm(file, &format!("{p}.encoder_attn_layer_norm"), d)?,
                cross_attn: attention(file, &format!("{p}.encoder_attn"), d)?,
                mlp_ln: layer_norm(file, &format!("{p}.final_layer_norm"), d)?,
                fc1: linear(file, &format!("{p}.fc1"), d, ff, true)?,
                fc2: linear(file, &format!("{p}.fc2"), ff, d, true)?,
            });
        }
        let dec_ln_post = layer_norm(file, "model.decoder.layer_norm", d)?;

        Ok(Self {
            encoder: EncoderWeights {
                conv1_w,
                conv1_b,
                conv2_w,
                conv2_b,
                pos_emb: enc_pos,
                layers: enc_layers,
                ln_post: enc_ln_post,
            },
            decoder: DecoderWeights {
                token_emb,
                pos_emb: dec_pos,
                layers: dec_layers,
                ln_post: dec_ln_post,
            },
        })
    }
}

fn err(name: &str, msg: impl std::fmt::Display) -> VokraError {
    VokraError::ModelLoad(format!("whisper weight `{name}`: {msg}"))
}

/// Reads a tensor as owned f32 and checks its shape.
///
/// Decoding (dense `F32` / `F16` or K-quant dequant) goes through the shared
/// [`GgufFile::tensor_f32`] path, so a K-quantized Whisper GGUF loads with no
/// changes to this module.
fn tensor(file: &GgufFile, name: &str, want: &[usize]) -> Result<Vec<f32>> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| err(name, "missing from GGUF"))?;
    let got: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    if got != want {
        return Err(err(name, format!("shape {got:?} != expected {want:?}")));
    }
    file.tensor_f32(name).map_err(|e| err(name, e))
}

/// Loads an `nn.Linear`, transposing the `[out, in]` weight to `[in, out]`.
fn linear(
    file: &GgufFile,
    prefix: &str,
    in_features: usize,
    out_features: usize,
    has_bias: bool,
) -> Result<Linear> {
    let w = tensor(
        file,
        &format!("{prefix}.weight"),
        &[out_features, in_features],
    )?;
    // Transpose [out, in] -> [in, out].
    let mut w_t = vec![0.0f32; in_features * out_features];
    for o in 0..out_features {
        let row = &w[o * in_features..(o + 1) * in_features];
        for (i, &v) in row.iter().enumerate() {
            w_t[i * out_features + o] = v;
        }
    }
    let bias = if has_bias {
        Some(tensor(file, &format!("{prefix}.bias"), &[out_features])?)
    } else {
        None
    };
    Ok(Linear {
        w_t,
        in_features,
        out_features,
        bias,
    })
}

/// Loads a LayerNorm (`weight` = γ, `bias` = β, width `d`).
fn layer_norm(file: &GgufFile, prefix: &str, d: usize) -> Result<LayerNorm> {
    Ok(LayerNorm {
        gamma: tensor(file, &format!("{prefix}.weight"), &[d])?,
        beta: tensor(file, &format!("{prefix}.bias"), &[d])?,
    })
}

/// Loads a `q/k/v/out` attention block; `k_proj` has no bias in Whisper.
fn attention(file: &GgufFile, prefix: &str, d: usize) -> Result<Attention> {
    Ok(Attention {
        q: linear(file, &format!("{prefix}.q_proj"), d, d, true)?,
        k: linear(file, &format!("{prefix}.k_proj"), d, d, false)?,
        v: linear(file, &format!("{prefix}.v_proj"), d, d, true)?,
        out: linear(file, &format!("{prefix}.out_proj"), d, d, true)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile};

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    #[test]
    fn tensor_decodes_f16_through_shared_path() {
        // 1.0, 2.0, -2.0 as half precision; `tensor` must route F16 through the
        // shared core dequant, not a private loop.
        let f16: Vec<u8> = [0x3C00u16, 0x4000, 0xC000]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        let mut b = GgufBuilder::new();
        b.add_tensor("h", GgmlType::F16, vec![3], f16).unwrap();
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert_eq!(tensor(&file, "h", &[3]).unwrap(), vec![1.0, 2.0, -2.0]);
    }

    #[test]
    fn tensor_loads_kquant_weight() {
        // A Q6_K tensor (one all-zero super-block) loads through `tensor` and
        // dequantizes to zeros — the K-quant weight path works in the model
        // layer with no dtype-specific code here.
        let mut b = GgufBuilder::new();
        b.add_tensor("q", GgmlType::Q6K, vec![256], vec![0u8; 210])
            .unwrap();
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let out = tensor(&file, "q", &[256]).unwrap();
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn linear_is_transposed_to_in_out() {
        // weight [out=2, in=3] = [[1,2,3],[4,5,6]]; bias [10,20].
        let mut b = GgufBuilder::new();
        b.add_tensor(
            "lin.weight",
            GgmlType::F32,
            vec![2, 3],
            f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        .unwrap();
        b.add_tensor("lin.bias", GgmlType::F32, vec![2], f32_bytes(&[10.0, 20.0]))
            .unwrap();
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let lin = linear(&file, "lin", 3, 2, true).unwrap();
        // w_t is [in=3, out=2] row-major: column-major read of the original.
        assert_eq!(lin.w_t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(lin.bias, Some(vec![10.0, 20.0]));
    }

    #[test]
    fn missing_tensor_is_model_load_error() {
        let file = GgufFile::parse(GgufBuilder::new().to_bytes().unwrap()).unwrap();
        let e = tensor(&file, "nope", &[1]).unwrap_err();
        assert!(matches!(e, VokraError::ModelLoad(_)));
    }

    /// Turbo-shaped WhisperConfig using minimal per-layer dims to keep the
    /// synthetic tensor bodies small while still exercising the 32/4 layer
    /// counts (M2-06-T05).
    fn turbo_config() -> WhisperConfig {
        WhisperConfig {
            n_mels: 8,
            d_model: 32,
            n_audio_ctx: 4,
            n_audio_head: 4,
            n_audio_layer: 32,
            n_text_ctx: 4,
            n_text_head: 4,
            n_text_layer: 4,
            n_vocab: 16,
            ffn_dim: 64,
            eot: 15,
            decoder_start_ids: vec![0, 1, 2, 3],
            alignment_heads: Vec::new(),
        }
    }

    /// Populates a GGUF with every Whisper tensor named by `WhisperWeights::load`
    /// under `cfg`, with all-zero f32 bodies of the correct shape. Optionally
    /// skips a single tensor whose name matches `skip` — used to drive the
    /// negative branch of the missing-tensor path.
    fn build_turbo_gguf(cfg: &WhisperConfig, skip: Option<&str>) -> GgufFile {
        let mut b = GgufBuilder::new();
        let d = cfg.d_model;
        let ff = cfg.ffn_dim;

        let mut add = |name: &str, dims: Vec<u64>| {
            if Some(name) == skip {
                return;
            }
            let n: usize = dims.iter().map(|&x| x as usize).product();
            b.add_tensor(name, GgmlType::F32, dims, vec![0u8; n * 4])
                .unwrap();
        };

        // Encoder-level tensors.
        add(
            "model.encoder.conv1.weight",
            vec![d as u64, cfg.n_mels as u64, 3],
        );
        add("model.encoder.conv1.bias", vec![d as u64]);
        add("model.encoder.conv2.weight", vec![d as u64, d as u64, 3]);
        add("model.encoder.conv2.bias", vec![d as u64]);
        add(
            "model.encoder.embed_positions.weight",
            vec![cfg.n_audio_ctx as u64, d as u64],
        );

        // Encoder layers × n_audio_layer.
        for i in 0..cfg.n_audio_layer {
            let p = format!("model.encoder.layers.{i}");
            add(&format!("{p}.self_attn_layer_norm.weight"), vec![d as u64]);
            add(&format!("{p}.self_attn_layer_norm.bias"), vec![d as u64]);
            add(
                &format!("{p}.self_attn.q_proj.weight"),
                vec![d as u64, d as u64],
            );
            add(&format!("{p}.self_attn.q_proj.bias"), vec![d as u64]);
            add(
                &format!("{p}.self_attn.k_proj.weight"),
                vec![d as u64, d as u64],
            );
            // k_proj has no bias in Whisper.
            add(
                &format!("{p}.self_attn.v_proj.weight"),
                vec![d as u64, d as u64],
            );
            add(&format!("{p}.self_attn.v_proj.bias"), vec![d as u64]);
            add(
                &format!("{p}.self_attn.out_proj.weight"),
                vec![d as u64, d as u64],
            );
            add(&format!("{p}.self_attn.out_proj.bias"), vec![d as u64]);
            add(&format!("{p}.final_layer_norm.weight"), vec![d as u64]);
            add(&format!("{p}.final_layer_norm.bias"), vec![d as u64]);
            add(&format!("{p}.fc1.weight"), vec![ff as u64, d as u64]);
            add(&format!("{p}.fc1.bias"), vec![ff as u64]);
            add(&format!("{p}.fc2.weight"), vec![d as u64, ff as u64]);
            add(&format!("{p}.fc2.bias"), vec![d as u64]);
        }
        add("model.encoder.layer_norm.weight", vec![d as u64]);
        add("model.encoder.layer_norm.bias", vec![d as u64]);

        // Decoder-level tensors.
        add(
            "model.decoder.embed_tokens.weight",
            vec![cfg.n_vocab as u64, d as u64],
        );
        add(
            "model.decoder.embed_positions.weight",
            vec![cfg.n_text_ctx as u64, d as u64],
        );

        // Decoder layers × n_text_layer.
        for i in 0..cfg.n_text_layer {
            let p = format!("model.decoder.layers.{i}");
            add(&format!("{p}.self_attn_layer_norm.weight"), vec![d as u64]);
            add(&format!("{p}.self_attn_layer_norm.bias"), vec![d as u64]);
            add(
                &format!("{p}.self_attn.q_proj.weight"),
                vec![d as u64, d as u64],
            );
            add(&format!("{p}.self_attn.q_proj.bias"), vec![d as u64]);
            add(
                &format!("{p}.self_attn.k_proj.weight"),
                vec![d as u64, d as u64],
            );
            add(
                &format!("{p}.self_attn.v_proj.weight"),
                vec![d as u64, d as u64],
            );
            add(&format!("{p}.self_attn.v_proj.bias"), vec![d as u64]);
            add(
                &format!("{p}.self_attn.out_proj.weight"),
                vec![d as u64, d as u64],
            );
            add(&format!("{p}.self_attn.out_proj.bias"), vec![d as u64]);
            add(
                &format!("{p}.encoder_attn_layer_norm.weight"),
                vec![d as u64],
            );
            add(&format!("{p}.encoder_attn_layer_norm.bias"), vec![d as u64]);
            add(
                &format!("{p}.encoder_attn.q_proj.weight"),
                vec![d as u64, d as u64],
            );
            add(&format!("{p}.encoder_attn.q_proj.bias"), vec![d as u64]);
            add(
                &format!("{p}.encoder_attn.k_proj.weight"),
                vec![d as u64, d as u64],
            );
            add(
                &format!("{p}.encoder_attn.v_proj.weight"),
                vec![d as u64, d as u64],
            );
            add(&format!("{p}.encoder_attn.v_proj.bias"), vec![d as u64]);
            add(
                &format!("{p}.encoder_attn.out_proj.weight"),
                vec![d as u64, d as u64],
            );
            add(&format!("{p}.encoder_attn.out_proj.bias"), vec![d as u64]);
            add(&format!("{p}.final_layer_norm.weight"), vec![d as u64]);
            add(&format!("{p}.final_layer_norm.bias"), vec![d as u64]);
            add(&format!("{p}.fc1.weight"), vec![ff as u64, d as u64]);
            add(&format!("{p}.fc1.bias"), vec![ff as u64]);
            add(&format!("{p}.fc2.weight"), vec![d as u64, ff as u64]);
            add(&format!("{p}.fc2.bias"), vec![d as u64]);
        }
        add("model.decoder.layer_norm.weight", vec![d as u64]);
        add("model.decoder.layer_norm.bias", vec![d as u64]);

        GgufFile::parse(b.to_bytes().unwrap()).unwrap()
    }

    #[test]
    fn loads_turbo_layer_counts() {
        // Turbo's asymmetric split (32 encoder blocks, 4 decoder blocks) must
        // round-trip through `WhisperWeights::load` driven only by cfg counts.
        let cfg = turbo_config();
        let file = build_turbo_gguf(&cfg, None);
        let weights = WhisperWeights::load(&file, &cfg).expect("turbo weights load");
        assert_eq!(weights.encoder.layers.len(), 32);
        assert_eq!(weights.decoder.layers.len(), 4);

        // Removing one decoder tensor must surface as an explicit ModelLoad
        // error (FR-EX-08: no silent fallback).
        let cfg2 = turbo_config();
        let file2 = build_turbo_gguf(
            &cfg2,
            Some("model.decoder.layers.3.self_attn.q_proj.weight"),
        );
        // `WhisperWeights` intentionally lacks `Debug` (owned weight buffers
        // would produce enormous dumps), so match the Result explicitly instead
        // of `unwrap_err`.
        match WhisperWeights::load(&file2, &cfg2) {
            Ok(_) => panic!("expected ModelLoad error, weights loaded successfully"),
            Err(VokraError::ModelLoad(_)) => {}
            Err(other) => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn shape_mismatch_is_rejected() {
        let mut b = GgufBuilder::new();
        b.add_tensor("t", GgmlType::F32, vec![2, 2], f32_bytes(&[0.0; 4]))
            .unwrap();
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(matches!(
            tensor(&file, "t", &[4]),
            Err(VokraError::ModelLoad(_))
        ));
    }
}
