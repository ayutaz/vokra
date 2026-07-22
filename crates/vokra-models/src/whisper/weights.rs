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

use vokra_backend_cpu::kernels::KQuantDtype;
use vokra_core::gguf::{GgufFile, tensor::QK_K};
use vokra_core::{Result, VokraError};

use super::config::WhisperConfig;

/// How a [`Linear`]'s weight matrix is held in memory.
///
/// The dense arm is the pre-M5-15 behaviour and stays the default; the
/// K-quant arm keeps the GGUF super-blocks verbatim so the fused INT8 kernels
/// can consume them (M5-15-T26).
pub(crate) enum LinearWeight {
    /// Transposed `[in_features, out_features]` f32, row-major — the layout
    /// `Compute::gemm_f32` consumes directly.
    Dense(Vec<f32>),
    /// On-disk K-quant super-blocks, **untransposed** `[out_features,
    /// in_features]` — exactly the layout `Compute::gemm_q_f32` wants, so the
    /// quant path pays no transpose at all.
    KQuant {
        /// Raw super-block payload, `out_features × (in_features / 256)`
        /// blocks of `dtype.block_bytes()`.
        bytes: Vec<u8>,
        /// Super-block format.
        dtype: KQuantDtype,
    },
}

/// A HF `nn.Linear` decoded for direct row-major GEMM.
///
/// The weight is either dense f32 (`[in, out]`, so
/// `y[t, o] = bias[o] + sum_i x[t, i] * w_t[i, o]`) or K-quant super-blocks
/// (`[out, in]`, consumed by the fused INT8 GEMM). The bias is **always**
/// dense f32 — biases are never quantized.
pub(crate) struct Linear {
    /// Weight storage (dense f32 or K-quant super-blocks).
    pub(crate) w: LinearWeight,
    /// Input width.
    pub(crate) in_features: usize,
    /// Output width.
    pub(crate) out_features: usize,
    /// Optional per-output bias (`None` for Whisper `k_proj`).
    pub(crate) bias: Option<Vec<f32>>,
}

impl Linear {
    /// A dense `Linear` from an explicit `[in_features, out_features]` weight
    /// (test fixtures and the synthetic-weight builders).
    pub(crate) fn dense(
        w_t: Vec<f32>,
        in_features: usize,
        out_features: usize,
        bias: Option<Vec<f32>>,
    ) -> Self {
        Self {
            w: LinearWeight::Dense(w_t),
            in_features,
            out_features,
            bias,
        }
    }

    /// The dense `[in_features, out_features]` weight.
    ///
    /// # Errors
    ///
    /// [`VokraError::UnsupportedOp`] when this `Linear` kept its K-quant form.
    /// Callers that cannot consume super-blocks — the device-resident fused
    /// seams (`Compute::attn_f32`, the pre-norm encoder stack, the decoder
    /// step session), all of which want an f32 pointer to upload — go through
    /// here so a fused-quant model on a GPU backend fails **loudly** instead
    /// of uploading an empty slice (FR-EX-08).
    pub(crate) fn dense_w_t(&self) -> Result<&[f32]> {
        match &self.w {
            LinearWeight::Dense(w_t) => Ok(w_t),
            LinearWeight::KQuant { dtype, .. } => Err(VokraError::UnsupportedOp(format!(
                "this Linear holds {dtype:?} super-blocks (fused-quant weights were requested at \
                 load), and the caller needs a dense f32 weight. The fused K-quant path is \
                 CPU-only: load without `WhisperLoadOptions::fused_quant_weights` to get the \
                 dequantized weights every backend can use."
            ))),
        }
    }

    /// The K-quant payload and format, or `None` for a dense weight.
    pub(crate) fn kquant(&self) -> Option<(&[u8], KQuantDtype)> {
        match &self.w {
            LinearWeight::Dense(_) => None,
            LinearWeight::KQuant { bytes, dtype } => Some((bytes, *dtype)),
        }
    }
}

/// Load-time options for [`WhisperWeights::load_with`] (M5-15-T26/T28).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WhisperLoadOptions {
    /// Keep K-quantized `nn.Linear` weights in their on-disk super-block form
    /// and run the fused INT8 kernels, instead of dequantizing every tensor to
    /// f32 at load.
    ///
    /// **Default `false`**, and deliberately so: the fused route is *not*
    /// bit-identical to the dequant route (the INT8 surface is bounded by the
    /// activation-quantization band, `kquant.rs` `UnpackedBlock` docs), so
    /// turning it on changes model output. Whether it becomes the default is
    /// gated on the M5-15-T10 WER measurement — see `docs/adr/M5-15-quant.md`
    /// §D4. It is also **CPU-only**: a GPU backend with these weights raises an
    /// explicit error rather than silently dequantizing or silently running on
    /// the CPU (FR-EX-08).
    pub fused_quant_weights: bool,
}

/// What the fused-quant binding actually did, per load (M5-15-T26).
///
/// Reported rather than silent: a K-quantized GGUF whose rows are not
/// 256-element aligned still loads, but through the f32 dequant path, and the
/// caller must be able to see that its "quantized" model is partly dense.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QuantBindReport {
    /// `nn.Linear` weights bound as K-quant super-blocks.
    pub fused: usize,
    /// `nn.Linear` weights that are K-quant **on disk** but were dequantized
    /// to f32 because `in_features % 256 != 0`, so a super-block would
    /// straddle two weight rows and the fused GEMV contract
    /// (`k` a positive multiple of `QK_K`) cannot hold.
    pub dequantized_unaligned: usize,
}

/// Load-time state threaded through the weight binders.
pub(crate) struct LoadCtx {
    /// Whether to attempt the fused-quant binding at all.
    fused_quant_weights: bool,
    /// Accumulated per-load counters.
    pub(crate) report: QuantBindReport,
}

impl LoadCtx {
    /// The pre-M5-15 behaviour: dequantize everything to f32 at load.
    pub(crate) fn dense() -> Self {
        Self {
            fused_quant_weights: false,
            report: QuantBindReport::default(),
        }
    }

    fn from_options(opts: WhisperLoadOptions) -> Self {
        Self {
            fused_quant_weights: opts.fused_quant_weights,
            report: QuantBindReport::default(),
        }
    }
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
    /// What the M5-15 fused-quant binding did (all-zero on the default load).
    pub(crate) quant_report: QuantBindReport,
}

impl WhisperWeights {
    /// Binds every weight from `file`, validating each tensor's presence,
    /// dtype and shape against `cfg`. Every tensor is decoded to f32 — a
    /// K-quantized GGUF is dequantized at load.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] naming the offending tensor if it is missing,
    /// has an unsupported dtype, or has an unexpected shape.
    pub fn load(file: &GgufFile, cfg: &WhisperConfig) -> Result<Self> {
        Self::load_with(file, cfg, WhisperLoadOptions::default())
    }

    /// [`load`](Self::load) with the M5-15 fused-quant binding options.
    ///
    /// With [`WhisperLoadOptions::fused_quant_weights`] set, `nn.Linear`
    /// weights that are K-quant on disk **and** 256-aligned per row keep their
    /// super-blocks; everything else (biases, norms, embeddings, unaligned
    /// rows) still decodes to f32. [`Self::quant_report`] says how many took
    /// each route.
    ///
    /// # Errors
    ///
    /// As [`load`](Self::load), plus [`VokraError::ModelLoad`] when a K-quant
    /// weight's payload length disagrees with its declared shape.
    pub fn load_with(
        file: &GgufFile,
        cfg: &WhisperConfig,
        opts: WhisperLoadOptions,
    ) -> Result<Self> {
        let ctx = &mut LoadCtx::from_options(opts);
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
                attn: attention_with(file, &format!("{p}.self_attn"), d, ctx)?,
                mlp_ln: layer_norm(file, &format!("{p}.final_layer_norm"), d)?,
                fc1: linear_with(file, &format!("{p}.fc1"), d, ff, true, ctx)?,
                fc2: linear_with(file, &format!("{p}.fc2"), ff, d, true, ctx)?,
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
                self_attn: attention_with(file, &format!("{p}.self_attn"), d, ctx)?,
                cross_ln: layer_norm(file, &format!("{p}.encoder_attn_layer_norm"), d)?,
                cross_attn: attention_with(file, &format!("{p}.encoder_attn"), d, ctx)?,
                mlp_ln: layer_norm(file, &format!("{p}.final_layer_norm"), d)?,
                fc1: linear_with(file, &format!("{p}.fc1"), d, ff, true, ctx)?,
                fc2: linear_with(file, &format!("{p}.fc2"), ff, d, true, ctx)?,
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
            quant_report: ctx.report,
        })
    }

    /// What the fused-quant binding did on this load (all-zero unless
    /// [`WhisperLoadOptions::fused_quant_weights`] was set).
    pub fn quant_report(&self) -> QuantBindReport {
        self.quant_report
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
///
/// `pub(crate)`: the Voxtral audio encoder (`crate::voxtral::audio_encoder`)
/// is a Whisper-derived stack whose per-layer tensors use the identical HF
/// sub-names under a different prefix (`audio_tower.` instead of
/// `model.encoder.`) — it reuses these loaders verbatim so the two models
/// share ONE audited weight-binding path (no second implementation).
pub(crate) fn tensor(file: &GgufFile, name: &str, want: &[usize]) -> Result<Vec<f32>> {
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
///
/// The dense binder — unchanged pre-M5-15 behaviour, kept as the signature the
/// Voxtral audio tower and every existing caller use.
pub(crate) fn linear(
    file: &GgufFile,
    prefix: &str,
    in_features: usize,
    out_features: usize,
    has_bias: bool,
) -> Result<Linear> {
    linear_with(
        file,
        prefix,
        in_features,
        out_features,
        has_bias,
        &mut LoadCtx::dense(),
    )
}

/// [`linear`] with the M5-15 fused-quant binding decision.
///
/// When `ctx` asks for fused-quant weights **and** the on-disk tensor is a
/// K-quant **and** `in_features % QK_K == 0`, the super-blocks are kept
/// verbatim (no transpose, no dequant). Every other case falls back to the
/// dense binder — and a K-quant tensor that falls back only because of the row
/// alignment is *counted* in the report, never silently absorbed.
pub(crate) fn linear_with(
    file: &GgufFile,
    prefix: &str,
    in_features: usize,
    out_features: usize,
    has_bias: bool,
    ctx: &mut LoadCtx,
) -> Result<Linear> {
    let wname = format!("{prefix}.weight");
    let bias = if has_bias {
        Some(tensor(file, &format!("{prefix}.bias"), &[out_features])?)
    } else {
        None
    };

    if ctx.fused_quant_weights
        && let Some(info) = file.tensor_info(&wname)
        && let Some(dtype) = KQuantDtype::from_ggml(info.dtype)
    {
        // Shape is validated here (not by `tensor`) because the quant path
        // never decodes the payload: a wrong `[out, in]` would otherwise be
        // discovered only as a byte-count mismatch below, with a worse message.
        let got: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
        if got != [out_features, in_features] {
            return Err(err(
                &wname,
                format!(
                    "shape {got:?} != expected {:?}",
                    [out_features, in_features]
                ),
            ));
        }
        if in_features % QK_K == 0 {
            let bytes = file
                .tensor_data(&wname)
                .ok_or_else(|| err(&wname, "missing from GGUF"))?;
            // Defense-in-depth: for any GGUF that *parses*, `tensor_bytes`
            // already slices to `byte_len()` recomputed from `(dtype, dims)`
            // and validated at parse, and `GgufBuilder::add_tensor` rejects a
            // mismatched payload up front — so with the dims checked above
            // this can only fire on a hand-crafted file. Kept because the
            // kernel indexes these bytes by super-block arithmetic.
            let want = out_features * (in_features / QK_K) * dtype.block_bytes();
            if bytes.len() != want {
                return Err(err(
                    &wname,
                    format!(
                        "{dtype:?} payload is {} bytes, want {want} \
                         ({out_features} rows x {} super-blocks x {} bytes)",
                        bytes.len(),
                        in_features / QK_K,
                        dtype.block_bytes(),
                    ),
                ));
            }
            ctx.report.fused += 1;
            return Ok(Linear {
                w: LinearWeight::KQuant {
                    bytes: bytes.to_vec(),
                    dtype,
                },
                in_features,
                out_features,
                bias,
            });
        }
        // K-quant on disk but the row straddles super-blocks: the fused GEMV
        // contract (`k` a positive multiple of QK_K) cannot hold, so this
        // tensor stays on the dequant path. Counted, not hidden.
        ctx.report.dequantized_unaligned += 1;
    }

    let w = tensor(file, &wname, &[out_features, in_features])?;
    // Transpose [out, in] -> [in, out].
    let mut w_t = vec![0.0f32; in_features * out_features];
    for o in 0..out_features {
        let row = &w[o * in_features..(o + 1) * in_features];
        for (i, &v) in row.iter().enumerate() {
            w_t[i * out_features + o] = v;
        }
    }
    Ok(Linear::dense(w_t, in_features, out_features, bias))
}

/// Loads a LayerNorm (`weight` = γ, `bias` = β, width `d`).
pub(crate) fn layer_norm(file: &GgufFile, prefix: &str, d: usize) -> Result<LayerNorm> {
    Ok(LayerNorm {
        gamma: tensor(file, &format!("{prefix}.weight"), &[d])?,
        beta: tensor(file, &format!("{prefix}.bias"), &[d])?,
    })
}

/// Loads a `q/k/v/out` attention block; `k_proj` has no bias in Whisper
/// (nor in the Whisper-derived Voxtral audio tower — HF
/// `modeling_voxtral.py` `VoxtralAttention.__init__`:
/// `k_proj = nn.Linear(embed_dim, embed_dim, bias=False)`).
pub(crate) fn attention(file: &GgufFile, prefix: &str, d: usize) -> Result<Attention> {
    attention_with(file, prefix, d, &mut LoadCtx::dense())
}

/// [`attention`] threading the M5-15 fused-quant binding decision.
pub(crate) fn attention_with(
    file: &GgufFile,
    prefix: &str,
    d: usize,
    ctx: &mut LoadCtx,
) -> Result<Attention> {
    Ok(Attention {
        q: linear_with(file, &format!("{prefix}.q_proj"), d, d, true, ctx)?,
        k: linear_with(file, &format!("{prefix}.k_proj"), d, d, false, ctx)?,
        v: linear_with(file, &format!("{prefix}.v_proj"), d, d, true, ctx)?,
        out: linear_with(file, &format!("{prefix}.out_proj"), d, d, true, ctx)?,
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
        assert_eq!(lin.dense_w_t().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
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

    // ---- M5-15: fused-quant weight binding (T26 / T28 / T29) -------------

    /// Deterministic byte source (xorshift; no external crate).
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn f32s(&mut self, n: usize) -> Vec<f32> {
            (0..n)
                .map(|_| ((self.next_u64() >> 40) as u32) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0)
                .collect()
        }
    }

    /// A K-quant payload whose f16 scales are pinned to small finite
    /// magnitudes; every other byte is pseudo-random (a valid super-block for
    /// all three formats). Transcribed from
    /// `vokra-backend-cpu/tests/server_tier_parity.rs::random_blocks`.
    fn kquant_blocks(rng: &mut Rng, dtype: GgmlType, nb: usize) -> Vec<u8> {
        let bs = match dtype {
            GgmlType::Q4K => 144,
            GgmlType::Q5K => 176,
            GgmlType::Q6K => 210,
            other => panic!("not a K-quant dtype: {other:?}"),
        };
        let mut bytes: Vec<u8> = (0..nb * bs).map(|_| (rng.next_u64() >> 32) as u8).collect();
        for b in 0..nb {
            let base = b * bs;
            match dtype {
                GgmlType::Q4K | GgmlType::Q5K => {
                    bytes[base + 1] = 0x2C;
                    bytes[base + 3] = 0x24;
                }
                GgmlType::Q6K => bytes[base + 209] = 0x2C,
                _ => unreachable!(),
            }
        }
        bytes
    }

    /// A single-`nn.Linear` GGUF: `{prefix}.weight` `[out, in]` in `dtype`
    /// (K-quant payloads are synthesized, f32 payloads are zero) plus a bias.
    fn linear_gguf(
        rng: &mut Rng,
        prefix: &str,
        in_f: usize,
        out_f: usize,
        dtype: GgmlType,
    ) -> GgufFile {
        let mut b = GgufBuilder::new();
        let payload = match dtype {
            GgmlType::F32 => f32_bytes(&vec![0.0; in_f * out_f]),
            q => kquant_blocks(rng, q, out_f * in_f / 256),
        };
        b.add_tensor(
            &format!("{prefix}.weight"),
            dtype,
            vec![out_f as u64, in_f as u64],
            payload,
        )
        .unwrap();
        b.add_tensor(
            &format!("{prefix}.bias"),
            GgmlType::F32,
            vec![out_f as u64],
            f32_bytes(&vec![0.25; out_f]),
        )
        .unwrap();
        GgufFile::parse(b.to_bytes().unwrap()).unwrap()
    }

    /// T26 (i): a K-quant weight with 256-aligned rows binds as super-blocks
    /// and is counted; the default load still dequantizes it.
    #[test]
    fn aligned_kquant_linear_binds_fused_and_is_counted() {
        for dtype in [GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q6K] {
            let mut rng = Rng::new(0x5115_2601);
            let file = linear_gguf(&mut rng, "lin", 256, 4, dtype);

            let mut ctx = LoadCtx::from_options(WhisperLoadOptions {
                fused_quant_weights: true,
            });
            let lin = linear_with(&file, "lin", 256, 4, true, &mut ctx).unwrap();
            assert!(lin.kquant().is_some(), "{dtype:?} should bind fused");
            assert!(lin.dense_w_t().is_err(), "{dtype:?} has no dense weight");
            assert_eq!(ctx.report.fused, 1);
            assert_eq!(ctx.report.dequantized_unaligned, 0);
            // The bias is never quantized.
            assert_eq!(lin.bias.as_deref(), Some(&[0.25f32; 4][..]));

            // Default (opt-out) load takes the dequant path, as before.
            let mut dense_ctx = LoadCtx::dense();
            let dense = linear_with(&file, "lin", 256, 4, true, &mut dense_ctx).unwrap();
            assert!(
                dense.kquant().is_none(),
                "{dtype:?} default must dequantize"
            );
            assert_eq!(dense.dense_w_t().unwrap().len(), 256 * 4);
            assert_eq!(dense_ctx.report, QuantBindReport::default());
        }
    }

    /// T26 (ii): a K-quant weight whose **row** is not 256-aligned cannot feed
    /// the fused GEMV (a super-block would straddle two rows), so it stays on
    /// the dequant path — and that is reported, not silent.
    #[test]
    fn unaligned_kquant_rows_fall_back_to_dequant_and_are_reported() {
        let mut rng = Rng::new(0x5115_2602);
        // in_features = 128: the tensor as a whole is 128*4 = 512 elements
        // (block-aligned, so the converter accepts it) but each ROW is only
        // 128 long.
        let file = linear_gguf(&mut rng, "lin", 128, 4, GgmlType::Q6K);
        let mut ctx = LoadCtx::from_options(WhisperLoadOptions {
            fused_quant_weights: true,
        });
        let lin = linear_with(&file, "lin", 128, 4, true, &mut ctx).unwrap();
        assert!(lin.kquant().is_none(), "unaligned rows must not bind fused");
        assert_eq!(ctx.report.fused, 0);
        assert_eq!(ctx.report.dequantized_unaligned, 1);
    }

    /// T26 (iii): a K-quant weight whose GGUF dims disagree with the shape the
    /// config demands is a loud `ModelLoad` on the fused path too — the quant
    /// binder never decodes the payload, so without this check a wrong shape
    /// would slip past into the kernel (FR-EX-08).
    ///
    /// Only `in_features` is varied: `linear_with` loads the bias (checked
    /// against `out_features`) *before* the weight, so perturbing
    /// `out_features` would be caught by the bias shape check and never reach
    /// the fused weight guard this test exists to pin.
    ///
    /// # Why there is no truncated-payload case here
    ///
    /// `linear_with` also asserts `bytes.len() == out × (in / QK_K) ×
    /// block_bytes`, but that branch is **unreachable through any GGUF that
    /// exists**: `GgufBuilder::add_tensor` rejects a payload whose length
    /// disagrees with `(dtype, dims)` (`GgufError::TensorSizeMismatch`), and
    /// `GgufFile::tensor_bytes` slices to `byte_len()` recomputed from
    /// `(dtype, dims)` and validated at parse. The check is kept as
    /// defense-in-depth against a hand-crafted file, and is deliberately not
    /// given a fake test that would only prove the fixture builder works.
    #[test]
    fn kquant_shape_mismatch_is_a_loud_model_load_error() {
        let mut rng = Rng::new(0x5115_2603);
        let file = linear_gguf(&mut rng, "lin", 256, 4, GgmlType::Q6K);
        let mut ctx = LoadCtx::from_options(WhisperLoadOptions {
            fused_quant_weights: true,
        });
        // GGUF says [4, 256]; ask for [4, 512]. The bias `[4]` still matches,
        // so the failure can only come from the fused weight-shape guard.
        match linear_with(&file, "lin", 512, 4, true, &mut ctx) {
            Err(VokraError::ModelLoad(m)) => {
                assert!(
                    m.contains("lin.weight") && m.contains("shape [4, 256]"),
                    "expected the fused weight-shape guard, got: {m}"
                );
            }
            other => panic!("expected ModelLoad, got {:?}", other.map(|_| "Ok(Linear)")),
        }
        assert_eq!(
            ctx.report,
            QuantBindReport::default(),
            "no partial counting"
        );
    }

    /// T28 regression guard: a **non**-quantized GGUF must be bit-identical
    /// with and without the fused option — the new dispatch adds nothing to
    /// the f32 path.
    #[test]
    fn f32_weights_are_bit_identical_with_and_without_the_fused_option() {
        let mut rng = Rng::new(0x5115_2804);
        let file = linear_gguf(&mut rng, "lin", 256, 4, GgmlType::F32);
        let mut on = LoadCtx::from_options(WhisperLoadOptions {
            fused_quant_weights: true,
        });
        let a = linear_with(&file, "lin", 256, 4, true, &mut on).unwrap();
        let b = linear_with(&file, "lin", 256, 4, true, &mut LoadCtx::dense()).unwrap();
        assert!(a.kquant().is_none() && b.kquant().is_none());
        assert_eq!(a.dense_w_t().unwrap(), b.dense_w_t().unwrap());
        assert_eq!(on.report, QuantBindReport::default());

        let compute = crate::compute::Compute::cpu();
        let x = Rng::new(9).f32s(3 * 256);
        let mut ya = vec![0.0f32; 3 * 4];
        let mut yb = vec![0.0f32; 3 * 4];
        crate::whisper::nn::linear_apply(&compute, &mut ya, &x, 3, &a).unwrap();
        crate::whisper::nn::linear_apply(&compute, &mut yb, &x, 3, &b).unwrap();
        assert_eq!(ya, yb, "f32 dispatch must be bit-identical");
    }

    /// T29: the fused projection tracks the dequant projection within the
    /// **derived** activation-quantization bound (`int8_error_bound × 2`), for
    /// every K-quant dtype and for both the GEMV (`t = 1`, decoder step) and
    /// GEMM (`t > 1`, encoder) shapes. Not a hand-picked constant, and not an
    /// equality claim — the fused route is deliberately not bit-identical.
    #[test]
    fn fused_projection_tracks_dequant_within_the_derived_bound() {
        use vokra_backend_cpu::kernels::int8_error_bound;

        let compute = crate::compute::Compute::cpu();
        let (in_f, out_f) = (256usize, 6usize);
        for dtype in [GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q6K] {
            let mut rng = Rng::new(0x5115_2900);
            let file = linear_gguf(&mut rng, "lin", in_f, out_f, dtype);
            let fused = linear_with(
                &file,
                "lin",
                in_f,
                out_f,
                true,
                &mut LoadCtx::from_options(WhisperLoadOptions {
                    fused_quant_weights: true,
                }),
            )
            .unwrap();
            let dense =
                linear_with(&file, "lin", in_f, out_f, true, &mut LoadCtx::dense()).unwrap();
            let (wq, kdtype) = fused.kquant().expect("fused bind");
            let row_bytes = wq.len() / out_f;

            for t in [1usize, 5] {
                let x = Rng::new(0x5115_2901 + t as u64).f32s(t * in_f);
                let mut got = vec![0.0f32; t * out_f];
                let mut want = vec![0.0f32; t * out_f];
                crate::whisper::nn::linear_apply(&compute, &mut got, &x, t, &fused).unwrap();
                crate::whisper::nn::linear_apply(&compute, &mut want, &x, t, &dense).unwrap();
                for r in 0..t {
                    let xr = &x[r * in_f..(r + 1) * in_f];
                    for o in 0..out_f {
                        let bound =
                            int8_error_bound(kdtype, &wq[o * row_bytes..(o + 1) * row_bytes], xr)
                                .max(1e-6);
                        let diff = (got[r * out_f + o] - want[r * out_f + o]).abs();
                        assert!(
                            diff <= 2.0 * bound,
                            "{dtype:?} t={t} row {r} col {o}: fused {} vs dequant {} \
                             (|diff| {diff}) exceeds 2x derived bound {bound}",
                            got[r * out_f + o],
                            want[r * out_f + o],
                        );
                    }
                }
            }
        }
    }

    /// T28 backend gate: the device-resident views cannot upload a weight that
    /// is not there, so a fused-quant layer makes them fail loudly instead
    /// (FR-EX-08). This is what keeps a GPU backend from silently degrading.
    #[test]
    fn fused_quant_layer_is_rejected_by_the_device_resident_view() {
        let mut rng = Rng::new(0x5115_2805);
        let file = linear_gguf(&mut rng, "lin", 256, 256, GgmlType::Q6K);
        let mut ctx = LoadCtx::from_options(WhisperLoadOptions {
            fused_quant_weights: true,
        });
        let q = linear_with(&file, "lin", 256, 256, true, &mut ctx).unwrap();
        let dense = linear_with(&file, "lin", 256, 256, true, &mut LoadCtx::dense()).unwrap();
        let ln = || LayerNorm {
            gamma: vec![1.0; 256],
            beta: vec![0.0; 256],
        };
        let layer = EncoderLayer {
            attn_ln: ln(),
            attn: Attention {
                q,
                k: linear_with(&file, "lin", 256, 256, false, &mut LoadCtx::dense()).unwrap(),
                v: linear_with(&file, "lin", 256, 256, true, &mut LoadCtx::dense()).unwrap(),
                out: linear_with(&file, "lin", 256, 256, true, &mut LoadCtx::dense()).unwrap(),
            },
            mlp_ln: ln(),
            fc1: dense,
            fc2: linear_with(&file, "lin", 256, 256, true, &mut LoadCtx::dense()).unwrap(),
        };
        match crate::whisper::encoder::prenorm_view(&layer) {
            Err(VokraError::UnsupportedOp(m)) => {
                assert!(m.contains("fused_quant_weights"), "message: {m}");
            }
            Err(other) => panic!("expected UnsupportedOp, got {other:?}"),
            Ok(_) => panic!("device view must reject fused-quant weights"),
        }
    }
}
