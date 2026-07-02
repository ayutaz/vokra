//! VITS text encoder (M0-07-T12/T13): phoneme embedding + a 6-layer
//! relative-position transformer + the `m_p` / `logs_p` projection.
//!
//! Follows piper-plus `vits/models.py::TextEncoder` and
//! `vits/attentions.py::{Encoder, MultiHeadAttention, FFN}` exactly. Every
//! weight is a named tensor in the GGUF (no `onnx::` recovery needed here), so
//! this is the most directly parity-checkable component (`m_p` / `logs_p`
//! against the onnxruntime reference, M0-07-T13). All tensors are the
//! `[channels, time]` layout; batch is always 1 and every position is valid
//! (no padding mask) in the M0 single-utterance path.

use vokra_core::{Result, VokraError};

use super::config::{
    self, FFN_CHANNELS, FFN_KERNEL, GIN, HIDDEN, K_CHANNELS, N_HEADS, N_LAYERS, WINDOW_SIZE,
};
use super::nn;
use super::weights::TensorStore;

/// One relative-position self-attention layer's weights.
struct AttnLayer {
    conv_q: (Vec<f32>, Vec<f32>),
    conv_k: (Vec<f32>, Vec<f32>),
    conv_v: (Vec<f32>, Vec<f32>),
    conv_o: (Vec<f32>, Vec<f32>),
    /// Relative key/value embeddings, `[2·window+1, k_channels]`.
    emb_rel_k: Vec<f32>,
    emb_rel_v: Vec<f32>,
}

/// One FFN layer's weights (`conv_1` 192→768 k3, `conv_2` 768→192 k3, ReLU).
struct FfnLayer {
    conv_1: (Vec<f32>, Vec<f32>),
    conv_2: (Vec<f32>, Vec<f32>),
}

/// LayerNorm affine params over the channel axis.
struct Norm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

/// The full text encoder.
pub(super) struct TextEncoder {
    emb: Vec<f32>, // [n_vocab, HIDDEN]
    n_vocab: usize,
    attn: Vec<AttnLayer>,
    norm1: Vec<Norm>,
    ffn: Vec<FfnLayer>,
    norm2: Vec<Norm>,
    cond_layer: (Vec<f32>, Vec<f32>), // [HIDDEN, GIN, 1]
    emb_lang: Vec<f32>,               // [n_lang, GIN]
    proj: (Vec<f32>, Vec<f32>),       // [2*HIDDEN, HIDDEN, 1]
}

/// Encoder output: the encoder features `x` and the split statistics.
pub(crate) struct EncoderOut {
    /// Encoder features `[HIDDEN, T]` (feeds the duration predictor).
    pub x: Vec<f32>,
    /// Prior mean `m_p` `[HIDDEN, T]`.
    pub m_p: Vec<f32>,
    /// Prior log-std `logs_p` `[HIDDEN, T]`.
    pub logs_p: Vec<f32>,
    /// Phoneme count `T`.
    pub t: usize,
}

impl TextEncoder {
    pub(super) fn load(store: &TensorStore, n_vocab: usize, n_lang: usize) -> Result<Self> {
        let conv1x1 = |name: &str| -> Result<(Vec<f32>, Vec<f32>)> {
            Ok((
                store.tensor_shaped(&format!("{name}.weight"), &[HIDDEN, HIDDEN, 1])?,
                store.tensor_shaped(&format!("{name}.bias"), &[HIDDEN])?,
            ))
        };
        let mut attn = Vec::with_capacity(N_LAYERS);
        let mut ffn = Vec::with_capacity(N_LAYERS);
        let mut norm1 = Vec::with_capacity(N_LAYERS);
        let mut norm2 = Vec::with_capacity(N_LAYERS);
        let rel = 2 * WINDOW_SIZE + 1;
        for i in 0..N_LAYERS {
            let a = format!("enc_p.encoder.attn_layers.{i}");
            attn.push(AttnLayer {
                conv_q: conv1x1(&format!("{a}.conv_q"))?,
                conv_k: conv1x1(&format!("{a}.conv_k"))?,
                conv_v: conv1x1(&format!("{a}.conv_v"))?,
                conv_o: conv1x1(&format!("{a}.conv_o"))?,
                emb_rel_k: store.tensor_shaped(&format!("{a}.emb_rel_k"), &[1, rel, K_CHANNELS])?,
                emb_rel_v: store.tensor_shaped(&format!("{a}.emb_rel_v"), &[1, rel, K_CHANNELS])?,
            });
            let f = format!("enc_p.encoder.ffn_layers.{i}");
            ffn.push(FfnLayer {
                conv_1: (
                    store.tensor_shaped(
                        &format!("{f}.conv_1.weight"),
                        &[FFN_CHANNELS, HIDDEN, FFN_KERNEL],
                    )?,
                    store.tensor_shaped(&format!("{f}.conv_1.bias"), &[FFN_CHANNELS])?,
                ),
                conv_2: (
                    store.tensor_shaped(
                        &format!("{f}.conv_2.weight"),
                        &[HIDDEN, FFN_CHANNELS, FFN_KERNEL],
                    )?,
                    store.tensor_shaped(&format!("{f}.conv_2.bias"), &[HIDDEN])?,
                ),
            });
            norm1.push(load_norm(
                store,
                &format!("enc_p.encoder.norm_layers_1.{i}"),
            )?);
            norm2.push(load_norm(
                store,
                &format!("enc_p.encoder.norm_layers_2.{i}"),
            )?);
        }

        Ok(Self {
            emb: store.tensor_shaped("enc_p.emb.weight", &[n_vocab, HIDDEN])?,
            n_vocab,
            attn,
            norm1,
            ffn,
            norm2,
            cond_layer: (
                store.tensor_shaped("enc_p.cond_layer.weight", &[HIDDEN, GIN, 1])?,
                store.tensor_shaped("enc_p.cond_layer.bias", &[HIDDEN])?,
            ),
            emb_lang: store.tensor_shaped("emb_lang.weight", &[n_lang, GIN])?,
            proj: (
                store.tensor_shaped("enc_p.proj.weight", &[2 * HIDDEN, HIDDEN, 1])?,
                store.tensor_shaped("enc_p.proj.bias", &[2 * HIDDEN])?,
            ),
        })
    }

    /// The global conditioning vector `g = emb_lang[lid]` `[GIN]` (shared by the
    /// encoder, duration predictor, flow and decoder).
    pub(super) fn lang_conditioning(&self, lid: i64) -> Vec<f32> {
        let lid = lid.max(0) as usize;
        self.emb_lang[lid * GIN..lid * GIN + GIN].to_vec()
    }

    /// Runs the encoder for a phoneme id sequence under language `lid`.
    pub(super) fn forward(&self, phoneme_ids: &[i64], lid: i64) -> Result<EncoderOut> {
        let t = phoneme_ids.len();
        if t == 0 {
            return Err(VokraError::InvalidArgument(
                "text encoder: empty phoneme sequence".to_owned(),
            ));
        }

        // Embedding × sqrt(HIDDEN), laid out [HIDDEN, T].
        let scale = (HIDDEN as f32).sqrt();
        let mut x = vec![0.0f32; HIDDEN * t];
        for (ti, &id) in phoneme_ids.iter().enumerate() {
            if id < 0 || id as usize >= self.n_vocab {
                return Err(VokraError::InvalidArgument(format!(
                    "phoneme id {id} out of range 0..{}",
                    self.n_vocab
                )));
            }
            let row = id as usize * HIDDEN;
            for c in 0..HIDDEN {
                x[c * t + ti] = self.emb[row + c] * scale;
            }
        }

        // 6 transformer layers.
        for l in 0..N_LAYERS {
            let y = self.attention(&x, l, t);
            let sum = add(&x, &y);
            x = nn::layer_norm_channels(
                &sum,
                HIDDEN,
                t,
                &self.norm1[l].gamma,
                &self.norm1[l].beta,
                config::LAYER_NORM_EPS,
            );
            let y = self.ffn(&x, l, t);
            let sum = add(&x, &y);
            x = nn::layer_norm_channels(
                &sum,
                HIDDEN,
                t,
                &self.norm2[l].gamma,
                &self.norm2[l].beta,
                config::LAYER_NORM_EPS,
            );
        }

        // Global conditioning: x += cond_layer(emb_lang[lid]) broadcast over T.
        let lid = lid.max(0) as usize;
        let g = &self.emb_lang[lid * GIN..lid * GIN + GIN];
        // cond_layer is Conv1d(GIN, HIDDEN, 1): out[c] = sum_i W[c,i]·g[i] + b[c].
        let mut cg = self.cond_layer.1.clone();
        let (cw, _) = &self.cond_layer;
        #[allow(clippy::needless_range_loop)] // channel-major matrix indexing
        for c in 0..HIDDEN {
            let wrow = c * GIN;
            let mut acc = cg[c];
            for i in 0..GIN {
                acc += cw[wrow + i] * g[i];
            }
            cg[c] = acc;
        }
        for c in 0..HIDDEN {
            for ti in 0..t {
                x[c * t + ti] += cg[c];
            }
        }

        // proj → stats [2*HIDDEN, T]; split into m_p / logs_p.
        let (pw, pb) = &self.proj;
        let (stats, _) = nn::conv1d(&x, HIDDEN, t, pw, 2 * HIDDEN, 1, Some(pb), 1, 0, 1, 1);
        let m_p = stats[..HIDDEN * t].to_vec();
        let logs_p = stats[HIDDEN * t..].to_vec();
        Ok(EncoderOut { x, m_p, logs_p, t })
    }

    /// FFN: conv_1 (same-pad k3) → ReLU → conv_2 (same-pad k3).
    fn ffn(&self, x: &[f32], layer: usize, t: usize) -> Vec<f32> {
        let f = &self.ffn[layer];
        let pad = (FFN_KERNEL - 1) / 2;
        let (w1, b1) = &f.conv_1;
        let (mut h, _) = nn::conv1d(
            x,
            HIDDEN,
            t,
            w1,
            FFN_CHANNELS,
            FFN_KERNEL,
            Some(b1),
            1,
            pad,
            1,
            1,
        );
        for v in &mut h {
            *v = v.max(0.0); // ReLU (default FFN activation)
        }
        let (w2, b2) = &f.conv_2;
        let (out, _) = nn::conv1d(
            &h,
            FFN_CHANNELS,
            t,
            w2,
            HIDDEN,
            FFN_KERNEL,
            Some(b2),
            1,
            pad,
            1,
            1,
        );
        out
    }

    /// Relative-position multi-head self-attention (`window_size = 4`).
    fn attention(&self, x: &[f32], layer: usize, t: usize) -> Vec<f32> {
        let a = &self.attn[layer];
        let q = conv1x1(x, &a.conv_q, t);
        let k = conv1x1(x, &a.conv_k, t);
        let v = conv1x1(x, &a.conv_v, t);
        let s = (K_CHANNELS as f32).sqrt();

        // Relative embeddings sliced/padded to length 2T-1 (shared over heads).
        let rel_k = get_relative_embeddings(&a.emb_rel_k, t);
        let rel_v = get_relative_embeddings(&a.emb_rel_v, t);
        let rel_len = 2 * t - 1;

        let mut out = vec![0.0f32; HIDDEN * t]; // [n_heads*k_channels, T]
        for h in 0..N_HEADS {
            let base = h * K_CHANNELS;
            // scores[i][j] = sum_d (q_h[i][d]/s)·k_h[j][d] + rel_local.
            let mut scores = vec![0.0f32; t * t];
            for i in 0..t {
                for j in 0..t {
                    let mut acc = 0.0f32;
                    for d in 0..K_CHANNELS {
                        acc += q[(base + d) * t + i] * k[(base + d) * t + j];
                    }
                    scores[i * t + j] = acc / s;
                }
            }
            // rel_logits[i][m] = sum_d (q_h[i][d]/s)·rel_k[m][d]; → abs via reshape.
            let mut rel_logits = vec![0.0f32; t * rel_len];
            for i in 0..t {
                for m in 0..rel_len {
                    let mut acc = 0.0f32;
                    for d in 0..K_CHANNELS {
                        acc += q[(base + d) * t + i] * rel_k[m * K_CHANNELS + d];
                    }
                    rel_logits[i * rel_len + m] = acc / s;
                }
            }
            let local = rel_to_abs(&rel_logits, t);
            for idx in 0..t * t {
                scores[idx] += local[idx];
            }
            nn::softmax_rows(&mut scores, t, t);

            // out_h[i][d] = sum_j p[i][j]·v_h[j][d] + rel-value term.
            let rel_weights = abs_to_rel(&scores, t);
            for i in 0..t {
                for d in 0..K_CHANNELS {
                    let mut acc = 0.0f32;
                    for j in 0..t {
                        acc += scores[i * t + j] * v[(base + d) * t + j];
                    }
                    for m in 0..rel_len {
                        acc += rel_weights[i * rel_len + m] * rel_v[m * K_CHANNELS + d];
                    }
                    out[(base + d) * t + i] = acc;
                }
            }
        }
        conv1x1(&out, &a.conv_o, t)
    }
}

fn load_norm(store: &TensorStore, prefix: &str) -> Result<Norm> {
    Ok(Norm {
        gamma: store.tensor_shaped(&format!("{prefix}.gamma"), &[HIDDEN])?,
        beta: store.tensor_shaped(&format!("{prefix}.beta"), &[HIDDEN])?,
    })
}

/// Applies a `Conv1d(HIDDEN, HIDDEN, 1)` (a per-position linear) to `[HIDDEN, T]`.
fn conv1x1(x: &[f32], layer: &(Vec<f32>, Vec<f32>), t: usize) -> Vec<f32> {
    let (w, b) = layer;
    let (out, _) = nn::conv1d(x, HIDDEN, t, w, HIDDEN, 1, Some(b), 1, 0, 1, 1);
    out
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// Slices/pads the `[2·window+1, k_channels]` relative embedding table to the
/// `[2T-1, k_channels]` window this sequence length needs
/// (`attentions.py::_get_relative_embeddings`).
fn get_relative_embeddings(emb: &[f32], t: usize) -> Vec<f32> {
    // emb stored as [1, 2*window+1, k] — drop the leading unit dim.
    let src_rows = 2 * WINDOW_SIZE + 1;
    let k = K_CHANNELS;
    let pad = (t as isize - (WINDOW_SIZE as isize + 1)).max(0) as usize;
    let slice_start = ((WINDOW_SIZE as isize + 1) - t as isize).max(0) as usize;
    let rel_len = 2 * t - 1;

    // Padded table has `pad` zero rows on each side.
    let padded_rows = src_rows + 2 * pad;
    let mut padded = vec![0.0f32; padded_rows * k];
    for r in 0..src_rows {
        padded[(r + pad) * k..(r + pad) * k + k].copy_from_slice(&emb[r * k..r * k + k]);
    }
    padded[slice_start * k..(slice_start + rel_len) * k].to_vec()
}

/// `_relative_position_to_absolute_position` for one head: `[T, 2T-1]` → `[T, T]`.
fn rel_to_abs(x: &[f32], t: usize) -> Vec<f32> {
    let rel_len = 2 * t - 1;
    // 1. pad each row by one trailing zero → [T, 2T].
    let mut flat = Vec::with_capacity(t * 2 * t + (t.saturating_sub(1)));
    for i in 0..t {
        flat.extend_from_slice(&x[i * rel_len..i * rel_len + rel_len]);
        flat.push(0.0);
    }
    // 2. pad the flat vector by T-1 trailing zeros.
    flat.extend(std::iter::repeat_n(0.0, t - 1));
    // 3. view as [T+1, 2T-1] and take [:T, T-1:].
    let cols = 2 * t - 1;
    let mut out = vec![0.0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            out[i * t + j] = flat[i * cols + (t - 1) + j];
        }
    }
    out
}

/// `_absolute_position_to_relative_position` for one head: `[T, T]` → `[T, 2T-1]`.
fn abs_to_rel(x: &[f32], t: usize) -> Vec<f32> {
    let rel_len = 2 * t - 1;
    // 1. pad each row by T-1 trailing zeros → [T, 2T-1].
    // 2. flatten, then 3. prepend T zeros.
    let mut flat = vec![0.0f32; t];
    for i in 0..t {
        flat.extend_from_slice(&x[i * t..i * t + t]);
        flat.extend(std::iter::repeat_n(0.0, t - 1));
    }
    // 4. view as [T, 2T] and drop the first column.
    let mut out = vec![0.0f32; t * rel_len];
    for i in 0..t {
        for j in 0..rel_len {
            out[i * rel_len + j] = flat[i * 2 * t + 1 + j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rel_to_abs_shapes_and_shift() {
        // T=2: rel_len=3. Input rows [a b c; d e f].
        // PyTorch result [:2, 1:] after the pad/reshape is:
        //   flat = [a b c 0 d e f 0] + [0]  (len 9)
        //   view [3,3] = [[a b c],[0 d e],[f 0 0]]; take [:2,1:] = [[b c],[d e]].
        let x = [1., 2., 3., 4., 5., 6.];
        let out = rel_to_abs(&x, 2);
        assert_eq!(out, [2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn abs_to_rel_inverse_shape() {
        // T=2 → output [2,3].
        let x = [1., 2., 3., 4.];
        let out = abs_to_rel(&x, 2);
        assert_eq!(out.len(), 2 * (2 * 2 - 1));
    }

    #[test]
    fn abs_to_rel_shifts_values() {
        // T=2, x = [[1,2],[3,4]]. Algorithm: prepend T=2 zeros; per row append
        // the row then T-1=1 zero → flat = [0,0, 1,2,0, 3,4,0]; view as [T,2T]
        // = [[0,0,1,2],[0,3,4,0]] and drop the first column → [[0,1,2],[3,4,0]].
        let x = [1., 2., 3., 4.];
        let out = abs_to_rel(&x, 2);
        assert_eq!(out, [0.0, 1.0, 2.0, 3.0, 4.0, 0.0]);
    }

    #[test]
    fn get_rel_embeddings_length() {
        // window=4 → 9 rows. T=3 → need 2T-1=5 rows, no padding (slice inside).
        let emb: Vec<f32> = (0..9 * K_CHANNELS).map(|i| i as f32).collect();
        let out = get_relative_embeddings(&emb, 3);
        assert_eq!(out.len(), 5 * K_CHANNELS);
    }
}
