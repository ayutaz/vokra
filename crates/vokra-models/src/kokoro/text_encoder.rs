//! Kokoro-82M text encoder — phoneme_ids → `[t, hidden_dim]` features
//! (M2-07-T12 → **T13-alpha rewrite, 2026-07-07**).
//!
//! # Architecture — bound to the upstream manifest
//!
//! ```text
//! Embedding(178, 512)
//! → 3× [ WeightNormedConv1d(512 → 512, k=5, pad=2)
//!        + per-channel affine (γ, β)
//!        + LeakyReLU(0.1) ]
//! → BiLSTM(input=512, hidden=256, bidirectional → out=512)
//! ```
//!
//! The layout above is the T13-alpha rewrite bound to the upstream tensor
//! manifest at `crates/vokra-models/src/kokoro/data/upstream_tensors_v1_0.tsv`
//! (dumped from `hexgrad/Kokoro-82M kokoro-v1_0.pth` on 2026-07-07 — see
//! `docs/adr/0007-kokoro-native.md` §"T02 upstream inspection findings").
//!
//! # Design notes
//!
//! * The **per-block "LN"** (`cnn.i.1.gamma` / `cnn.i.1.beta`) is a plain
//!   channel-wise affine (γ · x + β), NOT a full LayerNorm — the two tensors
//!   are `[512]` (per-channel) and there is no normalisation kernel between
//!   them. Verified against the manifest shapes.
//! * The **WeightNormed conv** is reconstructed at load time via
//!   [`super::nn::weight_norm_reconstruct_1d`] from `weight_g[512, 1, 1]`
//!   and `weight_v[512, 512, 5]`; the runtime does not see the two-tensor
//!   parameterisation.
//! * The **BiLSTM** uses PyTorch's canonical `weight_ih_l0[4·H, I]` +
//!   `weight_hh_l0[4·H, H]` + `bias_ih_l0[4·H]` + `bias_hh_l0[4·H]` layout
//!   with a mirrored `..._reverse` set for the backward direction. Gates are
//!   stacked `i | f | g | o`. Hidden dim is derived as `hidden_dim / 2`
//!   (`hidden_dim` must be even — 512 in the real checkpoint).
//! * Every tensor is bound at load time via [`super::weights::TensorStore::tensor_shaped`];
//!   a missing tensor or a shape mismatch is a loud
//!   [`VokraError::InvalidArgument`] (FR-EX-08 — no silent architecture drift).
//!
//! Determinism: no RNG. Two identical inputs produce identical outputs
//! (asserted by the deterministic-forward test below).

use vokra_core::{Result, VokraError};

use super::config::KokoroConfig;
use super::nn::{BiLstm1d, LRELU_SLOPE, conv1d, leaky_relu, weight_norm_reconstruct_1d};
use super::weights::TensorStore;
use crate::compute::Compute;

/// Kernel size of the text-encoder Conv1d blocks — fixed at 5 by the upstream
/// checkpoint (`weight_v` shape `[512, 512, 5]`). Not runtime-configurable.
const CNN_KERNEL: usize = 5;

/// Number of stacked `WeightNormedConv1d + affine + LeakyReLU` blocks
/// (`cnn.{0,1,2}` in the manifest).
const NUM_CNN_BLOCKS: usize = 3;

/// Same-padding for a kernel-5 stride-1 dilation-1 Conv1d
/// (`out_len = in_len + 2·pad − (kernel − 1) − 1 + 1`, so `pad = 2`).
const CNN_PAD: usize = 2;

/// Minimal row-major 2-D array used as the encoder output. Kept private to
/// this file (the crate uses raw `Vec<f32>` + shape for its other layers);
/// the M2-07 change spec asks for `Array2<f32>` as the return type, so we
/// name it that way here.
///
/// Shape is `[rows, cols]` and `data.len() == rows * cols`; `data` is stored
/// in row-major order (`data[r * cols + c]`).
#[derive(Debug, Clone)]
#[allow(dead_code)] // fields consumed by tests + the T18 e2e wire-up
pub struct Array2<T> {
    /// Row-major backing storage of length `rows * cols`.
    pub data: Vec<T>,
    /// Number of rows (phoneme count `t` for this encoder's output).
    pub rows: usize,
    /// Number of columns (hidden dimension for this encoder's output).
    pub cols: usize,
}

/// The Kokoro text encoder, bound to the upstream
/// `text_encoder.module.*` tensor names (see the module docstring).
///
/// * `emb`      — `[n_vocab, hidden_dim]` row-major phoneme embedding table.
/// * `conv_ws[i]` — reconstructed Conv1d weights `[hidden_dim, hidden_dim, 5]`
///   for block `i` (row-major, `[out_ch, in_ch, kernel]`).
/// * `conv_bs[i]` — `[hidden_dim]` Conv1d bias for block `i`.
/// * `norm_gs[i]` / `norm_bs[i]` — `[hidden_dim]` per-channel affine
///   (γ · x + β) applied after the Conv1d bias.
/// * `lstm`     — bidirectional LSTM with input=`hidden_dim`,
///   hidden=`hidden_dim / 2`, output width=`hidden_dim`.
#[derive(Debug)]
pub struct TextEncoder {
    n_vocab: usize,
    hidden_dim: usize,
    emb: Vec<f32>,
    conv_ws: [Vec<f32>; NUM_CNN_BLOCKS],
    conv_bs: [Vec<f32>; NUM_CNN_BLOCKS],
    norm_gs: [Vec<f32>; NUM_CNN_BLOCKS],
    norm_bs: [Vec<f32>; NUM_CNN_BLOCKS],
    lstm: BiLstm1d,
}

impl TextEncoder {
    /// Loads the encoder weights from `store` and cross-checks each tensor's
    /// shape against `config` (FR-EX-08: any missing tensor / wrong shape /
    /// wrong dtype fails loudly at load time rather than corrupting a forward
    /// pass).
    ///
    /// The tensor name catalog is verbatim from the upstream manifest at
    /// `crates/vokra-models/src/kokoro/data/upstream_tensors_v1_0.tsv` — the
    /// `.module.` prefix comes from PyTorch's `nn.DataParallel` wrap around
    /// every top-level submodule in the original training script.
    pub(crate) fn new(store: &TensorStore, config: &KokoroConfig) -> Result<Self> {
        let hidden = config.hidden_dim;
        let n_vocab = config.phoneme_symbols.len();
        if hidden == 0 {
            return Err(VokraError::InvalidArgument(
                "kokoro text encoder: config.hidden_dim is 0".to_owned(),
            ));
        }
        if hidden % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro text encoder: config.hidden_dim ({hidden}) must be even \
                 (the BiLSTM hidden width is hidden_dim / 2)"
            )));
        }
        if n_vocab == 0 {
            return Err(VokraError::InvalidArgument(
                "kokoro text encoder: config.phoneme_symbols is empty".to_owned(),
            ));
        }
        let lstm_hidden = hidden / 2;

        // 1. Embedding [n_vocab, hidden].
        let emb =
            store.tensor_shaped("text_encoder.module.embedding.weight", &[n_vocab, hidden])?;

        // 2. Three CNN blocks: WeightNormed Conv1d + per-channel affine.
        //    Rust doesn't let us [_; NUM_CNN_BLOCKS] a Vec directly without
        //    Copy, so build empties and fill by index.
        let mut conv_ws: [Vec<f32>; NUM_CNN_BLOCKS] = Default::default();
        let mut conv_bs: [Vec<f32>; NUM_CNN_BLOCKS] = Default::default();
        let mut norm_gs: [Vec<f32>; NUM_CNN_BLOCKS] = Default::default();
        let mut norm_bs: [Vec<f32>; NUM_CNN_BLOCKS] = Default::default();
        for i in 0..NUM_CNN_BLOCKS {
            // WeightNorm split: weight_g[out_ch, 1, 1] + weight_v[out_ch, in_ch, kernel].
            let g_name = format!("text_encoder.module.cnn.{i}.0.weight_g");
            let v_name = format!("text_encoder.module.cnn.{i}.0.weight_v");
            let b_name = format!("text_encoder.module.cnn.{i}.0.bias");
            let gamma_name = format!("text_encoder.module.cnn.{i}.1.gamma");
            let beta_name = format!("text_encoder.module.cnn.{i}.1.beta");
            let w_g = store.tensor_shaped(&g_name, &[hidden, 1, 1])?;
            let w_v = store.tensor_shaped(&v_name, &[hidden, hidden, CNN_KERNEL])?;
            let bias = store.tensor_shaped(&b_name, &[hidden])?;
            let gamma = store.tensor_shaped(&gamma_name, &[hidden])?;
            let beta = store.tensor_shaped(&beta_name, &[hidden])?;
            conv_ws[i] = weight_norm_reconstruct_1d(&w_g, &w_v, hidden, hidden, CNN_KERNEL);
            conv_bs[i] = bias;
            norm_gs[i] = gamma;
            norm_bs[i] = beta;
        }

        // 3. Bidirectional LSTM (input=hidden, hidden=hidden/2, output=hidden).
        let four_h = 4 * lstm_hidden;
        let w_ih_fwd =
            store.tensor_shaped("text_encoder.module.lstm.weight_ih_l0", &[four_h, hidden])?;
        let w_hh_fwd = store.tensor_shaped(
            "text_encoder.module.lstm.weight_hh_l0",
            &[four_h, lstm_hidden],
        )?;
        let b_ih_fwd = store.tensor_shaped("text_encoder.module.lstm.bias_ih_l0", &[four_h])?;
        let b_hh_fwd = store.tensor_shaped("text_encoder.module.lstm.bias_hh_l0", &[four_h])?;
        let w_ih_rev = store.tensor_shaped(
            "text_encoder.module.lstm.weight_ih_l0_reverse",
            &[four_h, hidden],
        )?;
        let w_hh_rev = store.tensor_shaped(
            "text_encoder.module.lstm.weight_hh_l0_reverse",
            &[four_h, lstm_hidden],
        )?;
        let b_ih_rev =
            store.tensor_shaped("text_encoder.module.lstm.bias_ih_l0_reverse", &[four_h])?;
        let b_hh_rev =
            store.tensor_shaped("text_encoder.module.lstm.bias_hh_l0_reverse", &[four_h])?;

        let lstm = BiLstm1d::new(
            hidden,
            lstm_hidden,
            w_ih_fwd,
            w_hh_fwd,
            b_ih_fwd,
            b_hh_fwd,
            w_ih_rev,
            w_hh_rev,
            b_ih_rev,
            b_hh_rev,
        )?;

        Ok(Self {
            n_vocab,
            hidden_dim: hidden,
            emb,
            conv_ws,
            conv_bs,
            norm_gs,
            norm_bs,
            lstm,
        })
    }

    /// Alias for [`Self::new`], kept so the [`super::KokoroTts`] loader
    /// (M2-07-T09, `from_gguf_with_policy`) continues to compile unchanged.
    #[allow(dead_code)] // called from `KokoroTts::from_gguf_with_policy`
    pub(crate) fn load(store: &TensorStore, config: &KokoroConfig) -> Result<Self> {
        Self::new(store, config)
    }

    /// The output's hidden dim (== the phoneme embedding dim == 2·lstm_hidden).
    #[cfg(test)]
    pub(crate) fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Runs the encoder for one phoneme id sequence and returns `[t, hidden_dim]`.
    ///
    /// Pipeline:
    ///
    /// 1. Embedding lookup → `[t, hidden]` row-major.
    /// 2. Transpose → `[hidden, t]` channel-major (the layout
    ///    [`super::nn::conv1d`] expects).
    /// 3. For each of the 3 CNN blocks: Conv1d(k=5, pad=2) → add bias →
    ///    per-channel affine (`γ · x + β`) → LeakyReLU(0.1).
    /// 4. Transpose back → `[t, hidden]` row-major (BiLSTM input layout).
    /// 5. BiLSTM forward → `[t, hidden]` (2·lstm_hidden = hidden).
    ///
    /// Errors on empty input or on any id outside `0..n_vocab`
    /// (FR-EX-08 — no silent fallback / clamping).
    #[allow(dead_code)] // called by the T18 e2e path
    pub(crate) fn forward(&self, phoneme_ids: &[i64]) -> Result<Array2<f32>> {
        if phoneme_ids.is_empty() {
            return Err(VokraError::InvalidArgument(
                "kokoro text encoder: empty phoneme id sequence".to_owned(),
            ));
        }
        let hidden = self.hidden_dim;
        let t = phoneme_ids.len();

        // 1. Embedding lookup → [t, hidden] row-major.
        let mut x_row = vec![0.0f32; t * hidden];
        for (ti, &id) in phoneme_ids.iter().enumerate() {
            if id < 0 || (id as usize) >= self.n_vocab {
                return Err(VokraError::InvalidArgument(format!(
                    "kokoro text encoder: phoneme id {id} out of range 0..{}",
                    self.n_vocab
                )));
            }
            let src = (id as usize) * hidden;
            let dst = ti * hidden;
            x_row[dst..dst + hidden].copy_from_slice(&self.emb[src..src + hidden]);
        }

        // 2. Transpose to [hidden, t] channel-major for Conv1d.
        let mut x_ch = vec![0.0f32; hidden * t];
        for ti in 0..t {
            for c in 0..hidden {
                x_ch[c * t + ti] = x_row[ti * hidden + c];
            }
        }

        // 3. CNN stack. All blocks run on CPU via the same im2col + GEMM path
        //    the piper decoder uses; a future GPU dispatch swaps this
        //    `Compute::cpu()` for `Compute::for_backend(...)` at the T18
        //    wire-up. Each block: Conv1d(k=5, pad=2) → +bias → γ·x+β → LeakyReLU.
        let compute = Compute::cpu();
        let mut cur = x_ch;
        let mut cur_len = t;
        for i in 0..NUM_CNN_BLOCKS {
            let (mut conv_out, out_len) = conv1d(
                &compute,
                &cur,
                hidden,
                cur_len,
                &self.conv_ws[i],
                hidden,
                CNN_KERNEL,
                Some(&self.conv_bs[i]),
                /*stride*/ 1,
                CNN_PAD,
                /*dilation*/ 1,
                /*groups*/ 1,
            );
            // Per-channel affine (γ · x + β) applied to the [hidden, out_len]
            // channel-major buffer.
            for c in 0..hidden {
                let g = self.norm_gs[i][c];
                let b = self.norm_bs[i][c];
                let row = &mut conv_out[c * out_len..c * out_len + out_len];
                for v in row.iter_mut() {
                    *v = *v * g + b;
                }
            }
            leaky_relu(&mut conv_out, LRELU_SLOPE);
            cur = conv_out;
            cur_len = out_len;
        }

        // 4. Transpose back [hidden, t] → [t, hidden] for BiLSTM.
        let mut lstm_input = vec![0.0f32; cur_len * hidden];
        for c in 0..hidden {
            for ti in 0..cur_len {
                lstm_input[ti * hidden + c] = cur[c * cur_len + ti];
            }
        }

        // 5. BiLSTM forward → [t, hidden] row-major.
        let lstm_out = self.lstm.forward(&lstm_input, cur_len);
        debug_assert_eq!(lstm_out.len(), cur_len * hidden);

        Ok(Array2 {
            data: lstm_out,
            rows: cur_len,
            cols: hidden,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::{
        KEY_HIDDEN_DIM, KEY_ISTFT_HOP, KEY_ISTFT_N_FFT, KEY_ISTFT_WIN_LENGTH, KEY_N_DECODER_LAYERS,
        KEY_N_TEXT_LAYERS, KEY_NUM_VOICES, KEY_PHONEME_SYMBOLS, KEY_SAMPLE_RATE, KEY_STYLE_DIM,
        KEY_VOICE_NAMES,
    };
    use super::*;
    use vokra_core::gguf::{
        GgmlType, GgufArray, GgufBuilder, GgufFile, GgufMetadataValue, GgufValueType,
    };

    /// F32 helper: write `n` zeros as GGUF-ready LE bytes.
    fn zeros_bytes(n: usize) -> Vec<u8> {
        vec![0u8; n * 4]
    }

    /// F32 helper: write a deterministic ramp `seed + i·step` as GGUF-ready
    /// LE bytes so a wrong index inside the encoder is visible in the output.
    fn ramp_bytes(n: usize, seed: f32, step: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|i| (seed + i as f32 * step).to_le_bytes())
            .collect()
    }

    fn str_array(items: &[&str]) -> GgufMetadataValue {
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: items
                .iter()
                .map(|s| GgufMetadataValue::String((*s).to_owned()))
                .collect(),
        })
    }

    /// Builds a synthetic Kokoro voice GGUF that carries every
    /// `text_encoder.module.*` tensor the [`TextEncoder`] loader binds.
    ///
    /// * `hidden` — must be even; `lstm_hidden = hidden / 2`.
    /// * `n_vocab` — the phoneme symbol count; ids `< n_vocab` are valid.
    /// * `ramp_weights` — if true, embedding + Conv1d weights are non-zero
    ///   ramps so a mis-index during the forward propagates to the output
    ///   (used by the deterministic-output test). If false, all weights are
    ///   zero, useful for the "loads and stays finite" smoke test.
    fn build_synthetic_gguf(hidden: usize, n_vocab: usize, ramp_weights: bool) -> Vec<u8> {
        assert!(hidden % 2 == 0, "test hidden must be even");
        let lstm_hidden = hidden / 2;
        let four_h = 4 * lstm_hidden;

        let mut b = GgufBuilder::new();
        // Config: everything required by [`KokoroConfig::from_gguf`]. Values are
        // arbitrary; only `hidden_dim` and `phoneme_symbols` are consumed by the
        // text encoder.
        b.add_u32(KEY_SAMPLE_RATE, 24_000);
        b.add_u32(KEY_STYLE_DIM, 8);
        b.add_u32(KEY_NUM_VOICES, 2);
        b.add_u32(KEY_HIDDEN_DIM, hidden as u32);
        b.add_u32(KEY_N_TEXT_LAYERS, 3);
        b.add_u32(KEY_N_DECODER_LAYERS, 2);
        b.add_u32(KEY_ISTFT_N_FFT, 20);
        b.add_u32(KEY_ISTFT_HOP, 5);
        b.add_u32(KEY_ISTFT_WIN_LENGTH, 20);
        let phoneme_symbols: Vec<String> = (0..n_vocab).map(|i| format!("p{i}")).collect();
        let phoneme_refs: Vec<&str> = phoneme_symbols.iter().map(String::as_str).collect();
        b.add_metadata(KEY_PHONEME_SYMBOLS, str_array(&phoneme_refs));
        b.add_metadata(KEY_VOICE_NAMES, str_array(&["af", "am"]));

        // Embedding [n_vocab, hidden]. Non-zero ramp lets deterministic tests
        // observe a real signal; zero payloads keep the smoke test bounded.
        let emb_bytes = if ramp_weights {
            ramp_bytes(n_vocab * hidden, 0.01, 0.03)
        } else {
            zeros_bytes(n_vocab * hidden)
        };
        b.add_tensor(
            "text_encoder.module.embedding.weight",
            GgmlType::F32,
            vec![n_vocab as u64, hidden as u64],
            emb_bytes,
        )
        .expect("emb");

        // Three CNN blocks.
        for i in 0..NUM_CNN_BLOCKS {
            let plane = hidden * hidden * CNN_KERNEL;
            let g_bytes = if ramp_weights {
                // Nonzero `g` so `w = g·v/||v||` scales are meaningful.
                ramp_bytes(hidden, 1.0 + 0.1 * i as f32, 0.01)
            } else {
                zeros_bytes(hidden)
            };
            let v_bytes = if ramp_weights {
                // Nonzero `v` so the reconstructed weight is finite (see also
                // the T16 zero-norm guard in `weight_norm_reconstruct_1d`).
                ramp_bytes(plane, 0.001 * (i as f32 + 1.0), 0.001)
            } else {
                zeros_bytes(plane)
            };
            b.add_tensor(
                &format!("text_encoder.module.cnn.{i}.0.weight_g"),
                GgmlType::F32,
                vec![hidden as u64, 1, 1],
                g_bytes,
            )
            .expect("weight_g");
            b.add_tensor(
                &format!("text_encoder.module.cnn.{i}.0.weight_v"),
                GgmlType::F32,
                vec![hidden as u64, hidden as u64, CNN_KERNEL as u64],
                v_bytes,
            )
            .expect("weight_v");
            b.add_tensor(
                &format!("text_encoder.module.cnn.{i}.0.bias"),
                GgmlType::F32,
                vec![hidden as u64],
                zeros_bytes(hidden),
            )
            .expect("cnn bias");
            // γ = 1s so the affine is non-trivial (a γ=0 collapse would
            // silently skip the scale-path regression).
            let gamma_bytes: Vec<u8> = (0..hidden).flat_map(|_| 1.0f32.to_le_bytes()).collect();
            b.add_tensor(
                &format!("text_encoder.module.cnn.{i}.1.gamma"),
                GgmlType::F32,
                vec![hidden as u64],
                gamma_bytes,
            )
            .expect("gamma");
            b.add_tensor(
                &format!("text_encoder.module.cnn.{i}.1.beta"),
                GgmlType::F32,
                vec![hidden as u64],
                zeros_bytes(hidden),
            )
            .expect("beta");
        }

        // LSTM: forward + reverse, each 4 tensors.
        for suffix in ["", "_reverse"] {
            b.add_tensor(
                &format!("text_encoder.module.lstm.weight_ih_l0{suffix}"),
                GgmlType::F32,
                vec![four_h as u64, hidden as u64],
                zeros_bytes(four_h * hidden),
            )
            .expect("lstm w_ih");
            b.add_tensor(
                &format!("text_encoder.module.lstm.weight_hh_l0{suffix}"),
                GgmlType::F32,
                vec![four_h as u64, lstm_hidden as u64],
                zeros_bytes(four_h * lstm_hidden),
            )
            .expect("lstm w_hh");
            b.add_tensor(
                &format!("text_encoder.module.lstm.bias_ih_l0{suffix}"),
                GgmlType::F32,
                vec![four_h as u64],
                zeros_bytes(four_h),
            )
            .expect("lstm b_ih");
            b.add_tensor(
                &format!("text_encoder.module.lstm.bias_hh_l0{suffix}"),
                GgmlType::F32,
                vec![four_h as u64],
                zeros_bytes(four_h),
            )
            .expect("lstm b_hh");
        }

        b.to_bytes().expect("serialize")
    }

    /// Builds a [`TextEncoder`] from the synthetic GGUF above — the only path
    /// the T13-alpha rewrite exercises, since the real-weight path requires
    /// a full 82M-parameter checkpoint and a bert branch that this WP does not
    /// yet build.
    fn build_encoder(hidden: usize, n_vocab: usize, ramp_weights: bool) -> TextEncoder {
        let bytes = build_synthetic_gguf(hidden, n_vocab, ramp_weights);
        let file = GgufFile::parse(bytes).expect("parse synthetic Kokoro GGUF");
        let config = KokoroConfig::from_gguf(&file).expect("valid config");
        let store = TensorStore::new(file);
        TextEncoder::new(&store, &config).expect("valid synthetic tensors")
    }

    /// The T13-alpha loader must bind every `text_encoder.module.*` tensor at
    /// its documented shape; a synthetic GGUF that carries them all builds
    /// successfully.
    #[test]
    fn loads_all_tensors_from_synthetic_gguf() {
        let enc = build_encoder(
            /*hidden=*/ 16, /*n_vocab=*/ 6, /*ramp_weights=*/ false,
        );
        assert_eq!(enc.hidden_dim(), 16);
    }

    /// The forward output shape must be `[t, hidden_dim]` with `t = phoneme
    /// count` — the invariant every downstream stage (prosody, length
    /// regulator, decoder) shape-checks against.
    #[test]
    fn forward_returns_expected_shape() {
        let enc = build_encoder(16, 6, false);
        let out = enc.forward(&[1, 2, 3]).expect("forward should succeed");
        assert_eq!(out.rows, 3, "rows must equal phoneme count t");
        assert_eq!(out.cols, enc.hidden_dim(), "cols must equal hidden_dim");
        assert_eq!(
            out.data.len(),
            out.rows * out.cols,
            "row-major storage length must equal rows*cols"
        );
        assert!(
            out.data.iter().all(|v| v.is_finite()),
            "forward must produce only finite values"
        );
    }

    /// The encoder must be deterministic: same input → bit-identical output.
    /// Any RNG or uninitialised buffer would fail this. Uses ramp weights so
    /// the output is not a trivial zero vector.
    #[test]
    fn forward_is_deterministic_across_two_calls() {
        let enc = build_encoder(16, 6, /*ramp_weights=*/ true);
        let a = enc.forward(&[1, 2, 3]).expect("first call");
        let b = enc.forward(&[1, 2, 3]).expect("second call");
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.cols, b.cols);
        assert_eq!(
            a.data, b.data,
            "text encoder must be bit-exact deterministic for identical inputs"
        );
    }

    #[test]
    fn forward_rejects_empty_input() {
        let enc = build_encoder(16, 6, false);
        let err = enc.forward(&[]).expect_err("empty input must error");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn forward_rejects_out_of_range_id() {
        let enc = build_encoder(16, 6, false);
        // n_vocab is 6; id 99 must fail (FR-EX-08 — no silent clamping).
        let err = enc
            .forward(&[1, 99, 3])
            .expect_err("out-of-range id must error");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn forward_rejects_negative_id() {
        let enc = build_encoder(16, 6, false);
        let err = enc
            .forward(&[-1, 1, 2])
            .expect_err("negative id must error");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    /// The loader must reject an odd `hidden_dim` since the BiLSTM hidden
    /// width is `hidden_dim / 2` — 511 would silently truncate to 255 rather
    /// than fail (FR-EX-08).
    #[test]
    fn new_rejects_odd_hidden_dim() {
        // hidden=15 is odd; the loader must error before touching any tensor.
        let mut b = GgufBuilder::new();
        b.add_u32(KEY_SAMPLE_RATE, 24_000);
        b.add_u32(KEY_STYLE_DIM, 8);
        b.add_u32(KEY_NUM_VOICES, 2);
        b.add_u32(KEY_HIDDEN_DIM, 15);
        b.add_u32(KEY_N_TEXT_LAYERS, 3);
        b.add_u32(KEY_N_DECODER_LAYERS, 2);
        b.add_u32(KEY_ISTFT_N_FFT, 20);
        b.add_u32(KEY_ISTFT_HOP, 5);
        b.add_u32(KEY_ISTFT_WIN_LENGTH, 20);
        b.add_metadata(KEY_PHONEME_SYMBOLS, str_array(&["a"]));
        b.add_metadata(KEY_VOICE_NAMES, str_array(&["af"]));
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let config = KokoroConfig::from_gguf(&file).expect("valid config");
        let store = TensorStore::new(file);
        let err = TextEncoder::new(&store, &config).expect_err("odd hidden must fail");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("even"),
                    "error should mention 'even'; got: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// A missing `text_encoder.module.embedding.weight` must fail at the
    /// very first `tensor_shaped` call with a message that names the tensor
    /// (FR-EX-08 red line R4 — no silent architecture drift).
    #[test]
    fn new_reports_missing_embedding_tensor() {
        // Build a config-only GGUF (no tensors) — the loader must fail on the
        // first tensor lookup.
        let mut b = GgufBuilder::new();
        b.add_u32(KEY_SAMPLE_RATE, 24_000);
        b.add_u32(KEY_STYLE_DIM, 8);
        b.add_u32(KEY_NUM_VOICES, 2);
        b.add_u32(KEY_HIDDEN_DIM, 16);
        b.add_u32(KEY_N_TEXT_LAYERS, 3);
        b.add_u32(KEY_N_DECODER_LAYERS, 2);
        b.add_u32(KEY_ISTFT_N_FFT, 20);
        b.add_u32(KEY_ISTFT_HOP, 5);
        b.add_u32(KEY_ISTFT_WIN_LENGTH, 20);
        b.add_metadata(KEY_PHONEME_SYMBOLS, str_array(&["a"]));
        b.add_metadata(KEY_VOICE_NAMES, str_array(&["af"]));
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let config = KokoroConfig::from_gguf(&file).expect("valid config");
        let store = TensorStore::new(file);
        let err = TextEncoder::new(&store, &config).expect_err("missing tensor must fail");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("text_encoder.module.embedding.weight"),
                    "error should name the missing tensor; got: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }
}
