//! Kokoro-82M text encoder — phoneme_ids → `[t, hidden_dim]` features
//! (M2-07-T12).
//!
//! Scaffold-level concrete forward, as documented in the M2-07 plan §3
//! "Edit group C": embedding lookup → per-token LayerNorm(gamma, beta) →
//! linear projection. Every weight is loaded verbatim from the GGUF the
//! `vokra-convert::models::kokoro` converter (M2-07-T07) will emit, using
//! the tensor-name mirror rule (safetensors → GGUF verbatim). The exact
//! Kokoro upstream module wiring (BiLSTM vs. transformer, activation kind)
//! is TBD pending the M2-07-T02 upstream inspection; when the real
//! architecture lands, this file's `forward` is the seam that is refined,
//! and the loader-side shape checks in [`Self::new`] keep guarding against
//! silent mis-loads (FR-EX-08).
//!
//! Determinism: no RNG. Two identical inputs produce identical outputs
//! (asserted by the synthetic parity test at the bottom of this file).

use vokra_core::{Result, VokraError};

use super::config::KokoroConfig;
use super::weights::TensorStore;

/// LayerNorm epsilon, kept explicit to make future PyTorch parity easy to
/// audit (StyleTTS 2 派生 uses 1e-5; see M2-07-T02 upstream inspection).
const LAYER_NORM_EPS: f32 = 1e-5;

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

/// The Kokoro text encoder.
///
/// The layout is deliberately simple for the M2-07 scaffold:
/// - `emb`      : `[n_vocab, hidden_dim]` row-major phoneme embedding table;
/// - `ln_gamma` / `ln_beta` : LayerNorm affine params of length `hidden_dim`;
/// - `proj_w`   : `[hidden_dim, hidden_dim]` row-major linear projection;
/// - `proj_b`   : `[hidden_dim]` projection bias.
///
/// The verbatim tensor names below mirror what the safetensors → GGUF
/// converter (M2-07-T07) will write; they use the "text_encoder.*" module
/// prefix from the upstream Kokoro-82M checkpoint and will be re-audited
/// against T02's upstream fact-finding.
#[derive(Debug)]
pub struct TextEncoder {
    n_vocab: usize,
    hidden_dim: usize,
    emb: Vec<f32>,
    ln_gamma: Vec<f32>,
    ln_beta: Vec<f32>,
    proj_w: Vec<f32>,
    proj_b: Vec<f32>,
}

impl TextEncoder {
    /// Loads the encoder weights from `store` and cross-checks each tensor's
    /// shape against `config` (FR-EX-08: any missing tensor / wrong shape /
    /// wrong dtype fails loudly at load time rather than corrupting a forward
    /// pass).
    ///
    /// This is the primary constructor mandated by the M2-07-T12 change spec.
    /// The [`Self::load`] alias is kept for the [`super::KokoroTts`]
    /// wire-up (M2-07-T09) that already spells the call as `load`.
    pub(crate) fn new(store: &TensorStore, config: &KokoroConfig) -> Result<Self> {
        let hidden = config.hidden_dim;
        let n_vocab = config.phoneme_symbols.len();
        if hidden == 0 {
            return Err(VokraError::InvalidArgument(
                "kokoro text encoder: config.hidden_dim is 0".to_owned(),
            ));
        }
        if n_vocab == 0 {
            return Err(VokraError::InvalidArgument(
                "kokoro text encoder: config.phoneme_symbols is empty".to_owned(),
            ));
        }
        let emb = store.tensor_shaped("text_encoder.embedding.weight", &[n_vocab, hidden])?;
        let ln_gamma = store.tensor_shaped("text_encoder.norm.weight", &[hidden])?;
        let ln_beta = store.tensor_shaped("text_encoder.norm.bias", &[hidden])?;
        let proj_w = store.tensor_shaped("text_encoder.proj.weight", &[hidden, hidden])?;
        let proj_b = store.tensor_shaped("text_encoder.proj.bias", &[hidden])?;
        Ok(Self {
            n_vocab,
            hidden_dim: hidden,
            emb,
            ln_gamma,
            ln_beta,
            proj_w,
            proj_b,
        })
    }

    /// Alias for [`Self::new`], kept so the [`super::KokoroTts`] loader
    /// (M2-07-T09, `from_gguf_with_policy`) continues to compile unchanged.
    #[allow(dead_code)] // called from `KokoroTts::from_gguf_with_policy`
    pub(crate) fn load(store: &TensorStore, config: &KokoroConfig) -> Result<Self> {
        Self::new(store, config)
    }

    /// Test-only constructor that skips the GGUF/TensorStore hop.
    ///
    /// Every buffer must have the exact expected length; a mismatch is a
    /// hard error (matches the [`Self::new`] contract on the real path).
    /// This is the seam the synthetic-parity test uses so we can exercise
    /// the forward path without spinning up a full GGUF fixture.
    #[cfg(test)]
    pub(crate) fn from_weights(
        n_vocab: usize,
        hidden_dim: usize,
        emb: Vec<f32>,
        ln_gamma: Vec<f32>,
        ln_beta: Vec<f32>,
        proj_w: Vec<f32>,
        proj_b: Vec<f32>,
    ) -> Result<Self> {
        if hidden_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "text encoder: hidden_dim must be > 0".to_owned(),
            ));
        }
        if n_vocab == 0 {
            return Err(VokraError::InvalidArgument(
                "text encoder: n_vocab must be > 0".to_owned(),
            ));
        }
        if emb.len() != n_vocab * hidden_dim {
            return Err(VokraError::InvalidArgument(format!(
                "text encoder: emb len {} != n_vocab*hidden_dim {}",
                emb.len(),
                n_vocab * hidden_dim
            )));
        }
        if ln_gamma.len() != hidden_dim || ln_beta.len() != hidden_dim {
            return Err(VokraError::InvalidArgument(
                "text encoder: ln_gamma/ln_beta must have length hidden_dim".to_owned(),
            ));
        }
        if proj_w.len() != hidden_dim * hidden_dim {
            return Err(VokraError::InvalidArgument(
                "text encoder: proj_w must have length hidden_dim*hidden_dim".to_owned(),
            ));
        }
        if proj_b.len() != hidden_dim {
            return Err(VokraError::InvalidArgument(
                "text encoder: proj_b must have length hidden_dim".to_owned(),
            ));
        }
        Ok(Self {
            n_vocab,
            hidden_dim,
            emb,
            ln_gamma,
            ln_beta,
            proj_w,
            proj_b,
        })
    }

    /// The output's hidden dim (== the phoneme embedding dim).
    #[cfg(test)]
    pub(crate) fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Runs the encoder for one phoneme id sequence and returns `[t, hidden_dim]`.
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

        // 1. Embedding lookup, laid out `[t, hidden]` row-major.
        let mut x = vec![0.0f32; t * hidden];
        for (ti, &id) in phoneme_ids.iter().enumerate() {
            if id < 0 || (id as usize) >= self.n_vocab {
                return Err(VokraError::InvalidArgument(format!(
                    "kokoro text encoder: phoneme id {id} out of range 0..{}",
                    self.n_vocab
                )));
            }
            let src = (id as usize) * hidden;
            let dst = ti * hidden;
            x[dst..dst + hidden].copy_from_slice(&self.emb[src..src + hidden]);
        }

        // 2. Per-token LayerNorm over the hidden dim.
        for ti in 0..t {
            let row = &mut x[ti * hidden..(ti + 1) * hidden];
            let mean: f32 = row.iter().sum::<f32>() / hidden as f32;
            let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden as f32;
            let denom = (var + LAYER_NORM_EPS).sqrt();
            for (i, v) in row.iter_mut().enumerate() {
                *v = (*v - mean) / denom * self.ln_gamma[i] + self.ln_beta[i];
            }
        }

        // 3. Linear projection: y[t, c] = Σ_i W[c, i]·x[t, i] + b[c].
        let mut y = vec![0.0f32; t * hidden];
        for ti in 0..t {
            let xrow = ti * hidden;
            let yrow = ti * hidden;
            for c in 0..hidden {
                let wrow = c * hidden;
                let mut acc = self.proj_b[c];
                for i in 0..hidden {
                    acc += self.proj_w[wrow + i] * x[xrow + i];
                }
                y[yrow + c] = acc;
            }
        }

        Ok(Array2 {
            data: y,
            rows: t,
            cols: hidden,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic small-tensor factory (no RNG): fills a length-`n` vector
    /// with the reproducible ramp `seed + i * step` so each test builds the
    /// same weights every run.
    fn ramp(n: usize, seed: f32, step: f32) -> Vec<f32> {
        (0..n).map(|i| seed + i as f32 * step).collect()
    }

    fn build_encoder() -> TextEncoder {
        let n_vocab = 8;
        let hidden = 4;
        let emb = ramp(n_vocab * hidden, 0.01, 0.03);
        // Identity-ish LayerNorm (gamma=1, beta=0) keeps the forward
        // interpretable if we ever hand-check numbers.
        let ln_gamma = vec![1.0f32; hidden];
        let ln_beta = vec![0.0f32; hidden];
        // Non-trivial (but hand-authored) projection so the linear step is
        // exercised for real.
        let proj_w = ramp(hidden * hidden, -0.1, 0.05);
        let proj_b = ramp(hidden, 0.02, 0.01);
        TextEncoder::from_weights(n_vocab, hidden, emb, ln_gamma, ln_beta, proj_w, proj_b)
            .expect("valid synthetic weights should build the encoder")
    }

    #[test]
    fn forward_returns_expected_shape() {
        let enc = build_encoder();
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

    #[test]
    fn forward_is_deterministic_across_two_calls() {
        let enc = build_encoder();
        let a = enc.forward(&[1, 2, 3]).expect("first call");
        let b = enc.forward(&[1, 2, 3]).expect("second call");
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.cols, b.cols);
        // Bit-exact: no RNG anywhere in the encoder.
        assert_eq!(
            a.data, b.data,
            "text encoder must be bit-exact deterministic for identical inputs"
        );
    }

    #[test]
    fn forward_rejects_empty_input() {
        let enc = build_encoder();
        let err = enc.forward(&[]).expect_err("empty input must error");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn forward_rejects_out_of_range_id() {
        let enc = build_encoder();
        // n_vocab is 8; id 99 must fail (FR-EX-08 — no silent clamping).
        let err = enc
            .forward(&[1, 99, 3])
            .expect_err("out-of-range id must error");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn forward_rejects_negative_id() {
        let enc = build_encoder();
        let err = enc
            .forward(&[-1, 1, 2])
            .expect_err("negative id must error");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn from_weights_rejects_wrong_emb_length() {
        // n_vocab*hidden_dim would be 8, but we supply 7 → hard error.
        let err = TextEncoder::from_weights(
            4,
            2,
            vec![0.0f32; 7],
            vec![1.0f32; 2],
            vec![0.0f32; 2],
            vec![0.0f32; 4],
            vec![0.0f32; 2],
        )
        .expect_err("wrong emb length must error");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }
}
