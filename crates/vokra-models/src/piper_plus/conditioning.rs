//! Global speaker/language conditioning `g` for piper-plus (M1 zero-shot v7).
//!
//! The MB-iSTFT-VITS2 conditioning vector is
//! `g = spk_proj(speaker_embedding) + emb_lang[lid]` (`[gin]`) — shared by the
//! text encoder, duration predictor, flow and decoder. `spk_proj` is a
//! `Linear → LayerNorm → GELU(erf) → Linear` MLP over the external speaker
//! embedding; with a *zero* speaker embedding it still contributes the
//! bias / LayerNorm / GELU path (it is **not** the identity), so the reference
//! parity exercises it. The single-speaker distributed voices never carried
//! `spk_proj`, so the whole module is only loaded for a FiLM (v7) voice.
//!
//! Verified against the committed v7 fixture `g.f32` (`parity_v7`).

use super::config::{Dims, LAYER_NORM_EPS};
use super::nn;
use super::weights::TensorStore;
use vokra_core::Result;

/// Speaker-projection MLP (`spk_proj`) plus the language embedding table.
pub(super) struct Conditioning {
    /// `spk_proj.0`: Linear `spk_emb_dim → gin` (`weight [gin, spk_emb_dim]`).
    l0: (Vec<f32>, Vec<f32>),
    /// `spk_proj.1`: LayerNorm over `gin` (`weight`/`bias` `[gin]`).
    ln: (Vec<f32>, Vec<f32>),
    /// `spk_proj.3`: Linear `gin → gin` (`weight [gin, gin]`).
    l3: (Vec<f32>, Vec<f32>),
    /// `emb_lang.weight` `[n_lang, gin]`.
    emb_lang: Vec<f32>,
    gin: usize,
    spk_emb_dim: usize,
    n_lang: usize,
}

impl Conditioning {
    /// Loads `spk_proj` + `emb_lang` from the voice, sized from [`Dims`].
    pub(super) fn load(store: &TensorStore, dims: &Dims, n_lang: usize) -> Result<Self> {
        let gin = dims.gin;
        let spk_emb_dim = dims.spk_emb_dim;
        Ok(Self {
            l0: (
                store.tensor_shaped("spk_proj.0.weight", &[gin, spk_emb_dim])?,
                store.tensor_shaped("spk_proj.0.bias", &[gin])?,
            ),
            ln: (
                store.tensor_shaped("spk_proj.1.weight", &[gin])?,
                store.tensor_shaped("spk_proj.1.bias", &[gin])?,
            ),
            l3: (
                store.tensor_shaped("spk_proj.3.weight", &[gin, gin])?,
                store.tensor_shaped("spk_proj.3.bias", &[gin])?,
            ),
            emb_lang: store.tensor_shaped("emb_lang.weight", &[n_lang, gin])?,
            gin,
            spk_emb_dim,
            n_lang,
        })
    }

    /// Global conditioning `g = spk_proj(speaker_embedding) + emb_lang[lid]`
    /// (`[gin]`).
    ///
    /// A `None` (or wrong-length) speaker embedding uses the zero vector — the
    /// deterministic zero-shot default and the reference-parity input (note
    /// `spk_proj(0) ≠ 0`). `lid` is clamped to the language table.
    pub(super) fn g(&self, speaker_embedding: Option<&[f32]>, lid: i64) -> Vec<f32> {
        let zeros = vec![0.0f32; self.spk_emb_dim];
        let spk = match speaker_embedding {
            Some(s) if s.len() == self.spk_emb_dim => s,
            _ => zeros.as_slice(),
        };
        // spk_proj: Linear → LayerNorm(gin) → GELU(erf) → Linear.
        let (w0, b0) = &self.l0;
        let mut h = nn::linear(w0, b0, spk);
        let (lw, lb) = &self.ln;
        h = nn::layer_norm_channels(&h, self.gin, 1, lw, lb, LAYER_NORM_EPS);
        for v in &mut h {
            *v = nn::gelu(*v);
        }
        let (w3, b3) = &self.l3;
        let mut g = nn::linear(w3, b3, &h);
        // + emb_lang[lid], broadcast add of the language row.
        let lid = (lid.max(0) as usize).min(self.n_lang.saturating_sub(1));
        let base = lid * self.gin;
        for (c, gv) in g.iter_mut().enumerate() {
            *gv += self.emb_lang[base + c];
        }
        g
    }
}
