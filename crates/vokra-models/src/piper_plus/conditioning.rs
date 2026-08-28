//! Global language and optional speaker conditioning `g` for piper-plus.
//!
//! Legacy distributed voices use `g = emb_lang[lid]`. Zero-shot voices add
//! `spk_proj` and use `g = spk_proj(speaker_embedding) + emb_lang[lid]`.
//! `spk_proj` is a `Linear → LayerNorm → GELU(erf) → Linear` MLP; with a
//! *zero* speaker embedding it still contributes its bias / LayerNorm / GELU
//! path (it is **not** the identity), so the v7 reference parity exercises it.
//!
//! Verified against the committed v7 fixture `g.f32` (`parity_v7`).

use super::config::{Dims, LAYER_NORM_EPS};
use super::nn;
use super::weights::TensorStore;
use vokra_core::{Result, VokraError};

/// Complete zero-shot speaker-projection MLP. It is optional as a group, but
/// once `spk_proj.0.weight` is present every member is required and
/// shape-checked at load time.
struct SpeakerProjection {
    /// `spk_proj.0`: Linear `spk_emb_dim → gin`.
    l0: (Vec<f32>, Vec<f32>),
    /// `spk_proj.1`: LayerNorm over `gin`.
    ln: (Vec<f32>, Vec<f32>),
    /// `spk_proj.3`: Linear `gin → gin`.
    l3: (Vec<f32>, Vec<f32>),
    spk_emb_dim: usize,
}

/// Optional speaker-projection MLP plus the always-present language table.
pub(super) struct Conditioning {
    speaker_projection: Option<SpeakerProjection>,
    /// `emb_lang.weight` `[n_lang, gin]`.
    emb_lang: Vec<f32>,
    gin: usize,
    n_lang: usize,
}

impl Conditioning {
    /// Loads `emb_lang` and, when present, the complete `spk_proj` group.
    pub(super) fn load(store: &TensorStore, dims: &Dims, n_lang: usize) -> Result<Self> {
        let gin = dims.gin;
        const PROJECTION_TENSORS: &[&str] = &[
            "spk_proj.0.weight",
            "spk_proj.0.bias",
            "spk_proj.1.weight",
            "spk_proj.1.bias",
            "spk_proj.3.weight",
            "spk_proj.3.bias",
        ];
        let any_projection = PROJECTION_TENSORS.iter().any(|name| store.contains(name));
        let all_projection = PROJECTION_TENSORS.iter().all(|name| store.contains(name));
        if any_projection && !all_projection {
            let missing = PROJECTION_TENSORS
                .iter()
                .filter(|name| !store.contains(name))
                .copied()
                .collect::<Vec<_>>()
                .join(", ");
            return Err(VokraError::InvalidArgument(format!(
                "piper voice GGUF has incomplete spk_proj group; missing {missing}"
            )));
        }
        if all_projection != dims.spk_emb_dim.is_some() {
            return Err(VokraError::InvalidArgument(
                "piper voice GGUF spk_proj presence changed while loading".into(),
            ));
        }
        let speaker_projection = if let Some(spk_emb_dim) = dims.spk_emb_dim {
            Some(SpeakerProjection {
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
                spk_emb_dim,
            })
        } else {
            None
        };
        Ok(Self {
            speaker_projection,
            emb_lang: store.tensor_shaped("emb_lang.weight", &[n_lang, gin])?,
            gin,
            n_lang,
        })
    }

    /// The external speaker-embedding width this voice's `spk_proj` expects
    /// (`spk_proj.0.weight` dim 1). For the zero-shot v7 voice this is 192 — the
    /// CAM++ output — so [`PiperPlusTts::embed_reference`](super::PiperPlusTts::embed_reference)
    /// can check the encoder matches before wiring its embedding in.
    pub(super) fn spk_emb_dim(&self) -> Option<usize> {
        self.speaker_projection.as_ref().map(|p| p.spk_emb_dim)
    }

    /// Global conditioning (`[gin]`): language-only for a legacy voice, or
    /// `spk_proj(speaker_embedding) + emb_lang[lid]` for a zero-shot voice.
    ///
    /// `None` uses the zero vector for a zero-shot voice (where
    /// `spk_proj(0) ≠ 0`). A supplied embedding on a legacy voice, or a wrong
    /// width on a zero-shot voice, is an explicit error rather than a silent
    /// zero-vector substitution. `lid` is clamped to the language table.
    pub(super) fn g(&self, speaker_embedding: Option<&[f32]>, lid: i64) -> Result<Vec<f32>> {
        let mut g = match (&self.speaker_projection, speaker_embedding) {
            (None, None) => vec![0.0; self.gin],
            (None, Some(_)) => {
                return Err(VokraError::InvalidArgument(
                    "piper TTS: this legacy voice has no spk_proj; speaker_embedding is unsupported"
                        .into(),
                ));
            }
            (Some(projection), embedding) => {
                let zeros = vec![0.0f32; projection.spk_emb_dim];
                let spk = embedding.unwrap_or(zeros.as_slice());
                if spk.len() != projection.spk_emb_dim {
                    return Err(VokraError::InvalidArgument(format!(
                        "piper TTS: speaker_embedding has {} values, expected {}",
                        spk.len(),
                        projection.spk_emb_dim
                    )));
                }
                // spk_proj: Linear → LayerNorm(gin) → GELU(erf) → Linear.
                let (w0, b0) = &projection.l0;
                let mut h = nn::linear(w0, b0, spk);
                let (lw, lb) = &projection.ln;
                h = nn::layer_norm_channels(&h, self.gin, 1, lw, lb, LAYER_NORM_EPS);
                for v in &mut h {
                    *v = nn::gelu(*v);
                }
                let (w3, b3) = &projection.l3;
                nn::linear(w3, b3, &h)
            }
        };
        // + emb_lang[lid], broadcast add of the language row.
        let lid = (lid.max(0) as usize).min(self.n_lang.saturating_sub(1));
        let base = lid * self.gin;
        for (c, gv) in g.iter_mut().enumerate() {
            *gv += self.emb_lang[base + c];
        }
        Ok(g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_conditioning_is_language_only_and_rejects_speaker_input() {
        let conditioning = Conditioning {
            speaker_projection: None,
            emb_lang: vec![1.0, 2.0, 3.0, 4.0],
            gin: 2,
            n_lang: 2,
        };

        assert_eq!(conditioning.g(None, 1).expect("language row"), [3.0, 4.0]);
        let error = conditioning
            .g(Some(&[0.0]), 0)
            .expect_err("legacy voice must reject speaker input");
        assert!(
            matches!(error, VokraError::InvalidArgument(ref message) if message.contains("has no spk_proj"))
        );
    }

    #[test]
    fn zero_shot_conditioning_rejects_wrong_speaker_width() {
        let conditioning = Conditioning {
            speaker_projection: Some(SpeakerProjection {
                l0: (vec![1.0, 1.0], vec![0.0]),
                ln: (vec![1.0], vec![0.0]),
                l3: (vec![1.0], vec![0.0]),
                spk_emb_dim: 2,
            }),
            emb_lang: vec![0.0],
            gin: 1,
            n_lang: 1,
        };

        let error = conditioning
            .g(Some(&[0.0]), 0)
            .expect_err("wrong speaker width must fail loudly");
        assert!(
            matches!(error, VokraError::InvalidArgument(ref message) if message.contains("expected 2"))
        );
        assert_eq!(
            conditioning.g(None, 0).expect("zero speaker default").len(),
            1
        );
    }
}
