//! Mimi neural chain — encoder (audio → RVQ tokens) and neural decoder
//! (RVQ features → 24 kHz PCM), shared by CSM (M4-05) and Moshi (M4-06).
//!
//! # Boundary (ADR M4-05 §D1-(c), settled with M4-04 / M4-06)
//!
//! `vokra_ops::mimi_rvq` (M3-06 / M4-04) is the RVQ **op** family:
//! codes → summed `[time, quantizer_dim]` feature vectors, plus the paged
//! per-codebook store. Its output is *features, not PCM* (its rustdoc says
//! so explicitly). The two ends of the neural chain —
//!
//! - [`MimiEncoder`]: PCM → SEANet conv downsampling → transformer
//!   bottleneck → RVQ **quantize** (nearest-codebook search, the decode
//!   lookup's inverse), and
//! - [`MimiNeuralDecoder`]: features → transformer → upsampling →
//!   SEANet decoder → PCM
//!
//! — are **model components with learned weights** (conv stacks,
//! transformer, projections), not pure ops, so they live here in
//! `vokra-models` as a shared module. M4-05 lands them (first consumer =
//! CSM); M4-06 Moshi consumes them unchanged (its T04〜T08 duplicate
//! tickets collapse to "consume + Moshi-specific deltas" — the exclusivity
//! rule both WP ADRs record).
//!
//! The RVQ codebook tables themselves stay `vokra_ops::mimi_rvq::
//! CodebookTable` — the encoder's quantizer searches the same tables the
//! decoder looks up, so the quantization table is never held twice.
//!
//! # Upstream anchor (ADR M4-05 §D2 — transcribed, never invented)
//!
//! `kyutai-labs/moshi` `moshi/models/loaders.py` + `moshi/modules/
//! seanet.py`: SEANet (dimension 512, n_filters 64, ratios [8, 6, 5, 4],
//! ELU, causal), 8-layer d=512 transformer (LayerNorm + RoPE +
//! layer_scale 0.01), SplitResidualVectorQuantizer (dim 256, n_q 32,
//! bins 2048, in/out 512), 24 kHz / 12.5 Hz. The Rust implementation is
//! **config-driven** (`vokra.mimi.*` GGUF chunk) — tiny synthesized
//! configs drive the tests; the real shapes arrive with the converter +
//! T29 weights.
//!
//! # Weight license (NOTICE)
//!
//! Mimi weights are Kyutai **CC-BY 4.0** (`AttributionRequired`) — the
//! NOTICE attribution covers the codec *including* the encoder / neural
//! decoder weights consumed here (M4-05-T27).

pub mod config;

pub use config::{MimiNeuralConfig, MimiQuantizerConfig, MimiSeanetConfig, MimiTransformerConfig};
