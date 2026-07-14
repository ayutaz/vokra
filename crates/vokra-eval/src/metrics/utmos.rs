//! UTMOS neural MOS predictor — config-driven wav2vec2-SSL + regression-head
//! **skeleton** (M4-18 T06/T07; FR-OP-93, FR-TL-03, NFR-QL-02).
//!
//! # Status: weight-deferred skeleton (M4-18 kickoff gate = NO-GO-defer)
//!
//! The M4-18 kickoff-week gate fired the ratified auto-defer rule (UTMOS
//! weight + license had not arrived from the owner — see
//! `docs/adr/M4-18-utmos-gate.md`): the **weight-dependent** halves (upstream
//! checkpoint converter mapping, real-reference parity fixtures, upstream
//! architecture pinning) are deferred to a v1.0.x patch. What lands here is
//! the **weight-independent** half:
//!
//! - a fully config-driven forward skeleton for the ratified
//!   characterization "wav2vec2 SSL + regression head"
//!   (`docs/m4-scope-expansion-2026-07-13.md` §BIG-7) — CNN feature encoder →
//!   bidirectional transformer encoder → regression head → one MOS scalar;
//! - synthesized, seed-deterministic weights ([`UtmosWeights::synthesized`],
//!   SplitMix64 + Xavier — the M3-09 `LlmWeights::synthesized` precedent) so
//!   shape / determinism / finiteness are machine-verified **without** the
//!   real checkpoint;
//! - the GGUF binding ([`Utmos::from_gguf`]) for the `vokra.utmos.*` schema
//!   (ADR `docs/adr/M4-18-utmos-arch.md` §(c)/(d)) so the owner-side
//!   flip-the-switch only needs the converter + fixtures, no runtime change.
//!
//! **No numerical agreement with upstream SaruLab UTMOS22 is claimed here.**
//! The exact upstream layer stack (feature-encoder normalization, positional
//! conv, listener/domain embeddings, BLSTM …) is pinned at flip time against
//! the upstream implementation — inventing those constants now would violate
//! the CLAUDE.md hallucination ban. The [`ARCH_VARIANT_V0`] string is the
//! guard: a GGUF converted for a *different* variant is rejected loudly
//! (never silently mis-scored, FR-EX-08).
//!
//! # Crate placement (ADR `docs/adr/M4-18-utmos-arch.md` §(a))
//!
//! Lives in `vokra-eval` (option B'): the CLI (`vokra-eval utmos …`) and the
//! degradation gates wire it without the banned `vokra-eval → vokra-models`
//! edge. The GEMM / conv / softmax / layer-norm / GELU bodies are the
//! first-party `vokra-backend-cpu::kernels` safe wrappers — the same kernels
//! the models' Compute seam dispatches, so nothing is re-implemented
//! (NFR-DS-02 stays zero-dep; this is a *downward* first-party edge).
//! CPU-only by design: eval is an offline/CI path, not an RTF surface.
//!
//! # No silent fallback (FR-EX-08)
//!
//! - sample-rate mismatch → loud [`VokraError::InvalidArgument`] (no silent
//!   resample);
//! - input shorter than the conv stack's receptive field → loud error;
//! - non-finite input samples → loud error (a NaN would silently poison the
//!   score);
//! - missing / mis-shaped GGUF tensor → loud [`VokraError::ModelLoad`]
//!   naming the tensor (never a zero-fill);
//! - unknown arch variant / activation / norm / pool string → loud error.

use vokra_backend_cpu::kernels;
use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::rng::SplitMix64;
use vokra_core::{Result, VokraError};

use super::{AudioMosMetric, Direction, Metric};

/// `vokra.model.arch` value for UTMOS GGUFs.
pub const ARCH: &str = "utmos";

/// The only architecture variant this skeleton implements. The flip-time
/// upstream pin bumps this (e.g. `wav2vec2_regression.v1`) if the real
/// SaruLab UTMOS22 stack differs — an unknown variant is a loud
/// [`VokraError::ModelLoad`], never a silent mis-score.
pub const ARCH_VARIANT_V0: &str = "wav2vec2_regression.v0";

// --- `vokra.utmos.*` metadata keys (ADR M4-18-utmos-arch §(c)) --------------

/// `vokra.utmos.arch.variant` — implemented-variant guard (STRING).
pub const KEY_ARCH_VARIANT: &str = "vokra.utmos.arch.variant";
/// `vokra.utmos.sample_rate` — required input sample rate (UINT32).
pub const KEY_SAMPLE_RATE: &str = "vokra.utmos.sample_rate";
/// `vokra.utmos.conv.channels` — feature-encoder per-layer out-channels
/// (ARRAY<UINT32>).
pub const KEY_CONV_CHANNELS: &str = "vokra.utmos.conv.channels";
/// `vokra.utmos.conv.kernels` — per-layer kernel widths (ARRAY<UINT32>).
pub const KEY_CONV_KERNELS: &str = "vokra.utmos.conv.kernels";
/// `vokra.utmos.conv.strides` — per-layer strides (ARRAY<UINT32>).
pub const KEY_CONV_STRIDES: &str = "vokra.utmos.conv.strides";
/// `vokra.utmos.conv.activation` — feature-encoder activation (STRING;
/// `"gelu"` is the only v0 value).
pub const KEY_CONV_ACTIVATION: &str = "vokra.utmos.conv.activation";
/// `vokra.utmos.transformer.n_layer` — encoder block count (UINT32).
pub const KEY_TF_N_LAYER: &str = "vokra.utmos.transformer.n_layer";
/// `vokra.utmos.transformer.n_head` — attention head count (UINT32).
pub const KEY_TF_N_HEAD: &str = "vokra.utmos.transformer.n_head";
/// `vokra.utmos.transformer.hidden_dim` — transformer width `d` (UINT32).
pub const KEY_TF_HIDDEN_DIM: &str = "vokra.utmos.transformer.hidden_dim";
/// `vokra.utmos.transformer.ffn_dim` — MLP intermediate width (UINT32).
pub const KEY_TF_FFN_DIM: &str = "vokra.utmos.transformer.ffn_dim";
/// `vokra.utmos.transformer.norm` — block norm placement (STRING;
/// `"pre"` / `"post"`).
pub const KEY_TF_NORM: &str = "vokra.utmos.transformer.norm";
/// `vokra.utmos.transformer.ln_eps` — LayerNorm epsilon (FLOAT32).
pub const KEY_TF_LN_EPS: &str = "vokra.utmos.transformer.ln_eps";
/// `vokra.utmos.head.dims` — regression-head linear output dims, last must
/// be 1 (ARRAY<UINT32>).
pub const KEY_HEAD_DIMS: &str = "vokra.utmos.head.dims";
/// `vokra.utmos.head.pool` — time pooling placement (STRING;
/// `"mean_before"` / `"mean_after"`).
pub const KEY_HEAD_POOL: &str = "vokra.utmos.head.pool";
/// `vokra.utmos.head.scale` — score affine scale (FLOAT32, optional,
/// default `1.0` — identity is the only safe default).
pub const KEY_HEAD_SCALE: &str = "vokra.utmos.head.scale";
/// `vokra.utmos.head.offset` — score affine offset (FLOAT32, optional,
/// default `0.0`).
pub const KEY_HEAD_OFFSET: &str = "vokra.utmos.head.offset";

/// Feature-encoder activation. v0 implements GELU only (the wav2vec2-family
/// staple); anything else in the GGUF is a loud error, and the enum exists so
/// the flip-time pin can add variants without stringly-typed dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvActivation {
    /// Exact (erf-based) GELU — `vokra_backend_cpu::kernels::gelu_f32`.
    Gelu,
}

/// Transformer block norm placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformerNorm {
    /// Pre-norm: `x + Attn(LN1(x))`, `x + MLP(LN2(x))`, plus a required
    /// final `utmos.enc_ln` after the last block.
    Pre,
    /// Post-norm: `LN1(x + Attn(x))`, `LN2(x + MLP(x))`; `utmos.enc_ln`
    /// must be absent.
    Post,
}

/// Regression-head time pooling placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadPool {
    /// Mean-pool the `[t, d]` hidden over time first, then run the head
    /// linears on the pooled `[1, d]` row.
    MeanBefore,
    /// Run the head linears frame-wise (`[t, 1]`), then mean-pool the
    /// per-frame scores.
    MeanAfter,
}

/// Resolved `vokra.utmos.*` hyper-parameters. Every field is read from the
/// GGUF metadata ([`UtmosConfig::from_gguf`]) — nothing architecture-defining
/// is hard-coded, and required keys have **no silent defaults** (CLAUDE.md
/// hallucination ban + FR-EX-08). Only `head_scale` / `head_offset` default
/// (to the identity affine `1.0` / `0.0`).
#[derive(Debug, Clone, PartialEq)]
pub struct UtmosConfig {
    /// Required input waveform sample rate (Hz). A mismatching clip is a
    /// loud error — the metric never silently resamples.
    pub sample_rate: u32,
    /// Feature-encoder per-layer out-channels (layer 0 input is the mono
    /// waveform, i.e. 1 channel).
    pub conv_channels: Vec<usize>,
    /// Per-layer kernel widths (same length as `conv_channels`).
    pub conv_kernels: Vec<usize>,
    /// Per-layer strides (same length as `conv_channels`).
    pub conv_strides: Vec<usize>,
    /// Feature-encoder activation.
    pub conv_activation: ConvActivation,
    /// Transformer encoder block count (>= 1).
    pub n_layer: usize,
    /// Attention head count (`hidden_dim % n_head == 0`).
    pub n_head: usize,
    /// Transformer width `d`.
    pub hidden_dim: usize,
    /// MLP intermediate width.
    pub ffn_dim: usize,
    /// Block norm placement.
    pub norm: TransformerNorm,
    /// LayerNorm epsilon.
    pub ln_eps: f32,
    /// Regression-head linear output dims; the last entry must be `1`
    /// (the MOS scalar).
    pub head_dims: Vec<usize>,
    /// Head time-pooling placement.
    pub head_pool: HeadPool,
    /// Score affine scale (`score = raw * scale + offset`).
    pub head_scale: f32,
    /// Score affine offset.
    pub head_offset: f32,
}

impl UtmosConfig {
    /// Validates the config shape. Every violation is a loud
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate(&self) -> Result<()> {
        Err(VokraError::NotImplemented("UtmosConfig::validate: stub"))
    }

    /// Reads the config from a `vokra.utmos.*` GGUF metadata block,
    /// rejecting an unknown [`KEY_ARCH_VARIANT`] loudly.
    pub fn from_gguf(_file: &GgufFile) -> Result<Self> {
        Err(VokraError::NotImplemented("UtmosConfig::from_gguf: stub"))
    }

    /// The frame count the conv feature encoder yields for `in_len` input
    /// samples (`out = (in - kernel) / stride + 1` per layer, "valid"
    /// padding). An input shorter than a layer's kernel is a loud error
    /// (FR-EX-08 — never an empty silent output).
    pub fn feature_len(&self, _in_len: usize) -> Result<usize> {
        Err(VokraError::NotImplemented("UtmosConfig::feature_len: stub"))
    }
}

/// One conv layer of the feature encoder — weight `[c_out, c_in, k]`
/// row-major plus optional per-out-channel bias.
struct ConvLayer {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    c_in: usize,
    c_out: usize,
    kernel: usize,
    stride: usize,
}

/// A linear layer stored **transposed** (`w_t` is `[d_in, d_out]` row-major)
/// so `Y[t, d_out] = X[t, d_in] @ w_t` is a single row-major GEMM with the
/// optional bias broadcast per output column.
struct Linear {
    w_t: Vec<f32>,
    bias: Option<Vec<f32>>,
    d_in: usize,
    d_out: usize,
}

/// LayerNorm affine parameters.
struct LayerNormW {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

/// One transformer encoder block (bidirectional MHSA + GELU MLP).
struct EncBlock {
    ln1: LayerNormW,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ln2: LayerNormW,
    fc1: Linear,
    fc2: Linear,
}

/// UTMOS weight store. Built either synthesized
/// ([`UtmosWeights::synthesized`]) or from a `vokra.utmos.*` GGUF
/// ([`UtmosWeights::from_gguf`]).
pub struct UtmosWeights {
    conv: Vec<ConvLayer>,
    /// Present iff the last conv channel count differs from `hidden_dim`
    /// (or the GGUF ships `utmos.feat_proj.weight`).
    feat_proj: Option<Linear>,
    blocks: Vec<EncBlock>,
    /// Final LayerNorm — required for [`TransformerNorm::Pre`], forbidden
    /// for [`TransformerNorm::Post`].
    enc_ln: Option<LayerNormW>,
    head: Vec<Linear>,
    /// `true` when built by [`UtmosWeights::synthesized`] — the parity
    /// harness refuses to compare synthesized weights against a real
    /// reference (no fabricated pass).
    pub is_synthesized: bool,
}

impl UtmosWeights {
    /// Builds a synthesized, seed-deterministic weight store (SplitMix64 →
    /// Xavier/Glorot uniform; LayerNorm γ=1, β=0 — the M3-09
    /// `LlmWeights::synthesized` recipe). **Not** a reproduction of any real
    /// checkpoint: it exists so shape / determinism / finiteness are
    /// verifiable without the deferred weights.
    pub fn synthesized(_config: &UtmosConfig, _seed: u64) -> Result<Self> {
        Err(VokraError::NotImplemented(
            "UtmosWeights::synthesized: stub",
        ))
    }

    /// Binds the weight store from a parsed GGUF, verifying every tensor's
    /// dims against `config` (ADR M4-18-utmos-arch §(d) naming). A missing
    /// or mis-shaped tensor is a loud [`VokraError::ModelLoad`] naming it.
    pub fn from_gguf(_file: &GgufFile, _config: &UtmosConfig) -> Result<Self> {
        Err(VokraError::NotImplemented("UtmosWeights::from_gguf: stub"))
    }
}

/// The UTMOS scorer: one waveform in, one MOS scalar out.
///
/// v0 skeleton semantics — see the module docs: config-driven forward,
/// synthesized or GGUF weights, **no upstream numerical claim until the
/// flip-time pin**.
pub struct Utmos {
    config: UtmosConfig,
    weights: UtmosWeights,
}

impl Utmos {
    /// Builds a scorer over synthesized weights (see
    /// [`UtmosWeights::synthesized`]).
    pub fn synthesized(config: UtmosConfig, seed: u64) -> Result<Self> {
        let _ = (&config, seed);
        Err(VokraError::NotImplemented("Utmos::synthesized: stub"))
    }

    /// Binds a scorer from a parsed `vokra.utmos.*` GGUF.
    pub fn from_gguf(_file: &GgufFile) -> Result<Self> {
        Err(VokraError::NotImplemented("Utmos::from_gguf: stub"))
    }

    /// Opens and binds a UTMOS GGUF from `path`.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// The resolved config.
    #[must_use]
    pub fn config(&self) -> &UtmosConfig {
        &self.config
    }

    /// `true` when the weights are synthesized (never claim real-reference
    /// parity over these).
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        self.weights.is_synthesized
    }

    /// Scores one mono clip (`[-1, 1]` PCM at `sample_rate`) → MOS scalar.
    ///
    /// # Errors (FR-EX-08 — all loud, nothing silent)
    ///
    /// - `sample_rate` differing from the config's rate (no silent
    ///   resample);
    /// - empty / non-finite input;
    /// - input shorter than the conv stack's receptive field.
    pub fn score(&self, _audio: &[f32], _sample_rate: u32) -> Result<f64> {
        Err(VokraError::NotImplemented("Utmos::score: stub"))
    }
}

impl Metric for Utmos {
    fn name(&self) -> &str {
        "utmos"
    }

    fn direction(&self) -> Direction {
        Direction::HigherIsBetter
    }
}

impl AudioMosMetric for Utmos {
    fn eval_mos(&self, audio: &[f32], sample_rate: u32) -> Result<f64> {
        self.score(audio, sample_rate)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::chunks::KEY_MODEL_ARCH;
    use vokra_core::gguf::{GgmlType, GgufArray, GgufBuilder, GgufValueType};

    const SEED: u64 = 0x4D34_5F31_385F_5530; // ASCII-ish "M4_18_U0"

    /// A tiny but structurally complete config: 2 conv layers whose last
    /// channel count equals `hidden_dim` (identity feat-proj path), 2
    /// post-norm blocks, a 2-linear head, mean-after pooling.
    fn tiny_config() -> UtmosConfig {
        UtmosConfig {
            sample_rate: 16_000,
            conv_channels: vec![4, 6],
            conv_kernels: vec![5, 3],
            conv_strides: vec![3, 2],
            conv_activation: ConvActivation::Gelu,
            n_layer: 2,
            n_head: 2,
            hidden_dim: 6,
            ffn_dim: 12,
            norm: TransformerNorm::Post,
            ln_eps: 1e-5,
            head_dims: vec![4, 1],
            head_pool: HeadPool::MeanAfter,
            head_scale: 1.0,
            head_offset: 0.0,
        }
    }

    /// A variant that exercises the non-identity feature projection
    /// (`c_last = 4 != d = 6`), pre-norm blocks and mean-before pooling.
    fn proj_pre_config() -> UtmosConfig {
        UtmosConfig {
            conv_channels: vec![4, 4],
            norm: TransformerNorm::Pre,
            head_pool: HeadPool::MeanBefore,
            ..tiny_config()
        }
    }

    fn sine(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 16_000.0).sin())
            .collect()
    }

    fn u32_array(values: &[u32]) -> GgufMetadataValue {
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: values.iter().map(|&v| GgufMetadataValue::U32(v)).collect(),
        })
    }

    /// Writes `config`'s metadata block into `b` (the schema of ADR
    /// M4-18-utmos-arch §(c)) — the same block the flip-time converter
    /// will emit.
    fn seed_metadata(b: &mut GgufBuilder, config: &UtmosConfig) {
        b.add_string(KEY_MODEL_ARCH, ARCH);
        b.add_string(KEY_ARCH_VARIANT, ARCH_VARIANT_V0);
        b.add_u32(KEY_SAMPLE_RATE, config.sample_rate);
        let as_u32 = |v: &[usize]| v.iter().map(|&x| x as u32).collect::<Vec<_>>();
        b.add_metadata(KEY_CONV_CHANNELS, u32_array(&as_u32(&config.conv_channels)));
        b.add_metadata(KEY_CONV_KERNELS, u32_array(&as_u32(&config.conv_kernels)));
        b.add_metadata(KEY_CONV_STRIDES, u32_array(&as_u32(&config.conv_strides)));
        b.add_string(KEY_CONV_ACTIVATION, "gelu");
        b.add_u32(KEY_TF_N_LAYER, config.n_layer as u32);
        b.add_u32(KEY_TF_N_HEAD, config.n_head as u32);
        b.add_u32(KEY_TF_HIDDEN_DIM, config.hidden_dim as u32);
        b.add_u32(KEY_TF_FFN_DIM, config.ffn_dim as u32);
        b.add_string(
            KEY_TF_NORM,
            match config.norm {
                TransformerNorm::Pre => "pre",
                TransformerNorm::Post => "post",
            },
        );
        b.add_f32(KEY_TF_LN_EPS, config.ln_eps);
        b.add_metadata(KEY_HEAD_DIMS, u32_array(&as_u32(&config.head_dims)));
        b.add_string(
            KEY_HEAD_POOL,
            match config.head_pool {
                HeadPool::MeanBefore => "mean_before",
                HeadPool::MeanAfter => "mean_after",
            },
        );
        b.add_f32(KEY_HEAD_SCALE, config.head_scale);
        b.add_f32(KEY_HEAD_OFFSET, config.head_offset);
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn add_f32_tensor(b: &mut GgufBuilder, name: &str, dims: &[usize], data: &[f32]) {
        b.add_tensor(
            name,
            GgmlType::F32,
            dims.iter().map(|&d| d as u64).collect(),
            f32_bytes(data),
        )
        .unwrap_or_else(|e| panic!("add_tensor {name}: {e}"));
    }

    /// Transposes a `[d_in, d_out]`-stored `Linear::w_t` back to the GGUF's
    /// semantic `[d_out, d_in]` layout (ADR §(d): tensors ship `y = W x`
    /// row-major).
    fn untranspose(w_t: &[f32], d_in: usize, d_out: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; d_in * d_out];
        for i in 0..d_in {
            for o in 0..d_out {
                w[o * d_in + i] = w_t[i * d_out + o];
            }
        }
        w
    }

    fn add_linear(b: &mut GgufBuilder, name: &str, lin: &Linear) {
        add_f32_tensor(
            b,
            &format!("{name}.weight"),
            &[lin.d_out, lin.d_in],
            &untranspose(&lin.w_t, lin.d_in, lin.d_out),
        );
        if let Some(bias) = &lin.bias {
            add_f32_tensor(b, &format!("{name}.bias"), &[lin.d_out], bias);
        }
    }

    fn add_ln(b: &mut GgufBuilder, name: &str, ln: &LayerNormW) {
        add_f32_tensor(b, &format!("{name}.weight"), &[ln.gamma.len()], &ln.gamma);
        add_f32_tensor(b, &format!("{name}.bias"), &[ln.beta.len()], &ln.beta);
    }

    /// Serializes synthesized weights into the ADR §(c)/(d) GGUF schema —
    /// the executable documentation of what the flip-time converter emits,
    /// and the writer half of the reader round-trip oracle below.
    fn write_synthesized_gguf(config: &UtmosConfig, seed: u64) -> Vec<u8> {
        let w = UtmosWeights::synthesized(config, seed).expect("synthesized");
        let mut b = GgufBuilder::new();
        seed_metadata(&mut b, config);
        for (i, c) in w.conv.iter().enumerate() {
            add_f32_tensor(
                &mut b,
                &format!("utmos.conv.{i}.weight"),
                &[c.c_out, c.c_in, c.kernel],
                &c.weight,
            );
            if let Some(bias) = &c.bias {
                add_f32_tensor(&mut b, &format!("utmos.conv.{i}.bias"), &[c.c_out], bias);
            }
        }
        if let Some(proj) = &w.feat_proj {
            add_linear(&mut b, "utmos.feat_proj", proj);
        }
        for (i, blk) in w.blocks.iter().enumerate() {
            add_ln(&mut b, &format!("utmos.enc.{i}.ln1"), &blk.ln1);
            add_linear(&mut b, &format!("utmos.enc.{i}.attn.q"), &blk.q);
            add_linear(&mut b, &format!("utmos.enc.{i}.attn.k"), &blk.k);
            add_linear(&mut b, &format!("utmos.enc.{i}.attn.v"), &blk.v);
            add_linear(&mut b, &format!("utmos.enc.{i}.attn.o"), &blk.o);
            add_ln(&mut b, &format!("utmos.enc.{i}.ln2"), &blk.ln2);
            add_linear(&mut b, &format!("utmos.enc.{i}.mlp.fc1"), &blk.fc1);
            add_linear(&mut b, &format!("utmos.enc.{i}.mlp.fc2"), &blk.fc2);
        }
        if let Some(ln) = &w.enc_ln {
            add_ln(&mut b, "utmos.enc_ln", ln);
        }
        for (i, lin) in w.head.iter().enumerate() {
            add_linear(&mut b, &format!("utmos.head.{i}"), lin);
        }
        b.to_bytes().expect("serialize")
    }

    // ---- config validation ------------------------------------------------

    #[test]
    fn tiny_config_validates() {
        tiny_config()
            .validate()
            .expect("tiny config is well-formed");
        proj_pre_config().validate().expect("proj/pre config too");
    }

    #[test]
    fn validate_rejects_malformed_configs() {
        let mut c = tiny_config();
        c.conv_kernels = vec![5]; // length mismatch vs channels
        assert!(matches!(c.validate(), Err(VokraError::InvalidArgument(_))));

        let mut c = tiny_config();
        c.conv_strides[0] = 0;
        assert!(c.validate().is_err(), "zero stride must be rejected");

        let mut c = tiny_config();
        c.n_head = 4; // 6 % 4 != 0
        assert!(c.validate().is_err(), "d % n_head != 0 must be rejected");

        let mut c = tiny_config();
        c.head_dims = vec![4, 2]; // last != 1
        assert!(c.validate().is_err(), "head must end in 1");

        let mut c = tiny_config();
        c.head_dims = vec![];
        assert!(c.validate().is_err(), "empty head must be rejected");

        let mut c = tiny_config();
        c.sample_rate = 0;
        assert!(c.validate().is_err(), "zero sample rate must be rejected");

        let mut c = tiny_config();
        c.n_layer = 0;
        assert!(c.validate().is_err(), "zero-layer encoder must be rejected");

        let mut c = tiny_config();
        c.ln_eps = 0.0;
        assert!(
            c.validate().is_err(),
            "non-positive ln_eps must be rejected"
        );

        let mut c = tiny_config();
        c.head_scale = f32::NAN;
        assert!(c.validate().is_err(), "non-finite affine must be rejected");
    }

    // ---- feature length ---------------------------------------------------

    #[test]
    fn feature_len_matches_valid_conv_formula() {
        let c = tiny_config();
        // layer0: (64 - 5) / 3 + 1 = 20; layer1: (20 - 3) / 2 + 1 = 9.
        assert_eq!(c.feature_len(64).unwrap(), 9);
        // Exactly one frame everywhere: layer0 needs >= 5.
        // in = 7 → layer0 (7-5)/3+1 = 1 → layer1 needs >= 3 → error.
        assert!(c.feature_len(7).is_err());
    }

    #[test]
    fn feature_len_rejects_too_short_input_loudly() {
        let c = tiny_config();
        let err = c.feature_len(4).expect_err("shorter than kernel 5");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
        let msg = format!("{err}");
        assert!(
            msg.contains("too short"),
            "error must say the input is too short, got: {msg}"
        );
    }

    // ---- metadata round-trip ----------------------------------------------

    #[test]
    fn config_gguf_metadata_round_trips() {
        for config in [tiny_config(), proj_pre_config()] {
            let mut b = GgufBuilder::new();
            seed_metadata(&mut b, &config);
            let file = GgufFile::parse(b.to_bytes().unwrap()).expect("parse");
            let read = UtmosConfig::from_gguf(&file).expect("read config");
            assert_eq!(read, config);
        }
    }

    #[test]
    fn config_from_gguf_rejects_unknown_variant() {
        let mut b = GgufBuilder::new();
        seed_metadata(&mut b, &tiny_config());
        // Overwrite the variant with a future/unknown one.
        let mut b2 = GgufBuilder::new();
        let config = tiny_config();
        seed_metadata(&mut b2, &config);
        drop(b);
        // Rebuild with a bogus variant: seed everything, then shadow the key.
        let mut b3 = GgufBuilder::new();
        b3.add_string(KEY_MODEL_ARCH, ARCH);
        b3.add_string(KEY_ARCH_VARIANT, "wav2vec2_regression.v999");
        b3.add_u32(KEY_SAMPLE_RATE, config.sample_rate);
        let file = GgufFile::parse(b3.to_bytes().unwrap()).expect("parse");
        let err = UtmosConfig::from_gguf(&file).expect_err("unknown variant");
        let msg = format!("{err}");
        assert!(
            msg.contains("wav2vec2_regression.v999"),
            "error must name the offending variant, got: {msg}"
        );
    }

    #[test]
    fn config_from_gguf_names_missing_keys() {
        let mut b = GgufBuilder::new();
        b.add_string(KEY_MODEL_ARCH, ARCH);
        b.add_string(KEY_ARCH_VARIANT, ARCH_VARIANT_V0);
        // sample_rate (and everything else) missing.
        let file = GgufFile::parse(b.to_bytes().unwrap()).expect("parse");
        let err = UtmosConfig::from_gguf(&file).expect_err("missing keys");
        let msg = format!("{err}");
        assert!(
            msg.contains(KEY_SAMPLE_RATE),
            "error must name the first missing key, got: {msg}"
        );
    }

    #[test]
    fn config_from_gguf_rejects_unknown_activation_and_pool() {
        let config = tiny_config();
        let mut b = GgufBuilder::new();
        seed_metadata(&mut b, &config);
        // No add_string overwrite API — rebuild with a bad activation.
        let bytes = {
            let mut bad = GgufBuilder::new();
            bad.add_string(KEY_MODEL_ARCH, ARCH);
            bad.add_string(KEY_ARCH_VARIANT, ARCH_VARIANT_V0);
            bad.add_u32(KEY_SAMPLE_RATE, config.sample_rate);
            let as_u32 = |v: &[usize]| v.iter().map(|&x| x as u32).collect::<Vec<_>>();
            bad.add_metadata(KEY_CONV_CHANNELS, u32_array(&as_u32(&config.conv_channels)));
            bad.add_metadata(KEY_CONV_KERNELS, u32_array(&as_u32(&config.conv_kernels)));
            bad.add_metadata(KEY_CONV_STRIDES, u32_array(&as_u32(&config.conv_strides)));
            bad.add_string(KEY_CONV_ACTIVATION, "relu6"); // not implemented in v0
            bad.add_u32(KEY_TF_N_LAYER, config.n_layer as u32);
            bad.add_u32(KEY_TF_N_HEAD, config.n_head as u32);
            bad.add_u32(KEY_TF_HIDDEN_DIM, config.hidden_dim as u32);
            bad.add_u32(KEY_TF_FFN_DIM, config.ffn_dim as u32);
            bad.add_string(KEY_TF_NORM, "post");
            bad.add_f32(KEY_TF_LN_EPS, config.ln_eps);
            let as_u32h = |v: &[usize]| v.iter().map(|&x| x as u32).collect::<Vec<_>>();
            bad.add_metadata(KEY_HEAD_DIMS, u32_array(&as_u32h(&config.head_dims)));
            bad.add_string(KEY_HEAD_POOL, "mean_after");
            bad.to_bytes().unwrap()
        };
        let file = GgufFile::parse(bytes).expect("parse");
        let err = UtmosConfig::from_gguf(&file).expect_err("unknown activation");
        assert!(format!("{err}").contains("relu6"));
    }

    // ---- synthesized weights ----------------------------------------------

    #[test]
    fn synthesized_weights_are_seed_deterministic() {
        let config = tiny_config();
        let a = UtmosWeights::synthesized(&config, SEED).unwrap();
        let b = UtmosWeights::synthesized(&config, SEED).unwrap();
        assert_eq!(a.conv[0].weight, b.conv[0].weight);
        assert_eq!(a.blocks[0].q.w_t, b.blocks[0].q.w_t);
        assert_eq!(a.head.last().unwrap().w_t, b.head.last().unwrap().w_t);
        assert!(a.is_synthesized);

        let c = UtmosWeights::synthesized(&config, SEED ^ 1).unwrap();
        assert_ne!(
            a.conv[0].weight, c.conv[0].weight,
            "different seeds must differ"
        );
    }

    #[test]
    fn synthesized_weights_shapes_follow_config() {
        let config = tiny_config();
        let w = UtmosWeights::synthesized(&config, SEED).unwrap();
        assert_eq!(w.conv.len(), 2);
        assert_eq!(w.conv[0].c_in, 1);
        assert_eq!(w.conv[0].c_out, 4);
        assert_eq!(w.conv[0].weight.len(), 4 * 1 * 5);
        assert_eq!(w.conv[1].c_in, 4);
        assert_eq!(w.conv[1].weight.len(), 6 * 4 * 3);
        // c_last == d → identity projection.
        assert!(w.feat_proj.is_none());
        assert_eq!(w.blocks.len(), 2);
        assert_eq!(w.blocks[0].q.w_t.len(), 6 * 6);
        assert_eq!(w.blocks[0].fc1.w_t.len(), 6 * 12);
        // Post-norm → no final LN.
        assert!(w.enc_ln.is_none());
        assert_eq!(w.head.len(), 2);
        assert_eq!(w.head[0].w_t.len(), 6 * 4);
        assert_eq!(w.head[1].w_t.len(), 4 * 1);

        // Projection + pre-norm variant.
        let config = proj_pre_config();
        let w = UtmosWeights::synthesized(&config, SEED).unwrap();
        let proj = w.feat_proj.as_ref().expect("c_last != d needs a proj");
        assert_eq!(proj.d_in, 4);
        assert_eq!(proj.d_out, 6);
        assert!(w.enc_ln.is_some(), "pre-norm requires the final LN");
    }

    // ---- score (e2e over synthesized weights) ------------------------------

    #[test]
    fn score_is_finite_and_deterministic() {
        for config in [tiny_config(), proj_pre_config()] {
            let m = Utmos::synthesized(config, SEED).unwrap();
            let x = sine(64);
            let s1 = m.score(&x, 16_000).expect("score");
            let s2 = m.score(&x, 16_000).expect("score again");
            assert!(s1.is_finite(), "score must be finite, got {s1}");
            assert_eq!(s1.to_bits(), s2.to_bits(), "bit-identical reruns");
        }
    }

    #[test]
    fn score_depends_on_input_and_seed() {
        let m = Utmos::synthesized(tiny_config(), SEED).unwrap();
        let s_sine = m.score(&sine(64), 16_000).unwrap();
        let zeros = vec![0.0f32; 64];
        let s_zero = m.score(&zeros, 16_000).unwrap();
        assert_ne!(
            s_sine.to_bits(),
            s_zero.to_bits(),
            "different inputs should score differently"
        );

        let m2 = Utmos::synthesized(tiny_config(), SEED ^ 7).unwrap();
        let s_other = m2.score(&sine(64), 16_000).unwrap();
        assert_ne!(
            s_sine.to_bits(),
            s_other.to_bits(),
            "different seeds should score differently"
        );
    }

    #[test]
    fn score_applies_the_affine() {
        let base = Utmos::synthesized(tiny_config(), SEED).unwrap();
        let s = base.score(&sine(64), 16_000).unwrap();

        let mut scaled_cfg = tiny_config();
        scaled_cfg.head_scale = 2.0;
        scaled_cfg.head_offset = 3.0;
        let scaled = Utmos::synthesized(scaled_cfg, SEED).unwrap();
        let s2 = scaled.score(&sine(64), 16_000).unwrap();
        let expected = s * 2.0 + 3.0;
        assert!(
            (s2 - expected).abs() < 1e-9,
            "affine must apply: got {s2}, expected {expected}"
        );
    }

    #[test]
    fn score_rejects_bad_inputs_loudly() {
        let m = Utmos::synthesized(tiny_config(), SEED).unwrap();
        // Wrong sample rate — never silently resampled (FR-EX-08).
        assert!(matches!(
            m.score(&sine(64), 22_050),
            Err(VokraError::InvalidArgument(_))
        ));
        // Empty input.
        assert!(m.score(&[], 16_000).is_err());
        // Too short for the conv stack.
        assert!(m.score(&sine(4), 16_000).is_err());
        // Non-finite samples would silently poison the score.
        let mut x = sine(64);
        x[10] = f32::NAN;
        assert!(m.score(&x, 16_000).is_err());
    }

    // ---- Metric / AudioMosMetric traits ------------------------------------

    #[test]
    fn metric_traits_are_wired() {
        let m = Utmos::synthesized(tiny_config(), SEED).unwrap();
        assert_eq!(m.name(), "utmos");
        assert_eq!(m.direction(), Direction::HigherIsBetter);
        let x = sine(64);
        let direct = m.score(&x, 16_000).unwrap();
        let via_trait = (&m as &dyn AudioMosMetric).eval_mos(&x, 16_000).unwrap();
        assert_eq!(direct.to_bits(), via_trait.to_bits());
    }

    // ---- GGUF weight round-trip (reader oracle) ----------------------------

    #[test]
    fn gguf_round_trip_scores_bit_identically() {
        for config in [tiny_config(), proj_pre_config()] {
            let bytes = write_synthesized_gguf(&config, SEED);
            let file = GgufFile::parse(bytes).expect("parse");
            let loaded = Utmos::from_gguf(&file).expect("bind");
            assert!(
                !loaded.is_synthesized(),
                "GGUF-loaded weights are not flagged synthesized"
            );
            let reference = Utmos::synthesized(config, SEED).unwrap();
            let x = sine(96);
            let a = reference.score(&x, 16_000).unwrap();
            let b = loaded.score(&x, 16_000).unwrap();
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "GGUF round-trip must reproduce the in-memory forward bit-for-bit"
            );
        }
    }

    #[test]
    fn gguf_missing_tensor_is_a_loud_model_load_error() {
        let config = tiny_config();
        let w = UtmosWeights::synthesized(&config, SEED).unwrap();
        let mut b = GgufBuilder::new();
        seed_metadata(&mut b, &config);
        // Ship ONLY the first conv layer — everything else missing.
        add_f32_tensor(
            &mut b,
            "utmos.conv.0.weight",
            &[w.conv[0].c_out, w.conv[0].c_in, w.conv[0].kernel],
            &w.conv[0].weight,
        );
        let file = GgufFile::parse(b.to_bytes().unwrap()).expect("parse");
        let err = Utmos::from_gguf(&file).expect_err("missing tensors");
        let msg = format!("{err}");
        assert!(
            matches!(err, VokraError::ModelLoad(_)) && msg.contains("utmos.conv.1.weight"),
            "must name the first missing tensor, got: {msg}"
        );
    }

    #[test]
    fn gguf_mis_shaped_tensor_is_rejected() {
        let config = tiny_config();
        let mut b = GgufBuilder::new();
        seed_metadata(&mut b, &config);
        // conv.0.weight with the wrong kernel width (4 instead of 5).
        add_f32_tensor(&mut b, "utmos.conv.0.weight", &[4, 1, 4], &vec![0.0; 16]);
        let file = GgufFile::parse(b.to_bytes().unwrap()).expect("parse");
        let err = Utmos::from_gguf(&file).expect_err("mis-shaped tensor");
        let msg = format!("{err}");
        assert!(
            msg.contains("utmos.conv.0.weight"),
            "must name the mis-shaped tensor, got: {msg}"
        );
    }

    #[test]
    fn gguf_post_norm_with_enc_ln_is_rejected() {
        // Post-norm forbids `utmos.enc_ln.*` (strict — no invented semantics
        // for a tensor the variant does not define).
        let config = tiny_config();
        let mut bytes = write_synthesized_gguf(&config, SEED);
        // Re-parse and rebuild with an extra enc_ln pair appended.
        let file = GgufFile::parse(bytes.clone()).expect("parse");
        let mut b = GgufBuilder::new();
        seed_metadata(&mut b, &config);
        for info in file.tensors() {
            let data = file.tensor_bytes(info).to_vec();
            b.add_tensor(&info.name, info.dtype, info.dimensions.clone(), data)
                .unwrap();
        }
        add_f32_tensor(&mut b, "utmos.enc_ln.weight", &[6], &vec![1.0; 6]);
        add_f32_tensor(&mut b, "utmos.enc_ln.bias", &[6], &vec![0.0; 6]);
        bytes = b.to_bytes().unwrap();
        let file = GgufFile::parse(bytes).expect("parse");
        let err = Utmos::from_gguf(&file).expect_err("post-norm + enc_ln");
        assert!(format!("{err}").contains("enc_ln"));
    }
}
