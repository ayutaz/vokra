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
use vokra_core::gguf::GgufFile;
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
        let fail = |what: String| Err(VokraError::InvalidArgument(format!("utmos config: {what}")));
        if self.sample_rate == 0 {
            return fail("sample_rate must be > 0".into());
        }
        if self.conv_channels.is_empty() {
            return fail("conv stack must have at least one layer (conv.channels empty)".into());
        }
        if self.conv_kernels.len() != self.conv_channels.len()
            || self.conv_strides.len() != self.conv_channels.len()
        {
            return fail(format!(
                "conv.channels/kernels/strides length mismatch ({} / {} / {})",
                self.conv_channels.len(),
                self.conv_kernels.len(),
                self.conv_strides.len()
            ));
        }
        for (i, ((&c, &k), &s)) in self
            .conv_channels
            .iter()
            .zip(&self.conv_kernels)
            .zip(&self.conv_strides)
            .enumerate()
        {
            if c == 0 {
                return fail(format!("conv.channels[{i}] must be > 0"));
            }
            if k == 0 {
                return fail(format!("conv.kernels[{i}] must be > 0"));
            }
            if s == 0 {
                return fail(format!("conv.strides[{i}] must be > 0 (zero stride)"));
            }
        }
        if self.n_layer == 0 {
            return fail("transformer.n_layer must be >= 1".into());
        }
        if self.n_head == 0 {
            return fail("transformer.n_head must be >= 1".into());
        }
        if self.hidden_dim == 0 || self.ffn_dim == 0 {
            return fail(format!(
                "transformer dims must be > 0 (hidden_dim={}, ffn_dim={})",
                self.hidden_dim, self.ffn_dim
            ));
        }
        if self.hidden_dim % self.n_head != 0 {
            return fail(format!(
                "hidden_dim {} is not divisible by n_head {}",
                self.hidden_dim, self.n_head
            ));
        }
        if !(self.ln_eps.is_finite() && self.ln_eps > 0.0) {
            return fail(format!(
                "transformer.ln_eps must be finite and > 0, got {}",
                self.ln_eps
            ));
        }
        if self.head_dims.is_empty() {
            return fail("head.dims must have at least one linear".into());
        }
        if let Some((i, _)) = self.head_dims.iter().enumerate().find(|&(_, &d)| d == 0) {
            return fail(format!("head.dims[{i}] must be > 0"));
        }
        if self.head_dims.last() != Some(&1) {
            return fail(format!(
                "head.dims must end in 1 (the MOS scalar), got {:?}",
                self.head_dims
            ));
        }
        if !self.head_scale.is_finite() || !self.head_offset.is_finite() {
            return fail(format!(
                "head affine must be finite (scale={}, offset={})",
                self.head_scale, self.head_offset
            ));
        }
        Ok(())
    }

    /// Reads the config from a `vokra.utmos.*` GGUF metadata block,
    /// rejecting an unknown [`KEY_ARCH_VARIANT`] loudly.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        use vokra_core::gguf::chunks::KEY_MODEL_ARCH;
        let arch = meta_str(file, KEY_MODEL_ARCH)?;
        if arch != ARCH {
            return Err(VokraError::ModelLoad(format!(
                "utmos GGUF: {KEY_MODEL_ARCH} is {arch:?}, expected {ARCH:?}"
            )));
        }
        let variant = meta_str(file, KEY_ARCH_VARIANT)?;
        if variant != ARCH_VARIANT_V0 {
            return Err(VokraError::ModelLoad(format!(
                "utmos GGUF: unknown arch variant {variant:?} — this build implements only \
                 {ARCH_VARIANT_V0:?} and refuses to mis-score a different stack (FR-EX-08; \
                 the flip-time upstream pin bumps the variant, see ADR M4-18-utmos-arch)"
            )));
        }
        let sample_rate = meta_u32(file, KEY_SAMPLE_RATE)?;
        let conv_channels = meta_usize_array(file, KEY_CONV_CHANNELS)?;
        let conv_kernels = meta_usize_array(file, KEY_CONV_KERNELS)?;
        let conv_strides = meta_usize_array(file, KEY_CONV_STRIDES)?;
        let conv_activation = match meta_str(file, KEY_CONV_ACTIVATION)? {
            "gelu" => ConvActivation::Gelu,
            other => {
                return Err(VokraError::ModelLoad(format!(
                    "utmos GGUF: unknown conv activation {other:?} (v0 implements \"gelu\" only)"
                )));
            }
        };
        let n_layer = meta_u32(file, KEY_TF_N_LAYER)? as usize;
        let n_head = meta_u32(file, KEY_TF_N_HEAD)? as usize;
        let hidden_dim = meta_u32(file, KEY_TF_HIDDEN_DIM)? as usize;
        let ffn_dim = meta_u32(file, KEY_TF_FFN_DIM)? as usize;
        let norm = match meta_str(file, KEY_TF_NORM)? {
            "pre" => TransformerNorm::Pre,
            "post" => TransformerNorm::Post,
            other => {
                return Err(VokraError::ModelLoad(format!(
                    "utmos GGUF: unknown transformer norm {other:?} (expected \"pre\" or \"post\")"
                )));
            }
        };
        let ln_eps = meta_f32(file, KEY_TF_LN_EPS)?;
        let head_dims = meta_usize_array(file, KEY_HEAD_DIMS)?;
        let head_pool = match meta_str(file, KEY_HEAD_POOL)? {
            "mean_before" => HeadPool::MeanBefore,
            "mean_after" => HeadPool::MeanAfter,
            other => {
                return Err(VokraError::ModelLoad(format!(
                    "utmos GGUF: unknown head pool {other:?} (expected \"mean_before\" or \
                     \"mean_after\")"
                )));
            }
        };
        let head_scale = meta_f32_or(file, KEY_HEAD_SCALE, 1.0)?;
        let head_offset = meta_f32_or(file, KEY_HEAD_OFFSET, 0.0)?;
        let config = Self {
            sample_rate,
            conv_channels,
            conv_kernels,
            conv_strides,
            conv_activation,
            n_layer,
            n_head,
            hidden_dim,
            ffn_dim,
            norm,
            ln_eps,
            head_dims,
            head_pool,
            head_scale,
            head_offset,
        };
        config.validate()?;
        Ok(config)
    }

    /// The frame count the conv feature encoder yields for `in_len` input
    /// samples (`out = (in - kernel) / stride + 1` per layer, "valid"
    /// padding). An input shorter than a layer's kernel is a loud error
    /// (FR-EX-08 — never an empty silent output).
    pub fn feature_len(&self, in_len: usize) -> Result<usize> {
        let mut len = in_len;
        for (i, (&k, &s)) in self.conv_kernels.iter().zip(&self.conv_strides).enumerate() {
            if len < k {
                return Err(VokraError::InvalidArgument(format!(
                    "utmos feature encoder: input too short at conv layer {i} — {len} sample(s) \
                     reach a kernel of width {k}, so not even one output frame exists (FR-EX-08: \
                     an empty feature sequence is announced, never silently produced)"
                )));
            }
            len = (len - k) / s + 1;
        }
        Ok(len)
    }
}

// --- GGUF metadata read helpers (loud on missing / mistyped keys) -----------

/// Reads a required STRING metadata value, loudly naming a missing or
/// mistyped key ([`VokraError::ModelLoad`]).
fn meta_str<'a>(file: &'a GgufFile, key: &str) -> Result<&'a str> {
    let v = file.get(key).ok_or_else(|| {
        VokraError::ModelLoad(format!("utmos GGUF: missing metadata key `{key}`"))
    })?;
    v.as_str().ok_or_else(|| {
        VokraError::ModelLoad(format!("utmos GGUF: metadata key `{key}` is not a STRING"))
    })
}

/// Reads a required unsigned-integer metadata value as `u32`.
fn meta_u32(file: &GgufFile, key: &str) -> Result<u32> {
    let v = file.get(key).ok_or_else(|| {
        VokraError::ModelLoad(format!("utmos GGUF: missing metadata key `{key}`"))
    })?;
    let wide = v.as_u64().ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "utmos GGUF: metadata key `{key}` is not an unsigned integer"
        ))
    })?;
    u32::try_from(wide).map_err(|_| {
        VokraError::ModelLoad(format!(
            "utmos GGUF: metadata key `{key}` = {wide} does not fit in u32"
        ))
    })
}

/// Reads a required FLOAT32 metadata value.
fn meta_f32(file: &GgufFile, key: &str) -> Result<f32> {
    let v = file.get(key).ok_or_else(|| {
        VokraError::ModelLoad(format!("utmos GGUF: missing metadata key `{key}`"))
    })?;
    let wide = v.as_f64().ok_or_else(|| {
        VokraError::ModelLoad(format!("utmos GGUF: metadata key `{key}` is not a float"))
    })?;
    Ok(wide as f32)
}

/// Reads an optional FLOAT32 metadata value: a missing key yields `default`
/// (the identity affine — the only safe default, see [`KEY_HEAD_SCALE`]), but
/// a present-and-mistyped key is still a loud error, never a silent default.
fn meta_f32_or(file: &GgufFile, key: &str, default: f32) -> Result<f32> {
    match file.get(key) {
        None => Ok(default),
        Some(v) => v.as_f64().map(|w| w as f32).ok_or_else(|| {
            VokraError::ModelLoad(format!("utmos GGUF: metadata key `{key}` is not a float"))
        }),
    }
}

/// Reads a required ARRAY<UINT32> metadata value as `Vec<usize>`.
fn meta_usize_array(file: &GgufFile, key: &str) -> Result<Vec<usize>> {
    let v = file.get(key).ok_or_else(|| {
        VokraError::ModelLoad(format!("utmos GGUF: missing metadata key `{key}`"))
    })?;
    let arr = v.as_array().ok_or_else(|| {
        VokraError::ModelLoad(format!("utmos GGUF: metadata key `{key}` is not an ARRAY"))
    })?;
    arr.values
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let wide = e.as_u64().ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "utmos GGUF: metadata key `{key}`[{i}] is not an unsigned integer"
                ))
            })?;
            usize::try_from(wide).map_err(|_| {
                VokraError::ModelLoad(format!(
                    "utmos GGUF: metadata key `{key}`[{i}] = {wide} does not fit in usize"
                ))
            })
        })
        .collect()
}

/// One conv layer of the feature encoder — weight `[c_out, c_in, k]`
/// row-major plus optional per-out-channel bias.
#[derive(Debug)]
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
#[derive(Debug)]
struct Linear {
    w_t: Vec<f32>,
    bias: Option<Vec<f32>>,
    d_in: usize,
    d_out: usize,
}

/// LayerNorm affine parameters.
#[derive(Debug)]
struct LayerNormW {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

/// One transformer encoder block (bidirectional MHSA + GELU MLP).
#[derive(Debug)]
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
#[derive(Debug)]
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
    /// Xavier/Glorot uniform; LayerNorm γ=1, β=0, biases 0 — the M3-09
    /// `LlmWeights::synthesized` recipe). **Not** a reproduction of any real
    /// checkpoint: it exists so shape / determinism / finiteness are
    /// verifiable without the deferred weights.
    pub fn synthesized(config: &UtmosConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        let mut rng = SplitMix64::new(seed);
        let mut conv = Vec::with_capacity(config.conv_channels.len());
        let mut c_in = 1usize;
        for ((&c_out, &kernel), &stride) in config
            .conv_channels
            .iter()
            .zip(&config.conv_kernels)
            .zip(&config.conv_strides)
        {
            // Conv fan counts follow the PyTorch convention: the receptive
            // field multiplies both fans (fan_in = c_in * k, fan_out = c_out
            // * k). A synthesized-recipe choice, not an upstream claim.
            let weight = xavier_uniform(
                &mut rng,
                c_out * c_in * kernel,
                c_in * kernel,
                c_out * kernel,
            );
            conv.push(ConvLayer {
                weight,
                bias: Some(vec![0.0; c_out]),
                c_in,
                c_out,
                kernel,
                stride,
            });
            c_in = c_out;
        }
        let c_last = c_in;
        let d = config.hidden_dim;
        let feat_proj = if c_last != d {
            Some(synth_linear(&mut rng, c_last, d))
        } else {
            None
        };
        let mut blocks = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            blocks.push(EncBlock {
                ln1: identity_ln(d),
                q: synth_linear(&mut rng, d, d),
                k: synth_linear(&mut rng, d, d),
                v: synth_linear(&mut rng, d, d),
                o: synth_linear(&mut rng, d, d),
                ln2: identity_ln(d),
                fc1: synth_linear(&mut rng, d, config.ffn_dim),
                fc2: synth_linear(&mut rng, config.ffn_dim, d),
            });
        }
        let enc_ln = match config.norm {
            TransformerNorm::Pre => Some(identity_ln(d)),
            TransformerNorm::Post => None,
        };
        let mut head = Vec::with_capacity(config.head_dims.len());
        let mut h_in = d;
        for &h_out in &config.head_dims {
            head.push(synth_linear(&mut rng, h_in, h_out));
            h_in = h_out;
        }
        Ok(Self {
            conv,
            feat_proj,
            blocks,
            enc_ln,
            head,
            is_synthesized: true,
        })
    }

    /// Binds the weight store from a parsed GGUF, verifying every tensor's
    /// dims against `config` (ADR M4-18-utmos-arch §(d) naming). A missing
    /// or mis-shaped tensor is a loud [`VokraError::ModelLoad`] naming it.
    pub fn from_gguf(file: &GgufFile, config: &UtmosConfig) -> Result<Self> {
        config.validate()?;
        let mut conv = Vec::with_capacity(config.conv_channels.len());
        let mut c_in = 1usize;
        for (i, ((&c_out, &kernel), &stride)) in config
            .conv_channels
            .iter()
            .zip(&config.conv_kernels)
            .zip(&config.conv_strides)
            .enumerate()
        {
            let weight = tensor_f32_shaped(
                file,
                &format!("utmos.conv.{i}.weight"),
                &[c_out, c_in, kernel],
            )?;
            let bias = opt_tensor_f32_shaped(file, &format!("utmos.conv.{i}.bias"), &[c_out])?;
            conv.push(ConvLayer {
                weight,
                bias,
                c_in,
                c_out,
                kernel,
                stride,
            });
            c_in = c_out;
        }
        let c_last = c_in;
        let d = config.hidden_dim;
        let feat_proj = if c_last != d {
            Some(load_linear(file, "utmos.feat_proj", c_last, d)?)
        } else if file.tensor_info("utmos.feat_proj.weight").is_some() {
            // c_last == d GGUFs may still ship an explicit square projection.
            Some(load_linear(file, "utmos.feat_proj", c_last, d)?)
        } else {
            None
        };
        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            blocks.push(EncBlock {
                ln1: load_ln(file, &format!("utmos.enc.{i}.ln1"), d)?,
                q: load_linear(file, &format!("utmos.enc.{i}.attn.q"), d, d)?,
                k: load_linear(file, &format!("utmos.enc.{i}.attn.k"), d, d)?,
                v: load_linear(file, &format!("utmos.enc.{i}.attn.v"), d, d)?,
                o: load_linear(file, &format!("utmos.enc.{i}.attn.o"), d, d)?,
                ln2: load_ln(file, &format!("utmos.enc.{i}.ln2"), d)?,
                fc1: load_linear(file, &format!("utmos.enc.{i}.mlp.fc1"), d, config.ffn_dim)?,
                fc2: load_linear(file, &format!("utmos.enc.{i}.mlp.fc2"), config.ffn_dim, d)?,
            });
        }
        let enc_ln = match config.norm {
            TransformerNorm::Pre => Some(load_ln(file, "utmos.enc_ln", d)?),
            TransformerNorm::Post => {
                // Post-norm defines no final LayerNorm; a shipped
                // `utmos.enc_ln.*` would be a tensor this variant has no
                // semantics for — reject it rather than inventing one
                // (FR-EX-08: never silently ignore weights either).
                if file.tensor_info("utmos.enc_ln.weight").is_some()
                    || file.tensor_info("utmos.enc_ln.bias").is_some()
                {
                    return Err(VokraError::ModelLoad(
                        "utmos GGUF: post-norm variant must not ship `utmos.enc_ln.*` (the \
                         variant defines no final LayerNorm; refusing to guess its placement)"
                            .to_owned(),
                    ));
                }
                None
            }
        };
        let mut head = Vec::with_capacity(config.head_dims.len());
        let mut h_in = d;
        for (i, &h_out) in config.head_dims.iter().enumerate() {
            head.push(load_linear(file, &format!("utmos.head.{i}"), h_in, h_out)?);
            h_in = h_out;
        }
        Ok(Self {
            conv,
            feat_proj,
            blocks,
            enc_ln,
            head,
            is_synthesized: false,
        })
    }
}

// --- weight construction helpers --------------------------------------------

/// Xavier / Glorot uniform draw: `n` values in `(-bound, +bound)` with
/// `bound = sqrt(6 / (fan_in + fan_out))` — verbatim the M3-09
/// `LlmWeights::synthesized` recipe (`cosyvoice2/llm.rs`).
fn xavier_uniform(rng: &mut SplitMix64, n: usize, fan_in: usize, fan_out: usize) -> Vec<f32> {
    let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
    (0..n)
        .map(|_| (rng.next_unit_f32() * 2.0 - 1.0) * bound)
        .collect()
}

/// A synthesized linear: Xavier weight (drawn directly in the transposed
/// `[d_in, d_out]` storage order), zero bias.
fn synth_linear(rng: &mut SplitMix64, d_in: usize, d_out: usize) -> Linear {
    Linear {
        w_t: xavier_uniform(rng, d_in * d_out, d_in, d_out),
        bias: Some(vec![0.0; d_out]),
        d_in,
        d_out,
    }
}

/// The identity LayerNorm affine (γ=1, β=0 — the PyTorch default init).
fn identity_ln(d: usize) -> LayerNormW {
    LayerNormW {
        gamma: vec![1.0; d],
        beta: vec![0.0; d],
    }
}

// --- GGUF tensor read helpers (loud on missing / mis-shaped tensors) --------

/// Reads a required F32 tensor with exactly `dims`, loudly naming a missing
/// or mis-shaped tensor.
fn tensor_f32_shaped(file: &GgufFile, name: &str, dims: &[usize]) -> Result<Vec<f32>> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("utmos GGUF: missing tensor `{name}`")))?;
    let expected: Vec<u64> = dims.iter().map(|&d| d as u64).collect();
    if info.dimensions != expected {
        return Err(VokraError::ModelLoad(format!(
            "utmos GGUF: tensor `{name}` has dims {:?}, expected {expected:?}",
            info.dimensions
        )));
    }
    file.tensor_f32(name)
        .map_err(|e| VokraError::ModelLoad(format!("utmos GGUF: reading tensor `{name}`: {e}")))
}

/// Reads an optional F32 tensor: absent yields `None`, present-but-mis-shaped
/// is still a loud error (never a silent zero-fill).
fn opt_tensor_f32_shaped(file: &GgufFile, name: &str, dims: &[usize]) -> Result<Option<Vec<f32>>> {
    if file.tensor_info(name).is_none() {
        return Ok(None);
    }
    tensor_f32_shaped(file, name, dims).map(Some)
}

/// Loads `{prefix}.weight` (`[d_out, d_in]`, the `y = W x` semantic of ADR
/// M4-18-utmos-arch §(c)) transposed into the `[d_in, d_out]` GEMM storage
/// order, plus the optional `{prefix}.bias`.
fn load_linear(file: &GgufFile, prefix: &str, d_in: usize, d_out: usize) -> Result<Linear> {
    let w = tensor_f32_shaped(file, &format!("{prefix}.weight"), &[d_out, d_in])?;
    let mut w_t = vec![0.0f32; d_in * d_out];
    for (o, row) in w.chunks_exact(d_in).enumerate() {
        for (i, &val) in row.iter().enumerate() {
            w_t[i * d_out + o] = val;
        }
    }
    let bias = opt_tensor_f32_shaped(file, &format!("{prefix}.bias"), &[d_out])?;
    Ok(Linear {
        w_t,
        bias,
        d_in,
        d_out,
    })
}

/// Loads a LayerNorm affine pair `{prefix}.weight` / `{prefix}.bias` (both
/// required, length `d`).
fn load_ln(file: &GgufFile, prefix: &str, d: usize) -> Result<LayerNormW> {
    Ok(LayerNormW {
        gamma: tensor_f32_shaped(file, &format!("{prefix}.weight"), &[d])?,
        beta: tensor_f32_shaped(file, &format!("{prefix}.bias"), &[d])?,
    })
}

/// The UTMOS scorer: one waveform in, one MOS scalar out.
///
/// v0 skeleton semantics — see the module docs: config-driven forward,
/// synthesized or GGUF weights, **no upstream numerical claim until the
/// flip-time pin**.
#[derive(Debug)]
pub struct Utmos {
    config: UtmosConfig,
    weights: UtmosWeights,
}

impl Utmos {
    /// Builds a scorer over synthesized weights (see
    /// [`UtmosWeights::synthesized`]).
    pub fn synthesized(config: UtmosConfig, seed: u64) -> Result<Self> {
        let weights = UtmosWeights::synthesized(&config, seed)?;
        Ok(Self { config, weights })
    }

    /// Binds a scorer from a parsed `vokra.utmos.*` GGUF.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let config = UtmosConfig::from_gguf(file)?;
        let weights = UtmosWeights::from_gguf(file, &config)?;
        Ok(Self { config, weights })
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
    pub fn score(&self, audio: &[f32], sample_rate: u32) -> Result<f64> {
        let c = &self.config;
        if sample_rate != c.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "utmos: input sample rate {sample_rate} != model rate {} — the metric never \
                 silently resamples (FR-EX-08); resample the clip explicitly first",
                c.sample_rate
            )));
        }
        if audio.is_empty() {
            return Err(VokraError::InvalidArgument(
                "utmos: empty input clip".to_owned(),
            ));
        }
        if let Some(pos) = audio.iter().position(|x| !x.is_finite()) {
            return Err(VokraError::InvalidArgument(format!(
                "utmos: non-finite sample at index {pos} — a NaN/Inf would silently poison the \
                 score (FR-EX-08)"
            )));
        }
        // Validates the length against the conv receptive field (loud
        // "too short" error) and pins the frame count the loop must yield.
        let t_frames = c.feature_len(audio.len())?;

        // ---- feature encoder: conv stack + GELU, channel-major [c, len] ----
        let mut cur = audio.to_vec();
        let mut len = audio.len();
        for layer in &self.weights.conv {
            let out_len = (len - layer.kernel) / layer.stride + 1;
            let mut out = vec![0.0f32; layer.c_out * out_len];
            kernels::conv1d_f32(
                &cur,
                layer.c_in,
                len,
                &layer.weight,
                layer.c_out,
                layer.kernel,
                layer.bias.as_deref(),
                layer.stride,
                0,
                &mut out,
            )?;
            let mut act = vec![0.0f32; out.len()];
            match self.config.conv_activation {
                ConvActivation::Gelu => kernels::gelu_f32(&out, &mut act)?,
            }
            cur = act;
            len = out_len;
        }
        debug_assert_eq!(len, t_frames, "feature_len oracle vs conv loop");
        let t = t_frames;

        // ---- [c_last, t] → frame-major [t, c_last] --------------------------
        let c_last = self.weights.conv.last().map_or(1, |l| l.c_out);
        let mut x = vec![0.0f32; t * c_last];
        for (ch, channel) in cur.chunks_exact(t).enumerate() {
            for (tt, &val) in channel.iter().enumerate() {
                x[tt * c_last + ch] = val;
            }
        }

        // ---- optional feature projection to the transformer width ----------
        let mut h = match &self.weights.feat_proj {
            Some(proj) => linear_forward(proj, &x, t)?,
            None => x,
        };

        // ---- bidirectional transformer encoder ------------------------------
        for blk in &self.weights.blocks {
            h = self.encoder_block(blk, h, t)?;
        }
        if let Some(ln) = &self.weights.enc_ln {
            h = layer_norm(ln, &h, t, c.hidden_dim, c.ln_eps)?;
        }

        // ---- regression head + pooling --------------------------------------
        let raw = match c.head_pool {
            HeadPool::MeanBefore => {
                let mut cur = mean_over_time(&h, t, c.hidden_dim);
                for lin in &self.weights.head {
                    cur = linear_forward(lin, &cur, 1)?;
                }
                cur[0]
            }
            HeadPool::MeanAfter => {
                let mut cur = h;
                for lin in &self.weights.head {
                    cur = linear_forward(lin, &cur, t)?;
                }
                mean_over_time(&cur, t, 1)[0]
            }
        };

        // The affine is applied in f64 so `score = raw * scale + offset`
        // is exact for the identity (`1.0` / `0.0`) case.
        Ok(f64::from(raw) * f64::from(c.head_scale) + f64::from(c.head_offset))
    }

    /// One encoder block over frame-major `h = [t, d]`, honoring the
    /// config's norm placement.
    fn encoder_block(&self, blk: &EncBlock, h: Vec<f32>, t: usize) -> Result<Vec<f32>> {
        let d = self.config.hidden_dim;
        let eps = self.config.ln_eps;
        match self.config.norm {
            TransformerNorm::Pre => {
                // x + Attn(LN1(x)); x + MLP(LN2(x)).
                let n1 = layer_norm(&blk.ln1, &h, t, d, eps)?;
                let attn = self.mhsa(blk, &n1, t)?;
                let h1 = add(&h, &attn)?;
                let n2 = layer_norm(&blk.ln2, &h1, t, d, eps)?;
                let mlp = mlp_forward(blk, &n2, t)?;
                add(&h1, &mlp)
            }
            TransformerNorm::Post => {
                // LN1(x + Attn(x)); LN2(x + MLP(x)).
                let attn = self.mhsa(blk, &h, t)?;
                let h1 = layer_norm(&blk.ln1, &add(&h, &attn)?, t, d, eps)?;
                let mlp = mlp_forward(blk, &h1, t)?;
                layer_norm(&blk.ln2, &add(&h1, &mlp)?, t, d, eps)
            }
        }
    }

    /// Bidirectional (unmasked) multi-head self-attention over `[t, d]`,
    /// scale `1/sqrt(d_head)`, per-head GEMM + row softmax.
    fn mhsa(&self, blk: &EncBlock, x: &[f32], t: usize) -> Result<Vec<f32>> {
        let d = self.config.hidden_dim;
        let n_head = self.config.n_head;
        let dh = d / n_head;
        let scale = 1.0 / (dh as f32).sqrt();
        let q = linear_forward(&blk.q, x, t)?;
        let k = linear_forward(&blk.k, x, t)?;
        let v = linear_forward(&blk.v, x, t)?;
        let mut ctx = vec![0.0f32; t * d];
        let mut qh = vec![0.0f32; t * dh];
        let mut kh_t = vec![0.0f32; dh * t];
        let mut vh = vec![0.0f32; t * dh];
        let mut scores = vec![0.0f32; t * t];
        let mut probs = vec![0.0f32; t * t];
        let mut out_h = vec![0.0f32; t * dh];
        for head in 0..n_head {
            let off = head * dh;
            for tt in 0..t {
                for j in 0..dh {
                    qh[tt * dh + j] = q[tt * d + off + j];
                    // K is materialized pre-transposed ([dh, t]) so the
                    // score GEMM is a plain row-major product.
                    kh_t[j * t + tt] = k[tt * d + off + j];
                    vh[tt * dh + j] = v[tt * d + off + j];
                }
            }
            kernels::gemm_f32(t, t, dh, &qh, &kh_t, None, &mut scores)?;
            for s in scores.iter_mut() {
                *s *= scale;
            }
            kernels::softmax_f32(&scores, &mut probs, t, t)?;
            kernels::gemm_f32(t, dh, t, &probs, &vh, None, &mut out_h)?;
            for tt in 0..t {
                ctx[tt * d + off..tt * d + off + dh]
                    .copy_from_slice(&out_h[tt * dh..(tt + 1) * dh]);
            }
        }
        linear_forward(&blk.o, &ctx, t)
    }
}

// --- forward primitives (thin wrappers over vokra-backend-cpu kernels) ------

/// `Y[rows, d_out] = X[rows, d_in] @ w_t (+ bias)` — one row-major GEMM.
fn linear_forward(lin: &Linear, x: &[f32], rows: usize) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; rows * lin.d_out];
    kernels::gemm_f32(
        rows,
        lin.d_out,
        lin.d_in,
        x,
        &lin.w_t,
        lin.bias.as_deref(),
        &mut out,
    )?;
    Ok(out)
}

/// Row-wise LayerNorm over `[rows, cols]` with the block's affine.
fn layer_norm(ln: &LayerNormW, x: &[f32], rows: usize, cols: usize, eps: f32) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; x.len()];
    kernels::layer_norm_f32(x, &mut out, rows, cols, &ln.gamma, &ln.beta, eps)?;
    Ok(out)
}

/// The GELU MLP: `fc2(gelu(fc1(x)))`.
fn mlp_forward(blk: &EncBlock, x: &[f32], t: usize) -> Result<Vec<f32>> {
    let inner = linear_forward(&blk.fc1, x, t)?;
    let mut act = vec![0.0f32; inner.len()];
    kernels::gelu_f32(&inner, &mut act)?;
    linear_forward(&blk.fc2, &act, t)
}

/// Element-wise residual add.
fn add(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; a.len()];
    kernels::add_f32(a, b, &mut out)?;
    Ok(out)
}

/// Mean over the time axis of a frame-major `[t, cols]` buffer → `[1, cols]`.
fn mean_over_time(x: &[f32], t: usize, cols: usize) -> Vec<f32> {
    let mut pooled = vec![0.0f32; cols];
    for frame in x.chunks_exact(cols) {
        for (acc, &val) in pooled.iter_mut().zip(frame) {
            *acc += val;
        }
    }
    let inv = 1.0 / t as f32;
    for acc in pooled.iter_mut() {
        *acc *= inv;
    }
    pooled
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
    use vokra_core::gguf::{GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType};

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
        assert_eq!(w.conv[0].weight.len(), 4 * 5); // c_out=4, c_in=1, k=5
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
        assert_eq!(w.head[1].w_t.len(), 4); // d_in=4, d_out=1

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
        add_f32_tensor(&mut b, "utmos.conv.0.weight", &[4, 1, 4], &[0.0; 16]);
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
        add_f32_tensor(&mut b, "utmos.enc_ln.weight", &[6], &[1.0; 6]);
        add_f32_tensor(&mut b, "utmos.enc_ln.bias", &[6], &[0.0; 6]);
        bytes = b.to_bytes().unwrap();
        let file = GgufFile::parse(bytes).expect("parse");
        let err = Utmos::from_gguf(&file).expect_err("post-norm + enc_ln");
        assert!(format!("{err}").contains("enc_ln"));
    }
}
