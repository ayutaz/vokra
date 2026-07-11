//! Voxtral audio adapter (M3-10 Wave 8) — encoder-hidden → text-decoder soft prefix.
//!
//! # Purpose
//!
//! Upstream Voxtral projects the audio encoder's hidden state through an
//! **audio adapter** (a small linear / MLP / down-sample projection) and
//! consumes the projected sequence as a soft-prefix embedding at the start
//! of the Mistral text decoder — i.e. real ASR conditioning.
//!
//! Wave 7 landed the greedy decode without adapter (LM continuation from
//! `bos_id`). This module adds the adapter *framework* so a real Voxtral
//! checkpoint that ships adapter weights can be conditioned properly, while
//! preserving the honest LM-continuation posture for GGUFs whose adapter is
//! absent (`AdapterKind::None`).
//!
//! # Pluggable via GGUF metadata (FR-LD-02 / FR-MD-02, no invented literals)
//!
//! Neither tensor names nor shape numbers are hard-coded in this file. The
//! adapter kind, shape, activation, tensor prefix and tensor sub-names are all
//! read from the GGUF's `vokra.voxtral.adapter.*` metadata chunk which the
//! offline converter (`convert_voxtral_file` — see `vokra-convert`) writes
//! from a caller-supplied side-car JSON. This means the runtime never invents
//! upstream tensor names — a checkpoint that spells its adapter tensors as
//! `audio_adapter.linear.weight` and another that uses
//! `mm_projector.proj.weight` are both handled through the same code path.
//!
//! # Honest limitation posture (FR-EX-08)
//!
//! `AdapterKind::None` is a first-class variant: it is the value the converter
//! writes when the caller has no adapter to embed. Runtime code that sees
//! `None` **must** continue the Wave 7 LM continuation — never fabricate audio
//! conditioning. Real conditioning only happens for `Linear` / `Mlp` /
//! `DownsampleLinear`.
//!
//! # Zero-dep contract (NFR-DS-02)
//!
//! Every projection dispatches through the shared [`crate::compute::Compute`]
//! seam (backed by `vokra-backend-cpu` today, Metal / CUDA when the caller
//! selects those backends). No new crate dependency is introduced.

use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use crate::compute::Compute;

// ---------- GGUF metadata keys ---------------------------------------------

/// `vokra.voxtral.adapter.kind` — string: one of `"none"` (no conditioning),
/// `"linear"` (single Linear + optional bias + optional LayerNorm),
/// `"mlp"` (multi-Linear + activation stack), or `"downsample_linear"`
/// (time-axis avg-pool then Linear).
const KEY_ADAPTER_KIND: &str = "vokra.voxtral.adapter.kind";

/// `vokra.voxtral.adapter.in_dim` — u32.
const KEY_ADAPTER_IN_DIM: &str = "vokra.voxtral.adapter.in_dim";

/// `vokra.voxtral.adapter.out_dim` — u32.
const KEY_ADAPTER_OUT_DIM: &str = "vokra.voxtral.adapter.out_dim";

/// `vokra.voxtral.adapter.has_bias` — bool, defaults to false.
const KEY_ADAPTER_HAS_BIAS: &str = "vokra.voxtral.adapter.has_bias";

/// `vokra.voxtral.adapter.has_layernorm` — bool, defaults to false.
const KEY_ADAPTER_HAS_LN: &str = "vokra.voxtral.adapter.has_layernorm";

/// `vokra.voxtral.adapter.activation` — string, one of `"gelu"`, `"silu"`,
/// `"relu"`, `"identity"`. Only meaningful for MLP layers.
const KEY_ADAPTER_ACTIVATION: &str = "vokra.voxtral.adapter.activation";

/// `vokra.voxtral.adapter.time_stride` — u32, only for `downsample_linear`.
const KEY_ADAPTER_TIME_STRIDE: &str = "vokra.voxtral.adapter.time_stride";

/// `vokra.voxtral.adapter.tensor_prefix` — string, e.g. `"audio_adapter."` or
/// `"mm_projector."`. All adapter tensor names below use this prefix.
const KEY_ADAPTER_TENSOR_PREFIX: &str = "vokra.voxtral.adapter.tensor_prefix";

/// `vokra.voxtral.adapter.weight_name` — sub-name of the (first) linear
/// weight tensor. Combined with `tensor_prefix` produces the full GGUF tensor
/// name. Optional — defaults to `"weight"`.
const KEY_ADAPTER_WEIGHT_NAME: &str = "vokra.voxtral.adapter.weight_name";

/// `vokra.voxtral.adapter.bias_name` — sub-name of the (first) linear bias
/// tensor. Optional — defaults to `"bias"` when `has_bias` is true.
const KEY_ADAPTER_BIAS_NAME: &str = "vokra.voxtral.adapter.bias_name";

/// `vokra.voxtral.adapter.layernorm_gamma_name` — sub-name of the LayerNorm
/// scale tensor. Optional — defaults to `"layernorm.weight"` when has_layernorm.
const KEY_ADAPTER_LN_GAMMA_NAME: &str = "vokra.voxtral.adapter.layernorm_gamma_name";

/// `vokra.voxtral.adapter.layernorm_beta_name` — sub-name of the LayerNorm
/// bias tensor. Optional — defaults to `"layernorm.bias"` when has_layernorm.
const KEY_ADAPTER_LN_BETA_NAME: &str = "vokra.voxtral.adapter.layernorm_beta_name";

/// `vokra.voxtral.adapter.mlp_hidden_dims` — comma-separated u32 list (as a
/// GGUF STRING) of intermediate hidden dims when kind = `"mlp"`. Empty string
/// / absent = single linear (equivalent to `"linear"`).
const KEY_ADAPTER_MLP_HIDDEN_DIMS: &str = "vokra.voxtral.adapter.mlp_hidden_dims";

/// `vokra.voxtral.adapter.mlp_layer_names` — comma-separated string of layer
/// tensor sub-names (e.g. `"layers.0,layers.1,layers.2"`). When kind = `"mlp"`
/// each layer's weight is at `{prefix}{layer_name}.{weight_name}` (same for
/// bias / LN — using the shared sub-names above). Empty = single-layer.
const KEY_ADAPTER_MLP_LAYER_NAMES: &str = "vokra.voxtral.adapter.mlp_layer_names";

// ---------- Types -----------------------------------------------------------

/// Activation applied between adapter MLP layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterActivation {
    /// Exact (erf) GELU — Whisper conv stem shape, matches Voxtral audio_tower
    /// activations.
    Gelu,
    /// SiLU / Swish — matches Mistral / Voxtral text decoder SwiGLU inner
    /// activation.
    Silu,
    /// ReLU — some older adapter variants.
    Relu,
    /// No activation (linear stack).
    Identity,
}

impl AdapterActivation {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "gelu" => Some(Self::Gelu),
            "silu" | "swish" => Some(Self::Silu),
            "relu" => Some(Self::Relu),
            "identity" | "none" | "" => Some(Self::Identity),
            _ => None,
        }
    }
}

/// Which projection shape the adapter has.
#[derive(Debug, Clone)]
pub enum AdapterKind {
    /// No adapter — the caller should skip the projection and stay on the
    /// LM-continuation path (honest Wave 7 posture).
    None,
    /// A single `Linear(in_dim → out_dim)` projection with optional bias +
    /// optional post-LayerNorm.
    Linear {
        /// Input channels (matches audio encoder `hidden_dim`).
        in_dim: usize,
        /// Output channels (matches text decoder `hidden_dim`).
        out_dim: usize,
        /// Whether the weight ships a bias tensor.
        has_bias: bool,
        /// Whether the weight ships a post-projection LayerNorm.
        has_layernorm: bool,
    },
    /// A stack of `Linear + activation` layers.
    ///
    /// `layers` describes each stage's `(in_dim, out_dim)`; the activation
    /// applies **between** layers (not after the last one). `bias` / LN are
    /// per-stage flags matching the [`Linear`](Self::Linear) semantics.
    Mlp {
        /// Per-stage `[in, out]` widths (`layers.len() >= 1`).
        layers: Vec<MlpLayerShape>,
        /// Activation between stages.
        activation: AdapterActivation,
    },
    /// Time-axis avg-pool by `time_stride`, then a `Linear(in_dim → out_dim)`.
    /// Matches the `AudioMultiModalProjector` shape some Voxtral / Voxtral-derived
    /// releases use to shrink 1500-frame encoder output to ~300 tokens before
    /// the text decoder consumes them.
    DownsampleLinear {
        /// Downsample factor along the time axis.
        time_stride: usize,
        /// Input channels.
        in_dim: usize,
        /// Output channels.
        out_dim: usize,
        /// Whether the projection ships a bias tensor.
        has_bias: bool,
        /// Whether the projection ships a post-LayerNorm.
        has_layernorm: bool,
    },
}

/// One `Linear` stage inside an [`AdapterKind::Mlp`].
#[derive(Debug, Clone)]
pub struct MlpLayerShape {
    /// Input channels of this stage.
    pub in_dim: usize,
    /// Output channels of this stage.
    pub out_dim: usize,
    /// Whether this stage ships a bias tensor.
    pub has_bias: bool,
    /// Whether this stage ships a post-projection LayerNorm.
    pub has_layernorm: bool,
}

/// A loaded `Linear` (weight in row-major `[in, out]` shape for direct GEMM,
/// optional bias `[out]`, optional LayerNorm gamma / beta `[out]`).
#[derive(Debug, Clone)]
pub(crate) struct AdapterLinear {
    /// `[in_features, out_features]` row-major.
    pub(crate) w_t: Vec<f32>,
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
    /// `[out_features]` — bias, `None` if the stage has no bias.
    pub(crate) bias: Option<Vec<f32>>,
    /// LayerNorm `gamma` / `beta` (`[out_features]` each) applied after the
    /// linear (before the activation, matches HF's `LayerNorm(x + linear(x))`
    /// order in existing adapters). `None` if `has_layernorm` is false.
    pub(crate) ln_gamma: Option<Vec<f32>>,
    pub(crate) ln_beta: Option<Vec<f32>>,
}

/// A parsed audio adapter, ready to project encoder hidden states into a
/// text-decoder soft-prefix.
pub struct AudioAdapter {
    kind: AdapterKind,
    activation: AdapterActivation,
    stages: Vec<AdapterLinear>,
    time_stride: usize,
    ln_eps: f32,
}

impl AudioAdapter {
    /// A stub `AdapterKind::None` adapter — [`apply`](Self::apply) is a no-op
    /// pass-through (returns the input unchanged). This is what the loader
    /// installs for a GGUF that declares `vokra.voxtral.adapter.kind = "none"`
    /// (or omits the chunk entirely). Callers that see `None` should stay on
    /// the LM-continuation path (Wave 7 honest posture).
    #[must_use]
    pub fn none() -> Self {
        Self {
            kind: AdapterKind::None,
            activation: AdapterActivation::Identity,
            stages: Vec::new(),
            time_stride: 1,
            ln_eps: 1e-5,
        }
    }

    /// The parsed adapter shape.
    #[must_use]
    pub fn kind(&self) -> &AdapterKind {
        &self.kind
    }

    /// Whether this adapter actually projects (i.e. is anything other than
    /// [`AdapterKind::None`]). Callers use this to branch between the audio-
    /// conditioned soft-prefix path and the honest LM continuation.
    #[must_use]
    pub fn is_active(&self) -> bool {
        !matches!(self.kind, AdapterKind::None)
    }

    /// Reads the adapter chunk from a GGUF and binds every referenced tensor.
    ///
    /// A GGUF with no `vokra.voxtral.adapter.*` metadata is a valid case —
    /// this returns [`Self::none()`]. A malformed chunk (e.g. `kind = "linear"`
    /// but the weight tensor is missing) is an explicit
    /// [`VokraError::ModelLoad`] naming the offending key or tensor.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // If no `kind` key is present, treat the model as adapter-less. Same
        // semantics as `kind = "none"`. This preserves backwards compatibility
        // with the Wave 7 GGUFs the converter shipped before this WP.
        let Some(kind_str) = file
            .get(KEY_ADAPTER_KIND)
            .and_then(|v| v.as_str())
            .map(str::to_owned)
        else {
            return Ok(Self::none());
        };

        // Empty / "none" — explicit absent adapter.
        if kind_str.is_empty() || kind_str == "none" {
            return Ok(Self::none());
        }

        let prefix = file
            .get(KEY_ADAPTER_TENSOR_PREFIX)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned();

        let weight_sub = file
            .get(KEY_ADAPTER_WEIGHT_NAME)
            .and_then(|v| v.as_str())
            .unwrap_or("weight")
            .to_owned();
        let bias_sub = file
            .get(KEY_ADAPTER_BIAS_NAME)
            .and_then(|v| v.as_str())
            .unwrap_or("bias")
            .to_owned();
        let ln_gamma_sub = file
            .get(KEY_ADAPTER_LN_GAMMA_NAME)
            .and_then(|v| v.as_str())
            .unwrap_or("layernorm.weight")
            .to_owned();
        let ln_beta_sub = file
            .get(KEY_ADAPTER_LN_BETA_NAME)
            .and_then(|v| v.as_str())
            .unwrap_or("layernorm.bias")
            .to_owned();

        let has_bias = file
            .get(KEY_ADAPTER_HAS_BIAS)
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let has_ln = file
            .get(KEY_ADAPTER_HAS_LN)
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let activation_str = file
            .get(KEY_ADAPTER_ACTIVATION)
            .and_then(|v| v.as_str())
            .unwrap_or("identity");
        let activation = AdapterActivation::parse(activation_str).ok_or_else(|| {
            bad(format!(
                "unknown activation `{activation_str}` (expected gelu|silu|relu|identity)"
            ))
        })?;

        match kind_str.as_str() {
            "linear" => {
                let (in_dim, out_dim) = read_dims(file)?;
                let stage = load_linear(
                    file,
                    &prefix,
                    "",
                    &weight_sub,
                    &bias_sub,
                    &ln_gamma_sub,
                    &ln_beta_sub,
                    in_dim,
                    out_dim,
                    has_bias,
                    has_ln,
                )?;
                Ok(Self {
                    kind: AdapterKind::Linear {
                        in_dim,
                        out_dim,
                        has_bias,
                        has_layernorm: has_ln,
                    },
                    activation: AdapterActivation::Identity,
                    stages: vec![stage],
                    time_stride: 1,
                    ln_eps: 1e-5,
                })
            }
            "mlp" => {
                let (in_dim, out_dim) = read_dims(file)?;
                let hidden_dims = parse_dim_list(
                    file.get(KEY_ADAPTER_MLP_HIDDEN_DIMS)
                        .and_then(|v| v.as_str())
                        .unwrap_or(""),
                )?;
                let layer_names = parse_str_list(
                    file.get(KEY_ADAPTER_MLP_LAYER_NAMES)
                        .and_then(|v| v.as_str())
                        .unwrap_or(""),
                );
                // Compute per-stage in/out widths.
                let mut widths: Vec<(usize, usize)> = Vec::new();
                let mut prev = in_dim;
                for &h in &hidden_dims {
                    widths.push((prev, h));
                    prev = h;
                }
                widths.push((prev, out_dim));
                if !layer_names.is_empty() && layer_names.len() != widths.len() {
                    return Err(bad(format!(
                        "mlp_layer_names has {} entries but computed layer count is {}",
                        layer_names.len(),
                        widths.len()
                    )));
                }
                let mut layers: Vec<MlpLayerShape> = Vec::with_capacity(widths.len());
                let mut stages: Vec<AdapterLinear> = Vec::with_capacity(widths.len());
                for (i, &(li, lo)) in widths.iter().enumerate() {
                    let sub = if layer_names.is_empty() {
                        format!("layers.{i}")
                    } else {
                        layer_names[i].clone()
                    };
                    let stage = load_linear(
                        file,
                        &prefix,
                        &sub,
                        &weight_sub,
                        &bias_sub,
                        &ln_gamma_sub,
                        &ln_beta_sub,
                        li,
                        lo,
                        has_bias,
                        has_ln,
                    )?;
                    stages.push(stage);
                    layers.push(MlpLayerShape {
                        in_dim: li,
                        out_dim: lo,
                        has_bias,
                        has_layernorm: has_ln,
                    });
                }
                Ok(Self {
                    kind: AdapterKind::Mlp { layers, activation },
                    activation,
                    stages,
                    time_stride: 1,
                    ln_eps: 1e-5,
                })
            }
            "downsample_linear" => {
                let (in_dim, out_dim) = read_dims(file)?;
                let time_stride = file
                    .get(KEY_ADAPTER_TIME_STRIDE)
                    .and_then(|v| v.as_u64())
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| {
                        bad(
                            "downsample_linear requires vokra.voxtral.adapter.time_stride (u32 >= 1)".to_owned(),
                        )
                    })?;
                if time_stride == 0 {
                    return Err(bad("time_stride must be >= 1".to_owned()));
                }
                let stage = load_linear(
                    file,
                    &prefix,
                    "",
                    &weight_sub,
                    &bias_sub,
                    &ln_gamma_sub,
                    &ln_beta_sub,
                    in_dim,
                    out_dim,
                    has_bias,
                    has_ln,
                )?;
                Ok(Self {
                    kind: AdapterKind::DownsampleLinear {
                        time_stride,
                        in_dim,
                        out_dim,
                        has_bias,
                        has_layernorm: has_ln,
                    },
                    activation: AdapterActivation::Identity,
                    stages: vec![stage],
                    time_stride,
                    ln_eps: 1e-5,
                })
            }
            other => Err(bad(format!(
                "unknown adapter kind `{other}` (expected none|linear|mlp|downsample_linear)"
            ))),
        }
    }

    /// Projects an encoder hidden-state sequence `[t_in, hidden_in]` into a
    /// text-decoder soft prefix `[t_out, hidden_out]`.
    ///
    /// * `AdapterKind::None` — returns `Ok(input.to_vec())` (identity). The
    ///   caller must **not** use this as audio conditioning; it exists so the
    ///   caller can uniformly write `let prefix = adapter.apply(...)` and then
    ///   decide *whether* to condition based on [`Self::is_active`].
    /// * `AdapterKind::Linear` — `t_out = t_in`, one GEMM (+bias +LN).
    /// * `AdapterKind::Mlp` — `t_out = t_in`, `layers` GEMMs with the
    ///   configured activation between stages.
    /// * `AdapterKind::DownsampleLinear` — `t_out = t_in / time_stride` (avg
    ///   pool over `time_stride` consecutive rows, then a GEMM).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on any shape mismatch (`input.len()`
    ///   vs `t_in * hidden_in`).
    pub fn apply(
        &self,
        compute: &Compute,
        input: &[f32],
        t_in: usize,
        hidden_in: usize,
    ) -> Result<Vec<f32>> {
        if input.len() != t_in * hidden_in {
            return Err(VokraError::InvalidArgument(format!(
                "voxtral audio adapter: input len {} != t_in*hidden_in ({}*{}={})",
                input.len(),
                t_in,
                hidden_in,
                t_in * hidden_in
            )));
        }
        match &self.kind {
            AdapterKind::None => Ok(input.to_vec()),
            AdapterKind::Linear {
                in_dim, out_dim, ..
            } => {
                self.check_hidden("Linear.in", *in_dim, hidden_in)?;
                let out = self.apply_stage(compute, &self.stages[0], input, t_in, *in_dim)?;
                debug_assert_eq!(out.len(), t_in * *out_dim);
                Ok(out)
            }
            AdapterKind::Mlp { layers, .. } => {
                let first = layers.first().ok_or_else(|| {
                    VokraError::InvalidArgument("voxtral audio adapter: Mlp has zero layers".into())
                })?;
                self.check_hidden("Mlp.first.in", first.in_dim, hidden_in)?;
                let mut cur = input.to_vec();
                let mut cur_h = hidden_in;
                let cur_t = t_in;
                for (i, (stage_meta, stage_weights)) in
                    layers.iter().zip(self.stages.iter()).enumerate()
                {
                    self.check_hidden(&format!("Mlp.layer[{i}].in"), stage_meta.in_dim, cur_h)?;
                    let mut out = self.apply_stage(compute, stage_weights, &cur, cur_t, cur_h)?;
                    // Activation between stages (skip after final).
                    let last = i + 1 == layers.len();
                    if !last {
                        self.activate(compute, &mut out)?;
                    }
                    cur = out;
                    cur_h = stage_meta.out_dim;
                }
                Ok(cur)
            }
            AdapterKind::DownsampleLinear {
                time_stride,
                in_dim,
                out_dim,
                ..
            } => {
                self.check_hidden("DownsampleLinear.in", *in_dim, hidden_in)?;
                let stride = *time_stride;
                if stride == 0 {
                    return Err(VokraError::InvalidArgument(
                        "voxtral audio adapter: time_stride must be >= 1".into(),
                    ));
                }
                // Floor division — drop any tail rows that don't complete a
                // window (matches PyTorch avg_pool1d default `ceil_mode=False`).
                let t_out = t_in / stride;
                let mut pooled = vec![0.0f32; t_out * hidden_in];
                let scale = 1.0f32 / stride as f32;
                for r in 0..t_out {
                    let src_start = r * stride;
                    let dst = &mut pooled[r * hidden_in..(r + 1) * hidden_in];
                    for s in 0..stride {
                        let src =
                            &input[(src_start + s) * hidden_in..(src_start + s + 1) * hidden_in];
                        for (d, &v) in dst.iter_mut().zip(src.iter()) {
                            *d += v * scale;
                        }
                    }
                }
                let out = self.apply_stage(compute, &self.stages[0], &pooled, t_out, *in_dim)?;
                debug_assert_eq!(out.len(), t_out * *out_dim);
                Ok(out)
            }
        }
    }

    fn check_hidden(&self, tag: &str, expected: usize, got: usize) -> Result<()> {
        if expected != got {
            return Err(VokraError::InvalidArgument(format!(
                "voxtral audio adapter {tag}: hidden {got} != expected {expected}"
            )));
        }
        Ok(())
    }

    fn apply_stage(
        &self,
        compute: &Compute,
        stage: &AdapterLinear,
        input: &[f32],
        rows: usize,
        _hidden_in: usize,
    ) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; rows * stage.out_features];
        // GEMM: [rows, in] × [in, out] → [rows, out].
        compute.gemm_f32(
            rows,
            stage.out_features,
            stage.in_features,
            input,
            &stage.w_t,
            stage.bias.as_deref(),
            &mut out,
        )?;
        // Optional post-LayerNorm.
        if let (Some(g), Some(b)) = (stage.ln_gamma.as_deref(), stage.ln_beta.as_deref()) {
            let mut ln_out = vec![0.0f32; rows * stage.out_features];
            compute.layer_norm_f32(
                &out,
                &mut ln_out,
                rows,
                stage.out_features,
                g,
                b,
                self.ln_eps,
            )?;
            return Ok(ln_out);
        }
        Ok(out)
    }

    fn activate(&self, compute: &Compute, x: &mut [f32]) -> Result<()> {
        match self.activation {
            AdapterActivation::Identity => Ok(()),
            AdapterActivation::Gelu => {
                let mut y = vec![0.0f32; x.len()];
                compute.gelu_f32(x, &mut y)?;
                x.copy_from_slice(&y);
                Ok(())
            }
            AdapterActivation::Silu => {
                // silu(x) = x * sigmoid(x). Scalar host loop — matches
                // super::text_decoder::silu_inplace (a text-decoder
                // scalar util, not a HotOp).
                for v in x.iter_mut() {
                    let s = 1.0 / (1.0 + (-*v).exp());
                    *v *= s;
                }
                Ok(())
            }
            AdapterActivation::Relu => {
                for v in x.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
                Ok(())
            }
        }
    }
}

impl std::fmt::Debug for AudioAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioAdapter")
            .field("kind", &self.kind)
            .field("activation", &self.activation)
            .field("stages", &self.stages.len())
            .field("time_stride", &self.time_stride)
            .finish()
    }
}

// ---------- Adapter shape queries ------------------------------------------

/// Returns the output channel width the adapter projects into. `AdapterKind::None`
/// returns 0 (there is no projection) — the caller must guard on
/// [`AudioAdapter::is_active`] before treating this as a meaningful width.
///
/// Used by [`super::AsrHead::transcribe`] to gate a misconfigured adapter that
/// projects into a different hidden width than the text decoder expects.
#[must_use]
pub(crate) fn out_dim(kind: &AdapterKind) -> usize {
    match kind {
        AdapterKind::None => 0,
        AdapterKind::Linear { out_dim, .. } => *out_dim,
        AdapterKind::Mlp { layers, .. } => layers.last().map(|l| l.out_dim).unwrap_or(0),
        AdapterKind::DownsampleLinear { out_dim, .. } => *out_dim,
    }
}

// ---------- Internals -------------------------------------------------------

fn bad(msg: String) -> VokraError {
    VokraError::ModelLoad(format!("voxtral adapter: {msg}"))
}

fn read_dims(file: &GgufFile) -> Result<(usize, usize)> {
    let in_dim = file
        .get(KEY_ADAPTER_IN_DIM)
        .and_then(|v| v.as_u64())
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| bad(format!("missing / non-u32 `{KEY_ADAPTER_IN_DIM}`")))?;
    let out_dim = file
        .get(KEY_ADAPTER_OUT_DIM)
        .and_then(|v| v.as_u64())
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| bad(format!("missing / non-u32 `{KEY_ADAPTER_OUT_DIM}`")))?;
    if in_dim == 0 || out_dim == 0 {
        return Err(bad(format!(
            "adapter dims must be non-zero (got in_dim={in_dim}, out_dim={out_dim})"
        )));
    }
    Ok((in_dim, out_dim))
}

/// `"128,256"` -> `[128, 256]`. Empty string -> empty vec.
fn parse_dim_list(s: &str) -> Result<Vec<usize>> {
    if s.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for part in s.split(',') {
        let t = part.trim();
        if t.is_empty() {
            continue;
        }
        let n: usize = t
            .parse()
            .map_err(|_| bad(format!("mlp_hidden_dims: cannot parse `{t}` as u32")))?;
        out.push(n);
    }
    Ok(out)
}

/// `"a,b,c"` -> `["a", "b", "c"]`. Empty string -> empty vec.
fn parse_str_list(s: &str) -> Vec<String> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Assembles a full tensor name `"{prefix}{layer_sub}.{weight_sub}"` (skipping
/// empty joins so a single-linear layout stays as `"{prefix}{weight_sub}"`).
fn join(prefix: &str, layer_sub: &str, sub: &str) -> String {
    match (layer_sub.is_empty(), sub.is_empty()) {
        (true, true) => prefix.to_owned(),
        (true, false) => format!("{prefix}{sub}"),
        (false, true) => format!("{prefix}{layer_sub}"),
        (false, false) => format!("{prefix}{layer_sub}.{sub}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn load_linear(
    file: &GgufFile,
    prefix: &str,
    layer_sub: &str,
    weight_sub: &str,
    bias_sub: &str,
    ln_gamma_sub: &str,
    ln_beta_sub: &str,
    in_dim: usize,
    out_dim: usize,
    has_bias: bool,
    has_ln: bool,
) -> Result<AdapterLinear> {
    let weight_name = join(prefix, layer_sub, weight_sub);
    // Convention: safetensors stores linear weights as [out, in]; we transpose
    // once at load into [in, out] for row-major GEMM.
    let w = tensor(file, &weight_name, &[out_dim, in_dim])?;
    let mut w_t = vec![0.0f32; in_dim * out_dim];
    for o in 0..out_dim {
        for i in 0..in_dim {
            w_t[i * out_dim + o] = w[o * in_dim + i];
        }
    }
    let bias = if has_bias {
        let bias_name = join(prefix, layer_sub, bias_sub);
        Some(tensor(file, &bias_name, &[out_dim])?)
    } else {
        None
    };
    let (ln_gamma, ln_beta) = if has_ln {
        let g = tensor(file, &join(prefix, layer_sub, ln_gamma_sub), &[out_dim])?;
        let b = tensor(file, &join(prefix, layer_sub, ln_beta_sub), &[out_dim])?;
        (Some(g), Some(b))
    } else {
        (None, None)
    };
    Ok(AdapterLinear {
        w_t,
        in_features: in_dim,
        out_features: out_dim,
        bias,
        ln_gamma,
        ln_beta,
    })
}

fn tensor(file: &GgufFile, name: &str, want: &[usize]) -> Result<Vec<f32>> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| bad(format!("`{name}` missing from GGUF")))?;
    let got: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    if got != want {
        return Err(bad(format!("`{name}` shape {got:?} != expected {want:?}")));
    }
    file.tensor_f32(name)
        .map_err(|e| bad(format!("`{name}`: {e}")))
}

#[cfg(test)]
// The tests below use `vec![0.0f32; N]` scratch buffers for readability where
// the size is compile-time known — clippy suggests arrays, but the test
// intent (mutable scratch shared by `iter().flat_map(...).collect()` bytes
// generation) reads cleaner as a Vec, so silence the whole module.
#[allow(clippy::useless_vec)]
mod tests {
    use super::*;
    use vokra_core::backend::BackendKind;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    fn compute() -> Compute {
        Compute::for_backend(BackendKind::Cpu, &[]).unwrap()
    }

    #[test]
    fn kind_none_apply_is_identity() {
        let a = AudioAdapter::none();
        assert!(matches!(a.kind(), AdapterKind::None));
        assert!(!a.is_active());
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = a
            .apply(&compute(), &x, /*t_in*/ 3, /*hidden_in*/ 2)
            .unwrap();
        assert_eq!(x, y, "None must be identity");
    }

    #[test]
    fn kind_none_from_gguf_when_key_absent() {
        // Empty GGUF: `kind` key absent → treat as None (backward compat).
        let file = GgufFile::parse(GgufBuilder::new().to_bytes().unwrap()).unwrap();
        let a = AudioAdapter::from_gguf(&file).unwrap();
        assert!(matches!(a.kind(), AdapterKind::None));
    }

    #[test]
    fn kind_none_from_gguf_when_key_explicitly_none() {
        let mut b = GgufBuilder::new();
        b.add_string(KEY_ADAPTER_KIND, "none");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let a = AudioAdapter::from_gguf(&file).unwrap();
        assert!(matches!(a.kind(), AdapterKind::None));
    }

    /// Builds a GGUF carrying a linear adapter `Linear(in_dim=4 → out_dim=6)`
    /// with `has_bias=true` and identity-initialized weight (`w[o,i]=1 iff
    /// o==i`, else 0), bias all zero. This lets tests exercise the load +
    /// apply path with an oracle that must reproduce the input in the
    /// first out_dim columns.
    fn linear_identity_gguf(prefix: &str) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(KEY_ADAPTER_KIND, "linear");
        b.add_string(KEY_ADAPTER_TENSOR_PREFIX, prefix);
        b.add_u32(KEY_ADAPTER_IN_DIM, 4);
        b.add_u32(KEY_ADAPTER_OUT_DIM, 4);
        b.add_bool(KEY_ADAPTER_HAS_BIAS, true);
        b.add_bool(KEY_ADAPTER_HAS_LN, false);
        // weight [out=4, in=4] identity.
        let mut w = vec![0.0f32; 16];
        for i in 0..4 {
            w[i * 4 + i] = 1.0;
        }
        let w_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        b.add_tensor(
            &format!("{prefix}weight"),
            GgmlType::F32,
            vec![4, 4],
            w_bytes,
        )
        .unwrap();
        // bias [4] zero.
        let bias_bytes = vec![0u8; 4 * 4];
        b.add_tensor(&format!("{prefix}bias"), GgmlType::F32, vec![4], bias_bytes)
            .unwrap();
        GgufFile::parse(b.to_bytes().unwrap()).unwrap()
    }

    #[test]
    fn kind_linear_identity_roundtrip() {
        let file = linear_identity_gguf("audio_adapter.");
        let a = AudioAdapter::from_gguf(&file).unwrap();
        assert!(a.is_active());
        assert!(matches!(
            a.kind(),
            AdapterKind::Linear {
                in_dim: 4,
                out_dim: 4,
                ..
            }
        ));
        let input: Vec<f32> = (0..8).map(|i| i as f32).collect(); // t_in=2, h=4
        let out = a.apply(&compute(), &input, 2, 4).unwrap();
        // With identity weight + zero bias, projection is identity.
        assert_eq!(out, input);
    }

    #[test]
    fn kind_linear_rejects_missing_weight_tensor() {
        let mut b = GgufBuilder::new();
        b.add_string(KEY_ADAPTER_KIND, "linear");
        b.add_string(KEY_ADAPTER_TENSOR_PREFIX, "audio_adapter.");
        b.add_u32(KEY_ADAPTER_IN_DIM, 4);
        b.add_u32(KEY_ADAPTER_OUT_DIM, 4);
        // No tensors added — load must surface the missing weight by name.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let err = AudioAdapter::from_gguf(&file).unwrap_err();
        assert!(
            matches!(err, VokraError::ModelLoad(ref m) if m.contains("weight")),
            "{err:?}"
        );
    }

    #[test]
    fn kind_linear_rejects_shape_mismatch_input() {
        let file = linear_identity_gguf("audio_adapter.");
        let a = AudioAdapter::from_gguf(&file).unwrap();
        // Wrong hidden width.
        let input = vec![0.0f32; 6]; // t_in=2, h=3 — mismatched
        assert!(matches!(
            a.apply(&compute(), &input, 2, 3),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn kind_linear_zero_dim_is_model_load_error() {
        let mut b = GgufBuilder::new();
        b.add_string(KEY_ADAPTER_KIND, "linear");
        b.add_u32(KEY_ADAPTER_IN_DIM, 0);
        b.add_u32(KEY_ADAPTER_OUT_DIM, 4);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(matches!(
            AudioAdapter::from_gguf(&file),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn kind_linear_with_layernorm() {
        // Linear with LN gamma=1, beta=0 -> LN normalises row to zero-mean
        // unit-variance, then linear (identity). We can't check exact values
        // without a full LN oracle, but we can verify the apply runs to
        // completion and produces a finite `[t_in, out_dim]`.
        let mut b = GgufBuilder::new();
        b.add_string(KEY_ADAPTER_KIND, "linear");
        b.add_string(KEY_ADAPTER_TENSOR_PREFIX, "adap.");
        b.add_u32(KEY_ADAPTER_IN_DIM, 4);
        b.add_u32(KEY_ADAPTER_OUT_DIM, 4);
        b.add_bool(KEY_ADAPTER_HAS_BIAS, false);
        b.add_bool(KEY_ADAPTER_HAS_LN, true);
        // Identity weight (transposed same in [out, in] layout).
        let mut w = vec![0.0f32; 16];
        for i in 0..4 {
            w[i * 4 + i] = 1.0;
        }
        let w_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        b.add_tensor("adap.weight", GgmlType::F32, vec![4, 4], w_bytes)
            .unwrap();
        let g: Vec<f32> = vec![1.0; 4];
        let z: Vec<f32> = vec![0.0; 4];
        b.add_tensor(
            "adap.layernorm.weight",
            GgmlType::F32,
            vec![4],
            g.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap();
        b.add_tensor(
            "adap.layernorm.bias",
            GgmlType::F32,
            vec![4],
            z.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap();
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let a = AudioAdapter::from_gguf(&file).unwrap();
        assert!(a.is_active());
        // Non-uniform input row so LN has something to normalise.
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
        let out = a.apply(&compute(), &input, 2, 4).unwrap();
        assert_eq!(out.len(), 2 * 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn kind_mlp_multi_layer_shape_correctness() {
        // MLP: 4 -> 8 -> 4 with GELU between. Weights all identity where
        // possible; the second layer has shape [4, 8] so we pack a specific
        // pattern (down-project by taking first 4 of 8). We check the *shape*
        // and that it runs — a numeric oracle would require GELU internals.
        let mut b = GgufBuilder::new();
        b.add_string(KEY_ADAPTER_KIND, "mlp");
        b.add_string(KEY_ADAPTER_TENSOR_PREFIX, "mlp.");
        b.add_u32(KEY_ADAPTER_IN_DIM, 4);
        b.add_u32(KEY_ADAPTER_OUT_DIM, 4);
        b.add_string(KEY_ADAPTER_MLP_HIDDEN_DIMS, "8");
        b.add_string(KEY_ADAPTER_ACTIVATION, "gelu");
        b.add_bool(KEY_ADAPTER_HAS_BIAS, false);
        b.add_bool(KEY_ADAPTER_HAS_LN, false);
        // layer 0: [out=8, in=4].
        let mut w0 = vec![0.0f32; 32];
        for i in 0..4 {
            w0[i * 4 + i] = 1.0; // top 4 rows = identity in first 4 cols
        }
        b.add_tensor(
            "mlp.layers.0.weight",
            GgmlType::F32,
            vec![8, 4],
            w0.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap();
        // layer 1: [out=4, in=8].
        let mut w1 = vec![0.0f32; 32];
        for i in 0..4 {
            w1[i * 8 + i] = 1.0; // take first 4 of 8
        }
        b.add_tensor(
            "mlp.layers.1.weight",
            GgmlType::F32,
            vec![4, 8],
            w1.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap();
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let a = AudioAdapter::from_gguf(&file).unwrap();
        assert!(matches!(a.kind(), AdapterKind::Mlp { .. }));
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
        let out = a.apply(&compute(), &input, 2, 4).unwrap();
        assert_eq!(out.len(), 2 * 4, "shape must be preserved through MLP");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn kind_downsample_linear_shrinks_time_axis() {
        let mut b = GgufBuilder::new();
        b.add_string(KEY_ADAPTER_KIND, "downsample_linear");
        b.add_string(KEY_ADAPTER_TENSOR_PREFIX, "ds.");
        b.add_u32(KEY_ADAPTER_IN_DIM, 4);
        b.add_u32(KEY_ADAPTER_OUT_DIM, 4);
        b.add_u32(KEY_ADAPTER_TIME_STRIDE, 2);
        b.add_bool(KEY_ADAPTER_HAS_BIAS, false);
        b.add_bool(KEY_ADAPTER_HAS_LN, false);
        // Identity weight.
        let mut w = vec![0.0f32; 16];
        for i in 0..4 {
            w[i * 4 + i] = 1.0;
        }
        b.add_tensor(
            "ds.weight",
            GgmlType::F32,
            vec![4, 4],
            w.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap();
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let a = AudioAdapter::from_gguf(&file).unwrap();
        assert!(matches!(
            a.kind(),
            AdapterKind::DownsampleLinear { time_stride: 2, .. }
        ));
        // 4 rows of 4 cols → 2 rows of 4 cols after avg-pool with stride 2.
        let input: Vec<f32> = vec![
            1.0, 1.0, 1.0, 1.0, // row 0
            3.0, 3.0, 3.0, 3.0, // row 1
            5.0, 5.0, 5.0, 5.0, // row 2
            7.0, 7.0, 7.0, 7.0, // row 3
        ];
        let out = a.apply(&compute(), &input, 4, 4).unwrap();
        assert_eq!(out.len(), 2 * 4);
        // avg((1,3))=2, avg((5,7))=6 → identity projection preserves it.
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[4] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn kind_downsample_zero_stride_is_model_load_error() {
        let mut b = GgufBuilder::new();
        b.add_string(KEY_ADAPTER_KIND, "downsample_linear");
        b.add_u32(KEY_ADAPTER_IN_DIM, 4);
        b.add_u32(KEY_ADAPTER_OUT_DIM, 4);
        b.add_u32(KEY_ADAPTER_TIME_STRIDE, 0);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(matches!(
            AudioAdapter::from_gguf(&file),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn unknown_kind_is_model_load_error() {
        let mut b = GgufBuilder::new();
        b.add_string(KEY_ADAPTER_KIND, "attention_pool");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let err = AudioAdapter::from_gguf(&file).unwrap_err();
        assert!(matches!(err, VokraError::ModelLoad(_)));
    }

    #[test]
    fn unknown_activation_is_model_load_error() {
        let mut b = GgufBuilder::new();
        b.add_string(KEY_ADAPTER_KIND, "mlp");
        b.add_u32(KEY_ADAPTER_IN_DIM, 4);
        b.add_u32(KEY_ADAPTER_OUT_DIM, 4);
        b.add_string(KEY_ADAPTER_ACTIVATION, "quantum_swish");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let err = AudioAdapter::from_gguf(&file).unwrap_err();
        assert!(matches!(err, VokraError::ModelLoad(_)));
    }

    #[test]
    fn parse_dim_list_handles_edges() {
        assert!(parse_dim_list("").unwrap().is_empty());
        assert_eq!(parse_dim_list("128").unwrap(), vec![128]);
        assert_eq!(parse_dim_list("128,256, 512").unwrap(), vec![128, 256, 512]);
        assert!(parse_dim_list("abc").is_err());
    }

    #[test]
    fn parse_str_list_trims_and_filters() {
        assert!(parse_str_list("").is_empty());
        assert_eq!(parse_str_list("a"), vec!["a"]);
        assert_eq!(parse_str_list("a, b ,c"), vec!["a", "b", "c"]);
        assert_eq!(parse_str_list("a,,b"), vec!["a", "b"]);
    }

    #[test]
    fn activation_parse_covers_all_variants() {
        assert_eq!(
            AdapterActivation::parse("gelu"),
            Some(AdapterActivation::Gelu)
        );
        assert_eq!(
            AdapterActivation::parse("silu"),
            Some(AdapterActivation::Silu)
        );
        assert_eq!(
            AdapterActivation::parse("swish"),
            Some(AdapterActivation::Silu)
        );
        assert_eq!(
            AdapterActivation::parse("relu"),
            Some(AdapterActivation::Relu)
        );
        assert_eq!(
            AdapterActivation::parse("identity"),
            Some(AdapterActivation::Identity)
        );
        assert_eq!(
            AdapterActivation::parse("none"),
            Some(AdapterActivation::Identity)
        );
        assert_eq!(AdapterActivation::parse("banana"), None);
    }
}
