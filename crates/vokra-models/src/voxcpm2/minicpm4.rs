//! Source-shaped MiniCPM-4 transformer primitives used by VoxCPM.
//!
//! This module deliberately contains no checkpoint discovery or permissive
//! tensor loader.  Callers construct the typed weights with the exact source
//! dimensions; malformed dimensions are rejected before a backend is chosen.
//! Every learned matrix multiplication, softmax, RMSNorm, and SiLU operation
//! is dispatched through [`Compute`].  In particular, selecting Metal never
//! causes an uncovered operation to run on the CPU.

use vokra_core::backend::BackendKind;
use vokra_core::{Result, VokraError};

use crate::compute::{Compute, HotOp};

/// Operations required by every MiniCPM-4 stack.
pub const MINICPM4_HOT_OPS: &[HotOp] = &[HotOp::Gemm, HotOp::Softmax, HotOp::RmsNorm, HotOp::Silu];

/// MiniCPM-4 axes shared by the base LM, residual LM, local encoder, and
/// local DiT blocks.  `n_layers` is carried by the generic stack rather than
/// being baked into its implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniCpm4Config {
    hidden_dim: usize,
    ffn_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    max_positions: usize,
    /// Effective-length threshold at which the source selects long factors.
    /// This is distinct from the maximum supported context.
    original_max_positions: usize,
    rope_theta: f32,
    rms_norm_eps: f32,
    use_mup: bool,
    /// LongRoPE factors for the `head_dim / 2` rotary pairs.
    rope_short_factor: Vec<f32>,
    rope_long_factor: Vec<f32>,
    rope_scale: f32,
}

impl MiniCpm4Config {
    /// Construct and validate a generic MiniCPM-4 configuration.
    pub fn new(
        hidden_dim: usize,
        ffn_dim: usize,
        n_layers: usize,
        n_heads: usize,
        n_kv_heads: usize,
        max_positions: usize,
        rope_theta: f32,
        rms_norm_eps: f32,
        use_mup: bool,
        rope_short_factor: Vec<f32>,
        rope_long_factor: Vec<f32>,
    ) -> Result<Self> {
        Self::new_with_original_max_positions(
            hidden_dim,
            ffn_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            max_positions,
            max_positions,
            rope_theta,
            rms_norm_eps,
            use_mup,
            rope_short_factor,
            rope_long_factor,
        )
    }

    /// Construct with an explicit original context and supported maximum.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_original_max_positions(
        hidden_dim: usize,
        ffn_dim: usize,
        n_layers: usize,
        n_heads: usize,
        n_kv_heads: usize,
        max_positions: usize,
        original_max_positions: usize,
        rope_theta: f32,
        rms_norm_eps: f32,
        use_mup: bool,
        rope_short_factor: Vec<f32>,
        rope_long_factor: Vec<f32>,
    ) -> Result<Self> {
        let config = Self {
            hidden_dim,
            ffn_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            max_positions,
            original_max_positions,
            rope_theta,
            rms_norm_eps,
            use_mup,
            rope_short_factor,
            rope_long_factor,
            rope_scale: if original_max_positions > 1 {
                (max_positions as f32 / original_max_positions as f32)
                    .ln()
                    .mul_add(1.0 / (original_max_positions as f32).ln(), 1.0)
                    .sqrt()
            } else if original_max_positions == 1 && max_positions > 0 {
                1.0
            } else {
                f32::NAN
            },
        };
        config.validate()?;
        Ok(config)
    }

    /// The authenticated 0.5B axes and LongRoPE arrays from the fixed
    /// `config.json` companion.
    pub fn voxcpm_0_5b() -> Result<Self> {
        const FACTORS: [f32; 32] = [
            1.0004360675811768,
            1.0668443441390991,
            1.1631425619125366,
            1.3025742769241333,
            1.5040205717086792,
            1.7941505908966064,
            2.2101221084594727,
            2.802666664123535,
            3.6389970779418945,
            4.804192543029785,
            6.39855432510376,
            8.527148246765137,
            11.277542114257812,
            14.684998512268066,
            18.69317054748535,
            23.13019371032715,
            27.72362518310547,
            32.1606559753418,
            36.168827056884766,
            39.57627868652344,
            42.32667541503906,
            44.45526885986328,
            46.049629974365234,
            47.21482849121094,
            48.05115509033203,
            48.64370346069336,
            49.05967712402344,
            49.34980392456055,
            49.551246643066406,
            49.69068145751953,
            49.78697967529297,
            49.85338592529297,
        ];
        Self::new(
            1024,
            4096,
            24,
            16,
            2,
            32768,
            10_000.0,
            1e-5,
            false,
            FACTORS.to_vec(),
            FACTORS.to_vec(),
        )
    }

    fn validate(&self) -> Result<()> {
        if self.hidden_dim == 0
            || self.ffn_dim == 0
            || self.n_layers == 0
            || self.n_heads == 0
            || self.n_kv_heads == 0
            || self.max_positions == 0
            || self.original_max_positions == 0
        {
            return Err(VokraError::InvalidArgument(
                "MiniCPM-4 dimensions and max_positions must be non-zero".to_owned(),
            ));
        }
        if self.original_max_positions > self.max_positions {
            return Err(VokraError::InvalidArgument(format!(
                "MiniCPM-4 original_max_positions {} exceeds max_positions {}",
                self.original_max_positions, self.max_positions
            )));
        }
        if self.hidden_dim % self.n_heads != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "MiniCPM-4 hidden_dim {} is not divisible by n_heads {}",
                self.hidden_dim, self.n_heads
            )));
        }
        if self.n_heads % self.n_kv_heads != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "MiniCPM-4 n_heads {} is not divisible by n_kv_heads {}",
                self.n_heads, self.n_kv_heads
            )));
        }
        if self.head_dim() % 2 != 0
            || self.rope_short_factor.len() != self.head_dim() / 2
            || self.rope_long_factor.len() != self.head_dim() / 2
        {
            return Err(VokraError::InvalidArgument(format!(
                "MiniCPM-4 LongRoPE factors must each have head_dim/2={} entries",
                self.head_dim() / 2
            )));
        }
        if !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
        {
            return Err(VokraError::InvalidArgument(
                "MiniCPM-4 rope_theta and rms_norm_eps must be positive finite values".to_owned(),
            ));
        }
        if !self.rope_scale.is_finite() || self.rope_scale <= 0.0 {
            return Err(VokraError::InvalidArgument(
                "MiniCPM-4 LongRoPE scale must be positive and finite".to_owned(),
            ));
        }
        if self.use_mup {
            return Err(VokraError::UnsupportedOp(
                "MiniCPM-4 µ-parametrized residual scaling is not part of the authenticated VoxCPM route"
                    .to_owned(),
            ));
        }
        if self
            .rope_short_factor
            .iter()
            .chain(self.rope_long_factor.iter())
            .any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err(VokraError::InvalidArgument(
                "MiniCPM-4 LongRoPE factors must be positive finite values".to_owned(),
            ));
        }
        Ok(())
    }

    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.n_heads
    }

    #[must_use]
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim()
    }

    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    #[must_use]
    pub fn ffn_dim(&self) -> usize {
        self.ffn_dim
    }

    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    #[must_use]
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    #[must_use]
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    #[must_use]
    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    #[must_use]
    pub fn original_max_positions(&self) -> usize {
        self.original_max_positions
    }

    #[must_use]
    pub fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    #[must_use]
    pub fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }

    #[must_use]
    pub fn rope_short_factor(&self) -> &[f32] {
        &self.rope_short_factor
    }

    #[must_use]
    pub fn rope_long_factor(&self) -> &[f32] {
        &self.rope_long_factor
    }

    #[must_use]
    pub fn rope_scale(&self) -> f32 {
        self.rope_scale
    }
}

/// A strict source-layout linear weight.  Source checkpoints store
/// `[out_features, in_features]`; the Compute seam consumes `[in, out]`, so
/// the transpose is performed once during authenticated binding.
#[derive(Debug, Clone)]
pub struct MiniCpm4Linear {
    w_t: Vec<f32>,
    bias: Option<Vec<f32>>,
    in_features: usize,
    out_features: usize,
}

impl MiniCpm4Linear {
    pub fn from_source(
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
        out_features: usize,
        in_features: usize,
    ) -> Result<Self> {
        if out_features == 0 || in_features == 0 {
            return Err(VokraError::InvalidArgument(
                "MiniCPM-4 linear dimensions must be non-zero".to_owned(),
            ));
        }
        let expected = out_features.checked_mul(in_features).ok_or_else(|| {
            VokraError::InvalidArgument("MiniCPM-4 linear dimensions overflow".to_owned())
        })?;
        if weight.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "MiniCPM-4 linear weight has {} values, expected {} ({out_features}x{in_features})",
                weight.len(),
                expected
            )));
        }
        if weight.iter().any(|x| !x.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "MiniCPM-4 linear weight contains non-finite values".to_owned(),
            ));
        }
        if let Some(ref b) = bias {
            if b.len() != out_features || b.iter().any(|x| !x.is_finite()) {
                return Err(VokraError::InvalidArgument(
                    "MiniCPM-4 linear bias has wrong shape or non-finite values".to_owned(),
                ));
            }
        }
        let mut w_t = vec![0.0; expected];
        for out in 0..out_features {
            for input in 0..in_features {
                w_t[input * out_features + out] = weight[out * in_features + input];
            }
        }
        Ok(Self {
            w_t,
            bias,
            in_features,
            out_features,
        })
    }

    pub(crate) fn apply(
        &self,
        compute: &Compute,
        input: &[f32],
        rows: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let expected_input = rows.checked_mul(self.in_features).ok_or_else(|| {
            VokraError::InvalidArgument("MiniCPM-4 linear row count overflow".to_owned())
        })?;
        let expected_output = rows.checked_mul(self.out_features).ok_or_else(|| {
            VokraError::InvalidArgument("MiniCPM-4 linear output row count overflow".to_owned())
        })?;
        if input.len() != expected_input || output.len() != expected_output {
            return Err(VokraError::InvalidArgument(format!(
                "MiniCPM-4 linear expected input/output {expected_input}/{expected_output}, got {}/{}",
                input.len(),
                output.len()
            )));
        }
        compute.gemm_f32(
            rows,
            self.out_features,
            self.in_features,
            input,
            &self.w_t,
            self.bias.as_deref(),
            output,
        )
    }

    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

/// One strict MiniCPM-4 transformer block.
#[derive(Debug, Clone)]
pub struct MiniCpm4BlockWeights {
    input_layernorm: Vec<f32>,
    post_attention_layernorm: Vec<f32>,
    q_proj: MiniCpm4Linear,
    k_proj: MiniCpm4Linear,
    v_proj: MiniCpm4Linear,
    o_proj: MiniCpm4Linear,
    gate_proj: MiniCpm4Linear,
    up_proj: MiniCpm4Linear,
    down_proj: MiniCpm4Linear,
}

impl MiniCpm4BlockWeights {
    /// Bind exactly the source names' matrix axes.  All MiniCPM-4 linear
    /// layers are bias-free; callers pass `None` for each bias.
    #[allow(clippy::too_many_arguments)]
    pub fn from_source(
        config: &MiniCpm4Config,
        input_layernorm: Vec<f32>,
        post_attention_layernorm: Vec<f32>,
        q_proj: Vec<f32>,
        k_proj: Vec<f32>,
        v_proj: Vec<f32>,
        o_proj: Vec<f32>,
        gate_proj: Vec<f32>,
        up_proj: Vec<f32>,
        down_proj: Vec<f32>,
    ) -> Result<Self> {
        check_vector("input_layernorm", &input_layernorm, config.hidden_dim)?;
        check_vector(
            "post_attention_layernorm",
            &post_attention_layernorm,
            config.hidden_dim,
        )?;
        let q_proj =
            MiniCpm4Linear::from_source(q_proj, None, config.hidden_dim, config.hidden_dim)?;
        let k_proj = MiniCpm4Linear::from_source(k_proj, None, config.kv_dim(), config.hidden_dim)?;
        let v_proj = MiniCpm4Linear::from_source(v_proj, None, config.kv_dim(), config.hidden_dim)?;
        let o_proj =
            MiniCpm4Linear::from_source(o_proj, None, config.hidden_dim, config.hidden_dim)?;
        let gate_proj =
            MiniCpm4Linear::from_source(gate_proj, None, config.ffn_dim, config.hidden_dim)?;
        let up_proj =
            MiniCpm4Linear::from_source(up_proj, None, config.ffn_dim, config.hidden_dim)?;
        let down_proj =
            MiniCpm4Linear::from_source(down_proj, None, config.hidden_dim, config.ffn_dim)?;
        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

/// Strict weights for a reusable stack of MiniCPM-4 blocks.
#[derive(Debug, Clone)]
pub struct MiniCpm4StackWeights {
    blocks: Vec<MiniCpm4BlockWeights>,
    final_norm: Vec<f32>,
}

impl MiniCpm4StackWeights {
    /// Construct the validated stack payload. Shape and finiteness checks are
    /// performed when it is attached to a [`MiniCpm4Stack`].
    pub(crate) fn from_source(blocks: Vec<MiniCpm4BlockWeights>, final_norm: Vec<f32>) -> Self {
        Self { blocks, final_norm }
    }
}

/// A reusable MiniCPM-4 transformer stack (base LM, residual LM, local
/// encoder, or local DiT).  It consumes hidden rows rather than assuming a
/// particular token vocabulary.
#[derive(Debug, Clone)]
pub struct MiniCpm4Stack {
    config: MiniCpm4Config,
    weights: MiniCpm4StackWeights,
}

impl MiniCpm4Stack {
    pub fn new(config: MiniCpm4Config, weights: MiniCpm4StackWeights) -> Result<Self> {
        config.validate()?;
        if weights.blocks.len() != config.n_layers {
            return Err(VokraError::InvalidArgument(format!(
                "MiniCPM-4 stack has {} blocks, expected {}",
                weights.blocks.len(),
                config.n_layers
            )));
        }
        check_vector("final_norm", &weights.final_norm, config.hidden_dim)?;
        for (index, block) in weights.blocks.iter().enumerate() {
            check_vector(
                &format!("block {index} input_layernorm"),
                &block.input_layernorm,
                config.hidden_dim,
            )?;
            check_vector(
                &format!("block {index} post_attention_layernorm"),
                &block.post_attention_layernorm,
                config.hidden_dim,
            )?;
            for (name, linear, input, output) in [
                (
                    "q_proj",
                    &block.q_proj,
                    config.hidden_dim,
                    config.hidden_dim,
                ),
                ("k_proj", &block.k_proj, config.hidden_dim, config.kv_dim()),
                ("v_proj", &block.v_proj, config.hidden_dim, config.kv_dim()),
                (
                    "o_proj",
                    &block.o_proj,
                    config.hidden_dim,
                    config.hidden_dim,
                ),
                (
                    "gate_proj",
                    &block.gate_proj,
                    config.hidden_dim,
                    config.ffn_dim,
                ),
                ("up_proj", &block.up_proj, config.hidden_dim, config.ffn_dim),
                (
                    "down_proj",
                    &block.down_proj,
                    config.ffn_dim,
                    config.hidden_dim,
                ),
            ] {
                if linear.in_features != input || linear.out_features != output {
                    return Err(VokraError::InvalidArgument(format!(
                        "MiniCPM-4 block {index} {name} has {}x{}, expected {output}x{input}",
                        linear.out_features, linear.in_features
                    )));
                }
            }
        }
        Ok(Self { config, weights })
    }

    #[must_use]
    pub fn config(&self) -> &MiniCpm4Config {
        &self.config
    }

    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.weights.blocks.len()
    }

    /// Full sequence forward. `causal=false` is used by the local encoder /
    /// DiT stack; causal mode is used by the language models.
    pub fn forward(
        &self,
        hidden: &[f32],
        rows: usize,
        causal: bool,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        self.forward_with_cache(
            hidden,
            rows,
            causal,
            &mut MiniCpm4KvCache::new(self),
            compute,
        )
    }

    /// One-token forward with a per-layer KV cache.  `hidden` is one row and
    /// must be the same input representation as the full sequence path.
    pub fn forward_step(
        &self,
        hidden: &[f32],
        cache: &mut MiniCpm4KvCache,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        if hidden.len() != self.config.hidden_dim {
            return Err(VokraError::InvalidArgument(format!(
                "MiniCPM-4 forward_step hidden length {} != {}",
                hidden.len(),
                self.config.hidden_dim
            )));
        }
        let out = self.forward_with_cache(hidden, 1, true, cache, compute)?;
        Ok(out)
    }

    fn forward_with_cache(
        &self,
        hidden: &[f32],
        rows: usize,
        causal: bool,
        cache: &mut MiniCpm4KvCache,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        let checkpoint = cache.checkpoint();
        match self.forward_with_cache_inner(hidden, rows, causal, cache, compute) {
            Ok(output) => Ok(output),
            Err(error) => {
                cache.restore(&checkpoint);
                Err(error)
            }
        }
    }

    /// Cache-backed sequence forward used by the source-shaped generation
    /// runtime.  The caller supplies the persistent cache; the same
    /// transaction/rollback contract as `forward_step` applies.
    pub(crate) fn forward_cached(
        &self,
        hidden: &[f32],
        rows: usize,
        causal: bool,
        cache: &mut MiniCpm4KvCache,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        self.forward_with_cache(hidden, rows, causal, cache, compute)
    }

    fn forward_with_cache_inner(
        &self,
        hidden: &[f32],
        rows: usize,
        causal: bool,
        cache: &mut MiniCpm4KvCache,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        if rows == 0 || hidden.len() != rows * self.config.hidden_dim {
            return Err(VokraError::InvalidArgument(format!(
                "MiniCPM-4 forward expected {} hidden values, got {}",
                rows * self.config.hidden_dim,
                hidden.len()
            )));
        }
        if cache.layers.len() != self.weights.blocks.len() {
            return Err(VokraError::InvalidArgument(
                "MiniCPM-4 KV cache layer count does not match stack".to_owned(),
            ));
        }
        let start = cache.positions;
        if start
            .checked_add(rows)
            .is_none_or(|end| end > self.config.max_positions)
        {
            return Err(VokraError::InvalidArgument(format!(
                "MiniCPM-4 sequence end exceeds max_positions {}",
                self.config.max_positions
            )));
        }
        let hidden_dim = self.config.hidden_dim;
        let kv_dim = self.config.kv_dim();
        let head_dim = self.config.head_dim();
        let mut h = hidden.to_vec();
        for (layer_index, block) in self.weights.blocks.iter().enumerate() {
            let mut norm = vec![0.0; rows * hidden_dim];
            compute.rms_norm_f32(
                &h,
                &mut norm,
                rows,
                hidden_dim,
                &block.input_layernorm,
                self.config.rms_norm_eps,
            )?;
            let mut q = vec![0.0; rows * hidden_dim];
            let mut k = vec![0.0; rows * kv_dim];
            let mut v = vec![0.0; rows * kv_dim];
            block.q_proj.apply(compute, &norm, rows, &mut q)?;
            block.k_proj.apply(compute, &norm, rows, &mut k)?;
            block.v_proj.apply(compute, &norm, rows, &mut v)?;
            for row in 0..rows {
                apply_rope(
                    &mut q[row * hidden_dim..(row + 1) * hidden_dim],
                    start + row,
                    self.config.n_heads,
                    head_dim,
                    &self.config,
                    cache.use_long_rope,
                );
                apply_rope(
                    &mut k[row * kv_dim..(row + 1) * kv_dim],
                    start + row,
                    self.config.n_kv_heads,
                    head_dim,
                    &self.config,
                    cache.use_long_rope,
                );
            }
            cache.layers[layer_index].append(&k, &v);
            let total = cache.layers[layer_index].keys.len() / kv_dim;
            let mut attended = vec![0.0; rows * hidden_dim];
            attention(
                compute,
                &q,
                &cache.layers[layer_index].keys,
                &cache.layers[layer_index].values,
                rows,
                total,
                self.config.n_heads,
                self.config.n_kv_heads,
                head_dim,
                start,
                causal,
                &mut attended,
            )?;
            let mut projected = vec![0.0; rows * hidden_dim];
            block
                .o_proj
                .apply(compute, &attended, rows, &mut projected)?;
            for (value, residual) in h.iter_mut().zip(projected) {
                *value += residual;
            }

            compute.rms_norm_f32(
                &h,
                &mut norm,
                rows,
                hidden_dim,
                &block.post_attention_layernorm,
                self.config.rms_norm_eps,
            )?;
            let mut gate = vec![0.0; rows * self.config.ffn_dim];
            let mut up = vec![0.0; rows * self.config.ffn_dim];
            block.gate_proj.apply(compute, &norm, rows, &mut gate)?;
            block.up_proj.apply(compute, &norm, rows, &mut up)?;
            let mut activated = vec![0.0; gate.len()];
            compute.silu_f32(&gate, &mut activated)?;
            for (a, b) in activated.iter_mut().zip(up) {
                *a *= b;
            }
            let mut down = vec![0.0; rows * hidden_dim];
            block
                .down_proj
                .apply(compute, &activated, rows, &mut down)?;
            for (value, residual) in h.iter_mut().zip(down) {
                *value += residual;
            }
        }
        cache.positions += rows;
        let mut output = vec![0.0; h.len()];
        compute.rms_norm_f32(
            &h,
            &mut output,
            rows,
            hidden_dim,
            &self.weights.final_norm,
            self.config.rms_norm_eps,
        )?;
        Ok(output)
    }
}

/// Token-embedding front end for a base MiniCPM-4 language model.
#[derive(Debug, Clone)]
pub struct MiniCpm4Model {
    stack: MiniCpm4Stack,
    embedding: Vec<f32>,
    vocab_size: usize,
}

impl MiniCpm4Model {
    pub fn from_source_embedding(stack: MiniCpm4Stack, embedding: Vec<f32>) -> Result<Self> {
        if embedding.is_empty() || embedding.len() % stack.config.hidden_dim != 0 {
            return Err(VokraError::InvalidArgument(
                "MiniCPM-4 embedding must be a non-empty [vocab, hidden] matrix".to_owned(),
            ));
        }
        if embedding.iter().any(|x| !x.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "MiniCPM-4 embedding contains non-finite values".to_owned(),
            ));
        }
        Ok(Self {
            vocab_size: embedding.len() / stack.config.hidden_dim,
            stack,
            embedding,
        })
    }

    #[must_use]
    pub fn stack(&self) -> &MiniCpm4Stack {
        &self.stack
    }

    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn forward_tokens(
        &self,
        tokens: &[u32],
        causal: bool,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(VokraError::InvalidArgument(
                "MiniCPM-4 token sequence must not be empty".to_owned(),
            ));
        }
        let hidden_dim = self.stack.config.hidden_dim;
        let mut hidden = vec![0.0; tokens.len() * hidden_dim];
        for (row, &token) in tokens.iter().enumerate() {
            let token = token as usize;
            if token >= self.vocab_size {
                return Err(VokraError::InvalidArgument(format!(
                    "MiniCPM-4 token {token} >= vocab_size {}",
                    self.vocab_size
                )));
            }
            hidden[row * hidden_dim..(row + 1) * hidden_dim]
                .copy_from_slice(&self.embedding[token * hidden_dim..(token + 1) * hidden_dim]);
        }
        self.stack.forward(&hidden, tokens.len(), causal, compute)
    }
}

/// Per-layer KV state for one reusable MiniCPM-4 stack.
#[derive(Debug, Clone)]
pub struct MiniCpm4KvCache {
    layers: Vec<LayerKvCache>,
    positions: usize,
    use_long_rope: bool,
}

impl MiniCpm4KvCache {
    #[must_use]
    pub fn new(stack: &MiniCpm4Stack) -> Self {
        Self {
            layers: (0..stack.weights.blocks.len())
                .map(|_| LayerKvCache {
                    keys: Vec::new(),
                    values: Vec::new(),
                })
                .collect(),
            positions: 0,
            // The source constructs its rotary cache at the configured
            // max_position_embeddings, so this decision is shared by every
            // full and incremental call on the stack.
            use_long_rope: stack.config.max_positions > stack.config.original_max_positions,
        }
    }

    /// Create a cache with the source's effective sequence-length RoPE mode.
    /// The mode is fixed for the cache so full and incremental calls cannot
    /// silently rotate old keys with a different frequency table.
    pub fn with_sequence_len(stack: &MiniCpm4Stack, sequence_len: usize) -> Result<Self> {
        if sequence_len == 0 || sequence_len > stack.config.max_positions {
            return Err(VokraError::InvalidArgument(format!(
                "MiniCPM-4 sequence_len {sequence_len} is outside 1..={}",
                stack.config.max_positions
            )));
        }
        let mut cache = Self::new(stack);
        cache.use_long_rope = sequence_len > stack.config.original_max_positions;
        Ok(cache)
    }

    #[must_use]
    pub fn positions(&self) -> usize {
        self.positions
    }

    pub(crate) fn checkpoint(&self) -> CacheCheckpoint {
        CacheCheckpoint {
            positions: self.positions,
            lengths: self
                .layers
                .iter()
                .map(|layer| (layer.keys.len(), layer.values.len()))
                .collect(),
        }
    }

    pub(crate) fn restore(&mut self, checkpoint: &CacheCheckpoint) {
        self.positions = checkpoint.positions;
        for (layer, &(key_len, value_len)) in self.layers.iter_mut().zip(&checkpoint.lengths) {
            layer.keys.truncate(key_len);
            layer.values.truncate(value_len);
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CacheCheckpoint {
    positions: usize,
    lengths: Vec<(usize, usize)>,
}

#[derive(Debug, Clone)]
struct LayerKvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl LayerKvCache {
    fn append(&mut self, keys: &[f32], values: &[f32]) {
        self.keys.extend_from_slice(keys);
        self.values.extend_from_slice(values);
    }
}

fn check_vector(name: &str, values: &[f32], expected: usize) -> Result<()> {
    if values.len() != expected || values.iter().any(|x| !x.is_finite()) {
        return Err(VokraError::InvalidArgument(format!(
            "MiniCPM-4 {name} expected {expected} finite values, got {}",
            values.len()
        )));
    }
    Ok(())
}

fn apply_rope(
    values: &mut [f32],
    position: usize,
    heads: usize,
    head_dim: usize,
    config: &MiniCpm4Config,
    use_long_rope: bool,
) {
    for head in 0..heads {
        let start = head * head_dim;
        for pair in 0..head_dim / 2 {
            let inv = config
                .rope_theta
                .powf(-(2.0 * pair as f32) / head_dim as f32);
            let factor = if use_long_rope {
                config.rope_long_factor[pair]
            } else {
                config.rope_short_factor[pair]
            };
            let angle = position as f32 * inv / factor;
            let (sin, cos) = angle.sin_cos();
            let sin = sin * config.rope_scale;
            let cos = cos * config.rope_scale;
            // Hugging Face MiniCPM uses rotate_half: the first half of the
            // head is paired with the second half, not adjacent lanes.
            let left = values[start + pair];
            let right = values[start + head_dim / 2 + pair];
            values[start + pair] = left * cos - right * sin;
            values[start + head_dim / 2 + pair] = left * sin + right * cos;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn attention(
    compute: &Compute,
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    query_rows: usize,
    key_rows: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    query_position: usize,
    causal: bool,
    output: &mut [f32],
) -> Result<()> {
    let hidden_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    if query.len() != query_rows * hidden_dim
        || keys.len() != key_rows * kv_dim
        || values.len() != key_rows * kv_dim
        || output.len() != query_rows * hidden_dim
    {
        return Err(VokraError::InvalidArgument(
            "MiniCPM-4 attention buffer shape mismatch".to_owned(),
        ));
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let groups = n_heads / n_kv_heads;
    for head in 0..n_heads {
        let kv_head = head / groups;
        let mut q_head = vec![0.0; query_rows * head_dim];
        let mut k_transposed = vec![0.0; head_dim * key_rows];
        let mut v_head = vec![0.0; key_rows * head_dim];
        for row in 0..query_rows {
            q_head[row * head_dim..(row + 1) * head_dim].copy_from_slice(
                &query
                    [row * hidden_dim + head * head_dim..row * hidden_dim + (head + 1) * head_dim],
            );
        }
        for row in 0..key_rows {
            let src =
                &keys[row * kv_dim + kv_head * head_dim..row * kv_dim + (kv_head + 1) * head_dim];
            for col in 0..head_dim {
                k_transposed[col * key_rows + row] = src[col];
            }
            v_head[row * head_dim..(row + 1) * head_dim].copy_from_slice(
                &values[row * kv_dim + kv_head * head_dim..row * kv_dim + (kv_head + 1) * head_dim],
            );
        }
        let mut scores = vec![0.0; query_rows * key_rows];
        compute.gemm_f32(
            query_rows,
            key_rows,
            head_dim,
            &q_head,
            &k_transposed,
            None,
            &mut scores,
        )?;
        for row in 0..query_rows {
            for key in 0..key_rows {
                scores[row * key_rows + key] *= scale;
                if causal && key > query_position + row {
                    scores[row * key_rows + key] = f32::NEG_INFINITY;
                }
            }
        }
        let mut probs = vec![0.0; query_rows * key_rows];
        compute.softmax_f32(&scores, &mut probs, query_rows, key_rows)?;
        let mut context = vec![0.0; query_rows * head_dim];
        compute.gemm_f32(
            query_rows,
            head_dim,
            key_rows,
            &probs,
            &v_head,
            None,
            &mut context,
        )?;
        for row in 0..query_rows {
            output[row * hidden_dim + head * head_dim..row * hidden_dim + (head + 1) * head_dim]
                .copy_from_slice(&context[row * head_dim..(row + 1) * head_dim]);
        }
    }
    Ok(())
}

/// Explicit backend selection helper used by VoxCPM callers.
pub fn minicpm4_compute(backend: BackendKind) -> Result<Compute> {
    Compute::for_backend(backend, MINICPM4_HOT_OPS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> MiniCpm4Config {
        MiniCpm4Config::new(
            4,
            8,
            1,
            2,
            1,
            32,
            10_000.0,
            1e-5,
            false,
            vec![1.0; 1],
            vec![1.0; 1],
        )
        .unwrap()
    }

    fn tiny_stack() -> MiniCpm4Stack {
        let config = tiny_config();
        let block = MiniCpm4BlockWeights::from_source(
            &config,
            vec![1.0; 4],
            vec![1.0; 4],
            vec![0.1; 16],
            vec![0.1; 8],
            vec![0.1; 8],
            vec![0.1; 16],
            vec![0.1; 32],
            vec![0.1; 32],
            vec![0.1; 32],
        )
        .unwrap();
        MiniCpm4Stack::new(
            config,
            MiniCpm4StackWeights {
                blocks: vec![block],
                final_norm: vec![1.0; 4],
            },
        )
        .unwrap()
    }

    #[test]
    fn config_rejects_invalid_gqa_and_pins_longrope_shape() {
        assert!(
            MiniCpm4Config::new(
                4,
                8,
                1,
                3,
                2,
                32,
                10_000.0,
                1e-5,
                false,
                vec![1.0],
                vec![1.0],
            )
            .is_err()
        );
        assert_eq!(
            MiniCpm4Config::voxcpm_0_5b()
                .unwrap()
                .rope_short_factor
                .len(),
            32
        );
    }

    #[test]
    fn linear_matches_scalar_cpu_oracle() {
        let linear = MiniCpm4Linear::from_source(vec![1.0, 2.0, 3.0, 4.0], None, 2, 2).unwrap();
        let mut output = [0.0; 2];
        linear
            .apply(&Compute::cpu(), &[5.0, 6.0], 1, &mut output)
            .unwrap();
        assert_eq!(output, [17.0, 39.0]);
    }

    #[test]
    fn full_sequence_equals_one_token_kv_steps() {
        let stack = tiny_stack();
        let compute = Compute::cpu();
        let input = [0.2, -0.1, 0.5, 0.7, -0.3, 0.4, 0.8, -0.2];
        let full = stack.forward(&input, 2, true, &compute).unwrap();
        let mut cache = MiniCpm4KvCache::new(&stack);
        let first = stack
            .forward_step(&input[..4], &mut cache, &compute)
            .unwrap();
        let second = stack
            .forward_step(&input[4..], &mut cache, &compute)
            .unwrap();
        assert_eq!(&full[..4], &first[..]);
        for (left, right) in full[4..].iter().zip(second) {
            assert!((left - right).abs() < 1e-6, "{left} != {right}");
        }
    }

    #[test]
    fn cached_prefill_advances_position_and_reject_rolls_back() {
        let stack = tiny_stack();
        let compute = Compute::cpu();
        let mut cache = MiniCpm4KvCache::new(&stack);
        stack
            .forward_cached(
                &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                2,
                true,
                &mut cache,
                &compute,
            )
            .unwrap();
        assert_eq!(cache.positions(), 2);
        assert!(
            stack
                .forward_cached(&[0.1, 0.2], 1, true, &mut cache, &compute)
                .is_err()
        );
        assert_eq!(cache.positions(), 2);
        stack
            .forward_cached(&[0.9, 1.0, 1.1, 1.2], 1, true, &mut cache, &compute)
            .unwrap();
        assert_eq!(cache.positions(), 3);
    }

    fn scalar_linear(linear: &MiniCpm4Linear, input: &[f32], rows: usize) -> Vec<f32> {
        let mut output = vec![0.0; rows * linear.out_features];
        for row in 0..rows {
            for out in 0..linear.out_features {
                let mut value = linear.bias.as_ref().map_or(0.0, |bias| bias[out]);
                for input_index in 0..linear.in_features {
                    value += input[row * linear.in_features + input_index]
                        * linear.w_t[input_index * linear.out_features + out];
                }
                output[row * linear.out_features + out] = value;
            }
        }
        output
    }

    fn scalar_rms(input: &[f32], rows: usize, cols: usize, gamma: &[f32], eps: f32) -> Vec<f32> {
        let mut output = vec![0.0; rows * cols];
        for row in 0..rows {
            let src = &input[row * cols..(row + 1) * cols];
            let mean_square = src.iter().map(|x| x * x).sum::<f32>() / cols as f32;
            let inverse = (mean_square + eps).sqrt().recip();
            for col in 0..cols {
                output[row * cols + col] = src[col] * inverse * gamma[col];
            }
        }
        output
    }

    /// Independent scalar implementation of one tiny source block.  This is
    /// intentionally separate from `attention`, `apply_rope`, and the Compute
    /// path so a shared bug cannot make the parity property pass.
    fn scalar_reference(stack: &MiniCpm4Stack, input: &[f32], rows: usize) -> Vec<f32> {
        let config = &stack.config;
        let block = &stack.weights.blocks[0];
        let d = config.hidden_dim;
        let kv_d = config.kv_dim();
        let hd = config.head_dim();
        let mut h = input.to_vec();
        let norm = scalar_rms(&h, rows, d, &block.input_layernorm, config.rms_norm_eps);
        let mut q = scalar_linear(&block.q_proj, &norm, rows);
        let mut k = scalar_linear(&block.k_proj, &norm, rows);
        let v = scalar_linear(&block.v_proj, &norm, rows);
        for row in 0..rows {
            for head in 0..config.n_heads {
                let q_start = row * d + head * hd;
                let k_start = row * kv_d + (head / (config.n_heads / config.n_kv_heads)) * hd;
                for pair in 0..hd / 2 {
                    let angle = row as f32
                        * config.rope_theta.powf(-(2.0 * pair as f32) / hd as f32)
                        / config.rope_short_factor[pair];
                    let (sin, cos) = angle.sin_cos();
                    let q_left = q[q_start + pair];
                    let q_right = q[q_start + hd / 2 + pair];
                    q[q_start + pair] = q_left * cos - q_right * sin;
                    q[q_start + hd / 2 + pair] = q_left * sin + q_right * cos;
                    if head < config.n_kv_heads {
                        let k_left = k[k_start + pair];
                        let k_right = k[k_start + hd / 2 + pair];
                        k[k_start + pair] = k_left * cos - k_right * sin;
                        k[k_start + hd / 2 + pair] = k_left * sin + k_right * cos;
                    }
                }
            }
        }
        let mut attended = vec![0.0; rows * d];
        for row in 0..rows {
            for head in 0..config.n_heads {
                let kv_head = head / (config.n_heads / config.n_kv_heads);
                let mut scores = vec![f32::NEG_INFINITY; rows];
                for key_row in 0..=row {
                    let mut score = 0.0;
                    for col in 0..hd {
                        score +=
                            q[row * d + head * hd + col] * k[key_row * kv_d + kv_head * hd + col];
                    }
                    scores[key_row] = score / (hd as f32).sqrt();
                }
                let max = scores[row];
                let denominator = scores[..=row]
                    .iter()
                    .map(|score| (*score - max).exp())
                    .sum::<f32>();
                for col in 0..hd {
                    attended[row * d + head * hd + col] = (0..=row)
                        .map(|key_row| {
                            let probability = (scores[key_row] - max).exp() / denominator;
                            probability * v[key_row * kv_d + kv_head * hd + col]
                        })
                        .sum();
                }
            }
        }
        let projected = scalar_linear(&block.o_proj, &attended, rows);
        for (value, residual) in h.iter_mut().zip(projected) {
            *value += residual;
        }
        let norm = scalar_rms(
            &h,
            rows,
            d,
            &block.post_attention_layernorm,
            config.rms_norm_eps,
        );
        let gate = scalar_linear(&block.gate_proj, &norm, rows);
        let up = scalar_linear(&block.up_proj, &norm, rows);
        let activated: Vec<f32> = gate
            .iter()
            .zip(up)
            .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
            .collect();
        let down = scalar_linear(&block.down_proj, &activated, rows);
        for (value, residual) in h.iter_mut().zip(down) {
            *value += residual;
        }
        scalar_rms(&h, rows, d, &stack.weights.final_norm, config.rms_norm_eps)
    }

    #[test]
    fn full_block_matches_independent_scalar_attention_rope_and_mlp() {
        let stack = tiny_stack();
        let input = [0.2, -0.1, 0.5, 0.7, -0.3, 0.4, 0.8, -0.2];
        let expected = scalar_reference(&stack, &input, 2);
        let actual = stack.forward(&input, 2, true, &Compute::cpu()).unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn rope_uses_split_halves_not_adjacent_pairs() {
        let config = MiniCpm4Config::new(
            4,
            8,
            1,
            1,
            1,
            16,
            1.0,
            1e-5,
            false,
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        )
        .unwrap();
        let mut actual = vec![1.0, 2.0, 3.0, 4.0];
        apply_rope(&mut actual, 1, 1, 4, &config, false);
        let (sin, cos) = 1.0f32.sin_cos();
        let expected = [
            cos - 3.0 * sin,
            2.0 * cos - 4.0 * sin,
            sin + 3.0 * cos,
            2.0 * sin + 4.0 * cos,
        ];
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
        let adjacent = [
            cos - 2.0 * sin,
            sin + 2.0 * cos,
            3.0 * cos - 4.0 * sin,
            3.0 * sin + 4.0 * cos,
        ];
        assert!(
            actual
                .iter()
                .zip(adjacent)
                .any(|(actual, adjacent)| (actual - adjacent).abs() > 1e-3)
        );
    }

    #[test]
    fn longrope_mode_is_selected_from_effective_length() {
        let config = MiniCpm4Config::new_with_original_max_positions(
            4,
            8,
            1,
            2,
            1,
            64,
            2,
            10_000.0,
            1e-5,
            false,
            vec![1.0],
            vec![2.0],
        )
        .unwrap();
        let expected_scale = (1.0 + (32.0f32).ln() / (2.0f32).ln()).sqrt();
        assert!((config.rope_scale - expected_scale).abs() < 1e-6);
        let block = MiniCpm4BlockWeights::from_source(
            &config,
            vec![1.0; 4],
            vec![1.0; 4],
            vec![0.1; 16],
            vec![0.1; 8],
            vec![0.1; 8],
            vec![0.1; 16],
            vec![0.1; 32],
            vec![0.1; 32],
            vec![0.1; 32],
        )
        .unwrap();
        let stack = MiniCpm4Stack::new(
            config,
            MiniCpm4StackWeights {
                blocks: vec![block],
                final_norm: vec![1.0; 4],
            },
        )
        .unwrap();
        assert!(MiniCpm4KvCache::new(&stack).use_long_rope);
        assert!(
            !MiniCpm4KvCache::with_sequence_len(&stack, 1)
                .unwrap()
                .use_long_rope
        );
    }
}
