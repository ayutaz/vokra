//! Plain-slice weight view for one **pre-norm transformer encoder block**
//! (FR-EX-05 device-residency support).
//!
//! [`PrenormLayer`] is a borrowed, model-agnostic description of the weights of a
//! single `h += attn(ln(h)); h += mlp(ln(h))` block: the two LayerNorms, the four
//! attention projections (`q` / `k` / `v` / `out`, each `d → d`, biases optional —
//! Whisper's `k` has none) and the two MLP linears (`fc1: d → ffn`,
//! `fc2: ffn → d`). Weights are row-major and pre-transposed to the `[in, out]`
//! layout the row-major GEMM consumes (the same layout `vokra-models`' `Linear`
//! stores).
//!
//! It exists so a GPU backend can run the **whole** encoder device-resident in one
//! submission: `vokra-models`' `whisper::encoder` builds a `&[PrenormLayer]` (one
//! per block, off the zero-alloc hot region) and hands it to a backend's fused
//! `encode_prenorm_stack`, which keeps `h` and every intermediate on the GPU
//! across all blocks — collapsing the per-op path's `6·N + 1` command-buffer syncs
//! to one. This type is the seam that lets the plain-slice `Compute` boundary carry
//! a whole block's weights without any backend-specific handle crossing into the
//! model layer.
//!
//! Both `vokra-backend-metal` and `vokra-backend-cuda` depend on `vokra-core`
//! only, so this shared view lives here rather than in either backend.

/// Borrowed row-major weight slices for one pre-norm transformer block.
///
/// All projections are `d → d` (`q_w` / `k_w` / `v_w` / `out_w` are `[d, d]`),
/// `fc1_w` is `[d, ffn]` and `fc2_w` is `[ffn, d]`, each pre-transposed to
/// `[in, out]`. LayerNorm `gamma` / `beta` are length `d`. A `None` bias is an
/// absent bias (Whisper's `k_proj`), which the backend treats as a zero bias
/// (`has_bias = 0`), exactly as the per-op path does.
#[derive(Clone, Copy, Debug)]
pub struct PrenormLayer<'a> {
    /// Pre-attention LayerNorm scale γ, length `d`.
    pub attn_ln_gamma: &'a [f32],
    /// Pre-attention LayerNorm shift β, length `d`.
    pub attn_ln_beta: &'a [f32],
    /// Query projection weight `[d, d]`.
    pub q_w: &'a [f32],
    /// Query projection bias `[d]` (Whisper: present).
    pub q_bias: Option<&'a [f32]>,
    /// Key projection weight `[d, d]`.
    pub k_w: &'a [f32],
    /// Key projection bias `[d]` (Whisper: `None`).
    pub k_bias: Option<&'a [f32]>,
    /// Value projection weight `[d, d]`.
    pub v_w: &'a [f32],
    /// Value projection bias `[d]` (Whisper: present).
    pub v_bias: Option<&'a [f32]>,
    /// Output projection weight `[d, d]`.
    pub out_w: &'a [f32],
    /// Output projection bias `[d]` (Whisper: present).
    pub out_bias: Option<&'a [f32]>,
    /// Pre-MLP LayerNorm scale γ, length `d`.
    pub mlp_ln_gamma: &'a [f32],
    /// Pre-MLP LayerNorm shift β, length `d`.
    pub mlp_ln_beta: &'a [f32],
    /// MLP up-projection weight `[d, ffn]`.
    pub fc1_w: &'a [f32],
    /// MLP up-projection bias `[ffn]`.
    pub fc1_bias: Option<&'a [f32]>,
    /// MLP down-projection weight `[ffn, d]`.
    pub fc2_w: &'a [f32],
    /// MLP down-projection bias `[d]`.
    pub fc2_bias: Option<&'a [f32]>,
}

/// Borrowed row-major weight slices for one **pre-norm transformer *decoder*
/// block** (causal self-attention → cross-attention → MLP) — the decoder sibling
/// of [`PrenormLayer`], for the device-resident autoregressive decode step
/// (FR-EX-02 / FR-EX-05).
///
/// A decoder block is `h += self_attn(ln(h)); h += cross_attn(ln(h)); h +=
/// mlp(ln(h))`. Unlike the encoder block, the self-attention is **causal** (masks
/// future keys) and grows a per-layer key/value cache, and the cross-attention
/// reads the encoder output — whose keys/values are the **same** every step, so
/// they are supplied here **already projected** (`cross_k` / `cross_v`, each
/// `[n_ctx, d]`) rather than as `k`/`v` weight matrices. All slices are row-major;
/// weight matrices are pre-transposed to `[in, out]` (the layout the row-major
/// GEMM consumes), exactly like [`PrenormLayer`]. A `None` bias is an absent bias
/// (Whisper's `k_proj`), treated as a zero bias.
///
/// Like [`PrenormLayer`], this is the plain-slice seam that lets a GPU backend
/// keep the whole decode step device-resident (weights uploaded once, the
/// self-attention K/V cache kept on the GPU and appended each step, the
/// cross-attention K/V uploaded once) without any backend-specific handle
/// crossing into the model layer.
#[derive(Clone, Copy, Debug)]
pub struct DecoderLayerView<'a> {
    /// Pre-self-attention LayerNorm scale γ, length `d`.
    pub self_ln_gamma: &'a [f32],
    /// Pre-self-attention LayerNorm shift β, length `d`.
    pub self_ln_beta: &'a [f32],
    /// Causal self-attention query projection weight `[d, d]`.
    pub self_q_w: &'a [f32],
    /// Causal self-attention query bias `[d]` (Whisper: present).
    pub self_q_bias: Option<&'a [f32]>,
    /// Causal self-attention key projection weight `[d, d]`.
    pub self_k_w: &'a [f32],
    /// Causal self-attention key bias `[d]` (Whisper: `None`).
    pub self_k_bias: Option<&'a [f32]>,
    /// Causal self-attention value projection weight `[d, d]`.
    pub self_v_w: &'a [f32],
    /// Causal self-attention value bias `[d]` (Whisper: present).
    pub self_v_bias: Option<&'a [f32]>,
    /// Causal self-attention output projection weight `[d, d]`.
    pub self_out_w: &'a [f32],
    /// Causal self-attention output bias `[d]` (Whisper: present).
    pub self_out_bias: Option<&'a [f32]>,
    /// Pre-cross-attention LayerNorm scale γ, length `d`.
    pub cross_ln_gamma: &'a [f32],
    /// Pre-cross-attention LayerNorm shift β, length `d`.
    pub cross_ln_beta: &'a [f32],
    /// Cross-attention query projection weight `[d, d]`.
    pub cross_q_w: &'a [f32],
    /// Cross-attention query bias `[d]` (Whisper: present).
    pub cross_q_bias: Option<&'a [f32]>,
    /// Cross-attention output projection weight `[d, d]`.
    pub cross_out_w: &'a [f32],
    /// Cross-attention output bias `[d]` (Whisper: present).
    pub cross_out_bias: Option<&'a [f32]>,
    /// Pre-projected cross-attention keys `[n_ctx, d]` (computed once from the
    /// encoder output; identical for every decode step).
    pub cross_k: &'a [f32],
    /// Pre-projected cross-attention values `[n_ctx, d]`.
    pub cross_v: &'a [f32],
    /// Pre-MLP LayerNorm scale γ, length `d`.
    pub mlp_ln_gamma: &'a [f32],
    /// Pre-MLP LayerNorm shift β, length `d`.
    pub mlp_ln_beta: &'a [f32],
    /// MLP up-projection weight `[d, ffn]`.
    pub fc1_w: &'a [f32],
    /// MLP up-projection bias `[ffn]`.
    pub fc1_bias: Option<&'a [f32]>,
    /// MLP down-projection weight `[ffn, d]`.
    pub fc2_w: &'a [f32],
    /// MLP down-projection bias `[d]`.
    pub fc2_bias: Option<&'a [f32]>,
}
