//! Mimi neural-chain building blocks: causal streaming conv /
//! conv-transpose and the bottleneck transformer (M4-05-T12/T32 shared
//! primitives).
//!
//! # Streaming contract (FR-LD-06 / NFR-QL-05 discipline)
//!
//! Every block carries its boundary state in an explicit `*State` value
//! (1:1 preserved, hidden from callers of the encoder/decoder facades —
//! the Silero-VAD posture: no silent internal resets). The **defining
//! property** is: processing one long buffer equals processing the same
//! samples in chunks with the state carried over, bit-for-bit — because a
//! causal conv's left context is exactly what the state holds and the
//! initial state is the constant (zero) pad the upstream
//! `StreamingConv1d(pad_mode="constant")` applies. The T33 test pins this.
//!
//! # Hot ops through the Compute seam
//!
//! Convolutions run as im2col gather + one `Compute::gemm_f32` per chunk
//! (the conv1d seam kernel has no dilation attribute, so the gather
//! handles dilation and the GEMM carries the FLOPs — the dominant cost
//! either way). The transformer reuses GEMM / softmax / LayerNorm / GELU
//! seam ops plus the adjacent-pair RoPE from `crate::csm::rope` (plain
//! frequencies at `max_period` — no Llama-3 scaling here).
//!
//! # Upstream anchors (ADR M4-05 §D2 — transcribed)
//!
//! `kyutai-labs/moshi` `moshi/modules/seanet.py` (block structure),
//! `resample.py` (`ConvDownsample1d` / `ConvTrUpsample1d`: causal,
//! `kernel_size = 2 * stride`, `bias=False`), `loaders.py`
//! (`_transformer_kwargs`: pre-norm LayerNorm, RoPE `max_period=10000`,
//! `layer_scale=0.01`, `gating="none"` = plain GELU MLP, `context=250`).
//! Linear-bias presence and the transformer's final-norm arrangement are
//! **T29 checkpoint-confirmed** items; the synthesized fixtures use
//! bias-less linears and no final norm, documented here so the real
//! binding can adjust shape-driven without an API break.

use vokra_core::{Result, VokraError};

use crate::compute::Compute;
use crate::csm::rope::{llama3_inv_freqs, rope_apply_adjacent};

/// ELU (α = 1) in place — the SEANet activation (`_seanet_kwargs`).
pub(crate) fn elu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = v.exp_m1();
        }
    }
}

/// Causal left-pad fill mode (upstream `StreamingConv1d.pad_mode`,
/// `moshi/modules/conv.py` — only these two values are asserted upstream).
///
/// - [`PadMode::Constant`]: zero left context (the SEANet convs —
///   `_seanet_kwargs["pad_mode"] = "constant"`).
/// - [`PadMode::Replicate`]: the **first** processed chunk fills the left
///   context with its own first input column (`init = x[..., :1]`;
///   `state.first` then flips) — the frame-resample
///   `ConvDownsample1d(pad_mode="replicate")` (`resample.py`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum PadMode {
    #[default]
    Constant,
    Replicate,
}

// ---------------------------------------------------------------------------
// Causal streaming Conv1d
// ---------------------------------------------------------------------------

/// A causal streaming 1-D convolution (upstream `StreamingConv1d`,
/// `pad_mode="constant"`): left pad `k_eff - stride` carried as state.
///
/// Weight layout: `[out_ch, in_ch * k]` row-major (the natural
/// `[out_ch, in_ch, k]` flattening); the GEMM path uses a pre-transposed
/// copy built at construction.
#[derive(Debug, Clone)]
pub(crate) struct CausalConv1d {
    pub(crate) in_ch: usize,
    pub(crate) out_ch: usize,
    pub(crate) k: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
    /// Left-context fill (module docs; `Constant` unless the block
    /// transcribes an upstream `pad_mode="replicate"` conv).
    pad_mode: PadMode,
    /// `[in_ch * k, out_ch]` — transposed for `gemm_f32`.
    w_t: Vec<f32>,
    bias: Option<Vec<f32>>,
}

/// Boundary state + scratch for one [`CausalConv1d`].
#[derive(Debug, Clone)]
pub(crate) struct ConvState {
    /// Left context `[in_ch, pad]` (zero-initialised = constant pad).
    hist: Vec<f32>,
    /// `true` until the first chunk is processed (drives the
    /// [`PadMode::Replicate`] first-call fill — upstream
    /// `_StreamingConv1dState.first`).
    first: bool,
    /// Assembled `[in_ch, pad + t_cap]` context.
    ctx: Vec<f32>,
    /// im2col gather `[n_out_cap, in_ch * k]`.
    gather: Vec<f32>,
    /// GEMM result `[n_out_cap, out_ch]`.
    gemm_out: Vec<f32>,
    t_cap: usize,
}

impl CausalConv1d {
    /// Builds the block from a `[out_ch, in_ch, k]` row-major weight.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on shape mismatch / zero dims /
    /// `k_eff < stride`.
    pub(crate) fn new(
        in_ch: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        dilation: usize,
        w: &[f32],
        bias: Option<Vec<f32>>,
    ) -> Result<Self> {
        if in_ch == 0 || out_ch == 0 || k == 0 || stride == 0 || dilation == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "mimi conv1d: zero dim (in={in_ch}, out={out_ch}, k={k}, stride={stride}, \
                 dilation={dilation})"
            )));
        }
        if w.len() != out_ch * in_ch * k {
            return Err(VokraError::InvalidArgument(format!(
                "mimi conv1d: weight len {} != out*in*k {}",
                w.len(),
                out_ch * in_ch * k
            )));
        }
        let k_eff = dilation * (k - 1) + 1;
        if k_eff < stride {
            return Err(VokraError::InvalidArgument(format!(
                "mimi conv1d: effective kernel {k_eff} < stride {stride} (causal left-pad \
                 would be negative)"
            )));
        }
        if let Some(b) = &bias {
            if b.len() != out_ch {
                return Err(VokraError::InvalidArgument(format!(
                    "mimi conv1d: bias len {} != out_ch {out_ch}",
                    b.len()
                )));
            }
        }
        // Transpose [out_ch, in_ch*k] → [in_ch*k, out_ch] once.
        let cols = in_ch * k;
        let mut w_t = vec![0.0f32; cols * out_ch];
        for o in 0..out_ch {
            for c in 0..cols {
                w_t[c * out_ch + o] = w[o * cols + c];
            }
        }
        Ok(Self {
            in_ch,
            out_ch,
            k,
            stride,
            dilation,
            pad_mode: PadMode::Constant,
            w_t,
            bias,
        })
    }

    /// Selects the left-context fill mode (builder — default
    /// [`PadMode::Constant`]).
    #[must_use]
    pub(crate) fn with_pad_mode(mut self, pad_mode: PadMode) -> Self {
        self.pad_mode = pad_mode;
        self
    }

    /// Causal left pad (`k_eff - stride`) — the state width.
    pub(crate) fn pad(&self) -> usize {
        self.dilation * (self.k - 1) + 1 - self.stride
    }

    /// Fresh zero state with scratch capacity for `t_cap` input samples.
    pub(crate) fn state(&self, t_cap: usize) -> ConvState {
        let pad = self.pad();
        let n_out_cap = t_cap / self.stride;
        ConvState {
            hist: vec![0.0; self.in_ch * pad],
            first: true,
            ctx: vec![0.0; self.in_ch * (pad + t_cap)],
            gather: vec![0.0; n_out_cap * self.in_ch * self.k],
            gemm_out: vec![0.0; n_out_cap * self.out_ch],
            t_cap,
        }
    }

    /// Output columns for `t` input columns (`t % stride == 0` required).
    pub(crate) fn out_len(&self, t: usize) -> usize {
        t / self.stride
    }

    /// Reconstructs the `[out_ch, in_ch, k]` row-major weight the
    /// constructor consumed (inverse of the internal `[in*k, out]`
    /// transpose). Used by the neural-chain GGUF round-trip.
    #[cfg(test)]
    pub(crate) fn weight_oik(&self) -> Vec<f32> {
        let cols = self.in_ch * self.k;
        let mut w = vec![0.0f32; self.out_ch * cols];
        for o in 0..self.out_ch {
            for c in 0..cols {
                w[o * cols + c] = self.w_t[c * self.out_ch + o];
            }
        }
        w
    }

    /// Optional bias `[out_ch]`.
    #[cfg(test)]
    pub(crate) fn bias(&self) -> Option<&[f32]> {
        self.bias.as_deref()
    }

    /// Processes `x = [in_ch, t]` channel-major into
    /// `out = [out_ch, t/stride]`, carrying the causal left context in
    /// `state`. Zero heap allocation (scratch pre-sized by
    /// [`Self::state`]).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on `t % stride != 0`, capacity
    /// overflow, or shape mismatch; propagates seam errors.
    pub(crate) fn process_into(
        &self,
        compute: &Compute,
        state: &mut ConvState,
        x: &[f32],
        t: usize,
        out: &mut [f32],
    ) -> Result<()> {
        if t == 0 || t % self.stride != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "mimi conv1d: t {t} must be a positive multiple of stride {}",
                self.stride
            )));
        }
        if t > state.t_cap {
            return Err(VokraError::InvalidArgument(format!(
                "mimi conv1d: t {t} > state capacity {}",
                state.t_cap
            )));
        }
        if x.len() != self.in_ch * t {
            return Err(VokraError::InvalidArgument(format!(
                "mimi conv1d: x len {} != in_ch*t {}",
                x.len(),
                self.in_ch * t
            )));
        }
        let n_out = self.out_len(t);
        if out.len() != self.out_ch * n_out {
            return Err(VokraError::InvalidArgument(format!(
                "mimi conv1d: out len {} != out_ch*n_out {}",
                out.len(),
                self.out_ch * n_out
            )));
        }
        let pad = self.pad();
        if pad > 0 && self.pad_mode == PadMode::Replicate {
            // Upstream conv.py replicate mode: needs at least `pad`
            // fresh columns ("Not enough content to pad streaming.").
            if t < pad {
                return Err(VokraError::InvalidArgument(format!(
                    "mimi conv1d: replicate pad needs t ({t}) >= pad ({pad}) \
                     (upstream StreamingConv1d assertion)"
                )));
            }
            if state.first {
                // First chunk: left context replicates this chunk's first
                // column (`init = x[..., :1]` — conv.py).
                for ci in 0..self.in_ch {
                    let v = x[ci * t];
                    state.hist[ci * pad..(ci + 1) * pad].fill(v);
                }
            }
        }
        state.first = false;
        let ctx_len = pad + t;
        // ctx[ci, :pad] = hist, ctx[ci, pad:] = x.
        for ci in 0..self.in_ch {
            state.ctx[ci * ctx_len..ci * ctx_len + pad]
                .copy_from_slice(&state.hist[ci * pad..(ci + 1) * pad]);
            state.ctx[ci * ctx_len + pad..ci * ctx_len + ctx_len]
                .copy_from_slice(&x[ci * t..(ci + 1) * t]);
        }
        // im2col gather (dilation-aware).
        let cols = self.in_ch * self.k;
        for i in 0..n_out {
            let row = &mut state.gather[i * cols..(i + 1) * cols];
            for ci in 0..self.in_ch {
                for j in 0..self.k {
                    row[ci * self.k + j] =
                        state.ctx[ci * ctx_len + i * self.stride + j * self.dilation];
                }
            }
        }
        // One GEMM: [n_out, cols] × [cols, out_ch] → [n_out, out_ch].
        compute.gemm_f32(
            n_out,
            self.out_ch,
            cols,
            &state.gather[..n_out * cols],
            &self.w_t,
            None,
            &mut state.gemm_out[..n_out * self.out_ch],
        )?;
        // Transpose to channel-major + bias.
        for o in 0..self.out_ch {
            let b = self.bias.as_ref().map_or(0.0, |b| b[o]);
            for i in 0..n_out {
                out[o * n_out + i] = state.gemm_out[i * self.out_ch + o] + b;
            }
        }
        // New hist = last `pad` context columns.
        for ci in 0..self.in_ch {
            state.hist[ci * pad..(ci + 1) * pad]
                .copy_from_slice(&state.ctx[ci * ctx_len + ctx_len - pad..(ci + 1) * ctx_len]);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Causal streaming ConvTranspose1d
// ---------------------------------------------------------------------------

/// A causal streaming transposed 1-D convolution (upstream
/// `StreamingConvTranspose1d` with full right trim — `trim_right_ratio`
/// causal semantics): each input column contributes `k` samples starting
/// at `t*stride`; the `k - stride` overlap runs ahead as carried state and
/// exactly `t*stride` samples are emitted.
#[derive(Debug, Clone)]
pub(crate) struct CausalConvTranspose1d {
    pub(crate) in_ch: usize,
    pub(crate) out_ch: usize,
    pub(crate) k: usize,
    pub(crate) stride: usize,
    /// `[in_ch, out_ch * k]` row-major (the natural `[in_ch, out_ch, k]`
    /// flattening — already GEMM-shaped).
    w: Vec<f32>,
    bias: Option<Vec<f32>>,
}

/// Boundary state + scratch for one [`CausalConvTranspose1d`].
#[derive(Debug, Clone)]
pub(crate) struct ConvTrState {
    /// Overlap tail `[out_ch, k - stride]` partial sums.
    tail: Vec<f32>,
    /// Input transposed `[t_cap, in_ch]`.
    x_rows: Vec<f32>,
    /// GEMM result `[t_cap, out_ch * k]`.
    contrib: Vec<f32>,
    t_cap: usize,
}

impl CausalConvTranspose1d {
    /// Builds from a `[in_ch, out_ch, k]` row-major weight.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on shape mismatch / zero dims /
    /// `k < stride`.
    pub(crate) fn new(
        in_ch: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        w: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> Result<Self> {
        if in_ch == 0 || out_ch == 0 || k == 0 || stride == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "mimi convtr1d: zero dim (in={in_ch}, out={out_ch}, k={k}, stride={stride})"
            )));
        }
        if k < stride {
            return Err(VokraError::InvalidArgument(format!(
                "mimi convtr1d: kernel {k} < stride {stride} (output would have holes)"
            )));
        }
        if w.len() != in_ch * out_ch * k {
            return Err(VokraError::InvalidArgument(format!(
                "mimi convtr1d: weight len {} != in*out*k {}",
                w.len(),
                in_ch * out_ch * k
            )));
        }
        if let Some(b) = &bias {
            if b.len() != out_ch {
                return Err(VokraError::InvalidArgument(format!(
                    "mimi convtr1d: bias len {} != out_ch {out_ch}",
                    b.len()
                )));
            }
        }
        Ok(Self {
            in_ch,
            out_ch,
            k,
            stride,
            w,
            bias,
        })
    }

    /// The `[in_ch, out_ch, k]` row-major weight (stored verbatim). Used by
    /// the neural-chain GGUF round-trip.
    #[cfg(test)]
    pub(crate) fn weight_iok(&self) -> &[f32] {
        &self.w
    }

    /// Optional bias `[out_ch]`.
    #[cfg(test)]
    pub(crate) fn bias(&self) -> Option<&[f32]> {
        self.bias.as_deref()
    }

    /// Fresh zero state with scratch capacity for `t_cap` input columns.
    pub(crate) fn state(&self, t_cap: usize) -> ConvTrState {
        ConvTrState {
            tail: vec![0.0; self.out_ch * (self.k - self.stride)],
            x_rows: vec![0.0; t_cap * self.in_ch],
            contrib: vec![0.0; t_cap * self.out_ch * self.k],
            t_cap,
        }
    }

    /// Processes `x = [in_ch, t]` into `out = [out_ch, t * stride]`,
    /// carrying the overlap tail. Zero heap allocation.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on shape/capacity mismatch;
    /// propagates seam errors.
    pub(crate) fn process_into(
        &self,
        compute: &Compute,
        state: &mut ConvTrState,
        x: &[f32],
        t: usize,
        out: &mut [f32],
    ) -> Result<()> {
        if t == 0 || t > state.t_cap {
            return Err(VokraError::InvalidArgument(format!(
                "mimi convtr1d: t {t} out of range (cap {})",
                state.t_cap
            )));
        }
        if x.len() != self.in_ch * t {
            return Err(VokraError::InvalidArgument(format!(
                "mimi convtr1d: x len {} != in_ch*t {}",
                x.len(),
                self.in_ch * t
            )));
        }
        let n_out = t * self.stride;
        if out.len() != self.out_ch * n_out {
            return Err(VokraError::InvalidArgument(format!(
                "mimi convtr1d: out len {} != out_ch*t*stride {}",
                out.len(),
                self.out_ch * n_out
            )));
        }
        // Transpose input to rows.
        for i in 0..t {
            for ci in 0..self.in_ch {
                state.x_rows[i * self.in_ch + ci] = x[ci * t + i];
            }
        }
        // One GEMM: [t, in_ch] × [in_ch, out_ch*k] → [t, out_ch*k].
        let cols = self.out_ch * self.k;
        compute.gemm_f32(
            t,
            cols,
            self.in_ch,
            &state.x_rows[..t * self.in_ch],
            &self.w,
            None,
            &mut state.contrib[..t * cols],
        )?;
        // Sequential overlap-add with the carried tail.
        let tail_len = self.k - self.stride;
        for i in 0..t {
            let contrib = &state.contrib[i * cols..(i + 1) * cols];
            for o in 0..self.out_ch {
                let b = self.bias.as_ref().map_or(0.0, |b| b[o]);
                let c_row = &contrib[o * self.k..(o + 1) * self.k];
                let tail_row = &mut state.tail[o * tail_len..(o + 1) * tail_len];
                // Emit the first `stride` samples (finalised).
                for s in 0..self.stride {
                    let v = c_row[s] + if s < tail_len { tail_row[s] } else { 0.0 };
                    out[o * n_out + i * self.stride + s] = v + b;
                }
                // Roll the tail: new_tail[j] = full[stride + j].
                for j in 0..tail_len {
                    let carried = if self.stride + j < tail_len {
                        tail_row[self.stride + j]
                    } else {
                        0.0
                    };
                    tail_row[j] = c_row[self.stride + j] + carried;
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Bottleneck transformer (pre-norm LayerNorm + RoPE MHA + LayerScale + GELU
// MLP, rolling causal context window)
// ---------------------------------------------------------------------------

/// One transformer layer's weights (bias-less linears — module docs).
#[derive(Debug, Clone)]
pub(crate) struct MimiTransformerLayer {
    pub(crate) ln1_gamma: Vec<f32>,
    pub(crate) ln1_beta: Vec<f32>,
    pub(crate) q_w_t: Vec<f32>,
    pub(crate) k_w_t: Vec<f32>,
    pub(crate) v_w_t: Vec<f32>,
    pub(crate) o_w_t: Vec<f32>,
    pub(crate) layer_scale_1: Vec<f32>,
    pub(crate) ln2_gamma: Vec<f32>,
    pub(crate) ln2_beta: Vec<f32>,
    pub(crate) fc1_w_t: Vec<f32>,
    pub(crate) fc2_w_t: Vec<f32>,
    pub(crate) layer_scale_2: Vec<f32>,
}

/// The bottleneck transformer (shared by the encoder / neural decoder —
/// each owns its own weights + state).
#[derive(Debug, Clone)]
pub(crate) struct MimiTransformer {
    pub(crate) d: usize,
    pub(crate) n_head: usize,
    pub(crate) ff: usize,
    pub(crate) context: usize,
    pub(crate) layers: Vec<MimiTransformerLayer>,
    inv_freqs: Vec<f32>,
}

/// Rolling KV window + step scratch for one [`MimiTransformer`].
#[derive(Debug, Clone)]
pub(crate) struct MimiTransformerState {
    /// `[n_layer, context, d]` K ring (RoPE already applied at append).
    k_ring: Vec<f32>,
    /// Same layout for V.
    v_ring: Vec<f32>,
    /// Absolute positions seen so far.
    pos: usize,
    h: Vec<f32>,
    norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    rope_buf: Vec<f32>,
    scores: Vec<f32>,
    probs: Vec<f32>,
    attn_out: Vec<f32>,
    attn_o: Vec<f32>,
    ff1: Vec<f32>,
    ff1_act: Vec<f32>,
    ff2: Vec<f32>,
}

impl MimiTransformer {
    /// Builds the transformer; `layers` shapes must match (`d`, `ff`).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on ill-formed dims or any layer
    /// shape mismatch.
    pub(crate) fn new(
        d: usize,
        n_head: usize,
        ff: usize,
        context: usize,
        max_period: usize,
        layers: Vec<MimiTransformerLayer>,
    ) -> Result<Self> {
        if d == 0 || n_head == 0 || d % n_head != 0 || (d / n_head) % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "mimi transformer: d {d} must split into even-width heads (n_head {n_head})"
            )));
        }
        if ff == 0 || context == 0 || max_period == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "mimi transformer: zero hparam (ff={ff}, context={context}, \
                 max_period={max_period})"
            )));
        }
        for (i, l) in layers.iter().enumerate() {
            let checks = [
                ("ln1_gamma", l.ln1_gamma.len(), d),
                ("ln1_beta", l.ln1_beta.len(), d),
                ("q_w_t", l.q_w_t.len(), d * d),
                ("k_w_t", l.k_w_t.len(), d * d),
                ("v_w_t", l.v_w_t.len(), d * d),
                ("o_w_t", l.o_w_t.len(), d * d),
                ("layer_scale_1", l.layer_scale_1.len(), d),
                ("ln2_gamma", l.ln2_gamma.len(), d),
                ("ln2_beta", l.ln2_beta.len(), d),
                ("fc1_w_t", l.fc1_w_t.len(), d * ff),
                ("fc2_w_t", l.fc2_w_t.len(), ff * d),
                ("layer_scale_2", l.layer_scale_2.len(), d),
            ];
            for (name, got, want) in checks {
                if got != want {
                    return Err(VokraError::InvalidArgument(format!(
                        "mimi transformer: layer[{i}].{name} len {got} != {want}"
                    )));
                }
            }
        }
        let head_dim = d / n_head;
        let inv_freqs = llama3_inv_freqs(head_dim, max_period as f32, None)?;
        Ok(Self {
            d,
            n_head,
            ff,
            context,
            layers,
            inv_freqs,
        })
    }

    /// Fresh state (zero KV window, position 0).
    pub(crate) fn state(&self) -> MimiTransformerState {
        let d = self.d;
        MimiTransformerState {
            k_ring: vec![0.0; self.layers.len() * self.context * d],
            v_ring: vec![0.0; self.layers.len() * self.context * d],
            pos: 0,
            h: vec![0.0; d],
            norm: vec![0.0; d],
            q: vec![0.0; d],
            k: vec![0.0; d],
            v: vec![0.0; d],
            rope_buf: vec![0.0; d / self.n_head],
            scores: vec![0.0; self.context],
            probs: vec![0.0; self.context],
            attn_out: vec![0.0; d],
            attn_o: vec![0.0; d],
            ff1: vec![0.0; self.ff],
            ff1_act: vec![0.0; self.ff],
            ff2: vec![0.0; d],
        }
    }

    /// Processes `t` positions in place over `x = [t, d]` row-major
    /// (sequentially — the bottleneck runs at ≤ 25 Hz, so per-position
    /// stepping is the streaming-native shape). Zero heap allocation.
    ///
    /// # Errors
    ///
    /// Shape errors are [`VokraError::InvalidArgument`]; seam errors
    /// propagate.
    pub(crate) fn process_inplace(
        &self,
        compute: &Compute,
        state: &mut MimiTransformerState,
        x: &mut [f32],
        t: usize,
    ) -> Result<()> {
        let d = self.d;
        if x.len() != t * d {
            return Err(VokraError::InvalidArgument(format!(
                "mimi transformer: x len {} != t*d {}",
                x.len(),
                t * d
            )));
        }
        for i in 0..t {
            // Work on one position; write back at the end.
            state.h.copy_from_slice(&x[i * d..(i + 1) * d]);
            self.step(compute, state)?;
            x[i * d..(i + 1) * d].copy_from_slice(&state.h);
        }
        Ok(())
    }

    /// One position through every layer (pre-norm + LayerScale residuals).
    fn step(&self, compute: &Compute, state: &mut MimiTransformerState) -> Result<()> {
        let d = self.d;
        let n_head = self.n_head;
        let head_dim = d / n_head;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let pos = state.pos;
        let ctx = self.context;
        let window = (pos + 1).min(ctx);
        let slot = pos % ctx;
        for (li, layer) in self.layers.iter().enumerate() {
            // ---- Attention sublayer ----
            compute.layer_norm_f32(
                &state.h,
                &mut state.norm,
                1,
                d,
                &layer.ln1_gamma,
                &layer.ln1_beta,
                1e-5,
            )?;
            compute.gemm_f32(1, d, d, &state.norm, &layer.q_w_t, None, &mut state.q)?;
            compute.gemm_f32(1, d, d, &state.norm, &layer.k_w_t, None, &mut state.k)?;
            compute.gemm_f32(1, d, d, &state.norm, &layer.v_w_t, None, &mut state.v)?;
            for h in 0..n_head {
                state
                    .rope_buf
                    .copy_from_slice(&state.q[h * head_dim..(h + 1) * head_dim]);
                rope_apply_adjacent(&mut state.rope_buf, 1, head_dim, &self.inv_freqs, pos)?;
                state.q[h * head_dim..(h + 1) * head_dim].copy_from_slice(&state.rope_buf);
                state
                    .rope_buf
                    .copy_from_slice(&state.k[h * head_dim..(h + 1) * head_dim]);
                rope_apply_adjacent(&mut state.rope_buf, 1, head_dim, &self.inv_freqs, pos)?;
                state.k[h * head_dim..(h + 1) * head_dim].copy_from_slice(&state.rope_buf);
            }
            let ring_base = li * ctx * d;
            state.k_ring[ring_base + slot * d..ring_base + (slot + 1) * d]
                .copy_from_slice(&state.k);
            state.v_ring[ring_base + slot * d..ring_base + (slot + 1) * d]
                .copy_from_slice(&state.v);
            // Attend over the rolling window (absolute order irrelevant to
            // the softmax sum; RoPE was applied at append time).
            for h in 0..n_head {
                let q_row = &state.q[h * head_dim..(h + 1) * head_dim];
                for (w, j) in window_positions(pos, window).enumerate() {
                    let js = j % ctx;
                    let k_row = &state.k_ring[ring_base + js * d + h * head_dim
                        ..ring_base + js * d + (h + 1) * head_dim];
                    let mut s = 0.0f32;
                    for c in 0..head_dim {
                        s += q_row[c] * k_row[c];
                    }
                    state.scores[w] = s * scale;
                }
                compute.softmax_f32(
                    &state.scores[..window],
                    &mut state.probs[..window],
                    1,
                    window,
                )?;
                let out_dst = &mut state.attn_out[h * head_dim..(h + 1) * head_dim];
                for (c, out) in out_dst.iter_mut().enumerate() {
                    let mut sum = 0.0f32;
                    for (w, j) in window_positions(pos, window).enumerate() {
                        let js = j % ctx;
                        sum += state.probs[w] * state.v_ring[ring_base + js * d + h * head_dim + c];
                    }
                    *out = sum;
                }
            }
            compute.gemm_f32(
                1,
                d,
                d,
                &state.attn_out,
                &layer.o_w_t,
                None,
                &mut state.attn_o,
            )?;
            for c in 0..d {
                state.h[c] += layer.layer_scale_1[c] * state.attn_o[c];
            }
            // ---- MLP sublayer (gating = none → GELU MLP) ----
            compute.layer_norm_f32(
                &state.h,
                &mut state.norm,
                1,
                d,
                &layer.ln2_gamma,
                &layer.ln2_beta,
                1e-5,
            )?;
            compute.gemm_f32(
                1,
                self.ff,
                d,
                &state.norm,
                &layer.fc1_w_t,
                None,
                &mut state.ff1,
            )?;
            compute.gelu_f32(&state.ff1, &mut state.ff1_act)?;
            compute.gemm_f32(
                1,
                d,
                self.ff,
                &state.ff1_act,
                &layer.fc2_w_t,
                None,
                &mut state.ff2,
            )?;
            for c in 0..d {
                state.h[c] += layer.layer_scale_2[c] * state.ff2[c];
            }
        }
        state.pos += 1;
        Ok(())
    }
}

/// The absolute positions inside the rolling window, oldest first:
/// `pos + 1 - window ..= pos`.
fn window_positions(pos: usize, window: usize) -> impl Iterator<Item = usize> {
    (pos + 1 - window)..=pos
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::BackendKind;
    use vokra_core::rng::SplitMix64;

    fn compute() -> Compute {
        Compute::for_backend(BackendKind::Cpu, &[]).expect("cpu compute")
    }

    fn rnd(rng: &mut SplitMix64, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.next_unit_f32() * 2.0 - 1.0).collect()
    }

    #[test]
    fn conv_full_buffer_equals_chunked_streaming() {
        let mut rng = SplitMix64::new(1);
        let (in_ch, out_ch, k, stride, dil) = (2, 3, 3, 2, 2);
        let w = rnd(&mut rng, out_ch * in_ch * k);
        let bias = Some(rnd(&mut rng, out_ch));
        let conv = CausalConv1d::new(in_ch, out_ch, k, stride, dil, &w, bias).unwrap();
        let compute = compute();
        let t = 16;
        let x = rnd(&mut rng, in_ch * t);

        let mut full_state = conv.state(t);
        let mut full = vec![0.0f32; out_ch * conv.out_len(t)];
        conv.process_into(&compute, &mut full_state, &x, t, &mut full)
            .unwrap();

        // Chunked: 4 chunks of 4 (channel-major slices per chunk).
        let mut st = conv.state(4);
        let mut got = vec![Vec::new(); out_ch];
        for c in 0..4 {
            let mut chunk = vec![0.0f32; in_ch * 4];
            for ci in 0..in_ch {
                chunk[ci * 4..(ci + 1) * 4]
                    .copy_from_slice(&x[ci * t + c * 4..ci * t + (c + 1) * 4]);
            }
            let n = conv.out_len(4);
            let mut out = vec![0.0f32; out_ch * n];
            conv.process_into(&compute, &mut st, &chunk, 4, &mut out)
                .unwrap();
            for o in 0..out_ch {
                got[o].extend_from_slice(&out[o * n..(o + 1) * n]);
            }
        }
        let n_full = conv.out_len(t);
        for o in 0..out_ch {
            for (i, v) in got[o].iter().enumerate() {
                assert!(
                    (v - full[o * n_full + i]).abs() < 1e-6,
                    "ch {o} col {i}: {v} vs {}",
                    full[o * n_full + i]
                );
            }
        }
    }

    #[test]
    fn convtr_full_buffer_equals_chunked_streaming_and_length() {
        let mut rng = SplitMix64::new(2);
        let (in_ch, out_ch, k, stride) = (3, 2, 4, 2);
        let w = rnd(&mut rng, in_ch * out_ch * k);
        let bias = Some(rnd(&mut rng, out_ch));
        let tr = CausalConvTranspose1d::new(in_ch, out_ch, k, stride, w, bias).unwrap();
        let compute = compute();
        let t = 6;
        let x = rnd(&mut rng, in_ch * t);

        let mut full_state = tr.state(t);
        let mut full = vec![0.0f32; out_ch * t * stride];
        tr.process_into(&compute, &mut full_state, &x, t, &mut full)
            .unwrap();

        let mut st = tr.state(2);
        let mut got = vec![Vec::new(); out_ch];
        for c in 0..3 {
            let mut chunk = vec![0.0f32; in_ch * 2];
            for ci in 0..in_ch {
                chunk[ci * 2..(ci + 1) * 2]
                    .copy_from_slice(&x[ci * t + c * 2..ci * t + (c + 1) * 2]);
            }
            let mut out = vec![0.0f32; out_ch * 2 * stride];
            tr.process_into(&compute, &mut st, &chunk, 2, &mut out)
                .unwrap();
            for o in 0..out_ch {
                got[o].extend_from_slice(&out[o * 2 * stride..(o + 1) * 2 * stride]);
            }
        }
        for o in 0..out_ch {
            assert_eq!(got[o].len(), t * stride);
            for (i, v) in got[o].iter().enumerate() {
                assert!(
                    (v - full[o * t * stride + i]).abs() < 1e-6,
                    "ch {o} col {i}"
                );
            }
        }
    }

    #[test]
    fn convtr_matches_hand_computed_reference() {
        // in_ch=1, out_ch=1, k=3, stride=2, w=[1,2,3], x=[1,1]:
        // full conv-transpose out (len (2-1)*2+3 = 5): [1,2,3+1,2,3] =
        // [1,2,4,2,3]; causal right-trim (k-stride = 1) → [1,2,4,2].
        let tr = CausalConvTranspose1d::new(1, 1, 3, 2, vec![1.0, 2.0, 3.0], None).unwrap();
        let compute = compute();
        let mut st = tr.state(2);
        let mut out = vec![0.0f32; 4];
        tr.process_into(&compute, &mut st, &[1.0, 1.0], 2, &mut out)
            .unwrap();
        assert_eq!(out, vec![1.0, 2.0, 4.0, 2.0]);
    }

    #[test]
    fn conv_replicate_pad_first_call_matches_hand_reference() {
        // in=1, out=1, k=3, stride=1, w=[1,1,1], x=[a,b,c] (pad = 2):
        //   constant : ctx [0,0,a,b,c] → [a, a+b, a+b+c]
        //   replicate: ctx [a,a,a,b,c] → [3a, 2a+b, a+b+c]
        // (upstream conv.py: first chunk's left context = its first column).
        let w = [1.0f32, 1.0, 1.0];
        let compute = compute();
        let (a, b, c) = (2.0f32, 3.0, 5.0);

        let conv_c = CausalConv1d::new(1, 1, 3, 1, 1, &w, None).unwrap();
        let mut st = conv_c.state(3);
        let mut out = vec![0.0f32; 3];
        conv_c
            .process_into(&compute, &mut st, &[a, b, c], 3, &mut out)
            .unwrap();
        assert_eq!(out, vec![a, a + b, a + b + c]);

        let conv_r = CausalConv1d::new(1, 1, 3, 1, 1, &w, None)
            .unwrap()
            .with_pad_mode(PadMode::Replicate);
        let mut st = conv_r.state(3);
        conv_r
            .process_into(&compute, &mut st, &[a, b, c], 3, &mut out)
            .unwrap();
        assert_eq!(out, vec![3.0 * a, 2.0 * a + b, a + b + c]);
    }

    #[test]
    fn conv_replicate_full_buffer_equals_chunked_streaming() {
        // The replicate fill happens exactly once (first chunk) — carried
        // state must reproduce the full-buffer result bit-for-bit.
        let mut rng = SplitMix64::new(7);
        let (in_ch, out_ch, k, stride) = (2, 3, 4, 2);
        let w = rnd(&mut rng, out_ch * in_ch * k);
        let conv = CausalConv1d::new(in_ch, out_ch, k, stride, 1, &w, None)
            .unwrap()
            .with_pad_mode(PadMode::Replicate);
        let compute = compute();
        let t = 8;
        let x = rnd(&mut rng, in_ch * t);

        let mut full_state = conv.state(t);
        let mut full = vec![0.0f32; out_ch * conv.out_len(t)];
        conv.process_into(&compute, &mut full_state, &x, t, &mut full)
            .unwrap();

        let mut st = conv.state(2);
        let mut got = vec![Vec::new(); out_ch];
        for c in 0..4 {
            let mut chunk = vec![0.0f32; in_ch * 2];
            for ci in 0..in_ch {
                chunk[ci * 2..(ci + 1) * 2]
                    .copy_from_slice(&x[ci * t + c * 2..ci * t + (c + 1) * 2]);
            }
            let n = conv.out_len(2);
            let mut out = vec![0.0f32; out_ch * n];
            conv.process_into(&compute, &mut st, &chunk, 2, &mut out)
                .unwrap();
            for o in 0..out_ch {
                got[o].extend_from_slice(&out[o * n..(o + 1) * n]);
            }
        }
        let n_full = conv.out_len(t);
        for o in 0..out_ch {
            for (i, v) in got[o].iter().enumerate() {
                assert_eq!(
                    *v,
                    full[o * n_full + i],
                    "ch {o} col {i}: replicate streaming must be bit-exact"
                );
            }
        }
    }

    #[test]
    fn conv_replicate_undersized_first_chunk_is_loud() {
        // k=4, stride=1 → pad=3; a 2-column chunk cannot replicate-fill
        // (upstream asserts T >= TP) — loud error, not a silent zero-pad.
        let w = vec![0.25f32; 4];
        let conv = CausalConv1d::new(1, 1, 4, 1, 1, &w, None)
            .unwrap()
            .with_pad_mode(PadMode::Replicate);
        let compute = compute();
        let mut st = conv.state(4);
        let mut out = vec![0.0f32; 2];
        assert!(
            conv.process_into(&compute, &mut st, &[1.0, 2.0], 2, &mut out)
                .is_err()
        );
    }

    #[test]
    fn conv_rejects_bad_shapes() {
        let w = vec![0.0f32; 2 * 3]; // out_ch(2) * in_ch(1) * k(3)
        let conv = CausalConv1d::new(1, 2, 3, 2, 1, &w, None).unwrap();
        let compute = compute();
        let mut st = conv.state(8);
        let mut out = vec![0.0f32; 2]; // out_ch(2) * n_out(1)
        // t not a multiple of stride.
        assert!(
            conv.process_into(&compute, &mut st, &[0.0; 3], 3, &mut out)
                .is_err()
        );
        // k_eff < stride rejected at construction.
        assert!(CausalConv1d::new(1, 1, 1, 2, 1, &[1.0], None).is_err());
        // Transposed kernel < stride rejected.
        assert!(CausalConvTranspose1d::new(1, 1, 1, 2, vec![1.0], None).is_err());
    }

    fn tiny_transformer(
        rng: &mut SplitMix64,
        d: usize,
        n_head: usize,
        ff: usize,
    ) -> MimiTransformer {
        let bound = |rng: &mut SplitMix64, n: usize| rnd(rng, n).iter().map(|v| v * 0.2).collect();
        let layer = MimiTransformerLayer {
            ln1_gamma: vec![1.0; d],
            ln1_beta: vec![0.0; d],
            q_w_t: bound(rng, d * d),
            k_w_t: bound(rng, d * d),
            v_w_t: bound(rng, d * d),
            o_w_t: bound(rng, d * d),
            layer_scale_1: vec![0.01; d],
            ln2_gamma: vec![1.0; d],
            ln2_beta: vec![0.0; d],
            fc1_w_t: bound(rng, d * ff),
            fc2_w_t: bound(rng, ff * d),
            layer_scale_2: vec![0.01; d],
        };
        MimiTransformer::new(d, n_head, ff, 4, 10_000, vec![layer]).unwrap()
    }

    #[test]
    fn transformer_bulk_equals_chunked_and_respects_context_window() {
        let mut rng = SplitMix64::new(3);
        let tf = tiny_transformer(&mut rng, 8, 2, 16);
        let compute = compute();
        let t = 10; // > context (4) → the window rolls.
        let x0 = rnd(&mut rng, t * 8);

        let mut bulk = x0.clone();
        let mut s1 = tf.state();
        tf.process_inplace(&compute, &mut s1, &mut bulk, t).unwrap();

        let mut chunked = x0.clone();
        let mut s2 = tf.state();
        for c in 0..5 {
            let seg = &mut chunked[c * 2 * 8..(c + 1) * 2 * 8];
            tf.process_inplace(&compute, &mut s2, seg, 2).unwrap();
        }
        for (i, (a, b)) in bulk.iter().zip(chunked.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "position {i}: {a} vs {b}");
        }
        assert!(bulk.iter().all(|v| v.is_finite()));
        assert_ne!(bulk, x0, "the stack must transform the input");
    }

    #[test]
    fn transformer_rejects_ill_formed_dims() {
        // d not divisible by heads.
        assert!(MimiTransformer::new(9, 2, 4, 4, 10_000, vec![]).is_err());
        // odd head width (6 / 2 = 3 — RoPE pairs need even).
        assert!(MimiTransformer::new(6, 2, 4, 4, 10_000, vec![]).is_err());
    }
}
