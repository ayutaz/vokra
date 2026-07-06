//! Imperative compute dispatcher for the native models (Phase 3 of the GPU
//! execution architecture; see `scratchpad/graph-engine-plan.md` §3).
//!
//! The graph evaluator ([`vokra_core::run_graph`]) drives one op at a time via
//! [`Backend::eval_op`](vokra_core::Backend); it is the right shape for new /
//! fused / graph-first models. The **existing** models (Whisper, piper-plus,
//! CAM++) are imperative: they call the compute kernels directly in a
//! zero-malloc hot path (`out: &mut [f32]`, caller-owned scratch — FR-EX-05).
//! Rewriting them onto the graph engine would add a large op surface and risk
//! the numeric parity for no speed gain (same kernels). Instead this module adds
//! a thin, typed seam — [`Compute`] — that those call sites dispatch through, so
//! the same GEMM the CPU backend runs can instead run on the GPU by swapping one
//! enum arm.
//!
//! # One kernel per (backend, op); two entry shapes
//!
//! [`Compute::gemm_f32`] on the CPU arm calls the very same
//! [`vokra_backend_cpu::kernels::gemm_f32`] that
//! [`Backend::eval_op`](vokra_core::Backend) does, and on the Metal arm the very
//! same `MetalContext::gemm_f32` — there is no second kernel. So the imperative
//! `Compute` path and the graph `eval_op` path stay bit-for-bit consistent on a
//! given backend, and a `Compute::cpu()` run reproduces the pre-seam output
//! **exactly** (the parity suites stay green).
//!
//! # One model = one backend, no silent fallback (FR-EX-08)
//!
//! [`Compute::for_backend`] takes the model's *required* hot-op set and refuses
//! to build a backend that does not cover **every** op in it — an explicit
//! [`VokraError::UnsupportedOp`], never a per-op CPU fall back. As of Phase 4
//! (M2-01 T09-T13) the Metal backend has a real GPU kernel for every hot op
//! (GEMM / GEMV / softmax / layer-norm / GELU / conv1d), so not only the
//! GEMM-only models (CAM++, piper-plus) but the **full Whisper forward** runs on
//! Metal through this seam. A backend that genuinely could not cover an op would
//! still be an explicit `UnsupportedOp` rather than a silent CPU fall back;
//! selecting the CPU instead is the caller's *explicit* [`BackendKind::Cpu`]
//! choice.
//!
//! # `!Send` `MetalContext`, `Send + Sync` engines
//!
//! `MetalContext` is `!Send` / `!Sync` (thread-affine `id` handles), whereas the
//! engine traits (`AsrEngine` / `TtsEngine` / …) are `Send + Sync`. So a model
//! **engine** must not *hold* a live backend; it holds a [`BackendKind`]
//! (`Copy`) and builds a `Compute` on the stack at each transcribe / synthesize
//! entry, threading `&Compute` down. That keeps the engines `Send + Sync` while
//! the `!Send` context lives only for the call.

use vokra_backend_cpu::kernels;
use vokra_core::backend::BackendKind;
use vokra_core::{Backend, DecoderLayerView, PrenormLayer, Result, VokraError};

/// A backend-dispatched hot op — the operators the imperative models route
/// through a backend (as opposed to the model-internal scalar glue like
/// LeakyReLU, embedding lookup or transpose, which always stays on the host and
/// is *not* a backend op, so is never a silent fall back).
///
/// A model declares the set it needs (`*_HOT_OPS`) so [`Compute::for_backend`]
/// can enforce whole-model backend coverage before running anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotOp {
    /// Row-major GEMM (`gemm_f32`) — the dominant matmul / linear / conv (via
    /// im2col) cost. The first op wired onto the GPU (M2-01 slice).
    Gemm,
    /// Row-major matrix-vector product (`gemv_f32`) — Whisper's tied logits head.
    Gemv,
    /// Row-wise softmax (`softmax_f32`) — attention.
    Softmax,
    /// Affine layer normalisation (`layer_norm_f32`) — Whisper pre-norm blocks.
    LayerNorm,
    /// Exact (erf) GELU (`gelu_f32`) — Whisper MLP / conv stem.
    Gelu,
    /// 1-D convolution (`conv1d_f32`) — Whisper encoder stem.
    Conv1d,
}

impl HotOp {
    /// Whether the Metal backend's imperative [`Compute`] seam covers this op.
    ///
    /// Kept in sync with the Metal arms of the [`Compute`] methods below; the
    /// `metal_coverage_is_consistent` test pins the two together. As of Phase 4
    /// (M2-01 T09-T13) every hot op has a `MetalContext` kernel, so the whole
    /// Whisper set runs on the GPU through this seam. (The *graph* backend
    /// `MetalBackend::supports` / `eval_op` is a separate path and still covers
    /// only `MatMul` — the two coverage surfaces are intentionally independent.)
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    fn covered_by_metal(self) -> bool {
        // Phase 4 complete: GEMM (M2-01 slice) plus the five T09-T13 kernels
        // (gemv / softmax / layer_norm / gelu / conv1d) are all real Metal
        // kernels on `MetalContext`, so every HotOp is covered.
        matches!(
            self,
            HotOp::Gemm
                | HotOp::Gemv
                | HotOp::Softmax
                | HotOp::LayerNorm
                | HotOp::Gelu
                | HotOp::Conv1d
        )
    }

    /// Whether the CUDA backend's imperative [`Compute`] seam covers this op.
    ///
    /// Kept in sync with the `Be::Cuda` arms of the [`Compute`] methods below;
    /// the `cuda_coverage_is_consistent` test pins the two together. As of
    /// Phase 4 (M2-03 T10-T14) the CUDA backend has a real NVRTC-compiled kernel
    /// for every hot op (GEMM plus gemv / softmax / layer_norm / gelu / conv1d),
    /// so the whole Whisper set runs on the GPU through this seam. (The *graph*
    /// backend `CudaBackend::supports` / `eval_op` is a separate path and still
    /// covers only `MatMul` — the two coverage surfaces are independent.)
    #[cfg(all(feature = "cuda", any(unix, windows)))]
    fn covered_by_cuda(self) -> bool {
        // Phase 4 complete: GEMM (M2-03 slice) plus the five T10-T14 kernels
        // (gemv / softmax / layer_norm / gelu / conv1d) are all real CUDA kernels
        // on `CudaContext`, so every HotOp is covered.
        matches!(
            self,
            HotOp::Gemm
                | HotOp::Gemv
                | HotOp::Softmax
                | HotOp::LayerNorm
                | HotOp::Gelu
                | HotOp::Conv1d
        )
    }
}

/// A typed, zero-malloc compute dispatcher the imperative model hot path calls
/// instead of the `vokra_backend_cpu::kernels::*` free functions directly.
///
/// Build one at a model entry point with [`Compute::for_backend`] (or the
/// infallible [`Compute::cpu`]) and thread `&Compute` down; the `out: &mut [f32]`
/// method shape preserves the zero-allocation hot path (FR-EX-05). It is a plain
/// `enum` dispatch (not `&dyn`), so the CPU per-call cost over calling the kernel
/// directly is a single branch.
pub struct Compute {
    be: Be,
}

/// The live backend behind a [`Compute`]. The `Metal` arm owns a `!Send`
/// `MetalContext`, which is why a `Compute` is built at a call entry and never
/// stored on a `Send + Sync` engine.
enum Be {
    /// CPU kernels (`vokra_backend_cpu::kernels`). Covers every [`HotOp`].
    Cpu,
    /// Metal GPU context. Covers every [`HotOp`] (Phase 4).
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    Metal(vokra_backend_metal::MetalContext),
    /// CUDA GPU context. Covers every [`HotOp`] (Phase 4). `Box`ed because
    /// `CudaContext` embeds the whole `CudaDriver` (≈20 dlopen'd fn pointers) by
    /// value, which would make the `Be` enum's inline size dwarf the other arms
    /// (`clippy::large_enum_variant`); the heap alloc is negligible — a `Compute`
    /// is built once per model entry, after a far costlier dlopen + NVRTC compile.
    #[cfg(all(feature = "cuda", any(unix, windows)))]
    Cuda(Box<vokra_backend_cuda::CudaContext>),
}

impl Compute {
    /// A CPU-backed dispatcher. Infallible: the CPU backend covers every op, and
    /// its methods reproduce the pre-seam kernel calls bit-for-bit.
    #[must_use]
    pub fn cpu() -> Self {
        Compute { be: Be::Cpu }
    }

    /// Builds a dispatcher for `kind`, requiring it to cover every op in
    /// `required` (one model = one backend, FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] if `kind` is a real backend that does not
    ///   cover some op in `required` (e.g. Metal for a model that needs softmax)
    ///   — never a per-op CPU fall back.
    /// - [`VokraError::BackendUnavailable`] if `kind` is not built into this
    ///   binary (e.g. `Metal` without the `metal` feature, or off an Apple
    ///   target), or if the device probe fails (no Metal device).
    pub fn for_backend(kind: BackendKind, required: &[HotOp]) -> Result<Self> {
        // `required` is consulted only by the Metal / CUDA coverage gates;
        // without either GPU arm compiled in, the CPU / unavailable arms do not
        // read it.
        #[cfg(not(any(
            all(feature = "metal", any(target_os = "macos", target_os = "ios")),
            all(feature = "cuda", any(unix, windows))
        )))]
        let _ = required;
        match kind {
            BackendKind::Cpu => Ok(Compute::cpu()),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            BackendKind::Metal => {
                if let Some(op) = required.iter().copied().find(|op| !op.covered_by_metal()) {
                    return Err(VokraError::UnsupportedOp(format!(
                        "metal backend does not cover {op:?} in this slice; the model requires \
                         {required:?}. One model = one backend — Vokra does not silently run the \
                         uncovered ops on the CPU (FR-EX-08). Select BackendKind::Cpu, or wait for \
                         the Metal {op:?} kernel (M2-01)."
                    )));
                }
                Ok(Compute {
                    be: Be::Metal(vokra_backend_metal::MetalContext::new()?),
                })
            }
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            BackendKind::Cuda => {
                if let Some(op) = required.iter().copied().find(|op| !op.covered_by_cuda()) {
                    return Err(VokraError::UnsupportedOp(format!(
                        "cuda backend does not cover {op:?} in this slice; the model requires \
                         {required:?}. One model = one backend — Vokra does not silently run the \
                         uncovered ops on the CPU (FR-EX-08). Select BackendKind::Cpu, or wait for \
                         the CUDA {op:?} kernel (M2-03 T10–T14)."
                    )));
                }
                Ok(Compute {
                    be: Be::Cuda(Box::new(vokra_backend_cuda::CudaContext::new()?)),
                })
            }
            other => Err(VokraError::BackendUnavailable(format!(
                "{other:?} backend is not built into vokra-models (build with the `metal` feature \
                 on macOS / iOS for Metal; CUDA / Vulkan / … are later roadmap backends)"
            ))),
        }
    }

    /// The backend this dispatcher runs on (`"cpu"` or `"metal"`).
    #[must_use]
    pub fn backend_name(&self) -> &'static str {
        match &self.be {
            Be::Cpu => "cpu",
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(_) => "metal",
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => "cuda",
        }
    }

    /// Whether this backend has the Phase-5 fused non-causal attention
    /// ([`Self::attn_f32`]): `true` on the GPU arms (Metal / CUDA), `false` on
    /// CPU.
    ///
    /// The caller (`whisper::nn::attention_from_kv_into`) gates the fused fast
    /// path on this: only a GPU backend routes a non-causal block through
    /// `attn_f32`; the CPU always runs the per-op head loop. This keeps the CPU
    /// arm of `attn_f32` an explicit [`VokraError::UnsupportedOp`] that correct
    /// code never reaches (no silent fall back, FR-EX-08), while `compute.rs`
    /// hosts **zero** duplicated attention math (nn.rs is the single source of
    /// truth for the head loop).
    #[must_use]
    pub fn attention_is_fused(&self) -> bool {
        match &self.be {
            Be::Cpu => false,
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(_) => true,
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => true,
        }
    }

    /// Whether this backend has the Phase-5-follow-on device-resident whole-encoder
    /// stack ([`Self::encode_prenorm_encoder`]): `true` on the GPU arms (Metal /
    /// CUDA), `false` on CPU.
    ///
    /// The caller (`whisper::encoder::encode`) gates the fused encoder on this:
    /// only a GPU backend routes the whole pre-norm block stack through
    /// `encode_prenorm_encoder` (one submission for the encoder); the CPU always
    /// runs the per-op `encoder_block` loop. This keeps the CPU arm of
    /// `encode_prenorm_encoder` an explicit [`VokraError::UnsupportedOp`] correct
    /// code never reaches (no silent fall back, FR-EX-08), while the block math
    /// lives in exactly one place (the CPU `encoder_block` loop is the single
    /// source of truth — `compute.rs` hosts no duplicated encoder loop).
    #[must_use]
    pub fn prenorm_stack_is_fused(&self) -> bool {
        match &self.be {
            Be::Cpu => false,
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(_) => true,
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => true,
        }
    }

    /// Row-major GEMM with optional per-column bias
    /// (`out[i,j] = bias[j] + Σ_l a[i,l]·b[l,j]`); `a` is `m×k`, `b` is `k×n`.
    ///
    /// The GPU-accelerated op in this slice: the CPU arm calls
    /// [`kernels::gemm_f32`], the Metal arm the identically-typed
    /// `MetalContext::gemm_f32` (drop-in, M2-01-T18).
    #[allow(clippy::too_many_arguments)] // intrinsic GEMM parameter set (matches kernels::gemm_f32)
    pub fn gemm_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => kernels::gemm_f32(m, n, k, a, b, bias, out),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.gemm_f32(m, n, k, a, b, bias, out),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(ctx) => ctx.gemm_f32(m, n, k, a, b, bias, out),
        }
    }

    /// Row-major matrix-vector product with optional per-row bias
    /// (`out[i] = bias[i] + Σ_l a[i,l]·x[l]`); `a` is `m×k`.
    pub fn gemv_f32(
        &self,
        m: usize,
        k: usize,
        a: &[f32],
        x: &[f32],
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => kernels::gemv_f32(m, k, a, x, bias, out),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.gemv_f32(m, k, a, x, bias, out),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(ctx) => ctx.gemv_f32(m, k, a, x, bias, out),
        }
    }

    /// Row-wise softmax over the innermost axis of a `rows × cols` buffer.
    pub fn softmax_f32(
        &self,
        input: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => kernels::softmax_f32(input, out, rows, cols),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.softmax_f32(input, out, rows, cols),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(ctx) => ctx.softmax_f32(input, out, rows, cols),
        }
    }

    /// Affine layer normalisation over the innermost axis of a `rows × cols`
    /// buffer (`gamma` / `beta` length `cols`).
    #[allow(clippy::too_many_arguments)] // intrinsic layer-norm parameter set (matches kernels::layer_norm_f32)
    pub fn layer_norm_f32(
        &self,
        input: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => kernels::layer_norm_f32(input, out, rows, cols, gamma, beta, eps),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.layer_norm_f32(input, out, rows, cols, gamma, beta, eps),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(ctx) => ctx.layer_norm_f32(input, out, rows, cols, gamma, beta, eps),
        }
    }

    /// Element-wise exact (erf) GELU (`x` and `out` equal length).
    pub fn gelu_f32(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        match &self.be {
            Be::Cpu => kernels::gelu_f32(x, out),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.gelu_f32(x, out),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(ctx) => ctx.gelu_f32(x, out),
        }
    }

    /// 1-D convolution via im2col + GEMM (`input` is `in_ch × in_len`, `weight`
    /// is `out_ch × in_ch × kernel`, `out` is `out_ch × out_len`).
    #[allow(clippy::too_many_arguments)] // convolution's intrinsic parameter set (matches kernels::conv1d_f32)
    pub fn conv1d_f32(
        &self,
        input: &[f32],
        in_ch: usize,
        in_len: usize,
        weight: &[f32],
        out_ch: usize,
        kernel: usize,
        bias: Option<&[f32]>,
        stride: usize,
        padding: usize,
        out: &mut [f32],
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => kernels::conv1d_f32(
                input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, out,
            ),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.conv1d_f32(
                input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, out,
            ),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(ctx) => ctx.conv1d_f32(
                input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, out,
            ),
        }
    }

    /// Fused MLP `fc2(gelu(fc1(x)))` — the Phase-5 device-residency slice.
    ///
    /// `x` is `[t, d]`; `fc1` maps `d → ffn` (`fc1_w` is `[d, ffn]`, bias
    /// `[ffn]`); `fc2` maps `ffn → d` (`fc2_w` is `[ffn, d]`, bias `[d]`); `out`
    /// is `[t, d]`. `mlp_h` / `mlp_a` are the two `[t, ffn]` intermediates.
    ///
    /// On the **CPU** arm this is the identical three-kernel sequence
    /// (`gemm_f32` → `gelu_f32` → `gemm_f32`, into `mlp_h` / `mlp_a`) the
    /// pre-fusion `whisper::nn::mlp_into` ran, so it is **bit-for-bit** the
    /// pre-seam result (the parity suites stay green). On the **Metal / CUDA**
    /// arms the same three kernels run in ONE GPU submission with the two
    /// `[t, ffn]` intermediates resident on the device — only `out` is read back
    /// — which is bit-identical to three separate GPU ops but pays one readback /
    /// one sync instead of three. `mlp_h` / `mlp_a` are unused on the GPU arms
    /// (the device holds those intermediates); the caller still sizes them so the
    /// CPU arm and the zero-alloc reserve are unaffected.
    #[allow(clippy::too_many_arguments)] // fused-MLP operand set (two Linears + scratch + dims)
    pub fn mlp_f32(
        &self,
        t: usize,
        d: usize,
        ffn: usize,
        x: &[f32],
        fc1_w: &[f32],
        fc1_bias: Option<&[f32]>,
        fc2_w: &[f32],
        fc2_bias: Option<&[f32]>,
        mlp_h: &mut [f32],
        mlp_a: &mut [f32],
        out: &mut [f32],
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => {
                // Bit-identical to the former `mlp_into`: fc1 GEMM → GELU → fc2
                // GEMM through the same CPU kernels, in the same order, into the
                // caller's scratch.
                kernels::gemm_f32(t, ffn, d, x, fc1_w, fc1_bias, mlp_h)?;
                kernels::gelu_f32(mlp_h, mlp_a)?;
                kernels::gemm_f32(t, d, ffn, mlp_a, fc2_w, fc2_bias, out)
            }
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.mlp_f32(t, d, ffn, x, fc1_w, fc1_bias, fc2_w, fc2_bias, out),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(ctx) => ctx.mlp_f32(t, d, ffn, x, fc1_w, fc1_bias, fc2_w, fc2_bias, out),
        }
    }

    /// Fused **non-causal** multi-head attention — the Phase-5 device-residency
    /// slice (the sibling of [`Self::mlp_f32`]).
    ///
    /// `xq` is `[t_q, d]`; `k` / `v` are the pre-projected `[t_kv, d]` keys /
    /// values; `q_w` / `out_w` are `[d, d]` (both projections `d → d`), biases
    /// `[d]`; `scale = head_dim^-0.5` (the caller folds the query scale in);
    /// `out` is `[t_q, d]`.
    ///
    /// **GPU-only.** On the Metal / CUDA arms this runs the q-proj → per-head
    /// {gather, QKᵀ, softmax, A·V, scatter} → out-proj chain in ONE GPU
    /// submission with every intermediate resident on the device (bit-identical
    /// to the per-op path, one readback instead of many). The **CPU arm is an
    /// explicit [`VokraError::UnsupportedOp`]**: the CPU never fuses attention —
    /// it runs the per-op head loop in `whisper::nn::attention_from_kv_into`,
    /// which gates this call behind [`Self::attention_is_fused`], so correct code
    /// never hits the CPU arm. This keeps the attention math in exactly one place
    /// (nn.rs) with no silent CPU fall back (FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::UnsupportedOp`] on the CPU arm; the GPU arms return
    /// [`VokraError::InvalidArgument`] on a shape mismatch and
    /// [`VokraError::BackendUnavailable`] on a device failure.
    #[allow(clippy::too_many_arguments)]
    // fused-attention operand set (two Linears + K/V + dims)
    // Without a GPU arm compiled in, only the CPU arm (which reads none of the
    // operands, just returns UnsupportedOp) remains, so every operand is unused —
    // exactly as `for_backend` cfg-silences its `required` argument.
    #[cfg_attr(
        not(any(
            all(feature = "metal", any(target_os = "macos", target_os = "ios")),
            all(feature = "cuda", any(unix, windows))
        )),
        allow(unused_variables)
    )]
    pub fn attn_f32(
        &self,
        t_q: usize,
        t_kv: usize,
        d: usize,
        n_head: usize,
        xq: &[f32],
        q_w: &[f32],
        q_bias: Option<&[f32]>,
        k: &[f32],
        v: &[f32],
        out_w: &[f32],
        out_bias: Option<&[f32]>,
        scale: f32,
        out: &mut [f32],
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => Err(VokraError::UnsupportedOp(
                "attn_f32 is the GPU fused attention path; the CPU uses the per-op attention loop \
                 (whisper::nn::attention_from_kv_into gates it behind Compute::attention_is_fused)"
                    .to_owned(),
            )),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.attn_f32(
                t_q, t_kv, d, n_head, xq, q_w, q_bias, k, v, out_w, out_bias, scale, out,
            ),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(ctx) => ctx.attn_f32(
                t_q, t_kv, d, n_head, xq, q_w, q_bias, k, v, out_w, out_bias, scale, out,
            ),
        }
    }

    /// Device-resident **whole pre-norm encoder** (Phase-5 follow-on): runs
    /// `n × [ln → attn → residual → ln → mlp → residual]` + a final LayerNorm with
    /// the hidden state and every intermediate kept on the GPU across all blocks,
    /// so the encoder pays ONE submission instead of the per-op path's `6·N + 1`.
    ///
    /// `hidden` is the `[t, d]` post-conv-stem input, `out` the `[t, d]` normed
    /// output; `layers` are the per-block weight slices; `n_head` splits `d`;
    /// `eps` is the LayerNorm epsilon (the caller passes the CPU-kernel constant,
    /// which the backend cannot import).
    ///
    /// **GPU-only.** On the Metal / CUDA arms this is bit-identical to running the
    /// blocks per-op on the GPU (same kernels, order, geometry) and matches the CPU
    /// within the FP32 bound. The **CPU arm is an explicit
    /// [`VokraError::UnsupportedOp`]**: the CPU never fuses the encoder — it runs
    /// the per-op `encoder_block` loop in `whisper::encoder`, which gates this call
    /// behind [`Self::prenorm_stack_is_fused`], so correct code never hits the CPU
    /// arm (no silent CPU fall back, FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::UnsupportedOp`] on the CPU arm; the GPU arms return
    /// [`VokraError::InvalidArgument`] on a shape mismatch and
    /// [`VokraError::BackendUnavailable`] on a device failure.
    #[allow(clippy::too_many_arguments)] // whole-encoder operand set (dims + weights + I/O)
    #[cfg_attr(
        not(any(
            all(feature = "metal", any(target_os = "macos", target_os = "ios")),
            all(feature = "cuda", any(unix, windows))
        )),
        allow(unused_variables)
    )]
    pub fn encode_prenorm_encoder(
        &self,
        t: usize,
        d: usize,
        ff: usize,
        n_head: usize,
        eps: f32,
        hidden: &[f32],
        layers: &[PrenormLayer<'_>],
        final_ln_gamma: &[f32],
        final_ln_beta: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        match &self.be {
            Be::Cpu => Err(VokraError::UnsupportedOp(
                "encode_prenorm_encoder is the GPU device-resident encoder path; the CPU uses the \
                 per-op encoder_block loop (whisper::encoder::encode gates it behind \
                 Compute::prenorm_stack_is_fused)"
                    .to_owned(),
            )),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(ctx) => ctx.encode_prenorm_stack(
                t,
                d,
                ff,
                n_head,
                eps,
                hidden,
                layers,
                final_ln_gamma,
                final_ln_beta,
                out,
            ),
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(ctx) => ctx.encode_prenorm_stack(
                t,
                d,
                ff,
                n_head,
                eps,
                hidden,
                layers,
                final_ln_gamma,
                final_ln_beta,
                out,
            ),
        }
    }

    /// Whether this backend has the Phase-3 device-resident **decoder-step**
    /// session ([`Self::new_decoder_step_session`] on Metal; CUDA is Phase 3b
    /// and returns `false` for now).
    ///
    /// The caller (`whisper::decoder::DecoderState`) gates the whole-step device
    /// path on this: only a Metal backend builds a [`DecoderStepSession`] at
    /// construction and routes every step through it; CPU and (for now) CUDA
    /// keep the per-op step loop untouched. This keeps
    /// [`Self::new_decoder_step_session`]'s CPU / CUDA arms an explicit
    /// [`VokraError::UnsupportedOp`] correct code never hits (no silent fall
    /// back, FR-EX-08), with zero duplicated decode-block math in `compute.rs`.
    #[must_use]
    pub fn decoder_step_is_session_backed(&self) -> bool {
        match &self.be {
            Be::Cpu => false,
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(_) => true,
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => false,
        }
    }

    /// Builds a device-resident **decoder-step session** — the Phase-3 device-
    /// residency slice sibling of [`Self::encode_prenorm_encoder`], for the
    /// autoregressive decode (weights uploaded once, self-attention K/V kept on
    /// the GPU and appended each step, cross-attention K/V uploaded once from
    /// the pre-projected slices in `layers`).
    ///
    /// `dims` names the model shape; `layers` carries every decoder block's
    /// weight slices (row-major, `[in, out]` layout — the same layout the CPU
    /// per-op path uses) plus the pre-projected cross-K/V; `token_emb` is the
    /// tied-head / embedding table `[n_vocab, d]`; `ln_post_gamma` /
    /// `ln_post_beta` are the final decoder LayerNorm.
    ///
    /// **GPU-only.** On the Metal arm this returns a session ready for
    /// [`DecoderStepSession::step`] (one command-buffer submission + one full
    /// `[t, n_vocab]` logits readback per step; bit-identical to running the
    /// step per-op on the GPU). The **CPU and CUDA arms are explicit
    /// [`VokraError::UnsupportedOp`]** — the CPU never fuses the decoder step,
    /// and CUDA is Phase 3b (not yet wired); the model layer gates this call
    /// behind [`Self::decoder_step_is_session_backed`], so correct code never
    /// hits either. No silent CPU fall back (FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::UnsupportedOp`] on the CPU / CUDA arms; the Metal arm
    /// returns [`VokraError::InvalidArgument`] on a shape mismatch and
    /// [`VokraError::BackendUnavailable`] on a device failure.
    #[cfg_attr(
        not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))),
        allow(unused_variables)
    )]
    pub fn new_decoder_step_session(
        &self,
        dims: DecoderStepDims,
        layers: &[DecoderLayerView<'_>],
        token_emb: &[f32],
        ln_post_gamma: &[f32],
        ln_post_beta: &[f32],
    ) -> Result<DecoderStepSession> {
        match &self.be {
            Be::Cpu => Err(VokraError::UnsupportedOp(
                "new_decoder_step_session is the GPU device-resident decoder-step driver; the CPU \
                 runs the per-op step loop (whisper::decoder::DecoderState gates it behind \
                 Compute::decoder_step_is_session_backed)"
                    .to_owned(),
            )),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Be::Metal(_) => {
                // The session owns its own `MetalContext` (weights + KV live inside
                // it; the outer `Compute`'s context is used only for the cross-KV
                // precompute at construction and dropped afterwards). Bit-identical
                // to the per-op Metal path within the FP32 bound: same kernels,
                // same launch geometry, one command-buffer submission per step.
                let s = vokra_backend_metal::MetalDecodeSession::new(
                    dims.d,
                    dims.n_head,
                    dims.ff,
                    dims.n_text_ctx,
                    dims.n_vocab,
                    dims.n_ctx,
                    dims.max_t_q,
                    dims.eps,
                    layers,
                    token_emb,
                    ln_post_gamma,
                    ln_post_beta,
                )?;
                Ok(DecoderStepSession::Metal(Box::new(s)))
            }
            #[cfg(all(feature = "cuda", any(unix, windows)))]
            Be::Cuda(_) => Err(VokraError::UnsupportedOp(
                "new_decoder_step_session on CUDA is Phase 3b (not yet wired); the CUDA backend \
                 continues to run the per-op step loop (no silent CPU fall back, FR-EX-08). Wait \
                 for CudaDecodeSession."
                    .to_owned(),
            )),
        }
    }
}

/// Immutable model shape for a device-resident decoder-step session
/// ([`Compute::new_decoder_step_session`]).
///
/// Names the dims the backend needs at build time to size its resident buffers
/// once (`n_text_ctx` bounds the self-attention KV cache; `max_t_q` bounds the
/// per-step scratch and the tied-head logits buffer; `n_ctx` matches the encoder
/// output width the pre-projected cross-K/V is `[n_ctx, d]` rows of). `eps` is
/// the LayerNorm epsilon (the caller passes the CPU-kernel constant, which the
/// backend cannot import).
#[derive(Clone, Copy, Debug)]
pub struct DecoderStepDims {
    /// Hidden width.
    pub d: usize,
    /// Attention head count (must divide `d`).
    pub n_head: usize,
    /// MLP inner width.
    pub ff: usize,
    /// Max decoder-context length (the hard self-attention KV cache bound).
    pub n_text_ctx: usize,
    /// Vocabulary size (the tied logits head output width).
    pub n_vocab: usize,
    /// Encoder context length (the cross-attention key window; the
    /// pre-projected `cross_k` / `cross_v` in each [`DecoderLayerView`] are
    /// `[n_ctx, d]` rows).
    pub n_ctx: usize,
    /// Widest single decode step's query length (the forced-prefix width;
    /// steady-state greedy decodes one token). Bounds the per-step scratch and
    /// the `[max_t_q, n_vocab]` logits buffer.
    pub max_t_q: usize,
    /// LayerNorm epsilon (the backend cannot import the CPU-kernel constant).
    pub eps: f32,
}

/// A backend-specific device-resident **decoder-step session** — the
/// autoregressive-decode sibling of [`Compute::encode_prenorm_encoder`].
///
/// Built once at [`Compute::new_decoder_step_session`] (Metal only in this
/// slice; CUDA is Phase 3b). Each [`Self::step`] runs the whole decode step
/// device-resident in ONE command-buffer submission, then reads back the full
/// `[t, n_vocab]` logits so the model layer can compare against the CPU
/// decoder's row-major output (not only the greedy last row). The session owns
/// its own backend context (weights + KV live inside it), so a Metal
/// [`DecoderStepSession`] holds Metal handles — see the `unsafe impl Send` note
/// below for why the model layer can still hold it inside a `Send`
/// `DecoderState`.
pub enum DecoderStepSession {
    /// Metal (M2 Phase 3a) device-resident decoder-step session.
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    Metal(Box<vokra_backend_metal::MetalDecodeSession>),
}

// `DecoderStepSession` is `Send` because its only variant boxes a
// `MetalDecodeSession`, which the backend crate declares `Send` (see the
// backend crate for the SAFETY rationale — Metal handles are thread-safe;
// unsafe impls live behind `vokra-backend-metal`'s `#![allow(unsafe_code)]`
// opt-out because vokra-models stays under the workspace `unsafe_code = deny`).
// The model-layer `DecoderState` therefore stays `Send` (its
// `assert_send::<DecoderState>()` compile-time bound + the cross-thread decode
// test both continue to hold) without either reuploading every weight per step
// or duplicating attention math in `compute.rs`.

impl DecoderStepSession {
    /// Advances the decode by the `t` tokens whose `[t, d]` token+positional
    /// embedding is `embedded`, starting at the committed position `start`.
    /// Runs the whole step device-resident in ONE command-buffer submission +
    /// ONE `[t, n_vocab]` logits readback (bit-identical to running the step
    /// per-op on the GPU).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a bad `t` / `start` / `embedded`
    /// length; [`VokraError::BackendUnavailable`] on a device failure.
    #[cfg_attr(
        not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))),
        allow(unused_variables)
    )]
    pub fn step(&mut self, embedded: &[f32], t: usize, start: usize) -> Result<()> {
        match self {
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            DecoderStepSession::Metal(s) => s.step(embedded, t, start),
            // Off the Metal build the enum is uninhabited (no variants) and
            // `Option<DecoderStepSession>` is only ever `None`; the model-layer
            // caller (`whisper::decoder::DecoderState`) never constructs a
            // session and so never calls this. `Self` still contains fields
            // (`&mut self` binding), so the match falls through the empty
            // never-reachable wildcard.
            #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
            _ => unreachable!("DecoderStepSession has no variants off the Metal build"),
        }
    }

    /// Rewinds the decode position to 0 for a fresh decode of the same audio
    /// (resident weights + cross-KV stay valid; the self-attention KV rows are
    /// simply overwritten from row 0). Mirrors [`vokra_core::KvCache::reset`].
    pub fn reset(&mut self) {
        match self {
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            DecoderStepSession::Metal(s) => s.reset(),
            #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
            _ => unreachable!("DecoderStepSession has no variants off the Metal build"),
        }
    }

    /// Committed token positions in the self-attention cache (the causal query
    /// offset for the next [`Self::step`]).
    #[must_use]
    pub fn positions(&self) -> usize {
        match self {
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            DecoderStepSession::Metal(s) => s.positions(),
            #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
            _ => unreachable!("DecoderStepSession has no variants off the Metal build"),
        }
    }

    /// The last decoded row of the last [`Self::step`] — `[n_vocab]` logits,
    /// the greedy / argmax read. Empty before the first step.
    #[must_use]
    pub fn last_logits(&self) -> &[f32] {
        match self {
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            DecoderStepSession::Metal(s) => s.last_logits(),
            #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
            _ => unreachable!("DecoderStepSession has no variants off the Metal build"),
        }
    }

    /// All `[t, n_vocab]` rows the last [`Self::step`] wrote, row-major
    /// (row `i` at offset `i·n_vocab`). This is the full-row output the model-
    /// layer path compares against the CPU decoder's `[t, n_vocab]` logits (not
    /// only the greedy last row). Empty before the first step.
    #[must_use]
    pub fn all_logits(&self) -> &[f32] {
        match self {
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            DecoderStepSession::Metal(s) => s.all_logits(),
            #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
            _ => unreachable!("DecoderStepSession has no variants off the Metal build"),
        }
    }
}

/// Builds a boxed [`Backend`] for the graph evaluator ([`vokra_core::run_graph`])
/// — the assembly-layer factory (§2.4). Distinct from [`Compute`], which is the
/// imperative seam; both ultimately drive the same per-(backend, op) kernels.
///
/// # Errors
///
/// [`VokraError::BackendUnavailable`] if `kind` is not built into this binary or
/// (for Metal) has no device.
pub fn make_backend(kind: BackendKind) -> Result<Box<dyn Backend>> {
    match kind {
        BackendKind::Cpu => Ok(Box::new(vokra_backend_cpu::CpuBackend::new())),
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        BackendKind::Metal => Ok(Box::new(vokra_backend_metal::MetalBackend::new()?)),
        #[cfg(all(feature = "cuda", any(unix, windows)))]
        BackendKind::Cuda => Ok(Box::new(vokra_backend_cuda::CudaBackend::new()?)),
        other => Err(VokraError::BackendUnavailable(format!(
            "{other:?} backend is not built into vokra-models (build with the `metal` feature on \
             macOS / iOS for Metal, or the `cuda` feature on Windows / Linux for CUDA)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_compute_matches_direct_kernel_bit_for_bit() {
        // The whole point of the seam: `Compute::cpu()` must reproduce the direct
        // kernel call exactly (atol = 0), so the model parity suites stay green.
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
        let bias = [0.5f32, -0.5];

        let mut via_compute = vec![0.0f32; 4];
        Compute::cpu()
            .gemm_f32(2, 2, 3, &a, &b, Some(&bias), &mut via_compute)
            .unwrap();

        let mut direct = vec![0.0f32; 4];
        kernels::gemm_f32(2, 2, 3, &a, &b, Some(&bias), &mut direct).unwrap();

        assert_eq!(via_compute, direct, "Compute::cpu gemm != direct kernel");
    }

    #[test]
    fn prenorm_stack_cpu_is_unsupported_no_silent_fallback() {
        // The CPU never fuses the encoder: `prenorm_stack_is_fused()` is false (so
        // `whisper::encoder::encode` keeps the per-op `encoder_block` loop), and
        // `encode_prenorm_encoder` on CPU is an explicit UnsupportedOp — never a
        // silent CPU substitute of the GPU-only device-resident path (FR-EX-08).
        let cpu = Compute::cpu();
        assert!(!cpu.prenorm_stack_is_fused());
        let layer = PrenormLayer {
            attn_ln_gamma: &[1.0, 1.0],
            attn_ln_beta: &[0.0, 0.0],
            q_w: &[1.0, 0.0, 0.0, 1.0],
            q_bias: None,
            k_w: &[1.0, 0.0, 0.0, 1.0],
            k_bias: None,
            v_w: &[1.0, 0.0, 0.0, 1.0],
            v_bias: None,
            out_w: &[1.0, 0.0, 0.0, 1.0],
            out_bias: None,
            mlp_ln_gamma: &[1.0, 1.0],
            mlp_ln_beta: &[0.0, 0.0],
            fc1_w: &[1.0, 0.0, 0.0, 1.0],
            fc1_bias: None,
            fc2_w: &[1.0, 0.0, 0.0, 1.0],
            fc2_bias: None,
        };
        let mut out = [0.0f32; 2];
        assert!(matches!(
            cpu.encode_prenorm_encoder(
                1,
                2,
                2,
                1,
                1e-5,
                &[0.0, 0.0],
                &[layer],
                &[1.0, 1.0],
                &[0.0, 0.0],
                &mut out,
            ),
            Err(VokraError::UnsupportedOp(_))
        ));
    }

    #[test]
    fn cpu_for_backend_covers_every_op() {
        // The CPU backend covers the full hot-op set unconditionally.
        let all = [
            HotOp::Gemm,
            HotOp::Gemv,
            HotOp::Softmax,
            HotOp::LayerNorm,
            HotOp::Gelu,
            HotOp::Conv1d,
        ];
        let c = Compute::for_backend(BackendKind::Cpu, &all).expect("cpu covers all");
        assert_eq!(c.backend_name(), "cpu");
    }

    #[test]
    fn make_backend_cpu_is_the_cpu_backend() {
        let b = make_backend(BackendKind::Cpu).expect("cpu backend");
        assert_eq!(b.name(), "cpu");
    }

    /// Off the Metal build (feature off or non-Apple), selecting Metal is an
    /// explicit unavailability error — never a silent CPU substitute.
    #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
    #[test]
    fn metal_without_the_feature_is_explicit_unavailable() {
        assert!(matches!(
            Compute::for_backend(BackendKind::Metal, &[HotOp::Gemm]),
            Err(VokraError::BackendUnavailable(_))
        ));
        assert!(matches!(
            make_backend(BackendKind::Metal),
            Err(VokraError::BackendUnavailable(_))
        ));
    }

    /// On a Metal build, coverage is enforced. As of Phase 4 the Metal backend
    /// covers the **whole** Whisper hot-op set, so `for_backend` never returns
    /// `UnsupportedOp` for it — it either builds (device present) or reports an
    /// explicit device unavailability (no silent CPU fall back).
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn metal_coverage_is_consistent() {
        // Every hot op is covered (this pins `covered_by_metal` to the wired
        // Metal method arms — all now dispatch to a `MetalContext` kernel).
        for op in [
            HotOp::Gemm,
            HotOp::Gemv,
            HotOp::Softmax,
            HotOp::LayerNorm,
            HotOp::Gelu,
            HotOp::Conv1d,
        ] {
            assert!(
                op.covered_by_metal(),
                "{op:?} unexpectedly NOT Metal-covered"
            );
        }

        // Whisper's full set is therefore a covered request: it either builds
        // (device present) or fails with an explicit device error — never a
        // coverage `UnsupportedOp`, never a silent CPU fall back (FR-EX-08).
        let whisper = [
            HotOp::Gemm,
            HotOp::Gemv,
            HotOp::Softmax,
            HotOp::LayerNorm,
            HotOp::Gelu,
            HotOp::Conv1d,
        ];
        match Compute::for_backend(BackendKind::Metal, &whisper) {
            Ok(c) => assert_eq!(c.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {
                eprintln!("no Metal device; full-coverage build path is device-gated");
            }
            Err(e) => panic!("unexpected error for a fully-covered request: {e}"),
        }
    }

    /// On a CUDA build, coverage is enforced. As of Phase 4 (M2-03 T10-T14) the
    /// CUDA backend covers the **whole** Whisper hot-op set, so `for_backend`
    /// never returns `UnsupportedOp` for it — it either builds (device present)
    /// or reports an explicit device unavailability (no silent CPU fall back,
    /// FR-EX-08 / NFR-RL-06). The device branch is exercised on the vast.ai
    /// RTX 4090 (M2-03-T25); here it skips.
    #[cfg(all(feature = "cuda", any(unix, windows)))]
    #[test]
    fn cuda_coverage_is_consistent() {
        // Every hot op is covered (this pins `covered_by_cuda` to the wired
        // CUDA method arms — all now dispatch to a `CudaContext` kernel).
        for op in [
            HotOp::Gemm,
            HotOp::Gemv,
            HotOp::Softmax,
            HotOp::LayerNorm,
            HotOp::Gelu,
            HotOp::Conv1d,
        ] {
            assert!(op.covered_by_cuda(), "{op:?} unexpectedly NOT CUDA-covered");
        }

        // Whisper's full set is therefore a covered request: it either builds
        // (device present) or fails with an explicit device error — never a
        // coverage `UnsupportedOp`, never a silent CPU fall back (FR-EX-08).
        let whisper = [
            HotOp::Gemm,
            HotOp::Gemv,
            HotOp::Softmax,
            HotOp::LayerNorm,
            HotOp::Gelu,
            HotOp::Conv1d,
        ];
        match Compute::for_backend(BackendKind::Cuda, &whisper) {
            Ok(c) => assert_eq!(c.backend_name(), "cuda"),
            Err(VokraError::BackendUnavailable(_)) => {
                eprintln!(
                    "no CUDA device; full-coverage build path is device-gated (run on vast.ai)"
                );
            }
            Err(e) => panic!("unexpected error for a fully-covered CUDA request: {e}"),
        }
    }
}
