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
use vokra_core::{Backend, Result, VokraError};

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
