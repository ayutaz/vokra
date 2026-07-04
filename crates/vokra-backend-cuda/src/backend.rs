//! [`CudaBackend`] — the `vokra-core` [`Backend`] implementation (M2-03-T16).
//!
//! Symmetry with `CpuBackend` (M0-08) and `MetalBackend` (M2-01) is deliberate,
//! two entry points:
//!
//! 1. **Direct kernels** — [`CudaBackend::gemm_f32`] (delegating to the wrapped
//!    [`CudaContext`]) is the surface the parity harness drives.
//! 2. **Graph execution** — [`Backend::eval_op`] evaluates one op on resolved
//!    [`Tensor`](vokra_core::Tensor) inputs on the GPU; `MatMul` routes into
//!    `CudaContext::gemm_f32`, and every uncovered op is an explicit
//!    [`VokraError::UnsupportedOp`], never a silent CPU fall back (FR-EX-08).
//!    [`Backend::execute`] stays a coverage-only check (symmetric with the CPU /
//!    Metal backends).
//!
//! On a target without a dynamic loader / driver the type still exists (so
//! downstream code can name it) but [`CudaBackend::new`] fails explicitly
//! (NFR-RL-06) — the CUDA backend is never a silent CPU substitute.

use vokra_core::{AudioGraph, Backend, OpKind, Result, Tensor, VokraError};

#[cfg(any(unix, windows))]
use crate::context::CudaContext;

/// CUDA backend handle (Unix / Windows) — owns a [`CudaContext`] (driver +
/// context + stream + compiled GEMM), created and device-probed in
/// [`CudaBackend::new`].
#[cfg(any(unix, windows))]
#[derive(Debug)]
pub struct CudaBackend {
    ctx: CudaContext,
}

/// CUDA backend handle (no-dynamic-loader stub — see the Unix/Windows docs).
#[cfg(not(any(unix, windows)))]
#[derive(Debug)]
pub struct CudaBackend {
    _private: (),
}

#[cfg(any(unix, windows))]
impl CudaBackend {
    /// Creates a CUDA backend, loading the driver and building the GPU context.
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] if there is no NVIDIA driver/GPU (e.g.
    /// on an Apple Mac), NVRTC is absent, or the GEMM kernel fails to build
    /// (NFR-RL-06: an unavailable/incompatible device is an explicit error, not
    /// a silent CPU fall back).
    pub fn new() -> Result<CudaBackend> {
        Ok(CudaBackend {
            ctx: CudaContext::new()?,
        })
    }

    /// The wrapped GPU context (driver + context + stream + GEMM kernel).
    #[must_use]
    pub fn context(&self) -> &CudaContext {
        &self.ctx
    }

    /// Row-major FP32 GEMM on the GPU (see [`CudaContext::gemm_f32`]).
    ///
    /// # Errors
    ///
    /// Propagates [`CudaContext::gemm_f32`]'s errors (shape / device failures).
    #[allow(clippy::too_many_arguments)] // intrinsic GEMM parameter set (matches CPU gemm_f32)
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
        self.ctx.gemm_f32(m, n, k, a, b, bias, out)
    }
}

#[cfg(not(any(unix, windows)))]
impl CudaBackend {
    /// No-dynamic-loader stub: always fails — the CUDA backend needs dlopen /
    /// LoadLibrary, and per FR-EX-08 that is an explicit error rather than a
    /// silent CPU substitute.
    ///
    /// # Errors
    ///
    /// Always [`VokraError::BackendUnavailable`].
    pub fn new() -> Result<CudaBackend> {
        Err(VokraError::BackendUnavailable(
            "CUDA backend requires a Unix or Windows target with a dynamic loader".to_owned(),
        ))
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn supports(&self, op: &OpKind) -> bool {
        // This slice ships one real CUDA kernel: the FP32 GEMM (`MatMul`). Every
        // other op is uncovered and must surface as an explicit error at
        // execution (FR-EX-08) — never a silent CPU fall back. Further CUDA
        // kernels (activation / softmax / conv1d / FlashAttention v2) are the
        // follow-on M2-03 tickets (T10–T14).
        #[cfg(any(unix, windows))]
        {
            matches!(op, OpKind::MatMul)
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = op;
            false
        }
    }

    fn execute(&self, graph: &AudioGraph) -> Result<()> {
        for node in graph.nodes() {
            if !self.supports(node.op()) {
                return Err(VokraError::UnsupportedOp(format!(
                    "cuda backend has no kernel for {:?} (no silent CPU fallback, FR-EX-08)",
                    node.op()
                )));
            }
        }
        // Coverage is satisfied. `execute` stays a coverage-only check; the
        // data-carrying path is `vokra_core::run_graph`, which drives `eval_op`
        // (symmetric with CpuBackend / MetalBackend).
        Err(VokraError::NotImplemented(
            "cuda graph-level execution is vokra_core::run_graph (drives eval_op); execute is coverage-only",
        ))
    }

    fn eval_op(&self, op: &OpKind, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        // On Unix/Windows the wrapped `CudaContext` runs the GPU kernel; only
        // `MatMul` is covered in this slice and the dispatcher rejects every
        // other op with an explicit `UnsupportedOp` (FR-EX-08, no silent CPU
        // fall back).
        #[cfg(any(unix, windows))]
        {
            crate::eval::eval_cuda_op(&self.ctx, op, inputs)
        }
        // On a target without the backend, `new()` errors, so this is
        // unreachable in practice; it must still compile and stays an explicit
        // error rather than a silent CPU substitute (FR-EX-08).
        #[cfg(not(any(unix, windows)))]
        {
            let _ = inputs;
            Err(VokraError::UnsupportedOp(format!(
                "cuda backend is not compiled for this target; no kernel for {op:?}"
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_name_is_cuda() {
        // On a CUDA host `new()` succeeds and the name/coverage are checkable;
        // on a CUDA-less host it is an explicit unavailability error (never a
        // silent CPU substitute). Both branches are valid.
        match CudaBackend::new() {
            Ok(backend) => {
                assert_eq!(backend.name(), "cuda");
                assert!(backend.supports(&OpKind::MatMul));
                assert!(!backend.supports(&OpKind::Softmax));
            }
            Err(VokraError::BackendUnavailable(_)) => {
                eprintln!("no CUDA backend on this host (expected off a CUDA GPU; run on vast.ai)");
            }
            Err(other) => panic!("unexpected error constructing CudaBackend: {other}"),
        }
    }
}
