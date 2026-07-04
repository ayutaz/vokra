//! [`MetalBackend`] — the `vokra-core` [`Backend`] implementation (M2-01-T05).
//!
//! Symmetry with `CpuBackend` (M0-08) is deliberate: two entry points.
//!
//! 1. **Direct kernels** — on Apple, [`MetalBackend::gemm_f32`] (delegating to
//!    the wrapped [`MetalContext`]) is the surface the parity harness drives.
//! 2. **Graph execution** — [`Backend::execute`] validates op coverage and, per
//!    FR-EX-08, returns an explicit error for uncovered ops; it never silently
//!    falls back to the CPU backend. The data-carrying graph evaluator is a
//!    later WP, so once coverage is satisfied it returns
//!    [`VokraError::NotImplemented`] — exactly as `CpuBackend::execute` does.

use vokra_core::{AudioGraph, Backend, OpKind, Result, VokraError};

#[cfg(any(target_os = "macos", target_os = "ios"))]
use crate::context::MetalContext;

/// Metal backend handle.
///
/// On Apple targets it owns a [`MetalContext`] (device + queue + compiled GEMM
/// pipeline), created — and device-probed — in [`MetalBackend::new`]. On every
/// other target the type still exists (so downstream code can name it) but
/// [`MetalBackend::new`] fails explicitly: the Metal backend is compiled out
/// (NFR-PT-01), never a silent CPU substitute (FR-EX-08).
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[derive(Debug)]
pub struct MetalBackend {
    ctx: MetalContext,
}

/// Metal backend handle (non-Apple stub — see the Apple-target docs above).
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
#[derive(Debug)]
pub struct MetalBackend {
    _private: (),
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
impl MetalBackend {
    /// Creates a Metal backend, probing the system default device.
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] if there is no Metal device or the
    /// GEMM pipeline fails to build (NFR-RL-06: an incompatible device is an
    /// explicit error, not a silent CPU fall back).
    pub fn new() -> Result<MetalBackend> {
        Ok(MetalBackend {
            ctx: MetalContext::new()?,
        })
    }

    /// The wrapped GPU context (device + queue + GEMM pipeline).
    pub fn context(&self) -> &MetalContext {
        &self.ctx
    }

    /// Row-major FP32 GEMM on the GPU (see [`MetalContext::gemm_f32`]).
    ///
    /// # Errors
    ///
    /// Propagates [`MetalContext::gemm_f32`]'s errors (shape / Metal failures).
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

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
impl MetalBackend {
    /// Non-Apple stub: always fails — the Metal backend is not compiled for
    /// this target (NFR-PT-01), and per FR-EX-08 that is an explicit error
    /// rather than a silent CPU substitute.
    ///
    /// # Errors
    ///
    /// Always [`VokraError::BackendUnavailable`].
    pub fn new() -> Result<MetalBackend> {
        Err(VokraError::BackendUnavailable(
            "Metal backend is not compiled for this target (only macOS / iOS)".to_owned(),
        ))
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &str {
        "metal"
    }

    fn supports(&self, op: &OpKind) -> bool {
        // This slice ships one real Metal kernel: the FP32 GEMM (`MatMul`).
        // Every other op is uncovered and must surface as an explicit error at
        // execution (FR-EX-08) — never a silent CPU fall back. Further Metal
        // kernels (activation / softmax / conv1d / attention / FFT / mel) are
        // the follow-on M2-01 tickets.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            matches!(op, OpKind::MatMul)
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            let _ = op;
            false
        }
    }

    fn execute(&self, graph: &AudioGraph) -> Result<()> {
        for node in graph.nodes() {
            if !self.supports(node.op()) {
                return Err(VokraError::UnsupportedOp(format!(
                    "metal backend has no kernel for {:?} (no silent CPU fallback, FR-EX-08)",
                    node.op()
                )));
            }
        }
        // Coverage is satisfied; the tensor-data-carrying graph engine is a
        // later WP (symmetric with CpuBackend). Until then, drive the kernels
        // directly via `MetalBackend::gemm_f32` / `MetalContext`.
        Err(VokraError::NotImplemented(
            "metal graph-level execution needs the data-carrying engine (later WP); use MetalBackend::gemm_f32 directly",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_name_is_metal() {
        // `name()` is target-independent and needs no device.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            if let Ok(backend) = MetalBackend::new() {
                assert_eq!(backend.name(), "metal");
                assert!(backend.supports(&OpKind::MatMul));
                assert!(!backend.supports(&OpKind::Softmax));
            }
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            assert!(matches!(
                MetalBackend::new(),
                Err(VokraError::BackendUnavailable(_))
            ));
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn execute_defers_covered_graph_and_rejects_uncovered() {
        use vokra_core::ir::graph::StftAttrs;
        use vokra_core::{DType, GraphBuilder, TensorDesc};

        let Ok(backend) = MetalBackend::new() else {
            eprintln!("no Metal device; skipping execute coverage test");
            return;
        };

        // A MatMul-only graph is covered → reaches the explicit "later WP" stub.
        let mut mb = GraphBuilder::new();
        let x = mb.add_tensor(TensorDesc::new("x", DType::F32, [2, 4]));
        let w = mb.add_tensor(TensorDesc::new("w", DType::F32, [4, 8]));
        let y = mb.add_tensor(TensorDesc::new("y", DType::F32, [2, 8]));
        mb.add_node(OpKind::MatMul, &[x, w], &[y]);
        mb.mark_input(x);
        mb.mark_output(y);
        let covered = mb.finish().expect("valid graph");
        assert!(matches!(
            backend.execute(&covered),
            Err(VokraError::NotImplemented(_))
        ));

        // An uncovered op (`Stft`) must be an explicit UnsupportedOp error —
        // no silent CPU fallback (FR-EX-08 / NFR-RL-06).
        let mut sb = GraphBuilder::new();
        let s = sb.add_tensor(TensorDesc::new("s", DType::F32, [400]));
        let o = sb.add_tensor(TensorDesc::new("o", DType::F32, [2, 201]));
        sb.add_node(OpKind::Stft(StftAttrs::new(400, 160)), &[s], &[o]);
        sb.mark_input(s);
        sb.mark_output(o);
        let uncovered = sb.finish().expect("structurally valid graph");
        assert!(matches!(
            backend.execute(&uncovered),
            Err(VokraError::UnsupportedOp(_))
        ));
    }
}
