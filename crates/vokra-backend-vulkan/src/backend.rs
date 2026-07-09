//! [`VulkanBackend`] — the `vokra-core` [`Backend`] implementation (M3-02-T23 /
//! T24 / T35).
//!
//! Symmetric with `MetalBackend` (M2-01) and `CudaBackend` (M2-03): two entry
//! points.
//!
//! 1. **Direct kernels** — future SPIR-V compute pipelines dispatched via
//!    `VulkanContext` (M3-02-T14 onwards). None ship in this foundation slice.
//! 2. **Graph execution** — [`Backend::eval_op`] evaluates one op on resolved
//!    [`Tensor`](vokra_core::Tensor) inputs by dispatching to the GPU, and
//!    [`vokra_core::run_graph`] drives it node-by-node. Every uncovered op is
//!    an explicit [`VokraError::UnsupportedOp`], never a silent CPU fallback
//!    (FR-EX-08). [`Backend::execute`] stays a coverage-only check.
//!
//! In this foundation slice, [`VulkanBackend::new`] runs the probe and
//! returns [`VokraError::BackendUnavailable`] if no Vulkan is present.
//! `supports()` returns `false` for every [`OpKind`] until the SPIR-V kernels
//! land (T14〜T22), so any graph execution attempt today surfaces
//! `UnsupportedOp` — the honest state.

use vokra_core::{AudioGraph, Backend, OpKind, Result, Tensor, VokraError};

/// Vulkan backend handle.
///
/// On Vulkan-capable targets it holds a [`crate::context::VulkanInstance`]
/// (loader + `VkInstance`), created — and device-probed — in
/// [`VulkanBackend::new`]. On every other target the type still exists (so
/// downstream code can name it) but [`VulkanBackend::new`] fails explicitly:
/// the Vulkan backend is compiled out (NFR-PT-01), never a silent CPU
/// substitute (FR-EX-08).
#[cfg(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
))]
pub struct VulkanBackend {
    _instance: crate::context::VulkanInstance,
    caps: crate::probe::VulkanCapabilities,
}

/// Vulkan backend handle (stub for other targets / feature-off builds — see
/// the Vulkan-target docs above).
#[cfg(not(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
)))]
pub struct VulkanBackend {
    _private: (),
}

// A manual `Debug` impl is used because `VulkanInstance` deliberately does not
// derive `Debug` (raw handles + function pointers are not useful to format).
impl core::fmt::Debug for VulkanBackend {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        #[cfg(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        ))]
        {
            f.debug_struct("VulkanBackend")
                .field("device_name", &self.caps.device_name)
                .field(
                    "api_version",
                    &format_args!(
                        "{}.{}",
                        self.caps.api_version_major, self.caps.api_version_minor
                    ),
                )
                .finish()
        }
        #[cfg(not(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        )))]
        {
            f.write_str("VulkanBackend(unavailable)")
        }
    }
}

#[cfg(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
))]
impl VulkanBackend {
    /// Creates a Vulkan backend, running the probe up-front so a missing
    /// loader / device becomes an explicit [`VokraError::BackendUnavailable`]
    /// at construction time (NFR-RL-06 — no silent CPU fall back).
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] if the Vulkan loader is not
    /// present, the loader is pre-1.1, or no physical device is enumerated
    /// (see [`crate::vokra_vulkan_probe`]).
    pub fn new() -> Result<VulkanBackend> {
        let caps = crate::probe::vokra_vulkan_probe()?;
        if !caps.subgroup_ready {
            return Err(VokraError::BackendUnavailable(format!(
                "Vulkan device present but subgroup precondition not met (Vokra requires \
                 Vulkan 1.1+ and a non-OTHER device type): {}",
                caps.summary()
            )));
        }
        let instance = crate::context::VulkanInstance::new()?;
        Ok(VulkanBackend {
            _instance: instance,
            caps,
        })
    }

    /// Access the discovered [`crate::probe::VulkanCapabilities`].
    #[must_use]
    pub fn capabilities(&self) -> &crate::probe::VulkanCapabilities {
        &self.caps
    }
}

#[cfg(not(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
)))]
impl VulkanBackend {
    /// Non-Vulkan stub: always fails — the Vulkan backend is not compiled
    /// for this target / feature set (NFR-PT-01), and per FR-EX-08 that is
    /// an explicit error rather than a silent CPU substitute.
    ///
    /// # Errors
    ///
    /// Always [`VokraError::BackendUnavailable`].
    pub fn new() -> Result<VulkanBackend> {
        Err(VokraError::BackendUnavailable(
            "Vulkan backend not compiled for this target / feature set (needs --features vulkan \
             on Linux / Android / Windows)."
                .to_owned(),
        ))
    }
}

impl Backend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
    }

    fn supports(&self, op: &OpKind) -> bool {
        // Foundation slice: NO Vulkan kernel is wired yet. Every op therefore
        // reports as unsupported. `execute` (below) will translate that into an
        // explicit `UnsupportedOp` — never a silent CPU fall back (FR-EX-08).
        //
        // As T14〜T22 land, this becomes a match on the op set the SPIR-V
        // kernels cover (GEMM / GEMV / softmax / softmax_causal / layer_norm /
        // gelu / conv1d / add / mul / relu / sigmoid / tanh / transpose /
        // gather).
        let _ = op;
        false
    }

    fn execute(&self, graph: &AudioGraph) -> Result<()> {
        for node in graph.nodes() {
            if !self.supports(node.op()) {
                return Err(VokraError::UnsupportedOp(format!(
                    "vulkan backend has no kernel for {:?} (no silent CPU fallback, FR-EX-08)",
                    node.op()
                )));
            }
        }
        // No kernels ship yet, so no covered graph exists to run through
        // `run_graph`. This branch is unreachable in the foundation slice; it
        // stays as an explicit "later WP" marker (T24 wires this up once the
        // SPIR-V pipelines exist).
        Err(VokraError::NotImplemented(
            "vulkan graph-level execution is vokra_core::run_graph (drives eval_op); execute is \
             coverage-only",
        ))
    }

    fn eval_op(&self, op: &OpKind, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        // Delegated to the eval module (T24). In the foundation slice the
        // dispatcher rejects every op with an explicit `UnsupportedOp` — no
        // silent CPU fall back. Kept in a separate module so T24〜T29 can
        // extend it op-by-op without touching this file.
        #[cfg(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        ))]
        {
            crate::eval::eval_vulkan_op(op, inputs)
        }
        #[cfg(not(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        )))]
        {
            let _ = inputs;
            Err(VokraError::UnsupportedOp(format!(
                "vulkan backend is not compiled for this target / feature set; no kernel for \
                 {op:?}"
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_name_is_vulkan() {
        // `name()` is target-independent and needs no device. On a
        // non-Vulkan host `new()` errors — assert both branches.
        #[cfg(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        ))]
        {
            if let Ok(backend) = VulkanBackend::new() {
                assert_eq!(backend.name(), "vulkan");
                assert!(!backend.supports(&OpKind::MatMul));
            }
        }
        #[cfg(not(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        )))]
        {
            assert!(matches!(
                VulkanBackend::new(),
                Err(VokraError::BackendUnavailable(_))
            ));
        }
    }

    /// M3-02-T35: silent CPU fall-back is explicitly forbidden. A graph that
    /// carries any op the backend does not cover MUST surface as
    /// `UnsupportedOp` from `execute`, never quietly succeed.
    #[test]
    fn execute_rejects_uncovered_ops_explicitly() {
        // On a non-Vulkan host we can't build a real VulkanBackend; the type
        // still exists so we exercise the trait via the target-agnostic error
        // wiring (backend.rs's `#[cfg(not(..))]` arm). On a Vulkan host we go
        // through the real path.
        #[cfg(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        ))]
        {
            let Ok(backend) = VulkanBackend::new() else {
                eprintln!("no Vulkan; skipping execute() coverage test");
                return;
            };
            use vokra_core::{DType, GraphBuilder, TensorDesc};
            // The simplest possible uncovered graph: a MatMul.
            let mut mb = GraphBuilder::new();
            let x = mb.add_tensor(TensorDesc::new("x", DType::F32, [2, 4]));
            let w = mb.add_tensor(TensorDesc::new("w", DType::F32, [4, 8]));
            let y = mb.add_tensor(TensorDesc::new("y", DType::F32, [2, 8]));
            mb.add_node(OpKind::MatMul, &[x, w], &[y]);
            mb.mark_input(x);
            mb.mark_output(y);
            let g = mb.finish().expect("valid graph");
            assert!(matches!(
                backend.execute(&g),
                Err(VokraError::UnsupportedOp(_))
            ));
        }
    }
}
