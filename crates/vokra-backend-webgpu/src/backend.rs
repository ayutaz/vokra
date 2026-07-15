//! `Backend` trait handle for the WebGPU backend (M4-01-T17).
//!
//! Mirrors `vokra-backend-vulkan/src/backend.rs`: the handle exists on every
//! target, with honest no-silent-fallback behaviour — off wasm32 (or without
//! the `webgpu` feature) construction is an explicit
//! [`VokraError::BackendUnavailable`](vokra_core::VokraError::BackendUnavailable)
//! and `supports()` reports `false` for everything.

use vokra_core::{AudioGraph, Backend, OpKind, Result, Tensor, VokraError};

// Used only by `graph_op_dispatched_shader` (same cfg gate) — the plan builders
// are the authoritative source of each arm's dispatched shader name.
#[cfg(any(test, all(feature = "webgpu", target_arch = "wasm32")))]
use crate::plan;

/// Maps a graph [`OpKind`] to the WGSL kernel that backs its WebGPU
/// graph-executor arm — the single source of truth both
/// [`WebGpuBackend::supports`] and the `eval_webgpu_op` dispatcher derive
/// from, so the lock-step invariant (M3-02-T35 posture) holds **by
/// construction**. Host-portable and compiled on every target so the
/// mapping itself is natively testable.
///
/// `None` means the op has no WebGPU graph arm — either it is a front-end
/// signal op executed by `vokra-ops` (`Stft` / `MelFilterbank` / … — covered
/// by NO backend graph arm, the honest all-backend gap in the M4-13-T14
/// coverage table), or its kernel exists only as an imperative Whisper
/// primitive with no `OpKind` variant (`gemv` / `layer_norm` / `gelu` /
/// `conv1d` / `softmax_causal` / `activation` — surface 2 of the M4-13-T01
/// two-surface distinction, reached through the `vokra-models` `Compute`
/// seam).
///
/// Unlike the Vulkan sibling there is no pipeline-variant parameter (a
/// single `gemm_f32` WGSL kernel, no coop-matrix split) and no blob gating
/// (WGSL sources are embedded text — always present).
#[must_use]
pub fn graph_op_backing_shader(op: &OpKind) -> Option<&'static str> {
    match op {
        OpKind::Copy => Some("copy_f32"),
        OpKind::Add => Some("add_f32"),
        OpKind::MatMul => Some("gemm_f32"),
        OpKind::Mul => Some("elementwise"),
        OpKind::Softmax => Some("softmax"),
        _ => None,
    }
}

/// The WGSL kernel the graph-executor arm for `op` **actually dispatches**,
/// derived from the very `plan::plan_*` builder the matching `eval_*` body in
/// [`crate::eval`] calls. That module compiles only on wasm32, so the
/// correspondence is encoded here (host-portable) for the native lock-step
/// test and cross-referenced from each `eval_*`:
///
/// | `OpKind`  | `eval_*` → context seam         | `plan::*` builder    |
/// |-----------|---------------------------------|----------------------|
/// | `Copy`    | `eval_copy`   → `copy_f32`       | `plan_copy`          |
/// | `Add`     | `eval_add`    → `add_f32`        | `plan_add`           |
/// | `MatMul`  | `eval_matmul` → `gemm_f32`       | `plan_gemm`          |
/// | `Mul`     | `eval_mul`    → `elementwise`    | `plan_elementwise`   |
/// | `Softmax` | `eval_softmax`→ `softmax`        | `plan_softmax`       |
///
/// [`graph_op_backing_shader`] is the single source of truth for the *gate*
/// (`supports()`) side; this is the *dispatch* side. The
/// [`tests::dispatch_matches_supports_gate_for_every_covered_op`] test asserts
/// the two agree for every covered op — the M4-01 #23 drift (the `add_f32`
/// gate token while `eval_add` borrowed the `elementwise` op-switch kernel) is
/// exactly a disagreement between them. Returns `None` on the same domain as
/// [`graph_op_backing_shader`] (ops with no graph arm).
///
/// The shader name is read from the real `plan::*` builder (not a string
/// literal), so a builder rename propagates here automatically.
///
/// Compiled only where it is used: the native test suite (`cfg(test)`) and the
/// wasm32 dispatch path (`eval_webgpu_op`'s on-target `debug_assert`); an
/// off-wasm non-test build has no consumer, so gating it here keeps
/// `-D dead_code` clean without an `#[allow]`.
#[cfg(any(test, all(feature = "webgpu", target_arch = "wasm32")))]
#[must_use]
pub(crate) fn graph_op_dispatched_shader(op: &OpKind) -> Option<&'static str> {
    // Representative dims: the shader name is dimension-independent.
    let plan = match op {
        OpKind::Copy => plan::plan_copy(1),
        OpKind::Add => plan::plan_add(1),
        OpKind::MatMul => plan::plan_gemm(1, 1, 1, false),
        OpKind::Mul => plan::plan_elementwise(plan::ElementwiseOp::Mul, 1),
        OpKind::Softmax => plan::plan_softmax(1, 1),
        _ => return None,
    };
    Some(
        plan.expect("representative dims are valid for the plan builder")
            .shader,
    )
}

/// The WebGPU [`Backend`] handle.
pub struct WebGpuBackend {
    #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
    context: crate::context::WebGpuContext,
}

impl WebGpuBackend {
    /// Opens the backend (probes the adapter/device through the JS glue).
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] off wasm32, without the `webgpu`
    /// feature, or when the host has no WebGPU adapter (FR-EX-08 — never a
    /// silent CPU substitute).
    #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
    pub fn new() -> Result<Self> {
        Ok(WebGpuBackend {
            context: crate::context::WebGpuContext::new()?,
        })
    }

    /// Stub for non-WebGPU builds/targets — explicit
    /// [`VokraError::BackendUnavailable`] (FR-EX-08 / NFR-RL-06).
    ///
    /// # Errors
    ///
    /// Always [`VokraError::BackendUnavailable`].
    #[cfg(not(all(feature = "webgpu", target_arch = "wasm32")))]
    pub fn new() -> Result<Self> {
        Err(VokraError::BackendUnavailable(
            "vokra-backend-webgpu compiled without the `webgpu` feature or off wasm32; the \
             WebGPU backend runs inside a browser WASM module (ADR M4-01-webgpu-wasm). Select \
             BackendKind::Cpu explicitly for a CPU run — Vokra never falls back silently \
             (FR-EX-08)."
                .to_owned(),
        ))
    }

    /// The live dispatch context (wasm32 + `webgpu` builds only).
    #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
    pub(crate) fn context(&self) -> &crate::context::WebGpuContext {
        &self.context
    }
}

impl Backend for WebGpuBackend {
    fn name(&self) -> &str {
        "webgpu"
    }

    fn supports(&self, op: &OpKind) -> bool {
        // Lock-step with `eval_webgpu_op` by construction: both sides derive
        // from `graph_op_backing_shader`. No blob gate (WGSL text is always
        // embedded), so on-target coverage is static; off-target the backend
        // cannot even be constructed (`new()` fails), so it honestly
        // supports nothing.
        #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
        {
            graph_op_backing_shader(op).is_some_and(crate::wgsl::has_shader)
        }
        #[cfg(not(all(feature = "webgpu", target_arch = "wasm32")))]
        {
            let _ = op;
            false
        }
    }

    fn execute(&self, graph: &AudioGraph) -> Result<()> {
        for node in graph.nodes() {
            if !self.supports(node.op()) {
                return Err(VokraError::UnsupportedOp(format!(
                    "webgpu backend has no kernel for {:?} (no silent CPU fallback, FR-EX-08)",
                    node.op()
                )));
            }
        }
        // Coverage satisfied. `execute` stays a coverage-only check; the
        // data-carrying path is `vokra_core::run_graph`, which drives
        // `eval_op` (symmetric with the CPU / Metal / CUDA / Vulkan
        // backends).
        Err(VokraError::NotImplemented(
            "webgpu graph-level execution is vokra_core::run_graph (drives eval_op); execute \
             is coverage-only",
        ))
    }

    fn eval_op(&self, op: &OpKind, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
        {
            crate::eval::eval_webgpu_op(self, op, inputs)
        }
        #[cfg(not(all(feature = "webgpu", target_arch = "wasm32")))]
        {
            let _ = inputs;
            Err(VokraError::UnsupportedOp(format!(
                "webgpu backend is not compiled for this target / feature set; no kernel for \
                 {op:?}"
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The graph-arm mapping is exactly {Copy, Add, MatMul, Mul, Softmax} —
    /// the M4-13 Vulkan target set (CUDA's set plus `Copy`); everything else
    /// is None. Host-portable: this is the native half of the supports()/
    /// eval_op lock-step (the wasm32 half is exercised by the browser
    /// harness).
    #[test]
    fn graph_arm_mapping_is_the_vulkan_target_set() {
        assert_eq!(graph_op_backing_shader(&OpKind::Copy), Some("copy_f32"));
        assert_eq!(graph_op_backing_shader(&OpKind::Add), Some("add_f32"));
        assert_eq!(graph_op_backing_shader(&OpKind::MatMul), Some("gemm_f32"));
        assert_eq!(graph_op_backing_shader(&OpKind::Mul), Some("elementwise"));
        assert_eq!(graph_op_backing_shader(&OpKind::Softmax), Some("softmax"));
        // Front-end signal ops: NO backend graph arm (all-backend honest gap).
        assert_eq!(
            graph_op_backing_shader(&OpKind::Stft(vokra_core::ir::graph::StftAttrs::new(
                400, 160
            ))),
            None
        );
    }

    /// Every backing shader named by the graph arm exists in the WGSL
    /// manifest (a rename in either place fails here — the same drift gate
    /// `spirv::has_blob` provides on the Vulkan side, minus the blob gate).
    #[test]
    fn graph_arm_shaders_exist_in_the_manifest() {
        for op in [
            OpKind::Copy,
            OpKind::Add,
            OpKind::MatMul,
            OpKind::Mul,
            OpKind::Softmax,
        ] {
            let name = graph_op_backing_shader(&op).unwrap();
            assert!(
                crate::wgsl::has_shader(name),
                "graph arm names `{name}` but the WGSL manifest has no such kernel"
            );
        }
    }

    /// M4-01 #23 lock-step pin: for EVERY covered graph `OpKind`, the shader
    /// the `eval_*` dispatch actually runs
    /// ([`graph_op_dispatched_shader`]) equals the token `supports()` gates on
    /// ([`graph_op_backing_shader`]). Before the #23 fix `eval_add` dispatched
    /// the `elementwise` op-switch kernel while the gate advertised the
    /// dedicated `add_f32` kernel — a silent drift this asserts can never
    /// return. Host-portable (the wasm32 dispatch path is exercised by the
    /// browser harness; this pins the mapping on every target).
    #[test]
    fn dispatch_matches_supports_gate_for_every_covered_op() {
        for op in [
            OpKind::Copy,
            OpKind::Add,
            OpKind::MatMul,
            OpKind::Mul,
            OpKind::Softmax,
        ] {
            let gate = graph_op_backing_shader(&op);
            let dispatched = graph_op_dispatched_shader(&op);
            assert!(
                gate.is_some(),
                "{op:?} is a covered op but the gate returns None"
            );
            assert_eq!(
                gate, dispatched,
                "{op:?}: supports() gates `{gate:?}` but eval_op dispatches `{dispatched:?}` \
                 (M4-01 #23 shader drift)"
            );
        }
        // Domain agreement: a non-covered op is `None` on BOTH sides, so a
        // dispatch arm can never outlive (or precede) its gate.
        let stft = OpKind::Stft(vokra_core::ir::graph::StftAttrs::new(400, 160));
        assert_eq!(graph_op_backing_shader(&stft), None);
        assert_eq!(graph_op_dispatched_shader(&stft), None);
    }

    /// Off-target contract (native test hosts): construction is an explicit
    /// BackendUnavailable and a default-constructed-by-force handle would
    /// support nothing — FR-EX-08, no silent CPU fall back.
    #[test]
    fn off_target_is_explicit_backend_unavailable() {
        if cfg!(not(all(feature = "webgpu", target_arch = "wasm32"))) {
            let Err(err) = WebGpuBackend::new() else {
                panic!("off-target WebGpuBackend::new must fail (FR-EX-08)");
            };
            assert!(matches!(err, VokraError::BackendUnavailable(_)));
        }
    }
}
