//! [`CoreMlBackend`] — the `vokra-core` [`Backend`] implementation (M5-01-T03).
//!
//! Shape mirrors `MetalBackend` (M2-01), but this is the **scaffold** slice:
//! op coverage is empty. Every op is an explicit
//! [`VokraError::UnsupportedOp`] and [`CoreMlBackend::new`] requires a reachable
//! ANE ([`vokra_coreml_probe`](crate::vokra_coreml_probe)); there is no silent
//! CPU fall back (FR-EX-08 / NFR-RL-06). Coverage grows once the execution path
//! lands after the M5-01-T02 model-supply ADR is ratified.

use vokra_core::{AudioGraph, Backend, OpKind, Result, Tensor, VokraError};

/// CoreML delegate backend handle.
///
/// On Apple targets [`CoreMlBackend::new`] probes for a reachable Apple Neural
/// Engine and fails explicitly if none is present. On every other target the
/// type still exists (so downstream code can name it) but
/// [`CoreMlBackend::new`] fails with
/// [`VokraError::BackendUnavailable`]: the CoreML backend is compiled out
/// (NFR-PT-01), never a silent CPU substitute (FR-EX-08).
#[derive(Debug)]
pub struct CoreMlBackend {
    /// The probed ANE core count, kept so callers can report what the handle
    /// was built against. Populated only on Apple targets with an ANE.
    ane_core_count: Option<u32>,
}

impl CoreMlBackend {
    /// Creates a CoreML backend, probing for a reachable Apple Neural Engine.
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] if there is no ANE (an Intel Mac, a
    /// runner that hides the Neural Engine, or any non-Apple target). Per
    /// NFR-RL-06 that is an explicit error, not a silent CPU fall back.
    pub fn new() -> Result<CoreMlBackend> {
        let caps = crate::probe::vokra_coreml_probe()?;
        Ok(CoreMlBackend {
            ane_core_count: caps.ane_core_count,
        })
    }

    /// The probed ANE core count the handle was built against (`None` off
    /// Apple, where `new()` cannot succeed).
    pub fn ane_core_count(&self) -> Option<u32> {
        self.ane_core_count
    }
}

impl Backend for CoreMlBackend {
    fn name(&self) -> &str {
        "coreml"
    }

    fn supports(&self, _op: &OpKind) -> bool {
        // Scaffold slice: no op has a wired CoreML execution path yet. The
        // execution path turns on the M5-01-T02 model-supply ADR, so until
        // that lands every op stays uncovered and must surface as an explicit
        // error (FR-EX-08), never a silent CPU fall back. Kept deliberately
        // empty rather than optimistic.
        false
    }

    fn execute(&self, graph: &AudioGraph) -> Result<()> {
        // With empty coverage, any non-empty graph has an uncovered op; report
        // the first one explicitly (FR-EX-08, no silent CPU fallback).
        for node in graph.nodes() {
            if !self.supports(node.op()) {
                return Err(VokraError::UnsupportedOp(format!(
                    "coreml backend has no execution path for {:?} yet (scaffold slice; \
                     the op path lands after the M5-01-T02 model-supply ADR — no silent \
                     CPU fallback, FR-EX-08)",
                    node.op()
                )));
            }
        }
        // An empty graph reaches here; there is still no execution path.
        Err(VokraError::NotImplemented(
            "coreml graph execution is not implemented in the M5-01 scaffold slice \
             (delegate submodel execution lands after the T02 ADR)",
        ))
    }

    fn eval_op(&self, op: &OpKind, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        let _ = inputs;
        Err(VokraError::UnsupportedOp(format!(
            "coreml backend has no kernel for {op:?} (scaffold slice; no silent CPU fallback, \
             FR-EX-08)"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_reports_empty_coverage_and_no_silent_fallback() {
        // `new()` needs a real ANE; where it succeeds, assert the honest-empty
        // coverage contract. Where it does not (no ANE / non-Apple), that is a
        // legitimate BackendUnavailable, not a fabricated pass.
        match CoreMlBackend::new() {
            Ok(backend) => {
                assert_eq!(backend.name(), "coreml");
                // Scaffold: nothing is covered yet.
                assert!(!backend.supports(&OpKind::MatMul));
                assert!(!backend.supports(&OpKind::Add));
                // eval_op on an uncovered op is an explicit UnsupportedOp.
                assert!(matches!(
                    backend.eval_op(&OpKind::MatMul, &[]),
                    Err(VokraError::UnsupportedOp(_))
                ));
            }
            Err(VokraError::BackendUnavailable(_)) => { /* no ANE here — skip */ }
            Err(other) => panic!("new() must be Ok or BackendUnavailable, got {other:?}"),
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn new_is_explicit_error_off_apple() {
        assert!(matches!(
            CoreMlBackend::new(),
            Err(VokraError::BackendUnavailable(_))
        ));
    }
}
