//! [`QnnBackend`] — the `vokra-core` [`Backend`] implementation (M5-02-T02).
//!
//! Shape mirrors `CoreMlBackend` (M5-01), the sister delegate: this is the
//! **scaffold** slice, so op coverage is empty. Every op is an explicit
//! [`VokraError::UnsupportedOp`] and [`QnnBackend::new`] requires a reachable
//! QNN runtime ([`vokra_qnn_probe`](crate::vokra_qnn_probe)); there is no silent
//! CPU fall back (FR-EX-08 / NFR-RL-06). Coverage grows in the SDK-gated
//! graph-construction re-issue wave (owner T11 gates it).

use vokra_core::{AudioGraph, Backend, OpKind, Result, Tensor, VokraError};

/// QNN (Qualcomm Hexagon NPU) delegate backend handle.
///
/// On Android / Linux / Windows with the `qnn` feature, [`QnnBackend::new`]
/// probes for a reachable QNN runtime and fails explicitly if none is present.
/// On every other target / feature combination the type still exists (so
/// downstream code can name it) but [`QnnBackend::new`] fails with
/// [`VokraError::BackendUnavailable`]: the QNN backend is compiled out
/// (NFR-PT-01), never a silent CPU substitute (FR-EX-08). QNN is **not** NNAPI
/// (FR-BE-07) and **not** an Apple backend.
#[derive(Debug)]
pub struct QnnBackend {
    /// The QNN runtime library the handle was built against (populated only
    /// where `new()` can succeed).
    library_name: Option<String>,
}

impl QnnBackend {
    /// Creates a QNN backend, probing for a reachable QNN runtime.
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] if there is no QNN runtime (no SDK
    /// installed, a runner without the Hexagon runtime, or any target/feature
    /// where the backend is compiled out). Per NFR-RL-06 that is an explicit
    /// error, not a silent CPU fall back.
    pub fn new() -> Result<QnnBackend> {
        let caps = crate::probe::vokra_qnn_probe()?;
        Ok(QnnBackend {
            library_name: Some(caps.library_name),
        })
    }

    /// The QNN runtime library the handle was built against (`None` where
    /// `new()` cannot succeed).
    pub fn library_name(&self) -> Option<&str> {
        self.library_name.as_deref()
    }
}

impl Backend for QnnBackend {
    fn name(&self) -> &str {
        "qnn"
    }

    fn supports(&self, _op: &OpKind) -> bool {
        // Scaffold slice: no op has a wired QNN execution path yet. The graph
        // construction (`QnnGraph_create` → `addNode` → `finalize` → `execute`)
        // lands in the SDK-gated re-issue wave, so until then every op stays
        // uncovered and must surface as an explicit error (FR-EX-08), never a
        // silent CPU fall back. Kept deliberately empty rather than optimistic.
        false
    }

    fn execute(&self, graph: &AudioGraph) -> Result<()> {
        // With empty coverage, any non-empty graph has an uncovered op; report
        // the first one explicitly (FR-EX-08, no silent CPU fallback).
        for node in graph.nodes() {
            if !self.supports(node.op()) {
                return Err(VokraError::UnsupportedOp(format!(
                    "qnn backend has no execution path for {:?} yet (scaffold slice; QNN graph \
                     construction lands in the SDK-gated re-issue wave — no silent CPU fallback, \
                     FR-EX-08)",
                    node.op()
                )));
            }
        }
        // An empty graph reaches here; there is still no execution path.
        Err(VokraError::NotImplemented(
            "qnn graph execution is not implemented in the M5-02 scaffold slice (delegate submodel \
             execution lands in the SDK-gated graph-construction re-issue wave)",
        ))
    }

    fn eval_op(&self, op: &OpKind, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        let _ = inputs;
        Err(VokraError::UnsupportedOp(format!(
            "qnn backend has no kernel for {op:?} (scaffold slice; no silent CPU fallback, FR-EX-08)"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_reports_empty_coverage_and_no_silent_fallback() {
        // `new()` needs a real QNN runtime; where it succeeds, assert the
        // honest-empty coverage contract. Where it does not (no SDK / off
        // target), that is a legitimate BackendUnavailable, not a fabricated
        // pass.
        match QnnBackend::new() {
            Ok(backend) => {
                assert_eq!(backend.name(), "qnn");
                // Scaffold: nothing is covered yet.
                assert!(!backend.supports(&OpKind::MatMul));
                assert!(!backend.supports(&OpKind::Add));
                // eval_op on an uncovered op is an explicit UnsupportedOp.
                assert!(matches!(
                    backend.eval_op(&OpKind::MatMul, &[]),
                    Err(VokraError::UnsupportedOp(_))
                ));
            }
            Err(VokraError::BackendUnavailable(_)) => { /* no QNN runtime here — skip */ }
            Err(other) => panic!("new() must be Ok or BackendUnavailable, got {other:?}"),
        }
    }

    #[cfg(not(all(
        feature = "qnn",
        any(target_os = "android", target_os = "linux", target_os = "windows")
    )))]
    #[test]
    fn new_is_explicit_error_off_target() {
        assert!(matches!(
            QnnBackend::new(),
            Err(VokraError::BackendUnavailable(_))
        ));
    }
}
