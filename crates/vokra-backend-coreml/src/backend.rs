//! [`CoreMlBackend`] — the `vokra-core` [`Backend`] implementation (M5-01-T03).
//!
//! Per-op [`Backend`] coverage remains empty because CoreML is a delegate, not
//! an operator-partitioning backend. The independent [`DelegateBackend`]
//! surface executes the complete Whisper encoder from a compiled, hash-bound
//! sidecar. Every per-op request and every unbound submodel remains an explicit
//! error; there is no Vokra CPU fallback (FR-EX-08 / NFR-RL-06).

use std::cell::RefCell;
use std::rc::Rc;

use vokra_core::{
    AudioGraph, Backend, DelegateBackend, DelegateSubmodel, OpKind, Result, Tensor, VokraError,
};

use crate::artifact::CoreMlArtifact;
use crate::context::CoreMlContext;

thread_local! {
    /// `MLModel` is deliberately !Send/!Sync. Cache it on the calling thread
    /// so a session's repeated predictions exclude model-load time without
    /// claiming an unaudited cross-thread Objective-C contract.
    static THREAD_MODELS: RefCell<Vec<Rc<CoreMlBackend>>> = const { RefCell::new(Vec::new()) };
}

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
    /// Loaded delegate artifact. `None` for the device-only handle returned by
    /// [`Self::new`]; only [`Self::from_artifact`] can execute a submodel.
    context: Option<CoreMlContext>,
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
            context: None,
        })
    }

    /// Loads a compiled `.mlmodelc` and binds it as a complete Whisper encoder
    /// delegate.
    ///
    /// Missing artifacts, unavailable ANE hardware, and CoreML load failures
    /// are all explicit errors. This method never constructs a CPU substitute.
    pub fn from_artifact(artifact: CoreMlArtifact) -> Result<CoreMlBackend> {
        if !artifact.compiled_model().is_dir() {
            return Err(VokraError::ModelLoad(format!(
                "CoreML compiled artifact `{}` is missing or is not a .mlmodelc directory",
                artifact.compiled_model().display()
            )));
        }
        let caps = crate::probe::vokra_coreml_probe()?;
        let context = CoreMlContext::load(artifact)?;
        Ok(CoreMlBackend {
            ane_core_count: caps.ane_core_count,
            context: Some(context),
        })
    }

    /// The probed ANE core count the handle was built against (`None` off
    /// Apple, where `new()` cannot succeed).
    pub fn ane_core_count(&self) -> Option<u32> {
        self.ane_core_count
    }

    /// Bound compiled artifact, if this handle was built with
    /// [`Self::from_artifact`].
    pub fn artifact(&self) -> Option<&CoreMlArtifact> {
        self.context.as_ref().map(CoreMlContext::artifact)
    }

    /// Runs `execute` with a model cached on the current thread.
    ///
    /// This is the session-safe bridge for [`vokra_core::engines::AsrEngine`],
    /// whose engine object is `Send + Sync`: only the portable artifact is
    /// stored in the engine, while the retained Objective-C model remains on
    /// the thread that performs prediction. Up to four distinct artifacts are
    /// retained per thread; eviction releases the oldest model.
    pub fn with_thread_local_artifact<T>(
        artifact: &CoreMlArtifact,
        execute: impl FnOnce(&CoreMlBackend) -> Result<T>,
    ) -> Result<T> {
        let backend = THREAD_MODELS
            .try_with(|models| -> Result<Rc<CoreMlBackend>> {
                let mut models = models.try_borrow_mut().map_err(|_| {
                    VokraError::BackendUnavailable(
                        "CoreML thread-local model cache was re-entered during prediction"
                            .to_owned(),
                    )
                })?;
                if let Some(backend) = models
                    .iter()
                    .find(|backend| backend.artifact() == Some(artifact))
                {
                    return Ok(Rc::clone(backend));
                }
                let backend = Rc::new(Self::from_artifact(artifact.clone())?);
                if models.len() == 4 {
                    models.remove(0);
                }
                models.push(Rc::clone(&backend));
                Ok(backend)
            })
            .map_err(|_| {
                VokraError::BackendUnavailable(
                    "CoreML thread-local model cache is unavailable during thread teardown"
                        .to_owned(),
                )
            })??;
        execute(&backend)
    }
}

impl Backend for CoreMlBackend {
    fn name(&self) -> &str {
        "coreml"
    }

    fn supports(&self, _op: &OpKind) -> bool {
        // CoreML owns declared whole-submodel graphs through DelegateBackend;
        // it does not claim individual Vokra operators. Keeping this surface
        // empty prevents accidental graph partitioning or CPU fallback.
        false
    }

    fn execute(&self, graph: &AudioGraph) -> Result<()> {
        // With empty coverage, any non-empty graph has an uncovered op; report
        // the first one explicitly (FR-EX-08, no silent CPU fallback).
        for node in graph.nodes() {
            if !self.supports(node.op()) {
                return Err(VokraError::UnsupportedOp(format!(
                    "coreml delegate does not execute individual {:?} nodes; use a declared \
                     DelegateSubmodel with a validated artifact (no silent CPU fallback, \
                     FR-EX-08)",
                    node.op()
                )));
            }
        }
        // An empty graph reaches here; there is still no execution path.
        Err(VokraError::NotImplemented(
            "coreml does not execute generic AudioGraph values; use DelegateBackend with a \
             declared whole-submodel artifact",
        ))
    }

    fn eval_op(&self, op: &OpKind, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        let _ = inputs;
        Err(VokraError::UnsupportedOp(format!(
            "coreml delegate has no per-op kernel for {op:?}; use a declared whole submodel \
             (no silent CPU fallback, FR-EX-08)"
        )))
    }
}

impl DelegateBackend for CoreMlBackend {
    fn delegate_name(&self) -> &str {
        "coreml"
    }

    fn supports_submodel(&self, submodel: DelegateSubmodel) -> bool {
        matches!(submodel, DelegateSubmodel::WhisperEncoder) && self.context.is_some()
    }

    fn execute_submodel(
        &self,
        submodel: DelegateSubmodel,
        inputs: &[&Tensor],
    ) -> Result<Vec<Tensor>> {
        if !self.supports_submodel(submodel) {
            return Err(VokraError::UnsupportedOp(format!(
                "coreml has no compiled artifact bound for {submodel:?}; construct the delegate \
                 from a validated CoreMlArtifact (no silent CPU fallback, FR-EX-08)"
            )));
        }
        if inputs.len() != 1 {
            return Err(VokraError::InvalidArgument(format!(
                "coreml {submodel:?} expects exactly one log-mel tensor, got {}",
                inputs.len()
            )));
        }
        let context = self.context.as_ref().ok_or_else(|| {
            VokraError::UnsupportedOp(format!(
                "coreml has no compiled artifact bound for {submodel:?}"
            ))
        })?;
        match submodel {
            DelegateSubmodel::WhisperEncoder => {
                Ok(vec![context.predict_whisper_encoder(inputs[0])?])
            }
            _ => Err(VokraError::UnsupportedOp(format!(
                "coreml does not support declared submodel {submodel:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::{DelegateBackend, DelegateSubmodel};

    #[test]
    fn backend_reports_empty_coverage_and_no_silent_fallback() {
        // `new()` needs a real ANE; where it succeeds, assert the honest-empty
        // coverage contract. Where it does not (no ANE / non-Apple), that is a
        // legitimate BackendUnavailable, not a fabricated pass.
        match CoreMlBackend::new() {
            Ok(backend) => {
                assert_eq!(backend.name(), "coreml");
                // Delegate execution never advertises per-op coverage.
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

    #[test]
    fn artifact_contract_rejects_non_compiled_model_and_zero_shapes() {
        let err = CoreMlArtifact::whisper_encoder(
            "whisper-encoder.mlpackage",
            [1, 80, 3000],
            [1, 1500, 512],
        )
        .expect_err("runtime artifact must be a compiled .mlmodelc directory");
        assert!(matches!(err, VokraError::InvalidArgument(_)));

        let err = CoreMlArtifact::whisper_encoder(
            "whisper-encoder.mlmodelc",
            [1, 0, 3000],
            [1, 1500, 512],
        )
        .expect_err("zero-sized delegate axes must fail before CoreML is called");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn unloaded_backend_does_not_claim_whisper_encoder() {
        match CoreMlBackend::new() {
            Ok(backend) => {
                assert!(!backend.supports_submodel(DelegateSubmodel::WhisperEncoder));
                assert!(matches!(
                    backend.execute_submodel(DelegateSubmodel::WhisperEncoder, &[]),
                    Err(VokraError::UnsupportedOp(_))
                ));
            }
            Err(VokraError::BackendUnavailable(_)) => { /* no ANE here — skip */ }
            Err(other) => panic!("new() must be Ok or BackendUnavailable, got {other:?}"),
        }
    }

    #[test]
    fn artifact_load_fails_loudly_when_mlmodelc_is_missing() {
        let path = std::env::temp_dir().join(format!(
            "vokra-coreml-missing-{}-{}.mlmodelc",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let artifact =
            CoreMlArtifact::whisper_encoder(path, [1, 80, 3000], [1, 1500, 512]).unwrap();
        let err = CoreMlBackend::from_artifact(artifact)
            .expect_err("missing compiled artifact must never fall back to CPU");
        assert!(matches!(err, VokraError::ModelLoad(_)));
        assert!(format!("{err}").contains(".mlmodelc"));
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn compiled_fixture_executes_declared_submodel() {
        let Some(path) = std::env::var_os("VOKRA_COREML_TEST_MODEL") else {
            eprintln!(
                "VOKRA_COREML_TEST_MODEL unset; generate the independent fixture with \
                 tools/coreml/build_test_fixture.sh"
            );
            return;
        };
        let artifact = CoreMlArtifact::whisper_encoder(path, [1, 2, 3], [1, 2, 3]).unwrap();
        let backend = match CoreMlBackend::from_artifact(artifact) {
            Ok(backend) => backend,
            Err(VokraError::BackendUnavailable(msg)) => {
                eprintln!("no ANE on this host; skipping CoreML execution fixture ({msg})");
                return;
            }
            Err(other) => panic!("CoreML fixture failed to load: {other:?}"),
        };
        assert!(backend.supports_submodel(DelegateSubmodel::WhisperEncoder));
        let input = Tensor::host_f32(vec![1, 2, 3], vec![0.0, 1.0, 2.0, -1.0, -2.0, 4.5]).unwrap();
        let output = backend
            .execute_submodel(DelegateSubmodel::WhisperEncoder, &[&input])
            .unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].shape, vec![1, 2, 3]);
        assert_eq!(
            output[0].as_f32().unwrap(),
            &[1.0, 2.0, 3.0, 0.0, -1.0, 5.5]
        );
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn compiled_whisper_graph_executes_through_verified_cached_sidecar() {
        let Some(gguf) = std::env::var_os("VOKRA_COREML_WHISPER_SIDECAR_GGUF") else {
            eprintln!(
                "VOKRA_COREML_WHISPER_SIDECAR_GGUF unset; generate the structural fixture with \
                 tools/coreml/generate_synthetic_whisper_gguf.py and \
                 tools/coreml/build_whisper_encoder.sh"
            );
            return;
        };
        let artifact = CoreMlArtifact::from_whisper_sidecar(&gguf, "whisper", [1, 2, 4], [1, 2, 4])
            .expect("source/tree-hashed synthetic Whisper sidecar");
        let input = Tensor::host_f32(
            vec![1, 2, 4],
            vec![0.0, 0.1, 0.2, 0.3, -0.4, -0.3, -0.2, -0.1],
        )
        .unwrap();
        let run = || {
            CoreMlBackend::with_thread_local_artifact(&artifact, |backend| {
                backend
                    .execute_submodel(DelegateSubmodel::WhisperEncoder, &[&input])
                    .map(|mut outputs| outputs.remove(0))
            })
        };
        let first = run().expect("first whole-encoder prediction");
        let second = run().expect("cached whole-encoder prediction");
        assert_eq!(first.shape, vec![1, 2, 4]);
        assert_eq!(first.as_f32().unwrap(), second.as_f32().unwrap());
        assert!(
            first
                .as_f32()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
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
