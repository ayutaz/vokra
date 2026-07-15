//! M4-05 T21/T22 — CSM GPU decode-session test bands (the M3-10 Voxtral
//! 3-band pattern: Metal-gated / CUDA-gated / off-GPU negative).
//!
//! - **Off-GPU band** (always compiled): a backend whose feature is off
//!   must be an explicit error through [`vokra_models::csm::gpu_backend_probe`]
//!   — never a silent CPU fall back (FR-EX-08).
//! - **Metal band** (`--features metal`, Apple): real-GPU parity on the
//!   M1 iMac (CLAUDE.md dev environment) — backbone hidden state within
//!   `atol ≤ 5e-4` of CPU and greedy frame codes identical. Skips cleanly
//!   (with a printed reason) when no device is present.
//! - **CUDA band** (`--features cuda`): API-symmetric; real-GPU parity is
//!   vast.ai-gated (spot RTX 4090) — CI skips cleanly off-device.

use vokra_core::BackendKind;
use vokra_models::csm::gpu_backend_probe;

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
#[test]
fn metal_probe_off_feature_is_an_explicit_error() {
    let err = gpu_backend_probe(BackendKind::Metal).unwrap_err();
    assert!(
        matches!(err, vokra_core::VokraError::BackendUnavailable(_)),
        "off-feature Metal must be BackendUnavailable, got {err:?}"
    );
}

#[cfg(not(all(feature = "cuda", any(unix, windows))))]
#[test]
fn cuda_probe_off_feature_is_an_explicit_error() {
    let err = gpu_backend_probe(BackendKind::Cuda).unwrap_err();
    assert!(
        matches!(err, vokra_core::VokraError::BackendUnavailable(_)),
        "off-feature CUDA must be BackendUnavailable, got {err:?}"
    );
}

#[test]
fn cpu_probe_always_succeeds() {
    gpu_backend_probe(BackendKind::Cpu).expect("CPU hosts the CSM hot-op set");
}

#[cfg(any(
    all(feature = "metal", any(target_os = "macos", target_os = "ios")),
    all(feature = "cuda", any(unix, windows))
))]
mod gpu_common {
    use vokra_core::{BackendKind, Result, VokraError};
    use vokra_models::csm::{
        CsmBackbone, CsmBackboneState, CsmConfig, CsmFrame, CsmGenerationState, CsmModel,
    };

    /// Greedy argmax closure shared by the parity drivers.
    pub(crate) fn greedy(logits: &mut [f32]) -> u32 {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        best as u32
    }

    /// Device-parity driver: same synthesized weights on CPU and `backend`;
    /// asserts backbone hidden `atol ≤ 5e-4` and greedy frame codes
    /// identical over 3 frames. Returns `Ok(false)` (clean skip) when the
    /// device is unavailable at construction (FR-EX-08 explicit error —
    /// distinguished from a wrong result, which panics).
    pub(crate) fn run_parity(backend: BackendKind) -> Result<bool> {
        let cfg = CsmConfig::tiny_for_tests();
        // Backbone hidden parity.
        let cpu_bb = CsmBackbone::synthesized(cfg.clone(), 99)?;
        // Same weights, GPU-routed hot ops; the device check fires at the
        // first step's Compute construction (loud, FR-EX-08).
        let gpu_bb = CsmBackbone::synthesized(cfg.clone(), 99)?.with_backend(backend);
        let frames = [CsmFrame::text(1), CsmFrame::text(3)];
        let mut cpu_state = CsmBackboneState::new(&cfg)?;
        let mut gpu_state = CsmBackboneState::new(&cfg)?;
        let mut cpu_h = vec![0.0f32; cfg.backbone.d_model];
        let mut gpu_h = vec![0.0f32; cfg.backbone.d_model];
        for f in &frames {
            cpu_bb.step_into(&mut cpu_state, f, &mut cpu_h)?;
            match gpu_bb.step_into(&mut gpu_state, f, &mut gpu_h) {
                Ok(()) => {}
                Err(VokraError::BackendUnavailable(_)) => return Ok(false),
                Err(e) => return Err(e),
            }
        }
        let max_delta = cpu_h
            .iter()
            .zip(gpu_h.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_delta <= 5e-4,
            "{backend:?} backbone hidden max |Δ| = {max_delta} > 5e-4"
        );

        // Greedy frame-code identity on the combined model.
        let cpu_model = CsmModel::synthesized(cfg.clone(), 99)?;
        let gpu_model = CsmModel::synthesized(cfg.clone(), 99)?.with_backend(backend);
        let context = vec![CsmFrame::text(2), CsmFrame::text(5)];
        let mut cpu_gen = CsmGenerationState::new(&cfg)?;
        let mut gpu_gen = CsmGenerationState::new(&cfg)?;
        cpu_model.prime(&mut cpu_gen, &context)?;
        gpu_model.prime(&mut gpu_gen, &context)?;
        let mut cpu_codes = vec![0u32; cfg.n_codebooks];
        let mut gpu_codes = vec![0u32; cfg.n_codebooks];
        for _ in 0..3 {
            let a = cpu_model.generate_frame_into(&mut cpu_gen, &mut greedy, &mut cpu_codes)?;
            let b = gpu_model.generate_frame_into(&mut gpu_gen, &mut greedy, &mut gpu_codes)?;
            assert_eq!(a, b, "frame kind must agree");
            assert_eq!(
                cpu_codes, gpu_codes,
                "{backend:?} greedy codes must match CPU"
            );
        }
        Ok(true)
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
mod metal_band {
    use super::gpu_common::{greedy, run_parity};
    use vokra_core::{BackendKind, VokraError};
    use vokra_models::csm::{CsmConfig, CsmFrame, CsmMetalDecodeSession, CsmModel};

    #[test]
    fn metal_session_parity_or_clean_skip() {
        match vokra_models::csm::gpu_backend_probe(BackendKind::Metal) {
            Ok(()) => {}
            Err(VokraError::BackendUnavailable(_)) => {
                println!("skip: no Metal device on this host (clean skip, never fabricated)");
                return;
            }
            Err(e) => panic!("unexpected probe error: {e:?}"),
        }
        let ran = run_parity(BackendKind::Metal).expect("parity driver");
        assert!(ran, "probe said available but construction skipped");
    }

    #[test]
    fn metal_session_type_steps_and_resets() {
        let cfg = CsmConfig::tiny_for_tests();
        let model = CsmModel::synthesized(cfg.clone(), 7).unwrap();
        let session = match CsmMetalDecodeSession::new_from_model(model) {
            Ok(s) => s,
            Err(VokraError::BackendUnavailable(_)) => {
                println!("skip: no Metal device (clean skip)");
                return;
            }
            Err(e) => panic!("unexpected construction error: {e:?}"),
        };
        let mut session = session;
        session.prime(&[CsmFrame::text(1)]).unwrap();
        let mut codes = vec![0u32; cfg.n_codebooks];
        session.step(&mut greedy, &mut codes).unwrap();
        assert!(session.context_len() >= 1);
        session.reset();
        assert_eq!(session.context_len(), 0);
        assert_eq!(session.backend_name(), "metal");
    }
}

#[cfg(all(feature = "cuda", any(unix, windows)))]
mod cuda_band {
    use super::gpu_common::{greedy, run_parity};
    use vokra_core::{BackendKind, VokraError};
    use vokra_models::csm::{CsmConfig, CsmCudaDecodeSession, CsmFrame, CsmModel};

    #[test]
    fn cuda_session_parity_or_clean_skip() {
        match vokra_models::csm::gpu_backend_probe(BackendKind::Cuda) {
            Ok(()) => {}
            Err(VokraError::BackendUnavailable(_)) => {
                println!("skip: no CUDA driver/device on this host (clean skip)");
                return;
            }
            Err(e) => panic!("unexpected probe error: {e:?}"),
        }
        let ran = run_parity(BackendKind::Cuda).expect("parity driver");
        assert!(ran, "probe said available but construction skipped");
    }

    #[test]
    fn cuda_session_type_steps_and_resets() {
        let cfg = CsmConfig::tiny_for_tests();
        let model = CsmModel::synthesized(cfg.clone(), 7).unwrap();
        let mut session = match CsmCudaDecodeSession::new_from_model(model) {
            Ok(s) => s,
            Err(VokraError::BackendUnavailable(_)) => {
                println!("skip: no CUDA device (clean skip)");
                return;
            }
            Err(e) => panic!("unexpected construction error: {e:?}"),
        };
        session.prime(&[CsmFrame::text(1)]).unwrap();
        let mut codes = vec![0u32; cfg.n_codebooks];
        session.step(&mut greedy, &mut codes).unwrap();
        session.reset();
        assert_eq!(session.backend_name(), "cuda");
    }
}
