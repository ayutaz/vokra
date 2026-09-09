//! SGMSE fixed `[1,3,3,1]` FIR Metal parity.
//!
//! This is a per-op device test, not a model run. It is intentionally gated
//! to Apple + the `metal` feature; the non-Metal build checks that the backend
//! coverage gate refuses the op rather than silently using CPU. Metal
//! execution is reserved for a disposable Scaleway Apple worker by policy.

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
mod off_feature {
    use vokra_core::{BackendKind, VokraError};
    use vokra_models::compute::{Compute, HotOp};

    #[test]
    fn fir_resample_metal_is_explicitly_unavailable_off_apple() {
        let Err(error) = Compute::for_backend(BackendKind::Metal, &[HotOp::FirResample2d]) else {
            panic!("off-feature Metal must not silently run FIR on the CPU");
        };
        assert!(matches!(error, VokraError::BackendUnavailable(_)));
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
mod metal_band {
    use vokra_core::{BackendKind, VokraError};
    use vokra_models::compute::{Compute, HotOp};

    // Fixed before observing a device result. The only expected difference is
    // FP32 multiply-add contraction in the Metal shader; this is deliberately
    // tighter than the model-level SGMSE parity envelope.
    const ATOL: f32 = 1.0e-5;

    fn max_delta(lhs: &[f32], rhs: &[f32]) -> f32 {
        lhs.iter()
            .zip(rhs)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    fn metal_compute() -> Compute {
        match Compute::for_backend(BackendKind::Metal, &[HotOp::FirResample2d]) {
            Ok(compute) => compute,
            Err(VokraError::BackendUnavailable(error)) => {
                panic!("Scaleway Metal worker must provide a Metal device: {error}")
            }
            Err(error) => panic!("unexpected Metal FIR backend error: {error}"),
        }
    }

    #[test]
    fn fixed_fir_up_and_down_match_cpu() {
        let metal = metal_compute();
        let cpu = Compute::cpu();
        let input: Vec<f32> = (0..2 * 8 * 8).map(|i| ((i as f32) - 31.5) / 17.0).collect();

        let mut cpu_up = vec![0.0; 2 * 16 * 16];
        let mut metal_up = vec![0.0; cpu_up.len()];
        cpu.fir_resample_2d_f32(&input, 2, 8, 8, true, &mut cpu_up)
            .unwrap();
        metal
            .fir_resample_2d_f32(&input, 2, 8, 8, true, &mut metal_up)
            .unwrap();
        assert!(max_delta(&cpu_up, &metal_up) <= ATOL);

        let mut cpu_down = vec![0.0; 2 * 4 * 4];
        let mut metal_down = vec![0.0; cpu_down.len()];
        cpu.fir_resample_2d_f32(&input, 2, 8, 8, false, &mut cpu_down)
            .unwrap();
        metal
            .fir_resample_2d_f32(&input, 2, 8, 8, false, &mut metal_down)
            .unwrap();
        assert!(max_delta(&cpu_down, &metal_down) <= ATOL);
    }
}
