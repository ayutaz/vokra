//! M4-13-T03/T04 — typed kernel dispatch arms (`gemm_f32` / `gemv_f32`) on
//! [`VulkanBackend`], exercising the placeholder-then-swap seam end-to-end.
//!
//! Three-way behaviour, all asserted here:
//!
//! 1. **No Vulkan host** (Apple authoring host / default build): every test
//!    skips cleanly at `VulkanBackend::new()` — never a silent CPU fallback.
//! 2. **Vulkan host, blob not committed** (foundation slice — the state
//!    lavapipe CI runs in until the owner's M4-13-T16 glslc commit): a
//!    correctly-shaped dispatch surfaces the explicit
//!    [`VokraError::UnsupportedOp`] from `spirv::require_blob`, while a
//!    badly-shaped call still fails host-side with `InvalidArgument`
//!    *before* the blob check (validation precedes dispatch).
//! 3. **Vulkan host, blob committed** (post-T16): the same call computes,
//!    and is checked against the CPU backend's kernel of identical
//!    shape/semantics within the NFR-QL-01 FP32 gate (atol = 0.01).
//!
//! The branch between 2 and 3 is `spirv::has_blob` — the tests stay green
//! across the owner's blob commit without edits (no fabricated pass: which
//! branch ran is visible in the test log).

use vokra_backend_vulkan::{GemmPipelinePreference, VulkanBackend, spirv};
use vokra_core::VokraError;

/// FP32 parity gate (NFR-QL-01).
const ATOL: f32 = 0.01;

/// Deterministic pseudo-random f32s in [-1, 1) — SplitMix64-based, the same
/// synthesized-input pattern the M3-09 LLM parity tests use (no external
/// crates, NFR-DS-02).
fn splitmix_f32s(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            // 24 mantissa-ish bits → [-1, 1).
            ((z >> 40) as f32) / ((1u64 << 23) as f32) - 1.0
        })
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= ATOL,
            "{what}: diverged at {i}: got {g}, want {w} (atol {ATOL})"
        );
    }
}

/// `gemm_f32` through the ForceSubgroup path: blob-gated compute vs CPU
/// `gemm_f32`, or explicit `UnsupportedOp` while the blob is absent.
#[test]
fn gemm_subgroup_dispatch_is_blob_gated_and_cpu_close() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan on this host; skipping gemm dispatch test");
        return;
    };
    // Ragged dims exercise both tile tails (16x16 workgroups).
    let (m, n, k) = (17usize, 33usize, 9usize);
    let a = splitmix_f32s(1, m * k);
    let b = splitmix_f32s(2, k * n);

    let result = backend.gemm_f32(GemmPipelinePreference::ForceSubgroup, m, n, k, &a, &b);
    if spirv::has_blob("gemm_subgroup") {
        let got = result.expect("blob committed; gemm must dispatch");
        let mut want = vec![0.0f32; m * n];
        vokra_backend_cpu::kernels::gemm_f32(m, n, k, &a, &b, None, &mut want)
            .expect("CPU reference");
        assert_close(&got, &want, "gemm_subgroup vs CPU");
        eprintln!("gemm_subgroup parity vs CPU green over {m}x{n}x{k}");
    } else {
        let err = result.expect_err("no .spv committed; gemm must be UnsupportedOp");
        assert!(
            matches!(err, VokraError::UnsupportedOp(_)),
            "expected UnsupportedOp while gemm_subgroup.spv is absent, got {err:?}"
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("gemm_subgroup"),
            "diagnostic names the missing blob: {msg}"
        );
        eprintln!("gemm_subgroup blob absent → explicit UnsupportedOp (placeholder slice)");
    }
}

/// The default preference resolves to whatever variant the probe selected;
/// dispatch must route to exactly that variant's blob.
#[test]
fn gemm_default_preference_follows_probe_selection() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan on this host; skipping gemm variant-routing test");
        return;
    };
    let variant = backend.select_gemm_pipeline_variant(GemmPipelinePreference::default());
    let shader = variant.shader_name();
    let (m, n, k) = (4usize, 4usize, 4usize);
    let a = splitmix_f32s(3, m * k);
    let b = splitmix_f32s(4, k * n);
    let result = backend.gemm_f32(GemmPipelinePreference::default(), m, n, k, &a, &b);
    if spirv::has_blob(shader) {
        assert!(
            result.is_ok(),
            "selected variant `{shader}` blob present; must dispatch"
        );
    } else {
        let msg = format!("{}", result.expect_err("selected variant blob absent"));
        assert!(
            msg.contains(shader),
            "UnsupportedOp must name the *selected* variant `{shader}`: {msg}"
        );
    }
}

/// Host-side shape validation fires BEFORE the blob check / any GPU work —
/// even in the foundation slice with no blob committed.
#[test]
fn gemm_shape_mismatch_is_invalid_argument_before_blob_check() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan on this host; skipping gemm validation test");
        return;
    };
    let a = vec![0.0f32; 7]; // not m*k = 8
    let b = vec![0.0f32; 12];
    let err = backend
        .gemm_f32(GemmPipelinePreference::ForceSubgroup, 2, 3, 4, &a, &b)
        .expect_err("shape mismatch must error");
    assert!(
        matches!(err, VokraError::InvalidArgument(_)),
        "expected InvalidArgument (validation precedes dispatch), got {err:?}"
    );
}

/// `gemv_f32` with and without bias: blob-gated compute vs CPU `gemv_f32`,
/// or explicit `UnsupportedOp`.
#[test]
fn gemv_dispatch_is_blob_gated_and_cpu_close() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan on this host; skipping gemv dispatch test");
        return;
    };
    // m=37 rows (one workgroup each), n=100 exercises the strided inner loop
    // (100 > 32 lanes → multiple strides + ragged tail).
    let (m, n) = (37usize, 100usize);
    let a = splitmix_f32s(5, m * n);
    let x = splitmix_f32s(6, n);
    let bias = splitmix_f32s(7, m);

    for bias_arg in [None, Some(bias.as_slice())] {
        let result = backend.gemv_f32(m, n, &a, &x, bias_arg);
        if spirv::has_blob("gemv") {
            let got = result.expect("blob committed; gemv must dispatch");
            let mut want = vec![0.0f32; m];
            vokra_backend_cpu::kernels::gemv_f32(m, n, &a, &x, bias_arg, &mut want)
                .expect("CPU reference");
            assert_close(&got, &want, "gemv vs CPU");
        } else {
            let err = result.expect_err("no .spv committed; gemv must be UnsupportedOp");
            assert!(
                matches!(err, VokraError::UnsupportedOp(_)),
                "expected UnsupportedOp while gemv.spv is absent, got {err:?}"
            );
        }
    }
}

/// gemv shape validation (x length) fires host-side on a Vulkan host too.
#[test]
fn gemv_shape_mismatch_is_invalid_argument_before_blob_check() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan on this host; skipping gemv validation test");
        return;
    };
    let a = vec![0.0f32; 6];
    let x = vec![0.0f32; 2]; // A is 2x3 → x must be 3
    let err = backend
        .gemv_f32(2, 3, &a, &x, None)
        .expect_err("x length mismatch must error");
    assert!(matches!(err, VokraError::InvalidArgument(_)));
}
