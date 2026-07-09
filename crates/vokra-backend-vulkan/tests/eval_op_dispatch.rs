//! M3-02-T24 — `VulkanBackend::eval_op` end-to-end for `OpKind::Copy` and
//! `OpKind::Add` on a live Vulkan host (Linux + lavapipe, or a real GPU).
//!
//! On the Apple Mac authoring host the tests skip cleanly via the deliberate
//! [`VokraError::BackendUnavailable`] stub (no `libvulkan` here) — never a
//! silent CPU fall back (FR-EX-08 / NFR-RL-06).
//!
//! # Contract this test enforces
//!
//! 1. `VulkanBackend::supports(Copy) == true`, `supports(Add) == true`.
//! 2. `VulkanBackend::supports(<any other op>) == false`.
//! 3. `eval_op(Copy)` returns the input bit-for-bit (the shader body is
//!    `dst[i] = src[i]`).
//! 4. `eval_op(Add)` returns `a + b` under IEEE-754 f32 semantics.
//! 5. `eval_op(<uncovered op>)` is an explicit `UnsupportedOp` — never
//!    a silent CPU fallback.

use vokra_backend_vulkan::{VulkanBackend, spirv};
use vokra_core::{Backend, OpKind, Tensor, VokraError};

/// `supports()` and `eval_op()` MUST advertise the exact same op set —
/// M3-02-T35 lock-step gate. This test is host-independent (probes just the
/// coverage decision + the arity-check paths of `eval_op`, both of which run
/// before any GPU dispatch).
#[test]
fn supports_and_eval_op_are_lock_step() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan; skipping lock-step test");
        return;
    };
    // The covered ops.
    assert!(backend.supports(&OpKind::Copy), "Copy must be supported");
    assert!(backend.supports(&OpKind::Add), "Add must be supported");
    // Uncovered ops.
    for op in [OpKind::MatMul, OpKind::Mul, OpKind::Softmax] {
        assert!(
            !backend.supports(&op),
            "{op:?} must NOT be supported in the foundation slice"
        );
        // `eval_op` on an uncovered op is an explicit UnsupportedOp — never
        // a silent CPU fall back (FR-EX-08).
        let a = Tensor::zeros_f32(vec![2, 2]);
        let inputs: Vec<&Tensor> = match op {
            OpKind::Softmax => vec![&a],
            _ => vec![&a, &a],
        };
        let err = backend.eval_op(&op, &inputs).unwrap_err();
        assert!(
            matches!(err, VokraError::UnsupportedOp(_)),
            "eval_op({op:?}) must be UnsupportedOp, got {err:?}"
        );
    }
}

/// `eval_op(Copy)` end-to-end: upload → dispatch → download → verify
/// bit-identical. On a Vulkan host this exercises the full T08〜T12 + T25
/// dispatch chain plus the T24 dispatcher; on macOS the whole thing skips
/// via `BackendUnavailable`.
#[test]
fn eval_op_copy_round_trips_input_bit_for_bit() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan; skipping eval_op(Copy) round-trip");
        return;
    };
    let local = spirv::handcrafted_copy_f32::LOCAL_SIZE_X as usize;
    // 2 workgroups so `group_count_x = 2 > 1` — proves the dispatch math wires
    // through the T24 arm correctly.
    let n = 2 * local;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 3.25).collect();
    let x = Tensor::host_f32(vec![n], input.clone()).unwrap();

    let outs = backend
        .eval_op(&OpKind::Copy, &[&x])
        .expect("eval_op(Copy) must succeed on a Vulkan host");
    assert_eq!(outs.len(), 1, "Copy produces exactly 1 output");
    let y = &outs[0];
    assert_eq!(y.shape, x.shape, "Copy preserves shape");
    let got = y.as_f32().unwrap();
    assert_eq!(got.len(), input.len());
    for (i, (want, got)) in input.iter().zip(got).enumerate() {
        assert_eq!(
            want.to_bits(),
            got.to_bits(),
            "GPU Copy diverged at index {i}: want {want} (bits {:#x}), got {got} (bits {:#x})",
            want.to_bits(),
            got.to_bits(),
        );
    }
    eprintln!("eval_op(Copy) bit-identical round trip over {n} f32s");
}

/// `eval_op(Add)` end-to-end: `c = a + b`, IEEE-754 f32. Also exercises the
/// 3-SSBO layout — the M3-02-T24 case `copy_f32` didn't cover.
#[test]
fn eval_op_add_matches_host_ieee754_sum() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan; skipping eval_op(Add) round-trip");
        return;
    };
    let local = spirv::handcrafted_add_f32::LOCAL_SIZE_X as usize;
    let n = 2 * local;
    let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25).collect();
    let b: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.125 + 1.0).collect();
    let ta = Tensor::host_f32(vec![n], a.clone()).unwrap();
    let tb = Tensor::host_f32(vec![n], b.clone()).unwrap();

    let outs = backend
        .eval_op(&OpKind::Add, &[&ta, &tb])
        .expect("eval_op(Add) must succeed on a Vulkan host");
    assert_eq!(outs.len(), 1, "Add produces exactly 1 output");
    let c = &outs[0];
    assert_eq!(c.shape, ta.shape, "Add preserves shape");
    let got = c.as_f32().unwrap();
    assert_eq!(got.len(), a.len());
    for (i, ((a_i, b_i), got_i)) in a.iter().zip(&b).zip(got).enumerate() {
        // IEEE-754 f32 `a + b` on the GPU matches the host sum bit-for-bit
        // for these small, finite inputs (no denormals, no NaNs, no overflow).
        let host_sum = a_i + b_i;
        assert_eq!(
            host_sum.to_bits(),
            got_i.to_bits(),
            "GPU Add diverged at index {i}: host_sum={host_sum} got={got_i}"
        );
    }
    eprintln!("eval_op(Add) matched host IEEE-754 sum over {n} f32s");
}

/// A copy of a single workgroup — boundary case for `group_count_x = 1`.
#[test]
fn eval_op_copy_single_workgroup_boundary() {
    let Ok(backend) = VulkanBackend::new() else {
        return;
    };
    let local = spirv::handcrafted_copy_f32::LOCAL_SIZE_X as usize;
    let input: Vec<f32> = (0..local).map(|i| (i as f32).sin()).collect();
    let x = Tensor::host_f32(vec![local], input.clone()).unwrap();
    let outs = backend.eval_op(&OpKind::Copy, &[&x]).unwrap();
    let y = outs[0].as_f32().unwrap();
    for (want, got) in input.iter().zip(y) {
        assert_eq!(want.to_bits(), got.to_bits());
    }
}

/// Arity / shape validation runs *before* any GPU dispatch, so these errors
/// fire on every host including macOS (where `VulkanBackend::new()` fails).
///
/// We can't build a real `VulkanBackend` off Vulkan, but we can call
/// `eval_vulkan_op` through the trait — the arity check is inside
/// `eval_vulkan_op`, so we need a Vulkan host to reach it. Off Vulkan we
/// simply skip this test.
#[test]
fn eval_op_arity_and_shape_errors_fire_before_gpu_dispatch() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan; skipping arity/shape validation test");
        return;
    };
    // Copy with 2 inputs → InvalidArgument.
    let a = Tensor::zeros_f32(vec![4]);
    let err = backend.eval_op(&OpKind::Copy, &[&a, &a]).unwrap_err();
    assert!(
        matches!(err, VokraError::InvalidArgument(_)),
        "Copy with 2 inputs must be InvalidArgument, got {err:?}"
    );
    // Add with shape mismatch → InvalidArgument.
    let b = Tensor::zeros_f32(vec![8]);
    let err = backend.eval_op(&OpKind::Add, &[&a, &b]).unwrap_err();
    assert!(
        matches!(err, VokraError::InvalidArgument(_)),
        "Add shape mismatch must be InvalidArgument, got {err:?}"
    );
    // Add with 1 input → InvalidArgument.
    let err = backend.eval_op(&OpKind::Add, &[&a]).unwrap_err();
    assert!(
        matches!(err, VokraError::InvalidArgument(_)),
        "Add with 1 input must be InvalidArgument, got {err:?}"
    );
}

/// Off-target (macOS / iOS / WASM / default-features): the backend
/// construction itself fails with `BackendUnavailable`. This locks in the
/// no-silent-fallback contract for the whole `impl Backend` surface.
#[cfg(not(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
)))]
#[test]
fn backend_construction_is_unavailable_off_target() {
    match VulkanBackend::new() {
        Err(VokraError::BackendUnavailable(_)) => {}
        Ok(_) => panic!("VulkanBackend::new() must fail off-target / feature off"),
        Err(other) => panic!("expected BackendUnavailable, got {other}"),
    }
}
