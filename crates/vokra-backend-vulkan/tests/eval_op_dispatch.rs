//! M3-02-T24 + M4-13-T09 — `VulkanBackend::eval_op` end-to-end on a live
//! Vulkan host (Linux + lavapipe, or a real GPU).
//!
//! On the Apple Mac authoring host the tests skip cleanly via the deliberate
//! [`VokraError::BackendUnavailable`] stub (no `libvulkan` here) — never a
//! silent CPU fall back (FR-EX-08 / NFR-RL-06).
//!
//! # Contract this test enforces
//!
//! 1. `supports(Copy) == true`, `supports(Add) == true` (hand-crafted blobs
//!    are always available).
//! 2. For the blob-gated arms (`MatMul` / `Mul` / `Softmax`, M4-13-T09):
//!    `supports(op)` equals blob availability of the op's backing shader,
//!    and `eval_op` agrees — Ok when the blob is loadable, explicit
//!    `UnsupportedOp` when not (lock-step BY MEASUREMENT, not by a
//!    hard-coded op list, so the owner's T16 blob commit flips both sides
//!    together with no test edit).
//! 3. Ops with no graph arm at all (`DcOffsetRemove`, `Stft`, …) are
//!    permanently `false` / `UnsupportedOp`.
//! 4. `eval_op(Copy)` returns the input bit-for-bit; `eval_op(Add)` returns
//!    the IEEE-754 f32 sum — including NON-multiples of the hand-crafted
//!    kernels' 64-lane workgroup (the M4-13-T09 padding fix; the M3-02 arms
//!    panicked on a `[2, 2]` tensor).

use vokra_backend_vulkan::{GemmPipelinePreference, VulkanBackend, graph_op_backing_shader, spirv};
use vokra_core::{Backend, OpKind, Tensor, VokraError};

/// `supports()` and `eval_op()` MUST advertise the exact same op set —
/// M3-02-T35 lock-step gate, blob-driven since M4-13-T09.
#[test]
fn supports_and_eval_op_are_lock_step() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan; skipping lock-step test");
        return;
    };
    // Hand-crafted-backed ops are always available.
    assert!(backend.supports(&OpKind::Copy), "Copy must be supported");
    assert!(backend.supports(&OpKind::Add), "Add must be supported");

    // Blob-gated arms: supports() must equal the backing shader's blob
    // availability, and eval_op must agree.
    let variant = backend
        .select_gemm_pipeline_variant(GemmPipelinePreference::default())
        .expect("default preference never errors");
    for op in [OpKind::MatMul, OpKind::Mul, OpKind::Softmax] {
        let shader = graph_op_backing_shader(&op, variant)
            .expect("covered graph op must have a backing shader");
        let expected = spirv::has_blob(shader);
        assert_eq!(
            backend.supports(&op),
            expected,
            "supports({op:?}) must track blob availability of `{shader}`"
        );
        // Valid small inputs for each arm.
        let a = Tensor::host_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let inputs: Vec<&Tensor> = match op {
            OpKind::Softmax => vec![&a],
            _ => vec![&a, &a],
        };
        match backend.eval_op(&op, &inputs) {
            Ok(outs) => {
                assert!(
                    expected,
                    "eval_op({op:?}) succeeded but supports() reported false — lock-step broken"
                );
                assert_eq!(outs.len(), 1);
                assert_eq!(outs[0].shape, vec![2, 2]);
            }
            Err(VokraError::UnsupportedOp(_)) => {
                assert!(
                    !expected,
                    "eval_op({op:?}) is UnsupportedOp but supports() reported true — lock-step \
                     broken"
                );
                eprintln!("{op:?}: blob `{shader}` absent → explicit UnsupportedOp (lock-step)");
            }
            Err(other) => panic!("unexpected error class for eval_op({op:?}): {other:?}"),
        }
    }

    // Ops with no graph arm at all: permanently unsupported.
    let a = Tensor::zeros_f32(vec![64]);
    assert!(!backend.supports(&OpKind::DcOffsetRemove));
    let err = backend.eval_op(&OpKind::DcOffsetRemove, &[&a]).unwrap_err();
    assert!(
        matches!(err, VokraError::UnsupportedOp(_)),
        "eval_op(DcOffsetRemove) must be UnsupportedOp, got {err:?}"
    );
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

/// M4-13-T09 bug fix: the M3-02 arms passed graph tensors straight into the
/// hand-crafted smoke kernels, whose impls ASSERT a multiple-of-64 length —
/// so `eval_op(Add)` on a `[2, 2]` tensor panicked on a live Vulkan host
/// (public API contract violation: Result, never a panic). The arms now
/// zero-pad + truncate; live regions stay bit-identical.
#[test]
fn eval_op_copy_and_add_handle_non_multiple_of_workgroup_lengths() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan; skipping non-multiple-length eval_op test");
        return;
    };
    // 2x2 (4 elements) and 5 elements — both << 64 and not multiples.
    let a = Tensor::host_f32(vec![2, 2], vec![1.5, -2.0, 0.25, 8.0]).unwrap();
    let outs = backend
        .eval_op(&OpKind::Copy, &[&a])
        .expect("Copy of a [2,2] tensor must not panic or error");
    assert_eq!(outs[0].shape, vec![2, 2]);
    for (want, got) in a.as_f32().unwrap().iter().zip(outs[0].as_f32().unwrap()) {
        assert_eq!(
            want.to_bits(),
            got.to_bits(),
            "padded Copy must stay bit-identical"
        );
    }

    let x = Tensor::host_f32(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y = Tensor::host_f32(vec![5], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
    let outs = backend
        .eval_op(&OpKind::Add, &[&x, &y])
        .expect("Add of length-5 tensors must not panic or error");
    let got = outs[0].as_f32().unwrap();
    assert_eq!(got.len(), 5, "readback truncated to the live region");
    for (i, ((a_i, b_i), g)) in x
        .as_f32()
        .unwrap()
        .iter()
        .zip(y.as_f32().unwrap())
        .zip(got)
        .enumerate()
    {
        assert_eq!((a_i + b_i).to_bits(), g.to_bits(), "Add diverged at {i}");
    }
}
