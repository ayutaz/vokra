//! M3-02-T13 / ADR M3-02-spirv-generation §4 (d) — end-to-end proof that the
//! Vulkan T08〜T12 + T25 object stack actually works, on top of the
//! hand-crafted `copy_f32` SPIR-V kernel.
//!
//! The `copy_f32` SPIR-V body is the smallest possible compute shader
//! (`dst[i] = src[i]`); its role here is to prove:
//!
//! - The Vulkan loader loads and creates an instance / device / queue.
//! - Host-visible + device-local buffers allocate + bind + copy correctly
//!   (T12 + T25).
//! - The descriptor set layout + pool + set + `vkUpdateDescriptorSets` chain
//!   binds SSBOs correctly (T10).
//! - The pipeline layout + shader module + compute pipeline chain accepts a
//!   SPIR-V blob (T11).
//! - `vkCmdBindPipeline` + `vkCmdBindDescriptorSets` + `vkCmdDispatch`
//!   + fence-sync submission dispatches the kernel and produces the
//!     expected memory result.
//!
//! On the Apple Mac authoring host (no `libvulkan`) the test skips cleanly
//! via the deliberate [`VokraError::BackendUnavailable`] stub — never a
//! silent CPU fall back (FR-EX-08 / NFR-RL-06). On Linux + lavapipe (the CI
//! runner target) it runs the full round-trip.

use vokra_backend_vulkan::{smoke_dispatch_add_f32, smoke_dispatch_copy_f32, spirv};
use vokra_core::VokraError;

/// A working Vulkan host returns the input bit-for-bit; a Vulkan-less host
/// returns [`VokraError::BackendUnavailable`], which we log and treat as
/// "not applicable" (the whole point of `BackendUnavailable`: no silent CPU
/// substitution, FR-EX-08).
#[test]
fn copy_f32_kernel_round_trips_a_multi_workgroup_input() {
    // 128 elements = 2 × LOCAL_SIZE_X = 2 workgroups. Big enough that
    // `group_count_x = 2 > 1` — proves the workgroup dispatch math wires
    // through correctly.
    let local = spirv::handcrafted_copy_f32::LOCAL_SIZE_X as usize;
    let n = 2 * local;
    // A distinctive, non-zero pattern so a "zero-fill by mistake" bug shows.
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 3.25).collect();

    match smoke_dispatch_copy_f32(&input) {
        Ok(output) => {
            assert_eq!(
                output.len(),
                input.len(),
                "GPU output length must match input length"
            );
            for (i, (want, got)) in input.iter().zip(output.iter()).enumerate() {
                assert_eq!(
                    want.to_bits(),
                    got.to_bits(),
                    "GPU output diverged at index {i}: want {want} (bits {:#x}), got {got} \
                     (bits {:#x}) — the hand-crafted `copy_f32` shader must be bit-identical \
                     to the input; a mismatch means the SPIR-V module encoding or the \
                     descriptor-set binding is subtly wrong",
                    want.to_bits(),
                    got.to_bits(),
                );
            }
            eprintln!(
                "smoke_dispatch_copy_f32: bit-identical round trip over {n} f32s (2 workgroups)"
            );
        }
        Err(VokraError::BackendUnavailable(msg)) => {
            // Apple Mac authoring host / any host without libvulkan / any
            // build with --features cpu (default). Clean skip.
            eprintln!("smoke_dispatch_copy_f32 unavailable (expected off Vulkan): {msg}");
        }
        Err(other) => panic!(
            "smoke_dispatch_copy_f32 returned an unexpected error kind: {other} — a Vulkan-host \
             failure must surface as BackendUnavailable (missing loader / ICD) or a driver-side \
             error worth investigating; a `NotImplemented`/`UnsupportedOp` at this level is a \
             regression"
        ),
    }
}

/// A single workgroup (`N == LOCAL_SIZE_X`) still round-trips correctly, and
/// the boundary case exercises `group_count_x = 1` explicitly.
#[test]
fn copy_f32_kernel_handles_single_workgroup() {
    let local = spirv::handcrafted_copy_f32::LOCAL_SIZE_X as usize;
    let input: Vec<f32> = (0..local).map(|i| (i as f32).sin()).collect();
    match smoke_dispatch_copy_f32(&input) {
        Ok(output) => {
            assert_eq!(output.len(), local);
            for (i, (want, got)) in input.iter().zip(output.iter()).enumerate() {
                assert_eq!(want.to_bits(), got.to_bits(), "mismatch at index {i}");
            }
        }
        Err(VokraError::BackendUnavailable(_)) => {
            eprintln!("skipping copy_f32 single-workgroup smoke — no Vulkan host");
        }
        Err(other) => panic!("unexpected: {other}"),
    }
}

/// Empty input is the trivial pass-through on a Vulkan host (the impl
/// short-circuits before any dispatch). Off Vulkan targets the stub still
/// surfaces `BackendUnavailable` — either outcome is honest; both prove
/// there is no panic on the empty case.
#[test]
fn copy_f32_kernel_handles_empty_input_without_panic() {
    match smoke_dispatch_copy_f32(&[]) {
        Ok(out) => assert!(out.is_empty(), "empty in => empty out (no dispatch)"),
        Err(VokraError::BackendUnavailable(_)) => {
            eprintln!("empty-input smoke skipped (no Vulkan host)");
        }
        Err(other) => panic!("unexpected error for empty input: {other}"),
    }
}

/// On the non-Vulkan build target (any macOS / iOS / WASM host, or the
/// default-features build without `vulkan`), the smoke dispatch surfaces
/// `BackendUnavailable`. This test locks in that contract so the "no silent
/// CPU fall back" invariant survives future refactors.
#[cfg(not(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
)))]
#[test]
fn copy_f32_kernel_stub_returns_backend_unavailable_off_target() {
    let local = spirv::handcrafted_copy_f32::LOCAL_SIZE_X as usize;
    let input: Vec<f32> = vec![1.0; local];
    match smoke_dispatch_copy_f32(&input) {
        Err(VokraError::BackendUnavailable(_)) => {}
        Ok(_) => panic!("stub must not succeed off Vulkan targets / feature off"),
        Err(other) => panic!("expected BackendUnavailable, got {other}"),
    }
}

/// Sanity: the manifest / handcrafted module co-locate on the same
/// `copy_f32` name — a rename must not silently break the smoke path.
#[test]
fn manifest_and_handcrafted_agree_on_copy_f32() {
    let entry = spirv::SHADERS
        .iter()
        .find(|s| s.name == "copy_f32")
        .expect("SHADERS manifest must include `copy_f32`");
    assert!(
        matches!(entry.variant, spirv::ShaderVariant::Handcrafted),
        "copy_f32 must be Handcrafted, was {}",
        entry.variant
    );
    // The pinned hash is what `verify_pinned_hashes` checks; assert it exists
    // so a future refactor cannot silently un-pin the smoke blob.
    assert!(
        entry.expected_sha256_hex.is_some(),
        "copy_f32 must have a pinned SHA-256"
    );
}

// ---------------------------------------------------------------------------
// add_f32 kernel smoke tests (M3-02-T24 partial). Same shape as the copy_f32
// tests above — a Vulkan-less host skips cleanly (BackendUnavailable), and
// on a Vulkan-capable host every element of the GPU-computed sum matches the
// host IEEE-754 f32 sum bit-for-bit.
// ---------------------------------------------------------------------------

/// Multi-workgroup add-round-trip: two workgroups so the dispatch math is
/// exercised for `group_count_x = 2 > 1`, and the GPU output matches the
/// host sum bit-for-bit.
#[test]
fn add_f32_kernel_matches_host_sum_over_multi_workgroup() {
    let local = spirv::handcrafted_add_f32::LOCAL_SIZE_X as usize;
    let n = 2 * local;
    // Distinctive, finite, no-denormals patterns so a zero-fill bug shows.
    let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 3.25).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32) * -0.125 + 1.75).collect();

    match smoke_dispatch_add_f32(&a, &b) {
        Ok(output) => {
            assert_eq!(
                output.len(),
                a.len(),
                "GPU output length must match input length"
            );
            for (i, ((av, bv), gv)) in a.iter().zip(&b).zip(output.iter()).enumerate() {
                let host_sum = av + bv;
                assert_eq!(
                    host_sum.to_bits(),
                    gv.to_bits(),
                    "GPU output diverged at index {i}: host_sum {host_sum} (bits {:#x}) vs GPU \
                     {gv} (bits {:#x}) — the hand-crafted `add_f32` shader must reproduce IEEE-754 \
                     f32 add bit-for-bit for finite inputs; a mismatch usually points at wrong \
                     3-SSBO descriptor binding order (a / b / c ordering)",
                    host_sum.to_bits(),
                    gv.to_bits(),
                );
            }
            eprintln!(
                "smoke_dispatch_add_f32: bit-identical IEEE-754 sum over {n} f32s (2 workgroups)"
            );
        }
        Err(VokraError::BackendUnavailable(msg)) => {
            eprintln!("smoke_dispatch_add_f32 unavailable (expected off Vulkan): {msg}");
        }
        Err(other) => panic!(
            "smoke_dispatch_add_f32 returned an unexpected error kind: {other} — a Vulkan-host \
             failure must surface as BackendUnavailable (missing loader / ICD) or a driver-side \
             error worth investigating"
        ),
    }
}

/// Single-workgroup boundary — `N == LOCAL_SIZE_X`, `group_count_x = 1`.
#[test]
fn add_f32_kernel_handles_single_workgroup() {
    let local = spirv::handcrafted_add_f32::LOCAL_SIZE_X as usize;
    let a: Vec<f32> = (0..local).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..local).map(|i| (i as f32).cos()).collect();
    match smoke_dispatch_add_f32(&a, &b) {
        Ok(output) => {
            assert_eq!(output.len(), local);
            for (i, ((av, bv), gv)) in a.iter().zip(&b).zip(output.iter()).enumerate() {
                let host = av + bv;
                assert_eq!(
                    host.to_bits(),
                    gv.to_bits(),
                    "mismatch at index {i}: host={host} gpu={gv}"
                );
            }
        }
        Err(VokraError::BackendUnavailable(_)) => {
            eprintln!("skipping add_f32 single-workgroup smoke — no Vulkan host");
        }
        Err(other) => panic!("unexpected: {other}"),
    }
}

/// Empty input is the trivial pass-through on a Vulkan host. Off Vulkan
/// targets the stub still surfaces `BackendUnavailable`.
#[test]
fn add_f32_kernel_handles_empty_input_without_panic() {
    match smoke_dispatch_add_f32(&[], &[]) {
        Ok(out) => assert!(out.is_empty(), "empty in => empty out (no dispatch)"),
        Err(VokraError::BackendUnavailable(_)) => {
            eprintln!("empty-input add smoke skipped (no Vulkan host)");
        }
        Err(other) => panic!("unexpected error for empty input: {other}"),
    }
}

/// Contract lock-in for the non-Vulkan build target: `smoke_dispatch_add_f32`
/// surfaces `BackendUnavailable`, never silently uses the CPU. Mirrors the
/// existing copy_f32 stub test — if this compiles, both kernels honour the
/// FR-EX-08 contract at the public API boundary.
#[cfg(not(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
)))]
#[test]
fn add_f32_kernel_stub_returns_backend_unavailable_off_target() {
    let local = spirv::handcrafted_add_f32::LOCAL_SIZE_X as usize;
    let a: Vec<f32> = vec![1.0; local];
    let b: Vec<f32> = vec![2.0; local];
    match smoke_dispatch_add_f32(&a, &b) {
        Err(VokraError::BackendUnavailable(_)) => {}
        Ok(_) => panic!("stub must not succeed off Vulkan targets / feature off"),
        Err(other) => panic!("expected BackendUnavailable, got {other}"),
    }
}

/// The `add_f32` manifest entry mirrors the same invariants as `copy_f32`.
#[test]
fn manifest_and_handcrafted_agree_on_add_f32() {
    let entry = spirv::SHADERS
        .iter()
        .find(|s| s.name == "add_f32")
        .expect("SHADERS manifest must include `add_f32`");
    assert!(
        matches!(entry.variant, spirv::ShaderVariant::Handcrafted),
        "add_f32 must be Handcrafted, was {}",
        entry.variant
    );
    assert!(
        entry.expected_sha256_hex.is_some(),
        "add_f32 must have a pinned SHA-256"
    );
}
