//! M4-07-T02: NVRTC `compute_90a` feasibility probe (gated).
//!
//! Answers the three boundary questions the FA v3 kernel work depends on —
//! **before** any full-kernel debugging (scout risk "NVRTC の sm_90a WGMMA
//! 対応境界"):
//!
//! (i)  NVRTC accepts `--gpu-architecture=compute_90a` and yields PTX;
//! (ii) inline-PTX `wgmma.mma_async` passes NVRTC;
//! (iii) the same source under `compute_89` fails at WHICH stage (NVRTC
//!       compile, or deferred to module load) — recorded, not asserted: the
//!       observation feeds the T01 ADR either way.
//!
//! NVRTC needs **no GPU** (it stops at PTX text), but it does need the CUDA
//! toolkit's `libnvrtc`. On a host without it (e.g. the Apple-silicon
//! authoring machine) every test here prints the reason and returns green —
//! a clean skip, never a fabricated pass: the actual (i)/(ii) proof happens
//! on a CUDA-toolkit host (vast.ai RTX 4090 route, or the owner's H100 run,
//! M4-07-T17) and its findings are appended to the ADR §(b).

#![cfg(any(unix, windows))]

use vokra_backend_cuda::{FA_V3_FEASIBILITY_SNIPPET, KERNELS_CUDA_FA_V3, nvrtc_compile_for_arch};
use vokra_core::VokraError;

/// Returns `true` when NVRTC is absent on this host (clean-skip condition).
/// Any *other* error from a trivial compile is a real failure and panics.
fn nvrtc_absent() -> bool {
    const TRIVIAL: &str = "extern \"C\" __global__ void vokra_t02_trivial() {}";
    match nvrtc_compile_for_arch(TRIVIAL, "compute_89") {
        Ok(_) => false,
        Err(VokraError::BackendUnavailable(msg)) if msg.contains("libnvrtc") => {
            eprintln!("skipping: NVRTC not installed on this host ({msg})");
            true
        }
        Err(other) => panic!("trivial compute_89 compile must succeed where NVRTC exists: {other}"),
    }
}

/// (i) + (ii): the minimal WGMMA snippet compiles under `compute_90a` and the
/// generated PTX actually carries the wgmma instruction (not silently
/// dropped/rewritten).
#[test]
fn fa_v3_snippet_compiles_for_compute_90a() {
    if nvrtc_absent() {
        return;
    }
    let ptx = nvrtc_compile_for_arch(FA_V3_FEASIBILITY_SNIPPET, "compute_90a")
        .expect("(i)/(ii): NVRTC must accept compute_90a + inline-PTX wgmma");
    let text = String::from_utf8_lossy(&ptx);
    assert!(
        text.contains("wgmma.mma_async"),
        "(ii): the emitted PTX must contain the wgmma.mma_async instruction"
    );
    assert!(
        text.contains("cvt.rna.tf32.f32"),
        "the explicit tf32 rounding conversion must survive into the PTX"
    );
    eprintln!(
        "fa_v3 feasibility: compute_90a OK, PTX {} bytes, wgmma.mma_async present",
        ptx.len()
    );
}

/// The full FA v3 program compiles under `compute_90a` (compile-only green
/// for T05–T08; runtime correctness is the H100-gated parity suite, T12/T17).
#[test]
fn fa_v3_full_program_compiles_for_compute_90a() {
    if nvrtc_absent() {
        return;
    }
    let ptx = nvrtc_compile_for_arch(KERNELS_CUDA_FA_V3, "compute_90a")
        .expect("KERNELS_CUDA_FA_V3 must compile for compute_90a");
    let text = String::from_utf8_lossy(&ptx);
    assert!(text.contains("wgmma.mma_async"));
    eprintln!(
        "fa_v3 full program: compute_90a OK, PTX {} bytes",
        ptx.len()
    );
}

/// (iii) — recorded, not asserted: compiling the WGMMA snippet for
/// `compute_89` shows where the arch check bites. Both outcomes are valid
/// findings for the ADR:
///   * `Err` here → NVRTC rejects wgmma at compile time for non-sm_90a
///     targets (arch check is front-loaded; module load never sees it);
///   * `Ok` here → the arch check is deferred to `cuModuleLoadData` (the
///     Hopper-only lazy-compile gate in `CudaContext::fa_v3_slot` is then
///     the load-time firewall).
///
/// The Hopper-only gate is required in either case; this probe documents
/// which stage the failure surfaces at so the ADR does not have to guess.
#[test]
fn fa_v3_snippet_under_compute_89_records_failure_stage() {
    if nvrtc_absent() {
        return;
    }
    match nvrtc_compile_for_arch(FA_V3_FEASIBILITY_SNIPPET, "compute_89") {
        Ok(ptx) => eprintln!(
            "(iii) FINDING for ADR: NVRTC ACCEPTED wgmma under compute_89 ({} B PTX) — \
             arch check is deferred to module load; the SM>=9.0 lazy-compile gate is the \
             load-time firewall",
            ptx.len()
        ),
        Err(e) => eprintln!(
            "(iii) FINDING for ADR: NVRTC REJECTED wgmma under compute_89 at compile time — \
             front-loaded arch check. Error: {e}"
        ),
    }
}

/// NUL-safety negatives of the diagnostic entry — CUDA-less green (both
/// reject *before* any NVRTC load can matter, at the CString boundary).
/// Note the `VOKRA_NVRTC_GPU_ARCH` non-applicability to FA v3 is structural
/// (the FA v3 path builds its arch from a literal and this entry takes it as
/// an explicit parameter — no env read exists on either path); the constant
/// is locked by the `fa_v3_gencode_is_fixed_90a` unit test in `fa_v3.rs`.
#[test]
fn nvrtc_compile_for_arch_rejects_interior_nul() {
    match nvrtc_compile_for_arch("extern \"C\" __global__ void k() {}", "compute_9\0a") {
        Err(VokraError::InvalidArgument(msg)) => {
            assert!(
                msg.contains("NUL"),
                "error must name the NUL problem: {msg}"
            )
        }
        other => panic!("interior-NUL arch must be InvalidArgument, got {other:?}"),
    }
    match nvrtc_compile_for_arch("extern \"C\" __global__ void k() {\0}", "compute_90a") {
        // Source NUL is caught at the source CString (InvalidArgument) — or,
        // on a host with no NVRTC, the library load fails first
        // (BackendUnavailable). Both are explicit, neither is a crash.
        Err(VokraError::InvalidArgument(_)) | Err(VokraError::BackendUnavailable(_)) => {}
        other => panic!("interior-NUL source must be an explicit error, got {other:?}"),
    }
}
