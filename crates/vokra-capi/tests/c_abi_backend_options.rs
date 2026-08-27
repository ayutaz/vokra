//! C ABI tests for GPU backend selection through an opaque options object
//! (design `docs/superpowers/specs/2026-08-14-c-abi-backend-speaker-design.md`,
//! §3 new symbols / §5 error handling / §6 T1-T6).
//!
//! # Why this file talks to the ABI, not to Rust
//!
//! Every `vokra-capi` module is private (`mod session;`, not `pub mod`), so an
//! integration test cannot reach the implementation through Rust paths. That is
//! deliberate: Unity / Godot / Python / Swift / Kotlin bindings only ever see
//! `include/vokra.h`, so these tests declare the symbols exactly as a C caller
//! would and exercise the linked `extern "C"` surface. Two mechanics are
//! required for that and are easy to lose in a refactor:
//!
//! - `extern crate vokra;` forces the rlib to be linked; without it the
//!   `#[unsafe(no_mangle)]` symbols are never pulled in and the test fails at
//!   link time with `undefined symbol: vokra_version`.
//! - `#![allow(unsafe_code)]` is needed per file: the workspace sets
//!   `unsafe_code = "deny"` and `crates/vokra-capi/src/lib.rs`'s crate-level
//!   allow does not reach a separate integration-test crate.
//!
//! # Enum parameters are declared as `i32` on purpose
//!
//! A field-less `#[repr(C)]` enum has the C ABI of `int` on every supported
//! target, so `i32` is layout-identical to `enum vokra_backend_t` /
//! `enum vokra_status_t`. Declaring them as plain integers is what lets T6 pass
//! an *out-of-range* discriminant (99, -1, `i32::MAX`) the way a C caller can,
//! without constructing an invalid Rust enum value (which would be UB).
//! [`header_declares_backend_enum_and_options_handle`] pins the mirrored
//! constants against the generated header so this mirror cannot silently drift.
//!
//! # Scope
//!
//! T1-T6 (backend selection + options lifecycle). The speaker surface
//! (§3.3 `vokra_speaker_embed` / `vokra_speaker_verify`, design T7-T9) is a
//! separate wave and is not covered here.

#![allow(unsafe_code)]

extern crate vokra;

use std::ffi::{CStr, CString, c_char, c_void};
use std::path::PathBuf;
use std::ptr;

// ---------------------------------------------------------------------------
// C ABI mirror
// ---------------------------------------------------------------------------

unsafe extern "C" {
    // --- Existing v1.0-rc baseline symbols (already in include/vokra.h) ---
    fn vokra_last_error() -> *const c_char;
    fn vokra_session_create_from_file(
        path_utf8: *const c_char,
        out_session: *mut *mut c_void,
    ) -> i32;
    fn vokra_session_create_from_bytes(
        data: *const u8,
        len: usize,
        out_session: *mut *mut c_void,
    ) -> i32;
    fn vokra_session_destroy(session: *mut c_void);
    fn vokra_stream_open(
        session: *const c_void,
        sample_rate: i32,
        out_stream: *mut *mut c_void,
    ) -> i32;
    fn vokra_stream_push_pcm(stream: *mut c_void, pcm: *const f32, num_samples: usize) -> i32;
    fn vokra_stream_poll(
        stream: *mut c_void,
        out_probs: *mut f32,
        capacity: usize,
        out_count: *mut usize,
    ) -> i32;
    fn vokra_stream_destroy(stream: *mut c_void);

    // --- New symbols under test (design §3.3). These do not exist yet. ---
    fn vokra_backend_available(backend: i32) -> bool;
    fn vokra_session_options_create() -> *mut c_void;
    fn vokra_session_options_destroy(opts: *mut c_void);
    fn vokra_session_options_set_backend(opts: *mut c_void, backend: i32) -> i32;
    fn vokra_session_create_from_file_with_options(
        path_utf8: *const c_char,
        opts: *const c_void,
        out_session: *mut *mut c_void,
    ) -> i32;
    fn vokra_session_create_from_bytes_with_options(
        data: *const u8,
        len: usize,
        opts: *const c_void,
        out_session: *mut *mut c_void,
    ) -> i32;
}

// `vokra_status_t` values, pinned numerically by
// `crates/vokra-capi/src/error.rs::status_codes_pin_numeric_abi` and re-checked
// against the generated header below.
const VOKRA_OK: i32 = 0;
const VOKRA_ERROR_UNSUPPORTED_OP: i32 = 3;
const VOKRA_ERROR_BACKEND_UNAVAILABLE: i32 = 4;
const VOKRA_ERROR_INVALID_ARGUMENT: i32 = 5;

// `vokra_backend_t` values (design §3.1). CoreML / QNN are deliberately *not*
// assigned values (D1: the delegate selector lands as a new symbol after the
// real-hardware ANE/Hexagon bakeoff, `docs/handoff/m4-12.md` §(e)-3).
const VOKRA_BACKEND_CPU: i32 = 0;
const VOKRA_BACKEND_METAL: i32 = 1;
const VOKRA_BACKEND_CUDA: i32 = 2;
const VOKRA_BACKEND_VULKAN: i32 = 3;
const VOKRA_BACKEND_WEBGPU: i32 = 4;

/// Every non-CPU backend in the exposed enum, in discriminant order.
const GPU_BACKENDS: [i32; 4] = [
    VOKRA_BACKEND_METAL,
    VOKRA_BACKEND_CUDA,
    VOKRA_BACKEND_VULKAN,
    VOKRA_BACKEND_WEBGPU,
];

fn backend_name(backend: i32) -> &'static str {
    match backend {
        VOKRA_BACKEND_CPU => "CPU",
        VOKRA_BACKEND_METAL => "METAL",
        VOKRA_BACKEND_CUDA => "CUDA",
        VOKRA_BACKEND_VULKAN => "VULKAN",
        VOKRA_BACKEND_WEBGPU => "WEBGPU",
        _ => "<out-of-range>",
    }
}

fn status_name(status: i32) -> &'static str {
    match status {
        VOKRA_OK => "VOKRA_OK",
        1 => "VOKRA_ERROR_IO",
        2 => "VOKRA_ERROR_MODEL_LOAD",
        VOKRA_ERROR_UNSUPPORTED_OP => "VOKRA_ERROR_UNSUPPORTED_OP",
        VOKRA_ERROR_BACKEND_UNAVAILABLE => "VOKRA_ERROR_BACKEND_UNAVAILABLE",
        VOKRA_ERROR_INVALID_ARGUMENT => "VOKRA_ERROR_INVALID_ARGUMENT",
        6 => "VOKRA_ERROR_GRAPH_VALIDATION",
        7 => "VOKRA_ERROR_NOT_IMPLEMENTED",
        8 => "VOKRA_ERROR_PANIC",
        9 => "VOKRA_ERROR_OTHER",
        _ => "<unknown status>",
    }
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The committed 2 MB Silero VAD v5 GGUF (M0-05 parity asset). Real weights, no
/// environment gate — the same fixture `tests/capi/smoke_vad.c` runs on.
fn silero_fixture() -> PathBuf {
    repo_root().join("tests/parity/silero_vad/silero-vad-v5.gguf")
}

/// 10 240 mono f32 samples @ 16 kHz (= 20 Silero frames of 512).
fn vad_input_fixture() -> PathBuf {
    repo_root().join("tests/capi/fixtures/vad_input_16k.f32")
}

fn read_f32_fixture(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("committed fixture {} must be readable: {e}", path.display()));
    assert_eq!(
        bytes.len() % 4,
        0,
        "{} is not a whole number of f32 samples",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn silero_path_cstring() -> CString {
    let path = silero_fixture();
    assert!(
        path.is_file(),
        "committed Silero fixture missing at {}",
        path.display()
    );
    CString::new(path.to_str().expect("fixture path is UTF-8")).expect("path has no interior NUL")
}

/// The calling thread's last recorded error message, if any.
fn last_error() -> Option<String> {
    // SAFETY: `vokra_last_error` returns either NULL or a NUL-terminated string
    // owned by Vokra and valid until this thread records its next error. The
    // borrow ends before this function returns.
    unsafe {
        let ptr = vokra_last_error();
        if ptr.is_null() {
            None
        } else {
            Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
        }
    }
}

// ---------------------------------------------------------------------------
// Observable model output: the real oracle for "same session"
// ---------------------------------------------------------------------------

/// Runs the committed VAD fixture through `session` and returns every speech
/// probability the stream produced.
///
/// This is the observable behaviour a C caller can compare. T2 / T4 assert the
/// options-based constructors reproduce it **bit-for-bit** against the legacy
/// constructors — same weights, same CPU backend, deterministic stepper — so a
/// merely-`VOKRA_OK` status cannot pass for equivalence.
fn vad_probs(session: *mut c_void) -> Vec<f32> {
    let pcm = read_f32_fixture(&vad_input_fixture());
    assert!(!pcm.is_empty(), "VAD fixture is empty");

    let mut stream: *mut c_void = ptr::null_mut();
    // SAFETY: `session` is a live handle owned by the caller; `stream` is a
    // writable out-slot.
    let st = unsafe { vokra_stream_open(session.cast_const(), 16_000, &mut stream) };
    assert_eq!(
        st,
        VOKRA_OK,
        "vokra_stream_open failed: {} ({:?})",
        status_name(st),
        last_error()
    );
    assert!(!stream.is_null(), "stream handle is NULL after VOKRA_OK");

    let mut probs = Vec::new();
    let mut offset = 0usize;
    while offset < pcm.len() {
        let chunk = (pcm.len() - offset).min(2048);
        // SAFETY: `stream` is live; `pcm[offset..]` is valid for `chunk` reads.
        let st = unsafe { vokra_stream_push_pcm(stream, pcm[offset..].as_ptr(), chunk) };
        assert_eq!(
            st,
            VOKRA_OK,
            "vokra_stream_push_pcm failed: {}",
            status_name(st)
        );
        offset += chunk;

        let mut batch = [0.0f32; 64];
        let mut count = 0usize;
        // SAFETY: `batch` is valid for 64 writes; `count` is a writable slot.
        let st = unsafe { vokra_stream_poll(stream, batch.as_mut_ptr(), batch.len(), &mut count) };
        assert_eq!(
            st,
            VOKRA_OK,
            "vokra_stream_poll failed: {}",
            status_name(st)
        );
        assert!(count <= batch.len(), "poll overran the caller buffer");
        probs.extend_from_slice(&batch[..count]);
    }

    // SAFETY: freshly opened handle, destroyed exactly once.
    unsafe { vokra_stream_destroy(stream) };

    assert!(
        !probs.is_empty(),
        "the VAD fixture produced no probabilities — the oracle would be vacuous"
    );
    probs
}

/// Bit-exact comparison via raw IEEE-754 bits: identical weights on the
/// identical backend must reproduce identical floats, not merely close ones.
fn assert_bit_identical(left: &[f32], right: &[f32], what: &str) {
    assert_eq!(
        left.len(),
        right.len(),
        "{what}: produced {} probabilities vs {} from the legacy constructor",
        left.len(),
        right.len()
    );
    for (i, (a, b)) in left.iter().zip(right.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{what}: probability #{i} differs ({a} vs {b}) — the options path is \
             not the same session"
        );
    }
}

/// The probabilities produced by the legacy `vokra_session_create_from_file`,
/// used as the reference for T2 / T4.
fn legacy_file_probs() -> Vec<f32> {
    let path = silero_path_cstring();
    let mut session: *mut c_void = ptr::null_mut();
    // SAFETY: valid C string path and a writable out-slot.
    let st = unsafe { vokra_session_create_from_file(path.as_ptr(), &mut session) };
    assert_eq!(
        st,
        VOKRA_OK,
        "legacy vokra_session_create_from_file failed: {}",
        status_name(st)
    );
    let probs = vad_probs(session);
    // SAFETY: freshly created handle, destroyed exactly once.
    unsafe { vokra_session_destroy(session) };
    probs
}

// ---------------------------------------------------------------------------
// Header guard (design §3.1 / §3.2)
// ---------------------------------------------------------------------------

/// Collapses every run of whitespace to a single space so the checks below are
/// insensitive to cbindgen's line wrapping.
fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The generated header must actually declare the new enum, the opaque options
/// handle and all eight functions — and must **not** reserve delegate slots.
///
/// This also pins the mirrored `vokra_status_t` constants used throughout this
/// file, so the mirror cannot drift away from the real ABI unnoticed.
#[test]
fn header_declares_backend_enum_and_options_handle() {
    let header_path = repo_root().join("include/vokra.h");
    let header = normalize_ws(
        &std::fs::read_to_string(&header_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", header_path.display())),
    );

    // The status codes this file mirrors (guards the i32 mirror).
    for (name, value) in [
        ("VOKRA_OK", VOKRA_OK),
        ("VOKRA_ERROR_UNSUPPORTED_OP", VOKRA_ERROR_UNSUPPORTED_OP),
        (
            "VOKRA_ERROR_BACKEND_UNAVAILABLE",
            VOKRA_ERROR_BACKEND_UNAVAILABLE,
        ),
        ("VOKRA_ERROR_INVALID_ARGUMENT", VOKRA_ERROR_INVALID_ARGUMENT),
    ] {
        let needle = format!("{name} = {value}");
        assert!(
            header.contains(&needle),
            "include/vokra.h does not declare `{needle}` — the i32 status mirror \
             in this test has drifted from the ABI"
        );
    }

    // §3.1: the backend enum, with exactly the documented discriminants.
    for (name, value) in [
        ("VOKRA_BACKEND_CPU", VOKRA_BACKEND_CPU),
        ("VOKRA_BACKEND_METAL", VOKRA_BACKEND_METAL),
        ("VOKRA_BACKEND_CUDA", VOKRA_BACKEND_CUDA),
        ("VOKRA_BACKEND_VULKAN", VOKRA_BACKEND_VULKAN),
        ("VOKRA_BACKEND_WEBGPU", VOKRA_BACKEND_WEBGPU),
    ] {
        let needle = format!("{name} = {value}");
        assert!(
            header.contains(&needle),
            "include/vokra.h does not declare `{needle}` (design §3.1) — run \
             scripts/gen-c-abi.sh after adding vokra_backend_t"
        );
    }

    // D1: CoreML / QNN values are NOT reserved. Adding them here now would
    // freeze a delegate selector before the real-hardware NPU bakeoff
    // (docs/handoff/m4-12.md §(e)-3).
    for forbidden in [
        "VOKRA_BACKEND_COREML",
        "VOKRA_BACKEND_CORE_ML",
        "VOKRA_BACKEND_QNN",
    ] {
        assert!(
            !header.contains(forbidden),
            "include/vokra.h declares `{forbidden}` — D1 forbids reserving \
             delegate slots in the frozen enum"
        );
    }

    // §3.2: the options handle must stay opaque (a forward declaration only).
    // A `#[repr(C)]` struct would expose its layout to C# marshalling and pin
    // it into the frozen surface.
    assert!(
        header.contains("typedef struct vokra_session_options_t vokra_session_options_t;"),
        "include/vokra.h does not forward-declare the opaque \
         vokra_session_options_t (design §3.2)"
    );

    // §3.3: all eight new functions are exported.
    for func in [
        "vokra_session_options_create",
        "vokra_session_options_destroy",
        "vokra_session_options_set_backend",
        "vokra_session_create_from_file_with_options",
        "vokra_session_create_from_bytes_with_options",
        "vokra_backend_available",
        "vokra_speaker_embed",
        "vokra_speaker_verify",
    ] {
        assert!(
            header.contains(func),
            "include/vokra.h does not export `{func}` (design §3.3)"
        );
    }
}

// ---------------------------------------------------------------------------
// T1 — vokra_backend_available
// ---------------------------------------------------------------------------

/// T1: the CPU backend is always available, and the query never fails.
///
/// `vokra_backend_available` returns `bool` precisely because a query has no
/// failure mode (§3.4), so it must also leave `vokra_last_error()` alone rather
/// than reporting "unavailable" as an error.
#[test]
fn t1_backend_available_reports_cpu() {
    // SAFETY: a pure query over an integer discriminant; no pointers involved.
    let cpu = unsafe { vokra_backend_available(VOKRA_BACKEND_CPU) };
    assert!(
        cpu,
        "vokra_backend_available(VOKRA_BACKEND_CPU) must be true — the CPU \
         backend is the always-present baseline (FR-BE-01)"
    );

    // The query is total: every exposed discriminant answers without panicking.
    for backend in GPU_BACKENDS {
        // SAFETY: pure query.
        let available = unsafe { vokra_backend_available(backend) };
        println!(
            "vokra_backend_available({}) = {available}",
            backend_name(backend)
        );
    }
}

/// T1 (consistency): the answer must agree with what the loader actually does.
///
/// `available(CPU) == true` is only meaningful if a CPU session then builds.
#[test]
fn t1_cpu_availability_agrees_with_session_creation() {
    // SAFETY: pure query.
    assert!(unsafe { vokra_backend_available(VOKRA_BACKEND_CPU) });

    let path = silero_path_cstring();
    // SAFETY: allocates an options object; NULL on allocation failure.
    let opts = unsafe { vokra_session_options_create() };
    assert!(
        !opts.is_null(),
        "vokra_session_options_create returned NULL"
    );

    // SAFETY: `opts` is a live options handle.
    let st = unsafe { vokra_session_options_set_backend(opts, VOKRA_BACKEND_CPU) };
    assert_eq!(
        st,
        VOKRA_OK,
        "set_backend(CPU) failed: {} ({:?})",
        status_name(st),
        last_error()
    );

    let mut session: *mut c_void = ptr::null_mut();
    // SAFETY: valid path, live options handle, writable out-slot.
    let st =
        unsafe { vokra_session_create_from_file_with_options(path.as_ptr(), opts, &mut session) };
    assert_eq!(
        st,
        VOKRA_OK,
        "CPU is reported available but creating a CPU session failed: {} ({:?})",
        status_name(st),
        last_error()
    );
    assert!(!session.is_null());

    // SAFETY: each handle freshly created and destroyed exactly once.
    unsafe {
        vokra_session_destroy(session);
        vokra_session_options_destroy(opts);
    }
}

// ---------------------------------------------------------------------------
// T2 — options(CPU) reproduces the legacy constructors
// ---------------------------------------------------------------------------

/// T2: `..._from_file_with_options` with an explicit CPU backend produces the
/// same session as the legacy `vokra_session_create_from_file`.
#[test]
fn t2_options_cpu_matches_legacy_create_from_file() {
    let reference = legacy_file_probs();

    let path = silero_path_cstring();
    // SAFETY: allocates an options object.
    let opts = unsafe { vokra_session_options_create() };
    assert!(
        !opts.is_null(),
        "vokra_session_options_create returned NULL"
    );
    // SAFETY: live options handle.
    let st = unsafe { vokra_session_options_set_backend(opts, VOKRA_BACKEND_CPU) };
    assert_eq!(st, VOKRA_OK, "set_backend(CPU): {}", status_name(st));

    let mut session: *mut c_void = ptr::null_mut();
    // SAFETY: valid path, live options handle, writable out-slot.
    let st =
        unsafe { vokra_session_create_from_file_with_options(path.as_ptr(), opts, &mut session) };
    assert_eq!(
        st,
        VOKRA_OK,
        "create_from_file_with_options(CPU) failed: {} ({:?})",
        status_name(st),
        last_error()
    );
    assert!(!session.is_null());

    let observed = vad_probs(session);
    // SAFETY: handles freshly created, destroyed exactly once.
    unsafe {
        vokra_session_destroy(session);
        vokra_session_options_destroy(opts);
    }

    assert_bit_identical(&observed, &reference, "from_file_with_options(CPU)");
}

/// T2 (bytes twin): `..._from_bytes_with_options` with CPU matches the legacy
/// `vokra_session_create_from_bytes` — the Unity WebGL path (ADR M4-02 §3)
/// must gain backend selection without changing its output.
#[test]
fn t2_options_cpu_matches_legacy_create_from_bytes() {
    let model = std::fs::read(silero_fixture()).expect("read Silero fixture");

    // Reference: the existing bytes constructor.
    let mut legacy: *mut c_void = ptr::null_mut();
    // SAFETY: `model` is a live buffer of `model.len()` bytes; writable out-slot.
    let st = unsafe { vokra_session_create_from_bytes(model.as_ptr(), model.len(), &mut legacy) };
    assert_eq!(
        st,
        VOKRA_OK,
        "legacy create_from_bytes failed: {}",
        status_name(st)
    );
    let reference = vad_probs(legacy);
    // SAFETY: freshly created handle, destroyed exactly once.
    unsafe { vokra_session_destroy(legacy) };

    // SAFETY: allocates an options object.
    let opts = unsafe { vokra_session_options_create() };
    assert!(!opts.is_null());
    // SAFETY: live options handle.
    let st = unsafe { vokra_session_options_set_backend(opts, VOKRA_BACKEND_CPU) };
    assert_eq!(st, VOKRA_OK, "set_backend(CPU): {}", status_name(st));

    let mut session: *mut c_void = ptr::null_mut();
    // SAFETY: live buffer, live options handle, writable out-slot.
    let st = unsafe {
        vokra_session_create_from_bytes_with_options(
            model.as_ptr(),
            model.len(),
            opts,
            &mut session,
        )
    };
    assert_eq!(
        st,
        VOKRA_OK,
        "create_from_bytes_with_options(CPU) failed: {} ({:?})",
        status_name(st),
        last_error()
    );
    assert!(!session.is_null());

    let observed = vad_probs(session);
    // SAFETY: handles freshly created, destroyed exactly once.
    unsafe {
        vokra_session_destroy(session);
        vokra_session_options_destroy(opts);
    }

    assert_bit_identical(&observed, &reference, "from_bytes_with_options(CPU)");
}

// ---------------------------------------------------------------------------
// T3 — no silent CPU fallback (FR-EX-08)
// ---------------------------------------------------------------------------

/// T3: selecting a backend this build cannot provide must fail loudly.
///
/// The assertion is driven by `vokra_backend_available` rather than by
/// `cfg(feature = ...)` so it holds in every build configuration: whenever the
/// runtime says a backend is unavailable, the loader must refuse it with
/// `VOKRA_ERROR_BACKEND_UNAVAILABLE` and must never hand back a session that
/// quietly ran on the CPU (FR-EX-08, design §5).
///
/// The rejection is accepted at either gate — `set_backend` may validate
/// eagerly, or the loader may validate at creation — but it must happen at one
/// of them. In the default build (no GPU features on `vokra-capi`) all four GPU
/// backends take the unavailable branch.
///
/// A backend that *is* available takes the second branch: an arch whose engine
/// honors the selection must produce real output on it, and an arch whose
/// engine cannot run every hot op it needs must be refused with
/// `VOKRA_ERROR_UNSUPPORTED_OP`. Either way, a session that quietly evaluated
/// on the CPU is a failure.
#[test]
fn t3_unavailable_backend_never_falls_back_to_cpu() {
    let path = silero_path_cstring();
    let reference_len = legacy_file_probs().len();
    let mut checked_unavailable = 0usize;
    let mut checked_available = 0usize;

    for backend in GPU_BACKENDS {
        // SAFETY: pure query.
        let available = unsafe { vokra_backend_available(backend) };

        // SAFETY: allocates an options object.
        let opts = unsafe { vokra_session_options_create() };
        assert!(
            !opts.is_null(),
            "vokra_session_options_create returned NULL"
        );
        // SAFETY: live options handle; `backend` is a documented discriminant.
        let set_status = unsafe { vokra_session_options_set_backend(opts, backend) };

        if !available {
            checked_unavailable += 1;

            if set_status != VOKRA_OK {
                // Eager validation: it must name the right failure.
                assert_eq!(
                    set_status,
                    VOKRA_ERROR_BACKEND_UNAVAILABLE,
                    "set_backend({}) rejected an unavailable backend with {} — \
                     an unavailable backend is BACKEND_UNAVAILABLE (design §5)",
                    backend_name(backend),
                    status_name(set_status)
                );
                assert!(
                    last_error().is_some(),
                    "set_backend({}) failed without recording a message for \
                     vokra_last_error()",
                    backend_name(backend)
                );
            } else {
                // Deferred validation: the loader must refuse.
                let mut session: *mut c_void = ptr::null_mut();
                // SAFETY: valid path, live options handle, writable out-slot.
                let st = unsafe {
                    vokra_session_create_from_file_with_options(path.as_ptr(), opts, &mut session)
                };
                assert_ne!(
                    st,
                    VOKRA_OK,
                    "create_from_file_with_options({}) returned VOKRA_OK for an \
                     unavailable backend — this is the silent CPU fallback \
                     FR-EX-08 forbids",
                    backend_name(backend)
                );
                assert_eq!(
                    st,
                    VOKRA_ERROR_BACKEND_UNAVAILABLE,
                    "create_from_file_with_options({}) rejected an unavailable \
                     backend with {} instead of BACKEND_UNAVAILABLE (design §5)",
                    backend_name(backend),
                    status_name(st)
                );
                assert!(
                    session.is_null(),
                    "out_session was written on the reject path for {}",
                    backend_name(backend)
                );
                assert!(
                    last_error().is_some(),
                    "rejecting {} recorded no message for vokra_last_error()",
                    backend_name(backend)
                );
            }
        } else {
            // The backend exists on this machine. Which outcome is correct
            // depends on whether *this arch's* engine is backend-parameterised:
            //
            //   - Silero VAD became backend-honoring in this branch
            //     (`SileroVadV5::with_backend`, `Compute::for_backend` over
            //     `SILERO_HOT_OPS`), so a backend that covers `Conv1d` and
            //     `Gemv` really does run the forward — VOKRA_OK is correct and
            //     the session must actually produce probabilities.
            //   - A backend that does not cover every hot op must be refused
            //     with UNSUPPORTED_OP, never handed back as a CPU session.
            //
            // The 2026-08-14 review is why the VOKRA_OK arm still drives the
            // session instead of accepting the status alone: a branch that
            // only looked at the status code stayed green when the CPU-only
            // guard was deleted. Engines that are *not* backend-parameterised
            // (Mimi, NanoCodec) keep that guard, and
            // `session::tests::reject_cpu_only_backend_passes_cpu_and_refuses_gpus`
            // covers it directly.
            checked_available += 1;
            let mut session: *mut c_void = ptr::null_mut();
            // SAFETY: valid path, live options handle, writable out-slot.
            let st = unsafe {
                vokra_session_create_from_file_with_options(path.as_ptr(), opts, &mut session)
            };
            if st == VOKRA_OK {
                assert!(
                    !session.is_null(),
                    "create_from_file_with_options({}) returned VOKRA_OK without \
                     writing a session handle",
                    backend_name(backend)
                );
                let probs = vad_probs(session);
                // SAFETY: handle freshly created, destroyed exactly once.
                unsafe { vokra_session_destroy(session) };
                assert_eq!(
                    probs.len(),
                    reference_len,
                    "{} produced {} probabilities for the shared fixture; the CPU \
                     reference produced {}",
                    backend_name(backend),
                    probs.len(),
                    reference_len
                );
                assert!(
                    probs
                        .iter()
                        .all(|p| p.is_finite() && (0.0..=1.0).contains(p)),
                    "{} produced a non-finite or out-of-range probability: {:?}",
                    backend_name(backend),
                    probs
                );
            } else {
                assert_eq!(
                    st,
                    VOKRA_ERROR_UNSUPPORTED_OP,
                    "create_from_file_with_options({}) returned {} for an available \
                     backend — a backend that cannot run every hot op this arch \
                     needs is refused with UNSUPPORTED_OP (design §5), never with \
                     a session that quietly ran on the CPU (FR-EX-08)",
                    backend_name(backend),
                    status_name(st)
                );
                assert!(
                    session.is_null(),
                    "out_session was written on the reject path for {}",
                    backend_name(backend)
                );
                assert!(
                    last_error().is_some(),
                    "rejecting {} recorded no message for vokra_last_error()",
                    backend_name(backend)
                );
            }
        }

        // SAFETY: freshly created options handle, destroyed exactly once.
        unsafe { vokra_session_options_destroy(opts) };
    }

    // Every backend must have gone down exactly one of the two branches. A
    // silent zero on both sides would mean the loop asserted nothing at all,
    // which a printed count alone does not catch.
    assert_eq!(
        checked_unavailable + checked_available,
        GPU_BACKENDS.len(),
        "t3 skipped a backend: {checked_unavailable} unavailable + \
         {checked_available} available != {} declared",
        GPU_BACKENDS.len()
    );
    println!(
        "t3: {checked_unavailable}/{} GPU backends were unavailable in this build \
         and took the no-fallback branch; {checked_available} were available and \
         took the CPU-only-arch refusal branch",
        GPU_BACKENDS.len()
    );
}

// ---------------------------------------------------------------------------
// T4 — opts = NULL means "default CPU"
// ---------------------------------------------------------------------------

/// T4: passing `NULL` options selects the default (CPU) and matches the legacy
/// constructor exactly, so a caller with no preference need not allocate an
/// options object (design §3.4).
#[test]
fn t4_null_options_defaults_to_cpu_and_matches_legacy() {
    let reference = legacy_file_probs();
    let path = silero_path_cstring();

    let mut session: *mut c_void = ptr::null_mut();
    // SAFETY: valid path; NULL options is the documented "defaults" input;
    // writable out-slot.
    let st = unsafe {
        vokra_session_create_from_file_with_options(path.as_ptr(), ptr::null(), &mut session)
    };
    assert_eq!(
        st,
        VOKRA_OK,
        "NULL options must mean default CPU, got {} ({:?})",
        status_name(st),
        last_error()
    );
    assert!(!session.is_null());

    let observed = vad_probs(session);
    // SAFETY: freshly created handle, destroyed exactly once.
    unsafe { vokra_session_destroy(session) };

    assert_bit_identical(&observed, &reference, "from_file_with_options(NULL opts)");
}

/// T4 (bytes twin): NULL options on the bytes constructor also defaults to CPU.
#[test]
fn t4_null_options_defaults_to_cpu_on_bytes_path() {
    let model = std::fs::read(silero_fixture()).expect("read Silero fixture");

    let mut legacy: *mut c_void = ptr::null_mut();
    // SAFETY: live buffer and writable out-slot.
    let st = unsafe { vokra_session_create_from_bytes(model.as_ptr(), model.len(), &mut legacy) };
    assert_eq!(
        st,
        VOKRA_OK,
        "legacy create_from_bytes: {}",
        status_name(st)
    );
    let reference = vad_probs(legacy);
    // SAFETY: freshly created handle, destroyed exactly once.
    unsafe { vokra_session_destroy(legacy) };

    let mut session: *mut c_void = ptr::null_mut();
    // SAFETY: live buffer; NULL options is the documented "defaults" input.
    let st = unsafe {
        vokra_session_create_from_bytes_with_options(
            model.as_ptr(),
            model.len(),
            ptr::null(),
            &mut session,
        )
    };
    assert_eq!(
        st,
        VOKRA_OK,
        "NULL options on the bytes path must mean default CPU, got {}",
        status_name(st)
    );
    assert!(!session.is_null());

    let observed = vad_probs(session);
    // SAFETY: freshly created handle, destroyed exactly once.
    unsafe { vokra_session_destroy(session) };

    assert_bit_identical(&observed, &reference, "from_bytes_with_options(NULL opts)");
}

// ---------------------------------------------------------------------------
// T5 — NULL arguments and options lifecycle
// ---------------------------------------------------------------------------

/// T5: every NULL-pointer argument is rejected with `INVALID_ARGUMENT`, no
/// panic crosses the boundary, and `out_session` is never written on a reject
/// path (the house rule pinned by `session.rs`'s existing tests).
#[test]
fn t5_null_arguments_are_rejected_without_panic() {
    let path = silero_path_cstring();
    let model = [0u8; 8];

    // NULL options handle into the setter.
    // SAFETY: NULL is the rejected branch.
    let st = unsafe { vokra_session_options_set_backend(ptr::null_mut(), VOKRA_BACKEND_CPU) };
    assert_eq!(
        st,
        VOKRA_ERROR_INVALID_ARGUMENT,
        "set_backend(NULL) must be INVALID_ARGUMENT, got {}",
        status_name(st)
    );
    assert!(
        last_error().is_some(),
        "set_backend(NULL) recorded no error message"
    );

    // SAFETY: allocates an options object used by the cases below.
    let opts = unsafe { vokra_session_options_create() };
    assert!(
        !opts.is_null(),
        "vokra_session_options_create returned NULL"
    );

    // NULL path.
    let mut session: *mut c_void = ptr::null_mut();
    // SAFETY: NULL path is the rejected branch; out-slot is writable.
    let st =
        unsafe { vokra_session_create_from_file_with_options(ptr::null(), opts, &mut session) };
    assert_eq!(
        st,
        VOKRA_ERROR_INVALID_ARGUMENT,
        "NULL path must be INVALID_ARGUMENT, got {}",
        status_name(st)
    );
    assert!(session.is_null(), "out_session written on the reject path");

    // NULL out_session.
    // SAFETY: NULL out_session is the rejected branch.
    let st = unsafe {
        vokra_session_create_from_file_with_options(path.as_ptr(), opts, ptr::null_mut())
    };
    assert_eq!(
        st,
        VOKRA_ERROR_INVALID_ARGUMENT,
        "NULL out_session must be INVALID_ARGUMENT, got {}",
        status_name(st)
    );

    // NULL data on the bytes path.
    let mut session: *mut c_void = ptr::null_mut();
    // SAFETY: NULL data is the rejected branch.
    let st =
        unsafe { vokra_session_create_from_bytes_with_options(ptr::null(), 4, opts, &mut session) };
    assert_eq!(
        st,
        VOKRA_ERROR_INVALID_ARGUMENT,
        "NULL bytes must be INVALID_ARGUMENT, got {}",
        status_name(st)
    );
    assert!(session.is_null(), "out_session written on the reject path");

    // Zero-length buffer: a 0-byte model is never a valid GGUF. Mirrors the
    // existing `vokra_session_create_from_bytes` behaviour.
    // SAFETY: valid pointer with len 0 is the rejected branch.
    let st = unsafe {
        vokra_session_create_from_bytes_with_options(model.as_ptr(), 0, opts, &mut session)
    };
    assert_eq!(
        st,
        VOKRA_ERROR_INVALID_ARGUMENT,
        "zero-length model must be INVALID_ARGUMENT, got {}",
        status_name(st)
    );
    assert!(session.is_null(), "out_session written on the reject path");

    // NULL out_session on the bytes path.
    // SAFETY: NULL out_session is the rejected branch.
    let st = unsafe {
        vokra_session_create_from_bytes_with_options(
            model.as_ptr(),
            model.len(),
            opts,
            ptr::null_mut(),
        )
    };
    assert_eq!(
        st,
        VOKRA_ERROR_INVALID_ARGUMENT,
        "NULL out_session on the bytes path must be INVALID_ARGUMENT, got {}",
        status_name(st)
    );

    // SAFETY: freshly created options handle, destroyed exactly once.
    unsafe { vokra_session_options_destroy(opts) };
}

/// T5: a non-UTF-8 path is rejected the way the legacy constructor rejects it,
/// never decoded with a locale-dependent fallback (NFR-RL-01).
#[test]
fn t5_non_utf8_path_is_rejected() {
    // 0xFF is not a valid UTF-8 lead byte.
    let bad = [0xFFu8, 0xFE, 0x00];
    let mut session: *mut c_void = ptr::null_mut();
    // SAFETY: NUL-terminated byte string that is not valid UTF-8 — the
    // rejected branch; out-slot is writable.
    let st = unsafe {
        vokra_session_create_from_file_with_options(
            bad.as_ptr().cast::<c_char>(),
            ptr::null(),
            &mut session,
        )
    };
    assert_eq!(
        st,
        VOKRA_ERROR_INVALID_ARGUMENT,
        "non-UTF-8 path must be INVALID_ARGUMENT, got {}",
        status_name(st)
    );
    assert!(session.is_null(), "out_session written on the reject path");
}

/// T5 (lifecycle): options objects are independent allocations and
/// `destroy(NULL)` is a no-op (ADR-0003 §3-a).
///
/// A true use-after-free check is not written here: reading a destroyed handle
/// is undefined behaviour, so a test that "expects INVALID_ARGUMENT" from a
/// dangling pointer would be asserting on UB rather than on the contract. What
/// *is* checkable — and what protects callers in practice — is that destroying
/// one options object never disturbs another, and that the documented NULL
/// no-op holds.
#[test]
fn t5_options_lifecycle_is_independent_and_destroy_null_is_noop() {
    // Destroying NULL is a documented no-op.
    // SAFETY: NULL is the documented no-op input for a destroy function.
    unsafe { vokra_session_options_destroy(ptr::null_mut()) };

    // SAFETY: allocates an options object.
    let a = unsafe { vokra_session_options_create() };
    // SAFETY: allocates a second, independent options object.
    let b = unsafe { vokra_session_options_create() };
    assert!(!a.is_null() && !b.is_null(), "options create returned NULL");
    assert_ne!(
        a, b,
        "two vokra_session_options_create calls returned the same pointer — \
         options must be independent objects, not a shared singleton"
    );

    // Configure `b`, then destroy `a`: `b` must be unaffected.
    // SAFETY: `b` is live.
    let st = unsafe { vokra_session_options_set_backend(b, VOKRA_BACKEND_CPU) };
    assert_eq!(st, VOKRA_OK, "set_backend(CPU) on b: {}", status_name(st));
    // SAFETY: `a` is live and destroyed exactly once.
    unsafe { vokra_session_options_destroy(a) };

    let path = silero_path_cstring();
    let mut session: *mut c_void = ptr::null_mut();
    // SAFETY: valid path, live options handle `b`, writable out-slot.
    let st = unsafe { vokra_session_create_from_file_with_options(path.as_ptr(), b, &mut session) };
    assert_eq!(
        st,
        VOKRA_OK,
        "destroying an unrelated options object invalidated this one: {}",
        status_name(st)
    );
    assert!(!session.is_null());

    // SAFETY: each handle destroyed exactly once.
    unsafe {
        vokra_session_destroy(session);
        vokra_session_options_destroy(b);
    }
}

// ---------------------------------------------------------------------------
// T6 — unknown enum values
// ---------------------------------------------------------------------------

/// T6: a discriminant outside the documented enum is rejected with
/// `INVALID_ARGUMENT` (design §5) — including the `5` / `6` slots a caller might
/// guess for CoreML / QNN, which D1 deliberately leaves unassigned.
#[test]
fn t6_unknown_backend_values_are_rejected() {
    // SAFETY: allocates an options object.
    let opts = unsafe { vokra_session_options_create() };
    assert!(
        !opts.is_null(),
        "vokra_session_options_create returned NULL"
    );

    // 5 and 6 are the slots CoreML / QNN would take: they must NOT work yet.
    for value in [5i32, 6, 7, 99, -1, i32::MAX, i32::MIN] {
        // SAFETY: live options handle; an out-of-range integer is exactly what a
        // C caller can pass through an `enum` parameter.
        let st = unsafe { vokra_session_options_set_backend(opts, value) };
        assert_eq!(
            st,
            VOKRA_ERROR_INVALID_ARGUMENT,
            "set_backend({value}) must be INVALID_ARGUMENT (design §5), got {}",
            status_name(st)
        );
        assert!(
            last_error().is_some(),
            "set_backend({value}) recorded no error message"
        );
    }

    // A rejected setter must not corrupt the object: it still builds a session
    // on the default backend.
    let path = silero_path_cstring();
    let mut session: *mut c_void = ptr::null_mut();
    // SAFETY: valid path, live options handle, writable out-slot.
    let st =
        unsafe { vokra_session_create_from_file_with_options(path.as_ptr(), opts, &mut session) };
    assert_eq!(
        st,
        VOKRA_OK,
        "a rejected set_backend left the options object unusable: {}",
        status_name(st)
    );
    assert!(!session.is_null());

    // SAFETY: each handle destroyed exactly once.
    unsafe {
        vokra_session_destroy(session);
        vokra_session_options_destroy(opts);
    }
}

/// T6 (query side): `vokra_backend_available` has no error channel (§3.4), so an
/// unknown discriminant is simply "not available" — and must not panic.
#[test]
fn t6_unknown_backend_value_is_not_available() {
    for value in [5i32, 6, 7, 99, -1, i32::MAX, i32::MIN] {
        // SAFETY: pure query over an integer.
        let available = unsafe { vokra_backend_available(value) };
        assert!(
            !available,
            "vokra_backend_available({value}) returned true for a discriminant \
             that is not in vokra_backend_t"
        );
    }
}
