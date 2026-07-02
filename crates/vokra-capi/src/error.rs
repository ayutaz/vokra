//! Status codes and the thread-local error channel (M0-09-T03).
//!
//! Every fallible C function returns a `vokra_status_t` (`VOKRA_OK` = 0 on
//! success, a non-zero code otherwise). The human-readable detail for the last
//! failure is kept in a **thread-local** slot and retrieved with
//! `vokra_last_error` (FR-API-01). Keeping it thread-local means concurrent
//! callers never clobber each other's message (ADR-0003 §3).

use std::cell::RefCell;
use std::ffi::{CString, c_char};

use vokra_core::VokraError;

/// Status code returned by the fallible Vokra C functions (ADR-0003 §3-d).
///
/// `VOKRA_OK` is 0; every error is a distinct non-zero code mirroring a
/// `VokraError` variant, plus `VOKRA_ERROR_PANIC` for a Rust panic caught at
/// the FFI boundary and `VOKRA_ERROR_OTHER` for any future `#[non_exhaustive]`
/// variant. The numeric values are part of the (M0-unstable) ABI.
//
// Named in C style (`vokra_status_t`, `VOKRA_*`) so cbindgen emits the enum
// verbatim without rename rules; this trips the Rust casing lints, hence the
// crate-local allow.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum vokra_status_t {
    /// Success.
    VOKRA_OK = 0,
    /// I/O failure (file open / read / metadata). Maps `VokraError::Io`.
    VOKRA_ERROR_IO = 1,
    /// A model file could not be loaded or parsed. Maps `VokraError::ModelLoad`.
    VOKRA_ERROR_MODEL_LOAD = 2,
    /// The selected backend does not support an op. Maps `VokraError::UnsupportedOp`.
    VOKRA_ERROR_UNSUPPORTED_OP = 3,
    /// The requested backend is unavailable. Maps `VokraError::BackendUnavailable`.
    VOKRA_ERROR_BACKEND_UNAVAILABLE = 4,
    /// A caller-supplied argument is invalid (NULL, non-UTF-8, wrong rate, ...).
    /// Maps `VokraError::InvalidArgument`.
    VOKRA_ERROR_INVALID_ARGUMENT = 5,
    /// An audio graph failed validation. Maps `VokraError::GraphValidation`.
    VOKRA_ERROR_GRAPH_VALIDATION = 6,
    /// The requested capability is not wired for this model/task (e.g. calling
    /// TTS on an ASR model). Maps `VokraError::NotImplemented`.
    VOKRA_ERROR_NOT_IMPLEMENTED = 7,
    /// A Rust panic was caught at the FFI boundary (never propagated across it).
    VOKRA_ERROR_PANIC = 8,
    /// Any other / future error variant.
    VOKRA_ERROR_OTHER = 9,
}

thread_local! {
    /// Last error message for the calling thread (`None` until one is set).
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Maps a `VokraError` to its status code (ADR-0003 §3-d).
///
/// `VokraError` is `#[non_exhaustive]`; unmapped future variants fall through
/// to `vokra_status_t::VOKRA_ERROR_OTHER`.
fn status_of(err: &VokraError) -> vokra_status_t {
    match err {
        VokraError::Io(_) => vokra_status_t::VOKRA_ERROR_IO,
        VokraError::ModelLoad(_) => vokra_status_t::VOKRA_ERROR_MODEL_LOAD,
        VokraError::UnsupportedOp(_) => vokra_status_t::VOKRA_ERROR_UNSUPPORTED_OP,
        VokraError::BackendUnavailable(_) => vokra_status_t::VOKRA_ERROR_BACKEND_UNAVAILABLE,
        VokraError::InvalidArgument(_) => vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT,
        VokraError::GraphValidation(_) => vokra_status_t::VOKRA_ERROR_GRAPH_VALIDATION,
        VokraError::NotImplemented(_) => vokra_status_t::VOKRA_ERROR_NOT_IMPLEMENTED,
        _ => vokra_status_t::VOKRA_ERROR_OTHER,
    }
}

/// Builds a NUL-terminated message, neutralising any interior NUL bytes (a
/// message must survive the trip through C as a single C string).
fn to_cstring(msg: &str) -> CString {
    match CString::new(msg) {
        Ok(c) => c,
        Err(_) => {
            let sanitized: String = msg.replace('\0', " ");
            CString::new(sanitized).unwrap_or_default()
        }
    }
}

/// Stores `msg` as the calling thread's last error message.
pub(crate) fn set_last_error(msg: &str) {
    let cstring = to_cstring(msg);
    LAST_ERROR.with(|cell| *cell.borrow_mut() = Some(cstring));
}

/// Records `err` as the thread's last error and returns its status code — the
/// single failure exit used across the C ABI.
pub(crate) fn fail(err: &VokraError) -> vokra_status_t {
    set_last_error(&err.to_string());
    status_of(err)
}

/// Records an ad-hoc `InvalidArgument`-class message (NULL/UTF-8 guards) and
/// returns `vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT`.
pub(crate) fn fail_invalid(msg: &str) -> vokra_status_t {
    set_last_error(msg);
    vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT
}

/// Returns the calling thread's last error message as a NUL-terminated UTF-8
/// C string, or `NULL` if no error has been recorded on this thread.
///
/// The returned pointer is owned by Vokra and stays valid until the next error
/// is recorded **on the same thread**; do not free it and do not use it across
/// threads. `vokra_last_error` itself never fails and never allocates
/// (ADR-0003 §3-b).
#[unsafe(no_mangle)]
pub extern "C" fn vokra_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| {
        cell.borrow()
            .as_ref()
            .map_or(std::ptr::null(), |cstr| cstr.as_ptr())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    /// Reads the current thread's last-error message as a Rust string.
    fn last_error_string() -> Option<String> {
        let ptr = vokra_last_error();
        if ptr.is_null() {
            return None;
        }
        // SAFETY: `ptr` is non-null and points at the thread-local CString set
        // by `set_last_error`, which is NUL-terminated and lives until the next
        // set on this thread (we do not set in between).
        let s = unsafe { CStr::from_ptr(ptr) };
        Some(s.to_str().expect("last error is UTF-8").to_owned())
    }

    #[test]
    fn no_error_returns_null() {
        // A pristine thread has no message.
        assert!(vokra_last_error().is_null());
        assert_eq!(last_error_string(), None);
    }

    #[test]
    fn set_and_get_roundtrips_utf8() {
        set_last_error("bad argument: café");
        assert_eq!(last_error_string().as_deref(), Some("bad argument: café"));
    }

    #[test]
    fn fail_maps_variant_and_records_message() {
        let status = fail(&VokraError::ModelLoad("bad magic".to_owned()));
        assert_eq!(status, vokra_status_t::VOKRA_ERROR_MODEL_LOAD);
        assert_eq!(
            last_error_string().as_deref(),
            Some("model load error: bad magic")
        );
    }

    #[test]
    fn interior_nul_is_neutralised() {
        set_last_error("before\0after");
        assert_eq!(last_error_string().as_deref(), Some("before after"));
    }

    #[test]
    fn errno_is_thread_local() {
        set_last_error("main-thread message");
        let other = std::thread::spawn(|| {
            // The spawned thread starts clean and sets its own message.
            let before = vokra_last_error().is_null();
            set_last_error("worker-thread message");
            (before, last_error_string())
        })
        .join()
        .unwrap();
        assert!(other.0, "worker thread must start with no error");
        assert_eq!(other.1.as_deref(), Some("worker-thread message"));
        // The main thread's message is untouched by the worker.
        assert_eq!(last_error_string().as_deref(), Some("main-thread message"));
    }

    #[test]
    fn ok_is_zero() {
        assert_eq!(vokra_status_t::VOKRA_OK as i32, 0);
        assert_ne!(vokra_status_t::VOKRA_ERROR_PANIC as i32, 0);
    }

    #[test]
    fn fail_maps_every_variant_to_documented_status() {
        // Each mapped `VokraError` variant must land on its distinct C status
        // AND on the documented numeric value (the M0 ABI other hosts depend
        // on). Oracle is the ADR-0003 §3-d spec pinned in this module's rustdoc;
        // no reference implementation. `_ => VOKRA_ERROR_OTHER` is unreachable
        // without an unmapped (future) variant, so it is left noted only.
        let cases: Vec<(VokraError, vokra_status_t, i32)> = vec![
            (
                VokraError::Io(std::io::Error::other("disk gone")),
                vokra_status_t::VOKRA_ERROR_IO,
                1,
            ),
            (
                VokraError::ModelLoad("bad magic".to_owned()),
                vokra_status_t::VOKRA_ERROR_MODEL_LOAD,
                2,
            ),
            (
                VokraError::UnsupportedOp("Stft on cpu".to_owned()),
                vokra_status_t::VOKRA_ERROR_UNSUPPORTED_OP,
                3,
            ),
            (
                VokraError::BackendUnavailable("cuda".to_owned()),
                vokra_status_t::VOKRA_ERROR_BACKEND_UNAVAILABLE,
                4,
            ),
            (
                VokraError::InvalidArgument("bad rate".to_owned()),
                vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT,
                5,
            ),
            (
                VokraError::GraphValidation("dangling output".to_owned()),
                vokra_status_t::VOKRA_ERROR_GRAPH_VALIDATION,
                6,
            ),
            (
                VokraError::NotImplemented("speech-to-speech"),
                vokra_status_t::VOKRA_ERROR_NOT_IMPLEMENTED,
                7,
            ),
        ];
        for (err, expected_status, expected_code) in cases {
            let status = fail(&err);
            assert_eq!(status, expected_status, "wrong status for {err:?}");
            assert_eq!(status as i32, expected_code, "wrong ABI number for {err:?}");
        }
    }

    #[test]
    fn status_codes_pin_numeric_abi() {
        // The full status enum's numeric layout is part of the M0 ABI; freeze
        // every discriminant so a reorder / inserted variant fails loudly.
        assert_eq!(vokra_status_t::VOKRA_OK as i32, 0);
        assert_eq!(vokra_status_t::VOKRA_ERROR_IO as i32, 1);
        assert_eq!(vokra_status_t::VOKRA_ERROR_MODEL_LOAD as i32, 2);
        assert_eq!(vokra_status_t::VOKRA_ERROR_UNSUPPORTED_OP as i32, 3);
        assert_eq!(vokra_status_t::VOKRA_ERROR_BACKEND_UNAVAILABLE as i32, 4);
        assert_eq!(vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT as i32, 5);
        assert_eq!(vokra_status_t::VOKRA_ERROR_GRAPH_VALIDATION as i32, 6);
        assert_eq!(vokra_status_t::VOKRA_ERROR_NOT_IMPLEMENTED as i32, 7);
        assert_eq!(vokra_status_t::VOKRA_ERROR_PANIC as i32, 8);
        assert_eq!(vokra_status_t::VOKRA_ERROR_OTHER as i32, 9);
    }
}
