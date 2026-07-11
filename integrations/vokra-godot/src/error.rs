//! Panic firewall + Vokra-status → Godot Error conversion (T10, ADR-0011 §D6).
//!
//! # Panic firewall (NFR-RL-07)
//!
//! Every function that Godot's C runtime can call — the exported entry point
//! (`vokra_gdextension_init`), the init/deinit callbacks, and any future
//! method-binding trampoline — MUST wrap its Rust body in [`catch_panic`].
//! A panic that crosses into Godot's C stack unwinds through code compiled
//! **without** unwind tables and is Undefined Behavior; the workspace
//! `panic = "unwind"` policy (root `Cargo.toml`) makes `catch_unwind`
//! functional, but only when we actually call it at every C boundary.
//!
//! # Silent-fallback discipline (FR-EX-08 / NFR-RL-06)
//!
//! The C ABI never falls back silently: `VOKRA_ERROR_BACKEND_UNAVAILABLE`
//! propagates unchanged. This module ONLY translates codes; it never retries
//! on a different backend, and it never converts a non-OK code into an OK
//! return. That is enforced by the exhaustive match in [`VokraError::from_raw`].

use core::ffi::{CStr, c_char};

use crate::ffi::capi::{VokraStatus, vokra_last_error};

/// A Vokra error surfaced across the Godot binding. Wraps the underlying
/// C-ABI status plus the thread-local `vokra_last_error()` string captured at
/// the call site (ADR-00xx §2, same-thread discipline).
#[derive(Debug, Clone)]
pub struct VokraError {
    /// Original C ABI status.
    pub status: VokraStatus,
    /// Detail from `vokra_last_error()` captured on the same thread as the
    /// failing call. Empty string if none was recorded.
    pub message: String,
}

impl VokraError {
    /// Read the thread-local `vokra_last_error()` right now and pair it with
    /// `status`. MUST be called on the same OS thread as the failing C ABI
    /// call (ADR-00xx §3 thread contract).
    pub fn from_status_and_last(status: VokraStatus) -> Self {
        // `vokra_last_error` is declared `safe fn` in the extern block (see
        // `ffi::capi`) because Vokra guarantees the returned pointer is
        // either NULL or a valid NUL-terminated UTF-8 string owned by the
        // thread-local errno slot — no caller preconditions apply. The
        // pointer's lifetime is bounded by the next ABI call on this
        // thread (ADR-00xx §4 string ownership), so the copy below MUST
        // happen before any other Vokra call.
        let ptr: *const c_char = vokra_last_error();
        let message = if ptr.is_null() {
            String::new()
        } else {
            // SAFETY: `ptr` is non-null; Vokra guarantees NUL-terminated UTF-8.
            // Lossy conversion so pathological non-UTF-8 (never produced by
            // Vokra itself but robust against future callers) does not panic.
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        };
        Self { status, message }
    }
}

impl core::fmt::Display for VokraError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Two forms so callers can tell whether a `vokra_last_error()`
        // string was captured on the same thread:
        //   empty  : "Vokra error <Status>"
        //   detail : "Vokra error <Status>: <message>"
        if self.message.is_empty() {
            write!(f, "Vokra error {:?}", self.status)
        } else {
            write!(f, "Vokra error {:?}: {}", self.status, self.message)
        }
    }
}

/// Convert a raw i32 status returned by the C ABI into either `Ok(())` or
/// a `VokraError` carrying the thread-local last-error string.
///
/// Central hook for FR-EX-08: this is the ONLY place a non-OK status can
/// become an in-band value on the Godot side, and it never lets one become
/// success.
pub fn check(status: i32) -> Result<(), VokraError> {
    let s = VokraStatus::from_raw(status);
    if s == VokraStatus::Ok {
        Ok(())
    } else {
        Err(VokraError::from_status_and_last(s))
    }
}

/// Catch every panic that would otherwise unwind into Godot's C stack.
///
/// - Returns `Some(value)` when `f` completes normally.
/// - Returns `None` when `f` panicked; the panic is swallowed, and a synthetic
///   [`VokraError`] with status `Panic` can be constructed by the caller if
///   needed. We deliberately do NOT re-panic and do NOT let the panic escape.
///
/// This is the exact posture of `crates/vokra-capi/src/ffi_guard.rs` — the
/// M0-09 firewall — applied at the Godot boundary. The workspace
/// `panic = "unwind"` policy (root `Cargo.toml`) is what makes this call
/// functional; a `panic = "abort"` cdylib would UB.
pub fn catch_panic<F, R>(f: F) -> Option<R>
where
    F: FnOnce() -> R + core::panic::UnwindSafe,
{
    std::panic::catch_unwind(f).ok()
}

/// Panic-firewalled variant of [`catch_panic`] that reports back a
/// `Result<R, VokraError>`, mapping a caught panic to `VokraError` with
/// status `Panic`. Preferred at method boundaries where the caller wants a
/// uniform error surface.
pub fn catch_panic_as_err<F, R>(f: F) -> Result<R, VokraError>
where
    F: FnOnce() -> Result<R, VokraError> + core::panic::UnwindSafe,
{
    match std::panic::catch_unwind(f) {
        Ok(inner) => inner,
        Err(_) => Err(VokraError {
            status: VokraStatus::Panic,
            message: String::from("Rust panic caught at the Godot boundary"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_zero_is_ok() {
        assert!(check(0).is_ok());
    }

    #[test]
    fn check_nonzero_is_err_with_status() {
        // Codes 1..=9 are the canonical Vokra errors.
        for code in 1..=9 {
            let err = check(code).expect_err("nonzero must be Err");
            assert_eq!(err.status as i32, code);
        }
    }

    #[test]
    fn check_unknown_maps_to_other() {
        // Forward-compat: any unknown positive/negative code becomes `Other`.
        let err = check(42).expect_err("unknown must be Err");
        assert_eq!(err.status, VokraStatus::Other);
    }

    #[test]
    fn catch_panic_swallows_panic_and_returns_none() {
        let out = catch_panic(|| -> i32 { panic!("boom") });
        assert!(out.is_none());
    }

    #[test]
    fn catch_panic_passes_value_through_on_no_panic() {
        let out = catch_panic(|| 42_i32);
        assert_eq!(out, Some(42));
    }

    #[test]
    fn catch_panic_as_err_maps_panic_to_panic_status() {
        let out: Result<(), VokraError> = catch_panic_as_err(|| -> Result<(), VokraError> {
            panic!("boom");
        });
        let err = out.expect_err("panic must produce Err");
        assert_eq!(err.status, VokraStatus::Panic);
        assert!(!err.message.is_empty());
    }

    #[test]
    fn catch_panic_as_err_forwards_normal_error() {
        // A non-panic Err from `f` must pass through unchanged (no
        // synthetic Panic remapping).
        let out: Result<(), VokraError> = catch_panic_as_err(|| -> Result<(), VokraError> {
            Err(VokraError {
                status: VokraStatus::InvalidArgument,
                message: String::from("nope"),
            })
        });
        let err = out.expect_err("Err must propagate");
        assert_eq!(err.status, VokraStatus::InvalidArgument);
        assert_eq!(err.message, "nope");
    }

    #[test]
    fn display_includes_status_and_message() {
        let err = VokraError {
            status: VokraStatus::Io,
            message: String::from("file not found"),
        };
        let s = format!("{err}");
        assert!(s.contains("Io"));
        assert!(s.contains("file not found"));
    }

    #[test]
    fn display_omits_empty_message() {
        // With an empty message we render "Vokra error: <Status>" (single
        // colon after "error"). With a non-empty message we add a second
        // colon before the message (see `display_includes_status_and_message`).
        // Distinguish the two by counting colons.
        let empty = VokraError {
            status: VokraStatus::Panic,
            message: String::new(),
        };
        let with_msg = VokraError {
            status: VokraStatus::Panic,
            message: String::from("oops"),
        };
        let s_empty = format!("{empty}");
        let s_full = format!("{with_msg}");
        assert!(s_empty.contains("Panic"), "status must appear: {s_empty}");
        assert!(
            s_empty.matches(':').count() < s_full.matches(':').count(),
            "empty-message form must have fewer colons than the with-message form\n  empty: {s_empty}\n  full:  {s_full}",
        );
        // And the empty form must NOT trail on any user-supplied text.
        assert!(!s_empty.trim_end().ends_with(':'));
    }
}
