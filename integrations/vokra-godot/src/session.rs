//! Rust-side representation of a Vokra session for the Godot binding (T05).
//!
//! `VokraSession` is a thin RAII wrapper over the opaque `*mut VokraSession`
//! handle returned by `vokra_session_create_from_file`. Godot's Object
//! subclass (registered at T05 via `classdb_register_extension_class3`)
//! stores one of these behind an atomic pointer; the method-binding
//! trampolines (T06 ASR / T07 TTS / T08 VAD) borrow it while dispatching to
//! the C ABI.
//!
//! # Ownership (ADR-00xx §1)
//!
//! One [`VokraSession`] value == exactly one refcount on the underlying
//! `vokra_session_t`. `Drop` releases it via `vokra_session_destroy`. Cloning
//! is done through [`retain`] (atomic bump), NOT through Rust `Clone`. The
//! type is deliberately `!Clone` to make double-destroy impossible from safe
//! code.

use core::ffi::c_char;
use core::ptr;
use std::ffi::CString;

use crate::error::{VokraError, check};
use crate::ffi::capi::{
    VokraSession as CVokraSession, vokra_session_create_from_file, vokra_session_destroy,
    vokra_session_retain,
};

/// RAII wrapper. See module docs.
pub struct VokraSession {
    handle: *mut CVokraSession,
}

// `vokra_session_t` is `Send + Sync` (see `crates/vokra-capi/src/session.rs`
// crate doc); the `Session` inside is `Arc<...>`. Handles are safe to move
// across threads and safe to share, but the Godot binding pattern is to keep
// them single-thread-owned (ADR-00xx §3).
//
// SAFETY: The C ABI guarantees the wrapped `Session` is `Send + Sync`. See
// `crates/vokra-capi/src/session.rs` doc comment. Our `!Clone` policy means
// no aliasing of the raw pointer inside safe Rust.
unsafe impl Send for VokraSession {}
// SAFETY: Same rationale as `Send` above — the C ABI Session is Sync.
unsafe impl Sync for VokraSession {}

impl VokraSession {
    /// Load a GGUF from `path` and produce a new session. Errors surface as
    /// [`VokraError`] carrying `vokra_last_error()` context (ADR-00xx §2).
    ///
    /// # Locale (NFR-RL-01)
    ///
    /// `path` is treated as UTF-8 bytes end-to-end. No `strtod`/`printf`
    /// involvement, no LC_NUMERIC risk (the C ABI is numeric-typed only —
    /// ADR-00xx §5).
    pub fn from_file(path: &str) -> Result<Self, VokraError> {
        // CString rejects interior NULs — surface as InvalidArgument so
        // the Godot side gets a typed error instead of a panic.
        let c_path = CString::new(path).map_err(|_| VokraError {
            status: crate::ffi::capi::VokraStatus::InvalidArgument,
            message: format!("path contains NUL byte: {path:?}"),
        })?;

        let mut out: *mut CVokraSession = ptr::null_mut();
        // SAFETY: `c_path.as_ptr()` is a valid NUL-terminated UTF-8 pointer
        // for the lifetime of `c_path` (this call); `&mut out` is a valid
        // writable `*mut *mut CVokraSession` slot. The C ABI writes `out`
        // only on VOKRA_OK.
        let status =
            unsafe { vokra_session_create_from_file(c_path.as_ptr() as *const c_char, &mut out) };
        check(status)?;
        if out.is_null() {
            // Defensive: C ABI SHOULD NOT return VOKRA_OK with a NULL handle;
            // treat as ModelLoad to preserve invariants (double-free proof
            // by construction — `!Clone` + Drop only sees non-null handles).
            return Err(VokraError {
                status: crate::ffi::capi::VokraStatus::ModelLoad,
                message: String::from("Vokra returned VOKRA_OK with a NULL session handle"),
            });
        }
        Ok(Self { handle: out })
    }

    /// Atomically retain the underlying session, producing an independent
    /// handle. Both handles must be dropped for the model to be freed
    /// (FR-API-03).
    pub fn retain(&self) -> Result<Self, VokraError> {
        let mut out: *mut CVokraSession = ptr::null_mut();
        // SAFETY: `self.handle` is a live, non-null session handle (Drop is
        // the only path to NULL). `&mut out` is a valid writable slot.
        let status = unsafe { vokra_session_retain(self.handle, &mut out) };
        check(status)?;
        if out.is_null() {
            return Err(VokraError {
                status: crate::ffi::capi::VokraStatus::ModelLoad,
                message: String::from("Vokra returned VOKRA_OK with a NULL retain handle"),
            });
        }
        Ok(Self { handle: out })
    }

    /// Access the raw handle for dispatching to the C ABI. Callers MUST NOT
    /// call `vokra_session_destroy` on the returned pointer — that is `Drop`'s
    /// job. The lifetime is bounded to `&self`; a returned `*const` cannot
    /// outlive the [`VokraSession`] value.
    pub fn as_raw(&self) -> *const CVokraSession {
        self.handle as *const CVokraSession
    }
}

impl Drop for VokraSession {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: `self.handle` was produced by `from_file`/`retain`,
            // hasn't been double-destroyed (Drop only runs once by
            // language guarantee), and no other alias exists in safe code
            // because `!Clone`.
            unsafe { vokra_session_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_file_rejects_nul_bytes() {
        // Interior NULs must be surfaced as InvalidArgument, NOT panic.
        // (`CString::new` is documented to reject them; we ensure the
        // mapping hits the error path before any C ABI call is made.)
        //
        // We match on the Result rather than `expect_err` because
        // `VokraSession` deliberately omits `Debug` — the RAII wrapper
        // holds a raw pointer whose "printed value" is meaningless and
        // could tempt callers to introspect it.
        match VokraSession::from_file("foo\0bar.gguf") {
            Ok(_) => panic!("interior NUL must fail early, not produce a session"),
            Err(err) => {
                assert_eq!(err.status, crate::ffi::capi::VokraStatus::InvalidArgument);
                assert!(err.message.contains("NUL"));
            }
        }
    }

    // Note: `from_file` with a real GGUF is exercised at the WP integration
    // test level (T14 ASR demo / T15 TTS demo in Godot), not here — this
    // crate is a cdylib without a GGUF fixture, and duplicating M0's
    // Silero/Whisper fixture bytes would violate the isolation principle
    // (integrations pull artifacts, they don't shadow them).
}
