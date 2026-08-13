//! `extern "C"` declarations of the subset of the Vokra C ABI
//! (`include/vokra.h`, ADR-0003) that this crate's 5-fn JNI scaffold calls.
//!
//! These are the SAME symbols that `crates/vokra-capi` defines as
//! `#[no_mangle] pub extern "C" fn`, folded into `libvokra_android.so`
//! through the rlib link edge declared in `Cargo.toml`. Every `extern "C"`
//! declaration below resolves at link time — no `dlopen("libvokra.so")`
//! runtime lookup is needed on Android; the JVM loads the single
//! `libvokra_android.so` via `System.loadLibrary` and finds both the
//! `Java_com_vokra_*` JNI trampolines and the folded `vokra_*` C ABI in
//! the same shared object.
//!
//! # Why redeclare instead of re-importing from `vokra-capi`?
//!
//! Same rationale as `integrations/vokra-godot/src/ffi/capi.rs`:
//! `vokra-capi`'s inner modules (`session`, `error`, ...) are private —
//! the crate exposes ONLY its `#[no_mangle]` C symbols. So we call them
//! via `extern "C"` just like any other C consumer. This keeps the
//! contract at the header (`include/vokra.h`), not at Rust internals.
//!
//! # Scope
//!
//! Only 3 vokra symbols are called from this scaffold:
//! - [`vokra_session_create_from_file`] — line 497 of `include/vokra.h`
//!   (v1.0-rc baseline).
//! - [`vokra_session_destroy`] — line 564.
//! - [`vokra_last_error`] — line 312.
//!
//! ASR / TTS / VAD / streaming / AEC / S2S wrappers are deferred to a
//! rolling follow-up per ADR M4-kotlin §7 "後続実装 WP の起票"; the C ABI
//! v1.0-rc baseline is 33 fn + 11 typedef and is not fully mirrored here.

use core::ffi::{c_char, c_int};

/// Mirror of `VOKRA_OK = 0` from `vokra_status_t` (`include/vokra.h` L51).
/// Only the success value is needed by the current scaffold — the Kotlin
/// caller reads the diagnostic message through `nativeGetLastError` rather
/// than switching on the raw integer.
pub const VOKRA_OK: c_int = 0;

// Opaque C ABI handle; we only ever pass a `*mut vokra_session_t` in and
// out and never dereference the pointee.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct vokra_session_t {
    _opaque: [u8; 0],
}

// Edition 2024 requires `unsafe extern` for FFI blocks; the callee-safety
// contract on each fn is inherited from `include/vokra.h` documentation.
unsafe extern "C" {
    /// `include/vokra.h` L497 — loads a GGUF and creates a CPU session.
    ///
    /// Returns `VOKRA_OK` on success; on failure leaves `out_session`
    /// untouched and populates the thread-local errno for the paired
    /// [`vokra_last_error`] read.
    pub fn vokra_session_create_from_file(
        path_utf8: *const c_char,
        out_session: *mut *mut vokra_session_t,
    ) -> c_int;

    /// `include/vokra.h` L564 — frees a session handle. NULL is a
    /// documented no-op.
    pub fn vokra_session_destroy(session: *mut vokra_session_t);

    /// `include/vokra.h` L312 — returns the calling thread's last error
    /// message (NUL-terminated UTF-8), or NULL if no error is recorded on
    /// this thread. The pointer is owned by Vokra and stays valid until
    /// the next error on the same thread; never `free()` it.
    pub fn vokra_last_error() -> *const c_char;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `vokra_last_error` must be safe to call from off-device unit tests —
    /// it never allocates and never panics (ADR-0003 §3-b). Result is
    /// either NULL (no error yet) or a valid C string; we only assert
    /// that the call itself does not crash.
    #[test]
    fn last_error_can_be_called() {
        // SAFETY: `vokra_last_error` is documented never-fail / never-alloc
        // and thread-local — safe to call from any thread.
        let _ = unsafe { vokra_last_error() };
    }

    /// `vokra_session_destroy(NULL)` is a documented no-op — this test
    /// prevents accidental removal of that contract on our side.
    #[test]
    fn session_destroy_null_is_noop() {
        // SAFETY: NULL is the documented no-op path (`include/vokra.h`
        // L560).
        unsafe { vokra_session_destroy(core::ptr::null_mut()) };
    }
}
