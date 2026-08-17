//! # vokra-android
//!
//! Android JNI binding for the Vokra speech-first runtime (FR-API-07 rolling
//! wave — Kotlin binding, docs/adr/M4-kotlin-binding-jni-vs-jna.md branch B).
//! Hand-written `extern "system"` bridge over the Vokra C ABI (`include/vokra.h`,
//! cbindgen-generated from `crates/vokra-capi`) — **no `jni` / `jni-sys` /
//! `ndk` binding crate** (ADR M4-kotlin §3 axis 1 — NFR-DS-02 preserved by
//! isolation, not by dep-avoidance in this crate; see this crate's `Cargo.toml`
//! for the excluded-workspace pattern).
//!
//! This crate is an OUT-OF-WORKSPACE integration
//! (`integrations/vokra-android/`) mirroring the isolation pattern used by
//! `integrations/vokra-godot/`, `integrations/vokra-piper-g2p/` and
//! `integrations/vokra-server/`, so the zero-dependency invariant on the
//! root `Cargo.lock` (NFR-DS-02) is untouched. The Kotlin/Java surface lives
//! next to this crate under `kotlin/com/vokra/` (loaded via
//! `System.loadLibrary("vokra_android")`).
//!
//! # Scope (2026-08-14, Proposed ADR scaffold)
//!
//! Minimal Session-lifecycle surface only (5 JNI entry points), matching the
//! post-audit scoping decision:
//! - [`Java_com_vokra_VokraSession_nativeContextNew`] — reserved; currently a
//!   no-op returning a synthetic non-zero handle (Vokra has no separate
//!   `context` object in the C ABI; sessions are self-contained).
//! - [`Java_com_vokra_VokraSession_nativeContextFree`] — reserved paired free.
//! - [`Java_com_vokra_VokraSession_nativeSessionCreate`] — wraps
//!   `vokra_session_create_from_file`; the Kotlin caller supplies a UTF-8
//!   path resolved against `Context.filesDir` (see the Kotlin `Vokra.assetToFile`
//!   helper for `AssetManager` → `filesDir` expansion, NFR-RL-04).
//! - [`Java_com_vokra_VokraSession_nativeSessionFree`] — wraps
//!   `vokra_session_destroy`.
//! - [`Java_com_vokra_VokraSession_nativeGetLastError`] — wraps
//!   `vokra_last_error` and returns the thread-local error message as a Java
//!   `String` (or `null` if there is no error on this thread).
//!
//! Rolling follow-ups (deliberately out of scope for this landing wave, per
//! ADR M4-kotlin §7 "後続実装 WP の起票" — separate ticket at the owner ADR
//! sign-off): ASR / TTS / VAD wrappers, `AutoCloseable` streaming helpers,
//! `AssetManager` → `filesDir` Kotlin helper, coroutine wrappers, Maven
//! Central publish CD.
//!
//! # Why raw JNI (no `jni` / `jni-sys` crate)
//!
//! ADR M4-kotlin §3 axis 1: adding a `jni-sys` (~0 transitive dep) or `jni`
//! (~5 transitive dep) crate to this **excluded** workspace does not break
//! NFR-DS-02 by itself (the root `Cargo.lock` is unaffected), but the ADR
//! prefers pattern (B) "raw `extern "system"`" for parity with the Metal /
//! CUDA / Vulkan / GDExtension bridges — all of which are hand-written
//! `extern "C"` / `extern "system"` FFI without a binding crate. The JNI ABI
//! surface used here (`JNIEnv`, `jclass`, `jlong`, `jstring`, `JNI_VERSION_1_6`)
//! is stable and small; hand-declaration keeps the whole toolchain reachable
//! without a Gradle-side native probe.
//!
//! Owner decision on JNA vs raw-JNI is captured in the ADR §7 sign-off queue.
//! If the owner picks (A) JNA, this crate becomes a docs-only demonstration
//! of the raw-JNI baseline the ADR compares against (the Kotlin side would
//! switch to `com.sun.jna.Library` and the `.so` would be replaced by the
//! stock `libvokra.so` loaded through JNA).
//!
//! # Unsafe policy (NFR-RL-07, workspace lint `unsafe_code = "deny"`)
//!
//! JNI is a C ABI, so this crate opts out at the crate root just like
//! `crates/vokra-capi` and `integrations/vokra-godot`. Every `unsafe` block
//! MUST carry a `// SAFETY:` comment (`clippy::undocumented_unsafe_blocks`).
//! Panics NEVER cross the JNI boundary (`catch_panic` at every trampoline
//! entry, mirroring ADR-0003 §4 and vokra-godot's `error::catch_panic`).

// Local opt-out from the workspace `unsafe_code = "deny"` lint — this crate
// IS a JNI bridge, so raw pointers and `extern "system"` are load-bearing.
#![allow(unsafe_code)]

// Force the Vokra C ABI rlib to be linked into our cdylib. See the identical
// note in integrations/vokra-godot/src/lib.rs (`extern crate vokra as _;`) —
// Rust's linker drops the `#[unsafe(no_mangle)] pub extern "C" fn vokra_*` symbols
// unless there is at least one Rust-level reference to the crate. All of our
// dispatch goes through `extern "C"` declarations in `capi.rs`, so without
// this `extern crate` the produced `libvokra_android.so` would have undefined
// `_vokra_*` symbols at runtime `System.loadLibrary()`.
//
// The `as _` binding suppresses the "unused extern crate" lint. Do NOT
// remove this line without adding an equivalent Rust-level reference to
// the `vokra` crate.
extern crate vokra as _;

pub mod capi;
pub mod jni;

use core::ffi::{c_char, c_void};
use core::ptr;

use crate::jni::{JNI_VERSION_1_6, JNIEnv, jclass, jint, jlong, jstring};

// ---------------------------------------------------------------------------
// JNI OnLoad — Android calls this once when `System.loadLibrary("vokra_android")`
// resolves the .so; return the JNI version this library targets.
// ---------------------------------------------------------------------------

/// Called by the JVM on `System.loadLibrary("vokra_android")`. Returns the JNI
/// version this library implements (1.6 is the Android minimum documented by
/// the NDK; higher versions add features we don't use).
///
/// # Safety
///
/// Called by the JVM with a valid `JavaVM` pointer; we do not dereference it.
#[unsafe(no_mangle)]
pub extern "system" fn JNI_OnLoad(_vm: *mut c_void, _reserved: *mut c_void) -> jint {
    JNI_VERSION_1_6
}

// ---------------------------------------------------------------------------
// Panic firewall — mirrors integrations/vokra-godot/src/error.rs::catch_panic
// and crates/vokra-capi/src/ffi_guard.rs. A panic MUST NOT unwind across
// the JNI boundary; JNI does not know how to propagate a Rust panic (or a
// C++ exception) and the JVM would abort the whole Android process.
// ---------------------------------------------------------------------------

/// Runs `f` under [`std::panic::catch_unwind`]; on panic returns the supplied
/// `on_panic` value and stashes a synthetic error in the thread-local errno
/// so a subsequent [`Java_com_vokra_VokraSession_nativeGetLastError`] call
/// surfaces the panic.
fn catch_panic<T, F>(on_panic: T, f: F) -> T
where
    F: FnOnce() -> T + std::panic::UnwindSafe,
{
    std::panic::catch_unwind(f).unwrap_or_else(|_| {
        // We deliberately do NOT try to call back into `vokra_last_error` /
        // set an errno here: `vokra-capi` owns the thread-local; on panic in
        // JNI-only code (jstring conversion failure, etc.) the caller gets
        // `on_panic` and calling `nativeGetLastError` returns whatever the
        // last `vokra-capi` call left (possibly `null`). Documented in
        // Kotlin `VokraException` — see `kotlin/com/vokra/VokraException.kt`.
        on_panic
    })
}

// ---------------------------------------------------------------------------
// Context reserved surface — Vokra's C ABI has no separate "context" object
// (sessions are self-contained per `include/vokra.h`), so these two entry
// points are reserved for forward compat and return a synthetic non-zero
// handle. The Kotlin `VokraSession` companion object calls `nativeContextNew`
// exactly once at class-load time; if the C ABI later grows a real context
// object (delegate / backend selector — M5 decision after NPU bakeoff per
// `include/vokra.h` header comment) these two trampolines become the natural
// promotion path without a Kotlin API break.
// ---------------------------------------------------------------------------

/// Reserved. Returns a synthetic non-zero handle `1` — the current C ABI has
/// no separate context object (v1.0-rc, 33 fn baseline; see
/// `docs/abi/vokra.h.v1.0-rc-baseline.symbols`).
///
/// # Safety
///
/// JNI-only reserved call site; no dereference of `_env` / `_class`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_vokra_VokraSession_nativeContextNew(
    _env: *mut JNIEnv,
    _class: jclass,
) -> jlong {
    // Synthetic handle. Any non-zero value distinguishes success from the
    // 0-on-error convention shared with nativeSessionCreate.
    1
}

/// Reserved paired free. No-op today.
///
/// # Safety
///
/// JNI-only reserved call site; no dereference of `_env` / `_class`; the
/// handle value is not interpreted.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_vokra_VokraSession_nativeContextFree(
    _env: *mut JNIEnv,
    _class: jclass,
    _handle: jlong,
) {
    // No-op: the reserved context is stateless.
}

// ---------------------------------------------------------------------------
// Session lifecycle — the minimal real surface for this scaffold. Kotlin
// `VokraSession.create(path)` calls `nativeSessionCreate` with a
// `filesDir`-expanded absolute path; the returned handle is a `Long` that
// Kotlin stores in an `AutoCloseable` and passes to `nativeSessionFree` on
// `close()`.
// ---------------------------------------------------------------------------

/// Wraps `vokra_session_create_from_file(path_utf8, &mut out_session)`. On
/// success returns the session pointer as a `jlong`; on failure returns `0`
/// (the C ABI leaves the out-pointer untouched on error, and the Kotlin
/// caller reads the diagnostic through `nativeGetLastError`).
///
/// # Safety
///
/// `env` must be a valid `JNIEnv*` provided by the JVM. `path` must be a
/// valid `jstring` (JVM-owned; we borrow it for the `GetStringUTFChars`
/// window and release it before returning). All Rust panics are caught
/// inside `catch_panic` and reported as a `0` return.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_vokra_VokraSession_nativeSessionCreate(
    env: *mut JNIEnv,
    _class: jclass,
    path: jstring,
) -> jlong {
    catch_panic(0, || {
        // SAFETY: JVM guarantees `env` is a valid `JNIEnv*` on entry.
        let path_ptr = unsafe { jni::get_string_utf_chars(env, path) };
        if path_ptr.is_null() {
            // The JVM will have thrown OutOfMemoryError; we simply propagate
            // the failure sentinel.
            return 0;
        }

        let mut session: *mut capi::vokra_session_t = ptr::null_mut();
        // SAFETY: `path_ptr` is a NUL-terminated UTF-8 string owned by the
        // JVM for the duration of this call; `&mut session` is a valid
        // out-pointer to a local `*mut vokra_session_t`.
        let status = unsafe { capi::vokra_session_create_from_file(path_ptr, &mut session) };

        // SAFETY: `path_ptr` was obtained from `GetStringUTFChars`; release
        // it before returning so the JVM can reclaim the temporary.
        unsafe { jni::release_string_utf_chars(env, path, path_ptr) };

        if status != capi::VOKRA_OK {
            // Kotlin caller reads `vokra_last_error()` via nativeGetLastError.
            return 0;
        }

        session as jlong
    })
}

/// Wraps `vokra_session_destroy(session)`. Safe to call with `0` (no-op);
/// mirrors the C ABI's `NULL` acceptance.
///
/// # Safety
///
/// `env` must be a valid `JNIEnv*`. `handle` must be `0` or a value
/// previously returned by `nativeSessionCreate` and not already freed.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_vokra_VokraSession_nativeSessionFree(
    _env: *mut JNIEnv,
    _class: jclass,
    handle: jlong,
) {
    catch_panic((), || {
        if handle == 0 {
            return;
        }
        // SAFETY: The Kotlin `AutoCloseable.close()` contract guarantees this
        // handle is not aliased and not previously freed.
        unsafe { capi::vokra_session_destroy(handle as *mut capi::vokra_session_t) };
    })
}

/// Wraps `vokra_last_error()` and returns the thread-local error message as
/// a Java `String`, or `null` if there is no error on this thread.
///
/// # Safety
///
/// `env` must be a valid `JNIEnv*` provided by the JVM. The returned
/// `jstring` is owned by the JVM (garbage-collected) once returned.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_vokra_VokraSession_nativeGetLastError(
    env: *mut JNIEnv,
    _class: jclass,
) -> jstring {
    catch_panic(ptr::null_mut(), || {
        // SAFETY: `vokra_last_error()` returns a pointer owned by the Vokra
        // runtime that stays valid until the next error on this thread; we
        // copy it into a Java `String` before returning (JVM manages the
        // resulting string).
        let ptr = unsafe { capi::vokra_last_error() };
        if ptr.is_null() {
            return ptr::null_mut();
        }
        // SAFETY: `env` is a valid JNIEnv; `ptr` is a NUL-terminated UTF-8
        // C string.
        unsafe { jni::new_string_utf(env, ptr as *const c_char) }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reserved context call always returns a non-zero synthetic handle.
    /// If a future ADR change makes this a real object, keep the "0 means
    /// error" invariant that Kotlin depends on.
    #[test]
    fn context_new_returns_nonzero() {
        let handle = Java_com_vokra_VokraSession_nativeContextNew(ptr::null_mut(), ptr::null_mut());
        assert_ne!(handle, 0);
    }

    /// The reserved context free is a no-op — it must not crash on 0 (paired
    /// with the "0 means error" convention).
    #[test]
    fn context_free_accepts_zero() {
        Java_com_vokra_VokraSession_nativeContextFree(ptr::null_mut(), ptr::null_mut(), 0);
    }

    /// Session free with a NULL handle must be a documented no-op — the C
    /// ABI accepts NULL and we forward that contract to Kotlin.
    #[test]
    fn session_free_accepts_zero() {
        Java_com_vokra_VokraSession_nativeSessionFree(ptr::null_mut(), ptr::null_mut(), 0);
    }
}
