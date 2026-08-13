//! Minimal hand-declared JNI ABI surface used by the 5 trampolines in
//! [`crate`]. Kept intentionally small — we do NOT redeclare the full
//! `jni.h` here; only the entry points our scaffold trampolines call.
//!
//! # Why hand-declare instead of pulling `jni-sys`
//!
//! ADR M4-kotlin §3 axis 1 — the ADR prefers pattern (B) raw
//! `extern "system"` for parity with the Metal / CUDA / Vulkan /
//! GDExtension bridges. The JNI ABI surface we touch here
//! (`JNIEnv` function table, `jstring` / `jlong` / `jclass` opaque
//! aliases, `GetStringUTFChars` / `ReleaseStringUTFChars` /
//! `NewStringUTF`) is stable across every JVM since ~2004 and small
//! enough to keep hand-written.
//!
//! # Function-table layout
//!
//! `JNIEnv` in Oracle's `jni.h` is a pointer to a `struct JNINativeInterface_`
//! containing ~230 function slots at well-known offsets. We only call three
//! (`GetStringUTFChars` at slot 169, `ReleaseStringUTFChars` at slot 170,
//! `NewStringUTF` at slot 167 — see `hotspot/share/prims/jni.h`), so we
//! declare a bespoke partial struct with `#[repr(C)]` and enough padding to
//! reach the slots we need. The exact offsets follow the Oracle-published
//! layout (unchanged since JNI 1.2, ~2000) and are safe to rely on: every
//! JVM (HotSpot, ART on Android, OpenJ9, Zulu, Corretto, ...) exposes the
//! same table shape. This is the same trick that `jni-sys` uses internally
//! — the crate's `sys::JNINativeInterface_` is a `#[repr(C)]` struct with
//! one field per slot; we take the same struct with all-but-3 fields
//! replaced by opaque padding.

use core::ffi::{c_char, c_void};

/// JNI 1.6 (matches the Android NDK minimum). Returned from `JNI_OnLoad`.
pub const JNI_VERSION_1_6: jint = 0x0001_0006;

// ---------------------------------------------------------------------------
// Opaque type aliases — same size as Oracle jni.h; we never dereference them
// as Rust values, only cast to/from raw pointers.
// ---------------------------------------------------------------------------

/// Java `int` — always 32-bit signed on every JVM.
#[allow(non_camel_case_types)]
pub type jint = i32;

/// Java `long` — always 64-bit signed on every JVM.
#[allow(non_camel_case_types)]
pub type jlong = i64;

/// Java `boolean` — always 8-bit unsigned on every JVM (`0` = false).
#[allow(non_camel_case_types)]
pub type jboolean = u8;

/// Java `String` reference — opaque JVM object.
#[allow(non_camel_case_types)]
pub type jstring = *mut c_void;

/// Java `Class` reference — opaque JVM object.
#[allow(non_camel_case_types)]
pub type jclass = *mut c_void;

// ---------------------------------------------------------------------------
// JNIEnv function table (partial). Every JVM lays these out at the same
// offsets; unused slots are padding.
// ---------------------------------------------------------------------------

/// Partial layout of `struct JNINativeInterface_` — enough slots to reach
/// `NewStringUTF` (167), `GetStringUTFChars` (169) and
/// `ReleaseStringUTFChars` (170). Slots we don't call are declared as
/// opaque `*const c_void` so that Rust can compute the same struct offsets
/// as the JVM's real vtable. Never construct one of these — only reach
/// them through a JVM-provided pointer.
///
/// See `hotspot/share/prims/jni.h` in OpenJDK for the canonical order.
#[repr(C)]
#[allow(non_snake_case)]
pub struct JNINativeInterface {
    // Slots 0..3 are reserved padding in the JNI spec.
    _reserved0: *const c_void,
    _reserved1: *const c_void,
    _reserved2: *const c_void,
    _reserved3: *const c_void,
    // Slots 4..166 are entry points we do not call — represent each as a
    // single opaque pointer so total offset math matches.
    _slots_4_166: [*const c_void; 163],
    // Slot 167: NewStringUTF(env, const char* bytes) -> jstring
    pub NewStringUTF: unsafe extern "system" fn(env: *mut JNIEnv, utf: *const c_char) -> jstring,
    // Slot 168: GetStringUTFLength(env, jstring)  — unused.
    _slot_168_get_string_utf_length: *const c_void,
    // Slot 169: GetStringUTFChars(env, jstring, jboolean* isCopy) -> const char*
    pub GetStringUTFChars: unsafe extern "system" fn(
        env: *mut JNIEnv,
        str: jstring,
        is_copy: *mut jboolean,
    ) -> *const c_char,
    // Slot 170: ReleaseStringUTFChars(env, jstring, const char*) -> void
    pub ReleaseStringUTFChars:
        unsafe extern "system" fn(env: *mut JNIEnv, str: jstring, chars: *const c_char),
    // Remaining slots after 170 are unused here — no padding needed because
    // we compute offsets from the start of the struct only.
}

/// `JNIEnv*` as passed by the JVM. It is a pointer-to-pointer-to-vtable:
/// `*JNIEnv == *JNINativeInterface`.
#[allow(non_camel_case_types)]
pub type JNIEnv = *const JNINativeInterface;

// ---------------------------------------------------------------------------
// Convenience wrappers — inline calls that indirect once through the vtable.
// ---------------------------------------------------------------------------

/// Wrapper for `(*env)->GetStringUTFChars(env, str, NULL)`. Returns a
/// pointer to a NUL-terminated UTF-8 byte buffer owned by the JVM; must be
/// released with [`release_string_utf_chars`].
///
/// # Safety
///
/// `env` must be a valid `JNIEnv*` provided by the JVM. `str` must be a
/// valid `jstring` reference (may be NULL — the JVM handles that and
/// returns NULL).
pub unsafe fn get_string_utf_chars(env: *mut JNIEnv, str: jstring) -> *const c_char {
    if env.is_null() {
        return core::ptr::null();
    }
    // SAFETY: `env` is a valid JNIEnv*; the vtable pointer is populated by
    // the JVM before any native method is called.
    let vtable = unsafe { *env };
    if vtable.is_null() {
        return core::ptr::null();
    }
    // SAFETY: The vtable pointer resolves to a JVM-owned
    // `JNINativeInterface_` struct; `GetStringUTFChars` at slot 169 is
    // present in every JNI 1.2+ implementation.
    unsafe { ((*vtable).GetStringUTFChars)(env, str, core::ptr::null_mut()) }
}

/// Wrapper for `(*env)->ReleaseStringUTFChars(env, str, chars)`. Pair with
/// every non-NULL [`get_string_utf_chars`] call.
///
/// # Safety
///
/// `env` must be a valid `JNIEnv*`; `str` must be the same `jstring` passed
/// to the paired `get_string_utf_chars`; `chars` must be the pointer that
/// call returned.
pub unsafe fn release_string_utf_chars(env: *mut JNIEnv, str: jstring, chars: *const c_char) {
    if env.is_null() || chars.is_null() {
        return;
    }
    // SAFETY: JVM-owned vtable; see `get_string_utf_chars`.
    let vtable = unsafe { *env };
    if vtable.is_null() {
        return;
    }
    // SAFETY: Slot 170 is present in every JNI 1.2+ implementation.
    unsafe { ((*vtable).ReleaseStringUTFChars)(env, str, chars) };
}

/// Wrapper for `(*env)->NewStringUTF(env, utf)`. Constructs a JVM-owned
/// Java `String` from a NUL-terminated UTF-8 C string; returns NULL if the
/// JVM cannot allocate.
///
/// # Safety
///
/// `env` must be a valid `JNIEnv*`; `utf` must be a NUL-terminated UTF-8 C
/// string (or NULL — the JVM's contract permits it, returning a Java
/// null).
pub unsafe fn new_string_utf(env: *mut JNIEnv, utf: *const c_char) -> jstring {
    if env.is_null() {
        return core::ptr::null_mut();
    }
    // SAFETY: JVM-owned vtable; see `get_string_utf_chars`.
    let vtable = unsafe { *env };
    if vtable.is_null() {
        return core::ptr::null_mut();
    }
    // SAFETY: Slot 167 is present in every JNI 1.2+ implementation.
    unsafe { ((*vtable).NewStringUTF)(env, utf) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The wrappers must accept a NULL `env` without dereferencing —
    /// off-device unit tests (this file) run without a real JVM, and the
    /// `catch_panic` guards in `lib.rs` rely on these being safe to call
    /// with a NULL env.
    #[test]
    fn get_string_utf_chars_null_env_returns_null() {
        // SAFETY: NULL env is the documented no-op path.
        let out = unsafe { get_string_utf_chars(core::ptr::null_mut(), core::ptr::null_mut()) };
        assert!(out.is_null());
    }

    #[test]
    fn release_string_utf_chars_null_env_is_noop() {
        // SAFETY: NULL env / NULL chars is the documented no-op path.
        unsafe {
            release_string_utf_chars(
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null(),
            )
        };
    }

    #[test]
    fn new_string_utf_null_env_returns_null() {
        // SAFETY: NULL env is the documented no-op path.
        let out = unsafe { new_string_utf(core::ptr::null_mut(), core::ptr::null()) };
        assert!(out.is_null());
    }

    /// `JNI_VERSION_1_6` is a fixed sentinel that the JVM checks — do not
    /// drift the constant.
    #[test]
    fn jni_version_pinned() {
        assert_eq!(JNI_VERSION_1_6, 0x0001_0006);
    }
}
