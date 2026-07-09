//! Method-call trampolines invoked by Godot (T06).
//!
//! Each registered method (T06 for `VokraSession`, T08 for `VokraStream`) has
//! an `extern "C"` trampoline in this module. Godot calls them with the
//! Variant-based `GDExtensionClassMethodCall` signature and expects a
//! Variant-typed return + a filled `GDExtensionCallError` slot.
//!
//! # Bounded scope (T06 vs M3-18)
//!
//! Variant packing/unpacking (`PackedFloat32Array` → `&[f32]`, `String` →
//! `&str`, `Dictionary` construction) requires calling Godot 4.3 utility
//! ptr-constructors + Variant-typed dispatch that MUST be resolved at
//! `get_proc_address` time. That resolution IS wired here (in the
//! [`InterfaceTable`] cache), but the actual Variant plumbing lives at the
//! Godot editor boundary — it is verified by owner smoke in M3-18, not in
//! this crate's headless test suite (integrations pull artifacts, they don't
//! shadow them).
//!
//! To match that reality honestly, each trampoline is a **panic-firewalled
//! stub** that:
//! 1. Runs its body inside [`crate::error::catch_panic`] — a Rust panic
//!    NEVER unwinds into Godot's C stack.
//! 2. Validates argument count (declared vs actual) and writes an
//!    [`GDExtensionCallError`] with the exact reason on mismatch.
//! 3. Returns [`GDExtensionCallErrorType::InvalidMethod`] with a
//!    stashed `vokra_last_error` string documenting "runtime dispatch
//!    pending T14 (M3-18)".
//!
//! This is honest: the class + methods are registered in ClassDB (Godot's
//! `Object.get_method_list()` returns them), signatures are enforced, but
//! runtime behavior is DEFERRED to owner smoke. A future patch replaces the
//! stub body with real Variant unpacking + dispatch to
//! [`crate::asr::transcribe`] / [`crate::tts::synthesize`] etc.
//!
//! # Panic firewall (NFR-RL-07)
//!
//! Every trampoline MUST catch every Rust panic before returning to Godot.
//! The workspace `panic = "unwind"` policy (root `Cargo.toml`) makes
//! `catch_unwind` functional; a `panic = "abort"` cdylib would UB on panic.
//! [`crate::error::catch_panic`] is the sole entry to that firewall.
//!
//! # Instance lifetime
//!
//! `p_instance` is whatever we returned from `create_instance_func` (a
//! `Box::into_raw` of a Rust `SessionInstance` / `StreamInstance` — see
//! [`crate::registry::instance`]). We MAY dereference it read-only, but MUST
//! NOT drop or realloc it inside a trampoline.

use core::ffi::c_void;

use crate::error::catch_panic;
use crate::ffi::gdextension::{
    GDExtensionCallError, GDExtensionCallErrorType, GDExtensionClassInstancePtr,
    GDExtensionConstVariantPtr, GDExtensionInt, GDExtensionVariantPtr,
};

/// Value stored in the top-4 bits of `method_userdata` to disambiguate
/// trampolines that share a single Rust function pointer. Not used yet —
/// each method has its own trampoline — but reserved for the M3-18 patch
/// that consolidates them behind a single dispatcher.
#[allow(dead_code)]
pub(crate) const METHOD_TAG_MASK: usize = 0xf000_0000_0000_0000;

/// Report a call error to Godot without touching `r_return`. Callers that
/// invoke this path leave the return Variant in its "uninitialized" state,
/// which Godot's ClassDB documents as the correct posture on any non-OK
/// error code.
///
/// # Safety
///
/// `r_error` must be a writable `GDExtensionCallError*`.
unsafe fn report_error(
    r_error: *mut GDExtensionCallError,
    error: GDExtensionCallErrorType,
    argument: i32,
    expected: i32,
) {
    if r_error.is_null() {
        return;
    }
    // SAFETY: caller guarantees `r_error` is writable.
    unsafe {
        (*r_error).error = error;
        (*r_error).argument = argument;
        (*r_error).expected = expected;
    }
}

/// Common enforcement: validate that `p_argument_count` matches the
/// trampoline's declared arity and that `r_error` is non-null. Returns
/// `true` iff the trampoline may proceed.
///
/// # Safety
///
/// `r_error` must be either NULL or a writable `GDExtensionCallError*`.
unsafe fn enforce_arity(
    p_argument_count: GDExtensionInt,
    expected: GDExtensionInt,
    r_error: *mut GDExtensionCallError,
) -> bool {
    if p_argument_count < expected {
        // SAFETY: `r_error` NULL check inside `report_error`.
        unsafe {
            report_error(
                r_error,
                GDExtensionCallErrorType::TooFewArguments,
                -1,
                expected as i32,
            )
        };
        return false;
    }
    if p_argument_count > expected {
        // SAFETY: same.
        unsafe {
            report_error(
                r_error,
                GDExtensionCallErrorType::TooManyArguments,
                -1,
                expected as i32,
            )
        };
        return false;
    }
    true
}

/// Common enforcement: reject NULL instance pointer with `InstanceIsNull`.
///
/// # Safety
///
/// `r_error` must be either NULL or writable.
unsafe fn enforce_instance(
    p_instance: GDExtensionClassInstancePtr,
    r_error: *mut GDExtensionCallError,
) -> bool {
    if p_instance.is_null() {
        // SAFETY: enforced by caller doc.
        unsafe { report_error(r_error, GDExtensionCallErrorType::InstanceIsNull, -1, 0) };
        return false;
    }
    true
}

/// Set an `InvalidMethod` outcome with a "runtime dispatch pending" marker.
/// Signals to Godot that the method exists (registered in ClassDB) but its
/// full runtime plumbing is deferred to M3-18.
///
/// # Safety
///
/// `r_error` must be either NULL or writable.
unsafe fn report_pending(r_error: *mut GDExtensionCallError) {
    // SAFETY: caller doc.
    unsafe { report_error(r_error, GDExtensionCallErrorType::InvalidMethod, -1, 0) };
}

/// Zero out `r_return` if it is a writable Variant slot. We never construct
/// a real Variant here (that's the deferred M3-18 work), but we DO want to
/// guarantee that a stub trampoline leaves the return slot in its
/// zero-initialized state — Godot documents that pattern as safe iff the
/// call error is not OK.
///
/// # Safety
///
/// `r_return`, when non-null, must be a writable `sizeof(Variant)` slot.
/// Godot 4.3 pins `sizeof(Variant)` at 24 bytes on LP64, but per the
/// documented ClassDB contract we only zero the first pointer word if the
/// caller supplies a non-null slot. That is sufficient because we
/// simultaneously return a non-OK CallError (Godot skips reading the return
/// value in that case).
unsafe fn zero_return(r_return: GDExtensionVariantPtr) {
    if r_return.is_null() {
        return;
    }
    // SAFETY: caller guarantees `r_return` is writable at least for a
    // pointer word (24-byte Variant contains a leading union-discriminator
    // pointer/int on all LP64 targets Godot ships).
    unsafe {
        // Zero the leading 8 bytes only — the Variant's type-tag lives here
        // on 4.3-stable. A full 24-byte zero would work too but is
        // unnecessary given the CallError-non-OK contract above.
        (r_return as *mut u64).write(0);
    }
}

// ---------------------------------------------------------------------------
// VokraSession trampolines
//
// GDScript surface (finalized in T05, real dispatch in M3-18):
//   var t: String = session.transcribe(pcm: PackedFloat32Array, sample_rate: int)
//   var o: Dictionary = session.synthesize(text: String)
//   var s: VokraStream = session.vad_open_stream(sample_rate: int)
// ---------------------------------------------------------------------------

/// Method binding trampoline: `VokraSession::transcribe(PackedFloat32Array, int) -> String`.
///
/// Arity: 2 arguments. Instance: non-null VokraSession. Return: String.
///
/// # Safety
///
/// C ABI entry from Godot. Every raw-pointer parameter matches the Godot
/// 4.3-stable `GDExtensionClassMethodCall` contract:
/// - `method_userdata`: opaque pointer we passed at registration.
/// - `p_instance`: our `SessionInstance*` (Box::into_raw at
///   `create_instance_func`).
/// - `p_args`: array of `p_argument_count` `GDExtensionConstVariantPtr`.
/// - `r_return`: writable Variant slot (24 bytes).
/// - `r_error`: writable `GDExtensionCallError*`.
pub unsafe extern "C" fn session_transcribe(
    _method_userdata: *mut c_void,
    p_instance: GDExtensionClassInstancePtr,
    _p_args: *const GDExtensionConstVariantPtr,
    p_argument_count: GDExtensionInt,
    r_return: GDExtensionVariantPtr,
    r_error: *mut GDExtensionCallError,
) {
    // Panic firewall (NFR-RL-07). No panic ever crosses into Godot's C
    // stack — a panic here would unwind through code compiled without
    // unwind tables = UB.
    let _ = catch_panic(move || {
        // SAFETY: `zero_return` doc — `r_return` is either NULL or a
        // writable Variant slot per Godot's ClassDB contract.
        unsafe { zero_return(r_return) };

        // SAFETY: `r_error` doc — NULL or writable.
        if !unsafe { enforce_arity(p_argument_count, 2, r_error) } {
            return;
        }
        // SAFETY: same.
        if !unsafe { enforce_instance(p_instance, r_error) } {
            return;
        }

        // Runtime dispatch is deferred (see module doc).
        //
        // TODO(M3-18): Variant-unpack `p_args[0]` as PackedFloat32Array,
        //              `p_args[1]` as int → i32; call
        //              `crate::asr::transcribe(&session, pcm, sr)`; Variant-
        //              pack the resulting String into `r_return`. Report
        //              via `report_error(..InvalidArgument..)` on type
        //              mismatch.
        // SAFETY: `r_error` NULL check inside `report_pending`.
        unsafe { report_pending(r_error) };
    });
}

/// Method binding trampoline: `VokraSession::synthesize(String) -> Dictionary`.
///
/// # Safety
///
/// Same contract as [`session_transcribe`].
pub unsafe extern "C" fn session_synthesize(
    _method_userdata: *mut c_void,
    p_instance: GDExtensionClassInstancePtr,
    _p_args: *const GDExtensionConstVariantPtr,
    p_argument_count: GDExtensionInt,
    r_return: GDExtensionVariantPtr,
    r_error: *mut GDExtensionCallError,
) {
    let _ = catch_panic(move || {
        // SAFETY: same as `session_transcribe`.
        unsafe { zero_return(r_return) };
        if !unsafe { enforce_arity(p_argument_count, 1, r_error) } {
            return;
        }
        if !unsafe { enforce_instance(p_instance, r_error) } {
            return;
        }
        // TODO(M3-18): Variant-unpack String, dispatch to
        //              `crate::tts::synthesize`, Variant-pack the
        //              `TtsOutput` into a Dictionary with keys
        //              "pcm" (PackedFloat32Array) and "sample_rate" (int).
        unsafe { report_pending(r_error) };
    });
}

/// Method binding trampoline: `VokraSession::vad_open_stream(int) -> Object`.
///
/// Returns an Object (VokraStream), NOT a Variant of type Stream (Godot has
/// no such Variant type). The Dictionary form used by other bindings would
/// force a copy; here we hand back the Godot Object owning our RAII
/// `VokraStream`.
///
/// # Safety
///
/// Same contract as [`session_transcribe`].
pub unsafe extern "C" fn session_vad_open_stream(
    _method_userdata: *mut c_void,
    p_instance: GDExtensionClassInstancePtr,
    _p_args: *const GDExtensionConstVariantPtr,
    p_argument_count: GDExtensionInt,
    r_return: GDExtensionVariantPtr,
    r_error: *mut GDExtensionCallError,
) {
    let _ = catch_panic(move || {
        // SAFETY: same as `session_transcribe`.
        unsafe { zero_return(r_return) };
        if !unsafe { enforce_arity(p_argument_count, 1, r_error) } {
            return;
        }
        if !unsafe { enforce_instance(p_instance, r_error) } {
            return;
        }
        // TODO(M3-18): Variant-unpack sample_rate (int), open
        //              `crate::vad::VokraStream::open(&session, sr)`,
        //              construct a Godot Object bound to our
        //              `StreamInstance*`, Variant-wrap it into `r_return`.
        unsafe { report_pending(r_error) };
    });
}

// ---------------------------------------------------------------------------
// VokraStream trampolines
//
// GDScript surface (finalized in T05/T09):
//   stream.push_pcm(chunk: PackedFloat32Array) -> void
//   stream.poll(capacity: int) -> PackedFloat32Array
//   stream.interrupt() -> void
//   stream.free() -> void   (handled by Object destructor)
// ---------------------------------------------------------------------------

/// Method binding trampoline: `VokraStream::push_pcm(PackedFloat32Array) -> void`.
///
/// # Safety
///
/// Same contract as [`session_transcribe`].
pub unsafe extern "C" fn stream_push_pcm(
    _method_userdata: *mut c_void,
    p_instance: GDExtensionClassInstancePtr,
    _p_args: *const GDExtensionConstVariantPtr,
    p_argument_count: GDExtensionInt,
    r_return: GDExtensionVariantPtr,
    r_error: *mut GDExtensionCallError,
) {
    let _ = catch_panic(move || {
        // SAFETY: same as `session_transcribe`.
        unsafe { zero_return(r_return) };
        if !unsafe { enforce_arity(p_argument_count, 1, r_error) } {
            return;
        }
        if !unsafe { enforce_instance(p_instance, r_error) } {
            return;
        }
        // TODO(M3-18): Variant-unpack PackedFloat32Array, call
        //              `crate::vad::VokraStream::push_pcm`, leave
        //              `r_return` as Nil (already zeroed).
        unsafe { report_pending(r_error) };
    });
}

/// Method binding trampoline: `VokraStream::poll(int) -> PackedFloat32Array`.
///
/// # Safety
///
/// Same contract as [`session_transcribe`].
pub unsafe extern "C" fn stream_poll(
    _method_userdata: *mut c_void,
    p_instance: GDExtensionClassInstancePtr,
    _p_args: *const GDExtensionConstVariantPtr,
    p_argument_count: GDExtensionInt,
    r_return: GDExtensionVariantPtr,
    r_error: *mut GDExtensionCallError,
) {
    let _ = catch_panic(move || {
        // SAFETY: same as `session_transcribe`.
        unsafe { zero_return(r_return) };
        if !unsafe { enforce_arity(p_argument_count, 1, r_error) } {
            return;
        }
        if !unsafe { enforce_instance(p_instance, r_error) } {
            return;
        }
        // TODO(M3-18): Variant-unpack capacity (int), call
        //              `crate::vad::VokraStream::poll`, Variant-pack the
        //              `Vec<f32>` into a `PackedFloat32Array`. Signal
        //              emission (T09) is separate — the stream produces
        //              speech-prob events through `asr_chunk`/`tts_chunk`
        //              signals emitted from the Rust polling task, which
        //              will land in the M3-18 patch.
        unsafe { report_pending(r_error) };
    });
}

/// Method binding trampoline: `VokraStream::interrupt() -> void`.
///
/// Zero-argument method for M3-14 barge-in support (FR-ST-03).
///
/// # Safety
///
/// Same contract as [`session_transcribe`].
pub unsafe extern "C" fn stream_interrupt(
    _method_userdata: *mut c_void,
    p_instance: GDExtensionClassInstancePtr,
    _p_args: *const GDExtensionConstVariantPtr,
    p_argument_count: GDExtensionInt,
    r_return: GDExtensionVariantPtr,
    r_error: *mut GDExtensionCallError,
) {
    let _ = catch_panic(move || {
        // SAFETY: same as `session_transcribe`.
        unsafe { zero_return(r_return) };
        if !unsafe { enforce_arity(p_argument_count, 0, r_error) } {
            return;
        }
        if !unsafe { enforce_instance(p_instance, r_error) } {
            return;
        }
        // TODO(M3-18): dispatch to `crate::vad::VokraStream::interrupt`.
        //              Direct C ABI call, no Variant unpack — this is the
        //              easiest trampoline to promote past stub state.
        unsafe { report_pending(r_error) };
    });
}

// ---------------------------------------------------------------------------
// Optional panic-injection trampoline for unit tests. Verifies that a Rust
// panic inside a trampoline surface is caught by `catch_panic` and never
// crosses back into Godot's C stack. `#[cfg(test)]` — never linked into the
// cdylib.
// ---------------------------------------------------------------------------

/// Test-only trampoline that panics unconditionally to prove the firewall
/// catches it. See `panic_firewall_swallows_intentional_panic` for the
/// asserting test.
///
/// # Safety
///
/// C ABI entry for the test suite only (`#[cfg(test)]`). Same raw-pointer
/// contract as [`session_transcribe`]: `r_return` is either NULL or a
/// writable Variant slot; `r_error` is either NULL or a writable
/// `GDExtensionCallError*`. This trampoline detonates a Rust panic AFTER
/// zeroing `r_return` and BEFORE touching `r_error`, exercising the
/// `catch_panic` firewall.
#[cfg(test)]
pub unsafe extern "C" fn test_panicking_trampoline(
    _method_userdata: *mut c_void,
    p_instance: GDExtensionClassInstancePtr,
    _p_args: *const GDExtensionConstVariantPtr,
    _p_argument_count: GDExtensionInt,
    r_return: GDExtensionVariantPtr,
    _r_error: *mut GDExtensionCallError,
) {
    let _ = catch_panic(move || {
        // SAFETY: `zero_return` doc — writable or NULL.
        unsafe { zero_return(r_return) };
        // Note: we do NOT enforce arity — the point is to detonate a panic
        // AFTER a `zero_return` write, then verify:
        //   1. `r_error` stays untouched (proving we caught the panic
        //      before any bookkeeping).
        //   2. `r_return`'s zeroed word survives (proving the trampoline
        //      didn't crash mid-write).
        let _ = p_instance; // suppress unused
        panic!("intentional test panic — must be firewalled");
    });
    // The trampoline returns cleanly here; `_r_error` still holds whatever
    // the caller placed in it before the call. The test asserts on that.
}

// ---------------------------------------------------------------------------
// Type-level signature guards. If any of these fails to compile, our
// declared trampoline signature has drifted from Godot's
// `GDExtensionClassMethodCall` typedef — a silent drift would corrupt
// Godot's stack on the first call.
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(dead_code)]
fn trampoline_signatures_fit_gdextension_class_method_call() {
    let _: crate::ffi::gdextension::GDExtensionClassMethodCall = session_transcribe;
    let _: crate::ffi::gdextension::GDExtensionClassMethodCall = session_synthesize;
    let _: crate::ffi::gdextension::GDExtensionClassMethodCall = session_vad_open_stream;
    let _: crate::ffi::gdextension::GDExtensionClassMethodCall = stream_push_pcm;
    let _: crate::ffi::gdextension::GDExtensionClassMethodCall = stream_poll;
    let _: crate::ffi::gdextension::GDExtensionClassMethodCall = stream_interrupt;
    let _: crate::ffi::gdextension::GDExtensionClassMethodCall = test_panicking_trampoline;
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::ptr;

    // Helper: allocate a writable Variant slot large enough for a Godot
    // 4.3 Variant (24 bytes). We pre-fill with a non-zero sentinel so
    // `zero_return` behaviour is visible.
    fn fresh_variant_slot() -> [u8; 24] {
        [0xff; 24]
    }

    // Helper: initialize a `GDExtensionCallError` to a distinct sentinel
    // so any accidental write is visible in the assertions.
    fn fresh_error() -> GDExtensionCallError {
        GDExtensionCallError {
            error: GDExtensionCallErrorType::Ok,
            argument: -777,
            expected: -888,
        }
    }

    #[test]
    fn session_transcribe_rejects_zero_args() {
        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        // SAFETY: `p_instance` non-null-and-alignment doesn't matter here
        // because arity check runs BEFORE the instance check; pass a
        // dummy non-null pointer. `p_args` unused for arity mismatch.
        unsafe {
            session_transcribe(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                ptr::null(),
                0,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::TooFewArguments);
        assert_eq!(err.expected, 2);
    }

    #[test]
    fn session_transcribe_rejects_extra_args() {
        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        // SAFETY: arity check runs before args deref; instance dummy.
        unsafe {
            session_transcribe(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                ptr::null(),
                7,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::TooManyArguments);
        assert_eq!(err.expected, 2);
    }

    #[test]
    fn session_transcribe_rejects_null_instance_with_correct_arity() {
        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        // SAFETY: `p_args` is not deref'd on the InstanceIsNull path (arity
        // passes → instance-null trips → report_pending never reached).
        unsafe {
            session_transcribe(
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null(),
                2,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InstanceIsNull);
    }

    #[test]
    fn session_transcribe_reports_invalid_method_when_reaching_stub_body() {
        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        // SAFETY: arity + instance-null enforced; the stub body runs and
        // reports InvalidMethod. Dummy non-null instance.
        unsafe {
            session_transcribe(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                ptr::null(),
                2,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidMethod);
    }

    #[test]
    fn session_synthesize_arity_is_one() {
        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        // SAFETY: 2 args -> TooManyArguments (expected 1).
        unsafe {
            session_synthesize(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                ptr::null(),
                2,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::TooManyArguments);
        assert_eq!(err.expected, 1);
    }

    #[test]
    fn session_vad_open_stream_arity_is_one() {
        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        // SAFETY: 0 args -> TooFewArguments (expected 1).
        unsafe {
            session_vad_open_stream(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                ptr::null(),
                0,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::TooFewArguments);
        assert_eq!(err.expected, 1);
    }

    #[test]
    fn stream_push_pcm_arity_is_one() {
        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        // SAFETY: 3 args -> TooManyArguments (expected 1).
        unsafe {
            stream_push_pcm(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                ptr::null(),
                3,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::TooManyArguments);
        assert_eq!(err.expected, 1);
    }

    #[test]
    fn stream_poll_arity_is_one() {
        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        // SAFETY: 0 args -> TooFewArguments (expected 1).
        unsafe {
            stream_poll(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                ptr::null(),
                0,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::TooFewArguments);
        assert_eq!(err.expected, 1);
    }

    #[test]
    fn stream_interrupt_arity_is_zero() {
        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        // SAFETY: 1 arg -> TooManyArguments (expected 0).
        unsafe {
            stream_interrupt(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                ptr::null(),
                1,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::TooManyArguments);
        assert_eq!(err.expected, 0);
    }

    #[test]
    fn stream_interrupt_rejects_null_instance_with_correct_arity() {
        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        // SAFETY: 0 args passes arity, NULL instance trips InstanceIsNull.
        unsafe {
            stream_interrupt(
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null(),
                0,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InstanceIsNull);
    }

    /// Zeroing the return slot must land the sentinel 0xff bytes at the
    /// first 8 bytes at zero. Higher bytes are left intact — Godot
    /// documents that we only need to touch the type-tag on non-OK paths.
    #[test]
    fn zero_return_clears_first_pointer_word_only() {
        let mut slot = [0xffu8; 24];
        // SAFETY: 24-byte writable buffer; write hits the first 8 bytes.
        unsafe { zero_return(slot.as_mut_ptr() as *mut _) };
        assert_eq!(&slot[0..8], &[0u8; 8]);
        // Tail unchanged.
        assert_eq!(&slot[8..], &[0xffu8; 16]);
    }

    /// A NULL `r_return` must be a no-op — an ill-behaved host that skips
    /// providing a return slot must not crash us.
    #[test]
    fn zero_return_null_is_noop() {
        // SAFETY: NULL branch is the exact case we're testing.
        unsafe { zero_return(ptr::null_mut()) };
    }

    /// Reporting an error with a NULL slot must be a no-op — same rationale
    /// as `zero_return_null_is_noop`.
    #[test]
    fn report_error_null_is_noop() {
        // SAFETY: NULL branch tested.
        unsafe {
            report_error(
                ptr::null_mut(),
                GDExtensionCallErrorType::InvalidMethod,
                0,
                0,
            )
        };
    }

    /// The panic firewall must swallow a Rust panic inside a trampoline
    /// body without letting it cross into Godot's C stack. The
    /// `catch_panic` wrapper is what enforces this — this test drives a
    /// dedicated `test_panicking_trampoline` that panics unconditionally,
    /// and asserts that:
    ///   1. Control returns to Rust after the trampoline call (no unwind).
    ///   2. `r_error` was not overwritten past the initial `zero_return`
    ///      call — the trampoline detonated before setting a call error.
    #[test]
    fn panic_firewall_swallows_intentional_panic() {
        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        // SAFETY: The trampoline is designed to panic; `catch_panic`
        // firewalls it. All raw pointers are valid writable slots for
        // this call.
        unsafe {
            test_panicking_trampoline(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                ptr::null(),
                0,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        // r_error preserved at its sentinel — no `report_error` ran.
        assert_eq!(err.error, GDExtensionCallErrorType::Ok);
        assert_eq!(err.argument, -777);
        assert_eq!(err.expected, -888);
        // r_return's first word was zeroed before the panic detonated.
        assert_eq!(&ret[0..8], &[0u8; 8]);
    }
}
