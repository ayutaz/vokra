//! Method-call trampolines invoked by Godot (T06).
//!
//! Each registered method (T06 for `VokraSession`, T08 for `VokraStream`) has
//! an `extern "C"` trampoline in this module. Godot calls them with the
//! Variant-based `GDExtensionClassMethodCall` signature and expects a
//! Variant-typed return + a filled `GDExtensionCallError` slot.
//!
//! # T14 promotion (M3-11 partial land)
//!
//! What ships in this file today:
//!
//! - **Full dispatch** for [`stream_interrupt`]: zero-arg / void-return
//!   method that needs no Variant unpack — writes a proper Nil Variant to
//!   `r_return` on success, `InvalidMethod` on backend error.
//! - **Int Variant type validation** for [`stream_poll`] and
//!   [`session_vad_open_stream`]: the int-typed arg (capacity /
//!   sample_rate) is Variant-unpacked via
//!   [`crate::variant::variant_to_i64`], and a type mismatch surfaces as
//!   `InvalidArgument` with the exact offending index. The unpacked value
//!   is currently unused — full dispatch still returns `InvalidMethod`
//!   because the return path (PackedFloat32Array / Object) requires
//!   additional Variant plumbing.
//! - **Full dispatch** for [`session_transcribe`], [`session_synthesize`],
//!   [`stream_push_pcm`], [`stream_poll`]: T14-followup promotions.
//!   `session_transcribe` unpacks `PackedFloat32Array` + `Int` and packs a
//!   `String` return. `session_synthesize` unpacks `String` and packs a
//!   `Dictionary` return (`{"pcm": PackedFloat32Array, "sample_rate":
//!   int}`) — see [`crate::variant::pack_tts_output_into_dict_variant`].
//!   `stream_push_pcm` unpacks `PackedFloat32Array`. `stream_poll`
//!   unpacks `Int` and packs a `PackedFloat32Array` return.
//!
//! # FR-EX-08 (no silent CPU fallback)
//!
//! Every promoted trampoline surfaces backend errors as an explicit
//! `InvalidMethod` CallError; the underlying `vokra_last_error()` string
//! remains available on the same thread for GDScript-side introspection
//! via the C ABI. There is NO path that swallows a non-OK C ABI status
//! into a success Variant.
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
    GDExtensionConstVariantPtr, GDExtensionInt, GDExtensionVariantPtr, GDExtensionVariantType,
};
use crate::ffi::interface::InterfaceTable;
use crate::variant;

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
/// full runtime plumbing is deferred (see individual `TODO(future)`
/// markers on the un-promoted trampolines).
///
/// # Safety
///
/// `r_error` must be either NULL or writable.
unsafe fn report_pending(r_error: *mut GDExtensionCallError) {
    // SAFETY: caller doc.
    unsafe { report_error(r_error, GDExtensionCallErrorType::InvalidMethod, -1, 0) };
}

/// Zero out `r_return` if it is a writable Variant slot. Used on the
/// failure path (non-OK CallError) where Godot documents it will skip
/// reading the return value; success paths use [`write_nil_return`] to
/// emit a proper Nil Variant instead.
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

/// Load `p_args[i]` as a `GDExtensionConstVariantPtr`. Godot documents the
/// `p_args` array as `const GDExtensionConstVariantPtr *` — an array of Variant
/// pointers, indexed by argument position.
///
/// # Safety
///
/// - `p_args` must be a valid array of at least `i + 1` pointers.
/// - Each pointer read from that array must be a live Variant pointer for
///   the duration of its use.
#[inline]
unsafe fn arg_ptr(
    p_args: *const GDExtensionConstVariantPtr,
    i: usize,
) -> GDExtensionConstVariantPtr {
    // SAFETY: caller doc.
    unsafe { *p_args.add(i) }
}

/// Try to unpack `p_args[i]` as an `i64`. Reports
/// [`GDExtensionCallErrorType::InvalidArgument`] with the offending index and
/// the expected type ([`GDExtensionVariantType::Int`] = 2) on a type mismatch,
/// and returns `None` in that case; returns `Some(i64)` on success.
///
/// # Safety
///
/// - `p_args` is a valid array of at least `i + 1` Variant pointers.
/// - `interface` is the extension's live resolved [`InterfaceTable`].
/// - `r_error` is either NULL or a writable `GDExtensionCallError*`.
unsafe fn unpack_i64_or_report(
    interface: &InterfaceTable,
    p_args: *const GDExtensionConstVariantPtr,
    i: usize,
    r_error: *mut GDExtensionCallError,
) -> Option<i64> {
    // SAFETY: caller doc.
    let v = unsafe { arg_ptr(p_args, i) };
    // SAFETY: same.
    match unsafe { variant::variant_to_i64(interface, v) } {
        Ok(x) => Some(x),
        Err(_actual) => {
            // Report the offending index + the expected type code. Godot
            // treats `expected` as a `GDExtensionVariantType` numeric for
            // `InvalidArgument`.
            // SAFETY: caller doc.
            unsafe {
                report_error(
                    r_error,
                    GDExtensionCallErrorType::InvalidArgument,
                    i as i32,
                    GDExtensionVariantType::Int as i32,
                );
            }
            None
        }
    }
}

/// Write a Nil Variant into `r_return` for void-return methods (success
/// path). Prefers the interface's `variant_new_nil` when available;
/// otherwise falls back to a full 24-byte zero (Nil's canonical layout on
/// Godot 4.3-stable is all-zero, so this is behaviourally equivalent).
///
/// # Safety
///
/// `r_return` must be either NULL or a writable 24-byte Variant slot.
unsafe fn write_nil_return(interface: Option<&InterfaceTable>, r_return: GDExtensionVariantPtr) {
    if r_return.is_null() {
        return;
    }
    match interface {
        Some(iface) => {
            // SAFETY: caller doc — writable 24-byte slot.
            unsafe { variant::write_nil_variant(iface, r_return) };
        }
        None => {
            // Fallback: full 24-byte zero. This is Nil's canonical layout
            // on Godot 4.3-stable and matches the `Variant()` default
            // constructor's `memset(0)` posture.
            //
            // SAFETY: caller doc — writable 24 bytes.
            unsafe {
                (r_return as *mut u8).write_bytes(0, 24);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// VokraSession trampolines
//
// GDScript surface (finalized in T05; promotion status per method — see
// module doc §T14 promotion for the full breakdown):
//   var t: String = session.transcribe(pcm: PackedFloat32Array, sample_rate: int)
//   var o: Dictionary = session.synthesize(text: String)
//   var s: VokraStream = session.vad_open_stream(sample_rate: int)
// ---------------------------------------------------------------------------

/// Method binding trampoline: `VokraSession::transcribe(PackedFloat32Array, int) -> String`.
///
/// Arity: 2 arguments. Instance: non-null VokraSession. Return: String.
///
/// # Runtime dispatch (M3-11 T14-followup — full promotion)
///
/// After the arity + instance + extension-init guards, the trampoline
/// walks the Variant args left-to-right (FR-EX-08 type-order discipline).
/// Argument-shaped failures (type / range) are surfaced BEFORE
/// instance-state failures so a GDScript coding error lands with the
/// typed CallError closest to the offending source location:
///
/// 1. Type-check `p_args[0]` as `PackedFloat32Array` — else
///    `InvalidArgument(0, PackedFloat32Array)`.
/// 2. Type-check + unpack `p_args[1]` as `Int` — else
///    `InvalidArgument(1, Int)`.
/// 3. Range-guard the unpacked `sample_rate` against `1..=i32::MAX`.
///    Godot Int is i64; the C ABI takes i32. Out-of-range values
///    (including 0 and negatives) surface as `InvalidArgument(1, Int)`
///    at the exact offending arg index — not left to `asr::transcribe`'s
///    inner check whose failure would return through the generic
///    `InvalidMethod` posture. Placing this ahead of the session-state
///    check matches the task-spec ordering (a) arg0 PFA → (b) arg1 int
///    + range guard → (c) recover session.
/// 4. Recover the `SessionInstance` behind `p_instance`. If `inner ==
///    None` (GDScript coded `transcribe` before `load(path)`), report
///    `InvalidMethod` — matches the same posture as [`stream_interrupt`]
///    when a stream is never opened.
/// 5. Unpack `p_args[0]` as `PackedFloat32Array` → `&[f32]` via
///    [`crate::variant::variant_to_packed_float32_slice`] and dispatch
///    to [`crate::asr::transcribe`]. On backend `Err`, report
///    `InvalidMethod` (the thread-local `vokra_last_error()` still
///    carries the detail for GDScript-side inspection). On backend
///    `Ok(text)`, pack the returned String into `r_return` via
///    [`crate::variant::variant_from_string_utf8`] and leave `r_error`
///    at its incoming `Ok` state.
///
/// # FR-EX-08 (no silent CPU fallback)
///
/// Every non-happy path surfaces an explicit CallError. There is NO
/// path that swallows a backend error into a Nil/empty Variant. The
/// only path that touches `r_return` with a real value is the ASR
/// success branch.
///
/// # Panic firewall (NFR-RL-07)
///
/// The whole body runs inside [`catch_panic`]. A panic inside
/// `variant_to_packed_float32_slice`'s closure still triggers the
/// RAII-guarded destructor for the temp PackedFloat32Array under
/// `panic = "unwind"` (workspace default) — no CowData refcount leak.
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
    p_args: *const GDExtensionConstVariantPtr,
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

        // Reach into the resolved GDExtension interface. `with_interface`
        // returns `None` iff the extension has not been initialised (or
        // was deinitialised) — that CANNOT happen on a live ClassDB
        // dispatch (the class registration itself lives inside the same
        // extension state), so a None here is a "corrupt / racing test
        // harness" scenario that we surface as an explicit
        // `InvalidMethod` (FR-EX-08 honest failure).
        //
        // SAFETY: `p_args` is a `p_argument_count == 2` array of Variant
        // pointers per Godot's contract. `p_instance` was validated non-null
        // above; it came from `create_session_instance`'s `Box::into_raw`
        // and points at a live `SessionInstance`. `r_return` and `r_error`
        // are writable per Godot's ClassDB contract (checked NULL / arity
        // above).
        let handled = unsafe {
            crate::with_interface(|iface| {
                dispatch_session_transcribe(iface, p_instance, p_args, r_return, r_error);
            })
        };
        if handled.is_none() {
            // Extension not initialised — FR-EX-08.
            // SAFETY: `r_error` doc — NULL or writable.
            unsafe { report_pending(r_error) };
        }
    });
}

/// Inner dispatch for [`session_transcribe`]. Kept out of the trampoline
/// body so the entry-point stays a thin panic-firewall + interface-lookup
/// wrapper — mirrors the split between `stream_interrupt`'s outer + inner
/// halves, and keeps the panic-safe closure inside `with_interface` small
/// enough to reason about.
///
/// Every arm of the match on `Result<Result<String, VokraError>, GDExtensionVariantType>`
/// writes to exactly one of `r_return` (Ok(Ok)) or `r_error` (everything
/// else); the caller-supplied slots are never double-written.
///
/// # Safety
///
/// - `p_instance` is a live `*mut SessionInstance` (`Box::into_raw`).
/// - `p_args` is a valid array of at least 2 `GDExtensionConstVariantPtr`
///   entries.
/// - `r_return` is NULL or a writable Variant slot (24 bytes).
/// - `r_error` is NULL or a writable `GDExtensionCallError*`.
/// - `iface` is a live resolved [`crate::ffi::interface::InterfaceTable`].
unsafe fn dispatch_session_transcribe(
    iface: &crate::ffi::interface::InterfaceTable,
    p_instance: GDExtensionClassInstancePtr,
    p_args: *const GDExtensionConstVariantPtr,
    r_return: GDExtensionVariantPtr,
    r_error: *mut GDExtensionCallError,
) {
    // (1) Type-check `p_args[0]` as PackedFloat32Array.
    //
    // SAFETY: `p_args` has at least 2 valid entries per caller doc.
    let arg0 = unsafe { arg_ptr(p_args, 0) };
    // SAFETY: `arg0` is a live Variant pointer (loaded from `p_args`).
    let ty0 = unsafe { crate::variant::variant_get_type(iface, arg0) };
    if ty0 != GDExtensionVariantType::PackedFloat32Array {
        // SAFETY: `r_error` doc.
        unsafe {
            report_error(
                r_error,
                GDExtensionCallErrorType::InvalidArgument,
                0,
                GDExtensionVariantType::PackedFloat32Array as i32,
            );
        }
        return;
    }

    // (2) Type-check `p_args[1]` as Int. `unpack_i64_or_report` combines
    // the type check + read; on mismatch it emits `InvalidArgument(1, Int)`
    // and returns None.
    //
    // SAFETY: `p_args` has at least 2 valid entries. `r_error` NULL check
    // inside `report_error`.
    let Some(sample_rate_i64) = (unsafe { unpack_i64_or_report(iface, p_args, 1, r_error) }) else {
        // Type mismatch already reported.
        return;
    };

    // (3) Range-guard `sample_rate`. Godot Int is i64; the C ABI takes
    // i32. Zero + negative are rejected here (asr.rs's inner check would
    // do the same but returning through `InvalidMethod` — surfacing
    // `InvalidArgument(1, Int)` at the exact offending arg gives the
    // GDScript surface a better error posture). Placed BEFORE the
    // `SessionInstance` deref so a bad-arg + never-loaded-session
    // combination still surfaces the argument error at the closest
    // source location (task-spec ordering (b) before (c)).
    if sample_rate_i64 <= 0 || sample_rate_i64 > i32::MAX as i64 {
        // SAFETY: `r_error` doc.
        unsafe {
            report_error(
                r_error,
                GDExtensionCallErrorType::InvalidArgument,
                1,
                GDExtensionVariantType::Int as i32,
            );
        }
        return;
    }
    let sample_rate = sample_rate_i64 as i32;

    // (4) Recover `SessionInstance` behind `p_instance`. If `inner == None`
    // (GDScript coded `transcribe` before `load(path)`), report
    // `InvalidMethod`. Matches the same posture as `stream_interrupt` on
    // an unopened stream.
    //
    // SAFETY: `p_instance` came from `create_session_instance` (see
    // `crate::registry`), has NOT been freed (Godot's ClassDB lifecycle
    // guarantees `free_instance_func` is called AFTER this trampoline
    // returns), and no other Rust alias exists because Godot's ClassDB
    // dispatch is single-threaded on the main thread. We take a shared
    // borrow only — `asr::transcribe` does not mutate the session state.
    let session_inst = unsafe { &*(p_instance as *const crate::registry::SessionInstance) };
    let Some(session) = session_inst.inner.as_ref() else {
        // Session was never loaded via `load(path)`. GDScript coding error.
        //
        // SAFETY: `r_error` doc.
        unsafe { report_pending(r_error) };
        return;
    };

    // (5) Unpack PackedFloat32Array → `&[f32]` and dispatch to
    // `asr::transcribe`. The unpack helper's Err path signals a type
    // mismatch — impossible here because we type-checked above, but
    // defense-in-depth: any Err surfaces as `InvalidArgument(0, PFA)`.
    //
    // SAFETY: `iface` is a live resolved interface; `arg0` is a live
    // PackedFloat32Array Variant per the type check above. The closure
    // borrows `&session` (a shared borrow, single-threaded main-thread
    // dispatch) and `sample_rate` (an `i32` copy).
    let unpack_result: Result<Result<String, crate::error::VokraError>, GDExtensionVariantType> = unsafe {
        crate::variant::variant_to_packed_float32_slice(iface, arg0, |pcm: &[f32]| {
            crate::asr::transcribe(session, pcm, sample_rate)
        })
    };

    match unpack_result {
        Err(_actual_ty) => {
            // Type check race — should be unreachable given the pre-check
            // above, but defensive.
            //
            // SAFETY: `r_error` doc.
            unsafe {
                report_error(
                    r_error,
                    GDExtensionCallErrorType::InvalidArgument,
                    0,
                    GDExtensionVariantType::PackedFloat32Array as i32,
                );
            }
        }
        Ok(Ok(text)) => {
            // Backend success. Pack the returned String into `r_return`.
            // `r_error` stays at its incoming `Ok` state (Godot passed it
            // in as CallError::Ok).
            //
            // A NULL `r_return` on the success path is a Godot host
            // contract violation (Godot 4.3 documents `r_return` as
            // non-null when the method has a declared return value —
            // and `transcribe` DOES). Skip the pack in that case rather
            // than UB'ing the constructor.
            if !r_return.is_null() {
                // SAFETY: `iface` is live; `r_return` is non-null and
                // writable 24 bytes (Godot ClassDB contract).
                unsafe { crate::variant::variant_from_string_utf8(iface, r_return, &text) };
            }
        }
        Ok(Err(_backend_err)) => {
            // Backend failure. Surface as `InvalidMethod`;
            // `vokra_last_error()` on this thread still holds the detail
            // for GDScript-side inspection via the C ABI.
            //
            // FR-EX-08: no silent success.
            //
            // SAFETY: `r_error` doc.
            unsafe { report_pending(r_error) };
        }
    }
}

/// Method binding trampoline: `VokraSession::synthesize(String) -> Dictionary`.
///
/// Arity: 1 argument. Instance: non-null VokraSession. Return: Dictionary
/// of shape `{"pcm": PackedFloat32Array, "sample_rate": int}`.
///
/// # Runtime dispatch (M3-11 T14-followup — full promotion)
///
/// After the arity + instance + extension-init guards, the trampoline
/// walks the Variant args left-to-right (FR-EX-08 type-order discipline):
///
/// 1. Type-check `p_args[0]` as `String` — else
///    `InvalidArgument(0, String)`. Uses
///    [`crate::variant::variant_to_string_owned`], whose Err path
///    surfaces the actual mismatched type (defense-in-depth vs. a
///    corrupt caller).
/// 2. Recover the `SessionInstance` behind `p_instance`. If `inner ==
///    None` (GDScript coded `synthesize` before `load(path)`), report
///    `InvalidMethod` — matches the same posture as
///    [`session_transcribe`] on an unloaded session.
/// 3. Dispatch to [`crate::tts::synthesize`] with the owned Rust
///    `String`. Any interior NUL (rejected by the underlying
///    [`std::ffi::CString`] gate) or backend failure surfaces as
///    `InvalidMethod` (the thread-local `vokra_last_error()` still
///    carries the detail).
/// 4. On backend `Ok(TtsOutput { pcm, sample_rate })`, pack the payload
///    into `r_return` via
///    [`crate::variant::pack_tts_output_into_dict_variant`] — the
///    {pcm, sample_rate} Dictionary is built entirely from resolved
///    fn pointers (no Rust panic points between typed-handle construct
///    and destroy).
///
/// # FR-EX-08 (no silent CPU fallback)
///
/// Every non-happy path surfaces an explicit CallError. There is NO
/// path that swallows a backend error into a Nil / empty Variant. The
/// only path that touches `r_return` with a real value is the TTS
/// success branch.
///
/// # Panic firewall (NFR-RL-07)
///
/// The whole body runs inside [`catch_panic`]. A panic inside
/// `variant_to_string_owned` still runs `StringGuard::drop` under
/// `panic = "unwind"` (workspace default) → the temp String CowData
/// refcount is released. A panic inside
/// `pack_tts_output_into_dict_variant` runs both `DictionaryGuard::drop`
/// and any nested `StringGuard` cleanup — no CowData leak.
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
pub unsafe extern "C" fn session_synthesize(
    _method_userdata: *mut c_void,
    p_instance: GDExtensionClassInstancePtr,
    p_args: *const GDExtensionConstVariantPtr,
    p_argument_count: GDExtensionInt,
    r_return: GDExtensionVariantPtr,
    r_error: *mut GDExtensionCallError,
) {
    // Panic firewall (NFR-RL-07). A panic across the C ABI would UB —
    // Godot's C stack has no unwind tables.
    let _ = catch_panic(move || {
        // SAFETY: same as `session_transcribe`.
        unsafe { zero_return(r_return) };
        if !unsafe { enforce_arity(p_argument_count, 1, r_error) } {
            return;
        }
        if !unsafe { enforce_instance(p_instance, r_error) } {
            return;
        }

        // Reach into the resolved GDExtension interface. `with_interface`
        // returns `None` iff the extension has not been initialised (or
        // was deinitialised) — that CANNOT happen on a live ClassDB
        // dispatch (the class registration itself lives inside the same
        // extension state), so a None here is a "corrupt / racing test
        // harness" scenario that we surface as an explicit
        // `InvalidMethod` (FR-EX-08 honest failure).
        //
        // SAFETY: `p_args` is a `p_argument_count == 1` array of Variant
        // pointers per Godot's contract. `p_instance` was validated
        // non-null above; it came from `create_session_instance`'s
        // `Box::into_raw` and points at a live `SessionInstance`.
        // `r_return` and `r_error` are writable per Godot's ClassDB
        // contract (checked NULL / arity above).
        let handled = unsafe {
            crate::with_interface(|iface| {
                dispatch_session_synthesize(iface, p_instance, p_args, r_return, r_error);
            })
        };
        if handled.is_none() {
            // Extension not initialised — FR-EX-08.
            // SAFETY: `r_error` doc — NULL or writable.
            unsafe { report_pending(r_error) };
        }
    });
}

/// Inner dispatch for [`session_synthesize`]. Same split rationale as
/// [`dispatch_session_transcribe`]: keeps the panic-safe closure inside
/// `with_interface` small enough to reason about, and every arm of the
/// match writes to exactly one of `r_return` (Ok success) or `r_error`
/// (everything else) — the caller-supplied slots are never double-written.
///
/// # Safety
///
/// - `p_instance` is a live `*mut SessionInstance` (`Box::into_raw`).
/// - `p_args` is a valid array of at least 1 `GDExtensionConstVariantPtr`
///   entry.
/// - `r_return` is NULL or a writable 24-byte Variant slot.
/// - `r_error` is NULL or a writable `GDExtensionCallError*`.
/// - `iface` is a live resolved [`crate::ffi::interface::InterfaceTable`].
unsafe fn dispatch_session_synthesize(
    iface: &crate::ffi::interface::InterfaceTable,
    p_instance: GDExtensionClassInstancePtr,
    p_args: *const GDExtensionConstVariantPtr,
    r_return: GDExtensionVariantPtr,
    r_error: *mut GDExtensionCallError,
) {
    // (1) Type-check + unpack `p_args[0]` as String → owned Rust String.
    //
    // SAFETY: `p_args` has at least 1 valid entry per caller doc.
    let arg0 = unsafe { arg_ptr(p_args, 0) };
    // SAFETY: `iface` is live; `arg0` is a live Variant pointer.
    let text = match unsafe { crate::variant::variant_to_string_owned(iface, arg0) } {
        Ok(s) => s,
        Err(_actual_ty) => {
            // Type mismatch → `InvalidArgument(0, String)`. FR-EX-08:
            // never a silent success.
            //
            // SAFETY: `r_error` doc.
            unsafe {
                report_error(
                    r_error,
                    GDExtensionCallErrorType::InvalidArgument,
                    0,
                    GDExtensionVariantType::String as i32,
                );
            }
            return;
        }
    };

    // (2) Recover `SessionInstance` behind `p_instance`. If `inner ==
    // None` (GDScript coded `synthesize` before `load(path)`), report
    // `InvalidMethod`. Matches the same posture as
    // `dispatch_session_transcribe` on an unloaded session.
    //
    // SAFETY: `p_instance` came from `create_session_instance` (see
    // `crate::registry`), has NOT been freed (Godot's ClassDB lifecycle
    // guarantees `free_instance_func` is called AFTER this trampoline
    // returns), and no other Rust alias exists because Godot's ClassDB
    // dispatch is single-threaded on the main thread. We take a shared
    // borrow only — `tts::synthesize` does not mutate the session
    // state.
    let session_inst = unsafe { &*(p_instance as *const crate::registry::SessionInstance) };
    let Some(session) = session_inst.inner.as_ref() else {
        // Session was never loaded via `load(path)`. GDScript coding
        // error.
        //
        // SAFETY: `r_error` doc.
        unsafe { report_pending(r_error) };
        return;
    };

    // (3) Dispatch to `tts::synthesize`. Interior NULs in the input
    // text (rejected by the underlying CString gate) or a backend
    // failure surface as `InvalidMethod`; `vokra_last_error()` on this
    // thread still holds the detail for GDScript-side inspection.
    let output = match crate::tts::synthesize(session, &text) {
        Ok(o) => o,
        Err(_backend_err) => {
            // Backend failure. FR-EX-08: no silent success.
            //
            // SAFETY: `r_error` doc.
            unsafe { report_pending(r_error) };
            return;
        }
    };

    // (4) Pack {"pcm": PackedFloat32Array, "sample_rate": int} into
    // `r_return`. A NULL `r_return` on the success path is a Godot
    // host contract violation (Godot 4.3 documents `r_return` as
    // non-null when the method has a declared return value — and
    // `synthesize` DOES). Skip the pack in that case rather than UB'ing
    // the constructor.
    if !r_return.is_null() {
        // SAFETY: `iface` is live; `r_return` is non-null and writable
        // 24 bytes (Godot ClassDB contract). `output.pcm.len()` fits
        // GDExtensionInt on LP64 (isize::MAX == i64::MAX).
        unsafe {
            crate::variant::pack_tts_output_into_dict_variant(
                iface,
                r_return,
                &output.pcm,
                output.sample_rate,
            );
        }
    }
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
    p_args: *const GDExtensionConstVariantPtr,
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

        // T14 partial promotion: validate `p_args[0]` (sample_rate) as
        // Int. Extension must be initialised for the Variant unpack to
        // reach the resolved constructor — a `None` from
        // `with_interface` means the extension is being called before
        // `vokra_gdextension_init` completed (or after deinit), which
        // FR-EX-08 surfaces as an explicit `InvalidMethod`.
        //
        // SAFETY: `p_args` is `p_argument_count == 1` valid pointer;
        // `unpack_i64_or_report` internally uses `p_args[0]`. `r_error`
        // and interface access follow their doc contracts.
        let unpacked = unsafe {
            crate::with_interface(|iface| unpack_i64_or_report(iface, p_args, 0, r_error))
        };
        let Some(inner) = unpacked else {
            // Extension not initialised — no interface available.
            // SAFETY: `r_error` doc.
            unsafe { report_pending(r_error) };
            return;
        };
        let Some(_sample_rate) = inner else {
            // `unpack_i64_or_report` already wrote `InvalidArgument` on
            // type mismatch; nothing more to do.
            return;
        };

        // TODO(future): Open `crate::vad::VokraStream::open(&session,
        //   sample_rate as i32)`; construct a Godot Object bound to our
        //   `StreamInstance*`; Variant-wrap it into `r_return` via
        //   `object_get_instance_binding` + object Variant constructor.
        //   Rationale for holding off: Object wrapping requires the
        //   `InstanceBinding` posture that `crate::registry` §Instance
        //   lifetime defers to owner smoke, and the object Variant
        //   packer requires resolving `object_new` + refcount
        //   arithmetic. The int-arg validation above IS a real
        //   improvement over the pre-T14 stub — a GDScript coding error
        //   like `session.vad_open_stream("16000")` now surfaces as
        //   InvalidArgument(index=0, expected=Int) instead of the vague
        //   "runtime dispatch pending" InvalidMethod.
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
/// Full dispatch promoted as of the M3-11 T14 followup. Flow:
///
/// 1. Enforce arity (`1`) and non-null instance (`InstanceIsNull` /
///    `TooFew`/`TooMany` on failure).
/// 2. Recover `&mut StreamInstance` from `p_instance`. If
///    `inner == None`, report `InvalidMethod` — GDScript coding error
///    (push_pcm before `session.vad_open_stream()`).
/// 3. Call [`crate::variant::variant_to_packed_float32_slice`] to type-check
///    `p_args[0]` and borrow the underlying `&[f32]` for one scope. On
///    type mismatch, report `InvalidArgument` with
///    `expected = PackedFloat32Array` (FR-EX-08 — no silent success).
/// 4. Inside the closure, call
///    [`crate::vad::VokraStream::push_pcm`]. On success, write a proper
///    Nil into `r_return`; on backend failure, surface `InvalidMethod`
///    (matching the [`stream_interrupt`] posture).
///
/// # Safety
///
/// Same contract as [`session_transcribe`].
pub unsafe extern "C" fn stream_push_pcm(
    _method_userdata: *mut c_void,
    p_instance: GDExtensionClassInstancePtr,
    p_args: *const GDExtensionConstVariantPtr,
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

        // Recover the `StreamInstance` mutable borrow. Godot's ClassDB
        // dispatch is single-threaded on the main thread (documented) and
        // `push_pcm` does NOT emit signals or reenter Godot — no aliasing
        // hazard exists. See [`stream_interrupt`] for the full rationale.
        //
        // SAFETY: `p_instance` came from `create_stream_instance` (see
        // `crate::registry`), has NOT been freed (Godot's ClassDB
        // lifecycle guarantees `free_instance_func` runs AFTER this
        // trampoline returns), and no other Rust alias exists because we
        // consume the `*mut c_void` uniquely on this main-thread dispatch
        // path.
        let stream_inst = unsafe { &mut *(p_instance as *mut crate::registry::StreamInstance) };
        let Some(stream) = stream_inst.inner.as_mut() else {
            // GDScript coding error: push_pcm on a stream that was never
            // opened via `session.vad_open_stream(sr)`. FR-EX-08 — no
            // silent success.
            //
            // SAFETY: `r_error` doc.
            unsafe { report_pending(r_error) };
            return;
        };

        // Borrow the arg-0 Variant → &[f32] via the resolved interface.
        // Extension MUST be initialised for `variant_to_packed_float32_slice`
        // to reach live fn pointers; a `None` from `with_interface` means
        // the extension is being called before `vokra_gdextension_init`
        // completed (or after deinit), which FR-EX-08 surfaces as
        // `InvalidMethod`.
        //
        // SAFETY: `p_args` is an array of exactly 1 valid pointer (arity
        // check passed). `p_args[0]` is a live Variant per Godot's ClassDB
        // dispatch contract. The interface reference is live for the
        // duration of the `with_interface` closure.
        let unpacked = unsafe {
            crate::with_interface(|iface| {
                let arg0 = arg_ptr(p_args, 0);
                crate::variant::variant_to_packed_float32_slice(iface, arg0, |pcm| {
                    // The slice is a direct borrow into Godot's CoW buffer
                    // and is valid for this closure only. Forward to
                    // VokraStream::push_pcm and translate its Result into a
                    // small enum for the outer scope.
                    stream.push_pcm(pcm)
                })
            })
        };
        let Some(unpack_result) = unpacked else {
            // Extension not initialised — no interface available.
            //
            // SAFETY: `r_error` doc.
            unsafe { report_pending(r_error) };
            return;
        };

        match unpack_result {
            Ok(Ok(())) => {
                // Success — write a proper Nil into `r_return` and leave
                // `r_error` untouched (Godot passes it in with
                // `error = Ok`).
                //
                // SAFETY: `r_return` is NULL or writable 24 bytes.
                unsafe {
                    crate::with_interface(|iface| {
                        write_nil_return(Some(iface), r_return);
                    });
                }
            }
            Ok(Err(_backend_err)) => {
                // Backend failure. Surface as `InvalidMethod`;
                // `vokra_last_error()` on this thread still holds the
                // detail for GDScript-side inspection via the C ABI.
                // FR-EX-08 — no silent success.
                //
                // SAFETY: `r_error` doc.
                unsafe { report_pending(r_error) };
            }
            Err(_actual_type) => {
                // Type mismatch on `p_args[0]`. Surface with the exact
                // expected type code so a GDScript coding error like
                // `stream.push_pcm("hi")` produces a diagnostic pointing
                // at the offending argument index + expected type.
                //
                // SAFETY: `r_error` doc.
                unsafe {
                    report_error(
                        r_error,
                        GDExtensionCallErrorType::InvalidArgument,
                        0,
                        GDExtensionVariantType::PackedFloat32Array as i32,
                    );
                }
            }
        }
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
    p_args: *const GDExtensionConstVariantPtr,
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

        // T14-followup: full dispatch. Steps:
        //   (a) Variant-unpack `p_args[0]` as Int.
        //   (b) Validate range fits `usize` (FR-EX-08 explicit
        //       InvalidArgument on negative / oversized).
        //   (c) Call `crate::vad::VokraStream::poll(capacity)`.
        //   (d) Variant-pack the returned `Vec<f32>` as a
        //       PackedFloat32Array via
        //       `crate::variant::pack_f32_slice_into_variant`.
        //
        // FR-EX-08: any non-viable path (extension not initialised,
        // stream never opened, backend `poll` error, out-of-range
        // capacity) surfaces an explicit CallError — never a silent
        // success.

        // (a) Unpack + (b) range-check.
        //
        // SAFETY: `p_args` is `p_argument_count == 1` valid pointer;
        // `unpack_i64_or_report` internally uses `p_args[0]`.
        let unpacked = unsafe {
            crate::with_interface(|iface| unpack_i64_or_report(iface, p_args, 0, r_error))
        };
        let Some(inner) = unpacked else {
            // Extension not initialised — no interface available.
            // SAFETY: `r_error` doc.
            unsafe { report_pending(r_error) };
            return;
        };
        let Some(capacity_i64) = inner else {
            // Type mismatch already reported by `unpack_i64_or_report`.
            return;
        };

        // Range validation: reject negative and portable > usize::MAX.
        // On LP64 the second branch is unreachable (i64::MAX < usize::MAX),
        // but we keep the check for 32-bit portability and to make the
        // intent explicit.
        if capacity_i64 < 0 || (capacity_i64 as u64) > usize::MAX as u64 {
            // FR-EX-08: explicit InvalidArgument, not a silent cap.
            // SAFETY: `r_error` doc.
            unsafe {
                report_error(
                    r_error,
                    GDExtensionCallErrorType::InvalidArgument,
                    0,
                    GDExtensionVariantType::Int as i32,
                );
            }
            return;
        }
        let capacity: usize = capacity_i64 as usize;

        // (c) Backend dispatch — cast `p_instance` to `&mut StreamInstance`.
        //
        // Godot's ClassDB dispatch is single-threaded on the main thread
        // and `VokraStream::poll` does NOT emit signals or reenter
        // Godot's dispatcher, so no aliasing hazard exists.
        //
        // SAFETY: `p_instance` came from `create_stream_instance` (see
        // `crate::registry`), has NOT been freed (Godot's ClassDB
        // lifecycle guarantees `free_instance_func` is called AFTER
        // this trampoline returns), and no other Rust alias exists
        // because we consume the `*mut c_void` uniquely on this
        // main-thread dispatch path. Mirrors the pattern in
        // [`stream_interrupt`].
        let stream_inst = unsafe { &mut *(p_instance as *mut crate::registry::StreamInstance) };

        match stream_inst.inner.as_mut() {
            Some(stream) => {
                match stream.poll(capacity) {
                    Ok(vec) => {
                        // (d) Variant-pack success. Route through the
                        // resolved interface's PackedFloat32Array packer
                        // pipeline.
                        //
                        // SAFETY: `r_return` is either NULL or a
                        // writable 24-byte Variant slot per the Godot
                        // ClassDB contract. `pack_f32_slice_into_variant`
                        // documents its safety requirements around that
                        // contract.
                        if !r_return.is_null() {
                            let packed = unsafe {
                                crate::with_interface(|iface| {
                                    crate::variant::pack_f32_slice_into_variant(
                                        iface, r_return, &vec,
                                    );
                                })
                            };
                            if packed.is_none() {
                                // Extension not initialised at the
                                // pack path — impossible on this
                                // branch (init is a precondition for
                                // ClassDB dispatch to reach us) but
                                // FR-EX-08 requires an explicit error
                                // rather than a silent bad-Variant
                                // return.
                                // SAFETY: `r_error` doc.
                                unsafe { report_pending(r_error) };
                            }
                        }
                    }
                    Err(_err) => {
                        // Backend `poll` failed (e.g. SPSC ring in a
                        // teardown state). FR-EX-08: surface as
                        // InvalidMethod; `vokra_last_error()` on this
                        // thread still holds the detail for
                        // GDScript-side inspection via the C ABI.
                        //
                        // SAFETY: `r_error` doc.
                        unsafe { report_pending(r_error) };
                    }
                }
            }
            None => {
                // Stream was never opened (inner=None). GDScript coding
                // error — surface as InvalidMethod, mirroring the
                // posture used by `stream_interrupt`.
                //
                // SAFETY: `r_error` doc.
                unsafe { report_pending(r_error) };
            }
        }
    });
}

/// Method binding trampoline: `VokraStream::interrupt() -> void`.
///
/// Zero-argument method for M3-14 barge-in support (FR-ST-03). Fully
/// promoted past stub as of T14 — no Variant unpack is needed (0 args,
/// Nil return), so this is the first trampoline to reach real dispatch.
///
/// Dispatch contract:
/// - `stream_instance.inner == None` → `InvalidMethod`. GDScript coding
///   error: `interrupt()` on a stream that was never opened via
///   `session.vad_open_stream(sr)`. Distinguished from backend failure
///   by looking at `vokra_last_error()` on the calling thread (empty
///   iff pre-open).
/// - `VokraStream::interrupt()` returns `Ok(())` → success path.
///   `r_return` gets a Nil Variant, `r_error` stays Ok.
/// - `VokraStream::interrupt()` returns `Err(VokraError)` → backend
///   failure. Report `InvalidMethod` with the vokra_last_error() detail
///   available on this thread (FR-EX-08: never silently succeed).
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

        // Cast `p_instance` back to the `StreamInstance` box we
        // originally handed to Godot at `create_stream_instance`
        // (`Box::into_raw`). Take a mutable borrow for the duration of
        // this call: Godot's ClassDB dispatch is single-threaded on the
        // main thread (documented) and `VokraStream::interrupt` does
        // NOT emit signals or reenter Godot's dispatcher, so no
        // aliasing hazard exists. If a future patch adds signal
        // emission inside `interrupt`, this borrow must be revisited.
        //
        // SAFETY: `p_instance` came from `create_stream_instance` (see
        // `crate::registry`), has NOT been freed (Godot's ClassDB
        // lifecycle guarantees `free_instance_func` is called AFTER
        // this trampoline returns), and no other Rust alias exists
        // because we consume the `*mut c_void` uniquely on this
        // main-thread dispatch path.
        let stream_inst = unsafe { &mut *(p_instance as *mut crate::registry::StreamInstance) };

        match stream_inst.inner.as_mut() {
            Some(stream) => {
                match stream.interrupt() {
                    Ok(()) => {
                        // Success — write a proper Nil into `r_return`
                        // and leave `r_error` untouched (Godot passes it
                        // in with `error = Ok`).
                        //
                        // SAFETY: `r_return` is NULL or writable 24 bytes.
                        unsafe {
                            crate::with_interface(|iface| {
                                write_nil_return(Some(iface), r_return);
                            });
                            // If `with_interface` returned None (extension
                            // not initialised — impossible on this path
                            // because trampoline dispatch requires ClassDB
                            // registration which requires init), fall
                            // back to a full 24-byte zero for a proper
                            // Nil layout.
                            if crate::with_interface(|_| ()).is_none() {
                                write_nil_return(None, r_return);
                            }
                        }
                    }
                    Err(_err) => {
                        // Backend failure. Surface as InvalidMethod;
                        // `vokra_last_error()` on this thread still
                        // holds the detail for GDScript-side inspection
                        // via the C ABI.
                        //
                        // FR-EX-08: no silent success.
                        // SAFETY: `r_error` doc.
                        unsafe { report_pending(r_error) };
                    }
                }
            }
            None => {
                // Stream was never opened (inner=None). GDScript coding
                // error — surface as InvalidMethod.
                //
                // SAFETY: `r_error` doc.
                unsafe { report_pending(r_error) };
            }
        }
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

    // ------------------------------------------------------------------
    // T14 promotion coverage.
    //
    // All tests that install or teardown a mock InterfaceTable in
    // `EXTENSION_STATE` must serialize against each other AND against
    // the sibling registry / lib tests that also touch that mutex. We
    // reuse `registry::tests::TEST_LOCK` as the single serialization
    // point. Test isolation is via the RAII guard `MockStateGuard`,
    // which clears `EXTENSION_STATE` on Drop even if the test panics.
    // ------------------------------------------------------------------

    /// RAII guard that installs a mock `ExtensionState` at construction
    /// and clears `EXTENSION_STATE` on drop. Ensures tests do not leak
    /// state between one another.
    struct MockStateGuard;

    impl MockStateGuard {
        /// Install a sig-aware mock InterfaceTable in EXTENSION_STATE.
        /// Panics if the mutex is poisoned (a fatal test-level bug).
        fn install() -> Self {
            let mut guard = crate::EXTENSION_STATE.lock().unwrap();
            *guard = Some(crate::ExtensionState {
                library: ptr::null_mut(),
                interface: crate::ffi::interface::tests::make_sig_aware_interface(),
            });
            Self
        }

        /// Install a mock InterfaceTable whose `variant_get_type`
        /// returns the given tag. Used to drive
        /// `unpack_i64_or_report`'s type-check path.
        fn install_with_variant_type_mock(
            get_type_fn: unsafe extern "C" fn(
                crate::ffi::gdextension::GDExtensionConstVariantPtr,
            ) -> GDExtensionVariantType,
            to_int_fn: unsafe extern "C" fn(
                crate::ffi::gdextension::GDExtensionUninitializedTypePtr,
                crate::ffi::gdextension::GDExtensionVariantPtr,
            ),
        ) -> Self {
            let mut iface = crate::ffi::interface::tests::make_sig_aware_interface();
            iface.variant_get_type = get_type_fn;
            iface.variant_to_int_ctor = to_int_fn;
            let mut guard = crate::EXTENSION_STATE.lock().unwrap();
            *guard = Some(crate::ExtensionState {
                library: ptr::null_mut(),
                interface: iface,
            });
            Self
        }
    }

    impl Drop for MockStateGuard {
        fn drop(&mut self) {
            if let Ok(mut guard) = crate::EXTENSION_STATE.lock() {
                *guard = None;
            }
        }
    }

    // Canned mocks for the variant type check.
    unsafe extern "C" fn mock_returns_int_type(
        _v: crate::ffi::gdextension::GDExtensionConstVariantPtr,
    ) -> GDExtensionVariantType {
        GDExtensionVariantType::Int
    }
    unsafe extern "C" fn mock_returns_string_type(
        _v: crate::ffi::gdextension::GDExtensionConstVariantPtr,
    ) -> GDExtensionVariantType {
        GDExtensionVariantType::String
    }
    unsafe extern "C" fn mock_to_int_writes_16000(
        r_out: crate::ffi::gdextension::GDExtensionUninitializedTypePtr,
        _p: crate::ffi::gdextension::GDExtensionVariantPtr,
    ) {
        // SAFETY: caller (`variant_to_i64`) passes a writable 8-byte slot.
        unsafe { (r_out as *mut i64).write(16000) };
    }
    unsafe extern "C" fn mock_to_int_writes_zero(
        r_out: crate::ffi::gdextension::GDExtensionUninitializedTypePtr,
        _p: crate::ffi::gdextension::GDExtensionVariantPtr,
    ) {
        // SAFETY: caller passes a writable 8-byte slot.
        unsafe { (r_out as *mut i64).write(0) };
    }
    unsafe extern "C" fn mock_to_int_writes_neg_one(
        r_out: crate::ffi::gdextension::GDExtensionUninitializedTypePtr,
        _p: crate::ffi::gdextension::GDExtensionVariantPtr,
    ) {
        // SAFETY: caller (`variant_to_i64`) passes a writable 8-byte slot.
        // Exercises the negative-capacity guard on the `stream_poll` range
        // check (`capacity_i64 < 0`) — see `stream_poll_negative_capacity_...`.
        unsafe { (r_out as *mut i64).write(-1) };
    }

    /// Legitimate Int argument passes the type check but the return
    /// path is still deferred: expect `InvalidMethod` (partial
    /// promotion). Verifies the type-check path takes the good branch
    /// (i.e. does NOT emit `InvalidArgument`).
    #[test]
    fn session_vad_open_stream_int_arg_reaches_deferred_return_stub() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _state = MockStateGuard::install_with_variant_type_mock(
            mock_returns_int_type,
            mock_to_int_writes_16000,
        );

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        // Fake Variant pointer array with 1 slot; the value is unused
        // because our mock ignores `p_in`.
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 1] = [fake_variant];

        // SAFETY: `args` has 1 valid slot; arity=1 matches; instance
        // dummy passes non-null check. Mock interface is installed so
        // `with_interface` sees Some.
        unsafe {
            session_vad_open_stream(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                args.as_ptr(),
                1,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        // Deferred return path → InvalidMethod, NOT InvalidArgument.
        assert_eq!(
            err.error,
            GDExtensionCallErrorType::InvalidMethod,
            "Int-typed arg must pass validation and reach the deferred stub",
        );
    }

    /// Type mismatch: pass a String Variant where Int is expected. The
    /// trampoline must surface `InvalidArgument(index=0, expected=Int)`
    /// BEFORE hitting the deferred stub.
    #[test]
    fn session_vad_open_stream_wrong_arg_type_reports_invalid_argument() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _state = MockStateGuard::install_with_variant_type_mock(
            mock_returns_string_type,
            mock_to_int_writes_zero,
        );

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 1] = [fake_variant];

        // SAFETY: same as above.
        unsafe {
            session_vad_open_stream(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                args.as_ptr(),
                1,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidArgument);
        assert_eq!(err.argument, 0, "offending arg index");
        assert_eq!(
            err.expected,
            GDExtensionVariantType::Int as i32,
            "expected type code = Int (2)",
        );
    }

    /// Symmetric coverage for `stream_poll` int-arg validation.
    ///
    /// `stream_poll` is fully promoted (unlike `session_vad_open_stream`
    /// which still has a deferred return path), so a valid `Int` arg passes
    /// the type check AND reaches the real dispatch body — which
    /// dereferences `p_instance` as `&mut StreamInstance` (line ~885). A
    /// `dangling_mut::<c_void>()` pointer has alignment 1 and misaligns
    /// the 8-byte-aligned StreamInstance load, UB'ing the deref. This test
    /// allocates a real `Box<StreamInstance>` with `inner = None`; the
    /// trampoline reads `inner`, sees `None`, and reports `InvalidMethod`
    /// — same posture as
    /// `stream_interrupt_on_unopened_stream_reports_invalid_method`.
    #[test]
    fn stream_poll_int_arg_typechecks_then_unopened_reports_invalid_method() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _state = MockStateGuard::install_with_variant_type_mock(
            mock_returns_int_type,
            mock_to_int_writes_16000,
        );

        // Real Box so `p_instance` is properly aligned + points at a live
        // StreamInstance. `inner = None` triggers the "unopened stream"
        // branch → InvalidMethod.
        let boxed = Box::new(crate::registry::StreamInstance { inner: None });
        let raw = Box::into_raw(boxed);

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 1] = [fake_variant];

        // SAFETY: `raw` is a live `*mut StreamInstance` with proper align.
        // `args` has 1 slot with a dangling Variant pointer — mock
        // `variant_get_type` ignores it.
        unsafe {
            stream_poll(
                ptr::null_mut(),
                raw as GDExtensionClassInstancePtr,
                args.as_ptr(),
                1,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidMethod);

        // Reclaim the Box (Godot never called free_instance_func for us).
        // SAFETY: `raw` still points at our Box; not consumed by trampoline.
        unsafe {
            drop(Box::from_raw(raw));
        }
    }

    #[test]
    fn stream_poll_wrong_arg_type_reports_invalid_argument() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _state = MockStateGuard::install_with_variant_type_mock(
            mock_returns_string_type,
            mock_to_int_writes_zero,
        );

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 1] = [fake_variant];

        // SAFETY: dangling p_instance is safe because `unpack_i64_or_report`
        // detects the String type tag and returns None BEFORE the
        // trampoline reaches the p_instance cast at ~line 885.
        unsafe {
            stream_poll(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                args.as_ptr(),
                1,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidArgument);
        assert_eq!(err.argument, 0);
        assert_eq!(err.expected, GDExtensionVariantType::Int as i32);
    }

    /// Negative capacity (`-1`) must be rejected with an explicit
    /// `InvalidArgument(index=0, expected=Int)` under FR-EX-08 — a silent
    /// clamp to `0` or `usize::MAX` would swallow the GDScript coding
    /// error. Since the range guard on `capacity_i64 < 0` runs BEFORE the
    /// p_instance dereference, dangling p_instance is safe here.
    #[test]
    fn stream_poll_negative_capacity_reports_invalid_argument() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _state = MockStateGuard::install_with_variant_type_mock(
            mock_returns_int_type,
            mock_to_int_writes_neg_one,
        );

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 1] = [fake_variant];

        // SAFETY: dangling p_instance is safe because the range guard on
        // `capacity_i64 < 0` triggers with the mocked `-1`, writes
        // InvalidArgument, and returns BEFORE the p_instance cast at
        // ~line 885. `args` is 1-slot valid; the mocks ignore the actual
        // Variant pointer.
        unsafe {
            stream_poll(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                args.as_ptr(),
                1,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidArgument);
        assert_eq!(err.argument, 0);
        assert_eq!(err.expected, GDExtensionVariantType::Int as i32);
    }

    /// If the extension is not initialised (`EXTENSION_STATE` is None) —
    /// impossible on a live ClassDB dispatch, but a defensive posture per
    /// FR-EX-08 — `stream_poll` must report `InvalidMethod` rather than
    /// silently succeed. Mirrors `session_vad_open_stream_pre_init_...`
    /// for symmetric coverage on the promoted stream trampoline.
    #[test]
    fn stream_poll_pre_init_reports_invalid_method() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Deliberately clear EXTENSION_STATE.
        {
            let mut guard = crate::EXTENSION_STATE.lock().unwrap();
            *guard = None;
        }

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 1] = [fake_variant];

        // SAFETY: `args` is a valid 1-slot array; the pre-init branch
        // never derefs the dangling Variant pointer nor the p_instance —
        // `with_interface` yields None, the trampoline reports
        // InvalidMethod, and returns BEFORE the p_instance cast.
        unsafe {
            stream_poll(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                args.as_ptr(),
                1,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidMethod);
    }

    /// If the extension is not initialised (`EXTENSION_STATE` is None),
    /// the type-validated trampolines must fall back to `InvalidMethod`
    /// per FR-EX-08 — never silently succeed.
    #[test]
    fn session_vad_open_stream_pre_init_reports_invalid_method() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Deliberately clear EXTENSION_STATE.
        {
            let mut guard = crate::EXTENSION_STATE.lock().unwrap();
            *guard = None;
        }

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 1] = [fake_variant];

        // SAFETY: `args` is a valid 1-slot array; `p_args[0]` is a
        // dangling pointer but the pre-init branch never derefs it.
        unsafe {
            session_vad_open_stream(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                args.as_ptr(),
                1,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidMethod);
    }

    /// `stream_interrupt` on a StreamInstance with `inner = None`
    /// (never opened) must report `InvalidMethod` — the GDScript coding
    /// error is "call interrupt() before vad_open_stream()".
    #[test]
    fn stream_interrupt_on_unopened_stream_reports_invalid_method() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _state = MockStateGuard::install();

        // Allocate a real StreamInstance with inner = None. The
        // trampoline will cast p_instance back to `&mut StreamInstance`
        // and see the None.
        let boxed = Box::new(crate::registry::StreamInstance { inner: None });
        let raw = Box::into_raw(boxed);

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        // SAFETY: `raw` is a live `*mut StreamInstance` with inner=None;
        // the trampoline reads inner, sees None, reports InvalidMethod
        // without dereferencing any VokraStream state.
        unsafe {
            stream_interrupt(
                ptr::null_mut(),
                raw as GDExtensionClassInstancePtr,
                ptr::null(),
                0,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidMethod);

        // Reclaim the Box so tests don't leak. The trampoline did NOT
        // consume the pointer (Godot's contract: only free_instance_func
        // consumes it).
        //
        // SAFETY: `raw` still points to the Box we allocated above.
        unsafe {
            drop(Box::from_raw(raw));
        }
    }

    /// The panic firewall on the promoted `stream_interrupt` must still
    /// catch a panic — verify that a NULL EXTENSION_STATE lookup during
    /// the success path (which reaches `crate::with_interface`) does not
    /// alter the firewall guarantee.
    #[test]
    fn stream_interrupt_pre_init_with_none_inner_still_firewalled() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // No MockStateGuard: EXTENSION_STATE stays None.
        {
            let mut guard = crate::EXTENSION_STATE.lock().unwrap();
            *guard = None;
        }

        let boxed = Box::new(crate::registry::StreamInstance { inner: None });
        let raw = Box::into_raw(boxed);

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        // SAFETY: same as `stream_interrupt_on_unopened_stream_...`.
        unsafe {
            stream_interrupt(
                ptr::null_mut(),
                raw as GDExtensionClassInstancePtr,
                ptr::null(),
                0,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        // Same posture — inner=None → InvalidMethod regardless of
        // EXTENSION_STATE availability.
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidMethod);

        // SAFETY: same reclamation reasoning.
        unsafe {
            drop(Box::from_raw(raw));
        }
    }

    // ------------------------------------------------------------------
    // `stream_push_pcm` full-dispatch coverage.
    //
    // The trampoline is fully promoted: it Variant-unpacks arg 0 as
    // `PackedFloat32Array` via
    // `crate::variant::variant_to_packed_float32_slice` and calls
    // `VokraStream::push_pcm` on the borrowed `&[f32]`. Four branches
    // deserve unit coverage (in the trampoline's evaluation order):
    //   (1) `inner == None` (GDScript coding error — stream never opened).
    //   (2) EXTENSION_STATE = None (interface not resolved / already
    //       deinit'd — `with_interface` returns None).
    //   (3) Arg 0 is a String Variant (wrong type — `variant_get_type`
    //       returns non-PackedFloat32Array → `Err(actual_type)` → the
    //       trampoline reports `InvalidArgument(0, PackedFloat32Array)`).
    //   (4) Type check passes but the backend push errors (NULL C
    //       handle — `ffi_guard::required_mut` returns non-OK →
    //       `check` returns Err → the trampoline reports InvalidMethod
    //       with the detail available via `vokra_last_error()`).
    //
    // Case (4) reaches the closure with an empty `&[f32]`. To keep
    // `variant_to_packed_float32_slice`'s `PackedFloat32ArraySlot` in a
    // fully-initialised state (the sig-aware mock's default
    // `variant_to_packed_float32_array_ctor` is a no-op, which would
    // leave the 16-byte slot uninit and UB the subsequent
    // `assume_init` — even for a POD `[u8; 16]`), we swap in a custom
    // mock that zeros the slot.
    // ------------------------------------------------------------------

    /// Custom `variant_to_packed_float32_array_ctor` mock that zeros the
    /// 16-byte typed slot so subsequent `assume_init` observes a valid
    /// (all-zero) `PackedFloat32ArraySlot`. The all-zero pattern matches
    /// Godot's default-constructed `PackedFloat32Array` state (empty
    /// CowData pointer + zero-length inline layout).
    ///
    /// # Safety
    ///
    /// `r_out` must be a writable 16-byte, 8-byte-aligned slot per the
    /// `variant.rs` `PackedFloat32ArraySlot` layout.
    unsafe extern "C" fn mock_pfa_ctor_zeroes_slot(
        r_out: crate::ffi::gdextension::GDExtensionUninitializedTypePtr,
        _v: crate::ffi::gdextension::GDExtensionVariantPtr,
    ) {
        // SAFETY: caller (`variant_to_packed_float32_slice`) provides a
        // writable `PACKED_FLOAT32_ARRAY_SIZE`-byte slot per its
        // `PackedFloat32ArraySlot` layout.
        unsafe {
            (r_out as *mut u8).write_bytes(0, crate::ffi::gdextension::PACKED_FLOAT32_ARRAY_SIZE);
        }
    }

    /// Mock `variant_get_type` that returns `PackedFloat32Array` — used to
    /// drive `variant_to_packed_float32_slice` past its type check.
    unsafe extern "C" fn mock_returns_packed_float32_array_type(
        _v: crate::ffi::gdextension::GDExtensionConstVariantPtr,
    ) -> GDExtensionVariantType {
        GDExtensionVariantType::PackedFloat32Array
    }

    /// Case (1): `stream_push_pcm` on a StreamInstance with `inner = None`
    /// (never opened via `session.vad_open_stream()`) must report
    /// `InvalidMethod` — matches the sibling `stream_interrupt` posture.
    #[test]
    fn stream_push_pcm_on_unopened_stream_reports_invalid_method() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _state = MockStateGuard::install();

        // Real Box: `p_instance` is properly aligned + points at a live
        // StreamInstance whose `inner = None`.
        let boxed = Box::new(crate::registry::StreamInstance { inner: None });
        let raw = Box::into_raw(boxed);

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 1] = [fake_variant];

        // SAFETY: `raw` is a live `*mut StreamInstance`, `args` has 1
        // valid slot; the trampoline checks `inner` and short-circuits to
        // `report_pending` before dereffing `args[0]`.
        unsafe {
            stream_push_pcm(
                ptr::null_mut(),
                raw as GDExtensionClassInstancePtr,
                args.as_ptr(),
                1,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidMethod);

        // SAFETY: reclaim the Box (Godot never called free_instance_func).
        unsafe {
            drop(Box::from_raw(raw));
        }
    }

    /// Case (2): `stream_push_pcm` with EXTENSION_STATE = None (extension
    /// not yet initialised or already deinit'd) must report
    /// `InvalidMethod` per FR-EX-08 — never a silent success. Uses
    /// `inner = Some(VokraStream::null_for_tests())` to prove the pre-init
    /// branch fires AFTER the `inner` check passes (i.e. we get all the
    /// way to `with_interface` in the trampoline body).
    #[test]
    fn stream_push_pcm_pre_init_reports_invalid_method() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Deliberately clear EXTENSION_STATE.
        {
            let mut guard = crate::EXTENSION_STATE.lock().unwrap();
            *guard = None;
        }

        let boxed = Box::new(crate::registry::StreamInstance {
            inner: Some(crate::vad::VokraStream::null_for_tests()),
        });
        let raw = Box::into_raw(boxed);

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 1] = [fake_variant];

        // SAFETY: `raw` is a live `*mut StreamInstance` with inner=Some;
        // the trampoline reaches `with_interface`, which returns None
        // (state cleared above), then falls through to `report_pending`.
        // The mock C handle is NULL so no C ABI call is ever made — the
        // pre-init branch fires first.
        unsafe {
            stream_push_pcm(
                ptr::null_mut(),
                raw as GDExtensionClassInstancePtr,
                args.as_ptr(),
                1,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidMethod);

        // SAFETY: reclaim the Box. Drop of VokraStream with NULL handle
        // is a no-op (see `impl Drop for VokraStream`), no C ABI call.
        unsafe {
            drop(Box::from_raw(raw));
        }
    }

    /// Case (3): `stream_push_pcm` with a wrong-type arg (String Variant
    /// where PackedFloat32Array is expected) must report
    /// `InvalidArgument(0, PackedFloat32Array)` — FR-EX-08 explicit
    /// arg-index + expected-type surface, not a silent
    /// `InvalidMethod`.
    #[test]
    fn stream_push_pcm_wrong_arg_type_reports_invalid_argument() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _state = MockStateGuard::install_with_variant_type_mock(
            mock_returns_string_type,
            mock_to_int_writes_zero,
        );

        // `inner = Some(null_for_tests)` so the trampoline reaches the
        // Variant unpack path. The closure inside
        // `variant_to_packed_float32_slice` is never invoked because the
        // type check fails; the NULL C handle is thus never called.
        let boxed = Box::new(crate::registry::StreamInstance {
            inner: Some(crate::vad::VokraStream::null_for_tests()),
        });
        let raw = Box::into_raw(boxed);

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 1] = [fake_variant];

        // SAFETY: `raw` is a live `*mut StreamInstance`; `args` has 1
        // valid slot; the mock `variant_get_type` returns String → the
        // trampoline's `Err(_actual_type)` branch reports InvalidArgument.
        unsafe {
            stream_push_pcm(
                ptr::null_mut(),
                raw as GDExtensionClassInstancePtr,
                args.as_ptr(),
                1,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidArgument);
        assert_eq!(err.argument, 0, "offending arg index is arg 0");
        assert_eq!(
            err.expected,
            GDExtensionVariantType::PackedFloat32Array as i32,
            "expected type code = PackedFloat32Array (32)",
        );

        // SAFETY: reclaim the Box.
        unsafe {
            drop(Box::from_raw(raw));
        }
    }

    /// Case (4): `stream_push_pcm` with a valid `PackedFloat32Array`-typed
    /// Variant passes the type check and reaches
    /// `VokraStream::push_pcm(&[f32])`, which delegates to the C ABI
    /// `vokra_stream_push_pcm`. On a NULL C handle,
    /// `ffi_guard::required_mut(stream)` returns non-OK; the trampoline
    /// then reports `InvalidMethod` — FR-EX-08 honest backend-error
    /// surface, no silent success.
    ///
    /// This test double-serves as evidence that:
    ///   - The Variant-unpack pipeline (`variant_to_packed_float32_slice`)
    ///     bit-forwards the borrowed `&[f32]` slice to the closure.
    ///   - The closure's captured `&mut VokraStream` (from
    ///     `stream_inst.inner.as_mut()`) actually receives the call.
    ///   - The `Ok(Err(_backend_err))` match arm in the trampoline maps
    ///     to `report_pending` — never `write_nil_return` — on backend
    ///     failure.
    #[test]
    fn stream_push_pcm_typecheck_passes_backend_null_reports_invalid_method() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        // Build the sig-aware interface, then override:
        //   - `variant_get_type` → returns PackedFloat32Array (type check
        //     passes).
        //   - `variant_to_packed_float32_array_ctor` → zeros the 16-byte
        //     typed slot so `assume_init` observes valid (all-zero) bytes.
        //     The sig-aware default is `mock_variant_to_int` (no-op),
        //     which would leave the slot uninit — UB on `assume_init`.
        // The default `pfa_size_method` (no-op) leaves the caller's
        // pre-initialised `size_i64 = 0`, so
        // `variant_to_packed_float32_slice` takes the empty-slice branch
        // and never invokes `operator_index_const`.
        let mut iface = crate::ffi::interface::tests::make_sig_aware_interface();
        iface.variant_get_type = mock_returns_packed_float32_array_type;
        iface.variant_to_packed_float32_array_ctor = mock_pfa_ctor_zeroes_slot;
        {
            let mut guard = crate::EXTENSION_STATE.lock().unwrap();
            *guard = Some(crate::ExtensionState {
                library: ptr::null_mut(),
                interface: iface,
            });
        }
        // RAII cleanup: clear EXTENSION_STATE on scope exit even on panic.
        let _state = MockStateGuard;

        // `inner = Some(null_for_tests)` — the closure will call
        // `push_pcm(&[])` on a NULL C handle. `vokra_stream_push_pcm`'s
        // `ffi_guard::required_mut` rejects NULL → returns non-OK →
        // `check` returns Err → trampoline sees `Ok(Err(_))` → reports
        // InvalidMethod.
        let boxed = Box::new(crate::registry::StreamInstance {
            inner: Some(crate::vad::VokraStream::null_for_tests()),
        });
        let raw = Box::into_raw(boxed);

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 1] = [fake_variant];

        // SAFETY: `raw` is a live `*mut StreamInstance`; `args` has 1
        // valid slot. The Variant unpack pipeline uses only resolved
        // mocks (no real Godot memory access).
        unsafe {
            stream_push_pcm(
                ptr::null_mut(),
                raw as GDExtensionClassInstancePtr,
                args.as_ptr(),
                1,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        // Backend failure surfaces as InvalidMethod (not
        // InvalidArgument — the type check DID pass).
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidMethod);

        // SAFETY: reclaim the Box.
        unsafe {
            drop(Box::from_raw(raw));
        }
    }

    /// `unpack_i64_or_report` returns `Some(value)` on Int-typed input.
    #[test]
    fn unpack_i64_or_report_returns_value_on_int() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _state = MockStateGuard::install_with_variant_type_mock(
            mock_returns_int_type,
            mock_to_int_writes_16000,
        );

        let mut err = fresh_error();
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 1] = [fake_variant];

        // SAFETY: `with_interface` yields Some (guard installed); args
        // has 1 slot; mock returns 16000 for Int.
        let result = unsafe {
            crate::with_interface(|iface| unpack_i64_or_report(iface, args.as_ptr(), 0, &mut err))
        }
        .expect("interface must be available");
        assert_eq!(result, Some(16000));
        // On the success path, err is not touched.
        assert_eq!(err.error, GDExtensionCallErrorType::Ok);
        assert_eq!(err.argument, -777);
        assert_eq!(err.expected, -888);
    }

    /// `unpack_i64_or_report` writes `InvalidArgument(index, Int)` on
    /// type mismatch and returns None.
    #[test]
    fn unpack_i64_or_report_reports_and_returns_none_on_mismatch() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _state = MockStateGuard::install_with_variant_type_mock(
            mock_returns_string_type,
            mock_to_int_writes_zero,
        );

        let mut err = fresh_error();
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 2] = [fake_variant, fake_variant];

        // SAFETY: same as above; index=1 exercises the offending-index
        // wiring.
        let result = unsafe {
            crate::with_interface(|iface| unpack_i64_or_report(iface, args.as_ptr(), 1, &mut err))
        }
        .expect("interface must be available");

        assert_eq!(result, None);
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidArgument);
        assert_eq!(err.argument, 1);
        assert_eq!(err.expected, GDExtensionVariantType::Int as i32);
    }

    /// `write_nil_return` with a live interface must dispatch to
    /// `variant_new_nil`.
    #[test]
    fn write_nil_return_uses_interface_new_nil() {
        use core::sync::atomic::{AtomicU8, Ordering};
        static CALLED: AtomicU8 = AtomicU8::new(0);
        unsafe extern "C" fn recording_nil(
            _r: crate::ffi::gdextension::GDExtensionUninitializedVariantPtr,
        ) {
            CALLED.fetch_add(1, Ordering::SeqCst);
        }

        let mut iface = crate::ffi::interface::tests::make_sig_aware_interface();
        iface.variant_new_nil = recording_nil;
        CALLED.store(0, Ordering::SeqCst);

        let mut slot = [0xffu8; 24];
        // SAFETY: writable 24-byte slot; recording mock does not deref.
        unsafe { write_nil_return(Some(&iface), slot.as_mut_ptr() as *mut _) };
        assert_eq!(CALLED.load(Ordering::SeqCst), 1);
    }

    /// `write_nil_return` with no interface must zero all 24 bytes of
    /// the slot (fallback Nil layout).
    #[test]
    fn write_nil_return_none_zeros_full_24_bytes() {
        let mut slot = [0xffu8; 24];
        // SAFETY: writable 24-byte slot.
        unsafe { write_nil_return(None, slot.as_mut_ptr() as *mut _) };
        assert_eq!(&slot[..], &[0u8; 24][..]);
    }

    /// `write_nil_return` with NULL slot must be a no-op (defensive).
    #[test]
    fn write_nil_return_null_slot_is_noop() {
        // SAFETY: NULL branch tested.
        unsafe { write_nil_return(None, ptr::null_mut()) };
    }

    // ------------------------------------------------------------------
    // `session_transcribe` full-dispatch coverage
    // (M3-11 T14 followup — var-A + var-C driven).
    //
    // These tests verify each rejection branch of
    // [`dispatch_session_transcribe`] in isolation. The happy path
    // (backend `Ok(text)` → String pack into `r_return`) is NOT covered
    // here because it requires a live `VokraSession` and a real GGUF —
    // this excluded workspace deliberately does not carry model
    // fixtures (see `session.rs` module docs / M3-18 owner smoke).
    //
    // Every test acquires `crate::registry::tests::TEST_LOCK` for
    // serialization AND uses `MockStateGuard` so `EXTENSION_STATE`
    // does not leak. The sequence-based `variant_get_type` mock resets
    // its call counter under the same lock so parallel test ordering
    // does not alter its returned values.
    // ------------------------------------------------------------------

    use core::sync::atomic::{AtomicI64, AtomicU32, Ordering as SeqOrder};

    /// Call counter for the `mock_seq_pfa_then_int` variant-type mock.
    /// Reset to 0 by [`reset_sequence_mock`] at the start of each test.
    static SEQ_MOCK_CALL: AtomicU32 = AtomicU32::new(0);

    /// Value that `mock_seq_to_int_from_atomic` writes for the arg-1
    /// int unpack. Set by [`reset_sequence_mock`] to drive the
    /// range-guard tests without a per-value mock function.
    static SEQ_MOCK_TO_INT_VALUE: AtomicI64 = AtomicI64::new(0);

    /// Reset the sequence-mock state. Callers pass the desired
    /// `variant_to_int_ctor` output. MUST be called BEFORE installing
    /// the `MockStateGuard`, and BEFORE entering the trampoline — the
    /// first `variant_get_type` call will observe `SEQ_MOCK_CALL == 0`
    /// and return `PackedFloat32Array`.
    fn reset_sequence_mock(to_int_value: i64) {
        SEQ_MOCK_CALL.store(0, SeqOrder::SeqCst);
        SEQ_MOCK_TO_INT_VALUE.store(to_int_value, SeqOrder::SeqCst);
    }

    /// Mock `variant_get_type` that returns `PackedFloat32Array` on the
    /// first call and `Int` on every subsequent call. Drives
    /// `session_transcribe`'s arg0 (PFA) + arg1 (Int) type checks in a
    /// single test invocation. The counter is reset to 0 by
    /// [`reset_sequence_mock`].
    unsafe extern "C" fn mock_seq_pfa_then_int(
        _v: crate::ffi::gdextension::GDExtensionConstVariantPtr,
    ) -> GDExtensionVariantType {
        let n = SEQ_MOCK_CALL.fetch_add(1, SeqOrder::SeqCst);
        if n == 0 {
            GDExtensionVariantType::PackedFloat32Array
        } else {
            GDExtensionVariantType::Int
        }
    }

    /// Mock `variant_to_int_ctor` that writes the current value of
    /// `SEQ_MOCK_TO_INT_VALUE`. Configured via [`reset_sequence_mock`].
    unsafe extern "C" fn mock_seq_to_int_from_atomic(
        r_out: crate::ffi::gdextension::GDExtensionUninitializedTypePtr,
        _p: crate::ffi::gdextension::GDExtensionVariantPtr,
    ) {
        // SAFETY: caller (`variant_to_i64`) provides a writable 8-byte
        // slot.
        unsafe { (r_out as *mut i64).write(SEQ_MOCK_TO_INT_VALUE.load(SeqOrder::SeqCst)) };
    }

    /// arg0 wrong-type path: mock reports the arg0 Variant as String.
    /// Dispatch must abort at step (1) with
    /// `InvalidArgument(0, PackedFloat32Array)`. No instance deref
    /// occurs on this branch — `dangling_mut()` is safe as p_instance.
    #[test]
    fn session_transcribe_wrong_arg0_type_reports_invalid_argument() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _state = MockStateGuard::install_with_variant_type_mock(
            mock_returns_string_type,
            mock_to_int_writes_zero,
        );

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 2] = [fake_variant, fake_variant];

        // SAFETY: `args` is a valid 2-slot array; arity=2 matches;
        // instance dummy passes non-null check; dispatch exits at the
        // arg0 type check BEFORE any instance deref, so a dangling
        // p_instance is safe here.
        unsafe {
            session_transcribe(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                args.as_ptr(),
                2,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidArgument);
        assert_eq!(err.argument, 0, "offending arg index");
        assert_eq!(
            err.expected,
            GDExtensionVariantType::PackedFloat32Array as i32,
            "expected type code = PackedFloat32Array (32)",
        );
    }

    /// arg1 wrong-type path: mock returns PFA for every call, so arg0
    /// passes (matches PFA) but arg1 fails the Int check inside
    /// `unpack_i64_or_report`. Dispatch aborts at step (2) with
    /// `InvalidArgument(1, Int)`. No instance deref on this branch.
    #[test]
    fn session_transcribe_wrong_arg1_type_reports_invalid_argument() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Reuse the existing "always PackedFloat32Array" mock authored
        // for the sibling `stream_push_pcm` tests — arg0 sees PFA
        // (passes), arg1 sees PFA (fails Int check).
        let _state = MockStateGuard::install_with_variant_type_mock(
            mock_returns_packed_float32_array_type,
            mock_to_int_writes_zero,
        );

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 2] = [fake_variant, fake_variant];

        // SAFETY: same as `session_transcribe_wrong_arg0_...` — dispatch
        // exits at the arg1 type check before touching the instance.
        unsafe {
            session_transcribe(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                args.as_ptr(),
                2,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidArgument);
        assert_eq!(err.argument, 1, "offending arg index");
        assert_eq!(
            err.expected,
            GDExtensionVariantType::Int as i32,
            "expected type code = Int (2)",
        );
    }

    /// Zero sample_rate: types pass but range guard fires. Dispatch
    /// exits at step (3) with `InvalidArgument(1, Int)`. Sequence mock
    /// yields PFA→Int for the two `variant_get_type` calls. Range
    /// guard is placed BEFORE the SessionInstance deref (task-spec
    /// ordering (b) → (c)), so `dangling_mut()` is safe.
    #[test]
    fn session_transcribe_zero_sample_rate_reports_invalid_argument() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        reset_sequence_mock(0);
        let _state = MockStateGuard::install_with_variant_type_mock(
            mock_seq_pfa_then_int,
            mock_seq_to_int_from_atomic,
        );

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 2] = [fake_variant, fake_variant];

        // SAFETY: dispatch exits at the range guard before touching
        // the instance — dangling is safe.
        unsafe {
            session_transcribe(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                args.as_ptr(),
                2,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidArgument);
        assert_eq!(err.argument, 1, "offending arg index = sample_rate");
        assert_eq!(err.expected, GDExtensionVariantType::Int as i32);
    }

    /// Negative sample_rate: same shape as zero, verifies the `<= 0`
    /// half of the range guard (`-1` specifically).
    #[test]
    fn session_transcribe_negative_sample_rate_reports_invalid_argument() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        reset_sequence_mock(-1);
        let _state = MockStateGuard::install_with_variant_type_mock(
            mock_seq_pfa_then_int,
            mock_seq_to_int_from_atomic,
        );

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 2] = [fake_variant, fake_variant];

        // SAFETY: same as zero-rate variant.
        unsafe {
            session_transcribe(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                args.as_ptr(),
                2,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidArgument);
        assert_eq!(err.argument, 1);
        assert_eq!(err.expected, GDExtensionVariantType::Int as i32);
    }

    /// Oversized sample_rate: verifies the `> i32::MAX` half of the
    /// range guard. `i32::MAX as i64 + 1` fits comfortably in i64.
    #[test]
    fn session_transcribe_oversized_sample_rate_reports_invalid_argument() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        reset_sequence_mock(i32::MAX as i64 + 1);
        let _state = MockStateGuard::install_with_variant_type_mock(
            mock_seq_pfa_then_int,
            mock_seq_to_int_from_atomic,
        );

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 2] = [fake_variant, fake_variant];

        // SAFETY: same as zero-rate variant.
        unsafe {
            session_transcribe(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                args.as_ptr(),
                2,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidArgument);
        assert_eq!(err.argument, 1);
        assert_eq!(err.expected, GDExtensionVariantType::Int as i32);
    }

    /// Positive sample_rate + never-loaded session: types + range pass,
    /// but `SessionInstance::inner == None` because GDScript coded
    /// `transcribe` before `load(path)`. Dispatch exits at step (4)
    /// with `InvalidMethod`. Requires a properly aligned
    /// `*mut SessionInstance` from `Box::into_raw` — the trampoline
    /// derefs it as `&SessionInstance` (alignment 8), and a dangling
    /// `*mut c_void` would fail the debug-mode alignment check.
    #[test]
    fn session_transcribe_positive_sample_rate_with_none_inner_reports_invalid_method() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        reset_sequence_mock(16000);
        let _state = MockStateGuard::install_with_variant_type_mock(
            mock_seq_pfa_then_int,
            mock_seq_to_int_from_atomic,
        );

        // Allocate a real SessionInstance with inner = None. The
        // trampoline casts p_instance back to `&SessionInstance` and
        // sees inner=None, reporting InvalidMethod.
        let boxed = Box::new(crate::registry::SessionInstance { inner: None });
        let raw = Box::into_raw(boxed);

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 2] = [fake_variant, fake_variant];

        // SAFETY: `raw` is a live `*mut SessionInstance` (aligned to
        // 8 by the global allocator). `args` is a valid 2-slot array;
        // arity=2 matches. Dispatch runs the arg checks + range guard
        // (all pass), then derefs `raw` as `&SessionInstance`, sees
        // inner=None, reports InvalidMethod without touching the
        // PFA-unpack helper or the `asr::transcribe` backend.
        unsafe {
            session_transcribe(
                ptr::null_mut(),
                raw as GDExtensionClassInstancePtr,
                args.as_ptr(),
                2,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidMethod);

        // Reclaim the Box so tests don't leak. The trampoline did NOT
        // consume the pointer (Godot's contract: only `free_instance_func`
        // consumes it).
        //
        // SAFETY: `raw` still points to the Box we allocated above.
        unsafe {
            drop(Box::from_raw(raw));
        }
    }

    // ------------------------------------------------------------------
    // `session_synthesize` full-dispatch T14-followup promotion coverage
    // (M3-11).
    //
    // The variant.rs test file exercises `variant_to_string_owned` and
    // `pack_tts_output_into_dict_variant` directly; the tests below drive
    // the trampoline surface end-to-end (minus a live VokraSession /
    // real GGUF, which is out of scope for a cdylib unit test) to lock
    // down the FR-EX-08 posture on:
    //   - arg0 wrong-type → `InvalidArgument(0, String)` (BEFORE any
    //     instance deref).
    //   - Extension pre-init → `InvalidMethod` (before touching args or
    //     instance).
    //   - Session `inner = None` (never loaded) → `InvalidMethod`
    //     after arg0 unpacks cleanly.
    //   - Full arity guard: existing `session_synthesize_arity_is_one`
    //     covers the too-many path; this section adds too-few (0 args
    //     → `TooFewArguments`) for symmetric coverage.
    // ------------------------------------------------------------------

    /// Mock `variant_to_string_ctor` that zeros the 8-byte typed slot
    /// (matches variant.rs's `mock_str_ctor_zeroes`). Trampoline tests
    /// that reach `variant_to_string_owned`'s unpack path need this
    /// override.
    unsafe extern "C" fn tramp_mock_str_ctor_zeroes(
        r_out: crate::ffi::gdextension::GDExtensionUninitializedTypePtr,
        _v: crate::ffi::gdextension::GDExtensionVariantPtr,
    ) {
        // SAFETY: caller provides exactly STRING_SIZE writable bytes.
        unsafe {
            (r_out as *mut u8).write_bytes(0, crate::ffi::gdextension::STRING_SIZE);
        }
    }

    /// Mock `string_to_utf8_chars` that always reports zero bytes
    /// (empty String payload). Used by the `session_synthesize` inner=None
    /// test to keep the arg0 unpack side-effect-free: the trampoline
    /// gets `Ok(String::new())` back, then hits the inner=None gate
    /// which is what the test asserts on.
    unsafe extern "C" fn tramp_mock_string_to_utf8_empty(
        _p_self: crate::ffi::gdextension::GDExtensionConstStringPtr,
        _r_text: *mut core::ffi::c_char,
        _p_max_write_length: crate::ffi::gdextension::GDExtensionInt,
    ) -> crate::ffi::gdextension::GDExtensionInt {
        0
    }

    /// Install a mock interface tuned for the `session_synthesize`
    /// String-unpack path: `variant_get_type` returns String on every
    /// call, `variant_to_string_ctor` zeros the typed slot, and
    /// `string_to_utf8_chars` reports zero bytes. Trampoline reaches
    /// the "session inner=None → InvalidMethod" gate cleanly.
    ///
    /// Returns the `MockStateGuard` — tests hold it for the trampoline
    /// call, then let it drop to clean up `EXTENSION_STATE`.
    fn install_synthesize_string_mock() -> MockStateGuard {
        let mut iface = crate::ffi::interface::tests::make_sig_aware_interface();
        iface.variant_get_type = mock_returns_string_type;
        iface.variant_to_string_ctor = tramp_mock_str_ctor_zeroes;
        iface.string_to_utf8_chars = tramp_mock_string_to_utf8_empty;
        let mut guard = crate::EXTENSION_STATE.lock().unwrap();
        *guard = Some(crate::ExtensionState {
            library: ptr::null_mut(),
            interface: iface,
        });
        MockStateGuard
    }

    /// Too-few args (0 for a 1-arity method) surfaces
    /// `TooFewArguments`. Symmetric to
    /// `session_synthesize_arity_is_one` (which drives the too-many
    /// side).
    #[test]
    fn session_synthesize_rejects_zero_args() {
        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        // SAFETY: arity check runs BEFORE any args / instance deref.
        // Dummy dangling instance is safe here.
        unsafe {
            session_synthesize(
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

    /// Null instance with correct arity trips `InstanceIsNull` before
    /// the args / interface deref reach. Symmetric to
    /// `session_transcribe_rejects_null_instance_with_correct_arity`.
    #[test]
    fn session_synthesize_rejects_null_instance_with_correct_arity() {
        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        // SAFETY: arity=1 passes; NULL instance trips InstanceIsNull.
        unsafe {
            session_synthesize(
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null(),
                1,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InstanceIsNull);
    }

    /// Extension pre-init path: `EXTENSION_STATE == None`. Every
    /// promoted trampoline reports `InvalidMethod` in this branch
    /// (FR-EX-08 — no silent success from an uninitialised host).
    #[test]
    fn session_synthesize_pre_init_reports_invalid_method() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Deliberately clear EXTENSION_STATE.
        {
            let mut guard = crate::EXTENSION_STATE.lock().unwrap();
            *guard = None;
        }

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 1] = [fake_variant];

        // SAFETY: `args` is a valid 1-slot array; `p_args[0]` is a
        // dangling pointer but the pre-init branch never derefs it —
        // `with_interface(..)` returns None before touching the args.
        unsafe {
            session_synthesize(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                args.as_ptr(),
                1,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidMethod);
    }

    /// arg0 wrong-type path: mock reports arg0 as Int → dispatch
    /// aborts at step (1) with `InvalidArgument(0, String)`. No
    /// instance deref on this branch — a `dangling_mut()` p_instance
    /// is safe.
    #[test]
    fn session_synthesize_wrong_arg0_type_reports_invalid_argument() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Reuse the existing "always Int" mock. arg0 sees Int → fails
        // the String type check.
        let _state = MockStateGuard::install_with_variant_type_mock(
            mock_returns_int_type,
            mock_to_int_writes_zero,
        );

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 1] = [fake_variant];

        // SAFETY: `args` is a valid 1-slot array; arity=1 matches;
        // instance dummy passes non-null check. Dispatch exits at
        // the arg0 type check BEFORE any instance deref.
        unsafe {
            session_synthesize(
                ptr::null_mut(),
                core::ptr::dangling_mut(),
                args.as_ptr(),
                1,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidArgument);
        assert_eq!(err.argument, 0, "offending arg index");
        assert_eq!(
            err.expected,
            GDExtensionVariantType::String as i32,
            "expected type code = String (4)",
        );
    }

    /// Session inner=None path: arg0 unpacks cleanly as an empty
    /// String (mock `variant_to_string_ctor` + `string_to_utf8_chars`
    /// return zero bytes), then dispatch derefs `p_instance` as
    /// `&SessionInstance`, finds `inner = None`, and reports
    /// `InvalidMethod`. Mirrors
    /// `session_transcribe_positive_sample_rate_with_none_inner_...`.
    ///
    /// This is the ONLY code path in the trampoline that reaches the
    /// instance deref without a live GGUF-backed session; every other
    /// success-side path requires a live `crate::tts::synthesize` C
    /// ABI call which is out of scope for a cdylib unit test.
    #[test]
    fn session_synthesize_string_with_none_inner_reports_invalid_method() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _state = install_synthesize_string_mock();

        // Allocate a real SessionInstance with inner = None. The
        // trampoline casts p_instance back to `&SessionInstance` and
        // sees inner=None, reporting InvalidMethod.
        let boxed = Box::new(crate::registry::SessionInstance { inner: None });
        let raw = Box::into_raw(boxed);

        let mut err = fresh_error();
        let mut ret = fresh_variant_slot();
        let fake_variant: *const c_void = ptr::dangling();
        let args: [GDExtensionConstVariantPtr; 1] = [fake_variant];

        // SAFETY: `raw` is a live `*mut SessionInstance` (aligned to
        // 8 by the global allocator). `args` is a valid 1-slot array;
        // arity=1 matches. Dispatch runs the arg unpack (empty String
        // via mocks), then derefs `raw` as `&SessionInstance`, sees
        // inner=None, reports InvalidMethod without touching the
        // `tts::synthesize` backend.
        unsafe {
            session_synthesize(
                ptr::null_mut(),
                raw as GDExtensionClassInstancePtr,
                args.as_ptr(),
                1,
                ret.as_mut_ptr() as *mut _,
                &mut err,
            )
        };
        assert_eq!(err.error, GDExtensionCallErrorType::InvalidMethod);

        // Reclaim the Box so tests don't leak. The trampoline did NOT
        // consume the pointer (Godot's contract: only `free_instance_func`
        // consumes it).
        //
        // SAFETY: `raw` still points to the Box we allocated above.
        unsafe {
            drop(Box::from_raw(raw));
        }
    }
}
