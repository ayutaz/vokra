//! Variant packing / unpacking helpers backed by the GDExtension type
//! constructors (T14 M3-11 promotion).
//!
//! # Scope
//!
//! Godot 4.3's `Variant` is a 24-byte union whose type tag is stored in the
//! leading word and whose payload is one of ~40 typed cases. Rather than
//! peek at the internal layout (which is documented as unstable across minor
//! versions) we route through Godot's own factory-produced constructors:
//! `get_variant_from_type_constructor(TYPE)` yields a per-type packer,
//! `get_variant_to_type_constructor(TYPE)` yields a per-type unpacker. Both
//! are resolved once at extension init and cached in
//! [`crate::ffi::interface::InterfaceTable`].
//!
//! # What this module covers today
//!
//! - [`variant_get_type`]: introspect the type tag of a Variant.
//! - [`variant_to_i64`]: unpack an Int Variant into an `i64`, with an
//!   explicit type check that surfaces mismatches as an
//!   `Err(GDExtensionVariantType)`.
//! - [`variant_from_i64`]: pack an `i64` into a Variant slot.
//! - [`write_nil_variant`]: write a Nil Variant into a return slot on the
//!   success path of void-return methods.
//! - [`variant_to_packed_float32_slice`]: unpack a PackedFloat32Array
//!   Variant into a scoped `&[f32]` callback (var-C, M3-11 session_transcribe
//!   dependency; RAII-guarded destructor).
//! - [`variant_from_string_utf8`]: pack a Rust `&str` into a Godot String
//!   Variant (var-A, M3-11 session_transcribe dependency; RAII-guarded
//!   destructor).
//!
//! # What this module deliberately DOES NOT cover
//!
//! - String unpack (Variant String → Rust `&str`). Requires
//!   `string_to_utf8_chars`, plus buffer sizing / decode round-trip.
//!   Deferred until a trampoline surface actually needs it (`tts::synthesize`
//!   is the next candidate).
//! - PackedFloat32Array pack (Rust `&[f32]` → Variant PackedFloat32Array).
//!   Handled by [`pack_f32_slice_into_variant`] (T14-followup, out of scope
//!   for the session_transcribe promotion but scaffolded in
//!   [`crate::ffi::interface`]).
//! - Object pack. Requires `object_get_instance_binding` and the
//!   `InstanceBinding` posture that
//!   [`crate::registry`] deliberately defers (see registry.rs §Instance
//!   lifetime).
//!
//! # var-A: String pack ([`variant_from_string_utf8`])
//!
//! Packs a Rust `&str` into a Godot `String` Variant. The typed String
//! opaque is stack-allocated ([`crate::ffi::gdextension::STRING_SIZE`] = 8
//! bytes on LP64, doubled to 16 for future-safety), built via
//! `string_new_with_utf8_chars_and_len`, copied into the Variant via
//! `variant_from_string_ctor`, then destroyed via the resolved String
//! destructor. RAII-guarded so a Rust panic between construction and pack
//! does NOT leak the CowData refcount (matches FR-EX-08 no-silent-leak
//! discipline).
//!
//! # var-C: PackedFloat32Array unpack ([`variant_to_packed_float32_slice`])
//!
//! Unpacks a Godot `PackedFloat32Array` Variant into a Rust `&[f32]`
//! callback. The typed PackedFloat32Array opaque is stack-allocated
//! ([`crate::ffi::gdextension::PACKED_FLOAT32_ARRAY_SIZE`] = 16 bytes on
//! LP64), unpacked from the Variant via `variant_to_packed_float32_array_ctor`,
//! its element count is retrieved through the resolved
//! [`crate::ffi::interface::InterfaceTable::pfa_size_method`]
//! (PackedFloat32Array.size hash `3173160232`, verified against Godot
//! 4.3-stable `extension_api.json`), a `*const f32` pointer to element 0
//! is obtained through `packed_float32_array_operator_index_const`, and
//! the resulting slice is passed to the callback. On callback return
//! (or panic) the temp PackedFloat32Array is destroyed via the resolved
//! per-type destructor (RAII guard) to release the CowData refcount.
//!
//! # Provenance
//!
//! The Godot API used by both helpers is pinned by the 4.3-stable header
//! (`gdextension_interface.h`). Every fn pointer is resolved once at extension
//! init in [`crate::ffi::interface::InterfaceTable::from_proc_address`] and
//! stored on `InterfaceTable`; the helpers here only invoke, never resolve.
//!
//! # Not exception-safe against `abort`
//!
//! The RAII guards below assume `panic = "unwind"` (workspace default); a
//! `panic = "abort"` build would skip Drop and leak the temp CowData
//! refcount. Godot itself continues to run after the abort, so the leak
//! is process-lifetime — bearable in an emergency, undesirable in normal
//! operation.

use core::mem::MaybeUninit;
use core::slice;

use crate::ffi::gdextension::{
    DICTIONARY_SIZE, GDExtensionConstStringPtr, GDExtensionConstTypePtr,
    GDExtensionConstVariantPtr, GDExtensionInt, GDExtensionTypePtr,
    GDExtensionUninitializedTypePtr, GDExtensionUninitializedVariantPtr, GDExtensionVariantPtr,
    GDExtensionVariantType, PACKED_FLOAT32_ARRAY_SIZE, STRING_SIZE,
};
use crate::ffi::interface::InterfaceTable;

/// Read a Variant's type tag via `variant_get_type`. Never allocates, never
/// mutates the Variant.
///
/// # Safety
///
/// `v` must point to a fully-constructed Variant for the duration of the
/// call. Godot's `variant_get_type` is documented as safe on any live
/// Variant.
#[inline]
pub unsafe fn variant_get_type(
    interface: &InterfaceTable,
    v: GDExtensionConstVariantPtr,
) -> GDExtensionVariantType {
    // SAFETY: caller doc.
    unsafe { (interface.variant_get_type)(v) }
}

/// Unpack a Variant of type `Int` into an `i64`.
///
/// Returns `Err(actual_type)` iff the Variant's type tag is not
/// [`GDExtensionVariantType::Int`], letting the caller surface a Godot
/// `InvalidArgument` CallError with the offending index and expected type.
///
/// # Safety
///
/// `v` must point to a fully-constructed Variant for the duration of the
/// call.
pub unsafe fn variant_to_i64(
    interface: &InterfaceTable,
    v: GDExtensionConstVariantPtr,
) -> Result<i64, GDExtensionVariantType> {
    // SAFETY: caller doc — `v` is a live Variant.
    let ty = unsafe { variant_get_type(interface, v) };
    if ty != GDExtensionVariantType::Int {
        return Err(ty);
    }
    let mut out = MaybeUninit::<i64>::uninit();
    // SAFETY: `variant_to_int_ctor` is a live fn pointer (resolved at init).
    // `out.as_mut_ptr()` is a writable 8-byte slot; the constructor is
    // documented to write exactly 8 bytes (i64) into it. `v` is a live
    // Int Variant per the type-tag check above; the constructor takes a
    // non-const `GDExtensionVariantPtr` in the header even though it only
    // reads — cast is standard.
    unsafe {
        (interface.variant_to_int_ctor)(out.as_mut_ptr() as _, v as GDExtensionVariantPtr);
        Ok(out.assume_init())
    }
}

/// Pack an `i64` into `r_dest` as a Variant of type `Int`.
///
/// # Safety
///
/// `r_dest` must be a writable 24-byte Variant slot (Godot 4.3-stable pins
/// `sizeof(Variant) == 24` on LP64). The constructor writes the full 24
/// bytes (type tag + payload + tail padding) so the caller may not skip
/// zeroing beforehand.
#[inline]
pub unsafe fn variant_from_i64(
    interface: &InterfaceTable,
    r_dest: GDExtensionUninitializedVariantPtr,
    value: i64,
) {
    let mut val = value;
    // SAFETY: `variant_from_int_ctor` is a live fn pointer. `r_dest` is a
    // writable 24-byte slot per caller doc. `&mut val` is a writable
    // 8-byte slot from which the constructor reads the payload.
    unsafe {
        (interface.variant_from_int_ctor)(r_dest, &mut val as *mut i64 as _);
    }
}

/// Write a Nil Variant into `r_dest`. Called on the success path of
/// void-return methods (e.g. `VokraStream::interrupt` → GDScript `void`).
///
/// # Safety
///
/// `r_dest` must be a writable 24-byte Variant slot.
#[inline]
pub unsafe fn write_nil_variant(
    interface: &InterfaceTable,
    r_dest: GDExtensionUninitializedVariantPtr,
) {
    // SAFETY: `variant_new_nil` is a live fn pointer; `r_dest` is a
    // writable slot per caller doc.
    unsafe { (interface.variant_new_nil)(r_dest) };
}

// ---------------------------------------------------------------------------
// var-C: PackedFloat32Array unpack (M3-11 T14-followup, `stream_push_pcm`).
// ---------------------------------------------------------------------------

/// 8-byte-aligned stack storage for a Godot `PackedFloat32Array` typed
/// handle. Godot's builtin-class layout on LP64 pins `sizeof(PackedFloat32Array)
/// = 16` in `extension_api.json` builtin_class_sizes (float_64 config,
/// verified 4.1..4.3-stable — see [`PACKED_FLOAT32_ARRAY_SIZE`]). The struct
/// enforces 8-byte alignment because a PackedFloat32Array's leading field is
/// a `CowData<float>` pointer.
///
/// This is deliberately NOT `Copy` / `Clone`: the underlying handle holds a
/// refcount on Godot's shared CoW buffer, so a bitwise copy would silently
/// duplicate the refcount without a matching destroy. The
/// [`variant_to_packed_float32_slice`] flow constructs one, borrows a slice
/// out of it, and destroys via
/// [`crate::ffi::interface::InterfaceTable::pfa_destructor`] — all within a
/// single Rust scope.
#[repr(C, align(8))]
struct PackedFloat32ArraySlot {
    bytes: [u8; PACKED_FLOAT32_ARRAY_SIZE],
}

/// RAII guard that runs the typed-handle destructor on drop. Ensures the
/// CowData refcount is released even if the borrow closure panics. Matches
/// the module-doc "Not exception-safe against `abort`" clause: with
/// `panic = "unwind"` (workspace default) this drops on unwind; with
/// `panic = "abort"` the whole process ends and Godot's refcount leak is
/// unavoidable but bounded to process lifetime.
struct PackedFloat32ArrayGuard<'a> {
    slot: *mut PackedFloat32ArraySlot,
    destructor: crate::ffi::gdextension::PackedFloat32ArrayDestructor,
    _marker: core::marker::PhantomData<&'a mut PackedFloat32ArraySlot>,
}

impl<'a> Drop for PackedFloat32ArrayGuard<'a> {
    fn drop(&mut self) {
        // SAFETY: `self.slot` is a live pointer to a `PackedFloat32ArraySlot`
        // (constructed on the caller's stack, address taken via
        // `&mut slot as *mut _`). `self.destructor` is the resolved per-type
        // destructor (Option-unwrapped at
        // `InterfaceTable::from_proc_address`). The destructor consumes the
        // handle bytes; after Drop returns, the slot must not be re-used.
        unsafe { (self.destructor)(self.slot as GDExtensionTypePtr) };
    }
}

/// Unpack a Godot `PackedFloat32Array` Variant into a Rust `&[f32]` slice
/// bound to the closure `f`'s scope.
///
/// The slice is a direct borrow into Godot's shared CoW buffer (no memcpy);
/// audio-hot-path callers pay one type check, one Variant→typed-handle
/// construction, one `.size()` builtin-method call, one element-pointer
/// resolve, and one per-type destructor call. NO Rust allocation.
///
/// # Type check + FR-EX-08
///
/// Returns `Err(actual_type)` when the Variant's type tag is not
/// [`GDExtensionVariantType::PackedFloat32Array`]. The trampoline caller
/// converts this into a Godot `InvalidArgument` CallError with
/// `expected = PackedFloat32Array` — never silently succeeds.
///
/// # Bit-exact size query
///
/// The element count is retrieved via
/// [`crate::ffi::interface::InterfaceTable::pfa_size_method`] (resolved as
/// `variant_get_ptr_builtin_method(PACKED_FLOAT32_ARRAY, "size", 3173160232)`).
/// The `.size()` builtin is documented as const + zero-arg + returns Int
/// (`extension_api.json` — verified against Godot 4.3-stable). A negative
/// return (impossible from Godot but defensive) is clamped to `0`.
///
/// # Slice safety
///
/// - When `size == 0`, returns `&[]` (empty slice) without touching the
///   operator_index resolver.
/// - When `size > 0`, calls
///   [`crate::ffi::interface::InterfaceTable::packed_float32_array_operator_index_const`]
///   with `p_index = 0` to obtain a `*const f32` to element 0. If Godot
///   ever returns NULL for `size > 0` (documented not to, but defensive),
///   we still return `&[]` to the closure — the CowData refcount is
///   released normally on Drop.
///
/// # Safety
///
/// - `v` must point to a fully-constructed Variant for the duration of the
///   call. Godot's ClassDB dispatch guarantees this contract on the
///   trampoline entry path.
/// - `interface` must be the extension's live resolved [`InterfaceTable`].
/// - The `&[f32]` passed to `f` is valid ONLY for the duration of `f`.
///   Callers MUST NOT store, leak, or transmute-to-'static the slice.
pub unsafe fn variant_to_packed_float32_slice<F, R>(
    interface: &InterfaceTable,
    v: GDExtensionConstVariantPtr,
    f: F,
) -> Result<R, GDExtensionVariantType>
where
    F: FnOnce(&[f32]) -> R,
{
    // 1. Type check.
    // SAFETY: `v` is a live Variant per caller doc.
    let ty = unsafe { variant_get_type(interface, v) };
    if ty != GDExtensionVariantType::PackedFloat32Array {
        return Err(ty);
    }

    // 2. Unpack Variant → typed PackedFloat32Array handle. The unpacker is
    //    `void (*)(GDExtensionUninitializedTypePtr r_out, GDExtensionVariantPtr
    //    p_in)`; it placement-constructs a `PackedFloat32Array` at `r_out`
    //    (16 bytes on LP64) with a CowData refcount++ (share of the source
    //    Variant's internal buffer).
    let mut slot: MaybeUninit<PackedFloat32ArraySlot> = MaybeUninit::uninit();
    // SAFETY: `slot.as_mut_ptr()` is a writable 16-byte, 8-byte-aligned
    // slot (`PackedFloat32ArraySlot` layout). The resolved unpacker writes
    // exactly `PACKED_FLOAT32_ARRAY_SIZE` bytes into it. `v` is a live
    // PackedFloat32Array Variant per the type check above; the const→mut
    // cast is standard (Godot's typedef is non-const but the ctor is
    // read-only on the input).
    unsafe {
        (interface.variant_to_packed_float32_array_ctor)(
            slot.as_mut_ptr() as GDExtensionUninitializedTypePtr,
            v as GDExtensionVariantPtr,
        );
    }
    // SAFETY: the unpacker fully initialised `slot` per the C ABI contract.
    let mut slot = unsafe { slot.assume_init() };

    // 3. RAII guard: destroy the typed handle on scope exit (including
    //    panic unwind). MUST outlive every read of `slot`.
    let _guard = PackedFloat32ArrayGuard {
        slot: &mut slot as *mut PackedFloat32ArraySlot,
        destructor: interface.pfa_destructor,
        _marker: core::marker::PhantomData,
    };

    // 4. Query element count via the cached `PackedFloat32Array::size()`
    //    builtin method (hash 3173160232). Signature:
    //    `void (*)(GDExtensionTypePtr p_base, const GDExtensionConstTypePtr *p_args,
    //              GDExtensionTypePtr r_return, int p_argument_count)`.
    //    - `p_base` = &slot (typed handle).
    //    - `p_args` = NULL (zero args).
    //    - `r_return` = &size_i64 (writable 8-byte slot).
    //    - `p_argument_count` = 0.
    let mut size_i64: GDExtensionInt = 0;
    // SAFETY: `slot` is a live typed handle (constructed above, alive until
    // `_guard` drops). `size_i64` is a writable 8-byte slot. `pfa_size_method`
    // is a live fn pointer (resolved at init).
    unsafe {
        (interface.pfa_size_method)(
            &mut slot as *mut PackedFloat32ArraySlot as GDExtensionTypePtr,
            core::ptr::null(),
            &mut size_i64 as *mut GDExtensionInt as GDExtensionTypePtr,
            0,
        );
    }
    let size = if size_i64 < 0 { 0 } else { size_i64 as usize };

    // 5. Element pointer + slice construction.
    let slice_ref: &[f32] = if size == 0 {
        &[]
    } else {
        // SAFETY: `slot` is a live typed handle.
        // `packed_float32_array_operator_index_const` is
        // `const float *(*)(GDExtensionConstTypePtr p_self, GDExtensionInt
        // p_index)`. Godot returns a valid pointer into the CowData for any
        // in-range index; index 0 is always in-range when `size > 0`.
        let ptr: *const f32 = unsafe {
            (interface.packed_float32_array_operator_index_const)(
                &slot as *const PackedFloat32ArraySlot as GDExtensionConstTypePtr,
                0 as GDExtensionInt,
            )
        };
        if ptr.is_null() {
            // Defensive: Godot documents this API as non-fallible for
            // in-range indices, but a NULL return would UB our slice
            // construction. Fall back to an empty slice — the closure sees
            // an empty PCM, and FR-EX-08 is preserved because the empty
            // path is functionally equivalent to a zero-length push.
            &[]
        } else {
            // SAFETY: `ptr` is a valid pointer to `size` contiguous f32
            // elements in the CoW-backed array. The slice lifetime is
            // bounded by `slot`'s live scope (Drop of `_guard` happens
            // AFTER this closure returns).
            unsafe { slice::from_raw_parts(ptr, size) }
        }
    };

    // 6. Hand the borrow to the closure. On return (or panic) `_guard`
    //    drops and the typed handle is destroyed.
    Ok(f(slice_ref))
}

// ---------------------------------------------------------------------------
// var-A: String pack (M3-11 session_transcribe result → return Variant).
// ---------------------------------------------------------------------------

/// 8-byte-aligned stack storage for a Godot `String` typed handle. Godot 4.3
/// pins `sizeof(String) == 8` on LP64 in `extension_api.json`
/// builtin_class_sizes (float_64 config, verified 4.3-stable — see
/// [`STRING_SIZE`]). Same non-Copy / non-Clone discipline as
/// [`PackedFloat32ArraySlot`]: the underlying handle owns a CowData
/// refcount which MUST be released exactly once via the resolved
/// [`crate::ffi::interface::InterfaceTable::string_destructor`].
#[repr(C, align(8))]
struct StringSlot {
    bytes: [u8; STRING_SIZE],
}

/// RAII guard that runs the typed-String destructor on drop. Mirrors
/// [`PackedFloat32ArrayGuard`]: ensures the CowData refcount is released
/// even on panic unwind between construction and Variant pack, matching
/// the module-doc "Not exception-safe against `abort`" clause.
struct StringGuard<'a> {
    slot: *mut StringSlot,
    destructor: crate::ffi::gdextension::StringDestructor,
    _marker: core::marker::PhantomData<&'a mut StringSlot>,
}

impl<'a> Drop for StringGuard<'a> {
    fn drop(&mut self) {
        // SAFETY: `self.slot` is a live pointer to a `StringSlot`
        // constructed on the caller's stack. `self.destructor` is the
        // resolved per-type String destructor (Option-unwrapped at
        // `InterfaceTable::from_proc_address`). Godot's destructor consumes
        // the handle bytes; the slot must not be reused after Drop.
        unsafe { (self.destructor)(self.slot as GDExtensionTypePtr) };
    }
}

/// Pack a Rust `&str` into a Godot `String` Variant, writing into `r_dest`.
///
/// The pipeline (verified against Godot 4.3-stable
/// `core/extension/gdextension_interface.h`):
///
/// 1. Construct a typed `String` handle from the UTF-8 buffer via
///    [`crate::ffi::interface::InterfaceTable::string_new_with_utf8_chars_and_len`].
///    Godot allocates a fresh CowData and copies the codepoints in; our
///    `StringSlot` now owns one refcount.
/// 2. Register the RAII guard so the typed handle is destroyed on any
///    exit path (normal return or panic unwind, matching the module-doc
///    `panic = "unwind"` assumption).
/// 3. Pack the typed handle INTO the Variant slot via
///    [`crate::ffi::interface::InterfaceTable::variant_from_string_ctor`].
///    Godot's Variant packer clones-refcounts the CowData (see
///    `Variant::Variant(const String &)` in `core/variant/variant.cpp` —
///    the assignment operator increments the shared refcount and takes
///    an independent reference). The output Variant now holds its own
///    refcount; ours is released by the guard's Drop.
///
/// # Empty strings
///
/// A zero-length `&str` still constructs a valid empty typed String and
/// packs it into the Variant. Godot's `string_new_with_utf8_chars_and_len`
/// is documented safe for `p_size = 0` (verified against the 4.3-stable
/// header — the ctor short-circuits before any read from `p_contents`,
/// so a NULL `p_contents` in that case would also be safe, but we always
/// pass the `&str`'s `as_ptr()` because that is defined behavior on empty
/// slices in Rust).
///
/// # UTF-8 correctness
///
/// Rust's `&str` invariant guarantees valid UTF-8. Godot decodes the
/// buffer as UTF-8 → char32_t. No transcoding needed on the Rust side.
///
/// # Safety
///
/// - `r_dest` must be a writable 24-byte Variant slot (Godot 4.3-stable
///   pins `sizeof(Variant) = 24` — see the layout guard in
///   [`crate::ffi::gdextension`]).
/// - `interface` must be the extension's live resolved [`InterfaceTable`].
/// - The caller MUST NOT touch `r_dest` before this fn returns; the
///   Variant packer overwrites its full 24 bytes.
pub unsafe fn variant_from_string_utf8(
    interface: &InterfaceTable,
    r_dest: GDExtensionUninitializedVariantPtr,
    utf8: &str,
) {
    // 1. Construct a typed String handle from the UTF-8 buffer.
    //    Godot's signature (4.3-stable header):
    //      void (*)(GDExtensionUninitializedStringPtr r_dest,
    //               const char *p_contents,
    //               GDExtensionInt p_size);
    //    `p_size` is BYTE count (not codepoints); `utf8.len()` matches.
    //    Cast `usize → i64` is safe on LP64 because any legal `&str` fits
    //    in isize::MAX bytes (Rust invariant), which fits in i64.
    let mut slot: MaybeUninit<StringSlot> = MaybeUninit::uninit();
    // SAFETY: `slot.as_mut_ptr()` is a writable STRING_SIZE-byte,
    // 8-byte-aligned buffer (StringSlot layout). `utf8.as_ptr()` is a
    // valid pointer to `utf8.len()` valid UTF-8 bytes (Rust's `&str`
    // invariant). The Godot ctor writes exactly STRING_SIZE bytes into
    // `slot` and copies the payload into a fresh CowData allocation.
    unsafe {
        (interface.string_new_with_utf8_chars_and_len)(
            slot.as_mut_ptr() as crate::ffi::gdextension::GDExtensionUninitializedStringPtr,
            utf8.as_ptr() as *const core::ffi::c_char,
            utf8.len() as GDExtensionInt,
        );
    }
    // SAFETY: the Godot ctor fully initialised `slot` per the C ABI
    // contract (writes all STRING_SIZE bytes).
    let mut slot = unsafe { slot.assume_init() };

    // 2. RAII guard: release the CowData refcount on scope exit.
    //    Registered BEFORE the pack call so a panic in `variant_from_string_ctor`
    //    (impossible in practice — it's a memcpy + refcount++ — but
    //    defensive) still cleans up.
    let _guard = StringGuard {
        slot: &mut slot as *mut StringSlot,
        destructor: interface.string_destructor,
        _marker: core::marker::PhantomData,
    };

    // 3. Pack typed handle → Variant. Signature (4.3-stable):
    //    void (*)(GDExtensionUninitializedVariantPtr r_out,
    //             GDExtensionTypePtr p_in);
    //    Godot's `_from_string_ctor` implementation invokes the String
    //    copy-assignment operator, which increments the CowData refcount
    //    and stores an independent reference in the output Variant.
    //
    // SAFETY: `r_dest` is a writable 24-byte Variant slot (caller doc).
    // `slot` is a live, fully-constructed typed String handle. The packer
    // writes exactly 24 bytes to `r_dest`.
    unsafe {
        (interface.variant_from_string_ctor)(
            r_dest,
            &mut slot as *mut StringSlot as GDExtensionTypePtr,
        );
    }

    // 4. `_guard` drops here: our local CowData refcount is released.
    //    The Variant retains its own independent refcount from step 3.
}

// ---------------------------------------------------------------------------
// var-A (unpack side): String Variant → owned Rust `String`
// (M3-11 session_synthesize argument path).
// ---------------------------------------------------------------------------

/// Unpack a Godot `String` Variant into an owned Rust [`String`].
///
/// The pipeline (verified against Godot 4.3-stable
/// `core/extension/gdextension_interface.h`):
///
/// 1. Type-check the Variant tag via [`variant_get_type`]. On mismatch,
///    return `Err(actual_type)` so the trampoline caller can surface an
///    [`crate::ffi::gdextension::GDExtensionCallErrorType::InvalidArgument`]
///    with `expected = String` (FR-EX-08).
/// 2. Unpack the Variant into a typed `String` handle via
///    [`InterfaceTable::variant_to_string_ctor`]. The handle owns one
///    CowData refcount into Godot's shared string buffer; the
///    [`StringGuard`] RAII releases it on scope exit (including panic
///    unwind under `panic = "unwind"`).
/// 3. Two-phase probe through
///    [`InterfaceTable::string_to_utf8_chars`]:
///    - **Phase A**: pass `r_text = NULL` and `p_max_write_length = 0`.
///      Godot's header pins the return value as the encoded UTF-8 BYTE
///      length (not codepoints, no trailing NUL). A `< 0` return is
///      defensive-clamped to `0` — the header documents non-negative
///      returns but we do not trust corrupt hosts.
///    - **Phase B**: allocate a `Vec<u8>` of the reported length and
///      call again with the full cap. `written.min(byte_len)` clamps a
///      pathological over-report to the allocated buffer.
/// 4. `String::from_utf8_lossy` produces a valid Rust `String`. Godot's
///    internal `char32_t` representation supports unpaired surrogates
///    (U+D800-U+DFFF) which encode to invalid UTF-8 bytes; the lossy
///    conversion replaces those with U+FFFD rather than failing the
///    unpack outright. In practice TTS input text does not exercise this
///    path (surrogate halves are not typical text), but the defensive
///    path preserves the module-doc "no silent leak" discipline: the
///    caller gets a usable String, not an error swallow.
/// 5. `_guard` drops → typed handle destructor releases the CowData
///    refcount; only the Rust-owned `String` remains.
///
/// # Empty strings
///
/// Phase A returning `0` short-circuits Phase B — an empty Rust `String`
/// is returned without allocating.
///
/// # Safety
///
/// - `v` must point to a fully-constructed Variant for the duration of the
///   call. Godot's ClassDB dispatch guarantees this on the trampoline
///   entry path.
/// - `interface` must be the extension's live resolved [`InterfaceTable`].
/// - The returned `String` is Rust-owned; the caller may store it, move
///   it across scopes, or copy it — it is entirely independent of Godot's
///   CowData buffer once this fn returns.
pub unsafe fn variant_to_string_owned(
    interface: &InterfaceTable,
    v: GDExtensionConstVariantPtr,
) -> Result<String, GDExtensionVariantType> {
    // 1. Type check.
    //
    // SAFETY: `v` is a live Variant per caller doc.
    let ty = unsafe { variant_get_type(interface, v) };
    if ty != GDExtensionVariantType::String {
        return Err(ty);
    }

    // 2. Unpack Variant → typed String handle. Same RAII pattern used by
    //    `variant_to_packed_float32_slice`.
    let mut slot: MaybeUninit<StringSlot> = MaybeUninit::uninit();
    // SAFETY: `slot.as_mut_ptr()` is a writable STRING_SIZE-byte,
    // 8-byte-aligned buffer (StringSlot layout). The resolved unpacker
    // writes exactly `STRING_SIZE` bytes into it and takes a shared
    // CowData refcount with the source Variant. `v` is a live String
    // Variant per the type-tag check above; the const→mut cast is
    // standard (Godot's typedef is non-const but the ctor is read-only
    // on the input).
    unsafe {
        (interface.variant_to_string_ctor)(
            slot.as_mut_ptr() as GDExtensionUninitializedTypePtr,
            v as GDExtensionVariantPtr,
        );
    }
    // SAFETY: the Godot ctor fully initialised `slot` per the C ABI
    // contract (writes all STRING_SIZE bytes).
    let mut slot = unsafe { slot.assume_init() };

    // 3. RAII guard: destroy the typed handle on scope exit (including
    //    panic unwind under `panic = "unwind"`).
    let _guard = StringGuard {
        slot: &mut slot as *mut StringSlot,
        destructor: interface.string_destructor,
        _marker: core::marker::PhantomData,
    };

    // 4a. Phase A probe: NULL r_text yields the encoded UTF-8 byte length.
    //     Godot's header (`string_to_utf8_chars`, `@since 4.1`) pins the
    //     return as the encoded byte count (no trailing NUL). A `< 0`
    //     return is defensive-clamped — the header documents non-negative
    //     returns but a corrupt host that reports negative would otherwise
    //     underflow the `as usize` cast.
    //
    // SAFETY: `slot` is a live typed String handle (constructed above,
    // alive until `_guard` drops). NULL `r_text` + zero cap is the
    // documented probe form.
    let byte_len_i64 = unsafe {
        (interface.string_to_utf8_chars)(
            &slot as *const StringSlot as GDExtensionConstStringPtr,
            core::ptr::null_mut(),
            0,
        )
    };
    let byte_len: usize = if byte_len_i64 < 0 {
        0
    } else {
        byte_len_i64 as usize
    };

    // 4b. Fast-path: empty string. Skip Phase B allocation entirely.
    if byte_len == 0 {
        // `_guard` drops here → CowData refcount released.
        return Ok(String::new());
    }

    // 4c. Phase B: allocate a `Vec<u8>` of the reported length and fill.
    //     `Vec::with_capacity(byte_len)` + `set_len` would avoid the
    //     zero-fill of `vec![0u8; byte_len]`, but the caller (TTS input
    //     text) is not hot-path; correctness + readability wins over the
    //     ~byte_len wasted stores.
    let mut buf: Vec<u8> = vec![0u8; byte_len];
    // SAFETY: `slot` still lives; `buf.as_mut_ptr()` is a writable
    // `byte_len`-byte buffer. `p_max_write_length = byte_len` caps the
    // write to the allocation — Godot will not exceed this cap even if
    // its internal representation ever changed to require more bytes.
    let written_i64 = unsafe {
        (interface.string_to_utf8_chars)(
            &slot as *const StringSlot as GDExtensionConstStringPtr,
            buf.as_mut_ptr() as *mut core::ffi::c_char,
            byte_len as GDExtensionInt,
        )
    };
    let written: usize = if written_i64 < 0 {
        0
    } else {
        // Clamp to `byte_len` — a pathological over-report would otherwise
        // let a downstream `truncate` walk off the end of the buffer.
        (written_i64 as usize).min(byte_len)
    };
    buf.truncate(written);

    // 5. Godot's char32_t → UTF-8 re-encoder produces valid UTF-8 for
    //    every legal Unicode scalar value; unpaired surrogates
    //    (U+D800-U+DFFF, valid char32_t but not Unicode scalar values)
    //    round-trip as invalid UTF-8 byte sequences. `from_utf8_lossy`
    //    replaces those with U+FFFD (`\u{FFFD}`) rather than failing the
    //    entire unpack — matches the module-doc "no silent leak"
    //    discipline (caller gets a usable String, not a swallowed
    //    error).
    let s = String::from_utf8_lossy(&buf).into_owned();

    // 6. `_guard` drops here → typed handle destructor releases CowData
    //    refcount. Only the Rust-owned `s` survives.
    Ok(s)
}

// ---------------------------------------------------------------------------
// var-D: Dictionary pack for `TtsOutput` (M3-11 session_synthesize return).
// ---------------------------------------------------------------------------

/// 8-byte-aligned stack storage for a Godot `Dictionary` typed handle.
/// Godot pins `sizeof(Dictionary) == 8` on LP64 per `extension_api.json`
/// builtin_class_sizes (verified against Godot 4.1..4.3 — see
/// [`DICTIONARY_SIZE`]). A single opaque handle to `DictionaryPrivate`.
///
/// Same non-Copy / non-Clone discipline as [`StringSlot`] and
/// [`PackedFloat32ArraySlot`]: the underlying handle owns a refcount that
/// MUST be released exactly once via the resolved
/// [`crate::ffi::interface::InterfaceTable::dict_destructor`].
#[repr(C, align(8))]
struct DictionarySlot {
    _bytes: [u8; DICTIONARY_SIZE],
}

/// RAII guard that runs the typed-Dictionary destructor on drop. Mirrors
/// [`StringGuard`] / [`PackedFloat32ArrayGuard`]: ensures the underlying
/// Dictionary refcount is released even on panic unwind between the
/// default-construct and the eventual Variant pack.
struct DictionaryGuard<'a> {
    slot: *mut DictionarySlot,
    destructor: crate::ffi::gdextension::DictionaryDestructor,
    _marker: core::marker::PhantomData<&'a mut DictionarySlot>,
}

impl<'a> Drop for DictionaryGuard<'a> {
    fn drop(&mut self) {
        // SAFETY: `self.slot` is a live pointer to a `DictionarySlot`
        // constructed on the caller's stack. `self.destructor` is the
        // resolved per-type Dictionary destructor (Option-unwrapped at
        // `InterfaceTable::from_proc_address`). Godot's destructor
        // consumes the handle bytes; the slot must not be reused after
        // Drop.
        unsafe { (self.destructor)(self.slot as GDExtensionTypePtr) };
    }
}

/// Pack a [`crate::tts::TtsOutput`]-shaped payload (PCM samples + sample
/// rate) into `r_dest` as a Godot `Dictionary` Variant of shape
/// `{"pcm": PackedFloat32Array, "sample_rate": int}`.
///
/// Called on the success path of `VokraSession::synthesize(text) ->
/// Dictionary` (M3-11 session_synthesize full dispatch).
///
/// # Pipeline (5 stages)
///
/// The build-a-Dictionary-from-typed-values pattern is not directly
/// exposed by GDExtension; the standard sequence (matching godot-cpp's
/// `Dictionary::operator[]` implementation) is:
///
/// 1. **Default-construct** an empty `Dictionary` in a [`DICTIONARY_SIZE`]
///    -byte stack buffer via
///    [`InterfaceTable::dict_default_ctor`][ct]
///    (`variant_get_ptr_constructor(DICTIONARY, 0)`). Guarded by
///    [`DictionaryGuard`] so a panic between here and stage 4 releases
///    the Dictionary refcount.
/// 2. **Build the value Variants** on stack Variant slots:
///    - `pcm_variant` via [`pack_f32_slice_into_variant`] (existing
///      var-B helper).
///    - `sr_variant` via [`variant_from_i64`] (existing helper).
///
///    Both Variants own their own CowData refcount (the PFA one holds
///    the sole reference to the freshly-allocated float buffer).
/// 3. **Insert each key** via
///    [`InterfaceTable::dictionary_operator_index`] with a per-key temp
///    String Variant:
///    - Build a `key_variant` on stack via [`variant_from_string_utf8`].
///    - Call `dictionary_operator_index(dict_handle, key_variant)` →
///      `*mut Variant` slot pointer. Godot creates the slot as Nil (the
///      key is fresh). Overwriting a Nil slot via `variant_new_copy`
///      does NOT leak (Nil holds no resource) — matches the per-typedef
///      rustdoc contract on `dictionary_operator_index`.
///    - Deep-copy the value Variant into the slot via
///      [`InterfaceTable::variant_new_copy`]. Godot bumps the CowData
///      refcount on the value; the dictionary now holds its own
///      independent reference.
///    - `variant_destroy(key_variant)` releases the local String
///      refcount; the slot's key inside the dictionary retains its
///      independent reference.
/// 4. **Pack Dictionary → Variant** via
///    [`InterfaceTable::variant_from_dictionary_ctor`]. Godot's assignment
///    operator increments the internal Dictionary refcount; the output
///    Variant now holds its own independent reference.
/// 5. **Destroy value Variants + Dictionary handle**:
///    - `variant_destroy(pcm_variant)` releases the local PackedFloat32Array
///      refcount (the dict slot retains an independent reference).
///    - `variant_destroy(sr_variant)` (idempotent on Int — no resource).
///    - `_dict_guard` drops → Dictionary destructor releases the local
///      handle's refcount (the Variant retains an independent reference).
///
/// [ct]: crate::ffi::interface::InterfaceTable::dict_default_ctor
///
/// # Panic safety
///
/// Every temp typed handle (Dictionary, String key, PackedFloat32Array
/// inside `pack_f32_slice_into_variant`) is either RAII-guarded (this
/// helper's [`DictionaryGuard`], `variant_from_string_utf8`'s
/// [`StringGuard`]) or destroyed inline before the next Rust statement
/// (see `pack_f32_slice_into_variant`'s stage 5 destructor call). The
/// temp value Variants (`pcm_variant`, `sr_variant`) are NOT RAII-guarded
/// — a panic between their construction and their `variant_destroy` call
/// would leak the CowData refcount to process lifetime. This matches the
/// existing `pack_f32_slice_into_variant` precedent and is acceptable
/// because the only code between construction and destroy is fn-pointer
/// calls into Godot (no Rust panic points).
///
/// # Empty PCM / zero sample_rate
///
/// A zero-length `pcm` still executes the full pipeline — the resulting
/// PackedFloat32Array Variant is a valid empty array. A `sample_rate = 0`
/// still packs (Godot's Int Variant accepts any i64); GDScript-side
/// callers can spot the zero and skip playback. Both edge cases preserve
/// the FR-EX-08 no-silent-fabrication rule (an honest empty result rather
/// than a hidden error).
///
/// # Safety
///
/// - `r_dest` must be a writable 24-byte Variant slot per Godot's
///   ClassDB return contract.
/// - `interface` must be the extension's live resolved [`InterfaceTable`].
/// - `pcm.len()` must fit into `GDExtensionInt` (`i64`). On LP64 targets
///   this is trivially true (`isize::MAX == i64::MAX`).
/// - The caller MUST NOT touch `r_dest` before this fn returns; the
///   Variant packer overwrites its full 24 bytes.
pub unsafe fn pack_tts_output_into_dict_variant(
    interface: &InterfaceTable,
    r_dest: GDExtensionUninitializedVariantPtr,
    pcm: &[f32],
    sample_rate: i32,
) {
    // Stage 1: default-construct the Dictionary on a stack buffer.
    #[repr(C, align(8))]
    struct DictBuf([u8; DICTIONARY_SIZE]);
    let mut dict_buf = DictBuf([0u8; DICTIONARY_SIZE]);
    let dict_ptr: *mut core::ffi::c_void = dict_buf.0.as_mut_ptr() as *mut core::ffi::c_void;

    // SAFETY: `dict_default_ctor` is a live fn pointer resolved from
    // `variant_get_ptr_constructor(DICTIONARY, 0)`. `dict_ptr` is a
    // writable DICTIONARY_SIZE-byte 8-aligned slot. `p_args = NULL` is
    // the standard 0-arg default constructor calling convention.
    unsafe {
        (interface.dict_default_ctor)(dict_ptr, core::ptr::null::<GDExtensionConstTypePtr>());
    }

    // Guard the freshly-constructed Dictionary handle. If any subsequent
    // stage panics, the Drop runs the resolved destructor to release the
    // internal refcount. The guard holds a `*mut DictionarySlot`, whose
    // byte layout (`repr(C, align(8))` over `[u8; DICTIONARY_SIZE]`) is
    // identical to `DictBuf`'s — we cast the address; no aliasing hazard
    // because we only touch `dict_buf` through `dict_ptr` until the
    // destructor runs, and the destructor consumes the bytes.
    let _dict_guard = DictionaryGuard {
        slot: dict_buf.0.as_mut_ptr() as *mut DictionarySlot,
        destructor: interface.dict_destructor,
        _marker: core::marker::PhantomData,
    };

    // Stage 2: build the two value Variants on stack Variant slots.
    // Variant is 24 bytes on LP64 (Godot 4.3-stable pins this — see
    // compile-time guards in `crate::ffi::gdextension`).
    #[repr(C, align(8))]
    struct VariantBuf([u8; 24]);

    let mut pcm_variant = VariantBuf([0u8; 24]);
    // SAFETY: `pack_f32_slice_into_variant` writes exactly 24 bytes into
    // its `r_dest`. `pcm.len()` is bounded by `isize::MAX == i64::MAX`
    // on LP64. On return, `pcm_variant` holds a live PackedFloat32Array
    // Variant with a CowData refcount on the fresh float buffer.
    unsafe {
        pack_f32_slice_into_variant(
            interface,
            pcm_variant.0.as_mut_ptr() as GDExtensionUninitializedVariantPtr,
            pcm,
        );
    }

    let mut sr_variant = VariantBuf([0u8; 24]);
    // SAFETY: `variant_from_i64` writes exactly 24 bytes into `r_dest`.
    // `sample_rate as i64` widens without loss (i32 → i64).
    unsafe {
        variant_from_i64(
            interface,
            sr_variant.0.as_mut_ptr() as GDExtensionUninitializedVariantPtr,
            sample_rate as i64,
        );
    }

    // Stage 3: insert both key/value pairs. `insert_dict_entry_from_str`
    // encapsulates the "build a String Variant key + `operator_index` +
    // `variant_new_copy` value + destroy String key" sequence.
    //
    // SAFETY: `dict_ptr` is a live Dictionary handle (stage 1). Each
    // value Variant is a live 24-byte Variant (stage 2). `interface` is
    // live per caller doc.
    unsafe {
        insert_dict_entry_from_str(
            interface,
            dict_ptr,
            "pcm",
            pcm_variant.0.as_ptr() as GDExtensionConstVariantPtr,
        );
        insert_dict_entry_from_str(
            interface,
            dict_ptr,
            "sample_rate",
            sr_variant.0.as_ptr() as GDExtensionConstVariantPtr,
        );
    }

    // Stage 4: pack the Dictionary INTO the return Variant. Godot's
    // Variant-from-Dictionary ctor bumps the internal refcount; the
    // output Variant owns its own reference.
    //
    // SAFETY: `variant_from_dictionary_ctor` is a live fn pointer.
    // `r_dest` is a writable 24-byte Variant slot per caller doc.
    // `dict_ptr` is the just-constructed Dictionary handle.
    unsafe {
        (interface.variant_from_dictionary_ctor)(r_dest, dict_ptr);
    }

    // Stage 5a: destroy the value Variants. Their CowData refcounts drop
    // to whatever the dictionary holds (independent references retained
    // inside the dict's own storage — see `variant_new_copy` in
    // `insert_dict_entry_from_str`).
    //
    // SAFETY: `variant_destroy` is a live fn pointer. Both Variants were
    // constructed above and are live 24-byte Variants.
    unsafe {
        (interface.variant_destroy)(pcm_variant.0.as_mut_ptr() as GDExtensionVariantPtr);
        (interface.variant_destroy)(sr_variant.0.as_mut_ptr() as GDExtensionVariantPtr);
    }

    // Stage 5b: `_dict_guard` drops here → the temp Dictionary handle's
    // internal refcount is released. The output Variant retains its own
    // independent reference from stage 4.
}

/// Helper: insert a `(String key, Variant value)` pair into a live
/// Dictionary handle. Encapsulates the "build a key Variant on a stack
/// slot, resolve the value slot via `dictionary_operator_index`, deep-copy
/// the value Variant into the slot, destroy the key Variant" sequence.
///
/// # Contract
///
/// - `dict_handle` must be a live `Dictionary` typed handle (not a
///   Variant) — i.e. `dict_default_ctor` has run on it and no destructor
///   has run yet.
/// - `key` must NOT already exist in the dictionary (this helper does not
///   handle the destroy-old-slot case — matches the fresh-dict shape used
///   by `pack_tts_output_into_dict_variant`).
/// - `value_variant` must point to a live 24-byte Variant. Its CowData
///   refcount is bumped by `variant_new_copy`; the caller retains
///   ownership of its own reference.
///
/// # Safety
///
/// C ABI internals. All raw-pointer parameters follow the same contract
/// as [`pack_tts_output_into_dict_variant`].
unsafe fn insert_dict_entry_from_str(
    interface: &InterfaceTable,
    dict_handle: *mut core::ffi::c_void,
    key: &str,
    value_variant: GDExtensionConstVariantPtr,
) {
    // 1. Build the key Variant on a stack Variant slot.
    #[repr(C, align(8))]
    struct VariantBuf([u8; 24]);
    let mut key_variant = VariantBuf([0u8; 24]);
    // SAFETY: `variant_from_string_utf8` writes exactly 24 bytes into
    // `r_dest`. Its RAII guard cleans up the temp String handle on
    // panic. On return, `key_variant` holds a live String Variant.
    unsafe {
        variant_from_string_utf8(
            interface,
            key_variant.0.as_mut_ptr() as GDExtensionUninitializedVariantPtr,
            key,
        );
    }

    // 2. Resolve the value slot. Godot creates a fresh Nil slot for
    //    unknown keys (documented — see per-typedef rustdoc on
    //    `dictionary_operator_index`).
    //
    // SAFETY: `dictionary_operator_index` is a live fn pointer.
    // `dict_handle` is a live Dictionary handle per caller doc.
    // `&key_variant` is a live String Variant. The returned slot pointer
    // aliases storage inside `dict_handle` for as long as the Dictionary
    // lives — we use it only for the single `variant_new_copy` call
    // below, no re-entrance possible.
    let slot: GDExtensionVariantPtr = unsafe {
        (interface.dictionary_operator_index)(
            dict_handle as GDExtensionTypePtr,
            key_variant.0.as_ptr() as GDExtensionConstVariantPtr,
        )
    };

    // Defensive: a NULL slot would indicate a corrupt Godot host (the
    // header documents `dictionary_operator_index` as never returning
    // NULL for a valid Dictionary + Variant key). Skip the copy in that
    // pathological case rather than UB'ing the slot write — the result
    // is a Dictionary missing the entry, which the caller surfaces to
    // GDScript as a Nil / missing-key lookup (honest failure rather
    // than a crash).
    if !slot.is_null() {
        // 3. Deep-copy the value Variant into the slot. Godot bumps the
        //    CowData refcount on the value; the dictionary now holds its
        //    own independent reference.
        //
        // SAFETY: `variant_new_copy` is a live fn pointer. `slot` is a
        // live Nil Variant (freshly allocated by Godot for this key —
        // overwriting Nil does not leak). `value_variant` is a live
        // 24-byte Variant per caller doc.
        unsafe {
            (interface.variant_new_copy)(slot as GDExtensionUninitializedVariantPtr, value_variant);
        }
    }

    // 4. Destroy the local key Variant. The dictionary's internal key
    //    storage retains its own independent String reference (bumped
    //    during `dictionary_operator_index`).
    //
    // SAFETY: `variant_destroy` is a live fn pointer. `key_variant` is
    // a live 24-byte Variant.
    unsafe {
        (interface.variant_destroy)(key_variant.0.as_mut_ptr() as GDExtensionVariantPtr);
    }
}

// ---------------------------------------------------------------------------
// var-B: PackedFloat32Array pack (M3-11 T14-followup, `stream_poll`).
// ---------------------------------------------------------------------------

/// Pack a Rust `&[f32]` slice into `r_dest` as a Variant of type
/// `PackedFloat32Array`. Called on the success path of
/// `VokraStream::poll(capacity) -> PackedFloat32Array` (M3-11 T14-followup).
///
/// # Pipeline (5 stages)
///
/// Godot's GDExtension header does not expose a "construct a
/// PackedFloat32Array from raw f32 data" API in one call; the standard
/// pattern (matching godot-cpp internals) is a five-step sequence, all of
/// which route through resolved fn pointers stored on [`InterfaceTable`]:
///
/// 1. **Default-construct** an empty `PackedFloat32Array` in a
///    [`PACKED_FLOAT32_ARRAY_SIZE`]-byte (16-byte) stack buffer via
///    [`InterfaceTable::pfa_default_ctor`][ct]
///    (`variant_get_ptr_constructor(PACKED_FLOAT32_ARRAY, 0)`).
/// 2. **Resize** to `slice.len()` via
///    [`InterfaceTable::pfa_resize_method`][rm]
///    (`variant_get_ptr_builtin_method(PACKED_FLOAT32_ARRAY, "resize",
///    848867239)`). Godot's `resize` allocates the backing
///    `CowData<float>` buffer; the returned int is ignored (we already
///    know the target size).
/// 3. **Copy payload** through
///    [`InterfaceTable::packed_float32_array_operator_index`][oi] +
///    `core::ptr::copy_nonoverlapping`. `operator_index(base, 0)` yields
///    a mutable pointer to element 0; we bulk-copy `slice.len()` f32
///    values in one shot. Skipped for `slice.is_empty()`.
/// 4. **Pack into Variant** via
///    [`InterfaceTable::variant_from_packed_float32_array_ctor`][vp].
///    `CowData` refcount-increment, not a deep copy — the Variant now
///    shares the same backing buffer with our temp handle.
/// 5. **Destroy the temp handle** via
///    [`InterfaceTable::pfa_destructor`][dr], decrementing the refcount
///    back to 1 (owned solely by the Variant). Skipping this step would
///    leak one refcount per call.
///
/// [ct]: crate::ffi::interface::InterfaceTable::pfa_default_ctor
/// [rm]: crate::ffi::interface::InterfaceTable::pfa_resize_method
/// [oi]: crate::ffi::interface::InterfaceTable::packed_float32_array_operator_index
/// [vp]: crate::ffi::interface::InterfaceTable::variant_from_packed_float32_array_ctor
/// [dr]: crate::ffi::interface::InterfaceTable::pfa_destructor
///
/// # Empty slice
///
/// `slice.len() == 0` still executes stages 1, 2, 4, and 5 — the resulting
/// Variant is a valid empty `PackedFloat32Array`. Stage 3 is short-circuited.
///
/// # Panic safety
///
/// This helper does NOT wrap its stack buffer in a `Drop` guard because it
/// contains only fn-pointer calls into Godot (no Rust code between
/// constructor and destructor that can panic). Consistent with the
/// module-doc "Not exception-safe against `abort`" clause: with
/// `panic = "unwind"` (workspace default) the destructor still runs; with
/// `panic = "abort"` the process ends and Godot's CowData leak is
/// process-lifetime.
///
/// # Safety
///
/// - `r_dest` must be a writable 24-byte Variant slot (Godot 4.3-stable
///   pins `sizeof(Variant) == 24` on LP64). On return, `r_dest` holds a
///   fully-constructed PackedFloat32Array Variant; the caller retains
///   ownership.
/// - `slice.len()` must fit into `GDExtensionInt` (`i64`). On LP64 targets
///   this is trivially true (`isize::MAX == i64::MAX`).
/// - The `InterfaceTable` fields consulted here MUST be live fn pointers
///   (resolved by `InterfaceTable::from_proc_address`).
pub unsafe fn pack_f32_slice_into_variant(
    interface: &InterfaceTable,
    r_dest: GDExtensionUninitializedVariantPtr,
    slice: &[f32],
) {
    use crate::ffi::gdextension::GDExtensionInt;
    use core::ptr;

    // 16-byte, 8-aligned stack buffer sized for `PackedFloat32Array` on
    // LP64 (verified constant across Godot 4.1..4.3 `extension_api.json`
    // — see `PACKED_FLOAT32_ARRAY_SIZE`). Wrapping in
    // `#[repr(C, align(8))]` pins the alignment; on ARM64 macOS an
    // under-aligned Godot type write would fault.
    #[repr(C, align(8))]
    struct PfaBuf([u8; PACKED_FLOAT32_ARRAY_SIZE]);
    let mut buf = PfaBuf([0u8; PACKED_FLOAT32_ARRAY_SIZE]);
    let base_ptr = buf.0.as_mut_ptr() as *mut core::ffi::c_void;

    // Stage 1: default-construct. Zero-arg constructor → `p_args` is a
    // valid NULL per the ptrcall contract.
    //
    // SAFETY: `pfa_default_ctor` is a live fn pointer resolved from
    // `variant_get_ptr_constructor(PACKED_FLOAT32_ARRAY, 0)`. `base_ptr`
    // is a writable 16-byte 8-aligned slot.
    unsafe {
        (interface.pfa_default_ctor)(base_ptr, ptr::null::<GDExtensionConstTypePtr>());
    }

    // Stage 2: resize to `slice.len()`. Godot's `resize(new_size: int) ->
    // int` takes a single Int arg (`*const i64`) via the standard
    // ptrcall pattern: `p_args` is a pointer to an array of `*const
    // c_void` where each element points to the raw typed argument.
    // `r_return` is a writable slot for the int return — we discard it
    // because we know the target size.
    let new_size: GDExtensionInt = slice.len() as GDExtensionInt;
    let new_size_ptr = (&new_size) as *const GDExtensionInt as GDExtensionConstTypePtr;
    let args: [GDExtensionConstTypePtr; 1] = [new_size_ptr];
    let mut ret_int: GDExtensionInt = 0;

    // SAFETY: `pfa_resize_method` is a live fn pointer resolved from
    // `variant_get_ptr_builtin_method(PACKED_FLOAT32_ARRAY, "resize",
    // PACKED_FLOAT32_ARRAY_RESIZE_HASH)`. `base_ptr` is the
    // just-constructed PackedFloat32Array. `args` is a valid 1-slot
    // array; `new_size_ptr` is a live `*const i64` for the duration of
    // the call. `ret_int` is a writable 8-byte slot for the int return.
    unsafe {
        (interface.pfa_resize_method)(
            base_ptr,
            args.as_ptr(),
            &mut ret_int as *mut GDExtensionInt as *mut core::ffi::c_void,
            1,
        );
    }

    // Stage 3: bulk-copy payload. `operator_index(base, 0)` returns a
    // pointer to element 0 (mutable); the following `slice.len() - 1`
    // elements are contiguous in memory per Godot's `Vector<float>` /
    // `CowData` guarantees. Skipped for empty slice (`operator_index`
    // may return NULL when the underlying buffer is 0-length).
    if !slice.is_empty() {
        // SAFETY: `packed_float32_array_operator_index` is a live fn
        // pointer. `base_ptr` is the just-resized PackedFloat32Array.
        let dst = unsafe { (interface.packed_float32_array_operator_index)(base_ptr, 0) };
        if !dst.is_null() {
            // SAFETY: `dst` points to `slice.len()` contiguous f32
            // slots (Godot allocated them at stage 2); source slice
            // pointer is valid for `slice.len()` reads by the `&[T]`
            // contract; the two ranges do not overlap (Godot allocation
            // vs. caller's `Vec<f32>`).
            unsafe {
                ptr::copy_nonoverlapping(slice.as_ptr(), dst, slice.len());
            }
        }
    }

    // Stage 4: pack into Variant (CowData refcount increment).
    //
    // SAFETY: `variant_from_packed_float32_array_ctor` is a live fn
    // pointer resolved from
    // `get_variant_from_type_constructor(PACKED_FLOAT32_ARRAY)`.
    // `r_dest` is a writable 24-byte Variant slot per caller doc.
    // `base_ptr` is a valid PackedFloat32Array containing the payload.
    unsafe {
        (interface.variant_from_packed_float32_array_ctor)(r_dest, base_ptr);
    }

    // Stage 5: destroy the temp buffer (decrements refcount to 1).
    //
    // SAFETY: `pfa_destructor` is a live fn pointer. `base_ptr` is the
    // just-constructed PackedFloat32Array; no aliasing hazard because
    // we do not touch `buf` again after this point.
    unsafe {
        (interface.pfa_destructor)(base_ptr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::interface::tests::{
        make_sig_aware_interface, mock_variant_from_int, mock_variant_to_int,
    };
    use core::ptr;
    use core::sync::atomic::{AtomicI64, AtomicU8, Ordering};

    // ------------------------------------------------------------------
    // A minimal per-test mock interface. `make_sig_aware_interface`
    // gives us a fully-populated table whose Int constructors are the
    // no-op `mock_variant_from_int` / `mock_variant_to_int`; we override
    // individual fields where a test wants a specific behaviour.
    // ------------------------------------------------------------------

    /// Storage for the last "input" and "output" observed by a per-test
    /// mock. AtomicI64 keeps the two mocks lock-free.
    static LAST_VARIANT_TYPE_CALL_COUNT: AtomicU8 = AtomicU8::new(0);
    static LAST_VARIANT_TYPE_RETURN: AtomicU8 = AtomicU8::new(0);
    static LAST_TO_INT_OUT: AtomicI64 = AtomicI64::new(0);
    static LAST_FROM_INT_IN: AtomicI64 = AtomicI64::new(0);

    /// Mock `variant_get_type` that returns whatever value was last stored
    /// in `LAST_VARIANT_TYPE_RETURN` (interpreted as `GDExtensionVariantType`).
    unsafe extern "C" fn mock_get_type_configurable(
        _v: GDExtensionConstVariantPtr,
    ) -> GDExtensionVariantType {
        LAST_VARIANT_TYPE_CALL_COUNT.fetch_add(1, Ordering::SeqCst);
        match LAST_VARIANT_TYPE_RETURN.load(Ordering::SeqCst) {
            0 => GDExtensionVariantType::Nil,
            1 => GDExtensionVariantType::Bool,
            2 => GDExtensionVariantType::Int,
            3 => GDExtensionVariantType::Float,
            4 => GDExtensionVariantType::String,
            24 => GDExtensionVariantType::Object,
            27 => GDExtensionVariantType::Dictionary,
            32 => GDExtensionVariantType::PackedFloat32Array,
            _ => GDExtensionVariantType::Nil,
        }
    }

    /// Mock `variant_to_int_ctor` that writes a canned value to `r_out`.
    /// The canned value is `LAST_TO_INT_OUT`.
    unsafe extern "C" fn mock_to_int_canned(
        r_out: crate::ffi::gdextension::GDExtensionUninitializedTypePtr,
        _p: GDExtensionVariantPtr,
    ) {
        // SAFETY: caller (variant_to_i64) provides a writable 8-byte slot.
        unsafe {
            (r_out as *mut i64).write(LAST_TO_INT_OUT.load(Ordering::SeqCst));
        }
    }

    /// Mock `variant_from_int_ctor` that records the input value.
    unsafe extern "C" fn mock_from_int_recording(
        _r: GDExtensionUninitializedVariantPtr,
        p: crate::ffi::gdextension::GDExtensionTypePtr,
    ) {
        // SAFETY: caller passes an 8-byte pointer to i64.
        let v = unsafe { (p as *const i64).read() };
        LAST_FROM_INT_IN.store(v, Ordering::SeqCst);
    }

    /// Build an interface table whose type/int-ctor fields go through the
    /// configurable mocks above. Non-Variant fields are the sig-aware
    /// mock's sentinel-backed fields — unused by variant.rs tests.
    fn make_configurable_interface() -> InterfaceTable {
        let mut table = make_sig_aware_interface();
        table.variant_get_type = mock_get_type_configurable;
        table.variant_to_int_ctor = mock_to_int_canned;
        table.variant_from_int_ctor = mock_from_int_recording;
        table
    }

    #[test]
    fn variant_get_type_dispatches_to_interface() {
        let iface = make_configurable_interface();
        LAST_VARIANT_TYPE_CALL_COUNT.store(0, Ordering::SeqCst);
        LAST_VARIANT_TYPE_RETURN.store(2, Ordering::SeqCst); // Int

        // SAFETY: `v` is unused by our mock; passing NULL is safe here
        // because the mock does not deref it. (Real Godot would UB.)
        let ty = unsafe { variant_get_type(&iface, ptr::null()) };
        assert_eq!(ty, GDExtensionVariantType::Int);
        assert_eq!(LAST_VARIANT_TYPE_CALL_COUNT.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn variant_to_i64_returns_value_on_int_type() {
        let iface = make_configurable_interface();
        LAST_VARIANT_TYPE_RETURN.store(2, Ordering::SeqCst); // Int
        LAST_TO_INT_OUT.store(42, Ordering::SeqCst);

        // SAFETY: unused by mock.
        let result = unsafe { variant_to_i64(&iface, ptr::null()) };
        assert_eq!(result, Ok(42));
    }

    #[test]
    fn variant_to_i64_returns_err_with_actual_type_on_mismatch() {
        let iface = make_configurable_interface();
        // Set the type to String → mismatch (Int expected).
        LAST_VARIANT_TYPE_RETURN.store(4, Ordering::SeqCst);
        LAST_TO_INT_OUT.store(999, Ordering::SeqCst); // must be ignored

        // SAFETY: unused by mock.
        let result = unsafe { variant_to_i64(&iface, ptr::null()) };
        assert_eq!(result, Err(GDExtensionVariantType::String));
    }

    #[test]
    fn variant_to_i64_returns_err_on_nil_variant() {
        let iface = make_configurable_interface();
        LAST_VARIANT_TYPE_RETURN.store(0, Ordering::SeqCst);

        // SAFETY: unused by mock.
        let result = unsafe { variant_to_i64(&iface, ptr::null()) };
        assert_eq!(result, Err(GDExtensionVariantType::Nil));
    }

    #[test]
    fn variant_to_i64_returns_err_on_packed_float32_array() {
        let iface = make_configurable_interface();
        // 32 = `GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY` in Godot
        // 4.3-stable (`variant.h` / `gdextension_interface.h`). A previous
        // revision of this file used 30, which is actually
        // `PACKED_INT32_ARRAY` — the mock got away with it because the
        // mock's return-value table listed 30→PackedFloat32Array too. The
        // fix locks both ends of the mock to the header-verified value.
        LAST_VARIANT_TYPE_RETURN.store(32, Ordering::SeqCst);

        // SAFETY: unused by mock.
        let result = unsafe { variant_to_i64(&iface, ptr::null()) };
        assert_eq!(result, Err(GDExtensionVariantType::PackedFloat32Array));
    }

    #[test]
    fn variant_to_i64_negative_value_roundtrip() {
        let iface = make_configurable_interface();
        LAST_VARIANT_TYPE_RETURN.store(2, Ordering::SeqCst); // Int
        LAST_TO_INT_OUT.store(-9_999_999_999, Ordering::SeqCst);

        // SAFETY: unused by mock.
        let result = unsafe { variant_to_i64(&iface, ptr::null()) };
        assert_eq!(result, Ok(-9_999_999_999));
    }

    #[test]
    fn variant_from_i64_forwards_value_to_constructor() {
        let iface = make_configurable_interface();
        LAST_FROM_INT_IN.store(0, Ordering::SeqCst);

        let mut slot = [0u8; 24];
        // SAFETY: `slot` is a writable 24-byte buffer; mock does not
        // deref `r_dest` beyond side-effect recording of `p_in`.
        unsafe { variant_from_i64(&iface, slot.as_mut_ptr() as _, 12345) };
        assert_eq!(LAST_FROM_INT_IN.load(Ordering::SeqCst), 12345);
    }

    #[test]
    fn write_nil_variant_dispatches_to_interface() {
        // Wire a mock that records its call.
        static CALLED: AtomicU8 = AtomicU8::new(0);
        unsafe extern "C" fn recording_nil(_r: GDExtensionUninitializedVariantPtr) {
            CALLED.fetch_add(1, Ordering::SeqCst);
        }

        let mut iface = make_configurable_interface();
        iface.variant_new_nil = recording_nil;
        CALLED.store(0, Ordering::SeqCst);

        let mut slot = [0u8; 24];
        // SAFETY: writable 24-byte slot; recorder mock does not deref.
        unsafe { write_nil_variant(&iface, slot.as_mut_ptr() as _) };

        assert_eq!(CALLED.load(Ordering::SeqCst), 1);
    }

    // Compile-only sanity: the from/to i64 helpers accept the actual
    // fn-pointer types stored in the InterfaceTable (no coercion drift).
    #[cfg(test)]
    #[allow(dead_code)]
    fn compile_only_signatures_match(iface: &InterfaceTable) {
        // Take addresses to prevent LTO from eliminating them. The
        // `as *const ()` step suppresses the
        // `function_casts_as_integer` lint (function items must round-trip
        // through a raw pointer before an integer cast).
        let _a = iface.variant_from_int_ctor as *const () as usize;
        let _b = iface.variant_to_int_ctor as *const () as usize;
        let _c = mock_variant_from_int as *const () as usize;
        let _d = mock_variant_to_int as *const () as usize;
    }

    // ------------------------------------------------------------------
    // `variant_to_packed_float32_slice` (var-C) direct coverage.
    //
    // The `stream_push_pcm` trampoline delegates its
    // `PackedFloat32Array` unpack to this helper; a bug here would
    // silently miroute a valid PCM chunk. These tests exercise the
    // helper directly (without the trampoline wrapper) to lock down:
    //   - Type-check rejection with `Err(actual_type)` on a non-PFA
    //     Variant (String → the exact wrong-type surface used by the
    //     trampoline to emit `InvalidArgument(0, PackedFloat32Array)`).
    //   - Empty-slice branch: `size == 0` → closure receives `&[]`
    //     WITHOUT touching `operator_index_const` (Godot documents this
    //     API as valid to return NULL for a 0-length backing buffer,
    //     so the empty short-circuit is a correctness feature).
    //   - The closure's return value round-trips through
    //     `Ok(closure_result)`.
    // ------------------------------------------------------------------

    /// Mock `variant_get_type` that returns `PackedFloat32Array`.
    unsafe extern "C" fn mock_gt_pfa(
        _v: crate::ffi::gdextension::GDExtensionConstVariantPtr,
    ) -> GDExtensionVariantType {
        GDExtensionVariantType::PackedFloat32Array
    }

    /// Mock `variant_get_type` that returns `String` (wrong type for a
    /// PFA unpack — surfaces as `Err(String)`).
    unsafe extern "C" fn mock_gt_string(
        _v: crate::ffi::gdextension::GDExtensionConstVariantPtr,
    ) -> GDExtensionVariantType {
        GDExtensionVariantType::String
    }

    /// Mock `variant_to_packed_float32_array_ctor` that zeros the
    /// 16-byte typed slot. Matches the "default-constructed empty
    /// PackedFloat32Array" pattern (empty CowData pointer).
    ///
    /// # Safety
    ///
    /// `r_out` must be a writable 16-byte, 8-byte-aligned slot.
    unsafe extern "C" fn mock_pfa_ctor_zeroes(
        r_out: crate::ffi::gdextension::GDExtensionUninitializedTypePtr,
        _v: crate::ffi::gdextension::GDExtensionVariantPtr,
    ) {
        // SAFETY: caller (`variant_to_packed_float32_slice`) provides
        // exactly `PACKED_FLOAT32_ARRAY_SIZE` writable bytes per the
        // `PackedFloat32ArraySlot` layout.
        unsafe {
            (r_out as *mut u8).write_bytes(0, PACKED_FLOAT32_ARRAY_SIZE);
        }
    }

    /// Type-check rejection: a String-typed Variant unpacks as
    /// `Err(GDExtensionVariantType::String)` — the exact "wrong type"
    /// value the trampoline needs to route to
    /// `InvalidArgument(0, PackedFloat32Array)`.
    #[test]
    fn variant_to_packed_float32_slice_rejects_wrong_type() {
        let mut iface = crate::ffi::interface::tests::make_sig_aware_interface();
        iface.variant_get_type = mock_gt_string;

        // If the type check WERE to fail, the closure would run and
        // panic (fail the test). Also assert the return path in the
        // check-fail branch is a clean `Err(actual_type)`.
        static WAS_CALLED: AtomicU8 = AtomicU8::new(0);
        WAS_CALLED.store(0, Ordering::SeqCst);

        // SAFETY: `p_variant` is unused by the mock (mock just returns
        // a constant).
        let result = unsafe {
            variant_to_packed_float32_slice(&iface, ptr::null(), |_pcm: &[f32]| {
                WAS_CALLED.fetch_add(1, Ordering::SeqCst);
            })
        };
        assert_eq!(result, Err(GDExtensionVariantType::String));
        assert_eq!(
            WAS_CALLED.load(Ordering::SeqCst),
            0,
            "closure must not be invoked on type-check failure",
        );
    }

    /// Empty-size fast path: when `pfa_size_method` yields 0, the
    /// closure receives `&[]` and its return value round-trips through
    /// `Ok(_)`. The `operator_index_const` fn is NEVER invoked
    /// (documented as valid to return NULL for a 0-length buffer, so
    /// the short-circuit is a correctness feature not an optimisation).
    #[test]
    fn variant_to_packed_float32_slice_empty_size_yields_empty_slice() {
        let mut iface = crate::ffi::interface::tests::make_sig_aware_interface();
        iface.variant_get_type = mock_gt_pfa;
        iface.variant_to_packed_float32_array_ctor = mock_pfa_ctor_zeroes;
        // sig-aware default `pfa_size_method` is a no-op that does NOT
        // touch its `r_return` slot; the caller pre-initialises
        // `size_i64 = 0`, so size stays 0.

        // Track that `operator_index_const` is NOT called on the
        // empty-size branch. If it WERE called, we'd notice because
        // `sentinel` (used as `operator_index_const` by the sig-aware
        // GPA path) takes zero args and calling it as a
        // `PackedFloat32ArrayOperatorIndexConst` with (base, index)
        // would push extra bytes onto the C ABI stack — undefined but
        // observable in practice. Instead we rely on the empty-slice
        // branch's short-circuit: no `operator_index_const` in the
        // called path.

        let observed_len = core::cell::Cell::new(usize::MAX);
        // SAFETY: `p_variant` is unused by the mock.
        let result = unsafe {
            variant_to_packed_float32_slice(&iface, ptr::null(), |pcm: &[f32]| {
                observed_len.set(pcm.len());
                42u32 // closure return value
            })
        };
        assert_eq!(result, Ok(42));
        assert_eq!(
            observed_len.get(),
            0,
            "closure must receive an empty slice when size == 0",
        );
    }

    /// Round-trip: the closure's return value bubbles up through the
    /// `Ok` layer. Locks in the trampoline's match on
    /// `Result<Result<(), VokraError>, GDExtensionVariantType>` — the
    /// outer `Ok` from `variant_to_packed_float32_slice` distinguishes
    /// backend errors (inner `Err`) from type mismatches (outer `Err`).
    #[test]
    fn variant_to_packed_float32_slice_forwards_closure_return_value() {
        let mut iface = crate::ffi::interface::tests::make_sig_aware_interface();
        iface.variant_get_type = mock_gt_pfa;
        iface.variant_to_packed_float32_array_ctor = mock_pfa_ctor_zeroes;

        // Mirror the trampoline's shape: closure returns
        // Result<(), &'static str>. Test both outcomes.

        // (a) closure returns Ok(())
        // SAFETY: `p_variant` is unused by the mock.
        let ok_result: Result<Result<(), &'static str>, GDExtensionVariantType> =
            unsafe { variant_to_packed_float32_slice(&iface, ptr::null(), |_pcm| Ok(())) };
        assert_eq!(ok_result, Ok(Ok(())));

        // (b) closure returns Err — the trampoline's "backend error"
        // shape.
        // SAFETY: same.
        let err_result: Result<Result<(), &'static str>, GDExtensionVariantType> =
            unsafe { variant_to_packed_float32_slice(&iface, ptr::null(), |_pcm| Err("boom")) };
        assert_eq!(err_result, Ok(Err("boom")));
    }

    // ------------------------------------------------------------------
    // var-B: PackedFloat32Array pack pipeline (M3-11 T14-followup).
    // Consumed by the `stream_poll` trampoline on the success path
    // (`VokraStream::poll(capacity) -> Vec<f32>` → PackedFloat32Array
    // Variant).
    //
    // Tests record the 5-stage sequence (default_ctor, resize_method,
    // operator_index, variant_from_packed_float32_array_ctor,
    // pfa_destructor) via atomic counters. Serialised against sibling
    // tests through `registry::tests::TEST_LOCK` because we mutate
    // static counters + a static backing buffer.
    // ------------------------------------------------------------------

    /// Per-stage call counters for the pack pipeline. Loaded/stored by
    /// the recording mocks below; assertions read them after each test.
    static PACK_DEFAULT_CTOR_CALLS: AtomicU8 = AtomicU8::new(0);
    static PACK_RESIZE_CALLS: AtomicU8 = AtomicU8::new(0);
    static PACK_RESIZE_LAST_NEW_SIZE: AtomicI64 = AtomicI64::new(-1);
    static PACK_OPERATOR_INDEX_CALLS: AtomicU8 = AtomicU8::new(0);
    static PACK_OPERATOR_INDEX_LAST_INDEX: AtomicI64 = AtomicI64::new(-1);
    static PACK_VARIANT_FROM_CALLS: AtomicU8 = AtomicU8::new(0);
    static PACK_DESTRUCTOR_CALLS: AtomicU8 = AtomicU8::new(0);

    /// Backing store the recording `operator_index` mock returns. Sized
    /// generously so the test can copy any small `[f32]` payload without
    /// stack overflow.
    static mut PACK_BACKING_STORE: [f32; 16] = [0.0f32; 16];

    fn reset_pack_counters() {
        PACK_DEFAULT_CTOR_CALLS.store(0, Ordering::SeqCst);
        PACK_RESIZE_CALLS.store(0, Ordering::SeqCst);
        PACK_RESIZE_LAST_NEW_SIZE.store(-1, Ordering::SeqCst);
        PACK_OPERATOR_INDEX_CALLS.store(0, Ordering::SeqCst);
        PACK_OPERATOR_INDEX_LAST_INDEX.store(-1, Ordering::SeqCst);
        PACK_VARIANT_FROM_CALLS.store(0, Ordering::SeqCst);
        PACK_DESTRUCTOR_CALLS.store(0, Ordering::SeqCst);
        // SAFETY: single-threaded test scope guarded by TEST_LOCK.
        #[allow(static_mut_refs)]
        unsafe {
            for x in PACK_BACKING_STORE.iter_mut() {
                *x = 0.0;
            }
        }
    }

    unsafe extern "C" fn recording_pack_default_ctor(
        _p_base: crate::ffi::gdextension::GDExtensionUninitializedTypePtr,
        _p_args: *const crate::ffi::gdextension::GDExtensionConstTypePtr,
    ) {
        PACK_DEFAULT_CTOR_CALLS.fetch_add(1, Ordering::SeqCst);
    }

    unsafe extern "C" fn recording_pack_resize(
        _p_base: crate::ffi::gdextension::GDExtensionTypePtr,
        p_args: *const crate::ffi::gdextension::GDExtensionConstTypePtr,
        _r_return: crate::ffi::gdextension::GDExtensionTypePtr,
        _p_argument_count: i32,
    ) {
        PACK_RESIZE_CALLS.fetch_add(1, Ordering::SeqCst);
        // Read `p_args[0]` as `*const i64` and record the requested new
        // size. `pack_f32_slice_into_variant` always passes a valid
        // 1-element array pointing at a stack-owned `GDExtensionInt`.
        //
        // SAFETY: `p_args` is the caller's 1-element array; `*p_args`
        // is the `new_size_ptr` cast from `&GDExtensionInt`.
        unsafe {
            let new_size_ptr = *p_args as *const crate::ffi::gdextension::GDExtensionInt;
            let new_size = new_size_ptr.read();
            PACK_RESIZE_LAST_NEW_SIZE.store(new_size, Ordering::SeqCst);
        }
    }

    unsafe extern "C" fn recording_pack_operator_index(
        _p_self: crate::ffi::gdextension::GDExtensionTypePtr,
        p_index: crate::ffi::gdextension::GDExtensionInt,
    ) -> *mut f32 {
        PACK_OPERATOR_INDEX_CALLS.fetch_add(1, Ordering::SeqCst);
        PACK_OPERATOR_INDEX_LAST_INDEX.store(p_index, Ordering::SeqCst);
        // Return a pointer into the static backing store so the caller
        // can `copy_nonoverlapping` a small slice into it. The static
        // is only touched under `TEST_LOCK`, so the aliasing is fine.
        //
        // SAFETY: single-threaded test; TEST_LOCK held. `p_index` is
        // always 0 for our stage-3 copy path.
        #[allow(static_mut_refs)]
        unsafe {
            PACK_BACKING_STORE.as_mut_ptr()
        }
    }

    unsafe extern "C" fn recording_pack_variant_from(
        _r_out: GDExtensionUninitializedVariantPtr,
        _p_in: crate::ffi::gdextension::GDExtensionTypePtr,
    ) {
        PACK_VARIANT_FROM_CALLS.fetch_add(1, Ordering::SeqCst);
    }

    unsafe extern "C" fn recording_pack_destructor(
        _p_base: crate::ffi::gdextension::GDExtensionTypePtr,
    ) {
        PACK_DESTRUCTOR_CALLS.fetch_add(1, Ordering::SeqCst);
    }

    /// Install a fully-recording pack interface. The 5 stages run
    /// through per-stage atomic counters; `operator_index` returns a
    /// pointer into a static backing store so the copy-nonoverlapping
    /// stage can safely land its bytes.
    fn install_recording_pack_interface() -> InterfaceTable {
        let mut iface = make_sig_aware_interface();
        iface.pfa_default_ctor = recording_pack_default_ctor;
        iface.pfa_resize_method = recording_pack_resize;
        iface.packed_float32_array_operator_index = recording_pack_operator_index;
        iface.variant_from_packed_float32_array_ctor = recording_pack_variant_from;
        iface.pfa_destructor = recording_pack_destructor;
        iface
    }

    /// Empty `&[f32]` slice: stages 1, 2, 4, 5 must fire exactly once;
    /// stage 3 (`operator_index`) is skipped because there's no payload
    /// to copy. The RAII posture also requires that the destructor
    /// (stage 5) runs even on the empty path, else the CowData refcount
    /// leaks.
    #[test]
    fn pack_f32_slice_into_variant_empty_slice_skips_operator_index() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        reset_pack_counters();
        let iface = install_recording_pack_interface();

        let mut slot = [0u8; 24];
        let slice: &[f32] = &[];
        // SAFETY: 24-byte writable Variant slot; all pack-stage mocks
        // ignore `r_dest` beyond the recording side-effect.
        unsafe {
            pack_f32_slice_into_variant(&iface, slot.as_mut_ptr() as _, slice);
        }

        assert_eq!(
            PACK_DEFAULT_CTOR_CALLS.load(Ordering::SeqCst),
            1,
            "stage 1 (default_ctor) must fire exactly once",
        );
        assert_eq!(
            PACK_RESIZE_CALLS.load(Ordering::SeqCst),
            1,
            "stage 2 (resize) must fire exactly once even on empty",
        );
        assert_eq!(
            PACK_RESIZE_LAST_NEW_SIZE.load(Ordering::SeqCst),
            0,
            "stage 2 new_size must be 0 for empty slice",
        );
        assert_eq!(
            PACK_OPERATOR_INDEX_CALLS.load(Ordering::SeqCst),
            0,
            "stage 3 (operator_index) MUST be skipped for empty slice",
        );
        assert_eq!(
            PACK_VARIANT_FROM_CALLS.load(Ordering::SeqCst),
            1,
            "stage 4 (variant_from_ctor) must fire exactly once",
        );
        assert_eq!(
            PACK_DESTRUCTOR_CALLS.load(Ordering::SeqCst),
            1,
            "stage 5 (destructor) must fire exactly once to release CowData",
        );
    }

    /// Non-empty `&[f32]`: all 5 stages must fire exactly once; the
    /// payload must land in the backing buffer returned by the
    /// `operator_index` mock (verifies the `copy_nonoverlapping` step).
    #[test]
    fn pack_f32_slice_into_variant_non_empty_copies_payload_via_operator_index() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        reset_pack_counters();
        let iface = install_recording_pack_interface();

        let mut slot = [0u8; 24];
        let payload: [f32; 3] = [1.5, -2.25, 3.75];
        // SAFETY: 24-byte writable Variant slot; recording mocks land
        // the payload into the static backing store.
        unsafe {
            pack_f32_slice_into_variant(&iface, slot.as_mut_ptr() as _, &payload);
        }

        assert_eq!(PACK_DEFAULT_CTOR_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(PACK_RESIZE_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(PACK_RESIZE_LAST_NEW_SIZE.load(Ordering::SeqCst), 3);
        assert_eq!(
            PACK_OPERATOR_INDEX_CALLS.load(Ordering::SeqCst),
            1,
            "stage 3 must fire once for non-empty",
        );
        assert_eq!(
            PACK_OPERATOR_INDEX_LAST_INDEX.load(Ordering::SeqCst),
            0,
            "operator_index MUST be called with p_index=0 (base element)",
        );
        assert_eq!(PACK_VARIANT_FROM_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(PACK_DESTRUCTOR_CALLS.load(Ordering::SeqCst), 1);

        // Verify the bytes landed in the backing store. Bit-exact so a
        // silent width-mismatch (e.g. accidentally treating &[f64]) is
        // caught.
        //
        // SAFETY: single-threaded test scope guarded by TEST_LOCK.
        let landed: [f32; 3] = {
            #[allow(static_mut_refs)]
            unsafe {
                [
                    PACK_BACKING_STORE[0],
                    PACK_BACKING_STORE[1],
                    PACK_BACKING_STORE[2],
                ]
            }
        };
        assert_eq!(landed, payload);
    }

    /// A larger slice (10 f32s) exercises the `resize(new_size)` +
    /// `copy_nonoverlapping` path with a non-trivial size and verifies
    /// the resize argument is threaded through correctly. This is the
    /// realistic shape for a VAD `poll(capacity=512)` returning e.g. 10
    /// speech probabilities.
    #[test]
    fn pack_f32_slice_into_variant_resize_new_size_matches_slice_len() {
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        reset_pack_counters();
        let iface = install_recording_pack_interface();

        let mut slot = [0u8; 24];
        let payload: [f32; 10] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        // SAFETY: 24-byte writable Variant slot; recording mocks land
        // the payload into the static backing store (16 slots ≥ 10).
        unsafe {
            pack_f32_slice_into_variant(&iface, slot.as_mut_ptr() as _, &payload);
        }

        assert_eq!(PACK_RESIZE_LAST_NEW_SIZE.load(Ordering::SeqCst), 10);
        // Bit-exact copy for the full payload.
        //
        // SAFETY: single-threaded test scope guarded by TEST_LOCK.
        let landed: [f32; 10] = {
            #[allow(static_mut_refs)]
            unsafe {
                [
                    PACK_BACKING_STORE[0],
                    PACK_BACKING_STORE[1],
                    PACK_BACKING_STORE[2],
                    PACK_BACKING_STORE[3],
                    PACK_BACKING_STORE[4],
                    PACK_BACKING_STORE[5],
                    PACK_BACKING_STORE[6],
                    PACK_BACKING_STORE[7],
                    PACK_BACKING_STORE[8],
                    PACK_BACKING_STORE[9],
                ]
            }
        };
        assert_eq!(landed, payload);
    }

    // ------------------------------------------------------------------
    // `variant_to_string_owned` (var-A unpack) direct coverage.
    //
    // The `session_synthesize` trampoline delegates its `String` unpack
    // to this helper; a bug here would silently miroute a valid TTS
    // input text. These tests exercise the helper directly (without
    // the trampoline wrapper) to lock down:
    //   - Type-check rejection with `Err(actual_type)` on a non-String
    //     Variant (Int → the exact wrong-type surface used by the
    //     trampoline to emit `InvalidArgument(0, String)`).
    //   - Empty-string fast path: Phase A returning 0 → `Ok(String::new())`
    //     WITHOUT Phase B running.
    //   - Non-empty round-trip: mocked Phase A returns a length, Phase
    //     B fills a canned UTF-8 payload → returned `String` matches.
    //   - Truncated over-report defence: mocked Phase B reporting a
    //     `written > byte_len` value is clamped to `byte_len`.
    // ------------------------------------------------------------------

    /// Mock `variant_get_type` that returns `String`.
    unsafe extern "C" fn mock_gt_str_ok(
        _v: crate::ffi::gdextension::GDExtensionConstVariantPtr,
    ) -> GDExtensionVariantType {
        GDExtensionVariantType::String
    }

    /// Mock `variant_get_type` that returns `Int` (wrong type for a
    /// String unpack — surfaces as `Err(Int)`).
    unsafe extern "C" fn mock_gt_str_wrong(
        _v: crate::ffi::gdextension::GDExtensionConstVariantPtr,
    ) -> GDExtensionVariantType {
        GDExtensionVariantType::Int
    }

    /// Mock `variant_to_string_ctor` that zeros the 8-byte typed slot.
    /// Matches the "default-constructed empty String" pattern (empty
    /// CowData pointer) — Godot's real ctor would populate the bytes
    /// with a CowData handle.
    unsafe extern "C" fn mock_str_ctor_zeroes(
        r_out: crate::ffi::gdextension::GDExtensionUninitializedTypePtr,
        _v: crate::ffi::gdextension::GDExtensionVariantPtr,
    ) {
        // SAFETY: caller provides exactly STRING_SIZE writable bytes.
        unsafe {
            (r_out as *mut u8).write_bytes(0, STRING_SIZE);
        }
    }

    /// Canned UTF-8 payload + per-phase call counter for
    /// `mock_string_to_utf8_chars_configurable`. Individual tests reset
    /// the atomics before use.
    static UTF8_LEN_PHASE_A: core::sync::atomic::AtomicUsize =
        core::sync::atomic::AtomicUsize::new(0);
    static UTF8_LEN_PHASE_B: core::sync::atomic::AtomicUsize =
        core::sync::atomic::AtomicUsize::new(0);
    static UTF8_PAYLOAD_LEN: core::sync::atomic::AtomicUsize =
        core::sync::atomic::AtomicUsize::new(0);
    static UTF8_PHASE_CALL_COUNT: core::sync::atomic::AtomicUsize =
        core::sync::atomic::AtomicUsize::new(0);
    /// 32 bytes of scratch UTF-8 payload; long enough for the tests.
    static UTF8_PAYLOAD: [u8; 32] = *b"hello, world! (utf8 test buffer)";

    /// Configurable `string_to_utf8_chars` mock. First call returns
    /// `UTF8_LEN_PHASE_A` (byte-length probe); subsequent non-NULL
    /// calls fill `min(cap, UTF8_PAYLOAD_LEN)` bytes into `r_text` and
    /// return `UTF8_LEN_PHASE_B`. The dual counter lets tests assert
    /// the two-phase probe visits the mock in the expected order.
    unsafe extern "C" fn mock_string_to_utf8_chars_configurable(
        _p_self: GDExtensionConstStringPtr,
        r_text: *mut core::ffi::c_char,
        p_max_write_length: GDExtensionInt,
    ) -> GDExtensionInt {
        use core::sync::atomic::Ordering;
        let n = UTF8_PHASE_CALL_COUNT.fetch_add(1, Ordering::SeqCst);
        if n == 0 || r_text.is_null() {
            // Phase A probe (or any subsequent NULL-buffer call).
            return UTF8_LEN_PHASE_A.load(Ordering::SeqCst) as GDExtensionInt;
        }
        // Phase B fill. Write min(cap, PAYLOAD_LEN) bytes.
        let cap = if p_max_write_length < 0 {
            0
        } else {
            p_max_write_length as usize
        };
        let payload_len = UTF8_PAYLOAD_LEN.load(Ordering::SeqCst);
        let n_write = cap.min(payload_len).min(UTF8_PAYLOAD.len());
        if n_write > 0 {
            // SAFETY: caller (`variant_to_string_owned` Phase B) provides
            // a writable `cap`-byte buffer; `n_write <= cap`. The
            // payload is a `'static` UTF-8 constant; no aliasing with
            // Godot's typed String handle.
            unsafe {
                core::ptr::copy_nonoverlapping(UTF8_PAYLOAD.as_ptr(), r_text as *mut u8, n_write);
            }
        }
        UTF8_LEN_PHASE_B.load(Ordering::SeqCst) as GDExtensionInt
    }

    /// Type-check rejection: an Int-typed Variant unpacks as
    /// `Err(GDExtensionVariantType::Int)` — the exact "wrong type"
    /// value the trampoline needs to route to
    /// `InvalidArgument(0, String)`.
    #[test]
    fn variant_to_string_owned_rejects_wrong_type() {
        let mut iface = crate::ffi::interface::tests::make_sig_aware_interface();
        iface.variant_get_type = mock_gt_str_wrong;

        // SAFETY: `p_variant` is unused by the mock.
        let result = unsafe { variant_to_string_owned(&iface, ptr::null()) };
        assert_eq!(result, Err(GDExtensionVariantType::Int));
    }

    /// Empty-string fast path: Phase A returns 0 → `Ok(String::new())`.
    /// The mock's Phase B is short-circuited (the helper returns before
    /// the second call).
    #[test]
    fn variant_to_string_owned_empty_string_shortcircuits() {
        use core::sync::atomic::Ordering;
        // The `UTF8_*` payload/phase atomics driving
        // `mock_string_to_utf8_chars_configurable` are process-global.
        // All three `variant_to_string_owned_*` tests arm them with
        // different values, so they MUST NOT interleave — without this
        // lock one test observes another's payload (seen in the wild as
        // `left: Ok("hello"), right: Ok("")`).
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut iface = crate::ffi::interface::tests::make_sig_aware_interface();
        iface.variant_get_type = mock_gt_str_ok;
        iface.variant_to_string_ctor = mock_str_ctor_zeroes;
        iface.string_to_utf8_chars = mock_string_to_utf8_chars_configurable;

        UTF8_LEN_PHASE_A.store(0, Ordering::SeqCst);
        UTF8_LEN_PHASE_B.store(999, Ordering::SeqCst); // must NOT be consumed
        UTF8_PAYLOAD_LEN.store(0, Ordering::SeqCst);
        UTF8_PHASE_CALL_COUNT.store(0, Ordering::SeqCst);

        // SAFETY: `p_variant` unused by mocks; interface fields all
        // populated by sig-aware base + our overrides.
        let result = unsafe { variant_to_string_owned(&iface, ptr::null()) };
        assert_eq!(result, Ok(String::new()));
        assert_eq!(
            UTF8_PHASE_CALL_COUNT.load(Ordering::SeqCst),
            1,
            "Phase B must not run when Phase A reports 0 bytes",
        );
    }

    /// Non-empty round-trip: Phase A returns a length, Phase B fills the
    /// buffer with UTF-8 bytes, the resulting Rust `String` matches the
    /// payload exactly.
    #[test]
    fn variant_to_string_owned_roundtrips_utf8_payload() {
        use core::sync::atomic::Ordering;
        // See `..._empty_string_shortcircuits`: shared `UTF8_*` globals.
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut iface = crate::ffi::interface::tests::make_sig_aware_interface();
        iface.variant_get_type = mock_gt_str_ok;
        iface.variant_to_string_ctor = mock_str_ctor_zeroes;
        iface.string_to_utf8_chars = mock_string_to_utf8_chars_configurable;

        // "hello, world!" = 13 bytes. Store into the payload prefix.
        const N: usize = 13;
        UTF8_LEN_PHASE_A.store(N, Ordering::SeqCst);
        UTF8_LEN_PHASE_B.store(N, Ordering::SeqCst);
        UTF8_PAYLOAD_LEN.store(N, Ordering::SeqCst);
        UTF8_PHASE_CALL_COUNT.store(0, Ordering::SeqCst);

        // SAFETY: same as above.
        let result = unsafe { variant_to_string_owned(&iface, ptr::null()) };
        let expected = String::from("hello, world!");
        assert_eq!(result, Ok(expected));
        assert_eq!(
            UTF8_PHASE_CALL_COUNT.load(Ordering::SeqCst),
            2,
            "expected Phase A + Phase B (=2 calls)",
        );
    }

    /// Over-report defence: if Phase B claims to have written MORE
    /// bytes than the allocated buffer, the helper clamps to
    /// `byte_len`. Godot's header pins this as non-fallible in
    /// practice, but the defensive path prevents a downstream
    /// `truncate` from walking off the end of the buffer.
    #[test]
    fn variant_to_string_owned_defensive_clamps_over_report() {
        use core::sync::atomic::Ordering;
        // See `..._empty_string_shortcircuits`: shared `UTF8_*` globals.
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut iface = crate::ffi::interface::tests::make_sig_aware_interface();
        iface.variant_get_type = mock_gt_str_ok;
        iface.variant_to_string_ctor = mock_str_ctor_zeroes;
        iface.string_to_utf8_chars = mock_string_to_utf8_chars_configurable;

        const N: usize = 5;
        UTF8_LEN_PHASE_A.store(N, Ordering::SeqCst);
        UTF8_LEN_PHASE_B.store(999_999, Ordering::SeqCst); // over-report!
        UTF8_PAYLOAD_LEN.store(N, Ordering::SeqCst);
        UTF8_PHASE_CALL_COUNT.store(0, Ordering::SeqCst);

        // SAFETY: same as above.
        let result = unsafe { variant_to_string_owned(&iface, ptr::null()) };
        // Clamped to 5 bytes → "hello".
        assert_eq!(result, Ok(String::from("hello")));
    }

    // ------------------------------------------------------------------
    // `pack_tts_output_into_dict_variant` (var-D) direct coverage.
    //
    // Testing the full happy path requires a live Godot host (real
    // Dictionary + Variant packers with side-effect-visible outputs).
    // In our unit-test environment the sig-aware mock's ctors are
    // typed no-ops — they don't populate the output Variant with any
    // observable shape. What we CAN lock down without a live host:
    //   - The pipeline visits every Interface fn we depend on
    //     (default_ctor / from_ctor / operator_index / new_copy /
    //     destroy / dict_destructor) in the right order and count.
    //   - Empty `pcm` does not skip stages; the destroy count still
    //     matches the construct count.
    // ------------------------------------------------------------------

    /// Per-fn call-counters. Reset before each test.
    static DICT_DEFAULT_CTOR_CALLS: core::sync::atomic::AtomicUsize =
        core::sync::atomic::AtomicUsize::new(0);
    static DICT_DESTRUCTOR_CALLS: core::sync::atomic::AtomicUsize =
        core::sync::atomic::AtomicUsize::new(0);
    static DICT_OP_INDEX_CALLS: core::sync::atomic::AtomicUsize =
        core::sync::atomic::AtomicUsize::new(0);
    static VARIANT_NEW_COPY_CALLS: core::sync::atomic::AtomicUsize =
        core::sync::atomic::AtomicUsize::new(0);
    static VARIANT_DESTROY_CALLS: core::sync::atomic::AtomicUsize =
        core::sync::atomic::AtomicUsize::new(0);
    static VARIANT_FROM_DICT_CTOR_CALLS: core::sync::atomic::AtomicUsize =
        core::sync::atomic::AtomicUsize::new(0);

    /// Stable NON-NULL storage that the mock `dictionary_operator_index`
    /// returns so `slot.is_null()` in the helper is false and
    /// `variant_new_copy` is exercised.
    static mut DICT_OP_INDEX_STORAGE: [u8; 24] = [0u8; 24];

    unsafe extern "C" fn mock_recording_dict_default_ctor(
        _p_base: crate::ffi::gdextension::GDExtensionUninitializedTypePtr,
        _p_args: *const crate::ffi::gdextension::GDExtensionConstTypePtr,
    ) {
        DICT_DEFAULT_CTOR_CALLS.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
    }
    unsafe extern "C" fn mock_recording_dict_destructor(
        _p_base: crate::ffi::gdextension::GDExtensionTypePtr,
    ) {
        DICT_DESTRUCTOR_CALLS.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
    }
    unsafe extern "C" fn mock_recording_dict_op_index(
        _p_self: crate::ffi::gdextension::GDExtensionTypePtr,
        _p_key: GDExtensionConstVariantPtr,
    ) -> GDExtensionVariantPtr {
        DICT_OP_INDEX_CALLS.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        // Return non-NULL so the caller runs `variant_new_copy` on it.
        // SAFETY: static storage; the mock `variant_new_copy` is a
        // recording no-op that does not write past 24 bytes.
        let ptr = &raw mut DICT_OP_INDEX_STORAGE;
        ptr as GDExtensionVariantPtr
    }
    unsafe extern "C" fn mock_recording_variant_new_copy(
        _r_dest: GDExtensionUninitializedVariantPtr,
        _p_src: GDExtensionConstVariantPtr,
    ) {
        VARIANT_NEW_COPY_CALLS.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
    }
    unsafe extern "C" fn mock_recording_variant_destroy(_p_self: GDExtensionVariantPtr) {
        VARIANT_DESTROY_CALLS.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
    }
    unsafe extern "C" fn mock_recording_variant_from_dict_ctor(
        _r_dest: GDExtensionUninitializedVariantPtr,
        _p_in: crate::ffi::gdextension::GDExtensionTypePtr,
    ) {
        VARIANT_FROM_DICT_CTOR_CALLS.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
    }

    /// Build an interface with all six recording mocks wired.
    fn make_dict_recording_interface() -> InterfaceTable {
        let mut iface = crate::ffi::interface::tests::make_sig_aware_interface();
        iface.dict_default_ctor = mock_recording_dict_default_ctor;
        iface.dict_destructor = mock_recording_dict_destructor;
        iface.dictionary_operator_index = mock_recording_dict_op_index;
        iface.variant_new_copy = mock_recording_variant_new_copy;
        iface.variant_destroy = mock_recording_variant_destroy;
        iface.variant_from_dictionary_ctor = mock_recording_variant_from_dict_ctor;
        iface
    }

    fn reset_dict_counters() {
        use core::sync::atomic::Ordering;
        DICT_DEFAULT_CTOR_CALLS.store(0, Ordering::SeqCst);
        DICT_DESTRUCTOR_CALLS.store(0, Ordering::SeqCst);
        DICT_OP_INDEX_CALLS.store(0, Ordering::SeqCst);
        VARIANT_NEW_COPY_CALLS.store(0, Ordering::SeqCst);
        VARIANT_DESTROY_CALLS.store(0, Ordering::SeqCst);
        VARIANT_FROM_DICT_CTOR_CALLS.store(0, Ordering::SeqCst);
    }

    /// Empty PCM still visits every pipeline stage the right number of
    /// times: one default_ctor + one from_dict_ctor + one
    /// dict_destructor + two operator_index (pcm, sample_rate) + two
    /// variant_new_copy (pcm value, sr value) + four variant_destroy
    /// (pcm_variant, sr_variant + two internal key destroys inside
    /// `insert_dict_entry_from_str`).
    #[test]
    fn pack_tts_output_into_dict_variant_empty_pcm_visits_every_stage() {
        use core::sync::atomic::Ordering;
        // The `DICT_*` / `VARIANT_*` call counters are process-global, so
        // this test and its `nonempty_pcm` sibling MUST NOT run
        // concurrently — interleaved runs double every count. Same
        // serialization point the sibling `PACK_*` tests already use.
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let iface = make_dict_recording_interface();
        reset_dict_counters();

        let mut r_dest = [0u8; 24];
        // SAFETY: `r_dest` is a writable 24-byte 8-aligned slot; every
        // fn pointer is a recording no-op that does not deref its slot
        // beyond mock bookkeeping. Empty PCM → skip stage 3's
        // `operator_index` copy path in `pack_f32_slice_into_variant`,
        // but the pipeline still visits `pfa_default_ctor` / `resize`
        // / `pfa_destructor` inside `pack_f32_slice_into_variant`.
        unsafe {
            pack_tts_output_into_dict_variant(&iface, r_dest.as_mut_ptr() as _, &[], 22050);
        }

        assert_eq!(DICT_DEFAULT_CTOR_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(DICT_DESTRUCTOR_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(
            DICT_OP_INDEX_CALLS.load(Ordering::SeqCst),
            2,
            "one operator_index per key (pcm, sample_rate)",
        );
        assert_eq!(
            VARIANT_NEW_COPY_CALLS.load(Ordering::SeqCst),
            2,
            "one new_copy per value (pcm, sample_rate)",
        );
        assert_eq!(
            VARIANT_DESTROY_CALLS.load(Ordering::SeqCst),
            4,
            "two per key sequence (key + value)",
        );
        assert_eq!(VARIANT_FROM_DICT_CTOR_CALLS.load(Ordering::SeqCst), 1);
    }

    /// Non-empty PCM path: same call counts as the empty case (the
    /// PFA-fill logic inside `pack_f32_slice_into_variant` is verified
    /// by the var-B tests above, so we do NOT re-audit it here — this
    /// test locks the dict-level shape only).
    #[test]
    fn pack_tts_output_into_dict_variant_nonempty_pcm_visits_every_stage() {
        use core::sync::atomic::Ordering;
        // See the `empty_pcm` sibling: shared process-global counters.
        let _lock = crate::registry::tests::TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let iface = make_dict_recording_interface();
        reset_dict_counters();

        let pcm: [f32; 4] = [0.1, 0.2, 0.3, 0.4];
        let mut r_dest = [0u8; 24];
        // SAFETY: `r_dest` writable 24-byte slot; recording mocks.
        unsafe {
            pack_tts_output_into_dict_variant(&iface, r_dest.as_mut_ptr() as _, &pcm, 16000);
        }

        assert_eq!(DICT_DEFAULT_CTOR_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(DICT_DESTRUCTOR_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(DICT_OP_INDEX_CALLS.load(Ordering::SeqCst), 2);
        assert_eq!(VARIANT_NEW_COPY_CALLS.load(Ordering::SeqCst), 2);
        assert_eq!(VARIANT_DESTROY_CALLS.load(Ordering::SeqCst), 4);
        assert_eq!(VARIANT_FROM_DICT_CTOR_CALLS.load(Ordering::SeqCst), 1);
    }

    /// Defensive path: if `dictionary_operator_index` ever returns
    /// NULL, the helper skips the `variant_new_copy` call rather than
    /// UB'ing the write. The key Variant is still destroyed (no leak).
    #[test]
    fn pack_tts_output_into_dict_variant_null_slot_skips_new_copy() {
        use core::sync::atomic::Ordering;
        let mut iface = make_dict_recording_interface();
        // Override `dictionary_operator_index` to return NULL.
        unsafe extern "C" fn mock_null_op_index(
            _p_self: crate::ffi::gdextension::GDExtensionTypePtr,
            _p_key: GDExtensionConstVariantPtr,
        ) -> GDExtensionVariantPtr {
            DICT_OP_INDEX_CALLS.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
            core::ptr::null_mut()
        }
        iface.dictionary_operator_index = mock_null_op_index;
        reset_dict_counters();

        let mut r_dest = [0u8; 24];
        // SAFETY: `r_dest` writable 24-byte slot; recording mocks.
        unsafe {
            pack_tts_output_into_dict_variant(&iface, r_dest.as_mut_ptr() as _, &[], 8000);
        }

        assert_eq!(DICT_OP_INDEX_CALLS.load(Ordering::SeqCst), 2);
        assert_eq!(
            VARIANT_NEW_COPY_CALLS.load(Ordering::SeqCst),
            0,
            "NULL slot short-circuits new_copy per FR-EX-08 (no UB write)",
        );
        // Still one destroy per key (2) + one destroy per value (2) = 4.
        assert_eq!(VARIANT_DESTROY_CALLS.load(Ordering::SeqCst), 4);
    }
}
