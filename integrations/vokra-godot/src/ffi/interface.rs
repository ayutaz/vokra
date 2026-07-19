//! GDExtension interface table — the resolved subset of Godot APIs we call.
//!
//! At extension init (`vokra_gdextension_init`) Godot hands us
//! `p_get_proc_address`, an `extern "C"` lookup by name. We use it to resolve
//! every GDExtension interface we depend on (T05: class registration,
//! T06: method binding, T09: signal registration, T10: string-name helpers)
//! and cache the typed function pointers in this table.
//!
//! # Why cache at init?
//!
//! GDExtension is documented as a low-overhead ABI, but every
//! `get_proc_address` call is a lookup in Godot's internal map keyed by
//! `const char*`. Resolving once, at init, guarantees the class-registration
//! and method-binding paths never touch a lookup during runtime dispatch.
//! Godot's own `gdextension.cpp` follows the same pattern.
//!
//! # Storage
//!
//! One-shot init: [`InterfaceTable::from_proc_address`] performs every
//! lookup during `vokra_gdextension_init` and returns a fully-populated
//! table. The table is stored in a `Mutex<Option<...>>` in [`crate::state`]
//! so `vokra_initialize` / `vokra_deinitialize` (called by Godot after the
//! entry point) can find it. The mutex is contested at most once per
//! extension load — we do not touch it from method trampolines (those hold
//! their own resolved pointers, if any).

use core::ffi::c_char;

use crate::ffi::gdextension::{
    DictionaryDefaultCtor, DictionaryDestructor, GDExtensionInterfaceClassdbConstructObject,
    GDExtensionInterfaceClassdbRegisterExtensionClass3,
    GDExtensionInterfaceClassdbRegisterExtensionClassMethod,
    GDExtensionInterfaceClassdbRegisterExtensionClassSignal,
    GDExtensionInterfaceClassdbUnregisterExtensionClass,
    GDExtensionInterfaceDictionaryOperatorIndex, GDExtensionInterfaceGetProcAddress,
    GDExtensionInterfaceGetVariantFromTypeConstructor,
    GDExtensionInterfaceGetVariantToTypeConstructor, GDExtensionInterfaceMemAlloc,
    GDExtensionInterfaceMemFree, GDExtensionInterfaceObjectSetInstance,
    GDExtensionInterfacePackedFloat32ArrayOperatorIndex,
    GDExtensionInterfacePackedFloat32ArrayOperatorIndexConst,
    GDExtensionInterfaceStringNameNewWithLatin1Chars,
    GDExtensionInterfaceStringNameNewWithUtf8Chars,
    GDExtensionInterfaceStringNewWithUtf8CharsAndLen, GDExtensionInterfaceStringToUtf8Chars,
    GDExtensionInterfaceVariantDestroy, GDExtensionInterfaceVariantGetPtrBuiltinMethod,
    GDExtensionInterfaceVariantGetPtrConstructor, GDExtensionInterfaceVariantGetPtrDestructor,
    GDExtensionInterfaceVariantGetType, GDExtensionInterfaceVariantNewCopy,
    GDExtensionInterfaceVariantNewNil, GDExtensionVariantType, PACKED_FLOAT32_ARRAY_RESIZE_HASH,
    PACKED_FLOAT32_ARRAY_SIZE_HASH, PackedFloat32ArrayDefaultCtor, PackedFloat32ArrayDestructor,
    PackedFloat32ArrayResizeMethod, PackedFloat32ArraySizeMethod, StringDestructor,
    VariantFromIntCtor, VariantFromTypeCtor, VariantToIntCtor, VariantToTypeCtor,
};

/// The subset of GDExtension APIs Vokra depends on. Every field is populated
/// by [`Self::from_proc_address`] at extension init.
///
/// The struct is `Copy` because every field is a raw function pointer, but we
/// deliberately do NOT derive `Copy` to prevent accidental value-shadow — the
/// caller must call `.classdb_register_extension_class3()` etc. on the shared
/// reference stored in the state module.
#[derive(Clone)]
pub struct InterfaceTable {
    pub classdb_register_extension_class3: GDExtensionInterfaceClassdbRegisterExtensionClass3,
    pub classdb_register_extension_class_method:
        GDExtensionInterfaceClassdbRegisterExtensionClassMethod,
    pub classdb_register_extension_class_signal:
        GDExtensionInterfaceClassdbRegisterExtensionClassSignal,
    pub classdb_unregister_extension_class: GDExtensionInterfaceClassdbUnregisterExtensionClass,
    /// Object construction pair, required by `create_instance_func`. Godot
    /// `dynamic_cast`s whatever that callback returns as an `Object *`, so
    /// the extension MUST hand back a real Godot Object (built by
    /// `classdb_construct_object` from the *parent* class name) with its
    /// own instance pointer attached via `object_set_instance`.
    pub classdb_construct_object: GDExtensionInterfaceClassdbConstructObject,
    pub object_set_instance: GDExtensionInterfaceObjectSetInstance,
    pub string_name_new_with_utf8_chars: GDExtensionInterfaceStringNameNewWithUtf8Chars,
    pub string_name_new_with_latin1_chars: GDExtensionInterfaceStringNameNewWithLatin1Chars,
    pub mem_alloc: GDExtensionInterfaceMemAlloc,
    pub mem_free: GDExtensionInterfaceMemFree,

    // --- Variant support (T14 M3-11 promotion). ---
    //
    // Introspection.
    pub variant_get_type: GDExtensionInterfaceVariantGetType,
    pub variant_new_nil: GDExtensionInterfaceVariantNewNil,
    // Cached type constructors for Int (the original T14 land — kept as
    // distinct `_int_ctor` fields for backward compatibility with sibling
    // modules `variant.rs` / `trampoline.rs`). The additional fields below
    // (String / PackedFloat32Array / Dictionary / Object, both from + to)
    // are the M3-18 Variant-unpack foundation prerequisite: they enable
    // downstream trampoline promotions listed as `TODO(M3-18)` in
    // `crate::trampoline` without a further init-time resolver walk.
    //
    // The null case for every cached factory result is discharged at
    // resolution time inside [`Self::from_proc_address`] (`?` propagation);
    // any Godot host returning NULL for a fundamental type's constructor
    // fails init cleanly (FR-EX-08 honest failure).
    pub variant_from_int_ctor: VariantFromIntCtor,
    pub variant_to_int_ctor: VariantToIntCtor,

    // --- Variant lifecycle (M3-11 T14/M3-18 unpack foundation). ---
    //
    // Copy/destroy pair for Variant lifetime management on the trampoline
    // hot path (e.g. extending an arg-Variant lifetime by copying it into
    // a local, then destroying after the call). Header names are pinned in
    // `mod names::VARIANT_NEW_COPY` / `VARIANT_DESTROY`.
    pub variant_new_copy: GDExtensionInterfaceVariantNewCopy,
    pub variant_destroy: GDExtensionInterfaceVariantDestroy,

    // --- Additional cached typed constructors (STRING / PACKED_FLOAT32_ARRAY
    //     / DICTIONARY / OBJECT). ---
    //
    // Signature shape is identical to the Int ctors (`VariantFromTypeCtor`
    // / `VariantToTypeCtor` are documented aliases for the shared fn-ptr
    // shape); per-type semantics are pinned by the factory type argument
    // Godot resolves them from. The naming `_from_TYPE_ctor` / `_to_TYPE_ctor`
    // matches the direction in the fn signature: `_from_TYPE_ctor` packs a
    // typed handle INTO a Variant, `_to_TYPE_ctor` unpacks a Variant INTO
    // a typed handle.
    pub variant_from_string_ctor: VariantFromTypeCtor,
    pub variant_to_string_ctor: VariantToTypeCtor,
    pub variant_from_packed_float32_array_ctor: VariantFromTypeCtor,
    pub variant_to_packed_float32_array_ctor: VariantToTypeCtor,
    pub variant_from_dictionary_ctor: VariantFromTypeCtor,
    pub variant_to_dictionary_ctor: VariantToTypeCtor,
    pub variant_from_object_ctor: VariantFromTypeCtor,
    pub variant_to_object_ctor: VariantToTypeCtor,

    // --- Typed String / PackedFloat32Array helpers. ---
    //
    // Direct fn pointers for typed-container access. Each is resolved by
    // its literal header name (see `mod names::*` block). See per-typedef
    // rustdoc in [`crate::ffi::gdextension`] for the call-order contract
    // relative to the constructors above.
    pub packed_float32_array_operator_index_const:
        GDExtensionInterfacePackedFloat32ArrayOperatorIndexConst,
    /// Mutable variant of `packed_float32_array_operator_index` used to
    /// bulk-write into a freshly-resized `PackedFloat32Array` from a Rust
    /// `&[f32]` slice (see [`crate::variant::pack_f32_slice_into_variant`]).
    pub packed_float32_array_operator_index: GDExtensionInterfacePackedFloat32ArrayOperatorIndex,
    pub string_new_with_utf8_chars_and_len: GDExtensionInterfaceStringNewWithUtf8CharsAndLen,
    pub string_to_utf8_chars: GDExtensionInterfaceStringToUtf8Chars,

    // --- PackedFloat32Array packing pipeline (T14-followup: `stream_poll`
    //     full dispatch). ---
    //
    // Raw factory fn pointers. Only used at
    // [`Self::from_proc_address`] time to resolve the three cached PFA
    // handles below; keeping them in the table means a future promotion
    // (e.g. Object / Dictionary packing) can re-use them without a second
    // resolver walk.
    pub variant_get_ptr_constructor: GDExtensionInterfaceVariantGetPtrConstructor,
    pub variant_get_ptr_builtin_method: GDExtensionInterfaceVariantGetPtrBuiltinMethod,
    pub variant_get_ptr_destructor: GDExtensionInterfaceVariantGetPtrDestructor,

    /// Cached default `PackedFloat32Array` constructor (`variant_get_ptr_constructor(PFA, 0)`
    /// → `Option<...>`, Option-unwrapped at resolution). Constructs an empty
    /// PFA into a 16-byte stack buffer.
    pub pfa_default_ctor: PackedFloat32ArrayDefaultCtor,

    /// Cached `PackedFloat32Array::resize(new_size: int) -> int` builtin
    /// method (hash [`PACKED_FLOAT32_ARRAY_RESIZE_HASH`] = 848867239, verified
    /// stable across Godot 4.1..4.3 `extension_api.json`).
    pub pfa_resize_method: PackedFloat32ArrayResizeMethod,

    /// Cached `PackedFloat32Array` destructor (`variant_get_ptr_destructor(PFA)`,
    /// Option-unwrapped at resolution). Called on the stack-allocated temp
    /// buffer after Variant packing to decrement the internal `CowData`
    /// refcount (the Variant now owns the storage).
    pub pfa_destructor: PackedFloat32ArrayDestructor,

    /// Cached `PackedFloat32Array::size() -> int` builtin method (hash
    /// [`PACKED_FLOAT32_ARRAY_SIZE_HASH`] = 3173160232, verified against
    /// Godot 4.3-stable `extension_api.json`). Used by session_transcribe's
    /// PackedFloat32Array unpack path to compute the element count of an
    /// arg PackedFloat32Array before `slice::from_raw_parts`; no dedicated
    /// `packed_float32_array_size` resolver exists in the header (verified
    /// against 4.3-stable — only `operator_index` / `operator_index_const`
    /// variants for every packed array type), so we route through the
    /// generic `variant_get_ptr_builtin_method` factory.
    pub pfa_size_method: PackedFloat32ArraySizeMethod,

    /// Cached `String` destructor (`variant_get_ptr_destructor(STRING)`,
    /// Option-unwrapped at resolution). Called on a typed String opaque
    /// built by [`Self::string_new_with_utf8_chars_and_len`] to release the
    /// CowData refcount after packing into a Variant via
    /// [`Self::variant_from_string_ctor`]. Godot's header exposes no
    /// dedicated `string_destroy` resolver — routed through the generic
    /// `variant_get_ptr_destructor` factory (verified against 4.3-stable
    /// `gdextension_interface.h`; `variant_destroy` operates on Variants,
    /// not raw typed String opaques).
    pub string_destructor: StringDestructor,

    // --- Dictionary packing pipeline (M3-11 T14 followup: `session_synthesize`
    //     full dispatch, dictionary side of the TTS output bag). ---
    //
    // The three cached typed handles below (default ctor / destructor +
    // `dictionary_operator_index`) let `session_synthesize` build the
    // `{"pcm": PackedFloat32Array, "sample_rate": int}` Dictionary Variant
    // at the raw typed level (Type ↔ Variant round-trip only at the
    // Dictionary boundary; the two values stay raw-inserted). Overhead is
    // dominated by the PackedFloat32Array pack pipeline (see the PFA
    // section above), so this is essentially free for the trampoline.
    /// Cached default `Dictionary` constructor (`variant_get_ptr_constructor(DICTIONARY, 0)`
    /// → `Option<...>`, Option-unwrapped at resolution). Constructs an empty
    /// Dictionary into an 8-byte stack buffer (Godot 4.1..4.3 pin
    /// `sizeof(Dictionary) = 8` on LP64 per `extension_api.json`; a 16-byte
    /// stack buffer stays honest across future header widening).
    pub dict_default_ctor: DictionaryDefaultCtor,

    /// Cached `Dictionary` destructor (`variant_get_ptr_destructor(DICTIONARY)`,
    /// Option-unwrapped at resolution). Called on the stack-allocated temp
    /// Dictionary after packing into a Variant to release the internal
    /// refcount.
    pub dict_destructor: DictionaryDestructor,

    /// `dictionary_operator_index` (Godot 4.1, direct API) — takes a raw
    /// Dictionary handle and a Variant key, returns a `*mut Variant`
    /// pointing at the value slot for that key. Godot creates the slot as
    /// Nil if the key did not previously exist; overwriting a non-Nil slot
    /// via `variant_new_copy` would leak the previous Variant. The
    /// [`crate::variant::dict_variant_set_from_str_key`] helper only calls
    /// this for fresh keys, so no cleanup is required at the callsite.
    pub dictionary_operator_index: GDExtensionInterfaceDictionaryOperatorIndex,
}

// The cached `Variant`-from-`PackedFloat32Array` packer is
// [`InterfaceTable::variant_from_packed_float32_array_ctor`] — its
// `VariantFromTypeCtor` alias has an identical fn-pointer signature to
// [`VariantFromPackedFloat32ArrayCtor`], so
// [`crate::variant::pack_f32_slice_into_variant`] reuses it directly.

/// The list of names we resolve. Each entry MUST end with a NUL byte
/// (`get_proc_address` takes a C string). Every string constant here matches
/// a `#define`d resolver key in Godot 4.3-stable
/// `core/extension/gdextension_interface.h`.
mod names {
    pub const CLASSDB_REGISTER_EXTENSION_CLASS3: &[u8] = b"classdb_register_extension_class3\0";
    pub const CLASSDB_REGISTER_EXTENSION_CLASS_METHOD: &[u8] =
        b"classdb_register_extension_class_method\0";
    pub const CLASSDB_REGISTER_EXTENSION_CLASS_SIGNAL: &[u8] =
        b"classdb_register_extension_class_signal\0";
    pub const CLASSDB_UNREGISTER_EXTENSION_CLASS: &[u8] = b"classdb_unregister_extension_class\0";
    /// `classdb_construct_object` — header line 2692 (@since 4.1).
    pub const CLASSDB_CONSTRUCT_OBJECT: &[u8] = b"classdb_construct_object\0";
    /// `object_set_instance` — header line 2440 (@since 4.1).
    pub const OBJECT_SET_INSTANCE: &[u8] = b"object_set_instance\0";
    pub const STRING_NAME_NEW_WITH_UTF8_CHARS: &[u8] = b"string_name_new_with_utf8_chars\0";
    pub const STRING_NAME_NEW_WITH_LATIN1_CHARS: &[u8] = b"string_name_new_with_latin1_chars\0";
    pub const MEM_ALLOC: &[u8] = b"mem_alloc\0";
    pub const MEM_FREE: &[u8] = b"mem_free\0";

    // Variant support (T14 M3-11 promotion). All four names are declared in
    // Godot 4.3-stable `gdextension_interface.h` next to the corresponding
    // `GDExtensionInterface*` typedefs and are the exact keys the header
    // documents for `get_proc_address(...)`.
    pub const VARIANT_GET_TYPE: &[u8] = b"variant_get_type\0";
    pub const VARIANT_NEW_NIL: &[u8] = b"variant_new_nil\0";
    pub const GET_VARIANT_FROM_TYPE_CONSTRUCTOR: &[u8] = b"get_variant_from_type_constructor\0";
    pub const GET_VARIANT_TO_TYPE_CONSTRUCTOR: &[u8] = b"get_variant_to_type_constructor\0";

    // M3-11 T14/M3-18 unpack foundation additions. Every name below is a
    // literal `#define`d resolver key in Godot 4.3-stable
    // `core/extension/gdextension_interface.h`, verified against the
    // header blob (SHA fetched into the scratchpad and grep'd at
    // implementation time). Line numbers are the typedef locations in the
    // header for auditability.
    /// `variant_new_copy` — header line 912 (@since 4.1).
    pub const VARIANT_NEW_COPY: &[u8] = b"variant_new_copy\0";
    /// `variant_destroy` — header line 932 (@since 4.1).
    pub const VARIANT_DESTROY: &[u8] = b"variant_destroy\0";
    /// `packed_float32_array_operator_index_const` — header line 2052
    /// (@since 4.1).
    pub const PACKED_FLOAT32_ARRAY_OPERATOR_INDEX_CONST: &[u8] =
        b"packed_float32_array_operator_index_const\0";
    /// `packed_float32_array_operator_index` — mutable variant (`@since 4.1`,
    /// header line 2043). Used by `stream_poll` to bulk-write payload into a
    /// freshly-resized PackedFloat32Array before packing into the return
    /// Variant.
    pub const PACKED_FLOAT32_ARRAY_OPERATOR_INDEX: &[u8] = b"packed_float32_array_operator_index\0";
    /// `variant_get_ptr_constructor` (`@since 4.1`). Factory for per-type
    /// in-place constructors — used to resolve the default PackedFloat32Array
    /// constructor (index 0) at init.
    pub const VARIANT_GET_PTR_CONSTRUCTOR: &[u8] = b"variant_get_ptr_constructor\0";
    /// `variant_get_ptr_builtin_method` (`@since 4.1`). Factory for per-type
    /// builtin methods — used to resolve `PackedFloat32Array::resize` at
    /// init, guarded by [`super::PACKED_FLOAT32_ARRAY_RESIZE_HASH`] against
    /// silent signature drift.
    pub const VARIANT_GET_PTR_BUILTIN_METHOD: &[u8] = b"variant_get_ptr_builtin_method\0";
    /// `variant_get_ptr_destructor` (`@since 4.1`). Factory for per-type
    /// destructors — used to resolve the PackedFloat32Array destructor at
    /// init, called on the temp buffer after Variant packing.
    pub const VARIANT_GET_PTR_DESTRUCTOR: &[u8] = b"variant_get_ptr_destructor\0";
    /// `string_new_with_utf8_chars_and_len` — header line 1593 (@since 4.1,
    /// `@deprecated in 4.3`). We pin the 4.1 shape because
    /// `vokra.gdextension`'s `compatibility_minimum = "4.1"` — see the
    /// per-typedef rustdoc in [`crate::ffi::gdextension`] for the
    /// promotion path to `..._and_len2` when we bump the pin.
    pub const STRING_NEW_WITH_UTF8_CHARS_AND_LEN: &[u8] = b"string_new_with_utf8_chars_and_len\0";
    /// `string_to_utf8_chars` — header line 1691 (@since 4.1).
    pub const STRING_TO_UTF8_CHARS: &[u8] = b"string_to_utf8_chars\0";

    // Dictionary packing pipeline (M3-11 T14 followup: session_synthesize
    // full dispatch). `dictionary_operator_index` is the only new named
    // resolver; the Dictionary default constructor + destructor are
    // resolved through the already-cached factories
    // `variant_get_ptr_constructor` and `variant_get_ptr_destructor`.
    /// `dictionary_operator_index` — Godot 4.1, direct API (verified in the
    /// 4.3-stable `gdextension_interface.h`).
    pub const DICTIONARY_OPERATOR_INDEX: &[u8] = b"dictionary_operator_index\0";
}

impl InterfaceTable {
    /// Resolve every interface function via `get_proc_address` and pack them
    /// into a table. Returns `None` if ANY required interface is missing;
    /// Godot documents that its own `get_proc_address` returns NULL for
    /// unknown names, so this is the honest failure mode.
    ///
    /// # Safety
    ///
    /// - `get_proc_address` must be a live function pointer for the duration
    ///   of the call, matching the `GDExtensionInterfaceGetProcAddress`
    ///   signature (checked by the entry point's static-cast test).
    /// - The returned table's function pointers are only valid for the
    ///   lifetime of the loaded Godot process. Godot never rebinds them, and
    ///   they become stale only after the cdylib is unloaded.
    pub unsafe fn from_proc_address(
        get_proc_address: GDExtensionInterfaceGetProcAddress,
    ) -> Option<Self> {
        // SAFETY: `get_proc_address` is a live fn pointer (see doc above).
        // Each name byte-slice is NUL-terminated by construction. The
        // returned `GDExtensionInterfaceFunctionPtr` is either NULL (miss)
        // or a live fn pointer with the documented Godot 4.3 signature.
        unsafe {
            let raw_class3 = get_proc_address(
                names::CLASSDB_REGISTER_EXTENSION_CLASS3.as_ptr() as *const c_char
            )?;
            let raw_method = get_proc_address(
                names::CLASSDB_REGISTER_EXTENSION_CLASS_METHOD.as_ptr() as *const c_char,
            )?;
            let raw_signal = get_proc_address(
                names::CLASSDB_REGISTER_EXTENSION_CLASS_SIGNAL.as_ptr() as *const c_char,
            )?;
            let raw_unreg = get_proc_address(
                names::CLASSDB_UNREGISTER_EXTENSION_CLASS.as_ptr() as *const c_char
            )?;
            // Object construction pair — see the field docs on
            // `InterfaceTable::classdb_construct_object`. A miss on either
            // means `create_instance_func` could never return a valid
            // Object, so bail rather than register a class that would
            // crash the host on `.new()`.
            let raw_construct_object =
                get_proc_address(names::CLASSDB_CONSTRUCT_OBJECT.as_ptr() as *const c_char)?;
            let raw_object_set_instance =
                get_proc_address(names::OBJECT_SET_INSTANCE.as_ptr() as *const c_char)?;
            let raw_sn_utf8 =
                get_proc_address(names::STRING_NAME_NEW_WITH_UTF8_CHARS.as_ptr() as *const c_char)?;
            let raw_sn_latin1 = get_proc_address(
                names::STRING_NAME_NEW_WITH_LATIN1_CHARS.as_ptr() as *const c_char,
            )?;
            let raw_alloc = get_proc_address(names::MEM_ALLOC.as_ptr() as *const c_char)?;
            let raw_free = get_proc_address(names::MEM_FREE.as_ptr() as *const c_char)?;

            // Variant support APIs (T14 M3-11 promotion). All four are
            // documented in Godot 4.3-stable `gdextension_interface.h` and
            // MUST resolve for a compatible Godot host; a miss on any is a
            // signal that the extension is being loaded by an
            // incompatible Godot version → bail with `None`.
            let raw_variant_get_type =
                get_proc_address(names::VARIANT_GET_TYPE.as_ptr() as *const c_char)?;
            let raw_variant_new_nil =
                get_proc_address(names::VARIANT_NEW_NIL.as_ptr() as *const c_char)?;
            let raw_get_from_ctor = get_proc_address(
                names::GET_VARIANT_FROM_TYPE_CONSTRUCTOR.as_ptr() as *const c_char,
            )?;
            let raw_get_to_ctor =
                get_proc_address(names::GET_VARIANT_TO_TYPE_CONSTRUCTOR.as_ptr() as *const c_char)?;

            // Transmute each opaque fn pointer to its typed signature.
            // Layout guarantee: `Option<unsafe extern "C" fn()>` is
            // `#[repr(transparent)]` over a raw fn pointer, so a NULL
            // Option decodes to a NULL fn pointer. `?` above already
            // discharged the NULL case.
            let variant_get_type: GDExtensionInterfaceVariantGetType =
                core::mem::transmute::<unsafe extern "C" fn(), GDExtensionInterfaceVariantGetType>(
                    raw_variant_get_type,
                );
            let variant_new_nil: GDExtensionInterfaceVariantNewNil =
                core::mem::transmute::<unsafe extern "C" fn(), GDExtensionInterfaceVariantNewNil>(
                    raw_variant_new_nil,
                );
            let get_variant_from_type_constructor: GDExtensionInterfaceGetVariantFromTypeConstructor =
                core::mem::transmute::<
                    unsafe extern "C" fn(),
                    GDExtensionInterfaceGetVariantFromTypeConstructor,
                >(raw_get_from_ctor);
            let get_variant_to_type_constructor: GDExtensionInterfaceGetVariantToTypeConstructor =
                core::mem::transmute::<
                    unsafe extern "C" fn(),
                    GDExtensionInterfaceGetVariantToTypeConstructor,
                >(raw_get_to_ctor);

            // M3-11 T14/M3-18 unpack foundation: resolve the additional
            // named APIs required by Variant lifecycle + typed String /
            // PackedFloat32Array unpack. Every miss on any of these
            // signals an incompatible Godot host → bail with `None`
            // (FR-EX-08 honest failure). Names + line numbers are pinned
            // in `mod names::*` (see per-const rustdoc).
            let raw_variant_new_copy =
                get_proc_address(names::VARIANT_NEW_COPY.as_ptr() as *const c_char)?;
            let raw_variant_destroy =
                get_proc_address(names::VARIANT_DESTROY.as_ptr() as *const c_char)?;
            let raw_packed_float32_array_operator_index_const = get_proc_address(
                names::PACKED_FLOAT32_ARRAY_OPERATOR_INDEX_CONST.as_ptr() as *const c_char,
            )?;
            let raw_string_new_with_utf8_chars_and_len = get_proc_address(
                names::STRING_NEW_WITH_UTF8_CHARS_AND_LEN.as_ptr() as *const c_char,
            )?;
            let raw_string_to_utf8_chars =
                get_proc_address(names::STRING_TO_UTF8_CHARS.as_ptr() as *const c_char)?;

            // T14-followup (`stream_poll` full dispatch): resolve the four
            // named APIs required to pack a Rust `&[f32]` into a
            // PackedFloat32Array Variant. A miss on any is treated as a hard
            // init failure (FR-EX-08) — the extension refuses to load rather
            // than binding partial functionality.
            let raw_packed_float32_array_operator_index = get_proc_address(
                names::PACKED_FLOAT32_ARRAY_OPERATOR_INDEX.as_ptr() as *const c_char,
            )?;
            let raw_variant_get_ptr_constructor =
                get_proc_address(names::VARIANT_GET_PTR_CONSTRUCTOR.as_ptr() as *const c_char)?;
            let raw_variant_get_ptr_builtin_method =
                get_proc_address(names::VARIANT_GET_PTR_BUILTIN_METHOD.as_ptr() as *const c_char)?;
            let raw_variant_get_ptr_destructor =
                get_proc_address(names::VARIANT_GET_PTR_DESTRUCTOR.as_ptr() as *const c_char)?;

            // M3-11 T14-followup Dictionary side (`session_synthesize` full
            // dispatch). `dictionary_operator_index` is the only new named
            // resolver; the default ctor + destructor for Dictionary come
            // through the already-resolved `variant_get_ptr_constructor`
            // + `variant_get_ptr_destructor` factories below.
            let raw_dictionary_operator_index =
                get_proc_address(names::DICTIONARY_OPERATOR_INDEX.as_ptr() as *const c_char)?;

            // Resolve the Int packer / unpacker. Godot documents these
            // factories return NULL for unknown types; Int is a
            // fundamental type so a NULL here would indicate a corrupt
            // host — bail. `?` unwraps the `Option<fn>` produced by the
            // factory into the non-Option alias types
            // `VariantFromIntCtor` / `VariantToIntCtor`.
            let variant_from_int_ctor: VariantFromIntCtor =
                get_variant_from_type_constructor(GDExtensionVariantType::Int)?;
            let variant_to_int_ctor: VariantToIntCtor =
                get_variant_to_type_constructor(GDExtensionVariantType::Int)?;

            // M3-11 T14/M3-18 unpack foundation: cache the packer /
            // unpacker for every additional fundamental Variant type the
            // trampoline surface will need (STRING / PACKED_FLOAT32_ARRAY
            // / DICTIONARY / OBJECT). Each factory result is
            // `?`-propagated because — as with Int — a NULL for any of
            // these fundamental types would indicate a corrupt Godot
            // host. Signature shape is universal
            // (`VariantFromTypeCtor` / `VariantToTypeCtor`); per-type
            // semantics are pinned by the factory type argument.
            let variant_from_string_ctor: VariantFromTypeCtor =
                get_variant_from_type_constructor(GDExtensionVariantType::String)?;
            let variant_to_string_ctor: VariantToTypeCtor =
                get_variant_to_type_constructor(GDExtensionVariantType::String)?;
            let variant_from_packed_float32_array_ctor: VariantFromTypeCtor =
                get_variant_from_type_constructor(GDExtensionVariantType::PackedFloat32Array)?;
            let variant_to_packed_float32_array_ctor: VariantToTypeCtor =
                get_variant_to_type_constructor(GDExtensionVariantType::PackedFloat32Array)?;
            let variant_from_dictionary_ctor: VariantFromTypeCtor =
                get_variant_from_type_constructor(GDExtensionVariantType::Dictionary)?;
            let variant_to_dictionary_ctor: VariantToTypeCtor =
                get_variant_to_type_constructor(GDExtensionVariantType::Dictionary)?;
            let variant_from_object_ctor: VariantFromTypeCtor =
                get_variant_from_type_constructor(GDExtensionVariantType::Object)?;
            let variant_to_object_ctor: VariantToTypeCtor =
                get_variant_to_type_constructor(GDExtensionVariantType::Object)?;

            // Transmute the new named-API opaque fn pointers to their
            // typed signatures. Same one-word round-trip rationale as the
            // legacy resolvers above.
            let variant_new_copy: GDExtensionInterfaceVariantNewCopy =
                core::mem::transmute::<unsafe extern "C" fn(), GDExtensionInterfaceVariantNewCopy>(
                    raw_variant_new_copy,
                );
            let variant_destroy: GDExtensionInterfaceVariantDestroy =
                core::mem::transmute::<unsafe extern "C" fn(), GDExtensionInterfaceVariantDestroy>(
                    raw_variant_destroy,
                );
            let packed_float32_array_operator_index_const:
                GDExtensionInterfacePackedFloat32ArrayOperatorIndexConst = core::mem::transmute::<
                unsafe extern "C" fn(),
                GDExtensionInterfacePackedFloat32ArrayOperatorIndexConst,
            >(
                raw_packed_float32_array_operator_index_const,
            );
            let string_new_with_utf8_chars_and_len:
                GDExtensionInterfaceStringNewWithUtf8CharsAndLen = core::mem::transmute::<
                unsafe extern "C" fn(),
                GDExtensionInterfaceStringNewWithUtf8CharsAndLen,
            >(
                raw_string_new_with_utf8_chars_and_len,
            );
            let string_to_utf8_chars: GDExtensionInterfaceStringToUtf8Chars = core::mem::transmute::<
                unsafe extern "C" fn(),
                GDExtensionInterfaceStringToUtf8Chars,
            >(
                raw_string_to_utf8_chars,
            );

            // T14-followup PackedFloat32Array packing pipeline. Transmute
            // the four new raw fn pointers to their typed signatures.
            let packed_float32_array_operator_index:
                GDExtensionInterfacePackedFloat32ArrayOperatorIndex = core::mem::transmute::<
                unsafe extern "C" fn(),
                GDExtensionInterfacePackedFloat32ArrayOperatorIndex,
            >(
                raw_packed_float32_array_operator_index,
            );
            let variant_get_ptr_constructor: GDExtensionInterfaceVariantGetPtrConstructor =
                core::mem::transmute::<
                    unsafe extern "C" fn(),
                    GDExtensionInterfaceVariantGetPtrConstructor,
                >(raw_variant_get_ptr_constructor);
            let variant_get_ptr_builtin_method: GDExtensionInterfaceVariantGetPtrBuiltinMethod =
                core::mem::transmute::<
                    unsafe extern "C" fn(),
                    GDExtensionInterfaceVariantGetPtrBuiltinMethod,
                >(raw_variant_get_ptr_builtin_method);
            let variant_get_ptr_destructor: GDExtensionInterfaceVariantGetPtrDestructor =
                core::mem::transmute::<
                    unsafe extern "C" fn(),
                    GDExtensionInterfaceVariantGetPtrDestructor,
                >(raw_variant_get_ptr_destructor);

            // Resolve the default PackedFloat32Array constructor (index 0,
            // zero-arg). Godot returns NULL for an unknown type/index — a
            // NULL for PackedFloat32Array/0 would indicate a corrupt host,
            // so `?` bails cleanly.
            let pfa_default_ctor: PackedFloat32ArrayDefaultCtor =
                variant_get_ptr_constructor(GDExtensionVariantType::PackedFloat32Array, 0)?;

            // Resolve the PackedFloat32Array destructor. Same NULL semantics
            // as the constructor factory.
            let pfa_destructor: PackedFloat32ArrayDestructor =
                variant_get_ptr_destructor(GDExtensionVariantType::PackedFloat32Array)?;

            // Resolve `PackedFloat32Array::resize(new_size: int) -> int`.
            // Godot requires a StringName for the method name plus the exact
            // method hash. We build the StringName on a stack buffer via the
            // just-resolved `string_name_new_with_utf8_chars` — the
            // StringName does not need to outlive this call because Godot
            // resolves the method pointer eagerly (the returned fn pointer
            // is process-lifetime).
            //
            // The hash [`PACKED_FLOAT32_ARRAY_RESIZE_HASH`] = 848867239 is
            // verified stable across `godot-cpp/gdextension/extension_api.json`
            // at tags `godot-4.1-stable`, `godot-4.2-stable`, and
            // `godot-4.3-stable` — see the per-const rustdoc in
            // [`crate::ffi::gdextension`]. A drift on any of those tags
            // would surface here as a NULL return (bail cleanly).
            #[repr(C, align(8))]
            struct StringNameBuf([u8; 16]);
            let mut sn = StringNameBuf([0u8; 16]);
            let string_name_new_with_utf8_chars: GDExtensionInterfaceStringNameNewWithUtf8Chars =
                core::mem::transmute::<
                    unsafe extern "C" fn(),
                    GDExtensionInterfaceStringNameNewWithUtf8Chars,
                >(raw_sn_utf8);
            let resize_name_bytes: &[u8] = b"resize\0";
            string_name_new_with_utf8_chars(
                sn.0.as_mut_ptr() as *mut core::ffi::c_void,
                resize_name_bytes.as_ptr() as *const c_char,
            );
            let pfa_resize_method: PackedFloat32ArrayResizeMethod = variant_get_ptr_builtin_method(
                GDExtensionVariantType::PackedFloat32Array,
                sn.0.as_ptr() as *const core::ffi::c_void,
                PACKED_FLOAT32_ARRAY_RESIZE_HASH,
            )?;

            // Resolve `PackedFloat32Array::size() -> int` (const, zero-arg).
            // Same StringName-plus-hash resolver as `resize` above; reuse the
            // same stack buffer (Godot's `string_name_new_with_utf8_chars`
            // overwrites the destination each call, so a second init on `sn`
            // is safe as long as we do not keep the previous `size` alive —
            // we don't, because `variant_get_ptr_builtin_method` resolves
            // eagerly and returns a process-lifetime fn pointer). The hash
            // [`PACKED_FLOAT32_ARRAY_SIZE_HASH`] = 3173160232 was verified
            // against Godot 4.3-stable `extension_api.json`; a drift on a
            // future tag would surface as a NULL return here (bail cleanly).
            let size_name_bytes: &[u8] = b"size\0";
            string_name_new_with_utf8_chars(
                sn.0.as_mut_ptr() as *mut core::ffi::c_void,
                size_name_bytes.as_ptr() as *const c_char,
            );
            let pfa_size_method: PackedFloat32ArraySizeMethod = variant_get_ptr_builtin_method(
                GDExtensionVariantType::PackedFloat32Array,
                sn.0.as_ptr() as *const core::ffi::c_void,
                PACKED_FLOAT32_ARRAY_SIZE_HASH,
            )?;

            // Resolve the `String` destructor. Called on the temp typed
            // String produced by `string_new_with_utf8_chars_and_len` after
            // the Variant packer copies its CowData refcount into the return
            // Variant. Godot's header has no `string_destroy` resolver
            // (verified against 4.3-stable — only `variant_destroy` exists,
            // which operates on Variants, not raw typed String opaques), so
            // we route through the per-type destructor factory.
            let string_destructor: StringDestructor =
                variant_get_ptr_destructor(GDExtensionVariantType::String)?;

            // M3-11 T14-followup: cache the default Dictionary constructor
            // (index 0, empty dict) and destructor for the
            // `session_synthesize` pack path. `?` propagation surfaces a
            // corrupt Godot host (Dictionary is a fundamental type — a NULL
            // here indicates the running host is not a real Godot binary)
            // as an init failure per FR-EX-08.
            let dict_default_ctor: DictionaryDefaultCtor =
                variant_get_ptr_constructor(GDExtensionVariantType::Dictionary, 0)?;
            let dict_destructor: DictionaryDestructor =
                variant_get_ptr_destructor(GDExtensionVariantType::Dictionary)?;

            // Transmute the direct `dictionary_operator_index` opaque
            // resolver into its typed shape. Same one-word round-trip
            // rationale as the other typed transmutes above.
            let dictionary_operator_index: GDExtensionInterfaceDictionaryOperatorIndex =
                core::mem::transmute::<
                    unsafe extern "C" fn(),
                    GDExtensionInterfaceDictionaryOperatorIndex,
                >(raw_dictionary_operator_index);

            Some(Self {
                classdb_register_extension_class3: core::mem::transmute::<
                    unsafe extern "C" fn(),
                    GDExtensionInterfaceClassdbRegisterExtensionClass3,
                >(raw_class3),
                classdb_register_extension_class_method: core::mem::transmute::<
                    unsafe extern "C" fn(),
                    GDExtensionInterfaceClassdbRegisterExtensionClassMethod,
                >(raw_method),
                classdb_register_extension_class_signal: core::mem::transmute::<
                    unsafe extern "C" fn(),
                    GDExtensionInterfaceClassdbRegisterExtensionClassSignal,
                >(raw_signal),
                classdb_unregister_extension_class: core::mem::transmute::<
                    unsafe extern "C" fn(),
                    GDExtensionInterfaceClassdbUnregisterExtensionClass,
                >(raw_unreg),
                classdb_construct_object: core::mem::transmute::<
                    unsafe extern "C" fn(),
                    GDExtensionInterfaceClassdbConstructObject,
                >(raw_construct_object),
                object_set_instance: core::mem::transmute::<
                    unsafe extern "C" fn(),
                    GDExtensionInterfaceObjectSetInstance,
                >(raw_object_set_instance),
                string_name_new_with_utf8_chars: core::mem::transmute::<
                    unsafe extern "C" fn(),
                    GDExtensionInterfaceStringNameNewWithUtf8Chars,
                >(raw_sn_utf8),
                string_name_new_with_latin1_chars: core::mem::transmute::<
                    unsafe extern "C" fn(),
                    GDExtensionInterfaceStringNameNewWithLatin1Chars,
                >(raw_sn_latin1),
                mem_alloc: core::mem::transmute::<
                    unsafe extern "C" fn(),
                    GDExtensionInterfaceMemAlloc,
                >(raw_alloc),
                mem_free: core::mem::transmute::<unsafe extern "C" fn(), GDExtensionInterfaceMemFree>(
                    raw_free,
                ),
                variant_get_type,
                variant_new_nil,
                variant_from_int_ctor,
                variant_to_int_ctor,
                variant_new_copy,
                variant_destroy,
                variant_from_string_ctor,
                variant_to_string_ctor,
                variant_from_packed_float32_array_ctor,
                variant_to_packed_float32_array_ctor,
                variant_from_dictionary_ctor,
                variant_to_dictionary_ctor,
                variant_from_object_ctor,
                variant_to_object_ctor,
                packed_float32_array_operator_index_const,
                packed_float32_array_operator_index,
                string_new_with_utf8_chars_and_len,
                string_to_utf8_chars,
                variant_get_ptr_constructor,
                variant_get_ptr_builtin_method,
                variant_get_ptr_destructor,
                pfa_default_ctor,
                pfa_resize_method,
                pfa_destructor,
                pfa_size_method,
                string_destructor,
                dict_default_ctor,
                dict_destructor,
                dictionary_operator_index,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Test-suite mocks.
//
// Two mock realities live here:
//
//   1. Plain-sentinel mock: every resolver name returns the SAME opaque fn
//      pointer. Fine for exercising the `?` propagation of missing
//      resolvers and the "every-field-populated" pass. NOT safe when
//      `from_proc_address` actually CALLS a resolved fn pointer (as it
//      now does for `get_variant_{from,to}_type_constructor` factories) —
//      calling a fn(A, B) as fn() is UB.
//
//   2. Sig-aware mock (`mock_gpa`): for names whose returned fn pointer
//      is ACTUALLY invoked by `from_proc_address`, dispatch to a mock fn
//      with the correct signature. Round-trips through
//      `Option<unsafe extern "C" fn()>` are ABI-safe because Rust fn
//      pointers are all one word; the caller re-transmutes to the true
//      signature before calling.
// ---------------------------------------------------------------------------
#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Verify every resolver name is NUL-terminated. Godot's `get_proc_address`
    /// takes `const char*`; a missing NUL would read past the buffer end
    /// (Undefined Behaviour in the host). The T05 registration path is
    /// entirely built on these names, so a silent typo would corrupt Godot.
    #[test]
    fn every_resolver_name_is_nul_terminated() {
        for (label, buf) in [
            (
                "classdb_register_extension_class3",
                names::CLASSDB_REGISTER_EXTENSION_CLASS3,
            ),
            (
                "classdb_register_extension_class_method",
                names::CLASSDB_REGISTER_EXTENSION_CLASS_METHOD,
            ),
            (
                "classdb_register_extension_class_signal",
                names::CLASSDB_REGISTER_EXTENSION_CLASS_SIGNAL,
            ),
            (
                "classdb_unregister_extension_class",
                names::CLASSDB_UNREGISTER_EXTENSION_CLASS,
            ),
            (
                "string_name_new_with_utf8_chars",
                names::STRING_NAME_NEW_WITH_UTF8_CHARS,
            ),
            (
                "string_name_new_with_latin1_chars",
                names::STRING_NAME_NEW_WITH_LATIN1_CHARS,
            ),
            ("mem_alloc", names::MEM_ALLOC),
            ("mem_free", names::MEM_FREE),
            ("variant_get_type", names::VARIANT_GET_TYPE),
            ("variant_new_nil", names::VARIANT_NEW_NIL),
            (
                "get_variant_from_type_constructor",
                names::GET_VARIANT_FROM_TYPE_CONSTRUCTOR,
            ),
            (
                "get_variant_to_type_constructor",
                names::GET_VARIANT_TO_TYPE_CONSTRUCTOR,
            ),
            // M3-11 T14/M3-18 unpack foundation.
            ("variant_new_copy", names::VARIANT_NEW_COPY),
            ("variant_destroy", names::VARIANT_DESTROY),
            (
                "packed_float32_array_operator_index_const",
                names::PACKED_FLOAT32_ARRAY_OPERATOR_INDEX_CONST,
            ),
            (
                "string_new_with_utf8_chars_and_len",
                names::STRING_NEW_WITH_UTF8_CHARS_AND_LEN,
            ),
            ("string_to_utf8_chars", names::STRING_TO_UTF8_CHARS),
            // M3-11 T14-followup PackedFloat32Array / Dictionary pipeline.
            (
                "packed_float32_array_operator_index",
                names::PACKED_FLOAT32_ARRAY_OPERATOR_INDEX,
            ),
            (
                "variant_get_ptr_constructor",
                names::VARIANT_GET_PTR_CONSTRUCTOR,
            ),
            (
                "variant_get_ptr_builtin_method",
                names::VARIANT_GET_PTR_BUILTIN_METHOD,
            ),
            (
                "variant_get_ptr_destructor",
                names::VARIANT_GET_PTR_DESTRUCTOR,
            ),
            (
                "dictionary_operator_index",
                names::DICTIONARY_OPERATOR_INDEX,
            ),
        ] {
            assert!(
                buf.last() == Some(&0u8),
                "resolver name {label:?} is missing terminal NUL"
            );
            // And exactly one NUL, at the end.
            let nul_count = buf.iter().filter(|&&b| b == 0).count();
            assert_eq!(nul_count, 1, "resolver name {label:?} has interior NUL");
        }
    }

    // ------------------------------------------------------------------
    // Sig-aware mocks — expose to sibling test modules through
    // `pub(crate)` so `trampoline` unit tests can build a valid mock
    // interface without duplicating this plumbing. All mocks are
    // no-ops from Godot's perspective.
    // ------------------------------------------------------------------

    pub(crate) unsafe extern "C" fn sentinel() {}

    pub(crate) unsafe extern "C" fn mock_variant_get_type(
        _v: crate::ffi::gdextension::GDExtensionConstVariantPtr,
    ) -> GDExtensionVariantType {
        // Return Nil by default; sig-aware trampoline tests override the
        // whole `variant_get_type` field with a per-test mock that
        // returns the type they want to exercise (see
        // `trampoline::tests::with_variant_type`).
        GDExtensionVariantType::Nil
    }

    pub(crate) unsafe extern "C" fn mock_variant_new_nil(
        _r: crate::ffi::gdextension::GDExtensionUninitializedVariantPtr,
    ) {
    }

    pub(crate) unsafe extern "C" fn mock_variant_from_int(
        _r: crate::ffi::gdextension::GDExtensionUninitializedVariantPtr,
        _p: crate::ffi::gdextension::GDExtensionTypePtr,
    ) {
    }

    pub(crate) unsafe extern "C" fn mock_variant_to_int(
        _r: crate::ffi::gdextension::GDExtensionUninitializedTypePtr,
        _p: crate::ffi::gdextension::GDExtensionVariantPtr,
    ) {
    }

    pub(crate) unsafe extern "C" fn mock_get_variant_from_type_ctor(
        _ty: GDExtensionVariantType,
    ) -> crate::ffi::gdextension::GDExtensionVariantFromTypeConstructorFunc {
        Some(mock_variant_from_int)
    }

    pub(crate) unsafe extern "C" fn mock_get_variant_to_type_ctor(
        _ty: GDExtensionVariantType,
    ) -> crate::ffi::gdextension::GDExtensionTypeFromVariantConstructorFunc {
        Some(mock_variant_to_int)
    }

    // ------------------------------------------------------------------
    // M3-11 T14-followup mocks: `from_proc_address` calls the following
    // resolved fn pointers during init (StringName construction + factory
    // dispatch), so their sig-aware bucket MUST return typed no-ops with
    // the correct signature. Providing `sentinel` would be UB the moment
    // the resolver casts to a typed factory + invokes it.
    // ------------------------------------------------------------------

    /// Mock `string_name_new_with_utf8_chars` — a no-op that ignores its
    /// output slot. `from_proc_address` uses this to build "resize" +
    /// "size" StringNames on a stack buffer before feeding them to the
    /// builtin_method factory. The factory ignores the actual bytes in
    /// our mock (see `mock_variant_get_ptr_builtin_method`), so leaving
    /// the buffer untouched is safe.
    pub(crate) unsafe extern "C" fn mock_string_name_new_with_utf8_chars(
        _r_dest: crate::ffi::gdextension::GDExtensionUninitializedStringNamePtr,
        _p_contents: *const c_char,
    ) {
    }

    /// Mock per-type in-place constructor — no-op. Cast to
    /// `PackedFloat32ArrayDefaultCtor` and stored; the trampoline test
    /// path never invokes it (real Godot would default-construct into
    /// the 16-byte stack buffer).
    pub(crate) unsafe extern "C" fn mock_ptr_constructor(
        _p_base: crate::ffi::gdextension::GDExtensionUninitializedTypePtr,
        _p_args: *const crate::ffi::gdextension::GDExtensionConstTypePtr,
    ) {
    }

    /// Mock per-type builtin method — no-op. Cast to
    /// `PackedFloat32ArrayResizeMethod` or `PackedFloat32ArraySizeMethod`
    /// and stored; the trampoline test path overrides these in a
    /// dedicated per-test wire-up when it needs a canned size.
    pub(crate) unsafe extern "C" fn mock_ptr_builtin_method(
        _p_base: crate::ffi::gdextension::GDExtensionTypePtr,
        _p_args: *const crate::ffi::gdextension::GDExtensionConstTypePtr,
        _r_return: crate::ffi::gdextension::GDExtensionTypePtr,
        _p_argument_count: i32,
    ) {
    }

    /// Mock per-type destructor — no-op. Cast to
    /// `PackedFloat32ArrayDestructor` / `StringDestructor` and stored.
    pub(crate) unsafe extern "C" fn mock_ptr_destructor(
        _p_base: crate::ffi::gdextension::GDExtensionTypePtr,
    ) {
    }

    /// Mock `variant_get_ptr_constructor` factory. Returns a typed no-op
    /// constructor regardless of the requested type or index. Signature
    /// match with `GDExtensionInterfaceVariantGetPtrConstructor` is
    /// asserted by the compile-only cast at the callsite of
    /// `from_proc_address`.
    pub(crate) unsafe extern "C" fn mock_variant_get_ptr_constructor(
        _ty: GDExtensionVariantType,
        _p_constructor: i32,
    ) -> Option<crate::ffi::gdextension::GDExtensionPtrConstructor> {
        Some(mock_ptr_constructor)
    }

    /// Mock `variant_get_ptr_builtin_method` factory. Returns a typed
    /// no-op method regardless of type / method / hash.
    pub(crate) unsafe extern "C" fn mock_variant_get_ptr_builtin_method(
        _ty: GDExtensionVariantType,
        _p_method: crate::ffi::gdextension::GDExtensionConstStringNamePtr,
        _p_hash: crate::ffi::gdextension::GDExtensionInt,
    ) -> Option<crate::ffi::gdextension::GDExtensionPtrBuiltInMethod> {
        Some(mock_ptr_builtin_method)
    }

    /// Mock `variant_get_ptr_destructor` factory. Returns a typed no-op
    /// destructor regardless of the requested type.
    pub(crate) unsafe extern "C" fn mock_variant_get_ptr_destructor(
        _ty: GDExtensionVariantType,
    ) -> Option<crate::ffi::gdextension::GDExtensionPtrDestructor> {
        Some(mock_ptr_destructor)
    }

    /// Sig-aware `get_proc_address`: returns per-name typed mocks for the
    /// names `from_proc_address` will actually CALL, and a plain
    /// `sentinel` for names that are only stored as raw pointers. Safe
    /// to use with the real `from_proc_address` — the factory calls hit
    /// typed mocks with the correct signatures.
    ///
    /// # Safety
    ///
    /// `p_name` must be a NUL-terminated C string (Godot's contract).
    pub(crate) unsafe extern "C" fn sig_aware_gpa(
        p_name: *const c_char,
    ) -> crate::ffi::gdextension::GDExtensionInterfaceFunctionPtr {
        // SAFETY: `p_name` is NUL-terminated per Godot's contract.
        let name = unsafe { core::ffi::CStr::from_ptr(p_name) }.to_bytes_with_nul();

        // Match each name to its typed mock. Round-trip via
        // Option<unsafe extern "C" fn()> — a fn ptr is one word regardless
        // of typed signature; the caller re-transmutes before calling.
        //
        // SAFETY: transmuting one fn-ptr type to another is defined
        // behaviour for storage; UB only manifests if the caller then
        // invokes the wrong signature. `from_proc_address` transmutes
        // back to the correct type before invocation.
        unsafe {
            if name == names::VARIANT_GET_TYPE {
                Some(core::mem::transmute::<
                    GDExtensionInterfaceVariantGetType,
                    unsafe extern "C" fn(),
                >(mock_variant_get_type))
            } else if name == names::VARIANT_NEW_NIL {
                Some(core::mem::transmute::<
                    GDExtensionInterfaceVariantNewNil,
                    unsafe extern "C" fn(),
                >(mock_variant_new_nil))
            } else if name == names::GET_VARIANT_FROM_TYPE_CONSTRUCTOR {
                Some(core::mem::transmute::<
                    GDExtensionInterfaceGetVariantFromTypeConstructor,
                    unsafe extern "C" fn(),
                >(mock_get_variant_from_type_ctor))
            } else if name == names::GET_VARIANT_TO_TYPE_CONSTRUCTOR {
                Some(core::mem::transmute::<
                    GDExtensionInterfaceGetVariantToTypeConstructor,
                    unsafe extern "C" fn(),
                >(mock_get_variant_to_type_ctor))
            } else if name == names::STRING_NAME_NEW_WITH_UTF8_CHARS {
                // M3-11 T14-followup: `from_proc_address` calls the
                // resolved `string_name_new_with_utf8_chars` to build
                // "resize"/"size" StringNames — sig-aware mock must
                // return a typed no-op.
                Some(core::mem::transmute::<
                    GDExtensionInterfaceStringNameNewWithUtf8Chars,
                    unsafe extern "C" fn(),
                >(mock_string_name_new_with_utf8_chars))
            } else if name == names::VARIANT_GET_PTR_CONSTRUCTOR {
                Some(core::mem::transmute::<
                    GDExtensionInterfaceVariantGetPtrConstructor,
                    unsafe extern "C" fn(),
                >(mock_variant_get_ptr_constructor))
            } else if name == names::VARIANT_GET_PTR_BUILTIN_METHOD {
                Some(core::mem::transmute::<
                    GDExtensionInterfaceVariantGetPtrBuiltinMethod,
                    unsafe extern "C" fn(),
                >(mock_variant_get_ptr_builtin_method))
            } else if name == names::VARIANT_GET_PTR_DESTRUCTOR {
                Some(core::mem::transmute::<
                    GDExtensionInterfaceVariantGetPtrDestructor,
                    unsafe extern "C" fn(),
                >(mock_variant_get_ptr_destructor))
            } else {
                Some(sentinel)
            }
        }
    }

    /// Return a fully-populated `InterfaceTable` built from `sig_aware_gpa`.
    /// Shared by sibling test modules that need to inject a live-shape
    /// interface into `EXTENSION_STATE`.
    pub(crate) fn make_sig_aware_interface() -> InterfaceTable {
        // SAFETY: `sig_aware_gpa` matches the resolver signature; the
        // typed mocks it returns match the true signatures of every
        // interface fn `from_proc_address` invokes.
        unsafe { InterfaceTable::from_proc_address(sig_aware_gpa) }
            .expect("sig-aware mock must yield a fully populated interface")
    }

    /// Mock resolver: verifies every field is populated and no name is
    /// silently short-circuited. Uses the sig-aware mock because
    /// `from_proc_address` now invokes the factory-typed fn pointers.
    #[test]
    fn from_proc_address_populates_every_field_when_resolver_returns_fn() {
        let table = make_sig_aware_interface();

        // Non-variant fields still point at `sentinel` — verify.
        let sentinel_addr = sentinel as *const () as usize;
        assert_eq!(
            table.classdb_register_extension_class3 as usize,
            sentinel_addr
        );
        assert_eq!(
            table.classdb_register_extension_class_method as usize,
            sentinel_addr
        );
        assert_eq!(
            table.classdb_register_extension_class_signal as usize,
            sentinel_addr
        );
        assert_eq!(
            table.classdb_unregister_extension_class as usize,
            sentinel_addr
        );
        // After the T14-followup PackedFloat32Array packing pipeline land,
        // `from_proc_address` INVOKES the resolved `string_name_new_with_utf8_chars`
        // to build the "resize"/"size" StringNames on a stack buffer at init
        // time. The sig-aware mock returns a typed no-op mock for this name
        // (see `mock_string_name_new_with_utf8_chars`), so the stored fn ptr
        // is the mock's address rather than the plain `sentinel`.
        assert_eq!(
            table.string_name_new_with_utf8_chars as usize,
            mock_string_name_new_with_utf8_chars as *const () as usize
        );
        assert_eq!(
            table.string_name_new_with_latin1_chars as usize,
            sentinel_addr
        );
        assert_eq!(table.mem_alloc as usize, sentinel_addr);
        assert_eq!(table.mem_free as usize, sentinel_addr);

        // Variant-support fields point at their typed mocks — they DO NOT
        // dispatch to sentinel because `from_proc_address` re-transmutes
        // the round-tripped `unsafe fn()` back to the true type. Compare
        // by fn-pointer address.
        assert_eq!(
            table.variant_get_type as usize,
            mock_variant_get_type as *const () as usize
        );
        assert_eq!(
            table.variant_new_nil as usize,
            mock_variant_new_nil as *const () as usize
        );
        assert_eq!(
            table.variant_from_int_ctor as usize,
            mock_variant_from_int as *const () as usize
        );
        assert_eq!(
            table.variant_to_int_ctor as usize,
            mock_variant_to_int as *const () as usize
        );

        // M3-11 T14/M3-18 unpack foundation: 5 new named APIs are only
        // STORED (never invoked by `from_proc_address`), so they land on
        // the `sentinel` bucket in the sig-aware mock.
        assert_eq!(table.variant_new_copy as usize, sentinel_addr);
        assert_eq!(table.variant_destroy as usize, sentinel_addr);
        assert_eq!(
            table.packed_float32_array_operator_index_const as usize,
            sentinel_addr
        );
        assert_eq!(
            table.string_new_with_utf8_chars_and_len as usize,
            sentinel_addr
        );
        assert_eq!(table.string_to_utf8_chars as usize, sentinel_addr);

        // Additional cached typed constructors. `mock_get_variant_from_type_ctor`
        // returns `mock_variant_from_int` regardless of the type argument,
        // and the `_to_type_ctor` mock returns `mock_variant_to_int` — so
        // every cached from/to ctor decays to those two addresses. This is
        // a valid mock posture because the storage shape is universal
        // (`VariantFromTypeCtor` / `VariantToTypeCtor`); real Godot returns
        // different addresses per type but the mock does not need to
        // distinguish for the "every-field-populated" pass.
        assert_eq!(
            table.variant_from_string_ctor as usize,
            mock_variant_from_int as *const () as usize
        );
        assert_eq!(
            table.variant_to_string_ctor as usize,
            mock_variant_to_int as *const () as usize
        );
        assert_eq!(
            table.variant_from_packed_float32_array_ctor as usize,
            mock_variant_from_int as *const () as usize
        );
        assert_eq!(
            table.variant_to_packed_float32_array_ctor as usize,
            mock_variant_to_int as *const () as usize
        );
        assert_eq!(
            table.variant_from_dictionary_ctor as usize,
            mock_variant_from_int as *const () as usize
        );
        assert_eq!(
            table.variant_to_dictionary_ctor as usize,
            mock_variant_to_int as *const () as usize
        );
        assert_eq!(
            table.variant_from_object_ctor as usize,
            mock_variant_from_int as *const () as usize
        );
        assert_eq!(
            table.variant_to_object_ctor as usize,
            mock_variant_to_int as *const () as usize
        );
    }

    /// Mock resolver that returns NULL for a specific name. Verifies the
    /// resolver bails cleanly (returns `None`) instead of populating the
    /// table with NULL fn pointers.
    ///
    /// After the M3-11 T14-followup Dictionary + PackedFloat32Array pipeline
    /// additions the resolver walks 22 names (8 legacy + 4 Variant
    /// introspection/factory + 5 unpack-foundation + 1 mutable PFA index +
    /// 3 pipeline factories + 1 dictionary_operator_index), so the position
    /// loop iterates 0..22.
    #[test]
    fn from_proc_address_returns_none_when_any_lookup_fails() {
        use core::sync::atomic::{AtomicU32, Ordering};

        // 22 slots (fits u32 mask trivially).
        static NULL_MASK: AtomicU32 = AtomicU32::new(0);
        static COUNTER: AtomicU32 = AtomicU32::new(0);

        /// Number of named `get_proc_address` calls performed by
        /// `from_proc_address` — pinned by the walk-through in that fn
        /// (8 legacy + 4 Variant introspection + 5 unpack foundation +
        /// 4 T14-followup PackedFloat32Array pipeline + 1 Dictionary
        /// pipeline). A drift here is a signal that a resolver was added
        /// or removed without updating this test.
        const NAMED_RESOLVER_COUNT: u32 = 22;

        unsafe extern "C" fn mock_gpa(
            p_name: *const c_char,
        ) -> crate::ffi::gdextension::GDExtensionInterfaceFunctionPtr {
            let n = COUNTER.fetch_add(1, Ordering::SeqCst);
            let mask = NULL_MASK.load(Ordering::SeqCst);
            if (mask >> n) & 1 == 1 {
                None
            } else {
                // Delegate to sig-aware mock so factory-call names produce
                // valid factories. Position-specific NULL trumps this.
                //
                // SAFETY: `sig_aware_gpa` matches the resolver signature.
                unsafe { sig_aware_gpa(p_name) }
            }
        }

        // Test failing on every one of the 17 positions individually.
        for pos in 0..NAMED_RESOLVER_COUNT {
            NULL_MASK.store(1 << pos, Ordering::SeqCst);
            COUNTER.store(0, Ordering::SeqCst);
            // SAFETY: same mock, single-threaded test.
            let result = unsafe { InterfaceTable::from_proc_address(mock_gpa) };
            assert!(
                result.is_none(),
                "NULL at resolver position {pos} must propagate to None",
            );
        }
    }

    /// Even when every resolver returns non-NULL, `from_proc_address`
    /// must bail if the Int constructor factory itself returns None
    /// (Godot documents that unknown-type factories return NULL — a
    /// factory returning NULL for Int would indicate a corrupt host).
    #[test]
    fn from_proc_address_returns_none_when_int_from_factory_returns_null() {
        unsafe extern "C" fn null_from_factory(
            _ty: GDExtensionVariantType,
        ) -> crate::ffi::gdextension::GDExtensionVariantFromTypeConstructorFunc {
            None
        }

        unsafe extern "C" fn gpa(
            p_name: *const c_char,
        ) -> crate::ffi::gdextension::GDExtensionInterfaceFunctionPtr {
            // SAFETY: `p_name` is NUL-terminated per Godot's contract.
            let name = unsafe { core::ffi::CStr::from_ptr(p_name) }.to_bytes_with_nul();
            if name == names::GET_VARIANT_FROM_TYPE_CONSTRUCTOR {
                // SAFETY: transmute one-word fn ptr to storage type.
                unsafe {
                    Some(core::mem::transmute::<
                        GDExtensionInterfaceGetVariantFromTypeConstructor,
                        unsafe extern "C" fn(),
                    >(null_from_factory))
                }
            } else {
                // SAFETY: same rationale as `sig_aware_gpa`.
                unsafe { sig_aware_gpa(p_name) }
            }
        }

        // SAFETY: `gpa` matches the resolver signature.
        let result = unsafe { InterfaceTable::from_proc_address(gpa) };
        assert!(
            result.is_none(),
            "NULL from get_variant_from_type_constructor(Int) must propagate",
        );
    }

    /// Symmetric: the `get_variant_to_type_constructor(Int)` factory
    /// returning NULL must also bail.
    #[test]
    fn from_proc_address_returns_none_when_int_to_factory_returns_null() {
        unsafe extern "C" fn null_to_factory(
            _ty: GDExtensionVariantType,
        ) -> crate::ffi::gdextension::GDExtensionTypeFromVariantConstructorFunc {
            None
        }

        unsafe extern "C" fn gpa(
            p_name: *const c_char,
        ) -> crate::ffi::gdextension::GDExtensionInterfaceFunctionPtr {
            // SAFETY: `p_name` is NUL-terminated per Godot's contract.
            let name = unsafe { core::ffi::CStr::from_ptr(p_name) }.to_bytes_with_nul();
            if name == names::GET_VARIANT_TO_TYPE_CONSTRUCTOR {
                // SAFETY: transmute one-word fn ptr to storage type.
                unsafe {
                    Some(core::mem::transmute::<
                        GDExtensionInterfaceGetVariantToTypeConstructor,
                        unsafe extern "C" fn(),
                    >(null_to_factory))
                }
            } else {
                // SAFETY: same rationale as `sig_aware_gpa`.
                unsafe { sig_aware_gpa(p_name) }
            }
        }

        // SAFETY: `gpa` matches the resolver signature.
        let result = unsafe { InterfaceTable::from_proc_address(gpa) };
        assert!(
            result.is_none(),
            "NULL from get_variant_to_type_constructor(Int) must propagate",
        );
    }

    // ------------------------------------------------------------------
    // M3-11 T14/M3-18 unpack foundation: parametric coverage.
    //
    // Every additional cached constructor MUST propagate a NULL factory
    // result as `None` from `from_proc_address` (FR-EX-08 honest failure).
    // The Int-specific tests above prove the pattern; the tests below
    // cover the 4 new types (STRING / PACKED_FLOAT32_ARRAY / DICTIONARY /
    // OBJECT) x 2 directions parametrically. Rather than 8 near-copies,
    // one shared helper drives per-type factory NULLs and a single test
    // asserts every case.
    // ------------------------------------------------------------------

    /// Global switches used by the parametric factory mocks. Kept as
    /// static so `unsafe extern "C" fn` factories (which cannot capture
    /// environment) can consult them.
    ///
    /// Layout is bit-packed:
    ///   bits 0..3   = FROM factory returns NULL for {Nil, Bool, Int, Float}
    ///                 (unused — Int case is covered by dedicated test).
    ///   bit 4       = FROM factory returns NULL for STRING.
    ///   bit 8       = FROM factory returns NULL for OBJECT (24).
    ///   bit 10      = FROM factory returns NULL for DICTIONARY (26).
    ///   bit 14      = FROM factory returns NULL for PACKED_FLOAT32_ARRAY (30).
    /// The TO factory reads a parallel mask.
    static NULL_FROM_TYPE_MASK: core::sync::atomic::AtomicU32 =
        core::sync::atomic::AtomicU32::new(0);
    static NULL_TO_TYPE_MASK: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

    /// Encode a variant type as a bit position in the mask above. Maps
    /// the sparse enum-numeric layout (STRING=4, OBJECT=24, DICTIONARY=26,
    /// PACKED_FLOAT32_ARRAY=30) into a small dense mask keyed by the type
    /// discriminant.
    fn type_to_mask_bit(ty: GDExtensionVariantType) -> u32 {
        // Map every 4.3-stable type discriminant we cache directly to a
        // bit position. Any other type would never be requested by
        // `from_proc_address` (the resolver walk is closed).
        match ty {
            GDExtensionVariantType::Nil => 0,
            GDExtensionVariantType::Bool => 1,
            GDExtensionVariantType::Int => 2,
            GDExtensionVariantType::Float => 3,
            GDExtensionVariantType::String => 4,
            GDExtensionVariantType::Object => 8,
            GDExtensionVariantType::Dictionary => 10,
            GDExtensionVariantType::PackedFloat32Array => 14,
        }
    }

    /// Parametric FROM factory: returns `None` for every type whose bit
    /// is set in `NULL_FROM_TYPE_MASK`; otherwise dispatches to the
    /// sig-aware no-op packer.
    unsafe extern "C" fn parametric_from_factory(
        ty: GDExtensionVariantType,
    ) -> crate::ffi::gdextension::GDExtensionVariantFromTypeConstructorFunc {
        let bit = type_to_mask_bit(ty);
        let mask = NULL_FROM_TYPE_MASK.load(core::sync::atomic::Ordering::SeqCst);
        if (mask >> bit) & 1 == 1 {
            None
        } else {
            Some(mock_variant_from_int)
        }
    }

    /// Parametric TO factory — symmetric to `parametric_from_factory`.
    unsafe extern "C" fn parametric_to_factory(
        ty: GDExtensionVariantType,
    ) -> crate::ffi::gdextension::GDExtensionTypeFromVariantConstructorFunc {
        let bit = type_to_mask_bit(ty);
        let mask = NULL_TO_TYPE_MASK.load(core::sync::atomic::Ordering::SeqCst);
        if (mask >> bit) & 1 == 1 {
            None
        } else {
            Some(mock_variant_to_int)
        }
    }

    /// Resolver that wires the parametric factories into
    /// `from_proc_address`. Every other name is delegated to
    /// `sig_aware_gpa`.
    unsafe extern "C" fn parametric_gpa(
        p_name: *const c_char,
    ) -> crate::ffi::gdextension::GDExtensionInterfaceFunctionPtr {
        // SAFETY: `p_name` is NUL-terminated per Godot's contract.
        let name = unsafe { core::ffi::CStr::from_ptr(p_name) }.to_bytes_with_nul();
        if name == names::GET_VARIANT_FROM_TYPE_CONSTRUCTOR {
            // SAFETY: fn-ptr storage transmute — one word regardless of
            // typed signature.
            unsafe {
                Some(core::mem::transmute::<
                    GDExtensionInterfaceGetVariantFromTypeConstructor,
                    unsafe extern "C" fn(),
                >(parametric_from_factory))
            }
        } else if name == names::GET_VARIANT_TO_TYPE_CONSTRUCTOR {
            // SAFETY: same rationale.
            unsafe {
                Some(core::mem::transmute::<
                    GDExtensionInterfaceGetVariantToTypeConstructor,
                    unsafe extern "C" fn(),
                >(parametric_to_factory))
            }
        } else {
            // SAFETY: `sig_aware_gpa` matches the resolver signature.
            unsafe { sig_aware_gpa(p_name) }
        }
    }

    /// Every additional cached typed constructor
    /// (STRING/PACKED_FLOAT32_ARRAY/DICTIONARY/OBJECT × from/to = 8 cases)
    /// must propagate a NULL factory result to `None`. The Int-specific
    /// tests above prove the identical invariant for Int.
    #[test]
    fn from_proc_address_returns_none_when_any_new_type_factory_returns_null() {
        use core::sync::atomic::Ordering;

        let cases: [(GDExtensionVariantType, bool); 8] = [
            (GDExtensionVariantType::String, true),  // FROM null
            (GDExtensionVariantType::String, false), // TO null
            (GDExtensionVariantType::PackedFloat32Array, true),
            (GDExtensionVariantType::PackedFloat32Array, false),
            (GDExtensionVariantType::Dictionary, true),
            (GDExtensionVariantType::Dictionary, false),
            (GDExtensionVariantType::Object, true),
            (GDExtensionVariantType::Object, false),
        ];

        for (ty, is_from_null) in cases {
            // Reset masks: only one bit set per iteration to prove each
            // case independently propagates NULL.
            NULL_FROM_TYPE_MASK.store(0, Ordering::SeqCst);
            NULL_TO_TYPE_MASK.store(0, Ordering::SeqCst);
            let bit = type_to_mask_bit(ty);
            if is_from_null {
                NULL_FROM_TYPE_MASK.store(1 << bit, Ordering::SeqCst);
            } else {
                NULL_TO_TYPE_MASK.store(1 << bit, Ordering::SeqCst);
            }

            // SAFETY: `parametric_gpa` matches the resolver signature.
            let result = unsafe { InterfaceTable::from_proc_address(parametric_gpa) };
            assert!(
                result.is_none(),
                "NULL from get_variant_{}_type_constructor({:?}) must propagate",
                if is_from_null { "from" } else { "to" },
                ty,
            );
        }

        // Reset for hygiene — subsequent tests see all-zero masks.
        NULL_FROM_TYPE_MASK.store(0, Ordering::SeqCst);
        NULL_TO_TYPE_MASK.store(0, Ordering::SeqCst);
    }

    /// Symmetric positive case: with all masks cleared, `from_proc_address`
    /// must succeed. Guards against a `parametric_gpa` regression that
    /// would silently null-out a case the previous test doesn't cover.
    #[test]
    fn from_proc_address_succeeds_when_parametric_factories_are_populated() {
        use core::sync::atomic::Ordering;
        NULL_FROM_TYPE_MASK.store(0, Ordering::SeqCst);
        NULL_TO_TYPE_MASK.store(0, Ordering::SeqCst);

        // SAFETY: parametric mocks match resolver signatures.
        let result = unsafe { InterfaceTable::from_proc_address(parametric_gpa) };
        assert!(
            result.is_some(),
            "all factories non-NULL must yield a populated table"
        );
    }
}
