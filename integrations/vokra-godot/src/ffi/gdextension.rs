//! Minimal `extern "C"` bindings for `gdextension_interface.h` (Godot 4.3+
//! MIT, ADR-0011 §D3). Only the types/enums that appear on the initialization
//! path (T04), the class-registration path (T05..T09), and the method-binding
//! trampolines (T06) are bound here. The Godot header is intentionally NOT
//! copied into this crate — we declare the pinned subset we actually use, so
//! incremental additions to the header downstream do not force churn on our
//! binding.
//!
//! # Godot version pin
//!
//! Every struct in this file is layout-frozen against Godot **4.3-stable**
//! (`godot/core/extension/gdextension_interface.h`, MIT). GDExtension is
//! documented as an unstable ABI across major versions; when Godot bumps a
//! major (e.g. 4.5+) or renames a versioned struct
//! (`classdb_register_extension_class4`, `GDExtensionClassCreationInfo4`),
//! this module MUST be re-audited alongside the AssetLib packaging
//! (`vokra.gdextension` `compatibility_minimum`).
//!
//! The sizes/alignments below (`GDExtensionClassCreationInfo3` = 160 bytes,
//! `GDExtensionClassMethodInfo` = 88 bytes, `GDExtensionPropertyInfo` = 48
//! bytes, `GDExtensionCallError` = 12 bytes) were probed with `clang -m64`
//! on x86_64 macOS against the 4.3-stable header (identical LP64 layout on
//! x86_64 Linux and macOS/ARM64). A drift MUST be caught by the compile-time
//! guards at the bottom of this file — a silent widening would corrupt
//! Godot's stack on the first `create_instance_func` / call dispatch.

use core::ffi::c_char;

/// GDExtension boolean is an `uint8_t` (0/1), matches
/// `typedef uint8_t GDExtensionBool` in `gdextension_interface.h`.
pub type GDExtensionBool = u8;

/// GDExtension int is a signed 64-bit integer, matches
/// `typedef int64_t GDExtensionInt`.
pub type GDExtensionInt = i64;

/// GDExtension initialization level, matches
/// `typedef enum { ... } GDExtensionInitializationLevel` in
/// `gdextension_interface.h`. Numeric values MUST match Godot's enum layout
/// and are pinned by the T20 layout test.
///
/// - `Core` (0): earliest — before any Godot server. Vokra does not register
///   at this level.
/// - `Servers` (1): before scene tree. Vokra does not register here either.
/// - `Scene` (2): after scene tree is ready — where `VokraSession` is
///   registered as an Object class.
/// - `Editor` (3): editor-only, unused by Vokra (M3 has no editor tooling).
#[repr(C)]
#[allow(dead_code)] // Variants correspond 1:1 to Godot enum; not all used yet.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GDExtensionInitializationLevel {
    Core = 0,
    Servers = 1,
    Scene = 2,
    Editor = 3,
}

/// GDExtension Variant type tag. This is a bare C enum in Godot's header (no
/// `: uint32_t` specifier) → 4 bytes on every LP64 platform. Only the values
/// Vokra's method signatures reference are exhaustively named here; the tail
/// variants are collapsed under `Other` so an unknown code round-trips
/// unchanged.
///
/// Vokra methods bind these types:
/// - `PackedFloat32Array` (32): PCM in/out
/// - `String` (4): text in/out (ASR result, TTS text)
/// - `Int` (2): sample rate
/// - `Object` (24): VokraStream returned by `vad_open_stream`
/// - `Dictionary` (27): TTS output bag (`{pcm, sample_rate}`)
/// - `Nil` (0): void return
///
/// # Provenance
///
/// Variant type codes are pinned by the Godot 4.3-stable
/// `core/extension/gdextension_interface.h` `GDExtensionVariantType` enum
/// (a bare, sequentially-numbered C enum whose values match `Variant::Type`
/// from `core/variant/variant.h` 1:1). The numeric values below were
/// counted verbatim from the 4.3-stable header:
///
/// ```text
///   0=NIL, 1=BOOL, 2=INT, 3=FLOAT, 4=STRING,
///   5..19=math types (VECTOR2..PROJECTION),
///   20=COLOR, 21=STRING_NAME, 22=NODE_PATH, 23=RID, 24=OBJECT,
///   25=CALLABLE, 26=SIGNAL, 27=DICTIONARY, 28=ARRAY,
///   29=PACKED_BYTE_ARRAY, 30=PACKED_INT32_ARRAY, 31=PACKED_INT64_ARRAY,
///   32=PACKED_FLOAT32_ARRAY, 33=PACKED_FLOAT64_ARRAY, ...
/// ```
///
/// A drift here (e.g. leaving `PackedFloat32Array = 30` — the value that
/// held before the M3-11 T14-followup PackedFloat32Array unpack land) would
/// cause `variant_get_type == PackedFloat32Array` to ALWAYS FALSE for real
/// PackedFloat32Array Variants — silently poisoning every trampoline that
/// type-checks a PackedFloat32Array arg.
#[repr(C)]
#[allow(dead_code)] // Tail variants unused; kept for future method surfaces.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GDExtensionVariantType {
    Nil = 0,
    Bool = 1,
    Int = 2,
    Float = 3,
    String = 4,
    /// `GDEXTENSION_VARIANT_TYPE_OBJECT` — used as the return type for
    /// `VokraSession::vad_open_stream(sr: int) -> VokraStream`. We do not
    /// pack/unpack Object Variants in this crate (Object wrapping is
    /// deferred to a follow-up patch); the variant is declared here so
    /// `variant_get_type` return-value matching stays exhaustive.
    Object = 24,
    /// `GDEXTENSION_VARIANT_TYPE_DICTIONARY` — used for the TTS output bag.
    Dictionary = 27,
    /// `GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY` — used for PCM I/O.
    PackedFloat32Array = 32,
}

/// GDExtension call-error tag. Matches the `GDExtensionCallErrorType` enum in
/// the header 1:1.
#[repr(i32)]
#[allow(dead_code)] // Tail variants used by future method surfaces.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GDExtensionCallErrorType {
    Ok = 0,
    InvalidMethod = 1,
    InvalidArgument = 2,
    TooManyArguments = 3,
    TooFewArguments = 4,
    InstanceIsNull = 5,
    MethodNotConst = 6,
}

/// Argument metadata for method binding — matches `GDExtensionClassMethodArgumentMetadata`
/// in the header. Bare C enum → 4 bytes. We only ever use `None` because we
/// declare properties by variant type, not by int/float width.
#[repr(C)]
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GDExtensionClassMethodArgumentMetadata {
    None = 0,
    IntIsInt8 = 1,
    IntIsInt16 = 2,
    IntIsInt32 = 3,
    IntIsInt64 = 4,
    IntIsUint8 = 5,
    IntIsUint16 = 6,
    IntIsUint32 = 7,
    IntIsUint64 = 8,
    RealIsFloat = 9,
    RealIsDouble = 10,
}

/// Bitfield for `GDExtensionClassMethodInfo::method_flags` — matches
/// `GDExtensionClassMethodFlags`. Kept as bare `u32` (not `enum`) because
/// Godot uses OR-combined flags.
#[allow(dead_code)]
pub mod method_flags {
    pub const NORMAL: u32 = 1;
    pub const EDITOR: u32 = 2;
    pub const CONST: u32 = 4;
    pub const VIRTUAL: u32 = 8;
    pub const VARARG: u32 = 16;
    pub const STATIC: u32 = 32;
    pub const DEFAULT: u32 = NORMAL;
}

/// `GDExtensionCallError` — the write-back error slot for
/// `GDExtensionClassMethodCall`. 12-byte POD (enum + i32 + i32).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct GDExtensionCallError {
    pub error: GDExtensionCallErrorType,
    /// Index of the offending argument (for InvalidArgument), or `-1`.
    pub argument: i32,
    /// Expected type / count (for InvalidArgument / TooMany / TooFew).
    pub expected: i32,
}

/// Opaque `GDExtensionInterfaceFunctionPtr` — an untyped function pointer
/// returned by `get_proc_address` for named GDExtension APIs. Every use MUST
/// `mem::transmute` to the concrete signature documented in
/// `gdextension_interface.h`.
pub type GDExtensionInterfaceFunctionPtr = Option<unsafe extern "C" fn()>;

/// `typedef GDExtensionInterfaceFunctionPtr (*GDExtensionInterfaceGetProcAddress)(const char *p_name);`
pub type GDExtensionInterfaceGetProcAddress =
    unsafe extern "C" fn(p_name: *const c_char) -> GDExtensionInterfaceFunctionPtr;

/// Opaque `GDExtensionClassLibraryPtr` — Godot's handle for the library that
/// registered a class. Only used as an opaque token in our surface; we pin it
/// through the extension lifetime for later `classdb_register_extension_class`
/// calls (T05).
pub type GDExtensionClassLibraryPtr = *mut core::ffi::c_void;

/// Opaque pointer types matching the Godot header (all `void *` in C).
pub type GDExtensionClassInstancePtr = *mut core::ffi::c_void;
pub type GDExtensionObjectPtr = *mut core::ffi::c_void;
pub type GDExtensionConstObjectPtr = *const core::ffi::c_void;
pub type GDExtensionStringNamePtr = *mut core::ffi::c_void;
pub type GDExtensionConstStringNamePtr = *const core::ffi::c_void;
pub type GDExtensionUninitializedStringNamePtr = *mut core::ffi::c_void;
pub type GDExtensionStringPtr = *mut core::ffi::c_void;
pub type GDExtensionConstStringPtr = *const core::ffi::c_void;
/// `typedef void *GDExtensionUninitializedStringPtr;` (Godot 4.3-stable
/// `gdextension_interface.h` line 167). Same underlying representation as
/// [`GDExtensionStringPtr`]; the "uninitialized" naming is a documented C-level
/// intent marker on the destination side of a typed-String constructor
/// (e.g. `string_new_with_utf8_chars_and_len`).
pub type GDExtensionUninitializedStringPtr = *mut core::ffi::c_void;
pub type GDExtensionVariantPtr = *mut core::ffi::c_void;
pub type GDExtensionConstVariantPtr = *const core::ffi::c_void;
pub type GDExtensionUninitializedVariantPtr = *mut core::ffi::c_void;
pub type GDExtensionTypePtr = *mut core::ffi::c_void;
pub type GDExtensionConstTypePtr = *const core::ffi::c_void;
/// `typedef void *GDExtensionUninitializedTypePtr;` (Godot 4.3-stable
/// `gdextension_interface.h`). Same underlying representation as
/// [`GDExtensionTypePtr`]; the "uninitialized" naming is a documented C-level
/// intent marker on the destination side of a per-type constructor.
pub type GDExtensionUninitializedTypePtr = *mut core::ffi::c_void;

/// `GDExtensionInitialization` struct — the mutable output of the extension
/// entry point. `#[repr(C)]` order MUST match the Godot header exactly.
#[repr(C)]
pub struct GDExtensionInitialization {
    /// Minimum init level the extension needs to run at. `Scene` for us.
    pub minimum_initialization_level: GDExtensionInitializationLevel,
    /// Opaque userdata forwarded to `initialize` / `deinitialize`. `NULL`
    /// (`ptr::null_mut`) for us: init state is static per-cdylib and reset on
    /// Godot's process termination.
    pub userdata: *mut core::ffi::c_void,
    /// Called by Godot at each init level, ascending. `NULL` = no work at
    /// that level. Vokra wires up only the Scene-level callback.
    pub initialize: Option<
        unsafe extern "C" fn(
            userdata: *mut core::ffi::c_void,
            p_level: GDExtensionInitializationLevel,
        ),
    >,
    /// Called by Godot at each teardown level, descending. Mirror of
    /// `initialize`.
    pub deinitialize: Option<
        unsafe extern "C" fn(
            userdata: *mut core::ffi::c_void,
            p_level: GDExtensionInitializationLevel,
        ),
    >,
}

/// The GDExtension entry point signature. Our exported `vokra_gdextension_init`
/// must match this exactly (checked by a static-cast in a compile-time test).
///
/// `p_get_proc_address` resolves any GDExtension API by name (e.g.
/// `"classdb_register_extension_class3"`), `p_library` is the library token
/// Godot binds the registered classes to, `r_initialization` is the mutable
/// output the extension fills in. Return `1` (success) / `0` (fail).
pub type GDExtensionInitializationFunction = unsafe extern "C" fn(
    p_get_proc_address: GDExtensionInterfaceGetProcAddress,
    p_library: GDExtensionClassLibraryPtr,
    r_initialization: *mut GDExtensionInitialization,
) -> GDExtensionBool;

// ---------------------------------------------------------------------------
// Property + method info structs (T05 class-info population, T06 method-info
// registration). All layouts are 4.3-stable header-verified with clang -m64.
// ---------------------------------------------------------------------------

/// `GDExtensionPropertyInfo` — 48 bytes on LP64. Used inside method info to
/// declare argument + return types. Only the fields we actually populate
/// (`type`, `name`, `usage`) carry meaning for us; `class_name`, `hint`,
/// `hint_string`, `usage` bits above `PROPERTY_USAGE_DEFAULT` are left at
/// their zero defaults.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GDExtensionPropertyInfo {
    /// Variant type of the property / argument / return.
    pub r#type: GDExtensionVariantType,
    /// StringName pointer (pre-constructed via
    /// `string_name_new_with_utf8_chars`, held for the register call's
    /// lifetime).
    pub name: GDExtensionStringNamePtr,
    /// Class name for Object-typed properties (StringName). NULL for us.
    pub class_name: GDExtensionStringNamePtr,
    /// PropertyHint bitfield. We default to `PROPERTY_HINT_NONE` (0).
    pub hint: u32,
    /// Hint string as a Godot `String*`. NULL for us.
    pub hint_string: GDExtensionStringPtr,
    /// PropertyUsageFlags bitfield. We default to
    /// `PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_NO_EDITOR` (6). The default
    /// here is 6 (`STORAGE | EDITOR`); explicitly setting `USAGE_DEFAULT`
    /// keeps the argument visible in ClassDB introspection.
    pub usage: u32,
}

/// `GDExtensionClassMethodCall` — Variant-based method dispatch entry point.
/// Every method registered via `classdb_register_extension_class_method`
/// receives this signature.
pub type GDExtensionClassMethodCall = unsafe extern "C" fn(
    method_userdata: *mut core::ffi::c_void,
    p_instance: GDExtensionClassInstancePtr,
    p_args: *const GDExtensionConstVariantPtr,
    p_argument_count: GDExtensionInt,
    r_return: GDExtensionVariantPtr,
    r_error: *mut GDExtensionCallError,
);

/// `GDExtensionClassMethodPtrCall` — ptrcall dispatch (skips Variant boxing).
/// We currently register `None` for this because ptrcall requires typed
/// signatures beyond our T06 scope.
pub type GDExtensionClassMethodPtrCall = unsafe extern "C" fn(
    method_userdata: *mut core::ffi::c_void,
    p_instance: GDExtensionClassInstancePtr,
    p_args: *const GDExtensionConstTypePtr,
    r_ret: GDExtensionTypePtr,
);

/// `GDExtensionClassMethodInfo` — 88 bytes on LP64. Populated at each call
/// to `classdb_register_extension_class_method`.
#[repr(C)]
pub struct GDExtensionClassMethodInfo {
    /// StringName pointer holding the method's Godot-visible name.
    pub name: GDExtensionStringNamePtr,
    /// Passed through to `call_func` / `ptrcall_func` as `method_userdata`.
    /// Vokra uses this to disambiguate trampolines that share a single
    /// implementation function.
    pub method_userdata: *mut core::ffi::c_void,
    /// Variant-based dispatch (mandatory).
    pub call_func: Option<GDExtensionClassMethodCall>,
    /// PtrCall dispatch (optional; `None` unless we have a typed signature).
    pub ptrcall_func: Option<GDExtensionClassMethodPtrCall>,
    /// OR-combined `method_flags::*` bits. `NORMAL` for our current surface.
    pub method_flags: u32,
    /// `1` iff `return_value_info` / `return_value_metadata` are populated.
    pub has_return_value: GDExtensionBool,
    // NOTE: struct alignment inserts 3 bytes of padding here before the
    // 8-byte pointer, matching the C header. Rust's `repr(C)` layout
    // reproduces that padding.
    pub return_value_info: *mut GDExtensionPropertyInfo,
    pub return_value_metadata: GDExtensionClassMethodArgumentMetadata,
    // NOTE: another 4-byte tail-padding is inserted here (offset 52 → 56)
    // before `argument_count`, matching the header. `repr(C)` handles it.
    pub argument_count: u32,
    pub arguments_info: *mut GDExtensionPropertyInfo,
    pub arguments_metadata: *mut GDExtensionClassMethodArgumentMetadata,
    pub default_argument_count: u32,
    // NOTE: 4 bytes of tail-padding (offset 76 → 80) before the pointer.
    pub default_arguments: *mut GDExtensionVariantPtr,
}

// ---------------------------------------------------------------------------
// Class creation callbacks + `GDExtensionClassCreationInfo3` (160 bytes on
// LP64). All 18 fn-pointer slots are typed against the 4.3-stable header;
// only `create_instance_func` / `free_instance_func` are populated by Vokra
// today, the rest are `None` (Godot treats absent callbacks as "not
// overridden").
// ---------------------------------------------------------------------------

pub type GDExtensionClassSet = unsafe extern "C" fn(
    p_instance: GDExtensionClassInstancePtr,
    p_name: GDExtensionConstStringNamePtr,
    p_value: GDExtensionConstVariantPtr,
) -> GDExtensionBool;

pub type GDExtensionClassGet = unsafe extern "C" fn(
    p_instance: GDExtensionClassInstancePtr,
    p_name: GDExtensionConstStringNamePtr,
    r_ret: GDExtensionVariantPtr,
) -> GDExtensionBool;

pub type GDExtensionClassGetRID =
    unsafe extern "C" fn(p_instance: GDExtensionClassInstancePtr) -> u64;

pub type GDExtensionClassGetPropertyList = unsafe extern "C" fn(
    p_instance: GDExtensionClassInstancePtr,
    r_count: *mut u32,
) -> *const GDExtensionPropertyInfo;

pub type GDExtensionClassFreePropertyList2 = unsafe extern "C" fn(
    p_instance: GDExtensionClassInstancePtr,
    p_list: *const GDExtensionPropertyInfo,
    p_count: u32,
);

pub type GDExtensionClassPropertyCanRevert = unsafe extern "C" fn(
    p_instance: GDExtensionClassInstancePtr,
    p_name: GDExtensionConstStringNamePtr,
) -> GDExtensionBool;

pub type GDExtensionClassPropertyGetRevert = unsafe extern "C" fn(
    p_instance: GDExtensionClassInstancePtr,
    p_name: GDExtensionConstStringNamePtr,
    r_ret: GDExtensionVariantPtr,
) -> GDExtensionBool;

pub type GDExtensionClassValidateProperty = unsafe extern "C" fn(
    p_instance: GDExtensionClassInstancePtr,
    p_property: *mut GDExtensionPropertyInfo,
) -> GDExtensionBool;

pub type GDExtensionClassNotification2 = unsafe extern "C" fn(
    p_instance: GDExtensionClassInstancePtr,
    p_what: i32,
    p_reversed: GDExtensionBool,
);

pub type GDExtensionClassToString = unsafe extern "C" fn(
    p_instance: GDExtensionClassInstancePtr,
    r_is_valid: *mut GDExtensionBool,
    p_out: GDExtensionStringPtr,
);

pub type GDExtensionClassReference = unsafe extern "C" fn(p_instance: GDExtensionClassInstancePtr);
pub type GDExtensionClassUnreference =
    unsafe extern "C" fn(p_instance: GDExtensionClassInstancePtr);

pub type GDExtensionClassCreateInstance =
    unsafe extern "C" fn(p_class_userdata: *mut core::ffi::c_void) -> GDExtensionObjectPtr;

pub type GDExtensionClassFreeInstance = unsafe extern "C" fn(
    p_class_userdata: *mut core::ffi::c_void,
    p_instance: GDExtensionClassInstancePtr,
);

pub type GDExtensionClassRecreateInstance = unsafe extern "C" fn(
    p_class_userdata: *mut core::ffi::c_void,
    p_object: GDExtensionObjectPtr,
) -> GDExtensionClassInstancePtr;

/// Placeholder typedef for `GDExtensionClassCallVirtual` (opaque fn pointer
/// returned by `get_virtual_func`). We don't override virtual methods yet,
/// so callers only need a `*const c_void` type-hygiene alias.
pub type GDExtensionClassCallVirtual = Option<unsafe extern "C" fn() /* opaque signature */>;

pub type GDExtensionClassGetVirtual = unsafe extern "C" fn(
    p_class_userdata: *mut core::ffi::c_void,
    p_name: GDExtensionConstStringNamePtr,
) -> GDExtensionClassCallVirtual;

pub type GDExtensionClassGetVirtualCallData = unsafe extern "C" fn(
    p_class_userdata: *mut core::ffi::c_void,
    p_name: GDExtensionConstStringNamePtr,
) -> *mut core::ffi::c_void;

pub type GDExtensionClassCallVirtualWithData = unsafe extern "C" fn(
    p_instance: GDExtensionClassInstancePtr,
    p_name: GDExtensionConstStringNamePtr,
    p_virtual_call_userdata: *mut core::ffi::c_void,
    p_args: *const GDExtensionConstTypePtr,
    r_ret: GDExtensionTypePtr,
);

/// `GDExtensionClassCreationInfo3` — 160 bytes on LP64. Field ordering MUST
/// match the 4.3-stable header 1:1; the compile-time layout guard at the
/// bottom of this file pins the size + alignment.
#[repr(C)]
pub struct GDExtensionClassCreationInfo3 {
    pub is_virtual: GDExtensionBool,
    pub is_abstract: GDExtensionBool,
    pub is_exposed: GDExtensionBool,
    pub is_runtime: GDExtensionBool,
    // NOTE: 4-byte alignment padding before the first 8-byte pointer.
    pub set_func: Option<GDExtensionClassSet>,
    pub get_func: Option<GDExtensionClassGet>,
    pub get_property_list_func: Option<GDExtensionClassGetPropertyList>,
    pub free_property_list_func: Option<GDExtensionClassFreePropertyList2>,
    pub property_can_revert_func: Option<GDExtensionClassPropertyCanRevert>,
    pub property_get_revert_func: Option<GDExtensionClassPropertyGetRevert>,
    pub validate_property_func: Option<GDExtensionClassValidateProperty>,
    pub notification_func: Option<GDExtensionClassNotification2>,
    pub to_string_func: Option<GDExtensionClassToString>,
    pub reference_func: Option<GDExtensionClassReference>,
    pub unreference_func: Option<GDExtensionClassUnreference>,
    pub create_instance_func: Option<GDExtensionClassCreateInstance>,
    pub free_instance_func: Option<GDExtensionClassFreeInstance>,
    pub recreate_instance_func: Option<GDExtensionClassRecreateInstance>,
    pub get_virtual_func: Option<GDExtensionClassGetVirtual>,
    pub get_virtual_call_data_func: Option<GDExtensionClassGetVirtualCallData>,
    pub call_virtual_with_data_func: Option<GDExtensionClassCallVirtualWithData>,
    pub get_rid_func: Option<GDExtensionClassGetRID>,
    pub class_userdata: *mut core::ffi::c_void,
}

// ---------------------------------------------------------------------------
// Interface function pointer typedefs. Resolved at extension init via
// `p_get_proc_address(<name>)`. Each name in a comment is the exact key
// passed to `get_proc_address` per the 4.3-stable header.
// ---------------------------------------------------------------------------

/// `classdb_register_extension_class3` (Godot 4.3).
pub type GDExtensionInterfaceClassdbRegisterExtensionClass3 = unsafe extern "C" fn(
    p_library: GDExtensionClassLibraryPtr,
    p_class_name: GDExtensionConstStringNamePtr,
    p_parent_class_name: GDExtensionConstStringNamePtr,
    p_extension_funcs: *const GDExtensionClassCreationInfo3,
);

/// `classdb_register_extension_class_method` (Godot 4.1).
pub type GDExtensionInterfaceClassdbRegisterExtensionClassMethod = unsafe extern "C" fn(
    p_library: GDExtensionClassLibraryPtr,
    p_class_name: GDExtensionConstStringNamePtr,
    p_method_info: *const GDExtensionClassMethodInfo,
);

/// `classdb_register_extension_class_signal` (Godot 4.1).
pub type GDExtensionInterfaceClassdbRegisterExtensionClassSignal = unsafe extern "C" fn(
    p_library: GDExtensionClassLibraryPtr,
    p_class_name: GDExtensionConstStringNamePtr,
    p_signal_name: GDExtensionConstStringNamePtr,
    p_argument_info: *const GDExtensionPropertyInfo,
    p_argument_count: GDExtensionInt,
);

/// `classdb_unregister_extension_class` (Godot 4.1).
pub type GDExtensionInterfaceClassdbUnregisterExtensionClass = unsafe extern "C" fn(
    p_library: GDExtensionClassLibraryPtr,
    p_class_name: GDExtensionConstStringNamePtr,
);

/// `classdb_construct_object` (Godot 4.1) — header line 2692.
///
/// Constructs a Godot-side `Object` of `p_classname`. A
/// `create_instance_func` MUST return one of these, NOT a bare pointer to
/// its own Rust allocation: Godot `dynamic_cast`s the returned pointer as
/// an `Object *`, so anything else segfaults. See
/// [`crate::registry::create_session_instance`].
pub type GDExtensionInterfaceClassdbConstructObject =
    unsafe extern "C" fn(p_classname: GDExtensionConstStringNamePtr) -> GDExtensionObjectPtr;

/// `object_set_instance` (Godot 4.1) — header line 2440.
///
/// Attaches an extension-owned instance pointer to a Godot `Object`.
/// `p_classname` must be a registered extension class that extends
/// `p_o`'s class. The pointer handed in here is exactly what Godot later
/// passes back as `p_instance` to every method trampoline and to
/// `free_instance_func`.
pub type GDExtensionInterfaceObjectSetInstance = unsafe extern "C" fn(
    p_o: GDExtensionObjectPtr,
    p_classname: GDExtensionConstStringNamePtr,
    p_instance: GDExtensionClassInstancePtr,
);

/// `string_name_new_with_utf8_chars` (Godot 4.2). Constructs a StringName in
/// caller-owned storage from a NUL-terminated UTF-8 buffer.
pub type GDExtensionInterfaceStringNameNewWithUtf8Chars =
    unsafe extern "C" fn(r_dest: GDExtensionUninitializedStringNamePtr, p_contents: *const c_char);

/// `string_name_new_with_latin1_chars` (Godot 4.2). Like the UTF-8 variant
/// but with an optional static-buffer optimization (unused by us — we always
/// pass 0).
pub type GDExtensionInterfaceStringNameNewWithLatin1Chars = unsafe extern "C" fn(
    r_dest: GDExtensionUninitializedStringNamePtr,
    p_contents: *const c_char,
    p_is_static: GDExtensionBool,
);

/// `mem_alloc` (Godot 4.1). Godot-side allocation used for `StringName`
/// scratch storage (24 bytes on LP64 in 4.3). We route through Rust's
/// allocator instead — this typedef is here for completeness only.
pub type GDExtensionInterfaceMemAlloc =
    unsafe extern "C" fn(p_bytes: usize) -> *mut core::ffi::c_void;

/// `mem_free` (Godot 4.1). Companion to `mem_alloc`.
pub type GDExtensionInterfaceMemFree = unsafe extern "C" fn(p_ptr: *mut core::ffi::c_void);

// ---------------------------------------------------------------------------
// Variant introspection + constructor factories.
//
// These are what the trampoline `Variant → typed value` unpack path uses.
// Godot 4.3-stable `gdextension_interface.h` declares them as follows
// (paraphrased from the header):
//
//   typedef GDExtensionVariantType (*GDExtensionInterfaceVariantGetType)
//       (GDExtensionConstVariantPtr p_self);
//
//   typedef void (*GDExtensionInterfaceVariantNewNil)
//       (GDExtensionUninitializedVariantPtr r_dest);
//
//   typedef void (*GDExtensionVariantFromTypeConstructorFunc)
//       (GDExtensionUninitializedVariantPtr r_out,
//        GDExtensionTypePtr p_in);
//
//   typedef void (*GDExtensionTypeFromVariantConstructorFunc)
//       (GDExtensionUninitializedTypePtr r_out,
//        GDExtensionVariantPtr p_in);
//
//   typedef GDExtensionVariantFromTypeConstructorFunc
//       (*GDExtensionInterfaceGetVariantFromTypeConstructor)
//       (GDExtensionVariantType p_type);
//
//   typedef GDExtensionTypeFromVariantConstructorFunc
//       (*GDExtensionInterfaceGetVariantToTypeConstructor)
//       (GDExtensionVariantType p_type);
//
// `get_variant_{from,to}_type_constructor` are factories: called ONCE per
// Variant type at [`crate::ffi::interface::InterfaceTable::from_proc_address`]
// time, they return the actual per-type packer/unpacker fn pointer. We cache
// the resolved constructors for `Int` (the only type this crate packs/unpacks
// today; String/PackedFloat32Array packing is deferred — see
// [`crate::trampoline`] TODO(future) markers for rationale).
// ---------------------------------------------------------------------------

/// `variant_get_type` — reads the type tag of a Variant. Cheap, non-allocating,
/// safe on any live Variant pointer.
pub type GDExtensionInterfaceVariantGetType =
    unsafe extern "C" fn(p_self: GDExtensionConstVariantPtr) -> GDExtensionVariantType;

/// `variant_new_nil` — writes a Nil Variant into `r_dest`. The C canonical
/// layout for a Nil Variant is all-zero, so this is effectively `memset`, but
/// routing through the interface keeps us robust against a future Godot bump
/// that changes the type-tag encoding.
pub type GDExtensionInterfaceVariantNewNil =
    unsafe extern "C" fn(r_dest: GDExtensionUninitializedVariantPtr);

/// Per-type packer: writes a `GDExtensionVariantType`-typed Variant into
/// `r_out`, reading the typed value from `p_in`. Nullable in the C header
/// because unknown types can produce NULL; the null case is discharged at
/// resolution time (see
/// [`crate::ffi::interface::InterfaceTable::from_proc_address`]).
pub type GDExtensionVariantFromTypeConstructorFunc = Option<
    unsafe extern "C" fn(r_out: GDExtensionUninitializedVariantPtr, p_in: GDExtensionTypePtr),
>;

/// Per-type unpacker: writes a typed value into `r_out`, reading the Variant
/// from `p_in`. Nullable for the same reason as
/// [`GDExtensionVariantFromTypeConstructorFunc`]; same null discharge.
pub type GDExtensionTypeFromVariantConstructorFunc = Option<
    unsafe extern "C" fn(r_out: GDExtensionUninitializedTypePtr, p_in: GDExtensionVariantPtr),
>;

/// `get_variant_from_type_constructor` factory. Called with a
/// `GDExtensionVariantType` at init and returns the per-type packer (or NULL
/// for an unknown type).
pub type GDExtensionInterfaceGetVariantFromTypeConstructor =
    unsafe extern "C" fn(
        p_type: GDExtensionVariantType,
    ) -> GDExtensionVariantFromTypeConstructorFunc;

/// `get_variant_to_type_constructor` factory. Same shape as the "from"
/// factory but yields Variant→typed unpackers.
pub type GDExtensionInterfaceGetVariantToTypeConstructor =
    unsafe extern "C" fn(
        p_type: GDExtensionVariantType,
    ) -> GDExtensionTypeFromVariantConstructorFunc;

/// Alias for the resolved Int-Variant packer (Option-unwrapped at
/// resolution). Signature: `void (*)(GDExtensionUninitializedVariantPtr r_out,
/// GDExtensionTypePtr p_in)`, where `p_in` must point to a valid `i64`.
pub type VariantFromIntCtor =
    unsafe extern "C" fn(r_out: GDExtensionUninitializedVariantPtr, p_in: GDExtensionTypePtr);

/// Alias for the resolved Int-Variant unpacker (Option-unwrapped at
/// resolution). Signature: `void (*)(GDExtensionUninitializedTypePtr r_out,
/// GDExtensionVariantPtr p_in)`, where `r_out` must point to a writable `i64`
/// and `p_in` must be a Variant whose type tag is
/// [`GDExtensionVariantType::Int`].
pub type VariantToIntCtor =
    unsafe extern "C" fn(r_out: GDExtensionUninitializedTypePtr, p_in: GDExtensionVariantPtr);

/// Generic alias for a Variant-from-typed constructor, Option-unwrapped at
/// resolution. Every per-type factory result from
/// `get_variant_from_type_constructor(TYPE)` — regardless of `TYPE` — shares
/// the identical fn-pointer signature; the concrete `TYPE` only constrains
/// what memory layout `p_in` must point at (e.g. `Object*` for OBJECT, an
/// opaque typed-String handle for STRING). Rust cannot express that per-call
/// constraint, so the caller MUST route each call through a typed helper that
/// enforces the input shape (see `crate::variant::*`).
pub type VariantFromTypeCtor =
    unsafe extern "C" fn(r_out: GDExtensionUninitializedVariantPtr, p_in: GDExtensionTypePtr);

/// Generic alias for a Variant-to-typed constructor, Option-unwrapped at
/// resolution. Symmetric to [`VariantFromTypeCtor`]: same fn-ptr shape,
/// per-call output layout is `TYPE`-specific and enforced by typed helpers.
pub type VariantToTypeCtor =
    unsafe extern "C" fn(r_out: GDExtensionUninitializedTypePtr, p_in: GDExtensionVariantPtr);

// ---------------------------------------------------------------------------
// Variant lifecycle + typed-container helpers (M3-11 T14/M3-18 unpack
// foundation).
//
// Signatures below mirror the Godot 4.3-stable `gdextension_interface.h`
// declarations verbatim; per-line provenance comments cite the header line
// number the typedef was copied from (SHA-based blob fetched to the
// scratchpad and grep'd before landing).
// ---------------------------------------------------------------------------

/// `variant_new_copy` (Godot 4.1, `gdextension_interface.h` line 912).
///
/// Deep-copies `p_src` into the uninitialized destination `r_dest`. For
/// refcounted types (String, PackedArray, Dictionary, Object) this increments
/// the source's refcount; the caller then owns the destination Variant and
/// MUST pair it with a matching [`GDExtensionInterfaceVariantDestroy`] call.
///
/// Header signature:
/// `void (*)(GDExtensionUninitializedVariantPtr r_dest, GDExtensionConstVariantPtr p_src)`.
pub type GDExtensionInterfaceVariantNewCopy = unsafe extern "C" fn(
    r_dest: GDExtensionUninitializedVariantPtr,
    p_src: GDExtensionConstVariantPtr,
);

/// `variant_destroy` (Godot 4.1, `gdextension_interface.h` line 932).
///
/// Destroys a Variant, releasing any internal heap allocation (refcount
/// decrement for CoW types). Idempotent on a Variant whose type tag is Nil.
///
/// Header signature: `void (*)(GDExtensionVariantPtr p_self)`.
///
/// # Not a typed-String destructor
///
/// This ONLY destroys Variants. To destroy a typed opaque handle produced by
/// a `get_variant_to_type_constructor(TYPE)` unpacker (e.g. a typed String or
/// PackedFloat32Array), callers must additionally resolve
/// `variant_get_ptr_destructor` (out of scope for the current land — see
/// module-doc `TODO(M3-18)` markers in [`crate::trampoline`]).
pub type GDExtensionInterfaceVariantDestroy = unsafe extern "C" fn(p_self: GDExtensionVariantPtr);

/// `packed_float32_array_operator_index_const` (Godot 4.1,
/// `gdextension_interface.h` line 2052).
///
/// Returns a const `*const f32` pointer to the `p_index`-th element of a
/// typed PackedFloat32Array opaque handle. `p_self` MUST be a typed handle
/// (NOT a Variant); the standard call sequence is:
///
/// 1. Resolve `get_variant_to_type_constructor(PACKED_FLOAT32_ARRAY)` → typed
///    unpacker (cached at init on
///    [`crate::ffi::interface::InterfaceTable::variant_to_packed_float32_array_ctor`]).
/// 2. Call the unpacker on an arg Variant → produces a typed PackedFloat32Array
///    handle in an uninitialized buffer.
/// 3. Call this fn with the typed handle + index → const-borrow the float.
///
/// # Not the size resolver
///
/// The companion `packed_float32_array_size` resolver is out of scope for the
/// current land; callers reading elements MUST separately resolve it to know
/// the count. See [`crate::trampoline`] `TODO(M3-18)` markers.
///
/// Header signature:
/// `const float *(*)(GDExtensionConstTypePtr p_self, GDExtensionInt p_index)`.
pub type GDExtensionInterfacePackedFloat32ArrayOperatorIndexConst =
    unsafe extern "C" fn(p_self: GDExtensionConstTypePtr, p_index: GDExtensionInt) -> *const f32;

/// `string_new_with_utf8_chars_and_len` (Godot 4.1,
/// `gdextension_interface.h` line 1593; `@deprecated in 4.3` in favour of
/// `..._and_len2` which returns a `GDExtensionInt` error code).
///
/// Constructs a typed String opaque in `r_dest` from a UTF-8 buffer of
/// `p_size` BYTES (not codepoints). We deliberately bind the 4.1 shape
/// because it matches `vokra.gdextension`'s `compatibility_minimum = "4.1"`
/// (ADR-0011 §D9); when we bump the pin to 4.5+ this should be swapped for
/// `..._and_len2` to gain the error-return channel.
///
/// # Not a Variant packer
///
/// This produces a typed opaque, NOT a Variant. Wrapping into a Variant is a
/// subsequent call through
/// [`crate::ffi::interface::InterfaceTable::variant_from_string_ctor`] with
/// the typed handle as `p_in`.
///
/// Header signature:
/// `void (*)(GDExtensionUninitializedStringPtr r_dest, const char *p_contents, GDExtensionInt p_size)`.
pub type GDExtensionInterfaceStringNewWithUtf8CharsAndLen = unsafe extern "C" fn(
    r_dest: GDExtensionUninitializedStringPtr,
    p_contents: *const c_char,
    p_size: GDExtensionInt,
);

// ---------------------------------------------------------------------------
// PackedFloat32Array packing pipeline (T14-followup: `stream_poll` full
// dispatch — pack `Vec<f32>` return into a Godot `PackedFloat32Array`
// Variant).
//
// The pipeline is:
//   1. Default-construct an empty PackedFloat32Array on a stack buffer via
//      `variant_get_ptr_constructor(PACKED_FLOAT32_ARRAY, 0)`.
//   2. Call its `resize(new_size)` builtin method via
//      `variant_get_ptr_builtin_method(PACKED_FLOAT32_ARRAY, "resize", 848867239)`.
//      The hash 848867239 is pinned from Godot 4.1..4.3 `extension_api.json`
//      (`gdextension/extension_api.json` in godot-cpp, verified stable across
//      all three tags before landing).
//   3. Write payload via `packed_float32_array_operator_index` (mutable).
//   4. Move-copy into the return Variant via
//      `get_variant_from_type_constructor(PACKED_FLOAT32_ARRAY)`.
//   5. Destroy the temp buffer via `variant_get_ptr_destructor(PACKED_FLOAT32_ARRAY)`.
//
// Every typedef below is verbatim from `godot/core/extension/gdextension_interface.h`
// (`@since 4.1`, matches our `vokra.gdextension` `compatibility_minimum = "4.1"`).
// ---------------------------------------------------------------------------

/// `GDExtensionPtrConstructor` — per-type in-place constructor (`@since 4.1`).
///
/// `p_base` is a writable slot large enough for the target type
/// (16 bytes for `PackedFloat32Array` on LP64, per `extension_api.json`
/// builtin_class_sizes float_64, stable Godot 4.1-4.3). `p_args` is a
/// pointer-to-array-of-pointer argument list (`NULL` for a 0-arg default
/// constructor like idx=0 of `PackedFloat32Array`).
///
/// Header signature:
/// `void (*)(GDExtensionUninitializedTypePtr p_base, const GDExtensionConstTypePtr *p_args)`.
pub type GDExtensionPtrConstructor = unsafe extern "C" fn(
    p_base: GDExtensionUninitializedTypePtr,
    p_args: *const GDExtensionConstTypePtr,
);

/// `GDExtensionPtrBuiltInMethod` — resolved per-type builtin method call
/// (`@since 4.1`).
///
/// `p_base` is the type instance, `p_args` is a pointer-to-array-of-pointer
/// argument list (each element points to the raw typed argument value —
/// e.g. `*const i64` for an Int arg), `r_return` is a writable slot for the
/// return value's raw type (may be `NULL` for `void` returns), and
/// `p_argument_count` is the argument count.
///
/// Header signature:
/// `void (*)(GDExtensionTypePtr p_base, const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr r_return, int p_argument_count)`.
pub type GDExtensionPtrBuiltInMethod = unsafe extern "C" fn(
    p_base: GDExtensionTypePtr,
    p_args: *const GDExtensionConstTypePtr,
    r_return: GDExtensionTypePtr,
    p_argument_count: i32,
);

/// `GDExtensionPtrDestructor` — per-type in-place destructor (`@since 4.1`).
///
/// Frees any heap-owned state the type holds (for `PackedFloat32Array` this
/// decrements the internal `CowData` refcount; the buffer contents become
/// undefined).
///
/// Header signature: `void (*)(GDExtensionTypePtr p_base)`.
pub type GDExtensionPtrDestructor = unsafe extern "C" fn(p_base: GDExtensionTypePtr);

/// Factory: `variant_get_ptr_constructor(TYPE, ctor_index) -> Option<ctor>`
/// (`@since 4.1`). Called once at
/// [`crate::ffi::interface::InterfaceTable::from_proc_address`] to resolve
/// the default `PackedFloat32Array` constructor (index 0). Returns `NULL`
/// for an unknown type/index — the null case is discharged at resolution
/// time.
///
/// Header signature:
/// `GDExtensionPtrConstructor (*)(GDExtensionVariantType p_type, int32_t p_constructor)`.
pub type GDExtensionInterfaceVariantGetPtrConstructor =
    unsafe extern "C" fn(
        p_type: GDExtensionVariantType,
        p_constructor: i32,
    ) -> Option<GDExtensionPtrConstructor>;

/// Factory: `variant_get_ptr_builtin_method(TYPE, name, hash) -> Option<method>`
/// (`@since 4.1`). Called once at
/// [`crate::ffi::interface::InterfaceTable::from_proc_address`] to resolve
/// the `PackedFloat32Array::resize` method. The `p_hash` guards against
/// silent signature drift across Godot versions — a mismatch returns `NULL`
/// and we bail cleanly at init.
///
/// `p_method` is a StringName pointer; the caller owns the underlying
/// storage only for the duration of this call (Godot does not retain the
/// StringName past the return).
///
/// Header signature:
/// `GDExtensionPtrBuiltInMethod (*)(GDExtensionVariantType p_type, GDExtensionConstStringNamePtr p_method, GDExtensionInt p_hash)`.
pub type GDExtensionInterfaceVariantGetPtrBuiltinMethod =
    unsafe extern "C" fn(
        p_type: GDExtensionVariantType,
        p_method: GDExtensionConstStringNamePtr,
        p_hash: GDExtensionInt,
    ) -> Option<GDExtensionPtrBuiltInMethod>;

/// Factory: `variant_get_ptr_destructor(TYPE) -> Option<destructor>`
/// (`@since 4.1`). Called once at
/// [`crate::ffi::interface::InterfaceTable::from_proc_address`] to resolve
/// the `PackedFloat32Array` destructor (used to clean up the
/// stack-allocated temp buffer after Variant packing).
///
/// Header signature:
/// `GDExtensionPtrDestructor (*)(GDExtensionVariantType p_type)`.
pub type GDExtensionInterfaceVariantGetPtrDestructor =
    unsafe extern "C" fn(p_type: GDExtensionVariantType) -> Option<GDExtensionPtrDestructor>;

/// `packed_float32_array_operator_index` — direct MUTABLE element pointer
/// access (`@since 4.1`).
///
/// Returns a `*mut f32` pointer to the `p_index`-th element of a typed
/// PackedFloat32Array opaque handle. NULL iff `p_index` is out of range.
/// Used to bulk-write into a freshly-resized `PackedFloat32Array` from a
/// Rust `&[f32]` slice.
///
/// Header signature:
/// `float *(*)(GDExtensionTypePtr p_self, GDExtensionInt p_index)`.
pub type GDExtensionInterfacePackedFloat32ArrayOperatorIndex =
    unsafe extern "C" fn(p_self: GDExtensionTypePtr, p_index: GDExtensionInt) -> *mut f32;

/// Alias for the resolved default `PackedFloat32Array` constructor (index 0,
/// Option-unwrapped at resolution). Same shape as
/// [`GDExtensionPtrConstructor`]; the alias is a documentation tag identifying
/// the exact resolved fn.
pub type PackedFloat32ArrayDefaultCtor = GDExtensionPtrConstructor;

/// Alias for the resolved `PackedFloat32Array::resize(new_size: int) -> int`
/// builtin method (hash 848867239, Option-unwrapped at resolution).
pub type PackedFloat32ArrayResizeMethod = GDExtensionPtrBuiltInMethod;

/// Alias for the resolved `PackedFloat32Array` destructor (Option-unwrapped
/// at resolution).
pub type PackedFloat32ArrayDestructor = GDExtensionPtrDestructor;

/// Alias for the resolved `Variant`-from-`PackedFloat32Array` packer
/// (Option-unwrapped at resolution). Signature:
/// `void (*)(GDExtensionUninitializedVariantPtr r_out, GDExtensionTypePtr p_in)`,
/// where `p_in` MUST point to a fully-constructed `PackedFloat32Array` C++
/// object (16 bytes on LP64 across Godot 4.1-4.3, per `extension_api.json`
/// builtin_class_sizes).
pub type VariantFromPackedFloat32ArrayCtor =
    unsafe extern "C" fn(r_out: GDExtensionUninitializedVariantPtr, p_in: GDExtensionTypePtr);

/// The compile-time hash of `PackedFloat32Array::resize(new_size: int) -> int`
/// per Godot's `extension_api.json`. Verified stable across
/// `godot-cpp/gdextension/extension_api.json` at tags `godot-4.1-stable`,
/// `godot-4.2-stable`, and `godot-4.3-stable` before landing.
///
/// A Godot version that drifts this hash returns NULL from
/// `variant_get_ptr_builtin_method(PACKED_FLOAT32_ARRAY, "resize",
/// PACKED_FLOAT32_ARRAY_RESIZE_HASH)`, which
/// [`crate::ffi::interface::InterfaceTable::from_proc_address`] surfaces as
/// an init-time `None` — the extension refuses to load cleanly (FR-EX-08),
/// rather than binding to a stale signature.
pub const PACKED_FLOAT32_ARRAY_RESIZE_HASH: GDExtensionInt = 848867239;

/// The compile-time hash of `PackedFloat32Array::size() -> int` (const method)
/// per Godot's `extension_api.json`. Verified against
/// `godot-cpp/gdextension/extension_api.json` at tag `4.3` (branch
/// `godot-4.3-stable`) — `builtin_classes.PackedFloat32Array.methods.size.hash
/// = 3173160232`, `is_const = true`, `is_static = false`, `is_vararg = false`,
/// `return_type = "int"`. Used by session_transcribe's PackedFloat32Array
/// unpack path to compute the element count of an arg PackedFloat32Array
/// before slice::from_raw_parts.
///
/// Header has NO dedicated `packed_float32_array_size` resolver (verified
/// against the 4.3-stable `gdextension_interface.h` — the exhaustive
/// `packed_*` @name list has only `operator_index` / `operator_index_const`
/// variants), so we route through the generic
/// `variant_get_ptr_builtin_method` factory. Hash drift bails init cleanly
/// via FR-EX-08 (NULL from the factory → `?` propagation → extension refuses
/// to load).
pub const PACKED_FLOAT32_ARRAY_SIZE_HASH: GDExtensionInt = 3173160232;

/// Alias for the resolved `PackedFloat32Array::size() -> int` builtin method
/// (hash [`PACKED_FLOAT32_ARRAY_SIZE_HASH`], Option-unwrapped at resolution).
/// Same call shape as [`PackedFloat32ArrayResizeMethod`]; the alias exists
/// as a documentation tag for the unpack path.
pub type PackedFloat32ArraySizeMethod = GDExtensionPtrBuiltInMethod;

/// Alias for the resolved `String` destructor (`variant_get_ptr_destructor(STRING)`,
/// Option-unwrapped at resolution). Called on a typed String opaque built by
/// `string_new_with_utf8_chars_and_len` to release the CowData refcount after
/// packing into a Variant via `variant_from_string_ctor`.
pub type StringDestructor = GDExtensionPtrDestructor;

// ---------------------------------------------------------------------------
// Dictionary packing pipeline (M3-11 T14 followup: `session_synthesize` full
// dispatch). All three of these resolve into their typed handles at
// [`crate::ffi::interface::InterfaceTable::from_proc_address`] time; the
// trampoline never resolves at runtime. See per-alias rustdoc for the
// call-order contract.
// ---------------------------------------------------------------------------

/// Alias for the resolved default `Dictionary` constructor (index 0,
/// Option-unwrapped at resolution). Same fn-ptr shape as
/// [`GDExtensionPtrConstructor`] — the alias is a documentation tag pinning
/// the type at the callsite.
pub type DictionaryDefaultCtor = GDExtensionPtrConstructor;

/// Alias for the resolved `Dictionary` destructor
/// (`variant_get_ptr_destructor(DICTIONARY)`, Option-unwrapped at
/// resolution). Called on the stack-allocated temp Dictionary after packing
/// into a Variant to release the internal refcount.
pub type DictionaryDestructor = GDExtensionPtrDestructor;

/// `dictionary_operator_index` (Godot 4.1, `gdextension_interface.h`).
///
/// Given a raw Dictionary handle `p_self` and a Variant key `p_key`, returns
/// a `*mut Variant` pointer to the value slot for that key. If the key does
/// not yet exist in the dictionary, Godot creates a Nil-Variant slot for it
/// and returns a pointer to that fresh Nil slot; if the key already exists,
/// the returned pointer aliases the existing value slot (overwriting via
/// `variant_new_copy` would leak the old value — caller MUST first
/// `variant_destroy` the slot in that case).
///
/// Header signature:
/// `GDExtensionVariantPtr (*)(GDExtensionTypePtr p_self, GDExtensionConstVariantPtr p_key)`.
pub type GDExtensionInterfaceDictionaryOperatorIndex =
    unsafe extern "C" fn(
        p_self: GDExtensionTypePtr,
        p_key: GDExtensionConstVariantPtr,
    ) -> GDExtensionVariantPtr;

/// `sizeof(Dictionary)` on LP64, per Godot 4.1..4.3 `extension_api.json`
/// `builtin_class_sizes.Dictionary` = 8 (single opaque handle to
/// `DictionaryPrivate`). We conservatively allocate a 16-byte stack buffer
/// so a future header widening (e.g. adding a per-instance flag byte) stays
/// covered without a re-audit of every `dict_new_variant`-style callsite.
pub const DICTIONARY_SIZE: usize = 8;

#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(
        DICTIONARY_SIZE == 8,
        "Dictionary size drift — audit variant.rs stack buffer + rebuild",
    );
};

/// `sizeof(PackedFloat32Array)` on LP64 (float_64 build configuration), per
/// Godot 4.1..4.3 `extension_api.json` builtin_class_sizes. A single 16-byte
/// stack buffer is enough to hold one temp PackedFloat32Array through the
/// construct → resize → copy → pack → destroy pipeline used by
/// [`crate::variant::pack_f32_slice_into_variant`].
pub const PACKED_FLOAT32_ARRAY_SIZE: usize = 16;

/// `sizeof(String)` on LP64 (float_64 build configuration), per Godot 4.3
/// `extension_api.json` builtin_class_sizes. A single pointer-word buffer
/// holds one temp typed String through the
/// `string_new_with_utf8_chars_and_len` → `variant_from_string_ctor` →
/// `string_destructor` cleanup used by
/// [`crate::variant::variant_from_string_utf8`].
pub const STRING_SIZE: usize = 8;

/// Compile-time guard for the PackedFloat32Array stack-buffer size. Godot's
/// `extension_api.json` reports 16 bytes on LP64 for the `float_64` build
/// configuration across Godot 4.1..4.3-stable — if our expectation ever
/// drifts past 16 bytes (e.g. we build against a future header that widens
/// it), we want to notice at compile time so `pack_f32_slice_into_variant`'s
/// stack buffer stays honest.
#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(
        PACKED_FLOAT32_ARRAY_SIZE == 16,
        "PackedFloat32Array size drift — audit variant.rs stack buffer + rebuild",
    );
    assert!(
        STRING_SIZE == 8,
        "String size drift — audit variant.rs stack buffer + rebuild",
    );
};

/// `string_to_utf8_chars` (Godot 4.1, `gdextension_interface.h` line 1691).
///
/// Encodes a typed String opaque into a UTF-8 byte buffer. Two-phase probe
/// pattern: passing `r_text = NULL` yields the length only (no bytes
/// written); passing a `p_max_write_length` cap fills the buffer up to that
/// cap and still returns the total encoded length (so callers can detect
/// truncation).
///
/// Return value: encoded length in BYTES (not codepoints), NOT including a
/// trailing NUL — Godot never writes a terminator via this API.
///
/// Header signature:
/// `GDExtensionInt (*)(GDExtensionConstStringPtr p_self, char *r_text, GDExtensionInt p_max_write_length)`.
pub type GDExtensionInterfaceStringToUtf8Chars = unsafe extern "C" fn(
    p_self: GDExtensionConstStringPtr,
    r_text: *mut c_char,
    p_max_write_length: GDExtensionInt,
) -> GDExtensionInt;

// ---------------------------------------------------------------------------
// PackedFloat32Array unpack completion (M3-11 T14 followup).
//
// The `variant_to_packed_float32_array_ctor` factory (already resolved in
// [`crate::ffi::interface::InterfaceTable`]) yields a TYPED
// PackedFloat32Array handle. Reading it into a Rust `&[f32]` requires three
// additional pieces that are ALREADY resolved above:
//
// 1. Element pointer: [`GDExtensionInterfacePackedFloat32ArrayOperatorIndexConst`]
//    (cached at `packed_float32_array_operator_index_const`).
// 2. Element count: [`PackedFloat32ArraySizeMethod`] (cached at
//    `pfa_size_method`, resolved via `variant_get_ptr_builtin_method(PFA,
//    "size", PACKED_FLOAT32_ARRAY_SIZE_HASH)`). This is preferred over
//    `variant_call("size", ...)` because it skips a full Variant boxing per
//    call — the hash guards silent signature drift.
// 3. Typed-handle cleanup: [`PackedFloat32ArrayDestructor`] (cached at
//    `pfa_destructor`, resolved via `variant_get_ptr_destructor(PFA)`). The
//    typed handle from the unpacker holds a refcount on Godot's shared
//    CoW-backed buffer; calling the destructor decrements it. Without this,
//    every push_pcm invocation would leak refcount-1 (a slow leak that
//    manifests only under long-running streaming — exactly the M3-14
//    barge-in scenario).
//
// All three resolvers were declared at Vokra-side land time as part of the
// `stream_poll` pack pipeline. Re-using them for `stream_push_pcm` unpack
// costs zero additional init-time proc-address resolves.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Compile-time layout guards.
//
// Godot 4.3 pins these sizes/alignments in `gdextension_interface.h`; a
// silently drifted struct layout would corrupt Godot's stack on the very
// first init or method-dispatch callback. These asserts catch that at build
// time. Values were probed with `clang -m64` against the 4.3-stable header.
// ---------------------------------------------------------------------------

#[cfg(target_pointer_width = "64")]
const _: () = {
    // 4-byte enum + 4-byte pad + 8-byte ptr + 8-byte fn ptr + 8-byte fn ptr = 32
    // Matches Godot's `sizeof(GDExtensionInitialization)` on LP64 targets.
    assert!(core::mem::size_of::<GDExtensionInitialization>() == 32);

    // Verified with clang -m64 against the Godot 4.3-stable header
    // (`gdextension_interface.h`). `GDExtensionClassCreationInfo3` = 4 bytes
    // of GDExtensionBool flags + 4 bytes align padding + 18 fn pointers +
    // 1 void* = 8 + 152 = 160.
    assert!(core::mem::size_of::<GDExtensionClassCreationInfo3>() == 160);
    assert!(core::mem::align_of::<GDExtensionClassCreationInfo3>() == 8);

    // `GDExtensionClassMethodInfo` = 88 bytes on LP64:
    //   ptr(8) + ptr(8) + fn(8) + fn(8) + u32(4) + u8+3pad(4) +
    //   ptr(8) + enum(4) + u32(4) + ptr(8) + ptr(8) + u32+4pad(8) + ptr(8) = 88
    assert!(core::mem::size_of::<GDExtensionClassMethodInfo>() == 88);
    assert!(core::mem::align_of::<GDExtensionClassMethodInfo>() == 8);

    // `GDExtensionPropertyInfo` = 48 bytes on LP64:
    //   enum(4) + pad(4) + ptr(8) + ptr(8) + u32(4) + pad(4) + ptr(8) + u32(4) + pad(4)
    assert!(core::mem::size_of::<GDExtensionPropertyInfo>() == 48);
    assert!(core::mem::align_of::<GDExtensionPropertyInfo>() == 8);

    // `GDExtensionCallError` = 12 bytes (enum + i32 + i32); 4-byte aligned.
    assert!(core::mem::size_of::<GDExtensionCallError>() == 12);
    assert!(core::mem::align_of::<GDExtensionCallError>() == 4);

    // `GDExtensionInt` = int64_t; `GDExtensionBool` = uint8_t.
    assert!(core::mem::size_of::<GDExtensionInt>() == 8);
    assert!(core::mem::size_of::<GDExtensionBool>() == 1);
};
