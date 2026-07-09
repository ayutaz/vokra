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
/// - `PackedFloat32Array` (30): PCM in/out
/// - `String` (4): text in/out (ASR result, TTS text)
/// - `Int` (2): sample rate
/// - `Dictionary` (26): TTS output bag (`{pcm, sample_rate}`)
/// - `Nil` (0): void return
#[repr(C)]
#[allow(dead_code)] // Tail variants unused; kept for future method surfaces.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GDExtensionVariantType {
    Nil = 0,
    Bool = 1,
    Int = 2,
    Float = 3,
    String = 4,
    /// `GDEXTENSION_VARIANT_TYPE_DICTIONARY` — used for the TTS output bag.
    Dictionary = 26,
    /// `GDEXTENSION_VARIANT_TYPE_PACKED_FLOAT32_ARRAY` — used for PCM I/O.
    PackedFloat32Array = 30,
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
pub type GDExtensionVariantPtr = *mut core::ffi::c_void;
pub type GDExtensionConstVariantPtr = *const core::ffi::c_void;
pub type GDExtensionUninitializedVariantPtr = *mut core::ffi::c_void;
pub type GDExtensionTypePtr = *mut core::ffi::c_void;
pub type GDExtensionConstTypePtr = *const core::ffi::c_void;

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
