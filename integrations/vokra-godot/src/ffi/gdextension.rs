//! Minimal `extern "C"` bindings for `gdextension_interface.h` (Godot 4.1+
//! MIT, ADR-0011 §D3). Only the types/enums that appear on the initialization
//! path (T04) and on the future class-registration path (T05..T09) are bound
//! here. The Godot header is intentionally NOT copied into this crate — we
//! declare the pinned subset we actually use, so incremental additions to the
//! header downstream do not force churn on our binding.

use core::ffi::c_char;

/// GDExtension boolean is an `uint8_t` (0/1), matches
/// `typedef uint8_t GDExtensionBool` in `gdextension_interface.h`.
pub type GDExtensionBool = u8;

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
// Compile-time layout guards.
//
// Godot 4.x pins these sizes/alignments in `gdextension_interface.h`; a
// silently drifted struct layout would corrupt Godot's stack on the very
// first init callback. These asserts catch that at build time.
// ---------------------------------------------------------------------------

#[cfg(target_pointer_width = "64")]
const _: () = {
    // 4-byte enum + 4-byte pad + 8-byte ptr + 8-byte fn ptr + 8-byte fn ptr = 32
    // Matches Godot's `sizeof(GDExtensionInitialization)` on LP64 targets.
    assert!(core::mem::size_of::<GDExtensionInitialization>() == 32);
};
