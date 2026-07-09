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
    GDExtensionInterfaceClassdbRegisterExtensionClass3,
    GDExtensionInterfaceClassdbRegisterExtensionClassMethod,
    GDExtensionInterfaceClassdbRegisterExtensionClassSignal,
    GDExtensionInterfaceClassdbUnregisterExtensionClass, GDExtensionInterfaceGetProcAddress,
    GDExtensionInterfaceMemAlloc, GDExtensionInterfaceMemFree,
    GDExtensionInterfaceStringNameNewWithLatin1Chars,
    GDExtensionInterfaceStringNameNewWithUtf8Chars,
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
    pub string_name_new_with_utf8_chars: GDExtensionInterfaceStringNameNewWithUtf8Chars,
    pub string_name_new_with_latin1_chars: GDExtensionInterfaceStringNameNewWithLatin1Chars,
    pub mem_alloc: GDExtensionInterfaceMemAlloc,
    pub mem_free: GDExtensionInterfaceMemFree,
}

/// The list of names we resolve. Each entry MUST end with a NUL byte
/// (`get_proc_address` takes a C string).
mod names {
    pub const CLASSDB_REGISTER_EXTENSION_CLASS3: &[u8] = b"classdb_register_extension_class3\0";
    pub const CLASSDB_REGISTER_EXTENSION_CLASS_METHOD: &[u8] =
        b"classdb_register_extension_class_method\0";
    pub const CLASSDB_REGISTER_EXTENSION_CLASS_SIGNAL: &[u8] =
        b"classdb_register_extension_class_signal\0";
    pub const CLASSDB_UNREGISTER_EXTENSION_CLASS: &[u8] = b"classdb_unregister_extension_class\0";
    pub const STRING_NAME_NEW_WITH_UTF8_CHARS: &[u8] = b"string_name_new_with_utf8_chars\0";
    pub const STRING_NAME_NEW_WITH_LATIN1_CHARS: &[u8] = b"string_name_new_with_latin1_chars\0";
    pub const MEM_ALLOC: &[u8] = b"mem_alloc\0";
    pub const MEM_FREE: &[u8] = b"mem_free\0";
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
            let raw_sn_utf8 =
                get_proc_address(names::STRING_NAME_NEW_WITH_UTF8_CHARS.as_ptr() as *const c_char)?;
            let raw_sn_latin1 = get_proc_address(
                names::STRING_NAME_NEW_WITH_LATIN1_CHARS.as_ptr() as *const c_char,
            )?;
            let raw_alloc = get_proc_address(names::MEM_ALLOC.as_ptr() as *const c_char)?;
            let raw_free = get_proc_address(names::MEM_FREE.as_ptr() as *const c_char)?;

            // Transmute each opaque fn pointer to its typed signature.
            // Layout guarantee: `Option<unsafe extern "C" fn()>` is
            // `#[repr(transparent)]` over a raw fn pointer, so a NULL
            // Option decodes to a NULL fn pointer. `?` above already
            // discharged the NULL case.
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
            })
        }
    }
}

#[cfg(test)]
mod tests {
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

    /// Mock resolver: returns a canned fn pointer for every request, so
    /// `from_proc_address` succeeds and every field of the returned table
    /// is populated. Verifies the resolver plumbing doesn't accidentally
    /// short-circuit any name.
    #[test]
    fn from_proc_address_populates_every_field_when_resolver_returns_fn() {
        // Sentinel function — never actually called; used only for its
        // address (all fields of the mocked table point to it).
        unsafe extern "C" fn sentinel() {}

        unsafe extern "C" fn mock_gpa(
            _p_name: *const c_char,
        ) -> crate::ffi::gdextension::GDExtensionInterfaceFunctionPtr {
            Some(sentinel)
        }

        // SAFETY: `mock_gpa` is a live fn pointer matching the resolver
        // signature; each string constant is NUL-terminated (checked in
        // `every_resolver_name_is_nul_terminated`).
        let table = unsafe { InterfaceTable::from_proc_address(mock_gpa) }
            .expect("resolver returning Some for every name must yield a table");

        // Every field must dispatch to the sentinel — cast to raw fn ptr
        // and compare addresses.
        let expected = sentinel as *const () as usize;
        assert_eq!(table.classdb_register_extension_class3 as usize, expected);
        assert_eq!(
            table.classdb_register_extension_class_method as usize,
            expected
        );
        assert_eq!(
            table.classdb_register_extension_class_signal as usize,
            expected
        );
        assert_eq!(table.classdb_unregister_extension_class as usize, expected);
        assert_eq!(table.string_name_new_with_utf8_chars as usize, expected);
        assert_eq!(table.string_name_new_with_latin1_chars as usize, expected);
        assert_eq!(table.mem_alloc as usize, expected);
        assert_eq!(table.mem_free as usize, expected);
    }

    /// Mock resolver that returns NULL for a specific name. Verifies the
    /// resolver bails cleanly (returns `None`) instead of populating the
    /// table with NULL fn pointers.
    #[test]
    fn from_proc_address_returns_none_when_any_lookup_fails() {
        use core::sync::atomic::{AtomicU32, Ordering};

        // Each mock returns NULL on the Nth lookup and Some(sentinel)
        // otherwise. This exercises the `?` propagation.
        unsafe extern "C" fn sentinel() {}

        // 8 slots; index into `[u8; 8]` = 1 iff this slot returns NULL.
        static NULL_MASK: AtomicU32 = AtomicU32::new(0);
        static COUNTER: AtomicU32 = AtomicU32::new(0);

        unsafe extern "C" fn mock_gpa(
            _p_name: *const c_char,
        ) -> crate::ffi::gdextension::GDExtensionInterfaceFunctionPtr {
            let n = COUNTER.fetch_add(1, Ordering::SeqCst);
            let mask = NULL_MASK.load(Ordering::SeqCst);
            if (mask >> n) & 1 == 1 {
                None
            } else {
                Some(sentinel)
            }
        }

        // Test failing on every one of the 8 positions individually.
        for pos in 0..8 {
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
}
