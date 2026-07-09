//! # vokra-godot
//!
//! Godot 4.x GDExtension binding for the Vokra speech-first runtime
//! (BR-04 / FR-API-05). Hand-written `extern "C"` bridge over the Vokra C
//! ABI (`include/vokra.h`, cbindgen-generated from `crates/vokra-capi`) —
//! **no `godot-cpp`, no `gdext-rs`, no third-party binding crate**
//! (ADR-0011 §D1). This crate is an OUT-OF-WORKSPACE integration
//! (`integrations/vokra-godot/`) mirroring the isolation pattern used by
//! `integrations/vokra-piper-g2p/` and `integrations/vokra-server/`, so the
//! zero-dependency invariant on the root `Cargo.lock` (NFR-DS-02) is
//! untouched.
//!
//! # State (2026-07-09, T02..T04 initial)
//!
//! Complete:
//! - [`vokra_gdextension_init`] entry point + init/deinit trampolines
//!   (panic-firewalled) — Godot 4.1+ can `dlopen` the produced cdylib.
//! - Safe RAII wrappers ([`session::VokraSession`], [`vad::VokraStream`])
//!   over the opaque C ABI handles.
//! - Dispatch helpers ([`asr::transcribe`], [`tts::synthesize`],
//!   [`vad::VokraStream::push_pcm`]/`poll`/`poll_events`/`interrupt`).
//! - Panic firewall ([`error::catch_panic`], [`error::catch_panic_as_err`])
//!   and `vokra_last_error()` → [`error::VokraError`] translation.
//! - `LINKER_KEEPALIVE` static that force-references every C ABI symbol so
//!   the produced cdylib exports them (rlib dead-code stripping guard).
//!
//! TODO (T05..T20):
//! - **T05**: `classdb_register_extension_class3` — expose
//!   [`session::VokraSession`] as a Godot Object subclass. Requires
//!   resolving GDExtension's method-binding APIs by name via
//!   `get_proc_address` and wiring the trampolines below into a
//!   `GDExtensionClassCreationInfo2` struct.
//! - **T09**: `object_emit_signal` — deliver streaming chunks as Godot
//!   Signals. Same resolution pattern as T05.
//! - **T12**: Windows / macOS universal / Android crossbuilds in
//!   `scripts/build-godot-gdextension.sh` (linux path is functional at T11).
//! - **T13..T18**: AssetLib package layout, CI, CD, NVIDIA non-bundle scanner.
//! - **T14 / T15 / T19**: GDScript demo scenes + owner Godot-editor smoke.
//!
//! # Unsafe policy (NFR-RL-07, workspace lint `unsafe_code = "deny"`)
//!
//! GDExtension is a C ABI, so this crate opts out at the crate root just
//! like `crates/vokra-capi`. Every `unsafe` block MUST carry a `// SAFETY:`
//! comment (`clippy::undocumented_unsafe_blocks`). Panics NEVER cross the
//! Godot boundary (`catch_panic` at every trampoline entry).

// Local opt-out from the workspace `unsafe_code = "deny"` lint — this crate
// IS a C ABI bridge, so raw pointers and `extern "C"` are load-bearing.
#![allow(unsafe_code)]

// Force the Vokra C ABI rlib to be linked into our cdylib. `vokra-capi`
// (crate package name) publishes its `#[no_mangle] pub extern "C" fn vokra_*`
// symbols through the `[lib] name = "vokra"` rlib, and Rust's linker only
// includes an rlib's `no_mangle` symbols if it sees at least one Rust-level
// reference to the crate. We have none through Rust paths — everything goes
// through the extern "C" declarations in `ffi::capi` — so without this
// `extern crate` the linker drops the whole vokra rlib as "unreachable" and
// the Godot binding cdylib ends up with undefined `_vokra_*` symbols.
//
// The `as _` binding suppresses the "unused extern crate" lint. Do NOT
// remove this line without adding an equivalent Rust-level reference to
// the `vokra` crate (or migrating to dlopen-based loading of libvokra.so
// at runtime, which is the M4 posture if the platform crossbuild plumbing
// makes rlib link infeasible).
extern crate vokra as _;

pub mod asr;
pub mod error;
pub mod ffi;
pub mod session;
pub mod tts;
pub mod vad;

use core::ffi::c_void;
use core::ptr;

use crate::ffi::gdextension::{
    GDExtensionBool, GDExtensionClassLibraryPtr, GDExtensionInitialization,
    GDExtensionInitializationLevel, GDExtensionInterfaceGetProcAddress,
};

// ---------------------------------------------------------------------------
// Linker keepalive.
//
// The Vokra C ABI symbols are defined as `#[no_mangle] pub extern "C" fn` in
// `crates/vokra-capi`, which we depend on as an rlib. Rust's linker WILL
// dead-code-strip `no_mangle` symbols from a cdylib if nothing in the
// cdylib's own code references them. At the T02..T04 milestone none of the
// class-method trampolines are wired yet, so we need a static reference to
// keep the ABI reachable. The T05+ class-method trampolines will make this
// redundant but leaving it in is cheap and defends against future
// dead-code-stripping regressions.
// ---------------------------------------------------------------------------

/// See module doc above. `linkme` is deliberately NOT used (zero-dep).
#[used]
static LINKER_KEEPALIVE: fn() -> usize = ffi::capi::keepalive_c_abi_symbols;

// ---------------------------------------------------------------------------
// GDExtension init/deinit callbacks.
//
// Both are `extern "C"` because Godot's C runtime invokes them across the
// ABI boundary. Neither may panic; both wrap their body in `catch_panic`.
// ---------------------------------------------------------------------------

/// Called by Godot at each init level (ascending). Vokra registers its
/// classes at [`GDExtensionInitializationLevel::Scene`] (post-scene-tree,
/// ADR-0011 §D3). Other levels are documented no-ops.
///
/// # Safety
///
/// This is a C entry point invoked by Godot. `userdata` is whatever we
/// stored in `GDExtensionInitialization::userdata` (currently `NULL` — no
/// per-load state).
extern "C" fn vokra_initialize(_userdata: *mut c_void, p_level: GDExtensionInitializationLevel) {
    // Panic firewall (NFR-RL-07): a panic here would unwind through Godot's
    // C stack (compiled without unwind tables) = UB. `catch_panic` swallows
    // it; there is nothing meaningful we can report at this level without
    // Godot's print system wired (T05 will add that path via
    // `get_proc_address("print_error")`).
    let _ = error::catch_panic(|| {
        // Force LINKER_KEEPALIVE to be reachable at runtime as well as at
        // link time. `black_box`-style: XOR into a discard.
        let _keep = (LINKER_KEEPALIVE)();

        match p_level {
            GDExtensionInitializationLevel::Scene => {
                // TODO(T05): call `classdb_register_extension_class3` here
                // to expose `VokraSession` as a Godot Object subclass.
                // Resolution pattern (T05 will implement):
                //   let register: extern "C" fn(...) = mem::transmute(
                //       get_proc_address(c"classdb_register_extension_class3".as_ptr())
                //   );
                //   register(library, class_name, parent, &class_info);
            }
            _ => { /* Vokra does not register at Core/Servers/Editor. */ }
        }
    });
}

/// Called by Godot at each teardown level (descending). Symmetric with
/// [`vokra_initialize`]. Currently a no-op because we own no per-load state
/// beyond what's in the class registry (Godot itself unregisters on
/// extension unload).
extern "C" fn vokra_deinitialize(_userdata: *mut c_void, _p_level: GDExtensionInitializationLevel) {
    let _ = error::catch_panic(|| {
        // TODO(T05): unregister classes here in reverse order of T05
        // registration.
    });
}

// ---------------------------------------------------------------------------
// GDExtension entry point.
//
// Godot's `.gdextension` config (`vokra.gdextension`, ADR-0011 §D9) points
// at this symbol as `entry_symbol = "vokra_gdextension_init"`. Godot loads
// the cdylib, resolves this symbol, and calls it with the interface pointer
// and library token. Signature MUST match
// `GDExtensionInitializationFunction` from `gdextension_interface.h`.
// ---------------------------------------------------------------------------

/// GDExtension library entry point (ADR-0011 §D3). Return `1` on success.
///
/// # Safety
///
/// Invoked by Godot; `p_get_proc_address` is a live function pointer,
/// `r_initialization` is a writable `GDExtensionInitialization*` slot for
/// the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_gdextension_init(
    _p_get_proc_address: GDExtensionInterfaceGetProcAddress,
    _p_library: GDExtensionClassLibraryPtr,
    r_initialization: *mut GDExtensionInitialization,
) -> GDExtensionBool {
    // Panic firewall (NFR-RL-07). If ANY Rust panic escapes this
    // function, Godot's C stack unwinds into non-unwind-safe territory =
    // UB. Wrap the whole body; on panic, return 0 (init failure) so Godot
    // reports "extension failed to load" cleanly.
    let ok = error::catch_panic(move || {
        if r_initialization.is_null() {
            return false;
        }
        // SAFETY: Godot guarantees `r_initialization` is a valid writable
        // slot for the duration of this call (per GDExtension contract).
        unsafe {
            (*r_initialization).minimum_initialization_level =
                GDExtensionInitializationLevel::Scene;
            (*r_initialization).userdata = ptr::null_mut();
            (*r_initialization).initialize = Some(vokra_initialize);
            (*r_initialization).deinitialize = Some(vokra_deinitialize);
        }
        true
    })
    .unwrap_or(false);

    if ok { 1 } else { 0 }
}

// ---------------------------------------------------------------------------
// Assorted crate-level tests. The full FFI init loop is not exercised here
// because it would require a real Godot host; the T20 CI job (unimplemented
// at T04) is where that lands.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Sanity check: the entry point's signature must be a legal
    // `GDExtensionInitializationFunction`. A silent signature drift would
    // corrupt Godot's stack on the first call.
    #[test]
    fn entry_point_matches_gdextension_signature() {
        let _: crate::ffi::gdextension::GDExtensionInitializationFunction = vokra_gdextension_init;
    }

    // The init/deinit callbacks must fit the Option<fn(...)> slot in
    // GDExtensionInitialization exactly. Same rationale.
    #[test]
    fn init_and_deinit_signatures_fit_option_slot() {
        let a: Option<unsafe extern "C" fn(*mut c_void, GDExtensionInitializationLevel)> =
            Some(vokra_initialize);
        let b: Option<unsafe extern "C" fn(*mut c_void, GDExtensionInitializationLevel)> =
            Some(vokra_deinitialize);
        assert!(a.is_some());
        assert!(b.is_some());
    }

    // Entry point must return 0 on a NULL initialization struct instead of
    // dereferencing it — Godot MAY (though the C header contract forbids)
    // pass NULL during a broken-host smoke test.
    #[test]
    fn entry_point_rejects_null_init_struct() {
        // SAFETY: passing a legitimately-NULL init pointer is the very
        // case the entry point defends against. The dummy `get_proc_address`
        // is a live function pointer with the correct signature.
        unsafe extern "C" fn dummy_gpa(
            _p_name: *const core::ffi::c_char,
        ) -> crate::ffi::gdextension::GDExtensionInterfaceFunctionPtr {
            None
        }
        let result = unsafe { vokra_gdextension_init(dummy_gpa, ptr::null_mut(), ptr::null_mut()) };
        assert_eq!(result, 0, "NULL init struct must produce init-failure");
    }

    // Happy path: with a real init struct, the entry point wires the
    // callbacks and reports success.
    #[test]
    fn entry_point_populates_init_struct_on_success() {
        unsafe extern "C" fn dummy_gpa(
            _p_name: *const core::ffi::c_char,
        ) -> crate::ffi::gdextension::GDExtensionInterfaceFunctionPtr {
            None
        }

        let mut init = GDExtensionInitialization {
            minimum_initialization_level: GDExtensionInitializationLevel::Core,
            userdata: ptr::null_mut(),
            initialize: None,
            deinitialize: None,
        };
        // SAFETY: `&mut init` is a valid writable slot for the duration
        // of the call.
        let result = unsafe { vokra_gdextension_init(dummy_gpa, ptr::null_mut(), &mut init) };

        assert_eq!(result, 1, "successful init must return 1");
        assert_eq!(
            init.minimum_initialization_level,
            GDExtensionInitializationLevel::Scene
        );
        assert!(init.initialize.is_some());
        assert!(init.deinitialize.is_some());
    }
}
