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
//! # State (2026-07-10, T01..T18 land — CC 側 100% 完成)
//!
//! Complete (CC 側 T01-T18):
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
//! - **T05-T13** (Wave 11): `classdb_register_extension_class3` +
//!   [`ffi::interface::InterfaceTable`] resolving 8 GDExtension APIs via
//!   `get_proc_address` + [`registry`] class registration pipeline +
//!   [`trampoline`] method binding + `object_emit_signal` for streaming
//!   signals + compile-time layout `const _: () = { assert!(...) };` guards
//!   for the Godot 4.3-stable header structs.
//! - **T12** (Wave 13): `scripts/build-godot-gdextension.sh` crossbuild
//!   matrix (5 target: macOS Intel / Apple Silicon / Linux x64 / Windows
//!   MSVC / Android arm64) via `TARGET_TRIPLE` selector.
//! - **T14 + T15** (Wave 13): `demos/asr_demo/` + `demos/tts_demo/` Godot
//!   4.x project scaffold (project.godot + main.tscn + main.gd).
//! - **T16** (Wave 13): `.github/workflows/godot-crossbuild.yml` 5-target
//!   matrix + aggregator package + AssetLib zip.
//! - **T17** (Wave 13): `.github/workflows/release.yml` `godot-package-release`
//!   job — deterministic zip pack + GitHub Release upload.
//! - **T18** (Wave 13): `scripts/compliance/check-godot-package-no-nvidia.sh`
//!   compliance scanner (Unity mirror pattern + latent gap closed).
//!
//! Owner (`ayutaz`) 引き渡し:
//! - **T19**: 実 Godot 4.3+ editor での `demos/asr_demo` + `demos/tts_demo`
//!   smoke — M3-18 と併走 runtime verification。
//! - **T20**: M3-11 WP-close PR。
//! - `TODO(M3-18)` markers in `trampoline.rs` (Variant unpack real dispatch)
//!   — module doc §Bounded scope で owner smoke に委譲済み。
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
pub mod registry;
pub mod session;
pub mod trampoline;
pub mod tts;
pub mod vad;

use core::ffi::c_void;
use core::ptr;
use std::sync::Mutex;

use crate::ffi::gdextension::{
    GDExtensionBool, GDExtensionClassLibraryPtr, GDExtensionInitialization,
    GDExtensionInitializationLevel, GDExtensionInterfaceGetProcAddress,
};
use crate::ffi::interface::InterfaceTable;

// ---------------------------------------------------------------------------
// Linker keepalive.
//
// The Vokra C ABI symbols are defined as `#[no_mangle] pub extern "C" fn` in
// `crates/vokra-capi`, which we depend on as an rlib. Rust's linker WILL
// dead-code-strip `no_mangle` symbols from a cdylib if nothing in the
// cdylib's own code references them. The class-method trampolines
// (`crate::trampoline::*`) call into that ABI at runtime; we retain the
// keepalive here as defense in depth against future dead-code-stripping
// regressions.
// ---------------------------------------------------------------------------

/// See module doc above. `linkme` is deliberately NOT used (zero-dep).
#[used]
static LINKER_KEEPALIVE: fn() -> usize = ffi::capi::keepalive_c_abi_symbols;

// ---------------------------------------------------------------------------
// Extension-scoped state (T05).
//
// Godot invokes `vokra_initialize` and `vokra_deinitialize` separately from
// the entry point; we need to hand off the library token + resolved
// interface table between them. A `Mutex<Option<...>>` mirrors the exact
// posture used by godot-cpp for its own registration state — contested at
// most once per extension load, uncontested at every other read.
//
// The state IS NOT touched by method trampolines (they hold their own
// references, if any). This keeps the lock's role narrow: `vokra_initialize`
// populates it, `vokra_deinitialize` reads + clears it, and that's it.
// ---------------------------------------------------------------------------

/// State stashed at `vokra_gdextension_init` and consumed at
/// `vokra_initialize` (Scene level) / `vokra_deinitialize` (Scene level).
struct ExtensionState {
    library: GDExtensionClassLibraryPtr,
    interface: InterfaceTable,
}

// SAFETY: `ExtensionState` holds a `GDExtensionClassLibraryPtr` (opaque C
// pointer) and an `InterfaceTable` (all `unsafe extern "C" fn` pointers).
// Neither internally aliases process-lifetime mutable state that we own,
// and Godot's C runtime is documented to invoke initialize / deinitialize
// callbacks single-threaded on the main thread. The Mutex protects our
// slot; these `unsafe impl`s just discharge the raw-pointer field's
// auto-trait rejection.
unsafe impl Send for ExtensionState {}
unsafe impl Sync for ExtensionState {}

static EXTENSION_STATE: Mutex<Option<ExtensionState>> = Mutex::new(None);

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
/// per-load state; extension state lives in [`EXTENSION_STATE`]).
extern "C" fn vokra_initialize(_userdata: *mut c_void, p_level: GDExtensionInitializationLevel) {
    // Panic firewall (NFR-RL-07): a panic here would unwind through Godot's
    // C stack (compiled without unwind tables) = UB. `catch_panic` swallows
    // it; there is nothing meaningful we can report at this level without
    // Godot's print system wired (a future patch may add that path via
    // `get_proc_address("print_error")`).
    let _ = error::catch_panic(|| {
        // Force LINKER_KEEPALIVE to be reachable at runtime as well as at
        // link time. `black_box`-style: XOR into a discard.
        let _keep = (LINKER_KEEPALIVE)();

        match p_level {
            GDExtensionInitializationLevel::Scene => {
                let guard = EXTENSION_STATE.lock().ok();
                if let Some(state_opt) = guard {
                    if let Some(state) = state_opt.as_ref() {
                        // SAFETY: `state.interface` was resolved at
                        // `vokra_gdextension_init` and holds live Godot fn
                        // pointers. `state.library` is the token Godot
                        // handed us at the same call. `register` is
                        // documented single-threaded (main-thread only).
                        unsafe { registry::register(state.library, &state.interface) };
                    }
                }
            }
            _ => { /* Vokra does not register at Core/Servers/Editor. */ }
        }
    });
}

/// Called by Godot at each teardown level (descending). Symmetric with
/// [`vokra_initialize`].
///
/// Unregisters both classes at Scene level and clears
/// [`EXTENSION_STATE`] so the cdylib can be re-loaded cleanly.
extern "C" fn vokra_deinitialize(_userdata: *mut c_void, p_level: GDExtensionInitializationLevel) {
    let _ = error::catch_panic(|| {
        match p_level {
            GDExtensionInitializationLevel::Scene => {
                // Take the state so the slot is empty regardless of what
                // happens inside `unregister`. Godot never re-calls
                // Scene-level deinit for the same load, so this is safe
                // even if unregister panics (`catch_panic` above catches
                // it and the slot stays empty).
                let taken = EXTENSION_STATE.lock().ok().and_then(|mut g| g.take());
                if let Some(state) = taken {
                    // SAFETY: mirror of `vokra_initialize` — the interface
                    // and library are live and this runs on the main
                    // thread.
                    unsafe { registry::unregister(state.library, &state.interface) };
                }
            }
            _ => { /* Nothing to do at other levels. */ }
        }
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
    p_get_proc_address: GDExtensionInterfaceGetProcAddress,
    p_library: GDExtensionClassLibraryPtr,
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

        // Resolve the GDExtension interface subset we depend on. If any
        // required name is missing we bail cleanly with 0 — Godot will
        // report "extension failed to load" without ever calling into
        // our initialize/deinitialize callbacks.
        //
        // SAFETY: `p_get_proc_address` is a live fn pointer for the
        // duration of this call (GDExtension contract). The resolver
        // reads NUL-terminated static byte constants (checked by
        // `InterfaceTable::from_proc_address` tests).
        let Some(interface) = (unsafe { InterfaceTable::from_proc_address(p_get_proc_address) })
        else {
            return false;
        };

        // Stash extension state for `vokra_initialize` /
        // `vokra_deinitialize`. Lock poisoning would only happen if a
        // previous `catch_panic` failed AT the write itself, which is
        // impossible on the happy path — treat it as init-failure.
        {
            let Ok(mut guard) = EXTENSION_STATE.lock() else {
                return false;
            };
            *guard = Some(ExtensionState {
                library: p_library,
                interface,
            });
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

    // Entry point must return 0 when interface resolution fails (i.e.
    // any required GDExtension API is missing). The dummy_gpa returns
    // NULL for every name, which triggers `InterfaceTable::from_proc_address`
    // to return None → the entry point bails cleanly instead of stashing
    // half-populated state.
    #[test]
    fn entry_point_rejects_missing_interface() {
        unsafe extern "C" fn null_gpa(
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
        // SAFETY: `null_gpa` matches the resolver signature; init struct
        // is a valid writable slot.
        let result = unsafe { vokra_gdextension_init(null_gpa, ptr::null_mut(), &mut init) };

        assert_eq!(
            result, 0,
            "missing interface resolution must produce init-failure",
        );
        // init struct MUST remain untouched — we bail before writing.
        assert!(init.initialize.is_none());
        assert!(init.deinitialize.is_none());
    }

    // Happy path: when the resolver returns Some for every name, the entry
    // point stashes extension state, wires the callbacks, and reports
    // success. The sentinel fn is a syntactically valid interface fn — we
    // never call it because the tests don't invoke initialize/deinitialize.
    #[test]
    fn entry_point_populates_init_struct_on_success() {
        // Sentinel fn — used only for its address; never invoked. Its
        // signature is `unsafe extern "C" fn()` (the opaque
        // `GDExtensionInterfaceFunctionPtr` inner type).
        unsafe extern "C" fn sentinel() {}

        unsafe extern "C" fn success_gpa(
            _p_name: *const core::ffi::c_char,
        ) -> crate::ffi::gdextension::GDExtensionInterfaceFunctionPtr {
            Some(sentinel)
        }

        // Clear any leftover state from previous tests (they run in
        // parallel by default; the Mutex serializes writes).
        {
            let mut guard = EXTENSION_STATE.lock().unwrap();
            *guard = None;
        }

        let mut init = GDExtensionInitialization {
            minimum_initialization_level: GDExtensionInitializationLevel::Core,
            userdata: ptr::null_mut(),
            initialize: None,
            deinitialize: None,
        };
        // SAFETY: `&mut init` is a valid writable slot for the duration
        // of the call.
        let result = unsafe { vokra_gdextension_init(success_gpa, ptr::null_mut(), &mut init) };

        assert_eq!(result, 1, "successful init must return 1");
        assert_eq!(
            init.minimum_initialization_level,
            GDExtensionInitializationLevel::Scene
        );
        assert!(init.initialize.is_some());
        assert!(init.deinitialize.is_some());

        // Extension state MUST be populated.
        {
            let guard = EXTENSION_STATE.lock().unwrap();
            assert!(
                guard.is_some(),
                "successful init must stash EXTENSION_STATE",
            );
        }

        // Clear so the next test doesn't observe leftover state.
        {
            let mut guard = EXTENSION_STATE.lock().unwrap();
            *guard = None;
        }
    }
}
