//! Raw QNN (Qualcomm AI Engine Direct) FFI, loaded at runtime with dlopen /
//! LoadLibrary (Android / Linux / Windows only — gated by the parent module,
//! and only under the `qnn` feature).
//!
//! This module is the **only** place that talks to the QNN runtime, and it does
//! so with runtime dynamic loading — **no `qnn-sys` / `hexagon` binding crate**
//! (M5-02 red line; keeps the root `Cargo.lock` free of non-`vokra-*` crates,
//! NFR-DS-02).
//!
//! # Why dlopen instead of `#[link(name = "QnnHtp")]`
//!
//! Two reasons, both load-bearing (identical to `vokra-backend-cuda`):
//!
//! 1. **Qualcomm EULA install model** (`third_party/QUALCOMM-QNN-NOTES.md`):
//!    Vokra bundles and statically links no part of the Qualcomm AI Engine
//!    Direct SDK. The developer installs the SDK/runtime; Vokra detects it at
//!    runtime via `dlopen("libQnnHtp.so")` / `LoadLibrary("QnnHtp.dll")`.
//! 2. **All-target build** (NFR-PT-01): a link-time dependency on `libQnnHtp`
//!    would break `cargo build` on any host without the QNN runtime. Runtime
//!    loading means the crate **compiles wherever this cfg is active** and only
//!    *fails at runtime* (an explicit
//!    [`VokraError::BackendUnavailable`], never a silent CPU fall back —
//!    NFR-RL-06) where no QNN runtime / Hexagon device exists.
//!
//! The dynamic-loader primitives (`dlopen`/`dlsym`/`dlclose` on Unix;
//! `LoadLibraryA`/`GetProcAddress`/`FreeLibrary` on Windows) are symbols `std`
//! already links (libdl / kernel32), declared inline here — the same technique
//! `vokra-backend-cuda` / `-vulkan` and `vokra-mmap` use, so nothing is added
//! to `Cargo.lock`.
//!
//! # ⚠ Nothing here is verified against a QNN SDK header
//!
//! No Qualcomm AI Engine Direct SDK is on the authoring host. The library
//! candidate names, the interface entry symbol name, and the placeholder struct
//! layout below are **from the WP instruction, not first-hand verified**. The
//! compile-time layout assert is a *self-consistency guard* (sum of the
//! hand-written field sizes), **not** a real-header check — owner T11 confirms
//! the values against the SDK header (see `docs/handoff/m5-02.md`). The scaffold
//! probe therefore only checks that the library loads and a representative
//! symbol *resolves* — it never calls a QNN entry point (which would require a
//! verified signature and struct layout that do not exist yet).

// `c_int` is used only inside the platform `dl` submodules (each imports it
// itself); the top level needs just `c_char` (dlopen name casts) and `c_void`
// (the library handle). Keeping `c_int` here would be an unused import on the
// Linux/Windows CI arm where this module actually compiles (-D warnings).
use core::ffi::{c_char, c_void};

use vokra_core::{Result, VokraError};

// ---------------------------------------------------------------------------
// Platform dynamic-loader primitives (dlopen / LoadLibrary). No binding crate.
// Mirrors `crates/vokra-backend-cuda/src/sys.rs`.
// ---------------------------------------------------------------------------

#[cfg(unix)]
mod dl {
    use core::ffi::{c_char, c_int, c_void};

    /// `RTLD_NOW` — resolve all undefined symbols before `dlopen` returns.
    /// Value `0x2` on both Linux (glibc/musl / Android bionic) and BSD.
    const RTLD_NOW: c_int = 2;

    // `std` links libc (and libdl on Linux / Android), which export these.
    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
    }

    /// Opens a shared library by NUL-terminated name; null on failure.
    ///
    /// # Safety
    /// `name` must be a valid NUL-terminated C string.
    pub(super) unsafe fn open(name: *const c_char) -> *mut c_void {
        // SAFETY: `name` is a valid NUL-terminated C string per the caller.
        unsafe { dlopen(name, RTLD_NOW) }
    }

    /// Resolves a symbol in an open library; null if absent.
    ///
    /// # Safety
    /// `handle` must be a live handle from [`open`]; `symbol` NUL-terminated.
    pub(super) unsafe fn sym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void {
        // SAFETY: `handle` is live and `symbol` is a valid C string per caller.
        unsafe { dlsym(handle, symbol) }
    }

    /// Closes a library handle.
    ///
    /// # Safety
    /// `handle` must be a live handle from [`open`], closed at most once.
    pub(super) unsafe fn close(handle: *mut c_void) {
        // SAFETY: `handle` is a live, not-yet-closed handle per the caller.
        unsafe {
            dlclose(handle);
        }
    }
}

#[cfg(windows)]
mod dl {
    use core::ffi::{c_char, c_void};

    // `std` links `kernel32`, which exports these. `extern "system"` is the
    // Win32 calling convention. `GetProcAddress` really returns `FARPROC`;
    // declaring it `-> *mut c_void` is ABI-compatible (a pointer-sized return).
    unsafe extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
    }

    /// Opens a DLL by NUL-terminated name; null on failure.
    ///
    /// # Safety
    /// `name` must be a valid NUL-terminated C string.
    pub(super) unsafe fn open(name: *const c_char) -> *mut c_void {
        // SAFETY: `name` is a valid NUL-terminated C string per the caller.
        unsafe { LoadLibraryA(name) }
    }

    /// Resolves a symbol in an open DLL; null if absent.
    ///
    /// # Safety
    /// `handle` must be a live module handle; `symbol` NUL-terminated.
    pub(super) unsafe fn sym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void {
        // SAFETY: `handle` is live and `symbol` is a valid C string per caller.
        unsafe { GetProcAddress(handle, symbol) }
    }

    /// Frees a module handle.
    ///
    /// # Safety
    /// `handle` must be a live module handle, freed at most once.
    pub(super) unsafe fn close(handle: *mut c_void) {
        // SAFETY: `handle` is a live, not-yet-freed handle per the caller.
        unsafe {
            FreeLibrary(handle);
        }
    }
}

/// A loaded QNN shared library. Closes its handle on drop.
///
/// Not `Send`/`Sync`: consistent with the other backends' loader handles.
pub(crate) struct DynLib {
    handle: *mut c_void,
    /// The name (candidate or `VOKRA_QNN_LIB` override) that actually loaded —
    /// reported through [`crate::QnnCapabilities::library_name`].
    name: String,
}

impl DynLib {
    /// Tries to open a NUL-terminated library name, recording `display` as the
    /// human-readable name on success. Returns `None` if it does not load.
    fn open_one(name_c: &[u8], display: &str) -> Option<DynLib> {
        debug_assert_eq!(
            name_c.last(),
            Some(&0),
            "library name must be NUL-terminated"
        );
        // SAFETY: `name_c` is a NUL-terminated C string; `dl::open`
        // (dlopen / LoadLibraryA) returns null on failure, checked here.
        let handle = unsafe { dl::open(name_c.as_ptr() as *const c_char) };
        if handle.is_null() {
            None
        } else {
            Some(DynLib {
                handle,
                name: display.to_owned(),
            })
        }
    }

    /// The library name that loaded.
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// Whether NUL-terminated `symbol` resolves in this library.
    ///
    /// This is a *presence* check only: the scaffold never `transmute`s the
    /// address to a typed function pointer, because the true QNN signature is
    /// UNKNOWN without the SDK header (owner T11). The graph-construction
    /// re-issue wave adds a typed `get<F>` (the `vokra-backend-cuda` shape) once
    /// the signatures are verified.
    pub(crate) fn has_symbol(&self, symbol: &[u8]) -> bool {
        debug_assert_eq!(
            symbol.last(),
            Some(&0),
            "symbol name must be NUL-terminated"
        );
        // SAFETY: `handle` is live; `symbol` is a valid NUL-terminated C string.
        let ptr = unsafe { dl::sym(self.handle, symbol.as_ptr() as *const c_char) };
        !ptr.is_null()
    }
}

impl Drop for DynLib {
    fn drop(&mut self) {
        // SAFETY: `handle` came from `dl::open` and is closed exactly once here.
        unsafe { dl::close(self.handle) }
    }
}

/// Candidate library names for the QNN HTP (Hexagon) backend, tried in order.
///
/// **UNVERIFIED (no SDK on the authoring host).** From the WP instruction:
/// `libQnnHtp` is the HTP/Hexagon backend of the Qualcomm AI Engine Direct SDK;
/// the exact SONAME / DLL name is confirmed by owner T11 against the SDK. The
/// candidate-array + `VOKRA_QNN_LIB` override shape mirrors
/// `vokra-backend-cuda` / `-vulkan`, so a corrected name needs no code change.
pub(crate) const QNN_HTP_LIB_CANDIDATES: &[&[u8]] = &[
    // Android / Linux.
    b"libQnnHtp.so\0",
    // Windows (Windows-on-ARM Snapdragon).
    b"QnnHtp.dll\0",
];

/// Environment override — a full path to the QNN HTP library, for
/// developer-controlled test environments (mirrors `VOKRA_CUDA_LIB` /
/// `VOKRA_VULKAN_LIB`). This is how an owner points the probe at their SDK's
/// `libQnnHtp.so` during T11/T12 without editing the candidate list.
pub(crate) const ENV_VOKRA_QNN_LIB: &str = "VOKRA_QNN_LIB";

/// A representative QNN interface entry symbol whose presence the probe checks.
///
/// **UNVERIFIED (no SDK).** From the WP instruction: `QnnInterface_getProviders`
/// is the documented interface-enumeration entry point. Owner T11 confirms the
/// exact spelling and any API-version negotiation. The probe only checks that it
/// *resolves*; it never calls it (the signature is UNKNOWN — see [`DynLib::has_symbol`]).
pub(crate) const QNN_INTERFACE_ENTRY_SYMBOL: &[u8] = b"QnnInterface_getProviders\0";

/// Loads the QNN HTP library: the `VOKRA_QNN_LIB` override first (if set), then
/// the candidate list.
///
/// # Errors
///
/// [`VokraError::BackendUnavailable`] if no QNN library loads (no SDK/runtime
/// installed on this host) — never a silent fall back (NFR-RL-06).
pub(crate) fn load_qnn_library() -> Result<DynLib> {
    // 1. Explicit override path (owner test environments).
    if let Some(path) = std::env::var_os(ENV_VOKRA_QNN_LIB) {
        // Build a NUL-terminated byte string for dlopen / LoadLibraryA.
        #[cfg(unix)]
        let bytes = {
            use std::os::unix::ffi::OsStrExt;
            let mut v = path.as_os_str().as_bytes().to_vec();
            v.push(0);
            v
        };
        #[cfg(windows)]
        let bytes = {
            let mut v = path.to_string_lossy().into_owned().into_bytes();
            v.push(0);
            v
        };
        let display = path.to_string_lossy().into_owned();
        if let Some(lib) = DynLib::open_one(&bytes, &display) {
            return Ok(lib);
        }
        return Err(VokraError::BackendUnavailable(format!(
            "VOKRA_QNN_LIB is set to `{display}` but that QNN library did not load (wrong path, or \
             not a loadable QNN runtime). Unset it to fall back to the candidate names, or point it \
             at a real libQnnHtp."
        )));
    }
    // 2. Candidate names on the dynamic-linker search path.
    for cand in QNN_HTP_LIB_CANDIDATES {
        // Display name = candidate without the trailing NUL.
        let display = String::from_utf8_lossy(&cand[..cand.len().saturating_sub(1)]).into_owned();
        if let Some(lib) = DynLib::open_one(cand, &display) {
            return Ok(lib);
        }
    }
    Err(VokraError::BackendUnavailable(
        "QNN runtime library (libQnnHtp.so / QnnHtp.dll) not found: no Qualcomm AI Engine Direct \
         SDK / runtime on this host, or it is not on the dynamic-linker search path. Vokra bundles \
         no QNN runtime (Qualcomm EULA install model, NFR-PT-01 all-target build); install the SDK \
         or set VOKRA_QNN_LIB to use the QNN backend."
            .to_owned(),
    ))
}

// ---------------------------------------------------------------------------
// Placeholder struct layout + compile-time self-consistency guard (M5-02-T04).
//
// ⚠ NOT a real-header transcription. No QNN SDK is on the authoring host, so the
// field set below is a PLACEHOLDER that only demonstrates the compile-time
// layout-assert mechanism (the M3-11 GDExtension pattern). The asserted size /
// align are the sum of the hand-written fields — a self-consistency guard that
// catches an accidental field-set edit, NOT proof that the layout matches the
// SDK. Owner T11 verifies the real `Qnn_Version_t` (and every other struct the
// graph-construction re-issue wave needs) against the header and corrects these
// values, exactly as M3-11 probed the Godot header with `clang -m64`.
//
// `allow(dead_code)`: the fields are never read (only `size_of` / `align_of` are
// taken by the assert) — a real read appears in the re-issue wave once the
// struct is actually passed to a QNN entry point with a verified signature.
// ---------------------------------------------------------------------------

/// **PLACEHOLDER, UNVERIFIED** QNN API-version struct skeleton (a common
/// version-triple convention — *not* a claim about the SDK's real layout).
#[cfg(target_pointer_width = "64")]
#[repr(C)]
#[allow(dead_code)]
pub(crate) struct Qnn_Version_t {
    major: u32,
    minor: u32,
    patch: u32,
}

#[cfg(target_pointer_width = "64")]
const _: () = {
    // 3 × u32 = 12 bytes, 4-byte aligned. Self-consistency guard only (owner
    // T11 replaces with the SDK-verified value).
    assert!(core::mem::size_of::<Qnn_Version_t>() == 12);
    assert!(core::mem::align_of::<Qnn_Version_t>() == 4);
};
