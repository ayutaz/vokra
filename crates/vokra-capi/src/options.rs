//! Session options: the C-visible backend selector and the opaque options
//! object that carries it (design
//! `docs/superpowers/specs/2026-08-14-c-abi-backend-speaker-design.md` §3.1-3.2).
//!
//! # Why an options object and not an `_ex` overload
//!
//! `vokra_session_create_from_file` hard-wired `BackendKind::Cpu`, and its own
//! source comment recorded that "a backend selector argument is a future
//! breaking change". Adding a parameter to a symbol that is about to freeze
//! (IF-01, M5-13) is not an option, and a chain of `_ex` / `_ex2` overloads
//! ages badly. Instead the two constructors gain `_with_options` siblings that
//! take an **opaque** handle (design D2): every future knob — KV quantization,
//! thread count — is one more setter, never a new constructor and never a
//! struct layout pinned into the frozen surface (a `#[repr(C)]` options struct
//! would also pin its field order into every C# `[StructLayout]` marshaller).
//!
//! # Which backends are exposed
//!
//! [`vokra_backend_t`] carries the five *compute* backends only. The CoreML and
//! QNN **delegates** deliberately have no C value yet: `docs/handoff/m4-12.md`
//! §(e)-3 says not to reserve variant slots for delegate selection before the
//! real-hardware ANE / Hexagon bakeoff (M5-01 / M5-02), and to land that API as
//! a new symbol afterwards. Appending `= 5` / `= 6` later is a backward
//! compatible additive change; guessing them now would freeze an unvalidated
//! surface.
//!
//! # No silent CPU fall back (FR-EX-08)
//!
//! Selecting a backend this build cannot provide is an error, never a quiet
//! downgrade — see [`crate::session`] for where the check fires and why it
//! fires there rather than in [`vokra_session_options_set_backend`].

use vokra_core::BackendKind;

use crate::error::{fail_invalid, vokra_status_t};
use crate::ffi_guard::{guard, guard_bool, guard_ptr, guard_void, required_mut};
use crate::handle::{drop_raw, into_raw};

/// The backend a session runs its hot ops on.
///
/// Pass one of these to `vokra_session_options_set_backend`. The delegate
/// backends (Apple ANE via CoreML, Qualcomm Hexagon via QNN) are **not** in
/// this enum yet — they land as a separate symbol after the real-hardware
/// bakeoff (M5-01 / M5-02). NNAPI will never be added (FR-BE-07: Google
/// deprecated it in Android 15); Android GPU support is Vulkan.
//
// C-style spelling so cbindgen emits the enum verbatim (the `error.rs`
// `vokra_status_t` pattern). Exported through `[export] include` in
// cbindgen.toml because the C functions take the discriminant as `int32_t` —
// see `backend_from_c` for why.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum vokra_backend_t {
    /// Portable CPU backend (SSE2/AVX2 on x86-64, NEON on ARM64). Always
    /// available — the baseline every build ships (FR-BE-01).
    VOKRA_BACKEND_CPU = 0,
    /// Apple Metal (macOS / iOS). Requires a `metal`-featured build and a
    /// Metal device.
    VOKRA_BACKEND_METAL = 1,
    /// NVIDIA CUDA (Windows / Linux). Requires a `cuda`-featured build; the
    /// driver and NVRTC are `dlopen`ed at run time (NVIDIA EULA install
    /// model — nothing is bundled).
    VOKRA_BACKEND_CUDA = 2,
    /// Vulkan (Linux / Android / Windows). Requires a `vulkan`-featured build
    /// and a loadable `libvulkan`.
    VOKRA_BACKEND_VULKAN = 3,
    /// Browser WebGPU (`wasm32` targets). Requires a `webgpu`-featured build.
    VOKRA_BACKEND_WEBGPU = 4,
}

/// The backend a session gets when the caller supplies no options — the same
/// backend the pre-options constructors always used, so `opts = NULL` is
/// exactly the legacy behaviour (design §3.4).
pub(crate) const DEFAULT_BACKEND: BackendKind = BackendKind::Cpu;

/// Opaque session-construction options.
///
/// Created by `vokra_session_options_create`, configured with the
/// `vokra_session_options_set_*` setters, consumed (by reference, never taken
/// over) by `vokra_session_create_from_file_with_options` /
/// `vokra_session_create_from_bytes_with_options`, and released with
/// `vokra_session_options_destroy`. Opaque to C: the layout is not part of the
/// ABI, so new knobs never break existing binaries.
///
/// An options object is an independent, plain-old allocation with no shared
/// state; destroying one never affects another, and one object may configure
/// any number of sessions.
//
// Deliberately NOT `#[repr(C)]`: that is what makes cbindgen emit a bare
// forward declaration instead of the field list (design D2).
#[allow(non_camel_case_types)]
pub struct vokra_session_options_t {
    pub(crate) backend: BackendKind,
}

/// Maps a C `vokra_backend_t` discriminant onto its [`BackendKind`], or `None`
/// when the value is not one this ABI defines.
///
/// The C functions take the selector as a plain integer rather than as
/// `vokra_backend_t` on the Rust side. C permits any `int` to travel through an
/// `enum` parameter, but *materialising* an out-of-range discriminant as a Rust
/// enum is undefined behaviour — and the contract requires exactly that case to
/// come back as `VOKRA_ERROR_INVALID_ARGUMENT` (design §5), which is only
/// definable if the value arrives as an integer. The `const` arms below read
/// the enum's own discriminants, so the header and this mapping cannot drift.
fn backend_from_c(value: i32) -> Option<BackendKind> {
    const CPU: i32 = vokra_backend_t::VOKRA_BACKEND_CPU as i32;
    const METAL: i32 = vokra_backend_t::VOKRA_BACKEND_METAL as i32;
    const CUDA: i32 = vokra_backend_t::VOKRA_BACKEND_CUDA as i32;
    const VULKAN: i32 = vokra_backend_t::VOKRA_BACKEND_VULKAN as i32;
    const WEBGPU: i32 = vokra_backend_t::VOKRA_BACKEND_WEBGPU as i32;
    match value {
        CPU => Some(BackendKind::Cpu),
        METAL => Some(BackendKind::Metal),
        CUDA => Some(BackendKind::Cuda),
        VULKAN => Some(BackendKind::Vulkan),
        WEBGPU => Some(BackendKind::WebGpu),
        // Includes the 5 / 6 a caller might guess for CoreML / QNN: those are
        // not part of this ABI yet (module docs).
        _ => None,
    }
}

/// The backend an options pointer selects. `NULL` is the documented "use the
/// defaults" input and yields [`DEFAULT_BACKEND`] (design §3.4).
///
/// # Safety
///
/// `opts` must be `NULL` or a live handle from `vokra_session_options_create`
/// that has not been destroyed.
pub(crate) unsafe fn selected_backend(opts: *const vokra_session_options_t) -> BackendKind {
    // SAFETY: `as_ref` null-checks; a non-null `opts` is a live options object
    // per the contract, borrowed only for this read.
    match unsafe { opts.as_ref() } {
        Some(options) => options.backend,
        None => DEFAULT_BACKEND,
    }
}

/// Allocates a session-options object preset to the library defaults (CPU
/// backend), or returns `NULL` if the allocation fails.
///
/// # Returns
///
/// A handle to release with `vokra_session_options_destroy`, or `NULL`. Unlike
/// the status-returning functions this one has a single failure mode, so the
/// `NULL` check the caller writes anyway is the whole error contract and
/// `vokra_last_error()` is left untouched.
#[unsafe(no_mangle)]
pub extern "C" fn vokra_session_options_create() -> *mut vokra_session_options_t {
    guard_ptr(|| {
        into_raw(vokra_session_options_t {
            backend: DEFAULT_BACKEND,
        })
    })
}

/// Frees an options object from `vokra_session_options_create`. `NULL` is a
/// no-op; using the handle afterwards is undefined behaviour.
///
/// Options are copied into the session at creation time, so destroying an
/// options object never affects sessions already built from it, and the object
/// may be destroyed immediately after the create call returns.
///
/// # Safety
///
/// `opts` must be `NULL` or a handle from `vokra_session_options_create` that
/// has not already been destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_session_options_destroy(opts: *mut vokra_session_options_t) {
    guard_void(|| {
        // SAFETY: `opts` is NULL or a live handle from `into_raw`; `drop_raw`
        // frees it once and treats NULL as a no-op (ADR-0003 §3-a).
        unsafe { drop_raw(opts) };
    });
}

/// Selects the backend a session built from these options will run on.
///
/// # Parameters
///
/// - `opts`: options handle from `vokra_session_options_create`.
/// - `backend`: a `vokra_backend_t` value (`VOKRA_BACKEND_CPU` = 0,
///   `_METAL` = 1, `_CUDA` = 2, `_VULKAN` = 3, `_WEBGPU` = 4). Any other value
///   is rejected.
///
/// # Returns
///
/// `VOKRA_OK`, or `VOKRA_ERROR_INVALID_ARGUMENT` for a NULL handle or an
/// unknown `backend` value. **A rejected call changes nothing** — the object
/// keeps the backend it had and stays usable.
///
/// This setter deliberately does **not** check whether the selected backend
/// exists on this machine: it records an intent, and probing a GPU would make a
/// plain setter allocate a device. Availability is resolved when the session is
/// created (`VOKRA_ERROR_BACKEND_UNAVAILABLE`), and can be queried up front
/// with `vokra_backend_available`.
///
/// # Safety
///
/// `opts` must be a live handle from `vokra_session_options_create`, not used
/// concurrently from another thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_session_options_set_backend(
    opts: *mut vokra_session_options_t,
    backend: i32,
) -> vokra_status_t {
    guard(|| {
        // SAFETY: `opts` is validated (NULL rejected) by `required_mut`; the
        // caller guarantees exclusive access for the duration of the call.
        let options = unsafe { required_mut(opts, "opts") }?;
        let kind = backend_from_c(backend).ok_or_else(|| {
            fail_invalid(&format!(
                "argument `backend` = {backend} is not a vokra_backend_t value (expected \
                 VOKRA_BACKEND_CPU = 0, VOKRA_BACKEND_METAL = 1, VOKRA_BACKEND_CUDA = 2, \
                 VOKRA_BACKEND_VULKAN = 3 or VOKRA_BACKEND_WEBGPU = 4; the CoreML / QNN \
                 delegates have no C value yet)"
            ))
        })?;
        // Assigned only after validation, so a rejected call leaves a usable
        // object (documented above).
        options.backend = kind;
        Ok(())
    })
}

/// Reports whether `backend` can actually be used by this build on this
/// machine.
///
/// `true` means the backend is compiled into this binary **and** its device /
/// driver is present, so `vokra_session_create_*_with_options` will not reject
/// it with `VOKRA_ERROR_BACKEND_UNAVAILABLE`. It does not promise that every
/// model runs there: a backend that lacks a kernel some model needs still
/// fails loudly at inference with `VOKRA_ERROR_UNSUPPORTED_OP`, never by
/// falling back to the CPU (FR-EX-08).
///
/// A query has no failure mode, so this returns a plain `bool`, leaves
/// `vokra_last_error()` untouched, and answers `false` — rather than erroring —
/// for a `backend` value outside `vokra_backend_t`.
///
/// The probe genuinely opens the backend (a Metal device, the CUDA driver,
/// `libvulkan`) and closes it again, so it is far cheaper than loading a model
/// but is not free; cache the answer if you poll it in a UI loop.
#[unsafe(no_mangle)]
pub extern "C" fn vokra_backend_available(backend: i32) -> bool {
    guard_bool(|| {
        // `make_backend` is the same constructor the graph evaluator uses, so
        // "available" here means exactly what the loader will decide.
        backend_from_c(backend).is_some_and(|kind| vokra_models::make_backend(kind).is_ok())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The C discriminants are ABI: freeze every one of them, and freeze the
    /// fact that CoreML / QNN have no value (design D1).
    #[test]
    fn backend_discriminants_pin_numeric_abi() {
        assert_eq!(vokra_backend_t::VOKRA_BACKEND_CPU as i32, 0);
        assert_eq!(vokra_backend_t::VOKRA_BACKEND_METAL as i32, 1);
        assert_eq!(vokra_backend_t::VOKRA_BACKEND_CUDA as i32, 2);
        assert_eq!(vokra_backend_t::VOKRA_BACKEND_VULKAN as i32, 3);
        assert_eq!(vokra_backend_t::VOKRA_BACKEND_WEBGPU as i32, 4);
        // The delegate slots stay unassigned until the NPU bakeoff.
        assert_eq!(backend_from_c(5), None);
        assert_eq!(backend_from_c(6), None);
    }

    #[test]
    fn backend_from_c_maps_every_exposed_value() {
        assert_eq!(backend_from_c(0), Some(BackendKind::Cpu));
        assert_eq!(backend_from_c(1), Some(BackendKind::Metal));
        assert_eq!(backend_from_c(2), Some(BackendKind::Cuda));
        assert_eq!(backend_from_c(3), Some(BackendKind::Vulkan));
        assert_eq!(backend_from_c(4), Some(BackendKind::WebGpu));
    }

    #[test]
    fn backend_from_c_rejects_out_of_range_values() {
        for value in [7, 99, -1, i32::MAX, i32::MIN] {
            assert_eq!(
                backend_from_c(value),
                None,
                "value {value} must be rejected"
            );
        }
    }

    #[test]
    fn create_defaults_to_cpu_and_destroy_is_paired() {
        let opts = vokra_session_options_create();
        assert!(!opts.is_null());
        // SAFETY: `opts` is a live handle from the line above.
        let backend = unsafe { selected_backend(opts.cast_const()) };
        assert_eq!(backend, BackendKind::Cpu, "default backend is CPU");
        // SAFETY: live handle, destroyed exactly once.
        unsafe { vokra_session_options_destroy(opts) };
    }

    #[test]
    fn null_options_pointer_selects_the_default_backend() {
        // SAFETY: NULL is the documented "use the defaults" input.
        let backend = unsafe { selected_backend(std::ptr::null()) };
        assert_eq!(backend, DEFAULT_BACKEND);
    }

    #[test]
    fn set_backend_records_the_selection() {
        let opts = vokra_session_options_create();
        // SAFETY: live handle.
        let st = unsafe {
            vokra_session_options_set_backend(opts, vokra_backend_t::VOKRA_BACKEND_METAL as i32)
        };
        assert_eq!(st, vokra_status_t::VOKRA_OK);
        // SAFETY: live handle.
        let recorded = unsafe { selected_backend(opts.cast_const()) };
        assert_eq!(recorded, BackendKind::Metal);
        // SAFETY: live handle, destroyed exactly once.
        unsafe { vokra_session_options_destroy(opts) };
    }

    #[test]
    fn rejected_set_backend_leaves_the_previous_selection() {
        let opts = vokra_session_options_create();
        // SAFETY: live handle.
        let st = unsafe {
            vokra_session_options_set_backend(opts, vokra_backend_t::VOKRA_BACKEND_CUDA as i32)
        };
        assert_eq!(st, vokra_status_t::VOKRA_OK);
        // A rejected value must not clobber the recorded selection.
        // SAFETY: live handle.
        let st = unsafe { vokra_session_options_set_backend(opts, 99) };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
        // SAFETY: live handle.
        let recorded = unsafe { selected_backend(opts.cast_const()) };
        assert_eq!(
            recorded,
            BackendKind::Cuda,
            "a rejected set_backend must leave the object untouched"
        );
        // SAFETY: live handle, destroyed exactly once.
        unsafe { vokra_session_options_destroy(opts) };
    }

    #[test]
    fn set_backend_rejects_null_handle() {
        // SAFETY: NULL is the rejected branch; no deref happens.
        let st = unsafe { vokra_session_options_set_backend(std::ptr::null_mut(), 0) };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
    }

    #[test]
    fn destroy_null_is_a_noop() {
        // SAFETY: NULL is the explicit no-op branch (ADR-0003 §3-a).
        unsafe { vokra_session_options_destroy(std::ptr::null_mut()) };
    }

    #[test]
    fn cpu_is_always_available_and_unknown_values_are_not() {
        assert!(
            vokra_backend_available(vokra_backend_t::VOKRA_BACKEND_CPU as i32),
            "the CPU backend is the always-present baseline (FR-BE-01)"
        );
        for value in [5, 6, 99, -1, i32::MAX, i32::MIN] {
            assert!(
                !vokra_backend_available(value),
                "value {value} is not a vokra_backend_t and cannot be available"
            );
        }
    }

    /// Availability must agree with what `make_backend` decides, for every
    /// exposed backend and in whatever feature configuration this build has —
    /// that agreement is what lets a caller trust the query.
    #[test]
    fn availability_agrees_with_the_backend_constructor() {
        for (value, kind) in [
            (0, BackendKind::Cpu),
            (1, BackendKind::Metal),
            (2, BackendKind::Cuda),
            (3, BackendKind::Vulkan),
            (4, BackendKind::WebGpu),
        ] {
            assert_eq!(
                vokra_backend_available(value),
                vokra_models::make_backend(kind).is_ok(),
                "vokra_backend_available disagrees with make_backend for {kind:?}"
            );
        }
    }
}
