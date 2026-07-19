//! WebGPU adapter/device probe (M4-01-T09).
//!
//! Mirrors `vokra-backend-vulkan/src/probe.rs`: an honest capability report
//! on a live WebGPU host, an explicit
//! [`VokraError::BackendUnavailable`](vokra_core::VokraError::BackendUnavailable)
//! everywhere else (FR-EX-08 — never a silent CPU fall back). "Everywhere
//! else" covers three distinct absences, each with its own message:
//!
//! 1. **native / non-wasm32 target or feature `webgpu` off** — the shim is
//!    not even compiled;
//! 2. **wasm32 + glue in "unavailable" mode** — the import object exists
//!    (instantiation demands it) but reported no adapter
//!    (`navigator.gpu` absent / `requestAdapter` null — the dlopen-failure
//!    analogue);
//! 3. **wasm32 + glue init error** — adapter present but device request or
//!    SAB bridge setup failed (message forwarded from the glue, e.g. the
//!    COOP/COEP guidance).

use vokra_core::{Result, VokraError};

/// What the probe learned about the WebGPU host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebGpuCapabilities {
    /// An adapter + device pair is initialised and the SAB bridge is live.
    pub adapter_ready: bool,
}

/// Probes the WebGPU adapter through the extern-import shim.
///
/// # Errors
///
/// [`VokraError::BackendUnavailable`] on non-wasm32 targets, builds without
/// the `webgpu` feature, hosts without a WebGPU adapter, or glue init
/// failures (the glue's message — e.g. the COOP/COEP deployment guidance —
/// is embedded).
#[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
pub fn vokra_webgpu_probe() -> Result<WebGpuCapabilities> {
    // SAFETY: sys.rs import contract — the glue implements `probe` (a wasm
    // module with unsatisfied imports cannot instantiate), takes no
    // pointers, and returns a plain i32.
    let status = unsafe { crate::sys::vokra_webgpu_probe() };
    match status {
        1 => Ok(WebGpuCapabilities {
            adapter_ready: true,
        }),
        0 => Err(VokraError::BackendUnavailable(
            "no WebGPU adapter: navigator.gpu is absent or requestAdapter() returned null in \
             this browser/context. Vokra does not silently fall back to the CPU (FR-EX-08) — \
             select BackendKind::Cpu explicitly to run on the WASM SIMD128/scalar path, or use \
             a WebGPU-enabled browser (see docs/tutorials/web.md)."
                .to_owned(),
        )),
        err => {
            let detail = crate::sys::last_glue_error();
            Err(VokraError::BackendUnavailable(format!(
                "WebGPU glue initialisation failed (status {err}): {detail}"
            )))
        }
    }
}

/// Stub for non-WebGPU builds/targets — explicit
/// [`VokraError::BackendUnavailable`], never a silent CPU substitute
/// (FR-EX-08 / NFR-RL-06).
///
/// # Errors
///
/// Always [`VokraError::BackendUnavailable`] (that is this stub's contract).
#[cfg(not(all(feature = "webgpu", target_arch = "wasm32")))]
pub fn vokra_webgpu_probe() -> Result<WebGpuCapabilities> {
    Err(VokraError::BackendUnavailable(
        "vokra-backend-webgpu compiled without the `webgpu` feature or off wasm32. The WebGPU \
         backend runs inside a browser WASM module whose JS glue satisfies the vokra_webgpu \
         imports (ADR M4-01-webgpu-wasm); rebuild with `--features webgpu --target \
         wasm32-unknown-unknown` and load it through web/pkg."
            .to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On every native test host the probe is an explicit BackendUnavailable
    /// (FR-EX-08) — the same off-target contract the Vulkan probe pins.
    #[test]
    fn probe_off_target_is_explicit_backend_unavailable() {
        if cfg!(not(all(feature = "webgpu", target_arch = "wasm32"))) {
            let err = vokra_webgpu_probe().unwrap_err();
            assert!(matches!(err, VokraError::BackendUnavailable(_)));
            let msg = format!("{err}");
            assert!(
                msg.contains("webgpu"),
                "unavailable message should name the feature: {msg}"
            );
        }
    }
}
