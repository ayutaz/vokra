//! Metal device / GPU-family probe (M2-01-T04; FR-EX-08 / NFR-RL-06).
//!
//! [`vokra_metal_probe`] is the Metal analogue of the (future) CUDA
//! `vokra_cuda_probe`: it reports whether a usable Metal device exists and, if
//! so, its name and GPU family. **A missing / incompatible device is an
//! explicit [`VokraError::BackendUnavailable`] — never a silent fall back to
//! the CPU backend** (FR-EX-08 permanent constraint, NFR-RL-06). Whether to run
//! on the CPU instead is the *caller's* explicit choice at backend-selection
//! time (the `Session` wiring is M2-01-T22), not something this layer decides.

use vokra_core::{Result, VokraError};

/// What [`vokra_metal_probe`] discovered about the host's Metal device.
///
/// FP16 / quantised fast-path capabilities are intentionally out of scope for
/// this slice (parity runs FP32, NFR-QL-01); they are a later work package
/// (M2-08).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalCapabilities {
    /// The `MTLDevice.name` string (e.g. `"Apple M1"`).
    pub device_name: String,
    /// Highest `MTLGPUFamilyApple<N>` the device reports supporting, if any
    /// (e.g. `Some(7)` on an M1). `None` on a non-Apple-silicon Metal device.
    pub apple_family: Option<u32>,
    /// Whether the device supports the Metal 3 feature set
    /// (`MTLGPUFamilyMetal3`).
    pub supports_metal3: bool,
}

impl MetalCapabilities {
    /// Human-readable one-line summary, e.g. `"Apple M1 (GPU family Apple7,
    /// Metal3)"`.
    pub fn summary(&self) -> String {
        let fam = match self.apple_family {
            Some(n) => format!("Apple{n}"),
            None => "non-Apple".to_owned(),
        };
        let metal3 = if self.supports_metal3 { ", Metal3" } else { "" };
        format!("{} (GPU family {fam}{metal3})", self.device_name)
    }
}

/// Detects the system default Metal device and its capabilities.
///
/// # Errors
///
/// Returns [`VokraError::BackendUnavailable`] when no Metal device is present —
/// on a host with no Metal GPU **and** on every non-Apple target (where the
/// backend is compiled out). This is the deliberate *explicit error* of
/// FR-EX-08 / NFR-RL-06: the Metal backend never silently degrades to CPU.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn vokra_metal_probe() -> Result<MetalCapabilities> {
    use crate::sys;

    // SAFETY: `MTLCreateSystemDefaultDevice` takes no arguments and returns an
    // owned `id` (or null). We release the device before returning, so it does
    // not leak.
    let device = unsafe { sys::MTLCreateSystemDefaultDevice() };
    if device.is_null() {
        return Err(VokraError::BackendUnavailable(
            "no system default Metal device (MTLCreateSystemDefaultDevice returned nil)".to_owned(),
        ));
    }

    // SAFETY: `device` is a valid, non-null `MTLDevice`. `name` is its
    // `-(NSString*)name` getter (autoreleased); `supportsFamily:` takes an
    // `NSInteger` and returns `BOOL`. The autorelease pool around this scope
    // drains the autoreleased NSString.
    let caps = unsafe {
        let pool = sys::objc_autoreleasePoolPush();

        let name_sel = sys::sel(b"name\0");
        let name_obj = sys::send_id(device, name_sel);
        let device_name =
            sys::nsstring_to_string(name_obj).unwrap_or_else(|| "unknown Metal device".to_owned());

        let supports_family = sys::sel(b"supportsFamily:\0");
        // Highest Apple family the device claims (scan Apple9 -> Apple1).
        let mut apple_family = None;
        let mut fam = sys::gpu_family::APPLE9;
        while fam >= sys::gpu_family::APPLE1 {
            if sys::send_bool_isize(device, supports_family, fam) {
                apple_family = Some((fam - sys::gpu_family::APPLE1) as u32 + 1);
                break;
            }
            fam -= 1;
        }
        let supports_metal3 =
            sys::send_bool_isize(device, supports_family, sys::gpu_family::METAL3);

        sys::objc_autoreleasePoolPop(pool);

        // Release the +1 owned device (Create Rule).
        sys::send_void(device, sys::sel(b"release\0"));

        MetalCapabilities {
            device_name,
            apple_family,
            supports_metal3,
        }
    };

    Ok(caps)
}

/// Non-Apple stub: the Metal backend is compiled out, so probing always fails
/// explicitly (FR-EX-08 / NFR-RL-06 — never a silent CPU fall back).
///
/// # Errors
///
/// Always returns [`VokraError::BackendUnavailable`].
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub fn vokra_metal_probe() -> Result<MetalCapabilities> {
    Err(VokraError::BackendUnavailable(
        "Metal backend is not compiled for this target (only macOS / iOS)".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn probe_detects_a_device_on_apple_host() {
        // This test runs on a Metal-capable Apple host (dev machine / CI
        // Apple-silicon runner). It asserts the probe surfaces a real device
        // rather than a silent failure.
        let caps = vokra_metal_probe().expect("probe must find a Metal device on an Apple host");
        assert!(!caps.device_name.is_empty());
        // Apple silicon reports an Apple GPU family; keep the assertion loose
        // (>=1) so it holds on any Apple7+ device.
        if let Some(fam) = caps.apple_family {
            assert!(fam >= 1, "Apple GPU family must be >= 1, got {fam}");
        }
        eprintln!("vokra_metal_probe: {}", caps.summary());
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn probe_is_explicit_error_off_apple() {
        assert!(matches!(
            vokra_metal_probe(),
            Err(VokraError::BackendUnavailable(_))
        ));
    }
}
