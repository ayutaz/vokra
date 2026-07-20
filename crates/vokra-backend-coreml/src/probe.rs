//! CoreML compute-device probe (M5-01-T06; FR-EX-08 / NFR-RL-06).
//!
//! [`vokra_coreml_probe`] is the CoreML analogue of `vokra_metal_probe`: it
//! reports whether the **Apple Neural Engine** is reachable through CoreML and,
//! if so, its core count. It is built on the public `MLAllComputeDevices()` API
//! (macOS 14.0+ / iOS 17.0+), which is the framework's own enumeration of the
//! devices it may schedule onto — so "the ANE is present" is answered by the OS
//! rather than assumed from the chip name.
//!
//! **A missing ANE is an explicit [`VokraError::BackendUnavailable`] — never a
//! silent fall back to the CPU backend** (FR-EX-08 permanent constraint,
//! NFR-RL-06). Whether to run on the CPU instead is the *caller's* explicit
//! choice at backend-selection time, not something this layer decides.

use vokra_core::{Result, VokraError};

/// What [`vokra_coreml_probe`] discovered about the host's CoreML compute
/// devices.
///
/// FP16 execution precision and per-op device placement are intentionally out
/// of scope for this scaffold; they are answered at execution time
/// (`MLComputePlan`, a later ticket) once the op path lands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreMlCapabilities {
    /// Total number of compute devices `MLAllComputeDevices()` reported
    /// (CPU + GPU + ANE, however many the host exposes).
    pub total_devices: usize,
    /// `MLNeuralEngineComputeDevice.totalCoreCount` when the ANE is present,
    /// else `None`. `Some(n)` is the affirmative "the ANE is reachable" signal.
    pub ane_core_count: Option<u32>,
}

impl CoreMlCapabilities {
    /// Whether the Apple Neural Engine is reachable through CoreML on this host.
    pub fn has_ane(&self) -> bool {
        self.ane_core_count.is_some()
    }

    /// Human-readable one-line summary, e.g.
    /// `"CoreML: 3 compute device(s), ANE 16 core(s)"`.
    pub fn summary(&self) -> String {
        match self.ane_core_count {
            Some(n) => format!(
                "CoreML: {} compute device(s), ANE {n} core(s)",
                self.total_devices
            ),
            None => format!(
                "CoreML: {} compute device(s), no Neural Engine",
                self.total_devices
            ),
        }
    }
}

/// Detects the host's CoreML compute devices and whether the ANE is among them.
///
/// # Errors
///
/// Returns [`VokraError::BackendUnavailable`] when no Neural Engine is present —
/// on an Apple host without an ANE **and** on every non-Apple target (where the
/// backend is compiled out). This is the deliberate *explicit error* of
/// FR-EX-08 / NFR-RL-06: the CoreML backend never silently degrades to CPU.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn vokra_coreml_probe() -> Result<CoreMlCapabilities> {
    use crate::sys;

    // SAFETY: `MLAllComputeDevices()` takes no arguments and returns an
    // autoreleased `NSArray` (or null on an OS too old to have the symbol,
    // which the weak-link resolves to null). We only read it inside an
    // autorelease pool and never over-release it (it is not owned by us).
    let caps = unsafe {
        let pool = sys::objc_autoreleasePoolPush();

        let devices = sys::MLAllComputeDevices();
        if devices.is_null() {
            sys::objc_autoreleasePoolPop(pool);
            return Err(VokraError::BackendUnavailable(
                "MLAllComputeDevices() returned nil (CoreML compute-device enumeration \
                 unavailable on this OS)"
                    .to_owned(),
            ));
        }

        let count_sel = sys::sel(b"count\0");
        let object_at_sel = sys::sel(b"objectAtIndex:\0");
        let is_kind_sel = sys::sel(b"isKindOfClass:\0");
        let core_count_sel = sys::sel(b"totalCoreCount\0");
        let ane_class = sys::class(b"MLNeuralEngineComputeDevice\0");

        let total_devices = sys::send_usize(devices, count_sel);

        // Scan for an MLNeuralEngineComputeDevice element and read its core
        // count. `isKindOfClass:` is the documented way to discriminate the
        // MLComputeDeviceProtocol conformers; `ane_class` is null only if the
        // class is unavailable, in which case no element can match it.
        let mut ane_core_count = None;
        if !ane_class.is_null() {
            let mut i = 0usize;
            while i < total_devices {
                let dev = sys::send_id_usize(devices, object_at_sel, i);
                if !dev.is_null() && sys::send_bool_class(dev, is_kind_sel, ane_class) {
                    let cores = sys::send_isize(dev, core_count_sel);
                    // totalCoreCount is a non-negative NSInteger; clamp defensively.
                    ane_core_count = Some(cores.max(0) as u32);
                    break;
                }
                i += 1;
            }
        }

        sys::objc_autoreleasePoolPop(pool);

        CoreMlCapabilities {
            total_devices,
            ane_core_count,
        }
    };

    if caps.ane_core_count.is_none() {
        return Err(VokraError::BackendUnavailable(format!(
            "no Apple Neural Engine among the {} CoreML compute device(s) on this host \
             (CoreML backend requires the ANE; no silent CPU fall back, FR-EX-08)",
            caps.total_devices
        )));
    }

    Ok(caps)
}

/// Non-Apple stub: the CoreML backend is compiled out, so probing always fails
/// explicitly (FR-EX-08 / NFR-RL-06 — never a silent CPU fall back).
///
/// # Errors
///
/// Always returns [`VokraError::BackendUnavailable`].
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub fn vokra_coreml_probe() -> Result<CoreMlCapabilities> {
    Err(VokraError::BackendUnavailable(
        "CoreML backend is not compiled for this target (only macOS / iOS)".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn probe_detects_the_ane_on_apple_silicon() {
        // Runs on an Apple-silicon host (dev machine / CI Apple-silicon
        // runner). A host with no ANE (an Intel Mac, or a hosted runner that
        // hides the Neural Engine) is a legitimate BackendUnavailable — a
        // probe-gated skip, never a fabricated pass (FR-EX-08).
        match vokra_coreml_probe() {
            Ok(caps) => {
                assert!(caps.has_ane(), "Ok(_) must carry an ANE core count");
                assert!(
                    caps.ane_core_count.unwrap_or(0) >= 1,
                    "ANE core count must be >= 1 when present"
                );
                assert!(caps.total_devices >= 1, "at least CPU + ANE expected");
                eprintln!("vokra_coreml_probe: {}", caps.summary());
            }
            Err(VokraError::BackendUnavailable(msg)) => {
                eprintln!("no ANE on this host; skipping ({msg})");
            }
            Err(other) => panic!("probe must be Ok or BackendUnavailable, got {other:?}"),
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn probe_is_explicit_error_off_apple() {
        assert!(matches!(
            vokra_coreml_probe(),
            Err(VokraError::BackendUnavailable(_))
        ));
    }
}
