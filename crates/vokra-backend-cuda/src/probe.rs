//! CUDA device probe (M2-03-T05/T06; FR-BE-08 / NFR-RL-06).
//!
//! [`vokra_cuda_probe`] is the CUDA analogue of `vokra_metal_probe`: it reports
//! whether a usable NVIDIA driver + CUDA-capable GPU exists and, if so, the
//! driver version, device count, device 0's name and its compute capability.
//! **A missing driver / absent GPU / incompatible driver is an explicit
//! [`VokraError::BackendUnavailable`] — never a silent fall back to the CPU
//! backend** (FR-BE-08 / NFR-RL-06 permanent constraint). Whether to run on the
//! CPU instead is the *caller's* explicit backend choice, not a decision this
//! layer makes.
//!
//! It also embodies the NVIDIA EULA "install model": the driver is *detected*
//! at runtime via dlopen (no bundling — `third_party/NVIDIA-EULA.md`), so this
//! function compiles on a CUDA-less host (e.g. an Apple Mac) and returns
//! `BackendUnavailable` there at runtime.
//!
//! cuDNN detection (optional, non-required — FR-BE-08 point 4) and the full
//! CUDA/cuDNN version-compatibility gate are follow-on M2-03 tickets (T06);
//! this foundation slice reports the driver + device essentials.

use vokra_core::{Result, VokraError};

/// What [`vokra_cuda_probe`] discovered about the host's CUDA device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaCapabilities {
    /// The CUDA driver version as reported by `cuDriverGetVersion`, encoded as
    /// `major * 1000 + minor * 10` (e.g. `12040` = CUDA 12.4).
    pub driver_version: i32,
    /// Number of CUDA-capable devices (`cuDeviceGetCount`); always `>= 1` when
    /// this struct is returned.
    pub device_count: u32,
    /// `cuDeviceGetName` of device 0 (e.g. `"NVIDIA GeForce RTX 4090"`).
    pub device_name: String,
    /// Compute-capability major version of device 0 (e.g. `8` for Ada/Ampere).
    pub compute_capability_major: u32,
    /// Compute-capability minor version of device 0 (e.g. `9` on an RTX 4090).
    pub compute_capability_minor: u32,
}

impl CudaCapabilities {
    /// Human-readable one-line summary, e.g.
    /// `"NVIDIA GeForce RTX 4090 (compute 8.9, driver 12040, 1 device(s))"`.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} (compute {}.{}, driver {}, {} device(s))",
            self.device_name,
            self.compute_capability_major,
            self.compute_capability_minor,
            self.driver_version,
            self.device_count,
        )
    }
}

/// Detects the NVIDIA driver and device 0's capabilities (Unix / Windows).
///
/// # Errors
///
/// Returns [`VokraError::BackendUnavailable`] when the CUDA driver library is
/// absent (no NVIDIA GPU / driver not installed / non-NVIDIA host), when no
/// CUDA-capable device is present, or when a driver call fails. This is the
/// deliberate *explicit error* of FR-BE-08 / NFR-RL-06: the CUDA backend never
/// silently degrades to the CPU.
#[cfg(any(unix, windows))]
pub fn vokra_cuda_probe() -> Result<CudaCapabilities> {
    use core::ffi::{c_char, c_int};

    use crate::sys;

    // Load the driver (dlopen). Absent on a CUDA-less host → BackendUnavailable.
    let driver = sys::CudaDriver::load()?;

    // SAFETY: `cu_init` is the resolved `cuInit`; flag 0 is the only defined
    // value. Must run before any other driver call.
    let r = unsafe { (driver.cu_init)(0) };
    sys::check(&driver, r, "cuInit")?;

    let mut version: c_int = 0;
    // SAFETY: `cu_driver_get_version` writes the encoded version into `version`.
    let r = unsafe { (driver.cu_driver_get_version)(&mut version) };
    sys::check(&driver, r, "cuDriverGetVersion")?;

    let mut count: c_int = 0;
    // SAFETY: `cu_device_get_count` writes the device count into `count`.
    let r = unsafe { (driver.cu_device_get_count)(&mut count) };
    sys::check(&driver, r, "cuDeviceGetCount")?;
    if count <= 0 {
        return Err(VokraError::BackendUnavailable(
            "CUDA driver present but no CUDA-capable GPU detected (device count 0)".to_owned(),
        ));
    }

    // Device 0 identity + compute capability.
    let mut dev: sys::CUdevice = 0;
    // SAFETY: `cu_device_get` writes the ordinal-0 device handle into `dev`.
    let r = unsafe { (driver.cu_device_get)(&mut dev, 0) };
    sys::check(&driver, r, "cuDeviceGet")?;

    let mut name_buf = [0u8; 256];
    // SAFETY: `cu_device_get_name` writes up to `name_buf.len()` bytes (a
    // NUL-terminated name) into the buffer; `dev` is a valid device handle.
    let r = unsafe {
        (driver.cu_device_get_name)(
            name_buf.as_mut_ptr() as *mut c_char,
            name_buf.len() as c_int,
            dev,
        )
    };
    sys::check(&driver, r, "cuDeviceGetName")?;
    let device_name = sys::name_from_buf(&name_buf);

    let mut major: c_int = 0;
    let mut minor: c_int = 0;
    // SAFETY: `cu_device_get_attribute` writes the attribute value into the
    // out-param; the enum value is the compute-capability major selector; `dev`
    // is valid.
    let r = unsafe {
        (driver.cu_device_get_attribute)(
            &mut major,
            sys::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            dev,
        )
    };
    sys::check(&driver, r, "cuDeviceGetAttribute(compute_capability_major)")?;
    // SAFETY: as above, for the compute-capability minor selector.
    let r = unsafe {
        (driver.cu_device_get_attribute)(
            &mut minor,
            sys::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            dev,
        )
    };
    sys::check(&driver, r, "cuDeviceGetAttribute(compute_capability_minor)")?;

    Ok(CudaCapabilities {
        driver_version: version,
        device_count: count as u32,
        device_name,
        compute_capability_major: major.max(0) as u32,
        compute_capability_minor: minor.max(0) as u32,
    })
}

/// No-dynamic-loader stub (targets without dlopen / LoadLibrary, e.g. WASM): the
/// CUDA backend cannot be reached, so probing always fails explicitly (FR-BE-08
/// / NFR-RL-06 — never a silent CPU fall back).
///
/// # Errors
///
/// Always returns [`VokraError::BackendUnavailable`].
#[cfg(not(any(unix, windows)))]
pub fn vokra_cuda_probe() -> Result<CudaCapabilities> {
    Err(VokraError::BackendUnavailable(
        "CUDA backend requires a Unix or Windows target with a dynamic loader (dlopen / \
         LoadLibrary); not available on this target"
            .to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probe never panics and gives an honest verdict on any host:
    /// - on a real CUDA host (vast.ai RTX 4090): `Ok` with a non-empty name;
    /// - on this Apple Mac / any CUDA-less host: an explicit
    ///   `BackendUnavailable` (NFR-RL-06, no silent CPU fall back).
    ///
    /// The `Ok` branch is only exercised on the vast.ai GPU runner.
    #[test]
    fn probe_is_honest_and_never_panics() {
        match vokra_cuda_probe() {
            Ok(caps) => {
                assert!(
                    !caps.device_name.is_empty(),
                    "a detected device must be named"
                );
                assert!(caps.device_count >= 1);
                eprintln!("vokra_cuda_probe: {}", caps.summary());
            }
            Err(VokraError::BackendUnavailable(msg)) => {
                eprintln!("vokra_cuda_probe unavailable (expected off a CUDA host): {msg}");
            }
            Err(other) => {
                panic!("probe must return BackendUnavailable off a CUDA host, got {other}")
            }
        }
    }
}
