//! Vulkan device / GPU-family probe (M3-02-T30 / T31; FR-EX-08 / NFR-RL-06).
//!
//! [`vokra_vulkan_probe`] is the Vulkan analogue of `vokra_metal_probe`
//! (M2-01) / `vokra_cuda_probe` (M2-03): it reports whether a usable Vulkan
//! loader + physical device exists and, if so, the device's identity, Vulkan
//! version, vendor family, and — where cheaply available — a coarse
//! capability bit for subgroup / cooperative-matrix support.
//!
//! **A missing loader / absent device / driver too old is an explicit
//! [`VokraError::BackendUnavailable`] — never a silent fall back to the CPU
//! backend** (FR-EX-08 permanent constraint, NFR-RL-06). Whether to run on the
//! CPU instead is the *caller's* explicit backend choice, not a decision this
//! layer makes.
//!
//! # Scope of this slice
//!
//! - Loader load + version query;
//! - `VkInstance` create + `VkPhysicalDevice` enumerate;
//! - Vendor / device-name / device-type readout for the first physical device;
//! - Coarse subgroup-vs-cooperative-matrix capability inference from the
//!   Vulkan API version reported (`>= 1.1` for subgroup, `>= 1.3` +
//!   VK_KHR_cooperative_matrix extension check for coop-matrix).
//!
//! Detailed extension-list walks / driver-version cross-checks (Adreno / Mali
//! / Immortalis generation gates) are M3-02-T30 follow-up once a real Android
//! runtime is available (M3-18 owner run). This foundation slice keeps the
//! probe honest on any host today.

use vokra_core::{Result, VokraError};

/// What [`vokra_vulkan_probe`] discovered about the host's Vulkan
/// installation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VulkanCapabilities {
    /// Vulkan API version reported by the loader
    /// (`vkEnumerateInstanceVersion`), encoded per the Vulkan spec
    /// (`VK_MAKE_API_VERSION`). Split into `api_version_major` /
    /// `api_version_minor` for convenience.
    pub api_version: u32,
    /// Convenience: major component of `api_version`.
    pub api_version_major: u32,
    /// Convenience: minor component of `api_version`.
    pub api_version_minor: u32,
    /// Number of physical devices enumerated (`vkEnumeratePhysicalDevices`);
    /// always `>= 1` when this struct is returned.
    pub device_count: u32,
    /// `VkPhysicalDeviceProperties.deviceName` of device 0 (e.g.
    /// `"llvmpipe (LLVM 15.0.7, 256 bits)"` on lavapipe or
    /// `"NVIDIA GeForce RTX 4090"` on a discrete card).
    pub device_name: String,
    /// `VkPhysicalDeviceProperties.vendorID` of device 0 (`0x10de` = NVIDIA,
    /// `0x1002` = AMD, `0x8086` = Intel, `0x5143` = Qualcomm Adreno,
    /// `0x13b5` = ARM Mali, `0x10005` = Mesa software renderer / lavapipe).
    pub vendor_id: u32,
    /// `VkPhysicalDeviceProperties.deviceType` of device 0 — one of
    /// `VK_PHYSICAL_DEVICE_TYPE_*` (0=other, 1=integrated, 2=discrete,
    /// 3=virtual, 4=CPU/software).
    pub device_type: u32,
    /// Whether Vokra classifies this loader/device pair as "usable" for the
    /// M3-02 subgroup-only fallback path (Vulkan 1.1+, device type is not
    /// `OTHER`).
    pub subgroup_ready: bool,
    /// Whether the reported Vulkan API version is `>= 1.3` **AND** either
    /// `VK_KHR_cooperative_matrix` or `VK_NV_cooperative_matrix` is present
    /// on the device (M3-02-T30 real extension walk, upgraded from the
    /// foundation-slice coarse "API >= 1.3" gate). Both preconditions are
    /// necessary; the coop-matrix pipeline is only bound when this is `true`.
    pub coop_matrix_precondition_met: bool,
    /// Whether the device exposes the promoted `VK_KHR_cooperative_matrix`
    /// (Vulkan 1.3+ KHR extension). `false` on either an older device or one
    /// with only the NVIDIA vendor extension.
    pub has_khr_cooperative_matrix: bool,
    /// Whether the device exposes the NVIDIA `VK_NV_cooperative_matrix`
    /// (older, pre-KHR promotion; still shipped on Turing / Ampere / Ada).
    pub has_nv_cooperative_matrix: bool,
    /// The index of the compute-capable queue family on device 0
    /// (M3-02-T07). `None` on a physical device that exposes no queue family
    /// with the compute bit set — impossible on any Vulkan-conformant GPU
    /// (spec §5.3.1 requires it), so `None` means Vokra will reject the
    /// device upstream with an explicit `BackendUnavailable`.
    ///
    /// The selection policy prefers a compute-*only* family (no graphics bit)
    /// where available — dedicated compute queues avoid contention with the
    /// display path on Adreno / Mali. See
    /// [`crate::context::VulkanInstance::find_compute_queue_family`] for the
    /// full algorithm.
    pub compute_queue_family_index: Option<u32>,
}

impl VulkanCapabilities {
    /// Human-readable one-line summary, e.g.
    /// `"llvmpipe (LLVM 15.0.7) — Vulkan 1.3, type=CPU, vendor=0x10005 — subgroup:yes coop-matrix:precondition"`.
    #[must_use]
    pub fn summary(&self) -> String {
        // VkPhysicalDeviceType enum values (spec §37.1). Repeated here so
        // `probe.rs` remains compilable on non-Vulkan / feature-off builds
        // (where `crate::sys` is not built).
        let ty = match self.device_type {
            0 => "OTHER",
            1 => "INTEGRATED",
            2 => "DISCRETE",
            3 => "VIRTUAL",
            4 => "CPU",
            _ => "unknown",
        };
        // Report the extension source when we detect coop-matrix support.
        let coop = if self.coop_matrix_precondition_met {
            if self.has_khr_cooperative_matrix {
                "khr"
            } else if self.has_nv_cooperative_matrix {
                "nv"
            } else {
                // Both false with precondition true is impossible (see
                // `vokra_vulkan_probe`), but keep the surface honest.
                "unknown"
            }
        } else {
            "no"
        };
        let subgroup = if self.subgroup_ready { "yes" } else { "no" };
        let queue = match self.compute_queue_family_index {
            Some(idx) => format!("compute-q:{idx}"),
            None => "compute-q:none".to_owned(),
        };
        format!(
            "{} — Vulkan {}.{}, type={ty}, vendor=0x{:x} — subgroup:{subgroup} coop-matrix:{coop} \
             {queue}",
            self.device_name, self.api_version_major, self.api_version_minor, self.vendor_id,
        )
    }

    /// Vendor-family classification, used by T30 to reject boards below the
    /// Adreno 6xx / Mali G7x / Immortalis floor once a real Android runtime is
    /// in the loop. Reports the parent family only — the driver-side
    /// generation gate (Adreno 6xx vs 7xx, Mali G51 vs G710) is a follow-up.
    #[must_use]
    pub fn vendor_family(&self) -> VendorFamily {
        match self.vendor_id {
            0x10de => VendorFamily::Nvidia,
            0x1002 => VendorFamily::Amd,
            0x8086 => VendorFamily::Intel,
            0x5143 => VendorFamily::Adreno,
            0x13b5 => VendorFamily::Mali,
            // Imagination PowerVR
            0x1010 => VendorFamily::Imagination,
            // Mesa software (lavapipe) reports 0x10005; llvmpipe reports the
            // same after Vulkan-loader 1.3.
            0x10005 => VendorFamily::MesaSoftware,
            _ => VendorFamily::Unknown,
        }
    }
}

/// Coarse vendor family, driven off `VkPhysicalDeviceProperties.vendorID`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VendorFamily {
    /// NVIDIA discrete GPU (Turing / Ampere / Ada / Hopper — Vokra uses the
    /// CUDA backend on desktop NVIDIA, but Vulkan is the Android and
    /// non-CUDA-desktop path).
    Nvidia,
    /// AMD discrete / integrated GPU.
    Amd,
    /// Intel integrated / discrete GPU.
    Intel,
    /// Qualcomm Adreno (Android SoCs). Primary target for M3-02.
    Adreno,
    /// ARM Mali / Immortalis (Android SoCs). Primary target for M3-02.
    Mali,
    /// Imagination Technologies PowerVR.
    Imagination,
    /// Mesa software renderer — `lavapipe`, the CI fallback that runs Vulkan
    /// on the CPU. Numerical parity only; performance is not representative.
    MesaSoftware,
    /// Unknown vendor ID — Vokra treats this as usable if `subgroup_ready` is
    /// true, but flags it in the probe for logging.
    Unknown,
}

/// Detects the Vulkan loader and device 0's capabilities.
///
/// # Errors
///
/// Returns [`VokraError::BackendUnavailable`] when:
/// - the Vulkan loader library is absent (no `libvulkan` on this host, no
///   MoltenVK installed, Apple Mac case);
/// - the loader is pre-1.1 (Vokra targets Vulkan 1.1+);
/// - `vkCreateInstance` fails;
/// - no physical device is enumerated (loader present, no ICDs — install
///   `mesa-vulkan-drivers` for lavapipe, or the vendor's Vulkan userland).
///
/// This is the deliberate *explicit error* of FR-EX-08 / NFR-RL-06: the
/// Vulkan backend never silently degrades to the CPU.
#[cfg(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
))]
pub fn vokra_vulkan_probe() -> Result<VulkanCapabilities> {
    let instance = crate::context::VulkanInstance::new()?;

    let devices = instance.enumerate_physical_devices()?;
    if devices.is_empty() {
        return Err(VokraError::BackendUnavailable(
            "Vulkan loader present but no physical devices enumerated (install a Vulkan ICD, \
             e.g. mesa-vulkan-drivers for lavapipe, or the vendor driver)."
                .to_owned(),
        ));
    }
    let props = instance.get_physical_device_properties(devices[0]);
    let api = props.api_version;
    let major = crate::sys::api_version_major(api);
    let minor = crate::sys::api_version_minor(api);
    // `subgroup_ready` = Vulkan 1.1+ AND the device is not
    // `VK_PHYSICAL_DEVICE_TYPE_OTHER` (spec-defined "unclassified" — Vokra
    // does not commit to shipping subgroup pipelines against unclassified
    // hardware).
    let subgroup_ready = (major > 1 || (major == 1 && minor >= 1))
        && props.device_type != crate::sys::VK_PHYSICAL_DEVICE_TYPE_OTHER;
    // Cooperative-matrix path (M3-02-T30 upgraded from the coarse "API >= 1.3"
    // check to a real extension walk):
    //
    //   coop_matrix_precondition_met = api_1.3+ AND (VK_KHR_cooperative_matrix
    //                                                 OR VK_NV_cooperative_matrix)
    //
    // Vokra checks BOTH the promoted KHR and the NVIDIA vendor extension.
    // KHR was promoted in Vulkan 1.3, but NV's older extension is still the
    // only shipping form on Turing / Ampere / early Ada, so we accept either
    // for the "coop-matrix capable" verdict — the actual pipeline binding
    // (T14+) will select the correct dispatch path based on which extension
    // is present.
    let exts = instance.enumerate_device_extensions(devices[0])?;
    let has_khr_cooperative_matrix =
        crate::context::VulkanInstance::has_extension(&exts, "VK_KHR_cooperative_matrix");
    let has_nv_cooperative_matrix =
        crate::context::VulkanInstance::has_extension(&exts, "VK_NV_cooperative_matrix");
    let api_geq_1_3 = major > 1 || (major == 1 && minor >= 3);
    let coop_matrix_precondition_met =
        api_geq_1_3 && (has_khr_cooperative_matrix || has_nv_cooperative_matrix);
    // M3-02-T07 compute queue-family selection.
    let compute_queue_family_index = instance.find_compute_queue_family(devices[0]);

    Ok(VulkanCapabilities {
        api_version: api,
        api_version_major: major,
        api_version_minor: minor,
        device_count: devices.len() as u32,
        device_name: crate::sys::name_from_buf(&props.device_name),
        vendor_id: props.vendor_id,
        device_type: props.device_type,
        subgroup_ready,
        coop_matrix_precondition_met,
        has_khr_cooperative_matrix,
        has_nv_cooperative_matrix,
        compute_queue_family_index,
    })
}

/// Stub for targets that cannot host a Vulkan loader (`macos` / `ios` /
/// `wasm`) or for default builds without the `vulkan` feature. The Vulkan
/// backend cannot be reached, so probing always fails explicitly (FR-EX-08 /
/// NFR-RL-06 — never a silent CPU fall back).
///
/// # Errors
///
/// Always returns [`VokraError::BackendUnavailable`].
#[cfg(not(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
)))]
pub fn vokra_vulkan_probe() -> Result<VulkanCapabilities> {
    Err(VokraError::BackendUnavailable(
        "Vulkan backend not compiled for this target / feature set (needs --features vulkan on \
         Linux / Android / Windows)."
            .to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probe never panics and gives an honest verdict on any host:
    /// - on a Vulkan-capable Linux + lavapipe host: `Ok` with a non-empty
    ///   device name (parity CI, T36);
    /// - on this Apple Mac / non-Vulkan host: an explicit
    ///   `BackendUnavailable` (NFR-RL-06, no silent CPU fall back).
    #[test]
    fn probe_is_honest_and_never_panics() {
        match vokra_vulkan_probe() {
            Ok(caps) => {
                assert!(
                    !caps.device_name.is_empty(),
                    "a detected device must be named"
                );
                assert!(caps.device_count >= 1);
                assert!(caps.api_version_major >= 1);
                eprintln!("vokra_vulkan_probe: {}", caps.summary());
            }
            Err(VokraError::BackendUnavailable(msg)) => {
                eprintln!("vokra_vulkan_probe unavailable (expected off Vulkan host): {msg}");
            }
            Err(other) => {
                panic!("probe must return BackendUnavailable off a Vulkan host, got {other}")
            }
        }
    }

    #[test]
    fn vendor_family_maps_known_ids() {
        for (id, expected) in [
            (0x10de_u32, VendorFamily::Nvidia),
            (0x1002, VendorFamily::Amd),
            (0x8086, VendorFamily::Intel),
            (0x5143, VendorFamily::Adreno),
            (0x13b5, VendorFamily::Mali),
            (0x1010, VendorFamily::Imagination),
            (0x10005, VendorFamily::MesaSoftware),
            (0xdead_beef, VendorFamily::Unknown),
        ] {
            let caps = VulkanCapabilities {
                api_version: 0,
                api_version_major: 1,
                api_version_minor: 1,
                device_count: 1,
                device_name: "test".to_owned(),
                vendor_id: id,
                device_type: 2, // VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU

                subgroup_ready: true,
                coop_matrix_precondition_met: false,
                has_khr_cooperative_matrix: false,
                has_nv_cooperative_matrix: false,
                compute_queue_family_index: Some(0),
            };
            assert_eq!(caps.vendor_family(), expected, "vendor 0x{id:x}");
        }
    }

    #[test]
    fn summary_mentions_compute_queue_and_subgroup() {
        // The summary format is human-readable log surface; keep the queue-
        // index reporting explicit so an owner-side log dump on Android tells
        // us at a glance which family Vokra picked (M3-02-T07 observability).
        let mut caps = VulkanCapabilities {
            api_version: 0x0040_1000,
            api_version_major: 1,
            api_version_minor: 1,
            device_count: 1,
            device_name: "test-device".to_owned(),
            vendor_id: 0x5143, // Adreno
            device_type: 1,    // integrated
            subgroup_ready: true,
            coop_matrix_precondition_met: false,
            has_khr_cooperative_matrix: false,
            has_nv_cooperative_matrix: false,
            compute_queue_family_index: Some(1),
        };
        let s = caps.summary();
        assert!(s.contains("compute-q:1"), "summary missing queue: `{s}`");
        assert!(s.contains("subgroup:yes"));
        assert!(s.contains("coop-matrix:no"));

        // A device with no compute queue family (impossible in the spec, but
        // the format must still be honest).
        caps.compute_queue_family_index = None;
        let s = caps.summary();
        assert!(s.contains("compute-q:none"));
    }

    /// M3-02-T30 upgrade — the "precondition met" verdict now requires an
    /// actual extension (KHR or NV) in addition to Vulkan 1.3+. Report the
    /// source (khr / nv) in the summary so debugging on Android is
    /// unambiguous.
    #[test]
    fn summary_reports_coop_matrix_source() {
        let base = VulkanCapabilities {
            api_version: 0x0040_3000,
            api_version_major: 1,
            api_version_minor: 3,
            device_count: 1,
            device_name: "gemm-capable".to_owned(),
            vendor_id: 0x10de, // NVIDIA
            device_type: 2,
            subgroup_ready: true,
            coop_matrix_precondition_met: true,
            has_khr_cooperative_matrix: true,
            has_nv_cooperative_matrix: false,
            compute_queue_family_index: Some(0),
        };
        assert!(base.summary().contains("coop-matrix:khr"));
        let nv = VulkanCapabilities {
            has_khr_cooperative_matrix: false,
            has_nv_cooperative_matrix: true,
            ..base.clone()
        };
        assert!(nv.summary().contains("coop-matrix:nv"));
        let none = VulkanCapabilities {
            coop_matrix_precondition_met: false,
            has_khr_cooperative_matrix: false,
            has_nv_cooperative_matrix: false,
            ..base
        };
        assert!(none.summary().contains("coop-matrix:no"));
    }
}
