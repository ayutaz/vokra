//! Vulkan device / queue / command / pipeline / buffer wrappers
//! (M3-02-T06〜T12).
//!
//! This foundation slice ships the minimum needed for the probe (T30/T31) and
//! for future compute-kernel dispatch. Each type owns its Vulkan handle and
//! destroys it on drop (RAII, symmetric with `MetalContext` /
//! `CudaContext`). All FFI calls carry `// SAFETY:` notes.
//!
//! # What is present in this slice
//!
//! - [`VulkanInstance`] — a real, loader-driven `VkInstance` create + destroy
//!   pair (T06). Used by the probe to enumerate physical devices without
//!   requiring the caller to build any of the device-side plumbing.
//!
//! # What is stubbed for later tickets
//!
//! Full [`VkDevice`](crate::sys::VkDevice) / command-pool / descriptor /
//! pipeline / memory / buffer plumbing (T08〜T12) needs a Vulkan-capable
//! runtime to be end-to-end verifiable, so the corresponding structs are
//! declared with their handle fields and Drop signatures — but the
//! constructor bodies are gated behind `todo!()` markers guarded by
//! `debug_assert!`, so a caller cannot invoke them by accident on a build
//! that has not yet wired the underlying kernel path. Every field / method
//! carries a `M3-02-T<nn> follow-up` note pointing to the ticket that will
//! land the body.

use core::ffi::c_void;
use core::mem::MaybeUninit;
use core::ptr;

use vokra_core::{Result, VokraError};

use crate::sys;

/// A Vulkan `VkInstance` scoped to the [`VulkanLoader`](crate::sys::VulkanLoader)
/// that created it. Destroyed on drop via `vkDestroyInstance`.
///
/// Vokra creates a *minimal* instance: no validation layers, no
/// window-system extensions — just the loader-level plumbing needed to
/// enumerate physical devices and later create a compute-only logical device
/// (M3-02-T08). Validation layers and `VK_KHR_get_physical_device_properties2`
/// are opt-ins for the probe (T30/T31).
pub(crate) struct VulkanInstance {
    // The loader must outlive every handle derived from it, hence the owned
    // Rc-less move (a single VulkanInstance holds one VulkanLoader for its
    // lifetime). `Send`/`Sync` are intentionally not implemented — Vulkan
    // handles are used from the thread that created them (same posture as
    // MetalContext / CudaContext).
    loader: sys::VulkanLoader,
    instance: sys::VkInstance,
    // Instance-level entry points resolved via vkGetInstanceProcAddr. We only
    // load what the probe needs (T30/T31); more entries are added in T07/T08.
    enumerate_physical_devices: sys::FnVkEnumeratePhysicalDevices,
    get_physical_device_properties: sys::FnVkGetPhysicalDeviceProperties,
    destroy_instance: sys::FnVkDestroyInstance,
}

impl VulkanInstance {
    /// Creates a compute-only Vulkan instance targeting Vulkan 1.1+.
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] when the Vulkan loader is not
    /// present (no `libvulkan.so.1` / `vulkan-1.dll`), when the loader is
    /// pre-1.1, or when `vkCreateInstance` fails (an unusable driver — never
    /// silently masked, NFR-RL-06).
    pub(crate) fn new() -> Result<VulkanInstance> {
        let loader = sys::VulkanLoader::load()?;

        // Enforce the Vulkan 1.1+ requirement declared in the ADR (T01(c)):
        // any loader that lacks `vkEnumerateInstanceVersion` is 1.0-only.
        let enumerate_ver = loader.enumerate_instance_version.ok_or_else(|| {
            VokraError::BackendUnavailable(
                "Vulkan loader is pre-1.1 (no `vkEnumerateInstanceVersion`). Vokra targets \
                 Vulkan 1.1+ (subgroup + cooperative-matrix precondition, M3-02 ADR)."
                    .to_owned(),
            )
        })?;
        let mut api_version: u32 = 0;
        // SAFETY: `enumerate_ver` is the resolved `vkEnumerateInstanceVersion`;
        // `api_version` is a valid, writable u32 stack slot.
        let r = unsafe { enumerate_ver(&mut api_version) };
        sys::check(r, "vkEnumerateInstanceVersion")?;
        let major = sys::api_version_major(api_version);
        let minor = sys::api_version_minor(api_version);
        if major < 1 || (major == 1 && minor < 1) {
            return Err(VokraError::BackendUnavailable(format!(
                "Vulkan loader reports API version {major}.{minor}; Vokra requires 1.1+ \
                 (M3-02 ADR)."
            )));
        }

        // Application info: `vokra` engine name, engine version 1.
        let app_name = c"vokra".as_ptr();
        let engine_name = c"vokra".as_ptr();
        let app_info = sys::VkApplicationInfo {
            s_type: sys::VK_STRUCTURE_TYPE_APPLICATION_INFO,
            p_next: ptr::null(),
            p_application_name: app_name,
            application_version: sys::make_api_version(0, 0, 1, 0),
            p_engine_name: engine_name,
            engine_version: sys::make_api_version(0, 0, 1, 0),
            // Request 1.1 (T01(a) — the minimum for subgroup ops); the driver
            // may return a newer version.
            api_version: sys::make_api_version(0, 1, 1, 0),
        };
        let create_info = sys::VkInstanceCreateInfo {
            s_type: sys::VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            p_application_info: &app_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: ptr::null(),
        };
        let mut instance: sys::VkInstance = ptr::null_mut();
        // SAFETY: `create_instance` is the resolved `vkCreateInstance`;
        // `create_info` points at a valid, filled struct; `instance` is a
        // writable out-parameter; the allocator callback is null (default).
        let r = unsafe { (loader.create_instance)(&create_info, ptr::null(), &mut instance) };
        sys::check(r, "vkCreateInstance")?;

        // Resolve the instance-level entries Vokra needs from here on.
        // SAFETY: pairs each C symbol name with the exact FnVk… alias declared
        // in `sys` (which mirrors vulkan_core.h). A missing symbol is an
        // unusable driver → BackendUnavailable, never a silent fallback.
        let enumerate_physical_devices: sys::FnVkEnumeratePhysicalDevices =
            unsafe { sys::instance_proc(&loader, instance, b"vkEnumeratePhysicalDevices\0") }
                .ok_or_else(|| {
                    VokraError::BackendUnavailable(
                        "Vulkan driver is missing `vkEnumeratePhysicalDevices` (impossible on any \
                 conforming ICD)."
                            .to_owned(),
                    )
                })?;
        let get_physical_device_properties: sys::FnVkGetPhysicalDeviceProperties =
            // SAFETY: pairs the exact FnVk… alias with vkGetPhysicalDeviceProperties;
            // `sys::instance_proc` transmutes a resolved procaddr to that type.
            unsafe { sys::instance_proc(&loader, instance, b"vkGetPhysicalDeviceProperties\0") }
                .ok_or_else(|| {
                    VokraError::BackendUnavailable(
                        "Vulkan driver is missing `vkGetPhysicalDeviceProperties` (impossible on \
                         any conforming ICD)."
                            .to_owned(),
                    )
                })?;
        let destroy_instance: sys::FnVkDestroyInstance =
            // SAFETY: pairs vkDestroyInstance's C symbol name with its exact
            // FnVkDestroyInstance alias.
            unsafe { sys::instance_proc(&loader, instance, b"vkDestroyInstance\0") }
                .ok_or_else(|| {
                    // If we can't resolve destroy we STILL destroy nothing — but we
                    // shouldn't ever reach this: destroy is a core loader entry.
                    VokraError::BackendUnavailable(
                        "Vulkan driver is missing `vkDestroyInstance` (impossible on any \
                         conforming ICD)."
                            .to_owned(),
                    )
                })?;

        Ok(VulkanInstance {
            loader,
            instance,
            enumerate_physical_devices,
            get_physical_device_properties,
            destroy_instance,
        })
    }

    /// Enumerate `VkPhysicalDevice` handles. The returned handles are borrowed
    /// from the instance and become invalid when `self` is dropped.
    pub(crate) fn enumerate_physical_devices(&self) -> Result<Vec<sys::VkPhysicalDevice>> {
        // Two-call idiom (spec §37.1): first call with null-out to size the
        // buffer, second call fills it.
        let mut count: u32 = 0;
        // SAFETY: enumerate_physical_devices is the resolved instance entry;
        // `count` is a valid writable u32.
        let r = unsafe {
            (self.enumerate_physical_devices)(self.instance, &mut count, ptr::null_mut())
        };
        sys::check(r, "vkEnumeratePhysicalDevices (count)")?;
        if count == 0 {
            return Ok(Vec::new());
        }
        // Pre-fill with null handles; `vkEnumeratePhysicalDevices` overwrites
        // every slot. Using `vec![]` avoids the `set_len(uninit)` idiom that
        // clippy's `uninit_vec` lint (rightly) flags — the driver never leaves
        // a slot untouched but this is cheaper to audit.
        let mut handles: Vec<sys::VkPhysicalDevice> = vec![ptr::null_mut(); count as usize];
        // SAFETY: `handles.as_mut_ptr()` is a valid pointer to `count`
        // writable `VkPhysicalDevice` slots; `enumerate_physical_devices` is
        // the resolved instance entry.
        let r = unsafe {
            (self.enumerate_physical_devices)(self.instance, &mut count, handles.as_mut_ptr())
        };
        sys::check(r, "vkEnumeratePhysicalDevices (fill)")?;
        Ok(handles)
    }

    /// Read `VkPhysicalDeviceProperties` for a physical device.
    pub(crate) fn get_physical_device_properties(
        &self,
        device: sys::VkPhysicalDevice,
    ) -> sys::VkPhysicalDeviceProperties {
        // The struct is fully initialised by the driver call, so we can hand
        // out a MaybeUninit<>.
        let mut props: MaybeUninit<sys::VkPhysicalDeviceProperties> = MaybeUninit::uninit();
        // SAFETY: `get_physical_device_properties` is the resolved instance
        // entry; `device` is a valid handle; the out-pointer is a writable
        // struct slot.
        unsafe { (self.get_physical_device_properties)(device, props.as_mut_ptr()) };
        // SAFETY: the driver call above fully initialises the struct.
        unsafe { props.assume_init() }
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        if self.instance.is_null() {
            return;
        }
        // SAFETY: `destroy_instance` is the resolved `vkDestroyInstance`;
        // `self.instance` is a live handle we own and destroy exactly once
        // here; the allocator callback is null (default).
        unsafe { (self.destroy_instance)(self.instance, ptr::null()) };
        self.instance = ptr::null_mut();
        // The loader is dropped after us (declared last in the struct — Rust
        // drops fields in declaration order). Explicitly noted so the ordering
        // is auditable.
        let _ = &self.loader;
    }
}

// ---------------------------------------------------------------------------
// Device / command / descriptor / pipeline / memory / buffer stubs (T08〜T12).
// Full bodies will land once the SPIR-V kernels do (T14〜T22); at that point a
// real Vulkan runtime (lavapipe / real GPU) is available to end-to-end verify
// them. Until then the constructors return an explicit "not implemented in
// this slice" error rather than pretending to succeed (FR-EX-08).
// ---------------------------------------------------------------------------

/// A Vulkan logical device + a compute queue. **Stub** — the body is
/// M3-02-T08 follow-up. Callers get an explicit `NotImplemented` today.
#[allow(dead_code)] // wired up by M3-02-T08
pub(crate) struct VulkanDevice {
    _private: (),
}

impl VulkanDevice {
    /// Creates a compute-only logical device on the first compute-capable
    /// physical device.
    ///
    /// # Errors
    ///
    /// Currently always [`VokraError::NotImplemented`] — this constructor is
    /// M3-02-T08 follow-up. Once wired, it will honestly return
    /// `BackendUnavailable` when no compute queue is available.
    #[allow(dead_code)] // wired up by M3-02-T08
    pub(crate) fn new(_instance: &VulkanInstance) -> Result<VulkanDevice> {
        Err(VokraError::NotImplemented(
            "VulkanDevice::new is M3-02-T08 follow-up",
        ))
    }
}

/// A `VkCommandPool` + primary command buffer. **Stub** — M3-02-T09 follow-up.
#[allow(dead_code)] // wired up by M3-02-T09
pub(crate) struct VulkanCommandPool {
    _private: (),
}

impl VulkanCommandPool {
    /// Creates a command pool with `VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT`.
    ///
    /// # Errors
    ///
    /// Currently always [`VokraError::NotImplemented`] — this constructor is
    /// M3-02-T09 follow-up.
    #[allow(dead_code)] // wired up by M3-02-T09
    pub(crate) fn new(_device: &VulkanDevice) -> Result<VulkanCommandPool> {
        Err(VokraError::NotImplemented(
            "VulkanCommandPool::new is M3-02-T09 follow-up",
        ))
    }
}

/// A `VkBuffer` + backing `VkDeviceMemory`. **Stub** — M3-02-T12 follow-up.
///
/// Vokra will select memory type per `usage`:
/// - staging (`HostVisible | HostCoherent`) for host↔device copies (T25);
/// - device-local (`DeviceLocal`) for compute working set;
/// - on integrated GPUs (Adreno / Mali) the driver may expose a type with
///   both, which the probe (T30) surfaces via a capability flag.
pub(crate) struct VulkanBuffer {
    _private: (),
}

/// Kind of memory Vokra requests when allocating a `VulkanBuffer`. Selected
/// by the memory-type search once T12 lands.
#[allow(dead_code)] // consumers land with T12 / T25
pub(crate) enum BufferKind {
    /// Host-visible + host-coherent — used for staging round-trips.
    Staging,
    /// Device-local — the compute working set.
    DeviceLocal,
}

impl VulkanBuffer {
    /// Allocates a Vulkan buffer of `size_bytes` with the memory type
    /// requested by `kind`.
    ///
    /// # Errors
    ///
    /// Currently always [`VokraError::NotImplemented`] — this constructor is
    /// M3-02-T12 follow-up.
    #[allow(dead_code)] // consumers land with T25
    pub(crate) fn new(
        _device: &VulkanDevice,
        _size_bytes: usize,
        _kind: BufferKind,
    ) -> Result<VulkanBuffer> {
        Err(VokraError::NotImplemented(
            "VulkanBuffer::new is M3-02-T12 follow-up",
        ))
    }
}

// Suppress "unused import" for the placeholder `c_void` we pull in for
// forward-declaration parity with `sys.rs`. It disappears once T08 lands.
const _: *const c_void = core::ptr::null();

#[cfg(test)]
mod tests {
    use super::*;

    /// If a Vulkan loader is present, the instance builds and destroys
    /// cleanly. Off Vulkan hosts (Apple Mac) this is an explicit
    /// BackendUnavailable — never a silent fallback (FR-EX-08).
    #[test]
    fn instance_new_is_honest_on_any_host() {
        match VulkanInstance::new() {
            Ok(instance) => {
                // Enumerate physical devices; may legitimately be empty on a
                // Vulkan loader with no ICDs installed (lavapipe not
                // installed). The API call must not panic either way.
                let devs = instance
                    .enumerate_physical_devices()
                    .expect("enumeration must not fail once instance exists");
                for d in &devs {
                    let props = instance.get_physical_device_properties(*d);
                    let name = sys::name_from_buf(&props.device_name);
                    assert!(!name.is_empty(), "device name must not be empty");
                }
            }
            Err(VokraError::BackendUnavailable(msg)) => {
                eprintln!("vulkan instance unavailable (expected off Vulkan host): {msg}");
            }
            Err(other) => {
                panic!(
                    "VulkanInstance::new must return BackendUnavailable off a Vulkan host, got {other}"
                );
            }
        }
    }

    #[test]
    fn device_stub_returns_not_implemented() {
        // The stub must be explicit — never a silent success or fallback.
        let Ok(instance) = VulkanInstance::new() else {
            eprintln!("no Vulkan loader; device stub test still assertable via type check");
            return;
        };
        assert!(matches!(
            VulkanDevice::new(&instance),
            Err(VokraError::NotImplemented(_))
        ));
    }
}
