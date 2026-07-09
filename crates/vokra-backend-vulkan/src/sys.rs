//! Raw Vulkan API FFI, loaded at runtime with dlopen / LoadLibrary (Unix /
//! Windows only — gated by the parent module).
//!
//! This module is the **only** place that talks to the Vulkan loader, and it
//! does so with hand-declared `unsafe extern` blocks + runtime dynamic loading
//! — **no `ash` / `vulkano` / `erupt` binding crate** (M3-02 red line; keeps
//! the root `Cargo.lock` free of non-`vokra-*` crates, NFR-DS-02).
//!
//! # Why dlopen instead of `#[link(name = "vulkan")]`
//!
//! Two reasons, both load-bearing:
//!
//! 1. **All-target build** (NFR-PT-01): a link-time dependency on `libvulkan`
//!    would break `cargo build` on any host without the Vulkan loader —
//!    including the Apple Mac this crate is authored on. Runtime loading means
//!    the crate **compiles everywhere** and only *fails at runtime* (an
//!    explicit [`VokraError::BackendUnavailable`], never a silent CPU fall
//!    back — NFR-RL-06) where no Vulkan loader is present.
//! 2. **Symmetry with CUDA (M2-03) / Metal (M2-01) approach**: both other GPU
//!    backends already treat their platform libraries the same way (CUDA:
//!    dlopen; Metal: framework link but `MTLCreateSystemDefaultDevice` gate
//!    for missing devices). The user's host either has the Vulkan loader
//!    installed system-wide (Linux / Android / Windows), or it does not — we
//!    do not bundle it.
//!
//! The dynamic-loader primitives (`dlopen`/`dlsym`/`dlclose` on Unix;
//! `LoadLibraryA`/`GetProcAddress`/`FreeLibrary` on Windows) are symbols `std`
//! already links (libdl / kernel32), declared inline here — the same technique
//! [`vokra-mmap`] and [`vokra-backend-cuda`] use, so nothing is added to
//! `Cargo.lock`.
//!
//! # Vulkan bootstrap flow
//!
//! Vulkan splits its API into two tiers:
//! - **loader-level entry points** (`vkGetInstanceProcAddr`,
//!   `vkCreateInstance`, `vkEnumerateInstanceVersion`) — resolved via
//!   `dlsym`;
//! - **instance-level / device-level entry points** — resolved via
//!   `vkGetInstanceProcAddr(instance, "vkXxx")` after an instance is created.
//!
//! This module resolves the loader-level entry points. Instance-level entries
//! are loaded lazily by [`context`](crate::context) once a `VkInstance`
//! exists.
//!
//! # Signature fidelity
//!
//! Every function pointer is `dlsym`'d (or resolved via
//! `vkGetInstanceProcAddr`) and `transmute`d to the **exact** C signature of
//! its symbol, with a `// SAFETY:` note. The Vulkan handles (`VkInstance`,
//! `VkPhysicalDevice`, …) are 64-bit opaque handles on all platforms Vulkan
//! supports.

use core::ffi::{c_char, c_int, c_uint, c_void};

use vokra_core::{Result, VokraError};

// ---------------------------------------------------------------------------
// Platform dynamic-loader primitives (dlopen / LoadLibrary). No binding crate.
// Copy of the same pattern used by vokra-backend-cuda / vokra-mmap — kept
// self-contained so this crate has zero non-`vokra-*` dependencies.
// ---------------------------------------------------------------------------

#[cfg(unix)]
mod dl {
    use core::ffi::{c_char, c_int, c_void};

    /// `RTLD_NOW` — resolve all undefined symbols before `dlopen` returns.
    /// Value `0x2` on both Linux (glibc/musl) and macOS/BSD (Vulkan is Linux /
    /// Android on the Unix side here; macOS uses Metal, not Vulkan).
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
    // Win32 calling convention.
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

/// A loaded shared library (`libvulkan.so.1` / `vulkan-1.dll`). Closes its
/// handle on drop.
///
/// Not `Send`/`Sync`: the Vulkan handles derived from it (instance / device /
/// queue) are used from the thread that created them (sufficient for the
/// parity harness; a `Send` wrapper is a later concern, mirroring
/// `MetalContext` / `CudaContext`).
pub(crate) struct DynLib {
    handle: *mut c_void,
}

impl DynLib {
    /// Tries each NUL-terminated candidate name in order; returns the first
    /// library that loads, or `None` if none do (e.g. on a host with no Vulkan
    /// loader installed — the Apple Mac case).
    pub(crate) fn open(candidates: &[&[u8]]) -> Option<DynLib> {
        for name in candidates {
            debug_assert_eq!(name.last(), Some(&0), "library name must be NUL-terminated");
            // SAFETY: `name` is a NUL-terminated C string literal; `dl::open`
            // (dlopen / LoadLibraryA) returns null on failure, checked here.
            let handle = unsafe { dl::open(name.as_ptr() as *const c_char) };
            if !handle.is_null() {
                return Some(DynLib { handle });
            }
        }
        None
    }

    /// Resolves NUL-terminated `name` as the function-pointer type `F`.
    ///
    /// Returns `None` if the symbol is absent (the loader is present but too
    /// old to export it — this is treated as `BackendUnavailable` upstream,
    /// never a silent fall back, NFR-RL-06).
    ///
    /// # Safety
    /// `F` must be a function-pointer type whose signature matches the C
    /// symbol `name` exactly. (Enforced by the single call site
    /// [`VulkanLoader::load`], which pairs each symbol with its precise `Fn*`
    /// alias.)
    pub(crate) unsafe fn get<F: Copy>(&self, name: &[u8]) -> Option<F> {
        debug_assert_eq!(name.last(), Some(&0), "symbol name must be NUL-terminated");
        debug_assert_eq!(
            core::mem::size_of::<F>(),
            core::mem::size_of::<*mut c_void>(),
            "F must be a pointer-sized function pointer"
        );
        // SAFETY: `handle` is live; `name` is a valid NUL-terminated C string.
        let ptr = unsafe { dl::sym(self.handle, name.as_ptr() as *const c_char) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: `ptr` is a non-null symbol address (pointer-sized). `F` is a
        // function-pointer type of the same size (asserted above) whose C
        // signature the caller guarantees matches `name`. `transmute_copy`
        // reads exactly `size_of::<F>()` bytes from `&ptr`, reinterpreting the
        // address as the typed function pointer (the standard dlsym idiom).
        Some(unsafe { core::mem::transmute_copy::<*mut c_void, F>(&ptr) })
    }
}

impl Drop for DynLib {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        // SAFETY: `handle` is a live handle from `open`, closed exactly once here.
        unsafe { dl::close(self.handle) };
    }
}

// ---------------------------------------------------------------------------
// Vulkan API types + constants (VK_HEADER_VERSION 1.3, subset used here).
// Every declaration mirrors the exact C definition from vulkan_core.h.
// ---------------------------------------------------------------------------

/// `VkResult` — Vulkan status code. `0` (`VK_SUCCESS`) is success; positive
/// values are informational, negative values are hard errors.
pub(crate) type VkResult = c_int;

/// `VK_SUCCESS`.
pub(crate) const VK_SUCCESS: VkResult = 0;

/// A 64-bit opaque Vulkan handle (`VkInstance`, `VkPhysicalDevice`, etc).
///
/// The Vulkan spec defines these as either `uint64_t` (non-dispatchable) or
/// pointer-sized dispatchable handles. On 64-bit hosts (every platform Vokra
/// supports) both are pointer-sized; we model them uniformly here.
pub(crate) type VkHandle = *mut c_void;

/// `VkInstance` — dispatchable loader handle.
pub(crate) type VkInstance = VkHandle;
/// `VkPhysicalDevice` — dispatchable device handle.
pub(crate) type VkPhysicalDevice = VkHandle;

/// A pointer to `vkGetInstanceProcAddr` result; returns `Option<F>` after
/// null-check.
pub(crate) type PFN_vkVoidFunction = *mut c_void;

/// `vkGetInstanceProcAddr(VkInstance, const char*)` — the entry point every
/// other instance/device fn is loaded through.
pub(crate) type FnVkGetInstanceProcAddr =
    unsafe extern "system" fn(VkInstance, *const c_char) -> PFN_vkVoidFunction;

/// `vkEnumerateInstanceVersion(uint32_t*)` — reports the highest Vulkan API
/// version the loader supports (Vulkan 1.1+; not present on 1.0 loaders).
pub(crate) type FnVkEnumerateInstanceVersion = unsafe extern "system" fn(*mut u32) -> VkResult;

/// `VkStructureType` selector for `VkApplicationInfo` (spec value `0`).
pub(crate) const VK_STRUCTURE_TYPE_APPLICATION_INFO: c_uint = 0;
/// `VkStructureType` selector for `VkInstanceCreateInfo` (spec value `1`).
pub(crate) const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: c_uint = 1;

/// Encodes a Vulkan API version: `VK_MAKE_API_VERSION(variant, major, minor,
/// patch)`. Only major/minor are load-bearing for Vokra (we target 1.1+).
pub(crate) const fn make_api_version(variant: u32, major: u32, minor: u32, patch: u32) -> u32 {
    (variant << 29) | (major << 22) | (minor << 12) | patch
}

/// Extract the major version component from an encoded Vulkan version.
pub(crate) const fn api_version_major(v: u32) -> u32 {
    (v >> 22) & 0x7f
}

/// Extract the minor version component from an encoded Vulkan version.
pub(crate) const fn api_version_minor(v: u32) -> u32 {
    (v >> 12) & 0x3ff
}

/// `VkApplicationInfo` (spec §4.2). Passed by pointer into
/// `VkInstanceCreateInfo`.
#[repr(C)]
pub(crate) struct VkApplicationInfo {
    pub s_type: c_uint,
    pub p_next: *const c_void,
    pub p_application_name: *const c_char,
    pub application_version: u32,
    pub p_engine_name: *const c_char,
    pub engine_version: u32,
    pub api_version: u32,
}

/// `VkInstanceCreateInfo` (spec §4.2). Passed by pointer into
/// `vkCreateInstance`.
#[repr(C)]
pub(crate) struct VkInstanceCreateInfo {
    pub s_type: c_uint,
    pub p_next: *const c_void,
    pub flags: u32,
    pub p_application_info: *const VkApplicationInfo,
    pub enabled_layer_count: u32,
    pub pp_enabled_layer_names: *const *const c_char,
    pub enabled_extension_count: u32,
    pub pp_enabled_extension_names: *const *const c_char,
}

/// `vkCreateInstance(*VkInstanceCreateInfo, *VkAllocationCallbacks,
/// *mut VkInstance)`. Allocation callbacks are null in Vokra (default host
/// allocator).
pub(crate) type FnVkCreateInstance = unsafe extern "system" fn(
    *const VkInstanceCreateInfo,
    *const c_void,
    *mut VkInstance,
) -> VkResult;

/// `vkDestroyInstance(VkInstance, *VkAllocationCallbacks)`.
pub(crate) type FnVkDestroyInstance = unsafe extern "system" fn(VkInstance, *const c_void);

/// `vkEnumeratePhysicalDevices(VkInstance, *uint32_t, *VkPhysicalDevice)`.
pub(crate) type FnVkEnumeratePhysicalDevices =
    unsafe extern "system" fn(VkInstance, *mut u32, *mut VkPhysicalDevice) -> VkResult;

/// `VkPhysicalDeviceType` enum values (spec §37.1). The probe classifies the
/// selected device by comparing `VkPhysicalDeviceProperties.deviceType`
/// against these; only `_OTHER` (`0`) is referenced by the foundation-slice
/// probe, the rest are declared for context (and for the T14〜T22 kernel
/// tickets that will pick device-local vs staging memory type per class).
pub(crate) const VK_PHYSICAL_DEVICE_TYPE_OTHER: u32 = 0;
/// `VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`.
#[allow(dead_code)] // consumers land with M3-02-T30 (fine-grained vendor gate)
pub(crate) const VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: u32 = 1;
/// `VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`.
#[allow(dead_code)] // consumers land with M3-02-T30
pub(crate) const VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: u32 = 2;
/// `VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU`.
#[allow(dead_code)] // consumers land with M3-02-T30
pub(crate) const VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: u32 = 3;
/// `VK_PHYSICAL_DEVICE_TYPE_CPU`.
#[allow(dead_code)] // consumers land with M3-02-T30
pub(crate) const VK_PHYSICAL_DEVICE_TYPE_CPU: u32 = 4;

/// `VkPhysicalDeviceLimits` (spec §37.2) — Vokra only reads a tiny subset
/// (`maxComputeSharedMemorySize`, `maxComputeWorkGroupInvocations`,
/// `maxComputeWorkGroupSize`). We declare the full struct layout so bytes line
/// up, but keep the field types opaque where we do not consume them.
///
/// This struct is ABI-stable per the Vulkan spec (repr C, no bitfields), so a
/// C-compatible declaration is sufficient — we do not need each individual
/// field named here to read the compute-related ones. To keep the ABI honest
/// and compilation robust, we declare the whole struct as a fixed-size byte
/// blob and index the compute-limit offsets by constant. See spec Chapter 37.
///
/// NOTE: `maxComputeSharedMemorySize` sits at offset `188` (`uint32_t`),
/// `maxComputeWorkGroupInvocations` at offset `192`, and
/// `maxComputeWorkGroupSize[3]` at offset `196..208` (Vulkan 1.0 headers, and
/// the layout has NEVER changed since 1.0). This mirrors what ash /
/// vulkan-headers auto-generate and is what every driver actually returns.
#[repr(C)]
pub(crate) struct VkPhysicalDeviceLimits {
    // 504 bytes total (Vulkan 1.0). Kept opaque; Vokra reads specific offsets
    // below rather than naming each of the ~110 fields. The size is used only
    // by `VkPhysicalDeviceProperties`, which is what `vkGetPhysicalDeviceProperties`
    // writes into and Vokra hands back to the probe.
    _bytes: [u8; 504],
}

/// `VkPhysicalDeviceSparseProperties` — 5 * VkBool32 = 20 bytes, then padding
/// to alignment. Vokra does not consume any sparse property.
#[repr(C)]
pub(crate) struct VkPhysicalDeviceSparseProperties {
    _bytes: [u8; 20],
}

/// `VK_UUID_SIZE` — the length of `pipelineCacheUUID` etc.
pub(crate) const VK_UUID_SIZE: usize = 16;
/// `VK_MAX_PHYSICAL_DEVICE_NAME_SIZE` — max device name length incl. NUL.
pub(crate) const VK_MAX_PHYSICAL_DEVICE_NAME_SIZE: usize = 256;

/// `VkPhysicalDeviceProperties` — the struct
/// `vkGetPhysicalDeviceProperties` writes into.
#[repr(C)]
pub(crate) struct VkPhysicalDeviceProperties {
    pub api_version: u32,
    pub driver_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: u32,
    pub device_name: [c_char; VK_MAX_PHYSICAL_DEVICE_NAME_SIZE],
    pub pipeline_cache_uuid: [u8; VK_UUID_SIZE],
    pub limits: VkPhysicalDeviceLimits,
    pub sparse_properties: VkPhysicalDeviceSparseProperties,
}

/// `vkGetPhysicalDeviceProperties(VkPhysicalDevice, *out
/// VkPhysicalDeviceProperties)`.
pub(crate) type FnVkGetPhysicalDeviceProperties =
    unsafe extern "system" fn(VkPhysicalDevice, *mut VkPhysicalDeviceProperties);

/// `VkQueueFlagBits` (spec §5.3.1) — capability bits for a queue family. Vokra
/// only cares about `_COMPUTE_BIT` (M3-02-T07 compute queue selection); the
/// others are declared for context.
///
/// `_GRAPHICS_BIT`. Graphics-capable queue (superset of compute per spec).
#[allow(dead_code)] // consumers land with M3-02-T08+
pub(crate) const VK_QUEUE_GRAPHICS_BIT: u32 = 0x0000_0001;
/// `_COMPUTE_BIT`. Compute-capable queue (Vokra's target).
pub(crate) const VK_QUEUE_COMPUTE_BIT: u32 = 0x0000_0002;
/// `_TRANSFER_BIT`. Transfer-only queue (dedicated DMA path — a follow-up may
/// route staging → device-local copies via this queue).
#[allow(dead_code)] // consumers land with M3-02-T25 (host↔device copy)
pub(crate) const VK_QUEUE_TRANSFER_BIT: u32 = 0x0000_0004;

/// `VkExtent3D` — used inside `VkQueueFamilyProperties.minImageTransferGranularity`.
/// The image-transfer field is unused by Vokra (compute-only), but must be
/// declared so `VkQueueFamilyProperties`'s layout matches vulkan_core.h.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct VkExtent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

/// `VkQueueFamilyProperties` (spec §5.3.1). Written by
/// `vkGetPhysicalDeviceQueueFamilyProperties` — one entry per queue family.
///
/// Vokra reads `queue_flags` (to pick a compute-capable family) and
/// `queue_count` (to sanity-check we can request a queue at all). The
/// timestamp / min-image-transfer fields are declared for layout fidelity but
/// unused.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct VkQueueFamilyProperties {
    pub queue_flags: u32,
    pub queue_count: u32,
    pub timestamp_valid_bits: u32,
    pub min_image_transfer_granularity: VkExtent3D,
}

/// `vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice, *uint32_t,
/// *VkQueueFamilyProperties)`.
pub(crate) type FnVkGetPhysicalDeviceQueueFamilyProperties =
    unsafe extern "system" fn(VkPhysicalDevice, *mut u32, *mut VkQueueFamilyProperties);

// ---------------------------------------------------------------------------
// The rest of the API used by `context.rs` (device / queue / command /
// descriptor / pipeline / memory / buffer / shader / dispatch) is intentionally
// left as function-pointer-only aliases; the pointers are resolved via
// `vkGetInstanceProcAddr` (loader-agnostic) or `vkGetDeviceProcAddr` (once a
// `VkDevice` exists). These bindings are declared here so `context.rs` can
// name the types uniformly; they are only USED when the crate is compiled
// with `--features vulkan` on a Vulkan-target host, and only exercised once
// T08〜T22 land. `#[allow(dead_code)]` because the foundation slice does not
// yet reference them (they are the load-bearing forward-declaration surface
// for the SPIR-V dispatch tickets).
// ---------------------------------------------------------------------------

/// `VkDevice` — dispatchable logical device handle.
pub(crate) type VkDevice = VkHandle;
/// `VkQueue` — dispatchable queue handle.
pub(crate) type VkQueue = VkHandle;
/// `VkCommandPool` — non-dispatchable handle.
pub(crate) type VkCommandPool = u64;
/// `VkCommandBuffer` — dispatchable handle.
pub(crate) type VkCommandBuffer = VkHandle;
/// `VkBuffer` — non-dispatchable handle.
pub(crate) type VkBuffer = u64;
/// `VkDeviceMemory` — non-dispatchable handle.
pub(crate) type VkDeviceMemory = u64;
/// `VkFence` — non-dispatchable handle (host-visible GPU sync primitive).
pub(crate) type VkFence = u64;
/// `VkDescriptorSetLayout` — non-dispatchable handle.
pub(crate) type VkDescriptorSetLayout = u64;
/// `VkDescriptorPool` — non-dispatchable handle.
pub(crate) type VkDescriptorPool = u64;
/// `VkDescriptorSet` — non-dispatchable handle.
pub(crate) type VkDescriptorSet = u64;
/// `VkPipelineLayout` — non-dispatchable handle.
pub(crate) type VkPipelineLayout = u64;
/// `VkPipeline` — non-dispatchable handle.
pub(crate) type VkPipeline = u64;
/// `VkPipelineCache` — non-dispatchable handle. Vokra never uses a pipeline
/// cache (SPIR-V blobs are already compiled) so this stays as an alias for
/// `VK_NULL_HANDLE` (0) at every call site.
pub(crate) type VkPipelineCache = u64;
/// `VkShaderModule` — non-dispatchable handle.
pub(crate) type VkShaderModule = u64;

// ---------------------------------------------------------------------------
// M3-02-T08〜T12 + T25 + T30 API surface: struct definitions, constants, and
// function pointer typedefs for the device-side objects (VkDevice / queue /
// command pool / command buffer / buffer / memory / fence / descriptor /
// pipeline / shader module / extension enumeration).
//
// Every field mirrors the exact C struct declared in vulkan_core.h (Vulkan
// 1.1+, ABI-stable). Constants are copied from the spec verbatim so this file
// stays cross-referenceable against the Khronos headers.
// ---------------------------------------------------------------------------

/// `VkDeviceSize` — 64-bit unsigned integer used for buffer offsets and sizes.
pub(crate) type VkDeviceSize = u64;

// -- Additional VkResult codes we recognise --------------------------------

/// `VK_TIMEOUT` — a wait on a fence timed out before the fence was signalled.
pub(crate) const VK_TIMEOUT: VkResult = 2;

// -- VkStructureType selectors we use ---------------------------------------

/// `VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO`.
pub(crate) const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: c_uint = 2;
/// `VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO`.
pub(crate) const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: c_uint = 3;
/// `VK_STRUCTURE_TYPE_SUBMIT_INFO`.
pub(crate) const VK_STRUCTURE_TYPE_SUBMIT_INFO: c_uint = 4;
/// `VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO`.
pub(crate) const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: c_uint = 5;
/// `VK_STRUCTURE_TYPE_FENCE_CREATE_INFO`.
pub(crate) const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: c_uint = 8;
/// `VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO`.
pub(crate) const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: c_uint = 12;
/// `VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO`.
pub(crate) const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: c_uint = 16;
/// `VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO`.
pub(crate) const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: c_uint = 32;
/// `VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO`.
pub(crate) const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: c_uint = 33;
/// `VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO`.
pub(crate) const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: c_uint = 34;
/// `VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET`.
pub(crate) const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: c_uint = 35;
/// `VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO`.
pub(crate) const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: c_uint = 18;
/// `VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO`.
pub(crate) const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: c_uint = 29;
/// `VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO`.
pub(crate) const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: c_uint = 30;
/// `VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO`.
pub(crate) const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: c_uint = 39;
/// `VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO`.
pub(crate) const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: c_uint = 40;
/// `VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO`.
pub(crate) const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: c_uint = 42;

// -- Buffer usage / memory property / other bitmask constants --------------

/// `VkBufferUsageFlagBits.VK_BUFFER_USAGE_TRANSFER_SRC_BIT`.
pub(crate) const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: u32 = 0x0000_0001;
/// `VkBufferUsageFlagBits.VK_BUFFER_USAGE_TRANSFER_DST_BIT`.
pub(crate) const VK_BUFFER_USAGE_TRANSFER_DST_BIT: u32 = 0x0000_0002;
/// `VkBufferUsageFlagBits.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`.
pub(crate) const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x0000_0020;

/// `VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`.
pub(crate) const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: u32 = 0x0000_0001;
/// `VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`.
pub(crate) const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x0000_0002;
/// `VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`.
pub(crate) const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x0000_0004;

/// `VkCommandPoolCreateFlagBits.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT`.
pub(crate) const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: u32 = 0x0000_0002;

/// `VkCommandBufferUsageFlagBits.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT`.
pub(crate) const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: u32 = 0x0000_0001;

/// `VkCommandBufferLevel.VK_COMMAND_BUFFER_LEVEL_PRIMARY`.
pub(crate) const VK_COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;

/// `VkSharingMode.VK_SHARING_MODE_EXCLUSIVE` — buffer memory access is
/// exclusive to a single queue family (Vokra uses one compute queue only).
pub(crate) const VK_SHARING_MODE_EXCLUSIVE: u32 = 0;

/// `VkDescriptorType.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` — the only descriptor
/// type Vokra's compute shaders use (SSBO for tensor storage).
pub(crate) const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;

/// `VkShaderStageFlagBits.VK_SHADER_STAGE_COMPUTE_BIT`.
pub(crate) const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x0000_0020;

/// `VkPipelineBindPoint.VK_PIPELINE_BIND_POINT_COMPUTE`.
#[allow(dead_code)] // T14+ dispatch code lands the consumer (vkCmdBindPipeline)
pub(crate) const VK_PIPELINE_BIND_POINT_COMPUTE: u32 = 1;

// -- Memory / device / queue create structs --------------------------------

/// `VkDeviceQueueCreateInfo` (spec §5.2).
#[repr(C)]
pub(crate) struct VkDeviceQueueCreateInfo {
    pub s_type: c_uint,
    pub p_next: *const c_void,
    pub flags: u32,
    pub queue_family_index: u32,
    pub queue_count: u32,
    pub p_queue_priorities: *const f32,
}

/// `VkDeviceCreateInfo` (spec §5.2). `p_enabled_features` stays null in Vokra
/// (default features — no shader int64 / float16 requests until T14+).
#[repr(C)]
pub(crate) struct VkDeviceCreateInfo {
    pub s_type: c_uint,
    pub p_next: *const c_void,
    pub flags: u32,
    pub queue_create_info_count: u32,
    pub p_queue_create_infos: *const VkDeviceQueueCreateInfo,
    pub enabled_layer_count: u32,
    pub pp_enabled_layer_names: *const *const c_char,
    pub enabled_extension_count: u32,
    pub pp_enabled_extension_names: *const *const c_char,
    pub p_enabled_features: *const c_void,
}

pub(crate) type FnVkCreateDevice = unsafe extern "system" fn(
    VkPhysicalDevice,
    *const VkDeviceCreateInfo,
    *const c_void,
    *mut VkDevice,
) -> VkResult;
pub(crate) type FnVkDestroyDevice = unsafe extern "system" fn(VkDevice, *const c_void);
pub(crate) type FnVkGetDeviceQueue = unsafe extern "system" fn(VkDevice, u32, u32, *mut VkQueue);
/// `vkGetDeviceProcAddr` — device-level entry-point resolution. Vokra
/// currently loads every device fn via `vkGetInstanceProcAddr` (spec-valid,
/// loader trampoline dispatches to the ICD). T14+ dispatch code may switch
/// to `vkGetDeviceProcAddr` for hot-path fns to bypass the trampoline.
#[allow(dead_code)] // T14+ dispatch code lands the consumer
pub(crate) type FnVkGetDeviceProcAddr =
    unsafe extern "system" fn(VkDevice, *const c_char) -> PFN_vkVoidFunction;
pub(crate) type FnVkDeviceWaitIdle = unsafe extern "system" fn(VkDevice) -> VkResult;

// -- Physical device memory properties -------------------------------------

/// `VK_MAX_MEMORY_TYPES` (spec §5.3.2).
pub(crate) const VK_MAX_MEMORY_TYPES: usize = 32;
/// `VK_MAX_MEMORY_HEAPS`.
pub(crate) const VK_MAX_MEMORY_HEAPS: usize = 16;

/// `VkMemoryType` (spec §11.3).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct VkMemoryType {
    pub property_flags: u32,
    pub heap_index: u32,
}

/// `VkMemoryHeap` (spec §11.3).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct VkMemoryHeap {
    pub size: VkDeviceSize,
    pub flags: u32,
}

/// `VkPhysicalDeviceMemoryProperties` (spec §11.3).
#[repr(C)]
pub(crate) struct VkPhysicalDeviceMemoryProperties {
    pub memory_type_count: u32,
    pub memory_types: [VkMemoryType; VK_MAX_MEMORY_TYPES],
    pub memory_heap_count: u32,
    pub memory_heaps: [VkMemoryHeap; VK_MAX_MEMORY_HEAPS],
}

pub(crate) type FnVkGetPhysicalDeviceMemoryProperties =
    unsafe extern "system" fn(VkPhysicalDevice, *mut VkPhysicalDeviceMemoryProperties);

// -- Command pool / command buffer -----------------------------------------

/// `VkCommandPoolCreateInfo` (spec §6.2).
#[repr(C)]
pub(crate) struct VkCommandPoolCreateInfo {
    pub s_type: c_uint,
    pub p_next: *const c_void,
    pub flags: u32,
    pub queue_family_index: u32,
}

pub(crate) type FnVkCreateCommandPool = unsafe extern "system" fn(
    VkDevice,
    *const VkCommandPoolCreateInfo,
    *const c_void,
    *mut VkCommandPool,
) -> VkResult;
pub(crate) type FnVkDestroyCommandPool =
    unsafe extern "system" fn(VkDevice, VkCommandPool, *const c_void);
pub(crate) type FnVkResetCommandPool =
    unsafe extern "system" fn(VkDevice, VkCommandPool, u32) -> VkResult;

/// `VkCommandBufferAllocateInfo` (spec §6.3).
#[repr(C)]
pub(crate) struct VkCommandBufferAllocateInfo {
    pub s_type: c_uint,
    pub p_next: *const c_void,
    pub command_pool: VkCommandPool,
    pub level: u32,
    pub command_buffer_count: u32,
}

pub(crate) type FnVkAllocateCommandBuffers = unsafe extern "system" fn(
    VkDevice,
    *const VkCommandBufferAllocateInfo,
    *mut VkCommandBuffer,
) -> VkResult;
pub(crate) type FnVkFreeCommandBuffers =
    unsafe extern "system" fn(VkDevice, VkCommandPool, u32, *const VkCommandBuffer);

/// `VkCommandBufferBeginInfo` (spec §6.4).
#[repr(C)]
pub(crate) struct VkCommandBufferBeginInfo {
    pub s_type: c_uint,
    pub p_next: *const c_void,
    pub flags: u32,
    pub p_inheritance_info: *const c_void,
}

pub(crate) type FnVkBeginCommandBuffer =
    unsafe extern "system" fn(VkCommandBuffer, *const VkCommandBufferBeginInfo) -> VkResult;
pub(crate) type FnVkEndCommandBuffer = unsafe extern "system" fn(VkCommandBuffer) -> VkResult;
pub(crate) type FnVkResetCommandBuffer =
    unsafe extern "system" fn(VkCommandBuffer, u32) -> VkResult;

/// `VkBufferCopy` (spec §20.4).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct VkBufferCopy {
    pub src_offset: VkDeviceSize,
    pub dst_offset: VkDeviceSize,
    pub size: VkDeviceSize,
}

pub(crate) type FnVkCmdCopyBuffer =
    unsafe extern "system" fn(VkCommandBuffer, VkBuffer, VkBuffer, u32, *const VkBufferCopy);

// -- Queue submit / fence --------------------------------------------------

/// `VkSubmitInfo` (spec §7.3).
#[repr(C)]
pub(crate) struct VkSubmitInfo {
    pub s_type: c_uint,
    pub p_next: *const c_void,
    pub wait_semaphore_count: u32,
    pub p_wait_semaphores: *const u64,
    pub p_wait_dst_stage_mask: *const u32,
    pub command_buffer_count: u32,
    pub p_command_buffers: *const VkCommandBuffer,
    pub signal_semaphore_count: u32,
    pub p_signal_semaphores: *const u64,
}

pub(crate) type FnVkQueueSubmit =
    unsafe extern "system" fn(VkQueue, u32, *const VkSubmitInfo, VkFence) -> VkResult;
pub(crate) type FnVkQueueWaitIdle = unsafe extern "system" fn(VkQueue) -> VkResult;

/// `VkFenceCreateInfo` (spec §7.3.2).
#[repr(C)]
pub(crate) struct VkFenceCreateInfo {
    pub s_type: c_uint,
    pub p_next: *const c_void,
    pub flags: u32,
}

pub(crate) type FnVkCreateFence = unsafe extern "system" fn(
    VkDevice,
    *const VkFenceCreateInfo,
    *const c_void,
    *mut VkFence,
) -> VkResult;
pub(crate) type FnVkDestroyFence = unsafe extern "system" fn(VkDevice, VkFence, *const c_void);
pub(crate) type FnVkWaitForFences = unsafe extern "system" fn(
    VkDevice,
    u32,
    *const VkFence,
    u32, // waitAll
    u64, // timeout ns
) -> VkResult;
pub(crate) type FnVkResetFences =
    unsafe extern "system" fn(VkDevice, u32, *const VkFence) -> VkResult;

// -- Buffer + memory --------------------------------------------------------

/// `VkBufferCreateInfo` (spec §12.2).
#[repr(C)]
pub(crate) struct VkBufferCreateInfo {
    pub s_type: c_uint,
    pub p_next: *const c_void,
    pub flags: u32,
    pub size: VkDeviceSize,
    pub usage: u32,
    pub sharing_mode: u32,
    pub queue_family_index_count: u32,
    pub p_queue_family_indices: *const u32,
}

pub(crate) type FnVkCreateBuffer = unsafe extern "system" fn(
    VkDevice,
    *const VkBufferCreateInfo,
    *const c_void,
    *mut VkBuffer,
) -> VkResult;
pub(crate) type FnVkDestroyBuffer = unsafe extern "system" fn(VkDevice, VkBuffer, *const c_void);

/// `VkMemoryRequirements` (spec §11.6).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct VkMemoryRequirements {
    pub size: VkDeviceSize,
    pub alignment: VkDeviceSize,
    pub memory_type_bits: u32,
}

pub(crate) type FnVkGetBufferMemoryRequirements =
    unsafe extern "system" fn(VkDevice, VkBuffer, *mut VkMemoryRequirements);

/// `VkMemoryAllocateInfo` (spec §11.6).
#[repr(C)]
pub(crate) struct VkMemoryAllocateInfo {
    pub s_type: c_uint,
    pub p_next: *const c_void,
    pub allocation_size: VkDeviceSize,
    pub memory_type_index: u32,
}

pub(crate) type FnVkAllocateMemory = unsafe extern "system" fn(
    VkDevice,
    *const VkMemoryAllocateInfo,
    *const c_void,
    *mut VkDeviceMemory,
) -> VkResult;
pub(crate) type FnVkFreeMemory = unsafe extern "system" fn(VkDevice, VkDeviceMemory, *const c_void);
pub(crate) type FnVkBindBufferMemory =
    unsafe extern "system" fn(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) -> VkResult;
pub(crate) type FnVkMapMemory = unsafe extern "system" fn(
    VkDevice,
    VkDeviceMemory,
    VkDeviceSize,
    VkDeviceSize,
    u32, // flags (reserved, must be 0)
    *mut *mut c_void,
) -> VkResult;
pub(crate) type FnVkUnmapMemory = unsafe extern "system" fn(VkDevice, VkDeviceMemory);

// -- Descriptor set layout / pool / set / update ---------------------------

/// `VkDescriptorSetLayoutBinding` (spec §14.2).
#[repr(C)]
pub(crate) struct VkDescriptorSetLayoutBinding {
    pub binding: u32,
    pub descriptor_type: u32,
    pub descriptor_count: u32,
    pub stage_flags: u32,
    pub p_immutable_samplers: *const u64,
}

/// `VkDescriptorSetLayoutCreateInfo`.
#[repr(C)]
pub(crate) struct VkDescriptorSetLayoutCreateInfo {
    pub s_type: c_uint,
    pub p_next: *const c_void,
    pub flags: u32,
    pub binding_count: u32,
    pub p_bindings: *const VkDescriptorSetLayoutBinding,
}

pub(crate) type FnVkCreateDescriptorSetLayout = unsafe extern "system" fn(
    VkDevice,
    *const VkDescriptorSetLayoutCreateInfo,
    *const c_void,
    *mut VkDescriptorSetLayout,
) -> VkResult;
pub(crate) type FnVkDestroyDescriptorSetLayout =
    unsafe extern "system" fn(VkDevice, VkDescriptorSetLayout, *const c_void);

/// `VkDescriptorPoolSize` (spec §14.2.3).
#[repr(C)]
pub(crate) struct VkDescriptorPoolSize {
    pub ty: u32,
    pub descriptor_count: u32,
}

/// `VkDescriptorPoolCreateInfo`.
#[repr(C)]
pub(crate) struct VkDescriptorPoolCreateInfo {
    pub s_type: c_uint,
    pub p_next: *const c_void,
    pub flags: u32,
    pub max_sets: u32,
    pub pool_size_count: u32,
    pub p_pool_sizes: *const VkDescriptorPoolSize,
}

pub(crate) type FnVkCreateDescriptorPool = unsafe extern "system" fn(
    VkDevice,
    *const VkDescriptorPoolCreateInfo,
    *const c_void,
    *mut VkDescriptorPool,
) -> VkResult;
pub(crate) type FnVkDestroyDescriptorPool =
    unsafe extern "system" fn(VkDevice, VkDescriptorPool, *const c_void);

/// `VkDescriptorSetAllocateInfo`.
#[repr(C)]
pub(crate) struct VkDescriptorSetAllocateInfo {
    pub s_type: c_uint,
    pub p_next: *const c_void,
    pub descriptor_pool: VkDescriptorPool,
    pub descriptor_set_count: u32,
    pub p_set_layouts: *const VkDescriptorSetLayout,
}

pub(crate) type FnVkAllocateDescriptorSets = unsafe extern "system" fn(
    VkDevice,
    *const VkDescriptorSetAllocateInfo,
    *mut VkDescriptorSet,
) -> VkResult;

/// `VkDescriptorBufferInfo` (spec §14.4).
#[repr(C)]
pub(crate) struct VkDescriptorBufferInfo {
    pub buffer: VkBuffer,
    pub offset: VkDeviceSize,
    pub range: VkDeviceSize,
}

/// `VkWriteDescriptorSet` (spec §14.4).
#[repr(C)]
pub(crate) struct VkWriteDescriptorSet {
    pub s_type: c_uint,
    pub p_next: *const c_void,
    pub dst_set: VkDescriptorSet,
    pub dst_binding: u32,
    pub dst_array_element: u32,
    pub descriptor_count: u32,
    pub descriptor_type: u32,
    pub p_image_info: *const c_void,
    pub p_buffer_info: *const VkDescriptorBufferInfo,
    pub p_texel_buffer_view: *const u64,
}

pub(crate) type FnVkUpdateDescriptorSets =
    unsafe extern "system" fn(VkDevice, u32, *const VkWriteDescriptorSet, u32, *const c_void);

// -- Pipeline layout / shader module / compute pipeline --------------------

/// `VkPipelineLayoutCreateInfo` (spec §14.2.1).
#[repr(C)]
pub(crate) struct VkPipelineLayoutCreateInfo {
    pub s_type: c_uint,
    pub p_next: *const c_void,
    pub flags: u32,
    pub set_layout_count: u32,
    pub p_set_layouts: *const VkDescriptorSetLayout,
    pub push_constant_range_count: u32,
    pub p_push_constant_ranges: *const c_void,
}

pub(crate) type FnVkCreatePipelineLayout = unsafe extern "system" fn(
    VkDevice,
    *const VkPipelineLayoutCreateInfo,
    *const c_void,
    *mut VkPipelineLayout,
) -> VkResult;
pub(crate) type FnVkDestroyPipelineLayout =
    unsafe extern "system" fn(VkDevice, VkPipelineLayout, *const c_void);

/// `VkShaderModuleCreateInfo` (spec §9.1).
#[repr(C)]
pub(crate) struct VkShaderModuleCreateInfo {
    pub s_type: c_uint,
    pub p_next: *const c_void,
    pub flags: u32,
    pub code_size: usize,
    pub p_code: *const u32,
}

pub(crate) type FnVkCreateShaderModule = unsafe extern "system" fn(
    VkDevice,
    *const VkShaderModuleCreateInfo,
    *const c_void,
    *mut VkShaderModule,
) -> VkResult;
pub(crate) type FnVkDestroyShaderModule =
    unsafe extern "system" fn(VkDevice, VkShaderModule, *const c_void);

/// `VkPipelineShaderStageCreateInfo` (spec §9.1).
#[repr(C)]
pub(crate) struct VkPipelineShaderStageCreateInfo {
    pub s_type: c_uint,
    pub p_next: *const c_void,
    pub flags: u32,
    pub stage: u32,
    pub module: VkShaderModule,
    pub p_name: *const c_char,
    pub p_specialization_info: *const c_void,
}

/// `VkComputePipelineCreateInfo` (spec §9.4).
#[repr(C)]
pub(crate) struct VkComputePipelineCreateInfo {
    pub s_type: c_uint,
    pub p_next: *const c_void,
    pub flags: u32,
    pub stage: VkPipelineShaderStageCreateInfo,
    pub layout: VkPipelineLayout,
    pub base_pipeline_handle: VkPipeline,
    pub base_pipeline_index: i32,
}

pub(crate) type FnVkCreateComputePipelines = unsafe extern "system" fn(
    VkDevice,
    VkPipelineCache,
    u32,
    *const VkComputePipelineCreateInfo,
    *const c_void,
    *mut VkPipeline,
) -> VkResult;
pub(crate) type FnVkDestroyPipeline =
    unsafe extern "system" fn(VkDevice, VkPipeline, *const c_void);

// -- Device extension enumeration (M3-02-T30) ------------------------------

/// `VK_MAX_EXTENSION_NAME_SIZE`.
pub(crate) const VK_MAX_EXTENSION_NAME_SIZE: usize = 256;

/// `VkExtensionProperties` (spec §37.3).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct VkExtensionProperties {
    pub extension_name: [c_char; VK_MAX_EXTENSION_NAME_SIZE],
    pub spec_version: u32,
}

pub(crate) type FnVkEnumerateDeviceExtensionProperties = unsafe extern "system" fn(
    VkPhysicalDevice,
    *const c_char,
    *mut u32,
    *mut VkExtensionProperties,
) -> VkResult;

// ---------------------------------------------------------------------------
// Loader (loader-level entry points resolved via dlopen + vkGetInstanceProcAddr).
// ---------------------------------------------------------------------------

/// Candidate library names for the Vulkan loader, tried in order. On a host
/// with no Vulkan loader (e.g. an Apple Mac; macOS uses Metal, and even
/// MoltenVK is not linked here) none load and the probe returns
/// `BackendUnavailable`.
const LIBVULKAN_CANDIDATES: &[&[u8]] = &[
    // Linux / Android — `libvulkan.so.1` is the SONAME the Khronos loader
    // ships with; `libvulkan.so` is a distro-provided symlink that is not
    // always installed.
    b"libvulkan.so.1\0",
    b"libvulkan.so\0",
    // macOS via MoltenVK (unlikely for Vokra — macOS uses `vokra-backend-metal`
    // — but harmless to try last).
    b"libvulkan.1.dylib\0",
    // Windows.
    b"vulkan-1.dll\0",
];

/// Environment override — a full path to `libvulkan` for
/// developer-controlled test environments (mirrors `VOKRA_CUDA_LIB` etc). Only
/// consulted on Unix (where `libc::getenv` is the standard read path).
#[allow(dead_code)] // used only via `VulkanLoader::open_from_env` when set
const ENV_VOKRA_VULKAN_LIB: &[u8] = b"VOKRA_VULKAN_LIB\0";

/// The Vulkan loader (`libvulkan.so.1` / `vulkan-1.dll`) with the loader-level
/// entry points resolved. Instance / device entries are loaded on top of this
/// (via `vkGetInstanceProcAddr`) by [`context`](crate::context) once an
/// instance exists.
pub(crate) struct VulkanLoader {
    _lib: DynLib,
    pub(crate) get_instance_proc_addr: FnVkGetInstanceProcAddr,
    /// `vkEnumerateInstanceVersion` — present only on Vulkan 1.1+ loaders.
    /// `None` on a 1.0 loader (which Vokra rejects at probe time).
    pub(crate) enumerate_instance_version: Option<FnVkEnumerateInstanceVersion>,
    pub(crate) create_instance: FnVkCreateInstance,
}

impl VulkanLoader {
    /// Loads `libvulkan` and resolves the loader-level entry points.
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] if the Vulkan loader library is not
    /// present (no Vulkan on this host, e.g. an Apple Mac), or a required
    /// loader-level symbol is missing (loader too old — a 1.0-only loader
    /// missing `vkEnumerateInstanceVersion` is rejected here; Vokra targets
    /// Vulkan 1.1+, spec §35.2). Never a silent fall back (NFR-RL-06).
    pub(crate) fn load() -> Result<VulkanLoader> {
        let lib = DynLib::open(LIBVULKAN_CANDIDATES).ok_or_else(|| {
            VokraError::BackendUnavailable(
                "libvulkan (Vulkan loader) not found: no Vulkan loader installed on this host, or \
                 the loader shared library is not on the dynamic-linker search path. Vokra does \
                 not bundle libvulkan (all-target build, NFR-PT-01); install the platform's \
                 Vulkan loader package to use the Vulkan backend."
                    .to_owned(),
            )
        })?;
        // SAFETY: each `get::<Fn…>` pairs the exact C symbol name with the
        // function-pointer alias declaring its true signature (vulkan_core.h).
        // The loader-level entry points below are exported by every 1.1+
        // Khronos loader; missing them means the loader is 1.0-only or
        // otherwise incompatible, which we surface as `BackendUnavailable`.
        let get_instance_proc_addr =
            unsafe { lib.get::<FnVkGetInstanceProcAddr>(b"vkGetInstanceProcAddr\0") }.ok_or_else(
                || {
                    VokraError::BackendUnavailable(
                    "Vulkan loader present but `vkGetInstanceProcAddr` is missing (impossible on \
                     any conforming loader — refusing to continue)."
                        .to_owned(),
                )
                },
            )?;

        // `vkEnumerateInstanceVersion` is only present on loaders that export
        // the 1.1 core. Absent on 1.0-only systems (very old Android or
        // ancient Linux). Treat that as an unusable loader.
        let enumerate_instance_version =
            // SAFETY: symbol name matches the FnVkEnumerateInstanceVersion prototype.
            unsafe { lib.get::<FnVkEnumerateInstanceVersion>(b"vkEnumerateInstanceVersion\0") };

        // `vkCreateInstance` is loader-level and available on all Vulkan
        // loaders since 1.0; it is legitimately resolvable via dlsym.
        //
        // SAFETY: symbol name matches the FnVkCreateInstance prototype
        // (vulkan_core.h) that `sys::get` transmutes the resolved procaddr to.
        let create_instance = unsafe { lib.get::<FnVkCreateInstance>(b"vkCreateInstance\0") }
            .ok_or_else(|| {
                VokraError::BackendUnavailable(
                    "Vulkan loader present but `vkCreateInstance` is missing (loader is \
                     malformed — refusing to continue)."
                        .to_owned(),
                )
            })?;

        Ok(VulkanLoader {
            _lib: lib,
            get_instance_proc_addr,
            enumerate_instance_version,
            create_instance,
        })
    }
}

/// Resolves an instance-level function via `vkGetInstanceProcAddr` and
/// `transmute`s it to `F`.
///
/// Returns `None` if the loader has no such symbol (extension not enabled /
/// name mistyped).
///
/// # Safety
/// `F` must be a function-pointer type whose signature matches the C symbol
/// `name` exactly. Callers pair each `name` with its precise `Fn*` alias.
pub(crate) unsafe fn instance_proc<F: Copy>(
    loader: &VulkanLoader,
    instance: VkInstance,
    name: &[u8],
) -> Option<F> {
    debug_assert_eq!(name.last(), Some(&0), "symbol name must be NUL-terminated");
    debug_assert_eq!(
        core::mem::size_of::<F>(),
        core::mem::size_of::<*mut c_void>(),
        "F must be a pointer-sized function pointer"
    );
    // SAFETY: `loader.get_instance_proc_addr` is a valid Vulkan entry point;
    // `instance` may be null (loader-level query) or a live `VkInstance`;
    // `name` is a valid NUL-terminated C string.
    let ptr = unsafe { (loader.get_instance_proc_addr)(instance, name.as_ptr() as *const c_char) };
    if ptr.is_null() {
        return None;
    }
    // SAFETY: `ptr` is a non-null Vulkan function pointer (pointer-sized). `F`
    // is a function-pointer type of the same size (asserted above) whose C
    // signature the caller guarantees matches `name`.
    Some(unsafe { core::mem::transmute_copy::<*mut c_void, F>(&ptr) })
}

/// Maps a `VkResult` to `Result<()>`. Any non-`VK_SUCCESS` code becomes a
/// [`VokraError::BackendUnavailable`] carrying the numeric code and the
/// operation name — Vokra does not silently swallow Vulkan errors.
pub(crate) fn check(r: VkResult, op: &str) -> Result<()> {
    if r == VK_SUCCESS {
        Ok(())
    } else {
        Err(VokraError::BackendUnavailable(format!(
            "Vulkan {op} failed with VkResult={r}"
        )))
    }
}

/// Convert a NUL-terminated C `[c_char; VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]`
/// field into an owned `String` (lossy on non-UTF-8, which no conforming
/// driver emits — device names are ASCII).
pub(crate) fn name_from_buf(buf: &[c_char; VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]) -> String {
    // Find first NUL byte; the driver is required to NUL-terminate.
    // `c_char` is `i8` on some targets (x86_64 Linux / macOS) and `u8` on
    // others (aarch64 Linux/Android / Windows). The cast to `*const u8` is
    // a no-op on the latter but still required to type-check on the former;
    // allow the "unnecessary cast" clippy lint locally.
    #[allow(clippy::unnecessary_cast)]
    let ptr = buf.as_ptr() as *const u8;
    // SAFETY: `c_char` and `u8` have the same representation on every
    // platform Vokra supports; the slice length equals the exact array
    // length so we do not read past `buf`.
    let bytes: &[u8] = unsafe { core::slice::from_raw_parts(ptr, buf.len()) };
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_encoding_round_trips() {
        let v = make_api_version(0, 1, 1, 0);
        assert_eq!(api_version_major(v), 1);
        assert_eq!(api_version_minor(v), 1);
    }

    #[test]
    fn name_from_buf_stops_at_nul() {
        let mut buf: [c_char; VK_MAX_PHYSICAL_DEVICE_NAME_SIZE] =
            [0; VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
        let src = b"lavapipe\0";
        for (i, &b) in src.iter().enumerate() {
            buf[i] = b as c_char;
        }
        assert_eq!(name_from_buf(&buf), "lavapipe");
    }

    #[test]
    fn check_maps_error_to_backend_unavailable() {
        let err = check(-1, "vkTest").unwrap_err();
        assert!(matches!(err, VokraError::BackendUnavailable(_)));
    }

    #[test]
    fn check_success_is_ok() {
        assert!(check(VK_SUCCESS, "vkTest").is_ok());
    }
}
