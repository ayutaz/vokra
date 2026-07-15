//! Vulkan runtime-object stack (M3-02-T06〜T12 + T25 + T30 extension walk).
//!
//! This module owns every Vulkan handle Vokra creates on the host side of the
//! runtime — the RAII-managed hierarchy is:
//!
//! ```text
//! VulkanInstance                    (T06 — vkCreateInstance / vkDestroyInstance)
//!   └── VulkanDevice                (T08 — vkCreateDevice + compute VkQueue)
//!         ├── VulkanCommandPool     (T09 — VkCommandPool + primary VkCommandBuffer)
//!         ├── VulkanBuffer          (T12 — VkBuffer + backing VkDeviceMemory)
//!         ├── VulkanFence           (T25 helper — host-visible submit sync)
//!         ├── VulkanDescriptorSetLayout / Pool / Set  (T10)
//!         └── VulkanShaderModule / VulkanPipelineLayout /
//!             VulkanComputePipeline (T11 — kernel-dispatch shell)
//! ```
//!
//! Every Vulkan API call is a raw FFI transmute — no `ash` / `vulkano` /
//! `erupt` / `gpu-alloc` binding crate (M3-02 red-line, NFR-DS-02). Every
//! `unsafe` block carries a `// SAFETY:` comment enumerating the invariant
//! it relies on; every constructor returns `Result`, and every wrapper's
//! `Drop` frees the underlying handle in the correct order.
//!
//! # Drop order
//!
//! Rust drops struct fields in declaration order. `VulkanDevice` owns its
//! parent `VulkanInstance` and destroys the `VkDevice` in its own
//! `Drop::drop` (before field drops run), so the loader handle stays alive
//! until after `vkDestroyDevice` has returned. Children of `VulkanDevice`
//! (command pool / buffer / descriptor set / pipeline / …) borrow
//! `&VulkanDevice`, so the borrow checker enforces that the device outlives
//! them — a resource-leak-preventing invariant Rust gives us "for free".
//!
//! # Host↔device round-trip (T25)
//!
//! [`VulkanDevice::upload_bytes`] and [`VulkanDevice::download_bytes`] copy
//! between a host-visible staging buffer and a device-local buffer via a
//! transient primary command buffer + `vkCmdCopyBuffer` + fence submit. The
//! transient command buffer is allocated from a dedicated one-shot pool
//! (`VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT`) and reset between
//! calls; the fence is the sync point ("submit → wait_for_fence" — the
//! simplest correct pattern, ADR-driven: no timeline semaphores yet).

use core::ffi::{c_char, c_void};
use core::mem::MaybeUninit;
use core::ptr;

use vokra_core::{Result, VokraError};

use crate::sys;

// ---------------------------------------------------------------------------
// VulkanInstance — T06 + additional instance-level fn pointers for T08 / T12 /
// T30.
// ---------------------------------------------------------------------------

/// A Vulkan `VkInstance` scoped to the [`VulkanLoader`](crate::sys::VulkanLoader)
/// that created it. Destroyed on drop via `vkDestroyInstance`.
///
/// Vokra creates a *minimal* instance: no validation layers, no
/// window-system extensions — just the loader-level plumbing needed to
/// enumerate physical devices and later create a compute-only logical device
/// (M3-02-T08).
pub(crate) struct VulkanInstance {
    // The loader must outlive every handle derived from it. `Send`/`Sync` are
    // intentionally not implemented — Vulkan handles are used from the thread
    // that created them (same posture as MetalContext / CudaContext).
    loader: sys::VulkanLoader,
    instance: sys::VkInstance,
    // Instance-level entry points resolved via vkGetInstanceProcAddr. The
    // foundation-slice probe needed only enumerate + properties + queue-family
    // + destroy; T08〜T12 add memory-properties / device create / extension
    // enumeration.
    enumerate_physical_devices: sys::FnVkEnumeratePhysicalDevices,
    get_physical_device_properties: sys::FnVkGetPhysicalDeviceProperties,
    get_physical_device_queue_family_properties: sys::FnVkGetPhysicalDeviceQueueFamilyProperties,
    get_physical_device_memory_properties: sys::FnVkGetPhysicalDeviceMemoryProperties,
    create_device: sys::FnVkCreateDevice,
    enumerate_device_extension_properties: sys::FnVkEnumerateDeviceExtensionProperties,
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

        // Resolve the instance-level entries Vokra needs from here on. Every
        // symbol lookup pairs the C name with the exact FnVk… alias declared
        // in `sys` (which mirrors vulkan_core.h). A missing symbol is an
        // unusable driver → BackendUnavailable, never a silent fallback.
        //
        // SAFETY (applies to every `unsafe { sys::instance_proc(...) }` block
        // below): each call pairs a NUL-terminated ASCII C symbol name with
        // the exact FnVk… type alias declared in `sys` (which mirrors
        // vulkan_core.h). The loader / instance handles are live for the
        // duration of the call.
        let enumerate_physical_devices: sys::FnVkEnumeratePhysicalDevices =
            // SAFETY: see block-level comment above (instance_proc symbol resolution).
            unsafe { sys::instance_proc(&loader, instance, b"vkEnumeratePhysicalDevices\0") }
                .ok_or_else(|| missing_instance_symbol("vkEnumeratePhysicalDevices"))?;
        let get_physical_device_properties: sys::FnVkGetPhysicalDeviceProperties =
            // SAFETY: see block-level comment above.
            unsafe { sys::instance_proc(&loader, instance, b"vkGetPhysicalDeviceProperties\0") }
                .ok_or_else(|| missing_instance_symbol("vkGetPhysicalDeviceProperties"))?;
        let get_physical_device_queue_family_properties: sys::FnVkGetPhysicalDeviceQueueFamilyProperties =
            // SAFETY: see block-level comment above.
            unsafe {
                sys::instance_proc(&loader, instance, b"vkGetPhysicalDeviceQueueFamilyProperties\0")
            }
            .ok_or_else(|| missing_instance_symbol("vkGetPhysicalDeviceQueueFamilyProperties"))?;
        let get_physical_device_memory_properties: sys::FnVkGetPhysicalDeviceMemoryProperties =
            // SAFETY: see block-level comment above.
            unsafe {
                sys::instance_proc(&loader, instance, b"vkGetPhysicalDeviceMemoryProperties\0")
            }
            .ok_or_else(|| missing_instance_symbol("vkGetPhysicalDeviceMemoryProperties"))?;
        let create_device: sys::FnVkCreateDevice =
            // SAFETY: see block-level comment above.
            unsafe { sys::instance_proc(&loader, instance, b"vkCreateDevice\0") }
                .ok_or_else(|| missing_instance_symbol("vkCreateDevice"))?;
        let enumerate_device_extension_properties: sys::FnVkEnumerateDeviceExtensionProperties =
            // SAFETY: see block-level comment above.
            unsafe {
                sys::instance_proc(&loader, instance, b"vkEnumerateDeviceExtensionProperties\0")
            }
            .ok_or_else(|| missing_instance_symbol("vkEnumerateDeviceExtensionProperties"))?;
        let destroy_instance: sys::FnVkDestroyInstance =
            // SAFETY: see block-level comment above.
            unsafe { sys::instance_proc(&loader, instance, b"vkDestroyInstance\0") }
                .ok_or_else(|| missing_instance_symbol("vkDestroyInstance"))?;

        Ok(VulkanInstance {
            loader,
            instance,
            enumerate_physical_devices,
            get_physical_device_properties,
            get_physical_device_queue_family_properties,
            get_physical_device_memory_properties,
            create_device,
            enumerate_device_extension_properties,
            destroy_instance,
        })
    }

    /// Enumerate `VkPhysicalDevice` handles.
    pub(crate) fn enumerate_physical_devices(&self) -> Result<Vec<sys::VkPhysicalDevice>> {
        let mut count: u32 = 0;
        // SAFETY: `enumerate_physical_devices` is the resolved instance entry;
        // `count` is a valid writable u32.
        let r = unsafe {
            (self.enumerate_physical_devices)(self.instance, &mut count, ptr::null_mut())
        };
        sys::check(r, "vkEnumeratePhysicalDevices (count)")?;
        if count == 0 {
            return Ok(Vec::new());
        }
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
        let mut props: MaybeUninit<sys::VkPhysicalDeviceProperties> = MaybeUninit::uninit();
        // SAFETY: `get_physical_device_properties` is the resolved instance
        // entry; `device` is a valid handle; the out-pointer is a writable
        // struct slot.
        unsafe { (self.get_physical_device_properties)(device, props.as_mut_ptr()) };
        // SAFETY: the driver call above fully initialises the struct.
        unsafe { props.assume_init() }
    }

    /// Read `VkPhysicalDeviceMemoryProperties` for a physical device (T12).
    pub(crate) fn get_physical_device_memory_properties(
        &self,
        device: sys::VkPhysicalDevice,
    ) -> sys::VkPhysicalDeviceMemoryProperties {
        let mut props: MaybeUninit<sys::VkPhysicalDeviceMemoryProperties> = MaybeUninit::uninit();
        // SAFETY: `get_physical_device_memory_properties` is the resolved
        // instance entry; `device` is a valid physical device handle; the out-
        // pointer is a writable struct slot the driver fully initialises.
        unsafe { (self.get_physical_device_memory_properties)(device, props.as_mut_ptr()) };
        // SAFETY: the driver call above fully initialises the struct.
        unsafe { props.assume_init() }
    }

    /// Enumerate queue-family properties for a physical device (T07).
    pub(crate) fn get_queue_family_properties(
        &self,
        device: sys::VkPhysicalDevice,
    ) -> Vec<sys::VkQueueFamilyProperties> {
        let mut count: u32 = 0;
        // SAFETY: resolved instance entry; `count` is a valid writable u32;
        // null out-pointer selects the spec-defined "count only" mode.
        unsafe {
            (self.get_physical_device_queue_family_properties)(
                device,
                &mut count,
                core::ptr::null_mut(),
            );
        }
        if count == 0 {
            return Vec::new();
        }
        let mut props: Vec<sys::VkQueueFamilyProperties> = vec![
            sys::VkQueueFamilyProperties {
                queue_flags: 0,
                queue_count: 0,
                timestamp_valid_bits: 0,
                min_image_transfer_granularity: sys::VkExtent3D {
                    width: 0,
                    height: 0,
                    depth: 0,
                },
            };
            count as usize
        ];
        // SAFETY: `props.as_mut_ptr()` is a valid pointer to `count` writable
        // `VkQueueFamilyProperties` slots; the driver call fully initialises
        // every entry.
        unsafe {
            (self.get_physical_device_queue_family_properties)(
                device,
                &mut count,
                props.as_mut_ptr(),
            );
        }
        props
    }

    /// Find the index of a compute-capable queue family on `device`.
    #[must_use]
    pub(crate) fn find_compute_queue_family(&self, device: sys::VkPhysicalDevice) -> Option<u32> {
        let families = self.get_queue_family_properties(device);
        // Pass 1 — compute-only family with at least one queue.
        for (i, f) in families.iter().enumerate() {
            if f.queue_count > 0
                && (f.queue_flags & sys::VK_QUEUE_COMPUTE_BIT) != 0
                && (f.queue_flags & sys::VK_QUEUE_GRAPHICS_BIT) == 0
            {
                return Some(i as u32);
            }
        }
        // Pass 2 — first compute-capable family (graphics + compute).
        for (i, f) in families.iter().enumerate() {
            if f.queue_count > 0 && (f.queue_flags & sys::VK_QUEUE_COMPUTE_BIT) != 0 {
                return Some(i as u32);
            }
        }
        None
    }

    /// Enumerate the device-level extensions the driver reports for `device`
    /// (M3-02-T30).
    ///
    /// Layer name is passed as null to enumerate the ICD-implicit extension
    /// set (the KHR / EXT extensions the vendor driver actually implements).
    pub(crate) fn enumerate_device_extensions(
        &self,
        device: sys::VkPhysicalDevice,
    ) -> Result<Vec<sys::VkExtensionProperties>> {
        let mut count: u32 = 0;
        // SAFETY: resolved instance entry; null layer name = "core + ICD-
        // implicit extensions"; `count` is a valid writable u32.
        let r = unsafe {
            (self.enumerate_device_extension_properties)(
                device,
                ptr::null(),
                &mut count,
                ptr::null_mut(),
            )
        };
        sys::check(r, "vkEnumerateDeviceExtensionProperties (count)")?;
        if count == 0 {
            return Ok(Vec::new());
        }
        // Pre-fill with zeroed `VkExtensionProperties`; the driver overwrites
        // every slot.
        let zero_ext = sys::VkExtensionProperties {
            extension_name: [0; sys::VK_MAX_EXTENSION_NAME_SIZE],
            spec_version: 0,
        };
        let mut props: Vec<sys::VkExtensionProperties> = vec![zero_ext; count as usize];
        // SAFETY: `props.as_mut_ptr()` is a valid pointer to `count` writable
        // slots; the driver call fully initialises each entry.
        let r = unsafe {
            (self.enumerate_device_extension_properties)(
                device,
                ptr::null(),
                &mut count,
                props.as_mut_ptr(),
            )
        };
        sys::check(r, "vkEnumerateDeviceExtensionProperties (fill)")?;
        Ok(props)
    }

    /// Returns `true` iff **any** `VkExtensionProperties.extension_name`
    /// entry equals `expected` (spec-defined ASCII, NUL-terminated). Used by
    /// the cooperative-matrix extension walk (T30) — checks either the KHR
    /// promoted `VK_KHR_cooperative_matrix` or the NVIDIA vendor
    /// `VK_NV_cooperative_matrix`.
    pub(crate) fn has_extension(exts: &[sys::VkExtensionProperties], expected: &str) -> bool {
        let expected_bytes = expected.as_bytes();
        for ext in exts {
            let bytes = extension_name_bytes(&ext.extension_name);
            if bytes == expected_bytes {
                return true;
            }
        }
        false
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
        // drops fields in declaration order after this Drop impl returns).
        let _ = &self.loader;
    }
}

/// Extract a byte-slice view of an extension_name up to (but excluding) the
/// first NUL. Vulkan guarantees NUL-termination and ASCII contents for both
/// device names and extension names.
fn extension_name_bytes(buf: &[c_char; sys::VK_MAX_EXTENSION_NAME_SIZE]) -> &[u8] {
    #[allow(clippy::unnecessary_cast)]
    let ptr = buf.as_ptr() as *const u8;
    // SAFETY: `c_char` and `u8` have the same representation on every
    // platform Vokra supports; slice length equals the array length so we do
    // not read past `buf`.
    let bytes: &[u8] = unsafe { core::slice::from_raw_parts(ptr, buf.len()) };
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    &bytes[..end]
}

fn missing_instance_symbol(name: &str) -> VokraError {
    VokraError::BackendUnavailable(format!(
        "Vulkan driver is missing `{name}` (impossible on any conforming ICD)."
    ))
}

// ---------------------------------------------------------------------------
// VulkanDevice — T08 (vkCreateDevice + compute queue) + T30 (coop-matrix ext
// walk). Owns its parent VulkanInstance so the loader stays alive until after
// vkDestroyDevice returns.
// ---------------------------------------------------------------------------

/// Bundle of all device-level function pointers Vokra resolves in
/// [`VulkanDevice::new`]. Kept as a struct-of-fn-pointers so children
/// ([`VulkanCommandPool`] / [`VulkanBuffer`] / …) can hold a `&DeviceFns`
/// borrow via `&VulkanDevice`, avoiding one indirection per FFI call.
///
/// Every field pairs a C symbol name with its exact `Fn*` alias declared in
/// [`crate::sys`] (which mirrors `vulkan_core.h`).
///
/// Not every fn pointer is consumed by the foundation slice — the fields
/// tagged as unused (e.g. `queue_wait_idle`, `reset_command_pool`,
/// `reset_command_buffer`) will be exercised by M3-02-T14+ dispatch code.
/// Keeping them resolved at construction time is intentional: symbol
/// resolution errors surface at backend init rather than mid-dispatch.
#[allow(dead_code)] // T14+ dispatch code lands the consumers
pub(crate) struct DeviceFns {
    // Device lifecycle.
    pub(crate) destroy_device: sys::FnVkDestroyDevice,
    pub(crate) device_wait_idle: sys::FnVkDeviceWaitIdle,
    pub(crate) get_device_queue: sys::FnVkGetDeviceQueue,
    // Queue submit + wait.
    pub(crate) queue_submit: sys::FnVkQueueSubmit,
    pub(crate) queue_wait_idle: sys::FnVkQueueWaitIdle,
    // Command pool / command buffer.
    pub(crate) create_command_pool: sys::FnVkCreateCommandPool,
    pub(crate) destroy_command_pool: sys::FnVkDestroyCommandPool,
    pub(crate) reset_command_pool: sys::FnVkResetCommandPool,
    pub(crate) allocate_command_buffers: sys::FnVkAllocateCommandBuffers,
    pub(crate) free_command_buffers: sys::FnVkFreeCommandBuffers,
    pub(crate) begin_command_buffer: sys::FnVkBeginCommandBuffer,
    pub(crate) end_command_buffer: sys::FnVkEndCommandBuffer,
    pub(crate) reset_command_buffer: sys::FnVkResetCommandBuffer,
    pub(crate) cmd_copy_buffer: sys::FnVkCmdCopyBuffer,
    // Compute dispatch commands (M3-02 handcrafted smoke + T14+ dispatch).
    pub(crate) cmd_bind_pipeline: sys::FnVkCmdBindPipeline,
    pub(crate) cmd_bind_descriptor_sets: sys::FnVkCmdBindDescriptorSets,
    pub(crate) cmd_dispatch: sys::FnVkCmdDispatch,
    pub(crate) cmd_push_constants: sys::FnVkCmdPushConstants,
    // Buffer + memory.
    pub(crate) create_buffer: sys::FnVkCreateBuffer,
    pub(crate) destroy_buffer: sys::FnVkDestroyBuffer,
    pub(crate) get_buffer_memory_requirements: sys::FnVkGetBufferMemoryRequirements,
    pub(crate) allocate_memory: sys::FnVkAllocateMemory,
    pub(crate) free_memory: sys::FnVkFreeMemory,
    pub(crate) bind_buffer_memory: sys::FnVkBindBufferMemory,
    pub(crate) map_memory: sys::FnVkMapMemory,
    pub(crate) unmap_memory: sys::FnVkUnmapMemory,
    // Fence.
    pub(crate) create_fence: sys::FnVkCreateFence,
    pub(crate) destroy_fence: sys::FnVkDestroyFence,
    pub(crate) wait_for_fences: sys::FnVkWaitForFences,
    pub(crate) reset_fences: sys::FnVkResetFences,
    // Descriptor set layout / pool / update.
    pub(crate) create_descriptor_set_layout: sys::FnVkCreateDescriptorSetLayout,
    pub(crate) destroy_descriptor_set_layout: sys::FnVkDestroyDescriptorSetLayout,
    pub(crate) create_descriptor_pool: sys::FnVkCreateDescriptorPool,
    pub(crate) destroy_descriptor_pool: sys::FnVkDestroyDescriptorPool,
    pub(crate) allocate_descriptor_sets: sys::FnVkAllocateDescriptorSets,
    pub(crate) update_descriptor_sets: sys::FnVkUpdateDescriptorSets,
    // Pipeline layout / shader module / compute pipeline.
    pub(crate) create_pipeline_layout: sys::FnVkCreatePipelineLayout,
    pub(crate) destroy_pipeline_layout: sys::FnVkDestroyPipelineLayout,
    pub(crate) create_shader_module: sys::FnVkCreateShaderModule,
    pub(crate) destroy_shader_module: sys::FnVkDestroyShaderModule,
    pub(crate) create_compute_pipelines: sys::FnVkCreateComputePipelines,
    pub(crate) destroy_pipeline: sys::FnVkDestroyPipeline,
}

/// A Vulkan logical device + one compute queue + the memory-type table
/// (M3-02-T08). Owns its parent [`VulkanInstance`] so the loader stays alive
/// until after `vkDestroyDevice` returns.
///
/// All children of the device (command pool, buffer, descriptor set, pipeline,
/// …) borrow `&VulkanDevice`, so the borrow checker enforces that the device
/// outlives its children — a resource-leak-preventing invariant.
pub(crate) struct VulkanDevice {
    // The VulkanInstance MUST be the first non-fn-pointer field: fields drop
    // in declaration order AFTER `Drop::drop(&mut self)` runs. `drop` calls
    // `vkDestroyDevice`, so the loader still needs to be alive at that point
    // — which is guaranteed because Drop::drop runs BEFORE field drops.
    _instance: VulkanInstance,
    physical_device: sys::VkPhysicalDevice,
    device: sys::VkDevice,
    queue: sys::VkQueue,
    queue_family_index: u32,
    memory_props: sys::VkPhysicalDeviceMemoryProperties,
    fns: DeviceFns,
}

impl VulkanDevice {
    /// Creates a compute-only logical device on the first compute-capable
    /// physical device.
    ///
    /// # Errors
    ///
    /// - [`VokraError::BackendUnavailable`] if no physical device is
    ///   enumerated;
    /// - [`VokraError::BackendUnavailable`] if the selected device has no
    ///   compute-capable queue family (impossible on any Vulkan-conformant
    ///   GPU — spec §5.3.1);
    /// - [`VokraError::BackendUnavailable`] if `vkCreateDevice` fails, or any
    ///   required device-level symbol is missing.
    pub(crate) fn new(instance: VulkanInstance) -> Result<VulkanDevice> {
        let physical_devices = instance.enumerate_physical_devices()?;
        if physical_devices.is_empty() {
            return Err(VokraError::BackendUnavailable(
                "Vulkan loader present but no physical devices enumerated (install a Vulkan ICD, \
                 e.g. mesa-vulkan-drivers for lavapipe, or the vendor driver)."
                    .to_owned(),
            ));
        }
        let physical_device = physical_devices[0];
        let queue_family_index = instance
            .find_compute_queue_family(physical_device)
            .ok_or_else(|| {
                VokraError::BackendUnavailable(
                    "selected Vulkan physical device exposes no compute queue family — \
                     non-conformant driver (spec §5.3.1 requires one)."
                        .to_owned(),
                )
            })?;
        let memory_props = instance.get_physical_device_memory_properties(physical_device);

        let queue_priorities: [f32; 1] = [1.0];
        let queue_ci = sys::VkDeviceQueueCreateInfo {
            s_type: sys::VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            queue_family_index,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
        };
        let device_ci = sys::VkDeviceCreateInfo {
            s_type: sys::VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_ci,
            enabled_layer_count: 0,
            pp_enabled_layer_names: ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: ptr::null(),
            p_enabled_features: ptr::null(),
        };
        let mut device: sys::VkDevice = ptr::null_mut();
        // SAFETY: `create_device` is the resolved `vkCreateDevice`; the
        // physical device is a live handle; the create-info struct is valid
        // for the duration of the call; `device` is a writable out-parameter;
        // the allocator callback is null (default).
        let r = unsafe {
            (instance.create_device)(physical_device, &device_ci, ptr::null(), &mut device)
        };
        sys::check(r, "vkCreateDevice")?;

        // Resolve device-level fn pointers via vkGetInstanceProcAddr (spec-
        // valid; the returned pointer is a loader trampoline dispatching to
        // the ICD's device-level implementation).
        //
        // SAFETY: every symbol name below matches its FnVk… alias
        // (vulkan_core.h). The instance/loader pair is live for the duration
        // of the calls.
        let fns = resolve_device_fns(&instance)?;

        // vkGetDeviceQueue — resolved separately because it is called
        // immediately below to fetch the compute queue handle.
        let get_device_queue = fns.get_device_queue;
        let mut queue: sys::VkQueue = ptr::null_mut();
        // SAFETY: `get_device_queue` is the resolved entry; `device` is the
        // freshly-created live handle; `queue` is a writable out-parameter;
        // queue-family-index / queue-index are within the range advertised in
        // the VkDeviceQueueCreateInfo above.
        unsafe { get_device_queue(device, queue_family_index, 0, &mut queue) };
        if queue.is_null() {
            // Undefined by the spec (vkGetDeviceQueue does not fail), but be
            // defensive against non-conformant drivers.
            // SAFETY: `device` is a live handle; `destroy_device` is the
            // resolved entry; allocator callback null.
            unsafe { (fns.destroy_device)(device, ptr::null()) };
            return Err(VokraError::BackendUnavailable(
                "Vulkan driver returned a null queue for a compute-capable queue family — \
                 non-conformant."
                    .to_owned(),
            ));
        }

        Ok(VulkanDevice {
            _instance: instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            memory_props,
            fns,
        })
    }

    /// The compute queue family index in use.
    #[must_use]
    #[allow(dead_code)] // consumers land with M3-02-T14+ (kept for symmetry with `.queue()`)
    pub(crate) fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    /// Access to the parent instance (for extension enumeration etc.).
    #[must_use]
    pub(crate) fn instance(&self) -> &VulkanInstance {
        &self._instance
    }

    /// Access to the physical device handle backing this device.
    #[must_use]
    pub(crate) fn physical_device(&self) -> sys::VkPhysicalDevice {
        self.physical_device
    }

    /// Locate a memory-type index satisfying `type_filter` (bit-mask from
    /// [`sys::VkMemoryRequirements::memory_type_bits`]) and the flag set in
    /// `required_props` (e.g. `HOST_VISIBLE | HOST_COHERENT` for a staging
    /// buffer, `DEVICE_LOCAL` for a compute working set).
    ///
    /// Returns the index into `self.memory_props.memory_types`, or an error
    /// if no matching type exists (unusual — every conformant driver exposes
    /// at least one host-visible and one device-local type).
    fn find_memory_type(&self, type_filter: u32, required_props: u32) -> Result<u32> {
        let count = self.memory_props.memory_type_count as usize;
        for i in 0..count {
            let bit = 1u32 << i;
            let mt = &self.memory_props.memory_types[i];
            if (type_filter & bit) != 0 && (mt.property_flags & required_props) == required_props {
                return Ok(i as u32);
            }
        }
        Err(VokraError::BackendUnavailable(format!(
            "no Vulkan memory type matches filter=0x{type_filter:x} required=0x{required_props:x} \
             — non-conformant driver or unusual heap layout"
        )))
    }

    /// Wait for all pending GPU work on this device to complete. Called
    /// during shutdown so no submissions outlive `Drop`.
    fn wait_idle(&self) -> Result<()> {
        // SAFETY: `device_wait_idle` is the resolved `vkDeviceWaitIdle`;
        // `self.device` is a live handle.
        let r = unsafe { (self.fns.device_wait_idle)(self.device) };
        sys::check(r, "vkDeviceWaitIdle")
    }

    /// Upload `data` into a fresh device-local [`VulkanBuffer`] via a
    /// transient staging buffer (M3-02-T25 upload half).
    ///
    /// Returns a device-local buffer with `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    /// VK_BUFFER_USAGE_TRANSFER_DST_BIT` — ready to feed into an SSBO
    /// descriptor for a compute pipeline (T14+).
    pub(crate) fn upload_bytes<'d>(&'d self, data: &[u8]) -> Result<VulkanBuffer<'d>> {
        let size = data.len();
        assert!(size > 0, "upload_bytes: empty input");
        // Staging: host-visible + host-coherent.
        let mut staging = VulkanBuffer::new(
            self,
            size,
            sys::VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            sys::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | sys::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )?;
        staging.write_bytes(data)?;
        // Device-local target buffer.
        let target = VulkanBuffer::new(
            self,
            size,
            sys::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | sys::VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sys::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )?;
        self.copy_buffer(&staging, &target, size)?;
        // `staging` drops here — its staging memory is freed automatically.
        drop(staging);
        Ok(target)
    }

    /// Download the full contents of `buffer` into `out` via a transient
    /// staging buffer (M3-02-T25 download half). `out.len()` must equal
    /// the buffer's byte size.
    pub(crate) fn download_bytes(&self, buffer: &VulkanBuffer<'_>, out: &mut [u8]) -> Result<()> {
        assert_eq!(
            out.len(),
            buffer.size,
            "download_bytes: destination slice length must match buffer size"
        );
        // Staging: host-visible + host-coherent.
        let staging = VulkanBuffer::new(
            self,
            buffer.size,
            sys::VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sys::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | sys::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )?;
        // Ensure `buffer` has TRANSFER_SRC usage. Currently every buffer we
        // create is either staging or a device-local target with only
        // STORAGE_BUFFER + TRANSFER_DST. For a general download, we should
        // ensure sources have TRANSFER_SRC; enforce that here.
        assert!(
            (buffer.usage & sys::VK_BUFFER_USAGE_TRANSFER_SRC_BIT) != 0,
            "download_bytes: source buffer must have TRANSFER_SRC usage",
        );
        self.copy_buffer(buffer, &staging, buffer.size)?;
        staging.read_bytes(out)?;
        Ok(())
    }

    /// Record a buffer→buffer copy of `size` bytes into a transient command
    /// buffer, submit to the compute queue, and wait on a fence for
    /// completion (M3-02-T25 core sync path).
    fn copy_buffer(
        &self,
        src: &VulkanBuffer<'_>,
        dst: &VulkanBuffer<'_>,
        size: usize,
    ) -> Result<()> {
        // A dedicated one-shot pool: transient + resettable command buffers.
        let pool_ci = sys::VkCommandPoolCreateInfo {
            s_type: sys::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: sys::VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queue_family_index: self.queue_family_index,
        };
        let mut pool: sys::VkCommandPool = 0;
        // SAFETY: resolved entry; live device; valid create-info.
        let r = unsafe {
            (self.fns.create_command_pool)(self.device, &pool_ci, ptr::null(), &mut pool)
        };
        sys::check(r, "vkCreateCommandPool (transient)")?;

        // Guard the pool so we destroy it even if a later step returns Err.
        let pool_guard = CommandPoolGuard {
            device: &self.fns,
            device_handle: self.device,
            pool,
        };

        let alloc_info = sys::VkCommandBufferAllocateInfo {
            s_type: sys::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool: pool,
            level: sys::VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            command_buffer_count: 1,
        };
        let mut cmd: sys::VkCommandBuffer = ptr::null_mut();
        // SAFETY: resolved entry; live device; valid alloc-info; writable out-
        // pointer.
        let r = unsafe { (self.fns.allocate_command_buffers)(self.device, &alloc_info, &mut cmd) };
        sys::check(r, "vkAllocateCommandBuffers (transient)")?;

        // Begin (ONE_TIME_SUBMIT — the driver may optimise on this hint).
        let begin_info = sys::VkCommandBufferBeginInfo {
            s_type: sys::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: sys::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: ptr::null(),
        };
        // SAFETY: resolved entry; live command buffer; valid begin-info.
        let r = unsafe { (self.fns.begin_command_buffer)(cmd, &begin_info) };
        sys::check(r, "vkBeginCommandBuffer (transient)")?;

        // Record vkCmdCopyBuffer.
        let region = sys::VkBufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: size as sys::VkDeviceSize,
        };
        // SAFETY: resolved entry; live command buffer; live source/target
        // buffers; single region on the stack.
        unsafe { (self.fns.cmd_copy_buffer)(cmd, src.handle, dst.handle, 1, &region) };

        // End + submit + wait.
        // SAFETY: resolved entry; live command buffer.
        let r = unsafe { (self.fns.end_command_buffer)(cmd) };
        sys::check(r, "vkEndCommandBuffer (transient)")?;

        let fence = VulkanFence::new(self)?;
        let submit_info = sys::VkSubmitInfo {
            s_type: sys::VK_STRUCTURE_TYPE_SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: ptr::null(),
            p_wait_dst_stage_mask: ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            signal_semaphore_count: 0,
            p_signal_semaphores: ptr::null(),
        };
        // SAFETY: resolved entry; live queue; single submit info on the stack;
        // fence handle is live.
        let r = unsafe { (self.fns.queue_submit)(self.queue, 1, &submit_info, fence.handle) };
        sys::check(r, "vkQueueSubmit (transient copy)")?;
        fence.wait()?;

        // Free the command buffer explicitly before the pool guard runs
        // (spec-mandated: command buffers freed via vkFreeCommandBuffers OR
        // destroyed together with the pool). Pool destroy also frees the
        // buffer, so the explicit free is a belt-and-suspenders.
        // SAFETY: resolved entry; live device; live pool; count matches
        // allocation.
        unsafe { (self.fns.free_command_buffers)(self.device, pool, 1, &cmd) };
        drop(pool_guard); // destroys the pool
        Ok(())
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        if self.device.is_null() {
            return;
        }
        // Best-effort idle before destroy — matches ash / vulkan-tutorial
        // conventions. If the wait itself errors, we still proceed to
        // destroy: leaking is worse than aborting a shutdown path.
        let _ = self.wait_idle();
        // SAFETY: `destroy_device` is the resolved `vkDestroyDevice`; the
        // device is live and destroyed exactly once here; allocator callback
        // null.
        unsafe { (self.fns.destroy_device)(self.device, ptr::null()) };
        self.device = ptr::null_mut();
        // `_instance` is dropped last (field order), so the loader is still
        // alive at this point — required by the Vulkan spec for
        // vkDestroyDevice.
    }
}

/// Smoke-test the T08〜T12 runtime object stack against a live device.
///
/// Creates + destroys a transient command pool, a small host→device→host
/// buffer round-trip, a descriptor set layout / pool / set, and a pipeline
/// layout — the exact objects M3-02-T14+ needs to build a compute dispatch.
///
/// Called once from [`crate::backend::VulkanBackend::new`] so a broken driver
/// surfaces as [`VokraError::BackendUnavailable`] at backend-construction
/// time (never a silent CPU fall back, FR-EX-08). Also serves as the "IS
/// this main-lib code reachable" anchor for the dead-code checker — every
/// helper in this module is transitively used by this function.
pub(crate) fn smoke_test_runtime_object_stack(device: &VulkanDevice) -> Result<()> {
    // Command pool + primary command buffer (T09) — the field is read here
    // to prove the allocation succeeded.
    let pool = VulkanCommandPool::new(device)?;
    if pool.command_buffer.is_null() {
        return Err(VokraError::BackendUnavailable(
            "Vulkan command pool allocated a null primary command buffer — driver bug".to_owned(),
        ));
    }

    // Buffer round-trip via the T25 upload/download helpers.
    let src: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(31));
    let buf = device.upload_bytes(&src)?;
    // A device-local buffer from upload_bytes has TRANSFER_DST but NOT
    // TRANSFER_SRC (the M3-02 host↔device path only writes to it). For the
    // round-trip smoke test we allocate a second target buffer that has
    // TRANSFER_SRC and stage into it, then read back.
    let buf_readable = VulkanBuffer::new(
        device,
        src.len(),
        sys::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | sys::VK_BUFFER_USAGE_TRANSFER_DST_BIT
            | sys::VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sys::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    )?;
    if buf_readable.size() != src.len() {
        return Err(VokraError::BackendUnavailable(format!(
            "Vulkan buffer reported wrong size: expected {}, got {}",
            src.len(),
            buf_readable.size(),
        )));
    }
    device.copy_buffer(&buf, &buf_readable, src.len())?;
    drop(buf);
    let mut got = [0u8; 64];
    device.download_bytes(&buf_readable, &mut got)?;
    if got != src {
        return Err(VokraError::BackendUnavailable(format!(
            "Vulkan host↔device round-trip returned corrupted bytes: expected {src:?}, got \
             {got:?} — driver bug (M3-02-T25 smoke)"
        )));
    }

    // Descriptor set layout / pool / set (T10) + pipeline layout (T11 without
    // a pipeline — the pipeline itself needs a SPIR-V blob, which the
    // foundation slice does not ship).
    let dsl = VulkanDescriptorSetLayout::new_storage_buffers(device, 1)?;
    let dpool = VulkanDescriptorPool::new_storage_buffers(device, 1, 1)?;
    let set = dpool.allocate_set(&dsl)?;
    // Bind the buffer to the descriptor set at binding 0 (T10 vkUpdateDescriptorSets).
    let buffer_info = sys::VkDescriptorBufferInfo {
        buffer: buf_readable.handle(),
        offset: 0,
        // VK_WHOLE_SIZE = ~0u64.
        range: sys::VkDeviceSize::MAX,
    };
    let write = sys::VkWriteDescriptorSet {
        s_type: sys::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        p_next: core::ptr::null(),
        dst_set: set,
        dst_binding: 0,
        dst_array_element: 0,
        descriptor_count: 1,
        descriptor_type: sys::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        p_image_info: core::ptr::null(),
        p_buffer_info: &buffer_info,
        p_texel_buffer_view: core::ptr::null(),
    };
    // SAFETY: resolved entry; live device; `write` is a single valid
    // VkWriteDescriptorSet on the stack; `buffer_info` outlives the call.
    unsafe {
        (device.fns.update_descriptor_sets)(device.device, 1, &write, 0, core::ptr::null());
    }

    // Pipeline layout wrapping the descriptor set layout (T11).
    let _pl = VulkanPipelineLayout::new(device, &dsl)?;

    // Extension walk (T30) — check whether cooperative-matrix precondition
    // extensions are present. Non-blocking (we don't require them here); this
    // exercises `instance().enumerate_device_extensions` + `has_extension`.
    let exts = device
        .instance()
        .enumerate_device_extensions(device.physical_device())?;
    let _has_coop = VulkanInstance::has_extension(&exts, "VK_KHR_cooperative_matrix")
        || VulkanInstance::has_extension(&exts, "VK_NV_cooperative_matrix");

    // Drop-order: buf_readable/set/dsl/dpool/_pl/pool released as this block ends.
    Ok(())
}

/// Small RAII helper for the transient command pool used by
/// [`VulkanDevice::copy_buffer`]. Destroys the pool on drop.
struct CommandPoolGuard<'d> {
    device: &'d DeviceFns,
    device_handle: sys::VkDevice,
    pool: sys::VkCommandPool,
}

impl Drop for CommandPoolGuard<'_> {
    fn drop(&mut self) {
        if self.pool == 0 {
            return;
        }
        // SAFETY: resolved entry; live device + pool handles; destroyed
        // exactly once here.
        unsafe { (self.device.destroy_command_pool)(self.device_handle, self.pool, ptr::null()) };
        self.pool = 0;
    }
}

/// Resolve every device-level fn pointer Vokra needs, via
/// `vkGetInstanceProcAddr` — a spec-valid path that returns a loader
/// trampoline dispatching to the ICD's device-level implementation. Used
/// once by [`VulkanDevice::new`] to build the [`DeviceFns`] table.
fn resolve_device_fns(instance: &VulkanInstance) -> Result<DeviceFns> {
    // The convenience closure resolves one symbol; the SAFETY note applies
    // to every unsafe block inside `instance_proc_or_err` calls below.
    macro_rules! resolve {
        ($ty:ty, $name:literal) => {{
            // SAFETY: `$name` is a NUL-terminated ASCII C string literal
            // paired with the FnVk… alias `$ty` (which mirrors the exact C
            // signature declared in vulkan_core.h). The loader/instance are
            // live for the duration of the call.
            let ptr: Option<$ty> = unsafe {
                sys::instance_proc(
                    &instance.loader,
                    instance.instance,
                    concat!($name, "\0").as_bytes(),
                )
            };
            ptr.ok_or_else(|| missing_instance_symbol($name))?
        }};
    }
    Ok(DeviceFns {
        destroy_device: resolve!(sys::FnVkDestroyDevice, "vkDestroyDevice"),
        device_wait_idle: resolve!(sys::FnVkDeviceWaitIdle, "vkDeviceWaitIdle"),
        get_device_queue: resolve!(sys::FnVkGetDeviceQueue, "vkGetDeviceQueue"),
        queue_submit: resolve!(sys::FnVkQueueSubmit, "vkQueueSubmit"),
        queue_wait_idle: resolve!(sys::FnVkQueueWaitIdle, "vkQueueWaitIdle"),
        create_command_pool: resolve!(sys::FnVkCreateCommandPool, "vkCreateCommandPool"),
        destroy_command_pool: resolve!(sys::FnVkDestroyCommandPool, "vkDestroyCommandPool"),
        reset_command_pool: resolve!(sys::FnVkResetCommandPool, "vkResetCommandPool"),
        allocate_command_buffers: resolve!(
            sys::FnVkAllocateCommandBuffers,
            "vkAllocateCommandBuffers"
        ),
        free_command_buffers: resolve!(sys::FnVkFreeCommandBuffers, "vkFreeCommandBuffers"),
        begin_command_buffer: resolve!(sys::FnVkBeginCommandBuffer, "vkBeginCommandBuffer"),
        end_command_buffer: resolve!(sys::FnVkEndCommandBuffer, "vkEndCommandBuffer"),
        reset_command_buffer: resolve!(sys::FnVkResetCommandBuffer, "vkResetCommandBuffer"),
        cmd_copy_buffer: resolve!(sys::FnVkCmdCopyBuffer, "vkCmdCopyBuffer"),
        cmd_bind_pipeline: resolve!(sys::FnVkCmdBindPipeline, "vkCmdBindPipeline"),
        cmd_bind_descriptor_sets: resolve!(
            sys::FnVkCmdBindDescriptorSets,
            "vkCmdBindDescriptorSets"
        ),
        cmd_dispatch: resolve!(sys::FnVkCmdDispatch, "vkCmdDispatch"),
        cmd_push_constants: resolve!(sys::FnVkCmdPushConstants, "vkCmdPushConstants"),
        create_buffer: resolve!(sys::FnVkCreateBuffer, "vkCreateBuffer"),
        destroy_buffer: resolve!(sys::FnVkDestroyBuffer, "vkDestroyBuffer"),
        get_buffer_memory_requirements: resolve!(
            sys::FnVkGetBufferMemoryRequirements,
            "vkGetBufferMemoryRequirements"
        ),
        allocate_memory: resolve!(sys::FnVkAllocateMemory, "vkAllocateMemory"),
        free_memory: resolve!(sys::FnVkFreeMemory, "vkFreeMemory"),
        bind_buffer_memory: resolve!(sys::FnVkBindBufferMemory, "vkBindBufferMemory"),
        map_memory: resolve!(sys::FnVkMapMemory, "vkMapMemory"),
        unmap_memory: resolve!(sys::FnVkUnmapMemory, "vkUnmapMemory"),
        create_fence: resolve!(sys::FnVkCreateFence, "vkCreateFence"),
        destroy_fence: resolve!(sys::FnVkDestroyFence, "vkDestroyFence"),
        wait_for_fences: resolve!(sys::FnVkWaitForFences, "vkWaitForFences"),
        reset_fences: resolve!(sys::FnVkResetFences, "vkResetFences"),
        create_descriptor_set_layout: resolve!(
            sys::FnVkCreateDescriptorSetLayout,
            "vkCreateDescriptorSetLayout"
        ),
        destroy_descriptor_set_layout: resolve!(
            sys::FnVkDestroyDescriptorSetLayout,
            "vkDestroyDescriptorSetLayout"
        ),
        create_descriptor_pool: resolve!(sys::FnVkCreateDescriptorPool, "vkCreateDescriptorPool"),
        destroy_descriptor_pool: resolve!(
            sys::FnVkDestroyDescriptorPool,
            "vkDestroyDescriptorPool"
        ),
        allocate_descriptor_sets: resolve!(
            sys::FnVkAllocateDescriptorSets,
            "vkAllocateDescriptorSets"
        ),
        update_descriptor_sets: resolve!(sys::FnVkUpdateDescriptorSets, "vkUpdateDescriptorSets"),
        create_pipeline_layout: resolve!(sys::FnVkCreatePipelineLayout, "vkCreatePipelineLayout"),
        destroy_pipeline_layout: resolve!(
            sys::FnVkDestroyPipelineLayout,
            "vkDestroyPipelineLayout"
        ),
        create_shader_module: resolve!(sys::FnVkCreateShaderModule, "vkCreateShaderModule"),
        destroy_shader_module: resolve!(sys::FnVkDestroyShaderModule, "vkDestroyShaderModule"),
        create_compute_pipelines: resolve!(
            sys::FnVkCreateComputePipelines,
            "vkCreateComputePipelines"
        ),
        destroy_pipeline: resolve!(sys::FnVkDestroyPipeline, "vkDestroyPipeline"),
    })
}

// ---------------------------------------------------------------------------
// VulkanCommandPool — T09. Command pool + one primary command buffer.
// ---------------------------------------------------------------------------

/// A Vulkan `VkCommandPool` + a preallocated primary `VkCommandBuffer`
/// (M3-02-T09). Created with
/// `VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT` so the caller can reuse
/// the command buffer across submissions via `vkResetCommandBuffer`.
///
/// Borrows `&VulkanDevice`, so the borrow checker enforces that the device
/// outlives the pool.
pub(crate) struct VulkanCommandPool<'d> {
    device: &'d VulkanDevice,
    pool: sys::VkCommandPool,
    /// Preallocated primary command buffer. The caller can reset + rerecord
    /// it across submissions.
    pub(crate) command_buffer: sys::VkCommandBuffer,
}

impl<'d> VulkanCommandPool<'d> {
    /// Creates a command pool with reset-command-buffer capability + a single
    /// primary command buffer.
    pub(crate) fn new(device: &'d VulkanDevice) -> Result<VulkanCommandPool<'d>> {
        let pool_ci = sys::VkCommandPoolCreateInfo {
            s_type: sys::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: sys::VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queue_family_index: device.queue_family_index,
        };
        let mut pool: sys::VkCommandPool = 0;
        // SAFETY: resolved entry; live device; valid create-info; writable
        // out-parameter.
        let r = unsafe {
            (device.fns.create_command_pool)(device.device, &pool_ci, ptr::null(), &mut pool)
        };
        sys::check(r, "vkCreateCommandPool")?;

        let alloc_info = sys::VkCommandBufferAllocateInfo {
            s_type: sys::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool: pool,
            level: sys::VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            command_buffer_count: 1,
        };
        let mut cmd: sys::VkCommandBuffer = ptr::null_mut();
        // SAFETY: resolved entry; live device; valid alloc-info; writable
        // out-pointer.
        let r =
            unsafe { (device.fns.allocate_command_buffers)(device.device, &alloc_info, &mut cmd) };
        if r != sys::VK_SUCCESS {
            // SAFETY: resolved entry; pool is live; allocator callback null.
            unsafe { (device.fns.destroy_command_pool)(device.device, pool, ptr::null()) };
            return Err(VokraError::BackendUnavailable(format!(
                "vkAllocateCommandBuffers failed with VkResult={r}"
            )));
        }
        Ok(VulkanCommandPool {
            device,
            pool,
            command_buffer: cmd,
        })
    }
}

impl Drop for VulkanCommandPool<'_> {
    fn drop(&mut self) {
        if self.pool == 0 {
            return;
        }
        // Destroying the pool also frees any command buffers allocated from
        // it (spec §6.2), so we do not need an explicit vkFreeCommandBuffers
        // call here.
        // SAFETY: resolved entry; live device + pool; destroyed exactly once.
        unsafe {
            (self.device.fns.destroy_command_pool)(self.device.device, self.pool, ptr::null())
        };
        self.pool = 0;
    }
}

// ---------------------------------------------------------------------------
// VulkanBuffer + VulkanFence — T12 + T25. VkBuffer + backing VkDeviceMemory.
// ---------------------------------------------------------------------------

/// Kind of memory Vokra requests when allocating a [`VulkanBuffer`].
///
/// This is a *high-level* selector; [`VulkanBuffer::new`] takes the raw usage
/// / property masks so it can express combinations like "host-visible +
/// transfer-src" that the enum would need extra variants for.
#[allow(dead_code)] // convenience constructors added in T25 follow-up
pub(crate) enum BufferKind {
    /// Host-visible + host-coherent — used for staging round-trips.
    Staging,
    /// Device-local — the compute working set.
    DeviceLocal,
}

/// A `VkBuffer` + backing `VkDeviceMemory` (M3-02-T12). Bound and ready to
/// use as an SSBO / staging buffer / transfer source or destination.
pub(crate) struct VulkanBuffer<'d> {
    device: &'d VulkanDevice,
    handle: sys::VkBuffer,
    memory: sys::VkDeviceMemory,
    size: usize,
    usage: u32,
    property_flags: u32,
}

impl<'d> VulkanBuffer<'d> {
    /// Allocate a Vulkan buffer of `size` bytes with the given usage +
    /// memory-property masks. The buffer is bound to a fresh VkDeviceMemory
    /// allocation immediately.
    pub(crate) fn new(
        device: &'d VulkanDevice,
        size: usize,
        usage: u32,
        property_flags: u32,
    ) -> Result<VulkanBuffer<'d>> {
        assert!(size > 0, "VulkanBuffer::new: size must be > 0");
        let buffer_ci = sys::VkBufferCreateInfo {
            s_type: sys::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            size: size as sys::VkDeviceSize,
            usage,
            sharing_mode: sys::VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
        };
        let mut handle: sys::VkBuffer = 0;
        // SAFETY: resolved entry; live device; valid create-info; writable
        // out-parameter.
        let r = unsafe {
            (device.fns.create_buffer)(device.device, &buffer_ci, ptr::null(), &mut handle)
        };
        sys::check(r, "vkCreateBuffer")?;

        // Query memory requirements.
        let mut mem_req = MaybeUninit::<sys::VkMemoryRequirements>::uninit();
        // SAFETY: resolved entry; live device + buffer; writable out-pointer.
        unsafe {
            (device.fns.get_buffer_memory_requirements)(device.device, handle, mem_req.as_mut_ptr())
        };
        // SAFETY: fully initialised by the call above.
        let mem_req = unsafe { mem_req.assume_init() };

        // Pick a memory type index matching filter + required properties.
        let mem_type_index = match device.find_memory_type(mem_req.memory_type_bits, property_flags)
        {
            Ok(idx) => idx,
            Err(e) => {
                // SAFETY: buffer is live; destroyed exactly once here on
                // the error path.
                unsafe { (device.fns.destroy_buffer)(device.device, handle, ptr::null()) };
                return Err(e);
            }
        };

        let alloc_info = sys::VkMemoryAllocateInfo {
            s_type: sys::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: mem_req.size,
            memory_type_index: mem_type_index,
        };
        let mut memory: sys::VkDeviceMemory = 0;
        // SAFETY: resolved entry; live device; valid alloc-info; writable
        // out-parameter.
        let r = unsafe {
            (device.fns.allocate_memory)(device.device, &alloc_info, ptr::null(), &mut memory)
        };
        if r != sys::VK_SUCCESS {
            // SAFETY: buffer is live; destroyed exactly once here on the
            // error path.
            unsafe { (device.fns.destroy_buffer)(device.device, handle, ptr::null()) };
            return Err(VokraError::BackendUnavailable(format!(
                "vkAllocateMemory failed with VkResult={r}"
            )));
        }

        // Bind memory to buffer.
        // SAFETY: resolved entry; live device + buffer + memory; offset 0.
        let r = unsafe { (device.fns.bind_buffer_memory)(device.device, handle, memory, 0) };
        if r != sys::VK_SUCCESS {
            // SAFETY: buffer + memory are live; destroyed exactly once here
            // on the error path.
            unsafe {
                (device.fns.free_memory)(device.device, memory, ptr::null());
                (device.fns.destroy_buffer)(device.device, handle, ptr::null());
            };
            return Err(VokraError::BackendUnavailable(format!(
                "vkBindBufferMemory failed with VkResult={r}"
            )));
        }

        Ok(VulkanBuffer {
            device,
            handle,
            memory,
            size,
            usage,
            property_flags,
        })
    }

    /// Byte size of the buffer.
    #[must_use]
    pub(crate) fn size(&self) -> usize {
        self.size
    }

    /// Raw `VkBuffer` handle (for T14+ descriptor set writes).
    #[must_use]
    pub(crate) fn handle(&self) -> sys::VkBuffer {
        self.handle
    }

    /// Copy `data` into this buffer's memory. Requires `HOST_VISIBLE`.
    fn write_bytes(&mut self, data: &[u8]) -> Result<()> {
        assert!(
            (self.property_flags & sys::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0,
            "VulkanBuffer::write_bytes requires HOST_VISIBLE memory",
        );
        assert!(
            data.len() <= self.size,
            "VulkanBuffer::write_bytes: data larger than buffer",
        );
        let mut mapped: *mut c_void = ptr::null_mut();
        // SAFETY: resolved entry; live device + memory; VK_WHOLE_SIZE
        // encoded as `sys::VkDeviceSize::MAX` (spec §11.6).
        let r = unsafe {
            (self.device.fns.map_memory)(
                self.device.device,
                self.memory,
                0,
                sys::VkDeviceSize::MAX,
                0,
                &mut mapped,
            )
        };
        sys::check(r, "vkMapMemory (write_bytes)")?;
        // SAFETY: `mapped` is a valid pointer to at least `self.size` writable
        // bytes returned by the driver; we copy at most `data.len() <= size`
        // bytes into it.
        unsafe {
            core::ptr::copy_nonoverlapping(data.as_ptr(), mapped as *mut u8, data.len());
        }
        // SAFETY: resolved entry; live device + memory (mapped by the
        // matching vkMapMemory above).
        unsafe { (self.device.fns.unmap_memory)(self.device.device, self.memory) };
        // HOST_COHERENT ⇒ no explicit flush/invalidate needed.
        Ok(())
    }

    /// Copy this buffer's memory into `out`. Requires `HOST_VISIBLE`.
    fn read_bytes(&self, out: &mut [u8]) -> Result<()> {
        assert!(
            (self.property_flags & sys::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0,
            "VulkanBuffer::read_bytes requires HOST_VISIBLE memory",
        );
        assert!(
            out.len() <= self.size,
            "VulkanBuffer::read_bytes: destination larger than buffer",
        );
        let mut mapped: *mut c_void = ptr::null_mut();
        // SAFETY: resolved entry; live device + memory.
        let r = unsafe {
            (self.device.fns.map_memory)(
                self.device.device,
                self.memory,
                0,
                sys::VkDeviceSize::MAX,
                0,
                &mut mapped,
            )
        };
        sys::check(r, "vkMapMemory (read_bytes)")?;
        // SAFETY: `mapped` is a valid pointer to at least `self.size` readable
        // bytes; we copy at most `out.len() <= size` bytes out.
        unsafe {
            core::ptr::copy_nonoverlapping(mapped as *const u8, out.as_mut_ptr(), out.len());
        }
        // SAFETY: resolved entry; live device + memory (mapped above).
        unsafe { (self.device.fns.unmap_memory)(self.device.device, self.memory) };
        Ok(())
    }
}

impl Drop for VulkanBuffer<'_> {
    fn drop(&mut self) {
        // Buffer + backing memory are destroyed exactly once here.
        if self.handle != 0 {
            // SAFETY: resolved entry; live device + buffer; destroyed once.
            unsafe {
                (self.device.fns.destroy_buffer)(self.device.device, self.handle, ptr::null())
            };
            self.handle = 0;
        }
        if self.memory != 0 {
            // SAFETY: resolved entry; live device + memory; freed once.
            unsafe { (self.device.fns.free_memory)(self.device.device, self.memory, ptr::null()) };
            self.memory = 0;
        }
    }
}

/// A `VkFence` — host-visible GPU sync primitive (T25 core sync path).
struct VulkanFence<'d> {
    device: &'d VulkanDevice,
    handle: sys::VkFence,
}

impl<'d> VulkanFence<'d> {
    fn new(device: &'d VulkanDevice) -> Result<VulkanFence<'d>> {
        let ci = sys::VkFenceCreateInfo {
            s_type: sys::VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0, // not signalled at creation
        };
        let mut handle: sys::VkFence = 0;
        // SAFETY: resolved entry; live device; valid create-info; writable
        // out-parameter.
        let r = unsafe { (device.fns.create_fence)(device.device, &ci, ptr::null(), &mut handle) };
        sys::check(r, "vkCreateFence")?;
        Ok(VulkanFence { device, handle })
    }

    /// Wait for the fence to be signalled (`vkWaitForFences waitAll=true`,
    /// 5-second timeout). Timeouts are treated as errors — a stuck GPU is
    /// not something Vokra silently ignores.
    fn wait(&self) -> Result<()> {
        const FIVE_SECONDS_NS: u64 = 5_000_000_000;
        // SAFETY: resolved entry; live device + fence; single-fence array on
        // the stack; timeout in ns.
        let r = unsafe {
            (self.device.fns.wait_for_fences)(
                self.device.device,
                1,
                &self.handle,
                1, // waitAll
                FIVE_SECONDS_NS,
            )
        };
        if r == sys::VK_TIMEOUT {
            return Err(VokraError::BackendUnavailable(
                "vkWaitForFences timed out after 5 s (transient buffer copy) — GPU appears \
                 unresponsive."
                    .to_owned(),
            ));
        }
        sys::check(r, "vkWaitForFences")
    }
}

impl Drop for VulkanFence<'_> {
    fn drop(&mut self) {
        if self.handle == 0 {
            return;
        }
        // SAFETY: resolved entry; live device + fence; destroyed exactly once.
        unsafe { (self.device.fns.destroy_fence)(self.device.device, self.handle, ptr::null()) };
        self.handle = 0;
    }
}

// ---------------------------------------------------------------------------
// Descriptor set layout / pool / set (T10) + pipeline layout / shader module
// / compute pipeline (T11). Foundation-slice: create + destroy are wired,
// pipeline creation requires a SPIR-V blob (returns NotImplemented until
// T14+ blobs land).
// ---------------------------------------------------------------------------

/// A `VkDescriptorSetLayout` describing storage-buffer bindings for a
/// compute pipeline (M3-02-T10). Owned by the caller and destroyed on drop.
pub(crate) struct VulkanDescriptorSetLayout<'d> {
    device: &'d VulkanDevice,
    handle: sys::VkDescriptorSetLayout,
}

impl<'d> VulkanDescriptorSetLayout<'d> {
    /// Create a layout with `binding_count` storage-buffer bindings at
    /// contiguous binding indices `[0, binding_count)`, all for the compute
    /// stage.
    pub(crate) fn new_storage_buffers(
        device: &'d VulkanDevice,
        binding_count: u32,
    ) -> Result<VulkanDescriptorSetLayout<'d>> {
        assert!(
            binding_count > 0 && binding_count <= 64,
            "storage-buffer binding count out of sane range",
        );
        let bindings: Vec<sys::VkDescriptorSetLayoutBinding> = (0..binding_count)
            .map(|i| sys::VkDescriptorSetLayoutBinding {
                binding: i,
                descriptor_type: sys::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: sys::VK_SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: ptr::null(),
            })
            .collect();
        let ci = sys::VkDescriptorSetLayoutCreateInfo {
            s_type: sys::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            binding_count,
            p_bindings: bindings.as_ptr(),
        };
        let mut handle: sys::VkDescriptorSetLayout = 0;
        // SAFETY: resolved entry; live device; valid create-info; writable
        // out-parameter.
        let r = unsafe {
            (device.fns.create_descriptor_set_layout)(device.device, &ci, ptr::null(), &mut handle)
        };
        sys::check(r, "vkCreateDescriptorSetLayout")?;
        Ok(VulkanDescriptorSetLayout { device, handle })
    }

    #[must_use]
    #[allow(dead_code)] // T14+ dispatch code lands the consumer
    pub(crate) fn handle(&self) -> sys::VkDescriptorSetLayout {
        self.handle
    }
}

impl Drop for VulkanDescriptorSetLayout<'_> {
    fn drop(&mut self) {
        if self.handle == 0 {
            return;
        }
        // SAFETY: resolved entry; live device + layout; destroyed once.
        unsafe {
            (self.device.fns.destroy_descriptor_set_layout)(
                self.device.device,
                self.handle,
                ptr::null(),
            )
        };
        self.handle = 0;
    }
}

/// A `VkDescriptorPool` from which descriptor sets are allocated (M3-02-T10).
/// Vokra sizes the pool for a single-set, storage-buffer-only workload — the
/// T14+ tickets will grow this as needed.
pub(crate) struct VulkanDescriptorPool<'d> {
    device: &'d VulkanDevice,
    handle: sys::VkDescriptorPool,
}

impl<'d> VulkanDescriptorPool<'d> {
    /// Create a descriptor pool with capacity for `max_sets` descriptor sets
    /// and `total_storage_buffers` storage-buffer descriptors across them.
    pub(crate) fn new_storage_buffers(
        device: &'d VulkanDevice,
        max_sets: u32,
        total_storage_buffers: u32,
    ) -> Result<VulkanDescriptorPool<'d>> {
        assert!(max_sets > 0);
        assert!(total_storage_buffers > 0);
        let pool_size = sys::VkDescriptorPoolSize {
            ty: sys::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count: total_storage_buffers,
        };
        let ci = sys::VkDescriptorPoolCreateInfo {
            s_type: sys::VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            max_sets,
            pool_size_count: 1,
            p_pool_sizes: &pool_size,
        };
        let mut handle: sys::VkDescriptorPool = 0;
        // SAFETY: resolved entry; live device; valid create-info; writable
        // out-parameter.
        let r = unsafe {
            (device.fns.create_descriptor_pool)(device.device, &ci, ptr::null(), &mut handle)
        };
        sys::check(r, "vkCreateDescriptorPool")?;
        Ok(VulkanDescriptorPool { device, handle })
    }

    /// Allocate one descriptor set from the pool using `layout`. The returned
    /// set's lifetime is tied to the pool (destroying the pool implicitly
    /// frees all its sets).
    pub(crate) fn allocate_set(
        &self,
        layout: &VulkanDescriptorSetLayout<'_>,
    ) -> Result<sys::VkDescriptorSet> {
        let alloc_info = sys::VkDescriptorSetAllocateInfo {
            s_type: sys::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool: self.handle,
            descriptor_set_count: 1,
            p_set_layouts: &layout.handle,
        };
        let mut set: sys::VkDescriptorSet = 0;
        // SAFETY: resolved entry; live device + pool; valid alloc-info;
        // writable out-parameter.
        let r = unsafe {
            (self.device.fns.allocate_descriptor_sets)(self.device.device, &alloc_info, &mut set)
        };
        sys::check(r, "vkAllocateDescriptorSets")?;
        Ok(set)
    }
}

impl Drop for VulkanDescriptorPool<'_> {
    fn drop(&mut self) {
        if self.handle == 0 {
            return;
        }
        // SAFETY: resolved entry; live device + pool; destroyed once
        // (implicitly frees all sets allocated from it).
        unsafe {
            (self.device.fns.destroy_descriptor_pool)(self.device.device, self.handle, ptr::null())
        };
        self.handle = 0;
    }
}

/// A `VkPipelineLayout` derived from a set of descriptor set layouts
/// (M3-02-T11). Used to create compute pipelines with matching binding
/// layouts.
pub(crate) struct VulkanPipelineLayout<'d> {
    device: &'d VulkanDevice,
    handle: sys::VkPipelineLayout,
}

impl<'d> VulkanPipelineLayout<'d> {
    /// Create a pipeline layout wrapping a single descriptor set layout.
    pub(crate) fn new(
        device: &'d VulkanDevice,
        set_layout: &VulkanDescriptorSetLayout<'_>,
    ) -> Result<VulkanPipelineLayout<'d>> {
        Self::new_with_push_constants(device, set_layout, 0)
    }

    /// Create a pipeline layout wrapping a single descriptor set layout plus
    /// a compute-stage push-constant range of `push_constant_size` bytes at
    /// offset 0 (M4-13-T02). `push_constant_size == 0` declares no range —
    /// the M3-02 hand-crafted smoke kernels take no push constants.
    ///
    /// `push_constant_size` must be a multiple of 4 and at most 128 bytes —
    /// the Vulkan spec's *guaranteed minimum* for
    /// `VkPhysicalDeviceLimits::maxPushConstantsSize` (spec §42.1 "Required
    /// Limits"), so a block that fits 128 bytes is portable to every
    /// conformant device without querying the limit.
    pub(crate) fn new_with_push_constants(
        device: &'d VulkanDevice,
        set_layout: &VulkanDescriptorSetLayout<'_>,
        push_constant_size: u32,
    ) -> Result<VulkanPipelineLayout<'d>> {
        if push_constant_size % 4 != 0 || push_constant_size > 128 {
            return Err(VokraError::InvalidArgument(format!(
                "vulkan pipeline layout: push-constant size {push_constant_size} must be a \
                 multiple of 4 and <= 128 bytes (spec §42.1 guaranteed minimum for \
                 maxPushConstantsSize)"
            )));
        }
        let push_range = sys::VkPushConstantRange {
            stage_flags: sys::VK_SHADER_STAGE_COMPUTE_BIT,
            offset: 0,
            size: push_constant_size,
        };
        let ci = sys::VkPipelineLayoutCreateInfo {
            s_type: sys::VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: &set_layout.handle,
            push_constant_range_count: u32::from(push_constant_size > 0),
            p_push_constant_ranges: if push_constant_size > 0 {
                &push_range
            } else {
                ptr::null()
            },
        };
        let mut handle: sys::VkPipelineLayout = 0;
        // SAFETY: resolved entry; live device; valid create-info (the
        // push-constant range on the stack outlives the call); writable
        // out-parameter.
        let r = unsafe {
            (device.fns.create_pipeline_layout)(device.device, &ci, ptr::null(), &mut handle)
        };
        sys::check(r, "vkCreatePipelineLayout")?;
        Ok(VulkanPipelineLayout { device, handle })
    }

    #[must_use]
    #[allow(dead_code)] // T14+ dispatch code lands the consumer
    pub(crate) fn handle(&self) -> sys::VkPipelineLayout {
        self.handle
    }
}

impl Drop for VulkanPipelineLayout<'_> {
    fn drop(&mut self) {
        if self.handle == 0 {
            return;
        }
        // SAFETY: resolved entry; live device + layout; destroyed once.
        unsafe {
            (self.device.fns.destroy_pipeline_layout)(self.device.device, self.handle, ptr::null())
        };
        self.handle = 0;
    }
}

/// A `VkShaderModule` wrapping a SPIR-V blob (M3-02-T11). Scaffolding for
/// M3-02-T14+ — the foundation slice ships no `.spv` blobs, so the
/// constructor surfaces `UnsupportedOp` on an empty input.
#[allow(dead_code)] // T14+ dispatch code lands the consumer
pub(crate) struct VulkanShaderModule<'d> {
    device: &'d VulkanDevice,
    handle: sys::VkShaderModule,
}

#[allow(dead_code)] // T14+ dispatch code lands the consumer
impl<'d> VulkanShaderModule<'d> {
    /// Create a shader module from a SPIR-V blob (`spv_bytes.len() % 4 == 0`
    /// required by the Vulkan spec).
    pub(crate) fn new(
        device: &'d VulkanDevice,
        spv_bytes: &[u8],
    ) -> Result<VulkanShaderModule<'d>> {
        if spv_bytes.is_empty() {
            return Err(VokraError::UnsupportedOp(
                "vulkan shader module: empty SPIR-V blob (foundation slice ships no `.spv`; \
                 M3-02-T14+ will land the actual blobs — no silent CPU fallback, FR-EX-08)"
                    .to_owned(),
            ));
        }
        if spv_bytes.len() % 4 != 0 {
            return Err(VokraError::UnsupportedOp(format!(
                "vulkan shader module: SPIR-V blob length {} is not a multiple of 4 (spec §9.1)",
                spv_bytes.len()
            )));
        }
        let ci = sys::VkShaderModuleCreateInfo {
            s_type: sys::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            code_size: spv_bytes.len(),
            p_code: spv_bytes.as_ptr() as *const u32,
        };
        let mut handle: sys::VkShaderModule = 0;
        // SAFETY: resolved entry; live device; valid create-info (blob slice
        // outlives the call); writable out-parameter.
        let r = unsafe {
            (device.fns.create_shader_module)(device.device, &ci, ptr::null(), &mut handle)
        };
        sys::check(r, "vkCreateShaderModule")?;
        Ok(VulkanShaderModule { device, handle })
    }

    #[must_use]
    pub(crate) fn handle(&self) -> sys::VkShaderModule {
        self.handle
    }
}

impl Drop for VulkanShaderModule<'_> {
    fn drop(&mut self) {
        if self.handle == 0 {
            return;
        }
        // SAFETY: resolved entry; live device + shader module; destroyed once.
        unsafe {
            (self.device.fns.destroy_shader_module)(self.device.device, self.handle, ptr::null())
        };
        self.handle = 0;
    }
}

/// A `VkPipeline` for compute dispatch (M3-02-T11). The foundation-slice
/// constructor requires a SPIR-V blob + entry-point name; when the blob is
/// absent [`VulkanShaderModule::new`] surfaces `UnsupportedOp` upstream.
#[allow(dead_code)] // T14+ dispatch code lands the consumer
pub(crate) struct VulkanComputePipeline<'d> {
    device: &'d VulkanDevice,
    handle: sys::VkPipeline,
}

#[allow(dead_code)] // T14+ dispatch code lands the consumer
impl<'d> VulkanComputePipeline<'d> {
    /// Create a compute pipeline bound to `layout` using the SPIR-V module
    /// `shader` with entry point `entry_name` (usually `"main"`).
    pub(crate) fn new(
        device: &'d VulkanDevice,
        layout: &VulkanPipelineLayout<'_>,
        shader: &VulkanShaderModule<'_>,
        entry_name: &core::ffi::CStr,
    ) -> Result<VulkanComputePipeline<'d>> {
        Self::new_specialized(device, layout, shader, entry_name, &[])
    }

    /// [`VulkanComputePipeline::new`] with `layout(constant_id = N)` GLSL
    /// specialization constants applied at pipeline-creation time
    /// (M4-13-T02/T07 — `elementwise` OP and `activation` KIND selection).
    /// An empty `spec_constants` slice creates an unspecialised pipeline
    /// (every `constant_id` keeps its GLSL default).
    pub(crate) fn new_specialized(
        device: &'d VulkanDevice,
        layout: &VulkanPipelineLayout<'_>,
        shader: &VulkanShaderModule<'_>,
        entry_name: &core::ffi::CStr,
        spec_constants: &[SpecConstantU32],
    ) -> Result<VulkanComputePipeline<'d>> {
        // Pack the u32 specialization values into a contiguous LE data blob
        // with one map entry per constant (spec §9.8). Both vectors must
        // outlive the vkCreateComputePipelines call below.
        let mut spec_data: Vec<u8> = Vec::with_capacity(spec_constants.len() * 4);
        let mut spec_entries: Vec<sys::VkSpecializationMapEntry> =
            Vec::with_capacity(spec_constants.len());
        for (i, sc) in spec_constants.iter().enumerate() {
            spec_entries.push(sys::VkSpecializationMapEntry {
                constant_id: sc.constant_id,
                offset: (i * 4) as u32,
                size: 4,
            });
            spec_data.extend_from_slice(&sc.value.to_le_bytes());
        }
        let spec_info = sys::VkSpecializationInfo {
            map_entry_count: spec_entries.len() as u32,
            p_map_entries: spec_entries.as_ptr(),
            data_size: spec_data.len(),
            p_data: spec_data.as_ptr() as *const c_void,
        };
        let stage_ci = sys::VkPipelineShaderStageCreateInfo {
            s_type: sys::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            stage: sys::VK_SHADER_STAGE_COMPUTE_BIT,
            module: shader.handle,
            p_name: entry_name.as_ptr(),
            p_specialization_info: if spec_constants.is_empty() {
                ptr::null()
            } else {
                &spec_info
            },
        };
        let pipe_ci = sys::VkComputePipelineCreateInfo {
            s_type: sys::VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            stage: stage_ci,
            layout: layout.handle,
            base_pipeline_handle: 0,
            base_pipeline_index: -1,
        };
        let mut handle: sys::VkPipeline = 0;
        // SAFETY: resolved entry; live device; valid create-info; writable
        // out-parameter; pipeline cache handle is null (VK_NULL_HANDLE = 0).
        let r = unsafe {
            (device.fns.create_compute_pipelines)(
                device.device,
                0,
                1,
                &pipe_ci,
                ptr::null(),
                &mut handle,
            )
        };
        sys::check(r, "vkCreateComputePipelines")?;
        Ok(VulkanComputePipeline { device, handle })
    }

    #[must_use]
    pub(crate) fn handle(&self) -> sys::VkPipeline {
        self.handle
    }
}

impl Drop for VulkanComputePipeline<'_> {
    fn drop(&mut self) {
        if self.handle == 0 {
            return;
        }
        // SAFETY: resolved entry; live device + pipeline; destroyed once.
        unsafe { (self.device.fns.destroy_pipeline)(self.device.device, self.handle, ptr::null()) };
        self.handle = 0;
    }
}

// Suppress "unused import" for the placeholder `c_void` we pull in.
const _: *const c_void = core::ptr::null();

// ---------------------------------------------------------------------------
// smoke_dispatch_copy_f32_impl — the end-to-end `Vulkan is real` proof point
// (M3-02-T13 / ADR M3-02-spirv-generation §4 (d)). Uses the hand-crafted
// `copy_f32` SPIR-V blob to dispatch a copy of an f32 array on the GPU and
// reads the result back. Every T08〜T12 + T25 primitive is exercised by this
// single function: device, command pool, buffer / memory, descriptor set,
// pipeline layout, shader module, compute pipeline, fence — plus the three
// new dispatch commands (`vkCmdBindPipeline` / `vkCmdBindDescriptorSets` /
// `vkCmdDispatch`).
// ---------------------------------------------------------------------------

/// Round-trip `input` through the hand-crafted `copy_f32` compute kernel and
/// return the GPU-observed output. On a working Vulkan host this is
/// bit-identical to `input` — the shader body is `dst[i] = src[i]`.
///
/// The public wrapper is [`crate::smoke_dispatch_copy_f32`] (in `lib.rs`),
/// which surfaces [`VokraError::BackendUnavailable`] on non-Vulkan targets.
///
/// # Errors
///
/// - [`VokraError::BackendUnavailable`] — no Vulkan loader / no ICD / no
///   compute queue. The caller's test skips (no CPU fall back, FR-EX-08).
/// - [`VokraError::UnsupportedOp`] — the SPIR-V module was rejected by the
///   driver (VkResult != SUCCESS on `vkCreateShaderModule` /
///   `vkCreateComputePipelines`). Bubble up so the caller can log the driver
///   name.
pub(crate) fn smoke_dispatch_copy_f32_impl(input: &[f32]) -> Result<Vec<f32>> {
    // Trivial pass-through — no dispatch, no allocation.
    if input.is_empty() {
        return Ok(Vec::new());
    }

    // 1. Build the device (loader + instance + logical device + queue).
    let instance = VulkanInstance::new()?;
    let device = VulkanDevice::new(instance)?;

    // 2. Encode input as little-endian bytes and upload to a device-local
    //    SSBO. `upload_bytes` uses a host-visible staging buffer + a
    //    device-local target buffer with STORAGE_BUFFER | TRANSFER_DST usage,
    //    which is exactly what the shader wants.
    let byte_len = input.len() * 4;
    let mut src_bytes = Vec::with_capacity(byte_len);
    for f in input {
        src_bytes.extend_from_slice(&f.to_le_bytes());
    }
    let src_buf = device.upload_bytes(&src_bytes)?;

    // 3. Allocate the destination SSBO — device-local, STORAGE_BUFFER +
    //    TRANSFER_SRC so the compute pipeline can write to it and the
    //    subsequent `download_bytes` can read it back via a staging copy.
    let dst_buf = VulkanBuffer::new(
        &device,
        byte_len,
        sys::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | sys::VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sys::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    )?;

    // 4. Descriptor set layout + pool + set (2 storage-buffer bindings).
    let dsl = VulkanDescriptorSetLayout::new_storage_buffers(
        &device,
        crate::spirv::handcrafted_copy_f32::BINDING_COUNT,
    )?;
    let dpool = VulkanDescriptorPool::new_storage_buffers(
        &device,
        1,
        crate::spirv::handcrafted_copy_f32::BINDING_COUNT,
    )?;
    let set = dpool.allocate_set(&dsl)?;

    // Update the set: binding 0 = src, binding 1 = dst.
    let buffer_infos = [
        sys::VkDescriptorBufferInfo {
            buffer: src_buf.handle(),
            offset: 0,
            range: sys::VkDeviceSize::MAX, // VK_WHOLE_SIZE
        },
        sys::VkDescriptorBufferInfo {
            buffer: dst_buf.handle(),
            offset: 0,
            range: sys::VkDeviceSize::MAX,
        },
    ];
    let writes = [
        sys::VkWriteDescriptorSet {
            s_type: sys::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(),
            dst_set: set,
            dst_binding: 0,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: sys::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            p_image_info: ptr::null(),
            p_buffer_info: &buffer_infos[0],
            p_texel_buffer_view: ptr::null(),
        },
        sys::VkWriteDescriptorSet {
            s_type: sys::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(),
            dst_set: set,
            dst_binding: 1,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: sys::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            p_image_info: ptr::null(),
            p_buffer_info: &buffer_infos[1],
            p_texel_buffer_view: ptr::null(),
        },
    ];
    // SAFETY: resolved entry; live device; array of 2 valid VkWriteDescriptorSet
    // on the stack; buffer_infos outlive the call.
    unsafe {
        (device.fns.update_descriptor_sets)(
            device.device,
            writes.len() as u32,
            writes.as_ptr(),
            0,
            ptr::null(),
        );
    }

    // 5. Pipeline layout (T11) + shader module (T11) + compute pipeline (T11).
    let pl = VulkanPipelineLayout::new(&device, &dsl)?;
    let spv = crate::spirv::handcrafted_copy_f32::bytes();
    let shader = VulkanShaderModule::new(&device, &spv)?;
    // `entry_name` MUST be NUL-terminated; a `c""` literal is a
    // compile-time-validated `&'static CStr` matching the SPIR-V module's
    // OpEntryPoint name ("main").
    let entry_name = c"main";
    let pipeline = VulkanComputePipeline::new(&device, &pl, &shader, entry_name)?;

    // 6. Record the dispatch into a fresh primary command buffer.
    let cmd_pool = VulkanCommandPool::new(&device)?;
    let cmd = cmd_pool.command_buffer;

    let begin_info = sys::VkCommandBufferBeginInfo {
        s_type: sys::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        p_next: ptr::null(),
        flags: sys::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        p_inheritance_info: ptr::null(),
    };
    // SAFETY: resolved entry; live command buffer; valid begin-info.
    let r = unsafe { (device.fns.begin_command_buffer)(cmd, &begin_info) };
    sys::check(r, "vkBeginCommandBuffer (copy_f32 dispatch)")?;

    // SAFETY: resolved entry; live command buffer; live pipeline handle;
    // bind point = compute.
    unsafe {
        (device.fns.cmd_bind_pipeline)(cmd, sys::VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
    }
    let set_arr = [set];
    // SAFETY: resolved entry; live command buffer; live layout + set; array
    // of 1 descriptor set on the stack; no dynamic offsets.
    unsafe {
        (device.fns.cmd_bind_descriptor_sets)(
            cmd,
            sys::VK_PIPELINE_BIND_POINT_COMPUTE,
            pl.handle(),
            0, // firstSet
            1, // descriptorSetCount
            set_arr.as_ptr(),
            0,
            ptr::null(),
        );
    }
    let local_size_x = crate::spirv::handcrafted_copy_f32::LOCAL_SIZE_X as usize;
    // `group_count_x = ceil(N / local_size_x)` — the shader tolerates trailing
    // invocations that read/write out-of-range only because we round to an
    // exact multiple: for arbitrary N we would need a bounds check in the
    // shader. Assert the caller passed N that is a multiple of local_size_x
    // to keep the smoke kernel small (no bounds check in the SPIR-V body).
    assert!(
        input.len() % local_size_x == 0,
        "smoke_dispatch_copy_f32 requires input.len() ({}) to be a multiple of LOCAL_SIZE_X ({}) \
         to avoid out-of-bounds shader invocations (the handcrafted kernel omits a bounds \
         check by design; see kernels/handcrafted/copy_f32.spv.rs)",
        input.len(),
        local_size_x,
    );
    let group_x = input.len() / local_size_x;
    // SAFETY: resolved entry; live command buffer; positive group counts.
    unsafe {
        (device.fns.cmd_dispatch)(cmd, group_x as u32, 1, 1);
    }
    // SAFETY: resolved entry; live command buffer.
    let r = unsafe { (device.fns.end_command_buffer)(cmd) };
    sys::check(r, "vkEndCommandBuffer (copy_f32 dispatch)")?;

    // 7. Submit + wait for fence.
    let fence = VulkanFence::new(&device)?;
    let submit = sys::VkSubmitInfo {
        s_type: sys::VK_STRUCTURE_TYPE_SUBMIT_INFO,
        p_next: ptr::null(),
        wait_semaphore_count: 0,
        p_wait_semaphores: ptr::null(),
        p_wait_dst_stage_mask: ptr::null(),
        command_buffer_count: 1,
        p_command_buffers: &cmd,
        signal_semaphore_count: 0,
        p_signal_semaphores: ptr::null(),
    };
    // SAFETY: resolved entry; live queue; single submit-info on the stack;
    // fence handle is live.
    let r = unsafe { (device.fns.queue_submit)(device.queue, 1, &submit, fence.handle) };
    sys::check(r, "vkQueueSubmit (copy_f32 dispatch)")?;
    fence.wait()?;

    // Guarantee any transient handles are dropped before we start the
    // read-back submission. The pipeline/shader/pool objects reference the
    // device via `&VulkanDevice`, so they must be released before the device
    // itself goes out of scope; explicit drops keep the ordering clear.
    drop(pipeline);
    drop(shader);
    drop(pl);
    // `set` is a `u64` handle (Copy), not an RAII wrapper; the descriptor set
    // is implicitly freed when its parent pool (`dpool`) is dropped below.
    let _ = set;
    drop(dpool);
    drop(dsl);

    // 8. Read the dst buffer back to host memory and decode f32s.
    let mut out_bytes = vec![0u8; byte_len];
    device.download_bytes(&dst_buf, &mut out_bytes)?;
    let mut out = Vec::with_capacity(input.len());
    for chunk in out_bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// smoke_dispatch_add_f32_impl — the M3-02-T24 three-SSBO dispatch-chain proof
// point. Uses the hand-crafted `add_f32` SPIR-V blob to compute the
// element-wise sum `c[i] = a[i] + b[i]` on the GPU. Same structural pattern
// as `smoke_dispatch_copy_f32_impl` (device / buffer / descriptor set /
// pipeline / command buffer / fence / dispatch), but with THREE storage
// buffers bound to descriptor set 0 (a @ binding 0, b @ binding 1, c @
// binding 2). This is what the `OpKind::Add` arm of `eval::eval_vulkan_op`
// routes into (M3-02-T24 / T26).
// ---------------------------------------------------------------------------

/// Element-wise add `a[i] + b[i] → c[i]` through the hand-crafted `add_f32`
/// SPIR-V kernel and return the GPU-observed output. On a working Vulkan
/// host the output is `a + b` under IEEE-754 f32 semantics — which for the
/// smoke-test inputs (all pairs of finite floats we send) is bit-identical
/// to the host sum.
///
/// The public wrapper is [`crate::smoke_dispatch_add_f32`] (in `lib.rs`),
/// which surfaces [`VokraError::BackendUnavailable`] on non-Vulkan targets.
///
/// # Panics
///
/// Panics if `a.len() != b.len()` or if the length is not a multiple of
/// [`crate::spirv::handcrafted_add_f32::LOCAL_SIZE_X`] — the hand-crafted
/// shader has no bounds check by design (kept the bytecode small). Callers
/// must pad their inputs, exactly the same contract as `smoke_dispatch_copy_f32`.
///
/// # Errors
///
/// - [`VokraError::BackendUnavailable`] — no Vulkan loader / no ICD / no
///   compute queue. The caller's test skips (no CPU fall back, FR-EX-08).
/// - [`VokraError::UnsupportedOp`] — the SPIR-V module was rejected by the
///   driver (VkResult != SUCCESS on `vkCreateShaderModule` /
///   `vkCreateComputePipelines`).
pub(crate) fn smoke_dispatch_add_f32_impl(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    assert_eq!(
        a.len(),
        b.len(),
        "smoke_dispatch_add_f32 requires a.len() == b.len(); got {} and {}",
        a.len(),
        b.len()
    );
    // Trivial pass-through — no dispatch, no allocation.
    if a.is_empty() {
        return Ok(Vec::new());
    }

    // 1. Build the device (loader + instance + logical device + queue).
    let instance = VulkanInstance::new()?;
    let device = VulkanDevice::new(instance)?;

    // 2. Encode both inputs as little-endian bytes and upload to two
    //    device-local SSBOs (a @ binding 0, b @ binding 1).
    let byte_len = a.len() * 4;
    let mut a_bytes = Vec::with_capacity(byte_len);
    for f in a {
        a_bytes.extend_from_slice(&f.to_le_bytes());
    }
    let a_buf = device.upload_bytes(&a_bytes)?;

    let mut b_bytes = Vec::with_capacity(byte_len);
    for f in b {
        b_bytes.extend_from_slice(&f.to_le_bytes());
    }
    let b_buf = device.upload_bytes(&b_bytes)?;

    // 3. Allocate the output SSBO (c @ binding 2) — device-local,
    //    STORAGE_BUFFER + TRANSFER_SRC so the pipeline can write to it and
    //    the subsequent `download_bytes` can read it back via a staging
    //    copy.
    let c_buf = VulkanBuffer::new(
        &device,
        byte_len,
        sys::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | sys::VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sys::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    )?;

    // 4. Descriptor set layout + pool + set (3 storage-buffer bindings).
    //    This is what `add_f32` exercises differently from `copy_f32` — the
    //    3-SSBO layout the M3-02-T24 graph arm needs.
    let dsl = VulkanDescriptorSetLayout::new_storage_buffers(
        &device,
        crate::spirv::handcrafted_add_f32::BINDING_COUNT,
    )?;
    let dpool = VulkanDescriptorPool::new_storage_buffers(
        &device,
        1,
        crate::spirv::handcrafted_add_f32::BINDING_COUNT,
    )?;
    let set = dpool.allocate_set(&dsl)?;

    // Update the set: binding 0 = a, binding 1 = b, binding 2 = c.
    let buffer_infos = [
        sys::VkDescriptorBufferInfo {
            buffer: a_buf.handle(),
            offset: 0,
            range: sys::VkDeviceSize::MAX, // VK_WHOLE_SIZE
        },
        sys::VkDescriptorBufferInfo {
            buffer: b_buf.handle(),
            offset: 0,
            range: sys::VkDeviceSize::MAX,
        },
        sys::VkDescriptorBufferInfo {
            buffer: c_buf.handle(),
            offset: 0,
            range: sys::VkDeviceSize::MAX,
        },
    ];
    let writes = [
        sys::VkWriteDescriptorSet {
            s_type: sys::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(),
            dst_set: set,
            dst_binding: 0,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: sys::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            p_image_info: ptr::null(),
            p_buffer_info: &buffer_infos[0],
            p_texel_buffer_view: ptr::null(),
        },
        sys::VkWriteDescriptorSet {
            s_type: sys::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(),
            dst_set: set,
            dst_binding: 1,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: sys::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            p_image_info: ptr::null(),
            p_buffer_info: &buffer_infos[1],
            p_texel_buffer_view: ptr::null(),
        },
        sys::VkWriteDescriptorSet {
            s_type: sys::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(),
            dst_set: set,
            dst_binding: 2,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: sys::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            p_image_info: ptr::null(),
            p_buffer_info: &buffer_infos[2],
            p_texel_buffer_view: ptr::null(),
        },
    ];
    // SAFETY: resolved entry; live device; array of 3 valid VkWriteDescriptorSet
    // on the stack; buffer_infos outlive the call.
    unsafe {
        (device.fns.update_descriptor_sets)(
            device.device,
            writes.len() as u32,
            writes.as_ptr(),
            0,
            ptr::null(),
        );
    }

    // 5. Pipeline layout + shader module + compute pipeline.
    let pl = VulkanPipelineLayout::new(&device, &dsl)?;
    let spv = crate::spirv::handcrafted_add_f32::bytes();
    let shader = VulkanShaderModule::new(&device, &spv)?;
    // `entry_name` MUST be NUL-terminated; a `c""` literal is a
    // compile-time-validated `&'static CStr` matching the SPIR-V module's
    // OpEntryPoint name ("main").
    let entry_name = c"main";
    let pipeline = VulkanComputePipeline::new(&device, &pl, &shader, entry_name)?;

    // 6. Record the dispatch into a fresh primary command buffer.
    let cmd_pool = VulkanCommandPool::new(&device)?;
    let cmd = cmd_pool.command_buffer;

    let begin_info = sys::VkCommandBufferBeginInfo {
        s_type: sys::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        p_next: ptr::null(),
        flags: sys::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        p_inheritance_info: ptr::null(),
    };
    // SAFETY: resolved entry; live command buffer; valid begin-info.
    let r = unsafe { (device.fns.begin_command_buffer)(cmd, &begin_info) };
    sys::check(r, "vkBeginCommandBuffer (add_f32 dispatch)")?;

    // SAFETY: resolved entry; live command buffer; live pipeline handle;
    // bind point = compute.
    unsafe {
        (device.fns.cmd_bind_pipeline)(cmd, sys::VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
    }
    let set_arr = [set];
    // SAFETY: resolved entry; live command buffer; live layout + set; array
    // of 1 descriptor set on the stack; no dynamic offsets.
    unsafe {
        (device.fns.cmd_bind_descriptor_sets)(
            cmd,
            sys::VK_PIPELINE_BIND_POINT_COMPUTE,
            pl.handle(),
            0, // firstSet
            1, // descriptorSetCount
            set_arr.as_ptr(),
            0,
            ptr::null(),
        );
    }
    let local_size_x = crate::spirv::handcrafted_add_f32::LOCAL_SIZE_X as usize;
    assert!(
        a.len() % local_size_x == 0,
        "smoke_dispatch_add_f32 requires input.len() ({}) to be a multiple of LOCAL_SIZE_X ({}) \
         to avoid out-of-bounds shader invocations (the handcrafted kernel omits a bounds \
         check by design; see kernels/handcrafted/add_f32.spv.rs)",
        a.len(),
        local_size_x,
    );
    let group_x = a.len() / local_size_x;
    // SAFETY: resolved entry; live command buffer; positive group counts.
    unsafe {
        (device.fns.cmd_dispatch)(cmd, group_x as u32, 1, 1);
    }
    // SAFETY: resolved entry; live command buffer.
    let r = unsafe { (device.fns.end_command_buffer)(cmd) };
    sys::check(r, "vkEndCommandBuffer (add_f32 dispatch)")?;

    // 7. Submit + wait for fence.
    let fence = VulkanFence::new(&device)?;
    let submit = sys::VkSubmitInfo {
        s_type: sys::VK_STRUCTURE_TYPE_SUBMIT_INFO,
        p_next: ptr::null(),
        wait_semaphore_count: 0,
        p_wait_semaphores: ptr::null(),
        p_wait_dst_stage_mask: ptr::null(),
        command_buffer_count: 1,
        p_command_buffers: &cmd,
        signal_semaphore_count: 0,
        p_signal_semaphores: ptr::null(),
    };
    // SAFETY: resolved entry; live queue; single submit-info on the stack;
    // fence handle is live.
    let r = unsafe { (device.fns.queue_submit)(device.queue, 1, &submit, fence.handle) };
    sys::check(r, "vkQueueSubmit (add_f32 dispatch)")?;
    fence.wait()?;

    // Drop transient handles in dependency order before the device goes out
    // of scope. Same ordering rationale as `smoke_dispatch_copy_f32_impl`.
    drop(pipeline);
    drop(shader);
    drop(pl);
    let _ = set;
    drop(dpool);
    drop(dsl);

    // 8. Read the c buffer back to host memory and decode f32s.
    let mut out_bytes = vec![0u8; byte_len];
    device.download_bytes(&c_buf, &mut out_bytes)?;
    let mut out = Vec::with_capacity(a.len());
    for chunk in out_bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// dispatch_kernel — the M4-13-T02 generic SPIR-V kernel dispatch helper.
//
// Generalises the hand-crafted `smoke_dispatch_{copy,add}_f32_impl` pattern
// (device / buffers / descriptor set / pipeline / command buffer / fence)
// to ANY manifest kernel: N read-only input SSBOs + 1 writable output SSBO
// (always the LAST binding — the convention every `kernels/glsl/*.comp`
// skeleton follows), an optional push-constant block, optional u32
// specialization constants, and an explicit [x, y, z] workgroup count.
//
// Placeholder-then-swap seam (FR-EX-08): the blob is fetched through
// `spirv::load_spv_owned`; a `None` (glslc `.spv` not yet committed —
// M4-13-T16 owner task) surfaces as the explicit `UnsupportedOp` that
// `spirv::require_blob` formats. Never a silent CPU fall back.
//
// Hot-path allocation note (FR-EX-05): this helper allocates its staging /
// device buffers, descriptor pool and command pool per dispatch — the same
// posture as the M3-02-T25 smoke path. A session-owned pre-allocated pool
// (upload once, reuse across steps — the Metal/CUDA decode-session pattern)
// is the M4 kernel-fusion follow-up; per-dispatch allocation is acceptable
// for the parity-first M4-13 scope and is NOT on a model hot path yet.
//
// Synchronisation: upload / dispatch / download are separate fence-waited
// submissions to the same queue — the exact structure the hand-crafted smoke
// kernels prove out on lavapipe (each submission fully completes before the
// next is recorded, so no intra-command-buffer barriers are needed).
// ---------------------------------------------------------------------------

// The specialization-constant descriptor lives in the host-portable `plan`
// module (M4-13-T02) so plans built off-target carry the same type this
// gated dispatch side consumes.
pub(crate) use crate::plan::SpecConstantU32;

/// A fully-described generic kernel dispatch (M4-13-T02): which manifest
/// kernel to run, its SSBO inputs, output size, push constants,
/// specialization constants and workgroup counts.
pub(crate) struct KernelInvocation<'a> {
    /// Manifest kernel name (`spirv::SHADERS` entry).
    pub(crate) name: &'a str,
    /// Read-only input SSBOs, bound at `binding = 0..inputs.len()` in order.
    pub(crate) inputs: &'a [&'a [u8]],
    /// Byte length of the writable output SSBO, bound at
    /// `binding = inputs.len()` (the last binding).
    pub(crate) output_byte_len: usize,
    /// Raw push-constant block (LE-packed scalars; may be empty). Must be a
    /// multiple of 4 bytes and at most 128 bytes (spec §42.1 minimum).
    pub(crate) push_constants: &'a [u8],
    /// Pipeline specialization constants (may be empty).
    pub(crate) spec_constants: &'a [SpecConstantU32],
    /// `vkCmdDispatch` group counts `[x, y, z]` — all three must be in
    /// `1..=65535` (spec §42.1 guaranteed minimum for
    /// `maxComputeWorkGroupCount`).
    pub(crate) workgroups: [u32; 3],
}

/// Dispatch one generic compute kernel and read its output SSBO back to host
/// memory as raw little-endian bytes (M4-13-T02).
///
/// # Errors
///
/// - [`VokraError::UnsupportedOp`] — the kernel's `.spv` blob has not landed
///   yet (`spirv::require_blob` seam; owner M4-13-T16 lights it up), or the
///   driver rejected the SPIR-V module.
/// - [`VokraError::InvalidArgument`] — empty input / output, malformed
///   push-constant block, or out-of-range workgroup counts. All validated
///   BEFORE any GPU work.
/// - [`VokraError::BackendUnavailable`] — driver-side failures (bubbled from
///   the object-stack constructors).
pub(crate) fn dispatch_kernel(
    device: &VulkanDevice,
    inv: &KernelInvocation<'_>,
) -> Result<Vec<u8>> {
    // ---- host-side validation, before any GPU object is created ----
    if inv.output_byte_len == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "dispatch_kernel({}): output_byte_len must be > 0",
            inv.name
        )));
    }
    for (i, input) in inv.inputs.iter().enumerate() {
        if input.is_empty() {
            return Err(VokraError::InvalidArgument(format!(
                "dispatch_kernel({}): input SSBO #{i} is empty (bind a 4-byte dummy for \
                 optional buffers instead — Vulkan requires every declared binding bound)",
                inv.name
            )));
        }
    }
    if inv.push_constants.len() % 4 != 0 || inv.push_constants.len() > 128 {
        return Err(VokraError::InvalidArgument(format!(
            "dispatch_kernel({}): push-constant block of {} bytes must be a multiple of 4 and \
             <= 128 (spec §42.1 guaranteed minimum)",
            inv.name,
            inv.push_constants.len()
        )));
    }
    for (axis, &g) in ["x", "y", "z"].iter().zip(&inv.workgroups) {
        if g == 0 || g > 65_535 {
            return Err(VokraError::InvalidArgument(format!(
                "dispatch_kernel({}): workgroup count {axis}={g} out of the portable range \
                 1..=65535 (spec §42.1 guaranteed minimum for maxComputeWorkGroupCount)",
                inv.name
            )));
        }
    }

    // ---- placeholder-then-swap seam: blob or explicit UnsupportedOp ----
    crate::spirv::require_blob(inv.name)?;
    let spv = crate::spirv::load_spv_owned(inv.name)
        .expect("require_blob passed; load_spv_owned must return the same blob");

    // ---- input upload + output allocation ----
    let mut in_bufs: Vec<VulkanBuffer<'_>> = Vec::with_capacity(inv.inputs.len());
    for input in inv.inputs {
        in_bufs.push(device.upload_bytes(input)?);
    }
    let out_buf = VulkanBuffer::new(
        device,
        inv.output_byte_len,
        sys::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | sys::VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sys::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    )?;

    // ---- descriptor set: inputs at binding 0.., output at the last binding ----
    let binding_count = (inv.inputs.len() + 1) as u32;
    let dsl = VulkanDescriptorSetLayout::new_storage_buffers(device, binding_count)?;
    let dpool = VulkanDescriptorPool::new_storage_buffers(device, 1, binding_count)?;
    let set = dpool.allocate_set(&dsl)?;

    // Collect ALL buffer infos first so their addresses stay stable while the
    // write array below points at them.
    let mut buffer_infos: Vec<sys::VkDescriptorBufferInfo> =
        Vec::with_capacity(binding_count as usize);
    for buf in &in_bufs {
        buffer_infos.push(sys::VkDescriptorBufferInfo {
            buffer: buf.handle(),
            offset: 0,
            range: sys::VkDeviceSize::MAX, // VK_WHOLE_SIZE
        });
    }
    buffer_infos.push(sys::VkDescriptorBufferInfo {
        buffer: out_buf.handle(),
        offset: 0,
        range: sys::VkDeviceSize::MAX,
    });
    let writes: Vec<sys::VkWriteDescriptorSet> = buffer_infos
        .iter()
        .enumerate()
        .map(|(i, info)| sys::VkWriteDescriptorSet {
            s_type: sys::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(),
            dst_set: set,
            dst_binding: i as u32,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: sys::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            p_image_info: ptr::null(),
            p_buffer_info: info,
            p_texel_buffer_view: ptr::null(),
        })
        .collect();
    // SAFETY: resolved entry; live device; `writes` is a valid array of
    // initialised VkWriteDescriptorSet whose buffer-info pointers reference
    // `buffer_infos`, which outlives the call.
    unsafe {
        (device.fns.update_descriptor_sets)(
            device.device,
            writes.len() as u32,
            writes.as_ptr(),
            0,
            ptr::null(),
        );
    }

    // ---- pipeline: layout (+ push range) + module + (specialised) pipeline ----
    let pl = VulkanPipelineLayout::new_with_push_constants(
        device,
        &dsl,
        inv.push_constants.len() as u32,
    )?;
    let shader = VulkanShaderModule::new(device, &spv)?;
    // Every Vokra kernel's OpEntryPoint is named "main" (glslc default; the
    // hand-crafted modules mirror it).
    let entry_name = c"main";
    let pipeline = VulkanComputePipeline::new_specialized(
        device,
        &pl,
        &shader,
        entry_name,
        inv.spec_constants,
    )?;

    // ---- record + submit + fence-wait ----
    let cmd_pool = VulkanCommandPool::new(device)?;
    let cmd = cmd_pool.command_buffer;
    let begin_info = sys::VkCommandBufferBeginInfo {
        s_type: sys::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        p_next: ptr::null(),
        flags: sys::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        p_inheritance_info: ptr::null(),
    };
    // SAFETY: resolved entry; live command buffer; valid begin-info.
    let r = unsafe { (device.fns.begin_command_buffer)(cmd, &begin_info) };
    sys::check(r, "vkBeginCommandBuffer (dispatch_kernel)")?;
    // SAFETY: resolved entry; live command buffer + pipeline; compute bind point.
    unsafe {
        (device.fns.cmd_bind_pipeline)(cmd, sys::VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
    }
    let set_arr = [set];
    // SAFETY: resolved entry; live command buffer / layout / set; single-set
    // array on the stack; no dynamic offsets.
    unsafe {
        (device.fns.cmd_bind_descriptor_sets)(
            cmd,
            sys::VK_PIPELINE_BIND_POINT_COMPUTE,
            pl.handle(),
            0,
            1,
            set_arr.as_ptr(),
            0,
            ptr::null(),
        );
    }
    if !inv.push_constants.is_empty() {
        // SAFETY: resolved entry; live command buffer + layout; the byte size
        // matches the range declared on the pipeline layout above; the data
        // slice outlives the call.
        unsafe {
            (device.fns.cmd_push_constants)(
                cmd,
                pl.handle(),
                sys::VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                inv.push_constants.len() as u32,
                inv.push_constants.as_ptr() as *const c_void,
            );
        }
    }
    // SAFETY: resolved entry; live command buffer; group counts validated
    // into 1..=65535 above.
    unsafe {
        (device.fns.cmd_dispatch)(cmd, inv.workgroups[0], inv.workgroups[1], inv.workgroups[2]);
    }
    // SAFETY: resolved entry; live command buffer.
    let r = unsafe { (device.fns.end_command_buffer)(cmd) };
    sys::check(r, "vkEndCommandBuffer (dispatch_kernel)")?;

    let fence = VulkanFence::new(device)?;
    let submit = sys::VkSubmitInfo {
        s_type: sys::VK_STRUCTURE_TYPE_SUBMIT_INFO,
        p_next: ptr::null(),
        wait_semaphore_count: 0,
        p_wait_semaphores: ptr::null(),
        p_wait_dst_stage_mask: ptr::null(),
        command_buffer_count: 1,
        p_command_buffers: &cmd,
        signal_semaphore_count: 0,
        p_signal_semaphores: ptr::null(),
    };
    // SAFETY: resolved entry; live queue; single submit-info on the stack;
    // live fence handle.
    let r = unsafe { (device.fns.queue_submit)(device.queue, 1, &submit, fence.handle) };
    sys::check(r, "vkQueueSubmit (dispatch_kernel)")?;
    fence.wait()?;

    // Release transients in dependency order before the read-back submission
    // (same rationale as the smoke impls).
    drop(pipeline);
    drop(shader);
    drop(pl);
    let _ = set;
    drop(dpool);
    drop(dsl);
    drop(in_bufs);

    // ---- read back the output SSBO ----
    let mut out_bytes = vec![0u8; inv.output_byte_len];
    device.download_bytes(&out_buf, &mut out_bytes)?;
    Ok(out_bytes)
}

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
                    "VulkanInstance::new must return BackendUnavailable off a Vulkan host, got \
                     {other}"
                );
            }
        }
    }

    /// Device create + destroy round-trips cleanly on a Vulkan host, and is
    /// an explicit `BackendUnavailable` off Vulkan.
    #[test]
    fn device_new_is_honest() {
        let Ok(instance) = VulkanInstance::new() else {
            eprintln!("no Vulkan loader; skipping device create test");
            return;
        };
        match VulkanDevice::new(instance) {
            Ok(dev) => {
                assert!(!dev.device.is_null());
                assert!(!dev.queue.is_null());
                // Extensions enumeration must succeed on a live device.
                let exts = dev
                    .instance()
                    .enumerate_device_extensions(dev.physical_device())
                    .expect("device extensions enumeration must succeed on a live device");
                eprintln!("device reports {} extensions", exts.len());
            }
            Err(VokraError::BackendUnavailable(msg)) => {
                eprintln!("vulkan device unavailable (probably no ICD): {msg}");
            }
            Err(other) => panic!("unexpected error from device create: {other}"),
        }
    }

    #[test]
    fn command_pool_new_is_honest() {
        let Ok(instance) = VulkanInstance::new() else {
            return;
        };
        let Ok(device) = VulkanDevice::new(instance) else {
            return;
        };
        let pool = VulkanCommandPool::new(&device).expect("command pool create should succeed");
        assert!(!pool.command_buffer.is_null());
    }

    #[test]
    fn buffer_upload_download_round_trip_matches() {
        let Ok(instance) = VulkanInstance::new() else {
            eprintln!("no Vulkan loader; skipping buffer round-trip test");
            return;
        };
        let device = match VulkanDevice::new(instance) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("no Vulkan device; skipping buffer round-trip test");
                return;
            }
        };
        // Bulk of the T25 round-trip: staging → device-local → staging → host.
        let src: Vec<u8> = (0..4096u32).map(|i| (i * 3 + 7) as u8).collect();
        // Upload_bytes needs the target to have TRANSFER_SRC so download works.
        let buffer = VulkanBuffer::new(
            &device,
            src.len(),
            sys::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | sys::VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | sys::VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            sys::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )
        .expect("device-local buffer create should succeed");
        // Push data via a staging buffer + a single-shot copy.
        let mut staging = VulkanBuffer::new(
            &device,
            src.len(),
            sys::VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            sys::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | sys::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        .expect("staging buffer create should succeed");
        staging.write_bytes(&src).expect("write_bytes must succeed");
        device
            .copy_buffer(&staging, &buffer, src.len())
            .expect("host→device copy must succeed");
        drop(staging);
        // Download back to host via download_bytes.
        let mut got = vec![0u8; src.len()];
        device
            .download_bytes(&buffer, &mut got)
            .expect("download_bytes must succeed");
        assert_eq!(got, src, "round-trip must be bit-identical");
        // Ensure the buffer's usage field survived construction round-trip
        // (a light sanity check for the wrapper's book-keeping).
        assert_eq!(
            buffer.usage & sys::VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            sys::VK_BUFFER_USAGE_TRANSFER_SRC_BIT
        );
        // Explicit drop to sequence resource teardown before the device.
        drop(buffer);
    }

    #[test]
    fn descriptor_layout_and_pool_round_trip() {
        let Ok(instance) = VulkanInstance::new() else {
            return;
        };
        let Ok(device) = VulkanDevice::new(instance) else {
            return;
        };
        let layout = VulkanDescriptorSetLayout::new_storage_buffers(&device, 3)
            .expect("descriptor set layout create should succeed");
        let pool = VulkanDescriptorPool::new_storage_buffers(&device, 1, 3)
            .expect("descriptor pool create should succeed");
        let _set = pool
            .allocate_set(&layout)
            .expect("descriptor set alloc should succeed");
        // Also exercise the pipeline layout wrapping this.
        let pl = VulkanPipelineLayout::new(&device, &layout)
            .expect("pipeline layout create should succeed");
        assert_ne!(pl.handle(), 0);
    }

    #[test]
    fn shader_module_rejects_empty_spirv() {
        let Ok(instance) = VulkanInstance::new() else {
            return;
        };
        let Ok(device) = VulkanDevice::new(instance) else {
            return;
        };
        assert!(matches!(
            VulkanShaderModule::new(&device, &[]),
            Err(VokraError::UnsupportedOp(_))
        ));
    }
}
