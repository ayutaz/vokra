//! Raw extern-import shim for the browser WebGPU API (M4-01-T09).
//!
//! The WASM analogue of the Metal / CUDA / Vulkan raw-FFI layers: instead of
//! dlopen + `dlsym`, a wasm module declares **imports** that the embedder's
//! import object satisfies at instantiate time — the runtime-linking step.
//! The hand-written JS glue (`crates/vokra-backend-webgpu/glue/vokra_webgpu.js`)
//! implements every import below against `navigator.gpu`; a no-GPU host
//! (Node, non-WebGPU browsers) provides the same import surface in
//! "unavailable" mode where [`vokra_webgpu_probe`] reports 0 and every op
//! fails with a readable message — the exact dlopen-failure analogue, mapped
//! to [`VokraError::BackendUnavailable`](vokra_core::VokraError::BackendUnavailable)
//! by `probe.rs` (FR-EX-08: explicit error, never a silent CPU fall back).
//!
//! # Import contract (the `// SAFETY:` basis for every call site)
//!
//! - module name: `"vokra_webgpu"`; the glue guarantees every import exists
//!   (instantiation would otherwise fail — there is no partially-linked
//!   state).
//! - handles are opaque `u32` ids into glue-side tables (`0` is never a
//!   valid handle; creation returns `0` on failure and stores a message
//!   readable via [`vokra_webgpu_error_len`] / [`vokra_webgpu_error_read`]).
//! - pointer + length pairs refer to the module's own linear memory; the
//!   glue reads/writes through the exported `memory` at those offsets. The
//!   Rust caller must pass in-bounds, live allocations (ordinary slice
//!   borrows suffice — wasm linear memory addresses ARE the pointer values).
//! - negative `i32` returns signal failure; `0` success.
//! - async WebGPU steps (`requestAdapter` / `requestDevice` at init,
//!   `mapAsync` inside `vokra_webgpu_buffer_read`) are resolved by the glue
//!   through the Worker + SharedArrayBuffer + `Atomics.wait` bridge (ADR
//!   M4-01 §3) — from the wasm side every import is synchronous.
//!
//! Individual import names and shapes map 1:1 onto the `navigator.gpu`
//! surface the glue drives: `buffer_create` → `device.createBuffer`,
//! `buffer_write` → `queue.writeBuffer`, `buffer_read` →
//! `copyBufferToBuffer` + `mapAsync` (proxied), `shader_create` →
//! `device.createShaderModule`, `pipeline_create` →
//! `device.createComputePipeline`, `dispatch` → bind group + command encoder
//! + `dispatchWorkgroups` + `submit`.

/// Buffer usage: kernel input (STORAGE | COPY_DST on the glue side).
pub const USAGE_STORAGE_INPUT: u32 = 0;
/// Buffer usage: kernel output (STORAGE | COPY_SRC | COPY_DST).
pub const USAGE_STORAGE_OUTPUT: u32 = 1;

#[link(wasm_import_module = "vokra_webgpu")]
unsafe extern "C" {
    /// Probe the adapter/device the glue initialised: `1` = ready, `0` = no
    /// WebGPU adapter (`navigator.gpu` absent or `requestAdapter` returned
    /// null), negative = init error (message via the error imports).
    pub fn vokra_webgpu_probe() -> i32;

    /// Byte length of the glue's last error message (UTF-8; 0 = none).
    pub fn vokra_webgpu_error_len() -> u32;

    /// Copies up to `cap` bytes of the last error message into `dst`;
    /// returns the copied length.
    pub fn vokra_webgpu_error_read(dst: *mut u8, cap: u32) -> u32;

    /// Creates a GPU buffer of `size` bytes with the [`USAGE_STORAGE_INPUT`]
    /// / [`USAGE_STORAGE_OUTPUT`] usage class. Returns a non-zero handle, or
    /// `0` on failure.
    pub fn vokra_webgpu_buffer_create(size: u32, usage: u32) -> u32;

    /// Writes `len` bytes from linear memory at `src` into `buf` at
    /// `offset` (`queue.writeBuffer`). Returns 0 on success.
    pub fn vokra_webgpu_buffer_write(buf: u32, offset: u32, src: *const u8, len: u32) -> i32;

    /// Reads `len` bytes from `buf` at `offset` into linear memory at `dst`.
    /// This is the **synchronous readback bridge**: the glue proxies
    /// `copyBufferToBuffer` + `mapAsync` through the GPU-proxy context and
    /// blocks the calling worker on `Atomics.wait` until the copy lands
    /// (ADR M4-01 §3). Returns 0 on success. Call at run boundaries only —
    /// the M2-01 readback-consolidation lesson.
    pub fn vokra_webgpu_buffer_read(buf: u32, offset: u32, dst: *mut u8, len: u32) -> i32;

    /// Destroys a buffer handle (idempotent; unknown handles are ignored).
    pub fn vokra_webgpu_buffer_destroy(buf: u32);

    /// Creates (or returns the cached) shader module compiled from the WGSL
    /// `source` (`device.createShaderModule` — compilation is the browser /
    /// driver's responsibility, NFR-RL-05). `name` keys the glue-side cache.
    /// Returns a non-zero handle, or `0` on failure.
    pub fn vokra_webgpu_shader_create(
        name_ptr: *const u8,
        name_len: u32,
        src_ptr: *const u8,
        src_len: u32,
    ) -> u32;

    /// Creates a compute pipeline for `shader`'s `entry` function
    /// (`device.createComputePipeline`, `layout: "auto"`). Returns a
    /// non-zero handle, or `0` on failure.
    pub fn vokra_webgpu_pipeline_create(shader: u32, entry_ptr: *const u8, entry_len: u32) -> u32;

    /// Binds `bufs_len` storage buffers (ids at `bufs_ptr`, bind indices
    /// `0..bufs_len` in order) plus — when `uniform_len > 0` — a uniform
    /// buffer at bind index `bufs_len` filled from linear memory, then
    /// encodes `dispatchWorkgroups(wg_x, wg_y, wg_z)` and submits. Returns
    /// 0 on success.
    pub fn vokra_webgpu_dispatch(
        pipeline: u32,
        bufs_ptr: *const u32,
        bufs_len: u32,
        uniform_ptr: *const u8,
        uniform_len: u32,
        wg_x: u32,
        wg_y: u32,
        wg_z: u32,
    ) -> i32;

    /// Monotonic milliseconds from the embedder (`performance.now()`) — the
    /// wasm32 stand-in for `std::time::Instant` (which unconditionally
    /// panics on wasm32-unknown-unknown). RTF measurement hook (M4-01-T24).
    pub fn vokra_webgpu_now_ms() -> f64;
}

/// Fetches the glue's last error message (empty string when none).
pub(crate) fn last_glue_error() -> String {
    // SAFETY: import contract above — the glue implements both imports; the
    // buffer is a live, writable Vec allocation of exactly `len` bytes in
    // this module's linear memory.
    unsafe {
        let len = vokra_webgpu_error_len();
        if len == 0 {
            return String::new();
        }
        let mut buf = vec![0u8; len as usize];
        let copied = vokra_webgpu_error_read(buf.as_mut_ptr(), len);
        buf.truncate(copied as usize);
        String::from_utf8_lossy(&buf).into_owned()
    }
}
