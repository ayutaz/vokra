//! Raw CUDA Driver API + NVRTC FFI, loaded at runtime with dlopen / LoadLibrary
//! (Unix / Windows only — gated by the parent module).
//!
//! This module is the **only** place that talks to the NVIDIA driver, and it
//! does so with hand-declared `unsafe extern` blocks + runtime dynamic loading —
//! **no `cudarc` / `cust` / `rustacuda` binding crate** (M2-03 red line; keeps
//! the root `Cargo.lock` free of non-`vokra-*` crates, NFR-DS-02).
//!
//! # Why dlopen instead of `#[link(name = "cuda")]`
//!
//! Two reasons, both load-bearing:
//!
//! 1. **NVIDIA EULA install model** (FR-BE-08, `third_party/NVIDIA-EULA.md`):
//!    Vokra must not bundle or statically link `cudart` / `cudnn` / `cublas` /
//!    the driver. The developer installs the CUDA driver system-wide; Vokra
//!    detects it at runtime via `dlopen("libcuda.so.1")` /
//!    `LoadLibrary("nvcuda.dll")`.
//! 2. **All-target build** (NFR-PT-01): a link-time dependency on `libcuda`
//!    would break `cargo build` on any host without the CUDA driver — including
//!    the Apple Mac this slice is authored on. Runtime loading means the crate
//!    **compiles everywhere** and only *fails at runtime* (an explicit
//!    [`VokraError::BackendUnavailable`], never a silent CPU fall back —
//!    NFR-RL-06) where no NVIDIA GPU/driver exists.
//!
//! The dynamic-loader primitives themselves (`dlopen`/`dlsym`/`dlclose` on Unix;
//! `LoadLibraryA`/`GetProcAddress`/`FreeLibrary` on Windows) are symbols `std`
//! already links (libdl / kernel32), declared inline here — the same technique
//! `vokra-mmap` uses for POSIX `mmap` / Win32 file mapping, so nothing is added
//! to `Cargo.lock`.
//!
//! # Signature fidelity
//!
//! Every function pointer is `dlsym`'d and `transmute`d to the **exact** C
//! signature of its symbol, with a `// SAFETY:` note. The memory-management and
//! stream/context entry points use the versioned `_v2` symbol names
//! (`cuCtxCreate_v2`, `cuMemAlloc_v2`, …) that the CUDA headers `#define` the
//! bare names to on 64-bit platforms — that is what a modern `libcuda` actually
//! exports.

use core::ffi::{c_char, c_int, c_uint, c_void};

use vokra_core::{Result, VokraError};

// ---------------------------------------------------------------------------
// Platform dynamic-loader primitives (dlopen / LoadLibrary). No binding crate.
// ---------------------------------------------------------------------------

#[cfg(unix)]
mod dl {
    use core::ffi::{c_char, c_int, c_void};

    /// `RTLD_NOW` — resolve all undefined symbols before `dlopen` returns.
    /// Value `0x2` on both Linux (glibc/musl) and macOS/BSD.
    const RTLD_NOW: c_int = 2;

    // `std` links libc (and libdl on the Unix targets Vokra ships: macOS / iOS /
    // Linux / Android), which export these symbols.
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
    // Win32 calling convention. `GetProcAddress` really returns `FARPROC` (a
    // function pointer); declaring it `-> *mut c_void` is ABI-compatible (a
    // pointer-sized return) and lets the shared loader code stay uniform.
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

/// A loaded shared library (`libcuda` / `libnvrtc`). Closes its handle on drop.
///
/// Not `Send`/`Sync`: the driver handles derived from it (context / stream /
/// module) are used from the thread that created them (sufficient for the
/// parity harness; a `Send` wrapper is a later concern, mirroring
/// `MetalContext`).
pub(crate) struct DynLib {
    handle: *mut c_void,
}

impl DynLib {
    /// Tries each NUL-terminated candidate name in order; returns the first
    /// library that loads, or `None` if none do (e.g. on a host with no NVIDIA
    /// driver — the Apple Mac case).
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
    /// # Errors
    /// [`VokraError::BackendUnavailable`] if the symbol is absent (a driver too
    /// old to export it — treated as an incompatible/unavailable backend, never
    /// a silent fall back, NFR-RL-06).
    ///
    /// # Safety
    /// `F` must be a function-pointer type whose signature matches the C symbol
    /// `name` exactly. (Enforced by the single call site [`CudaDriver::load`] /
    /// [`Nvrtc::load`], which pairs each symbol with its precise `Fn*` alias.)
    pub(crate) unsafe fn get<F: Copy>(&self, name: &[u8]) -> Result<F> {
        debug_assert_eq!(name.last(), Some(&0), "symbol name must be NUL-terminated");
        debug_assert_eq!(
            core::mem::size_of::<F>(),
            core::mem::size_of::<*mut c_void>(),
            "F must be a pointer-sized function pointer"
        );
        // SAFETY: `handle` is live; `name` is a valid NUL-terminated C string.
        let ptr = unsafe { dl::sym(self.handle, name.as_ptr() as *const c_char) };
        if ptr.is_null() {
            let sym = String::from_utf8_lossy(&name[..name.len().saturating_sub(1)]);
            return Err(VokraError::BackendUnavailable(format!(
                "CUDA/NVRTC symbol `{sym}` not found in the loaded library (driver/NVRTC too old?)"
            )));
        }
        // SAFETY: `ptr` is a non-null symbol address (pointer-sized). `F` is a
        // function-pointer type of the same size (asserted above) whose C
        // signature the caller guarantees matches `name`. `transmute_copy` reads
        // exactly `size_of::<F>()` bytes from `&ptr`, reinterpreting the address
        // as the typed function pointer (the standard dlsym idiom).
        Ok(unsafe { core::mem::transmute_copy::<*mut c_void, F>(&ptr) })
    }

    /// [`Self::get`] variant for **optional** symbols (M4-07-T04): returns
    /// `None` instead of an error when the symbol is absent. Used for the
    /// Hopper-era Driver API entry points (`cuFuncSetAttribute`,
    /// `cuTensorMapEncodeTiled`) that the FA v3 path needs but the FA v2 /
    /// decomposed paths do not — a missing optional symbol must **never** fail
    /// the whole backend load; it only degrades FA v3 (explicit stderr log at
    /// the degrade site, FR-EX-08 — the log lives with the capability
    /// decision, not here).
    ///
    /// # Safety
    /// Same contract as [`Self::get`]: `F` must be a function-pointer type
    /// whose signature matches the C symbol `name` exactly.
    pub(crate) unsafe fn get_opt<F: Copy>(&self, name: &[u8]) -> Option<F> {
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
        // SAFETY: as in [`Self::get`] — non-null pointer-sized symbol address
        // reinterpreted as the typed function pointer the caller vouched for.
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
// CUDA Driver API types + constants.
// ---------------------------------------------------------------------------

/// `CUresult` — driver API status code. `0` (`CUDA_SUCCESS`) is success.
pub(crate) type CUresult = c_int;
/// `CUdevice` — an ordinal device handle (`typedef int CUdevice`).
pub(crate) type CUdevice = c_int;
/// `CUcontext` — an opaque driver context handle.
pub(crate) type CUcontext = *mut c_void;
/// `CUmodule` — an opaque loaded-module (PTX/cubin) handle.
pub(crate) type CUmodule = *mut c_void;
/// `CUfunction` — an opaque kernel function handle.
pub(crate) type CUfunction = *mut c_void;
/// `CUstream` — an opaque stream handle.
pub(crate) type CUstream = *mut c_void;
/// `CUdeviceptr` — a device memory address. `unsigned long long` (64-bit) on
/// every platform CUDA supports.
pub(crate) type CUdeviceptr = u64;

/// `CUDA_SUCCESS`.
pub(crate) const CUDA_SUCCESS: CUresult = 0;

/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR` (cuda.h enum value 75).
pub(crate) const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR` (cuda.h enum value 76).
pub(crate) const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

/// `NVRTC` status code. `0` (`NVRTC_SUCCESS`) is success.
pub(crate) type NvrtcResult = c_int;
/// `nvrtcProgram` — an opaque NVRTC compilation-unit handle.
pub(crate) type NvrtcProgram = *mut c_void;

/// `NVRTC_SUCCESS`.
pub(crate) const NVRTC_SUCCESS: NvrtcResult = 0;

// Candidate library names, tried in order. On a host with no NVIDIA driver
// (e.g. an Apple Mac) none load and the probe/context returns BackendUnavailable.
const LIBCUDA_CANDIDATES: &[&[u8]] = &[
    b"libcuda.so.1\0", // Linux (the driver stub library)
    b"libcuda.so\0",
    b"nvcuda.dll\0", // Windows
];

// NVRTC ships a version-suffixed name on each platform; the exact Windows suffix
// tracks the toolkit version, so several are tried. (vast.ai TODO: confirm the
// precise `nvrtc64_<major><minor>_0.dll` present on the RTX 4090 image.)
const LIBNVRTC_CANDIDATES: &[&[u8]] = &[
    b"libnvrtc.so\0",
    b"libnvrtc.so.12\0",
    b"libnvrtc.so.11\0",
    b"nvrtc64_120_0.dll\0",
    b"nvrtc64_112_0.dll\0",
];

// Driver API function-pointer signatures (exact C prototypes from cuda.h).
pub(crate) type FnCuInit = unsafe extern "C" fn(c_uint) -> CUresult;
pub(crate) type FnCuDriverGetVersion = unsafe extern "C" fn(*mut c_int) -> CUresult;
pub(crate) type FnCuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CUresult;
pub(crate) type FnCuDeviceGet = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
pub(crate) type FnCuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult;
pub(crate) type FnCuDeviceGetAttribute =
    unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult;
pub(crate) type FnCuGetErrorString = unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult;
pub(crate) type FnCuCtxCreate = unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult;
pub(crate) type FnCuCtxDestroy = unsafe extern "C" fn(CUcontext) -> CUresult;
pub(crate) type FnCuStreamCreate = unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult;
pub(crate) type FnCuStreamDestroy = unsafe extern "C" fn(CUstream) -> CUresult;
pub(crate) type FnCuStreamSynchronize = unsafe extern "C" fn(CUstream) -> CUresult;
pub(crate) type FnCuMemAlloc = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult;
pub(crate) type FnCuMemFree = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
pub(crate) type FnCuMemcpyHtoD =
    unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
pub(crate) type FnCuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
pub(crate) type FnCuModuleLoadData = unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;
pub(crate) type FnCuModuleGetFunction =
    unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
pub(crate) type FnCuModuleUnload = unsafe extern "C" fn(CUmodule) -> CUresult;
#[allow(clippy::type_complexity)] // exact cuLaunchKernel prototype (7 launch dims + stream + 2 arg arrays)
pub(crate) type FnCuLaunchKernel = unsafe extern "C" fn(
    CUfunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    CUstream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;
/// `cuFuncSetAttribute(CUfunction, CUfunction_attribute, int)` — exact cuda.h
/// prototype. Optional (M4-07): the FA v3 kernel's 82 944-byte dynamic
/// shared-memory tile exceeds the 48 KiB default per-block cap, so the lazy
/// FA v3 module init must opt in via
/// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`. FA v2 (40 KiB, under the
/// default cap) and the decomposed chain never need it.
pub(crate) type FnCuFuncSetAttribute = unsafe extern "C" fn(CUfunction, c_int, c_int) -> CUresult;
/// `cuTensorMapEncodeTiled` — exact cuda.h (12.x) prototype, with the opaque
/// out-param `CUtensorMap*` (a 128-byte, 64-byte-aligned struct) and the enum
/// parameters (`CUtensorMapDataType` / `Interleave` / `Swizzle` /
/// `L2promotion` / `FloatOOBfill`) passed as `c_uint` (C enums are int-sized
/// here). Optional (M4-07-T04): resolved as the **Hopper-era-driver canary +
/// TMA descriptor plumbing** for the FA v3 path — see
/// `docs/adr/M4-07-fa-v3-hopper.md` §(e). The initial FA v3 kernel streams
/// K/V with `cp.async` and does not consume a tensor map yet; the resolve +
/// degrade wiring keeps the TMA follow-up purely kernel-side.
pub(crate) type FnCuTensorMapEncodeTiled = unsafe extern "C" fn(
    *mut c_void, // CUtensorMap* tensorMap (out, 128 B / 64-B-aligned)
    c_uint,      // CUtensorMapDataType tensorDataType
    c_uint,      // cuuint32_t tensorRank
    *mut c_void, // void* globalAddress
    *const u64,  // const cuuint64_t* globalDim   [tensorRank]
    *const u64,  // const cuuint64_t* globalStrides [tensorRank - 1], bytes
    *const u32,  // const cuuint32_t* boxDim      [tensorRank]
    *const u32,  // const cuuint32_t* elementStrides [tensorRank]
    c_uint,      // CUtensorMapInterleave interleave
    c_uint,      // CUtensorMapSwizzle swizzle
    c_uint,      // CUtensorMapL2promotion l2Promotion
    c_uint,      // CUtensorMapFloatOOBfill oobFill
) -> CUresult;

/// The resolved CUDA Driver API entry points, plus the `libcuda` handle that
/// keeps them valid. Loaded once by [`CudaDriver::load`].
///
/// One unified table (rather than a probe-only subset) keeps the code small;
/// every symbol here is exported by any `libcuda` since CUDA 4.x, so requiring
/// them is equivalent to requiring a modern driver.
pub(crate) struct CudaDriver {
    _lib: DynLib,
    pub(crate) cu_init: FnCuInit,
    pub(crate) cu_driver_get_version: FnCuDriverGetVersion,
    pub(crate) cu_device_get_count: FnCuDeviceGetCount,
    pub(crate) cu_device_get: FnCuDeviceGet,
    pub(crate) cu_device_get_name: FnCuDeviceGetName,
    pub(crate) cu_device_get_attribute: FnCuDeviceGetAttribute,
    pub(crate) cu_get_error_string: FnCuGetErrorString,
    pub(crate) cu_ctx_create: FnCuCtxCreate,
    pub(crate) cu_ctx_destroy: FnCuCtxDestroy,
    pub(crate) cu_stream_create: FnCuStreamCreate,
    pub(crate) cu_stream_destroy: FnCuStreamDestroy,
    pub(crate) cu_stream_synchronize: FnCuStreamSynchronize,
    pub(crate) cu_mem_alloc: FnCuMemAlloc,
    pub(crate) cu_mem_free: FnCuMemFree,
    pub(crate) cu_memcpy_htod: FnCuMemcpyHtoD,
    pub(crate) cu_memcpy_dtoh: FnCuMemcpyDtoH,
    pub(crate) cu_module_load_data: FnCuModuleLoadData,
    pub(crate) cu_module_get_function: FnCuModuleGetFunction,
    pub(crate) cu_module_unload: FnCuModuleUnload,
    pub(crate) cu_launch_kernel: FnCuLaunchKernel,
    /// Optional (M4-07): `cuFuncSetAttribute` — required only by the FA v3
    /// shared-memory opt-in. `None` (a pre-CUDA-9-era driver) degrades FA v3
    /// explicitly and leaves FA v2 / decomposed untouched.
    pub(crate) cu_func_set_attribute: Option<FnCuFuncSetAttribute>,
    /// Optional (M4-07-T04): `cuTensorMapEncodeTiled` — the TMA descriptor
    /// encoder, exported by CUDA 12-era drivers. Doubles as the
    /// Hopper-generation-driver canary for the FA v3 gate (ADR M4-07 §(e)):
    /// a driver too old to export it cannot JIT `compute_90a` PTX either.
    pub(crate) cu_tensor_map_encode_tiled: Option<FnCuTensorMapEncodeTiled>,
}

impl CudaDriver {
    /// Loads `libcuda` and resolves every driver entry point.
    ///
    /// # Errors
    /// [`VokraError::BackendUnavailable`] if the NVIDIA driver library is not
    /// present (no GPU / driver not installed / non-NVIDIA host such as an Apple
    /// Mac), or a required symbol is missing (driver too old). Never a silent
    /// fall back (NFR-RL-06).
    pub(crate) fn load() -> Result<CudaDriver> {
        let lib = DynLib::open(LIBCUDA_CANDIDATES).ok_or_else(|| {
            VokraError::BackendUnavailable(
                "libcuda (NVIDIA driver) not found: no NVIDIA GPU/driver on this host, or CUDA is \
                 not installed. Vokra does not bundle the CUDA runtime (NVIDIA EULA install model, \
                 FR-BE-08); install the NVIDIA driver to use the CUDA backend."
                    .to_owned(),
            )
        })?;
        // SAFETY: each `get::<Fn…>` pairs the exact C symbol name with the
        // function-pointer alias declaring its true signature (cuda.h). The
        // `_v2` names are the symbols a modern `libcuda` actually exports for
        // the context/stream/memory entry points.
        unsafe {
            Ok(CudaDriver {
                cu_init: lib.get(b"cuInit\0")?,
                cu_driver_get_version: lib.get(b"cuDriverGetVersion\0")?,
                cu_device_get_count: lib.get(b"cuDeviceGetCount\0")?,
                cu_device_get: lib.get(b"cuDeviceGet\0")?,
                cu_device_get_name: lib.get(b"cuDeviceGetName\0")?,
                cu_device_get_attribute: lib.get(b"cuDeviceGetAttribute\0")?,
                cu_get_error_string: lib.get(b"cuGetErrorString\0")?,
                cu_ctx_create: lib.get(b"cuCtxCreate_v2\0")?,
                cu_ctx_destroy: lib.get(b"cuCtxDestroy_v2\0")?,
                cu_stream_create: lib.get(b"cuStreamCreate\0")?,
                cu_stream_destroy: lib.get(b"cuStreamDestroy_v2\0")?,
                cu_stream_synchronize: lib.get(b"cuStreamSynchronize\0")?,
                cu_mem_alloc: lib.get(b"cuMemAlloc_v2\0")?,
                cu_mem_free: lib.get(b"cuMemFree_v2\0")?,
                cu_memcpy_htod: lib.get(b"cuMemcpyHtoD_v2\0")?,
                cu_memcpy_dtoh: lib.get(b"cuMemcpyDtoH_v2\0")?,
                cu_module_load_data: lib.get(b"cuModuleLoadData\0")?,
                cu_module_get_function: lib.get(b"cuModuleGetFunction\0")?,
                cu_module_unload: lib.get(b"cuModuleUnload\0")?,
                cu_launch_kernel: lib.get(b"cuLaunchKernel\0")?,
                // Optional Hopper-era symbols (M4-07): absence is NOT an
                // error — it only degrades the FA v3 path (explicit stderr
                // log at the capability-decision site, FR-EX-08).
                cu_func_set_attribute: lib.get_opt(b"cuFuncSetAttribute\0"),
                cu_tensor_map_encode_tiled: lib.get_opt(b"cuTensorMapEncodeTiled\0"),
                _lib: lib,
            })
        }
    }
}

// NVRTC function-pointer signatures (exact C prototypes from nvrtc.h).
#[allow(clippy::type_complexity)] // exact nvrtcCreateProgram prototype
pub(crate) type FnNvrtcCreateProgram = unsafe extern "C" fn(
    *mut NvrtcProgram,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> NvrtcResult;
pub(crate) type FnNvrtcCompileProgram =
    unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> NvrtcResult;
pub(crate) type FnNvrtcGetPtxSize = unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult;
pub(crate) type FnNvrtcGetPtx = unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult;
pub(crate) type FnNvrtcDestroyProgram = unsafe extern "C" fn(*mut NvrtcProgram) -> NvrtcResult;
pub(crate) type FnNvrtcGetProgramLogSize =
    unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult;
pub(crate) type FnNvrtcGetProgramLog =
    unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult;
pub(crate) type FnNvrtcGetErrorString = unsafe extern "C" fn(NvrtcResult) -> *const c_char;

/// The resolved NVRTC entry points, plus the `libnvrtc` handle. NVRTC compiles
/// the CUDA C GEMM to PTX **on the GPU host at runtime** (device-side JIT, not
/// CPU code generation — NFR-RL-05).
pub(crate) struct Nvrtc {
    _lib: DynLib,
    pub(crate) create_program: FnNvrtcCreateProgram,
    pub(crate) compile_program: FnNvrtcCompileProgram,
    pub(crate) get_ptx_size: FnNvrtcGetPtxSize,
    pub(crate) get_ptx: FnNvrtcGetPtx,
    pub(crate) destroy_program: FnNvrtcDestroyProgram,
    pub(crate) get_program_log_size: FnNvrtcGetProgramLogSize,
    pub(crate) get_program_log: FnNvrtcGetProgramLog,
    pub(crate) get_error_string: FnNvrtcGetErrorString,
}

impl Nvrtc {
    /// Loads `libnvrtc` and resolves every NVRTC entry point.
    ///
    /// # M3-01-T09: library search fallback
    ///
    /// Search order:
    /// 1. `VOKRA_NVRTC_PATH` env var — an absolute path to `libnvrtc.so*` /
    ///    `nvrtc64_*.dll` (owner escape hatch for hosts where the toolkit is in
    ///    a non-standard prefix, e.g. `/opt/cuda-12.6/lib64/libnvrtc.so.12`).
    /// 2. [`LIBNVRTC_CANDIDATES`] — the bare filenames the platform loader
    ///    searches (`LD_LIBRARY_PATH` / `PATH` / the standard toolkit prefixes
    ///    the loader consults).
    ///
    /// **Not** a silent CPU fall back on failure: on either 1 or 2 missing the
    /// library, the returned error is an explicit
    /// [`VokraError::BackendUnavailable`] (FR-EX-08 / NFR-RL-06).
    ///
    /// # Errors
    /// [`VokraError::BackendUnavailable`] if NVRTC is not installed (it ships
    /// with the CUDA toolkit, not the bare driver) or a symbol is missing.
    pub(crate) fn load() -> Result<Nvrtc> {
        let lib = Self::load_lib_with_env_fallback()?;
        // SAFETY: each `get::<Fn…>` pairs the exact NVRTC symbol name with the
        // function-pointer alias declaring its true signature (nvrtc.h).
        unsafe {
            Ok(Nvrtc {
                create_program: lib.get(b"nvrtcCreateProgram\0")?,
                compile_program: lib.get(b"nvrtcCompileProgram\0")?,
                get_ptx_size: lib.get(b"nvrtcGetPTXSize\0")?,
                get_ptx: lib.get(b"nvrtcGetPTX\0")?,
                destroy_program: lib.get(b"nvrtcDestroyProgram\0")?,
                get_program_log_size: lib.get(b"nvrtcGetProgramLogSize\0")?,
                get_program_log: lib.get(b"nvrtcGetProgramLog\0")?,
                get_error_string: lib.get(b"nvrtcGetErrorString\0")?,
                _lib: lib,
            })
        }
    }

    /// Tries `VOKRA_NVRTC_PATH` env var first (M3-01-T09 fallback #1); on env
    /// unset / empty / invalid it falls through to
    /// [`LIBNVRTC_CANDIDATES`] (fallback #2). Both misses collapse into one
    /// explicit [`VokraError::BackendUnavailable`] (FR-EX-08 / NFR-RL-06).
    fn load_lib_with_env_fallback() -> Result<DynLib> {
        // (1) env-var absolute path (if set + non-empty + NUL-free).
        if let Ok(path) = std::env::var("VOKRA_NVRTC_PATH") {
            if !path.is_empty() && !path.bytes().any(|b| b == 0) {
                // Build a NUL-terminated candidate — DynLib::open expects one.
                let mut c_path = path.into_bytes();
                c_path.push(0);
                if let Some(lib) = DynLib::open(&[&c_path]) {
                    return Ok(lib);
                }
                // env var was set but load failed → explicit error, do NOT
                // silently fall back to the platform candidates. The intent of
                // setting VOKRA_NVRTC_PATH is to *pin* the location; falling
                // through would hide a typo behind a lucky system default.
                return Err(VokraError::BackendUnavailable(format!(
                    "VOKRA_NVRTC_PATH set but library failed to load ({} bytes); \
                     unset the env var to fall back to the system search path, \
                     or point it at a valid libnvrtc / nvrtc64_XX_0.dll",
                    c_path.len() - 1
                )));
            }
        }
        // (2) Platform bare-name candidates (the loader's own search path).
        DynLib::open(LIBNVRTC_CANDIDATES).ok_or_else(|| {
            VokraError::BackendUnavailable(
                "libnvrtc (NVRTC runtime compiler) not found: install the CUDA toolkit's NVRTC \
                 component (or set VOKRA_NVRTC_PATH to its absolute path). Vokra bundles no \
                 NVIDIA library (EULA install model, FR-BE-08)."
                    .to_owned(),
            )
        })
    }
}

/// Maps a non-success [`CUresult`] to an explicit [`VokraError::BackendUnavailable`]
/// carrying the driver's own `cuGetErrorString` message (no silent success,
/// NFR-RL-06).
pub(crate) fn check(driver: &CudaDriver, res: CUresult, what: &str) -> Result<()> {
    if res == CUDA_SUCCESS {
        return Ok(());
    }
    let mut msg_ptr: *const c_char = core::ptr::null();
    // SAFETY: `cu_get_error_string` writes a pointer to a static, NUL-terminated
    // driver string (or leaves it null for an unknown code) into `msg_ptr`.
    unsafe { (driver.cu_get_error_string)(res, &mut msg_ptr) };
    let detail = cstr_to_string(msg_ptr).unwrap_or_else(|| format!("CUDA error code {res}"));
    Err(VokraError::BackendUnavailable(format!(
        "{what} failed: {detail} (CUresult {res})"
    )))
}

/// Maps a non-success [`NvrtcResult`] to an explicit
/// [`VokraError::BackendUnavailable`] with the NVRTC error string.
pub(crate) fn check_nvrtc(nvrtc: &Nvrtc, res: NvrtcResult, what: &str) -> Result<()> {
    if res == NVRTC_SUCCESS {
        return Ok(());
    }
    // SAFETY: `nvrtcGetErrorString` returns a static NUL-terminated string for
    // any result code.
    let ptr = unsafe { (nvrtc.get_error_string)(res) };
    let detail = cstr_to_string(ptr).unwrap_or_else(|| format!("NVRTC error code {res}"));
    Err(VokraError::BackendUnavailable(format!(
        "{what} failed: {detail} (nvrtcResult {res})"
    )))
}

/// Reads a NUL-terminated C string pointer into an owned `String`; `None` if
/// the pointer is null.
///
/// # Safety note
/// The pointer must be null or point to a valid NUL-terminated C string that is
/// alive for the duration of this call (the driver/NVRTC strings are static).
fn cstr_to_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: `ptr` is non-null and points to a static NUL-terminated string
    // owned by the driver/NVRTC; we copy it into an owned String immediately.
    let cstr = unsafe { core::ffi::CStr::from_ptr(ptr) };
    Some(cstr.to_string_lossy().into_owned())
}

/// Interprets a fixed byte buffer written by `cuDeviceGetName` (NUL-terminated
/// within the buffer) as an owned `String`.
pub(crate) fn name_from_buf(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}
