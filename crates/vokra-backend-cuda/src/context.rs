//! CUDA working context: driver + device + context + stream + the FP32 GEMM
//! kernel (NVRTC-compiled PTX). Unix / Windows only.
//!
//! This is the **directly callable** compute surface, mirroring
//! `vokra-backend-metal`'s `MetalContext` and `vokra-backend-cpu`'s `kernels::*`:
//! [`CudaContext::gemm_f32`] runs a row-major single-precision GEMM on the GPU
//! with the **exact** shape/semantics contract of
//! `vokra_backend_cpu::kernels::gemm_f32` (row-major, per-column bias,
//! `out = A·B + bias`), so the two are differentially comparable
//! (M2-03-T18/T19; NFR-QL-01, FP32 `atol = 0.01`). This foundation slice ships
//! **only** the GEMM (`MatMul`); GEMV / softmax / layer-norm / GELU / conv1d /
//! FlashAttention-v2 are the follow-on M2-03 tickets (T10–T14).
//!
//! # Precision (FP32, red line)
//!
//! The kernel is authored in explicit `float` (FP32). It uses no cuBLAS Tensor
//! Core / TF32 fast path, so there is no implicit precision reduction (FP16 /
//! quantised tiers are M2-08).
//!
//! # Kernel build (NVRTC → PTX, device-side JIT — not CPU codegen)
//!
//! The CUDA C GEMM is compiled to PTX at runtime with NVRTC and loaded via
//! `cuModuleLoadData`. This is **GPU** just-in-time compilation performed by the
//! NVIDIA toolchain; the host emits no executable CPU pages (NFR-RL-05). The
//! `cuBLAS`-based GEMM path (FR-BE-03) is a follow-on ticket; this slice proves
//! the end-to-end driver path (module load → alloc → H2D → launch → D2H → free)
//! with a self-contained kernel and no extra NVIDIA runtime dependency.
//!
//! # No bundling (NVIDIA EULA, FR-BE-08)
//!
//! Every NVIDIA entry point is resolved at runtime via dlopen ([`crate::sys`]);
//! nothing is linked or shipped. On a host with no driver (e.g. an Apple Mac)
//! [`CudaContext::new`] returns an explicit [`VokraError::BackendUnavailable`]
//! (never a silent CPU fall back — NFR-RL-06).

use core::ffi::{c_char, c_uint, c_void};

use vokra_core::{Result, VokraError};

use crate::sys::{self, CUcontext, CUdeviceptr, CUfunction, CUmodule, CUstream, CudaDriver, Nvrtc};

/// The GEMM kernel, compiled once per [`CudaContext`]. Row-major, FP32:
/// `C[r, c] = (has_bias ? bias[c] : 0) + Σ_k A[r, k] · B[k, c]` — identical
/// semantics to `vokra_backend_cpu::kernels::gemm_f32`. `extern "C"` suppresses
/// C++ name mangling so `cuModuleGetFunction("vokra_gemm_f32")` resolves it.
const GEMM_CUDA: &str = r#"
extern "C" __global__ void vokra_gemm_f32(
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int has_bias)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) {
        return;
    }
    float acc = 0.0f;
    unsigned int arow = row * K;
    for (unsigned int k = 0; k < K; ++k) {
        acc += A[arow + k] * B[k * N + col];
    }
    if (has_bias != 0u) {
        acc += bias[col];
    }
    C[row * N + col] = acc;
}
"#;

/// 16×16 thread block (matches the Metal GEMM launch); the kernel guards the
/// ragged tail against `M`/`N`.
const BLOCK: u32 = 16;

/// A `+`-owned device allocation, freed exactly once on drop via the borrowed
/// driver. Borrowing (rather than storing a fn pointer) keeps `sys`' private
/// signature aliases encapsulated, and an early `?`-return mid-`run_gemm`
/// still frees every buffer already allocated.
struct DeviceBuf<'a> {
    driver: &'a CudaDriver,
    ptr: CUdeviceptr,
}

impl Drop for DeviceBuf<'_> {
    fn drop(&mut self) {
        if self.ptr == 0 {
            return;
        }
        // SAFETY: `ptr` is a live device allocation from `cuMemAlloc`, freed once.
        unsafe { (self.driver.cu_mem_free)(self.ptr) };
    }
}

/// A CUDA driver + device context + stream + compiled GEMM kernel.
///
/// Holds the owned driver context, stream and module, released in [`Drop`] in
/// reverse creation order. Not `Send`/`Sync`: the driver handles are used from
/// the thread that created them (sufficient for the parity harness; a
/// thread-affine / `Send` wrapper is a later concern, mirroring `MetalContext`).
pub struct CudaContext {
    driver: CudaDriver,
    context: CUcontext,
    stream: CUstream,
    module: CUmodule,
    gemm: CUfunction,
}

impl CudaContext {
    /// Loads the driver, creates a context + stream on device 0, and
    /// NVRTC-compiles + loads the FP32 GEMM kernel.
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] if there is no NVIDIA driver/GPU (e.g.
    /// on an Apple Mac — dlopen finds no `libcuda`), if a driver call fails, or
    /// if NVRTC is absent / the kernel fails to compile (the NVRTC log is
    /// included). Never a silent CPU fall back (NFR-RL-06).
    pub fn new() -> Result<CudaContext> {
        let driver = CudaDriver::load()?;

        // SAFETY: `cuInit(0)` must precede any other driver call; flag 0 is the
        // only defined value.
        let r = unsafe { (driver.cu_init)(0) };
        sys::check(&driver, r, "cuInit")?;

        let mut count = 0;
        // SAFETY: writes the device count into `count`.
        let r = unsafe { (driver.cu_device_get_count)(&mut count) };
        sys::check(&driver, r, "cuDeviceGetCount")?;
        if count <= 0 {
            return Err(VokraError::BackendUnavailable(
                "CUDA driver present but no CUDA-capable GPU (device count 0)".to_owned(),
            ));
        }

        let mut dev: sys::CUdevice = 0;
        // SAFETY: writes the ordinal-0 device handle into `dev`.
        let r = unsafe { (driver.cu_device_get)(&mut dev, 0) };
        sys::check(&driver, r, "cuDeviceGet")?;

        let mut context: CUcontext = core::ptr::null_mut();
        // SAFETY: creates a context (flags 0) on the valid device `dev`, writing
        // the owned handle into `context`.
        let r = unsafe { (driver.cu_ctx_create)(&mut context, 0, dev) };
        sys::check(&driver, r, "cuCtxCreate")?;

        // From here a failure must destroy `context`; `build_pipeline` cleans up
        // its own stream/module on partial failure.
        match build_pipeline(&driver) {
            Ok((stream, module, gemm)) => Ok(CudaContext {
                driver,
                context,
                stream,
                module,
                gemm,
            }),
            Err(e) => {
                // SAFETY: `context` is the just-created owned context; destroy it
                // before propagating the error (no leak).
                unsafe { (driver.cu_ctx_destroy)(context) };
                Err(e)
            }
        }
    }

    /// Row-major FP32 GEMM on the GPU with optional per-column bias:
    /// `out[i, j] = bias[j] + Σ_l a[i, l] · b[l, j]`.
    ///
    /// `a` is `m×k`, `b` is `k×n`, `out` is `m×n`, and `bias` (when `Some`) has
    /// length `n` — the exact contract of
    /// `vokra_backend_cpu::kernels::gemm_f32`, so the two are differentially
    /// comparable (M2-03-T19).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any shape mismatch or a zero
    /// dimension; [`VokraError::BackendUnavailable`] if a device allocation /
    /// copy / launch fails.
    #[allow(clippy::too_many_arguments)] // intrinsic GEMM parameter set (matches CPU gemm_f32)
    pub fn gemm_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        validate_gemm(m, n, k, a, b, bias, out)?;
        self.run_gemm(m, n, k, a, b, bias, out)
    }

    /// GEMM body: allocate device buffers, H2D, launch, synchronise, D2H, free.
    /// Shapes are already validated.
    #[allow(clippy::too_many_arguments)] // intrinsic GEMM parameter set
    fn run_gemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        let d = &self.driver;

        // Device inputs (copied H2D). A failed alloc `?`-returns; already-built
        // DeviceBufs free on drop.
        let a_buf = self.alloc(size_of_val(a))?;
        self.htod(&a_buf, a)?;
        let b_buf = self.alloc(size_of_val(b))?;
        self.htod(&b_buf, b)?;

        // Bias: the real bias when present, else a 1-float dummy the kernel never
        // reads (has_bias = 0). Always allocated so the kernel arg is bound.
        let dummy = [0.0f32];
        let bias_slice = bias.unwrap_or(&dummy);
        let bias_buf = self.alloc(size_of_val(bias_slice))?;
        self.htod(&bias_buf, bias_slice)?;

        // Output (uninitialised device storage of m*n floats).
        let c_buf = self.alloc(size_of_val(out))?;

        // Scalar kernel args (must outlive the launch call, which captures them).
        let m_u = m as c_uint;
        let n_u = n as c_uint;
        let k_u = k as c_uint;
        let has_bias: c_uint = u32::from(bias.is_some());

        // `kernelParams`: one pointer per argument, in the kernel's declared
        // order (A, B, bias, C, M, N, K, has_bias). Each points to the value:
        // for the pointer args that is the CUdeviceptr; for the scalars the u32.
        let mut params: [*mut c_void; 8] = [
            (&a_buf.ptr as *const CUdeviceptr)
                .cast::<c_void>()
                .cast_mut(),
            (&b_buf.ptr as *const CUdeviceptr)
                .cast::<c_void>()
                .cast_mut(),
            (&bias_buf.ptr as *const CUdeviceptr)
                .cast::<c_void>()
                .cast_mut(),
            (&c_buf.ptr as *const CUdeviceptr)
                .cast::<c_void>()
                .cast_mut(),
            (&m_u as *const c_uint).cast::<c_void>().cast_mut(),
            (&n_u as *const c_uint).cast::<c_void>().cast_mut(),
            (&k_u as *const c_uint).cast::<c_void>().cast_mut(),
            (&has_bias as *const c_uint).cast::<c_void>().cast_mut(),
        ];

        // Grid: x = columns (N), y = rows (M), measured in blocks; the kernel
        // guards row/col against M/N for the ragged edges.
        let grid_x = n.div_ceil(BLOCK as usize) as c_uint;
        let grid_y = m.div_ceil(BLOCK as usize) as c_uint;

        // SAFETY: `self.gemm` is the loaded `vokra_gemm_f32` function; the launch
        // dims are non-zero (validated m,n,k >= 1); `self.stream` is the owned
        // stream; `params` holds one valid pointer per kernel argument, matching
        // the kernel's signature and alive across this synchronous launch; no
        // dynamic shared memory (0) and no `extra` (null).
        let launch = unsafe {
            (d.cu_launch_kernel)(
                self.gemm,
                grid_x,
                grid_y,
                1,
                BLOCK,
                BLOCK,
                1,
                0,
                self.stream,
                params.as_mut_ptr(),
                core::ptr::null_mut(),
            )
        };
        sys::check(d, launch, "cuLaunchKernel(vokra_gemm_f32)")?;

        // SAFETY: waits for the launch on the owned stream to complete before D2H.
        let sync = unsafe { (d.cu_stream_synchronize)(self.stream) };
        sys::check(d, sync, "cuStreamSynchronize")?;

        self.dtoh(&c_buf, out)
    }

    /// Allocates `bytes` (min one float) of device memory, tied to `&self`.
    fn alloc(&self, bytes: usize) -> Result<DeviceBuf<'_>> {
        let bytes = bytes.max(size_of::<f32>());
        let mut ptr: CUdeviceptr = 0;
        // SAFETY: `cuMemAlloc` writes a device pointer to `bytes` into `ptr`.
        let r = unsafe { (self.driver.cu_mem_alloc)(&mut ptr, bytes) };
        sys::check(&self.driver, r, "cuMemAlloc")?;
        Ok(DeviceBuf {
            driver: &self.driver,
            ptr,
        })
    }

    /// Copies `host` into device buffer `buf` (host-to-device).
    fn htod(&self, buf: &DeviceBuf<'_>, host: &[f32]) -> Result<()> {
        // SAFETY: `buf.ptr` is a device allocation of at least `size_of_val(host)`
        // bytes; `host.as_ptr()` is valid for that many bytes.
        let r = unsafe {
            (self.driver.cu_memcpy_htod)(buf.ptr, host.as_ptr().cast::<c_void>(), size_of_val(host))
        };
        sys::check(&self.driver, r, "cuMemcpyHtoD")
    }

    /// Copies device buffer `buf` into `host` (device-to-host).
    fn dtoh(&self, buf: &DeviceBuf<'_>, host: &mut [f32]) -> Result<()> {
        // SAFETY: `host.as_mut_ptr()` is valid for `size_of_val(host)` bytes;
        // `buf.ptr` is a device allocation of at least that many bytes.
        let r = unsafe {
            (self.driver.cu_memcpy_dtoh)(
                host.as_mut_ptr().cast::<c_void>(),
                buf.ptr,
                size_of_val(host),
            )
        };
        sys::check(&self.driver, r, "cuMemcpyDtoH")
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        // SAFETY: each handle is a valid owned object created in `new` /
        // `build_pipeline`; released once, in reverse creation order. The
        // `driver` (which owns the libcuda handle whose fn pointers these are)
        // is dropped only after this method returns, so the pointers are live.
        unsafe {
            if !self.stream.is_null() {
                (self.driver.cu_stream_synchronize)(self.stream);
                (self.driver.cu_stream_destroy)(self.stream);
            }
            if !self.module.is_null() {
                (self.driver.cu_module_unload)(self.module);
            }
            if !self.context.is_null() {
                (self.driver.cu_ctx_destroy)(self.context);
            }
        }
    }
}

impl core::fmt::Debug for CudaContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CudaContext").finish_non_exhaustive()
    }
}

/// Creates the stream, then NVRTC-compiles + loads the GEMM module. On a partial
/// failure (module build fails after the stream exists) the stream is destroyed
/// before the error propagates.
fn build_pipeline(driver: &CudaDriver) -> Result<(CUstream, CUmodule, CUfunction)> {
    let mut stream: CUstream = core::ptr::null_mut();
    // SAFETY: creates a stream (flags 0 = default) on the current context.
    let r = unsafe { (driver.cu_stream_create)(&mut stream, 0) };
    sys::check(driver, r, "cuStreamCreate")?;
    match load_gemm_module(driver) {
        Ok((module, gemm)) => Ok((stream, module, gemm)),
        Err(e) => {
            // SAFETY: `stream` is the just-created owned stream; destroy it.
            unsafe { (driver.cu_stream_destroy)(stream) };
            Err(e)
        }
    }
}

/// NVRTC-compiles the GEMM to PTX, loads it as a module, and resolves the
/// kernel function. Unloads the module if the function lookup fails.
fn load_gemm_module(driver: &CudaDriver) -> Result<(CUmodule, CUfunction)> {
    let nvrtc = Nvrtc::load()?;
    let ptx = compile_gemm_ptx(&nvrtc)?;
    drop(nvrtc); // libnvrtc no longer needed once the PTX text is in hand

    let mut module: CUmodule = core::ptr::null_mut();
    // SAFETY: `ptx` is a NUL-terminated PTX image produced by NVRTC; the driver
    // parses it and writes the owned module handle into `module`.
    let r = unsafe { (driver.cu_module_load_data)(&mut module, ptx.as_ptr().cast::<c_void>()) };
    sys::check(driver, r, "cuModuleLoadData")?;

    let mut func: CUfunction = core::ptr::null_mut();
    // SAFETY: `module` is valid; the C string names the `extern "C"` kernel.
    let fret =
        unsafe { (driver.cu_module_get_function)(&mut func, module, c"vokra_gemm_f32".as_ptr()) };
    let got = sys::check(driver, fret, "cuModuleGetFunction(vokra_gemm_f32)");
    match got {
        Ok(()) => Ok((module, func)),
        Err(e) => {
            // SAFETY: `module` is the just-loaded owned module; unload it.
            unsafe { (driver.cu_module_unload)(module) };
            Err(e)
        }
    }
}

/// NVRTC-compiles [`GEMM_CUDA`] to a PTX byte buffer (NUL-terminated). The
/// program handle is always destroyed before returning.
fn compile_gemm_ptx(nvrtc: &Nvrtc) -> Result<Vec<u8>> {
    let src = std::ffi::CString::new(GEMM_CUDA).map_err(|_| {
        VokraError::InvalidArgument("GEMM CUDA source contains an interior NUL".to_owned())
    })?;

    let mut prog: sys::NvrtcProgram = core::ptr::null_mut();
    // SAFETY: `src` is a valid NUL-terminated C string; the name literal is a
    // C string; 0 headers with null header/include arrays. Writes the program
    // handle into `prog`.
    let r = unsafe {
        (nvrtc.create_program)(
            &mut prog,
            src.as_ptr(),
            c"vokra_gemm.cu".as_ptr(),
            0,
            core::ptr::null(),
            core::ptr::null(),
        )
    };
    sys::check_nvrtc(nvrtc, r, "nvrtcCreateProgram")?;

    // `prog` now exists; ensure it is destroyed on every exit path.
    let result = compile_and_extract_ptx(nvrtc, prog);
    // SAFETY: `prog` is a valid program handle; destroyed exactly once here.
    unsafe { (nvrtc.destroy_program)(&mut prog) };
    result
}

/// Compiles `prog` (no options → NVRTC's default target arch) and extracts its
/// PTX. On a compile failure the NVRTC log is surfaced in the error.
///
/// vast.ai TODO: pin `--gpu-architecture=compute_89` (Ada / RTX 4090) once the
/// runner's toolkit version is confirmed, rather than relying on the default.
fn compile_and_extract_ptx(nvrtc: &Nvrtc, prog: sys::NvrtcProgram) -> Result<Vec<u8>> {
    // SAFETY: `prog` is a valid program; 0 options with a null options array.
    let compile_res = unsafe { (nvrtc.compile_program)(prog, 0, core::ptr::null()) };
    if compile_res != sys::NVRTC_SUCCESS {
        return Err(VokraError::BackendUnavailable(format!(
            "NVRTC GEMM compile failed: {}",
            fetch_nvrtc_log(nvrtc, prog)
        )));
    }

    let mut size = 0usize;
    // SAFETY: writes the PTX byte length (incl. trailing NUL) into `size`.
    let r = unsafe { (nvrtc.get_ptx_size)(prog, &mut size) };
    sys::check_nvrtc(nvrtc, r, "nvrtcGetPTXSize")?;

    let mut ptx = vec![0u8; size.max(1)];
    // SAFETY: `ptx` has `size` writable bytes; NVRTC writes the NUL-terminated
    // PTX text into it.
    let r = unsafe { (nvrtc.get_ptx)(prog, ptx.as_mut_ptr().cast::<c_char>()) };
    sys::check_nvrtc(nvrtc, r, "nvrtcGetPTX")?;
    Ok(ptx)
}

/// Best-effort NVRTC compile log for a program (`"(no log)"` if empty).
fn fetch_nvrtc_log(nvrtc: &Nvrtc, prog: sys::NvrtcProgram) -> String {
    let mut size = 0usize;
    // SAFETY: writes the log byte length into `size`.
    let ok = unsafe { (nvrtc.get_program_log_size)(prog, &mut size) };
    if ok != sys::NVRTC_SUCCESS || size <= 1 {
        return "(no log)".to_owned();
    }
    let mut buf = vec![0u8; size];
    // SAFETY: `buf` has `size` writable bytes; NVRTC writes the NUL-terminated
    // log into it.
    let ok = unsafe { (nvrtc.get_program_log)(prog, buf.as_mut_ptr().cast::<c_char>()) };
    if ok != sys::NVRTC_SUCCESS {
        return "(log unavailable)".to_owned();
    }
    sys::name_from_buf(&buf)
}

// ---- shape validation (mirrors the CPU / Metal gemm validator) ----

fn checked_mul(a: usize, b: usize, what: &str) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| {
        VokraError::InvalidArgument(format!("{what}: dimension product overflows usize"))
    })
}

fn expect_len(name: &str, got: usize, want: usize) -> Result<()> {
    if got == want {
        Ok(())
    } else {
        Err(VokraError::InvalidArgument(format!(
            "{name} length {got} does not match expected {want}"
        )))
    }
}

fn validate_gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &[f32],
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(VokraError::InvalidArgument(
            "gemm dimensions m, n, k must all be >= 1".to_owned(),
        ));
    }
    expect_len("gemm a", a.len(), checked_mul(m, k, "gemm m*k")?)?;
    expect_len("gemm b", b.len(), checked_mul(k, n, "gemm k*n")?)?;
    expect_len("gemm out", out.len(), checked_mul(m, n, "gemm m*n")?)?;
    if let Some(bias) = bias {
        expect_len("gemm bias", bias.len(), n)?;
    }
    Ok(())
}
