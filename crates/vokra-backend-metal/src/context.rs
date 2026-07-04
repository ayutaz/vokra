//! Metal working context: device + command queue + the FP32 GEMM compute
//! pipeline (M2-01-T05/T06/T08). Apple targets only.
//!
//! This is the **directly callable** compute surface (mirroring
//! `vokra-backend-cpu`'s `kernels::gemm_f32`): [`MetalContext::gemm_f32`] runs a
//! row-major single-precision GEMM on the GPU and is what the parity tests call
//! (M2-01-T17/T18). [`crate::MetalBackend`] wraps a context for the `Backend`
//! trait but, exactly like `CpuBackend`, keeps graph-level `execute` an honest
//! stub until the data-carrying graph engine lands (a later WP).
//!
//! # Precision (FP32, red line)
//!
//! The kernel is authored in explicit `float` (FP32) — Vokra does **not** run
//! this parity path through MPS/MPSGraph, so there is no implicit FP16 fast
//! path to fall into (M2-01 scope note; the FP16 / quantised tiers are M2-08).
//!
//! # Shader build (`newLibraryWithSource:`, no CPU JIT)
//!
//! The MSL is compiled at runtime with
//! `-[MTLDevice newLibraryWithSource:options:error:]`. This is **not** CPU-side
//! W^X code generation (NFR-RL-05): the host emits no executable code; the Metal
//! framework / GPU driver compiles GPU shader code. iOS ships a W^X constraint
//! on *CPU* pages, and Apple's guidance there is to precompile to a `.metallib`
//! at build time — that iOS precompile path is a followup for M2-02 (this slice
//! is macOS, where `newLibraryWithSource:` is the pragmatic route).

use core::ffi::c_void;

use vokra_core::{Result, VokraError};

use crate::sys::{self, Id, MtlSize};

/// The GEMM shader, compiled once per [`MetalContext`]. Row-major, FP32:
/// `C[r, c] = (has_bias ? bias[c] : 0) + Σ_k A[r, k] · B[k, c]` — identical
/// semantics to `vokra_backend_cpu::kernels::gemm_f32`.
const GEMM_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct GemmDims {
    uint M;
    uint N;
    uint K;
    uint has_bias;
};

kernel void vokra_gemm_f32(
    device const float*   A    [[buffer(0)]],
    device const float*   B    [[buffer(1)]],
    device const float*   bias [[buffer(2)]],
    device float*         C    [[buffer(3)]],
    constant GemmDims&    dims [[buffer(4)]],
    uint2                 gid  [[thread_position_in_grid]])
{
    const uint row = gid.y;
    const uint col = gid.x;
    if (row >= dims.M || col >= dims.N) {
        return;
    }
    float acc = 0.0f;
    const uint arow = row * dims.K;
    for (uint k = 0; k < dims.K; ++k) {
        acc += A[arow + k] * B[k * dims.N + col];
    }
    if (dims.has_bias != 0u) {
        acc += bias[col];
    }
    C[row * dims.N + col] = acc;
}
"#;

/// GEMM dimension block handed to the kernel via `setBytes:` (buffer index 4).
/// Field order and `u32` widths mirror the MSL `struct GemmDims`.
#[repr(C)]
#[derive(Clone, Copy)]
struct GemmDims {
    m: u32,
    n: u32,
    k: u32,
    has_bias: u32,
}

/// RAII wrapper for a `+1`-owned `MTLBuffer`, released exactly once on drop.
///
/// Using drop (rather than a manual release ladder) means an early `?`-return
/// mid-setup still releases every buffer already allocated.
struct OwnedBuf(Id);

impl Drop for OwnedBuf {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a valid `+1`-owned MTLBuffer (or null) obtained
        // from a `newBuffer…` call; `release` is sent once.
        unsafe { release(self.0) };
    }
}

/// A Metal device + command queue + compiled GEMM pipeline.
///
/// Holds three `+1`-owned Objective-C objects (device, queue, pipeline),
/// released in [`Drop`]. Not `Send`/`Sync`: the raw `id` handles must be used
/// from the thread that created them (sufficient for the parity harness; a
/// thread-affine or `Send` wrapper is a later concern).
pub struct MetalContext {
    device: Id,
    queue: Id,
    gemm_pipeline: Id,
}

impl MetalContext {
    /// Creates the system default device, a command queue, and compiles the
    /// FP32 GEMM pipeline.
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] if there is no Metal device, the
    /// command queue cannot be created, or the shader fails to compile /
    /// pipeline creation fails (the Metal error description is included).
    pub fn new() -> Result<MetalContext> {
        // SAFETY: `MTLCreateSystemDefaultDevice` takes no arguments and returns
        // an owned `id` (or null), checked below.
        let device = unsafe { sys::MTLCreateSystemDefaultDevice() };
        if device.is_null() {
            return Err(VokraError::BackendUnavailable(
                "no system default Metal device".to_owned(),
            ));
        }

        // SAFETY: `objc_autoreleasePoolPush` returns a token consumed by the one
        // matching pop below; `build` sends only documented selectors to the
        // just-created device.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        // SAFETY: `device` is a valid, non-null MTLDevice owned by us.
        let result = unsafe { Self::build(device) };
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };

        if result.is_err() {
            // SAFETY: release the device we owned before the failure.
            unsafe { release(device) };
        }
        result
    }

    /// Builds queue + pipeline for an already-owned `device`. Runs inside the
    /// caller's autorelease pool.
    ///
    /// # Safety
    /// `device` must be a valid, non-null `MTLDevice` owned by the caller.
    unsafe fn build(device: Id) -> Result<MetalContext> {
        // SAFETY: `device` is a valid MTLDevice per the caller contract.
        let queue = unsafe { sys::send_id(device, sys::sel(b"newCommandQueue\0")) };
        if queue.is_null() {
            return Err(VokraError::BackendUnavailable(
                "MTLDevice newCommandQueue returned nil".to_owned(),
            ));
        }

        // Compile the MSL source into a library.
        let source = std::ffi::CString::new(GEMM_MSL).map_err(|_| {
            VokraError::InvalidArgument("GEMM MSL source contains an interior NUL".to_owned())
        })?;
        // SAFETY: NSString class is loaded (Foundation linked); `source` is a
        // valid NUL-terminated C string. The returned NSString is autoreleased.
        let ns_source = unsafe {
            let nsstring = sys::class(b"NSString\0");
            sys::send_id_cstr(
                nsstring,
                sys::sel(b"stringWithUTF8String:\0"),
                source.as_ptr(),
            )
        };

        let mut err: Id = core::ptr::null_mut();
        // SAFETY: `newLibraryWithSource:options:error:` on the device; a nil
        // options selects defaults, `&mut err` receives an autoreleased NSError
        // on failure.
        let library = unsafe {
            sys::send_new_library(
                device,
                sys::sel(b"newLibraryWithSource:options:error:\0"),
                ns_source,
                core::ptr::null_mut(),
                &mut err,
            )
        };
        if library.is_null() {
            // SAFETY: `err` is null or a valid autoreleased NSError.
            let detail = unsafe { error_description(err) };
            // SAFETY: release the owned queue before erroring out.
            unsafe { release(queue) };
            return Err(VokraError::BackendUnavailable(format!(
                "MSL GEMM shader failed to compile: {detail}"
            )));
        }

        // Fetch the kernel function by name, then build the pipeline.
        // SAFETY: `newFunctionWithName:` takes an NSString built from the
        // kernel's name; `newComputePipelineStateWithFunction:error:` consumes
        // it. `library` / `function` are released afterwards (owned transients).
        let (pipeline, perr) = unsafe {
            let fname_c = c"vokra_gemm_f32";
            let fname = sys::send_id_cstr(
                sys::class(b"NSString\0"),
                sys::sel(b"stringWithUTF8String:\0"),
                fname_c.as_ptr(),
            );
            let function = sys::send_id_id(library, sys::sel(b"newFunctionWithName:\0"), fname);
            if function.is_null() {
                release(library);
                release(queue);
                return Err(VokraError::BackendUnavailable(
                    "MTLLibrary has no function named vokra_gemm_f32".to_owned(),
                ));
            }
            let mut perr: Id = core::ptr::null_mut();
            let pipeline = sys::send_new_pipeline(
                device,
                sys::sel(b"newComputePipelineStateWithFunction:error:\0"),
                function,
                &mut perr,
            );
            release(function);
            release(library);
            (pipeline, perr)
        };

        if pipeline.is_null() {
            // SAFETY: `perr` is null or a valid autoreleased NSError.
            let detail = unsafe { error_description(perr) };
            // SAFETY: release the owned queue before erroring out.
            unsafe { release(queue) };
            return Err(VokraError::BackendUnavailable(format!(
                "compute pipeline creation failed: {detail}"
            )));
        }

        Ok(MetalContext {
            device,
            queue,
            gemm_pipeline: pipeline,
        })
    }

    /// Row-major FP32 GEMM on the GPU with optional per-column bias:
    /// `out[i, j] = bias[j] + Σ_l a[i, l] · b[l, j]`.
    ///
    /// `a` is `m×k`, `b` is `k×n`, `out` is `m×n`, and `bias` (when `Some`) has
    /// length `n` — the exact contract of
    /// `vokra_backend_cpu::kernels::gemm_f32`, so the two are differentially
    /// comparable (M2-01-T18).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any shape mismatch or a zero
    /// dimension; [`VokraError::BackendUnavailable`] if a Metal buffer /
    /// command object cannot be created or the command buffer reports an error.
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

        // Bracket the GPU work in an autorelease pool so the autoreleased
        // command buffer / encoder / any NSError drain here rather than leaking
        // until some outer pool (there is none on a plain worker thread).
        // SAFETY: `objc_autoreleasePoolPush` returns a token consumed by the one
        // matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_gemm(m, n, k, a, b, bias, out);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    /// GEMM body: allocate shared buffers, encode + run, read back. Runs inside
    /// `gemm_f32`'s autorelease pool. Shapes are already validated.
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
        // Input buffers copy host data into shared storage (Apple silicon: one
        // physical pool, so the later `contents` readback is copy-free). A
        // failed alloc `?`-returns; already-built `OwnedBuf`s release on drop.
        let a_buf = self.new_buffer_from_slice(a)?;
        let b_buf = self.new_buffer_from_slice(b)?;

        // Bias buffer: the real bias when present, else a 1-float dummy the
        // kernel never reads (has_bias = 0). Always bound so buffer(2) is set.
        let dummy = [0.0f32];
        let bias_slice = bias.unwrap_or(&dummy);
        let bias_buf = self.new_buffer_from_slice(bias_slice)?;

        // Output buffer (uninitialised shared storage of m*n floats).
        let c_buf = self.new_buffer_output(out.len())?;

        let dims = GemmDims {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            has_bias: u32::from(bias.is_some()),
        };

        self.encode_and_run(&a_buf, &b_buf, &bias_buf, &c_buf, &dims, m, n)?;

        // SAFETY: `c_buf` is a valid shared buffer of `m*n` floats; after
        // `waitUntilCompleted` its `contents` pointer is host-readable.
        let contents = unsafe { sys::send_ptr(c_buf.0, sys::sel(b"contents\0")) } as *const f32;
        if contents.is_null() {
            return Err(VokraError::BackendUnavailable(
                "output MTLBuffer contents pointer is null".to_owned(),
            ));
        }
        // SAFETY: `contents` is the base of `out.len()` valid, non-overlapping
        // f32s in shared memory; copy them into the caller's slice.
        unsafe { core::ptr::copy_nonoverlapping(contents, out.as_mut_ptr(), out.len()) };
        Ok(())
    }

    /// Encodes and submits the GEMM, waiting for completion. Returns an error if
    /// the command buffer reports one.
    #[allow(clippy::too_many_arguments)] // encoder + four buffers + dims + m/n
    fn encode_and_run(
        &self,
        a_buf: &OwnedBuf,
        b_buf: &OwnedBuf,
        bias_buf: &OwnedBuf,
        c_buf: &OwnedBuf,
        dims: &GemmDims,
        m: usize,
        n: usize,
    ) -> Result<()> {
        // SAFETY: `queue` and `gemm_pipeline` are valid for the context's
        // lifetime; `commandBuffer` / `computeCommandEncoder` return
        // autoreleased objects (drained by the caller's pool). Each setter uses
        // the argument contract documented in `sys`; the four buffers are valid
        // and `dims` matches them. The two `MtlSize`s are passed per AAPCS64.
        unsafe {
            let cmd = sys::send_id(self.queue, sys::sel(b"commandBuffer\0"));
            if cmd.is_null() {
                return Err(VokraError::BackendUnavailable(
                    "MTLCommandQueue commandBuffer returned nil".to_owned(),
                ));
            }
            let enc = sys::send_id(cmd, sys::sel(b"computeCommandEncoder\0"));
            if enc.is_null() {
                return Err(VokraError::BackendUnavailable(
                    "MTLCommandBuffer computeCommandEncoder returned nil".to_owned(),
                ));
            }

            sys::send_void_id(
                enc,
                sys::sel(b"setComputePipelineState:\0"),
                self.gemm_pipeline,
            );
            let set_buffer = sys::sel(b"setBuffer:offset:atIndex:\0");
            sys::send_set_buffer(enc, set_buffer, a_buf.0, 0, 0);
            sys::send_set_buffer(enc, set_buffer, b_buf.0, 0, 1);
            sys::send_set_buffer(enc, set_buffer, bias_buf.0, 0, 2);
            sys::send_set_buffer(enc, set_buffer, c_buf.0, 0, 3);
            sys::send_set_bytes(
                enc,
                sys::sel(b"setBytes:length:atIndex:\0"),
                (dims as *const GemmDims).cast::<c_void>(),
                size_of::<GemmDims>(),
                4,
            );

            // Grid: x = columns (N), y = rows (M). 16x16 threadgroups; the
            // kernel guards row/col against M/N for ragged edges.
            const TG: usize = 16;
            let grid = MtlSize {
                width: n.div_ceil(TG),
                height: m.div_ceil(TG),
                depth: 1,
            };
            let tg = MtlSize {
                width: TG,
                height: TG,
                depth: 1,
            };
            sys::send_dispatch(
                enc,
                sys::sel(b"dispatchThreadgroups:threadsPerThreadgroup:\0"),
                grid,
                tg,
            );

            sys::send_void(enc, sys::sel(b"endEncoding\0"));
            sys::send_void(cmd, sys::sel(b"commit\0"));
            sys::send_void(cmd, sys::sel(b"waitUntilCompleted\0"));

            // Surface a GPU-side execution error explicitly (no silent success).
            let cmd_err = sys::send_id(cmd, sys::sel(b"error\0"));
            if !cmd_err.is_null() {
                let detail = error_description(cmd_err);
                return Err(VokraError::BackendUnavailable(format!(
                    "GEMM command buffer failed: {detail}"
                )));
            }
            Ok(())
        }
    }

    /// Allocates a shared-storage `MTLBuffer` initialised from `data`.
    ///
    /// A safe wrapper: `data` is a valid slice, so its pointer is valid for
    /// `size_of_val(data)` bytes, which is what `newBufferWithBytes:` copies.
    fn new_buffer_from_slice(&self, data: &[f32]) -> Result<OwnedBuf> {
        let bytes = size_of_val(data).max(size_of::<f32>());
        // SAFETY: `device` is valid; `data.as_ptr()` is valid for
        // `size_of_val(data)` bytes; shared storage mode (0). +1-owned buffer.
        let buf = unsafe {
            sys::send_new_buffer_bytes(
                self.device,
                sys::sel(b"newBufferWithBytes:length:options:\0"),
                data.as_ptr().cast::<c_void>(),
                bytes,
                sys::STORAGE_MODE_SHARED,
            )
        };
        if buf.is_null() {
            return Err(VokraError::BackendUnavailable(
                "MTLDevice newBufferWithBytes returned nil".to_owned(),
            ));
        }
        Ok(OwnedBuf(buf))
    }

    /// Allocates an uninitialised shared-storage `MTLBuffer` of `len` f32s.
    fn new_buffer_output(&self, len: usize) -> Result<OwnedBuf> {
        let bytes = (len * size_of::<f32>()).max(size_of::<f32>());
        // SAFETY: `device` is valid; shared storage mode (0). +1-owned buffer.
        let buf = unsafe {
            sys::send_new_buffer_len(
                self.device,
                sys::sel(b"newBufferWithLength:options:\0"),
                bytes,
                sys::STORAGE_MODE_SHARED,
            )
        };
        if buf.is_null() {
            return Err(VokraError::BackendUnavailable(
                "MTLDevice newBufferWithLength returned nil".to_owned(),
            ));
        }
        Ok(OwnedBuf(buf))
    }
}

impl Drop for MetalContext {
    fn drop(&mut self) {
        // SAFETY: the three handles are valid `+1`-owned objects created in
        // `new` / `build`; release each exactly once.
        unsafe {
            release(self.gemm_pipeline);
            release(self.queue);
            release(self.device);
        }
    }
}

impl core::fmt::Debug for MetalContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalContext").finish_non_exhaustive()
    }
}

/// Sends `-release` to a non-null owned object.
///
/// # Safety
/// `obj` must be a valid `+1`-owned Objective-C object (or null).
#[inline]
unsafe fn release(obj: Id) {
    if !obj.is_null() {
        // SAFETY: `obj` is a valid owned object per the caller contract.
        unsafe { sys::send_void(obj, sys::sel(b"release\0")) };
    }
}

/// Extracts `-[NSError localizedDescription]` as a String (best effort).
///
/// # Safety
/// `err` must be null or a valid `NSError`.
unsafe fn error_description(err: Id) -> String {
    if err.is_null() {
        return "(no error object)".to_owned();
    }
    // SAFETY: `localizedDescription` is a valid `-(NSString*)` selector on
    // NSError; the result is autoreleased and read within the caller's pool.
    let desc = unsafe { sys::send_id(err, sys::sel(b"localizedDescription\0")) };
    // SAFETY: `desc` is null or a valid NSString.
    unsafe { sys::nsstring_to_string(desc) }.unwrap_or_else(|| "(no description)".to_owned())
}

// ---- shape validation (mirrors vokra-backend-cpu's gemm validator) ----

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
