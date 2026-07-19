//! Live WebGPU dispatch context (M4-01-T10/T12〜T15; wasm32 + feature
//! `webgpu` only).
//!
//! Executes [`crate::plan::KernelPlan`]s through the extern-import shim
//! ([`crate::sys`]): pipeline cache → buffer create/write → dispatch →
//! readback → destroy. The public methods mirror the
//! `vokra_backend_cpu::kernels::*` signatures exactly so the `vokra-models`
//! `Compute` seam can swap arms without reshaping call sites (the
//! `MetalContext` precedent).
//!
//! # Per-op readback (honest scope note)
//!
//! Each seam method uploads its inputs, dispatches once, and reads the
//! output back — the M2-01 "per-op GPU" stage. Whole-run device residency
//! (readback 6N+1 → 1) is what the `buffer_read`-at-run-boundary shim
//! contract enables, but wiring the Whisper encoder chain device-resident is
//! the M4-02+ follow-up recorded in the WP hand-over; the browser parity
//! harness measures the per-op mode as-is (no fabricated performance
//! claims). Buffer pooling likewise: buffers are created per call today
//! (session-lifetime pre-allocation is the FR-EX-05 follow-up, M3-02-T25
//! posture).
//!
//! # `!Send` by construction
//!
//! Handles index glue-side tables owned by the worker's JS realm, so a
//! context must not cross threads. `PhantomData<*const ()>` makes the type
//! `!Send + !Sync` (same idiom as `MetalContext`).

use core::marker::PhantomData;
use std::cell::RefCell;
use std::collections::HashMap;

use vokra_core::{Result, VokraError};

use crate::plan::{self, ActivationKind, ElementwiseOp, KernelPlan};
use crate::sys;
use crate::wgsl;

/// A live WebGPU device context reached through the JS glue.
pub struct WebGpuContext {
    /// name → pipeline handle (glue-side object) cache.
    pipelines: RefCell<HashMap<&'static str, u32>>,
    /// `!Send + !Sync`: glue handles are realm-affine.
    _not_send: PhantomData<*const ()>,
}

/// RAII wrapper so buffers are destroyed on every exit path.
struct GpuBuffer(u32);

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        // SAFETY: sys.rs import contract — destroy is idempotent and ignores
        // unknown handles.
        unsafe { sys::vokra_webgpu_buffer_destroy(self.0) };
    }
}

fn glue_err(what: &str) -> VokraError {
    VokraError::BackendUnavailable(format!(
        "webgpu glue call failed ({what}): {}",
        sys::last_glue_error()
    ))
}

impl WebGpuContext {
    /// Opens the context: probes the adapter/device (explicit
    /// [`VokraError::BackendUnavailable`] when absent — FR-EX-08).
    ///
    /// # Errors
    ///
    /// Propagates [`crate::probe::vokra_webgpu_probe`] failures.
    pub fn new() -> Result<Self> {
        let caps = crate::probe::vokra_webgpu_probe()?;
        debug_assert!(caps.adapter_ready);
        Ok(WebGpuContext {
            pipelines: RefCell::new(HashMap::new()),
            _not_send: PhantomData,
        })
    }

    /// Cached pipeline lookup; creates shader module + pipeline on first
    /// use of each kernel.
    fn pipeline(&self, name: &'static str) -> Result<u32> {
        if let Some(&p) = self.pipelines.borrow().get(name) {
            return Ok(p);
        }
        let shader = wgsl::get(name).ok_or_else(|| {
            VokraError::UnsupportedOp(format!(
                "webgpu backend has no WGSL kernel named `{name}` (no silent CPU fallback, \
                 FR-EX-08)"
            ))
        })?;
        // SAFETY: sys.rs import contract — the name/source pointers are live
        // borrows of 'static strs in linear memory with exact lengths.
        let module = unsafe {
            sys::vokra_webgpu_shader_create(
                shader.name.as_ptr(),
                shader.name.len() as u32,
                shader.source.as_ptr(),
                shader.source.len() as u32,
            )
        };
        if module == 0 {
            return Err(glue_err("shader_create"));
        }
        // SAFETY: sys.rs import contract — entry-point pointer/length are a
        // live 'static str borrow.
        let pipeline = unsafe {
            sys::vokra_webgpu_pipeline_create(
                module,
                shader.entry_point.as_ptr(),
                shader.entry_point.len() as u32,
            )
        };
        if pipeline == 0 {
            return Err(glue_err("pipeline_create"));
        }
        self.pipelines.borrow_mut().insert(name, pipeline);
        Ok(pipeline)
    }

    fn upload(&self, data: &[f32]) -> Result<GpuBuffer> {
        let bytes = core::mem::size_of_val(data) as u32;
        // SAFETY: sys.rs import contract — size is the exact byte length of
        // the live slice below.
        let buf = unsafe { sys::vokra_webgpu_buffer_create(bytes, sys::USAGE_STORAGE_INPUT) };
        if buf == 0 {
            return Err(glue_err("buffer_create(input)"));
        }
        let buf = GpuBuffer(buf);
        // SAFETY: sys.rs import contract — `data` is a live borrow; the glue
        // copies `bytes` bytes out of linear memory at the given offset.
        let rc =
            unsafe { sys::vokra_webgpu_buffer_write(buf.0, 0, data.as_ptr().cast::<u8>(), bytes) };
        if rc != 0 {
            return Err(glue_err("buffer_write"));
        }
        Ok(buf)
    }

    fn alloc_output(&self, len: usize) -> Result<GpuBuffer> {
        let bytes = (len * core::mem::size_of::<f32>()) as u32;
        // SAFETY: sys.rs import contract.
        let buf = unsafe { sys::vokra_webgpu_buffer_create(bytes, sys::USAGE_STORAGE_OUTPUT) };
        if buf == 0 {
            return Err(glue_err("buffer_create(output)"));
        }
        Ok(GpuBuffer(buf))
    }

    fn dispatch(&self, plan: &KernelPlan, bufs: &[u32]) -> Result<()> {
        debug_assert_eq!(bufs.len() as u32, plan.n_storage_buffers);
        let pipeline = self.pipeline(plan.shader)?;
        // SAFETY: sys.rs import contract — `bufs` and `plan.uniform` are
        // live borrows with exact lengths; the glue binds storage buffers
        // 0..len and the uniform at index len (the plan bind contract).
        let rc = unsafe {
            sys::vokra_webgpu_dispatch(
                pipeline,
                bufs.as_ptr(),
                bufs.len() as u32,
                plan.uniform.as_ptr(),
                plan.uniform.len() as u32,
                plan.workgroups[0],
                plan.workgroups[1],
                plan.workgroups[2],
            )
        };
        if rc != 0 {
            return Err(glue_err(plan.shader));
        }
        Ok(())
    }

    fn read_back(&self, buf: &GpuBuffer, out: &mut [f32]) -> Result<()> {
        let bytes = core::mem::size_of_val(out) as u32;
        // SAFETY: sys.rs import contract — `out` is a live mutable borrow of
        // exactly `bytes` bytes; the glue writes through linear memory at
        // that offset (synchronous via the SAB bridge).
        let rc = unsafe {
            sys::vokra_webgpu_buffer_read(buf.0, 0, out.as_mut_ptr().cast::<u8>(), bytes)
        };
        if rc != 0 {
            return Err(glue_err("buffer_read"));
        }
        Ok(())
    }

    /// upload-inputs → dispatch → readback helper for the seam methods.
    fn run(&self, plan: &KernelPlan, inputs: &[&[f32]], out: &mut [f32]) -> Result<()> {
        let mut handles: Vec<GpuBuffer> = Vec::with_capacity(inputs.len() + 1);
        for input in inputs {
            handles.push(self.upload(input)?);
        }
        let out_buf = self.alloc_output(out.len())?;
        let mut ids: Vec<u32> = handles.iter().map(|h| h.0).collect();
        ids.push(out_buf.0);
        self.dispatch(plan, &ids)?;
        self.read_back(&out_buf, out)?;
        Ok(())
        // GpuBuffer Drop destroys every buffer on all paths.
    }

    /// Wall-clock milliseconds from the embedder (`performance.now()`), the
    /// wasm32 RTF measurement hook (M4-01-T24; `std::time::Instant` panics
    /// on wasm32-unknown-unknown).
    #[must_use]
    pub fn now_ms(&self) -> f64 {
        // SAFETY: sys.rs import contract — no pointers, plain f64 return.
        unsafe { sys::vokra_webgpu_now_ms() }
    }

    // -- seam methods (signatures mirror vokra_backend_cpu::kernels) --------

    /// Row-major GEMM with optional per-column bias (see
    /// `kernels::gemm_f32`).
    ///
    /// # Errors
    /// Shape mismatches are [`VokraError::InvalidArgument`]; glue/device
    /// failures are [`VokraError::BackendUnavailable`].
    #[allow(clippy::too_many_arguments)] // intrinsic GEMM parameter set (matches kernels::gemm_f32)
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
        plan::expect_lens(&[
            ("gemm a", a.len(), m * k),
            ("gemm b", b.len(), k * n),
            ("gemm out", out.len(), m * n),
        ])?;
        if let Some(bias) = bias {
            plan::expect_lens(&[("gemm bias", bias.len(), n)])?;
        }
        let plan = plan::plan_gemm(m, n, k, bias.is_some())?;
        let dummy = [0.0f32];
        let bias_slice = bias.unwrap_or(&dummy);
        self.run(&plan, &[a, b, bias_slice], out)
    }

    /// Row-major matrix-vector product with optional per-row bias (see
    /// `kernels::gemv_f32`).
    ///
    /// # Errors
    /// As [`Self::gemm_f32`].
    pub fn gemv_f32(
        &self,
        m: usize,
        k: usize,
        a: &[f32],
        x: &[f32],
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        plan::expect_lens(&[
            ("gemv a", a.len(), m * k),
            ("gemv x", x.len(), k),
            ("gemv out", out.len(), m),
        ])?;
        if let Some(bias) = bias {
            plan::expect_lens(&[("gemv bias", bias.len(), m)])?;
        }
        let plan = plan::plan_gemv(m, k, bias.is_some())?;
        let dummy = [0.0f32];
        let bias_slice = bias.unwrap_or(&dummy);
        self.run(&plan, &[a, x, bias_slice], out)
    }

    /// Row-wise numerically-stable softmax (see `kernels::softmax_f32`).
    ///
    /// # Errors
    /// As [`Self::gemm_f32`].
    pub fn softmax_f32(
        &self,
        input: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        plan::expect_lens(&[
            ("softmax input", input.len(), rows * cols),
            ("softmax out", out.len(), rows * cols),
        ])?;
        let plan = plan::plan_softmax(rows, cols)?;
        self.run(&plan, &[input], out)
    }

    /// Causal-masked row softmax (row `r` sees columns `c <= r + offset`) —
    /// the host-mask + softmax equivalence kernel (M2-01 Phase 1 posture).
    ///
    /// # Errors
    /// As [`Self::gemm_f32`].
    pub fn softmax_causal_f32(
        &self,
        input: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        offset: usize,
    ) -> Result<()> {
        plan::expect_lens(&[
            ("softmax_causal input", input.len(), rows * cols),
            ("softmax_causal out", out.len(), rows * cols),
        ])?;
        let plan = plan::plan_softmax_causal(rows, cols, offset)?;
        self.run(&plan, &[input], out)
    }

    /// Affine layer norm; `eps` comes from the model config (see
    /// `kernels::layer_norm_f32`).
    ///
    /// # Errors
    /// As [`Self::gemm_f32`].
    #[allow(clippy::too_many_arguments)] // intrinsic layer-norm parameter set (matches kernels::layer_norm_f32)
    pub fn layer_norm_f32(
        &self,
        input: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<()> {
        plan::expect_lens(&[
            ("layer_norm input", input.len(), rows * cols),
            ("layer_norm out", out.len(), rows * cols),
            ("layer_norm gamma", gamma.len(), cols),
            ("layer_norm beta", beta.len(), cols),
        ])?;
        let plan = plan::plan_layer_norm(rows, cols, eps)?;
        self.run(&plan, &[input, gamma, beta], out)
    }

    /// Element-wise exact (erf/A&S 7.1.26) GELU (see `kernels::gelu_f32`).
    ///
    /// # Errors
    /// As [`Self::gemm_f32`].
    pub fn gelu_f32(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        plan::expect_lens(&[("gelu out", out.len(), x.len())])?;
        let plan = plan::plan_gelu(x.len())?;
        self.run(&plan, &[x], out)
    }

    /// 1-D convolution, Whisper stem envelope (see `kernels::conv1d_f32`).
    ///
    /// # Errors
    /// As [`Self::gemm_f32`].
    #[allow(clippy::too_many_arguments)] // convolution's intrinsic parameter set (matches kernels::conv1d_f32)
    pub fn conv1d_f32(
        &self,
        input: &[f32],
        in_ch: usize,
        in_len: usize,
        weight: &[f32],
        out_ch: usize,
        kernel: usize,
        bias: Option<&[f32]>,
        stride: usize,
        padding: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let out_len = plan::conv1d_out_len(in_len, kernel, stride, padding)?;
        plan::expect_lens(&[
            ("conv1d input", input.len(), in_ch * in_len),
            ("conv1d weight", weight.len(), out_ch * in_ch * kernel),
            ("conv1d out", out.len(), out_ch * out_len),
        ])?;
        if let Some(bias) = bias {
            plan::expect_lens(&[("conv1d bias", bias.len(), out_ch)])?;
        }
        let plan = plan::plan_conv1d(
            in_ch,
            in_len,
            out_ch,
            kernel,
            stride,
            padding,
            bias.is_some(),
        )?;
        let dummy = [0.0f32];
        let bias_slice = bias.unwrap_or(&dummy);
        self.run(&plan, &[input, weight, bias_slice], out)
    }

    /// Element-wise binary op (add / mul) through the `elementwise` kernel.
    ///
    /// # Errors
    /// As [`Self::gemm_f32`].
    pub fn elementwise_f32(
        &self,
        op: ElementwiseOp,
        a: &[f32],
        b: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        plan::expect_lens(&[
            ("elementwise b", b.len(), a.len()),
            ("elementwise out", out.len(), a.len()),
        ])?;
        let plan = plan::plan_elementwise(op, a.len())?;
        self.run(&plan, &[a, b], out)
    }

    /// Element-wise sum `out[i] = a[i] + b[i]` through the **dedicated**
    /// `add_f32` kernel — the `OpKind::Add` graph arm's kernel (mirrors the
    /// Vulkan hand-crafted `add_f32` arm; distinct from the `elementwise`
    /// op-switch kernel, which backs `OpKind::Mul`). Keeping Add on its own
    /// kernel is what makes the dispatch match the `supports()` gate token
    /// `add_f32` in lock-step (M4-01 #23:
    /// [`crate::backend::graph_op_dispatched_shader`]).
    ///
    /// # Errors
    /// As [`Self::gemm_f32`].
    pub fn add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        plan::expect_lens(&[
            ("add_f32 b", b.len(), a.len()),
            ("add_f32 out", out.len(), a.len()),
        ])?;
        let plan = plan::plan_add(a.len())?;
        self.run(&plan, &[a, b], out)
    }

    /// Identity copy through the `copy_f32` kernel — the round-trip smoke op
    /// (M4-01-T10/T11).
    ///
    /// # Errors
    /// As [`Self::gemm_f32`].
    pub fn copy_f32(&self, src: &[f32], out: &mut [f32]) -> Result<()> {
        plan::expect_lens(&[("copy out", out.len(), src.len())])?;
        let plan = plan::plan_copy(src.len())?;
        self.run(&plan, &[src], out)
    }

    /// Element-wise activation (relu / sigmoid / tanh).
    ///
    /// # Errors
    /// As [`Self::gemm_f32`].
    pub fn activation_f32(&self, kind: ActivationKind, x: &[f32], out: &mut [f32]) -> Result<()> {
        plan::expect_lens(&[("activation out", out.len(), x.len())])?;
        let plan = plan::plan_activation(kind, x.len())?;
        self.run(&plan, &[x], out)
    }
}
