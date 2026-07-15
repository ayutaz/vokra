//! Host-portable kernel plans (M4-01: the WebGPU analogue of
//! `vokra-backend-vulkan/src/plan.rs`).
//!
//! A [`KernelPlan`] is everything a dispatch needs *except* the live GPU
//! objects: shader name, workgroup grid, packed uniform bytes, and the
//! storage-buffer bind arity. Building one performs the **shape validation**
//! for the op (explicit [`VokraError::InvalidArgument`] on any mismatch —
//! NFR-RL-07 boundary rule), so this module is the host-portable, natively
//! unit-tested half of the dispatch chain; `context.rs` (wasm32-only)
//! executes plans through the extern-import shim, and the native test suite
//! exercises the plan surface as the headless "mock dispatch" check the
//! M4-01 spec asks for (workgroup math, uniform packing, bind arity, and the
//! WGSL `@workgroup_size` lock-step below).
//!
//! # Bind contract (shared with the JS glue)
//!
//! Storage buffers bind at indices `0..n_storage_buffers` in plan order with
//! the **output always last**; the packed uniform binds at index
//! `n_storage_buffers`. `wgsl.rs`'s structural test pins the WGSL side of
//! this contract; [`tests::plans_match_manifest_arity`] pins the plan side.
//!
//! # Uniform layout
//!
//! Fields are packed little-endian in WGSL struct order and padded to a
//! 16-byte multiple (safe against strict uniform-binding size validators).
//! wasm32 is little-endian, and WebGPU buffer views are host-endian, so the
//! packed bytes land in the GPU-visible layout WGSL expects.

use vokra_core::{Result, VokraError};

/// Workgroup x-size of the 1-D element-wise kernels
/// (`copy_f32` / `add_f32` / `elementwise` / `gelu` / `activation`).
pub const WG_ELEMENTWISE: u32 = 256;
/// Square tile edge of `gemm_f32` (`@workgroup_size(16, 16)`).
pub const GEMM_TILE: u32 = 16;
/// Workgroup x-size of `gemv_f32` (one workgroup per output row).
pub const WG_GEMV: u32 = 64;
/// Workgroup x-size of the row-reduction kernels
/// (`softmax` / `softmax_causal` / `layer_norm` — one workgroup per row).
pub const WG_ROW_REDUCE: u32 = 256;
/// Workgroup x-size of `conv1d` (output positions per workgroup).
pub const WG_CONV1D: u32 = 64;

/// WebGPU's guaranteed per-dimension `dispatchWorkgroups` limit
/// (`maxComputeWorkgroupsPerDimension` default = 65535). Exceeding it is an
/// explicit host-side error, never a truncated dispatch.
pub const MAX_WORKGROUPS_PER_DIM: u32 = 65_535;

/// Element-wise binary op selector for the `elementwise` kernel (uniform
/// flag, matching the WGSL `params.op` encoding).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementwiseOp {
    /// `out[i] = a[i] + b[i]` (op flag 0).
    Add = 0,
    /// `out[i] = a[i] * b[i]` (op flag 1).
    Mul = 1,
}

/// Activation selector for the `activation` kernel (uniform flag, matching
/// the WGSL `params.kind` encoding).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationKind {
    /// `max(v, 0)` (kind 0).
    Relu = 0,
    /// `1 / (1 + exp(-v))` (kind 1).
    Sigmoid = 1,
    /// `tanh(v)` (kind 2).
    Tanh = 2,
}

/// A validated, ready-to-dispatch kernel invocation (host-portable half).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelPlan {
    /// Manifest key into [`crate::wgsl::SHADERS`] (also the glue pipeline
    /// cache key).
    pub shader: &'static str,
    /// Workgroup grid for `dispatchWorkgroups(x, y, z)`.
    pub workgroups: [u32; 3],
    /// Packed uniform bytes (little-endian, 16-byte padded).
    pub uniform: Vec<u8>,
    /// Number of storage buffers the kernel binds (output last); the
    /// uniform binds at this index.
    pub n_storage_buffers: u32,
}

/// Little-endian uniform packer (WGSL struct order).
#[derive(Default)]
struct UniformPacker {
    bytes: Vec<u8>,
}

impl UniformPacker {
    fn u32(mut self, v: u32) -> Self {
        self.bytes.extend_from_slice(&v.to_le_bytes());
        self
    }

    fn f32(mut self, v: f32) -> Self {
        self.bytes.extend_from_slice(&v.to_le_bytes());
        self
    }

    /// Pads to a 16-byte multiple and finishes.
    fn finish(mut self) -> Vec<u8> {
        while self.bytes.len() % 16 != 0 {
            self.bytes.push(0);
        }
        self.bytes
    }
}

fn ceil_div(a: usize, b: u32) -> Result<u32> {
    let b = b as usize;
    let groups = a.div_ceil(b);
    let groups32 = u32::try_from(groups).map_err(|_| {
        VokraError::InvalidArgument(format!("workgroup count {groups} exceeds u32"))
    })?;
    if groups32 > MAX_WORKGROUPS_PER_DIM {
        return Err(VokraError::InvalidArgument(format!(
            "workgroup count {groups32} exceeds the WebGPU per-dimension limit \
             {MAX_WORKGROUPS_PER_DIM} — split the dispatch host-side (explicit error, never a \
             truncated dispatch)"
        )));
    }
    Ok(groups32)
}

fn dim_u32(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v)
        .map_err(|_| VokraError::InvalidArgument(format!("{what} = {v} exceeds u32 (wasm32)")))
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

fn nonzero(v: usize, what: &str) -> Result<()> {
    if v == 0 {
        Err(VokraError::InvalidArgument(format!(
            "{what} must be non-zero"
        )))
    } else {
        Ok(())
    }
}

/// `copy_f32`: identity copy of `n` elements (src, dst).
pub fn plan_copy(n: usize) -> Result<KernelPlan> {
    nonzero(n, "copy n")?;
    Ok(KernelPlan {
        shader: "copy_f32",
        workgroups: [ceil_div(n, WG_ELEMENTWISE)?, 1, 1],
        uniform: UniformPacker::default().u32(dim_u32(n, "copy n")?).finish(),
        n_storage_buffers: 2,
    })
}

/// `add_f32`: element-wise sum of `n` elements (a, b, out).
pub fn plan_add(n: usize) -> Result<KernelPlan> {
    nonzero(n, "add n")?;
    Ok(KernelPlan {
        shader: "add_f32",
        workgroups: [ceil_div(n, WG_ELEMENTWISE)?, 1, 1],
        uniform: UniformPacker::default().u32(dim_u32(n, "add n")?).finish(),
        n_storage_buffers: 3,
    })
}

/// `elementwise`: binary op switch over `n` elements (a, b, out).
pub fn plan_elementwise(op: ElementwiseOp, n: usize) -> Result<KernelPlan> {
    nonzero(n, "elementwise n")?;
    Ok(KernelPlan {
        shader: "elementwise",
        workgroups: [ceil_div(n, WG_ELEMENTWISE)?, 1, 1],
        uniform: UniformPacker::default()
            .u32(dim_u32(n, "elementwise n")?)
            .u32(op as u32)
            .finish(),
        n_storage_buffers: 3,
    })
}

/// `gemm_f32`: `out[m,n] = bias?[n] + a[m,k] @ b[k,n]` (a, b, bias, out).
pub fn plan_gemm(m: usize, n: usize, k: usize, use_bias: bool) -> Result<KernelPlan> {
    nonzero(m, "gemm m")?;
    nonzero(n, "gemm n")?;
    nonzero(k, "gemm k")?;
    Ok(KernelPlan {
        shader: "gemm_f32",
        workgroups: [ceil_div(n, GEMM_TILE)?, ceil_div(m, GEMM_TILE)?, 1],
        uniform: UniformPacker::default()
            .u32(dim_u32(m, "gemm m")?)
            .u32(dim_u32(n, "gemm n")?)
            .u32(dim_u32(k, "gemm k")?)
            .u32(u32::from(use_bias))
            .finish(),
        n_storage_buffers: 4,
    })
}

/// `gemv_f32`: `out[m] = bias?[m] + a[m,k] @ x[k]` (a, x, bias, out).
pub fn plan_gemv(m: usize, k: usize, use_bias: bool) -> Result<KernelPlan> {
    nonzero(m, "gemv m")?;
    nonzero(k, "gemv k")?;
    Ok(KernelPlan {
        shader: "gemv_f32",
        // One workgroup per output row.
        workgroups: [ceil_div(m, 1)?, 1, 1],
        uniform: UniformPacker::default()
            .u32(dim_u32(m, "gemv m")?)
            .u32(dim_u32(k, "gemv k")?)
            .u32(u32::from(use_bias))
            .finish(),
        n_storage_buffers: 4,
    })
}

/// `softmax`: row softmax over `rows x cols` (x, out).
pub fn plan_softmax(rows: usize, cols: usize) -> Result<KernelPlan> {
    nonzero(rows, "softmax rows")?;
    nonzero(cols, "softmax cols")?;
    Ok(KernelPlan {
        shader: "softmax",
        workgroups: [ceil_div(rows, 1)?, 1, 1],
        uniform: UniformPacker::default()
            .u32(dim_u32(rows, "softmax rows")?)
            .u32(dim_u32(cols, "softmax cols")?)
            .finish(),
        n_storage_buffers: 2,
    })
}

/// `softmax_causal`: causal-masked row softmax; row `r` sees columns
/// `c <= r + offset` (x, out).
pub fn plan_softmax_causal(rows: usize, cols: usize, offset: usize) -> Result<KernelPlan> {
    nonzero(rows, "softmax_causal rows")?;
    nonzero(cols, "softmax_causal cols")?;
    Ok(KernelPlan {
        shader: "softmax_causal",
        workgroups: [ceil_div(rows, 1)?, 1, 1],
        uniform: UniformPacker::default()
            .u32(dim_u32(rows, "softmax_causal rows")?)
            .u32(dim_u32(cols, "softmax_causal cols")?)
            .u32(dim_u32(offset, "softmax_causal offset")?)
            .finish(),
        n_storage_buffers: 2,
    })
}

/// `layer_norm`: affine layer norm over `rows x cols`; `eps` comes from the
/// model config (never invented — M4-01 spec T14) (x, gamma, beta, out).
pub fn plan_layer_norm(rows: usize, cols: usize, eps: f32) -> Result<KernelPlan> {
    nonzero(rows, "layer_norm rows")?;
    nonzero(cols, "layer_norm cols")?;
    Ok(KernelPlan {
        shader: "layer_norm",
        workgroups: [ceil_div(rows, 1)?, 1, 1],
        uniform: UniformPacker::default()
            .u32(dim_u32(rows, "layer_norm rows")?)
            .u32(dim_u32(cols, "layer_norm cols")?)
            .f32(eps)
            .finish(),
        n_storage_buffers: 4,
    })
}

/// `gelu`: element-wise exact (erf/A&S-7.1.26) GELU over `n` elements
/// (x, out).
pub fn plan_gelu(n: usize) -> Result<KernelPlan> {
    nonzero(n, "gelu n")?;
    Ok(KernelPlan {
        shader: "gelu",
        workgroups: [ceil_div(n, WG_ELEMENTWISE)?, 1, 1],
        uniform: UniformPacker::default().u32(dim_u32(n, "gelu n")?).finish(),
        n_storage_buffers: 2,
    })
}

/// `activation`: element-wise relu / sigmoid / tanh over `n` elements
/// (x, out).
pub fn plan_activation(kind: ActivationKind, n: usize) -> Result<KernelPlan> {
    nonzero(n, "activation n")?;
    Ok(KernelPlan {
        shader: "activation",
        workgroups: [ceil_div(n, WG_ELEMENTWISE)?, 1, 1],
        uniform: UniformPacker::default()
            .u32(dim_u32(n, "activation n")?)
            .u32(kind as u32)
            .finish(),
        n_storage_buffers: 2,
    })
}

/// The `(in_len + 2*padding - kernel) / stride + 1` output-length rule the
/// CPU kernel uses (`vokra-backend-cpu` conv1d docs). Exposed for callers
/// sizing the output buffer.
pub fn conv1d_out_len(
    in_len: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<usize> {
    nonzero(kernel, "conv1d kernel")?;
    nonzero(stride, "conv1d stride")?;
    let padded = in_len + 2 * padding;
    if padded < kernel {
        return Err(VokraError::InvalidArgument(format!(
            "conv1d: padded length {padded} shorter than kernel {kernel}"
        )));
    }
    Ok((padded - kernel) / stride + 1)
}

/// `conv1d`: Whisper-stem 1-D convolution (x, w, bias, out).
pub fn plan_conv1d(
    in_ch: usize,
    in_len: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    use_bias: bool,
) -> Result<KernelPlan> {
    nonzero(in_ch, "conv1d in_ch")?;
    nonzero(in_len, "conv1d in_len")?;
    nonzero(out_ch, "conv1d out_ch")?;
    let out_len = conv1d_out_len(in_len, kernel, stride, padding)?;
    Ok(KernelPlan {
        shader: "conv1d",
        workgroups: [ceil_div(out_len, WG_CONV1D)?, ceil_div(out_ch, 1)?, 1],
        uniform: UniformPacker::default()
            .u32(dim_u32(in_ch, "conv1d in_ch")?)
            .u32(dim_u32(in_len, "conv1d in_len")?)
            .u32(dim_u32(out_ch, "conv1d out_ch")?)
            .u32(dim_u32(kernel, "conv1d kernel")?)
            .u32(dim_u32(stride, "conv1d stride")?)
            .u32(dim_u32(padding, "conv1d padding")?)
            .u32(dim_u32(out_len, "conv1d out_len")?)
            .u32(u32::from(use_bias))
            .finish(),
        n_storage_buffers: 4,
    })
}

/// Validates a plan's input/output slice lengths against the caller's
/// buffers — the boundary check `context.rs` runs before creating GPU
/// buffers (kept here so the native tests can exercise it).
pub fn expect_lens(pairs: &[(&str, usize, usize)]) -> Result<()> {
    for (name, got, want) in pairs {
        expect_len(name, *got, *want)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl;

    #[test]
    fn workgroup_math_is_ceil_div() {
        assert_eq!(plan_copy(1).unwrap().workgroups, [1, 1, 1]);
        assert_eq!(plan_copy(256).unwrap().workgroups, [1, 1, 1]);
        assert_eq!(plan_copy(257).unwrap().workgroups, [2, 1, 1]);
        assert_eq!(plan_gemm(17, 33, 5, false).unwrap().workgroups, [3, 2, 1]);
        assert_eq!(plan_gemv(3, 1000, true).unwrap().workgroups, [3, 1, 1]);
        assert_eq!(plan_softmax(7, 9).unwrap().workgroups, [7, 1, 1]);
        // conv1d: whisper stem shape (k=3, s=2, p=1) halves the length.
        let p = plan_conv1d(80, 3000, 384, 3, 2, 1, true).unwrap();
        assert_eq!(p.workgroups, [ceil_div(1500, WG_CONV1D).unwrap(), 384, 1]);
    }

    #[test]
    fn uniform_packing_is_little_endian_and_16_byte_padded() {
        let p = plan_elementwise(ElementwiseOp::Mul, 5).unwrap();
        assert_eq!(p.uniform.len() % 16, 0);
        assert_eq!(&p.uniform[0..4], &5u32.to_le_bytes());
        assert_eq!(&p.uniform[4..8], &1u32.to_le_bytes());
        let p = plan_layer_norm(2, 4, 1e-5).unwrap();
        assert_eq!(&p.uniform[0..4], &2u32.to_le_bytes());
        assert_eq!(&p.uniform[4..8], &4u32.to_le_bytes());
        assert_eq!(&p.uniform[8..12], &1e-5f32.to_le_bytes());
        let p = plan_conv1d(2, 10, 3, 3, 1, 1, false).unwrap();
        // 8 fields * 4 bytes = 32 bytes, already 16-aligned.
        assert_eq!(p.uniform.len(), 32);
        assert_eq!(&p.uniform[24..28], &10u32.to_le_bytes()); // out_len (k=3,s=1,p=1 keeps len)
        assert_eq!(&p.uniform[28..32], &0u32.to_le_bytes()); // use_bias = 0
    }

    #[test]
    fn shape_validation_is_explicit_error() {
        assert!(matches!(plan_copy(0), Err(VokraError::InvalidArgument(_))));
        assert!(matches!(
            plan_gemm(0, 1, 1, false),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            conv1d_out_len(1, 5, 1, 1),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            plan_conv1d(1, 1, 1, 0, 1, 0, false),
            Err(VokraError::InvalidArgument(_))
        ));
        // Workgroup-per-dimension ceiling: explicit error, never truncation.
        assert!(matches!(
            plan_softmax(70_000, 4),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// conv1d_out_len matches the CPU kernel's documented formula on the
    /// M0-06 Whisper stem envelope (kernel 3; stride 1 and 2; padding 1).
    #[test]
    fn conv1d_out_len_matches_whisper_stem() {
        assert_eq!(conv1d_out_len(3000, 3, 1, 1).unwrap(), 3000); // conv1
        assert_eq!(conv1d_out_len(3000, 3, 2, 1).unwrap(), 1500); // conv2
    }

    /// Every plan's shader exists in the manifest and the storage-buffer
    /// arity matches the manifest (the glue bind contract, both sides).
    #[test]
    fn plans_match_manifest_arity() {
        let plans = vec![
            plan_copy(4).unwrap(),
            plan_add(4).unwrap(),
            plan_elementwise(ElementwiseOp::Add, 4).unwrap(),
            plan_gemm(2, 2, 2, true).unwrap(),
            plan_gemv(2, 2, false).unwrap(),
            plan_softmax(2, 2).unwrap(),
            plan_softmax_causal(2, 2, 0).unwrap(),
            plan_layer_norm(2, 2, 1e-5).unwrap(),
            plan_gelu(4).unwrap(),
            plan_conv1d(1, 8, 1, 3, 1, 1, true).unwrap(),
            plan_activation(ActivationKind::Relu, 4).unwrap(),
        ];
        for p in &plans {
            let shader = wgsl::get(p.shader)
                .unwrap_or_else(|| panic!("plan shader `{}` not in manifest", p.shader));
            assert_eq!(
                shader.n_storage_buffers, p.n_storage_buffers,
                "{}: plan/manifest storage-buffer arity drifted",
                p.shader
            );
            assert!(
                p.uniform.len() % 16 == 0,
                "{}: uniform not 16-padded",
                p.shader
            );
        }
        // All 11 manifest kernels have a plan above (drift the other way).
        assert_eq!(plans.len(), wgsl::SHADERS.len());
    }

    /// The plan-side workgroup constants match the literal
    /// `@workgroup_size(...)` in each WGSL source (drift gate between the
    /// dispatch math and the shader text).
    #[test]
    fn workgroup_constants_match_wgsl_sources() {
        let expect = [
            ("copy_f32", format!("@workgroup_size({WG_ELEMENTWISE})")),
            ("add_f32", format!("@workgroup_size({WG_ELEMENTWISE})")),
            ("elementwise", format!("@workgroup_size({WG_ELEMENTWISE})")),
            (
                "gemm_f32",
                format!("@workgroup_size({GEMM_TILE}, {GEMM_TILE})"),
            ),
            ("gemv_f32", format!("@workgroup_size({WG_GEMV})")),
            ("softmax", format!("@workgroup_size({WG_ROW_REDUCE})")),
            (
                "softmax_causal",
                format!("@workgroup_size({WG_ROW_REDUCE})"),
            ),
            ("layer_norm", format!("@workgroup_size({WG_ROW_REDUCE})")),
            ("gelu", format!("@workgroup_size({WG_ELEMENTWISE})")),
            ("conv1d", format!("@workgroup_size({WG_CONV1D})")),
            ("activation", format!("@workgroup_size({WG_ELEMENTWISE})")),
        ];
        for (name, needle) in expect {
            let src = wgsl::get(name).unwrap().source;
            assert!(
                src.contains(&needle),
                "{name}: WGSL text does not contain `{needle}` — plan constants drifted from \
                 the shader"
            );
        }
    }

    /// M4-01-T14 formula-transcription pin: re-evaluates the EXACT expression
    /// the gelu.wgsl kernel encodes (A&S 7.1.26 with the CPU kernel's
    /// coefficients) in Rust and diffs against `vokra_backend_cpu`'s
    /// dispatched `gelu_f32` over a dense sweep. Both sides use the same
    /// approximation, so the only free variable is the WGSL transcription
    /// itself — a sign/coefficient typo in the shader text would show up
    /// here without a GPU. Measured on this sweep the max |Δ| is 0.0 (the
    /// mirror reproduces the CPU chain exactly); the true GPU-side residual
    /// (driver `exp` rounding) is measured in the browser harness (T18) and
    /// bounded by atol = 0.01 (NFR-QL-01).
    #[test]
    fn gelu_wgsl_formula_transcription_matches_cpu_kernel() {
        // Mirror of gelu.wgsl's erf_approx + main body, kept in WGSL
        // evaluation order. Canonical A&S 7.1.26 constants kept verbatim
        // (auditable — the same allow the CPU kernel carries); excess digits
        // round to f32 harmlessly.
        #[allow(clippy::excessive_precision)]
        const ERF_P: f32 = 0.3275911;
        #[allow(clippy::excessive_precision)]
        const ERF_A1: f32 = 0.254829592;
        #[allow(clippy::excessive_precision)]
        const ERF_A2: f32 = -0.284496736;
        #[allow(clippy::excessive_precision)]
        const ERF_A3: f32 = 1.421413741;
        #[allow(clippy::excessive_precision)]
        const ERF_A4: f32 = -1.453152027;
        #[allow(clippy::excessive_precision)]
        const ERF_A5: f32 = 1.061405429;
        const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
        fn erf_approx(v: f32) -> f32 {
            let s = if v < 0.0 { -1.0 } else { 1.0 };
            let ax = v.abs();
            let t = 1.0 / (1.0 + ERF_P * ax);
            let poly = ((((ERF_A5 * t + ERF_A4) * t + ERF_A3) * t + ERF_A2) * t + ERF_A1) * t;
            let y = 1.0 - poly * (-ax * ax).exp();
            s * y
        }
        fn gelu_wgsl_mirror(v: f32) -> f32 {
            0.5 * v * (1.0 + erf_approx(v * FRAC_1_SQRT_2))
        }

        // Dense sweep over the numerically interesting band.
        let xs: Vec<f32> = (-8000..=8000).map(|i| i as f32 * 1e-3).collect();
        let mut cpu = vec![0.0f32; xs.len()];
        vokra_backend_cpu::kernels::gelu_f32(&xs, &mut cpu).unwrap();
        let mut max_abs = 0.0f32;
        for (i, &x) in xs.iter().enumerate() {
            let d = (gelu_wgsl_mirror(x) - cpu[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        // Same formula, same coefficients, same f32 op order → the mirror
        // must reproduce the CPU kernel to within a couple of ULP (measured
        // 0.0 on the authoring host; 1e-6 leaves ULP headroom while staying
        // 4 orders of magnitude under atol = 0.01).
        assert!(
            max_abs <= 1e-6,
            "gelu WGSL formula transcription drifted from the CPU kernel: max |Δ| = {max_abs:e}"
        );
    }
}
