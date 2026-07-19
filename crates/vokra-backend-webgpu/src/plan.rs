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

    // ===================================================================
    // M4-01-T13/T14/T15 (#13): WGSL formula-transcription pins for the rest
    // of the kernel set — softmax / softmax_causal / layer_norm / gemm_f32 /
    // gemv_f32 / conv1d / elementwise (+ the dedicated add_f32 arm the #23 fix
    // dispatches). Same design as the gelu pin above: a pure-Rust mirror
    // re-evaluates the EXACT expression each WGSL kernel encodes, in the
    // kernel's own evaluation order, and diffs against the dispatched CPU
    // scalar oracle (`vokra_backend_cpu::kernels::*_on(IsaPath::Scalar, ..)` —
    // the canonical single-thread numeric reference the SIMD paths are checked
    // against). No GPU / browser / wasm: the only free variable is the WGSL
    // transcription, so a sign / coefficient / index / mask typo in a shader
    // (surfaced by `wgsl::tests::wgsl_sources_match_pinned_hashes`, which forces
    // the mirror to be re-derived on any source edit) shows up here. The true
    // GPU-side residual (driver `exp` / `fma` rounding) is the browser harness's
    // job (T18), bounded by atol = 0.01 (NFR-QL-01).
    //
    // Two residual classes, both measured on the authoring host (Apple M1):
    //   * op-order coincides with the scalar oracle → BIT-EXACT (max |Δ| = 0):
    //     `elementwise` / `add_f32` (element-wise), `gemm_f32` (bias-seeded
    //     ascending-k), `conv1d` with no bias (ascending-l, padding is a no-op
    //     `+ 0.0`). Asserted `== 0.0`.
    //   * the WGSL reassociates the reduction (tree-reduced `softmax` /
    //     `softmax_causal` / `layer_norm` / `gemv_f32`) or appends bias instead
    //     of seeding it (`gemv_f32`, biased `conv1d`) → pure FP32 reassociation,
    //     measured « atol and asserted at a documented bound with ULP headroom.

    use vokra_backend_cpu::IsaPath;
    use vokra_backend_cpu::kernels as cpu;

    /// The `-inf` stand-in the softmax kernels use for empty / masked lanes.
    /// softmax.wgsl / softmax_causal.wgsl spell it `-3.402823466e+38`, which
    /// rounds to exactly [`f32::MIN`] — used here so the value is bit-identical
    /// to the shader constant without carrying excess decimal digits.
    const WGSL_NEG_INF: f32 = f32::MIN;

    /// Deterministic, dependency-free pseudo-random f32 in `[-1, 1)` (an LCG —
    /// no `rand` crate, NFR-DS-02). Distinct `seed`s give distinct streams.
    fn seeded(n: usize, seed: u32) -> Vec<f32> {
        // 2^24: the top 24 bits of the LCG state are exactly representable in
        // f32, so `(s >> 8) as f32 / 2^24` is an exact map into [0, 1).
        const SCALE: f32 = 16_777_216.0;
        let mut s = seed ^ 0x9E37_79B9;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / SCALE) * 2.0 - 1.0
            })
            .collect()
    }

    /// Max absolute element-wise difference between two equal-length slices.
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "diff over unequal lengths");
        a.iter()
            .zip(b)
            .fold(0.0f32, |m, (&x, &y)| m.max((x - y).abs()))
    }

    /// Workgroup tree reduction by `+`, transcribing the shared
    /// `stride = WG/2; while stride>0 { scratch[i] += scratch[i+stride] } stride/=2`
    /// pattern of softmax / layer_norm (WG = 256) and gemv (WG = 64).
    /// Sequential evaluation is bit-identical to the WGSL parallel form: each
    /// step writes `scratch[0..stride]` and reads `scratch[0..2*stride]`, and a
    /// lane's read of `scratch[i+stride]` is outside the `[0, stride)` write
    /// range, so no lane observes another lane's same-step write. `scratch.len()`
    /// must be a power of two (256 or 64 here).
    fn wg_tree_sum(scratch: &mut [f32]) -> f32 {
        let mut stride = scratch.len() / 2;
        while stride > 0 {
            for i in 0..stride {
                scratch[i] += scratch[i + stride];
            }
            stride /= 2;
        }
        scratch[0]
    }

    /// Workgroup tree reduction by `max` (softmax pass 1). `max` returns one
    /// input unchanged, so association is irrelevant — bit-identical to the
    /// scalar `fold(NEG_INFINITY, f32::max)` regardless of order.
    fn wg_tree_max(scratch: &mut [f32]) -> f32 {
        let mut stride = scratch.len() / 2;
        while stride > 0 {
            for i in 0..stride {
                scratch[i] = scratch[i].max(scratch[i + stride]);
            }
            stride /= 2;
        }
        scratch[0]
    }

    // ---- softmax.wgsl mirror (WG = WG_ROW_REDUCE) ----
    fn softmax_wgsl_mirror(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        const WG: usize = WG_ROW_REDUCE as usize;
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let base = r * cols;
            // Pass 1: row max (strided partials + tree reduce).
            let mut scratch = vec![WGSL_NEG_INF; WG];
            for (l, slot) in scratch.iter_mut().enumerate() {
                let mut m = WGSL_NEG_INF;
                let mut c = l;
                while c < cols {
                    m = m.max(x[base + c]);
                    c += WG;
                }
                *slot = m;
            }
            let row_max = wg_tree_max(&mut scratch);
            // Pass 2: exp(x - max) into out, accumulating the sum.
            let mut ssum = vec![0.0f32; WG];
            for (l, slot) in ssum.iter_mut().enumerate() {
                let mut s = 0.0f32;
                let mut c = l;
                while c < cols {
                    let e = (x[base + c] - row_max).exp();
                    out[base + c] = e;
                    s += e;
                    c += WG;
                }
                *slot = s;
            }
            let inv = 1.0 / wg_tree_sum(&mut ssum);
            // Pass 3: normalize.
            for c in 0..cols {
                out[base + c] *= inv;
            }
        }
        out
    }

    // ---- softmax_causal.wgsl mirror ----
    fn softmax_causal_wgsl_mirror(x: &[f32], rows: usize, cols: usize, offset: usize) -> Vec<f32> {
        const WG: usize = WG_ROW_REDUCE as usize;
        let mut out = vec![0.0f32; rows * cols];
        for row in 0..rows {
            let base = row * cols;
            let budget = row + offset; // row r may attend to columns c <= r + offset
            // Pass 1: masked row max.
            let mut scratch = vec![WGSL_NEG_INF; WG];
            for (l, slot) in scratch.iter_mut().enumerate() {
                let mut m = WGSL_NEG_INF;
                let mut c = l;
                while c < cols {
                    let v = if c > budget {
                        WGSL_NEG_INF
                    } else {
                        x[base + c]
                    };
                    m = m.max(v);
                    c += WG;
                }
                *slot = m;
            }
            let row_max = wg_tree_max(&mut scratch);
            // Pass 2: exp of masked value (masked lanes write exactly 0.0).
            let mut ssum = vec![0.0f32; WG];
            for (l, slot) in ssum.iter_mut().enumerate() {
                let mut s = 0.0f32;
                let mut c = l;
                while c < cols {
                    let e = if c <= budget {
                        (x[base + c] - row_max).exp()
                    } else {
                        0.0
                    };
                    out[base + c] = e;
                    s += e;
                    c += WG;
                }
                *slot = s;
            }
            let inv = 1.0 / wg_tree_sum(&mut ssum);
            // Pass 3: normalize (masked lanes stay exactly 0.0: 0 * inv = 0).
            for c in 0..cols {
                out[base + c] *= inv;
            }
        }
        out
    }

    // ---- layer_norm.wgsl mirror ----
    fn layer_norm_wgsl_mirror(
        x: &[f32],
        rows: usize,
        cols: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Vec<f32> {
        const WG: usize = WG_ROW_REDUCE as usize;
        let inv_cols = 1.0 / cols as f32;
        let mut out = vec![0.0f32; rows * cols];
        for row in 0..rows {
            let base = row * cols;
            // Pass 1: mean.
            let mut scratch = vec![0.0f32; WG];
            for (l, slot) in scratch.iter_mut().enumerate() {
                let mut s = 0.0f32;
                let mut c = l;
                while c < cols {
                    s += x[base + c];
                    c += WG;
                }
                *slot = s;
            }
            let mean = wg_tree_sum(&mut scratch) * inv_cols;
            // Pass 2: biased variance.
            let mut scratch2 = vec![0.0f32; WG];
            for (l, slot) in scratch2.iter_mut().enumerate() {
                let mut q = 0.0f32;
                let mut c = l;
                while c < cols {
                    let d = x[base + c] - mean;
                    q += d * d;
                    c += WG;
                }
                *slot = q;
            }
            let variance = wg_tree_sum(&mut scratch2) * inv_cols;
            let inv_std = 1.0 / (variance + eps).sqrt();
            // Pass 3: normalize + affine.
            for c in 0..cols {
                out[base + c] = (x[base + c] - mean) * inv_std * gamma[c] + beta[c];
            }
        }
        out
    }

    // ---- gemm_f32.wgsl mirror (bias-seeded, ascending-k tiles, zero pad) ----
    fn gemm_wgsl_mirror(
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
    ) -> Vec<f32> {
        const TILE: usize = GEMM_TILE as usize;
        let tiles = k.div_ceil(TILE);
        let mut out = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = bias.map_or(0.0, |bias| bias[col]);
                for t in 0..tiles {
                    for i in 0..TILE {
                        let kk = t * TILE + i;
                        // Out-of-range tile lanes are zero-padded on both
                        // operands: `acc += 0.0 * 0.0` is an exact no-op.
                        let av = if kk < k { a[row * k + kk] } else { 0.0 };
                        let bv = if kk < k { b[kk * n + col] } else { 0.0 };
                        acc += av * bv;
                    }
                }
                out[row * n + col] = acc;
            }
        }
        out
    }

    // ---- gemv_f32.wgsl mirror (strided partials + WG=64 tree, bias appended) --
    fn gemv_wgsl_mirror(
        m: usize,
        k: usize,
        a: &[f32],
        x: &[f32],
        bias: Option<&[f32]>,
    ) -> Vec<f32> {
        const WG: usize = WG_GEMV as usize;
        let mut out = vec![0.0f32; m];
        for (row, slot) in out.iter_mut().enumerate() {
            let mut partial = vec![0.0f32; WG];
            for (l, p) in partial.iter_mut().enumerate() {
                let mut s = 0.0f32;
                let mut i = l;
                while i < k {
                    s += a[row * k + i] * x[i];
                    i += WG;
                }
                *p = s;
            }
            let sum = wg_tree_sum(&mut partial);
            // The kernel adds bias AFTER the reduction: `r = bias[row] + r`.
            *slot = bias.map_or(sum, |bias| bias[row] + sum);
        }
        out
    }

    // ---- conv1d.wgsl mirror (bias-seeded, ic-major kk-minor, padding skipped) --
    #[allow(clippy::too_many_arguments)]
    fn conv1d_wgsl_mirror(
        x: &[f32],
        in_ch: usize,
        in_len: usize,
        w: &[f32],
        out_ch: usize,
        kernel: usize,
        bias: Option<&[f32]>,
        stride: usize,
        padding: usize,
    ) -> Vec<f32> {
        let out_len = conv1d_out_len(in_len, kernel, stride, padding).unwrap();
        let mut out = vec![0.0f32; out_ch * out_len];
        for oc in 0..out_ch {
            for t in 0..out_len {
                let mut acc = bias.map_or(0.0, |bias| bias[oc]);
                // Signed origin: t*stride - padding can be negative at the left
                // edge (WGSL uses i32; the guard skips out-of-range positions).
                let origin = (t * stride) as i32 - padding as i32;
                for ic in 0..in_ch {
                    let w_base = (oc * in_ch + ic) * kernel;
                    let x_base = ic * in_len;
                    for kk in 0..kernel {
                        let pos = origin + kk as i32;
                        if pos >= 0 && pos < in_len as i32 {
                            acc += w[w_base + kk] * x[x_base + pos as usize];
                        }
                    }
                }
                out[oc * out_len + t] = acc;
            }
        }
        out
    }

    // ---- elementwise.wgsl mirror (op switch: 0 = add, 1 = mul) ----
    fn elementwise_wgsl_mirror(op: ElementwiseOp, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| match op {
                ElementwiseOp::Mul => x * y, // params.op == 1u
                ElementwiseOp::Add => x + y, // else
            })
            .collect()
    }

    // ---- add_f32.wgsl mirror (dedicated Add arm: out[i] = a[i] + b[i]) ----
    fn add_f32_wgsl_mirror(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(&x, &y)| x + y).collect()
    }

    #[test]
    fn softmax_wgsl_formula_transcription_matches_cpu_kernel() {
        // Two shapes: cols <= WG (one element per lane) and cols > WG (strided
        // multi-element lanes + tail), each with a wide input range.
        for (rows, cols) in [(4usize, 48usize), (3, 300)] {
            let x: Vec<f32> = seeded(rows * cols, 0x5017 ^ cols as u32)
                .iter()
                .map(|v| v * 6.0) // widen to exercise the max-shift stabilization
                .collect();
            let mirror = softmax_wgsl_mirror(&x, rows, cols);
            let mut oracle = vec![0.0f32; rows * cols];
            cpu::softmax_f32_on(IsaPath::Scalar, &x, &mut oracle, rows, cols).unwrap();
            let d = max_abs_diff(&mirror, &oracle);
            // exp(x - max) and the max are bit-identical to the oracle; only the
            // tree-reduced sum reassociates. Measured max |Δ| = 2.98e-8 (48
            // cols) / 7.45e-9 (300 cols); outputs are in [0, 1]. 1e-6 leaves ULP
            // headroom and stays 4 orders under atol = 0.01.
            assert!(
                d <= 1e-6,
                "softmax WGSL transcription drifted ({rows}x{cols}): max |Δ| = {d:e}"
            );
        }
    }

    #[test]
    fn softmax_causal_wgsl_formula_transcription_matches_cpu_kernel() {
        // Oracle = scalar softmax over a `-inf`-masked copy (exp(-inf) = 0
        // zeroes the masked columns — the host-mask + softmax equivalence the
        // Metal/CUDA causal kernels pin). offset = t_k - t_q; 0 = square
        // self-attention, > 0 = a decoder step over a KV cache.
        for (rows, cols, offset) in [(6usize, 6usize, 0usize), (5, 40, 3)] {
            let x: Vec<f32> = seeded(rows * cols, 0x0CA5)
                .iter()
                .map(|v| v * 6.0)
                .collect();
            let mirror = softmax_causal_wgsl_mirror(&x, rows, cols, offset);
            let mut masked = x.clone();
            for row in 0..rows {
                for c in 0..cols {
                    if c > row + offset {
                        masked[row * cols + c] = f32::NEG_INFINITY;
                    }
                }
            }
            let mut oracle = vec![0.0f32; rows * cols];
            cpu::softmax_f32_on(IsaPath::Scalar, &masked, &mut oracle, rows, cols).unwrap();
            let d = max_abs_diff(&mirror, &oracle);
            // Masked lanes are exactly 0.0 on both sides; only the unmasked
            // exp-sum reassociates. Measured max |Δ| = 0.0 (6x6, off 0) /
            // 1.19e-7 (5x40, off 3). 1e-6 leaves ULP headroom.
            assert!(
                d <= 1e-6,
                "softmax_causal WGSL transcription drifted ({rows}x{cols}, off {offset}): \
                 max |Δ| = {d:e}"
            );
        }
    }

    #[test]
    fn layer_norm_wgsl_formula_transcription_matches_cpu_kernel() {
        for (rows, cols) in [(4usize, 64usize), (3, 384)] {
            let x: Vec<f32> = seeded(rows * cols, 0x1AE7)
                .iter()
                .map(|v| v * 3.0)
                .collect();
            let gamma: Vec<f32> = seeded(cols, 0x6A33)
                .iter()
                .map(|v| 1.0 + 0.25 * v)
                .collect();
            let beta = seeded(cols, 0xBE7A);
            let eps = 1e-5f32; // PyTorch nn.LayerNorm default (never invented — T14).
            let mirror = layer_norm_wgsl_mirror(&x, rows, cols, &gamma, &beta, eps);
            let mut oracle = vec![0.0f32; rows * cols];
            cpu::layer_norm_f32_on(
                IsaPath::Scalar,
                &x,
                &mut oracle,
                rows,
                cols,
                &gamma,
                &beta,
                eps,
            )
            .unwrap();
            let d = max_abs_diff(&mirror, &oracle);
            // Only the tree-reduced mean and variance reassociate; Pass 3 is
            // identical given them. Measured max |Δ| = 4.77e-7 (both shapes);
            // 1e-4 leaves ample headroom, 2 orders under atol = 0.01.
            assert!(
                d <= 1e-4,
                "layer_norm WGSL transcription drifted ({rows}x{cols}): max |Δ| = {d:e}"
            );
        }
    }

    #[test]
    fn gemm_wgsl_formula_transcription_matches_cpu_kernel() {
        // k = 40 = 2*TILE + 8 exercises multiple tiles + a partial (zero-padded)
        // last tile. With and without bias.
        let (m, n, k) = (6usize, 5usize, 40usize);
        let a = seeded(m * k, 0x9E44);
        let b = seeded(k * n, 0x0B15);
        let bias = seeded(n, 0xB1A5);
        for use_bias in [false, true] {
            let bias_opt = use_bias.then_some(bias.as_slice());
            let mirror = gemm_wgsl_mirror(m, n, k, &a, &b, bias_opt);
            let mut oracle = vec![0.0f32; m * n];
            cpu::gemm_f32_on(IsaPath::Scalar, m, n, k, &a, &b, bias_opt, &mut oracle).unwrap();
            let d = max_abs_diff(&mirror, &oracle);
            // Bias-seeded ascending-k with zero-pad no-ops == the scalar oracle's
            // op order exactly → BIT-EXACT.
            assert!(
                d == 0.0,
                "gemm WGSL transcription is not bit-exact (bias {use_bias}): max |Δ| = {d:e}"
            );
        }
    }

    #[test]
    fn gemv_wgsl_formula_transcription_matches_cpu_kernel() {
        // k = 200 > WG_GEMV exercises strided partials (lane l sums l, l+64,
        // l+128, ...) then the 64-lane tree.
        let (m, k) = (5usize, 200usize);
        let a = seeded(m * k, 0x6E44);
        let x = seeded(k, 0x0417);
        let bias = seeded(m, 0xB1A6);
        for use_bias in [false, true] {
            let bias_opt = use_bias.then_some(bias.as_slice());
            let mirror = gemv_wgsl_mirror(m, k, &a, &x, bias_opt);
            let mut oracle = vec![0.0f32; m];
            cpu::gemv_f32_on(IsaPath::Scalar, m, k, &a, &x, bias_opt, &mut oracle).unwrap();
            let d = max_abs_diff(&mirror, &oracle);
            // Reassociation: strided-partials + tree vs the scalar left-to-right
            // fold, and bias appended vs seeded. Measured max |Δ| = 2.38e-6 (no
            // bias) / 1.91e-6 (bias), sums of ~200 unit-scale products; 1e-3
            // leaves generous headroom, still an order under atol = 0.01.
            assert!(
                d <= 1e-3,
                "gemv WGSL transcription drifted (bias {use_bias}): max |Δ| = {d:e}"
            );
        }
    }

    #[test]
    fn conv1d_wgsl_formula_transcription_matches_cpu_kernel() {
        // Whisper stem envelope: kernel 3, padding 1, stride 1 (keeps length)
        // and stride 2 (halves it).
        let (in_ch, in_len, out_ch, kernel, padding) = (3usize, 20usize, 4usize, 3usize, 1usize);
        let x = seeded(in_ch * in_len, 0xC0A1);
        let w = seeded(out_ch * in_ch * kernel, 0x0FED);
        let bias = seeded(out_ch, 0xB1A7);
        for stride in [1usize, 2usize] {
            // No bias: ascending-l with padding as an exact `+ 0.0` no-op →
            // BIT-EXACT vs the scalar im2col + GEMM oracle.
            let mut oracle_nb =
                vec![0.0f32; out_ch * conv1d_out_len(in_len, kernel, stride, padding).unwrap()];
            cpu::conv1d_f32_on(
                IsaPath::Scalar,
                &x,
                in_ch,
                in_len,
                &w,
                out_ch,
                kernel,
                None,
                stride,
                padding,
                &mut oracle_nb,
            )
            .unwrap();
            let mirror_nb =
                conv1d_wgsl_mirror(&x, in_ch, in_len, &w, out_ch, kernel, None, stride, padding);
            let d_nb = max_abs_diff(&mirror_nb, &oracle_nb);
            assert!(
                d_nb == 0.0,
                "conv1d WGSL transcription is not bit-exact (stride {stride}, no bias): \
                 max |Δ| = {d_nb:e}"
            );
            // With bias: the WGSL SEEDS bias before the sum while the CPU appends
            // it after the GEMM → a single end reassociation. Measured max |Δ| =
            // 3.58e-7 (stride 1) / 2.38e-7 (stride 2); 1e-5 leaves ULP headroom.
            let mut oracle_b = vec![0.0f32; oracle_nb.len()];
            cpu::conv1d_f32_on(
                IsaPath::Scalar,
                &x,
                in_ch,
                in_len,
                &w,
                out_ch,
                kernel,
                Some(&bias),
                stride,
                padding,
                &mut oracle_b,
            )
            .unwrap();
            let mirror_b = conv1d_wgsl_mirror(
                &x,
                in_ch,
                in_len,
                &w,
                out_ch,
                kernel,
                Some(&bias),
                stride,
                padding,
            );
            let d_b = max_abs_diff(&mirror_b, &oracle_b);
            assert!(
                d_b <= 1e-5,
                "conv1d WGSL transcription drifted (stride {stride}, bias): max |Δ| = {d_b:e}"
            );
        }
    }

    #[test]
    fn elementwise_wgsl_formula_transcription_matches_cpu_kernel() {
        // The `elementwise` op-switch backs OpKind::Mul (op = 1) and is the
        // generic add/mul seam (op = 0/1); pin both flags. Element-wise → the
        // op order coincides with the scalar oracle → BIT-EXACT.
        let a = seeded(257, 0xE1E1);
        let b = seeded(257, 0xE2E2);
        // op = 0 → add.
        let mut add_oracle = vec![0.0f32; a.len()];
        cpu::add_f32_on(IsaPath::Scalar, &a, &b, &mut add_oracle).unwrap();
        let add_mirror = elementwise_wgsl_mirror(ElementwiseOp::Add, &a, &b);
        assert!(
            max_abs_diff(&add_mirror, &add_oracle) == 0.0,
            "elementwise(add) WGSL transcription is not bit-exact"
        );
        // op = 1 → mul.
        let mut mul_oracle = vec![0.0f32; a.len()];
        cpu::mul_f32_on(IsaPath::Scalar, &a, &b, &mut mul_oracle).unwrap();
        let mul_mirror = elementwise_wgsl_mirror(ElementwiseOp::Mul, &a, &b);
        assert!(
            max_abs_diff(&mul_mirror, &mul_oracle) == 0.0,
            "elementwise(mul) WGSL transcription is not bit-exact"
        );
    }

    #[test]
    fn add_f32_wgsl_formula_transcription_matches_cpu_kernel() {
        // The DEDICATED add_f32 kernel the #23 fix routes OpKind::Add through
        // (distinct from the elementwise op-switch above). Element-wise `+` is
        // IEEE-754 exact → BIT-EXACT vs the scalar oracle.
        let a = seeded(257, 0xADD1);
        let b = seeded(257, 0xADD2);
        let mut oracle = vec![0.0f32; a.len()];
        cpu::add_f32_on(IsaPath::Scalar, &a, &b, &mut oracle).unwrap();
        let mirror = add_f32_wgsl_mirror(&a, &b);
        assert!(
            max_abs_diff(&mirror, &oracle) == 0.0,
            "add_f32 WGSL transcription is not bit-exact"
        );
    }
}
