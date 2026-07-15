//! Host-portable SPIR-V kernel dispatch *planning* (M4-13-T02〜T08).
//!
//! A [`KernelPlan`] fully describes one compute dispatch — which manifest
//! kernel to bind, its LE-packed push-constant block, its pipeline
//! specialization constants, the `vkCmdDispatch` workgroup counts, and the
//! output SSBO length — **without touching any Vulkan object**. The gated
//! dispatch side (`context::dispatch_kernel`, Vulkan targets only) consumes a
//! plan verbatim; this module is compiled on **every** target so that the
//! whole host-side contract (shape validation, push-constant layout,
//! workgroup math) is unit-testable on the Apple-Silicon authoring host with
//! no Vulkan loader present (the M3-02 / M4-13 split of "host-portable
//! validation" vs "lavapipe/Android real dispatch").
//!
//! # The frozen `.comp` contract
//!
//! Every `plan_*` constructor below mirrors — field-for-field — the
//! `layout(push_constant)` block, the SSBO binding order, and the
//! `layout(local_size_*)` of its GLSL source under `kernels/glsl/*.comp`
//! (M3-02 skeletons, frozen so the Rust dispatcher can rely on them).
//! The [`shader_local_size`] table is cross-checked against the committed
//! GLSL sources by the `local_size_table_matches_glsl_sources` test, so a
//! `.comp` edit that silently changes workgroup geometry fails the suite
//! instead of corrupting dispatch math (no fabricated pass).
//!
//! # Binding convention
//!
//! Read-only input SSBOs occupy `binding = 0..N-1` in the order the plan's
//! consumer uploads them; the writable output SSBO is **always the last
//! binding** (`binding = N-1 + 1`) — the convention every committed `.comp`
//! skeleton follows and `context::dispatch_kernel` encodes.

use vokra_core::{Result, VokraError};

use crate::backend::GemmPipelineVariant;

/// `vkCmdDispatch` per-axis group-count ceiling Vokra treats as portable:
/// the Vulkan spec's *guaranteed minimum* for
/// `VkPhysicalDeviceLimits::maxComputeWorkGroupCount` (spec §42.1 "Required
/// Limits") is 65535 per axis, so staying at or below it needs no per-device
/// limit query.
pub const MAX_WORKGROUPS_PER_AXIS: u32 = 65_535;

/// Push-constant block ceiling Vokra treats as portable: the Vulkan spec's
/// guaranteed minimum for `VkPhysicalDeviceLimits::maxPushConstantsSize`
/// (spec §42.1) is 128 bytes.
pub const MAX_PUSH_CONSTANT_BYTES: usize = 128;

/// One `layout(constant_id = N)` u32 specialization constant, applied at
/// pipeline-creation time (M4-13-T02/T07 — `elementwise` OP and `activation`
/// KIND selection).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecConstantU32 {
    /// The GLSL `constant_id`.
    pub constant_id: u32,
    /// The u32 value the pipeline is specialised with.
    pub value: u32,
}

/// A fully-described generic kernel dispatch: which manifest kernel to run,
/// its push constants, specialization constants, workgroup counts, and the
/// f32 element count of its output SSBO. Produced by the `plan_*`
/// constructors below (host-portable); consumed by the Vulkan-target
/// dispatch chain (`context::dispatch_kernel` via `kernels`).
#[derive(Debug, Clone)]
pub struct KernelPlan {
    /// Manifest kernel name (`spirv::SHADERS` entry).
    pub shader: &'static str,
    /// Raw push-constant block (LE-packed scalars; may be empty). Always a
    /// multiple of 4 bytes and at most [`MAX_PUSH_CONSTANT_BYTES`].
    pub push_constants: Vec<u8>,
    /// Pipeline specialization constants (may be empty).
    pub spec_constants: Vec<SpecConstantU32>,
    /// `vkCmdDispatch` group counts `[x, y, z]`, all validated into
    /// `1..=`[`MAX_WORKGROUPS_PER_AXIS`].
    pub workgroups: [u32; 3],
    /// Number of f32 elements in the writable output SSBO (the last binding).
    pub output_len: usize,
}

impl KernelPlan {
    /// Byte length of the output SSBO (`output_len * 4`).
    #[must_use]
    pub fn output_byte_len(&self) -> usize {
        self.output_len * 4
    }
}

// ---------------------------------------------------------------------------
// Small shared plumbing: push-constant packing + dispatch-grid math.
// ---------------------------------------------------------------------------

/// LE-packing push-constant writer. GLSL `layout(push_constant) uniform PC`
/// blocks in the committed kernels contain only 4-byte scalars (`uint` /
/// `float`), laid out in declaration order with std430-natural 4-byte
/// alignment — so a plain LE byte concatenation reproduces the block.
#[derive(Debug, Default)]
struct PcWriter(Vec<u8>);

impl PcWriter {
    fn u32(mut self, v: u32) -> Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }

    fn f32(mut self, v: f32) -> Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }

    fn finish(self, shader: &'static str) -> Result<Vec<u8>> {
        debug_assert_eq!(self.0.len() % 4, 0, "PcWriter only writes 4-byte scalars");
        if self.0.len() > MAX_PUSH_CONSTANT_BYTES {
            return Err(VokraError::InvalidArgument(format!(
                "plan({shader}): push-constant block of {} bytes exceeds the portable \
                 {MAX_PUSH_CONSTANT_BYTES}-byte ceiling (spec §42.1)",
                self.0.len()
            )));
        }
        Ok(self.0)
    }
}

/// `ceil(num / den)` as a dispatch group count, validated into
/// `1..=`[`MAX_WORKGROUPS_PER_AXIS`]. `num` must be non-zero (a zero-sized
/// dispatch axis is a caller bug surfaced as an explicit error, never a
/// silent no-op dispatch).
fn group_count(shader: &'static str, axis: &str, num: usize, den: u32) -> Result<u32> {
    debug_assert!(den > 0, "local size is a non-zero compile-time constant");
    if num == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "plan({shader}): dispatch axis {axis} has zero elements"
        )));
    }
    let den = den as usize;
    let groups = num.div_ceil(den);
    if groups > MAX_WORKGROUPS_PER_AXIS as usize {
        return Err(VokraError::InvalidArgument(format!(
            "plan({shader}): dispatch axis {axis} needs {groups} workgroups, above the portable \
             {MAX_WORKGROUPS_PER_AXIS} ceiling (spec §42.1 guaranteed minimum for \
             maxComputeWorkGroupCount)"
        )));
    }
    // `groups` fits u32: it is <= MAX_WORKGROUPS_PER_AXIS.
    Ok(groups as u32)
}

/// Casts a dimension to `u32` for a push constant, with an explicit error
/// when it does not fit (the GLSL PC blocks declare `uint`).
fn pc_u32(shader: &'static str, what: &str, v: usize) -> Result<u32> {
    u32::try_from(v).map_err(|_| {
        VokraError::InvalidArgument(format!(
            "plan({shader}): {what} = {v} does not fit the shader's u32 push constant"
        ))
    })
}

/// Validates that a caller-supplied buffer has exactly the expected element
/// count (explicit `InvalidArgument`, mirroring the CPU backend's kernel
/// validators).
fn expect_len(shader: &'static str, what: &str, got: usize, want: usize) -> Result<()> {
    if got != want {
        return Err(VokraError::InvalidArgument(format!(
            "plan({shader}): {what} has {got} elements, expected {want}"
        )));
    }
    Ok(())
}

/// Frozen mirror of `layout(local_size_x/y/z)` for every glslc-produced
/// kernel under `kernels/glsl/*.comp`. The dispatch-grid math in the
/// `plan_*` constructors divides by these values, so the table MUST match
/// the GLSL sources — the `local_size_table_matches_glsl_sources` test
/// parses the committed `.comp` files and asserts equality (drift gate; a
/// `.comp` workgroup-geometry edit without a matching update here fails the
/// suite). The two hand-crafted smoke kernels keep their `LOCAL_SIZE_X`
/// consts in `kernels/handcrafted/*.spv.rs` and are not planned through this
/// module.
#[must_use]
pub fn shader_local_size(name: &str) -> Option<[u32; 3]> {
    match name {
        "gemm_subgroup" => Some([16, 16, 1]),
        "gemm_coopmat" => Some([32, 1, 1]),
        "gemv" => Some([32, 1, 1]),
        "softmax" => Some([32, 1, 1]),
        "softmax_causal" => Some([32, 1, 1]),
        "layer_norm" => Some([32, 1, 1]),
        "gelu" => Some([256, 1, 1]),
        "conv1d" => Some([64, 1, 1]),
        "elementwise" => Some([256, 1, 1]),
        "activation" => Some([256, 1, 1]),
        "transpose" => Some([16, 16, 1]),
        "gather" => Some([64, 4, 1]),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Per-kernel plans (M4-13-T03〜T08). Each mirrors its `.comp` source
// field-for-field; see the module docs.
// ---------------------------------------------------------------------------

/// Plans `out[m,n] = Σ_k a[m,k]·b[k,n]` on the GEMM pipeline `variant`
/// selected by the probe (M4-13-T03).
///
/// `.comp` contract (`gemm_subgroup.comp` / `gemm_coopmat.comp`): PC
/// `{m, n, k}`; SSBOs `lhs(0), rhs(1), out(2)`; one invocation per output
/// element with `gl_GlobalInvocationID.x` walking columns and `.y` rows.
pub fn plan_gemm(
    variant: GemmPipelineVariant,
    m: usize,
    n: usize,
    k: usize,
    a_len: usize,
    b_len: usize,
) -> Result<KernelPlan> {
    let shader = variant.shader_name();
    expect_len(shader, "lhs (m*k)", a_len, m.saturating_mul(k))?;
    expect_len(shader, "rhs (k*n)", b_len, k.saturating_mul(n))?;
    if m == 0 || n == 0 || k == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "plan({shader}): GEMM dims must be non-zero; got m={m} n={n} k={k}"
        )));
    }
    let local = shader_local_size(shader).expect("gemm variants are in the local-size table");
    // Column axis is X, row axis is Y in both GEMM shaders.
    let workgroups = [
        group_count(shader, "x (cols)", n, local[0])?,
        group_count(shader, "y (rows)", m, local[1])?,
        1,
    ];
    Ok(KernelPlan {
        shader,
        push_constants: PcWriter::default()
            .u32(pc_u32(shader, "m", m)?)
            .u32(pc_u32(shader, "n", n)?)
            .u32(pc_u32(shader, "k", k)?)
            .finish(shader)?,
        spec_constants: Vec::new(),
        workgroups,
        output_len: m * n,
    })
}

/// Plans `y[i] = Σ_j A[i,j]·x[j] (+ b[i])` (M4-13-T04, Whisper decoder-step
/// hot path).
///
/// `.comp` contract (`gemv.comp`): PC `{m, n, bias_present}`; SSBOs
/// `A(0), x(1), b(2), y(3)`; **one workgroup per output row** (the kernel
/// reduces the inner sum across the workgroup), so the dispatch is `[m,1,1]`.
/// When `bias` is absent the caller binds a 4-byte dummy SSBO at binding 2
/// (Vulkan requires every declared binding bound) and `bias_present = 0`
/// makes the shader never read it.
pub fn plan_gemv(
    m: usize,
    n: usize,
    a_len: usize,
    x_len: usize,
    bias_len: Option<usize>,
) -> Result<KernelPlan> {
    let shader = "gemv";
    expect_len(shader, "A (m*n)", a_len, m.saturating_mul(n))?;
    expect_len(shader, "x (n)", x_len, n)?;
    if let Some(b_len) = bias_len {
        expect_len(shader, "bias (m)", b_len, m)?;
    }
    if m == 0 || n == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "plan({shader}): GEMV dims must be non-zero; got m={m} n={n}"
        )));
    }
    // One workgroup per row — the group count IS m (not ceil(m/local)).
    let workgroups = [group_count(shader, "x (rows)", m, 1)?, 1, 1];
    Ok(KernelPlan {
        shader,
        push_constants: PcWriter::default()
            .u32(pc_u32(shader, "m", m)?)
            .u32(pc_u32(shader, "n", n)?)
            .u32(u32::from(bias_len.is_some()))
            .finish(shader)?,
        spec_constants: Vec::new(),
        workgroups,
        output_len: m,
    })
}

/// Plans a numerically-stable row softmax `[rows, cols]` (M4-13-T05).
///
/// `.comp` contract (`softmax.comp`): PC `{rows, cols}`; SSBOs
/// `in(0), out(1)`; one workgroup per row.
pub fn plan_softmax(rows: usize, cols: usize, x_len: usize) -> Result<KernelPlan> {
    plan_rowwise("softmax", rows, cols, x_len)
}

/// Plans a causal-masked row softmax (M4-13-T05): row `i` normalises over
/// columns `0..=i`; masked columns are written as exactly `0.0`
/// (`exp(-inf) = 0` semantics, matching the Metal / CUDA `softmax_causal`
/// kernels' host-mask equivalence).
///
/// `.comp` contract (`softmax_causal.comp`): identical surface to
/// [`plan_softmax`].
pub fn plan_softmax_causal(rows: usize, cols: usize, x_len: usize) -> Result<KernelPlan> {
    plan_rowwise("softmax_causal", rows, cols, x_len)
}

fn plan_rowwise(
    shader: &'static str,
    rows: usize,
    cols: usize,
    x_len: usize,
) -> Result<KernelPlan> {
    expect_len(
        shader,
        "input (rows*cols)",
        x_len,
        rows.saturating_mul(cols),
    )?;
    if rows == 0 || cols == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "plan({shader}): rows/cols must be non-zero; got rows={rows} cols={cols}"
        )));
    }
    let workgroups = [group_count(shader, "x (rows)", rows, 1)?, 1, 1];
    Ok(KernelPlan {
        shader,
        push_constants: PcWriter::default()
            .u32(pc_u32(shader, "rows", rows)?)
            .u32(pc_u32(shader, "cols", cols)?)
            .finish(shader)?,
        spec_constants: Vec::new(),
        workgroups,
        output_len: rows * cols,
    })
}

/// Plans row-wise layer normalisation with affine parameters (M4-13-T06).
///
/// `eps` is **the model's configured value** passed through verbatim (same
/// contract as the CPU backend's `layer_norm_f32` — the plan never invents
/// an epsilon; Whisper reads it from the checkpoint config, M0-08-T07).
///
/// `.comp` contract (`layer_norm.comp`): PC `{rows, cols, eps}` (two `uint`
/// plus one `float`); SSBOs `in(0), gamma(1), beta(2), out(3)`; one
/// workgroup per row.
pub fn plan_layer_norm(
    rows: usize,
    cols: usize,
    eps: f32,
    x_len: usize,
    gamma_len: usize,
    beta_len: usize,
) -> Result<KernelPlan> {
    let shader = "layer_norm";
    expect_len(
        shader,
        "input (rows*cols)",
        x_len,
        rows.saturating_mul(cols),
    )?;
    expect_len(shader, "gamma (cols)", gamma_len, cols)?;
    expect_len(shader, "beta (cols)", beta_len, cols)?;
    if rows == 0 || cols == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "plan({shader}): rows/cols must be non-zero; got rows={rows} cols={cols}"
        )));
    }
    let workgroups = [group_count(shader, "x (rows)", rows, 1)?, 1, 1];
    Ok(KernelPlan {
        shader,
        push_constants: PcWriter::default()
            .u32(pc_u32(shader, "rows", rows)?)
            .u32(pc_u32(shader, "cols", cols)?)
            .f32(eps)
            .finish(shader)?,
        spec_constants: Vec::new(),
        workgroups,
        output_len: rows * cols,
    })
}

/// Plans element-wise GELU (M4-13-T06). The kernel implements the **exact
/// (erf-based) form** with the CPU backend's A&S 7.1.26 coefficients — see
/// `kernels/glsl/gelu.comp` and `vokra-backend-cpu`'s `gelu_f32` (matching
/// OpenAI Whisper's `nn.GELU()` default; formula parity is a hard
/// requirement, M2-01-T09 spirit).
///
/// `.comp` contract (`gelu.comp`): PC `{n}`; SSBOs `in(0), out(1)`.
pub fn plan_gelu(x_len: usize) -> Result<KernelPlan> {
    plan_pointwise_unary("gelu", x_len, Vec::new())
}

/// Unary activation selector for [`plan_activation`] (M4-13-T07). The
/// discriminants are the GLSL `KIND` specialization-constant values in
/// `activation.comp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationKind {
    /// `out = max(0, x)`.
    Relu = 0,
    /// `out = 1 / (1 + exp(-x))`.
    Sigmoid = 1,
    /// `out = tanh(x)`.
    Tanh = 2,
}

/// Plans an element-wise unary activation (M4-13-T07): relu / sigmoid /
/// tanh, selected by the `KIND` pipeline specialization constant
/// (`constant_id = 0`) — one shader, three pipelines, no runtime branch on
/// the hot path.
///
/// `.comp` contract (`activation.comp`): PC `{n}`; SSBOs `in(0), out(1)`;
/// spec constant `KIND @ constant_id 0`.
pub fn plan_activation(kind: ActivationKind, x_len: usize) -> Result<KernelPlan> {
    plan_pointwise_unary(
        "activation",
        x_len,
        vec![SpecConstantU32 {
            constant_id: 0,
            value: kind as u32,
        }],
    )
}

fn plan_pointwise_unary(
    shader: &'static str,
    x_len: usize,
    spec_constants: Vec<SpecConstantU32>,
) -> Result<KernelPlan> {
    if x_len == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "plan({shader}): input must be non-empty"
        )));
    }
    let local = shader_local_size(shader).expect("pointwise shaders are in the local-size table");
    let workgroups = [group_count(shader, "x", x_len, local[0])?, 1, 1];
    Ok(KernelPlan {
        shader,
        push_constants: PcWriter::default()
            .u32(pc_u32(shader, "n", x_len)?)
            .finish(shader)?,
        spec_constants,
        workgroups,
        output_len: x_len,
    })
}

/// Binary element-wise selector for [`plan_elementwise`] (M4-13-T07). The
/// discriminants are the GLSL `OP` specialization-constant values in
/// `elementwise.comp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementwiseOp {
    /// `out[i] = a[i] + b[i]`.
    Add = 0,
    /// `out[i] = a[i] * b[i]`.
    Mul = 1,
}

/// Plans an element-wise binary op (M4-13-T07): add / mul selected by the
/// `OP` specialization constant (`constant_id = 0`). `Add` duplicates the
/// hand-crafted `add_f32` smoke kernel's semantics but with a bounds-checked
/// body (arbitrary `n`, no multiple-of-local-size requirement); `Mul` backs
/// the graph executor's `OpKind::Mul` arm (M4-13-T09).
///
/// `.comp` contract (`elementwise.comp`): PC `{n}`; SSBOs
/// `a(0), b(1), out(2)`; spec constant `OP @ constant_id 0`.
pub fn plan_elementwise(op: ElementwiseOp, a_len: usize, b_len: usize) -> Result<KernelPlan> {
    let shader = "elementwise";
    expect_len(shader, "b (must match a)", b_len, a_len)?;
    if a_len == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "plan({shader}): inputs must be non-empty"
        )));
    }
    let local = shader_local_size(shader).expect("elementwise is in the local-size table");
    let workgroups = [group_count(shader, "x", a_len, local[0])?, 1, 1];
    Ok(KernelPlan {
        shader,
        push_constants: PcWriter::default()
            .u32(pc_u32(shader, "n", a_len)?)
            .finish(shader)?,
        spec_constants: vec![SpecConstantU32 {
            constant_id: 0,
            value: op as u32,
        }],
        workgroups,
        output_len: a_len,
    })
}

/// Shape descriptor for [`plan_conv1d`] (M4-13-T07). Mirrors the PC block of
/// `conv1d.comp` minus the derived `out_len` / `bias_present` fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv1dDims {
    /// Batch count (Whisper front-end runs batch = 1).
    pub batch: usize,
    /// Input channels.
    pub in_ch: usize,
    /// Output channels.
    pub out_ch: usize,
    /// Input length (time axis).
    pub in_len: usize,
    /// Kernel width.
    pub kernel_len: usize,
    /// Stride (Whisper conv1 = 1, conv2 = 2).
    pub stride: usize,
    /// Symmetric zero padding (Whisper conv1/conv2 = 1).
    pub padding: usize,
}

impl Conv1dDims {
    /// `out_len = (in_len + 2*padding - kernel_len) / stride + 1`, or an
    /// explicit error when the padded input is shorter than the kernel or a
    /// dimension is zero.
    pub fn out_len(&self) -> Result<usize> {
        if self.batch == 0
            || self.in_ch == 0
            || self.out_ch == 0
            || self.in_len == 0
            || self.kernel_len == 0
            || self.stride == 0
        {
            return Err(VokraError::InvalidArgument(format!(
                "plan(conv1d): all dims must be non-zero; got {self:?}"
            )));
        }
        let padded = self.in_len + 2 * self.padding;
        if padded < self.kernel_len {
            return Err(VokraError::InvalidArgument(format!(
                "plan(conv1d): padded input length {padded} is shorter than kernel_len {}",
                self.kernel_len
            )));
        }
        Ok((padded - self.kernel_len) / self.stride + 1)
    }
}

/// Plans a 1-D convolution for the Whisper front-end conv1 / conv2 layers
/// (M4-13-T07). Direct (non-im2col) kernel: one invocation per output
/// element `[b, oc, t]`.
///
/// `.comp` contract (`conv1d.comp`): PC `{batch, in_ch, out_ch, in_len,
/// out_len, kernel_len, stride, padding, bias_present}` (9 × `uint` = 36
/// bytes); SSBOs `in(0), weight(1), bias(2), out(3)`; dispatch axes
/// `x = out_len`, `y = out_ch` (one group per channel), `z = batch` (one
/// group per batch item). Absent bias binds a 4-byte dummy at binding 2.
pub fn plan_conv1d(
    dims: &Conv1dDims,
    input_len: usize,
    weight_len: usize,
    bias_len: Option<usize>,
) -> Result<KernelPlan> {
    let shader = "conv1d";
    let out_len = dims.out_len()?;
    expect_len(
        shader,
        "input (batch*in_ch*in_len)",
        input_len,
        dims.batch * dims.in_ch * dims.in_len,
    )?;
    expect_len(
        shader,
        "weight (out_ch*in_ch*kernel_len)",
        weight_len,
        dims.out_ch * dims.in_ch * dims.kernel_len,
    )?;
    if let Some(b_len) = bias_len {
        expect_len(shader, "bias (out_ch)", b_len, dims.out_ch)?;
    }
    let local = shader_local_size(shader).expect("conv1d is in the local-size table");
    let workgroups = [
        group_count(shader, "x (out_len)", out_len, local[0])?,
        group_count(shader, "y (out_ch)", dims.out_ch, local[1])?,
        group_count(shader, "z (batch)", dims.batch, local[2])?,
    ];
    Ok(KernelPlan {
        shader,
        push_constants: PcWriter::default()
            .u32(pc_u32(shader, "batch", dims.batch)?)
            .u32(pc_u32(shader, "in_ch", dims.in_ch)?)
            .u32(pc_u32(shader, "out_ch", dims.out_ch)?)
            .u32(pc_u32(shader, "in_len", dims.in_len)?)
            .u32(pc_u32(shader, "out_len", out_len)?)
            .u32(pc_u32(shader, "kernel_len", dims.kernel_len)?)
            .u32(pc_u32(shader, "stride", dims.stride)?)
            .u32(pc_u32(shader, "padding", dims.padding)?)
            .u32(u32::from(bias_len.is_some()))
            .finish(shader)?,
        spec_constants: Vec::new(),
        workgroups,
        output_len: dims.batch * dims.out_ch * out_len,
    })
}

/// Plans a 2-D transpose `[m, n] → [n, m]` (M4-13-T08). Reshape needs **no
/// shader at all** — it is a buffer reinterpretation on the host side
/// (M3-02-T22 note); only the axis swap moves memory.
///
/// `.comp` contract (`transpose.comp`): PC `{m, n}`; SSBOs `in(0), out(1)`;
/// dispatch `x` walks columns (`n`), `y` walks rows (`m`).
pub fn plan_transpose(m: usize, n: usize, x_len: usize) -> Result<KernelPlan> {
    let shader = "transpose";
    expect_len(shader, "input (m*n)", x_len, m.saturating_mul(n))?;
    if m == 0 || n == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "plan({shader}): dims must be non-zero; got m={m} n={n}"
        )));
    }
    let local = shader_local_size(shader).expect("transpose is in the local-size table");
    let workgroups = [
        group_count(shader, "x (cols)", n, local[0])?,
        group_count(shader, "y (rows)", m, local[1])?,
        1,
    ];
    Ok(KernelPlan {
        shader,
        push_constants: PcWriter::default()
            .u32(pc_u32(shader, "m", m)?)
            .u32(pc_u32(shader, "n", n)?)
            .finish(shader)?,
        spec_constants: Vec::new(),
        workgroups,
        output_len: m * n,
    })
}

/// Plans an embedding-lookup gather `out[i, :] = table[indices[i], :]`
/// (M4-13-T08).
///
/// Every index is bounds-checked **host-side before dispatch**: an
/// out-of-range index is an explicit
/// [`VokraError::InvalidArgument`] (NFR-RL-07 — the shader additionally
/// zero-fills OOB rows defensively, but garbage-in is rejected before any
/// GPU work, per the `gather.comp` header contract).
///
/// `.comp` contract (`gather.comp`): PC `{n, vocab, dim}`; SSBOs
/// `table(0), indices(1) (uint), out(2)`; dispatch `x` walks the embedding
/// dim, `y` walks the `n` looked-up rows.
pub fn plan_gather(
    vocab: usize,
    dim: usize,
    table_len: usize,
    indices: &[u32],
) -> Result<KernelPlan> {
    let shader = "gather";
    expect_len(
        shader,
        "table (vocab*dim)",
        table_len,
        vocab.saturating_mul(dim),
    )?;
    if vocab == 0 || dim == 0 || indices.is_empty() {
        return Err(VokraError::InvalidArgument(format!(
            "plan({shader}): vocab/dim/indices must be non-empty; got vocab={vocab} dim={dim} \
             n={}",
            indices.len()
        )));
    }
    for (pos, &idx) in indices.iter().enumerate() {
        if idx as usize >= vocab {
            return Err(VokraError::InvalidArgument(format!(
                "plan({shader}): index {idx} at position {pos} is out of range for vocab {vocab} \
                 (explicit reject before dispatch, NFR-RL-07)"
            )));
        }
    }
    let n = indices.len();
    let local = shader_local_size(shader).expect("gather is in the local-size table");
    let workgroups = [
        group_count(shader, "x (dim)", dim, local[0])?,
        group_count(shader, "y (n)", n, local[1])?,
        1,
    ];
    Ok(KernelPlan {
        shader,
        push_constants: PcWriter::default()
            .u32(pc_u32(shader, "n", n)?)
            .u32(pc_u32(shader, "vocab", vocab)?)
            .u32(pc_u32(shader, "dim", dim)?)
            .finish(shader)?,
        spec_constants: Vec::new(),
        workgroups,
        output_len: n * dim,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- shared plumbing -------------------------------------------------

    #[test]
    fn pc_writer_packs_le_scalars_in_order() {
        // Hand-computed layout: u32 3 → 03 00 00 00; f32 1.0 → 00 00 80 3f.
        let pc = PcWriter::default().u32(3).f32(1.0).finish("test").unwrap();
        assert_eq!(pc, vec![0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f]);
    }

    #[test]
    fn pc_writer_rejects_oversized_block() {
        let mut w = PcWriter::default();
        for i in 0..33 {
            w = w.u32(i); // 33 * 4 = 132 bytes > 128
        }
        assert!(matches!(
            w.finish("test"),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn group_count_math_covers_exact_and_ragged() {
        assert_eq!(group_count("t", "x", 256, 256).unwrap(), 1);
        assert_eq!(group_count("t", "x", 257, 256).unwrap(), 2);
        assert_eq!(group_count("t", "x", 1, 256).unwrap(), 1);
        // Zero elements and over-ceiling both explicit errors.
        assert!(group_count("t", "x", 0, 256).is_err());
        assert!(group_count("t", "x", 65_536, 1).is_err());
        assert_eq!(group_count("t", "x", 65_535, 1).unwrap(), 65_535);
    }

    /// The local-size table mirrors the committed GLSL sources. Parses each
    /// `kernels/glsl/<name>.comp` for its `layout(local_size_x = A,
    /// local_size_y = B, local_size_z = C) in;` declaration — a geometry
    /// edit in either place without the other fails here (drift gate, no
    /// fabricated pass).
    #[test]
    fn local_size_table_matches_glsl_sources() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let mut checked = 0usize;
        for shader in crate::spirv::SHADERS {
            if matches!(shader.variant, crate::spirv::ShaderVariant::Handcrafted) {
                continue;
            }
            let table = shader_local_size(shader.name).unwrap_or_else(|| {
                panic!(
                    "glslc manifest entry `{}` missing from shader_local_size",
                    shader.name
                )
            });
            let path = format!("{manifest_dir}/kernels/glsl/{}.comp", shader.name);
            let src = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
            let parsed = parse_local_size(&src)
                .unwrap_or_else(|| panic!("no local_size layout found in {path}"));
            assert_eq!(
                table, parsed,
                "shader_local_size(\"{}\") = {table:?} but {path} declares {parsed:?}",
                shader.name
            );
            checked += 1;
        }
        assert_eq!(checked, 12, "expected 12 glslc kernels cross-checked");
    }

    /// Minimal parser for `layout(local_size_x = A, local_size_y = B,
    /// local_size_z = C) in;` — accepts arbitrary whitespace, requires the
    /// x component, defaults y/z to 1 (GLSL default).
    fn parse_local_size(src: &str) -> Option<[u32; 3]> {
        let start = src.find("local_size_x")?;
        let decl_end = src[start..].find(')')? + start;
        let decl = &src[start..decl_end];
        let mut out = [0u32, 1, 1];
        for (axis, slot) in [
            ("local_size_x", 0),
            ("local_size_y", 1),
            ("local_size_z", 2),
        ] {
            if let Some(pos) = decl.find(axis) {
                let rest = &decl[pos + axis.len()..];
                let digits: String = rest
                    .chars()
                    .skip_while(|c| *c == ' ' || *c == '=')
                    .take_while(char::is_ascii_digit)
                    .collect();
                if !digits.is_empty() {
                    out[slot] = digits.parse().ok()?;
                }
            }
        }
        if out[0] == 0 { None } else { Some(out) }
    }

    // ---- per-kernel plans ------------------------------------------------

    #[test]
    fn gemm_plan_carries_dims_and_tiles_both_axes() {
        // m=17, n=33, k=9 with 16x16 tiles → x = ceil(33/16) = 3, y = ceil(17/16) = 2.
        let p = plan_gemm(GemmPipelineVariant::Subgroup, 17, 33, 9, 17 * 9, 9 * 33).unwrap();
        assert_eq!(p.shader, "gemm_subgroup");
        assert_eq!(p.workgroups, [3, 2, 1]);
        assert_eq!(p.output_len, 17 * 33);
        // PC = m, n, k as LE u32.
        assert_eq!(p.push_constants.len(), 12);
        assert_eq!(&p.push_constants[0..4], &17u32.to_le_bytes());
        assert_eq!(&p.push_constants[4..8], &33u32.to_le_bytes());
        assert_eq!(&p.push_constants[8..12], &9u32.to_le_bytes());
        assert!(p.spec_constants.is_empty());

        // Coop-matrix variant: local (32,1) → x = ceil(33/32) = 2, y = m = 17.
        let p = plan_gemm(GemmPipelineVariant::CoopMatrix, 17, 33, 9, 17 * 9, 9 * 33).unwrap();
        assert_eq!(p.shader, "gemm_coopmat");
        assert_eq!(p.workgroups, [2, 17, 1]);
    }

    #[test]
    fn gemm_plan_rejects_shape_mismatch_and_zero_dims() {
        assert!(matches!(
            plan_gemm(GemmPipelineVariant::Subgroup, 2, 3, 4, 7, 12),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            plan_gemm(GemmPipelineVariant::Subgroup, 2, 3, 4, 8, 11),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            plan_gemm(GemmPipelineVariant::Subgroup, 0, 3, 4, 0, 12),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn gemv_plan_is_one_workgroup_per_row_with_bias_flag() {
        let p = plan_gemv(5, 7, 35, 7, Some(5)).unwrap();
        assert_eq!(p.shader, "gemv");
        assert_eq!(p.workgroups, [5, 1, 1]);
        assert_eq!(p.output_len, 5);
        assert_eq!(
            &p.push_constants[8..12],
            &1u32.to_le_bytes(),
            "bias_present = 1"
        );

        let p = plan_gemv(5, 7, 35, 7, None).unwrap();
        assert_eq!(
            &p.push_constants[8..12],
            &0u32.to_le_bytes(),
            "bias_present = 0"
        );

        // Wrong bias length is an explicit error.
        assert!(plan_gemv(5, 7, 35, 7, Some(4)).is_err());
        // Wrong x length.
        assert!(plan_gemv(5, 7, 35, 6, None).is_err());
    }

    #[test]
    fn rowwise_plans_dispatch_one_group_per_row() {
        for f in [plan_softmax, plan_softmax_causal] {
            let p = f(4, 100, 400).unwrap();
            assert_eq!(p.workgroups, [4, 1, 1]);
            assert_eq!(p.output_len, 400);
            assert_eq!(&p.push_constants[0..4], &4u32.to_le_bytes());
            assert_eq!(&p.push_constants[4..8], &100u32.to_le_bytes());
            assert!(f(4, 100, 399).is_err(), "length mismatch must error");
            assert!(f(0, 100, 0).is_err(), "zero rows must error");
        }
        assert_eq!(plan_softmax(1, 8, 8).unwrap().shader, "softmax");
        assert_eq!(
            plan_softmax_causal(1, 8, 8).unwrap().shader,
            "softmax_causal"
        );
    }

    #[test]
    fn layer_norm_plan_packs_eps_verbatim() {
        let eps = 1e-5f32;
        let p = plan_layer_norm(3, 8, eps, 24, 8, 8).unwrap();
        assert_eq!(p.workgroups, [3, 1, 1]);
        assert_eq!(
            &p.push_constants[8..12],
            &eps.to_le_bytes(),
            "eps passes through verbatim"
        );
        // Affine params must match cols.
        assert!(plan_layer_norm(3, 8, eps, 24, 7, 8).is_err());
        assert!(plan_layer_norm(3, 8, eps, 24, 8, 9).is_err());
    }

    #[test]
    fn pointwise_plans_tile_by_256() {
        let p = plan_gelu(257).unwrap();
        assert_eq!(p.shader, "gelu");
        assert_eq!(p.workgroups, [2, 1, 1]);
        assert!(p.spec_constants.is_empty());

        let p = plan_activation(ActivationKind::Tanh, 256).unwrap();
        assert_eq!(p.shader, "activation");
        assert_eq!(p.workgroups, [1, 1, 1]);
        assert_eq!(
            p.spec_constants,
            vec![SpecConstantU32 {
                constant_id: 0,
                value: 2
            }],
            "KIND spec constant carries the GLSL discriminant"
        );

        assert!(plan_gelu(0).is_err());
    }

    #[test]
    fn elementwise_plan_selects_op_via_spec_constant() {
        let p = plan_elementwise(ElementwiseOp::Mul, 10, 10).unwrap();
        assert_eq!(p.shader, "elementwise");
        assert_eq!(
            p.spec_constants,
            vec![SpecConstantU32 {
                constant_id: 0,
                value: 1
            }]
        );
        let p = plan_elementwise(ElementwiseOp::Add, 10, 10).unwrap();
        assert_eq!(
            p.spec_constants,
            vec![SpecConstantU32 {
                constant_id: 0,
                value: 0
            }]
        );
        assert!(plan_elementwise(ElementwiseOp::Add, 10, 9).is_err());
    }

    #[test]
    fn conv1d_plan_matches_whisper_frontend_shapes() {
        // Whisper conv1: 80 → 512, k=3, s=1, p=1 keeps length; conv2: s=2 halves.
        let conv1 = Conv1dDims {
            batch: 1,
            in_ch: 80,
            out_ch: 512,
            in_len: 100,
            kernel_len: 3,
            stride: 1,
            padding: 1,
        };
        assert_eq!(conv1.out_len().unwrap(), 100);
        let p = plan_conv1d(&conv1, 80 * 100, 512 * 80 * 3, Some(512)).unwrap();
        // x = ceil(100/64) = 2, y = out_ch = 512, z = batch = 1.
        assert_eq!(p.workgroups, [2, 512, 1]);
        assert_eq!(p.output_len, 512 * 100);
        assert_eq!(p.push_constants.len(), 36, "9 u32 fields");
        // out_len is the 5th field (offset 16).
        assert_eq!(&p.push_constants[16..20], &100u32.to_le_bytes());
        // bias_present is the 9th field (offset 32).
        assert_eq!(&p.push_constants[32..36], &1u32.to_le_bytes());

        let conv2 = Conv1dDims {
            stride: 2,
            in_ch: 512,
            in_len: 100,
            ..conv1
        };
        assert_eq!(conv2.out_len().unwrap(), 50);

        // Kernel longer than padded input → explicit error.
        let bad = Conv1dDims {
            in_len: 1,
            padding: 0,
            kernel_len: 3,
            ..conv1
        };
        assert!(bad.out_len().is_err());
        // Weight length mismatch → explicit error.
        assert!(plan_conv1d(&conv1, 80 * 100, 512 * 80 * 2, None).is_err());
    }

    #[test]
    fn transpose_plan_tiles_16x16() {
        let p = plan_transpose(17, 33, 17 * 33).unwrap();
        assert_eq!(p.workgroups, [3, 2, 1]);
        assert_eq!(p.output_len, 17 * 33);
        assert!(plan_transpose(17, 33, 5).is_err());
    }

    #[test]
    fn gather_plan_rejects_out_of_range_index_before_dispatch() {
        // vocab=10, dim=4; index 10 is OOB (NFR-RL-07 explicit reject).
        let err = plan_gather(10, 4, 40, &[0, 9, 10]).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
        let msg = format!("{err}");
        assert!(
            msg.contains("out of range"),
            "diagnostic names the OOB: {msg}"
        );

        let p = plan_gather(10, 4, 40, &[0, 9, 3]).unwrap();
        assert_eq!(p.shader, "gather");
        // x = ceil(dim/64) = 1, y = ceil(n/4) = 1.
        assert_eq!(p.workgroups, [1, 1, 1]);
        assert_eq!(p.output_len, 3 * 4);
        assert_eq!(&p.push_constants[0..4], &3u32.to_le_bytes());
        assert_eq!(&p.push_constants[4..8], &10u32.to_le_bytes());
        assert_eq!(&p.push_constants[8..12], &4u32.to_le_bytes());
    }

    #[test]
    fn every_planned_shader_is_a_manifest_entry() {
        // A plan for a shader the manifest doesn't know would dispatch into
        // a guaranteed `UnsupportedOp`; catch the drift here instead.
        let planned = [
            plan_gemm(GemmPipelineVariant::Subgroup, 1, 1, 1, 1, 1).unwrap(),
            plan_gemm(GemmPipelineVariant::CoopMatrix, 1, 1, 1, 1, 1).unwrap(),
            plan_gemv(1, 1, 1, 1, None).unwrap(),
            plan_softmax(1, 1, 1).unwrap(),
            plan_softmax_causal(1, 1, 1).unwrap(),
            plan_layer_norm(1, 1, 1e-5, 1, 1, 1).unwrap(),
            plan_gelu(1).unwrap(),
            plan_activation(ActivationKind::Relu, 1).unwrap(),
            plan_elementwise(ElementwiseOp::Mul, 1, 1).unwrap(),
            plan_conv1d(
                &Conv1dDims {
                    batch: 1,
                    in_ch: 1,
                    out_ch: 1,
                    in_len: 1,
                    kernel_len: 1,
                    stride: 1,
                    padding: 0,
                },
                1,
                1,
                None,
            )
            .unwrap(),
            plan_transpose(1, 1, 1).unwrap(),
            plan_gather(1, 1, 1, &[0]).unwrap(),
        ];
        for p in &planned {
            assert!(
                crate::spirv::SHADERS.iter().any(|s| s.name == p.shader),
                "plan targets `{}` which is not in the SPIR-V manifest",
                p.shader
            );
        }
        // All 11 glslc op categories are reachable through plans (12 shaders;
        // GEMM's two variants share one op).
        let mut names: Vec<&str> = planned.iter().map(|p| p.shader).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 12, "12 distinct glslc shaders planned");
    }
}
