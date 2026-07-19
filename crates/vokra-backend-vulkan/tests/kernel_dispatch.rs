//! M4-13-T03/T04 — typed kernel dispatch arms (`gemm_f32` / `gemv_f32`) on
//! [`VulkanBackend`], exercising the placeholder-then-swap seam end-to-end.
//!
//! Three-way behaviour, all asserted here:
//!
//! 1. **No Vulkan host** (Apple authoring host / default build): every test
//!    skips cleanly at `VulkanBackend::new()` — never a silent CPU fallback.
//! 2. **Vulkan host, blob not committed** (foundation slice — the state
//!    lavapipe CI runs in until the owner's M4-13-T16 glslc commit): a
//!    correctly-shaped dispatch surfaces the explicit
//!    [`VokraError::UnsupportedOp`] from `spirv::require_blob`, while a
//!    badly-shaped call still fails host-side with `InvalidArgument`
//!    *before* the blob check (validation precedes dispatch).
//! 3. **Vulkan host, blob committed** (post-T16): the same call computes,
//!    and is checked against the CPU backend's kernel of identical
//!    shape/semantics within the NFR-QL-01 FP32 gate (atol = 0.01).
//!
//! The branch between 2 and 3 is `spirv::has_blob` — the tests stay green
//! across the owner's blob commit without edits (no fabricated pass: which
//! branch ran is visible in the test log).

use vokra_backend_vulkan::plan::{ActivationKind, Conv1dDims, ElementwiseOp};
use vokra_backend_vulkan::{GemmPipelinePreference, VulkanBackend, spirv};
use vokra_core::VokraError;

/// FP32 parity gate (NFR-QL-01).
const ATOL: f32 = 0.01;

/// Deterministic pseudo-random f32s in [-1, 1) — SplitMix64-based, the same
/// synthesized-input pattern the M3-09 LLM parity tests use (no external
/// crates, NFR-DS-02).
fn splitmix_f32s(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            // 24 mantissa-ish bits → [-1, 1).
            ((z >> 40) as f32) / ((1u64 << 23) as f32) - 1.0
        })
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= ATOL,
            "{what}: diverged at {i}: got {g}, want {w} (atol {ATOL})"
        );
    }
}

/// `gemm_f32` through the ForceSubgroup path: blob-gated compute vs CPU
/// `gemm_f32`, or explicit `UnsupportedOp` while the blob is absent.
#[test]
fn gemm_subgroup_dispatch_is_blob_gated_and_cpu_close() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan on this host; skipping gemm dispatch test");
        return;
    };
    // Ragged dims exercise both tile tails (16x16 workgroups).
    let (m, n, k) = (17usize, 33usize, 9usize);
    let a = splitmix_f32s(1, m * k);
    let b = splitmix_f32s(2, k * n);

    let result = backend.gemm_f32(GemmPipelinePreference::ForceSubgroup, m, n, k, &a, &b);
    if spirv::has_blob("gemm_subgroup") {
        let got = result.expect("blob committed; gemm must dispatch");
        let mut want = vec![0.0f32; m * n];
        vokra_backend_cpu::kernels::gemm_f32(m, n, k, &a, &b, None, &mut want)
            .expect("CPU reference");
        assert_close(&got, &want, "gemm_subgroup vs CPU");
        eprintln!("gemm_subgroup parity vs CPU green over {m}x{n}x{k}");
    } else {
        let err = result.expect_err("no .spv committed; gemm must be UnsupportedOp");
        assert!(
            matches!(err, VokraError::UnsupportedOp(_)),
            "expected UnsupportedOp while gemm_subgroup.spv is absent, got {err:?}"
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("gemm_subgroup"),
            "diagnostic names the missing blob: {msg}"
        );
        eprintln!("gemm_subgroup blob absent → explicit UnsupportedOp (placeholder slice)");
    }
}

/// The default preference resolves to whatever variant the probe selected;
/// dispatch must route to exactly that variant's blob.
#[test]
fn gemm_default_preference_follows_probe_selection() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan on this host; skipping gemm variant-routing test");
        return;
    };
    let variant = backend
        .select_gemm_pipeline_variant(GemmPipelinePreference::default())
        .expect("the default preference (PreferCoopMatrix) never errors");
    let shader = variant.shader_name();
    let (m, n, k) = (4usize, 4usize, 4usize);
    let a = splitmix_f32s(3, m * k);
    let b = splitmix_f32s(4, k * n);
    let result = backend.gemm_f32(GemmPipelinePreference::default(), m, n, k, &a, &b);
    if spirv::has_blob(shader) {
        assert!(
            result.is_ok(),
            "selected variant `{shader}` blob present; must dispatch"
        );
    } else {
        let msg = format!("{}", result.expect_err("selected variant blob absent"));
        assert!(
            msg.contains(shader),
            "UnsupportedOp must name the *selected* variant `{shader}`: {msg}"
        );
    }
}

/// Host-side shape validation fires BEFORE the blob check / any GPU work —
/// even in the foundation slice with no blob committed.
#[test]
fn gemm_shape_mismatch_is_invalid_argument_before_blob_check() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan on this host; skipping gemm validation test");
        return;
    };
    let a = vec![0.0f32; 7]; // not m*k = 8
    let b = vec![0.0f32; 12];
    let err = backend
        .gemm_f32(GemmPipelinePreference::ForceSubgroup, 2, 3, 4, &a, &b)
        .expect_err("shape mismatch must error");
    assert!(
        matches!(err, VokraError::InvalidArgument(_)),
        "expected InvalidArgument (validation precedes dispatch), got {err:?}"
    );
}

/// `gemv_f32` with and without bias: blob-gated compute vs CPU `gemv_f32`,
/// or explicit `UnsupportedOp`.
#[test]
fn gemv_dispatch_is_blob_gated_and_cpu_close() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan on this host; skipping gemv dispatch test");
        return;
    };
    // m=37 rows (one workgroup each), n=100 exercises the strided inner loop
    // (100 > 32 lanes → multiple strides + ragged tail).
    let (m, n) = (37usize, 100usize);
    let a = splitmix_f32s(5, m * n);
    let x = splitmix_f32s(6, n);
    let bias = splitmix_f32s(7, m);

    for bias_arg in [None, Some(bias.as_slice())] {
        let result = backend.gemv_f32(m, n, &a, &x, bias_arg);
        if spirv::has_blob("gemv") {
            let got = result.expect("blob committed; gemv must dispatch");
            let mut want = vec![0.0f32; m];
            vokra_backend_cpu::kernels::gemv_f32(m, n, &a, &x, bias_arg, &mut want)
                .expect("CPU reference");
            assert_close(&got, &want, "gemv vs CPU");
        } else {
            let err = result.expect_err("no .spv committed; gemv must be UnsupportedOp");
            assert!(
                matches!(err, VokraError::UnsupportedOp(_)),
                "expected UnsupportedOp while gemv.spv is absent, got {err:?}"
            );
        }
    }
}

/// gemv shape validation (x length) fires host-side on a Vulkan host too.
#[test]
fn gemv_shape_mismatch_is_invalid_argument_before_blob_check() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan on this host; skipping gemv validation test");
        return;
    };
    let a = vec![0.0f32; 6];
    let x = vec![0.0f32; 2]; // A is 2x3 → x must be 3
    let err = backend
        .gemv_f32(2, 3, &a, &x, None)
        .expect_err("x length mismatch must error");
    assert!(matches!(err, VokraError::InvalidArgument(_)));
}

/// `softmax_f32` (M4-13-T05): blob-gated compute checked against the CPU
/// kernel AND the mathematical invariants (each row sums to 1; constant
/// shift leaves the output unchanged within FP32).
#[test]
fn softmax_dispatch_is_blob_gated_and_cpu_close() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan on this host; skipping softmax dispatch test");
        return;
    };
    let (rows, cols) = (5usize, 100usize); // cols > 32 lanes → strided loop + tail
    let x = splitmix_f32s(8, rows * cols);

    let result = backend.softmax_f32(rows, cols, &x);
    if spirv::has_blob("softmax") {
        let got = result.expect("blob committed; softmax must dispatch");
        let mut want = vec![0.0f32; rows * cols];
        vokra_backend_cpu::kernels::softmax_f32(&x, &mut want, rows, cols).expect("CPU reference");
        assert_close(&got, &want, "softmax vs CPU");
        // Invariant: each row sums to 1 within FP32 rounding.
        for r in 0..rows {
            let s: f32 = got[r * cols..(r + 1) * cols].iter().sum();
            assert!((s - 1.0).abs() <= 1e-4, "row {r} sums to {s}, want 1.0");
        }
        // Invariant: softmax(x + c) == softmax(x).
        let shifted: Vec<f32> = x.iter().map(|v| v + 3.5).collect();
        let got_shifted = backend
            .softmax_f32(rows, cols, &shifted)
            .expect("shifted softmax");
        assert_close(&got_shifted, &got, "softmax shift invariance");
    } else {
        let err = result.expect_err("no .spv committed; softmax must be UnsupportedOp");
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
        eprintln!("softmax blob absent → explicit UnsupportedOp (placeholder slice)");
    }
}

/// `softmax_causal_f32` (M4-13-T05): masked columns are exactly 0.0 and the
/// unmasked region matches a host-masked CPU softmax (`exp(-inf) = 0`
/// equivalence — the Metal / CUDA causal contract).
#[test]
fn softmax_causal_dispatch_matches_host_masked_cpu_softmax() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan on this host; skipping softmax_causal dispatch test");
        return;
    };
    let (rows, cols) = (6usize, 6usize);
    let x = splitmix_f32s(9, rows * cols);

    let result = backend.softmax_causal_f32(rows, cols, &x);
    if spirv::has_blob("softmax_causal") {
        let got = result.expect("blob committed; softmax_causal must dispatch");
        // Host reference: mask j > i with -inf, then CPU softmax.
        let mut masked = x.clone();
        for i in 0..rows {
            for j in 0..cols {
                if j > i {
                    masked[i * cols + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mut want = vec![0.0f32; rows * cols];
        vokra_backend_cpu::kernels::softmax_f32(&masked, &mut want, rows, cols)
            .expect("CPU reference");
        assert_close(&got, &want, "softmax_causal vs host-masked CPU softmax");
        // Masked cols are written as EXACTLY 0.0 (not merely small).
        for i in 0..rows {
            for j in (i + 1)..cols {
                assert_eq!(
                    got[i * cols + j].to_bits(),
                    0.0f32.to_bits(),
                    "masked ({i},{j}) must be exactly 0.0"
                );
            }
        }
    } else {
        let err = result.expect_err("no .spv committed; softmax_causal must be UnsupportedOp");
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
        eprintln!("softmax_causal blob absent → explicit UnsupportedOp (placeholder slice)");
    }
}

/// `layer_norm_f32` (M4-13-T06): blob-gated compute vs the CPU kernel with
/// the SAME eps (passed through verbatim, never invented), plus the γ=1 /
/// β=0 zero-mean-unit-variance invariant.
#[test]
fn layer_norm_dispatch_is_blob_gated_and_cpu_close() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan on this host; skipping layer_norm dispatch test");
        return;
    };
    let (rows, cols) = (4usize, 100usize);
    let eps = 1e-5f32; // the CPU-default documented value; models pass their config's
    let x = splitmix_f32s(10, rows * cols);
    let gamma = splitmix_f32s(11, cols);
    let beta = splitmix_f32s(12, cols);

    let result = backend.layer_norm_f32(rows, cols, eps, &x, &gamma, &beta);
    if spirv::has_blob("layer_norm") {
        let got = result.expect("blob committed; layer_norm must dispatch");
        let mut want = vec![0.0f32; rows * cols];
        vokra_backend_cpu::kernels::layer_norm_f32(&x, &mut want, rows, cols, &gamma, &beta, eps)
            .expect("CPU reference");
        assert_close(&got, &want, "layer_norm vs CPU");

        // γ=1, β=0 → each output row has mean ~0 and variance ~1.
        let ones = vec![1.0f32; cols];
        let zeros = vec![0.0f32; cols];
        let unit = backend
            .layer_norm_f32(rows, cols, eps, &x, &ones, &zeros)
            .expect("unit-affine layer_norm");
        for r in 0..rows {
            let row = &unit[r * cols..(r + 1) * cols];
            let mean: f32 = row.iter().sum::<f32>() / cols as f32;
            let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
            assert!(mean.abs() <= 1e-3, "row {r} mean {mean} not ~0");
            assert!((var - 1.0).abs() <= 1e-2, "row {r} var {var} not ~1");
        }
    } else {
        let err = result.expect_err("no .spv committed; layer_norm must be UnsupportedOp");
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
        eprintln!("layer_norm blob absent → explicit UnsupportedOp (placeholder slice)");
    }
}

/// `gelu_f32` (M4-13-T06): blob-gated compute vs the CPU's exact (erf-based)
/// GELU — the A&S 7.1.26 coefficients are shared between the GLSL and the
/// CPU kernel, so agreement is far tighter than the 0.01 gate.
#[test]
fn gelu_dispatch_matches_cpu_erf_form() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan on this host; skipping gelu dispatch test");
        return;
    };
    // 300 elements → 2 workgroups of 256 with a ragged tail; range stretched
    // to ±4 to cover both erf saturation regions.
    let x: Vec<f32> = splitmix_f32s(13, 300).iter().map(|v| v * 4.0).collect();

    let result = backend.gelu_f32(&x);
    if spirv::has_blob("gelu") {
        let got = result.expect("blob committed; gelu must dispatch");
        let mut want = vec![0.0f32; x.len()];
        vokra_backend_cpu::kernels::gelu_f32(&x, &mut want).expect("CPU reference");
        assert_close(&got, &want, "gelu vs CPU (erf form)");
        // Hand-computed anchors: gelu(0) = 0; gelu(x) → x for large x;
        // gelu(-x) → 0 for large x.
        let anchors = backend
            .gelu_f32(&[0.0, 6.0, -6.0])
            .expect("anchor dispatch");
        assert!(anchors[0].abs() <= 1e-6, "gelu(0) = 0");
        assert!((anchors[1] - 6.0).abs() <= 1e-3, "gelu(6) ≈ 6");
        assert!(anchors[2].abs() <= 1e-3, "gelu(-6) ≈ 0");
    } else {
        let err = result.expect_err("no .spv committed; gelu must be UnsupportedOp");
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
        eprintln!("gelu blob absent → explicit UnsupportedOp (placeholder slice)");
    }
}

/// `conv1d_f32` (M4-13-T07): Whisper front-end stride/padding envelope,
/// batch=1 vs the CPU `conv1d_f32` oracle, and batch=2 vs two independent
/// CPU calls (the CPU kernel is single-batch).
#[test]
fn conv1d_dispatch_is_blob_gated_and_cpu_close() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan on this host; skipping conv1d dispatch test");
        return;
    };
    // Whisper-shaped (downscaled channels): k=3, conv1 s=1 p=1, conv2 s=2 p=1.
    let (in_ch, out_ch, in_len) = (8usize, 12usize, 20usize);
    for (stride, padding) in [(1usize, 1usize), (2, 1)] {
        let dims = Conv1dDims {
            batch: 2,
            in_ch,
            out_ch,
            in_len,
            kernel_len: 3,
            stride,
            padding,
        };
        let out_len = dims.out_len().expect("valid dims");
        let input = splitmix_f32s(20 + stride as u64, dims.batch * in_ch * in_len);
        let weight = splitmix_f32s(22, out_ch * in_ch * 3);
        let bias = splitmix_f32s(23, out_ch);

        let result = backend.conv1d_f32(&dims, &input, &weight, Some(&bias));
        if spirv::has_blob("conv1d") {
            let got = result.expect("blob committed; conv1d must dispatch");
            // CPU oracle is single-batch: run it once per batch item.
            let mut want = Vec::with_capacity(dims.batch * out_ch * out_len);
            for b in 0..dims.batch {
                let slice = &input[b * in_ch * in_len..(b + 1) * in_ch * in_len];
                let mut out_b = vec![0.0f32; out_ch * out_len];
                vokra_backend_cpu::kernels::conv1d_f32(
                    slice,
                    in_ch,
                    in_len,
                    &weight,
                    out_ch,
                    3,
                    Some(&bias),
                    stride,
                    padding,
                    &mut out_b,
                )
                .expect("CPU reference");
                want.extend_from_slice(&out_b);
            }
            assert_close(&got, &want, "conv1d vs CPU (per-batch)");
        } else {
            let err = result.expect_err("no .spv committed; conv1d must be UnsupportedOp");
            assert!(matches!(err, VokraError::UnsupportedOp(_)));
            eprintln!(
                "conv1d blob absent → explicit UnsupportedOp (placeholder slice, s={stride})"
            );
        }
    }
}

/// `elementwise_f32` (M4-13-T07): the OP specialization constant selects
/// add vs mul; both checked against the CPU kernels.
#[test]
fn elementwise_dispatch_selects_add_and_mul_via_spec_constant() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan on this host; skipping elementwise dispatch test");
        return;
    };
    let n = 300usize; // 2 workgroups of 256 + ragged tail
    let a = splitmix_f32s(24, n);
    let b = splitmix_f32s(25, n);

    for (op, name) in [(ElementwiseOp::Add, "add"), (ElementwiseOp::Mul, "mul")] {
        let result = backend.elementwise_f32(op, &a, &b);
        if spirv::has_blob("elementwise") {
            let got = result.expect("blob committed; elementwise must dispatch");
            let mut want = vec![0.0f32; n];
            match op {
                ElementwiseOp::Add => {
                    vokra_backend_cpu::kernels::add_f32(&a, &b, &mut want).expect("CPU add")
                }
                ElementwiseOp::Mul => {
                    vokra_backend_cpu::kernels::mul_f32(&a, &b, &mut want).expect("CPU mul")
                }
            }
            assert_close(&got, &want, name);
        } else {
            let err = result.expect_err("no .spv committed; elementwise must be UnsupportedOp");
            assert!(matches!(err, VokraError::UnsupportedOp(_)));
            eprintln!("elementwise({name}) blob absent → explicit UnsupportedOp");
        }
    }
}

/// `activation_f32` (M4-13-T07): the KIND specialization constant selects
/// relu / sigmoid / tanh; each checked against its CPU kernel.
#[test]
fn activation_dispatch_selects_kind_via_spec_constant() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan on this host; skipping activation dispatch test");
        return;
    };
    let x: Vec<f32> = splitmix_f32s(26, 300).iter().map(|v| v * 3.0).collect();

    type CpuFn = fn(&[f32], &mut [f32]) -> vokra_core::Result<()>;
    let cases: [(ActivationKind, &str, CpuFn); 3] = [
        (
            ActivationKind::Relu,
            "relu",
            vokra_backend_cpu::kernels::relu_f32,
        ),
        (
            ActivationKind::Sigmoid,
            "sigmoid",
            vokra_backend_cpu::kernels::sigmoid_f32,
        ),
        (
            ActivationKind::Tanh,
            "tanh",
            vokra_backend_cpu::kernels::tanh_f32,
        ),
    ];
    for (kind, name, cpu) in cases {
        let result = backend.activation_f32(kind, &x);
        if spirv::has_blob("activation") {
            let got = result.expect("blob committed; activation must dispatch");
            let mut want = vec![0.0f32; x.len()];
            cpu(&x, &mut want).expect("CPU reference");
            assert_close(&got, &want, name);
        } else {
            let err = result.expect_err("no .spv committed; activation must be UnsupportedOp");
            assert!(matches!(err, VokraError::UnsupportedOp(_)));
            eprintln!("activation({name}) blob absent → explicit UnsupportedOp");
        }
    }
}

/// `transpose_f32` (M4-13-T08): checked against a host reference; a double
/// transpose round-trips bit-for-bit (pure data movement).
#[test]
fn transpose_dispatch_round_trips() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan on this host; skipping transpose dispatch test");
        return;
    };
    let (m, n) = (17usize, 33usize); // ragged vs the 16x16 tiles
    let x = splitmix_f32s(27, m * n);

    let result = backend.transpose_f32(m, n, &x);
    if spirv::has_blob("transpose") {
        let got = result.expect("blob committed; transpose must dispatch");
        // Host reference.
        let mut want = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                want[j * m + i] = x[i * n + j];
            }
        }
        assert_close(&got, &want, "transpose vs host");
        // Double transpose is the identity, bit-for-bit (no arithmetic).
        let back = backend.transpose_f32(n, m, &got).expect("second transpose");
        for (i, (orig, rt)) in x.iter().zip(&back).enumerate() {
            assert_eq!(orig.to_bits(), rt.to_bits(), "round-trip diverged at {i}");
        }
    } else {
        let err = result.expect_err("no .spv committed; transpose must be UnsupportedOp");
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
        eprintln!("transpose blob absent → explicit UnsupportedOp (placeholder slice)");
    }
}

/// `gather_f32` (M4-13-T08): embedding rows come back verbatim, and an
/// out-of-range index is rejected host-side (InvalidArgument) BEFORE any
/// dispatch — even in the foundation slice (validation precedes the blob
/// check).
#[test]
fn gather_dispatch_looks_up_rows_and_rejects_oob_indices() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan on this host; skipping gather dispatch test");
        return;
    };
    let (vocab, dim) = (50usize, 70usize); // dim > 64 → 2 x-groups
    let table = splitmix_f32s(28, vocab * dim);
    let indices: Vec<u32> = vec![0, 49, 7, 7, 21];

    // OOB index fires host-side regardless of blob availability.
    let err = backend
        .gather_f32(vocab, dim, &table, &[0, 50])
        .expect_err("index 50 is OOB for vocab 50");
    assert!(
        matches!(err, VokraError::InvalidArgument(_)),
        "OOB index must be InvalidArgument before dispatch, got {err:?}"
    );

    let result = backend.gather_f32(vocab, dim, &table, &indices);
    if spirv::has_blob("gather") {
        let got = result.expect("blob committed; gather must dispatch");
        assert_eq!(got.len(), indices.len() * dim);
        for (row, &idx) in indices.iter().enumerate() {
            let got_row = &got[row * dim..(row + 1) * dim];
            let want_row = &table[idx as usize * dim..(idx as usize + 1) * dim];
            for (c, (g, w)) in got_row.iter().zip(want_row).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "gather row {row} col {c} must be a verbatim copy"
                );
            }
        }
    } else {
        let err = result.expect_err("no .spv committed; gather must be UnsupportedOp");
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
        eprintln!("gather blob absent → explicit UnsupportedOp (placeholder slice)");
    }
}
