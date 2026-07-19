//! M5-15-T31/T32/T33: the multi-activation K-quant INT8 kernels.
//!
//! The oracle is **internal and exact**: `kquant_gemm_i8_on` must equal
//! `n_act` separate [`kquant_gemv_i8_on`] calls, element for element, with no
//! tolerance at all. That holds because the existing INT8 contract
//! (`kquant.rs` "All paths share the combine and exact integer sums, so they
//! are bit-identical to each other") makes every accepted ISA path produce the
//! same floats, and because this WP quantizes activations **per vector** — no
//! Q8 scale is ever shared across activation rows, so widening the batch
//! cannot move a single result bit (ADR `M5-15-quant.md` §D2).
//!
//! Every SIMD leg is runtime-detect gated and skips loudly, never fabricated
//! (`server_tier_parity.rs` conventions). On the Apple M1 dev machine the
//! `NeonDotprod` and `Scalar` legs run on real silicon; `NeonI8mm` needs
//! ARMv8.6 (Apple M2+) and the VNNI legs need x86-64, so those skip here.

use vokra_backend_cpu::kernels::{self, KQuantDtype};
use vokra_backend_cpu::{CpuFeatures, IsaPath};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as u32) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next_u64() >> 32) as u8).collect()
    }
}

/// K-quant payload with pinned-small f16 scales (finite magnitudes; every
/// other byte pattern is a valid payload for all three formats). Transcribed
/// from `server_tier_parity.rs::random_blocks` so both suites drive the
/// kernels with the same class of payload.
fn random_blocks(rng: &mut Rng, dtype: KQuantDtype, nb: usize) -> Vec<u8> {
    let mut bytes = rng.bytes(nb * dtype.block_bytes());
    for b in 0..nb {
        let base = b * dtype.block_bytes();
        match dtype {
            KQuantDtype::Q4K | KQuantDtype::Q5K => {
                bytes[base + 1] = 0x2C;
                bytes[base + 3] = 0x24;
            }
            KQuantDtype::Q6K => {
                bytes[base + 209] = 0x2C;
            }
        }
    }
    bytes
}

/// The INT8 tiers this host can actually run, plus `Scalar` (always present).
fn runnable_int8_paths() -> Vec<IsaPath> {
    let f = CpuFeatures::detect();
    [
        IsaPath::Scalar,
        IsaPath::Avx512Vnni,
        IsaPath::AvxVnni256,
        IsaPath::NeonDotprod,
        IsaPath::NeonI8mm,
    ]
    .into_iter()
    .filter(|&isa| isa == IsaPath::Scalar || f.supports(isa))
    .collect()
}

/// The tier the **single-vector** kernel runs an `isa` request on.
///
/// `kquant_gemv_i8_on` has no `NeonI8mm` form — SMMLA needs two activation
/// rows to fill a 2x2 tile, so a 1-activation GEMV cannot use it and the
/// kernel rejects that path outright. Asking the oracle for it would panic on
/// real i8mm silicon rather than test anything; every accepted INT8 tier is
/// bit-identical, so the dot-product tier is an exact reference for the tile
/// kernel's output. (Mirrors `kquant.rs::int8_tail_tier`, which is what the
/// batched kernels use for their odd tail.)
fn single_vector_tier(isa: IsaPath) -> IsaPath {
    if isa == IsaPath::NeonI8mm {
        CpuFeatures::detect()
            .best_int8_isa()
            .unwrap_or(IsaPath::Scalar)
    } else {
        isa
    }
}

/// The `[n_act, m]` reference: `n_act` independent single-activation GEMVs.
fn gemv_oracle(
    isa: IsaPath,
    dtype: KQuantDtype,
    m: usize,
    k: usize,
    n_act: usize,
    w: &[u8],
    x: &[f32],
) -> Vec<f32> {
    let isa = single_vector_tier(isa);
    let mut out = vec![0.0f32; n_act * m];
    for t in 0..n_act {
        kernels::kquant_gemv_i8_on(
            isa,
            dtype,
            m,
            k,
            w,
            &x[t * k..(t + 1) * k],
            &mut out[t * m..(t + 1) * m],
        )
        .unwrap();
    }
    out
}

/// T31: the GEMM entry equals `n_act` GEMV applications **exactly**, on every
/// runnable path and across ragged `(m, n_act, k)` shapes (including the odd
/// `n_act` that leaves an unpaired tail for the SMMLA tile path).
#[test]
fn quant_gemm_equals_repeated_gemv_on_every_path() {
    let mut rng = Rng::new(0x5115_0031);
    // n_act 1 and 3 exercise the odd tail; 2 and 4 the full tile pairs.
    // n = 1 exercises the single-row weight; k = 512 the multi-super-block row.
    let shapes = [
        (1usize, 1usize, 256usize),
        (3, 1, 256),
        (7, 2, 256),
        (5, 3, 512),
        (8, 4, 512),
        (33, 5, 256),
    ];
    for dtype in [KQuantDtype::Q4K, KQuantDtype::Q5K, KQuantDtype::Q6K] {
        for &(m, n_act, k) in &shapes {
            let nb = k / 256;
            let w = random_blocks(&mut rng, dtype, m * nb);
            let x = rng.vec(n_act * k);
            for isa in runnable_int8_paths() {
                let want = gemv_oracle(isa, dtype, m, k, n_act, &w, &x);
                let mut got = vec![0.0f32; n_act * m];
                kernels::kquant_gemm_i8_on(isa, dtype, m, k, n_act, &w, &x, &mut got).unwrap();
                assert_eq!(
                    got, want,
                    "{dtype:?} GEMM on {isa} (m={m}, n_act={n_act}, k={k}) != repeated GEMV"
                );
            }
        }
    }
}

/// T32: `kquant_gemvn_i8_on` generalizes `kquant_gemv2_i8_on` — for
/// `n_act == 2` it must reproduce the 2-activation kernel's output byte for
/// byte (same `[m][n_act]` interleaved layout), and for any `n_act` it must
/// agree with the repeated single-activation kernel.
#[test]
fn gemvn_generalizes_gemv2_and_matches_single_activation() {
    let mut rng = Rng::new(0x5115_0032);
    let dtype = KQuantDtype::Q5K;
    let (m, k) = (9usize, 512usize);
    let w = random_blocks(&mut rng, dtype, m * (k / 256));

    for isa in runnable_int8_paths() {
        // (a) n_act == 2 is bit-for-bit the existing gemv2 contract.
        let x2 = rng.vec(2 * k);
        let mut want2 = vec![0.0f32; 2 * m];
        kernels::kquant_gemv2_i8_on(isa, dtype, m, k, &w, &x2, &mut want2).unwrap();
        let mut got2 = vec![0.0f32; 2 * m];
        kernels::kquant_gemvn_i8_on(isa, dtype, m, k, 2, &w, &x2, &mut got2).unwrap();
        assert_eq!(got2, want2, "gemvn(n_act=2) on {isa} != gemv2");

        // (b) any n_act agrees with the single-activation kernel, transposed
        //     into the `[m][n_act]` layout.
        for n_act in [1usize, 3, 6] {
            let x = rng.vec(n_act * k);
            let flat = gemv_oracle(isa, dtype, m, k, n_act, &w, &x); // [n_act, m]
            let mut got = vec![0.0f32; n_act * m];
            kernels::kquant_gemvn_i8_on(isa, dtype, m, k, n_act, &w, &x, &mut got).unwrap();
            for i in 0..m {
                for c in 0..n_act {
                    assert_eq!(
                        got[n_act * i + c],
                        flat[c * m + i],
                        "gemvn on {isa} (n_act={n_act}) row {i} col {c}"
                    );
                }
            }
        }
    }
}

/// T31: the quant GEMM tracks the f32 dequant GEMM within the **derived**
/// activation-quantization bound (`int8_error_bound × 2`, the existing
/// `server_tier_parity.rs` rule), not a hand-tuned constant. Asserted
/// per output element with its own row/activation bound.
#[test]
fn quant_gemm_tracks_f32_within_derived_bound() {
    let mut rng = Rng::new(0x5115_0033);
    let (m, k, n_act) = (6usize, 512usize, 3usize);
    let nb = k / 256;
    for dtype in [KQuantDtype::Q4K, KQuantDtype::Q5K, KQuantDtype::Q6K] {
        let w = random_blocks(&mut rng, dtype, m * nb);
        let x = rng.vec(n_act * k);
        let mut got = vec![0.0f32; n_act * m];
        kernels::kquant_gemm_i8_on(IsaPath::Scalar, dtype, m, k, n_act, &w, &x, &mut got).unwrap();

        let row_bytes = nb * dtype.block_bytes();
        for i in 0..m {
            let row = &w[i * row_bytes..(i + 1) * row_bytes];
            let wf = kernels::kquant_dequant_on(IsaPath::Scalar, dtype, row, k).unwrap();
            for t in 0..n_act {
                let xt = &x[t * k..(t + 1) * k];
                let f32_dot: f32 = (0..k).map(|l| wf[l] * xt[l]).sum();
                let bound = kernels::int8_error_bound(dtype, row, xt).max(1e-6);
                let diff = (got[t * m + i] - f32_dot).abs();
                assert!(
                    diff <= 2.0 * bound,
                    "{dtype:?} row {i} act {t}: int8 {} vs f32 {f32_dot} \
                     (|diff| {diff}) exceeds 2x derived bound {bound}",
                    got[t * m + i]
                );
            }
        }
    }
}

/// FR-EX-08: malformed shapes are explicit errors, never a partial write or a
/// silent widen. `n_act == 0` in particular must not pass validation just
/// because `0 * k == 0` lengths line up.
#[test]
fn quant_gemm_rejects_malformed_shapes() {
    let mut rng = Rng::new(0x5115_0034);
    let dtype = KQuantDtype::Q6K;
    let (m, k) = (4usize, 256usize);
    let w = random_blocks(&mut rng, dtype, m);
    let x = rng.vec(2 * k);
    let mut out = vec![0.0f32; 2 * m];

    // n_act = 0.
    assert!(
        kernels::kquant_gemm_i8_on(IsaPath::Scalar, dtype, m, k, 0, &w, &[], &mut []).is_err(),
        "n_act = 0 must be rejected"
    );
    assert!(
        kernels::kquant_gemvn_i8_on(IsaPath::Scalar, dtype, m, k, 0, &w, &[], &mut []).is_err(),
        "n_act = 0 must be rejected"
    );
    // k not a multiple of the 256-element super-block.
    assert!(
        kernels::kquant_gemm_i8_on(IsaPath::Scalar, dtype, m, 255, 2, &w, &x, &mut out).is_err(),
        "k = 255 must be rejected"
    );
    // Output too short.
    let mut short = vec![0.0f32; m];
    assert!(
        kernels::kquant_gemm_i8_on(IsaPath::Scalar, dtype, m, k, 2, &w, &x, &mut short).is_err(),
        "short output must be rejected"
    );
}

/// T33: `kernels::gemm_q_f32` (the `gemm_driver` quant-B entry) computes the
/// `nn.Linear` shape `out[t, j] = bias[j] + Σ_l a[t, l] · dequant(wq[j, l])`
/// and equals the raw kernel plus the bias add. The f32 `gemm_f32` route is
/// untouched by construction (separate entry), which
/// `gemm_packed_parity.rs` continues to pin.
#[test]
fn gemm_q_f32_applies_bias_over_the_kernel_result() {
    let mut rng = Rng::new(0x5115_0035);
    let dtype = KQuantDtype::Q4K;
    let (m, n, k) = (3usize, 5usize, 256usize); // m = activations, n = out features
    let wq = random_blocks(&mut rng, dtype, n * (k / 256));
    let a = rng.vec(m * k);
    let bias = rng.vec(n);

    let mut raw = vec![0.0f32; m * n];
    kernels::kquant_gemm_i8(dtype, n, k, m, &wq, &a, &mut raw).unwrap();

    let mut got = vec![0.0f32; m * n];
    kernels::gemm_q_f32(m, n, k, &a, &wq, dtype, Some(&bias), &mut got).unwrap();
    for t in 0..m {
        for j in 0..n {
            assert_eq!(
                got[t * n + j],
                raw[t * n + j] + bias[j],
                "gemm_q_f32 bias at ({t}, {j})"
            );
        }
    }

    // Without a bias the wrapper is the kernel verbatim.
    let mut nobias = vec![0.0f32; m * n];
    kernels::gemm_q_f32(m, n, k, &a, &wq, dtype, None, &mut nobias).unwrap();
    assert_eq!(nobias, raw, "gemm_q_f32 without bias != kernel");
}

/// M5-15 regression — **odd activation count on the SMMLA tier**. The tile
/// consumes activations two at a time, so the last vector of an odd batch
/// falls through to the single-vector kernel; that kernel has no i8mm form,
/// and it used to be handed the caller's `isa` verbatim, making every odd
/// `n_act` on an i8mm host a hard `UnsupportedOp` instead of a result.
///
/// Runs only on real ARMv8.6 silicon (Apple M2+ / Neoverse V2 / Graviton3);
/// this suite's dev machine (Apple M1) has no i8mm, so the tier mapping
/// itself is pinned host-independently by the `kquant.rs` unit test
/// `i8mm_tail_tier_is_runnable_by_the_single_vector_kernel`, and the loud
/// error on a host *without* i8mm by
/// `i8mm_batch_on_a_host_without_i8mm_is_backend_unavailable`.
#[test]
fn i8mm_odd_activation_tail_computes_and_matches_the_single_vector_kernel() {
    if !CpuFeatures::detect().supports(IsaPath::NeonI8mm) {
        eprintln!(
            "skipping i8mm odd-tail leg: host has no ARMv8.6 SMMLA \
             (covered host-independently in kquant.rs unit tests)"
        );
        return;
    }
    let mut rng = Rng::new(0x5115_0037);
    // Odd n_act only — each leaves exactly one unpaired tail vector.
    let shapes = [
        (1usize, 1usize, 256usize),
        (5, 3, 256),
        (7, 5, 512),
        (4, 7, 256),
    ];
    for dtype in [KQuantDtype::Q4K, KQuantDtype::Q5K, KQuantDtype::Q6K] {
        for &(m, n_act, k) in &shapes {
            let w = random_blocks(&mut rng, dtype, m * (k / 256));
            let x = rng.vec(n_act * k);
            let want = gemv_oracle(IsaPath::NeonI8mm, dtype, m, k, n_act, &w, &x); // [n_act, m]

            let mut gemm = vec![0.0f32; n_act * m];
            kernels::kquant_gemm_i8_on(IsaPath::NeonI8mm, dtype, m, k, n_act, &w, &x, &mut gemm)
                .expect("odd n_act on the i8mm tier must compute, not error");
            assert_eq!(
                gemm, want,
                "{dtype:?} i8mm GEMM (m={m}, n_act={n_act}, k={k}) != repeated GEMV"
            );

            let mut gemvn = vec![0.0f32; n_act * m];
            kernels::kquant_gemvn_i8_on(IsaPath::NeonI8mm, dtype, m, k, n_act, &w, &x, &mut gemvn)
                .expect("odd n_act on the i8mm tier must compute, not error");
            for i in 0..m {
                for c in 0..n_act {
                    assert_eq!(
                        gemvn[n_act * i + c],
                        want[c * m + i],
                        "{dtype:?} i8mm gemvn (n_act={n_act}) row {i} col {c}"
                    );
                }
            }
        }
    }
}

/// The host-tier auto-selector agrees with the forced path it claims to pick.
/// (`best_int8_isa` never returns `NeonI8mm`; the GEMM selector prefers the
/// SMMLA tile only when there are ≥ 2 activations — every tier is
/// bit-identical, so this is a speed choice whose result must not move.)
#[test]
fn quant_gemm_auto_selector_matches_a_forced_runnable_path() {
    let mut rng = Rng::new(0x5115_0036);
    let dtype = KQuantDtype::Q6K;
    let (m, k, n_act) = (5usize, 256usize, 4usize);
    let w = random_blocks(&mut rng, dtype, m);
    let x = rng.vec(n_act * k);

    let mut auto = vec![0.0f32; n_act * m];
    kernels::kquant_gemm_i8(dtype, m, k, n_act, &w, &x, &mut auto).unwrap();
    // Every accepted INT8 tier is bit-identical, so the scalar reference is a
    // valid oracle for whichever tier the selector chose.
    let mut scalar = vec![0.0f32; n_act * m];
    kernels::kquant_gemm_i8_on(IsaPath::Scalar, dtype, m, k, n_act, &w, &x, &mut scalar).unwrap();
    assert_eq!(auto, scalar, "auto INT8 GEMM selection != scalar reference");
}
