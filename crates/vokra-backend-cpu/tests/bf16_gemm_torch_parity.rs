//! Independent PyTorch-reference parity for the AVX-512 BF16 GEMM path.
//!
//! The fixtures are generated on VAST by
//! `tools/parity/bf16_gemm/dump_reference.py`.  This test is intentionally
//! ignored until that packet is populated.  It is never a model test: the
//! only inputs are the small deterministic tensors committed with the parity
//! fixture, and the reference output comes from PyTorch `torch.matmul` over
//! BF16-rounded inputs.

#[path = "support/bf16_gemm_fixture.rs"]
mod fixture;

use fixture::Fixture;
use vokra_backend_cpu::kernels;
use vokra_backend_cpu::{CpuFeatures, IsaPath};

fn compare_case(case: &Fixture) {
    let mut actual = vec![f32::NAN; case.m * case.n];
    kernels::gemm_bf16_on(
        IsaPath::Avx512Bf16,
        case.m,
        case.n,
        case.k,
        &case.a,
        &case.b,
        &mut actual,
    )
    .unwrap_or_else(|error| panic!("{}: AVX-512 BF16 GEMM failed: {error}", case.name));
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "{}: non-finite output",
        case.name
    );
    assert_eq!(
        actual.len(),
        case.output.len(),
        "{}: output length",
        case.name
    );
    for (index, (&got, &expected)) in actual.iter().zip(&case.output).enumerate() {
        let tolerance = case.atol + case.rtol * expected.abs();
        assert!(
            (got - expected).abs() <= tolerance,
            "{}: index {index}: AVX512-BF16={got:?}, PyTorch={expected:?}, |diff|={} > tolerance {tolerance}",
            case.name,
            (got - expected).abs()
        );
    }
}

#[test]
#[ignore = "run after VAST PyTorch fixture generation; no local Torch/model execution"]
fn avx512_bf16_gemm_matches_pytorch_reference() {
    let features = CpuFeatures::detect();
    if !features.supports(IsaPath::Avx512Bf16) {
        eprintln!("skip: AVX-512 BF16 is unavailable on this host");
        return;
    }
    for case in fixture::load_all() {
        compare_case(&case);
    }
}
