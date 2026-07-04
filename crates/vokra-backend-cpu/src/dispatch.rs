//! Kernel dispatch table and process-wide ISA selection (M0-08-T04).
//!
//! Dispatch is the llama.cpp / OpenBLAS style (FR-EX-06): a table of function
//! pointers to statically compiled kernels, selected **once** per process
//! from [`crate::features::resolve_isa`] and cached in a [`OnceLock`]. There
//! is no JIT and no runtime code generation (NFR-RL-05) — only a choice of
//! which already-compiled function to call.
//!
//! # Relationship to FR-EX-08
//!
//! FR-EX-08 ("uniform op coverage; unsupported ops are an explicit error, no
//! silent CPU fallback") is a rule **between backends**. The ISA selection
//! here is **inside** the CPU backend: every path (`Scalar` / `Avx2` /
//! `Neon`) computes the *same* op with the *same* result (within FP32
//! rounding). Selecting a SIMD path is therefore never a silent fallback.

use std::sync::OnceLock;

use vokra_core::{Result, VokraError};

use crate::features::{self, CpuFeatures, IsaPath};
use crate::kernels::scalar;

/// GEMM kernel signature (see [`scalar::gemm`]); inputs are pre-validated.
pub(crate) type GemmKernel = fn(usize, usize, usize, &[f32], &[f32], Option<&[f32]>, &mut [f32]);
/// GEMV (matrix-vector) kernel signature (see [`scalar::gemv`]); inputs are
/// pre-validated. `(m, k, a[m*k], x[k], bias?[m], out[m])`.
pub(crate) type GemvKernel = fn(usize, usize, &[f32], &[f32], Option<&[f32]>, &mut [f32]);
/// Element-wise binary kernel signature (`add` / `mul`).
pub(crate) type BinaryKernel = fn(&[f32], &[f32], &mut [f32]);
/// Element-wise unary kernel signature (`relu` / `sigmoid` / `tanh` / `gelu`).
pub(crate) type UnaryKernel = fn(&[f32], &mut [f32]);
/// Row-wise softmax kernel signature.
pub(crate) type SoftmaxKernel = fn(&[f32], &mut [f32], usize, usize);
/// Row-wise layer-norm kernel signature.
pub(crate) type LayerNormKernel = fn(&[f32], &mut [f32], usize, usize, &[f32], &[f32], f32);

/// A bundle of function pointers, one per kernel kind, all resolved to the
/// same [`IsaPath`]. Populated by [`build_table`] and cached in [`selected`].
#[derive(Clone, Copy)]
pub(crate) struct KernelTable {
    pub(crate) gemm: GemmKernel,
    pub(crate) gemv: GemvKernel,
    pub(crate) add: BinaryKernel,
    pub(crate) mul: BinaryKernel,
    pub(crate) relu: UnaryKernel,
    pub(crate) sigmoid: UnaryKernel,
    pub(crate) tanh: UnaryKernel,
    pub(crate) gelu: UnaryKernel,
    pub(crate) softmax: SoftmaxKernel,
    pub(crate) layer_norm: LayerNormKernel,
}

fn scalar_table() -> KernelTable {
    KernelTable {
        gemm: scalar::gemm,
        gemv: scalar::gemv,
        add: scalar::add,
        mul: scalar::mul,
        relu: scalar::relu,
        sigmoid: scalar::sigmoid,
        tanh: scalar::tanh,
        gelu: scalar::gelu,
        softmax: scalar::softmax,
        layer_norm: scalar::layer_norm,
    }
}

#[cfg(target_arch = "x86_64")]
fn avx2_table() -> KernelTable {
    use crate::kernels::avx2;
    KernelTable {
        gemm: avx2::gemm,
        gemv: avx2::gemv,
        add: avx2::add,
        mul: avx2::mul,
        relu: avx2::relu,
        sigmoid: avx2::sigmoid,
        tanh: avx2::tanh,
        gelu: avx2::gelu,
        softmax: avx2::softmax,
        layer_norm: avx2::layer_norm,
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_table() -> KernelTable {
    // Unreachable: `features::select_isa` never yields `Avx2` off x86-64, and
    // `table_for` rejects it via `CpuFeatures::supports`.
    unreachable!("AVX2 kernel table requested on a non-x86-64 target")
}

#[cfg(target_arch = "aarch64")]
fn neon_table() -> KernelTable {
    use crate::kernels::neon;
    KernelTable {
        gemm: neon::gemm,
        gemv: neon::gemv,
        add: neon::add,
        mul: neon::mul,
        relu: neon::relu,
        sigmoid: neon::sigmoid,
        tanh: neon::tanh,
        gelu: neon::gelu,
        softmax: neon::softmax,
        layer_norm: neon::layer_norm,
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn neon_table() -> KernelTable {
    // Unreachable: `features::select_isa` never yields `Neon` off aarch64, and
    // `table_for` rejects it via `CpuFeatures::supports`.
    unreachable!("NEON kernel table requested on a non-aarch64 target")
}

/// Maps an [`IsaPath`] to its kernel table — the single source of truth for
/// the ISA → implementation mapping (used by both production dispatch and the
/// `*_on` test entry points).
fn build_table(isa: IsaPath) -> KernelTable {
    match isa {
        IsaPath::Scalar => scalar_table(),
        IsaPath::Avx2 => avx2_table(),
        IsaPath::Neon => neon_table(),
    }
}

struct Selected {
    isa: IsaPath,
    table: KernelTable,
}

static SELECTED: OnceLock<Selected> = OnceLock::new();

fn selected() -> &'static Selected {
    SELECTED.get_or_init(|| {
        // A malformed / unsupported `VOKRA_CPU_ISA` is a hard configuration
        // error: fail fast and loudly (an explicit panic, not a silent
        // fallback to another path — FR-EX-08 principle). The pure
        // `resolve_isa` path is unit-tested in `features`; production only
        // reaches this panic on genuine misconfiguration.
        let isa = features::resolve_isa()
            .unwrap_or_else(|e| panic!("invalid {} override: {e}", features::ENV_ISA_OVERRIDE));
        Selected {
            isa,
            table: build_table(isa),
        }
    })
}

/// The ISA path this process selected (host default, or the `VOKRA_CPU_ISA`
/// override). Fixed on first use and stable thereafter.
///
/// Intended for diagnostics: the ASR demo's one-line ISA log (M0-06-T26) and
/// the CI assertion that a binary picks the runner's path (M0-08-T18).
pub fn active_isa() -> IsaPath {
    selected().isa
}

/// The cached production kernel table for [`active_isa`].
pub(crate) fn table() -> &'static KernelTable {
    &selected().table
}

/// Builds a kernel table for an explicitly requested `isa`, used by the
/// `*_on` entry points (differential tests, forced-path comparison).
///
/// Requesting a path the host cannot run is an explicit
/// [`VokraError::BackendUnavailable`] (FR-EX-08 principle), never a silent
/// switch to another path.
pub(crate) fn table_for(isa: IsaPath) -> Result<KernelTable> {
    if CpuFeatures::detect().supports(isa) {
        Ok(build_table(isa))
    } else {
        Err(VokraError::BackendUnavailable(format!(
            "the {isa} kernel path is not available on this host CPU"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_isa_is_stable_and_host_supported() {
        let a = active_isa();
        let b = active_isa();
        assert_eq!(a, b, "selection must be fixed after the first call");
        assert!(CpuFeatures::detect().supports(a));

        // On this test process (no override) the active path equals the host
        // best path.
        assert_eq!(a, CpuFeatures::detect().best_isa());
    }

    #[test]
    fn table_for_scalar_always_available() {
        assert!(table_for(IsaPath::Scalar).is_ok());
    }

    #[test]
    fn table_for_rejects_unavailable_path() {
        // Exactly one of AVX2 / NEON is unavailable on any given host arch;
        // whichever it is must be an explicit error.
        let feats = CpuFeatures::detect();
        for isa in [IsaPath::Avx2, IsaPath::Neon] {
            if !feats.supports(isa) {
                assert!(matches!(
                    table_for(isa),
                    Err(VokraError::BackendUnavailable(_))
                ));
            }
        }
    }
}
