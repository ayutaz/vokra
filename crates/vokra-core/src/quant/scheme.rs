//! WxAy quantization scheme tag (T02).
//!
//! [`QuantScheme`] names the weight / activation dtype pair a policy rule
//! asks for. It commits to *no* kernel — resolution (T04) and validation
//! (T09) work purely off these tags. Kernels for the INT8 arm land in a
//! follow-up WP; M2-08 stops at policy validation errors when a scheme
//! resolves to an unsupported activation dtype (FR-EX-08 pattern).

use crate::backend::BackendKind;
use crate::error::{Result, VokraError};
use crate::gguf::GgmlType;

/// Weight/activation dtype pair carried by a quantization policy rule.
///
/// The alias strings round-trip through [`Self::as_str`] / [`Self::from_alias_str`]
/// and are what the `vokra.quant.*` GGUF chunk (T05) stores on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QuantScheme {
    /// Weight = F32, activation = F32.
    Fp32,
    /// Weight = F16, activation = F16 (metadata-only in M2-08 — see module
    /// doc; kernels stay F32 until the fp16 activation path lands).
    Fp16,
    /// Weight = Q4_K, activation = F16. The default 4-bit tier ("w4a16"
    /// with no suffix resolves here).
    W4A16Q4K,
    /// Weight = Q5_K, activation = F16.
    W4A16Q5K,
    /// Weight = Q6_K, activation = F16.
    W4A16Q6K,
    /// Weight = INT8, activation = INT8. No kernel in M2-08 — resolves to
    /// [`ActivationDtype::Int8`], which the runtime rejects with
    /// `VokraError::InvalidArgument` at session ctor (FR-EX-08).
    W8A8Int8,
}

/// Weight dtype implied by a [`QuantScheme`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightDtype {
    /// Dense F32 / F16 / K-quant weight (dequantized via
    /// `crate::gguf::quant::dequantize`).
    Ggml(GgmlType),
    /// INT8 weight — no on-disk decoder yet (M2-08 policy-only).
    Int8Reserved,
}

/// Activation dtype implied by a [`QuantScheme`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationDtype {
    /// FP32 activation path (M2-08 kernels).
    F32,
    /// FP16 activation path (metadata-only in M2-08).
    F16,
    /// INT8 activation — kernel deferred; validation must reject before
    /// reaching a backend.
    Int8,
}

impl QuantScheme {
    /// Weight dtype this scheme requests.
    pub fn weight_dtype(&self) -> WeightDtype {
        match self {
            Self::Fp32 => WeightDtype::Ggml(GgmlType::F32),
            Self::Fp16 => WeightDtype::Ggml(GgmlType::F16),
            Self::W4A16Q4K => WeightDtype::Ggml(GgmlType::Q4K),
            Self::W4A16Q5K => WeightDtype::Ggml(GgmlType::Q5K),
            Self::W4A16Q6K => WeightDtype::Ggml(GgmlType::Q6K),
            Self::W8A8Int8 => WeightDtype::Int8Reserved,
        }
    }

    /// Activation dtype this scheme requests.
    pub fn activation_dtype(&self) -> ActivationDtype {
        match self {
            Self::Fp32 => ActivationDtype::F32,
            Self::Fp16 | Self::W4A16Q4K | Self::W4A16Q5K | Self::W4A16Q6K => ActivationDtype::F16,
            Self::W8A8Int8 => ActivationDtype::Int8,
        }
    }

    /// Canonical alias string (round-trips through [`Self::from_alias_str`]).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
            Self::W4A16Q4K => "w4a16-q4k",
            Self::W4A16Q5K => "w4a16-q5k",
            Self::W4A16Q6K => "w4a16-q6k",
            Self::W8A8Int8 => "w8a8",
        }
    }

    /// Whether the given backend has a kernel path for this scheme in M2-08.
    ///
    /// Kernel coverage in M2-08 is F32-only across all backends
    /// (`vokra-backend-cpu/src/kernels/mod.rs:6-8` — "dtype is f32 only in the
    /// spike"; Metal / CUDA reuse F32 GEMM on the imperative hot path via the
    /// `vokra-models` `Compute` dispatcher). [`Self::Fp16`] and the W4A16
    /// sub-tiers dequantize to F32 at load and therefore execute on the F32
    /// kernels — reported as *supported* here. [`Self::W8A8Int8`] has **no**
    /// INT8 GEMM kernel on any backend (AVX-VNNI / SDOT / i8mm paths are
    /// deferred per `vokra-backend-cpu/src/lib.rs:16-17`) and is reported as
    /// *not* supported on every [`BackendKind`] — the T09 validate pass in a
    /// follow-up ticket uses this to raise
    /// [`VokraError::UnsupportedOp`](crate::VokraError::UnsupportedOp) rather
    /// than silently widen (FR-EX-08).
    pub fn backend_supported(&self, backend: BackendKind) -> bool {
        match self {
            Self::Fp32 | Self::Fp16 | Self::W4A16Q4K | Self::W4A16Q5K | Self::W4A16Q6K => {
                // Vulkan is explicitly excluded in the M3-02 foundation slice —
                // no SPIR-V compute kernel is wired yet (`kernels/precompiled/`
                // ships no `.spv`; kernels land in M3-02-T14 onwards). Reporting
                // `false` here means the T09 validate pass will surface an
                // explicit `UnsupportedOp` when a caller pairs Vulkan with a
                // scheme, rather than a silent CPU fall back (FR-EX-08).
                matches!(
                    backend,
                    BackendKind::Cpu | BackendKind::Metal | BackendKind::Cuda
                )
            }
            // W8A8 INT8 has no kernel on any backend in M2-08 / M3-02.
            Self::W8A8Int8 => false,
        }
    }

    /// Parse a canonical or shorthand alias. `"w4a16"` (no sub-tier) resolves
    /// to [`Self::W4A16Q4K`] (default 4-bit tier).
    pub fn from_alias_str(s: &str) -> Result<Self> {
        match s {
            "fp32" => Ok(Self::Fp32),
            "fp16" => Ok(Self::Fp16),
            "w4a16" | "w4a16-q4k" => Ok(Self::W4A16Q4K),
            "w4a16-q5k" => Ok(Self::W4A16Q5K),
            "w4a16-q6k" => Ok(Self::W4A16Q6K),
            "w8a8" => Ok(Self::W8A8Int8),
            other => Err(VokraError::InvalidArgument(format!(
                "unknown QuantScheme alias: `{other}` (expected one of fp32, fp16, w4a16, \
                 w4a16-q4k, w4a16-q5k, w4a16-q6k, w8a8)"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alias_round_trip() {
        for scheme in [
            QuantScheme::Fp32,
            QuantScheme::Fp16,
            QuantScheme::W4A16Q4K,
            QuantScheme::W4A16Q5K,
            QuantScheme::W4A16Q6K,
            QuantScheme::W8A8Int8,
        ] {
            let s = scheme.as_str();
            assert_eq!(QuantScheme::from_alias_str(s).unwrap(), scheme);
        }
    }

    #[test]
    fn w4a16_shorthand_defaults_to_q4k() {
        assert_eq!(
            QuantScheme::from_alias_str("w4a16").unwrap(),
            QuantScheme::W4A16Q4K
        );
    }

    #[test]
    fn unknown_alias_errors() {
        assert!(matches!(
            QuantScheme::from_alias_str("nope"),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn dtype_mapping() {
        assert!(matches!(
            QuantScheme::W4A16Q4K.weight_dtype(),
            WeightDtype::Ggml(GgmlType::Q4K)
        ));
        assert!(matches!(
            QuantScheme::W4A16Q5K.weight_dtype(),
            WeightDtype::Ggml(GgmlType::Q5K)
        ));
        assert!(matches!(
            QuantScheme::W4A16Q6K.weight_dtype(),
            WeightDtype::Ggml(GgmlType::Q6K)
        ));
        assert!(matches!(
            QuantScheme::Fp32.weight_dtype(),
            WeightDtype::Ggml(GgmlType::F32)
        ));
        assert!(matches!(
            QuantScheme::Fp16.weight_dtype(),
            WeightDtype::Ggml(GgmlType::F16)
        ));
        assert!(matches!(
            QuantScheme::W8A8Int8.weight_dtype(),
            WeightDtype::Int8Reserved
        ));
        assert_eq!(QuantScheme::Fp32.activation_dtype(), ActivationDtype::F32);
        assert_eq!(QuantScheme::Fp16.activation_dtype(), ActivationDtype::F16);
        assert_eq!(
            QuantScheme::W4A16Q4K.activation_dtype(),
            ActivationDtype::F16
        );
        assert_eq!(
            QuantScheme::W8A8Int8.activation_dtype(),
            ActivationDtype::Int8
        );
    }

    #[test]
    fn backend_supported_matches_kernel_coverage() {
        // W8A8 INT8 has no kernel on any backend in M2-08 / M3-02.
        for backend in [
            BackendKind::Cpu,
            BackendKind::Metal,
            BackendKind::Cuda,
            BackendKind::Vulkan,
        ] {
            assert!(!QuantScheme::W8A8Int8.backend_supported(backend));
        }
        // Everything else lands on the F32 kernel path (weights dequantize to
        // F32 at load) — supported on every backend Vokra ships in M2-08.
        for scheme in [
            QuantScheme::Fp32,
            QuantScheme::Fp16,
            QuantScheme::W4A16Q4K,
            QuantScheme::W4A16Q5K,
            QuantScheme::W4A16Q6K,
        ] {
            for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
                assert!(
                    scheme.backend_supported(backend),
                    "scheme {:?} expected supported on {:?}",
                    scheme,
                    backend
                );
            }
            // Vulkan foundation slice (M3-02): no SPIR-V kernel wired, so no
            // scheme is supported yet — this becomes an explicit `UnsupportedOp`
            // upstream (no silent CPU fall back, FR-EX-08).
            assert!(
                !scheme.backend_supported(BackendKind::Vulkan),
                "scheme {:?} unexpectedly reported supported on Vulkan (foundation slice \
                 has no SPIR-V kernel)",
                scheme,
            );
        }
    }
}
