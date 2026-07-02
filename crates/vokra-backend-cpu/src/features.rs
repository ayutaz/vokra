//! Runtime ISA detection and the `VOKRA_CPU_ISA` override (M0-08-T03).
//!
//! Detection uses only the standard library's CPUID-based
//! `std::arch::is_x86_feature_detected!` on x86-64 and the compile-time
//! guarantee that NEON is an ARMv8-A baseline on AArch64 (CLAUDE.md). No
//! extra dependency is introduced (NFR-DS-02), and **no JIT / runtime code
//! generation** is involved (NFR-RL-05): selection only picks which
//! statically compiled kernel to call.
//!
//! The override environment variable `VOKRA_CPU_ISA=scalar|avx2|neon` lets
//! tests and CI force a specific path on one machine (M0-08-T18 forced-path
//! job). Requesting a path the host cannot run is an **explicit error**
//! (never a silent switch to another path), consistent with the explicit-op
//! error principle of FR-EX-08.

use std::fmt;

use vokra_core::{Result, VokraError};

/// Name of the ISA-path override environment variable (M0-08-T03).
pub const ENV_ISA_OVERRIDE: &str = "VOKRA_CPU_ISA";

/// Which SIMD kernel path the CPU backend runs.
///
/// This selects between behaviourally identical implementations (the same
/// results within FP32 rounding); it is *not* the cross-backend op-coverage
/// concept of FR-EX-08. See [`crate::dispatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IsaPath {
    /// Portable scalar kernels (fallback on x86-64 without AVX2; oracle for
    /// the SIMD differential tests).
    Scalar,
    /// x86-64 AVX2 + FMA kernels (FR-BE-01 main path).
    Avx2,
    /// AArch64 NEON kernels (ARMv8-A baseline).
    Neon,
}

impl fmt::Display for IsaPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Neon => "neon",
        };
        f.write_str(s)
    }
}

/// CPU features relevant to the spike kernel set (f32 AVX2 / NEON).
///
/// F16C / BMI1/2 (part of the wider "main path" definition in CLAUDE.md) are
/// unused by the f32 spike kernels and therefore not probed here; they return
/// with the later ISA-tier expansion (FR-BE-01 "→ 拡張").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuFeatures {
    /// x86-64 AVX2 (256-bit integer/float vectors).
    pub avx2: bool,
    /// x86-64 FMA3 (fused multiply-add).
    pub fma: bool,
    /// AArch64 NEON (ARMv8-A baseline).
    pub neon: bool,
}

impl CpuFeatures {
    /// Detects the running host's features.
    ///
    /// On x86-64 this consults `is_x86_feature_detected!` (CPUID). On AArch64
    /// NEON is a baseline feature and reported unconditionally. On any other
    /// architecture all features are `false` and only [`IsaPath::Scalar`] is
    /// available.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2: std::arch::is_x86_feature_detected!("avx2"),
                fma: std::arch::is_x86_feature_detected!("fma"),
                neon: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                avx2: false,
                fma: false,
                neon: true,
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                avx2: false,
                fma: false,
                neon: false,
            }
        }
    }

    /// Whether `isa` can actually run on this host.
    ///
    /// [`IsaPath::Scalar`] is always available; `Avx2` needs AVX2+FMA; `Neon`
    /// needs NEON.
    pub fn supports(&self, isa: IsaPath) -> bool {
        match isa {
            IsaPath::Scalar => true,
            IsaPath::Avx2 => self.avx2 && self.fma,
            IsaPath::Neon => self.neon,
        }
    }

    /// The fastest path this host supports: AVX2 if present, else NEON, else
    /// scalar (M0-08-T03 selection rule).
    pub fn best_isa(&self) -> IsaPath {
        if self.avx2 && self.fma {
            IsaPath::Avx2
        } else if self.neon {
            IsaPath::Neon
        } else {
            IsaPath::Scalar
        }
    }
}

/// Parses a `VOKRA_CPU_ISA` value into an [`IsaPath`] (case-insensitive).
///
/// This is a pure function (no environment / host access) so it can be unit
/// tested exhaustively and deterministically on any machine. An unrecognised
/// value is an explicit [`VokraError::InvalidArgument`].
pub fn parse_isa_override(value: &str) -> Result<IsaPath> {
    match value.trim().to_ascii_lowercase().as_str() {
        "scalar" => Ok(IsaPath::Scalar),
        "avx2" => Ok(IsaPath::Avx2),
        "neon" => Ok(IsaPath::Neon),
        other => Err(VokraError::InvalidArgument(format!(
            "{ENV_ISA_OVERRIDE} must be one of scalar|avx2|neon, got `{other}`"
        ))),
    }
}

/// Applies the selection rule: an explicit `override_isa` (if the host
/// supports it) else `features.best_isa()`.
///
/// Pure function (M0-08-T03). A requested override the host cannot run is an
/// explicit [`VokraError::BackendUnavailable`] — never a silent fallback
/// (FR-EX-08 principle).
pub fn select_isa(override_isa: Option<IsaPath>, features: &CpuFeatures) -> Result<IsaPath> {
    match override_isa {
        Some(isa) if features.supports(isa) => Ok(isa),
        Some(isa) => Err(VokraError::BackendUnavailable(format!(
            "{ENV_ISA_OVERRIDE}={isa} was requested but this host CPU does not support the {isa} path"
        ))),
        None => Ok(features.best_isa()),
    }
}

/// Resolves the effective [`IsaPath`] from the real environment and host.
///
/// Reads `VOKRA_CPU_ISA` (if set), parses it and applies [`select_isa`]
/// against [`CpuFeatures::detect`]. Only reads the environment (never writes
/// it), so it is safe under the edition-2024 `set_var` restrictions.
pub fn resolve_isa() -> Result<IsaPath> {
    let features = CpuFeatures::detect();
    let override_isa = match std::env::var(ENV_ISA_OVERRIDE) {
        Ok(value) => Some(parse_isa_override(&value)?),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(VokraError::InvalidArgument(format!(
                "{ENV_ISA_OVERRIDE} contains non-UTF-8 data"
            )));
        }
    };
    select_isa(override_isa, &features)
}

#[cfg(test)]
mod tests {
    use super::*;

    const X86: CpuFeatures = CpuFeatures {
        avx2: true,
        fma: true,
        neon: false,
    };
    const X86_NO_AVX2: CpuFeatures = CpuFeatures {
        avx2: false,
        fma: false,
        neon: false,
    };
    const ARM: CpuFeatures = CpuFeatures {
        avx2: false,
        fma: false,
        neon: true,
    };
    // AVX2 present but FMA absent: the AVX2 kernels use `_mm256_fmadd_ps`, so
    // this combination must NOT select the Avx2 path (it would SIGILL).
    const AVX2_NO_FMA: CpuFeatures = CpuFeatures {
        avx2: true,
        fma: false,
        neon: false,
    };

    #[test]
    fn detect_matches_compiled_target() {
        let f = CpuFeatures::detect();
        // NEON is always true on aarch64 and always false elsewhere.
        if cfg!(target_arch = "aarch64") {
            assert!(f.neon);
            assert_eq!(f.best_isa(), IsaPath::Neon);
        }
        if cfg!(not(target_arch = "aarch64")) {
            assert!(!f.neon);
        }
        // Scalar is always a supported path.
        assert!(f.supports(IsaPath::Scalar));
    }

    #[test]
    fn best_isa_selection_rule() {
        assert_eq!(X86.best_isa(), IsaPath::Avx2);
        assert_eq!(X86_NO_AVX2.best_isa(), IsaPath::Scalar);
        assert_eq!(ARM.best_isa(), IsaPath::Neon);
    }

    #[test]
    fn avx2_without_fma_never_selects_the_fma_kernels() {
        // The AVX2 gemm / layer_norm kernels emit `_mm256_fmadd_ps`; selecting
        // the Avx2 path on an AVX2-without-FMA host would execute an illegal
        // instruction. The `&& self.fma` guard in `best_isa` / `supports` must
        // keep this feature set on the scalar path.
        assert_eq!(AVX2_NO_FMA.best_isa(), IsaPath::Scalar);
        assert!(!AVX2_NO_FMA.supports(IsaPath::Avx2));
        // Scalar is still always available.
        assert!(AVX2_NO_FMA.supports(IsaPath::Scalar));
        // Explicitly forcing Avx2 is an explicit error, never a silent switch.
        let err = select_isa(Some(IsaPath::Avx2), &AVX2_NO_FMA).unwrap_err();
        assert!(matches!(err, VokraError::BackendUnavailable(_)));
    }

    #[test]
    fn parse_override_accepts_known_values_case_insensitively() {
        assert_eq!(parse_isa_override("scalar").unwrap(), IsaPath::Scalar);
        assert_eq!(parse_isa_override("AVX2").unwrap(), IsaPath::Avx2);
        assert_eq!(parse_isa_override("  Neon ").unwrap(), IsaPath::Neon);
    }

    #[test]
    fn parse_override_rejects_garbage_with_explicit_error() {
        let err = parse_isa_override("sse2").unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn select_forces_supported_override() {
        assert_eq!(
            select_isa(Some(IsaPath::Scalar), &X86).unwrap(),
            IsaPath::Scalar
        );
        assert_eq!(
            select_isa(Some(IsaPath::Avx2), &X86).unwrap(),
            IsaPath::Avx2
        );
    }

    #[test]
    fn select_rejects_unsupported_override_with_explicit_error() {
        // AVX2 forced on an ARM-like feature set is unavailable.
        let err = select_isa(Some(IsaPath::Avx2), &ARM).unwrap_err();
        assert!(matches!(err, VokraError::BackendUnavailable(_)));
        // NEON forced on an x86-like feature set is unavailable.
        let err = select_isa(Some(IsaPath::Neon), &X86).unwrap_err();
        assert!(matches!(err, VokraError::BackendUnavailable(_)));
    }

    #[test]
    fn select_without_override_uses_best() {
        assert_eq!(select_isa(None, &X86).unwrap(), IsaPath::Avx2);
        assert_eq!(select_isa(None, &ARM).unwrap(), IsaPath::Neon);
    }

    #[test]
    fn resolve_isa_on_this_host_is_supported() {
        // Whatever the CI runner is, the resolved path must be one the host
        // actually supports (unless the env forces an error, which this test
        // process does not set).
        let isa = resolve_isa().expect("no override set in this test process");
        assert!(CpuFeatures::detect().supports(isa));
    }
}
