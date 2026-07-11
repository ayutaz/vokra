//! Runtime ISA detection and the `VOKRA_CPU_ISA` override (M0-08-T03; extended
//! for RVV 1.0 by M3-13-T02/T03).
//!
//! Detection uses only the standard library's CPUID-based
//! `std::arch::is_x86_feature_detected!` on x86-64, the compile-time guarantee
//! that NEON is an ARMv8-A baseline on AArch64, and `/proc/cpuinfo` parsing on
//! `riscv64-*-linux-*` for the RVV 1.0 `v` extension + Zvfh/ZvfBFmin/Zvbb
//! optional extensions (CLAUDE.md, ADR M3-13). No extra dependency is
//! introduced (NFR-DS-02), and **no JIT / runtime code generation** is
//! involved (NFR-RL-05): selection only picks which statically compiled
//! kernel to call.
//!
//! The override environment variable `VOKRA_CPU_ISA=scalar|avx2|neon|rvv`
//! lets tests and CI force a specific path on one machine (M0-08-T18
//! forced-path job; M3-13-T02 rvv variant). Requesting a path the host
//! cannot run is an **explicit error** (never a silent switch to another
//! path), consistent with the explicit-op error principle of FR-EX-08.

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
    /// RISC-V RVV 1.0 base kernels (SpacemiT K1 / Banana Pi BPI-F3 世代,
    /// M3-13). Optional Zvfh / ZvfBFmin / Zvbb extensions are reported by
    /// [`CpuFeatures`] separately; selecting `Rvv` requires at least
    /// `rvv_v = true`. RVV 0.7.1 fallback (LicheePi 4A C910 / Milk-V Duo
    /// C906) is deferred to v1.5 = M4-08 (前倒し禁止, CLAUDE.md).
    Rvv,
}

impl fmt::Display for IsaPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Neon => "neon",
            Self::Rvv => "rvv",
        };
        f.write_str(s)
    }
}

/// CPU features relevant to the spike kernel set (f32 AVX2 / NEON) and the
/// M3-13 RVV 1.0 scaffold (SpacemiT K1 / Banana Pi BPI-F3).
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
    /// RISC-V RVV 1.0 base vector extension (`v`), M3-13. When true the host
    /// implements the ratified RVV 1.0 spec at least at rv64gcv baseline
    /// (SpacemiT K1 / Banana Pi BPI-F3).
    pub rvv_v: bool,
    /// RISC-V Zvfh (FP16 vector arithmetic), an RVV 1.0 optional extension.
    /// Gates the [`crate::kernels::rvv`] fp16 GEMM opt-in path (M3-13-T09).
    pub rvv_zvfh: bool,
    /// RISC-V ZvfBFmin (BF16 minimum subset), an RVV 1.0 optional extension.
    /// Probed but no BF16 kernel is wired in M3 — reserved for M4+.
    pub rvv_zvfbfmin: bool,
    /// RISC-V Zvbb (vector bit-manipulation), an RVV 1.0 optional extension.
    /// Probed but no kernel wired in M3 — reserved for M4+.
    pub rvv_zvbb: bool,
}

impl CpuFeatures {
    /// Detects the running host's features.
    ///
    /// On x86-64 this consults `is_x86_feature_detected!` (CPUID). On AArch64
    /// NEON is a baseline feature and reported unconditionally. On riscv64
    /// (Linux) it parses `/proc/cpuinfo` for the `isa` line and looks for the
    /// `v`, `zvfh`, `zvfbfmin`, `zvbb` extension names (M3-13-T02). On any
    /// other architecture all features are `false` and only
    /// [`IsaPath::Scalar`] is available.
    ///
    /// Rationale for `/proc/cpuinfo` over `getauxval(AT_HWCAP)`: `getauxval`
    /// is a `libc` symbol and pulling it in would break the crate's
    /// no-external-crate rule (NFR-DS-02); a Linux syscall wrapper would
    /// require unsafe FFI. `std::fs::read_to_string("/proc/cpuinfo")` is
    /// pure-std, zero-alloc-critical (we read once at process init), and
    /// covers the two shipping RVV 1.0 boards (K1 / BPI-F3, ADR M3-13).
    /// stable Rust has no `is_riscv_feature_detected!` at 1.85 (unstable
    /// `riscv_ext_intrinsics`), so this path parses the ISA string
    /// ourselves.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2: std::arch::is_x86_feature_detected!("avx2"),
                fma: std::arch::is_x86_feature_detected!("fma"),
                neon: false,
                rvv_v: false,
                rvv_zvfh: false,
                rvv_zvfbfmin: false,
                rvv_zvbb: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                avx2: false,
                fma: false,
                neon: true,
                rvv_v: false,
                rvv_zvfh: false,
                rvv_zvfbfmin: false,
                rvv_zvbb: false,
            }
        }
        #[cfg(target_arch = "riscv64")]
        {
            let caps = detect_riscv_caps();
            Self {
                avx2: false,
                fma: false,
                neon: false,
                rvv_v: caps.v,
                rvv_zvfh: caps.zvfh,
                rvv_zvfbfmin: caps.zvfbfmin,
                rvv_zvbb: caps.zvbb,
            }
        }
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        )))]
        {
            Self {
                avx2: false,
                fma: false,
                neon: false,
                rvv_v: false,
                rvv_zvfh: false,
                rvv_zvfbfmin: false,
                rvv_zvbb: false,
            }
        }
    }

    /// Whether `isa` can actually run on this host.
    ///
    /// [`IsaPath::Scalar`] is always available; `Avx2` needs AVX2+FMA; `Neon`
    /// needs NEON; `Rvv` needs the RVV 1.0 base `v` extension.
    pub fn supports(&self, isa: IsaPath) -> bool {
        match isa {
            IsaPath::Scalar => true,
            IsaPath::Avx2 => self.avx2 && self.fma,
            IsaPath::Neon => self.neon,
            IsaPath::Rvv => self.rvv_v,
        }
    }

    /// The fastest path this host supports: AVX2 if present, else NEON, else
    /// RVV, else scalar (M0-08-T03 + M3-13-T03 selection rule). Only one of
    /// AVX2 / NEON / RVV can be true on any given host — they are
    /// arch-exclusive.
    pub fn best_isa(&self) -> IsaPath {
        if self.avx2 && self.fma {
            IsaPath::Avx2
        } else if self.neon {
            IsaPath::Neon
        } else if self.rvv_v {
            IsaPath::Rvv
        } else {
            IsaPath::Scalar
        }
    }
}

/// Internal RISC-V capability bundle (M3-13-T02). Only produced on
/// `target_arch = "riscv64"` — the `detect_riscv_caps()` fn is
/// `cfg(target_arch = "riscv64")`-gated so it never appears on other targets.
#[cfg(target_arch = "riscv64")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct RiscvIsaCaps {
    v: bool,
    zvfh: bool,
    zvfbfmin: bool,
    zvbb: bool,
}

/// Probes `/proc/cpuinfo` for RVV 1.0 + optional extensions on Linux riscv64.
///
/// Returns all-false on non-Linux riscv64 (BSD / bare-metal) since we have no
/// portable probe surface there; the runtime dispatch then falls back to the
/// scalar path — this is the same within-CPU-backend ISA fallback used when
/// AVX2 / NEON detection returns false, not the cross-backend silent-fallback
/// forbidden by FR-EX-08.
///
/// # Format
///
/// Linux `/proc/cpuinfo` on RISC-V exposes one CPU per stanza with an `isa :`
/// line like `rv64imafdcv_zvfh_zvfbfmin_zvbb_zicsr_zifencei`. We scan for the
/// token `v` (RVV 1.0 base) plus the three optional extension names we care
/// about (Zvfh / ZvfBFmin / Zvbb). Token match is case-insensitive and
/// underscore-separated so a heterogeneous descriptor cannot be spoofed by a
/// prefix like `vhole` or `virt`.
#[cfg(target_arch = "riscv64")]
fn detect_riscv_caps() -> RiscvIsaCaps {
    // Only Linux exposes `/proc/cpuinfo`; other riscv64 OSes return the
    // all-false default. `std::fs::read_to_string` is fallible (e.g. the
    // /proc filesystem may be missing under chroots or `unshare(--mount)`
    // sandboxes) — a failed read is not an error, just "no evidence of RVV",
    // and the caller falls back to scalar.
    #[cfg(target_os = "linux")]
    {
        match std::fs::read_to_string("/proc/cpuinfo") {
            Ok(text) => parse_riscv_isa_string(&text),
            Err(_) => RiscvIsaCaps::default(),
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        RiscvIsaCaps::default()
    }
}

/// Parses the `/proc/cpuinfo` payload for RISC-V ISA extensions.
///
/// Pure function so it can be unit-tested off riscv64 (we still gate it by
/// `cfg(target_arch = "riscv64")` because it is only used from
/// `detect_riscv_caps` on that target).
#[cfg(target_arch = "riscv64")]
fn parse_riscv_isa_string(cpuinfo: &str) -> RiscvIsaCaps {
    let mut caps = RiscvIsaCaps::default();
    for line in cpuinfo.lines() {
        // Case-insensitive prefix match on `isa` before the `:` — the kernel
        // may render this as `isa` or `ISA` depending on version.
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        if !k.trim().eq_ignore_ascii_case("isa") {
            continue;
        }
        // Tokenise on `_` and any ASCII whitespace so `rv64imafdcv_zvfh` and
        // `rv64gcv zvfh zvbb` both yield the same set. Also strip the
        // leading `rvXX` base descriptor (e.g. `rv64imafdcv`) by scanning its
        // last few chars.
        for token in v.split(|c: char| c == '_' || c.is_ascii_whitespace()) {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            let low = token.to_ascii_lowercase();
            // The base descriptor `rv64...` may contain the `v` extension
            // fused at the tail (e.g. `rv64imafdcv`). Peel it off by
            // checking membership of common single-letter extensions after
            // `rv64` / `rv32`.
            if low.starts_with("rv64") || low.starts_with("rv32") {
                let tail = &low[4..];
                if tail.contains('v') {
                    caps.v = true;
                }
                continue;
            }
            match low.as_str() {
                "v" => caps.v = true,
                "zvfh" => caps.zvfh = true,
                "zvfbfmin" => caps.zvfbfmin = true,
                "zvbb" => caps.zvbb = true,
                _ => {}
            }
        }
    }
    caps
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
        "rvv" => Ok(IsaPath::Rvv),
        other => Err(VokraError::InvalidArgument(format!(
            "{ENV_ISA_OVERRIDE} must be one of scalar|avx2|neon|rvv, got `{other}`"
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
        rvv_v: false,
        rvv_zvfh: false,
        rvv_zvfbfmin: false,
        rvv_zvbb: false,
    };
    const X86_NO_AVX2: CpuFeatures = CpuFeatures {
        avx2: false,
        fma: false,
        neon: false,
        rvv_v: false,
        rvv_zvfh: false,
        rvv_zvfbfmin: false,
        rvv_zvbb: false,
    };
    const ARM: CpuFeatures = CpuFeatures {
        avx2: false,
        fma: false,
        neon: true,
        rvv_v: false,
        rvv_zvfh: false,
        rvv_zvfbfmin: false,
        rvv_zvbb: false,
    };
    // AVX2 present but FMA absent: the AVX2 kernels use `_mm256_fmadd_ps`, so
    // this combination must NOT select the Avx2 path (it would SIGILL).
    const AVX2_NO_FMA: CpuFeatures = CpuFeatures {
        avx2: true,
        fma: false,
        neon: false,
        rvv_v: false,
        rvv_zvfh: false,
        rvv_zvfbfmin: false,
        rvv_zvbb: false,
    };
    // Synthetic feature set for M3-13-T02 unit tests: RVV 1.0 base present
    // (SpacemiT K1 / BPI-F3 baseline); optional Zvfh added in a second variant.
    const RVV_BASE: CpuFeatures = CpuFeatures {
        avx2: false,
        fma: false,
        neon: false,
        rvv_v: true,
        rvv_zvfh: false,
        rvv_zvfbfmin: false,
        rvv_zvbb: false,
    };
    const RVV_WITH_ZVFH: CpuFeatures = CpuFeatures {
        avx2: false,
        fma: false,
        neon: false,
        rvv_v: true,
        rvv_zvfh: true,
        rvv_zvfbfmin: false,
        rvv_zvbb: false,
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

    // -------------------------------------------------------------------
    // M3-13-T02 RVV 1.0 detection + selection unit tests
    //
    // These are pure-function tests over synthetic feature sets so they
    // execute on every host in CI (the actual /proc/cpuinfo parser is
    // riscv64-only and gated separately below).
    // -------------------------------------------------------------------

    #[test]
    fn rvv_selection_prefers_avx2_neon_over_rvv() {
        // AVX2 / NEON / RVV are arch-exclusive in practice, but the selector
        // must still express the priority order correctly for hosts where
        // multiple bits happen to be true (e.g. a future emulator surface).
        assert_eq!(RVV_BASE.best_isa(), IsaPath::Rvv);
        assert_eq!(RVV_WITH_ZVFH.best_isa(), IsaPath::Rvv);
        assert!(RVV_BASE.supports(IsaPath::Rvv));
        assert!(!RVV_BASE.supports(IsaPath::Avx2));
        assert!(!RVV_BASE.supports(IsaPath::Neon));
        assert!(RVV_BASE.supports(IsaPath::Scalar));
    }

    #[test]
    fn rvv_override_rejected_on_non_rvv_host_with_explicit_error() {
        // Forcing Rvv on an x86-like feature set must be an explicit error
        // (never a silent switch to another path — FR-EX-08 principle).
        let err = select_isa(Some(IsaPath::Rvv), &X86).unwrap_err();
        assert!(matches!(err, VokraError::BackendUnavailable(_)));
        let err = select_isa(Some(IsaPath::Rvv), &ARM).unwrap_err();
        assert!(matches!(err, VokraError::BackendUnavailable(_)));
    }

    #[test]
    fn rvv_override_accepted_on_rvv_host() {
        assert_eq!(
            select_isa(Some(IsaPath::Rvv), &RVV_BASE).unwrap(),
            IsaPath::Rvv
        );
        assert_eq!(
            select_isa(Some(IsaPath::Rvv), &RVV_WITH_ZVFH).unwrap(),
            IsaPath::Rvv
        );
    }

    #[test]
    fn parse_override_accepts_rvv_case_insensitively() {
        assert_eq!(parse_isa_override("rvv").unwrap(), IsaPath::Rvv);
        assert_eq!(parse_isa_override("RVV").unwrap(), IsaPath::Rvv);
        assert_eq!(parse_isa_override("  Rvv ").unwrap(), IsaPath::Rvv);
    }

    #[test]
    fn isa_path_display_includes_rvv() {
        assert_eq!(IsaPath::Rvv.to_string(), "rvv");
    }

    #[test]
    fn detect_on_non_rvv_host_reports_no_rvv_extensions() {
        // On x86-64 / ARM64 CI runners the RVV bits must all be false — the
        // /proc/cpuinfo parser is riscv64-gated and cannot leak features onto
        // other targets.
        let f = CpuFeatures::detect();
        if cfg!(not(target_arch = "riscv64")) {
            assert!(!f.rvv_v);
            assert!(!f.rvv_zvfh);
            assert!(!f.rvv_zvfbfmin);
            assert!(!f.rvv_zvbb);
        }
    }

    // -------------------------------------------------------------------
    // M3-13-T02 /proc/cpuinfo parser (riscv64-only): pure-function tests
    // for the ISA string parser. Compiled only on riscv64 so it doesn't
    // affect other targets; the SpacemiT K1 sample string comes from the
    // vendor's board bring-up documentation and is used verbatim as the
    // canonical input.
    // -------------------------------------------------------------------

    #[cfg(target_arch = "riscv64")]
    #[test]
    fn parse_riscv_isa_string_detects_v_in_rv64_descriptor() {
        // rv64imafdcv — the base descriptor contains `v`. This matches a
        // SpacemiT K1 / BPI-F3 minimal RVV 1.0 line.
        let cpuinfo = "processor\t: 0\nisa\t: rv64imafdcv\n";
        let caps = super::parse_riscv_isa_string(cpuinfo);
        assert!(caps.v, "rv64imafdcv must set caps.v");
        assert!(!caps.zvfh);
        assert!(!caps.zvfbfmin);
        assert!(!caps.zvbb);
    }

    #[cfg(target_arch = "riscv64")]
    #[test]
    fn parse_riscv_isa_string_detects_optional_extensions() {
        // Composite line with all four extensions.
        let cpuinfo = "isa: rv64imafdcv_zvfh_zvfbfmin_zvbb_zicsr_zifencei\n";
        let caps = super::parse_riscv_isa_string(cpuinfo);
        assert!(caps.v);
        assert!(caps.zvfh);
        assert!(caps.zvfbfmin);
        assert!(caps.zvbb);
    }

    #[cfg(target_arch = "riscv64")]
    #[test]
    fn parse_riscv_isa_string_returns_default_on_pre_rvv_hart() {
        // rv64gc (no `v`) — the C910-class LicheePi 4A / older harts. M4-08
        // will add RVV 0.7.1 fallback for these; in M3 they land on scalar.
        let cpuinfo = "isa: rv64imafdc_zicsr_zifencei\n";
        let caps = super::parse_riscv_isa_string(cpuinfo);
        assert!(!caps.v, "rv64imafdc (no v) must NOT set caps.v");
    }

    #[cfg(target_arch = "riscv64")]
    #[test]
    fn parse_riscv_isa_string_is_case_insensitive_on_key() {
        // Some kernels emit `ISA :` in uppercase.
        let cpuinfo = "ISA : rv64gcv\n";
        let caps = super::parse_riscv_isa_string(cpuinfo);
        assert!(caps.v);
    }

    #[cfg(target_arch = "riscv64")]
    #[test]
    fn parse_riscv_isa_string_ignores_prefixed_tokens() {
        // `virt` and `vhole` must not be misread as the `v` extension.
        let cpuinfo = "isa: rv64imafdc_virt_vhole_zicsr\n";
        let caps = super::parse_riscv_isa_string(cpuinfo);
        assert!(!caps.v);
    }
}
