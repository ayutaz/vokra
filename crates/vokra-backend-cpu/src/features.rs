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
    /// WASM SIMD128 f32x4 kernels (M4-01). **Compile-time, not runtime,
    /// dispatch**: WASM has no runtime CPU feature detection — SIMD
    /// acceptance is decided when the engine *validates* the module, so the
    /// AVX2/NEON CPUID-style probe cannot exist. A wasm32 build either has
    /// `target_feature = "simd128"` baked in (`RUSTFLAGS="-C
    /// target-feature=+simd128"`) and always selects this path, or does not
    /// and always selects [`IsaPath::Scalar`]. Distribution ships BOTH
    /// artifacts (`scripts/build-wasm.sh`) and the JS loader picks one with
    /// a `WebAssembly.validate` feature probe (ADR M4-01-webgpu-wasm §4).
    ///
    /// **Relaxed SIMD is NOT adopted** (Safari-partial only per the CLAUDE.md
    /// quarterly ISA watch; relaxed-fma result nondeterminism conflicts with
    /// the parity discipline NFR-QL-01) — the kernels use deterministic
    /// mul + add, never fma.
    WasmSimd128,
}

impl fmt::Display for IsaPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Neon => "neon",
            Self::Rvv => "rvv",
            Self::WasmSimd128 => "wasm-simd128",
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
    /// WASM SIMD128 (M4-01). **Compile-time constant on wasm32**
    /// (`cfg!(target_feature = "simd128")`), always `false` on native
    /// targets. WASM has no runtime feature detection — see
    /// [`IsaPath::WasmSimd128`] for the 2-artifact distribution policy.
    pub wasm_simd128: bool,

    // ---- M4-17 server-tier features, x86-64 (ADR M4-17 §(a)/(c)) ----
    /// x86-64 AVX-512 Foundation (512-bit zmm; Skylake-X 2017+ server main
    /// path per FR-BE-01). The f32 [`IsaPath::Avx512`] tier additionally
    /// requires DQ/BW/VL (the kernels are compiled with all four enabled;
    /// they ship together on every Skylake-X+/Zen4 part — ADR M4-17 §(b)-4).
    pub avx512f: bool,
    /// x86-64 AVX-512DQ (doubleword/quadword ops; part of the F/DQ/BW/VL
    /// bundle the [`IsaPath::Avx512`] f32 kernels are compiled against).
    pub avx512dq: bool,
    /// x86-64 AVX-512BW (byte/word ops; same bundle as above, and used by the
    /// VNNI INT8 kernel's byte manipulation).
    pub avx512bw: bool,
    /// x86-64 AVX-512VL (128/256-bit encodings of AVX-512 ops; same bundle).
    pub avx512vl: bool,
    /// x86-64 AVX-512 VNNI (`vpdpbusd`; Cascade Lake 2019+ — the server INT8
    /// main path per FR-BE-01). Gates [`IsaPath::Avx512Vnni`].
    pub avx512vnni: bool,
    /// x86-64 AVX-512 BF16 (`vdpbf16ps`; Cooper Lake 2020+). Gates the
    /// opt-in [`IsaPath::Avx512Bf16`] matmul tier.
    pub avx512bf16: bool,
    /// x86-64 AVX-VNNI 256-bit (`vpdpbusd` in VEX encoding; Alder Lake 2021+,
    /// present on both P- and E-cores). **The client INT8 main path**: Intel
    /// client parts since Alder Lake fuse AVX-512 off platform-wide (E-cores
    /// lack it), so this 256-bit tier is what a hybrid CPU actually reports
    /// (ADR M4-17 §(d) tier collapse). Probed via the std_detect feature
    /// string `"avxvnni"`; the field keeps the FR-BE-01 name `avxvnni256`.
    pub avxvnni256: bool,

    // ---- M4-17 server-tier features, ARM64 (ADR M4-17 §(a)/(c)) ----
    /// ARM64 fp16 arithmetic (ARMv8.2, Cortex-A75+ 2018+ per FR-BE-01).
    /// Gates the opt-in [`IsaPath::NeonFp16`] fp16 GEMM tier.
    pub neon_fp16: bool,
    /// ARM64 dotprod SDOT/UDOT (ARMv8.2-DotProd; Cortex-A55/A75 2017 initial
    /// cores, Apple A13+ per FR-BE-01). The ARM INT8 main path — gates
    /// [`IsaPath::NeonDotprod`].
    pub neon_dotprod: bool,
    /// ARM64 i8mm SMMLA/UMMLA (ARMv8.6; Apple M2+ per FR-BE-01 — this dev
    /// machine, an Apple M1, reports `false`). Gates [`IsaPath::NeonI8mm`].
    pub neon_i8mm: bool,
    /// ARM64 bf16 BFMMLA (ARMv8.6). Gates the opt-in [`IsaPath::NeonBf16`]
    /// matmul tier.
    pub neon_bf16: bool,
}

impl CpuFeatures {
    /// The all-features-false value: base for the arch-specific `detect()`
    /// arms (each fills in only the fields its architecture can probe) and
    /// for the synthetic feature sets in unit tests (M4-17-T02/T03; keeps
    /// future field additions from touching every construction site).
    pub const NONE: CpuFeatures = CpuFeatures {
        avx2: false,
        fma: false,
        neon: false,
        rvv_v: false,
        rvv_zvfh: false,
        rvv_zvfbfmin: false,
        rvv_zvbb: false,
        wasm_simd128: false,
        avx512f: false,
        avx512dq: false,
        avx512bw: false,
        avx512vl: false,
        avx512vnni: false,
        avx512bf16: false,
        avxvnni256: false,
        neon_fp16: false,
        neon_dotprod: false,
        neon_i8mm: false,
        neon_bf16: false,
    };

    /// Detects the running host's features.
    ///
    /// On x86-64 this consults `is_x86_feature_detected!` (CPUID), including
    /// the M4-17 server tiers (AVX-512F/DQ/BW/VL + VNNI + BF16, AVX-VNNI
    /// 256-bit). On AArch64 NEON is a baseline feature and reported
    /// unconditionally; the ARMv8.2+ tiers (fp16 / dotprod / i8mm / bf16) are
    /// probed via `is_aarch64_feature_detected!`. On riscv64 (Linux) it
    /// parses `/proc/cpuinfo` for the `isa` line and looks for the `v`,
    /// `zvfh`, `zvfbfmin`, `zvbb` extension names (M3-13-T02). On any other
    /// architecture all features are `false` and only [`IsaPath::Scalar`] is
    /// available.
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
    ///
    /// The same zero-dep rule governs the M4-17 AArch64 probes: the
    /// milestones.md "AT_HWCAP/HWCAP2" wording describes the mechanism
    /// `is_aarch64_feature_detected!` consults internally (auxv on Linux,
    /// sysctl on macOS) — it is **not** an instruction to FFI into
    /// `libc::getauxval` directly, which would violate NFR-DS-02 (ADR M4-17
    /// §(c), following the RISC-V probe judgment above).
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2: std::arch::is_x86_feature_detected!("avx2"),
                fma: std::arch::is_x86_feature_detected!("fma"),
                // M4-17-T02 server tiers. Feature strings confirmed against
                // the rustc 1.95 std_detect surface (ADR M4-17 §(c)); note
                // the AVX-VNNI-256 std_detect name is "avxvnni".
                avx512f: std::arch::is_x86_feature_detected!("avx512f"),
                avx512dq: std::arch::is_x86_feature_detected!("avx512dq"),
                avx512bw: std::arch::is_x86_feature_detected!("avx512bw"),
                avx512vl: std::arch::is_x86_feature_detected!("avx512vl"),
                avx512vnni: std::arch::is_x86_feature_detected!("avx512vnni"),
                avx512bf16: std::arch::is_x86_feature_detected!("avx512bf16"),
                avxvnni256: std::arch::is_x86_feature_detected!("avxvnni"),
                ..Self::NONE
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                neon: true,
                // M4-17-T03 server tiers (std macro only — no getauxval FFI,
                // see the docstring above / ADR M4-17 §(c)).
                neon_fp16: std::arch::is_aarch64_feature_detected!("fp16"),
                neon_dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),
                neon_i8mm: std::arch::is_aarch64_feature_detected!("i8mm"),
                neon_bf16: std::arch::is_aarch64_feature_detected!("bf16"),
                ..Self::NONE
            }
        }
        #[cfg(target_arch = "riscv64")]
        {
            let caps = detect_riscv_caps();
            Self {
                rvv_v: caps.v,
                rvv_zvfh: caps.zvfh,
                rvv_zvfbfmin: caps.zvfbfmin,
                rvv_zvbb: caps.zvbb,
                ..Self::NONE
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            // COMPILE-TIME: WASM has no runtime feature detection (module
            // validation decides SIMD acceptance), so this is `cfg!`, not a
            // probe. The 2-artifact build (`scripts/build-wasm.sh`) makes
            // exactly one of {simd128, base} true per shipped .wasm
            // (M4-01-T04, ADR M4-01 §4).
            Self {
                wasm_simd128: cfg!(target_feature = "simd128"),
                ..Self::NONE
            }
        }
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64",
            target_arch = "wasm32"
        )))]
        {
            Self::NONE
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
            IsaPath::WasmSimd128 => self.wasm_simd128,
        }
    }

    /// The fastest path this host supports: AVX2 if present, else NEON, else
    /// RVV, else WASM SIMD128, else scalar (M0-08-T03 + M3-13-T03 + M4-01-T04
    /// selection rule). Only one of AVX2 / NEON / RVV / WasmSimd128 can be
    /// true on any given host — they are arch-exclusive.
    pub fn best_isa(&self) -> IsaPath {
        if self.avx2 && self.fma {
            IsaPath::Avx2
        } else if self.neon {
            IsaPath::Neon
        } else if self.rvv_v {
            IsaPath::Rvv
        } else if self.wasm_simd128 {
            IsaPath::WasmSimd128
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
        "wasm-simd128" | "wasm_simd128" => Ok(IsaPath::WasmSimd128),
        other => Err(VokraError::InvalidArgument(format!(
            "{ENV_ISA_OVERRIDE} must be one of scalar|avx2|neon|rvv|wasm-simd128, got `{other}`"
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
        ..CpuFeatures::NONE
    };
    const X86_NO_AVX2: CpuFeatures = CpuFeatures::NONE;
    const ARM: CpuFeatures = CpuFeatures {
        neon: true,
        ..CpuFeatures::NONE
    };
    // AVX2 present but FMA absent: the AVX2 kernels use `_mm256_fmadd_ps`, so
    // this combination must NOT select the Avx2 path (it would SIGILL).
    const AVX2_NO_FMA: CpuFeatures = CpuFeatures {
        avx2: true,
        fma: false,
        ..CpuFeatures::NONE
    };
    // Synthetic feature set for M3-13-T02 unit tests: RVV 1.0 base present
    // (SpacemiT K1 / BPI-F3 baseline); optional Zvfh added in a second variant.
    const RVV_BASE: CpuFeatures = CpuFeatures {
        rvv_v: true,
        ..CpuFeatures::NONE
    };
    const RVV_WITH_ZVFH: CpuFeatures = CpuFeatures {
        rvv_v: true,
        rvv_zvfh: true,
        ..CpuFeatures::NONE
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

    // -------------------------------------------------------------------
    // M4-01-T04 WASM SIMD128 detection + selection unit tests
    //
    // Pure-function tests over synthetic feature sets so they execute on
    // every host in CI. The wasm32 bit is COMPILE-TIME (`cfg!(target_feature
    // = "simd128")`) — WASM has no runtime CPU feature detection, SIMD
    // acceptance is decided at module validation time (ADR M4-01 §4) — so
    // unlike AVX2/NEON there is no probe to exercise, only the selection
    // rules.
    // -------------------------------------------------------------------

    const WASM_SIMD: CpuFeatures = CpuFeatures {
        wasm_simd128: true,
        ..CpuFeatures::NONE
    };

    #[test]
    fn wasm_simd128_selection_rules() {
        // A wasm32+simd128 build selects the WasmSimd128 path; the scalar
        // fallback is always available.
        assert_eq!(WASM_SIMD.best_isa(), IsaPath::WasmSimd128);
        assert!(WASM_SIMD.supports(IsaPath::WasmSimd128));
        assert!(WASM_SIMD.supports(IsaPath::Scalar));
        assert!(!WASM_SIMD.supports(IsaPath::Avx2));
        assert!(!WASM_SIMD.supports(IsaPath::Neon));
        assert!(!WASM_SIMD.supports(IsaPath::Rvv));
        // A base (no-SIMD) wasm build reduces to scalar.
        assert_eq!(X86_NO_AVX2.best_isa(), IsaPath::Scalar);
        assert!(!X86_NO_AVX2.supports(IsaPath::WasmSimd128));
    }

    #[test]
    fn wasm_simd128_override_rejected_on_non_wasm_host_with_explicit_error() {
        // Forcing WasmSimd128 where the module was not built with simd128 is
        // an explicit error (never a silent switch — FR-EX-08 principle).
        let err = select_isa(Some(IsaPath::WasmSimd128), &X86).unwrap_err();
        assert!(matches!(err, VokraError::BackendUnavailable(_)));
        let err = select_isa(Some(IsaPath::WasmSimd128), &ARM).unwrap_err();
        assert!(matches!(err, VokraError::BackendUnavailable(_)));
    }

    #[test]
    fn wasm_simd128_override_accepted_on_simd_wasm_build() {
        assert_eq!(
            select_isa(Some(IsaPath::WasmSimd128), &WASM_SIMD).unwrap(),
            IsaPath::WasmSimd128
        );
    }

    #[test]
    fn parse_override_accepts_wasm_simd128() {
        assert_eq!(
            parse_isa_override("wasm-simd128").unwrap(),
            IsaPath::WasmSimd128
        );
        assert_eq!(
            parse_isa_override("WASM-SIMD128").unwrap(),
            IsaPath::WasmSimd128
        );
    }

    #[test]
    fn isa_path_display_includes_wasm_simd128() {
        assert_eq!(IsaPath::WasmSimd128.to_string(), "wasm-simd128");
    }

    #[test]
    fn detect_off_wasm32_reports_no_wasm_simd128() {
        // The wasm_simd128 bit is `cfg!(all(target_arch = "wasm32",
        // target_feature = "simd128"))` — it can never leak onto native
        // targets (this test suite runs on native CI hosts only; the wasm32
        // artifact is exercised by tools/wasm/run-kernel-parity.mjs, T06).
        let f = CpuFeatures::detect();
        if cfg!(not(target_arch = "wasm32")) {
            assert!(!f.wasm_simd128);
            assert_ne!(f.best_isa(), IsaPath::WasmSimd128);
        }
    }

    // -------------------------------------------------------------------
    // M4-17-T02/T03 server-tier probe unit tests
    //
    // The probe itself is the std macro (CPUID on x86-64, auxv/sysctl-backed
    // std_detect on AArch64 — never a direct getauxval FFI, NFR-DS-02); the
    // tests here pin (a) that detect() plumbs each new field from the right
    // macro invocation on the compiled arch, and (b) that no field can leak
    // onto a foreign arch (the SIGILL-guard precondition: `supports` gates on
    // these bits before any upper-tier kernel is reachable).
    // -------------------------------------------------------------------

    #[test]
    fn detect_x86_server_tier_fields_reflect_host_cpuid() {
        let f = CpuFeatures::detect();
        #[cfg(target_arch = "x86_64")]
        {
            assert_eq!(f.avx512f, std::arch::is_x86_feature_detected!("avx512f"));
            assert_eq!(f.avx512dq, std::arch::is_x86_feature_detected!("avx512dq"));
            assert_eq!(f.avx512bw, std::arch::is_x86_feature_detected!("avx512bw"));
            assert_eq!(f.avx512vl, std::arch::is_x86_feature_detected!("avx512vl"));
            assert_eq!(
                f.avx512vnni,
                std::arch::is_x86_feature_detected!("avx512vnni")
            );
            assert_eq!(
                f.avx512bf16,
                std::arch::is_x86_feature_detected!("avx512bf16")
            );
            // std_detect names AVX-VNNI-256 "avxvnni" (ADR M4-17 §(c)).
            assert_eq!(f.avxvnni256, std::arch::is_x86_feature_detected!("avxvnni"));
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            assert!(!f.avx512f);
            assert!(!f.avx512dq);
            assert!(!f.avx512bw);
            assert!(!f.avx512vl);
            assert!(!f.avx512vnni);
            assert!(!f.avx512bf16);
            assert!(!f.avxvnni256);
        }
    }

    #[test]
    fn detect_arm_server_tier_fields_reflect_host_hwcaps() {
        let f = CpuFeatures::detect();
        #[cfg(target_arch = "aarch64")]
        {
            assert_eq!(f.neon_fp16, std::arch::is_aarch64_feature_detected!("fp16"));
            assert_eq!(
                f.neon_dotprod,
                std::arch::is_aarch64_feature_detected!("dotprod")
            );
            assert_eq!(f.neon_i8mm, std::arch::is_aarch64_feature_detected!("i8mm"));
            assert_eq!(f.neon_bf16, std::arch::is_aarch64_feature_detected!("bf16"));
            // The upper tiers imply the NEON baseline is present.
            assert!(f.neon);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            assert!(!f.neon_fp16);
            assert!(!f.neon_dotprod);
            assert!(!f.neon_i8mm);
            assert!(!f.neon_bf16);
        }
    }

    #[test]
    fn none_constant_is_all_false_and_scalar_only() {
        let f = CpuFeatures::NONE;
        assert_eq!(f.best_isa(), IsaPath::Scalar);
        assert!(f.supports(IsaPath::Scalar));
        assert!(!f.supports(IsaPath::Avx2));
        assert!(!f.supports(IsaPath::Neon));
    }
}
