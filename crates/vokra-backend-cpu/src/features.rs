//! Runtime ISA detection and the `VOKRA_CPU_ISA` override (M0-08-T03; extended
//! for RVV 1.0 by M3-13-T02/T03).
//!
//! Detection uses only the standard library's CPUID-based
//! `std::arch::is_x86_feature_detected!` on x86-64, the compile-time guarantee
//! that NEON is an ARMv8-A baseline on AArch64, and `/proc/cpuinfo` parsing on
//! `riscv64-*-linux-*` for the RVV 1.0 `v` extension + Zvfh/ZvfBFmin/Zvbb
//! optional extensions (CLAUDE.md, ADR M3-13) as well as the M4-08 RVV
//! draft-0.7.1 signals (`xtheadvector` isa token / vendor-kernel
//! `cpu-vector : 0.7.1` line) with the RVV 1.0 misdetection guard
//! (ADR M4-08). No extra dependency is
//! introduced (NFR-DS-02), and **no JIT / runtime code generation** is
//! involved (NFR-RL-05): selection only picks which statically compiled
//! kernel to call.
//!
//! The override environment variable
//! `VOKRA_CPU_ISA=scalar|avx2|neon|rvv|rvv071|wasm-simd128` lets tests and
//! CI force a specific path on one machine (M0-08-T18 forced-path job;
//! M3-13-T02 rvv variant; M4-08-T04 rvv071 variant). Requesting a path the host
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
///
/// # `#[non_exhaustive]` semver contract (M4-17-T04, `docs/handoff/m4-12.md` §(e)-2)
///
/// New tiers keep landing after the v1.0 GA C-ABI freeze (M5-13): the
/// reserved names `Amx*` (M5), `Sme*` (M5) and `RvvZvfh*` are pre-recorded in
/// `docs/abi-changelog.md` `## Reserved additions`. `#[non_exhaustive]` makes
/// every such landing a **backward-compatible variant addition**: downstream
/// crates matching on `IsaPath` must carry a `_` arm (the attribute forces
/// it at compile time), so a new variant cannot break them. **Within this
/// crate the attribute has no effect** — `dispatch::build_table` keeps its
/// exhaustive match on purpose, so adding a variant without a kernel table
/// arm is a compile error, not a runtime surprise.
#[non_exhaustive]
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
    /// `rvv_v = true`. The RVV 0.7.1 fallback for pre-ratification T-Head
    /// harts is the separate [`IsaPath::Rvv071`] tier (M4-08) — the two are
    /// instruction-encoding incompatible and generation-exclusive.
    Rvv,
    /// RISC-V RVV **draft 0.7.1** kernels (M4-08) for T-Head XuanTie
    /// C910/C906 harts — LicheePi 4A (TH1520, Tier 1) and Milk-V Duo
    /// (Sophgo CV1800B, Tier 2 = Silero VAD only, NFR-PT-03). Upstream
    /// toolchains treat this generation as the `xtheadvector` vendor
    /// extension; its instruction encodings are **incompatible with the
    /// ratified RVV 1.0** (`IsaPath::Rvv`) — e.g. 0.7.1 `vle.v` uses
    /// width=111 where 1.0 encodes `vle64.v`, and the 0.7.1 `vtype` has no
    /// ta/ma bits (riscv-v-spec tag 0.7.1; ADR M4-08). Selecting `Rvv071`
    /// requires `rvv_071 = true`; auto-selection additionally requires
    /// `rvv_071_auto = true` (kernel-managed vector state, ADR M4-08 §c).
    Rvv071,
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
    /// x86-64 AVX-512 f32 tier (M4-17-T04/T07..T09): 16-lane zmm kernels
    /// compiled with `avx512f,avx512dq,avx512bw,avx512vl` — the four ship
    /// together on every Skylake-X 2017+ / Zen4 server part (FR-BE-01), so
    /// [`CpuFeatures::supports`] gates on the full bundle plus the AVX2+FMA
    /// base the transcendental kernels delegate to (ADR M4-17 §(b)-4).
    Avx512,
    /// x86-64 AVX-512 VNNI INT8 tier (`vpdpbusd`, Cascade Lake 2019+ — the
    /// server INT8 main path, FR-BE-01). Its f32 [`crate::dispatch`] table
    /// delegates to the [`IsaPath::Avx512`] kernels (the gate includes the
    /// full f32 bundle); the INT8 dot-product itself is a separate dispatch
    /// surface (`kernels::kquant_gemv_i8*`, ADR M4-17 §(b)-2).
    Avx512Vnni,
    /// x86-64 AVX-512 BF16 matmul tier (`vdpbf16ps`, Cooper Lake 2020+).
    /// **Opt-in**: never picked implicitly for f32-precision ops — reached
    /// via `kernels::gemm_bf16_on` / [`CpuFeatures::best_bf16_isa`] only
    /// (bf16's 8-bit mantissa is an accuracy cliff, CLAUDE.md "BF16 mantissa
    /// 損失"). Its f32 table delegates to [`IsaPath::Avx512`].
    Avx512Bf16,
    /// x86-64 AVX-VNNI 256-bit INT8 tier (Alder Lake 2021+, P- and E-core).
    /// **The client INT8 main path**: Alder Lake+ client parts fuse AVX-512
    /// off platform-wide, so hybrid CPUs land here (ADR M4-17 §(d) tier
    /// collapse — the probe reports platform-common features, keeping the
    /// process-wide `OnceLock` selection safe under P/E-core migration).
    /// Its f32 table delegates to [`IsaPath::Avx2`].
    AvxVnni256,
    /// ARM64 fp16 arithmetic tier (ARMv8.2, Cortex-A75+ 2018+). **Opt-in**
    /// fp16 GEMM (`kernels::gemm_fp16_on` / [`CpuFeatures::best_fp16_isa`];
    /// fp16's 10-bit mantissa is avoided for f32-precision ops). Its f32
    /// table delegates to [`IsaPath::Neon`].
    NeonFp16,
    /// ARM64 dotprod SDOT/UDOT INT8 tier (ARMv8.2-DotProd; Cortex-A55/A75
    /// 2017 initial cores, Apple A13+ — the ARM INT8 main path, FR-BE-01).
    /// INT8 surface mirrors [`IsaPath::Avx512Vnni`]; f32 table delegates to
    /// [`IsaPath::Neon`].
    NeonDotprod,
    /// ARM64 i8mm SMMLA INT8 matmul tier (ARMv8.6, Apple M2+ — this dev
    /// machine (M1) cannot execute it, so its differential runs on owner
    /// silicon, M4-17-T24). 2x2-tile INT8 matmul (`kernels::kquant_gemv2_i8_on`);
    /// f32 table delegates to [`IsaPath::Neon`].
    NeonI8mm,
    /// ARM64 bf16 BFMMLA matmul tier (ARMv8.6). **Opt-in** like
    /// [`IsaPath::Avx512Bf16`]; f32 table delegates to [`IsaPath::Neon`].
    NeonBf16,
}

impl IsaPath {
    /// Every non-scalar path, for "check all host-supported SIMD tiers"
    /// loops (`selftest::checked_paths`, the differential harnesses, the
    /// forced-path negative tests). Scalar is the oracle, not a checked
    /// path. Kept in the crate so new variants extend one list (the
    /// exhaustive `dispatch::build_table` match still catches a variant
    /// added without a kernel table).
    pub const ALL_SIMD: [IsaPath; 13] = [
        IsaPath::Avx2,
        IsaPath::Neon,
        IsaPath::Rvv,
        IsaPath::Rvv071,
        IsaPath::WasmSimd128,
        IsaPath::Avx512,
        IsaPath::Avx512Vnni,
        IsaPath::Avx512Bf16,
        IsaPath::AvxVnni256,
        IsaPath::NeonFp16,
        IsaPath::NeonDotprod,
        IsaPath::NeonI8mm,
        IsaPath::NeonBf16,
    ];
}

impl fmt::Display for IsaPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Neon => "neon",
            Self::Rvv => "rvv",
            Self::Rvv071 => "rvv071",
            Self::WasmSimd128 => "wasm-simd128",
            Self::Avx512 => "avx512",
            Self::Avx512Vnni => "avx512vnni",
            Self::Avx512Bf16 => "avx512bf16",
            Self::AvxVnni256 => "avxvnni256",
            Self::NeonFp16 => "neon-fp16",
            Self::NeonDotprod => "neon-dotprod",
            Self::NeonI8mm => "neon-i8mm",
            Self::NeonBf16 => "neon-bf16",
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
    /// RISC-V RVV **draft 0.7.1** (T-Head `xtheadvector` lineage, M4-08).
    /// True when the host is positively identified as a 0.7.1 hart whose
    /// shipping kernel manages the vector state: either the mainline
    /// (>= 6.10) `xtheadvector` isa token, or the T-Head vendor-kernel
    /// `cpu-vector : 0.7.1` cpuinfo line (ADR M4-08 §b signals A/B).
    /// Grants [`CpuFeatures::supports`] for [`IsaPath::Rvv071`] — i.e. the
    /// `VOKRA_CPU_ISA=rvv071` override. **Generation-exclusive with
    /// `rvv_v`**: the parser never sets both (SIGILL guard, M4-08-T05).
    pub rvv_071: bool,
    /// Auto-select eligibility for the 0.7.1 tier (ADR M4-08 §c): true only
    /// on the mainline `xtheadvector` signal, where the kernel's vendor
    /// extension framework advertises (and context-switches) the T-Head
    /// vector state. On vendor 5.10 kernels (`cpu-vector : 0.7.1`) this
    /// stays false — the tier is override-only there because per-build
    /// CONFIG_VECTOR enablement cannot be proven from cpuinfo. Invariant:
    /// `rvv_071_auto` implies `rvv_071`.
    pub rvv_071_auto: bool,
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
        rvv_071: false,
        rvv_071_auto: false,
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
                rvv_071: caps.v071,
                rvv_071_auto: caps.v071_auto,
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
    /// needs NEON; `Rvv` needs the RVV 1.0 base `v` extension. The M4-17
    /// server tiers gate on exactly the feature bundles their kernels are
    /// compiled against (ADR M4-17 §(b)-4) — this is the SIGILL guard: a
    /// path `supports` rejects is never dispatched to and can only be forced
    /// into an explicit [`VokraError::BackendUnavailable`].
    pub fn supports(&self, isa: IsaPath) -> bool {
        // The AVX-512 f32 kernel bundle (compiled with
        // `avx512f,avx512dq,avx512bw,avx512vl`) plus the AVX2+FMA base its
        // transcendental kernels delegate to.
        let avx512_f32 = self.avx2
            && self.fma
            && self.avx512f
            && self.avx512dq
            && self.avx512bw
            && self.avx512vl;
        match isa {
            IsaPath::Scalar => true,
            IsaPath::Avx2 => self.avx2 && self.fma,
            IsaPath::Neon => self.neon,
            IsaPath::Rvv => self.rvv_v,
            IsaPath::Rvv071 => self.rvv_071,
            IsaPath::WasmSimd128 => self.wasm_simd128,
            IsaPath::Avx512 => avx512_f32,
            // VNNI / BF16 tiers include the f32 bundle: their f32 kernel
            // table delegates to the Avx512 kernels (ADR M4-17 §(b)-1), and
            // on real silicon VNNI/BF16 never ship without F/DQ/BW/VL
            // (Cascade Lake+ / Cooper Lake / Zen4).
            IsaPath::Avx512Vnni => avx512_f32 && self.avx512vnni,
            IsaPath::Avx512Bf16 => avx512_f32 && self.avx512bf16,
            IsaPath::AvxVnni256 => self.avx2 && self.fma && self.avxvnni256,
            IsaPath::NeonFp16 => self.neon && self.neon_fp16,
            IsaPath::NeonDotprod => self.neon && self.neon_dotprod,
            IsaPath::NeonI8mm => self.neon && self.neon_i8mm,
            IsaPath::NeonBf16 => self.neon && self.neon_bf16,
        }
    }

    /// The most capable path this host supports (M0-08-T03 + M3-13-T03 +
    /// M4-01-T04 + M4-17-T04 selection rule). Arch families are exclusive on
    /// any real host; within a family the ladder is:
    ///
    /// - x86-64: `Avx512Bf16 > Avx512Vnni > Avx512 > AvxVnni256 > Avx2`
    /// - ARM64:  `NeonI8mm > NeonDotprod > NeonBf16 > NeonFp16 > Neon`
    /// - then `Rvv`, `WasmSimd128`, `Scalar` as before.
    ///
    /// Selecting a specialized (INT8/BF16/FP16) tier here never regresses
    /// f32 throughput: those tiers' f32 kernel tables delegate to the best
    /// f32 kernels their gate guarantees (ADR M4-17 §(b)-1/3), and the
    /// specialized kernels are reached through the op-kind selectors
    /// ([`Self::best_int8_isa`] / [`Self::best_bf16_isa`] /
    /// [`Self::best_fp16_isa`]) rather than this ladder. Hybrid-CPU note
    /// (ADR M4-17 §(d)): Alder Lake+ client parts report AVX-512 fused off +
    /// AVX-VNNI-256 present on both core types, so the ladder lands on
    /// `AvxVnni256` platform-wide — no per-core logic is needed and the
    /// process-wide `OnceLock` selection stays sound under P/E migration.
    pub fn best_isa(&self) -> IsaPath {
        // x86-64 family.
        if self.supports(IsaPath::Avx512Bf16) {
            IsaPath::Avx512Bf16
        } else if self.supports(IsaPath::Avx512Vnni) {
            IsaPath::Avx512Vnni
        } else if self.supports(IsaPath::Avx512) {
            IsaPath::Avx512
        } else if self.supports(IsaPath::AvxVnni256) {
            IsaPath::AvxVnni256
        } else if self.avx2 && self.fma {
            IsaPath::Avx2
        // ARM64 family.
        } else if self.supports(IsaPath::NeonI8mm) {
            IsaPath::NeonI8mm
        } else if self.supports(IsaPath::NeonDotprod) {
            IsaPath::NeonDotprod
        } else if self.supports(IsaPath::NeonBf16) {
            IsaPath::NeonBf16
        } else if self.supports(IsaPath::NeonFp16) {
            IsaPath::NeonFp16
        } else if self.neon {
            IsaPath::Neon
        } else if self.rvv_v {
            IsaPath::Rvv
        } else if self.rvv_071_auto {
            // ADR M4-08 §c: the 0.7.1 tier auto-selects only on the
            // kernel-managed signal (`xtheadvector` isa token). Vendor-kernel
            // hosts probe as `rvv_071 = true, rvv_071_auto = false` and stay
            // on scalar unless `VOKRA_CPU_ISA=rvv071` is set explicitly.
            IsaPath::Rvv071
        } else if self.wasm_simd128 {
            IsaPath::WasmSimd128
        } else {
            IsaPath::Scalar
        }
    }

    /// The best K-quant INT8 dot-product tier this host can run, or `None`
    /// when only the scalar-int8 reference path is available (M4-17, ADR
    /// §(b)-2 op-kind selector). x86-64: server VNNI-512 over client
    /// VNNI-256; ARM64: dotprod (i8mm serves the 2-activation matmul shape
    /// through `kernels::kquant_gemv2_i8_on`, not this selector).
    pub fn best_int8_isa(&self) -> Option<IsaPath> {
        if self.supports(IsaPath::Avx512Vnni) {
            Some(IsaPath::Avx512Vnni)
        } else if self.supports(IsaPath::AvxVnni256) {
            Some(IsaPath::AvxVnni256)
        } else if self.supports(IsaPath::NeonDotprod) {
            Some(IsaPath::NeonDotprod)
        } else {
            None
        }
    }

    /// The best **opt-in** BF16 matmul tier, or `None`. Callers must opt in
    /// per-op: bf16's 8-bit mantissa is architecturally lossy (CLAUDE.md
    /// "BF16 mantissa 損失"), so f32-precision ops never route here
    /// implicitly (ADR M4-17 §(b)-2).
    pub fn best_bf16_isa(&self) -> Option<IsaPath> {
        if self.supports(IsaPath::Avx512Bf16) {
            Some(IsaPath::Avx512Bf16)
        } else if self.supports(IsaPath::NeonBf16) {
            Some(IsaPath::NeonBf16)
        } else {
            None
        }
    }

    /// The best **opt-in** fp16 GEMM tier, or `None` (ARM64-only in M4-17;
    /// x86-64 fp16 compute is AMX-FP16 = v1.5+ anchor, out of scope).
    pub fn best_fp16_isa(&self) -> Option<IsaPath> {
        if self.supports(IsaPath::NeonFp16) {
            Some(IsaPath::NeonFp16)
        } else {
            None
        }
    }
}

/// Internal RISC-V capability bundle (M3-13-T02). Only produced on
/// `target_arch = "riscv64"` — the `detect_riscv_caps()` fn is
/// `cfg(target_arch = "riscv64")`-gated so it never appears on other targets.
#[cfg_attr(not(target_arch = "riscv64"), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct RiscvIsaCaps {
    v: bool,
    zvfh: bool,
    zvfbfmin: bool,
    zvbb: bool,
    /// RVV draft 0.7.1 (T-Head xtheadvector lineage) probe — see
    /// [`CpuFeatures::rvv_071`] and ADR M4-08 §b.
    v071: bool,
    /// Auto-select eligibility for the 0.7.1 tier — see
    /// [`CpuFeatures::rvv_071_auto`]. Invariant: `v071_auto` implies `v071`.
    v071_auto: bool,
}

/// Probes `/proc/cpuinfo` for RVV 1.0 / RVV 0.7.1 + optional extensions on
/// Linux riscv64.
///
/// Returns all-false on non-Linux riscv64 (BSD / bare-metal) since we have no
/// portable probe surface there; the runtime dispatch then falls back to the
/// scalar path — this is the same within-CPU-backend ISA fallback used when
/// AVX2 / NEON detection returns false, not the cross-backend silent-fallback
/// forbidden by FR-EX-08.
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
            Ok(text) => parse_riscv_cpuinfo(&text),
            Err(_) => RiscvIsaCaps::default(),
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        RiscvIsaCaps::default()
    }
}

/// Parses the `/proc/cpuinfo` payload for RISC-V ISA extensions and the
/// M4-08 RVV 0.7.1 detection signals.
///
/// Pure function, compiled on every target so the RVV-1.0-misdetection guard
/// is unit-tested on all CI hosts (only [`detect_riscv_caps`] — the actual
/// `/proc/cpuinfo` read — stays riscv64-gated). Extended from the M3-13
/// isa-line scan (`parse_riscv_isa_string`) to the full cpuinfo because the
/// 0.7.1 signals live outside the isa line.
///
/// # Format
///
/// Linux `/proc/cpuinfo` on RISC-V exposes one CPU per stanza with an `isa :`
/// line like `rv64imafdcv_zvfh_zvfbfmin_zvbb_zicsr_zifencei`. We scan for the
/// token `v` (RVV 1.0 base) plus the optional extension names we care about
/// (Zvfh / ZvfBFmin / Zvbb) and the vendor token `xtheadvector`. Token match
/// is case-insensitive and underscore-separated so a heterogeneous
/// descriptor cannot be spoofed by a prefix like `vhole` or `virt`. The
/// mainline "hart isa" per-hart line is deliberately not parsed — the plain
/// "isa" line is the harts' common subset, which is the conservative choice.
///
/// # RVV 0.7.1 signals and the RVV 1.0 misdetection guard (M4-08-T05)
///
/// All facts below are source-verified in ADR M4-08 §T02:
///
/// - **Signal A (auto)**: isa token `xtheadvector` — printed by the mainline
///   (>= 6.10) vendor-extension framework (`vendor_extensions/thead.c`), i.e.
///   the kernel manages the T-Head vector state. Sets `v071` + `v071_auto`.
/// - **Signal B (override-only)**: a `cpu-vector : 0.7.1` line — printed by
///   the T-Head vendor 5.10 kernels from the DT `cpu-vector` property
///   (LicheePi 4A `th1520.dtsi`), combined with a `v` in the isa line. Sets
///   `v071` only.
/// - **Guard C**: the shipping vendor kernels print the DT `riscv,isa`
///   string *verbatim*, and those legacy strings carry the non-extension
///   letters `s`/`u` (`rv64imafdcvsu` on LicheePi 4A, `rv64imafdvcsu` on
///   Milk-V Duo) which mainline never prints. A base descriptor whose tail
///   contains `v` together with `s`/`u` is therefore a legacy verbatim
///   vendor string — every verified instance is a T-Head 0.7.1 hart, so the
///   `v` must NOT probe as ratified RVV 1.0 (SIGILL guard). It does not
///   positively prove a usable 0.7.1 unit either ⇒ scalar (spec §4;
///   Milk-V Duo lands here day-one pending the T14 owner dump backfill).
/// - **Guard D**: an `mvendorid : 0x5b7` (THEAD_VENDOR_ID,
///   `vendorid_list.h`) line paired with `marchid : 0x0` in the same CPU
///   stanza — the userspace mirror of the mainline kernel's own rule
///   (`cpufeature.c`: T-Head cores with marchid 0 implement 0.7.1 and put
///   a bogus `v` in their DTs; ratified-spec cores have non-zero marchid).
///
/// Invariant: `v` and `v071` are never both true (the generations are
/// encoding-incompatible and hart-exclusive).
#[cfg_attr(not(target_arch = "riscv64"), allow(dead_code))]
fn parse_riscv_cpuinfo(cpuinfo: &str) -> RiscvIsaCaps {
    // Raw accumulators (guards applied after the scan).
    let mut saw_v = false;
    let mut zvfh = false;
    let mut zvfbfmin = false;
    let mut zvbb = false;
    let mut xtheadvector = false; // signal A
    let mut cpu_vector_071 = false; // signal B
    let mut legacy_su_fingerprint = false; // guard C
    let mut thead_zero_marchid = false; // guard D
    // Per-stanza mvendorid/marchid pairing for guard D (stanzas are
    // blank-line separated; pairing across stanzas would mis-attribute
    // registers of different harts).
    let mut stanza_mvendorid: Option<u64> = None;
    let mut stanza_marchid: Option<u64> = None;

    fn parse_hex_field(value: &str) -> Option<u64> {
        let v = value.trim();
        let v = v.strip_prefix("0x").or_else(|| v.strip_prefix("0X"))?;
        u64::from_str_radix(v, 16).ok()
    }

    fn fold_stanza(mv: &mut Option<u64>, ma: &mut Option<u64>, out: &mut bool) {
        /// Vendor ID of T-Head in the RISC-V `mvendorid` CSR — transcribed
        /// from Linux `arch/riscv/include/asm/vendorid_list.h`
        /// (`THEAD_VENDOR_ID`), the same constant the mainline kernel's own
        /// clear-the-bogus-v rule keys on (cpufeature.c; ADR M4-08 §T02).
        const THEAD_VENDOR_ID: u64 = 0x5b7;
        if *mv == Some(THEAD_VENDOR_ID) && *ma == Some(0) {
            *out = true;
        }
        *mv = None;
        *ma = None;
    }

    for line in cpuinfo.lines() {
        if line.trim().is_empty() {
            fold_stanza(
                &mut stanza_mvendorid,
                &mut stanza_marchid,
                &mut thead_zero_marchid,
            );
            continue;
        }
        // Case-insensitive match on the key before the `:` — the kernel may
        // render keys as `isa` or `ISA` depending on version.
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        let key = k.trim().to_ascii_lowercase();
        match key.as_str() {
            "isa" => {
                // Tokenise on `_` and any ASCII whitespace so
                // `rv64imafdcv_zvfh` and `rv64gcv zvfh zvbb` both yield the
                // same set.
                for token in v.split(|c: char| c == '_' || c.is_ascii_whitespace()) {
                    let token = token.trim();
                    if token.is_empty() {
                        continue;
                    }
                    let low = token.to_ascii_lowercase();
                    // The base descriptor `rv64...` may contain the `v`
                    // extension fused at the tail (e.g. `rv64imafdcv`), and
                    // on verbatim vendor kernels also the legacy `s`/`u`
                    // letters (guard C above).
                    if low.starts_with("rv64") || low.starts_with("rv32") {
                        let tail = &low[4..];
                        if tail.contains('v') {
                            saw_v = true;
                            if tail.contains('s') || tail.contains('u') {
                                legacy_su_fingerprint = true;
                            }
                        }
                        continue;
                    }
                    match low.as_str() {
                        "v" => saw_v = true,
                        "zvfh" => zvfh = true,
                        "zvfbfmin" => zvfbfmin = true,
                        "zvbb" => zvbb = true,
                        "xtheadvector" => xtheadvector = true,
                        _ => {}
                    }
                }
            }
            // T-Head vendor 5.10 kernel line, verbatim from the DT
            // `cpu-vector` property. Exact "0.7.1" match only — any other
            // value is a future/unknown vector claim we must not guess
            // about.
            "cpu-vector" if v.trim() == "0.7.1" => cpu_vector_071 = true,
            "mvendorid" => stanza_mvendorid = parse_hex_field(v),
            "marchid" => stanza_marchid = parse_hex_field(v),
            _ => {}
        }
    }
    fold_stanza(
        &mut stanza_mvendorid,
        &mut stanza_marchid,
        &mut thead_zero_marchid,
    );

    // Apply the M4-08 signal/guard rules (rustdoc above; ADR M4-08 §b).
    let thead_071_evidence =
        xtheadvector || cpu_vector_071 || legacy_su_fingerprint || thead_zero_marchid;
    let v071 = xtheadvector || (cpu_vector_071 && saw_v);
    RiscvIsaCaps {
        // The RVV 1.0 probe survives only when nothing marks this host as a
        // T-Head 0.7.1-lineage hart (the SIGILL guard: 1.0 encodings must
        // never dispatch onto a 0.7.1 hart).
        v: saw_v && !thead_071_evidence,
        zvfh,
        zvfbfmin,
        zvbb,
        v071,
        v071_auto: xtheadvector,
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
        "rvv" => Ok(IsaPath::Rvv),
        "rvv071" => Ok(IsaPath::Rvv071),
        "wasm-simd128" | "wasm_simd128" => Ok(IsaPath::WasmSimd128),
        // M4-17 server tiers (hyphen and underscore spellings both accepted,
        // matching the wasm-simd128 precedent).
        "avx512" => Ok(IsaPath::Avx512),
        "avx512vnni" => Ok(IsaPath::Avx512Vnni),
        "avx512bf16" => Ok(IsaPath::Avx512Bf16),
        "avxvnni256" => Ok(IsaPath::AvxVnni256),
        "neon-fp16" | "neon_fp16" => Ok(IsaPath::NeonFp16),
        "neon-dotprod" | "neon_dotprod" => Ok(IsaPath::NeonDotprod),
        "neon-i8mm" | "neon_i8mm" => Ok(IsaPath::NeonI8mm),
        "neon-bf16" | "neon_bf16" => Ok(IsaPath::NeonBf16),
        other => Err(VokraError::InvalidArgument(format!(
            "{ENV_ISA_OVERRIDE} must be one of scalar|avx2|avx512|avx512vnni|avx512bf16|avxvnni256|neon|neon-fp16|neon-dotprod|neon-i8mm|neon-bf16|rvv|rvv071|wasm-simd128, got `{other}`"
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
            // Since M4-17 the ladder may land on an upper NEON-family tier
            // (this Apple M1 dev machine picks NeonDotprod); whatever it is,
            // it must be NEON-family and host-supported.
            let best = f.best_isa();
            assert!(
                matches!(
                    best,
                    IsaPath::Neon
                        | IsaPath::NeonFp16
                        | IsaPath::NeonDotprod
                        | IsaPath::NeonI8mm
                        | IsaPath::NeonBf16
                ),
                "aarch64 best_isa must be NEON-family, got {best}"
            );
            assert!(f.supports(best));
            assert!(f.supports(IsaPath::Neon));
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
    // M3-13-T02 /proc/cpuinfo parser: pure-function tests for the ISA
    // string scan. Ungated since M4-08 (the parser itself now compiles on
    // every target so the 0.7.1 guard logic is exercised on all CI hosts);
    // the SpacemiT K1 sample string comes from the vendor's board bring-up
    // documentation and is used verbatim as the canonical input.
    // -------------------------------------------------------------------

    #[test]
    fn parse_riscv_cpuinfo_detects_v_in_rv64_descriptor() {
        // rv64imafdcv — the base descriptor contains `v`. This matches a
        // SpacemiT K1 / BPI-F3 minimal RVV 1.0 line.
        let cpuinfo = "processor\t: 0\nisa\t: rv64imafdcv\n";
        let caps = super::parse_riscv_cpuinfo(cpuinfo);
        assert!(caps.v, "rv64imafdcv must set caps.v");
        assert!(!caps.zvfh);
        assert!(!caps.zvfbfmin);
        assert!(!caps.zvbb);
        assert!(!caps.v071);
    }

    #[test]
    fn parse_riscv_cpuinfo_detects_optional_extensions() {
        // Composite line with all four extensions.
        let cpuinfo = "isa: rv64imafdcv_zvfh_zvfbfmin_zvbb_zicsr_zifencei\n";
        let caps = super::parse_riscv_cpuinfo(cpuinfo);
        assert!(caps.v);
        assert!(caps.zvfh);
        assert!(caps.zvfbfmin);
        assert!(caps.zvbb);
    }

    #[test]
    fn parse_riscv_cpuinfo_returns_default_on_pre_rvv_hart() {
        // rv64gc (no `v`) — what a *mainline* (< 6.10, or >= 6.10 with a
        // legacy DT) kernel reports for the C910-class LicheePi 4A: the
        // kernel's own THEAD+marchid==0 rule strips the bogus DT `v`
        // (cpufeature.c, ADR M4-08 §T02 表 4). Note the shipping *vendor*
        // 5.10 kernel instead prints `rv64imafdcvsu` verbatim — that case
        // is covered by `parse_cpuinfo_licheepi_vendor_kernel_*` above.
        let cpuinfo = "isa: rv64imafdc_zicsr_zifencei\n";
        let caps = super::parse_riscv_cpuinfo(cpuinfo);
        assert!(!caps.v, "rv64imafdc (no v) must NOT set caps.v");
        assert!(!caps.v071, "no 0.7.1 signal on a plain rv64gc line");
    }

    #[test]
    fn parse_riscv_cpuinfo_is_case_insensitive_on_key() {
        // Some kernels emit `ISA :` in uppercase.
        let cpuinfo = "ISA : rv64gcv\n";
        let caps = super::parse_riscv_cpuinfo(cpuinfo);
        assert!(caps.v);
    }

    #[test]
    fn parse_riscv_cpuinfo_ignores_prefixed_tokens() {
        // `virt` and `vhole` must not be misread as the `v` extension.
        let cpuinfo = "isa: rv64imafdc_virt_vhole_zicsr\n";
        let caps = super::parse_riscv_cpuinfo(cpuinfo);
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

    // -------------------------------------------------------------------
    // M4-08 RVV 0.7.1 fallback tier (T-Head C910/C906 = LicheePi 4A /
    // Milk-V Duo) — selection-rule unit tests over synthetic feature sets.
    // Pure functions, so they run on every CI host. Signal semantics are
    // fixed by ADR M4-08 (docs/adr/M4-08-rvv-071-fallback.md §b/§c).
    // -------------------------------------------------------------------

    /// Models a mainline (>= 6.10) kernel advertising the `xtheadvector`
    /// vendor extension (ADR M4-08 signal A): probe passes AND the tier is
    /// auto-select eligible.
    const RVV071_AUTO: CpuFeatures = CpuFeatures {
        rvv_071: true,
        rvv_071_auto: true,
        ..CpuFeatures::NONE
    };

    /// Models the LicheePi 4A vendor 5.10 kernel (`cpu-vector : 0.7.1`,
    /// ADR M4-08 signal B): probe passes (so `VOKRA_CPU_ISA=rvv071` works)
    /// but auto-select stays off — `best_isa` must return `Scalar`.
    const RVV071_OVERRIDE_ONLY: CpuFeatures = CpuFeatures {
        rvv_071: true,
        rvv_071_auto: false,
        ..CpuFeatures::NONE
    };

    #[test]
    fn parse_override_accepts_rvv071_case_insensitively() {
        assert_eq!(parse_isa_override("rvv071").unwrap(), IsaPath::Rvv071);
        assert_eq!(parse_isa_override("RVV071").unwrap(), IsaPath::Rvv071);
        assert_eq!(parse_isa_override("  Rvv071 ").unwrap(), IsaPath::Rvv071);
    }

    #[test]
    fn parse_override_error_lists_rvv071_candidate() {
        // The explicit-error message must enumerate the new token so a
        // misconfigured `VOKRA_CPU_ISA` points the user at the full choice
        // set (FR-EX-08 diagnostics style).
        let err = parse_isa_override("rvv0.7.1").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("rvv071"),
            "override error must list rvv071, got: {msg}"
        );
    }

    #[test]
    fn isa_path_display_includes_rvv071() {
        assert_eq!(IsaPath::Rvv071.to_string(), "rvv071");
    }

    #[test]
    fn rvv071_supports_and_best_isa_selection_rules() {
        // Signal A host (xtheadvector): auto-select eligible.
        assert!(RVV071_AUTO.supports(IsaPath::Rvv071));
        assert_eq!(RVV071_AUTO.best_isa(), IsaPath::Rvv071);
        // Signal B host (cpu-vector 0.7.1 vendor kernel): override works,
        // auto-select must stay scalar (ADR M4-08 §c — fabricated
        // auto-detect is forbidden; kernel CONFIG_VECTOR state is not
        // provable from cpuinfo on vendor 5.10 kernels).
        assert!(RVV071_OVERRIDE_ONLY.supports(IsaPath::Rvv071));
        assert_eq!(RVV071_OVERRIDE_ONLY.best_isa(), IsaPath::Scalar);
        // An RVV 1.0 host never supports the 0.7.1 tier (encoding
        // incompatible — generation-exclusive harts).
        assert!(!RVV_BASE.supports(IsaPath::Rvv071));
        // And a 0.7.1 host never supports the 1.0 tier.
        assert!(!RVV071_AUTO.supports(IsaPath::Rvv));
        assert!(!RVV071_OVERRIDE_ONLY.supports(IsaPath::Rvv));
    }

    #[test]
    fn rvv071_override_rejected_on_non_071_host_with_explicit_error() {
        // Forcing rvv071 on x86 / ARM / RVV 1.0 hosts must be an explicit
        // error (never a silent switch to another path — FR-EX-08).
        for feats in [X86, ARM, X86_NO_AVX2, RVV_BASE, RVV_WITH_ZVFH] {
            let err = select_isa(Some(IsaPath::Rvv071), &feats).unwrap_err();
            assert!(matches!(err, VokraError::BackendUnavailable(_)));
        }
    }

    #[test]
    fn rvv071_override_accepted_on_071_host() {
        // Both signal classes accept the explicit override — this is the
        // first-class enablement path on vendor-kernel boards (spec §4).
        assert_eq!(
            select_isa(Some(IsaPath::Rvv071), &RVV071_AUTO).unwrap(),
            IsaPath::Rvv071
        );
        assert_eq!(
            select_isa(Some(IsaPath::Rvv071), &RVV071_OVERRIDE_ONLY).unwrap(),
            IsaPath::Rvv071
        );
    }

    // -------------------------------------------------------------------
    // M4-08-T05 /proc/cpuinfo parser tests (pure function; ungated so the
    // RVV-1.0-misdetection guard is exercised on every CI host, not only
    // on riscv64 cross-builds). Input strings mirror the shipping-board
    // sources verified in ADR M4-08 §T02 (revyos/thead-kernel th1520.dtsi,
    // milkv-duo/duo-buildroot-sdk cv180x_base_riscv.dtsi, torvalds cpu.c).
    // -------------------------------------------------------------------

    /// LicheePi 4A on the shipping T-Head vendor 5.10 kernel: isa line is
    /// the DT string `rv64imafdcvsu` verbatim (bare `v`!) plus the
    /// vendor-only `cpu-vector : 0.7.1` line. No mvendorid/marchid lines.
    const LICHEEPI_4A_VENDOR_CPUINFO: &str = "processor\t: 0\nhart\t\t: 0\n\
isa\t\t: rv64imafdcvsu\nmmu\t\t: sv39\ncpu-freq\t: 1.848Ghz\n\
cpu-icache\t: 64KB\ncpu-dcache\t: 64KB\ncpu-l2cache\t: 1MB\n\
cpu-tlb\t\t: 1024 4-ways\ncpu-cacheline\t: 64Bytes\ncpu-vector\t: 0.7.1\n";

    /// Milk-V Duo (CV1800B C906) on the sophgo 5.10 SDK kernel: verbatim DT
    /// string `rv64imafdvcsu` (note v BEFORE c), and no `cpu-vector`
    /// property in its DT — no unambiguous positive signal.
    const MILKV_DUO_VENDOR_CPUINFO: &str =
        "processor\t: 0\nhart\t\t: 0\nisa\t\t: rv64imafdvcsu\nmmu\t\t: sv39\n";

    /// Mainline >= 6.10 with the vendor-extension framework: the kernel has
    /// already cleared the bogus `v` (cpufeature.c THEAD+marchid==0 rule)
    /// and advertises `xtheadvector` as an isa token instead.
    const MAINLINE_XTHEADVECTOR_CPUINFO: &str = "processor\t: 0\nhart\t\t: 0\n\
isa\t\t: rv64imafdc_zicsr_zifencei_xtheadvector\nmmu\t\t: sv39\n\
uarch\t\t: thead,c910\nmvendorid\t: 0x5b7\nmarchid\t\t: 0x0\nmimpid\t\t: 0x0\n";

    #[test]
    fn parse_cpuinfo_mainline_rvv10_sets_v_only() {
        // (a) mainline RVV 1.0 notation (SpacemiT K1 / BPI-F3 class).
        let caps = super::parse_riscv_cpuinfo("isa\t: rv64imafdcv_zicsr_zifencei\n");
        assert!(caps.v, "modern bare-v isa line must still probe as RVV 1.0");
        assert!(!caps.v071);
        assert!(!caps.v071_auto);
    }

    #[test]
    fn parse_cpuinfo_licheepi_vendor_kernel_is_override_only_071() {
        // (b) The RVV 1.0 misdetection hazard this WP closes: the shipping
        // LicheePi 4A cpuinfo contains a bare `v` (verbatim DT string), and
        // before M4-08 the parser probed it as RVV 1.0 — dispatching 1.0
        // encodings onto a 0.7.1 hart (SIGILL). The `cpu-vector : 0.7.1`
        // line identifies the hart; probe passes as override-only.
        let caps = super::parse_riscv_cpuinfo(LICHEEPI_4A_VENDOR_CPUINFO);
        assert!(
            !caps.v,
            "0.7.1 hart must NOT probe as RVV 1.0 (SIGILL guard)"
        );
        assert!(
            caps.v071,
            "cpu-vector 0.7.1 + v must enable the 0.7.1 probe"
        );
        assert!(!caps.v071_auto, "vendor-kernel signal is override-only");
    }

    #[test]
    fn parse_cpuinfo_milkv_duo_lands_on_scalar() {
        // (d) Milk-V Duo day-one: bare `v` inside a legacy verbatim DT
        // string (`…vcsu` — the s/u letters never appear on mainline), and
        // no cpu-vector line. Ambiguous ⇒ BOTH probes stay false (scalar;
        // never guess 1.0 or 0.7.1 — spec §4). T14 owner dump backfill may
        // add a positive signal later.
        let caps = super::parse_riscv_cpuinfo(MILKV_DUO_VENDOR_CPUINFO);
        assert!(
            !caps.v,
            "legacy su-fingerprinted v must not probe as RVV 1.0"
        );
        assert!(!caps.v071, "no unambiguous 0.7.1 signal on the Duo cpuinfo");
        assert!(!caps.v071_auto);
    }

    #[test]
    fn parse_cpuinfo_mainline_xtheadvector_is_auto_071() {
        // (e) mainline >= 6.10 vendor-extension framework: kernel manages
        // the T-Head vector state, so auto-select is safe (ADR signal A).
        let caps = super::parse_riscv_cpuinfo(MAINLINE_XTHEADVECTOR_CPUINFO);
        assert!(!caps.v);
        assert!(caps.v071);
        assert!(caps.v071_auto);
    }

    #[test]
    fn parse_cpuinfo_thead_zero_marchid_guards_v() {
        // (f) Belt-and-suspenders mirror of the mainline kernel rule
        // (cpufeature.c: THEAD_VENDOR_ID 0x5b7 + marchid 0x0 ⇒ the DT `v`
        // means 0.7.1, not ratified 1.0): a hypothetical kernel that passes
        // the bogus `v` through alongside mvendorid/marchid lines must not
        // probe as RVV 1.0. No positive 0.7.1 signal either (no proof the
        // kernel manages the vector state).
        let cpuinfo = "isa\t: rv64imafdcv\nmvendorid\t: 0x5b7\nmarchid\t\t: 0x0\nmimpid\t\t: 0x0\n";
        let caps = super::parse_riscv_cpuinfo(cpuinfo);
        assert!(!caps.v);
        assert!(!caps.v071);
    }

    #[test]
    fn parse_cpuinfo_non_thead_vendorid_keeps_v() {
        // (g) A non-T-Head mvendorid with a ratified-1.0 marchid must keep
        // the normal RVV 1.0 probe (the guard is T-Head + marchid==0 only,
        // exactly like the mainline kernel rule).
        let cpuinfo =
            "isa\t: rv64imafdcv_zvfh\nmvendorid\t: 0x710\nmarchid\t\t: 0x8000000058000001\n";
        let caps = super::parse_riscv_cpuinfo(cpuinfo);
        assert!(caps.v);
        assert!(caps.zvfh);
        assert!(!caps.v071);
    }

    #[test]
    fn parse_cpuinfo_thead_ratified_v_nonzero_marchid_keeps_v() {
        // (g') Future T-Head cores with the ratified spec carry a non-zero
        // marchid (mainline cpufeature.c comment) — the guard must NOT
        // penalise them.
        let cpuinfo = "isa\t: rv64imafdcv\nmvendorid\t: 0x5b7\nmarchid\t\t: 0x8000000009140d00\n";
        let caps = super::parse_riscv_cpuinfo(cpuinfo);
        assert!(caps.v, "T-Head + non-zero marchid is a ratified-V hart");
        assert!(!caps.v071);
    }

    #[test]
    fn parse_cpuinfo_marchid_pairing_is_per_stanza() {
        // (i) mvendorid/marchid pairing must not leak across per-CPU
        // stanzas: stanza 0 = non-T-Head vendor, stanza 1 = T-Head with
        // marchid 0 ⇒ the guard fires (any 0.7.1-lineage hart in the
        // system disqualifies the 1.0 probe — conservative direction).
        let cpuinfo = "processor\t: 0\nisa\t: rv64imafdcv\nmvendorid\t: 0x710\nmarchid\t\t: 0x1\n\
\nprocessor\t: 1\nisa\t: rv64imafdcv\nmvendorid\t: 0x5b7\nmarchid\t\t: 0x0\n";
        let caps = super::parse_riscv_cpuinfo(cpuinfo);
        assert!(!caps.v, "a T-Head marchid==0 stanza anywhere must guard v");
        // And the reverse split (fields in different stanzas must NOT be
        // combined into a false T-Head match).
        let cpuinfo_split = "processor\t: 0\nisa\t: rv64imafdcv\nmvendorid\t: 0x5b7\nmarchid\t\t: 0x1\n\
\nprocessor\t: 1\nisa\t: rv64imafdcv\nmvendorid\t: 0x710\nmarchid\t\t: 0x0\n";
        let caps = super::parse_riscv_cpuinfo(cpuinfo_split);
        assert!(
            caps.v,
            "vendor/marchid from different stanzas must not pair"
        );
    }

    #[test]
    fn parse_cpuinfo_never_reports_both_vector_generations() {
        // (h) Invariant: rvv_v and rvv_071 are generation-exclusive — no
        // input may yield both (the SIGILL-guard core of M4-08-T05).
        for cpuinfo in [
            "isa\t: rv64imafdcv_zicsr\n",
            LICHEEPI_4A_VENDOR_CPUINFO,
            MILKV_DUO_VENDOR_CPUINFO,
            MAINLINE_XTHEADVECTOR_CPUINFO,
            // Pathological: both a bare v and an xtheadvector token.
            "isa\t: rv64imafdcv_xtheadvector\n",
            // Pathological: cpu-vector plus a modern-styled v.
            "isa\t: rv64imafdcv\ncpu-vector\t: 0.7.1\n",
        ] {
            let caps = super::parse_riscv_cpuinfo(cpuinfo);
            assert!(
                !(caps.v && caps.v071),
                "rvv_v and rvv_071 must never both be true for: {cpuinfo:?}"
            );
            // auto implies the base probe.
            assert!(!caps.v071_auto || caps.v071);
        }
    }

    #[test]
    fn parse_cpuinfo_cpu_vector_other_versions_do_not_probe_071() {
        // A future vendor kernel reporting a different cpu-vector version
        // must not enable the 0.7.1 tier (exact "0.7.1" match only), but
        // the legacy su-fingerprint still guards the bogus v.
        let cpuinfo = "isa\t: rv64imafdcvsu\ncpu-vector\t: 1.0\n";
        let caps = super::parse_riscv_cpuinfo(cpuinfo);
        assert!(!caps.v071);
        assert!(!caps.v, "su fingerprint guards v regardless of cpu-vector");
    }

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

    // -------------------------------------------------------------------
    // M4-17-T04 server-tier IsaPath ladder unit tests (synthetic feature
    // sets modeled on the silicon classes from FR-BE-01 / ADR M4-17 §(d)).
    // -------------------------------------------------------------------

    /// Skylake-X-class server: AVX-512 F/DQ/BW/VL, no VNNI/BF16.
    const SKYLAKE_X: CpuFeatures = CpuFeatures {
        avx2: true,
        fma: true,
        avx512f: true,
        avx512dq: true,
        avx512bw: true,
        avx512vl: true,
        ..CpuFeatures::NONE
    };
    /// Cascade Lake / Ice Lake server-class: full AVX-512 f32 set + VNNI.
    const CASCADE_LAKE: CpuFeatures = CpuFeatures {
        avx512vnni: true,
        ..SKYLAKE_X
    };
    /// Cooper Lake / Zen4-class: VNNI + BF16 on top of the f32 set.
    const ZEN4_LIKE: CpuFeatures = CpuFeatures {
        avx512vnni: true,
        avx512bf16: true,
        ..SKYLAKE_X
    };
    /// Alder Lake+ client: AVX-512 fused off platform-wide (E-cores lack
    /// it), AVX-VNNI-256 present on both core types (ADR M4-17 §(d)).
    const ALDER_LAKE: CpuFeatures = CpuFeatures {
        avx2: true,
        fma: true,
        avxvnni256: true,
        ..CpuFeatures::NONE
    };
    /// Apple-M1-class ARM64: fp16 + dotprod, no i8mm / bf16.
    const ARM_M1_LIKE: CpuFeatures = CpuFeatures {
        neon: true,
        neon_fp16: true,
        neon_dotprod: true,
        ..CpuFeatures::NONE
    };
    /// Apple-M2+-class / Graviton3-class ARM64: full server tier set.
    const ARM_M2_LIKE: CpuFeatures = CpuFeatures {
        neon: true,
        neon_fp16: true,
        neon_dotprod: true,
        neon_i8mm: true,
        neon_bf16: true,
        ..CpuFeatures::NONE
    };

    #[test]
    fn avx512_tier_requires_the_full_f_dq_bw_vl_bundle() {
        // The f32 kernels are compiled with all four features enabled
        // (ADR M4-17 §(b)-4): an avx512f-only host must NOT select Avx512
        // (running the kernel there could SIGILL on a DQ/BW/VL encoding).
        assert!(SKYLAKE_X.supports(IsaPath::Avx512));
        let f_only = CpuFeatures {
            avx2: true,
            fma: true,
            avx512f: true,
            ..CpuFeatures::NONE
        };
        assert!(!f_only.supports(IsaPath::Avx512));
        assert_eq!(f_only.best_isa(), IsaPath::Avx2);
        // ... and Avx512 without the AVX2+FMA base is likewise rejected
        // (the transcendental kernels delegate to the AVX2 implementations).
        let no_avx2 = CpuFeatures {
            avx2: false,
            fma: false,
            ..SKYLAKE_X
        };
        assert!(!no_avx2.supports(IsaPath::Avx512));
    }

    #[test]
    fn x86_server_ladder_selects_most_capable_tier() {
        assert_eq!(SKYLAKE_X.best_isa(), IsaPath::Avx512);
        assert_eq!(CASCADE_LAKE.best_isa(), IsaPath::Avx512Vnni);
        assert_eq!(ZEN4_LIKE.best_isa(), IsaPath::Avx512Bf16);
        // M4-17-T13 completion criterion: a synthetic Alder Lake set
        // (AVX-512 false / AVX-VNNI-256 true) selects AvxVnni256.
        assert_eq!(ALDER_LAKE.best_isa(), IsaPath::AvxVnni256);
        // Plain AVX2 hosts are unchanged by the ladder extension.
        assert_eq!(X86.best_isa(), IsaPath::Avx2);
    }

    #[test]
    fn vnni_and_bf16_tiers_gate_on_the_f32_bundle_too() {
        // A (hypothetical) VNNI bit without the f32 bundle must not unlock
        // the tier: its f32 KernelTable delegates to the AVX-512 kernels,
        // so the gate is inclusive (ADR M4-17 §(b)-4).
        let vnni_only = CpuFeatures {
            avx2: true,
            fma: true,
            avx512f: true,
            avx512vnni: true,
            ..CpuFeatures::NONE
        };
        assert!(!vnni_only.supports(IsaPath::Avx512Vnni));
        assert!(CASCADE_LAKE.supports(IsaPath::Avx512Vnni));
        assert!(!CASCADE_LAKE.supports(IsaPath::Avx512Bf16));
        assert!(ZEN4_LIKE.supports(IsaPath::Avx512Bf16));
        // AVX-VNNI-256 needs only the AVX2+FMA base.
        assert!(ALDER_LAKE.supports(IsaPath::AvxVnni256));
        assert!(!X86.supports(IsaPath::AvxVnni256));
    }

    #[test]
    fn arm_server_ladder_selects_most_capable_tier() {
        assert_eq!(ARM.best_isa(), IsaPath::Neon);
        assert_eq!(ARM_M1_LIKE.best_isa(), IsaPath::NeonDotprod);
        assert_eq!(ARM_M2_LIKE.best_isa(), IsaPath::NeonI8mm);
        // fp16-only (A75-class without dotprod) picks NeonFp16.
        let fp16_only = CpuFeatures {
            neon: true,
            neon_fp16: true,
            ..CpuFeatures::NONE
        };
        assert_eq!(fp16_only.best_isa(), IsaPath::NeonFp16);
    }

    #[test]
    fn arm_server_tiers_gate_on_their_feature_bits() {
        assert!(ARM_M1_LIKE.supports(IsaPath::NeonFp16));
        assert!(ARM_M1_LIKE.supports(IsaPath::NeonDotprod));
        assert!(!ARM_M1_LIKE.supports(IsaPath::NeonI8mm));
        assert!(!ARM_M1_LIKE.supports(IsaPath::NeonBf16));
        assert!(ARM_M2_LIKE.supports(IsaPath::NeonI8mm));
        assert!(ARM_M2_LIKE.supports(IsaPath::NeonBf16));
        // The plain NEON baseline never unlocks an upper tier.
        for isa in [
            IsaPath::NeonFp16,
            IsaPath::NeonDotprod,
            IsaPath::NeonI8mm,
            IsaPath::NeonBf16,
        ] {
            assert!(!ARM.supports(isa), "plain NEON must not support {isa}");
        }
    }

    #[test]
    fn server_tier_overrides_parse_and_display_round_trip() {
        for (isa, name) in [
            (IsaPath::Avx512, "avx512"),
            (IsaPath::Avx512Vnni, "avx512vnni"),
            (IsaPath::Avx512Bf16, "avx512bf16"),
            (IsaPath::AvxVnni256, "avxvnni256"),
            (IsaPath::NeonFp16, "neon-fp16"),
            (IsaPath::NeonDotprod, "neon-dotprod"),
            (IsaPath::NeonI8mm, "neon-i8mm"),
            (IsaPath::NeonBf16, "neon-bf16"),
        ] {
            assert_eq!(isa.to_string(), name);
            assert_eq!(parse_isa_override(name).unwrap(), isa, "{name}");
            // Case-insensitive and underscore-tolerant (the T21 example
            // names use `neon_dotprod` spelling).
            assert_eq!(parse_isa_override(&name.to_ascii_uppercase()).unwrap(), isa);
            assert_eq!(parse_isa_override(&name.replace('-', "_")).unwrap(), isa);
        }
    }

    #[test]
    fn server_tier_override_rejected_on_unsupporting_host_with_explicit_error() {
        // FR-EX-08 principle: forcing a tier the host cannot run is an
        // explicit BackendUnavailable, never a silent switch (SIGILL guard).
        for isa in [
            IsaPath::Avx512,
            IsaPath::Avx512Vnni,
            IsaPath::Avx512Bf16,
            IsaPath::AvxVnni256,
        ] {
            let err = select_isa(Some(isa), &ARM_M2_LIKE).unwrap_err();
            assert!(matches!(err, VokraError::BackendUnavailable(_)));
        }
        for isa in [
            IsaPath::NeonFp16,
            IsaPath::NeonDotprod,
            IsaPath::NeonI8mm,
            IsaPath::NeonBf16,
        ] {
            let err = select_isa(Some(isa), &ZEN4_LIKE).unwrap_err();
            assert!(matches!(err, VokraError::BackendUnavailable(_)));
        }
        // ... while a supported force is honored.
        assert_eq!(
            select_isa(Some(IsaPath::Avx512Vnni), &ZEN4_LIKE).unwrap(),
            IsaPath::Avx512Vnni
        );
        assert_eq!(
            select_isa(Some(IsaPath::NeonDotprod), &ARM_M1_LIKE).unwrap(),
            IsaPath::NeonDotprod
        );
    }

    #[test]
    fn op_kind_selectors_pick_the_specialized_tiers() {
        // INT8 (ADR M4-17 §(b)-2): server VNNI-512 > client VNNI-256 on
        // x86-64; dotprod on ARM64 (i8mm is the 2-activation matmul shape,
        // selected by the gemv2 surface, not this selector).
        assert_eq!(ZEN4_LIKE.best_int8_isa(), Some(IsaPath::Avx512Vnni));
        assert_eq!(ALDER_LAKE.best_int8_isa(), Some(IsaPath::AvxVnni256));
        assert_eq!(ARM_M1_LIKE.best_int8_isa(), Some(IsaPath::NeonDotprod));
        assert_eq!(X86.best_int8_isa(), None);
        assert_eq!(ARM.best_int8_isa(), None);
        // BF16 (opt-in tier).
        assert_eq!(ZEN4_LIKE.best_bf16_isa(), Some(IsaPath::Avx512Bf16));
        assert_eq!(ARM_M2_LIKE.best_bf16_isa(), Some(IsaPath::NeonBf16));
        assert_eq!(CASCADE_LAKE.best_bf16_isa(), None);
        assert_eq!(ARM_M1_LIKE.best_bf16_isa(), None);
        // FP16 (opt-in tier; ARM64-only in this WP).
        assert_eq!(ARM_M1_LIKE.best_fp16_isa(), Some(IsaPath::NeonFp16));
        assert_eq!(ZEN4_LIKE.best_fp16_isa(), None);
    }
}
