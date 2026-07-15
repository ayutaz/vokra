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
                rvv_071: false,
                rvv_071_auto: false,
                wasm_simd128: false,
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
                rvv_071: false,
                rvv_071_auto: false,
                wasm_simd128: false,
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
                rvv_071: caps.v071,
                rvv_071_auto: caps.v071_auto,
                wasm_simd128: false,
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
                avx2: false,
                fma: false,
                neon: false,
                rvv_v: false,
                rvv_zvfh: false,
                rvv_zvfbfmin: false,
                rvv_zvbb: false,
                rvv_071: false,
                rvv_071_auto: false,
                wasm_simd128: cfg!(target_feature = "simd128"),
            }
        }
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64",
            target_arch = "wasm32"
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
                rvv_071: false,
                rvv_071_auto: false,
                wasm_simd128: false,
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
            IsaPath::Rvv071 => self.rvv_071,
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
        other => Err(VokraError::InvalidArgument(format!(
            "{ENV_ISA_OVERRIDE} must be one of scalar|avx2|neon|rvv|rvv071|wasm-simd128, got `{other}`"
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
        rvv_071: false,
        rvv_071_auto: false,
        wasm_simd128: false,
    };
    const X86_NO_AVX2: CpuFeatures = CpuFeatures {
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
    };
    const ARM: CpuFeatures = CpuFeatures {
        avx2: false,
        fma: false,
        neon: true,
        rvv_v: false,
        rvv_zvfh: false,
        rvv_zvfbfmin: false,
        rvv_zvbb: false,
        rvv_071: false,
        rvv_071_auto: false,
        wasm_simd128: false,
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
        rvv_071: false,
        rvv_071_auto: false,
        wasm_simd128: false,
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
        rvv_071: false,
        rvv_071_auto: false,
        wasm_simd128: false,
    };
    const RVV_WITH_ZVFH: CpuFeatures = CpuFeatures {
        avx2: false,
        fma: false,
        neon: false,
        rvv_v: true,
        rvv_zvfh: true,
        rvv_zvfbfmin: false,
        rvv_zvbb: false,
        rvv_071: false,
        rvv_071_auto: false,
        wasm_simd128: false,
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
        avx2: false,
        fma: false,
        neon: false,
        rvv_v: false,
        rvv_zvfh: false,
        rvv_zvfbfmin: false,
        rvv_zvbb: false,
        rvv_071: false,
        rvv_071_auto: false,
        wasm_simd128: true,
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
        avx2: false,
        fma: false,
        neon: false,
        rvv_v: false,
        rvv_zvfh: false,
        rvv_zvfbfmin: false,
        rvv_zvbb: false,
        rvv_071: true,
        rvv_071_auto: true,
        wasm_simd128: false,
    };

    /// Models the LicheePi 4A vendor 5.10 kernel (`cpu-vector : 0.7.1`,
    /// ADR M4-08 signal B): probe passes (so `VOKRA_CPU_ISA=rvv071` works)
    /// but auto-select stays off — `best_isa` must return `Scalar`.
    const RVV071_OVERRIDE_ONLY: CpuFeatures = CpuFeatures {
        avx2: false,
        fma: false,
        neon: false,
        rvv_v: false,
        rvv_zvfh: false,
        rvv_zvfbfmin: false,
        rvv_zvbb: false,
        rvv_071: true,
        rvv_071_auto: false,
        wasm_simd128: false,
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
}
