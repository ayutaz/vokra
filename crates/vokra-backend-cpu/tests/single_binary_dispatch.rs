//! Single-binary runtime-dispatch acceptance (M1-05).
//!
//! Proves, at `cargo test` level (independent of any CI wiring), the two
//! things the "one binary runs on x86-64 and ARM64 via runtime dispatch"
//! claim rests on (FR-BE-01, FR-EX-06):
//!
//! 1. the shipped binary selects a host-supported ISA path, stably, and honors
//!    the `VOKRA_CPU_ISA` override when set (the CI forced-path leg reads this
//!    directly); and
//! 2. the selected SIMD path is self-consistent with the scalar oracle on this
//!    host ([`vokra_backend_cpu::selftest`]).
//!
//! It also lifts the pure-function `features` unit coverage to the *binary*
//! level: a subprocess forces each ISA path via `VOKRA_CPU_ISA` and asserts a
//! supported force is honored while an unsupported force fails fast (never a
//! silent switch — FR-EX-08 principle).

use vokra_backend_cpu::{CpuFeatures, IsaPath, SELFTEST_ATOL, active_isa, selftest};

/// Environment variable the CPU backend reads to force an ISA path (kept in
/// sync with `features::ENV_ISA_OVERRIDE`; it is a stable user-facing name).
const ENV_ISA_OVERRIDE: &str = "VOKRA_CPU_ISA";

#[test]
fn active_isa_is_host_supported_and_reflects_env_or_best() {
    let isa = active_isa();
    // Selection is fixed after the first call (OnceLock).
    assert_eq!(isa, active_isa(), "active_isa must be stable across calls");
    // Whatever was selected, this host can actually run it.
    assert!(
        CpuFeatures::detect().supports(isa),
        "active_isa {isa} is not supported by the detected host features"
    );

    match std::env::var(ENV_ISA_OVERRIDE) {
        // CI forced-path leg: the single binary must pick exactly the forced
        // path (the leg only ever forces a host-supported path; an unsupported
        // force would fail fast in `active_isa()` — covered by the subprocess
        // test below).
        Ok(forced) => assert_eq!(
            isa.to_string(),
            forced.trim().to_ascii_lowercase(),
            "with {ENV_ISA_OVERRIDE} set, the forced path must win"
        ),
        // No override: the single binary picks the host's fastest path.
        Err(_) => assert_eq!(
            isa,
            CpuFeatures::detect().best_isa(),
            "without an override, active_isa must equal best_isa"
        ),
    }
}

#[test]
fn selftest_succeeds_and_matches_active_isa() {
    let report = selftest().expect("cpu selftest must pass on this host");
    assert_eq!(report.active_isa, active_isa());
    assert!(report.features.supports(report.active_isa));
    assert!(
        report.max_abs_diff <= SELFTEST_ATOL,
        "selftest max_abs_diff {} exceeds atol {SELFTEST_ATOL}",
        report.max_abs_diff
    );
    // A non-scalar host default is among the cross-checked SIMD paths.
    if report.active_isa != IsaPath::Scalar {
        assert!(report.checked_paths.contains(&report.active_isa));
    }
}

/// Subprocess child: no-op in a normal test run; when invoked by
/// [`forced_isa_paths_via_subprocess`] with `VOKRA_FORCED_CHILD=1` it asserts
/// that `active_isa()` equals the forced `VOKRA_EXPECT_ISA`. Forcing an
/// unsupported path makes `active_isa()` panic (fail-fast), which the parent
/// observes as a non-zero exit.
#[test]
fn forced_isa_child_helper() {
    if std::env::var("VOKRA_FORCED_CHILD").is_err() {
        return; // ordinary run: nothing to do.
    }
    let expect = std::env::var("VOKRA_EXPECT_ISA").expect("child needs VOKRA_EXPECT_ISA");
    let got = active_isa().to_string();
    assert_eq!(got, expect, "forced {ENV_ISA_OVERRIDE} path mismatch");
}

#[test]
fn forced_isa_paths_via_subprocess() {
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(_) => {
            eprintln!("skip: current_exe() unavailable in this environment");
            return;
        }
    };
    let feats = CpuFeatures::detect();

    // Re-exec this test binary running ONLY the child helper, forcing `isa`.
    let spawn = |isa: &str, expect: &str| {
        std::process::Command::new(&exe)
            .args([
                "forced_isa_child_helper",
                "--exact",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("VOKRA_FORCED_CHILD", "1")
            .env(ENV_ISA_OVERRIDE, isa)
            .env("VOKRA_EXPECT_ISA", expect)
            .output()
    };

    // Probe: if the sandbox forbids re-exec, skip cleanly (the in-process
    // tests above still cover selection + self-consistency).
    let probe = match spawn("scalar", "scalar") {
        Ok(out) => out,
        Err(e) => {
            eprintln!("skip: cannot re-exec test binary ({e})");
            return;
        }
    };
    assert!(
        probe.status.success(),
        "forcing the always-available scalar path must be honored"
    );

    let run = |isa: &str, expect: &str| {
        spawn(isa, expect)
            .expect("subprocess spawn")
            .status
            .success()
    };

    // The host's fastest SIMD path, forced by name, must be honored.
    let best = feats.best_isa();
    if best != IsaPath::Scalar {
        let name = best.to_string();
        assert!(
            run(&name, &name),
            "forcing the host SIMD path {name} must be honored"
        );
    }

    // Forcing a path the host cannot run must fail fast, never silently switch.
    // Includes IsaPath::Rvv (M3-13) and IsaPath::Rvv071 (M4-08) — on x86-64 /
    // aarch64 CI runners both RISC-V vector paths are unavailable and forcing
    // either must be an explicit error.
    for isa in IsaPath::ALL_SIMD {
        if !feats.supports(isa) {
            let name = isa.to_string();
            assert!(
                !run(&name, &name),
                "forcing the unsupported path {name} must fail fast (BackendUnavailable)"
            );
        }
    }
}
