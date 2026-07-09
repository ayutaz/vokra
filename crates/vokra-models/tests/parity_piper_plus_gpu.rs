//! piper-plus native TTS: CPU vs GPU numerical parity (M3-12-T10..T13).
//!
//! Diffs the deterministic (noise scales zeroed) synthesize pipeline between
//! the CPU reference and the Metal / CUDA backends, per component isolation
//! point: (T10) text encoder `m_p` / `logs_p`, (T11) post-flow decoder-input
//! latent `z`, (T12) MB-iSTFT decoder PCM, (T13) full e2e PCM. The parity
//! bounds come from `docs/adr/0012-piper-plus-gpu.md` §D3: `atol = 0.01` on
//! the components, `atol = 0.05` on PCM (the MB-iSTFT / PQMF tail rounding is
//! looser than the linear-algebra components).
//!
//! # Triple gate (env var + feature + device)
//!
//! Real parity requires the converted voice GGUF, the corresponding backend
//! feature, and a real GPU. The tests are therefore **triply gated** — they
//! run only when
//!
//! 1. `VOKRA_PIPER_V7_GGUF` is set (a converted piper-plus zero-shot v7 voice
//!    with the M0-07 fixtures, `docs/piper-plus-integration.md` §5). This
//!    matches the existing `piper_plus/parity_v7.rs` env var: the same voice
//!    covers both suites.
//! 2. The `metal` or `cuda` feature is enabled at compile time.
//! 3. `Compute::for_backend` succeeds (`BackendUnavailable` = no device →
//!    skip, never a silent CPU fall back per FR-EX-08).
//!
//! CI (which sets neither the env var nor the backend feature) skips silently.
//! Local Metal parity runs on the M1 iMac; CUDA parity runs on vast.ai RTX
//! 4090 (M3-12-T15 owner sanity).
//!
//! # No silent CPU fall back
//!
//! Every skip path prints the reason (`eprintln!`) so a real regression cannot
//! be confused with a device-absent skip. A coverage `UnsupportedOp` for a
//! backend that Phase 4 said is fully covered would be a bug (never a silent
//! substitute), so those errors panic rather than skip.

#![allow(dead_code)] // helpers used only in feature-gated tests below

use vokra_models::piper_plus::{PiperIntermediates, PiperPlusTts};

/// FP32 component-level parity bound (`docs/adr/0012-piper-plus-gpu.md` §D3).
const ATOL_COMPONENT: f32 = 0.01;
/// PCM parity bound (§D3): MB-iSTFT + PQMF tail rounding is looser than the
/// linear-algebra components, so PCM comparisons use `0.05` (matches the
/// existing `piper_plus/parity_v7.rs` decoder-PCM tolerance vs onnxruntime).
const ATOL_PCM: f32 = 0.05;

/// Loads the voice named by `$VOKRA_PIPER_V7_GGUF`, or `None` to skip cleanly
/// (CI has neither the env var nor the GGUF file). The path is the *same* env
/// var the existing `piper_plus/parity_v7.rs` unit-test suite uses, so one
/// converted voice covers both suites without duplication.
fn load_voice() -> Option<PiperPlusTts> {
    let path = std::env::var("VOKRA_PIPER_V7_GGUF").ok()?;
    Some(PiperPlusTts::from_path(&path).expect("load piper-plus v7 voice GGUF"))
}

/// Fixed synthesize inputs — a short deterministic 3-phoneme sequence, no
/// speaker embedding, no prosody. Uses the language `lid = 0` (the
/// zero-language-id path collapses `conditioning::g` to the projected zero
/// vector). `length_scale = 1.0`.
fn fixed_inputs() -> (Vec<i64>, i64, f32) {
    // Phoneme ids valid for any voice with `num_symbols >= 10` (the v7 voice
    // has 256; every downstream index is in-bounds). Three distinct symbols
    // exercise the encoder / duration / flow / decoder path with a non-trivial
    // `T_frames` (each phoneme expands to several frames after the duration
    // predictor's exp).
    (vec![1, 5, 9], 0, 1.0)
}

/// Largest absolute difference between two equal-length slices, with a
/// descriptive panic on a length mismatch (so a shape divergence between CPU
/// and GPU falls out here rather than deep in a length-arithmetic panic).
fn max_abs_diff(a: &[f32], b: &[f32], ctx: &str) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "{ctx}: length {} != expected {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Asserts closeness with a diagnostic on the worst offender and the observed
/// margin (always printed with `--nocapture`, so parity headroom is
/// measurable — not just pass/fail).
fn assert_close(got: &[f32], expected: &[f32], atol: f32, ctx: &str) {
    let mut worst = 0.0f32;
    let mut worst_i = 0usize;
    assert_eq!(
        got.len(),
        expected.len(),
        "{ctx}: length {} != expected {}",
        got.len(),
        expected.len()
    );
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        let d = (g - e).abs();
        if d > worst {
            worst = d;
            worst_i = i;
        }
    }
    eprintln!(
        "[parity] {ctx}: max |Δ| = {worst:.3e} over {} elems (atol {atol})",
        got.len()
    );
    assert!(
        worst <= atol,
        "{ctx}: max |Δ| = {worst} at index {worst_i} (got {} vs {}) exceeds atol {atol}",
        got[worst_i],
        expected[worst_i]
    );
}

/// Runs the deterministic component-capturing synthesize on `backend`; delegates
/// the triple-gate skip logic to the caller (they decide on missing GGUF vs
/// missing device separately).
fn intermediates_on(
    voice: &PiperPlusTts,
    backend: vokra_core::BackendKind,
) -> Result<PiperIntermediates, vokra_core::VokraError> {
    let (ids, lid, length_scale) = fixed_inputs();
    voice.synthesize_with_intermediates(&ids, lid, backend, None, None, length_scale)
}

/// The four M3-12 parity assertions between two [`PiperIntermediates`] runs:
///
/// - T10 text encoder (`m_p` + `logs_p`, atol=0.01)
/// - T11 flow latent (`z`, atol=0.01)
/// - T12 decoder PCM (`pcm.samples`, atol=0.05)
/// - T13 e2e PCM (same PCM diff at atol=0.05; a MEL loss cross-check on top
///   of an already-bit-agreeing PCM would add no signal, so T13 is folded
///   into the PCM diff here — a real MEL loss gate lives in the vokra-eval
///   integration once its fixtures ship, see ADR-0012 §D4)
///
/// Every layer's atol is enforced with the same diagnostic (worst |Δ|, index,
/// values), so a real regression pins the failure to the layer that widened.
fn assert_all_layers_within_atol(
    cpu: &PiperIntermediates,
    gpu: &PiperIntermediates,
    backend_name: &str,
) {
    // T10: text encoder split statistics.
    assert_eq!(
        cpu.t_phonemes, gpu.t_phonemes,
        "{backend_name}: T_phonemes diverged ({} vs {}) — encoder produced a different phoneme count",
        cpu.t_phonemes, gpu.t_phonemes,
    );
    assert_close(
        &gpu.m_p,
        &cpu.m_p,
        ATOL_COMPONENT,
        &format!("{backend_name} text encoder m_p"),
    );
    assert_close(
        &gpu.logs_p,
        &cpu.logs_p,
        ATOL_COMPONENT,
        &format!("{backend_name} text encoder logs_p"),
    );

    // T11: flow latent (post length-regulation + reverse Normalizing Flow).
    assert_eq!(
        cpu.t_frames, gpu.t_frames,
        "{backend_name}: T_frames diverged ({} vs {}) — duration predictor produced a different \
         w_ceil, upstream m_p/logs_p regression",
        cpu.t_frames, gpu.t_frames,
    );
    assert_close(
        &gpu.z,
        &cpu.z,
        ATOL_COMPONENT,
        &format!("{backend_name} flow latent z"),
    );

    // T12 + T13: decoder PCM (and by extension the e2e PCM, which is the same
    // buffer here — the deterministic path has no stochastic tail).
    assert_eq!(
        cpu.pcm.sample_rate, gpu.pcm.sample_rate,
        "{backend_name}: sample rate diverged ({} vs {})",
        cpu.pcm.sample_rate, gpu.pcm.sample_rate,
    );
    assert_close(
        &gpu.pcm.samples,
        &cpu.pcm.samples,
        ATOL_PCM,
        &format!("{backend_name} decoder PCM"),
    );
    // Positive fail-fast on NaN/Inf that would otherwise slip past a
    // "close enough" bound.
    assert!(
        gpu.pcm.samples.iter().all(|s| s.is_finite()),
        "{backend_name}: PCM has NaN/Inf"
    );

    // Report the aggregate margin so a PR reviewer sees parity headroom, not
    // just pass/fail.
    let dmp = max_abs_diff(&gpu.m_p, &cpu.m_p, "m_p");
    let dlp = max_abs_diff(&gpu.logs_p, &cpu.logs_p, "logs_p");
    let dz = max_abs_diff(&gpu.z, &cpu.z, "z");
    let dpcm = max_abs_diff(&gpu.pcm.samples, &cpu.pcm.samples, "pcm");
    eprintln!(
        "[parity] {backend_name} piper-plus e2e: max|Δm_p|={dmp:.3e} max|Δlogs_p|={dlp:.3e} \
         max|Δz|={dz:.3e} max|Δpcm|={dpcm:.3e} (T_phonemes={} T_frames={} samples={})",
        cpu.t_phonemes,
        cpu.t_frames,
        cpu.pcm.samples.len(),
    );
}

// --- Metal parity (M3-12-T10..T13, Metal arm) --------------------------------

/// Full piper-plus on Metal: encoder → duration → flow → MB-iSTFT decoder
/// routed through `Compute::Metal` must match the `Compute::Cpu` reference
/// within the FP32 bounds (NFR-QL-01: `atol = 0.01` per component,
/// `atol = 0.05` on PCM per ADR-0012 §D3). Triply gated on the voice GGUF, the
/// `metal` feature, and a real Metal device — never a silent CPU substitute.
///
/// ```text
/// VOKRA_PIPER_V7_GGUF=voice.gguf \
///     cargo test -p vokra-models --features metal piper_plus_metal_e2e -- --nocapture
/// ```
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
fn piper_plus_metal_e2e_matches_cpu() {
    use vokra_core::BackendKind;
    use vokra_core::VokraError;

    let Some(voice) = load_voice() else {
        eprintln!("skip piper-plus Metal e2e parity: set VOKRA_PIPER_V7_GGUF to a converted voice");
        return;
    };

    // CPU reference (Compute::cpu() is infallible on any target).
    let cpu = intermediates_on(&voice, BackendKind::Cpu).expect("cpu synthesize");

    // Metal path: device-gated. `BackendUnavailable` = no device → skip
    // cleanly. A coverage `UnsupportedOp` would be a real bug (Phase 4 covers
    // every HotOp in PIPER_HOT_OPS = &[HotOp::Gemm]), so it panics.
    let metal = match intermediates_on(&voice, BackendKind::Metal) {
        Ok(m) => m,
        Err(VokraError::BackendUnavailable(e)) => {
            eprintln!("skip Metal piper-plus e2e (no Metal device): {e}");
            return;
        }
        Err(e) => panic!("unexpected error on Metal piper-plus synthesize: {e}"),
    };

    assert_all_layers_within_atol(&cpu, &metal, "Metal");
}

// --- CUDA parity (M3-12-T10..T13, CUDA arm) ----------------------------------

/// Full piper-plus on CUDA: same shape as the Metal arm. This crate is
/// authored on an Apple Mac with no NVIDIA GPU, so it skips cleanly here; it
/// runs for real on the vast.ai RTX 4090 (M3-12-T15 owner sanity), matching
/// the M2-03-T25 / M3-01 CUDA gate:
///
/// ```text
/// VOKRA_PIPER_V7_GGUF=voice.gguf \
///     cargo test -p vokra-models --features cuda piper_plus_cuda_e2e -- --nocapture
/// ```
#[cfg(all(feature = "cuda", any(unix, windows)))]
#[test]
fn piper_plus_cuda_e2e_matches_cpu() {
    use vokra_core::BackendKind;
    use vokra_core::VokraError;

    let Some(voice) = load_voice() else {
        eprintln!("skip piper-plus CUDA e2e parity: set VOKRA_PIPER_V7_GGUF to a converted voice");
        return;
    };

    let cpu = intermediates_on(&voice, BackendKind::Cpu).expect("cpu synthesize");

    // CUDA path: dlopen + device-gated. A missing driver / no device is
    // `BackendUnavailable` → skip. A coverage `UnsupportedOp` would be a real
    // bug (Phase 4 covers HotOp::Gemm on CUDA), so it panics.
    let cuda = match intermediates_on(&voice, BackendKind::Cuda) {
        Ok(c) => c,
        Err(VokraError::BackendUnavailable(e)) => {
            eprintln!("skip CUDA piper-plus e2e (no CUDA device): {e}");
            return;
        }
        Err(e) => panic!("unexpected error on CUDA piper-plus synthesize: {e}"),
    };

    assert_all_layers_within_atol(&cpu, &cuda, "CUDA");
}

// --- Fast smoke tests (compile-only sanity, always run) ----------------------

/// The `PiperIntermediates` struct is re-exported at
/// `vokra_models::piper_plus::PiperIntermediates` and its fields are `pub`;
/// this compile-time check catches an accidental visibility narrowing that
/// would break the parity test's downstream users (e.g. an owner-side
/// diagnostic script). Runs on every `cargo test` regardless of feature /
/// env var.
/// The tuple returned by [`piper_intermediates_is_publicly_reachable`]'s
/// compile-time visibility check. A private-field rename inside
/// [`PiperIntermediates`] would break the corresponding `.len()` / `.sample_rate`
/// projection in the check function and fail this file to compile — that
/// early miss is the whole point of the check.
type Shape = (usize, usize, usize, usize, usize, u32);

#[test]
fn piper_intermediates_is_publicly_reachable() {
    // Verifies that the fields are pub by name (a private-field rename would
    // break this function at compile time). Also nails down the fn item type
    // of the constructor to catch an accidental visibility narrowing.
    fn shape_fields(i: &PiperIntermediates) -> Shape {
        (
            i.m_p.len(),
            i.logs_p.len(),
            i.z.len(),
            i.pcm.samples.len(),
            i.t_phonemes + i.t_frames,
            i.pcm.sample_rate,
        )
    }
    let _: fn(&PiperIntermediates) -> Shape = shape_fields;
}
