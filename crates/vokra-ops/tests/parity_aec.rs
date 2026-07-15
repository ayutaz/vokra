//! M4-03 T09/T10 — `aec` upstream parity + convergence e2e against the
//! committed SpeexDSP float-build reference (`tests/parity/aec/`, generated
//! by `tools/parity/aec_dump.py` at the upstream commit pinned in its
//! manifest; CI needs no C toolchain or Python — M3-06 fixture operating
//! model).
//!
//! # Two-tier judgment (ADR M4-03 §D-(h) — why not per-sample everywhere)
//!
//! The canceller is a *recurrent adaptive* system: every frame's weights
//! feed the next frame, so an FP32 rounding difference is amplified along
//! the run and a long-window per-sample comparison is architecturally
//! meaningless (the Kokoro `PROSODY_F0_ATOL` precedent). The judgment is
//! therefore:
//!
//! - **Short window, strict**: the first [`STRICT_FRAMES`] frames are
//!   compared per-sample. Measured on the generating machine (2026-07-15,
//!   Apple M1, `cc -O2 -ffp-contract=off`): max |Δ| = **0 LSB** over the
//!   first 8 frames and ≤ 1 LSB over the whole 250-frame run — the port is
//!   rounding-identical to upstream there. The gate is set at
//!   [`STRICT_ATOL_LSB`] = 16 int16-LSB (≈ 4.9e-4 in the [-1, 1] domain):
//!   16× the measured bound, headroom for cross-platform libm ULP
//!   differences (FFT twiddles / Hann / exp / sqrt are f64 libm calls; the
//!   fixture is macOS-generated, CI runs glibc), and still ~20× tighter
//!   than the NFR-QL-01 FP32 default atol 0.01.
//! - **Long window, metric-domain**: ERLE (echo-return-loss enhancement,
//!   `10·log10(E_near/E_out)` over the converged tail). Measured: upstream
//!   38.69 dB vs port 38.69 dB (identical to 0.01 dB). Gate:
//!   |ΔERLE| ≤ [`ERLE_PARITY_TOL_DB`] = 1.0 dB — an energy metric over 100
//!   frames is robust to trajectory-level FP divergence, and 1 dB is 100×
//!   the measured delta while still far below the 38 dB signal.
//!
//! The e2e (T09) thresholds are **self-calibrating**: the upstream ERLE /
//! double-talk distortion are computed from the committed fixture at test
//! time and ours must reach `0.6×` ERLE (dB) / stay within `1.25×`
//! distortion — the margins of the spec's 0.5-0.8 window, chosen mid-range
//! because adaptive trajectories may diverge across platforms while the
//! converged solution does not (measured ratios: 1.00 / 1.00). No absolute
//! dB threshold is invented anywhere in this file (fabricated-pass 禁止).
//!
//! Behaviour negatives asked by T09 that live in the `aec.rs` unit suite
//! (not duplicated here): rate-mismatch explicit error, far-end-absent
//! pass-through, NaN divergence guard.

use std::path::PathBuf;

use vokra_core::stream::aec_ref_queue;
use vokra_ops::{Aec, AecAttrs, AecStatus};

/// Fixture parameters (mirrors `tests/parity/aec/manifest.txt`).
const SAMPLE_RATE: u32 = 16_000;
const FRAME: usize = 256;
const FILTER_LENGTH: usize = 2048;
const ST_FRAMES: usize = 250;
const DT_SPEECH_START_FRAME: usize = 100;

/// Strict-window width and bound (see module doc for the measured basis).
const STRICT_FRAMES: usize = 8;
const STRICT_ATOL_LSB: i32 = 16;

/// Long-window ERLE parity tolerance (measured Δ = 0.00 dB).
const ERLE_PARITY_TOL_DB: f64 = 1.0;

/// e2e factors (spec M4-03-T09: margin 0.5-0.8 of the reference
/// measurement; mid-range 0.6 / distortion mirror 1.25 = 1/0.8).
const E2E_ERLE_FACTOR: f64 = 0.6;
const E2E_DISTORTION_FACTOR: f64 = 1.25;

fn fixture_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/aec");
    dir.canonicalize()
        .ok()
        .filter(|d| d.join("manifest.txt").exists())
}

/// Gated skip (the committed-fixture twin of the GGUF-gated model parity
/// tests): a missing fixture skips loudly instead of fabricating a pass.
macro_rules! require_fixtures {
    () => {
        match fixture_dir() {
            Some(d) => d,
            None => {
                eprintln!(
                    "SKIP: tests/parity/aec fixtures not found; regenerate with \
                     `python3 tools/parity/aec_dump.py`"
                );
                return;
            }
        }
    };
}

fn read_i16le(dir: &std::path::Path, name: &str) -> Vec<i16> {
    let path = dir.join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn to_unit(v: &[i16]) -> Vec<f32> {
    v.iter().map(|&s| f32::from(s) / 32768.0).collect()
}

/// Upstream's float-build `WORD2INT` (arch.h): round-half-up with clamping —
/// applied to our unit-domain output to compare against the int16 reference.
fn word2int(x: f32) -> i16 {
    if x < -32767.5 {
        -32768
    } else if x > 32766.5 {
        32767
    } else {
        (0.5 + f64::from(x)).floor() as i16
    }
}

fn energy(x: &[f32]) -> f64 {
    x.iter().map(|&v| f64::from(v) * f64::from(v)).sum()
}

fn erle_db(near: &[f32], out: &[f32]) -> f64 {
    10.0 * (energy(near) / energy(out)).log10()
}

fn attrs() -> AecAttrs {
    AecAttrs {
        sample_rate: SAMPLE_RATE,
        frame_size: FRAME,
        filter_length: FILTER_LENGTH,
    }
}

/// Runs our port over the whole fixture via the queue-less parity surface,
/// returning the unit-domain output.
fn run_port(far: &[f32], near: &[f32]) -> Vec<f32> {
    let mut aec = Aec::new(&attrs()).unwrap();
    let frames = far.len() / FRAME;
    let mut ours = vec![0.0f32; far.len()];
    let mut out = vec![0.0f32; FRAME];
    for f in 0..frames {
        let status = aec
            .process_with_far(
                &near[f * FRAME..(f + 1) * FRAME],
                &far[f * FRAME..(f + 1) * FRAME],
                &mut out,
            )
            .unwrap();
        assert_ne!(
            status,
            AecStatus::Reset,
            "divergence guard must not fire on the clean fixture (frame {f})"
        );
        ours[f * FRAME..(f + 1) * FRAME].copy_from_slice(&out);
    }
    ours
}

// ---------------------------------------------------------------------------
// T10 — upstream parity
// ---------------------------------------------------------------------------

/// Short-window strict: the deterministic pre-divergence region must match
/// upstream per-sample within [`STRICT_ATOL_LSB`] (measured: 0 LSB).
#[test]
fn parity_short_window_strict() {
    let dir = require_fixtures!();
    let far = to_unit(&read_i16le(&dir, "st_far.i16le"));
    let near = to_unit(&read_i16le(&dir, "st_near.i16le"));
    let up = read_i16le(&dir, "st_out_speexdsp.i16le");

    let ours = run_port(&far, &near);
    let mut max_d = 0i32;
    for i in 0..STRICT_FRAMES * FRAME {
        let got = word2int(ours[i] * 32768.0);
        let d = (i32::from(got) - i32::from(up[i])).abs();
        if d > max_d {
            max_d = d;
        }
        assert!(
            d <= STRICT_ATOL_LSB,
            "sample {i}: ours {got} vs upstream {} — |Δ| {d} LSB > {STRICT_ATOL_LSB}",
            up[i]
        );
    }
    eprintln!("short-window max|Δ| = {max_d} LSB (gate {STRICT_ATOL_LSB})");
}

/// Long-window metric-domain: converged-tail ERLE within
/// [`ERLE_PARITY_TOL_DB`] of upstream (measured Δ = 0.00 dB). Per-sample
/// comparison is architecturally meaningless here — see the module doc.
#[test]
fn parity_long_window_erle() {
    let dir = require_fixtures!();
    let far = to_unit(&read_i16le(&dir, "st_far.i16le"));
    let near = to_unit(&read_i16le(&dir, "st_near.i16le"));
    let up = to_unit(&read_i16le(&dir, "st_out_speexdsp.i16le"));

    let ours = run_port(&far, &near);
    let a = (ST_FRAMES - 100) * FRAME; // converged tail: last 100 frames
    let b = ST_FRAMES * FRAME;
    let erle_up = erle_db(&near[a..b], &up[a..b]);
    let erle_ours = erle_db(&near[a..b], &ours[a..b]);
    eprintln!("ERLE upstream {erle_up:.2} dB vs ours {erle_ours:.2} dB");
    assert!(
        (erle_up - erle_ours).abs() <= ERLE_PARITY_TOL_DB,
        "converged ERLE diverged: upstream {erle_up:.2} dB vs ours {erle_ours:.2} dB \
         (tol {ERLE_PARITY_TOL_DB} dB)"
    );
}

// ---------------------------------------------------------------------------
// T09 — convergence e2e (full queue path) + behaviour negatives
// ---------------------------------------------------------------------------

/// The real runtime path (AecRefWriter::push + Aec::process with sample-clock
/// alignment) must reach the reference-calibrated ERLE on the same fixture.
#[test]
fn e2e_queue_path_reaches_reference_calibrated_erle() {
    let dir = require_fixtures!();
    let far = to_unit(&read_i16le(&dir, "st_far.i16le"));
    let near = to_unit(&read_i16le(&dir, "st_near.i16le"));
    let up = to_unit(&read_i16le(&dir, "st_out_speexdsp.i16le"));

    let mut aec = Aec::new(&attrs()).unwrap();
    let (mut tx, mut rx) = aec_ref_queue(4 * FILTER_LENGTH, SAMPLE_RATE).unwrap();
    let frames = far.len() / FRAME;
    let mut ours = vec![0.0f32; far.len()];
    let mut out = vec![0.0f32; FRAME];
    for f in 0..frames {
        let pos = (f * FRAME) as u64;
        assert_eq!(
            tx.push(&far[f * FRAME..(f + 1) * FRAME], pos).unwrap(),
            FRAME
        );
        let status = aec
            .process(&near[f * FRAME..(f + 1) * FRAME], pos, &mut rx, &mut out)
            .unwrap();
        assert_eq!(status, AecStatus::Cancelled, "frame {f}");
        ours[f * FRAME..(f + 1) * FRAME].copy_from_slice(&out);
    }

    let a = (ST_FRAMES - 100) * FRAME;
    let b = ST_FRAMES * FRAME;
    let erle_up = erle_db(&near[a..b], &up[a..b]);
    let erle_ours = erle_db(&near[a..b], &ours[a..b]);
    eprintln!(
        "e2e ERLE: ours {erle_ours:.2} dB, reference {erle_up:.2} dB, \
         gate {:.2} dB",
        E2E_ERLE_FACTOR * erle_up
    );
    assert!(
        erle_ours >= E2E_ERLE_FACTOR * erle_up,
        "e2e ERLE {erle_ours:.2} dB below the reference-calibrated gate \
         {:.2} dB (= {E2E_ERLE_FACTOR} × upstream {erle_up:.2} dB)",
        E2E_ERLE_FACTOR * erle_up
    );
}

/// Double-talk: the near-end speech must be preserved — our distortion
/// against the clean near-end component stays within the reference-
/// calibrated bound (measured ratio 1.00).
#[test]
fn e2e_double_talk_preserves_near_end() {
    let dir = require_fixtures!();
    let far = to_unit(&read_i16le(&dir, "dt_far.i16le"));
    let near = to_unit(&read_i16le(&dir, "dt_near.i16le"));
    let speech = to_unit(&read_i16le(&dir, "dt_speech.i16le"));
    let up = to_unit(&read_i16le(&dir, "dt_out_speexdsp.i16le"));

    let ours = run_port(&far, &near);
    let a = DT_SPEECH_START_FRAME * FRAME;
    let e_speech = energy(&speech[a..]);
    let mut d_up = 0.0f64;
    let mut d_ours = 0.0f64;
    for i in a..far.len() {
        d_up += f64::from(up[i] - speech[i]).powi(2);
        d_ours += f64::from(ours[i] - speech[i]).powi(2);
    }
    let up_ratio = d_up / e_speech;
    let ours_ratio = d_ours / e_speech;
    eprintln!("double-talk distortion: ours {ours_ratio:.4} vs reference {up_ratio:.4}");
    assert!(
        ours_ratio <= E2E_DISTORTION_FACTOR * up_ratio,
        "near-end distortion {ours_ratio:.4} exceeds {E2E_DISTORTION_FACTOR} × \
         reference {up_ratio:.4}"
    );
}

/// Mid-run reset() re-converges: after a hard reset at half time, the tail
/// still reaches the calibrated ERLE gate (the tail is long enough for a
/// fresh convergence: 125 frames = 2 s).
#[test]
fn e2e_reset_then_reconverges() {
    let dir = require_fixtures!();
    let far = to_unit(&read_i16le(&dir, "st_far.i16le"));
    let near = to_unit(&read_i16le(&dir, "st_near.i16le"));
    let up = to_unit(&read_i16le(&dir, "st_out_speexdsp.i16le"));

    let mut aec = Aec::new(&attrs()).unwrap();
    let frames = far.len() / FRAME;
    let mut ours = vec![0.0f32; far.len()];
    let mut out = vec![0.0f32; FRAME];
    for f in 0..frames {
        if f == frames / 2 {
            aec.reset();
        }
        aec.process_with_far(
            &near[f * FRAME..(f + 1) * FRAME],
            &far[f * FRAME..(f + 1) * FRAME],
            &mut out,
        )
        .unwrap();
        ours[f * FRAME..(f + 1) * FRAME].copy_from_slice(&out);
    }

    // Judge only the final 60 frames (well after the mid-run reset).
    let a = (ST_FRAMES - 60) * FRAME;
    let b = ST_FRAMES * FRAME;
    let erle_up = erle_db(&near[a..b], &up[a..b]);
    let erle_ours = erle_db(&near[a..b], &ours[a..b]);
    eprintln!("post-reset ERLE: ours {erle_ours:.2} dB (reference {erle_up:.2} dB)");
    assert!(
        erle_ours >= E2E_ERLE_FACTOR * erle_up,
        "post-reset re-convergence too weak: {erle_ours:.2} dB < {:.2} dB",
        E2E_ERLE_FACTOR * erle_up
    );
}
