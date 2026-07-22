//! Real-voice regression guard for the piper decoder per-stage shape-driven
//! generalization (M4-RESIDUAL-B (A), T01/T08).
//!
//! The (A) change makes the MB-iSTFT decoder derive its upsample kernel per
//! stage from the tensor shapes, links the ResBlock table to `n_ups`, and
//! generalizes `samples_per_frame` to a per-stage stride product. For the
//! shipping **css10-ja-6lang** voice (2 uniform stride-4 / kernel-16 stages,
//! last upsample width 64) the generalization must reduce **exactly** to the
//! former constants — i.e. produce **bit-identical** PCM.
//!
//! This test loads the real css10 GGUF and pins the synthesized PCM by a stable
//! FNV-1a digest over its little-endian bytes plus the sample count and first /
//! last samples. Proving Vokra-vs-Vokra bit-identity before and after the (A)
//! change is a *sufficient and stronger* proof that the real-weight-eval metric
//! against onnxruntime (mel-L1 ≈ 0.0033–0.0035, ≤3 int16 LSB, `docs/bench-
//! baselines/m1-real-weight-eval-2026-07-16/report.md` §4) is unchanged — that
//! metric is a function of this exact PCM, so an unchanged PCM leaves it
//! unchanged. It does **not** re-run onnxruntime (a separate, out-of-tree
//! harness); the two vehicles are deliberately not conflated.
//!
//! Gated on `VOKRA_PIPER_CSS10_GGUF` (the voice GGUF is ~77 MB, uncommittable —
//! `piper_plus/parity.rs` records the same rationale). It skips cleanly when
//! unset (CI); a run with the var **set** must report "1 test RAN / 0 skipped".
//!
//! ```text
//! VOKRA_PIPER_CSS10_GGUF=~/.cache/vokra-eval/gguf/piper-plus-css10-ja-6lang.gguf \
//!     cargo test -p vokra-models --test piper_css10_geometry -- --nocapture
//! ```

use vokra_models::piper_plus::PiperPlusTts;

/// Deterministic, in-range phoneme ids (fixed so the digest is reproducible).
/// The actual phonemes are irrelevant to a bit-identity regression; the point
/// is to drive the full encoder → duration → flow → **decoder** path on the
/// real weights. `id = i·7 mod num_symbols` is a spread, non-degenerate walk.
fn fixed_ids(num_symbols: usize) -> Vec<i64> {
    (0..24).map(|i| ((i * 7) % num_symbols) as i64).collect()
}

/// FNV-1a 64-bit over the PCM's little-endian f32 bytes (bit-exact digest — a
/// single-ULP change flips it).
fn fnv1a(samples: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for s in samples {
        for b in s.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The captured baseline for the shipping css10-ja-6lang voice, taken on the
/// pre-(A) decoder (uniform stride-4 / kernel-16 constants). After the (A)
/// generalization the same run must reproduce these EXACTLY.
///
/// `None` until captured; a `Some` value turns the test into a hard bit-identity
/// gate. Baseline captured 2026-07-20 on branch `feat/m5-plan-and-wave1` from
/// `piper-plus-css10-ja-6lang-neutralspk.gguf` (the plain css10 voice is
/// single-speaker and lacks `spk_proj`; the neutralspk variant carries the
/// mathematically-neutral speaker projection the real-weight-eval used).
const CSS10_BASELINE: Option<Css10Digest> = Some(Css10Digest {
    len: 6656,
    fnv: 0x2842_024e_bcaa_6a09,
    first_bits: 0xbabe_41c1,
    last_bits: 0x3817_bdfc,
});

#[derive(Debug, PartialEq, Eq)]
struct Css10Digest {
    len: usize,
    fnv: u64,
    first_bits: u32,
    last_bits: u32,
}

#[test]
fn css10_decoder_geometry_regression() {
    let Ok(path) = std::env::var("VOKRA_PIPER_CSS10_GGUF") else {
        eprintln!("skipping css10 regression: VOKRA_PIPER_CSS10_GGUF unset");
        return;
    };
    let voice = PiperPlusTts::from_path(&path).expect("load css10 voice GGUF");
    let num_symbols = voice.config().num_symbols;
    let ids = fixed_ids(num_symbols);

    // Deterministic synthesis (both noise knobs zero, length_scale 1.0), lid 0.
    let audio = voice
        .synthesize_phonemes(&ids, 0, None, None, 0.0, 1.0, 0.0)
        .expect("synthesize css10 phonemes");

    assert!(!audio.samples.is_empty(), "css10 produced no PCM");
    assert!(
        audio.samples.iter().all(|s| s.is_finite()),
        "css10 PCM has non-finite samples"
    );

    let digest = Css10Digest {
        len: audio.samples.len(),
        fnv: fnv1a(&audio.samples),
        first_bits: audio.samples[0].to_bits(),
        last_bits: audio.samples[audio.samples.len() - 1].to_bits(),
    };
    eprintln!(
        "css10 digest: len={} fnv=0x{:016x} first_bits=0x{:08x} last_bits=0x{:08x}",
        digest.len, digest.fnv, digest.first_bits, digest.last_bits
    );

    if let Some(baseline) = CSS10_BASELINE {
        assert_eq!(
            digest, baseline,
            "css10 PCM changed vs the pre-(A) baseline — the per-stage \
             generalization did NOT reduce to the former constants (regression)"
        );
    }
}
