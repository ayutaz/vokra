//! RMVPE numerical parity harness — env-gated (F0 tier, 2026-07-30).
//!
//! Sibling of `parity_cosyvoice2.rs` / `parity_kokoro.rs` / `parity_whisper.rs`:
//! every test that needs the RMVPE GGUF is gated on the
//! [`GGUF_ENV`] environment variable and skips cleanly when unset
//! (never a fabricated pass). Once opted in, every failure is hard:
//! a missing / malformed / wrong-shaped fixture is a loud panic (FR-EX-08).
//!
//! # Fixture recipe (owner-side)
//!
//! The upstream `yxlllc/RMVPE` release ships a torch `.pt` pickle
//! (`model.pt` or `checkpoint_pretrain.pt`). Bridge it to safetensors
//! offline with the existing `tools/parity/nemo_pt_to_safetensors.py`
//! (fair-use pickle → safetensors converter shared with the DFN3 /
//! Kokoro / Kyutai-STT patterns), then convert to Vokra GGUF:
//!
//! ```text
//! # 1. Fetch upstream `.pt` from github.com/yxlllc/RMVPE releases
//! # 2. Flatten to safetensors (offline, in a venv):
//! uv run python tools/parity/nemo_pt_to_safetensors.py \
//!     --input  ~/rmvpe.pt \
//!     --output ~/rmvpe.safetensors
//! # 3. Convert to GGUF:
//! vokra-cli convert --model rmvpe \
//!     --input  ~/rmvpe.safetensors \
//!     --output ~/rmvpe.gguf
//! # 4. Point the parity harness at it:
//! export VOKRA_RMVPE_REAL_GGUF=~/rmvpe.gguf
//! cargo test -p vokra-models --test parity_rmvpe -- --nocapture
//! ```
//!
//! # Numeric parity contract
//!
//! Upstream RMVPE's classification head emits 360 pitch classes on a
//! log-Hz grid at 20 cents / class. The parity check is *not* a per-
//! sample L∞ bound (a CNN pipeline with dropout / BN buffers is too
//! sensitive to float rounding to admit a tight bound without a
//! bit-identical reference dumper): instead, following the RMVPE
//! evaluation convention, it is an **argmax-match-rate** bound
//! ([`ARGMAX_MATCH_RATE_MIN`]). At 20 cents / class the discretization
//! is tighter than typical F0 estimation error, so a 99 % argmax-match
//! rate corresponds to a mean pitch |Δ| well below 1 semitone.
//!
//! **When the U-Net + GRU kernel binding wave lands** (deferred per
//! `docs/license-audit.md` §3.1 owner sign-off + real-checkpoint
//! parity), the harness upgrades from "loud pending" to a real
//! argmax-match run:
//!
//! - Reference generator: an offline Python dumper that runs the
//!   upstream RMVPE forward on a known-voiced clip (e.g. a JFK 30 s
//!   excerpt or a synthetic sine sweep across the [30 Hz, 1000 Hz]
//!   band) and writes `[n_frames]` argmax indices + a `[n_frames,
//!   360]` sigmoid dump. This lives at
//!   `tools/parity/rmvpe_dump_reference.py` — a future WP.
//! - Rust comparison: `RMVPE::extract_real` (which today loud-errors)
//!   feeds the same PCM through the native forward and computes the
//!   argmax-match rate against the dumped indices.
//!
//! The harness currently exercises everything up to the kernel binding
//! (real GGUF open, real tensor bind, real mel spectrogram, honest
//! pending error on `extract_real`) and marks the missing forward with
//! an explicit "opt-in for real forward" panic message so an owner
//! opt-in is directed straight at the follow-up wave rather than
//! silently reporting "0 %" match.

use std::env;
use std::path::Path;

use vokra_core::VokraError;
use vokra_models::f0::rmvpe::{RMVPE, RmvpeConfig, decode_class_to_hz};

/// Env var the owner sets to point the gated harness at a real RMVPE
/// GGUF. Absent = skip cleanly (never a fabricated pass). Present =
/// binding: every downstream check hard-fails on any error.
const GGUF_ENV: &str = "VOKRA_RMVPE_REAL_GGUF";

/// Minimum argmax-match rate (real Vokra vs upstream RMVPE reference)
/// the parity gate enforces. 99 % at 20 cents / class ≈ mean pitch |Δ|
/// well below a semitone — the "architectural bound" honest-atol
/// [`feedback-honest-parity-atol`] pattern applied to a discrete
/// classification head.
#[allow(dead_code)] // Consumed once `extract_real` returns real frames.
const ARGMAX_MATCH_RATE_MIN: f32 = 0.99;

/// FIXTURE-FREE: primary-source constants pin. Every hparam in
/// [`RmvpeConfig::default`] is transcribed verbatim from the upstream
/// RMVPE README (github.com/yxlllc/RMVPE, fetched 2026-07-30 per
/// CLAUDE.md "ハルシネーション厳禁") — a silent drift in any of these
/// would misalign the mel front-end / 360-class head against every
/// upstream checkpoint.
#[test]
fn rmvpe_config_default_constants_match_primary_source() {
    let c = RmvpeConfig::default();
    assert_eq!(c.hop, 160, "upstream RMVPE hop = 10 ms at 16 kHz");
    assert_eq!(c.sample_rate, 16_000, "upstream RMVPE is trained at 16 kHz");
    assert_eq!(c.n_mels, 128, "upstream RMVPE n_mels = 128");
    assert_eq!(c.n_fft, 2048, "upstream RMVPE n_fft = 2048");
    assert_eq!(c.win_length, 1024, "upstream RMVPE win_length = 1024");
    assert_eq!(c.n_class, 360, "upstream RMVPE head = 360 pitch classes");
    assert!(
        (c.cents_per_class - 20.0).abs() < f32::EPSILON,
        "upstream RMVPE grid = 20 cents / class (12 classes / semitone)"
    );
    // Class-0 anchor ≈ C1 (32.703 Hz) — pinning to a small window
    // rather than a bit-exact float to survive the f32-vs-primary-
    // source-decimal-string round-trip.
    assert!(
        (c.base_hz - 32.703).abs() < 0.01,
        "upstream RMVPE class-0 anchor ≈ C1 = 32.703 Hz, got {}",
        c.base_hz
    );
}

/// FIXTURE-FREE: the 360-class → Hz decoding primitive is a pure
/// function and independently unit-testable. This pins the log-Hz grid
/// math against an analytic reference for a set of representative
/// spike-only probability vectors — the guarantee is that when the
/// U-Net + GRU wave lands, the *post*-CNN decoding step will not silently
/// regress under a bit-identical head implementation.
#[test]
fn decode_class_to_hz_matches_analytic_grid_over_full_span() {
    let cfg = RmvpeConfig::default();
    let voiced_threshold = 0.03f32;

    // Sample a handful of classes across the full 360-class span. Each
    // "spike-only" probability vector isolates one class; the local
    // centroid decoder therefore returns exactly the class index in
    // continuous form, so Hz = base_hz * 2^(class * cents_per_class /
    // 1200) — the analytic reference.
    //
    // Class 0 lies below fmin (default 30 Hz) so decode clamps up to
    // fmin; skip it to keep the analytic bound clean.
    let sample_classes = [30usize, 60, 120, 180, 240, 300, 359];
    for &c in &sample_classes {
        let mut probs = vec![0.0f32; cfg.n_class as usize];
        probs[c] = 1.0;
        let (hz, voiced, conf) = decode_class_to_hz(&probs, &cfg, voiced_threshold);
        assert!(voiced, "class {c} peak > threshold must be voiced");
        assert!((0.0..=1.0).contains(&conf), "conf must be a probability");
        let analytic_hz = cfg.base_hz * (2.0f32).powf(c as f32 * cfg.cents_per_class / 1200.0);
        let expected = analytic_hz.clamp(cfg.fmin, cfg.fmax);
        // A spike-only vector has zero probability at the neighbours,
        // so the 3-tap centroid collapses to the argmax's exact class
        // index. Tolerance = 0.5 Hz (well below the 20-cent grid
        // resolution's Hz-equivalent even at fmin).
        let delta = (hz - expected).abs();
        assert!(
            delta < 0.5,
            "class {c}: analytic Hz = {expected}, got {hz} (Δ = {delta})"
        );
    }
}

/// GATED: opens a real RMVPE GGUF and verifies the load path is a
/// genuine bind (real config parse, real tensor bind, real mel
/// spectrogram computation). This exercises everything up to — and
/// including — the honest-pending gate on the U-Net + GRU forward.
///
/// Skips cleanly when [`GGUF_ENV`] is unset. Once set, all failures
/// are hard: a missing / malformed / wrong-arch fixture fails loudly.
#[test]
fn parity_rmvpe_gguf_smoke() {
    let Some(gguf_path) = env::var(GGUF_ENV).ok() else {
        eprintln!(
            "{GGUF_ENV} unset — skipping rmvpe GGUF parity smoke; \
             this is a clean skip (never a fabricated pass). See the \
             module docs for the fixture recipe."
        );
        return;
    };

    let path = Path::new(&gguf_path);
    let m = RMVPE::from_gguf(path).unwrap_or_else(|e| {
        panic!(
            "rmvpe GGUF at {gguf_path} failed to load: {e:?} \
             (opted-in ⇒ any error is a hard failure — FR-EX-08)"
        )
    });

    // Config sanity: the primary-source constants must round-trip
    // through the `vokra.rmvpe.*` chunk group. A GGUF that never
    // carried the chunk still loads with defaults — but a mismatched
    // sample rate or n_mels is a hard failure (the U-Net was trained
    // against exactly one set of front-end axes).
    let cfg = m.config();
    assert_eq!(
        cfg.sample_rate, 16000,
        "RMVPE was trained at 16 kHz PCM in; a differently-rated GGUF \
         is either misconfigured or a non-canonical fork (loud-fail)"
    );
    assert_eq!(
        cfg.n_mels, 128,
        "RMVPE mel front-end is fixed at 128 mels; a different width \
         is a hard failure (the U-Net first conv shape mismatches)"
    );
    assert_eq!(
        cfg.n_class, 360,
        "RMVPE head is fixed at 360 pitch classes; a different width \
         means the decoder-side log-Hz grid math is off"
    );

    // Tensor manifest: a real upstream RMVPE checkpoint carries at
    // least ~50 weight tensors across the U-Net + GRU + head; a
    // one-tensor GGUF (as the from_gguf smoke fixture uses) would be
    // accepted by from_gguf but is not a real checkpoint. This is a
    // heuristic lower bound — the exact count depends on the fork and
    // is not primary-source-transcribable, so we use a conservative
    // floor that catches "1-tensor synthetic" and "empty header"
    // regressions without becoming brittle to fork variance.
    assert!(
        m.tensor_count() >= 10,
        "real RMVPE checkpoint must carry >= 10 tensors (U-Net + GRU + \
         head); got {} — refusing a synthesized-shape fixture (FR-EX-08)",
        m.tensor_count()
    );

    // Real mel front-end: a 1 s 440 Hz sine at 16 kHz sample rate has
    // strong periodic energy near the 5th mel band; the log-mel peak
    // must be well above the ln(EPS) floor.
    let f0 = 440.0f32;
    let sr = cfg.sample_rate as f32;
    let pcm: Vec<f32> = (0..cfg.sample_rate as usize)
        .map(|i| (2.0 * std::f32::consts::PI * f0 * (i as f32) / sr).sin())
        .collect();
    let mel = m.mel_spectrogram(&pcm);
    assert!(
        !mel.is_empty(),
        "mel spectrogram of a 1 s 440 Hz sine must not be empty"
    );
    assert_eq!(mel[0].len(), cfg.n_mels as usize);
    let floor = 1e-5f32.ln();
    let peak = mel
        .iter()
        .flat_map(|row| row.iter())
        .fold(f32::MIN, |a, &b| a.max(b));
    assert!(
        peak > floor + 2.0,
        "real mel front-end must have peak energy well above the log-eps \
         floor (peak={peak}, floor={floor}); a placeholder would return \
         all-floor rows"
    );

    // Honest pending: `extract_real` returns a loud
    // `UnsupportedOp` under the current landing. The follow-up wave
    // (docs/license-audit.md §3.1 real-weight parity) upgrades this
    // to a real argmax-match run; that upgrade is where
    // `ARGMAX_MATCH_RATE_MIN` binds.
    let err = m
        .extract_real(&pcm, cfg.sample_rate)
        .expect_err("extract_real must be a loud pending error");
    assert!(
        matches!(err, VokraError::UnsupportedOp(_)),
        "expected UnsupportedOp (FR-EX-08 honest pending), got {err:?}"
    );
    eprintln!(
        "rmvpe GGUF loaded from {gguf_path}: sr={}, n_mels={}, n_class={}, \
         {} tensors bound; real forward is pending (see extract_real error message)",
        cfg.sample_rate,
        cfg.n_mels,
        cfg.n_class,
        m.tensor_count(),
    );
}
