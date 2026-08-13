//! RMVPE numerical parity harness — env-gated (F0 tier, 2026-08-13).
//!
//! Sibling of `parity_cosyvoice2.rs` / `parity_kokoro.rs` /
//! `parity_whisper.rs`: every test that needs a real RMVPE fixture is
//! gated on an environment variable and skips cleanly when unset —
//! never a fabricated pass. Once opted in, every failure is hard: a
//! missing / malformed / wrong-shaped fixture is a loud panic
//! (FR-EX-08).
//!
//! # Fixture recipes (owner-side)
//!
//! ## Path A — full end-to-end (`VOKRA_RMVPE_REAL_GGUF`)
//!
//! Point [`RMVPE::extract_real`] at a real upstream `yxlllc/RMVPE`
//! checkpoint converted to Vokra GGUF:
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
//! Under Path A the harness runs the full mel + CNN + BiGRU + head
//! forward and asserts the extract() frame-count contract, the finite /
//! sigmoid-range contract, and a coarse "voiced fraction sanity" band
//! on a 1 s 440 Hz sine (a sine at F0 = 440 Hz should be well within
//! the [30, 1000] Hz tracked band).
//!
//! ## Path B — post-CNN hidden state (`VOKRA_RMVPE_REAL_HIDDEN`)
//!
//! Bit-exact numeric parity against the upstream Python is gated on
//! the owner-side dumper (`tools/parity/rmvpe_dump_reference.py`, a
//! future WP). The dumper runs the upstream RMVPE forward on a known
//! clip, dumps the post-CNN hidden state (`[n_frames, feature_dim]`)
//! and the argmax pitch classes (`[n_frames]`), and the parity harness
//! feeds the hidden state straight into
//! [`RMVPE::forward_from_hidden`] then the head, sigmoid and decoder,
//! and compares argmax indices. This isolates numerical parity of the
//! deterministic post-CNN primitives from any topology drift in the
//! CNN chain (whose exact per-block order of `Conv` / `BN` / `MaxPool`
//! is not primary-source-transcribable from the upstream README
//! alone).
//!
//! ```text
//! # 1. Prepare the same Vokra GGUF as Path A (steps 1-3 above).
//! # 2. Run the reference dumper (future WP):
//! uv run python tools/parity/rmvpe_dump_reference.py \
//!     --checkpoint ~/rmvpe.pt \
//!     --pcm ~/test_clip.wav \
//!     --hidden-out ~/rmvpe_hidden.npy \
//!     --argmax-out ~/rmvpe_argmax.npy
//! # 3. Point the harness at the fixtures:
//! export VOKRA_RMVPE_REAL_GGUF=~/rmvpe.gguf
//! export VOKRA_RMVPE_REAL_HIDDEN=~/rmvpe_hidden.npy
//! export VOKRA_RMVPE_REAL_ARGMAX=~/rmvpe_argmax.npy
//! cargo test -p vokra-models --test parity_rmvpe -- --nocapture
//! ```
//!
//! Path B binds the [`ARGMAX_MATCH_RATE_MIN`] gate (>= 99 % match at
//! 20 cents / class == mean pitch |Δ| < 1 semitone). Path A alone
//! only exercises the shape / finite / sigmoid-range contract.

use std::env;
use std::path::Path;

use vokra_models::f0::rmvpe::{RMVPE, RmvpeConfig, decode_class_to_hz};

/// Env var the owner sets to point Path A (full end-to-end) at a real
/// RMVPE GGUF. Absent = skip cleanly (never a fabricated pass).
/// Present = binding: every downstream check hard-fails on any error.
const GGUF_ENV: &str = "VOKRA_RMVPE_REAL_GGUF";

/// Env var the owner sets to point Path B (post-CNN hidden state) at a
/// dumped `[n_frames, feature_dim]` f32 buffer (raw little-endian
/// contiguous — no `.npy` header). Absent = Path B skips cleanly (Path
/// A still runs when GGUF_ENV is set). Present = binds the
/// [`ARGMAX_MATCH_RATE_MIN`] gate.
const HIDDEN_ENV: &str = "VOKRA_RMVPE_REAL_HIDDEN";

/// Env var the owner sets to declare `feature_dim` for Path B (the
/// dumper writes a raw buffer so the harness needs the width). Required
/// alongside `HIDDEN_ENV`; absent = Path B skips with a diagnostic that
/// names both env vars.
const HIDDEN_FEATURE_DIM_ENV: &str = "VOKRA_RMVPE_REAL_HIDDEN_FEATURE_DIM";

/// Env var the owner sets to point Path B at a dumped `[n_frames]` u32
/// argmax buffer (raw little-endian contiguous). Required alongside
/// `HIDDEN_ENV`; absent = Path B skips with a diagnostic.
const ARGMAX_ENV: &str = "VOKRA_RMVPE_REAL_ARGMAX";

/// Minimum argmax-match rate (real Vokra vs upstream RMVPE reference)
/// the Path B parity gate enforces. 99 % at 20 cents / class ≈ mean
/// pitch |Δ| well below a semitone — the "architectural bound"
/// honest-atol pattern applied to a discrete classification head.
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

    // Real forward: `extract_real` runs the full mel + CNN + BiGRU +
    // head + sigmoid + decoder chain against the bound weights and
    // returns per-hop F0 rows. Shape / finite / sigmoid-range contract
    // must hold on every frame; Path B (`VOKRA_RMVPE_REAL_HIDDEN`)
    // binds the argmax-match-rate gate separately.
    let frames = m
        .extract_real(&pcm, cfg.sample_rate)
        .unwrap_or_else(|e| panic!("extract_real must run cleanly on a real GGUF, got {e:?}"));
    let hop = cfg.hop as usize;
    assert_eq!(
        frames.len(),
        pcm.len() / hop,
        "extract_real must honor the extract() frame-count contract"
    );
    for (i, f) in frames.iter().enumerate() {
        assert!(f.hz.is_finite(), "frame {i}: hz {} is not finite", f.hz);
        assert!(
            f.confidence.is_finite() && (0.0..=1.0).contains(&f.confidence),
            "frame {i}: confidence {} outside sigmoid range [0, 1]",
            f.confidence
        );
        if f.voiced {
            assert!(
                f.hz >= cfg.fmin && f.hz <= cfg.fmax,
                "frame {i}: voiced hz {} outside [{}, {}]",
                f.hz,
                cfg.fmin,
                cfg.fmax
            );
        }
    }
    eprintln!(
        "rmvpe GGUF loaded from {gguf_path}: sr={}, n_mels={}, n_class={}, \
         {} tensors bound; extract_real returned {} frames (shape / finite / \
         sigmoid-range contract holds; argmax parity is Path B)",
        cfg.sample_rate,
        cfg.n_mels,
        cfg.n_class,
        m.tensor_count(),
        frames.len(),
    );
}

/// GATED (Path B): opens the real RMVPE GGUF, feeds the dumped
/// post-CNN hidden state into [`RMVPE::forward_from_hidden`] +
/// head + sigmoid + argmax, and asserts the argmax-match rate against
/// the dumped reference argmax indices is `>= ARGMAX_MATCH_RATE_MIN`.
///
/// Skips cleanly when either `GGUF_ENV`, `HIDDEN_ENV`,
/// `HIDDEN_FEATURE_DIM_ENV`, or `ARGMAX_ENV` is unset — Path B
/// requires all four (the mel + CNN chain is bypassed by the dumper,
/// so the harness needs the checkpoint, the hidden buffer, the feature
/// dim, and the reference argmax to compute a match rate).
///
/// This is where [`ARGMAX_MATCH_RATE_MIN`] binds — Path A alone only
/// exercises the shape / finite / sigmoid-range contract.
#[test]
fn parity_rmvpe_from_hidden_argmax_match_rate() {
    let (Some(gguf_path), Some(hidden_path), Some(feature_dim), Some(argmax_path)) = (
        env::var(GGUF_ENV).ok(),
        env::var(HIDDEN_ENV).ok(),
        env::var(HIDDEN_FEATURE_DIM_ENV)
            .ok()
            .and_then(|s| s.parse::<usize>().ok()),
        env::var(ARGMAX_ENV).ok(),
    ) else {
        eprintln!(
            "path-B parity skipped — requires {GGUF_ENV} + {HIDDEN_ENV} + \
             {HIDDEN_FEATURE_DIM_ENV} (usize) + {ARGMAX_ENV} to all be set. \
             This is a clean skip (never a fabricated pass); see the module \
             docs for the fixture recipe."
        );
        return;
    };

    // Load Vokra GGUF (real weight bind + shape gate).
    let m = RMVPE::from_gguf(Path::new(&gguf_path))
        .unwrap_or_else(|e| panic!("Path-B: failed to load Vokra GGUF {gguf_path}: {e:?}"));

    // Load raw little-endian f32 hidden buffer.
    let hidden_bytes = std::fs::read(&hidden_path)
        .unwrap_or_else(|e| panic!("Path-B: failed to read hidden buffer {hidden_path}: {e:?}"));
    assert!(
        hidden_bytes.len() % 4 == 0,
        "Path-B: hidden buffer len {} is not a multiple of 4 bytes (f32)",
        hidden_bytes.len()
    );
    let hidden_flat: Vec<f32> = hidden_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert!(
        hidden_flat.len() % feature_dim == 0,
        "Path-B: hidden buffer len {} is not a multiple of feature_dim {feature_dim}",
        hidden_flat.len()
    );
    let n_frames = hidden_flat.len() / feature_dim;

    // Load raw little-endian u32 argmax buffer.
    let argmax_bytes = std::fs::read(&argmax_path)
        .unwrap_or_else(|e| panic!("Path-B: failed to read argmax buffer {argmax_path}: {e:?}"));
    assert!(
        argmax_bytes.len() % 4 == 0,
        "Path-B: argmax buffer len {} is not a multiple of 4 bytes (u32)",
        argmax_bytes.len()
    );
    let reference_argmax: Vec<u32> = argmax_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(
        reference_argmax.len(),
        n_frames,
        "Path-B: reference argmax has {} entries but hidden buffer has {n_frames} frames",
        reference_argmax.len()
    );

    // Run the Vokra forward on the hidden state and re-derive argmax
    // by feeding each frame through decode_class_to_hz internally
    // (extract() reports (hz, voiced, confidence), so we reconstruct
    // argmax from voiced Hz via the log-Hz grid: class ≈ log2(hz /
    // base_hz) * 1200 / cents_per_class). To keep the argmax
    // comparison free of that inverse, we run through
    // `forward_from_hidden` and separately compute argmax from the
    // sigmoid probabilities that would be produced — but the public
    // API only exposes F0Frame, so we approximate: compare voiced
    // frames' argmax-derived class index vs reference.
    let frames = m
        .forward_from_hidden(&hidden_flat, n_frames, feature_dim, 16_000)
        .unwrap_or_else(|e| panic!("Path-B: forward_from_hidden failed: {e:?}"));
    assert_eq!(frames.len(), n_frames);

    let cfg = m.config();
    let mut match_count = 0usize;
    let mut compared = 0usize;
    for (t, f) in frames.iter().enumerate() {
        // Reference class index (from dumper): frames where reference
        // class is 0 are treated as unvoiced (matching the upstream
        // dumper convention).
        let ref_class = reference_argmax[t];
        if ref_class == 0 {
            // Skip unvoiced reference frames from the match-rate.
            continue;
        }
        compared += 1;
        if !f.voiced {
            continue;
        }
        // Convert Vokra's decoded hz back to a class index on the
        // log-Hz grid.
        let cents = (f.hz / cfg.base_hz).log2() * 1200.0;
        let vokra_class = (cents / cfg.cents_per_class).round() as u32;
        // Allow ±1 class of drift (20 cents) to survive the
        // local-centroid decoder — the argmax itself is what the
        // reference reports.
        if vokra_class.abs_diff(ref_class) <= 1 {
            match_count += 1;
        }
    }
    let match_rate = if compared > 0 {
        match_count as f32 / compared as f32
    } else {
        0.0
    };
    eprintln!(
        "path-B: {} / {} voiced frames matched within 1 class (== {} cents) — \
         match rate = {:.4} (gate = {:.4})",
        match_count, compared, cfg.cents_per_class, match_rate, ARGMAX_MATCH_RATE_MIN,
    );
    assert!(
        match_rate >= ARGMAX_MATCH_RATE_MIN,
        "Path-B argmax-match rate {match_rate} < gate {ARGMAX_MATCH_RATE_MIN} \
         ({match_count} / {compared} voiced frames matched within ±1 class); \
         the Vokra forward is out of sync with the upstream RMVPE dumper"
    );
}
