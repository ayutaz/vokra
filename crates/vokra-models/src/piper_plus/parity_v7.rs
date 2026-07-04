//! Numerical parity vs the piper-plus **v7** (zero-shot multi-speaker/lang)
//! onnxruntime reference — through the duration predictor and flow.
//!
//! Fixtures live in `tests/parity/piper_plus_v7/` (committed, generated offline
//! by onnxruntime, deterministic scales `[0,1,0]`, zero speaker embedding + zero
//! prosody, `lid = 0`). The reference is FP16-weight onnxruntime (which casts to
//! FP32 per op), compared against this FP32 native implementation.
//!
//! The 76 MB voice GGUF is far too large to commit, so these tests are gated on
//! `VOKRA_PIPER_V7_GGUF` and skip cleanly when unset (CI stays green). The var
//! is distinct from `VOKRA_PIPER_GGUF` (the older single-speaker voice) so both
//! parity suites coexist:
//!
//! ```text
//! VOKRA_PIPER_V7_GGUF=v7-voice.gguf cargo test -p vokra-models parity_v7
//! ```
//!
//! Coverage (atol 0.01 for the component tensors): `g`, `m_p`, `logs_p`,
//! `sdp_body`, `durations` (plus `ceil(durations)` summing to exactly
//! `T_FRAMES`) and the flow latent `flow_z` (== `dec_input`); the
//! multi-stage-FiLM decoder from `dec_input` to the fullband `pcm` (atol 0.05 —
//! the reference is FP16-weight onnxruntime, whose iSTFT/PQMF tail accumulates
//! more error than the FP32 component stages — with RMS / correlation sanity);
//! and the **end-to-end** `synthesize_phonemes` path from the fixed phoneme ids
//! to `pcm` (same atol 0.05), chaining the native `g` / encoder / duration /
//! flow / decoder rather than the reference intermediates.

use std::path::PathBuf;

use super::PiperPlusTts;
use super::config::HIDDEN;

/// Component parity bound (design: g / m_p / logs_p / sdp / durations / flow_z
/// atol 0.01).
const ATOL: f32 = 0.01;

/// End-to-end PCM parity bound (design: pcm atol 0.05). The reference is
/// FP16-weight onnxruntime, so the decoder / iSTFT / PQMF tail accumulates more
/// error than the FP32 component stages; peak error is bounded by this, and the
/// aggregate RMS / correlation must stay tight (see [`v7_decoder_pcm_parity`]).
const PCM_ATOL: f32 = 0.05;

/// Language id of the fixed reference input (`ja`).
const LID: i64 = 0;

/// Length scale of the fixed reference input (`scales = [0, 1, 0]` → the middle
/// entry, `length_scale = 1`; the outer two zero the noise, making the path
/// deterministic, per `manifest.txt`).
const LENGTH_SCALE: f32 = 1.0;

/// Frame count of the reference `flow_z` / `dec_input` (`[1, 192, 27]`) and the
/// exact `sum(ceil(durations))` the length regulator must produce (`manifest`).
const T_FRAMES: usize = 27;

/// The fixed v7 reference phoneme ids: `[1, 2, …, 13, 0]` (T = 14), per the
/// committed `manifest.txt`.
fn phoneme_ids() -> Vec<i64> {
    (1..=13).chain(std::iter::once(0)).collect()
}

fn fixtures_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = <repo>/crates/vokra-models.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("parity")
        .join("piper_plus_v7")
}

/// Reads a little-endian f32 fixture file.
fn read_f32(name: &str) -> Vec<f32> {
    let path = fixtures_dir().join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(bytes.len() % 4, 0, "{name}: not a whole number of f32");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Loads the voice named by `$VOKRA_PIPER_V7_GGUF`, or `None` to skip (CI).
fn load_voice() -> Option<PiperPlusTts> {
    let path = std::env::var("VOKRA_PIPER_V7_GGUF").ok()?;
    Some(PiperPlusTts::from_path(&path).expect("load piper v7 voice GGUF"))
}

/// Largest absolute difference between two equal-length slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Root-mean-square error and Pearson correlation between two equal-length
/// signals — aggregate sanity for the FP16-reference PCM (a few samples may
/// approach the peak `atol`, but the waveforms must track closely). Accumulated
/// in f64 so the 6912-sample reduction does not lose precision.
fn rms_error_and_correlation(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let (mut se, mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        se += (x - y) * (x - y);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let rms = (se / a.len() as f64).sqrt() as f32;
    let corr = if na > 0.0 && nb > 0.0 {
        (dot / (na.sqrt() * nb.sqrt())) as f32
    } else {
        0.0
    };
    (rms, corr)
}

#[test]
fn v7_global_g_parity() {
    // g = spk_proj(zeros_192) + emb_lang[0]. `spk_proj(0) ≠ 0` (bias / LayerNorm
    // / GELU), so this exercises the whole speaker-projection path — the key fix
    // over the language-only conditioning.
    let Some(voice) = load_voice() else {
        eprintln!("skipping piper v7 g parity: set VOKRA_PIPER_V7_GGUF to run");
        return;
    };
    let g = voice.global_g(None, LID);
    let ref_g = read_f32("g.f32");
    let d = max_abs_diff(&g, &ref_g);
    eprintln!("v7 g parity: max|Δg|={d:.6}, len={} (atol={ATOL})", g.len());
    assert!(d <= ATOL, "g parity {d} exceeds atol {ATOL}");
}

#[test]
fn v7_encoder_m_p_logs_p_parity() {
    // Native text encoder under the full g → prior statistics, vs the reference
    // m_p / logs_p (`/enc_p/Split_output_{0,1}`).
    let Some(voice) = load_voice() else {
        eprintln!("skipping piper v7 encoder parity: set VOKRA_PIPER_V7_GGUF to run");
        return;
    };
    let ids = phoneme_ids();
    let out = voice.encode(&ids, LID).expect("encode");
    let ref_m_p = read_f32("m_p.f32");
    let ref_logs_p = read_f32("logs_p.f32");

    let dm = max_abs_diff(&out.m_p, &ref_m_p);
    let dl = max_abs_diff(&out.logs_p, &ref_logs_p);
    eprintln!("v7 encoder parity: max|Δm_p|={dm:.6}, max|Δlogs_p|={dl:.6} (atol={ATOL})");
    assert!(dm <= ATOL, "m_p parity {dm} exceeds atol {ATOL}");
    assert!(dl <= ATOL, "logs_p parity {dl} exceeds atol {ATOL}");
}

#[test]
fn v7_sdp_body_parity() {
    // The stochastic-duration-predictor body (`pre → + dp.cond(g) → DDSConv →
    // proj`, `/dp/proj/Conv_output_0` `[1, 208, 14]`) under the *full* g
    // (`spk_proj(0) + emb_lang[0]`, not the old language-only conditioning) and
    // the prosody-padded encoder output. Isolates the SDP conditioning body from
    // its spline flows.
    let Some(voice) = load_voice() else {
        eprintln!("skipping piper v7 sdp body parity: set VOKRA_PIPER_V7_GGUF to run");
        return;
    };
    let ids = phoneme_ids();
    let (body, t) = voice.sdp_body(&ids, LID).expect("sdp body");
    let ref_body = read_f32("sdp_body.f32");
    assert_eq!(t, ids.len(), "sdp body phoneme count");
    let db = max_abs_diff(&body, &ref_body);
    eprintln!(
        "v7 sdp body parity: max|Δbody|={db:.6}, len={} (atol={ATOL})",
        body.len()
    );
    assert!(db <= ATOL, "sdp body parity {db} exceeds atol {ATOL}");
}

#[test]
fn v7_duration_parity() {
    // Native encoder + SDP reverse flow (deterministic, `noise_w = 0`) under the
    // full g, vs the reference `durations = exp(logw)·length_scale` (pre-ceil).
    // The integer frame counts `ceil(durations)` must both match the reference
    // element-wise AND sum to exactly `T_FRAMES` (drives length regulation).
    let Some(voice) = load_voice() else {
        eprintln!("skipping piper v7 duration parity: set VOKRA_PIPER_V7_GGUF to run");
        return;
    };
    let ids = phoneme_ids();
    let dur = voice.durations(&ids, LID, LENGTH_SCALE).expect("durations");
    let ref_dur = read_f32("durations.f32");
    let d = max_abs_diff(&dur, &ref_dur);

    let w_ceil: Vec<usize> = dur.iter().map(|x| x.ceil() as usize).collect();
    let ref_ceil: Vec<usize> = ref_dur.iter().map(|x| x.ceil() as usize).collect();
    let sum: usize = w_ceil.iter().sum();
    eprintln!("v7 duration parity: max|Δdur|={d:.6} w_ceil_sum={sum} (atol={ATOL})");
    assert!(d <= ATOL, "duration parity {d} exceeds atol {ATOL}");
    assert_eq!(
        w_ceil, ref_ceil,
        "ceil(durations) must match the reference exactly"
    );
    assert_eq!(sum, T_FRAMES, "sum(ceil(durations)) must equal T_FRAMES");
}

#[test]
fn v7_flow_latent_parity() {
    // Reference `m_p` + reference durations → length-regulate → reverse flow → z,
    // under the full g, vs the reference post-flow latent. `flow_z`
    // (`/flow/flows.0/Concat_output_0`) and `dec_input` (`/Mul_9_output_0`, the
    // decoder input) are the same tensor (`z·y_mask`, mask all ones) — assert the
    // committed fixtures agree, then compare the native flow output to it.
    let Some(voice) = load_voice() else {
        eprintln!("skipping piper v7 flow parity: set VOKRA_PIPER_V7_GGUF to run");
        return;
    };
    let ref_m_p = read_f32("m_p.f32");
    let ref_dur = read_f32("durations.f32");
    let ref_flow_z = read_f32("flow_z.f32");
    let ref_dec_input = read_f32("dec_input.f32");
    assert_eq!(
        ref_flow_z, ref_dec_input,
        "flow_z == dec_input in the reference"
    );

    let t_phonemes = ref_dur.len();
    let w_ceil: Vec<usize> = ref_dur.iter().map(|d| d.ceil() as usize).collect();
    let t_frames: usize = w_ceil.iter().sum();
    assert_eq!(t_frames, T_FRAMES, "sum(ceil(durations)) == T_FRAMES");
    assert_eq!(
        ref_flow_z.len(),
        HIDDEN * T_FRAMES,
        "flow_z shape [HIDDEN, T_FRAMES]"
    );

    let (z, frames) = voice.expand_and_flow(&ref_m_p, t_phonemes, &w_ceil, LID);
    assert_eq!(frames, T_FRAMES, "flow output frame count");
    assert_eq!(z.len(), ref_flow_z.len(), "flow output shape");
    let d = max_abs_diff(&z, &ref_flow_z);
    eprintln!(
        "v7 flow parity: max|Δz|={d:.6}, len={} (atol={ATOL})",
        z.len()
    );
    assert!(d <= ATOL, "flow latent parity {d} exceeds atol {ATOL}");
}

#[test]
fn v7_decoder_pcm_parity() {
    // The reference decoder-input latent `dec_input` (== `flow_z`, the post-flow
    // `z·y_mask`) through the native multi-stage gated-FiLM MB-iSTFT decoder,
    // versus the reference fullband `pcm`. Fed verbatim so this isolates the
    // decoder — the three FiLM stages (`dec.cond` after conv_pre, then
    // `dec.cond_layers.{0,1}` after each upsample+MRF) plus the iSTFT / PQMF
    // head — from the flow / duration stages.
    let Some(voice) = load_voice() else {
        eprintln!("skipping piper v7 decoder parity: set VOKRA_PIPER_V7_GGUF to run");
        return;
    };
    let dec_input = read_f32("dec_input.f32");
    assert_eq!(
        dec_input.len(),
        HIDDEN * T_FRAMES,
        "dec_input shape [HIDDEN, T_FRAMES]"
    );

    let pcm = voice.decode(&dec_input, T_FRAMES, LID).expect("decode");
    let ref_pcm = read_f32("pcm.f32");
    assert_eq!(pcm.len(), ref_pcm.len(), "pcm length");
    assert!(pcm.iter().all(|s| s.is_finite()), "PCM has NaN/Inf");

    let d = max_abs_diff(&pcm, &ref_pcm);
    let (rms_err, corr) = rms_error_and_correlation(&pcm, &ref_pcm);
    eprintln!(
        "v7 decoder pcm parity: max|Δ|={d:.6} rms_err={rms_err:.6} corr={corr:.6} \
         len={} (atol={PCM_ATOL})",
        pcm.len()
    );
    assert!(
        d <= PCM_ATOL,
        "decoder PCM parity {d} exceeds atol {PCM_ATOL}"
    );
    // Aggregate sanity: near-perfect correlation and an RMS error far below the
    // peak tolerance (the FP16-reference floor lives in the peaks, not the bulk).
    assert!(corr >= 0.999, "PCM correlation {corr} below 0.999");
    assert!(rms_err <= 0.01, "PCM rms error {rms_err} exceeds 0.01");
}

#[test]
fn v7_e2e_pcm_parity() {
    // The full native path end-to-end from the fixed reference input: g =
    // spk_proj(zeros) + emb_lang[0] → text encoder → SDP durations → length
    // regulation → reverse flow → multi-stage gated-FiLM MB-iSTFT decoder → PCM,
    // deterministic (noise scales 0, zero speaker embedding, zero prosody →
    // bias-only prosody channels), versus the onnxruntime reference `pcm`. Every
    // component is parity-checked in isolation above; this exercises the public
    // `synthesize_phonemes` API and confirms the chained native outputs (native
    // `m_p` / durations / flow feed the decoder) stay within the FP16-reference
    // PCM tolerance — the stage-4 completion criterion.
    let Some(voice) = load_voice() else {
        eprintln!("skipping piper v7 e2e parity: set VOKRA_PIPER_V7_GGUF to run");
        return;
    };
    let ids = phoneme_ids();
    // None speaker embedding → zeros (spk_proj still non-zero); None prosody →
    // bias-only channels: exactly the fixed reference conditioning.
    let audio = voice
        .synthesize_phonemes(&ids, LID, None, None, 0.0, LENGTH_SCALE, 0.0)
        .expect("synthesize");
    let ref_pcm = read_f32("pcm.f32");
    assert_eq!(audio.samples.len(), ref_pcm.len(), "pcm length");
    assert!(
        audio.samples.iter().all(|s| s.is_finite()),
        "PCM has NaN/Inf"
    );

    let d = max_abs_diff(&audio.samples, &ref_pcm);
    let (rms_err, corr) = rms_error_and_correlation(&audio.samples, &ref_pcm);
    eprintln!(
        "v7 e2e pcm parity: max|Δ|={d:.6} rms_err={rms_err:.6} corr={corr:.6} \
         len={} (atol={PCM_ATOL})",
        audio.samples.len()
    );
    assert!(d <= PCM_ATOL, "e2e PCM parity {d} exceeds atol {PCM_ATOL}");
    // Aggregate sanity (as for the isolated decoder): near-perfect correlation
    // and an RMS error far below the peak tolerance.
    assert!(corr >= 0.999, "PCM correlation {corr} below 0.999");
    assert!(rms_err <= 0.01, "PCM rms error {rms_err} exceeds 0.01");
}
