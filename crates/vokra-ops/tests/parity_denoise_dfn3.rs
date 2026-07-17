//! `denoise` real-weight parity vs the upstream DeepFilterNet3 reference
//! (M4-20 T17, numerical-parity discipline).
//!
//! Env-gated: runs only when both env vars point at the prepared assets
//! (skips cleanly otherwise — CI without the checkpoint stays green without
//! fabricating a pass):
//!
//! * `VOKRA_DFN3_GGUF` — the converted real checkpoint
//!   (`vokra-cli convert --model denoise` over the
//!   `tools/parity/dfn3_prepare_checkpoint.py` safetensors; release
//!   checkpoint sha256 `49c52edc…`).
//! * `VOKRA_DFN3_DATA` — the reference dir holding `clean_48k.f32` /
//!   `noisy_48k.f32` / `enhanced_upstream.f32` (11 s JFK utterance + seeded
//!   5 dB white noise @ 48 kHz, `prep_noisy.py`) and `taps/` (per-stage
//!   dumps from `tools/parity/dfn3_dump_reference.py` — the REAL upstream
//!   `deepfilternet` 0.5.6 package running the REAL released checkpoint).
//!
//! Reference provenance (2026-07-17 measured run, Apple M1, torch 2.1.2
//! CPU): every tolerance below is the measured max |Δ| of that run × a
//! small headroom factor — never a CI-green-seeking constant. Measured
//! stage maxima are recorded inline next to each bound.

use std::path::PathBuf;

use vokra_core::gguf::GgufFile;
use vokra_ops::DenoiseModel;

fn env_paths() -> Option<(PathBuf, PathBuf)> {
    let gguf = std::env::var_os("VOKRA_DFN3_GGUF")?;
    let data = std::env::var_os("VOKRA_DFN3_DATA")?;
    Some((PathBuf::from(gguf), PathBuf::from(data)))
}

fn read_f32(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(bytes.len() % 4, 0, "{path:?} not a raw f32 file");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn max_abs_delta(a: &[f32], b: &[f32], what: &str) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "{what}: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// De-interleave an upstream `[.., 2]` as_real dump into (re, im).
fn split_complex(x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let re = x.iter().step_by(2).copied().collect();
    let im = x.iter().skip(1).step_by(2).copied().collect();
    (re, im)
}

/// analyze.py `si_snr`: zero-mean scale-invariant SNR in dB (f64 math).
fn si_snr_db(est: &[f32], reference: &[f32]) -> f64 {
    assert_eq!(est.len(), reference.len());
    let n = est.len() as f64;
    let me: f64 = est.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mr: f64 = reference.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mut dot = 0.0f64;
    let mut rr = 0.0f64;
    for (&e, &r) in est.iter().zip(reference) {
        let (e, r) = (e as f64 - me, r as f64 - mr);
        dot += e * r;
        rr += r * r;
    }
    let alpha = dot / rr;
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (&e, &r) in est.iter().zip(reference) {
        let (e, r) = (e as f64 - me, r as f64 - mr);
        let s = alpha * r;
        sig += s * s;
        let d = e - s;
        err += d * d;
    }
    10.0 * (sig / err).log10()
}

#[test]
fn dfn3_real_weight_stage_and_output_parity() {
    let Some((gguf_path, data)) = env_paths() else {
        eprintln!("skipping: set VOKRA_DFN3_GGUF + VOKRA_DFN3_DATA to run the real-weight parity");
        return;
    };
    let gguf = GgufFile::parse(std::fs::read(&gguf_path).expect("read gguf")).expect("parse gguf");
    let model = DenoiseModel::from_gguf(&gguf).expect("bind real DFN3");

    let noisy = read_f32(&data.join("noisy_48k.f32"));
    let clean = read_f32(&data.join("clean_48k.f32"));
    let upstream = read_f32(&data.join("enhanced_upstream.f32"));
    assert_eq!(noisy.len(), 528_000);

    let (enhanced, taps) = model.enhance_with_taps(&noisy).expect("enhance");
    assert_eq!(enhanced.len(), noisy.len());
    let t = taps.frames;
    assert_eq!(t, 1102, "528000 + n_fft over hop 480");

    let taps_dir = data.join("taps");
    let rd = |name: &str| read_f32(&taps_dir.join(name));

    // ---- per-stage parity ----
    //
    // Every bound = the measured max |Δ| of the 2026-07-17 run (recorded per
    // stage below) × ~3-10 headroom for cross-machine libm / accumulation-
    // order variance. FP32 accumulation-order noise only — no behavioural
    // tolerance is hidden here (the honest-parity-atol discipline).
    let mut table: Vec<(&str, f32, f32)> = Vec::new();
    let mut check = |what: &'static str, got: &[f32], want: &[f32], bound: f32| {
        let d = max_abs_delta(got, want, what);
        table.push((what, d, bound));
        assert!(
            d <= bound,
            "{what}: max |Δ| {d:.3e} exceeds bound {bound:.1e}"
        );
    };

    // Frontend (measured: raw spec 2.24e-8, feat_erb 4.77e-7,
    // feat_spec 3.58e-7 / 2.38e-7).
    let (sp_re, sp_im) = split_complex(&rd("spec.f32"));
    check("spec.re", &taps.spec_re, &sp_re, 1e-6);
    check("spec.im", &taps.spec_im, &sp_im, 1e-6);
    check("feat_erb", &taps.feat_erb, &rd("feat_erb.f32"), 5e-6);
    let (fs_re, fs_im) = split_complex(&rd("feat_spec.f32"));
    check("feat_spec.re", &taps.feat_spec_re, &fs_re, 5e-6);
    check("feat_spec.im", &taps.feat_spec_im, &fs_im, 5e-6);
    // Encoder convs (measured: e0 1.44e-6, e1 1.01e-6, e2 5.35e-6,
    // e3 1.46e-5, c0 2.86e-6).
    check("enc.e0", &taps.e0, &rd("e0.f32"), 2e-5);
    check("enc.e1", &taps.e1, &rd("e1.f32"), 2e-5);
    check("enc.e2", &taps.e2, &rd("e2.f32"), 5e-5);
    check("enc.e3", &taps.e3, &rd("e3.f32"), 1e-4);
    check("enc.c0", &taps.c0, &rd("c0.f32"), 2e-5);
    // Embedding path (measured: cemb 6.68e-6, emb_in 1.46e-5, emb 3.04e-6,
    // lsnr 2.10e-5 dB on a ±50 dB scale).
    check("enc.cemb", &taps.cemb, &rd("cemb.f32"), 5e-5);
    check("enc.emb_in", &taps.emb_in, &rd("emb_in.f32"), 1e-4);
    check("enc.emb", &taps.emb, &rd("emb.f32"), 5e-5);
    check("enc.lsnr", &taps.lsnr, &rd("lsnr.f32"), 2e-4);
    // Decoders (measured: m 1.70e-6 on [0, 1]; df_gru_out 4.72e-6;
    // coefs 2.06e-6).
    check("erb_dec.m", &taps.m, &rd("m.f32"), 1e-5);
    check(
        "df_dec.gru_out",
        &taps.df_gru_out,
        &rd("df_gru_out.f32"),
        5e-5,
    );
    check("df_dec.coefs", &taps.coefs, &rd("coefs.f32"), 2e-5);
    // Final spectrum (measured: 5.96e-8 / 8.94e-8 — the mask + DF assembly
    // over near-bit-identical raw spectra).
    let (se_re, se_im) = split_complex(&rd("spec_e.f32"));
    check("spec_e.re", &taps.spec_e_re, &se_re, 1e-6);
    check("spec_e.im", &taps.spec_e_im, &se_im, 1e-6);
    // Enhanced waveform (measured 4.17e-7 vs taps/enhanced.f32 AND vs
    // enhanced_upstream.f32 — the two upstream dumps are bit-identical).
    check("enhanced (taps ref)", &enhanced, &rd("enhanced.f32"), 5e-6);
    check("enhanced (enhance() ref)", &enhanced, &upstream, 5e-6);

    // ---- quality: SI-SNR vs clean (upstream measured +14.768 dB) ----
    let snr_noisy = si_snr_db(&noisy, &clean);
    let snr_up = si_snr_db(&upstream, &clean);
    let snr_vokra = si_snr_db(&enhanced, &clean);
    eprintln!("== dfn3 real-weight parity (max |Δ| / bound) ==");
    for (what, d, bound) in &table {
        eprintln!("  {what:<24} {d:9.3e} / {bound:.1e}");
    }
    eprintln!("  SI-SNR dB: noisy {snr_noisy:.3} | upstream {snr_up:.3} | vokra {snr_vokra:.3}");
    assert!(
        (snr_noisy - 5.002).abs() < 0.01,
        "noisy baseline drifted: {snr_noisy}"
    );
    assert!(
        (snr_up - 14.768).abs() < 0.01,
        "upstream reference drifted: {snr_up}"
    );
    // The port must land within 0.01 dB of the upstream enhancement
    // (measured 2026-07-17: upstream 14.768398 dB, vokra 14.768399 dB —
    // gap 2.0e-7 dB at sample-level max |Δ| 4.17e-7).
    assert!(
        (snr_vokra - snr_up).abs() < 0.01,
        "vokra SI-SNR {snr_vokra:.3} deviates from upstream {snr_up:.3}"
    );
}
