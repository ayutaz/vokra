//! Real-weight Mimi PCM roundtrip (env-gated — CI-skipped without the
//! converted GGUF).
//!
//! Gate: `VOKRA_MIMI_GGUF` must point at a standalone Mimi GGUF produced
//! by `vokra-cli convert --model mimi` from the real kyutai checkpoint
//! (`kyutai/moshiko-pytorch-bf16
//! tokenizer-e351c8d8-checkpoint125.safetensors`, CC-BY 4.0 — NOTICE §5).
//! Unset → the test prints a skip note and passes (the M2 parity-test
//! gating pattern; never a fabricated pass).
//!
//! What it pins (the T29 chain the converter adapter + runtime binders
//! close):
//!
//! 1. `vokra.mimi.*` config chunk group reads back and validates.
//! 2. [`MimiEncoder`] / [`MimiNeuralDecoder`] bind the real weights
//!    (structural naming, split input projections present).
//! 3. PCM → codes at the 12.5 Hz token rate (11.04 s → 138 × 32 codes).
//! 4. codes → effective-table features → PCM at 24 kHz, finite, energy-
//!    carrying, length-exact.
//!
//! Numerical parity vs the upstream `moshi` implementation is measured by
//! the eval harness (campaign record), not asserted here — this test is
//! the *wiring* gate.

use std::f32::consts::PI;

use vokra_core::gguf::GgufFile;
use vokra_models::codec::MimiCodecGguf;
use vokra_models::mimi::{MimiEncoder, MimiNeuralConfig, MimiNeuralDecoder};
use vokra_ops::{MimiRvqAttrs, mimi_rvq_decode};

#[test]
fn real_mimi_pcm_roundtrip_via_gguf() {
    let Ok(path) = std::env::var("VOKRA_MIMI_GGUF") else {
        eprintln!("skip: VOKRA_MIMI_GGUF not set (real-weight gated test)");
        return;
    };
    let file = GgufFile::open(&path).expect("open GGUF");

    // 1. Config chunk group (converter-emitted) reads back and validates.
    let cfg = MimiNeuralConfig::from_gguf(&file).expect("vokra.mimi.* config chunk group");
    cfg.validate().expect("converter-emitted config validates");
    assert_eq!(cfg.sample_rate, 24_000);
    assert_eq!(cfg.frame_rate_mhz, 12_500, "12.5 Hz token rate");
    let hop = cfg.frame_hop_samples().expect("rate arithmetic closes");
    assert_eq!(hop, 1920, "24 kHz / 12.5 Hz = 1920 samples per frame");

    // 2. Real-weight binding.
    let enc = MimiEncoder::from_gguf(&file, &cfg).expect("encoder binds real weights");
    assert!(
        enc.has_split_projection(),
        "real checkpoint carries rvq_first + rvq_rest input projections"
    );
    let dec = MimiNeuralDecoder::from_gguf(&file, &cfg).expect("decoder binds real weights");
    assert_eq!(
        dec.expected_feature_dim(),
        cfg.seanet.dimension,
        "standalone GGUF decodes on the effective-table path (no feature_proj)"
    );
    let codec = MimiCodecGguf::from_gguf(&file).expect("effective codebook tables");

    // 3. Encode 11.04 s (138 frames) of deterministic band-limited audio.
    let n_frames = 138usize;
    let n = n_frames * hop; // 264 960 samples = 11.04 s at 24 kHz
    let pcm: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / 24_000.0;
            0.4 * (2.0 * PI * 220.0 * t).sin() + 0.2 * (2.0 * PI * 523.25 * t).sin()
        })
        .collect();
    let codes = enc.encode_all(&pcm).expect("encode");
    assert_eq!(
        codes.len(),
        n_frames * cfg.quantizer.n_q,
        "12.5 Hz code count for 11.04 s"
    );
    assert!(codes.iter().all(|&c| (c as usize) < cfg.quantizer.bins));
    let first = &codes[..cfg.quantizer.n_q];
    assert!(
        codes.chunks(cfg.quantizer.n_q).any(|f| f != first),
        "codes must respond to signal content"
    );

    // 4. Decode back to PCM through the shared RVQ decode op.
    let attrs = MimiRvqAttrs {
        n_codebooks: cfg.quantizer.n_q,
        codebook_size: cfg.quantizer.bins,
        d_model: cfg.seanet.dimension,
    };
    let features =
        mimi_rvq_decode(&codes, n_frames, &codec.tables, &attrs).expect("codes → features");
    let out = dec.decode_all(&features).expect("decode");
    assert_eq!(out.len(), n, "PCM length: 11.04 s at 24 kHz");
    assert!(out.iter().all(|v| v.is_finite()));
    let rms = (out
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        / out.len() as f64)
        .sqrt();
    assert!(
        rms > 1e-4,
        "decoded PCM must carry signal energy, rms = {rms}"
    );
    assert!(
        out.iter().all(|v| v.abs() < 4.0),
        "decoded PCM amplitude sanity"
    );
    eprintln!(
        "real mimi roundtrip: {} frames -> {} codes -> {} samples (rms {rms:.4})",
        n_frames,
        codes.len(),
        out.len()
    );
}
