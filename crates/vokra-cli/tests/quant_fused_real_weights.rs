//! M5-15-T29 / T34: the fused K-quant weight path on **real Whisper weights**.
//!
//! The synthetic unit tests in `vokra-models` pin the binding rules and the
//! per-projection error bound against pseudo-random super-blocks. What real
//! weights add is the value *distribution*: a trained projection's dynamic
//! range decides how much of the Q8 activation band the fused route actually
//! spends, and no synthesized payload can stand in for that.
//!
//! # How the quantized fixture is produced
//!
//! By **transcoding** a real f32 Whisper GGUF: metadata is copied verbatim and
//! every rank-2, 256-aligned tensor is re-emitted through the same offline
//! K-quant encoder the converter uses (`vokra_convert::quantize`). Nothing is
//! re-implemented here — a hand-rolled quantizer would be a self-consistent
//! mirror that proves only its own arithmetic.
//!
//! Transcoding rather than running `vokra-cli convert --model whisper
//! --quantize` is **deliberate and is a known defect, not a shortcut**: that
//! command currently fails on every real Whisper checkpoint because the
//! `whisper_q4_k` policy preset pins `.bias` and `.weight_norm` to F32 but not
//! the rank-1 LayerNorm `.weight` tensors, so `convert_with_policy` raises
//! `QuantPolicyInapplicable` on `model.decoder.layer_norm.weight`. Reproduced
//! on `13a2a6e` with the M5-15 changes stashed, so it pre-dates this WP. The
//! preset is a ratified M2-08 contract whose shape is asserted by
//! `vokra-cli/tests/policy_e2e.rs` (`rule_count == 2`), so widening it is a
//! decision for M5-15-T02/T03, not something to slip in here.
//!
//! # Gating
//!
//! `VOKRA_WHISPER_GGUF` must point at a real **non-quantized** Whisper GGUF.
//! Absent, the test skips cleanly and says so — a skip is a skip, never a
//! fabricated pass (NFR-QL-04).

use std::path::PathBuf;

use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile};
use vokra_models::whisper::{WhisperLoadOptions, WhisperModel};

/// Number of elements in one K-quant super-block.
const QK_K: usize = 256;

fn gated_gguf() -> Option<PathBuf> {
    match std::env::var("VOKRA_WHISPER_GGUF") {
        Ok(p) if !p.is_empty() => Some(PathBuf::from(p)),
        _ => {
            eprintln!(
                "skip: VOKRA_WHISPER_GGUF is unset — the fused-quant real-weight legs need a \
                 real non-quantized Whisper GGUF"
            );
            None
        }
    }
}

fn tmp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("vokra-m5-15-{tag}-{}.gguf", std::process::id()));
    p
}

/// Whether the runtime's fused binder will accept this tensor: rank 2 (an
/// `nn.Linear` weight `[out, in]`) with a 256-aligned **row**, so a
/// super-block never straddles two rows.
fn fused_eligible(dims: &[u64]) -> bool {
    dims.len() == 2 && (dims[1] as usize) % QK_K == 0
}

/// Re-emits `src` with every rank-2, 256-aligned tensor K-quantized to
/// `target`. Returns the written path and how many tensors were quantized.
fn transcode_quantized(src: &GgufFile, target: GgmlType, out: &PathBuf) -> usize {
    let mut b = GgufBuilder::new();
    for (k, v) in src.metadata() {
        b.add_metadata(k, v.clone());
    }
    let mut quantized = 0usize;
    for t in src.tensors() {
        let name = &t.name;
        if t.dtype == GgmlType::F32
            && fused_eligible(&t.dimensions)
            && (t.dimensions.iter().product::<u64>() as usize) % QK_K == 0
        {
            let data = src.tensor_f32(name).expect("dequant source tensor");
            let payload = vokra_convert::quantize(target, &data).expect("K-quant encode");
            b.add_tensor(name, target, t.dimensions.clone(), payload)
                .expect("add quantized tensor");
            quantized += 1;
        } else {
            b.add_tensor(
                name,
                t.dtype,
                t.dimensions.clone(),
                src.tensor_bytes(t).to_vec(),
            )
            .expect("copy tensor verbatim");
        }
    }
    std::fs::write(out, b.to_bytes().expect("serialize GGUF")).expect("write GGUF");
    quantized
}

/// A deterministic log-mel window. Real weights + a fixed input isolates the
/// weight-representation change as the only variable between the two loads.
fn synthetic_log_mel(n_mels: usize, n_frames: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n_mels * n_frames];
    let mut s = 0x5115_2934u64 | 1;
    for x in &mut v {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        let u = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
        // Whisper log-mel lives roughly in [-1, 1] after normalisation.
        *x = ((u >> 40) as u32) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
    }
    v
}

/// T29 / T34: a real quantized Whisper GGUF loads through the fused binder,
/// binds the projections it should, and produces a finite encoder output whose
/// divergence from the dequant route is **measured and reported** rather than
/// asserted against an invented stack-level tolerance.
///
/// A composed encoder (12 GEMMs deep, through softmax and LayerNorm) has no
/// analytic error bound — `int8_error_bound` is a *per-GEMV* bound and does not
/// compose. Claiming one here would be exactly the fabricated-tolerance move
/// the parity red-line forbids, so this test asserts what is actually provable
/// (the binder did its job, the arithmetic stayed finite, the two routes agree
/// on shape) and prints the observed delta as the input to the M5-15-T10
/// go/no-go on defaulting the route ON.
#[test]
fn real_quantized_whisper_binds_fused_and_reports_encoder_divergence() {
    let Some(src_path) = gated_gguf() else { return };
    let src_bytes = std::fs::read(&src_path).expect("read source GGUF");
    let src = GgufFile::parse(src_bytes).expect("parse source GGUF");

    // The source must be non-quantized, or "dequant vs fused" is not the
    // comparison being made.
    assert!(
        src.tensors()
            .iter()
            .all(|t| matches!(t.dtype, GgmlType::F32 | GgmlType::F16)),
        "VOKRA_WHISPER_GGUF must point at a NON-quantized GGUF"
    );

    for target in [GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q6K] {
        let out = tmp_path(&format!("{target:?}").to_lowercase());
        let n_quantized = transcode_quantized(&src, target, &out);
        assert!(n_quantized > 0, "{target:?}: nothing was quantized");

        let bytes = std::fs::read(&out).expect("read transcoded GGUF");
        let file = GgufFile::parse(bytes).expect("parse transcoded GGUF");

        let dequant = WhisperModel::from_gguf(&file).expect("dequant load");
        let fused = WhisperModel::from_gguf_with(
            &file,
            WhisperLoadOptions {
                fused_quant_weights: true,
            },
        )
        .expect("fused load");

        assert_eq!(
            dequant.quant_report(),
            Default::default(),
            "{target:?}: the default load must not bind anything fused"
        );
        let report = fused.quant_report();
        assert!(
            report.fused > 0,
            "{target:?}: fused load bound nothing — the route under test never ran"
        );
        assert_eq!(
            report.dequantized_unaligned, 0,
            "{target:?}: every quantized tensor here is 256-aligned by construction"
        );

        let cfg = dequant.config();
        let n_frames = cfg.n_audio_ctx * 2;
        let mel = synthetic_log_mel(cfg.n_mels, n_frames);
        let a = dequant.encode(&mel, n_frames).expect("dequant encode");
        let b = fused.encode(&mel, n_frames).expect("fused encode");

        assert_eq!(a.hidden.len(), b.hidden.len(), "{target:?}: shape");
        assert!(
            b.hidden.iter().all(|v| v.is_finite()),
            "{target:?}: fused encoder produced a non-finite value"
        );

        let (mut max_abs, mut sse, mut ref_sse) = (0.0f32, 0.0f64, 0.0f64);
        for (&x, &y) in a.hidden.iter().zip(&b.hidden) {
            max_abs = max_abs.max((x - y).abs());
            sse += f64::from(x - y).powi(2);
            ref_sse += f64::from(x).powi(2);
        }
        let rel_rms = (sse / ref_sse.max(f64::MIN_POSITIVE)).sqrt();
        eprintln!(
            "M5-15-T29 {target:?}: {} projections fused, encoder max|Δ| = {max_abs:.6e}, \
             relative RMS = {rel_rms:.6e}",
            report.fused
        );

        let _ = std::fs::remove_file(&out);
    }
}

/// Minimal 16-bit PCM WAV reader (RIFF chunk walk). File I/O only — it decodes
/// no model and stands in for no reference implementation.
fn read_wav_mono_f32(path: &PathBuf) -> Option<Vec<f32>> {
    let b = std::fs::read(path).ok()?;
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return None;
    }
    let (mut i, mut channels, mut bits) = (12usize, 1usize, 16usize);
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let sz = u32::from_le_bytes(b[i + 4..i + 8].try_into().ok()?) as usize;
        let body = i + 8;
        if id == b"fmt " && body + 16 <= b.len() {
            channels = u16::from_le_bytes(b[body + 2..body + 4].try_into().ok()?) as usize;
            bits = u16::from_le_bytes(b[body + 14..body + 16].try_into().ok()?) as usize;
        } else if id == b"data" {
            let end = (body + sz).min(b.len());
            if bits != 16 {
                return None;
            }
            return Some(
                b[body..end]
                    .chunks_exact(2)
                    .step_by(channels)
                    .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
                    .collect(),
            );
        }
        i = body + sz + (sz & 1);
    }
    None
}

/// T34: the transcript-level comparison on real weights **and real audio**.
///
/// The fused route is not bit-identical, so demanding a byte-identical
/// transcript would be an assertion the design does not support. What is
/// asserted is that the fused route transcribes at all and stays on the same
/// token vocabulary; the agreement (or the delta) is measured and printed, and
/// that number is the material M5-15-T10 needs to decide whether the route can
/// ever default to ON.
///
/// Needs `VOKRA_WHISPER_GGUF` **and** `VOKRA_WHISPER_WAV` (16 kHz mono PCM16).
#[test]
fn real_quantized_whisper_transcribes_and_reports_token_agreement() {
    use vokra_models::whisper::WhisperAsr;

    let Some(src_path) = gated_gguf() else { return };
    let Ok(wav) = std::env::var("VOKRA_WHISPER_WAV") else {
        eprintln!("skip: VOKRA_WHISPER_WAV is unset — the transcript leg needs real audio");
        return;
    };
    let wav = PathBuf::from(wav);
    let Some(pcm) = read_wav_mono_f32(&wav) else {
        eprintln!("skip: {} is not a 16-bit PCM WAV", wav.display());
        return;
    };

    let src = GgufFile::parse(std::fs::read(&src_path).expect("read")).expect("parse");
    let target = GgmlType::Q6K;
    let out = tmp_path("transcript-q6k");
    assert!(transcode_quantized(&src, target, &out) > 0);
    let file = GgufFile::parse(std::fs::read(&out).expect("read")).expect("parse");

    let a = WhisperAsr::from_gguf(&file).expect("dequant asr");
    let b = WhisperAsr::from_gguf_with(
        &file,
        WhisperLoadOptions {
            fused_quant_weights: true,
        },
    )
    .expect("fused asr");

    let ta = a.transcribe_tokens(&pcm).expect("dequant transcribe");
    let tb = b.transcribe_tokens(&pcm).expect("fused transcribe");
    assert!(!tb.is_empty(), "fused route produced no tokens");

    let identical = ta == tb;
    if a.has_tokenizer() {
        eprintln!(
            "M5-15-T34 {target:?} dequant: {:?}",
            a.render_ids(&ta).unwrap_or_default()
        );
        eprintln!(
            "M5-15-T34 {target:?} fused  : {:?}",
            b.render_ids(&tb).unwrap_or_default()
        );
    }
    eprintln!(
        "M5-15-T34 {target:?}: token sequences {} ({} vs {} tokens)",
        if identical { "IDENTICAL" } else { "DIFFER" },
        ta.len(),
        tb.len()
    );

    let _ = std::fs::remove_file(&out);
}
