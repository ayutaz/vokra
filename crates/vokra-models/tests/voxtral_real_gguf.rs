//! Real-checkpoint Voxtral GGUF load + text-only greedy-step smoke test
//! (2026-07-16 P1 fix acceptance: GQA decoupled `head_dim` + untied
//! `lm_head`).
//!
//! # Gate posture
//!
//! Gated on `VOKRA_VOXTRAL_GGUF` (same env var `parity_voxtral.rs` uses):
//! unset → clean skip with a diagnostic, never a fabricated pass. The GGUF
//! is opened through the first-party mmap loader (`vokra_mmap::open_gguf`)
//! so the 9+ GB file is paged, not heap-copied; the decoded f32 decoder
//! weights still need ~16 GB of (compressed/swapped) memory on the mini —
//! run on a machine with real swap headroom.
//!
//! # What this proves (and what it does not)
//!
//! - `VoxtralConfig::from_gguf` reads the real hparams (explicit
//!   `head_dim` decoupled from `hidden_dim / n_head_q` on the mini);
//! - `TextDecoder::load` binds the real GQA-shaped projections and the
//!   untied `lm_head`;
//! - a text-only greedy step from BOS produces finite logits, and one
//!   incremental step through the KV cache does too (the top-5 token ids
//!   are printed for the eval report).
//!
//! It does NOT claim transcription: the 32-layer audio-encoder transformer
//! is still the explicit `UnsupportedOp` stub (T19+ deliberate deferral),
//! so no audio conditioning happens here.

use std::path::PathBuf;

use vokra_core::BackendKind;
use vokra_models::voxtral::{TextDecoder, TextDecoderSession, VoxtralConfig};

fn require_voxtral_gguf() -> Option<PathBuf> {
    std::env::var_os("VOKRA_VOXTRAL_GGUF").map(PathBuf::from)
}

/// BOS id override for non-mini checkpoints; the shipping mini's
/// `generation_config.json` says `bos_token_id = 1`.
fn bos_id() -> u32 {
    std::env::var("VOKRA_VOXTRAL_BOS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

fn top_k(row: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&a, &b| {
        row[b]
            .partial_cmp(&row[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.into_iter().take(k).map(|i| (i, row[i])).collect()
}

#[test]
fn real_gguf_loads_gqa_decoder_and_greedy_step_is_finite() {
    let Some(path) = require_voxtral_gguf() else {
        eprintln!(
            "[voxtral_real_gguf] SKIP: set VOKRA_VOXTRAL_GGUF to a converted Voxtral GGUF \
             (e.g. the raw-BF16-shards conversion) to enable this test."
        );
        return;
    };

    // mmap open — the tensor payloads are paged on demand.
    let file = vokra_mmap::open_gguf(&path).expect("mmap-parse the Voxtral GGUF");

    let cfg = VoxtralConfig::from_gguf(&file).expect("read vokra.voxtral.* hparams");
    eprintln!(
        "[voxtral_real_gguf] hparams: n_layer={} hidden={} n_head_q={} n_head_kv={} \
         head_dim={} (q_hidden={} kv_hidden={}) ffn={} vocab={} n_ctx={} rope_base={} eps={}",
        cfg.text.n_layer,
        cfg.text.hidden_dim,
        cfg.text.n_head_q,
        cfg.text.n_head_kv,
        cfg.text.head_dim(),
        cfg.text.q_hidden(),
        cfg.text.kv_hidden(),
        cfg.text.ffn_dim,
        cfg.text.vocab_size,
        cfg.text.n_ctx,
        cfg.text.rope_base,
        cfg.text.rms_norm_eps,
    );
    assert!(cfg.text.head_dim() > 0, "head_dim must resolve");
    assert!(cfg.text.n_head_q >= cfg.text.n_head_kv, "GQA split");

    // On the shipping mini the head width is decoupled: 32 x 128 = 4096
    // while hidden_dim = 3072 — the exact shape class the pre-fix loader
    // rejected ("shape [4096, 3072] != expected [3072, 3072]").
    let name = file
        .get("vokra.model.name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_owned();
    if name == "voxtral-mini-3b" {
        assert_eq!(cfg.text.n_layer, 30);
        assert_eq!(cfg.text.hidden_dim, 3072);
        assert_eq!(cfg.text.n_head_q, 32);
        assert_eq!(cfg.text.n_head_kv, 8);
        assert_eq!(cfg.text.head_dim(), 128);
        assert_eq!(cfg.text.q_hidden(), 4096);
        assert_eq!(cfg.text.kv_hidden(), 1024);
        assert_eq!(cfg.text.vocab_size, 131_072);
    }

    let decoder = TextDecoder::load(&file, &cfg).expect("bind the real GQA text decoder");
    eprintln!(
        "[voxtral_real_gguf] loaded: n_layer={} prefix={:?} untied_lm_head={}",
        decoder.n_layer(),
        decoder.source_prefix(),
        decoder.has_untied_lm_head(),
    );
    if name == "voxtral-mini-3b" {
        assert!(
            decoder.has_untied_lm_head(),
            "the shipping mini carries an untied language_model.lm_head.weight \
             (byte-compared != embed_tokens in the 2026-07-16 eval)"
        );
    }

    // Text-only greedy step from BOS (no audio conditioning — the audio
    // encoder transformer is the T19+ stub).
    let bos = bos_id();
    let mut session = TextDecoderSession::new(&cfg, &decoder, BackendKind::Cpu)
        .expect("construct CPU decode session");
    session.step_into(&[bos]).expect("BOS step");
    let row = session.last_logits_row();
    assert_eq!(row.len(), cfg.text.vocab_size);
    let finite = row.iter().all(|v| v.is_finite());
    assert!(finite, "BOS-step logits must be finite");
    let top5 = top_k(row, 5);
    eprintln!("[voxtral_real_gguf] BOS={bos} step top-5 (id, logit): {top5:?}");

    // One incremental step through the KV cache on the real shapes.
    let next = top5[0].0 as u32;
    session.step_into(&[next]).expect("incremental step");
    let row2 = session.last_logits_row();
    assert!(row2.iter().all(|v| v.is_finite()), "step-2 logits finite");
    let top5_2 = top_k(row2, 5);
    eprintln!("[voxtral_real_gguf] step-2 (after id {next}) top-5: {top5_2:?}");
}
