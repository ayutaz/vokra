//! CosyVoice2 hyper-parameters read from the `vokra.cosyvoice2.*` metadata
//! chunk group (M3-09-T04 chunk design).
//!
//! Every runtime parameter (sample rate, vocab size, LLM backbone shape,
//! flow / mimi / streaming hparams) is read from the GGUF metadata the
//! converter wrote — never hard-coded, never given a silent default. A
//! missing key raises [`VokraError::InvalidArgument`] with the offending
//! key name (FR-EX-08). The upstream CosyVoice2 is Apache 2.0 code + weight
//! (docs/license-audit.md), so the resulting GGUF is公式 zoo eligible; a
//! non-commercial provenance tag is rejected by the shared
//! [`vokra_core::check_weight_license`] gate (M2-13), not here.
//!
//! # Zero-placeholder hparams (scaffold policy, T02 / T04)
//!
//! The M3-09 T02 upstream inspection is still open (the checkpoint is not
//! bound to this scaffold), so the numeric hparams (`n_layer`, `n_head`,
//! `hidden_dim`, `ffn_dim`, `flow.nfe`, mimi shapes …) are accepted with
//! a `0` placeholder — the pattern Whisper and Kokoro established for
//! shape-driven fields whose upstream values are not yet pinned. The
//! runtime forward-path lands with T07/T08/T10/T13 and will enforce
//! `!= 0` at that point (per-field, not here) so a `0`-placeholder GGUF
//! fails loudly at the first missing shape rather than silently in the
//! middle of a forward.

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::{Result, VokraError};

// --- `vokra.cosyvoice2.*` metadata key names --------------------------------
//
// Kept as constants inside this module (mirror the piper-plus / kokoro /
// silero patterns): CosyVoice2-specific keys live with the CosyVoice2 model,
// not in `vokra-core::gguf::chunks`.

pub(crate) const KEY_SAMPLE_RATE: &str = "vokra.cosyvoice2.sample_rate";
pub(crate) const KEY_VOCAB_SIZE: &str = "vokra.cosyvoice2.arch.vocab_size";
pub(crate) const KEY_HIDDEN_DIM: &str = "vokra.cosyvoice2.arch.hidden_dim";
pub(crate) const KEY_N_LAYER: &str = "vokra.cosyvoice2.arch.n_layer";
pub(crate) const KEY_N_HEAD: &str = "vokra.cosyvoice2.arch.n_head";
pub(crate) const KEY_FFN_DIM: &str = "vokra.cosyvoice2.arch.ffn_dim";
pub(crate) const KEY_FLOW_NFE: &str = "vokra.cosyvoice2.flow.nfe";
pub(crate) const KEY_FLOW_SCHEDULE: &str = "vokra.cosyvoice2.flow.schedule";
pub(crate) const KEY_MIMI_N_CODEBOOKS: &str = "vokra.cosyvoice2.mimi.n_codebooks";
pub(crate) const KEY_MIMI_CODEBOOK_SIZE: &str = "vokra.cosyvoice2.mimi.codebook_size";
pub(crate) const KEY_MIMI_D_MODEL: &str = "vokra.cosyvoice2.mimi.d_model";
pub(crate) const KEY_STREAMING_CHUNK_SIZE: &str = "vokra.cosyvoice2.streaming.chunk_size";
pub(crate) const KEY_STREAMING_CHUNK_HOP: &str = "vokra.cosyvoice2.streaming.chunk_hop";

/// Resolved runtime configuration read from a CosyVoice2 GGUF.
///
/// Grows under `#[non_exhaustive]` so the follow-on tickets (T04 extension
/// for tokenizer / voicepack keys, T05 `vokra.frontend.*` bit-exact chunk
/// binding) can add fields without breaking downstream matches.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CosyVoice2Config {
    /// Output PCM sample rate, Hz. The upstream CosyVoice2 model card fixes
    /// this at 24 kHz (Mimi codec native rate); read from the GGUF to keep
    /// the runtime data-driven.
    pub sample_rate: u32,
    /// Text tokenizer vocabulary size. Runtime uses this to size the LLM
    /// backbone's input embedding table (T07 lookup).
    pub vocab_size: u32,
    /// LLM backbone hidden dimension.
    pub hidden_dim: u32,
    /// LLM backbone transformer block count.
    pub n_layer: u32,
    /// LLM backbone attention head count.
    pub n_head: u32,
    /// LLM backbone FFN inner dimension.
    pub ffn_dim: u32,
    /// Flow Matching sampler default NFE (number of function evaluations
    /// per chunk). Runtime-overridable per invocation (FR-EX-10).
    pub flow_nfe: u32,
    /// Flow Matching schedule tag (`linear` / `sway` / `epss` — matches
    /// `vokra_ops::Schedule` variants; the mapping lives in
    /// `flow_matching::FlowMatchingRuntimeParams::schedule_from_tag`).
    pub flow_schedule_tag: String,
    /// Mimi codec: number of RVQ codebooks (base + residuals).
    pub mimi_n_codebooks: u32,
    /// Mimi codec: entries per codebook.
    pub mimi_codebook_size: u32,
    /// Mimi codec: feature dim per codebook entry.
    pub mimi_d_model: u32,
    /// Chunk-aware streaming chunk size (frames per chunk boundary).
    pub streaming_chunk_size: u32,
    /// Chunk-aware streaming chunk hop (frames between chunk starts).
    pub streaming_chunk_hop: u32,
}

impl CosyVoice2Config {
    /// Reads the configuration from a loaded CosyVoice2 GGUF.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if any `vokra.cosyvoice2.*` key is
    /// missing or of the wrong type (FR-EX-08 — never a silent default).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        Ok(Self {
            sample_rate: u32v(file, KEY_SAMPLE_RATE)?,
            vocab_size: u32v(file, KEY_VOCAB_SIZE)?,
            hidden_dim: u32v(file, KEY_HIDDEN_DIM)?,
            n_layer: u32v(file, KEY_N_LAYER)?,
            n_head: u32v(file, KEY_N_HEAD)?,
            ffn_dim: u32v(file, KEY_FFN_DIM)?,
            flow_nfe: u32v(file, KEY_FLOW_NFE)?,
            flow_schedule_tag: strv(file, KEY_FLOW_SCHEDULE)?,
            mimi_n_codebooks: u32v(file, KEY_MIMI_N_CODEBOOKS)?,
            mimi_codebook_size: u32v(file, KEY_MIMI_CODEBOOK_SIZE)?,
            mimi_d_model: u32v(file, KEY_MIMI_D_MODEL)?,
            streaming_chunk_size: u32v(file, KEY_STREAMING_CHUNK_SIZE)?,
            streaming_chunk_hop: u32v(file, KEY_STREAMING_CHUNK_HOP)?,
        })
    }
}

fn u32v(file: &GgufFile, key: &str) -> Result<u32> {
    match file.get(key) {
        Some(GgufMetadataValue::U32(v)) => Ok(*v),
        Some(_) => Err(VokraError::InvalidArgument(format!(
            "cosyvoice2 config: `{key}` is not a UINT32"
        ))),
        None => Err(VokraError::InvalidArgument(format!(
            "cosyvoice2 config: missing `{key}` (expected UINT32)"
        ))),
    }
}

fn strv(file: &GgufFile, key: &str) -> Result<String> {
    match file.get(key) {
        Some(GgufMetadataValue::String(s)) => Ok(s.clone()),
        Some(_) => Err(VokraError::InvalidArgument(format!(
            "cosyvoice2 config: `{key}` is not a STRING"
        ))),
        None => Err(VokraError::InvalidArgument(format!(
            "cosyvoice2 config: missing `{key}` (expected STRING)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufBuilder;
    use vokra_core::gguf::chunks::KEY_MODEL_ARCH;

    fn seed(b: &mut GgufBuilder) {
        b.add_string(KEY_MODEL_ARCH, "cosyvoice2");
        b.add_u32(KEY_SAMPLE_RATE, 24_000);
        b.add_u32(KEY_VOCAB_SIZE, 65_536);
        b.add_u32(KEY_HIDDEN_DIM, 1024);
        b.add_u32(KEY_N_LAYER, 24);
        b.add_u32(KEY_N_HEAD, 16);
        b.add_u32(KEY_FFN_DIM, 4096);
        b.add_u32(KEY_FLOW_NFE, 10);
        b.add_string(KEY_FLOW_SCHEDULE, "linear");
        b.add_u32(KEY_MIMI_N_CODEBOOKS, 8);
        b.add_u32(KEY_MIMI_CODEBOOK_SIZE, 2048);
        b.add_u32(KEY_MIMI_D_MODEL, 512);
        b.add_u32(KEY_STREAMING_CHUNK_SIZE, 25);
        b.add_u32(KEY_STREAMING_CHUNK_HOP, 25);
    }

    #[test]
    fn happy_path_reads_every_field() {
        // The seed values above are illustrative (T02 upstream inspection
        // is still open), not authoritative — this test only checks that
        // every key is read back verbatim.
        let mut b = GgufBuilder::new();
        seed(&mut b);
        let bytes = b.to_bytes().expect("serialize");
        let file = GgufFile::parse(bytes).expect("parse");
        let cfg = CosyVoice2Config::from_gguf(&file).expect("read");
        assert_eq!(cfg.sample_rate, 24_000);
        assert_eq!(cfg.vocab_size, 65_536);
        assert_eq!(cfg.hidden_dim, 1024);
        assert_eq!(cfg.n_layer, 24);
        assert_eq!(cfg.n_head, 16);
        assert_eq!(cfg.ffn_dim, 4096);
        assert_eq!(cfg.flow_nfe, 10);
        assert_eq!(cfg.flow_schedule_tag, "linear");
        assert_eq!(cfg.mimi_n_codebooks, 8);
        assert_eq!(cfg.mimi_codebook_size, 2048);
        assert_eq!(cfg.mimi_d_model, 512);
        assert_eq!(cfg.streaming_chunk_size, 25);
        assert_eq!(cfg.streaming_chunk_hop, 25);
    }

    #[test]
    fn missing_key_fails_loudly() {
        // Drop KEY_HIDDEN_DIM and expect the offending key name in the
        // error — that is what makes FR-EX-08 an actionable failure.
        let mut b = GgufBuilder::new();
        seed(&mut b);
        // Overwrite the key with a wrong-type value (string instead of
        // u32) so the builder does not accidentally deduplicate; the
        // reader treats "wrong type" and "missing" identically at the
        // API level (both produce InvalidArgument naming the key).
        b.add_metadata(
            KEY_HIDDEN_DIM,
            GgufMetadataValue::String("not-a-u32".to_owned()),
        );
        let bytes = b.to_bytes().expect("serialize");
        let file = GgufFile::parse(bytes).expect("parse");
        let err = CosyVoice2Config::from_gguf(&file).expect_err("wrong type must fail");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(msg.contains(KEY_HIDDEN_DIM), "unexpected: {msg}")
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }
}
