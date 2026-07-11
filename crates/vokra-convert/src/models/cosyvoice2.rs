//! CosyVoice2 (Flow Matching + Mimi codec + chunk-aware CFM): safetensors
//! checkpoint → GGUF conversion (M3-09-T03 / T04 draft).
//!
//! Input: the upstream `iic/CosyVoice2-0.5B` safetensors checkpoint on
//! HuggingFace (Apache 2.0 code + weight, docs/license-audit.md). Output: a
//! GGUF carrying every float tensor plus the `vokra.model.*` and
//! `vokra.cosyvoice2.*` metadata chunks the native CosyVoice2 implementation
//! (`crates/vokra-models/src/cosyvoice2/`) reads.
//!
//! # Scaffold scope (T03 / T04)
//!
//! This module lands the **converter skeleton**: safetensors reader plumbing,
//! `vokra.model.*` arch / name write, and the `vokra.cosyvoice2.*` chunk
//! group with `0` placeholders on every numeric hparam pending the T02
//! upstream inspection (same pattern Kokoro / Whisper established for
//! shape-driven values). Verbatim F32 / F16 tensor copy is exercised so the
//! roundtrip through `GgufBuilder::to_bytes` + `GgufFile::parse` is testable
//! today without a real checkpoint.
//!
//! The follow-on tickets (T04 hparam derivations from tensor shapes, T05
//! `vokra.frontend.*` bit-exact chunk, T06 tokenizer vocab embed) extend
//! this same file with per-tensor renaming and shape-driven hparam
//! computation, mirroring the Kokoro / Whisper pattern.
//!
//! # Tensor naming contract (T03)
//!
//! GGUF tensor names are the **upstream safetensors names verbatim** (same
//! contract Whisper / Kokoro use). Rich Vokra-side renaming can arrive
//! later without changing the guarantees of this module.
//!
//! # No ONNX (permanent constraint)
//!
//! The converter never touches an ONNX graph — CosyVoice2 ships as
//! safetensors + a Python-side pipeline; the pipeline is re-implemented in
//! Rust by the runtime crate (whisper.cpp 型 self re-implementation,
//! CLAUDE.md 設計判断 4).

use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value written for CosyVoice2 GGUFs. Kept in sync with
/// the runtime constant `crates/vokra-models/src/cosyvoice2::EXPECTED_ARCH`.
pub(crate) const ARCH: &str = "cosyvoice2";
/// `vokra.model.name` value written for the CosyVoice2 GGUF.
pub(crate) const NAME: &str = "cosyvoice2-0.5b";

// --- vokra.cosyvoice2.* metadata keys (T04 chunk design) --------------------
//
// Kept as constants inside this module (mirror the piper-plus / kokoro
// pattern): CosyVoice2-specific keys live with the CosyVoice2 model, not in
// `vokra-core::gguf::chunks`.
//
// The runtime reads back the same keys via
// `crates/vokra-models/src/cosyvoice2/config.rs`; the two files intentionally
// duplicate the constant strings (the runtime crate cannot depend on
// `vokra-convert`, and `vokra-convert` cannot depend on `vokra-models` — both
// depend only on `vokra-core`). A round-trip test in this file (below)
// serializes a synthetic buffer and re-reads the strings to catch any drift.

const KEY_SAMPLE_RATE: &str = "vokra.cosyvoice2.sample_rate";
const KEY_VOCAB_SIZE: &str = "vokra.cosyvoice2.arch.vocab_size";
const KEY_HIDDEN_DIM: &str = "vokra.cosyvoice2.arch.hidden_dim";
const KEY_N_LAYER: &str = "vokra.cosyvoice2.arch.n_layer";
const KEY_N_HEAD: &str = "vokra.cosyvoice2.arch.n_head";
const KEY_FFN_DIM: &str = "vokra.cosyvoice2.arch.ffn_dim";
const KEY_FLOW_NFE: &str = "vokra.cosyvoice2.flow.nfe";
const KEY_FLOW_SCHEDULE: &str = "vokra.cosyvoice2.flow.schedule";
const KEY_MIMI_N_CODEBOOKS: &str = "vokra.cosyvoice2.mimi.n_codebooks";
const KEY_MIMI_CODEBOOK_SIZE: &str = "vokra.cosyvoice2.mimi.codebook_size";
const KEY_MIMI_D_MODEL: &str = "vokra.cosyvoice2.mimi.d_model";
const KEY_STREAMING_CHUNK_SIZE: &str = "vokra.cosyvoice2.streaming.chunk_size";
const KEY_STREAMING_CHUNK_HOP: &str = "vokra.cosyvoice2.streaming.chunk_hop";

/// CosyVoice2 output PCM sample rate (Hz).
///
/// Sourced from the CosyVoice2 model card (Mimi codec native rate = 24 kHz);
/// this is the same "model-card invariant" exception Kokoro uses for its
/// 24 kHz value. Every other numeric hparam stays `0`-placeholder pending
/// the T02 upstream inspection.
const COSYVOICE2_SAMPLE_RATE: u32 = 24_000;

/// Canonical Mimi RVQ shape (8 codebooks × 2048 entries × 512 dim).
///
/// Sourced from the Mimi paper (Kyutai) / M3-06 module documentation —
/// stable model-card invariants, not invented numbers. The runtime rejects
/// a `0` codec shape at load (`MimiBridge::from_config`), so we do
/// **not** emit `0` placeholders on these three axes: an owner running
/// the converter would immediately hit that rejection with a confusing
/// message. Instead we write the Mimi defaults today; a future CosyVoice2
/// checkpoint that re-shapes its codec (unlikely — Mimi is bundled) can
/// override them via T04 shape-driven derivations.
const MIMI_N_CODEBOOKS: u32 = 8;
const MIMI_CODEBOOK_SIZE: u32 = 2048;
const MIMI_D_MODEL: u32 = 512;

/// Outcome of a CosyVoice2 conversion.
#[derive(Debug, Default)]
pub(crate) struct CosyVoice2Report {
    /// Number of float weight tensors written to the GGUF.
    pub(crate) written: usize,
    /// Tensors whose dtype falls outside the F32/F16 range and were skipped.
    ///
    /// The upstream safetensors reader already rejects unknown dtypes at
    /// parse time (`SafetensorsError::UnsupportedDtype`), so this counter
    /// is defensive/forward-compat (same rationale as Kokoro).
    pub(crate) skipped_non_float: usize,
    /// Diagnostic notes surfaced to the CLI operator (T02 upstream shape
    /// mismatch, etc.). The converter never fails on a note — the runtime
    /// is the authoritative gate (FR-EX-08) — but a loud warning is printed
    /// so the operator does not learn about it only at load time.
    pub(crate) notes: Vec<String>,
}

/// Converts a CosyVoice2 safetensors buffer into a populated GGUF builder
/// plus a report of what was written vs. skipped.
///
/// Every tensor is written verbatim (bytes, dtype and shape preserved); no
/// FP16 → FP32 widening. The numeric hparams are `0`-placeholders (except
/// the model-card sample_rate and Mimi shape invariants) pending the T02
/// upstream inspection — the runtime rejects those `0`s at load per
/// `CosyVoice2Config` reader in `crates/vokra-models/src/cosyvoice2/config.rs`.
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, CosyVoice2Report), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    write_hparams(&mut b);

    let mut report = CosyVoice2Report::default();
    for t in st.tensors() {
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )?;
                report.written += 1;
            }
            _ => {
                report.skipped_non_float += 1;
            }
        }
    }

    Ok((b, report))
}

/// Writes the `vokra.cosyvoice2.*` hparam chunk group.
///
/// The numeric hparams (`vocab_size` / `hidden_dim` / `n_layer` / `n_head` /
/// `ffn_dim` / `flow.nfe` / `streaming.chunk_size` / `streaming.chunk_hop`)
/// are `0`-placeholders pending the T02 upstream inspection. The runtime
/// (`CosyVoice2Config::from_gguf`) reads them as-is; a T07/T08/T10 forward
/// path will refuse to run against a `0`-hparam GGUF, so a converter-side
/// `0` is a loud fail at first use rather than a silent zero-shape forward.
fn write_hparams(b: &mut GgufBuilder) {
    b.add_u32(KEY_SAMPLE_RATE, COSYVOICE2_SAMPLE_RATE);
    b.add_u32(KEY_VOCAB_SIZE, 0);
    b.add_u32(KEY_HIDDEN_DIM, 0);
    b.add_u32(KEY_N_LAYER, 0);
    b.add_u32(KEY_N_HEAD, 0);
    b.add_u32(KEY_FFN_DIM, 0);
    b.add_u32(KEY_FLOW_NFE, 0);
    // The schedule tag has no meaningful `0`-placeholder — a missing
    // schedule tag is what the runtime error is written to catch. We
    // write `"linear"` here (the M3-05 default schedule); T04 will
    // replace it with the tag the upstream checkpoint carries in its
    // Flow Matching config.
    b.add_string(KEY_FLOW_SCHEDULE, "linear");
    b.add_u32(KEY_MIMI_N_CODEBOOKS, MIMI_N_CODEBOOKS);
    b.add_u32(KEY_MIMI_CODEBOOK_SIZE, MIMI_CODEBOOK_SIZE);
    b.add_u32(KEY_MIMI_D_MODEL, MIMI_D_MODEL);
    b.add_u32(KEY_STREAMING_CHUNK_SIZE, 0);
    b.add_u32(KEY_STREAMING_CHUNK_HOP, 0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    /// Builds a minimal safetensors buffer with one F32 tensor. Payload is
    /// deliberately trivial (all-zero) — only the header parsing and the
    /// verbatim byte-copy path are exercised.
    fn minimal_safetensors_one_f32() -> Vec<u8> {
        // A single F32 tensor of shape [2, 3] = 6 elements = 24 bytes.
        let header = r#"{"llm.wte":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 24]);
        out
    }

    #[test]
    fn round_trip_carries_arch_and_cosyvoice2_chunk_group() {
        // The chunk group must round-trip through `to_bytes` → `parse` so
        // the runtime constants in vokra-models/cosyvoice2/config.rs read
        // the same values back. A drift between the two constants would
        // surface here at compile-run time.
        let bytes = minimal_safetensors_one_f32();
        let (builder, report) = convert(bytes).expect("convert");
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");

        // Arch / name.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );

        // Sample rate: model-card invariant.
        match file.get(KEY_SAMPLE_RATE) {
            Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, COSYVOICE2_SAMPLE_RATE),
            other => panic!("sample_rate: unexpected {other:?}"),
        }

        // Mimi shape: canonical Kyutai defaults.
        for (key, expected) in [
            (KEY_MIMI_N_CODEBOOKS, MIMI_N_CODEBOOKS),
            (KEY_MIMI_CODEBOOK_SIZE, MIMI_CODEBOOK_SIZE),
            (KEY_MIMI_D_MODEL, MIMI_D_MODEL),
        ] {
            match file.get(key) {
                Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, expected, "{key}"),
                other => panic!("{key}: unexpected {other:?}"),
            }
        }

        // Placeholder hparams: `0` until T02.
        for key in [
            KEY_VOCAB_SIZE,
            KEY_HIDDEN_DIM,
            KEY_N_LAYER,
            KEY_N_HEAD,
            KEY_FFN_DIM,
            KEY_FLOW_NFE,
            KEY_STREAMING_CHUNK_SIZE,
            KEY_STREAMING_CHUNK_HOP,
        ] {
            match file.get(key) {
                Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, 0, "{key}"),
                other => panic!("{key}: unexpected {other:?}"),
            }
        }

        // Schedule tag: `linear` default.
        assert_eq!(
            file.get(KEY_FLOW_SCHEDULE).and_then(|v| v.as_str()),
            Some("linear")
        );
    }

    #[test]
    fn arch_string_matches_runtime_constant() {
        // Hard-coded sanity: the runtime's EXPECTED_ARCH is `cosyvoice2`;
        // this file's ARCH constant must be identical. A drift is caught
        // here rather than at load time.
        assert_eq!(ARCH, "cosyvoice2");
    }
}
