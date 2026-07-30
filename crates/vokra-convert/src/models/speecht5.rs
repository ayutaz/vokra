//! **SpeechT5 (TTS variant)** (`microsoft/speecht5_tts`, MIT):
//! safetensors → GGUF conversion (implementer C wave, 2026-07-30).
//!
//! Input: the upstream `microsoft/speecht5_tts` release — the upstream
//! ships `pytorch_model.bin` (torch pickle); callers must offline-flatten
//! to safetensors first (mirror of the CSM / DAC / DFN3 prepare-script
//! pattern — `crates/vokra-convert/src/models/{csm,dac,denoise}.rs`).
//! Output: a GGUF carrying every float tensor plus the
//! `vokra.provenance.*` / `vokra.model.*` metadata chunks a future
//! native SpeechT5 loader will read.
//!
//! # Architecture (primary source, 2026-07-30 CC fetch)
//!
//! SpeechT5 is Microsoft's **unified encoder-decoder pre-training**
//! architecture (Ao et al., 2022; ACL 2022 highlight); the `_tts` head
//! is the text-to-speech specialisation. TTS pipeline: text (SentencePiece
//! character) → 12-layer Transformer encoder → cross-attention → 6-layer
//! Transformer decoder → speech-decoder pre-net (256-unit MLP) → mel
//! spectrogram (80 mel bins, reduction factor 2) → speech-decoder
//! post-net (5-layer conv). Speaker conditioning is a **512-d x-vector**
//! supplied by the caller (mirror of CAM++ / ECAPA-TDNN — SpeechT5 does
//! NOT include a speaker encoder inside the checkpoint).
//!
//! Every hparam below is transcribed verbatim from `huggingface.co/
//! microsoft/speecht5_tts/raw/main/config.json` (fetched 2026-07-30 —
//! CLAUDE.md「ハルシネーション厳禁」):
//!
//! - `architectures = ["SpeechT5ForTextToSpeech"]`
//! - `model_type = "speecht5"`
//! - `hidden_size = 768`
//! - `encoder_layers = 12` (shared unified encoder)
//! - `decoder_layers = 6`
//! - `encoder_attention_heads = 12`, `decoder_attention_heads = 12`
//! - `encoder_ffn_dim = 3072`, `decoder_ffn_dim = 3072`
//! - `vocab_size = 81` (SentencePiece character tokenizer)
//! - `num_mel_bins = 80`
//! - `reduction_factor = 2` (mel frames per decoder step)
//! - `speech_decoder_prenet_units = 256`
//! - `speech_decoder_postnet_layers = 5`
//! - `speaker_embedding_dim = 512` (x-vector, caller-supplied)
//!
//! The `vokra.speecht5.*` hparam chunk pins every one of these fields
//! so a future `SpeechT5Weights::from_gguf` reader can walk them without
//! re-parsing the upstream `config.json`.
//!
//! # BF16 pass-through (mirror of `qwen3_tts` / `wespeaker` / `neucodec`)
//!
//! F32 / F16 / BF16 tensors pass through **verbatim** under their
//! upstream safetensors names. BF16 stays GGUF type 30
//! (`GgmlType::BF16`) — no convert-time widening; runtime widens BF16 →
//! f32 losslessly at load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM /
//! VibeVoice / Neucodec contract). Real-weight binding is a follow-up
//! wave gated on the upstream tensor-name manifest fetch; this converter
//! passes every F32 / F16 / BF16 tensor through unchanged so a future
//! `SpeechT5Weights::from_gguf` can walk the same names.
//!
//! # Scope — **only the TTS variant** (implementer C constraint)
//!
//! The sibling `microsoft/speecht5_vc` (voice-conversion) is deliberately
//! **out of scope** per implementer C task spec: voice-conversion is a
//! `vokra-voiceclone-experimental` (tier-5 別リポ) concern by CLAUDE.md
//! 設計判断 8 (ELVIS Act 分離). This converter must never accept a
//! `_vc` checkpoint — but shape-wise a `_vc` checkpoint is close enough
//! that we cannot loudly reject it here (no shape difference identifies
//! it); the discipline is enforced through the CLI slug (`speecht5-tts`
//! is the only spelling that dispatches here) and the docstring above.
//!
//! # Real-weight parity
//!
//! Real-weight parity vs the upstream Microsoft SpeechT5 Python
//! pipeline is deferred to owner (`docs/license-audit.md` §3.1
//! sign-off) — this converter provides the byte-parallel GGUF surface
//! only.
//!
//! # No ONNX (permanent)
//!
//! SpeechT5 is distributed as a torch pickle (`pytorch_model.bin`) via
//! the HuggingFace `transformers` release; the converter **never**
//! touches ONNX (FR-LD-05); the pipeline is re-implemented natively
//! in a future `crates/vokra-models/src/speecht5/` module (whisper.cpp
//! 型 self re-implementation, CLAUDE.md 設計判断 4).

#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for SpeechT5 TTS GGUFs. Distinct from every
/// sibling TTS arch tag — silently sharing would misroute the runtime
/// dispatch (SpeechT5 has a unique architecture: unified encoder +
/// speech-decoder-prenet + speech-decoder-postnet + external speaker
/// x-vector conditioning).
pub(crate) const ARCH: &str = "speecht5";
/// `vokra.model.name` value — pins the TTS variant explicitly so a
/// consumer inspecting the artifact can tell it apart from the (future,
/// out-of-scope) `_vc` release.
pub(crate) const NAME: &str = "speecht5-tts";
/// Model category tag — `tts`.
pub(crate) const CATEGORY: &str = "tts";
/// Upstream HuggingFace repo slug.
pub(crate) const UPSTREAM_HF: &str = "microsoft/speecht5_tts";
/// SPDX default license (upstream ships MIT end-to-end).
pub(crate) const DEFAULT_LICENSE: &str = "mit";

// ---- Hparams (transcribed verbatim from upstream config.json) -----------

/// `hidden_size = 768`.
pub(crate) const HIDDEN_SIZE: u32 = 768;
/// `encoder_layers = 12`.
pub(crate) const ENCODER_LAYERS: u32 = 12;
/// `decoder_layers = 6`.
pub(crate) const DECODER_LAYERS: u32 = 6;
/// `encoder_attention_heads = 12`.
pub(crate) const ENCODER_ATTENTION_HEADS: u32 = 12;
/// `decoder_attention_heads = 12`.
pub(crate) const DECODER_ATTENTION_HEADS: u32 = 12;
/// `encoder_ffn_dim = 3072`.
pub(crate) const ENCODER_FFN_DIM: u32 = 3_072;
/// `decoder_ffn_dim = 3072`.
pub(crate) const DECODER_FFN_DIM: u32 = 3_072;
/// `vocab_size = 81` (SentencePiece character tokenizer, `spm_char.model`).
pub(crate) const VOCAB_SIZE: u32 = 81;
/// `num_mel_bins = 80`.
pub(crate) const NUM_MEL_BINS: u32 = 80;
/// `reduction_factor = 2` — mel frames emitted per decoder step.
pub(crate) const REDUCTION_FACTOR: u32 = 2;
/// `speech_decoder_prenet_units = 256`.
pub(crate) const SPEECH_DECODER_PRENET_UNITS: u32 = 256;
/// `speech_decoder_postnet_layers = 5`.
pub(crate) const SPEECH_DECODER_POSTNET_LAYERS: u32 = 5;
/// `speaker_embedding_dim = 512` — caller-supplied x-vector.
pub(crate) const SPEAKER_EMBEDDING_DIM: u32 = 512;

// ---- Additive metadata keys ---------------------------------------------

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_HIDDEN_SIZE: &str = "vokra.speecht5.hidden_size";
const KEY_ENCODER_LAYERS: &str = "vokra.speecht5.encoder_layers";
const KEY_DECODER_LAYERS: &str = "vokra.speecht5.decoder_layers";
const KEY_ENCODER_ATTENTION_HEADS: &str = "vokra.speecht5.encoder_attention_heads";
const KEY_DECODER_ATTENTION_HEADS: &str = "vokra.speecht5.decoder_attention_heads";
const KEY_ENCODER_FFN_DIM: &str = "vokra.speecht5.encoder_ffn_dim";
const KEY_DECODER_FFN_DIM: &str = "vokra.speecht5.decoder_ffn_dim";
const KEY_VOCAB_SIZE: &str = "vokra.speecht5.vocab_size";
const KEY_NUM_MEL_BINS: &str = "vokra.speecht5.num_mel_bins";
const KEY_REDUCTION_FACTOR: &str = "vokra.speecht5.reduction_factor";
const KEY_SPEECH_DECODER_PRENET_UNITS: &str = "vokra.speecht5.speech_decoder_prenet_units";
const KEY_SPEECH_DECODER_POSTNET_LAYERS: &str = "vokra.speecht5.speech_decoder_postnet_layers";
const KEY_SPEAKER_EMBEDDING_DIM: &str = "vokra.speecht5.speaker_embedding_dim";

/// Outcome of a SpeechT5 TTS conversion. Mirrors the sibling BF16
/// pass-through report shape.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SpeechT5Report {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter).
    pub skipped_non_float: usize,
    /// BF16 tensors on the pass-through arm.
    pub bf16_passthrough: usize,
}

/// File-based SpeechT5 TTS converter (`vokra-cli convert --model
/// speecht5-tts`).
///
/// Reads `input` (upstream `microsoft/speecht5_tts` flattened to
/// safetensors — the upstream ships `pytorch_model.bin`, callers
/// pre-flatten offline mirror of the CSM / DAC / DFN3 pattern), writes
/// a Vokra GGUF to `output`. `license` overrides the default `mit`
/// provenance stamp (see `convert_file_licensed` in `lib.rs`); pass
/// `None` to keep the built-in `mit` stamp.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_speecht5_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<SpeechT5Report, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_UPSTREAM_HF, UPSTREAM_HF);

    // Hparam chunk — transcribed verbatim from upstream config.json.
    b.add_u32(KEY_HIDDEN_SIZE, HIDDEN_SIZE);
    b.add_u32(KEY_ENCODER_LAYERS, ENCODER_LAYERS);
    b.add_u32(KEY_DECODER_LAYERS, DECODER_LAYERS);
    b.add_u32(KEY_ENCODER_ATTENTION_HEADS, ENCODER_ATTENTION_HEADS);
    b.add_u32(KEY_DECODER_ATTENTION_HEADS, DECODER_ATTENTION_HEADS);
    b.add_u32(KEY_ENCODER_FFN_DIM, ENCODER_FFN_DIM);
    b.add_u32(KEY_DECODER_FFN_DIM, DECODER_FFN_DIM);
    b.add_u32(KEY_VOCAB_SIZE, VOCAB_SIZE);
    b.add_u32(KEY_NUM_MEL_BINS, NUM_MEL_BINS);
    b.add_u32(KEY_REDUCTION_FACTOR, REDUCTION_FACTOR);
    b.add_u32(KEY_SPEECH_DECODER_PRENET_UNITS, SPEECH_DECODER_PRENET_UNITS);
    b.add_u32(
        KEY_SPEECH_DECODER_POSTNET_LAYERS,
        SPEECH_DECODER_POSTNET_LAYERS,
    );
    b.add_u32(KEY_SPEAKER_EMBEDDING_DIM, SPEAKER_EMBEDDING_DIM);

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = mit (upstream `microsoft/speecht5_tts` model
    // card, fetched 2026-07-30 — CLAUDE.md「ハルシネーション厳禁」).
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(&mut b, class, &spdx, Some(NAME), Some(UPSTREAM_HF));

    let mut report = SpeechT5Report::default();
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )
                .map_err(|e| ConvertError::Gguf(e.to_string()))?;
                report.written += 1;
                if t.dtype == GgmlType::BF16 {
                    report.bf16_passthrough += 1;
                }
            }
            _ => {
                report.skipped_non_float += 1;
            }
        }
    }

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-speecht5-{}-{}-{}.bin",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        p
    }

    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        assert_eq!(bf16_bytes.len(), elems as usize * 2);
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"BF16","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            bf16_bytes.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(bf16_bytes);
        out
    }

    fn synth_bf16_input() -> (Vec<u8>, Vec<u8>) {
        // Non-zero BF16 bits so a byte-identity assert catches a silent
        // widen / downcast. Use an upstream-realistic tensor name.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let bytes = safetensors_one_bf16(
            "speecht5.encoder.wrapped_encoder.layer.0.attention.k_proj.weight",
            &[2, 3],
            &bf16,
        );
        (bytes, bf16)
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (input_bytes, bf16_payload) = synth_bf16_input();
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).unwrap();

        let report = convert_speecht5_file(&input, &output, None).expect("convert must succeed");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);

        let file = GgufFile::parse(std::fs::read(&output).unwrap()).unwrap();
        let info = file
            .tensor_info("speecht5.encoder.wrapped_encoder.layer.0.attention.k_proj.weight")
            .expect("BF16 tensor present after pass-through");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), bf16_payload.as_slice());
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn hparam_chunk_is_stamped() {
        let (input_bytes, _) = synth_bf16_input();
        let input = scratch_path("hp-in");
        let output = scratch_path("hp-out");
        std::fs::write(&input, &input_bytes).unwrap();

        convert_speecht5_file(&input, &output, None).expect("convert must succeed");
        let file = GgufFile::parse(std::fs::read(&output).unwrap()).unwrap();

        // Identity + provenance chunks.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );
        assert_eq!(
            file.get(KEY_UPSTREAM_HF).and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE)
        );

        // Every one of the 13 hparam pins matches the transcribed
        // upstream config.json values.
        for (k, expect) in [
            (KEY_HIDDEN_SIZE, 768u64),
            (KEY_ENCODER_LAYERS, 12),
            (KEY_DECODER_LAYERS, 6),
            (KEY_ENCODER_ATTENTION_HEADS, 12),
            (KEY_DECODER_ATTENTION_HEADS, 12),
            (KEY_ENCODER_FFN_DIM, 3_072),
            (KEY_DECODER_FFN_DIM, 3_072),
            (KEY_VOCAB_SIZE, 81),
            (KEY_NUM_MEL_BINS, 80),
            (KEY_REDUCTION_FACTOR, 2),
            (KEY_SPEECH_DECODER_PRENET_UNITS, 256),
            (KEY_SPEECH_DECODER_POSTNET_LAYERS, 5),
            (KEY_SPEAKER_EMBEDDING_DIM, 512),
        ] {
            assert_eq!(
                file.get(k).and_then(|v| v.as_u64()),
                Some(expect),
                "{k} must be stamped as {expect}"
            );
        }
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn license_override_reclassifies_provenance() {
        let (input_bytes, _) = synth_bf16_input();
        let input = scratch_path("lic-in");
        let output = scratch_path("lic-out");
        std::fs::write(&input, &input_bytes).unwrap();

        convert_speecht5_file(&input, &output, Some("apache-2.0"))
            .expect("convert must accept license override");
        let file = GgufFile::parse(std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
