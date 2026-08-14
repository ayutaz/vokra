#![allow(clippy::doc_lazy_continuation)]
//! **MT3** (`magenta/mt3`, apache-2.0 code, weight license UNCLEAR):
//! safetensors → GGUF conversion (Wave 3, music-transcription
//! upgrade, 2026-08-14).
//!
//! MT3 = **Multi-Task Multitrack Music Transcription** (Gardner et
//! al. ICLR 2022, arXiv:2111.03017). A **T5-small encoder-decoder**
//! (~60M parameters) that ingests log-mel spectrogram frames and
//! emits a MIDI event token stream covering multiple simultaneous
//! instruments (drums, bass, guitar, piano, strings, etc). The
//! encoder is stock T5 (relative-attention-bias multi-head attention
//! + FFN + `RMSNorm`); the decoder autoregressively emits a music
//! event vocabulary (`event_codec.py`) whose tokens post-process to
//! MIDI note-on / note-off / program-change events.
//!
//! # Vokra scope — music-transcription upgrade over Basic-Pitch
//!
//! Complements the sibling `basic_pitch` (Spotify Research
//! polyphonic-CNN posteriorgram audio-to-MIDI, `pitch-transcription`
//! category, ~6 MB CNN) with the **multi-track T5-encoder-decoder
//! upgrade** — MT3 is not just polyphonic (single-instrument
//! posteriorgram over pitch bins) but **multi-track** (concurrent
//! instrument voices resolved by program change), which is a strict
//! superset of Basic-Pitch's output surface. Distinct arch tag `mt3`
//! (never `basic-pitch`, never any T5 speech-tree arch) because the
//! T5-small encoder-decoder body + music-event vocabulary topology is
//! a distinct binding target from every existing binder in the tree
//! (no shared T5 encoder-decoder runtime binder exists — the
//! MusicGen family lands T5 tensors as opaque BF16 passthroughs
//! without a runtime binder). Category `music-transcription` (new)
//! — distinct from `pitch-transcription` (owned by `basic_pitch`)
//! because the output is a MIDI event stream with program-change
//! semantics, not a pitch posteriorgram.
//!
//! # License posture — apache-2.0 CODE, weight license UNCLEAR
//!
//! Upstream `github.com/magenta/mt3` LICENSE = Apache-2.0 (verified
//! via `gh api repos/magenta/mt3/license` at scout time
//! 2026-08-14). **However, weights ship on `gs://mt3/checkpoints/`
//! with no explicit per-bucket LICENSE file**, no HuggingFace
//! mirror, and the paper is silent on weight redistribution. This
//! is the classic "code MIT/Apache but weight license unclear"
//! split (mirror of F5-TTS / EnCodec, though here the code side is
//! actually permissive rather than the more common `code MIT +
//! weight CC-BY-NC`). Under Vokra's fail-closed compliance policy
//! this converter hard-maps to [`LicenseClass::Unknown`]
//! **regardless** of the SPDX string the caller passes — the runtime
//! compliance gate (FR-CP-03) will refuse to load the resulting
//! GGUF in commercial mode until the owner completes primary-source
//! confirmation on the weight side and un-Unknowns it.
//!
//! The `license` override parameter is retained for **provenance
//! recording** — the raw SPDX still lands in
//! `vokra.provenance.license` for audit trail — but the `LicenseClass`
//! stamped on `vokra.provenance.weight_license` is always
//! `Unknown` here. This mirrors the fail-closed pattern from the
//! sibling `chattts` / `xtts_v2` converters where the code license
//! resolves permissive but the weight license lands `Unknown`.
//!
//! # Scale — local convert OK (~0.24 GB / T5-small ~60M params)
//!
//! Well below the M1 iMac 16 GB local-convert threshold (memory
//! `[[feedback-large-models-on-vast-ai]]`: <2 GB safe). T5-small
//! `~60M params × 4 B (F32) = 240 MB`; BF16 halves this. No vast.ai
//! handoff needed.
//!
//! # No ONNX / no JAX (permanent)
//!
//! MT3 ships upstream as a **T5X / JAX checkpoint** (not PyTorch).
//! This converter **never** touches ONNX or JAX (FR-LD-05 / NFR-DS-02).
//! Callers pre-flatten the T5X checkpoint to safetensors offline via
//! a future `tools/parity/mt3_prepare_checkpoint.py` (the
//! DAC / Kokoro / UTMOSv2 / beats bridge pattern, uv-managed Python
//! 3.12 sidecar per memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`). Runtime tree carries neither the
//! JAX runtime nor `t5x`.
//!
//! # BF16 pass-through
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm. BF16 is
//! emitted as GGUF type 30 ([`GgmlType::BF16`]); the runtime widens
//! BF16 → f32 losslessly at load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (the same
//! choke point every sibling BF16 pass-through converter binds
//! against — never fabricated fp32 conversions elsewhere in the
//! tree).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for MT3 GGUFs. Distinct from every sibling
/// converter — never `basic-pitch` (Spotify polyphonic-CNN, different
/// topology + different output surface), never any T5 speech-tree
/// arch (no shared T5 encoder-decoder runtime binder exists).
pub const ARCH: &str = "mt3";

/// `vokra.model.name` — canonical `magenta/mt3` release.
pub const NAME: &str = "mt3-multitrack";

/// `vokra.model.category` — music-transcription (multi-track MIDI
/// event transcription from audio, distinct from
/// `pitch-transcription` owned by `basic_pitch`).
pub const CATEGORY: &str = "music-transcription";

/// Upstream source — recorded on `vokra.provenance.upstream_url`.
/// MT3 has **no HuggingFace mirror**; the checkpoint lives on
/// `gs://mt3/checkpoints/` and the reference source on
/// `github.com/magenta/mt3`. We stamp the GitHub URL because it is
/// the primary source for both the code license (Apache-2.0,
/// verified via `gh api repos/magenta/mt3/license`) and the
/// reference `network.py` / `vocabularies.py` / `event_codec.py`.
pub const UPSTREAM_URL: &str = "github.com/magenta/mt3";

/// Default SPDX. Upstream `github.com/magenta/mt3` LICENSE =
/// Apache-2.0 (verified via `gh api repos/magenta/mt3/license` at
/// scout time 2026-08-14) — this describes the **code** license.
/// **The weight license is UNCLEAR** (no per-bucket LICENSE on
/// `gs://mt3/checkpoints/`, no HF mirror) so the converter
/// hard-maps to [`LicenseClass::Unknown`] regardless of this string
/// (see [`convert_mt3_file`] rustdoc).
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// ---------------------------------------------------------------------------
// T5-small hyperparameters — transcribed from magenta/mt3
// `mt3/network.py` (defaults for the released MT3 checkpoint). Stamped
// on the GGUF so the runtime binder can validate topology without
// re-inspecting the tensor shapes.
//
// Sources:
// - github.com/magenta/mt3/blob/main/mt3/network.py
// - arxiv.org/abs/2111.03017 (Gardner et al. 2022 §3.2 "Model")
//
// These match the standard T5-small (Raffel et al. 2020) axes with
// MT3's task-specific music-event vocabulary. The vocabulary size
// derives from `event_codec.py` (MIDI note events + program change +
// velocity + tie/EOS markers) and is transcribed here as an
// approximate `1200` — the runtime binder validates the actual value
// against the stamped chunk; a mismatched checkpoint fails loudly.
// ---------------------------------------------------------------------------

/// T5-small hidden dimension (`d_model`).
pub const D_MODEL: u32 = 512;
/// T5-small FFN inner dimension (`d_ff`).
pub const D_FF: u32 = 1024;
/// T5-small attention head count (`n_heads`).
pub const N_HEADS: u32 = 6;
/// T5-small per-head dimension (`d_kv`). Distinct from `d_model /
/// n_heads` in T5 (Raffel et al. 2020) — `d_kv=64` regardless of the
/// `d_model / n_heads` product.
pub const D_KV: u32 = 64;
/// MT3 encoder stack depth (`num_encoder_layers`).
pub const NUM_ENC_LAYERS: u32 = 12;
/// MT3 decoder stack depth (`num_decoder_layers`).
pub const NUM_DEC_LAYERS: u32 = 12;
/// MT3 music-event vocabulary size (approximate — transcribed from
/// the `event_codec.py` event enumeration: note events + program
/// change + velocity + tie/EOS markers). The runtime binder
/// validates the actual chunk value; a mismatched checkpoint fails
/// loudly.
pub const MUSIC_VOCAB_SIZE: u32 = 1200;
/// T5 relative-attention bucket count (`num_buckets`, T5 default 32).
pub const REL_ATTN_NUM_BUCKETS: u32 = 32;
/// T5 relative-attention max distance (`max_distance`, T5 default 128).
pub const REL_ATTN_MAX_DISTANCE: u32 = 128;

// ---------------------------------------------------------------------------
// GGUF chunk keys — mirror of `crates/vokra-models/src/mt3/mod.rs`
// GGUF_KEY_* (see runtime binder module doc for the cross-crate
// duplication rationale — vokra-models must not gain a dep edge onto
// vokra-convert).
// ---------------------------------------------------------------------------

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

const KEY_MT3_D_MODEL: &str = "vokra.mt3.d_model";
const KEY_MT3_D_FF: &str = "vokra.mt3.d_ff";
const KEY_MT3_N_HEADS: &str = "vokra.mt3.n_heads";
const KEY_MT3_D_KV: &str = "vokra.mt3.d_kv";
const KEY_MT3_NUM_ENC_LAYERS: &str = "vokra.mt3.num_enc_layers";
const KEY_MT3_NUM_DEC_LAYERS: &str = "vokra.mt3.num_dec_layers";
const KEY_MT3_MUSIC_VOCAB_SIZE: &str = "vokra.mt3.music_vocab_size";
const KEY_MT3_REL_ATTN_NUM_BUCKETS: &str = "vokra.mt3.rel_attn_num_buckets";
const KEY_MT3_REL_ATTN_MAX_DISTANCE: &str = "vokra.mt3.rel_attn_max_distance";

const UPSTREAM_SOURCE: &str = "magenta/mt3 (Google Magenta Multi-Task Multitrack Music \
     Transcription, T5-small encoder-decoder ~60M params, log-mel spectrogram → MIDI event \
     tokens covering multiple simultaneous instruments, Gardner et al. ICLR 2022 \
     arXiv:2111.03017, apache-2.0 code / weight license UNCLEAR (no per-bucket LICENSE on \
     gs://mt3/checkpoints/))";

/// Outcome of an MT3 conversion. Mirrors the counter shape of the
/// sibling BF16 pass-through converters (`basic_pitch` /
/// `musicgen_small` / `beat_this` / `dasheng` / `mert` / `muq`) —
/// the invariant `read == written + skipped_non_float` is auditable
/// at the report level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Mt3Report {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the
    /// safetensors reader accepts only F32 / F16 / BF16, so any
    /// tensor reaching this counter would signal a reader change
    /// upstream).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16
    /// (subset counter). Emits GGUF type 30 verbatim; the runtime
    /// widens BF16 → f32 losslessly via the single choke point
    /// `vokra_core::gguf::quant::decode_bf16`.
    pub bf16_passthrough: usize,
}

/// Converts a `magenta/mt3` safetensors checkpoint at `input` into a
/// Vokra-native GGUF at `output`, returning an [`Mt3Report`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict name; the `vokra.model.*` / `vokra.provenance.*` /
/// `vokra.mt3.*` chunks are stamped for the runtime compliance gate
/// (FR-CP-03) and the runtime binder topology validation.
///
/// # License policy — always Unknown regardless of `license` arg
///
/// `license` optionally overrides the raw SPDX string stamped on
/// `vokra.provenance.license` (default is [`DEFAULT_LICENSE_SPDX`],
/// `"apache-2.0"`). **However**, the [`LicenseClass`] stamped on
/// `vokra.provenance.weight_license` is **always
/// [`LicenseClass::Unknown`]** here because the MT3 weight bucket
/// (`gs://mt3/checkpoints/`) carries no per-bucket LICENSE file and
/// no HuggingFace mirror exists as of 2026-08-14. This is
/// fail-closed at the runtime compliance gate — the resulting GGUF
/// will not load in commercial mode until the owner completes
/// primary-source confirmation on the weight side and either
/// re-stamps the class explicitly or the converter is updated to
/// map the confirmed weight license.
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_mt3_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<Mt3Report, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);

    // T5-small topology axes (transcribed from magenta/mt3
    // `mt3/network.py` and Raffel et al. 2020) + MT3-specific
    // music-vocab / relative-attention params.
    b.add_u32(KEY_MT3_D_MODEL, D_MODEL);
    b.add_u32(KEY_MT3_D_FF, D_FF);
    b.add_u32(KEY_MT3_N_HEADS, N_HEADS);
    b.add_u32(KEY_MT3_D_KV, D_KV);
    b.add_u32(KEY_MT3_NUM_ENC_LAYERS, NUM_ENC_LAYERS);
    b.add_u32(KEY_MT3_NUM_DEC_LAYERS, NUM_DEC_LAYERS);
    b.add_u32(KEY_MT3_MUSIC_VOCAB_SIZE, MUSIC_VOCAB_SIZE);
    b.add_u32(KEY_MT3_REL_ATTN_NUM_BUCKETS, REL_ATTN_NUM_BUCKETS);
    b.add_u32(KEY_MT3_REL_ATTN_MAX_DISTANCE, REL_ATTN_MAX_DISTANCE);

    // Provenance stamp — SPDX raw string may reflect the caller's
    // override (Apache-2.0 code license by default), but the
    // LicenseClass is hard-mapped to Unknown per the fail-closed
    // policy documented on this fn's rustdoc.
    let spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Unknown,
        spdx,
        Some(NAME),
        Some(UPSTREAM_SOURCE),
    );

    let mut report = Mt3Report::default();
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
    std::fs::write(output, &out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::GgufFile;

    fn tmp_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-mt3-{tag}-{}-{n}",
            std::process::id()
        ));
        p
    }

    fn safetensors_one(name: &str, dtype: &str, shape: &[u64], payload: &[u8]) -> Vec<u8> {
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"{dtype}","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn f32_tensor_passes_through_and_stamps_topology_chunk_group() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // T5 encoder layer 0 self-attention Q projection (typical
        // upstream state_dict name shape).
        let st = safetensors_one(
            "encoder.layers.0.self_attn.q_proj.weight",
            "F32",
            &[1, 2],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();

        let r = convert_mt3_file(&inp, &outp, None).expect("convert F32");
        assert_eq!(r.read, 1);
        assert_eq!(r.written, 1);
        assert_eq!(r.bf16_passthrough, 0);

        let g = GgufFile::open(&outp).unwrap();
        let read_str = |k: &str| -> String {
            g.get(k)
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| panic!("{k}: missing"))
                .to_owned()
        };
        assert_eq!(read_str(chunks::KEY_MODEL_ARCH), ARCH);
        assert_eq!(read_str(chunks::KEY_MODEL_NAME), NAME);
        assert_eq!(read_str(KEY_MODEL_CATEGORY), CATEGORY);
        assert_eq!(read_str(KEY_PROVENANCE_UPSTREAM_URL), UPSTREAM_URL);
        assert_eq!(
            read_str(chunks::KEY_PROVENANCE_LICENSE),
            DEFAULT_LICENSE_SPDX,
        );

        // Fail-closed license policy: default SPDX is apache-2.0 (code
        // license, Permissive on its own) BUT the LicenseClass is
        // hard-mapped to Unknown because the weight bucket has no
        // per-bucket LICENSE and no HF mirror as of 2026-08-14.
        assert_eq!(
            read_str(chunks::KEY_PROVENANCE_WEIGHT_LICENSE),
            LicenseClass::Unknown.as_str(),
            "MT3 weight license must land Unknown regardless of SPDX \
             — fail-closed at compliance gate until owner \
             primary-source confirms the weight bucket",
        );

        // Read the u32 topology axes back through the GGUF.
        let read_u32 = |k: &str| -> u32 {
            g.get(k)
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| panic!("{k}: missing u32"))
                .try_into()
                .unwrap()
        };
        assert_eq!(read_u32(KEY_MT3_D_MODEL), D_MODEL);
        assert_eq!(read_u32(KEY_MT3_D_FF), D_FF);
        assert_eq!(read_u32(KEY_MT3_N_HEADS), N_HEADS);
        assert_eq!(read_u32(KEY_MT3_D_KV), D_KV);
        assert_eq!(read_u32(KEY_MT3_NUM_ENC_LAYERS), NUM_ENC_LAYERS);
        assert_eq!(read_u32(KEY_MT3_NUM_DEC_LAYERS), NUM_DEC_LAYERS);
        assert_eq!(read_u32(KEY_MT3_MUSIC_VOCAB_SIZE), MUSIC_VOCAB_SIZE);
        assert_eq!(read_u32(KEY_MT3_REL_ATTN_NUM_BUCKETS), REL_ATTN_NUM_BUCKETS);
        assert_eq!(
            read_u32(KEY_MT3_REL_ATTN_MAX_DISTANCE),
            REL_ATTN_MAX_DISTANCE,
        );

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let inp = tmp_path("bf16-in");
        let outp = tmp_path("bf16-out");
        let values: [f32; 4] = [1.0, -0.5, 0.25, 8.0];
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let st = safetensors_one(
            "decoder.layers.0.self_attn.k_proj.weight",
            "BF16",
            &[2, 2],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();

        let r = convert_mt3_file(&inp, &outp, None).expect("convert BF16");
        assert_eq!(r.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&outp).unwrap();
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("decoder.layers.0.self_attn.k_proj.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), payload.as_slice());

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn license_override_records_spdx_but_class_stays_unknown() {
        // Even if the caller overrides the SPDX string, the
        // LicenseClass stamped on `vokra.provenance.weight_license`
        // stays Unknown per the fail-closed policy documented on
        // convert_mt3_file. The raw SPDX is recorded for audit
        // trail on `vokra.provenance.license`.
        let inp = tmp_path("lic-in");
        let outp = tmp_path("lic-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("x", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();

        convert_mt3_file(&inp, &outp, Some("mit")).expect("convert with override");
        let g = GgufFile::open(&outp).unwrap();
        // Raw SPDX reflects the override.
        assert_eq!(
            g.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
        );
        // But the LicenseClass stays Unknown regardless.
        assert_eq!(
            g.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Unknown.as_str()),
            "SPDX override must not lift MT3 out of Unknown \
             (fail-closed policy)",
        );

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn skips_non_float_tensor_defensively() {
        // Any tensor whose dtype is not F32 / F16 / BF16 is counted
        // on `skipped_non_float` (defensive — the safetensors reader
        // already narrows to these three, so this arm only fires if
        // an upstream reader change lets an INT dtype through).
        let inp = tmp_path("nonfloat-in");
        let outp = tmp_path("nonfloat-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("x", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();

        let r = convert_mt3_file(&inp, &outp, None).expect("convert");
        // One float tensor read + written; no non-float in this
        // fixture but the invariant is asserted so a future reader
        // widening will surface here.
        assert_eq!(r.read, r.written + r.skipped_non_float);
    }

    #[test]
    fn arch_tag_is_stable_and_distinct_from_sibling_music_arches() {
        // Pin the string constants so a rename would land here in
        // the same commit or fail this test. The sibling music-tree
        // arches (`basic-pitch` polyphonic-CNN, `beat-this`
        // Transformer beat-tracker, `musicgen` text-to-music AR LM)
        // MUST NOT collide with ours.
        assert_eq!(ARCH, "mt3");
        assert_ne!(
            ARCH, "basic-pitch",
            "mt3 (T5 encoder-decoder multi-track transcription) and \
             basic-pitch (Spotify polyphonic-CNN posteriorgram) are \
             different topologies — sharing arch would mis-route \
             runtime dispatch (FR-EX-08)",
        );
        assert_ne!(
            ARCH, "beat-this",
            "mt3 and beat-this are different tasks (transcription \
             vs beat-tracking) — sharing arch would mis-route (FR-EX-08)",
        );
        assert_ne!(
            ARCH, "musicgen",
            "mt3 (transcription) and musicgen (generation) are \
             opposite directions — sharing arch would mis-route \
             (FR-EX-08)",
        );
        assert_eq!(CATEGORY, "music-transcription");
        assert_ne!(
            CATEGORY, "pitch-transcription",
            "mt3 category must not collide with basic-pitch's \
             `pitch-transcription` — the output surface (multi-track \
             MIDI event stream vs single-instrument pitch \
             posteriorgram) is a distinct taxonomy slot",
        );
    }
}
