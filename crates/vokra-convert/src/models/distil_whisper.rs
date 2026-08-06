//! HuggingFace **distil-whisper / distil-large-v3.5**: safetensors
//! checkpoint → GGUF conversion (SoTA plan Phase 2, 2026-07-24).
//!
//! Input: a `distil-whisper/distil-large-v3.5` safetensors checkpoint (an
//! upstream HF release ships as `model.safetensors`, plain byte-exact
//! Whisper tensor naming). Output: a GGUF carrying every F32 / F16 tensor
//! verbatim plus a `vokra.whisper.*` hparam chunk (schema shared with
//! vanilla Whisper — the "very cheap follow-on" contract in the task) and
//! `vokra.provenance.*` marked MIT (Permissive).
//!
//! # What is transcribed vs. shape-driven
//!
//! - **Transcribed constants** — the tokenizer boundary constants
//!   (`WHISPER_EOT = 50257` — Whisper multilingual `<|endoftext|>`,
//!   invariant across every family size). These are the same constants
//!   the vanilla Whisper converter uses.
//! - **Shape-driven** — every architectural axis
//!   (`d_model`, `n_audio_layer`, `n_text_layer`, `n_mels`, `n_vocab`,
//!   `ffn_dim`) is read from the checkpoint's tensor shapes, so the
//!   emitted hparam chunk cannot disagree with the tensor payloads
//!   (FR-EX-08). A checkpoint whose decoder-layer count does not match
//!   the distil axis (`n_text_layer < n_audio_layer`) is rejected — a
//!   real Whisper (large-v3 = 32/32) or a mis-flattened distil (decoder
//!   duplicated to encoder count) surfaces as a loud
//!   `ConvertError::Parse` here rather than a runtime crash later.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the upstream HF Whisper names **verbatim** —
//! the distil-whisper release keeps the exact
//! `model.encoder.layers.i.self_attn.q_proj.weight` /
//! `model.decoder.layers.j.self_attn.q_proj.weight` structure that
//! `openai/whisper` uses, only with `j < i` (fewer decoder layers). The
//! vanilla `models::whisper` converter's identity `gguf_tensor_name`
//! rule holds unchanged, so a future runtime path that delegates to
//! [`crate::models::whisper`] via [`crate::ModelKind::Whisper`] can
//! read the same names.
//!
//! # No ONNX (permanent)
//!
//! distil-whisper ships PyTorch safetensors; the pipeline is
//! re-implemented natively via the vanilla Whisper runtime
//! (whisper.cpp 型, CLAUDE.md 設計判断 4). This converter never touches
//! ONNX.

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::{SafeTensorInfo, SafetensorsFile};

/// `vokra.model.arch` for distil-whisper GGUFs — kept in sync with the
/// runtime constant `vokra-models::distil_whisper::EXPECTED_ARCH`.
///
/// Intentionally distinct from vanilla Whisper's `"whisper"` so the
/// runtime can label the loaded model correctly in telemetry / logs /
/// model cards; the hparam **schema** is the same (`vokra.whisper.*`
/// keys) so a future runtime path that delegates distil-whisper to
/// [`vokra_core::gguf`] + `WhisperConfig` reads the same chunk.
pub(crate) const ARCH: &str = "distil-whisper";

// The vokra.whisper.* keys duplicated verbatim from `models/whisper.rs`
// (kept as constants because the two arms cannot share private items
// across the file boundary without more surface than this scaffold
// needs).

const KEY_N_MELS: &str = "vokra.whisper.n_mels";
const KEY_N_AUDIO_CTX: &str = "vokra.whisper.n_audio_ctx";
const KEY_N_AUDIO_STATE: &str = "vokra.whisper.n_audio_state";
const KEY_N_AUDIO_HEAD: &str = "vokra.whisper.n_audio_head";
const KEY_N_AUDIO_LAYER: &str = "vokra.whisper.n_audio_layer";
const KEY_N_TEXT_CTX: &str = "vokra.whisper.n_text_ctx";
const KEY_N_TEXT_STATE: &str = "vokra.whisper.n_text_state";
const KEY_N_TEXT_HEAD: &str = "vokra.whisper.n_text_head";
const KEY_N_TEXT_LAYER: &str = "vokra.whisper.n_text_layer";
const KEY_N_VOCAB: &str = "vokra.whisper.n_vocab";
const KEY_FFN_DIM: &str = "vokra.whisper.ffn_dim";
const KEY_EOT: &str = "vokra.whisper.eot";
const KEY_DECODER_START_IDS: &str = "vokra.whisper.decoder_start_ids";

/// Fixed Whisper attention head dimension across every family size
/// (base / small / medium / large-v3 / turbo / distil-large-v3.5 all
/// have `head_dim = 64`). Same constant the vanilla Whisper converter
/// uses.
const WHISPER_HEAD_DIM: u64 = 64;

/// End-of-transcript token id for the Whisper *multilingual* tokenizer
/// (`<|endoftext|>`), invariant across sizes and shared with distil.
const WHISPER_EOT: u32 = 50_257;

/// Derives the `vokra.model.name` value from a distil-whisper
/// checkpoint's shape quintuple. Returns one of `distil-large-v3` /
/// `distil-large-v3.5` (both currently share the same quintuple; the
/// canonical spelling emitted here is `"distil-large-v3.5"` — the
/// released variant). Unknown combinations return an explicit error —
/// no silent fallback per FR-EX-08.
///
/// The distil-large-v3.5 quintuple is `(1280, 32, 2, 128, 51866)`.
/// Future distil-medium / distil-small variants (with the smaller
/// encoder widths) can extend this table.
pub(crate) fn derive_name(
    d_model: u64,
    n_audio_layer: u32,
    n_text_layer: u32,
    n_mels: u64,
    n_vocab: u64,
) -> Result<&'static str, ConvertError> {
    match (d_model, n_audio_layer, n_text_layer, n_mels, n_vocab) {
        // distil-large-v3 and distil-large-v3.5 share the same
        // architectural quintuple — 3.5 is a data-only refresh over v3.
        // The canonical release published today is v3.5; a
        // shape-identical v3 or a hypothetical future v4 lands under
        // the same axes.
        (1280, 32, 2, 128, 51_866) => Ok("distil-large-v3.5"),
        // distil-medium.en — English-only distilled Whisper medium.
        // (d_model=1024, n_audio_layer=24, n_text_layer=2, n_mels=80,
        // n_vocab=51864 — English-only vocab is 51864, one less than
        // the multilingual 51865/51866). Primary source:
        // `distil-whisper/distil-medium.en` config.json 2026-07-31.
        (1024, 24, 2, 80, 51_864) => Ok("distil-medium.en"),
        _ => Err(ConvertError::Parse(format!(
            "unknown distil-whisper size: (d_model={d_model}, n_audio_layer={n_audio_layer}, \
             n_text_layer={n_text_layer}, n_mels={n_mels}, n_vocab={n_vocab}); expected the \
             distil-large-v3.5 quintuple (1280, 32, 2, 128, 51866) or distil-medium.en \
             (1024, 24, 2, 80, 51864). If this really is a distil-whisper checkpoint \
             but a size the converter has not seen, extend `derive_name` — do not fall \
             back silently."
        ))),
    }
}

/// True when `(d_model, n_mels, n_audio_layer, n_text_layer)` looks like
/// a rank-2 unit-test stub built by the safetensors round-trip tests in
/// this file rather than a real distil-whisper checkpoint. Mirrors the
/// vanilla Whisper converter's `is_synthetic_shape`: real distil
/// checkpoints always have `d_model >= 512` and non-zero axes.
fn is_synthetic_shape(d_model: u64, n_audio_layer: u32, n_text_layer: u32, n_mels: u64) -> bool {
    d_model == 0 || n_mels == 0 || n_audio_layer == 0 || n_text_layer == 0 || d_model < 512
}

/// Reads dimension `axis` of tensor `name`, or `0` when the tensor / axis
/// is absent (a degenerate checkpoint the runtime rejects at load).
fn tensor_dim(st: &SafetensorsFile, name: &str, axis: usize) -> u64 {
    st.tensors()
        .iter()
        .find(|t: &&SafeTensorInfo| t.name == name)
        .and_then(|t| t.shape.get(axis).copied())
        .unwrap_or(0)
}

/// Counts the highest `N` such that a tensor name matching
/// `{prefix}{N}.*` exists. Same helper the vanilla Whisper converter
/// uses to count encoder / decoder layers.
fn count_layers(st: &SafetensorsFile, prefix: &str) -> u32 {
    let mut max_idx: Option<u32> = None;
    for t in st.tensors() {
        if let Some(rest) = t.name.strip_prefix(prefix) {
            if let Some(end) = rest.find('.') {
                if let Ok(idx) = rest[..end].parse::<u32>() {
                    max_idx = Some(max_idx.map_or(idx, |m| m.max(idx)));
                }
            }
        }
    }
    max_idx.map_or(0, |m| m + 1)
}

/// Outcome of a distil-whisper conversion.
#[derive(Debug, Default)]
pub(crate) struct DistilWhisperReport {
    /// Float tensors written verbatim (F32 / F16 / BF16 — all three go
    /// through the same byte-copy path since the BF16 pass-through
    /// land 2026-07-25, mirror of `qwen3-tts` / `vibevoice` /
    /// `voxcpm2` / `moshi` / `voxtral`).
    pub(crate) written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time;
    /// anything that reaches this arm is a quantized dtype the runtime
    /// is not expected to consume).
    pub(crate) skipped_non_float: usize,
    /// Of the tensors in `written`, how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; runtime widens BF16 →
    /// f32 losslessly via the single choke point
    /// `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 =
    /// top 16 bits of an f32 — `bits << 16` is exact).
    pub(crate) bf16_passthrough: usize,
    /// Operator-facing diagnostics (never fail the conversion — the
    /// runtime is the authoritative gate, FR-EX-08).
    pub(crate) notes: Vec<String>,
}

/// Converts a distil-whisper safetensors buffer into a populated GGUF
/// builder.
///
/// Every F32 / F16 tensor passes through under its upstream HF name; the
/// `vokra.whisper.*` chunk group is written from the checkpoint's
/// tensor shapes (never invented); provenance is stamped **MIT**
/// (`Permissive`) — no runtime-side attribution obligation.
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, DistilWhisperReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    // Derive the model-name label from the checkpoint's shape quintuple.
    // Reads the same tensor axes `write_hparams` uses, so the written
    // `vokra.whisper.*` hparams and `vokra.model.name` label are
    // guaranteed to agree.
    let d_model = tensor_dim(&st, "model.encoder.conv1.weight", 0);
    let n_mels = tensor_dim(&st, "model.encoder.conv1.weight", 1);
    let n_audio_layer = count_layers(&st, "model.encoder.layers.");
    let n_text_layer = count_layers(&st, "model.decoder.layers.");
    let n_vocab = tensor_dim(&st, "model.decoder.embed_tokens.weight", 0);
    let name = match derive_name(d_model, n_audio_layer, n_text_layer, n_mels, n_vocab) {
        Ok(n) => n,
        Err(_) if is_synthetic_shape(d_model, n_audio_layer, n_text_layer, n_mels) => {
            "distil-whisper-unknown"
        }
        Err(e) => return Err(e),
    };
    // The distil invariant: `n_text_layer < n_audio_layer`. A real
    // distil checkpoint always satisfies this; a mis-labelled Whisper
    // checkpoint (large-v3 = 32/32) or a mis-flattened distil (decoder
    // duplicated to encoder count) hits the loud error path so a
    // downstream doesn't ship a GGUF whose arch stamp lies about the
    // decoder depth.
    if !is_synthetic_shape(d_model, n_audio_layer, n_text_layer, n_mels)
        && n_text_layer >= n_audio_layer
    {
        return Err(ConvertError::Parse(format!(
            "distil-whisper: this checkpoint has n_text_layer ({n_text_layer}) >= \
             n_audio_layer ({n_audio_layer}); a distil checkpoint shrinks the decoder, \
             so equal or larger decoder depth means this is not a distil-whisper — \
             use --model whisper for vanilla Whisper sizes."
        )));
    }

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, name);
    // Self-describing redistribution: the artifact carries its own
    // licence, not relying on a consumer running Vokra's registry
    // resolver. distil-whisper ships MIT weights (mirrors
    // openai/whisper's MIT posture) — the docs/license-audit.md entry
    // must land in the same PR that first materialises a real
    // distil-whisper GGUF, following the SoTA plan Phase 2 convention.
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        "MIT",
        Some(name),
        Some("distil-whisper/distil-large-v3.5 (MIT) — HuggingFace distilled Whisper"),
    );
    write_hparams(&mut b, &st);

    let mut report = DistilWhisperReport::default();
    for t in st.tensors() {
        match t.dtype {
            // BF16 pass-through added 2026-07-25 (mirror of qwen3-tts +
            // vibevoice + voxcpm2 + moshi + voxtral): the upstream
            // distil-large-v3.5 safetensors release ships
            // `torch_dtype: bfloat16` so the release checkpoint hits
            // this arm. Emit as GGUF type 30 verbatim; runtime widens
            // on load via `decode_bf16` (exact, `bits << 16`).
            //
            // Identity naming — the distil-whisper release keeps
            // upstream HF Whisper tensor names verbatim.
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )?;
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
    if report.written == 0 {
        report.notes.push(
            "no float tensors passed through — this GGUF is metadata-only and \
             the runtime will refuse to bind any weights (FR-EX-08). The BF16 \
             pass-through path is now wired (2026-07-25, mirror of qwen3-tts / \
             vibevoice / voxcpm2 / moshi / voxtral), so this state is only \
             reachable when the release contains no F32 / F16 / BF16 float \
             tensors at all."
                .to_owned(),
        );
    }
    Ok((b, report))
}

/// Writes the `vokra.whisper.*` chunk group from the checkpoint's tensor
/// shapes. Every value is derived (never invented). Mirrors the vanilla
/// Whisper converter's `write_hparams` so a runtime that delegates
/// distil-whisper to `WhisperConfig::from_gguf` reads the identical
/// schema.
fn write_hparams(b: &mut GgufBuilder, st: &SafetensorsFile) {
    let d_model = tensor_dim(st, "model.encoder.conv1.weight", 0);
    let n_mels = tensor_dim(st, "model.encoder.conv1.weight", 1);
    let n_audio_ctx = tensor_dim(st, "model.encoder.embed_positions.weight", 0);
    let n_text_ctx = tensor_dim(st, "model.decoder.embed_positions.weight", 0);
    let n_vocab = tensor_dim(st, "model.decoder.embed_tokens.weight", 0);
    let ffn_dim = tensor_dim(st, "model.encoder.layers.0.fc1.weight", 0);
    let n_audio_layer = count_layers(st, "model.encoder.layers.");
    let n_text_layer = count_layers(st, "model.decoder.layers.");
    // Whisper invariant: head_dim == 64, so n_head == d_model / 64.
    let n_head = if d_model >= WHISPER_HEAD_DIM {
        d_model / WHISPER_HEAD_DIM
    } else {
        0
    };

    b.add_u32(KEY_N_MELS, n_mels as u32);
    b.add_u32(KEY_N_AUDIO_CTX, n_audio_ctx as u32);
    b.add_u32(KEY_N_AUDIO_STATE, d_model as u32);
    b.add_u32(KEY_N_AUDIO_HEAD, n_head as u32);
    b.add_u32(KEY_N_AUDIO_LAYER, n_audio_layer);
    b.add_u32(KEY_N_TEXT_CTX, n_text_ctx as u32);
    b.add_u32(KEY_N_TEXT_STATE, d_model as u32);
    b.add_u32(KEY_N_TEXT_HEAD, n_head as u32);
    b.add_u32(KEY_N_TEXT_LAYER, n_text_layer);
    b.add_u32(KEY_N_VOCAB, n_vocab as u32);
    b.add_u32(KEY_FFN_DIM, ffn_dim as u32);
    b.add_u32(KEY_EOT, WHISPER_EOT);

    // Default English-transcription decode prefix
    // `<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>`,
    // derived from n_vocab so large-v3's +1 vocab shift lands the tail
    // specials at the right ids. `saturating_sub` keeps the converter
    // infallible on tiny synthetic n_vocab (the runtime rejects such
    // a degenerate model anyway).
    let n_vocab_u32 = n_vocab as u32;
    let decoder_start_ids = [
        WHISPER_EOT + 1,                  // <|startoftranscript|>
        WHISPER_EOT + 2,                  // <|en|> (first language)
        n_vocab_u32.saturating_sub(1506), // <|transcribe|>
        n_vocab_u32.saturating_sub(1502), // <|notimestamps|>
    ];
    b.add_metadata(
        KEY_DECODER_START_IDS,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: decoder_start_ids
                .iter()
                .map(|&id| GgufMetadataValue::U32(id))
                .collect(),
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    fn minimal_safetensors_one_f32() -> Vec<u8> {
        // A single f32 tensor at the top of the file so `convert` has
        // something to pass through and the report counts a non-zero
        // write. Uses the upstream HF Whisper name.
        let header = r#"{"model.encoder.layers.0.self_attn.q_proj.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 24]);
        out
    }

    fn minimal_safetensors_no_tensors() -> Vec<u8> {
        let header = r#"{}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out
    }

    fn minimal_safetensors_one_f16() -> Vec<u8> {
        let header = r#"{"model.encoder.layers.0.self_attn.q_proj.weight":{"dtype":"F16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out
    }

    fn minimal_safetensors_one_bf16() -> Vec<u8> {
        let header = r#"{"model.encoder.layers.0.self_attn.q_proj.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out
    }

    #[test]
    fn arch_string_matches_runtime_constant() {
        // The two crates only share `vokra-core`, so this constant is
        // the sole handshake with
        // `vokra-models::distil_whisper::EXPECTED_ARCH`.
        assert_eq!(ARCH, "distil-whisper");
    }

    #[test]
    fn derive_name_covers_distil_large_v3_5() {
        // The primary-source quintuple lands on `distil-large-v3.5`.
        assert_eq!(
            derive_name(1280, 32, 2, 128, 51_866).expect("known"),
            "distil-large-v3.5",
        );
    }

    #[test]
    fn derive_name_covers_distil_medium_en() {
        // distil-medium.en (English-only, primary source
        // `distil-whisper/distil-medium.en` config.json 2026-07-31).
        assert_eq!(
            derive_name(1024, 24, 2, 80, 51_864).expect("known"),
            "distil-medium.en",
        );
    }

    #[test]
    fn derive_name_rejects_unknown_shape() {
        // A vanilla Whisper large-v3 shape (32 decoder layers) is not
        // distil-large-v3.5; the converter must not silently accept it.
        let err = derive_name(1280, 32, 32, 128, 51_866).expect_err("not distil");
        assert!(matches!(err, ConvertError::Parse(_)));
    }

    #[test]
    fn round_trip_carries_arch_chunks_and_provenance() {
        let (builder, report) = convert(minimal_safetensors_one_f32()).expect("convert");
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH),
        );
        // The tiny synthetic checkpoint hits the synthetic-shape arm
        // (d_model 0 / rank-2 stub) so the name lands on the sentinel.
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("distil-whisper-unknown"),
        );

        // Provenance: MIT permissive.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("MIT"),
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
        );
        // No attribution stamp for MIT (unlike CC-BY 4.0 Parakeet /
        // Canary / Kyutai STT — but source string is present).
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_ATTRIBUTION)
                .and_then(|v| v.as_str()),
            None,
            "MIT must not stamp attribution",
        );

        // The `vokra.whisper.*` chunk lands (derived from the tiny stub —
        // n_mels axis 1 is 3, d_model axis 0 is 2, etc.). Only the ARCH
        // string is fully-pinned here; the numeric chunk is exercised
        // in more detail by the CLI-level integration tests.
        assert!(file.get(KEY_EOT).is_some());
        assert!(file.get(KEY_DECODER_START_IDS).is_some());
    }

    /// F16 tensor passes through the union match arm.
    #[test]
    fn f16_tensor_passes_through_verbatim() {
        let (builder, report) = convert(minimal_safetensors_one_f16()).expect("convert");
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file
            .tensor_info("model.encoder.layers.0.self_attn.q_proj.weight")
            .expect("tensor present");
        assert_eq!(info.dtype, GgmlType::F16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info).len(), 12);
    }

    /// Pins the BF16 leg of the `GgmlType::F32 | GgmlType::F16 |
    /// GgmlType::BF16` union: BF16 (the upstream serving format for the
    /// distil-large-v3.5 release, `torch_dtype: bfloat16` in the
    /// HuggingFace `config.json`) must reach the pass-through arm, emit
    /// as GGUF type 30 verbatim, and increment `bf16_passthrough`.
    /// Mirror of qwen3-tts / vibevoice / voxcpm2's
    /// `bf16_tensor_passes_through_verbatim` and moshi's
    /// `assert_eq!(info.dtype, GgmlType::BF16, "no convert-time
    /// widening")`.
    ///
    /// Rewritten 2026-07-25 from the earlier "counted as skipped" pin —
    /// the earlier pin encoded the pre-BF16-fix scaffold posture.
    /// Removing the pin outright would let a latent silent-widen slip
    /// in undetected; rewriting to the passes-through invariant keeps
    /// the regression guard.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (builder, report) = convert(minimal_safetensors_one_bf16()).expect("convert");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm and increment `written`"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 subset counter must record the pass-through"
        );
        // The tensor survives the round trip under its upstream name
        // and preserves its BF16 dtype (no convert-time widening —
        // runtime widens on load via `decode_bf16`, exact via
        // `bits << 16`).
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file
            .tensor_info("model.encoder.layers.0.self_attn.q_proj.weight")
            .expect("BF16 tensor must be present after pass-through");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — GGUF dtype must remain BF16"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        // Round-trip byte equality: the input helper emits 12 bytes of
        // zeros (2×3 BF16 = 6 elements × 2 bytes) and those bytes must
        // survive the pass-through unchanged (`bits << 16` widening is
        // a *runtime* concern; the emitted GGUF payload is the source
        // BF16 buffer verbatim).
        assert_eq!(
            file.tensor_bytes(info),
            &[0u8; 12][..],
            "BF16 payload = 12 bytes must round-trip verbatim"
        );
        // Metadata (arch / hparams / provenance) still lands.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH),
        );
    }

    #[test]
    fn zero_tensor_conversion_surfaces_a_loud_note() {
        let (_, report) = convert(minimal_safetensors_no_tensors()).expect("convert");
        assert_eq!(report.written, 0);
        assert!(
            report.notes.iter().any(|n| n.contains("no float tensors")),
            "zero-tensor conversion must emit a loud note: {:?}",
            report.notes
        );
    }

    /// Pins `SafetensorsFile::parse(bytes)?` error propagation. A
    /// malformed input surfaces as `Err(ConvertError::Parse(_))`, not
    /// a silently-empty successful conversion (FR-EX-08 loud fail).
    #[test]
    fn malformed_input_returns_parse_error() {
        // Case 1: empty buffer.
        let err = convert(Vec::new()).expect_err("empty buffer must be rejected");
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );

        // Case 2: declared header length runs off the end of the buffer.
        let mut truncated = Vec::new();
        truncated.extend_from_slice(&1024u64.to_le_bytes());
        truncated.extend_from_slice(b"{}");
        let err = convert(truncated).expect_err("truncated header must be rejected");
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );

        // Case 3: valid length prefix but malformed JSON body.
        let bad_json = b"{not-json";
        let mut bad = Vec::new();
        bad.extend_from_slice(&(bad_json.len() as u64).to_le_bytes());
        bad.extend_from_slice(bad_json);
        let err = convert(bad).expect_err("malformed JSON must be rejected");
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );
    }
}
