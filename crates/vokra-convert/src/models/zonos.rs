//! Zyphra **Zonos-v0.1-transformer**: safetensors checkpoint → GGUF conversion
//! (SoTA plan Phase 1-5, 2026-07-24).
//!
//! Input: an upstream `Zyphra/Zonos-v0.1-transformer` safetensors checkpoint
//! (Apache 2.0 code + weight, docs/license-audit.md). The reference release
//! ships safetensors directly; no `.pth` prepare step is required (unlike
//! Dia). Output: a GGUF carrying every float tensor verbatim plus the
//! `vokra.zonos.*` / `vokra.provenance.*` metadata chunks.
//!
//! # What is transcribed vs. shape-driven
//!
//! - **Transcribed constants** — every hparam of the `vokra.zonos.*` chunk
//!   group is transcribed **verbatim** from the upstream `config.json`
//!   (see the top of this module for the full table). No axis is invented;
//!   any tensor whose shape disagrees with these values in a real conversion
//!   fails the runtime shape gate loudly (FR-EX-08,
//!   `ZonosConfig::validate_for_forward`).
//! - **Runtime-supplied** — the DAC 44.1 kHz codec (`vokra.dac.*`) travels
//!   in a separate standalone codec GGUF (M4-04 T11), *not* embedded here.
//!   Zonos and DAC are two independent Apache 2.0 / MIT projects; keeping
//!   them as two GGUFs preserves the M2-13 provenance chain and lets a
//!   caller mix & match Zonos weights with the same DAC 44.1 kHz codec Dia
//!   uses.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the upstream safetensors names **verbatim** (the
//! CSM / Kokoro / CosyVoice2 / Dia contract). Real-weight binding is a
//! follow-up wave gated on the upstream tensor-name manifest fetch; this
//! converter passes every F32 / F16 tensor through unchanged so a future
//! `ZonosWeights::from_gguf` can walk the same names.
//!
//! # No ONNX (permanent)
//!
//! Zonos ships as safetensors / a Python pipeline; the pipeline is
//! re-implemented natively in `vokra-models/src/zonos/` (whisper.cpp 型,
//! CLAUDE.md 設計判断 4). This converter never touches ONNX.

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, GgufMetadataValue, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Zonos GGUFs — kept in sync with the runtime
/// constant `vokra-models::zonos::EXPECTED_ARCH`.
pub(crate) const ARCH: &str = "zonos";
/// `vokra.model.name` for Zonos GGUFs.
pub(crate) const NAME: &str = "zonos-v0.1";

// --- vokra.zonos.* keys (kept as constants in the converter; the runtime
// duplicates the strings in `crates/vokra-models/src/zonos/mod.rs` — a
// round-trip test on the converter side catches drift, following the
// cross-crate pattern established by CSM / CosyVoice2 / Kokoro / Dia) ---

const KEY_SAMPLE_RATE: &str = "vokra.zonos.sample_rate";

// Backbone
const KEY_BB_N_LAYER: &str = "vokra.zonos.arch.backbone.n_layer";
const KEY_BB_D_MODEL: &str = "vokra.zonos.arch.backbone.d_model";
const KEY_BB_D_INTERMEDIATE: &str = "vokra.zonos.arch.backbone.d_intermediate";
const KEY_BB_NUM_HEADS: &str = "vokra.zonos.arch.backbone.num_heads";
const KEY_BB_NUM_HEADS_KV: &str = "vokra.zonos.arch.backbone.num_heads_kv";
const KEY_BB_ROTARY_EMB_DIM: &str = "vokra.zonos.arch.backbone.rotary_emb_dim";
const KEY_BB_ROTARY_EMB_INTERLEAVED: &str = "vokra.zonos.arch.backbone.rotary_emb_interleaved";
const KEY_BB_CAUSAL: &str = "vokra.zonos.arch.backbone.causal";
const KEY_BB_QKV_PROJ_BIAS: &str = "vokra.zonos.arch.backbone.qkv_proj_bias";
const KEY_BB_OUT_PROJ_BIAS: &str = "vokra.zonos.arch.backbone.out_proj_bias";
const KEY_BB_NORM_EPSILON: &str = "vokra.zonos.arch.backbone.norm_epsilon";
const KEY_BB_RMS_NORM: &str = "vokra.zonos.arch.backbone.rms_norm";

// Vocab / codebook / heads
const KEY_NUM_CODEBOOKS: &str = "vokra.zonos.num_codebooks";
const KEY_CODEBOOK_VOCAB: &str = "vokra.zonos.codebook_vocab";
const KEY_HEAD_VOCAB: &str = "vokra.zonos.head_vocab";
const KEY_EOS_TOKEN_ID: &str = "vokra.zonos.eos_token_id";
const KEY_MASKED_TOKEN_ID: &str = "vokra.zonos.masked_token_id";

// Delay pattern (indexed keys — the CSM / Mimi pattern for array metadata)
const KEY_DELAY_PATTERN_COUNT: &str = "vokra.zonos.delay_pattern_count";
const PREFIX_DELAY_PATTERN: &str = "vokra.zonos.delay_pattern.";

// Prefix-conditioner descriptor (indexed keys, one group per conditioner)
const KEY_CONDITIONER_COUNT: &str = "vokra.zonos.prefix_conditioner.count";
const PREFIX_CONDITIONER: &str = "vokra.zonos.prefix_conditioner.";

// --- Transcribed constants (primary source: config.json fetched verbatim) --
//
// Zyphra/Zonos-v0.1-transformer config.json
// (huggingface.co/Zyphra/Zonos-v0.1-transformer/raw/main/config.json). Every
// value here is transcribed verbatim; nothing is invented.

// PCM sample rate — not written in config.json; inherited from DAC 44.1 kHz
// (the codec Zonos loads via `DacModel.from_pretrained("descript/dac_44khz")`
// upstream in `zonos/autoencoder.py::DACAutoencoder.__init__`).
const ZONOS_SAMPLE_RATE: u32 = 44_100;

// Backbone
const BB_N_LAYER: u32 = 26;
const BB_D_MODEL: u32 = 2048;
const BB_D_INTERMEDIATE: u32 = 8192;
const BB_NUM_HEADS: u32 = 16;
const BB_NUM_HEADS_KV: u32 = 4;
const BB_ROTARY_EMB_DIM: u32 = 128;
const BB_ROTARY_EMB_INTERLEAVED: bool = true;
const BB_CAUSAL: bool = true;
const BB_QKV_PROJ_BIAS: bool = false;
const BB_OUT_PROJ_BIAS: bool = false;
const BB_NORM_EPSILON: f32 = 1e-5;
const BB_RMS_NORM: bool = false;

// Vocab / codebook / heads
const NUM_CODEBOOKS: u32 = 9;
const CODEBOOK_VOCAB: u32 = 1026;
const HEAD_VOCAB: u32 = 1025;
const EOS_TOKEN_ID: u32 = 1024;
const MASKED_TOKEN_ID: u32 = 1025;

// Delay pattern from `zonos/codebook_pattern.py::apply_delay_pattern`:
// codebook k is rolled by k+1 → [1, 2, ..., NUM_CODEBOOKS].
const DELAY_PATTERN: [u32; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];

/// One typed conditioner descriptor as it rides in the GGUF (name +
/// discriminant + numeric bounds where they apply). This mirrors the
/// runtime `ZonosConditionerKind` enum but is expressed as `u32`s / a
/// short string so the GGUF metadata surface stays primitive.
struct ConditionerDesc {
    name: &'static str,
    /// Discriminant string: `"espeak"` | `"speaker"` | `"fourier"` |
    /// `"integer"`.
    kind: &'static str,
    /// For `speaker`: `cond_dim`; for `fourier`: `input_dim`; unused
    /// otherwise.
    input_dim: u32,
    /// `min_val` (fourier scalars). `0.0` where inapplicable.
    min_val_f: f32,
    /// `max_val` (fourier scalars). `0.0` where inapplicable.
    max_val_f: f32,
    /// `min_val` (integer conditioners). `0` where inapplicable.
    min_val_i: i32,
    /// `max_val` (integer conditioners). `0` where inapplicable.
    max_val_i: i32,
}

/// The 7 typed prefix conditioners, verbatim from `config.prefix_conditioner
/// .conditioners`.
const CONDITIONERS: [ConditionerDesc; 7] = [
    ConditionerDesc {
        name: "espeak",
        kind: "espeak",
        input_dim: 0,
        min_val_f: 0.0,
        max_val_f: 0.0,
        min_val_i: 0,
        max_val_i: 0,
    },
    ConditionerDesc {
        name: "speaker",
        kind: "speaker",
        input_dim: 128,
        min_val_f: 0.0,
        max_val_f: 0.0,
        min_val_i: 0,
        max_val_i: 0,
    },
    ConditionerDesc {
        name: "emotion",
        kind: "fourier",
        input_dim: 8,
        min_val_f: 0.0,
        max_val_f: 0.0,
        min_val_i: 0,
        max_val_i: 0,
    },
    ConditionerDesc {
        name: "fmax",
        kind: "fourier",
        input_dim: 1,
        min_val_f: 0.0,
        max_val_f: 24_000.0,
        min_val_i: 0,
        max_val_i: 0,
    },
    ConditionerDesc {
        name: "pitch_std",
        kind: "fourier",
        input_dim: 1,
        min_val_f: 0.0,
        max_val_f: 400.0,
        min_val_i: 0,
        max_val_i: 0,
    },
    ConditionerDesc {
        name: "speaking_rate",
        kind: "fourier",
        input_dim: 1,
        min_val_f: 0.0,
        max_val_f: 40.0,
        min_val_i: 0,
        max_val_i: 0,
    },
    ConditionerDesc {
        name: "language_id",
        kind: "integer",
        input_dim: 0,
        min_val_f: 0.0,
        max_val_f: 0.0,
        min_val_i: -1,
        max_val_i: 126,
    },
];

/// Outcome of a Zonos conversion.
#[derive(Debug, Default)]
pub(crate) struct ZonosReport {
    /// Float tensors written verbatim.
    pub(crate) written: usize,
    /// Non-F32 / F16 tensors skipped (defensive counter — the safetensors
    /// reader rejects unknown dtypes at parse time).
    pub(crate) skipped_non_float: usize,
    /// Operator-facing diagnostics (never fail the conversion — the
    /// runtime is the authoritative gate, FR-EX-08).
    pub(crate) notes: Vec<String>,
}

/// Converts a Zonos safetensors buffer into a populated GGUF builder.
///
/// Every F32 / F16 tensor passes through under its upstream name; the
/// `vokra.zonos.*` chunk group is written from the transcribed constants
/// above; provenance stamps mark the weight as `Permissive` (Apache 2.0).
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, ZonosReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    write_hparams(&mut b);
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        "Apache-2.0",
        Some("Zyphra/Zonos-v0.1-transformer"),
        Some("huggingface"),
    );

    let mut report = ZonosReport::default();
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
    if report.written == 0 {
        report.notes.push(
            "no float tensors passed through — this GGUF is metadata-only and \
             the runtime will refuse to bind any weights (FR-EX-08). The upstream \
             Zonos-v0.1-transformer release ships safetensors directly; check that \
             the input path is a Zonos safetensors and not a config-only shard."
                .into(),
        );
    }
    Ok((b, report))
}

/// Writes the `vokra.zonos.*` chunk group from the transcribed constants
/// above (primary source: `config.json`). Delay pattern and the 7
/// prefix-conditioner descriptors ride as count + N indexed keys (the CSM
/// / mimi pattern for array metadata).
fn write_hparams(b: &mut GgufBuilder) {
    b.add_u32(KEY_SAMPLE_RATE, ZONOS_SAMPLE_RATE);

    // Backbone
    b.add_u32(KEY_BB_N_LAYER, BB_N_LAYER);
    b.add_u32(KEY_BB_D_MODEL, BB_D_MODEL);
    b.add_u32(KEY_BB_D_INTERMEDIATE, BB_D_INTERMEDIATE);
    b.add_u32(KEY_BB_NUM_HEADS, BB_NUM_HEADS);
    b.add_u32(KEY_BB_NUM_HEADS_KV, BB_NUM_HEADS_KV);
    b.add_u32(KEY_BB_ROTARY_EMB_DIM, BB_ROTARY_EMB_DIM);
    // Booleans ride as u32 0/1 for GGUF portability (matches the CSM
    // scalar-flag convention).
    b.add_u32(
        KEY_BB_ROTARY_EMB_INTERLEAVED,
        u32::from(BB_ROTARY_EMB_INTERLEAVED),
    );
    b.add_u32(KEY_BB_CAUSAL, u32::from(BB_CAUSAL));
    b.add_u32(KEY_BB_QKV_PROJ_BIAS, u32::from(BB_QKV_PROJ_BIAS));
    b.add_u32(KEY_BB_OUT_PROJ_BIAS, u32::from(BB_OUT_PROJ_BIAS));
    b.add_f32(KEY_BB_NORM_EPSILON, BB_NORM_EPSILON);
    b.add_u32(KEY_BB_RMS_NORM, u32::from(BB_RMS_NORM));

    // Vocab / codebook / heads
    b.add_u32(KEY_NUM_CODEBOOKS, NUM_CODEBOOKS);
    b.add_u32(KEY_CODEBOOK_VOCAB, CODEBOOK_VOCAB);
    b.add_u32(KEY_HEAD_VOCAB, HEAD_VOCAB);
    b.add_u32(KEY_EOS_TOKEN_ID, EOS_TOKEN_ID);
    b.add_u32(KEY_MASKED_TOKEN_ID, MASKED_TOKEN_ID);

    // Delay pattern
    b.add_u32(KEY_DELAY_PATTERN_COUNT, DELAY_PATTERN.len() as u32);
    for (i, d) in DELAY_PATTERN.iter().enumerate() {
        b.add_u32(&format!("{PREFIX_DELAY_PATTERN}{i}"), *d);
    }

    // Prefix conditioners — count + one group per conditioner. Each group
    // has {name, kind, input_dim, min_val_f, max_val_f, min_val_i, max_val_i}.
    b.add_u32(KEY_CONDITIONER_COUNT, CONDITIONERS.len() as u32);
    for (i, c) in CONDITIONERS.iter().enumerate() {
        let base = format!("{PREFIX_CONDITIONER}{i}");
        b.add_string(&format!("{base}.name"), c.name);
        b.add_string(&format!("{base}.kind"), c.kind);
        b.add_u32(&format!("{base}.input_dim"), c.input_dim);
        b.add_f32(&format!("{base}.min_val_f"), c.min_val_f);
        b.add_f32(&format!("{base}.max_val_f"), c.max_val_f);
        b.add_metadata(
            &format!("{base}.min_val_i"),
            GgufMetadataValue::I32(c.min_val_i),
        );
        b.add_metadata(
            &format!("{base}.max_val_i"),
            GgufMetadataValue::I32(c.max_val_i),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    fn minimal_safetensors_one_f32() -> Vec<u8> {
        // A single f32 tensor at the top of the file so `convert` has
        // something to pass through and the report counts a non-zero write.
        let header = r#"{"backbone.embeddings.0.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
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
        // Single f16 tensor with a distinctive, non-zero payload so the
        // byte-verbatim round-trip test can prove no silent truncation /
        // byte-swap happened: shape [2, 3] × sizeof(f16) = 12 bytes. The
        // half-precision encodings 0x3C00…0x4600 decode to 1.0, 2.0, 3.0,
        // 4.0, 5.0, 6.0 respectively, but the converter never dequantizes
        // — the raw u16 bytes are what round-trips.
        let payload: [u16; 6] = [0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600];
        let mut data_region = Vec::new();
        for v in payload {
            data_region.extend_from_slice(&v.to_le_bytes());
        }
        let header = r#"{"backbone.embeddings.0.weight":{"dtype":"F16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&data_region);
        out
    }

    fn minimal_safetensors_one_bf16() -> Vec<u8> {
        // Single bf16 tensor: the `vokra-core` safetensors reader accepts
        // BF16 (M4-06, the all-BF16 `kyutai/moshiko` release), but the
        // Zonos `convert()` loop only pipes F32 / F16 through — so this
        // fixture exercises the `_ => report.skipped_non_float += 1` arm.
        // Shape [2, 3] × sizeof(bf16) = 12 bytes (same byte span as F16).
        let header = r#"{"backbone.embeddings.0.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out
    }

    fn minimal_safetensors_duplicate_name() -> Vec<u8> {
        // Two tensor entries with the same key. The `vokra-core` JSON
        // parser preserves duplicate keys (objects are stored as
        // `Vec<(String, JsonValue)>`, not a map), and the safetensors
        // header parser pushes both descriptors onto its `Vec` (the
        // secondary name index is a HashMap that overwrites, but
        // `.tensors()` returns the full Vec). The Zonos converter walks
        // `.tensors()`, so the second `GgufBuilder::add_tensor` call hits
        // `GgufError::DuplicateTensor`. Two F32 shape-[1] tensors, 4
        // bytes each, contiguous data region.
        let header = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 8]);
        out
    }

    #[test]
    fn arch_string_matches_runtime_constant() {
        // The two crates only share `vokra-core`, so this constant is the
        // sole handshake with `vokra-models::zonos::EXPECTED_ARCH`.
        assert_eq!(ARCH, "zonos");
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
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        // Every transcribed U32 hparam round-trips verbatim.
        for (key, want) in [
            (KEY_SAMPLE_RATE, ZONOS_SAMPLE_RATE),
            (KEY_BB_N_LAYER, BB_N_LAYER),
            (KEY_BB_D_MODEL, BB_D_MODEL),
            (KEY_BB_D_INTERMEDIATE, BB_D_INTERMEDIATE),
            (KEY_BB_NUM_HEADS, BB_NUM_HEADS),
            (KEY_BB_NUM_HEADS_KV, BB_NUM_HEADS_KV),
            (KEY_BB_ROTARY_EMB_DIM, BB_ROTARY_EMB_DIM),
            (
                KEY_BB_ROTARY_EMB_INTERLEAVED,
                u32::from(BB_ROTARY_EMB_INTERLEAVED),
            ),
            (KEY_BB_CAUSAL, u32::from(BB_CAUSAL)),
            (KEY_BB_QKV_PROJ_BIAS, u32::from(BB_QKV_PROJ_BIAS)),
            (KEY_BB_OUT_PROJ_BIAS, u32::from(BB_OUT_PROJ_BIAS)),
            (KEY_BB_RMS_NORM, u32::from(BB_RMS_NORM)),
            (KEY_NUM_CODEBOOKS, NUM_CODEBOOKS),
            (KEY_CODEBOOK_VOCAB, CODEBOOK_VOCAB),
            (KEY_HEAD_VOCAB, HEAD_VOCAB),
            (KEY_EOS_TOKEN_ID, EOS_TOKEN_ID),
            (KEY_MASKED_TOKEN_ID, MASKED_TOKEN_ID),
            (KEY_DELAY_PATTERN_COUNT, DELAY_PATTERN.len() as u32),
            (KEY_CONDITIONER_COUNT, CONDITIONERS.len() as u32),
        ] {
            match file.get(key) {
                Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, want, "{key}"),
                other => panic!("{key}: unexpected {other:?}"),
            }
        }
        // Delay pattern indexed keys.
        for (i, want) in DELAY_PATTERN.iter().enumerate() {
            let k = format!("{PREFIX_DELAY_PATTERN}{i}");
            match file.get(&k) {
                Some(GgufMetadataValue::U32(v)) => assert_eq!(v, want, "{k}"),
                other => panic!("{k}: unexpected {other:?}"),
            }
        }
        // F32 hparams.
        match file.get(KEY_BB_NORM_EPSILON) {
            Some(GgufMetadataValue::F32(v)) => assert_eq!(*v, BB_NORM_EPSILON),
            other => panic!("{KEY_BB_NORM_EPSILON}: unexpected {other:?}"),
        }
        // Prefix conditioner descriptors — every field round-trips.
        for (i, c) in CONDITIONERS.iter().enumerate() {
            let base = format!("{PREFIX_CONDITIONER}{i}");
            assert_eq!(
                file.get(&format!("{base}.name")).and_then(|v| v.as_str()),
                Some(c.name),
                "{base}.name"
            );
            assert_eq!(
                file.get(&format!("{base}.kind")).and_then(|v| v.as_str()),
                Some(c.kind),
                "{base}.kind"
            );
            match file.get(&format!("{base}.input_dim")) {
                Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, c.input_dim, "{base}.input_dim"),
                other => panic!("{base}.input_dim: unexpected {other:?}"),
            }
            match file.get(&format!("{base}.min_val_f")) {
                Some(GgufMetadataValue::F32(v)) => assert_eq!(*v, c.min_val_f, "{base}.min_val_f"),
                other => panic!("{base}.min_val_f: unexpected {other:?}"),
            }
            match file.get(&format!("{base}.max_val_f")) {
                Some(GgufMetadataValue::F32(v)) => assert_eq!(*v, c.max_val_f, "{base}.max_val_f"),
                other => panic!("{base}.max_val_f: unexpected {other:?}"),
            }
            match file.get(&format!("{base}.min_val_i")) {
                Some(GgufMetadataValue::I32(v)) => assert_eq!(*v, c.min_val_i, "{base}.min_val_i"),
                other => panic!("{base}.min_val_i: unexpected {other:?}"),
            }
            match file.get(&format!("{base}.max_val_i")) {
                Some(GgufMetadataValue::I32(v)) => assert_eq!(*v, c.max_val_i, "{base}.max_val_i"),
                other => panic!("{base}.max_val_i: unexpected {other:?}"),
            }
        }
        // Provenance: Apache 2.0 permissive.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some("Zyphra/Zonos-v0.1-transformer")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("Apache-2.0")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
    }

    #[test]
    fn zero_tensor_conversion_surfaces_a_loud_note() {
        // Empty safetensors → the runtime's `ZonosWeights::from_gguf` would
        // fail loudly at bind time, but the converter itself succeeds and
        // reports the situation so the operator sees it now.
        let (_, report) = convert(minimal_safetensors_no_tensors()).expect("convert");
        assert_eq!(report.written, 0);
        assert!(
            report.notes.iter().any(|n| n.contains("no float tensors")),
            "zero-tensor conversion must emit a loud note: {:?}",
            report.notes
        );
    }

    /// The seven conditioner descriptors match the upstream `config.json`
    /// `prefix_conditioner.conditioners` list one-for-one (name, kind and
    /// numeric bounds).
    #[test]
    fn conditioner_descriptors_match_primary_source() {
        let names: Vec<&str> = CONDITIONERS.iter().map(|c| c.name).collect();
        assert_eq!(
            names,
            vec![
                "espeak",
                "speaker",
                "emotion",
                "fmax",
                "pitch_std",
                "speaking_rate",
                "language_id"
            ]
        );
        assert_eq!(CONDITIONERS[0].kind, "espeak");
        assert_eq!(CONDITIONERS[1].kind, "speaker");
        assert_eq!(CONDITIONERS[1].input_dim, 128);
        assert_eq!(CONDITIONERS[3].kind, "fourier");
        assert_eq!(CONDITIONERS[3].max_val_f, 24_000.0);
        assert_eq!(CONDITIONERS[6].kind, "integer");
        assert_eq!(CONDITIONERS[6].min_val_i, -1);
        assert_eq!(CONDITIONERS[6].max_val_i, 126);
    }

    /// Pins the `GgmlType::F16` half of the `match` arm at the top of the
    /// `convert()` tensor loop: the existing `round_trip_carries_arch_chunks
    /// _and_provenance` test only fed F32, leaving ~50% of the passthrough
    /// arm untested. A refactor that accidentally routed F16 out of the
    /// verbatim path (e.g. into a quantize branch) would flip
    /// `written == 1` to `written == 0` here.
    #[test]
    fn f16_tensor_passes_through_and_counts_written() {
        let (_, report) = convert(minimal_safetensors_one_f16()).expect("convert");
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert!(
            report.notes.is_empty(),
            "F16 pass-through must not fire the loud note: {:?}",
            report.notes
        );
    }

    /// Pins the `_ => report.skipped_non_float += 1` arm at line 259: the
    /// `vokra-core` safetensors reader graduated BF16 from "future
    /// extension" to "supported" in M4-06, so the arm is genuinely
    /// reachable — contradicting the `ZonosReport::skipped_non_float`
    /// doc comment that calls it a "defensive counter". This also covers
    /// the second path into the loud note (`written == 0` via a skipped
    /// non-float tensor) which the existing `zero_tensor_conversion_
    /// surfaces_a_loud_note` test did not exercise (that one hits an
    /// empty `{}` header).
    #[test]
    fn bf16_tensor_increments_skipped_and_surfaces_loud_note() {
        let (_, report) = convert(minimal_safetensors_one_bf16()).expect("convert");
        assert_eq!(report.written, 0);
        assert_eq!(report.skipped_non_float, 1);
        assert!(
            report.notes.iter().any(|n| n.contains("no float tensors")),
            "BF16-only conversion must emit the loud note: {:?}",
            report.notes
        );
    }

    /// Pins the `?` on `SafetensorsFile::parse(bytes)?` at line 232: any
    /// buffer shorter than the 8-byte header-length prefix returns
    /// `SafetensorsError::Truncated`, which the `?` must forward as
    /// `ConvertError::Parse` (FR-EX-08: no silent success, no panic).
    /// Covers both the "garbage bytes" error branch and the "empty
    /// bytes" edge case in one shot — both hit the same `data.len() < 8`
    /// check, and a regression that swallowed either would show here.
    #[test]
    fn short_buffers_propagate_safetensors_parse_error() {
        for garbage in [Vec::new(), vec![0u8; 4], vec![0u8; 7]] {
            let n = garbage.len();
            let err = convert(garbage).expect_err(&format!("len {n}: expected error, got Ok"));
            match err {
                ConvertError::Parse(_) => {}
                other => panic!("len {n}: expected ConvertError::Parse, got {other:?}"),
            }
        }
    }

    /// Pins the `?` on `b.add_tensor(...)?` at line 255: the safetensors
    /// JSON header parser preserves duplicate keys (see the fixture
    /// doc), so `st.tensors()` yields both entries; the second
    /// `GgufBuilder::add_tensor` call must then hit
    /// `GgufError::DuplicateTensor`, which the `?` must forward as
    /// `ConvertError::Gguf`. Without this test a future refactor that
    /// silently deduped tensor names would ship broken (FR-EX-08).
    #[test]
    fn duplicate_tensor_name_propagates_gguf_error() {
        let err = convert(minimal_safetensors_duplicate_name())
            .expect_err("duplicate tensor name must error");
        match err {
            ConvertError::Gguf(msg) => {
                assert!(
                    msg.to_lowercase().contains("duplicate"),
                    "unexpected Gguf error message: {msg}"
                );
            }
            other => panic!("expected ConvertError::Gguf, got {other:?}"),
        }
    }

    /// Pins the module doc contract on lines 27-31: "GGUF tensor names
    /// are the upstream safetensors names verbatim" and "passes every
    /// F32 / F16 tensor through unchanged". The existing round-trip
    /// test asserts `report.written == 1` but never re-parses the output
    /// GGUF, so a regression that silently renamed the tensor, swapped
    /// shape order, or byte-scrambled the payload would still pass the
    /// metadata assertions. This walks the fixture through the
    /// safetensors reader to capture the input (name, dtype, shape,
    /// bytes) triple, then converts, serializes, re-parses, and diffs
    /// every field. Uses the F16 fixture because its payload is a
    /// distinctive non-zero bit pattern (0x3C00…0x4600) that a partial
    /// truncation or byte-swap cannot silently mimic.
    #[test]
    fn round_trip_carries_tensor_bytes_verbatim() {
        let bytes = minimal_safetensors_one_f16();
        let st = SafetensorsFile::parse(bytes.clone()).expect("parse safetensors");
        let (input_name, input_shape, input_dtype, input_payload) = {
            let t = st.tensors().first().expect("one tensor in fixture");
            (
                t.name.clone(),
                t.shape.clone(),
                t.dtype,
                st.tensor_bytes(t).to_vec(),
            )
        };

        let (builder, report) = convert(bytes).expect("convert");
        assert_eq!(report.written, 1);

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");

        let info = file
            .tensor_info(&input_name)
            .unwrap_or_else(|| panic!("output GGUF must carry tensor `{input_name}`"));
        assert_eq!(info.dtype, input_dtype, "dtype must be preserved verbatim");
        assert_eq!(
            info.dimensions, input_shape,
            "shape must be preserved verbatim"
        );
        assert_eq!(
            file.tensor_bytes(info),
            input_payload.as_slice(),
            "payload bytes must round-trip byte-for-byte (verbatim contract)"
        );
    }

    /// Pins the invariant documented at lines 119-121: "codebook k is
    /// rolled by k+1 → [1, 2, ..., NUM_CODEBOOKS]". If a future v0.2
    /// bumped `NUM_CODEBOOKS` without extending `DELAY_PATTERN` (or vice
    /// versa) the drift would surface only at model bind time as a
    /// runtime shape mismatch; this asserts the k+1 rule at CI time so
    /// the converter never emits a `vokra.zonos.delay_pattern_count`
    /// that disagrees with `vokra.zonos.num_codebooks`.
    #[test]
    fn delay_pattern_length_matches_num_codebooks() {
        assert_eq!(NUM_CODEBOOKS as usize, DELAY_PATTERN.len());
        for (i, &d) in DELAY_PATTERN.iter().enumerate() {
            assert_eq!(
                d,
                (i + 1) as u32,
                "delay_pattern[{i}] must be {} (k+1 rule)",
                i + 1
            );
        }
    }
}
