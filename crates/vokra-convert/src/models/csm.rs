//! Sesame CSM-1B: safetensors checkpoint → GGUF conversion
//! (M4-05-T03 skeleton / T04 `vokra.csm.*` chunk design / T05 tokenizer
//! embed).
//!
//! Input: the upstream `sesame/csm-1b` safetensors checkpoint on
//! HuggingFace (Apache 2.0 code + weight, docs/license-audit.md; the repo
//! is gated — T29 owner hand-off). Output: a GGUF carrying every float
//! tensor verbatim plus the `vokra.csm.*` / `vokra.mimi.*` /
//! `vokra.provenance.*` metadata chunks and (optionally) the embedded text
//! tokenizer blob.
//!
//! # What is transcribed vs. shape-driven (ADR M4-05 §D2/§D9)
//!
//! - **Transcribed constants** (primary source = `SesameAILabs/csm`
//!   `models.py` / `generator.py`, `kyutai-labs/moshi` `loaders.py`,
//!   torchtune `Llama3ScaledRoPE` — all fetched and recorded in the ADR;
//!   the "model-card invariant" exception Kokoro / CosyVoice2 established):
//!   the two flavor dims (`llama3_2_1B` / `llama3_2_100M`), RMSNorm ε,
//!   RoPE base + Llama-3 scaling parameters, `max_seq_len`, sample /
//!   frame rates, `audio_num_codebooks = 32`, and the whole Mimi neural
//!   chunk group.
//! - **`0` placeholders pending T29** (checkpoint shapes / gated config):
//!   `audio_vocab_size` and `text_vocab_size`. The runtime rejects a `0`
//!   there at load (`CsmConfig::validate_for_forward`) — loud, never a
//!   silent zero-shape forward (FR-EX-08).
//!
//! # Tensor naming contract (T03)
//!
//! GGUF tensor names are the upstream safetensors names **verbatim**
//! (Whisper / Kokoro / CosyVoice2 contract). The runtime's real-weight
//! binding is a `NotImplemented` stub until the T29 tensor manifest lands,
//! so no name mapping is invented here.
//!
//! # No ONNX (permanent)
//!
//! CSM ships as safetensors + a Python pipeline; the pipeline is
//! re-implemented natively in `vokra-models/src/csm/` (whisper.cpp 型,
//! CLAUDE.md 設計判断 4). This converter never touches ONNX.

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for CSM GGUFs — kept in sync with the runtime
/// constant `vokra-models::csm::EXPECTED_ARCH`.
pub(crate) const ARCH: &str = "csm";
/// `vokra.model.name` for the CSM GGUF.
pub(crate) const NAME: &str = "sesame-csm-1b";

// --- vokra.csm.* keys (duplicated verbatim from
// `vokra-models/src/csm/config.rs` — the two crates only share vokra-core;
// the round-trip test below catches drift) ----------------------------------

const KEY_SAMPLE_RATE: &str = "vokra.csm.sample_rate";
const KEY_FRAME_RATE_MHZ: &str = "vokra.csm.frame_rate_mhz";
const KEY_BB_N_LAYER: &str = "vokra.csm.arch.backbone.n_layer";
const KEY_BB_D_MODEL: &str = "vokra.csm.arch.backbone.d_model";
const KEY_BB_N_HEAD_Q: &str = "vokra.csm.arch.backbone.n_head_q";
const KEY_BB_N_HEAD_KV: &str = "vokra.csm.arch.backbone.n_head_kv";
const KEY_BB_FFN_DIM: &str = "vokra.csm.arch.backbone.ffn_dim";
const KEY_DT_N_LAYER: &str = "vokra.csm.arch.depth.n_layer";
const KEY_DT_D_MODEL: &str = "vokra.csm.arch.depth.d_model";
const KEY_DT_N_HEAD_Q: &str = "vokra.csm.arch.depth.n_head_q";
const KEY_DT_N_HEAD_KV: &str = "vokra.csm.arch.depth.n_head_kv";
const KEY_DT_FFN_DIM: &str = "vokra.csm.arch.depth.ffn_dim";
const KEY_RMS_NORM_EPS: &str = "vokra.csm.arch.rms_norm_eps";
const KEY_ROPE_BASE: &str = "vokra.csm.arch.rope_base";
const KEY_N_CTX: &str = "vokra.csm.arch.n_ctx";
const KEY_ROPE_SCALE_FACTOR: &str = "vokra.csm.rope.scale_factor";
const KEY_ROPE_LOW_FREQ_FACTOR: &str = "vokra.csm.rope.low_freq_factor";
const KEY_ROPE_HIGH_FREQ_FACTOR: &str = "vokra.csm.rope.high_freq_factor";
const KEY_ROPE_OLD_CONTEXT_LEN: &str = "vokra.csm.rope.old_context_len";
const KEY_AUDIO_N_CODEBOOKS: &str = "vokra.csm.audio.n_codebooks";
const KEY_AUDIO_VOCAB_SIZE: &str = "vokra.csm.audio.vocab_size";
const KEY_TEXT_VOCAB_SIZE: &str = "vokra.csm.text.vocab_size";

// --- vokra.mimi.* keys (duplicated from vokra-models/src/mimi/config.rs) ---

const KEY_MIMI_SAMPLE_RATE: &str = "vokra.mimi.sample_rate";
const KEY_MIMI_FRAME_RATE_MHZ: &str = "vokra.mimi.frame_rate_mhz";
const KEY_MIMI_SEANET_DIMENSION: &str = "vokra.mimi.seanet.dimension";
const KEY_MIMI_SEANET_N_FILTERS: &str = "vokra.mimi.seanet.n_filters";
const KEY_MIMI_SEANET_N_RESIDUAL_LAYERS: &str = "vokra.mimi.seanet.n_residual_layers";
const KEY_MIMI_SEANET_KERNEL_SIZE: &str = "vokra.mimi.seanet.kernel_size";
const KEY_MIMI_SEANET_RESIDUAL_KERNEL_SIZE: &str = "vokra.mimi.seanet.residual_kernel_size";
const KEY_MIMI_SEANET_LAST_KERNEL_SIZE: &str = "vokra.mimi.seanet.last_kernel_size";
const KEY_MIMI_SEANET_COMPRESS: &str = "vokra.mimi.seanet.compress";
const KEY_MIMI_SEANET_DILATION_BASE: &str = "vokra.mimi.seanet.dilation_base";
const KEY_MIMI_SEANET_N_RATIOS: &str = "vokra.mimi.seanet.n_ratios";
const PREFIX_MIMI_SEANET_RATIO: &str = "vokra.mimi.seanet.ratio.";
const KEY_MIMI_QUANTIZER_DIMENSION: &str = "vokra.mimi.quantizer.dimension";
const KEY_MIMI_QUANTIZER_N_Q: &str = "vokra.mimi.quantizer.n_q";
const KEY_MIMI_QUANTIZER_BINS: &str = "vokra.mimi.quantizer.bins";
const KEY_MIMI_QUANTIZER_INPUT_DIMENSION: &str = "vokra.mimi.quantizer.input_dimension";
const KEY_MIMI_QUANTIZER_OUTPUT_DIMENSION: &str = "vokra.mimi.quantizer.output_dimension";
const KEY_MIMI_TRANSFORMER_D_MODEL: &str = "vokra.mimi.transformer.d_model";
const KEY_MIMI_TRANSFORMER_N_HEAD: &str = "vokra.mimi.transformer.n_head";
const KEY_MIMI_TRANSFORMER_N_LAYER: &str = "vokra.mimi.transformer.n_layer";
const KEY_MIMI_TRANSFORMER_FF_DIM: &str = "vokra.mimi.transformer.ff_dim";
const KEY_MIMI_TRANSFORMER_CONTEXT: &str = "vokra.mimi.transformer.context";
const KEY_MIMI_TRANSFORMER_MAX_PERIOD: &str = "vokra.mimi.transformer.max_period";
const KEY_MIMI_TRANSFORMER_LAYER_SCALE: &str = "vokra.mimi.transformer.layer_scale";

/// `vokra.tokenizer.model` — the raw text-tokenizer blob embedded verbatim
/// (M2-06 Whisper / M3-10 Voxtral pattern; zero-dep U8 array). For CSM
/// this is the `meta-llama/Llama-3.2-1B` tokenizer file (gated repo — the
/// owner supplies it at T29; conversions without it embed nothing and the
/// runtime tokenizer fails loudly).
const KEY_TOKENIZER_MODEL: &str = "vokra.tokenizer.model";

// --- Transcribed constants (ADR M4-05 §D2 — sources quoted per block) -------

/// Mimi native PCM rate (Hz) — `loaders.py` `SAMPLE_RATE = 24000`.
const CSM_SAMPLE_RATE: u32 = 24_000;
/// Mimi token frame rate in milli-Hz — `loaders.py` `FRAME_RATE = 12.5`.
const CSM_FRAME_RATE_MHZ: u32 = 12_500;
/// `models.py llama3_2_1B`: 16 layers / 2048 dim / 32 Q / 8 KV / 8192 FFN.
const BB: [u32; 5] = [16, 2048, 32, 8, 8192];
/// `models.py llama3_2_100M`: 4 layers / 1024 dim / 8 Q / 2 KV / 8192 FFN.
const DT: [u32; 5] = [4, 1024, 8, 2, 8192];
/// `models.py`: `norm_eps=1e-5` (both flavors).
const CSM_RMS_NORM_EPS: f32 = 1e-5;
/// `models.py`: `rope_base=500_000` (both flavors).
const CSM_ROPE_BASE: f32 = 500_000.0;
/// `models.py`: `max_seq_len=2048` (both flavors).
const CSM_N_CTX: u32 = 2048;
/// `models.py`: `scale_factor=32` → torchtune `Llama3ScaledRoPE`; its
/// defaults `low_freq_factor=1`, `high_freq_factor=4`,
/// `old_context_len=8192` apply (ADR §D3).
const CSM_ROPE_SCALE_FACTOR: f32 = 32.0;
const CSM_ROPE_LOW_FREQ_FACTOR: f32 = 1.0;
const CSM_ROPE_HIGH_FREQ_FACTOR: f32 = 4.0;
const CSM_ROPE_OLD_CONTEXT_LEN: u32 = 8192;
/// `generator.py`: `mimi.set_num_codebooks(32)`.
const CSM_AUDIO_N_CODEBOOKS: u32 = 32;

/// Mimi neural chunk constants — `loaders.py` `_seanet_kwargs` /
/// `_quantizer_kwargs` / `_transformer_kwargs` verbatim (ADR §D2 table).
const MIMI_SEANET_DIMENSION: u32 = 512;
const MIMI_SEANET_N_FILTERS: u32 = 64;
const MIMI_SEANET_N_RESIDUAL_LAYERS: u32 = 1;
const MIMI_SEANET_KERNEL_SIZE: u32 = 7;
const MIMI_SEANET_RESIDUAL_KERNEL_SIZE: u32 = 3;
const MIMI_SEANET_LAST_KERNEL_SIZE: u32 = 3;
const MIMI_SEANET_COMPRESS: u32 = 2;
const MIMI_SEANET_DILATION_BASE: u32 = 2;
const MIMI_SEANET_RATIOS: [u32; 4] = [8, 6, 5, 4];
const MIMI_QUANTIZER_DIMENSION: u32 = 256;
const MIMI_QUANTIZER_N_Q: u32 = 32;
const MIMI_QUANTIZER_BINS: u32 = 2048;
const MIMI_QUANTIZER_IO_DIMENSION: u32 = 512;
const MIMI_TRANSFORMER_D_MODEL: u32 = 512;
const MIMI_TRANSFORMER_N_HEAD: u32 = 8;
const MIMI_TRANSFORMER_N_LAYER: u32 = 8;
const MIMI_TRANSFORMER_FF_DIM: u32 = 2048;
const MIMI_TRANSFORMER_CONTEXT: u32 = 250;
const MIMI_TRANSFORMER_MAX_PERIOD: u32 = 10_000;
const MIMI_TRANSFORMER_LAYER_SCALE: f32 = 0.01;

/// Outcome of a CSM conversion.
#[derive(Debug, Default)]
pub(crate) struct CsmReport {
    /// Float tensors written verbatim.
    pub(crate) written: usize,
    /// Non-F32/F16 tensors skipped (defensive counter — the safetensors
    /// reader rejects unknown dtypes at parse time).
    pub(crate) skipped_non_float: usize,
    /// Whether a tokenizer blob was embedded.
    pub(crate) tokenizer_embedded: bool,
    /// Operator-facing diagnostics (never fail the conversion — the
    /// runtime is the authoritative gate, FR-EX-08).
    pub(crate) notes: Vec<String>,
}

/// Converts a CSM safetensors buffer into a populated GGUF builder.
///
/// `tokenizer_bytes` — the raw `meta-llama/Llama-3.2-1B` tokenizer file
/// (T29 owner supply; `None` skips the embed with a loud note).
pub(crate) fn convert(
    bytes: Vec<u8>,
    tokenizer_bytes: Option<Vec<u8>>,
) -> Result<(GgufBuilder, CsmReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    write_hparams(&mut b);
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        "Apache-2.0",
        Some("sesame/csm-1b"),
        Some("huggingface (gated acceptance — T29 owner hand-off)"),
    );

    let mut report = CsmReport::default();
    if let Some(tok) = tokenizer_bytes {
        if tok.is_empty() {
            report
                .notes
                .push("tokenizer file was empty — nothing embedded".into());
        } else {
            // U8 array, bytes verbatim (M2-06 Whisper / M3-10 Voxtral
            // pattern; runtime tokenizer is self-implemented, zero-dep).
            b.add_metadata(
                KEY_TOKENIZER_MODEL,
                GgufMetadataValue::Array(GgufArray {
                    element_type: GgufValueType::U8,
                    values: tok.iter().map(|&x| GgufMetadataValue::U8(x)).collect(),
                }),
            );
            report.tokenizer_embedded = true;
        }
    } else {
        report.notes.push(
            "no tokenizer supplied — `vokra.tokenizer.model` not embedded; the \
             runtime text path will fail loudly until a tokenizer-carrying GGUF \
             is converted (meta-llama/Llama-3.2-1B is a gated repo — T29)"
                .into(),
        );
    }

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

/// Writes the `vokra.csm.*` + `vokra.mimi.*` chunk groups: transcribed
/// primary-source constants (module docs) plus `0` placeholders on the two
/// vocab axes pending the T29 checkpoint (the runtime rejects `0` at
/// load — FR-EX-08 loud fail at first use).
fn write_hparams(b: &mut GgufBuilder) {
    b.add_u32(KEY_SAMPLE_RATE, CSM_SAMPLE_RATE);
    b.add_u32(KEY_FRAME_RATE_MHZ, CSM_FRAME_RATE_MHZ);
    for (keys, v) in [
        (
            [
                KEY_BB_N_LAYER,
                KEY_BB_D_MODEL,
                KEY_BB_N_HEAD_Q,
                KEY_BB_N_HEAD_KV,
                KEY_BB_FFN_DIM,
            ],
            BB,
        ),
        (
            [
                KEY_DT_N_LAYER,
                KEY_DT_D_MODEL,
                KEY_DT_N_HEAD_Q,
                KEY_DT_N_HEAD_KV,
                KEY_DT_FFN_DIM,
            ],
            DT,
        ),
    ] {
        for (key, value) in keys.iter().zip(v.iter()) {
            b.add_u32(key, *value);
        }
    }
    b.add_f32(KEY_RMS_NORM_EPS, CSM_RMS_NORM_EPS);
    b.add_f32(KEY_ROPE_BASE, CSM_ROPE_BASE);
    b.add_u32(KEY_N_CTX, CSM_N_CTX);
    b.add_f32(KEY_ROPE_SCALE_FACTOR, CSM_ROPE_SCALE_FACTOR);
    b.add_f32(KEY_ROPE_LOW_FREQ_FACTOR, CSM_ROPE_LOW_FREQ_FACTOR);
    b.add_f32(KEY_ROPE_HIGH_FREQ_FACTOR, CSM_ROPE_HIGH_FREQ_FACTOR);
    b.add_u32(KEY_ROPE_OLD_CONTEXT_LEN, CSM_ROPE_OLD_CONTEXT_LEN);
    b.add_u32(KEY_AUDIO_N_CODEBOOKS, CSM_AUDIO_N_CODEBOOKS);
    // Shape-driven at T29 (gated checkpoint): audio_vocab_size from the
    // codebook0_head rows / audio_embeddings rows ÷ 32; text_vocab_size
    // from the text_embeddings rows. `0` until then — never invented.
    b.add_u32(KEY_AUDIO_VOCAB_SIZE, 0);
    b.add_u32(KEY_TEXT_VOCAB_SIZE, 0);

    // Mimi neural chunk group (loaders.py constants — the CSM engine reads
    // its codec shape from its own GGUF; the Mimi *weights* travel in the
    // standalone M4-04 mimi GGUF).
    b.add_u32(KEY_MIMI_SAMPLE_RATE, CSM_SAMPLE_RATE);
    b.add_u32(KEY_MIMI_FRAME_RATE_MHZ, CSM_FRAME_RATE_MHZ);
    b.add_u32(KEY_MIMI_SEANET_DIMENSION, MIMI_SEANET_DIMENSION);
    b.add_u32(KEY_MIMI_SEANET_N_FILTERS, MIMI_SEANET_N_FILTERS);
    b.add_u32(
        KEY_MIMI_SEANET_N_RESIDUAL_LAYERS,
        MIMI_SEANET_N_RESIDUAL_LAYERS,
    );
    b.add_u32(KEY_MIMI_SEANET_KERNEL_SIZE, MIMI_SEANET_KERNEL_SIZE);
    b.add_u32(
        KEY_MIMI_SEANET_RESIDUAL_KERNEL_SIZE,
        MIMI_SEANET_RESIDUAL_KERNEL_SIZE,
    );
    b.add_u32(
        KEY_MIMI_SEANET_LAST_KERNEL_SIZE,
        MIMI_SEANET_LAST_KERNEL_SIZE,
    );
    b.add_u32(KEY_MIMI_SEANET_COMPRESS, MIMI_SEANET_COMPRESS);
    b.add_u32(KEY_MIMI_SEANET_DILATION_BASE, MIMI_SEANET_DILATION_BASE);
    b.add_u32(KEY_MIMI_SEANET_N_RATIOS, MIMI_SEANET_RATIOS.len() as u32);
    for (i, r) in MIMI_SEANET_RATIOS.iter().enumerate() {
        b.add_u32(&format!("{PREFIX_MIMI_SEANET_RATIO}{i}"), *r);
    }
    b.add_u32(KEY_MIMI_QUANTIZER_DIMENSION, MIMI_QUANTIZER_DIMENSION);
    b.add_u32(KEY_MIMI_QUANTIZER_N_Q, MIMI_QUANTIZER_N_Q);
    b.add_u32(KEY_MIMI_QUANTIZER_BINS, MIMI_QUANTIZER_BINS);
    b.add_u32(
        KEY_MIMI_QUANTIZER_INPUT_DIMENSION,
        MIMI_QUANTIZER_IO_DIMENSION,
    );
    b.add_u32(
        KEY_MIMI_QUANTIZER_OUTPUT_DIMENSION,
        MIMI_QUANTIZER_IO_DIMENSION,
    );
    b.add_u32(KEY_MIMI_TRANSFORMER_D_MODEL, MIMI_TRANSFORMER_D_MODEL);
    b.add_u32(KEY_MIMI_TRANSFORMER_N_HEAD, MIMI_TRANSFORMER_N_HEAD);
    b.add_u32(KEY_MIMI_TRANSFORMER_N_LAYER, MIMI_TRANSFORMER_N_LAYER);
    b.add_u32(KEY_MIMI_TRANSFORMER_FF_DIM, MIMI_TRANSFORMER_FF_DIM);
    b.add_u32(KEY_MIMI_TRANSFORMER_CONTEXT, MIMI_TRANSFORMER_CONTEXT);
    b.add_u32(KEY_MIMI_TRANSFORMER_MAX_PERIOD, MIMI_TRANSFORMER_MAX_PERIOD);
    b.add_f32(
        KEY_MIMI_TRANSFORMER_LAYER_SCALE,
        MIMI_TRANSFORMER_LAYER_SCALE,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    fn minimal_safetensors_one_f32() -> Vec<u8> {
        let header = r#"{"backbone.layers.0.attn.q_proj.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 24]);
        out
    }

    #[test]
    fn round_trip_carries_arch_chunks_and_provenance() {
        let (builder, report) = convert(minimal_safetensors_one_f32(), None).expect("convert");
        assert_eq!(report.written, 1);
        assert!(!report.tokenizer_embedded);
        assert!(
            report.notes.iter().any(|n| n.contains("tokenizer")),
            "missing tokenizer must be a loud note"
        );

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        // Transcribed flavor dims (models.py — ADR M4-05 §D2).
        for (key, want) in [
            (KEY_BB_N_LAYER, 16u32),
            (KEY_BB_D_MODEL, 2048),
            (KEY_BB_N_HEAD_Q, 32),
            (KEY_BB_N_HEAD_KV, 8),
            (KEY_BB_FFN_DIM, 8192),
            (KEY_DT_N_LAYER, 4),
            (KEY_DT_D_MODEL, 1024),
            (KEY_DT_N_HEAD_Q, 8),
            (KEY_DT_N_HEAD_KV, 2),
            (KEY_DT_FFN_DIM, 8192),
            (KEY_N_CTX, 2048),
            (KEY_AUDIO_N_CODEBOOKS, 32),
            (KEY_ROPE_OLD_CONTEXT_LEN, 8192),
            (KEY_SAMPLE_RATE, 24_000),
            (KEY_FRAME_RATE_MHZ, 12_500),
        ] {
            match file.get(key) {
                Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, want, "{key}"),
                other => panic!("{key}: unexpected {other:?}"),
            }
        }
        // T29-pending placeholders.
        for key in [KEY_AUDIO_VOCAB_SIZE, KEY_TEXT_VOCAB_SIZE] {
            match file.get(key) {
                Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, 0, "{key} placeholder"),
                other => panic!("{key}: unexpected {other:?}"),
            }
        }
        // Provenance: Apache-2.0 permissive (compliance gate input).
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some("sesame/csm-1b")
        );
        // Mimi neural chunk group present (indexed ratios).
        match file.get(KEY_MIMI_SEANET_N_RATIOS) {
            Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, 4),
            other => panic!("n_ratios: unexpected {other:?}"),
        }
        for (i, want) in [8u32, 6, 5, 4].iter().enumerate() {
            match file.get(&format!("{PREFIX_MIMI_SEANET_RATIO}{i}")) {
                Some(GgufMetadataValue::U32(v)) => assert_eq!(v, want),
                other => panic!("ratio.{i}: unexpected {other:?}"),
            }
        }
    }

    #[test]
    fn tokenizer_bytes_are_embedded_verbatim() {
        let tok = b"fake-tokenizer-blob".to_vec();
        let (builder, report) =
            convert(minimal_safetensors_one_f32(), Some(tok.clone())).expect("convert");
        assert!(report.tokenizer_embedded);
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        let arr = file
            .get(KEY_TOKENIZER_MODEL)
            .and_then(|v| v.as_array())
            .expect("tokenizer array present");
        let bytes: Vec<u8> = arr
            .values
            .iter()
            .map(|v| match v {
                GgufMetadataValue::U8(x) => *x,
                other => panic!("non-U8 element {other:?}"),
            })
            .collect();
        assert_eq!(bytes, tok);
    }

    #[test]
    fn arch_string_matches_runtime_constant() {
        assert_eq!(ARCH, "csm");
    }
}
