//! Strict Kyutai STT-2.6B-EN decoder-component conversion.
//!
//! This converter accepts only the source-authenticated decoder checkpoint
//! manifest from `kyutai/stt-2.6b-en`: 323 dense BF16 tensors, with the exact
//! names and shapes used by the dep_q=0 decoder seam.  Mimi, the tokenizer,
//! streaming state, and whole-file identity remain separate runtime gates;
//! this output is intentionally not a complete ASR model.

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use super::canary_1b_flash::{hex, sha256};
use crate::ConvertError;
use crate::safetensors::SafetensorsFile;
use crate::spm_proto::{PieceType, parse_model};

pub(crate) const ARCH: &str = "kyutai-stt";
pub(crate) const NAME: &str = "kyutai-stt-2.6b-en";

const SAMPLE_RATE: u32 = 24_000;
const BB_N_LAYER: u32 = 48;
const BB_D_MODEL: u32 = 2048;
const BB_N_HEAD: u32 = 32;
const BB_HIDDEN_SCALE: f32 = 4.125;
const BB_FFN_HIDDEN: u32 = 5632;
const BB_CONTEXT: u32 = 375;
const BB_ROPE_MAX_PERIOD: f32 = 100_000.0;
const BB_CAUSAL: u32 = 1;
const BB_RMS_NORM_EPS: f32 = 1e-8;
const DEP_N_LAYER: u32 = 6;
const DEP_D_MODEL: u32 = 1024;
const DEP_N_HEAD: u32 = 16;
const DEP_MULTI_LINEAR: u32 = 1;
const DEP_WEIGHTS_PER_STEP: u32 = 1;
const N_Q: u32 = 32;
const DEP_Q: u32 = 0;
const AUDIO_CARD: u32 = 2048;
const TEXT_CARD: u32 = 4000;
const TEXT_PAD_ID: u32 = 3;
const AUDIO_DELAY_SECS: f32 = 2.5;
const AUDIO_SILENCE_PREFIX_SECS: f32 = 1.0;
const N_DELAYS: u32 = 33;

const PROVENANCE_LICENSE: &str = "cc-by-4.0";
const PROVENANCE_MODEL_ID: &str = "kyutai/stt-2.6b-en";
const PROVENANCE_SOURCE: &str = "https://huggingface.co/kyutai/stt-2.6b-en";
const ATTRIBUTION: &str = "This application uses the Kyutai STT-2.6B-EN model (decoder-only English streaming ASR over Mimi audio tokens). Model weights are licensed under CC-BY 4.0 (attribution required; commercial use permitted). Copyright (c) Kyutai. Source: https://github.com/kyutai-labs/delayed-streams-modeling / https://huggingface.co/kyutai/stt-2.6b-en";

const KEY_SAMPLE_RATE: &str = "vokra.kyutai_stt.sample_rate";
const KEY_BB_N_LAYER: &str = "vokra.kyutai_stt.arch.backbone.n_layer";
const KEY_BB_D_MODEL: &str = "vokra.kyutai_stt.arch.backbone.d_model";
const KEY_BB_N_HEAD: &str = "vokra.kyutai_stt.arch.backbone.n_head";
const KEY_BB_HIDDEN_SCALE: &str = "vokra.kyutai_stt.arch.backbone.hidden_scale";
const KEY_BB_FFN_HIDDEN: &str = "vokra.kyutai_stt.arch.backbone.ffn_hidden";
const KEY_BB_CONTEXT: &str = "vokra.kyutai_stt.arch.backbone.context";
const KEY_BB_ROPE_MAX_PERIOD: &str = "vokra.kyutai_stt.arch.backbone.rope_max_period";
const KEY_BB_CAUSAL: &str = "vokra.kyutai_stt.arch.backbone.causal";
const KEY_BB_RMS_NORM_EPS: &str = "vokra.kyutai_stt.arch.backbone.rms_norm_eps";
const KEY_DEP_N_LAYER: &str = "vokra.kyutai_stt.arch.depformer.n_layer";
const KEY_DEP_D_MODEL: &str = "vokra.kyutai_stt.arch.depformer.d_model";
const KEY_DEP_N_HEAD: &str = "vokra.kyutai_stt.arch.depformer.n_head";
const KEY_DEP_MULTI_LINEAR: &str = "vokra.kyutai_stt.arch.depformer.multi_linear";
const KEY_DEP_WEIGHTS_PER_STEP: &str = "vokra.kyutai_stt.arch.depformer.weights_per_step";
const KEY_N_Q: &str = "vokra.kyutai_stt.audio.n_q";
const KEY_DEP_Q: &str = "vokra.kyutai_stt.audio.dep_q";
const KEY_AUDIO_CARD: &str = "vokra.kyutai_stt.audio.card";
const KEY_TEXT_CARD: &str = "vokra.kyutai_stt.text.card";
const KEY_TEXT_PAD_ID: &str = "vokra.kyutai_stt.text.pad_id";
const KEY_AUDIO_DELAY_SECS: &str = "vokra.kyutai_stt.stream.audio_delay_seconds";
const KEY_AUDIO_SILENCE_PREFIX_SECS: &str = "vokra.kyutai_stt.stream.audio_silence_prefix_seconds";
const KEY_N_DELAYS: &str = "vokra.kyutai_stt.n_delays";
const PREFIX_DELAY: &str = "vokra.kyutai_stt.delay.";

/// Dedicated tokenizer component schema.  It is deliberately a distinct
/// GGUF arch from the decoder component: pairing a tokenizer with a decoder
/// is an explicit caller operation, never an accidental metadata alias.
pub(crate) const TOKENIZER_ARCH: &str = "kyutai-stt-tokenizer";
pub(crate) const TOKENIZER_COMPONENT_NAME: &str = "kyutai-stt-2.6b-en-tokenizer";
pub(crate) const TOKENIZER_ASSET_NAME: &str = "tokenizer_en_audio_4000.model";
pub(crate) const MIMI_ASSET_NAME: &str = "mimi-pytorch-e351c8d8@125.safetensors";
pub(crate) const KEY_TOKENIZER_SCHEMA: &str = "vokra.kyutai_stt.tokenizer.schema";
pub(crate) const KEY_TOKENIZER_CARD: &str = "vokra.kyutai_stt.tokenizer.card";
pub(crate) const KEY_TOKENIZER_PIECES: &str = "vokra.kyutai_stt.tokenizer.pieces";
pub(crate) const KEY_TOKENIZER_TYPES: &str = "vokra.kyutai_stt.tokenizer.types";
pub(crate) const KEY_TOKENIZER_UNK_ID: &str = "vokra.kyutai_stt.tokenizer.unk_id";
pub(crate) const KEY_TOKENIZER_BOS_ID: &str = "vokra.kyutai_stt.tokenizer.bos_id";
pub(crate) const KEY_TOKENIZER_EOS_ID: &str = "vokra.kyutai_stt.tokenizer.eos_id";
pub(crate) const KEY_TOKENIZER_PAD_ID: &str = "vokra.kyutai_stt.tokenizer.pad_id";
pub(crate) const KEY_TOKENIZER_BYTES: &str = "vokra.kyutai_stt.tokenizer.bytes";
pub(crate) const KEY_TOKENIZER_SHA256: &str = "vokra.kyutai_stt.tokenizer.sha256";
pub(crate) const KEY_TOKENIZER_TABLE_SHA256: &str = "vokra.kyutai_stt.tokenizer.table_sha256";
pub(crate) const KEY_TOKENIZER_GIT_BLOB_SHA1: &str = "vokra.kyutai_stt.tokenizer.git_blob_sha1";
pub(crate) const KEY_TOKENIZER_MIMI_FILE: &str = "vokra.kyutai_stt.tokenizer.mimi.file";
pub(crate) const KEY_TOKENIZER_MIMI_BYTES: &str = "vokra.kyutai_stt.tokenizer.mimi.bytes";
pub(crate) const KEY_TOKENIZER_MIMI_SHA256: &str = "vokra.kyutai_stt.tokenizer.mimi.sha256";
pub(crate) const KEY_TOKENIZER_ADD_DUMMY_PREFIX: &str =
    "vokra.kyutai_stt.tokenizer.normalizer.add_dummy_prefix";
pub(crate) const KEY_TOKENIZER_REMOVE_EXTRA_WHITESPACES: &str =
    "vokra.kyutai_stt.tokenizer.normalizer.remove_extra_whitespaces";
pub(crate) const KEY_TOKENIZER_DENORMALIZER_PRESENT: &str =
    "vokra.kyutai_stt.tokenizer.denormalizer_present";
const TOKENIZER_SCHEMA: &str = "sentencepiece-decode-v1";
const TOKENIZER_TABLE_DIGEST_PREFIX: &[u8] = b"vokra.kyutai_stt.tokenizer.table.v1\0";

/// Conversion accounting retained by the generic converter dispatch API.
#[derive(Debug, Default)]
pub(crate) struct KyutaiSttReport {
    /// Number of exact-manifest tensors written verbatim as BF16.
    pub(crate) written: usize,
    /// Always zero for this strict converter; non-BF16 input is rejected.
    pub(crate) skipped_non_float: usize,
    /// Number of BF16 tensors written verbatim.
    pub(crate) bf16_passthrough: usize,
    /// Operator-facing diagnostics.
    pub(crate) notes: Vec<String>,
}

type TensorSpec = (String, Vec<u64>);

fn checked_mul(label: &str, lhs: usize, rhs: usize) -> Result<usize, ConvertError> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| ConvertError::Parse(format!("{label} overflows: {lhs} * {rhs}")))
}

fn checked_add(label: &str, lhs: usize, rhs: usize) -> Result<usize, ConvertError> {
    lhs.checked_add(rhs)
        .ok_or_else(|| ConvertError::Parse(format!("{label} overflows: {lhs} + {rhs}")))
}

fn axis(value: usize, label: &str) -> Result<u64, ConvertError> {
    u64::try_from(value)
        .map_err(|_| ConvertError::Parse(format!("{label} does not fit in u64: {value}")))
}

fn matrix(rows: usize, cols: usize, label: &str) -> Result<Vec<u64>, ConvertError> {
    let _ = checked_mul(label, rows, cols)?;
    Ok(vec![axis(rows, label)?, axis(cols, label)?])
}

fn expected_specs() -> Result<Vec<TensorSpec>, ConvertError> {
    let d_model = usize::try_from(BB_D_MODEL)
        .map_err(|_| ConvertError::Parse("d_model does not fit usize".into()))?;
    let n_layers = usize::try_from(BB_N_LAYER)
        .map_err(|_| ConvertError::Parse("n_layer does not fit usize".into()))?;
    let audio_rows = usize::try_from(AUDIO_CARD)
        .map_err(|_| ConvertError::Parse("audio card does not fit usize".into()))?;
    let text_rows = usize::try_from(TEXT_CARD)
        .map_err(|_| ConvertError::Parse("text card does not fit usize".into()))?;
    let audio_rows = checked_add("audio embedding rows", audio_rows, 1)?;
    let text_rows = checked_add("text embedding rows", text_rows, 1)?;
    let qkv_rows = checked_mul("QKV rows", 3, d_model)?;
    let ffn_hidden = usize::try_from(BB_FFN_HIDDEN)
        .map_err(|_| ConvertError::Parse("FFN hidden does not fit usize".into()))?;
    let ffn_rows = checked_mul("gating input rows", 2, ffn_hidden)?;
    let per_layer = 6usize;
    let layer_tensors = checked_mul("layer tensor count", per_layer, n_layers)?;
    let n_q =
        usize::try_from(N_Q).map_err(|_| ConvertError::Parse("n_q does not fit usize".into()))?;
    let count = checked_add("manifest tensor count", 1, n_q)?;
    let count = checked_add("manifest tensor count", count, layer_tensors)?;
    let count = checked_add("manifest tensor count", count, 2)?;
    let text_output_rows = text_rows
        .checked_sub(1)
        .ok_or_else(|| ConvertError::Parse("text output vocabulary rows underflow".into()))?;
    let mut specs = Vec::with_capacity(count);
    specs.push((
        "text_emb.weight".to_owned(),
        matrix(text_rows, d_model, "text embedding shape")?,
    ));
    for i in 0..N_Q {
        specs.push((
            format!("emb.{i}.weight"),
            matrix(audio_rows, d_model, "audio embedding shape")?,
        ));
    }
    for layer in 0..n_layers {
        let prefix = format!("transformer.layers.{layer}");
        specs.push((
            format!("{prefix}.self_attn.in_proj_weight"),
            matrix(qkv_rows, d_model, "attention input projection shape")?,
        ));
        specs.push((
            format!("{prefix}.self_attn.out_proj.weight"),
            matrix(d_model, d_model, "attention output projection shape")?,
        ));
        specs.push((
            format!("{prefix}.gating.linear_in.weight"),
            matrix(ffn_rows, d_model, "gating input projection shape")?,
        ));
        specs.push((
            format!("{prefix}.gating.linear_out.weight"),
            matrix(d_model, ffn_hidden, "gating output projection shape")?,
        ));
        specs.push((
            format!("{prefix}.norm1.alpha"),
            vec![axis(d_model, "norm shape")?],
        ));
        specs.push((
            format!("{prefix}.norm2.alpha"),
            vec![axis(d_model, "norm shape")?],
        ));
    }
    specs.push((
        "out_norm.alpha".to_owned(),
        vec![axis(d_model, "output norm shape")?],
    ));
    specs.push((
        "text_linear.weight".to_owned(),
        matrix(text_output_rows, d_model, "text output projection shape")?,
    ));
    Ok(specs)
}

fn validate_names(actual: &[String], expected: &[TensorSpec]) -> Result<(), ConvertError> {
    let mut actual_names = actual.to_vec();
    actual_names.sort_unstable();
    let mut expected_names: Vec<&str> = expected.iter().map(|(name, _)| name.as_str()).collect();
    expected_names.sort_unstable();
    if actual_names.len() != expected_names.len() {
        return Err(ConvertError::Parse(format!(
            "Kyutai decoder tensor manifest has {} entries; expected {}",
            actual_names.len(),
            expected_names.len()
        )));
    }
    for (actual, expected) in actual_names.iter().zip(expected_names.iter()) {
        if actual.as_str() != *expected {
            return Err(ConvertError::Parse(format!(
                "Kyutai decoder tensor manifest mismatch: got `{actual}`, expected `{expected}`"
            )));
        }
    }
    Ok(())
}

fn checked_element_count(shape: &[u64], name: &str) -> Result<usize, ConvertError> {
    shape.iter().try_fold(1usize, |count, axis| {
        let axis = usize::try_from(*axis)
            .map_err(|_| ConvertError::Parse(format!("tensor `{name}` axis does not fit usize")))?;
        checked_mul("tensor element count", count, axis)
    })
}

fn validate_payload(
    name: &str,
    dtype: GgmlType,
    shape: &[u64],
    bytes: &[u8],
    expected_shape: &[u64],
) -> Result<(), ConvertError> {
    if dtype != GgmlType::BF16 {
        return Err(ConvertError::Parse(format!(
            "Kyutai decoder tensor `{name}` has dtype {dtype:?}; only BF16 is accepted"
        )));
    }
    if shape != expected_shape {
        return Err(ConvertError::Parse(format!(
            "Kyutai decoder tensor `{name}` has shape {shape:?}; expected {expected_shape:?}"
        )));
    }
    let elements = checked_element_count(shape, name)?;
    let expected_bytes = checked_mul("BF16 payload size", elements, 2)?;
    if bytes.len() != expected_bytes {
        return Err(ConvertError::Parse(format!(
            "Kyutai decoder tensor `{name}` has {} payload bytes; expected {expected_bytes}",
            bytes.len()
        )));
    }
    for pair in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        let value = f32::from_bits(u32::from(bits) << 16);
        if !value.is_finite() {
            return Err(ConvertError::Parse(format!(
                "Kyutai decoder tensor `{name}` contains a non-finite BF16 value"
            )));
        }
    }
    Ok(())
}

fn write_hparams(builder: &mut GgufBuilder) {
    builder.add_u32(KEY_SAMPLE_RATE, SAMPLE_RATE);
    builder.add_u32(KEY_BB_N_LAYER, BB_N_LAYER);
    builder.add_u32(KEY_BB_D_MODEL, BB_D_MODEL);
    builder.add_u32(KEY_BB_N_HEAD, BB_N_HEAD);
    builder.add_f32(KEY_BB_HIDDEN_SCALE, BB_HIDDEN_SCALE);
    builder.add_u32(KEY_BB_FFN_HIDDEN, BB_FFN_HIDDEN);
    builder.add_u32(KEY_BB_CONTEXT, BB_CONTEXT);
    builder.add_f32(KEY_BB_ROPE_MAX_PERIOD, BB_ROPE_MAX_PERIOD);
    builder.add_u32(KEY_BB_CAUSAL, BB_CAUSAL);
    builder.add_f32(KEY_BB_RMS_NORM_EPS, BB_RMS_NORM_EPS);
    builder.add_u32(KEY_DEP_N_LAYER, DEP_N_LAYER);
    builder.add_u32(KEY_DEP_D_MODEL, DEP_D_MODEL);
    builder.add_u32(KEY_DEP_N_HEAD, DEP_N_HEAD);
    builder.add_u32(KEY_DEP_MULTI_LINEAR, DEP_MULTI_LINEAR);
    builder.add_u32(KEY_DEP_WEIGHTS_PER_STEP, DEP_WEIGHTS_PER_STEP);
    builder.add_u32(KEY_N_Q, N_Q);
    builder.add_u32(KEY_DEP_Q, DEP_Q);
    builder.add_u32(KEY_AUDIO_CARD, AUDIO_CARD);
    builder.add_u32(KEY_TEXT_CARD, TEXT_CARD);
    builder.add_u32(KEY_TEXT_PAD_ID, TEXT_PAD_ID);
    builder.add_f32(KEY_AUDIO_DELAY_SECS, AUDIO_DELAY_SECS);
    builder.add_f32(KEY_AUDIO_SILENCE_PREFIX_SECS, AUDIO_SILENCE_PREFIX_SECS);
    builder.add_u32(KEY_N_DELAYS, N_DELAYS);
    for index in 0..N_DELAYS {
        builder.add_u32(&format!("{PREFIX_DELAY}{index}"), 0);
    }
}

fn component_builder() -> GgufBuilder {
    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    write_hparams(&mut builder);
    vokra_core::stamp_provenance(
        &mut builder,
        LicenseClass::AttributionRequired,
        PROVENANCE_LICENSE,
        Some(PROVENANCE_MODEL_ID),
        Some(PROVENANCE_SOURCE),
    );
    vokra_core::stamp_attribution(&mut builder, ATTRIBUTION);
    builder
}

/// Summary for the metadata-only tokenizer companion.  The tokenizer is
/// intentionally not a tensor component and must be shipped separately from
/// the 384 MB Mimi weights. Its Mimi fields declare the expected companion;
/// runtime composition still requires the independently authenticated bytes.
#[derive(Debug, Default)]
pub(crate) struct KyutaiSttTokenizerReport {
    pub(crate) pieces: usize,
    pub(crate) byte_fallback_pieces: usize,
}

fn piece_type_value(piece_type: PieceType) -> Result<u32, ConvertError> {
    let value = match piece_type {
        PieceType::Unspecified => {
            return Err(ConvertError::Parse(
                "Kyutai tokenizer contains explicit invalid SentencePiece type 0".into(),
            ));
        }
        PieceType::Normal => 1,
        PieceType::Unknown => 2,
        PieceType::Control => 3,
        PieceType::UserDefined => 4,
        PieceType::Unused => 5,
        PieceType::Byte => 6,
        PieceType::Other(value) => {
            return Err(ConvertError::Parse(format!(
                "Kyutai tokenizer contains unsupported SentencePiece type {value}"
            )));
        }
    };
    Ok(value)
}

fn validate_tokenizer_model(
    bytes: &[u8],
) -> Result<(crate::spm_proto::ModelProto, String), ConvertError> {
    if bytes.len() != 59_339 {
        return Err(ConvertError::Parse(format!(
            "Kyutai tokenizer `{}` has {} bytes; expected 59339",
            TOKENIZER_ASSET_NAME,
            bytes.len()
        )));
    }
    let digest = hex(&sha256(bytes));
    if digest != "d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6" {
        return Err(ConvertError::Parse(format!(
            "Kyutai tokenizer SHA-256 {digest} does not match the authenticated sidecar"
        )));
    }
    let model = parse_model(bytes).map_err(|error| {
        ConvertError::Parse(format!(
            "Kyutai tokenizer SentencePiece ModelProto: {error}"
        ))
    })?;
    if model.pieces.len() != TEXT_CARD as usize {
        return Err(ConvertError::Parse(format!(
            "Kyutai tokenizer has {} pieces; expected text_card={TEXT_CARD}",
            model.pieces.len()
        )));
    }
    if !model.normalizer_add_dummy_prefix || !model.normalizer_remove_extra_whitespaces {
        return Err(ConvertError::Parse(
            "Kyutai tokenizer normalizer flags differ from the authenticated SentencePiece decode contract"
                .into(),
        ));
    }
    if model.denormalizer_present {
        return Err(ConvertError::Parse(
            "Kyutai tokenizer carries an unsupported denormalizer_spec".into(),
        ));
    }
    let expected_specials = [
        (0usize, "<unk>", PieceType::Unknown),
        (1, "<s>", PieceType::Control),
        (2, "</s>", PieceType::Control),
        (3, "<pad>", PieceType::Control),
    ];
    for (id, expected, expected_type) in expected_specials {
        let piece = &model.pieces[id];
        if piece.piece != expected || piece.piece_type != expected_type {
            return Err(ConvertError::Parse(format!(
                "Kyutai tokenizer special id {id} is {:?}/{:?}; expected {expected:?}/{expected_type:?}",
                piece.piece, piece.piece_type
            )));
        }
    }
    for (id, piece) in model.pieces.iter().enumerate() {
        if piece.piece.is_empty() {
            return Err(ConvertError::Parse(format!(
                "Kyutai tokenizer piece {id} is empty"
            )));
        }
        if !piece.score.is_finite() {
            return Err(ConvertError::Parse(format!(
                "Kyutai tokenizer piece {id} has a non-finite score"
            )));
        }
        let _ = piece_type_value(piece.piece_type)?;
    }
    Ok((model, digest))
}

fn tokenizer_builder(model: &crate::spm_proto::ModelProto, digest: &str) -> GgufBuilder {
    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, TOKENIZER_ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, TOKENIZER_COMPONENT_NAME);
    builder.add_string(KEY_TOKENIZER_SCHEMA, TOKENIZER_SCHEMA);
    builder.add_u32(KEY_TOKENIZER_CARD, model.pieces.len() as u32);
    builder.add_metadata(
        KEY_TOKENIZER_PIECES,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: model
                .pieces
                .iter()
                .map(|piece| GgufMetadataValue::String(piece.piece.clone()))
                .collect(),
        }),
    );
    builder.add_metadata(
        KEY_TOKENIZER_TYPES,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: model
                .pieces
                .iter()
                .map(|piece| {
                    GgufMetadataValue::U32(
                        piece_type_value(piece.piece_type).expect("validated type"),
                    )
                })
                .collect(),
        }),
    );
    builder.add_u32(KEY_TOKENIZER_UNK_ID, 0);
    builder.add_u32(KEY_TOKENIZER_BOS_ID, 1);
    builder.add_u32(KEY_TOKENIZER_EOS_ID, 2);
    builder.add_u32(KEY_TOKENIZER_PAD_ID, 3);
    builder.add_u32(KEY_TOKENIZER_BYTES, 59_339);
    builder.add_string(KEY_TOKENIZER_SHA256, digest);
    builder.add_string(KEY_TOKENIZER_TABLE_SHA256, &tokenizer_table_sha256(model));
    builder.add_string(
        KEY_TOKENIZER_GIT_BLOB_SHA1,
        "1820a7cbb15efc6a33dd365113c07e3df9d28d80",
    );
    builder.add_string(KEY_TOKENIZER_MIMI_FILE, MIMI_ASSET_NAME);
    builder.add_u32(KEY_TOKENIZER_MIMI_BYTES, 384_644_900);
    builder.add_string(
        KEY_TOKENIZER_MIMI_SHA256,
        "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50",
    );
    builder.add_bool(
        KEY_TOKENIZER_ADD_DUMMY_PREFIX,
        model.normalizer_add_dummy_prefix,
    );
    builder.add_bool(
        KEY_TOKENIZER_REMOVE_EXTRA_WHITESPACES,
        model.normalizer_remove_extra_whitespaces,
    );
    builder.add_bool(
        KEY_TOKENIZER_DENORMALIZER_PRESENT,
        model.denormalizer_present,
    );
    vokra_core::stamp_provenance(
        &mut builder,
        LicenseClass::AttributionRequired,
        PROVENANCE_LICENSE,
        Some(PROVENANCE_MODEL_ID),
        Some(PROVENANCE_SOURCE),
    );
    vokra_core::stamp_attribution(&mut builder, ATTRIBUTION);
    builder
}

/// Hash the complete ordered tokenizer table, not just the raw sidecar.
/// Length-prefixing every UTF-8 piece and including its ID and exact enum
/// value makes the encoding deterministic and unambiguous. This digest is an
/// internal table-integrity check; it does not replace the external whole-file
/// SHA-256 identity of the raw tokenizer asset or output GGUF.
fn tokenizer_table_sha256(model: &crate::spm_proto::ModelProto) -> String {
    let mut canonical = Vec::new();
    canonical.extend_from_slice(TOKENIZER_TABLE_DIGEST_PREFIX);
    for (id, piece) in model.pieces.iter().enumerate() {
        canonical.extend_from_slice(&(id as u32).to_le_bytes());
        canonical.extend_from_slice(&(piece.piece.len() as u32).to_le_bytes());
        canonical.extend_from_slice(piece.piece.as_bytes());
        canonical.extend_from_slice(
            &piece_type_value(piece.piece_type)
                .expect("validated type")
                .to_le_bytes(),
        );
    }
    hex(&sha256(&canonical))
}

/// Convert the exact Kyutai STT SentencePiece sidecar into a metadata-only
/// GGUF tokenizer component. The raw 59,339-byte asset is authenticated and
/// parsed offline; no model/Mimi weights are read or embedded. The GGUF also
/// carries a canonical digest of the complete ordered piece/type table so
/// runtime readback can detect metadata tampering independently of the raw
/// sidecar SHA-256.
pub(crate) fn convert_tokenizer(
    bytes: Vec<u8>,
) -> Result<(GgufBuilder, KyutaiSttTokenizerReport), ConvertError> {
    let (model, digest) = validate_tokenizer_model(&bytes)?;
    let byte_fallback_pieces = model
        .pieces
        .iter()
        .filter(|piece| {
            piece.piece_type == PieceType::Byte
                && piece.piece.starts_with("<0x")
                && piece.piece.ends_with('>')
        })
        .count();
    let report = KyutaiSttTokenizerReport {
        pieces: model.pieces.len(),
        byte_fallback_pieces,
    };
    Ok((tokenizer_builder(&model, &digest), report))
}

/// Convert the authenticated decoder component, preserving every BF16 payload
/// verbatim. No scalar or tensor defaults are synthesized.
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, KyutaiSttReport), ConvertError> {
    let safetensors = SafetensorsFile::parse(bytes)?;
    let expected = expected_specs()?;
    let actual_names: Vec<String> = safetensors
        .tensors()
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect();
    validate_names(&actual_names, &expected)?;

    let mut builder = component_builder();

    let mut report = KyutaiSttReport::default();
    for tensor in safetensors.tensors() {
        let expected_shape = expected
            .iter()
            .find(|(name, _)| name == &tensor.name)
            .map(|(_, shape)| shape.as_slice())
            .ok_or_else(|| {
                ConvertError::Parse(format!(
                    "unexpected Kyutai decoder tensor `{}`",
                    tensor.name
                ))
            })?;
        let payload = safetensors.tensor_bytes(tensor);
        validate_payload(
            &tensor.name,
            tensor.dtype,
            &tensor.shape,
            payload,
            expected_shape,
        )?;
        builder.add_tensor(
            &tensor.name,
            GgmlType::BF16,
            tensor.shape.clone(),
            payload.to_vec(),
        )?;
        report.written += 1;
        report.bf16_passthrough += 1;
    }
    if report.written != expected.len() || report.bf16_passthrough != expected.len() {
        return Err(ConvertError::Parse(format!(
            "Kyutai decoder report invariant failed: wrote {} tensors / {} BF16 passthrough; expected {}",
            report.written,
            report.bf16_passthrough,
            expected.len()
        )));
    }
    report.notes.push(
        "decoder-component-only: 323 BF16 tensors preserved verbatim; Mimi, tokenizer, streaming state, and public ASR remain fail-closed".to_owned(),
    );
    Ok((builder, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    #[test]
    fn official_decoder_manifest_is_exactly_323_tensors() {
        let specs = expected_specs().expect("manifest");
        assert_eq!(specs.len(), 323);
        assert_eq!(specs[0], ("text_emb.weight".to_owned(), vec![4001, 2048]));
        assert_eq!(
            specs[33],
            (
                "transformer.layers.0.self_attn.in_proj_weight".to_owned(),
                vec![6144, 2048]
            )
        );
        assert_eq!(
            specs[35],
            (
                "transformer.layers.0.gating.linear_in.weight".to_owned(),
                vec![11264, 2048]
            )
        );
        assert_eq!(
            specs[36],
            (
                "transformer.layers.0.gating.linear_out.weight".to_owned(),
                vec![2048, 5632]
            )
        );
        assert_eq!(
            specs[322],
            ("text_linear.weight".to_owned(), vec![4000, 2048])
        );
    }

    #[test]
    fn manifest_rejects_missing_extra_and_duplicate_names() {
        let expected = expected_specs().expect("manifest");
        let mut names: Vec<String> = expected.iter().map(|(name, _)| name.clone()).collect();
        names.pop();
        assert!(validate_names(&names, &expected).is_err());
        names.push("unexpected.weight".to_owned());
        assert!(validate_names(&names, &expected).is_err());
        names[0] = names[1].clone();
        assert!(validate_names(&names, &expected).is_err());
    }

    #[test]
    fn payload_gate_rejects_wrong_dtype_shape_size_and_nonfinite_bf16() {
        let shape = vec![2, 2];
        let finite = [0u8; 8];
        assert!(validate_payload("x", GgmlType::F32, &shape, &finite, &shape).is_err());
        assert!(validate_payload("x", GgmlType::BF16, &[4, 1], &finite, &shape).is_err());
        assert!(validate_payload("x", GgmlType::BF16, &shape, &[0u8; 2], &shape).is_err());
        let nan = 0x7fc0u16.to_le_bytes();
        let nonfinite = [nan[0], nan[1], 0, 0, 0, 0, 0, 0];
        assert!(validate_payload("x", GgmlType::BF16, &shape, &nonfinite, &shape).is_err());
    }

    #[test]
    fn provenance_and_fixed_config_are_canonical() {
        assert_eq!(ARCH, "kyutai-stt");
        assert_eq!(NAME, "kyutai-stt-2.6b-en");
        assert_eq!(PROVENANCE_LICENSE, "cc-by-4.0");
        assert_eq!(
            LicenseClass::AttributionRequired.as_str(),
            "attribution-required"
        );
        assert_eq!(PROVENANCE_MODEL_ID, "kyutai/stt-2.6b-en");
        assert_eq!(
            PROVENANCE_SOURCE,
            "https://huggingface.co/kyutai/stt-2.6b-en"
        );
        assert_eq!(BB_FFN_HIDDEN, 5632);
        assert_eq!(DEP_Q, 0);
        assert_eq!(N_DELAYS, 33);
    }

    #[test]
    fn tokenizer_converter_is_exact_and_rejects_unknown_piece_types() {
        assert!(validate_tokenizer_model(&[]).is_err());
        assert!(piece_type_value(PieceType::Unspecified).is_err());
        assert!(piece_type_value(PieceType::Other(99)).is_err());
        assert_eq!(piece_type_value(PieceType::Unused).unwrap(), 5);
        assert_eq!(piece_type_value(PieceType::Byte).unwrap(), 6);
        assert_eq!(TOKENIZER_ARCH, "kyutai-stt-tokenizer");
        assert_eq!(TOKENIZER_ASSET_NAME, "tokenizer_en_audio_4000.model");
    }

    #[test]
    fn emitted_metadata_is_strict_and_component_scoped() {
        let builder = component_builder();
        let file = GgufFile::parse(builder.to_bytes().expect("metadata-only GGUF"))
            .expect("parse metadata-only GGUF");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH)
                .and_then(|value| value.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME)
                .and_then(|value| value.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|value| value.as_str()),
            Some(PROVENANCE_LICENSE)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|value| value.as_str()),
            Some("attribution-required")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|value| value.as_str()),
            Some(PROVENANCE_MODEL_ID)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_SOURCE)
                .and_then(|value| value.as_str()),
            Some(PROVENANCE_SOURCE)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_ATTRIBUTION)
                .and_then(|value| value.as_str()),
            Some(ATTRIBUTION)
        );
        for (key, value) in [
            (KEY_SAMPLE_RATE, SAMPLE_RATE),
            (KEY_BB_N_LAYER, BB_N_LAYER),
            (KEY_BB_D_MODEL, BB_D_MODEL),
            (KEY_BB_N_HEAD, BB_N_HEAD),
            (KEY_BB_FFN_HIDDEN, BB_FFN_HIDDEN),
            (KEY_BB_CONTEXT, BB_CONTEXT),
            (KEY_BB_CAUSAL, BB_CAUSAL),
            (KEY_DEP_N_LAYER, DEP_N_LAYER),
            (KEY_DEP_D_MODEL, DEP_D_MODEL),
            (KEY_DEP_N_HEAD, DEP_N_HEAD),
            (KEY_DEP_MULTI_LINEAR, DEP_MULTI_LINEAR),
            (KEY_DEP_WEIGHTS_PER_STEP, DEP_WEIGHTS_PER_STEP),
            (KEY_N_Q, N_Q),
            (KEY_DEP_Q, DEP_Q),
            (KEY_AUDIO_CARD, AUDIO_CARD),
            (KEY_TEXT_CARD, TEXT_CARD),
            (KEY_TEXT_PAD_ID, TEXT_PAD_ID),
            (KEY_N_DELAYS, N_DELAYS),
        ] {
            assert_eq!(file.get(key), Some(&GgufMetadataValue::U32(value)), "{key}");
        }
        for (key, value) in [
            (KEY_BB_HIDDEN_SCALE, BB_HIDDEN_SCALE),
            (KEY_BB_ROPE_MAX_PERIOD, BB_ROPE_MAX_PERIOD),
            (KEY_BB_RMS_NORM_EPS, BB_RMS_NORM_EPS),
            (KEY_AUDIO_DELAY_SECS, AUDIO_DELAY_SECS),
            (KEY_AUDIO_SILENCE_PREFIX_SECS, AUDIO_SILENCE_PREFIX_SECS),
        ] {
            assert_eq!(file.get(key), Some(&GgufMetadataValue::F32(value)), "{key}");
        }
        for index in 0..N_DELAYS {
            assert_eq!(
                file.get(&format!("{PREFIX_DELAY}{index}")),
                Some(&GgufMetadataValue::U32(0))
            );
        }
    }
}
