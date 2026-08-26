//! Strict Microsoft SpeechT5 TTS conversion.
//!
//! The released checkpoint is a 393-float-tensor encoder/decoder after the
//! prepare tool explicitly removes five named integer BatchNorm counters.
//! Conversion is total and fail-closed: the exact
//! names, shapes, F32 dtypes, fixed upstream revision and exact 79-piece
//! `spm_char.model` are required. The two fixed Hugging Face added tokens
//! (`<mask>` and `<ctc_blank>`) extend that base vocabulary to the model's
//! 81 embedding rows. The runtime never parses torch pickle or a generic
//! SentencePiece protobuf; both are offline-converter concerns.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;
use crate::spm_proto::{PieceType, parse_model};

pub(crate) const ARCH: &str = "speecht5";
pub(crate) const NAME: &str = "speecht5-tts";
pub(crate) const CATEGORY: &str = "tts";
pub(crate) const UPSTREAM_HF: &str = "microsoft/speecht5_tts";
pub(crate) const UPSTREAM_REVISION: &str = "30fcde30f19b87502b8435427b5f5068e401d5f6";
/// Git-LFS/Xet content SHA-256 for `pytorch_model.bin` at the pinned revision.
pub(crate) const SOURCE_WEIGHT_SHA256: &str =
    "d60d28067349ef66b50d8cd643ae56b6d6b8f27def929bc4ef6fcad907954190";
/// SHA-256 for the exact upstream `spm_char.model`.
pub(crate) const TOKENIZER_MODEL_SHA256: &str =
    "7fcc48f3e225f627b1641db410ceb0c8649bd2b0c982e150b03f8be3728ab560";
/// SHA-256 for `added_tokens.json` at the pinned upstream revision.
pub(crate) const TOKENIZER_ADDED_TOKENS_SHA256: &str =
    "74be21ecff0a1fb1f304fe7c72ab21e4f0c046f8359fdf2852eb1b80967069ad";
/// SHA-256 over each final tokenizer entry encoded as
/// `u32-le byte_len || UTF-8 piece || f32-le score`.
pub(crate) const TOKENIZER_VOCAB_MANIFEST_SHA256: &str =
    "2b04363543fae9615b30cc91e1b0ed76fba73f91dd23aefb60eed984dc85ee96";
/// Canonical hash over the 393 inference tensor names and shapes.
pub(crate) const TENSOR_MANIFEST_SHA256: &str =
    "fd6a1323b4994781daf6b657e690cca1e741ee2f7810fab03d0d22bf62301e04";
pub(crate) const DEFAULT_LICENSE: &str = "mit";

pub(crate) const HIDDEN_SIZE: u32 = 768;
pub(crate) const ENCODER_LAYERS: u32 = 12;
pub(crate) const DECODER_LAYERS: u32 = 6;
pub(crate) const ENCODER_ATTENTION_HEADS: u32 = 12;
pub(crate) const DECODER_ATTENTION_HEADS: u32 = 12;
pub(crate) const ENCODER_FFN_DIM: u32 = 3_072;
pub(crate) const DECODER_FFN_DIM: u32 = 3_072;
pub(crate) const VOCAB_SIZE: u32 = 81;
pub(crate) const NUM_MEL_BINS: u32 = 80;
pub(crate) const REDUCTION_FACTOR: u32 = 2;
pub(crate) const SPEECH_DECODER_PRENET_UNITS: u32 = 256;
pub(crate) const SPEECH_DECODER_PRENET_LAYERS: u32 = 2;
pub(crate) const SPEECH_DECODER_POSTNET_UNITS: u32 = 256;
pub(crate) const SPEECH_DECODER_POSTNET_LAYERS: u32 = 5;
pub(crate) const SPEECH_DECODER_POSTNET_KERNEL: u32 = 5;
pub(crate) const SPEAKER_EMBEDDING_DIM: u32 = 512;
pub(crate) const MAX_TEXT_POSITIONS: u32 = 600;
pub(crate) const MAX_SPEECH_POSITIONS: u32 = 1_876;
pub(crate) const ENCODER_MAX_RELATIVE_POSITION: u32 = 160;
pub(crate) const PAD_TOKEN_ID: u32 = 1;
pub(crate) const EOS_TOKEN_ID: u32 = 2;
pub(crate) const UNK_TOKEN_ID: u32 = 3;
pub(crate) const MASK_TOKEN_ID: u32 = 79;
pub(crate) const CTC_BLANK_TOKEN_ID: u32 = 80;

const TENSOR_COUNT: usize = 393;
const SOURCE_TENSOR_COUNT: usize = 393;
const TOKENIZER_BASE_PIECES: usize = 79;
const TOKENIZER_PIECES: usize = VOCAB_SIZE as usize;
const TOKENIZER_PREFIX: &str = "vokra.speecht5.tokenizer";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_SOURCE_REVISION: &str = "vokra.speecht5.source_revision";
const KEY_SOURCE_WEIGHT_SHA256: &str = "vokra.speecht5.source_weight_sha256";
const KEY_TENSOR_MANIFEST_SHA256: &str = "vokra.speecht5.tensor_manifest_sha256";
const KEY_TOKENIZER_MODEL_SHA256: &str = "vokra.speecht5.tokenizer.model_sha256";
const KEY_TOKENIZER_ADDED_TOKENS_SHA256: &str = "vokra.speecht5.tokenizer.added_tokens_sha256";
const KEY_TOKENIZER_VOCAB_MANIFEST_SHA256: &str = "vokra.speecht5.tokenizer.vocab_manifest_sha256";
const KEY_EXCLUDED_BATCH_COUNTERS: &str = "vokra.speecht5.excluded_batch_norm_counters";
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
const KEY_SPEECH_DECODER_PRENET_LAYERS: &str = "vokra.speecht5.speech_decoder_prenet_layers";
const KEY_SPEECH_DECODER_PRENET_DROPOUT: &str = "vokra.speecht5.speech_decoder_prenet_dropout";
const KEY_SPEECH_DECODER_POSTNET_UNITS: &str = "vokra.speecht5.speech_decoder_postnet_units";
const KEY_SPEECH_DECODER_POSTNET_LAYERS: &str = "vokra.speecht5.speech_decoder_postnet_layers";
const KEY_SPEECH_DECODER_POSTNET_KERNEL: &str = "vokra.speecht5.speech_decoder_postnet_kernel";
const KEY_SPEECH_DECODER_POSTNET_DROPOUT: &str = "vokra.speecht5.speech_decoder_postnet_dropout";
const KEY_SPEAKER_EMBEDDING_DIM: &str = "vokra.speecht5.speaker_embedding_dim";
const KEY_MAX_TEXT_POSITIONS: &str = "vokra.speecht5.max_text_positions";
const KEY_MAX_SPEECH_POSITIONS: &str = "vokra.speecht5.max_speech_positions";
const KEY_ENCODER_MAX_RELATIVE_POSITION: &str = "vokra.speecht5.encoder_max_relative_position";
const KEY_LAYER_NORM_EPS: &str = "vokra.speecht5.layer_norm_eps";
const KEY_PAD_TOKEN_ID: &str = "vokra.speecht5.pad_token_id";
const KEY_EOS_TOKEN_ID: &str = "vokra.speecht5.eos_token_id";
const KEY_GENERATION_MAXLEN_RATIO: &str = "vokra.speecht5.generation.maxlen_ratio";
const KEY_GENERATION_STOP_THRESHOLD: &str = "vokra.speecht5.generation.stop_threshold";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
/// Counters from one successful strict SpeechT5 TTS conversion.
pub struct SpeechT5Report {
    /// All prepared float tensors.
    pub read: usize,
    /// Exact inference tensors written to GGUF.
    pub written: usize,
    /// Always zero; the pinned prepare tool removes the five named integer
    /// `num_batches_tracked` scalars before this float-only reader runs.
    pub skipped_non_float: usize,
    /// Always zero for the pinned F32 release.
    pub bf16_passthrough: usize,
    /// True only on the strict side-car entry point.
    pub tokenizer_embedded: bool,
}

/// The compatibility entry point is intentionally unusable: a runnable text
/// TTS artifact requires the exact tokenizer side-car.
pub fn convert_speecht5_file(
    _input: &Path,
    _output: &Path,
    _license: Option<&str>,
) -> Result<SpeechT5Report, ConvertError> {
    Err(ConvertError::Usage(
        "speecht5-tts requires the exact upstream spm_char.model; use \
         convert_speecht5_file_with_tokenizer (CLI: --tokenizer spm_char.model)"
            .to_owned(),
    ))
}

/// Convert the pinned complete release and embed its exact SentencePiece
/// vocabulary as dependency-free GGUF metadata.
pub fn convert_speecht5_file_with_tokenizer(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    tokenizer_model: &Path,
) -> Result<SpeechT5Report, ConvertError> {
    validate_license_override(license)?;
    let tokenizer_bytes = std::fs::read(tokenizer_model)?;
    let tokenizer = validate_tokenizer(&tokenizer_bytes)?;
    let bytes = std::fs::read(input)?;
    let checkpoint = SafetensorsFile::parse(bytes)?;
    validate_checkpoint(&checkpoint)?;

    let mut builder = GgufBuilder::new();
    stamp_model_metadata(&mut builder);
    stamp_tokenizer_metadata(&mut builder, &tokenizer);

    let mut report = SpeechT5Report {
        read: checkpoint.tensors().len(),
        tokenizer_embedded: true,
        ..SpeechT5Report::default()
    };
    for tensor in checkpoint.tensors() {
        builder
            .add_tensor(
                &tensor.name,
                tensor.dtype,
                tensor.shape.clone(),
                checkpoint.tensor_bytes(tensor).to_vec(),
            )
            .map_err(|error| ConvertError::Gguf(error.to_string()))?;
        report.written += 1;
    }
    debug_assert_eq!(report.written, TENSOR_COUNT);
    debug_assert_eq!(report.skipped_non_float, 0);

    let output_bytes = builder
        .to_bytes()
        .map_err(|error| ConvertError::Gguf(error.to_string()))?;
    std::fs::write(output, output_bytes)?;
    Ok(report)
}

fn validate_license_override(license: Option<&str>) -> Result<(), ConvertError> {
    if let Some(value) = license.filter(|value| !value.is_empty()) {
        if !value.eq_ignore_ascii_case(DEFAULT_LICENSE) {
            return Err(ConvertError::Usage(format!(
                "speecht5-tts is pinned to the official {DEFAULT_LICENSE} checkpoint; refusing license override {value:?}"
            )));
        }
    }
    Ok(())
}

fn stamp_model_metadata(builder: &mut GgufBuilder) {
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    builder.add_string(KEY_UPSTREAM_HF, UPSTREAM_HF);
    builder.add_string(KEY_SOURCE_REVISION, UPSTREAM_REVISION);
    builder.add_string(KEY_SOURCE_WEIGHT_SHA256, SOURCE_WEIGHT_SHA256);
    builder.add_string(KEY_TENSOR_MANIFEST_SHA256, TENSOR_MANIFEST_SHA256);
    builder.add_string(KEY_TOKENIZER_MODEL_SHA256, TOKENIZER_MODEL_SHA256);
    builder.add_u32(KEY_EXCLUDED_BATCH_COUNTERS, 5);

    for (key, value) in [
        (KEY_HIDDEN_SIZE, HIDDEN_SIZE),
        (KEY_ENCODER_LAYERS, ENCODER_LAYERS),
        (KEY_DECODER_LAYERS, DECODER_LAYERS),
        (KEY_ENCODER_ATTENTION_HEADS, ENCODER_ATTENTION_HEADS),
        (KEY_DECODER_ATTENTION_HEADS, DECODER_ATTENTION_HEADS),
        (KEY_ENCODER_FFN_DIM, ENCODER_FFN_DIM),
        (KEY_DECODER_FFN_DIM, DECODER_FFN_DIM),
        (KEY_VOCAB_SIZE, VOCAB_SIZE),
        (KEY_NUM_MEL_BINS, NUM_MEL_BINS),
        (KEY_REDUCTION_FACTOR, REDUCTION_FACTOR),
        (KEY_SPEECH_DECODER_PRENET_UNITS, SPEECH_DECODER_PRENET_UNITS),
        (
            KEY_SPEECH_DECODER_PRENET_LAYERS,
            SPEECH_DECODER_PRENET_LAYERS,
        ),
        (
            KEY_SPEECH_DECODER_POSTNET_UNITS,
            SPEECH_DECODER_POSTNET_UNITS,
        ),
        (
            KEY_SPEECH_DECODER_POSTNET_LAYERS,
            SPEECH_DECODER_POSTNET_LAYERS,
        ),
        (
            KEY_SPEECH_DECODER_POSTNET_KERNEL,
            SPEECH_DECODER_POSTNET_KERNEL,
        ),
        (KEY_SPEAKER_EMBEDDING_DIM, SPEAKER_EMBEDDING_DIM),
        (KEY_MAX_TEXT_POSITIONS, MAX_TEXT_POSITIONS),
        (KEY_MAX_SPEECH_POSITIONS, MAX_SPEECH_POSITIONS),
        (
            KEY_ENCODER_MAX_RELATIVE_POSITION,
            ENCODER_MAX_RELATIVE_POSITION,
        ),
        (KEY_PAD_TOKEN_ID, PAD_TOKEN_ID),
        (KEY_EOS_TOKEN_ID, EOS_TOKEN_ID),
    ] {
        builder.add_u32(key, value);
    }
    builder.add_f32(KEY_LAYER_NORM_EPS, 1.0e-5);
    builder.add_f32(KEY_SPEECH_DECODER_PRENET_DROPOUT, 0.5);
    builder.add_f32(KEY_SPEECH_DECODER_POSTNET_DROPOUT, 0.5);
    builder.add_f32(KEY_GENERATION_MAXLEN_RATIO, 20.0);
    builder.add_f32(KEY_GENERATION_STOP_THRESHOLD, 0.5);

    vokra_core::stamp_provenance(
        builder,
        LicenseClass::Permissive,
        DEFAULT_LICENSE,
        Some(NAME),
        Some(UPSTREAM_HF),
    );
}

#[derive(Debug)]
struct TokenizerMetadata {
    pieces: Vec<String>,
    scores: Vec<f32>,
    model_bytes: Vec<u8>,
}

fn validate_tokenizer(bytes: &[u8]) -> Result<TokenizerMetadata, ConvertError> {
    let actual = super::canary_1b_flash::hex(&super::canary_1b_flash::sha256(bytes));
    if actual != TOKENIZER_MODEL_SHA256 {
        return Err(ConvertError::Parse(format!(
            "SpeechT5 spm_char.model SHA-256 {actual}, expected {TOKENIZER_MODEL_SHA256}"
        )));
    }
    let model = parse_model(bytes)
        .map_err(|error| ConvertError::Parse(format!("SpeechT5 spm_char.model: {error}")))?;
    if model.pieces.len() != TOKENIZER_BASE_PIECES {
        return Err(ConvertError::Parse(format!(
            "SpeechT5 spm_char.model has {} pieces, expected {TOKENIZER_BASE_PIECES}",
            model.pieces.len()
        )));
    }
    for (id, expected, expected_type) in [
        (0usize, "<s>", PieceType::Control),
        (1usize, "<pad>", PieceType::Control),
        (2usize, "</s>", PieceType::Control),
        (3usize, "<unk>", PieceType::Unknown),
    ] {
        let piece = &model.pieces[id];
        if piece.piece != expected || piece.piece_type != expected_type {
            return Err(ConvertError::Parse(format!(
                "SpeechT5 tokenizer id {id} is {:?}/{:?}, expected {expected:?}/{expected_type:?}",
                piece.piece, piece.piece_type
            )));
        }
    }
    let mut pieces = model
        .pieces
        .iter()
        .map(|piece| piece.piece.clone())
        .collect::<Vec<_>>();
    let mut scores = model
        .pieces
        .iter()
        .map(|piece| piece.score)
        .collect::<Vec<_>>();
    // `added_tokens.json` at the pinned revision is exactly
    // `{ "<mask>": 79, "<ctc_blank>": 80 }`. These tokens are not part
    // of the 79-piece SentencePiece ModelProto, but they account for the two
    // final rows of the checkpoint's 81-row embedding table.
    pieces.extend(["<mask>".to_owned(), "<ctc_blank>".to_owned()]);
    scores.extend([0.0, 0.0]);
    debug_assert_eq!(pieces.len(), TOKENIZER_PIECES);
    Ok(TokenizerMetadata {
        pieces,
        scores,
        model_bytes: bytes.to_vec(),
    })
}

fn stamp_tokenizer_metadata(builder: &mut GgufBuilder, tokenizer: &TokenizerMetadata) {
    builder.add_string(&format!("{TOKENIZER_PREFIX}.scheme"), "char");
    builder.add_string(&format!("{TOKENIZER_PREFIX}.kind"), "sentencepiece-char");
    builder.add_string(
        KEY_TOKENIZER_ADDED_TOKENS_SHA256,
        TOKENIZER_ADDED_TOKENS_SHA256,
    );
    builder.add_string(
        KEY_TOKENIZER_VOCAB_MANIFEST_SHA256,
        TOKENIZER_VOCAB_MANIFEST_SHA256,
    );
    builder.add_u32(
        &format!("{TOKENIZER_PREFIX}.base_vocab_size"),
        TOKENIZER_BASE_PIECES as u32,
    );
    builder.add_string(&format!("{TOKENIZER_PREFIX}.normalizer"), "nmt_nfkc");
    builder.add_bool(
        &format!("{TOKENIZER_PREFIX}.normalizer.add_dummy_prefix"),
        true,
    );
    builder.add_bool(
        &format!("{TOKENIZER_PREFIX}.normalizer.remove_extra_whitespaces"),
        true,
    );
    builder.add_bool(
        &format!("{TOKENIZER_PREFIX}.normalizer.escape_whitespaces"),
        true,
    );
    builder.add_metadata(
        &format!("{TOKENIZER_PREFIX}.model"),
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U8,
            values: tokenizer
                .model_bytes
                .iter()
                .map(|&byte| GgufMetadataValue::U8(byte))
                .collect(),
        }),
    );
    builder.add_metadata(
        &format!("{TOKENIZER_PREFIX}.pieces"),
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: tokenizer
                .pieces
                .iter()
                .map(|piece| GgufMetadataValue::String(piece.clone()))
                .collect(),
        }),
    );
    builder.add_metadata(
        &format!("{TOKENIZER_PREFIX}.scores"),
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::F32,
            values: tokenizer
                .scores
                .iter()
                .map(|score| GgufMetadataValue::F32(*score))
                .collect(),
        }),
    );
    builder.add_u32(&format!("{TOKENIZER_PREFIX}.unk_id"), UNK_TOKEN_ID);
    builder.add_u32(&format!("{TOKENIZER_PREFIX}.bos_id"), 0);
    builder.add_u32(&format!("{TOKENIZER_PREFIX}.pad_id"), PAD_TOKEN_ID);
    builder.add_u32(&format!("{TOKENIZER_PREFIX}.eos_id"), EOS_TOKEN_ID);
    builder.add_u32(&format!("{TOKENIZER_PREFIX}.mask_id"), MASK_TOKEN_ID);
    builder.add_u32(
        &format!("{TOKENIZER_PREFIX}.ctc_blank_id"),
        CTC_BLANK_TOKEN_ID,
    );
    builder.add_u32(&format!("{TOKENIZER_PREFIX}.vocab_size"), VOCAB_SIZE);
}

fn validate_checkpoint(checkpoint: &SafetensorsFile) -> Result<(), ConvertError> {
    let expected = expected_manifest();
    let internal_hash =
        super::canary_1b_flash::hex(&super::canary_1b_flash::manifest_sha256(&expected));
    if expected.len() != TENSOR_COUNT || internal_hash != TENSOR_MANIFEST_SHA256 {
        return Err(ConvertError::Parse(format!(
            "SpeechT5 internal manifest drift: count={}, sha256={internal_hash}",
            expected.len()
        )));
    }

    let expected_names = expected.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let actual_names = checkpoint
        .tensors()
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<BTreeSet<_>>();
    if actual_names != expected_names || checkpoint.tensors().len() != SOURCE_TENSOR_COUNT {
        let missing = expected_names
            .difference(&actual_names)
            .take(8)
            .copied()
            .collect::<Vec<_>>();
        let extra = actual_names
            .difference(&expected_names)
            .take(8)
            .copied()
            .collect::<Vec<_>>();
        return Err(ConvertError::Parse(format!(
            "SpeechT5 prepared checkpoint manifest mismatch: found {}, expected \
             {SOURCE_TENSOR_COUNT}; missing={missing:?}, extra={extra:?}",
            checkpoint.tensors().len()
        )));
    }

    for tensor in checkpoint.tensors() {
        let expected_shape = &expected[&tensor.name];
        if &tensor.shape != expected_shape || tensor.dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "SpeechT5 tensor {:?} is {:?} {:?}, expected F32 {:?}",
                tensor.name, tensor.dtype, tensor.shape, expected_shape
            )));
        }
    }
    Ok(())
}

fn expected_manifest() -> BTreeMap<String, Vec<u64>> {
    let mut tensors = BTreeMap::new();
    let mut add = |name: String, shape: &[u64]| {
        assert!(tensors.insert(name, shape.to_vec()).is_none());
    };

    add("speech_decoder_postnet.feat_out.bias".into(), &[160]);
    add("speech_decoder_postnet.feat_out.weight".into(), &[160, 768]);
    for layer in 0..5 {
        let channels = if layer == 4 { 80 } else { 256 };
        let input_channels = if layer == 0 { 80 } else { 256 };
        let prefix = format!("speech_decoder_postnet.layers.{layer}");
        for name in ["bias", "running_mean", "running_var", "weight"] {
            add(format!("{prefix}.batch_norm.{name}"), &[channels]);
        }
        add(
            format!("{prefix}.conv.weight"),
            &[channels, input_channels, 5],
        );
    }
    add("speech_decoder_postnet.prob_out.bias".into(), &[2]);
    add("speech_decoder_postnet.prob_out.weight".into(), &[2, 768]);

    add("speecht5.decoder.prenet.encode_positions.alpha".into(), &[]);
    add(
        "speecht5.decoder.prenet.encode_positions.pe".into(),
        &[1, 1_876, 768],
    );
    add("speecht5.decoder.prenet.final_layer.bias".into(), &[768]);
    add(
        "speecht5.decoder.prenet.final_layer.weight".into(),
        &[768, 256],
    );
    for layer in 0..2 {
        let input = if layer == 0 { 80 } else { 256 };
        add(
            format!("speecht5.decoder.prenet.layers.{layer}.bias"),
            &[256],
        );
        add(
            format!("speecht5.decoder.prenet.layers.{layer}.weight"),
            &[256, input],
        );
    }
    add(
        "speecht5.decoder.prenet.speaker_embeds_layer.bias".into(),
        &[768],
    );
    add(
        "speecht5.decoder.prenet.speaker_embeds_layer.weight".into(),
        &[768, 1_280],
    );
    for layer in 0..6 {
        let prefix = format!("speecht5.decoder.wrapped_decoder.layers.{layer}");
        for attention in ["encoder_attn", "self_attn"] {
            for projection in ["k_proj", "out_proj", "q_proj", "v_proj"] {
                add(format!("{prefix}.{attention}.{projection}.bias"), &[768]);
                add(
                    format!("{prefix}.{attention}.{projection}.weight"),
                    &[768, 768],
                );
            }
        }
        for norm in [
            "encoder_attn_layer_norm",
            "final_layer_norm",
            "self_attn_layer_norm",
        ] {
            add(format!("{prefix}.{norm}.bias"), &[768]);
            add(format!("{prefix}.{norm}.weight"), &[768]);
        }
        add(
            format!("{prefix}.feed_forward.intermediate_dense.bias"),
            &[3_072],
        );
        add(
            format!("{prefix}.feed_forward.intermediate_dense.weight"),
            &[3_072, 768],
        );
        add(format!("{prefix}.feed_forward.output_dense.bias"), &[768]);
        add(
            format!("{prefix}.feed_forward.output_dense.weight"),
            &[768, 3_072],
        );
    }

    add(
        "speecht5.encoder.prenet.embed_tokens.weight".into(),
        &[81, 768],
    );
    add("speecht5.encoder.prenet.encode_positions.alpha".into(), &[]);
    add(
        "speecht5.encoder.prenet.encode_positions.pe".into(),
        &[1, 600, 768],
    );
    add(
        "speecht5.encoder.wrapped_encoder.embed_positions.pe_k.weight".into(),
        &[320, 64],
    );
    add(
        "speecht5.encoder.wrapped_encoder.layer_norm.bias".into(),
        &[768],
    );
    add(
        "speecht5.encoder.wrapped_encoder.layer_norm.weight".into(),
        &[768],
    );
    for layer in 0..12 {
        let prefix = format!("speecht5.encoder.wrapped_encoder.layers.{layer}");
        for projection in ["k_proj", "out_proj", "q_proj", "v_proj"] {
            add(format!("{prefix}.attention.{projection}.bias"), &[768]);
            add(
                format!("{prefix}.attention.{projection}.weight"),
                &[768, 768],
            );
        }
        for norm in ["final_layer_norm", "layer_norm"] {
            add(format!("{prefix}.{norm}.bias"), &[768]);
            add(format!("{prefix}.{norm}.weight"), &[768]);
        }
        add(
            format!("{prefix}.feed_forward.intermediate_dense.bias"),
            &[3_072],
        );
        add(
            format!("{prefix}.feed_forward.intermediate_dense.weight"),
            &[3_072, 768],
        );
        add(format!("{prefix}.feed_forward.output_dense.bias"), &[768]);
        add(
            format!("{prefix}.feed_forward.output_dense.weight"),
            &[768, 3_072],
        );
    }

    tensors
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    fn scratch_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "vokra-speecht5-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or_default()
        ))
    }

    #[test]
    fn official_manifest_count_and_hash_are_pinned() {
        let manifest = expected_manifest();
        assert_eq!(manifest.len(), TENSOR_COUNT);
        assert_eq!(
            super::super::canary_1b_flash::hex(&super::super::canary_1b_flash::manifest_sha256(
                &manifest
            )),
            TENSOR_MANIFEST_SHA256
        );
        assert_eq!(
            manifest["speecht5.encoder.wrapped_encoder.embed_positions.pe_k.weight"],
            vec![320, 64]
        );
        assert_eq!(
            manifest["speech_decoder_postnet.layers.4.conv.weight"],
            vec![80, 256, 5]
        );
    }

    #[test]
    fn compatibility_entrypoint_requires_tokenizer() {
        let error = convert_speecht5_file(Path::new("missing"), Path::new("out"), None)
            .expect_err("weight-only conversion must fail");
        assert!(error.to_string().contains("spm_char.model"));
    }

    #[test]
    fn wrong_tokenizer_fails_before_checkpoint_read() {
        let tokenizer = scratch_path("bad-tokenizer");
        let output = scratch_path("bad-tokenizer-output");
        std::fs::write(&tokenizer, b"not the pinned tokenizer").unwrap();
        let error = convert_speecht5_file_with_tokenizer(
            Path::new("definitely-missing-checkpoint"),
            &output,
            None,
            &tokenizer,
        )
        .expect_err("wrong tokenizer must fail first");
        assert!(error.to_string().contains("SHA-256"));
        assert!(!output.exists());
        std::fs::remove_file(tokenizer).ok();
    }

    #[test]
    fn official_tokenizer_sidecar_if_configured() {
        let Some(path) = std::env::var_os("VOKRA_SPEECHT5_TOKENIZER") else {
            return;
        };
        let bytes = std::fs::read(path).expect("read configured SpeechT5 tokenizer");
        let tokenizer = validate_tokenizer(&bytes).expect("bind official SpeechT5 tokenizer");
        assert_eq!(tokenizer.pieces.len(), 81);
        assert_eq!(tokenizer.pieces[0], "<s>");
        assert_eq!(tokenizer.pieces[3], "<unk>");
        assert_eq!(tokenizer.pieces[79], "<mask>");
        assert_eq!(tokenizer.pieces[80], "<ctc_blank>");
        assert_eq!(tokenizer.model_bytes, bytes);
    }

    #[test]
    fn complete_runtime_metadata_is_stamped() {
        let mut builder = GgufBuilder::new();
        stamp_model_metadata(&mut builder);
        stamp_tokenizer_metadata(
            &mut builder,
            &TokenizerMetadata {
                pieces: (0..TOKENIZER_PIECES)
                    .map(|index| format!("piece-{index}"))
                    .collect(),
                scores: vec![0.0; TOKENIZER_PIECES],
                model_bytes: vec![0x08, 0x01],
            },
        );
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert_eq!(
            file.get(KEY_SOURCE_REVISION)
                .and_then(|value| value.as_str()),
            Some(UPSTREAM_REVISION)
        );
        assert_eq!(
            file.get(KEY_TENSOR_MANIFEST_SHA256)
                .and_then(|value| value.as_str()),
            Some(TENSOR_MANIFEST_SHA256)
        );
        assert_eq!(
            file.get(KEY_MAX_TEXT_POSITIONS)
                .and_then(|value| value.as_u64()),
            Some(600)
        );
        assert_eq!(
            file.get(KEY_SPEECH_DECODER_PRENET_LAYERS)
                .and_then(|value| value.as_u64()),
            Some(2)
        );
        assert_eq!(
            file.get(&format!("{TOKENIZER_PREFIX}.vocab_size"))
                .and_then(|value| value.as_u64()),
            Some(81)
        );
        assert_eq!(
            file.get(&format!("{TOKENIZER_PREFIX}.base_vocab_size"))
                .and_then(|value| value.as_u64()),
            Some(79)
        );
        assert_eq!(
            file.get(&format!("{TOKENIZER_PREFIX}.kind"))
                .and_then(|value| value.as_str()),
            Some("sentencepiece-char")
        );
        assert_eq!(
            file.get(&format!("{TOKENIZER_PREFIX}.unk_id"))
                .and_then(|value| value.as_u64()),
            Some(3)
        );
        assert_eq!(
            file.get(&format!("{TOKENIZER_PREFIX}.mask_id"))
                .and_then(|value| value.as_u64()),
            Some(79)
        );
        assert_eq!(
            file.get(&format!("{TOKENIZER_PREFIX}.ctc_blank_id"))
                .and_then(|value| value.as_u64()),
            Some(80)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|value| value.as_str()),
            Some(DEFAULT_LICENSE)
        );
    }

    #[test]
    fn license_override_is_canonical_and_fail_closed() {
        validate_license_override(None).unwrap();
        validate_license_override(Some("")).unwrap();
        validate_license_override(Some("MIT")).unwrap();
        let error = validate_license_override(Some("apache-2.0"))
            .expect_err("a conflicting license must be refused");
        assert!(error.to_string().contains("refusing license override"));
    }
}
