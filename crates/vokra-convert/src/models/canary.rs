//! Strict NVIDIA Canary-1B-v2 main-checkpoint conversion.
//!
//! The upstream `.nemo` contains both a timestamp auxiliary CTC checkpoint
//! and the actual eight-layer Transformer-AED model. Preparation must select
//! `./model_weights.ckpt` explicitly. This converter then accepts only the
//! main checkpoint manifest with exactly 1,478 F32 tensors and the immutable 16,384-line
//! aggregate `tokenizer.vocab`; it never emits a plausible partial GGUF.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Component, Path};
use std::sync::atomic::{AtomicU64, Ordering};

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use super::canary_1b_flash::{expected_canary_aed_manifest, hex, manifest_sha256, sha256};
use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub(crate) const ARCH: &str = "canary";
pub(crate) const NAME: &str = "canary-1b-v2";
pub(crate) const CATEGORY: &str = "asr";
pub(crate) const UPSTREAM_HF: &str = "nvidia/canary-1b-v2";
pub(crate) const DEFAULT_LICENSE: &str = "cc-by-4.0";

pub(crate) const CANARY_ATTRIBUTION_TEXT: &str = "This application uses NVIDIA Canary-1B-v2 (multilingual ASR / AST across 25 languages). Model weights are licensed under CC-BY 4.0. Copyright (c) NVIDIA. Source: https://huggingface.co/nvidia/canary-1b-v2";

const TENSOR_COUNT: usize = 1_478;
const TENSOR_MANIFEST_SHA256: &str =
    "a7a50151cdf5503430492a0d610600ba901c180e249e25e202f7294ddbafae34";
const TOKENIZER_VOCAB_SHA256: &str =
    "4d10723a8bef5b8b186c3d2bb1449c849cc25c6b811969a7d170261b0ceed178";
const VOCAB_SIZE: usize = 16_384;
const SPECIAL_VOCAB_SIZE: usize = 1_163;
static OUTPUT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_SAMPLE_RATE: &str = "vokra.canary.sample_rate";
const KEY_ENC_N_LAYER: &str = "vokra.canary.arch.encoder.n_layer";
const KEY_ENC_D_MODEL: &str = "vokra.canary.arch.encoder.d_model";
const KEY_ENC_N_HEAD: &str = "vokra.canary.arch.encoder.n_head";
const KEY_ENC_N_HEAD_KV: &str = "vokra.canary.arch.encoder.n_head_kv";
const KEY_ENC_FFN_DIM: &str = "vokra.canary.arch.encoder.ffn_dim";
const KEY_ENC_CONV_KERNEL: &str = "vokra.canary.arch.encoder.conv_kernel_size";
const KEY_ENC_IN_DIM: &str = "vokra.canary.arch.encoder.in_dim";
const KEY_ENC_SUB_FACTOR: &str = "vokra.canary.arch.encoder.subsampling_factor";
const KEY_ENC_SUB_KERNEL: &str = "vokra.canary.arch.encoder.subsampling_conv_kernel_size";
const KEY_ENC_SUB_STRIDE: &str = "vokra.canary.arch.encoder.subsampling_conv_stride";
const KEY_ENC_SUB_CHANNELS: &str = "vokra.canary.arch.encoder.subsampling_conv_channels";
const KEY_ENC_MAX_POS: &str = "vokra.canary.arch.encoder.max_position_embeddings";
const KEY_ENC_ATTN_BIAS: &str = "vokra.canary.arch.encoder.attention_bias";
const KEY_ENC_CONV_BIAS: &str = "vokra.canary.arch.encoder.convolution_bias";
const KEY_ENC_SCALE_INPUT: &str = "vokra.canary.arch.encoder.scale_input";
const KEY_DEC_N_LAYER: &str = "vokra.canary.arch.decoder.n_layer";
const KEY_DEC_D_MODEL: &str = "vokra.canary.arch.decoder.d_model";
const KEY_DEC_N_HEAD: &str = "vokra.canary.arch.decoder.n_head";
const KEY_DEC_FFN_DIM: &str = "vokra.canary.arch.decoder.ffn_dim";
const KEY_DEC_MAX_SEQ: &str = "vokra.canary.arch.decoder.max_sequence_length";
const KEY_DEC_PRE_LN: &str = "vokra.canary.arch.decoder.pre_ln";
const KEY_DEC_HIDDEN_ACT: &str = "vokra.canary.arch.decoder.hidden_act";
const KEY_HEAD_VOCAB_SIZE: &str = "vokra.canary.head.vocab_size";
const KEY_HEAD_PAD_ID: &str = "vokra.canary.head.pad_token_id";
const KEY_HEAD_BOS_ID: &str = "vokra.canary.head.bos_token_id";
const KEY_HEAD_EOS_ID: &str = "vokra.canary.head.eos_token_id";
const KEY_SOURCE_REVISION: &str = "vokra.canary.source_revision";
const KEY_SOURCE_NEMO_SHA256: &str = "vokra.canary.source_nemo_sha256";
const KEY_MODEL_CONFIG_SHA256: &str = "vokra.canary.model_config_sha256";
const KEY_DATA_PICKLE_SHA256: &str = "vokra.canary.data_pickle_sha256";
const KEY_TENSOR_MANIFEST_SHA256: &str = "vokra.canary.tensor_manifest_sha256";
const KEY_FRONTEND_N_FFT: &str = "vokra.canary.frontend.n_fft";
const KEY_FRONTEND_HOP: &str = "vokra.canary.frontend.hop_length";
const KEY_FRONTEND_WIN: &str = "vokra.canary.frontend.win_length";
const KEY_FRONTEND_N_MELS: &str = "vokra.canary.frontend.n_mels";
const KEY_FRONTEND_PREEMPHASIS: &str = "vokra.canary.frontend.preemphasis";
const KEY_FRONTEND_WINDOW: &str = "vokra.canary.frontend.window";
const KEY_FRONTEND_WINDOW_PERIODIC: &str = "vokra.canary.frontend.window_periodic";
const KEY_FRONTEND_NORMALIZE: &str = "vokra.canary.frontend.normalize";
const KEY_FRONTEND_PAD_MODE: &str = "vokra.canary.frontend.pad_mode";
const KEY_TOKENIZER_VOCAB: &str = "vokra.canary.tokenizer.vocab";
const KEY_TOKENIZER_VOCAB_SHA256: &str = "vokra.canary.tokenizer.vocab_sha256";

const SOURCE_REVISION: &str = "87bc52657add533cd0156b3fc1aef027280754bf";
const SOURCE_NEMO_SHA256: &str = "ae5ef1bf06812a95a1594a8f5f0ee9c51f35418e5ba96939fa6b98ab00431094";
const MODEL_CONFIG_SHA256: &str =
    "202542a45eb4ad656a47044c5db8c02926259d7232b436d77ca6af21dc84deae";
const DATA_PICKLE_SHA256: &str = "9d8020dacbb2cb97c32614a0460365c8ed7ad3809e942183774e9af94269dba6";

#[derive(Debug, Default, Clone, PartialEq, Eq)]
/// Summary of a strict Canary-1B-v2 main-checkpoint conversion.
pub struct CanaryReport {
    /// Number of authenticated source tensors read.
    pub read: usize,
    /// Number of tensors written to the GGUF.
    pub written: usize,
    /// Number of rejected non-floating tensors (always zero on success).
    pub skipped_non_float: usize,
    /// Number of BF16 tensors passed through (zero for the pinned F32 input).
    pub bf16_passthrough: usize,
}

/// Converts the complete prepared main checkpoint. This path is VAST-only by
/// repository policy because the prepared checkpoint exceeds 2 GB.
pub fn convert_canary_file_with_tokenizer(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    tokenizer_vocab: &Path,
) -> Result<CanaryReport, ConvertError> {
    validate_io_paths(input, tokenizer_vocab, output)?;
    // Authenticate the small tokenizer sidecar before touching or stat'ing
    // the multi-gigabyte checkpoint. This keeps malformed/unauthenticated
    // tokenizer inputs fail-closed without probing the large payload.
    if tokenizer_vocab.is_symlink() || !tokenizer_vocab.is_file() {
        return Err(ConvertError::Usage(format!(
            "canary-1b-v2 tokenizer must be a regular non-symlink file: {}",
            tokenizer_vocab.display()
        )));
    }
    if let Some(value) = license.filter(|value| !value.is_empty()) {
        if !value.eq_ignore_ascii_case(DEFAULT_LICENSE) {
            return Err(ConvertError::Usage(format!(
                "canary-1b-v2 is pinned to the official {DEFAULT_LICENSE} checkpoint; refusing license override {value:?}"
            )));
        }
    }

    let tokenizer = std::fs::read(tokenizer_vocab).map_err(ConvertError::Io)?;
    validate_tokenizer_vocab(&tokenizer)?;

    if output.exists() || output.is_symlink() {
        return Err(ConvertError::Usage(format!(
            "canary-1b-v2 output already exists or is symlinked: {}",
            output.display()
        )));
    }
    if input.is_symlink() || !input.is_file() {
        return Err(ConvertError::Usage(format!(
            "canary-1b-v2 checkpoint must be a regular non-symlink file: {}",
            input.display()
        )));
    }

    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let safetensors = SafetensorsFile::parse(bytes)?;
    validate_checkpoint(&safetensors)?;

    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    builder.add_string(KEY_UPSTREAM_HF, UPSTREAM_HF);
    write_runtime_metadata(&mut builder, &tokenizer);
    vokra_core::stamp_provenance(
        &mut builder,
        LicenseClass::AttributionRequired,
        DEFAULT_LICENSE,
        Some(NAME),
        Some("https://huggingface.co/nvidia/canary-1b-v2"),
    );
    vokra_core::stamp_attribution(&mut builder, CANARY_ATTRIBUTION_TEXT);

    for tensor in safetensors.tensors() {
        builder
            .add_tensor(
                &tensor.name,
                tensor.dtype,
                tensor.shape.clone(),
                safetensors.tensor_bytes(tensor).to_vec(),
            )
            .map_err(|error| ConvertError::Gguf(error.to_string()))?;
    }
    let output_bytes = builder
        .to_bytes()
        .map_err(|error| ConvertError::Gguf(error.to_string()))?;
    write_output_no_clobber(output, &output_bytes)?;

    Ok(CanaryReport {
        read: TENSOR_COUNT,
        written: TENSOR_COUNT,
        skipped_non_float: 0,
        bf16_passthrough: 0,
    })
}

fn write_output_no_clobber(output: &Path, bytes: &[u8]) -> Result<(), ConvertError> {
    let parent = output.parent().ok_or_else(|| {
        ConvertError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "output must have a parent directory",
        ))
    })?;
    let mut temporary = None;
    for _ in 0..32_u64 {
        let sequence = OUTPUT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            ".{}.{}.{}.tmp",
            output
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("output"),
            std::process::id(),
            sequence
        ));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                temporary = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(ConvertError::Io(error)),
        }
    }
    let Some((temporary_path, mut temporary_file)) = temporary else {
        return Err(ConvertError::Io(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not allocate a unique temporary output",
        )));
    };
    let result = (|| {
        temporary_file.write_all(bytes)?;
        temporary_file.sync_all()?;
        std::fs::hard_link(&temporary_path, output)?;
        Ok::<(), std::io::Error>(())
    })();
    drop(temporary_file);
    // hard_link publishes a complete, synced artifact. Cleanup is best effort
    // so a post-publication unlink error never reports a false conversion
    // failure alongside a valid final output.
    match result {
        Ok(()) => {
            let _ = std::fs::remove_file(&temporary_path);
            Ok(())
        }
        Err(error) => {
            let _ = std::fs::remove_file(&temporary_path);
            Err(ConvertError::Io(error))
        }
    }
}

fn validate_io_paths(
    input: &Path,
    tokenizer_vocab: &Path,
    output: &Path,
) -> Result<(), ConvertError> {
    reject_unsafe_path(input, "checkpoint")?;
    reject_unsafe_path(tokenizer_vocab, "tokenizer")?;
    reject_unsafe_path(output, "output")?;
    if tokenizer_vocab.is_symlink() || !tokenizer_vocab.is_file() {
        return Err(ConvertError::Usage(format!(
            "canary-1b-v2 tokenizer must be a regular non-symlink file: {}",
            tokenizer_vocab.display()
        )));
    }
    if output.exists() || output.is_symlink() {
        return Err(ConvertError::Usage(format!(
            "canary-1b-v2 output already exists or is symlinked: {}",
            output.display()
        )));
    }
    let parent = output.parent().ok_or_else(|| {
        ConvertError::Usage("canary-1b-v2 output must have a parent directory".to_owned())
    })?;
    if !parent.is_dir() || parent.is_symlink() {
        return Err(ConvertError::Usage(format!(
            "canary-1b-v2 output parent must be an existing regular non-symlink directory: {}",
            parent.display()
        )));
    }
    Ok(())
}

fn reject_unsafe_path(path: &Path, label: &str) -> Result<(), ConvertError> {
    let raw = path.to_string_lossy();
    #[cfg(windows)]
    let has_lexical_dot = raw
        .split(['/', '\\'])
        .any(|component| matches!(component, "." | ".."));
    #[cfg(not(windows))]
    let has_lexical_dot = raw
        .split('/')
        .any(|component| matches!(component, "." | ".."));
    if has_lexical_dot
        || path
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        return Err(ConvertError::Usage(format!(
            "canary-1b-v2 {label} must not contain lexical dot components"
        )));
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(ConvertError::Io)?
            .join(path)
    };
    let mut current = absolute.as_path();
    loop {
        if current.is_symlink() && current != Path::new("/var") {
            return Err(ConvertError::Usage(format!(
                "canary-1b-v2 {label} has symlink ancestry: {}",
                current.display()
            )));
        }
        let Some(parent) = current.parent() else {
            break;
        };
        if parent == current {
            break;
        }
        current = parent;
    }
    Ok(())
}

fn write_runtime_metadata(builder: &mut GgufBuilder, tokenizer: &[u8]) {
    for (key, value) in [
        (KEY_SOURCE_REVISION, SOURCE_REVISION),
        (KEY_SOURCE_NEMO_SHA256, SOURCE_NEMO_SHA256),
        (KEY_MODEL_CONFIG_SHA256, MODEL_CONFIG_SHA256),
        (KEY_DATA_PICKLE_SHA256, DATA_PICKLE_SHA256),
        (KEY_TENSOR_MANIFEST_SHA256, TENSOR_MANIFEST_SHA256),
        (KEY_DEC_HIDDEN_ACT, "relu"),
        (KEY_FRONTEND_WINDOW, "hann"),
        (KEY_FRONTEND_NORMALIZE, "per_feature"),
        (KEY_FRONTEND_PAD_MODE, "constant"),
    ] {
        builder.add_string(key, value);
    }
    for (key, value) in [
        (KEY_SAMPLE_RATE, 16_000),
        (KEY_ENC_N_LAYER, 32),
        (KEY_ENC_D_MODEL, 1_024),
        (KEY_ENC_N_HEAD, 8),
        (KEY_ENC_N_HEAD_KV, 8),
        (KEY_ENC_FFN_DIM, 4_096),
        (KEY_ENC_CONV_KERNEL, 9),
        (KEY_ENC_IN_DIM, 128),
        (KEY_ENC_SUB_FACTOR, 8),
        (KEY_ENC_SUB_KERNEL, 3),
        (KEY_ENC_SUB_STRIDE, 2),
        (KEY_ENC_SUB_CHANNELS, 256),
        (KEY_ENC_MAX_POS, 5_000),
        (KEY_ENC_ATTN_BIAS, 1),
        (KEY_ENC_CONV_BIAS, 1),
        (KEY_ENC_SCALE_INPUT, 0),
        (KEY_DEC_N_LAYER, 8),
        (KEY_DEC_D_MODEL, 1_024),
        (KEY_DEC_N_HEAD, 8),
        (KEY_DEC_FFN_DIM, 4_096),
        (KEY_DEC_MAX_SEQ, 1_024),
        (KEY_DEC_PRE_LN, 1),
        (KEY_HEAD_VOCAB_SIZE, 16_384),
        (KEY_HEAD_PAD_ID, 2),
        (KEY_HEAD_BOS_ID, 4),
        (KEY_HEAD_EOS_ID, 3),
        (KEY_FRONTEND_N_FFT, 512),
        (KEY_FRONTEND_HOP, 160),
        (KEY_FRONTEND_WIN, 400),
        (KEY_FRONTEND_N_MELS, 128),
        (KEY_FRONTEND_WINDOW_PERIODIC, 0),
    ] {
        builder.add_u32(key, value);
    }
    builder.add_f32(KEY_FRONTEND_PREEMPHASIS, 0.97);
    builder.add_metadata(
        KEY_TOKENIZER_VOCAB,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U8,
            values: tokenizer
                .iter()
                .map(|&byte| GgufMetadataValue::U8(byte))
                .collect(),
        }),
    );
    builder.add_string(KEY_TOKENIZER_VOCAB_SHA256, TOKENIZER_VOCAB_SHA256);
}

fn validate_checkpoint(safetensors: &SafetensorsFile) -> Result<(), ConvertError> {
    let expected = expected_canary_aed_manifest(8, 16_384);
    let internal_hash = hex(&manifest_sha256(&expected));
    if expected.len() != TENSOR_COUNT || internal_hash != TENSOR_MANIFEST_SHA256 {
        return Err(ConvertError::Parse(format!(
            "Canary-1B-v2 internal manifest drift: count={}, sha256={internal_hash}",
            expected.len()
        )));
    }

    // Compare the descriptor count before collapsing names into a set.  The
    // safetensors header is ordered JSON and the lightweight reader preserves
    // duplicate keys; an exact count plus exact name set therefore makes the
    // main-checkpoint contract reject duplicate/partial timestamp artifacts
    // before any tensor payload is copied to GGUF.
    if safetensors.tensors().len() != TENSOR_COUNT {
        return Err(ConvertError::Parse(format!(
            "Canary-1B-v2 prepared main-checkpoint tensor count {}, expected {TENSOR_COUNT}; the public 688-tensor timestamp auxiliary or duplicate/partial artifact is not executable",
            safetensors.tensors().len()
        )));
    }

    let actual_names = safetensors
        .tensors()
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<BTreeSet<_>>();
    let expected_names = expected.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if actual_names != expected_names {
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
            "Canary-1B-v2 prepared main-checkpoint manifest mismatch: found {}, expected {TENSOR_COUNT}; missing={missing:?}, extra={extra:?}. The public 688-tensor timestamp auxiliary checkpoint is not executable as Canary-1B-v2",
            safetensors.tensors().len()
        )));
    }
    for tensor in safetensors.tensors() {
        let expected_shape = &expected[&tensor.name];
        if &tensor.shape != expected_shape {
            return Err(ConvertError::Parse(format!(
                "Canary-1B-v2 tensor {:?} shape {:?}, expected {:?}",
                tensor.name, tensor.shape, expected_shape
            )));
        }
        if tensor.dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "Canary-1B-v2 tensor {:?} dtype {:?}, expected the pinned F32 main checkpoint",
                tensor.name, tensor.dtype
            )));
        }
    }
    Ok(())
}

fn validate_tokenizer_vocab(bytes: &[u8]) -> Result<(), ConvertError> {
    let actual_hash = hex(&sha256(bytes));
    if actual_hash != TOKENIZER_VOCAB_SHA256 {
        return Err(ConvertError::Parse(format!(
            "Canary-1B-v2 tokenizer SHA-256 {actual_hash}, expected {TOKENIZER_VOCAB_SHA256}"
        )));
    }
    let document = std::str::from_utf8(bytes).map_err(|error| {
        ConvertError::Parse(format!("Canary-1B-v2 tokenizer is not UTF-8: {error}"))
    })?;
    let mut pieces = Vec::with_capacity(VOCAB_SIZE);
    for (index, line) in document.lines().enumerate() {
        let (piece, score) = line.rsplit_once('\t').ok_or_else(|| {
            ConvertError::Parse(format!(
                "Canary-1B-v2 tokenizer line {} is not `piece<TAB>score`",
                index + 1
            ))
        })?;
        let score = score.parse::<f32>().map_err(|error| {
            ConvertError::Parse(format!(
                "Canary-1B-v2 tokenizer line {} score: {error}",
                index + 1
            ))
        })?;
        if piece.is_empty() || !score.is_finite() {
            return Err(ConvertError::Parse(format!(
                "Canary-1B-v2 tokenizer line {} is malformed",
                index + 1
            )));
        }
        pieces.push(piece);
    }
    if pieces.len() != VOCAB_SIZE {
        return Err(ConvertError::Parse(format!(
            "Canary-1B-v2 tokenizer has {} pieces, expected {VOCAB_SIZE}",
            pieces.len()
        )));
    }
    for (id, expected) in [
        (0, "<unk>"),
        (2, "<pad>"),
        (3, "<|endoftext|>"),
        (4, "<|startoftranscript|>"),
        (7, "<|startofcontext|>"),
        (16, "<|emo:undefined|>"),
        (64, "<|en|>"),
        (192, "<|uk|>"),
        (SPECIAL_VOCAB_SIZE, "en"),
    ] {
        if pieces[id] != expected {
            return Err(ConvertError::Parse(format!(
                "Canary-1B-v2 tokenizer id {id} must be {expected:?}, found {:?}",
                pieces[id]
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_manifest_count_and_hash_are_pinned() {
        let manifest = expected_canary_aed_manifest(8, 16_384);
        assert_eq!(manifest.len(), TENSOR_COUNT);
        assert_eq!(hex(&manifest_sha256(&manifest)), TENSOR_MANIFEST_SHA256);
        assert_eq!(
            manifest["transf_decoder._decoder.layers.7.third_sub_layer.dense_out.weight"],
            vec![1_024, 4_096]
        );
        assert_eq!(
            manifest["log_softmax.mlp.layer0.weight"],
            vec![16_384, 1_024]
        );
    }

    #[test]
    fn wrong_tokenizer_hash_is_rejected() {
        let error = validate_tokenizer_vocab(b"<unk>\t0\n").expect_err("wrong hash");
        assert!(error.to_string().contains("SHA-256"));
    }

    #[test]
    fn timestamp_or_duplicate_partial_checkpoint_is_rejected_by_exact_count() {
        let header = br#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0; 8]);
        let checkpoint = SafetensorsFile::parse(bytes).expect("synthetic duplicate header");
        let error = validate_checkpoint(&checkpoint).expect_err("partial checkpoint");
        let message = error.to_string();
        assert!(message.contains("tensor count 2"));
        assert!(message.contains("timestamp auxiliary"));
    }

    #[test]
    fn output_writer_never_clobbers_claimed_final_path() {
        let root = std::env::temp_dir().join(format!("vokra-canary-output-{}", std::process::id()));
        std::fs::create_dir_all(&root).expect("create output test directory");
        let output = root.join("model.gguf");
        write_output_no_clobber(&output, b"first").expect("publish first output");
        assert!(write_output_no_clobber(&output, b"second").is_err());
        assert_eq!(
            std::fs::read(&output).expect("read preserved output"),
            b"first"
        );
        std::fs::remove_dir_all(root).expect("remove output test directory");
    }

    #[test]
    fn io_paths_reject_dot_components_and_symlink_ancestors() {
        let root = std::env::temp_dir().join(format!("vokra-canary-paths-{}", std::process::id()));
        std::fs::create_dir_all(&root).expect("create path-test directory");
        let input = root.join("checkpoint.safetensors");
        let tokenizer = root.join("tokenizer.vocab");
        std::fs::write(&input, []).expect("write checkpoint placeholder");
        std::fs::write(&tokenizer, []).expect("write tokenizer placeholder");
        let output = root.join("output.gguf");
        validate_io_paths(&input, &tokenizer, &output).expect("regular paths accepted");
        let dotted = root.join(".").join("output.gguf");
        assert!(reject_unsafe_path(&dotted, "output").is_err());
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let real_parent = root.join("real-parent");
            let link_parent = root.join("link-parent");
            std::fs::create_dir(&real_parent).expect("create real parent");
            symlink(&real_parent, &link_parent).expect("create symlink parent");
            let linked_output = link_parent.join("output.gguf");
            assert!(reject_unsafe_path(&linked_output, "output").is_err());
        }
        std::fs::remove_dir_all(root).expect("remove path-test directory");
    }

    #[test]
    fn dedicated_path_authenticates_tokenizer_before_checkpoint_io() {
        let root = std::env::temp_dir().join(format!(
            "vokra-canary-tokenizer-first-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).expect("create test directory");
        let checkpoint = root.join("does-not-exist.safetensors");
        let tokenizer = root.join("tokenizer.vocab");
        let output = root.join("output.gguf");
        std::fs::write(&tokenizer, b"unauthenticated").expect("write tokenizer fixture");
        let error = convert_canary_file_with_tokenizer(&checkpoint, &output, None, &tokenizer)
            .expect_err("tokenizer authentication must precede checkpoint I/O");
        assert!(error.to_string().contains("SHA-256"));
        std::fs::remove_dir_all(root).expect("remove test directory");
    }
}
