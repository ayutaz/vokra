//! NVIDIA Parakeet-TDT-1.1B safetensors → executable GGUF conversion.
//!
//! The immutable upstream `.nemo` archive contains 1,667 F32 tensors, a
//! 42-layer FastConformer config and a 1,024-piece SentencePiece Unigram
//! tokenizer. The offline preparation step flattens `model_weights.ckpt`
//! without renaming tensors. This converter preserves those payload bytes,
//! stamps the complete verified hparam contract and optionally embeds the
//! byte-exact plaintext `tokenizer.vocab` needed for native decoding.
//!
//! The 4.28 GB artifact is VAST-only under the repository's >=2 GB safety
//! policy. The runtime remains Rust-only and never loads Python, pickle,
//! SentencePiece protobuf, ONNX or ONNX Runtime.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Parakeet-TDT-1.1B GGUFs. Distinct from every
/// sibling arch tag — silently aliasing with `"parakeet-tdt"` (which the
/// `parakeet-tdt-0.6b-v3` module owns) would misroute the runtime
/// dispatch because that arm hard-codes the 0.6B-v3 axes (24-layer / 128
/// mel bins / `attention_bias=false`), and aliasing with
/// `"parakeet-ctc"` (which the 1.1B CTC sibling owns) would misroute
/// into a CTC decoder walk. The `_1_1b` suffix pins the SKU on the arch
/// tag itself so a downstream reader can dispatch without a second
/// hparam lookup.
pub const ARCH: &str = "parakeet-tdt-1_1b";

/// `vokra.model.name` value written for the canonical Parakeet-TDT-1.1B
/// GGUF. Matches the `huggingface.co/vokra/parakeet-tdt-1.1b` publish
/// slug and the `as_arg` return value in `lib.rs` so the CLI /
/// model-card / publish pipe all agree on a single identifier.
pub const NAME: &str = "parakeet-tdt-1.1b";

/// `vokra.model.category` value — the third `asr` model in the Parakeet
/// family (after `parakeet-tdt-0.6b-v3` and `parakeet-ctc-1.1b`).
/// Consumed by the model-card generator + zoo manifest tier gate so a
/// downstream picks the correct decode path.
pub const CATEGORY: &str = "asr";

/// Ad-hoc metadata key for the model category. Kept as a converter-side
/// constant (not a `chunks::KEY_*` alias) until a sibling `category`
/// consumer lands in `vokra-core` — mirror of the wespeaker /
/// speaker_3d / emotion2vec / frcrn / rnnoise local constant.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Upstream repository slug (`org/name`) recorded under
/// `vokra.provenance.upstream_hf` so a downstream consumer can trace the
/// artifact back to its serving location.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
/// Value written to [`KEY_PROVENANCE_UPSTREAM_HF`] — the canonical
/// NVIDIA HuggingFace slug.
pub const UPSTREAM_HF: &str = "nvidia/parakeet-tdt-1.1b";

/// Canonical weight license SPDX (`cc-by-4.0`). Overrides via the
/// [`convert_parakeet_tdt_1_1b_file`] `license` parameter — the
/// standing mechanism for "implementation is clean-room but the
/// upstream distributed checkpoint is another SPDX" scenarios (mirror
/// of `convert_file_licensed` in `lib.rs` and the `license` arg on
/// `convert_frcrn_file` / `convert_rnnoise_file`).
pub const DEFAULT_LICENSE: &str = "cc-by-4.0";

const KEY_SOURCE_REVISION: &str = "vokra.parakeet_tdt_1_1b.source_revision";
const KEY_SOURCE_NEMO_SHA256: &str = "vokra.parakeet_tdt_1_1b.source_nemo_sha256";
const KEY_SAMPLE_RATE: &str = "vokra.parakeet_tdt_1_1b.sample_rate";
const KEY_N_FFT: &str = "vokra.parakeet_tdt_1_1b.frontend.n_fft";
const KEY_HOP_LENGTH: &str = "vokra.parakeet_tdt_1_1b.frontend.hop_length";
const KEY_WIN_LENGTH: &str = "vokra.parakeet_tdt_1_1b.frontend.win_length";
const KEY_N_MELS: &str = "vokra.parakeet_tdt_1_1b.frontend.n_mels";
const KEY_PREEMPHASIS: &str = "vokra.parakeet_tdt_1_1b.frontend.preemphasis";
const KEY_ENC_N_LAYER: &str = "vokra.parakeet_tdt_1_1b.encoder.n_layer";
const KEY_ENC_D_MODEL: &str = "vokra.parakeet_tdt_1_1b.encoder.d_model";
const KEY_ENC_N_HEAD: &str = "vokra.parakeet_tdt_1_1b.encoder.n_head";
const KEY_ENC_N_HEAD_KV: &str = "vokra.parakeet_tdt_1_1b.encoder.n_head_kv";
const KEY_ENC_FFN_DIM: &str = "vokra.parakeet_tdt_1_1b.encoder.ffn_dim";
const KEY_ENC_CONV_KERNEL: &str = "vokra.parakeet_tdt_1_1b.encoder.conv_kernel_size";
const KEY_ENC_SUB_FACTOR: &str = "vokra.parakeet_tdt_1_1b.encoder.subsampling_factor";
const KEY_ENC_SUB_KERNEL: &str = "vokra.parakeet_tdt_1_1b.encoder.subsampling_kernel";
const KEY_ENC_SUB_STRIDE: &str = "vokra.parakeet_tdt_1_1b.encoder.subsampling_stride";
const KEY_ENC_SUB_CHANNELS: &str = "vokra.parakeet_tdt_1_1b.encoder.subsampling_channels";
const KEY_ENC_MAX_POS: &str = "vokra.parakeet_tdt_1_1b.encoder.max_position_embeddings";
const KEY_ENC_USE_BIAS: &str = "vokra.parakeet_tdt_1_1b.encoder.use_bias";
const KEY_ENC_SCALE_INPUT: &str = "vokra.parakeet_tdt_1_1b.encoder.scale_input";
const KEY_DEC_N_LAYER: &str = "vokra.parakeet_tdt_1_1b.decoder.n_layer";
const KEY_DEC_D_MODEL: &str = "vokra.parakeet_tdt_1_1b.decoder.d_model";
const KEY_JOINT_VOCAB_SIZE: &str = "vokra.parakeet_tdt_1_1b.joint.vocab_size";
const KEY_JOINT_BLANK_ID: &str = "vokra.parakeet_tdt_1_1b.joint.blank_token_id";
const KEY_JOINT_PAD_ID: &str = "vokra.parakeet_tdt_1_1b.joint.pad_token_id";
const KEY_JOINT_N_DURATIONS: &str = "vokra.parakeet_tdt_1_1b.joint.n_durations";
const PREFIX_JOINT_DURATION: &str = "vokra.parakeet_tdt_1_1b.joint.duration.";
const KEY_JOINT_MAX_SYMBOLS: &str = "vokra.parakeet_tdt_1_1b.joint.max_symbols_per_step";
const KEY_JOINT_ACT: &str = "vokra.parakeet_tdt_1_1b.joint.activation";
const KEY_TOKENIZER_VOCAB: &str = "vokra.parakeet_tdt_1_1b.tokenizer.vocab";
const KEY_TOKENIZER_VOCAB_SHA256: &str = "vokra.parakeet_tdt_1_1b.tokenizer.vocab_sha256";

pub const SOURCE_REVISION: &str = "53276c6469d1f17a1352e30c4d11be3d0d7e9575";
pub const SOURCE_NEMO_SHA256: &str =
    "9c563d52bdffeacbac0c5b894fdea9be82fea3a6bd8bb8018ff57888e2b5d988";
pub const TOKENIZER_VOCAB_SHA256: &str =
    "dc8f48909c2d3a0374f45b7478226d26a7de16bbc5334448a8e989f4538384d1";

const DURATIONS: [u32; 5] = [0, 1, 2, 3, 4];

/// The FR-MD-09 attribution text stamped into
/// `vokra.provenance.attribution` — wording aligned with `NOTICE` and
/// the `docs/license-audit.md` NVIDIA Parakeet family rows. Final legal
/// sufficiency = owner-facing publish gate (mirror of the T29 sign-off
/// posture the 0.6B-v3 sibling uses).
pub const PARAKEET_TDT_1_1B_ATTRIBUTION_TEXT: &str = "This application uses NVIDIA Parakeet-TDT-1.1B \
     (English ASR — FastConformer encoder + TDT decoder, 1.1B scale-up \
     of the Parakeet-TDT-0.6B-v3 topology). Model weights are licensed \
     under CC-BY 4.0 (attribution required; commercial use permitted). \
     Copyright (c) NVIDIA. Source: \
     https://huggingface.co/nvidia/parakeet-tdt-1.1b";

/// Outcome of a Parakeet-TDT-1.1B conversion.
///
/// All counters are additive and default to zero — a zero-tensor
/// checkpoint returns `ParakeetTdt11bReport::default()` and the caller
/// remains responsible for surfacing the "no float tensors" loud note
/// (mirror of the frcrn / rnnoise / qwen3_tts / vibevoice `Report`
/// pattern). `read == written + skipped_non_float` is an invariant
/// preserved by [`convert_parakeet_tdt_1_1b_file`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ParakeetTdt11bReport {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 all go through
    /// the same byte-copy path since the BF16 pass-through landed
    /// 2026-07-25).
    pub written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time; anything
    /// that reaches this arm signals a reader change upstream).
    pub skipped_non_float: usize,
    /// Of the tensors in `written`, how many were BF16 (subset counter).
    /// Emits GGUF type 30 verbatim; runtime widens BF16 → f32 losslessly
    /// via the single choke point `crates/vokra-core/src/gguf/quant/mod.rs
    /// decode_bf16` (BF16 = top 16 bits of an f32 — `bits << 16` is exact).
    pub bf16_passthrough: usize,
}

fn write_runtime_metadata(builder: &mut GgufBuilder, tokenizer_vocab: Option<&[u8]>) {
    builder.add_string(KEY_SOURCE_REVISION, SOURCE_REVISION);
    builder.add_string(KEY_SOURCE_NEMO_SHA256, SOURCE_NEMO_SHA256);
    for (key, value) in [
        (KEY_SAMPLE_RATE, 16_000),
        (KEY_N_FFT, 512),
        (KEY_HOP_LENGTH, 160),
        (KEY_WIN_LENGTH, 400),
        (KEY_N_MELS, 80),
        (KEY_ENC_N_LAYER, 42),
        (KEY_ENC_D_MODEL, 1024),
        (KEY_ENC_N_HEAD, 8),
        (KEY_ENC_N_HEAD_KV, 8),
        (KEY_ENC_FFN_DIM, 4096),
        (KEY_ENC_CONV_KERNEL, 9),
        (KEY_ENC_SUB_FACTOR, 8),
        (KEY_ENC_SUB_KERNEL, 3),
        (KEY_ENC_SUB_STRIDE, 2),
        (KEY_ENC_SUB_CHANNELS, 256),
        (KEY_ENC_MAX_POS, 5000),
        (KEY_ENC_USE_BIAS, 1),
        (KEY_ENC_SCALE_INPUT, 0),
        (KEY_DEC_N_LAYER, 2),
        (KEY_DEC_D_MODEL, 640),
        (KEY_JOINT_VOCAB_SIZE, 1025),
        (KEY_JOINT_BLANK_ID, 1024),
        (KEY_JOINT_PAD_ID, 1024),
        (KEY_JOINT_N_DURATIONS, DURATIONS.len() as u32),
        (KEY_JOINT_MAX_SYMBOLS, 10),
    ] {
        builder.add_u32(key, value);
    }
    builder.add_f32(KEY_PREEMPHASIS, 0.97);
    builder.add_string(KEY_JOINT_ACT, "relu");
    for (index, duration) in DURATIONS.iter().enumerate() {
        builder.add_u32(&format!("{PREFIX_JOINT_DURATION}{index}"), *duration);
    }
    if let Some(bytes) = tokenizer_vocab {
        builder.add_metadata(
            KEY_TOKENIZER_VOCAB,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::U8,
                values: bytes
                    .iter()
                    .map(|&byte| GgufMetadataValue::U8(byte))
                    .collect(),
            }),
        );
        builder.add_string(KEY_TOKENIZER_VOCAB_SHA256, TOKENIZER_VOCAB_SHA256);
    }
}

/// Reads a safetensors checkpoint at `input` and writes a
/// Parakeet-TDT-1.1B GGUF to `output`.
///
/// Every F32 / F16 / BF16 tensor is emitted verbatim under its upstream
/// name; the `vokra.provenance.*` + `vokra.model.*` chunk groups pin the
/// upstream slug, weight license, and model category so the zoo
/// manifest + model-card generator can gate on the artifact alone (no
/// side-car lookup). `vokra.schema.*` is written unconditionally by the
/// GGUF writer.
///
/// `license` overrides `DEFAULT_LICENSE` (`"cc-by-4.0"`) — the same
/// mechanism `lib.rs::convert_file_licensed` uses when the implementation
/// is clean-room but the redistributed checkpoint carries a different
/// SPDX. The class is re-derived from the override string via
/// [`LicenseClass::from_license_str`] so an override to `mit` /
/// `apache-2.0` correctly re-tags to `Permissive` rather than staying on
/// the CC-BY `AttributionRequired` class the default carries.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_parakeet_tdt_1_1b_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<ParakeetTdt11bReport, ConvertError> {
    convert_parakeet_tdt_1_1b_file_with_tokenizer(input, output, license, None)
}

/// Converts the official checkpoint and optionally embeds its byte-exact
/// plaintext SentencePiece vocabulary.
pub fn convert_parakeet_tdt_1_1b_file_with_tokenizer(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    tokenizer_vocab: Option<&Path>,
) -> Result<ParakeetTdt11bReport, ConvertError> {
    // Whole-file conversion of this 4.28 GB F32 release is intentionally
    // VAST-only under AGENTS.md. Do not run this path on the maintainer Mac.
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;

    let tokenizer_vocab = tokenizer_vocab
        .map(std::fs::read)
        .transpose()
        .map_err(ConvertError::Io)?;
    if let Some(bytes) = tokenizer_vocab.as_deref() {
        validate_tokenizer_vocab(bytes)?;
    }

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    write_runtime_metadata(&mut b, tokenizer_vocab.as_deref());

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = cc-by-4.0 (upstream `nvidia/parakeet-tdt-1.1b`
    // model-card + LICENSE, primary-source verified per the wave-b
    // ticket). `license` overrides for callers who obtained the weight
    // under a different SPDX (see `convert_file_licensed` in `lib.rs`).
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (
            DEFAULT_LICENSE.to_owned(),
            LicenseClass::AttributionRequired,
        ),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some("https://huggingface.co/nvidia/parakeet-tdt-1.1b"),
    );
    vokra_core::stamp_attribution(&mut b, PARAKEET_TDT_1_1B_ATTRIBUTION_TEXT);

    let mut report = ParakeetTdt11bReport::default();
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (docs/adr/qwen3-tts-bf16.md, strategy A_passthrough); the runtime
    // widens BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. Mirrors
    // `parakeet_ctc::convert` / `frcrn::convert` / `rnnoise::convert`.
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
    std::fs::write(output, out_bytes).map_err(ConvertError::Io)?;
    Ok(report)
}

fn validate_tokenizer_vocab(bytes: &[u8]) -> Result<(), ConvertError> {
    let document = std::str::from_utf8(bytes).map_err(|error| {
        ConvertError::Parse(format!(
            "Parakeet-TDT-1.1B tokenizer.vocab is not UTF-8: {error}"
        ))
    })?;
    let lines = document.lines().collect::<Vec<_>>();
    if lines.len() != 1024
        || lines
            .first()
            .and_then(|line| line.split_once('\t'))
            .map(|v| v.0)
            != Some("<unk>")
    {
        return Err(ConvertError::Parse(format!(
            "Parakeet-TDT-1.1B tokenizer.vocab must contain the official 1024-piece SentencePiece export beginning with `<unk>`; found {} lines",
            lines.len()
        )));
    }
    for (index, line) in lines.iter().enumerate() {
        let Some((piece, score)) = line.rsplit_once('\t') else {
            return Err(ConvertError::Parse(format!(
                "Parakeet-TDT-1.1B tokenizer.vocab line {} is not `piece<TAB>score`",
                index + 1
            )));
        };
        if piece.is_empty() || score.parse::<f32>().is_err() {
            return Err(ConvertError::Parse(format!(
                "Parakeet-TDT-1.1B tokenizer.vocab line {} is malformed",
                index + 1
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Per-process, per-test scratch path in the system temp dir (frcrn
    /// / rnnoise / emotion2vec test pattern — no external `tempfile`
    /// dep, preserving zero-dep NFR-DS-02). The nanosecond suffix
    /// separates parallel `cargo test` runs so they cannot clobber each
    /// other's files.
    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-parakeet-tdt-1-1b-{}-{}-{}.bin",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        p
    }

    /// Builds a synthetic safetensors buffer with a single BF16 tensor.
    ///
    /// The payload is chosen from a known set of non-zero BF16 bit
    /// patterns so a byte-identity assert catches any silent widen /
    /// downcast attempt — a zeroed payload would round-trip trivially
    /// through F32 / F16 widen and defeat the pin (mirror of frcrn's
    /// fixture).
    fn synthetic_bf16_safetensors() -> (Vec<u8>, Vec<u8>) {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");
        // Tensor name modelled on the Parakeet FastConformer encoder
        // topology (`encoder.blocks.0.attn.qkv_proj.weight`, per the
        // parakeet.rs / parakeet_ctc.rs test fixtures) — the shape here
        // is a stand-in `[2, 3]` for the synthetic pass-through pin;
        // the real prep-script tensor names are the follow-up.
        let header = r#"{"encoder.blocks.0.attn.qkv_proj.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&bf16);
        (buf, bf16)
    }

    /// Builds a synthetic safetensors buffer with one F32 tensor
    /// (`shape=[2,3]`, 24 B) followed by one F16 tensor
    /// (`shape=[1,4]`, 8 B). The offsets are chosen so the tensors are
    /// contiguous in the data region — mirror of frcrn / rnnoise's
    /// fixtures.
    fn synthetic_f32_and_f16_safetensors() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let f32_vals: [f32; 6] = [1.0, -2.0, 3.5, -0.25, 100.0, 0.001];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 24, "6 elements × 4 bytes F32 payload");
        let f16_patterns: [u16; 4] = [0x3C00, 0xC000, 0x4200, 0x0001];
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|p| p.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 8, "4 elements × 2 bytes F16 payload");
        // Tensor names track the Parakeet FastConformer / RNN-T
        // prediction-net topology (`encoder.blocks.0.mlp.fc1.weight` /
        // `decoder.pred_net.embed.weight`); shapes are synthetic
        // stand-ins for the pass-through pin.
        let header = r#"{"encoder.blocks.0.mlp.fc1.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"decoder.pred_net.embed.weight":{"dtype":"F16","shape":[1,4],"data_offsets":[24,32]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&f32_bytes);
        buf.extend_from_slice(&f16_bytes);
        (buf, f32_bytes, f16_bytes)
    }

    /// BF16 pass-through: the upstream BF16 checkpoint must survive the
    /// file-based converter round-trip with its dtype preserved (GGUF
    /// type 30 = `GgmlType::BF16`) and its payload byte-identical to the
    /// input. Mirror of the frcrn / rnnoise / neucodec / emotion2vec
    /// equivalent.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (input_bytes, bf16_payload) = synthetic_bf16_safetensors();
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_parakeet_tdt_1_1b_file(&input, &output, None).expect("convert");

        // Counters: single BF16 tensor read + written + BF16 subset.
        assert_eq!(report.read, 1, "one tensor visible in safetensors header");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of parakeet_ctc)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 subset counter must record the pass-through"
        );
        assert_eq!(
            report.read,
            report.written + report.skipped_non_float,
            "read = written + skipped invariant (mirror of qwen3_tts pattern)"
        );

        // Round-trip: dtype preserved, payload byte-identical (no silent widen).
        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        let info = file
            .tensor_info("encoder.blocks.0.attn.qkv_proj.weight")
            .expect("BF16 tensor present after pass-through");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16_payload.as_slice(),
            "BF16 payload must be byte-identical to input"
        );

        // Provenance + category chunks pinned on the artifact itself.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH),
            "arch stamp distinct from parakeet-tdt (0.6B-v3) / parakeet-ctc (1.1B CTC) — silent alias would misroute"
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "category groups Parakeet-TDT-1.1B with the ASR family for the zoo manifest"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF),
            "upstream slug pins traceability back to nvidia/parakeet-tdt-1.1b"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE),
            "default license is cc-by-4.0 (AttributionRequired)"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::AttributionRequired.as_str())
        );
        // Attribution text is non-empty and NVIDIA-named — mirror of
        // the parakeet 0.6B-v3 assertion.
        let attr = file
            .get(chunks::KEY_PROVENANCE_ATTRIBUTION)
            .and_then(|v| v.as_str())
            .expect("attribution present");
        assert!(
            attr.contains("NVIDIA") && attr.contains("CC-BY 4.0"),
            "attribution names NVIDIA + CC-BY 4.0: {attr}"
        );
        assert!(
            file.get(chunks::KEY_SCHEMA_VERSION).is_some(),
            "vokra.schema.version must be stamped"
        );
        assert!(
            file.get(chunks::KEY_SCHEMA_PRODUCER).is_some(),
            "vokra.schema.producer must be stamped"
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// F32 + F16 pass-through: two float tensors of distinct dtypes in
    /// the same input must both reach the pass-through arm without
    /// collapsing into a single dtype branch, and the BF16 counter must
    /// remain 0 (default). Guards against a naive `if bf16 { ... } else`
    /// refactor.
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        let (input_bytes, f32_payload, f16_payload) = synthetic_f32_and_f16_safetensors();
        let input = scratch_path("f32f16-in");
        let output = scratch_path("f32f16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_parakeet_tdt_1_1b_file(&input, &output, None).expect("convert");

        assert_eq!(report.read, 2, "two tensors visible in header");
        assert_eq!(report.written, 2, "both F32 and F16 must pass through");
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32+F16-only input must leave the BF16 subset counter at the Default 0"
        );

        // Both tensors survive the round-trip with their upstream names
        // and dtypes preserved.
        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        let f32_info = file
            .tensor_info("encoder.blocks.0.mlp.fc1.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(f32_info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(f32_info), f32_payload.as_slice());

        let f16_info = file
            .tensor_info("decoder.pred_net.embed.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(f16_info.dimensions, vec![1, 4]);
        assert_eq!(file.tensor_bytes(f16_info), f16_payload.as_slice());

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// License override: a caller with an SPDX id distinct from the
    /// default (`cc-by-4.0`) must land on the artifact's license stamp;
    /// the license class is re-derived from the override string (mirror
    /// of the `convert_file_licensed` pattern in `lib.rs`). Uses
    /// `apache-2.0` (Permissive) so the class ALSO changes — an
    /// attribution-required-only override would flip the SPDX but keep
    /// the class, missing the class-derivation regression window.
    #[test]
    fn license_override_lands_on_the_artifact_and_reshapes_the_class() {
        let (input_bytes, _) = synthetic_bf16_safetensors();
        let input = scratch_path("lic-in");
        let output = scratch_path("lic-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let _ = convert_parakeet_tdt_1_1b_file(&input, &output, Some("apache-2.0"))
            .expect("convert with license override");

        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
            "override MUST land on the raw licence slot"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "override to apache-2.0 MUST re-derive the class to Permissive \
             (a stale AttributionRequired stamp would tag the artifact as \
             CC-BY-family in the publish gate)"
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// Empty header → zero tensors written, no schema drift. Guards
    /// against a regression where a metadata-only GGUF accidentally
    /// binds a placeholder tensor.
    #[test]
    fn zero_tensor_input_writes_metadata_only_gguf() {
        let empty_header = r#"{}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(empty_header.len() as u64).to_le_bytes());
        buf.extend_from_slice(empty_header.as_bytes());
        let input = scratch_path("empty-in");
        let output = scratch_path("empty-out");
        std::fs::write(&input, &buf).expect("write empty safetensors input");

        let report = convert_parakeet_tdt_1_1b_file(&input, &output, None).expect("convert empty");

        assert_eq!(report.read, 0);
        assert_eq!(report.written, 0);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0);

        // Metadata still lands (arch / name / provenance / attribution) —
        // the runtime's FR-EX-08 gate at load time is the authoritative
        // "no float tensors bound" refuser, not this converter.
        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(file.tensors().len(), 0, "no tensors bound in empty case");

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// The `ARCH` constant must not collide with either Parakeet sibling
    /// (`parakeet-tdt` for 0.6B-v3, `parakeet-ctc` for 1.1B CTC).
    /// Silently aliasing would misroute the runtime dispatch — this
    /// pins the arch tag as a compile-time invariant.
    #[test]
    fn arch_string_is_distinct_from_parakeet_siblings() {
        assert_eq!(ARCH, "parakeet-tdt-1_1b");
        assert_ne!(
            ARCH,
            crate::models::parakeet::ARCH,
            "must not alias with parakeet-tdt (0.6B-v3) arch tag"
        );
        assert_ne!(
            ARCH,
            crate::models::parakeet_ctc::ARCH,
            "must not alias with parakeet-ctc (1.1B CTC) arch tag"
        );
    }
}
