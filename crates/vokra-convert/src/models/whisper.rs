//! Whisper base: safetensors checkpoint to GGUF conversion.
//!
//! Input: the upstream `openai/whisper-base` safetensors checkpoint (weights
//! only — no model code is imported, per IF-06 / FR-MD-02). Output: a GGUF with
//! every tensor plus the `vokra.model.*` and `vokra.frontend.*` chunks.
//!
//! # Tensor naming contract (M0 proposal, shared with M0-06)
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! ([`gguf_tensor_name`] is the identity function in M0). This makes coverage
//! total by construction — the converter writes exactly the tensors the file
//! contains, so there can be no "unknown" or "missing" tensor — and gives
//! M0-06 (the native Whisper implementation) an unambiguous contract: look up
//! weights by their Hugging Face names. A richer Vokra-side renaming can be
//! introduced later without changing this module's guarantees.
//!
//! # Dimension order
//!
//! Dimensions are stored in **source order** (safetensors/PyTorch row-major,
//! outermost dimension first), not reversed to the ggml `ne[]` convention. The
//! consumer (M0-06) reads them in the same order; consistency within Vokra is
//! the contract.

use vokra_core::gguf::{
    FrontendSpec, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::{SafeTensor, SafeTensors};

/// `vokra.model.arch` value written for Whisper GGUFs.
pub(crate) const ARCH: &str = "whisper";
/// `vokra.model.name` value written for the Whisper base GGUF.
pub(crate) const NAME: &str = "whisper-base";

// ---------------------------------------------------------------------------
// `vokra.whisper.*` hyperparameter chunk (M0-06-T04)
// ---------------------------------------------------------------------------
//
// The native Whisper implementation (M0-06, `vokra-models`) must read every
// hyperparameter from GGUF metadata rather than hard-coding it (FR-LD-02 /
// FR-MD-02). The M0-03 converter previously wrote only `vokra.model.*` and
// `vokra.frontend.*`; this WP adds the architectural hyperparameters, derived
// from the checkpoint's tensor shapes (never invented). Keys mirror the
// familiar whisper.cpp names under the `vokra.` prefix (IF-07 / no collision
// with llama.cpp's `general.*` / `tokenizer.*`).
//
// These key strings are duplicated verbatim in
// `vokra-models/src/whisper/config.rs` because the two crates cannot depend on
// each other (converter -> vokra-core only; model -> vokra-core / vokra-ops).
// Centralising them in `vokra-core::gguf::chunks` is a follow-up once that
// module is not owned by a parallel WP.

/// `vokra.whisper.n_mels` — number of mel input channels (`UINT32`).
const KEY_N_MELS: &str = "vokra.whisper.n_mels";
/// `vokra.whisper.n_audio_ctx` — encoder positional length, 1500 (`UINT32`).
const KEY_N_AUDIO_CTX: &str = "vokra.whisper.n_audio_ctx";
/// `vokra.whisper.n_audio_state` — encoder/decoder hidden width `d_model` (`UINT32`).
const KEY_N_AUDIO_STATE: &str = "vokra.whisper.n_audio_state";
/// `vokra.whisper.n_audio_head` — encoder attention heads (`UINT32`).
const KEY_N_AUDIO_HEAD: &str = "vokra.whisper.n_audio_head";
/// `vokra.whisper.n_audio_layer` — encoder block count (`UINT32`).
const KEY_N_AUDIO_LAYER: &str = "vokra.whisper.n_audio_layer";
/// `vokra.whisper.n_text_ctx` — decoder positional length, 448 (`UINT32`).
const KEY_N_TEXT_CTX: &str = "vokra.whisper.n_text_ctx";
/// `vokra.whisper.n_text_state` — decoder hidden width (`UINT32`).
const KEY_N_TEXT_STATE: &str = "vokra.whisper.n_text_state";
/// `vokra.whisper.n_text_head` — decoder attention heads (`UINT32`).
const KEY_N_TEXT_HEAD: &str = "vokra.whisper.n_text_head";
/// `vokra.whisper.n_text_layer` — decoder block count (`UINT32`).
const KEY_N_TEXT_LAYER: &str = "vokra.whisper.n_text_layer";
/// `vokra.whisper.n_vocab` — token vocabulary size (`UINT32`).
const KEY_N_VOCAB: &str = "vokra.whisper.n_vocab";
/// `vokra.whisper.ffn_dim` — feed-forward inner width (`UINT32`).
const KEY_FFN_DIM: &str = "vokra.whisper.ffn_dim";
/// `vokra.whisper.eot` — end-of-transcript token id (`UINT32`).
const KEY_EOT: &str = "vokra.whisper.eot";
/// `vokra.whisper.decoder_start_ids` — default decode prefix (`UINT32` array).
const KEY_DECODER_START_IDS: &str = "vokra.whisper.decoder_start_ids";

/// Fixed Whisper attention head dimension across every model size (base /
/// small / medium / large all use `head_dim = 64`); the head count is
/// therefore `d_model / 64`. Source: openai/whisper `whisper/model.py`
/// (`MultiHeadAttention`, `n_state // n_head` with the sizes tabulated so
/// `head_dim == 64`).
const WHISPER_HEAD_DIM: u64 = 64;

/// End-of-transcript token id for the Whisper *multilingual* tokenizer
/// (`<|endoftext|>`), fixed for every multilingual model including base.
/// Source: openai/whisper `whisper/tokenizer.py`.
const WHISPER_EOT: u32 = 50257;

/// Default decode prefix for **English transcription** with the multilingual
/// tokenizer: `<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>`.
/// Source: openai/whisper `whisper/tokenizer.py` special-token layout, verified
/// against `transformers` `WhisperProcessor.get_decoder_prompt_ids`. Non-English
/// / translation prefixes are a later (M1) concern; the runtime reads this
/// array from metadata rather than hard-coding it.
const WHISPER_DECODER_START_IDS: [u32; 4] = [50258, 50259, 50359, 50363];

/// Maps an upstream safetensors tensor name to its GGUF name (identity in M0).
pub(crate) fn gguf_tensor_name(hf_name: &str) -> String {
    hf_name.to_owned()
}

/// The Whisper front-end feature-extraction parameters.
///
/// Every value is transcribed from the upstream Whisper implementation, not
/// invented (frontend bit-exactness, reviewer C note #2). Sources:
///
/// - `openai/whisper` `whisper/audio.py`: `SAMPLE_RATE = 16000`,
///   `N_FFT = 400`, `HOP_LENGTH = 160`, `N_MELS = 80` (base),
///   `window = torch.hann_window(N_FFT)`, `torch.stft(..., center=True)`.
/// - `win_length` defaults to `n_fft` in `torch.stft`; `pad_mode` defaults to
///   `"reflect"` in `torch.stft`.
/// - The mel filterbank is `librosa.filters.mel(sr=16000, n_fft=400,
///   n_mels=80)`; librosa defaults give Slaney normalization, non-HTK,
///   `fmin = 0.0`, `fmax = sr/2 = 8000.0`.
/// - Whisper applies no DC-offset removal and no pre-emphasis.
pub(crate) fn frontend_spec() -> FrontendSpec {
    FrontendSpec {
        n_fft: 400,
        hop: 160,
        win_length: 400,
        window_type: "hann".to_owned(),
        mel_norm: "slaney".to_owned(),
        htk_mode: false,
        fmin: 0.0,
        fmax: 8000.0,
        n_mels: 80,
        pad_mode: "reflect".to_owned(),
        dc_offset_removal: false,
        pre_emphasis: 0.0,
        sample_rate: 16_000,
    }
}

/// Converts a Whisper base safetensors buffer into a populated GGUF builder.
pub(crate) fn convert(bytes: Vec<u8>) -> Result<GgufBuilder, ConvertError> {
    let st = SafeTensors::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    frontend_spec().write_into(&mut b);
    write_hparams(&mut b, &st);

    for t in st.tensors() {
        b.add_tensor(
            &gguf_tensor_name(&t.name),
            t.dtype,
            t.shape.clone(),
            st.tensor_bytes(t).to_vec(),
        )?;
    }

    Ok(b)
}

/// Derives the `vokra.whisper.*` hyperparameters from the checkpoint's tensor
/// shapes and writes them into `b`.
///
/// Every value is read from a tensor shape (or a documented Whisper invariant),
/// never invented. Derivation is best-effort: a checkpoint missing an expected
/// tensor writes `0` for that key, which the runtime's `WhisperConfig` loader
/// rejects at load time — the converter stays infallible so degenerate inputs
/// still round-trip.
fn write_hparams(b: &mut GgufBuilder, st: &SafeTensors) {
    let dim = |name: &str, axis: usize| -> u64 {
        st.tensors()
            .iter()
            .find(|t: &&SafeTensor| t.name == name)
            .and_then(|t| t.shape.get(axis).copied())
            .unwrap_or(0)
    };

    // d_model / n_mels from the first conv weight [d_model, n_mels, 3].
    let d_model = dim("model.encoder.conv1.weight", 0);
    let n_mels = dim("model.encoder.conv1.weight", 1);
    let n_audio_ctx = dim("model.encoder.embed_positions.weight", 0);
    let n_text_ctx = dim("model.decoder.embed_positions.weight", 0);
    let n_vocab = dim("model.decoder.embed_tokens.weight", 0);
    let ffn_dim = dim("model.encoder.layers.0.fc1.weight", 0);
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
    b.add_metadata(
        KEY_DECODER_START_IDS,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: WHISPER_DECODER_START_IDS
                .iter()
                .map(|&id| GgufMetadataValue::U32(id))
                .collect(),
        }),
    );
}

/// Counts contiguous transformer blocks named `<prefix><i>.` for `i = 0, 1, …`.
fn count_layers(st: &SafeTensors, prefix: &str) -> u32 {
    let mut n = 0u32;
    loop {
        let probe = format!("{prefix}{n}.");
        if st.tensors().iter().any(|t| t.name.starts_with(&probe)) {
            n += 1;
        } else {
            return n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufFile};

    /// Builds a tiny synthetic safetensors buffer with Whisper-like names.
    fn synthetic_whisper() -> Vec<u8> {
        // Two F32 tensors: names mimic HF Whisper naming.
        let a: Vec<u8> = [0.1f32, 0.2, 0.3, 0.4]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let bdat: Vec<u8> = [1.0f32, -1.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let header = r#"{"model.encoder.conv1.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"model.decoder.embed_tokens.weight":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&a);
        out.extend_from_slice(&bdat);
        out
    }

    #[test]
    fn converts_and_roundtrips_through_gguf() {
        let gguf_bytes = convert(synthetic_whisper()).unwrap().to_bytes().unwrap();
        let file = GgufFile::parse(gguf_bytes).unwrap();

        // Model + frontend metadata present (2 model keys + 13 frontend keys).
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some("whisper")
        );
        let spec = FrontendSpec::from_gguf(&file).unwrap();
        assert_eq!(spec, frontend_spec());

        // Both tensors present verbatim, bytes intact.
        assert_eq!(file.tensors().len(), 2);
        let w = file.tensor_info("model.encoder.conv1.weight").unwrap();
        assert_eq!(w.dtype, GgmlType::F32);
        assert_eq!(w.dimensions, vec![2, 2]);
        assert_eq!(
            file.tensor_data("model.decoder.embed_tokens.weight")
                .unwrap(),
            [1.0f32, -1.0]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<_>>()
                .as_slice()
        );
    }

    #[test]
    fn coverage_is_total_by_construction() {
        // Every input tensor name appears in the output.
        let st = SafeTensors::parse(synthetic_whisper()).unwrap();
        let input_names: Vec<String> = st.tensors().iter().map(|t| t.name.clone()).collect();
        let file =
            GgufFile::parse(convert(synthetic_whisper()).unwrap().to_bytes().unwrap()).unwrap();
        for name in input_names {
            assert!(
                file.tensor_info(&gguf_tensor_name(&name)).is_some(),
                "missing {name}"
            );
        }
    }
}
