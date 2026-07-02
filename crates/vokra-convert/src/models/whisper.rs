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

use vokra_core::gguf::{FrontendSpec, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafeTensors;

/// `vokra.model.arch` value written for Whisper GGUFs.
pub(crate) const ARCH: &str = "whisper";
/// `vokra.model.name` value written for the Whisper base GGUF.
pub(crate) const NAME: &str = "whisper-base";

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
