//! CSM text tokenizer surface (M4-05-T05).
//!
//! The upstream text tokenizer is `meta-llama/Llama-3.2-1B`
//! (`generator.py` — ADR M4-05 §D2), a **gated** HF repo: the raw
//! tokenizer file arrives with the T29 owner hand-off and is embedded by
//! the converter as the `vokra.tokenizer.model` u8-array (the M2-06
//! Whisper / M3-10 Voxtral pattern, zero-dep).
//!
//! # Honest posture (FR-EX-08 / hallucination ban)
//!
//! - [`GgufCsmTokenizer::from_gguf`] reads the embedded blob back today;
//!   **`encode` is a `NotImplemented` stub** until T29 confirms the exact
//!   file format in hand (a half-invented BPE would silently mis-tokenize
//!   — worse than a loud error).
//! - [`FixtureByteTokenizer`] is the **explicit, opt-in** fixture path for
//!   synthesized-weight tests / host-only CLI smoke: it maps UTF-8 bytes
//!   into the tiny fixture vocab deterministically. It is *not* a language
//!   tokenizer and never a fallback — callers construct it by name, the
//!   same way they opt into synthesized weights.

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::{Result, VokraError};

/// The GGUF key carrying the raw tokenizer blob (shared with Whisper /
/// Voxtral — see the converter-side constant of the same value).
pub const KEY_TOKENIZER_MODEL: &str = "vokra.tokenizer.model";

/// Text → token ids for the CSM text slots. Implemented by the real
/// Llama-3.2 tokenizer (T29 flip-the-switch) and the fixture tokenizer.
pub trait CsmTextTokenizer: Send + Sync {
    /// Encodes `text` to token ids (each `< vocab_size`).
    ///
    /// # Errors
    ///
    /// Implementation-defined; the GGUF-backed tokenizer returns
    /// [`VokraError::NotImplemented`] until T29.
    fn encode(&self, text: &str) -> Result<Vec<u32>>;

    /// The vocab bound every returned id respects.
    fn vocab_size(&self) -> usize;
}

/// The GGUF-embedded Llama-3.2 tokenizer blob (raw bytes read back;
/// parsing = T29 flip-the-switch).
pub struct GgufCsmTokenizer {
    bytes: Vec<u8>,
    vocab_size: usize,
}

impl std::fmt::Debug for GgufCsmTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufCsmTokenizer")
            .field("bytes.len", &self.bytes.len())
            .field("vocab_size", &self.vocab_size)
            .finish()
    }
}

impl GgufCsmTokenizer {
    /// Reads the embedded tokenizer blob from a CSM GGUF.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] when the GGUF carries no
    /// `vokra.tokenizer.model` (the converter warns loudly in that case —
    /// gated repo, T29) or the array is not u8.
    pub fn from_gguf(file: &GgufFile, vocab_size: usize) -> Result<Self> {
        let arr = match file.get(KEY_TOKENIZER_MODEL) {
            Some(GgufMetadataValue::Array(a)) => a,
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "csm tokenizer: `{KEY_TOKENIZER_MODEL}` is not an array (got {:?})",
                    other.value_type()
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(format!(
                    "csm tokenizer: `{KEY_TOKENIZER_MODEL}` missing — the \
                     meta-llama/Llama-3.2-1B tokenizer is a gated download (T29); \
                     re-convert with `vokra-cli convert --model csm --config \
                     <tokenizer file>` once the owner supplies it"
                )));
            }
        };
        let mut bytes = Vec::with_capacity(arr.values.len());
        for v in &arr.values {
            match v {
                GgufMetadataValue::U8(x) => bytes.push(*x),
                other => {
                    return Err(VokraError::ModelLoad(format!(
                        "csm tokenizer: `{KEY_TOKENIZER_MODEL}` carries a non-U8 \
                         element ({:?})",
                        other.value_type()
                    )));
                }
            }
        }
        if bytes.is_empty() {
            return Err(VokraError::ModelLoad(
                "csm tokenizer: embedded blob is empty".into(),
            ));
        }
        Ok(Self { bytes, vocab_size })
    }

    /// The raw embedded bytes (parity / T29 inspection).
    #[must_use]
    pub fn raw_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl CsmTextTokenizer for GgufCsmTokenizer {
    fn encode(&self, _text: &str) -> Result<Vec<u32>> {
        Err(VokraError::NotImplemented(
            "csm tokenizer: the Llama-3.2 tokenizer parser lands with the T29 \
             owner hand-off (the gated file's exact format is confirmed then — \
             a half-invented BPE would silently mis-tokenize, FR-EX-08). Use \
             FixtureByteTokenizer for synthesized-weight fixtures.",
        ))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// Explicit **fixture** tokenizer: UTF-8 bytes → `byte % vocab_size`.
/// Deterministic, linguistically meaningless — pairs with the synthesized
/// weight fixtures to exercise the numeric pipeline. Never a fallback:
/// callers opt in by constructing it (FR-EX-08 posture).
#[derive(Debug, Clone)]
pub struct FixtureByteTokenizer {
    vocab_size: usize,
}

impl FixtureByteTokenizer {
    /// Builds the fixture tokenizer for a `vocab_size`-bounded id space.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on `vocab_size == 0`.
    pub fn new(vocab_size: usize) -> Result<Self> {
        if vocab_size == 0 {
            return Err(VokraError::InvalidArgument(
                "csm fixture tokenizer: vocab_size must be > 0".into(),
            ));
        }
        Ok(Self { vocab_size })
    }
}

impl CsmTextTokenizer for FixtureByteTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "csm fixture tokenizer: text must be non-empty".into(),
            ));
        }
        Ok(text
            .bytes()
            .map(|b| u32::from(b) % self.vocab_size as u32)
            .collect())
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufArray, GgufBuilder, GgufValueType};

    fn gguf_with_tokenizer(bytes: Option<&[u8]>) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "csm");
        if let Some(bytes) = bytes {
            b.add_metadata(
                KEY_TOKENIZER_MODEL,
                GgufMetadataValue::Array(GgufArray {
                    element_type: GgufValueType::U8,
                    values: bytes.iter().map(|&x| GgufMetadataValue::U8(x)).collect(),
                }),
            );
        }
        GgufFile::parse(b.to_bytes().unwrap()).unwrap()
    }

    #[test]
    fn blob_reads_back_verbatim_and_encode_is_honestly_stubbed() {
        let file = gguf_with_tokenizer(Some(b"blob"));
        let tok = GgufCsmTokenizer::from_gguf(&file, 100).expect("read back");
        assert_eq!(tok.raw_bytes(), b"blob");
        assert_eq!(tok.vocab_size(), 100);
        assert!(matches!(
            tok.encode("hello"),
            Err(VokraError::NotImplemented(_))
        ));
    }

    #[test]
    fn missing_blob_is_a_loud_model_load_error() {
        let file = gguf_with_tokenizer(None);
        assert!(matches!(
            GgufCsmTokenizer::from_gguf(&file, 100),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn fixture_tokenizer_is_deterministic_and_bounded() {
        let tok = FixtureByteTokenizer::new(11).unwrap();
        let a = tok.encode("Vokra CSM").unwrap();
        let b = tok.encode("Vokra CSM").unwrap();
        assert_eq!(a, b);
        assert!(a.iter().all(|&t| t < 11));
        assert!(tok.encode("").is_err(), "empty text is loud");
        assert!(FixtureByteTokenizer::new(0).is_err());
    }
}
