//! The authenticated Dia text boundary.
//!
//! The pinned upstream `_encode_text` path is authenticated to first encode
//! the complete string as UTF-8 bytes, replace the two speaker markers with
//! byte ids `1` and `2`, and truncate to `data.text_length`.  This module
//! exposes that observed boundary without claiming any unverified text
//! normalization or sidecar vocabulary.

use vokra_core::{Result, VokraError};

use super::DiaConfig;

/// Dia's fixed byte-level source vocabulary.
pub const DIA_TEXT_SOURCE_VOCAB_SIZE: usize = 256;
/// Official speaker-one marker in Dia's text input.
pub const DIA_SPEAKER_ONE_MARKER: &[u8] = b"[S1]";
/// Official speaker-two marker in Dia's text input.
pub const DIA_SPEAKER_TWO_MARKER: &[u8] = b"[S2]";
/// Byte id emitted for [`DIA_SPEAKER_ONE_MARKER`].
pub const DIA_SPEAKER_ONE_ID: u32 = 1;
/// Byte id emitted for [`DIA_SPEAKER_TWO_MARKER`].
pub const DIA_SPEAKER_TWO_ID: u32 = 2;

/// Strict tokenizer contract for the pinned Dia-1.6B text encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiaTokenizer {
    text_length: usize,
}

impl DiaTokenizer {
    /// Binds the byte tokenizer to an authenticated Dia-1.6B configuration.
    ///
    /// A miniature test configuration is intentionally rejected: accepting a
    /// source vocabulary smaller than the byte range would silently turn
    /// ordinary UTF-8 input into invalid embedding ids.
    pub fn from_config(config: &DiaConfig) -> Result<Self> {
        config.validate_for_forward()?;
        if config.src_vocab_size != DIA_TEXT_SOURCE_VOCAB_SIZE {
            return Err(VokraError::InvalidArgument(format!(
                "dia tokenizer: source vocabulary {} is not the authenticated byte vocabulary {}",
                config.src_vocab_size, DIA_TEXT_SOURCE_VOCAB_SIZE
            )));
        }
        Ok(Self {
            text_length: config.text_length,
        })
    }

    /// Encodes one complete input string using the official byte boundary.
    ///
    /// Marker replacement happens before truncation, matching the upstream
    /// source adapter.  An empty string produces an empty sequence; the
    /// generation route separately rejects an empty sequence because Dia's
    /// encoder requires at least one valid source position.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if self.text_length == 0 {
            return Err(VokraError::InvalidArgument(
                "dia tokenizer: text length cap is zero".to_owned(),
            ));
        }
        let bytes = replace_marker(
            &replace_marker(
                text.as_bytes(),
                DIA_SPEAKER_ONE_MARKER,
                DIA_SPEAKER_ONE_ID as u8,
            ),
            DIA_SPEAKER_TWO_MARKER,
            DIA_SPEAKER_TWO_ID as u8,
        );
        Ok(bytes
            .into_iter()
            .take(self.text_length)
            .map(u32::from)
            .collect())
    }
}

fn replace_marker(input: &[u8], marker: &[u8], replacement: u8) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len());
    let mut cursor = 0;
    while cursor < input.len() {
        if input[cursor..].starts_with(marker) {
            output.push(replacement);
            cursor += marker.len();
        } else {
            output.push(input[cursor]);
            cursor += 1;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_marker_and_utf8_boundary_is_exact() {
        let tokenizer = DiaTokenizer::from_config(&DiaConfig::dia_1_6b()).expect("config");
        assert_eq!(
            tokenizer.encode("A[S1]é[S2]").expect("encode"),
            vec![65, 1, 195, 169, 2]
        );
    }

    #[test]
    fn replacement_happens_before_text_length_truncation() {
        let mut config = DiaConfig::dia_1_6b();
        config.text_length = 2;
        let tokenizer = DiaTokenizer::from_config(&config).expect("config");
        assert_eq!(tokenizer.encode("[S1]x").expect("encode"), vec![1, 120]);
    }

    #[test]
    fn non_byte_fixture_is_rejected() {
        assert!(DiaTokenizer::from_config(&DiaConfig::tiny_for_tests()).is_err());
    }
}
