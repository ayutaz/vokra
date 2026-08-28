//! Exact tokenizer contract for the pinned Microsoft SpeechT5 TTS release.
//!
//! The upstream vocabulary is a 79-piece SentencePiece `CHAR` model. Hugging
//! Face appends `<mask>` and `<ctc_blank>` at ids 79 and 80, matching the
//! checkpoint's 81-row text embedding. The converter stores the raw model,
//! the expanded piece/score arrays, and the immutable NMT-NFKC normalizer
//! contract. This runtime validates all of them before exposing tokenization.

use std::collections::HashMap;

use vokra_core::gguf::{GgufFile, GgufMetadataValue, GgufValueType};
use vokra_core::{Result, VokraError};

use crate::strict_checkpoint::sha256_bytes;

use super::{EOS_TOKEN_ID, MAX_TEXT_POSITIONS, VOCAB_SIZE};

const LABEL: &str = "SpeechT5-TTS tokenizer";
const PREFIX: &str = "vokra.speecht5.tokenizer";

pub(super) const KEY_MODEL_SHA256: &str = "vokra.speecht5.tokenizer.model_sha256";
const KEY_ADDED_TOKENS_SHA256: &str = "vokra.speecht5.tokenizer.added_tokens_sha256";
const KEY_VOCAB_MANIFEST_SHA256: &str = "vokra.speecht5.tokenizer.vocab_manifest_sha256";
const KEY_MODEL: &str = "vokra.speecht5.tokenizer.model";
const KEY_PIECES: &str = "vokra.speecht5.tokenizer.pieces";
const KEY_SCORES: &str = "vokra.speecht5.tokenizer.scores";

pub(super) const MODEL_SHA256: &str =
    "7fcc48f3e225f627b1641db410ceb0c8649bd2b0c982e150b03f8be3728ab560";
const MODEL_SHA256_BYTES: [u8; 32] = [
    0x7f, 0xcc, 0x48, 0xf3, 0xe2, 0x25, 0xf6, 0x27, 0xb1, 0x64, 0x1d, 0xb4, 0x10, 0xce, 0xb0, 0xc8,
    0x64, 0x9b, 0xd2, 0xb0, 0xc9, 0x82, 0xe1, 0x50, 0xb0, 0x3f, 0x8b, 0xe3, 0x72, 0x8a, 0xb5, 0x60,
];
const ADDED_TOKENS_SHA256: &str =
    "74be21ecff0a1fb1f304fe7c72ab21e4f0c046f8359fdf2852eb1b80967069ad";
const VOCAB_MANIFEST_SHA256: &str =
    "2b04363543fae9615b30cc91e1b0ed76fba73f91dd23aefb60eed984dc85ee96";
const VOCAB_MANIFEST_SHA256_BYTES: [u8; 32] = [
    0x2b, 0x04, 0x36, 0x35, 0x43, 0xfa, 0xe9, 0x61, 0x5b, 0x30, 0xcc, 0x91, 0xe1, 0xb0, 0xed, 0x76,
    0xfb, 0xa7, 0x3f, 0x91, 0xdd, 0x23, 0xae, 0xfb, 0x60, 0xee, 0xd9, 0x84, 0xdc, 0x85, 0xee, 0x96,
];

const BASE_VOCAB_SIZE: usize = 79;
const BOS_ID: u32 = 0;
const PAD_ID: u32 = 1;
const UNK_ID: u32 = 3;
const MASK_ID: u32 = 79;
const CTC_BLANK_ID: u32 = 80;
const SPACE_PIECE_ID: u32 = 4;

// Exact expanded vocabulary authenticated from microsoft/speecht5_tts at
// SOURCE_REVISION. Keeping the already-audited pieces and scores in the
// zero-dependency runtime lets the exact historical Vokra GGUF recover its
// omitted tokenizer without accepting caller-provided or inferred data.
const OFFICIAL_PIECES: [&str; VOCAB_SIZE] = [
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "▁",
    "e",
    "t",
    "a",
    "o",
    "n",
    "i",
    "h",
    "s",
    "r",
    "d",
    "l",
    "u",
    "c",
    "m",
    "f",
    "w",
    "g",
    "y",
    ",",
    "p",
    "b",
    ".",
    "v",
    "k",
    "\"",
    "I",
    "'",
    "T",
    "A",
    "S",
    "H",
    ";",
    "x",
    "W",
    "-",
    "B",
    "?",
    "C",
    "M",
    "!",
    "q",
    "j",
    "E",
    "N",
    "P",
    "O",
    "D",
    "L",
    "G",
    "R",
    "F",
    "Y",
    "z",
    "J",
    ":",
    "K",
    "U",
    "V",
    ")",
    "(",
    "Q",
    "Z",
    "]",
    "[",
    "X",
    "—",
    "/",
    "æ",
    "é",
    "{",
    "}",
    "ê",
    "œ",
    "̄",
    "<mask>",
    "<ctc_blank>",
];

const OFFICIAL_SCORE_BITS: [u32; VOCAB_SIZE] = [
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xbfdad4ee, 0xc0144b19, 0xc029bbf7, 0xc033608a,
    0xc0356c63, 0xc03cceb9, 0xc0408fc8, 0xc04107d2, 0xc042aade, 0xc0468320, 0xc0586b8b, 0xc05de7a9,
    0xc07427a1, 0xc07ee1f4, 0xc07f1e67, 0xc081bc95, 0xc081e942, 0xc08607d9, 0xc0867cd7, 0xc0899ec9,
    0xc08a5dd0, 0xc090e0d6, 0xc09519f9, 0xc09e3f33, 0xc0a271c1, 0xc0a806b5, 0xc0b20357, 0xc0bb0c55,
    0xc0bdf983, 0xc0cc9862, 0xc0d19ff8, 0xc0d1f8e5, 0xc0d68b7f, 0xc0d70a8b, 0xc0d888b9, 0xc0d966bc,
    0xc0d9a725, 0xc0dfe98d, 0xc0e08c1b, 0xc0e19b80, 0xc0e5477a, 0xc0e592a7, 0xc0e605d6, 0xc0e7b454,
    0xc0e8c199, 0xc0e98082, 0xc0ea2577, 0xc0ed7ab8, 0xc0ee6306, 0xc0f1770d, 0xc0f1b0aa, 0xc0f24280,
    0xc0f5ad01, 0xc0f91a48, 0xc101d045, 0xc102e2a2, 0xc1094073, 0xc10be725, 0xc10de4c3, 0xc118aae6,
    0xc1194e85, 0xc11ee7a3, 0xc12b88a0, 0xc13d4fbe, 0xc13e6794, 0xc1401bbc, 0xc144a6c4, 0xc14d4f3d,
    0xc171beed, 0xc17cd60f, 0xc183f698, 0xc183f698, 0xc183f698, 0xc183f698, 0xc183f698, 0x00000000,
    0x00000000,
];

fn official_parts() -> (Vec<String>, Vec<f32>) {
    (
        OFFICIAL_PIECES
            .iter()
            .map(|piece| (*piece).to_owned())
            .collect(),
        OFFICIAL_SCORE_BITS
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect(),
    )
}

/// Self-contained, fail-closed SpeechT5 text tokenizer.
#[derive(Debug, Clone)]
pub struct SpeechT5Tokenizer {
    character_ids: HashMap<char, u32>,
}

impl SpeechT5Tokenizer {
    /// Constructs the authenticated expanded vocabulary without requiring the
    /// raw SentencePiece blob omitted by the historical public Vokra GGUF.
    pub(super) fn official() -> Result<Self> {
        let (pieces, scores) = official_parts();
        Self::from_parts(pieces, scores)
    }

    pub(super) fn from_gguf(file: &GgufFile) -> Result<Self> {
        required_string(file, &format!("{PREFIX}.scheme"), "char")?;
        required_string(file, &format!("{PREFIX}.kind"), "sentencepiece-char")?;
        required_string(file, KEY_MODEL_SHA256, MODEL_SHA256)?;
        required_string(file, KEY_ADDED_TOKENS_SHA256, ADDED_TOKENS_SHA256)?;
        required_string(file, KEY_VOCAB_MANIFEST_SHA256, VOCAB_MANIFEST_SHA256)?;
        required_string(file, &format!("{PREFIX}.normalizer"), "nmt_nfkc")?;
        required_bool(file, &format!("{PREFIX}.normalizer.add_dummy_prefix"), true)?;
        required_bool(
            file,
            &format!("{PREFIX}.normalizer.remove_extra_whitespaces"),
            true,
        )?;
        required_bool(
            file,
            &format!("{PREFIX}.normalizer.escape_whitespaces"),
            true,
        )?;
        for (key, expected) in [
            (&format!("{PREFIX}.base_vocab_size"), BASE_VOCAB_SIZE as u32),
            (&format!("{PREFIX}.vocab_size"), VOCAB_SIZE as u32),
            (&format!("{PREFIX}.bos_id"), BOS_ID),
            (&format!("{PREFIX}.pad_id"), PAD_ID),
            (&format!("{PREFIX}.eos_id"), EOS_TOKEN_ID),
            (&format!("{PREFIX}.unk_id"), UNK_ID),
            (&format!("{PREFIX}.mask_id"), MASK_ID),
            (&format!("{PREFIX}.ctc_blank_id"), CTC_BLANK_ID),
        ] {
            required_u32(file, key, expected)?;
        }

        let model = required_u8_array(file, KEY_MODEL)?;
        if sha256_bytes(&model) != MODEL_SHA256_BYTES {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: raw `{KEY_MODEL}` does not match pinned SHA-256 {MODEL_SHA256}"
            )));
        }
        let pieces = required_string_array(file, KEY_PIECES)?;
        let scores = required_f32_array(file, KEY_SCORES)?;
        Self::from_parts(pieces, scores)
    }

    fn from_parts(pieces: Vec<String>, scores: Vec<f32>) -> Result<Self> {
        if pieces.len() != VOCAB_SIZE || scores.len() != VOCAB_SIZE {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: pieces/scores lengths are {}/{}, expected {VOCAB_SIZE}/{VOCAB_SIZE}",
                pieces.len(),
                scores.len()
            )));
        }
        if scores.iter().any(|score| !score.is_finite()) {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: score array contains a non-finite value"
            )));
        }
        for (id, expected) in [
            (BOS_ID, "<s>"),
            (PAD_ID, "<pad>"),
            (EOS_TOKEN_ID, "</s>"),
            (UNK_ID, "<unk>"),
            (SPACE_PIECE_ID, "▁"),
            (MASK_ID, "<mask>"),
            (CTC_BLANK_ID, "<ctc_blank>"),
        ] {
            if pieces[id as usize] != expected {
                return Err(VokraError::ModelLoad(format!(
                    "{LABEL}: token id {id} is {:?}, expected {expected:?}",
                    pieces[id as usize]
                )));
            }
        }

        let mut canonical = Vec::new();
        for (piece, score) in pieces.iter().zip(&scores) {
            let piece_len = u32::try_from(piece.len()).map_err(|_| {
                VokraError::ModelLoad(format!("{LABEL}: tokenizer piece is too large"))
            })?;
            canonical.extend_from_slice(&piece_len.to_le_bytes());
            canonical.extend_from_slice(piece.as_bytes());
            canonical.extend_from_slice(&score.to_le_bytes());
        }
        if sha256_bytes(&canonical) != VOCAB_MANIFEST_SHA256_BYTES {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: expanded piece/score manifest does not match pinned SHA-256 {VOCAB_MANIFEST_SHA256}"
            )));
        }

        let mut character_ids = HashMap::with_capacity(BASE_VOCAB_SIZE - 4);
        for (id, piece) in pieces
            .iter()
            .enumerate()
            .take(BASE_VOCAB_SIZE)
            .skip(SPACE_PIECE_ID as usize)
        {
            let mut characters = piece.chars();
            let character = characters
                .next()
                .ok_or_else(|| VokraError::ModelLoad(format!("{LABEL}: token id {id} is empty")))?;
            if characters.next().is_some() {
                return Err(VokraError::ModelLoad(format!(
                    "{LABEL}: CHAR-model token id {id} is not one Unicode scalar: {piece:?}"
                )));
            }
            if character_ids.insert(character, id as u32).is_some() {
                return Err(VokraError::ModelLoad(format!(
                    "{LABEL}: duplicate character piece {piece:?}"
                )));
            }
        }
        Ok(Self { character_ids })
    }

    /// Encodes ordinary SpeechT5 TTS text and appends EOS exactly as
    /// `SpeechT5Tokenizer.build_inputs_with_special_tokens` does.
    ///
    /// ASCII whitespace is collapsed and escaped as `▁`. The pinned model's
    /// stable precomposed non-ASCII character pieces are accepted, while any
    /// code point requiring the full NMT-NFKC rewrite table fails explicitly.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: text is empty"
            )));
        }
        if text.contains("<mask>") || text.contains("<ctc_blank>") {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: training-only added tokens are not accepted by the TTS text API"
            )));
        }
        let mut ids = Vec::with_capacity(text.len().min(MAX_TEXT_POSITIONS));
        let mut pending_space = true; // SentencePiece add_dummy_prefix=true.
        let mut emitted_text = false;
        let mut in_unknown_ascii_span = false;

        for character in text.chars() {
            if character.is_ascii_whitespace() {
                if emitted_text {
                    pending_space = true;
                }
                in_unknown_ascii_span = false;
                continue;
            }
            if character.is_whitespace() {
                return Err(VokraError::InvalidArgument(format!(
                    "{LABEL}: non-ASCII whitespace U+{:04X} requires the full NMT-NFKC normalizer and is not accepted by the strict runtime path",
                    character as u32
                )));
            }
            if pending_space {
                ids.push(SPACE_PIECE_ID);
                pending_space = false;
                in_unknown_ascii_span = false;
            }
            if let Some(&id) = self.character_ids.get(&character) {
                // U+0304 COMBINING MACRON is present in the raw vocabulary,
                // but NMT-NFKC may compose it with the preceding scalar. A
                // scalar-only implementation would produce the wrong ids.
                if character == '\u{0304}' {
                    return Err(VokraError::InvalidArgument(format!(
                        "{LABEL}: combining marks require the full NMT-NFKC normalizer"
                    )));
                }
                ids.push(id);
                in_unknown_ascii_span = false;
            } else if character.is_ascii() {
                // SentencePiece CHAR emits one `<unk>` for a consecutive run
                // of unknown normalized scalars (for example `123`).
                if !in_unknown_ascii_span {
                    ids.push(UNK_ID);
                    in_unknown_ascii_span = true;
                }
            } else {
                return Err(VokraError::InvalidArgument(format!(
                    "{LABEL}: U+{:04X} is outside the pinned already-normalized character vocabulary; refusing an inexact NMT-NFKC fallback",
                    character as u32
                )));
            }
            emitted_text = true;
            if ids.len() >= MAX_TEXT_POSITIONS {
                return Err(VokraError::InvalidArgument(format!(
                    "{LABEL}: normalized input reaches {MAX_TEXT_POSITIONS} tokens before EOS"
                )));
            }
        }
        if !emitted_text {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: text contains only whitespace"
            )));
        }
        ids.push(EOS_TOKEN_ID);
        Ok(ids)
    }
}

fn required_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| VokraError::ModelLoad(format!("{LABEL}: missing/non-string `{key}`")))?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}`={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn required_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    let actual = match file.get(key) {
        Some(GgufMetadataValue::U32(value)) => *value,
        Some(other) => {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: `{key}` must be u32, found {other:?}"
            )));
        }
        None => return Err(VokraError::ModelLoad(format!("{LABEL}: missing `{key}`"))),
    };
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}`={actual}, expected {expected}"
        )));
    }
    Ok(())
}

fn required_bool(file: &GgufFile, key: &str, expected: bool) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(GgufMetadataValue::as_bool)
        .ok_or_else(|| VokraError::ModelLoad(format!("{LABEL}: missing/non-bool `{key}`")))?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}`={actual}, expected {expected}"
        )));
    }
    Ok(())
}

fn required_u8_array(file: &GgufFile, key: &str) -> Result<Vec<u8>> {
    let array = file
        .get(key)
        .and_then(GgufMetadataValue::as_array)
        .ok_or_else(|| VokraError::ModelLoad(format!("{LABEL}: missing/non-array `{key}`")))?;
    if array.element_type != GgufValueType::U8 {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}` must declare u8 elements"
        )));
    }
    array
        .values
        .iter()
        .map(|value| match value {
            GgufMetadataValue::U8(byte) => Ok(*byte),
            other => Err(VokraError::ModelLoad(format!(
                "{LABEL}: `{key}` contains non-u8 element {other:?}"
            ))),
        })
        .collect()
}

fn required_string_array(file: &GgufFile, key: &str) -> Result<Vec<String>> {
    let array = file
        .get(key)
        .and_then(GgufMetadataValue::as_array)
        .ok_or_else(|| VokraError::ModelLoad(format!("{LABEL}: missing/non-array `{key}`")))?;
    if array.element_type != GgufValueType::String {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}` must declare string elements"
        )));
    }
    array
        .values
        .iter()
        .map(|value| match value {
            GgufMetadataValue::String(piece) => Ok(piece.clone()),
            other => Err(VokraError::ModelLoad(format!(
                "{LABEL}: `{key}` contains non-string element {other:?}"
            ))),
        })
        .collect()
}

fn required_f32_array(file: &GgufFile, key: &str) -> Result<Vec<f32>> {
    let array = file
        .get(key)
        .and_then(GgufMetadataValue::as_array)
        .ok_or_else(|| VokraError::ModelLoad(format!("{LABEL}: missing/non-array `{key}`")))?;
    if array.element_type != GgufValueType::F32 {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}` must declare f32 elements"
        )));
    }
    array
        .values
        .iter()
        .map(|value| match value {
            GgufMetadataValue::F32(score) => Ok(*score),
            other => Err(VokraError::ModelLoad(format!(
                "{LABEL}: `{key}` contains non-f32 element {other:?}"
            ))),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_char_ids_match_sentencepiece_reference() {
        let (pieces, scores) = official_parts();
        let tokenizer = SpeechT5Tokenizer::from_parts(pieces, scores).unwrap();
        assert_eq!(
            tokenizer.encode("Hello world").unwrap(),
            vec![4, 35, 5, 15, 15, 8, 4, 20, 8, 13, 15, 14, 2]
        );
        assert_eq!(
            tokenizer.encode("  hello  world  ").unwrap(),
            vec![4, 11, 5, 15, 15, 8, 4, 20, 8, 13, 15, 14, 2]
        );
        assert_eq!(tokenizer.encode("123").unwrap(), vec![4, 3, 2]);
    }

    #[test]
    fn unsupported_normalization_is_explicit() {
        let (pieces, scores) = official_parts();
        let tokenizer = SpeechT5Tokenizer::from_parts(pieces, scores).unwrap();
        let error = tokenizer.encode("１２３").unwrap_err();
        assert!(error.to_string().contains("NMT-NFKC"));
        let error = tokenizer.encode("a\u{0304}").unwrap_err();
        assert!(error.to_string().contains("combining marks"));
    }

    #[test]
    fn tampered_vocab_manifest_fails_closed() {
        let (mut pieces, scores) = official_parts();
        pieces[5] = "E".to_owned();
        let error = SpeechT5Tokenizer::from_parts(pieces, scores).unwrap_err();
        assert!(error.to_string().contains("manifest"));
    }
}
