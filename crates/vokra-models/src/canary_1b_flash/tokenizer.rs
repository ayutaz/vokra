//! Decode-only aggregate SentencePiece vocabulary and Canary2 prompt.

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::{Result, VokraError};

use crate::strict_checkpoint::sha256_bytes;

pub const KEY_TOKENIZER_VOCAB: &str = "vokra.canary_1b_flash.tokenizer.vocab";
pub const KEY_TOKENIZER_VOCAB_SHA256: &str = "vokra.canary_1b_flash.tokenizer.vocab_sha256";
pub const TOKENIZER_VOCAB_SHA256: &str =
    "08cb29d15437dbd3f45c26046c2f5994b3b92c86a3aa4a6e27d253d40837db79";

const TOKENIZER_VOCAB_SHA256_BYTES: [u8; 32] = [
    0x08, 0xcb, 0x29, 0xd1, 0x54, 0x37, 0xdb, 0xd3, 0xf4, 0x5c, 0x26, 0x04, 0x6c, 0x2f, 0x59, 0x94,
    0xb3, 0xb9, 0x2c, 0x86, 0xa3, 0xaa, 0x4a, 0x6e, 0x27, 0xd2, 0x53, 0xd4, 0x08, 0x37, 0xdb, 0x79,
];

pub const VOCAB_SIZE: usize = 5_248;
pub const SPECIAL_VOCAB_SIZE: usize = 1_152;
const LANGUAGE_VOCAB_SIZE: usize = 1_024;

pub const PAD_ID: u32 = 2;
pub const EOS_ID: u32 = 3;
pub const BOS_ID: u32 = 4;
const START_OF_CONTEXT_ID: u32 = 7;
const PNC_ID: u32 = 5;
const NO_PNC_ID: u32 = 6;
const ITN_ID: u32 = 8;
const NO_ITN_ID: u32 = 9;
const TIMESTAMP_ID: u32 = 10;
const NO_TIMESTAMP_ID: u32 = 11;
const DIARIZE_ID: u32 = 12;
const NO_DIARIZE_ID: u32 = 13;

/// Four released Canary-1B-Flash languages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CanaryLanguage {
    English,
    German,
    Spanish,
    French,
}

impl CanaryLanguage {
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::English => "en",
            Self::German => "de",
            Self::Spanish => "es",
            Self::French => "fr",
        }
    }

    #[must_use]
    pub const fn prompt_token_id(self) -> u32 {
        match self {
            Self::English => 62,
            Self::French => 69,
            Self::German => 76,
            Self::Spanish => 169,
        }
    }
}

/// Emotion slot supported by the released Canary2 prompt formatter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CanaryEmotion {
    Undefined,
    Neutral,
    Happy,
    Sad,
    Angry,
}

impl CanaryEmotion {
    #[must_use]
    const fn token_id(self) -> u32 {
        match self {
            Self::Undefined => 16,
            Self::Neutral => 17,
            Self::Happy => 18,
            Self::Sad => 19,
            Self::Angry => 20,
        }
    }
}

/// Canary2 generation controls. ASR uses equal source/target languages;
/// translation uses a different target language.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Canary1bFlashOptions {
    pub source_language: CanaryLanguage,
    pub target_language: CanaryLanguage,
    pub punctuation: bool,
    pub inverse_text_normalization: bool,
    pub timestamps: bool,
    pub diarize: bool,
    pub emotion: CanaryEmotion,
    pub max_new_tokens: Option<usize>,
}

impl Default for Canary1bFlashOptions {
    fn default() -> Self {
        Self {
            source_language: CanaryLanguage::English,
            target_language: CanaryLanguage::English,
            punctuation: true,
            inverse_text_normalization: false,
            timestamps: false,
            diarize: false,
            emotion: CanaryEmotion::Undefined,
            max_new_tokens: None,
        }
    }
}

impl Canary1bFlashOptions {
    /// Exact nine-token `Canary2PromptFormatter` user template with an empty
    /// decoder context, as recorded in the released `model_config.yaml`.
    #[must_use]
    pub fn prompt_tokens(self) -> [u32; 9] {
        [
            START_OF_CONTEXT_ID,
            BOS_ID,
            self.emotion.token_id(),
            self.source_language.prompt_token_id(),
            self.target_language.prompt_token_id(),
            if self.punctuation { PNC_ID } else { NO_PNC_ID },
            if self.inverse_text_normalization {
                ITN_ID
            } else {
                NO_ITN_ID
            },
            if self.timestamps {
                TIMESTAMP_ID
            } else {
                NO_TIMESTAMP_ID
            },
            if self.diarize {
                DIARIZE_ID
            } else {
                NO_DIARIZE_ID
            },
        ]
    }
}

/// Aggregate decode table in released YAML order:
/// `spl_tokens, en, de, es, fr`.
#[derive(Debug, Clone)]
pub struct CanaryTokenizer {
    pieces: Vec<String>,
}

impl CanaryTokenizer {
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let bytes = required_u8_array(file, KEY_TOKENIZER_VOCAB)?;
        let actual_hash = sha256_bytes(&bytes);
        if actual_hash != TOKENIZER_VOCAB_SHA256_BYTES {
            return Err(VokraError::ModelLoad(format!(
                "Canary-1B-Flash tokenizer: `{KEY_TOKENIZER_VOCAB}` SHA-256 does not match the pinned released aggregate vocabulary `{TOKENIZER_VOCAB_SHA256}`"
            )));
        }
        let stamped = file
            .get(KEY_TOKENIZER_VOCAB_SHA256)
            .and_then(GgufMetadataValue::as_str)
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "Canary-1B-Flash tokenizer: missing/non-string `{KEY_TOKENIZER_VOCAB_SHA256}`"
                ))
            })?;
        if stamped != TOKENIZER_VOCAB_SHA256 {
            return Err(VokraError::ModelLoad(format!(
                "Canary-1B-Flash tokenizer: stamped SHA-256 {stamped:?}, expected {TOKENIZER_VOCAB_SHA256:?}"
            )));
        }
        Self::from_vocab_bytes(&bytes)
    }

    pub fn from_vocab_bytes(bytes: &[u8]) -> Result<Self> {
        if sha256_bytes(bytes) != TOKENIZER_VOCAB_SHA256_BYTES {
            return Err(VokraError::ModelLoad(format!(
                "Canary-1B-Flash tokenizer vocabulary SHA-256 does not match the pinned released aggregate vocabulary `{TOKENIZER_VOCAB_SHA256}`"
            )));
        }
        let document = std::str::from_utf8(bytes).map_err(|error| {
            VokraError::ModelLoad(format!(
                "Canary-1B-Flash tokenizer vocabulary is not UTF-8: {error}"
            ))
        })?;
        let mut pieces = Vec::with_capacity(VOCAB_SIZE);
        for (line_index, line) in document.lines().enumerate() {
            let (piece, score) = line.rsplit_once('\t').ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "Canary-1B-Flash tokenizer line {} is not `piece<TAB>score`",
                    line_index + 1
                ))
            })?;
            if piece.is_empty() {
                return Err(VokraError::ModelLoad(format!(
                    "Canary-1B-Flash tokenizer line {} has an empty piece",
                    line_index + 1
                )));
            }
            let score = score.parse::<f32>().map_err(|error| {
                VokraError::ModelLoad(format!(
                    "Canary-1B-Flash tokenizer line {} has invalid score: {error}",
                    line_index + 1
                ))
            })?;
            if !score.is_finite() {
                return Err(VokraError::ModelLoad(format!(
                    "Canary-1B-Flash tokenizer line {} has non-finite score",
                    line_index + 1
                )));
            }
            pieces.push(piece.to_owned());
        }
        if pieces.len() != VOCAB_SIZE {
            return Err(VokraError::ModelLoad(format!(
                "Canary-1B-Flash aggregate tokenizer has {} pieces, expected {VOCAB_SIZE}",
                pieces.len()
            )));
        }
        for offset in [
            0,
            SPECIAL_VOCAB_SIZE,
            SPECIAL_VOCAB_SIZE + LANGUAGE_VOCAB_SIZE,
            SPECIAL_VOCAB_SIZE + 2 * LANGUAGE_VOCAB_SIZE,
            SPECIAL_VOCAB_SIZE + 3 * LANGUAGE_VOCAB_SIZE,
        ] {
            if pieces[offset] != "<unk>" {
                return Err(VokraError::ModelLoad(format!(
                    "Canary-1B-Flash aggregate tokenizer component at offset {offset} must begin with `<unk>`, found {:?}",
                    pieces[offset]
                )));
            }
        }
        for (id, expected) in [
            (PAD_ID, "<pad>"),
            (EOS_ID, "<|endoftext|>"),
            (BOS_ID, "<|startoftranscript|>"),
            (PNC_ID, "<|pnc|>"),
            (NO_PNC_ID, "<|nopnc|>"),
            (START_OF_CONTEXT_ID, "<|startofcontext|>"),
            (ITN_ID, "<|itn|>"),
            (NO_ITN_ID, "<|noitn|>"),
            (TIMESTAMP_ID, "<|timestamp|>"),
            (NO_TIMESTAMP_ID, "<|notimestamp|>"),
            (DIARIZE_ID, "<|diarize|>"),
            (NO_DIARIZE_ID, "<|nodiarize|>"),
            (16, "<|emo:undefined|>"),
            (17, "<|emo:neutral|>"),
            (18, "<|emo:happy|>"),
            (19, "<|emo:sad|>"),
            (20, "<|emo:angry|>"),
            (62, "<|en|>"),
            (69, "<|fr|>"),
            (76, "<|de|>"),
            (169, "<|es|>"),
        ] {
            if pieces[id as usize] != expected {
                return Err(VokraError::ModelLoad(format!(
                    "Canary-1B-Flash tokenizer id {id} must be {expected:?}, found {:?}",
                    pieces[id as usize]
                )));
            }
        }
        Ok(Self { pieces })
    }

    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let mut encoded = String::new();
        let component_unknown_ids = [
            SPECIAL_VOCAB_SIZE,
            SPECIAL_VOCAB_SIZE + LANGUAGE_VOCAB_SIZE,
            SPECIAL_VOCAB_SIZE + 2 * LANGUAGE_VOCAB_SIZE,
            SPECIAL_VOCAB_SIZE + 3 * LANGUAGE_VOCAB_SIZE,
        ];
        for &token_id in token_ids {
            let id = token_id as usize;
            let piece = self.pieces.get(id).ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "Canary-1B-Flash tokenizer id {token_id} outside 0..{}",
                    self.pieces.len()
                ))
            })?;
            if id < SPECIAL_VOCAB_SIZE || component_unknown_ids.contains(&id) {
                continue;
            }
            encoded.push_str(piece);
        }
        let decoded = encoded.replace('▁', " ");
        Ok(decoded.trim().to_owned())
    }
}

fn required_u8_array(file: &GgufFile, key: &str) -> Result<Vec<u8>> {
    match file.get(key) {
        Some(GgufMetadataValue::Array(array)) => array
            .values
            .iter()
            .map(|value| match value {
                GgufMetadataValue::U8(byte) => Ok(*byte),
                _ => Err(VokraError::ModelLoad(format!(
                    "Canary-1B-Flash tokenizer: `{key}` contains a non-u8 element"
                ))),
            })
            .collect(),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "Canary-1B-Flash tokenizer: `{key}` must be a u8 array, found {other:?}"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "Canary-1B-Flash tokenizer: `{key}` is absent; reconvert the complete `.nemo` checkpoint with its five aggregate `*_tokenizer.vocab` sidecars"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_canary2_prompt_is_exact_english_asr() {
        assert_eq!(
            Canary1bFlashOptions::default().prompt_tokens(),
            [7, 4, 16, 62, 62, 5, 9, 11, 13]
        );
    }

    #[test]
    fn translation_prompt_changes_only_target_slot() {
        let options = Canary1bFlashOptions {
            target_language: CanaryLanguage::German,
            ..Canary1bFlashOptions::default()
        };
        assert_eq!(options.prompt_tokens(), [7, 4, 16, 62, 76, 5, 9, 11, 13]);
    }
}
