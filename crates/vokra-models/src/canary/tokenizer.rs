//! Decode-only unified SentencePiece vocabulary and Canary2 prompt for v2.

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::{Result, VokraError};

use crate::strict_checkpoint::sha256_bytes;

/// GGUF metadata key holding the exact tab-separated aggregate vocabulary.
pub const KEY_TOKENIZER_VOCAB: &str = "vokra.canary.tokenizer.vocab";
/// GGUF metadata key holding the aggregate vocabulary SHA-256.
pub const KEY_TOKENIZER_VOCAB_SHA256: &str = "vokra.canary.tokenizer.vocab_sha256";
/// SHA-256 of the official 16,384-line aggregate vocabulary.
pub const TOKENIZER_VOCAB_SHA256: &str =
    "4d10723a8bef5b8b186c3d2bb1449c849cc25c6b811969a7d170261b0ceed178";

const TOKENIZER_VOCAB_SHA256_BYTES: [u8; 32] = [
    0x4d, 0x10, 0x72, 0x3a, 0x8b, 0xef, 0x5b, 0x8b, 0x18, 0x6c, 0x3d, 0x2b, 0xb1, 0x44, 0x9c, 0x84,
    0x9c, 0xc2, 0x5c, 0x6b, 0x81, 0x19, 0x69, 0xa7, 0xd1, 0x70, 0x26, 0x1b, 0x0c, 0xee, 0xd1, 0x78,
];

/// Number of pieces in the official aggregate tokenizer.
pub const VOCAB_SIZE: usize = 16_384;
/// Number of reserved special pieces before ordinary SentencePiece tokens.
pub const SPECIAL_VOCAB_SIZE: usize = 1_163;

/// Padding-token identifier in the official aggregate vocabulary.
pub const PAD_ID: u32 = 2;
/// End-of-text token identifier in the official aggregate vocabulary.
pub const EOS_ID: u32 = 3;
/// Start-of-transcript token identifier in the official aggregate vocabulary.
pub const BOS_ID: u32 = 4;
const PNC_ID: u32 = 5;
const NO_PNC_ID: u32 = 6;
const START_OF_CONTEXT_ID: u32 = 7;
const ITN_ID: u32 = 8;
const NO_ITN_ID: u32 = 9;
const TIMESTAMP_ID: u32 = 10;
const NO_TIMESTAMP_ID: u32 = 11;
const DIARIZE_ID: u32 = 12;
const NO_DIARIZE_ID: u32 = 13;

/// The 25 languages in the official Canary-1B-v2 aggregate tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CanaryLanguage {
    /// Bulgarian (`bg`).
    Bulgarian,
    /// Croatian (`hr`).
    Croatian,
    /// Czech (`cs`).
    Czech,
    /// Danish (`da`).
    Danish,
    /// Dutch (`nl`).
    Dutch,
    /// English (`en`).
    English,
    /// Estonian (`et`).
    Estonian,
    /// Finnish (`fi`).
    Finnish,
    /// French (`fr`).
    French,
    /// German (`de`).
    German,
    /// Greek (`el`).
    Greek,
    /// Hungarian (`hu`).
    Hungarian,
    /// Italian (`it`).
    Italian,
    /// Latvian (`lv`).
    Latvian,
    /// Lithuanian (`lt`).
    Lithuanian,
    /// Maltese (`mt`).
    Maltese,
    /// Polish (`pl`).
    Polish,
    /// Portuguese (`pt`).
    Portuguese,
    /// Romanian (`ro`).
    Romanian,
    /// Russian (`ru`).
    Russian,
    /// Slovak (`sk`).
    Slovak,
    /// Slovenian (`sl`).
    Slovenian,
    /// Spanish (`es`).
    Spanish,
    /// Swedish (`sv`).
    Swedish,
    /// Ukrainian (`uk`).
    Ukrainian,
}

impl CanaryLanguage {
    /// Returns the official two-letter language code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Bulgarian => "bg",
            Self::Croatian => "hr",
            Self::Czech => "cs",
            Self::Danish => "da",
            Self::Dutch => "nl",
            Self::English => "en",
            Self::Estonian => "et",
            Self::Finnish => "fi",
            Self::French => "fr",
            Self::German => "de",
            Self::Greek => "el",
            Self::Hungarian => "hu",
            Self::Italian => "it",
            Self::Latvian => "lv",
            Self::Lithuanian => "lt",
            Self::Maltese => "mt",
            Self::Polish => "pl",
            Self::Portuguese => "pt",
            Self::Romanian => "ro",
            Self::Russian => "ru",
            Self::Slovak => "sk",
            Self::Slovenian => "sl",
            Self::Spanish => "es",
            Self::Swedish => "sv",
            Self::Ukrainian => "uk",
        }
    }

    /// Returns the exact aggregate-tokenizer prompt token identifier.
    #[must_use]
    pub const fn prompt_token_id(self) -> u32 {
        match self {
            Self::Bulgarian => 46,
            Self::Croatian => 58,
            Self::Czech => 59,
            Self::Danish => 60,
            Self::Dutch => 62,
            Self::English => 64,
            Self::Estonian => 66,
            Self::Finnish => 70,
            Self::French => 71,
            Self::German => 78,
            Self::Greek => 79,
            Self::Hungarian => 89,
            Self::Italian => 99,
            Self::Latvian => 117,
            Self::Lithuanian => 120,
            Self::Maltese => 127,
            Self::Polish => 150,
            Self::Portuguese => 151,
            Self::Romanian => 154,
            Self::Russian => 157,
            Self::Slovak => 167,
            Self::Slovenian => 168,
            Self::Spanish => 171,
            Self::Swedish => 175,
            Self::Ukrainian => 192,
        }
    }

    /// Parses one of the 25 supported two-letter language codes.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] for an unsupported code.
    pub fn parse(code: &str) -> Result<Self> {
        let value = match code {
            "bg" => Self::Bulgarian,
            "hr" => Self::Croatian,
            "cs" => Self::Czech,
            "da" => Self::Danish,
            "nl" => Self::Dutch,
            "en" => Self::English,
            "et" => Self::Estonian,
            "fi" => Self::Finnish,
            "fr" => Self::French,
            "de" => Self::German,
            "el" => Self::Greek,
            "hu" => Self::Hungarian,
            "it" => Self::Italian,
            "lv" => Self::Latvian,
            "lt" => Self::Lithuanian,
            "mt" => Self::Maltese,
            "pl" => Self::Polish,
            "pt" => Self::Portuguese,
            "ro" => Self::Romanian,
            "ru" => Self::Russian,
            "sk" => Self::Slovak,
            "sl" => Self::Slovenian,
            "es" => Self::Spanish,
            "sv" => Self::Swedish,
            "uk" => Self::Ukrainian,
            other => {
                return Err(VokraError::InvalidArgument(format!(
                    "Canary-1B-v2 language {other:?} is unsupported; expected one of bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, ru, sk, sl, es, sv, uk"
                )));
            }
        };
        Ok(value)
    }
}

/// Emotion control token encoded in the Canary2 prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CanaryEmotion {
    /// Leave emotion unspecified, matching the default NeMo prompt.
    Undefined,
    /// Request neutral speech tagging.
    Neutral,
    /// Request happy speech tagging.
    Happy,
    /// Request sad speech tagging.
    Sad,
    /// Request angry speech tagging.
    Angry,
}

impl CanaryEmotion {
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

/// Exact `canary2` prompt controls for Canary-1B-v2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Canary1bV2Options {
    /// Language spoken in the input audio.
    pub source_language: CanaryLanguage,
    /// Requested transcript or translation language.
    pub target_language: CanaryLanguage,
    /// Whether the output should contain punctuation and capitalization.
    pub punctuation: bool,
    /// Whether inverse text normalization should be requested.
    pub inverse_text_normalization: bool,
    /// Whether timestamp tokens should be requested.
    pub timestamps: bool,
    /// Whether speaker diarization tokens should be requested.
    pub diarize: bool,
    /// Emotion prompt control.
    pub emotion: CanaryEmotion,
    /// Optional bound on tokens generated after the nine-token prompt.
    pub max_new_tokens: Option<usize>,
}

impl Default for Canary1bV2Options {
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

impl Canary1bV2Options {
    /// Exact nine-token prompt with an empty decoder-context slot.
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

/// Authenticated decode-only aggregate tokenizer embedded in the GGUF.
#[derive(Debug, Clone)]
pub struct CanaryTokenizer {
    pieces: Vec<String>,
}

impl CanaryTokenizer {
    /// Loads and authenticates the aggregate vocabulary from GGUF metadata.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::ModelLoad`] when metadata, hash, vocabulary
    /// length, or required special-token positions diverge from the release.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let bytes = required_u8_array(file, KEY_TOKENIZER_VOCAB)?;
        let actual_hash = sha256_bytes(&bytes);
        if actual_hash != TOKENIZER_VOCAB_SHA256_BYTES {
            return Err(VokraError::ModelLoad(format!(
                "Canary-1B-v2 tokenizer: `{KEY_TOKENIZER_VOCAB}` SHA-256 does not match `{TOKENIZER_VOCAB_SHA256}`"
            )));
        }
        let stamped = file
            .get(KEY_TOKENIZER_VOCAB_SHA256)
            .and_then(GgufMetadataValue::as_str)
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "Canary-1B-v2 tokenizer: missing/non-string `{KEY_TOKENIZER_VOCAB_SHA256}`"
                ))
            })?;
        if stamped != TOKENIZER_VOCAB_SHA256 {
            return Err(VokraError::ModelLoad(format!(
                "Canary-1B-v2 tokenizer: stamped SHA-256 {stamped:?}, expected {TOKENIZER_VOCAB_SHA256:?}"
            )));
        }
        Self::from_vocab_bytes(&bytes)
    }

    /// Builds a tokenizer from the exact official tab-separated vocabulary.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::ModelLoad`] unless the bytes match the immutable
    /// hash and all required release boundaries.
    pub fn from_vocab_bytes(bytes: &[u8]) -> Result<Self> {
        if sha256_bytes(bytes) != TOKENIZER_VOCAB_SHA256_BYTES {
            return Err(VokraError::ModelLoad(format!(
                "Canary-1B-v2 tokenizer vocabulary SHA-256 does not match `{TOKENIZER_VOCAB_SHA256}`"
            )));
        }
        let document = std::str::from_utf8(bytes).map_err(|error| {
            VokraError::ModelLoad(format!(
                "Canary-1B-v2 tokenizer vocabulary is not UTF-8: {error}"
            ))
        })?;
        let mut pieces = Vec::with_capacity(VOCAB_SIZE);
        for (line_index, line) in document.lines().enumerate() {
            let (piece, score) = line.rsplit_once('\t').ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "Canary-1B-v2 tokenizer line {} is not `piece<TAB>score`",
                    line_index + 1
                ))
            })?;
            let score = score.parse::<f32>().map_err(|error| {
                VokraError::ModelLoad(format!(
                    "Canary-1B-v2 tokenizer line {} has invalid score: {error}",
                    line_index + 1
                ))
            })?;
            if piece.is_empty() || !score.is_finite() {
                return Err(VokraError::ModelLoad(format!(
                    "Canary-1B-v2 tokenizer line {} is malformed",
                    line_index + 1
                )));
            }
            pieces.push(piece.to_owned());
        }
        if pieces.len() != VOCAB_SIZE {
            return Err(VokraError::ModelLoad(format!(
                "Canary-1B-v2 unified tokenizer has {} pieces, expected {VOCAB_SIZE}",
                pieces.len()
            )));
        }
        for (id, expected) in required_special_pieces() {
            if pieces[id as usize] != expected {
                return Err(VokraError::ModelLoad(format!(
                    "Canary-1B-v2 tokenizer id {id} must be {expected:?}, found {:?}",
                    pieces[id as usize]
                )));
            }
        }
        if pieces[SPECIAL_VOCAB_SIZE] != "en" {
            return Err(VokraError::ModelLoad(format!(
                "Canary-1B-v2 first non-special piece at {SPECIAL_VOCAB_SIZE} must be `en`, found {:?}",
                pieces[SPECIAL_VOCAB_SIZE]
            )));
        }
        Ok(Self { pieces })
    }

    #[must_use]
    /// Returns the authenticated vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// Decodes generated IDs while suppressing the reserved prompt region.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] for an out-of-range ID.
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let mut encoded = String::new();
        for &token_id in token_ids {
            let id = token_id as usize;
            let piece = self.pieces.get(id).ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "Canary-1B-v2 tokenizer id {token_id} outside 0..{}",
                    self.pieces.len()
                ))
            })?;
            if id >= SPECIAL_VOCAB_SIZE {
                encoded.push_str(piece);
            }
        }
        Ok(encoded.replace('▁', " ").trim().to_owned())
    }
}

fn required_special_pieces() -> impl Iterator<Item = (u32, &'static str)> {
    [
        (0, "<unk>"),
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
        (46, "<|bg|>"),
        (58, "<|hr|>"),
        (59, "<|cs|>"),
        (60, "<|da|>"),
        (62, "<|nl|>"),
        (64, "<|en|>"),
        (66, "<|et|>"),
        (70, "<|fi|>"),
        (71, "<|fr|>"),
        (78, "<|de|>"),
        (79, "<|el|>"),
        (89, "<|hu|>"),
        (99, "<|it|>"),
        (117, "<|lv|>"),
        (120, "<|lt|>"),
        (127, "<|mt|>"),
        (150, "<|pl|>"),
        (151, "<|pt|>"),
        (154, "<|ro|>"),
        (157, "<|ru|>"),
        (167, "<|sk|>"),
        (168, "<|sl|>"),
        (171, "<|es|>"),
        (175, "<|sv|>"),
        (192, "<|uk|>"),
    ]
    .into_iter()
}

fn required_u8_array(file: &GgufFile, key: &str) -> Result<Vec<u8>> {
    match file.get(key) {
        Some(GgufMetadataValue::Array(array)) => array
            .values
            .iter()
            .map(|value| match value {
                GgufMetadataValue::U8(byte) => Ok(*byte),
                _ => Err(VokraError::ModelLoad(format!(
                    "Canary-1B-v2 tokenizer: `{key}` contains a non-u8 element"
                ))),
            })
            .collect(),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "Canary-1B-v2 tokenizer: `{key}` must be a u8 array, found {other:?}"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "Canary-1B-v2 tokenizer: `{key}` is absent; reconvert the complete main checkpoint with the official tokenizer.vocab"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_canary2_prompt_is_exact_english_asr() {
        assert_eq!(
            Canary1bV2Options::default().prompt_tokens(),
            [7, 4, 16, 64, 64, 5, 9, 11, 13]
        );
    }

    #[test]
    fn all_25_language_codes_round_trip() {
        for language in [
            CanaryLanguage::Bulgarian,
            CanaryLanguage::Croatian,
            CanaryLanguage::Czech,
            CanaryLanguage::Danish,
            CanaryLanguage::Dutch,
            CanaryLanguage::English,
            CanaryLanguage::Estonian,
            CanaryLanguage::Finnish,
            CanaryLanguage::French,
            CanaryLanguage::German,
            CanaryLanguage::Greek,
            CanaryLanguage::Hungarian,
            CanaryLanguage::Italian,
            CanaryLanguage::Latvian,
            CanaryLanguage::Lithuanian,
            CanaryLanguage::Maltese,
            CanaryLanguage::Polish,
            CanaryLanguage::Portuguese,
            CanaryLanguage::Romanian,
            CanaryLanguage::Russian,
            CanaryLanguage::Slovak,
            CanaryLanguage::Slovenian,
            CanaryLanguage::Spanish,
            CanaryLanguage::Swedish,
            CanaryLanguage::Ukrainian,
        ] {
            assert_eq!(CanaryLanguage::parse(language.code()).unwrap(), language);
        }
    }

    #[test]
    fn translation_prompt_changes_only_target_slot() {
        let options = Canary1bV2Options {
            target_language: CanaryLanguage::German,
            ..Canary1bV2Options::default()
        };
        assert_eq!(options.prompt_tokens(), [7, 4, 16, 64, 78, 5, 9, 11, 13]);
    }
}
