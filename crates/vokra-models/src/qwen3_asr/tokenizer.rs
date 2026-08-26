//! Fixed-revision Qwen2 byte-BPE and prompt contract for Qwen3-ASR.
//!
//! Executable Qwen3-ASR GGUFs carry the five tokenizer, chat-template and
//! generation sidecars from the same immutable upstream revision as the
//! weights.  Every blob is checked by exact byte length and SHA-256 before
//! the BPE tables are exposed.  Runtime downloads and mutable local sidecars
//! are deliberately unsupported.

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::{Result, VokraError};

use crate::cosyvoice2::CosyVoice2Tokenizer;
use crate::strict_checkpoint::sha256_bytes;

/// Raw upstream `vocab.json` embedded as a GGUF U8 array.
pub const KEY_TOKENIZER_VOCAB: &str = "vokra.qwen3_asr.tokenizer.vocab_json";
/// Raw upstream `merges.txt` embedded as a GGUF U8 array.
pub const KEY_TOKENIZER_MERGES: &str = "vokra.qwen3_asr.tokenizer.merges_txt";
/// Raw upstream `tokenizer_config.json` embedded as a GGUF U8 array.
pub const KEY_TOKENIZER_CONFIG: &str = "vokra.qwen3_asr.tokenizer.config_json";
/// Raw upstream `chat_template.json` embedded as a GGUF U8 array.
pub const KEY_CHAT_TEMPLATE: &str = "vokra.qwen3_asr.tokenizer.chat_template_json";
/// Raw upstream `generation_config.json` embedded as a GGUF U8 array.
pub const KEY_GENERATION_CONFIG: &str = "vokra.qwen3_asr.generation.config_json";

/// Number of ordinary byte-BPE entries before Qwen's added tokens.
pub const BASE_VOCAB_SIZE: usize = 151_643;
/// `<|endoftext|>`; also the padding token and one released EOS id.
pub const END_OF_TEXT_TOKEN_ID: u32 = 151_643;
/// `<|im_start|>`.
pub const IM_START_TOKEN_ID: u32 = 151_644;
/// `<|im_end|>` and the second released EOS id.
pub const IM_END_TOKEN_ID: u32 = 151_645;
/// `<|audio_start|>`.
pub const AUDIO_START_TOKEN_ID: u32 = 151_669;
/// `<|audio_end|>`.
pub const AUDIO_END_TOKEN_ID: u32 = 151_670;
/// `<|audio_pad|>` placeholder replaced by one projected audio row.
pub const AUDIO_PAD_TOKEN_ID: u32 = 151_676;
/// `<asr_text>` output separator and forced-language prompt suffix.
pub const ASR_TEXT_TOKEN_ID: u32 = 151_704;

const ASR_TEXT_TAG: &str = "<asr_text>";
const LANG_PREFIX: &str = "language ";

const VOCAB: ExactAsset = ExactAsset {
    key: KEY_TOKENIZER_VOCAB,
    file_name: "vocab.json",
    bytes: 2_776_833,
    sha256: "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
};
const MERGES: ExactAsset = ExactAsset {
    key: KEY_TOKENIZER_MERGES,
    file_name: "merges.txt",
    bytes: 1_671_853,
    sha256: "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
};
const TOKENIZER_CONFIG: ExactAsset = ExactAsset {
    key: KEY_TOKENIZER_CONFIG,
    file_name: "tokenizer_config.json",
    bytes: 12_487,
    sha256: "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c",
};
const CHAT_TEMPLATE: ExactAsset = ExactAsset {
    key: KEY_CHAT_TEMPLATE,
    file_name: "chat_template.json",
    bytes: 1_161,
    sha256: "75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff",
};
const GENERATION_CONFIG: ExactAsset = ExactAsset {
    key: KEY_GENERATION_CONFIG,
    file_name: "generation_config.json",
    bytes: 142,
    sha256: "1da527824d81e07118facff437e03f2e24a23311e3bdeb2368973fe77e5f275c",
};

/// Language names accepted by the official Qwen3-ASR inference wrapper.
pub const SUPPORTED_LANGUAGES: &[&str] = &[
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
];

#[derive(Debug, Clone, Copy)]
struct ExactAsset {
    key: &'static str,
    file_name: &'static str,
    bytes: usize,
    sha256: &'static str,
}

/// One parsed Qwen3-ASR result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3AsrTranscription {
    /// Canonical detected or caller-forced language, empty when unknown.
    pub language: String,
    /// Transcript with the model's metadata prefix removed.
    pub text: String,
}

/// Exact Qwen3-ASR byte-BPE, ChatML prompt and output parser.
#[derive(Debug, Clone)]
pub struct Qwen3AsrTokenizer {
    bpe: CosyVoice2Tokenizer,
}

impl Qwen3AsrTokenizer {
    /// Authenticates all five embedded release sidecars and builds the BPE.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::ModelLoad`] for a missing, mistyped, truncated,
    /// modified or malformed sidecar.  A historical GGUF without the assets
    /// remains descriptor-bindable, but is intentionally not executable.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let vocab = read_exact_u8_array(file, VOCAB)?;
        let merges = read_exact_u8_array(file, MERGES)?;
        // These blobs define the special-token ids, exact ChatML formatting
        // and deterministic generation defaults. Their bytes are pinned even
        // though execution uses the audited constants below.
        read_exact_u8_array(file, TOKENIZER_CONFIG)?;
        read_exact_u8_array(file, CHAT_TEMPLATE)?;
        read_exact_u8_array(file, GENERATION_CONFIG)?;

        let bpe = CosyVoice2Tokenizer::from_parts(&vocab, &merges).map_err(|error| {
            VokraError::ModelLoad(format!(
                "qwen3_asr tokenizer: fixed Qwen2 vocab/merges failed to parse: {error}"
            ))
        })?;
        if bpe.vocab_size() != BASE_VOCAB_SIZE {
            return Err(VokraError::ModelLoad(format!(
                "qwen3_asr tokenizer: vocab.json has {} entries, expected exactly {BASE_VOCAB_SIZE}",
                bpe.vocab_size()
            )));
        }
        Ok(Self { bpe })
    }

    /// Produces the exact ChatML token sequence consumed by the decoder.
    ///
    /// One `<|audio_pad|>` is emitted per projected audio row. When
    /// `language` is present, the official `language X<asr_text>` suffix is
    /// appended and the decoder is expected to emit transcript text only.
    pub fn prompt_ids(
        &self,
        audio_frames: usize,
        context: Option<&str>,
        language: Option<&str>,
    ) -> Result<Vec<u32>> {
        if audio_frames == 0 {
            return Err(VokraError::InvalidArgument(
                "qwen3_asr tokenizer: audio_frames must be greater than zero".to_owned(),
            ));
        }
        let language = language.map(normalize_language).transpose()?;
        let mut ids = Vec::with_capacity(audio_frames.saturating_add(64));
        ids.push(IM_START_TOKEN_ID);
        self.push_text(&mut ids, "system\n")?;
        self.push_text(&mut ids, context.unwrap_or_default())?;
        ids.push(IM_END_TOKEN_ID);
        self.push_text(&mut ids, "\n")?;
        ids.push(IM_START_TOKEN_ID);
        self.push_text(&mut ids, "user\n")?;
        ids.push(AUDIO_START_TOKEN_ID);
        ids.extend(std::iter::repeat_n(AUDIO_PAD_TOKEN_ID, audio_frames));
        ids.push(AUDIO_END_TOKEN_ID);
        ids.push(IM_END_TOKEN_ID);
        self.push_text(&mut ids, "\n")?;
        ids.push(IM_START_TOKEN_ID);
        self.push_text(&mut ids, "assistant\n")?;
        if let Some(language) = language {
            self.push_text(&mut ids, LANG_PREFIX)?;
            self.push_text(&mut ids, &language)?;
            ids.push(ASR_TEXT_TOKEN_ID);
        }
        Ok(ids)
    }

    /// Decodes generated ids, preserving the Qwen3-ASR metadata separator.
    ///
    /// Decoding stops at either released EOS id. Any other added/control token
    /// in model output is rejected explicitly rather than silently removed.
    pub fn decode_generated_ids(&self, ids: &[u32]) -> Result<String> {
        let mut decoded = String::new();
        let mut ordinary = Vec::new();
        for &id in ids {
            if is_eos(id) {
                break;
            }
            if id == ASR_TEXT_TOKEN_ID {
                self.flush_decoded(&mut ordinary, &mut decoded)?;
                decoded.push_str(ASR_TEXT_TAG);
            } else if (id as usize) < BASE_VOCAB_SIZE {
                ordinary.push(id);
            } else {
                return Err(VokraError::InvalidArgument(format!(
                    "qwen3_asr tokenizer: generated unexpected added/control token id {id}; only base BPE, <asr_text> and EOS are valid"
                )));
            }
        }
        self.flush_decoded(&mut ordinary, &mut decoded)?;
        Ok(decoded)
    }

    /// Decodes and structurally parses one generated sequence.
    pub fn parse_generated_ids(
        &self,
        ids: &[u32],
        forced_language: Option<&str>,
    ) -> Result<Qwen3AsrTranscription> {
        let raw = self.decode_generated_ids(ids)?;
        parse_asr_output(&raw, forced_language)
    }

    fn push_text(&self, ids: &mut Vec<u32>, text: &str) -> Result<()> {
        let encoded = self.bpe.encode(text).map_err(|error| {
            VokraError::InvalidArgument(format!(
                "qwen3_asr tokenizer: byte-BPE encode failed: {error}"
            ))
        })?;
        ids.extend(encoded);
        Ok(())
    }

    fn flush_decoded(&self, ids: &mut Vec<u32>, output: &mut String) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        output.push_str(&self.bpe.decode(ids).map_err(|error| {
            VokraError::InvalidArgument(format!(
                "qwen3_asr tokenizer: byte-BPE decode failed: {error}"
            ))
        })?);
        ids.clear();
        Ok(())
    }
}

/// Returns whether `id` is one of the two exact EOS ids in the release.
#[must_use]
pub const fn is_eos(id: u32) -> bool {
    id == END_OF_TEXT_TOKEN_ID || id == IM_END_TOKEN_ID
}

fn normalize_language(language: &str) -> Result<String> {
    let trimmed = language.trim();
    if trimmed.is_empty() {
        return Err(VokraError::InvalidArgument(
            "qwen3_asr tokenizer: language is empty".to_owned(),
        ));
    }
    let mut chars = trimmed.chars();
    let first = chars.next().expect("non-empty checked above");
    let normalized = first
        .to_uppercase()
        .chain(chars.flat_map(char::to_lowercase))
        .collect::<String>();
    if !SUPPORTED_LANGUAGES.contains(&normalized.as_str()) {
        return Err(VokraError::InvalidArgument(format!(
            "qwen3_asr tokenizer: unsupported language {language:?}; expected one of {}",
            SUPPORTED_LANGUAGES.join(", ")
        )));
    }
    Ok(normalized)
}

fn parse_asr_output(raw: &str, forced_language: Option<&str>) -> Result<Qwen3AsrTranscription> {
    let value = raw.trim();
    if value.is_empty() {
        return Ok(Qwen3AsrTranscription {
            language: String::new(),
            text: String::new(),
        });
    }
    if let Some(language) = forced_language {
        return Ok(Qwen3AsrTranscription {
            language: normalize_language(language)?,
            text: value.to_owned(),
        });
    }
    let Some((metadata, text)) = value.split_once(ASR_TEXT_TAG) else {
        return Ok(Qwen3AsrTranscription {
            language: String::new(),
            text: value.to_owned(),
        });
    };
    if metadata.to_ascii_lowercase().contains("language none") {
        return Ok(Qwen3AsrTranscription {
            language: String::new(),
            text: text.trim().to_owned(),
        });
    }
    let mut language = String::new();
    for line in metadata
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        if line
            .get(..LANG_PREFIX.len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(LANG_PREFIX))
        {
            let candidate = line[LANG_PREFIX.len()..].trim();
            if !candidate.is_empty() {
                language = normalize_language(candidate)?;
            }
            break;
        }
    }
    Ok(Qwen3AsrTranscription {
        language,
        text: text.trim().to_owned(),
    })
}

fn read_exact_u8_array(file: &GgufFile, asset: ExactAsset) -> Result<Vec<u8>> {
    let array = match file.get(asset.key) {
        Some(GgufMetadataValue::Array(array)) => array,
        Some(other) => {
            return Err(VokraError::ModelLoad(format!(
                "qwen3_asr tokenizer: `{}` for {} is not a U8 array (got {:?})",
                asset.key,
                asset.file_name,
                other.value_type()
            )));
        }
        None => {
            return Err(VokraError::ModelLoad(format!(
                "qwen3_asr tokenizer: missing `{}` ({}); re-convert the exact release with all five tokenizer/chat/generation sidecars",
                asset.key, asset.file_name
            )));
        }
    };
    if array.values.len() != asset.bytes {
        return Err(VokraError::ModelLoad(format!(
            "qwen3_asr tokenizer: `{}` ({}) is {} bytes, expected exactly {}",
            asset.key,
            asset.file_name,
            array.values.len(),
            asset.bytes
        )));
    }
    let mut bytes = Vec::with_capacity(array.values.len());
    for value in &array.values {
        match value {
            GgufMetadataValue::U8(byte) => bytes.push(*byte),
            other => {
                return Err(VokraError::ModelLoad(format!(
                    "qwen3_asr tokenizer: `{}` ({}) contains non-U8 {:?}",
                    asset.key,
                    asset.file_name,
                    other.value_type()
                )));
            }
        }
    }
    let actual = hex_sha256(&bytes);
    if actual != asset.sha256 {
        return Err(VokraError::ModelLoad(format!(
            "qwen3_asr tokenizer: `{}` ({}) SHA-256 {actual}, expected {}",
            asset.key, asset.file_name, asset.sha256
        )));
    }
    Ok(bytes)
}

fn hex_sha256(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = sha256_bytes(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(char::from(HEX[(byte >> 4) as usize]));
        output.push(char::from(HEX[(byte & 0x0f) as usize]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufArray, GgufBuilder, GgufValueType};

    fn tiny_tokenizer() -> Qwen3AsrTokenizer {
        // No merges are needed for structural prompt tests; each ASCII byte
        // is represented by its GPT-2 byte-unicode character.
        let symbols = [
            ("a", 0),
            ("e", 1),
            ("g", 2),
            ("h", 3),
            ("i", 4),
            ("l", 5),
            ("m", 6),
            ("n", 7),
            ("r", 8),
            ("s", 9),
            ("t", 10),
            ("u", 11),
            ("y", 12),
            ("E", 13),
            ("Ġ", 14),
            ("Ċ", 15),
        ];
        let entries = symbols
            .iter()
            .map(|(token, id)| format!("\"{token}\":{id}"))
            .collect::<Vec<_>>()
            .join(",");
        let vocab = format!("{{{entries}}}");
        Qwen3AsrTokenizer {
            bpe: CosyVoice2Tokenizer::from_parts(vocab.as_bytes(), b"").expect("tiny byte BPE"),
        }
    }

    fn u8_array(bytes: &[u8]) -> GgufMetadataValue {
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U8,
            values: bytes.iter().copied().map(GgufMetadataValue::U8).collect(),
        })
    }

    #[test]
    fn prompt_has_exact_audio_span_and_forced_language_suffix() {
        let tokenizer = tiny_tokenizer();
        let ids = tokenizer
            .prompt_ids(3, Some("hint"), Some("eNGLISH"))
            .expect("prompt");
        assert_eq!(ids.first(), Some(&IM_START_TOKEN_ID));
        assert_eq!(
            ids.iter().filter(|&&id| id == AUDIO_PAD_TOKEN_ID).count(),
            3
        );
        assert_eq!(ids.last(), Some(&ASR_TEXT_TOKEN_ID));
        let audio_start = ids
            .iter()
            .position(|&id| id == AUDIO_START_TOKEN_ID)
            .expect("audio start");
        assert_eq!(
            &ids[audio_start..audio_start + 5],
            &[
                AUDIO_START_TOKEN_ID,
                AUDIO_PAD_TOKEN_ID,
                AUDIO_PAD_TOKEN_ID,
                AUDIO_PAD_TOKEN_ID,
                AUDIO_END_TOKEN_ID,
            ]
        );
    }

    #[test]
    fn generated_ids_preserve_asr_separator_and_reject_other_controls() {
        let tokenizer = tiny_tokenizer();
        assert_eq!(
            tokenizer
                .decode_generated_ids(&[3, 4, ASR_TEXT_TOKEN_ID, 3, IM_END_TOKEN_ID, 4])
                .expect("decode"),
            "hi<asr_text>h"
        );
        assert!(
            tokenizer
                .decode_generated_ids(&[IM_START_TOKEN_ID])
                .expect_err("unexpected control")
                .to_string()
                .contains("unexpected added/control")
        );
    }

    #[test]
    fn output_parser_matches_official_structural_cases() {
        assert_eq!(
            parse_asr_output("language Chinese<asr_text>你好", None).expect("metadata"),
            Qwen3AsrTranscription {
                language: "Chinese".to_owned(),
                text: "你好".to_owned(),
            }
        );
        assert_eq!(
            parse_asr_output("language None<asr_text>", None).expect("silence"),
            Qwen3AsrTranscription {
                language: String::new(),
                text: String::new(),
            }
        );
        assert_eq!(
            parse_asr_output("plain transcript", None).expect("plain"),
            Qwen3AsrTranscription {
                language: String::new(),
                text: "plain transcript".to_owned(),
            }
        );
        assert_eq!(
            parse_asr_output("forced text", Some("japanese")).expect("forced"),
            Qwen3AsrTranscription {
                language: "Japanese".to_owned(),
                text: "forced text".to_owned(),
            }
        );
    }

    #[test]
    fn exact_asset_reader_rejects_size_hash_and_type_drift() {
        let exact = ExactAsset {
            key: "test.asset",
            file_name: "test.bin",
            bytes: 3,
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        };
        let mut builder = GgufBuilder::new();
        builder.add_metadata(exact.key, u8_array(b"abc"));
        let file = GgufFile::parse(builder.to_bytes().expect("serialize")).expect("parse");
        assert_eq!(read_exact_u8_array(&file, exact).expect("exact"), b"abc");

        let short = ExactAsset { bytes: 4, ..exact };
        assert!(
            read_exact_u8_array(&file, short)
                .expect_err("size")
                .to_string()
                .contains("expected exactly 4")
        );
        let wrong_hash = ExactAsset {
            sha256: "0000000000000000000000000000000000000000000000000000000000000000",
            ..exact
        };
        assert!(
            read_exact_u8_array(&file, wrong_hash)
                .expect_err("hash")
                .to_string()
                .contains("SHA-256")
        );

        let mut wrong_type = GgufBuilder::new();
        wrong_type.add_string(exact.key, "abc");
        let wrong_type = GgufFile::parse(wrong_type.to_bytes().expect("serialize")).expect("parse");
        assert!(
            read_exact_u8_array(&wrong_type, exact)
                .expect_err("type")
                .to_string()
                .contains("not a U8 array")
        );
    }
}
