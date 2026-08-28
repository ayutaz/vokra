//! Fixed-revision Qwen2 byte-BPE and prompt contract for Qwen3-TTS.
//!
//! Executable main-model GGUFs carry the exact release `config.json`,
//! tokenizer and generation sidecars beside the weights. Every blob is
//! checked by byte length and SHA-256 before text can be tokenized. Runtime
//! downloads and mutable local sidecars are deliberately unsupported.

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::{Result, VokraError};

use crate::cosyvoice2::CosyVoice2Tokenizer;
use crate::strict_checkpoint::sha256_bytes;

use super::Qwen3TtsCheckpointVariant;

/// Exact pinned main-model revision.
pub const KEY_SOURCE_REVISION: &str = "vokra.qwen3_tts.source_revision";
/// Raw upstream `config.json` embedded as a GGUF U8 array.
pub const KEY_CONFIG_JSON: &str = "vokra.qwen3_tts.config_json";
/// Raw upstream `vocab.json` embedded as a GGUF U8 array.
pub const KEY_TOKENIZER_VOCAB: &str = "vokra.qwen3_tts.tokenizer.vocab_json";
/// Raw upstream `merges.txt` embedded as a GGUF U8 array.
pub const KEY_TOKENIZER_MERGES: &str = "vokra.qwen3_tts.tokenizer.merges_txt";
/// Raw upstream `tokenizer_config.json` embedded as a GGUF U8 array.
pub const KEY_TOKENIZER_CONFIG: &str = "vokra.qwen3_tts.tokenizer.config_json";
/// Raw upstream `generation_config.json` embedded as a GGUF U8 array.
pub const KEY_GENERATION_CONFIG: &str = "vokra.qwen3_tts.generation.config_json";

/// Number of ordinary byte-BPE entries before Qwen's added tokens.
pub const BASE_VOCAB_SIZE: usize = 151_643;
/// `<|im_start|>`.
pub const IM_START_TOKEN_ID: u32 = 151_644;
/// `<|im_end|>`.
pub const IM_END_TOKEN_ID: u32 = 151_645;
/// `<tts_pad>`.
pub const TTS_PAD_TOKEN_ID: u32 = 151_671;
/// `<tts_text_bos>`.
pub const TTS_BOS_TOKEN_ID: u32 = 151_672;
/// `<tts_text_eod>`; used as the TTS text EOS by this release.
pub const TTS_EOS_TOKEN_ID: u32 = 151_673;

/// First-codebook EOS.
pub const CODEC_EOS_TOKEN_ID: u32 = 2_150;
/// Codec padding token.
pub const CODEC_PAD_TOKEN_ID: u32 = 2_148;
/// Codec BOS token.
pub const CODEC_BOS_TOKEN_ID: u32 = 2_149;
/// Automatic-language prefill marker.
pub const CODEC_NOTHINK_TOKEN_ID: u32 = 2_155;
/// Explicit-language prefill marker.
pub const CODEC_THINK_TOKEN_ID: u32 = 2_154;
/// Language-thought BOS.
pub const CODEC_THINK_BOS_TOKEN_ID: u32 = 2_156;
/// Language-thought EOS.
pub const CODEC_THINK_EOS_TOKEN_ID: u32 = 2_157;

/// Languages accepted by every released Qwen3-TTS main checkpoint.
pub const SUPPORTED_LANGUAGES: &[&str] = &[
    "auto",
    "chinese",
    "english",
    "german",
    "italian",
    "portuguese",
    "spanish",
    "japanese",
    "korean",
    "french",
    "russian",
];

const VOCAB: ExactAsset = ExactAsset {
    key: KEY_TOKENIZER_VOCAB,
    file_name: "vocab.json",
    bytes: 2_776_833,
    sha256: "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
};
const MERGES: ExactAsset = ExactAsset {
    key: KEY_TOKENIZER_MERGES,
    file_name: "merges.txt",
    bytes: 1_671_839,
    sha256: "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
};
const TOKENIZER_CONFIG: ExactAsset = ExactAsset {
    key: KEY_TOKENIZER_CONFIG,
    file_name: "tokenizer_config.json",
    bytes: 7_344,
    sha256: "dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670",
};
const GENERATION_CONFIG: ExactAsset = ExactAsset {
    key: KEY_GENERATION_CONFIG,
    file_name: "generation_config.json",
    bytes: 245,
    sha256: "f1b90b4513f3b34c62851049e2492d7b4c5940daf1276f89c82b8ef04127f3aa",
};

#[derive(Debug, Clone, Copy)]
struct ExactAsset {
    key: &'static str,
    file_name: &'static str,
    bytes: usize,
    sha256: &'static str,
}

/// Exact authenticated Qwen3-TTS byte-BPE and release prompt vocabulary.
#[derive(Debug, Clone)]
pub struct Qwen3TtsTokenizer {
    bpe: CosyVoice2Tokenizer,
    variant: Qwen3TtsCheckpointVariant,
}

impl Qwen3TtsTokenizer {
    /// Authenticates all five embedded release sidecars and builds the BPE.
    ///
    /// Historical GGUFs without these blobs remain descriptor-bindable but
    /// cannot enter string generation.
    pub fn from_gguf(file: &GgufFile, variant: Qwen3TtsCheckpointVariant) -> Result<Self> {
        let expected_revision = source_revision(variant);
        let actual_revision = file
            .get(KEY_SOURCE_REVISION)
            .and_then(GgufMetadataValue::as_str)
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "qwen3_tts tokenizer: missing string `{KEY_SOURCE_REVISION}`; re-convert the exact release with authenticated config/tokenizer/generation sidecars"
                ))
            })?;
        if actual_revision != expected_revision {
            return Err(VokraError::ModelLoad(format!(
                "qwen3_tts tokenizer: `{KEY_SOURCE_REVISION}` is {actual_revision:?}, expected pinned {expected_revision:?} for {variant:?}"
            )));
        }

        read_exact_u8_array(file, config_asset(variant))?;
        let vocab = read_exact_u8_array(file, VOCAB)?;
        let merges = read_exact_u8_array(file, MERGES)?;
        read_exact_u8_array(file, TOKENIZER_CONFIG)?;
        read_exact_u8_array(file, GENERATION_CONFIG)?;
        let bpe = CosyVoice2Tokenizer::from_parts(&vocab, &merges).map_err(|error| {
            VokraError::ModelLoad(format!(
                "qwen3_tts tokenizer: fixed Qwen2 vocab/merges failed to parse: {error}"
            ))
        })?;
        if bpe.vocab_size() != BASE_VOCAB_SIZE {
            return Err(VokraError::ModelLoad(format!(
                "qwen3_tts tokenizer: vocab.json has {} entries, expected exactly {BASE_VOCAB_SIZE}",
                bpe.vocab_size()
            )));
        }
        Ok(Self { bpe, variant })
    }

    /// Exact release variant whose sidecars were authenticated.
    #[must_use]
    pub const fn variant(&self) -> Qwen3TtsCheckpointVariant {
        self.variant
    }

    /// Tokenizes the exact assistant wrapper consumed by Qwen3-TTS.
    pub fn assistant_ids(&self, text: &str) -> Result<Vec<u32>> {
        reject_empty_or_control("text", text)?;
        let mut ids = Vec::with_capacity(text.len().saturating_add(16));
        ids.push(IM_START_TOKEN_ID);
        self.push_text(&mut ids, "assistant\n")?;
        self.push_text(&mut ids, text)?;
        ids.push(IM_END_TOKEN_ID);
        self.push_text(&mut ids, "\n")?;
        ids.push(IM_START_TOKEN_ID);
        self.push_text(&mut ids, "assistant\n")?;
        Ok(ids)
    }

    /// Tokenizes the exact optional user instruction wrapper.
    pub fn instruction_ids(&self, instruction: &str) -> Result<Vec<u32>> {
        reject_empty_or_control("instruction", instruction)?;
        let mut ids = Vec::with_capacity(instruction.len().saturating_add(8));
        ids.push(IM_START_TOKEN_ID);
        self.push_text(&mut ids, "user\n")?;
        self.push_text(&mut ids, instruction)?;
        ids.push(IM_END_TOKEN_ID);
        self.push_text(&mut ids, "\n")?;
        Ok(ids)
    }

    /// Tokenizes the exact reference-transcript wrapper used by Base ICL.
    pub fn reference_ids(&self, text: &str) -> Result<Vec<u32>> {
        reject_empty_or_control("reference text", text)?;
        let mut ids = Vec::with_capacity(text.len().saturating_add(8));
        ids.push(IM_START_TOKEN_ID);
        self.push_text(&mut ids, "assistant\n")?;
        self.push_text(&mut ids, text)?;
        ids.push(IM_END_TOKEN_ID);
        self.push_text(&mut ids, "\n")?;
        Ok(ids)
    }

    /// Resolves an official language name to its codec prefill id. `auto`
    /// intentionally returns `None` and uses the no-think prefill.
    pub fn language_id(&self, language: &str) -> Result<Option<u32>> {
        let normalized = language.trim().to_ascii_lowercase();
        if normalized == "auto" {
            return Ok(None);
        }
        language_id(&normalized).map(Some).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "qwen3_tts tokenizer: unsupported language {language:?}; expected one of {}",
                SUPPORTED_LANGUAGES.join(", ")
            ))
        })
    }

    /// Resolves a fixed CustomVoice speaker name. Base and VoiceDesign have
    /// no fixed-speaker table and reject this call explicitly.
    pub fn speaker_id(&self, speaker: &str) -> Result<u32> {
        if !matches!(
            self.variant,
            Qwen3TtsCheckpointVariant::CustomVoice0_6B | Qwen3TtsCheckpointVariant::CustomVoice1_7B
        ) {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts tokenizer: {variant:?} has no fixed speaker ids",
                variant = self.variant
            )));
        }
        let normalized = speaker.trim().to_ascii_lowercase();
        speaker_id(&normalized).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "qwen3_tts tokenizer: unsupported CustomVoice speaker {speaker:?}; expected one of serena, vivian, uncle_fu, ryan, aiden, ono_anna, sohee, eric, dylan"
            ))
        })
    }

    /// Applies the official Chinese/auto dialect override for Eric and Dylan.
    pub fn language_id_for_speaker(
        &self,
        language: &str,
        speaker: Option<&str>,
    ) -> Result<Option<u32>> {
        let normalized_language = language.trim().to_ascii_lowercase();
        let mut id = self.language_id(&normalized_language)?;
        if let Some(speaker) = speaker {
            // Validate the fixed-speaker contract even when no dialect
            // override applies. Base and VoiceDesign must never silently
            // accept a CustomVoice-only speaker name.
            self.speaker_id(speaker)?;
        }
        if matches!(normalized_language.as_str(), "chinese" | "auto")
            && let Some(speaker) = speaker
        {
            match speaker.trim().to_ascii_lowercase().as_str() {
                "eric" => id = language_id("sichuan_dialect"),
                "dylan" => id = language_id("beijing_dialect"),
                _ => {}
            }
        }
        Ok(id)
    }

    fn push_text(&self, ids: &mut Vec<u32>, text: &str) -> Result<()> {
        let encoded = self.bpe.encode(text).map_err(|error| {
            VokraError::InvalidArgument(format!(
                "qwen3_tts tokenizer: byte-BPE encode failed: {error}"
            ))
        })?;
        ids.extend(encoded);
        Ok(())
    }
}

const fn source_revision(variant: Qwen3TtsCheckpointVariant) -> &'static str {
    match variant {
        Qwen3TtsCheckpointVariant::Base0_6B => "5d83992436eae1d760afd27aff78a71d676296fc",
        Qwen3TtsCheckpointVariant::CustomVoice0_6B => "85e237c12c027371202489a0ec509ded67b5e4b5",
        Qwen3TtsCheckpointVariant::Base1_7B => "fd4b254389122332181a7c3db7f27e918eec64e3",
        Qwen3TtsCheckpointVariant::CustomVoice1_7B => "0c0e3051f131929182e2c023b9537f8b1c68adfe",
        Qwen3TtsCheckpointVariant::VoiceDesign1_7B => "5ecdb67327fd37bb2e042aab12ff7391903235d3",
    }
}

const fn config_asset(variant: Qwen3TtsCheckpointVariant) -> ExactAsset {
    match variant {
        Qwen3TtsCheckpointVariant::Base0_6B => ExactAsset {
            key: KEY_CONFIG_JSON,
            file_name: "config.json",
            bytes: 4_494,
            sha256: "2e714c787c8edb98b05432685cddb634add2de4d4e645f653d68251ef72ba011",
        },
        Qwen3TtsCheckpointVariant::CustomVoice0_6B => ExactAsset {
            key: KEY_CONFIG_JSON,
            file_name: "config.json",
            bytes: 4_908,
            sha256: "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455",
        },
        Qwen3TtsCheckpointVariant::Base1_7B => ExactAsset {
            key: KEY_CONFIG_JSON,
            file_name: "config.json",
            bytes: 4_494,
            sha256: "b4f01752d15a488abde3e1ab44723ae4f4b9e68a4037257b098b3737893cc1f9",
        },
        Qwen3TtsCheckpointVariant::CustomVoice1_7B => ExactAsset {
            key: KEY_CONFIG_JSON,
            file_name: "config.json",
            bytes: 4_908,
            sha256: "17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9",
        },
        Qwen3TtsCheckpointVariant::VoiceDesign1_7B => ExactAsset {
            key: KEY_CONFIG_JSON,
            file_name: "config.json",
            bytes: 4_421,
            sha256: "aecd2cc4c1fe9edef1cb7ca7c401685a43879ad43f3f9e883f1c6760b61731e0",
        },
    }
}

fn language_id(language: &str) -> Option<u32> {
    match language {
        "english" => Some(2_050),
        "german" => Some(2_053),
        "spanish" => Some(2_054),
        "chinese" => Some(2_055),
        "japanese" => Some(2_058),
        "french" => Some(2_061),
        "sichuan_dialect" => Some(2_062),
        "korean" => Some(2_064),
        "russian" => Some(2_069),
        "italian" => Some(2_070),
        "portuguese" => Some(2_071),
        "beijing_dialect" => Some(2_074),
        _ => None,
    }
}

fn speaker_id(speaker: &str) -> Option<u32> {
    match speaker {
        "aiden" => Some(2_861),
        "sohee" => Some(2_864),
        "ono_anna" => Some(2_873),
        "eric" => Some(2_875),
        "dylan" => Some(2_878),
        "uncle_fu" => Some(3_010),
        "ryan" => Some(3_061),
        "vivian" => Some(3_065),
        "serena" => Some(3_066),
        _ => None,
    }
}

fn reject_empty_or_control(label: &str, value: &str) -> Result<()> {
    if value.is_empty() {
        return Err(VokraError::InvalidArgument(format!(
            "qwen3_tts tokenizer: {label} is empty"
        )));
    }
    if value.contains("<|") || value.contains("<tts_") {
        return Err(VokraError::InvalidArgument(format!(
            "qwen3_tts tokenizer: {label} contains a reserved control-token spelling"
        )));
    }
    Ok(())
}

fn read_exact_u8_array(file: &GgufFile, asset: ExactAsset) -> Result<Vec<u8>> {
    let array = match file.get(asset.key) {
        Some(GgufMetadataValue::Array(array)) => array,
        Some(other) => {
            return Err(VokraError::ModelLoad(format!(
                "qwen3_tts tokenizer: `{}` for {} is not a U8 array (got {:?})",
                asset.key,
                asset.file_name,
                other.value_type()
            )));
        }
        None => {
            return Err(VokraError::ModelLoad(format!(
                "qwen3_tts tokenizer: missing `{}` ({}); re-convert the exact release with all five config/tokenizer/generation sidecars",
                asset.key, asset.file_name
            )));
        }
    };
    if array.values.len() != asset.bytes {
        return Err(VokraError::ModelLoad(format!(
            "qwen3_tts tokenizer: `{}` ({}) is {} bytes, expected exactly {}",
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
                    "qwen3_tts tokenizer: `{}` ({}) contains non-U8 {:?}",
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
            "qwen3_tts tokenizer: `{}` ({}) SHA-256 {actual}, expected {}",
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

    fn tiny_tokenizer(variant: Qwen3TtsCheckpointVariant) -> Qwen3TtsTokenizer {
        let symbols = [
            ("a", 0),
            ("e", 1),
            ("h", 2),
            ("i", 3),
            ("n", 4),
            ("r", 5),
            ("s", 6),
            ("t", 7),
            ("u", 8),
            ("Ċ", 9),
        ];
        let entries = symbols
            .iter()
            .map(|(token, id)| format!("\"{token}\":{id}"))
            .collect::<Vec<_>>()
            .join(",");
        let vocab = format!("{{{entries}}}");
        Qwen3TtsTokenizer {
            bpe: CosyVoice2Tokenizer::from_parts(vocab.as_bytes(), b"").expect("tiny byte BPE"),
            variant,
        }
    }

    fn u8_array(bytes: &[u8]) -> GgufMetadataValue {
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U8,
            values: bytes.iter().copied().map(GgufMetadataValue::U8).collect(),
        })
    }

    #[test]
    fn assistant_and_instruction_wrappers_pin_control_positions() {
        let tokenizer = tiny_tokenizer(Qwen3TtsCheckpointVariant::CustomVoice0_6B);
        let assistant = tokenizer.assistant_ids("hi").expect("assistant");
        assert_eq!(assistant.first(), Some(&IM_START_TOKEN_ID));
        assert_eq!(
            assistant
                .iter()
                .filter(|&&id| id == IM_START_TOKEN_ID)
                .count(),
            2
        );
        assert_eq!(
            assistant
                .iter()
                .filter(|&&id| id == IM_END_TOKEN_ID)
                .count(),
            1
        );
        let instruction = tokenizer.instruction_ids("hi").expect("instruction");
        assert_eq!(instruction.first(), Some(&IM_START_TOKEN_ID));
        assert_eq!(
            instruction
                .iter()
                .filter(|&&id| id == IM_END_TOKEN_ID)
                .count(),
            1
        );
        assert!(
            tokenizer
                .assistant_ids("<|im_end|>")
                .expect_err("control injection")
                .to_string()
                .contains("reserved control-token")
        );
    }

    #[test]
    fn languages_speakers_and_dialects_match_release_config() {
        let custom = tiny_tokenizer(Qwen3TtsCheckpointVariant::CustomVoice1_7B);
        assert_eq!(custom.language_id("Auto").unwrap(), None);
        assert_eq!(custom.language_id("Japanese").unwrap(), Some(2_058));
        assert_eq!(custom.speaker_id("Serena").unwrap(), 3_066);
        assert_eq!(
            custom
                .language_id_for_speaker("Chinese", Some("Eric"))
                .unwrap(),
            Some(2_062)
        );
        assert_eq!(
            custom
                .language_id_for_speaker("Auto", Some("Dylan"))
                .unwrap(),
            Some(2_074)
        );
        let base = tiny_tokenizer(Qwen3TtsCheckpointVariant::Base0_6B);
        assert!(base.speaker_id("Serena").is_err());
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

        assert!(
            read_exact_u8_array(&file, ExactAsset { bytes: 4, ..exact })
                .expect_err("size")
                .to_string()
                .contains("expected exactly 4")
        );
        assert!(
            read_exact_u8_array(
                &file,
                ExactAsset {
                    sha256: "0000000000000000000000000000000000000000000000000000000000000000",
                    ..exact
                }
            )
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
