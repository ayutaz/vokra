//! Fixed-revision Qwen2 byte-BPE and prompt contract for MOSS-Audio.
//!
//! Executable corrected GGUFs carry six exact sidecars from the same immutable
//! 4B/8B release as the weights. The runtime authenticates every byte before
//! exposing the BPE, official default prompt, two-second time markers or
//! generated-text decoder. Historical public GGUFs remain usable through the
//! token-level API only; no mutable local tokenizer is consulted.

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::{Result, VokraError};

use crate::cosyvoice2::CosyVoice2Tokenizer;
use crate::strict_checkpoint::sha256_bytes;

use super::MossAudioVariant;

/// Raw upstream `vocab.json` embedded as a GGUF U8 array.
pub const KEY_TOKENIZER_VOCAB: &str = "vokra.moss_audio.tokenizer.vocab_json";
/// Raw upstream `merges.txt` embedded as a GGUF U8 array.
pub const KEY_TOKENIZER_MERGES: &str = "vokra.moss_audio.tokenizer.merges_txt";
/// Raw upstream `tokenizer_config.json` embedded as a GGUF U8 array.
pub const KEY_TOKENIZER_CONFIG: &str = "vokra.moss_audio.tokenizer.config_json";
/// Raw upstream `chat_template.jinja` embedded as a GGUF U8 array.
pub const KEY_CHAT_TEMPLATE: &str = "vokra.moss_audio.tokenizer.chat_template_jinja";
/// Raw upstream `generation_config.json` embedded as a GGUF U8 array.
pub const KEY_GENERATION_CONFIG: &str = "vokra.moss_audio.generation.config_json";
/// Raw upstream `processor_config.json` embedded as a GGUF U8 array.
pub const KEY_PROCESSOR_CONFIG: &str = "vokra.moss_audio.processor.config_json";

/// Number of ordinary byte-BPE entries before Qwen's added tokens.
pub const BASE_VOCAB_SIZE: usize = 151_643;
/// `<|endoftext|>`; the released padding token.
pub const END_OF_TEXT_TOKEN_ID: u32 = 151_643;
/// `<|im_start|>`.
pub const IM_START_TOKEN_ID: u32 = 151_644;
/// `<|im_end|>` and the released generation EOS.
pub const IM_END_TOKEN_ID: u32 = 151_645;
/// `<|AUDIO|>` / `<|vision_pad|>` audio replacement row.
pub const AUDIO_TOKEN_ID: u32 = 151_654;
/// Processor alias `<|audio_bos|>`.
pub const AUDIO_START_TOKEN_ID: u32 = 151_669;
/// Processor alias `<|audio_eos|>`.
pub const AUDIO_END_TOKEN_ID: u32 = 151_670;

const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";
/// Prompt used by the fixed-revision official `infer.py` example.
pub const DEFAULT_USER_PROMPT: &str = "Describe this audio.";
const AUDIO_TOKENS_PER_TIME_MARKER: usize = 25;
const SECONDS_PER_TIME_MARKER: usize = 2;
const DIGIT_TOKEN_IDS: [u32; 10] = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24];

const VOCAB: ExactAsset = ExactAsset {
    key: KEY_TOKENIZER_VOCAB,
    file_name: "vocab.json",
    b4: AssetIdentity {
        bytes: 3_383_407,
        sha256: "87a257b04b17642a0688c98cd1df89c398bda4fee532d6f88b38a659ecb4ac8d",
    },
    b8: AssetIdentity {
        bytes: 3_383_407,
        sha256: "87a257b04b17642a0688c98cd1df89c398bda4fee532d6f88b38a659ecb4ac8d",
    },
};
const MERGES: ExactAsset = ExactAsset {
    key: KEY_TOKENIZER_MERGES,
    file_name: "merges.txt",
    b4: AssetIdentity {
        bytes: 1_671_853,
        sha256: "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    },
    b8: AssetIdentity {
        bytes: 1_671_853,
        sha256: "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    },
};
const TOKENIZER_CONFIG: ExactAsset = ExactAsset {
    key: KEY_TOKENIZER_CONFIG,
    file_name: "tokenizer_config.json",
    b4: AssetIdentity {
        bytes: 5_404,
        sha256: "443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0",
    },
    b8: AssetIdentity {
        bytes: 6_114,
        sha256: "0869e41f5d123ff144a811f0d83c5d18871dcd4b4064f46bf9def194bfbc6f41",
    },
};
const CHAT_TEMPLATE: ExactAsset = ExactAsset {
    key: KEY_CHAT_TEMPLATE,
    file_name: "chat_template.jinja",
    b4: AssetIdentity {
        bytes: 4_116,
        sha256: "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5",
    },
    b8: AssetIdentity {
        bytes: 4_116,
        sha256: "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5",
    },
};
const GENERATION_CONFIG: ExactAsset = ExactAsset {
    key: KEY_GENERATION_CONFIG,
    file_name: "generation_config.json",
    b4: AssetIdentity {
        bytes: 121,
        sha256: "bb52bfdd308deaea4ec800bf0165e75770b0a4e5c105963bee1b0398f4043d3e",
    },
    b8: AssetIdentity {
        bytes: 121,
        sha256: "bb52bfdd308deaea4ec800bf0165e75770b0a4e5c105963bee1b0398f4043d3e",
    },
};
const PROCESSOR_CONFIG: ExactAsset = ExactAsset {
    key: KEY_PROCESSOR_CONFIG,
    file_name: "processor_config.json",
    b4: AssetIdentity {
        bytes: 426,
        sha256: "0749d81701d2a2a2e83ca4d549fbebb1a205acac1ac7bdccea7965c1913b2cbf",
    },
    b8: AssetIdentity {
        bytes: 427,
        sha256: "6a5c462858acb299db0d2d967b63d520b72d178f44d1619c33fc860f25fdccbf",
    },
};

#[derive(Debug, Clone, Copy)]
struct AssetIdentity {
    bytes: usize,
    sha256: &'static str,
}

#[derive(Debug, Clone, Copy)]
struct ExactAsset {
    key: &'static str,
    file_name: &'static str,
    b4: AssetIdentity,
    b8: AssetIdentity,
}

impl ExactAsset {
    const fn identity(self, variant: MossAudioVariant) -> AssetIdentity {
        match variant {
            MossAudioVariant::B4Instruct => self.b4,
            MossAudioVariant::B8Instruct => self.b8,
        }
    }
}

/// Exact fixed-revision MOSS-Audio byte-BPE and processor prompt contract.
#[derive(Debug, Clone)]
pub struct MossAudioTextTokenizer {
    bpe: CosyVoice2Tokenizer,
    variant: MossAudioVariant,
}

impl MossAudioTextTokenizer {
    /// Authenticates all six embedded release sidecars and builds the BPE.
    pub fn from_gguf(file: &GgufFile, variant: MossAudioVariant) -> Result<Self> {
        let vocab = read_exact_u8_array(file, VOCAB, variant)?;
        let merges = read_exact_u8_array(file, MERGES, variant)?;
        // These exact bytes pin special-token ids, ChatML formatting,
        // generation EOS and time-marker/audio aliases. Execution below uses
        // audited constants so no JSON/Jinja interpreter enters the runtime.
        read_exact_u8_array(file, TOKENIZER_CONFIG, variant)?;
        read_exact_u8_array(file, CHAT_TEMPLATE, variant)?;
        read_exact_u8_array(file, GENERATION_CONFIG, variant)?;
        read_exact_u8_array(file, PROCESSOR_CONFIG, variant)?;

        let bpe = CosyVoice2Tokenizer::from_parts(&vocab, &merges).map_err(|error| {
            VokraError::ModelLoad(format!(
                "moss_audio tokenizer: fixed Qwen2 vocab/merges failed to parse: {error}"
            ))
        })?;
        if bpe.vocab_size() != BASE_VOCAB_SIZE {
            return Err(VokraError::ModelLoad(format!(
                "moss_audio tokenizer: vocab.json has {} entries, expected exactly {BASE_VOCAB_SIZE}",
                bpe.vocab_size()
            )));
        }
        Ok(Self { bpe, variant })
    }

    /// Produces the official processor's default one-audio ChatML sequence.
    ///
    /// The fixed processor inserts one audio replacement id per encoder row
    /// and, because `enable_time_marker=true`, appends the decimal second at
    /// every 25 rows (12.5 audio rows/s × 2 seconds).
    pub fn prompt_ids(&self, audio_frames: usize, text: &str) -> Result<Vec<u32>> {
        if audio_frames == 0 {
            return Err(VokraError::InvalidArgument(
                "moss_audio tokenizer: audio_frames must be greater than zero".to_owned(),
            ));
        }
        reject_reserved_prompt_text(text)?;
        let marker_digits = (1..=audio_frames / AUDIO_TOKENS_PER_TIME_MARKER)
            .map(|marker| (marker * SECONDS_PER_TIME_MARKER).to_string().len())
            .sum::<usize>();
        let mut ids = Vec::with_capacity(
            audio_frames
                .saturating_add(marker_digits)
                .saturating_add(64),
        );
        ids.push(IM_START_TOKEN_ID);
        self.push_text(&mut ids, "system\n")?;
        self.push_text(&mut ids, DEFAULT_SYSTEM_PROMPT)?;
        ids.push(IM_END_TOKEN_ID);
        self.push_text(&mut ids, "\n")?;
        ids.push(IM_START_TOKEN_ID);
        self.push_text(&mut ids, "user\n")?;
        ids.push(AUDIO_START_TOKEN_ID);
        push_audio_placeholders(&mut ids, audio_frames);
        ids.push(AUDIO_END_TOKEN_ID);
        self.push_text(&mut ids, "\n")?;
        self.push_text(&mut ids, text)?;
        ids.push(IM_END_TOKEN_ID);
        self.push_text(&mut ids, "\n")?;
        ids.push(IM_START_TOKEN_ID);
        self.push_text(&mut ids, "assistant\n")?;
        Ok(ids)
    }

    /// Decodes generated ids like the official `skip_special_tokens=true`
    /// path, stopping at `<|im_end|>` and preserving non-special Qwen tags.
    pub fn decode_generated_ids(&self, ids: &[u32]) -> Result<String> {
        let mut decoded = String::new();
        let mut ordinary = Vec::new();
        for &id in ids {
            if id == IM_END_TOKEN_ID {
                break;
            }
            if (id as usize) < BASE_VOCAB_SIZE {
                ordinary.push(id);
                continue;
            }
            self.flush_decoded(&mut ordinary, &mut decoded)?;
            if is_skipped_special(id) {
                continue;
            }
            if let Some(token) = non_special_added_token(self.variant, id) {
                decoded.push_str(token);
                continue;
            }
            return Err(VokraError::InvalidArgument(format!(
                "moss_audio tokenizer: generated unexpected unmapped token id {id}; corrected releases admit base BPE, authenticated non-special added tokens and EOS only"
            )));
        }
        self.flush_decoded(&mut ordinary, &mut decoded)?;
        Ok(decoded)
    }

    fn push_text(&self, ids: &mut Vec<u32>, text: &str) -> Result<()> {
        ids.extend(self.bpe.encode(text).map_err(|error| {
            VokraError::InvalidArgument(format!(
                "moss_audio tokenizer: byte-BPE encode failed: {error}"
            ))
        })?);
        Ok(())
    }

    fn flush_decoded(&self, ids: &mut Vec<u32>, output: &mut String) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        output.push_str(&self.bpe.decode(ids).map_err(|error| {
            VokraError::InvalidArgument(format!(
                "moss_audio tokenizer: byte-BPE decode failed: {error}"
            ))
        })?);
        ids.clear();
        Ok(())
    }
}

fn reject_reserved_prompt_text(text: &str) -> Result<()> {
    if text.contains("<|") || text.contains("<think>") || text.contains("</think>") {
        return Err(VokraError::InvalidArgument(
            "moss_audio tokenizer: prompt contains an upstream reserved-token spelling; the safe one-audio string route does not permit control-token injection"
                .to_owned(),
        ));
    }
    Ok(())
}

fn push_audio_placeholders(ids: &mut Vec<u32>, audio_frames: usize) {
    for frame in 1..=audio_frames {
        ids.push(AUDIO_TOKEN_ID);
        if frame % AUDIO_TOKENS_PER_TIME_MARKER == 0 {
            let seconds = frame / AUDIO_TOKENS_PER_TIME_MARKER * SECONDS_PER_TIME_MARKER;
            for digit in seconds.to_string().bytes() {
                ids.push(DIGIT_TOKEN_IDS[(digit - b'0') as usize]);
            }
        }
    }
}

const fn is_skipped_special(id: u32) -> bool {
    id == END_OF_TEXT_TOKEN_ID || (id >= IM_START_TOKEN_ID && id <= 151_656)
}

const fn non_special_added_token(variant: MossAudioVariant, id: u32) -> Option<&'static str> {
    match id {
        151_657 => Some("<tool_call>"),
        151_658 => Some("</tool_call>"),
        151_659 => Some("<|fim_prefix|>"),
        151_660 => Some("<|fim_middle|>"),
        151_661 => Some("<|fim_suffix|>"),
        151_662 => Some("<|fim_pad|>"),
        151_663 => Some("<|repo_name|>"),
        151_664 => Some("<|file_sep|>"),
        151_665 => Some("<tool_response>"),
        151_666 => Some("</tool_response>"),
        151_667 => Some("<think>"),
        151_668 => Some("</think>"),
        151_669 if matches!(variant, MossAudioVariant::B8Instruct) => Some("<|system|>"),
        151_670 if matches!(variant, MossAudioVariant::B8Instruct) => Some("<|user|>"),
        151_671 if matches!(variant, MossAudioVariant::B8Instruct) => Some("<|assistant|>"),
        151_672 if matches!(variant, MossAudioVariant::B8Instruct) => Some("<|eot|>"),
        _ => None,
    }
}

fn read_exact_u8_array(
    file: &GgufFile,
    asset: ExactAsset,
    variant: MossAudioVariant,
) -> Result<Vec<u8>> {
    let identity = asset.identity(variant);
    let array = match file.get(asset.key) {
        Some(GgufMetadataValue::Array(array)) => array,
        Some(other) => {
            return Err(VokraError::ModelLoad(format!(
                "moss_audio tokenizer: `{}` for {} is not a U8 array (got {:?})",
                asset.key,
                asset.file_name,
                other.value_type()
            )));
        }
        None => {
            return Err(VokraError::ModelLoad(format!(
                "moss_audio tokenizer: missing `{}` ({}); re-convert the exact {} release with all six tokenizer/chat/generation/processor sidecars",
                asset.key,
                asset.file_name,
                variant.model_name()
            )));
        }
    };
    if array.values.len() != identity.bytes {
        return Err(VokraError::ModelLoad(format!(
            "moss_audio tokenizer: `{}` ({}) is {} bytes, expected exactly {} for {}",
            asset.key,
            asset.file_name,
            array.values.len(),
            identity.bytes,
            variant.model_name()
        )));
    }
    let mut bytes = Vec::with_capacity(array.values.len());
    for value in &array.values {
        match value {
            GgufMetadataValue::U8(byte) => bytes.push(*byte),
            other => {
                return Err(VokraError::ModelLoad(format!(
                    "moss_audio tokenizer: `{}` ({}) contains non-U8 {:?}",
                    asset.key,
                    asset.file_name,
                    other.value_type()
                )));
            }
        }
    }
    let actual = hex_sha256(&bytes);
    if actual != identity.sha256 {
        return Err(VokraError::ModelLoad(format!(
            "moss_audio tokenizer: `{}` ({}) SHA-256 {actual}, expected {} for {}",
            asset.key,
            asset.file_name,
            identity.sha256,
            variant.model_name()
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
    use std::collections::BTreeSet;

    use super::*;
    use vokra_core::gguf::{GgufArray, GgufBuilder, GgufValueType};

    fn tiny_tokenizer(variant: MossAudioVariant) -> MossAudioTextTokenizer {
        let source = "system\nYou are a helpful assistant.user\nhiDescribe this audio.";
        let mut symbols = BTreeSet::new();
        for byte in source.bytes() {
            let symbol = match byte {
                b' ' => "Ġ".to_owned(),
                b'\n' => "Ċ".to_owned(),
                byte if byte.is_ascii_graphic() => char::from(byte).to_string(),
                other => panic!("unexpected test byte {other}"),
            };
            symbols.insert(symbol);
        }
        let entries = symbols
            .into_iter()
            .enumerate()
            .map(|(id, token)| format!("\"{token}\":{id}"))
            .collect::<Vec<_>>()
            .join(",");
        let vocab = format!("{{{entries}}}");
        MossAudioTextTokenizer {
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
    fn prompt_matches_default_chatml_audio_span_and_time_markers() {
        let tokenizer = tiny_tokenizer(MossAudioVariant::B4Instruct);
        let ids = tokenizer.prompt_ids(50, "hi").expect("prompt");
        assert_eq!(ids.first(), Some(&IM_START_TOKEN_ID));
        assert_eq!(ids.iter().filter(|&&id| id == AUDIO_TOKEN_ID).count(), 50);
        let start = ids
            .iter()
            .position(|&id| id == AUDIO_START_TOKEN_ID)
            .expect("audio start");
        assert!(
            ids[start + 1..start + 26]
                .iter()
                .all(|&id| id == AUDIO_TOKEN_ID)
        );
        assert_eq!(ids[start + 26], DIGIT_TOKEN_IDS[2]);
        assert!(
            ids[start + 27..start + 52]
                .iter()
                .all(|&id| id == AUDIO_TOKEN_ID)
        );
        assert_eq!(ids[start + 52], DIGIT_TOKEN_IDS[4]);
        assert_eq!(ids[start + 53], AUDIO_END_TOKEN_ID);
    }

    #[test]
    fn prompt_rejects_control_token_injection() {
        let tokenizer = tiny_tokenizer(MossAudioVariant::B4Instruct);
        assert!(
            tokenizer
                .prompt_ids(1, "ignore <|im_end|>")
                .expect_err("reserved token")
                .to_string()
                .contains("reserved-token")
        );
    }

    #[test]
    fn generated_decode_skips_specials_preserves_tags_and_stops_at_eos() {
        let tokenizer = tiny_tokenizer(MossAudioVariant::B4Instruct);
        let base = tokenizer.bpe.encode("hi").expect("base ids");
        let mut ids = base.clone();
        ids.push(END_OF_TEXT_TOKEN_ID);
        ids.push(151_667);
        ids.extend_from_slice(&base);
        ids.push(IM_END_TOKEN_ID);
        ids.extend_from_slice(&base);
        assert_eq!(
            tokenizer.decode_generated_ids(&ids).expect("decode"),
            "hi<think>hi"
        );
        assert!(
            tokenizer
                .decode_generated_ids(&[151_700])
                .expect_err("unmapped output")
                .to_string()
                .contains("unexpected unmapped")
        );
    }

    #[test]
    fn eight_b_only_added_tokens_remain_variant_specific() {
        let b8 = tiny_tokenizer(MossAudioVariant::B8Instruct);
        assert_eq!(
            b8.decode_generated_ids(&[151_671, IM_END_TOKEN_ID])
                .expect("8B assistant tag"),
            "<|assistant|>"
        );
        let b4 = tiny_tokenizer(MossAudioVariant::B4Instruct);
        assert!(b4.decode_generated_ids(&[151_671]).is_err());
    }

    #[test]
    fn exact_asset_reader_rejects_size_hash_and_type_drift() {
        let exact = ExactAsset {
            key: "test.asset",
            file_name: "test.bin",
            b4: AssetIdentity {
                bytes: 3,
                sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            },
            b8: AssetIdentity {
                bytes: 3,
                sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            },
        };
        let mut builder = GgufBuilder::new();
        builder.add_metadata(exact.key, u8_array(b"abc"));
        let file = GgufFile::parse(builder.to_bytes().expect("serialize")).expect("parse");
        assert_eq!(
            read_exact_u8_array(&file, exact, MossAudioVariant::B4Instruct).expect("exact"),
            b"abc"
        );

        let short = ExactAsset {
            b4: AssetIdentity {
                bytes: 4,
                ..exact.b4
            },
            ..exact
        };
        assert!(
            read_exact_u8_array(&file, short, MossAudioVariant::B4Instruct)
                .expect_err("size")
                .to_string()
                .contains("expected exactly 4")
        );
        let wrong_hash = ExactAsset {
            b4: AssetIdentity {
                sha256: "0000000000000000000000000000000000000000000000000000000000000000",
                ..exact.b4
            },
            ..exact
        };
        assert!(
            read_exact_u8_array(&file, wrong_hash, MossAudioVariant::B4Instruct)
                .expect_err("hash")
                .to_string()
                .contains("SHA-256")
        );

        let mut wrong_type = GgufBuilder::new();
        wrong_type.add_string(exact.key, "abc");
        let wrong_type = GgufFile::parse(wrong_type.to_bytes().expect("serialize")).expect("parse");
        assert!(
            read_exact_u8_array(&wrong_type, exact, MossAudioVariant::B4Instruct)
                .expect_err("type")
                .to_string()
                .contains("not a U8 array")
        );
    }
}
