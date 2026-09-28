//! Exact Qwen2 text-tokenizer contract for VibeVoice Realtime.
//!
//! The tokenizer is a companion sidecar, not a model weight.  This module
//! accepts only the byte-identical files authenticated by the fixed Realtime
//! packet and delegates byte-level BPE execution to the existing first-party
//! [`CosyVoice2Tokenizer`].  The speech boundary ids are taken from the
//! pinned upstream VibeVoice text-tokenizer source; no tokenizer library or
//! mutable runtime download is used here.

use std::path::Path;

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::{Result, VokraError};

use crate::cosyvoice2::CosyVoice2Tokenizer;
use crate::strict_checkpoint::sha256_bytes;

/// Fixed Qwen tokenizer repository used by the Realtime release.
pub const TOKENIZER_REPOSITORY: &str = "Qwen/Qwen2.5-0.5B";
/// Fixed Qwen tokenizer revision authenticated by the VAST packet.
pub const TOKENIZER_REVISION: &str = "060db6499f32faf8b98477b0a26969ef7d8b9987";
/// Fixed VibeVoice source revision containing the text-tokenizer boundary.
pub const SOURCE_REVISION: &str = "94da20d98b2fa7688e9cbfaf7692ddb4954f7600";
/// Fixed source file whose exact blob defines the speech boundary properties.
pub const SOURCE_TOKENIZER_FILE: &str = "vibevoice/modular/modular_vibevoice_text_tokenizer.py";
/// Git blob identity of [`SOURCE_TOKENIZER_FILE`].
pub const SOURCE_TOKENIZER_GIT_BLOB_SHA1: &str = "9532d9ffe7120eb47b18c52c0a23db9e2d4e3bbf";

/// GGUF key for the fixed Qwen repository revision.
pub const KEY_TOKENIZER_REVISION: &str = "vokra.vibevoice.tokenizer.revision";
/// GGUF key for raw `vocab.json` bytes.
pub const KEY_TOKENIZER_VOCAB: &str = "vokra.vibevoice.tokenizer.vocab_json";
/// GGUF key for raw `merges.txt` bytes.
pub const KEY_TOKENIZER_MERGES: &str = "vokra.vibevoice.tokenizer.merges_txt";
/// GGUF key for raw `tokenizer_config.json` bytes.
pub const KEY_TOKENIZER_CONFIG: &str = "vokra.vibevoice.tokenizer.config_json";
/// GGUF key for raw `tokenizer.json` bytes.
pub const KEY_TOKENIZER_JSON: &str = "vokra.vibevoice.tokenizer.tokenizer_json";

/// Number of ordinary Qwen2 BPE entries in `vocab.json`.
pub const BASE_VOCAB_SIZE: usize = 151_643;
/// Full Qwen2 vocabulary width after added tokens.
pub const VOCAB_SIZE: usize = 151_936;
/// `<|vision_start|>`; VibeVoice's speech-start boundary.
pub const SPEECH_START_ID: u32 = 151_652;
/// `<|vision_end|>`; VibeVoice's speech-end boundary.
pub const SPEECH_END_ID: u32 = 151_653;
/// `<|vision_pad|>`; VibeVoice's speech-diffusion boundary.
pub const SPEECH_PAD_ID: u32 = 151_654;

const SPEECH_START_TOKEN: &str = "<|vision_start|>";
const SPEECH_END_TOKEN: &str = "<|vision_end|>";
const SPEECH_PAD_TOKEN: &str = "<|vision_pad|>";

const VOCAB_BYTES: usize = 2_776_833;
const VOCAB_SHA256: &str = "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910";
const MERGES_BYTES: usize = 1_671_839;
const MERGES_SHA256: &str = "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3";
const CONFIG_BYTES: usize = 7_228;
const CONFIG_SHA256: &str = "c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9";
const TOKENIZER_JSON_BYTES: usize = 7_031_645;
const TOKENIZER_JSON_SHA256: &str =
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539";

/// The authenticated VibeVoice Realtime text-tokenizer primitive.
#[derive(Debug, Clone)]
pub struct VibeVoiceRealtimeTokenizer {
    bpe: CosyVoice2Tokenizer,
}

impl VibeVoiceRealtimeTokenizer {
    /// Builds a tokenizer from the four exact upstream sidecar byte payloads.
    ///
    /// Hash and byte-length checks run before JSON/BPE parsing.  This is the
    /// only constructor that can authorize text IDs; a syntactically valid
    /// but different Qwen tokenizer is rejected as provenance/role drift.
    pub fn from_parts(
        vocab_json: &[u8],
        merges_txt: &[u8],
        tokenizer_config_json: &[u8],
        tokenizer_json: &[u8],
    ) -> Result<Self> {
        require_exact_asset("vocab.json", vocab_json, VOCAB_BYTES, VOCAB_SHA256)?;
        require_exact_asset("merges.txt", merges_txt, MERGES_BYTES, MERGES_SHA256)?;
        require_exact_asset(
            "tokenizer_config.json",
            tokenizer_config_json,
            CONFIG_BYTES,
            CONFIG_SHA256,
        )?;
        require_exact_asset(
            "tokenizer.json",
            tokenizer_json,
            TOKENIZER_JSON_BYTES,
            TOKENIZER_JSON_SHA256,
        )?;
        validate_tokenizer_config(tokenizer_config_json)?;
        validate_tokenizer_json(tokenizer_json)?;
        let bpe = CosyVoice2Tokenizer::from_parts(vocab_json, merges_txt).map_err(|error| {
            VokraError::ModelLoad(format!(
                "vibevoice-realtime tokenizer: fixed Qwen vocab/merges failed to parse: {error}"
            ))
        })?;
        if bpe.vocab_size() != BASE_VOCAB_SIZE {
            return Err(VokraError::ModelLoad(format!(
                "vibevoice-realtime tokenizer: vocab.json has {} entries, expected exactly {BASE_VOCAB_SIZE}",
                bpe.vocab_size()
            )));
        }
        Ok(Self { bpe })
    }

    /// Loads exact tokenizer sidecars embedded as U8 GGUF metadata arrays.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let revision = file
            .get(KEY_TOKENIZER_REVISION)
            .and_then(GgufMetadataValue::as_str)
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "vibevoice-realtime tokenizer: missing `{KEY_TOKENIZER_REVISION}`"
                ))
            })?;
        if revision != TOKENIZER_REVISION {
            return Err(VokraError::ModelLoad(format!(
                "vibevoice-realtime tokenizer: tokenizer revision {revision:?} differs from pinned {TOKENIZER_REVISION:?}"
            )));
        }
        let vocab = read_u8_array(file, KEY_TOKENIZER_VOCAB, "vocab.json")?;
        let merges = read_u8_array(file, KEY_TOKENIZER_MERGES, "merges.txt")?;
        let config = read_u8_array(file, KEY_TOKENIZER_CONFIG, "tokenizer_config.json")?;
        let tokenizer = read_u8_array(file, KEY_TOKENIZER_JSON, "tokenizer.json")?;
        Self::from_parts(&vocab, &merges, &config, &tokenizer)
    }

    /// Reads four exact sidecar files from a VAST-produced evidence directory.
    pub fn from_files(root: &Path) -> Result<Self> {
        fn read(root: &Path, name: &str) -> Result<Vec<u8>> {
            let path = root.join(name);
            if path.is_symlink() || !path.is_file() {
                return Err(VokraError::ModelLoad(format!(
                    "vibevoice-realtime tokenizer: {name:?} is not a regular non-symlink file"
                )));
            }
            if root.ancestors().any(|ancestor| ancestor.is_symlink()) {
                return Err(VokraError::ModelLoad(
                    "vibevoice-realtime tokenizer: sidecar directory has symlink ancestry".into(),
                ));
            }
            std::fs::read(path).map_err(|error| {
                VokraError::ModelLoad(format!(
                    "vibevoice-realtime tokenizer: unable to read {name:?}: {error}"
                ))
            })
        }
        Self::from_parts(
            &read(root, "vocab.json")?,
            &read(root, "merges.txt")?,
            &read(root, "tokenizer_config.json")?,
            &read(root, "tokenizer.json")?,
        )
    }

    /// Encodes raw text with the authenticated Qwen2 byte-level BPE.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        reject_speech_boundary_literals(text)?;
        let ids = self.bpe.encode(text)?;
        validate_text_ids(&ids)?;
        Ok(ids)
    }

    /// Applies the fixed streaming processor boundary: stripped text plus a
    /// terminal newline, with no added special tokens.
    pub fn streaming_text_ids(&self, text: &str) -> Result<Vec<u32>> {
        let mut normalized = text.trim().to_owned();
        normalized.push('\n');
        self.encode(&normalized)
    }

    /// Returns the upstream speech-start boundary id.
    #[must_use]
    pub const fn speech_start_id(&self) -> u32 {
        SPEECH_START_ID
    }

    /// Returns the upstream speech-end boundary id.
    #[must_use]
    pub const fn speech_end_id(&self) -> u32 {
        SPEECH_END_ID
    }

    /// Returns the upstream speech-diffusion/padding boundary id.
    #[must_use]
    pub const fn speech_pad_id(&self) -> u32 {
        SPEECH_PAD_ID
    }

    /// Reports whether an ID is one of VibeVoice's speech boundary tokens.
    #[must_use]
    pub const fn is_speech_boundary(token_id: u32) -> bool {
        matches!(token_id, SPEECH_START_ID | SPEECH_END_ID | SPEECH_PAD_ID)
    }
}

fn require_exact_asset(
    name: &str,
    bytes: &[u8],
    expected_bytes: usize,
    expected_sha256: &str,
) -> Result<()> {
    if bytes.len() != expected_bytes {
        return Err(VokraError::ModelLoad(format!(
            "vibevoice-realtime tokenizer: {name} has {} bytes, expected exactly {expected_bytes}",
            bytes.len()
        )));
    }
    let actual = hex_sha256(bytes);
    if actual != expected_sha256 {
        return Err(VokraError::ModelLoad(format!(
            "vibevoice-realtime tokenizer: {name} SHA-256 {actual}, expected {expected_sha256}"
        )));
    }
    Ok(())
}

fn validate_tokenizer_config(bytes: &[u8]) -> Result<()> {
    let root = vokra_core::json::parse(bytes).map_err(|error| {
        VokraError::ModelLoad(format!(
            "vibevoice-realtime tokenizer_config.json is invalid: {error}"
        ))
    })?;
    let class = root.get("tokenizer_class").and_then(|value| value.as_str());
    if !matches!(class, Some("Qwen2Tokenizer" | "Qwen2TokenizerFast")) {
        return Err(VokraError::ModelLoad(
            "vibevoice-realtime tokenizer_config.json has an unexpected tokenizer_class".into(),
        ));
    }
    let decoder = root
        .get("added_tokens_decoder")
        .and_then(|value| value.as_object())
        .ok_or_else(|| {
            VokraError::ModelLoad(
                "vibevoice-realtime tokenizer_config.json is missing added_tokens_decoder".into(),
            )
        })?;
    for (id, token) in [
        (SPEECH_START_ID, "<|vision_start|>"),
        (SPEECH_END_ID, "<|vision_end|>"),
        (SPEECH_PAD_ID, "<|vision_pad|>"),
    ] {
        let record = decoder
            .iter()
            .find(|(key, _)| key == &id.to_string())
            .and_then(|(_, value)| value.as_object())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "vibevoice-realtime tokenizer_config.json is missing added token id {id}"
                ))
            })?;
        if record.get("content").and_then(|value| value.as_str()) != Some(token)
            || !record.get("special").is_some_and(is_json_true)
        {
            return Err(VokraError::ModelLoad(format!(
                "vibevoice-realtime tokenizer_config.json added token id {id} does not match {token:?}"
            )));
        }
    }
    Ok(())
}

fn validate_tokenizer_json(bytes: &[u8]) -> Result<()> {
    let root = vokra_core::json::parse(bytes).map_err(|error| {
        VokraError::ModelLoad(format!(
            "vibevoice-realtime tokenizer.json is invalid: {error}"
        ))
    })?;
    let model = root.get("model").ok_or_else(|| {
        VokraError::ModelLoad("vibevoice-realtime tokenizer.json is missing model".into())
    })?;
    if model.get("type").and_then(|value| value.as_str()) != Some("BPE") {
        return Err(VokraError::ModelLoad(
            "vibevoice-realtime tokenizer.json model.type must be BPE".into(),
        ));
    }
    let vocab = model
        .get("vocab")
        .and_then(|value| value.as_object())
        .ok_or_else(|| {
            VokraError::ModelLoad("vibevoice-realtime tokenizer.json model.vocab is missing".into())
        })?;
    if vocab.len() != BASE_VOCAB_SIZE {
        return Err(VokraError::ModelLoad(format!(
            "vibevoice-realtime tokenizer.json model.vocab has {} entries, expected {BASE_VOCAB_SIZE}",
            vocab.len()
        )));
    }
    let merges = model
        .get("merges")
        .and_then(|value| value.as_array())
        .ok_or_else(|| {
            VokraError::ModelLoad(
                "vibevoice-realtime tokenizer.json model.merges is missing".into(),
            )
        })?;
    if merges.is_empty() {
        return Err(VokraError::ModelLoad(
            "vibevoice-realtime tokenizer.json model.merges is empty".into(),
        ));
    }
    let added = root
        .get("added_tokens")
        .and_then(|value| value.as_array())
        .ok_or_else(|| {
            VokraError::ModelLoad(
                "vibevoice-realtime tokenizer.json added_tokens is missing".into(),
            )
        })?;
    for (id, token) in [
        (SPEECH_START_ID, "<|vision_start|>"),
        (SPEECH_END_ID, "<|vision_end|>"),
        (SPEECH_PAD_ID, "<|vision_pad|>"),
    ] {
        let found = added.iter().any(|record| {
            record.get("id").and_then(|value| value.as_u64()) == Some(u64::from(id))
                && record.get("content").and_then(|value| value.as_str()) == Some(token)
                && record.get("special").is_some_and(is_json_true)
        });
        if !found {
            return Err(VokraError::ModelLoad(format!(
                "vibevoice-realtime tokenizer.json is missing special token {token:?} at id {id}"
            )));
        }
    }
    Ok(())
}

fn is_json_true(value: &vokra_core::json::JsonValue) -> bool {
    matches!(value, vokra_core::json::JsonValue::Bool(true))
}

fn validate_text_ids(ids: &[u32]) -> Result<()> {
    if ids.iter().any(|&id| id >= VOCAB_SIZE as u32) {
        return Err(VokraError::InvalidArgument(
            "vibevoice-realtime tokenizer produced an id outside the authenticated vocabulary"
                .into(),
        ));
    }
    if ids
        .iter()
        .any(|&id| VibeVoiceRealtimeTokenizer::is_speech_boundary(id))
    {
        return Err(VokraError::InvalidArgument(
            "vibevoice-realtime tokenizer text unexpectedly contains a speech boundary token"
                .into(),
        ));
    }
    Ok(())
}

fn reject_speech_boundary_literals(text: &str) -> Result<()> {
    for token in [SPEECH_START_TOKEN, SPEECH_END_TOKEN, SPEECH_PAD_TOKEN] {
        if text.contains(token) {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice-realtime tokenizer text must not contain reserved speech boundary {token:?}"
            )));
        }
    }
    Ok(())
}

fn read_u8_array(file: &GgufFile, key: &str, name: &str) -> Result<Vec<u8>> {
    let array = match file.get(key) {
        Some(GgufMetadataValue::Array(array)) => array,
        Some(other) => {
            return Err(VokraError::ModelLoad(format!(
                "vibevoice-realtime tokenizer: `{key}` for {name} is not a U8 array (got {:?})",
                other.value_type()
            )));
        }
        None => {
            return Err(VokraError::ModelLoad(format!(
                "vibevoice-realtime tokenizer: missing `{key}` ({name})"
            )));
        }
    };
    let mut bytes = Vec::with_capacity(array.values.len());
    for value in &array.values {
        match value {
            GgufMetadataValue::U8(byte) => bytes.push(*byte),
            other => {
                return Err(VokraError::ModelLoad(format!(
                    "vibevoice-realtime tokenizer: `{key}` ({name}) contains non-U8 {:?}",
                    other.value_type()
                )));
            }
        }
    }
    Ok(bytes)
}

fn hex_sha256(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in sha256_bytes(bytes) {
        output.push(char::from(HEX[(byte >> 4) as usize]));
        output.push(char::from(HEX[(byte & 0x0f) as usize]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_asset_drift_fails_before_parsing() {
        let error = VibeVoiceRealtimeTokenizer::from_parts(b"{}", b"", b"{}", b"{}").unwrap_err();
        assert!(error.to_string().contains("vocab.json has 2 bytes"));
    }

    #[test]
    fn speech_boundaries_are_fixed_and_distinct() {
        assert_eq!(SPEECH_START_ID, 151_652);
        assert_eq!(SPEECH_END_ID, 151_653);
        assert_eq!(SPEECH_PAD_ID, 151_654);
        assert!(VibeVoiceRealtimeTokenizer::is_speech_boundary(
            SPEECH_START_ID
        ));
        assert!(!VibeVoiceRealtimeTokenizer::is_speech_boundary(151_655));
    }

    #[test]
    fn reserved_speech_boundary_literals_are_rejected() {
        let error = reject_speech_boundary_literals("hello <|vision_start|>").unwrap_err();
        assert!(error.to_string().contains("reserved speech boundary"));
    }

    #[test]
    #[ignore = "requires exact VAST-recovered Qwen sidecars; no local tokenizer execution"]
    fn fixed_sidecars_bind_and_encode_on_remote_evidence() {
        let root = std::env::var_os("VOKRA_VIBEVOICE_TOKENIZER_DIR")
            .as_deref()
            .map(Path::new)
            .expect("VOKRA_VIBEVOICE_TOKENIZER_DIR must point to exact sidecars");
        let tokenizer = VibeVoiceRealtimeTokenizer::from_files(root).unwrap();
        let ids = tokenizer.streaming_text_ids("hello").unwrap();
        assert!(!ids.is_empty());
        assert!(ids.iter().all(|&id| id < VOCAB_SIZE as u32));
        assert!(
            ids.iter()
                .all(|&id| !VibeVoiceRealtimeTokenizer::is_speech_boundary(id))
        );
    }
}
