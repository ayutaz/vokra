//! Decode-only Hugging Face BPE tokenizer used by Moonshine generation.

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::{Result, VokraError};

pub(super) const KEY_TOKENIZER_MODEL: &str = "vokra.tokenizer.model";

#[derive(Debug, Clone)]
struct Piece {
    text: String,
    special: bool,
}

#[derive(Debug, Clone)]
pub(super) struct MoonshineTokenizer {
    pieces: Vec<Option<Piece>>,
}

impl MoonshineTokenizer {
    pub(super) fn from_gguf(file: &GgufFile, vocab_size: usize) -> Result<Self> {
        let array = match file.get(KEY_TOKENIZER_MODEL) {
            Some(GgufMetadataValue::Array(array)) => array,
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "moonshine tokenizer: `{KEY_TOKENIZER_MODEL}` must be a U8 array, got {other:?}"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(format!(
                    "moonshine tokenizer: `{KEY_TOKENIZER_MODEL}` is absent; reconvert with `--tokenizer tokenizer.json`"
                )));
            }
        };
        let mut bytes = Vec::with_capacity(array.values.len());
        for value in &array.values {
            let byte = value
                .as_u64()
                .and_then(|value| u8::try_from(value).ok())
                .ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "moonshine tokenizer: `{KEY_TOKENIZER_MODEL}` contains a non-byte element"
                    ))
                })?;
            bytes.push(byte);
        }
        let root = vokra_core::json::parse(&bytes).map_err(|error| {
            VokraError::ModelLoad(format!("moonshine tokenizer.json is invalid: {error}"))
        })?;
        let model = root.get("model").ok_or_else(|| {
            VokraError::ModelLoad("moonshine tokenizer.json: missing `model`".into())
        })?;
        if model.get("type").and_then(|value| value.as_str()) != Some("BPE") {
            return Err(VokraError::ModelLoad(
                "moonshine tokenizer.json: `model.type` must be `BPE`".into(),
            ));
        }
        let vocab = model
            .get("vocab")
            .and_then(|value| value.as_object())
            .ok_or_else(|| {
                VokraError::ModelLoad(
                    "moonshine tokenizer.json: `model.vocab` must be an object".into(),
                )
            })?;
        if vocab.len() != 32_000 {
            return Err(VokraError::ModelLoad(format!(
                "moonshine tokenizer.json: model vocab has {} entries, expected 32000",
                vocab.len()
            )));
        }
        let mut pieces = vec![None; vocab_size];
        for (text, id) in vocab {
            let id = id
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|&value| value < vocab_size)
                .ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "moonshine tokenizer.json: invalid id for piece {text:?}"
                    ))
                })?;
            if pieces[id].is_some() {
                return Err(VokraError::ModelLoad(format!(
                    "moonshine tokenizer.json: duplicate token id {id}"
                )));
            }
            pieces[id] = Some(Piece {
                text: text.clone(),
                special: false,
            });
        }
        let added = root
            .get("added_tokens")
            .and_then(|value| value.as_array())
            .ok_or_else(|| {
                VokraError::ModelLoad(
                    "moonshine tokenizer.json: `added_tokens` must be an array".into(),
                )
            })?;
        if added.len() != 771 {
            return Err(VokraError::ModelLoad(format!(
                "moonshine tokenizer.json: added_tokens has {} entries, expected 771",
                added.len()
            )));
        }
        let mut seen_added = vec![false; vocab_size];
        for token in added {
            let id = token
                .get("id")
                .and_then(|value| value.as_u64())
                .and_then(|value| usize::try_from(value).ok())
                .filter(|&value| value < vocab_size)
                .ok_or_else(|| {
                    VokraError::ModelLoad(
                        "moonshine tokenizer.json: added token has an invalid id".into(),
                    )
                })?;
            if std::mem::replace(&mut seen_added[id], true) {
                return Err(VokraError::ModelLoad(format!(
                    "moonshine tokenizer.json: duplicate added-token id {id}"
                )));
            }
            let text = token
                .get("content")
                .and_then(|value| value.as_str())
                .ok_or_else(|| {
                    VokraError::ModelLoad(
                        "moonshine tokenizer.json: added token is missing `content`".into(),
                    )
                })?;
            let special = matches!(
                token.get("special"),
                Some(vokra_core::json::JsonValue::Bool(true))
            );
            if !special {
                return Err(VokraError::ModelLoad(format!(
                    "moonshine tokenizer.json: added token id {id} is not special"
                )));
            }
            if let Some(existing) = pieces[id].as_ref() {
                if existing.text != text {
                    return Err(VokraError::ModelLoad(format!(
                        "moonshine tokenizer.json: added token id {id} conflicts with model-vocab piece"
                    )));
                }
            }
            pieces[id] = Some(Piece {
                text: text.into(),
                special,
            });
        }
        if pieces.iter().any(Option::is_none) {
            return Err(VokraError::ModelLoad(format!(
                "moonshine tokenizer.json does not cover every id in 0..{vocab_size}"
            )));
        }
        Ok(Self { pieces })
    }

    pub(super) fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            let piece = self
                .pieces
                .get(id as usize)
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    VokraError::InvalidArgument(format!(
                        "moonshine tokenizer: token id {id} is outside the vocabulary"
                    ))
                })?;
            if piece.special {
                continue;
            }
            if let Some(byte) = byte_fallback(&piece.text) {
                bytes.push(byte);
            } else {
                bytes.extend_from_slice(piece.text.replace('▁', " ").as_bytes());
            }
        }
        let rendered = String::from_utf8_lossy(&bytes);
        Ok(rendered.strip_prefix(' ').unwrap_or(&rendered).to_owned())
    }
}

fn byte_fallback(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    (hex.len() == 2).then(|| u8::from_str_radix(hex, 16).ok())?
}

#[cfg(test)]
mod tests {
    use super::byte_fallback;

    #[test]
    fn byte_fallback_accepts_only_canonical_piece() {
        assert_eq!(byte_fallback("<0xE3>"), Some(0xe3));
        assert_eq!(byte_fallback("<0x0>"), None);
        assert_eq!(byte_fallback("hello"), None);
    }
}
