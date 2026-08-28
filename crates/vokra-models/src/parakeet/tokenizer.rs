//! Decode-only Hugging Face BPE + Metaspace tokenizer for Parakeet-TDT.
//!
//! The official `nvidia/parakeet-tdt-0.6b-v3` release carries a
//! `tokenizer.json` whose model is BPE and whose decoder is Metaspace with
//! replacement `▁` and `prepend_scheme = "always"`.  TDT inference only
//! needs id-to-piece decoding, so the runtime deliberately does not implement
//! the training/encoding half (merges are irrelevant when decoding ids).

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::json;
use vokra_core::{Result, VokraError};

/// Raw official Hugging Face `tokenizer.json` bytes embedded by the converter.
pub const KEY_TOKENIZER_JSON: &str = "vokra.parakeet.tokenizer.json";

/// Plaintext SentencePiece vocabulary embedded for the original
/// Parakeet-TDT-1.1B NeMo release. The runtime needs only id-to-piece
/// decoding, so it deliberately does not parse the protobuf `.model`.
pub const KEY_SENTENCEPIECE_VOCAB: &str = "vokra.parakeet_tdt_1_1b.tokenizer.vocab";

/// Decode-only Parakeet tokenizer.
#[derive(Debug, Clone)]
pub struct ParakeetTokenizer {
    pieces: Vec<String>,
    special: Vec<bool>,
}

impl ParakeetTokenizer {
    /// Parses the embedded official tokenizer JSON.
    pub fn from_gguf(file: &GgufFile, vocab_size: usize) -> Result<Self> {
        let bytes = required_u8_array(file, KEY_TOKENIZER_JSON)?;
        Self::from_bytes(&bytes, vocab_size)
    }

    /// Parses the embedded decode-only SentencePiece vocabulary used by
    /// Parakeet-TDT-1.1B.
    pub fn from_gguf_sentencepiece_vocab(
        file: &GgufFile,
        vocab_size: usize,
        blank_id: u32,
    ) -> Result<Self> {
        let bytes = required_u8_array(file, KEY_SENTENCEPIECE_VOCAB)?;
        Self::from_sentencepiece_vocab_bytes(&bytes, vocab_size, blank_id)
    }

    /// Parses SentencePiece's plaintext `piece<TAB>score` export for
    /// decode-only inference and appends the model's blank symbol at its
    /// declared id. Scores are syntax-checked but are not needed to decode
    /// emitted token ids.
    pub fn from_sentencepiece_vocab_bytes(
        bytes: &[u8],
        vocab_size: usize,
        blank_id: u32,
    ) -> Result<Self> {
        let blank_id = blank_id as usize;
        if vocab_size == 0 || blank_id >= vocab_size {
            return Err(VokraError::ModelLoad(format!(
                "Parakeet SentencePiece vocab: blank id {blank_id} is outside 0..{vocab_size}"
            )));
        }
        let document = std::str::from_utf8(bytes).map_err(|error| {
            VokraError::ModelLoad(format!(
                "Parakeet SentencePiece vocab is not UTF-8: {error}"
            ))
        })?;
        let mut source_pieces = Vec::new();
        for (line_index, line) in document.lines().enumerate() {
            let (piece, score) = line.rsplit_once('\t').ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "Parakeet SentencePiece vocab line {} is not `piece<TAB>score`",
                    line_index + 1
                ))
            })?;
            if piece.is_empty() {
                return Err(VokraError::ModelLoad(format!(
                    "Parakeet SentencePiece vocab line {} has an empty piece",
                    line_index + 1
                )));
            }
            let score = score.parse::<f32>().map_err(|error| {
                VokraError::ModelLoad(format!(
                    "Parakeet SentencePiece vocab line {} has invalid score {score:?}: {error}",
                    line_index + 1
                ))
            })?;
            if !score.is_finite() {
                return Err(VokraError::ModelLoad(format!(
                    "Parakeet SentencePiece vocab line {} has non-finite score {score}",
                    line_index + 1
                )));
            }
            source_pieces.push(piece.to_owned());
        }
        if source_pieces.len() + 1 != vocab_size || source_pieces.len() != blank_id {
            return Err(VokraError::ModelLoad(format!(
                "Parakeet SentencePiece vocab has {} pieces; expected {} pieces followed by blank id {blank_id} in a head of width {vocab_size}",
                source_pieces.len(),
                vocab_size.saturating_sub(1),
            )));
        }
        let mut pieces = source_pieces;
        pieces.push(String::new());
        let mut special = vec![false; vocab_size];
        special[blank_id] = true;
        Ok(Self { pieces, special })
    }

    /// Parses an official Hugging Face `tokenizer.json` payload.
    pub fn from_bytes(bytes: &[u8], vocab_size: usize) -> Result<Self> {
        let root = json::parse(bytes).map_err(|error| {
            VokraError::ModelLoad(format!("Parakeet tokenizer JSON parse failed: {error}"))
        })?;
        let model = root.get("model").ok_or_else(|| {
            VokraError::ModelLoad("Parakeet tokenizer: missing `model` object".to_owned())
        })?;
        if model.get("type").and_then(|value| value.as_str()) != Some("BPE") {
            return Err(VokraError::ModelLoad(
                "Parakeet tokenizer: `model.type` must be `BPE`".to_owned(),
            ));
        }
        let decoder = root.get("decoder").ok_or_else(|| {
            VokraError::ModelLoad("Parakeet tokenizer: missing `decoder` object".to_owned())
        })?;
        if decoder.get("type").and_then(|value| value.as_str()) != Some("Metaspace")
            || decoder.get("replacement").and_then(|value| value.as_str()) != Some("▁")
            || decoder
                .get("prepend_scheme")
                .and_then(|value| value.as_str())
                != Some("always")
        {
            return Err(VokraError::ModelLoad(
                "Parakeet tokenizer: expected Metaspace decoder with replacement `▁` and prepend_scheme `always`"
                    .to_owned(),
            ));
        }

        let mut pieces = vec![String::new(); vocab_size];
        let mut occupied = vec![false; vocab_size];
        let vocab = model
            .get("vocab")
            .and_then(|value| value.as_object())
            .ok_or_else(|| {
                VokraError::ModelLoad("Parakeet tokenizer: missing `model.vocab` object".to_owned())
            })?;
        for (piece, id_value) in vocab {
            let id = id_value.as_u64().ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "Parakeet tokenizer: vocab id for {piece:?} is not a non-negative integer"
                ))
            })? as usize;
            if id >= vocab_size {
                return Err(VokraError::ModelLoad(format!(
                    "Parakeet tokenizer: vocab id {id} for {piece:?} is outside 0..{vocab_size}"
                )));
            }
            if occupied[id] {
                return Err(VokraError::ModelLoad(format!(
                    "Parakeet tokenizer: duplicate vocab id {id}"
                )));
            }
            pieces[id] = piece.clone();
            occupied[id] = true;
        }

        let mut special = vec![false; vocab_size];
        if let Some(added) = root.get("added_tokens").and_then(|value| value.as_array()) {
            for entry in added {
                let Some(id) = entry.get("id").and_then(|value| value.as_u64()) else {
                    return Err(VokraError::ModelLoad(
                        "Parakeet tokenizer: added token missing integer `id`".to_owned(),
                    ));
                };
                let id = id as usize;
                if id >= vocab_size {
                    return Err(VokraError::ModelLoad(format!(
                        "Parakeet tokenizer: added-token id {id} is outside 0..{vocab_size}"
                    )));
                }
                let content = entry
                    .get("content")
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| {
                        VokraError::ModelLoad(
                            "Parakeet tokenizer: added token missing string `content`".to_owned(),
                        )
                    })?;
                if !occupied[id] {
                    pieces[id] = content.to_owned();
                    occupied[id] = true;
                } else if pieces[id] != content {
                    return Err(VokraError::ModelLoad(format!(
                        "Parakeet tokenizer: added-token id {id} content {content:?} disagrees with vocab {:?}",
                        pieces[id]
                    )));
                }
                special[id] = matches!(entry.get("special"), Some(json::JsonValue::Bool(true)));
            }
        }
        if let Some(id) = occupied.iter().position(|present| !present) {
            return Err(VokraError::ModelLoad(format!(
                "Parakeet tokenizer: no piece is defined for id {id}"
            )));
        }
        Ok(Self { pieces, special })
    }

    /// Decodes emitted TDT ids. Repeated ids are intentionally retained;
    /// unlike CTC, a repeated TDT token is a real second emission.
    pub fn decode(
        &self,
        token_ids: &[u32],
        blank_id: u32,
        pad_id: u32,
        eos_id: Option<u32>,
    ) -> Result<String> {
        let mut encoded = String::new();
        for &token_id in token_ids {
            if token_id == blank_id || token_id == pad_id || eos_id == Some(token_id) {
                continue;
            }
            let id = token_id as usize;
            let piece = self.pieces.get(id).ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "Parakeet tokenizer: token id {token_id} outside 0..{}",
                    self.pieces.len()
                ))
            })?;
            if self.special[id] {
                continue;
            }
            encoded.push_str(piece);
        }
        let decoded = encoded.replace('▁', " ");
        Ok(decoded.strip_prefix(' ').unwrap_or(&decoded).to_owned())
    }
}

pub(super) fn required_u8_array(file: &GgufFile, key: &str) -> Result<Vec<u8>> {
    match file.get(key) {
        Some(GgufMetadataValue::Array(array)) => array
            .values
            .iter()
            .map(|value| match value {
                GgufMetadataValue::U8(byte) => Ok(*byte),
                _ => Err(VokraError::ModelLoad(format!(
                    "Parakeet tokenizer: `{key}` contains a non-u8 element"
                ))),
            })
            .collect(),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "Parakeet tokenizer: `{key}` must be a u8 array, found {other:?}"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "Parakeet tokenizer: `{key}` is absent; supply the pinned official tokenizer sidecar or reconvert with `--tokenizer`"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINI: &[u8] = br#"{
      "model":{"type":"BPE","vocab":{"<unk>":0,"a":1,"<pad>":2,"\u2581hello":3}},
      "decoder":{"type":"Metaspace","replacement":"\u2581","prepend_scheme":"always","split":true},
      "added_tokens":[
        {"id":0,"content":"<unk>","special":true},
        {"id":2,"content":"<pad>","special":true},
        {"id":4,"content":"<blank>","special":true}
      ]
    }"#;

    #[test]
    fn bpe_metaspace_decode_keeps_repeated_tdt_tokens() {
        let tokenizer = ParakeetTokenizer::from_bytes(MINI, 5).expect("parse mini tokenizer");
        assert_eq!(
            tokenizer.decode(&[3, 1, 1, 4, 2], 4, 2, Some(0)).unwrap(),
            "helloaa"
        );
    }

    #[test]
    fn tokenizer_rejects_sentencepiece_misclassification() {
        let document = std::str::from_utf8(MINI).expect("fixture is UTF-8");
        let bytes = document.replacen("\"BPE\"", "\"Unigram\"", 1);
        let error = ParakeetTokenizer::from_bytes(bytes.as_bytes(), 5).unwrap_err();
        assert!(error.to_string().contains("must be `BPE`"));
    }

    #[test]
    fn sentencepiece_vocab_decodes_without_a_protobuf_runtime() {
        let tokenizer = ParakeetTokenizer::from_sentencepiece_vocab_bytes(
            b"<unk>\t0\n\xe2\x96\x81hello\t-1\na\t-2\n",
            4,
            3,
        )
        .expect("parse plaintext vocab");
        assert_eq!(
            tokenizer.decode(&[1, 2, 2, 3], 3, 3, None).unwrap(),
            "helloaa"
        );
    }
}
