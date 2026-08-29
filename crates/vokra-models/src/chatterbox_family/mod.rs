//! Source-authenticated T3 contracts shared by the Chatterbox family.
//!
//! This module deliberately stops at T3.  Chatterbox's public artefacts are
//! composite pipelines (voice encoder, S3 tokenizer/S3Gen, HiFT and
//! watermark); a T3-only GGUF is therefore not a runnable PCM checkpoint.
//! The contracts here make the native boundary explicit without opening the
//! historical T3-only loaders.

use std::collections::HashMap;

use vokra_core::{Result, VokraError};

/// Fixed upstream family source revision.
pub const SOURCE_REVISION: &str = "5de7a54aa4e5e2baadb0182dde554908b48b85c2";
/// Fixed upstream source URL.
pub const SOURCE_URL: &str = "https://github.com/resemble-ai/chatterbox.git";
/// Explicitly selected base checkpoint (never the upstream v2 default).
pub const BASE_V3_CHECKPOINT: &str = "t3_mtl23ls_v3.safetensors";
/// Maximum sequence length of the base Llama backbone.
pub const BASE_BACKBONE_MAX_POSITION_EMBEDDINGS: usize = 131_072;
/// Number of learned base text-position rows (`max_text + 2`).
pub const BASE_LEARNED_TEXT_POSITIONS: usize = 2_050;
/// Number of learned base speech-position rows (`max_speech + 4`).
pub const BASE_LEARNED_SPEECH_POSITIONS: usize = 4_100;
/// Fixed number of rows emitted by the base prompt-speech Perceiver.
pub const BASE_PERCEIVER_OUTPUT_ROWS: usize = 32;

// No complete composite manifest has been authenticated yet.  Keeping this
// sentinel empty is intentional: a future VAST-reviewed composite manifest
// must replace it with its exact digest before this method can ever bind PCM.
const AUTHENTICATED_COMPOSITE_MANIFEST_SHA256: &str = "";

/// Metadata required before a future composite GGUF can be bound to the
/// native T3 implementation.  Tensor names are intentionally supplied by an
/// authenticated conversion manifest; this contract never invents names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeBinderEvidence {
    pub variant: ChatterboxVariant,
    pub source_revision: String,
    pub checkpoint_filename: String,
    pub tensor_manifest_sha256: String,
    pub includes_voice_encoder: bool,
    pub includes_s3_tokenizer: bool,
    pub includes_s3gen: bool,
    pub includes_hift: bool,
    pub includes_watermark: bool,
}

impl CompositeBinderEvidence {
    /// Fail closed unless every PCM pipeline component is authenticated.
    pub fn require_complete_pcm(&self) -> Result<()> {
        let manifest_is_authenticated = self.tensor_manifest_sha256.len() == 64
            && self
                .tensor_manifest_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            && self.tensor_manifest_sha256.bytes().any(|byte| byte != b'0')
            && self.tensor_manifest_sha256 == AUTHENTICATED_COMPOSITE_MANIFEST_SHA256;
        if self.source_revision != SOURCE_REVISION || !manifest_is_authenticated {
            return Err(VokraError::ModelLoad(
                "chatterbox composite binder: source/manifest identity is not authenticated".into(),
            ));
        }
        if !(self.includes_voice_encoder
            && self.includes_s3_tokenizer
            && self.includes_s3gen
            && self.includes_hift
            && self.includes_watermark)
        {
            return Err(VokraError::NotImplemented(
                "chatterbox composite binder: T3-only checkpoint is incomplete; voice encoder, S3 tokenizer, S3Gen, HiFT and watermark are all required before PCM".into(),
            ));
        }
        Ok(())
    }
}

/// The three T3 variants supported by the source-native slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatterboxVariant {
    /// Multilingual Llama T3, explicitly v3.
    MultilingualV3,
    /// GPT-2 small T3.
    Nano,
    /// GPT-2 medium T3.
    Turbo,
}

/// Authenticated architecture axes.  This is intentionally independent from
/// the old shape-only weight structs in the per-variant modules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct T3Architecture {
    pub variant: ChatterboxVariant,
    pub backbone: Backbone,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub ffn: usize,
    pub positions: usize,
    pub text_vocab: usize,
    pub speech_vocab: usize,
}

/// Exact GPT-2 T3 structural contract.  The weight vectors are deliberately
/// not synthesized here: a future composite binder must provide an
/// authenticated tensor manifest before materializing these names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gpt2T3Contract {
    pub architecture: T3Architecture,
    pub has_learned_wpe: bool,
    pub has_custom_text_pos_emb: bool,
    pub has_custom_speech_pos_emb: bool,
    pub has_fused_qkv_bias: bool,
    pub has_layer_norm_bias: bool,
    pub has_swiglu: bool,
    pub has_perceiver: bool,
    pub has_emotion_projection: bool,
}

impl Gpt2T3Contract {
    /// Source-authenticated contract for Nano or Turbo.
    pub fn for_variant(variant: ChatterboxVariant) -> Result<Self> {
        match variant {
            ChatterboxVariant::MultilingualV3 => Err(VokraError::InvalidArgument(
                "chatterbox GPT-2 contract requested for multilingual Llama variant".into(),
            )),
            ChatterboxVariant::Nano | ChatterboxVariant::Turbo => Ok(Self {
                architecture: variant.architecture(),
                has_learned_wpe: true,
                has_custom_text_pos_emb: false,
                has_custom_speech_pos_emb: false,
                has_fused_qkv_bias: true,
                has_layer_norm_bias: true,
                has_swiglu: false,
                has_perceiver: false,
                has_emotion_projection: false,
            }),
        }
    }
}

/// T3 transformer family.  Nano/Turbo must never be represented as Llama.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backbone {
    Llama520M,
    Gpt2Small,
    Gpt2Medium,
}

impl ChatterboxVariant {
    /// Return the exact effective T3 axes authenticated from upstream code.
    #[must_use]
    pub const fn architecture(self) -> T3Architecture {
        match self {
            Self::MultilingualV3 => T3Architecture {
                variant: self,
                backbone: Backbone::Llama520M,
                hidden: 1024,
                layers: 30,
                heads: 16,
                kv_heads: 16,
                ffn: 4096,
                positions: BASE_BACKBONE_MAX_POSITION_EMBEDDINGS,
                text_vocab: 2454,
                speech_vocab: 8194,
            },
            Self::Nano => T3Architecture {
                variant: self,
                backbone: Backbone::Gpt2Small,
                hidden: 768,
                layers: 12,
                heads: 12,
                kv_heads: 12,
                ffn: 3072,
                positions: 8196,
                text_vocab: 50276,
                speech_vocab: 6563,
            },
            Self::Turbo => T3Architecture {
                variant: self,
                backbone: Backbone::Gpt2Medium,
                hidden: 1024,
                layers: 24,
                heads: 16,
                kv_heads: 16,
                ffn: 4096,
                positions: 8196,
                text_vocab: 50276,
                speech_vocab: 6563,
            },
        }
    }

    /// Stable source/config identity, used in evidence packets and errors.
    #[must_use]
    pub const fn identity(self) -> &'static str {
        match self {
            Self::MultilingualV3 => "chatterbox-multilingual-v3",
            Self::Nano => "chatterbox-nano-v1",
            Self::Turbo => "chatterbox-turbo-v1",
        }
    }
}

/// Exact non-text conditioning stages in upstream `T3CondEnc.forward`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditioningStage {
    SpeakerProjection,
    PromptSpeechPerceiver,
    EmotionProjection,
}

/// Return the source order.  An absent prompt contributes zero rows; CLAP is
/// intentionally not present because upstream asserts it is unsupported.
#[must_use]
pub const fn conditioning_stages(variant: ChatterboxVariant) -> &'static [ConditioningStage] {
    match variant {
        ChatterboxVariant::MultilingualV3 => &[
            ConditioningStage::SpeakerProjection,
            ConditioningStage::PromptSpeechPerceiver,
            ConditioningStage::EmotionProjection,
        ],
        ChatterboxVariant::Nano | ChatterboxVariant::Turbo => {
            &[ConditioningStage::SpeakerProjection]
        }
    }
}

/// Rows resulting from source-ordered conditioning concatenation.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditioningRows {
    pub stages: Vec<ConditioningStage>,
    pub rows: Vec<Vec<f32>>,
}

/// Concatenate already-computed source-native conditioning outputs.
///
/// The projections and (for base) Perceiver are intentionally supplied by the
/// caller: they are learned tensors and cannot be fabricated here.  This
/// function owns the ordering, empty-prompt behavior, dimension checks, and
/// CLAP fail-closed boundary.
pub fn assemble_conditioning(
    variant: ChatterboxVariant,
    speaker_projected: &[f32],
    prompt_perceiver_rows: &[Vec<f32>],
    emotion_projected: Option<&[f32]>,
    clap_present: bool,
) -> Result<ConditioningRows> {
    if clap_present {
        return Err(VokraError::UnsupportedOp(
            "chatterbox T3 conditioning: upstream T3CondEnc asserts CLAP is unsupported; refusing an approximation".into(),
        ));
    }
    if speaker_projected.is_empty() {
        return Err(VokraError::InvalidArgument(
            "chatterbox T3 conditioning: projected speaker row is empty".into(),
        ));
    }
    let dim = speaker_projected.len();
    if variant == ChatterboxVariant::MultilingualV3
        && !prompt_perceiver_rows.is_empty()
        && prompt_perceiver_rows.len() != BASE_PERCEIVER_OUTPUT_ROWS
    {
        return Err(VokraError::InvalidArgument(format!(
            "chatterbox multilingual v3 conditioning: Perceiver must emit exactly {BASE_PERCEIVER_OUTPUT_ROWS} rows"
        )));
    }
    if prompt_perceiver_rows.iter().any(|row| row.len() != dim) {
        return Err(VokraError::InvalidArgument(
            "chatterbox T3 conditioning: prompt rows must match hidden width".into(),
        ));
    }
    let mut rows = vec![speaker_projected.to_vec()];
    let mut stages = vec![ConditioningStage::SpeakerProjection];
    if variant == ChatterboxVariant::MultilingualV3 {
        rows.extend(prompt_perceiver_rows.iter().cloned());
        stages.push(ConditioningStage::PromptSpeechPerceiver);
        let emotion = emotion_projected.ok_or_else(|| {
            VokraError::InvalidArgument(
                "chatterbox multilingual v3 conditioning: emotion_adv projection is required"
                    .into(),
            )
        })?;
        if emotion.len() != dim {
            return Err(VokraError::InvalidArgument(
                "chatterbox multilingual v3 conditioning: emotion row must match hidden width"
                    .into(),
            ));
        }
        rows.push(emotion.to_vec());
        stages.push(ConditioningStage::EmotionProjection);
    } else if emotion_projected.is_some() || !prompt_perceiver_rows.is_empty() {
        return Err(VokraError::InvalidArgument(
            "chatterbox GPT-2 conditioning: prompt Perceiver/emotion rows are not in the upstream route".into(),
        ));
    }
    Ok(ConditioningRows { stages, rows })
}

/// Exact punctuation normalization used by the Turbo/Nano upstream wrapper.
pub fn punc_norm_gpt2(text: &str) -> String {
    if text.is_empty() {
        return "You need to add some text for me to talk.".into();
    }
    let mut chars = text.chars();
    let mut normalized = match chars.next() {
        Some(c) if c.is_lowercase() => c.to_uppercase().collect::<String>() + chars.as_str(),
        _ => text.to_owned(),
    };
    normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");
    for (from, to) in [
        ('…', ", "),
        (':', ","),
        ('—', "-"),
        ('–', "-"),
        ('“', "\""),
        ('”', "\""),
        ('‘', "'"),
        ('’', "'"),
    ] {
        normalized = normalized.replace(from, to);
    }
    normalized = normalized.replace(" ,", ",").trim_end().to_owned();
    if !ends_with_sentence_punctuation(&normalized) {
        normalized.push('.');
    }
    normalized
}

/// Exact punctuation normalization used by the base multilingual wrapper.
pub fn punc_norm_multilingual(text: &str) -> String {
    if text.is_empty() {
        return "You need to add some text for me to talk.".into();
    }
    let mut text = text.to_owned();
    if text.chars().next().is_some_and(char::is_lowercase) {
        let first = text.remove(0);
        text = first.to_uppercase().collect::<String>() + &text;
    }
    text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    for (from, to) in [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ] {
        text = text.replace(from, to);
    }
    text = text.trim_end().to_owned();
    if !ends_with_sentence_punctuation(&text) {
        text.push('.');
    }
    text
}

fn ends_with_sentence_punctuation(text: &str) -> bool {
    text.chars().last().is_some_and(|c| {
        matches!(
            c,
            '.' | '!' | '?' | '-' | ',' | '、' | '，' | '。' | '？' | '！'
        )
    })
}

/// Base tokenizer's exact preprocessing boundary.  NFKD and language
/// specific transforms are not available in the dependency-free runtime;
/// non-ASCII or unsupported language inputs therefore fail closed.
pub fn normalize_multilingual_input(text: &str, language: Option<&str>) -> Result<String> {
    let lang = language.map(str::to_ascii_lowercase);
    if let Some(lang) = &lang {
        const LANGS: &[&str] = &[
            "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms",
            "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
        ];
        if !LANGS.contains(&lang.as_str()) {
            return Err(VokraError::InvalidArgument(format!(
                "chatterbox multilingual tokenizer: unsupported language {lang:?}"
            )));
        }
        if matches!(lang.as_str(), "zh" | "ja" | "he" | "ru" | "ko") {
            return Err(VokraError::UnsupportedOp(format!(
                "chatterbox multilingual tokenizer: {lang} requires upstream language-specific preprocessing data"
            )));
        }
    }
    if !text.is_ascii() {
        return Err(VokraError::UnsupportedOp(
            "chatterbox multilingual tokenizer: authenticated NFKD/language preprocessing is unavailable for non-ASCII input".into(),
        ));
    }
    let mut out = text.to_ascii_lowercase();
    if let Some(lang) = lang {
        out = format!("[{lang}]{out}");
    }
    Ok(out.replace(' ', "[SPACE]"))
}

/// GPT-2 BPE tokenizer loaded from an authenticated `tokenizer.json`-like
/// blob. Nano/Turbo use separate `vocab.json` + `merges.txt`; use
/// [`Gpt2Tokenizer::from_parts`] for those files.
#[derive(Debug, Clone)]
pub struct Gpt2Tokenizer {
    vocab: HashMap<String, u32>,
    merges: HashMap<(String, String), usize>,
    unk_id: u32,
    added_tokens: HashMap<String, u32>,
}

/// Marker for the base multilingual tokenizer boundary.
///
/// The pinned grapheme tokenizer uses language-specific pre-tokenization and
/// cannot be represented by the GPT-2 byte-BPE implementation below. Keep this
/// type fail-closed until an authenticated source-compatible parser exists.
#[derive(Debug, Clone)]
pub struct MultilingualTokenizer;

impl MultilingualTokenizer {
    /// Parse enough JSON to distinguish malformed data, then refuse the
    /// unauthenticated pre-tokenizer/BPE subset. In particular, this must not
    /// route the multilingual JSON through GPT-2 byte-BPE.
    pub fn from_tokenizer_json(data: &[u8]) -> Result<Self> {
        vokra_core::json::parse(data).map_err(|e| {
            VokraError::ModelLoad(format!(
                "chatterbox multilingual tokenizer: invalid JSON: {e}"
            ))
        })?;
        Err(VokraError::UnsupportedOp(
            "chatterbox multilingual tokenizer: authenticated grapheme pre-tokenizer/BPE execution is unavailable; official ID parity is blocked".into(),
        ))
    }

    /// Encode after the exact lower-case/NFKD/language-prefix boundary.
    pub fn encode(&self, _text: &str, _language: Option<&str>) -> Result<Vec<u32>> {
        Err(VokraError::UnsupportedOp(
            "chatterbox multilingual tokenizer: authenticated grapheme pre-tokenizer/BPE execution is unavailable; official ID parity is blocked".into(),
        ))
    }
}

impl Gpt2Tokenizer {
    /// Parse GPT-2 vocab and merge files. `added_tokens` must contain exactly
    /// the authenticated 19 tags and ids 50257..50275.
    pub fn from_parts(
        vocab_json: &[u8],
        merges_txt: &[u8],
        added_tokens_json: &[u8],
    ) -> Result<Self> {
        let vocab = parse_string_id_object(vocab_json, "GPT-2 vocab")?;
        if vocab.len() != 50_257 {
            return Err(VokraError::ModelLoad(format!(
                "chatterbox GPT-2 tokenizer: vocab has {} entries, expected 50257",
                vocab.len()
            )));
        }
        let added = parse_string_id_object(added_tokens_json, "added_tokens")?;
        if added.len() != 19
            || added
                .values()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                != (50_257..50_276).collect()
        {
            return Err(VokraError::ModelLoad(
                "chatterbox GPT-2 tokenizer: added_tokens must be the authenticated 19 ids 50257..50275".into(),
            ));
        }
        let text = std::str::from_utf8(merges_txt).map_err(|_| {
            VokraError::ModelLoad("chatterbox GPT-2 tokenizer: merges are not UTF-8".into())
        })?;
        let mut merges = HashMap::new();
        let mut lines = text.lines();
        if lines.next() != Some("#version: 0.2") {
            return Err(VokraError::ModelLoad(
                "chatterbox GPT-2 tokenizer: invalid merges header".into(),
            ));
        }
        for (rank, line) in lines.enumerate() {
            let mut fields = line.split(' ');
            let left = fields.next().unwrap_or_default();
            let right = fields.next().unwrap_or_default();
            if left.is_empty() || right.is_empty() || fields.next().is_some() {
                return Err(VokraError::ModelLoad(
                    "chatterbox GPT-2 tokenizer: malformed merge row".into(),
                ));
            }
            merges.insert((left.to_owned(), right.to_owned()), rank);
        }
        if merges.len() != 50_000 {
            return Err(VokraError::ModelLoad(format!(
                "chatterbox GPT-2 tokenizer: {} merges, expected 50000",
                merges.len()
            )));
        }
        let unk_id = *vocab.get("<|endoftext|>").ok_or_else(|| {
            VokraError::ModelLoad("chatterbox GPT-2 tokenizer: missing <|endoftext|>".into())
        })?;
        Ok(Self {
            vocab,
            merges,
            unk_id,
            added_tokens: added,
        })
    }

    /// Encode normalized GPT-2 text.  This implements the byte-level BPE
    /// merge loop; callers own the upstream punc normalization.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        encode_bpe_with_atomic_markers(
            &self.vocab,
            &self.merges,
            self.unk_id,
            &self.added_tokens,
            text,
        )
    }

    /// Encode a GPT-2 wrapper input using upstream punc normalization.
    pub fn encode_wrapper(&self, text: &str) -> Result<Vec<u32>> {
        self.encode(&punc_norm_gpt2(text))
    }
}

fn parse_string_id_object(data: &[u8], label: &str) -> Result<HashMap<String, u32>> {
    let root = vokra_core::json::parse(data)
        .map_err(|e| VokraError::ModelLoad(format!("chatterbox {label}: invalid JSON: {e}")))?;
    let object = root
        .as_object()
        .ok_or_else(|| VokraError::ModelLoad(format!("chatterbox {label}: root must be object")))?;
    let mut out = HashMap::with_capacity(object.len());
    for (key, value) in object {
        let id = value
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!("chatterbox {label}: token {key:?} id is not u32"))
            })?;
        if out.insert(key.clone(), id).is_some() {
            return Err(VokraError::ModelLoad(format!(
                "chatterbox {label}: duplicate token {key:?}"
            )));
        }
    }
    Ok(out)
}

fn encode_bpe_with_atomic_markers(
    vocab: &HashMap<String, u32>,
    merges: &HashMap<(String, String), usize>,
    unk_id: u32,
    atomic_tokens: &HashMap<String, u32>,
    text: &str,
) -> Result<Vec<u32>> {
    let mut ids = Vec::new();
    let mut cursor = 0;
    while cursor < text.len() {
        let marker = atomic_tokens
            .keys()
            .filter_map(|key| {
                text[cursor..]
                    .find(key)
                    .map(|offset| (cursor + offset, key))
            })
            .min_by_key(|(offset, _)| *offset);
        let Some((marker_start, marker)) = marker else {
            for piece in gpt2_pieces(&text[cursor..])? {
                append_bpe_piece(vocab, merges, unk_id, &piece, &mut ids);
            }
            break;
        };
        if marker_start > cursor {
            for piece in gpt2_pieces(&text[cursor..marker_start])? {
                append_bpe_piece(vocab, merges, unk_id, &piece, &mut ids);
            }
        }
        ids.push(atomic_tokens[marker]);
        cursor = marker_start + marker.len();
    }
    if text.is_empty() || ids.is_empty() {
        return Err(VokraError::InvalidArgument(
            "chatterbox tokenizer: input produced no tokens".into(),
        ));
    }
    Ok(ids)
}

fn append_bpe_piece(
    vocab: &HashMap<String, u32>,
    merges: &HashMap<(String, String), usize>,
    unk_id: u32,
    piece: &str,
    ids: &mut Vec<u32>,
) {
    if piece.is_empty() {
        return;
    }
    if let Some(&id) = vocab.get(piece) {
        ids.push(id);
        return;
    }
    let mut symbols = bytes_to_unicode(piece)
        .chars()
        .map(|c| c.to_string())
        .collect::<Vec<_>>();
    while symbols.len() > 1 {
        let mut best = None;
        for i in 0..symbols.len() - 1 {
            if let Some(&rank) = merges.get(&(symbols[i].clone(), symbols[i + 1].clone())) {
                if best.is_none_or(|(_, old)| rank < old) {
                    best = Some((i, rank));
                }
            }
        }
        let Some((i, _)) = best else { break };
        let right = symbols.remove(i + 1);
        symbols[i].push_str(&right);
    }
    for symbol in symbols {
        ids.push(*vocab.get(&symbol).unwrap_or(&unk_id));
    }
}

fn gpt2_byte_to_unicode(byte: u8) -> char {
    // This is the canonical OpenAI/Transformers bytes_to_unicode table. The
    // excluded bytes are assigned consecutive code points, not 256 + byte.
    let mut rank = 0u32;
    let mut value = 0u32;
    for candidate in 0u16..=255 {
        let candidate = candidate as u8;
        let printable = matches!(candidate, 33..=126 | 161..=172 | 174..=255);
        if candidate == byte {
            value = if printable {
                u32::from(candidate)
            } else {
                256 + rank
            };
            break;
        }
        if !printable {
            rank += 1;
        }
    }
    char::from_u32(value).expect("GPT-2 byte map is valid")
}

fn bytes_to_unicode(text: &str) -> String {
    text.bytes().map(gpt2_byte_to_unicode).collect()
}

/// A dependency-free ASCII implementation of the fixed GPT-2 pre-tokenizer.
/// It follows the upstream GPT-2 regex's important byte-level behavior:
/// ordinary words/numbers/punctuation retain one leading space. Non-ASCII
/// input is rejected because the Unicode regex path needs the authenticated
/// tokenizer runtime.
fn gpt2_pieces(text: &str) -> Result<Vec<String>> {
    if !text.is_ascii() {
        return Err(VokraError::UnsupportedOp("chatterbox GPT-2 tokenizer: non-ASCII GPT-2 regex path is unavailable without the authenticated tokenizer runtime".into()));
    }
    let bytes = text.as_bytes();
    let mut pieces = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_whitespace() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i == bytes.len() {
                pieces.push(text[start..i].to_owned());
            } else if i - start > 1 {
                // GPT-2's `\s+(?!\S)` consumes a whitespace run ending at a
                // non-space only through the final single-space alternative.
                pieces.push(text[start..i - 1].to_owned());
            }
            continue;
        }
        let start = i;
        let has_leading_space = start > 0 && bytes[start - 1].is_ascii_whitespace();
        let class = if bytes[i].is_ascii_alphabetic() {
            0 // \p{L}
        } else if bytes[i].is_ascii_digit() {
            1 // \p{N}
        } else {
            2 // punctuation/symbol
        };
        if class == 0 {
            while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
                i += 1;
            }
        } else if class == 1 {
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        } else if bytes[i] == b'\'' && i + 1 < bytes.len() {
            let rest = &bytes[i + 1..];
            let contraction_len = if matches!(
                rest.first().map(u8::to_ascii_lowercase),
                Some(b's' | b'd' | b't' | b'm')
            ) {
                Some(2)
            } else if rest
                .get(..2)
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"ll"))
                || rest
                    .get(..2)
                    .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"ve"))
                || rest
                    .get(..2)
                    .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"re"))
            {
                Some(3)
            } else {
                None
            };
            if let Some(length) = contraction_len {
                i += length;
            } else {
                while i < bytes.len()
                    && !bytes[i].is_ascii_whitespace()
                    && !bytes[i].is_ascii_alphabetic()
                    && !bytes[i].is_ascii_digit()
                {
                    i += 1;
                }
            }
        } else {
            while i < bytes.len()
                && !bytes[i].is_ascii_whitespace()
                && !bytes[i].is_ascii_alphabetic()
                && !bytes[i].is_ascii_digit()
            {
                i += 1;
            }
        }
        let mut piece_start = start;
        if has_leading_space {
            piece_start -= 1;
        }
        pieces.push(text[piece_start..i].to_owned());
    }
    Ok(pieces)
}

/// Exact logits processor order for a generation route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Processor {
    Temperature,
    TopK,
    MinP,
    TopP,
    RepetitionPenalty,
}

/// Sampling controls mirrored from the upstream T3 wrappers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 1000,
            top_p: 0.95,
            min_p: 0.05,
            repetition_penalty: 1.2,
        }
    }
}

/// Apply the exact source processor sequence in-place. `-INFINITY` is used
/// for filtered candidates, matching Transformers' logits warpers.
pub fn apply_processors(
    variant: ChatterboxVariant,
    logits: &mut [f32],
    history: &[u32],
    cfg: SamplingConfig,
) -> Result<()> {
    if logits.is_empty()
        // Upstream processors intentionally use -inf for filtered entries;
        // reject NaN/+inf before processing so softmax cannot hide corrupt
        // logits while preserving that explicit filtered sentinel.
        || logits.iter().any(|x| x.is_nan() || *x == f32::INFINITY)
        || !cfg.temperature.is_finite()
        || cfg.temperature <= 0.0
        || !cfg.top_p.is_finite()
        || !(0.0 < cfg.top_p && cfg.top_p <= 1.0)
        || !cfg.min_p.is_finite()
        || !(0.0..=1.0).contains(&cfg.min_p)
        || !cfg.repetition_penalty.is_finite()
        || cfg.repetition_penalty <= 0.0
    {
        return Err(VokraError::InvalidArgument(
            "chatterbox sampling: invalid processor configuration or logits".into(),
        ));
    }
    for processor in processor_order(variant) {
        match processor {
            Processor::Temperature if cfg.temperature > 0.0 && cfg.temperature != 1.0 => {
                for score in logits.iter_mut() {
                    *score /= cfg.temperature;
                }
            }
            Processor::TopK if cfg.top_k > 0 && cfg.top_k < logits.len() => {
                // Transformers' TopKLogitsWarper computes a kth-value
                // threshold and removes scores strictly below it. Equal
                // values therefore all survive, even if this keeps > k rows.
                let mut values = logits
                    .iter()
                    .copied()
                    .filter(|score| score.is_finite())
                    .collect::<Vec<_>>();
                if !values.is_empty() {
                    values.sort_by(|a, b| b.total_cmp(a));
                    let threshold = values[cfg.top_k.min(values.len()) - 1];
                    for score in logits.iter_mut() {
                        if score.is_finite() && *score < threshold {
                            *score = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            Processor::MinP if cfg.min_p > 0.0 && cfg.min_p < 1.0 => {
                let max = logits
                    .iter()
                    .copied()
                    .filter(|x| x.is_finite())
                    .fold(f32::NEG_INFINITY, f32::max);
                if max.is_finite() {
                    let threshold = max + cfg.min_p.ln();
                    for score in logits.iter_mut() {
                        if score.is_finite() && *score < threshold {
                            *score = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            Processor::TopP if cfg.top_p < 1.0 => {
                let max = logits
                    .iter()
                    .copied()
                    .filter(|x| x.is_finite())
                    .fold(f32::NEG_INFINITY, f32::max);
                if max.is_finite() {
                    let mut indices: Vec<usize> = (0..logits.len())
                        .filter(|&i| logits[i].is_finite())
                        .collect();
                    // `TopPLogitsWarper` in the pinned Transformers 5.2.0
                    // release sorts ascending and marks the tail whose
                    // cumulative probability is <= 1 - top_p. Transformers
                    // does not specify an index order for equal logits, so a
                    // mask that would split an equal-logit group is rejected
                    // below until an official trace fixes that tie order.
                    indices.sort_by(|&a, &b| logits[a].total_cmp(&logits[b]));
                    let mut tail_cumulative = 0.0f32;
                    let denom: f32 = indices.iter().map(|&i| (logits[i] - max).exp()).sum();
                    if denom > 0.0 {
                        let cutoff = 1.0f32 - cfg.top_p;
                        let mut remove = vec![false; indices.len()];
                        for (position, &i) in indices.iter().enumerate() {
                            let p = (logits[i] - max).exp() / denom;
                            tail_cumulative += p;
                            remove[position] = tail_cumulative <= cutoff;
                        }
                        // Transformers sets the final ascending row to
                        // `remove=false` (`min_tokens_to_keep=1`) after the
                        // cumulative comparison. This also protects against
                        // f32 rounding at extremely small positive top-p.
                        if let Some(last) = remove.last_mut() {
                            *last = false;
                        }
                        for position in 1..indices.len() {
                            if remove[position] != remove[position - 1]
                                && logits[indices[position]] == logits[indices[position - 1]]
                            {
                                return Err(VokraError::InvalidArgument(
                                    "chatterbox sampling: top-p tie boundary is unverified".into(),
                                ));
                            }
                        }
                        for (position, &i) in indices.iter().enumerate() {
                            if remove[position] {
                                logits[i] = f32::NEG_INFINITY;
                            }
                        }
                    }
                }
            }
            Processor::RepetitionPenalty if cfg.repetition_penalty != 1.0 => {
                let mut unique_history = history.to_vec();
                unique_history.sort_unstable();
                unique_history.dedup();
                for id in unique_history {
                    let Some(score) = logits.get_mut(id as usize) else {
                        continue;
                    };
                    if *score < 0.0 {
                        *score *= cfg.repetition_penalty;
                    } else {
                        *score /= cfg.repetition_penalty;
                    }
                }
            }
            _ => {}
        }
    }
    if !logits.iter().any(|x| x.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "chatterbox sampling: processors removed every candidate".into(),
        ));
    }
    Ok(())
}

#[must_use]
pub const fn processor_order(variant: ChatterboxVariant) -> &'static [Processor] {
    match variant {
        ChatterboxVariant::MultilingualV3 => &[
            Processor::RepetitionPenalty,
            Processor::Temperature,
            Processor::MinP,
            Processor::TopP,
        ],
        ChatterboxVariant::Nano | ChatterboxVariant::Turbo => &[
            Processor::Temperature,
            Processor::TopK,
            Processor::TopP,
            Processor::RepetitionPenalty,
        ],
    }
}

/// Caller-owned deterministic random contract. Values must be in [0,1).
#[derive(Debug, Clone)]
pub struct RandomDraws {
    draws: Vec<f32>,
    cursor: usize,
}

impl RandomDraws {
    pub fn new(draws: Vec<f32>) -> Result<Self> {
        if draws
            .iter()
            .any(|x| !x.is_finite() || *x < 0.0 || *x >= 1.0)
        {
            return Err(VokraError::InvalidArgument(
                "chatterbox random draw must be finite and in [0,1)".into(),
            ));
        }
        Ok(Self { draws, cursor: 0 })
    }
    pub fn next(&mut self) -> Result<f32> {
        let value = self.draws.get(self.cursor).copied().ok_or_else(|| {
            VokraError::InvalidArgument(
                "chatterbox generation: caller-owned random draws exhausted".into(),
            )
        })?;
        self.cursor += 1;
        Ok(value)
    }
}

/// Sample one token from logits with source-compatible caller-owned draw.
pub fn sample_with_draw(logits: &[f32], draw: f32) -> Result<u32> {
    if logits.is_empty() || !draw.is_finite() || !(0.0..1.0).contains(&draw) {
        return Err(VokraError::InvalidArgument(
            "chatterbox sampling: invalid logits or draw".into(),
        ));
    }
    let max = logits
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return Err(VokraError::InvalidArgument(
            "chatterbox sampling: all logits are non-finite".into(),
        ));
    }
    let mut sum = 0.0;
    let mut probs = Vec::with_capacity(logits.len());
    for &x in logits {
        let p = if x.is_finite() { (x - max).exp() } else { 0.0 };
        probs.push(p);
        sum += p;
    }
    if !(sum > 0.0) {
        return Err(VokraError::InvalidArgument(
            "chatterbox sampling: no finite probability mass".into(),
        ));
    }
    let target = draw * sum;
    let mut cumulative = 0.0;
    for (i, p) in probs.iter().enumerate() {
        cumulative += p;
        if target < cumulative {
            return Ok(i as u32);
        }
    }
    Ok((probs.len() - 1) as u32)
}

/// CFG combination used by the base multilingual upstream path.
pub fn combine_cfg_logits(cond: &[f32], uncond: &[f32], cfg_weight: f32) -> Result<Vec<f32>> {
    if cond.len() != uncond.len()
        || cond.is_empty()
        || !cfg_weight.is_finite()
        || cond.iter().chain(uncond).any(|value| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(
            "chatterbox CFG: conditional and unconditional logits must have equal non-empty finite shape".into(),
        ));
    }
    Ok(cond
        .iter()
        .zip(uncond)
        .map(|(&c, &u)| c + cfg_weight * (c - u))
        .collect())
}

/// Validate a caller-provided T3 prefix before the autoregressive loop.
pub fn validate_prefix(variant: ChatterboxVariant, text_len: usize, cond_len: usize) -> Result<()> {
    if text_len == 0 {
        return Err(VokraError::InvalidArgument(
            "chatterbox T3: text prefix is empty".into(),
        ));
    }
    if variant == ChatterboxVariant::MultilingualV3 {
        if text_len > BASE_LEARNED_TEXT_POSITIONS {
            return Err(VokraError::InvalidArgument(format!(
                "chatterbox multilingual v3: text positions {text_len} exceed {BASE_LEARNED_TEXT_POSITIONS}"
            )));
        }
    } else if text_len + cond_len + 1 > variant.architecture().positions {
        return Err(VokraError::InvalidArgument(format!(
            "chatterbox {}: prefix length {} exceeds authenticated context {}",
            variant.identity(),
            text_len + cond_len + 1,
            variant.architecture().positions
        )));
    }
    Ok(())
}

/// Validate the three independent learned-position tables used by base v3.
/// The backbone context and text/speech learned embeddings are distinct
/// limits; a combined prefix arithmetic check would be the wrong contract.
pub fn validate_base_position_lengths(
    backbone_positions: usize,
    text_positions: usize,
    speech_positions: usize,
    perceiver_rows: usize,
) -> Result<()> {
    if backbone_positions != BASE_BACKBONE_MAX_POSITION_EMBEDDINGS {
        return Err(VokraError::InvalidArgument(format!(
            "chatterbox multilingual v3: backbone max_position_embeddings must be {BASE_BACKBONE_MAX_POSITION_EMBEDDINGS}, got {backbone_positions}"
        )));
    }
    if text_positions != BASE_LEARNED_TEXT_POSITIONS {
        return Err(VokraError::InvalidArgument(format!(
            "chatterbox multilingual v3: learned text positions must be {BASE_LEARNED_TEXT_POSITIONS}, got {text_positions}"
        )));
    }
    if speech_positions != BASE_LEARNED_SPEECH_POSITIONS {
        return Err(VokraError::InvalidArgument(format!(
            "chatterbox multilingual v3: learned speech positions must be {BASE_LEARNED_SPEECH_POSITIONS}, got {speech_positions}"
        )));
    }
    if perceiver_rows != BASE_PERCEIVER_OUTPUT_ROWS {
        return Err(VokraError::InvalidArgument(format!(
            "chatterbox multilingual v3: Perceiver output rows must be {BASE_PERCEIVER_OUTPUT_ROWS}, got {perceiver_rows}"
        )));
    }
    Ok(())
}

/// Generation topology exposed to a native binder and parity harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationTopology {
    /// Whether CFG duplicates the conditional prefix (base only).
    pub cfg_batch: bool,
    /// Whether the first transformer call consumes the full prefix.
    pub initial_full_prefix: bool,
    /// Whether later calls consume one speech token each.
    pub incremental_kv: bool,
    /// Whether terminal EOS is omitted from the returned tokens.
    pub remove_eos: bool,
}

#[must_use]
pub const fn generation_topology(variant: ChatterboxVariant) -> GenerationTopology {
    match variant {
        ChatterboxVariant::MultilingualV3 => GenerationTopology {
            cfg_batch: true,
            initial_full_prefix: true,
            incremental_kv: true,
            remove_eos: false,
        },
        ChatterboxVariant::Nano | ChatterboxVariant::Turbo => GenerationTopology {
            cfg_batch: false,
            initial_full_prefix: true,
            incremental_kv: true,
            remove_eos: true,
        },
    }
}

/// Remove EOS from the returned Turbo/Nano sequence exactly as upstream does.
#[must_use]
pub fn remove_terminal_eos(tokens: &mut Vec<u32>, eos: u32) {
    if tokens.last().copied() == Some(eos) {
        tokens.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axes_are_source_faithful_and_gpt2_is_not_llama() {
        let base = ChatterboxVariant::MultilingualV3.architecture();
        assert_eq!(
            (base.layers, base.hidden, base.ffn, base.text_vocab),
            (30, 1024, 4096, 2454)
        );
        let nano = ChatterboxVariant::Nano.architecture();
        assert_eq!(nano.backbone, Backbone::Gpt2Small);
        assert_eq!(
            (nano.layers, nano.hidden, nano.ffn, nano.positions),
            (12, 768, 3072, 8196)
        );
        let turbo = ChatterboxVariant::Turbo.architecture();
        assert_eq!(turbo.backbone, Backbone::Gpt2Medium);
        assert_eq!(
            (turbo.layers, turbo.hidden, turbo.ffn, turbo.positions),
            (24, 1024, 4096, 8196)
        );
    }

    #[test]
    fn conditioning_order_and_variant_boundaries_are_exact() {
        let spk = [1.0, 2.0];
        let prompt = (0..BASE_PERCEIVER_OUTPUT_ROWS)
            .map(|row| vec![3.0 + row as f32, 4.0 + row as f32])
            .collect::<Vec<_>>();
        let emotion = [5.0, 6.0];
        let rows = assemble_conditioning(
            ChatterboxVariant::MultilingualV3,
            &spk,
            &prompt,
            Some(&emotion),
            false,
        )
        .unwrap();
        assert_eq!(
            rows.stages,
            vec![
                ConditioningStage::SpeakerProjection,
                ConditioningStage::PromptSpeechPerceiver,
                ConditioningStage::EmotionProjection
            ]
        );
        assert_eq!(
            rows.rows,
            std::iter::once(spk.to_vec())
                .chain(prompt.iter().cloned())
                .chain(std::iter::once(emotion.to_vec()))
                .collect::<Vec<_>>()
        );
        assert!(assemble_conditioning(ChatterboxVariant::Turbo, &spk, &[], None, false).is_ok());
        assert!(
            assemble_conditioning(ChatterboxVariant::Turbo, &spk, &prompt, None, false).is_err()
        );
        assert!(
            assemble_conditioning(
                ChatterboxVariant::MultilingualV3,
                &spk,
                &[],
                Some(&emotion),
                true
            )
            .is_err()
        );
        let short_prompt = vec![vec![3.0, 4.0]; BASE_PERCEIVER_OUTPUT_ROWS - 1];
        assert!(
            assemble_conditioning(
                ChatterboxVariant::MultilingualV3,
                &spk,
                &short_prompt,
                Some(&emotion),
                false
            )
            .is_err()
        );
    }

    #[test]
    fn punctuation_and_multilingual_fail_closed_contract() {
        assert_eq!(punc_norm_gpt2("hello… world"), "Hello, world.");
        assert_eq!(punc_norm_multilingual("hello... world"), "Hello, world.");
        assert_eq!(punc_norm_multilingual("你好。"), "你好。");
        assert_eq!(punc_norm_multilingual("こんにちは？"), "こんにちは？");
        assert_eq!(
            normalize_multilingual_input("Hello world", Some("en")).unwrap(),
            "[en]hello[SPACE]world"
        );
        assert!(normalize_multilingual_input("こんにちは", Some("ja")).is_err());
        assert!(matches!(
            MultilingualTokenizer::from_tokenizer_json(br#"{"model":{"type":"BPE"}}"#),
            Err(VokraError::UnsupportedOp(_))
        ));
    }

    #[test]
    fn eos_and_processor_contracts() {
        assert_eq!(
            processor_order(ChatterboxVariant::MultilingualV3),
            &[
                Processor::RepetitionPenalty,
                Processor::Temperature,
                Processor::MinP,
                Processor::TopP
            ]
        );
        assert_eq!(
            processor_order(ChatterboxVariant::Turbo),
            &[
                Processor::Temperature,
                Processor::TopK,
                Processor::TopP,
                Processor::RepetitionPenalty
            ]
        );
        let mut ids = vec![1, 2, 6562];
        remove_terminal_eos(&mut ids, 6562);
        assert_eq!(ids, vec![1, 2]);
        let cfg = combine_cfg_logits(&[2.0, 4.0], &[1.0, 3.0], 0.5).unwrap();
        assert_eq!(cfg, vec![2.5, 4.5]);
        let mut draws = RandomDraws::new(vec![0.0]).unwrap();
        assert_eq!(sample_with_draw(&[0.0, 1.0], draws.next().unwrap()), 0);
        assert!(draws.next().is_err());
    }

    #[test]
    fn base_position_tables_are_independent_and_exact() {
        assert_eq!(
            ChatterboxVariant::MultilingualV3.architecture().positions,
            BASE_BACKBONE_MAX_POSITION_EMBEDDINGS
        );
        validate_base_position_lengths(
            BASE_BACKBONE_MAX_POSITION_EMBEDDINGS,
            BASE_LEARNED_TEXT_POSITIONS,
            BASE_LEARNED_SPEECH_POSITIONS,
            BASE_PERCEIVER_OUTPUT_ROWS,
        )
        .unwrap();
        assert!(
            validate_base_position_lengths(
                BASE_BACKBONE_MAX_POSITION_EMBEDDINGS,
                BASE_LEARNED_TEXT_POSITIONS + 1,
                BASE_LEARNED_SPEECH_POSITIONS,
                BASE_PERCEIVER_OUTPUT_ROWS,
            )
            .is_err()
        );
        assert!(validate_prefix(ChatterboxVariant::MultilingualV3, 2050, 10000).is_ok());
        assert!(validate_prefix(ChatterboxVariant::MultilingualV3, 2051, 1).is_err());
    }

    #[test]
    fn gpt2_scanner_separates_letters_digits_and_contractions() {
        assert_eq!(gpt2_pieces("abc123").unwrap(), vec!["abc", "123"]);
        assert_eq!(
            gpt2_pieces("we're you'll").unwrap(),
            vec!["we", "'re", " you", "'ll"]
        );
        assert_eq!(
            gpt2_pieces("WE'RE YOU'LL").unwrap(),
            vec!["WE", "'RE", " YOU", "'LL"]
        );
        assert_eq!(gpt2_pieces("a\x01b").unwrap(), vec!["a", "\x01", "b"]);
        assert!(gpt2_pieces("é").is_err());
    }

    #[test]
    fn gpt2_bytes_to_unicode_uses_canonical_excluded_byte_ranks() {
        assert_eq!(gpt2_byte_to_unicode(0) as u32, 256);
        assert_eq!(gpt2_byte_to_unicode(32) as u32, 288);
        assert_eq!(gpt2_byte_to_unicode(127) as u32, 289);
        assert_eq!(gpt2_byte_to_unicode(160) as u32, 322);
        assert_eq!(gpt2_byte_to_unicode(173) as u32, 323);
        assert_eq!(gpt2_byte_to_unicode(174) as u32, 174);
    }

    #[test]
    fn transformers_threshold_semantics_keep_topk_ties_and_topp_boundary() {
        let mut topk = [3.0, 3.0, 1.0];
        apply_processors(
            ChatterboxVariant::Turbo,
            &mut topk,
            &[],
            SamplingConfig {
                temperature: 1.0,
                top_k: 1,
                top_p: 1.0,
                min_p: 0.0,
                repetition_penalty: 1.0,
            },
        )
        .unwrap();
        assert!(topk[0].is_finite() && topk[1].is_finite() && topk[2].is_infinite());

        let mut topp = [3.0, 2.0, 1.0];
        apply_processors(
            ChatterboxVariant::Turbo,
            &mut topp,
            &[],
            SamplingConfig {
                temperature: 1.0,
                top_k: 0,
                top_p: 0.5,
                min_p: 0.0,
                repetition_penalty: 1.0,
            },
        )
        .unwrap();
        assert!(
            topp[0].is_finite() && topp[1] == f32::NEG_INFINITY && topp[2] == f32::NEG_INFINITY
        );

        let mut tied = [0.0, 0.0];
        assert!(
            apply_processors(
                ChatterboxVariant::Turbo,
                &mut tied,
                &[],
                SamplingConfig {
                    temperature: 1.0,
                    top_k: 0,
                    top_p: 0.5,
                    min_p: 0.0,
                    repetition_penalty: 1.0,
                },
            )
            .is_err()
        );

        // The pinned TopP warper keeps one candidate even when f32 rounding
        // makes every cumulative tail comparison pass for a tiny positive
        // threshold.
        let mut tiny_top_p = [4.0, 0.0, -1.0];
        apply_processors(
            ChatterboxVariant::Turbo,
            &mut tiny_top_p,
            &[],
            SamplingConfig {
                temperature: 1.0,
                top_k: 0,
                top_p: f32::MIN_POSITIVE,
                min_p: 0.0,
                repetition_penalty: 1.0,
            },
        )
        .unwrap();
        assert!(tiny_top_p[0].is_finite());

        // Numeric equality, rather than bit equality, is the source-safe
        // tie relation: +0 and -0 cannot be split by an unverified boundary.
        let mut signed_zero_tie = [0.0, -0.0];
        assert!(
            apply_processors(
                ChatterboxVariant::Turbo,
                &mut signed_zero_tie,
                &[],
                SamplingConfig {
                    temperature: 1.0,
                    top_k: 0,
                    top_p: 0.5,
                    min_p: 0.0,
                    repetition_penalty: 1.0,
                },
            )
            .is_err()
        );
    }

    #[test]
    fn sampling_rejects_zero_temperature_and_deduplicates_history() {
        let mut positive_infinity = [f32::INFINITY, 0.0];
        assert!(
            apply_processors(
                ChatterboxVariant::Turbo,
                &mut positive_infinity,
                &[],
                SamplingConfig::default(),
            )
            .is_err()
        );
        let mut filtered = [f32::NEG_INFINITY, 0.0];
        assert!(
            apply_processors(
                ChatterboxVariant::Turbo,
                &mut filtered,
                &[],
                SamplingConfig::default(),
            )
            .is_ok()
        );

        let mut zero_temperature = [1.0, 0.0];
        assert!(
            apply_processors(
                ChatterboxVariant::Turbo,
                &mut zero_temperature,
                &[],
                SamplingConfig {
                    temperature: 0.0,
                    top_k: 0,
                    top_p: 1.0,
                    min_p: 0.0,
                    repetition_penalty: 1.0,
                },
            )
            .is_err()
        );

        let mut zero_top_p = [1.0, 0.0];
        assert!(
            apply_processors(
                ChatterboxVariant::Turbo,
                &mut zero_top_p,
                &[],
                SamplingConfig {
                    temperature: 1.0,
                    top_k: 0,
                    top_p: 0.0,
                    min_p: 0.0,
                    repetition_penalty: 1.0,
                },
            )
            .is_err()
        );

        let mut no_top_p = [1.0, 2.0];
        apply_processors(
            ChatterboxVariant::Turbo,
            &mut no_top_p,
            &[],
            SamplingConfig {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                repetition_penalty: 1.0,
            },
        )
        .unwrap();
        assert_eq!(no_top_p, [1.0, 2.0]);

        let mut low_top_p = [4.0, 0.0, -1.0];
        apply_processors(
            ChatterboxVariant::Turbo,
            &mut low_top_p,
            &[],
            SamplingConfig {
                temperature: 1.0,
                top_k: 0,
                top_p: 0.01,
                min_p: 0.0,
                repetition_penalty: 1.0,
            },
        )
        .unwrap();
        assert!(low_top_p[0].is_finite());

        let mut logits = [0.0, 4.0];
        apply_processors(
            ChatterboxVariant::Turbo,
            &mut logits,
            &[1, 1],
            SamplingConfig {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                repetition_penalty: 2.0,
            },
        )
        .unwrap();
        assert_eq!(logits, [0.0, 2.0]);
    }

    #[test]
    fn cfg_rejects_nonfinite_inputs() {
        assert!(combine_cfg_logits(&[f32::NAN], &[0.0], 1.0).is_err());
        assert!(combine_cfg_logits(&[0.0], &[f32::INFINITY], 1.0).is_err());
    }

    #[test]
    fn composite_evidence_rejects_unverified_manifest_strings() {
        let mut evidence = CompositeBinderEvidence {
            variant: ChatterboxVariant::MultilingualV3,
            source_revision: SOURCE_REVISION.to_owned(),
            checkpoint_filename: BASE_V3_CHECKPOINT.to_owned(),
            tensor_manifest_sha256: "0".repeat(64),
            includes_voice_encoder: true,
            includes_s3_tokenizer: true,
            includes_s3gen: true,
            includes_hift: true,
            includes_watermark: true,
        };
        assert!(evidence.require_complete_pcm().is_err());
        evidence.tensor_manifest_sha256 = "g".repeat(64);
        assert!(evidence.require_complete_pcm().is_err());
        evidence.tensor_manifest_sha256 = "a".repeat(64);
        assert!(evidence.require_complete_pcm().is_err());
    }
}
