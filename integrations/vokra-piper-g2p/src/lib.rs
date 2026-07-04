//! `PiperPlusG2p` — a [`vokra_piper_plus::Phonemizer`] backed by the real
//! 8-language [`piper_plus_g2p`] crate.
//!
//! This is the concrete G2P that the zero-dependency core deliberately does not
//! contain. It turns text into the exact `(phoneme ids, per-phoneme prosody,
//! language id)` triple the multilingual piper-plus models consume, then hands
//! it to Vokra's native TTS via [`vokra_piper_plus::Phonemizer::phonemize_full`]
//! → [`vokra_models::piper_plus::PiperPlusTts::synthesize_full`].
//!
//! # Faithfulness to piper-plus
//!
//! The pipeline mirrors piper-plus's own inference
//! (`piper_plus.api.PiperTTS._phonemize` / `piper-core::voice`):
//!
//! 1. Build a [`piper_plus_g2p::multilingual::MultilingualPhonemizer`] over the
//!    voice's languages (JA = bundled NAIST-JDIC, EN/others = rule/dict based),
//!    exactly as `piper-core::voice::create_language_g2p_phonemizer` does.
//! 2. `phonemize_with_prosody(text)` → clean IPA tokens + `(A1, A2, A3)` prosody.
//! 3. [`piper_plus_g2p::PiperEncoder`] `encode_with_prosody` inserts BOS/PAD/EOS
//!    and maps tokens (via PUA) to ids using the **voice's own** phoneme id map —
//!    reconstructed by inverting the GGUF `vokra.piper.phoneme_symbols` table, so
//!    the ids are byte-correct for that specific checkpoint.
//! 4. The language id is the voice's own index for the detected dominant
//!    language.
//!
//! The `[3 → 16]` `ProsodyProj` inside the native model is parity-checked
//! against onnxruntime with a non-zero prosody buffer
//! (`tests/parity/piper_plus_v7_prosody/`), so the prosody produced here is
//! consumed correctly.

use std::collections::HashMap;

use piper_plus_g2p::{PiperEncoder, UnknownTokenMode};
// Brings `phonemize_with_prosody` (a piper_plus_g2p trait method) into scope for
// the `MultilingualPhonemizer` value.
use piper_plus_g2p::phonemizer::Phonemizer as _;

use vokra_core::{Result, VokraError};
use vokra_models::piper_plus::PiperPlusTts;
use vokra_piper_plus::{PhonemizedUtterance, Phonemizer};

/// Real-G2P phonemizer bound to one voice's phoneme inventory and languages.
pub struct PiperPlusG2p {
    inner: piper_plus_g2p::multilingual::MultilingualPhonemizer,
    encoder: PiperEncoder,
    /// Language code → the voice's language id (index into `language_codes`).
    lang_to_lid: HashMap<String, i64>,
}

impl PiperPlusG2p {
    /// Builds the G2P for `voice`, reusing that voice's phoneme id map and
    /// language set so the emitted ids/lids match the checkpoint exactly.
    ///
    /// # Errors
    ///
    /// Fails if the voice's phoneme table cannot form a usable id map (e.g. it
    /// lacks the `_`/`^`/`$` framing symbols the encoder requires).
    pub fn from_voice(voice: &PiperPlusTts) -> Result<Self> {
        let cfg = voice.config();

        // Invert the voice's id-indexed symbol table (`vokra.piper.phoneme_symbols`,
        // itself the transcription of the checkpoint's `phoneme_id_map`) into the
        // symbol → [id] map the encoder wants. The multilingual piper map is a
        // bijection, so each symbol maps to a single id; PUA symbols are carried
        // verbatim (the encoder emits the same PUA chars).
        let mut id_map: piper_plus_g2p::PhonemeIdMap = HashMap::new();
        for (id, sym) in cfg.phoneme_symbols.iter().enumerate() {
            if !sym.is_empty() {
                id_map.entry(sym.clone()).or_insert_with(|| vec![id as i64]);
            }
        }
        let encoder = PiperEncoder::new(id_map, UnknownTokenMode::Skip)
            .map_err(|e| VokraError::InvalidArgument(format!("piper-plus-g2p encoder: {e}")))?;

        // One phonemizer per language the voice supports (index = language id).
        let languages: Vec<String> = cfg.language_codes.clone();
        let default_latin = if languages.iter().any(|l| l == "en") {
            "en".to_string()
        } else {
            languages
                .iter()
                .find(|l| matches!(l.as_str(), "es" | "fr" | "pt" | "sv"))
                .cloned()
                .unwrap_or_else(|| {
                    languages
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "en".to_string())
                })
        };
        let mut phonemizers: HashMap<String, Box<dyn piper_plus_g2p::phonemizer::Phonemizer>> =
            HashMap::new();
        for lang in &languages {
            phonemizers.insert(lang.clone(), build_lang_phonemizer(lang));
        }
        let inner = piper_plus_g2p::multilingual::MultilingualPhonemizer::new(
            languages.clone(),
            default_latin,
            phonemizers,
        );

        let lang_to_lid = languages
            .iter()
            .enumerate()
            .map(|(i, l)| (l.clone(), i as i64))
            .collect();

        Ok(Self {
            inner,
            encoder,
            lang_to_lid,
        })
    }
}

/// Constructs the per-language phonemizer, mirroring
/// `piper-core::voice::create_language_g2p_phonemizer`: real linguistics for the
/// languages with a dictionary/ruleset available, and a graceful
/// `PassthroughPhonemizer` fallback for any that cannot be built here (so an
/// unavailable dict degrades that language rather than failing construction).
fn build_lang_phonemizer(lang: &str) -> Box<dyn piper_plus_g2p::phonemizer::Phonemizer> {
    let passthrough = || {
        Box::new(piper_plus_g2p::multilingual::PassthroughPhonemizer::new(
            lang,
        )) as Box<_>
    };
    match lang {
        "ja" => piper_plus_g2p::japanese::JapanesePhonemizer::new_bundled()
            .map(|p| Box::new(p) as Box<dyn piper_plus_g2p::phonemizer::Phonemizer>)
            .unwrap_or_else(|_| passthrough()),
        "en" => piper_plus_g2p::english::EnglishPhonemizer::new()
            .map(|p| Box::new(p) as Box<dyn piper_plus_g2p::phonemizer::Phonemizer>)
            .unwrap_or_else(|_| passthrough()),
        "ko" => Box::new(piper_plus_g2p::korean::KoreanPhonemizer::new()),
        "es" => Box::new(piper_plus_g2p::spanish::SpanishPhonemizer::new()),
        "fr" => Box::new(piper_plus_g2p::french::FrenchPhonemizer::new()),
        "pt" => Box::new(piper_plus_g2p::portuguese::PortuguesePhonemizer::new()),
        "sv" => Box::new(piper_plus_g2p::swedish::SwedishPhonemizer::new()),
        // zh needs a loanword/pinyin dict that isn't shipped here; passthrough
        // keeps construction total. (JA/EN — the priority path — are exact.)
        _ => passthrough(),
    }
}

impl Phonemizer for PiperPlusG2p {
    fn phonemize(&self, text: &str) -> Result<Vec<i64>> {
        Ok(self.phonemize_full(text)?.ids)
    }

    fn phonemize_full(&self, text: &str) -> Result<PhonemizedUtterance> {
        let (tokens, prosody) = self
            .inner
            .phonemize_with_prosody(text)
            .map_err(|e| VokraError::InvalidArgument(format!("piper-plus-g2p phonemize: {e}")))?;
        let (ids, feats) = self
            .encoder
            .encode_with_prosody(&tokens, &prosody)
            .map_err(|e| VokraError::InvalidArgument(format!("piper-plus-g2p encode: {e}")))?;
        // `feats` are `[i32; 3]` aligned 1:1 with `ids` (BOS/PAD/EOS → [0,0,0]).
        let prosody = feats
            .iter()
            .map(|f| [f[0] as i64, f[1] as i64, f[2] as i64])
            .collect();
        // Language id = the voice's index for the detected dominant language.
        let lang = self.inner.detect_primary_language(text);
        let lid = self.lang_to_lid.get(lang).copied().unwrap_or(0);
        Ok(PhonemizedUtterance { ids, prosody, lid })
    }
}
