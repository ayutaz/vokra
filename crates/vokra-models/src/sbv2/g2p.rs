//! SBV2 G2P wrapper: piper-plus 8-language G2P を SBV2 phoneme table に mapping.
//! (Clean-room comment: see mod.rs)

use std::collections::HashMap;
use vokra_core::Result;

/// SBV2 input language selector — drives which char-level mapping table
/// (and which tone convention) `SbV2Phonemizer::phonemize` uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    /// Japanese input (SBV2 pitch-accent tones, 0-2).
    JA,
    /// English input (tones are always 0 — SBV2 has no EN pitch accent).
    EN,
}

/// The G2P output for one input string: phoneme ids in SBV2 vocabulary
/// space, per-phoneme pitch-accent tones, per-phoneme word-boundary flags,
/// and the original text (fed separately to the BERT bridge).
pub struct PhonemizeResult {
    /// Phoneme ids in SBV2 phoneme-table space (one per output phoneme).
    pub phoneme_ids: Vec<u16>,
    /// Pitch-accent tone per phoneme: 0-2 for JA, all 0 for EN.
    pub tones: Vec<u8>,
    /// Word-boundary flag per phoneme (true = first phoneme of a word).
    pub word_boundaries: Vec<bool>,
    /// The original input text, passed through for the BERT bridge (Task 16+).
    pub bert_input_text: String,
}

/// SBV2 grapheme-to-phoneme wrapper: maps input text to the SBV2 phoneme
/// vocabulary (ids, tones, word boundaries) for the JA and EN language
/// families.
pub struct SbV2Phonemizer {
    // Piper-plus phonemizer references. Real land uses PassthroughPhonemizer +
    // out-of-workspace piper-plus-g2p per vokra-piper-plus's M1-01-B path.
    // For Phase 1 land, we use a simplified inline JA/EN mapper below and
    // wire the real piper-plus once the g2p bridge is available in-crate.
    ja_mapping: HashMap<char, (u16, u8)>, // char → (phoneme_id, tone)
    en_mapping: HashMap<char, u16>,
    sbv2_default_phoneme_id: u16,
}

impl SbV2Phonemizer {
    #[doc(hidden)]
    pub fn synthetic_for_test() -> Self {
        // 決定的な最小 mapping: JA hiragana → id 100+, EN letter → id 200+
        let mut ja = HashMap::new();
        for (i, c) in "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんこんにちは"
            .chars()
            .enumerate()
        {
            ja.insert(c, (100 + i as u16, (i % 3) as u8));
        }
        let mut en = HashMap::new();
        for (i, c) in "abcdefghijklmnopqrstuvwxyz ".chars().enumerate() {
            en.insert(c, 200 + i as u16);
        }
        Self {
            ja_mapping: ja,
            en_mapping: en,
            sbv2_default_phoneme_id: 0,
        }
    }

    /// Phonemize `text` under the given `language`, producing SBV2 phoneme
    /// ids, tones, word boundaries and the pass-through BERT input text.
    pub fn phonemize(&self, text: &str, language: Language) -> Result<PhonemizeResult> {
        match language {
            Language::JA => self.phonemize_ja(text),
            Language::EN => self.phonemize_en(text),
        }
    }

    fn phonemize_ja(&self, text: &str) -> Result<PhonemizeResult> {
        let mut ids = Vec::new();
        let mut tones = Vec::new();
        let mut wb = Vec::new();
        for c in text.chars() {
            let (id, tone) = self
                .ja_mapping
                .get(&c)
                .copied()
                .unwrap_or((self.sbv2_default_phoneme_id, 0));
            ids.push(id);
            tones.push(tone);
            wb.push(false);
        }
        // Mark first phoneme as word boundary start.
        if !wb.is_empty() {
            wb[0] = true;
        }
        Ok(PhonemizeResult {
            phoneme_ids: ids,
            tones,
            word_boundaries: wb,
            bert_input_text: text.to_string(),
        })
    }

    fn phonemize_en(&self, text: &str) -> Result<PhonemizeResult> {
        let mut ids = Vec::new();
        let mut wb = Vec::new();
        for c in text.to_ascii_lowercase().chars() {
            if c == ' ' {
                wb.push(true);
                continue;
            }
            let id = self
                .en_mapping
                .get(&c)
                .copied()
                .unwrap_or(self.sbv2_default_phoneme_id);
            ids.push(id);
            wb.push(false);
        }
        if !wb.is_empty() {
            wb[0] = true;
        }
        let tones = vec![0u8; ids.len()];
        // Pad wb to same length as ids (space boundaries collapsed).
        let wb = if wb.len() > ids.len() {
            wb[..ids.len()].to_vec()
        } else {
            let mut w = wb;
            w.resize(ids.len(), false);
            w
        };
        Ok(PhonemizeResult {
            phoneme_ids: ids,
            tones,
            word_boundaries: wb,
            bert_input_text: text.to_string(),
        })
    }
}
