//! SBV2 G2P wrapper: piper-plus 8-language G2P を SBV2 phoneme table に mapping.
//! (Clean-room comment: see mod.rs)

use std::collections::HashMap;
use vokra_core::Result;
use vokra_piper_plus::Phonemizer;

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
///
/// Two construction paths select the routing strategy:
///
/// - [`from_piper_g2p`](SbV2Phonemizer::from_piper_g2p) (Task 15) wires the
///   real piper-plus [`Phonemizer`] reuse boundary (M1-01-A,
///   `docs/piper-plus-integration.md` §7): input text is phonemized by the
///   injected `ja_g2p` / `en_g2p`, and the resulting piper-plus phoneme id
///   sequence is routed into SBV2 phoneme-table space through `ja_mapping` /
///   `en_mapping`.
/// - [`synthetic_for_test`](SbV2Phonemizer::synthetic_for_test) (Task 14)
///   uses a deterministic char-level mapping instead, so this crate's own
///   tests can prove the module wiring without depending on a real G2P
///   instance or a real SBV2 phoneme table.
pub struct SbV2Phonemizer {
    // Real piper-plus G2P (M1-01-A reuse boundary), when wired via
    // `from_piper_g2p`. `None` for `synthetic_for_test()` builds, where
    // `phonemize_ja`/`phonemize_en` fall back to the `*_char_mapping` tables
    // below instead.
    ja_g2p: Option<Box<dyn Phonemizer>>,
    en_g2p: Option<Box<dyn Phonemizer>>,
    // Piper-plus phoneme id -> (SBV2 phoneme id, tone) / SBV2 phoneme id,
    // for the real-G2P path above. Keyed by `i64` (the piper-plus voice's
    // OWN phoneme id) rather than a phoneme *symbol* string:
    // `Phonemizer::phonemize` (`crates/vokra-piper-plus/src/phonemizer.rs`)
    // returns only an already BOS/PAD/EOS-framed `Vec<i64>` id sequence —
    // the trait boundary never hands the caller a phoneme symbol string to
    // key a mapping on. A piper-plus id absent from these maps falls back
    // to `sbv2_default_phoneme_id` (a documented mapping fallback, not a
    // silent no-op — FR-EX-08).
    ja_mapping: HashMap<i64, (u16, u8)>,
    en_mapping: HashMap<i64, u16>,
    // Fallback SBV2 phoneme id for any input (piper-plus id or char) absent
    // from the active mapping table.
    sbv2_default_phoneme_id: u16,
    // Char -> (SBV2 phoneme id, tone) / SBV2 phoneme id, for the synthetic
    // path (`synthetic_for_test`). Renamed from `ja_mapping`/`en_mapping`
    // (Task 14) when Task 15 added the id-keyed real-G2P mapping above.
    ja_char_mapping: HashMap<char, (u16, u8)>,
    en_char_mapping: HashMap<char, u16>,
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
            ja_g2p: None,
            en_g2p: None,
            ja_mapping: HashMap::new(),
            en_mapping: HashMap::new(),
            sbv2_default_phoneme_id: 0,
            ja_char_mapping: ja,
            en_char_mapping: en,
        }
    }

    /// Wires the real piper-plus G2P (M1-01-A reuse boundary,
    /// `docs/piper-plus-integration.md` §7): `ja_g2p` / `en_g2p` phonemize
    /// the input text, and their (already BOS/PAD/EOS-framed) piper-plus
    /// phoneme id sequence is routed into SBV2 phoneme-table space through
    /// `ja_mapping` / `en_mapping`.
    ///
    /// A piper-plus id produced by `ja_g2p` / `en_g2p` that is absent from
    /// the corresponding mapping falls back to the default SBV2 phoneme id
    /// (`0`) — a documented mapping fallback, not a silent no-op. A `text`
    /// that `ja_g2p` / `en_g2p` itself cannot phonemize instead propagates
    /// that `Err` out of [`phonemize`](SbV2Phonemizer::phonemize): the
    /// real-G2P path never falls through to the synthetic char mapping
    /// (FR-EX-08).
    pub fn from_piper_g2p(
        ja_g2p: Box<dyn Phonemizer>,
        en_g2p: Box<dyn Phonemizer>,
        ja_mapping: HashMap<i64, (u16, u8)>,
        en_mapping: HashMap<i64, u16>,
    ) -> Self {
        Self {
            ja_g2p: Some(ja_g2p),
            en_g2p: Some(en_g2p),
            ja_mapping,
            en_mapping,
            sbv2_default_phoneme_id: 0,
            ja_char_mapping: HashMap::new(),
            en_char_mapping: HashMap::new(),
        }
    }

    /// Phonemize `text` under the given `language`, producing SBV2 phoneme
    /// ids, tones, word boundaries and the pass-through BERT input text.
    ///
    /// # Errors
    ///
    /// When wired via [`from_piper_g2p`](SbV2Phonemizer::from_piper_g2p),
    /// propagates any error the injected piper-plus [`Phonemizer`] returns.
    /// The synthetic char-mapping path
    /// ([`synthetic_for_test`](SbV2Phonemizer::synthetic_for_test)) never
    /// fails.
    pub fn phonemize(&self, text: &str, language: Language) -> Result<PhonemizeResult> {
        match language {
            Language::JA => self.phonemize_ja(text),
            Language::EN => self.phonemize_en(text),
        }
    }

    fn phonemize_ja(&self, text: &str) -> Result<PhonemizeResult> {
        match &self.ja_g2p {
            Some(g2p) => self.phonemize_ja_via_piper(g2p.as_ref(), text),
            None => self.phonemize_ja_char_mapping(text),
        }
    }

    /// Real-G2P JA path: routes `g2p`'s piper-plus phoneme id sequence
    /// through `ja_mapping` into SBV2 phoneme-table space.
    fn phonemize_ja_via_piper(&self, g2p: &dyn Phonemizer, text: &str) -> Result<PhonemizeResult> {
        let piper_ids = g2p.phonemize(text)?;
        let mut ids = Vec::with_capacity(piper_ids.len());
        let mut tones = Vec::with_capacity(piper_ids.len());
        let mut wb = Vec::with_capacity(piper_ids.len());
        for (i, piper_id) in piper_ids.iter().enumerate() {
            let (id, tone) = self
                .ja_mapping
                .get(piper_id)
                .copied()
                .unwrap_or((self.sbv2_default_phoneme_id, 0));
            ids.push(id);
            tones.push(tone);
            // The piper-plus id sequence carries only BOS/PAD/EOS framing
            // (`PhonemeTable::frame`), no word segmentation of its own.
            // Conservatively mark only the first emitted phoneme as a word
            // start.
            // TODO(Task 17-19): tighten word-boundary detection when text
            // encoder lands.
            wb.push(i == 0);
        }
        Ok(PhonemizeResult {
            phoneme_ids: ids,
            tones,
            word_boundaries: wb,
            bert_input_text: text.to_string(),
        })
    }

    /// Synthetic-mapping JA path (Task 14; test-only, see
    /// [`synthetic_for_test`](SbV2Phonemizer::synthetic_for_test)).
    fn phonemize_ja_char_mapping(&self, text: &str) -> Result<PhonemizeResult> {
        let mut ids = Vec::new();
        let mut tones = Vec::new();
        let mut wb = Vec::new();
        for c in text.chars() {
            let (id, tone) = self
                .ja_char_mapping
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
        match &self.en_g2p {
            Some(g2p) => self.phonemize_en_via_piper(g2p.as_ref(), text),
            None => self.phonemize_en_char_mapping(text),
        }
    }

    /// Real-G2P EN path: routes `g2p`'s piper-plus phoneme id sequence
    /// through `en_mapping` into SBV2 phoneme-table space. EN carries no
    /// pitch-accent tone (SBV2 has none for English), so `tones` is all `0`.
    fn phonemize_en_via_piper(&self, g2p: &dyn Phonemizer, text: &str) -> Result<PhonemizeResult> {
        let piper_ids = g2p.phonemize(text)?;
        let mut ids = Vec::with_capacity(piper_ids.len());
        let mut wb = Vec::with_capacity(piper_ids.len());
        for (i, piper_id) in piper_ids.iter().enumerate() {
            let id = self
                .en_mapping
                .get(piper_id)
                .copied()
                .unwrap_or(self.sbv2_default_phoneme_id);
            ids.push(id);
            // TODO(Task 17-19): tighten word-boundary detection when text
            // encoder lands (see `phonemize_ja_via_piper`).
            wb.push(i == 0);
        }
        let tones = vec![0u8; ids.len()];
        Ok(PhonemizeResult {
            phoneme_ids: ids,
            tones,
            word_boundaries: wb,
            bert_input_text: text.to_string(),
        })
    }

    /// Synthetic-mapping EN path (Task 14; test-only, see
    /// [`synthetic_for_test`](SbV2Phonemizer::synthetic_for_test)).
    fn phonemize_en_char_mapping(&self, text: &str) -> Result<PhonemizeResult> {
        let mut ids = Vec::new();
        let mut wb = Vec::new();
        // Tracks whether the next non-space char starts a new word. Starts
        // `true` so the first character of the input begins word 1.
        //
        // Space characters are consumed without pushing any `wb` entry —
        // the flag is carried forward and applied to the next real
        // character instead. This keeps `wb` and `ids` the same length by
        // construction (one entry per emitted phoneme), avoiding the
        // previous approach's per-space phantom push + tail-truncation
        // reconciliation, which misaligned boundaries for 3+ word inputs
        // (see `en_phonemize_multiword_word_boundaries_aligned` regression
        // test).
        let mut next_is_word_start = true;
        for c in text.to_ascii_lowercase().chars() {
            if c == ' ' {
                next_is_word_start = true;
                continue;
            }
            let id = self
                .en_char_mapping
                .get(&c)
                .copied()
                .unwrap_or(self.sbv2_default_phoneme_id);
            ids.push(id);
            wb.push(next_is_word_start);
            next_is_word_start = false;
        }
        let tones = vec![0u8; ids.len()];
        Ok(PhonemizeResult {
            phoneme_ids: ids,
            tones,
            word_boundaries: wb,
            bert_input_text: text.to_string(),
        })
    }
}
