//! G2P bridge trait boundary + a mock implementation (M0-07-T08).
//!
//! # Why a trait boundary
//!
//! The 8-language piper-plus G2P is *reused*, not reimplemented (client
//! decision 2026-07-02, FR-API-06). The reuse form — depending on the upstream
//! pure-Rust `piper-plus-g2p` crate, vendoring it, or a future in-tree port —
//! is not finalised (it is T04 client confirmation, see
//! `docs/piper-plus-integration.md` §7/§8). [`Phonemizer`] pins the interface
//! the native TTS path consumes (text → phoneme id sequence) so the concrete
//! G2P can be swapped behind it later (M4-09 / M5-09 re-evaluation) without
//! touching the model.
//!
//! # M0 scope: mock only
//!
//! This ships [`MockPhonemizer`], a deterministic CI scaffold that lets the
//! demo, integration tests and CI run end to end **without** the real G2P
//! (which is T09, blocked on the T04 confirmation of the reuse form). The mock
//! is not linguistically correct: it looks each input character up in the
//! voice's phoneme table and drops the rest. Numerical parity (M0-07-T21/T22)
//! is therefore run on **phoneme ids fed directly** — never through the mock —
//! exactly as the WP splits text→phoneme checking from phoneme→PCM checking.

use std::collections::HashMap;

use vokra_core::{Result, VokraError};

/// Maps a text string to a phoneme id sequence ready for the native
/// MB-iSTFT-VITS2 model (M0-07-T11..T20).
///
/// The returned ids index the voice's phoneme embedding table. The trait is
/// the swap point for the eventual real G2P reuse (T09 / M4-09); M0 ships only
/// [`MockPhonemizer`].
pub trait Phonemizer {
    /// Converts `text` to a phoneme id sequence (already wrapped with the
    /// voice's BOS/EOS/PAD framing).
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] if `text` cannot be mapped to any
    /// in-vocabulary phoneme.
    fn phonemize(&self, text: &str) -> Result<Vec<i64>>;
}

/// A voice's phoneme symbol → id table plus the special framing ids.
///
/// Built from the `vokra.piper.phoneme_symbols` GGUF metadata (id → symbol),
/// which the converter (M0-07-T07) transcribes from the piper-plus
/// `config.json` `phoneme_id_map`.
#[derive(Debug, Clone)]
pub struct PhonemeTable {
    symbol_to_id: HashMap<String, i64>,
    pad: i64,
    bos: i64,
    eos: i64,
}

impl PhonemeTable {
    /// piper-plus fixes PAD = `_`, BOS = `^`, EOS = `$` (see
    /// `piper/const.py`); their ids come from the voice table.
    const PAD_SYMBOL: &'static str = "_";
    const BOS_SYMBOL: &'static str = "^";
    const EOS_SYMBOL: &'static str = "$";

    /// Builds a table from an id-indexed symbol list (`symbols[id] = symbol`),
    /// as stored in `vokra.piper.phoneme_symbols`.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] if the PAD/BOS/EOS symbols are
    /// absent (the table would be unusable for framing).
    pub fn from_symbols(symbols: &[String]) -> Result<Self> {
        let mut symbol_to_id = HashMap::with_capacity(symbols.len());
        for (id, symbol) in symbols.iter().enumerate() {
            if !symbol.is_empty() {
                // First id wins; the piper map is a bijection so this is moot.
                symbol_to_id.entry(symbol.clone()).or_insert(id as i64);
            }
        }
        let lookup = |s: &str| -> Result<i64> {
            symbol_to_id.get(s).copied().ok_or_else(|| {
                VokraError::InvalidArgument(format!("phoneme table missing special symbol `{s}`"))
            })
        };
        let pad = lookup(Self::PAD_SYMBOL)?;
        let bos = lookup(Self::BOS_SYMBOL)?;
        let eos = lookup(Self::EOS_SYMBOL)?;
        Ok(Self {
            symbol_to_id,
            pad,
            bos,
            eos,
        })
    }

    /// Looks up a single phoneme symbol's id.
    pub fn id_of(&self, symbol: &str) -> Option<i64> {
        self.symbol_to_id.get(symbol).copied()
    }

    /// PAD id (`_`).
    pub fn pad(&self) -> i64 {
        self.pad
    }

    /// Wraps a phoneme-id sequence in piper-plus multilingual framing: BOS, then
    /// each id followed by a PAD, then EOS (mirrors
    /// `piper/voice.py` for `phoneme_type != openjtalk`).
    pub fn frame(&self, phoneme_ids: &[i64]) -> Vec<i64> {
        let mut out = Vec::with_capacity(phoneme_ids.len() * 2 + 2);
        out.push(self.bos);
        for &id in phoneme_ids {
            out.push(id);
            out.push(self.pad);
        }
        out.push(self.eos);
        out
    }
}

/// A deterministic mock G2P (CI scaffold — **not** linguistically correct).
///
/// Maps each input character to a phoneme id by looking the character up (as a
/// one-character symbol) in the voice's [`PhonemeTable`], skipping characters
/// with no table entry, then applies the voice's BOS/EOS/PAD framing. This is
/// enough to drive the demo and integration tests end to end while the real
/// G2P reuse (T09) is pending; it must never be used for parity.
#[derive(Debug, Clone)]
pub struct MockPhonemizer {
    table: PhonemeTable,
}

impl MockPhonemizer {
    /// Creates a mock over the voice's phoneme table.
    pub fn new(table: PhonemeTable) -> Self {
        Self { table }
    }
}

impl Phonemizer for MockPhonemizer {
    fn phonemize(&self, text: &str) -> Result<Vec<i64>> {
        let ids: Vec<i64> = text
            .chars()
            .filter_map(|c| {
                let mut buf = [0u8; 4];
                self.table.id_of(c.encode_utf8(&mut buf))
            })
            .collect();
        if ids.is_empty() {
            return Err(VokraError::InvalidArgument(
                "MockPhonemizer: no input character maps to a phoneme (mock G2P covers only literal phoneme symbols)".to_owned(),
            ));
        }
        Ok(self.table.frame(&ids))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny table: `_`=0 (PAD), `^`=1 (BOS), `$`=2 (EOS), then `a`/`i`.
    fn table() -> PhonemeTable {
        let symbols = vec![
            "_".to_owned(),
            "^".to_owned(),
            "$".to_owned(),
            "a".to_owned(),
            "i".to_owned(),
        ];
        PhonemeTable::from_symbols(&symbols).unwrap()
    }

    #[test]
    fn table_resolves_special_ids() {
        let t = table();
        assert_eq!(t.pad(), 0);
        assert_eq!(t.id_of("a"), Some(3));
        assert_eq!(t.id_of("z"), None);
    }

    #[test]
    fn from_symbols_requires_framing_symbols() {
        // Missing `$` (EOS) -> unusable table.
        let err = PhonemeTable::from_symbols(&["_".to_owned(), "^".to_owned()]).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn mock_frames_known_phonemes_and_drops_unknown() {
        let mock = MockPhonemizer::new(table());
        // "a?i" -> known a(3), i(4); '?' dropped. Framed: BOS a PAD i PAD EOS.
        let ids = mock.phonemize("a?i").unwrap();
        assert_eq!(ids, vec![1, 3, 0, 4, 0, 2]);
    }

    #[test]
    fn mock_errors_when_nothing_maps() {
        let mock = MockPhonemizer::new(table());
        assert!(matches!(
            mock.phonemize("xyz"),
            Err(VokraError::InvalidArgument(_))
        ));
    }
}
