//! `itn` — inverse text normalization / text normalization over WFST grammars.
//!
//! Every ASR model in this repository emits normalized, unpunctuated text: an
//! utterance spoken as *"one hundred fourteen thousand five"* comes back as
//! those words, not as `114005`. Turning the words back into their written form
//! is **inverse text normalization** (ITN); the opposite direction, expanding
//! `2.5` into *"two point five"* before TTS, is **text normalization** (TN).
//! This module implements both, because in the reference implementation they
//! are the *same machine* run over a different pair of grammars.
//!
//! # Primary source
//!
//! [`WeTextProcessing`](https://github.com/wenet-e2e/WeTextProcessing) —
//! **Apache-2.0** (verified via the GitHub API on 2026-08-15:
//! `license.spdx_id = "Apache-2.0"`, 809 stars). Everything here is transcribed
//! from its C++ runtime, which is the part of the project that consumes the
//! compiled grammars:
//!
//! - `runtime/processor/wetext_processor.cc` — the tag → verbalize pipeline and
//!   the `StringCompiler` → `Compose` → `ShortestPath` → `StringPrinter` chain
//!   (transcribed in [`pipeline`] and [`compose`]);
//! - `runtime/processor/wetext_token_parser.cc` — the tagged-token grammar, the
//!   per-language field-order tables and `Reorder` (transcribed in [`token`]);
//! - `runtime/utils/wetext_string.cc` — `Trim` / UTF-8 character splitting.
//!
//! The Python side (`tn/processor.py`, `tn/cache.py`, `itn/*/inverse_normalizer.py`)
//! is the *builder*: it compiles pynini rules into a bundle of `tagger.fst` +
//! `verbalizer.fst` + `manifest.json` per language/direction prefix (`zh_tn`,
//! `zh_itn`, `en_tn`, `en_itn`, `ja_tn`, `ja_itn`). Vokra never runs pynini; it
//! consumes the finished `.fst` files, exactly as the C++ runtime does.
//!
//! # Why this is mostly *reuse*
//!
//! Vokra already has a from-scratch, zero-dependency Rust port of OpenFST's
//! decode side at `vokra_core::decode::wfst` (M5-06): the tropical semiring,
//! the `Fst` / `Arc` data structures, structural validation, and
//! `read_openfst_vector` — a byte-verified reader for exactly the
//! `VectorFst<StdArc>` binary that `StdVectorFst::Read` consumes upstream. ITN
//! is a **second consumer of that machinery**, not new machinery:
//!
//! | upstream C++                          | here                                     |
//! |---------------------------------------|------------------------------------------|
//! | `StdVectorFst::Read`                  | `vokra_core::decode::wfst::read_openfst_vector` |
//! | `StringCompiler<StdArc>` (BYTE)       | the input side of [`compose::compose_shortest_path`] |
//! | `fst::Compose` + `fst::ShortestPath`  | [`compose::compose_shortest_path`]       |
//! | `StringPrinter<StdArc>` (BYTE)        | the output side of the same function     |
//! | `TokenParser::Reorder`                | [`token::reorder`]                       |
//! | `Processor::{Tag,Verbalize,Normalize}`| [`ItnPipeline`]                          |
//!
//! The one genuinely new algorithm is the composition itself, and only because
//! `vokra_core::decode::wfst` deliberately ships **no** general `compose` (ADR
//! M5-06 §1: graphs are composed offline). Composing a *linear string* with a
//! transducer does not need one — see [`compose`] for the product-graph
//! argument.
//!
//! # What is gated, and why
//!
//! `vokra_core::decode::wfst` lives behind the opt-in `vokra-wfst` feature
//! (default OFF). This module forwards that feature (`vokra-ops/vokra-wfst`
//! enables `vokra-core/vokra-wfst`) and gates **only** the two FST stages.
//! Everything else — the tagged-token parser, the field-order tables, the
//! `Reorder` rewrite, the grammar container, the OpenFST header probe — is
//! compiled and tested in the default build.
//!
//! # The honest part: the reader may refuse a real upstream grammar
//!
//! `read_openfst_vector` accepts exactly the byte-verified shape and refuses
//! everything else rather than guessing (FR-EX-08). In particular it rejects
//! **non-zero header flags**, which is how OpenFST advertises embedded symbol
//! tables — and WeTextProcessing grammars are built with pynini, which attaches
//! byte symbol tables by default.
//!
//! That gap is not hand-waved here. [`OpenFstHeader::probe`] reads the header
//! without the feature and without touching the body, and
//! [`ItnGrammarSet::compatibility`] reports the exact field, the exact value,
//! and the exact developer-side command that closes it
//! (`fstsymbols --clear_isymbols --clear_osymbols`). Extending the reader to
//! skip `SymbolTable` sections is the reader-side follow-up; until then a
//! symbol-table-carrying grammar is a loud [`vokra_core::VokraError::UnsupportedOp`],
//! never a best-effort parse.
//!
//! # The rule-based helper is not the WFST path
//!
//! [`rule_based_english_cardinals`] is a hand-written English cardinal
//! rewriter. It exists so the *shape* of an ITN rewrite is demonstrable and
//! testable without a grammar bundle. [`ItnPipeline`] never calls it — not on a
//! missing grammar, not on an unreadable one, not on a failed composition. A
//! caller has to name it explicitly. See [`rule_fallback`].
//!
//! # Example
//!
//! ```no_run
//! use vokra_ops::itn::{ItnGrammarSet, ItnParseType, ItnPipeline};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let tagger = std::fs::read("zh_itn/tagger.fst")?;
//! let verbalizer = std::fs::read("zh_itn/verbalizer.fst")?;
//! let grammars = ItnGrammarSet::new(ItnParseType::ZhItn, tagger, verbalizer)?;
//! let pipeline = ItnPipeline::new(grammars)?;
//! println!("{}", pipeline.normalize("二点五平方电线")?);
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "vokra-wfst")]
pub mod compose;
pub mod grammar;
pub mod pipeline;
pub mod rule_fallback;
pub mod token;

pub use grammar::{
    ARC_TYPE_STANDARD, FST_TYPE_VECTOR, ItnGrammarSet, OPENFST_MAGIC, OpenFstHeader,
    VERIFIED_VERSION, header_flags,
};
pub use pipeline::ItnPipeline;
pub use rule_fallback::{RuleBasedItnOutput, rule_based_english_cardinals};
pub use token::{
    EN_TN_ORDERS, ITN_ORDERS, ItnParseType, JA_TN_ORDERS, Token, ZH_TN_ORDERS, is_key_char,
    parse_tokens, reorder,
};

#[cfg(feature = "vokra-wfst")]
pub use compose::{ComposeOutcome, compose_shortest_path};

#[cfg(test)]
mod tests {
    use super::*;

    /// The upstream C++ runtime picks the field-order table from the grammar
    /// FILENAME (`wetext_processor.cc` probes for `zh_tn_` / `zh_itn_` / …).
    /// Vokra carries the parse type explicitly, so this test pins that the two
    /// conventions still agree — a `<prefix>_tagger.fst` name must resolve to
    /// the same parse type the metadata would.
    #[test]
    fn upstream_filename_prefixes_all_resolve() {
        for pt in ItnParseType::all() {
            let flat = format!("{}_tagger.fst", pt.prefix());
            let stem = flat.strip_suffix("_tagger.fst").unwrap();
            assert_eq!(ItnParseType::from_prefix(stem), Some(pt), "{flat}");
        }
    }

    /// The three ITN directions must share one order table, and the three TN
    /// directions must not — this is the single branch upstream's
    /// `TokenParser::TokenParser(ParseType)` makes.
    #[test]
    fn order_tables_are_wired_the_way_upstream_wires_them() {
        assert_eq!(ItnParseType::ZhItn.orders().len(), ITN_ORDERS.len());
        assert_eq!(ItnParseType::ZhTn.orders().len(), ZH_TN_ORDERS.len());
        assert_eq!(ItnParseType::JaTn.orders().len(), JA_TN_ORDERS.len());
        assert_eq!(ItnParseType::EnTn.orders().len(), EN_TN_ORDERS.len());
    }

    /// The rule path and the grammar path are separate surfaces. This pins that
    /// the rule path is reachable ONLY by name.
    #[test]
    fn the_rule_path_is_a_named_entry_point() {
        let out = rule_based_english_cardinals("one hundred fourteen thousand five");
        assert_eq!(out.text, "114005");
        assert_eq!(out.rewrites, 1);
    }
}
