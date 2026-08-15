//! The two-stage `tag → reorder → verbalize` pipeline.
//!
//! Transcribed from `runtime/processor/wetext_processor.cc`
//! (WeTextProcessing, Apache-2.0):
//!
//! ```cpp
//! std::string Processor::Tag(const std::string& input) {
//!   if (input.empty()) return "";
//!   return Compose(input, tagger_.get());
//! }
//!
//! std::string Processor::Verbalize(const std::string& input) {
//!   if (input.empty()) return "";
//!   TokenParser parser(parse_type_);
//!   std::string output = parser.Reorder(input);
//!   output = Compose(output, verbalizer_.get());
//!   output.erase(std::remove(output.begin(), output.end(), '\\0'), output.end());
//!   return output;
//! }
//!
//! std::string Processor::Normalize(const std::string& input) {
//!   return Verbalize(Tag(input));
//! }
//! ```
//!
//! Every one of those behaviours is reproduced here, including the two
//! empty-input short-circuits and the NUL strip on the verbalizer's output.
//!
//! # What is real and what is loud-partial
//!
//! - **Real, always compiled**: the tagged-token parser and `Reorder` (the
//!   whole middle stage — see [`super::token`]), the grammar container, the
//!   OpenFST header probe, and the format-gap diagnosis.
//! - **Real, behind the `vokra-wfst` feature**: the two FST stages, which reuse
//!   `vokra_core::decode::wfst`'s `Fst` / `Arc` / `TropicalWeight` and the
//!   byte-verified `read_openfst_vector` reader (see [`super::compose`]).
//! - **Loud-partial**: without the feature, or when the supplied grammar is
//!   outside the shape `read_openfst_vector` was byte-verified against, every
//!   FST-touching entry point returns [`VokraError::UnsupportedOp`] naming the
//!   exact gap. It never falls back to [`super::rule_based_english_cardinals`],
//!   and it never returns a partially-normalised string as if it were the
//!   grammar's output (FR-EX-08).

use vokra_core::error::{Result, VokraError};

use super::grammar::ItnGrammarSet;
use super::token::{ItnParseType, reorder};

#[cfg(feature = "vokra-wfst")]
use vokra_core::decode::wfst::{Fst, TropicalWeight, read_openfst_vector};

/// The compiled grammar pair, held only when the WFST feature is on.
#[cfg(feature = "vokra-wfst")]
#[derive(Debug)]
struct Compiled {
    tagger: Fst<TropicalWeight>,
    verbalizer: Fst<TropicalWeight>,
}

/// A ready-to-run WeTextProcessing pipeline over one grammar bundle.
///
/// Build it once and reuse it: with the `vokra-wfst` feature on,
/// [`ItnPipeline::new`] parses both compiled FSTs up front, which is the only
/// expensive step (a real `zh_itn` tagger is on the order of 10^5–10^6 states).
#[derive(Debug)]
pub struct ItnPipeline {
    grammars: ItnGrammarSet,
    #[cfg(feature = "vokra-wfst")]
    compiled: Compiled,
}

impl ItnPipeline {
    /// Builds a pipeline from a grammar bundle.
    ///
    /// The compatibility check runs in **both** builds: a grammar outside the
    /// shape Vokra's OpenFST reader was byte-verified against is unusable
    /// whether or not the FST stages are compiled in, so the same bundle is
    /// rejected with the same message either way. Keeping that answer
    /// feature-independent means a caller cannot be told "enable `vokra-wfst`"
    /// about a bundle that would still not work with the feature on.
    ///
    /// With the feature **on** this additionally parses both compiled FSTs and
    /// structurally validates them, so a malformed body also fails here rather
    /// than at first use.
    ///
    /// # Errors
    ///
    /// [`VokraError::UnsupportedOp`] when a grammar is outside the byte-verified
    /// OpenFST shape (see [`ItnGrammarSet::compatibility`]);
    /// [`VokraError::ModelLoad`] / [`VokraError::InvalidArgument`] when the
    /// reader or `Fst::validate` rejects the body (feature on only).
    pub fn new(grammars: ItnGrammarSet) -> Result<Self> {
        // Diagnose the header first: `read_openfst_vector` would also refuse,
        // but with a message about the reader rather than about the grammar
        // bundle and how to fix it.
        grammars.compatibility()?;
        #[cfg(feature = "vokra-wfst")]
        {
            let tagger = parse_one(grammars.tagger_bytes(), "tagger", grammars.parse_type())?;
            let verbalizer = parse_one(
                grammars.verbalizer_bytes(),
                "verbalizer",
                grammars.parse_type(),
            )?;
            Ok(Self {
                grammars,
                compiled: Compiled { tagger, verbalizer },
            })
        }
        #[cfg(not(feature = "vokra-wfst"))]
        {
            Ok(Self { grammars })
        }
    }

    /// The language + direction this pipeline runs.
    #[must_use]
    pub const fn parse_type(&self) -> ItnParseType {
        self.grammars.parse_type()
    }

    /// The grammar bundle backing this pipeline.
    #[must_use]
    pub const fn grammars(&self) -> &ItnGrammarSet {
        &self.grammars
    }

    /// Stage 1 — classification and raw-field tagging (upstream `Processor::Tag`).
    ///
    /// Returns the tagged-token stream, e.g.
    /// `cardinal { integer: "114005" }`. An empty input returns an empty string,
    /// exactly as upstream does.
    ///
    /// # Errors
    ///
    /// [`VokraError::UnsupportedOp`] when the `vokra-wfst` feature is off;
    /// otherwise the composition errors from [`super::compose`].
    pub fn tag(&self, input: &str) -> Result<String> {
        if input.is_empty() {
            return Ok(String::new());
        }
        self.compose_stage(input.as_bytes(), Stage::Tagger)
    }

    /// Stage 2 — reorder the tagged fields, then verbalize (upstream
    /// `Processor::Verbalize`).
    ///
    /// The reorder step is real and unconditional; only the FST composition
    /// after it needs the feature. Trailing NUL bytes are stripped from the
    /// composed output, mirroring upstream's `output.erase(std::remove(...,
    /// '\0'), ...)`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] when `tagged` is not a well-formed
    /// tagged-token stream; [`VokraError::UnsupportedOp`] when the `vokra-wfst`
    /// feature is off; otherwise the composition errors.
    pub fn verbalize(&self, tagged: &str) -> Result<String> {
        if tagged.is_empty() {
            return Ok(String::new());
        }
        let reordered = reorder(tagged, self.parse_type())?;
        let out = self.compose_stage(reordered.as_bytes(), Stage::Verbalizer)?;
        Ok(out.replace('\0', ""))
    }

    /// The whole pipeline: `Verbalize(Tag(input))` (upstream
    /// `Processor::Normalize`).
    ///
    /// For an inverse (ITN) bundle this is the spoken→written direction that
    /// turns `"one hundred fourteen thousand five"` into `"114005"`; for a
    /// forward (TN) bundle it is written→spoken.
    ///
    /// # Errors
    ///
    /// As [`Self::tag`] and [`Self::verbalize`].
    pub fn normalize(&self, input: &str) -> Result<String> {
        let tagged = self.tag(input)?;
        self.verbalize(&tagged)
    }

    /// The reorder stage on its own — real with or without the feature.
    ///
    /// Exposed because it is genuinely useful standalone (it is the only part
    /// of the pipeline that is pure string rewriting) and because it is what
    /// the feature-off build can still be tested against end to end.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] for a malformed tagged-token stream.
    pub fn reorder_tagged(&self, tagged: &str) -> Result<String> {
        reorder(tagged, self.parse_type())
    }

    #[cfg(feature = "vokra-wfst")]
    fn compose_stage(&self, input: &[u8], stage: Stage) -> Result<String> {
        let fst = match stage {
            Stage::Tagger => &self.compiled.tagger,
            Stage::Verbalizer => &self.compiled.verbalizer,
        };
        let outcome = super::compose::compose_shortest_path(input, fst)?;
        String::from_utf8(outcome.output).map_err(|e| {
            VokraError::UnsupportedOp(format!(
                "itn: the `{}` {stage} emitted a byte sequence that is not valid UTF-8: {e}. \
                 WeTextProcessing grammars transduce UTF-8 bytes, so a non-UTF-8 result means \
                 the grammar and the input disagree about encoding.",
                self.grammars.parse_type().prefix()
            ))
        })
    }

    #[cfg(not(feature = "vokra-wfst"))]
    fn compose_stage(&self, _input: &[u8], stage: Stage) -> Result<String> {
        Err(VokraError::UnsupportedOp(format!(
            "itn: the {stage} FST stage is not compiled in — `vokra-ops` was built without the \
             `vokra-wfst` feature, so `vokra_core::decode::wfst` (the OpenFST binary reader, the \
             tropical semiring and the `Fst` type this stage composes over) is absent. Rebuild \
             with `--features vokra-wfst`. The tagged-token parser and the `Reorder` stage \
             (`ItnPipeline::reorder_tagged`) are real in this build; only the two FST \
             compositions are gated. Grammar bundle: `{}` \
             (tagger {} bytes, verbalizer {} bytes). Upstream reference: \
             https://github.com/wenet-e2e/WeTextProcessing",
            self.grammars.parse_type().prefix(),
            self.grammars.tagger_bytes().len(),
            self.grammars.verbalizer_bytes().len()
        )))
    }
}

/// Which of the two grammars a composition is running against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    Tagger,
    Verbalizer,
}

impl std::fmt::Display for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tagger => f.write_str("tagger"),
            Self::Verbalizer => f.write_str("verbalizer"),
        }
    }
}

/// Parses one compiled grammar, attributing any reader error to the grammar it
/// came from.
#[cfg(feature = "vokra-wfst")]
fn parse_one(bytes: &[u8], which: &str, parse_type: ItnParseType) -> Result<Fst<TropicalWeight>> {
    let fst = read_openfst_vector(bytes).map_err(|e| {
        VokraError::ModelLoad(format!(
            "itn: failed to read the `{}` {which} grammar ({} bytes): {e}",
            parse_type.prefix(),
            bytes.len()
        ))
    })?;
    fst.validate().map_err(|e| {
        VokraError::InvalidArgument(format!(
            "itn: the `{}` {which} grammar is structurally invalid: {e}",
            parse_type.prefix()
        ))
    })?;
    Ok(fst)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::itn::grammar::{
        ARC_TYPE_STANDARD, FST_TYPE_VECTOR, VERIFIED_VERSION, test_good_grammar, test_openfst_bytes,
    };

    fn pipeline(parse_type: ItnParseType) -> ItnPipeline {
        let g = ItnGrammarSet::new(parse_type, test_good_grammar(), test_good_grammar()).unwrap();
        ItnPipeline::new(g).expect("the byte-verified 1-state grammar builds a pipeline")
    }

    #[test]
    fn parse_type_and_grammars_are_surfaced() {
        let p = pipeline(ItnParseType::ZhItn);
        assert_eq!(p.parse_type(), ItnParseType::ZhItn);
        assert_eq!(p.grammars().parse_type(), ItnParseType::ZhItn);
        assert!(!p.grammars().tagger_bytes().is_empty());
    }

    #[test]
    fn empty_input_short_circuits_both_stages() {
        // Upstream returns "" for an empty input WITHOUT touching the FST, so
        // this holds identically with the feature on or off.
        let p = pipeline(ItnParseType::EnItn);
        assert_eq!(p.tag("").unwrap(), "");
        assert_eq!(p.verbalize("").unwrap(), "");
        assert_eq!(p.normalize("").unwrap(), "");
    }

    #[test]
    fn reorder_stage_is_real_regardless_of_the_feature() {
        let p = pipeline(ItnParseType::ZhItn);
        let out = p
            .reorder_tagged(r#"date { month: "01" day: "28" year: "2002" }"#)
            .unwrap();
        assert_eq!(out, r#"date { year: "2002" month: "01" day: "28" }"#);
    }

    #[test]
    fn reorder_stage_rejects_a_malformed_tagged_stream() {
        let p = pipeline(ItnParseType::ZhItn);
        let Err(err) = p.reorder_tagged("cardinal {") else {
            panic!("expected an error for a malformed tagged stream");
        };
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err}");
    }

    #[test]
    fn verbalize_reports_a_malformed_tagged_stream_before_touching_the_fst() {
        // The parse error must win over the feature gate: a caller who passes
        // garbage should be told that, not told to enable a feature.
        let p = pipeline(ItnParseType::ZhItn);
        let Err(err) = p.verbalize("not a token stream") else {
            panic!("expected an error for a malformed tagged stream");
        };
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err}");
    }

    #[test]
    fn a_symbol_table_carrying_grammar_is_refused_with_the_exact_gap() {
        // pynini attaches byte symbol tables by default, so this is the shape a
        // real upstream bundle is most likely to arrive in.
        let bad = test_openfst_bytes(FST_TYPE_VECTOR, ARC_TYPE_STANDARD, VERIFIED_VERSION, 0x3);
        let g = ItnGrammarSet::new(ItnParseType::ZhItn, bad, test_good_grammar()).unwrap();
        // The gap is visible without building a pipeline at all...
        let Err(err) = g.compatibility() else {
            panic!("expected a compatibility gap for a symbol-table-carrying grammar");
        };
        let msg = format!("{err}");
        assert!(msg.contains("fstsymbols --clear_isymbols"), "{msg}");
        assert!(matches!(err, VokraError::UnsupportedOp(_)));

        // ...and building a pipeline refuses in BOTH builds, so a caller is
        // never told "enable vokra-wfst" about a bundle that would still not
        // work with the feature on.
        let bad2 = test_openfst_bytes(FST_TYPE_VECTOR, ARC_TYPE_STANDARD, VERIFIED_VERSION, 0x3);
        let g2 = ItnGrammarSet::new(ItnParseType::ZhItn, bad2, test_good_grammar()).unwrap();
        assert!(matches!(
            ItnPipeline::new(g2),
            Err(VokraError::UnsupportedOp(_))
        ));
        // A clean bundle still builds.
        let g3 = ItnGrammarSet::new(
            ItnParseType::ZhItn,
            test_good_grammar(),
            test_good_grammar(),
        )
        .unwrap();
        assert!(ItnPipeline::new(g3).is_ok());
    }

    /// The load-bearing honesty test: a pipeline that cannot run the grammar
    /// must ERROR, never quietly hand back rule-based output.
    #[test]
    fn broken_grammar_never_falls_back_to_the_rule_path() {
        let p = pipeline(ItnParseType::EnItn);
        // The scaffold grammar is a 1-state acceptor with no arcs, so it accepts
        // nothing but the empty string. Any real input must therefore fail —
        // with the feature ON because the composition finds no path, with the
        // feature OFF because the stage is not compiled in. Either way it must
        // NOT be the rule-based rewrite.
        let src = "one hundred fourteen thousand five";
        let rule_output = crate::itn::rule_based_english_cardinals(src).text;
        assert_eq!(rule_output, "114005", "the rule path does produce this");

        let Err(err) = p.normalize(src) else {
            panic!("a pipeline that cannot run the grammar must not return a normalised string");
        };
        let msg = format!("{err}");
        assert!(
            !msg.contains("114005"),
            "the error must not smuggle rule-based output: {msg}"
        );
    }

    #[cfg(not(feature = "vokra-wfst"))]
    #[test]
    fn feature_off_names_the_feature_and_the_stage() {
        let p = pipeline(ItnParseType::ZhItn);
        let Err(err) = p.tag("一百") else {
            panic!("expected a loud-partial when the wfst feature is off");
        };
        let msg = format!("{err}");
        assert!(msg.contains("vokra-wfst"), "{msg}");
        assert!(msg.contains("tagger"), "{msg}");
        assert!(msg.contains("zh_itn"), "{msg}");
        assert!(msg.contains("WeTextProcessing"), "{msg}");
        assert!(matches!(err, VokraError::UnsupportedOp(_)));

        // The verbalizer stage names itself, and only AFTER the reorder stage
        // has done its (real) work.
        let Err(err) = p.verbalize(r#"cardinal { integer: "5" }"#) else {
            panic!("expected a loud-partial from the verbalizer stage");
        };
        assert!(format!("{err}").contains("verbalizer"), "{err}");
    }

    #[cfg(feature = "vokra-wfst")]
    #[test]
    fn feature_on_runs_the_real_composition() {
        // `Fst` / `TropicalWeight` already reach here through the module-level
        // import; `Arc` and `Semiring` do not.
        use vokra_core::decode::wfst::{Arc, Semiring};

        // Build a real tagger: "5" -> `cardinal { integer: "5" }`, and a real
        // verbalizer: that tagged stream -> "5". Emitting the whole output on a
        // single arc keeps the fixture small while still exercising
        // read_openfst_vector -> compose -> reorder -> compose end to end.
        //
        // These FSTs are hand-built rather than read from bytes because the
        // byte-level reader is already verified against real OpenFST fixtures
        // in `vokra-core/tests/parity_wfst.rs`; re-deriving that here would be
        // a self-mirror.
        // A single-path transducer: consume every `input` byte emitting
        // nothing, then emit every `output` byte on epsilon-input arcs, then
        // terminate. Hand-built rather than read from bytes because the
        // byte-level reader is already verified against real OpenFST fixtures
        // in `vokra-core/tests/parity_wfst.rs` — re-deriving that here would be
        // a self-mirror that proves nothing about the real format.
        fn emit(input: &[u8], output: &[u8]) -> Fst<TropicalWeight> {
            let mut f = Fst::new();
            let s0 = f.add_state(TropicalWeight::zero());
            f.set_start(s0);
            let mut cur = s0;
            for &b in input {
                let next = f.add_state(TropicalWeight::zero());
                f.add_arc(
                    cur,
                    Arc {
                        ilabel: u32::from(b),
                        olabel: 0,
                        weight: TropicalWeight::new(0.0),
                        nextstate: next,
                    },
                )
                .unwrap();
                cur = next;
            }
            for &ob in output {
                let next = f.add_state(TropicalWeight::zero());
                f.add_arc(
                    cur,
                    Arc {
                        ilabel: 0,
                        olabel: u32::from(ob),
                        weight: TropicalWeight::new(0.0),
                        nextstate: next,
                    },
                )
                .unwrap();
                cur = next;
            }
            f.set_final(cur, TropicalWeight::new(0.0)).unwrap();
            f
        }

        let tagged = br#"cardinal { integer: "5" }"#;
        let tagger = emit(b"5", tagged);
        let verbalizer = emit(tagged, b"5");

        let g = ItnGrammarSet::new(
            ItnParseType::EnItn,
            test_good_grammar(),
            test_good_grammar(),
        )
        .unwrap();
        let p = ItnPipeline {
            grammars: g,
            compiled: Compiled { tagger, verbalizer },
        };

        assert_eq!(
            p.tag("5").unwrap(),
            r#"cardinal { integer: "5" }"#,
            "stage 1 must produce the tagged form"
        );
        assert_eq!(p.normalize("5").unwrap(), "5", "the full pipeline must run");
    }
}
