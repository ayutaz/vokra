//! A small, deterministic, **rule-based** English cardinal rewriter.
//!
//! # Read this before using it
//!
//! This is **not** the WFST path and it is **not** WeTextProcessing. It is a
//! hand-written rule that recognises runs of English cardinal number words and
//! rewrites them as digits. It handles exactly one non-standard-word class
//! (cardinals) in exactly one language (English) — no dates, no money, no
//! measures, no fractions, no telephone numbers, no whitelist, no blacklist.
//!
//! It exists for one reason: `vokra-ops` should be able to demonstrate the
//! *shape* of an ITN rewrite (and be tested for it) on a machine that has no
//! compiled grammar bundle. It is not a substitute for the grammar.
//!
//! # It is never substituted for the grammar path
//!
//! [`super::ItnPipeline`] does not call anything in this module — not on a
//! missing grammar, not on an unparseable grammar, not on a composition
//! failure. Every one of those is a loud error (FR-EX-08). A caller who wants
//! this behaviour has to name [`rule_based_english_cardinals`] explicitly at
//! the call site, where the name says what it is. The test
//! `pipeline::tests::broken_grammar_never_falls_back_to_the_rule_path` pins
//! that.
//!
//! # Algorithm
//!
//! The classic scale accumulator over a fixed word table: units add into a
//! running `current`, `hundred` multiplies `current`, and `thousand` /
//! `million` / `billion` flush `current` into `total` at their scale. Worked
//! example, the one from the ASR motivation:
//!
//! ```text
//! "one hundred fourteen thousand five"
//!   one       current = 1
//!   hundred   current = 1 * 100          = 100
//!   fourteen  current = 100 + 14         = 114
//!   thousand  total   = 114 * 1000       = 114000 ; current = 0
//!   five      current = 5
//!   ------------------------------------------------------
//!   total + current                      = 114005
//! ```
//!
//! `and` is absorbed inside a run ("one hundred and five") but does not start
//! one. Anything the table does not know ends the run and passes through
//! untouched.

/// The outcome of a rule-based rewrite.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleBasedItnOutput {
    /// The rewritten text.
    pub text: String,
    /// How many distinct number-word runs were replaced with digits. `0` means
    /// the input passed through byte-for-byte.
    pub rewrites: usize,
}

/// A cardinal word: either a value to add, a multiplier, or a scale flush.
#[derive(Clone, Copy)]
enum Word {
    /// Adds to the current group (`one` … `ninety`).
    Unit(u64),
    /// Multiplies the current group (`hundred`).
    Hundred,
    /// Flushes the current group into the total at this scale.
    Scale(u64),
    /// The filler `and`, absorbed inside a run.
    And,
}

/// Looks up one lower-cased word in the cardinal table.
fn classify(word: &str) -> Option<Word> {
    // `"oh"` is deliberately NOT mapped to 0. Reading "oh" as a zero is a
    // *digit-sequence* convention ("four oh four" = 404), which this cardinal
    // accumulator does not implement — and "oh" is an extremely common English
    // interjection, so mapping it would corrupt ordinary prose.
    let unit = match word {
        "zero" => 0,
        "one" => 1,
        "two" => 2,
        "three" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        "eleven" => 11,
        "twelve" => 12,
        "thirteen" => 13,
        "fourteen" => 14,
        "fifteen" => 15,
        "sixteen" => 16,
        "seventeen" => 17,
        "eighteen" => 18,
        "nineteen" => 19,
        "twenty" => 20,
        "thirty" => 30,
        "forty" => 40,
        "fifty" => 50,
        "sixty" => 60,
        "seventy" => 70,
        "eighty" => 80,
        "ninety" => 90,
        "hundred" => return Some(Word::Hundred),
        "thousand" => return Some(Word::Scale(1_000)),
        "million" => return Some(Word::Scale(1_000_000)),
        "billion" => return Some(Word::Scale(1_000_000_000)),
        "and" => return Some(Word::And),
        _ => return None,
    };
    Some(Word::Unit(unit))
}

/// Splits `text` into whitespace-delimited tokens, keeping each token's
/// leading/trailing punctuation so it can be re-emitted verbatim.
fn core_of(token: &str) -> (&str, &str, &str) {
    let start = token
        .find(|c: char| c.is_alphanumeric())
        .unwrap_or(token.len());
    let end = token
        .rfind(|c: char| c.is_alphanumeric())
        .map_or(start, |i| {
            i + token[i..].chars().next().map_or(1, char::len_utf8)
        });
    (&token[..start], &token[start..end], &token[end..])
}

/// Rewrites runs of English cardinal number words in `text` as digits.
///
/// **This is the rule-based path, not the WFST grammar path.** See the module
/// docs. It never fails: unknown words pass through unchanged.
///
/// ```
/// use vokra_ops::itn::rule_based_english_cardinals;
///
/// let out = rule_based_english_cardinals("one hundred fourteen thousand five");
/// assert_eq!(out.text, "114005");
/// assert_eq!(out.rewrites, 1);
/// ```
#[must_use]
pub fn rule_based_english_cardinals(text: &str) -> RuleBasedItnOutput {
    // Tokenise once, splitting each token into (leading punctuation, core,
    // trailing punctuation) so a rewritten run can re-emit the punctuation the
    // last word carried.
    let raw: Vec<&str> = text.split_whitespace().collect();
    let parts: Vec<(&str, String, &str)> = raw
        .iter()
        .map(|t| {
            let (lead, core, trail) = core_of(t);
            (lead, core.to_ascii_lowercase(), trail)
        })
        .collect();

    let mut out: Vec<String> = Vec::with_capacity(raw.len());
    let mut rewrites = 0usize;
    let mut i = 0usize;

    while i < raw.len() {
        // A run may only START on a value word with no leading punctuation:
        // `and` never starts one, a bare `thousand` is a word not a number, and
        // `(one` keeps its bracket rather than becoming an ambiguous `(1`.
        let starts_run =
            parts[i].0.is_empty() && matches!(classify(&parts[i].1), Some(Word::Unit(_)));
        if !starts_run {
            out.push(raw[i].to_owned());
            i += 1;
            continue;
        }

        // Extend greedily over cardinal words with no leading punctuation.
        let mut run_end = i;
        while run_end < raw.len()
            && parts[run_end].0.is_empty()
            && classify(&parts[run_end].1).is_some()
        {
            run_end += 1;
        }
        // Trailing `and`s belong to the sentence, not to the number; they are
        // re-emitted verbatim below. The run starts on a `Unit`, so `end > i`
        // still holds after trimming.
        let mut end = run_end;
        while end > i && matches!(classify(&parts[end - 1].1), Some(Word::And)) {
            end -= 1;
        }

        // Scale accumulator over the run. Saturating arithmetic: a nonsense
        // input like "billion billion billion" must not wrap around silently.
        let mut total: u64 = 0;
        let mut current: u64 = 0;
        for (_, core, _) in &parts[i..end] {
            match classify(core) {
                Some(Word::Unit(v)) => current = current.saturating_add(v),
                Some(Word::Hundred) => current = current.saturating_mul(100),
                Some(Word::Scale(s)) => {
                    total = total.saturating_add(current.saturating_mul(s));
                    current = 0;
                }
                Some(Word::And) | None => {}
            }
        }
        let value = total.saturating_add(current);
        out.push(format!("{value}{}", parts[end - 1].2));
        rewrites += 1;

        // Re-emit the `and`s that were trimmed off the run's tail, verbatim.
        for token in &raw[end..run_end] {
            out.push((*token).to_owned());
        }
        i = run_end;
    }

    RuleBasedItnOutput {
        text: out.join(" "),
        rewrites,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_motivating_example() {
        let out = rule_based_english_cardinals("one hundred fourteen thousand five");
        assert_eq!(out.text, "114005");
        assert_eq!(out.rewrites, 1);
    }

    #[test]
    fn simple_cardinals() {
        for (src, want) in [
            ("zero", "0"),
            ("seven", "7"),
            ("twelve", "12"),
            ("twenty three", "23"),
            ("ninety nine", "99"),
            ("two thousand two", "2002"),
            ("three million", "3000000"),
        ] {
            assert_eq!(rule_based_english_cardinals(src).text, want, "src={src}");
        }
    }

    #[test]
    fn absorbs_and_inside_a_run_but_not_at_the_start() {
        assert_eq!(
            rule_based_english_cardinals("one hundred and five").text,
            "105"
        );
        // `and` alone is not a number run.
        let out = rule_based_english_cardinals("and then");
        assert_eq!(out.text, "and then");
        assert_eq!(out.rewrites, 0);
    }

    #[test]
    fn surrounding_words_pass_through() {
        let out = rule_based_english_cardinals("call me at forty two in the morning");
        assert_eq!(out.text, "call me at 42 in the morning");
        assert_eq!(out.rewrites, 1);
    }

    /// "oh" stays a word: it is not a cardinal, and mapping it to zero would
    /// rewrite the interjection in ordinary prose. See the note in `classify`.
    #[test]
    fn oh_is_not_read_as_zero() {
        let out = rule_based_english_cardinals("oh dear");
        assert_eq!(out.text, "oh dear");
        assert_eq!(out.rewrites, 0);
    }

    #[test]
    fn a_bare_digit_sequence_is_summed_as_a_cardinal_not_concatenated() {
        // Pins the documented boundary rather than a nicety: this is a scale
        // ACCUMULATOR, so "four two" is the cardinal 4 + 2 = 6, not the digit
        // string "42". Reading a run of bare digits positionally is the
        // telephone-number class, which the module docs put out of scope and
        // which belongs to the WFST grammar path.
        //
        // Worth a test of its own because the two readings are both plausible
        // to a reader and differ silently — nothing crashes, the number is
        // just wrong.
        let out = rule_based_english_cardinals("four two");
        assert_eq!(out.text, "6");
        assert_eq!(out.rewrites, 1);
    }

    #[test]
    fn multiple_runs_are_counted_separately() {
        let out = rule_based_english_cardinals("five apples and twelve pears");
        assert_eq!(out.text, "5 apples and 12 pears");
        assert_eq!(out.rewrites, 2);
    }

    #[test]
    fn text_without_numbers_is_untouched() {
        let out = rule_based_english_cardinals("the quick brown fox");
        assert_eq!(out.text, "the quick brown fox");
        assert_eq!(out.rewrites, 0);
    }

    #[test]
    fn trailing_punctuation_is_preserved() {
        let out = rule_based_english_cardinals("there were twelve, then more");
        assert_eq!(out.text, "there were 12, then more");
    }

    #[test]
    fn case_is_ignored() {
        assert_eq!(rule_based_english_cardinals("Twenty One").text, "21");
    }

    #[test]
    fn a_bare_scale_word_is_not_a_number() {
        // "thousand" with nothing before it is a word, not the value 1000.
        let out = rule_based_english_cardinals("a thousand words");
        assert_eq!(out.text, "a thousand words");
        assert_eq!(out.rewrites, 0);
    }

    #[test]
    fn empty_input_is_empty_output() {
        let out = rule_based_english_cardinals("");
        assert_eq!(out.text, "");
        assert_eq!(out.rewrites, 0);
    }

    #[test]
    fn core_of_splits_punctuation() {
        assert_eq!(core_of("twelve,"), ("", "twelve", ","));
        assert_eq!(core_of("(one"), ("(", "one", ""));
        assert_eq!(core_of("---"), ("---", "", ""));
    }
}
