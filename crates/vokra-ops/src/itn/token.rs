//! The WeTextProcessing **tagged-token** grammar: parser, field-order tables,
//! and the `Reorder` rewrite that sits between the tagger and the verbalizer.
//!
//! This module is a **line-by-line transcription** of the upstream C++ runtime
//! file `runtime/processor/wetext_token_parser.cc` (+ its header) from
//! `github.com/wenet-e2e/WeTextProcessing` (**Apache-2.0**, verified via the
//! GitHub API 2026-08-15: `spdx_id = "Apache-2.0"`). Every constant table below
//! (`ZH_TN_ORDERS` / `JA_TN_ORDERS` / `EN_TN_ORDERS` / `ITN_ORDERS`,
//! `ASCII_LETTERS`, the whitespace set) is copied from that file rather than
//! recalled — CLAUDE.md「ハルシネーション厳禁」.
//!
//! # Why this half is NOT behind the `vokra-wfst` feature
//!
//! The two-stage pipeline is `tagger-FST → REORDER → verbalizer-FST`. The two
//! FST stages need the OpenFST machinery (and so the opt-in feature), but the
//! middle stage is pure string rewriting with no FST involved at all. Keeping
//! it unconditional means the default `cargo test --workspace` build actually
//! compiles and exercises it, instead of it disappearing into a feature that
//! nobody enables.
//!
//! # The tagged-token grammar (upstream `TokenParser::Parse`)
//!
//! ```text
//! stream := WS* token (WS+ token)* WS*
//! token  := KEY " { " field* "}"
//! field  := KEY ": \"" VALUE "\"" WS*
//! KEY    := [A-Za-z_]+
//! VALUE  := any run of characters up to an unescaped '"';
//!           a backslash escapes the next character and BOTH the backslash
//!           and the escaped character stay in the value verbatim.
//! ```
//!
//! A concrete ITN tagger output looks like:
//!
//! ```text
//! cardinal { integer: "114005" } char { value: "人" }
//! ```
//!
//! # Faithfulness quirks that are reproduced on purpose
//!
//! Two upstream behaviours look like bugs but are **load-bearing** — the
//! verbalizer FST was compiled against the exact byte stream the upstream
//! `Reorder` emits, so "fixing" either would silently desynchronise Vokra's
//! output from the grammar it is feeding:
//!
//! 1. **Duplicate keys are emitted twice.** `Token::Append` pushes the key onto
//!    an insertion-order vector *and* writes into a map. A repeated key
//!    therefore appears twice in the order vector while the map holds only the
//!    last value, so `Token::String` prints `key: "<last>"` twice. See
//!    [`Token::render`].
//! 2. **`preserve_order` is only consulted for token names that have an order
//!    table.** The upstream condition is
//!    `orders.count(name) > 0 && (no preserve_order || preserve_order != "true")`,
//!    so a token whose name is absent from the table always keeps insertion
//!    order regardless of the flag. Transcribed verbatim in [`Token::render`].
//!
//! A third upstream quirk is *dead code* rather than behaviour: the whitespace
//! set in `wetext_token_parser.cc` is
//! `{" ", "\t", "\n", "\r", "\x0b\x0c"}` — the last entry is a **two**-character
//! string, which can never compare equal to a single UTF-8 character, so it can
//! never match. It is transcribed as the four reachable entries plus this note.

use std::collections::BTreeMap;

use vokra_core::error::{Result, VokraError};

/// Which language + direction the upstream pipeline was compiled for.
///
/// The upstream C++ runtime derives this from the grammar **filename**
/// (`wetext_processor.cc`: it probes the tagger path for `zh_tn_` / `zh_itn_` /
/// `en_tn_` / `en_itn_` / `ja_tn_` / `ja_itn_` and `LOG(FATAL)`s otherwise).
/// Vokra carries it as explicit metadata in the GGUF instead of re-deriving it
/// from a path, because a renamed file must not silently pick the wrong field
/// order (FR-EX-08).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ItnParseType {
    /// Chinese **text normalization** (written → spoken). Uses `ZH_TN_ORDERS`.
    ZhTn,
    /// Chinese **inverse text normalization** (spoken → written). Uses `ITN_ORDERS`.
    ZhItn,
    /// English text normalization. Uses `EN_TN_ORDERS`.
    EnTn,
    /// English inverse text normalization. Uses `ITN_ORDERS`.
    EnItn,
    /// Japanese text normalization. Uses `JA_TN_ORDERS`.
    JaTn,
    /// Japanese inverse text normalization. Uses `ITN_ORDERS`.
    JaItn,
}

impl ItnParseType {
    /// The upstream grammar-bundle prefix (`tn/processor.py::build_fst(prefix)`
    /// and the substring the C++ runtime probes the tagger path for).
    #[must_use]
    pub const fn prefix(self) -> &'static str {
        match self {
            Self::ZhTn => "zh_tn",
            Self::ZhItn => "zh_itn",
            Self::EnTn => "en_tn",
            Self::EnItn => "en_itn",
            Self::JaTn => "ja_tn",
            Self::JaItn => "ja_itn",
        }
    }

    /// The ISO-639-1 language part (`zh` / `en` / `ja`).
    #[must_use]
    pub const fn language(self) -> &'static str {
        match self {
            Self::ZhTn | Self::ZhItn => "zh",
            Self::EnTn | Self::EnItn => "en",
            Self::JaTn | Self::JaItn => "ja",
        }
    }

    /// The direction part (`tn` = written→spoken, `itn` = spoken→written).
    #[must_use]
    pub const fn direction(self) -> &'static str {
        match self {
            Self::ZhTn | Self::EnTn | Self::JaTn => "tn",
            Self::ZhItn | Self::EnItn | Self::JaItn => "itn",
        }
    }

    /// `true` for the three inverse (spoken→written) variants — the ones that
    /// turn `"one hundred fourteen thousand five"` into `"114005"`.
    #[must_use]
    pub const fn is_inverse(self) -> bool {
        matches!(self, Self::ZhItn | Self::EnItn | Self::JaItn)
    }

    /// Parses a bundle prefix (`"zh_itn"`, `"en-tn"`, `"ja_itn"`, …).
    ///
    /// Hyphens are accepted as separators alongside underscores and the input
    /// is lower-cased, because the CLI slug and the upstream directory name
    /// disagree on separator style. Anything else is [`None`] — this never
    /// falls back to a default (FR-EX-08: a typo'd language must not silently
    /// select Chinese field orders).
    #[must_use]
    pub fn from_prefix(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().replace('-', "_").as_str() {
            "zh_tn" => Some(Self::ZhTn),
            "zh_itn" => Some(Self::ZhItn),
            "en_tn" => Some(Self::EnTn),
            "en_itn" => Some(Self::EnItn),
            "ja_tn" => Some(Self::JaTn),
            "ja_itn" => Some(Self::JaItn),
            _ => None,
        }
    }

    /// Builds the parse type from a separate language + direction pair, the
    /// shape the GGUF metadata stores.
    ///
    /// Returns [`None`] for an unknown language or direction rather than
    /// guessing.
    #[must_use]
    pub fn from_language_direction(language: &str, direction: &str) -> Option<Self> {
        let lang = language.to_ascii_lowercase();
        let dir = direction.to_ascii_lowercase();
        match (lang.as_str(), dir.as_str()) {
            ("zh", "tn") => Some(Self::ZhTn),
            ("zh", "itn") => Some(Self::ZhItn),
            ("en", "tn") => Some(Self::EnTn),
            ("en", "itn") => Some(Self::EnItn),
            ("ja", "tn") => Some(Self::JaTn),
            ("ja", "itn") => Some(Self::JaItn),
            _ => None,
        }
    }

    /// Every parse type, for exhaustive iteration in tests and CLI help.
    #[must_use]
    pub const fn all() -> [Self; 6] {
        [
            Self::ZhTn,
            Self::ZhItn,
            Self::EnTn,
            Self::EnItn,
            Self::JaTn,
            Self::JaItn,
        ]
    }

    /// The field-order table this parse type reorders tokens with.
    ///
    /// Transcribed from `TokenParser::TokenParser(ParseType)`: all three
    /// inverse variants share the single `ITN_ORDERS` table, while each
    /// forward (TN) language has its own.
    #[must_use]
    pub const fn orders(self) -> &'static [(&'static str, &'static [&'static str])] {
        match self {
            Self::ZhTn => ZH_TN_ORDERS,
            Self::JaTn => JA_TN_ORDERS,
            Self::EnTn => EN_TN_ORDERS,
            Self::ZhItn | Self::EnItn | Self::JaItn => ITN_ORDERS,
        }
    }
}

/// `ZH_TN_ORDERS` — verbatim from `wetext_token_parser.cc`.
pub const ZH_TN_ORDERS: &[(&str, &[&str])] = &[
    ("date", &["year", "month", "day"]),
    ("fraction", &["denominator", "numerator"]),
    ("measure", &["denominator", "numerator", "value"]),
    ("money", &["value", "currency"]),
    ("time", &["noon", "hour", "minute", "second"]),
];

/// `JA_TN_ORDERS` — verbatim from `wetext_token_parser.cc`.
pub const JA_TN_ORDERS: &[(&str, &[&str])] = &[
    ("date", &["year", "month", "day"]),
    ("money", &["value", "currency"]),
];

/// `EN_TN_ORDERS` — verbatim from `wetext_token_parser.cc`.
///
/// Note that `date` lists `preserve_order` as its *first* field: in the
/// forward English grammar the flag is itself a printed field, not only a
/// control switch.
pub const EN_TN_ORDERS: &[(&str, &[&str])] = &[
    ("date", &["preserve_order", "text", "day", "month", "year"]),
    (
        "money",
        &[
            "integer_part",
            "fractional_part",
            "quantity",
            "currency_maj",
        ],
    ),
];

/// `ITN_ORDERS` — verbatim from `wetext_token_parser.cc`. Shared by the
/// Chinese, English and Japanese **inverse** grammars.
pub const ITN_ORDERS: &[(&str, &[&str])] = &[
    ("date", &["year", "month", "day", "preserve_order"]),
    ("fraction", &["sign", "numerator", "denominator"]),
    ("measure", &["numerator", "denominator", "value", "units"]),
    ("money", &["currency", "value", "decimal", "quantity"]),
    ("time", &["hour", "minute", "second", "noon", "zone"]),
    ("telephone", &["country_code", "number_part"]),
    ("electronic", &["username", "domain", "protocol"]),
];

/// One parsed tagged token: its name, its fields in insertion order, and the
/// field values.
///
/// Mirrors the upstream `struct Token` (name / order / members).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// Token name, e.g. `cardinal`, `date`, `char`.
    pub name: String,
    /// Field keys in **insertion** order. A duplicate key appears more than
    /// once here on purpose (see the module docs, quirk 1).
    pub order: Vec<String>,
    /// Field values, keyed by field name. A duplicate key keeps the LAST
    /// value, matching the upstream `members[key] = value` assignment.
    pub members: BTreeMap<String, String>,
}

impl Token {
    /// A new, field-less token.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            order: Vec::new(),
            members: BTreeMap::new(),
        }
    }

    /// Appends a field, exactly as upstream `Token::Append` does: the key is
    /// pushed onto the order vector *and* assigned into the member map, so a
    /// repeated key is recorded twice in the order and once (last-wins) in the
    /// map.
    pub fn append(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into();
        self.order.push(key.clone());
        self.members.insert(key, value.into());
    }

    /// Renders the token back to the tagged-string form the verbalizer FST
    /// expects — the transcription of upstream `Token::String`.
    ///
    /// `orders` is the parse type's field-order table. When the token name has
    /// an entry there **and** the token does not carry `preserve_order: "true"`,
    /// the table order replaces the insertion order; otherwise insertion order
    /// is kept. Keys listed in the table but absent from the token are skipped.
    #[must_use]
    pub fn render(&self, orders: &[(&str, &[&str])]) -> String {
        let mut out = String::with_capacity(self.name.len() + 8);
        out.push_str(&self.name);
        out.push_str(" {");

        let table = orders
            .iter()
            .find(|(n, _)| *n == self.name)
            .map(|(_, fields)| *fields);

        // Upstream condition, verbatim: the order table wins UNLESS the token
        // explicitly asks to preserve insertion order. Note the table lookup is
        // the outer guard, so `preserve_order` on a table-less token is inert.
        let preserve = self.members.get("preserve_order").map(String::as_str) == Some("true");
        let effective: Vec<&str> = match table {
            Some(fields) if !preserve => fields.to_vec(),
            _ => self.order.iter().map(String::as_str).collect(),
        };

        for key in effective {
            let Some(value) = self.members.get(key) else {
                continue;
            };
            out.push(' ');
            out.push_str(key);
            out.push_str(": \"");
            out.push_str(value);
            out.push('"');
        }
        out.push_str(" }");
        out
    }
}

/// The set of characters the upstream `ASCII_LETTERS` table accepts inside a
/// token name or a field key: `[A-Za-z_]`.
#[must_use]
pub const fn is_key_char(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

/// The reachable entries of upstream `UTF8_WHITESPACE`. (The upstream set also
/// contains the two-character string `"\x0b\x0c"`, which cannot equal any
/// single character and is therefore dead — see the module docs.)
const UTF8_WHITESPACE: [char; 4] = [' ', '\t', '\n', '\r'];

/// The characters upstream `Trim` strips (`WHITESPACE` in
/// `runtime/utils/wetext_string.cc` = `" \n\r\t\f\v"`).
const TRIM_WHITESPACE: [char; 6] = [' ', '\n', '\r', '\t', '\u{0c}', '\u{0b}'];

/// Parses a tagged-token stream into [`Token`]s.
///
/// Every malformed input is a loud [`VokraError::InvalidArgument`] naming the
/// character offset — the upstream throws `std::invalid_argument`, and a silent
/// "skip the bad token" would hand the verbalizer FST a stream it was never
/// compiled against (FR-EX-08).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] for an empty stream, a missing/invalid key,
/// a missing `" { "` / `": \""` / closing-quote delimiter, an unterminated
/// value or escape, or an unterminated token.
pub fn parse_tokens(input: &str) -> Result<Vec<Token>> {
    Parser::new(input)?.parse()
}

/// Applies the tagger→verbalizer middle stage: parse the tagged stream, rewrite
/// each token with `parse_type`'s field order, and join with single spaces.
///
/// This is upstream `TokenParser::Reorder` — the exact byte stream that is then
/// fed to the verbalizer FST.
///
/// # Errors
///
/// As [`parse_tokens`].
pub fn reorder(input: &str, parse_type: ItnParseType) -> Result<String> {
    let tokens = parse_tokens(input)?;
    let orders = parse_type.orders();
    let mut out = String::new();
    for token in &tokens {
        out.push_str(&token.render(orders));
        out.push(' ');
    }
    Ok(trim(&out).to_owned())
}

/// Upstream `Trim` (`Rtrim(Ltrim(s))`) over `" \n\r\t\f\v"`.
fn trim(s: &str) -> &str {
    s.trim_matches(|c| TRIM_WHITESPACE.contains(&c))
}

/// A character cursor with the upstream `Read` / `ParseChar` / `ParseChars`
/// semantics, including its end-of-stream sentinel behaviour.
///
/// Upstream keeps `text_` as a vector of UTF-8 character strings and signals
/// end-of-stream by assigning the literal `"<EOS>"` to the current character.
/// Rust models that with `Option<char>` (`None` == `<EOS>`), which also removes
/// the upstream footgun where a stream containing the literal text `<EOS>`
/// could be mistaken for the sentinel.
struct Parser {
    text: Vec<char>,
    index: usize,
    ch: Option<char>,
}

impl Parser {
    fn new(input: &str) -> Result<Self> {
        let text: Vec<char> = input.chars().collect();
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "itn: token stream must not be empty (upstream TokenParser::Load throws \
                 `token stream must not be empty`)"
                    .to_owned(),
            ));
        }
        let ch = Some(text[0]);
        Ok(Self { text, index: 0, ch })
    }

    /// Upstream `Read`: advance one character; at the last index the current
    /// character becomes the end-of-stream sentinel.
    fn read(&mut self) -> bool {
        if self.index + 1 < self.text.len() {
            self.index += 1;
            self.ch = Some(self.text[self.index]);
            true
        } else {
            self.ch = None;
            false
        }
    }

    /// Upstream `ParseWs`: skip spaces; `false` once the stream is exhausted.
    fn parse_ws(&mut self) -> bool {
        let mut not_eos = self.ch.is_some();
        while not_eos && self.ch == Some(' ') {
            not_eos = self.read();
        }
        not_eos
    }

    /// Upstream `ParseChar`.
    fn parse_char(&mut self, exp: char) -> bool {
        if self.ch == Some(exp) {
            self.read();
            true
        } else {
            false
        }
    }

    /// Upstream `ParseChars`: all-or-nothing, restoring the cursor on failure.
    fn parse_chars(&mut self, exp: &str) -> bool {
        let start = self.index;
        for c in exp.chars() {
            if !self.parse_char(c) {
                self.index = start;
                self.ch = Some(self.text[start]);
                return false;
            }
        }
        true
    }

    /// Upstream `ParseKey`.
    fn parse_key(&mut self) -> Result<String> {
        match self.ch {
            None => {
                return Err(VokraError::InvalidArgument(format!(
                    "itn: expected token key at end of stream (offset {})",
                    self.index
                )));
            }
            Some(c) if UTF8_WHITESPACE.contains(&c) => {
                return Err(VokraError::InvalidArgument(format!(
                    "itn: expected token key at offset {}, found whitespace {c:?}",
                    self.index
                )));
            }
            Some(_) => {}
        }
        let mut key = String::new();
        while let Some(c) = self.ch {
            if !is_key_char(c) {
                break;
            }
            key.push(c);
            self.read();
        }
        if key.is_empty() {
            return Err(VokraError::InvalidArgument(format!(
                "itn: invalid token key at offset {} — keys are [A-Za-z_]+ (upstream \
                 ASCII_LETTERS)",
                self.index
            )));
        }
        Ok(key)
    }

    /// Upstream `ParseValue`: read to the next unescaped `"`, keeping the
    /// backslash AND the escaped character in the value verbatim.
    fn parse_value(&mut self) -> Result<String> {
        if self.ch.is_none() {
            return Err(VokraError::InvalidArgument(
                "itn: expected token value at end of stream".to_owned(),
            ));
        }
        let mut value = String::new();
        loop {
            let Some(c) = self.ch else {
                return Err(VokraError::InvalidArgument(format!(
                    "itn: unterminated token value (no closing quote before end of stream, \
                     offset {})",
                    self.index
                )));
            };
            if c == '"' {
                return Ok(value);
            }
            value.push(c);
            let escape = c == '\\';
            self.read();
            if escape {
                let Some(esc) = self.ch else {
                    return Err(VokraError::InvalidArgument(format!(
                        "itn: unterminated escape in token value (offset {})",
                        self.index
                    )));
                };
                value.push(esc);
                self.read();
            }
        }
    }

    /// Upstream `TokenParser::Parse`.
    fn parse(mut self) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        while self.parse_ws() {
            let name = self.parse_key()?;
            if !self.parse_chars(" { ") {
                return Err(VokraError::InvalidArgument(format!(
                    "itn: expected token opening delimiter `{{` after `{name}` (offset {}) — \
                     the tagged form is `name {{ key: \"value\" }}`",
                    self.index
                )));
            }
            let mut token = Token::new(name.clone());
            let mut closed = false;
            while self.parse_ws() {
                if self.ch == Some('}') {
                    self.parse_char('}');
                    closed = true;
                    break;
                }
                let key = self.parse_key()?;
                if !self.parse_chars(": \"") {
                    return Err(VokraError::InvalidArgument(format!(
                        "itn: expected field delimiter `: \"` after key `{key}` in token \
                         `{name}` (offset {})",
                        self.index
                    )));
                }
                let value = self.parse_value()?;
                if !self.parse_char('"') {
                    return Err(VokraError::InvalidArgument(format!(
                        "itn: expected closing quote for field `{key}` in token `{name}` \
                         (offset {})",
                        self.index
                    )));
                }
                token.append(key, value);
            }
            if !closed {
                return Err(VokraError::InvalidArgument(format!(
                    "itn: unterminated token `{name}` (no closing `}}` before end of stream)"
                )));
            }
            tokens.push(token);
        }
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_type_prefix_language_direction_roundtrip() {
        for pt in ItnParseType::all() {
            assert_eq!(ItnParseType::from_prefix(pt.prefix()), Some(pt));
            assert_eq!(
                ItnParseType::from_language_direction(pt.language(), pt.direction()),
                Some(pt)
            );
            // `<lang>_<dir>` must reconstruct the prefix exactly — this is the
            // string the upstream C++ runtime probes the filename for.
            assert_eq!(format!("{}_{}", pt.language(), pt.direction()), pt.prefix());
        }
    }

    #[test]
    fn parse_type_rejects_unknown_without_defaulting() {
        assert_eq!(ItnParseType::from_prefix("fr_itn"), None);
        assert_eq!(ItnParseType::from_prefix("zh"), None);
        assert_eq!(ItnParseType::from_prefix(""), None);
        assert_eq!(ItnParseType::from_language_direction("fr", "itn"), None);
        assert_eq!(ItnParseType::from_language_direction("zh", "xx"), None);
    }

    #[test]
    fn hyphenated_and_uppercase_prefixes_are_accepted() {
        assert_eq!(
            ItnParseType::from_prefix("zh-itn"),
            Some(ItnParseType::ZhItn)
        );
        assert_eq!(
            ItnParseType::from_prefix("ZH_ITN"),
            Some(ItnParseType::ZhItn)
        );
    }

    #[test]
    fn all_three_inverse_variants_share_the_itn_order_table() {
        // Compared by VALUE, not by `std::ptr::eq`. `ITN_ORDERS` is a `const`,
        // so each use site gets its own inlined temporary and the addresses
        // legitimately differ even though the tables are identical — a
        // pointer-identity assertion here fails against a perfectly correct
        // implementation. The property that actually matters is that the
        // three inverse directions resolve to the same table, and that the
        // forward directions do not.
        for pt in [
            ItnParseType::ZhItn,
            ItnParseType::EnItn,
            ItnParseType::JaItn,
        ] {
            assert!(pt.is_inverse());
            assert_eq!(pt.orders(), ITN_ORDERS, "{pt:?} must use the ITN table");
        }
        for pt in [ItnParseType::ZhTn, ItnParseType::EnTn, ItnParseType::JaTn] {
            assert!(!pt.is_inverse());
            assert_ne!(
                pt.orders(),
                ITN_ORDERS,
                "{pt:?} is a forward direction and must NOT use the ITN table"
            );
        }
    }

    #[test]
    fn parses_a_single_token() {
        let toks = parse_tokens(r#"cardinal { integer: "114005" }"#).unwrap();
        assert_eq!(toks.len(), 1);
        assert_eq!(toks[0].name, "cardinal");
        assert_eq!(toks[0].order, vec!["integer".to_owned()]);
        assert_eq!(toks[0].members["integer"], "114005");
    }

    #[test]
    fn parses_multiple_tokens_and_multiple_fields() {
        let toks =
            parse_tokens(r#"date { month: "01" day: "28" year: "2002" } char { value: "の" }"#)
                .unwrap();
        assert_eq!(toks.len(), 2);
        assert_eq!(toks[0].order, vec!["month", "day", "year"]);
        assert_eq!(toks[1].name, "char");
        assert_eq!(toks[1].members["value"], "の");
    }

    #[test]
    fn parses_a_field_less_token() {
        let toks = parse_tokens("sil { }").unwrap();
        assert_eq!(toks.len(), 1);
        assert!(toks[0].order.is_empty());
        assert_eq!(toks[0].render(ITN_ORDERS), "sil { }");
    }

    #[test]
    fn escapes_keep_both_characters_verbatim() {
        // Upstream ParseValue appends the backslash AND the escaped char.
        let toks = parse_tokens(r#"char { value: "a\"b" }"#).unwrap();
        assert_eq!(toks[0].members["value"], r#"a\"b"#);
    }

    #[test]
    fn empty_stream_is_loud() {
        let Err(err) = parse_tokens("") else {
            panic!("expected an error for an empty token stream");
        };
        assert!(format!("{err}").contains("must not be empty"), "{err}");
    }

    #[test]
    fn missing_opening_delimiter_is_loud() {
        let Err(err) = parse_tokens(r#"cardinal{ integer: "1" }"#) else {
            panic!("expected an error for a missing ` {{ ` delimiter");
        };
        let msg = format!("{err}");
        assert!(msg.contains("opening delimiter"), "{msg}");
        assert!(msg.contains("cardinal"), "{msg}");
    }

    #[test]
    fn missing_field_delimiter_is_loud() {
        let Err(err) = parse_tokens(r#"cardinal { integer："1" }"#) else {
            panic!("expected an error for a missing `: \"` delimiter");
        };
        assert!(format!("{err}").contains("field delimiter"), "{err}");
    }

    #[test]
    fn unterminated_value_is_loud() {
        let Err(err) = parse_tokens(r#"cardinal { integer: "114005 }"#) else {
            panic!("expected an error for an unterminated value");
        };
        assert!(format!("{err}").contains("unterminated"), "{err}");
    }

    #[test]
    fn unterminated_token_is_loud() {
        let Err(err) = parse_tokens(r#"cardinal { integer: "1" "#) else {
            panic!("expected an error for an unterminated token");
        };
        let msg = format!("{err}");
        assert!(msg.contains("unterminated token"), "{msg}");
        assert!(msg.contains("cardinal"), "{msg}");
    }

    #[test]
    fn unterminated_escape_is_loud() {
        let Err(err) = parse_tokens(r#"char { value: "a\"#) else {
            panic!("expected an error for an unterminated escape");
        };
        assert!(format!("{err}").contains("escape"), "{err}");
    }

    #[test]
    fn invalid_key_is_loud() {
        let Err(err) = parse_tokens(r#"1cardinal { integer: "1" }"#) else {
            panic!("expected an error for a non-[A-Za-z_] key");
        };
        assert!(format!("{err}").contains("invalid token key"), "{err}");
    }

    #[test]
    fn reorder_applies_the_itn_date_order() {
        // Tagger emits insertion order month/day/year; ITN_ORDERS is
        // year/month/day/preserve_order.
        let out = reorder(
            r#"date { month: "01" day: "28" year: "2002" }"#,
            ItnParseType::ZhItn,
        )
        .unwrap();
        assert_eq!(out, r#"date { year: "2002" month: "01" day: "28" }"#);
    }

    #[test]
    fn reorder_honours_preserve_order_true() {
        let out = reorder(
            r#"date { month: "01" day: "28" preserve_order: "true" }"#,
            ItnParseType::ZhItn,
        )
        .unwrap();
        assert_eq!(
            out,
            r#"date { month: "01" day: "28" preserve_order: "true" }"#
        );
    }

    #[test]
    fn reorder_ignores_preserve_order_when_not_literally_true() {
        // Upstream compares against the literal string "true".
        let out = reorder(
            r#"date { month: "01" year: "2002" preserve_order: "yes" }"#,
            ItnParseType::ZhItn,
        )
        .unwrap();
        assert_eq!(
            out,
            r#"date { year: "2002" month: "01" preserve_order: "yes" }"#
        );
    }

    #[test]
    fn reorder_keeps_insertion_order_for_table_less_tokens() {
        let out = reorder(
            r#"cardinal { integer: "114005" sign: "-" }"#,
            ItnParseType::EnItn,
        )
        .unwrap();
        assert_eq!(out, r#"cardinal { integer: "114005" sign: "-" }"#);
    }

    #[test]
    fn preserve_order_is_inert_on_a_table_less_token() {
        // Quirk 2: the order-table lookup guards the whole condition upstream,
        // so `preserve_order` on a name with no table changes nothing.
        let with = reorder(
            r#"cardinal { integer: "5" preserve_order: "true" }"#,
            ItnParseType::EnItn,
        )
        .unwrap();
        assert_eq!(with, r#"cardinal { integer: "5" preserve_order: "true" }"#);
    }

    #[test]
    fn duplicate_keys_are_rendered_twice_last_value_wins() {
        // Quirk 1, transcribed on purpose: order holds the key twice, members
        // holds only the last value.
        let out = reorder(
            r#"cardinal { integer: "1" integer: "2" }"#,
            ItnParseType::EnItn,
        )
        .unwrap();
        assert_eq!(out, r#"cardinal { integer: "2" integer: "2" }"#);
    }

    #[test]
    fn reorder_drops_table_fields_that_are_absent() {
        let out = reorder(r#"time { minute: "02" hour: "5" }"#, ItnParseType::ZhItn).unwrap();
        assert_eq!(out, r#"time { hour: "5" minute: "02" }"#);
    }

    #[test]
    fn tn_and_itn_orders_differ_for_the_same_token() {
        let tagged = r#"money { currency: "$" value: "13.5" }"#;
        let itn = reorder(tagged, ItnParseType::ZhItn).unwrap();
        let tn = reorder(tagged, ItnParseType::ZhTn).unwrap();
        assert_eq!(itn, r#"money { currency: "$" value: "13.5" }"#);
        assert_eq!(tn, r#"money { value: "13.5" currency: "$" }"#);
        assert_ne!(itn, tn);
    }

    #[test]
    fn reorder_joins_tokens_with_single_spaces_and_trims() {
        let out = reorder(
            r#"  cardinal { integer: "1" }   char { value: "x" }  "#,
            ItnParseType::EnItn,
        )
        .unwrap();
        assert_eq!(out, r#"cardinal { integer: "1" } char { value: "x" }"#);
        assert!(!out.ends_with(' '));
        assert!(!out.starts_with(' '));
    }

    #[test]
    fn reorder_output_reparses_to_the_same_tokens() {
        // Reorder's output is the verbalizer's input, so it MUST be a valid
        // tagged stream itself.
        let src = r#"date { month: "01" day: "28" year: "2002" } char { value: "x" }"#;
        let once = reorder(src, ItnParseType::ZhItn).unwrap();
        let twice = reorder(&once, ItnParseType::ZhItn).unwrap();
        assert_eq!(once, twice, "reorder must be idempotent");
        assert_eq!(parse_tokens(&once).unwrap().len(), 2);
    }

    #[test]
    fn is_key_char_matches_upstream_ascii_letters() {
        assert!(is_key_char('a') && is_key_char('Z') && is_key_char('_'));
        assert!(!is_key_char('0') && !is_key_char(' ') && !is_key_char('-'));
    }
}
