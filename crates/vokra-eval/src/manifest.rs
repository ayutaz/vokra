//! Tiny dependency-free manifest reader for batch evaluation.
//!
//! A manifest is a UTF-8 text file of **records** separated by one or more
//! blank lines. Each record is a set of `key = value` lines (the same
//! `key = value` convention the parity fixtures use); a `#` starts a comment.
//! Which keys a record needs depends on the metric — the text metrics read
//! `hyp` and `ref`, the audio mel-loss reads `hyp_wav` and `ref_wav` (WAV
//! paths). Precomputed hypotheses are supplied here; running a model to
//! *produce* the hypothesis is a separate (dataset/GGUF-gated) concern.

use std::collections::HashMap;

/// One manifest record: a `key -> value` map plus the 1-based line where the
/// record started (for diagnostics).
#[derive(Debug, Clone)]
pub struct Record {
    fields: HashMap<String, String>,
    /// 1-based line number of the record's first field.
    pub line: usize,
}

impl Record {
    /// The value for `key`, if the record has it.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(String::as_str)
    }

    /// Number of key/value pairs in the record.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether the record has no fields.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

/// A parsed manifest: an ordered list of [`Record`]s.
#[derive(Debug, Clone, Default)]
pub struct Manifest {
    /// Records in file order.
    pub records: Vec<Record>,
}

impl Manifest {
    /// Parses a manifest from text. Blank lines separate records; `#` comments
    /// and surrounding whitespace are ignored. Consecutive `key = value` lines
    /// (no blank line between them) extend the same record.
    pub fn parse(text: &str) -> Self {
        let mut records = Vec::new();
        let mut cur: HashMap<String, String> = HashMap::new();
        let mut cur_line = 0usize;
        for (idx, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() {
                if !cur.is_empty() {
                    records.push(Record {
                        fields: std::mem::take(&mut cur),
                        line: cur_line,
                    });
                }
                continue;
            }
            if line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                if cur.is_empty() {
                    cur_line = idx + 1;
                }
                cur.insert(k.trim().to_owned(), v.trim().to_owned());
            }
        }
        if !cur.is_empty() {
            records.push(Record {
                fields: cur,
                line: cur_line,
            });
        }
        Self { records }
    }

    /// Loads and parses a manifest file.
    ///
    /// # Errors
    ///
    /// Propagates any I/O error from reading `path`.
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        Ok(Self::parse(&std::fs::read_to_string(path)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_records_split_by_blank_lines() {
        let text = "\
# a comment
hyp = the cat
ref = the cat sat

hyp = hello
ref = hello
";
        let man = Manifest::parse(text);
        assert_eq!(man.records.len(), 2);
        assert_eq!(man.records[0].get("hyp"), Some("the cat"));
        assert_eq!(man.records[0].get("ref"), Some("the cat sat"));
        assert_eq!(man.records[0].line, 2);
        assert_eq!(man.records[1].get("hyp"), Some("hello"));
        assert_eq!(man.records[1].len(), 2);
        assert!(!man.records[1].is_empty());
        assert_eq!(man.records[1].get("missing"), None);
    }

    #[test]
    fn multiple_blank_lines_do_not_create_empty_records() {
        let man = Manifest::parse("hyp = a\n\n\n\nhyp = b\n");
        assert_eq!(man.records.len(), 2);
    }

    #[test]
    fn trailing_record_without_final_blank_line_is_kept() {
        let man = Manifest::parse("hyp = only");
        assert_eq!(man.records.len(), 1);
        assert_eq!(man.records[0].get("hyp"), Some("only"));
    }
}
