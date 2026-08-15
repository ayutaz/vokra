//! The grammar container and the OpenFST **header probe**.
//!
//! A WeTextProcessing bundle is two compiled OpenFST binaries — `tagger.fst`
//! and `verbalizer.fst` — plus the language/direction that selects the
//! field-order table. [`ItnGrammarSet`] is that triple, carried as raw bytes so
//! it can be stored in (and read straight back out of) a GGUF.
//!
//! # Why a second, header-only parser exists
//!
//! `vokra_core::decode::wfst::read_openfst_vector` parses the whole file, but
//! it lives behind the opt-in `vokra-wfst` feature and — deliberately — refuses
//! anything outside the byte-verified shape it was written against. That makes
//! it a poor diagnostician: a grammar that Vokra cannot consume should say
//! *which field* is out of range, at conversion time, in a build that does not
//! have the feature enabled at all.
//!
//! [`OpenFstHeader::probe`] therefore reads **only the fixed header** (it never
//! touches the body, so it is O(1) on a multi-megabyte grammar) and reports
//! every field verbatim. It is used by the converter to refuse a non-FST input
//! loudly, and by [`ItnGrammarSet::compatibility`] to name the exact format gap
//! when the full reader would reject the file.
//!
//! # Header layout and where the constants come from (NOT invented)
//!
//! The field order and the magic number are the ones already byte-verified in
//! this repository against a real OpenFST 1.8.4 `fstcompile` output — see the
//! table in `vokra-core/src/decode/wfst/reader.rs`. The magic is independently
//! confirmed against upstream OpenFST itself
//! (`src/include/fst/fst.h`: `constexpr int32 kFstMagicNumber = 2125659606`,
//! which is `0x7EB2_FDD6`).
//!
//! ```text
//! magic       u32   0x7EB2FDD6
//! fst_type    i32 length-prefixed UTF-8   ("vector")
//! arc_type    i32 length-prefixed UTF-8   ("standard" = tropical StdArc)
//! version     i32
//! flags       i32   bitmask, see `HeaderFlags`
//! properties  u64   (ignored for decode)
//! start       i64   (-1 = kNoStateId)
//! num_states  i64
//! num_arcs    i64   (VectorFst writes 0 here; per-state counts are authoritative)
//! ```
//!
//! The `flags` bit meanings are transcribed from upstream OpenFST
//! `src/include/fst/fst.h`, `class FstHeader::Flags`:
//! `HAS_ISYMBOLS = 0x1`, `HAS_OSYMBOLS = 0x2`, `IS_ALIGNED = 0x4`.

use vokra_core::error::{Result, VokraError};

use super::token::ItnParseType;

/// OpenFST file magic (`kFstMagicNumber` = 2125659606).
pub const OPENFST_MAGIC: u32 = 0x7EB2_FDD6;

/// The only `fst_type` Vokra's WFST reader accepts.
pub const FST_TYPE_VECTOR: &str = "vector";

/// The only `arc_type` Vokra's WFST reader accepts (tropical `StdArc`).
pub const ARC_TYPE_STANDARD: &str = "standard";

/// The single `VectorFst` header version byte-verified in this repository.
pub const VERIFIED_VERSION: i32 = 2;

/// `FstHeader::Flags` bit meanings, transcribed from upstream OpenFST
/// `src/include/fst/fst.h`.
pub mod header_flags {
    /// The file carries an embedded **input** symbol table.
    pub const HAS_ISYMBOLS: i32 = 0x1;
    /// The file carries an embedded **output** symbol table.
    pub const HAS_OSYMBOLS: i32 = 0x2;
    /// The body is memory-aligned.
    pub const IS_ALIGNED: i32 = 0x4;
}

/// The fixed header of an OpenFST binary, read without touching the body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenFstHeader {
    /// `fst_type` string, e.g. `"vector"` / `"const"`.
    pub fst_type: String,
    /// `arc_type` string, e.g. `"standard"` (tropical) / `"log"`.
    pub arc_type: String,
    /// Serialisation version.
    pub version: i32,
    /// Header flags bitmask — see [`header_flags`].
    pub flags: i32,
    /// Property bitmask (not needed for decode; recorded for diagnostics).
    pub properties: u64,
    /// Start state, or `-1` for OpenFST's `kNoStateId`.
    pub start: i64,
    /// Number of states.
    pub num_states: i64,
    /// The header's `num_arcs` field. `VectorFst` writes `0` here and stores
    /// per-state arc counts in the body, so this is **not** authoritative.
    pub num_arcs_header: i64,
    /// Byte length of the header itself (the body starts here).
    pub header_len: usize,
}

impl OpenFstHeader {
    /// Reads the fixed header of an OpenFST binary.
    ///
    /// This accepts *any* `fst_type` / `arc_type` / `version` / `flags` — it is
    /// a describer, not a gate. Use [`Self::vokra_reader_gap`] to find out
    /// whether Vokra's full reader can consume the file.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] when the buffer is too short for the header or
    /// the magic number does not match (i.e. it is not an OpenFST binary at
    /// all), or when a length-prefixed type string is negative / non-UTF-8.
    pub fn probe(bytes: &[u8]) -> Result<Self> {
        let mut c = HeadCursor { buf: bytes, pos: 0 };
        let magic = c.u32()?;
        if magic != OPENFST_MAGIC {
            return Err(VokraError::ModelLoad(format!(
                "itn: not an OpenFST binary — magic {magic:#010x} != {OPENFST_MAGIC:#010x} \
                 (upstream OpenFST `kFstMagicNumber` = 2125659606). Expected a compiled \
                 WeTextProcessing grammar (`tagger.fst` / `verbalizer.fst`)."
            )));
        }
        let fst_type = c.lp_string("fst_type")?;
        let arc_type = c.lp_string("arc_type")?;
        let version = c.i32()?;
        let flags = c.i32()?;
        let properties = c.u64()?;
        let start = c.i64()?;
        let num_states = c.i64()?;
        let num_arcs_header = c.i64()?;
        Ok(Self {
            fst_type,
            arc_type,
            version,
            flags,
            properties,
            start,
            num_states,
            num_arcs_header,
            header_len: c.pos,
        })
    }

    /// `true` if the header advertises an embedded input symbol table.
    #[must_use]
    pub const fn has_isymbols(&self) -> bool {
        self.flags & header_flags::HAS_ISYMBOLS != 0
    }

    /// `true` if the header advertises an embedded output symbol table.
    #[must_use]
    pub const fn has_osymbols(&self) -> bool {
        self.flags & header_flags::HAS_OSYMBOLS != 0
    }

    /// `true` if the header advertises memory-aligned body storage.
    #[must_use]
    pub const fn is_aligned(&self) -> bool {
        self.flags & header_flags::IS_ALIGNED != 0
    }

    /// Describes, in one sentence per gap, why Vokra's WFST reader cannot
    /// consume this file — or [`None`] when it can.
    ///
    /// This is the **honest** half of the ITN landing: the upstream grammars
    /// are produced by pynini, which routinely attaches byte symbol tables, and
    /// `read_openfst_vector` refuses any file with non-zero header flags rather
    /// than guessing at the trailing symbol-table sections. Rather than
    /// hand-waving that, the gap is named here with the exact field, the exact
    /// value, and the exact developer-side command that removes it.
    #[must_use]
    pub fn vokra_reader_gap(&self) -> Option<String> {
        if self.fst_type != FST_TYPE_VECTOR {
            return Some(format!(
                "fst_type is `{}`, but Vokra's reader only accepts `{FST_TYPE_VECTOR}` \
                 (a decode-only VectorFst). Re-emit with \
                 `fstconvert --fst_type=vector in.fst out.fst`.",
                self.fst_type
            ));
        }
        if self.arc_type != ARC_TYPE_STANDARD {
            return Some(format!(
                "arc_type is `{}`, but Vokra's reader only accepts the tropical \
                 `{ARC_TYPE_STANDARD}` (StdArc, f32 costs). The `log` semiring is a \
                 documented future additive in `vokra-core::decode::wfst::semiring` — \
                 re-emit with `fstconvert --fst_type=vector` from a standard-arc source, \
                 or wait for LogWeight.",
                self.arc_type
            ));
        }
        if self.version != VERIFIED_VERSION {
            return Some(format!(
                "VectorFst header version is {}, but only version {VERIFIED_VERSION} has \
                 been byte-verified in this repository (see the fixture table in \
                 `vokra-core/src/decode/wfst/reader.rs`). Vokra refuses unverified \
                 versions rather than guessing at a changed layout (FR-EX-08).",
                self.version
            ));
        }
        if self.flags != 0 {
            let mut carried = Vec::new();
            if self.has_isymbols() {
                carried.push("an input symbol table (HAS_ISYMBOLS 0x1)");
            }
            if self.has_osymbols() {
                carried.push("an output symbol table (HAS_OSYMBOLS 0x2)");
            }
            if self.is_aligned() {
                carried.push("aligned body storage (IS_ALIGNED 0x4)");
            }
            let unknown = self.flags
                & !(header_flags::HAS_ISYMBOLS
                    | header_flags::HAS_OSYMBOLS
                    | header_flags::IS_ALIGNED);
            if unknown != 0 {
                carried.push("unrecognised header flag bits");
            }
            return Some(format!(
                "header flags are {:#x} — the file carries {}. Vokra's \
                 `read_openfst_vector` only parses the flags==0 shape: it does not skip \
                 the trailing symbol-table sections, so it would mis-read the body. \
                 WeTextProcessing grammars are built with pynini, which attaches byte \
                 symbol tables by default. Strip them developer-side before conversion: \
                 `fstsymbols --clear_isymbols --clear_osymbols in.fst out.fst` \
                 (the labels are already plain UTF-8 byte values 1..=255, so nothing is \
                 lost). The alternative is to extend the reader to parse and skip \
                 SymbolTable sections — tracked as the reader-side follow-up.",
                self.flags,
                carried.join(" and ")
            ));
        }
        None
    }
}

/// A bounds-checked little-endian cursor over the fixed header only.
struct HeadCursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> HeadCursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or_else(|| {
            VokraError::ModelLoad("itn: OpenFST header cursor overflow".to_owned())
        })?;
        if end > self.buf.len() {
            return Err(VokraError::ModelLoad(format!(
                "itn: truncated OpenFST header — need {n} bytes at offset {}, only {} left \
                 (file is {} bytes)",
                self.pos,
                self.buf.len().saturating_sub(self.pos),
                self.buf.len()
            )));
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }

    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }

    fn lp_string(&mut self, which: &str) -> Result<String> {
        let len = self.i32()?;
        if len < 0 {
            return Err(VokraError::ModelLoad(format!(
                "itn: negative {which} string length {len} in OpenFST header"
            )));
        }
        let bytes = self.take(len as usize)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| {
            VokraError::ModelLoad(format!(
                "itn: non-UTF-8 {which} string in OpenFST header: {e}"
            ))
        })
    }
}

/// A WeTextProcessing grammar bundle: the two compiled FSTs plus the parse type.
///
/// Constructed from raw bytes (a GGUF blob, or files read off disk by the
/// converter). Construction **probes both headers** and refuses anything that
/// is not an OpenFST binary, so a mis-specified path fails at build time rather
/// than at first `normalize()`.
#[derive(Debug, Clone)]
pub struct ItnGrammarSet {
    parse_type: ItnParseType,
    tagger: Vec<u8>,
    verbalizer: Vec<u8>,
    tagger_header: OpenFstHeader,
    verbalizer_header: OpenFstHeader,
}

impl ItnGrammarSet {
    /// Builds a grammar set from the two compiled-FST byte blobs.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] when either blob is empty, is not an OpenFST
    /// binary (magic mismatch), or has a truncated header. The message names
    /// **which** of the two grammars is at fault.
    pub fn new(parse_type: ItnParseType, tagger: Vec<u8>, verbalizer: Vec<u8>) -> Result<Self> {
        if tagger.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "itn: `{}` tagger grammar is empty — expected a compiled OpenFST \
                 `tagger.fst` from a WeTextProcessing bundle",
                parse_type.prefix()
            )));
        }
        if verbalizer.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "itn: `{}` verbalizer grammar is empty — expected a compiled OpenFST \
                 `verbalizer.fst` from a WeTextProcessing bundle",
                parse_type.prefix()
            )));
        }
        let tagger_header = OpenFstHeader::probe(&tagger).map_err(|e| {
            VokraError::ModelLoad(format!(
                "itn: `{}` tagger grammar header is unreadable: {e}",
                parse_type.prefix()
            ))
        })?;
        let verbalizer_header = OpenFstHeader::probe(&verbalizer).map_err(|e| {
            VokraError::ModelLoad(format!(
                "itn: `{}` verbalizer grammar header is unreadable: {e}",
                parse_type.prefix()
            ))
        })?;
        Ok(Self {
            parse_type,
            tagger,
            verbalizer,
            tagger_header,
            verbalizer_header,
        })
    }

    /// The language + direction this bundle was compiled for.
    #[must_use]
    pub const fn parse_type(&self) -> ItnParseType {
        self.parse_type
    }

    /// The raw `tagger.fst` bytes.
    #[must_use]
    pub fn tagger_bytes(&self) -> &[u8] {
        &self.tagger
    }

    /// The raw `verbalizer.fst` bytes.
    #[must_use]
    pub fn verbalizer_bytes(&self) -> &[u8] {
        &self.verbalizer
    }

    /// The probed `tagger.fst` header.
    #[must_use]
    pub const fn tagger_header(&self) -> &OpenFstHeader {
        &self.tagger_header
    }

    /// The probed `verbalizer.fst` header.
    #[must_use]
    pub const fn verbalizer_header(&self) -> &OpenFstHeader {
        &self.verbalizer_header
    }

    /// `Ok(())` when Vokra's WFST reader can parse **both** grammars; otherwise
    /// a loud [`VokraError::UnsupportedOp`] naming the exact format gap and the
    /// developer-side command that closes it.
    ///
    /// # Errors
    ///
    /// [`VokraError::UnsupportedOp`] — see [`OpenFstHeader::vokra_reader_gap`].
    pub fn compatibility(&self) -> Result<()> {
        for (which, header) in [
            ("tagger", &self.tagger_header),
            ("verbalizer", &self.verbalizer_header),
        ] {
            if let Some(gap) = header.vokra_reader_gap() {
                return Err(VokraError::UnsupportedOp(format!(
                    "itn: the `{}` {which} grammar cannot be parsed by Vokra's OpenFST \
                     reader — {gap} Upstream reference: \
                     https://github.com/wenet-e2e/WeTextProcessing \
                     (runtime/processor/wetext_processor.cc reads these with \
                     `StdVectorFst::Read`).",
                    self.parse_type.prefix()
                )));
            }
        }
        Ok(())
    }
}

/// Test-only builder for a minimal, well-formed OpenFST `VectorFst<StdArc>`
/// binary: the fixed header plus a one-state, zero-arc, final body.
///
/// This is a **scaffold for rejection tests**, not a format oracle. The
/// positive proof that Vokra parses a real OpenFST binary lives in
/// `vokra-core/tests/parity_wfst.rs` against fixtures produced by real OpenFST
/// 1.8.4; building bytes here and parsing them back would be a self-mirror and
/// would verify nothing about the real format (numerical-parity discipline).
/// The layout it emits is the byte-verified one from
/// `vokra-core/src/decode/wfst/reader.rs`.
#[cfg(test)]
pub(crate) fn test_openfst_bytes(
    fst_type: &str,
    arc_type: &str,
    version: i32,
    flags: i32,
) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&OPENFST_MAGIC.to_le_bytes());
    v.extend_from_slice(&(fst_type.len() as i32).to_le_bytes());
    v.extend_from_slice(fst_type.as_bytes());
    v.extend_from_slice(&(arc_type.len() as i32).to_le_bytes());
    v.extend_from_slice(arc_type.as_bytes());
    v.extend_from_slice(&version.to_le_bytes());
    v.extend_from_slice(&flags.to_le_bytes());
    v.extend_from_slice(&0u64.to_le_bytes()); // properties
    v.extend_from_slice(&0i64.to_le_bytes()); // start = 0
    v.extend_from_slice(&1i64.to_le_bytes()); // num_states = 1
    v.extend_from_slice(&0i64.to_le_bytes()); // num_arcs (header) = 0
    v.extend_from_slice(&0.0f32.to_le_bytes()); // state 0 final weight
    v.extend_from_slice(&0i64.to_le_bytes()); // state 0 narcs
    v
}

/// Test-only shorthand for a well-formed, Vokra-readable grammar blob.
#[cfg(test)]
pub(crate) fn test_good_grammar() -> Vec<u8> {
    test_openfst_bytes(FST_TYPE_VECTOR, ARC_TYPE_STANDARD, VERIFIED_VERSION, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::test_openfst_bytes as header_bytes;

    fn good() -> Vec<u8> {
        test_good_grammar()
    }

    #[test]
    fn probe_reads_every_header_field() {
        let h = OpenFstHeader::probe(&good()).unwrap();
        assert_eq!(h.fst_type, "vector");
        assert_eq!(h.arc_type, "standard");
        assert_eq!(h.version, VERIFIED_VERSION);
        assert_eq!(h.flags, 0);
        assert_eq!(h.start, 0);
        assert_eq!(h.num_states, 1);
        assert_eq!(h.num_arcs_header, 0);
        // magic(4) + 4+6 + 4+8 + version(4) + flags(4) + props(8) + 3*i64(24)
        assert_eq!(h.header_len, 4 + 10 + 12 + 4 + 4 + 8 + 24);
        assert!(h.vokra_reader_gap().is_none());
    }

    #[test]
    fn magic_mismatch_is_loud() {
        let Err(err) = OpenFstHeader::probe(&[0u8; 64]) else {
            panic!("expected an error for a non-OpenFST buffer");
        };
        let msg = format!("{err}");
        assert!(msg.contains("magic"), "{msg}");
        assert!(msg.contains("2125659606"), "{msg}");
    }

    #[test]
    fn truncated_header_is_loud() {
        let bytes = OPENFST_MAGIC.to_le_bytes().to_vec();
        let Err(err) = OpenFstHeader::probe(&bytes) else {
            panic!("expected an error for a truncated header");
        };
        assert!(format!("{err}").contains("truncated"), "{err}");
    }

    #[test]
    fn symbol_table_flags_name_the_exact_gap_and_the_fix() {
        let bytes = header_bytes(FST_TYPE_VECTOR, ARC_TYPE_STANDARD, VERIFIED_VERSION, 0x3);
        let h = OpenFstHeader::probe(&bytes).unwrap();
        assert!(h.has_isymbols() && h.has_osymbols() && !h.is_aligned());
        let gap = h.vokra_reader_gap().expect("flags != 0 must be a gap");
        assert!(gap.contains("HAS_ISYMBOLS"), "{gap}");
        assert!(gap.contains("HAS_OSYMBOLS"), "{gap}");
        assert!(gap.contains("fstsymbols --clear_isymbols"), "{gap}");
    }

    #[test]
    fn aligned_flag_is_named_too() {
        let bytes = header_bytes(FST_TYPE_VECTOR, ARC_TYPE_STANDARD, VERIFIED_VERSION, 0x4);
        let h = OpenFstHeader::probe(&bytes).unwrap();
        assert!(h.is_aligned());
        assert!(
            h.vokra_reader_gap()
                .expect("aligned is a gap")
                .contains("IS_ALIGNED")
        );
    }

    #[test]
    fn non_vector_and_non_standard_and_bad_version_each_name_their_field() {
        let cases = [
            (
                header_bytes("const", ARC_TYPE_STANDARD, VERIFIED_VERSION, 0),
                "fst_type",
            ),
            (
                header_bytes(FST_TYPE_VECTOR, "log", VERIFIED_VERSION, 0),
                "arc_type",
            ),
            (
                header_bytes(FST_TYPE_VECTOR, ARC_TYPE_STANDARD, 99, 0),
                "version",
            ),
        ];
        for (bytes, needle) in cases {
            let gap = OpenFstHeader::probe(&bytes)
                .unwrap()
                .vokra_reader_gap()
                .unwrap_or_else(|| panic!("expected a gap naming `{needle}`"));
            assert!(gap.contains(needle), "gap `{gap}` should name `{needle}`");
        }
    }

    #[test]
    fn grammar_set_probes_both_headers() {
        let g = ItnGrammarSet::new(ItnParseType::ZhItn, good(), good()).unwrap();
        assert_eq!(g.parse_type(), ItnParseType::ZhItn);
        assert_eq!(g.tagger_header().num_states, 1);
        assert_eq!(g.verbalizer_header().num_states, 1);
        assert!(g.compatibility().is_ok());
    }

    #[test]
    fn empty_tagger_is_loud_and_named() {
        let Err(err) = ItnGrammarSet::new(ItnParseType::ZhItn, Vec::new(), good()) else {
            panic!("expected an error for an empty tagger grammar");
        };
        let msg = format!("{err}");
        assert!(msg.contains("tagger"), "{msg}");
        assert!(msg.contains("zh_itn"), "{msg}");
    }

    #[test]
    fn empty_verbalizer_is_loud_and_named() {
        let Err(err) = ItnGrammarSet::new(ItnParseType::EnItn, good(), Vec::new()) else {
            panic!("expected an error for an empty verbalizer grammar");
        };
        let msg = format!("{err}");
        assert!(msg.contains("verbalizer"), "{msg}");
        assert!(msg.contains("en_itn"), "{msg}");
    }

    #[test]
    fn non_fst_blob_is_refused_at_construction() {
        let Err(err) = ItnGrammarSet::new(ItnParseType::ZhItn, b"not an fst".to_vec(), good())
        else {
            panic!("expected an error for a non-OpenFST tagger blob");
        };
        assert!(format!("{err}").contains("tagger"), "{err}");
    }

    #[test]
    fn compatibility_points_at_the_offending_grammar() {
        let bad = header_bytes(FST_TYPE_VECTOR, ARC_TYPE_STANDARD, VERIFIED_VERSION, 0x1);
        let g = ItnGrammarSet::new(ItnParseType::ZhItn, good(), bad).unwrap();
        let Err(err) = g.compatibility() else {
            panic!("expected a compatibility gap for a symbol-table-carrying verbalizer");
        };
        let msg = format!("{err}");
        assert!(msg.contains("verbalizer"), "{msg}");
        assert!(msg.contains("WeTextProcessing"), "{msg}");
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
    }
}
