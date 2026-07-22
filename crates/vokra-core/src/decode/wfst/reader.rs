//! OpenFST binary format reader (M5-06 T05 + T06).
//!
//! Reads the on-disk `VectorFst<StdArc>` binary that
//! `fstcompile --arc_type=standard --fst_type=vector` writes, into a decode-only
//! [`Fst<TropicalWeight>`]. This is a **from-scratch** parser: the runtime
//! depends on no OpenFST code (NFR-DS-02); only the *dev-time* parity dumper
//! (`tools/parity/wfst_dump_reference.py`) touches real OpenFST, to produce the
//! committed fixtures this parser is verified against.
//!
//! # Where the format constants come from (they are NOT invented)
//!
//! Every magic / version / layout constant below was derived by
//! byte-inspecting a real `.fst` produced by OpenFST 1.8.4 `fstcompile`, the
//! same tool the parity dumper uses. A 3-state linear acceptor
//! (`0 -(1:1/0.5)-> 1 -(2:2/0.25)-> 2/0.125`) compiled to exactly 134 bytes:
//!
//! ```text
//! offset  field                       bytes (LE)              value
//! 0x00    magic         (i32)         d6 fd b2 7e             2125659606
//! 0x04    fsttype len   (i32)         06 00 00 00             6
//! 0x08    fsttype                     "vector"
//! 0x0E    arctype len   (i32)         08 00 00 00             8
//! 0x12    arctype                     "standard"
//! 0x1A    version       (i32)         02 00 00 00             2
//! 0x1E    flags         (i32)         00 00 00 00             0
//! 0x22    properties    (u64)         03 00 81 5a 69 00 00 00 (bitmask, ignored)
//! 0x2A    start         (i64)         00 …                    0
//! 0x32    numstates     (i64)         03 …                    3
//! 0x3A    numarcs       (i64)         00 …                    0  (unreliable*)
//! 0x42    <body>
//! ```
//! `*` VectorFst writes `numarcs = 0` in the header (arcs are counted per
//! state), so this parser reads it for the record but never trusts it — the
//! per-state `narcs` in the body is authoritative.
//!
//! Body, per state `s` in `0..numstates` (StdArc = tropical, weight = f32):
//! ```text
//!   final_weight (f32)         +inf ⇒ non-final
//!   narcs        (i64)
//!   narcs × arc {
//!     ilabel     (i32)  (>= 0; 0 = epsilon)
//!     olabel     (i32)  (>= 0; 0 = epsilon)
//!     weight     (f32)
//!     nextstate  (i32)  (>= 0)
//!   }
//! ```
//! An arc is exactly 16 bytes; the total for the fixture is
//! `0x42 + 3·(4+8) + 2·16 = 134`, matching `wc -c`.
//!
//! # Unsupported → explicit error (FR-EX-08)
//!
//! Only the byte-verified shape is accepted. A different `fst_type`, a
//! non-tropical `arc_type`, an unverified `version`, non-zero `flags`
//! (i.e. embedded symbol tables or aligned storage), or a negative label are
//! each an explicit [`VokraError::UnsupportedOp`] — never a best-effort parse.
//! Truncation / bad magic is [`VokraError::ModelLoad`].

use crate::error::{Result, VokraError};

use super::fst::{Arc, Fst};
use super::semiring::{Semiring, TropicalWeight};

/// OpenFST file magic (`kFstMagicNumber`). Derived from the real fixture header
/// (`d6 fd b2 7e` LE); do **not** change without a new byte-verified fixture.
const OPENFST_MAGIC: u32 = 0x7EB2_FDD6;

/// The only `fst_type` this reader accepts (decode-only VectorFst).
const FST_TYPE_VECTOR: &str = "vector";

/// The only `arc_type` this reader accepts (tropical / `StdArc`, weight = f32).
const ARC_TYPE_STANDARD: &str = "standard";

/// The single VectorFst header `version` byte-verified against a real fixture.
/// Other versions are rejected rather than guessed (FR-EX-08).
const VERIFIED_VERSION: i32 = 2;

/// Reads an OpenFST `VectorFst<StdArc>` binary into a decode-only tropical FST.
///
/// See the module docs for the byte layout and the accepted / rejected shapes.
/// The returned FST is **not** yet validated for decode — call
/// [`Fst::validate`] before handing it to the decoder.
///
/// # Errors
/// - [`VokraError::ModelLoad`] — truncated input or a magic-number mismatch;
/// - [`VokraError::UnsupportedOp`] — a recognised-but-unsupported format
///   variant (non-vector type, non-standard arc, unverified version, non-zero
///   flags = symbol tables / aligned storage, or a negative label).
pub fn read_openfst_vector(bytes: &[u8]) -> Result<Fst<TropicalWeight>> {
    let mut c = Cursor::new(bytes);

    // ---- Header ----
    let magic = c.read_u32()?;
    if magic != OPENFST_MAGIC {
        return Err(VokraError::ModelLoad(format!(
            "wfst: not an OpenFST binary — magic {magic:#010x} != {OPENFST_MAGIC:#010x}"
        )));
    }
    let fst_type = c.read_len_prefixed_string()?;
    if fst_type != FST_TYPE_VECTOR {
        return Err(VokraError::UnsupportedOp(format!(
            "wfst: fst_type `{fst_type}` unsupported (only `{FST_TYPE_VECTOR}`) — FR-EX-08"
        )));
    }
    let arc_type = c.read_len_prefixed_string()?;
    if arc_type != ARC_TYPE_STANDARD {
        return Err(VokraError::UnsupportedOp(format!(
            "wfst: arc_type `{arc_type}` unsupported (only tropical `{ARC_TYPE_STANDARD}`) — \
             FR-EX-08"
        )));
    }
    let version = c.read_i32()?;
    if version != VERIFIED_VERSION {
        return Err(VokraError::UnsupportedOp(format!(
            "wfst: VectorFst version {version} unsupported (only byte-verified version \
             {VERIFIED_VERSION}) — FR-EX-08"
        )));
    }
    let flags = c.read_i32()?;
    if flags != 0 {
        return Err(VokraError::UnsupportedOp(format!(
            "wfst: header flags {flags:#x} unsupported — embedded symbol tables / aligned \
             storage are out of the decode-only scope; supply a numeric-label FST (FR-EX-08)"
        )));
    }
    let _properties = c.read_u64()?; // bitmask; not needed for decode.
    let start_raw = c.read_i64()?;
    let numstates = c.read_i64()?;
    let _numarcs_hdr = c.read_i64()?; // unreliable for VectorFst — see module docs.

    if numstates < 0 {
        return Err(VokraError::ModelLoad(format!(
            "wfst: negative numstates {numstates}"
        )));
    }
    let numstates = usize::try_from(numstates)
        .map_err(|_| VokraError::ModelLoad("wfst: numstates too large".into()))?;

    // ---- Body ----
    // VectorFst stores each state INTERLEAVED: `final_weight, narcs, arcs…`,
    // repeated per state (NOT all final weights first). So states must be
    // created up front (arcs' `nextstate` may point forward) and then their
    // final weight + arcs read in a single per-state pass.
    let mut fst = Fst::new();
    for _ in 0..numstates {
        fst.add_state(TropicalWeight::zero());
    }
    for s in 0..numstates {
        let final_w = TropicalWeight::new(c.read_f32()?);
        fst.set_final(s as u32, final_w)?;
        let narcs = c.read_i64()?;
        if narcs < 0 {
            return Err(VokraError::ModelLoad(format!(
                "wfst: state {s} has negative narcs {narcs}"
            )));
        }
        for a in 0..narcs {
            let ilabel = c.read_label(s, a, "ilabel")?;
            let olabel = c.read_label(s, a, "olabel")?;
            let weight = TropicalWeight::new(c.read_f32()?);
            let nextstate = c.read_nextstate(s, a)?;
            fst.add_arc(
                s as u32,
                Arc {
                    ilabel,
                    olabel,
                    weight,
                    nextstate,
                },
            )?;
        }
    }

    // Resolve the start state (kNoStateId = -1 ⇒ no start).
    if start_raw == -1 {
        // leave `start` = None
    } else if start_raw < 0 {
        return Err(VokraError::ModelLoad(format!(
            "wfst: invalid start state {start_raw}"
        )));
    } else {
        let s = u32::try_from(start_raw)
            .map_err(|_| VokraError::ModelLoad("wfst: start state too large".into()))?;
        if (s as usize) >= numstates {
            return Err(VokraError::ModelLoad(format!(
                "wfst: start state {s} out of range (numstates={numstates})"
            )));
        }
        fst.set_start(s);
    }

    // Trailing bytes would mean we mis-parsed the body — surface it, don't
    // silently ignore (a truncation the other direction is already caught by
    // the cursor's bounds checks).
    if !c.at_end() {
        return Err(VokraError::ModelLoad(format!(
            "wfst: {} trailing bytes after body — parser/format mismatch",
            c.remaining()
        )));
    }

    Ok(fst)
}

/// A little-endian byte cursor with bounds-checked primitive reads. Every read
/// that would run past the end is a [`VokraError::ModelLoad`] (truncation),
/// never a panic.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or_else(|| {
            VokraError::ModelLoad("wfst: cursor overflow (corrupt length)".into())
        })?;
        if end > self.buf.len() {
            return Err(VokraError::ModelLoad(format!(
                "wfst: truncated — need {n} bytes at offset {}, only {} left",
                self.pos,
                self.remaining()
            )));
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Reads an OpenFST length-prefixed string: `i32` length + that many bytes,
    /// interpreted as UTF-8.
    fn read_len_prefixed_string(&mut self) -> Result<String> {
        let len = self.read_i32()?;
        if len < 0 {
            return Err(VokraError::ModelLoad(format!(
                "wfst: negative string length {len}"
            )));
        }
        let bytes = self.take(len as usize)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| VokraError::ModelLoad(format!("wfst: non-UTF-8 type string: {e}")))
    }

    /// Reads an arc label (`i32`, must be non-negative; negative labels are OpenFST
    /// special labels outside the decode-only scope → FR-EX-08).
    fn read_label(&mut self, state: usize, arc: i64, which: &str) -> Result<u32> {
        let v = self.read_i32()?;
        if v < 0 {
            return Err(VokraError::UnsupportedOp(format!(
                "wfst: state {state} arc {arc} has negative {which} {v} — special labels are \
                 out of the decode-only scope (FR-EX-08)"
            )));
        }
        Ok(v as u32)
    }

    /// Reads an arc `nextstate` (`i32`, must be non-negative).
    fn read_nextstate(&mut self, state: usize, arc: i64) -> Result<u32> {
        let v = self.read_i32()?;
        if v < 0 {
            return Err(VokraError::ModelLoad(format!(
                "wfst: state {state} arc {arc} has negative nextstate {v}"
            )));
        }
        Ok(v as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These are ERROR-PATH tests. The positive parse (that the reader
    // reconstructs a real OpenFST-produced FST byte-for-byte) is verified in
    // `tests/parity_wfst.rs` against committed fixtures generated by real
    // OpenFST — a self-built byte buffer here would be a self-mirror and prove
    // nothing about the real format (numerical-parity discipline).

    /// The smallest valid header this reader accepts, for a `numstates`-state,
    /// zero-arc, start-0 FST with `numstates >= 1` and state 0 final. Uses ONLY
    /// the constants derived from the real fixture — it is a *rejection-test
    /// scaffold*, not a format oracle.
    fn header_only(fst_type: &str, arc_type: &str, version: i32, flags: i32) -> Vec<u8> {
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
        v.extend_from_slice(&1i64.to_le_bytes()); // numstates = 1
        v.extend_from_slice(&0i64.to_le_bytes()); // numarcs (hdr) = 0
        // one final state, no arcs: final weight 0.0, narcs 0
        v.extend_from_slice(&0.0f32.to_le_bytes());
        v.extend_from_slice(&0i64.to_le_bytes());
        v
    }

    #[test]
    fn bad_magic_is_model_load_error() {
        let bytes = vec![0u8; 64];
        match read_openfst_vector(&bytes) {
            Err(VokraError::ModelLoad(m)) => assert!(m.contains("magic"), "{m}"),
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn truncated_after_magic_is_model_load_error() {
        let bytes = OPENFST_MAGIC.to_le_bytes().to_vec();
        assert!(matches!(
            read_openfst_vector(&bytes),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn non_vector_type_is_unsupported() {
        let bytes = header_only("const", "standard", VERIFIED_VERSION, 0);
        match read_openfst_vector(&bytes) {
            Err(VokraError::UnsupportedOp(m)) => assert!(m.contains("fst_type"), "{m}"),
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }
    }

    #[test]
    fn non_standard_arc_is_unsupported() {
        let bytes = header_only("vector", "log", VERIFIED_VERSION, 0);
        match read_openfst_vector(&bytes) {
            Err(VokraError::UnsupportedOp(m)) => assert!(m.contains("arc_type"), "{m}"),
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }
    }

    #[test]
    fn unverified_version_is_unsupported() {
        let bytes = header_only("vector", "standard", 99, 0);
        match read_openfst_vector(&bytes) {
            Err(VokraError::UnsupportedOp(m)) => assert!(m.contains("version"), "{m}"),
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }
    }

    #[test]
    fn nonzero_flags_are_unsupported() {
        // flag bit for symbol tables set → out of decode-only scope.
        let bytes = header_only("vector", "standard", VERIFIED_VERSION, 1);
        match read_openfst_vector(&bytes) {
            Err(VokraError::UnsupportedOp(m)) => assert!(m.contains("flags"), "{m}"),
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut bytes = header_only("vector", "standard", VERIFIED_VERSION, 0);
        bytes.extend_from_slice(&[0xAB, 0xCD]); // junk after the body
        match read_openfst_vector(&bytes) {
            Err(VokraError::ModelLoad(m)) => assert!(m.contains("trailing"), "{m}"),
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn header_only_scaffold_parses_when_well_formed() {
        // Sanity that the rejection scaffold itself is a parseable 1-state FST
        // (so the rejection tests above isolate exactly the field they mutate).
        let bytes = header_only("vector", "standard", VERIFIED_VERSION, 0);
        let fst = read_openfst_vector(&bytes).expect("well-formed 1-state FST parses");
        assert_eq!(fst.num_states(), 1);
        assert_eq!(fst.start(), Some(0));
        assert_eq!(fst.num_arcs(), 0);
    }
}
