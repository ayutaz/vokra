//! `wfst_decode` — WFST token-passing decoder (M5-06, FR-OP-43).
//!
//! **Opt-in**: this whole module is compiled only under the `vokra-wfst`
//! feature (default OFF). A from-scratch, zero-dependency Rust port of just the
//! decode-side of OpenFST — the runtime links **no** OpenFST code (NFR-DS-02).
//!
//! # What lands here
//!
//! - [`semiring`] — the [`Semiring`] trait + the tropical [`TropicalWeight`]
//!   (Viterbi min/plus). The `log` semiring is a documented future additive
//!   (ADR M5-06 §2).
//! - [`fst`] — the decode-only [`Fst`] / [`Arc`] / [`StateId`] / [`Label`] data
//!   structures with structural [`Fst::validate`] (no `compose` / `determinize`
//!   — the HCLG graph is composed offline, ADR M5-06 §1).
//! - [`reader`] — [`read_openfst_vector`], the OpenFST `VectorFst<StdArc>`
//!   binary reader, its format constants byte-verified against real fixtures.
//! - [`decoder`] — [`WfstDecoder`], the frame-synchronous token-passing sweep
//!   over a per-frame acoustic emission matrix.
//! - [`lattice`] — [`WfstLattice`] + best-path + [`WfstHypothesis`] n-best.
//!
//! # The honest scope (do not overstate)
//!
//! `wfst_decode` consumes a per-frame **emission** matrix (a CTC / RNN-T
//! posterior). The decoders themselves have since landed as runtime
//! primitives (`vokra_ops::ctc_decode_greedy` / `ctc_decode_beam` /
//! `rnnt_decode`); what [`crate::m5_residual_ops`] still reserves is only
//! their graph-side `OpKind` variant. The constraint that actually binds M5-06
//! is the **feeder**: no model forward yet produces an emission matrix
//! end-to-end (the CTC / TDT binders are loud-partial on their encoder
//! forward), and Vokra's only live acoustic decoder (`beam_search`) returns
//! token sequences, not an emission matrix. So M5-06 is **decode-only**: verified against offline
//! reference emissions (`tests/parity_wfst.rs`, OpenFST oracle) and usable for
//! lexicon / grammar rescoring at the word level; a classic HCLG
//! frame-synchronous e2e ASR waits on the CTC/RNN-T feeder. See ADR M5-06 and
//! `docs/handoff/m5-06.md`.
//!
//! # C ABI
//!
//! Not exported to `include/vokra.h` in M5-06 — the C-surface decision is
//! deferred to the M5-13 freeze (ADR M5-06). So a C consumer cannot call
//! `wfst_decode` during the v1.0-rc window; the Rust API here is the only
//! surface.

pub mod decoder;
pub mod fst;
pub mod lattice;
pub mod reader;
pub mod semiring;

pub use decoder::{WfstDecodeConfig, WfstDecoder};
pub use fst::{Arc, Fst, Label, StateId};
pub use lattice::{LatArc, WfstHypothesis, WfstLattice};
pub use reader::read_openfst_vector;
pub use semiring::{Semiring, TropicalWeight};
