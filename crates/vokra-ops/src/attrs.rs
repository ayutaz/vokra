//! Audio front-end operator attribute types (M0-04-T01).
//!
//! The attribute types for the `stft` / `istft` / `mel_filterbank` / `mfcc` /
//! `dct` operators are **defined in `vokra-core`** (embedded in the
//! [`vokra_core::OpKind`] variants). The crate dependency edge runs
//! `vokra-ops → vokra-core`, so a type an `OpKind` variant carries cannot live
//! here — this is a documented deviation from M0-04-T01, which proposed
//! `crates/vokra-ops/src/attrs.rs` as the definition site. This module instead
//! re-exports them so the operator implementations and downstream callers get
//! an ergonomic `vokra_ops::attrs::*` path.
//!
//! # FR-OP-01 attribute checklist (all present on [`StftAttrs`])
//!
//! `window` · `hop_length` · `n_fft` · `center` (center-padding) · `pad_mode` ·
//! `normalization` (forward/backward/ortho) · `causal` (causal-mode) ·
//! `real_input` (RFFT) — plus `win_length` for `vokra.frontend.*` parity.
//!
//! # FR-OP-03 (Slaney/HTK)
//!
//! [`MelScale`] selects the Hz→mel warp and [`MelNorm`] the filter-bank
//! normalization; both `mel_filterbank` and `mfcc` route through them.

pub use vokra_core::ir::graph::{
    DctAttrs, IstftAttrs, IstftStreamingAttrs, MelAttrs, MelNorm, MelScale, MfccAttrs,
    Normalization, PadMode, StftAttrs, Window, WindowSymmetry,
};
