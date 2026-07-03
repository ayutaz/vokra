//! Re-export of the shared JSON parser, now hosted in
//! [`vokra_core::json`](vokra_core::json).
//!
//! Promoted from `vokra-convert` to `vokra-core` in M1-02 so the *runtime*
//! safetensors direct-load path can share one std-only parser (still zero
//! external dependencies). This thin re-export keeps the existing
//! `crate::json` paths in the safetensors reader and the piper-plus
//! `config.json` parser working unchanged.

pub(crate) use vokra_core::json::*;
