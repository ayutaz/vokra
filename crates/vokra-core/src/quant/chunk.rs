//! `vokra.quant.*` GGUF chunk reader/writer (M2-08 T05).
//!
//! **Landing pad only — implementation deferred to the T05 ticket.**
//!
//! The c03 change (T03 + T04 + T10) delivers the policy builder, ordered rule
//! table, HiFi-GAN INT8 opt-in gate, and the [`resolve`](super::resolve::resolve)
//! function. Chunk (de)serialisation depends on symbols that don't exist yet
//! (`gguf::chunks::KEY_QUANT_*` constants, `GgufBuilder::add_bool`,
//! `VokraError::UnknownQuantScheme`); a follow-up c0N change will add those
//! and populate this module with the reader/writer per the design in
//! `docs/tickets/m2/quantization-policy.md` §T05.
//!
//! Keeping the file present (even empty) as a stable module path so downstream
//! consumers can `use crate::quant::chunk;` once T05 lands without a rename.
