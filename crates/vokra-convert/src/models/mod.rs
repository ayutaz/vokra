//! Per-model conversion routines (upstream checkpoint to GGUF builder).

pub(crate) mod campplus;
pub(crate) mod cosyvoice2;
pub(crate) mod csm;
pub(crate) mod dac;
// SoTA plan Phase 1-4 (2026-07-24): nari-labs Dia-1.6B (Apache 2.0)
// safetensors → GGUF with the `vokra.dia.*` chunk group. Every tensor passes
// through verbatim; every hparam is transcribed from the upstream config.json.
pub(crate) mod dia;
// M4-20 T12/T17: DeepFilterNet3 `denoise` → `vokra.denoise.*` GGUF (real
// checkpoint parse from the prepared safetensors, verbatim upstream names).
pub mod denoise;
pub(crate) mod kokoro;
pub(crate) mod mimi;
pub(crate) mod moshi;
pub(crate) mod piper_plus;
pub(crate) mod silero;
pub(crate) mod utmos;
pub(crate) mod voxtral;
pub(crate) mod whisper;
