//! Per-model conversion routines (upstream checkpoint to GGUF builder).

pub(crate) mod campplus;
pub(crate) mod cosyvoice2;
// M4-20 T12: DeepFilterNet `denoise` → `vokra.denoise.*` GGUF writer (real
// checkpoint parse is owner, T17).
pub mod denoise;
pub(crate) mod kokoro;
pub(crate) mod piper_plus;
pub(crate) mod silero;
pub(crate) mod voxtral;
pub(crate) mod whisper;
