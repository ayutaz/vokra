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
// SoTA plan Phase 2 (2026-07-24): Kyutai STT-2.6B-EN (CC-BY 4.0 weight,
// AttributionRequired) safetensors → GGUF with the `vokra.kyutai_stt.*`
// chunk group. Every F32 / F16 tensor passes through verbatim; every
// hparam is transcribed from the upstream config.json. The upstream
// release is BF16 and the streaming-BF16 pass-through path is a follow-up
// (T29-equivalent — the Moshi pattern).
pub(crate) mod kyutai_stt;
pub(crate) mod mimi;
pub(crate) mod moshi;
pub(crate) mod piper_plus;
pub(crate) mod silero;
pub(crate) mod utmos;
pub(crate) mod voxtral;
pub(crate) mod whisper;
// SoTA plan Phase 1-5 (2026-07-24): Zyphra Zonos-v0.1-transformer
// (Apache 2.0) safetensors → GGUF with the `vokra.zonos.*` chunk group.
// Every float tensor passes through verbatim; every hparam (including the
// 7 typed prefix conditioners) is transcribed from the upstream config.json.
pub(crate) mod zonos;
