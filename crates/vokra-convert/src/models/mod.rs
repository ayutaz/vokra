//! Per-model conversion routines (upstream checkpoint to GGUF builder).

pub(crate) mod campplus;
// SoTA plan Phase 2 (2026-07-24): NVIDIA Canary-1B-v2 — multilingual
// multi-task ASR / AST (25 European languages). FastConformer encoder
// (32 layers) + Transformer decoder (8 layers, AED). CC-BY 4.0 weight
// (AttributionRequired). Every F32 / F16 tensor passes through
// verbatim; every hparam on the model card is transcribed from it and
// every remaining axis from the shared FastConformer-Transformer AED
// reference config (fast-conformer_aed.yaml — the whole Canary family's
// reference). Reuses the `vokra_ops::conformer` (FastConformer encoder
// body via `Stacking { factor: 8 }`) and `vokra_ops::beam_search`
// (attention-decoder search) primitives — no per-model op duplication.
pub(crate) mod canary;
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
// SoTA plan Phase 2 (2026-07-24): NVIDIA Parakeet-TDT-0.6B-v3 — English
// ASR (FastConformer encoder + TDT decoder). CC-BY 4.0 weight
// (AttributionRequired). Every F32 / F16 tensor passes through
// verbatim; every hparam is transcribed from the upstream
// `config.json` (encoder_config + decoder + joint / TDT). Reuses the
// `vokra_ops::conformer` + `vokra_ops::rnnt_decode` primitives — no
// per-model op duplication.
pub(crate) mod parakeet;
// SoTA plan Phase 2 (2026-07-24): NVIDIA Parakeet-CTC-1.1B — English ASR
// (FastConformer encoder + CTC head, no RNN-T prediction network). CC-BY
// 4.0 weight (AttributionRequired). Every F32 / F16 tensor passes through
// verbatim; every hparam is transcribed from the upstream `config.json`
// (encoder_config + top-level vocab_size / pad_token_id — no decoder or
// joint section exists for CTC). Reuses the `vokra_ops::conformer` +
// `vokra_ops::ctc_decode` primitives — no per-model op duplication.
pub(crate) mod parakeet_ctc;
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
