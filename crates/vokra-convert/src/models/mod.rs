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
// SoTA plan Phase 3 (2026-07-24): Resemble AI Chatterbox-Multilingual TTS
// (MIT weight, Permissive) safetensors → GGUF with the `vokra.chatterbox.*`
// chunk group. Every F32 / F16 tensor passes through verbatim; every hparam
// is transcribed from the primary source (`github.com/resemble-ai/chatterbox`
// — `src/chatterbox/models/t3/`). No `config.json` on HF (the release stores
// hparams in Python code), so the converter takes no config side-car; the
// variant tag (multilingual vs english-only) is a caller argument, defaulted
// to Multilingual by [`convert`]. Reuses the shared HiFTChain seam
// (`vokra-models::cosyvoice2::hift_chain::HiFTChain`, SoTA plan §1(a) 訂正
// 2026-07-22) — no new op or backend kernel is added.
pub(crate) mod chatterbox;
pub(crate) mod cosyvoice2;
// SoTA plan Phase 3 (2026-07-24): FunAudioLLM Fun-CosyVoice3-0.5B (apache-2.0
// permissive) — same architecture as CosyVoice2 (Qwen2 LLM backbone +
// chunk-aware CFM + HiFTNet vocoder). The tensor walk / shape derivation /
// Q/K/V bias uniformity check is delegated to `cosyvoice2::convert_*`; this
// module rewrites the arch label, model name, provenance, and metadata chunk
// prefix so the runtime dispatches to `vokra-models::cosyvoice3` (Phase 3
// refinements DRSR + Core-Cocktail are training-side and leave the runtime
// operators byte-identical to CosyVoice2). No new op — a very cheap follow-on.
pub(crate) mod cosyvoice3;
pub(crate) mod csm;
pub(crate) mod dac;
// SoTA plan Phase 1-4 (2026-07-24): nari-labs Dia-1.6B (Apache 2.0)
// safetensors → GGUF with the `vokra.dia.*` chunk group. Every tensor passes
// through verbatim; every hparam is transcribed from the upstream config.json.
pub(crate) mod dia;
// SoTA plan Phase 2 (2026-07-24): HuggingFace distil-whisper /
// distil-large-v3.5 (MIT weight, Permissive) — a distilled Whisper checkpoint
// that keeps the large-v3 encoder intact (32 layers / d_model=1280 /
// n_mels=128) and shrinks the decoder to 2 layers. Every F32 / F16 tensor
// passes through verbatim under the upstream HF Whisper name; hparams ride
// the `vokra.whisper.*` chunk schema (schema shared with vanilla Whisper —
// the "very cheap follow-on" contract in the task). Reuses the shared
// Whisper op inventory (STFT / mel filterbank / GEMM / GEMV / softmax /
// layer-norm / GELU / conv1d) — no new op is added.
pub(crate) mod distil_whisper;
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
// SoTA plan Phase 2 (2026-07-24): Meta omniASR-CTC-1B — 1600+ language
// multilingual ASR (wav2vec 2.0 waveform-in encoder + single-Linear CTC
// head, no RNN-T prediction network). Apache-2.0 weight (Permissive).
// Every F32 / F16 tensor passes through verbatim; every hparam is
// transcribed from the upstream fairseq2 registry walk
// (`omnilingual_asr/models/wav2vec2_asr/config.py::_1b_asr` →
// `wav2vec2_ssl/config.py::_1b_ssl` →
// `fairseq2/models/wav2vec2/config.py::large_lv60k`) — the HF release
// carries no `config.json`. Reuses the `vokra_ops::ctc_decode`
// primitive; the wav2vec 2.0 encoder body is a distinct topology
// from FastConformer (no shared `vokra_ops::wav2vec2_encoder` op today —
// deliberate "may need new op" follow-up).
pub(crate) mod omniasr_ctc;
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
