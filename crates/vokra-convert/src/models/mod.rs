//! Per-model conversion routines (upstream checkpoint to GGUF builder).

// SoTA plan follow-on (2026-07-25): baichuan-inc/Baichuan-Audio
// (apache-2.0). Category: s2s. Baichuan Omni-1.5 = Whisper-Large enc +
// 8-layer RVQ 12.5Hz + Flow Matching mel + CosyVoice2 HiFi-GAN. Every
// F32 / F16 / BF16 tensor passes through verbatim following the
// qwen3_tts / vibevoice / voxcpm2 pattern; real-weight parity is
// deferred to owner (`docs/license-audit.md` §3.1 sign-off).
pub mod baichuan_audio;
pub mod bicodec;
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
// SoTA plan reuse bundle (2026-07-30): NVIDIA Canary-Qwen-2.5B —
// multimodal ASR + Qwen LLM head-swap on top of Canary FastConformer
// encoder. CC-BY 4.0 weight (AttributionRequired via `canary-` prefix
// walk). Every F32 / F16 / BF16 tensor passes through verbatim; every
// encoder hparam reuses the primary-source Canary-1B-v2 defaults; every
// decoder hparam carries canonical Qwen-family constants (GQA 16 Q /
// 8 KV, head_dim=128, rope=1_000_000, rms_norm_eps=1e-6) with
// `0`-placeholder dims (n_layer / hidden_dim / ffn_dim / vocab_size /
// n_ctx) pending `.nemo` extraction — runtime validator rejects `0`
// loudly (FR-EX-08). Reuses the shared `canary` encoder + Voxtral
// text-decoder session primitives — no per-model op duplication.
pub(crate) mod canary_qwen;
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
// SoTA plan Phase 3 (2026-07-24): Resemble AI Chatterbox-Turbo TTS
// (MIT weight, Permissive) safetensors → GGUF with the
// `vokra.chatterbox_turbo.*` chunk group. Every F32 / F16 tensor
// passes through verbatim; every hparam is transcribed from the
// primary source `t3_turbo_v1.yaml`
// (`huggingface.co/ResembleAI/chatterbox-turbo`, fetched 2026-07-24).
// Distinct arch tag from base Chatterbox because Turbo swaps backbone
// family (gpt2-medium vs Llama_520M) + sample rate (32 kHz vs 24 kHz)
// + text vocabulary (50 276 vs 2454/704 — GPT-2 base 50 257 + 19
// paralinguistic tags from `added_tokens.json`); the terminal vocoder
// is still S3Gen HiFT-GAN so the shared HiFTChain seam
// (`vokra-models::cosyvoice2::hift_chain::HiFTChain`) applies —
// no new op or backend kernel is added.
pub(crate) mod chatterbox_turbo;
// SoTA plan Phase 3 (2026-07-24): Resemble AI Chatterbox-Nano TTS
// (MIT weight, Permissive) safetensors → GGUF with the
// `vokra.chatterbox_nano.*` chunk group. Every F32 / F16 tensor
// passes through verbatim; every hparam is transcribed from the
// primary source `t3_nano_v1.yaml`
// (`huggingface.co/ResembleAI/chatterbox-nano`, fetched 2026-07-24).
// Distinct arch tag from base Chatterbox + Turbo because Nano keeps
// base's Llama_520M backbone family but swaps sample rate (32 kHz vs
// 24 kHz) + text vocabulary (50 276 GPT-2 vs 2454/704) +
// stop-text sentinel (50 256 GPT-2 EOT vs 0); the terminal vocoder
// is still S3Gen HiFT-GAN so the shared HiFTChain seam
// (`vokra-models::cosyvoice2::hift_chain::HiFTChain`) applies —
// no new op or backend kernel is added.
pub(crate) mod chatterbox_nano;
pub(crate) mod cosyvoice2;
// M5 gap follow-up (2026-07-30): marl/crepe (Kim et al. 2018) — a
// monophonic F0 (fundamental-frequency) extractor. The upstream release
// ships a Keras / TensorFlow `.h5`, so `tools/parity/keras_h5_to_safetensors.py`
// bridges the checkpoint into safetensors + a JSON config side-car this
// converter consumes (the DAC / Kokoro / UTMOS split — zero-dep, no
// TensorFlow / Keras / torch in the runtime, NFR-DS-02 / FR-LD-05).
// License = MIT (`marl/crepe/main/LICENSE.txt`, "Copyright (c) 2018 Jong
// Wook Kim et al.", fetched 2026-07-30 — CLAUDE.md「ハルシネーション厳禁」).
pub(crate) mod crepe;
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
// SBV2 v2 plan Task 11 (2026-07-26): DeBERTa v2 (`ku-nlp/deberta-v2-large-
// japanese-char-wwm`, cc-by-sa-4.0) and v3 (`microsoft/deberta-v3-large`,
// mit) safetensors → GGUF, category `bert`. BF16 pass-through mirror of
// `funcodec` / `wespeaker`; hparams are checkpoint-shape-derived where
// possible (never invented) with a documented, unverified "large"-variant
// placeholder for the axes no single tensor shape can carry (`n_heads`,
// `n_pos_buckets`, `max_pos_dist`). Tensor names pass through verbatim —
// the HF -> `bert.*` rename table `DebertaV2Encoder::from_gguf` /
// `DebertaV3Encoder::from_gguf` (`crates/vokra-bert`) expect is a real-
// checkpoint-header question deferred to Task 30 (TODO(owner) markers in
// both files). Lives here rather than in `vokra-bert` specifically to
// avoid a `vokra-bert <-> vokra-convert` dependency cycle the original
// plan's task split would have created — see `deberta_v2`'s module doc.
pub mod deberta_v2;
pub mod deberta_v3;
// SoTA plan Phase 1-4 (2026-07-24): nari-labs Dia-1.6B (Apache 2.0)
// safetensors → GGUF with the `vokra.dia.*` chunk group. Every tensor passes
// through verbatim; every hparam is transcribed from the upstream config.json.
pub(crate) mod dia;
pub mod ecapa_tdnn;
pub mod emotion2vec;
// M5-16 (FR-OP-83): FCPE — Fast Context-based Pitch Estimator (CNChTu/FCPE,
// MIT permissive). safetensors → GGUF pass-through (F32 / F16 / BF16
// verbatim, `vokra.fcpe.*` / `vokra.provenance.*` stamps). Reuses the
// shared `vokra_ops::conformer::ConformerEncoder` primitive on the runtime
// side (SoTA Phase 2 landed op — no per-model op duplication); the upstream
// release is a torch-pickle `.pt`, so callers pre-flatten it to safetensors
// via `tools/parity/fcpe_prepare_checkpoint.py` (the DFN3 / DAC / CSM
// bridge pattern — no pickle ever enters the runtime, FR-LD-05).
pub(crate) mod fcpe;
pub mod freevc;
// SoTA plan Phase 5 VAD-2 (2026-07-30): FunASR **FSMN-VAD**
// (`iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`, mit) safetensors → GGUF
// with the `vokra.fsmn_vad.*` chunk group. Feed-forward Sequential Memory
// Network for voice activity detection — first-class audio-dialect op
// posture (distinct from Silero VAD v5's FR-LD-06 1:1 subgraph). Every
// F32 / F16 / BF16 tensor passes through verbatim under the upstream
// state-dict name; every hparam axis is stamped unconditionally (the
// released FunASR checkpoint has fixed axes — see the module docstring).
// Real-weight parity deferred to owner sign-off (§3.1 row landed
// 2026-07-30 yousan).
pub mod fsmn_vad;
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
// SoTA plan Phase 5 JA-ASR-2 (2026-07-24): Kotoba Technologies
// **kotoba-whisper** family — Japanese-distilled Whisper (large-v3
// encoder + shrunk 2-layer decoder — same tensor topology as
// distil-large-v3.5, but distilled on ReazonSpeech Japanese audio
// and released under **apache-2.0**, distinct from distil-whisper's
// MIT). Every F32 / F16 tensor passes through verbatim under the
// upstream HF Whisper name; hparams ride the `vokra.whisper.*` chunk
// schema (schema shared with vanilla Whisper — the "very cheap
// follow-on" contract in the task). The JA-ASR-2 axis (data-driven
// decoder depth) is honored by the shape-driven `count_layers` walk
// — the converter reads `n_text_layer=2` from the checkpoint's
// tensor names, never hard-coding to 32. Reuses the shared Whisper
// op inventory (STFT / mel filterbank / GEMM / GEMV / softmax /
// layer-norm / GELU / conv1d) — no new op is added.
pub(crate) mod kotoba_whisper;
// M4-20 T12/T17: DeepFilterNet3 `denoise` → `vokra.denoise.*` GGUF (real
// checkpoint parse from the prepared safetensors, verbatim upstream names).
pub mod denoise;
pub mod funcodec;
pub mod kimi_audio;
pub mod knn_vc;
pub(crate) mod kokoro;
// SoTA plan Phase 2 (2026-07-24): Kyutai STT-2.6B-EN (CC-BY 4.0 weight,
// AttributionRequired) safetensors → GGUF with the `vokra.kyutai_stt.*`
// chunk group. Every F32 / F16 tensor passes through verbatim; every
// hparam is transcribed from the upstream config.json. The upstream
// release is BF16 and the streaming-BF16 pass-through path is a follow-up
// (T29-equivalent — the Moshi pattern).
pub(crate) mod kyutai_stt;
pub mod meanvc;
pub(crate) mod mimi;
pub(crate) mod moshi;
pub mod neucodec;
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
pub mod openvoice_v2;
// pyannote/segmentation-3.0 (Bredin, CNRS, MIT — 2026-07-30 §3.1 row 263
// yousan sign-off, DIARIZE_OP blocker text "trigger + license double" →
// "trigger only" this wave). PyanNet voice-activity-detection /
// speaker-segmentation backbone (SincNet → BiLSTM x2 → Linear x2 →
// powerset multiclass classifier). Category = `vad`. BF16 pass-through
// skeleton + `vokra.pyannote.*` hparam chunk group so the future runtime
// binder (`crates/vokra-models/src/pyannote/`) can bring the graph up
// without a side-car config lookup. Runtime forward is Wave 2 loud-partial
// (SincNet primitive is Vokra-new op, Wave 3 scope) —
// `docs/handoff/pyannote-implementation-plan-2026-07-30.md`.
pub(crate) mod piper_plus;
pub mod pyannote_segmentation;
// pyannote/speaker-diarization-3.1 (Bredin, CNRS, MIT — 2026-08-01 Wave 5
// pipeline orchestration add, docs/license-audit.md §3.1 sign-off row).
// SpeakerDiarization pipeline that composes the sibling MIT weight repos
// pyannote/segmentation-3.0 (VAD backbone — pub mod pyannote_segmentation
// above) + pyannote/wespeaker-voxceleb-resnet34-LM (speaker encoder —
// pub mod wespeaker below) via AgglomerativeClustering. Category `diarize`.
// Weightless GGUF — the converter reads a config.yaml sanity buffer and
// emits primary-source-verified pipeline hparams under
// `vokra.pyannote_pipeline.*`; no YAML parser enters the runtime tree
// (NFR-DS-02 zero-dep). The Rust runtime pipeline dispatch is a
// separate WP — this converter only stamps orchestration metadata.
pub mod pyannote_speaker_diarization_3_1;
// SoTA plan Phase 3 (2026-07-24): Alibaba **Qwen3-TTS-12Hz-0.6B-Base**
// (apache-2.0 end-to-end weight) safetensors → GGUF with the
// `vokra.qwen3_tts.*` chunk group. Discrete multi-codebook LM
// (Qwen3-flavour 28-layer talker + 5-layer parallel code predictor +
// shared Qwen3-TTS-Codec seam via `vokra_ops::qwen3_tts_codec`).
// Every F32 / F16 tensor passes through verbatim; every hparam is
// transcribed from the primary source
// `huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/raw/main/config.json`
// (talker.* + code_predictor.*) plus README.md (speaker encoder
// 24 kHz / 1024-dim). Distinct arch tag from CosyVoice2/3 because
// Qwen3-TTS is codec-LM not vocoder-LM — the terminal step is
// qwen3_tts_codec, NOT HiFTChain.
pub(crate) mod qwen3_tts;
// SBV2 v2 plan Task 25 (2026-07-26): Style-Bert-VITS2 v2
// (`litagin02/style_bert_vits2` family, AGPL-3.0 -> LicenseClass::Copyleft
// default) safetensors -> GGUF, category `tts`. BF16 pass-through mirror of
// `deberta_v2` / `funcodec` / `wespeaker`; the `vokra.sbv2.*` hparam chunk
// (22 required + 1 optional keys) is written only when a JSON config
// side-car is supplied -- never filled with invented placeholders (see
// `sbv2.rs`'s module doc "Hparams" section). Tensor names pass through
// verbatim -- the upstream-name -> `sbv2.*` rename table
// `SbV2Model::from_gguf` (`crates/vokra-models/src/sbv2/mod.rs`, Task 24)
// expects is a real-checkpoint-header question deferred to Task 30. Lives
// here rather than in `vokra-models` specifically to avoid a
// `vokra-models <-> vokra-convert` dependency cycle the design doc's
// original task split would have created -- see `sbv2`'s module doc (same
// rationale as Task 11's `deberta_v2` / `deberta_v3`).
// F0 pitch-extractor tier (2026-07-30): **RMVPE** (`yxlllc/RMVPE` fork of
// `Dream-High/RMVPE`, MIT weight + code — Permissive). Safetensors →
// GGUF with the `vokra.rmvpe.*` hparam chunk group; every F32 / F16 /
// BF16 tensor passes through verbatim under upstream state_dict names.
// Distinct arch tag (`rmvpe`) — the first `category = "f0"` binder in
// the converter tree. Consumed by a native `vokra-models::f0::rmvpe`
// runtime (U-Net + GRU CNN over a 128-mel spectrogram at 16 kHz →
// 360-class pitch head → argmax → Hz via a log-cents grid). Sibling
// pattern of `emotion2vec` (BF16 pass-through, dedicated arch tag,
// converter-side hparam stamping).
pub mod rmvpe;
pub(crate) mod sbv2;
// StyleTTS 2 (yl4579, Li et al. 2023 arXiv:2306.07691) — config-only
// scaffold. **Weight license = voice-consent / disclosure usage
// agreement (NOT standard SPDX permissive)** so provenance stamps
// [`LicenseClass::Unknown`] (fail-closed under M2-13); the runtime
// `StyleTts2Tts::from_gguf` is deliberately unwired
// (`docs/license-audit.md` §3.1 StyleTTS 2 sign-off = ☑ Rejected
// 2026-07-23 yousan). A user who trained their own StyleTTS 2 on a
// permissive corpus overrides at the `--license <spdx>` boundary — the
// same escape hatch vits-ja / kokoro / whisper use. Architecture MIT
// (upstream repo LICENSE) and always independently implementable
// (whisper.cpp 型, CLAUDE.md 設計判断 4).
pub(crate) mod silero;
pub mod speaker_3d;
pub mod speechtokenizer;
pub(crate) mod styletts2;
// SoTA follow-on (2026-07-30): NVIDIA TitaNet-Large speaker verification
// (`nvidia/speakerverification_en_titanet_large`, CC-BY-4.0 = AttributionRequired
// — HF cardData primary source verified 2026-07-30). Category = `speaker`.
// Depth-wise-separable Conv1D speaker-embedding extractor, 16 kHz mono →
// 192-d embedding, ~23 M params. Every F32 / F16 / BF16 tensor passes
// through verbatim (mirror of wespeaker / ecapa_tdnn / speaker_3d
// skeleton pattern) plus the FR-MD-09 attribution chunk (stamp_attribution)
// for the CC-BY-4.0 display obligation — the sibling mimi / moshi /
// parakeet / canary / kyutai_stt CC-BY-4.0 posture. The `.nemo` tarball
// distribution is bridged offline through `tools/parity/nemo_pt_to_safetensors.py`;
// this converter accepts safetensors only (int-tensor strip like
// BatchNorm `num_batches_tracked` is done at the bridge script — the
// safetensors reader admits only F32 / F16 / BF16 so any surviving int
// would fail parse before reaching us). Runtime port is out-of-scope
// (M5-residual `titanet_speaker_encode`, FR-OP-80 variant); consumers
// needing a speaker embedding today should use CAM++ (`vokra-models::speaker_encode`)
// under Apache-2.0 with no attribution overhead.
pub mod titanet;
// SoTA plan Phase 3 (2026-07-25): StepFun **Step-Audio-2-mini**
// (apache-2.0 end-to-end weight) safetensors → GGUF skeleton. 8B S2S
// with a dual codebook (semantic 1024 + acoustic 4096) and a
// flow-matching mel decoder. This is the pass-through skeleton
// (`convert_step_audio2_mini_file`) — every F32 / F16 / BF16 tensor
// passes through verbatim; real-weight parity is deferred to owner
// (docs/license-audit.md §3.1 sign-off). Distinct arch tag from every
// sibling — silently sharing would mis-route the runtime dispatch.
pub mod step_audio2_mini;
pub(crate) mod utmos;
// SoTA plan Phase 4 (2026-07-24): OpenBMB **VoxCPM-0.5B** (apache-2.0
// end-to-end weight) safetensors → GGUF with the `vokra.voxcpm2.*` and
// `vokra.vae_continuous.*` chunk groups. NEW class of TTS vs every
// earlier target — the terminal decoding hop is a continuous VAE
// decoder consuming flow-matching sampler output (not vocoder-LM
// HiFTChain, not codec-LM RVQ / FSQ). Topology: MiniCPM-4 LM backbone
// (24-layer / 1024d / GQA 16 Q ÷ 2 KV / SwiGLU 4096 / RoPE θ=10000
// with longrope scaling / RMSNorm ε=1e-5 / vocab=73448) + 6-layer
// residual acoustic LM + 4-layer local encoder + 4-layer local DiT +
// UnifiedCFM flow-matching sampler (Euler / inference_cfg_rate=2.0) +
// AudioVAE V2 continuous encoder / decoder (16 kHz PCM in → 25 Hz
// latents → 48 kHz PCM out) + inline scalar-quantization bottleneck
// (`scalar_quantization_latent_dim=256`, `scalar_quantization_scale=9`).
// Every F32 / F16 tensor passes through verbatim; every hparam is
// transcribed from the primary source
// `huggingface.co/openbmb/VoxCPM-0.5B/raw/main/config.json` +
// `openbmb/VoxCPM/src/voxcpm/modules/audiovae/audio_vae_v2.py`
// (`AudioVAEConfig` defaults). Distinct arch tag from CosyVoice2/3 /
// Qwen3-TTS / Chatterbox family because VoxCPM's terminal step is
// vae_continuous_decode, NOT HiFTChain or any RVQ / FSQ codec.
pub(crate) mod voxcpm2;
// SoTA plan Phase 4 (2026-07-24): Microsoft **VibeVoice-1.5B** (MIT
// end-to-end weight) safetensors → GGUF with the `vokra.vibevoice.*`
// chunk group. SECOND consumer of the continuous VAE + diffusion
// decoder class (after VoxCPM-0.5B) — but where VoxCPM uses a
// UnifiedCFM flow-matching sampler, VibeVoice uses a **DDPM** sampler
// (v-prediction + cosine β schedule). Topology: Qwen2 decoder LM
// (28-layer / 1536d / MHA n_head=12 n_head_kv=2 (GQA ratio 6) /
// SwiGLU 8960 / RoPE θ=1_000_000 / RMSNorm ε=1e-6 / vocab=151_936 /
// max_positions=65_536 / tie_word_embeddings=true) + acoustic σ-VAE
// tokenizer (vae_dim=64, mirror-symmetric encoder/decoder at 24 kHz,
// encoder_ratios=[8,5,5,4,2,2] → 7.5 Hz frame rate) + semantic
// tokenizer (encoder-only deterministic, vae_dim=128) + 4-layer
// AdaLN-modulated MLP diffusion head (hidden=1536, head_ffn_ratio=3.0 →
// ffn_dim=4608, latent_size=64, prediction_type="v_prediction",
// diffusion_type="ddpm", ddpm_num_steps=1000,
// ddpm_num_inference_steps=20, ddpm_beta_schedule="cosine"). Every
// F32 / F16 tensor passes through verbatim; every hparam is
// transcribed from the primary sources
// `huggingface.co/microsoft/VibeVoice-1.5B/raw/main/config.json` and
// `github.com/microsoft/VibeVoice/blob/main/vibevoice/modular/
// configuration_vibevoice.py`. Distinct arch tag from VoxCPM /
// CosyVoice2/3 / Qwen3-TTS / Chatterbox family — silently sharing
// would misroute the runtime dispatch (VoxCPM → flow_sample,
// VibeVoice → ddpm_sample).
pub(crate) mod vibevoice;
// SoTA plan Phase 5 JA-TTS-1 (2026-07-24): Aratako **Irodori-TTS-500M-v3**
// Japanese TTS (MIT weight + code, verified via
// `gh api /repos/Aratako/Irodori-TTS/license` → `MIT`, fetched
// 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」) safetensors → GGUF
// with the `vokra.irodori.*` chunk group. A Rectified-Flow Diffusion
// Transformer (RF-DiT) over the paired `Semantic-DACVAE-Japanese-32dim`
// codec (32-d continuous latent → 48 kHz PCM). Every F32 / F16 tensor
// passes through verbatim; every hparam is transcribed from
// `train_500m_v3_phase1_body.yaml` + `train_500m_v3_phase2_duration.yaml`
// + `irodori_tts/config.py::ModelConfig` at
// `github.com/Aratako/Irodori-TTS`. Distinct arch tag from every
// sibling — silently sharing would misroute the runtime dispatch
// (VibeVoice → ddpm_sample, VoxCPM → EpsS flow_sample, Irodori →
// Linear/Sway flow_sample with a distinct latent width 32 vs the
// Phase-4 siblings' 64). No side-car config today: every field is
// fixed for the 500M-v3 release and byte-parallel to the transcribed
// constants; a future 600M VoiceDesign / 2.5B variant that reshapes
// the DiT or adds caption conditioning would demand `--config`. No
// new op or backend kernel — reuses the shared
// `vokra_ops::flow_sampler` primitive (M3-05, `OdeSolver::Euler` +
// `Schedule::Linear` | `Schedule::Sway`) and the shared
// `crate::codec::DacCodecGguf` seam for the paired DACVAE decode.
pub(crate) mod irodori;
// SoTA plan Phase 5 JA-TTS-2 (2026-07-24): ESPnet-family Japanese
// **plain VITS** (Kim et al. 2021 VITS + HiFi-GAN generator) —
// architecture Apache-2.0 (ESPnet `espnet2/gan_tts/vits/`) + MIT
// (`jaywalnut310/vits` reference). Distinct arch tag from piper-plus
// because plain VITS decodes through a HiFi-GAN generator directly
// while piper-plus (MB-iSTFT-VITS2) decodes through a sub-band iSTFT
// + PQMF post-net (silently sharing an arch tag would misroute the
// runtime dispatch). Every F32 / F16 tensor passes through verbatim;
// every hparam is transcribed from the primary sources
// `egs2/jsut/tts1/conf/tuning/train_vits.yaml` +
// `egs2/jvs/tts1/conf/tuning/finetune_vits.yaml` +
// `espnet2/gan_tts/vits/{vits,generator}.py` (fetched 2026-07-24 —
// CLAUDE.md「ハルシネーション厳禁」). Reuses the shared
// `vokra_ops::hifigan_generator` (M3-07, FR-OP-10) primitive via
// `VitsJaConfig::to_hifigan_attrs`; no new op or backend kernel is
// added. **⚠️  Weight redistribution default**: the JSUT / JVS /
// COEIROINK corpus terms forbid trained-weight redistribution, so
// the provenance stamp defaults to
// `LicenseClass::RedistributionForbidden`. A user who trained on a
// permissive corpus overrides at the outer
// `convert_file --license <spdx>` boundary.
pub(crate) mod vits_ja;
pub(crate) mod voxtral;
pub mod wespeaker;
pub(crate) mod whisper;
// SoTA plan Phase 5 codec (2026-07-28): HKUSTAudio/xcodec2 (**cc-by-nc-4.0**
// weight — HF card front-matter, CC-verified 2026-07-15, sign-off 2026-07-23
// yousan = ☑ Research-only, `docs/license-audit.md` §3.1). FSQ codec paired
// with the Llasa TTS family — the M4-16 landing implemented the FSQ decode
// op-side (`xcodec2_fsq`, `crates/vokra-ops/src/fsq_codec.rs`); this
// converter completes the missing "safetensors → GGUF" side. F32 / F16 /
// BF16 pass-through mirrors the neucodec / step_audio2_mini contract; the
// license default is NonCommercial (fail-closed) so a commercial-mode
// caller cannot silently bring up NC weights.
pub(crate) mod xcodec2;
// SoTA plan Phase 5 codec (2026-07-25): fnlp XY_Tokenizer_TTSD_V0
// (apache-2.0) safetensors → GGUF. 1 kbps RVQ-8 @ 12.5 Hz — the codec
// half of MOSS-TTSD. F32 / F16 / BF16 pass-through following the
// qwen3_tts / vibevoice / voxcpm2 landed contract.
pub mod xy_tokenizer;
// SoTA plan Phase 1-5 (2026-07-24): Zyphra Zonos-v0.1-transformer
// (Apache 2.0) safetensors → GGUF with the `vokra.zonos.*` chunk group.
// Every float tensor passes through verbatim; every hparam (including the
// 7 typed prefix conditioners) is transcribed from the upstream config.json.
pub(crate) mod zonos;

// ---------------------------------------------------------------------------
// TIER 1+2 audio-gap implementation wave (2026-07-30 ultracode workflow
// `wf_022575ce-077`, 7 parallel worktree implementers). Each module is a
// self-contained BF16 pass-through converter mirror of the wespeaker /
// rmvpe / neucodec pattern. **CLI wiring (ModelKind / from_arg / dispatch
// arm / as_arg / license class registry) is DEFERRED to a follow-up wave**
// because integrating 7 worktrees' changes to the 5 shared files
// (crates/vokra-cli/src/convert.rs, crates/vokra-convert/src/lib.rs,
// crates/vokra-convert/src/main.rs, crates/vokra-convert/src/models/mod.rs,
// crates/vokra-core/src/compliance/license_class.rs) hit fundamental
// 25-commit drift between the WT common base (`d05ab7d`) and the current
// branch tip (pyannote Wave 3+4). See `docs/handoff/tier1-tier2-audio-
// impl-2026-07-30.md` for the deferred-wiring detail + per-model
// primary-source URLs + license sign-off list.
//
// Under this partial land the modules exist as library-callable code
// (`crate::models::qwen3_asr::convert_qwen3_asr_file` etc.); the
// `vokra-cli convert --model qwen3-asr` CLI dispatch is the follow-up.
pub mod ast;
pub mod audiobox_aesthetics;
// 2026-08-01 Wave 5 music-separation add: BS-Roformer / Mel-Band Roformer
// (Lu et al. 2023 arXiv:2310.01809 "Music Source Separation with Band-Split
// RoPE Transformer", **weight provenance unclear**). Third-party mirror
// `chenmozhijin/BSRoformer-GGUF` aggregates converted GGUFs from multiple
// trainers under mixed licenses (GPL-3.0 / CC-BY-NC-4.0 / unspecified); the
// clean-room MIT reference lives at `github.com/lucidrains/BS-RoFormer` but
// ships no pretrained weights. `category = "separation"` (sibling of the
// SepFormer speech-separation family — BS-Roformer is the music-vocals
// analogue, mask an STFT spectrogram to isolate vocals / drums / bass /
// other stems). BF16 pass-through skeleton mirror of vits_ja / musicgen_large.
// **LicenseClass::RedistributionForbidden default** — a converter cannot know
// which SPDX id covers the caller's checkpoint, so the fail-closed publish
// gate applies until a caller supplies `--license <spdx>` at the outer
// boundary (same escape hatch vits-ja / Whisper / kokoro use). **Publish
// blocked (unclear-provenance-defer)** — no entry in
// `scripts/publish/signoff_match.py::REPO_TO_SIGNOFF_ROWS`, §3.1 sign-off
// blank (owner ADR selecting a specific checkpoint + license required).
// Real-weight parity + runtime binder (new op surface: band-split RoPE
// transformer with time-axis + band-axis alternating attention, mask
// estimator) deferred to owner sign-off.
pub mod bs_roformer;
// 2026-08-01 Wave 5 music-generation add: AudioLDM 2 (`cvssp/audioldm2`,
// **cc-by-nc-sa-4.0**). Text-to-audio latent-diffusion generator
// (Liu et al. 2024 ICML, arXiv:2308.05734) — VAE encoder/decoder + U-Net
// latent diffusion + HiFi-GAN vocoder + GPT-2 audio-caption LM + T5-base
// + CLAP text encoder, ~8.5 GB bundle. `category = "music"` (shared with
// sibling musicgen family per 2026-07-30 scope expansion). BF16
// pass-through skeleton mirror of musicgen_medium / xcodec2. Doubly-
// restrictive `LicenseClass::NonCommercialShareAlike` default — NC gate
// + SA cascade both fail-closed; **publish blocked** (no
// REPO_TO_SIGNOFF_ROWS entry, no §3.1 sign-off ☑ — owner ADR required
// to resolve the SA cascade onto Vokra-added artifacts). Real-weight
// parity + runtime binder (new op surface: latent-diffusion sampler +
// VAE + HiFi-GAN — distinct from `flow_sampler` which targets flow-
// matching) deferred to owner sign-off (`docs/license-audit.md` §3.1
// sign-off queue). Scale ~8.5 GB = vast.ai handoff per memory
// [[feedback-large-models-on-vast-ai]] (M1 iMac 16 GB unsafe on the
// upper edge — multi-encoder bundle doubles peak resident to ~17 GB).
pub mod audioldm2;
pub mod bark;
pub mod bigvgan;
pub mod clap;
pub mod deepfake_detection;
pub mod firered_vad;
pub mod focalcodec;
// 2026-08-01 wave: IBM Granite Speech 4.1-2B (`ibm-granite/granite-speech-4.1-2b`,
// apache-2.0). Category = `asr` (audio-LLM ASR). Conformer CTC encoder
// (16L × d_model 1024 × 8 heads × 128 head_dim × conv_kernel 15) +
// Granite-4.0-1b-base LLM decoder (40L × hidden 2048 × GQA 16Q ÷ 4KV ×
// ffn 4096 × RoPE θ 10000 × RMSNorm ε 1e-5 × vocab 100 353, distinctive
// Granite scalars attention_multiplier 0.0078125 / embedding_multiplier
// 12.0 / logits_scaling 8.0 / residual_multiplier 0.22) + BLIP-2 q-former
// projector (2L × hidden 1024 × 16 heads × downsample_rate 5) + optional
// LoRA adapter. BF16 pass-through skeleton mirror of speecht5_hifigan /
// canary_qwen; real-weight parity + runtime binding deferred to owner
// (`docs/license-audit.md` §3.1 sign-off).
pub mod granite_speech;
pub mod hifigan_vocoder;
pub mod kyutai_tts;
pub mod melotts;
pub mod metricgan_plus;
// 2026-08-01 Wave 3 codec add: OpenMOSS MOSS-Audio-Tokenizer
// (`OpenMOSS-Team/MOSS-Audio-Tokenizer` + `-Nano`, apache-2.0). The
// codec half of the MOSS-TTS pipeline (waveform → discrete tokens
// fed into the sibling `moss_tts` LLM). BF16 pass-through skeleton
// mirror of snac / neucodec / focalcodec (variant-taking = Full
// ~1.77B params 6.6 GB / Nano ~22M params 88 MB). Both variants ship
// as sharded safetensors + `model.safetensors.index.json` weight-map;
// callers pre-flatten via `tools/parity/moss_audio_tokenizer_prepare_checkpoint.py`.
// Real-weight parity + runtime binder deferred to owner sign-off (§3.1).
pub mod moss_audio_tokenizer;
pub mod moss_tts;
// 2026-08-01 Wave 5 music-generation add: Meta AudioCraft MusicGen-Medium
// (`facebook/musicgen-medium`, **cc-by-nc-4.0**). First music-generation
// target to land a converter (post-2026-07-30 scope expansion
// `[[project-scope-expansion-2026-07-30]]`). 1.5B autoregressive transformer
// LM over EnCodec RVQ tokens conditioned on frozen T5 text encoder
// (Copet et al. 2023, arXiv:2306.05284). BF16 pass-through skeleton mirror
// of xcodec2 / wavtokenizer — the same T4 (Research-only) tier as X-Codec 2
// with `LicenseClass::NonCommercial` fail-closed default. First use of
// `category = "music"` in the tree (distinct from the speech-tree tags
// tts / asr / codec / vocoder / s2s / vad / speaker / f0 / separator /
// bert). Real-weight parity + runtime binder deferred to owner sign-off
// (`docs/license-audit.md` §3.1 sign-off queue). Scale ~11.4 GB = vast.ai
// handoff per memory [[feedback-large-models-on-vast-ai]] (M1 iMac 16 GB
// unsafe for this class of publish).
pub mod musicgen_medium;
// 2026-08-01 Wave 5 music-generation add: Meta AudioCraft MusicGen-Large
// (`facebook/musicgen-large`, **cc-by-nc-4.0**). 3.3B autoregressive
// transformer LM over EnCodec RVQ tokens conditioned on frozen T5 text
// encoder (top rung of the MusicGen family — `-small` 300M / `-medium`
// 1.5B / **`-large` 3.3B**). Sibling file to musicgen_medium.rs (the
// chatterbox / chatterbox_turbo / chatterbox_nano split) rather than a
// shared musicgen.rs variant enum — zero-churn on the medium landing +
// distinct upstream HF repo. Same T4 (Research-only) tier as X-Codec 2
// and MusicGen-Medium with `LicenseClass::NonCommercial` fail-closed
// default. Real-weight parity + runtime binder deferred to owner sign-off
// (`docs/license-audit.md` §3.1 sign-off queue). Scale ~19.5 GB = vast.ai
// handoff per memory [[feedback-large-models-on-vast-ai]] (M1 iMac 16 GB
// unsafe for this class of publish — larger than the sibling MusicGen-
// Medium ~11.4 GB, both routed to vast.ai per the runbook).
pub mod musicgen_large;
// 2026-08-01 Wave 5 audio-generation add: Meta AudioCraft AudioGen-Medium
// (`facebook/audiogen-medium`, **cc-by-nc-4.0**). 1.5B autoregressive
// transformer LM over EnCodec RVQ tokens conditioned on frozen T5 text
// encoder — MusicGen sibling with identical topology, tuned on
// environmental sounds / SFX (not music). Shares `musicgen` arch tag
// and `music` category with MusicGen family. Same T4 (Research-only)
// tier as X-Codec 2 / MusicGen-Medium / MusicGen-Large with
// `LicenseClass::NonCommercial` fail-closed default. Scale ~3.7 GB =
// local convert safe on M1 iMac 16 GB (below the vast.ai threshold).
pub mod audiogen_medium;
// 2026-08-01 Wave 6 residual: Meta AudioCraft MusicGen-Small
// (`facebook/musicgen-small`, cc-by-nc-4.0). 300M smallest of the
// MusicGen family. Shares `musicgen` arch tag + `music` category.
// Scale ~5.5 GB = vast.ai handoff per owner directive.
pub mod musicgen_small;
// 2026-08-01 Wave 6 residual: Alibaba Qwen2-Audio-7B-Instruct
// (`Qwen/Qwen2-Audio-7B-Instruct`, apache-2.0). Whisper audio
// encoder + Qwen2-7B LM = audio-LLM omni. Distinct arch tag
// `qwen2_audio`. Scale ~16 GB (5-shard) = vast.ai handoff.
pub mod qwen2_audio;
// 2026-08-01 Wave 6 residual: Microsoft VibeVoice-ASR
// (`microsoft/VibeVoice-ASR`, MIT). VibeVoice sibling with ASR
// head. Distinct arch tag `vibevoice_asr`. Scale ~16.5 GB = vast.ai.
pub mod vibevoice_asr;
// 2026-08-01 Wave 6 residual: ACE-Step 1.5 (`ACE-Step/Ace-Step1.5`,
// MIT). Multi-component music-generation bundle. Distinct arch tag
// `ace_step`, category `music`. Scale ~9.6 GB = vast.ai handoff.
pub mod ace_step;
// 2026-08-01 Wave 7 residual: Meta HuBERT-Large-LS960
// (`facebook/hubert-large-ls960-ft`, apache-2.0). 317M self-supervised
// speech encoder (BERT-style masked-feature prediction, distinct from
// wav2vec 2.0's contrastive convnet + Gumbel objective) + CTC head
// fine-tuned on LibriSpeech 960h. Distinct arch tag `hubert` — future
// native forward is expected to share ops with `wav2vec2_ctc`
// (Conv1D feature-extractor + Transformer encoder + CTC decode) but
// the arch tag stays distinct so runtime dispatch cannot misroute a
// HuBERT checkpoint into a wav2vec2 loader silently (FR-EX-08).
// Scale ~1.26 GB = local convert safe on M1 iMac 16 GB.
pub mod hubert_large_ls960;
// 2026-08-01 Wave 3 codec add: Amphion NaturalSpeech 3 FACodec
// (`amphion/naturalspeech3_facodec`, apache-2.0). Factorized VQ (FVQ)
// neural audio codec at 16 kHz, 3 parallel quantizer heads over
// disentangled subspaces (prosody 1cb + content 2cb + detail 3cb, 6
// codebooks total). Distinct arch tag `facodec` — first FVQ codec in
// the tree, silently sharing with sibling RVQ (Mimi/DAC/SNAC) or FSQ
// (WavTokenizer/X-Codec 2) codecs would misroute runtime dispatch.
// BF16 pass-through skeleton mirror of snac / neucodec /
// moss_audio_tokenizer. Upstream ships 5 separate `.bin` pickles;
// callers pre-merge the variant subset via
// `tools/parity/naturalspeech3_facodec_prepare_checkpoint.py` (the
// sepformer multi-file bridge precedent). Real-weight parity + runtime
// binder deferred to owner sign-off (§3.1). Redecoder variants enable
// zero-shot voice conversion — owner routing decision whether they
// belong in main zoo or `vokra-voiceclone-experimental` (ELVIS Act
// policy, CLAUDE.md 設計判断 8).
pub mod mp_senet;
pub mod naturalspeech3_facodec;
pub mod nemotron_asr;
pub mod parler;
pub mod qwen3_asr;
pub mod sepformer;
// SNAC — Multi-Scale Neural Audio Codec (Siuzdak et al. 2024,
// hubertsiuzdak/snac_{24khz,44khz}, MIT). BF16 pass-through skeleton
// mirror of focalcodec / bigvgan (variant-taking, sample-rate specialised).
// 3 RVQ levels @ ~12/23/47 Hz for the 24 kHz variant (no attention);
// 4 RVQ levels + 32-frame local attention for the 44.1 kHz music-quality
// variant. Consumed by Orpheus-TTS + MOSS voice family + CSM-1B-adjacent
// stacks (upstream ~452k monthly downloads on the 24 kHz release).
pub mod smart_turn;
pub mod snac;
pub mod speechbrain_lang_id;
pub mod speecht5;
pub mod speecht5_hifigan;
pub mod tiger;
pub mod vieneu;
// 2026-08-01 wave: Charactr AI Vocos vocoder (`charactr/vocos-mel-24khz`
// = HF audio-vocoder category top by download, 2.85M dl; and
// `charactr/vocos-encodec-24khz`, both MIT). Category: vocoder.
// Fourier-space vocoder (ConvNeXt V2 backbone + iSTFT head,
// arXiv:2306.00814) — distinct arch tag `vocos` from every HiFi-GAN
// sibling because Vocos does NOT time-domain upsample + MRF; it
// spectrum-space processes then inverse-STFTs. Every F32 / F16 /
// BF16 tensor passes through verbatim following the
// speecht5_hifigan / bigvgan / focalcodec BF16 pass-through
// contract; real-weight parity is deferred to owner
// (`docs/license-audit.md` §3.1 sign-off). Upstream ships torch
// pickle `pytorch_model.bin` + `config.yaml` only — pre-flatten to
// safetensors offline via
// `tools/parity/vocos_prepare_checkpoint.py` (a thin wrapper over
// `bin_to_safetensors.py`, mirror of speecht5_hifigan).
pub mod vocos;
pub mod wav2vec2_ctc;
pub mod xvector;
// Wave 3 codec add (2026-08-01): novateur/WavTokenizer-large-speech-75token
// (MIT). Single-codebook FSQ audio codec at 24 kHz, 75 tokens/sec
// (hop_length=320, arXiv:2408.16532). Upstream ships a torch pickle
// Lightning `.ckpt` (1.75 GB) so callers pre-flatten to safetensors via
// a dedicated `tools/parity/wavtokenizer_prepare_checkpoint.py` bridge
// (the DFN3 / DAC / CSM / SpeechT5-HiFi-GAN precedent — no pickle in the
// runtime, NFR-DS-02 zero-dep + FR-LD-05). Real-weight parity deferred
// to owner sign-off (§3.1). Runtime forward reuses the M4-16 landed
// `wavtokenizer_vq` op (`crates/vokra-ops/src/fsq_codec.rs`).
pub mod wavtokenizer;
// 2026-08-01 Wave 3 sibling-pair (codec + vocoder) add: YuE bundle
// (`m-a-p/YuE-upsampler` + `m-a-p/xcodec_mini_infer`, apache-2.0). The
// codec / vocoder half of the YuE full-song music-generation system
// (Yuan et al. 2025, arXiv:2503.08638). Two distinct HF repos share
// one Rust converter module: `YueUpsampler` variant = Vocos backbone
// + iSTFT head @ 44.1 kHz (145 MB); `YueXcodecMini` variant =
// SoundStream RVQ codec + HuBERT-base semantic encoder + Vocos decoder
// head @ 16 kHz / 25 Hz (~2.2 GB). Both upstream repos ship torch
// pickle only (`.pth` / `.bin`) — callers pre-flatten via
// `tools/parity/yue_bundle_prepare_checkpoint.py` (multi-file bridge
// mirror of `naturalspeech3_facodec_prepare_checkpoint.py` +
// `sepformer_prepare_checkpoint.py` + `bin_to_safetensors.py`). BF16
// pass-through skeleton mirror of vocos / snac / focalcodec /
// speecht5_hifigan; runtime binder + real-weight parity deferred to
// owner sign-off (§3.1). Distinct arch tags `yue_upsampler` +
// `yue_xcodec_mini` from every sibling — silently sharing with vocos
// (different config axes / training corpus) or with any RVQ / FSQ
// codec (semantic-encoder fusion is what distinguishes YuE) would
// mis-route runtime dispatch. Attribution note: RepCodec (ByteDance
// / Chutong Meng, MIT) + Descript-Audio-Codec (MIT) source trees
// ship inside `xcodec_mini_infer` at `RepCodec/` /
// `descriptaudiocodec/dac/` but are inference-tree artefacts, not
// loaded weights — NOTICE keeps credit for design influence, this
// converter ignores them.
pub mod yue_bundle;
// 2026-08-02 Wave residual: dscripka/openWakeWord (custom-KWS MLP/CNN
// over precomputed melspec, apache-2.0). Small wake-word family
// (~1–5 MB each) — audio-dialect `kws` op entry (FR-OP `kws`).
// Distinct arch tag `openwakeword`, category `kws`. Scale ~0.01 GB =
// local convert safe on M1 iMac (well below vast.ai threshold).
pub mod openwakeword;
// 2026-08-02 Wave residual: UsefulSensors/moonshine-tiny (27M raw-audio
// transformer enc-dec ASR, MIT). Distinct from Whisper: no mel front-end
// (raw 16 kHz audio via Conv1D stack) + rotary + SwiGLU. Distinct arch
// tag `moonshine`, category `asr`. Scale ~0.11 GB = local convert safe.
pub mod moonshine_tiny;
// 2026-08-02 Wave residual: UsefulSensors/moonshine-base (61.5M raw-audio
// transformer enc-dec ASR, MIT). Sibling to `moonshine_tiny` — same
// arch family (raw-audio Conv1D + rotary + SwiGLU), wider/deeper
// backbone. Distinct arch tag `moonshine` (shared with sibling Tiny),
// category `asr`. Scale ~0.25 GB = local convert safe.
pub mod moonshine_base;
// 2026-08-02 Wave residual: facebook/demucs (HT-Demucs, MIT). Hybrid
// transformer Demucs (Rouard et al. 2023, arXiv:2211.08553) — U-Net
// waveform branch + spectrogram branch joined by cross-domain self-
// attention, 4-source music separation (drums / bass / other / vocals).
// Distinct from SepFormer (waveform-only dual-path Transformer) and
// TIGER (time-frequency dual-branch) siblings; distinct arch tag
// `demucs`, category `separation`. Scale ~0.50 GB = local convert safe.
pub mod demucs_htdemucs;
// 2026-08-02 Wave residual: fixie-ai/ultravox-v0_5-llama-3_2-1b (MIT).
// Ultravox v0.5 (Llama-3.2-1B) — audio-text-to-text multimodal model =
// Llama-3.2-1B decoder + Whisper encoder + lightweight projection adapter.
// Both underlying arches (Llama + Whisper) already supported by sibling
// converters + runtime primitives; new wiring is the adapter projection +
// multimodal prompt template (runtime-side, not converter-side). Distinct
// arch tag `ultravox` from sibling Voxtral (Mistral decoder) / Qwen2-Audio
// (Qwen2 decoder) — the decoder backbone fixes tensor layout + tokenizer +
// rope base, so FR-EX-08 forbids silent shape misroute across the three.
// Category `audio-llm`. Scale ~1.83 GB = local convert safe.
pub mod ultravox_v0_5_llama_3_2_1b;
// 2026-08-02 Wave residual: coqui/XTTS-v2 (multilingual zero-shot voice-
// cloning TTS = GPT-2 backbone + DVAE + HiFi-GAN, coqui-public-model-license
// = NonCommercial T4 tier). ~1.90 GB = local convert safe on M1 iMac.
// Distinct arch tag `xtts` from sibling piper-plus (VITS2) / Kokoro
// (iSTFTNet) / CosyVoice2 (FSQ + HiFTNet) — FR-EX-08.
pub mod xtts_v2;
// 2026-08-02 Wave residual: JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k
// (Asteroid ConvTasNet, cc-by-sa-4.0). First Copyleft-tier separator
// entry — single-speaker enhancement head on Libri1Mix 16 kHz (one
// clean speaker + additive noise, one output stream). Distinct arch
// tag `conv_tasnet` from sibling separator families (sepformer /
// demucs / tiger_separator / bs_roformer / mp_senet) — FR-EX-08
// forbids silent shape misroute across separator families. Category
// `enhancement` (mirrors SepFormer WHAM / WHAMR / DNS-4 sibling
// posture). Weight license Copyleft (SA cascade — a derived GGUF is
// itself CC-BY-SA), T3 tier redistributable with original licence
// preserved. Scale ~20 MB = local convert safe on M1 iMac.
pub mod conv_tasnet_libri1mix;
// ---------------------------------------------------------------------------
