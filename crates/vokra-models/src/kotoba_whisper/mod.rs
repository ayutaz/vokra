//! Kotoba Technologies **kotoba-whisper** — Whisper large-v3 encoder +
//! a 2-layer decoder distilled on Japanese ReazonSpeech audio (SoTA plan
//! Phase 5 JA-ASR-2, 2026-07-24).
//!
//! # What kotoba-whisper is (primary source)
//!
//! `kotoba-tech/kotoba-whisper-v2.0` is a Japanese-distilled Whisper
//! checkpoint: the **large-v3 encoder is kept intact** (32 layers,
//! d_model=1280, n_mels=128, encoder_attention_heads=20) and the
//! **decoder is shrunk to 2 layers** (same width / head count as
//! large-v3). The tokenizer is the large-v3 multilingual byte-level
//! BPE (`vocab_size=51866`, `eos_token_id=50257`,
//! `decoder_start_token_id=50258`).
//!
//! Every hparam below is transcribed **verbatim** from
//! `huggingface.co/kotoba-tech/kotoba-whisper-v2.0/raw/main/config.json`
//! (fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」):
//!
//! - **Model type** (`model_type`): `"whisper"`
//!   (`architectures = ["WhisperForConditionalGeneration"]`). The
//!   underlying architecture is Whisper, only the decoder depth changes,
//!   which is why this module is a thin scaffold over the existing
//!   [`crate::whisper`] plumbing — every op / kernel / attention detail
//!   is shared verbatim.
//! - **Encoder** (identical to Whisper `large-v3`):
//!   - `d_model` = 1280,
//!   - `encoder_layers` = 32,
//!   - `encoder_attention_heads` = 20 (`head_dim = 1280 / 20 = 64`,
//!     the Whisper family invariant),
//!   - `encoder_ffn_dim` = 5120,
//!   - `num_mel_bins` = 128 (large-v3 log-mel front-end),
//!   - `max_source_positions` = 1500 (encoder positional length —
//!     `n_audio_ctx`).
//! - **Decoder** (the distil axis — this is where it differs from
//!   large-v3):
//!   - `decoder_layers` = **2** (large-v3 has 32),
//!   - `decoder_attention_heads` = 20 (unchanged),
//!   - `decoder_ffn_dim` = 5120 (unchanged),
//!   - `max_target_positions` = 448 (decoder positional length —
//!     `n_text_ctx`).
//! - **Tokenizer**:
//!   - `vocab_size` = 51866 (large-v3 multilingual +1 for `<|yue|>`),
//!   - `eos_token_id` = 50257 (`<|endoftext|>`),
//!   - `decoder_start_token_id` = 50258 (`<|startoftranscript|>`),
//!   - `pad_token_id` = 50256.
//! - **Audio boundary**: `sample_rate` = 16 000 (Whisper convention).
//! - **Weight license**: **Apache-2.0** per every HF model card in the
//!   family (`kotoba-tech/kotoba-whisper-v1.0`, `-v1.1`, `-v2.0`,
//!   `-v2.1`, `-bilingual-v1.0`) — resolves to
//!   [`vokra_core::LicenseClass::Permissive`] via the `kotoba-whisper-`
//!   family walk, so the M2-13 gate passes commercially without any
//!   attribution obligation on the runtime side.
//!
//! # Distinct from distil-whisper (same shape, different upstream)
//!
//! `kotoba-whisper` and `distil-whisper/distil-large-v3.5` share the
//! **exact same architectural shape** (`d_model=1280`, `n_audio_layer=32`,
//! `n_text_layer=2`, `n_mels=128`, `vocab_size=51866`,
//! `ffn_dim=5120`). Both are Whisper large-v3 with a shrunk 2-layer
//! decoder. The differentiators are:
//!
//! - **Distillation corpus**: kotoba-whisper is distilled on
//!   ReazonSpeech Japanese audio (7000+ hours); distil-large-v3.5
//!   is distilled on multilingual audio.
//! - **License**: kotoba-whisper is **Apache-2.0** (upstream); distil-
//!   whisper is **MIT** (upstream). The compliance registry resolves
//!   both to Permissive, but the GGUF provenance stamp differs.
//! - **Upstream identifier**: `kotoba-tech/kotoba-whisper-vN.N` vs
//!   `distil-whisper/distil-large-v3.5`.
//! - **Language specialization**: kotoba-whisper is Japanese-specialized
//!   (JA CER 6-9% on the eval triad); distil-whisper covers many
//!   languages with more moderate quality per language.
//!
//! Because the *runtime forward* is identical (same op inventory, same
//! tensor names, same shape flow), both share the [`crate::whisper`]
//! plumbing. This module carries the primary-source config + a distinct
//! arch tag ("kotoba-whisper") so provenance / telemetry / model cards
//! label the loaded model correctly.
//!
//! # Very-cheap follow-on — reuses Whisper verbatim
//!
//! Because the topology is a Whisper checkpoint whose only difference is
//! `n_text_layer = 2`, kotoba-whisper **does not add any new op**
//! (`vokra-ops`) or backend kernel: the same STFT / mel filterbank / GEMM /
//! GEMV / softmax / layer-norm / GELU / conv1d inventory Whisper base
//! consumes (see [`crate::whisper`] docstring §Operator inventory) is
//! also what kotoba-whisper uses. The runtime forward is a follow-up
//! wave (T29-equivalent — the Moshi / CSM / Zonos / Kyutai STT /
//! Parakeet-CTC / distil-whisper pattern): when it lands it will
//! delegate to [`crate::whisper::WhisperModel`] with an appropriately-
//! shrunk `WhisperConfig`, since the checkpoint's tensor names follow
//! the upstream HF Whisper convention verbatim
//! (`model.encoder.layers.*` / `model.decoder.layers.*`) and the
//! converter (`vokra-convert::models::kotoba_whisper`) writes them
//! through unchanged.
//!
//! # What lands in this Phase 5 slice
//!
//! - [`KotobaWhisperConfig`] — every hparam transcribed from the primary
//!   source (v2.0 canonical), plus a `distil_invariant` sanity check
//!   (`n_text_layer < n_audio_layer`) that catches a checkpoint whose
//!   decoder depth was left at the source (large-v3 = 32) instead of the
//!   shrunk kotoba count.
//! - [`KotobaWhisperAsr`] — engine handle carrying config.
//!   [`KotobaWhisperAsr::transcribe`] returns
//!   [`VokraError::NotImplemented`] until real weights are bound (the
//!   real forward — log-mel front-end → 32-layer encoder → 2-layer
//!   decoder → BPE detokenize — is a follow-up wave gated on the real
//!   HF checkpoint T29 hand-off).
//!
//! # No ONNX (permanent)
//!
//! `kotoba-tech/kotoba-whisper-*` ships PyTorch safetensors; the
//! pipeline is re-implemented natively via [`crate::whisper`]
//! (whisper.cpp 型, CLAUDE.md 設計判断 4). This module never touches ONNX.

use vokra_core::{Result, VokraError};

/// `vokra.model.arch` a kotoba-whisper GGUF must carry. Written by
/// `vokra-convert::models::kotoba_whisper::ARCH`; the compliance registry
/// (`vokra_core::compliance`) knows `kotoba-whisper` / `kotoba-whisper-v1.0` /
/// `kotoba-whisper-v2.0` (and every family variant that lands later) as
/// [`vokra_core::LicenseClass::Permissive`] via the `kotoba-whisper-`
/// family prefix walk (apache-2.0 — the M2-13 gate passes commercially).
///
/// This arch string is intentionally **distinct** from Whisper's
/// (`"whisper"`) and distil-whisper's (`"distil-whisper"`) so the runtime
/// can label the loaded model correctly in telemetry / logs / model cards
/// while still consuming the same `vokra.whisper.*` hparam chunk schema
/// and Whisper decoder plumbing — the "very cheap follow-on" contract
/// in the task.
pub const EXPECTED_ARCH: &str = "kotoba-whisper";

/// PCM sample rate kotoba-whisper expects. Same as vanilla Whisper —
/// 16 kHz mono, per the openai/whisper convention (not written directly
/// in `config.json` but inherited from the Whisper feature extractor
/// preprocessor).
pub const KOTOBA_WHISPER_SAMPLE_RATE: u32 = 16_000;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// kotoba-whisper architectural hyperparameters.
///
/// A deliberate subset of the Whisper `config.json` schema — every field
/// maps 1-to-1 to the corresponding Whisper axis (see
/// [`crate::whisper::WhisperConfig`]). The distinguishing invariant is
/// **`n_text_layer < n_audio_layer`** ("the decoder is smaller than the
/// encoder"), enforced by [`Self::validate_for_forward`] — a real
/// non-distil checkpoint would have `n_text_layer == n_audio_layer`
/// (Whisper base…large-v3) or `n_text_layer < n_audio_layer` only for
/// the turbo variant (which is a separate arch, `"whisper"`, at
/// `whisper-turbo`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KotobaWhisperConfig {
    /// Mel input channels (encoder conv1 in-channels). **128** for
    /// kotoba-whisper v1.x/v2.x (matching large-v3's 128-bin front-end).
    pub n_mels: usize,
    /// Hidden width `d_model` shared by encoder and decoder — 1280 for
    /// kotoba-whisper v1.x/v2.x.
    pub d_model: usize,
    /// Encoder positional length (`max_source_positions`), 1500.
    pub n_audio_ctx: usize,
    /// Encoder attention heads — 20 for kotoba-whisper v1.x/v2.x
    /// (`head_dim = 1280 / 20 = 64`).
    pub n_audio_head: usize,
    /// Encoder block count. **32** for kotoba-whisper v1.x/v2.x — the
    /// family keeps the large-v3 encoder intact.
    pub n_audio_layer: usize,
    /// Decoder positional length (`max_target_positions`), 448.
    pub n_text_ctx: usize,
    /// Decoder attention heads — same as `n_audio_head` for kotoba-whisper.
    pub n_text_head: usize,
    /// Decoder block count. **2** for kotoba-whisper v1.x/v2.x — the
    /// distil axis (large-v3 has 32). This is the JA-ASR-2 axis: the
    /// converter and runtime must both honor it as data-driven from
    /// GGUF metadata (never hard-coded to 32).
    pub n_text_layer: usize,
    /// Token vocabulary size — **51 866** for kotoba-whisper (the
    /// large-v3 multilingual vocab including `<|yue|>`).
    pub n_vocab: usize,
    /// Feed-forward inner width — 5120 for kotoba-whisper v1.x/v2.x.
    pub ffn_dim: usize,
    /// End-of-transcript token id (decode stop condition) — 50257 for
    /// the Whisper multilingual tokenizer.
    pub eot: u32,
    /// Start-of-transcript token id (decoder prompt seed) — 50258 for
    /// the Whisper multilingual tokenizer.
    pub sot: u32,
    /// PCM sample rate — 16 000 (Whisper convention).
    pub sample_rate: u32,
}

impl KotobaWhisperConfig {
    /// Per-head width. Whisper fixes this at 64 across every family
    /// size, so it is simply `d_model / n_audio_head` (validated
    /// non-zero and exact in [`Self::validate_for_forward`]).
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.d_model.checked_div(self.n_audio_head).unwrap_or(0)
    }

    /// Primary-source `kotoba-tech/kotoba-whisper-v2.0` config (every
    /// value transcribed verbatim from the upstream `config.json` — see
    /// module docstring). The v1.0 / v1.1 / v2.1 releases share the
    /// same architectural quintuple; only the distilled weights differ.
    #[must_use]
    pub fn kotoba_whisper_v2_0() -> Self {
        Self {
            n_mels: 128,
            d_model: 1280,
            n_audio_ctx: 1500,
            n_audio_head: 20,
            n_audio_layer: 32,
            n_text_ctx: 448,
            n_text_head: 20,
            n_text_layer: 2,
            n_vocab: 51_866,
            ffn_dim: 5120,
            eot: 50_257,
            sot: 50_258,
            sample_rate: KOTOBA_WHISPER_SAMPLE_RATE,
        }
    }

    /// Miniature well-formed config for shape / stability tests. Dims
    /// are tiny so synthesized-weight builds fit in KB; the *shape
    /// relationships* (encoder deeper than decoder, MHA head split,
    /// even head_dim, vocab non-zero) mirror the real model.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            n_mels: 16,
            d_model: 16,
            n_audio_ctx: 24,
            n_audio_head: 4,
            n_audio_layer: 4,
            n_text_ctx: 16,
            n_text_head: 4,
            n_text_layer: 2,
            n_vocab: 32,
            ffn_dim: 32,
            eot: 30,
            sot: 31,
            sample_rate: KOTOBA_WHISPER_SAMPLE_RATE,
        }
    }

    /// Rejects `0`-placeholder / ill-formed configs before any forward
    /// runs (FR-EX-08 — a shape-only converter path fails loudly here,
    /// not deep inside a GEMM).
    ///
    /// Enforces the Whisper cross-checks (`d_model % n_head == 0` for
    /// both encoder and decoder, non-zero axes, EOT/SOT inside the
    /// vocab) plus the distil invariant
    /// (`n_text_layer < n_audio_layer`).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate_for_forward(&self) -> Result<()> {
        if self.d_model == 0
            || self.n_mels == 0
            || self.n_audio_ctx == 0
            || self.n_audio_layer == 0
            || self.n_text_ctx == 0
            || self.n_text_layer == 0
            || self.n_vocab == 0
            || self.ffn_dim == 0
            || self.n_audio_head == 0
            || self.n_text_head == 0
            || self.sample_rate == 0
        {
            return Err(VokraError::InvalidArgument(
                "kotoba-whisper config: every architectural axis must be > 0".to_owned(),
            ));
        }
        if self.d_model % self.n_audio_head != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kotoba-whisper config: n_audio_head ({}) must divide d_model ({})",
                self.n_audio_head, self.d_model,
            )));
        }
        if self.d_model % self.n_text_head != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kotoba-whisper config: n_text_head ({}) must divide d_model ({})",
                self.n_text_head, self.d_model,
            )));
        }
        if self.head_dim() % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kotoba-whisper config: head_dim {} must be even (attention K/V pair layout)",
                self.head_dim(),
            )));
        }
        // The distil axis: a kotoba-whisper checkpoint has fewer decoder
        // layers than encoder layers. A checkpoint where the two are
        // equal is (a) a real Whisper (large-v3, medium, etc.) that
        // landed on the kotoba path by mistake, or (b) a mis-flattened
        // checkpoint where the decoder tensors were duplicated to the
        // encoder count. Either way this must fail loudly (FR-EX-08).
        if self.n_text_layer >= self.n_audio_layer {
            return Err(VokraError::InvalidArgument(format!(
                "kotoba-whisper config: n_text_layer ({}) must be < n_audio_layer ({}); \
                 kotoba-whisper is Japanese-distilled from Whisper large-v3 with a shrunk \
                 decoder, so equal or larger decoder depth means this is not a kotoba-whisper \
                 checkpoint (use --model whisper for vanilla Whisper sizes)",
                self.n_text_layer, self.n_audio_layer,
            )));
        }
        if (self.eot as usize) >= self.n_vocab {
            return Err(VokraError::InvalidArgument(format!(
                "kotoba-whisper config: eot ({}) must be < n_vocab ({})",
                self.eot, self.n_vocab,
            )));
        }
        if (self.sot as usize) >= self.n_vocab {
            return Err(VokraError::InvalidArgument(format!(
                "kotoba-whisper config: sot ({}) must be < n_vocab ({})",
                self.sot, self.n_vocab,
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// kotoba-whisper ASR engine handle.
///
/// Carries the resolved config. [`Self::transcribe`] is the primary
/// waveform → text entry point; until real weights are bound (see the
/// module docstring) it returns [`VokraError::NotImplemented`] with a
/// message naming the blocker (FR-EX-08 — never a silent zero-fill or
/// empty transcript).
///
/// # Data-driven decoder depth (JA-ASR-2)
///
/// The [`config`](Self::config) carries `n_text_layer = 2` and the
/// engine honors it end-to-end — the runtime never assumes 32 layers
/// (that would be a large-v3-specific hard-code). This is the JA-ASR-2
/// entry point: the shared [`crate::whisper::WhisperConfig`] loader
/// reads `n_text_layer` from GGUF metadata, and every downstream
/// component (weight binding, KV cache, greedy loop, beam search)
/// iterates over `w.layers.len()` — no fixed constant.
#[derive(Debug, Clone)]
pub struct KotobaWhisperAsr {
    cfg: KotobaWhisperConfig,
}

impl KotobaWhisperAsr {
    /// Assembles an engine from `cfg`. Config is cross-checked so a
    /// mismatched shape fails loudly here rather than deep inside a
    /// forward.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] from `cfg.validate_for_forward`.
    pub fn new(cfg: KotobaWhisperConfig) -> Result<Self> {
        cfg.validate_for_forward()?;
        Ok(Self { cfg })
    }

    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &KotobaWhisperConfig {
        &self.cfg
    }

    /// Transcribes a mono `f32` PCM slice at [`Self::config`]'s sample
    /// rate.
    ///
    /// This is the primary waveform → text entry point. **Real weights
    /// required**: this scaffold does not yet bind kotoba-whisper
    /// weights, so it returns [`VokraError::NotImplemented`] naming the
    /// blocker. Callers verify the shape flow through
    /// [`KotobaWhisperAsr::new`] today; a follow-up wave binds real
    /// kotoba-whisper weights and wires the forward through
    /// [`crate::whisper::WhisperModel`] with the kotoba-shrunk decoder
    /// depth (`n_text_layer = 2`) — the JA-ASR-2 payload.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `pcm` is empty.
    /// - [`VokraError::NotImplemented`] otherwise (real forward not yet
    ///   bound — FR-EX-08).
    pub fn transcribe(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "kotoba-whisper transcribe: pcm slice is empty".to_owned(),
            ));
        }
        Err(VokraError::NotImplemented(
            "kotoba-whisper transcribe: the 16 kHz waveform → log-mel front-end \
             (n_mels = 128 for kotoba-whisper v2.0) → Whisper large-v3 encoder \
             (32 layers) → kotoba-shrunk decoder (2 layers) → byte-level BPE \
             detokenize (vocab_size = 51 866) forward path has not landed yet. \
             Follow-up wave: delegate to crate::whisper::WhisperModel with the \
             kotoba-shrunk n_text_layer — the op set (STFT / mel filterbank / \
             GEMM / GEMV / softmax / layer-norm / GELU / conv1d) and every kernel \
             are already shared with vanilla Whisper. The JA-ASR-2 axis \
             (data-driven decoder depth) is already honored by the shared \
             WhisperConfig loader — this scaffold rides on top of it.",
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Every hparam matches `huggingface.co/kotoba-tech/kotoba-whisper-v2.0/
    /// raw/main/config.json` (fetched 2026-07-24).
    #[test]
    fn kotoba_whisper_v2_0_matches_primary_source_config_json() {
        let c = KotobaWhisperConfig::kotoba_whisper_v2_0();
        // Encoder — identical to whisper-large-v3.
        assert_eq!(c.d_model, 1280);
        assert_eq!(c.n_audio_layer, 32);
        assert_eq!(c.n_audio_head, 20);
        assert_eq!(c.ffn_dim, 5120);
        assert_eq!(c.n_mels, 128);
        assert_eq!(c.n_audio_ctx, 1500);
        // Decoder — the JA-ASR-2 axis.
        assert_eq!(
            c.n_text_layer, 2,
            "kotoba-whisper v2.0 shrinks decoder to 2 layers"
        );
        assert_eq!(c.n_text_head, 20);
        assert_eq!(c.n_text_ctx, 448);
        // Tokenizer — large-v3 multilingual (+1 vocab for <|yue|>).
        assert_eq!(c.n_vocab, 51_866);
        assert_eq!(c.eot, 50_257);
        assert_eq!(c.sot, 50_258);
        // Audio boundary.
        assert_eq!(c.sample_rate, 16_000);
        // Derived — head_dim is the Whisper invariant.
        assert_eq!(c.head_dim(), 64, "kotoba-whisper head_dim = 1280/20 = 64");
        // Distil invariant holds.
        assert!(
            c.n_text_layer < c.n_audio_layer,
            "kotoba-whisper must have decoder < encoder depth"
        );
        c.validate_for_forward()
            .expect("kotoba-whisper v2.0 is well-formed");
    }

    #[test]
    fn tiny_config_is_well_formed() {
        KotobaWhisperConfig::tiny_for_tests()
            .validate_for_forward()
            .expect("tiny config is well-formed");
    }

    /// A checkpoint whose decoder depth equals the encoder depth is
    /// **not** kotoba-whisper — it is vanilla Whisper. The validator
    /// must catch this so a mis-flattened checkpoint (decoder tensors
    /// duplicated to the encoder count) fails loudly at
    /// `KotobaWhisperAsr::new`, not silently deep in a forward.
    #[test]
    fn config_rejects_equal_encoder_decoder_depth() {
        let mut c = KotobaWhisperConfig::tiny_for_tests();
        c.n_text_layer = c.n_audio_layer;
        let err = c
            .validate_for_forward()
            .expect_err("equal depth is not kotoba");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument, got {err:?}"
        );
    }

    #[test]
    fn config_rejects_decoder_larger_than_encoder() {
        let mut c = KotobaWhisperConfig::tiny_for_tests();
        c.n_text_layer = c.n_audio_layer + 1;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_rejects_zero_axis() {
        for mutate in [
            |c: &mut KotobaWhisperConfig| c.d_model = 0,
            |c: &mut KotobaWhisperConfig| c.n_mels = 0,
            |c: &mut KotobaWhisperConfig| c.n_audio_ctx = 0,
            |c: &mut KotobaWhisperConfig| c.n_audio_layer = 0,
            |c: &mut KotobaWhisperConfig| c.n_text_ctx = 0,
            |c: &mut KotobaWhisperConfig| c.n_text_layer = 0,
            |c: &mut KotobaWhisperConfig| c.n_vocab = 0,
            |c: &mut KotobaWhisperConfig| c.ffn_dim = 0,
            |c: &mut KotobaWhisperConfig| c.n_audio_head = 0,
            |c: &mut KotobaWhisperConfig| c.n_text_head = 0,
            |c: &mut KotobaWhisperConfig| c.sample_rate = 0,
        ] {
            let mut c = KotobaWhisperConfig::tiny_for_tests();
            mutate(&mut c);
            assert!(matches!(
                c.validate_for_forward(),
                Err(VokraError::InvalidArgument(_))
            ));
        }
    }

    #[test]
    fn config_rejects_head_not_dividing_d_model() {
        let mut c = KotobaWhisperConfig::tiny_for_tests();
        c.n_audio_head = 3; // 16 % 3 != 0
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_rejects_odd_head_dim() {
        let mut c = KotobaWhisperConfig::tiny_for_tests();
        // 12 / 4 = 3 (odd).
        c.d_model = 12;
        c.n_audio_head = 4;
        c.n_text_head = 4;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_rejects_eot_or_sot_outside_vocab() {
        let mut c = KotobaWhisperConfig::tiny_for_tests();
        c.eot = c.n_vocab as u32;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));

        let mut c = KotobaWhisperConfig::tiny_for_tests();
        c.sot = c.n_vocab as u32 + 10;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_accepts_well_formed_config() {
        let c = KotobaWhisperConfig::tiny_for_tests();
        let asr = KotobaWhisperAsr::new(c.clone()).expect("kotoba-whisper asr");
        assert_eq!(asr.config().d_model, c.d_model);
        assert_eq!(asr.config().n_text_layer, c.n_text_layer);
    }

    #[test]
    fn transcribe_rejects_empty_pcm() {
        let c = KotobaWhisperConfig::tiny_for_tests();
        let asr = KotobaWhisperAsr::new(c).expect("kotoba-whisper asr");
        assert!(matches!(
            asr.transcribe(&[]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// The primary NotImplemented path names the follow-up blocker
    /// (FR-EX-08 — never a silent zero-fill / hallucinated transcript).
    #[test]
    fn transcribe_is_loud_not_implemented_until_real_forward_lands() {
        let c = KotobaWhisperConfig::tiny_for_tests();
        let asr = KotobaWhisperAsr::new(c).expect("kotoba-whisper asr");
        let pcm = vec![0.0f32; 1024];
        let err = asr.transcribe(&pcm).unwrap_err();
        match err {
            VokraError::NotImplemented(msg) => {
                assert!(
                    msg.contains("kotoba-whisper"),
                    "message must name the model: {msg}"
                );
                assert!(
                    msg.contains("JA-ASR-2"),
                    "message must reference JA-ASR-2 axis: {msg}"
                );
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    #[test]
    fn expected_arch_is_kotoba_whisper() {
        assert_eq!(EXPECTED_ARCH, "kotoba-whisper");
    }

    #[test]
    fn expected_arch_is_distinct_from_siblings() {
        assert_ne!(EXPECTED_ARCH, "whisper");
        assert_ne!(EXPECTED_ARCH, crate::distil_whisper::EXPECTED_ARCH);
    }

    #[test]
    fn sample_rate_matches_whisper_convention() {
        assert_eq!(KOTOBA_WHISPER_SAMPLE_RATE, 16_000);
    }

    /// The M2-13 compliance registry must resolve every canonical
    /// kotoba-whisper id to Permissive (Apache-2.0). Cross-crate test
    /// to keep this module's registry-side contract honest.
    ///
    /// Scout A-5 follow-up (2026-07-29): `kotoba-whisper-v2.2` is the
    /// slug the parity-CI workflow (`parity-whisper-extras-real.yml`)
    /// pins today via `env.KOTOBA_WHISPER_REPO`. It resolves Permissive
    /// transitively via the `kotoba-whisper-` prefix walk in
    /// `vokra_core::compliance::license_class`, but pinning it here
    /// makes the workflow-pinned literal a machine-checked invariant so
    /// a future prefix-walk removal surfaces red on any `cargo test`
    /// rather than only during a paid HF-download workflow run.
    #[test]
    fn registry_lookup_maps_kotoba_whisper_to_permissive_apache_2_0() {
        use vokra_core::compliance::{LicenseClass, registry_lookup};
        for id in [
            "kotoba-whisper",
            "kotoba-whisper-v1.0",
            "kotoba-whisper-v1.1",
            "kotoba-whisper-v2.0",
            "kotoba-whisper-v2.1",
            // v2.2 = workflow-pinned literal (see rustdoc above).
            // Prefix walk covers it today; the pin binds the contract.
            "kotoba-whisper-v2.2",
            "kotoba-whisper-bilingual",
            "kotoba-whisper-bilingual-v1.0",
        ] {
            assert_eq!(
                registry_lookup(id),
                Some(LicenseClass::Permissive),
                "registry must map `{id}` to Permissive (Apache-2.0)"
            );
        }
    }
}
