//! Voxtral `AsrEngine` — [`vokra_core::AsrEngine`] adaptor over a loaded
//! [`VoxtralModel`].
//!
//! # Purpose
//!
//! The vokra-server (`integrations/vokra-server`) treats every ASR model
//! uniformly through the [`vokra_core::AsrEngine`] trait. This file supplies
//! the Voxtral side of that trait — a thin adaptor that owns a
//! [`VoxtralModel`] plus a runtime [`BackendKind`] and dispatches
//! [`AsrEngine::transcribe`](vokra_core::AsrEngine::transcribe) through the
//! [`AsrHead`] + [`VoxtralTokenizer`].
//!
//! # Autoregressive greedy decode (M3-10 core)
//!
//! [`AsrHead::transcribe`] runs:
//! 1. Whisper-shape log-mel front-end (`n_mels=128`, matches the declared
//!    `vokra.frontend.*` chunk) — shared with the Whisper front-end because
//!    the specs match bit-for-bit;
//! 2. audio encoder forward (shape / dispatch coverage);
//! 3. Mistral text decoder greedy loop (KV cache, RoPE, GQA, SwiGLU,
//!    RMSNorm, tied logits, EOS stop, `max_new_tokens` cap).
//!
//! # Audio conditioning — Wave 8 pluggable adapter (see [`AsrHead`] docs)
//!
//! Wave 8 lands the pluggable [`AudioAdapter`] framework: a GGUF that carries
//! a `vokra.voxtral.adapter.*` chunk with a non-`"none"` kind routes the
//! encoder output through the adapter (linear / MLP / downsample-linear) and
//! feeds the projection as a soft-prefix to the greedy decode — this is real
//! audio-conditioned ASR.
//!
//! A GGUF whose adapter chunk is absent or declared `kind = "none"` keeps
//! the Wave 7 posture: the returned tokens reflect the language-model prior
//! of the greedy decode from `bos_id`, not audio-conditioned ASR. This is
//! intentional per FR-EX-08 — callers see either a real (audio-conditioned)
//! token sequence, an honest (LM-prior) token sequence, or an explicit error;
//! never a fabricated audio-shaped transcript. Real ASR accuracy against a
//! Voxtral checkpoint requires (a) the adapter tensors + hparams from the
//! upstream release passed via `convert_voxtral_file`'s `--adapter-config`
//! side-car, and (b) a real-checkpoint parity dump (T19+).
//!
//! [`AudioAdapter`]: super::AudioAdapter
//!
//! # Front-end
//!
//! The 16 kHz mono `f32` PCM is turned into log-mel through
//! [`crate::whisper::mel::log_mel`] with `n_mels=128` (the Voxtral spec).
//! The same helper Whisper uses — no second implementation.
//!
//! [`VoxtralTokenizer`]: super::VoxtralTokenizer

use std::sync::Arc;

use vokra_core::{AsrEngine, BackendKind, Result, Transcription, VokraError};

use super::asr_head::{MISTRAL_BOS_ID, MISTRAL_EOS_ID};
use super::{AsrHead, VoxtralModel, VoxtralTokenizer};

/// A Voxtral engine that speaks the [`AsrEngine`] trait. Holds the loaded
/// [`VoxtralModel`] plus its embedded tokenizer, the runtime backend and
/// max-new-token cap.
///
/// Cloned freely on the hot path (the model / tokenizer are behind an
/// [`Arc`]).
pub struct VoxtralAsr {
    /// The parsed model, shared. `Arc` because the registry holds one and
    /// hot-path handlers borrow it read-only.
    model: Arc<VoxtralModel>,
    /// The embedded Mistral tokenizer, shared. Loaded from the GGUF's
    /// `vokra.tokenizer.model` chunk at construction. Optional because a
    /// GGUF converted without the tokenizer chunk (older paths, or a
    /// tokenizer-less test double) still parses; a `None` tokenizer
    /// surfaces at [`transcribe`] time as an explicit error.
    tokenizer: Option<Arc<VoxtralTokenizer>>,
    /// Whether the model was declared as ASR- or S2S-capable in its config.
    /// ASR mode is the default; an S2S-tagged model can still be routed
    /// through this adaptor (S2S produces text on the inner stream) but the
    /// caller sees an ASR interface.
    #[allow(dead_code)]
    is_configured_for_asr: bool,
    /// Runtime backend selector for the encoder + decoder session.
    backend: BackendKind,
    /// Upper bound on generated tokens per transcribe call. `0` means
    /// [`super::text_decoder_session::DEFAULT_MAX_NEW_TOKENS`].
    max_new_tokens: usize,
    /// BOS token id (Mistral's `<s>` = 1 unless overridden).
    bos_id: u32,
    /// EOS token id (Mistral's `</s>` = 2 unless overridden).
    eos_id: u32,
}

impl VoxtralAsr {
    /// Wraps a loaded [`VoxtralModel`] as an [`AsrEngine`] with the CPU
    /// backend and default (Mistral shipping) BOS/EOS ids.
    ///
    /// A model whose declared `mode` is not `"asr"` or `"s2s"` is rejected
    /// with an explicit [`VokraError::ModelLoad`] — never silently coerced
    /// (FR-EX-08).
    pub fn new(model: VoxtralModel) -> Result<Self> {
        Self::new_with_backend(model, BackendKind::Cpu)
    }

    /// Like [`Self::new`] but on an explicit backend. The backend is
    /// consulted at each [`transcribe`] call so a runtime toggle can
    /// switch between CPU and a GPU seam without rebuilding the adaptor.
    pub fn new_with_backend(model: VoxtralModel, backend: BackendKind) -> Result<Self> {
        let is_asr = matches!(model.config().mode.as_str(), "asr" | "s2s");
        if !is_asr {
            return Err(VokraError::ModelLoad(format!(
                "voxtral::VoxtralAsr: unknown mode `{}` — expected `asr` or `s2s`",
                model.config().mode
            )));
        }
        Ok(Self {
            model: Arc::new(model),
            tokenizer: None,
            is_configured_for_asr: is_asr,
            backend,
            max_new_tokens: 0, // 0 => DEFAULT_MAX_NEW_TOKENS
            bos_id: MISTRAL_BOS_ID,
            eos_id: MISTRAL_EOS_ID,
        })
    }

    /// Loads a Voxtral model from a GGUF file and wraps it as an
    /// [`AsrEngine`], also loading the embedded tokenizer if present. A
    /// missing tokenizer chunk is NOT a hard error at construction (some
    /// converter paths write shape-only GGUFs) — it surfaces at
    /// [`transcribe`] time as an explicit [`VokraError::ModelLoad`] naming
    /// the missing chunk. Same posture as other model surfaces here (never
    /// a silent fabrication).
    pub fn from_gguf(file: &vokra_core::gguf::GgufFile) -> Result<Self> {
        let model = VoxtralModel::from_gguf(file)?;
        let mut asr = Self::new(model)?;
        // Tokenizer load is optional at construction (see docstring).
        if let Ok(tok) = VoxtralTokenizer::from_gguf(file, MISTRAL_EOS_ID) {
            asr.tokenizer = Some(Arc::new(tok));
        }
        Ok(asr)
    }

    /// Shared handle to the underlying model.
    #[must_use]
    pub fn model(&self) -> &Arc<VoxtralModel> {
        &self.model
    }

    /// Shared handle to the loaded tokenizer (`None` if the GGUF did not
    /// embed one).
    #[must_use]
    pub fn tokenizer(&self) -> Option<&Arc<VoxtralTokenizer>> {
        self.tokenizer.as_ref()
    }

    /// Attaches (or replaces) an externally-loaded tokenizer. Used by tests
    /// and by callers that resolve the tokenizer bytes out-of-band.
    pub fn with_tokenizer(mut self, tokenizer: VoxtralTokenizer) -> Self {
        self.tokenizer = Some(Arc::new(tokenizer));
        self
    }

    /// Overrides the max-new-token cap for greedy decode. `0` restores the
    /// default ([`super::text_decoder_session::DEFAULT_MAX_NEW_TOKENS`]).
    #[must_use]
    pub fn with_max_new_tokens(mut self, max_new: usize) -> Self {
        self.max_new_tokens = max_new;
        self
    }

    /// Overrides the greedy BOS/EOS token ids. Defaults to Mistral's
    /// shipping ids (1 / 2).
    #[must_use]
    pub fn with_bos_eos(mut self, bos_id: u32, eos_id: u32) -> Self {
        self.bos_id = bos_id;
        self.eos_id = eos_id;
        self
    }
}

impl AsrEngine for VoxtralAsr {
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "voxtral::VoxtralAsr::transcribe: pcm slice is empty".into(),
            ));
        }
        let cfg = self.model.config();
        let n_mels = cfg.audio.n_mels;
        if n_mels == 0 {
            return Err(VokraError::ModelLoad(
                "voxtral::VoxtralAsr::transcribe: config carries n_mels = 0 (shape-only path). \
                 Re-convert with a full VoxtralConfig (FR-EX-08 — no silent default)."
                    .into(),
            ));
        }
        // 1) Log-mel front-end: PCM (16 kHz mono) → [n_mels, N_FRAMES] with
        //    Whisper's fixed spec (Voxtral inherits the same n_fft=400,
        //    hop=160, Slaney mel — the front-end check in `from_gguf`
        //    already validated bit-for-bit equality to Voxtral's spec).
        let log_mel = crate::whisper::mel::log_mel(pcm, n_mels);
        let n_frames = crate::whisper::mel::N_FRAMES;

        // 2) Autoregressive greedy through the AsrHead (encoder + text
        //    decoder session + KV cache). When the loaded GGUF carries an
        //    active audio adapter (M3-10 Wave 8) the head routes through the
        //    soft-prefix audio-conditioning path; otherwise it stays on the
        //    honest Wave 7 LM-continuation path.
        let head = AsrHead::new(
            self.model.config(),
            self.model.audio_encoder(),
            self.model.text_decoder(),
        )
        .with_adapter(self.model.audio_adapter());
        let ids = head.transcribe(
            self.backend,
            &log_mel,
            n_frames,
            self.bos_id,
            self.eos_id,
            self.max_new_tokens,
        )?;

        // 3) Detokenise. A GGUF without an embedded tokenizer surfaces
        //    an explicit error — no fabrication.
        let text = match &self.tokenizer {
            Some(tok) => tok.decode(&ids)?,
            None => {
                return Err(VokraError::ModelLoad(
                    "voxtral::VoxtralAsr::transcribe: model has no embedded tokenizer \
                     (`vokra.tokenizer.model` chunk absent). Re-convert with tokenizer bytes \
                     in the side-car, or attach one via `with_tokenizer(...)` (FR-EX-08 — \
                     never fabricate detokenised text)."
                        .into(),
                ));
            }
        };
        Ok(Transcription::new(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxtral::config::{AudioEncoderConfig, TextDecoderConfig};
    use crate::voxtral::text_decoder::{DecoderBlock, GqaAttention, Linear, SwiGluFfn};
    use crate::voxtral::{AudioEncoder, TextDecoder, VoxtralConfig};

    /// Config large enough to run the full autoregressive decode: text
    /// hidden = 4, GQA 2/1, n_ctx = 16 so `bos + max_new = 8` tokens fit.
    fn tiny_config() -> VoxtralConfig {
        VoxtralConfig {
            audio: AudioEncoderConfig {
                n_layer: 1,
                n_head: 2,
                hidden_dim: 4,
                n_ctx: 8,
                n_mels: 2,
                ffn_dim: 8,
            },
            text: TextDecoderConfig {
                n_layer: 1,
                n_head_q: 2,
                n_head_kv: 1,
                hidden_dim: 4,
                ffn_dim: 8,
                vocab_size: 8,
                n_ctx: 16,
                rope_base: 10_000.0,
                rms_norm_eps: 1e-5,
            },
            cross_attn_hidden_dim: 4,
            mode: "asr".to_owned(),
            s2s_codec_type: "none".to_owned(),
        }
    }

    /// Deterministic-weight TextDecoder shared with the other Voxtral test
    /// modules — the same seed pattern.
    fn tiny_decoder(cfg: &VoxtralConfig) -> TextDecoder {
        let d = cfg.text.hidden_dim;
        let ffn = cfg.text.ffn_dim;
        let vocab = cfg.text.vocab_size;
        let head_dim = d / cfg.text.n_head_q;
        let kv_hidden = cfg.text.n_head_kv * head_dim;
        let mut token_emb = vec![0.0f32; vocab * d];
        for (i, v) in token_emb.iter_mut().enumerate() {
            *v = ((i as i32 % 7) - 3) as f32 * 0.05;
        }
        fn linear(rows: usize, cols: usize, base: f32) -> Linear {
            let mut w_t = vec![0.0f32; rows * cols];
            for (i, v) in w_t.iter_mut().enumerate() {
                *v = base + 0.01 * ((i as i32 % 5) - 2) as f32;
            }
            Linear {
                w_t,
                in_features: rows,
                out_features: cols,
            }
        }
        let blocks = (0..cfg.text.n_layer)
            .map(|_| DecoderBlock {
                attn_norm_gamma: vec![1.0f32; d],
                attn: GqaAttention {
                    q: linear(d, d, 0.10),
                    k: linear(d, kv_hidden, -0.07),
                    v: linear(d, kv_hidden, 0.05),
                    o: linear(d, d, -0.04),
                },
                ffn_norm_gamma: vec![1.0f32; d],
                ffn: SwiGluFfn {
                    gate: linear(d, ffn, 0.06),
                    up: linear(d, ffn, -0.02),
                    down: linear(ffn, d, 0.03),
                },
            })
            .collect();
        TextDecoder {
            token_emb,
            blocks,
            final_norm_gamma: vec![1.0f32; d],
            prefix: "",
        }
    }

    fn tiny_model() -> VoxtralModel {
        let cfg = tiny_config();
        let audio = AudioEncoder {
            conv1_w: vec![0.0; cfg.audio.hidden_dim * cfg.audio.n_mels * 3],
            conv1_b: vec![0.0; cfg.audio.hidden_dim],
            conv2_w: vec![0.0; cfg.audio.hidden_dim * cfg.audio.hidden_dim * 3],
            conv2_b: vec![0.0; cfg.audio.hidden_dim],
            pos_emb: vec![0.0; cfg.audio.n_ctx * cfg.audio.hidden_dim],
            has_learned_pos_emb: true,
        };
        let text = tiny_decoder(&cfg);
        VoxtralModel {
            config: cfg,
            audio,
            text,
            audio_adapter: crate::voxtral::AudioAdapter::none(),
        }
    }

    /// Empty-decoder tiny model (n_layer=0-shaped `TextDecoder`) — used to
    /// exercise the tokenizer / config-only error paths.
    fn tiny_shape_only_model() -> VoxtralModel {
        let cfg = tiny_config();
        let audio = AudioEncoder {
            conv1_w: vec![0.0; cfg.audio.hidden_dim * cfg.audio.n_mels * 3],
            conv1_b: vec![0.0; cfg.audio.hidden_dim],
            conv2_w: vec![0.0; cfg.audio.hidden_dim * cfg.audio.hidden_dim * 3],
            conv2_b: vec![0.0; cfg.audio.hidden_dim],
            pos_emb: vec![0.0; cfg.audio.n_ctx * cfg.audio.hidden_dim],
            has_learned_pos_emb: true,
        };
        let text = TextDecoder {
            token_emb: Vec::new(),
            blocks: Vec::new(),
            final_norm_gamma: Vec::new(),
            prefix: "",
        };
        VoxtralModel {
            config: cfg,
            audio,
            text,
            audio_adapter: crate::voxtral::AudioAdapter::none(),
        }
    }

    /// A minimum-viable compact-vocab tokenizer covering ids 0..vocab_size
    /// with `id -> "t{id} "` renderings, for the 200-dispatch test.
    fn tiny_tokenizer(vocab_size: usize, eos: u32) -> VoxtralTokenizer {
        // Compact-vocab dump format: u32 count + records.
        let mut blob = (vocab_size as u32).to_le_bytes().to_vec();
        for id in 0..vocab_size {
            let s = format!("t{id} ");
            let bytes = s.as_bytes();
            blob.push(0u8); // not special
            blob.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
            blob.extend_from_slice(bytes);
        }
        VoxtralTokenizer::from_bytes(blob, eos).unwrap()
    }

    #[test]
    fn new_accepts_asr_and_s2s_modes() {
        let asr = VoxtralAsr::new(tiny_model()).unwrap();
        assert!(asr.is_configured_for_asr);
    }

    #[test]
    fn new_rejects_unknown_mode() {
        let mut model = tiny_model();
        model.config.mode = "unknown".to_owned();
        assert!(matches!(
            VoxtralAsr::new(model),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn transcribe_empty_pcm_is_invalid_argument() {
        let asr = VoxtralAsr::new(tiny_model()).unwrap();
        assert!(matches!(
            asr.transcribe(&[]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn transcribe_without_tokenizer_is_model_load_error() {
        // A VoxtralAsr with no tokenizer attached must surface an explicit
        // ModelLoad on transcribe — not fabricate a text string.
        let asr = VoxtralAsr::new(tiny_model())
            .unwrap()
            .with_max_new_tokens(2)
            .with_bos_eos(1, /*unreachable*/ 999);
        let pcm = vec![0.0f32; 16_000];
        let err = asr.transcribe(&pcm).unwrap_err();
        assert!(matches!(err, VokraError::ModelLoad(_)), "{err:?}");
    }

    #[test]
    fn transcribe_with_tokenizer_returns_200_shaped_transcription() {
        // The 501 → 200 acceptance test: given a tiny model + tokenizer,
        // transcribe must return Ok(Transcription) with UTF-8 text.
        let vocab = tiny_config().text.vocab_size;
        let asr = VoxtralAsr::new(tiny_model())
            .unwrap()
            .with_max_new_tokens(3)
            .with_bos_eos(1, /*unreachable*/ vocab as u32 + 10)
            .with_tokenizer(tiny_tokenizer(vocab, 999));
        let pcm = vec![0.5f32; 16_000]; // 1 s @ 16 kHz
        let t = asr.transcribe(&pcm).expect("transcribe must return Ok");
        // With eos unreachable and max_new=3, exactly 3 tokens emitted →
        // exactly 3 "t{id} " chunks. Non-empty is the load-bearing check.
        assert!(
            !t.text.is_empty(),
            "transcription text must not be empty: {:?}",
            t.text
        );
        // Deterministic decode: a repeated call must produce the same text
        // (proves greedy + tokenizer are pure over the same input).
        let t2 = asr.transcribe(&pcm).unwrap();
        assert_eq!(t.text, t2.text);
    }

    #[test]
    fn transcribe_zero_n_mels_is_model_load_error() {
        // Shape-only converter path: n_mels=0 must not silently pass.
        let mut model = tiny_shape_only_model();
        model.config.audio.n_mels = 0;
        let asr = VoxtralAsr::new(model).unwrap();
        let pcm = vec![0.0f32; 16_000];
        assert!(matches!(
            asr.transcribe(&pcm),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn with_tokenizer_replaces_previously_attached_tokenizer() {
        let vocab = tiny_config().text.vocab_size;
        let asr = VoxtralAsr::new(tiny_model()).unwrap();
        assert!(asr.tokenizer().is_none());
        let asr = asr.with_tokenizer(tiny_tokenizer(vocab, 2));
        assert!(asr.tokenizer().is_some());
    }

    #[test]
    fn is_asr_engine_object_safe() {
        // If AsrEngine goes non-object-safe, this line stops compiling.
        // The vokra-server registry stores engines behind Arc<dyn AsrEngine>
        // so this is a load-bearing property.
        let _engine: Arc<dyn AsrEngine> = Arc::new(VoxtralAsr::new(tiny_model()).unwrap());
    }
}
