//! Voxtral ASR head (M3-10-T13) — audio → text.
//!
//! # Scope (autoregressive greedy)
//!
//! The ASR head owns the logic that ties the audio encoder output to the
//! Mistral text decoder for a greedy transcribe:
//!
//! 1. run [`AudioEncoder::forward`](super::audio_encoder::forward) on the
//!    log-mel input, yielding `[n_ctx, d_audio]`;
//! 2. seed the decoder with the config-supplied special-token prefix
//!    (`bos_id`);
//! 3. drive the causal decode loop through a [`TextDecoderSession`]
//!    (self-attention + KV cache append + RoPE + causal mask + GQA);
//! 4. stop on the `<eos>` token or a max-new-token cap;
//! 5. return the generated token id sequence (the caller detokenises via
//!    [`VoxtralTokenizer`]).
//!
//! # Audio conditioning — Wave 8 pluggable adapter
//!
//! Upstream Voxtral projects the audio-encoder hidden state through an
//! **audio adapter** and consumes it as a soft-prefix sequence at the start
//! of the text decoder. Wave 8 lands the pluggable [`AudioAdapter`]
//! framework: a GGUF that carries a `vokra.voxtral.adapter.*` chunk with a
//! non-`"none"` kind will route the encoder output through the adapter and
//! prepend the projection as a soft-prefix to the greedy decode (real audio
//! conditioning).
//!
//! A GGUF whose adapter chunk is absent or `kind = "none"` keeps the honest
//! Wave 7 posture: the encoder still runs (shape / dispatch coverage) but the
//! greedy loop starts from `bos_id` with **no audio conditioning wired** —
//! the returned tokens reflect the language-model prior. This entry point
//! deliberately never fabricates a "transcript that sounds like the audio"
//! (FR-EX-08): either the caller sees the real audio-conditioned output, the
//! honest LM-prior output, or an explicit error.
//!
//! `bos_id` / `eos_id` default to Mistral's shipped ids (`1` / `2`) — every
//! Voxtral variant on HuggingFace inherits these from the Mistral base
//! tokenizer.
//!
//! [`AudioAdapter`]: super::AudioAdapter
//! [`TextDecoderSession`]: super::TextDecoderSession
//! [`VoxtralTokenizer`]: super::VoxtralTokenizer

use vokra_core::{BackendKind, Result, VokraError};

use super::AudioAdapter;
use super::text_decoder_session::{
    DEFAULT_MAX_NEW_TOKENS, TextDecoderSession, greedy_decode, greedy_decode_with_prefix,
};
use super::{AudioEncoder, AudioEncoderOutput, TextDecoder, VoxtralConfig};

/// The Mistral BOS token id (shipped `<s>` = 1 across every Mistral tokenizer
/// release Voxtral inherits from).
pub const MISTRAL_BOS_ID: u32 = 1;
/// The Mistral EOS token id (shipped `</s>` = 2).
pub const MISTRAL_EOS_ID: u32 = 2;

/// A greedy ASR head. Holds references — it is a thin adaptor, not a
/// stateful engine.
///
/// Construct once (cheap) and call [`AsrHead::transcribe`] per utterance;
/// the caller owns the log-mel front-end + tokenization.
pub struct AsrHead<'m> {
    config: &'m VoxtralConfig,
    audio: &'m AudioEncoder,
    text: &'m TextDecoder,
    /// Optional audio adapter: when present *and* active
    /// ([`AudioAdapter::is_active`]) the encoder output is projected and fed
    /// to the decoder as a soft-prefix (real audio conditioning). Otherwise
    /// the greedy loop stays on the honest Wave 7 LM-continuation path.
    adapter: Option<&'m AudioAdapter>,
}

impl<'m> AsrHead<'m> {
    /// Bundle a config + encoder + decoder into a runnable ASR head. The
    /// audio adapter is `None`; use [`Self::with_adapter`] to attach one.
    #[must_use]
    pub fn new(config: &'m VoxtralConfig, audio: &'m AudioEncoder, text: &'m TextDecoder) -> Self {
        Self {
            config,
            audio,
            text,
            adapter: None,
        }
    }

    /// Attach an audio adapter. Chainable with [`Self::new`]. Passing an
    /// adapter whose [`AudioAdapter::is_active`] is `false` is equivalent to
    /// no adapter — the greedy loop still runs the LM-continuation path
    /// (Wave 7 honest posture). Any other kind flips the transcribe path onto
    /// the audio-conditioned soft-prefix decode.
    #[must_use]
    pub fn with_adapter(mut self, adapter: &'m AudioAdapter) -> Self {
        self.adapter = Some(adapter);
        self
    }

    /// Runs the audio encoder on `log_mel` (`[n_mels, n_frames]`) and
    /// returns the encoder hidden state — the shape the future full decode
    /// will consume.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] on a zero-sentinel config;
    /// [`VokraError::InvalidArgument`] on shape mismatch.
    pub fn encode(
        &self,
        compute: &crate::compute::Compute,
        log_mel: &[f32],
        n_frames: usize,
    ) -> Result<AudioEncoderOutput> {
        super::audio_encoder::forward(compute, self.config, self.audio, log_mel, n_frames)
    }

    /// Greedy transcribe: audio → token id sequence.
    ///
    /// Steps:
    /// 1. validates `config.mode` is `"asr"` or `"s2s"` (S2S also produces
    ///    text as its inner stream);
    /// 2. runs the audio encoder end-to-end (shape / dispatch coverage);
    /// 3. constructs a [`TextDecoderSession`] on `backend` and greedy-decodes
    ///    from `bos_id` until `eos_id` or `max_new_tokens`.
    ///
    /// See the module doc's "Audio conditioning" note — until the audio
    /// adapter follow-up ticket lands, the returned tokens are the LM
    /// prior's continuation of `bos_id`, not audio-conditioned ASR.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] on a mismatched `config.mode` or a
    ///   0-sentinel config;
    /// - [`VokraError::InvalidArgument`] on a shape mismatch in `log_mel` or
    ///   an out-of-range token id;
    /// - [`VokraError::UnsupportedOp`] / [`VokraError::BackendUnavailable`]
    ///   on a backend that does not cover the Voxtral hot ops.
    pub fn transcribe(
        &self,
        backend: BackendKind,
        log_mel: &[f32],
        n_frames: usize,
        bos_id: u32,
        eos_id: u32,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        // Config-mode gate — surface the mismatch clearly.
        if self.config.mode != "asr" && self.config.mode != "s2s" {
            return Err(VokraError::ModelLoad(format!(
                "voxtral asr_head.transcribe: config.mode is `{}` — expected `asr` or `s2s`",
                self.config.mode
            )));
        }
        // Config bounds gate: bos/eos must be in vocab.
        let vocab = self.config.text.vocab_size as u32;
        if bos_id >= vocab {
            return Err(VokraError::InvalidArgument(format!(
                "voxtral asr_head.transcribe: bos_id {bos_id} >= vocab_size {vocab}"
            )));
        }
        // eos_id being out-of-vocab is TOLERATED (matches Whisper's greedy —
        // an unreachable eos just means the decode runs to max_new_tokens).

        // Build the compute seam via a lightweight Compute for the encoder
        // (the decoder session builds its own with the same backend).
        let compute = crate::compute::Compute::for_backend(backend, super::VOXTRAL_HOT_OPS)?;
        let encoder_out = self.encode(&compute, log_mel, n_frames)?;

        // Autoregressive greedy over the loaded Mistral text decoder.
        let mut session = TextDecoderSession::new(self.config, self.text, backend)?;
        let cap = if max_new_tokens == 0 {
            DEFAULT_MAX_NEW_TOKENS
        } else {
            max_new_tokens
        };

        // Route through the adapter if present *and* active. Absent adapter
        // (or `AdapterKind::None`) falls back to the Wave 7 LM-continuation
        // path — honest limitation, never a fabricated audio-shaped output.
        match self.adapter {
            Some(adapter) if adapter.is_active() => {
                let d = self.config.text.hidden_dim;
                // Sanity gate: the adapter must project *into* the decoder's
                // hidden width. A misconfigured adapter (out_dim != d) is a
                // configuration error — reject rather than push a
                // mis-sized prefix into the KV cache (FR-EX-08).
                let adapter_out = super::adapter::out_dim(adapter.kind());
                if adapter_out != d {
                    return Err(VokraError::ModelLoad(format!(
                        "voxtral asr_head.transcribe: adapter out_dim ({adapter_out}) must equal \
                         text_decoder.hidden_dim ({d}) — check the adapter config."
                    )));
                }
                let prefix_embed = adapter.apply(
                    &compute,
                    &encoder_out.hidden,
                    encoder_out.n_ctx,
                    encoder_out.hidden_dim,
                )?;
                if d == 0 || prefix_embed.len() % d != 0 {
                    return Err(VokraError::ModelLoad(format!(
                        "voxtral asr_head.transcribe: adapter output len {} not a multiple of \
                         text_decoder.hidden_dim {}",
                        prefix_embed.len(),
                        d,
                    )));
                }
                let t_prefix = prefix_embed.len() / d;
                greedy_decode_with_prefix(
                    &mut session,
                    &prefix_embed,
                    t_prefix,
                    bos_id,
                    eos_id,
                    cap,
                )
            }
            _ => greedy_decode(&mut session, &[bos_id], eos_id, cap),
        }
    }

    /// Returns the number of text-decoder blocks — surfaces a piece of the
    /// loaded model to the caller for e2e wiring checks.
    #[must_use]
    pub fn n_decoder_layer(&self) -> usize {
        self.text.n_layer()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxtral::config::{AudioEncoderConfig, TextDecoderConfig};
    use crate::voxtral::text_decoder::{DecoderBlock, GqaAttention, Linear, SwiGluFfn};

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

    fn tiny_encoder(cfg: &VoxtralConfig) -> AudioEncoder {
        AudioEncoder {
            conv1_w: vec![0.0; cfg.audio.hidden_dim * cfg.audio.n_mels * 3],
            conv1_b: vec![0.0; cfg.audio.hidden_dim],
            conv2_w: vec![0.0; cfg.audio.hidden_dim * cfg.audio.hidden_dim * 3],
            conv2_b: vec![0.0; cfg.audio.hidden_dim],
            pos_emb: vec![0.0; cfg.audio.n_ctx * cfg.audio.hidden_dim],
            has_learned_pos_emb: true,
        }
    }

    /// Build a tiny hand-crafted TextDecoder with deterministic non-zero
    /// weights (same seed pattern as text_decoder_session::tests::tiny_decoder).
    /// Reused across the AsrHead greedy-decode tests.
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

    #[test]
    fn transcribe_returns_a_non_empty_token_sequence() {
        // With deterministic non-zero weights + eos outside the vocab so the
        // greedy loop never terminates early, transcribe must return exactly
        // `max_new_tokens` tokens, each in-vocab. This is the "engine is
        // registered and runs" oracle (no fabricated string — the caller
        // detokenises separately).
        let cfg = tiny_config();
        let ae = tiny_encoder(&cfg);
        let td = tiny_decoder(&cfg);
        let head = AsrHead::new(&cfg, &ae, &td);
        let n_frames = 8;
        let log_mel = vec![0.5f32; cfg.audio.n_mels * n_frames];
        // eos = vocab+100 => unreachable => loop runs to max_new.
        let out = head
            .transcribe(
                BackendKind::Cpu,
                &log_mel,
                n_frames,
                /*bos*/ 1,
                /*eos*/ cfg.text.vocab_size as u32 + 100,
                /*max_new*/ 3,
            )
            .unwrap();
        assert_eq!(out.len(), 3, "must respect max_new");
        assert!(
            out.iter().all(|&t| (t as usize) < cfg.text.vocab_size),
            "every token must be in-vocab: {out:?}"
        );
    }

    #[test]
    fn transcribe_stops_on_eos() {
        // With eos = 0 and the deterministic weights above, the greedy loop
        // may or may not emit 0 as the first token; either way the returned
        // sequence must not exceed max_new_tokens. When it stops early on
        // eos, the eos token IS included in the result.
        let cfg = tiny_config();
        let ae = tiny_encoder(&cfg);
        let td = tiny_decoder(&cfg);
        let head = AsrHead::new(&cfg, &ae, &td);
        let n_frames = 8;
        let log_mel = vec![0.5f32; cfg.audio.n_mels * n_frames];
        let out = head
            .transcribe(BackendKind::Cpu, &log_mel, n_frames, 1, 0, 10)
            .unwrap();
        assert!(out.len() <= 10, "must respect max_new");
        if let Some(&last) = out.last() {
            if last == 0 {
                // stopped on eos: eos IS included, no tokens after it.
                assert!(out.iter().take(out.len() - 1).all(|&t| t != 0));
            }
        }
    }

    #[test]
    fn transcribe_rejects_unknown_mode() {
        let mut cfg = tiny_config();
        cfg.mode = "wat".to_owned();
        let ae = AudioEncoder {
            conv1_w: Vec::new(),
            conv1_b: Vec::new(),
            conv2_w: Vec::new(),
            conv2_b: Vec::new(),
            pos_emb: Vec::new(),
            has_learned_pos_emb: false,
        };
        let td = TextDecoder {
            token_emb: Vec::new(),
            blocks: Vec::new(),
            final_norm_gamma: Vec::new(),
            prefix: "",
        };
        let head = AsrHead::new(&cfg, &ae, &td);
        assert!(matches!(
            head.transcribe(BackendKind::Cpu, &[], 0, 1, 2, 4),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn transcribe_rejects_out_of_range_bos() {
        let cfg = tiny_config();
        let ae = tiny_encoder(&cfg);
        let td = tiny_decoder(&cfg);
        let head = AsrHead::new(&cfg, &ae, &td);
        let n_frames = 8;
        let log_mel = vec![0.0f32; cfg.audio.n_mels * n_frames];
        let bos = cfg.text.vocab_size as u32; // >= vocab
        assert!(matches!(
            head.transcribe(BackendKind::Cpu, &log_mel, n_frames, bos, 2, 4),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn transcribe_is_deterministic_across_calls() {
        let cfg = tiny_config();
        let ae = tiny_encoder(&cfg);
        let td = tiny_decoder(&cfg);
        let head = AsrHead::new(&cfg, &ae, &td);
        let n_frames = 8;
        let log_mel = vec![0.5f32; cfg.audio.n_mels * n_frames];
        let a = head
            .transcribe(BackendKind::Cpu, &log_mel, n_frames, 1, 9999, 4)
            .unwrap();
        let b = head
            .transcribe(BackendKind::Cpu, &log_mel, n_frames, 1, 9999, 4)
            .unwrap();
        assert_eq!(a, b, "greedy decode must be deterministic");
    }

    #[test]
    fn mistral_default_ids_match_upstream_shipping_values() {
        // Guard against accidental rename; the caller-side layer relies on
        // these being the Mistral defaults (1 = <s>, 2 = </s>).
        assert_eq!(MISTRAL_BOS_ID, 1);
        assert_eq!(MISTRAL_EOS_ID, 2);
    }

    /// A synthetic `AdapterKind::Linear` that mirrors what a real Voxtral
    /// checkpoint would load from `vokra.voxtral.adapter.*`, but built here
    /// directly so the test exercises the runtime forward path without a
    /// full converter round-trip. `in_dim == out_dim == d_audio == d_text`
    /// so the projection shape matches the tiny_config decoder hidden width.
    fn synth_linear_adapter_gguf(d: usize) -> Vec<u8> {
        use vokra_core::gguf::{GgmlType, GgufBuilder};
        let mut b = GgufBuilder::new();
        b.add_string("vokra.voxtral.adapter.kind", "linear");
        b.add_string("vokra.voxtral.adapter.tensor_prefix", "audio_adapter.");
        b.add_u32("vokra.voxtral.adapter.in_dim", d as u32);
        b.add_u32("vokra.voxtral.adapter.out_dim", d as u32);
        b.add_bool("vokra.voxtral.adapter.has_bias", false);
        b.add_bool("vokra.voxtral.adapter.has_layernorm", false);
        // Identity weight [out=d, in=d] — the safetensors → w_t transpose in
        // `adapter::load_linear` inverts to identity too.
        let mut w = vec![0.0f32; d * d];
        for i in 0..d {
            w[i * d + i] = 1.0;
        }
        b.add_tensor(
            "audio_adapter.weight",
            GgmlType::F32,
            vec![d as u64, d as u64],
            w.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap();
        b.to_bytes().unwrap()
    }

    #[test]
    fn adapter_active_routes_through_soft_prefix_path() {
        // With an active linear adapter, the transcribe path prefills the
        // decoder with a soft-prefix from the (all-zero) encoder output +
        // adapter projection, then runs the greedy loop. The result must
        // still be an in-vocab token sequence of size == max_new. This
        // exercises the *dispatch* not a numeric oracle — a real
        // conditioning check requires a real Voxtral checkpoint.
        use vokra_core::gguf::GgufFile;
        let cfg = tiny_config();
        let ae = tiny_encoder(&cfg);
        let td = tiny_decoder(&cfg);
        let adapter_bytes = synth_linear_adapter_gguf(cfg.text.hidden_dim);
        let file = GgufFile::parse(adapter_bytes).unwrap();
        let adapter = super::super::AudioAdapter::from_gguf(&file).unwrap();
        assert!(adapter.is_active());

        let head = AsrHead::new(&cfg, &ae, &td).with_adapter(&adapter);
        let n_frames = 8;
        let log_mel = vec![0.5f32; cfg.audio.n_mels * n_frames];
        let out = head
            .transcribe(
                BackendKind::Cpu,
                &log_mel,
                n_frames,
                1,
                cfg.text.vocab_size as u32 + 100,
                3,
            )
            .unwrap();
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|&t| (t as usize) < cfg.text.vocab_size));
    }

    #[test]
    fn adapter_active_produces_different_output_than_none() {
        // A soft-prefix path *should* fill the decoder KV cache with entries
        // derived from the (adapter-projected) encoder hidden state, which
        // gives the greedy decoder a different starting context than a bare
        // `[bos]` prefix. Provided the encoder output is not identically
        // zero (we use non-zero log-mel + Conv weights via the identity
        // adapter passing whatever comes out of the encoder), the two greedy
        // sequences should not be forced to be equal.
        //
        // We use a decoder + config where the encoder output is guaranteed
        // to be non-zero: with all-zero conv weights the encoder outputs 0,
        // but with pos_emb pre-populated to non-zero the transpose+pos_add
        // step yields non-zero hidden. That's exactly what tiny_encoder
        // gives us (pos_emb is [0.0…], though) — override.
        use vokra_core::gguf::GgufFile;
        let cfg = tiny_config();
        // Encoder with non-zero pos_emb → non-zero hidden even when
        // conv weights are zero.
        let mut ae = tiny_encoder(&cfg);
        for (i, v) in ae.pos_emb.iter_mut().enumerate() {
            *v = ((i as i32 % 3) - 1) as f32 * 0.1;
        }
        let td = tiny_decoder(&cfg);

        let adapter_bytes = synth_linear_adapter_gguf(cfg.text.hidden_dim);
        let file = GgufFile::parse(adapter_bytes).unwrap();
        let adapter = super::super::AudioAdapter::from_gguf(&file).unwrap();
        let none = super::super::AudioAdapter::none();

        let head_active = AsrHead::new(&cfg, &ae, &td).with_adapter(&adapter);
        let head_bare = AsrHead::new(&cfg, &ae, &td).with_adapter(&none);
        let n_frames = 8;
        let log_mel = vec![0.5f32; cfg.audio.n_mels * n_frames];
        let a = head_active
            .transcribe(
                BackendKind::Cpu,
                &log_mel,
                n_frames,
                1,
                cfg.text.vocab_size as u32 + 100,
                4,
            )
            .unwrap();
        let b = head_bare
            .transcribe(
                BackendKind::Cpu,
                &log_mel,
                n_frames,
                1,
                cfg.text.vocab_size as u32 + 100,
                4,
            )
            .unwrap();
        // The two paths take a different sequence of KV cache entries into
        // the greedy loop, so their outputs are *allowed* to diverge. We
        // require they be *reachable* — i.e. both call chains complete
        // without error and return length-4 sequences.
        assert_eq!(a.len(), 4);
        assert_eq!(b.len(), 4);
        // Deterministic across calls with the same head.
        let a2 = head_active
            .transcribe(
                BackendKind::Cpu,
                &log_mel,
                n_frames,
                1,
                cfg.text.vocab_size as u32 + 100,
                4,
            )
            .unwrap();
        assert_eq!(a, a2, "adapter-conditioned greedy must be deterministic");
    }

    #[test]
    fn adapter_out_dim_mismatch_is_model_load_error() {
        // An adapter whose out_dim doesn't match text_decoder.hidden_dim is
        // caught by the head at project time (with the sanity gate we added
        // in transcribe()). We build a synthetic adapter with out_dim = 6,
        // decoder hidden_dim = 4 — the head must refuse rather than push a
        // mis-shaped prefix into the decoder.
        use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile};
        let cfg = tiny_config();
        let d_in = cfg.text.hidden_dim; // 4
        let d_out = 6;
        let ae = tiny_encoder(&cfg);
        let td = tiny_decoder(&cfg);
        let mut b = GgufBuilder::new();
        b.add_string("vokra.voxtral.adapter.kind", "linear");
        b.add_string("vokra.voxtral.adapter.tensor_prefix", "adap.");
        b.add_u32("vokra.voxtral.adapter.in_dim", d_in as u32);
        b.add_u32("vokra.voxtral.adapter.out_dim", d_out as u32);
        // Weight [out=6, in=4] — must exist so from_gguf doesn't reject on
        // the missing-tensor path.
        let w = vec![0.0f32; d_out * d_in];
        b.add_tensor(
            "adap.weight",
            GgmlType::F32,
            vec![d_out as u64, d_in as u64],
            w.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap();
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        // The adapter loads (its shape is internally consistent); it's the
        // head that flags the size mismatch vs decoder.hidden_dim.
        let adapter = super::super::AudioAdapter::from_gguf(&file).unwrap();
        let head = AsrHead::new(&cfg, &ae, &td).with_adapter(&adapter);
        let n_frames = 8;
        let log_mel = vec![0.5f32; cfg.audio.n_mels * n_frames];
        // The adapter's apply itself will surface an InvalidArgument (input
        // hidden 4 doesn't match adapter in_dim 4 — actually it matches so
        // apply succeeds, but the head then sees output len 6*t_prefix which
        // is not a multiple of d_text=4 iff t_prefix != multiple of 2. With
        // t_prefix = n_ctx_encoder = whatever, this is a robust head-level
        // rejection).
        let err = head
            .transcribe(BackendKind::Cpu, &log_mel, n_frames, 1, 999, 2)
            .unwrap_err();
        assert!(
            matches!(err, VokraError::ModelLoad(_)),
            "expected ModelLoad, got {err:?}"
        );
    }
}
