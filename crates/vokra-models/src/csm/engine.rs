//! `CsmEngine` — the [`S2sEngine`] implementation wrapping the CSM
//! pipeline (M4-05-T17): context management + frame generation (T10) +
//! audio decode chain (T15) + AEC front (T16) + Mimi encoder input path.
//!
//! # I/O contract (ADR M4-05 §D1-(b))
//!
//! CSM speaks the **caller-supplied** `reply_text` in dialog context; it
//! does not run ASR and does not generate text. An empty `reply_text` is a
//! loud [`VokraError::InvalidArgument`] (never a silent empty reply);
//! [`DialogTurn::text`] echoes the reply text verbatim.
//!
//! # Speaker conditioning note (T29-pending)
//!
//! The upstream tokenizes each segment's text with the speaker id folded
//! into the text string (generator.py `_tokenize_text_segment(text,
//! speaker)`). The exact prefix format was **not transcribed** by the T02
//! fetch; [`turn_text`] uses `"[{speaker}]{text}"` as the (提案) fixture
//! format and the T29 tokenizer hand-off pins the real one (the real
//! tokenizer's `encode` is a `NotImplemented` stub until then anyway, so
//! no real-checkpoint behaviour depends on this placeholder).
//!
//! # Loading (FR-EX-08 posture)
//!
//! [`CsmEngine::from_gguf_with_policy`] mirrors the CosyVoice2 pattern:
//! arch check → M2-13 weight-license gate → config read. Real weight
//! binding is T29-gated, so the engine binds **synthesized** weights
//! against the GGUF's shape config (the documented honest bridge —
//! `LlmBackbone::from_gguf` precedent) and the GGUF tokenizer's `encode`
//! stays `NotImplemented`; fixture flows inject
//! [`FixtureByteTokenizer`](super::tokenizer::FixtureByteTokenizer) by
//! name. A `0`-placeholder GGUF (scaffold converter) is rejected loudly at
//! load.

use std::sync::{Arc, Mutex};

use vokra_core::stream::AecRefWriter;
use vokra_core::{
    CompliancePolicy, DialogRequest, Result, S2sEngine, Sampler, SamplerConfig, SynthesizedAudio,
    VokraError, WatermarkConfig, check_weight_license,
};
use vokra_core::{DialogTurn, gguf::GgufFile};
use vokra_ops::aec::AecAttrs;
use vokra_ops::mimi_rvq::MimiRvqAttrs;

use super::aec_front::{AecFront, EchoPath, require_echo_path_wiring};
use super::audio::{CsmAudioDecodeChain, CsmAudioDecodeState};
use super::backbone::CsmFrame;
use super::config::CsmConfig;
use super::frame::{CsmFrameKind, CsmGenerationState, CsmModel};
use super::tokenizer::{CsmTextTokenizer, FixtureByteTokenizer, GgufCsmTokenizer};
use crate::mimi::{MimiEncoder, MimiNeuralConfig, MimiNeuralDecoder};

/// Upstream default generation cap: `max_audio_length_ms = 90_000`
/// (`generator.py` — ADR M4-05 §D2). Converted to frames against the
/// config's frame rate at engine construction.
pub const DEFAULT_MAX_AUDIO_MS: u64 = 90_000;

/// Upstream default sampling (`generator.py`: `temperature=0.9`,
/// `topk=50`).
pub const DEFAULT_TEMPERATURE: f32 = 0.9;
/// See [`DEFAULT_TEMPERATURE`].
pub const DEFAULT_TOP_K: usize = 50;

/// The AEC session the engine drives for interactive input
/// (mic clock + playback clock ride together).
struct AecSession {
    front: AecFront,
    writer: AecRefWriter,
    mic_pos: u64,
    play_pos: u64,
}

/// The CSM S2S engine (module docs).
pub struct CsmEngine {
    model: CsmModel,
    chain: CsmAudioDecodeChain,
    encoder: MimiEncoder,
    tokenizer: Arc<dyn CsmTextTokenizer>,
    echo_path: EchoPath,
    aec: Option<Mutex<AecSession>>,
    watermark: WatermarkConfig,
    default_max_frames: usize,
}

impl std::fmt::Debug for CsmEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsmEngine")
            .field("config", self.model.config())
            .field("echo_path", &self.echo_path)
            .field("aec_wired", &self.aec.is_some())
            .field("default_max_frames", &self.default_max_frames)
            .finish()
    }
}

impl CsmEngine {
    /// Assembles an engine from explicit components (the loaders below
    /// route here). Validates the codec seams: encoder table shape ==
    /// chain attrs, encoder `n_q` == the model's `n_codebooks`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the mismatched seam.
    pub fn new(
        model: CsmModel,
        chain: CsmAudioDecodeChain,
        encoder: MimiEncoder,
        tokenizer: Arc<dyn CsmTextTokenizer>,
    ) -> Result<Self> {
        let cfg = model.config();
        if chain.attrs().n_codebooks != cfg.n_codebooks {
            return Err(VokraError::InvalidArgument(format!(
                "csm engine: decode chain has {} codebooks but the model generates {}",
                chain.attrs().n_codebooks,
                cfg.n_codebooks
            )));
        }
        if encoder.config().quantizer.n_q != cfg.n_codebooks {
            return Err(VokraError::InvalidArgument(format!(
                "csm engine: mimi encoder quantizes {} codebooks but the model expects {}",
                encoder.config().quantizer.n_q,
                cfg.n_codebooks
            )));
        }
        if tokenizer.vocab_size() != cfg.text_vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "csm engine: tokenizer vocab {} != config text_vocab {}",
                tokenizer.vocab_size(),
                cfg.text_vocab_size
            )));
        }
        let max_ms = DEFAULT_MAX_AUDIO_MS;
        // frames = ms * frame_rate / 1000 = ms * frame_rate_mhz / 1e6.
        let default_max_frames =
            ((max_ms as u128 * cfg.frame_rate_mhz as u128) / 1_000_000) as usize;
        Ok(Self {
            model,
            chain,
            encoder,
            tokenizer,
            echo_path: EchoPath::AecRequired,
            aec: None,
            watermark: WatermarkConfig::default(),
            default_max_frames: default_max_frames.max(1),
        })
    }

    /// A fully synthesized fixture engine (tiny dims, deterministic):
    /// the numeric pipeline end to end without any real weight. The
    /// fixture Mimi config is reshaped so its quantizer matches the tiny
    /// CSM audio vocab (`n_q = n_codebooks`, `bins = audio_vocab_size`)
    /// — every generated non-special code decodes.
    ///
    /// # Errors
    ///
    /// Propagates component construction errors.
    pub fn synthesized_fixture(seed: u64) -> Result<Self> {
        let cfg = CsmConfig::tiny_for_tests();
        Self::synthesized_with_config(cfg, seed)
    }

    /// [`Self::synthesized_fixture`] with an explicit (tiny) config.
    ///
    /// # Errors
    ///
    /// Propagates component construction errors.
    pub fn synthesized_with_config(cfg: CsmConfig, seed: u64) -> Result<Self> {
        let mut mimi_cfg = MimiNeuralConfig::tiny_for_tests();
        mimi_cfg.quantizer.n_q = cfg.n_codebooks;
        mimi_cfg.quantizer.bins = cfg.audio_vocab_size;
        mimi_cfg.validate()?;
        // The fixture rides the tiny Mimi rates (16 kHz / 8-sample hop) —
        // the *relationships* (exact hop, quantizer↔model codebook match)
        // mirror the real model; the real 24 kHz / 12.5 Hz rates arrive
        // with the real config.
        let mut csm_cfg = cfg;
        csm_cfg.sample_rate = mimi_cfg.sample_rate;
        csm_cfg.frame_rate_mhz = mimi_cfg.frame_rate_mhz;

        let model = CsmModel::synthesized(csm_cfg.clone(), seed)?;
        let encoder = MimiEncoder::synthesized(&mimi_cfg, seed ^ 0x5EED_5EED)?;
        let neural = MimiNeuralDecoder::synthesized(&mimi_cfg, seed ^ 0xDEC0_DEC0, true)?;
        let attrs = MimiRvqAttrs {
            n_codebooks: mimi_cfg.quantizer.n_q,
            codebook_size: mimi_cfg.quantizer.bins,
            d_model: mimi_cfg.quantizer.dimension,
        };
        let chain = CsmAudioDecodeChain::new(encoder.tables().to_vec(), attrs, neural)?;
        let tokenizer = Arc::new(FixtureByteTokenizer::new(csm_cfg.text_vocab_size)?);
        Self::new(model, chain, encoder, tokenizer)
    }

    /// Loads a CSM GGUF from raw bytes under `policy` (M2-13 gate — a
    /// CC-BY-NC provenance without a research flag is refused).
    ///
    /// Weight posture: **synthesized bridge** until T29 (module docs);
    /// the GGUF tokenizer's `encode` is `NotImplemented` until T29, so a
    /// dialog attempt fails loudly unless a fixture tokenizer is injected
    /// via [`Self::with_tokenizer`].
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] on a wrong arch / missing tokenizer
    /// blob; compliance-gate refusals verbatim;
    /// [`VokraError::InvalidArgument`] on a `0`-placeholder shape config.
    pub fn from_gguf_with_policy(bytes: &[u8], policy: &CompliancePolicy) -> Result<Self> {
        let file = GgufFile::parse(bytes.to_vec())
            .map_err(|e| VokraError::ModelLoad(format!("csm GGUF: {e}")))?;
        let arch = file
            .get(vokra_core::gguf::chunks::KEY_MODEL_ARCH)
            .and_then(|v| v.as_str());
        if arch != Some(super::EXPECTED_ARCH) {
            return Err(VokraError::ModelLoad(format!(
                "not a CSM GGUF: vokra.model.arch = {arch:?}, expected `{}`",
                super::EXPECTED_ARCH
            )));
        }
        check_weight_license(&file, policy)?;
        let cfg = CsmConfig::from_gguf(&file)?;
        cfg.validate_for_forward()?;
        let mimi_cfg = MimiNeuralConfig::from_gguf(&file)?;
        mimi_cfg.validate()?;
        // Synthesized bridge (T29 pending — LlmBackbone::from_gguf
        // precedent, documented in the module docs).
        let model = CsmModel::synthesized(cfg.clone(), super::CSM_FROM_GGUF_DEFAULT_SEED)?;
        let encoder =
            MimiEncoder::synthesized(&mimi_cfg, super::CSM_FROM_GGUF_DEFAULT_SEED ^ 0x5EED)?;
        let neural = MimiNeuralDecoder::synthesized(
            &mimi_cfg,
            super::CSM_FROM_GGUF_DEFAULT_SEED ^ 0xDEC0,
            true,
        )?;
        let attrs = MimiRvqAttrs {
            n_codebooks: mimi_cfg.quantizer.n_q,
            codebook_size: mimi_cfg.quantizer.bins,
            d_model: mimi_cfg.quantizer.dimension,
        };
        let chain = CsmAudioDecodeChain::new(encoder.tables().to_vec(), attrs, neural)?;
        let tokenizer: Arc<dyn CsmTextTokenizer> =
            Arc::new(GgufCsmTokenizer::from_gguf(&file, cfg.text_vocab_size)?);
        Self::new(model, chain, encoder, tokenizer)
    }

    /// Loads from a file path with the fail-closed strict policy.
    ///
    /// # Errors
    ///
    /// See [`Self::from_gguf_with_policy`].
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).map_err(VokraError::Io)?;
        Self::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
    }

    /// Overrides the text tokenizer (fixture flows — module docs).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a vocab mismatch.
    pub fn with_tokenizer(mut self, tokenizer: Arc<dyn CsmTextTokenizer>) -> Result<Self> {
        if tokenizer.vocab_size() != self.model.config().text_vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "csm engine: tokenizer vocab {} != config text_vocab {}",
                tokenizer.vocab_size(),
                self.model.config().text_vocab_size
            )));
        }
        self.tokenizer = tokenizer;
        Ok(self)
    }

    /// Selects the echo path (default [`EchoPath::AecRequired`] —
    /// interactive posture; the bypass is an explicit opt-in for
    /// recorded-file input, T16 rustdoc).
    #[must_use]
    pub fn with_echo_path(mut self, path: EchoPath) -> Self {
        self.echo_path = path;
        self
    }

    /// Wires the AEC front (interactive dialog). The far-end queue is fed
    /// by this engine with every generated turn's PCM (playback clock).
    ///
    /// # Errors
    ///
    /// Propagates [`AecFront::new`] validation.
    pub fn with_aec(mut self, attrs: &AecAttrs, queue_capacity_samples: usize) -> Result<Self> {
        let (front, writer) = AecFront::new(attrs, queue_capacity_samples)?;
        self.aec = Some(Mutex::new(AecSession {
            front,
            writer,
            mic_pos: 0,
            play_pos: 0,
        }));
        Ok(self)
    }

    /// Overrides the watermark configuration (default **ON** — T26;
    /// embedding backend is Deferred, see [`WatermarkConfig::backend_status`],
    /// and deployer-side disclosure stays a MUST — docs/legal-compliance.md
    /// §1.4).
    #[must_use]
    pub fn with_watermark(mut self, watermark: WatermarkConfig) -> Self {
        self.watermark = watermark;
        self
    }

    /// The current watermark configuration.
    #[must_use]
    pub fn watermark(&self) -> &WatermarkConfig {
        &self.watermark
    }

    /// The resolved model config.
    #[must_use]
    pub fn config(&self) -> &CsmConfig {
        self.model.config()
    }

    /// The generation model (streaming session construction).
    #[must_use]
    pub(crate) fn model(&self) -> &CsmModel {
        &self.model
    }

    /// The decode chain (streaming session construction).
    #[must_use]
    pub(crate) fn chain(&self) -> &CsmAudioDecodeChain {
        &self.chain
    }

    /// Sampler for a request: deterministic → greedy (a plain
    /// [`vokra_core::decode::argmax`] — allocation-free, the T18 hot
    /// path); stochastic → the M1 [`Sampler`] at the upstream defaults
    /// (temperature 0.9 / top-k 50) with the request seed.
    #[must_use]
    pub fn sampler_for(request: &DialogRequest) -> CsmFrameSampler {
        if request.deterministic {
            CsmFrameSampler::Greedy
        } else {
            CsmFrameSampler::Stochastic(Sampler::new(SamplerConfig {
                temperature: DEFAULT_TEMPERATURE,
                top_k: Some(DEFAULT_TOP_K),
                top_p: None,
                repetition_penalty: None,
                seed: request.seed,
            }))
        }
    }

    /// Frame cap for a request (`min(request cap, engine default,
    /// remaining context)` — the backbone rejects `n_ctx` overflow anyway,
    /// this keeps the loop clean).
    fn max_frames_for(&self, request: &DialogRequest, context_len: usize) -> usize {
        let remaining = self.model.config().n_ctx.saturating_sub(context_len);
        request
            .max_frames
            .unwrap_or(self.default_max_frames)
            .min(self.default_max_frames)
            .min(remaining)
    }

    /// Builds the backbone context frames for a request: context turns
    /// (text → tokenizer, audio → Mimi encode), the cleaned input
    /// utterance, then the reply text (priming generation).
    pub(crate) fn build_context_frames(
        &self,
        request: &DialogRequest,
        cleaned_input: Option<&[f32]>,
    ) -> Result<Vec<CsmFrame>> {
        if request.reply_text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "csm dialog: reply_text is empty — CSM does not generate reply text \
                 (ADR M4-05 §D1-(b)); supply DialogRequest::reply_text (an upstream \
                 text LLM or caller-authored) and use S2s::dialog_request"
                    .into(),
            ));
        }
        let mut frames = Vec::new();
        for (i, turn) in request.context.iter().enumerate() {
            if turn.text.is_none() && turn.audio.is_none() {
                return Err(VokraError::InvalidArgument(format!(
                    "csm dialog: context turn {i} has neither text nor audio"
                )));
            }
            if let Some(text) = &turn.text {
                self.push_text_frames(turn.speaker, text, &mut frames)?;
            }
            if let Some(audio) = &turn.audio {
                self.push_audio_frames(audio, &mut frames)?;
            }
        }
        if let Some(pcm) = cleaned_input {
            self.push_audio_frames(pcm, &mut frames)?;
        }
        self.push_text_frames(request.reply_speaker, &request.reply_text, &mut frames)?;
        Ok(frames)
    }

    fn push_text_frames(&self, speaker: u32, text: &str, frames: &mut Vec<CsmFrame>) -> Result<()> {
        let ids = self.tokenizer.encode(&turn_text(speaker, text))?;
        let vocab = self.model.config().text_vocab_size as u32;
        for id in ids {
            if id >= vocab {
                return Err(VokraError::InvalidArgument(format!(
                    "csm dialog: tokenizer produced id {id} >= text_vocab {vocab}"
                )));
            }
            frames.push(CsmFrame::text(id));
        }
        Ok(())
    }

    fn push_audio_frames(&self, pcm: &[f32], frames: &mut Vec<CsmFrame>) -> Result<()> {
        let hop = self.encoder.frame_hop()?;
        if pcm.is_empty() || pcm.len() % hop != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "csm dialog: audio length {} is not a positive multiple of the frame \
                 hop {hop} — buffer whole frames or pad explicitly \
                 (pad_to_whole_frames); no silent zero-pad (FR-EX-08)",
                pcm.len()
            )));
        }
        let n_q = self.encoder.config().quantizer.n_q;
        let n_frames = pcm.len() / hop;
        let mut codes = vec![0u32; n_frames * n_q];
        // Fresh encoder state per segment (the upstream encodes each
        // segment one-shot).
        let mut state = self.encoder.state(n_frames)?;
        self.encoder.encode_into(&mut state, pcm, &mut codes)?;
        for f in 0..n_frames {
            frames.push(CsmFrame::audio(codes[f * n_q..(f + 1) * n_q].to_vec()));
        }
        Ok(())
    }

    /// Runs the AEC front over `mic` (interactive path) or passes it
    /// through on the explicit recorded-input bypass. Shared by the batch
    /// dialog and the streaming session (one AEC clock per engine —
    /// turn-interleaved).
    pub(crate) fn clean_input(&self, mic: &[f32]) -> Result<Vec<f32>> {
        require_echo_path_wiring(self.echo_path, self.aec.is_some())?;
        match self.echo_path {
            EchoPath::BypassRecordedInput => Ok(mic.to_vec()),
            EchoPath::AecRequired => {
                let session = self.aec.as_ref().expect("wiring guard passed");
                let mut s = session.lock().map_err(|_| {
                    VokraError::InvalidArgument(
                        "csm dialog: AEC session mutex poisoned (a prior panic mid-frame)".into(),
                    )
                })?;
                let fs = s.front.frame_size();
                if mic.is_empty() || mic.len() % fs != 0 {
                    return Err(VokraError::InvalidArgument(format!(
                        "csm dialog: mic length {} is not a positive multiple of the AEC \
                         frame size {fs} (FR-EX-08 — buffer whole frames)",
                        mic.len()
                    )));
                }
                let mut out = vec![0.0f32; mic.len()];
                for (i, chunk) in mic.chunks_exact(fs).enumerate() {
                    let pos = s.mic_pos;
                    let dst = &mut out[i * fs..(i + 1) * fs];
                    // Split-borrow: front and the clock live in one guard.
                    let AecSession { front, mic_pos, .. } = &mut *s;
                    front.process_mic(chunk, pos, dst)?;
                    *mic_pos += fs as u64;
                }
                Ok(out)
            }
        }
    }

    /// Feeds a generated turn's PCM into the far-end reference queue
    /// (playback clock) so the *next* turn's mic frames cancel against it.
    fn push_playback_reference(&self, pcm: &[f32]) -> Result<()> {
        if let Some(session) = &self.aec {
            let mut s = session.lock().map_err(|_| {
                VokraError::InvalidArgument("csm dialog: AEC session mutex poisoned".into())
            })?;
            let pos = s.play_pos;
            s.writer.push(pcm, pos)?;
            s.play_pos += pcm.len() as u64;
        }
        Ok(())
    }
}

/// Per-frame code sampler: the greedy arm is a stateless
/// [`vokra_core::decode::argmax`] (zero allocation — the M1 `Sampler`'s
/// repetition-penalty context grows on every draw, which the T18
/// counting-allocator proof would flag); the stochastic arm is the M1
/// [`Sampler`] with the upstream defaults.
#[derive(Debug)]
pub enum CsmFrameSampler {
    /// Temperature-0 argmax (deterministic parity anchor).
    Greedy,
    /// Seeded M1 sampler (temperature 0.9 / top-k 50 — generator.py).
    Stochastic(Sampler),
}

impl CsmFrameSampler {
    /// Draws one id from `logits`.
    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        match self {
            Self::Greedy => vokra_core::decode::argmax(logits),
            Self::Stochastic(s) => s.sample(logits),
        }
    }
}

/// The (提案) fixture turn-text format — module docs "Speaker
/// conditioning note": the real prefix format is pinned at T29.
pub(crate) fn turn_text(speaker: u32, text: &str) -> String {
    format!("[{speaker}]{text}")
}

/// Explicit zero-pad helper: pads `pcm` up to a whole multiple of
/// `frame_hop` (the engine itself never pads silently — FR-EX-08).
#[must_use]
pub fn pad_to_whole_frames(pcm: &[f32], frame_hop: usize) -> Vec<f32> {
    let mut out = pcm.to_vec();
    if frame_hop > 0 {
        let rem = out.len() % frame_hop;
        if rem != 0 {
            out.resize(out.len() + (frame_hop - rem), 0.0);
        }
    }
    out
}

impl S2sEngine for CsmEngine {
    fn dialog(&self, request: &DialogRequest) -> Result<DialogTurn> {
        // Input front (AEC / explicit bypass).
        let cleaned = match &request.input_audio {
            Some(mic) => Some(self.clean_input(mic)?),
            None => None,
        };
        let frames = self.build_context_frames(request, cleaned.as_deref())?;
        let mut generation = CsmGenerationState::new(self.model.config())?;
        self.model.prime(&mut generation, &frames)?;
        let max_frames = self.max_frames_for(request, generation.context_len());
        let mut audio_state: CsmAudioDecodeState = self.chain.state(max_frames.max(1))?;
        let mut sampler = Self::sampler_for(request);
        let hop = self.chain.frame_hop()?;
        let n_cb = self.model.config().n_codebooks;
        let mut codes = vec![0u32; n_cb];
        let mut pcm = Vec::with_capacity(max_frames * hop);
        let mut frame_pcm = vec![0.0f32; hop];
        for _ in 0..max_frames {
            match self.model.generate_frame_into(
                &mut generation,
                &mut |l| sampler.sample(l),
                &mut codes,
            )? {
                CsmFrameKind::Eos => break,
                CsmFrameKind::Audio => {
                    self.chain
                        .decode_frame_into(&mut audio_state, &codes, &mut frame_pcm)?;
                    pcm.extend_from_slice(&frame_pcm);
                }
            }
        }
        // Feed the far-end queue so the next turn's mic cancels against
        // this turn's playback.
        self.push_playback_reference(&pcm)?;
        Ok(DialogTurn::new(
            request.reply_text.clone(),
            Some(SynthesizedAudio::new(pcm, self.model.config().sample_rate)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> CsmEngine {
        CsmEngine::synthesized_fixture(21).expect("fixture engine")
    }

    fn request() -> DialogRequest {
        DialogRequest::new("hello vokra")
            .with_reply_speaker(1)
            .deterministic()
    }

    #[test]
    fn dialog_speaks_the_reply_text_and_returns_audio() {
        let e = engine().with_echo_path(EchoPath::BypassRecordedInput);
        let turn = e.dialog(&request().with_max_frames(4)).unwrap();
        assert_eq!(turn.text, "hello vokra");
        let audio = turn.audio.expect("audio produced");
        assert_eq!(audio.sample_rate, e.config().sample_rate);
        let hop = e.chain().frame_hop().unwrap();
        assert_eq!(audio.samples.len() % hop, 0, "whole frames");
        assert!(audio.samples.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn deterministic_dialog_is_reproducible() {
        let e = engine().with_echo_path(EchoPath::BypassRecordedInput);
        let a = e.dialog(&request().with_max_frames(3)).unwrap();
        let b = e.dialog(&request().with_max_frames(3)).unwrap();
        assert_eq!(a.audio.unwrap().samples, b.audio.unwrap().samples);
    }

    #[test]
    fn empty_reply_text_is_a_loud_contract_error() {
        let e = engine().with_echo_path(EchoPath::BypassRecordedInput);
        let err = e.dialog(&DialogRequest::new("")).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
        assert!(err.to_string().contains("reply_text"), "actionable message");
    }

    #[test]
    fn input_audio_without_aec_wiring_is_rejected_by_default() {
        // Default echo path = AecRequired; no front wired → loud error
        // that names the bypass (FR-OP-60 / T16).
        let e = engine();
        let hop = e.encoder.frame_hop().unwrap();
        let err = e
            .dialog(&request().with_input_audio(vec![0.0; hop]))
            .unwrap_err();
        assert!(err.to_string().contains("BypassRecordedInput"));
    }

    #[test]
    fn recorded_bypass_accepts_whole_frame_audio_and_rejects_partial() {
        let e = engine().with_echo_path(EchoPath::BypassRecordedInput);
        let hop = e.encoder.frame_hop().unwrap();
        let ok = e.dialog(
            &request()
                .with_input_audio(vec![0.1; hop * 2])
                .with_max_frames(2),
        );
        assert!(ok.is_ok());
        let err = e
            .dialog(&request().with_input_audio(vec![0.1; hop + 1]))
            .unwrap_err();
        assert!(err.to_string().contains("pad_to_whole_frames"));
    }

    #[test]
    fn aec_wired_dialog_processes_mic_and_feeds_playback_reference() {
        let attrs = AecAttrs {
            sample_rate: 16_000,
            frame_size: 128,
            filter_length: 512,
        };
        let e = CsmEngine::synthesized_fixture(3)
            .unwrap()
            .with_aec(&attrs, 16_000)
            .unwrap();
        // Mic must arrive in AEC frame multiples AND encode frame
        // multiples; 128 is both (1 AEC frame = 16 audio frames — inside
        // the tiny n_ctx budget).
        let mic = vec![0.05f32; 128];
        let turn = e
            .dialog(&request().with_input_audio(mic).with_max_frames(2))
            .unwrap();
        assert!(turn.audio.is_some());
        // The playback clock advanced (reference queue fed) — a second
        // turn keeps working.
        let turn2 = e.dialog(&request().with_max_frames(2)).unwrap();
        assert!(turn2.audio.is_some());
    }

    #[test]
    fn context_turns_condition_the_generation() {
        let e = engine().with_echo_path(EchoPath::BypassRecordedInput);
        let base = e.dialog(&request().with_max_frames(3)).unwrap();
        let with_ctx = e
            .dialog(
                &request()
                    .with_context_turn(vokra_core::DialogContextTurn::text(0, "context line"))
                    .with_max_frames(3),
            )
            .unwrap();
        // Different context → (deterministically) different audio.
        assert_ne!(
            base.audio.unwrap().samples,
            with_ctx.audio.unwrap().samples,
            "context must condition the frames"
        );
    }

    #[test]
    fn watermark_config_default_is_on_and_backend_deferred() {
        let e = engine();
        assert!(!e.watermark().audioseal_opted_out(), "default ON (T26)");
    }

    #[test]
    fn from_gguf_loads_fixture_metadata_and_rejects_placeholders() {
        use vokra_core::gguf::GgufBuilder;
        // Fixture-shaped GGUF (tiny config + mimi chunk + tokenizer blob).
        let cfg = CsmConfig::tiny_for_tests();
        let mut mimi_cfg = MimiNeuralConfig::tiny_for_tests();
        mimi_cfg.quantizer.n_q = cfg.n_codebooks;
        mimi_cfg.quantizer.bins = cfg.audio_vocab_size;
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "csm");
        // The converter always stamps provenance (T04); the gate resolves
        // the explicit weight-license class from it (T26).
        vokra_core::stamp_provenance(
            &mut b,
            vokra_core::LicenseClass::Permissive,
            "Apache-2.0",
            Some("sesame/csm-1b"),
            None,
        );
        let mut fixed = cfg.clone();
        fixed.sample_rate = mimi_cfg.sample_rate;
        fixed.frame_rate_mhz = mimi_cfg.frame_rate_mhz;
        fixed.write_gguf_metadata(&mut b);
        mimi_cfg.write_gguf_metadata(&mut b);
        b.add_metadata(
            "vokra.tokenizer.model",
            vokra_core::gguf::GgufMetadataValue::Array(vokra_core::gguf::GgufArray {
                element_type: vokra_core::gguf::GgufValueType::U8,
                values: vec![vokra_core::gguf::GgufMetadataValue::U8(1)],
            }),
        );
        let bytes = b.to_bytes().unwrap();
        let e = CsmEngine::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect("fixture-shaped GGUF loads (synthesized bridge)");
        // The GGUF tokenizer is honest: encode = NotImplemented until T29.
        let err = e
            .with_echo_path(EchoPath::BypassRecordedInput)
            .dialog(&request())
            .unwrap_err();
        assert!(matches!(err, VokraError::NotImplemented(_)));

        // A provenance-less GGUF is refused by the M2-13 gate first
        // (fail-closed: unknown weight license needs a research flag).
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "csm");
        let bytes = b.to_bytes().unwrap();
        assert!(matches!(
            CsmEngine::from_gguf_with_policy(&bytes, &CompliancePolicy::strict()),
            Err(VokraError::ResearchLicenseRequired { .. })
        ));

        // A 0-placeholder GGUF (scaffold converter posture, provenance
        // stamped) passes the gate and then fails the shape validation
        // loudly.
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "csm");
        vokra_core::stamp_provenance(
            &mut b,
            vokra_core::LicenseClass::Permissive,
            "Apache-2.0",
            Some("sesame/csm-1b"),
            None,
        );
        let bytes = b.to_bytes().unwrap();
        assert!(matches!(
            CsmEngine::from_gguf_with_policy(&bytes, &CompliancePolicy::strict()),
            Err(VokraError::InvalidArgument(_))
        ));

        // A wrong arch is a ModelLoad error.
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "whisper");
        let bytes = b.to_bytes().unwrap();
        assert!(matches!(
            CsmEngine::from_gguf_with_policy(&bytes, &CompliancePolicy::strict()),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn pad_helper_is_explicit_and_exact() {
        assert_eq!(pad_to_whole_frames(&[1.0; 5], 4).len(), 8);
        assert_eq!(pad_to_whole_frames(&[1.0; 8], 4).len(), 8);
        assert_eq!(pad_to_whole_frames(&[], 4).len(), 0);
    }
}
