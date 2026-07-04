//! piper-plus native TTS — MB-iSTFT-VITS2 (M0-07). Vokra's first native TTS.
//!
//! Native re-implementation of the piper-plus (MB-iSTFT-VITS2) inference core
//! — text encoder / duration predictor / flow / MB-iSTFT decoder — in the
//! whisper.cpp style (client decision 2026-07-02; the former wrap approach is
//! abolished, ADR-0002). The voice model is converted offline to GGUF by
//! `vokra-convert` (M0-03); no ONNX runs at runtime (FR-LD-05). G2P (8
//! languages) is bridged through the `vokra-piper-plus` crate for now.
//!
//! # Layout (M0-07)
//!
//! - model definition + GGUF load (config, phoneme table, iSTFT params);
//! - text encoder, (stochastic) duration predictor + length regulation,
//!   flow (residual coupling), MB-iSTFT decoder — the decoder is the first
//!   real consumer of the `vokra-ops` `istft` op (M0-04);
//! - a [`vokra_core::engines::TtsEngine`] implementation wired to
//!   `session.tts().synthesize()`, with deterministic (noise-off) synthesis
//!   for reference parity against piper-plus onnxruntime.
//!
//! Implementation lands with M0-07 (T06–T24); see `docs/piper-plus-integration.md`.

mod conditioning;
mod config;
mod decoder;
mod duration;
mod flow;
mod nn;
mod text_encoder;
mod weights;

#[cfg(test)]
mod clone_integration;
#[cfg(test)]
mod parity;
#[cfg(test)]
mod parity_v7;
#[cfg(test)]
mod parity_v7_prosody;

use std::path::Path;

use vokra_core::gguf::GgufFile;
use vokra_core::rng::GaussianSplitMix64;
use vokra_core::{
    CompliancePolicy, Result, SynthesisRequest, SynthesizedAudio, TtsEngine, VokraError,
    check_weight_license,
};
use vokra_ops::{KaldiFbankOpts, kaldi_fbank, resample};
use vokra_piper_plus::{PhonemeTable, Phonemizer};

use crate::speaker::SpeakerEncoder;

pub use config::PiperConfig;
// Re-export the G2P reuse-boundary types so downstreams can build the injected
// phonemizer without also depending on `vokra-piper-plus` directly (M1-01-A).
pub use vokra_piper_plus::{MockPhonemizer, PassthroughPhonemizer};

use config::{Dims, HIDDEN};

/// `vokra.model.arch` a piper-plus voice GGUF must carry (written by
/// `vokra-convert`'s `models::piper_plus::ARCH`; kept in sync by that
/// converter, M0-07-T06). A wrong arch fails loudly at load (M1-01-C).
const EXPECTED_ARCH: &str = "piper-plus-mb-istft-vits2";

use conditioning::Conditioning;
use decoder::Decoder;
use duration::DurationPredictor;
use flow::Flow;
use text_encoder::TextEncoder;
use weights::TensorStore;

/// A loaded piper-plus (MB-iSTFT-VITS2) voice: Vokra's first native TTS.
///
/// Built from a voice GGUF (produced offline by `vokra-convert`, M0-07-T07); no
/// ONNX is touched at runtime (FR-LD-05). The MB-iSTFT-VITS2 inference core is
/// assembled here from the loaded weights (M0-07-T11..T20).
pub struct PiperPlusTts {
    config: PiperConfig,
    /// Global speaker/language conditioning `g = spk_proj(speaker) + emb_lang`
    /// (M1 zero-shot v7): the single source of the vector shared by the encoder,
    /// duration predictor, flow and decoder.
    conditioning: Conditioning,
    encoder: TextEncoder,
    duration: DurationPredictor,
    flow: Flow,
    decoder: Decoder,
    /// Prosody projection (`prosody_proj`): the per-phoneme `(A1, A2, A3)` accent
    /// features → `PROSODY_DIM` channels appended to the encoder output for the
    /// duration predictor. With zero features (EN / mock-G2P path) it collapses
    /// to the bias broadcast over time.
    prosody_proj: ProsodyProj,
}

/// The v7 prosody projection: `channels = (features · gate) @ weight + bias`.
///
/// `weight` is the raw MatMul matrix `[in_dim, out_dim]` (row-major, **not** the
/// PyTorch `[out, in]` Linear layout) and `bias` is `[out_dim]`. The features
/// are gated to zero for every language except [`PROSODY_LANG_ID`] (JA), so a
/// `None` (or non-JA) request leaves the projection at its bias — the reference
/// parity setting (the committed v7 fixtures use zero prosody, so only the bias
/// path is reference-verified; the MatMul orientation follows the ONNX graph and
/// is pinned by a unit test).
struct ProsodyProj {
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl ProsodyProj {
    /// Prosody channels `[out_dim, t]` (channel-major) for the duration
    /// predictor. `features` is `[t, in_dim]` i64 (per-phoneme `(A1, A2, A3)`);
    /// `None`, or a non-JA `lid`, yields the bias broadcast over time. The caller
    /// guarantees `features.len() == in_dim · t` (checked in
    /// [`PiperPlusTts::check_prosody_len`]).
    fn channels(&self, features: Option<&[i64]>, lid: i64, t: usize) -> Vec<f32> {
        let (in_dim, out_dim) = (self.in_dim, self.out_dim);
        // Bias, broadcast over time: out[o, ti] = bias[o].
        let mut out = vec![0.0f32; out_dim * t];
        for (o, chunk) in out.chunks_exact_mut(t).enumerate() {
            chunk.fill(self.bias[o]);
        }
        // JA-only gate (`Equal(lid, PROSODY_LANG_ID)` in the graph): every other
        // language (and a featureless request) leaves the bias-only channels.
        if lid == config::PROSODY_LANG_ID {
            if let Some(f) = features {
                // out[o, ti] += Σ_i features[ti, i] · weight[i, o].
                for ti in 0..t {
                    for o in 0..out_dim {
                        let mut acc = 0.0f32;
                        for i in 0..in_dim {
                            acc += f[ti * in_dim + i] as f32 * self.weight[i * out_dim + o];
                        }
                        out[o * t + ti] += acc;
                    }
                }
            }
        }
        out
    }
}

impl PiperPlusTts {
    /// Loads a voice from a GGUF file on disk.
    ///
    /// # Errors
    ///
    /// Propagates GGUF parse errors and any weight shape/metadata mismatch
    /// (a wrong or corrupt voice fails loudly at load).
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_path_with_policy(path, &CompliancePolicy::strict())
    }

    /// Loads a voice from disk under an explicit compliance `policy` — the
    /// weight-license research-flag gate entry point (FR-CP-03, M2-13).
    ///
    /// [`from_path`](Self::from_path) is exactly this with the default
    /// fail-closed [`CompliancePolicy::strict`]. A stock piper-plus voice is
    /// MIT-weight (permissive) and always loads; the gate only bites a voice
    /// GGUF explicitly tagged non-commercial (`vokra.provenance.*`).
    pub fn from_path_with_policy(
        path: impl AsRef<Path>,
        policy: &CompliancePolicy,
    ) -> Result<Self> {
        let file = GgufFile::open(path)
            .map_err(|e| vokra_core::VokraError::ModelLoad(format!("piper voice GGUF: {e}")))?;
        Self::from_gguf_with_policy(file, policy)
    }

    /// Loads a voice from an already-parsed GGUF (default fail-closed policy).
    ///
    /// The GGUF's `vokra.model.arch` is checked first, so a non-piper (or wrong
    /// architecture) GGUF fails with a clear [`VokraError::ModelLoad`] rather
    /// than a confusing missing-tensor error deep in a component loader
    /// (M1-01-C). The retained GGUF backing bytes (~77 MB FP32) are dropped once
    /// every component has copied its tensors out, halving resident memory: the
    /// `TensorStore` is a function local and is freed at the end of load.
    pub fn from_gguf(file: GgufFile) -> Result<Self> {
        Self::from_gguf_with_policy(file, &CompliancePolicy::strict())
    }

    /// Loads a voice from an already-parsed GGUF under an explicit compliance
    /// `policy`.
    ///
    /// After the `vokra.model.arch` check, the **weight-license gate**
    /// ([`check_weight_license`], FR-CP-03) runs on the container *before* any
    /// weight tensor is bound: a non-commercial / unknown weight license
    /// (`vokra.provenance.*`) without a research flag is refused with
    /// [`VokraError::ResearchLicenseRequired`] — never a silent load. A stock
    /// MIT piper voice classifies permissive (built-in registry, arch
    /// `piper-plus-mb-istft-vits2`) and passes. This is the single M2-13
    /// enforcement path wired in `vokra-models`; the other model loaders
    /// (whisper / silero / campplus) are a follow-up.
    pub fn from_gguf_with_policy(file: GgufFile, policy: &CompliancePolicy) -> Result<Self> {
        let store = TensorStore::new(file);
        let arch = store
            .file()
            .get(vokra_core::gguf::chunks::KEY_MODEL_ARCH)
            .and_then(|v| v.as_str());
        if arch != Some(EXPECTED_ARCH) {
            return Err(vokra_core::VokraError::ModelLoad(format!(
                "not a piper-plus voice GGUF: vokra.model.arch = {arch:?}, expected `{EXPECTED_ARCH}`"
            )));
        }
        // Weight-license research-flag gate (FR-CP-03): refuse a non-commercial
        // / unknown-provenance voice unless the policy grants a research flag.
        // Runs before the heavy weight load so a gated model fails fast.
        check_weight_license(store.file(), policy)?;
        let config = PiperConfig::from_gguf(store.file())?;
        // Shape-derived dimensions (single-speaker medium vs zero-shot v7 FiLM);
        // threaded into the components, dropped after load.
        let dims = Dims::derive(&store)?;
        let conditioning = Conditioning::load(&store, &dims, config.num_languages)?;
        let encoder = TextEncoder::load(&store, &dims, config.num_symbols)?;
        let duration = DurationPredictor::load(&store, &dims)?;
        let prosody_proj = ProsodyProj {
            weight: store
                .tensor_shaped("prosody_proj.weight", &[dims.prosody_in, dims.prosody_out])?,
            bias: store.tensor_shaped("prosody_proj.bias", &[dims.prosody_out])?,
            in_dim: dims.prosody_in,
            out_dim: dims.prosody_out,
        };
        let flow = Flow::load(&store, &dims)?;
        let decoder = Decoder::load(
            &store,
            &dims,
            config.istft_n_fft,
            config.istft_hop,
            config.pqmf_subbands,
        )?;
        // `store` (and its GGUF backing bytes) drops here — every component owns
        // its own copies now, so nothing borrows it past load.
        Ok(Self {
            config,
            conditioning,
            encoder,
            duration,
            prosody_proj,
            flow,
            decoder,
        })
    }

    /// The resolved voice configuration (sample rate, tables, scales, ...).
    pub fn config(&self) -> &PiperConfig {
        &self.config
    }

    /// The external speaker-embedding width this voice's `spk_proj` expects —
    /// 192 (the CAM++ output) for the zero-shot v7 voice. Any embedding handed
    /// to a request via [`SynthesisRequest::speaker_embedding`] must have this
    /// length (a shorter/longer one falls back to the zero vector).
    pub fn speaker_embedding_dim(&self) -> usize {
        self.conditioning.spk_emb_dim()
    }

    /// Turns reference audio into the speaker embedding that drives zero-shot
    /// voice cloning — the replacement for the zero-embedding fallback (M0-08).
    ///
    /// `reference_pcm` is mono PCM at `sample_rate`; it is resampled to the CAM++
    /// encoder's 16 kHz when needed, converted to a Kaldi fbank
    /// ([`kaldi_fbank`]), and run through the native `encoder`. The returned
    /// `Vec<f32>` (length [`speaker_embedding_dim`](Self::speaker_embedding_dim))
    /// is dropped straight into [`SynthesisRequest::speaker_embedding`] /
    /// [`synthesize_phonemes`](Self::synthesize_phonemes). The encoder is passed
    /// in rather than owned, so a voice still loads from a single GGUF and a
    /// caller that never clones a voice pays nothing for the speaker model.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if the `encoder`'s output width does not
    ///   match this voice's `spk_proj` input (a mismatched speaker model), or if
    ///   `reference_pcm` is shorter than one fbank frame (too short to embed);
    /// - propagates resample / fbank / CAM++ forward errors.
    pub fn embed_reference(
        &self,
        encoder: &SpeakerEncoder,
        reference_pcm: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<f32>> {
        let opts = KaldiFbankOpts::camplus();
        // The CAM++ encoder emits EMBED_DIM (192); reject a voice whose spk_proj
        // was trained on a different width before doing any work.
        let want = self.speaker_embedding_dim();
        if crate::speaker::EMBED_DIM != want {
            return Err(VokraError::InvalidArgument(format!(
                "piper TTS: CAM++ embedding dim {} != voice speaker_embedding_dim {want}",
                crate::speaker::EMBED_DIM
            )));
        }
        // Resample to the encoder's rate (a bit-exact copy is avoided when the
        // reference is already 16 kHz — the common case).
        let owned;
        let pcm16k: &[f32] = if sample_rate == opts.sample_rate {
            reference_pcm
        } else {
            owned = resample(
                reference_pcm,
                sample_rate,
                opts.sample_rate,
                vokra_ops::resample::DEFAULT_QUALITY,
            )?;
            &owned
        };
        let (fbank, t) = kaldi_fbank(pcm16k, &opts)?;
        let emb = encoder.embed(&fbank, t)?;
        Ok(emb.to_vec())
    }

    /// Synthesizes PCM from a phoneme id sequence — the full native
    /// MB-iSTFT-VITS2 path (encoder → duration predictor → length regulation →
    /// flow → decoder), M0-07-T20.
    ///
    /// The zero-shot conditioning inputs are `speaker_embedding`
    /// (`speaker_embedding_dim` floats; `None` → the zero vector, which the
    /// speaker projection still maps to a non-zero contribution) and
    /// `prosody_features` (the flattened `[T_phonemes · 3]` `(A1, A2, A3)`
    /// triples for the JA prosody feed; `None` → the bias-only channels). They
    /// compose the global conditioning `g` shared by every stage.
    ///
    /// `noise_scale` / `noise_w` are the VITS stochastic knobs; passing `0` for
    /// both makes the whole path deterministic (the parity setting, docs §5).
    /// Non-zero scales draw Gaussian noise from a fixed-seed RNG (reproducible,
    /// but not bit-matched to onnxruntime — that path is exercised only for
    /// audio, not parity).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`](vokra_core::VokraError::InvalidArgument)
    /// if a phoneme / language id is out of range, or `prosody_features` is
    /// present with a length other than `3 · phoneme_ids.len()`.
    // The explicit low-level primitive takes the model's natural conditioning
    // inputs (speaker embedding + prosody) and VITS knobs (two noise scales +
    // length scale) individually; the ergonomic path is `synthesize_with` /
    // `SynthesisRequest`.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_phonemes(
        &self,
        phoneme_ids: &[i64],
        lid: i64,
        speaker_embedding: Option<&[f32]>,
        prosody_features: Option<&[i64]>,
        noise_scale: f32,
        length_scale: f32,
        noise_w: f32,
    ) -> Result<SynthesizedAudio> {
        self.check_ids(phoneme_ids, lid)?;
        self.check_prosody_len(prosody_features, phoneme_ids.len())?;
        let g = self.conditioning.g(speaker_embedding, lid);
        let enc = self.encoder.forward(phoneme_ids, &g)?;
        let prosody = self.prosody_proj.channels(prosody_features, lid, enc.t);
        let x_dp = build_x_dp(&enc.x, &prosody, enc.t);
        let logw = self.duration.logw(&x_dp, enc.t, &g, noise_w);
        let w_ceil: Vec<usize> = logw
            .iter()
            .map(|&l| ((l.exp() * length_scale).ceil() as i64).max(1) as usize)
            .collect();

        let (mut z_p, t_frames) = length_regulate(&enc.m_p, HIDDEN, enc.t, &w_ceil);
        if noise_scale != 0.0 {
            let (logs_exp, _) = length_regulate(&enc.logs_p, HIDDEN, enc.t, &w_ceil);
            let mut rng = GaussianSplitMix64::new(0x5eed_1234_abcd_0007);
            for (z, ls) in z_p.iter_mut().zip(&logs_exp) {
                *z += rng.next_gaussian() * ls.exp() * noise_scale;
            }
        }
        let z = self.flow.reverse(&z_p, t_frames, &g);
        let pcm = self.decoder.forward(&z, t_frames, &g)?;
        Ok(SynthesizedAudio::new(pcm, self.config.sample_rate))
    }

    /// Defensive bounds check on the inputs to the embedding lookups, run before
    /// any table indexing (M1-01-C). A real 8-language G2P can emit phoneme ids
    /// that exceed this voice's table (e.g. the sv/ko PUA ids 173–184 that the
    /// converter flags as over-range, `PiperPlusReport::phoneme_ids_over_range`,
    /// `docs/piper-plus-integration.md` §8 A-4); catching them here turns a
    /// panic in the embedding lookup into a clear [`VokraError::InvalidArgument`].
    fn check_ids(&self, phoneme_ids: &[i64], lid: i64) -> Result<()> {
        if let Some(&bad) = phoneme_ids
            .iter()
            .find(|&&id| id < 0 || id as usize >= self.config.num_symbols)
        {
            return Err(vokra_core::VokraError::InvalidArgument(format!(
                "piper TTS: phoneme id {bad} out of range (num_symbols = {})",
                self.config.num_symbols
            )));
        }
        if lid < 0 || lid as usize >= self.config.num_languages {
            return Err(vokra_core::VokraError::InvalidArgument(format!(
                "piper TTS: language id {lid} out of range (num_languages = {})",
                self.config.num_languages
            )));
        }
        Ok(())
    }

    /// Checks that a supplied flattened prosody buffer carries exactly one
    /// `(A1, A2, A3)` triple per phoneme (`in_dim · n_phonemes` values), so the
    /// [`ProsodyProj::channels`] indexing is in-bounds. A `None` buffer (the
    /// default / EN path) is always valid.
    fn check_prosody_len(&self, prosody_features: Option<&[i64]>, n_phonemes: usize) -> Result<()> {
        if let Some(pf) = prosody_features {
            let want = self.prosody_proj.in_dim * n_phonemes;
            if pf.len() != want {
                return Err(vokra_core::VokraError::InvalidArgument(format!(
                    "piper TTS: prosody_features has {} values, expected {want} \
                     ({} features × {n_phonemes} phonemes)",
                    pf.len(),
                    self.prosody_proj.in_dim,
                )));
            }
        }
        Ok(())
    }

    /// Builds a [`PhonemeTable`] from this voice's phoneme symbol table, for
    /// driving [`synthesize_with`](Self::synthesize_with) with a [`Phonemizer`]
    /// such as [`MockPhonemizer`] or [`PassthroughPhonemizer`] (M1-01-A).
    ///
    /// # Errors
    ///
    /// Fails if the voice's table lacks the piper framing symbols `_`/`^`/`$`.
    pub fn phoneme_table(&self) -> Result<PhonemeTable> {
        PhonemeTable::from_symbols(&self.config.phoneme_symbols)
    }

    /// Synthesizes PCM for `request`, converting text → phoneme ids through an
    /// injected [`Phonemizer`] — the **G2P reuse boundary** (M1-01-A,
    /// `docs/piper-plus-integration.md` §7).
    ///
    /// The default [`TtsEngine::synthesize`] path uses the built-in placeholder
    /// tokenizer so the zero-dependency build still emits audio for the demo.
    /// `synthesize_with` is the escape hatch: a downstream that runs the
    /// out-of-workspace 8-language `piper-plus-g2p` (or any other G2P) injects
    /// it here, so the core never takes a non-`vokra-*` dependency (NFR-DS-02).
    /// [`PassthroughPhonemizer`] covers callers that already hold phoneme
    /// content. The resulting ids are range-checked before the embedding lookup.
    pub fn synthesize_with(
        &self,
        request: &SynthesisRequest,
        phonemizer: &dyn Phonemizer,
    ) -> Result<SynthesizedAudio> {
        let lid = request
            .language
            .as_deref()
            .and_then(|c| self.config.language_id(c))
            .unwrap_or(0);
        let phoneme_ids = phonemizer.phonemize(&request.text)?;
        let (noise, noise_w) = if request.deterministic {
            (0.0, 0.0)
        } else {
            (self.config.noise_scale, self.config.noise_w)
        };
        let prosody = request
            .prosody_features
            .as_deref()
            .map(|p| p.as_flattened());
        self.synthesize_phonemes(
            &phoneme_ids,
            lid,
            request.speaker_embedding.as_deref(),
            prosody,
            noise,
            self.config.length_scale,
            noise_w,
        )
    }

    /// Synthesizes PCM by running a **prosody- and language-aware** G2P over
    /// `request.text`: the injected [`Phonemizer`] returns not just phoneme ids
    /// but the per-phoneme prosody triples and detected language id
    /// ([`Phonemizer::phonemize_full`]) that the multilingual piper-plus models
    /// (e.g. the zero-shot 6-language v7) consume. This is the full text→speech
    /// entry point for the out-of-workspace `piper-plus-g2p` reuse; the
    /// zero-dependency core still never links a non-`vokra-*` crate — the G2P is
    /// injected across the trait boundary (NFR-DS-02).
    ///
    /// Overrides applied to the phonemizer's output:
    /// - `request.language`, when it names a known language, pins `lid` over the
    ///   phonemizer's detected language;
    /// - `request.speaker_embedding` supplies the zero-shot speaker (`None` → the
    ///   zero vector, whose projection is still non-zero);
    /// - `request.prosody_features`, when `Some`, overrides the phonemizer's
    ///   prosody; otherwise the phonemizer's per-phoneme triples are used,
    ///   aligned 1:1 with the ids.
    ///
    /// # Errors
    ///
    /// Propagates phonemization errors, and [`VokraError::InvalidArgument`] if a
    /// phoneme / language id is out of range or a prosody length disagrees with
    /// the phoneme count (see [`synthesize_phonemes`](Self::synthesize_phonemes)).
    pub fn synthesize_full(
        &self,
        request: &SynthesisRequest,
        phonemizer: &dyn Phonemizer,
    ) -> Result<SynthesizedAudio> {
        let utt = phonemizer.phonemize_full(&request.text)?;
        // request.language override wins; otherwise trust the phonemizer's lid.
        let lid = request
            .language
            .as_deref()
            .and_then(|c| self.config.language_id(c))
            .unwrap_or(utt.lid);
        let (noise, noise_w) = if request.deterministic {
            (0.0, 0.0)
        } else {
            (self.config.noise_scale, self.config.noise_w)
        };
        // request prosody overrides the phonemizer's; both are flattened [T · 3].
        let g2p_prosody: Option<Vec<i64>> = if utt.prosody.is_empty() {
            None
        } else {
            Some(utt.prosody.as_flattened().to_vec())
        };
        let prosody = request
            .prosody_features
            .as_deref()
            .map(|p| p.as_flattened())
            .or(g2p_prosody.as_deref());
        self.synthesize_phonemes(
            &utt.ids,
            lid,
            request.speaker_embedding.as_deref(),
            prosody,
            noise,
            self.config.length_scale,
            noise_w,
        )
    }

    /// A placeholder tokenizer: maps each input character to a phoneme id via
    /// the voice's phoneme table and applies BOS/PAD/EOS framing (mirrors
    /// `vokra_piper_plus::MockPhonemizer` — real G2P reuse is T09). Unknown
    /// characters are dropped.
    pub fn tokenize(&self, text: &str) -> Vec<i64> {
        let id_of = |sym: &str| -> Option<i64> {
            self.config
                .phoneme_symbols
                .iter()
                .position(|s| s == sym)
                .map(|i| i as i64)
        };
        let (bos, eos, pad) = (id_of("^"), id_of("$"), id_of("_"));
        let mut ids: Vec<i64> = Vec::new();
        if let Some(b) = bos {
            ids.push(b);
        }
        let mut buf = [0u8; 4];
        for c in text.chars() {
            if let Some(id) = id_of(c.encode_utf8(&mut buf)) {
                ids.push(id);
                if let Some(p) = pad {
                    ids.push(p);
                }
            }
        }
        if let Some(e) = eos {
            ids.push(e);
        }
        ids
    }

    /// Runs the text encoder for a phoneme id sequence under language `lid`
    /// (component boundary used by the M0-07-T13 parity test).
    #[cfg(test)]
    pub(crate) fn encode(&self, phoneme_ids: &[i64], lid: i64) -> Result<text_encoder::EncoderOut> {
        let g = self.conditioning.g(None, lid);
        self.encoder.forward(phoneme_ids, &g)
    }

    /// The global conditioning `g = spk_proj(speaker) + emb_lang[lid]` `[gin]`
    /// (component boundary for the v7 parity test; `None` speaker = zeros).
    #[cfg(test)]
    pub(crate) fn global_g(&self, speaker_embedding: Option<&[f32]>, lid: i64) -> Vec<f32> {
        self.conditioning.g(speaker_embedding, lid)
    }

    /// Runs the MB-iSTFT decoder on a decoder-input latent `z` `[HIDDEN, T]`
    /// under language `lid` (component boundary used by the M0-07-T19 parity
    /// test: reference latent → PCM).
    #[cfg(test)]
    pub(crate) fn decode(&self, z: &[f32], t_frames: usize, lid: i64) -> Result<Vec<f32>> {
        let g = self.conditioning.g(None, lid);
        self.decoder.forward(z, t_frames, &g)
    }

    /// Runs the encoder + stochastic duration predictor and returns the raw
    /// (pre-ceil) durations `w = exp(logw)·length_scale` `[T_phonemes]`
    /// (component boundary for the M0-07-T14 parity test). Prosody features are
    /// zero (the mock-G2P / EN path), so `x_dp` is the encoder output padded
    /// with zero prosody channels.
    #[cfg(test)]
    pub(crate) fn durations(
        &self,
        phoneme_ids: &[i64],
        lid: i64,
        length_scale: f32,
    ) -> Result<Vec<f32>> {
        let g = self.conditioning.g(None, lid);
        let enc = self.encoder.forward(phoneme_ids, &g)?;
        let prosody = self.prosody_proj.channels(None, lid, enc.t);
        let x_dp = build_x_dp(&enc.x, &prosody, enc.t);
        // Deterministic (noise_w = 0) for parity (docs §5).
        let logw = self.duration.logw(&x_dp, enc.t, &g, 0.0);
        Ok(logw.iter().map(|&l| l.exp() * length_scale).collect())
    }

    /// The SDP body (proj output) for a phoneme sequence — component boundary
    /// used to isolate the duration-predictor body from its spline flows in the
    /// M0-07-T14 parity test.
    #[cfg(test)]
    pub(crate) fn sdp_body(&self, phoneme_ids: &[i64], lid: i64) -> Result<(Vec<f32>, usize)> {
        let g = self.conditioning.g(None, lid);
        let enc = self.encoder.forward(phoneme_ids, &g)?;
        let prosody = self.prosody_proj.channels(None, lid, enc.t);
        let x_dp = build_x_dp(&enc.x, &prosody, enc.t);
        Ok((self.duration.body(&x_dp, enc.t, &g), enc.t))
    }

    /// Expands `m_p` by `w_ceil` (length regulation, T15) and runs the reverse
    /// flow to the decoder-input latent `z` (component boundary for the
    /// M0-07-T17 parity test: reference `m_p` + durations → `z`).
    #[cfg(test)]
    pub(crate) fn expand_and_flow(
        &self,
        m_p: &[f32],
        t_phonemes: usize,
        w_ceil: &[usize],
        lid: i64,
    ) -> (Vec<f32>, usize) {
        let (z_p, t_frames) = length_regulate(m_p, HIDDEN, t_phonemes, w_ceil);
        let g = self.conditioning.g(None, lid);
        let z = self.flow.reverse(&z_p, t_frames, &g);
        (z, t_frames)
    }
}

impl TtsEngine for PiperPlusTts {
    /// Text → PCM via the placeholder [`tokenize`](Self::tokenize) then the
    /// native path (M0-07-T20). `request.deterministic` disables the VITS noise
    /// (parity mode); otherwise the voice's default scales apply. The language
    /// hint maps through the voice's language table (default: id 0).
    fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesizedAudio> {
        let lid = request
            .language
            .as_deref()
            .and_then(|c| self.config.language_id(c))
            .unwrap_or(0);
        let phoneme_ids = self.tokenize(&request.text);
        if phoneme_ids.is_empty() {
            return Err(vokra_core::VokraError::InvalidArgument(
                "piper TTS: text produced no in-vocabulary phonemes (placeholder tokenizer)"
                    .to_owned(),
            ));
        }
        let (noise, noise_w) = if request.deterministic {
            (0.0, 0.0)
        } else {
            (self.config.noise_scale, self.config.noise_w)
        };
        let prosody = request
            .prosody_features
            .as_deref()
            .map(|p| p.as_flattened());
        self.synthesize_phonemes(
            &phoneme_ids,
            lid,
            request.speaker_embedding.as_deref(),
            prosody,
            noise,
            self.config.length_scale,
            noise_w,
        )
    }
}

/// Builds the duration-predictor input `x_dp` `[HIDDEN + prosody, T]` by
/// concatenating (on the channel axis) the encoder output `[HIDDEN, T]` with the
/// prosody channels `[prosody, T]` from [`ProsodyProj::channels`]. Both are
/// channel-major, so the concatenation is a contiguous copy; for the v7 voice
/// the result is `[DP_FILTER, T]` = `[208, T]`.
fn build_x_dp(enc_x: &[f32], prosody: &[f32], t: usize) -> Vec<f32> {
    let hidden_len = HIDDEN * t;
    let mut x_dp = Vec::with_capacity(hidden_len + prosody.len());
    x_dp.extend_from_slice(&enc_x[..hidden_len]);
    x_dp.extend_from_slice(prosody);
    x_dp
}

/// Expands per-phoneme features to frame resolution by repeating each phoneme
/// column `w_ceil[j]` times (piper `commons.generate_path` — a monotonic,
/// search-free expansion). `features` is `[channels, t_phonemes]`; returns
/// `[channels, sum(w_ceil)]`.
fn length_regulate(
    features: &[f32],
    channels: usize,
    t_phonemes: usize,
    w_ceil: &[usize],
) -> (Vec<f32>, usize) {
    let t_frames: usize = w_ceil.iter().take(t_phonemes).sum();
    let mut out = vec![0.0f32; channels * t_frames];
    let mut tf = 0;
    for (j, &reps) in w_ceil.iter().take(t_phonemes).enumerate() {
        for _ in 0..reps {
            for c in 0..channels {
                out[c * t_frames + tf] = features[c * t_phonemes + j];
            }
            tf += 1;
        }
    }
    (out, t_frames)
}

#[cfg(test)]
mod tests {
    use super::config::{DP_FILTER, PROSODY_DIM};
    use super::{HIDDEN, ProsodyProj, build_x_dp, length_regulate};

    #[test]
    fn length_regulate_repeats_each_phoneme_column() {
        // channels=2, t_phonemes=3, channel-major [2,3]:
        //   ch0 = [1,2,3], ch1 = [4,5,6];  w_ceil = [2,1,3].
        let features = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (out, t_frames) = length_regulate(&features, 2, 3, &[2, 1, 3]);
        assert_eq!(t_frames, 6);
        // ch0 → [1,1,2,3,3,3], ch1 → [4,4,5,6,6,6] (channel-major).
        assert_eq!(
            out,
            [1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0]
        );
    }

    #[test]
    fn length_regulate_ignores_w_ceil_past_t_phonemes() {
        // A trailing 99 beyond t_phonemes=3 must be dropped (take(t_phonemes)).
        let features = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (out, t_frames) = length_regulate(&features, 2, 3, &[2, 1, 3, 99]);
        assert_eq!(t_frames, 6);
        assert_eq!(
            out,
            [1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0]
        );
    }

    #[test]
    fn build_x_dp_concatenates_hidden_then_prosody_channels() {
        let t = 2;
        // enc_x is a ramp over HIDDEN·t, laid out channel-major.
        let enc_x: Vec<f32> = (0..HIDDEN * t).map(|i| i as f32).collect();
        // Distinct value per (prosody channel, time) so a mis-index is caught.
        let prosody: Vec<f32> = (0..PROSODY_DIM * t).map(|i| 100.0 + i as f32).collect();
        let out = build_x_dp(&enc_x, &prosody, t);

        assert_eq!(out.len(), DP_FILTER * t);
        // First HIDDEN channels are the encoder output verbatim, the remaining
        // PROSODY_DIM channels the prosody block verbatim (channel-major concat).
        assert_eq!(&out[..HIDDEN * t], &enc_x[..]);
        assert_eq!(&out[HIDDEN * t..], &prosody[..]);
    }

    #[test]
    fn prosody_channels_bias_gate_and_matmul() {
        // in_dim=2, out_dim=3, t=2. weight is [in, out] row-major (raw MatMul):
        // W = [[1,2,3],[10,20,30]]; bias = [100,200,300].
        let proj = ProsodyProj {
            weight: vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            bias: vec![100.0, 200.0, 300.0],
            in_dim: 2,
            out_dim: 3,
        };
        // No features → bias broadcast over time, channel-major [out=3, t=2].
        let bias_only = proj.channels(None, 0, 2);
        assert_eq!(bias_only, [100.0, 100.0, 200.0, 200.0, 300.0, 300.0]);
        // Features present but a non-JA language (gate off) → still bias only.
        let feats = [1i64, 0, 0, 1]; // [t=2, in=2]: ti0=(1,0), ti1=(0,1)
        assert_eq!(proj.channels(Some(&feats), 1, 2), bias_only);
        // JA (lid == PROSODY_LANG_ID) with features → bias + features @ W:
        //   ti0 (1,0): (1,2,3);  ti1 (0,1): (10,20,30).
        // channel-major: o0=[101,110], o1=[202,220], o2=[303,330].
        let on = proj.channels(Some(&feats), super::config::PROSODY_LANG_ID, 2);
        assert_eq!(on, [101.0, 110.0, 202.0, 220.0, 303.0, 330.0]);
    }
}

/// Weight-license research-flag gate wiring (M2-13, FR-CP-03).
///
/// These prove the gate is live on the piper loader path. They use minimal
/// (weightless) piper-arch GGUFs: a gated model is rejected by the gate before
/// any weight is bound, and a permissive / research-unlocked model gets *past*
/// the gate (then fails later on the absent weights — a different error, which
/// is exactly what distinguishes "gate fired" from "gate passed").
#[cfg(test)]
mod compliance_gate_tests {
    use super::{EXPECTED_ARCH, PiperPlusTts};
    use vokra_core::gguf::{GgufBuilder, GgufFile, chunks};
    use vokra_core::{ComplianceLevel, CompliancePolicy, VokraError};

    /// A minimal piper-arch GGUF (no weight tensors), optionally carrying a
    /// `vokra.provenance.license` string.
    fn piper_gguf(license: Option<&str>) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        if let Some(l) = license {
            b.add_string(chunks::KEY_PROVENANCE_LICENSE, l);
        }
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    /// `PiperPlusTts` is not `Debug`, so unwrap the `Err` without touching `Ok`.
    fn load_err(res: Result<PiperPlusTts, VokraError>) -> VokraError {
        match res {
            Ok(_) => panic!("expected a load error (fixture carries no weights)"),
            Err(e) => e,
        }
    }

    #[test]
    fn noncommercial_voice_is_rejected_without_research_flag() {
        // WP core: a CC-BY-NC-tagged voice, strict policy -> explicit gate error
        // (before weight binding), never a silent load.
        let err = load_err(PiperPlusTts::from_gguf_with_policy(
            piper_gguf(Some("CC-BY-NC-4.0")),
            &CompliancePolicy::strict(),
        ));
        assert!(
            matches!(err, VokraError::ResearchLicenseRequired { .. }),
            "expected the gate to fire, got {err}"
        );
    }

    #[test]
    fn research_flag_unlocks_the_gate_on_this_path() {
        // Same tagged voice, but with a research flag: it clears the gate and
        // only then fails on the absent weights — so the error is NOT the
        // license gate. Proven for all three unlock routes.
        for policy in [
            CompliancePolicy::strict().with_research_license(true),
            CompliancePolicy::new(ComplianceLevel::Research),
            CompliancePolicy::new(ComplianceLevel::Disabled),
        ] {
            let err = load_err(PiperPlusTts::from_gguf_with_policy(
                piper_gguf(Some("CC-BY-NC-4.0")),
                &policy,
            ));
            assert!(
                !matches!(err, VokraError::ResearchLicenseRequired { .. }),
                "gate must be unlocked; got {err}"
            );
        }
    }

    #[test]
    fn permissive_voice_passes_the_gate_under_strict() {
        // A stock piper voice (no provenance -> registry: piper arch is
        // permissive) clears the gate under the strictest policy; it then fails
        // on the absent weights, NOT on the license gate.
        let err = load_err(PiperPlusTts::from_gguf_with_policy(
            piper_gguf(None),
            &CompliancePolicy::strict(),
        ));
        assert!(
            !matches!(err, VokraError::ResearchLicenseRequired { .. }),
            "permissive voice must pass the gate; got {err}"
        );
    }

    #[test]
    fn wrong_arch_still_fails_before_the_gate() {
        // The pre-existing arch check is unchanged: a non-piper GGUF is a plain
        // ModelLoad, not a license error.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "whisper");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let err = load_err(PiperPlusTts::from_gguf(file));
        assert!(matches!(err, VokraError::ModelLoad(_)), "got {err}");
    }
}
