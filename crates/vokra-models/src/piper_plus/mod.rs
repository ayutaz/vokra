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

mod config;
mod decoder;
mod duration;
mod flow;
mod nn;
mod text_encoder;
mod weights;

#[cfg(test)]
mod parity;

use std::path::Path;

use vokra_core::gguf::GgufFile;
use vokra_core::rng::GaussianSplitMix64;
use vokra_core::{Result, SynthesisRequest, SynthesizedAudio, TtsEngine};
use vokra_piper_plus::{PhonemeTable, Phonemizer};

pub use config::PiperConfig;
// Re-export the G2P reuse-boundary types so downstreams can build the injected
// phonemizer without also depending on `vokra-piper-plus` directly (M1-01-A).
pub use vokra_piper_plus::{MockPhonemizer, PassthroughPhonemizer};

use config::{DP_FILTER, HIDDEN};

/// `vokra.model.arch` a piper-plus voice GGUF must carry (written by
/// `vokra-convert`'s `models::piper_plus::ARCH`; kept in sync by that
/// converter, M0-07-T06). A wrong arch fails loudly at load (M1-01-C).
const EXPECTED_ARCH: &str = "piper-plus-mb-istft-vits2";

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
    encoder: TextEncoder,
    duration: DurationPredictor,
    flow: Flow,
    decoder: Decoder,
    /// `prosody_proj.bias` `[PROSODY_DIM]`. With zero prosody features
    /// (mock-G2P / EN path) `prosody_proj(0) = bias`, so these are the prosody
    /// channels appended to the encoder output for the duration predictor.
    /// Real A1/A2/A3 prosody (JA) needs the G2P bridge (T09) and the
    /// `prosody_proj` weight — a followup.
    prosody_bias: Vec<f32>,
}

impl PiperPlusTts {
    /// Loads a voice from a GGUF file on disk.
    ///
    /// # Errors
    ///
    /// Propagates GGUF parse errors and any weight shape/metadata mismatch
    /// (a wrong or corrupt voice fails loudly at load).
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let file = GgufFile::open(path)
            .map_err(|e| vokra_core::VokraError::ModelLoad(format!("piper voice GGUF: {e}")))?;
        Self::from_gguf(file)
    }

    /// Loads a voice from an already-parsed GGUF.
    ///
    /// The GGUF's `vokra.model.arch` is checked first, so a non-piper (or wrong
    /// architecture) GGUF fails with a clear [`VokraError::ModelLoad`] rather
    /// than a confusing missing-tensor error deep in a component loader
    /// (M1-01-C). The retained GGUF backing bytes (~77 MB FP32) are dropped once
    /// every component has copied its tensors out, halving resident memory: the
    /// `TensorStore` is a function local and is freed at the end of load.
    pub fn from_gguf(file: GgufFile) -> Result<Self> {
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
        let config = PiperConfig::from_gguf(store.file())?;
        let encoder = TextEncoder::load(&store, config.num_symbols, config.num_languages)?;
        let duration = DurationPredictor::load(&store)?;
        let prosody_bias = store.tensor_shaped("prosody_proj.bias", &[config::PROSODY_DIM])?;
        let flow = Flow::load(&store)?;
        let decoder = Decoder::load(
            &store,
            config.istft_n_fft,
            config.istft_hop,
            config.pqmf_subbands,
        )?;
        // `store` (and its GGUF backing bytes) drops here — every component owns
        // its own copies now, so nothing borrows it past load.
        Ok(Self {
            config,
            encoder,
            duration,
            prosody_bias,
            flow,
            decoder,
        })
    }

    /// The resolved voice configuration (sample rate, tables, scales, ...).
    pub fn config(&self) -> &PiperConfig {
        &self.config
    }

    /// Synthesizes PCM from a phoneme id sequence — the full native
    /// MB-iSTFT-VITS2 path (encoder → duration predictor → length regulation →
    /// flow → decoder), M0-07-T20.
    ///
    /// `noise_scale` / `noise_w` are the VITS stochastic knobs; passing `0` for
    /// both makes the whole path deterministic (the parity setting, docs §5).
    /// Non-zero scales draw Gaussian noise from a fixed-seed RNG (reproducible,
    /// but not bit-matched to onnxruntime — that path is exercised only for
    /// audio, not parity).
    pub fn synthesize_phonemes(
        &self,
        phoneme_ids: &[i64],
        lid: i64,
        noise_scale: f32,
        length_scale: f32,
        noise_w: f32,
    ) -> Result<SynthesizedAudio> {
        self.check_ids(phoneme_ids, lid)?;
        let enc = self.encoder.forward(phoneme_ids, lid)?;
        let g = self.encoder.lang_conditioning(lid);
        let x_dp = build_x_dp(&enc.x, enc.t, &self.prosody_bias);
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
        self.synthesize_phonemes(&phoneme_ids, lid, noise, self.config.length_scale, noise_w)
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
        self.encoder.forward(phoneme_ids, lid)
    }

    /// Runs the MB-iSTFT decoder on a decoder-input latent `z` `[HIDDEN, T]`
    /// under language `lid` (component boundary used by the M0-07-T19 parity
    /// test: reference latent → PCM).
    #[cfg(test)]
    pub(crate) fn decode(&self, z: &[f32], t_frames: usize, lid: i64) -> Result<Vec<f32>> {
        let g = self.encoder.lang_conditioning(lid);
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
        let enc = self.encoder.forward(phoneme_ids, lid)?;
        let g = self.encoder.lang_conditioning(lid);
        let x_dp = build_x_dp(&enc.x, enc.t, &self.prosody_bias);
        // Deterministic (noise_w = 0) for parity (docs §5).
        let logw = self.duration.logw(&x_dp, enc.t, &g, 0.0);
        Ok(logw.iter().map(|&l| l.exp() * length_scale).collect())
    }

    /// The SDP body (proj output) for a phoneme sequence — component boundary
    /// used to isolate the duration-predictor body from its spline flows in the
    /// M0-07-T14 parity test.
    #[cfg(test)]
    pub(crate) fn sdp_body(&self, phoneme_ids: &[i64], lid: i64) -> Result<(Vec<f32>, usize)> {
        let enc = self.encoder.forward(phoneme_ids, lid)?;
        let g = self.encoder.lang_conditioning(lid);
        let x_dp = build_x_dp(&enc.x, enc.t, &self.prosody_bias);
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
        let g = self.encoder.lang_conditioning(lid);
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
        self.synthesize_phonemes(&phoneme_ids, lid, noise, self.config.length_scale, noise_w)
    }
}

/// Builds the duration-predictor input `x_dp` `[DP_FILTER, T]` from the encoder
/// output `[HIDDEN, T]` by appending the prosody channels. With zero prosody
/// features `prosody_proj(0) = prosody_proj.bias`, so each appended channel is
/// the constant bias broadcast over time.
fn build_x_dp(x: &[f32], t: usize, prosody_bias: &[f32]) -> Vec<f32> {
    let mut x_dp = vec![0.0f32; DP_FILTER * t];
    x_dp[..HIDDEN * t].copy_from_slice(&x[..HIDDEN * t]);
    for (c, &b) in prosody_bias.iter().enumerate() {
        for ti in 0..t {
            x_dp[(HIDDEN + c) * t + ti] = b;
        }
    }
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
    use super::config::PROSODY_DIM;
    use super::{DP_FILTER, HIDDEN, build_x_dp, length_regulate};

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
    fn build_x_dp_copies_hidden_then_broadcasts_prosody_bias() {
        let t = 2;
        // x is a ramp over HIDDEN·t, laid out channel-major.
        let x: Vec<f32> = (0..HIDDEN * t).map(|i| i as f32).collect();
        // Distinct bias per prosody channel so a mis-index is caught.
        let bias: Vec<f32> = (0..PROSODY_DIM).map(|k| 100.0 + k as f32).collect();
        let out = build_x_dp(&x, t, &bias);

        assert_eq!(out.len(), DP_FILTER * t);
        // First HIDDEN channels are the encoder output verbatim.
        for c in 0..HIDDEN {
            for ti in 0..t {
                assert_eq!(out[c * t + ti], x[c * t + ti]);
            }
        }
        // Prosody channels HIDDEN..DP_FILTER hold the constant bias over time.
        for k in 0..PROSODY_DIM {
            for ti in 0..t {
                assert_eq!(out[(HIDDEN + k) * t + ti], bias[k]);
            }
        }
    }
}
