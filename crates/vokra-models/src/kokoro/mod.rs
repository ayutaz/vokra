//! Kokoro-82M native TTS (M2-07) — module skeleton.
//!
//! Native re-implementation of the Kokoro-82M inference core (StyleTTS 2 派生
//! iSTFTNet) in the whisper.cpp style: the model definition lives in Rust,
//! only the upstream **checkpoint** (Apache 2.0, converted offline to GGUF by
//! `vokra-convert`) is consumed at runtime. No ONNX runs at runtime
//! (FR-LD-05). G2P (misaki) is out of scope for M2-07; the runtime consumes
//! phoneme ids only (see [`docs/adr/0007-kokoro-native.md`]).
//!
//! # Layout (M2-07)
//!
//! - [`config`] — `vokra.kokoro.*` metadata + shape-cross-checked dims
//!   (T09);
//! - [`weights`] — F32-only [`GgufFile`](vokra_core::gguf::GgufFile) tensor
//!   store, rejecting a non-F32 tensor as a converter bug (T10);
//! - [`nn`] — 1-D dilated / grouped / transposed convolutions, activations
//!   plus a private [`nn::adain`] helper (StyleTTS 2 AdaIN as a composition
//!   of instance-norm + affine, **not** a new first-class op — FR-EX-08
//!   permits composition);
//! - [`text_encoder`] / [`bert`] / [`prosody`] / [`decoder`] — component
//!   skeletons; the concrete forward paths land at T12–T17. `bert` is the
//!   T13-beta PL-BERT branch (`bert.module.*` + `bert_encoder.module.*`,
//!   loaded only when the canary tensor is present — see
//!   [`BERT_CANARY_TENSOR`]). The iSTFT head uses FR-OP-01 `istft`, **not**
//!   the FR-OP-12 `vocos_head` — Kokoro is iSTFTNet 系.
//!
//! # Hot ops (M2-08 alignment)
//!
//! Kokoro dispatches **GEMM only** through the [`Compute`](crate::compute::Compute)
//! seam (every conv routes through [`nn::conv1d`]'s im2col + GEMM); the
//! LeakyReLU / GELU / sigmoid / AdaIN / iSTFT / voicepack lookup glue is
//! model-internal scalar work. Kokoro is **not** a FR-OP-12 `vocos_head`
//! consumer, so it does not opt in to any `vocos_head` FP16-forbidden
//! registry entry in M2-08 (`docs/adr/0007-kokoro-native.md` §Op gap).

mod bert;
mod config;
mod decoder;
mod nn;
mod prosody;
mod text_encoder;
mod weights;

use std::path::Path;

use vokra_core::gguf::GgufFile;
use vokra_core::{
    BackendKind, CompliancePolicy, Result, SynthesisRequest, SynthesizedAudio, TtsEngine,
    VokraError, check_weight_license,
};

use crate::compute::HotOp;

pub use config::KokoroConfig;

use bert::Bert;
use config::Dims;
use decoder::Decoder;
use prosody::ProsodyPredictor;
use text_encoder::TextEncoder;
use weights::TensorStore;

/// Canary tensor whose presence marks a GGUF as carrying the upstream Kokoro-82M
/// PL-BERT branch. Absent on slim fixture voices; when absent the runtime
/// bypasses [`Bert`] and falls back to the [`TextEncoder`] features as the
/// prosody-predictor input (documented at the wire-up call site).
const BERT_CANARY_TENSOR: &str = "bert.module.embeddings.word_embeddings.weight";

/// The backend hot ops the Kokoro-82M native TTS dispatches: **GEMM only**
/// (same rationale as [`crate::piper_plus`]).
#[allow(dead_code)] // consumed by the T18 e2e wire-up (`Compute::for_backend`)
pub(crate) const KOKORO_HOT_OPS: &[HotOp] = &[HotOp::Gemm];

/// `vokra.model.arch` a Kokoro-82M voice GGUF must carry. Written by
/// `vokra-convert::models::kokoro::ARCH`; kept in sync by that converter.
const EXPECTED_ARCH: &str = "kokoro-82m-istftnet";

/// A loaded Kokoro-82M voice.
///
/// Built from a voice GGUF (produced offline by `vokra-convert`, T07); no ONNX
/// is touched at runtime (FR-LD-05). The iSTFTNet inference core is assembled
/// here from the loaded weights (T09/T10 skeleton, T12–T17 wire-up).
pub struct KokoroTts {
    config: KokoroConfig,
    #[allow(dead_code)] // consumed by the T12–T17 forward path
    dims: Dims,
    text_encoder: TextEncoder,
    /// Upstream Kokoro-82M PL-BERT branch (`bert.module.*` +
    /// `bert_encoder.module.*`, 178 → 128 → 4× ALBERT → 768 → 512). When present
    /// its `[t, 512]` output replaces the [`TextEncoder`] output as the prosody
    /// predictor's input, matching the upstream Kokoro pipeline. Absent on slim
    /// fixture voices (fall-through to text-encoder features documented at the
    /// call site); dispatched by [`BERT_CANARY_TENSOR`] at load time.
    bert: Option<Bert>,
    prosody: ProsodyPredictor,
    decoder: Decoder,
    /// Backend selector (`Copy`; never a live `!Send` backend, same rationale
    /// as [`crate::piper_plus::PiperPlusTts`]).
    #[allow(dead_code)] // consumed by the T18 e2e wire-up
    backend_kind: BackendKind,
}

impl KokoroTts {
    /// Loads a voice from a GGUF file on disk.
    ///
    /// # Errors
    ///
    /// Propagates GGUF parse errors and any metadata / shape mismatch.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_path_with_policy(path, &CompliancePolicy::strict())
    }

    /// Loads a voice from disk under an explicit compliance `policy`.
    pub fn from_path_with_policy(
        path: impl AsRef<Path>,
        policy: &CompliancePolicy,
    ) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).map_err(VokraError::Io)?;
        Self::from_gguf_with_policy(&bytes, policy)
    }

    /// Loads a voice from raw GGUF bytes under an explicit compliance
    /// `policy`.
    ///
    /// The `vokra.model.arch` is checked first, so a non-Kokoro (or wrong
    /// architecture) GGUF fails with a clear [`VokraError::ModelLoad`] rather
    /// than a confusing missing-tensor error deep in a component loader.
    /// Then the shared **weight-license gate**
    /// ([`check_weight_license`], FR-CP-03) runs on the container *before*
    /// any weight tensor is bound — a non-commercial / unknown weight license
    /// (`vokra.provenance.*`) without a research flag is refused with
    /// [`VokraError::ResearchLicenseRequired`], not a silent load.
    ///
    /// Kokoro-82M is Apache 2.0 code + weight, so a stock (unlabelled)
    /// Kokoro voice classifies permissive (built-in registry, arch
    /// `kokoro-82m-istftnet`) and passes.
    pub fn from_gguf_with_policy(bytes: &[u8], policy: &CompliancePolicy) -> Result<Self> {
        let file = GgufFile::parse(bytes.to_vec())
            .map_err(|e| VokraError::ModelLoad(format!("kokoro voice GGUF: {e}")))?;
        let store = TensorStore::new(file);
        let arch = store
            .file()
            .get(vokra_core::gguf::chunks::KEY_MODEL_ARCH)
            .and_then(|v| v.as_str());
        if arch != Some(EXPECTED_ARCH) {
            return Err(VokraError::ModelLoad(format!(
                "not a Kokoro voice GGUF: vokra.model.arch = {arch:?}, expected `{EXPECTED_ARCH}`"
            )));
        }
        // Weight-license research-flag gate (FR-CP-03).
        check_weight_license(store.file(), policy)?;
        let config = KokoroConfig::from_gguf(store.file())?;
        let dims = Dims::derive(&store, &config)?;
        let text_encoder = TextEncoder::load(&store, &config)?;
        // Upstream Kokoro-82M carries a PL-BERT branch (`bert.module.*` +
        // `bert_encoder.module.*`); a slim fixture voice may omit it. The
        // canary tensor decides the dispatch — if absent the runtime uses the
        // text-encoder features as the prosody-predictor input (the T13-beta
        // seam documented in `docs/adr/0007-kokoro-native.md`). The load itself
        // is strict when the canary IS present: a partial bert set fails
        // loudly at [`Bert::new`] rather than silently falling back
        // (FR-EX-08).
        let bert = if store.shape(BERT_CANARY_TENSOR).is_ok() {
            Some(Bert::new(&store, &config)?)
        } else {
            None
        };
        let prosody = ProsodyPredictor::load(&store, &config)?;
        let decoder = Decoder::load(&store, &config)?;
        // `store` (and its GGUF backing bytes) drops here.
        Ok(Self {
            config,
            dims,
            text_encoder,
            bert,
            prosody,
            decoder,
            backend_kind: BackendKind::Cpu,
        })
    }

    /// The resolved voice configuration (sample rate, tables, iSTFT sizes, …).
    pub fn config(&self) -> &KokoroConfig {
        &self.config
    }

    /// Selects the backend the synthesis hot path runs on (default
    /// [`BackendKind::Cpu`]; wired at T18).
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend_kind = backend;
        self
    }

    /// Synthesizes PCM from a phoneme id sequence — the M2-07-T18 low-level
    /// native path, mirroring [`crate::piper_plus::PiperPlusTts::synthesize_phonemes`].
    ///
    /// The pipeline is
    /// `text_encoder → [bert →] prosody → length_regulate → decoder → PCM`.
    /// When the optional PL-BERT branch is loaded (upstream Kokoro-82M carries
    /// it), its `[t, 512]` output replaces the text-encoder output as the
    /// prosody-predictor input (the T13-beta seam documented in
    /// `docs/adr/0007-kokoro-native.md`). The chosen features are transposed
    /// from `[t, hidden]` row-major to `[hidden, t]` channel-major (the layout
    /// every downstream stage consumes; the layout mismatch is pinned at the
    /// module boundary here, not silently inside a component).
    ///
    /// # Style resolution
    ///
    /// Exactly one of the two style sources must be present; both absent is a
    /// loud [`VokraError::InvalidArgument`] rather than a silent zero-style
    /// default (FR-EX-08):
    ///
    /// - `style_override = Some(vec)` — the caller-supplied style vector wins;
    ///   `vec.len()` must equal `config.style_dim`.
    /// - `voice = Some(name)` — the name is looked up in the voice table
    ///   ([`KokoroConfig::voice_names`]); an unknown name is a loud
    ///   [`VokraError::InvalidArgument`]. Because the voicepack tensor schema
    ///   is TBD until M2-07-T02 upstream inspection
    ///   (`docs/adr/0007-kokoro-native.md` §Voicepack), a **known** name
    ///   returns [`VokraError::NotImplemented`] — never a silent zero-style
    ///   fallback.
    ///
    /// `style_override` takes precedence over `voice` when both are set.
    ///
    /// # Scales
    ///
    /// - `noise_scale` is reserved for the Kokoro stochastic prosody path
    ///   (M2-07-T13 follow-on); the current deterministic scaffold ignores it
    ///   rather than pretending to apply a noise it has not yet wired.
    /// - `length_scale` scales each per-phoneme duration before frame
    ///   expansion (`w = max(1, ceil(exp(log_dur) · length_scale))`), matching
    ///   the piper convention. `1.0` disables scaling.
    ///
    /// # Errors
    ///
    /// Any component error propagates verbatim (all typed): out-of-range
    /// phoneme id from the text encoder, shape mismatch inside prosody /
    /// decoder, or the two style-resolution errors above.
    pub fn synthesize_phonemes(
        &self,
        phoneme_ids: &[i64],
        voice: Option<&str>,
        style_override: Option<&[f32]>,
        noise_scale: f32,
        length_scale: f32,
    ) -> Result<SynthesizedAudio> {
        // Reserved for the stochastic prosody path (M2-07-T13 follow-on).
        // Consumed here so the parameter is not silently dropped.
        let _ = noise_scale;

        // 1) Resolve the style vector (FR-EX-08: never a silent zero default).
        let style: Vec<f32> = if let Some(s) = style_override {
            if s.len() != self.config.style_dim {
                return Err(VokraError::InvalidArgument(format!(
                    "kokoro TTS: style_override len {} != style_dim ({})",
                    s.len(),
                    self.config.style_dim,
                )));
            }
            s.to_vec()
        } else if let Some(name) = voice {
            let _voice_id = self.config.voice_id(name).ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "kokoro TTS: unknown voice `{name}` (voice_names = {:?})",
                    self.config.voice_names,
                ))
            })?;
            return Err(VokraError::NotImplemented(
                "kokoro voicepack style lookup TBD at M2-07-T02 (pass style_override in the meantime)",
            ));
        } else {
            return Err(VokraError::InvalidArgument(
                "kokoro TTS: no style — pass style_override or a voice name".to_owned(),
            ));
        };

        // 2) Text encoder → [t, hidden_dim] row-major.
        let enc_arr = self.text_encoder.forward(phoneme_ids)?;
        let t_in = enc_arr.rows;
        let hidden = enc_arr.cols;
        if hidden != self.config.hidden_dim {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro TTS: text encoder produced cols {} != config.hidden_dim ({})",
                hidden, self.config.hidden_dim,
            )));
        }

        // 3) Prosody-input features. Upstream Kokoro feeds the PL-BERT branch's
        //    `[t, 512]` output (not the text encoder's) to `predictor.text_encoder`;
        //    the T13-beta seam (`docs/adr/0007-kokoro-native.md`) makes bert the
        //    prosody source when the branch is present, and falls back to the
        //    text-encoder output otherwise. Both sources produce `[t, hidden_dim]`
        //    row-major features; a bert-vs-hidden width mismatch is a loud error
        //    rather than a silent fallback (FR-EX-08).
        let features_row: Vec<f32> = if let Some(bert) = &self.bert {
            let bert_out = bert.forward(phoneme_ids)?;
            let bert_cols = bert_out.len() / t_in;
            if bert_cols != hidden {
                return Err(VokraError::InvalidArgument(format!(
                    "kokoro TTS: bert output width {} != hidden_dim ({}); \
                     the bert branch expects a Kokoro-82M-shaped voice \
                     (hidden_dim = 512)",
                    bert_cols, hidden,
                )));
            }
            bert_out
        } else {
            enc_arr.data.clone()
        };

        // 4) Transpose to [hidden_dim, T] channel-major (the layout prosody /
        //    length regulation / decoder consume — piper's convention).
        let mut encoded_ch = vec![0.0f32; hidden * t_in];
        for ti in 0..t_in {
            for c in 0..hidden {
                encoded_ch[c * t_in + ti] = features_row[ti * hidden + c];
            }
        }

        // 5) Prosody predictor → (log_dur, f0, energy) each [T] via the
        //    backward-compat adapter. The upstream forward
        //    ([`ProsodyPredictor::forward_upstream`]) is used by the T17 parity
        //    landing; the adapter is called here so the wiring stays stable
        //    across the phase-3 → parity boundary. `deterministic = true`: the
        //    stochastic path is deferred and returns NotImplemented rather
        //    than being silently skipped.
        let (log_dur, _f0, _energy) =
            self.prosody
                .forward(&encoded_ch, &style, t_in, /*deterministic=*/ true)?;

        // 6) Length regulation: `w = max(1, ceil(exp(log_dur) · length_scale))`
        //    (piper convention). Values are clamped to `[1, 1024]` per phoneme
        //    to keep a degenerate scaffold-time `log_dur` from allocating an
        //    unbounded frame buffer via a `+inf as usize` saturation.
        let durations: Vec<usize> = log_dur
            .iter()
            .map(|&l| {
                let v = (l.exp() * length_scale).ceil();
                if v.is_finite() && v >= 1.0 {
                    (v as usize).min(1024)
                } else {
                    1
                }
            })
            .collect();
        let (z, t_frames) = nn::length_regulate(&encoded_ch, hidden, t_in, &durations);

        // 7) Decoder → PCM at `config.sample_rate`. [`Decoder::forward`]
        //    dispatches to stub / real mode internally based on the canary
        //    tensor seen at load time; real-mode currently feeds zero F0/N
        //    contours (M2-07-T17 landing wires the real prosody contours via
        //    [`Decoder::forward_full`]).
        let pcm = self.decoder.forward(&z, t_frames, &style)?;

        Ok(SynthesizedAudio::new(pcm, self.config.sample_rate))
    }

    /// Runs the internal text encoder forward for one phoneme id sequence and
    /// returns its `[t · hidden_dim]` row-major output. Test-only bridge for
    /// the M2-07-T17 per-module parity harness
    /// (`crates/vokra-models/tests/parity_kokoro.rs::text_encoder_forward_bit_parity`);
    /// hidden behind a `#[doc(hidden)]` so it stays out of the public API.
    ///
    /// The layout matches the T17 dumper's `text_encoder.f32` fixture: the
    /// first `enc_pos · hidden_dim` floats of the returned `Vec` are compared
    /// byte-for-byte against the reference at `atol = 0.01`.
    #[doc(hidden)]
    pub fn text_encoder_forward_for_parity(&self, phoneme_ids: &[i64]) -> Result<Vec<f32>> {
        let arr = self.text_encoder.forward(phoneme_ids)?;
        // The text encoder returns an internal `Array2<f32>` (row-major
        // `[t, hidden_dim]`); expose the raw `data` so the parity harness can
        // slice `[..enc_pos * hidden_dim]` without a shape-conversion loop.
        Ok(arr.data)
    }

    /// Runs the internal PL-BERT forward for one phoneme id sequence and
    /// returns its `[t · 512]` row-major output. Test-only bridge for the
    /// M2-07-T17 per-module parity harness. When the voice GGUF does not
    /// carry the PL-BERT branch (canary tensor
    /// [`BERT_CANARY_TENSOR`] absent), returns a loud
    /// [`VokraError::InvalidArgument`] naming the missing branch rather than
    /// a silent zero-shaped result (FR-EX-08).
    #[doc(hidden)]
    pub fn bert_forward_for_parity(&self, phoneme_ids: &[i64]) -> Result<Vec<f32>> {
        let Some(bert) = &self.bert else {
            return Err(VokraError::InvalidArgument(
                "kokoro TTS: bert branch absent — parity dump asserts bert_mode = full \
                 but the loaded voice GGUF has no `bert.module.*` tensors. Rebuild the \
                 GGUF from the upstream Kokoro-82M checkpoint (the canary tensor \
                 `bert.module.embeddings.word_embeddings.weight` must be present)."
                    .to_owned(),
            ));
        };
        bert.forward(phoneme_ids)
    }

    /// Runs the internal prosody predictor forward via the T14
    /// [`ProsodyPredictor::forward_upstream`] path (bypassing the
    /// per-phoneme downgrade adapter used by `synthesize_phonemes`).
    /// Test-only bridge for the M2-07-T17 per-module parity harness.
    ///
    /// Pipeline mirrors [`Self::synthesize_phonemes`] up to and including the
    /// prosody call:
    /// 1. `text_encoder.forward(phoneme_ids)` → `[t, hidden_dim]` row-major.
    /// 2. If `bert` present, `bert.forward(phoneme_ids)` overrides the
    ///    features; else falls through to the text-encoder output.
    /// 3. Transpose to `[hidden_dim, t]` channel-major (the layout prosody
    ///    consumes).
    /// 4. Call [`ProsodyPredictor::forward_upstream`] with the caller-supplied
    ///    `style` vector (validated against `config.style_dim`).
    ///
    /// Returns a tuple `(durations, f0, n, hidden, t_frames)`:
    /// * `durations` — per-phoneme integer duration counts as `Vec<i64>`
    ///   (converted from the internal `Vec<usize>` so callers can dump as
    ///   little-endian i64 without further conversion).
    /// * `f0` — F0 contour at 2·T_frames resolution.
    /// * `n` — N (energy) contour at 2·T_frames resolution.
    /// * `hidden` — `[d_model, T_frames]` channel-major frame-rate features
    ///   from `predictor.shared`.
    /// * `t_frames` — `sum(durations)`, so the caller can validate lengths.
    ///
    /// # Errors
    ///
    /// * `style` length mismatch vs `config.style_dim` — a loud
    ///   [`VokraError::InvalidArgument`] rather than a silent zero-pad.
    /// * Any component error propagates verbatim (text encoder / bert /
    ///   prosody shape mismatches).
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn prosody_forward_for_parity(
        &self,
        phoneme_ids: &[i64],
        style: &[f32],
    ) -> Result<(Vec<i64>, Vec<f32>, Vec<f32>, Vec<f32>, usize)> {
        if style.len() != self.config.style_dim {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro TTS: prosody parity style len {} != style_dim ({})",
                style.len(),
                self.config.style_dim,
            )));
        }
        let enc_arr = self.text_encoder.forward(phoneme_ids)?;
        let t_in = enc_arr.rows;
        let hidden = enc_arr.cols;
        if hidden != self.config.hidden_dim {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro TTS: text encoder produced cols {} != config.hidden_dim ({})",
                hidden, self.config.hidden_dim,
            )));
        }
        let features_row: Vec<f32> = if let Some(bert) = &self.bert {
            let bert_out = bert.forward(phoneme_ids)?;
            let bert_cols = bert_out.len() / t_in;
            if bert_cols != hidden {
                return Err(VokraError::InvalidArgument(format!(
                    "kokoro TTS: bert output width {} != hidden_dim ({})",
                    bert_cols, hidden,
                )));
            }
            bert_out
        } else {
            enc_arr.data.clone()
        };
        let mut encoded_ch = vec![0.0f32; hidden * t_in];
        for ti in 0..t_in {
            for c in 0..hidden {
                encoded_ch[c * t_in + ti] = features_row[ti * hidden + c];
            }
        }
        let out = self.prosody.forward_upstream(&encoded_ch, style, t_in)?;
        let t_frames: usize = out.durations.iter().sum();
        let durations_i64: Vec<i64> = out.durations.iter().map(|&d| d as i64).collect();
        Ok((durations_i64, out.f0, out.n, out.hidden, t_frames))
    }

    /// Runs the internal decoder forward for one phoneme id sequence and
    /// returns the pre-iSTFT `(x_mag, x_phase, pcm)` triple. Test-only bridge
    /// for the M2-07-T15 decoder parity harness
    /// (`crates/vokra-models/tests/parity_kokoro.rs::decoder_forward_bit_parity`).
    ///
    /// Pipeline mirrors [`Self::synthesize_phonemes`] up to the decoder call:
    /// 1. `text_encoder.forward(phoneme_ids)` → `[t, hidden_dim]` row-major.
    /// 2. If `bert` present, override with `bert.forward(phoneme_ids)`.
    /// 3. Transpose to `[hidden_dim, t_in]` channel-major.
    /// 4. Prosody predictor via `forward_upstream` for the durations (F0/N are
    ///    NOT fed downstream — mirrors the mainline `Decoder::forward` which
    ///    feeds zero F0 / N contours).
    /// 5. Length-regulate the encoder features → `[hidden, t_frames]`.
    /// 6. Call [`Decoder::forward_full_intermediate`] with zero F0 / N and
    ///    `PhaseActivation::Sin` (matches the mainline path).
    ///
    /// The mag / phase tensors returned are `[n_half · t_gen]` channel-major
    /// (same layout as the reference dumper's
    /// `decoder_pre_istft_mag.f32` / `decoder_pre_istft_phase.f32`).
    ///
    /// # Errors
    ///
    /// * `style` length mismatch vs `config.style_dim` — a loud
    ///   [`VokraError::InvalidArgument`] rather than a silent zero-pad.
    /// * Stub-mode voice (no decoder tensors) — the intermediate accessor
    ///   requires real-mode weights and fails loudly.
    /// * Any component error propagates verbatim (text encoder / bert /
    ///   prosody / decoder shape mismatches).
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn decoder_forward_for_parity(
        &self,
        phoneme_ids: &[i64],
        style: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if style.len() != self.config.style_dim {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro TTS: decoder parity style len {} != style_dim ({})",
                style.len(),
                self.config.style_dim,
            )));
        }
        // 1. Text encoder → [t_in, hidden] row-major.
        let enc_arr = self.text_encoder.forward(phoneme_ids)?;
        let t_in = enc_arr.rows;
        let hidden = enc_arr.cols;
        if hidden != self.config.hidden_dim {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro TTS: text encoder cols {} != hidden_dim ({})",
                hidden, self.config.hidden_dim,
            )));
        }
        // 2. Prefer bert output as encoder features when the branch is present
        //    (matches `synthesize_phonemes` and the reference dumper).
        let features_row: Vec<f32> = if let Some(bert) = &self.bert {
            let bert_out = bert.forward(phoneme_ids)?;
            let bert_cols = bert_out.len() / t_in;
            if bert_cols != hidden {
                return Err(VokraError::InvalidArgument(format!(
                    "kokoro TTS: bert output width {} != hidden_dim ({})",
                    bert_cols, hidden,
                )));
            }
            bert_out
        } else {
            enc_arr.data.clone()
        };
        // 3. Transpose to channel-major [hidden, t_in].
        let mut encoded_ch = vec![0.0f32; hidden * t_in];
        for ti in 0..t_in {
            for c in 0..hidden {
                encoded_ch[c * t_in + ti] = features_row[ti * hidden + c];
            }
        }
        // 4. Prosody (upstream path) — we only need durations. F0 / N are not
        //    fed to the decoder in this parity harness (matches the reference
        //    dumper's ``_forward_decoder``: both feed zero f0 / n contours,
        //    validating decoder math rather than the yet-unwired F0 handling).
        let pros = self.prosody.forward_upstream(&encoded_ch, style, t_in)?;
        // 5. Length-regulate encoder features → [hidden, t_frames].
        let (z, t_frames) = nn::length_regulate(&encoded_ch, hidden, t_in, &pros.durations);
        if t_frames == 0 {
            return Err(VokraError::InvalidArgument(
                "kokoro TTS: decoder parity produced t_frames = 0".to_owned(),
            ));
        }
        // 6. Zero F0 / N (mirrors mainline `Decoder::forward`).
        let f0 = vec![0.0f32; t_frames];
        let n = vec![0.0f32; t_frames];
        // Dispatch through the intermediate accessor.
        self.decoder.forward_full_intermediate(
            &z,
            &f0,
            &n,
            style,
            t_frames,
            decoder::PhaseActivation::Sin,
        )
    }
}

impl TtsEngine for KokoroTts {
    /// Text → PCM adapter around [`KokoroTts::synthesize_phonemes`].
    ///
    /// Style resolution (documented / wired here for the future G2P bridge):
    ///
    /// - `request.speaker_embedding = Some(vec)` (matching `style_dim`) is
    ///   used verbatim as the style vector — the parity of the low-level
    ///   `synthesize_phonemes(style_override = …)` path.
    /// - Otherwise the voicepack table is queried (TBD at M2-07-T02 follow-on;
    ///   returns [`VokraError::NotImplemented`]).
    /// - Both absent — the voice has no named voicepack **and** no embedding is
    ///   supplied — is a loud [`VokraError::InvalidArgument`], never a silent
    ///   zero-style default (FR-EX-08).
    ///
    /// The text → phoneme_ids step requires G2P (misaki; eSpeak-NG fallback
    /// GPL-3.0 excluded), which is out of scope for M2-07 (see
    /// `docs/adr/0007-kokoro-native.md` §Design). Until a G2P bridge lands,
    /// callers exercise the native pipeline via
    /// [`KokoroTts::synthesize_phonemes`] directly with phoneme ids from a
    /// separate G2P integration.
    fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesizedAudio> {
        let has_style = request.speaker_embedding.is_some() || !self.config.voice_names.is_empty();
        if !has_style {
            return Err(VokraError::InvalidArgument(
                "kokoro TTS: no style — pass request.speaker_embedding or use a voice GGUF \
                 with a named voicepack"
                    .to_owned(),
            ));
        }
        // The text is not silently dropped — its consumer is the future G2P
        // bridge. Reference it so the intent is documented in-source.
        let _ = request.text.as_str();
        Err(VokraError::NotImplemented(
            "kokoro TtsEngine::synthesize needs a G2P bridge (out of scope M2-07); \
             use KokoroTts::synthesize_phonemes with phoneme ids from a separate G2P",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::config::{
        KEY_HIDDEN_DIM, KEY_ISTFT_HOP, KEY_ISTFT_N_FFT, KEY_ISTFT_WIN_LENGTH, KEY_N_DECODER_LAYERS,
        KEY_N_TEXT_LAYERS, KEY_NUM_VOICES, KEY_PHONEME_SYMBOLS, KEY_SAMPLE_RATE, KEY_STYLE_DIM,
        KEY_VOICE_NAMES,
    };
    use super::*;
    use vokra_core::gguf::{
        GgmlType, GgufArray, GgufBuilder, GgufFile, GgufMetadataValue, GgufValueType,
    };

    fn str_array(items: &[&str]) -> GgufMetadataValue {
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: items
                .iter()
                .map(|s| GgufMetadataValue::String((*s).to_owned()))
                .collect(),
        })
    }

    /// A builder carrying all 11 `vokra.kokoro.*` keys with distinct values so
    /// a field-swap regression is caught.
    fn valid_kokoro_builder() -> GgufBuilder {
        let mut b = GgufBuilder::new();
        b.add_u32(KEY_SAMPLE_RATE, 24_000);
        b.add_u32(KEY_STYLE_DIM, 256);
        b.add_u32(KEY_NUM_VOICES, 3);
        b.add_u32(KEY_HIDDEN_DIM, 512);
        b.add_u32(KEY_N_TEXT_LAYERS, 4);
        b.add_u32(KEY_N_DECODER_LAYERS, 6);
        b.add_u32(KEY_ISTFT_N_FFT, 20);
        b.add_u32(KEY_ISTFT_HOP, 5);
        b.add_u32(KEY_ISTFT_WIN_LENGTH, 20);
        b.add_metadata(KEY_PHONEME_SYMBOLS, str_array(&["_", "^", "$", "a"]));
        b.add_metadata(KEY_VOICE_NAMES, str_array(&["af", "am", "bf"]));
        b
    }

    #[test]
    fn config_from_gguf_reads_all_keys() {
        let file =
            GgufFile::parse(valid_kokoro_builder().to_bytes().expect("serialize")).expect("parse");
        let cfg = KokoroConfig::from_gguf(&file).expect("valid config");
        // Every field is verified against the distinct value written above so a
        // field-swap regression is caught.
        assert_eq!(cfg.sample_rate, 24_000);
        assert_eq!(cfg.style_dim, 256);
        assert_eq!(cfg.num_voices, 3);
        assert_eq!(cfg.hidden_dim, 512);
        assert_eq!(cfg.n_text_layers, 4);
        assert_eq!(cfg.n_decoder_layers, 6);
        assert_eq!(cfg.istft_n_fft, 20);
        assert_eq!(cfg.istft_hop, 5);
        assert_eq!(cfg.istft_win_length, 20);
        assert_eq!(cfg.phoneme_symbols, ["_", "^", "$", "a"]);
        assert_eq!(cfg.voice_names, ["af", "am", "bf"]);
        // Voice-id lookup: index in the name table; absent name = None.
        assert_eq!(cfg.voice_id("af"), Some(0));
        assert_eq!(cfg.voice_id("bf"), Some(2));
        assert_eq!(cfg.voice_id("zz"), None);
    }

    #[test]
    fn config_from_gguf_rejects_missing_style_dim() {
        // Every key except `style_dim` — the loader must refuse rather than
        // silently defaulting (FR-EX-08).
        let mut b = valid_kokoro_builder();
        // GgufBuilder does not expose a delete API, so rebuild without the
        // key: reconstruct the builder skipping `style_dim`.
        let mut without_style_dim = GgufBuilder::new();
        without_style_dim.add_u32(KEY_SAMPLE_RATE, 24_000);
        // `style_dim` deliberately omitted.
        without_style_dim.add_u32(KEY_NUM_VOICES, 3);
        without_style_dim.add_u32(KEY_HIDDEN_DIM, 512);
        without_style_dim.add_u32(KEY_N_TEXT_LAYERS, 4);
        without_style_dim.add_u32(KEY_N_DECODER_LAYERS, 6);
        without_style_dim.add_u32(KEY_ISTFT_N_FFT, 20);
        without_style_dim.add_u32(KEY_ISTFT_HOP, 5);
        without_style_dim.add_u32(KEY_ISTFT_WIN_LENGTH, 20);
        without_style_dim.add_metadata(KEY_PHONEME_SYMBOLS, str_array(&["_", "^", "$", "a"]));
        without_style_dim.add_metadata(KEY_VOICE_NAMES, str_array(&["af", "am", "bf"]));
        // `b` is left mutated but unused — silence dead_code.
        let _ = &mut b;

        let file =
            GgufFile::parse(without_style_dim.to_bytes().expect("serialize")).expect("parse");
        match KokoroConfig::from_gguf(&file) {
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(
                    msg.contains(KEY_STYLE_DIM),
                    "error should name the missing key; got: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// Text-encoder-only in-memory fixture (arch string + config metadata +
    /// text-encoder tensors, everything else absent). Used by the wiring tests
    /// below to reach `from_gguf_with_policy` past the arch / config gates and
    /// exercise the downstream (bert / prosody / decoder) loader chain. Kept
    /// distinct from the T13-alpha text-encoder synthetic fixture so its
    /// coverage remains focused on the wiring seam.
    fn text_encoder_only_bytes(hidden: usize, n_vocab: usize, style_dim: usize) -> Vec<u8> {
        assert_eq!(hidden % 2, 0, "text encoder hidden must be even");
        let lstm_hidden = hidden / 2;
        let four_h = 4 * lstm_hidden;
        let zeros = |n: usize| -> Vec<u8> { vec![0u8; n * 4] };
        let ones = |n: usize| -> Vec<u8> { (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect() };

        let mut b = GgufBuilder::new();
        b.add_string(vokra_core::gguf::chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        b.add_u32(KEY_SAMPLE_RATE, 24_000);
        b.add_u32(KEY_STYLE_DIM, style_dim as u32);
        b.add_u32(KEY_NUM_VOICES, 2);
        b.add_u32(KEY_HIDDEN_DIM, hidden as u32);
        b.add_u32(KEY_N_TEXT_LAYERS, 2);
        b.add_u32(KEY_N_DECODER_LAYERS, 2);
        b.add_u32(KEY_ISTFT_N_FFT, 20);
        b.add_u32(KEY_ISTFT_HOP, 5);
        b.add_u32(KEY_ISTFT_WIN_LENGTH, 20);
        let phoneme_symbols: Vec<String> = (0..n_vocab).map(|i| format!("p{i}")).collect();
        let phoneme_refs: Vec<&str> = phoneme_symbols.iter().map(String::as_str).collect();
        b.add_metadata(KEY_PHONEME_SYMBOLS, str_array(&phoneme_refs));
        b.add_metadata(KEY_VOICE_NAMES, str_array(&["af", "am"]));

        b.add_tensor(
            "text_encoder.module.embedding.weight",
            GgmlType::F32,
            vec![n_vocab as u64, hidden as u64],
            zeros(n_vocab * hidden),
        )
        .expect("emb");
        for i in 0..3usize {
            b.add_tensor(
                &format!("text_encoder.module.cnn.{i}.0.weight_g"),
                GgmlType::F32,
                vec![hidden as u64, 1, 1],
                zeros(hidden),
            )
            .expect("weight_g");
            b.add_tensor(
                &format!("text_encoder.module.cnn.{i}.0.weight_v"),
                GgmlType::F32,
                vec![hidden as u64, hidden as u64, 5],
                zeros(hidden * hidden * 5),
            )
            .expect("weight_v");
            b.add_tensor(
                &format!("text_encoder.module.cnn.{i}.0.bias"),
                GgmlType::F32,
                vec![hidden as u64],
                zeros(hidden),
            )
            .expect("cnn bias");
            b.add_tensor(
                &format!("text_encoder.module.cnn.{i}.1.gamma"),
                GgmlType::F32,
                vec![hidden as u64],
                ones(hidden),
            )
            .expect("gamma");
            b.add_tensor(
                &format!("text_encoder.module.cnn.{i}.1.beta"),
                GgmlType::F32,
                vec![hidden as u64],
                zeros(hidden),
            )
            .expect("beta");
        }
        for suffix in ["", "_reverse"] {
            b.add_tensor(
                &format!("text_encoder.module.lstm.weight_ih_l0{suffix}"),
                GgmlType::F32,
                vec![four_h as u64, hidden as u64],
                zeros(four_h * hidden),
            )
            .expect("lstm w_ih");
            b.add_tensor(
                &format!("text_encoder.module.lstm.weight_hh_l0{suffix}"),
                GgmlType::F32,
                vec![four_h as u64, lstm_hidden as u64],
                zeros(four_h * lstm_hidden),
            )
            .expect("lstm w_hh");
            b.add_tensor(
                &format!("text_encoder.module.lstm.bias_ih_l0{suffix}"),
                GgmlType::F32,
                vec![four_h as u64],
                zeros(four_h),
            )
            .expect("lstm b_ih");
            b.add_tensor(
                &format!("text_encoder.module.lstm.bias_hh_l0{suffix}"),
                GgmlType::F32,
                vec![four_h as u64],
                zeros(four_h),
            )
            .expect("lstm b_hh");
        }

        b.to_bytes().expect("serialize")
    }

    /// FR-EX-08 wiring check: with only the text-encoder tensors present,
    /// [`KokoroTts::from_gguf_with_policy`] must reach the prosody loader and
    /// fail LOUDLY at the first missing `predictor.module.*` tensor. A
    /// silent stub (the pre-T13/T14/T15 placeholder) would return `Ok`; the
    /// wiring test pins the fail-fast contract that the phase-3 rewrite
    /// requires. Also confirms the bert branch stays optional — its absence is
    /// NOT what surfaces (the bert canary is checked first).
    #[test]
    fn from_gguf_reaches_prosody_loader_and_fails_loudly_on_missing_tensor() {
        let bytes = text_encoder_only_bytes(16, 6, 8);
        // Use `match` on the Result rather than `expect_err` — `KokoroTts`
        // does not implement `Debug` (it owns non-Debug component buffers).
        match KokoroTts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict()) {
            Ok(_) => panic!("prosody tensors absent — loader must fail loudly"),
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(
                    msg.contains("predictor.module."),
                    "error must name the missing predictor tensor (FR-EX-08); got: {msg}"
                );
            }
            Err(other) => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// Bert is optional — a voice GGUF whose only bert tensor is the canary
    /// (missing the rest) fails LOUDLY at [`Bert::new`] rather than silently
    /// falling back to text-encoder features (FR-EX-08). This pins the "if
    /// the canary is present, load strictly" half of the two-armed dispatch.
    ///
    /// Builds on the same text-encoder-only fixture (so the text-encoder loader
    /// clears) plus a bare bert canary at the exact upstream shape `[178, 128]`
    /// but WITHOUT any of the other bert tensors. `Bert::new` then fails at the
    /// second lookup (`position_embeddings.weight`), naming the offending
    /// tensor — the wiring test asserts the error surfaces from the bert
    /// subtree, not from a downstream loader.
    #[test]
    fn from_gguf_rejects_partial_bert_branch() {
        // Start from the text-encoder-only bytes (so the text encoder clears
        // and the loader reaches the bert canary check), then rebuild carrying
        // an extra bert canary tensor via a fresh builder mirroring
        // [`text_encoder_only_bytes`]. Rebuilding is necessary because
        // `GgufBuilder` does not expose an append-to-existing-file API.
        let hidden: usize = 16;
        let n_vocab: usize = 6;
        let style_dim: usize = 8;
        let lstm_hidden = hidden / 2;
        let four_h = 4 * lstm_hidden;
        let zeros = |n: usize| -> Vec<u8> { vec![0u8; n * 4] };
        let ones = |n: usize| -> Vec<u8> { (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect() };

        let mut b = GgufBuilder::new();
        b.add_string(vokra_core::gguf::chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        b.add_u32(KEY_SAMPLE_RATE, 24_000);
        b.add_u32(KEY_STYLE_DIM, style_dim as u32);
        b.add_u32(KEY_NUM_VOICES, 2);
        b.add_u32(KEY_HIDDEN_DIM, hidden as u32);
        b.add_u32(KEY_N_TEXT_LAYERS, 2);
        b.add_u32(KEY_N_DECODER_LAYERS, 2);
        b.add_u32(KEY_ISTFT_N_FFT, 20);
        b.add_u32(KEY_ISTFT_HOP, 5);
        b.add_u32(KEY_ISTFT_WIN_LENGTH, 20);
        let phoneme_symbols: Vec<String> = (0..n_vocab).map(|i| format!("p{i}")).collect();
        let phoneme_refs: Vec<&str> = phoneme_symbols.iter().map(String::as_str).collect();
        b.add_metadata(KEY_PHONEME_SYMBOLS, str_array(&phoneme_refs));
        b.add_metadata(KEY_VOICE_NAMES, str_array(&["af", "am"]));
        // Text-encoder tensors (so the text encoder loader clears).
        b.add_tensor(
            "text_encoder.module.embedding.weight",
            GgmlType::F32,
            vec![n_vocab as u64, hidden as u64],
            zeros(n_vocab * hidden),
        )
        .expect("emb");
        for i in 0..3usize {
            b.add_tensor(
                &format!("text_encoder.module.cnn.{i}.0.weight_g"),
                GgmlType::F32,
                vec![hidden as u64, 1, 1],
                zeros(hidden),
            )
            .expect("weight_g");
            b.add_tensor(
                &format!("text_encoder.module.cnn.{i}.0.weight_v"),
                GgmlType::F32,
                vec![hidden as u64, hidden as u64, 5],
                zeros(hidden * hidden * 5),
            )
            .expect("weight_v");
            b.add_tensor(
                &format!("text_encoder.module.cnn.{i}.0.bias"),
                GgmlType::F32,
                vec![hidden as u64],
                zeros(hidden),
            )
            .expect("cnn bias");
            b.add_tensor(
                &format!("text_encoder.module.cnn.{i}.1.gamma"),
                GgmlType::F32,
                vec![hidden as u64],
                ones(hidden),
            )
            .expect("gamma");
            b.add_tensor(
                &format!("text_encoder.module.cnn.{i}.1.beta"),
                GgmlType::F32,
                vec![hidden as u64],
                zeros(hidden),
            )
            .expect("beta");
        }
        for suffix in ["", "_reverse"] {
            b.add_tensor(
                &format!("text_encoder.module.lstm.weight_ih_l0{suffix}"),
                GgmlType::F32,
                vec![four_h as u64, hidden as u64],
                zeros(four_h * hidden),
            )
            .expect("lstm w_ih");
            b.add_tensor(
                &format!("text_encoder.module.lstm.weight_hh_l0{suffix}"),
                GgmlType::F32,
                vec![four_h as u64, lstm_hidden as u64],
                zeros(four_h * lstm_hidden),
            )
            .expect("lstm w_hh");
            b.add_tensor(
                &format!("text_encoder.module.lstm.bias_ih_l0{suffix}"),
                GgmlType::F32,
                vec![four_h as u64],
                zeros(four_h),
            )
            .expect("lstm b_ih");
            b.add_tensor(
                &format!("text_encoder.module.lstm.bias_hh_l0{suffix}"),
                GgmlType::F32,
                vec![four_h as u64],
                zeros(four_h),
            )
            .expect("lstm b_hh");
        }
        // Bert canary — real Kokoro-82M shape [178, 128] — but NO other bert
        // tensors. `Bert::new` must fail loudly at the second lookup
        // (`bert.module.embeddings.position_embeddings.weight`).
        b.add_tensor(
            BERT_CANARY_TENSOR,
            GgmlType::F32,
            vec![178, 128],
            zeros(178 * 128),
        )
        .expect("canary");
        let bytes = b.to_bytes().expect("serialize");

        // Use `match` on Result rather than `expect_err` — `KokoroTts` is not
        // `Debug`.
        match KokoroTts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict()) {
            Ok(_) => panic!("partial bert branch must fail loudly"),
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(
                    msg.contains("bert.module."),
                    "error must name a bert tensor (FR-EX-08); got: {msg}"
                );
            }
            Err(other) => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// End-to-end smoke against a REAL Kokoro-82M voice GGUF, gated on
    /// `VOKRA_KOKORO_GGUF` (same pattern as `tests/parity_kokoro.rs`).
    /// Skipped cleanly when the env var is unset, so CI stays green without
    /// the 82M-parameter fixture. The full loader chain (text_encoder + bert +
    /// prosody + decoder) must succeed and `synthesize_phonemes` must return
    /// non-empty finite PCM at the voice's declared sample rate.
    #[test]
    fn synthesize_from_real_gguf_gated() {
        let Some(gguf_path) = std::env::var_os("VOKRA_KOKORO_GGUF") else {
            eprintln!(
                "[kokoro::mod::synthesize_from_real_gguf_gated] SKIP: \
                 set VOKRA_KOKORO_GGUF to a converted Kokoro-82M voice GGUF."
            );
            return;
        };
        let tts = KokoroTts::from_path(&gguf_path).unwrap_or_else(|e| {
            panic!(
                "load VOKRA_KOKORO_GGUF = {gguf_path:?}: {e}. Convert via \
                 `vokra-cli convert --model kokoro-82m ...` first."
            )
        });
        // Style vector matched to the voice's declared style_dim; explicit
        // override so the voicepack lookup path (`NotImplemented` until T02)
        // is not exercised.
        let style = vec![0.0f32; tts.config().style_dim];
        // Two arbitrary in-range ids — the voice's phoneme table has ≥ 4
        // entries in every shipped Kokoro-82M voice.
        let audio = tts
            .synthesize_phonemes(&[0, 1, 2, 3], None, Some(&style), 0.0, 1.0)
            .expect("real GGUF synthesize");
        assert!(
            !audio.samples.is_empty(),
            "real GGUF synthesize must produce non-empty PCM"
        );
        assert!(
            audio.samples.iter().all(|s| s.is_finite()),
            "real GGUF PCM must be all-finite (FR-EX-08)"
        );
        assert_eq!(audio.sample_rate, tts.config().sample_rate);
    }

    #[test]
    fn tensor_store_rejects_wrong_shape() {
        // A tensor `w` shaped [3]; asking for [2] must fail loudly rather than
        // truncate silently (FR-EX-08).
        let mut b = GgufBuilder::new();
        b.add_string(vokra_core::gguf::chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        let bytes: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        b.add_tensor("w", GgmlType::F32, vec![3], bytes)
            .expect("add F32 tensor");
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let store = TensorStore::new(file);
        // Correct shape roundtrips.
        assert_eq!(
            store.tensor_shaped("w", &[3]).expect("shape ok"),
            vec![1.0, 2.0, 3.0]
        );
        // Wrong shape is rejected.
        assert!(matches!(
            store.tensor_shaped("w", &[2]),
            Err(VokraError::InvalidArgument(_))
        ));
    }
}
