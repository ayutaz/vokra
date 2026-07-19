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
    /// `bert_encoder.module.*`, 178 → 128 → 12× ALBERT → 768 → 512). When present
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
    /// The pipeline mirrors upstream `KModel.forward_with_tokens`
    /// (`kokoro==0.9.4` `model.py:86-119`):
    ///
    /// 1. `bert → bert_encoder` (`d_en`) feeds the prosody predictor;
    /// 2. the predictor yields per-phoneme durations
    ///    (`round(sigmoid.sum / speed).clamp(min=1)` — `model.py:107-109`)
    ///    plus the `F0Ntrain` F0 / energy contours at 2× frame rate
    ///    (`model.py:114-115`);
    /// 3. the **text-encoder output `t_en`** — NOT the BERT features — is
    ///    length-regulated into the decoder's `asr` input
    ///    (`asr = t_en @ pred_aln_trg`, `model.py:116-117`);
    /// 4. the decoder consumes `(asr, F0_pred, N_pred, ref_s[:, :128])`
    ///    (`model.py:118`).
    ///
    /// Feeding length-regulated BERT features + zero F0/N to the decoder
    /// (the pre-fix wiring) was the P1 upstream divergence found by the
    /// 2026-07-16 real-weight eval (round-trip WER 1.0).
    ///
    /// # Style resolution
    ///
    /// Exactly one of the two style sources must be present; both absent is a
    /// loud [`VokraError::InvalidArgument`] rather than a silent zero-style
    /// default (FR-EX-08):
    ///
    /// - `style_override = Some(vec)` — the caller-supplied style vector wins.
    ///   Two lengths are accepted:
    ///   * `2·style_dim` (= 256 for Kokoro-82M): upstream's full `ref_s`
    ///     voicepack row — `[:style_dim]` conditions the DECODER,
    ///     `[style_dim:]` conditions the PROSODY predictor
    ///     (`model.py:104` + `:118`). This is the fidelity path for real
    ///     voicepack styles (e.g. `af_heart.pt` rows).
    ///   * `style_dim` (= 128): one vector used for BOTH halves — equivalent
    ///     to `ref_s = concat([s, s])`. Kept for the parity fixtures and
    ///     backward compatibility.
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
    /// - `noise_scale` is reserved: the SineGen dither upstream injects is
    ///   deterministically neutralized (see
    ///   `decoder/generator.rs::Generator::forward` §Determinism); the
    ///   parameter is consumed, not silently dropped.
    /// - `length_scale` multiplies the per-phoneme sigmoid-sum before
    ///   rounding — the reciprocal of upstream's `speed`
    ///   (`duration = sigmoid(...).sum(-1) / speed`, `model.py:108`). `1.0`
    ///   reproduces upstream defaults.
    ///
    /// # Errors
    ///
    /// Any component error propagates verbatim (all typed): out-of-range
    /// phoneme id from the text encoder, shape mismatch inside prosody /
    /// decoder, or the style-resolution errors above.
    pub fn synthesize_phonemes(
        &self,
        phoneme_ids: &[i64],
        voice: Option<&str>,
        style_override: Option<&[f32]>,
        noise_scale: f32,
        length_scale: f32,
    ) -> Result<SynthesizedAudio> {
        // Reserved — the stochastic SineGen dither is deterministically
        // neutralized (generator.rs §Determinism). Consumed here so the
        // parameter is not silently dropped.
        let _ = noise_scale;

        // 1) Resolve the style vector (FR-EX-08: never a silent zero default).
        let style: Vec<f32> = if let Some(s) = style_override {
            let sd = self.config.style_dim;
            if s.len() != sd && s.len() != 2 * sd {
                return Err(VokraError::InvalidArgument(format!(
                    "kokoro TTS: style_override len {} — expected style_dim ({sd}) \
                     or 2·style_dim ({}) for a full ref_s voicepack row",
                    s.len(),
                    2 * sd,
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
        let (style_decoder, style_prosody) = split_ref_s(&style, self.config.style_dim);

        // 2) Text encoder → t_en [t, hidden_dim] row-major (`model.py:116`).
        let enc_arr = self.text_encoder.forward(phoneme_ids)?;
        let t_in = enc_arr.rows;
        let hidden = enc_arr.cols;
        if hidden != self.config.hidden_dim {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro TTS: text encoder produced cols {} != config.hidden_dim ({})",
                hidden, self.config.hidden_dim,
            )));
        }
        // Transpose t_en to [hidden, t] channel-major — the decoder's asr
        // source (length-regulated below).
        let mut t_en_ch = vec![0.0f32; hidden * t_in];
        for ti in 0..t_in {
            for c in 0..hidden {
                t_en_ch[c * t_in + ti] = enc_arr.data[ti * hidden + c];
            }
        }

        // 3) Prosody-input features `d_en`. Upstream feeds the PL-BERT
        //    branch's `[t, 512]` output (`bert → bert_encoder`,
        //    `model.py:102-103`) to `predictor.text_encoder` — never `t_en`.
        //    A slim fixture voice without the PL-BERT branch falls back to
        //    the text-encoder output (the T13-beta seam); real Kokoro-82M
        //    always carries the branch.
        let d_en_ch: Vec<f32> = if let Some(bert) = &self.bert {
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
            let mut ch = vec![0.0f32; hidden * t_in];
            for ti in 0..t_in {
                for c in 0..hidden {
                    ch[c * t_in + ti] = bert_out[ti * hidden + c];
                }
            }
            ch
        } else {
            t_en_ch.clone()
        };

        // 4) Prosody predictor (upstream path): durations via
        //    `round(sigmoid.sum · length_scale).clamp(min=1)` + F0/N contours
        //    at 2·t_frames (`model.py:105-115`). Style: the PROSODY half.
        let pros = self
            .prosody
            .forward_upstream(&d_en_ch, style_prosody, t_in, length_scale)?;
        let t_frames: usize = pros.durations.iter().sum();
        if t_frames == 0 {
            return Err(VokraError::InvalidArgument(
                "kokoro TTS: prosody predicted zero total frames".to_owned(),
            ));
        }

        // 5) Length regulation of t_en → asr [hidden, t_frames]
        //    (`asr = t_en @ pred_aln_trg`, `model.py:116-117`).
        let (asr, t_frames_actual) = nn::length_regulate(&t_en_ch, hidden, t_in, &pros.durations);
        debug_assert_eq!(t_frames_actual, t_frames);

        // 6) Decoder → PCM at `config.sample_rate`, with the REAL F0/N
        //    contours and the DECODER style half (`model.py:118`). The
        //    stub-mode branch (voice without decoder tensors — synthetic
        //    smoke fixtures only) keeps the legacy shape-only reduction.
        let pcm = if self.decoder.is_real() {
            self.decoder.forward_full(
                &asr,
                &pros.f0,
                &pros.n,
                style_decoder,
                t_frames,
                decoder::PhaseActivation::Sin,
            )?
        } else {
            self.decoder.forward(&asr, t_frames, style_decoder)?
        };

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
    ///    `style` (`style_dim` for both halves, or `2·style_dim` — the
    ///    PROSODY half `[style_dim:]` is used, matching upstream
    ///    `s = ref_s[:, 128:]`, `model.py:104`), `length_scale = 1.0`.
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
    /// * `style` length neither `style_dim` nor `2·style_dim` — a loud
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
        let sd = self.config.style_dim;
        if style.len() != sd && style.len() != 2 * sd {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro TTS: prosody parity style len {} — expected style_dim ({sd}) \
                 or 2·style_dim ({})",
                style.len(),
                2 * sd,
            )));
        }
        let (_style_decoder, style_prosody) = split_ref_s(style, sd);
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
        let out = self
            .prosody
            .forward_upstream(&encoded_ch, style_prosody, t_in, 1.0)?;
        let t_frames: usize = out.durations.iter().sum();
        let durations_i64: Vec<i64> = out.durations.iter().map(|&d| d as i64).collect();
        Ok((durations_i64, out.f0, out.n, out.hidden, t_frames))
    }

    /// Runs the internal decoder forward for one phoneme id sequence and
    /// returns the pre-iSTFT `(x_mag, x_phase, pcm)` triple. Test-only bridge
    /// for the M2-07-T15 decoder parity harness
    /// (`crates/vokra-models/tests/parity_kokoro.rs::decoder_forward_bit_parity`).
    ///
    /// Pipeline is the FULL upstream `forward_with_tokens` up to the
    /// pre-iSTFT split (`model.py:86-119`):
    /// 1. `text_encoder.forward(phoneme_ids)` → `t_en` `[t, hidden_dim]`.
    /// 2. `bert.forward(phoneme_ids)` → `d_en` (prosody-predictor input;
    ///    falls back to `t_en` on a slim voice without the PL-BERT branch).
    /// 3. Prosody `forward_upstream` → durations + REAL F0/N contours.
    /// 4. Length-regulate **`t_en`** → `asr` `[hidden, t_frames]`
    ///    (`asr = t_en @ pred_aln_trg`).
    /// 5. [`Decoder::forward_full_intermediate`] with the real F0/N, the
    ///    decoder style half, and `PhaseActivation::Sin`.
    ///
    /// This IS the `synthesize_phonemes` pipeline with intermediates
    /// exposed — the parity harness therefore exercises the exact mainline
    /// math (the pre-fix variant fed zero F0/N + BERT features, testing a
    /// wiring the mainline no longer has).
    ///
    /// The mag / phase tensors returned are `[n_half · t_gen]` channel-major
    /// (same layout as the reference dumper's
    /// `decoder_pre_istft_mag.f32` / `decoder_pre_istft_phase.f32`).
    ///
    /// # Errors
    ///
    /// * `style` length neither `style_dim` nor `2·style_dim` — a loud
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
        let sd = self.config.style_dim;
        if style.len() != sd && style.len() != 2 * sd {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro TTS: decoder parity style len {} — expected style_dim ({sd}) \
                 or 2·style_dim ({})",
                style.len(),
                2 * sd,
            )));
        }
        let (style_decoder, style_prosody) = split_ref_s(style, sd);
        // 1. Text encoder → t_en [t_in, hidden] row-major.
        let enc_arr = self.text_encoder.forward(phoneme_ids)?;
        let t_in = enc_arr.rows;
        let hidden = enc_arr.cols;
        if hidden != self.config.hidden_dim {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro TTS: text encoder cols {} != hidden_dim ({})",
                hidden, self.config.hidden_dim,
            )));
        }
        // Transpose t_en to channel-major [hidden, t_in] — the asr source.
        let mut t_en_ch = vec![0.0f32; hidden * t_in];
        for ti in 0..t_in {
            for c in 0..hidden {
                t_en_ch[c * t_in + ti] = enc_arr.data[ti * hidden + c];
            }
        }
        // 2. Prosody-predictor input d_en: bert output when the branch is
        //    present (upstream always), else the t_en fallback (slim voices).
        let d_en_ch: Vec<f32> = if let Some(bert) = &self.bert {
            let bert_out = bert.forward(phoneme_ids)?;
            let bert_cols = bert_out.len() / t_in;
            if bert_cols != hidden {
                return Err(VokraError::InvalidArgument(format!(
                    "kokoro TTS: bert output width {} != hidden_dim ({})",
                    bert_cols, hidden,
                )));
            }
            let mut ch = vec![0.0f32; hidden * t_in];
            for ti in 0..t_in {
                for c in 0..hidden {
                    ch[c * t_in + ti] = bert_out[ti * hidden + c];
                }
            }
            ch
        } else {
            t_en_ch.clone()
        };
        // 3. Prosody (upstream path): durations + REAL F0/N contours.
        let pros = self
            .prosody
            .forward_upstream(&d_en_ch, style_prosody, t_in, 1.0)?;
        // 4. Length-regulate t_en → asr [hidden, t_frames].
        let (asr, t_frames) = nn::length_regulate(&t_en_ch, hidden, t_in, &pros.durations);
        if t_frames == 0 {
            return Err(VokraError::InvalidArgument(
                "kokoro TTS: decoder parity produced t_frames = 0".to_owned(),
            ));
        }
        // 5. Dispatch through the intermediate accessor with the real
        //    contours and the decoder style half.
        self.decoder.forward_full_intermediate(
            &asr,
            &pros.f0,
            &pros.n,
            style_decoder,
            t_frames,
            decoder::PhaseActivation::Sin,
        )
    }
}

impl KokoroTts {
    /// Runs the decoder on CALLER-SUPPLIED prosody outputs — the module
    /// isolation bridge for the decoder parity harness.
    ///
    /// [`Self::decoder_forward_for_parity`] runs the full pipeline, so its
    /// decoder inputs carry the (honest, bounded) prosody deltas — and the
    /// NSF source path is **discontinuously** sensitive to them: the
    /// harmonic-source STFT's `angle` feature has an atan2 branch cut, so an
    /// ε difference in F0 flips near-zero-magnitude bins by 2π (measured on
    /// the fixture: f0 max |Δ| ≈ 3e-3 → ~1.2k flipped bins → decoder logit
    /// max |Δ| ≈ 2). Feeding the REFERENCE durations / F0 / N (from the
    /// upstream-true fixtures) isolates the decoder math itself, which is
    /// what the `decoder_*` fixtures gate. The composed pipeline is covered
    /// by the e2e acceptance (round-trip WER / mel-L1), not by pretending
    /// the branch cut away (FR-EX-08: the exclusion is documented, not
    /// silent).
    ///
    /// `durations` length must equal `phoneme_ids` length; `f0` / `n` length
    /// must equal `2 · sum(durations)`.
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn decoder_forward_with_reference_contours(
        &self,
        phoneme_ids: &[i64],
        style: &[f32],
        durations: &[usize],
        f0: &[f32],
        n: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let sd = self.config.style_dim;
        if style.len() != sd && style.len() != 2 * sd {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro TTS: decoder parity style len {} — expected style_dim ({sd}) \
                 or 2·style_dim ({})",
                style.len(),
                2 * sd,
            )));
        }
        let (style_decoder, _style_prosody) = split_ref_s(style, sd);
        if durations.len() != phoneme_ids.len() {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro TTS: durations len {} != phoneme_ids len {}",
                durations.len(),
                phoneme_ids.len(),
            )));
        }
        let enc_arr = self.text_encoder.forward(phoneme_ids)?;
        let t_in = enc_arr.rows;
        let hidden = enc_arr.cols;
        let mut t_en_ch = vec![0.0f32; hidden * t_in];
        for ti in 0..t_in {
            for c in 0..hidden {
                t_en_ch[c * t_in + ti] = enc_arr.data[ti * hidden + c];
            }
        }
        let (asr, t_frames) = nn::length_regulate(&t_en_ch, hidden, t_in, durations);
        if t_frames == 0 {
            return Err(VokraError::InvalidArgument(
                "kokoro TTS: reference durations sum to 0 frames".to_owned(),
            ));
        }
        self.decoder.forward_full_intermediate(
            &asr,
            f0,
            n,
            style_decoder,
            t_frames,
            decoder::PhaseActivation::Sin,
        )
    }
}

/// Splits a resolved style vector into `(decoder_half, prosody_half)`.
///
/// A `2·style_dim` vector is upstream's full `ref_s` voicepack row —
/// `[:style_dim]` conditions the decoder, `[style_dim:]` the prosody
/// predictor (`model.py:104` `s = ref_s[:, 128:]` + `:118`
/// `ref_s[:, :128]`). A plain `style_dim` vector is used for both halves
/// (equivalent to `ref_s = concat([s, s])` — the parity-fixture
/// convention). Callers validate the length BEFORE calling; any other
/// length is a caller bug caught by their loud checks.
fn split_ref_s(style: &[f32], style_dim: usize) -> (&[f32], &[f32]) {
    if style.len() == 2 * style_dim {
        (&style[..style_dim], &style[style_dim..])
    } else {
        (style, style)
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
