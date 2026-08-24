//! Kokoro-82M native TTS (M2-07) — module skeleton.
//!
//! Native re-implementation of the Kokoro-82M inference core (StyleTTS 2 派生
//! iSTFTNet) in the whisper.cpp style: the model definition lives in Rust,
//! only the upstream **checkpoint** (Apache 2.0, converted offline to GGUF by
//! `vokra-convert`) is consumed at runtime. No ONNX runs at runtime
//! (FR-LD-05). G2P (misaki) is out of scope for M2-07; the runtime consumes
//! phoneme ids only (see `docs/adr/0007-kokoro-native.md`).
//!
//! # Layout (M2-07)
//!
//! - `config` — `vokra.kokoro.*` metadata + shape-cross-checked dims
//!   (T09);
//! - `weights` — F32-only [`GgufFile`] tensor
//!   store, rejecting a non-F32 tensor as a converter bug (T10);
//! - `nn` — 1-D dilated / grouped / transposed convolutions, activations
//!   plus a private `nn::adain` helper (StyleTTS 2 AdaIN as a composition
//!   of instance-norm + affine, **not** a new first-class op — FR-EX-08
//!   permits composition);
//! - `text_encoder` / `bert` / `prosody` / `decoder` — component
//!   skeletons; the concrete forward paths land at T12–T17. `bert` is the
//!   T13-beta PL-BERT branch (`bert.module.*` + `bert_encoder.module.*`,
//!   loaded only when the canary tensor is present — see
//!   `BERT_CANARY_TENSOR`). The iSTFT head uses FR-OP-01 `istft`, **not**
//!   the FR-OP-12 `vocos_head` — Kokoro is iSTFTNet 系.
//!
//! # Hot ops (M2-08 alignment)
//!
//! Kokoro dispatches **GEMM only** through the [`crate::compute::Compute`]
//! seam (every conv routes through `nn::conv1d`'s im2col + GEMM); the
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
use vokra_ops::ProsodyControl;

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
    /// Stacked per-voice style table (`voices/*.pt` merged), when the voice
    /// GGUF carries a `voicepack` tensor. `None` for the canonical conversion
    /// (upstream ships voices as separate `voices/*.pt` files, so the default
    /// safetensors — and thus the GGUF — has no stacked voicepack). When
    /// `None`, a `voice = Some(name)` synthesis returns a loud, actionable
    /// error rather than a silent zero-style fallback (FR-EX-08). Populated by
    /// [`VoicePack::load`] at load time (M2-07-T02).
    voicepack: Option<VoicePack>,
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
        // Optional stacked per-voice style table. Absent on the canonical
        // conversion (voices ship as separate `voices/*.pt` files); present
        // when the safetensors was built with
        // `kokoro_prepare_checkpoint.py --stack-voicepack`. A malformed
        // voicepack (wrong rank / row width / voice count) fails loudly here
        // rather than mid-synthesis (FR-EX-08).
        let voicepack = VoicePack::load(&store, &config)?;
        // `store` (and its GGUF backing bytes) drops here.
        Ok(Self {
            config,
            dims,
            text_encoder,
            bert,
            prosody,
            decoder,
            voicepack,
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
    ///   [`VokraError::InvalidArgument`]. The resolved voice id then indexes
    ///   the stacked voicepack row (M2-07-T02): upstream Kokoro selects
    ///   `pack[len(ps) - 1]` from a per-voice `[max_tokens, 1, 2·style_dim]`
    ///   table (`pipeline.py:232`), so the row is
    ///   `min(phoneme_ids.len() - 1, max_tokens - 1)` and the resulting
    ///   `2·style_dim` `ref_s` feeds the same `split_ref_s` path as a full
    ///   voicepack `style_override`. A voice GGUF built the canonical way
    ///   (no stacked `voicepack` tensor — upstream ships voices as separate
    ///   `voices/*.pt` files) has nothing to look up: a **known** name then
    ///   returns a loud [`VokraError::InvalidArgument`] naming the rebuild
    ///   step (`--stack-voicepack`) — never a silent zero-style fallback
    ///   (FR-EX-08).
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
        // Thin delegate to the prosody-aware entry point with `prosody = None`.
        // Routing every synthesis through a single code path guarantees that
        // the pre-M3-17 output is bit-identical to a `prosody = None` call
        // (or an explicit identity [`ProsodyControl`]) by construction —
        // there is no shadow pipeline that could drift.
        self.synthesize_phonemes_with_prosody(
            phoneme_ids,
            voice,
            style_override,
            noise_scale,
            length_scale,
            None,
        )
    }

    /// Single-chunk **pseudo-streaming** wrapper around
    /// [`Self::synthesize_phonemes`] — the FR-ST-04 API surface for Kokoro.
    ///
    /// # Why "pseudo-streaming" and not "streaming" (FR-ST-04)
    ///
    /// Kokoro-82M (`iSTFTNet` head, upstream `kokoro==0.9.4` `model.py`) is a
    /// **full-utterance** synthesizer: `forward_with_tokens`
    /// (`model.py:86-119`) runs the text encoder → PL-BERT → prosody predictor
    /// → decoder → iSTFT chain over the entire phoneme sequence in one forward
    /// pass, and there is no chunk-boundary that would let the model emit
    /// audio while later frames are still being generated. Wrapping the
    /// returned PCM in a chunk-emitting adapter reduces first-byte latency to
    /// the caller, but the **generation** time is unchanged — the model has
    /// already finished by the time the single chunk is emitted (mirrors the
    /// same honesty red-line in
    /// [`crate::piper_plus::PiperPlusTts::synthesize_pseudo_streaming`]).
    ///
    /// To keep this honest at the API boundary (SRS §FR-ST-04
    /// `docs/system-requirements.md:190` — "真のストリーミング非対応モデルは
    /// pseudo-streaming と API 名で明示する（例: `synthesize_pseudo_streaming`）")
    /// this method is named `synthesize_phonemes_pseudo_streaming` rather
    /// than `synthesize_phonemes_streaming`. A truly-incremental Kokoro path
    /// (e.g. a per-phoneme downgrade adapter or a streaming iSTFT with
    /// state carry-over across frames) would land as a distinct method — it
    /// must not be silently swapped in behind this signature, or the
    /// pseudo-streaming honesty guarantee is broken.
    ///
    /// # Contract
    ///
    /// - Returns `Ok(iter)` on success where `iter` yields **exactly one
    ///   chunk** equal to the full PCM buffer of [`Self::synthesize_phonemes`],
    ///   then `None`.
    /// - Sync setup errors (unknown voice, style-length mismatch, empty
    ///   phoneme sequence, out-of-range phoneme id, decoder shape mismatch,
    ///   …) surface as the OUTER `Err` — the caller never sees an iterator
    ///   that would then yield `Err`. This matches the `?`-propagation
    ///   semantic and the FR-EX-08 red-line against silent-drop paths.
    ///
    /// The parameter set mirrors [`Self::synthesize_phonemes`] exactly (5
    /// axes: `phoneme_ids`, `voice`, `style_override`, `noise_scale`,
    /// `length_scale`) so the pseudo-streaming call is a drop-in replacement
    /// for callers that want an [`Iterator`]-shaped adapter but do not need
    /// M3-17 prosody control on this axis. A prosody-aware pseudo-streaming
    /// variant, if ever needed, lands as a distinct method rather than
    /// mutating this one.
    ///
    /// # Errors
    ///
    /// Any error [`Self::synthesize_phonemes`] can surface propagates as the
    /// outer `Err` (typed [`VokraError`]).
    pub fn synthesize_phonemes_pseudo_streaming(
        &self,
        phoneme_ids: &[i64],
        voice: Option<&str>,
        style_override: Option<&[f32]>,
        noise_scale: f32,
        length_scale: f32,
    ) -> Result<impl Iterator<Item = Result<Vec<f32>>>> {
        // Sync body has already finished by the time we return: the "streaming"
        // is a chunk-shaped adapter over the full buffer, NOT a true
        // per-chunk forward. `?` surfaces sync errors as the outer `Err`, so
        // the caller never gets an iterator that would yield `Err(_)` — see
        // the contract docstring above.
        let audio = self.synthesize_phonemes(
            phoneme_ids,
            voice,
            style_override,
            noise_scale,
            length_scale,
        )?;
        Ok(std::iter::once(Ok(audio.samples)))
    }

    /// Prosody-aware variant of [`Self::synthesize_phonemes`] — the M3-17
    /// unified prosody-control API wired into Kokoro.
    ///
    /// The pipeline is identical to [`Self::synthesize_phonemes`]; only the
    /// `prosody` axis differs:
    ///
    /// - `prosody = None` OR an identity [`ProsodyControl`] — bit-identical
    ///   to the plain [`Self::synthesize_phonemes`] entry (both routes
    ///   evaluate `effective_length_scale = length_scale` and skip the F0
    ///   scaling branch).
    /// - `prosody = Some(ctrl)` with numeric axes — Kokoro honours them
    ///   NATIVELY through its existing pipeline knobs, so no adapter trait
    ///   folding is invented (M3-17 red-line, CLAUDE.md hallucination ban):
    ///     * `pitch_shift` (semitones) becomes a multiplicative F0 factor
    ///       `2^(semitones / 12)` applied post-predictor, pre-decoder to
    ///       [`crate::kokoro::prosody::ProsodyOutput::f0`]. Energy `n` and
    ///       durations are untouched — pitch axis only.
    ///     * `speed_scale` (`0.5..=2.0`) is folded into `length_scale` as
    ///       `length_scale / speed_scale` — Kokoro's `length_scale` is the
    ///       reciprocal of upstream `speed` (§Scales above), so a
    ///       caller-side `speed = 2.0` request shortens the output by half.
    /// - `prosody = Some(ctrl)` with a TEXT axis — rejected LOUDLY with a
    ///   named blocker (FR-EX-08, never a silent drop):
    ///     * `pause_ms` — Kokoro's `duration_proj` has no phoneme-level
    ///       pause semantic (it yields one integer count per phoneme; there
    ///       is no side channel for a caller pause).
    ///     * `instruction` — Kokoro consumes phoneme ids only; per M3-17
    ///       the instruction-folding [`vokra_ops::ApplyProsody`] adapter is
    ///       CosyVoice2-only in v0.9.
    ///
    /// Kokoro deliberately does NOT implement [`vokra_ops::ApplyProsody`];
    /// that trait is for instruction-folding adapters (CosyVoice2 in v0.9,
    /// M3-17 rustdoc). Kokoro consumes the numeric axes directly through
    /// this method — no [`vokra_ops::ProsodyControl::instruction`] is ever
    /// produced or consumed here.
    ///
    /// # Errors
    ///
    /// In addition to every error [`Self::synthesize_phonemes`] can
    /// surface, [`VokraError::InvalidArgument`] on any of the loud
    /// rejections listed above (non-finite `pitch_shift`, non-finite or
    /// out-of-range `speed_scale`, non-`None` `pause_ms`, non-`None`
    /// `instruction`).
    pub fn synthesize_phonemes_with_prosody(
        &self,
        phoneme_ids: &[i64],
        voice: Option<&str>,
        style_override: Option<&[f32]>,
        noise_scale: f32,
        length_scale: f32,
        prosody: Option<&ProsodyControl>,
    ) -> Result<SynthesizedAudio> {
        // Resolve prosody up front so downstream sees only the effective
        // `(length_scale, pitch_factor)` pair. This is where the loud
        // rejections (un-honoured axes / out-of-range values) surface —
        // before any tensor work runs (FR-EX-08).
        let (effective_length_scale, pitch_factor) =
            resolve_prosody_for_kokoro(length_scale, prosody)?;

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
            let voice_id = self.config.voice_id(name).ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "kokoro TTS: unknown voice `{name}` (voice_names = {:?})",
                    self.config.voice_names,
                ))
            })?;
            let Some(voicepack) = self.voicepack.as_ref() else {
                return Err(VokraError::InvalidArgument(format!(
                    "kokoro TTS: voice `{name}` (id {voice_id}) resolved, but this \
                     voice GGUF carries no stacked `voicepack` tensor to look up \
                     its style vector. The canonical Kokoro-82M release ships \
                     voices as separate `voices/*.pt` files, so the default \
                     conversion omits them — rebuild the GGUF from a safetensors \
                     produced with \
                     `tools/parity/kokoro_prepare_checkpoint.py --stack-voicepack`, \
                     or pass `style_override` with the voice's `ref_s` row."
                )));
            };
            // Upstream selects `pack[len(ps) - 1]` (`pipeline.py:232`); the row
            // is `2·style_dim` by construction (validated at load), matching the
            // full-`ref_s` `style_override` length.
            voicepack.ref_s(voice_id, phoneme_ids.len())?.to_vec()
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
        //    `effective_length_scale` folds any `ProsodyControl::speed_scale`
        //    request into the caller-supplied `length_scale` (identity path
        //    yields `effective == length_scale` exactly, so pre-M3-17 output
        //    is bit-identical).
        let mut pros =
            self.prosody
                .forward_upstream(&d_en_ch, style_prosody, t_in, effective_length_scale)?;
        let t_frames: usize = pros.durations.iter().sum();
        if t_frames == 0 {
            return Err(VokraError::InvalidArgument(
                "kokoro TTS: prosody predicted zero total frames".to_owned(),
            ));
        }

        // 4b) Apply the pitch-shift request to the F0 contour before the
        //     decoder consumes it (M3-17 numeric axis). Energy `n` and
        //     durations are untouched — pitch axis only. The equality vs
        //     exact `1.0` guards a genuine no-op: `resolve_prosody_for_kokoro`
        //     returns `1.0` bit-exact on identity / `None` paths and on
        //     `pitch_shift = 0.0` (`2f32.powf(0.0) == 1.0`), so the
        //     bit-identical guarantee for those callers is preserved by
        //     construction.
        if pitch_factor != 1.0 {
            for f in pros.f0.iter_mut() {
                *f *= pitch_factor;
            }
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

/// Stacked per-voice style table recovered from a voice GGUF's `voicepack`
/// tensor (M2-07-T02).
///
/// Upstream Kokoro-82M ships one style tensor per voice in a separate
/// `voices/<name>.pt` file, each shaped `[max_tokens, 1, 2·style_dim]` — a
/// per-phoneme-count *history* of `ref_s` rows (`ref_s = [decoder ; prosody]`).
/// `tools/parity/kokoro_prepare_checkpoint.py --stack-voicepack` stacks them on
/// a new leading axis (in `voice_names` alphabetical order) into a single
/// `voicepack` tensor `[num_voices, max_tokens, 1, 2·style_dim]` that the Rust
/// converter writes verbatim into the GGUF. This struct owns that table as a
/// flat row-major buffer and resolves a `ref_s` row for `(voice_id,
/// phoneme_count)` exactly as upstream `KPipeline` does (`pack[len(ps) - 1]`,
/// `pipeline.py:232`), clamped to the table height.
///
/// A `[num_voices, max_tokens, 2·style_dim]` 3-D layout (singleton axis
/// squeezed) is accepted as well. The buffer is row-major so the singleton axis
/// does not change the flat layout: voice `v`, row `r`, channel `k` lives at
/// `(v · max_tokens + r) · row_len + k`.
struct VoicePack {
    /// Row-major `[num_voices, max_tokens, row_len]` (the singleton `1` axis, if
    /// present in the GGUF shape, is flattened out — it does not affect offsets).
    data: Vec<f32>,
    num_voices: usize,
    max_tokens: usize,
    /// `2·style_dim` — the full `ref_s` width (decoder half ++ prosody half).
    row_len: usize,
}

impl VoicePack {
    /// GGUF tensor name for the stacked voicepack (verbatim from the prep tool).
    const TENSOR: &'static str = "voicepack";

    /// Loads the stacked voicepack from a voice GGUF, if present.
    ///
    /// Returns `Ok(None)` when the `voicepack` tensor is absent — the canonical
    /// conversion path, where a `voice = Some(name)` synthesis then fails
    /// loudly at the call site (FR-EX-08). When present, the shape and element
    /// count are validated against `config` and a malformed voicepack (wrong
    /// rank, row width ≠ `2·style_dim`, voice count ≠ `config.num_voices`, or a
    /// degenerate dim) is rejected with a loud [`VokraError::InvalidArgument`]
    /// rather than loaded silently.
    fn load(store: &TensorStore, config: &KokoroConfig) -> Result<Option<Self>> {
        let shape = match store.shape(Self::TENSOR) {
            Ok(shape) => shape,
            // Absent tensor — the canonical (non-stacked) voice GGUF.
            Err(_) => return Ok(None),
        };
        let row_len = 2 * config.style_dim;
        // Accept the canonical 4-D `[nv, mt, 1, ch]` (singleton squeezed out) or
        // a 3-D `[nv, mt, ch]` layout; anything else is malformed.
        let (num_voices, max_tokens, ch) = match shape.as_slice() {
            [nv, mt, one, ch] if *one == 1 => (*nv, *mt, *ch),
            [nv, mt, ch] => (*nv, *mt, *ch),
            other => {
                return Err(VokraError::InvalidArgument(format!(
                    "kokoro voicepack tensor shape {other:?} unsupported — expected \
                     [num_voices, max_tokens, 1, 2·style_dim] or \
                     [num_voices, max_tokens, 2·style_dim]"
                )));
            }
        };
        if ch != row_len {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro voicepack row width {ch} != 2·style_dim ({row_len}); the \
                 voicepack must carry a full ref_s row per (voice, phoneme-count)"
            )));
        }
        if num_voices != config.num_voices {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro voicepack rows {num_voices} != vokra.kokoro.num_voices \
                 ({}); the voicepack axis 0 must be in voice_names order",
                config.num_voices,
            )));
        }
        if num_voices == 0 || max_tokens == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro voicepack degenerate dims (num_voices={num_voices}, \
                 max_tokens={max_tokens})"
            )));
        }
        let data = store.tensor(Self::TENSOR)?;
        let expected = num_voices
            .checked_mul(max_tokens)
            .and_then(|v| v.checked_mul(row_len))
            .ok_or_else(|| {
                VokraError::InvalidArgument("kokoro voicepack element count overflow".to_owned())
            })?;
        if data.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro voicepack element count {} != num_voices·max_tokens·row_len \
                 ({expected})",
                data.len(),
            )));
        }
        Ok(Some(Self {
            data,
            num_voices,
            max_tokens,
            row_len,
        }))
    }

    /// Resolves the `ref_s` row (`2·style_dim` floats) for `voice_id` at the
    /// given `phoneme_count`.
    ///
    /// Mirrors upstream `KPipeline.infer`: `pack[len(ps) - 1]`
    /// (`pipeline.py:232`), where `len(ps)` is the phoneme count and `pack` is
    /// the per-voice `[max_tokens, 1, 2·style_dim]` history. `phoneme_count` is
    /// the number of phoneme ids supplied to
    /// [`KokoroTts::synthesize_phonemes`] (which is upstream's `input_ids` to
    /// `forward_with_tokens`, i.e. `len(ps)` when the caller passes the raw,
    /// un-boundary-wrapped phonemes — the same convention the parity
    /// `phoneme_ids.i64` fixture uses). The row is clamped to
    /// `[0, max_tokens - 1]`; a `phoneme_count` of 0 maps to row 0.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `voice_id >= num_voices` (a defensive
    /// re-check — callers resolve it via [`KokoroConfig::voice_id`] first, but a
    /// voice-name table longer than the voicepack would otherwise index out of
    /// bounds).
    fn ref_s(&self, voice_id: usize, phoneme_count: usize) -> Result<&[f32]> {
        if voice_id >= self.num_voices {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro voicepack: voice id {voice_id} out of range 0..{}",
                self.num_voices,
            )));
        }
        let row = phoneme_count.saturating_sub(1).min(self.max_tokens - 1);
        let base = (voice_id * self.max_tokens + row) * self.row_len;
        Ok(&self.data[base..base + self.row_len])
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

/// Resolves a Kokoro `(effective_length_scale, pitch_factor)` pair from the
/// caller's base `length_scale` and an optional [`ProsodyControl`] request.
///
/// This is the single seam that decides which prosody axes Kokoro can honour
/// and which are refused loudly (FR-EX-08 — never a silent drop):
///
/// - `None` OR identity control → `(base_length_scale, 1.0)` EXACTLY, so the
///   default path is bit-identical to a pre-M3-17 synthesis (by construction).
/// - `pitch_shift = Some(semitones)` → `pitch_factor = 2^(semitones / 12)`
///   (standard semitone → ratio conversion; `powf(0.0) = 1.0` bit-exact so a
///   `Some(0.0)` still hits the identity F0 path).
/// - `speed_scale = Some(s)` with `0.5 ≤ s ≤ 2.0` → `effective_length_scale =
///   base_length_scale / s`. Kokoro's `length_scale` is the reciprocal of
///   upstream `speed` (see [`KokoroTts::synthesize_phonemes`] §Scales), so a
///   caller-side `speed = 2.0` maps to `length_scale = base / 2.0` and
///   shortens the output by half.
/// - `pause_ms = Some(_)` → refused (Kokoro's `duration_proj` has no
///   phoneme-level pause semantic).
/// - `instruction = Some(_)` → refused (Kokoro has no text-instruction
///   consumer; per M3-17 the instruction folding is CosyVoice2-only).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on any of:
/// - `pitch_shift` non-finite (NaN / ±∞);
/// - `speed_scale` non-finite or outside `[0.5, 2.0]`;
/// - `pause_ms` set;
/// - `instruction` set.
fn resolve_prosody_for_kokoro(
    base_length_scale: f32,
    prosody: Option<&ProsodyControl>,
) -> Result<(f32, f32)> {
    let Some(ctrl) = prosody else {
        return Ok((base_length_scale, 1.0));
    };
    if ctrl.is_identity() {
        return Ok((base_length_scale, 1.0));
    }
    // Text axes: Kokoro cannot honour these; refuse loudly.
    if ctrl.pause_ms.is_some() {
        return Err(VokraError::InvalidArgument(
            "kokoro TTS: ProsodyControl.pause_ms is not honoured — Kokoro's \
             duration predictor has no phoneme-level pause semantic \
             (FR-EX-08). Insert silence at the phoneme-id level, or leave \
             pause_ms = None."
                .to_owned(),
        ));
    }
    if ctrl.instruction.is_some() {
        return Err(VokraError::InvalidArgument(
            "kokoro TTS: ProsodyControl.instruction is not honoured — Kokoro \
             consumes phoneme ids only; per M3-17 the instruction folding is \
             CosyVoice2-only. Leave instruction = None."
                .to_owned(),
        ));
    }
    // Numeric axes: honoured natively.
    let pitch_factor = if let Some(semitones) = ctrl.pitch_shift {
        if !semitones.is_finite() {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro TTS: ProsodyControl.pitch_shift must be finite (got {semitones})"
            )));
        }
        2f32.powf(semitones / 12.0)
    } else {
        1.0
    };
    let effective_length_scale = if let Some(speed) = ctrl.speed_scale {
        if !speed.is_finite() {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro TTS: ProsodyControl.speed_scale must be finite (got {speed})"
            )));
        }
        if !(0.5..=2.0).contains(&speed) {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro TTS: ProsodyControl.speed_scale {speed} outside supported \
                 range [0.5, 2.0]"
            )));
        }
        // Kokoro's `length_scale` = 1 / upstream `speed`
        // (`synthesize_phonemes_with_prosody` §Scales), so a caller-side
        // `speed = 2.0` maps to `length_scale = base / 2.0`.
        base_length_scale / speed
    } else {
        base_length_scale
    };
    Ok((effective_length_scale, pitch_factor))
}

impl TtsEngine for KokoroTts {
    /// Text → PCM adapter around [`KokoroTts::synthesize_phonemes`].
    ///
    /// Style resolution (documented / wired here for the future G2P bridge):
    ///
    /// - `request.speaker_embedding = Some(vec)` (matching `style_dim`) is
    ///   used verbatim as the style vector — the parity of the low-level
    ///   `synthesize_phonemes(style_override = …)` path.
    /// - Otherwise a voice name would index the stacked voicepack via
    ///   [`KokoroTts::synthesize_phonemes`] (`voice = Some(name)`, M2-07-T02);
    ///   the low-level path already does that lookup. This high-level adapter
    ///   still returns [`VokraError::NotImplemented`] because it lacks the G2P
    ///   step below, not because of the voicepack.
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

    fn backend(&self) -> BackendKind {
        self.backend_kind
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
        // override so this smoke exercises the decoder chain independently of
        // whether the GGUF carries a stacked voicepack (the voice-name lookup
        // has its own gated test below).
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

    // -----------------------------------------------------------------------
    // FR-ST-04 pseudo-streaming (single-chunk fallback) — Kokoro-82M has no
    // chunked forward upstream (kokoro==0.9.4 `model.py` runs the full
    // pipeline in one pass), so the API name pins the honesty red-line and
    // the tests pin the "one chunk equal to the sync output, then None"
    // contract. See KokoroTts::synthesize_phonemes_pseudo_streaming for the
    // full rationale.
    // -----------------------------------------------------------------------

    /// Compile-time gate on the `synthesize_phonemes_pseudo_streaming` name.
    ///
    /// A silent rename (e.g. dropping `_pseudo_` to look like a true
    /// streaming method) would violate FR-ST-04 without any runtime test
    /// firing — the fn-pointer binding here fails to compile if the symbol
    /// moves or the signature drifts. Mirrors the piper_plus precedent at
    /// `crates/vokra-models/src/piper_plus/mod.rs::synthesize_pseudo_streaming_symbol_exists`.
    #[test]
    fn synthesize_phonemes_pseudo_streaming_symbol_exists() {
        // Bind through the fully-qualified path so this is a strict
        // compile-time gate against a silent rename / signature drift.
        // The `impl Iterator<...>` return means we can only bind the fn
        // pointer with a leaked concrete type parameter — but the intent is
        // symbol existence, not calling the function, so a reference-through
        // a fn item is sufficient and does NOT require naming the opaque
        // return type.
        let _ = super::KokoroTts::synthesize_phonemes_pseudo_streaming;
    }

    /// Pseudo-streaming yields exactly one chunk equal to the sync output.
    ///
    /// Gated on `VOKRA_KOKORO_GGUF` (same pattern as
    /// [`synthesize_from_real_gguf_gated`]) — CI stays green without the
    /// 82M-parameter fixture. When set, both the sync entry and the
    /// pseudo-streaming iterator are called with identical inputs; the
    /// iterator MUST yield one chunk byte-equal to `sync.samples`, and MUST
    /// be drained after (`.next().is_none()`).
    #[test]
    fn pseudo_streaming_matches_sync_from_real_gguf_gated() {
        let Some(gguf_path) = std::env::var_os("VOKRA_KOKORO_GGUF") else {
            eprintln!(
                "[kokoro::mod::pseudo_streaming_matches_sync_from_real_gguf_gated] \
                 SKIP: set VOKRA_KOKORO_GGUF to a converted Kokoro-82M voice GGUF."
            );
            return;
        };
        let tts = KokoroTts::from_path(&gguf_path).unwrap_or_else(|e| {
            panic!(
                "load VOKRA_KOKORO_GGUF = {gguf_path:?}: {e}. Convert via \
                 `vokra-cli convert --model kokoro-82m ...` first."
            )
        });
        let style = vec![0.0f32; tts.config().style_dim];
        let phoneme_ids: [i64; 4] = [0, 1, 2, 3];

        // 1) Sync path — the byte-equality baseline.
        let sync = tts
            .synthesize_phonemes(&phoneme_ids, None, Some(&style), 0.0, 1.0)
            .expect("sync synthesize");
        assert!(
            !sync.samples.is_empty(),
            "sync sample buffer must be non-empty to make the parity meaningful"
        );

        // 2) Pseudo-streaming path — must yield exactly one chunk == sync PCM.
        let iter = tts
            .synthesize_phonemes_pseudo_streaming(&phoneme_ids, None, Some(&style), 0.0, 1.0)
            .expect("pseudo-streaming synthesize");
        let chunks: Vec<vokra_core::Result<Vec<f32>>> = iter.collect();
        assert_eq!(
            chunks.len(),
            1,
            "single-chunk fallback must yield exactly ONE chunk (FR-ST-04 pseudo-streaming contract)"
        );
        let chunk = chunks
            .into_iter()
            .next()
            .expect("one chunk")
            .expect("chunk is not an inner Err (sync errors surface as outer Err)");
        assert_eq!(
            chunk, sync.samples,
            "pseudo-streaming chunk must be byte-equal to sync PCM"
        );

        // 3) A fresh call — iterator MUST drain to None on the second `.next()`.
        //    Bind as `mut` so we can consume it.
        let mut iter2 = tts
            .synthesize_phonemes_pseudo_streaming(&phoneme_ids, None, Some(&style), 0.0, 1.0)
            .expect("pseudo-streaming synthesize (drain check)");
        assert!(iter2.next().is_some(), "first .next() must yield the chunk");
        assert!(
            iter2.next().is_none(),
            "second .next() must yield None (single-chunk fallback drains after 1)"
        );
    }

    /// Sync setup errors surface as the OUTER `Err`, never as an iterator
    /// that then yields `Err`.
    ///
    /// Gated on `VOKRA_KOKORO_GGUF` because reaching the text encoder's
    /// empty-input error path (`text_encoder.rs:248` — "kokoro text encoder:
    /// empty phoneme id sequence") requires a real loaded voice. Passes an
    /// empty `phoneme_ids` slice and asserts the outer `Result` is
    /// `Err(VokraError::InvalidArgument(_))`. This pins the `?`-propagation
    /// semantic documented on the method (FR-EX-08 — no silent-Ok-with-Err
    /// paths).
    #[test]
    fn pseudo_streaming_propagates_sync_error_from_real_gguf_gated() {
        let Some(gguf_path) = std::env::var_os("VOKRA_KOKORO_GGUF") else {
            eprintln!(
                "[kokoro::mod::pseudo_streaming_propagates_sync_error_from_real_gguf_gated] \
                 SKIP: set VOKRA_KOKORO_GGUF to a converted Kokoro-82M voice GGUF."
            );
            return;
        };
        let tts = KokoroTts::from_path(&gguf_path).unwrap_or_else(|e| {
            panic!(
                "load VOKRA_KOKORO_GGUF = {gguf_path:?}: {e}. Convert via \
                 `vokra-cli convert --model kokoro-82m ...` first."
            )
        });
        let style = vec![0.0f32; tts.config().style_dim];
        // Empty phoneme_ids fires `VokraError::InvalidArgument("kokoro text
        // encoder: empty phoneme id sequence")` at text_encoder.rs:248 — a
        // sync-setup error that must surface as the OUTER Err, NOT as an
        // Ok(iter) that then yields Err.
        let empty: [i64; 0] = [];
        let result = tts.synthesize_phonemes_pseudo_streaming(&empty, None, Some(&style), 0.0, 1.0);
        // KokoroTts does not implement Debug (owns non-Debug component
        // buffers), so the inner Ok type on this Result carries an opaque
        // `impl Iterator<...>` that ALSO cannot be Debug-printed. Use the
        // let-else pattern rather than `.expect_err(...)` to match on the
        // outer Result.
        let Err(err) = result else {
            panic!(
                "empty phoneme_ids must surface as OUTER Err — pseudo-streaming \
                 contract (FR-EX-08); got Ok(iter)"
            );
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("empty phoneme id sequence"),
                    "outer Err must name the text-encoder empty-input path \
                     (FR-EX-08); got: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Voicepack lookup (M2-07-T02) — deterministic, no multi-MB GGUF needed
    // -----------------------------------------------------------------------

    /// Minimal GGUF builder carrying all 11 `vokra.kokoro.*` config keys with a
    /// caller-chosen `style_dim` / `num_voices` / voice list, and NO component
    /// tensors. Used by the voicepack tests below, which drive
    /// [`KokoroConfig::from_gguf`] + [`VoicePack::load`] directly (bypassing the
    /// full component loader chain, so no 82M-parameter fixture is needed).
    fn kokoro_config_builder(style_dim: u32, num_voices: u32, voices: &[&str]) -> GgufBuilder {
        let mut b = GgufBuilder::new();
        b.add_u32(KEY_SAMPLE_RATE, 24_000);
        b.add_u32(KEY_STYLE_DIM, style_dim);
        b.add_u32(KEY_NUM_VOICES, num_voices);
        b.add_u32(KEY_HIDDEN_DIM, 8);
        b.add_u32(KEY_N_TEXT_LAYERS, 2);
        b.add_u32(KEY_N_DECODER_LAYERS, 2);
        b.add_u32(KEY_ISTFT_N_FFT, 20);
        b.add_u32(KEY_ISTFT_HOP, 5);
        b.add_u32(KEY_ISTFT_WIN_LENGTH, 20);
        b.add_metadata(KEY_PHONEME_SYMBOLS, str_array(&["_", "a"]));
        b.add_metadata(KEY_VOICE_NAMES, str_array(voices));
        b
    }

    /// Builds a synthetic stacked voicepack `[num_voices, max_tokens, 1, row_len]`
    /// (F32, row-major) whose element `(v, r, k)` = `v·1000 + r·10 + k`, so the
    /// selected slice pins the exact `(voice, row)` chosen.
    fn synthetic_voicepack_bytes(num_voices: usize, max_tokens: usize, row_len: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(num_voices * max_tokens * row_len * 4);
        for v in 0..num_voices {
            for r in 0..max_tokens {
                for k in 0..row_len {
                    let val = (v * 1000 + r * 10 + k) as f32;
                    data.extend_from_slice(&val.to_le_bytes());
                }
            }
        }
        data
    }

    #[test]
    fn voicepack_ref_s_selects_expected_row() {
        // 2 voices, max_tokens = 4, row_len = 4 (= 2·style_dim, style_dim = 2).
        let num_voices = 2;
        let max_tokens = 4;
        let row_len = 4;
        let data: Vec<f32> = synthetic_voicepack_bytes(num_voices, max_tokens, row_len)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let vp = VoicePack {
            data,
            num_voices,
            max_tokens,
            row_len,
        };

        // Upstream `pack[len(ps) - 1]`, clamped: count → row.
        assert_eq!(vp.ref_s(0, 1).unwrap(), &[0.0, 1.0, 2.0, 3.0]); // count 1 → row 0
        assert_eq!(vp.ref_s(0, 2).unwrap(), &[10.0, 11.0, 12.0, 13.0]); // count 2 → row 1
        assert_eq!(vp.ref_s(0, 4).unwrap(), &[30.0, 31.0, 32.0, 33.0]); // count 4 → row 3
        assert_eq!(vp.ref_s(0, 99).unwrap(), &[30.0, 31.0, 32.0, 33.0]); // clamp to row 3
        assert_eq!(vp.ref_s(0, 0).unwrap(), &[0.0, 1.0, 2.0, 3.0]); // saturating → row 0
        // Second voice is offset by 1000; count 3 → row 2.
        assert_eq!(vp.ref_s(1, 3).unwrap(), &[1020.0, 1021.0, 1022.0, 1023.0]);
    }

    #[test]
    fn voicepack_ref_s_rejects_out_of_range_voice() {
        let vp = VoicePack {
            data: vec![0.0f32; 2 * 4 * 4],
            num_voices: 2,
            max_tokens: 4,
            row_len: 4,
        };
        match vp.ref_s(2, 1) {
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(msg.contains("voice id 2"), "got: {msg}");
            }
            Ok(_) => panic!("voice id 2 out of range must be rejected (FR-EX-08)"),
            Err(other) => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn voicepack_load_reads_stacked_tensor() {
        // style_dim = 2 → row_len = 4; voicepack [2, 3, 1, 4].
        let style_dim = 2usize;
        let (num_voices, max_tokens, row_len) = (2usize, 3usize, 2 * style_dim);
        let mut b = kokoro_config_builder(style_dim as u32, num_voices as u32, &["v0", "v1"]);
        b.add_tensor(
            VoicePack::TENSOR,
            GgmlType::F32,
            vec![num_voices as u64, max_tokens as u64, 1, row_len as u64],
            synthetic_voicepack_bytes(num_voices, max_tokens, row_len),
        )
        .expect("voicepack tensor");
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let cfg = KokoroConfig::from_gguf(&file).expect("config");
        let store = TensorStore::new(file);
        let vp = VoicePack::load(&store, &cfg)
            .expect("load ok")
            .expect("voicepack present");
        assert_eq!(vp.num_voices, 2);
        assert_eq!(vp.max_tokens, 3);
        assert_eq!(vp.row_len, 4);
        // voice 1, count 2 → row 1; fill is v·1000 + r·10 + k ⇒ 1000 + 10 + k.
        assert_eq!(vp.ref_s(1, 2).unwrap(), &[1010.0, 1011.0, 1012.0, 1013.0]);
    }

    #[test]
    fn voicepack_load_absent_is_none() {
        let b = kokoro_config_builder(2, 2, &["v0", "v1"]);
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let cfg = KokoroConfig::from_gguf(&file).expect("config");
        let store = TensorStore::new(file);
        assert!(
            VoicePack::load(&store, &cfg).expect("load ok").is_none(),
            "absent voicepack tensor must load as None (canonical conversion)"
        );
    }

    #[test]
    fn voicepack_load_rejects_wrong_row_width() {
        // style_dim = 2 → row_len must be 4; supply a last axis of 6.
        let mut b = kokoro_config_builder(2, 2, &["v0", "v1"]);
        b.add_tensor(
            VoicePack::TENSOR,
            GgmlType::F32,
            vec![2, 3, 1, 6],
            synthetic_voicepack_bytes(2, 3, 6),
        )
        .expect("voicepack tensor");
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let cfg = KokoroConfig::from_gguf(&file).expect("config");
        let store = TensorStore::new(file);
        match VoicePack::load(&store, &cfg) {
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(msg.contains("row width 6"), "got: {msg}");
            }
            Ok(_) => panic!("row width 6 != 2·style_dim (4) must be rejected (FR-EX-08)"),
            Err(other) => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn voicepack_load_rejects_voice_count_mismatch() {
        // config declares 3 voices; voicepack carries only 2 rows.
        let mut b = kokoro_config_builder(2, 3, &["v0", "v1", "v2"]);
        b.add_tensor(
            VoicePack::TENSOR,
            GgmlType::F32,
            vec![2, 3, 1, 4],
            synthetic_voicepack_bytes(2, 3, 4),
        )
        .expect("voicepack tensor");
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let cfg = KokoroConfig::from_gguf(&file).expect("config");
        let store = TensorStore::new(file);
        match VoicePack::load(&store, &cfg) {
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(msg.contains("voicepack rows 2"), "got: {msg}");
            }
            Ok(_) => panic!("voicepack rows != num_voices must be rejected (FR-EX-08)"),
            Err(other) => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// Voice-name lookup against a REAL Kokoro-82M voice GGUF, gated on
    /// `VOKRA_KOKORO_GGUF`. Skipped cleanly when unset. This test documents
    /// BOTH real-world states and asserts each is handled correctly:
    ///
    /// * A GGUF built the canonical way (no stacked voicepack — the default
    ///   `kokoro_prepare_checkpoint.py` path) must return a loud
    ///   [`VokraError::InvalidArgument`] naming the `--stack-voicepack` rebuild
    ///   step — never [`VokraError::NotImplemented`] and never a silent stub.
    /// * A GGUF built with `--stack-voicepack` must resolve the voice name to a
    ///   real style row and synthesize non-empty, all-finite PCM.
    ///
    /// Any other outcome (NotImplemented, non-finite PCM, wrong error) fails.
    #[test]
    fn voice_name_lookup_from_real_gguf_gated() {
        let Some(gguf_path) = std::env::var_os("VOKRA_KOKORO_GGUF") else {
            eprintln!(
                "[kokoro::mod::voice_name_lookup_from_real_gguf_gated] SKIP: \
                 set VOKRA_KOKORO_GGUF to a converted Kokoro-82M voice GGUF."
            );
            return;
        };
        let tts = KokoroTts::from_path(&gguf_path)
            .unwrap_or_else(|e| panic!("load VOKRA_KOKORO_GGUF = {gguf_path:?}: {e}"));
        let Some(voice) = tts.config().voice_names.first().cloned() else {
            eprintln!(
                "[kokoro::mod::voice_name_lookup_from_real_gguf_gated] SKIP: \
                 GGUF has no voice_names to look up."
            );
            return;
        };
        // 8 arbitrary in-range ids (every shipped voice has ≥ 178 symbols).
        let ids = [1i64, 2, 3, 4, 5, 6, 7, 8];
        match tts.synthesize_phonemes(&ids, Some(&voice), None, 0.0, 1.0) {
            Ok(audio) => {
                assert!(
                    !audio.samples.is_empty() && audio.samples.iter().all(|s| s.is_finite()),
                    "voicepack-backed GGUF: voice `{voice}` must synthesize finite PCM"
                );
                assert_eq!(audio.sample_rate, tts.config().sample_rate);
                eprintln!(
                    "[voice_name_lookup] voice `{voice}` synthesized {} samples \
                     (voicepack present)",
                    audio.samples.len()
                );
            }
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(
                    msg.contains("--stack-voicepack") && msg.contains(&voice),
                    "no-voicepack GGUF must name the rebuild step; got: {msg}"
                );
                eprintln!(
                    "[voice_name_lookup] voice `{voice}` → loud no-voicepack error \
                     (canonical GGUF, as expected)"
                );
            }
            Err(other) => {
                panic!("voice-name lookup must be Ok or a loud InvalidArgument, not: {other:?}")
            }
        }
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

    // -----------------------------------------------------------------------
    // resolve_prosody_for_kokoro (M3-17 wiring) — pure-unit, no fixture
    // -----------------------------------------------------------------------

    #[test]
    fn resolve_prosody_none_passthrough_is_bit_identical() {
        // The bit-identical guarantee for pre-M3-17 callers relies on this
        // path returning the base `length_scale` unchanged and `pitch_factor`
        // exactly 1.0 (both used as `==` guards downstream).
        let (ls, pf) = resolve_prosody_for_kokoro(1.0, None).expect("None passthrough");
        assert_eq!(ls, 1.0);
        assert_eq!(pf, 1.0);
        let (ls2, pf2) = resolve_prosody_for_kokoro(0.75, None).expect("None passthrough");
        assert_eq!(ls2, 0.75);
        assert_eq!(pf2, 1.0);
    }

    #[test]
    fn resolve_prosody_identity_control_is_bit_identical() {
        // Identity `ProsodyControl` and explicit `None` must both preserve
        // the base `length_scale` exactly.
        let identity = ProsodyControl::identity();
        let (ls, pf) =
            resolve_prosody_for_kokoro(1.25, Some(&identity)).expect("identity passthrough");
        assert_eq!(ls, 1.25);
        assert_eq!(pf, 1.0);
        // default() is the same observable value as identity() per M3-17
        // rustdoc — pin that too.
        let default = ProsodyControl::default();
        let (ls_d, pf_d) =
            resolve_prosody_for_kokoro(1.25, Some(&default)).expect("default passthrough");
        assert_eq!(ls_d, 1.25);
        assert_eq!(pf_d, 1.0);
    }

    #[test]
    fn resolve_prosody_zero_semitones_is_identity_bit_exact() {
        // `2f32.powf(0.0)` is 1.0 exactly, so a `pitch_shift = Some(0.0)`
        // request must not trigger the F0 scaling branch. Pin the bit-exact
        // 1.0 here so any future rewrite that loses this property fails.
        let ctrl = ProsodyControl::default().with_pitch_shift(0.0);
        let (ls, pf) = resolve_prosody_for_kokoro(1.0, Some(&ctrl)).expect("zero semitones ok");
        assert_eq!(ls, 1.0);
        assert_eq!(pf, 1.0);
    }

    #[test]
    fn resolve_prosody_unit_speed_is_identity_bit_exact() {
        // `base / 1.0 == base` bit-exact — pin it so the identity contract
        // survives future refactors.
        let ctrl = ProsodyControl::default().with_speed_scale(1.0);
        let (ls, pf) = resolve_prosody_for_kokoro(0.5, Some(&ctrl)).expect("unit speed ok");
        assert_eq!(ls, 0.5);
        assert_eq!(pf, 1.0);
    }

    #[test]
    fn resolve_prosody_speed_inverts_into_length_scale() {
        // `length_scale = base / speed`. Base 1.0 chosen so results are
        // exact powers of two (division by 2 / 0.5 is bit-exact in f32).
        let fast = ProsodyControl::default().with_speed_scale(2.0);
        let (ls, pf) = resolve_prosody_for_kokoro(1.0, Some(&fast)).expect("speed 2.0 ok");
        assert_eq!(ls, 0.5);
        assert_eq!(pf, 1.0);
        let slow = ProsodyControl::default().with_speed_scale(0.5);
        let (ls_s, pf_s) = resolve_prosody_for_kokoro(1.0, Some(&slow)).expect("speed 0.5 ok");
        assert_eq!(ls_s, 2.0);
        assert_eq!(pf_s, 1.0);
    }

    #[test]
    fn resolve_prosody_twelve_semitones_yields_two() {
        // One octave up = factor 2. `2f32.powf(1.0)` is close to but not
        // guaranteed to be bit-exact 2.0 across libm impls, so use a small
        // absolute tolerance for the pitch factor. length_scale unchanged.
        let ctrl = ProsodyControl::default().with_pitch_shift(12.0);
        let (ls, pf) = resolve_prosody_for_kokoro(1.0, Some(&ctrl)).expect("12 semitones ok");
        assert_eq!(ls, 1.0);
        assert!(
            (pf - 2.0).abs() < 1e-6,
            "12 semitones must yield pitch factor ≈ 2.0 (got {pf})"
        );
        // One octave down = factor 0.5.
        let ctrl_dn = ProsodyControl::default().with_pitch_shift(-12.0);
        let (_, pf_dn) = resolve_prosody_for_kokoro(1.0, Some(&ctrl_dn)).expect("-12 semitones ok");
        assert!(
            (pf_dn - 0.5).abs() < 1e-6,
            "-12 semitones must yield pitch factor ≈ 0.5 (got {pf_dn})"
        );
    }

    #[test]
    fn resolve_prosody_rejects_pause_ms() {
        // Kokoro cannot honour a per-caller pause — reject loudly rather
        // than silently ignore (FR-EX-08).
        let ctrl = ProsodyControl::default().with_pause_ms(200);
        match resolve_prosody_for_kokoro(1.0, Some(&ctrl)) {
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(
                    msg.contains("pause_ms"),
                    "error must name the un-honoured axis; got: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn resolve_prosody_rejects_instruction() {
        // Kokoro consumes phoneme ids only — no text-instruction path exists
        // (per M3-17: instruction folding is CosyVoice2-only).
        let ctrl = ProsodyControl::default().with_instruction("speak calmly");
        match resolve_prosody_for_kokoro(1.0, Some(&ctrl)) {
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(
                    msg.contains("instruction"),
                    "error must name the un-honoured axis; got: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn resolve_prosody_rejects_out_of_range_speed() {
        // Below the supported floor.
        let too_slow = ProsodyControl::default().with_speed_scale(0.4);
        assert!(matches!(
            resolve_prosody_for_kokoro(1.0, Some(&too_slow)),
            Err(VokraError::InvalidArgument(_))
        ));
        // Above the supported ceiling.
        let too_fast = ProsodyControl::default().with_speed_scale(2.5);
        assert!(matches!(
            resolve_prosody_for_kokoro(1.0, Some(&too_fast)),
            Err(VokraError::InvalidArgument(_))
        ));
        // 0.5 and 2.0 are the inclusive bounds — must NOT be rejected.
        let low_bound = ProsodyControl::default().with_speed_scale(0.5);
        assert!(resolve_prosody_for_kokoro(1.0, Some(&low_bound)).is_ok());
        let high_bound = ProsodyControl::default().with_speed_scale(2.0);
        assert!(resolve_prosody_for_kokoro(1.0, Some(&high_bound)).is_ok());
    }

    #[test]
    fn resolve_prosody_rejects_non_finite_pitch() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let ctrl = ProsodyControl::default().with_pitch_shift(bad);
            match resolve_prosody_for_kokoro(1.0, Some(&ctrl)) {
                Err(VokraError::InvalidArgument(msg)) => {
                    assert!(
                        msg.contains("pitch_shift"),
                        "error must name pitch_shift for {bad}; got: {msg}"
                    );
                }
                other => panic!("expected InvalidArgument for {bad}, got {other:?}"),
            }
        }
    }

    #[test]
    fn resolve_prosody_rejects_non_finite_speed() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let ctrl = ProsodyControl::default().with_speed_scale(bad);
            assert!(
                matches!(
                    resolve_prosody_for_kokoro(1.0, Some(&ctrl)),
                    Err(VokraError::InvalidArgument(_))
                ),
                "speed_scale {bad} must be rejected"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Real-checkpoint gated integration tests (VOKRA_KOKORO_GGUF)
    // -----------------------------------------------------------------------

    /// The M3-17 bit-identical contract on a REAL voice: the plain
    /// `synthesize_phonemes` entry, an explicit `None` prosody, and an
    /// identity `ProsodyControl` must all produce byte-equal PCM. Skipped
    /// cleanly when `VOKRA_KOKORO_GGUF` is unset.
    #[test]
    fn prosody_none_and_identity_bit_identical_pcm_on_real_gguf_gated() {
        let Some(gguf_path) = std::env::var_os("VOKRA_KOKORO_GGUF") else {
            eprintln!(
                "[kokoro::mod::prosody_none_and_identity_bit_identical_pcm_on_real_gguf_gated] \
                 SKIP: set VOKRA_KOKORO_GGUF to a converted Kokoro-82M voice GGUF."
            );
            return;
        };
        let tts = KokoroTts::from_path(&gguf_path).unwrap_or_else(|e| {
            panic!(
                "load VOKRA_KOKORO_GGUF = {gguf_path:?}: {e}. Convert via \
                 `vokra-cli convert --model kokoro-82m ...` first."
            )
        });
        let style = vec![0.0f32; tts.config().style_dim];
        let ids = [0i64, 1, 2, 3];
        let plain = tts
            .synthesize_phonemes(&ids, None, Some(&style), 0.0, 1.0)
            .expect("plain synthesize");
        let explicit_none = tts
            .synthesize_phonemes_with_prosody(&ids, None, Some(&style), 0.0, 1.0, None)
            .expect("explicit None prosody");
        let identity = ProsodyControl::identity();
        let identity_out = tts
            .synthesize_phonemes_with_prosody(&ids, None, Some(&style), 0.0, 1.0, Some(&identity))
            .expect("identity ProsodyControl");
        assert_eq!(
            plain.samples, explicit_none.samples,
            "plain and explicit-None prosody must produce bit-identical PCM \
             (single code path guarantees this by construction)"
        );
        assert_eq!(
            plain.samples, identity_out.samples,
            "plain and identity ProsodyControl must produce bit-identical PCM \
             (identity is passthrough per M3-17 contract)"
        );
        assert_eq!(plain.sample_rate, identity_out.sample_rate);
    }

    /// A caller-side `speed_scale = 1.5` must shorten the frame count vs
    /// the neutral baseline. Skipped when the env var is unset.
    #[test]
    fn prosody_speed_1_5_shortens_frame_count_on_real_gguf_gated() {
        let Some(gguf_path) = std::env::var_os("VOKRA_KOKORO_GGUF") else {
            eprintln!(
                "[kokoro::mod::prosody_speed_1_5_shortens_frame_count_on_real_gguf_gated] \
                 SKIP: set VOKRA_KOKORO_GGUF to a converted Kokoro-82M voice GGUF."
            );
            return;
        };
        let tts = KokoroTts::from_path(&gguf_path)
            .unwrap_or_else(|e| panic!("load VOKRA_KOKORO_GGUF = {gguf_path:?}: {e}"));
        let style = vec![0.0f32; tts.config().style_dim];
        // 8 in-range ids so the sigmoid-sum is comfortably above 1 for
        // every phoneme; the sped-up call must still round to fewer frames.
        let ids = [1i64, 2, 3, 4, 5, 6, 7, 8];
        let baseline = tts
            .synthesize_phonemes(&ids, None, Some(&style), 0.0, 1.0)
            .expect("baseline synthesize");
        let sped_up = tts
            .synthesize_phonemes_with_prosody(
                &ids,
                None,
                Some(&style),
                0.0,
                1.0,
                Some(&ProsodyControl::default().with_speed_scale(1.5)),
            )
            .expect("sped-up synthesize");
        assert!(
            sped_up.samples.len() < baseline.samples.len(),
            "speed_scale = 1.5 must shorten the output (baseline len {}, sped-up len {})",
            baseline.samples.len(),
            sped_up.samples.len()
        );
        assert_eq!(sped_up.sample_rate, baseline.sample_rate);
    }

    /// A caller-side `pitch_shift` must alter the PCM output. Frame count is
    /// preserved (pitch axis touches F0 only, never durations). Skipped when
    /// the env var is unset.
    #[test]
    fn prosody_pitch_shift_alters_pcm_on_real_gguf_gated() {
        let Some(gguf_path) = std::env::var_os("VOKRA_KOKORO_GGUF") else {
            eprintln!(
                "[kokoro::mod::prosody_pitch_shift_alters_pcm_on_real_gguf_gated] \
                 SKIP: set VOKRA_KOKORO_GGUF to a converted Kokoro-82M voice GGUF."
            );
            return;
        };
        let tts = KokoroTts::from_path(&gguf_path)
            .unwrap_or_else(|e| panic!("load VOKRA_KOKORO_GGUF = {gguf_path:?}: {e}"));
        let style = vec![0.0f32; tts.config().style_dim];
        let ids = [1i64, 2, 3, 4, 5, 6, 7, 8];
        let neutral = tts
            .synthesize_phonemes(&ids, None, Some(&style), 0.0, 1.0)
            .expect("neutral synthesize");
        let pitched = tts
            .synthesize_phonemes_with_prosody(
                &ids,
                None,
                Some(&style),
                0.0,
                1.0,
                // 4 semitones ≈ major third; well within any perceptual band
                // but a decisive enough factor that non-linear decoder output
                // must diverge from the neutral baseline.
                Some(&ProsodyControl::default().with_pitch_shift(4.0)),
            )
            .expect("pitched synthesize");
        assert_eq!(
            pitched.samples.len(),
            neutral.samples.len(),
            "pitch_shift must not alter frame count (pitch axis touches F0 only)"
        );
        assert_ne!(
            pitched.samples, neutral.samples,
            "pitch_shift must alter PCM (F0 scaling must reach the decoder output)"
        );
        assert_eq!(pitched.sample_rate, neutral.sample_rate);
    }

    /// Un-honoured axes must be refused loudly at the entry point too — not
    /// just inside the resolver — so the error surfaces before any tensor
    /// work runs. This test doesn't need a real GGUF because it exercises
    /// the resolver's error path at the entry, but we still need a loaded
    /// `KokoroTts` to call it; use the same gate as the other integration
    /// tests so it skips cleanly on hosts without the fixture.
    #[test]
    fn synthesize_with_prosody_rejects_pause_ms_on_real_gguf_gated() {
        let Some(gguf_path) = std::env::var_os("VOKRA_KOKORO_GGUF") else {
            eprintln!(
                "[kokoro::mod::synthesize_with_prosody_rejects_pause_ms_on_real_gguf_gated] \
                 SKIP: set VOKRA_KOKORO_GGUF to a converted Kokoro-82M voice GGUF."
            );
            return;
        };
        let tts = KokoroTts::from_path(&gguf_path)
            .unwrap_or_else(|e| panic!("load VOKRA_KOKORO_GGUF = {gguf_path:?}: {e}"));
        let style = vec![0.0f32; tts.config().style_dim];
        let ids = [0i64, 1, 2, 3];
        let ctrl = ProsodyControl::default().with_pause_ms(200);
        match tts.synthesize_phonemes_with_prosody(&ids, None, Some(&style), 0.0, 1.0, Some(&ctrl))
        {
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(
                    msg.contains("pause_ms"),
                    "entry-point rejection must name the un-honoured axis; got: {msg}"
                );
            }
            other => {
                panic!("un-honoured pause_ms must be rejected loudly at the entry, got: {other:?}")
            }
        }
    }
}
