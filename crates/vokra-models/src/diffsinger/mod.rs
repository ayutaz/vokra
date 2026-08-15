//! **DiffSinger** — singing voice **synthesis** (SVS) runtime binder
//! (Wave D 2026-08-15, the **first singing-voice entry in the whole Vokra
//! catalogue**). Loud-partial per the panns / redimnet / storm / musicgen /
//! emotion2vec precedent — CLAUDE.md 教訓 (a): "loud-partial は
//! fake-complete より honest".
//!
//! # Scope — score-to-singing, NOT singing-voice conversion. Do not relocate.
//!
//! **Read this before moving or deleting this module.** DiffSinger is
//! **score-to-singing**: lyrics *already converted to phonemes*, a per-note
//! MIDI pitch, and per-phoneme durations go in; singing comes out. **There
//! is no source singer recording anywhere in the signal path.**
//!
//! That makes it categorically different from singing-voice **conversion**
//! (SVC) and from RVC, which take *an existing recording of a real,
//! identifiable person* and re-timbre it onto another identity. Per
//! CLAUDE.md 設計判断 8, the Tennessee ELVIS Act (2024-07-01) and the
//! federal NO FAKES Act attach liability under a "primary purpose or
//! effect" test aimed squarely at cloning a specific person's voice — and
//! that is why RVC v2 / GPT-SoVITS and every other voice-clone **trigger**
//! model are confined to the separate `vokra-voiceclone-experimental`
//! repository.
//!
//! DiffSinger is **not** such a trigger. A written melody is not a
//! person's voice; rendering one through a trained voicebank is the
//! singing analogue of ordinary TTS (piper-plus, Kokoro), both of which
//! live in this repo. **DiffSinger therefore belongs in `ayutaz/vokra` and
//! must not be moved to the voiceclone repo.** Note the symmetry with the
//! landed F0 family (`rmvpe` / `fcpe` / `crepe`), which stayed in-repo for
//! exactly the same reason: they are pitch *primitives*, not identity
//! triggers. The ELVIS Act boundary follows the *trigger model*, not the
//! surrounding capability.
//!
//! # Primary sources
//!
//! - Reference code: <https://github.com/openvpi/DiffSinger> (Apache-2.0
//!   per the upstream README, "licensed under the Apache 2.0 License",
//!   fetched 2026-08-15; the actively-maintained community fork, 3184
//!   stars per the GitHub API at scout time — CLAUDE.md
//!   「ハルシネーション厳禁」)
//! - Paper: Liu, Li, Ren, Chen & Zhao 2021, *"DiffSinger: Singing Voice
//!   Synthesis via Shallow Diffusion Mechanism"*
//!   (<https://arxiv.org/abs/2105.02446>)
//!
//! # Architecture (transcribed from the primary sources)
//!
//! Verbatim from the abstract: DiffSinger "is a parameterized Markov chain
//! that iteratively converts the noise into mel-spectrogram conditioned on
//! the music score". The **shallow diffusion mechanism** — the paper's
//! title contribution — has the model "start generation at a shallow step
//! smaller than the total number of diffusion steps, according to the
//! intersection of the diffusion trajectories of the ground-truth
//! mel-spectrogram and the one predicted by a simple mel-spectrogram
//! decoder". That shallow start step is upstream's `K_step: 400` against
//! `timesteps: 1000`.
//!
//! ```text
//! Score { phoneme, midi_pitch, duration } x N          ← [`Score`], REAL (validated here)
//!   -> FFT-block phoneme encoder                        ← **loud-partial**
//!        (enc_layers=4, num_heads=2, hidden=384)
//!   -> score-to-frame duration expansion                ← REAL ([`Score::total_frames`],
//!        (per-phoneme seconds -> frame grid)               via `vokra_ops::length_conditioning`)
//!   -> shallow-diffusion denoiser backbone               ← **loud-partial**
//!        (`backbone_type: lynxnet2`, start at K_step=400
//!         of timesteps=1000, linear β schedule, DDIM)
//!   -> **mel-spectrogram** (n_mels=128 @ 44.1 kHz, hop 512)
//!   ==== hand-off boundary — DiffSinger stops here ====
//!   -> separate neural vocoder -> waveform
//! ```
//!
//! # The vocoder half is already landed — DiffSinger hands off, it does not embed
//!
//! The upstream README lists **HiFi-GAN, NSF and pc-ddsp** as
//! *interchangeable* vocoder options operating as separate components from
//! the synthesis models. Vokra already carries that half as first-class
//! binders: [`crate::hifigan`], [`crate::bigvgan`] and [`crate::vocos`].
//!
//! So [`DiffSinger::synthesize_mel`] is deliberately named for what it
//! returns: **a mel-spectrogram, not a waveform**. Embedding a vocoder
//! inside this arch would duplicate three landed binders *and* hard-wire a
//! choice the upstream project deliberately leaves free to the voicebank
//! author. The `vokra.diffsinger.{n_mels, sample_rate, hop}` axes are the
//! hand-off contract a downstream vocoder binder checks compatibility
//! against.
//!
//! # Phonemes in, not lyrics in — no G2P is invented here
//!
//! [`ScoreNote::phoneme`] takes a **phoneme symbol**, verbatim from the
//! caller's own dictionary. This binder deliberately does **not** ship a
//! lyric-to-phoneme (G2P) front-end, and must not grow one by accident:
//!
//! - singing G2P is *per-voicebank*, not per-language — a DiffSinger
//!   voicebank ships its own phoneme dictionary, and the same lyric maps to
//!   different symbol sets across voicebanks;
//! - inventing a G2P would silently mis-pronounce every voicebank whose
//!   dictionary disagreed with the guess, which is exactly the class of
//!   silent-wrong failure FR-EX-08 exists to prevent;
//! - CLAUDE.md's standing G2P posture (M4-09 ADR Accepted 2026-07-22,
//!   "流用継続") is to reuse piper-plus's existing implementation for
//!   *speech*, never to author a new one — and singing is not even covered
//!   by that.
//!
//! Phoneme symbols are therefore carried through as opaque strings and
//! validated only for non-emptiness. Mapping a symbol to a voicebank
//! embedding index is a follow-up wave that needs the voicebank's real
//! dictionary.
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**:
//!   - [`DiffSinger::from_gguf`] with strict `vokra.model.arch ==
//!     "diffsinger"` validation. A sibling speech-TTS or vocoder GGUF
//!     handed here by mistake fails with a specific mis-route
//!     [`VokraError::ModelLoad`] naming both expected and actual arch.
//!   - [`DiffSingerConfig::from_gguf`] — a strict 19-axis parse of the
//!     `vokra.diffsinger.*` chunk group. Every axis is **required**; a
//!     missing key or a `0`-sentinel on a u32 axis is a loud
//!     [`VokraError::ModelLoad`], never a silent default (FR-EX-08).
//!     Structural invariants (`k_step < timesteps`, Nyquist, `f0_max >
//!     f0_min`) are enforced too — a config that violates them describes a
//!     model that cannot be the shallow-diffusion architecture.
//!   - [`DiffSingerWeights::from_gguf`] with a non-empty tensor gate.
//!   - [`Score`] / [`ScoreNote`] — the explicit score input type, with
//!     real validation (non-empty, finite, positive durations, MIDI pitch
//!     in range, non-empty phoneme symbols).
//!   - [`Score::total_frames`] — real score-to-frame expansion, computed
//!     through the landed [`vokra_ops::length_conditioning`] op against the
//!     config's `sample_rate` / `hop`.
//!   - Weight-license class surfacing (fail-closed to
//!     [`LicenseClass::Unknown`] when the stamp is absent).
//!
//! - **Loud-partial (this WP)**: [`DiffSinger::synthesize_mel`] returns
//!   [`VokraError::UnsupportedOp`] naming the deferred FFT-block phoneme
//!   encoder + LynxNet2 shallow-diffusion denoiser backbone, echoing both
//!   primary source URLs. **No fabricated mel-spectrogram is ever
//!   emitted** (FR-EX-08 — no silent partial output).
//!
//! # Reused Vokra ops (no new primitives invented)
//!
//! - [`vokra_ops::length_conditioning`] — score-to-frame expansion, wired
//!   for real in [`Score::total_frames`].
//! - [`vokra_ops::ddpm_sampler`] — the diffusion loop the follow-up wave
//!   drives. DiffSinger's `schedule_type: 'linear'` maps onto
//!   `BetaSchedule::Linear`, `timesteps: 1000` onto `num_train_steps`, and
//!   the shallow `K_step: 400` onto the reduced-step walk the sampler
//!   already supports. The missing piece is the *denoiser closure*, not
//!   the sampler.
//!
//! # Cross-crate constant duplication
//!
//! Mirror of the converter's [`ARCH`] / [`NAME`] / [`CATEGORY`] /
//! [`UPSTREAM_URL`] — same rule the sibling binders use so `vokra-models`
//! does not gain a dependency edge onto `vokra-convert`, preserving the
//! layered convention `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF
//! reader`, `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! DiffSinger ships PyTorch checkpoints upstream, and upstream *does*
//! publish an ONNX export path which Vokra deliberately does not consume at
//! runtime (CLAUDE.md 設計判断 2 / FR-LD-05 / NFR-DS-02). The `.ckpt` →
//! safetensors bridge lives offline through the sibling
//! `tools/parity/nemo_pt_to_safetensors.py` (uv-managed Python 3.12 per
//! memory `[[feedback-python-uses-uv]]` + `[[feedback-python-3-12]]`), not
//! part of the runtime.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::ir::graph::LengthConditioningAttrs;
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Contract constants — mirror of `crates/vokra-convert/src/models/diffsinger.rs`.
// See module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model diffsinger`.
///
/// Distinct from every sibling speech-TTS arch (`piper`, `kokoro`,
/// `cosyvoice2`, `qwen3_tts`, `xtts_v2`, `sbv2`, `vits`) and every vocoder
/// arch (`hifigan`, `bigvgan`, `vocos`, `nsf`). Silent aliasing would
/// misroute runtime dispatch to a wrong-topology loader (FR-EX-08).
pub const ARCH: &str = "diffsinger";

/// Expected `vokra.model.name` value written by the converter.
pub const NAME: &str = "diffsinger";

/// Expected `vokra.model.category` value — `svs`, **singing voice
/// synthesis**, a brand-new category in the Vokra catalogue.
///
/// Deliberately not `tts` (no text front-end; the input is a music score)
/// and deliberately not `vc` (no source recording — see the module
/// docstring for why that distinction is load-bearing).
pub const CATEGORY: &str = "svs";

/// Upstream GitHub tree (mirror of the converter's `UPSTREAM_URL`).
pub const UPSTREAM_URL: &str = "github.com/openvpi/DiffSinger";

/// Primary-source anchor for the reference code.
pub const PRIMARY_SOURCE_CODE: &str = "github.com/openvpi/DiffSinger";
/// Primary-source anchor for the paper (Liu et al. 2021).
pub const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2105.02446";

// ---- `vokra.diffsinger.*` metadata keys (mirror of the converter) --------

const KEY_DS_SAMPLE_RATE: &str = "vokra.diffsinger.sample_rate";
const KEY_DS_N_FFT: &str = "vokra.diffsinger.n_fft";
const KEY_DS_HOP: &str = "vokra.diffsinger.hop";
const KEY_DS_WIN_SIZE: &str = "vokra.diffsinger.win_size";
const KEY_DS_N_MELS: &str = "vokra.diffsinger.n_mels";
const KEY_DS_FMIN: &str = "vokra.diffsinger.fmin";
const KEY_DS_FMAX: &str = "vokra.diffsinger.fmax";
const KEY_DS_HIDDEN_SIZE: &str = "vokra.diffsinger.hidden_size";
const KEY_DS_ENC_LAYERS: &str = "vokra.diffsinger.enc_layers";
const KEY_DS_NUM_HEADS: &str = "vokra.diffsinger.num_heads";
const KEY_DS_TIMESTEPS: &str = "vokra.diffsinger.timesteps";
const KEY_DS_K_STEP: &str = "vokra.diffsinger.k_step";
const KEY_DS_F0_MIN: &str = "vokra.diffsinger.f0_min";
const KEY_DS_F0_MAX: &str = "vokra.diffsinger.f0_max";
const KEY_DS_MAX_BETA: &str = "vokra.diffsinger.max_beta";
const KEY_DS_MEL_VMIN: &str = "vokra.diffsinger.mel_vmin";
const KEY_DS_MEL_VMAX: &str = "vokra.diffsinger.mel_vmax";
const KEY_DS_SCHEDULE_TYPE: &str = "vokra.diffsinger.schedule_type";
const KEY_DS_DIFF_ACCELERATOR: &str = "vokra.diffsinger.diff_accelerator";
const KEY_DS_BACKBONE_TYPE: &str = "vokra.diffsinger.backbone_type";

/// The lowest MIDI note number (`0` = C-1) accepted by
/// [`ScoreNote::validate`]. The MIDI 1.0 specification defines note
/// numbers over `0..=127`; anything outside is a caller bug, not a
/// pitch to clamp.
pub const MIDI_NOTE_MIN: f32 = 0.0;
/// The highest MIDI note number (`127` = G9) accepted by
/// [`ScoreNote::validate`].
pub const MIDI_NOTE_MAX: f32 = 127.0;

// ---------------------------------------------------------------------------
// DiffSingerConfig — strict 19-axis parse of the `vokra.diffsinger.*` group.
// ---------------------------------------------------------------------------

/// Topology read from the `vokra.diffsinger.*` chunk group.
///
/// **Every axis is required.** [`from_gguf`](Self::from_gguf) refuses a
/// missing key and refuses a `0` on any u32 axis rather than substituting
/// a default (FR-EX-08): a GGUF that does not describe its own topology is
/// a mis-produced artifact, and silently filling in the upstream defaults
/// would make a variant voicebank load as if it were the reference config.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffSingerConfig {
    /// PCM sample rate (Hz). Upstream `audio_sample_rate: 44100` — note
    /// the openvpi fork's README calls out 44.1 kHz as an improvement over
    /// the original paper's 24 kHz, so a reader must not assume the
    /// paper's rate.
    pub sample_rate: u32,
    /// STFT FFT length. Upstream `fft_size: 2048`.
    pub n_fft: u32,
    /// STFT hop in samples. Upstream `hop_size: 512`.
    pub hop: u32,
    /// STFT window length. Upstream `win_size: 2048`.
    pub win_size: u32,
    /// Mel bin count — the width of the emitted mel. Upstream
    /// `audio_num_mel_bins: 128`. Half of the mel hand-off contract a
    /// downstream vocoder binder validates against.
    pub n_mels: u32,
    /// Mel filterbank low edge (Hz). Upstream `fmin: 40`.
    pub fmin: u32,
    /// Mel filterbank high edge (Hz). Upstream `fmax: 16000`.
    pub fmax: u32,
    /// Encoder / denoiser hidden width. Upstream `hidden_size: 384`.
    pub hidden_size: u32,
    /// FFT-block phoneme encoder layer count. Upstream `enc_layers: 4`.
    pub enc_layers: u32,
    /// Encoder attention head count. Upstream `num_heads: 2`.
    pub num_heads: u32,
    /// Total diffusion timesteps the model was trained on. Upstream
    /// `timesteps: 1000`. Maps onto
    /// `vokra_ops::ddpm_sampler::DdpmSamplerConfig::num_train_steps`.
    pub timesteps: u32,
    /// **Shallow** diffusion start step. Upstream `K_step: 400`. This is
    /// the defining knob of arXiv:2105.02446 — generation starts here
    /// rather than at the full `timesteps`, which is what "shallow
    /// diffusion" means. Enforced `< timesteps`.
    pub k_step: u32,
    /// F0 search floor (Hz). Upstream `f0_min: 65`.
    pub f0_min: u32,
    /// F0 search ceiling (Hz). Upstream `f0_max: 1100` — far above the
    /// speech-TTS range, because singing covers soprano registers.
    pub f0_max: u32,
    /// Diffusion beta ceiling. Upstream `max_beta: 0.02`.
    pub max_beta: f32,
    /// Mel dynamic-range floor. Upstream `mel_vmin: -14.`.
    pub mel_vmin: f32,
    /// Mel dynamic-range ceiling. Upstream `mel_vmax: 4.`.
    pub mel_vmax: f32,
    /// Beta schedule name. Upstream `schedule_type: 'linear'` — maps onto
    /// `vokra_ops::ddpm_sampler::BetaSchedule::Linear`.
    pub schedule_type: String,
    /// Diffusion accelerator name. Upstream `diff_accelerator: ddim`.
    pub diff_accelerator: String,
    /// Denoiser backbone name. Upstream `backbone_type: 'lynxnet2'`.
    pub backbone_type: String,
}

/// Reads a required, non-zero `u32` axis or fails loudly.
fn required_u32(file: &GgufFile, key: &str) -> Result<u32> {
    let raw = file.get(key).and_then(|v| v.as_u64()).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "diffsinger: GGUF is missing required topology key `{key}` — the \
             `vokra.diffsinger.*` chunk group is mandatory so the artifact is \
             self-describing. Re-run `vokra-cli convert --model diffsinger` \
             (FR-EX-08 — no silent default)."
        ))
    })?;
    if raw == 0 {
        return Err(VokraError::ModelLoad(format!(
            "diffsinger: topology key `{key}` is 0 — a zero on any topology axis \
             is a mis-produced GGUF, not a value to substitute a default for \
             (FR-EX-08)."
        )));
    }
    u32::try_from(raw).map_err(|_| {
        VokraError::ModelLoad(format!(
            "diffsinger: topology key `{key}` = {raw} overflows u32 — mis-produced GGUF."
        ))
    })
}

/// Reads a required, finite `f32` axis or fails loudly. Unlike
/// [`required_u32`] a zero is legal here (`mel_vmax` could in principle be
/// 0), so only presence and finiteness are enforced.
fn required_f32(file: &GgufFile, key: &str) -> Result<f32> {
    let raw = file.get(key).and_then(|v| v.as_f64()).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "diffsinger: GGUF is missing required topology key `{key}` — the \
             `vokra.diffsinger.*` chunk group is mandatory (FR-EX-08 — no silent default)."
        ))
    })?;
    let v = raw as f32;
    if !v.is_finite() {
        return Err(VokraError::ModelLoad(format!(
            "diffsinger: topology key `{key}` is non-finite ({raw}) — mis-produced GGUF."
        )));
    }
    Ok(v)
}

/// Reads a required, non-empty string axis or fails loudly.
fn required_str(file: &GgufFile, key: &str) -> Result<String> {
    let s = file.get(key).and_then(|v| v.as_str()).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "diffsinger: GGUF is missing required topology key `{key}` — the \
             `vokra.diffsinger.*` chunk group is mandatory (FR-EX-08 — no silent default)."
        ))
    })?;
    if s.is_empty() {
        return Err(VokraError::ModelLoad(format!(
            "diffsinger: topology key `{key}` is an empty string — mis-produced GGUF."
        )));
    }
    Ok(s.to_owned())
}

impl DiffSingerConfig {
    /// Strictly parses the 19-axis `vokra.diffsinger.*` chunk group.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] when any axis is missing, when a u32 axis
    /// is `0`, when an f32 axis is non-finite, when a string axis is
    /// empty, or when a structural invariant is violated
    /// (`k_step >= timesteps`, `fmax * 2 > sample_rate`,
    /// `f0_max <= f0_min`).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let cfg = Self {
            sample_rate: required_u32(file, KEY_DS_SAMPLE_RATE)?,
            n_fft: required_u32(file, KEY_DS_N_FFT)?,
            hop: required_u32(file, KEY_DS_HOP)?,
            win_size: required_u32(file, KEY_DS_WIN_SIZE)?,
            n_mels: required_u32(file, KEY_DS_N_MELS)?,
            fmin: required_u32(file, KEY_DS_FMIN)?,
            fmax: required_u32(file, KEY_DS_FMAX)?,
            hidden_size: required_u32(file, KEY_DS_HIDDEN_SIZE)?,
            enc_layers: required_u32(file, KEY_DS_ENC_LAYERS)?,
            num_heads: required_u32(file, KEY_DS_NUM_HEADS)?,
            timesteps: required_u32(file, KEY_DS_TIMESTEPS)?,
            k_step: required_u32(file, KEY_DS_K_STEP)?,
            f0_min: required_u32(file, KEY_DS_F0_MIN)?,
            f0_max: required_u32(file, KEY_DS_F0_MAX)?,
            max_beta: required_f32(file, KEY_DS_MAX_BETA)?,
            mel_vmin: required_f32(file, KEY_DS_MEL_VMIN)?,
            mel_vmax: required_f32(file, KEY_DS_MEL_VMAX)?,
            schedule_type: required_str(file, KEY_DS_SCHEDULE_TYPE)?,
            diff_accelerator: required_str(file, KEY_DS_DIFF_ACCELERATOR)?,
            backbone_type: required_str(file, KEY_DS_BACKBONE_TYPE)?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Structural invariants that must hold for the config to describe a
    /// shallow-diffusion singing acoustic model at all.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] on any violation.
    pub fn validate(&self) -> Result<()> {
        if self.k_step >= self.timesteps {
            return Err(VokraError::ModelLoad(format!(
                "diffsinger: k_step ({}) must be strictly less than timesteps ({}) — \
                 the SHALLOW diffusion mechanism of arXiv:2105.02446 is defined by \
                 starting generation at a step *smaller* than the total number of \
                 diffusion steps. k_step >= timesteps describes a model where the \
                 mechanism is not engaged, so this GGUF cannot be a DiffSinger \
                 acoustic model (FR-EX-08).",
                self.k_step, self.timesteps
            )));
        }
        if self.fmax.saturating_mul(2) > self.sample_rate {
            return Err(VokraError::ModelLoad(format!(
                "diffsinger: mel fmax ({} Hz) exceeds the Nyquist limit of \
                 sample_rate ({} Hz) — mis-produced GGUF (FR-EX-08).",
                self.fmax, self.sample_rate
            )));
        }
        if self.f0_max <= self.f0_min {
            return Err(VokraError::ModelLoad(format!(
                "diffsinger: f0_max ({}) must exceed f0_min ({}) — mis-produced GGUF.",
                self.f0_max, self.f0_min
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Score input type — phonemes + per-note MIDI pitch + durations.
// ---------------------------------------------------------------------------

/// One scored note: a phoneme, the MIDI pitch it is sung at, and how long
/// it lasts.
///
/// # Phoneme, not lyric
///
/// [`phoneme`](Self::phoneme) is a **phoneme symbol taken verbatim from
/// the caller's own dictionary** — this binder ships no lyric-to-phoneme
/// (G2P) front-end and must not grow one (see the module docstring: singing
/// G2P is per-voicebank, and guessing it would silently mis-pronounce every
/// voicebank whose dictionary disagreed). The symbol is carried through
/// opaquely and validated only for non-emptiness; mapping it to a voicebank
/// embedding index needs the voicebank's real dictionary and is a follow-up
/// wave.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreNote {
    /// Phoneme symbol, verbatim from the caller's dictionary. Opaque to
    /// this binder — never parsed, never guessed at.
    pub phoneme: String,
    /// MIDI note number the phoneme is sung at. `f32` rather than integer
    /// so cents-level detune, portamento and vibrato targets survive:
    /// `60.5` is a quarter-tone above middle C. Must lie within
    /// [`MIDI_NOTE_MIN`]..=[`MIDI_NOTE_MAX`].
    pub midi_pitch: f32,
    /// How long the phoneme lasts, in seconds. Must be finite and
    /// strictly positive — a zero-length note is a caller bug, not a rest
    /// (rests are expressed as a note whose phoneme symbol is the
    /// voicebank's own silence symbol).
    pub duration_seconds: f32,
}

impl ScoreNote {
    /// Validates a single note.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on an empty phoneme symbol, a
    /// non-finite or out-of-range `midi_pitch`, or a non-finite /
    /// non-positive `duration_seconds`.
    pub fn validate(&self, index: usize) -> Result<()> {
        if self.phoneme.is_empty() {
            return Err(VokraError::InvalidArgument(format!(
                "diffsinger: score note {index} has an empty phoneme symbol — \
                 DiffSinger takes phonemes, not lyrics, and this binder ships no \
                 G2P front-end (see the module docs). Supply the symbol from your \
                 voicebank's own phoneme dictionary."
            )));
        }
        if !self.midi_pitch.is_finite() {
            return Err(VokraError::InvalidArgument(format!(
                "diffsinger: score note {index} (`{}`) has a non-finite midi_pitch ({})",
                self.phoneme, self.midi_pitch
            )));
        }
        if self.midi_pitch < MIDI_NOTE_MIN || self.midi_pitch > MIDI_NOTE_MAX {
            return Err(VokraError::InvalidArgument(format!(
                "diffsinger: score note {index} (`{}`) has midi_pitch {} outside the \
                 MIDI 1.0 note range {MIDI_NOTE_MIN}..={MIDI_NOTE_MAX} — refusing to \
                 clamp a caller bug into a plausible-sounding pitch (FR-EX-08).",
                self.phoneme, self.midi_pitch
            )));
        }
        if !self.duration_seconds.is_finite() || self.duration_seconds <= 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "diffsinger: score note {index} (`{}`) has duration_seconds {} — must \
                 be finite and strictly positive. A rest is expressed as a note \
                 carrying the voicebank's silence phoneme, not as a zero-length note.",
                self.phoneme, self.duration_seconds
            )));
        }
        Ok(())
    }
}

/// A music score: the complete input to singing voice synthesis.
///
/// This is the *whole* input surface — there is no audio input anywhere,
/// which is precisely why DiffSinger is not an ELVIS Act voice-clone
/// trigger (see the module docstring).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Score {
    /// The note sequence, in performance order.
    pub notes: Vec<ScoreNote>,
}

impl Score {
    /// Builds a score from a note sequence.
    #[must_use]
    pub fn new(notes: Vec<ScoreNote>) -> Self {
        Self { notes }
    }

    /// Validates the whole score.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] when the score is empty or any note
    /// fails [`ScoreNote::validate`].
    pub fn validate(&self) -> Result<()> {
        if self.notes.is_empty() {
            return Err(VokraError::InvalidArgument(
                "diffsinger: score is empty — singing voice synthesis needs at least \
                 one scored note (phoneme + midi_pitch + duration)."
                    .to_owned(),
            ));
        }
        for (i, n) in self.notes.iter().enumerate() {
            n.validate(i)?;
        }
        Ok(())
    }

    /// Total scored duration in seconds.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] when the score fails
    /// [`validate`](Self::validate), or when the accumulated duration
    /// overflows to a non-finite value.
    pub fn total_seconds(&self) -> Result<f32> {
        self.validate()?;
        let total: f32 = self.notes.iter().map(|n| n.duration_seconds).sum();
        if !total.is_finite() {
            return Err(VokraError::InvalidArgument(format!(
                "diffsinger: total scored duration overflowed to {total} — the score \
                 carries {} notes whose durations sum out of range.",
                self.notes.len()
            )));
        }
        Ok(total)
    }

    /// **Score-to-frame expansion** — the number of mel frames this score
    /// occupies at the model's `sample_rate` / `hop`.
    ///
    /// This is real work, not a placeholder: it routes the scored duration
    /// through the landed [`vokra_ops::length_conditioning`] op
    /// (`frames = round(seconds · sample_rate / hop)`), which is the same
    /// op the Flow Matching TTS path uses for full-length generation. The
    /// follow-up denoiser wave consumes this frame count as the shape of
    /// the mel it must produce.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] when the score fails
    /// [`validate`](Self::validate), or propagated from
    /// [`vokra_ops::length_conditioning`] when the derived frame count is
    /// out of range.
    pub fn total_frames(&self, config: &DiffSingerConfig) -> Result<u32> {
        let seconds = self.total_seconds()?;
        let attrs = LengthConditioningAttrs::user_specified_seconds(
            seconds,
            config.sample_rate,
            config.hop,
        );
        vokra_ops::length_conditioning(&attrs)
    }
}

// ---------------------------------------------------------------------------
// DiffSingerWeights — non-empty tensor gate.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a DiffSinger GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification
/// step. A GGUF that carries zero tensors is rejected with
/// [`VokraError::ModelLoad`] (FR-EX-08) rather than silently running an
/// all-zero forward.
#[derive(Debug)]
pub struct DiffSingerWeights {
    /// Tensors discovered on disk, indexed by upstream `state_dict` name
    /// with their GGUF-side dims. Consumed by the load-time non-emptiness
    /// gate and by the follow-up FFT-encoder + LynxNet2 denoiser wave.
    tensors: Vec<(String, Vec<usize>)>,
}

impl DiffSingerWeights {
    /// Scans `gguf` for the DiffSinger state_dict tensors. Refuses to bind
    /// if the GGUF carries zero tensors.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] when the GGUF carries zero tensors.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let mut tensors: Vec<(String, Vec<usize>)> = Vec::new();
        for info in gguf.tensors() {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            tensors.push((info.name.clone(), dims));
        }
        if tensors.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "diffsinger: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). A legitimate DiffSinger voicebank carries the \
                 FFT-block phoneme encoder plus the diffusion denoiser backbone, \
                 hundreds of parameters at minimum (arch={ARCH}, name={NAME}); zero \
                 tensors always signals a mis-produced GGUF. Re-run `vokra-cli \
                 convert --model diffsinger` against an upstream `{UPSTREAM_URL}` \
                 checkpoint."
            )));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF. Diagnostic accessor — the
    /// follow-up forward wave uses it to size its expectations.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

// ---------------------------------------------------------------------------
// DiffSinger — the runtime binder handle.
// ---------------------------------------------------------------------------

/// DiffSinger (`openvpi/DiffSinger`, Apache-2.0) runtime binder for
/// **singing voice synthesis** (score-to-singing).
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`synthesize_mel`](Self::synthesize_mel) with a [`Score`] to obtain a
/// mel-spectrogram — which is then handed to a **separate** vocoder binder
/// ([`crate::hifigan`] / [`crate::bigvgan`] / [`crate::vocos`]) to get a
/// waveform. See the module doc for the implementation-status matrix, the
/// FR-EX-08 loud-error contract on the deferred denoiser, and the ELVIS Act
/// scope note explaining why this module belongs in this repo.
#[derive(Debug)]
pub struct DiffSinger {
    config: DiffSingerConfig,
    // The bound weights are held (real, counted) but the FFT-block encoder
    // + LynxNet2 denoiser composition is a follow-up wave; the field is
    // deliberately `#[allow(dead_code)]` until it lands so a reader is not
    // misled by an unused field. Same posture as emotion2vec / panns /
    // storm / musicgen.
    #[allow(dead_code)]
    weights: DiffSingerWeights,
    weight_license: LicenseClass,
}

impl DiffSinger {
    /// Binds a DiffSinger GGUF: validates arch, strictly parses the
    /// 19-axis topology chunk group, discovers tensors, and surfaces the
    /// stamped weight-license class.
    ///
    /// Every failure is a distinct [`VokraError::ModelLoad`] naming the
    /// missing / wrong key so a reader diagnosing a mis-produced GGUF has
    /// exactly one place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent or is
    ///   not `"diffsinger"` (a sibling speech-TTS or vocoder GGUF handed
    ///   here by mistake fails with a clear message naming both expected
    ///   and actual arch).
    /// - [`VokraError::ModelLoad`] from [`DiffSingerConfig::from_gguf`]
    ///   when any topology axis is missing, zero, non-finite, empty, or
    ///   violates a structural invariant.
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed here
        //    fails with a specific message instead of a downstream
        //    missing-key error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "diffsinger: GGUF arch is `{other}`, expected `{ARCH}` (was this \
                     GGUF produced by `vokra-cli convert --model diffsinger`?). \
                     DiffSinger is a SINGING voice synthesis acoustic model: it is \
                     driven by a music score (phonemes + per-note MIDI pitch + \
                     durations) and emits a mel-spectrogram. Neighbouring arches are \
                     all incompatible with that contract — the speech-TTS family \
                     (`piper`, `kokoro`, `cosyvoice2`, `qwen3_tts`, `xtts_v2`, \
                     `sbv2`, `vits`) is driven by text and has no note/pitch score \
                     axis, while the vocoder family (`hifigan`, `bigvgan`, `vocos`, \
                     `nsf`) CONSUMES a mel rather than producing one and so is a \
                     downstream stage, not a substitute. Silently aliasing arch \
                     would misroute the runtime dispatch (FR-EX-08 — no silent \
                     partial load)."
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "diffsinger: GGUF is missing `vokra.model.arch` — this is not a \
                     Vokra-native diffsinger GGUF (was it produced by `vokra-cli \
                     convert --model diffsinger`?)"
                        .to_owned(),
                ));
            }
        }

        // 2. Strict topology parse — every axis required, no silent defaults.
        let config = DiffSingerConfig::from_gguf(file)?;

        // 3. Tensor manifest with the non-emptiness gate.
        let weights = DiffSingerWeights::from_gguf(file)?;

        // 4. Provenance surfacing. The converter stamps `Permissive` for
        //    the apache-2.0 framework grant; a voicebank converted with a
        //    `--license` override surfaces its own class. A GGUF missing
        //    the stamp reads back as `Unknown` (fail-closed per memory
        //    feedback-license-signoff-primary-source).
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        Ok(Self {
            config,
            weights,
            weight_license,
        })
    }

    /// The parsed topology.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &DiffSingerConfig {
        &self.config
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk.
    ///
    /// **Check this before redistributing a rendered performance.** The
    /// converter's apache-2.0 default is the *framework* grant from
    /// `openvpi/DiffSinger`; individual singer voicebanks are released by
    /// their own authors under their own, frequently non-commercial or
    /// consent-bound, terms. A GGUF missing the stamp reads back as
    /// [`LicenseClass::Unknown`] (fail-closed at the M2-13 compliance
    /// gate).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors bound from the GGUF. Diagnostic accessor.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Renders a [`Score`] to a **mel-spectrogram**.
    ///
    /// # Output contract — a mel, not a waveform
    ///
    /// The return value is a flattened mel of
    /// `config().n_mels × Score::total_frames(config())` values. Turning
    /// that into audio is a **separate** stage: hand it to
    /// [`crate::hifigan`], [`crate::bigvgan`] or [`crate::vocos`]. This
    /// split is upstream's own design — the openvpi README lists HiFi-GAN,
    /// NSF and pc-ddsp as interchangeable vocoder options — and embedding
    /// one here would duplicate three landed binders while hard-wiring a
    /// choice the voicebank author owns.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`]. The score is **validated
    /// first**, so a malformed score still fails with a precise
    /// score-level diagnostic rather than being masked by the deferral —
    /// callers can develop and test their score-building code against this
    /// binder today.
    ///
    /// The deferral itself is the FFT-block phoneme encoder plus the
    /// LynxNet2 shallow-diffusion denoiser backbone, which cannot be
    /// written from the README and paper abstract alone without guessing
    /// at topology — and a guessed topology is the silent-wrong failure
    /// mode CLAUDE.md 教訓 (a) exists to avoid. **No fabricated mel is
    /// ever emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] when `score` fails
    ///   [`Score::validate`] (checked before the deferral).
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred encoder + denoiser composition.
    pub fn synthesize_mel(&self, score: &Score) -> Result<Vec<f32>> {
        // Validate first: a caller with a malformed score deserves the
        // precise score-level error, not the deferral message. This also
        // means score-building code is testable against this binder today.
        score.validate()?;
        let frames = score.total_frames(&self.config)?;
        Err(synthesize_mel_loud_partial(&self.config, score, frames))
    }
}

/// Construct the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`DiffSinger::synthesize_mel`] until the FFT-block encoder + LynxNet2
/// shallow-diffusion denoiser composition lands.
///
/// Names both primary source URLs and the exact mel shape the follow-up
/// wave must produce, so a reader diagnosing the gap knows where to walk
/// and what to target. Mirror of the emotion2vec / panns / storm /
/// musicgen loud-partial-message precedent (CLAUDE.md 教訓 (a)).
fn synthesize_mel_loud_partial(
    config: &DiffSingerConfig,
    score: &Score,
    frames: u32,
) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "diffsinger synthesize_mel (loud-partial): the score was accepted \
         ({notes} notes, {frames} mel frames at {sr} Hz / hop {hop}) but the \
         acoustic forward is deferred; two missing pieces must land before a \
         real mel can be emitted: \
         (1) FFT-block phoneme encoder — enc_layers={enc_layers}, \
         num_heads={heads}, hidden_size={hidden}; \
         (2) shallow-diffusion denoiser backbone `{backbone}` driving the \
         landed `vokra_ops::ddpm_sampler` from the shallow start step \
         k_step={k_step} of timesteps={timesteps} on a `{schedule}` beta \
         schedule (max_beta={max_beta}) with the `{accel}` accelerator — the \
         sampler itself is landed and reusable, what is missing is the \
         denoiser closure it drives, whose topology cannot be transcribed \
         from the README and paper abstract without guessing. \
         Target output shape = n_mels {n_mels} x frames {frames} (mel range \
         [{vmin}, {vmax}]). Note the output is a MEL, not a waveform: the \
         vocoder stage is deliberately separate and already landed as the \
         `hifigan` / `bigvgan` / `vocos` binders (upstream lists HiFi-GAN / \
         NSF / pc-ddsp as interchangeable). Primary sources: reference code \
         {code}, paper {paper}. Runtime cannot fabricate a mel-spectrogram \
         (FR-EX-08 no silent partial output).",
        notes = score.notes.len(),
        frames = frames,
        sr = config.sample_rate,
        hop = config.hop,
        enc_layers = config.enc_layers,
        heads = config.num_heads,
        hidden = config.hidden_size,
        backbone = config.backbone_type,
        k_step = config.k_step,
        timesteps = config.timesteps,
        schedule = config.schedule_type,
        max_beta = config.max_beta,
        accel = config.diff_accelerator,
        n_mels = config.n_mels,
        vmin = config.mel_vmin,
        vmax = config.mel_vmax,
        code = PRIMARY_SOURCE_CODE,
        paper = PRIMARY_SOURCE_PAPER,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the DiffSinger runtime binder — contract-constant pins,
    //! strict-topology round-trip, real score validation, real
    //! score-to-frame expansion, and negative-space round-trip on every
    //! loud gate.
    //!
    //! # What is real vs deferred here
    //!
    //! The score type, its validation, and the score-to-frame expansion
    //! are **real** and tested as such — `total_frames` computes an actual
    //! frame count through the landed `vokra_ops::length_conditioning` op.
    //! The acoustic forward is deferred (see the module doc), so the mel
    //! itself is tested as a loud refusal rather than a fabricated array:
    //! CLAUDE.md 教訓 (a), "loud-partial は fake-complete より honest".

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    // -- fixtures ---------------------------------------------------------

    /// The upstream reference axes (`configs/acoustic.yaml` +
    /// `configs/base.yaml`, fetched 2026-08-15).
    fn upstream_axes() -> Vec<(&'static str, u32)> {
        vec![
            (KEY_DS_SAMPLE_RATE, 44_100),
            (KEY_DS_N_FFT, 2048),
            (KEY_DS_HOP, 512),
            (KEY_DS_WIN_SIZE, 2048),
            (KEY_DS_N_MELS, 128),
            (KEY_DS_FMIN, 40),
            (KEY_DS_FMAX, 16_000),
            (KEY_DS_HIDDEN_SIZE, 384),
            (KEY_DS_ENC_LAYERS, 4),
            (KEY_DS_NUM_HEADS, 2),
            (KEY_DS_TIMESTEPS, 1000),
            (KEY_DS_K_STEP, 400),
            (KEY_DS_F0_MIN, 65),
            (KEY_DS_F0_MAX, 1100),
        ]
    }

    /// Builds a synthetic DiffSinger GGUF with the upstream axes, one
    /// tensor, and the provenance stamps. `arch` and per-key overrides let
    /// individual tests poke holes in it.
    fn synth_gguf(
        arch: Option<&str>,
        with_tensor: bool,
        u32_overrides: &[(&str, u32)],
        drop_keys: &[&str],
    ) -> GgufFile {
        let mut b = GgufBuilder::new();
        if let Some(a) = arch {
            b.add_string(chunks::KEY_MODEL_ARCH, a);
        }
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string("vokra.model.category", CATEGORY);
        b.add_string(
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        );
        b.add_string(chunks::KEY_PROVENANCE_LICENSE, "apache-2.0");

        for (k, v) in upstream_axes() {
            if drop_keys.contains(&k) {
                continue;
            }
            let val = u32_overrides
                .iter()
                .find(|(ok, _)| *ok == k)
                .map_or(v, |(_, ov)| *ov);
            b.add_u32(k, val);
        }
        for (k, v) in [
            (KEY_DS_MAX_BETA, 0.02_f32),
            (KEY_DS_MEL_VMIN, -14.0),
            (KEY_DS_MEL_VMAX, 4.0),
        ] {
            if drop_keys.contains(&k) {
                continue;
            }
            b.add_f32(k, v);
        }
        for (k, v) in [
            (KEY_DS_SCHEDULE_TYPE, "linear"),
            (KEY_DS_DIFF_ACCELERATOR, "ddim"),
            (KEY_DS_BACKBONE_TYPE, "lynxnet2"),
        ] {
            if drop_keys.contains(&k) {
                continue;
            }
            b.add_string(k, v);
        }

        if with_tensor {
            let payload: Vec<u8> = [0.5_f32, -0.25]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            b.add_tensor(
                "fs2.encoder.layers.0.self_attn.in_proj_weight",
                GgmlType::F32,
                vec![1, 2],
                payload,
            )
            .expect("add tensor");
        }
        GgufFile::parse(b.to_bytes().expect("build gguf")).expect("parse gguf")
    }

    fn note(ph: &str, midi: f32, dur: f32) -> ScoreNote {
        ScoreNote {
            phoneme: ph.to_owned(),
            midi_pitch: midi,
            duration_seconds: dur,
        }
    }

    /// A short scored phrase. Phonemes are opaque symbols — this binder
    /// ships no G2P, so the caller's dictionary owns them.
    fn sample_score() -> Score {
        Score::new(vec![
            note("k", 60.0, 0.1),
            note("o", 60.0, 0.4),
            note("n", 62.0, 0.2),
            note("i", 64.0, 0.5),
        ])
    }

    // -- 1. contract-constant pins ---------------------------------------

    /// Cross-crate constant pin: `ARCH` / `NAME` / `CATEGORY` /
    /// `UPSTREAM_URL` must match the converter's values exactly. A
    /// converter-side drift without a binder follow-through lands here.
    #[test]
    fn contract_constants_are_pinned() {
        assert_eq!(ARCH, "diffsinger");
        assert_eq!(NAME, "diffsinger");
        assert_eq!(
            CATEGORY, "svs",
            "singing voice synthesis is its own category — not `tts` (no text \
             front-end; the input is a music score) and not `vc` (no source \
             recording, so not an ELVIS Act voice-clone trigger)"
        );
        assert_eq!(UPSTREAM_URL, "github.com/openvpi/DiffSinger");
        assert_eq!(PRIMARY_SOURCE_PAPER, "arxiv.org/abs/2105.02446");
    }

    /// Arch-tag distinctness pin against the speech-TTS and vocoder
    /// neighbourhoods (FR-EX-08).
    #[test]
    fn arch_is_distinct_from_speech_tts_and_vocoder_families() {
        for sibling in [
            "piper",
            "kokoro",
            "cosyvoice2",
            "qwen3_tts",
            "xtts_v2",
            "sbv2",
            "vits",
            "hifigan",
            "bigvgan",
            "vocos",
            "nsf",
            "storm",
            "vae_continuous",
        ] {
            assert_ne!(
                ARCH, sibling,
                "diffsinger (score-to-singing shallow-diffusion acoustic model) and \
                 `{sibling}` are distinct arches — sharing a tag would misroute \
                 runtime dispatch (FR-EX-08)"
            );
        }
    }

    // -- 2. metadata round-trip ------------------------------------------

    /// A well-formed synthetic GGUF binds, and every one of the 19
    /// topology axes round-trips to the upstream-transcribed value.
    #[test]
    fn synthetic_gguf_binds_and_all_axes_round_trip() {
        let file = synth_gguf(Some(ARCH), true, &[], &[]);
        let m = DiffSinger::from_gguf(&file).expect("well-formed GGUF must bind");
        let c = m.config();

        assert_eq!(c.sample_rate, 44_100, "upstream audio_sample_rate");
        assert_eq!(c.n_fft, 2048, "upstream fft_size");
        assert_eq!(c.hop, 512, "upstream hop_size");
        assert_eq!(c.win_size, 2048, "upstream win_size");
        assert_eq!(c.n_mels, 128, "upstream audio_num_mel_bins");
        assert_eq!(c.fmin, 40, "upstream fmin");
        assert_eq!(c.fmax, 16_000, "upstream fmax");
        assert_eq!(c.hidden_size, 384, "upstream hidden_size");
        assert_eq!(c.enc_layers, 4, "upstream enc_layers");
        assert_eq!(c.num_heads, 2, "upstream num_heads");
        assert_eq!(c.timesteps, 1000, "upstream timesteps");
        assert_eq!(c.k_step, 400, "upstream K_step");
        assert_eq!(c.f0_min, 65, "upstream f0_min");
        assert_eq!(c.f0_max, 1100, "upstream f0_max");
        assert_eq!(c.max_beta, 0.02, "upstream max_beta");
        assert_eq!(c.mel_vmin, -14.0, "upstream mel_vmin");
        assert_eq!(c.mel_vmax, 4.0, "upstream mel_vmax");
        assert_eq!(c.schedule_type, "linear", "upstream schedule_type");
        assert_eq!(c.diff_accelerator, "ddim", "upstream diff_accelerator");
        assert_eq!(c.backbone_type, "lynxnet2", "upstream backbone_type");

        assert_eq!(m.tensor_count(), 1);
        assert_eq!(
            m.weight_license(),
            LicenseClass::Permissive,
            "apache-2.0 framework grant surfaces as Permissive"
        );
    }

    /// A GGUF with no `vokra.provenance.weight_license` stamp must read
    /// back as `Unknown` (fail-closed), never as an optimistic
    /// `Permissive`.
    #[test]
    fn missing_license_stamp_falls_back_to_unknown_fail_closed() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        for (k, v) in upstream_axes() {
            b.add_u32(k, v);
        }
        b.add_f32(KEY_DS_MAX_BETA, 0.02);
        b.add_f32(KEY_DS_MEL_VMIN, -14.0);
        b.add_f32(KEY_DS_MEL_VMAX, 4.0);
        b.add_string(KEY_DS_SCHEDULE_TYPE, "linear");
        b.add_string(KEY_DS_DIFF_ACCELERATOR, "ddim");
        b.add_string(KEY_DS_BACKBONE_TYPE, "lynxnet2");
        b.add_tensor("t", GgmlType::F32, vec![1], vec![0, 0, 0, 0])
            .expect("add tensor");
        let file = GgufFile::parse(b.to_bytes().expect("build")).expect("parse");

        let m = DiffSinger::from_gguf(&file).expect("bind without license stamp");
        assert_eq!(
            m.weight_license(),
            LicenseClass::Unknown,
            "an unstamped GGUF must fail closed to Unknown — a voicebank whose \
             terms we cannot read must never surface as redistributable"
        );
    }

    // -- 3. loud arch gates ----------------------------------------------

    /// Missing `vokra.model.arch` → loud `ModelLoad`.
    #[test]
    fn missing_arch_is_loud() {
        let file = synth_gguf(None, true, &[], &[]);
        let Err(err) = DiffSinger::from_gguf(&file) else {
            panic!("expected an error when `vokra.model.arch` is absent");
        };
        assert!(matches!(err, VokraError::ModelLoad(_)), "got {err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("vokra.model.arch"),
            "the error must name the missing key: {msg}"
        );
    }

    /// A foreign arch → loud `ModelLoad` naming **both** expected and
    /// actual, plus the neighbouring families so a reader understands why
    /// the two are not interchangeable.
    #[test]
    fn foreign_arch_is_loud_and_names_expected_and_actual() {
        // `hifigan` is the most dangerous confusion: it is the very stage
        // DiffSinger hands off to, so a mixed-up pair would look plausible.
        let file = synth_gguf(Some("hifigan"), true, &[], &[]);
        let Err(err) = DiffSinger::from_gguf(&file) else {
            panic!("expected an error when a foreign arch GGUF is handed to DiffSinger");
        };
        assert!(matches!(err, VokraError::ModelLoad(_)), "got {err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("hifigan"),
            "the error must name the ACTUAL arch found: {msg}"
        );
        assert!(
            msg.contains("diffsinger"),
            "the error must name the EXPECTED arch: {msg}"
        );
        assert!(
            msg.contains("CONSUMES a mel"),
            "the error must explain that a vocoder is a downstream stage, not a \
             substitute: {msg}"
        );
    }

    // -- 4. strict topology gates ----------------------------------------

    /// A missing topology axis → loud `ModelLoad` naming the key. No
    /// silent default is ever substituted.
    #[test]
    fn missing_topology_axis_is_loud_and_names_the_key() {
        for dropped in [
            KEY_DS_SAMPLE_RATE,
            KEY_DS_N_MELS,
            KEY_DS_K_STEP,
            KEY_DS_MAX_BETA,
            KEY_DS_BACKBONE_TYPE,
        ] {
            let file = synth_gguf(Some(ARCH), true, &[], &[dropped]);
            let Err(err) = DiffSinger::from_gguf(&file) else {
                panic!("expected an error when topology key `{dropped}` is absent");
            };
            assert!(matches!(err, VokraError::ModelLoad(_)), "got {err:?}");
            let msg = err.to_string();
            assert!(
                msg.contains(dropped),
                "the error must name the missing key `{dropped}`: {msg}"
            );
        }
    }

    /// A `0`-sentinel on a u32 topology axis → loud `ModelLoad`. Silently
    /// treating 0 as "use the default" would let a variant voicebank load
    /// as if it were the reference config.
    #[test]
    fn zero_sentinel_on_u32_axis_is_loud() {
        let file = synth_gguf(Some(ARCH), true, &[(KEY_DS_N_MELS, 0)], &[]);
        let Err(err) = DiffSinger::from_gguf(&file) else {
            panic!("expected an error when a u32 topology axis is 0");
        };
        assert!(matches!(err, VokraError::ModelLoad(_)), "got {err:?}");
        assert!(
            err.to_string().contains(KEY_DS_N_MELS),
            "the error must name the offending key"
        );
    }

    /// `k_step >= timesteps` → loud `ModelLoad`. This is the structural
    /// signature of the *shallow* diffusion mechanism: if generation does
    /// not start below the full step count, the paper's title contribution
    /// is not engaged and the GGUF cannot be a DiffSinger acoustic model.
    #[test]
    fn k_step_not_below_timesteps_is_loud() {
        let file = synth_gguf(
            Some(ARCH),
            true,
            &[(KEY_DS_K_STEP, 1000), (KEY_DS_TIMESTEPS, 1000)],
            &[],
        );
        let Err(err) = DiffSinger::from_gguf(&file) else {
            panic!("expected an error when k_step >= timesteps");
        };
        assert!(matches!(err, VokraError::ModelLoad(_)), "got {err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("k_step") && msg.contains("timesteps"),
            "the error must name both axes: {msg}"
        );
        assert!(
            msg.to_ascii_uppercase().contains("SHALLOW"),
            "the error must explain that this breaks the shallow diffusion \
             mechanism: {msg}"
        );
    }

    /// `fmax * 2 > sample_rate` → loud `ModelLoad` (Nyquist).
    #[test]
    fn fmax_above_nyquist_is_loud() {
        let file = synth_gguf(Some(ARCH), true, &[(KEY_DS_FMAX, 30_000)], &[]);
        let Err(err) = DiffSinger::from_gguf(&file) else {
            panic!("expected an error when fmax exceeds Nyquist");
        };
        assert!(matches!(err, VokraError::ModelLoad(_)), "got {err:?}");
        assert!(err.to_string().contains("Nyquist"), "got {err}");
    }

    // -- 5. empty tensor gate --------------------------------------------

    /// Zero tensors → loud `ModelLoad`, never an all-zero forward.
    #[test]
    fn empty_tensor_list_is_loud() {
        let file = synth_gguf(Some(ARCH), false, &[], &[]);
        let Err(err) = DiffSinger::from_gguf(&file) else {
            panic!("expected an error when the GGUF carries zero tensors");
        };
        assert!(matches!(err, VokraError::ModelLoad(_)), "got {err:?}");
        assert!(
            err.to_string().contains("zero tensors"),
            "the error must say the manifest is empty: {err}"
        );
    }

    // -- 6. score type: real validation ----------------------------------

    /// A well-formed score validates and its total duration is the exact
    /// sum of the note durations.
    #[test]
    fn well_formed_score_validates_and_sums_duration() {
        let s = sample_score();
        s.validate().expect("well-formed score must validate");
        let total = s.total_seconds().expect("total_seconds");
        // 0.1 + 0.4 + 0.2 + 0.5
        assert!(
            (total - 1.2).abs() < 1e-5,
            "total scored duration must be the sum of note durations, got {total}"
        );
    }

    /// An empty score is refused — SVS needs at least one scored note.
    #[test]
    fn empty_score_is_refused() {
        let s = Score::default();
        let Err(err) = s.validate() else {
            panic!("expected an error when the score carries no notes");
        };
        assert!(matches!(err, VokraError::InvalidArgument(_)), "got {err:?}");
    }

    /// An empty phoneme symbol is refused, and the error explains that
    /// this binder takes phonemes rather than lyrics (no G2P here).
    #[test]
    fn empty_phoneme_symbol_is_refused_and_explains_no_g2p() {
        let s = Score::new(vec![note("", 60.0, 0.2)]);
        let Err(err) = s.validate() else {
            panic!("expected an error for an empty phoneme symbol");
        };
        assert!(matches!(err, VokraError::InvalidArgument(_)), "got {err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("phonemes, not lyrics"),
            "the error must tell the caller this binder ships no G2P: {msg}"
        );
    }

    /// Out-of-range and non-finite MIDI pitches are refused rather than
    /// clamped — clamping would turn a caller bug into a plausible-sounding
    /// wrong note (FR-EX-08).
    #[test]
    fn out_of_range_midi_pitch_is_refused_not_clamped() {
        for bad in [-1.0_f32, 128.0, f32::NAN, f32::INFINITY] {
            let s = Score::new(vec![note("a", bad, 0.2)]);
            let Err(err) = s.validate() else {
                panic!("expected an error for midi_pitch {bad}");
            };
            assert!(
                matches!(err, VokraError::InvalidArgument(_)),
                "midi_pitch {bad} must be refused, got {err:?}"
            );
        }
        // Fractional pitches inside the range ARE legal — cents-level
        // detune / portamento targets must survive.
        Score::new(vec![note("a", 60.5, 0.2)])
            .validate()
            .expect("a fractional MIDI pitch is legal (quarter-tone / portamento)");
    }

    /// Zero and negative durations are refused; the error points the
    /// caller at the silence phoneme for rests.
    #[test]
    fn non_positive_duration_is_refused() {
        for bad in [0.0_f32, -0.5, f32::NAN] {
            let s = Score::new(vec![note("a", 60.0, bad)]);
            let Err(err) = s.validate() else {
                panic!("expected an error for duration_seconds {bad}");
            };
            assert!(
                matches!(err, VokraError::InvalidArgument(_)),
                "duration {bad} must be refused, got {err:?}"
            );
        }
    }

    // -- 7. score-to-frame expansion: real ------------------------------

    /// **Real** score-to-frame expansion through the landed
    /// `vokra_ops::length_conditioning` op:
    /// `frames = round(seconds · sample_rate / hop)`.
    ///
    /// At the upstream 44 100 Hz / hop 512, a 1.2 s phrase is
    /// `1.2 · 44100 / 512 = 103.36…` → 103 frames.
    #[test]
    fn score_to_frame_expansion_is_real() {
        let file = synth_gguf(Some(ARCH), true, &[], &[]);
        let m = DiffSinger::from_gguf(&file).expect("bind");
        let frames = sample_score()
            .total_frames(m.config())
            .expect("frame expansion must succeed for a valid score");
        // Computed from the op's documented formula, not from a run.
        assert_eq!(
            frames, 103,
            "1.2 s at 44100 Hz / hop 512 = 103.36 -> 103 frames"
        );
    }

    /// The expansion scales with the score: doubling the duration roughly
    /// doubles the frame count, and a longer score is strictly longer.
    #[test]
    fn frame_expansion_scales_with_scored_duration() {
        let file = synth_gguf(Some(ARCH), true, &[], &[]);
        let m = DiffSinger::from_gguf(&file).expect("bind");
        let short = Score::new(vec![note("a", 60.0, 1.0)])
            .total_frames(m.config())
            .expect("short");
        let long = Score::new(vec![note("a", 60.0, 2.0)])
            .total_frames(m.config())
            .expect("long");
        assert!(
            long > short,
            "a longer score must occupy more mel frames ({long} vs {short})"
        );
        assert_eq!(
            long,
            short * 2,
            "doubling the duration must double the frame count at a fixed hop"
        );
    }

    /// An invalid score fails at expansion time too — the frame count is
    /// never derived from a score that has not validated.
    #[test]
    fn frame_expansion_refuses_an_invalid_score() {
        let file = synth_gguf(Some(ARCH), true, &[], &[]);
        let m = DiffSinger::from_gguf(&file).expect("bind");
        let Err(err) = Score::default().total_frames(m.config()) else {
            panic!("expected an error when expanding an empty score");
        };
        assert!(matches!(err, VokraError::InvalidArgument(_)), "got {err:?}");
    }

    // -- 8. loud-partial forward ----------------------------------------

    /// The forward is loud-partial: it returns `UnsupportedOp` naming the
    /// two deferred pieces, the target mel shape, both primary sources,
    /// and the vocoder hand-off — and never a fabricated mel.
    #[test]
    fn synthesize_mel_is_loud_partial_and_names_the_gap() {
        let file = synth_gguf(Some(ARCH), true, &[], &[]);
        let m = DiffSinger::from_gguf(&file).expect("bind");
        let Err(err) = m.synthesize_mel(&sample_score()) else {
            panic!("expected the deferred acoustic forward to refuse, not fabricate a mel");
        };
        assert!(matches!(err, VokraError::UnsupportedOp(_)), "got {err:?}");
        let msg = err.to_string();
        for needle in [
            "loud-partial",
            "FFT-block phoneme encoder",
            "lynxnet2",
            "ddpm_sampler",
            "github.com/openvpi/DiffSinger",
            "arxiv.org/abs/2105.02446",
            "hifigan",
        ] {
            assert!(
                msg.contains(needle),
                "the loud-partial message must mention `{needle}`: {msg}"
            );
        }
        // It must state the shape the follow-up wave has to produce.
        assert!(
            msg.contains("128") && msg.contains("103"),
            "the message must name the target mel shape (n_mels x frames): {msg}"
        );
    }

    /// A malformed score fails with the **score-level** diagnostic, not
    /// the deferral message — so callers can develop score-building code
    /// against this binder today.
    #[test]
    fn synthesize_mel_validates_the_score_before_deferring() {
        let file = synth_gguf(Some(ARCH), true, &[], &[]);
        let m = DiffSinger::from_gguf(&file).expect("bind");
        let Err(err) = m.synthesize_mel(&Score::default()) else {
            panic!("expected an error for an empty score");
        };
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "an invalid score must produce a score-level InvalidArgument, not be \
             masked by the loud-partial UnsupportedOp: {err:?}"
        );
    }
}
