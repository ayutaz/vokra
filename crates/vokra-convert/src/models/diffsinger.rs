#![allow(clippy::doc_lazy_continuation)]
//! **DiffSinger** (`openvpi/DiffSinger`, **Apache-2.0**) — singing voice
//! **synthesis** (SVS): safetensors → GGUF conversion (Wave D 2026-08-15,
//! the **first singing-voice entry in the whole Vokra catalogue**).
//!
//! # Model class — score-to-singing (SVS), NOT singing-voice *conversion*
//!
//! **Read this before relocating or deleting this module.** DiffSinger is
//! **score-to-singing**: the inputs are a *music score* — a phoneme
//! sequence, a per-note MIDI pitch, and per-phoneme durations — and the
//! output is *singing*. There is **no source singer recording anywhere in
//! the signal path**.
//!
//! That makes it categorically different from singing-voice **conversion**
//! (SVC) / RVC, which take *an existing recording of a real person* and
//! re-timbre it onto another identity. Per CLAUDE.md 設計判断 8, the
//! ELVIS Act (Tennessee, 2024-07-01) and the federal NO FAKES Act attach
//! liability on a "primary purpose or effect" test aimed at *cloning a
//! specific identifiable voice*; that is why RVC v2 / GPT-SoVITS and the
//! rest of the voice-clone **trigger** models are confined to the separate
//! `vokra-voiceclone-experimental` repository.
//!
//! DiffSinger is **not** such a trigger. A score is not a person's voice:
//! synthesising a written melody through a trained voicebank is the
//! singing analogue of ordinary TTS (piper-plus / Kokoro), both of which
//! live in this repo. **DiffSinger therefore belongs in `ayutaz/vokra` and
//! must not be moved to the voiceclone repo.** (Note the symmetry with the
//! landed F0 family — `rmvpe` / `fcpe` / `crepe` — which stayed in-repo for
//! the same reason: they are pitch *primitives*, not identity triggers.)
//!
//! # Architecture — shallow-diffusion acoustic model → mel → **separate** vocoder
//!
//! Liu, Li, Ren, Chen & Zhao 2021, *"DiffSinger: Singing Voice Synthesis
//! via Shallow Diffusion Mechanism"* (arXiv:2105.02446). Verbatim from the
//! abstract: DiffSinger "is a parameterized Markov chain that iteratively
//! converts the noise into mel-spectrogram conditioned on the music score",
//! and the shallow-diffusion mechanism has it "start generation at a
//! shallow step smaller than the total number of diffusion steps,
//! according to the intersection of the diffusion trajectories of the
//! ground-truth mel-spectrogram and the one predicted by a simple
//! mel-spectrogram decoder".
//!
//! ```text
//! music score (phonemes + per-note MIDI pitch + durations)
//!   -> FFT-block phoneme encoder            (enc_layers=4, num_heads=2, hidden=384)
//!   -> score-to-frame duration expansion    (per-phoneme durations -> frame grid)
//!   -> shallow-diffusion denoiser backbone  (`backbone_type: lynxnet2`, K_step=400
//!                                            of timesteps=1000, DDIM accelerator)
//!   -> **mel-spectrogram** (128 bins @ 44.1 kHz, hop 512)
//!   ==== hand-off boundary ====
//!   -> separate neural vocoder -> waveform
//! ```
//!
//! **The vocoder half is deliberately out of scope for this arch.** The
//! upstream README lists HiFi-GAN, NSF and pc-ddsp as *interchangeable*
//! vocoder options operating as separate components from the synthesis
//! models — so DiffSinger emits a mel and hands off. Vokra already has
//! that half landed as first-class binders: `crates/vokra-models/src/
//! hifigan/`, `crates/vokra-models/src/bigvgan/` and
//! `crates/vokra-models/src/vocos/`. Embedding a vocoder inside the
//! `diffsinger` arch would duplicate three landed binders and hard-wire a
//! choice the upstream project deliberately leaves free.
//!
//! # Distinct arch tag from every sibling TTS / vocoder family
//!
//! [`ARCH`] = `"diffsinger"` is **deliberately distinct** from every
//! sibling speech-synthesis and vocoder arch tag:
//!
//! - `piper` / `kokoro` / `cosyvoice2` / `qwen3_tts` / `xtts_v2` —
//!   *speech* TTS: text-to-speech with no note/pitch score axis at all.
//!   A DiffSinger checkpoint has no text front-end and cannot be driven
//!   by a sentence;
//! - `hifigan` / `bigvgan` / `vocos` / `nsf` — *vocoders*: mel-to-waveform.
//!   They consume what DiffSinger produces, they are not substitutes for
//!   it;
//! - `vits` / `sbv2` lineage — end-to-end waveform TTS that fuses acoustic
//!   model and vocoder, the exact opposite of DiffSinger's deliberate
//!   split;
//! - `vae_continuous` / `storm` — other diffusion-bearing arches whose
//!   denoisers operate on entirely different latent spaces.
//!
//! Silently sharing an arch tag would let runtime dispatch mis-route a
//! DiffSinger voicebank onto a speech-TTS or vocoder loader (FR-EX-08 —
//! no silent shape misroute).
//!
//! # License — Apache-2.0 (primary source: upstream repo)
//!
//! `github.com/openvpi/DiffSinger` states the project "is licensed under
//! the Apache 2.0 License" (README, fetched 2026-08-15; the repo is the
//! actively-maintained community fork, 3184 stars per the GitHub API at
//! scout time — CLAUDE.md「ハルシネーション厳禁」). Apache-2.0 maps to
//! [`LicenseClass::Permissive`].
//!
//! **Voicebank weights are a separate question from the code license.**
//! The openvpi project distributes the *framework*; individual singer
//! voicebanks are released by their own authors under their own terms
//! (frequently non-commercial or singer-consent-bound). The
//! [`convert_diffsinger_file`] `license` override exists precisely for
//! this: a repackager stamps the voicebank's own SPDX rather than
//! inheriting the framework's apache-2.0.
//!
//! §3.1 sign-off column in `docs/license-audit.md` is **BLANK**
//! (fail-closed default — CC MUST NOT sign a license row, that is
//! owner-only per memory `[[feedback-license-signoff-primary-source]]`).
//! Runtime binder land is unblocked; *publish* is blocked until §3.1 is
//! signed, and the per-voicebank weight question must be settled
//! separately from the framework grant.
//!
//! # `vokra.diffsinger.*` topology chunk group (19 axes)
//!
//! Every runtime hparam `vokra-models::diffsinger::DiffSinger::from_gguf`
//! needs is stamped here so a downstream reader is fully self-describing
//! (no external YAML side-car needed). **Every value below is transcribed
//! from the upstream config files**, fetched 2026-08-15:
//!
//! From `configs/acoustic.yaml`:
//! - `audio_num_mel_bins: 128`, `audio_sample_rate: 44100`,
//!   `hop_size: 512`, `fft_size: 2048`, `win_size: 2048`,
//!   `fmin: 40`, `fmax: 16000`, `mel_vmin: -14.`, `mel_vmax: 4.`
//! - `hidden_size: 384`, `backbone_type: 'lynxnet2'`
//! - `timesteps: 1000`, `max_beta: 0.02`, `schedule_type: 'linear'`,
//!   `diff_accelerator: ddim`, `K_step: 400`
//!
//! From `configs/base.yaml`:
//! - `enc_layers: 4`, `num_heads: 2`, `f0_min: 65`, `f0_max: 1100`
//!
//! The 44.1 kHz rate is called out in the upstream README as an explicit
//! improvement over the original paper's 24 kHz ("44.1 kHz instead of the
//! original 24 kHz") — a downstream reader must not assume the paper's
//! rate. The runtime binder holds the same values as `const`s; the
//! converter and the binder mirror the same primary-source-transcribed
//! defaults so an axis-value change must land in both crates in the same
//! commit.
//!
//! # Reused Vokra ops (no new primitives invented)
//!
//! - [`vokra_ops::ddpm_sampler`] — the diffusion loop. DiffSinger's
//!   `schedule_type: 'linear'` maps onto `BetaSchedule::Linear` and
//!   `timesteps: 1000` onto `num_train_steps`; the shallow `K_step: 400`
//!   is the reduced-step walk the sampler already supports.
//! - [`vokra_ops::length_conditioning`] — score-to-frame expansion
//!   (per-phoneme seconds → frame count at `sample_rate` / `hop`).
//!
//! # BF16 pass-through
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm.
//! BF16 stays GGUF type 30 ([`GgmlType::BF16`]); runtime widens
//! BF16 → f32 losslessly at load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream `state_dict` names verbatim**.
//! Real-weight parity + a native mel forward are **loud-partial** in the
//! runtime binder pending the FFT-block encoder + LynxNet2 denoiser
//! backbone composition (`crates/vokra-models/src/diffsinger/mod.rs` — no
//! fabricated mel-spectrogram is ever emitted, FR-EX-08).
//!
//! # No ONNX / no pickle (permanent)
//!
//! DiffSinger ships PyTorch checkpoints upstream; this converter **never**
//! touches ONNX or pickle (FR-LD-05 / NFR-DS-02) — note that upstream
//! *does* publish an ONNX export path, which Vokra deliberately does not
//! consume at runtime (CLAUDE.md 設計判断 2). The `.ckpt` → safetensors
//! bridge lives offline through the sibling
//! `tools/parity/nemo_pt_to_safetensors.py` (uv-managed Python 3.12 per
//! memory `[[feedback-python-uses-uv]]` + `[[feedback-python-3-12]]`),
//! not part of the runtime.

// Skeleton-only allowance: the public API is exercised by the in-module
// tests + wired to the CLI + `ModelKind` + `pub use` re-export in
// `lib.rs` in the same commit. Removed once callers exercise the API
// outside tests.
#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` = `diffsinger` — distinct from every sibling
/// speech-TTS arch (`piper`, `kokoro`, `cosyvoice2`, `qwen3_tts`,
/// `xtts_v2`) and every vocoder arch (`hifigan`, `bigvgan`, `vocos`,
/// `nsf`). FR-EX-08 forbids silent shape misroute across synthesis
/// families.
pub const ARCH: &str = "diffsinger";

/// `vokra.model.name` — canonical `diffsinger` release (the openvpi
/// community fork's acoustic model; individual singer voicebanks are
/// distributed separately by their own authors and carry their own
/// `vokra.model.name` when repackaged).
pub const NAME: &str = "diffsinger";

/// `vokra.model.category` = `svs` — **singing voice synthesis**, a
/// brand-new category in the Vokra catalogue (no prior `svs` entry
/// existed as of 2026-08-15).
///
/// Deliberately **not** `tts` (DiffSinger takes a music score, not a
/// sentence, and has no text front-end) and deliberately **not** `vc`
/// (there is no source recording — see the module docstring's ELVIS Act
/// section for why this distinction is load-bearing). Consumed by the
/// model-card generator + zoo manifest tier gate.
pub const CATEGORY: &str = "svs";

/// Upstream GitHub tree the release ships from. DiffSinger's canonical
/// home is the openvpi community fork on GitHub (individual voicebanks
/// are scattered across HF / other hosts), so this uses `upstream_url`
/// rather than `upstream_hf`; the model-card generator picks up either.
pub const UPSTREAM_URL: &str = "github.com/openvpi/DiffSinger";

/// Default weight license SPDX (`apache-2.0`) per the upstream repo
/// README ("licensed under the Apache 2.0 License", fetched 2026-08-15).
///
/// **This is the framework grant, not a voicebank grant.** Override via
/// the [`convert_diffsinger_file`] `license` parameter when converting a
/// singer voicebank released under its own (often non-commercial) terms
/// — the standing mechanism mirrored from `convert_file_licensed` in
/// `lib.rs`.
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

/// Ad-hoc metadata key for the model category. Kept as a converter-side
/// constant (not a `chunks::KEY_*` alias) matching the sibling `gtcrn` /
/// `nsnet2` / `emotion2vec` posture until a first-class `category`
/// consumer lands in `vokra-core`.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Ad-hoc metadata key for the upstream URL (used for non-HF sources
/// such as GitHub). Sibling to `gtcrn::KEY_PROVENANCE_UPSTREAM_URL`.
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

// ---- `vokra.diffsinger.*` hparam chunk group ----------------------------
//
// Mirror of the `gtcrn::KEY_*` posture: every runtime hparam the
// `vokra-models::diffsinger::DiffSinger::from_gguf` binder needs is
// stamped here so a downstream reader is fully self-describing. A
// `0`-sentinel on any u32 axis makes the runtime binder refuse to load
// (FR-EX-08 — no silent default).

/// GGUF metadata key: PCM sample rate (u32 Hz). Upstream
/// `audio_sample_rate: 44100`.
pub const KEY_DS_SAMPLE_RATE: &str = "vokra.diffsinger.sample_rate";
/// GGUF metadata key: STFT FFT length (u32). Upstream `fft_size: 2048`.
pub const KEY_DS_N_FFT: &str = "vokra.diffsinger.n_fft";
/// GGUF metadata key: STFT hop (u32 samples). Upstream `hop_size: 512`.
pub const KEY_DS_HOP: &str = "vokra.diffsinger.hop";
/// GGUF metadata key: STFT window length (u32). Upstream
/// `win_size: 2048`.
pub const KEY_DS_WIN_SIZE: &str = "vokra.diffsinger.win_size";
/// GGUF metadata key: mel bin count (u32). Upstream
/// `audio_num_mel_bins: 128`.
pub const KEY_DS_N_MELS: &str = "vokra.diffsinger.n_mels";
/// GGUF metadata key: mel filterbank low edge (u32 Hz). Upstream
/// `fmin: 40`.
pub const KEY_DS_FMIN: &str = "vokra.diffsinger.fmin";
/// GGUF metadata key: mel filterbank high edge (u32 Hz). Upstream
/// `fmax: 16000`.
pub const KEY_DS_FMAX: &str = "vokra.diffsinger.fmax";
/// GGUF metadata key: encoder / denoiser hidden width (u32). Upstream
/// `hidden_size: 384`.
pub const KEY_DS_HIDDEN_SIZE: &str = "vokra.diffsinger.hidden_size";
/// GGUF metadata key: FFT-block phoneme encoder layer count (u32).
/// Upstream `enc_layers: 4`.
pub const KEY_DS_ENC_LAYERS: &str = "vokra.diffsinger.enc_layers";
/// GGUF metadata key: encoder attention head count (u32). Upstream
/// `num_heads: 2`.
pub const KEY_DS_NUM_HEADS: &str = "vokra.diffsinger.num_heads";
/// GGUF metadata key: total diffusion timesteps the model was trained on
/// (u32). Upstream `timesteps: 1000`. Maps onto
/// `DdpmSamplerConfig::num_train_steps`.
pub const KEY_DS_TIMESTEPS: &str = "vokra.diffsinger.timesteps";
/// GGUF metadata key: **shallow** diffusion start step (u32). Upstream
/// `K_step: 400` / `K_step_infer: 400` — the defining knob of
/// arXiv:2105.02446: generation starts at step `K_step` rather than at
/// the full `timesteps`, which is what "shallow diffusion" *means*.
pub const KEY_DS_K_STEP: &str = "vokra.diffsinger.k_step";
/// GGUF metadata key: F0 search floor (u32 Hz). Upstream `f0_min: 65`.
pub const KEY_DS_F0_MIN: &str = "vokra.diffsinger.f0_min";
/// GGUF metadata key: F0 search ceiling (u32 Hz). Upstream
/// `f0_max: 1100` — note this is far above the speech-TTS range, because
/// singing covers soprano registers.
pub const KEY_DS_F0_MAX: &str = "vokra.diffsinger.f0_max";
/// GGUF metadata key: diffusion beta ceiling (f32). Upstream
/// `max_beta: 0.02`.
pub const KEY_DS_MAX_BETA: &str = "vokra.diffsinger.max_beta";
/// GGUF metadata key: mel dynamic-range floor (f32). Upstream
/// `mel_vmin: -14.`.
pub const KEY_DS_MEL_VMIN: &str = "vokra.diffsinger.mel_vmin";
/// GGUF metadata key: mel dynamic-range ceiling (f32). Upstream
/// `mel_vmax: 4.`.
pub const KEY_DS_MEL_VMAX: &str = "vokra.diffsinger.mel_vmax";
/// GGUF metadata key: beta schedule name (string). Upstream
/// `schedule_type: 'linear'` → maps onto
/// `vokra_ops::ddpm_sampler::BetaSchedule::Linear`.
pub const KEY_DS_SCHEDULE_TYPE: &str = "vokra.diffsinger.schedule_type";
/// GGUF metadata key: diffusion accelerator name (string). Upstream
/// `diff_accelerator: ddim`.
pub const KEY_DS_DIFF_ACCELERATOR: &str = "vokra.diffsinger.diff_accelerator";
/// GGUF metadata key: denoiser backbone name (string). Upstream
/// `backbone_type: 'lynxnet2'`.
pub const KEY_DS_BACKBONE_TYPE: &str = "vokra.diffsinger.backbone_type";

/// Upstream PCM sample rate (Hz) — `audio_sample_rate: 44100`.
/// The upstream README calls out 44.1 kHz as an explicit improvement over
/// the original paper's 24 kHz.
pub const DEFAULT_SAMPLE_RATE: u32 = 44_100;
/// Upstream STFT FFT length — `fft_size: 2048`.
pub const DEFAULT_N_FFT: u32 = 2048;
/// Upstream STFT hop (samples) — `hop_size: 512`.
pub const DEFAULT_HOP: u32 = 512;
/// Upstream STFT window length (samples) — `win_size: 2048`.
pub const DEFAULT_WIN_SIZE: u32 = 2048;
/// Upstream mel bin count — `audio_num_mel_bins: 128`.
pub const DEFAULT_N_MELS: u32 = 128;
/// Upstream mel low edge (Hz) — `fmin: 40`.
pub const DEFAULT_FMIN: u32 = 40;
/// Upstream mel high edge (Hz) — `fmax: 16000`.
pub const DEFAULT_FMAX: u32 = 16_000;
/// Upstream hidden width — `hidden_size: 384`.
pub const DEFAULT_HIDDEN_SIZE: u32 = 384;
/// Upstream encoder layer count — `enc_layers: 4`.
pub const DEFAULT_ENC_LAYERS: u32 = 4;
/// Upstream encoder head count — `num_heads: 2`.
pub const DEFAULT_NUM_HEADS: u32 = 2;
/// Upstream training diffusion timesteps — `timesteps: 1000`.
pub const DEFAULT_TIMESTEPS: u32 = 1000;
/// Upstream shallow-diffusion start step — `K_step: 400`.
pub const DEFAULT_K_STEP: u32 = 400;
/// Upstream F0 floor (Hz) — `f0_min: 65`.
pub const DEFAULT_F0_MIN: u32 = 65;
/// Upstream F0 ceiling (Hz) — `f0_max: 1100`.
pub const DEFAULT_F0_MAX: u32 = 1100;
/// Upstream diffusion beta ceiling — `max_beta: 0.02`.
pub const DEFAULT_MAX_BETA: f32 = 0.02;
/// Upstream mel floor — `mel_vmin: -14.`.
pub const DEFAULT_MEL_VMIN: f32 = -14.0;
/// Upstream mel ceiling — `mel_vmax: 4.`.
pub const DEFAULT_MEL_VMAX: f32 = 4.0;
/// Upstream beta schedule name — `schedule_type: 'linear'`.
pub const DEFAULT_SCHEDULE_TYPE: &str = "linear";
/// Upstream diffusion accelerator — `diff_accelerator: ddim`.
pub const DEFAULT_DIFF_ACCELERATOR: &str = "ddim";
/// Upstream denoiser backbone — `backbone_type: 'lynxnet2'`.
pub const DEFAULT_BACKBONE_TYPE: &str = "lynxnet2";

const UPSTREAM_SOURCE: &str = "openvpi/DiffSinger (DiffSinger: Singing Voice Synthesis via Shallow Diffusion \
     Mechanism, arXiv:2105.02446 — score-to-singing acoustic model emitting a \
     128-bin mel at 44.1 kHz for a separate neural vocoder, Apache-2.0)";

/// Outcome of a DiffSinger conversion.
///
/// All counters are additive and default to zero — a zero-tensor
/// checkpoint returns `DiffSingerReport::default()` and the caller
/// remains responsible for surfacing the "no float tensors" loud note
/// (mirror of the `gtcrn` / `nsnet2` / `emotion2vec` `Report` pattern).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DiffSingerReport {
    /// Total tensors surfaced by the safetensors reader (the sum of
    /// `written + skipped_non_float`). Pins the budget so a truncated
    /// header cannot silently drop tensors without the caller noticing.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 all go through
    /// the same byte-copy path).
    pub written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time).
    pub skipped_non_float: usize,
    /// Of the tensors in `written`, how many were BF16 (subset counter).
    /// Emits GGUF type 30 verbatim; runtime widens BF16 → f32 losslessly
    /// via the single choke point `crates/vokra-core/src/gguf/quant/mod.rs
    /// decode_bf16`.
    pub bf16_passthrough: usize,
}

/// Reads a safetensors checkpoint at `input` and writes a DiffSinger
/// GGUF to `output`.
///
/// Every F32 / F16 / BF16 tensor is emitted verbatim under its upstream
/// name; the `vokra.model.*` (arch / name / category) and
/// `vokra.provenance.*` chunk groups are stamped for the runtime
/// compliance gate (FR-CP-03) alongside the 19-axis
/// `vokra.diffsinger.*` topology chunk group. `vokra.schema.*` is
/// written unconditionally by the GGUF writer.
///
/// `license` overrides [`DEFAULT_LICENSE_SPDX`] (`"apache-2.0"`). **Use
/// it when converting a singer voicebank**: the apache-2.0 default is the
/// *framework* grant from `openvpi/DiffSinger`, and individual voicebanks
/// are released by their own authors under their own (frequently
/// non-commercial or consent-bound) terms.
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure; [`ConvertError::Parse`]
/// on a malformed safetensors input.
pub fn convert_diffsinger_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<DiffSingerReport, ConvertError> {
    // Whole-file read: a DiffSinger acoustic model is a few hundred MB at
    // most — well below the ≥8 GB vast.ai cutoff per memory
    // `[[feedback-large-models-on-vast-ai]]`, so no streaming path.
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // Self-describing redistribution: the artifact carries its own
    // licence. The apache-2.0 default is the framework grant per the
    // upstream README; a voicebank repackager passes `license` to stamp
    // the voicebank's own SPDX instead.
    let effective_license = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let effective_class = LicenseClass::from_license_str(effective_license);
    vokra_core::stamp_provenance(
        &mut b,
        effective_class,
        effective_license,
        Some(NAME),
        Some(UPSTREAM_SOURCE),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);

    // Topology axes, all transcribed from the upstream `configs/
    // acoustic.yaml` + `configs/base.yaml` (fetched 2026-08-15).
    // Stamping them here makes the artifact self-describing so
    // `vokra-models::diffsinger::DiffSinger::from_gguf` can validate
    // loudly (FR-EX-08 — a checkpoint from a different topology cannot
    // silently misload). CLAUDE.md「ハルシネーション厳禁」: owner MUST
    // re-confirm these axes against the upstream configs at land time
    // rather than trusting the transcribed constants alone, and MUST
    // re-check per voicebank since a voicebank may ship a variant config.
    b.add_u32(KEY_DS_SAMPLE_RATE, DEFAULT_SAMPLE_RATE);
    b.add_u32(KEY_DS_N_FFT, DEFAULT_N_FFT);
    b.add_u32(KEY_DS_HOP, DEFAULT_HOP);
    b.add_u32(KEY_DS_WIN_SIZE, DEFAULT_WIN_SIZE);
    b.add_u32(KEY_DS_N_MELS, DEFAULT_N_MELS);
    b.add_u32(KEY_DS_FMIN, DEFAULT_FMIN);
    b.add_u32(KEY_DS_FMAX, DEFAULT_FMAX);
    b.add_u32(KEY_DS_HIDDEN_SIZE, DEFAULT_HIDDEN_SIZE);
    b.add_u32(KEY_DS_ENC_LAYERS, DEFAULT_ENC_LAYERS);
    b.add_u32(KEY_DS_NUM_HEADS, DEFAULT_NUM_HEADS);
    b.add_u32(KEY_DS_TIMESTEPS, DEFAULT_TIMESTEPS);
    b.add_u32(KEY_DS_K_STEP, DEFAULT_K_STEP);
    b.add_u32(KEY_DS_F0_MIN, DEFAULT_F0_MIN);
    b.add_u32(KEY_DS_F0_MAX, DEFAULT_F0_MAX);
    b.add_f32(KEY_DS_MAX_BETA, DEFAULT_MAX_BETA);
    b.add_f32(KEY_DS_MEL_VMIN, DEFAULT_MEL_VMIN);
    b.add_f32(KEY_DS_MEL_VMAX, DEFAULT_MEL_VMAX);
    b.add_string(KEY_DS_SCHEDULE_TYPE, DEFAULT_SCHEDULE_TYPE);
    b.add_string(KEY_DS_DIFF_ACCELERATOR, DEFAULT_DIFF_ACCELERATOR);
    b.add_string(KEY_DS_BACKBONE_TYPE, DEFAULT_BACKBONE_TYPE);

    let mut report = DiffSingerReport::default();
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (`docs/adr/qwen3-tts-bf16.md`, strategy A_passthrough); the runtime
    // widens BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )
                .map_err(|e| ConvertError::Gguf(e.to_string()))?;
                report.written += 1;
                if t.dtype == GgmlType::BF16 {
                    report.bf16_passthrough += 1;
                }
            }
            _ => {
                report.skipped_non_float += 1;
            }
        }
    }

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Parse(e.to_string()))?;
    std::fs::write(output, out_bytes).map_err(ConvertError::Io)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::GgufFile;

    /// Per-test unique scratch path (PID + monotonically increasing
    /// sequence — the gtcrn / sepformer test pattern; no external
    /// `tempfile` dep, preserving zero-dep NFR-DS-02).
    fn scratch_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-diffsinger-{tag}-{}-{n}",
            std::process::id()
        ));
        p
    }

    fn safetensors_one(name: &str, dtype: &str, shape: &[u64], payload: &[u8]) -> Vec<u8> {
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"{dtype}","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn safetensors_f32_then_f16(
        f32_name: &str,
        f32_shape: &[u64],
        f32_bytes: &[u8],
        f16_name: &str,
        f16_shape: &[u64],
        f16_bytes: &[u8],
    ) -> Vec<u8> {
        let f32_elems: u64 = f32_shape.iter().product();
        assert_eq!(f32_bytes.len(), f32_elems as usize * 4);
        let f16_elems: u64 = f16_shape.iter().product();
        assert_eq!(f16_bytes.len(), f16_elems as usize * 2);
        let f32_shape_str = f32_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let f16_shape_str = f16_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let f32_len = f32_bytes.len();
        let total = f32_len + f16_bytes.len();
        let header = format!(
            r#"{{"{f32_name}":{{"dtype":"F32","shape":[{f32_shape_str}],"data_offsets":[0,{f32_len}]}},"{f16_name}":{{"dtype":"F16","shape":[{f16_shape_str}],"data_offsets":[{f32_len},{total}]}}}}"#
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(f32_bytes);
        out.extend_from_slice(f16_bytes);
        out
    }

    // -----------------------------------------------------------------
    // Test 1 — BF16 round-trip + full topology + provenance stamps
    // -----------------------------------------------------------------

    /// BF16 pass-through pin plus the full metadata surface: the dtype
    /// must stay BF16 (GGUF type 30) and the payload must be
    /// byte-identical (a silent widen would still round-trip values but
    /// would break the byte pin). Every `vokra.model.*` /
    /// `vokra.provenance.*` / `vokra.diffsinger.*` chunk MUST land.
    #[test]
    fn bf16_tensor_passes_through_and_full_metadata_lands() {
        // Non-zero BF16 bit patterns so any silent widen / downcast
        // attempt is caught by the subsequent byte-identity assert.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements x 2 bytes BF16 payload");

        // Mirror a plausible upstream DiffSinger state_dict tensor name.
        // The exact encoder block layout is pinned by the real manifest
        // at follow-up time; this fixture only exercises the byte-copy
        // path.
        let input_bytes = safetensors_one(
            "fs2.encoder.layers.0.self_attn.in_proj_weight",
            "BF16",
            &[2, 3],
            &bf16,
        );
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_diffsinger_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 1, "one tensor visible in header");
        assert_eq!(report.written, 1, "BF16 must reach the pass-through arm");
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 subset counter must record the pass-through"
        );

        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        let info = file
            .tensor_info("fs2.encoder.layers.0.self_attn.in_proj_weight")
            .expect("BF16 tensor present after pass-through");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input"
        );

        // Provenance + category chunks pinned on the artifact itself.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "category chunk pins DiffSinger as `svs` (singing voice synthesis), \
             not `tts` and not `vc`"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "apache-2.0 weight license normalises to LicenseClass::Permissive"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_URL)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_URL),
            "upstream_url chunk pins the GitHub tree the release ships from"
        );
        // Schema stamp is written unconditionally by the GGUF writer.
        assert!(
            file.get(chunks::KEY_SCHEMA_VERSION).is_some(),
            "vokra.schema.version must be stamped"
        );
        assert!(
            file.get(chunks::KEY_SCHEMA_PRODUCER).is_some(),
            "vokra.schema.producer must be stamped"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // -----------------------------------------------------------------
    // Test 2 — F32 + F16 mixed pass-through (BF16 counter stays at 0)
    // -----------------------------------------------------------------

    /// Mixed F32/F16 round-trip pin: both dtypes must ride the same
    /// pass-through arm; the BF16 subset counter MUST stay at zero
    /// (defence against a regression where a widen path silently upcasts
    /// F32 / F16 into BF16 to inflate the pass-through counter).
    #[test]
    fn f32_and_f16_tensors_pass_through_no_bf16_upcast() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // F16 exact-representable values via manual half bit-fiddling
        // (no external crate).
        let f16_words: [u16; 6] = [0x3C00, 0xC000, 0xB800, 0x4200, 0x3100, 0x5140];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 12);

        let input_bytes = safetensors_f32_then_f16(
            "fs2.encoder.embed_positions.weight",
            &[1, 2],
            &f32_bytes,
            "diffusion.denoise_fn.residual_layers.0.conv.weight",
            &[2, 3],
            &f16_bytes,
        );
        let input = scratch_path("mixed-in");
        let output = scratch_path("mixed-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_diffsinger_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 2, "two tensors observed");
        assert_eq!(
            report.written, 2,
            "both F32 and F16 tensors must pass through"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32/F16 must NOT increment the BF16 counter (no silent upcast)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );

        let out_bytes = std::fs::read(&output).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let f32_info = file
            .tensor_info("fs2.encoder.embed_positions.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());
        let f16_info = file
            .tensor_info("diffusion.denoise_fn.residual_layers.0.conv.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // -----------------------------------------------------------------
    // Test 3 — malformed safetensors input is refused loudly
    // -----------------------------------------------------------------

    /// A truncated / malformed safetensors header must produce a loud
    /// [`ConvertError`] rather than a silently-empty GGUF (FR-EX-08).
    /// A converter that swallowed a parse failure and wrote a
    /// zero-tensor artifact would produce a file that *looks* like a
    /// voicebank but synthesises nothing.
    #[test]
    fn malformed_safetensors_input_is_refused_loudly() {
        // Header length prefix claims 4096 bytes but the file has none.
        let mut bogus = Vec::new();
        bogus.extend_from_slice(&4096u64.to_le_bytes());
        bogus.extend_from_slice(b"{ this is not valid json");

        let input = scratch_path("bad-in");
        let output = scratch_path("bad-out");
        std::fs::write(&input, &bogus).expect("write malformed input");

        let err = convert_diffsinger_file(&input, &output, None)
            .expect_err("malformed safetensors must be refused, not silently converted");
        // The specific variant is the reader's business; what matters is
        // that it is an error and that no output artifact was produced.
        let msg = format!("{err:?}");
        assert!(
            !msg.is_empty(),
            "the refusal must carry a diagnosable message"
        );
        assert!(
            !output.exists(),
            "no GGUF may be written when the input failed to parse — a \
             zero-tensor artifact would look like a valid voicebank"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // -----------------------------------------------------------------
    // Test 4 — License override swaps the stamped SPDX + class
    // -----------------------------------------------------------------

    /// License override pin: this is the **voicebank** path. The
    /// apache-2.0 default is the openvpi *framework* grant; a singer
    /// voicebank released under CC-BY-NC-4.0 must stamp its own SPDX and
    /// re-derive a **NonCommercial** class rather than inheriting the
    /// framework's permissive verdict. Getting this wrong would let a
    /// non-commercial voicebank pass a commercial-redistribution gate.
    #[test]
    fn license_override_swaps_spdx_and_reclassifies_voicebank() {
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one("x", "F32", &[1], &payload);
        let input = scratch_path("lic-in");
        let output = scratch_path("lic-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        convert_diffsinger_file(&input, &output, Some("cc-by-nc-4.0"))
            .expect("convert with voicebank license override");
        let out_bytes = std::fs::read(&output).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("cc-by-nc-4.0"),
            "override SPDX lands verbatim"
        );
        let class = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str());
        assert_ne!(
            class,
            Some(LicenseClass::Permissive.as_str()),
            "a non-commercial voicebank MUST NOT inherit the framework's \
             Permissive class — that would let it pass a commercial \
             redistribution gate (fail-closed)"
        );
        assert_eq!(
            class,
            Some(LicenseClass::NonCommercial.as_str()),
            "cc-by-nc-4.0 must re-derive as LicenseClass::NonCommercial"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // -----------------------------------------------------------------
    // Test 5 — All 19 `vokra.diffsinger.*` axes emit the transcribed
    //          upstream values (rename / axis-value regression pin)
    // -----------------------------------------------------------------

    /// Pin every primary-source-transcribed axis
    /// (`configs/acoustic.yaml` + `configs/base.yaml`, fetched
    /// 2026-08-15). A rename or default-value change must land here in
    /// the same commit or fail this test.
    #[test]
    fn all_diffsinger_axes_emit_expected_upstream_values() {
        let payload: Vec<u8> = [0.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one("dummy.weight", "F32", &[1], &payload);
        let input = scratch_path("axes-in");
        let output = scratch_path("axes-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        convert_diffsinger_file(&input, &output, None).expect("convert");
        let out_bytes = std::fs::read(&output).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");

        // u32 axes — every value transcribed verbatim from upstream.
        for (k, want, src) in [
            (
                KEY_DS_SAMPLE_RATE,
                44_100u32,
                "acoustic.yaml audio_sample_rate",
            ),
            (KEY_DS_N_FFT, 2048, "acoustic.yaml fft_size"),
            (KEY_DS_HOP, 512, "acoustic.yaml hop_size"),
            (KEY_DS_WIN_SIZE, 2048, "acoustic.yaml win_size"),
            (KEY_DS_N_MELS, 128, "acoustic.yaml audio_num_mel_bins"),
            (KEY_DS_FMIN, 40, "acoustic.yaml fmin"),
            (KEY_DS_FMAX, 16_000, "acoustic.yaml fmax"),
            (KEY_DS_HIDDEN_SIZE, 384, "acoustic.yaml hidden_size"),
            (KEY_DS_ENC_LAYERS, 4, "base.yaml enc_layers"),
            (KEY_DS_NUM_HEADS, 2, "base.yaml num_heads"),
            (KEY_DS_TIMESTEPS, 1000, "acoustic.yaml timesteps"),
            (KEY_DS_K_STEP, 400, "acoustic.yaml K_step"),
            (KEY_DS_F0_MIN, 65, "base.yaml f0_min"),
            (KEY_DS_F0_MAX, 1100, "base.yaml f0_max"),
        ] {
            assert_eq!(
                file.get(k).and_then(|v| v.as_u64()),
                Some(u64::from(want)),
                "hparam `{k}` must be stamped as {want} (upstream {src})"
            );
        }

        // f32 axes.
        for (k, want, src) in [
            (KEY_DS_MAX_BETA, 0.02_f32, "acoustic.yaml max_beta"),
            (KEY_DS_MEL_VMIN, -14.0, "acoustic.yaml mel_vmin"),
            (KEY_DS_MEL_VMAX, 4.0, "acoustic.yaml mel_vmax"),
        ] {
            let got = file.get(k).and_then(|v| v.as_f64()).map(|v| v as f32);
            assert_eq!(
                got,
                Some(want),
                "hparam `{k}` must be stamped as {want} (upstream {src})"
            );
        }

        // string axes.
        assert_eq!(
            file.get(KEY_DS_SCHEDULE_TYPE).and_then(|v| v.as_str()),
            Some("linear"),
            "schedule_type = linear (upstream acoustic.yaml schedule_type)"
        );
        assert_eq!(
            file.get(KEY_DS_DIFF_ACCELERATOR).and_then(|v| v.as_str()),
            Some("ddim"),
            "diff_accelerator = ddim (upstream acoustic.yaml diff_accelerator)"
        );
        assert_eq!(
            file.get(KEY_DS_BACKBONE_TYPE).and_then(|v| v.as_str()),
            Some("lynxnet2"),
            "backbone_type = lynxnet2 (upstream acoustic.yaml backbone_type)"
        );

        // Structural invariants the upstream config satisfies. A future
        // axis edit that broke either would be a transcription bug.
        const {
            assert!(
                DEFAULT_K_STEP < DEFAULT_TIMESTEPS,
                // Plain literal, not a format string: this is a const
                // assertion and formatting macros are not const-callable.
                // Both operands are named right above, so the message does
                // not need to restate their values.
                "structural invariant: DEFAULT_K_STEP must be strictly less \
                 than DEFAULT_TIMESTEPS — K_step >= timesteps would mean the \
                 shallow diffusion mechanism of arXiv:2105.02446 is not \
                 actually engaged"
            )
        };
        const {
            assert!(
                DEFAULT_FMAX * 2 <= DEFAULT_SAMPLE_RATE,
                "structural invariant: DEFAULT_FMAX must respect the Nyquist \
                 limit of DEFAULT_SAMPLE_RATE"
            )
        };
        const {
            assert!(
                DEFAULT_F0_MAX > DEFAULT_F0_MIN,
                "structural invariant: f0_max must exceed f0_min"
            )
        };

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    // -----------------------------------------------------------------
    // Test 6 — arch / name / category tag pins (FR-EX-08 distinctness)
    // -----------------------------------------------------------------

    /// Pin `ARCH = "diffsinger"` and assert distinctness against every
    /// sibling speech-TTS and vocoder arch string. A future rename of any
    /// sibling arch tag would land here in the same commit or fail this
    /// test.
    #[test]
    fn arch_tag_distinct_from_speech_tts_and_vocoder_arches() {
        assert_eq!(ARCH, "diffsinger");
        assert_eq!(NAME, "diffsinger");
        assert_eq!(
            CATEGORY, "svs",
            "singing voice synthesis is its own category — NOT `tts` (no text \
             front-end, takes a music score) and NOT `vc` (no source recording, \
             so not an ELVIS Act voice-clone trigger)"
        );
        for sibling in [
            // speech TTS — text in, no score axis
            "piper",
            "kokoro",
            "cosyvoice2",
            "qwen3_tts",
            "xtts_v2",
            "sbv2",
            "vits",
            // vocoders — they CONSUME what diffsinger produces
            "hifigan",
            "bigvgan",
            "vocos",
            "nsf",
            // other diffusion-bearing arches over different latents
            "vae_continuous",
            "storm",
        ] {
            assert_ne!(
                ARCH, sibling,
                "diffsinger (score-to-singing shallow-diffusion acoustic model) \
                 and `{sibling}` are distinct arches — sharing an arch tag would \
                 misroute the runtime dispatch (FR-EX-08)"
            );
        }
    }

    // -----------------------------------------------------------------
    // Test 7 — the mel/vocoder hand-off boundary is a real contract
    // -----------------------------------------------------------------

    /// The converter must NOT stamp any vocoder-selection axis. The
    /// upstream README lists HiFi-GAN / NSF / pc-ddsp as interchangeable
    /// options, and Vokra already carries `hifigan` / `bigvgan` / `vocos`
    /// as separate first-class binders. Baking a vocoder choice into the
    /// acoustic model's metadata would hard-wire a decision upstream
    /// deliberately leaves free, and would duplicate three landed
    /// binders.
    #[test]
    fn no_vocoder_axis_is_stamped_mel_handoff_is_the_boundary() {
        let payload: Vec<u8> = [0.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one("dummy.weight", "F32", &[1], &payload);
        let input = scratch_path("vocoder-in");
        let output = scratch_path("vocoder-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");
        convert_diffsinger_file(&input, &output, None).expect("convert");
        let out_bytes = std::fs::read(&output).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");

        for forbidden in [
            "vokra.diffsinger.vocoder",
            "vokra.diffsinger.vocoder_type",
            "vokra.diffsinger.hifigan",
            "vokra.diffsinger.nsf",
        ] {
            assert!(
                file.get(forbidden).is_none(),
                "`{forbidden}` must NOT be stamped — DiffSinger emits a mel and \
                 hands off to a SEPARATE vocoder binder (hifigan / bigvgan / \
                 vocos), per the upstream README listing HiFi-GAN / NSF / \
                 pc-ddsp as interchangeable"
            );
        }
        // The mel axes that define the hand-off contract ARE stamped, so
        // a downstream vocoder binder can validate compatibility.
        assert!(
            file.get(KEY_DS_N_MELS).is_some() && file.get(KEY_DS_SAMPLE_RATE).is_some(),
            "the mel hand-off contract (n_mels + sample_rate + hop) MUST be \
             stamped so a downstream vocoder binder can check compatibility"
        );
        assert!(file.get(KEY_DS_HOP).is_some());

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
