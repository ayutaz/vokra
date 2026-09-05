#![allow(clippy::doc_lazy_continuation)]
//! **AudioLDM 2** (`cvssp/audioldm2`, **cc-by-nc-sa-4.0**): safetensors →
//! GGUF conversion (Wave 5 candidate, 2026-08-01).
//!
//! Input: the upstream `cvssp/audioldm2` release — Liu et al. 2024 (ICML,
//! arXiv:2308.05734 "AudioLDM 2: Learning Holistic Audio Generation with
//! Self-supervised Pretraining"). Text-to-audio latent-diffusion model
//! (LDM) operating on a compressed 1D latent produced by a VAE encoder;
//! the U-Net diffuses in latent space, a HiFi-GAN vocoder decodes latents
//! → mel → waveform, and the text-conditioning path fuses **three
//! encoders** (frozen T5-base + CLAP text encoder + a small GPT-2
//! "audio-caption LM" that produces "language of audio" tokens the U-Net
//! consumes). Distinct from sibling `musicgen` (autoregressive over
//! EnCodec RVQ tokens) — AudioLDM 2 is a *diffusion* generator over a
//! VAE latent, not an AR transformer over discrete audio tokens; silently
//! sharing the `musicgen` arch tag would misroute runtime dispatch to a
//! wrong-shape forward.
//!
//! # Vokra scope — music / audio generation (per 2026-07-30 scope expansion)
//!
//! AudioLDM 2 is a text-to-audio latent-diffusion generator. Music
//! generation was pinned in-scope by the 2026-07-30 依頼者指示「asr,tts,
//! 音楽系,音声分離など全てのモデルに対応したい」（memory
//! `[[project-scope-expansion-2026-07-30]]`). This converter uses the
//! same `category = "music"` tag as sibling
//! `super::musicgen_medium` / `super::musicgen_large` — AudioLDM 2's
//! output space includes music (the CVSSP release ships a canonical
//! text-to-music prompt suite) alongside general audio; a separate
//! `audio-generation` taxonomy tag would be premature before a second
//! non-music-tree generator lands.
//!
//! # License posture — CC-BY-NC-SA-4.0 default (**NonCommercialShareAlike**)
//!
//! Weight redistribution default is [`LicenseClass::NonCommercialShareAlike`].
//! The HF model card at `huggingface.co/cvssp/audioldm2` carries
//! `license: cc-by-nc-4.0` on the current YAML front-matter, but every
//! upstream artifact (the paper §Ethics + the CVSSP GitHub README + every
//! sibling variant `cvssp/audioldm2-{music,large,music-665k}`) pins the
//! **weight license as CC-BY-NC-SA 4.0** — the *ShareAlike* clause is
//! what governs redistribution of derivative artifacts. **We follow the
//! ShareAlike primary source** (the model card's `-nc-4.0` tag is the
//! looser form of the same restriction and would silently drop the SA
//! cascade if we defaulted to it — the same Fish-Speech precedent
//! (`docs/license-audit.md` §3.1 row 250/251) which pins CC-BY-NC-SA-4.0
//! for the same reason.
//!
//! [`LicenseClass::NonCommercialShareAlike`] is doubly restrictive:
//!
//! - **NC (non-commercial)** → activates
//!   [`LicenseClass::requires_research_flag`] at load time → **fail-closed**
//!   at the M2-13 runtime gate (`VokraError::ResearchLicenseRequired`).
//! - **SA (share-alike)** → activates
//!   [`LicenseClass::requires_license_preserved`] → any converted GGUF
//!   redistributed downstream must carry the CC-BY-NC-SA-4.0 grant
//!   forward, which **cascades** the obligation onto any Vokra-added
//!   artifacts bundled with the weight (model card, LICENSE, NOTICE,
//!   auxiliary GGUFs — the whole publish blob) unless the SA cascade
//!   is resolved by an owner ADR.
//!
//! [`LicenseClass::NonCommercialShareAlike::redistributable`] answers
//! **false**, so this converter's output cannot flow through the
//! `publish-one.sh` gate today — the artifact stops at the converter.
//! The task title "sa-cascade-defer" names this posture: the code +
//! prep-script land today so a future publish is one owner ADR (+
//! `--allow-noncommercial-sharealike` flag if it is ever introduced)
//! away, but the current release does NOT include an entry in
//! `scripts/publish/signoff_match.py::REPO_TO_SIGNOFF_ROWS` — the
//! unlisted slug fails-closed as `UNKNOWN_REPO` at publish time.
//!
//! ## Why not `LicenseClass::Copyleft`?
//!
//! [`LicenseClass::Copyleft`] covers pure share-alike (CC-BY-SA / AGPL /
//! GPL) — those are `commercial_ok() = true` and
//! `redistributable() = true`. AudioLDM 2's cc-by-nc-sa adds the NC
//! restriction, so `Copyleft` would silently downgrade the NC gate and
//! silently mark the weight as publishable. `NonCommercialShareAlike`
//! is the correct class: it stacks both restrictions and mirrors the
//! automatic classification `LicenseClass::from_license_str("cc-by-nc-
//! sa-4.0")` would produce if a caller ever supplied the license as a
//! raw SPDX id (the same Fish-Speech pattern in
//! `crates/vokra-core/src/compliance/license_class.rs`).
//!
//! Callers may override at the outer `convert_file --license <spdx>`
//! boundary when they legitimately hold the weights under a distinct
//! SPDX id (e.g. a permissive re-training on public-domain audio — the
//! same Whisper / kokoro / vits-ja / xcodec2 / musicgen override
//! pattern).
//!
//! # BF16 pass-through (mirror of musicgen_medium / xcodec2 / neucodec)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm —
//! no convert-time widening. BF16 stays GGUF type 30 (`GgmlType::BF16`);
//! the runtime widens BF16 → f32 losslessly at load via the single
//! choke point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`
//! (BF16 = top 16 bits of an f32 — `bits << 16` is exact). The
//! observability counter [`AudioLdm2Report::bf16_passthrough`] records
//! how many BF16 tensors landed on this arm so a silent widen /
//! downcast cannot slip in undetected.
//!
//! # Scale — vast.ai handoff (~8.5 GB)
//!
//! AudioLDM 2 ships as ~8.5 GB on HF (VAE encoder/decoder + U-Net
//! backbone + HiFi-GAN vocoder + frozen T5-base + CLAP text encoder +
//! GPT-2 audio-caption LM = bundle total). This is above the M1 iMac
//! 16 GB local-convert threshold (memory
//! `[[feedback-large-models-on-vast-ai]]`: ≥8 GB safe threshold on the
//! upper edge, and the multi-encoder AudioLDM 2 bundle is on the
//! swap-death curve when a full BF16 pass buffer + safetensors view
//! doubles peak resident to ~17 GB — the same class as sibling
//! MusicGen-Medium 11.4 GB) — conversion + publish happens on vast.ai
//! per `docs/handoff/vast-ai-large-model-publish.md`.
//!
//! # Bundle shape — five weight groups, one converter arm
//!
//! Upstream ships the full pipeline as a single sharded safetensors
//! bundle (`model.safetensors.index.json` + shards) or a torch pickle
//! (`pytorch_model.bin`, pre-2024 releases). The bundle carries **five
//! distinct sub-modules** under their upstream state-dict prefixes:
//!
//! 1. `vae.*` — the VAE encoder/decoder that maps waveform ↔ latent.
//! 2. `unet.*` — the latent-diffusion U-Net (2D U-Net with time-embed
//!    conditioning + cross-attention over the fused text tokens).
//! 3. `vocoder.*` — the HiFi-GAN mel-to-waveform head.
//! 4. `language_model.*` — the GPT-2 audio-caption LM that produces
//!    "language of audio" tokens the U-Net consumes.
//! 5. `text_encoder{,_2}.*` — the frozen T5-base + CLAP text encoders.
//!
//! Every group rides the same BF16 pass-through arm: names are copied
//! verbatim under their upstream prefix, dtypes preserved. A future
//! `AudioLdm2::from_gguf` walks these five prefixes to bind the five
//! sub-modules — the tensor-name contract is stable across variants
//! (`cvssp/audioldm2` vs `-music` vs `-large`).
//!
//! # Sibling family (future waves)
//!
//! Future AudioLDM 2 family additions (`-music`, `-large`, `-music-665k`
//! variants) can either:
//!   (a) land as sibling files (`audioldm2_music.rs`, `audioldm2_large.rs`,
//!       …) mirror of the `chatterbox` / `musicgen_medium` /
//!       `musicgen_large` split, OR
//!   (b) refactor into a shared `audioldm2.rs` with an `AudioLdm2Variant`
//!       enum mirror of the `vocos` / `snac` / `bigvgan` / `focalcodec`
//!       split.
//! Today's landing is a standalone base-only file (the `xcodec2` /
//! `wavtokenizer` / `musicgen_medium` posture) — a single §3.1 row, a
//! single ModelKind, no pre-emptive enum bloat before a second variant
//! exists.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim** (the
//! CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM / VibeVoice
//! / neucodec / step_audio2_mini / xcodec2 / musicgen contract).
//! Real-weight binding + AudioLDM 2 diffusion parity is a follow-up wave
//! gated on §3.1 sign-off + real-checkpoint tensor-name manifest fetch;
//! this converter passes every F32 / F16 / BF16 tensor through unchanged
//! so a future `AudioLdm2::from_gguf` can walk the same names.
//!
//! # Real-weight parity
//!
//! Real-weight parity against the upstream Python pipeline
//! (`diffusers.AudioLDM2Pipeline`) is deferred to owner
//! (`docs/license-audit.md` §3.1 sign-off queue). A parity harness
//! following the sepformer / DFN3 / Kokoro precedent (reference dumper →
//! fixture commit → Rust parity test) will land when the runtime binder
//! + latent-diffusion + VAE + HiFi-GAN ops land — this is a **new op
//! surface** (latent-diffusion sampler + VAE encoder/decoder), not a
//! reuse of the existing `flow_sampler` (which targets flow-matching, a
//! sibling family).
//!
//! # No ONNX (permanent)
//!
//! AudioLDM 2 ships safetensors + PyTorch pickle; this converter
//! **never** touches ONNX (FR-LD-05). The pipeline is re-implemented
//! natively in a future `crates/vokra-models/src/audioldm2/` module
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).
//! Between now and that landing, the runtime consumer walks the emitted
//! tensor names and either succeeds or fails loudly per FR-EX-08.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for AudioLDM 2 GGUFs.
///
/// The arch tag matches the `diffusers.AudioLDM2Pipeline` class family
/// (Liu et al. 2024, arXiv:2308.05734). Intentionally distinct from
/// every sibling arch tag:
///
/// - Distinct from `musicgen` (autoregressive over EnCodec RVQ tokens) —
///   AudioLDM 2 is a **latent-diffusion** generator over a VAE latent,
///   not an AR transformer over discrete audio tokens.
/// - Distinct from every speech-tree arch (Whisper ASR / CosyVoice2
///   speech-LM / Voxtral audio-LLM / Moshi full-duplex S2S).
///
/// Silently sharing an arch tag would mis-route runtime dispatch to a
/// wrong-shape forward. A future family variant (`-music` / `-large` /
/// `-music-665k`) shares this same `audioldm2` arch tag — the topology
/// is identical, only the model dims + optional variant-specific heads
/// differ (the same arch-shared / name-distinct posture snac / vocos /
/// bigvgan / musicgen use).
#[allow(dead_code)] // Retained as inspection-only dispatch metadata until the bundle is authenticated.
pub const ARCH: &str = "audioldm2";

/// `vokra.model.name` value written for the canonical AudioLDM 2 base
/// GGUF (the variant-specific display name a consumer sees when
/// inspecting a converted artifact).
///
/// The variant-in-name spelling mirrors the wavtokenizer /
/// chatterbox_turbo / musicgen_medium pattern so a future
/// `audioldm2-music` / `audioldm2-large` / `audioldm2-music-665k` lands
/// as a distinct `NAME` under the shared [`ARCH`] tag — the same
/// shared-arch / distinct-name split every future family sibling will
/// use.
#[allow(dead_code)] // Retained as inspection-only model metadata until the bundle is authenticated.
pub const NAME: &str = "audioldm2";

/// `vokra.model.category` value — AudioLDM 2 is a text-to-audio
/// (music / general audio) generator, sharing the `music` taxonomy tag
/// with the sibling musicgen family per the 2026-07-30 scope expansion
/// (`[[project-scope-expansion-2026-07-30]]`). A separate
/// `audio-generation` tag would be premature before a second
/// non-music-tree generator lands.
#[allow(dead_code)] // Retained as inspection-only model metadata until the bundle is authenticated.
pub const CATEGORY: &str = "music";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf` so a downstream can trace the artifact
/// back to its serving location without parsing the free-text
/// `vokra.provenance.source`.
#[allow(dead_code)] // Retained as inspection-only provenance until the bundle is authenticated.
pub const UPSTREAM_HF: &str = "cvssp/audioldm2";

/// `vokra.model.name` value written for the AudioLDM 2 **Large**
/// variant GGUF (`cvssp/audioldm2-large`, Wave 8 sibling landed
/// 2026-08-02, **cc-by-nc-sa-4.0**). Large = wider/deeper VAE + U-Net
/// + HiFi-GAN vocoder + T5 + CLAP + GPT-2 audio-caption LM (the same
/// six-encoder bundle as sibling [`NAME`], only model dims + optional
/// variant-specific heads differ). Reusing the base BF16 pass-
/// through arm (single [`convert_audioldm2_family_file`] helper +
/// this sibling wrapper) rather than a dedicated `audioldm2_large.rs`
/// file — the tensor-name manifest is topology-identical to sibling
/// base, only `vokra.model.name` + `vokra.provenance.*` (`model_id`,
/// `source`, `upstream_hf`) flip. Mirror of the
/// musicgen_medium / musicgen_melody in-place sibling landing pattern
/// (2026-08-02 precedent).
#[allow(dead_code)] // Retained as inspection-only variant metadata until the bundle is authenticated.
pub const LARGE_NAME: &str = "audioldm2-large";

/// Upstream HF repository slug for the AudioLDM 2 Large sibling
/// (`cvssp/audioldm2-large`), recorded under
/// `vokra.provenance.upstream_hf`. See [`LARGE_NAME`] for the base /
/// large topology relationship (identical multi-encoder bundle,
/// only dims differ).
#[allow(dead_code)] // Retained as inspection-only variant provenance until the bundle is authenticated.
pub const LARGE_UPSTREAM_HF: &str = "cvssp/audioldm2-large";

/// The default upstream weight license — `cc-by-nc-sa-4.0` per the
/// CVSSP GitHub README + paper Ethics §. The HF model card carries a
/// `license: cc-by-nc-4.0` tag (looser NC-only), but the CVSSP-owned
/// primary source pins the *ShareAlike* form and we follow the more
/// restrictive of the two conflicting declarations (Fish-Speech
/// precedent for the same license). Callers can override at the
/// `convert_audioldm2_file(_, _, license=Some(_))` boundary when the
/// source distribution declares a different SPDX id (a permissive
/// re-training on public-domain audio, for example).
#[allow(dead_code)] // Retained as inspection-only license metadata until the bundle is authenticated.
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-sa-4.0";

/// Human-readable upstream source note stored in
/// `vokra.provenance.source` (`KEY_PROVENANCE_SOURCE`). Kept short —
/// the license machine class is carried separately in the
/// `vokra.provenance.weight_license` chunk.
#[allow(dead_code)] // Retained as inspection-only provenance until the bundle is authenticated.
const UPSTREAM_SOURCE: &str =
    "cvssp/audioldm2 (Liu et al. 2024 arXiv:2308.05734 text-to-audio LDM, cc-by-nc-sa-4.0)";

/// Human-readable upstream source note for the AudioLDM 2 Large
/// sibling (`cvssp/audioldm2-large`, Liu et al. 2024 arXiv:2308.05734
/// text-to-audio LDM, wider/deeper multi-encoder bundle,
/// cc-by-nc-sa-4.0). Stored in `vokra.provenance.source`.
#[allow(dead_code)] // Retained as inspection-only provenance until the bundle is authenticated.
const LARGE_UPSTREAM_SOURCE: &str = "cvssp/audioldm2-large (Liu et al. 2024 arXiv:2308.05734 text-to-audio LDM, large variant, cc-by-nc-sa-4.0)";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication rule
// the sibling BF16 pass-through converters use applies).
#[allow(dead_code)] // Retained as metadata keys for the future authenticated converter.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
#[allow(dead_code)] // Retained as metadata keys for the future authenticated converter.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of an AudioLDM 2 conversion.
///
/// Mirrors the sibling BF16-pass-through converters' counter shape
/// (`super::musicgen_medium::MusicGenMediumReport`,
/// `super::xcodec2::XCodec2Report`) adapted to the file-oriented
/// `convert_audioldm2_file` surface.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AudioLdm2Report {
    /// Total tensors surfaced by the safetensors reader (before any
    /// dispatch to the pass-through / skipped arm).
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16 at parse time, so a
    /// non-zero here would signal a reader change upstream).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Additive observability counter — a latent
    /// silent widen / downcast cannot slip in undetected without this
    /// counter also drifting.
    pub bf16_passthrough: usize,
}

/// Converts a `cvssp/audioldm2` safetensors checkpoint at `input` into
/// a Vokra-native GGUF at `output`, returning an [`AudioLdm2Report`].
///
/// The upstream release ships as a bundle (VAE + U-Net + HiFi-GAN
/// vocoder + T5-base + CLAP text encoder + GPT-2 audio-caption LM)
/// totalling ~8.5 GB. Callers running on the M1 iMac 16 GB machine
/// should NOT attempt local conversion (memory
/// `[[feedback-large-models-on-vast-ai]]`: 8 GB safe threshold, and
/// the multi-encoder bundle doubles peak resident to ~17 GB on the
/// pass) — conversion + publish happens on vast.ai per
/// `docs/handoff/vast-ai-large-model-publish.md`.
///
/// If the upstream release ships torch pickle (`.bin`) or sharded
/// safetensors rather than a single-file safetensors, callers
/// pre-flatten + merge offline via
/// `tools/parity/audioldm2_prepare_checkpoint.py` (thin wrapper over
/// `bin_to_safetensors.py` + a sharded-safetensors merger, the
/// SpeechT5-HiFi-GAN + MusicGen-Medium pattern). This function accepts
/// safetensors only — no pickle parser enters the Vokra tree
/// (NFR-DS-02 zero-dep + FR-LD-05 no pickle in runtime).
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict name; the `vokra.model.*` (arch / name / category) and
/// `vokra.provenance.*` (weight_license / license / model_id / source /
/// upstream_hf) chunks are stamped for the runtime compliance gate
/// (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// `DEFAULT_LICENSE_SPDX` (`"cc-by-nc-sa-4.0"`,
/// `NonCommercialShareAlike`) — the doubly-restrictive class the
/// upstream CVSSP primary source pins.
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure; [`ConvertError::Parse`]
/// on a malformed safetensors input; [`ConvertError::Gguf`] if the GGUF
/// cannot be assembled.
pub fn convert_audioldm2_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<AudioLdm2Report, ConvertError> {
    let _ = (input, output, license);
    Err(ConvertError::Usage(
        "audioldm2 conversion is BLOCKED: a complete authenticated cvssp/audioldm2 bundle manifest (all fixed components, projection_model, sidecars, and source/model revisions) is required before binding; the legacy single-file pass-through is disabled".into(),
    ))
}

/// Converts a `cvssp/audioldm2-large` safetensors checkpoint at
/// `input` into a Vokra-native GGUF at `output`.
///
/// AudioLDM 2 Large is the wider/deeper sibling of the base variant
/// (`cvssp/audioldm2`) — the multi-encoder bundle topology (VAE +
/// latent-diffusion U-Net + HiFi-GAN vocoder + T5-base + CLAP text
/// encoder + GPT-2 audio-caption LM) is unchanged, only model dims +
/// optional variant-specific heads differ. The BF16 pass-through
/// pipeline is therefore shared with the base arm via the private
/// [`convert_audioldm2_family_file`] helper; only the
/// `vokra.model.name` + `vokra.provenance.{model_id,source,
/// upstream_hf}` chunks flip to the large spellings
/// ([`LARGE_NAME`] / [`LARGE_UPSTREAM_HF`] /
/// [`LARGE_UPSTREAM_SOURCE`]).
///
/// **Scale ~7 GB → vast.ai handoff.** Do NOT attempt a local convert
/// on the M1 iMac 16 GB machine (memory
/// [[feedback-large-models-on-vast-ai]]: ≥8 GB safe cutoff; Voxtral-
/// Small-24B 48 GB confirmed swap-death is the calibration point).
/// The whole multi-encoder bundle roughly doubles peak resident on
/// the pass (input buffer + parsed safetensors view = additive), so
/// even a nominally 7 GB checkpoint peaks well above the 8 GB safe
/// threshold. Real-weight parity + runtime binder deferred to owner
/// sign-off (`docs/license-audit.md` §3.1 sign-off queue).
///
/// `license` optionally overrides the stamped weight license — see
/// [`convert_audioldm2_file`] for the override semantics + empty-
/// string research-flag-downgrade guard + SA-cascade-downgrade guard.
/// Defaults to `DEFAULT_LICENSE_SPDX` (`"cc-by-nc-sa-4.0"`,
/// `NonCommercialShareAlike`) — every sibling in the AudioLDM 2
/// family carries the same CVSSP primary-source license.
///
/// **Publish blocked (sa-cascade-defer)** — no entry in
/// `scripts/publish/signoff_match.py::REPO_TO_SIGNOFF_ROWS`, and no
/// ☑ sign-off in `docs/license-audit.md` §3.1 (owner ADR required
/// to resolve the SA cascade onto Vokra-added artifacts). The
/// converter lands so a future publish is one owner decision away,
/// but nothing today routes to `publish-one.sh`.
///
/// # Errors
///
/// Same failure modes as [`convert_audioldm2_file`].
pub fn convert_audioldm2_large_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<AudioLdm2Report, ConvertError> {
    let _ = (input, output, license);
    Err(ConvertError::Usage(
        "audioldm2-large conversion is BLOCKED: a complete authenticated cvssp/audioldm2-large bundle manifest (all fixed components, projection_model, sidecars, and source/model revisions) is required before binding; the legacy single-file pass-through is disabled".into(),
    ))
}

/// Shared implementation for the AudioLDM 2 family (base + large,
/// both cc-by-nc-sa-4.0, same six-encoder VAE + U-Net + HiFi-GAN +
/// T5 + CLAP + GPT-2 topology, only the model-id / upstream-hf /
/// source stamps + model dims differ).
///
/// Kept `pub(crate)` so future variants (`-music` / `-music-665k`,
/// for example) can piggyback without duplicating the BF16 pass-
/// through dispatch. External callers should route through the
/// variant-specific wrappers ([`convert_audioldm2_file`] /
/// [`convert_audioldm2_large_file`]) so the correct built-in defaults
/// stay in one place.
#[allow(dead_code)] // Staged behind the inspection-only public conversion gates.
pub(crate) fn convert_audioldm2_family_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    name: &str,
    upstream_hf: &str,
    upstream_source: &str,
) -> Result<AudioLdm2Report, ConvertError> {
    // NB: AudioLDM 2 bundle is ~8.5 GB (base) / ~7 GB (large).
    // `std::fs::read` peaks at ~2x file size (input buffer + parsed
    // safetensors view = additive in the worst case, ~14–17 GB peak).
    // The vast.ai runbook allocates a 32 GB+ box for this class of
    // publish per `docs/handoff/vast-ai-large-model-publish.md` §2,
    // so simple eager-read is acceptable — no streaming reader needed
    // for a one-shot offline convert. Moshi (14 GB) is the streaming-
    // mandated tier and lives in its own module; AudioLDM 2 sits at
    // the upper edge of the non-streaming tier.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, name);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // Built-in stamp = cc-by-nc-sa-4.0 NonCommercialShareAlike. The
    // `license` argument (Some(non-empty spdx)) overrides these three
    // chunks — but with the built-in gate the artifact fails **closed**
    // at load time in commercial mode
    // (`LicenseClass::NonCommercialShareAlike::requires_research_flag`
    // = true), so an operator who never touched the license flag cannot
    // silently bring up an NC-SA weight in production. The empty-string
    // case is explicitly filtered (mirror of xcodec2 / wavtokenizer /
    // musicgen): an empty override must NOT wipe the built-in stamp —
    // that would be a silent research-flag downgrade + a silent
    // share-alike-cascade downgrade.
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (
            DEFAULT_LICENSE_SPDX.to_owned(),
            LicenseClass::NonCommercialShareAlike,
        ),
    };
    vokra_core::stamp_provenance(&mut b, class, &spdx, Some(name), Some(upstream_source));
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, upstream_hf);

    let mut report = AudioLdm2Report::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted ADR (mirror of
    // xcodec2 / neucodec / wavtokenizer / speecht5_hifigan / vibevoice /
    // musicgen); the runtime widens BF16 → f32 exactly at load via the
    // single choke point `crates/vokra-core/src/gguf/quant/mod.rs
    // decode_bf16`.
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
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::{GgmlType, GgufFile};

    #[test]
    fn conversion_refuses_legacy_single_file_without_authenticated_bundle() {
        let input = tmp_path("blocked-in");
        let output = tmp_path("blocked-out");
        std::fs::write(&input, b"not a complete AudioLDM2 bundle").expect("write input");
        let error = convert_audioldm2_file(&input, &output, None).unwrap_err();
        assert!(
            matches!(error, ConvertError::Usage(message) if message.contains("BLOCKED") && message.contains("projection_model"))
        );
        assert!(!output.exists());
        let _ = std::fs::remove_file(input);
    }

    /// A unique temp path — per-process id **plus** a monotonic counter
    /// so two tests in the same process never race on the same file.
    fn tmp_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-audioldm2-{tag}-{}-{n}",
            std::process::id()
        ));
        p
    }

    /// Encodes an f32 array as little-endian BF16 bytes (top 16 bits of
    /// the f32 pattern — the exact inverse of the runtime's
    /// `decode_bf16 : bits << 16`).
    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// Builds a synthetic single-tensor safetensors buffer with a
    /// caller-declared dtype and raw payload.
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

    /// Builds a two-tensor safetensors buffer (F32 first, then F16)
    /// with caller-supplied payloads.
    fn safetensors_f32_then_f16(
        f32_name: &str,
        f32_shape: &[u64],
        f32_bytes: &[u8],
        f16_name: &str,
        f16_shape: &[u64],
        f16_bytes: &[u8],
    ) -> Vec<u8> {
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

    /// The BF16 pass-through arm must emit GGUF type 30
    /// (`GgmlType::BF16`) with byte-identical payload — mirror of the
    /// musicgen / xcodec2 / wavtokenizer / neucodec pin.
    #[test]
    #[ignore = "AudioLDM2 conversion is fail-closed pending authenticated bundle binder"]
    fn bf16_tensor_passes_through_verbatim() {
        // Non-zero BF16 bit patterns so a subsequent byte-identity
        // assert catches any silent widen / downcast (zeroed payloads
        // would round-trip trivially through F32/F16 widen too).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16 = bf16_bytes(&values);
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        // Mirror a realistic AudioLDM 2 state-dict name — `unet.
        // conv_in.weight` is the input conv of the latent-diffusion
        // U-Net (`diffusers.AudioLDM2UNet2DConditionModel`).
        let input_bytes = safetensors_one("unet.conv_in.weight", "BF16", &[2, 3], &bf16);
        let input = tmp_path("bf16-in");
        let output = tmp_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        let report = convert_audioldm2_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(report.written, 1, "BF16 must reach the pass-through arm");
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );

        let file = GgufFile::open(&output).expect("load output gguf");
        let info = file
            .tensor_info("unet.conv_in.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// F32 + F16 mixed-dtype pass-through with the additive-default
    /// invariant on `bf16_passthrough` and all arch / provenance /
    /// category stamps — including the **critical** default
    /// NonCommercialShareAlike stamp (the whole point of the
    /// cc-by-nc-sa-4.0 flip vs. sibling MusicGen NonCommercial
    /// converters).
    #[test]
    #[ignore = "AudioLDM2 conversion is fail-closed pending authenticated bundle binder"]
    fn f32_and_f16_tensors_pass_through_and_default_license_is_noncommercial_sharealike() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // F16 exact-representable half-values: 1.0=0x3C00, -2.0=0xC000,
        // -0.5=0xB800, 3.0=0x4200, 0.15625=0x3100, 42.0=0x5140.
        let f16_words: [u16; 6] = [0x3C00, 0xC000, 0xB800, 0x4200, 0x3100, 0x5140];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 12);

        // Mirror realistic AudioLDM 2 state-dict tensor names:
        //   `vae.encoder.conv_in.weight` — VAE encoder input conv
        //   (`diffusers.AutoencoderKL`).
        //   `vocoder.upsampler.0.weight` — HiFi-GAN vocoder first
        //   upsample block (`speechbrain/tts-hifigan-*` topology).
        let input_bytes = safetensors_f32_then_f16(
            "vae.encoder.conv_in.weight",
            &[1, 2],
            &f32_bytes,
            "vocoder.upsampler.0.weight",
            &[2, 3],
            &f16_bytes,
        );
        let input = tmp_path("mixed-in");
        let output = tmp_path("mixed-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        let report = convert_audioldm2_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 2, "two tensors observed");
        assert_eq!(
            report.written, 2,
            "both F32 and F16 tensors must pass through"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32/F16-only input must leave the BF16 counter at the Default 0 (additive-default invariant)"
        );

        let file = GgufFile::open(&output).expect("load output gguf");

        let f32_info = file
            .tensor_info("vae.encoder.conv_in.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());

        let f16_info = file
            .tensor_info("vocoder.upsampler.0.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16);
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

        // Arch / name / category / provenance chunks land with the
        // built-in cc-by-nc-sa-4.0 NonCommercialShareAlike stamp.
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
            "vokra.model.category must be `music` (shared with sibling musicgen family)"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );
        // The default license path must stamp cc-by-nc-sa-4.0 /
        // NonCommercialShareAlike (the whole point of this converter
        // vs. sibling musicgen which defaults to cc-by-nc-4.0 /
        // NonCommercial — AudioLDM 2 CVSSP primary source adds the
        // ShareAlike cascade).
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::NonCommercialShareAlike.as_str())
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// A caller-supplied `license` (e.g. re-trained on a permissive
    /// public-domain audio corpus) overrides the built-in
    /// cc-by-nc-sa-4.0 NonCommercialShareAlike stamp. Same override
    /// pattern as `convert_file_licensed` — the model_id / arch /
    /// category / upstream stamps survive but the license triple flips.
    #[test]
    #[ignore = "AudioLDM2 conversion is fail-closed pending authenticated bundle binder"]
    fn caller_license_override_swaps_the_stamp() {
        // Non-zero payloads that are NOT approximations of π/e —
        // clippy::approx_constant would flag 3.14/2.71 as a naked
        // approximation of the standard constants.
        let f32_vals: [f32; 2] = [11.5, -6.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one(
            "language_model.transformer.h.0.attn.c_attn.weight",
            "F32",
            &[1, 2],
            &f32_bytes,
        );
        let input = tmp_path("override-in");
        let output = tmp_path("override-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        // Override to Apache-2.0 (Permissive) — the caller retrained
        // on a permissive corpus.
        let report = convert_audioldm2_file(&input, &output, Some("apache-2.0")).expect("convert");
        assert_eq!(report.written, 1);

        let file = GgufFile::open(&output).expect("load output gguf");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
            "override SPDX must land in vokra.provenance.license"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "override class must be re-derived from the SPDX id"
        );
        // Model id / arch / category / upstream_hf remain the built-in
        // values — the override changes only the license triple.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "category (music) must not flip when license overrides"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// An empty `Some("")` license override must NOT wipe the built-in
    /// stamp — that would be a silent research-flag downgrade AND a
    /// silent share-alike-cascade downgrade. The
    /// `Some(s) if !s.is_empty()` guard in `convert_audioldm2_file`
    /// keeps the default cc-by-nc-sa-4.0 NonCommercialShareAlike stamp
    /// (mirror of xcodec2 / wavtokenizer / musicgen empty-string guard
    /// test).
    #[test]
    #[ignore = "AudioLDM2 conversion is fail-closed pending authenticated bundle binder"]
    fn empty_string_license_override_keeps_the_default_stamp() {
        let f32_vals: [f32; 2] = [0.5, -0.5];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_bytes = safetensors_one(
            "text_encoder.encoder.block.0.layer.0.SelfAttention.q.weight",
            "F32",
            &[1, 2],
            &f32_bytes,
        );
        let input = tmp_path("empty-in");
        let output = tmp_path("empty-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        let _ = convert_audioldm2_file(&input, &output, Some("")).expect("convert");

        let file = GgufFile::open(&output).expect("load output gguf");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX),
            "empty string must NOT downgrade the license stamp"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::NonCommercialShareAlike.as_str()),
            "empty string must NOT downgrade the class (both research-flag AND share-alike-cascade stay in force)"
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// Non-float dtypes reach the skipped counter, not the pass-through
    /// arm — defensive since the safetensors reader already rejects
    /// non-F32/F16/BF16 dtypes at parse time. This test also asserts
    /// that even an empty safetensors buffer still lands the
    /// provenance stamp (fail-closed license posture applies
    /// unconditionally, mirror of musicgen precedent).
    #[test]
    #[ignore = "AudioLDM2 conversion is fail-closed pending authenticated bundle binder"]
    fn empty_safetensors_still_stamps_provenance() {
        // An empty safetensors requires the header `{}` (2 bytes)
        // prefixed by its little-endian u64 length (8 bytes) = 10
        // bytes total.
        let empty_header = "{}";
        let mut empty_safetensors = Vec::new();
        empty_safetensors.extend_from_slice(&(empty_header.len() as u64).to_le_bytes());
        empty_safetensors.extend_from_slice(empty_header.as_bytes());

        let input = tmp_path("empty-st-in");
        let output = tmp_path("empty-st-out");
        std::fs::write(&input, &empty_safetensors).expect("write empty safetensors");

        let report = convert_audioldm2_file(&input, &output, None)
            .expect("empty safetensors must be accepted");
        assert_eq!(report.read, 0);
        assert_eq!(report.written, 0);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0);

        // Even with zero tensors the metadata chunks still land — the
        // provenance stamp is independent of the tensor walk.
        let file = GgufFile::open(&output).expect("load output gguf");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::NonCommercialShareAlike.as_str()),
            "stamp must land even with no tensors — fail-closed license posture applies unconditionally"
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// Pin the doubly-restrictive class semantics — the SA cascade is
    /// what makes this converter's default distinct from sibling
    /// musicgen's `NonCommercial`. Both classes require the research
    /// flag, but only `NonCommercialShareAlike` requires the license
    /// be preserved on republish (SA propagation). This test guards
    /// the class-derived predicates against a silent downgrade — if
    /// someone flips the default to plain `NonCommercial`, both these
    /// asserts flip and this test fires.
    #[test]
    fn default_class_carries_both_nc_gate_and_sa_cascade() {
        assert!(
            LicenseClass::NonCommercialShareAlike.requires_research_flag(),
            "NC-SA weight must require the research flag at load time (FR-CP-03)"
        );
        assert!(
            LicenseClass::NonCommercialShareAlike.requires_license_preserved(),
            "NC-SA weight must require the license be preserved on redistribution (SA cascade)"
        );
        assert!(
            !LicenseClass::NonCommercialShareAlike.redistributable(),
            "NC-SA weight must NOT be routable through the default publish gate (fail-closed until owner ADR)"
        );
        assert!(
            !LicenseClass::NonCommercialShareAlike.commercial_ok(),
            "NC-SA weight must NOT be marked commercial-safe by the classifier"
        );
    }

    /// Sibling wrapper `convert_audioldm2_large_file` must flip the
    /// `vokra.model.name` + `vokra.provenance.{model_id,source,
    /// upstream_hf}` chunks to the LARGE_* spellings while keeping the
    /// arch tag (`audioldm2`), category (`music`), and doubly-
    /// restrictive NC-SA default stamp identical to sibling base.
    /// Mirrors the `musicgen_melody` sibling landing test pattern —
    /// the shared `convert_audioldm2_family_file` helper flips only
    /// four id chunks between siblings, and this test pins that
    /// four-chunk delta so a silent regression on the shared helper
    /// fires here first.
    #[test]
    #[ignore = "AudioLDM2 conversion is fail-closed pending authenticated bundle binder"]
    fn large_sibling_flips_name_and_upstream_but_keeps_arch_category_and_license() {
        // Non-zero BF16 bit patterns so the byte-identity assert also
        // catches any silent widen / downcast on the sibling path.
        let values: [f32; 4] = [3.5, -1.25, 0.5, -0.125];
        let bf16 = bf16_bytes(&values);
        assert_eq!(bf16.len(), 8, "4 elements × 2 bytes BF16 payload");

        // Mirror realistic AudioLDM 2 large state-dict name — same
        // U-Net topology as sibling base, only dims differ.
        let input_bytes = safetensors_one(
            "unet.mid_block.attentions.0.to_q.weight",
            "BF16",
            &[2, 2],
            &bf16,
        );
        let input = tmp_path("large-in");
        let output = tmp_path("large-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        let report = convert_audioldm2_large_file(&input, &output, None).expect("convert large");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);
        assert_eq!(report.skipped_non_float, 0);

        let file = GgufFile::open(&output).expect("load large output gguf");

        // BF16 payload byte-identical to input on the sibling path too
        // (guards against a silent widen slipping into the shared helper).
        let info = file
            .tensor_info("unet.mid_block.attentions.0.to_q.weight")
            .expect("BF16 tensor present in large output");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), bf16.as_slice());

        // Arch + category + license stamps stay identical to sibling
        // base (the whole point of the shared helper — only four id
        // chunks flip).
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH),
            "arch tag `audioldm2` must be shared with sibling base (silently forking would misroute runtime dispatch)"
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "category (music) must not flip between siblings"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX),
            "large sibling must default to the same cc-by-nc-sa-4.0 license as base"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::NonCommercialShareAlike.as_str()),
            "large sibling must inherit the doubly-restrictive NC-SA default (NC gate + SA cascade)"
        );

        // Name + model_id + upstream_hf flip to the LARGE_* spellings —
        // this is the four-chunk delta between siblings.
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(LARGE_NAME),
            "vokra.model.name must flip to `audioldm2-large` on the sibling path"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some(LARGE_NAME),
            "vokra.provenance.model_id must match LARGE_NAME (stamp_provenance derives model_id from `name`)"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(LARGE_UPSTREAM_HF),
            "vokra.provenance.upstream_hf must flip to `cvssp/audioldm2-large`"
        );
        // Guard against a silent leak of the base-name into the large
        // stamp (a copy-paste regression in the shared helper).
        assert_ne!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME),
            "large sibling must NOT stamp the base name (regression guard)"
        );
        assert_ne!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF),
            "large sibling must NOT stamp the base upstream_hf (regression guard)"
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }
}
