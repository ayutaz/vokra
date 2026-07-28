//! Weight-license classification primitives (M2-13).
//!
//! This module is intentionally **pure**: it maps license strings / model ids
//! to a [`LicenseClass`] and knows which classes require the research flag. It
//! reads no files and holds no policy — the GGUF reading and the load-time gate
//! live in the parent [`crate::compliance`] module.
//!
//! # Source of truth
//!
//! The built-in registry and the class of each license string are a **machine
//! transcription of `docs/license-audit.md` §3** (the single source of truth,
//! per M2-13-T02). No independent licensing judgement is made here; when a PR
//! adds a model it updates that table and this registry together (FR-MD-13).
//!
//! # weight license ≠ crate/code license
//!
//! [`LicenseClass`] describes the **model weight** license only. It is a wholly
//! separate mechanism from the dependency (crate) license gate, which is
//! `cargo-deny` (NFR-LC-02/04, a CI check). A model can be MIT *code* but
//! CC-BY-NC *weight* (F5-TTS, EnCodec); only the latter is what this classifies.

/// The license class of a model's **weights**, used to decide whether the
/// research flag is required to load it (FR-CP-03).
///
/// Ordering of severity (least to most restricted): [`Permissive`] <
/// [`AttributionRequired`] < [`InheritedRestriction`] / [`Copyleft`] <
/// [`NonCommercial`] / [`NonCommercialShareAlike`] < [`Unknown`]. Everything
/// from [`NonCommercial`] onward is gated ([`Self::requires_research_flag`]);
/// [`Unknown`] is gated deliberately so an unclassifiable weight fails
/// **closed** rather than open.
///
/// [`Permissive`]: LicenseClass::Permissive
/// [`AttributionRequired`]: LicenseClass::AttributionRequired
/// [`NonCommercial`]: LicenseClass::NonCommercial
/// [`NonCommercialShareAlike`]: LicenseClass::NonCommercialShareAlike
/// [`Copyleft`]: LicenseClass::Copyleft
/// [`RedistributionForbidden`]: LicenseClass::RedistributionForbidden
/// [`ConditionalCommercial`]: LicenseClass::ConditionalCommercial
/// [`InheritedRestriction`]: LicenseClass::InheritedRestriction
/// [`Unknown`]: LicenseClass::Unknown
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LicenseClass {
    /// Commercial-friendly with no attribution obligation encoded here:
    /// Apache-2.0 / MIT / BSD / CC0 / ISC / Unlicense (e.g. Whisper, Kokoro,
    /// piper-plus, DAC, WavTokenizer, X-Codec 2). Loads on any path.
    Permissive,
    /// Commercial-OK but attribution is required, e.g. CC-BY-4.0 (Mimi / Moshi
    /// weights). Loads on any path; downstream is expected to honour the
    /// attribution (NOTICE) — a separate, non-gating obligation.
    AttributionRequired,
    /// Non-commercial, e.g. CC-BY-NC-4.0 (F5-TTS, EnCodec) or NVIDIA
    /// Source-Code-License-NC. **Research flag required.**
    NonCommercial,
    /// Non-commercial *and* share-alike, e.g. CC-BY-NC-SA-4.0 (Fish-Speech
    /// v1.4/v1.5). **Research flag required.**
    NonCommercialShareAlike,
    /// Redistribution is permitted **only with the original licence
    /// preserved** — share-alike or strong copyleft (CC-BY-SA, AGPL, GPL,
    /// LGPL). Loading is unrestricted; the obligation is on *republishing*.
    ///
    /// Distinct from [`AttributionRequired`](Self::AttributionRequired):
    /// CC-BY asks only for credit, whereas share-alike propagates to a
    /// converted artifact, so a GGUF derived from a CC-BY-SA weight is itself
    /// CC-BY-SA and cannot be relabelled Apache-2.0. Getting this wrong is a
    /// misrepresentation, not merely a missing credit — which is why
    /// `from_license_str` tests share-alike *before* the plain `cc-by` arm.
    Copyleft,
    /// Redistribution is forbidden by contract or terms of use, **regardless
    /// of any licence string on the artifact**. Never publishable.
    ///
    /// This is categorically unlike the copyleft and non-commercial classes,
    /// which permit redistribution under conditions: here there are no
    /// conditions that make it lawful. Examples: VOICEVOX `.vvm` (its terms
    /// forbid `逆コンパイル・リバースエンジニアリング`, which a format
    /// conversion requires, and separately forbid publishing the method);
    /// CSJ-trained weights (the licence contract defines trained models as
    /// derivative works and the academic tier bars third-party provision);
    /// JSUT / JVS-trained weights (`Re-distribution is not permitted`).
    ///
    /// Never inferred from a licence string — only ever set from an explicit
    /// list, because the prohibition lives in a contract the artifact does not
    /// carry.
    RedistributionForbidden,
    /// Commercial use is permitted **below a stated threshold** (annual
    /// revenue or monthly active users) and needs a separate grant above it —
    /// e.g. LFM Open License v1.0 (revenue >= $10M), the Boson Higgs Audio 2
    /// Community License (>100k annual active users), IndexTTS-2 (>100M MAU
    /// or >CNY 1bn revenue).
    ///
    /// Loading is unrestricted. The threshold is the *downstream user's* to
    /// evaluate, so the obligation this class creates is disclosure: the
    /// threshold must be stated wherever the weight is published.
    ConditionalCommercial,
    /// The licence text carries **usage restrictions that flow to downstream
    /// users** — the Responsible-AI Licence family (`OpenRAIL-M` a.k.a.
    /// `creativeml-openrail-m`, `RAIL-D`, `BigScience-OpenRAIL-M`).
    ///
    /// Loading is unrestricted and commercial use is not per-se barred (the
    /// licences all state so), but the negative use-case list they carry
    /// (weapons, mass surveillance, targeting protected classes, etc.) must
    /// be preserved when a derivative artefact is republished — otherwise a
    /// downstream user cannot see the restriction they are bound by.
    ///
    /// Distinct from [`Copyleft`](Self::Copyleft) even though the two share
    /// the same publish-with-licence-preserved verdict: share-alike
    /// propagates the *licence* (a derivative's terms match the source),
    /// whereas OpenRAIL propagates *use restrictions* (the derivative's use
    /// is still bound by the source's negative use-case list). Same gate
    /// state, different reason — modelling them separately keeps the
    /// obligation legible to the caller.
    InheritedRestriction,
    /// Training-rights unclear / license unstated / unrecognized string
    /// ("要確認"): classification failed. **Research flag required** so an
    /// unknown weight fails closed (never mistaken for permissive).
    Unknown,
}

impl LicenseClass {
    /// Whether loading a weight of this class requires an explicit research
    /// flag (FR-CP-03). True for [`NonCommercial`](Self::NonCommercial),
    /// [`NonCommercialShareAlike`](Self::NonCommercialShareAlike) and
    /// [`Unknown`](Self::Unknown) (fail-closed).
    pub fn requires_research_flag(self) -> bool {
        matches!(
            self,
            Self::NonCommercial | Self::NonCommercialShareAlike | Self::Unknown
        )
    }

    /// Whether **Vokra may republish** a weight of this class — i.e. upload a
    /// converted artifact to a public model hub.
    ///
    /// Deliberately separate from [`Self::requires_research_flag`], which gates
    /// *loading*. The two answers differ for almost every non-permissive class,
    /// and conflating them produces one of two failures: refusing to publish
    /// something that is perfectly publishable under its own terms, or
    /// publishing something whose terms forbid it.
    ///
    /// - [`Copyleft`](Self::Copyleft) is **publishable** — with the original
    ///   licence preserved on the artifact, never relabelled.
    /// - [`ConditionalCommercial`](Self::ConditionalCommercial) is
    ///   **publishable** — the threshold is the downstream user's to evaluate,
    ///   so the obligation is to state it.
    /// - [`RedistributionForbidden`](Self::RedistributionForbidden) is **never**
    ///   publishable, and unlike every other class no condition makes it so.
    /// - The non-commercial classes are an **owner policy decision**, not a
    ///   code one, so they answer `false` here and are re-enabled (if ever)
    ///   explicitly rather than by default.
    /// - [`Unknown`](Self::Unknown) fails closed: an unclassifiable weight is
    ///   not republished.
    pub fn redistributable(self) -> bool {
        matches!(
            self,
            Self::Permissive
                | Self::AttributionRequired
                | Self::Copyleft
                | Self::ConditionalCommercial
                | Self::InheritedRestriction
        )
    }

    /// Whether republishing must carry the **original licence unchanged** —
    /// share-alike / copyleft. Relabelling such an artifact (e.g. publishing a
    /// CC-BY-SA-derived GGUF as Apache-2.0) is a misrepresentation, not a
    /// paperwork slip, so this is what a publishing gate keys on.
    pub fn requires_license_preserved(self) -> bool {
        matches!(
            self,
            Self::Copyleft | Self::NonCommercialShareAlike | Self::InheritedRestriction
        )
    }

    /// Whether **commercial use** of a weight of this class is permitted
    /// outright.
    ///
    /// [`Copyleft`](Self::Copyleft) answers `true`: AGPL / GPL / CC-BY-SA all
    /// permit commercial use, and restrict *redistribution terms* rather than
    /// use. [`ConditionalCommercial`](Self::ConditionalCommercial) answers
    /// `false` because the answer genuinely depends on the user's revenue or
    /// user count, which this type cannot know — the fail-safe reading is
    /// "must be evaluated", not "yes".
    ///
    /// This used to double as the official-zoo admission test (BR-10:
    /// Apache-2.0 / MIT only). Those two questions diverged when the zoo policy
    /// changed to admit copyleft weights under their own licences, so the
    /// publishing question now lives in [`Self::redistributable`] and this
    /// predicate answers only what its name says.
    pub fn commercial_ok(self) -> bool {
        matches!(
            self,
            Self::Permissive
                | Self::AttributionRequired
                | Self::Copyleft
                | Self::InheritedRestriction
        )
    }

    /// Whether downstream must display attribution for this class (CC-BY-4.0,
    /// and the copyleft family, whose licences all carry a BY / notice-
    /// retention term). Advisory (non-gating) — enforced via NOTICE, not this
    /// gate.
    pub fn requires_attribution(self) -> bool {
        matches!(
            self,
            Self::AttributionRequired
                | Self::Copyleft
                | Self::NonCommercialShareAlike
                | Self::InheritedRestriction
        )
    }

    /// The stable canonical name written to / read from
    /// `vokra.provenance.weight_license`. Round-trips with
    /// [`Self::from_class_str`].
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Permissive => "permissive",
            Self::AttributionRequired => "attribution-required",
            Self::NonCommercial => "non-commercial",
            Self::NonCommercialShareAlike => "non-commercial-share-alike",
            Self::Copyleft => "copyleft",
            Self::RedistributionForbidden => "redistribution-forbidden",
            Self::ConditionalCommercial => "conditional-commercial",
            Self::InheritedRestriction => "inherited-restriction",
            Self::Unknown => "unknown",
        }
    }

    /// Parses a canonical class name (the value of
    /// `vokra.provenance.weight_license`). Returns `None` for anything not
    /// produced by [`Self::as_str`] so the caller can fall through to the raw
    /// license string / registry rather than silently trusting a typo.
    pub fn from_class_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "permissive" => Some(Self::Permissive),
            "attribution-required" | "attribution" => Some(Self::AttributionRequired),
            "non-commercial" | "noncommercial" => Some(Self::NonCommercial),
            "non-commercial-share-alike" | "noncommercial-sharealike" => {
                Some(Self::NonCommercialShareAlike)
            }
            "copyleft" | "share-alike" | "sharealike" => Some(Self::Copyleft),
            "redistribution-forbidden" => Some(Self::RedistributionForbidden),
            "conditional-commercial" => Some(Self::ConditionalCommercial),
            "inherited-restriction" | "openrail" | "rail" => Some(Self::InheritedRestriction),
            "unknown" => Some(Self::Unknown),
            _ => None,
        }
    }

    /// Classifies a **raw** weight license string (the value of
    /// `vokra.provenance.license`, e.g. `"CC-BY-NC-4.0"`).
    ///
    /// Fail-closed: any empty / unrecognized string maps to
    /// [`Self::Unknown`] (gated), never to a permissive default. Matching is
    /// locale-independent (ASCII lower-case only; no `strtod`/locale parsing,
    /// NFR-RL-01) and order-sensitive — share-alike is tested before plain
    /// non-commercial, and non-commercial before attribution, because the
    /// tokens are substrings of one another.
    pub fn from_license_str(s: &str) -> Self {
        // Normalize: lower-case, unify separators (space / underscore / dot) to
        // '-' so "CC BY NC 4.0" and "cc-by-nc-4.0" compare equal.
        let norm: String = s
            .trim()
            .to_ascii_lowercase()
            .chars()
            .map(|c| match c {
                ' ' | '_' | '.' | '/' => '-',
                other => other,
            })
            .collect();
        if norm.is_empty() {
            return Self::Unknown;
        }
        let has_nc = norm.contains("-nc") || norm.contains("noncommercial") || norm.contains("nc-");
        let has_sa =
            norm.contains("-sa") || norm.contains("sharealike") || norm.contains("share-alike");
        // Non-commercial family first (CC-BY-NC / CC-BY-NC-SA / NVIDIA -NC).
        if has_nc {
            return if has_sa {
                Self::NonCommercialShareAlike
            } else {
                Self::NonCommercial
            };
        }
        // Responsible-AI licence family (OpenRAIL-M / creativeml-openrail-m /
        // RAIL-D) BEFORE the share-alike arm. These carry usage restrictions
        // that flow downstream — see [`Self::InheritedRestriction`]. Modelled
        // separately from share-alike even though both end up
        // `requires_license_preserved = true`, because the reason it must be
        // preserved differs (usage list vs licence terms). Matching before
        // the share-alike arm is deliberate: share-alike matches on `-sa` /
        // `sharealike` tokens, `openrail` matches on the licence family name
        // — the two sets do not collide, so the order is a semantic choice
        // rather than a substring hazard, but keeping it first keeps the
        // more specific classification in front of the more general one.
        if norm.contains("openrail") || norm.contains("rail-m") {
            return Self::InheritedRestriction;
        }
        // Share-alike and strong copyleft, BEFORE the plain `cc-by` arm.
        //
        // Order matters and the previous ordering was wrong: `cc-by-sa-4.0`
        // contains `cc-by`, so testing attribution first classified every
        // share-alike weight as merely attribution-required. That understates
        // the obligation — CC-BY asks for credit, CC-BY-SA propagates to a
        // converted artifact, so a GGUF built from a CC-BY-SA weight is itself
        // CC-BY-SA. Publishing it as Apache-2.0 would be a misrepresentation.
        //
        // AGPL/GPL/LGPL land here too: redistribution is permitted with the
        // licence preserved, which is a different disposition from `Unknown`
        // (where the fail-closed answer is "we do not know, so refuse").
        if has_sa || norm.contains("agpl") || norm.contains("gpl") || norm.contains("copyleft") {
            return Self::Copyleft;
        }
        // Attribution (CC-BY, not -NC, not -SA): matched after both.
        if norm.contains("cc-by") || norm.starts_with("by-") {
            return Self::AttributionRequired;
        }
        // Permissive families.
        const PERMISSIVE_TOKENS: [&str; 8] = [
            "mit",
            "apache",
            "bsd",
            "cc0",
            "isc",
            "unlicense",
            "mpl",
            "zlib",
        ];
        if PERMISSIVE_TOKENS.iter().any(|t| norm.contains(t)) {
            return Self::Permissive;
        }
        // Anything else (incl. "要確認" / "unknown" / "proprietary"): fail closed.
        Self::Unknown
    }
}

/// Built-in weight-license registry: a machine transcription of
/// `docs/license-audit.md` §3, keyed on a model identifier (the value of
/// `vokra.provenance.model_id`, or a `vokra.model.*` arch/name fallback).
///
/// Returns `None` when the id is not registered; the caller then falls back to
/// [`LicenseClass::Unknown`] (fail-closed). Matching is on the ASCII
/// lower-cased id.
///
/// The first-party runtime models (whisper / piper-plus / silero-vad / campplus)
/// are registered as [`LicenseClass::Permissive`] so their untagged GGUFs keep
/// loading on the default path; the CC-BY-NC entries (F5-TTS / Fish-Speech /
/// EnCodec) are registered as gated so they are rejected there.
pub fn registry_lookup(model_id: &str) -> Option<LicenseClass> {
    let id = model_id.trim().to_ascii_lowercase();
    let class = match id.as_str() {
        // --- first-party / official-zoo permissive (Apache-2.0 / MIT) --------
        "whisper" | "whisper-base" | "whisper-small" | "whisper-medium" | "whisper-large-v3"
        | "whisper-turbo" => LicenseClass::Permissive,
        // SoTA plan Phase 2 (2026-07-24): distil-whisper / distil-large-v3.5 —
        // HuggingFace's distilled Whisper (large-v3 encoder + shrunk 2-layer
        // decoder). MIT weight license per `distil-whisper/distil-large-v3.5`
        // model card (mirrors openai/whisper's MIT posture). Redundant with the
        // `distil-whisper-` / `distil-large-` family walks below, but the
        // exact canonical spellings are listed here so an id lookup returns
        // quickly without hitting the prefix arms.
        "distil-whisper"
        | "distil-whisper-large-v3"
        | "distil-whisper-large-v3.5"
        | "distil-whisper-large-v3_5"
        | "distil-large-v3"
        | "distil-large-v3.5"
        | "distil-large-v3_5" => LicenseClass::Permissive,
        // SoTA plan Phase 5 JA-ASR-2 (2026-07-24): Kotoba Technologies
        // **kotoba-whisper** family — Japanese-distilled Whisper (large-v3
        // encoder + shrunk 2-layer decoder — same tensor topology as
        // distil-large-v3.5, but distilled on ReazonSpeech Japanese audio
        // and released under a different upstream). Weight license =
        // **apache-2.0** per every HF model card in the family
        // (`kotoba-tech/kotoba-whisper-v1.0`, `-v1.1`, `-v2.0`, `-v2.1`,
        // `-bilingual-v1.0`, fetched 2026-07-24 — CLAUDE.md
        // 「ハルシネーション厳禁」). Redundant with the `kotoba-whisper-`
        // family walk below, but the exact canonical spellings are
        // listed here so an id lookup returns quickly without hitting
        // the prefix arms.
        "kotoba-whisper"
        | "kotoba-whisper-v1.0"
        | "kotoba-whisper-v1_0"
        | "kotoba-whisper-v1.1"
        | "kotoba-whisper-v1_1"
        | "kotoba-whisper-v2.0"
        | "kotoba-whisper-v2_0"
        | "kotoba-whisper-v2.1"
        | "kotoba-whisper-v2_1"
        | "kotoba-whisper-bilingual"
        | "kotoba-whisper-bilingual-v1.0"
        | "kotoba-whisper-bilingual-v1_0" => LicenseClass::Permissive,
        "piper-plus" | "piper-plus-mb-istft-vits2" => LicenseClass::Permissive,
        "silero-vad" | "silero-vad-v5" => LicenseClass::Permissive,
        "campplus" | "cam++" => LicenseClass::Permissive,
        "kokoro" | "kokoro-82m" | "cosyvoice" | "cosyvoice2" | "sesame-csm" | "csm-1b"
        | "voxtral" | "openwakeword" => LicenseClass::Permissive,
        // SoTA plan Phase 3 (2026-07-24): FunAudioLLM Fun-CosyVoice3-0.5B —
        // TTS with the CosyVoice2 topology (Qwen2 LLM + chunk-aware CFM +
        // HiFTNet vocoder). Weight license = **apache-2.0** per the model-
        // card header (`huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512`,
        // fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」). Redundant
        // with the `cosyvoice-` / `cosyvoice3-` family walks below, but the
        // exact canonical spellings + `fun-` HF-prefix variants are listed
        // here so an id lookup returns quickly without hitting the prefix
        // arms.
        "cosyvoice3"
        | "fun-cosyvoice3"
        | "fun-cosyvoice3-0.5b"
        | "fun-cosyvoice3-0.5b-2512"
        | "fun-cosyvoice3-0_5b"
        | "fun-cosyvoice3-0_5b-2512" => LicenseClass::Permissive,
        // SoTA plan Phase 3 (2026-07-24): Resemble AI Chatterbox-Multilingual
        // TTS — T3 (Llama_520M backbone) + HiFT-GAN vocoder. Weight license
        // = **MIT** per `github.com/resemble-ai/chatterbox/LICENSE`
        // (Copyright (c) 2025 Resemble AI, fetched 2026-07-24 — CLAUDE.md
        // 「ハルシネーション厳禁」). Redundant with the `chatterbox-` family
        // walk below, but the exact canonical spellings + the two variant
        // tags + the raw checkpoint stems are listed here so an id lookup
        // returns quickly without hitting the prefix arm.
        "chatterbox"
        | "chatterbox-multilingual"
        | "chatterbox-multilingual-v2"
        | "chatterbox-multilingual-v3"
        | "chatterbox-mtl23ls-v2"
        | "chatterbox-mtl23ls-v3"
        | "chatterbox-english"
        | "chatterbox_en" => LicenseClass::Permissive,
        // SoTA plan Phase 3 (2026-07-24): Resemble AI Chatterbox-Turbo —
        // 350M-parameter distilled Turbo variant (GPT-2-medium backbone
        // + 32 kHz S3Gen HiFT-GAN vocoder). Weight license = MIT
        // (same `github.com/resemble-ai/chatterbox/LICENSE` — the whole
        // Chatterbox family, base + Turbo + multilingual variants,
        // ships under a single MIT LICENSE, fetched 2026-07-24 —
        // CLAUDE.md「ハルシネーション厳禁」). Redundant with the
        // `chatterbox-` / `chatterbox_` family walks below, but the
        // canonical id + underscore spelling (== arch tag) + v1 stem +
        // sibling ONNX release id are listed here so an id lookup
        // returns quickly without hitting the prefix arm.
        "chatterbox-turbo"
        | "chatterbox_turbo"
        | "chatterbox-turbo-v1"
        | "chatterbox-turbo-onnx" => LicenseClass::Permissive,
        // SoTA plan Phase 3 (2026-07-24): Resemble AI Chatterbox-Nano —
        // compact 110M-parameter Chatterbox variant advertised at
        // ~3× realtime on an 8-core CPU (Llama_520M backbone + 32 kHz
        // S3Gen HiFT-GAN vocoder + GPT-2 text tokenizer + distilled
        // 1-step mel decoder). Weight license = MIT (same
        // `github.com/resemble-ai/chatterbox/LICENSE` — the whole
        // Chatterbox family, base + Turbo + Nano + multilingual variants,
        // ships under a single MIT LICENSE, fetched 2026-07-24 —
        // CLAUDE.md「ハルシネーション厳禁」). Redundant with the
        // `chatterbox-` / `chatterbox_` family walks below, but the
        // canonical id + underscore spelling (== arch tag) + v1 stem
        // are listed here so an id lookup returns quickly without
        // hitting the prefix arm.
        "chatterbox-nano" | "chatterbox_nano" | "chatterbox-nano-v1" => LicenseClass::Permissive,
        // SoTA plan Phase 3 (2026-07-24): Alibaba Qwen3-TTS-12Hz-0.6B-Base —
        // discrete multi-codebook LM TTS. Weight license = **apache-2.0**
        // **end-to-end** — LM + codec + tokenizer + speaker encoder all
        // under a single apache-2.0 grant
        // (`huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base` model-card
        // `license: apache-2.0`, fetched 2026-07-24 — CLAUDE.md
        // 「ハルシネーション厳禁」). The M2-13 gate passes commercially
        // without any attribution obligation on the runtime side. The
        // exact canonical spellings + arch-tag underscore variant + the
        // common short forms are listed here so an id lookup returns
        // quickly without hitting the `qwen3-tts-` prefix arm below.
        "qwen3-tts"
        | "qwen3_tts"
        | "qwen3-tts-0.6b"
        | "qwen3-tts-0_6b"
        | "qwen3-tts-12hz-0.6b-base"
        | "qwen3-tts-12hz-0_6b-base"
        | "qwen3-tts-12hz-0.6b" => LicenseClass::Permissive,
        // SoTA plan Phase 4 (2026-07-24): OpenBMB VoxCPM-0.5B — end-to-end
        // diffusion-autoregressive TTS. Weight license = **apache-2.0**
        // **end-to-end** — code + weight all under a single apache-2.0
        // grant (`huggingface.co/openbmb/VoxCPM-0.5B` model-card
        // `license: apache-2.0` + `github.com/OpenBMB/VoxCPM/LICENSE`,
        // fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」). The
        // M2-13 gate passes commercially without any attribution
        // obligation on the runtime side. The exact canonical spellings
        // + arch-tag underscore variant + the common short forms are
        // listed here so an id lookup returns quickly without hitting
        // the `voxcpm-` prefix arm below.
        "voxcpm" | "voxcpm2" | "voxcpm-0.5b" | "voxcpm-0_5b" | "voxcpm-0.5b-base"
        | "voxcpm-0_5b-base" => LicenseClass::Permissive,
        // SoTA plan Phase 4 (2026-07-24): Microsoft VibeVoice-1.5B —
        // long-form multi-speaker end-to-end diffusion-autoregressive
        // TTS. Weight license = **MIT** **end-to-end** — code + weight
        // all under a single MIT grant
        // (`huggingface.co/microsoft/VibeVoice-1.5B` model-card
        // `license: MIT` + `github.com/microsoft/VibeVoice/blob/main/
        // LICENSE`, fetched 2026-07-24 — CLAUDE.md「ハルシネーション
        // 厳禁」). MIT is a `Permissive` license class — same
        // commercial verdict as apache-2.0 (no runtime-side attribution
        // obligation on the runtime side), just a different SPDX
        // string. The exact canonical spellings + arch-tag variant +
        // common short forms are listed here so an id lookup returns
        // quickly without hitting the `vibevoice-` prefix arm below.
        //
        // Note the model card carries usage restrictions (no voice
        // impersonation without recorded consent, no deepfakes, no
        // non-English/Chinese, no non-speech audio) — those are
        // **policy obligations**, not license terms; the MIT LICENSE
        // itself carries no field-of-use restriction, so
        // `Permissive` remains the correct license class.
        "vibevoice"
        | "vibevoice-1.5b"
        | "vibevoice-1_5b"
        | "vibevoice-1.5b-base"
        | "vibevoice-1_5b-base" => LicenseClass::Permissive,
        // SoTA plan Phase 5 JA-TTS-1 (2026-07-24): Aratako Irodori-TTS
        // family — Japanese Rectified-Flow Diffusion Transformer TTS.
        // Weight license = **MIT** per `github.com/Aratako/Irodori-TTS/blob/main/LICENSE`
        // (verified via `gh api /repos/Aratako/Irodori-TTS/license` →
        // `MIT`, fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」).
        // Redundant with the `irodori-` prefix arm below, but the exact
        // canonical spellings — canonical id + underscore arch tag + v1
        // / v2 / v2-VoiceDesign / v3 / 600M-v3-VoiceDesign HF release ids
        // — are listed here so an id lookup returns quickly without
        // hitting the prefix arm.
        "irodori"
        | "irodori-tts"
        | "irodori_tts"
        | "irodori-tts-500m"
        | "irodori-tts-500m-v2"
        | "irodori-tts-500m-v2-voicedesign"
        | "irodori-tts-500m-v3"
        | "irodori-tts-500m-v3-base"
        | "irodori-tts-600m-v3-voicedesign" => LicenseClass::Permissive,
        // Commercial-OK codecs (FR-OP-32): DAC / WavTokenizer = MIT.
        //
        // ⚠ X-Codec 2 (`x-codec-2` / `xcodec2`) was previously listed here as
        // `Permissive`, based on the earlier reading that the whole family
        // shipped MIT. That was **wrong for the weight-distribution repo**:
        // the HF card at `huggingface.co/HKUSTAudio/xcodec2` carries
        // `license: cc-by-nc-4.0` on its YAML front-matter (CC-verified
        // 2026-07-15; sign-off 2026-07-23 yousan = ☑ Research-only,
        // `docs/license-audit.md` §3.1). The **code** at
        // `github.com/zhenye234/X-Codec-2.0` remains MIT — but the weight
        // class is what M2-13 gates on, and the weight-distribution repo
        // governs the license of the redistributed artifact. So xcodec2
        // now lives on the NonCommercial arm below (with F5-TTS / EnCodec),
        // fail-closed against silent commercial use.
        "dac" | "wavtokenizer" => LicenseClass::Permissive,
        // SoTA plan Phase 1-4 (2026-07-24): nari-labs Dia-1.6B — Apache 2.0
        // code + weight (docs/license-audit.md, model card).
        "dia" | "dia-1.6b" | "dia-1_6b" => LicenseClass::Permissive,
        // SoTA plan Phase 1-5 (2026-07-24): Zyphra Zonos-v0.1-transformer —
        // Apache 2.0 code + weight (HF `Zyphra/Zonos-v0.1-transformer`
        // model card `license: apache-2.0`, docs/tickets/sota-coverage-
        // plan-2026-07-22.md §3.3). Both HF variants (transformer, hybrid)
        // resolve permissive by prefix below; the canonical id + the
        // `zonos-v0.1` short form are listed here for a lookup that does
        // not require the prefix walk.
        "zonos" | "zonos-v0.1" | "zonos-v0_1" => LicenseClass::Permissive,
        // --- attribution-required (CC-BY-4.0) --------------------------------
        "mimi" | "moshi" => LicenseClass::AttributionRequired,
        // SoTA plan Phase 2 (2026-07-24): Kyutai STT-2.6B-EN — English
        // streaming ASR (decoder-only over Mimi tokens). Weight license =
        // CC-BY 4.0 (`huggingface.co/kyutai/stt-2.6b-en` model card;
        // `docs/license-audit.md` Kyutai row). The M2-13 gate passes
        // commercially *and* the FR-MD-09 attribution surface activates.
        "kyutai-stt" | "kyutai-stt-2.6b-en" | "kyutai-stt-2.6b" | "stt-2.6b-en" => {
            LicenseClass::AttributionRequired
        }
        // SoTA plan Phase 2 (2026-07-24): NVIDIA Parakeet-TDT-0.6B-v3 —
        // English ASR (FastConformer encoder + TDT decoder). Weight
        // license = CC-BY 4.0 (`huggingface.co/nvidia/parakeet-tdt-0.6b-v3`
        // model card explicitly states "Use of this model is governed by
        // the CC-BY-4.0 license"). The M2-13 gate passes commercially
        // *and* the FR-MD-09 attribution surface activates. Note this
        // resolves to CC-BY 4.0 (not the Apache-2.0 permissive one) —
        // NVIDIA's own model card carries the CC-BY-4.0 grant.
        "parakeet-tdt" | "parakeet-tdt-0.6b-v3" | "parakeet-tdt-0.6b" | "parakeet" => {
            LicenseClass::AttributionRequired
        }
        // SoTA plan Phase 2 (2026-07-24): NVIDIA Parakeet-CTC-1.1B —
        // English ASR (FastConformer encoder + CTC head, no RNN-T
        // prediction network). Weight license = CC-BY 4.0
        // (`huggingface.co/nvidia/parakeet-ctc-1.1b` model card explicitly
        // states "License to use this model is covered by the CC-BY-4.0").
        // The M2-13 gate passes commercially *and* the FR-MD-09
        // attribution surface activates. Redundant with the
        // `parakeet-` family walk below, but kept as an explicit
        // exact-match arm for parity with the parakeet-tdt arm above and
        // so an id search returns the canonical spellings quickly.
        "parakeet-ctc" | "parakeet-ctc-1.1b" => LicenseClass::AttributionRequired,
        // SoTA plan Phase 2 (2026-07-24): NVIDIA Canary-1B-v2 —
        // multilingual multi-task ASR / AST (25 European languages;
        // FastConformer encoder + Transformer AED decoder). Weight
        // license = CC-BY 4.0 (`huggingface.co/nvidia/canary-1b-v2`
        // model card explicitly states "CC-BY-4.0" for the model
        // weights). The M2-13 gate passes commercially *and* the
        // FR-MD-09 attribution surface activates. Redundant with the
        // `canary-` family walk below, but kept as an explicit
        // exact-match arm for parity with the parakeet-tdt /
        // parakeet-ctc arms above.
        "canary" | "canary-1b-v2" => LicenseClass::AttributionRequired,
        // SoTA plan Phase 2 (2026-07-24): Meta omniASR-CTC-1B — the
        // Omnilingual ASR family's 1B wav2vec 2.0 CTC checkpoint
        // (`facebook/omniASR-CTC-1B` — 1600+ languages). Weight
        // license = **Apache-2.0** (`huggingface.co/facebook/omniASR-CTC-1B`
        // model-card `license: apache-2.0` — confirmed via the HF
        // model API `cardData.license`). The corpus dataset ships
        // CC-BY-4.0 separately, but the model weights are Apache-2.0
        // — so this resolves to `Permissive`, NOT `AttributionRequired`
        // like NVIDIA's Parakeet-CTC / Canary. Redundant with the
        // `omniasr-ctc-` family walk below, but kept as an explicit
        // exact-match arm so an id lookup returns the canonical
        // spellings quickly.
        "omniasr-ctc" | "omniasr-ctc-1b" => LicenseClass::Permissive,
        // SoTA plan Phase 5 fleet (2026-07-28): 12 BF16 pass-through
        // skeleton wire-ups. All 12 modules default to Permissive
        // (MIT or Apache-2.0 per each module's docstring — see
        // `crates/vokra-convert/src/models/{kimi_audio,step_audio2_mini,
        // baichuan_audio,speechtokenizer,funcodec,xy_tokenizer,bicodec,
        // neucodec,ecapa_tdnn,wespeaker,speaker_3d,emotion2vec}.rs`).
        // For each, the arch tag (underscore, == `vokra.model.arch`),
        // the CLI slug (hyphen, canonical `--model` spelling), and the
        // NAME model-card id (== `vokra.provenance.model_id`) are all
        // registered so an untagged GGUF resolves permissive on the
        // fallback path regardless of which of the three id forms it
        // carries. A caller shipping the artifact under a non-permissive
        // SPDX id overrides at the outer `--license <spdx>` boundary.
        "kimi-audio" | "kimi_audio" | "kimi-audio-7b-instruct" | "kimi-audio-7b" => {
            LicenseClass::Permissive
        }
        "step-audio2-mini" | "step_audio2_mini" | "step-audio-2-mini" => LicenseClass::Permissive,
        "baichuan-audio" | "baichuan_audio" => LicenseClass::Permissive,
        "speechtokenizer" | "speech-tokenizer" | "speech_tokenizer" => LicenseClass::Permissive,
        "funcodec"
        | "fun-codec"
        | "fun_codec"
        | "funcodec-encodec-zh-en-16k-nq32-ds320"
        | "funcodec-encodec-zh_en"
        | "funcodec-encodec-zh-en" => LicenseClass::Permissive,
        "xy-tokenizer" | "xy_tokenizer" | "xy-tokenizer-ttsd-v0" | "xy_tokenizer_ttsd_v0" => {
            LicenseClass::Permissive
        }
        "bicodec" | "bi-codec" | "bi_codec" | "spark-tts-bicodec" => LicenseClass::Permissive,
        "neucodec" | "neu-codec" | "neu_codec" => LicenseClass::Permissive,
        "ecapa-tdnn" | "ecapa_tdnn" | "spkrec-ecapa-voxceleb" => LicenseClass::Permissive,
        "wespeaker" | "we-speaker" | "we_speaker" | "wespeaker-voxceleb-resnet34-lm" => {
            LicenseClass::Permissive
        }
        "speaker-3d"
        | "speaker_3d"
        | "3d-speaker"
        | "eres2net"
        | "speech_eres2net_sv_zh-cn_16k-common" => LicenseClass::Permissive,
        "emotion2vec" | "emotion-2vec" | "emotion2vec-plus-large" => LicenseClass::Permissive,
        // SoTA plan Phase 5 JA-TTS-2 (2026-07-24): ESPnet-family
        // Japanese plain VITS (JSUT / JVS / COEIROINK deployments +
        // any downstream that consumes the shared `vits-ja` arch tag).
        // The **architecture** rides Apache-2.0 (ESPnet's
        // `espnet2/gan_tts/vits/` source) + MIT (`jaywalnut310/vits`
        // reference implementation) and is always independently
        // implementable. The **trained weight**, however, is bound by
        // the corpus terms it was trained on:
        //
        // - **JSUT** (`sites.google.com/site/shinnosuketakamichi/publication/jsut`)
        //   pins single-speaker Japanese TTS training data with the
        //   explicit clause *"Re-distribution is not permitted"*.
        // - **JVS** (`sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus`)
        //   pins the 100-speaker Japanese corpus with the same
        //   re-distribution ban.
        // - **COEIROINK** ships per-character licence terms that a
        //   converter cannot machine-check.
        //
        // The default class is therefore `RedistributionForbidden` —
        // never a downstream permissive default — and a user who
        // trained on a permissive corpus overrides at conversion time
        // via `vokra-convert --license <spdx>`. Aliases cover the
        // canonical arch tag + the underscore variant + the three
        // upstream deployment ids (ESPnet-JSUT / ESPnet-JVS / COEIROINK).
        //
        // (See `crates/vokra-models/src/vits_ja/mod.rs` module doc for
        // the JSUT / JVS terms + the `support the architecture, refuse
        // the weights` rationale from
        // `docs/tickets/sota-coverage-plan-2026-07-22.md` §2.4.)
        "vits-ja" | "vits_ja" | "espnet-vits-ja" | "espnet-jsut-vits" | "espnet-jvs-vits"
        | "coeiroink-vits" => LicenseClass::RedistributionForbidden,
        // --- gated: CC-BY-NC (research flag) ---------------------------------
        //
        // X-Codec 2 (`x-codec-2` / `xcodec2`, SoTA plan Phase 5 codec)
        // joined this arm 2026-07-28 after the CC-verified check of
        // `huggingface.co/HKUSTAudio/xcodec2` (front-matter
        // `license: cc-by-nc-4.0`; sign-off 2026-07-23 yousan =
        // ☑ Research-only, `docs/license-audit.md` §3.1). See the note on
        // the DAC / WavTokenizer arm above for the reason the earlier
        // Permissive listing was wrong for the weight-distribution repo.
        "f5-tts" | "encodec" | "x-codec-2" | "xcodec2" => LicenseClass::NonCommercial,
        // --- gated: CC-BY-NC-SA (research flag) ------------------------------
        "fish-speech" | "fish-speech-v1.4" | "fish-speech-v1.5" => {
            LicenseClass::NonCommercialShareAlike
        }
        // --- gated: unknown training rights (research flag, fail-closed) -----
        "rvc" | "rvc-v2" | "gpt-sovits" | "e2-tts" | "styletts2" | "styletts-2" => {
            LicenseClass::Unknown
        }
        // First-party runtime model **families**: a specific variant id (e.g.
        // `piper-plus-multilingual-6lang`, `whisper-base.en`) is still one of the
        // Apache-2.0 / MIT first-party archs, so it resolves permissive like its
        // canonical id above — otherwise a stock voice's untagged GGUF would
        // fail-closed. The prefixes are the first-party families ONLY; the gated
        // CC-BY-NC families are matched exactly above and any *unlisted* variant
        // of them still falls through to `Unknown` (fail-closed), never permissive.
        _ if id.starts_with("piper-plus")
            || id.starts_with("whisper")
            || id.starts_with("silero-vad")
            || id.starts_with("campplus")
            || id.starts_with("cam++")
            || id.starts_with("kokoro")
            // CosyVoice2 first-party family (Apache 2.0 code + weight,
            // docs/license-audit.md, CLAUDE.md モデル表): a specific variant
            // id like `cosyvoice2-0.5b` is still Apache 2.0. Guarded on the
            // dash so `cosyvoicexyz` cannot slip through.
            || id.starts_with("cosyvoice2-")
            || id.starts_with("cosyvoice-")
            // Fun-CosyVoice3 first-party family (apache-2.0 weight — SoTA
            // plan Phase 3, 2026-07-24): a specific HF variant id like
            // `fun-cosyvoice3-0.5b-2512` or a future
            // `fun-cosyvoice3-3b` still resolves permissive. Guarded on
            // the dash so unrelated ids (`cosyvoice3` prefixed anything
            // the family doesn't ship) cannot slip through. The
            // `fun-cosyvoice3-` prefix covers the canonical HF release
            // spellings; `cosyvoice3-` covers the short-form aliases
            // (`cosyvoice3-0.5b`).
            || id.starts_with("fun-cosyvoice3-")
            || id.starts_with("cosyvoice3-")
            // nari-labs Dia first-party family (Apache 2.0 code + weight —
            // SoTA plan Phase 1-4, 2026-07-24): a specific variant id like
            // `dia-1.6b` or a future `dia-3b` still resolves permissive.
            // Guarded on the dash so `diagnostics` cannot slip through.
            || id.starts_with("dia-")
            // Zyphra Zonos first-party family (Apache 2.0 code + weight —
            // SoTA plan Phase 1-5, 2026-07-24): specific HF variant ids
            // like `zonos-v0.1-transformer` / `zonos-v0.1-hybrid` still
            // resolve permissive without being individually listed. The
            // dash guard keeps unrelated ids from matching.
            || id.starts_with("zonos-")
            // Meta omniASR-CTC family (Apache-2.0 weight — SoTA plan
            // Phase 2, 2026-07-24): specific HF variant ids like
            // `omniasr-ctc-1b` or a future `omniasr-ctc-3b` /
            // `omniasr-ctc-7b` (paired 300M / 3B / 7B checkpoints ship
            // in the same fairseq2 registry under Apache-2.0 too) still
            // resolve permissive. Guarded on the dash so unrelated ids
            // (`omniasr-ctc` prefixed anything the family doesn't
            // ship) cannot slip through. This is a POSITIVE-license
            // Meta family — distinct from NVIDIA's Parakeet-CTC /
            // Canary which ship CC-BY 4.0.
            || id.starts_with("omniasr-ctc-")
            // HuggingFace distil-whisper family (MIT weight — SoTA plan
            // Phase 2, 2026-07-24): a specific HF variant id like
            // `distil-whisper-large-v3.5` or a future
            // `distil-whisper-small.en` still resolves permissive.
            // Guarded on the dash so unrelated ids (`distil-whisper`
            // prefixed anything the family doesn't ship) cannot slip
            // through. Mirrors openai/whisper's MIT posture — the
            // family ships MIT code + MIT weights.
            || id.starts_with("distil-whisper-")
            // Canonical shorthand spelling used by the distil-whisper
            // release checkpoints themselves (`distil-large-v3`,
            // `distil-large-v3.5`, and any future `distil-large-*` /
            // `distil-medium-*` / `distil-small-*` sibling). Guarded
            // on the dash so unrelated ids (`distil-largely-anything`)
            // cannot slip through. All distil-whisper variants ship
            // MIT.
            || id.starts_with("distil-large-")
            || id.starts_with("distil-medium-")
            || id.starts_with("distil-small-")
            // Kotoba Technologies kotoba-whisper family (Apache-2.0 weight —
            // SoTA plan Phase 5 JA-ASR-2, 2026-07-24): specific HF variant
            // ids like `kotoba-whisper-v2.0` / `kotoba-whisper-bilingual-v1.0`
            // or a future `kotoba-whisper-v3.0` still resolve permissive.
            // Guarded on the dash so unrelated ids (`kotoba-whisper` prefixed
            // anything the family doesn't ship) cannot slip through. Distinct
            // upstream from distil-whisper (Japanese-fine-tuned by Kotoba
            // Technologies, released under apache-2.0 rather than MIT), but
            // shares the same tensor topology (large-v3 encoder + 2-layer
            // decoder).
            || id.starts_with("kotoba-whisper-")
            // Resemble AI Chatterbox family (MIT weight — SoTA plan
            // Phase 3, 2026-07-24): specific HF variant ids like
            // `chatterbox-multilingual-v3` / `chatterbox-nano` /
            // `chatterbox-turbo` (a future release) or the raw
            // `chatterbox-mtl23ls-v3` checkpoint stem still resolve
            // permissive. Guarded on the dash so unrelated ids
            // (`chatterbox-something-not-shipped-by-resemble`) still
            // land under this permissive arm — which is correct here:
            // the whole family ships under a single MIT LICENSE at
            // `github.com/resemble-ai/chatterbox/LICENSE`, unlike NVIDIA's
            // multi-checkpoint releases where different sizes can carry
            // different licences. The `chatterbox_` alias variant covers
            // the `chatterbox_en` underscore spelling and any future
            // underscore stem.
            || id.starts_with("chatterbox-")
            || id.starts_with("chatterbox_")
            // Alibaba Qwen3-TTS first-party family (apache-2.0
            // end-to-end — SoTA plan Phase 3, 2026-07-24): specific HF
            // variant ids like `qwen3-tts-12hz-0.6b-base` /
            // `qwen3-tts-12hz-0.6b-customvoice` /
            // `qwen3-tts-12hz-0.6b-voicedesign` or a future
            // `qwen3-tts-12hz-1.7b-*` still resolve permissive. Guarded
            // on the dash so unrelated ids (`qwen3-ttsomething`) cannot
            // slip through. The `qwen3_tts` (underscore) alias covers
            // the arch tag spelling only — a future underscore variant
            // family would need its own explicit prefix.
            || id.starts_with("qwen3-tts-")
            // OpenBMB VoxCPM first-party family (apache-2.0 end-to-end —
            // SoTA plan Phase 4, 2026-07-24): specific HF variant ids
            // like `voxcpm-0.5b` / a future `voxcpm-0.5b-customvoice` /
            // `voxcpm-1.5b` still resolve permissive. Guarded on the
            // dash so unrelated ids (`voxcpmsomething`) cannot slip
            // through. The `voxcpm2` (arch-tag) alias covers only the
            // arch-tag spelling — a future underscore variant family
            // would need its own explicit prefix.
            || id.starts_with("voxcpm-")
            // Microsoft VibeVoice first-party family (MIT end-to-end —
            // SoTA plan Phase 4, 2026-07-24): specific HF variant ids
            // like `vibevoice-1.5b` / a future `vibevoice-7b` /
            // `vibevoice-large` still resolve permissive. Guarded on
            // the dash so unrelated ids (`vibevoiceanything`) cannot
            // slip through. The `vibevoice` (arch-tag) alias covers
            // only the arch-tag spelling — a future underscore
            // variant family would need its own explicit prefix.
            //
            // MIT is a `Permissive` license class — same commercial
            // verdict as apache-2.0. The model card's usage
            // restrictions (no impersonation without recorded consent,
            // no deepfakes, English/Chinese only, no non-speech
            // audio) are **policy obligations**, not license terms;
            // the MIT LICENSE itself carries no field-of-use
            // restriction, so `Permissive` remains correct.
            || id.starts_with("vibevoice-")
            // Aratako Irodori-TTS first-party family (MIT end-to-end —
            // SoTA plan Phase 5 JA-TTS-1, 2026-07-24): specific HF
            // variant ids like `irodori-tts-500m-v3` /
            // `irodori-tts-600m-v3-voicedesign` / a future 2.5B variant
            // still resolve permissive. Guarded on the dash so unrelated
            // ids (`irodori-anything-else-unowned`) cannot slip through.
            // The `irodori` / `irodori-tts` / `irodori_tts` arch-tag
            // aliases are pinned in the fast-path arm above.
            //
            // MIT is a `Permissive` license class — same commercial
            // verdict as apache-2.0; the model card's `text_tokenizer_repo
            // = "llm-jp/llm-jp-3-150m"` (Apache-2.0) transitively inherits
            // Permissive too.
            || id.starts_with("irodori-tts-")
            || id.starts_with("irodori-") =>
        {
            LicenseClass::Permissive
        }
        // Kyutai STT family (SoTA plan Phase 2, 2026-07-24): a specific
        // variant id like `kyutai-stt-6.7b-multilingual` (or a future
        // `kyutai-stt-1b-en`) still resolves attribution-required. Guarded
        // on the dash so unrelated ids (`kyutai-stt-x-something-else`)
        // cannot slip through into the permissive bucket by accident. Note
        // this arm resolves to CC-BY 4.0, not the Apache-2.0 permissive
        // one — Kyutai's audio/text checkpoints ship CC-BY 4.0.
        _ if id.starts_with("kyutai-stt-") => LicenseClass::AttributionRequired,
        // Parakeet family (SoTA plan Phase 2, 2026-07-24): a specific
        // variant id like `parakeet-tdt-1.1b` or `parakeet-rnnt-1.1b`
        // still resolves attribution-required. Guarded on the dash so
        // unrelated ids cannot slip through. NVIDIA's whole Parakeet
        // family ships under CC-BY 4.0 (per the 0.6B-v3 model card + the
        // NeMo release convention).
        _ if id.starts_with("parakeet-") => LicenseClass::AttributionRequired,
        // Canary family (SoTA plan Phase 2, 2026-07-24): a specific
        // variant id like `canary-1b-v2-en` or a future `canary-3b-v3`
        // still resolves attribution-required. Guarded on the dash so
        // unrelated ids (`canary-` prefixed anything the family doesn't
        // ship) cannot slip through into the permissive bucket by
        // accident. NVIDIA's whole Canary family ships under CC-BY 4.0
        // (per the 1B-v2 model card).
        _ if id.starts_with("canary-") => LicenseClass::AttributionRequired,
        _ => return None,
    };
    Some(class)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The bug this class ordering exists to prevent.** `cc-by-sa-4.0`
    /// contains the substring `cc-by`, so an attribution-first match reported
    /// every share-alike weight as merely attribution-required. That is not a
    /// pedantic distinction: CC-BY asks for credit, CC-BY-SA propagates to a
    /// converted artifact, so a GGUF built from a CC-BY-SA weight is itself
    /// CC-BY-SA. Publishing it under an Apache-2.0 label would misstate the
    /// terms a downstream user is bound by.
    ///
    /// Concretely load-bearing today: Style-Bert-VITS2's mandatory runtime
    /// BERT (`ku-nlp/deberta-v2-large-japanese-char-wwm`) and the JVNV corpus
    /// weights are both `cc-by-sa-4.0`.
    #[test]
    fn share_alike_is_copyleft_not_merely_attribution() {
        for s in ["cc-by-sa-4.0", "CC BY SA 4.0", "cc_by_sa_3.0"] {
            assert_eq!(
                LicenseClass::from_license_str(s),
                LicenseClass::Copyleft,
                "{s} must classify as copyleft, not attribution-required"
            );
        }
        // Plain CC-BY must NOT be swept up by the same arm.
        for s in ["cc-by-4.0", "CC BY 4.0"] {
            assert_eq!(
                LicenseClass::from_license_str(s),
                LicenseClass::AttributionRequired,
                "{s} is attribution-only"
            );
        }
        // Non-commercial still wins over share-alike (it is the stronger bar).
        assert_eq!(
            LicenseClass::from_license_str("cc-by-nc-sa-4.0"),
            LicenseClass::NonCommercialShareAlike
        );
    }

    /// AGPL / GPL weights used to fall through to `Unknown`, which fails
    /// closed and therefore demanded a research flag to *load*. That was an
    /// artifact of not recognising the string, not a considered position:
    /// these licences do not restrict use at all. They restrict the terms of
    /// redistribution, which is a different gate.
    #[test]
    fn strong_copyleft_is_recognised_and_loadable() {
        for s in ["agpl-3.0", "AGPL-3.0-only", "gpl-3.0", "lgpl-2.1"] {
            let c = LicenseClass::from_license_str(s);
            assert_eq!(c, LicenseClass::Copyleft, "{s}");
            assert!(!c.requires_research_flag(), "{s}: loading is unrestricted");
            assert!(c.commercial_ok(), "{s}: commercial use is permitted");
            assert!(
                c.redistributable(),
                "{s}: republishable with the licence kept"
            );
            assert!(c.requires_license_preserved(), "{s}: may not be relabelled");
        }
    }

    /// The publishing gate and the loading gate answer different questions,
    /// and the classes where they disagree are exactly the ones worth pinning.
    #[test]
    fn redistribution_and_loading_are_separate_questions() {
        use LicenseClass::*;
        // (class, may load without a research flag, may Vokra republish)
        for (c, loadable, publishable) in [
            (Permissive, true, true),
            (AttributionRequired, true, true),
            (Copyleft, true, true),
            (ConditionalCommercial, true, true),
            // Use-restriction propagates downstream but licence itself allows
            // publishing with the restrictions preserved. Same publish verdict
            // as Copyleft, distinct semantic (see the variant docstring).
            (InheritedRestriction, true, true),
            // Contractually barred: loading a weight you legitimately hold is
            // fine; Vokra handing it to a third party is not.
            (RedistributionForbidden, true, false),
            // Owner policy decision, so `false` until explicitly re-enabled.
            (NonCommercial, false, false),
            (NonCommercialShareAlike, false, false),
            (Unknown, false, false),
        ] {
            assert_eq!(
                !c.requires_research_flag(),
                loadable,
                "{c:?}: loadable-without-flag"
            );
            assert_eq!(c.redistributable(), publishable, "{c:?}: publishable");
        }
    }

    /// `RedistributionForbidden` must never be reachable by parsing a licence
    /// string. The prohibition lives in a contract the artifact does not carry
    /// (VOICEVOX's reverse-engineering ban, the CSJ licence agreement, the
    /// JSUT/JVS corpus terms), so inferring it from text would be guessing —
    /// and guessing the *other* way would silently authorise a publish.
    #[test]
    fn redistribution_forbidden_is_never_inferred_from_a_string() {
        for s in [
            "voicevox",
            "csj",
            "jsut",
            "redistribution-forbidden",
            "do-not-redistribute",
            "proprietary",
        ] {
            assert_ne!(
                LicenseClass::from_license_str(s),
                LicenseClass::RedistributionForbidden,
                "{s} must not be inferred as redistribution-forbidden"
            );
        }
        // It round-trips through the canonical class name, which is how an
        // explicit list sets it.
        assert_eq!(
            LicenseClass::from_class_str("redistribution-forbidden"),
            Some(LicenseClass::RedistributionForbidden)
        );
    }

    /// Every class must round-trip through its canonical wire name, or a
    /// stamped GGUF would read back as something else.
    #[test]
    fn every_class_round_trips_through_its_wire_name() {
        use LicenseClass::*;
        for c in [
            Permissive,
            AttributionRequired,
            Copyleft,
            NonCommercial,
            NonCommercialShareAlike,
            RedistributionForbidden,
            ConditionalCommercial,
            InheritedRestriction,
            Unknown,
        ] {
            assert_eq!(
                LicenseClass::from_class_str(c.as_str()),
                Some(c),
                "{c:?} must round-trip via {:?}",
                c.as_str()
            );
        }
    }

    /// OpenRAIL-family licences carry usage restrictions that flow downstream
    /// but do not restrict commercial use or require a research flag to load.
    /// Modelled distinctly from [`LicenseClass::Copyleft`] because they share
    /// the same "preserve licence when republishing" verdict for different
    /// reasons (use-case list vs derivative-licence terms).
    #[test]
    fn openrail_is_inherited_restriction_not_copyleft() {
        for s in [
            "openrail",
            "OpenRAIL-M",
            "creativeml-openrail-m",
            "CreativeML OpenRAIL-M",
            "bigscience-openrail-m",
            "RAIL-M",
        ] {
            let c = LicenseClass::from_license_str(s);
            assert_eq!(
                c,
                LicenseClass::InheritedRestriction,
                "{s} must classify as inherited-restriction"
            );
            assert!(!c.requires_research_flag(), "{s}: loading is unrestricted");
            assert!(c.commercial_ok(), "{s}: commercial use is permitted");
            assert!(c.redistributable(), "{s}: republishable");
            assert!(
                c.requires_license_preserved(),
                "{s}: use-case list must travel with the artefact"
            );
            assert!(c.requires_attribution(), "{s}: attribution required");
        }
        // Canonical class name round-trips.
        assert_eq!(
            LicenseClass::from_class_str("inherited-restriction"),
            Some(LicenseClass::InheritedRestriction)
        );
        // Short aliases (`openrail`, `rail`) also parse.
        assert_eq!(
            LicenseClass::from_class_str("openrail"),
            Some(LicenseClass::InheritedRestriction)
        );
        assert_eq!(
            LicenseClass::from_class_str("rail"),
            Some(LicenseClass::InheritedRestriction)
        );
    }

    #[test]
    fn gate_membership_matches_severity() {
        assert!(!LicenseClass::Permissive.requires_research_flag());
        assert!(!LicenseClass::AttributionRequired.requires_research_flag());
        assert!(LicenseClass::NonCommercial.requires_research_flag());
        assert!(LicenseClass::NonCommercialShareAlike.requires_research_flag());
        assert!(LicenseClass::Unknown.requires_research_flag());
    }

    #[test]
    fn commercial_and_attribution_flags() {
        assert!(LicenseClass::Permissive.commercial_ok());
        assert!(LicenseClass::AttributionRequired.commercial_ok());
        assert!(!LicenseClass::NonCommercial.commercial_ok());
        assert!(!LicenseClass::Unknown.commercial_ok());
        // InheritedRestriction is loadable + commercial-OK (OpenRAIL family
        // constrains specific use cases, not commercial use itself) and
        // carries the same attribution obligation as the CC-BY family.
        assert!(LicenseClass::InheritedRestriction.commercial_ok());
        assert!(LicenseClass::InheritedRestriction.requires_attribution());
        assert!(LicenseClass::AttributionRequired.requires_attribution());
        assert!(!LicenseClass::Permissive.requires_attribution());
    }

    #[test]
    fn canonical_name_roundtrips() {
        for c in [
            LicenseClass::Permissive,
            LicenseClass::AttributionRequired,
            LicenseClass::NonCommercial,
            LicenseClass::NonCommercialShareAlike,
            LicenseClass::Unknown,
        ] {
            assert_eq!(LicenseClass::from_class_str(c.as_str()), Some(c));
        }
        assert_eq!(LicenseClass::from_class_str("garbage"), None);
    }

    #[test]
    fn license_string_classification_covers_audit_rows() {
        // The three CC-BY-NC rows from docs/license-audit.md §3, in a few
        // spellings each (case / separator variants), must all gate.
        for s in ["CC-BY-NC-4.0", "cc by nc 4.0", "CC_BY_NC_4.0"] {
            assert_eq!(
                LicenseClass::from_license_str(s),
                LicenseClass::NonCommercial,
                "{s}"
            );
        }
        for s in ["CC-BY-NC-SA-4.0", "cc-by-nc-sa 4.0"] {
            assert_eq!(
                LicenseClass::from_license_str(s),
                LicenseClass::NonCommercialShareAlike,
                "{s}"
            );
        }
        // `NVIDIA Source Code License-NC` (a licence family, not a specific
        // model): the string parses to NonCommercial regardless of which
        // model presently carries it. BigVGAN itself moved to MIT in 2024
        // (see `docs/license-audit.md` §3), so this assertion pins the
        // parser's behaviour on the licence text, NOT the current status
        // of any model that historically shipped under it.
        assert_eq!(
            LicenseClass::from_license_str("NVIDIA Source Code License-NC"),
            LicenseClass::NonCommercial
        );
        // Responsible-AI Licence family (OpenRAIL-M) — audit rows for
        // downstream OpenRAIL-tagged models parse to InheritedRestriction,
        // distinct from Copyleft even though both preserve the licence on
        // republishing.
        for s in [
            "openrail-m",
            "creativeml-openrail-m",
            "BigScience-OpenRAIL-M",
        ] {
            assert_eq!(
                LicenseClass::from_license_str(s),
                LicenseClass::InheritedRestriction,
                "{s}"
            );
        }
        // Attribution (CC-BY without NC) is NOT gated.
        assert_eq!(
            LicenseClass::from_license_str("CC-BY-4.0"),
            LicenseClass::AttributionRequired
        );
        // Permissive families.
        for s in ["MIT", "Apache-2.0", "apache 2.0", "BSD-3-Clause", "CC0-1.0"] {
            assert_eq!(
                LicenseClass::from_license_str(s),
                LicenseClass::Permissive,
                "{s}"
            );
        }
        // Fail-closed on empty / unknown.
        assert_eq!(LicenseClass::from_license_str(""), LicenseClass::Unknown);
        assert_eq!(
            LicenseClass::from_license_str("要確認"),
            LicenseClass::Unknown
        );
        assert_eq!(
            LicenseClass::from_license_str("proprietary"),
            LicenseClass::Unknown
        );
    }

    #[test]
    fn registry_maps_first_party_permissive_and_ccbync_gated() {
        // First-party runtime models load on the default path.
        for id in [
            "whisper",
            "piper-plus-mb-istft-vits2",
            "silero-vad",
            "campplus",
            // SoTA plan Phase 1-4 (2026-07-24) — nari-labs Dia canonical id
            // and the variant HF publishes.
            "dia",
            "dia-1.6b",
            // SoTA plan Phase 1-5 (2026-07-24) — Zyphra Zonos canonical id
            // and the short form the CLI accepts.
            "zonos",
            "zonos-v0.1",
        ] {
            assert_eq!(registry_lookup(id), Some(LicenseClass::Permissive), "{id}");
        }
        // docs/license-audit.md §3 CC-BY-NC / CC-BY-NC-SA weights are gated.
        assert_eq!(registry_lookup("f5-tts"), Some(LicenseClass::NonCommercial));
        assert_eq!(
            registry_lookup("encodec"),
            Some(LicenseClass::NonCommercial)
        );
        assert_eq!(
            registry_lookup("fish-speech-v1.5"),
            Some(LicenseClass::NonCommercialShareAlike)
        );
        // Attribution codec.
        assert_eq!(
            registry_lookup("mimi"),
            Some(LicenseClass::AttributionRequired)
        );
        // SoTA plan Phase 2 (2026-07-24): Kyutai STT-2.6B-EN. Canonical id
        // and the two variant spellings the CLI + converter accept — all
        // resolve to CC-BY 4.0 attribution-required (never permissive).
        for id in [
            "kyutai-stt",
            "kyutai-stt-2.6b-en",
            "kyutai-stt-2.6b",
            "stt-2.6b-en",
            // Case-insensitive.
            "Kyutai-STT-2.6B-EN",
            // Family prefix — a hypothetical future `kyutai-stt-1b-en`
            // variant still resolves attribution-required by the walk.
            "kyutai-stt-1b-en",
        ] {
            assert_eq!(
                registry_lookup(id),
                Some(LicenseClass::AttributionRequired),
                "{id}"
            );
        }
        // Guard: a random id containing "stt" that is NOT under the
        // Kyutai family prefix still fails closed to `None`.
        assert_eq!(registry_lookup("some-random-stt-model"), None);
        // SoTA plan Phase 2 (2026-07-24): NVIDIA Parakeet-TDT-0.6B-v3.
        // Canonical id + the variant spellings the CLI + converter
        // accept — all resolve to CC-BY 4.0 attribution-required
        // (NVIDIA's model card explicitly grants CC-BY-4.0).
        for id in [
            "parakeet-tdt",
            "parakeet-tdt-0.6b-v3",
            "parakeet-tdt-0.6b",
            "parakeet",
            // Case-insensitive.
            "Parakeet-TDT-0.6B-v3",
            // Family prefix — a hypothetical future
            // `parakeet-tdt-1.1b` / `parakeet-rnnt-1.1b` still
            // resolves attribution-required by the walk.
            "parakeet-tdt-1.1b",
            "parakeet-rnnt-1.1b",
        ] {
            assert_eq!(
                registry_lookup(id),
                Some(LicenseClass::AttributionRequired),
                "{id}"
            );
        }
        // Guard: a random id starting with "parakeetx" is NOT under
        // the family prefix (the dash guard rejects it).
        assert_eq!(registry_lookup("parakeetx-something"), None);
        // SoTA plan Phase 2 (2026-07-24): NVIDIA Parakeet-CTC-1.1B.
        // Canonical id + variant spellings — all resolve to CC-BY 4.0
        // attribution-required (NVIDIA's model card explicitly grants
        // CC-BY-4.0 for the CTC family too).
        for id in [
            "parakeet-ctc",
            "parakeet-ctc-1.1b",
            // Case-insensitive (via lower-casing before lookup).
            "Parakeet-CTC-1.1B",
            "PARAKEET-CTC",
            // Family prefix — a hypothetical future
            // `parakeet-ctc-0.6b` / `parakeet-ctc-6b` still resolves
            // attribution-required by the walk.
            "parakeet-ctc-0.6b",
            "parakeet-ctc-6b",
        ] {
            assert_eq!(
                registry_lookup(id),
                Some(LicenseClass::AttributionRequired),
                "{id}"
            );
        }
        // SoTA plan Phase 2 (2026-07-24): NVIDIA Canary-1B-v2. Canonical
        // id + variant spellings — all resolve to CC-BY 4.0
        // attribution-required (NVIDIA's model card explicitly states
        // CC-BY-4.0 for the whole Canary family).
        for id in [
            "canary",
            "canary-1b-v2",
            // Case-insensitive (via lower-casing before lookup).
            "Canary-1B-v2",
            "CANARY",
            // Family prefix — a hypothetical future
            // `canary-1b-v2-en` / `canary-3b-v3` / `canary-180m-flash`
            // still resolves attribution-required by the walk.
            "canary-1b-v2-en",
            "canary-3b-v3",
            "canary-180m-flash",
        ] {
            assert_eq!(
                registry_lookup(id),
                Some(LicenseClass::AttributionRequired),
                "{id}"
            );
        }
        // Guard: a random id starting with "canaryx" is NOT under the
        // family prefix (the dash guard rejects it).
        assert_eq!(registry_lookup("canaryx-something"), None);
        // SoTA plan Phase 2 (2026-07-24): Meta omniASR-CTC-1B. Canonical
        // id + variant spellings — all resolve to **Permissive**
        // (Apache-2.0), NOT AttributionRequired like the NVIDIA CTC /
        // AED families above. The whole omniASR-CTC family (paired
        // 300M / 1B / 3B / 7B checkpoints) ships Apache-2.0 per the
        // fairseq2 release.
        for id in [
            "omniasr-ctc",
            "omniasr-ctc-1b",
            // Case-insensitive (via lower-casing before lookup).
            "OmniASR-CTC-1B",
            "OMNIASR-CTC",
            // Family prefix — the paired 300M / 3B / 7B checkpoints (and
            // a hypothetical future size) still resolve permissive by
            // the walk. The dash guard keeps unrelated ids out.
            "omniasr-ctc-300m",
            "omniasr-ctc-3b",
            "omniasr-ctc-7b",
        ] {
            assert_eq!(registry_lookup(id), Some(LicenseClass::Permissive), "{id}");
        }
        // Guard: a random id starting with "omniasr-ctcxyz" (no dash) is
        // NOT under the family prefix walk.
        assert_eq!(registry_lookup("omniasr-ctcxyz-something"), None);
        // Case-insensitive.
        assert_eq!(registry_lookup("F5-TTS"), Some(LicenseClass::NonCommercial));
        // First-party **variant** ids (not canonical) still resolve permissive by
        // family prefix — a stock voice's untagged GGUF must not fail-closed.
        for id in [
            "piper-plus-multilingual-6lang", // the v7 zero-shot voice id
            "whisper-base.en",
            "silero-vad-v5",
            "kokoro-82m",
            // CosyVoice2 first-party family (M3-09 scaffold): Apache 2.0 code
            // + weight, so a variant id like `cosyvoice2-0.5b` still resolves
            // permissive (docs/license-audit.md).
            "cosyvoice2-0.5b",
            // Fun-CosyVoice3 first-party family (SoTA plan Phase 3): apache-2.0
            // weight, so canonical + variant ids all resolve permissive via
            // the `fun-cosyvoice3-` / `cosyvoice3-` prefix walks.
            "fun-cosyvoice3-0.5b",
            "fun-cosyvoice3-0.5b-2512",
            "cosyvoice3-0.5b",
            // Dia family (SoTA Phase 1-4): a future `dia-3b` still resolves
            // permissive without being individually listed.
            "dia-3b",
            // Zonos family (SoTA Phase 1-5): both HF variants and a future
            // `zonos-v0.2` still resolve permissive via the family prefix.
            "zonos-v0.1-transformer",
            "zonos-v0.1-hybrid",
            "zonos-v0.2",
        ] {
            assert_eq!(registry_lookup(id), Some(LicenseClass::Permissive), "{id}");
        }
        // But an unlisted variant of a GATED family still fails closed (not
        // permissive): the family prefixes cover first-party archs only.
        assert_eq!(registry_lookup("encodec-24khz-v2"), None);
        assert_eq!(registry_lookup("fish-speech-v9"), None);
        // Unregistered -> None (caller fails closed to Unknown).
        assert_eq!(registry_lookup("totally-unknown-model"), None);
    }

    /// SoTA plan Phase 2-5 (2026-07-24): every new family added to
    /// `registry_lookup` in this campaign must (a) resolve to the exact
    /// license class the module doc quotes from its primary source, (b) keep
    /// working when the id is case-perturbed (registry does an
    /// ASCII-lower-case up front), and (c) accept the family-prefix walk for
    /// a hypothetical future variant so an untagged variant GGUF loads on
    /// the correct gate.
    ///
    /// This test pins every new arm the campaign added:
    ///
    /// - `distil-whisper` / `distil-large` / `distil-medium` / `distil-small`
    ///   family (MIT, Permissive) — Phase 2.
    /// - `kotoba-whisper` family (Apache-2.0, Permissive) — Phase 5 JA-ASR-2.
    /// - `chatterbox` / `chatterbox-turbo` / `chatterbox-nano` +
    ///   `chatterbox_` alias family (MIT, Permissive) — Phase 3.
    /// - `qwen3-tts` family (Apache-2.0, Permissive) — Phase 3.
    /// - `voxcpm` / `voxcpm2` family (Apache-2.0, Permissive) — Phase 4.
    /// - `vibevoice` family (MIT, Permissive) — Phase 4.
    /// - `irodori` / `irodori-tts` family (MIT, Permissive) — Phase 5
    ///   JA-TTS-1.
    /// - `vits-ja` / `vits_ja` / ESPnet-JSUT-VITS / ESPnet-JVS-VITS /
    ///   COEIROINK-VITS (**RedistributionForbidden**) — Phase 5 JA-TTS-2.
    ///   This is the ONE arm that is *not* Permissive and where a wrong
    ///   verdict would silently authorise republishing a corpus that
    ///   explicitly bans it (JSUT / JVS terms).
    /// - `dac` / `wavtokenizer` (Permissive).
    /// - `x-codec-2` / `xcodec2` (**NonCommercial** — HF card at
    ///   `huggingface.co/HKUSTAudio/xcodec2` = `cc-by-nc-4.0`,
    ///   `docs/license-audit.md` §3.1 2026-07-23 yousan sign-off =
    ///   ☑ Research-only; flipped 2026-07-28 from an earlier Permissive
    ///   listing that mistakenly read the MIT code license as governing
    ///   weight redistribution).
    #[test]
    fn sota_plan_registry_entries_resolve_to_the_correct_class() {
        // ---- Phase 2: distil-whisper (MIT) -----------------------------
        for id in [
            "distil-whisper",
            "distil-whisper-large-v3",
            "distil-whisper-large-v3.5",
            "distil-whisper-large-v3_5",
            "distil-large-v3",
            "distil-large-v3.5",
            "distil-large-v3_5",
            // Case-insensitivity is provided by the top-of-fn lower-case.
            "Distil-Whisper-Large-v3.5",
            "DISTIL-LARGE-V3",
            // Family-prefix walk covers a hypothetical future variant.
            "distil-whisper-small.en",
            "distil-medium-en",
            "distil-small-multilingual",
        ] {
            assert_eq!(
                registry_lookup(id),
                Some(LicenseClass::Permissive),
                "distil-whisper family: {id}"
            );
        }

        // ---- Phase 5 JA-ASR-2: kotoba-whisper (Apache-2.0) --------------
        for id in [
            "kotoba-whisper",
            "kotoba-whisper-v1.0",
            "kotoba-whisper-v1_0",
            "kotoba-whisper-v1.1",
            "kotoba-whisper-v2.0",
            "kotoba-whisper-v2_1",
            "kotoba-whisper-bilingual",
            "kotoba-whisper-bilingual-v1.0",
            "kotoba-whisper-bilingual-v1_0",
            // Case-insensitive.
            "Kotoba-Whisper-v2.0",
            "KOTOBA-WHISPER-V1_1",
            // Family prefix — a hypothetical `v3.0` still resolves.
            "kotoba-whisper-v3.0",
        ] {
            assert_eq!(
                registry_lookup(id),
                Some(LicenseClass::Permissive),
                "kotoba-whisper family: {id}"
            );
        }

        // ---- Phase 3: chatterbox (base + multilingual + turbo + nano) (MIT)
        for id in [
            // base + multilingual arm
            "chatterbox",
            "chatterbox-multilingual",
            "chatterbox-multilingual-v2",
            "chatterbox-multilingual-v3",
            "chatterbox-mtl23ls-v2",
            "chatterbox-mtl23ls-v3",
            "chatterbox-english",
            "chatterbox_en",
            // turbo arm
            "chatterbox-turbo",
            "chatterbox_turbo",
            "chatterbox-turbo-v1",
            "chatterbox-turbo-onnx",
            // nano arm
            "chatterbox-nano",
            "chatterbox_nano",
            "chatterbox-nano-v1",
            // Case-insensitive.
            "Chatterbox-Turbo",
            "CHATTERBOX_NANO",
            // Family prefix — a hypothetical future variant.
            "chatterbox-japanese",
            "chatterbox-huge-v4",
            "chatterbox_multi",
        ] {
            assert_eq!(
                registry_lookup(id),
                Some(LicenseClass::Permissive),
                "chatterbox family: {id}"
            );
        }

        // ---- Phase 3: qwen3-tts (Apache-2.0) ----------------------------
        for id in [
            "qwen3-tts",
            "qwen3_tts",
            "qwen3-tts-0.6b",
            "qwen3-tts-0_6b",
            "qwen3-tts-12hz-0.6b-base",
            "qwen3-tts-12hz-0_6b-base",
            "qwen3-tts-12hz-0.6b",
            // Case-insensitive.
            "Qwen3-TTS-12Hz-0.6B-Base",
            // Family prefix — a hypothetical future variant.
            "qwen3-tts-24hz-1.7b-base",
            "qwen3-tts-12hz-0.6b-customvoice",
        ] {
            assert_eq!(
                registry_lookup(id),
                Some(LicenseClass::Permissive),
                "qwen3-tts family: {id}"
            );
        }

        // ---- Phase 4: voxcpm / voxcpm2 (Apache-2.0) ---------------------
        for id in [
            "voxcpm",
            "voxcpm2",
            "voxcpm-0.5b",
            "voxcpm-0_5b",
            "voxcpm-0.5b-base",
            "voxcpm-0_5b-base",
            // Case-insensitive.
            "VoxCPM-0.5B",
            // Family prefix — a hypothetical future variant.
            "voxcpm-1.5b",
        ] {
            assert_eq!(
                registry_lookup(id),
                Some(LicenseClass::Permissive),
                "voxcpm family: {id}"
            );
        }

        // ---- Phase 4: vibevoice (MIT) -----------------------------------
        for id in [
            "vibevoice",
            "vibevoice-1.5b",
            "vibevoice-1_5b",
            "vibevoice-1.5b-base",
            "vibevoice-1_5b-base",
            // Case-insensitive.
            "VibeVoice-1.5B",
            "VIBEVOICE",
            // Family prefix — a hypothetical future variant.
            "vibevoice-7b",
            "vibevoice-large",
        ] {
            assert_eq!(
                registry_lookup(id),
                Some(LicenseClass::Permissive),
                "vibevoice family: {id}"
            );
        }

        // ---- Phase 5 JA-TTS-1: irodori (MIT) ----------------------------
        for id in [
            "irodori",
            "irodori-tts",
            "irodori_tts",
            "irodori-tts-500m",
            "irodori-tts-500m-v2",
            "irodori-tts-500m-v2-voicedesign",
            "irodori-tts-500m-v3",
            "irodori-tts-500m-v3-base",
            "irodori-tts-600m-v3-voicedesign",
            // Case-insensitive.
            "Irodori-TTS-500M-v3",
            // Family prefix — a hypothetical future variant.
            "irodori-tts-2.5b-v4",
            // `irodori-` prefix walk also matches (a non-`-tts-` spelling).
            "irodori-japanese-v1",
        ] {
            assert_eq!(
                registry_lookup(id),
                Some(LicenseClass::Permissive),
                "irodori family: {id}"
            );
        }

        // ---- FR-OP-32 codecs (Permissive) --------------------------------
        //
        // DAC + WavTokenizer stay Permissive (MIT weight). X-Codec 2 was
        // previously in this list — flipped 2026-07-28 to NonCommercial
        // after CC-verification of `huggingface.co/HKUSTAudio/xcodec2`
        // (front-matter `license: cc-by-nc-4.0`); see the dedicated
        // xcodec2 arm below.
        for id in ["dac", "wavtokenizer"] {
            assert_eq!(
                registry_lookup(id),
                Some(LicenseClass::Permissive),
                "codec: {id}"
            );
        }

        // ---- SoTA plan Phase 5 codec: X-Codec 2 (NonCommercial) ---------
        //
        // The **weight** class flip that motivated the 2026-07-28 change.
        // The HF card at `huggingface.co/HKUSTAudio/xcodec2` carries
        // `license: cc-by-nc-4.0` on its YAML front-matter (CC-verified
        // 2026-07-15; `docs/license-audit.md` §3.1 sign-off 2026-07-23
        // yousan = ☑ Research-only). The code layer at
        // `github.com/zhenye234/X-Codec-2.0` is MIT — but M2-13 gates on
        // the **weight** class, and the weight-distribution repo governs
        // the class of the redistributed artifact. Fail-closed: a
        // commercial-mode caller cannot silently bring this up
        // (`requires_research_flag = true`), the publish gate refuses
        // (`redistributable = false`, `commercial_ok = false`).
        for id in [
            "x-codec-2",
            "xcodec2",
            // Case-insensitive.
            "X-Codec-2",
            "XCODEC2",
        ] {
            let c = registry_lookup(id);
            assert_eq!(
                c,
                Some(LicenseClass::NonCommercial),
                "xcodec2: {id} MUST be NonCommercial (HF card = cc-by-nc-4.0) \
                 — silently returning Permissive would authorise a commercial \
                 load of an NC weight."
            );
            let c = c.unwrap();
            assert!(
                c.requires_research_flag(),
                "{id}: NC must require the research flag to load"
            );
            assert!(
                !c.commercial_ok(),
                "{id}: commercial_ok must be false for NC"
            );
            assert!(
                !c.redistributable(),
                "{id}: NonCommercial is not on the publish gate's allow-list"
            );
        }

        // ---- Phase 5 JA-TTS-2: vits-ja (RedistributionForbidden) --------
        //
        // The one arm that is NOT Permissive. Every id here MUST resolve to
        // `RedistributionForbidden` because the trained weights carry
        // corpus-level redistribution bans (JSUT / JVS / COEIROINK). A
        // wrong verdict here would silently authorise publishing a corpus
        // that explicitly forbids it — the exact class of drift the audit
        // is designed to catch.
        for id in [
            "vits-ja",
            "vits_ja",
            "espnet-vits-ja",
            "espnet-jsut-vits",
            "espnet-jvs-vits",
            "coeiroink-vits",
            // Case-insensitive.
            "VITS-JA",
            "ESPnet-JSUT-VITS",
        ] {
            let c = registry_lookup(id);
            assert_eq!(
                c,
                Some(LicenseClass::RedistributionForbidden),
                "vits-ja: {id} MUST be RedistributionForbidden (JSUT / JVS \
                 corpus bans re-distribution) — silently returning Permissive \
                 would authorise a forbidden publish."
            );
            // Cross-check the derived predicates — the class must fail the
            // publish gate but stay loadable (owner may still hold the
            // weights locally for their own inference).
            let c = c.unwrap();
            assert!(
                !c.redistributable(),
                "{id}: publish gate must fail on vits-ja"
            );
            assert!(
                !c.requires_research_flag(),
                "{id}: loading is unrestricted (only republish is barred)"
            );
        }
    }

    /// SoTA plan Phase 2 (2026-07-24): the new CC-BY-4.0 ASR families
    /// (Kyutai STT, NVIDIA Parakeet CTC/TDT + Canary AED) must all resolve
    /// to `AttributionRequired`, keep the M2-13 gate green (commercial
    /// allowed) *and* activate the FR-MD-09 attribution surface. Meta's
    /// omniASR-CTC ships Apache-2.0 by contrast, so it must land under
    /// `Permissive` — proving the arms are family-specific and not a
    /// blanket "all ASR is attribution-required" default.
    #[test]
    fn sota_plan_phase2_asr_families_resolve_correctly() {
        // Kyutai STT (CC-BY-4.0) - explicit ids already covered above;
        // pin the derived predicates so a caller can rely on the class
        // semantics staying stable.
        for id in [
            "kyutai-stt",
            "kyutai-stt-2.6b-en",
            "kyutai-stt-2.6b",
            "stt-2.6b-en",
            "kyutai-stt-1b-en",
        ] {
            let c = registry_lookup(id).unwrap_or_else(|| panic!("{id} not registered"));
            assert_eq!(c, LicenseClass::AttributionRequired, "{id}");
            assert!(c.commercial_ok(), "{id}: CC-BY 4.0 commercial-ok");
            assert!(c.requires_attribution(), "{id}: attribution required");
            assert!(c.redistributable(), "{id}: republishable with credit");
            assert!(!c.requires_research_flag(), "{id}: loadable");
        }
        // NVIDIA Parakeet family (CC-BY-4.0).
        for id in [
            "parakeet-tdt",
            "parakeet-ctc-1.1b",
            "parakeet-rnnt-1.1b",
            "canary-1b-v2",
            "canary-3b-v3",
        ] {
            let c = registry_lookup(id).unwrap_or_else(|| panic!("{id} not registered"));
            assert_eq!(c, LicenseClass::AttributionRequired, "{id}");
            assert!(c.commercial_ok(), "{id}");
            assert!(c.requires_attribution(), "{id}");
        }
        // Meta omniASR-CTC ships Apache-2.0 — MUST be Permissive, not
        // AttributionRequired. Regression guard against a blanket
        // "all *-ctc- is CC-BY" default (NVIDIA vs Meta divergence).
        for id in [
            "omniasr-ctc",
            "omniasr-ctc-1b",
            "omniasr-ctc-300m",
            "omniasr-ctc-3b",
            "omniasr-ctc-7b",
        ] {
            assert_eq!(
                registry_lookup(id),
                Some(LicenseClass::Permissive),
                "{id}: Meta omniASR-CTC ships Apache-2.0 — must NOT be \
                 attribution-required like NVIDIA's Parakeet-CTC / Canary."
            );
        }
    }
}
