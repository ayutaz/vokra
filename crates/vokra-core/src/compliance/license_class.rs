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
        const PERMISSIVE_TOKENS: [&str; 9] = [
            "mit",
            "apache",
            "bsd",
            "cc0",
            "isc",
            "unlicense",
            "mpl",
            "zlib",
            // OpenMDW-1.1 (Open Model Derivatives Work 1.1, openmdw.ai/license/1-1/):
            // permissive MIT-analog for ML weights — commercial+redistribution
            // allowed, no share-alike / non-commercial / field-of-use
            // restrictions. NVIDIA Nemotron family uses this (2026-07-30 CC
            // primary source照合、`docs/license-audit.md` §3.1 row 更新済).
            "openmdw",
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
        // pyannote speaker diarization / VAD segmentation (2026-07-30 license
        // half unblock): weight license = **MIT** per HF `pyannote/segmentation-
        // 3.0` + `pyannote/speaker-diarization-3.1` cardData `license: mit`
        // (authenticated HF API primary source, `gated: auto` は access control
        // のみで追加条項なし、`docs/license-audit.md` §3.1 row 263 で 2026-07-30
        // yousan sign)。The `diarize` op (FR-OP-82) itself is still M5-residual
        // (op 実装 + trigger converter + real-checkpoint parity は残 wave の
        // scope)、but the license side is unblocked.
        "pyannote"
        | "pyannote-segmentation"
        | "pyannote-segmentation-3.0"
        | "pyannote-segmentation-3_0"
        | "pyannote-speaker-diarization"
        | "pyannote-speaker-diarization-3.1"
        | "pyannote-speaker-diarization-3_1" => LicenseClass::Permissive,
        "kokoro" | "kokoro-82m" | "cosyvoice" | "cosyvoice2" | "sesame-csm" | "csm-1b"
        | "voxtral" | "openwakeword" => LicenseClass::Permissive,
        // 2026-08-02 Wave residual: Moonshine-Tiny (UsefulSensors, MIT).
        // 27M raw-audio transformer enc-dec ASR (arXiv:2410.15608). Weight
        // license = **MIT** per upstream `UsefulSensors/moonshine-tiny`
        // model card (Apache-2.0 code + MIT weight release — the model
        // card canonical spelling). Sibling to the Whisper / piper-plus /
        // Silero first-party Permissive posture.
        "moonshine" | "moonshine-tiny" => LicenseClass::Permissive,
        // 2026-08-02 Wave residual: Moonshine-Base (UsefulSensors, MIT).
        // 61.5M raw-audio transformer enc-dec ASR (arXiv:2410.15608).
        // Sibling to Moonshine-Tiny — same arch family (raw-audio
        // Conv1D + rotary + SwiGLU), wider/deeper backbone. Weight
        // license = **MIT** per upstream `UsefulSensors/moonshine-base`
        // model card (same posture as Tiny sibling). Sibling to the
        // Whisper / piper-plus / Silero first-party Permissive posture.
        "moonshine-base" => LicenseClass::Permissive,
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
        | "qwen3-tts-12hz-0.6b"
        // 2026-08-01 Wave 4 slug-only add: 0.6B-CustomVoice fine-tune of
        // 0.6B-Base — same apache-2.0 grant end-to-end (`qwen3-tts-` prefix
        // family walk below would resolve these to `Permissive` too; the
        // exact-match arm is faster + keeps the canonical spellings visible
        // in one place, mirror of the sibling variant walks throughout this
        // registry).
        | "qwen3-tts-0.6b-customvoice"
        | "qwen3-tts-0_6b-customvoice"
        | "qwen3-tts-12hz-0.6b-customvoice"
        | "qwen3-tts-12hz-0_6b-customvoice" => LicenseClass::Permissive,
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
        | "voxcpm-0_5b-base"
        // 2026-07-30 Option C hybrid rename: `voxcpm2-*` names carry the
        // arch-family prefix so the parity harness dispatches on a single
        // string. Both variants ship apache-2.0 end-to-end so the class
        // is unchanged — the only novelty is the canonical `voxcpm2-2b`
        // name for `openbmb/VoxCPM2` (2B scale-up). The legacy
        // `voxcpm-0.5b` string above stays live for backward compat with
        // any pre-rename GGUF on disk.
        | "voxcpm2-0.5b" | "voxcpm2-0_5b" | "voxcpm2-2b" | "voxcpm2-2_0b"
        | "voxcpm2-2b-base" => LicenseClass::Permissive,
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
        "dac" | "dac-24khz" | "dac-16khz" | "dac-44khz" | "wavtokenizer" => {
            LicenseClass::Permissive
        }
        // 2026-08-01 Wave 8: SpeechBrain ECAPA-TDNN, voice-gender-classifier,
        // primeline whisper-de fine-tune, jonatasgrosman xlsr-53-arabic.
        // All apache-2.0 or MIT permissive (primary source verified via HF
        // cardData API 2026-08-01).
        "voice-gender-classifier"
        | "speechbrain-spkrec-ecapa-voxceleb"
        | "whisper-large-v3-turbo-german"
        | "wav2vec2-large-xlsr-53-arabic" => LicenseClass::Permissive,
        // 2026-08-01 Wave 8: pyannote/wespeaker CC-BY-4.0 attribution.
        "pyannote-wespeaker-voxceleb-resnet34-lm" => LicenseClass::AttributionRequired,
        // M5 gap follow-up (2026-07-30): marl/crepe — a monophonic F0
        // (fundamental-frequency) extractor. Weight license = **MIT**
        // (`marl/crepe/main/LICENSE.txt`, "MIT License / Copyright (c)
        // 2018 Jong Wook Kim et al.", CC-verified 2026-07-30 —
        // CLAUDE.md「ハルシネーション厳禁」). Every capacity size
        // (tiny/small/medium/large/full) shares the same MIT LICENSE.
        // Registered here so a converted GGUF with
        // `vokra.provenance.model_id = "crepe"` load-gates as commercial
        // without a caller-side override.
        "crepe" | "crepe-tiny" | "crepe-small" | "crepe-medium" | "crepe-large" | "crepe-full" => {
            LicenseClass::Permissive
        }
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
        // hf-audio-gap-comprehensive-2026-07-30 §3.8 JA-vocoder complement
        // wave (2026-08-04): Aratako/MioCodec-25Hz-44.1kHz-v2 — MIT
        // (Permissive) end-to-end weight per HF cardData primary source
        // 2026-08-04 (`api/models/Aratako/MioCodec-25Hz-44.1kHz-v2` →
        // `license: mit`). The arch tag (underscore == `vokra.model.arch`),
        // the CLI slug (hyphen), and the versioned publish repo slug
        // (`miocodec-25hz-44khz-v2` — matches
        // `huggingface.co/vokra/miocodec-25hz-44khz-v2`) are all
        // registered so an untagged GGUF resolves permissive on the
        // fallback path regardless of which spelling it carries.
        "miocodec"
        | "mio-codec"
        | "mio_codec"
        | "miocodec-25hz-44khz-v2"
        | "miocodec_25hz_44khz_v2"
        | "aratako/miocodec-25hz-44.1khz-v2" => LicenseClass::Permissive,
        // SoTA plan candidate wave (2026-08-04): Neuphonic NeuTTS Air —
        // apache-2.0 (Permissive) end-to-end weight per HF cardData
        // primary source 2026-08-04 (`api/models/neuphonic/neutts-air`
        // → `license: apache-2.0`). The arch tag (== `vokra.model.arch`
        // = "neutts-air"), the CLI slug / publish-repo id (same string
        // for this SKU — `vokra/neutts-air`), the underscore variant
        // (== `models::neutts_air` module filename), and the upstream
        // HF slug are all registered so an untagged GGUF resolves
        // permissive on the fallback path regardless of which spelling
        // it carries. Sibling to `neucodec` (same Neuphonic publisher,
        // same apache-2.0 tag) — kept as its own arm so a rename or
        // classification change on either side stays independent.
        "neutts-air"
        | "neutts_air"
        | "neu-tts-air"
        | "neu_tts_air"
        | "neuphonic/neutts-air" => LicenseClass::Permissive,
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
        // 2026-08-01 wave: IBM Granite Speech 4.1-2B — audio-LLM ASR
        // (Conformer CTC encoder + Granite-4.0-1b-base LLM decoder +
        // BLIP-2 q-former projector + optional LoRA adapter). Weight
        // license = apache-2.0 end-to-end (HF model page + docs page
        // linking apache.org/licenses/LICENSE-2.0, CC-verified
        // 2026-08-01). Redundant with the `granite-speech-` family walk
        // below, but kept as an explicit exact-match arm so an id
        // lookup returns quickly without hitting the prefix arm.
        "granite-speech"
        | "granite-speech-4.1-2b"
        | "granite_speech"
        | "granite-speech-4_1-2b"
        | "ibm-granite/granite-speech-4.1-2b" => LicenseClass::Permissive,
        // 2026-08-01 Wave 3: OpenMOSS MOSS-Audio-Tokenizer family
        // (`OpenMOSS-Team/MOSS-Audio-Tokenizer` + `-Nano`) — the codec
        // half of the MOSS-TTS pipeline. Weight license = **apache-2.0**
        // end-to-end on both variants (`cardData.license =
        // "apache-2.0"` verified 2026-08-01 via authenticated HF API
        // — no `LICENSE` file in the repos, declared via HF cardData
        // tag only). Redundant with the sibling `moss-` prefix walker
        // below (which handles `moss-tts` + this family via shared
        // OpenMOSS-Team apache-2.0 licensing), but kept as an
        // explicit exact-match arm so an id lookup returns quickly
        // without hitting the prefix arm.
        "moss-audio-tokenizer"
        | "moss_audio_tokenizer"
        | "moss-audio-tokenizer-full"
        | "moss-audio-tokenizer-nano"
        | "moss_audio_tokenizer_nano"
        | "openmoss-team/moss-audio-tokenizer"
        | "openmoss-team/moss-audio-tokenizer-nano" => LicenseClass::Permissive,
        // 2026-08-02 wave: OpenMOSS Team MOSS-Audio-4B-Instruct
        // (`OpenMOSS-Team/MOSS-Audio-4B-Instruct`) — the 4B audio-LLM
        // sibling of the four `moss_tts_*` releases. Custom-code
        // release (`configuration_moss_audio.py`,
        // `trust_remote_code=True`) distinct from every existing
        // `moss-tts` prefix arm below (`moss-audio-4b` does NOT start
        // with `moss-tts`) and from the `moss-audio-tokenizer` codec
        // family above. Weight license = **apache-2.0** end-to-end
        // per parent workflow task manifest (2026-08-02). Registered
        // as an explicit exact-match arm so an id lookup returns
        // quickly and so a hypothetical sibling id like
        // `moss-audio-something-else` cannot silently inherit the
        // classification without explicit review — fail-closed
        // default via the outer `Unknown` arm.
        "moss-audio-4b-instruct"
        | "moss_audio_4b_instruct"
        | "moss-audio-4b"
        | "moss_audio_4b"
        | "openmoss-team/moss-audio-4b-instruct"
        | "openmoss-team/moss-audio-4b" => LicenseClass::Permissive,
        // 2026-08-02 wave: OpenMOSS Team MOSS-Audio-8B-Instruct
        // (`OpenMOSS-Team/MOSS-Audio-8B-Instruct`) — the 8B audio-LLM
        // sibling of `MOSS-Audio-4B-Instruct` sharing the same
        // custom-code release (`configuration_moss_audio.py`,
        // `trust_remote_code=True`, 4 shards ~9.05 GB BF16 — vast.ai
        // required). Weight license = **apache-2.0** end-to-end per
        // parent workflow task manifest (2026-08-02). Registered as
        // an explicit exact-match arm so an id lookup returns
        // quickly and so a hypothetical sibling id like
        // `moss-audio-something-else` cannot silently inherit the
        // classification without explicit review — fail-closed
        // default via the outer `Unknown` arm.
        "moss-audio-8b-instruct"
        | "moss_audio_8b_instruct"
        | "moss-audio-8b"
        | "moss_audio_8b"
        | "openmoss-team/moss-audio-8b-instruct"
        | "openmoss-team/moss-audio-8b" => LicenseClass::Permissive,
        // 2026-08-01 Wave 3: Amphion NaturalSpeech 3 FACodec — factorized
        // VQ (FVQ) codec (`amphion/naturalspeech3_facodec`). Weight
        // license = **apache-2.0** end-to-end (HF cardData API + Amphion
        // GitHub `open-mmlab/Amphion/LICENSE` both apache-2.0, verified
        // 2026-08-01 — CLAUDE.md「ハルシネーション厳禁」). All four
        // variants (v1 / v2 / redecoder-v{1,2}) share the same repo and
        // the same license. Registered here so a converted GGUF with
        // `vokra.provenance.model_id = "facodec"` / `naturalspeech3-facodec`
        // / `naturalspeech3-facodec-v{1,2}` /
        // `naturalspeech3-facodec-redecoder-v{1,2}` load-gates as
        // commercial without a caller-side override.
        "facodec"
        | "naturalspeech3-facodec"
        | "naturalspeech3_facodec"
        | "ns3-facodec"
        | "ns3_facodec"
        | "amphion/naturalspeech3_facodec"
        | "naturalspeech3-facodec-v1"
        | "naturalspeech3-facodec-v2"
        | "naturalspeech3-facodec-redecoder-v1"
        | "naturalspeech3-facodec-redecoder-v2" => LicenseClass::Permissive,
        // 2026-08-01 wave: Charactr AI Vocos family — Fourier-space
        // vocoder (ConvNeXt V2 backbone + iSTFT head, arXiv:2306.00814).
        // Weight license = **MIT** end-to-end (Charactr AI code + trained
        // weights) per HF cardData API on both
        // `charactr/vocos-mel-24khz` and `charactr/vocos-encodec-24khz`
        // (verified 2026-08-01 — CLAUDE.md 「ハルシネーション厳禁」).
        // GitHub `charactr-platform/vocos/LICENSE` is also standard
        // MIT. Redundant with the `vocos` / `charactr/vocos-` prefix
        // arm below, but the exact canonical spellings are listed
        // here so an id lookup returns quickly without hitting the
        // prefix arm.
        "vocos"
        | "vocos-mel-24khz"
        | "vocos_mel_24khz"
        | "vocos-encodec-24khz"
        | "vocos_encodec_24khz" => LicenseClass::Permissive,
        // 2026-08-01 Wave 3 sibling-pair add: YuE bundle
        // (`m-a-p/YuE-upsampler` + `m-a-p/xcodec_mini_infer`). Weight
        // license = **apache-2.0** end-to-end on both variants (HF
        // cardData API `license: apache-2.0` on both repos, verified
        // 2026-08-01 — CLAUDE.md 「ハルシネーション厳禁」). Upstream
        // YuE code at `github.com/multimodal-art-projection/YuE` also
        // ships apache-2.0. Redundant with the `yue-` prefix walker
        // below, but the exact canonical spellings are listed here so
        // an id lookup returns quickly without hitting the prefix arm.
        "yue-upsampler"
        | "yue_upsampler"
        | "map-yue-upsampler"
        | "m-a-p/yue-upsampler"
        | "yue-xcodec-mini"
        | "yue_xcodec_mini"
        | "yue-xcodec-mini-infer"
        | "yue_xcodec_mini_infer"
        | "xcodec-mini"
        | "xcodec_mini"
        | "xcodec-mini-infer"
        | "xcodec_mini_infer"
        | "yue-codec"
        | "m-a-p/xcodec_mini_infer" => LicenseClass::Permissive,
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
        // BS-Roformer / Mel-Band Roformer (Wave 5 music-separation add,
        // 2026-08-01) — the fail-closed default is
        // `RedistributionForbidden` for a different failure mode than
        // vits-ja above. Where vits-ja refuses because the training
        // corpus (JSUT / JVS / COEIROINK) explicitly forbids trained-
        // weight redistribution, BS-Roformer refuses because a converter
        // cannot know which specific SPDX id applies to the caller's
        // checkpoint: the architecture / reference code is MIT
        // (`github.com/lucidrains/BS-RoFormer`, Phil Wang's clean-room
        // implementation of Lu et al. 2023 arXiv:2310.01809), the paper
        // released no reference weights, and every checkpoint in the
        // wild is a downstream retraining under mixed licenses (GPL-3.0
        // for some Ultimate-Vocal-Remover / MDX-Net-community
        // derivatives, CC-BY-NC-4.0 for some MoisesDB / MusDB fine-
        // tunes, no explicit license for the majority — hobbyist
        // releases). The third-party mirror `chenmozhijin/BSRoformer-
        // GGUF` aggregates converted GGUFs across trainers without a
        // uniform license clause; the converter registers the family
        // fail-closed and defers the per-checkpoint override to the
        // caller supplying `--license <spdx>` at conversion time (the
        // same escape hatch vits-ja / Whisper / kokoro use). Aliases
        // cover the arch tag (underscore + hyphen), the family-name
        // spellings (`bs-roformer` / `bsroformer` / `mel-band-roformer` /
        // `melband-roformer`), and the third-party HF mirror slug — same
        // spellings the converter `from_arg` walk in
        // `crates/vokra-convert/src/lib.rs` accepts. The
        // `mel-band-roformer` sibling shares the same arch tag because
        // the band-split module vs mel-filter-bank module is a runtime
        // hparam, not a distinct arch.
        //
        // Publish is blocked at
        // `scripts/publish/signoff_match.py::REPO_TO_SIGNOFF_ROWS` (no
        // entry for `bs-roformer`, unlisted slug fails closed as
        // `UNKNOWN_REPO`) until an owner ADR selects a specific
        // checkpoint + license — the license classifier here is the
        // upstream half of that gate.
        "bs-roformer"
        | "bs_roformer"
        | "bsroformer"
        | "mel-band-roformer"
        | "mel_band_roformer"
        | "melband-roformer"
        | "melband_roformer"
        | "chenmozhijin/bsroformer-gguf" => LicenseClass::RedistributionForbidden,
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
        // residual wave 4 (2026-08-02): CrisperWhisper
        // (`nyrahealth/CrisperWhisper`, cc-by-nc-4.0). Whisper-large-v3
        // fine-tune emphasising verbatim word-level timestamps —
        // architecturally byte-identical to whisper-large-v3, but the
        // trained weights are gated by CC-BY-NC-4.0 (T4 tier /
        // Research-only publish path per the X-Codec-2 (2026-07-28)
        // precedent). `crisper-whisper` covers the arch tag stamped by
        // the converter (distinct from vanilla `whisper`);
        // `crisperwhisper` / `crisper_whisper` cover the model-id stamp
        // spellings; `vokra/crisperwhisper` covers the publish repo slug
        // per the ELVIS-Act / T4 tier gate. The M2-13 runtime gate
        // refuses to load in commercial mode
        // (`requires_research_flag = true`); publish requires
        // `publish-one.sh --allow-noncommercial`.
        "crisper-whisper"
        | "crisperwhisper"
        | "crisper_whisper"
        | "vokra/crisperwhisper" => LicenseClass::NonCommercial,
        // 2026-08-02 wave: Meta MMS-1B-All (`facebook/mms-1b-all`,
        // **cc-by-nc-4.0** per HF cardData primary source verified
        // 2026-08-02 — CLAUDE.md「ハルシネーション厳禁」). Pratap et al.
        // 2023 (arXiv:2305.13516) — 1B wav2vec 2.0 backbone + 1000+
        // per-language CTC adapters. Registered as **explicit exact-
        // match arms** BEFORE the sibling `_ if id.starts_with("wav2vec2")
        // => Permissive` prefix walk below because MMS is a distinct
        // upstream release with a distinct weight-distribution licence
        // (cc-by-nc-4.0), and the fail-closed publish path requires the
        // `NonCommercial` classification to force `publish-one.sh
        // --allow-noncommercial` at publish time + the M2-13 runtime
        // gate refusal in commercial mode. The `mms-1b-all` /
        // `mms_1b_all` / `mms-1b` / `mms_1b` covers the arch-tag +
        // slug spellings the converter stamps; `facebook/mms-1b-all`
        // covers the upstream HF slug; `vokra/mms-1b-all` covers the
        // publish repo slug per the ELVIS-Act / T4 tier gate.
        "mms-1b-all"
        | "mms_1b_all"
        | "mms-1b"
        | "mms_1b"
        | "facebook/mms-1b-all"
        | "vokra/mms-1b-all" => LicenseClass::NonCommercial,
        // Meta AudioCraft MusicGen family (Wave 5 music-generation add,
        // 2026-08-01) — the trained weights ship **cc-by-nc-4.0** on HF
        // (`huggingface.co/facebook/musicgen-medium` model card
        // front-matter `license: cc-by-nc-4.0`), same posture X-Codec 2
        // uses above. The code layer at
        // `github.com/facebookresearch/audiocraft` is MIT, but this
        // registry lookups the weight distribution class (M2-13 runtime
        // gate anchors on `vokra.provenance.weight_license`, not on the
        // code license). Every future MusicGen family variant (small /
        // large / melody / stereo-*) inherits this arm — Meta's weight
        // policy is uniform across the family. `musicgen` covers the
        // bare arch tag; `musicgen-medium` covers the medium model-id
        // stamp; `musicgen-large` covers the large model-id stamp
        // (Wave 5 sibling landed 2026-08-01, 3.3B vs medium 1.5B, same
        // cc-by-nc-4.0 weight license per HF cardData
        // `huggingface.co/facebook/musicgen-large` primary source).
        "musicgen"
        | "musicgen-small"
        | "musicgen-medium"
        | "musicgen-large"
        // MusicGen-Melody sibling (2026-08-02 Wave 5, medium 1.5B LM +
        // chromagram conditioning frontend, `facebook/musicgen-melody`
        // cc-by-nc-4.0 per HF cardData primary source). Both the raw
        // arch/name variants and the `vokra/` publish repo slug route
        // here so the M2-13 runtime gate refuses commercial-mode loads
        // fail-closed regardless of which id form the caller supplies
        // (mirror of the `facebook/mms-1b-all` | `vokra/mms-1b-all`
        // arm above).
        | "musicgen-melody"
        | "musicgen_melody"
        | "facebook/musicgen-melody"
        | "vokra/musicgen-melody"
        | "audiogen-medium"
        | "audiogen" => LicenseClass::NonCommercial,
        // 2026-08-02 Wave residual: Coqui XTTS-v2 (`coqui/XTTS-v2`,
        // `coqui-public-model-license`). Coqui's bespoke research-only
        // license — not SPDX-listed, so the string-based
        // `from_license_str` classifier cannot recognise it (falls through
        // to `Unknown` = fail-closed refuse). This registry override anchors
        // the family on `LicenseClass::NonCommercial` so `vokra/xtts-v2`
        // routes through the T4 (Research-only) publish path per X-Codec-2
        // (2026-07-28) / MusicGen family (2026-08-01) precedent. `xtts`
        // covers the bare arch tag; `xtts-v2` covers the model-id stamp;
        // `xttsv2` covers the compact spelling. Coqui shut down Jan 2024
        // but the upstream repo remains the primary source. Publish
        // requires `publish-one.sh --allow-noncommercial`.
        "xtts" | "xtts-v2" | "xttsv2" => LicenseClass::NonCommercial,
        // 2026-08-02 Wave residual: Meta Seamless-M4T-v2-Large
        // (`facebook/seamless-m4t-v2-large`, **cc-by-nc-4.0** per HF
        // cardData primary source — CLAUDE.md「ハルシネーション厳禁」).
        // 2.3B unified any-to-any speech-and-text translation model
        // (Communication et al. 2023, arXiv:2312.05187) — ASR + T2TT +
        // S2TT + T2ST + S2ST across ~100 source / ~35 target speech
        // languages. Distinct arch tag `unity-2` (Meta's fairseq2
        // dispatch name) covering the 4 subgraphs (w2v-BERT enc + text
        // dec + T2U + HiFi-GAN vocoder). Registered as **explicit exact-
        // match arms** so the fail-closed publish path forces
        // `publish-one.sh --allow-noncommercial` at publish time + the
        // M2-13 runtime gate refusal in commercial mode. `seamless-m4t-
        // v2-large` / `seamless_m4t_v2_large` cover the model-id
        // spellings; `unity-2` / `unity_2` cover the arch tag stamped
        // by the converter; `facebook/seamless-m4t-v2-large` covers the
        // upstream HF slug; `vokra/seamless-m4t-v2-large` covers the
        // publish repo slug per the ELVIS-Act / T4 tier gate. Same T4
        // (Research-only) tier as X-Codec 2 (2026-07-28 precedent) /
        // MusicGen family (2026-08-01) / CrisperWhisper + MMS-1B-All
        // (2026-08-02 wave).
        "seamless-m4t-v2-large"
        | "seamless_m4t_v2_large"
        | "seamlessm4t-v2-large"
        | "seamlessm4t_v2_large"
        | "seamless-m4t-v2"
        | "seamless_m4t_v2"
        | "unity-2"
        | "unity_2"
        | "facebook/seamless-m4t-v2-large"
        | "vokra/seamless-m4t-v2-large" => LicenseClass::NonCommercial,
        // 2026-08-01 Wave 6 residual — permissive audio-LLM / VC-sibling /
        // multi-file bundle. All apache-2.0 / MIT clean.
        "qwen2-audio"
        | "qwen2-audio-7b"
        | "qwen2-audio-7b-instruct"
        | "vibevoice-asr"
        | "ace-step"
        | "ace-step-1.5"
        | "ace_step"
        | "ace-step-1_5" => LicenseClass::Permissive,
        // 2026-08-02 Wave residual: Alibaba Qwen2.5-Omni-7B
        // (`Qwen/Qwen2.5-Omni-7B`, apache-2.0 per HF primary source
        // cardData). Thinker + Talker unified any-to-any omni
        // multimodal LLM over a Qwen2.5-7B backbone. Distinct arch
        // tag `qwen2-omni` from sibling `qwen2_audio` (audio-only
        // Whisper + Qwen2-7B LM) — the two share a family lineage
        // but the fused Thinker + Talker pair fixes a different
        // tensor topology, so the arch tag must stay distinct
        // (FR-EX-08 no silent shape misroute). `qwen2-omni` covers
        // the arch stamp; `qwen2-5-omni-7b` covers the model-id
        // stamp; `qwen2-5-omni` covers the family stamp.
        "qwen2-omni"
        | "qwen2-5-omni-7b"
        | "qwen2-5-omni"
        | "qwen2_5_omni_7b"
        | "qwen/qwen2.5-omni-7b"
        | "vokra/qwen2-5-omni-7b" => LicenseClass::Permissive,
        // 2026-08-01 Wave 7 residual — Meta HuBERT-Large-LS960
        // (`facebook/hubert-large-ls960-ft`, apache-2.0 per HF cardData
        // primary source). 317M self-supervised speech encoder + CTC
        // head fine-tuned on LibriSpeech 960h. Distinct arch tag
        // `hubert` from sibling wav2vec2 (different pretraining
        // objective) — the two share ops but the arch tag stays
        // distinct so runtime dispatch cannot misroute silently.
        "hubert"
        | "hubert-large-ls960"
        | "hubert_large_ls960"
        | "hubert-large-ls960-ft"
        | "facebook/hubert-large-ls960-ft" => LicenseClass::Permissive,
        // 2026-08-02 Wave residual — Meta HT-Demucs (`facebook/demucs`, MIT
        // per upstream `github.com/facebookresearch/demucs` LICENSE primary
        // source; HF mirror returned 401 on the 2026-08-02 residual walk,
        // so the SPDX id anchors on the upstream GitHub `LICENSE` file per
        // memory `[[feedback-license-signoff-primary-source]]`). Hybrid
        // transformer Demucs (Rouard et al. 2023, arXiv:2211.08553) =
        // U-Net waveform branch + spectrogram branch + cross-domain self-
        // attention, 4-source music separation (drums / bass / other /
        // vocals). Distinct arch tag `demucs` from sibling SepFormer /
        // TIGER separators (different internal domain + different output
        // branching — FR-EX-08 forbids silent misroute across separator
        // families). Category `separation` shared with the sibling
        // separator families.
        "demucs"
        | "demucs-htdemucs"
        | "demucs_htdemucs"
        | "htdemucs"
        | "ht-demucs"
        | "facebook/demucs" => LicenseClass::Permissive,
        // 2026-08-02 Wave residual — Ultravox v0.5 (Llama-3.2-1B)
        // (`fixie-ai/ultravox-v0_5-llama-3_2-1b`, MIT). Audio-text-to-text
        // multimodal = Llama-3.2-1B decoder + Whisper encoder + projection
        // adapter. Weight license = **MIT** per HF cardData (SoTA scope-
        // expansion 2026-07-30 canary sweep). Sibling to the first-party
        // Whisper / piper-plus / Silero / CAM++ / Moonshine Permissive
        // posture. Distinct arch tag `ultravox` from sibling Voxtral
        // (Mistral decoder) / Qwen2-Audio (Qwen2 decoder) — the decoder
        // backbone fixes tensor layout + tokenizer + rope base, so FR-EX-08
        // forbids silent shape misroute across the three families.
        "ultravox"
        | "ultravox-v0-5-llama-3-2-1b"
        | "ultravox_v0_5_llama_3_2_1b"
        | "ultravox-v0_5-llama-3_2-1b"
        | "fixie-ai/ultravox-v0_5-llama-3_2-1b"
        | "vokra/ultravox-v0-5-llama-3-2-1b" => LicenseClass::Permissive,
        // --- Copyleft (share-alike, redistributable with LICENSE preserved) --
        //
        // 2026-08-02 Wave residual — JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k
        // (Asteroid ConvTasNet single-speaker enhancement, cc-by-sa-4.0 per
        // HF cardData primary source). **First entry on the
        // `LicenseClass::Copyleft` arm.** The SA cascade propagates to
        // derivatives — a GGUF built from a CC-BY-SA weight is itself
        // CC-BY-SA, so downstream re-labelling as Apache-2.0 is a
        // misrepresentation, not a mere attribution drop.
        // `from_license_str("cc-by-sa-4.0")` already lands the same
        // Copyleft class (share-alike arm is tested before plain cc-by
        // per the ordering pin in `Self::from_license_str`), but this
        // registry override anchors the family on Copyleft so a
        // `vokra/conv-tasnet-libri1mix` publish gate can look up the class
        // without re-parsing the SPDX id. Publish is **redistributable
        // with the original licence preserved** (T3 tier) — no
        // `--allow-noncommercial` required (Copyleft ≠ NonCommercial),
        // but the SA cascade must carry forward on every derivative.
        "conv-tasnet"
        | "conv_tasnet"
        | "convtasnet"
        | "conv-tasnet-libri1mix"
        | "conv_tasnet_libri1mix"
        | "convtasnet-libri1mix"
        | "conv-tasnet-libri1mix-enhsingle-16k"
        | "conv_tasnet_libri1mix_enhsingle_16k"
        | "joriscos/convtasnet_libri1mix_enhsingle_16k"
        | "vokra/conv-tasnet-libri1mix" => LicenseClass::Copyleft,
        // --- gated: CC-BY-NC-SA (research flag) ------------------------------
        "fish-speech" | "fish-speech-v1.4" | "fish-speech-v1.5" => {
            LicenseClass::NonCommercialShareAlike
        }
        // AudioLDM 2 (Wave 5 candidate, 2026-08-01) — CVSSP primary
        // source (Liu et al. 2024 ICML arXiv:2308.05734 paper §Ethics +
        // GitHub `haoheliu/AudioLDM2` README) pins CC-BY-NC-SA 4.0.
        // The HF card `cvssp/audioldm2` carries the looser `-nc-4.0`
        // tag, but the CVSSP-owned primary source is the ShareAlike
        // form — we follow the more restrictive of the two conflicting
        // declarations (same Fish-Speech pattern above; the SA cascade
        // is the load-bearing part of the classification and dropping
        // it silently would silently mark derivatives as re-licensable
        // outside CC-BY-NC-SA). **Publish blocked** at the
        // `signoff_match.py::REPO_TO_SIGNOFF_ROWS` layer until an
        // owner ADR resolves the SA cascade onto Vokra-added artifacts.
        "audioldm2"
        | "audio-ldm-2"
        | "audio_ldm_2"
        | "audioldm-2"
        | "audioldm_2"
        | "cvssp-audioldm2"
        | "cvssp/audioldm2" => LicenseClass::NonCommercialShareAlike,
        // AudioLDM 2 Large (Wave 8 sibling, 2026-08-02) — wider/deeper
        // sibling of the base AudioLDM 2 variant, same CVSSP primary-
        // source license (CC-BY-NC-SA-4.0). The `vokra/audioldm2-large`
        // repo slug resolves here so a future publish gate lookup finds
        // the same doubly-restrictive class (NC gate + SA cascade) as
        // sibling base. **Publish blocked** at the
        // `signoff_match.py::REPO_TO_SIGNOFF_ROWS` layer until an owner
        // ADR resolves the SA cascade onto Vokra-added artifacts (same
        // posture as sibling `cvssp/audioldm2` above).
        "audioldm2-large"
        | "audio-ldm-2-large"
        | "audio_ldm_2_large"
        | "audioldm-2-large"
        | "audioldm_2_large"
        | "cvssp-audioldm2-large"
        | "cvssp/audioldm2-large"
        | "vokra/audioldm2-large" => LicenseClass::NonCommercialShareAlike,
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
            // SoTA plan Phase 4, 2026-07-24; 2B variant land 2026-07-30):
            // specific HF variant ids like `voxcpm-0.5b` / a future
            // `voxcpm-0.5b-customvoice` / `voxcpm-1.5b` still resolve
            // permissive. Guarded on the dash so unrelated ids
            // (`voxcpmsomething`) cannot slip through. The `voxcpm2`
            // (arch-tag) alias covers only the arch-tag spelling; the
            // `voxcpm2-` prefix covers the 2026-07-30 rename family
            // (`voxcpm2-2b`, `voxcpm2-0.5b`) plus any future 2B-lineage
            // variant that keeps the arch-family name.
            || id.starts_with("voxcpm-")
            || id.starts_with("voxcpm2-")
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
        // pyannote first-party family (MIT weight — 2026-07-30 license half
        // unblock、`docs/license-audit.md` §3.1 row 263): specific HF variant
        // ids like `pyannote-segmentation-3.0` / `pyannote-speaker-
        // diarization-3.1` or future `pyannote-segmentation-4.0` still
        // resolve permissive. Guarded on the dash so unrelated ids
        // (`pyannote-something-not-shipped-by-pyannote`) cannot slip
        // through. Exact-name aliases (`pyannote` / `pyannote-segmentation`
        // 等) are pinned in the fast-path arm above; this prefix walk covers
        // future variants of the same family under the same MIT LICENSE at
        // `github.com/pyannote/pyannote-audio/LICENSE`.
        _ if id.starts_with("pyannote-") => LicenseClass::Permissive,
        // 2026-07-30 TIER 1+2 audio-gap family walks (ultracode
        // `wf_022575ce-077` land): each family under apache-2.0 or MIT per
        // its HF cardData primary source verified 2026-07-30. See
        // `docs/handoff/tier1-tier2-audio-impl-2026-07-30.md` for the
        // per-family primary source URL list.
        _ if id.starts_with("qwen3-asr-") => LicenseClass::Permissive,
        // 2026-08-01 wave: IBM Granite Speech family (apache-2.0 end-to-
        // end): future variants like `granite-speech-4.1-8b` /
        // `granite-speech-3.3-8b` still resolve permissive. Guarded on
        // the dash so unrelated ids (`granite-speechx-something`) cannot
        // slip through into the permissive bucket by accident.
        _ if id.starts_with("granite-speech-") => LicenseClass::Permissive,
        _ if id.starts_with("wav2vec2") => LicenseClass::Permissive,
        // 2026-08-02 wave: `facebook/data2vec-audio-base-960h` (apache-2.0).
        // Baevski et al. 2022 (arXiv:2202.03555): wav2vec 2.0 base
        // topology + data2vec pretraining objective + LibriSpeech 960h
        // English char CTC head. Every future `data2vec-audio-*` sibling
        // (base / large / bookish / whatever Meta releases next) stays
        // permissive by prefix walk — Meta's data2vec fleet is uniformly
        // apache-2.0 to date. Placed as its own arm rather than folded
        // into the `wav2vec2` prefix so a future non-apache release
        // cannot silently inherit the classification without explicit
        // review (the `data2vec` bucket is architecturally distinct
        // from wav2vec2 despite sharing the downstream inference
        // topology — different pretraining objective).
        _ if id.starts_with("data2vec-audio") || id.starts_with("data2vec_audio") => {
            LicenseClass::Permissive
        }
        _ if id.starts_with("moss-tts") || id.starts_with("openmoss-team/moss-tts") => {
            LicenseClass::Permissive
        }
        // 2026-08-01 Wave 4 slug-only add: OpenMOSS Team
        // **MOSS-VoiceGenerator** (`OpenMOSS-Team/MOSS-VoiceGenerator`,
        // apache-2.0). Sibling HF release of the `moss_tts` LLM family
        // under the same `moss_tts_delay` internal `model_type` tag, so
        // topology is already covered by [`MossTtsVariant::Delay`] and
        // no new converter arm is needed at this layer. Primary source
        // = HF cardData `license: apache-2.0` (CC 直接照合 2026-08-01,
        // `https://huggingface.co/api/models/OpenMOSS-Team/MOSS-VoiceGenerator`).
        // Registered as explicit exact-match arms rather than routed
        // through the `moss-tts` prefix walk because the ids do not
        // share that prefix (`moss-voice-generator` starts with
        // `moss-v`, not `moss-tts`), and this keeps a hypothetical
        // future `moss-voice-*` sibling from silently inheriting the
        // classification without an explicit review. Guarded by the
        // dash / underscore variants only — anything else fails
        // through to `Unknown` (fail-closed).
        "moss-voice-generator"
        | "moss_voice_generator"
        | "moss-voicegenerator"
        | "moss_voicegenerator"
        | "openmoss-team/moss-voice-generator"
        | "openmoss-team/moss-voicegenerator" => LicenseClass::Permissive,
        // 2026-08-01 Wave 3: MOSS-Audio-Tokenizer family — the codec
        // half of the MOSS-TTS pipeline (both Full + Nano apache-2.0
        // per HF cardData API verified 2026-08-01). Prefix walk
        // covers future variants OpenMOSS Team may ship (e.g. a
        // hypothetical v2 or additional distillations) under the same
        // apache-2.0 licensing. Guarded so unrelated ids like
        // `moss-audio-tokenizerx-something` cannot slip through.
        _ if id.starts_with("moss-audio-tokenizer")
            || id.starts_with("openmoss-team/moss-audio-tokenizer") =>
        {
            LicenseClass::Permissive
        }
        _ if id.starts_with("melotts-") || id.starts_with("myshell-ai/melotts-") => {
            LicenseClass::Permissive
        }
        _ if id.starts_with("speecht5-") || id == "speecht5" => LicenseClass::Permissive,
        _ if id.starts_with("parler-tts") || id.starts_with("indic-parler") => {
            LicenseClass::Permissive
        }
        _ if id.starts_with("vieneu-") || id.starts_with("pnnbao-ump/vieneu-") => {
            LicenseClass::Permissive
        }
        _ if id.starts_with("bark") || id.starts_with("suno/bark") => LicenseClass::Permissive,
        _ if id.starts_with("hifigan-vocoder") || id.starts_with("speechbrain/tts-hifigan-") => {
            LicenseClass::Permissive
        }
        _ if id.starts_with("bigvgan") || id.starts_with("nvidia/bigvgan") => {
            LicenseClass::Permissive
        }
        _ if id.starts_with("focalcodec") || id.starts_with("lucadellalib/focalcodec") => {
            LicenseClass::Permissive
        }
        // 2026-08-01 wave: Charactr AI Vocos family — MIT end-to-end
        // per HF cardData API `license: mit` on both mel-24khz and
        // encodec-24khz repos (verified 2026-08-01). Prefix walk
        // covers any future Vocos variant Charactr AI ships (e.g. a
        // future `charactr/vocos-mel-48khz`) so an untagged GGUF
        // resolves permissive without needing a rebuild of this arm.
        _ if id.starts_with("vocos") || id.starts_with("charactr/vocos-") => {
            LicenseClass::Permissive
        }
        // 2026-08-01 Wave 3 sibling-pair add: YuE bundle family
        // (`m-a-p/YuE-upsampler` + `m-a-p/xcodec_mini_infer`) —
        // apache-2.0 end-to-end per HF cardData API on both repos
        // (verified 2026-08-01). Prefix walk covers any future YuE
        // variant m-a-p ships (e.g. a hypothetical yue-upsampler-v2
        // or an xcodec_mini_v2 refresh) so an untagged GGUF resolves
        // permissive without needing a rebuild of this arm. Guarded
        // so unrelated ids (`yuejun-something` etc.) cannot slip
        // through into the permissive bucket by accident.
        _ if id.starts_with("yue-")
            || id.starts_with("yue_")
            || id.starts_with("xcodec-mini")
            || id.starts_with("xcodec_mini")
            || id.starts_with("m-a-p/yue-")
            || id.starts_with("m-a-p/xcodec_mini") =>
        {
            LicenseClass::Permissive
        }
        _ if id.starts_with("tiger-") || id.starts_with("jusperlee/tiger-") => {
            LicenseClass::Permissive
        }
        _ if id.starts_with("mp-senet") || id.starts_with("jacoblincool/mp-senet-") => {
            LicenseClass::Permissive
        }
        _ if id.starts_with("metricgan-") || id.starts_with("speechbrain/metricgan-") => {
            LicenseClass::Permissive
        }
        _ if id.starts_with("sepformer-") || id.starts_with("speechbrain/sepformer-") => {
            LicenseClass::Permissive
        }
        _ if id.starts_with("fsmn-vad") || id.starts_with("funasr/fsmn-") => {
            LicenseClass::Permissive
        }
        _ if id.starts_with("firered-vad") || id.starts_with("fireredteam/firered") => {
            LicenseClass::Permissive
        }
        _ if id.starts_with("smart-turn") || id.starts_with("pipecat-ai/smart-turn") => {
            LicenseClass::Permissive
        }
        _ if id.starts_with("clap") || id.starts_with("laion/clap-") => LicenseClass::Permissive,
        _ if id.starts_with("ast") || id.starts_with("mit/ast-") => LicenseClass::Permissive,
        _ if id.starts_with("lang-id-") || id.starts_with("speechbrain/lang-id-") => {
            LicenseClass::Permissive
        }
        _ if id.starts_with("xvector") || id.starts_with("speechbrain/spkrec-xvect-") => {
            LicenseClass::Permissive
        }
        _ if id.starts_with("deepfake-detection") || id.starts_with("melodymachine/deepfake-") => {
            LicenseClass::Permissive
        }
        _ if id.starts_with("kyutai-tts") => LicenseClass::AttributionRequired,
        _ if id.starts_with("audiobox-aesthetics") => LicenseClass::AttributionRequired,
        // Defer markers (vast.ai / gated / license-精査-要): fail-closed by
        // returning None here; owner ADR unblocks per model.
        _ if id.starts_with("voxtral-mini-realtime") => LicenseClass::Permissive,
        _ if id.starts_with("cohere-transcribe") => LicenseClass::Permissive,
        // nvidia/nemotron-3.5-asr-streaming-* family — OpenMDW-1.1
        // (Open Model Derivatives Work 1.1、openmdw.ai/license/1-1/、
        // 2026-07-30 CC 直接照合)。permissive MIT-analog for ML weights =
        // commercial + redistribution 可、no share-alike / no NC / no
        // field-of-use restriction、attribution = notice 保持のみ
        // (Apache-2.0 同 tier)。owner ADR 完了 = 暫定から確定 Permissive
        // へ (`docs/license-audit.md` §3.1 row 更新済)。
        _ if id.starts_with("nemotron-asr") => LicenseClass::Permissive,
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
        // 2026-07-30 license half unblock: pyannote family (MIT) — canonical
        // + variant spellings + case-insensitive + family prefix walk.
        // `docs/license-audit.md` §3.1 row 263 で 2026-07-30 yousan sign。
        for id in [
            "pyannote",
            "pyannote-segmentation",
            "pyannote-segmentation-3.0",
            "pyannote-segmentation-3_0",
            "pyannote-speaker-diarization",
            "pyannote-speaker-diarization-3.1",
            // Case-insensitive (via lower-casing before lookup).
            "PyAnnote-Segmentation-3.0",
            "PYANNOTE",
            // Family prefix — a hypothetical future `pyannote-segmentation-
            // 4.0` / `pyannote-vad-v2` still resolves permissive by the walk.
            "pyannote-segmentation-4.0",
            "pyannote-vad-v2",
        ] {
            assert_eq!(registry_lookup(id), Some(LicenseClass::Permissive), "{id}");
        }
        // Guard: a random id starting with "pyannotex" (no dash) is NOT under
        // the family prefix walk.
        assert_eq!(registry_lookup("pyannotex-something"), None);
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
            // 2026-07-30 Option C hybrid rename: `voxcpm2-*` names.
            "voxcpm2-0.5b",
            "voxcpm2-0_5b",
            "voxcpm2-2b",
            "voxcpm2-2_0b",
            "voxcpm2-2b-base",
            // Case-insensitive on the arch-family form too.
            "VoxCPM2",
            // Prefix arm (future 2B-lineage variants).
            "voxcpm2-2b-customvoice",
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

        // ---- 2026-08-02 wave: MMS-1B-All (NonCommercial) --------------
        //
        // Meta MMS-1B-All (`facebook/mms-1b-all`, cc-by-nc-4.0 per HF
        // cardData primary source verified 2026-08-02 —
        // CLAUDE.md「ハルシネーション厳禁」). Pratap et al. 2023
        // (arXiv:2305.13516) — 1B wav2vec 2.0 backbone + 1000+
        // per-language CTC adapters. Every id form MUST resolve to
        // `NonCommercial` (T4 tier / Research-only publish path per the
        // X-Codec-2 (2026-07-28) precedent). CRUCIALLY, this arm must
        // beat the sibling `_ if id.starts_with("wav2vec2")` prefix walk
        // (which would silently return `Permissive`) — the exact-match
        // arm is placed BEFORE the prefix walk for that reason.
        for id in [
            "mms-1b-all",
            "mms_1b_all",
            "mms-1b",
            "mms_1b",
            "facebook/mms-1b-all",
            "vokra/mms-1b-all",
            // Case-insensitive.
            "MMS-1B-ALL",
            "Facebook/MMS-1B-All",
        ] {
            let c = registry_lookup(id);
            assert_eq!(
                c,
                Some(LicenseClass::NonCommercial),
                "mms-1b-all: {id} MUST be NonCommercial (HF card = \
                 cc-by-nc-4.0) — silently returning Permissive would \
                 authorise a commercial load of an NC weight."
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
