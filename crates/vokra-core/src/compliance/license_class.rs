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
        "piper-plus" | "piper-plus-mb-istft-vits2" => LicenseClass::Permissive,
        "silero-vad" | "silero-vad-v5" => LicenseClass::Permissive,
        "campplus" | "cam++" => LicenseClass::Permissive,
        "kokoro" | "kokoro-82m" | "cosyvoice" | "cosyvoice2" | "sesame-csm" | "csm-1b"
        | "voxtral" | "openwakeword" => LicenseClass::Permissive,
        // Commercial-OK codecs (FR-OP-32): DAC / WavTokenizer / X-Codec 2 = MIT.
        "dac" | "wavtokenizer" | "x-codec-2" | "xcodec2" => LicenseClass::Permissive,
        // --- attribution-required (CC-BY-4.0) --------------------------------
        "mimi" | "moshi" => LicenseClass::AttributionRequired,
        // --- gated: CC-BY-NC (research flag) ---------------------------------
        "f5-tts" | "encodec" => LicenseClass::NonCommercial,
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
            || id.starts_with("cosyvoice-") =>
        {
            LicenseClass::Permissive
        }
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
}
