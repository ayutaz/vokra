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
/// [`AttributionRequired`] < [`NonCommercial`] / [`NonCommercialShareAlike`] <
/// [`Unknown`]. Everything from [`NonCommercial`] onward is gated
/// ([`Self::requires_research_flag`]); [`Unknown`] is gated deliberately so an
/// unclassifiable weight fails **closed** rather than open.
///
/// [`Permissive`]: LicenseClass::Permissive
/// [`AttributionRequired`]: LicenseClass::AttributionRequired
/// [`NonCommercial`]: LicenseClass::NonCommercial
/// [`NonCommercialShareAlike`]: LicenseClass::NonCommercialShareAlike
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

    /// Whether this class is cleared for commercial use / the official model
    /// zoo (Apache-2.0/MIT/BSD or CC-BY). The public zoo admits only these
    /// (BR-10); [`crate::compliance`] uses this to keep CC-BY-NC weights out of
    /// default paths.
    pub fn commercial_ok(self) -> bool {
        matches!(self, Self::Permissive | Self::AttributionRequired)
    }

    /// Whether downstream must display attribution for this class
    /// (CC-BY-4.0). Advisory (non-gating) — enforced via NOTICE, not this gate.
    pub fn requires_attribution(self) -> bool {
        matches!(self, Self::AttributionRequired)
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
        // Attribution (CC-BY, not -NC): matched only after ruling out -NC.
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
        _ => return None,
    };
    Some(class)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // NVIDIA BigVGAN reference is non-commercial too.
        assert_eq!(
            LicenseClass::from_license_str("NVIDIA Source Code License-NC"),
            LicenseClass::NonCommercial
        );
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
        // Unregistered -> None (caller fails closed to Unknown).
        assert_eq!(registry_lookup("totally-unknown-model"), None);
    }
}
