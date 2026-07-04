//! Compliance level and policy types (FR-CP-06, M2-13-T09).
//!
//! These types are the in-code form of the `docs/legal-compliance.md` §8
//! sketch (`ComplianceLevel` / `VoiceCloningPolicy` / `SpeakerEmbeddingPolicy` /
//! `DisclosureConfig`, grouped by [`ComplianceConfig`], the `VokraConfig`
//! equivalent). They are **configuration only**: the values that actually gate
//! a load are threaded through a [`CompliancePolicy`](super::CompliancePolicy)
//! derived from this config (see [`ComplianceConfig::policy`]).
//!
//! # What is *not* here (2026-07-04 client drop)
//!
//! The watermark **embedding** backends (AudioSeal / C2PA, FR-CP-01/02) were
//! dropped by the client; [`WatermarkConfig`] keeps its
//! design-intent defaults but performs no embedding (deferred). Region
//! auto-detection beyond a locale hint, and any IP-geolocation, are also
//! deferred to preserve the zero-dependency invariant (NFR-DS-02). See the
//! parent module docs.

use super::WatermarkConfig;

/// Runtime compliance posture (`docs/legal-compliance.md` §8).
///
/// Only [`Research`](Self::Research) (and the escape-hatch
/// [`Disabled`](Self::Disabled)) unlock the CC-BY-NC research-flag gate on the
/// weight loader; [`Strict`](Self::Strict) / [`Standard`](Self::Standard) keep
/// non-commercial weights rejected unless an explicit
/// [`CompliancePolicy::with_research_license`](super::CompliancePolicy::with_research_license)
/// opt-in is set. The default is [`Strict`](Self::Strict) (fail-closed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ComplianceLevel {
    /// Default. All watermark flags on (embedding deferred), voice cloning
    /// disabled, speaker embedding requires consent, EU/CA/TN region hints
    /// steer toward the safest posture. Does **not** unlock research weights.
    #[default]
    Strict,
    /// Watermark on (embedding deferred); voice cloning only via the separate
    /// `vokra-voiceclone-experimental` binary; speaker embedding allowed. Does
    /// **not** unlock research weights.
    Standard,
    /// Watermark opt-out permitted, consent manifest not required, and
    /// **research-flag models (CC-BY-NC: F5-TTS / Fish-Speech / EnCodec) are
    /// permitted to load** (`docs/legal-compliance.md` §8).
    Research,
    /// Compliance fully disabled (self-responsibility; the README carries the
    /// large warning). Also unlocks research weights.
    Disabled,
}

/// Voice-cloning posture. In **vokra-core this is always
/// [`Disabled`](Self::Disabled)** — the enum has no other variant — because
/// voice cloning is split into the separate `vokra-voiceclone-experimental`
/// repository / binary (FR-CP-04, CLAUDE.md design note 8: ELVIS Act / NO FAKES
/// Act tool-distributor liability). The type therefore makes an enabled state
/// *unrepresentable* in core, independent of [`ComplianceLevel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum VoiceCloningPolicy {
    /// The only value core can hold: voice cloning is not available.
    #[default]
    Disabled,
}

/// Speaker-embedding (speaker *feature extraction*, kept in core for zero-shot
/// TTS) posture (`docs/legal-compliance.md` §8). Distinct from voice cloning:
/// embedding extraction stays in core (CLAUDE.md design note 8), gated by a
/// consent policy rather than removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SpeakerEmbeddingPolicy {
    /// A signed consent manifest is required before extracting a speaker
    /// embedding (default under [`ComplianceLevel::Strict`]). Enforcement is
    /// wired to the speaker op in a later WP (`speaker_encode`, FR-OP-80,
    /// post-M2); this type reserves the policy.
    #[default]
    RequireConsent,
    /// Extraction permitted without a consent manifest (Standard/Research).
    Allow,
}

/// AI-disclosure beacon configuration (`docs/legal-compliance.md` §8).
///
/// Config-only in M2-13 (the audible/inaudible beacon emitter is deferred with
/// the watermark backends). Numbers are transcribed from the doc, not invented.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisclosureConfig {
    /// Frequency of the inaudible AI-disclosure beacon, Hz. Default `22050`
    /// (above the audible band), per `docs/legal-compliance.md` §8.
    pub default_beacon_frequency_hz: u32,
    /// Whether a visible AI-generated UI marker is required. Default `true`.
    pub require_visible_ui: bool,
}

impl Default for DisclosureConfig {
    fn default() -> Self {
        // Values transcribed verbatim from docs/legal-compliance.md §8.
        Self {
            default_beacon_frequency_hz: 22050,
            require_visible_ui: true,
        }
    }
}

/// The grouped compliance configuration — the `VokraConfig` of
/// `docs/legal-compliance.md` §8, minus the (deferred) init plumbing.
///
/// Build one and call [`ComplianceConfig::policy`] to get the
/// [`CompliancePolicy`](super::CompliancePolicy) the weight-license gate reads.
/// The default is the §8 default: `Strict`, watermark design-intent defaults,
/// voice cloning disabled, speaker embedding consent-required.
///
/// # Session wiring (deferred)
///
/// Threading this into `Session` / a global `Vokra::init` is a later WP; M2-13
/// wires the gate through an explicit [`CompliancePolicy`](super::CompliancePolicy)
/// argument on the model loader instead (see the parent module docs).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ComplianceConfig {
    /// Overall posture (default [`ComplianceLevel::Strict`]).
    pub level: ComplianceLevel,
    /// Watermark flags (design-intent defaults; embedding deferred).
    pub watermark: WatermarkConfig,
    /// Voice-cloning posture (core: always
    /// [`VoiceCloningPolicy::Disabled`]).
    pub voice_cloning: VoiceCloningPolicy,
    /// Speaker-embedding consent posture.
    pub speaker_embedding: SpeakerEmbeddingPolicy,
    /// AI-disclosure beacon / UI configuration.
    pub disclosure: DisclosureConfig,
}

impl ComplianceConfig {
    /// Derives the [`CompliancePolicy`](super::CompliancePolicy) the
    /// weight-license gate consumes. The research flag is unlocked purely by
    /// [`self.level`](Self::level) here (`Research` / `Disabled`); a Strict or
    /// Standard config that nonetheless needs a research weight must opt in
    /// explicitly via
    /// [`CompliancePolicy::with_research_license`](super::CompliancePolicy::with_research_license).
    pub fn policy(&self) -> super::CompliancePolicy {
        super::CompliancePolicy::new(self.level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_legal_compliance_section_8() {
        let c = ComplianceConfig::default();
        assert_eq!(c.level, ComplianceLevel::Strict);
        assert_eq!(c.voice_cloning, VoiceCloningPolicy::Disabled);
        assert_eq!(c.speaker_embedding, SpeakerEmbeddingPolicy::RequireConsent);
        assert_eq!(c.disclosure.default_beacon_frequency_hz, 22050);
        assert!(c.disclosure.require_visible_ui);
        // Watermark design-intent defaults are asserted in watermark.rs.
        assert!(c.watermark.audioseal);
    }

    #[test]
    fn voice_cloning_is_disabled_only() {
        // The type makes an enabled state unrepresentable in core (FR-CP-04):
        // Default is Disabled and there is no other variant to construct.
        assert_eq!(VoiceCloningPolicy::default(), VoiceCloningPolicy::Disabled);
    }

    #[test]
    fn config_policy_unlocks_only_on_research_or_disabled() {
        let unlock = |lvl| {
            ComplianceConfig {
                level: lvl,
                ..Default::default()
            }
            .policy()
            .research_license_allowed()
        };
        assert!(!unlock(ComplianceLevel::Strict));
        assert!(!unlock(ComplianceLevel::Standard));
        assert!(unlock(ComplianceLevel::Research));
        assert!(unlock(ComplianceLevel::Disabled));
    }
}
