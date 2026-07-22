//! Weight-license compliance: the CC-BY-NC research-flag gate and the
//! compliance settings API (M2-13, FR-CP-03 / FR-CP-06).
//!
//! # Scope boundary (M2-13, ADR-style note)
//!
//! This module implements two things and deliberately excludes a third:
//!
//! - **(in) research-flag gate** (FR-CP-03 / FR-MD-10 / FR-OP-32): a model's
//!   **weight** license is classified ([`LicenseClass`]); a non-commercial or
//!   unknown class is refused unless an explicit research flag is set. This is
//!   what keeps F5-TTS / Fish-Speech / EnCodec (CC-BY-NC) weights off the
//!   default / official-zoo path.
//! - **(in) compliance settings API** (FR-CP-06): [`ComplianceLevel`],
//!   [`WatermarkConfig`] and the policy types, mirroring
//!   `docs/legal-compliance.md` §8.
//! - **(out, deferred) watermark embedding** (FR-CP-01/02): dropped by the
//!   client on 2026-07-04. [`WatermarkConfig`] keeps its config surface with
//!   design-intent defaults, but no audio is watermarked
//!   ([`WatermarkConfig::backend_status`] == [`WatermarkBackendStatus::Deferred`]).
//!
//! # Two separate license mechanisms (do not conflate)
//!
//! - **weight license** (this module): the license of a model's *weights*,
//!   read from `vokra.provenance.*` GGUF metadata. Gated here.
//! - **crate license** (NOT this module): the license of a Rust *dependency*,
//!   gated by `cargo-deny` (NFR-LC-02/04, a CI check). GPL/LGPL crates are
//!   excluded there, not here.
//!
//! A model can be MIT-*code* yet CC-BY-NC-*weight* (F5-TTS, EnCodec); only the
//! weight side is what this gate concerns.
//!
//! # Source of truth
//!
//! Every license classification is a transcription of `docs/license-audit.md`
//! §3 (see [`LicenseClass`]); this module introduces no independent judgement.
//!
//! # Fail-closed
//!
//! An unclassifiable weight resolves to [`LicenseClass::Unknown`], which is
//! gated. Silent load of a gated weight is forbidden (the same "explicit error,
//! never a silent fallback" rule as the GPU-op gate, NFR-RL-06): the loader
//! returns [`VokraError::ResearchLicenseRequired`], it never skips or
//! substitutes.

mod consent;
mod level;
mod license_class;
mod watermark;

pub use consent::{ConsentManifest, ConsentScope, SignatureStatus};
pub use level::{
    ComplianceConfig, ComplianceLevel, DisclosureConfig, SpeakerEmbeddingPolicy, VoiceCloningPolicy,
};
pub use license_class::{LicenseClass, registry_lookup};
pub use watermark::{WatermarkBackendStatus, WatermarkConfig};

use crate::error::{Result, VokraError};
use crate::gguf::{GgufBuilder, GgufFile, chunks};

/// Environment variable that unlocks the research-flag gate (one of the three
/// opt-in routes, alongside a builder flag and [`ComplianceLevel::Research`]).
pub const ENV_ALLOW_RESEARCH_LICENSE: &str = "VOKRA_ALLOW_RESEARCH_LICENSE";

/// How a [`LicenseResolution`] was reached, for diagnostics and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResolutionSource {
    /// From an explicit `vokra.provenance.weight_license` canonical class.
    ExplicitClass,
    /// From a raw `vokra.provenance.license` string (normalized).
    ExplicitLicense,
    /// From the built-in registry keyed on a model id (`vokra.provenance.model_id`
    /// or a `vokra.model.*` fallback).
    Registry,
    /// No signal at all — fell back to [`LicenseClass::Unknown`] (fail-closed).
    Fallback,
}

/// The outcome of classifying a model's weight license.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LicenseResolution {
    /// The resolved class (drives the gate).
    pub class: LicenseClass,
    /// Best-effort license label for messages: the raw string when known, else
    /// the canonical class name.
    pub license: String,
    /// Best-effort model identifier for messages (may be `"unknown"`).
    pub model_id: String,
    /// How [`Self::class`] was determined.
    pub source: ResolutionSource,
}

impl LicenseResolution {
    /// Whether the resolved weight is research-only (non-commercial / unknown),
    /// i.e. gated. After a successful [`check_weight_license`] this is the
    /// "non-commercial marker" downstream keeps on the loaded model (T06).
    pub fn is_research_only(&self) -> bool {
        self.class.requires_research_flag()
    }
}

/// The compliance policy the weight-license gate reads at load time.
///
/// Constructed from a [`ComplianceLevel`] and/or the explicit research opt-in.
/// The research gate is unlocked when **any** of the three FR-CP-03 routes is
/// set: [`Self::with_research_license`] (builder), the
/// [`ENV_ALLOW_RESEARCH_LICENSE`] env var (via [`Self::from_env`]), or a
/// [`ComplianceLevel`] of [`Research`](ComplianceLevel::Research) /
/// [`Disabled`](ComplianceLevel::Disabled). Default is fail-closed
/// ([`ComplianceLevel::Strict`], no opt-in).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompliancePolicy {
    level: ComplianceLevel,
    allow_research_license: bool,
}

impl Default for CompliancePolicy {
    fn default() -> Self {
        Self::strict()
    }
}

impl CompliancePolicy {
    /// The default fail-closed policy: [`ComplianceLevel::Strict`], no research
    /// opt-in. A non-commercial / unknown weight is rejected.
    pub fn strict() -> Self {
        Self {
            level: ComplianceLevel::Strict,
            allow_research_license: false,
        }
    }

    /// A policy at `level` with no explicit research opt-in. (`Research` /
    /// `Disabled` levels unlock the gate on their own;
    /// `Strict` / `Standard` do not.)
    pub fn new(level: ComplianceLevel) -> Self {
        Self {
            level,
            allow_research_license: false,
        }
    }

    /// Sets the explicit research opt-in (the builder route of FR-CP-03;
    /// distinct from the `vokra-voiceclone-experimental` risk flags, which are
    /// a *different* mechanism — voice cloning is always disabled in core).
    #[must_use]
    pub fn with_research_license(mut self, allow: bool) -> Self {
        self.allow_research_license = allow;
        self
    }

    /// Sets the compliance level.
    #[must_use]
    pub fn with_level(mut self, level: ComplianceLevel) -> Self {
        self.level = level;
        self
    }

    /// A policy seeded from [`ENV_ALLOW_RESEARCH_LICENSE`] (the env route of
    /// FR-CP-03), otherwise [`Self::strict`]. Locale-independent parse
    /// (NFR-RL-01): `1` / `true` / `yes` / `on` (any case) enable it.
    pub fn from_env() -> Self {
        let allow = parse_research_env(std::env::var(ENV_ALLOW_RESEARCH_LICENSE).ok().as_deref());
        Self::strict().with_research_license(allow)
    }

    /// The configured level.
    pub fn level(&self) -> ComplianceLevel {
        self.level
    }

    /// Whether this policy unlocks research-flag (CC-BY-NC / unknown) weights.
    pub fn research_license_allowed(&self) -> bool {
        self.allow_research_license
            || matches!(
                self.level,
                ComplianceLevel::Research | ComplianceLevel::Disabled
            )
    }
}

/// Parses the [`ENV_ALLOW_RESEARCH_LICENSE`] value. Pure (no real env access)
/// so it is unit-testable without mutating process-global state.
fn parse_research_env(val: Option<&str>) -> bool {
    matches!(
        val.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

/// Classifies a model's **weight** license from its GGUF `vokra.provenance.*`
/// metadata, with a built-in-registry and fail-closed fallback.
///
/// Priority (M2-13-T02/T03): an explicit `vokra.provenance.weight_license`
/// canonical class → a raw `vokra.provenance.license` string → the built-in
/// [`registry_lookup`] keyed on `vokra.provenance.model_id` (else
/// `vokra.model.name` / `vokra.model.arch`) → [`LicenseClass::Unknown`]
/// (fail-closed, gated). Never returns a permissive default for missing data.
pub fn resolve_license_class(gguf: &GgufFile) -> LicenseResolution {
    let get = |k: &str| gguf.get(k).and_then(|v| v.as_str()).map(str::to_owned);

    // Best-effort model id for messages / registry (provenance id, else the
    // model name, else the arch tag).
    let model_id = get(chunks::KEY_PROVENANCE_MODEL_ID)
        .or_else(|| get(chunks::KEY_MODEL_NAME))
        .or_else(|| get(chunks::KEY_MODEL_ARCH))
        .unwrap_or_else(|| "unknown".to_owned());

    // 1. explicit canonical class override.
    if let Some(s) = get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE) {
        if let Some(class) = LicenseClass::from_class_str(&s) {
            return LicenseResolution {
                class,
                license: s,
                model_id,
                source: ResolutionSource::ExplicitClass,
            };
        }
        // An unparseable explicit class falls through (do not trust a typo);
        // resolution continues to the raw license string / registry.
    }

    // 2. raw license string.
    if let Some(s) = get(chunks::KEY_PROVENANCE_LICENSE) {
        if !s.trim().is_empty() {
            return LicenseResolution {
                class: LicenseClass::from_license_str(&s),
                license: s,
                model_id,
                source: ResolutionSource::ExplicitLicense,
            };
        }
    }

    // 3. built-in registry by model id.
    if let Some(class) = registry_lookup(&model_id) {
        return LicenseResolution {
            class,
            license: class.as_str().to_owned(),
            model_id,
            source: ResolutionSource::Registry,
        };
    }

    // 4. fail-closed.
    LicenseResolution {
        class: LicenseClass::Unknown,
        license: LicenseClass::Unknown.as_str().to_owned(),
        model_id,
        source: ResolutionSource::Fallback,
    }
}

/// The load-time weight-license gate (FR-CP-03).
///
/// Classifies `gguf`'s weight license and, if it is gated
/// ([`LicenseClass::requires_research_flag`]) while `policy` grants no research
/// opt-in, refuses the load with [`VokraError::ResearchLicenseRequired`] — an
/// **explicit** error carrying the detected license, model id and the unlock
/// hint (never a silent skip / substitution). When the flag *is* set for a
/// gated weight, the load is allowed, a one-line non-commercial warning is
/// emitted to stderr (T06), and the returned [`LicenseResolution::is_research_only`]
/// stays `true` so downstream can mark the model non-commercial.
///
/// Wire this at the model load boundary; see `vokra-models` for the
/// piper-plus loader integration (other model loaders are a follow-up).
pub fn check_weight_license(
    gguf: &GgufFile,
    policy: &CompliancePolicy,
) -> Result<LicenseResolution> {
    let res = resolve_license_class(gguf);
    if res.class.requires_research_flag() {
        if !policy.research_license_allowed() {
            let hint = unlock_hint(&res);
            return Err(VokraError::ResearchLicenseRequired {
                model_id: res.model_id,
                license: res.license,
                hint,
            });
        }
        // Allowed via a research flag: warn (research/eval only), keep marker.
        emit_research_warning(&res);
    }
    Ok(res)
}

/// Builds the human-readable unlock hint embedded in the gate error: the three
/// opt-in routes plus, where known, commercial alternatives.
fn unlock_hint(res: &LicenseResolution) -> String {
    let mut hint = String::from(
        "weight is research/evaluation-only and must not be used for commercial \
         distribution or inference; set one of: \
         CompliancePolicy::with_research_license(true), \
         env VOKRA_ALLOW_RESEARCH_LICENSE=1, or ComplianceLevel::Research",
    );
    // Commercial alternatives from docs/license-audit.md §3 where applicable.
    if res.model_id.to_ascii_lowercase().contains("encodec") {
        hint.push_str(". Commercial alternatives: DAC (MIT) / Mimi (CC-BY 4.0) / WavTokenizer (MIT) / X-Codec 2 (MIT)");
    }
    hint
}

/// Emits the one-line non-commercial warning when a gated weight loads via a
/// research flag (T06). Uses stderr directly — the zero-dependency invariant
/// (NFR-DS-02) forbids pulling in a logging crate; a `tracing`/`log` bridge is
/// a follow-up if the workspace adopts one. Legal sufficiency of the wording is
/// the client's call (FR-MD-13 / X-03).
fn emit_research_warning(res: &LicenseResolution) {
    eprintln!(
        "vokra: WARNING loading research-only weight `{}` (weight license `{}`): \
         non-commercial (CC-BY-NC / CC-BY-NC-SA / unknown), for research / evaluation \
         use only — not for commercial distribution or inference.",
        res.model_id, res.license
    );
}

/// The displayable attribution bundle for an `AttributionRequired` weight
/// (FR-MD-09 — M4-06). Deployers surface [`Self::text`] in their UI /
/// about screen to satisfy the CC-BY 4.0 display obligation (plus the
/// NFR-LG-03 store checklists); `license` and `source_url` are the
/// machine-readable companions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttributionInfo {
    /// Human-readable attribution text (converter-stamped
    /// `vokra.provenance.attribution`, or the registry fallback).
    pub text: String,
    /// The weight license label (e.g. `"CC-BY-4.0"`).
    pub license: String,
    /// Advisory upstream source (URL / repo) when known.
    pub source_url: Option<String>,
}

/// Resolves the attribution surface for a GGUF (FR-MD-09 — M4-06-T23).
///
/// - Weight class **not** [`LicenseClass::requires_attribution`] →
///   `None` (permissive weights carry no display obligation here; the
///   NOTICE file covers code-level attribution).
/// - Attribution-required with the converter-stamped
///   `vokra.provenance.attribution` chunk → that text verbatim.
/// - Attribution-required **without** the chunk (older conversion) → a
///   registry-derived fallback naming the model id, license and source —
///   the "attribution required but no attribution available" combination
///   is structurally unrepresentable (never `None` for a gated class).
pub fn resolve_attribution(gguf: &GgufFile) -> Option<AttributionInfo> {
    let res = resolve_license_class(gguf);
    if !res.class.requires_attribution() {
        return None;
    }
    let source_url = gguf
        .get(chunks::KEY_PROVENANCE_SOURCE)
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let text = match gguf
        .get(chunks::KEY_PROVENANCE_ATTRIBUTION)
        .and_then(|v| v.as_str())
    {
        Some(t) if !t.trim().is_empty() => t.to_owned(),
        // Fallback: never leave an AttributionRequired weight without a
        // displayable string (module docs). The wording mirrors NOTICE §5
        // (Kyutai / CC-BY 4.0) generically by model id.
        _ => format!(
            "This application uses the `{}` model weights (license: {}), \
             which require attribution to their authors{}.",
            res.model_id,
            res.license,
            source_url
                .as_deref()
                .map(|s| format!(" — source: {s}"))
                .unwrap_or_default()
        ),
    };
    Some(AttributionInfo {
        text,
        license: res.license,
        source_url,
    })
}

/// Writes the `vokra.provenance.attribution` chunk (FR-MD-09 — the
/// converter-side companion of [`stamp_provenance`]; M4-06-T22). Empty
/// text is ignored (the runtime fallback then applies).
pub fn stamp_attribution(builder: &mut GgufBuilder, text: &str) {
    if !text.trim().is_empty() {
        builder.add_string(chunks::KEY_PROVENANCE_ATTRIBUTION, text);
    }
}

/// Writes the `vokra.provenance.*` weight-license chunk into a [`GgufBuilder`]
/// — the minimal converter conduit (M2-13, "converter can write a license
/// class"). The offline `vokra-convert` tool calls this so a produced GGUF
/// carries its weight license for the runtime gate.
///
/// `class` is written as the explicit canonical override
/// (`vokra.provenance.weight_license`); `license` (the raw string) and the
/// optional `model_id` / `source` are written when provided. Only the
/// `vokra.*` metadata namespace is touched — no ONNX/protobuf enters the
/// runtime (NFR-DS-02).
pub fn stamp_provenance(
    builder: &mut GgufBuilder,
    class: LicenseClass,
    license: &str,
    model_id: Option<&str>,
    source: Option<&str>,
) {
    builder.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, class.as_str());
    if !license.is_empty() {
        builder.add_string(chunks::KEY_PROVENANCE_LICENSE, license);
    }
    if let Some(id) = model_id {
        builder.add_string(chunks::KEY_PROVENANCE_MODEL_ID, id);
    }
    if let Some(src) = source {
        builder.add_string(chunks::KEY_PROVENANCE_SOURCE, src);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufBuilder;

    /// Builds a minimal parseable GGUF from a metadata-only builder.
    fn parse(b: &GgufBuilder) -> GgufFile {
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    #[test]
    fn env_parser_is_locale_independent_and_strict() {
        for v in ["1", "true", "TRUE", "Yes", "on", " on "] {
            assert!(parse_research_env(Some(v)), "{v}");
        }
        for v in ["0", "false", "", "2", "oui"] {
            assert!(!parse_research_env(Some(v)), "{v}");
        }
        assert!(!parse_research_env(None));
    }

    #[test]
    fn policy_unlock_routes() {
        assert!(!CompliancePolicy::strict().research_license_allowed());
        assert!(
            CompliancePolicy::strict()
                .with_research_license(true)
                .research_license_allowed()
        );
        assert!(CompliancePolicy::new(ComplianceLevel::Research).research_license_allowed());
        assert!(CompliancePolicy::new(ComplianceLevel::Disabled).research_license_allowed());
        assert!(!CompliancePolicy::new(ComplianceLevel::Standard).research_license_allowed());
    }

    #[test]
    fn resolve_prefers_explicit_class_then_license_then_registry() {
        // Explicit canonical class wins (even over a conflicting raw string).
        let mut b = GgufBuilder::new();
        stamp_provenance(
            &mut b,
            LicenseClass::NonCommercial,
            "MIT", // deliberately conflicting raw string
            Some("some-model"),
            None,
        );
        let r = resolve_license_class(&parse(&b));
        assert_eq!(r.class, LicenseClass::NonCommercial);
        assert_eq!(r.source, ResolutionSource::ExplicitClass);

        // Raw license string only.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_PROVENANCE_LICENSE, "CC-BY-NC-SA-4.0");
        let r = resolve_license_class(&parse(&b));
        assert_eq!(r.class, LicenseClass::NonCommercialShareAlike);
        assert_eq!(r.source, ResolutionSource::ExplicitLicense);

        // Registry via model id.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_PROVENANCE_MODEL_ID, "encodec");
        let r = resolve_license_class(&parse(&b));
        assert_eq!(r.class, LicenseClass::NonCommercial);
        assert_eq!(r.source, ResolutionSource::Registry);

        // Registry via vokra.model.arch fallback (first-party permissive).
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "whisper");
        let r = resolve_license_class(&parse(&b));
        assert_eq!(r.class, LicenseClass::Permissive);
        assert_eq!(r.source, ResolutionSource::Registry);
    }

    #[test]
    fn resolve_fails_closed_to_unknown_when_no_signal() {
        // A GGUF with no provenance and an unregistered arch -> Unknown (gated).
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "mystery-arch");
        let r = resolve_license_class(&parse(&b));
        assert_eq!(r.class, LicenseClass::Unknown);
        assert_eq!(r.source, ResolutionSource::Fallback);
        assert!(r.is_research_only());
    }

    #[test]
    fn gate_rejects_noncommercial_without_flag() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_PROVENANCE_LICENSE, "CC-BY-NC-4.0");
        b.add_string(chunks::KEY_PROVENANCE_MODEL_ID, "f5-tts");
        let file = parse(&b);

        let err = check_weight_license(&file, &CompliancePolicy::strict()).unwrap_err();
        match err {
            VokraError::ResearchLicenseRequired {
                model_id,
                license,
                hint,
            } => {
                assert_eq!(model_id, "f5-tts");
                assert_eq!(license, "CC-BY-NC-4.0");
                // The hint lists all three unlock routes.
                assert!(hint.contains("with_research_license"));
                assert!(hint.contains("VOKRA_ALLOW_RESEARCH_LICENSE"));
                assert!(hint.contains("ComplianceLevel::Research"));
            }
            other => panic!("expected ResearchLicenseRequired, got {other:?}"),
        }
    }

    #[test]
    fn gate_allows_noncommercial_with_each_flag_route() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_PROVENANCE_LICENSE, "CC-BY-NC-4.0");
        b.add_string(chunks::KEY_PROVENANCE_MODEL_ID, "f5-tts");
        let file = parse(&b);

        for policy in [
            CompliancePolicy::strict().with_research_license(true),
            CompliancePolicy::new(ComplianceLevel::Research),
            CompliancePolicy::new(ComplianceLevel::Disabled),
        ] {
            let res = check_weight_license(&file, &policy).expect("unlocked");
            assert_eq!(res.class, LicenseClass::NonCommercial);
            assert!(res.is_research_only(), "non-commercial marker stays set");
        }
    }

    #[test]
    fn gate_encodec_via_registry_and_hints_alternatives() {
        // EnCodec is gated by model-id alone (no provenance license), and the
        // error suggests the commercial codec alternatives (FR-OP-32).
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_PROVENANCE_MODEL_ID, "encodec");
        let file = parse(&b);
        let err = check_weight_license(&file, &CompliancePolicy::strict()).unwrap_err();
        match err {
            VokraError::ResearchLicenseRequired { hint, .. } => {
                assert!(hint.contains("DAC"));
                assert!(hint.contains("WavTokenizer"));
            }
            other => panic!("expected ResearchLicenseRequired, got {other:?}"),
        }
    }

    #[test]
    fn gate_always_allows_permissive() {
        // Permissive loads under the strictest policy (no flag needed).
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_PROVENANCE_LICENSE, "MIT");
        b.add_string(chunks::KEY_PROVENANCE_MODEL_ID, "whisper-base");
        let file = parse(&b);
        let res = check_weight_license(&file, &CompliancePolicy::strict()).expect("permissive");
        assert_eq!(res.class, LicenseClass::Permissive);
        assert!(!res.is_research_only());
    }

    #[test]
    fn gate_rejects_unknown_provenance_fail_closed() {
        // No provenance, unregistered arch -> Unknown -> rejected without a flag
        // (the T08 fallback case).
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "mystery-arch");
        let file = parse(&b);
        assert!(matches!(
            check_weight_license(&file, &CompliancePolicy::strict()),
            Err(VokraError::ResearchLicenseRequired { .. })
        ));
    }

    #[test]
    fn attribution_is_none_for_permissive_and_present_for_attribution_required() {
        // Permissive → no display obligation surfaces here.
        let mut b = GgufBuilder::new();
        stamp_provenance(
            &mut b,
            LicenseClass::Permissive,
            "MIT",
            Some("whisper-base"),
            None,
        );
        assert!(resolve_attribution(&parse(&b)).is_none());

        // AttributionRequired + converter-stamped text → verbatim.
        let mut b = GgufBuilder::new();
        stamp_provenance(
            &mut b,
            LicenseClass::AttributionRequired,
            "CC-BY-4.0",
            Some("moshi"),
            Some("https://github.com/kyutai-labs/moshi"),
        );
        stamp_attribution(&mut b, "Moshi weights (c) Kyutai, CC-BY 4.0.");
        let info = resolve_attribution(&parse(&b)).expect("attribution surfaces");
        assert_eq!(info.text, "Moshi weights (c) Kyutai, CC-BY 4.0.");
        assert_eq!(info.license, "attribution-required");
        assert_eq!(
            info.source_url.as_deref(),
            Some("https://github.com/kyutai-labs/moshi")
        );
    }

    #[test]
    fn attribution_required_without_chunk_falls_back_to_registry_text() {
        // The "attribution required but nothing to display" combination is
        // structurally unrepresentable (M4-06-T23): an older conversion
        // without the chunk still yields a non-empty registry-derived text.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_PROVENANCE_MODEL_ID, "moshi");
        let info = resolve_attribution(&parse(&b)).expect("registry fallback fires");
        assert!(
            info.text.contains("moshi"),
            "names the model: {}",
            info.text
        );
        assert!(!info.text.trim().is_empty());

        // Empty stamped text is treated as absent (fallback, not blank UI).
        let mut b = GgufBuilder::new();
        stamp_provenance(
            &mut b,
            LicenseClass::AttributionRequired,
            "CC-BY-4.0",
            Some("mimi"),
            None,
        );
        stamp_attribution(&mut b, "   ");
        let info = resolve_attribution(&parse(&b)).expect("attribution surfaces");
        assert!(!info.text.trim().is_empty(), "fallback replaces blank text");
    }

    #[test]
    fn stamp_provenance_roundtrips_through_resolver() {
        // The converter conduit writes a class the runtime reads back.
        let mut b = GgufBuilder::new();
        stamp_provenance(
            &mut b,
            LicenseClass::Permissive,
            "MIT",
            Some("whisper-base"),
            Some("openai/whisper-base"),
        );
        let file = parse(&b);
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some("permissive")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_SOURCE)
                .and_then(|v| v.as_str()),
            Some("openai/whisper-base")
        );
        let res = check_weight_license(&file, &CompliancePolicy::strict()).expect("permissive");
        assert_eq!(res.class, LicenseClass::Permissive);
    }
}
