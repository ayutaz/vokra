//! Consent manifest schema + **structural** validator (FR-CP-04, M5-05).
//!
//! This is the in-code form of the signed consent manifest sketched in
//! `docs/legal-compliance.md` §3.3 (the `vokra-voiceclone-experimental`
//! start-up consent workflow, ELVIS Act §3 / NO FAKES Act §4). The schema is a
//! **transcription** of that example — it introduces no field the doc does not
//! list, and no independent legal judgement (the same "SoT is the doc" rule as
//! [`license_class`](super::license_class)).
//!
//! # Two things this module does — and one it deliberately does not
//!
//! - **(in) schema** — [`ConsentManifest`] + [`ConsentScope`], the five §3.3
//!   fields.
//! - **(in) structural validation** — [`ConsentManifest::parse`] reads a JSON
//!   blob with the zero-dependency [`vokra_core::json`](crate::json) parser
//!   (no `serde`, NFR-DS-02) and *fail-closed* rejects a manifest that is
//!   missing a field, carries an unknown [`ConsentScope`], or has an empty
//!   `vokra_session_id` / `grant_date` (FR-EX-08: an explicit reject, never a
//!   silent default).
//! - **(out) cryptographic signature verification** — NOT done here, and there
//!   is deliberately no `Verified` state (see [`SignatureStatus`]).
//!
//! # Why signature *verification* is out of scope (state the reason correctly)
//!
//! The reason is **not** "zero-dep, so no crypto can be written": that is false
//! and this tree disproves it — there are two hand-written, dependency-free
//! SHA-256 implementations already
//! (`vokra-backend-vulkan::spirv::sha256` and
//! `vokra-backend-webgpu::wgsl::sha256`, both FIPS-180-4 with NIST test
//! vectors). Hashing a payload is a settled zero-dep pattern here. The real
//! boundary is two-fold:
//!
//! 1. **A signature is not a hash.** PGP / Ed25519 *signature* verification
//!    needs bignum / elliptic-curve arithmetic **plus** OpenPGP packet parsing
//!    **plus** a trust-root model. A hand-rolled, security-critical verifier
//!    that fails *open* on a subtle bug is worse than none — this is not the
//!    kind of code to invent in-tree.
//! 2. **The trust root is an owner decision, not a code decision.** *Whose*
//!    key signs, how keys are distributed, and how they are revoked is a
//!    policy the maintainer sets (M5-05-T04); the runtime must not fabricate
//!    one.
//!
//! So the core validator observes the *structure* of the `signature` field
//! (present / absent, non-empty / empty) and never claims it verified anything.
//! Real verification, if adopted, lives in the separate
//! `vokra-voiceclone-experimental` binary (an isolated workspace that may take
//! an external crypto crate) or stays deferred — the M5-05
//! `docs/adr/M5-05-watermark-dependency.md` decision, owner's call.

use crate::error::{Result, VokraError};
use crate::json::{self, JsonValue};

/// The scope a voice owner granted consent for (`docs/legal-compliance.md`
/// §3.3 `"consent_scope": "commercial|personal|research"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ConsentScope {
    /// Commercial use of the cloned/derived voice is permitted.
    Commercial,
    /// Personal (non-commercial) use only.
    Personal,
    /// Research use only.
    Research,
}

impl ConsentScope {
    /// Parses the lowercase §3.3 token, or `None` for an unknown scope
    /// (the caller turns `None` into a fail-closed reject).
    pub fn from_token(s: &str) -> Option<Self> {
        match s {
            "commercial" => Some(Self::Commercial),
            "personal" => Some(Self::Personal),
            "research" => Some(Self::Research),
            _ => None,
        }
    }

    /// The canonical lowercase token (round-trips [`Self::from_token`]).
    pub fn as_token(self) -> &'static str {
        match self {
            Self::Commercial => "commercial",
            Self::Personal => "personal",
            Self::Research => "research",
        }
    }
}

/// A **structural** observation of the manifest's `signature` field. This is
/// NOT a verification result — there is deliberately no `Verified` variant
/// (see the module docs): core observes presence, an owner-chosen mechanism
/// verifies (or the field stays deferred).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SignatureStatus {
    /// A non-empty `signature` string is present in the manifest. **This says
    /// nothing about the signature being valid** — only that one was supplied.
    Present,
    /// The `signature` field parsed as a string but is empty — the manifest is
    /// effectively unsigned (`docs/legal-compliance.md` §3.2 "unsigned
    /// embedding → reject").
    Absent,
}

/// A signed consent manifest (`docs/legal-compliance.md` §3.3).
///
/// All five fields are transcribed verbatim from the §3.3 example; the type
/// adds nothing. Construct it via [`Self::parse`], which structurally validates
/// a JSON blob — the [`Default`]-free, explicit-field shape mirrors the doc so
/// no field can be silently defaulted (FR-EX-08).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsentManifest {
    /// `voice_owner_name` — the natural-language name of the voice owner.
    pub voice_owner_name: String,
    /// `consent_scope` — the granted [`ConsentScope`].
    pub consent_scope: ConsentScope,
    /// `grant_date` — the date consent was granted. Stored as the raw string;
    /// strict date-format validation is intentionally NOT done here
    /// (over-engineering — presence + non-empty is the core contract, a
    /// stricter format check is an owner/deployer extension point).
    pub grant_date: String,
    /// `signature` — a "PGP or similar cryptographic signature" (§3.3). Stored
    /// as the raw string; core does **not** verify it (see the module docs and
    /// [`SignatureStatus`]).
    pub signature: String,
    /// `vokra_session_id` — the session the consent binds to (a UUID in the
    /// §3.3 example; core requires only that it is non-empty).
    pub vokra_session_id: String,
}

impl ConsentManifest {
    /// Parses and **structurally** validates a consent-manifest JSON blob.
    ///
    /// Uses the zero-dependency [`vokra_core::json`](crate::json) parser (no
    /// `serde`, NFR-DS-02). Fail-closed (FR-EX-08): a malformed document, a
    /// missing / non-string field, an unknown [`ConsentScope`], or an empty
    /// `vokra_session_id` / `grant_date` is an explicit
    /// [`VokraError::InvalidArgument`] — never a silent default.
    ///
    /// The `signature` field is required to be **present** (a string) but is
    /// allowed to be empty; its emptiness surfaces via
    /// [`Self::signature_status`], not as a parse error, because core does not
    /// distinguish "unsigned" from "signed-but-unverified" — both are the
    /// caller's / owner's concern (see the module docs).
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let root = json::parse(bytes).map_err(|e| {
            VokraError::InvalidArgument(format!(
                "consent manifest: not valid JSON ({e}) — expected the \
                 docs/legal-compliance.md §3.3 object (FR-CP-04)"
            ))
        })?;
        if root.as_object().is_none() {
            return Err(VokraError::InvalidArgument(
                "consent manifest: top-level value must be a JSON object with the five \
                 §3.3 fields (FR-CP-04)"
                    .to_owned(),
            ));
        }

        // Every field must be PRESENT and a STRING (structural check 1). Missing
        // or non-string is a fail-closed reject, never a silent default.
        let string_field = |key: &str| -> Result<String> {
            match root.get(key) {
                Some(JsonValue::Str(s)) => Ok(s.clone()),
                _ => Err(VokraError::InvalidArgument(format!(
                    "consent manifest: required string field `{key}` is missing or not a \
                     string (docs/legal-compliance.md §3.3; FR-EX-08 fail-closed)"
                ))),
            }
        };

        let voice_owner_name = string_field("voice_owner_name")?;
        let scope_token = string_field("consent_scope")?;
        let grant_date = string_field("grant_date")?;
        // `signature` is required to be present as a string (structural check 4),
        // but may be empty — its emptiness surfaces via `signature_status()`,
        // not as a parse error (core does not conflate "unsigned" with
        // "signed-but-unverified"; see the module docs).
        let signature = string_field("signature")?;
        let vokra_session_id = string_field("vokra_session_id")?;

        // Check 2: consent_scope must be a known enum token.
        let consent_scope = ConsentScope::from_token(&scope_token).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "consent manifest: field `consent_scope` must be one of \
                 {{commercial, personal, research}} (§3.3), got `{scope_token}`"
            ))
        })?;

        // Check 3: vokra_session_id must be non-empty.
        if vokra_session_id.trim().is_empty() {
            return Err(VokraError::InvalidArgument(
                "consent manifest: field `vokra_session_id` must be non-empty (§3.3; \
                 FR-EX-08 fail-closed)"
                    .to_owned(),
            ));
        }

        // Check 5: grant_date must be non-empty (strict date-format validation
        // is deliberately not done here — see the field docs).
        if grant_date.trim().is_empty() {
            return Err(VokraError::InvalidArgument(
                "consent manifest: field `grant_date` must be non-empty (§3.3; \
                 FR-EX-08 fail-closed)"
                    .to_owned(),
            ));
        }

        Ok(Self {
            voice_owner_name,
            consent_scope,
            grant_date,
            signature,
            vokra_session_id,
        })
    }

    /// The **structural** signature observation — [`SignatureStatus::Present`]
    /// iff the `signature` string is non-empty (after trimming). This is not a
    /// cryptographic verification (see the module docs / [`SignatureStatus`]).
    pub fn signature_status(&self) -> SignatureStatus {
        if self.signature.trim().is_empty() {
            SignatureStatus::Absent
        } else {
            SignatureStatus::Present
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &[u8] = br#"{
        "voice_owner_name": "Yamada Taro",
        "consent_scope": "commercial",
        "grant_date": "2026-07-02",
        "signature": "-----BEGIN PGP SIGNATURE-----\nAB\n-----END PGP SIGNATURE-----",
        "vokra_session_id": "550e8400-e29b-41d4-a716-446655440000"
    }"#;

    #[test]
    fn scope_round_trips() {
        for s in [
            ConsentScope::Commercial,
            ConsentScope::Personal,
            ConsentScope::Research,
        ] {
            assert_eq!(ConsentScope::from_token(s.as_token()), Some(s));
        }
        assert_eq!(ConsentScope::from_token("bogus"), None);
    }

    #[test]
    fn parses_valid_manifest() {
        let m = ConsentManifest::parse(VALID).expect("valid §3.3 manifest");
        assert_eq!(m.voice_owner_name, "Yamada Taro");
        assert_eq!(m.consent_scope, ConsentScope::Commercial);
        assert_eq!(m.grant_date, "2026-07-02");
        assert_eq!(m.vokra_session_id, "550e8400-e29b-41d4-a716-446655440000");
        assert_eq!(m.signature_status(), SignatureStatus::Present);
    }

    #[test]
    fn all_three_scopes_parse() {
        for tok in ["commercial", "personal", "research"] {
            let blob = format!(
                r#"{{"voice_owner_name":"A","consent_scope":"{tok}","grant_date":"2026-01-01","signature":"s","vokra_session_id":"id"}}"#
            );
            let m = ConsentManifest::parse(blob.as_bytes()).expect("scope parses");
            assert_eq!(m.consent_scope.as_token(), tok);
        }
    }

    // Fail-closed (FR-EX-08): each required field missing is an explicit reject.
    #[test]
    fn each_missing_field_is_rejected() {
        for missing in [
            "voice_owner_name",
            "consent_scope",
            "grant_date",
            "signature",
            "vokra_session_id",
        ] {
            // Build a manifest with every field except `missing`.
            let mut parts: Vec<(&str, &str)> = vec![
                ("voice_owner_name", "\"A\""),
                ("consent_scope", "\"personal\""),
                ("grant_date", "\"2026-01-01\""),
                ("signature", "\"sig\""),
                ("vokra_session_id", "\"id\""),
            ];
            parts.retain(|(k, _)| *k != missing);
            let body = parts
                .iter()
                .map(|(k, v)| format!("\"{k}\":{v}"))
                .collect::<Vec<_>>()
                .join(",");
            let blob = format!("{{{body}}}");
            let err = ConsentManifest::parse(blob.as_bytes())
                .expect_err("missing field must reject (fail-closed)");
            assert!(
                matches!(err, VokraError::InvalidArgument(_)),
                "missing `{missing}` -> {err:?}"
            );
            assert!(
                err.to_string().contains(missing),
                "error should name the missing field `{missing}`: {err}"
            );
        }
    }

    #[test]
    fn non_string_field_is_rejected() {
        // consent_scope as a number, not a string.
        let blob = br#"{"voice_owner_name":"A","consent_scope":3,"grant_date":"2026-01-01","signature":"s","vokra_session_id":"id"}"#;
        let err = ConsentManifest::parse(blob).expect_err("non-string field must reject");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn unknown_scope_is_rejected() {
        let blob = br#"{"voice_owner_name":"A","consent_scope":"broadcast","grant_date":"2026-01-01","signature":"s","vokra_session_id":"id"}"#;
        let err = ConsentManifest::parse(blob).expect_err("unknown scope must reject");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
        assert!(err.to_string().contains("consent_scope"));
    }

    #[test]
    fn empty_session_id_is_rejected() {
        let blob = br#"{"voice_owner_name":"A","consent_scope":"personal","grant_date":"2026-01-01","signature":"s","vokra_session_id":"  "}"#;
        let err = ConsentManifest::parse(blob).expect_err("empty session id must reject");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
        assert!(err.to_string().contains("vokra_session_id"));
    }

    #[test]
    fn empty_grant_date_is_rejected() {
        let blob = br#"{"voice_owner_name":"A","consent_scope":"personal","grant_date":"","signature":"s","vokra_session_id":"id"}"#;
        let err = ConsentManifest::parse(blob).expect_err("empty grant_date must reject");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
        assert!(err.to_string().contains("grant_date"));
    }

    // `signature` present-but-empty PARSES (structural check 4 is presence only),
    // and surfaces as Absent — core does not conflate unsigned with unverified.
    #[test]
    fn empty_signature_parses_as_absent() {
        let blob = br#"{"voice_owner_name":"A","consent_scope":"personal","grant_date":"2026-01-01","signature":"","vokra_session_id":"id"}"#;
        let m = ConsentManifest::parse(blob).expect("empty signature still parses (present-only)");
        assert_eq!(m.signature_status(), SignatureStatus::Absent);
    }

    #[test]
    fn malformed_json_is_rejected() {
        let err = ConsentManifest::parse(b"{not json").expect_err("malformed JSON must reject");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn non_object_top_level_is_rejected() {
        let err = ConsentManifest::parse(b"[1,2,3]").expect_err("array top-level must reject");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    // The honesty boundary: there is no way to reach a "verified" state — the
    // status enum only carries Present / Absent (no Verified variant).
    #[test]
    fn signature_status_has_no_verified_state() {
        let m = ConsentManifest::parse(VALID).expect("valid");
        // A non-empty signature is Present, never "Verified": core does not
        // cryptographically verify (module docs). This test pins that the only
        // reachable states are Present / Absent.
        match m.signature_status() {
            SignatureStatus::Present | SignatureStatus::Absent => {}
        }
    }
}
