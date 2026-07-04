//! Error handling for all Vokra public APIs.
//!
//! FR-API-02 mandates that the public Rust API returns
//! [`Result<T, VokraError>`](crate::Result).

use std::fmt;

/// Error type returned by every Vokra public API (FR-API-02).
///
/// The enum is `#[non_exhaustive]`: variants will be added while the v0.1
/// spike progresses (M0-02-T06), so downstream matches must keep a wildcard
/// arm.
#[derive(Debug)]
#[non_exhaustive]
pub enum VokraError {
    /// An I/O operation failed (file open / read / metadata, ...).
    Io(std::io::Error),
    /// A model file could not be loaded or parsed.
    ModelLoad(String),
    /// The graph contains an op the selected backend does not support.
    ///
    /// Per FR-EX-08 (permanent constraint) this is an *explicit* error:
    /// Vokra never silently falls back to the CPU backend by default.
    UnsupportedOp(String),
    /// The requested backend is not available in this build / on this host.
    BackendUnavailable(String),
    /// A caller-supplied argument is invalid.
    InvalidArgument(String),
    /// An [`AudioGraph`](crate::AudioGraph) failed validation.
    GraphValidation(String),
    /// The model's `frontend_spec` (`vokra.frontend.*`) does not match the
    /// runtime's front-end bit-for-bit (FR-LD-03, M1-03).
    ///
    /// Raised at model load under [`FrontendPolicy::Fail`](crate::FrontendPolicy)
    /// when the declared feature-extraction parameters differ from what the
    /// consuming model actually computes; the message lists the differing
    /// fields. A distinct variant (rather than reusing [`Self::ModelLoad`]) so
    /// callers can special-case a front-end mismatch — e.g. downgrade to a
    /// warning — without string-matching.
    FrontendMismatch(String),
    /// A model's **weight** license is non-commercial (CC-BY-NC / CC-BY-NC-SA)
    /// or of unknown provenance, and no research flag was set to permit it
    /// (FR-CP-03, M2-13). The weight-license gate raises this *explicitly*
    /// rather than silently skipping or substituting the model — the same
    /// "never a silent fallback" rule as [`Self::UnsupportedOp`]. Distinct from
    /// [`Self::ModelLoad`] so callers can special-case it (e.g. surface the
    /// unlock hint) without string-matching. See
    /// [`crate::compliance::check_weight_license`].
    ///
    /// This is the *weight* license gate, wholly separate from the dependency
    /// (crate) license gate (`cargo-deny`, NFR-LC-02/04).
    ResearchLicenseRequired {
        /// Best-effort model identifier (from provenance / registry).
        model_id: String,
        /// The detected weight license label (raw string or class name),
        /// e.g. `"CC-BY-NC-4.0"`.
        license: String,
        /// Human-readable unlock hint: the research-flag routes and, where
        /// known, commercial alternatives.
        hint: String,
    },
    /// The API shape exists but its implementation has not landed yet
    /// (M0 skeleton; see the per-method rustdoc for the WP that wires it).
    NotImplemented(&'static str),
}

impl fmt::Display for VokraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::ModelLoad(msg) => write!(f, "model load error: {msg}"),
            Self::UnsupportedOp(msg) => write!(f, "unsupported op: {msg}"),
            Self::BackendUnavailable(msg) => write!(f, "backend unavailable: {msg}"),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::GraphValidation(msg) => write!(f, "graph validation error: {msg}"),
            Self::FrontendMismatch(msg) => write!(f, "frontend_spec mismatch: {msg}"),
            Self::ResearchLicenseRequired {
                model_id,
                license,
                hint,
            } => write!(
                f,
                "research license required: model `{model_id}` has non-commercial/unknown \
                 weight license `{license}` — {hint}"
            ),
            Self::NotImplemented(what) => write!(f, "not implemented (M0 skeleton): {what}"),
        }
    }
}

impl std::error::Error for VokraError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for VokraError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Convenience alias used across the Vokra API surface (FR-API-02).
pub type Result<T> = std::result::Result<T, VokraError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn display_strings_are_stable() {
        assert_eq!(
            VokraError::ModelLoad("bad magic".to_owned()).to_string(),
            "model load error: bad magic"
        );
        assert_eq!(
            VokraError::UnsupportedOp("Softmax on gpu".to_owned()).to_string(),
            "unsupported op: Softmax on gpu"
        );
        assert_eq!(
            VokraError::NotImplemented("wired in M0-06").to_string(),
            "not implemented (M0 skeleton): wired in M0-06"
        );
        assert_eq!(
            VokraError::FrontendMismatch("htk_mode: model=true runtime=false".to_owned())
                .to_string(),
            "frontend_spec mismatch: htk_mode: model=true runtime=false"
        );
        let gated = VokraError::ResearchLicenseRequired {
            model_id: "f5-tts".to_owned(),
            license: "CC-BY-NC-4.0".to_owned(),
            hint: "set ComplianceLevel::Research".to_owned(),
        }
        .to_string();
        assert!(gated.starts_with("research license required: model `f5-tts`"));
        assert!(gated.contains("CC-BY-NC-4.0"));
    }

    #[test]
    fn research_license_required_has_no_source() {
        // Not an I/O error, so it exposes no source (source() `_ => None`).
        assert!(
            VokraError::ResearchLicenseRequired {
                model_id: "m".to_owned(),
                license: "l".to_owned(),
                hint: "h".to_owned(),
            }
            .source()
            .is_none()
        );
    }

    #[test]
    fn io_error_converts_and_chains_source() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        let err: VokraError = io.into();
        assert!(matches!(err, VokraError::Io(_)));
        let source = err.source().expect("Io variant must expose a source");
        assert_eq!(source.to_string(), "gone");
    }

    #[test]
    fn non_io_variants_have_no_source() {
        assert!(
            VokraError::InvalidArgument("x".to_owned())
                .source()
                .is_none()
        );
    }

    #[test]
    fn result_alias_is_usable() {
        fn f(ok: bool) -> Result<u32> {
            if ok {
                Ok(7)
            } else {
                Err(VokraError::NotImplemented("f"))
            }
        }
        assert_eq!(f(true).unwrap(), 7);
        assert!(f(false).is_err());
    }
}
