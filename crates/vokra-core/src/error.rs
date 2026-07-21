//! Error handling for all Vokra public APIs.
//!
//! FR-API-02 mandates that the public Rust API returns
//! [`Result<T, VokraError>`](crate::Result).

// M5-03-T04: no_std-first. `core::fmt` is `std::fmt` under std, so this is
// unchanged for the default build (NFR-PT-01). `String` is an `alloc` type, in
// the prelude only under std; the no_std subset imports it explicitly (this
// import is inert under std — the prelude already provides `String`).
#[cfg(not(feature = "std"))]
use alloc::string::String;
use core::fmt;

/// Error type returned by every Vokra public API (FR-API-02).
///
/// The enum is `#[non_exhaustive]`: variants will be added while the v0.1
/// spike progresses (M0-02-T06), so downstream matches must keep a wildcard
/// arm.
#[derive(Debug)]
#[non_exhaustive]
pub enum VokraError {
    /// An I/O operation failed (file open / read / metadata, ...).
    ///
    /// M5-03-T04: `std::io::Error` does not exist under `#![no_std]`, so this
    /// variant is present only in std builds. The no_std subset (Cortex-M55
    /// Tier 3) has no filesystem, so no code path constructs it; GGUF is loaded
    /// there from an in-memory / flash-mapped `&[u8]` via `GgufFile::parse` /
    /// `from_external`, never `GgufFile::open`.
    #[cfg(feature = "std")]
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
    /// A quantization scheme requests an activation dtype (or weight/backend
    /// combination) whose kernel path is not implemented in this build
    /// (M2-08 T07 / T09). Per FR-EX-08 (permanent constraint) this is an
    /// *explicit* error — Vokra never silently downgrades activation
    /// precision or falls back to a different backend. Distinct from
    /// [`Self::UnsupportedOp`] so callers can special-case a quantization
    /// mismatch (e.g. surface a policy edit hint) without string-matching.
    UnsupportedQuantPath {
        /// Op-kind identifier that would consume the activation
        /// (e.g. `"whisper::mlp"`, `"vocos_head"`), or a coarser scope
        /// (`"whisper"`) when the mismatch is model-wide.
        op: String,
        /// The requested [`QuantScheme`](crate::quant::QuantScheme) as its
        /// canonical alias string (e.g. `"w8a8"`).
        scheme: String,
        /// The [`BackendKind`](crate::BackendKind) that would have run the op.
        backend: String,
    },
    /// A GGUF `vokra.quant.*` chunk (or a caller of
    /// [`QuantScheme::from_alias_str`](crate::quant::QuantScheme::from_alias_str))
    /// named a quantization scheme the runtime does not recognize.
    ///
    /// This is the "future scheme alias" case: a converter built with a newer
    /// [`QuantScheme`](crate::quant::QuantScheme) variant may write an alias an
    /// older runtime cannot parse, and this surfaces gracefully instead of
    /// panicking (FR-QT-02, M2-08-T05).
    UnknownQuantScheme(String),
    /// A [`QuantPolicy`](crate::quant::QuantPolicy) resolves to a scheme whose
    /// activation dtype falls below the minimum an op registered in the
    /// [`MinDtypeRegistry`](crate::quant::MinDtypeRegistry) tolerates
    /// (M2-08-T09). Per FR-EX-08 (permanent constraint) this is an *explicit*
    /// error raised before any backend is invoked — Vokra never silently
    /// widens activation precision or drops an op to CPU.
    ///
    /// Distinct from [`Self::UnsupportedQuantPath`] so callers can special-case
    /// the "policy exceeded the op's minimum" case (e.g. surface the FR-OP-*
    /// audit reference so the operator knows *why* the reject fired) without
    /// string-matching.
    MinDtypeViolation {
        /// Op-kind identifier that raised the violation
        /// (e.g. `"vocos_head"`, `"hifigan_generator"`).
        op: String,
        /// The requested [`QuantScheme`](crate::quant::QuantScheme) as its
        /// canonical alias string (e.g. `"w8a8"`).
        requested_scheme: String,
        /// Minimum activation dtype the op tolerates (canonical string of the
        /// [`MinDtype`](crate::quant::MinDtype) variant — `"fp16"` or `"fp32"`).
        min_required: String,
        /// FR-* identifier this violation anchors, from the registry entry
        /// (e.g. `"FR-OP-10"`, `"FR-OP-12"`).
        fr_ref: String,
    },
    /// A [`QuantPolicy`](crate::quant::QuantPolicy) enables HiFi-GAN INT8
    /// (`hifigan_int8_opt_in = true`) AND the loaded model exercises the
    /// HiFi-GAN op, but no fresh
    /// [`DegradationReport`](crate::quant::DegradationReport) was attached at
    /// session / bench construction (M2-08-T12).
    ///
    /// Per FR-EX-08 (permanent constraint) this is an *explicit* error — a
    /// HiFi-GAN INT8 build cannot be run without an accompanying eval
    /// verification, so the runtime refuses to start rather than silently
    /// shipping an unverified INT8 vocoder. Distinct from
    /// [`Self::HifiganInt8DegradationExceeded`] so callers can special-case
    /// "verify not attempted" vs "verify attempted and failed" without
    /// string-matching.
    HifiganInt8VerifyMissing,
    /// A [`QuantPolicy`](crate::quant::QuantPolicy) enables HiFi-GAN INT8
    /// AND the loaded model exercises the HiFi-GAN op AND a fresh
    /// [`DegradationReport`](crate::quant::DegradationReport) *was* attached,
    /// but its MEL-loss relative delta exceeds the NFR-QL-02 5% gate
    /// (M2-08-T12).
    ///
    /// Contrast with [`Self::HifiganInt8VerifyMissing`] (no report attached).
    HifiganInt8DegradationExceeded {
        /// Observed relative MEL-loss delta (`(loss_quant - loss_ref) /
        /// max(loss_ref, ε)`) from the attached
        /// [`DegradationReport`](crate::quant::DegradationReport).
        delta: f64,
        /// The gate threshold the delta exceeded (NFR-QL-02: 0.05).
        threshold: f64,
    },
    /// A [`PagedKvCache`](crate::cache::paged::PagedKvCache) exhausted its
    /// pre-allocated page pool (M3-03).
    ///
    /// Per FR-EX-05 the paged cache must never invoke a system allocator on the
    /// hot path, so page acquisition is a strictly O(1) free-list pop against a
    /// bounded arena sized at session construction. A miss surfaces here rather
    /// than growing the arena, so the caller can either (a) size the session
    /// with a larger `max_time` / `n_stream` hint, or (b) reset the cache
    /// between segments. Distinct from
    /// [`Self::InvalidArgument`] so callers can special-case the exhaustion
    /// (e.g. surface the pre-allocate hint) without string-matching.
    KvCacheExhausted {
        /// How many pages the arena was sized for at construction.
        capacity: usize,
        /// The number of pages that were already in use when the miss fired.
        in_use: usize,
    },
}

impl fmt::Display for VokraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "std")]
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
            Self::UnsupportedQuantPath {
                op,
                scheme,
                backend,
            } => write!(
                f,
                "unsupported quant path: op `{op}` with scheme `{scheme}` on backend \
                 `{backend}` — no silent fallback (FR-EX-08)"
            ),
            Self::UnknownQuantScheme(alias) => {
                write!(f, "unknown quantization scheme alias `{alias}`")
            }
            Self::MinDtypeViolation {
                op,
                requested_scheme,
                min_required,
                fr_ref,
            } => write!(
                f,
                "min-dtype violation: op `{op}` requires activation >= `{min_required}` \
                 ({fr_ref}), but policy resolves to scheme `{requested_scheme}` — no silent \
                 widen (FR-EX-08)"
            ),
            Self::HifiganInt8VerifyMissing => write!(
                f,
                "hifigan int8 verify missing: policy enables hifigan_int8_opt_in but no \
                 fresh DegradationReport was attached — attach one via \
                 `check_degradation` before construction (M2-08-T12, NFR-QL-02, FR-EX-08)"
            ),
            Self::HifiganInt8DegradationExceeded { delta, threshold } => write!(
                f,
                "hifigan int8 degradation exceeded: relative MEL-loss delta {delta:.4} > \
                 threshold {threshold:.4} (M2-08-T12, NFR-QL-02)"
            ),
            Self::KvCacheExhausted { capacity, in_use } => write!(
                f,
                "paged KV cache exhausted: arena has {capacity} pages, {in_use} in use — \
                 pre-allocate a larger max_time/n_stream at session construction \
                 (M3-03, FR-EX-05)"
            ),
        }
    }
}

// M5-03-T04: `core::error::Error` is stable since Rust 1.81 (this workspace's
// floor is well above that) and `std::error::Error` is a re-export of it, so
// this impl is identical for the default build (NFR-PT-01) yet also holds under
// `#![no_std]`. `std::io::Error` implements `core::error::Error`, so the
// std-gated `Io` source still type-checks under std.
impl core::error::Error for VokraError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            #[cfg(feature = "std")]
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

// The `From<std::io::Error>` bridge exists only under std (the no_std subset has
// no `std::io`). Loud-by-construction: no_std GGUF loading never touches fs.
#[cfg(feature = "std")]
impl From<std::io::Error> for VokraError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Convenience alias used across the Vokra API surface (FR-API-02).
pub type Result<T> = core::result::Result<T, VokraError>;

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

    /// M5-03-T04: the no_std migration (`std::fmt` → `core::fmt`,
    /// `std::error::Error` → `core::error::Error`, `std::result::Result` →
    /// `core::result::Result`) must not change the std build's observable
    /// behavior. A non-`Io` variant's `Display` stays byte-stable, it exposes no
    /// `source()`, and the `Result` alias still resolves — proving the migration
    /// is behavior-preserving for the default (std) build (NFR-PT-01).
    #[test]
    fn nostd_migration_preserves_std_display_and_source() {
        let e = VokraError::UnsupportedOp("Conv1d on vulkan".to_owned());
        assert_eq!(e.to_string(), "unsupported op: Conv1d on vulkan");
        assert!(e.source().is_none());
        // The alias is `core::result::Result`, structurally identical to the
        // former `std::result::Result` for every caller (exercised through a
        // function boundary so it is a real value, not a literal).
        fn via_alias(v: u8) -> Result<u8> {
            Ok(v)
        }
        assert!(matches!(via_alias(3), Ok(3)));
    }
}
