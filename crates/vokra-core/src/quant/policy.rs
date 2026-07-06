//! `QuantPolicy` builder + rule table (T03) plus the HiFi-GAN INT8 opt-in
//! gate (T10).
//!
//! The public shape is deliberately narrow:
//!
//! - [`QuantPolicy`] is a value type built via a chainable builder
//!   ([`QuantPolicy::new`], [`Self::with_rule`],
//!   [`Self::with_hifigan_int8_opt_in`]). No serde, no config parser
//!   (NFR-DS-02) — policies are Rust literals in application code, or
//!   read from the `vokra.quant.*` GGUF chunk (T05, separate WP).
//! - Rules are ordered and evaluated first-match-wins by
//!   [`crate::quant::resolve::resolve`] (T04). Order within the list is
//!   authoritative; the resolve helper additionally biases by pattern kind
//!   (Exact > Prefix > Suffix > Glob) at equal priority.
//! - HiFi-GAN INT8 (T10): the `opt_in` bool and the [`CalibrationRef`] are
//!   *private* and settable only together via
//!   [`Self::with_hifigan_int8_opt_in`], so the "opt-in without calibration"
//!   state is unrepresentable. Vocos / BigVGAN (registry
//!   `DowngradePolicy::Forbidden`) are rejected regardless of this flag —
//!   that's a validate-time concern (T09).

use crate::error::{Result, VokraError};
use crate::quant::scheme::QuantScheme;

/// Layer / tensor name matcher used by [`QuantRule`].
///
/// [`LayerPattern::Glob`] uses a hand-rolled matcher restricted to `*`
/// (matches any run of characters, including empty). No regex, no `?`,
/// no character classes — keeps the zero-dep invariant intact and matches
/// tensor names from `torch.nn.Module.named_parameters()` which are
/// dotted paths without regex metacharacters.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LayerPattern {
    /// Exact tensor name match, e.g. `"encoder.blocks.0.mlp.0.weight"`.
    Exact(String),
    /// Prefix match, e.g. `"encoder.blocks."`.
    Prefix(String),
    /// Suffix match, e.g. `".bias"` — biases/norms exception idiom.
    Suffix(String),
    /// Glob with `*` wildcards, e.g. `"encoder.blocks.*.attn.*"`. Literal
    /// segments separated by `*`; two adjacent `*` collapse to one.
    Glob(String),
}

impl LayerPattern {
    /// Returns `true` if `name` matches this pattern.
    ///
    /// Used by [`crate::quant::resolve::resolve`] (T04). Exposed on the
    /// pattern itself so unit tests can pin behavior per variant without
    /// materialising a full policy.
    pub fn matches(&self, name: &str) -> bool {
        match self {
            Self::Exact(p) => name == p,
            Self::Prefix(p) => name.starts_with(p.as_str()),
            Self::Suffix(p) => name.ends_with(p.as_str()),
            Self::Glob(p) => glob_matches(p, name),
        }
    }

    /// Priority tier used by the resolver's tie-break rule
    /// (Exact > Prefix > Suffix > Glob). Lower is more specific.
    pub(crate) fn priority(&self) -> u8 {
        match self {
            Self::Exact(_) => 0,
            Self::Prefix(_) => 1,
            Self::Suffix(_) => 2,
            Self::Glob(_) => 3,
        }
    }

    /// Canonical kind tag used by the `vokra.quant.rule.{i}.pattern_kind`
    /// chunk key (T05). Round-trips with [`Self::from_kind_and_pattern`].
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::Exact(_) => "exact",
            Self::Prefix(_) => "prefix",
            Self::Suffix(_) => "suffix",
            Self::Glob(_) => "glob",
        }
    }

    /// Raw pattern payload (variant-independent). Used by the T05 chunk
    /// writer to persist the pattern body next to its kind tag.
    pub fn pattern_str(&self) -> &str {
        match self {
            Self::Exact(p) | Self::Prefix(p) | Self::Suffix(p) | Self::Glob(p) => p.as_str(),
        }
    }
}

/// Hand-rolled `*`-only glob matcher (zero-dep).
///
/// - `*` matches any (possibly empty) run of characters.
/// - Every other char is a literal.
/// - `**` collapses to `*` (two-star is not a distinct operator).
fn glob_matches(pattern: &str, name: &str) -> bool {
    // Split pattern on `*`; every non-first segment must appear in-order after
    // the previous match, and the first/last segments anchor the ends unless
    // the pattern starts/ends with `*`.
    let raw_segments: Vec<&str> = pattern.split('*').collect();
    // Collapse runs of `*` (empty segments between them). Keep a single empty
    // segment when the pattern is all-`*` so we can distinguish "match
    // anything" from "no segments at all".
    let segments: Vec<&str> = if raw_segments.iter().all(|s| s.is_empty()) {
        Vec::new()
    } else {
        raw_segments.into_iter().filter(|s| !s.is_empty()).collect()
    };
    // Re-derive anchoring flags after retain: reparse original ends since
    // retain may drop anchoring info.
    let leading_star = pattern.starts_with('*');
    let trailing_star = pattern.ends_with('*');

    if segments.is_empty() {
        // Pattern was all `*`s → matches anything.
        return true;
    }

    let mut cursor = 0usize;
    let n = segments.len();
    for (i, seg) in segments.iter().enumerate() {
        let first = i == 0;
        let last = i == n - 1;
        let anchor_start = first && !leading_star;
        let anchor_end = last && !trailing_star;
        if anchor_start && anchor_end {
            // Single un-starred segment must equal the whole remainder.
            if &name[cursor..] != *seg {
                return false;
            }
            cursor = name.len();
        } else if anchor_start {
            if !name[cursor..].starts_with(*seg) {
                return false;
            }
            cursor += seg.len();
        } else if anchor_end {
            if let Some(pos) = name[cursor..].rfind(*seg) {
                if cursor + pos + seg.len() != name.len() {
                    return false;
                }
                cursor += pos + seg.len();
            } else {
                return false;
            }
        } else if let Some(pos) = name[cursor..].find(*seg) {
            cursor += pos + seg.len();
        } else {
            return false;
        }
    }
    true
}

/// One rule in a [`QuantPolicy`]: `pattern → scheme`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuantRule {
    /// Layer / tensor name matcher.
    pub pattern: LayerPattern,
    /// Scheme to apply when the pattern matches.
    pub scheme: QuantScheme,
}

/// Opaque reference to an INT8 calibration blob.
///
/// Stored on-disk under `vokra.quant.hifigan_int8_calibration_ref` (T05).
/// M2-08 leaves the blob storage format opaque — validate (T09) only checks
/// that a reference exists when the opt-in bool is `true`, which is
/// guaranteed at construction time via
/// [`QuantPolicy::with_hifigan_int8_opt_in`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CalibrationRef {
    /// Opaque handle string.
    pub handle: String,
}

impl CalibrationRef {
    /// Construct a calibration reference from an opaque handle string.
    pub fn new(handle: impl Into<String>) -> Self {
        Self {
            handle: handle.into(),
        }
    }
}

/// Quantization policy — default scheme + ordered rule table + HiFi-GAN
/// INT8 opt-in gate.
///
/// See the module-level doc for the design contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantPolicy {
    default: QuantScheme,
    rules: Vec<QuantRule>,
    /// **Private on purpose (T10)**: flipped only together with
    /// `hifigan_int8_calibration` via [`Self::with_hifigan_int8_opt_in`]. No
    /// public setter, so "opt-in without calibration" is unrepresentable by
    /// construction.
    hifigan_int8_opt_in: bool,
    hifigan_int8_calibration: Option<CalibrationRef>,
}

impl QuantPolicy {
    /// Build a policy with `default` as the fall-through scheme and no rules.
    pub fn new(default: QuantScheme) -> Self {
        Self {
            default,
            rules: Vec::new(),
            hifigan_int8_opt_in: false,
            hifigan_int8_calibration: None,
        }
    }

    /// Append a rule to the ordered rule list.
    ///
    /// Order matters when the resolver's Exact > Prefix > Suffix > Glob
    /// tie-break can't decide (e.g. two `Prefix` rules of the same length):
    /// earlier rules win. Callers wanting deterministic layering should
    /// register more-specific patterns first.
    pub fn with_rule(mut self, pattern: LayerPattern, scheme: QuantScheme) -> Self {
        self.rules.push(QuantRule { pattern, scheme });
        self
    }

    /// **Sole atomic path (T10)** for enabling HiFi-GAN INT8:
    ///
    /// Flips the internal `hifigan_int8_opt_in` bool *and* attaches the
    /// caller-supplied [`CalibrationRef`] in one call. There is no separate
    /// setter for either field, so a policy can never carry
    /// `opt_in=true` with `calibration=None`.
    ///
    /// Vocos / BigVGAN remain rejected regardless of this flag — that's a
    /// validate-time concern (registry `DowngradePolicy::Forbidden`, T09).
    pub fn with_hifigan_int8_opt_in(mut self, calibration: CalibrationRef) -> Self {
        self.hifigan_int8_opt_in = true;
        self.hifigan_int8_calibration = Some(calibration);
        self
    }

    /// Read-only accessor: default scheme.
    pub fn default_scheme(&self) -> QuantScheme {
        self.default
    }

    /// Read-only accessor: ordered rule slice.
    pub fn rules(&self) -> &[QuantRule] {
        &self.rules
    }

    /// Read-only accessor: HiFi-GAN INT8 opt-in flag.
    pub fn hifigan_int8_opt_in(&self) -> bool {
        self.hifigan_int8_opt_in
    }

    /// Read-only accessor: HiFi-GAN INT8 calibration reference (present iff
    /// [`Self::hifigan_int8_opt_in`] is `true`).
    pub fn hifigan_int8_calibration(&self) -> Option<&CalibrationRef> {
        self.hifigan_int8_calibration.as_ref()
    }

    /// Cross-field self-check for construction-time invariants.
    ///
    /// Returns `Err(VokraError::InvalidArgument)` if the policy is internally
    /// inconsistent. Cross-model checks (e.g. whether an INT8 default is
    /// compatible with the ops in a specific model) live in T09
    /// (`validate_policy_against_model`), not here.
    pub fn validate_self(&self) -> Result<()> {
        // The opt-in / calibration pair is enforced by construction (no
        // public setter for opt-in alone), but validate anyway so we catch
        // any future accessor that violates the invariant.
        if self.hifigan_int8_opt_in && self.hifigan_int8_calibration.is_none() {
            return Err(VokraError::InvalidArgument(
                "hifigan_int8_opt_in=true requires a CalibrationRef; use \
                 QuantPolicy::with_hifigan_int8_opt_in(calibration) — see T10"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

// -- Presets (T04 landing pad) ---------------------------------------------
//
// The `whisper_q4_k` and `default_vocoder_safe` presets live in `resolve.rs`
// so they can be tested together with the resolver behavior they encode
// (bias/weight_norm exceptions to Fp32).

#[cfg(test)]
mod tests {
    use super::*;

    // ----- LayerPattern ---------------------------------------------------

    #[test]
    fn exact_pattern_matches_only_exact() {
        let p = LayerPattern::Exact("encoder.blocks.0.mlp.0.weight".to_owned());
        assert!(p.matches("encoder.blocks.0.mlp.0.weight"));
        assert!(!p.matches("encoder.blocks.0.mlp.0.weight2"));
        assert!(!p.matches("encoder.blocks.0.mlp.0.weigh"));
    }

    #[test]
    fn prefix_pattern() {
        let p = LayerPattern::Prefix("encoder.blocks.".to_owned());
        assert!(p.matches("encoder.blocks.0.mlp.0.weight"));
        assert!(p.matches("encoder.blocks."));
        assert!(!p.matches("decoder.blocks.0.mlp.0.weight"));
    }

    #[test]
    fn suffix_pattern() {
        let p = LayerPattern::Suffix(".bias".to_owned());
        assert!(p.matches("encoder.blocks.0.mlp.0.bias"));
        assert!(!p.matches("encoder.blocks.0.mlp.0.weight"));
        // Empty suffix matches everything (edge case — documented).
        assert!(LayerPattern::Suffix(String::new()).matches("anything"));
    }

    #[test]
    fn glob_middle_wildcard() {
        let p = LayerPattern::Glob("encoder.blocks.*.attn.*".to_owned());
        assert!(p.matches("encoder.blocks.0.attn.qkv"));
        assert!(p.matches("encoder.blocks.11.attn.out.weight"));
        assert!(!p.matches("encoder.blocks.0.mlp.0.weight"));
        assert!(!p.matches("decoder.blocks.0.attn.qkv"));
    }

    #[test]
    fn glob_leading_and_trailing() {
        let p = LayerPattern::Glob("*.weight".to_owned());
        assert!(p.matches("encoder.blocks.0.mlp.0.weight"));
        assert!(!p.matches("encoder.blocks.0.mlp.0.bias"));

        let p2 = LayerPattern::Glob("encoder.*".to_owned());
        assert!(p2.matches("encoder.blocks.0"));
        assert!(!p2.matches("decoder.blocks.0"));

        // "*" alone matches anything.
        assert!(LayerPattern::Glob("*".to_owned()).matches(""));
        assert!(LayerPattern::Glob("*".to_owned()).matches("anything"));

        // "**" collapses to "*".
        assert!(LayerPattern::Glob("**".to_owned()).matches("anything"));
    }

    #[test]
    fn glob_no_wildcard_is_exact_semantics() {
        // A glob with no `*` should behave like an anchored exact match.
        let p = LayerPattern::Glob("encoder.weight".to_owned());
        assert!(p.matches("encoder.weight"));
        assert!(!p.matches("encoder.weight2"));
        assert!(!p.matches("xencoder.weight"));
    }

    #[test]
    fn pattern_priority_order() {
        assert!(
            LayerPattern::Exact("a".to_owned()).priority()
                < LayerPattern::Prefix("a".to_owned()).priority()
        );
        assert!(
            LayerPattern::Prefix("a".to_owned()).priority()
                < LayerPattern::Suffix("a".to_owned()).priority()
        );
        assert!(
            LayerPattern::Suffix("a".to_owned()).priority()
                < LayerPattern::Glob("a".to_owned()).priority()
        );
    }

    // ----- QuantPolicy builder -------------------------------------------

    #[test]
    fn builder_records_default_and_rules_in_order() {
        let policy = QuantPolicy::new(QuantScheme::W4A16Q4K)
            .with_rule(LayerPattern::Suffix(".bias".to_owned()), QuantScheme::Fp32)
            .with_rule(
                LayerPattern::Prefix("encoder.".to_owned()),
                QuantScheme::Fp16,
            );

        assert_eq!(policy.default_scheme(), QuantScheme::W4A16Q4K);
        assert_eq!(policy.rules().len(), 2);
        assert_eq!(policy.rules()[0].scheme, QuantScheme::Fp32);
        assert_eq!(policy.rules()[1].scheme, QuantScheme::Fp16);
    }

    #[test]
    fn builder_default_has_no_hifigan_opt_in() {
        let policy = QuantPolicy::new(QuantScheme::Fp16);
        assert!(!policy.hifigan_int8_opt_in());
        assert!(policy.hifigan_int8_calibration().is_none());
        policy.validate_self().unwrap();
    }

    // ----- T10: HiFi-GAN INT8 opt-in gate --------------------------------

    #[test]
    fn hifigan_opt_in_sets_both_fields_atomically() {
        let cal = CalibrationRef::new("hifigan-int8-cal-v1");
        let policy = QuantPolicy::new(QuantScheme::Fp16).with_hifigan_int8_opt_in(cal.clone());

        assert!(policy.hifigan_int8_opt_in());
        assert_eq!(policy.hifigan_int8_calibration(), Some(&cal));
        policy.validate_self().unwrap();
    }

    #[test]
    fn hifigan_opt_in_is_the_only_atomic_path() {
        // Structural check (compile-time via the type): there is no public
        // setter that toggles `opt_in` without a calibration. If a future
        // refactor adds one, `validate_self` also catches the inconsistent
        // state at runtime. Exercise that fallback path explicitly by
        // rebuilding the value with the invariant broken via a clone of the
        // struct fields — which requires field access we do not expose, so
        // the only way to break it is through a future breaking change,
        // which this test protects against by pinning the accessor shape.
        let policy = QuantPolicy::new(QuantScheme::W8A8Int8);
        // No public field access for `hifigan_int8_opt_in` — accessor is
        // read-only. Confirm.
        assert!(!policy.hifigan_int8_opt_in());
        assert!(policy.hifigan_int8_calibration().is_none());
    }

    #[test]
    fn validate_self_rejects_opt_in_without_calibration() {
        // Only reachable via a struct-literal outside the module (which is
        // impossible because the fields are private). Simulate the corrupt
        // state by mutating through a private test helper: we don't expose
        // one, so instead we assert `validate_self` reports OK on the two
        // legitimate states.
        //
        // The invariant is: `opt_in <=> calibration.is_some()`.
        let no_opt = QuantPolicy::new(QuantScheme::Fp16);
        no_opt.validate_self().unwrap();
        let with_opt = QuantPolicy::new(QuantScheme::Fp16)
            .with_hifigan_int8_opt_in(CalibrationRef::new("cal"));
        with_opt.validate_self().unwrap();
    }

    // A hand-rolled test-only mutator confirms `validate_self` DOES flag the
    // corrupt state if some future accessor breaks the invariant. We build
    // the corrupt value via the private field access this test module has
    // (same crate, private siblings visible).
    #[test]
    fn validate_self_catches_manufactured_corruption() {
        let mut p = QuantPolicy::new(QuantScheme::Fp16);
        p.hifigan_int8_opt_in = true; // simulate a future bad setter
        p.hifigan_int8_calibration = None;
        assert!(matches!(
            p.validate_self(),
            Err(VokraError::InvalidArgument(_))
        ));
    }
}
