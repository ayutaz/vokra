//! `MinDtypeRegistry` (T08) — minimum activation dtype per audio op.
//!
//! The registry pairs an **op-kind identifier** (a `&'static str`, not an
//! [`OpKind`](crate::ir::OpKind) variant — the mechanism-先行, kernel-後追い
//! idiom from `crates/vokra-core/src/ir/fusion/patterns/snake.rs`) with a
//! minimum activation dtype and a downgrade policy. It exists so
//! [`quant::validate`](crate::quant) can reject a policy that would drive any
//! of the FR-OP-10 / FR-OP-11 / FR-OP-12 / FR-OP-13 ops below their allowed
//! precision **before** hitting a backend (FR-EX-08: no silent widen).
//!
//! # Built-in entries
//!
//! - [`HIFIGAN_GENERATOR_OP`] (FR-OP-10): `min = Fp16`, `downgrade = HifiganOptIn`.
//! - [`BIGVGAN_GENERATOR_OP`] (FR-OP-11): `min = Fp16`, `downgrade = Forbidden`.
//! - [`VOCOS_HEAD_OP`]         (FR-OP-12): `min = Fp16`, `downgrade = Forbidden`.
//! - [`SNAKE_ACTIVATION_OP`]   (FR-OP-13): `min = Fp32`, `downgrade = Forbidden`
//!   (audit anchor; hard enforcement of `internal_precision` lives with the
//!   co-delivered kernel, per `crates/vokra-core/src/ir/fusion/patterns/snake.rs`).
//!
//! # Kokoro is *not* registered as `vocos_head`
//!
//! Kokoro's decoder is iSTFTNet / StyleTTS 2 派生 (CLAUDE.md レビュアー A
//! 修正), not a Vocos head. Registering Kokoro under [`VOCOS_HEAD_OP`] would
//! force fp16 minimum on every Kokoro config and break legitimate INT8
//! experiments once the M2-07 Kokoro decoder lands. The identifier
//! [`KOKORO_ISTFT_HEAD_OP`] is reserved for that follow-up but deliberately
//! **not** inserted into [`MinDtypeRegistry::builtin`] — see the test
//! `kokoro_istft_head_is_reserved_but_unregistered` below.
//!
//! # Zero-dep invariant
//!
//! Std-only: no serde / no toml. Op-kind identifiers are `&'static str`
//! constants (same pattern as `crate::compliance::ENV_ALLOW_RESEARCH_LICENSE`
//! and `ir::fusion::patterns::snake`) so downstream crates (`vokra-convert`,
//! `vokra-models`) can reference them without depending on an enum vocabulary
//! that would grow every time a vocoder op is added.

use crate::quant::scheme::ActivationDtype;

/// FR-OP-10 op-kind identifier — HiFi-GAN generator (Vocoder).
pub const HIFIGAN_GENERATOR_OP: &str = "hifigan_generator";

/// FR-OP-11 op-kind identifier — BigVGAN generator (Vocoder).
pub const BIGVGAN_GENERATOR_OP: &str = "bigvgan_generator";

/// FR-OP-12 op-kind identifier — Vocos iSTFT head.
pub const VOCOS_HEAD_OP: &str = "vocos_head";

/// FR-OP-13 op-kind identifier — Snake activation.
pub const SNAKE_ACTIVATION_OP: &str = "snake_activation";

/// Reserved op-kind identifier for Kokoro's iSTFTNet / StyleTTS 2 派生 decoder
/// head (M2-07). **Not** inserted into [`MinDtypeRegistry::builtin`] — Kokoro
/// is not a Vocos head (CLAUDE.md レビュアー A 修正). Consumers that need to
/// reason about the Kokoro head should reference this identifier so a future
/// registry entry lands on a stable name.
pub const KOKORO_ISTFT_HEAD_OP: &str = "kokoro_istft_head";

/// Minimum activation dtype a registered op tolerates.
///
/// Values are ordered from most-restrictive to least-restrictive:
/// [`Self::Fp32`] rejects both fp16 and int8 activation paths; [`Self::Fp16`]
/// rejects only int8. The concrete comparison lives with the validator
/// (T09 / c09); [`ActivationDtype`] is imported here so callers can wire the
/// check without a second dtype vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MinDtype {
    /// FP16 (or FP32) activations required. INT8 rejected.
    Fp16,
    /// FP32 activations required. Both FP16 and INT8 rejected.
    Fp32,
}

impl MinDtype {
    /// Whether `actual` satisfies this minimum.
    ///
    /// - `Fp16` accepts `F32` and `F16`, rejects `Int8`.
    /// - `Fp32` accepts only `F32`, rejects `F16` and `Int8`.
    pub fn is_satisfied_by(&self, actual: ActivationDtype) -> bool {
        match (self, actual) {
            (Self::Fp16, ActivationDtype::F32 | ActivationDtype::F16) => true,
            (Self::Fp16, ActivationDtype::Int8) => false,
            (Self::Fp32, ActivationDtype::F32) => true,
            (Self::Fp32, ActivationDtype::F16 | ActivationDtype::Int8) => false,
        }
    }
}

/// What the validator should do when a policy asks for a dtype below the
/// registered minimum.
///
/// - [`Self::Forbidden`] — reject unconditionally (Vocos, BigVGAN, and Snake's
///   audit entry). No flag flips this.
/// - [`Self::HifiganOptIn`] — reject **unless** the policy has
///   `hifigan_int8_opt_in = true` *and* an attached calibration reference
///   (see `QuantPolicy::with_hifigan_int8_opt_in`). The T12 eval verify
///   (NFR-QL-02 5% gate) still runs at deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DowngradePolicy {
    /// No downgrade allowed under any flag combination.
    Forbidden,
    /// Downgrade to INT8 allowed only through the HiFi-GAN opt-in path
    /// (T10 / c10).
    HifiganOptIn,
}

/// A single registry entry: op-kind identifier + minimum activation dtype +
/// downgrade policy + FR-* audit reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MinDtypeEntry {
    /// Op-kind identifier this entry constrains. Matches the string a policy
    /// / IR walk reports for the op (mechanism-先行, kernel-後追い).
    pub op_name: &'static str,
    /// Minimum activation dtype the op tolerates.
    pub min_activation: MinDtype,
    /// What the validator does when the resolved scheme falls below `min_activation`.
    pub downgrade: DowngradePolicy,
    /// FR-* identifier this entry anchors, used in error messages so the
    /// audit trail is unambiguous.
    pub fr_ref: &'static str,
}

/// Registry of minimum activation dtypes for audio ops.
///
/// [`Self::builtin`] returns the M2-08 built-in set (FR-OP-10 / FR-OP-11 /
/// FR-OP-12 / FR-OP-13). The struct is intentionally simple (linear-scan
/// [`Vec`]) — the built-in set is small (four entries) and lookups happen at
/// converter / session-ctor time, not on the hot path.
#[derive(Debug, Clone)]
pub struct MinDtypeRegistry {
    entries: Vec<MinDtypeEntry>,
}

impl MinDtypeRegistry {
    /// M2-08 built-in registry: FR-OP-10 hifigan, FR-OP-11 bigvgan,
    /// FR-OP-12 vocos_head, FR-OP-13 snake_activation.
    ///
    /// Kokoro's iSTFTNet head ([`KOKORO_ISTFT_HEAD_OP`]) is deliberately
    /// **not** registered — see module doc.
    pub fn builtin() -> Self {
        Self {
            entries: vec![
                MinDtypeEntry {
                    op_name: HIFIGAN_GENERATOR_OP,
                    min_activation: MinDtype::Fp16,
                    downgrade: DowngradePolicy::HifiganOptIn,
                    fr_ref: "FR-OP-10",
                },
                MinDtypeEntry {
                    op_name: BIGVGAN_GENERATOR_OP,
                    min_activation: MinDtype::Fp16,
                    downgrade: DowngradePolicy::Forbidden,
                    fr_ref: "FR-OP-11",
                },
                MinDtypeEntry {
                    op_name: VOCOS_HEAD_OP,
                    min_activation: MinDtype::Fp16,
                    downgrade: DowngradePolicy::Forbidden,
                    fr_ref: "FR-OP-12",
                },
                MinDtypeEntry {
                    op_name: SNAKE_ACTIVATION_OP,
                    min_activation: MinDtype::Fp32,
                    downgrade: DowngradePolicy::Forbidden,
                    fr_ref: "FR-OP-13",
                },
            ],
        }
    }

    /// Empty registry — for tests and downstream crates that want to
    /// hand-build a set without inheriting the built-ins.
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Look up the constraint for `op_name`. Returns [`None`] when the op is
    /// unregistered (the validator treats unregistered ops as unconstrained —
    /// they can accept any scheme the policy resolves to).
    pub fn lookup(&self, op_name: &str) -> Option<&MinDtypeEntry> {
        self.entries.iter().find(|e| e.op_name == op_name)
    }

    /// All entries in the registry (iteration order matches insertion).
    pub fn entries(&self) -> &[MinDtypeEntry] {
        &self.entries
    }
}

impl Default for MinDtypeRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_contains_fr_op_10_hifigan() {
        let reg = MinDtypeRegistry::builtin();
        let entry = reg
            .lookup(HIFIGAN_GENERATOR_OP)
            .expect("hifigan registered");
        assert_eq!(entry.op_name, "hifigan_generator");
        assert_eq!(entry.min_activation, MinDtype::Fp16);
        assert_eq!(entry.downgrade, DowngradePolicy::HifiganOptIn);
        assert_eq!(entry.fr_ref, "FR-OP-10");
    }

    #[test]
    fn builtin_contains_fr_op_11_bigvgan() {
        let reg = MinDtypeRegistry::builtin();
        let entry = reg
            .lookup(BIGVGAN_GENERATOR_OP)
            .expect("bigvgan registered");
        assert_eq!(entry.op_name, "bigvgan_generator");
        assert_eq!(entry.min_activation, MinDtype::Fp16);
        assert_eq!(entry.downgrade, DowngradePolicy::Forbidden);
        assert_eq!(entry.fr_ref, "FR-OP-11");
    }

    #[test]
    fn builtin_contains_fr_op_12_vocos_head() {
        let reg = MinDtypeRegistry::builtin();
        let entry = reg.lookup(VOCOS_HEAD_OP).expect("vocos_head registered");
        assert_eq!(entry.op_name, "vocos_head");
        assert_eq!(entry.min_activation, MinDtype::Fp16);
        assert_eq!(entry.downgrade, DowngradePolicy::Forbidden);
        assert_eq!(entry.fr_ref, "FR-OP-12");
    }

    #[test]
    fn builtin_contains_fr_op_13_snake_activation() {
        let reg = MinDtypeRegistry::builtin();
        let entry = reg
            .lookup(SNAKE_ACTIVATION_OP)
            .expect("snake_activation registered");
        assert_eq!(entry.op_name, "snake_activation");
        assert_eq!(entry.min_activation, MinDtype::Fp32);
        assert_eq!(entry.downgrade, DowngradePolicy::Forbidden);
        assert_eq!(entry.fr_ref, "FR-OP-13");
    }

    #[test]
    fn builtin_has_exactly_four_entries() {
        // The built-in set is M2-08's FR-OP-10 / FR-OP-11 / FR-OP-12 /
        // FR-OP-13 anchor set — nothing more. A change to this count is a
        // deliberate scope decision that needs a code review, not a silent
        // registry edit.
        let reg = MinDtypeRegistry::builtin();
        assert_eq!(reg.entries().len(), 4);
    }

    #[test]
    fn unknown_op_returns_none() {
        let reg = MinDtypeRegistry::builtin();
        assert!(reg.lookup("matmul").is_none());
        assert!(reg.lookup("").is_none());
        assert!(reg.lookup("HifiGAN_Generator").is_none()); // case-sensitive
    }

    #[test]
    fn kokoro_istft_head_is_reserved_but_unregistered() {
        // CLAUDE.md レビュアー A 修正 — Kokoro is iSTFTNet / StyleTTS 2 派生,
        // not a Vocos head. The identifier is reserved so a future M2-07
        // registry entry lands on a stable name, but M2-08's built-in
        // set must not include it — see module doc.
        let reg = MinDtypeRegistry::builtin();
        assert!(
            reg.lookup(KOKORO_ISTFT_HEAD_OP).is_none(),
            "Kokoro decoder must not be auto-registered under vocos_head; see module doc"
        );
        // But the constant itself is available for downstream reference.
        assert_eq!(KOKORO_ISTFT_HEAD_OP, "kokoro_istft_head");
    }

    #[test]
    fn empty_registry_looks_up_nothing() {
        let reg = MinDtypeRegistry::empty();
        assert!(reg.lookup(HIFIGAN_GENERATOR_OP).is_none());
        assert!(reg.lookup(VOCOS_HEAD_OP).is_none());
        assert_eq!(reg.entries().len(), 0);
    }

    #[test]
    fn default_matches_builtin() {
        let default_reg = MinDtypeRegistry::default();
        let builtin = MinDtypeRegistry::builtin();
        assert_eq!(default_reg.entries().len(), builtin.entries().len());
        for entry in builtin.entries() {
            let matched = default_reg
                .lookup(entry.op_name)
                .expect("default has builtin entry");
            assert_eq!(matched, entry);
        }
    }

    #[test]
    fn min_dtype_fp16_accepts_f32_and_f16_rejects_int8() {
        assert!(MinDtype::Fp16.is_satisfied_by(ActivationDtype::F32));
        assert!(MinDtype::Fp16.is_satisfied_by(ActivationDtype::F16));
        assert!(!MinDtype::Fp16.is_satisfied_by(ActivationDtype::Int8));
    }

    #[test]
    fn min_dtype_fp32_rejects_f16_and_int8() {
        assert!(MinDtype::Fp32.is_satisfied_by(ActivationDtype::F32));
        assert!(!MinDtype::Fp32.is_satisfied_by(ActivationDtype::F16));
        assert!(!MinDtype::Fp32.is_satisfied_by(ActivationDtype::Int8));
    }
}
