//! M5-residual op **mechanism anchors** (M4-20 T14).
//!
//! M4-20 lands only the *trigger-backed* audio-op subset; the rest of the
//! catalogue has no live trigger model and would leave unused C ABI symbols in
//! the M5-13 (旧 M4-12) freeze surface semi-permanently
//! (`docs/m4-scope-expansion-2026-07-13.md` §BIG-10, ADR M4-20 §D-6). Landing
//! them **before** the freeze would violate the mechanism-先行・実体-後追い
//! discipline.
//!
//! This module records the M5-residual ops as reserved `&'static str` op-kind
//! identifiers — declared, so a future M5 landing lands on a stable name. It
//! generalizes the
//! [`KOKORO_ISTFT_HEAD_OP`](crate::quant::registry::KOKORO_ISTFT_HEAD_OP)
//! reserved-but-unregistered pattern to the M4-20 catalogue, and it pairs with
//! the `docs/abi-changelog.md` "Reserved additions" entry so an M5 op landing
//! is a backward-compatible additive.
//!
//! # What "reserved" is guaranteed to mean here (and how)
//!
//! The reservation spans three dimensions, but they are **not** enforced the
//! same way — two are actively checked, the third is a structural property of
//! the type system with nothing to assert. Spelling them apart keeps this doc
//! honest: the original wording ("not inserted into any registry / `OpKind` …
//! adding no C ABI symbol … guarded by a test") implied one test guarded all
//! three, which is not true and cannot be made true (M5-ORPHAN-SCOPE-T06; the
//! `OpKind` dimension has no runtime assertion target — see the ADR
//! `docs/adr/M5-ORPHAN-SCOPE-residual-ops-amx-sme.md` §(6)).
//!
//! - **Not in [`MinDtypeRegistry`](crate::quant::registry::MinDtypeRegistry)** —
//!   *checked by a test*. `tests::new_anchors_are_reserved_but_unregistered`
//!   asserts `reg.lookup(id).is_none()` for the six new anchors. BigVGAN is the
//!   documented exception: its min-dtype anchor **is** registered (M2-08), only
//!   the generator op landing is M5 — see
//!   `tests::bigvgan_min_dtype_anchor_is_registered_but_op_is_m5`.
//! - **Adds no C ABI symbol** — *checked by a machine gate*, not by a unit test
//!   in this crate. `scripts/check-m5-residual-no-abi.sh` asserts that none of
//!   these op-kind ids appears in the exported C ABI **symbol list**
//!   (`scripts/check-abi-changelog.sh --list` — the FUNC/TYPEDEF names, not the
//!   raw `include/vokra.h` text, which carries rustdoc comments that would
//!   false-positive). It runs in the `abi-surface` CI job next to the ABI
//!   changelog gate.
//! - **Not an [`OpKind`](crate::ir::OpKind) variant** — *structural; no test,
//!   and none is possible*. These ids are `&'static str`; `OpKind` is a
//!   `#[non_exhaustive]` enum with **no string-resolution path** — no
//!   `FromStr`, no `TryFrom<&str>`, no name lookup exists for it anywhere in
//!   the workspace (only `Node::op` returns `&OpKind`). There is therefore no
//!   mechanism by which a `&str` could ever "register" as a variant, and hence
//!   no runtime state to assert; a would-be test would be vacuous theater. If a
//!   string-resolution path is ever added to `OpKind`, add the non-resolution
//!   assertion at that point (T06 followup, recorded in the ADR §(6) and the
//!   integrated report).
//!
//! # Blockers (why each is M5-residual)
//!
//! **Read this column as "what is still reserved", not "what does not
//! exist".** Per ADR M4-20 §D-5 these primitives are deliberately *runtime
//! functions*, not [`OpKind`](crate::ir::OpKind) variants — so a landed
//! runtime implementation and a still-reserved graph-side id coexist by
//! design, and the reservation stays defensible after the primitive ships.
//! The first three rows below are exactly that case: their runtime functions
//! have landed — `rnnt_decode` even has a live consumer — and what remains
//! reserved is only the graph-side `OpKind` variant plus the C ABI export,
//! both deferred to the M5-13 freeze policy.
//!
//! A blocker that instead asserts a *pending trigger* or an *absent
//! implementation* is a factual claim about the tree, and it goes stale the
//! moment that thing lands — which is what happened to the three rows below
//! before 2026-08-15. `scripts/check-m5-residual-blockers.sh` now refuses an
//! absence-claim on any op whose `vokra-ops` module exists, so this specific
//! drift cannot silently return.
//!
//! | op-kind id                       | FR-OP    | blocker                                            |
//! | -------------------------------- | -------- | -------------------------------------------------- |
//! | [`BIGVGAN_GENERATOR_OP`]         | FR-OP-11 | runtime primitive, strict real-weight binder, alias-free forward, CLI mel contract, and independent waveform parity landed; the min-dtype anchor is registered (M2-08). Reserved: the graph-side `OpKind` variant + C ABI export |
//! | [`CTC_DECODE_OP`]                | FR-OP-41 | runtime primitives landed (`vokra_ops::ctc_decode_greedy` / `ctc_decode_beam`, incl. n-gram LM shallow fusion + hotword boost) and the NeMo family landed (`parakeet_ctc`, `canary`, `canary_qwen`, `canary_1b_flash`, `omniasr_ctc`); those binders are loud-partial and name this primitive as the piece that already exists, so no live call site exists yet. Reserved: the graph-side `OpKind` variant + C ABI export |
//! | [`RNNT_DECODE_OP`]               | FR-OP-42 | runtime primitive landed (`vokra_ops::rnnt_decode`) with a **live consumer** in `ParakeetTdt11b::decode_tdt`; the same strict model now has a complete native PCM-to-token/text TDT forward. Reserved: the graph-side `OpKind` variant + C ABI export |
//! | [`ECAPA_TDNN_SPEAKER_ENCODE_OP`] | FR-OP-80 | native 200-tensor SpeechBrain ECAPA binder, CPU/Metal forward, CLI/C ABI speaker route and independent parity landed. Reserved: the graph-side `OpKind` variant + dedicated C ABI op export |
//! | [`WESPEAKER_SPEAKER_ENCODE_OP`]  | FR-OP-80 | native strict WeSpeaker binder, CPU/Metal forward, CLI/C ABI speaker route and independent parity landed. Reserved: the graph-side `OpKind` variant + dedicated C ABI op export |
//! | [`TITANET_SPEAKER_ENCODE_OP`]    | FR-OP-80 | native strict TitaNet-L binder, CPU/Metal forward, CLI/C ABI speaker route and independent parity landed. Reserved: the graph-side `OpKind` variant + dedicated C ABI op export |
//! | [`DIARIZE_OP`]                   | FR-OP-82 | trigger only (pyannote license MIT primary source 2026-07-30 signed = `docs/license-audit.md` §3.1 row 263, `gated: auto` は access control のみで追加条項なし) |

/// BigVGAN generator op-kind identifier. Re-exported from the M2-08 registry:
/// the min-dtype audit anchor (fp16 minimum) is already registered there, and
/// the runtime vocoder itself has landed (`vokra_ops::bigvgan_generator` plus
/// the `vokra_models::bigvgan` arch binder). What stays M5-residual is the
/// **graph-side `OpKind` variant + C ABI export**. ADR M4-20 §D-5 / §D-6.
pub use crate::quant::registry::BIGVGAN_GENERATOR_OP;

/// CTC decoder op-kind identifier (FR-OP-41). Reserved; unregistered.
///
/// The runtime primitives (`vokra_ops::ctc_decode_greedy` /
/// `ctc_decode_beam`) have landed; this reserves only the graph-side variant.
pub const CTC_DECODE_OP: &str = "ctc_decode";

/// RNN-T decoder op-kind identifier (FR-OP-42). Reserved; unregistered.
///
/// The runtime primitive (`vokra_ops::rnnt_decode`) has landed and has a live
/// consumer (`ParakeetTdt11b::decode_tdt`); this reserves only the graph-side
/// variant.
pub const RNNT_DECODE_OP: &str = "rnnt_decode";

/// ECAPA-TDNN speaker-encoder op-kind identifier. The native model forward is
/// live; only a graph-side `OpKind` variant and dedicated op-level C ABI export
/// remain reserved. The model-generic `vokra_speaker_embed` C ABI already
/// consumes ECAPA through `SpeakerEngine`.
pub const ECAPA_TDNN_SPEAKER_ENCODE_OP: &str = "ecapa_tdnn_speaker_encode";

/// WeSpeaker speaker-encoder op-kind identifier. The native model forward is
/// live; only a graph-side `OpKind` variant and dedicated op-level C ABI export
/// remain reserved. The generic `vokra_speaker_embed` C ABI already consumes
/// WeSpeaker through `SpeakerEngine`.
pub const WESPEAKER_SPEAKER_ENCODE_OP: &str = "wespeaker_speaker_encode";

/// TitaNet speaker-encoder op-kind identifier. The native model forward is
/// live; only a graph-side `OpKind` variant and dedicated op-level C ABI export
/// remain reserved. The generic `vokra_speaker_embed` C ABI already consumes
/// TitaNet through `SpeakerEngine`.
pub const TITANET_SPEAKER_ENCODE_OP: &str = "titanet_speaker_encode";

/// Diarization op-kind identifier (FR-OP-82). Reserved; unregistered.
/// **2026-07-30 license half unblock**: pyannote weight は MIT primary source
/// (HF cardData `license: mit, gated: auto` = access control のみ、追加条項なし、
/// `docs/license-audit.md` §3.1 row 263 で 2026-07-30 yousan sign)。従前の
/// "license + trigger double blocker" は "trigger only blocker" に縮小
/// (実 op 実装 + trigger converter + real-checkpoint parity harness が
/// 残 M5 wave の scope)。
pub const DIARIZE_OP: &str = "diarize";

/// One M5-residual op anchor: op-kind id + the FR-OP it will satisfy + the
/// reason it is deferred to M5. Used for documentation / tooling that wants to
/// enumerate the deferred catalogue without hard-coding the list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct M5ResidualAnchor {
    /// Reserved op-kind identifier.
    pub op_id: &'static str,
    /// FR-OP requirement this anchor will satisfy when landed in M5.
    pub fr_op: &'static str,
    /// What is still reserved, and why it is M5-residual.
    ///
    /// Reasons split into two shapes, and mixing them up is what let this
    /// column go stale: either the primitive genuinely does not exist yet
    /// (covered elsewhere / license), or the **runtime function has landed**
    /// and only the graph-side `OpKind` variant + C ABI export remain
    /// reserved (ADR M4-20 §D-5). Prefer the latter wording whenever the
    /// `vokra-ops` module exists — an absence-claim there is a factual claim
    /// that decays, and `scripts/check-m5-residual-blockers.sh` rejects it.
    pub blocker: &'static str,
}

/// The full M5-residual op catalogue (ADR M4-20 §D-6). Landing any of these in
/// M5 is a backward-compatible additive — this list + the abi-changelog
/// "Reserved additions" entry are the forward reservation.
pub fn m5_residual_op_anchors() -> &'static [M5ResidualAnchor] {
    &[
        M5ResidualAnchor {
            op_id: BIGVGAN_GENERATOR_OP,
            fr_op: "FR-OP-11",
            blocker: "graph-side OpKind variant + C ABI export reserved; the runtime vocoder \
                      landed (vokra_ops::bigvgan_generator, plus the vokra_models::bigvgan arch \
                      binder whose decode delegates verbatim) and the min-dtype anchor is \
                      registered (M2-08); strict real-weight binding and waveform parity landed",
        },
        M5ResidualAnchor {
            op_id: CTC_DECODE_OP,
            fr_op: "FR-OP-41",
            blocker: "graph-side OpKind variant + C ABI export reserved; the runtime primitives \
                      landed (vokra_ops::ctc_decode_greedy / ctc_decode_beam with LM shallow \
                      fusion + hotwords) and the NeMo family landed (parakeet_ctc, canary, \
                      canary_qwen, canary_1b_flash, omniasr_ctc), whose loud-partial binders \
                      name this primitive as already-existing — no live call site yet",
        },
        M5ResidualAnchor {
            op_id: RNNT_DECODE_OP,
            fr_op: "FR-OP-42",
            blocker: "graph-side OpKind variant + C ABI export reserved; the runtime primitive \
                      landed (vokra_ops::rnnt_decode) with a live consumer in \
                      ParakeetTdt11b::decode_tdt; the same strict model now has a complete \
                      native PCM-to-token/text TDT forward",
        },
        M5ResidualAnchor {
            op_id: ECAPA_TDNN_SPEAKER_ENCODE_OP,
            fr_op: "FR-OP-80",
            blocker: "native SpeechBrain ECAPA binder and CPU/Metal forward landed; reserved: the graph-side OpKind variant + dedicated op-level C ABI export",
        },
        M5ResidualAnchor {
            op_id: WESPEAKER_SPEAKER_ENCODE_OP,
            fr_op: "FR-OP-80",
            blocker: "native strict WeSpeaker binder and CPU/Metal forward landed; reserved: the graph-side OpKind variant + dedicated op-level C ABI export",
        },
        M5ResidualAnchor {
            op_id: TITANET_SPEAKER_ENCODE_OP,
            fr_op: "FR-OP-80",
            blocker: "native strict TitaNet-L binder and CPU/Metal forward landed; reserved: the graph-side OpKind variant + dedicated op-level C ABI export",
        },
        M5ResidualAnchor {
            op_id: DIARIZE_OP,
            fr_op: "FR-OP-82",
            blocker: "trigger only (pyannote license MIT signed 2026-07-30, §3.1 row 263)",
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::registry::MinDtypeRegistry;

    /// The six *new* M4-20 T14 anchors (everything except BigVGAN, whose
    /// min-dtype anchor legitimately lives in the registry) must be declared
    /// but **not** registered in `MinDtypeRegistry::builtin` and must carry
    /// their expected identifier strings — the reserved-but-unregistered
    /// guarantee (KOKORO_ISTFT_HEAD_OP pattern, ADR M4-20 §D-6).
    #[test]
    fn new_anchors_are_reserved_but_unregistered() {
        let reg = MinDtypeRegistry::builtin();
        for (constant, expected) in [
            (CTC_DECODE_OP, "ctc_decode"),
            (RNNT_DECODE_OP, "rnnt_decode"),
            (ECAPA_TDNN_SPEAKER_ENCODE_OP, "ecapa_tdnn_speaker_encode"),
            (WESPEAKER_SPEAKER_ENCODE_OP, "wespeaker_speaker_encode"),
            (TITANET_SPEAKER_ENCODE_OP, "titanet_speaker_encode"),
            (DIARIZE_OP, "diarize"),
        ] {
            assert_eq!(constant, expected, "anchor id must be stable");
            assert!(
                reg.lookup(constant).is_none(),
                "M5-residual op `{constant}` must NOT be registered before its M5 landing"
            );
        }
    }

    /// BigVGAN is the one exception: its min-dtype anchor IS registered (fp16
    /// minimum, M2-08), but the generator op landing is still M5. This documents
    /// the distinction so a reader does not mistake the registry entry for a
    /// landed op.
    #[test]
    fn bigvgan_min_dtype_anchor_is_registered_but_op_is_m5() {
        let reg = MinDtypeRegistry::builtin();
        assert!(
            reg.lookup(BIGVGAN_GENERATOR_OP).is_some(),
            "BigVGAN min-dtype anchor is registered (M2-08); only the op landing is M5"
        );
        assert_eq!(BIGVGAN_GENERATOR_OP, "bigvgan_generator");
    }

    /// The catalogue covers exactly the seven M5-residual ops with the correct
    /// FR-OP mapping; a change to this set is a deliberate scope decision, not a
    /// silent edit (mirrors the registry `builtin_has_exactly_four_entries`
    /// discipline).
    #[test]
    fn catalogue_is_the_seven_residual_ops() {
        let anchors = m5_residual_op_anchors();
        assert_eq!(anchors.len(), 7, "seven M5-residual ops (ADR M4-20 §D-6)");
        // Every op id is unique.
        for i in 0..anchors.len() {
            for j in (i + 1)..anchors.len() {
                assert_ne!(anchors[i].op_id, anchors[j].op_id, "op ids must be unique");
            }
        }
        // Speaker-encoder variants all anchor FR-OP-80.
        for a in anchors {
            if a.op_id.contains("speaker_encode") {
                assert_eq!(a.fr_op, "FR-OP-80", "speaker variants anchor FR-OP-80");
            }
        }
        // Spot-check a couple of mappings.
        assert!(
            anchors
                .iter()
                .any(|a| a.op_id == CTC_DECODE_OP && a.fr_op == "FR-OP-41")
        );
        assert!(
            anchors
                .iter()
                .any(|a| a.op_id == DIARIZE_OP && a.fr_op == "FR-OP-82")
        );
    }
}
