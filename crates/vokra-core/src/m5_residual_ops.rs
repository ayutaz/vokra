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
//! identifiers — declared, so a future M5 landing lands on a stable name, but
//! **not** inserted into any registry / [`OpKind`](crate::ir::OpKind) and
//! adding **no** C ABI symbol (the whole point). It is the exact
//! [`KOKORO_ISTFT_HEAD_OP`](crate::quant::registry::KOKORO_ISTFT_HEAD_OP)
//! pattern (reserved-but-unregistered, guarded by a test) generalized to the
//! M4-20 catalogue, and it pairs with the `docs/abi-changelog.md` "Reserved
//! additions" entry so an M5 op landing is a backward-compatible additive.
//!
//! # Blockers (why each is M5-residual)
//!
//! | op-kind id                       | FR-OP    | blocker                                            |
//! | -------------------------------- | -------- | -------------------------------------------------- |
//! | [`BIGVGAN_GENERATOR_OP`]         | FR-OP-11 | no trigger model (Kokoro=iSTFTNet, CosyVoice2=Mimi, piper-plus=MB-iSTFT); the min-dtype anchor is already in the registry, only the *generator op landing* is M5 |
//! | [`CTC_DECODE_OP`]                | FR-OP-41 | NeMo-family trigger pending                        |
//! | [`RNNT_DECODE_OP`]               | FR-OP-42 | NeMo-family trigger pending                        |
//! | [`ECAPA_TDNN_SPEAKER_ENCODE_OP`] | FR-OP-80 | CAM++ already covers speaker embedding             |
//! | [`WESPEAKER_SPEAKER_ENCODE_OP`]  | FR-OP-80 | CAM++ already covers speaker embedding             |
//! | [`TITANET_SPEAKER_ENCODE_OP`]    | FR-OP-80 | CAM++ covers it; TitaNet NVIDIA NC unconfirmed     |
//! | [`DIARIZE_OP`]                   | FR-OP-82 | trigger + license (pyannote HF-gated) double blocker |

/// BigVGAN generator op-kind identifier. Re-exported from the M2-08 registry:
/// the min-dtype audit anchor (fp16 minimum) is already registered there, but
/// the **generator op landing** is M5-residual (no trigger model). ADR M4-20
/// §D-6.
pub use crate::quant::registry::BIGVGAN_GENERATOR_OP;

/// CTC decoder op-kind identifier (FR-OP-41). Reserved; unregistered.
pub const CTC_DECODE_OP: &str = "ctc_decode";

/// RNN-T decoder op-kind identifier (FR-OP-42). Reserved; unregistered.
pub const RNNT_DECODE_OP: &str = "rnnt_decode";

/// ECAPA-TDNN speaker-encoder op-kind identifier (FR-OP-80 variant, covered by
/// CAM++). Reserved; unregistered.
pub const ECAPA_TDNN_SPEAKER_ENCODE_OP: &str = "ecapa_tdnn_speaker_encode";

/// WeSpeaker speaker-encoder op-kind identifier (FR-OP-80 variant, covered by
/// CAM++). Reserved; unregistered.
pub const WESPEAKER_SPEAKER_ENCODE_OP: &str = "wespeaker_speaker_encode";

/// TitaNet speaker-encoder op-kind identifier (FR-OP-80 variant; NVIDIA NC
/// restriction unconfirmed). Reserved; unregistered.
pub const TITANET_SPEAKER_ENCODE_OP: &str = "titanet_speaker_encode";

/// Diarization op-kind identifier (FR-OP-82; trigger + license double blocker).
/// Reserved; unregistered.
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
    /// Why it is M5-residual (no trigger model / license / covered elsewhere).
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
            blocker: "no trigger model; min-dtype anchor already registered, op landing is M5",
        },
        M5ResidualAnchor {
            op_id: CTC_DECODE_OP,
            fr_op: "FR-OP-41",
            blocker: "NeMo-family trigger pending",
        },
        M5ResidualAnchor {
            op_id: RNNT_DECODE_OP,
            fr_op: "FR-OP-42",
            blocker: "NeMo-family trigger pending",
        },
        M5ResidualAnchor {
            op_id: ECAPA_TDNN_SPEAKER_ENCODE_OP,
            fr_op: "FR-OP-80",
            blocker: "CAM++ already covers speaker embedding",
        },
        M5ResidualAnchor {
            op_id: WESPEAKER_SPEAKER_ENCODE_OP,
            fr_op: "FR-OP-80",
            blocker: "CAM++ already covers speaker embedding",
        },
        M5ResidualAnchor {
            op_id: TITANET_SPEAKER_ENCODE_OP,
            fr_op: "FR-OP-80",
            blocker: "CAM++ covers it; TitaNet NVIDIA NC unconfirmed",
        },
        M5ResidualAnchor {
            op_id: DIARIZE_OP,
            fr_op: "FR-OP-82",
            blocker: "trigger + license (pyannote HF-gated) double blocker",
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
