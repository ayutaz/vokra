//! EnCodec residual VQ decode — **engine op only; weights permanently
//! zoo-excluded** (M4-04; FR-OP-30 op / FR-OP-32 permanent constraint).
//!
//! # The op exists; the weights never ship
//!
//! EnCodec's **code** is MIT, but the pretrained EnCodec **weights** are
//! CC-BY-NC 4.0 (non-commercial). FR-OP-32 makes the exclusion permanent:
//!
//! - the official model zoo never carries an EnCodec GGUF;
//! - `crates/vokra-convert` has **no** `encodec` model kind (this is
//!   deliberate and load-bearing — `scripts/compliance/check-encodec-exclusion.sh`
//!   greps vokra-convert to keep it that way);
//! - the M2-13 runtime gate (`registry_lookup("encodec") ==
//!   LicenseClass::NonCommercial`) refuses to load an EnCodec weight without
//!   an explicit research flag;
//! - even the parity fixtures are generated from **fixed-seed synthetic
//!   codebooks** run through the MIT reference *code* — no pretrained weight
//!   is downloaded anywhere in this repository or its CI.
//!
//! A developer with their own locally obtained EnCodec checkpoint and the
//! research flag can still evaluate it — that is exactly what this op is for
//! (ADR M3-06 §D2 / ADR M4-04 §D-d).
//!
//! # Structure (source: facebookresearch/encodec, MIT — ADR M4-04 §T02)
//!
//! EnCodec's RVQ has **no input/output projections**: its
//! `VectorQuantization` only instantiates projections when
//! `codebook_dim != dim`, and the released models pass a single `dimension`
//! (= the SEANet encoder dim), so decode is a plain per-layer codebook
//! lookup (`EuclideanCodebook.decode = F.embedding(codes, embed)`) summed
//! across quantizers. That is byte-for-byte the shape-generic gather + FP32
//! fold this crate already ships for Mimi, so [`encodec_rvq_decode`] is a
//! **thin entry over the same core** (`rvq_fold_core`) — zero duplicated
//! math, but honest `encodec_rvq:`-prefixed argument errors.
//!
//! # No canonical attrs baked
//!
//! [`EncodecRvqAttrs`] has **no** canonical constructor: EnCodec is not in
//! the zoo, so there is no converter default to mirror — attrs are always
//! caller-supplied (from whatever research checkpoint the caller loaded).
//! Shape provenance for the parity fixture is recorded in
//! `tests/parity/encodec/manifest.txt`.
//!
//! # No paged variant
//!
//! FR-OP-30's paged variants are model-synced (Mimi → CSM/Moshi, DAC → zoo);
//! EnCodec has no zoo consumer, so no paged variant is provided. A research
//! caller who needs paging can decode to features and use the paged store
//! directly.
//!
//! # No silent fallback (FR-EX-08)
//!
//! Out-of-range indices and shape mismatches are explicit
//! [`VokraError::InvalidArgument`] — identical contract to
//! [`crate::mimi_rvq::mimi_rvq_decode`].

use vokra_core::{Result, VokraError};

use crate::mimi_rvq::{CodebookTable, rvq_fold_core};

// ---------------------------------------------------------------------------
// Op attributes
// ---------------------------------------------------------------------------

/// Static shape attributes for an EnCodec RVQ decode.
///
/// Deliberately has **no** canonical constructor (module docs): EnCodec is
/// permanently zoo-excluded (FR-OP-32), so there is no converter-emitted
/// default to mirror. Callers supply the shape of the research checkpoint
/// they loaded themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodecRvqAttrs {
    /// Number of quantizers (base + residuals).
    pub n_codebooks: usize,
    /// Number of entries per codebook.
    pub codebook_size: usize,
    /// Feature dimension per codebook entry (EnCodec has no factorized
    /// projection — the codebook rows live directly in the feature space).
    pub d_model: usize,
}

// ---------------------------------------------------------------------------
// Core op
// ---------------------------------------------------------------------------

/// Decodes a `[time, n_codebooks]` row-major `codes` block into a
/// `[time, d_model]` row-major feature buffer, summing every codebook's
/// contribution in FP32 — the shape-generic RVQ path (`rvq_fold_core`)
/// shared with `mimi_rvq`.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on any of:
/// - shape mismatch in `codes` / `codebook_tables` vs `attrs`;
/// - `codes[t, cb] >= attrs.codebook_size` (no silent clamp — FR-EX-08).
pub fn encodec_rvq_decode(
    codes: &[u32],
    time: usize,
    codebook_tables: &[CodebookTable],
    attrs: &EncodecRvqAttrs,
) -> Result<Vec<f32>> {
    if attrs.n_codebooks == 0 || attrs.codebook_size == 0 || attrs.d_model == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "encodec_rvq: attrs must have every axis > 0, got n_codebooks={} \
             codebook_size={} d_model={}",
            attrs.n_codebooks, attrs.codebook_size, attrs.d_model,
        )));
    }
    if codebook_tables.len() != attrs.n_codebooks {
        return Err(VokraError::InvalidArgument(format!(
            "encodec_rvq: codebook_tables.len() {} != attrs.n_codebooks {}",
            codebook_tables.len(),
            attrs.n_codebooks
        )));
    }
    for (i, t) in codebook_tables.iter().enumerate() {
        if t.codebook_size != attrs.codebook_size || t.d_model != attrs.d_model {
            return Err(VokraError::InvalidArgument(format!(
                "encodec_rvq: codebook_tables[{i}] shape [{},{}] != attrs [{},{}]",
                t.codebook_size, t.d_model, attrs.codebook_size, attrs.d_model
            )));
        }
    }
    let expected = time.checked_mul(attrs.n_codebooks).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "encodec_rvq: time ({time}) * n_codebooks ({}) overflows usize",
            attrs.n_codebooks
        ))
    })?;
    if codes.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "encodec_rvq: codes.len() {} != time * n_codebooks {expected}",
            codes.len()
        )));
    }
    rvq_fold_core(
        codes,
        time,
        codebook_tables,
        attrs.n_codebooks,
        attrs.d_model,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mimi_rvq::{MimiRvqAttrs, mimi_rvq_decode};
    use vokra_core::compliance::{LicenseClass, registry_lookup};

    fn tiny_attrs() -> EncodecRvqAttrs {
        EncodecRvqAttrs {
            n_codebooks: 3,
            codebook_size: 4,
            d_model: 5,
        }
    }

    fn make_ramp_tables(attrs: EncodecRvqAttrs) -> Vec<CodebookTable> {
        let mut tables = Vec::with_capacity(attrs.n_codebooks);
        for cb in 0..attrs.n_codebooks {
            let mut data = vec![0.0_f32; attrs.codebook_size * attrs.d_model];
            for i in 0..attrs.codebook_size {
                for d in 0..attrs.d_model {
                    data[i * attrs.d_model + d] = (i + d) as f32 + (cb as f32) * 100.0;
                }
            }
            tables.push(CodebookTable::new(attrs.codebook_size, attrs.d_model, data).unwrap());
        }
        tables
    }

    // ---- generic-path equivalence ------------------------------------------

    #[test]
    fn rides_the_same_generic_path_as_mimi_bit_for_bit() {
        // The whole point of the thin entry: identical inputs through
        // `encodec_rvq_decode` and `mimi_rvq_decode` produce bit-identical
        // output (both delegate to `rvq_fold_core`).
        let attrs = tiny_attrs();
        let tables = make_ramp_tables(attrs);
        let time = 4;
        let codes: Vec<u32> = (0..time as u32 * attrs.n_codebooks as u32)
            .map(|i| i % attrs.codebook_size as u32)
            .collect();

        let enc = encodec_rvq_decode(&codes, time, &tables, &attrs).unwrap();
        let mimi = mimi_rvq_decode(
            &codes,
            time,
            &tables,
            &MimiRvqAttrs {
                n_codebooks: attrs.n_codebooks,
                codebook_size: attrs.codebook_size,
                d_model: attrs.d_model,
            },
        )
        .unwrap();
        assert_eq!(enc, mimi);
    }

    #[test]
    fn decode_matches_hand_fold() {
        let attrs = tiny_attrs();
        let tables = make_ramp_tables(attrs);
        let codes: Vec<u32> = vec![0, 3, 1];
        let got = encodec_rvq_decode(&codes, 1, &tables, &attrs).unwrap();

        let mut want = vec![0.0_f32; attrs.d_model];
        for cb in 0..attrs.n_codebooks {
            let idx = codes[cb] as usize;
            for d in 0..attrs.d_model {
                want[d] += tables[cb].data[idx * attrs.d_model + d];
            }
        }
        assert_eq!(got, want);
    }

    #[test]
    fn decode_rejects_shape_mismatches_and_bad_index() {
        let attrs = tiny_attrs();
        let tables = make_ramp_tables(attrs);
        // Bad codes length.
        assert!(matches!(
            encodec_rvq_decode(&[0u32; 2], 1, &tables, &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
        // Bad table count.
        assert!(matches!(
            encodec_rvq_decode(&[0u32; 3], 1, &tables[..2], &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
        // Out-of-range index — FR-EX-08.
        assert!(matches!(
            encodec_rvq_decode(&[0, 0, 4], 1, &tables, &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
        // Zero-axis attrs.
        assert!(matches!(
            encodec_rvq_decode(
                &[],
                0,
                &[],
                &EncodecRvqAttrs {
                    n_codebooks: 0,
                    codebook_size: 0,
                    d_model: 0
                }
            ),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- T19: host-only fallback smoke -------------------------------------

    #[test]
    fn host_only_smoke_decode_end_to_end() {
        // The engine-op path runs on the CPU with zero external dependencies
        // (research-checkpoint decode is host-only today; GPU arms stay
        // explicit UnsupportedOp — FR-EX-08).
        let attrs = EncodecRvqAttrs {
            n_codebooks: 2,
            codebook_size: 3,
            d_model: 4,
        };
        let tables = make_ramp_tables(EncodecRvqAttrs {
            n_codebooks: 2,
            codebook_size: 3,
            d_model: 4,
        });
        let out = encodec_rvq_decode(&[0, 2, 1, 1], 2, &tables, &attrs).unwrap();
        assert_eq!(out.len(), 2 * attrs.d_model);
    }

    // ---- weight-exclusion posture (FR-OP-32) -------------------------------

    #[test]
    fn registry_still_classifies_encodec_as_non_commercial() {
        // Pin the M2-13 runtime gate: adding the engine op must not have
        // moved EnCodec weights out of the NonCommercial class. If this test
        // ever fails, FR-OP-32 (permanent constraint) is being violated —
        // do NOT "fix" the test; fix the registry.
        assert_eq!(
            registry_lookup("encodec"),
            Some(LicenseClass::NonCommercial),
            "FR-OP-32: EnCodec weights are permanently non-commercial; \
             the research-flag gate must keep refusing them by default"
        );
    }
}
