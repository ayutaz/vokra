//! T5 bidirectional relative-position buckets.
//!
//! T5 does not add an absolute position embedding to its token states.  The
//! first encoder block instead owns a learned `[num_buckets, num_heads]`
//! table and turns every `(query, key)` displacement into one bucket.  The
//! resulting per-head bias is reused by every encoder layer.  Keeping this
//! indexing rule in one runtime function avoids each T5 consumer (MusicGen,
//! AudioGen, JASCO, AudioLDM2 and MT3) growing a subtly different copy.
//!
//! This is deterministic index/gather glue, not a graph backend op: the
//! learned attention reductions still run through the selected model
//! backend.  The equations follow the Apache-2.0 Transformers
//! `T5Attention._relative_position_bucket` implementation and Raffel et al.
//! (2020), §2.1.

use vokra_core::{Result, VokraError};

/// Explicit T5 relative-position geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct T5RelativePositionAttrs {
    /// Total learned buckets, split equally between negative and positive
    /// displacements for a bidirectional encoder.
    pub num_buckets: usize,
    /// Largest logarithmic distance represented before clamping to the last
    /// bucket in each direction.
    pub max_distance: usize,
    /// Whether positive and negative displacements use distinct halves.
    pub bidirectional: bool,
}

impl T5RelativePositionAttrs {
    /// Canonical T5-base encoder geometry.
    pub const T5_BASE: Self = Self {
        num_buckets: 32,
        max_distance: 128,
        bidirectional: true,
    };

    /// Reject geometry that would make the exact/logarithmic split empty.
    pub fn validate(self) -> Result<()> {
        let directional = if self.bidirectional {
            if self.num_buckets % 2 != 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "T5 relative-position num_buckets must be even in bidirectional mode, got {}",
                    self.num_buckets
                )));
            }
            self.num_buckets / 2
        } else {
            self.num_buckets
        };
        let exact = directional / 2;
        if exact == 0 || self.max_distance <= exact {
            return Err(VokraError::InvalidArgument(format!(
                "T5 relative-position geometry requires directional buckets >= 2 and max_distance > exact split; got num_buckets={}, max_distance={}, bidirectional={}",
                self.num_buckets, self.max_distance, self.bidirectional
            )));
        }
        Ok(())
    }
}

/// Map one signed `key_position - query_position` displacement to a T5
/// relative-attention bucket.
pub fn t5_relative_position_bucket(
    relative_position: isize,
    attrs: T5RelativePositionAttrs,
) -> Result<usize> {
    attrs.validate()?;

    let mut buckets = attrs.num_buckets;
    let mut bucket = 0usize;
    let distance = if attrs.bidirectional {
        buckets /= 2;
        if relative_position > 0 {
            bucket += buckets;
        }
        relative_position.unsigned_abs()
    } else {
        // Decoder-style T5 attention clamps future positions to zero and
        // measures only how far the key lies in the past.
        relative_position.saturating_neg().max(0) as usize
    };

    let max_exact = buckets / 2;
    if distance < max_exact {
        return Ok(bucket + distance);
    }

    let ratio = distance as f64 / max_exact as f64;
    let log_range = (attrs.max_distance as f64 / max_exact as f64).ln();
    let logarithmic =
        max_exact + (ratio.ln() / log_range * (buckets - max_exact) as f64).floor() as usize;
    Ok(bucket + logarithmic.min(buckets - 1))
}

/// Build the `[query_len, key_len]` row-major bucket index matrix used by a
/// T5 attention layer.
pub fn t5_relative_position_buckets(
    query_len: usize,
    key_len: usize,
    attrs: T5RelativePositionAttrs,
) -> Result<Vec<usize>> {
    attrs.validate()?;
    if query_len == 0 || key_len == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "T5 relative-position query_len and key_len must be non-zero, got {query_len}x{key_len}"
        )));
    }
    if query_len > isize::MAX as usize || key_len > isize::MAX as usize {
        return Err(VokraError::InvalidArgument(format!(
            "T5 relative-position axes must fit isize, got {query_len}x{key_len}"
        )));
    }
    let len = query_len.checked_mul(key_len).ok_or_else(|| {
        VokraError::InvalidArgument("T5 relative-position matrix length overflow".to_owned())
    })?;
    let mut output = Vec::with_capacity(len);
    for query in 0..query_len {
        for key in 0..key_len {
            output.push(t5_relative_position_bucket(
                key as isize - query as isize,
                attrs,
            )?);
        }
    }
    Ok(output)
}

/// Gather a learned `[num_buckets, num_heads]` table into row-major
/// `[num_heads, query_len, key_len]` attention bias.
pub fn t5_relative_attention_bias(
    table: &[f32],
    num_heads: usize,
    query_len: usize,
    key_len: usize,
    attrs: T5RelativePositionAttrs,
) -> Result<Vec<f32>> {
    attrs.validate()?;
    let expected = attrs.num_buckets.checked_mul(num_heads).ok_or_else(|| {
        VokraError::InvalidArgument("T5 relative-bias table length overflow".to_owned())
    })?;
    if num_heads == 0 || table.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "T5 relative-bias table must be [num_buckets={}, num_heads={num_heads}] ({expected} values), got {}",
            attrs.num_buckets,
            table.len()
        )));
    }
    let buckets = t5_relative_position_buckets(query_len, key_len, attrs)?;
    let plane = query_len.checked_mul(key_len).ok_or_else(|| {
        VokraError::InvalidArgument("T5 relative-bias plane length overflow".to_owned())
    })?;
    let output_len = num_heads.checked_mul(plane).ok_or_else(|| {
        VokraError::InvalidArgument("T5 relative-bias output length overflow".to_owned())
    })?;
    let mut output = vec![0.0; output_len];
    for head in 0..num_heads {
        for (position, &bucket) in buckets.iter().enumerate() {
            output[head * plane + position] = table[bucket * num_heads + head];
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t5_base_bucket_boundaries_match_transformers() {
        let a = T5RelativePositionAttrs::T5_BASE;
        let cases = [
            (0, 0),
            (-1, 1),
            (-7, 7),
            (-8, 8),
            (-15, 9),
            (-16, 10),
            (-31, 11),
            (-32, 12),
            (-63, 13),
            (-64, 14),
            (-127, 15),
            (-128, 15),
            (1, 17),
            (7, 23),
            (8, 24),
            (16, 26),
            (32, 28),
            (64, 30),
            (128, 31),
            (10_000, 31),
        ];
        for (relative, expected) in cases {
            assert_eq!(
                t5_relative_position_bucket(relative, a).unwrap(),
                expected,
                "relative={relative}"
            );
        }
    }

    #[test]
    fn matrix_uses_key_minus_query_direction() {
        let got =
            t5_relative_position_buckets(2, 3, T5RelativePositionAttrs::T5_BASE).expect("buckets");
        assert_eq!(got, vec![0, 17, 18, 1, 0, 17]);
    }

    #[test]
    fn decoder_mode_clamps_future_and_uses_all_buckets_for_the_past() {
        let attrs = T5RelativePositionAttrs {
            bidirectional: false,
            ..T5RelativePositionAttrs::T5_BASE
        };
        for (relative, expected) in [(7, 0), (0, 0), (-1, 1), (-15, 15), (-16, 16), (-128, 31)] {
            assert_eq!(
                t5_relative_position_bucket(relative, attrs).unwrap(),
                expected,
                "relative={relative}"
            );
        }
    }

    #[test]
    fn learned_table_gather_is_head_major() {
        let mut table = vec![0.0; 32 * 2];
        for bucket in 0..32 {
            table[bucket * 2] = bucket as f32;
            table[bucket * 2 + 1] = 100.0 + bucket as f32;
        }
        let got = t5_relative_attention_bias(&table, 2, 2, 2, T5RelativePositionAttrs::T5_BASE)
            .expect("bias");
        assert_eq!(got, vec![0.0, 17.0, 1.0, 0.0, 100.0, 117.0, 101.0, 100.0]);
    }

    #[test]
    fn malformed_geometry_and_table_fail_closed() {
        assert!(
            t5_relative_position_bucket(
                0,
                T5RelativePositionAttrs {
                    num_buckets: 31,
                    max_distance: 128,
                    bidirectional: true,
                }
            )
            .is_err()
        );
        assert!(
            t5_relative_attention_bias(&[0.0; 7], 2, 1, 1, T5RelativePositionAttrs::T5_BASE,)
                .is_err()
        );
    }
}
