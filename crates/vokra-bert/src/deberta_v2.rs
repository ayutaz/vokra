//! DeBERTa v2 encoder — clean-room per arXiv:2006.03654.
//!
//! # References (permissive only)
//!
//! - He, Liu, Gao, Chen 2021 (arXiv:2006.03654)
//! - microsoft/DeBERTa (MIT)
//! - HuggingFace transformers `deberta_v2` (Apache-2.0)
//!
//! # NOT REFERENCED
//!
//! - github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)

/// Log-scale relative position bucket per DeBERTa v2 (§3.2, "disentangled
/// attention"). Positions closer to `q` get finer buckets; positions far
/// away get log-spaced buckets that saturate at `bucket_size - 1`.
///
/// # Arguments
/// - `q_pos`, `k_pos`: absolute positions of query and key
/// - `bucket_size`: number of buckets (typically 256)
/// - `max_dist`: distance beyond which all positions share the last bucket
///
/// # Algorithm (arXiv:2006.03654 eq. after §3.2)
///
/// rel = q_pos - k_pos
/// sign = sign(rel)
/// mid = bucket_size / 2
/// if |rel| < mid: bucket = mid + rel   # linear near-region
/// else: bucket = mid + sign * (mid + log(|rel|/mid) / log(max_dist/mid) * mid)
///                            .clamp(0, bucket_size - 1)
pub fn relative_position_bucket(q_pos: i32, k_pos: i32, bucket_size: i32, max_dist: i32) -> i32 {
    let rel = q_pos - k_pos;

    // Special case: same position → bucket 0
    if rel == 0 {
        return 0;
    }

    let sign = if rel > 0 { 1 } else { -1 };
    let mid = bucket_size / 2;
    let abs_rel = rel.abs();
    if abs_rel < mid {
        (mid + rel).clamp(0, bucket_size - 1)
    } else {
        let log_ratio = (abs_rel as f32 / mid as f32).ln() / (max_dist as f32 / mid as f32).ln();
        let far = mid + (log_ratio * mid as f32) as i32;
        let bucketed = mid + sign * far.min(mid);
        bucketed.clamp(0, bucket_size - 1)
    }
}
