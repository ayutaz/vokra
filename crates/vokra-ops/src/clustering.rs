//! Agglomerative hierarchical clustering — a runtime function pyannote-style
//! speaker-diarization pipelines call to aggregate segment-level speaker
//! embeddings into speaker clusters (pyannote Wave 4; supports the
//! FR-OP-82 `diarize` residual anchor —
//! [`vokra_core::m5_residual_ops::DIARIZE_OP`]).
//!
//! # Runtime function — NOT a graph node (FR-EX-10 / FR-OP-40 posture)
//!
//! [`AgglomerativeClustering::cluster`] is a **host-side runtime function**,
//! not an `OpKind` variant. Encoding a clustering pass as a graph op would
//! freeze the `threshold`, `metric`, and `linkage` into the model at
//! conversion time — precisely the axes a diarization caller varies most
//! often — and would break execution-provider compatibility (the "contrib
//! op" anti-pattern, FR-OP-40). This mirrors the [`ctc_decode`] /
//! [`flow_sample`](crate::flow_sampler) posture. The full `diarize` op
//! remains M5-residual (owner-side; HF-gated weight license + trigger model
//! blocker — `docs/license-audit.md` §3 row 122); this primitive is the
//! *clustering step alone* and lands independently.
//!
//! # Upstream reference
//!
//! Primary source (as directed):
//! [`pyannote/audio/pipelines/clustering.py`][pyannote-clustering]
//! (`develop` branch, MIT, `Copyright (c) 2020 CNRS`).
//! `AgglomerativeClustering.cluster_embeddings` delegates the linkage
//! recurrence and the distance-threshold cut to
//! [`scipy.cluster.hierarchy.linkage`][scipy-linkage] +
//! [`scipy.cluster.hierarchy.fcluster`][scipy-fcluster], parametrised by
//! `metric` (`"cosine"` / `"euclidean"`) and `method` (`"single"` /
//! `"complete"` / `"average"`) — the three linkage variants and two metrics
//! this module exposes.
//!
//! The algorithm implemented here is the textbook agglomerative
//! hierarchical clustering (bottom-up single/complete/average linkage with
//! a distance-threshold cut) — Sokal & Michener 1958 (UPGMA); the linkage
//! recurrence lives in Lance & Williams 1967, ["A General Theory of
//! Classificatory Sorting Strategies"][lance-williams]. `scipy`'s docs give
//! the standard reference for the same three method aliases pyannote passes
//! through.
//!
//! ## Distance metric definitions
//!
//! - **Cosine**: `d(a, b) = 1 - (a · b) / (||a||₂ · ||b||₂)`. Range
//!   `[0, 2]` for arbitrary sign; `[0, 1]` for non-negative embeddings.
//!   Zero-magnitude edge cases (an all-zero embedding vs any other) have
//!   no mathematically defined similarity — this module returns `0.0` when
//!   both operands are zero (they are byte-identical, hence "identical" by
//!   construction) and `1.0` when exactly one operand is zero (no directional
//!   overlap). This is a documented convention, not a silent fallback.
//! - **Euclidean**: `d(a, b) = √(Σᵢ (aᵢ - bᵢ)²)`. Well-defined for every
//!   `Vec<f32>` pair (no zero-magnitude edge case).
//!
//! ## Linkage recurrences (cluster-level distance from point-level)
//!
//! Given two clusters `A` and `B` and the point-level pairwise-distance
//! matrix `d`, the linkage distance is:
//!
//! - **Single** (nearest-neighbour): `min_{a∈A, b∈B} d(a, b)`.
//! - **Complete** (farthest-neighbour): `max_{a∈A, b∈B} d(a, b)`.
//! - **Average** (UPGMA — unweighted pair group method with arithmetic
//!   mean, `scipy` `method="average"`):
//!   `(1 / (|A|·|B|)) · Σ_{a∈A, b∈B} d(a, b)`. Equivalently the
//!   Lance-Williams update
//!   `d(k, i∪j) = (|i|·d(k,i) + |j|·d(k,j)) / (|i| + |j|)` — the direct
//!   pairwise mean is used here because `O(n²)` overall time keeps the
//!   Lance-Williams shortcut from mattering for the target working set
//!   (< 500 segment-level embeddings per typical audio file).
//!
//! ## Threshold semantics
//!
//! Two clusters merge iff their linkage distance is **strictly less than**
//! `threshold`. A pair at distance exactly equal to `threshold` does not
//! merge (this matches `scipy.cluster.hierarchy.fcluster(..., t=threshold,
//! criterion="distance")` when read as "cut at height `threshold`" — the
//! cut removes edges of length ≥ `threshold`).
//!
//! # Zero-dependency posture (NFR-DS-02)
//!
//! No third-party crate, no BLAS. The pairwise-distance matrix and the
//! linkage recurrence live in this file (each a handful of lines); the
//! root `Cargo.lock` continues to list only `vokra-*` packages.
//!
//! # Speaker-diarization scope (Design 判断 8, CLAUDE.md)
//!
//! Speaker diarization is **not** voice cloning — it identifies "who spoke
//! when", never re-synthesises a voice. This module therefore lives in the
//! main `ayutaz/vokra` repository and is unaffected by the
//! `vokra-voiceclone-experimental` separation (`docs/legal-compliance.md`
//! §voice-clone).
//!
//! [pyannote-clustering]: https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/pipelines/clustering.py
//! [scipy-linkage]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
//! [scipy-fcluster]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html
//! [lance-williams]: https://academic.oup.com/comjnl/article-abstract/9/4/373/480290
//! [`ctc_decode`]: crate::ctc_decode

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Distance metric between two embedding vectors.
///
/// pyannote passes the metric to `scipy.cluster.hierarchy.linkage` via the
/// `metric=` keyword; the two variants below are the two the pyannote
/// speaker-diarization pipeline actually configures. See the module docs
/// for the exact formula (including the zero-magnitude convention for
/// cosine).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Cosine distance: `1 - (a · b) / (||a||₂ · ||b||₂)`. Speaker
    /// embeddings from CAM++ / ECAPA-TDNN / pyannote embed models are
    /// direction-carrying by convention, so cosine is pyannote's default.
    Cosine,
    /// Euclidean distance: `√(Σᵢ (aᵢ - bᵢ)²)`. Useful when the caller has
    /// L2-normalised embeddings *and* wants a chord-length reading of
    /// similarity (chord² = 2 · (1 - cosine)).
    Euclidean,
}

/// Linkage rule for computing a cluster-to-cluster distance from the
/// point-level pairwise distances.
///
/// pyannote passes the linkage to `scipy.cluster.hierarchy.linkage` via
/// the `method=` keyword; the three variants below are the three the
/// pyannote pipeline configures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkageMethod {
    /// Nearest-neighbour: `d(A, B) = min_{a∈A, b∈B} d(a, b)`. Sensitive to
    /// chain effects — a bridging pair between two otherwise-separated
    /// clusters collapses them.
    Single,
    /// Farthest-neighbour: `d(A, B) = max_{a∈A, b∈B} d(a, b)`. Resists
    /// chaining; a bridging pair only merges when the *whole* neighbouring
    /// cluster is close.
    Complete,
    /// UPGMA (unweighted pair group method with arithmetic mean):
    /// `d(A, B) = (1 / (|A|·|B|)) · Σ_{a∈A, b∈B} d(a, b)`. `scipy`'s
    /// `method="average"`.
    Average,
}

/// Agglomerative hierarchical clustering (bottom-up single / complete /
/// average linkage with a distance-threshold cut).
///
/// See the module-level docs for the algorithm reference and the
/// zero-dep + no-silent-fallback posture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgglomerativeClustering {
    /// Distance cutoff. Two clusters merge iff their linkage distance is
    /// **strictly less than** `threshold`. Typical pyannote speaker-
    /// diarization value: `0.7` (cosine metric).
    pub threshold: f32,
    /// Distance metric on the point (embedding) level.
    pub metric: DistanceMetric,
    /// Linkage rule for lifting point-level distances to cluster-level
    /// distances.
    pub linkage: LinkageMethod,
}

impl AgglomerativeClustering {
    /// Clusters row vectors (embeddings) and returns per-row cluster
    /// assignments (0-indexed).
    ///
    /// # Contract
    ///
    /// - **Empty input → empty output** (well-defined: no rows, no
    ///   clusters).
    /// - **Single input → `[0]`** (one row is always its own cluster).
    /// - **Every row is assigned exactly one cluster id**; ids are
    ///   `0..k` with no gaps (dense, contiguous).
    /// - **Cluster ids are deterministic**: a cluster's id is the position
    ///   at which its smallest-index member first appears — i.e. the
    ///   cluster containing embedding `0` is always id `0`, the cluster
    ///   containing the next unassigned index gets id `1`, and so on.
    /// - **Threshold**: two clusters merge iff their linkage distance is
    ///   **strictly less than** `self.threshold`.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!` / `assert!`) if `embeddings` contains
    /// rows of inconsistent length — this is a caller-invariant violation
    /// (the same row-shape rule that [`crate::dct()`] and [`crate::mel`]
    /// panic on): a speaker encoder always emits a fixed-dimension
    /// embedding, so a length mismatch is a wiring bug, not runtime data.
    ///
    /// # Complexity
    ///
    /// `O(n² · d)` for the pairwise-distance precompute + `O(n³)` in the
    /// worst case for the merge loop (recomputing linkage from the point-
    /// level matrix each step). Acceptable for the target working set
    /// (< 500 segment-level embeddings per typical audio file); the
    /// Lance-Williams `O(n²)` shortcut is a deliberate follow-up when a
    /// larger working set emerges.
    pub fn cluster(&self, embeddings: &[Vec<f32>]) -> Vec<usize> {
        let n = embeddings.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![0];
        }

        // Row-shape invariant: every embedding must carry the same
        // dimension. A speaker encoder emits fixed-dim rows by construction,
        // so a mismatch is a caller-side wiring bug — panic like `dct` /
        // `mel_filterbank` do on the same class of contract violation.
        let dim = embeddings[0].len();
        for (i, e) in embeddings.iter().enumerate() {
            assert_eq!(
                e.len(),
                dim,
                "AgglomerativeClustering::cluster: embedding {i} has dim {} \
                 but embedding 0 has dim {dim} (speaker-encoder outputs must \
                 share a fixed dimension)",
                e.len()
            );
        }

        // Precompute the symmetric point-level pairwise-distance matrix
        // (row-major `n × n`, zeros on the diagonal).
        let point_dist = pairwise_distances(embeddings, self.metric, n);

        // Each row starts in its own singleton cluster; `clusters[i]` is the
        // list of embedding indices grouped together so far.
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        // Greedy merge loop: find the closest pair of clusters (under the
        // configured linkage rule) and merge them iff their distance is
        // strictly less than `threshold`. Exits either when fewer than two
        // clusters remain (`find_closest_pair` → `None`) or when the
        // closest pair is at or above the threshold cut.
        while let Some((i, j, dist)) = find_closest_pair(&clusters, &point_dist, n, self.linkage) {
            if dist >= self.threshold {
                // Closest pair is already farther than the cut — done.
                break;
            }
            // Merge cluster `j` into cluster `i` (j > i by construction of
            // `find_closest_pair`, so removing `j` first does not shift
            // `i`'s index).
            debug_assert!(j > i, "closest-pair enforces j > i");
            let mut j_members = clusters.remove(j);
            clusters[i].append(&mut j_members);
        }

        assignments_from_clusters(clusters, n)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Row-major `n × n` pairwise-distance matrix (symmetric, zero diagonal).
fn pairwise_distances(embeddings: &[Vec<f32>], metric: DistanceMetric, n: usize) -> Vec<f32> {
    let mut d = vec![0.0f32; n * n];
    for i in 0..n {
        // Diagonal is 0 by construction (self-distance).
        for j in (i + 1)..n {
            let v = point_distance(&embeddings[i], &embeddings[j], metric);
            d[i * n + j] = v;
            d[j * n + i] = v;
        }
    }
    d
}

/// Point-level distance dispatcher.
fn point_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::Euclidean => euclidean_distance(a, b),
    }
}

/// Cosine distance with an explicit zero-magnitude convention.
///
/// `1 - (a·b) / (||a||·||b||)`. When *both* operands are all-zero the
/// distance is `0.0` (they are byte-identical); when *exactly one* operand
/// is all-zero the distance is `1.0` (no directional overlap).
///
/// The dot product and both norms accumulate in `f64` (the sum of squares
/// is the numerical pinch point — `f32` loses precision past 2²⁴ terms and
/// for magnitude products past the f32 dynamic range).
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "cosine_distance: dim mismatch — caller must pre-check"
    );
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let xf = f64::from(*x);
        let yf = f64::from(*y);
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    if na == 0.0 && nb == 0.0 {
        // Both zero → identical by construction.
        return 0.0;
    }
    if na == 0.0 || nb == 0.0 {
        // One zero, one non-zero → no direction to compare.
        return 1.0;
    }
    let sim = dot / (na.sqrt() * nb.sqrt());
    // Clamp against f64 rounding drift (a `dot/|a||b|` should live in
    // `[-1, 1]` mathematically; clamp guarantees the returned distance
    // sits in `[0, 2]` exactly).
    (1.0 - sim.clamp(-1.0, 1.0)) as f32
}

/// Standard Euclidean distance; `f64` accumulator for the same numerical
/// reason as [`cosine_distance`].
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "euclidean_distance: dim mismatch — caller must pre-check"
    );
    let mut sum = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = f64::from(*x) - f64::from(*y);
        sum += d * d;
    }
    sum.sqrt() as f32
}

/// Finds the pair `(i, j)` (`j > i`) of clusters with the minimum linkage
/// distance under `method`. Returns `None` iff fewer than two clusters
/// exist. On ties the lexicographically smaller `(i, j)` wins — the same
/// deterministic-tie posture as [`crate::ctc_decode::ctc_decode_greedy`]
/// (left-to-right first-max).
fn find_closest_pair(
    clusters: &[Vec<usize>],
    point_dist: &[f32],
    n: usize,
    method: LinkageMethod,
) -> Option<(usize, usize, f32)> {
    if clusters.len() < 2 {
        return None;
    }
    let mut best: Option<(usize, usize, f32)> = None;
    for i in 0..clusters.len() {
        for j in (i + 1)..clusters.len() {
            let d = linkage_distance(&clusters[i], &clusters[j], point_dist, n, method);
            let take = match best {
                None => true,
                // Strict `<` keeps the earliest (i, j) on a tie —
                // deterministic ordering irrespective of iteration
                // details.
                Some((_, _, cur)) => d < cur,
            };
            if take {
                best = Some((i, j, d));
            }
        }
    }
    best
}

/// Lifts a point-level pairwise-distance matrix to a cluster-level
/// distance under the requested linkage rule (see module docs for the
/// three formulas). The direct pairwise summation is `O(|A|·|B|)`; keeping
/// the overall algorithm at `O(n³)` in the worst case is acceptable for
/// the target working set (< 500 embeddings; see
/// [`AgglomerativeClustering::cluster`] complexity note).
fn linkage_distance(
    a: &[usize],
    b: &[usize],
    point_dist: &[f32],
    n: usize,
    method: LinkageMethod,
) -> f32 {
    debug_assert!(!a.is_empty() && !b.is_empty(), "empty cluster in linkage");
    match method {
        LinkageMethod::Single => {
            let mut m = f32::INFINITY;
            for &i in a {
                for &j in b {
                    let v = point_dist[i * n + j];
                    if v < m {
                        m = v;
                    }
                }
            }
            m
        }
        LinkageMethod::Complete => {
            let mut m = f32::NEG_INFINITY;
            for &i in a {
                for &j in b {
                    let v = point_dist[i * n + j];
                    if v > m {
                        m = v;
                    }
                }
            }
            m
        }
        LinkageMethod::Average => {
            // f64 accumulator — cluster sizes stay small (< 500) but the
            // sum of many `f32` distances still benefits from the wider
            // running total.
            let mut sum = 0.0f64;
            for &i in a {
                for &j in b {
                    sum += f64::from(point_dist[i * n + j]);
                }
            }
            let denom = (a.len() * b.len()) as f64;
            (sum / denom) as f32
        }
    }
}

/// Turns the final list of clusters into a dense per-row assignment
/// vector. Cluster ids are assigned in the order the cluster's smallest
/// member first appears (0-based, contiguous), so the cluster containing
/// embedding `0` is always id `0`, the cluster containing the smallest
/// unassigned index gets `1`, and so on.
fn assignments_from_clusters(mut clusters: Vec<Vec<usize>>, n: usize) -> Vec<usize> {
    // Sort clusters by their minimum member index — deterministic id
    // assignment irrespective of the merge order.
    clusters.sort_by_key(|c| *c.iter().min().expect("cluster is never empty"));
    let mut out = vec![usize::MAX; n];
    for (cid, members) in clusters.iter().enumerate() {
        for &m in members {
            debug_assert!(
                out[m] == usize::MAX,
                "embedding {m} assigned to two clusters — merge invariant broken"
            );
            out[m] = cid;
        }
    }
    debug_assert!(
        out.iter().all(|&v| v != usize::MAX),
        "some embedding was not assigned to a cluster"
    );
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// A helper so a test can declare its intent (`Cosine` / `Complete` at
    /// `threshold`) as a one-line struct build.
    fn ahc(
        threshold: f32,
        metric: DistanceMetric,
        linkage: LinkageMethod,
    ) -> AgglomerativeClustering {
        AgglomerativeClustering {
            threshold,
            metric,
            linkage,
        }
    }

    /// Counts distinct cluster ids in an assignment vector.
    fn n_clusters(a: &[usize]) -> usize {
        a.iter().collect::<BTreeSet<_>>().len()
    }

    #[test]
    fn cluster_empty_returns_empty() {
        // Empty in, empty out — the well-defined identity that keeps the
        // caller from having to special-case a zero-segment audio file
        // (FR-EX-08 posture: no silent "one empty cluster" fallback).
        let a = ahc(0.5, DistanceMetric::Cosine, LinkageMethod::Complete);
        let out = a.cluster(&[]);
        assert!(out.is_empty(), "empty in must give empty out, got {out:?}");
    }

    #[test]
    fn cluster_single_returns_one_cluster() {
        // A single embedding is always its own cluster with id 0 — the
        // "n=1 is degenerate but well-defined" contract.
        let a = ahc(0.5, DistanceMetric::Cosine, LinkageMethod::Complete);
        let out = a.cluster(&[vec![1.0, 0.0, 0.0]]);
        assert_eq!(out, vec![0]);
    }

    #[test]
    fn cluster_identical_embeddings_merge_into_one_cluster() {
        // Three copies of the same embedding — every pairwise distance is
        // 0 (cosine and euclidean both), which is strictly less than any
        // positive threshold, so they all merge into one cluster.
        let e = vec![1.0, 0.0, 0.0];
        for linkage in [
            LinkageMethod::Single,
            LinkageMethod::Complete,
            LinkageMethod::Average,
        ] {
            let a = ahc(0.5, DistanceMetric::Cosine, linkage);
            let out = a.cluster(&[e.clone(), e.clone(), e.clone()]);
            assert_eq!(
                out,
                vec![0, 0, 0],
                "linkage {linkage:?}: identical embeddings must fold into id 0"
            );
        }
    }

    #[test]
    fn cluster_orthogonal_embeddings_stay_separate_at_low_threshold() {
        // Two orthogonal unit vectors have cosine distance = 1.0 exactly.
        // The strict-less-than merge rule + threshold=0.5 keeps them
        // apart; if the code drifted to `<=` a threshold of 1.0 would
        // silently fold orthogonal segments (that's the bug this pins).
        let a = ahc(0.5, DistanceMetric::Cosine, LinkageMethod::Complete);
        let out = a.cluster(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert_eq!(out, vec![0, 1]);
        // Also verify the strict-less-than boundary at threshold = 1.0:
        // orthogonal embeddings sit *at* the cut, so they still stay
        // separate.
        let boundary = ahc(1.0, DistanceMetric::Cosine, LinkageMethod::Complete);
        let out = boundary.cluster(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert_eq!(
            out,
            vec![0, 1],
            "distance == threshold must not merge (strict-less-than rule)"
        );
    }

    #[test]
    fn cluster_two_tight_groups_form_two_clusters() {
        // Group 0: three near-copies of (1, 0). Group 1: three near-copies
        // of (0, 1). Within-group cosine distance ≈ 0 (≤ 2e-4 in this
        // fixture); cross-group cosine distance ≈ 0.99. Threshold 0.5
        // cleanly separates them under every linkage rule.
        let e = vec![
            vec![1.0, 0.01],
            vec![1.0, 0.02],
            vec![1.0, 0.00],
            vec![0.01, 1.0],
            vec![0.02, 1.0],
            vec![0.00, 1.0],
        ];
        for linkage in [
            LinkageMethod::Single,
            LinkageMethod::Complete,
            LinkageMethod::Average,
        ] {
            let a = ahc(0.5, DistanceMetric::Cosine, linkage);
            let out = a.cluster(&e);
            assert_eq!(
                n_clusters(&out),
                2,
                "linkage {linkage:?}: expected two well-separated groups, got {out:?}"
            );
            // Deterministic id assignment: rows 0..3 (smallest member 0)
            // are cluster 0; rows 3..6 (smallest member 3) are cluster 1.
            assert_eq!(
                out,
                vec![0, 0, 0, 1, 1, 1],
                "linkage {linkage:?}: id assignment must follow smallest-member ordering"
            );
        }
    }

    #[test]
    fn cluster_linkage_methods_differ_on_bridging_pair() {
        // A chain of three points on the real line: 0.0, 0.4, 0.8.
        // Euclidean distances: d(0,1)=0.4, d(1,2)=0.4, d(0,2)=0.8.
        //
        // At threshold=0.5 the two rules diverge:
        //   • single link: after merging {0,1}, min({0,1}, 2) = min(0.4, 0.8) = 0.4 < 0.5
        //     → merges to one cluster.
        //   • complete link: after merging {0,1}, max({0,1}, 2) = max(0.4, 0.8) = 0.8 ≥ 0.5
        //     → stops; two clusters.
        // This is the textbook "chain effect" that distinguishes single
        // from complete linkage; the fixture is the smallest possible
        // one that surfaces the divergence.
        let e = vec![vec![0.0], vec![0.4], vec![0.8]];
        let out_s = ahc(0.5, DistanceMetric::Euclidean, LinkageMethod::Single).cluster(&e);
        let out_c = ahc(0.5, DistanceMetric::Euclidean, LinkageMethod::Complete).cluster(&e);
        assert_eq!(
            n_clusters(&out_s),
            1,
            "single-link chain must collapse to one cluster, got {out_s:?}"
        );
        assert_eq!(
            n_clusters(&out_c),
            2,
            "complete-link chain must remain two clusters, got {out_c:?}"
        );
        // Complete-link specific assignment: {0,1} merges but 2 stays
        // alone → smallest member of {0,1} is 0 (id 0), of {2} is 2
        // (id 1).
        assert_eq!(out_c, vec![0, 0, 1]);
    }

    // ---- edge cases ------------------------------------------------------

    #[test]
    fn cluster_all_zero_embeddings_merge_via_zero_cosine_distance() {
        // Two all-zero embeddings: mathematically the cosine similarity is
        // undefined (0/0), but the module's documented convention returns
        // distance 0.0 for the both-zero case (they are byte-identical).
        // At any positive threshold they must merge.
        let a = ahc(0.1, DistanceMetric::Cosine, LinkageMethod::Complete);
        let out = a.cluster(&[vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]]);
        assert_eq!(out, vec![0, 0]);
    }

    #[test]
    fn cluster_zero_and_nonzero_stay_separate_under_cosine() {
        // One zero + one non-zero: the documented convention returns
        // distance 1.0. Any threshold ≤ 1.0 keeps them separate.
        let a = ahc(0.5, DistanceMetric::Cosine, LinkageMethod::Complete);
        let out = a.cluster(&[vec![0.0, 0.0], vec![1.0, 0.0]]);
        assert_eq!(out, vec![0, 1]);
    }

    #[test]
    fn cluster_ids_are_dense_and_contiguous() {
        // A general-position input: three well-separated groups of
        // different sizes. Cluster ids must be exactly {0, 1, 2} — no
        // gaps, no ids ≥ 3.
        let e = vec![
            vec![1.0, 0.0, 0.0],  // group 0
            vec![1.0, 0.01, 0.0], // group 0
            vec![0.0, 1.0, 0.0],  // group 1
            vec![0.0, 1.0, 0.01], // group 1
            vec![0.0, 1.0, 0.02], // group 1
            vec![0.0, 0.0, 1.0],  // group 2
        ];
        let a = ahc(0.5, DistanceMetric::Cosine, LinkageMethod::Complete);
        let out = a.cluster(&e);
        let ids: BTreeSet<usize> = out.iter().copied().collect();
        assert_eq!(
            ids,
            BTreeSet::from([0, 1, 2]),
            "cluster ids must be dense {{0, 1, 2}}, got {out:?}"
        );
    }

    #[test]
    fn cluster_high_threshold_folds_everything_into_one_cluster() {
        // A threshold above every possible pairwise distance folds every
        // embedding into one cluster — the degenerate but well-defined
        // "everything merges" case. Cosine distance is bounded above by 2,
        // so `threshold = 10.0` guarantees the fold.
        let e = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
        ];
        let a = ahc(10.0, DistanceMetric::Cosine, LinkageMethod::Complete);
        let out = a.cluster(&e);
        assert_eq!(out, vec![0, 0, 0, 0]);
    }

    #[test]
    fn cluster_zero_threshold_keeps_every_embedding_separate() {
        // Threshold = 0.0 with the strict-less-than merge rule keeps every
        // embedding in its own cluster (even byte-identical ones — their
        // distance is 0.0, and 0.0 is not strictly < 0.0). This is the
        // documented "no merge at threshold 0" contract.
        let e = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let a = ahc(0.0, DistanceMetric::Cosine, LinkageMethod::Complete);
        let out = a.cluster(&e);
        assert_eq!(out, vec![0, 1, 2]);
    }

    // ---- shape validation ------------------------------------------------

    #[test]
    #[should_panic(expected = "AgglomerativeClustering::cluster")]
    fn cluster_panics_on_dim_mismatch() {
        // A speaker encoder always emits fixed-dim rows; a length mismatch
        // is a caller-invariant violation (the same class of contract
        // violation `dct` / `mel_filterbank` panic on).
        let a = ahc(0.5, DistanceMetric::Cosine, LinkageMethod::Complete);
        let _ = a.cluster(&[vec![1.0, 0.0, 0.0], vec![1.0, 0.0]]);
    }
}
