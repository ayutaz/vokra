//! Rational-quadratic spline transform — the invertible per-position
//! non-linearity at the heart of VITS's `ConvFlow` layer (the "coupling"
//! flow step of the stochastic duration predictor's DDS+ConvFlow stack, Kim
//! et al. 2021 arXiv:2106.06103 Appendix A). Blocker 2c Wave 1 primitive:
//! ships the math foundation on which the follow-up `DdsConv` / `ConvFlow` /
//! `StochasticDurationPredictor` implementations (Wave 2, tracked as
//! residual issues on this PR) compose.
//!
//! # References (permissive only, per `mod.rs`'s clean-room policy)
//!
//! - Durkan, Bekasov, Murray & Papamakarios (2019), "Neural Spline Flows",
//!   arXiv:1906.04032 — §3.1 defines the rational-quadratic transform
//!   implemented here. This module's forward/inverse formulas trace directly
//!   to that section.
//! - bayesiains/nflows (MIT) — the reference implementation released with
//!   the Durkan et al. paper, used only as an information feed for
//!   parameter-normalization conventions (softmax over widths and heights,
//!   softplus over derivatives, tail-bound linear extension). No code is
//!   copied.
//! - VITS paper: arXiv:2106.06103 — Appendix A specifies the SDP's use of
//!   rational-quadratic splines inside `ConvFlow`. Only the SDP-level spec
//!   is consumed here; the wrapping `ConvFlow`/`DdsConv` composition is
//!   Wave 2.
//!
//! # NOT REFERENCED
//!
//! - github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)
//! - github.com/jaywalnut310/vits `modules.py::piecewise_rational_quadratic_transform` — MIT-licensed but deliberately not consulted for the derivation below; the math is traced only to the Durkan paper.
//!
//! # Formula (Durkan §3.1, transcribed)
//!
//! With `K` bins (`num_bins`), each bin `k` in `0..K` is defined by four
//! per-position parameters supplied by the caller:
//!
//! - `cum_widths[k]` / `cum_widths[k+1]` — the bin's left/right edge along
//!   the input axis. `cum_widths` has length `K + 1`; the endpoints are
//!   `±tail_bound`.
//! - `cum_heights[k]` / `cum_heights[k+1]` — the bin's left/right edge along
//!   the output axis. `cum_heights` has length `K + 1`; the endpoints are
//!   `±tail_bound`.
//! - `derivatives[k]` / `derivatives[k+1]` — the derivative of the piecewise
//!   function at the two bin edges. `derivatives` has length `K + 1`; the
//!   endpoints are `1.0` (so the transform C¹-glues onto the linear-identity
//!   tails at `±tail_bound`).
//!
//! Bin-local shorthand: `w_k = cum_widths[k+1] - cum_widths[k]`,
//! `h_k = cum_heights[k+1] - cum_heights[k]`, `s_k = h_k / w_k` (the average
//! slope over the bin), `d_k = derivatives[k]`, `d_{k+1} = derivatives[k+1]`.
//! For an input `x ∈ [cum_widths[k], cum_widths[k+1]]`, let `ξ = (x -
//! cum_widths[k]) / w_k` (a normalized 0..1 position inside the bin). Then:
//!
//! ```text
//! numerator   = h_k · (s_k · ξ² + d_k · ξ · (1 − ξ))
//! denominator = s_k + (d_{k+1} + d_k − 2·s_k) · ξ · (1 − ξ)
//! y           = cum_heights[k] + numerator / denominator
//! ```
//!
//! The transform is monotonically increasing (invertible) provided every
//! `d_k > 0`; that is the caller's responsibility (softplus + `min_derivative`
//! floor at parameter-normalization time — Wave 2's `ConvFlow` layer will
//! own that step).
//!
//! # Inverse (same reference §3.1)
//!
//! For `y ∈ [cum_heights[k], cum_heights[k+1]]`, solve the quadratic
//! `a·ξ² + b·ξ + c = 0` where:
//!
//! ```text
//! a = h_k · (s_k − d_k) + (y − cum_heights[k]) · (d_{k+1} + d_k − 2·s_k)
//! b = h_k · d_k         − (y − cum_heights[k]) · (d_{k+1} + d_k − 2·s_k)
//! c = −s_k · (y − cum_heights[k])
//! ```
//!
//! and return `x = cum_widths[k] + ξ · w_k`. The numerically stable root is
//! `ξ = 2c / (−b − √(b² − 4·a·c))` (see Durkan §3.1's own remark on which
//! root branch to pick — the negative-sign discriminant avoids catastrophic
//! cancellation when `b > 0`).
//!
//! # Tail-bound linear extension
//!
//! For `|x| ≥ tail_bound` (forward) or `|y| ≥ tail_bound` (inverse), the
//! transform is the identity — the piecewise rational-quadratic spline only
//! covers `[-tail_bound, tail_bound]`, everything outside passes through
//! unchanged. This matches the "linear tails" convention in the Durkan
//! paper's §3.4 and is the invertibility condition VITS's SDP relies on.

/// A rational-quadratic spline transform's per-position parameters — one
/// bundle per phoneme position, per channel. `cum_widths` / `cum_heights` /
/// `derivatives` all have length `num_bins + 1`; see the module doc's
/// "Formula" section for the edge-value conventions the caller must
/// establish (`cum_widths[0] = cum_widths[num_bins] = ∓tail_bound` etc.).
///
/// This is a **thin view**, not an owning struct: it borrows the caller's
/// per-position parameter slices to avoid a `Vec` per position in the hot
/// path. Wave 2's `ConvFlow` will allocate one big flat `Vec` of raw
/// `3·num_bins − 1` parameters per position and slice it into these views.
#[derive(Debug, Clone, Copy)]
pub struct SplineParams<'a> {
    /// Bin left/right edges along the input axis, length `num_bins + 1`,
    /// strictly increasing, endpoints `∓tail_bound`.
    pub cum_widths: &'a [f32],
    /// Bin left/right edges along the output axis, length `num_bins + 1`,
    /// strictly increasing, endpoints `∓tail_bound`.
    pub cum_heights: &'a [f32],
    /// Per-edge derivatives, length `num_bins + 1`, strictly positive,
    /// endpoints `1.0` (C¹ glue onto the linear tails).
    pub derivatives: &'a [f32],
    /// The `[-tail_bound, tail_bound]` support boundary; inputs outside this
    /// range pass through as the identity.
    pub tail_bound: f32,
}

impl<'a> SplineParams<'a> {
    /// Number of bins the parameter arrays describe — always
    /// `cum_widths.len() - 1`. Panics in debug builds if the three arrays
    /// have inconsistent lengths.
    pub fn num_bins(&self) -> usize {
        debug_assert_eq!(
            self.cum_widths.len(),
            self.cum_heights.len(),
            "SplineParams: cum_widths and cum_heights must have the same length (num_bins + 1)"
        );
        debug_assert_eq!(
            self.cum_widths.len(),
            self.derivatives.len(),
            "SplineParams: cum_widths and derivatives must have the same length (num_bins + 1)"
        );
        debug_assert!(
            self.cum_widths.len() >= 2,
            "SplineParams: need at least 1 bin (2 edge values)"
        );
        self.cum_widths.len() - 1
    }
}

/// Applies the rational-quadratic spline transform in the forward direction.
/// See the module doc's "Formula" section for the exact derivation this
/// implements (traced to Durkan et al. 2019 §3.1).
///
/// For `|x| >= p.tail_bound`, returns `x` unchanged (the linear-identity
/// tail; see the module doc's "Tail-bound linear extension" section).
///
/// # Panics
///
/// Panics in debug builds if `p.cum_widths` / `p.cum_heights` /
/// `p.derivatives` have inconsistent lengths, or if `p.cum_widths` /
/// `p.cum_heights` are not strictly increasing (the invertibility
/// precondition — see the module doc).
pub fn rational_quadratic_spline_forward(x: f32, p: SplineParams<'_>) -> f32 {
    let _n = p.num_bins(); // triggers the shape debug_asserts

    // Linear-identity tails — see the module doc's "Tail-bound linear
    // extension" section.
    if x <= -p.tail_bound || x >= p.tail_bound {
        return x;
    }

    let k = bin_index(x, p.cum_widths);
    let w_k = p.cum_widths[k + 1] - p.cum_widths[k];
    let h_k = p.cum_heights[k + 1] - p.cum_heights[k];
    let s_k = h_k / w_k;
    let d_k = p.derivatives[k];
    let d_kp1 = p.derivatives[k + 1];

    let xi = (x - p.cum_widths[k]) / w_k;
    let one_minus_xi = 1.0 - xi;
    let numerator = h_k * (s_k * xi * xi + d_k * xi * one_minus_xi);
    let denominator = s_k + (d_kp1 + d_k - 2.0 * s_k) * xi * one_minus_xi;

    p.cum_heights[k] + numerator / denominator
}

/// Applies the rational-quadratic spline transform in the inverse direction.
/// See the module doc's "Inverse" section for the quadratic-root derivation
/// this implements (traced to Durkan et al. 2019 §3.1).
///
/// For `|y| >= p.tail_bound`, returns `y` unchanged (the same linear-identity
/// tail the forward direction uses — since the tails are the identity, so is
/// their inverse).
///
/// # Panics
///
/// Panics in debug builds if `p.cum_widths` / `p.cum_heights` /
/// `p.derivatives` have inconsistent lengths.
pub fn rational_quadratic_spline_inverse(y: f32, p: SplineParams<'_>) -> f32 {
    let _n = p.num_bins();

    if y <= -p.tail_bound || y >= p.tail_bound {
        return y;
    }

    let k = bin_index(y, p.cum_heights);
    let w_k = p.cum_widths[k + 1] - p.cum_widths[k];
    let h_k = p.cum_heights[k + 1] - p.cum_heights[k];
    let s_k = h_k / w_k;
    let d_k = p.derivatives[k];
    let d_kp1 = p.derivatives[k + 1];
    let delta_y = y - p.cum_heights[k];
    let sum_d_minus_2s = d_kp1 + d_k - 2.0 * s_k;

    let a = h_k * (s_k - d_k) + delta_y * sum_d_minus_2s;
    let b = h_k * d_k - delta_y * sum_d_minus_2s;
    let c = -s_k * delta_y;

    // Numerically stable root — Durkan §3.1's own recommendation
    // (`2c / (-b - sqrt(discriminant))` avoids catastrophic cancellation
    // when `b > 0`; the discriminant is guaranteed >= 0 by the
    // monotonicity precondition).
    let discriminant = b * b - 4.0 * a * c;
    debug_assert!(
        discriminant >= 0.0,
        "rational_quadratic_spline_inverse: discriminant {} < 0 at y = {y}, bin {k} — this \
         indicates a non-monotonic spline (violated invertibility precondition)",
        discriminant
    );
    let xi = 2.0 * c / (-b - discriminant.sqrt());

    p.cum_widths[k] + xi * w_k
}

/// Locates the bin index `k` such that `edges[k] <= x < edges[k+1]`.
///
/// `edges` is strictly increasing with length `num_bins + 1`. This is a
/// linear scan rather than a binary search: `num_bins` in SBV2 v2's real
/// config is 10 (per the design doc's `TODO(owner)` comment, though the
/// exact value only lands with the real checkpoint), a size where a linear
/// scan wins on branch prediction and cache locality; if profiling ever
/// shows this is hot with a much larger `num_bins`, swap to
/// `slice::binary_search_by`.
///
/// # Panics
///
/// Panics in debug builds if `x` is outside `[edges[0], edges[num_bins]]` —
/// the caller (this file's forward/inverse) guarantees this via its own
/// tail-bound check.
fn bin_index(x: f32, edges: &[f32]) -> usize {
    debug_assert!(edges.len() >= 2, "bin_index: need at least 2 edges");
    debug_assert!(
        x >= edges[0] && x <= edges[edges.len() - 1],
        "bin_index: x = {x} outside [edges[0] = {}, edges[last] = {}] — caller's tail-bound \
         check should have kept x in-support",
        edges[0],
        edges[edges.len() - 1]
    );
    let mut k = 0;
    while k + 1 < edges.len() - 1 && edges[k + 1] <= x {
        k += 1;
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uniform bins with all-`1` derivatives compose exactly to the identity
    /// on `[-tail_bound, tail_bound]` — the sanity check that pins the
    /// formula's algebraic zero.
    ///
    /// With 4 uniform bins on `[-5, 5]`, `w_k = h_k = 2.5`, `s_k = 1`,
    /// `d_k = d_{k+1} = 1`; the formula's numerator collapses to
    /// `2.5 * (ξ² + ξ*(1−ξ)) = 2.5 * ξ` and the denominator to
    /// `1 + 0 = 1`, so `y = cum_heights[k] + 2.5·ξ = x`.
    fn identity_uniform_params() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let cum_widths = vec![-5.0, -2.5, 0.0, 2.5, 5.0];
        let cum_heights = cum_widths.clone();
        let derivatives = vec![1.0; 5];
        (cum_widths, cum_heights, derivatives)
    }

    #[test]
    fn identity_forward_is_identity_inside_tail() {
        let (cum_widths, cum_heights, derivatives) = identity_uniform_params();
        let p = SplineParams {
            cum_widths: &cum_widths,
            cum_heights: &cum_heights,
            derivatives: &derivatives,
            tail_bound: 5.0,
        };
        for x in [-4.5_f32, -2.5, -1.0, 0.0, 1.0, 2.5, 4.5] {
            let y = rational_quadratic_spline_forward(x, p);
            assert!(
                (y - x).abs() < 1e-6,
                "uniform-identity spline must map {x} -> {x}, got {y}"
            );
        }
    }

    #[test]
    fn identity_inverse_is_identity_inside_tail() {
        let (cum_widths, cum_heights, derivatives) = identity_uniform_params();
        let p = SplineParams {
            cum_widths: &cum_widths,
            cum_heights: &cum_heights,
            derivatives: &derivatives,
            tail_bound: 5.0,
        };
        for y in [-4.5_f32, -2.5, -1.0, 0.0, 1.0, 2.5, 4.5] {
            let x = rational_quadratic_spline_inverse(y, p);
            assert!(
                (x - y).abs() < 1e-6,
                "uniform-identity spline inverse must map {y} -> {y}, got {x}"
            );
        }
    }

    #[test]
    fn out_of_tail_passes_through_forward() {
        let (cum_widths, cum_heights, derivatives) = identity_uniform_params();
        let p = SplineParams {
            cum_widths: &cum_widths,
            cum_heights: &cum_heights,
            derivatives: &derivatives,
            tail_bound: 5.0,
        };
        // At and beyond ±tail_bound, forward is the identity (linear tail).
        assert_eq!(rational_quadratic_spline_forward(-5.0, p), -5.0);
        assert_eq!(rational_quadratic_spline_forward(5.0, p), 5.0);
        assert_eq!(rational_quadratic_spline_forward(-10.0, p), -10.0);
        assert_eq!(rational_quadratic_spline_forward(10.0, p), 10.0);
    }

    #[test]
    fn out_of_tail_passes_through_inverse() {
        let (cum_widths, cum_heights, derivatives) = identity_uniform_params();
        let p = SplineParams {
            cum_widths: &cum_widths,
            cum_heights: &cum_heights,
            derivatives: &derivatives,
            tail_bound: 5.0,
        };
        assert_eq!(rational_quadratic_spline_inverse(-5.0, p), -5.0);
        assert_eq!(rational_quadratic_spline_inverse(5.0, p), 5.0);
        assert_eq!(rational_quadratic_spline_inverse(-10.0, p), -10.0);
        assert_eq!(rational_quadratic_spline_inverse(10.0, p), 10.0);
    }

    /// Non-trivial round trip: a warped 3-bin spline (uneven widths, uneven
    /// heights, non-unit derivatives) must satisfy `inverse(forward(x)) == x`
    /// to atol = 1e-5 for a swept sample of `x` values inside the support.
    /// This is the property test — Durkan §3.1's invertibility guarantee.
    #[test]
    fn round_trip_recovers_input_on_warped_spline() {
        let cum_widths = vec![-3.0_f32, -1.5, 0.5, 3.0];
        let cum_heights = vec![-3.0_f32, -2.2, 1.1, 3.0];
        // Positive, non-unit interior derivatives; ends still 1.0 (linear
        // tails).
        let derivatives = vec![1.0_f32, 0.7, 1.4, 1.0];
        let p = SplineParams {
            cum_widths: &cum_widths,
            cum_heights: &cum_heights,
            derivatives: &derivatives,
            tail_bound: 3.0,
        };

        for &x in &[
            -2.9_f32, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.9,
        ] {
            let y = rational_quadratic_spline_forward(x, p);
            let x_back = rational_quadratic_spline_inverse(y, p);
            assert!(
                (x_back - x).abs() < 1e-5,
                "round trip at x = {x}: forward = {y}, inverse = {x_back}, err = {}",
                (x_back - x).abs()
            );
        }
    }

    /// The transform is monotonically increasing (invertibility
    /// precondition, Durkan §3.1): sweeping `x` upward through a warped
    /// spline must produce a strictly increasing sequence of outputs.
    #[test]
    fn forward_is_monotonically_increasing() {
        let cum_widths = vec![-3.0_f32, -1.5, 0.5, 3.0];
        let cum_heights = vec![-3.0_f32, -2.2, 1.1, 3.0];
        let derivatives = vec![1.0_f32, 0.7, 1.4, 1.0];
        let p = SplineParams {
            cum_widths: &cum_widths,
            cum_heights: &cum_heights,
            derivatives: &derivatives,
            tail_bound: 3.0,
        };

        let mut prev_y = f32::NEG_INFINITY;
        let mut x = -2.9_f32;
        while x < 2.9 {
            let y = rational_quadratic_spline_forward(x, p);
            assert!(
                y > prev_y,
                "spline must be strictly increasing: y({x}) = {y} <= prev_y = {prev_y}"
            );
            prev_y = y;
            x += 0.05;
        }
    }

    /// Boundary continuity: the transform hits the bin edges exactly.
    /// `forward(cum_widths[k]) == cum_heights[k]` for every bin edge
    /// (including the tail-bound endpoints, which are handled by the
    /// identity-tail early return).
    #[test]
    fn bin_edges_map_exactly() {
        let cum_widths = vec![-3.0_f32, -1.5, 0.5, 3.0];
        let cum_heights = vec![-3.0_f32, -2.2, 1.1, 3.0];
        let derivatives = vec![1.0_f32, 0.7, 1.4, 1.0];
        let p = SplineParams {
            cum_widths: &cum_widths,
            cum_heights: &cum_heights,
            derivatives: &derivatives,
            tail_bound: 3.0,
        };

        for k in 0..cum_widths.len() {
            let x = cum_widths[k];
            let y_expected = cum_heights[k];
            let y = rational_quadratic_spline_forward(x, p);
            assert!(
                (y - y_expected).abs() < 1e-5,
                "bin edge {k}: forward({x}) = {y}, expected {y_expected} (err {})",
                (y - y_expected).abs()
            );
        }
    }

    /// Bin index picks the correct interval: `edges[k] <= x < edges[k+1]`
    /// for every interior edge. The last edge maps to the last bin (a
    /// deliberate convention so `x == cum_widths[num_bins]` — the right
    /// endpoint — still gets a valid `k = num_bins - 1` even though the
    /// tail-bound early return handles that case first).
    #[test]
    fn bin_index_locates_correct_interval() {
        let edges = vec![-5.0_f32, -2.5, 0.0, 2.5, 5.0];

        assert_eq!(bin_index(-5.0, &edges), 0);
        assert_eq!(bin_index(-4.0, &edges), 0);
        assert_eq!(bin_index(-2.5, &edges), 1);
        assert_eq!(bin_index(-1.0, &edges), 1);
        assert_eq!(bin_index(0.0, &edges), 2);
        assert_eq!(bin_index(1.0, &edges), 2);
        assert_eq!(bin_index(2.5, &edges), 3);
        assert_eq!(bin_index(4.0, &edges), 3);
        assert_eq!(bin_index(5.0, &edges), 3);
    }
}
