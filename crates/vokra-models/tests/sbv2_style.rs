//! `StyleVectorInjector` (AdaIN scale + bias) tests (Task 16).
//!
//! Both tests exercise `StyleVectorInjector::from_projections` with
//! zero-initialized projection weights: since the projection is linear
//! (no bias term, see `sbv2::style` module docs), a zero weight matrix
//! forces `scale = 0` and `bias = 0` for *any* `style_vec`, so
//! `inject(hidden, seq_len, style_vec)` must leave `hidden` unchanged
//! (`h * (1 + 0) + 0 == h`) regardless of what `style_vec` contains.

use vokra_models::sbv2::StyleVectorInjector;

/// Zero-init projections + arbitrary (nonzero) `style_vec` → injection is
/// the identity, because `scale` and `bias` are both forced to zero.
#[test]
fn zero_projections_produce_identity() {
    let d_style = 4;
    let d_target = 3;
    let seq_len = 5;
    let inj = StyleVectorInjector::from_projections(
        vec![0.0; d_target * d_style], // proj_scale weights = 0
        vec![0.0; d_target * d_style], // proj_bias weights = 0
        d_style,
        d_target,
    );
    let mut hidden = vec![1.5_f32; seq_len * d_target];
    let expected = hidden.clone();
    let style = vec![0.7, -0.2, 0.5, 0.1]; // arbitrary, nonzero
    inj.inject(&mut hidden, seq_len, &style);
    assert_eq!(hidden, expected, "zero projections must give identity");
}

/// Shape invariant: injection preserves `hidden`'s length (it mutates
/// values in place, never grows or truncates the buffer).
#[test]
fn preserves_hidden_shape() {
    let d_style = 2;
    let d_target = 4;
    let seq_len = 6;
    let inj = StyleVectorInjector::from_projections(
        vec![0.1; d_target * d_style],
        vec![-0.05; d_target * d_style],
        d_style,
        d_target,
    );
    let mut hidden = vec![0.5_f32; seq_len * d_target];
    let len_before = hidden.len();
    inj.inject(&mut hidden, seq_len, &[1.0, -1.0]);
    assert_eq!(
        hidden.len(),
        len_before,
        "inject must not change hidden length"
    );
}
