//! IR distinction between `length_conditioning` and `duration_expander`
//! (M3-08-T07; WP completion condition — milestones.md §7 M3-08).
//!
//! The completion condition of M3-08 is "`duration_expander` と別 op として
//! IR で区別が強制されることを検証テストで確認": the two ops share nothing
//! at the IR level, and mixing them must be either a compile-time error or
//! (for runtime-only mixups) a `dispatch` error.
//!
//! `duration_expander` is a **reserved placeholder** at the time this test
//! was written — no `OpKind::DurationExpander` variant exists yet (its own
//! WP will add one). So the "distinct from `duration_expander`" claim is
//! proven negatively — with `mas` (FR-OP-72) and any other placeholder /
//! existing variant standing in for "not-`length_conditioning`":
//!
//! - **Rust type system as the first defence (ADR 0010 §D6-1/§D6-4)**: the
//!   attrs are a distinct type (`LengthConditioningAttrs`) reachable only via
//!   `OpKind::LengthConditioning`; the compile-fail check below (as a
//!   commented, code-form doc) documents that an attempt to reuse the attrs
//!   under a different variant does not typecheck.
//! - **`dispatch` distinctness at runtime**: only the
//!   `OpKind::LengthConditioning` arm routes into the length-conditioning
//!   code path. Every other current variant that could be confused with it
//!   (arbitrary front-end / preprocessing ops that also carry attrs and no
//!   runtime inputs) is proven to *not* execute the length-conditioning
//!   computation.
//! - **`AudioGraph::validate` accepts `LengthConditioning` as a first-class
//!   node**: it walks tensor-id lists uniformly, so a `LengthConditioning`
//!   node in a graph passes validation without any custom shape check
//!   (`Fused(...)` precedent — M2-04-T02).

use vokra_core::{AudioGraph, DType, GraphBuilder, OpKind, Result, TensorDesc, VokraError};
use vokra_ops::attrs::{LengthConditioningAttrs, StftAttrs};
use vokra_ops::{OpValue, dispatch};

fn build_singleton(op: OpKind, out_shape: [usize; 1]) -> Result<AudioGraph> {
    let mut b = GraphBuilder::new();
    let y = b.add_tensor(TensorDesc::new("y", DType::F32, out_shape));
    b.add_node(op, &[], &[y]);
    b.mark_output(y);
    b.finish()
}

#[test]
fn opkind_variants_are_distinct_types_at_runtime() {
    // `OpKind::LengthConditioning` and every other variant are *different
    // discriminants*. Rust's enum tags make this a runtime-checkable claim;
    // matching on the wrong arm is what dispatch relies on.
    let lc = OpKind::LengthConditioning(LengthConditioningAttrs::ref_linear(100, 2.0));
    let stft = OpKind::Stft(StftAttrs::new(400, 160));
    let matmul = OpKind::MatMul;

    // Cross-comparison: no two of these should equal each other. This
    // exercises `PartialEq`, which is exactly what a graph rewrite / fusion
    // pass relies on when it checks "is this the length-conditioning node?".
    assert_ne!(lc, stft);
    assert_ne!(lc, matmul);
    assert_ne!(stft, matmul);

    // A same-attrs `LengthConditioning` node is equal to itself: the
    // *variant*, not the wrapped attrs, is the identity anchor.
    let lc2 = OpKind::LengthConditioning(LengthConditioningAttrs::ref_linear(100, 2.0));
    assert_eq!(lc, lc2);
}

#[test]
fn length_conditioning_node_validates_in_an_audio_graph() {
    // Structural check: a graph whose sole node is `OpKind::LengthConditioning`
    // must pass `validate()`. This is the same uniform tensor-id walk that
    // `Fused(...)` proved in M2-04-T02 — the new variant is not a bypass.
    let attrs = LengthConditioningAttrs::ref_linear(100, 2.0);
    let graph = build_singleton(OpKind::LengthConditioning(attrs), [1])
        .expect("LengthConditioning graph validates");
    assert_eq!(graph.nodes().len(), 1);
    assert!(matches!(
        graph.nodes()[0].op(),
        OpKind::LengthConditioning(_)
    ));
    assert!(graph.validate().is_ok());
}

#[test]
fn length_conditioning_validate_rejects_double_producer() {
    // Uniform structural checks apply. A `LengthConditioning` node and any
    // other node writing the same output tensor must be rejected — the new
    // variant does not carve out a single-producer exemption. This is the
    // externally-observable analogue of `graph::tests::double_producer_is_rejected`.
    let mut b = GraphBuilder::new();
    let y = b.add_tensor(TensorDesc::new("y", DType::F32, [1]));
    let attrs = LengthConditioningAttrs::ref_linear(100, 2.0);
    b.add_node(OpKind::LengthConditioning(attrs), &[], &[y]);
    // A second node writing `y` triggers the single-producer check.
    b.add_node(OpKind::DcOffsetRemove, &[], &[y]);
    b.mark_output(y);
    let err = b.finish().unwrap_err();
    assert!(
        matches!(err, VokraError::GraphValidation(_)),
        "expected GraphValidation, got {err:?}"
    );
}

#[test]
fn dispatch_only_length_conditioning_variant_computes_length() {
    // The core distinctness claim: dispatch routes ONLY
    // `OpKind::LengthConditioning` into `length_conditioning::apply`. Every
    // other variant is either a different op or (for ops outside dispatch's
    // set — e.g. MatMul, Softmax) an explicit `UnsupportedOp` error. This is
    // what stops a graph-builder mistake ("I meant duration_expander but I
    // typed length_conditioning") from silently doing the wrong thing.
    let attrs = LengthConditioningAttrs::ref_linear(100, 2.0);
    let out = dispatch(&OpKind::LengthConditioning(attrs), &[]).unwrap();
    let (shape, data) = out[0].as_real().unwrap();
    assert_eq!(shape, &[1]);
    assert_eq!(data, &[200.0]);

    // A same-arity, same-input-shape "not-length_conditioning" op runs a
    // completely different computation (or errors). We probe two:
    //   (1) `MatMul` — outside dispatch's set → `UnsupportedOp`, verifying
    //       that the fallthrough arm does NOT route a graph author's typo
    //       into the length-conditioning path;
    let e = dispatch(&OpKind::MatMul, &[]).unwrap_err();
    assert!(matches!(e, VokraError::UnsupportedOp(_)), "MatMul: {e:?}");
    //   (2) `DcOffsetRemove` — inside dispatch's set but a different op:
    //       running it with the same "no runtime inputs" arity used by
    //       `length_conditioning` fails with `InvalidArgument`, again
    //       proving no cross-routing to the length-conditioning code path.
    let e = dispatch(&OpKind::DcOffsetRemove, &[]).unwrap_err();
    assert!(
        matches!(e, VokraError::InvalidArgument(_)),
        "DcOffsetRemove: {e:?}"
    );
}

#[test]
fn dispatch_rejects_extra_runtime_input_for_length_conditioning() {
    // `length_conditioning` takes **no** runtime tensor inputs (mode B's
    // ref_speech_frames lives in the attrs, per ADR 0010 §D8). A caller
    // that supplies one — the shape a `duration_expander` node would take
    // (per-phoneme lengths tensor) — must be rejected. This is the
    // runtime-side second defence against the confusion the compile-time
    // type check catches.
    let attrs = LengthConditioningAttrs::ref_linear(100, 2.0);
    let fake_per_phoneme = OpValue::real(vec![3], vec![10.0, 20.0, 30.0]);
    let e = dispatch(&OpKind::LengthConditioning(attrs), &[fake_per_phoneme]).unwrap_err();
    assert!(matches!(e, VokraError::InvalidArgument(_)), "extra: {e:?}");
}

/// Compile-time distinctness (illustrative — this is NOT a runnable test).
///
/// Attempting to wrap `LengthConditioningAttrs` in a variant that does not
/// carry it fails to compile. The following would not typecheck (both lines
/// left as comments so the crate still builds):
///
/// ```ignore
/// use vokra_core::OpKind;
/// use vokra_ops::attrs::LengthConditioningAttrs;
///
/// // (a) A non-LengthConditioning variant does not accept the attrs.
/// let _bad: OpKind = OpKind::Stft(LengthConditioningAttrs::ref_linear(100, 2.0));
/// //                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/// //         expected `StftAttrs`, found `LengthConditioningAttrs`
///
/// // (b) `LengthConditioningAttrs` cannot silently coerce to another
/// //     attrs type.
/// fn takes_stft(_: vokra_ops::attrs::StftAttrs) {}
/// takes_stft(LengthConditioningAttrs::ref_linear(100, 2.0));
/// //         ^ mismatched types
/// ```
///
/// Rust's enum discriminant + attrs typing together enforce the IR
/// distinction claimed in ADR 0010 §D6 without any custom validation code.
#[test]
fn compile_time_distinctness_is_documented() {
    // Executable no-op: the meat of this test is the doc comment above,
    // which sits next to the test so `cargo doc` picks it up. We assert the
    // structural claim runtime-side too, so this test still exercises API
    // surface: `StftAttrs` and `LengthConditioningAttrs` are structurally
    // different types — different fields, different Rust identity.
    let stft: vokra_ops::attrs::StftAttrs = StftAttrs::new(400, 160);
    let lc = LengthConditioningAttrs::ref_linear(100, 2.0);
    // Format them: `StftAttrs` reports "StftAttrs {", `LengthConditioningAttrs`
    // reports "LengthConditioningAttrs {". A "mistakenly aliased" typedef
    // would collapse to the same struct name in the debug output.
    let a = format!("{stft:?}");
    let b = format!("{lc:?}");
    assert!(a.starts_with("StftAttrs"), "stft debug = {a}");
    assert!(b.starts_with("LengthConditioningAttrs"), "lc debug = {b}");
    assert_ne!(a, b);
}
