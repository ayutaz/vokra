//! M2-08 T13 — quantization parity test.
//!
//! Anchors two guarantees on the M2-08 policy machinery **without** invoking
//! PyTorch or a checkpoint:
//!
//! 1. **FP16-tier weight parity (NFR-QL-01, `atol = 0.01`).** A hand-built
//!    `Q4_K` super-block filled with analytic values dequantises to a known
//!    constant; feeding a constant `F32` activation through a scalar GEMM
//!    against the dequantised weight matches the closed-form reference within
//!    `atol = 0.01`. The block bytes go through the same
//!    [`gguf::quant::dequantize`] path the runtime uses at load
//!    (`crates/vokra-core/src/gguf/reader.rs:238-244`), so the test exercises
//!    the real `dequantize(Q4_K weight) → GEMM → F32` chain rather than a
//!    parallel implementation.
//!
//! 2. **INT8 branch is *rejected*, not silently widened (FR-EX-08).** A policy
//!    resolving to `QuantScheme::W8A8Int8` for an op registered in
//!    [`MinDtypeRegistry::builtin`] (e.g. FR-OP-12 `vocos_head`) must surface
//!    as either [`VokraError::MinDtypeViolation`] (op-level minimum breached)
//!    or [`VokraError::UnsupportedQuantPath`] (no INT8 kernel on the target
//!    backend). No INT8 GEMM path exists in M2-08 — see
//!    `crates/vokra-backend-cpu/src/kernels/mod.rs:6-8` — so the parity check
//!    is a *rejection* check, not a numeric one.
//!
//! # Zero-dep + zero-fixture-file
//!
//! The fixture is materialised in-test from analytic constants; no `.gguf`
//! blob is checked in under `tests/parity/fixtures/m2-08/`. See the sibling
//! `README.md` for the closed-form reference.
//!
//! # Follow-up
//!
//! The INT8 rejection currently composes [`MinDtypeRegistry::lookup`] +
//! [`MinDtype::is_satisfied_by`] + [`QuantScheme::backend_supported`] inline
//! — the same three primitives the T09 `validate_policy_against_model`
//! function will use once it lands. When T09 ships, the
//! `validate_int8_scheme_against_registered_op` helper here can be replaced
//! by a single call to the real validator without changing the test asserts.
//!
//! Runs via `cargo test -p vokra-core --test quant_parity`.

use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile};
use vokra_core::quant::{
    self, MinDtype, MinDtypeRegistry, QuantPolicy, QuantScheme, VOCOS_HEAD_OP,
    default_vocoder_safe, whisper_q4_k,
};
use vokra_core::{BackendKind, VokraError};

/// K-quant super-block size (`QK_K` in ggml `k_quants.h`).
const QK_K: usize = 256;

/// FP16-tier weight parity absolute tolerance (NFR-QL-01).
///
/// The fixture is engineered so the closed-form reference is bit-exact
/// (dequant rounds to zero because every `q`, `sc`, `m` and `d` land on
/// exactly-representable half/single values), so the test naturally passes at
/// far tighter tolerances — but the *contract* is `atol = 0.01`, and the
/// helper below emits a max-error line so a future fixture tweak (e.g. varying
/// sub-scales) still verifies against the tier the requirements pin.
const FP16_ATOL: f32 = 0.01;

// --------------------------------------------------------------------------
// Fixture builder
// --------------------------------------------------------------------------

/// Packs eight 6-bit sub-scales + eight 6-bit sub-mins into the 12-byte
/// `scales` layout — inverse of `get_scale_min_k4` in
/// `crates/vokra-core/src/gguf/quant/mod.rs`. Transcribed from the on-disk
/// data-format specification (ggml `k_quants.h`), same pattern as the private
/// `pack_scales` helper in `crates/vokra-core/src/gguf/quant/q4_k.rs`.
fn pack_scales_q4k(sc: [u8; 8], m: [u8; 8]) -> [u8; 12] {
    let mut s = [0u8; 12];
    for j in 0..8 {
        if j < 4 {
            s[j] = sc[j] & 63;
            s[j + 4] = m[j] & 63;
        } else {
            s[j + 4] = (sc[j] & 0xF) | ((m[j] & 0xF) << 4);
            s[j - 4] |= (sc[j] >> 4) << 6;
            s[j] |= (m[j] >> 4) << 6;
        }
    }
    s
}

/// Assembles one 144-byte `Q4_K` block from semantic fields, mirroring the
/// interleave the dequantiser expects (element `64k+l` is the low nibble,
/// `64k+32+l` the high nibble of `qs[32k+l]`).
fn build_q4k_block(d: u16, dmin: u16, scales: [u8; 12], quants: [u8; QK_K]) -> Vec<u8> {
    let mut b = Vec::with_capacity(144);
    b.extend_from_slice(&d.to_le_bytes());
    b.extend_from_slice(&dmin.to_le_bytes());
    b.extend_from_slice(&scales);
    let mut qs = [0u8; 128];
    for k in 0..4 {
        for l in 0..32 {
            let lo = quants[64 * k + l] & 0xF;
            let hi = quants[64 * k + 32 + l] & 0xF;
            qs[32 * k + l] = lo | (hi << 4);
        }
    }
    b.extend_from_slice(&qs);
    b
}

/// Builds the whisper-like tiny MLP-block GGUF fixture in-memory.
///
/// Contents (see `tests/parity/fixtures/m2-08/README.md`):
/// - `mlp.0.weight` — `Q4_K`, shape `[QK_K, 1]`, closed-form value `8.0`.
/// - `mlp.0.bias`   — `F32`,  shape `[1]`, value `0.5`.
fn build_tiny_mlp_gguf() -> Vec<u8> {
    // d = 1.0 (f16 = 0x3C00), dmin = 0.0 (f16 = 0x0000)
    // sc = 2, m = 0 (uniform sub-scales; sub-mins irrelevant when dmin=0)
    // q = 4 everywhere ⇒ y = d·sc·q − dmin·m = 1.0·2·4 − 0 = 8.0
    let scales = pack_scales_q4k([2u8; 8], [0u8; 8]);
    let block = build_q4k_block(0x3C00, 0x0000, scales, [4u8; QK_K]);
    assert_eq!(block.len(), 144, "Q4_K block must be exactly 144 bytes");

    let mut b = GgufBuilder::new();
    b.add_string("vokra.model.arch", "tiny_mlp");
    b.add_tensor("mlp.0.weight", GgmlType::Q4K, vec![QK_K as u64, 1], block)
        .expect("valid Q4_K block payload");
    b.add_tensor(
        "mlp.0.bias",
        GgmlType::F32,
        vec![1],
        0.5f32.to_le_bytes().to_vec(),
    )
    .expect("valid F32 scalar payload");
    b.to_bytes().expect("builder emits valid GGUF")
}

// --------------------------------------------------------------------------
// max-error helper (spec calls this out explicitly)
// --------------------------------------------------------------------------

/// Asserts `|actual - expected| <= atol` element-wise, printing the max
/// absolute error and its index on failure.
///
/// Precedent: the M0/M1 parity harness uses ad-hoc `assert!` + a formatted
/// panic message; this helper centralises the pattern so every quant-parity
/// assertion emits a consistent, greppable "max err {…} at index {…}" line —
/// exactly what T13 asks for.
fn assert_close_f32(actual: &[f32], expected: &[f32], atol: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch — actual={} expected={}",
        actual.len(),
        expected.len()
    );
    let mut max_err = 0.0f32;
    let mut max_idx = 0usize;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let err = (a - e).abs();
        if err > max_err {
            max_err = err;
            max_idx = i;
        }
    }
    // Emit the max-error line whether we pass or fail so a CI run with `--nocapture`
    // records the tightness we actually hit for the tier.
    println!("quant_parity[{label}]: max err {max_err:.6e} at index {max_idx} (atol {atol:.6e})");
    assert!(
        max_err <= atol,
        "{label}: max err {max_err:.6e} at index {max_idx} exceeds atol {atol:.6e}"
    );
}

// --------------------------------------------------------------------------
// FP16-tier weight parity: dequantize(Q4_K) → GEMM → F32
// --------------------------------------------------------------------------

#[test]
fn dequant_q4k_then_gemm_matches_reference_within_fp16_atol() {
    // Build the whisper-like tiny MLP-block fixture, then load it back through
    // the real GGUF reader + the single `dequantize` seam the runtime uses.
    let bytes = build_tiny_mlp_gguf();
    let file = GgufFile::parse(bytes).expect("parse in-memory GGUF");

    // Sanity-check the fixture landed with the dtypes the policy expects.
    let winfo = file.tensor_info("mlp.0.weight").expect("weight present");
    assert_eq!(winfo.dtype, GgmlType::Q4K, "weight must be Q4_K");
    assert_eq!(winfo.dimensions, vec![QK_K as u64, 1]);
    let binfo = file.tensor_info("mlp.0.bias").expect("bias present");
    assert_eq!(binfo.dtype, GgmlType::F32, "bias stays F32 (FR-QT-03)");

    // Confirm the policy the converter would apply (`whisper_q4_k` preset)
    // resolves the two tensors to the schemes we materialised on disk. This is
    // the round-trip we ultimately want to test end-to-end (M2-08 T06 wires
    // the converter to obey it); testing it here anchors that a whisper-like
    // model wants Q4_K weights and F32 biases *by policy*, not by accident.
    let policy = whisper_q4_k();
    assert_eq!(
        quant::resolve(&policy, "mlp.0.weight"),
        QuantScheme::W4A16Q4K,
        "whisper_q4_k must send `.weight` tensors to Q4_K"
    );
    assert_eq!(
        quant::resolve(&policy, "mlp.0.bias"),
        QuantScheme::Fp32,
        "whisper_q4_k must keep `.bias` tensors in F32"
    );

    // Dequantise through the canonical seam `tensor_f32` (which calls
    // `gguf::quant::dequantize` — the SINGLE decode path per FR-LD-07).
    let w = file.tensor_f32("mlp.0.weight").expect("dequantize Q4_K");
    let bias = file.tensor_f32("mlp.0.bias").expect("read F32 bias");
    assert_eq!(w.len(), QK_K);
    assert_eq!(bias, vec![0.5]);

    // Closed-form reference: every weight equals d·sc·q − dmin·m = 1·2·4 − 0
    // = 8.0. Assert against the tier tolerance so a future fixture edit that
    // varies sub-scales still verifies against NFR-QL-01's FP16 atol=0.01.
    let expected_weight = vec![8.0f32; QK_K];
    assert_close_f32(&w, &expected_weight, FP16_ATOL, "dequantize(Q4_K weight)");

    // Scalar reference GEMM: y = <x, w> + bias for a single output column.
    // Activation is all-ones ⇒ dot product = QK_K · 8.0 = 2048.0.
    let x = vec![1.0f32; QK_K];
    let dot: f32 = x.iter().zip(w.iter()).map(|(&a, &b)| a * b).sum();
    let y = dot + bias[0];

    // Closed-form GEMM reference: 256 · 8.0 + 0.5 = 2048.5.
    let y_ref = 256.0 * 8.0 + 0.5;
    assert_close_f32(&[y], &[y_ref], FP16_ATOL, "gemm(dequant(Q4_K), 1s) + bias");
}

// --------------------------------------------------------------------------
// INT8 branch: policy validation must reject, not widen (FR-EX-08)
// --------------------------------------------------------------------------

/// Inline composition of the T09 check `validate_policy_against_model` that
/// isn't landed yet (see `crates/vokra-core/src/quant/validate.rs` — skeleton
/// only). Returns the *same* error variants T09 will emit:
///
/// - [`VokraError::MinDtypeViolation`] when the resolved activation dtype
///   falls below a registered op's minimum and the downgrade policy is not
///   `HifiganOptIn` (or is `HifiganOptIn` but the opt-in flag is unset).
/// - [`VokraError::UnsupportedQuantPath`] when the resolved scheme has no
///   kernel path on the target backend (M2-08: `W8A8Int8` on every backend).
///
/// This helper is deliberately test-local: T09 will replace both call sites
/// with a single call to the real validator, and the asserts below already
/// pattern-match on the two variants without inspecting message strings.
fn validate_int8_scheme_against_registered_op(
    policy: &QuantPolicy,
    op_name: &'static str,
    tensor_name: &str,
    registry: &MinDtypeRegistry,
    backend: BackendKind,
) -> Result<(), VokraError> {
    let scheme = quant::resolve(policy, tensor_name);
    let activation = scheme.activation_dtype();

    // (a) Op-level minimum activation dtype (FR-QT-03 anchor set).
    if let Some(entry) = registry.lookup(op_name)
        && !entry.min_activation.is_satisfied_by(activation)
    {
        return Err(VokraError::MinDtypeViolation {
            op: op_name.to_owned(),
            requested_scheme: scheme.as_str().to_owned(),
            min_required: match entry.min_activation {
                MinDtype::Fp16 => "fp16".to_owned(),
                MinDtype::Fp32 => "fp32".to_owned(),
                // `MinDtype` is `#[non_exhaustive]`; a future variant should be
                // labelled distinctly so the parity assertion fails loudly
                // rather than silently mislabelling the violation.
                _ => "unknown".to_owned(),
            },
            fr_ref: entry.fr_ref.to_owned(),
        });
    }

    // (b) Backend has no kernel for this scheme (FR-EX-08, no silent fallback).
    if !scheme.backend_supported(backend) {
        return Err(VokraError::UnsupportedQuantPath {
            op: op_name.to_owned(),
            scheme: scheme.as_str().to_owned(),
            backend: format!("{backend:?}"),
        });
    }

    Ok(())
}

#[test]
fn int8_policy_on_registered_fp16_op_is_rejected() {
    // W8A8 default policy applied to a vocos_head op (FR-OP-12, min=Fp16,
    // Forbidden). Either error variant is acceptable per T13's spec — both
    // encode "the runtime refused to silently drop below the op's minimum".
    let policy = QuantPolicy::new(QuantScheme::W8A8Int8);
    let registry = MinDtypeRegistry::builtin();

    let err = validate_int8_scheme_against_registered_op(
        &policy,
        VOCOS_HEAD_OP,
        "vocos_head.out.weight",
        &registry,
        BackendKind::Cpu,
    )
    .expect_err("W8A8 on vocos_head must be rejected (FR-EX-08)");

    match err {
        VokraError::MinDtypeViolation {
            op,
            requested_scheme,
            min_required,
            fr_ref,
        } => {
            assert_eq!(op, "vocos_head");
            assert_eq!(requested_scheme, "w8a8");
            assert_eq!(min_required, "fp16");
            assert_eq!(fr_ref, "FR-OP-12");
        }
        VokraError::UnsupportedQuantPath {
            op,
            scheme,
            backend: _,
        } => {
            assert_eq!(op, "vocos_head");
            assert_eq!(scheme, "w8a8");
        }
        other => panic!(
            "expected MinDtypeViolation or UnsupportedQuantPath for W8A8 on vocos_head, got {other:?}"
        ),
    }
}

#[test]
fn int8_policy_on_backend_without_kernel_surfaces_unsupported_quant_path() {
    // Empty registry ⇒ MinDtype check falls through; then W8A8 has no kernel
    // on any backend in M2-08, so the second gate must fire with
    // UnsupportedQuantPath. Confirms the two-gate ordering is *both* wired,
    // not just the first.
    let policy = QuantPolicy::new(QuantScheme::W8A8Int8);
    let empty_registry = MinDtypeRegistry::empty();

    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        let err = validate_int8_scheme_against_registered_op(
            &policy,
            "matmul",
            "encoder.blocks.0.mlp.0.weight",
            &empty_registry,
            backend,
        )
        .expect_err("W8A8 must be rejected on every backend in M2-08");
        assert!(
            matches!(err, VokraError::UnsupportedQuantPath { .. }),
            "expected UnsupportedQuantPath, got {err:?}"
        );
    }
}

#[test]
fn vocoder_safe_default_survives_all_gates() {
    // Sanity: the shipping default preset (all-fp16) passes both the
    // registered-op gate AND the backend-kernel gate for every M2-08 backend.
    // This is the "positive" companion to the two rejection asserts above and
    // pins that the parity harness itself isn't inadvertently strict.
    let policy = default_vocoder_safe();
    let registry = MinDtypeRegistry::builtin();
    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        validate_int8_scheme_against_registered_op(
            &policy,
            VOCOS_HEAD_OP,
            "vocos_head.out.weight",
            &registry,
            backend,
        )
        .expect("vocoder_safe (Fp16) must pass on every backend");
    }
}
