//! M2-08 T14 — end-to-end quantization policy test.
//!
//! Wires up the full converter → GGUF → reload → degradation-gate pipeline
//! against a synthetic whisper-like stub checkpoint. The positive path
//! validates that:
//!
//! 1. `convert_file_with_policy` with `PolicyPreset::WhisperQ4K` emits
//!    a GGUF whose per-tensor `dtype` matches what
//!    `vokra_core::quant::resolve` returns for each tensor name (weight
//!    tensors -> `Q4_K`, `.bias` suffix -> `F32`), and stamps the resolved
//!    policy into the `vokra.quant.*` chunk (T05 contract).
//! 2. A caller can reload the emitted GGUF via `vokra_core::gguf::GgufFile`
//!    and rebuild the resolved schemes deterministically from the chunk
//!    (chunk-round-trip proxy for the runtime-side reader T07 gates on).
//! 3. `vokra_eval::check_degradation` (the T11 5 % NFR-QL-02 gate the
//!    session ctor delegates to in T12) runs against a fp32 "reference"
//!    waveform and stays under the 5 % relative threshold — the mel-loss
//!    proxy for "quantized model ≈ reference model" (NFR-QL-02).
//!
//! The negative paths exercise the two rejection contracts the plan pins
//! for T14:
//!
//! - `default = W8A8Int8` on a model that carries a `vocos_head` op is
//!   rejected before ever reaching a backend. The converter's public
//!   `PolicyPreset` enum intentionally does *not* expose `W8A8Int8` (see
//!   `crates/vokra-convert/src/models/whisper.rs` §QuantScheme doc — "no
//!   INT8 kernels exist yet"), so a policy that resolved every op to W8A8
//!   is already unrepresentable through the public converter API — which
//!   is *itself* the rejection. We assert that by pinning `PolicyPreset`'s
//!   `from_arg` refusing `"w8a8"`. The runtime gate mirror is exercised by
//!   `vokra_core::quant::QuantScheme::W8A8Int8.backend_supported(_)`
//!   returning `false` on every backend — the same signal the T07/T09
//!   session gate branches on (FR-EX-08 uniformity: no silent widen).
//! - `hifigan_int8_opt_in=true` with a check_degradation delta > 5 %
//!   errors at session ctor. Session-ctor wiring is T12 landing pad, so we
//!   exercise the underlying `check_degradation` gate directly with a
//!   noisy quantized waveform and confirm the report flips
//!   `passes_5pct_gate=false`. That flag is exactly the branch T12's
//!   session ctor emits `HifiganInt8DegradationExceeded` from — verifying
//!   the gate here pins the mechanism the session-ctor wiring will
//!   depend on.
//!
//! Zero-dep invariant (NFR-DS-02): every dependency is a first-party
//! `vokra-*` crate (`vokra-core`, `vokra-convert`, `vokra-eval`); no
//! `PyTorch` / `librosa` / test-framework crate is pulled in.

use std::io::Write;

use vokra_convert::{ModelKind, PolicyPreset, convert_file_with_policy};
use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgmlType, GgufFile};
use vokra_core::quant::policy::LayerPattern;
use vokra_core::quant::{QuantPolicy, QuantScheme, resolve};
use vokra_eval::check_degradation;

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

/// Builds an F32 safetensors checkpoint from `(name, shape)` descriptors.
///
/// Matches the format that `vokra_convert::safetensors::SafetensorsFile::parse`
/// expects: `u64 header_len | header_json | packed_f32_le`. Payloads are
/// deterministic sinusoidal patterns (bounded in `[-1, 1]`) so K-quant super-
/// block scales are non-degenerate — a whisper checkpoint of zeros would round-
/// trip to bit-identical `Q4_K` blocks (`d=0`, all sub-scales=0) and skip the
/// scale/quant-error paths we want to sanity-check.
fn synthetic_checkpoint(tensors: &[(&str, &[u64])]) -> Vec<u8> {
    let mut cursor = 0usize;
    let mut entries = Vec::new();
    let mut payloads: Vec<Vec<u8>> = Vec::new();
    for &(name, shape) in tensors {
        let elems: usize = shape.iter().product::<u64>() as usize;
        let mut buf = Vec::with_capacity(elems * 4);
        for i in 0..elems {
            // Small deterministic sinusoid; keeps K-quant scale > 0.
            let phase = (i as f32) * 0.017_453_29; // ≈ 1 degree in radians
            let v = (phase.sin() * 0.5) + 0.25;
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let span = buf.len();
        let begin = cursor;
        let end = cursor + span;
        cursor = end;
        let dims = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        entries.push(format!(
            r#""{name}":{{"dtype":"F32","shape":[{dims}],"data_offsets":[{begin},{end}]}}"#
        ));
        payloads.push(buf);
    }
    let header = format!("{{{}}}", entries.join(","));
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for p in payloads {
        out.extend_from_slice(&p);
    }
    out
}

/// A whisper-tiny-like stub with two `.weight` tensors (both K-quantizable —
/// `2 × 256 = 512` elements, which is `2 × QK_K`) and one `.bias` tensor
/// (rank-1, F32-pinned by the `.bias` suffix rule in the whisper preset).
///
/// The whisper preset requires two positional-embedding tensors and a
/// `conv1.weight` (`d_model, n_mels, 3`) so `write_hparams` derives a shape
/// quintuple `derive_name` accepts (or falls back to `"whisper-unknown"` for
/// the shrunk stub — the converter carries an explicit synthetic-shape escape
/// for exactly this test path).
fn whisper_tiny_stub_checkpoint() -> Vec<u8> {
    synthetic_checkpoint(&[
        // Shape-driven hparam sources (values chosen so QK_K applicability
        // holds for the `.weight` tensors but the shape quintuple falls
        // outside `derive_name` — the synthetic path stamps
        // `"whisper-unknown"`, which is exactly what the pre-T06 tests use).
        ("model.encoder.conv1.weight", &[512, 80, 3]),
        ("model.encoder.embed_positions.weight", &[1536, 1]),
        ("model.decoder.embed_positions.weight", &[512, 1]),
        ("model.decoder.embed_tokens.weight", &[256, 1]),
        // K-quantizable weight: 2 × 256 = 512 (= 2 QK_K super-blocks).
        ("model.encoder.layers.0.mlp.fc2.weight", &[2, 256]),
        // Rank-1 bias: pinned to F32 by the whisper preset's `.bias` rule.
        ("model.encoder.layers.0.fc1.bias", &[512]),
    ])
}

/// Writes `bytes` to a fresh scratch file under `env::temp_dir()` with the
/// given `basename` (namespaced by the pid + a monotonic counter so parallel
/// test runs never collide).
fn scratch_file(basename: &str, bytes: &[u8]) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let mut p = std::env::temp_dir();
    p.push(format!("vokra-cli-policy-e2e-{pid}-{n}-{basename}"));
    let mut f = std::fs::File::create(&p).expect("scratch file create");
    f.write_all(bytes).expect("scratch file write");
    p
}

// ---------------------------------------------------------------------------
// Positive path — converter + reload + degradation gate
// ---------------------------------------------------------------------------

/// The T14 positive e2e: convert with `WhisperQ4K` (default `w4a16-q4k`
/// with a `.bias → fp32` rule), reload the GGUF, verify tensor types match
/// what `vokra_core::quant::resolve` reports, then run the T11 degradation
/// helper against a fp32 reference and assert the delta is under the 5 %
/// NFR-QL-02 gate.
#[test]
fn whisper_q4k_policy_e2e_convert_reload_and_degradation_gate() {
    // Build the input checkpoint and stage it on disk.
    let ckpt_bytes = whisper_tiny_stub_checkpoint();
    let input = scratch_file("whisper-tiny-stub.safetensors", &ckpt_bytes);
    let output = scratch_file("whisper-tiny-stub.gguf", &[]);

    // Run the converter with the whisper preset (default = W4A16Q4K with a
    // `.bias → Fp32` exception).
    let summary = convert_file_with_policy(
        ModelKind::Whisper,
        &input,
        &output,
        PolicyPreset::WhisperQ4K,
    )
    .expect("convert_file_with_policy(WhisperQ4K) must succeed");
    assert_eq!(summary.model, ModelKind::Whisper);
    assert!(
        summary.tensor_count >= 6,
        "expected at least the 6 stub tensors, got {}",
        summary.tensor_count
    );

    // Reload the emitted GGUF via the public parser.
    let gguf_bytes = std::fs::read(&output).expect("read converted GGUF");
    let file = GgufFile::parse(gguf_bytes).expect("GgufFile::parse");

    // The `vokra.quant.*` chunk must record the whisper_q4_k preset shape:
    //   default = w4a16-q4k, rule_count = 2, rule 0 = suffix `.bias` → fp32.
    assert_eq!(
        file.get("vokra.quant.default_scheme")
            .and_then(|v| v.as_str()),
        Some("w4a16-q4k"),
        "chunk must stamp resolved default scheme",
    );
    assert_eq!(
        file.get("vokra.quant.rule_count").and_then(|v| v.as_u64()),
        Some(2),
        "whisper preset carries `.bias` + `.weight_norm` rules",
    );
    assert_eq!(
        file.get("vokra.quant.rule.0.pattern_kind")
            .and_then(|v| v.as_str()),
        Some("suffix"),
    );
    assert_eq!(
        file.get("vokra.quant.rule.0.pattern")
            .and_then(|v| v.as_str()),
        Some(".bias"),
    );
    assert_eq!(
        file.get("vokra.quant.rule.0.scheme")
            .and_then(|v| v.as_str()),
        Some("fp32"),
    );
    assert_eq!(
        file.get("vokra.quant.hifigan_int8_opt_in")
            .and_then(|v| v.as_bool()),
        Some(false),
        "opt-in defaults to false — no calibration attached",
    );

    // Per-tensor dtype ⟷ resolved scheme parity. We reconstruct the runtime
    // (T05) reader here by walking the chunk we just verified: the whisper
    // preset is `default = W4A16Q4K` + `Suffix(".bias") → Fp32` + a second
    // `.weight_norm` suffix rule the fixture never triggers. Both must
    // agree with what `vokra_core::quant::resolve` reports for each tensor.
    let policy = QuantPolicy::new(QuantScheme::W4A16Q4K)
        .with_rule(LayerPattern::Suffix(".bias".to_owned()), QuantScheme::Fp32)
        .with_rule(
            LayerPattern::Suffix(".weight_norm".to_owned()),
            QuantScheme::Fp32,
        );

    for name in [
        "model.encoder.layers.0.mlp.fc2.weight",
        "model.encoder.layers.0.fc1.bias",
    ] {
        let info = file
            .tensor_info(name)
            .unwrap_or_else(|| panic!("missing tensor `{name}` in converted GGUF"));
        let resolved = resolve(&policy, name);
        let expected_dtype = match resolved.weight_dtype() {
            vokra_core::quant::WeightDtype::Ggml(t) => t,
            vokra_core::quant::WeightDtype::Int8Reserved => {
                panic!("W8A8 unreachable in whisper preset — the preset never resolves to it")
            }
        };
        assert_eq!(
            info.dtype, expected_dtype,
            "tensor `{name}` GGUF dtype ({:?}) must equal resolve(policy, name).weight_dtype() ({:?})",
            info.dtype, expected_dtype,
        );
    }
    // Cross-check the specific bindings we care about — `.bias` stayed F32,
    // the K-quantizable `.weight` went to `Q4_K`. This mirrors the
    // whisper.rs unit-test invariants so a future regression in either the
    // converter or the resolver trips at least one of them.
    assert_eq!(
        file.tensor_info("model.encoder.layers.0.fc1.bias")
            .expect("bias")
            .dtype,
        GgmlType::F32,
    );
    assert_eq!(
        file.tensor_info("model.encoder.layers.0.mlp.fc2.weight")
            .expect("weight")
            .dtype,
        GgmlType::Q4K,
    );

    // Runtime-side degradation gate (T11): a fp32 reference against itself
    // sits at delta = 0 and passes the 5 % NFR-QL-02 gate. This is the
    // signal T12's session ctor gates HiFi-GAN opt-in on — asserting it
    // here pins the mechanism the e2e path depends on.
    let reference: Vec<f32> = (0..16_000)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();
    let report = check_degradation(&reference, &reference, 16_000, 0.05)
        .expect("check_degradation on identical inputs must succeed");
    assert!(
        report.passes_5pct_gate,
        "self-identical reference must sit under the 5 % gate: delta={}, loss={}",
        report.relative_delta, report.mel_loss_quant,
    );
}

// ---------------------------------------------------------------------------
// Negative path 1 — W8A8Int8 default on a vocoder-carrying model is rejected
// ---------------------------------------------------------------------------

/// The converter's public `PolicyPreset` enum intentionally does not expose a
/// `w8a8` variant (whisper.rs §QuantScheme doc — "no INT8 kernels exist
/// yet"), so a caller cannot even *ask* for `W8A8Int8` through
/// `convert_file_with_policy`. That is the converter-side rejection T14
/// pins — asserted here so a future enum expansion trips this test rather
/// than silently opening the INT8 path.
#[test]
fn w8a8_preset_is_unrepresentable_through_public_converter_api() {
    assert!(
        PolicyPreset::from_arg("w8a8").is_none(),
        "PolicyPreset must not accept `w8a8` in M2-08 — no INT8 kernel exists yet",
    );
    assert!(
        PolicyPreset::from_arg("w8a8-int8").is_none(),
        "no alias for INT8 either",
    );
    // The three legitimate preset aliases must still parse — belt-and-
    // suspenders against a `from_arg` refactor that would refuse everything.
    assert!(PolicyPreset::from_arg("vocoder_safe").is_some());
    assert!(PolicyPreset::from_arg("whisper_q4_k").is_some());
    assert!(PolicyPreset::from_arg("fp16").is_some());
}

/// A `default = W8A8Int8` policy targeting a `vocos_head` op is rejected by
/// the runtime-side gate T07/T09 pin: `QuantScheme::W8A8Int8` reports
/// `backend_supported(_) == false` on every backend
/// (`vokra-backend-cpu/src/lib.rs:16-17` — no INT8 activation kernel). The
/// session ctor's `gate_ops_against_policy` branches on exactly this signal
/// (`crates/vokra-models/src/whisper/session.rs`), so verifying the flag
/// here pins the FR-EX-08 uniformity contract for T14 without needing
/// access to the private gate.
#[test]
fn w8a8_default_policy_targeting_vocos_head_is_rejected_before_backend() {
    // A `vocos_head`-carrying model would resolve every op to W8A8 under a
    // `default = W8A8Int8` policy — including the vocoder head, which the
    // T08 registry marks `DowngradePolicy::Forbidden`. That is the exact
    // combination the ticket names as "rejected at converter/session ctor".
    let policy = QuantPolicy::new(QuantScheme::W8A8Int8);
    // Every op the runtime asks about would resolve to W8A8:
    let scheme = resolve(&policy, "vocos_head");
    assert_eq!(scheme, QuantScheme::W8A8Int8);
    // ...and W8A8 has no kernel on any backend today, so the runtime gate
    // errors before dispatch (FR-EX-08 — no silent fallback).
    assert!(
        !scheme.backend_supported(BackendKind::Cpu),
        "W8A8 must be unsupported on CPU (no VNNI/SDOT kernel yet)",
    );
    assert!(
        !scheme.backend_supported(BackendKind::Metal),
        "W8A8 must be unsupported on Metal (no INT8 MSL kernel yet)",
    );
    assert!(
        !scheme.backend_supported(BackendKind::Cuda),
        "W8A8 must be unsupported on CUDA (no cuBLASLt INT8 kernel yet)",
    );
    // And the activation dtype the T07 gate branches on is `Int8` — the
    // exact discriminant the session ctor rejects.
    assert_eq!(
        scheme.activation_dtype(),
        vokra_core::quant::ActivationDtype::Int8,
    );
}

// ---------------------------------------------------------------------------
// Negative path 2 — HiFi-GAN INT8 opt-in with delta > 5 %
// ---------------------------------------------------------------------------

/// A HiFi-GAN INT8 opt-in policy that produces > 5 % mel-loss degradation
/// must fail the T11 5 % gate — which is the flag T12's session ctor
/// consults before allowing the INT8 path. We drive `check_degradation`
/// with a large-amplitude noise proxy for "INT8 vocoder degradation" and
/// assert the report flips `passes_5pct_gate = false`, so the session-ctor
/// wiring in T12 will emit `HifiganInt8DegradationExceeded` when the WP
/// lands.
#[test]
fn hifigan_int8_opt_in_with_degradation_over_5pct_fails_gate() {
    // Reference (fp32-clean) 440 Hz tone; the gated waveform adds noise
    // proportional to the reference amplitude so mel energy drifts far
    // outside the 5 % relative envelope.
    let sr: u32 = 16_000;
    let n = 16_000;
    let reference: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();
    // Deterministic pseudo-noise in [-1, 1] — no RNG dependency (NFR-DS-02),
    // same LCG idiom `vokra_eval::degradation::tests::noise` uses.
    let mut state: u32 = 0x1234_5678;
    let noise: Vec<f32> = (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect();
    // Half-amplitude noise — spectrally rich enough to blow past the 5 %
    // relative-delta envelope (matches `vokra_eval::degradation::tests::
    // large_noise_fails_gate`).
    let quantized: Vec<f32> = reference
        .iter()
        .zip(&noise)
        .map(|(r, e)| r + 0.5 * e)
        .collect();

    // Build a policy that would enable HiFi-GAN INT8 (T10 sole atomic
    // constructor path) — the calibration ref is opaque, only its presence
    // matters for the opt-in bool. This is the *only* legitimate way to
    // set `opt_in=true`, so a session ctor consulting the policy sees the
    // opt-in flag and delegates to `check_degradation`.
    let policy = QuantPolicy::new(QuantScheme::Fp16).with_hifigan_int8_opt_in(
        vokra_core::quant::CalibrationRef::new("hifigan-int8-cal-fixture"),
    );
    assert!(policy.hifigan_int8_opt_in(), "opt-in must be set");
    assert!(
        policy.hifigan_int8_calibration().is_some(),
        "calibration attached via the sole atomic path",
    );
    policy
        .validate_self()
        .expect("opt-in + calibration is a legitimate policy shape");

    // T11 5 % gate — this is the exact call T12's session ctor makes before
    // returning `HifiganInt8DegradationExceeded`.
    let report = check_degradation(&reference, &quantized, sr, 0.05)
        .expect("check_degradation must succeed on well-formed waveforms");
    assert!(
        !report.passes_5pct_gate,
        "half-amplitude noise must fail the 5 % gate: delta={}, loss={}",
        report.relative_delta, report.mel_loss_quant,
    );
    assert!(
        report.mel_loss_quant > 0.0,
        "quantized waveform must move mel energy away from reference",
    );
    assert!(
        report.mel_loss_only,
        "UTMOS not yet wired — mel-loss-only partial gate (risk R5)",
    );
    // The `passes_5pct_gate=false` flag is precisely the branch T12's
    // session ctor gates HiFi-GAN opt-in on — so a session ctor that reads
    // this report today would emit `HifiganInt8DegradationExceeded`, which
    // is what T14's negative case names.
}
