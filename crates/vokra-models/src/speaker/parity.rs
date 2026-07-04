//! CAM++ speaker-encoder numerical parity vs the onnxruntime reference (M0-08,
//! NFR-QL-01, FP32 `atol = 0.01`).
//!
//! The oracle is `tests/parity/camplus/gen_reference.py`, which feeds a FIXED,
//! seeded fbank `[1, 200, 80]` through the reference `campplus.onnx` under
//! onnxruntime and dumps the final 192-d embedding plus intermediate node
//! outputs (`post_fcm_reshape`, `post_tdnn`, `post_block1/2/3`, `post_stats`)
//! that localize any divergence over the 3206-node graph. The committed
//! fixtures make the fbank→embedding NETWORK fully validatable here; the
//! audio→fbank front-end is validated separately once a Kaldi-fbank oracle
//! exists.
//!
//! The 27 MB CAM++ GGUF is not committed, so these tests are gated on
//! `VOKRA_CAMPLUS_GGUF` and skip cleanly when unset (CI stays green). Run with:
//!
//! ```text
//! VOKRA_CAMPLUS_GGUF=campplus.gguf cargo test -p vokra-models camplus
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use super::SpeakerEncoder;

/// FP32 parity bound (design: each stage + final embedding atol 0.01).
const ATOL: f32 = 0.01;
/// Reference input frame count (`manifest.txt`: `input_frames = 200`).
const T: usize = 200;

fn fixtures_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = <repo>/crates/vokra-models.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("parity")
        .join("camplus")
}

/// Reads a little-endian f32 fixture file.
fn read_f32(name: &str) -> Vec<f32> {
    let path = fixtures_dir().join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(bytes.len() % 4, 0, "{name}: not a whole number of f32");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Loads the encoder named by `$VOKRA_CAMPLUS_GGUF`, or `None` to skip (CI).
fn load_encoder() -> Option<SpeakerEncoder> {
    let path = std::env::var("VOKRA_CAMPLUS_GGUF").ok()?;
    Some(SpeakerEncoder::from_path(&path).expect("load CAM++ GGUF"))
}

/// Largest absolute difference between two equal-length slices.
fn max_abs_diff(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(
        got.len(),
        want.len(),
        "length mismatch {} vs {}",
        got.len(),
        want.len()
    );
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// Asserts a captured stage matches its fixture within [`ATOL`], printing the
/// peak error.
fn check(stages: &HashMap<String, Vec<f32>>, stage: &str, fixture: &str) {
    let got = stages
        .get(stage)
        .unwrap_or_else(|| panic!("stage `{stage}` not captured"));
    let want = read_f32(fixture);
    let d = max_abs_diff(got, want.as_slice());
    eprintln!(
        "camplus {stage}: max|Δ|={d:.6} len={} (atol={ATOL})",
        got.len()
    );
    assert!(d <= ATOL, "camplus {stage} parity {d} exceeds atol {ATOL}");
}

#[test]
fn camplus_network_parity_all_stages() {
    let Some(enc) = load_encoder() else {
        eprintln!("skipping CAM++ parity: set VOKRA_CAMPLUS_GGUF to run");
        return;
    };
    let fbank = read_f32("input_fbank.f32");
    assert_eq!(fbank.len(), T * 80, "fbank fixture is [1, {T}, 80]");

    // Capture every intermediate; the first stage mismatch localizes the fault.
    let mut stages: HashMap<String, Vec<f32>> = HashMap::new();
    let emb = enc
        .run(&fbank, T, |name, data| {
            stages.insert(name.to_owned(), data.to_vec());
        })
        .expect("CAM++ forward");

    check(&stages, "post_fcm_reshape", "post_fcm_reshape.f32");
    check(&stages, "post_tdnn", "post_tdnn.f32");
    check(&stages, "post_block1", "post_block1.f32");
    check(&stages, "post_block2", "post_block2.f32");
    check(&stages, "post_block3", "post_block3.f32");
    check(&stages, "post_stats", "post_stats.f32");
    check(&stages, "embedding", "embedding.f32");

    // The public `embed` API must return the same 192-d vector.
    let want = read_f32("embedding.f32");
    assert_eq!(emb.len(), want.len());
    let api = enc.embed(&fbank, T).expect("CAM++ embed");
    assert!(max_abs_diff(&api, &want) <= ATOL, "embed API parity");
}

#[test]
fn camplus_embed_rejects_wrong_length() {
    let Some(enc) = load_encoder() else {
        eprintln!("skipping CAM++ length check: set VOKRA_CAMPLUS_GGUF to run");
        return;
    };
    // fbank length must equal t * 80.
    assert!(enc.embed(&[0.0; 79], 1).is_err());
    assert!(enc.embed(&[], 0).is_err());
}

/// Metal-vs-CPU parity for the GEMM-dominated CAM++ forward (M2-01 Phase 3):
/// the same encoder run through the `Compute::Metal` GEMM path must match the
/// `Compute::Cpu` path within the FP32 bound (NFR-QL-01, `atol = 0.01`). Every
/// CAM++ conv is lowered to a GEMM, so this exercises the whole network on the
/// GPU. Doubly gated — needs the CAM++ GGUF (`VOKRA_CAMPLUS_GGUF`) **and** a real
/// Metal device — and skips cleanly (no silent CPU substitute) when either is
/// absent. Run with:
///
/// ```text
/// VOKRA_CAMPLUS_GGUF=campplus.gguf cargo test -p vokra-models --features metal camplus_metal -- --nocapture
/// ```
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
fn camplus_metal_matches_cpu() {
    use crate::compute::{Compute, HotOp};
    use vokra_core::BackendKind;

    let Some(enc) = load_encoder() else {
        eprintln!("skipping CAM++ Metal parity: set VOKRA_CAMPLUS_GGUF to run");
        return;
    };
    // Device-gated: build a GEMM-covering Metal dispatcher, or skip if there is
    // no Metal device (an explicit BackendUnavailable — never a CPU fall back).
    let metal = match Compute::for_backend(BackendKind::Metal, &[HotOp::Gemm]) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping CAM++ Metal parity (no Metal device): {e}");
            return;
        }
    };
    assert_eq!(metal.backend_name(), "metal");

    let fbank = read_f32("input_fbank.f32");
    assert_eq!(fbank.len(), T * 80, "fbank fixture is [1, {T}, 80]");

    // Same weights, two backends: the Metal GEMM path must match the CPU path.
    let cpu_emb = enc
        .run_with(&Compute::cpu(), &fbank, T, |_, _| {})
        .expect("CPU forward");
    let metal_emb = enc
        .run_with(&metal, &fbank, T, |_, _| {})
        .expect("Metal forward");

    let d = max_abs_diff(&cpu_emb, &metal_emb);
    eprintln!(
        "camplus Metal vs CPU: max|Δ|={d:.6} len={} (atol={ATOL})",
        cpu_emb.len()
    );
    assert!(
        d <= ATOL,
        "camplus Metal vs CPU parity {d} exceeds atol {ATOL}"
    );

    // And the GPU embedding still matches the committed onnxruntime fixture.
    let reference = read_f32("embedding.f32");
    let dref = max_abs_diff(&metal_emb, &reference);
    assert!(
        dref <= ATOL,
        "camplus Metal vs reference parity {dref} exceeds atol {ATOL}"
    );
}
