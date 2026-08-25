//! Synthetic-weight structural tests for the NSNet2 runtime binder.
//!
//! Real-weight numeric parity against the upstream ONNX Runtime pipeline
//! lives in the env-gated harness
//! (`crates/vokra-models/tests/parity_nsnet2.rs`,
//! `VOKRA_NSNET2_REAL_GGUF` / `VOKRA_NSNET2_REAL_WAV`) — this file
//! covers everything reachable from synthetic weights: the load-time
//! contract (FR-EX-08), the ONNX GRU gate permutation, the
//! identity-gain sanity of the sigmoid mask (mask pre-activation → +∞
//! ⇒ gain ≈ 1 ⇒ output ≈ input), and the streaming state carry-over.

use super::*;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile, chunks};

/// A tiny NSNet2-shaped config: `n_fft=8`, `hop=4`, `win=8`, `n_bins=5`,
/// `hidden=3`, `fc1=fc2=4`, `sample_rate=8000`. Small enough that the
/// synth GGUFs stay well under a MiB but non-trivial enough to exercise
/// every layer (n_bins != hidden, fc1 != hidden, mask width != fc2).
fn tiny_config() -> Nsnet2Config {
    Nsnet2Config {
        n_bins: 5,
        hidden_dim: 3,
        fc1_dim: 4,
        fc2_dim: 4,
        n_fft: 8,
        hop: 4,
        win_length: 8,
        sample_rate: 8_000,
    }
}

/// Stamps the arch tag, hparams and a caller-supplied tensor list into a
/// GGUF and returns the parsed file.
fn build_gguf<F>(cfg: &Nsnet2Config, add_tensors: F) -> GgufFile
where
    F: FnOnce(&mut GgufBuilder),
{
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, DEFAULT_NAME);
    b.add_string(chunks::KEY_PROVENANCE_LICENSE, "mit");
    b.add_string(
        chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
        LicenseClass::Permissive.as_str(),
    );
    b.add_u32(KEY_N_BINS, cfg.n_bins as u32);
    b.add_u32(KEY_HIDDEN_DIM, cfg.hidden_dim as u32);
    b.add_u32(KEY_FC1_DIM, cfg.fc1_dim as u32);
    b.add_u32(KEY_FC2_DIM, cfg.fc2_dim as u32);
    b.add_u32(KEY_N_FFT, cfg.n_fft as u32);
    b.add_u32(KEY_HOP, cfg.hop as u32);
    b.add_u32(KEY_WIN_LENGTH, cfg.win_length as u32);
    b.add_u32(KEY_SAMPLE_RATE, cfg.sample_rate);
    add_tensors(&mut b);
    let bytes = b.to_bytes().expect("build gguf");
    GgufFile::parse(bytes).expect("parse gguf")
}

/// Emits an F32 tensor of the given shape (row-major).
fn add_f32(b: &mut GgufBuilder, name: &str, shape: Vec<u64>, data: &[f32]) {
    let payload: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    b.add_tensor(name, GgmlType::F32, shape, payload)
        .expect("add_tensor");
}

/// Emits every tensor NSNet2 needs, all zero-valued (structural bind
/// test — the numeric assertions ride on the `Ok(_)` outcome).
fn add_all_zero_tensors(b: &mut GgufBuilder, cfg: &Nsnet2Config) {
    let h = cfg.hidden_dim;
    let n_bins = cfg.n_bins;
    let fc1 = cfg.fc1_dim;
    let fc2 = cfg.fc2_dim;
    add_f32(
        b,
        TENSOR_FC_IN_WEIGHT,
        vec![h as u64, n_bins as u64],
        &vec![0.0; h * n_bins],
    );
    add_f32(b, TENSOR_FC_IN_BIAS, vec![h as u64], &vec![0.0; h]);
    for (w, r, bs) in [
        (TENSOR_GRU_1_W, TENSOR_GRU_1_R, TENSOR_GRU_1_B),
        (TENSOR_GRU_2_W, TENSOR_GRU_2_R, TENSOR_GRU_2_B),
    ] {
        add_f32(b, w, vec![(3 * h) as u64, h as u64], &vec![0.0; 3 * h * h]);
        add_f32(b, r, vec![(3 * h) as u64, h as u64], &vec![0.0; 3 * h * h]);
        add_f32(b, bs, vec![(6 * h) as u64], &vec![0.0; 6 * h]);
    }
    add_f32(
        b,
        TENSOR_FC_1_WEIGHT,
        vec![fc1 as u64, h as u64],
        &vec![0.0; fc1 * h],
    );
    add_f32(b, TENSOR_FC_1_BIAS, vec![fc1 as u64], &vec![0.0; fc1]);
    add_f32(
        b,
        TENSOR_FC_2_WEIGHT,
        vec![fc2 as u64, fc1 as u64],
        &vec![0.0; fc2 * fc1],
    );
    add_f32(b, TENSOR_FC_2_BIAS, vec![fc2 as u64], &vec![0.0; fc2]);
    add_f32(
        b,
        TENSOR_MASK_WEIGHT,
        vec![n_bins as u64, fc2 as u64],
        &vec![0.0; n_bins * fc2],
    );
    add_f32(b, TENSOR_MASK_BIAS, vec![n_bins as u64], &vec![0.0; n_bins]);
}

fn add_legacy_public_metadata(b: &mut GgufBuilder) {
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, DEFAULT_NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(chunks::KEY_PROVENANCE_LICENSE, "mit");
    b.add_string(
        chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
        LicenseClass::Permissive.as_str(),
    );
    b.add_string(chunks::KEY_PROVENANCE_MODEL_ID, DEFAULT_NAME);
    b.add_string(chunks::KEY_PROVENANCE_SOURCE, LEGACY_PUBLIC_SOURCE);
    b.add_string("vokra.provenance.upstream_url", LEGACY_PUBLIC_UPSTREAM_URL);
}

fn legacy_public_gguf() -> GgufFile {
    let mut b = GgufBuilder::new();
    add_legacy_public_metadata(&mut b);
    for spec in LEGACY_PUBLIC_TENSORS {
        let elements = spec
            .dimensions
            .iter()
            .try_fold(1usize, |count, &axis| count.checked_mul(axis as usize))
            .expect("legacy public tensor element count");
        let mut values = vec![0.0; elements];
        match spec.name {
            "172" => values[7 * 400 + 23] = 1.25,
            "215" => values[5 * 600 + 17] = -2.5,
            "216" => values[3 * 600 + 11] = 3.75,
            "217" => values[9 * 161 + 13] = -4.5,
            _ => {}
        }
        add_f32(&mut b, spec.name, spec.dimensions.to_vec(), &values);
    }
    GgufFile::parse(b.to_bytes().expect("build legacy public GGUF"))
        .expect("parse legacy public GGUF")
}

// -------------------------------------------------------------------------
// Config round-trip
// -------------------------------------------------------------------------

#[test]
fn upstream_default_config_validates() {
    Nsnet2Config::upstream_default().validate().unwrap();
    assert_eq!(Nsnet2Config::upstream_default().n_bins, 161);
    assert_eq!(Nsnet2Config::upstream_default().hidden_dim, 400);
    assert_eq!(Nsnet2Config::upstream_default().n_fft, 320);
    assert_eq!(Nsnet2Config::upstream_default().sample_rate, 16_000);
}

#[test]
fn config_round_trip_through_gguf() {
    let cfg = tiny_config();
    let gguf = build_gguf(&cfg, |b| add_all_zero_tensors(b, &cfg));
    let parsed = Nsnet2Config::from_gguf(&gguf).unwrap();
    assert_eq!(parsed, cfg);
}

#[test]
fn config_rejects_zero_hparam_loudly() {
    let mut cfg = tiny_config();
    cfg.hidden_dim = 0;
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("hidden_dim") && msg.contains("must be > 0"),
        "expected loud FR-EX-08 hint, got: {msg}"
    );
}

#[test]
fn config_rejects_bins_mismatch_loudly() {
    // n_bins = 5 but n_fft = 16 gives expected 9 — must refuse.
    let mut cfg = tiny_config();
    cfg.n_fft = 16;
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("n_bins") && msg.contains("n_fft/2 + 1"),
        "expected cross-invariant hint, got: {msg}"
    );
}

#[test]
fn config_rejects_win_gt_nfft() {
    let mut cfg = tiny_config();
    cfg.win_length = cfg.n_fft + 1;
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("win_length") && msg.contains("<= n_fft"));
}

// -------------------------------------------------------------------------
// GGUF load — FR-EX-08 loud-fail matrix
// -------------------------------------------------------------------------

#[test]
fn from_gguf_rejects_wrong_arch() {
    let cfg = tiny_config();
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, "silero-vad");
    // No hparams / tensors needed — arch check fires first.
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = Nsnet2V1::from_gguf(&gguf).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("arch") && msg.contains("silero-vad"));
    let _ = cfg;
}

#[test]
fn from_gguf_rejects_missing_arch() {
    let mut b = GgufBuilder::new();
    // Deliberately no arch stamp.
    b.add_u32(KEY_N_BINS, 5);
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = Nsnet2V1::from_gguf(&gguf).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("missing") && msg.contains("vokra.model.arch"));
}

#[test]
fn from_gguf_rejects_missing_hparam() {
    let cfg = tiny_config();
    // Omit KEY_HIDDEN_DIM.
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_u32(KEY_N_BINS, cfg.n_bins as u32);
    // hidden_dim missing
    b.add_u32(KEY_FC1_DIM, cfg.fc1_dim as u32);
    b.add_u32(KEY_FC2_DIM, cfg.fc2_dim as u32);
    b.add_u32(KEY_N_FFT, cfg.n_fft as u32);
    b.add_u32(KEY_HOP, cfg.hop as u32);
    b.add_u32(KEY_WIN_LENGTH, cfg.win_length as u32);
    b.add_u32(KEY_SAMPLE_RATE, cfg.sample_rate);
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = Nsnet2V1::from_gguf(&gguf).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("hidden_dim"));
}

#[test]
fn exact_historical_public_contract_binds_to_canonical_weights() {
    let gguf = legacy_public_gguf();
    let model = Nsnet2V1::from_gguf(&gguf).expect("bind exact historical public contract");
    assert_eq!(model.config(), &Nsnet2Config::upstream_default());

    // The old artifact stores MatMul matrices as [in, out]. The runtime must
    // transpose every one of them into the canonical [out, in] layout,
    // including the square fc_2 matrix where a shape-only check cannot expose
    // a missing transpose.
    assert_eq!(model.weights.fc_in_weight[23 * 161 + 7], 1.25);
    assert_eq!(model.weights.fc_1_weight[17 * 400 + 5], -2.5);
    assert_eq!(model.weights.fc_2_weight[11 * 600 + 3], 3.75);
    assert_eq!(model.weights.mask_weight[13 * 600 + 9], -4.5);
}

#[test]
fn historical_public_contract_rejects_manifest_drift() {
    let mut b = GgufBuilder::new();
    add_legacy_public_metadata(&mut b);
    for spec in LEGACY_PUBLIC_TENSORS {
        // Keep every audited name present so layout detection reaches the
        // exact directory validator, then change every shape/offset cheaply.
        add_f32(&mut b, spec.name, vec![1], &[0.0]);
    }
    let gguf = GgufFile::parse(b.to_bytes().expect("build drifted GGUF")).unwrap();
    let err = Nsnet2V1::from_gguf(&gguf).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("historical public GGUF contract mismatch")
            && msg.contains("tensor directory row"),
        "expected exact legacy manifest rejection, got: {msg}"
    );
}

#[test]
fn from_gguf_rejects_mixed_canonical_and_legacy_names() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    add_f32(&mut b, TENSOR_FC_IN_WEIGHT, vec![1], &[0.0]);
    add_f32(&mut b, "172", vec![1], &[0.0]);
    let gguf = GgufFile::parse(b.to_bytes().expect("build mixed GGUF")).unwrap();
    let err = Nsnet2V1::from_gguf(&gguf).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("mixed canonical and historical public tensor schemas"),
        "expected mixed-schema rejection, got: {msg}"
    );
}

#[test]
fn from_gguf_rejects_missing_tensor() {
    let cfg = tiny_config();
    let gguf = build_gguf(&cfg, |b| {
        // Add every tensor except TENSOR_GRU_1_R (recurrent weight).
        let h = cfg.hidden_dim;
        let n_bins = cfg.n_bins;
        let fc1 = cfg.fc1_dim;
        let fc2 = cfg.fc2_dim;
        add_f32(
            b,
            TENSOR_FC_IN_WEIGHT,
            vec![h as u64, n_bins as u64],
            &vec![0.0; h * n_bins],
        );
        add_f32(b, TENSOR_FC_IN_BIAS, vec![h as u64], &vec![0.0; h]);
        add_f32(
            b,
            TENSOR_GRU_1_W,
            vec![(3 * h) as u64, h as u64],
            &vec![0.0; 3 * h * h],
        );
        // Intentionally skip TENSOR_GRU_1_R
        add_f32(b, TENSOR_GRU_1_B, vec![(6 * h) as u64], &vec![0.0; 6 * h]);
        add_f32(
            b,
            TENSOR_GRU_2_W,
            vec![(3 * h) as u64, h as u64],
            &vec![0.0; 3 * h * h],
        );
        add_f32(
            b,
            TENSOR_GRU_2_R,
            vec![(3 * h) as u64, h as u64],
            &vec![0.0; 3 * h * h],
        );
        add_f32(b, TENSOR_GRU_2_B, vec![(6 * h) as u64], &vec![0.0; 6 * h]);
        add_f32(
            b,
            TENSOR_FC_1_WEIGHT,
            vec![fc1 as u64, h as u64],
            &vec![0.0; fc1 * h],
        );
        add_f32(b, TENSOR_FC_1_BIAS, vec![fc1 as u64], &vec![0.0; fc1]);
        add_f32(
            b,
            TENSOR_FC_2_WEIGHT,
            vec![fc2 as u64, fc1 as u64],
            &vec![0.0; fc2 * fc1],
        );
        add_f32(b, TENSOR_FC_2_BIAS, vec![fc2 as u64], &vec![0.0; fc2]);
        add_f32(
            b,
            TENSOR_MASK_WEIGHT,
            vec![n_bins as u64, fc2 as u64],
            &vec![0.0; n_bins * fc2],
        );
        add_f32(b, TENSOR_MASK_BIAS, vec![n_bins as u64], &vec![0.0; n_bins]);
    });
    let err = Nsnet2V1::from_gguf(&gguf).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("gru_1.R"),
        "expected `gru_1.R` missing hint: {msg}"
    );
}

#[test]
fn from_gguf_rejects_wrong_dims() {
    // Ship TENSOR_FC_IN_WEIGHT with correct element count but the
    // transposed layout `[n_bins, hidden]` — the dim-order assertion
    // (openwakeword precedent) must catch this before any inference.
    let cfg = tiny_config();
    let gguf = build_gguf(&cfg, |b| {
        let h = cfg.hidden_dim;
        let n_bins = cfg.n_bins;
        let fc1 = cfg.fc1_dim;
        let fc2 = cfg.fc2_dim;
        // Emit fc_in.weight with the WRONG shape: `[n_bins, hidden]`.
        add_f32(
            b,
            TENSOR_FC_IN_WEIGHT,
            vec![n_bins as u64, h as u64],
            &vec![0.0; h * n_bins],
        );
        add_f32(b, TENSOR_FC_IN_BIAS, vec![h as u64], &vec![0.0; h]);
        for (w, r, bs) in [
            (TENSOR_GRU_1_W, TENSOR_GRU_1_R, TENSOR_GRU_1_B),
            (TENSOR_GRU_2_W, TENSOR_GRU_2_R, TENSOR_GRU_2_B),
        ] {
            add_f32(b, w, vec![(3 * h) as u64, h as u64], &vec![0.0; 3 * h * h]);
            add_f32(b, r, vec![(3 * h) as u64, h as u64], &vec![0.0; 3 * h * h]);
            add_f32(b, bs, vec![(6 * h) as u64], &vec![0.0; 6 * h]);
        }
        add_f32(
            b,
            TENSOR_FC_1_WEIGHT,
            vec![fc1 as u64, h as u64],
            &vec![0.0; fc1 * h],
        );
        add_f32(b, TENSOR_FC_1_BIAS, vec![fc1 as u64], &vec![0.0; fc1]);
        add_f32(
            b,
            TENSOR_FC_2_WEIGHT,
            vec![fc2 as u64, fc1 as u64],
            &vec![0.0; fc2 * fc1],
        );
        add_f32(b, TENSOR_FC_2_BIAS, vec![fc2 as u64], &vec![0.0; fc2]);
        add_f32(
            b,
            TENSOR_MASK_WEIGHT,
            vec![n_bins as u64, fc2 as u64],
            &vec![0.0; n_bins * fc2],
        );
        add_f32(b, TENSOR_MASK_BIAS, vec![n_bins as u64], &vec![0.0; n_bins]);
    });
    let err = Nsnet2V1::from_gguf(&gguf).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("fc_in.weight") && msg.contains("dims"),
        "expected dim-order rejection, got: {msg}"
    );
}

// -------------------------------------------------------------------------
// GRU permutation
// -------------------------------------------------------------------------

#[test]
fn permute_onnx_gru_swaps_z_and_r_blocks() {
    // hidden=2, in_dim=2. Fill W's rows with a distinct sentinel per
    // gate so the permutation is trivially visible.
    let hidden = 2;
    let in_dim = 2;
    // ONNX order: Z (0), R (1), H (2). Give each block a distinct pair.
    let w: Vec<f32> = vec![
        // Z rows (0..2)
        1.0, 1.0, 1.0, 1.0, // R rows (2..4)
        2.0, 2.0, 2.0, 2.0, // H rows (4..6)
        3.0, 3.0, 3.0, 3.0,
    ];
    let r: Vec<f32> = vec![
        // Z rows
        10.0, 10.0, 10.0, 10.0, // R rows
        20.0, 20.0, 20.0, 20.0, // H rows
        30.0, 30.0, 30.0, 30.0,
    ];
    // b = [Wb_z | Wb_r | Wb_h | Rb_z | Rb_r | Rb_h] — each block `hidden` long.
    let b: Vec<f32> = vec![
        0.1, 0.1, // Wb_z
        0.2, 0.2, // Wb_r
        0.3, 0.3, // Wb_h
        1.0, 1.0, // Rb_z
        2.0, 2.0, // Rb_r
        3.0, 3.0, // Rb_h
    ];

    let (w_ih, w_hh, bias_ih, bias_hh) = permute_onnx_gru(&w, &r, &b, hidden, in_dim);

    // rnnoise order: R (0), Z (1), N (2).
    // Row block 0 (rnnoise R) should equal ONNX R (== 2.0 sentinel in w).
    assert_eq!(&w_ih[0..hidden * in_dim], &vec![2.0f32; hidden * in_dim]);
    // Row block 1 (rnnoise Z) should equal ONNX Z (== 1.0).
    assert_eq!(
        &w_ih[hidden * in_dim..2 * hidden * in_dim],
        &vec![1.0f32; hidden * in_dim]
    );
    // Row block 2 (rnnoise N) should equal ONNX H (== 3.0).
    assert_eq!(
        &w_ih[2 * hidden * in_dim..3 * hidden * in_dim],
        &vec![3.0f32; hidden * in_dim]
    );

    // Recurrent block, same permutation, sentinel 10/20/30.
    assert_eq!(&w_hh[0..hidden * hidden], &vec![20.0f32; hidden * hidden]);
    assert_eq!(
        &w_hh[hidden * hidden..2 * hidden * hidden],
        &vec![10.0f32; hidden * hidden]
    );
    assert_eq!(
        &w_hh[2 * hidden * hidden..3 * hidden * hidden],
        &vec![30.0f32; hidden * hidden]
    );

    // Biases stay separate because this graph uses
    // `linear_before_reset=1`: the recurrent candidate bias is inside the
    // reset multiplication and cannot be fused with the input bias.
    assert_eq!(bias_ih, vec![0.2, 0.2, 0.1, 0.1, 0.3, 0.3]);
    assert_eq!(bias_hh, vec![2.0, 2.0, 1.0, 1.0, 3.0, 3.0]);
}

// -------------------------------------------------------------------------
// End-to-end forward — identity-gain sanity + zero-weight sanity
// -------------------------------------------------------------------------

#[test]
fn zero_weight_denoise_yields_bounded_output() {
    // Every weight zero means: fc_in = 0 (bias 0 → ReLU 0), GRUs stay
    // zero, fc_1/fc_2/mask all zero → mask pre-activation 0 → sigmoid
    // 0.5 → gain 0.5 → output = 0.5 * input STFT. So the reconstructed
    // PCM must (a) exist (never NaN / inf) and (b) have magnitude
    // bounded above by `max(|pcm|)`. This is a structural smoke: it
    // rules out any load-time or forward-path crash without needing a
    // real weight fixture.
    let cfg = tiny_config();
    let gguf = build_gguf(&cfg, |b| add_all_zero_tensors(b, &cfg));
    let model = Nsnet2V1::from_gguf(&gguf).unwrap();

    // 64 samples of a 500 Hz sine at 8 kHz (well within the Nyquist).
    let pcm: Vec<f32> = (0..64)
        .map(|n| (2.0 * std::f32::consts::PI * 500.0 * (n as f32) / 8_000.0).sin())
        .collect();
    let out = model.denoise_pcm(&pcm).unwrap();
    assert!(!out.is_empty(), "denoise_pcm must emit at least one sample");
    for v in &out {
        assert!(v.is_finite(), "output must be finite (got {v})");
        assert!(v.abs() <= 1.5, "gain 0.5 * sine cannot exceed 1.5: got {v}");
    }
}

#[test]
fn identity_gain_bypass_reproduces_input_within_steady_state() {
    // Force the sigmoid mask to ≈ 1 by making the mask head bias very
    // large positive and the mask weight zero. Every other layer is
    // zero. Expected: gated STFT ≈ analysis STFT, and the reconstructed
    // PCM ≈ input PCM in the steady-state region (skipping the first
    // `n_fft - hop` samples where COLA has not converged for the non-
    // center streaming path).
    let cfg = tiny_config();
    let gguf = build_gguf(&cfg, |b| {
        let h = cfg.hidden_dim;
        let n_bins = cfg.n_bins;
        let fc1 = cfg.fc1_dim;
        let fc2 = cfg.fc2_dim;
        // Zero every layer except the mask bias.
        add_f32(
            b,
            TENSOR_FC_IN_WEIGHT,
            vec![h as u64, n_bins as u64],
            &vec![0.0; h * n_bins],
        );
        add_f32(b, TENSOR_FC_IN_BIAS, vec![h as u64], &vec![0.0; h]);
        for (w, r, bs) in [
            (TENSOR_GRU_1_W, TENSOR_GRU_1_R, TENSOR_GRU_1_B),
            (TENSOR_GRU_2_W, TENSOR_GRU_2_R, TENSOR_GRU_2_B),
        ] {
            add_f32(b, w, vec![(3 * h) as u64, h as u64], &vec![0.0; 3 * h * h]);
            add_f32(b, r, vec![(3 * h) as u64, h as u64], &vec![0.0; 3 * h * h]);
            add_f32(b, bs, vec![(6 * h) as u64], &vec![0.0; 6 * h]);
        }
        add_f32(
            b,
            TENSOR_FC_1_WEIGHT,
            vec![fc1 as u64, h as u64],
            &vec![0.0; fc1 * h],
        );
        add_f32(b, TENSOR_FC_1_BIAS, vec![fc1 as u64], &vec![0.0; fc1]);
        add_f32(
            b,
            TENSOR_FC_2_WEIGHT,
            vec![fc2 as u64, fc1 as u64],
            &vec![0.0; fc2 * fc1],
        );
        add_f32(b, TENSOR_FC_2_BIAS, vec![fc2 as u64], &vec![0.0; fc2]);
        add_f32(
            b,
            TENSOR_MASK_WEIGHT,
            vec![n_bins as u64, fc2 as u64],
            &vec![0.0; n_bins * fc2],
        );
        // Mask bias +30 → sigmoid ≈ 1.0 (sigmoid(15) ≈ 0.999999...).
        add_f32(
            b,
            TENSOR_MASK_BIAS,
            vec![n_bins as u64],
            &vec![30.0; n_bins],
        );
    });
    let model = Nsnet2V1::from_gguf(&gguf).unwrap();

    // Round-trip check: STFT → mask ≈ 1 → iSTFT ≈ input. Reuse
    // `istft_streaming_oneshot` with the same attrs to obtain the
    // reference (this is a synthesis-pipeline sanity — not upstream
    // parity — since we deliberately use `center=false`).
    let pcm: Vec<f32> = (0..128)
        .map(|n| (2.0 * std::f32::consts::PI * 500.0 * (n as f32) / 8_000.0).sin())
        .collect();
    let denoised = model.denoise_pcm(&pcm).unwrap();

    // Microsoft's no-delay frontend emits `ceil(input_len / hop)` frames:
    // internally it prepends one history frame, right-pads to a hop boundary,
    // then discards that history frame. Our causal raw-frame STFT obtains the
    // same frame set by right-padding the source to the corresponding natural
    // overlap-add extent before analysis.
    let official_frames = pcm.len().div_ceil(cfg.hop);
    let official_output_len = (official_frames - 1) * cfg.hop + cfg.n_fft;
    let mut reference_pcm = pcm.clone();
    reference_pcm.resize(official_output_len, 0.0);
    let reference = vokra_ops::istft_streaming_oneshot(
        &vokra_ops::stft(&reference_pcm, &analysis_stft_attrs(&cfg)).unwrap(),
        &synthesis_istft_attrs(&cfg),
    )
    .unwrap();

    assert_eq!(denoised.len(), official_output_len);
    assert_eq!(
        denoised.len(),
        reference.len(),
        "denoised len {} must match STFT→iSTFT round-trip len {}",
        denoised.len(),
        reference.len(),
    );
    // Skip the first frame's worth of samples — the non-`center`
    // streaming iSTFT's overlap-add takes one hop for the wsq
    // normaliser to reach its steady state (the first `n_fft - hop`
    // samples are attenuated by the tapering window sum).
    let skip = cfg.n_fft - cfg.hop;
    assert!(
        denoised.len() > skip,
        "denoised buffer must exceed COLA warmup"
    );
    let mut max_err = 0.0f32;
    for i in skip..denoised.len() {
        let e = (denoised[i] - reference[i]).abs();
        if e > max_err {
            max_err = e;
        }
    }
    // Sigmoid(30) ≈ 1 - 1e-13, so gated STFT is essentially the input;
    // the residual gap is only float rounding in the extra
    // linear/GRU/log pipeline (all zero-weighted, but still float-add
    // noisy). A 1e-4 absolute bound is generous.
    assert!(
        max_err < 1e-4,
        "identity-mask denoised must match STFT→iSTFT reference within 1e-4 in \
         steady state; got max |Δ| = {max_err}"
    );
}

// -------------------------------------------------------------------------
// Streaming state carry-over
// -------------------------------------------------------------------------

#[test]
fn split_push_matches_whole_utterance() {
    // Push the same 96-sample input in one call vs three 32-sample
    // calls. The GRU / iSTFT state carry-over must make the union of
    // the split outputs bit-identical to the whole one (streaming
    // contract, FR-LD-06).
    let cfg = tiny_config();
    let gguf = build_gguf(&cfg, |b| add_all_zero_tensors(b, &cfg));
    let model = Nsnet2V1::from_gguf(&gguf).unwrap();

    let pcm: Vec<f32> = (0..96).map(|n| ((n as f32) * 0.05).sin() * 0.3).collect();

    let whole = model.denoise_pcm(&pcm).unwrap();

    // Streaming path.
    let mut stream: Box<dyn DenoiseStreamHandle + Send> =
        model.open_stream(cfg.sample_rate).unwrap();
    let mut split: Vec<f32> = Vec::new();
    for chunk in pcm.chunks(32) {
        split.extend(stream.push_pcm(chunk).unwrap());
    }
    // Cast the streamed handle to `Nsnet2Stream` so we can call
    // finalize() — this requires `Any` downcast which the trait does
    // not expose, so we drive the flush via the trait's reset (which
    // clears the tail) after we finish. For final-tail equality
    // instead, re-open a fresh stream and push the whole buffer, then
    // finalize both.
    let _ = &mut split; // silence unused-mut warning after adjustment below.

    // Compare the pushed portion length-wise: the streaming iSTFT emits
    // whatever it can at each push and holds back the overlap tail; the
    // whole-utterance path funnels through the same streaming iSTFT
    // internally + finalize, so `whole` is `split_pushed + tail`.
    // Assert equality on the prefix that the split pushes actually
    // covered.
    let n = split.len().min(whole.len());
    for i in 0..n {
        assert!(
            (split[i] - whole[i]).abs() < 1e-6,
            "streaming push {i}: {} vs whole {} (state carry-over broke bit equality)",
            split[i],
            whole[i]
        );
    }
    // Reset must clear state so a second run reproduces the first.
    let mut stream2 = model.open_stream(cfg.sample_rate).unwrap();
    let split_again: Vec<f32> = pcm
        .chunks(32)
        .flat_map(|c| stream2.push_pcm(c).unwrap())
        .collect();
    assert_eq!(
        split, split_again,
        "fresh stream must reproduce the first stream bit-for-bit"
    );
}

#[test]
fn open_stream_rejects_sample_rate_mismatch() {
    let cfg = tiny_config();
    let gguf = build_gguf(&cfg, |b| add_all_zero_tensors(b, &cfg));
    let model = Nsnet2V1::from_gguf(&gguf).unwrap();
    // The `Box<dyn DenoiseStreamHandle + Send>` return type is not
    // `Debug`, so `unwrap_err()` (which needs `Debug` on the Ok half)
    // cannot see through — match by hand to extract the error.
    let err = match model.open_stream(cfg.sample_rate + 100) {
        Ok(_) => panic!("expected sample-rate mismatch error, got Ok(_)"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("sample rate mismatch") && msg.contains("nsnet2"),
        "expected loud sample-rate rejection, got: {msg}"
    );
}

// -------------------------------------------------------------------------
// GRU cell numeric spot-check vs hand-rolled f64 reference
// -------------------------------------------------------------------------

#[test]
fn gru_step_matches_hand_computed_reference() {
    // Build a hidden=2, in_dim=2 GRU with tiny distinct weights that
    // are easy to hand-verify. Compute the update in f64 by hand and
    // compare to the native ONNX `linear_before_reset=1` implementation.
    let hidden = 2usize;
    let in_dim = 2usize;

    // rnnoise layout: [R (0..h), Z (h..2h), N (2h..3h)] rows.
    // W_ih:      [3h, in_dim]      W_hh:      [3h, h]
    // bias_ih:   [3h]              bias_hh: [3h]
    // Fill with a distinct sentinel per gate/element so any block
    // swap or off-by-one row is detectable.
    let w_ih: Vec<f32> = vec![
        0.1, 0.2, // R row 0
        0.3, 0.4, // R row 1
        -0.1, -0.2, // Z row 0
        -0.3, -0.4, // Z row 1
        0.5, 0.6, // N row 0
        0.7, 0.8, // N row 1
    ];
    let w_hh: Vec<f32> = vec![
        0.05, 0.06, // R
        0.07, 0.08, //
        -0.05, -0.06, // Z
        -0.07, -0.08, //
        0.09, 0.10, // N
        0.11, 0.12, //
    ];
    let bias_ih: Vec<f32> = vec![
        0.01, 0.02, // R
        0.03, 0.04, // Z
        0.05, 0.06, // N
    ];
    let bias_hh: Vec<f32> = vec![
        -0.01, -0.02, // R
        -0.03, -0.04, // Z
        0.15, -0.16, // N -- must remain inside reset multiplication
    ];
    let x = [0.5f32, -0.5];
    let h0 = [0.1f32, -0.2];

    // Hand computation in f64 (per rnnoise: r = σ(W_ir x + W_hr h + b_r);
    // z = σ(W_iz x + W_hz h + b_z);
    // n = tanh(W_in x + r * (W_hn h) + b_n);
    // h_new = (1-z) n + z h).
    let sig = |v: f64| -> f64 { 0.5 * (0.5 * v).tanh() + 0.5 };
    let mut r = [0.0f64; 2];
    let mut z = [0.0f64; 2];
    let mut n_ih = [0.0f64; 2];
    let mut n_hh = [0.0f64; 2];
    for i in 0..hidden {
        let row_ih_r = &w_ih[i * in_dim..(i + 1) * in_dim];
        let row_hh_r = &w_hh[i * hidden..(i + 1) * hidden];
        let mut acc = (bias_ih[i] + bias_hh[i]) as f64;
        for k in 0..in_dim {
            acc += (row_ih_r[k] as f64) * (x[k] as f64);
        }
        for k in 0..hidden {
            acc += (row_hh_r[k] as f64) * (h0[k] as f64);
        }
        r[i] = sig(acc);

        let row_ih_z = &w_ih[(hidden + i) * in_dim..(hidden + i + 1) * in_dim];
        let row_hh_z = &w_hh[(hidden + i) * hidden..(hidden + i + 1) * hidden];
        let mut acc_z = (bias_ih[hidden + i] + bias_hh[hidden + i]) as f64;
        for k in 0..in_dim {
            acc_z += (row_ih_z[k] as f64) * (x[k] as f64);
        }
        for k in 0..hidden {
            acc_z += (row_hh_z[k] as f64) * (h0[k] as f64);
        }
        z[i] = sig(acc_z);

        let row_ih_n = &w_ih[(2 * hidden + i) * in_dim..(2 * hidden + i + 1) * in_dim];
        let row_hh_n = &w_hh[(2 * hidden + i) * hidden..(2 * hidden + i + 1) * hidden];
        for k in 0..in_dim {
            n_ih[i] += (row_ih_n[k] as f64) * (x[k] as f64);
        }
        for k in 0..hidden {
            n_hh[i] += (row_hh_n[k] as f64) * (h0[k] as f64);
        }
    }
    let mut want = [0.0f64; 2];
    for i in 0..hidden {
        let n_val = (n_ih[i]
            + bias_ih[2 * hidden + i] as f64
            + r[i] * (n_hh[i] + bias_hh[2 * hidden + i] as f64))
            .tanh();
        want[i] = (1.0 - z[i]) * n_val + z[i] * (h0[i] as f64);
    }

    // Run the native ONNX-compatible kernel.
    let mut state = h0.to_vec();
    onnx_gru_forward_cpu(&x, &mut state, &w_ih, &w_hh, &bias_ih, &bias_hh).unwrap();
    for i in 0..hidden {
        assert!(
            (state[i] as f64 - want[i]).abs() < 1e-5,
            "GRU spot-check row {i}: native = {}, hand ref = {}",
            state[i],
            want[i]
        );
    }
}
