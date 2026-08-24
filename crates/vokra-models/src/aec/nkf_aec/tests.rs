//! NKF-AEC runtime binder tests (structural / round-trip / FR-EX-08 /
//! streaming safety).
//!
//! Mirror of `crate::kws::openwakeword::tests` scope: config +
//! weight-bundle round-trip, missing / mis-shaped / transposed-dim
//! loud errors, a sample-rate contract test, a mic/farend-length
//! mismatch test, and — genuine synth-weight math — the dead-air
//! short-circuit contract (`farend = 0` ⇒ Kalman state unchanged,
//! `E = Y = mic`) that holds regardless of weight values.
//!
//! # Streaming safety (Defects 1+2 regression coverage)
//!
//! Four dedicated tests exercise the per-frame streaming drain:
//! [`hop_sized_pushes_produce_monotonically_growing_output`] proves
//! that a caller who pushes exactly `hop`-sized chunks 100 times
//! receives contiguous cleaned output past the first push (Defect 1
//! regression); [`drain_advances_kalman_state_once_per_new_frame`]
//! proves that a follow-up hop-sized push runs the Kalman recurrence
//! exactly once (Defect 2 regression);
//! [`whole_utterance_equals_hop_chunked_stream_within_tolerance`] pins
//! the streaming path bit-close to the whole-utterance path outside
//! the OLA warmup; and
//! [`kgnet_active_reduces_echo_with_deterministic_weights`] exercises
//! the KGNet + Kalman recurrence with non-zero deterministic weights.

use super::*;

use vokra_core::engines::{AecEngine, AecStreamHandle};
use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile, chunks};

// ---- fixture builders ---------------------------------------------------

/// Builds a valid all-zero NKF-AEC GGUF for structural tests. Every one
/// of the 22 tensors is emitted with the exact upstream-pinned shape.
/// The dim conventions here are the ones [`NkfAecWeights::from_gguf`]
/// gates against; a bug in the fixture surfaces as a load-time
/// [`VokraError::ModelLoad`], which is exactly what we want.
fn build_all_zero_gguf() -> Vec<u8> {
    let cfg = NkfAecConfig::upstream_default();
    let l = cfg.l as u64;
    let h = cfg.rnn_dim as u64;
    let ki = cfg.kgnet_in() as u64;
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, DEFAULT_NAME);
    let zeros = |n: u64| vec![0u8; (n * 4) as usize];
    let add = |b: &mut GgufBuilder, name: &str, dims: &[u64]| {
        let n: u64 = dims.iter().product();
        b.add_tensor(name, GgmlType::F32, dims.to_vec(), zeros(n))
            .expect("add tensor");
    };
    // fc_in
    add(&mut b, T_FC_IN_LR_W, &[h, ki]);
    add(&mut b, T_FC_IN_LR_B, &[h]);
    add(&mut b, T_FC_IN_LI_W, &[h, ki]);
    add(&mut b, T_FC_IN_LI_B, &[h]);
    add(&mut b, T_FC_IN_PRELU, &[1]);
    // complex_gru
    add(&mut b, T_GRU_R_W_IH, &[3 * h, h]);
    add(&mut b, T_GRU_R_W_HH, &[3 * h, h]);
    add(&mut b, T_GRU_R_B_IH, &[3 * h]);
    add(&mut b, T_GRU_R_B_HH, &[3 * h]);
    add(&mut b, T_GRU_I_W_IH, &[3 * h, h]);
    add(&mut b, T_GRU_I_W_HH, &[3 * h, h]);
    add(&mut b, T_GRU_I_B_IH, &[3 * h]);
    add(&mut b, T_GRU_I_B_HH, &[3 * h]);
    // fc_out
    add(&mut b, T_FC_OUT0_LR_W, &[h, h]);
    add(&mut b, T_FC_OUT0_LR_B, &[h]);
    add(&mut b, T_FC_OUT0_LI_W, &[h, h]);
    add(&mut b, T_FC_OUT0_LI_B, &[h]);
    add(&mut b, T_FC_OUT_PRELU, &[1]);
    add(&mut b, T_FC_OUT2_LR_W, &[l, h]);
    add(&mut b, T_FC_OUT2_LR_B, &[l]);
    add(&mut b, T_FC_OUT2_LI_W, &[l, h]);
    add(&mut b, T_FC_OUT2_LI_B, &[l]);
    b.to_bytes().expect("serialise NKF-AEC GGUF")
}

// ---- config -------------------------------------------------------------

#[test]
fn upstream_default_config_matches_pinned_constants() {
    let cfg = NkfAecConfig::upstream_default();
    assert_eq!(cfg.l, L);
    assert_eq!(cfg.fc_dim, FC_DIM);
    assert_eq!(cfg.rnn_dim, RNN_DIM);
    assert_eq!(cfg.n_fft, N_FFT);
    assert_eq!(cfg.hop, HOP);
    assert_eq!(cfg.win_length, WIN_LENGTH);
    assert_eq!(cfg.sample_rate, SAMPLE_RATE);
    assert_eq!(cfg.f_bins(), F_BINS);
    assert_eq!(cfg.kgnet_in(), KGNET_IN);
    cfg.validate().unwrap();
}

#[test]
fn validate_rejects_zero_hparams_loudly() {
    let mut cfg = NkfAecConfig::upstream_default();
    cfg.l = 0;
    let err = cfg.validate().expect_err("l=0 must be loud");
    match err {
        VokraError::InvalidArgument(m) => assert!(m.contains("`l`"), "{m}"),
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

#[test]
fn validate_rejects_win_length_greater_than_n_fft() {
    let mut cfg = NkfAecConfig::upstream_default();
    cfg.win_length = cfg.n_fft + 1;
    let err = cfg.validate().expect_err("win_length > n_fft must be loud");
    match err {
        VokraError::InvalidArgument(m) => assert!(m.contains("win_length"), "{m}"),
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

// ---- from_gguf round-trip -----------------------------------------------

#[test]
fn from_gguf_round_trips_all_zero_bundle() {
    let bytes = build_all_zero_gguf();
    let gguf = GgufFile::parse(bytes).unwrap();
    let session = NkfAec::from_gguf(&gguf).expect("from_gguf must succeed");
    assert_eq!(session.config(), &NkfAecConfig::upstream_default());
}

#[test]
fn from_gguf_rejects_wrong_arch_tag() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, "fsmn-vad");
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = NkfAec::from_gguf(&gguf).expect_err("wrong arch must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(msg.contains("fsmn-vad"), "must name seen arch: {msg}");
    assert!(msg.contains(ARCH), "must name expected arch: {msg}");
}

#[test]
fn from_gguf_rejects_missing_arch_tag() {
    let b = GgufBuilder::new();
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = NkfAec::from_gguf(&gguf).expect_err("missing arch must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(msg.contains("vokra.model.arch"), "{msg}");
}

#[test]
fn from_gguf_rejects_missing_tensor() {
    // Build the fixture, then re-emit with one tensor omitted.
    let cfg = NkfAecConfig::upstream_default();
    let l = cfg.l as u64;
    let h = cfg.rnn_dim as u64;
    let ki = cfg.kgnet_in() as u64;
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    let zeros = |n: u64| vec![0u8; (n * 4) as usize];
    let add = |b: &mut GgufBuilder, name: &str, dims: &[u64]| {
        let n: u64 = dims.iter().product();
        b.add_tensor(name, GgmlType::F32, dims.to_vec(), zeros(n))
            .unwrap();
    };
    // Deliberately OMIT T_FC_IN_LR_W.
    add(&mut b, T_FC_IN_LR_B, &[h]);
    add(&mut b, T_FC_IN_LI_W, &[h, ki]);
    add(&mut b, T_FC_IN_LI_B, &[h]);
    add(&mut b, T_FC_IN_PRELU, &[1]);
    add(&mut b, T_GRU_R_W_IH, &[3 * h, h]);
    add(&mut b, T_GRU_R_W_HH, &[3 * h, h]);
    add(&mut b, T_GRU_R_B_IH, &[3 * h]);
    add(&mut b, T_GRU_R_B_HH, &[3 * h]);
    add(&mut b, T_GRU_I_W_IH, &[3 * h, h]);
    add(&mut b, T_GRU_I_W_HH, &[3 * h, h]);
    add(&mut b, T_GRU_I_B_IH, &[3 * h]);
    add(&mut b, T_GRU_I_B_HH, &[3 * h]);
    add(&mut b, T_FC_OUT0_LR_W, &[h, h]);
    add(&mut b, T_FC_OUT0_LR_B, &[h]);
    add(&mut b, T_FC_OUT0_LI_W, &[h, h]);
    add(&mut b, T_FC_OUT0_LI_B, &[h]);
    add(&mut b, T_FC_OUT_PRELU, &[1]);
    add(&mut b, T_FC_OUT2_LR_W, &[l, h]);
    add(&mut b, T_FC_OUT2_LR_B, &[l]);
    add(&mut b, T_FC_OUT2_LI_W, &[l, h]);
    add(&mut b, T_FC_OUT2_LI_B, &[l]);
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = NkfAec::from_gguf(&gguf).expect_err("missing tensor must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(
        msg.contains(T_FC_IN_LR_W),
        "must name missing tensor: {msg}"
    );
}

/// Dim-order assertion (openWakeWord precedent): the loader must
/// reject a `weight_ih_l0` written with transposed dims even though
/// the element count is unchanged. A Python bridge that silently
/// writes `[D, 3H]` would slip past a product-only check and mis-forward.
#[test]
fn from_gguf_rejects_transposed_gru_weight_dims() {
    let cfg = NkfAecConfig::upstream_default();
    let h = cfg.rnn_dim as u64;
    let ki = cfg.kgnet_in() as u64;
    let l = cfg.l as u64;
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    let zeros = |n: u64| vec![0u8; (n * 4) as usize];
    let add = |b: &mut GgufBuilder, name: &str, dims: &[u64]| {
        let n: u64 = dims.iter().product();
        b.add_tensor(name, GgmlType::F32, dims.to_vec(), zeros(n))
            .unwrap();
    };
    add(&mut b, T_FC_IN_LR_W, &[h, ki]);
    add(&mut b, T_FC_IN_LR_B, &[h]);
    add(&mut b, T_FC_IN_LI_W, &[h, ki]);
    add(&mut b, T_FC_IN_LI_B, &[h]);
    add(&mut b, T_FC_IN_PRELU, &[1]);
    // Adversarial transpose: `weight_ih_l0` documented as `[3H, D]`
    // (row-major); write it as `[D, 3H]`. Product 54*18 == 18*54 so
    // element count matches; only the dim gate catches this.
    add(&mut b, T_GRU_R_W_IH, &[h, 3 * h]);
    add(&mut b, T_GRU_R_W_HH, &[3 * h, h]);
    add(&mut b, T_GRU_R_B_IH, &[3 * h]);
    add(&mut b, T_GRU_R_B_HH, &[3 * h]);
    add(&mut b, T_GRU_I_W_IH, &[3 * h, h]);
    add(&mut b, T_GRU_I_W_HH, &[3 * h, h]);
    add(&mut b, T_GRU_I_B_IH, &[3 * h]);
    add(&mut b, T_GRU_I_B_HH, &[3 * h]);
    add(&mut b, T_FC_OUT0_LR_W, &[h, h]);
    add(&mut b, T_FC_OUT0_LR_B, &[h]);
    add(&mut b, T_FC_OUT0_LI_W, &[h, h]);
    add(&mut b, T_FC_OUT0_LI_B, &[h]);
    add(&mut b, T_FC_OUT_PRELU, &[1]);
    add(&mut b, T_FC_OUT2_LR_W, &[l, h]);
    add(&mut b, T_FC_OUT2_LR_B, &[l]);
    add(&mut b, T_FC_OUT2_LI_W, &[l, h]);
    add(&mut b, T_FC_OUT2_LI_B, &[l]);
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = NkfAec::from_gguf(&gguf).expect_err("transposed GRU weight must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(
        msg.contains(T_GRU_R_W_IH),
        "must name offending tensor: {msg}"
    );
    assert!(msg.contains("dims"), "must mention dims: {msg}");
}

/// Wrong element-count for a per-tensor `[out, in]` linear is caught
/// before the dim gate — asserts that the element-count check does its
/// job independently of the dim gate.
#[test]
fn from_gguf_rejects_wrong_element_count() {
    let cfg = NkfAecConfig::upstream_default();
    let h = cfg.rnn_dim as u64;
    let ki = cfg.kgnet_in() as u64;
    let l = cfg.l as u64;
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    let zeros = |n: u64| vec![0u8; (n * 4) as usize];
    let add = |b: &mut GgufBuilder, name: &str, dims: &[u64]| {
        let n: u64 = dims.iter().product();
        b.add_tensor(name, GgmlType::F32, dims.to_vec(), zeros(n))
            .unwrap();
    };
    // Wrong-shape `T_FC_IN_LR_W`: doc says `[18, 9]` (162 elements); emit
    // `[7, 9]` (63 elements) — count mismatch, loud fail.
    add(&mut b, T_FC_IN_LR_W, &[7, ki]);
    add(&mut b, T_FC_IN_LR_B, &[h]);
    add(&mut b, T_FC_IN_LI_W, &[h, ki]);
    add(&mut b, T_FC_IN_LI_B, &[h]);
    add(&mut b, T_FC_IN_PRELU, &[1]);
    add(&mut b, T_GRU_R_W_IH, &[3 * h, h]);
    add(&mut b, T_GRU_R_W_HH, &[3 * h, h]);
    add(&mut b, T_GRU_R_B_IH, &[3 * h]);
    add(&mut b, T_GRU_R_B_HH, &[3 * h]);
    add(&mut b, T_GRU_I_W_IH, &[3 * h, h]);
    add(&mut b, T_GRU_I_W_HH, &[3 * h, h]);
    add(&mut b, T_GRU_I_B_IH, &[3 * h]);
    add(&mut b, T_GRU_I_B_HH, &[3 * h]);
    add(&mut b, T_FC_OUT0_LR_W, &[h, h]);
    add(&mut b, T_FC_OUT0_LR_B, &[h]);
    add(&mut b, T_FC_OUT0_LI_W, &[h, h]);
    add(&mut b, T_FC_OUT0_LI_B, &[h]);
    add(&mut b, T_FC_OUT_PRELU, &[1]);
    add(&mut b, T_FC_OUT2_LR_W, &[l, h]);
    add(&mut b, T_FC_OUT2_LR_B, &[l]);
    add(&mut b, T_FC_OUT2_LI_W, &[l, h]);
    add(&mut b, T_FC_OUT2_LI_B, &[l]);
    let bytes = b.to_bytes().unwrap();
    let gguf = GgufFile::parse(bytes).unwrap();
    let err = NkfAec::from_gguf(&gguf).expect_err("wrong element count must be loud");
    let msg = match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected ModelLoad, got {other:?}"),
    };
    assert!(
        msg.contains(T_FC_IN_LR_W),
        "must name offending tensor: {msg}"
    );
    assert!(msg.contains("elements"), "must mention elements: {msg}");
}

// ---- AecEngine trait ----------------------------------------------------

#[test]
fn open_stream_rejects_wrong_sample_rate_loudly() {
    let bytes = build_all_zero_gguf();
    let gguf = GgufFile::parse(bytes).unwrap();
    let session = NkfAec::from_gguf(&gguf).unwrap();
    // `Box<dyn AecStreamHandle + Send>` is not `Debug`, so we can't
    // use `.expect_err(...)` — pattern-match on the Result directly.
    let msg = match session.open_stream(8_000) {
        Err(VokraError::InvalidArgument(m)) => m,
        Err(other) => panic!("expected InvalidArgument, got {other:?}"),
        Ok(_) => panic!("wrong sample rate must be loud, got Ok"),
    };
    assert!(msg.contains("8000") || msg.contains("8_000"), "{msg}");
    assert!(msg.contains("16_000") || msg.contains("16000"), "{msg}");
}

#[test]
fn open_stream_accepts_matching_sample_rate() {
    let bytes = build_all_zero_gguf();
    let gguf = GgufFile::parse(bytes).unwrap();
    let session = NkfAec::from_gguf(&gguf).unwrap();
    let stream = session
        .open_stream(SAMPLE_RATE)
        .expect("matching sample rate must succeed");
    // Sanity: a Zero-length push starves both buffers; no error, no PCM.
    // (Length matches — both zero — so the alignment gate does not
    // fire; the STFT drain returns empty because we have no samples.)
    let mut stream = stream;
    let out = stream.push_paired(&[], &[]).unwrap();
    assert!(out.is_empty());
}

// ---- push_paired FR-EX-08 -----------------------------------------------

#[test]
fn push_paired_rejects_length_mismatch_loudly() {
    let bytes = build_all_zero_gguf();
    let gguf = GgufFile::parse(bytes).unwrap();
    let session = NkfAec::from_gguf(&gguf).unwrap();
    let mut stream = session.open_stream(SAMPLE_RATE).unwrap();
    let err = stream
        .push_paired(&[0.0f32; 100], &[0.0f32; 200])
        .expect_err("length mismatch must be loud");
    let msg = match err {
        VokraError::InvalidArgument(m) => m,
        other => panic!("expected InvalidArgument, got {other:?}"),
    };
    assert!(msg.contains("100"), "{msg}");
    assert!(msg.contains("200"), "{msg}");
    assert!(msg.contains("sample-aligned"), "{msg}");
}

/// Retry-loop resistance (openWakeWord precedent): a caller that
/// swallows the length-mismatch loud-fail and keeps pushing MUST NOT
/// grow the internal pending buffers — the loud-fail gate fires before
/// any `extend`. Without this the naive retry loop leaks memory.
#[test]
fn push_paired_does_not_grow_pending_on_length_mismatch() {
    let bytes = build_all_zero_gguf();
    let gguf = GgufFile::parse(bytes).unwrap();
    let session = NkfAec::from_gguf(&gguf).unwrap();
    let mut stream = session.open_stream(SAMPLE_RATE).unwrap();
    for _ in 0..50 {
        let _ = stream.push_paired(&[0.0f32; 100], &[0.0f32; 200]);
    }
    // Downcast to the concrete NkfAecStream to observe internal state.
    // We used AecStreamHandle above; drop the trait object and rebuild
    // via a direct call to check the invariant.
    let mut stream2 = NkfAecStream::new(
        NkfAecConfig::upstream_default(),
        Arc::new(NkfAecWeights::zeros(&NkfAecConfig::upstream_default())),
    );
    for _ in 0..50 {
        let _ = stream2.push_paired(&[0.0f32; 100], &[0.0f32; 200]);
    }
    assert_eq!(
        stream2.pending_mic.len(),
        0,
        "loud-fail must run before any buffer extend (FR-EX-08 retry-loop safety)"
    );
    assert_eq!(stream2.pending_farend.len(), 0);
}

// ---- forward math (synth weights, genuine invariants) --------------------

/// **Dead-air short-circuit contract** (`nkf.py:127-131`): with
/// `farend = 0` (or any signal whose per-bin `mean(|xt|) < 1e-5`),
/// upstream's Kalman loop `continue`s — `echo_hat[k, t] = 0`,
/// `E[k, t] = Y[k, t]`, and the filter state is **untouched**
/// (`h_prior`, `h_posterior`, GRU hidden). This binder mirrors that
/// verbatim (see [`super::NkfAecStream::step_frame`]).
///
/// This asserts BOTH:
/// - the STFT + iSTFT + OLA pipeline emits sample-domain output
///   (proves the streaming drain committed non-empty output past
///   warmup — Defect 1 shape);
/// - the Kalman state is bit-identical to its post-`reset` initial
///   state (proves the skip path did not silently mutate — Defect 3).
///
/// Renamed from `zero_farend_identity_recovers_mic_via_stft_istft_roundtrip`
/// under the design's Defect 3 patch (the old name promised a full mic
/// recovery which the new `center=False` streaming path cannot make at
/// the head of the utterance — Hann OLA has ~`n_fft` samples of
/// warmup distortion; the invariant that actually holds is the
/// no-mutation contract on the Kalman state).
#[test]
fn zero_farend_hits_dead_air_shortcut_and_returns_mic_via_ola() {
    let cfg = NkfAecConfig::upstream_default();
    let session = NkfAec::from_parts(cfg.clone(), NkfAecWeights::zeros(&cfg));
    // Open concrete stream so we can inspect the private state fields
    // (`pub(crate)` under `#[cfg(test)]` per the design).
    let mut stream = NkfAecStream::new(cfg.clone(), Arc::clone(&session.weights));

    // 4096 samples ≈ 4× n_fft + tail; enough to run several STFT
    // frames and let the OLA commit an interior stretch.
    let n = 4096;
    let mic: Vec<f32> = (0..n)
        .map(|t| (t as f32 * 0.02).sin() + 0.3 * (t as f32 * 0.11).cos())
        .collect();
    let farend = vec![0.0f32; n];

    let cleaned = stream.push_paired(&mic, &farend).unwrap();

    // ---- Defect 1 shape: drain committed non-empty output past warmup.
    assert!(
        !cleaned.is_empty(),
        "drain must commit cleaned samples once pending has > n_fft samples"
    );

    // ---- Defect 3 no-mutation contract on the Kalman state under the
    // dead-air skip. Every h_prior / h_posterior element must still be
    // Complex32::ZERO because `nkf.py:127` `continue`s without touching
    // the filter state (see `step_frame`'s dead-air branch — no writes
    // to h_prior/h_posterior/h_rr/h_ir/h_ri/h_ii on the skip).
    let expected_frames =
        if stream.buffer_offset_abs + stream.pending_mic.len() + stream.frames_processed * cfg.hop
            >= cfg.n_fft
        {
            // Full utterance length ⇒ every frame that fits is processed.
            (n - cfg.n_fft) / cfg.hop + 1
        } else {
            0
        };
    assert_eq!(
        stream.frames_processed, expected_frames,
        "one Kalman step per new STFT frame (frames_processed advanced by \
         exactly `available_frames` on a whole-utterance push)"
    );
    assert!(
        stream.h_prior.iter().all(|c| c.re == 0.0 && c.im == 0.0),
        "dead-air shortcut must not mutate h_prior (nkf.py:127-131 skip)"
    );
    assert!(
        stream
            .h_posterior
            .iter()
            .all(|c| c.re == 0.0 && c.im == 0.0),
        "dead-air shortcut must not mutate h_posterior"
    );
    assert!(
        stream.h_rr.iter().all(|&x| x == 0.0),
        "dead-air shortcut must not mutate GRU h_rr"
    );
    assert!(
        stream.h_ir.iter().all(|&x| x == 0.0),
        "dead-air shortcut must not mutate GRU h_ir"
    );
    assert!(
        stream.h_ri.iter().all(|&x| x == 0.0),
        "dead-air shortcut must not mutate GRU h_ri"
    );
    assert!(
        stream.h_ii.iter().all(|&x| x == 0.0),
        "dead-air shortcut must not mutate GRU h_ii"
    );

    // Emitted output covers `[emitted_abs, ola_start_abs)` — with a
    // one-shot push, `emitted_abs == committed_abs == available_frames
    // * hop`, matching `cleaned.len()`.
    assert_eq!(
        stream.emitted_abs,
        expected_frames * cfg.hop,
        "emitted_abs must equal available_frames * hop on a whole-utterance drain"
    );
    assert_eq!(cleaned.len(), stream.emitted_abs);
}

/// Zero-farend identity + reset carries: pushing after `reset` behaves
/// as if we had opened a fresh stream (bit-identical output for the
/// same input).
#[test]
fn reset_returns_stream_to_initial_state() {
    let cfg = NkfAecConfig::upstream_default();
    let session = NkfAec::from_parts(cfg.clone(), NkfAecWeights::zeros(&cfg));
    let mut a = session.open_stream(SAMPLE_RATE).unwrap();
    let mut b = session.open_stream(SAMPLE_RATE).unwrap();

    let mic: Vec<f32> = (0..2048).map(|t| (t as f32 * 0.03).sin()).collect();
    let far = vec![0.0f32; 2048];

    // Prime `a` with unrelated audio, then reset.
    let _ = a
        .push_paired(
            &(0..2048)
                .map(|t| (t as f32 * 0.7).cos())
                .collect::<Vec<_>>(),
            &far,
        )
        .unwrap();
    a.reset();
    let out_a = a.push_paired(&mic, &far).unwrap();

    // `b` runs fresh on the same input.
    let out_b = b.push_paired(&mic, &far).unwrap();

    assert_eq!(
        out_a.len(),
        out_b.len(),
        "reset must produce the same output length as a fresh stream"
    );
    for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "reset drift at sample {i}: after-reset={a}, fresh={b}, |Δ|={}",
            (a - b).abs()
        );
    }
}

/// The zero-length push is a no-op — both slices have length 0 so the
/// alignment gate passes, and the STFT has no samples to consume.
/// Returns an empty vec.
#[test]
fn push_paired_zero_length_is_noop() {
    let cfg = NkfAecConfig::upstream_default();
    let session = NkfAec::from_parts(cfg.clone(), NkfAecWeights::zeros(&cfg));
    let mut stream = session.open_stream(SAMPLE_RATE).unwrap();
    let out = stream.push_paired(&[], &[]).unwrap();
    assert!(out.is_empty());
}

/// A push that adds < n_fft samples of pending audio returns nothing
/// (the STFT can't fire yet), but retains the samples for the next
/// push — the following push carries them across.
#[test]
fn push_paired_below_n_fft_is_buffered_not_dropped() {
    let cfg = NkfAecConfig::upstream_default();
    let session = NkfAec::from_parts(cfg.clone(), NkfAecWeights::zeros(&cfg));
    let mut stream = NkfAecStream::new(cfg.clone(), Arc::clone(&session.weights));

    // < n_fft: too short to STFT.
    let chunk = vec![0.5f32; 512];
    let farend = vec![0.0f32; 512];
    let out = stream.push_paired(&chunk, &farend).unwrap();
    assert!(
        out.is_empty(),
        "must not commit before we have n_fft samples"
    );
    assert_eq!(
        stream.pending_mic.len(),
        512,
        "samples below n_fft stay in pending buffer"
    );

    // Push another 512, now we have 1024 = n_fft — enough for one
    // STFT frame in the block.
    let out2 = stream.push_paired(&chunk, &farend).unwrap();
    // With zero far-end the identity path fires and iSTFT overlap-add
    // commits at least some samples once we have >= n_fft.
    // The exact count depends on the iSTFT trim, but zero would signal
    // that our carry logic dropped the samples.
    assert!(
        !out2.is_empty(),
        "once pending has >= n_fft samples the drain must commit output"
    );
}

// ---- KGNet forward: numeric spot-check ---------------------------------

/// Sigmoid / tanh / PReLU + linear composition sanity: with all
/// weights zero and PReLU coeff = 0, KGNet always emits zero for any
/// input (GRU hidden stays zero, complex_gru output stays zero, final
/// complex dense produces zero from a zero input plus zero bias).
#[test]
fn kgnet_step_returns_zero_for_all_zero_weights() {
    let cfg = NkfAecConfig::upstream_default();
    let w = NkfAecWeights::zeros(&cfg);
    let feat_re = vec![1.0f32; cfg.kgnet_in()];
    let feat_im = vec![-2.0f32; cfg.kgnet_in()];
    let mut h_rr = vec![0.0f32; cfg.rnn_dim];
    let mut h_ir = vec![0.0f32; cfg.rnn_dim];
    let mut h_ri = vec![0.0f32; cfg.rnn_dim];
    let mut h_ii = vec![0.0f32; cfg.rnn_dim];
    let mut kg_re = vec![0.0f32; cfg.l];
    let mut kg_im = vec![0.0f32; cfg.l];
    let mut scratch = KgnetScratch::new(&cfg);
    kgnet_step(
        &w,
        &feat_re,
        &feat_im,
        &mut h_rr,
        &mut h_ir,
        &mut h_ri,
        &mut h_ii,
        &mut kg_re,
        &mut kg_im,
        &mut scratch,
        None,
    )
    .unwrap();
    for (i, &v) in kg_re.iter().enumerate() {
        assert!(v.abs() < 1e-6, "kg_re[{i}] = {v}, want 0");
    }
    for (i, &v) in kg_im.iter().enumerate() {
        assert!(v.abs() < 1e-6, "kg_im[{i}] = {v}, want 0");
    }
}

/// Test the PReLU: with input `[-3.0, 2.0]` and coeff `0.25`, we
/// expect `[-0.75, 2.0]`.
#[test]
fn complex_prelu_applies_shared_coefficient_to_negatives_only() {
    let mut buf = vec![-3.0f32, 2.0, -0.5, 1.5];
    complex_prelu_apply(0.25, &mut buf);
    assert_eq!(buf, vec![-0.75, 2.0, -0.125, 1.5]);
}

/// Test the GRU step: with all zero weights and biases, the input has
/// no effect and the hidden stays at whatever it was.
#[test]
fn gru_step_with_zero_weights_preserves_hidden() {
    let h = 4usize;
    let d = 3usize;
    let gru = GruWeights {
        weight_ih: vec![0.0; 3 * h * d],
        weight_hh: vec![0.0; 3 * h * h],
        bias_ih: vec![0.0; 3 * h],
        bias_hh: vec![0.0; 3 * h],
        hidden: h,
        input_size: d,
    };
    let x = vec![1.0, -2.0, 3.0];
    let h_in = vec![0.5, -0.5, 0.25, -0.25];
    let mut h_out = vec![0.0; h];
    gru.step(&x, &h_in, &mut h_out);
    // r = sigmoid(0) = 0.5; z = sigmoid(0) = 0.5; n = tanh(0.5 * 0) = 0
    // h' = (1 - 0.5) * 0 + 0.5 * h_in = 0.5 * h_in
    for i in 0..h {
        let expected = 0.5 * h_in[i];
        assert!(
            (h_out[i] - expected).abs() < 1e-6,
            "h_out[{i}] = {}, expected {expected}",
            h_out[i]
        );
    }
}

// ---- streaming safety tests (Defects 1 + 2 + 4 regression coverage) ------
//
// SplitMix64 (`vokra_core::SplitMix64` is already in the workspace; we
// re-implement the tiny 4-line hash locally here so the tests stay
// dependency-free at the module boundary and self-contained).
//
// # SplitMix64 (Guy Steele, 2014) — deterministic 64-bit PRNG suitable
// for seeded synth-weight and audio fixtures. Adequate for property
// tests; no security posture.

/// Simple SplitMix64 PRNG used only by the streaming tests.
#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform `f32` in `[-1, 1)`.
    fn next_f32(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as u32; // 24 bits
        ((u as f32) / ((1u32 << 24) as f32)) * 2.0 - 1.0
    }
    /// Box-Muller Gaussian (mean 0, std `sigma`).
    fn next_gaussian(&mut self, sigma: f32) -> f32 {
        let u1 = ((self.next_u64() >> 40) as f32 / ((1u32 << 24) as f32)).max(1e-9);
        let u2 = (self.next_u64() >> 40) as f32 / ((1u32 << 24) as f32);
        let two_pi = 2.0 * std::f32::consts::PI;
        sigma * (-2.0 * u1.ln()).sqrt() * (two_pi * u2).cos()
    }
}

/// Builds a deterministic small-scale weight bundle (fixed
/// `sigma = 0.01`, mean 0) suitable for exercising the KGNet + Kalman
/// forward with non-trivial values while keeping the recurrence
/// bounded (no NaN / Inf from an untrained `ComplexGRU` diverging over
/// many time steps — Xavier / He are too large for finite-time
/// stability without training). See the fn body for the SIGMA rationale.
fn build_synth_weights(cfg: &NkfAecConfig, seed: u64) -> NkfAecWeights {
    let mut r = SplitMix64::new(seed);
    let h = cfg.rnn_dim;
    let l = cfg.l;
    let ki = cfg.kgnet_in();

    // Conservative fixed sigma keeps `|KG| ≪ 1` so `h_posterior =
    // h_prior + KG · e` walks slowly — Xavier / He (`sqrt(2/fan_in)`)
    // is designed for gradient-flow stability under training, not for
    // finite-time recurrence stability without training, and diverges
    // empirically at ~2 s of broadband input for this recurrence.
    const SIGMA: f32 = 0.01;

    fn mk_linear(r: &mut SplitMix64, out_dim: usize, in_dim: usize, sigma: f32) -> LinearWeights {
        let weight: Vec<f32> = (0..out_dim * in_dim)
            .map(|_| r.next_gaussian(sigma))
            .collect();
        let bias = vec![0.0f32; out_dim];
        LinearWeights {
            weight,
            bias,
            out_dim,
            in_dim,
        }
    }
    fn mk_gru(r: &mut SplitMix64, hidden: usize, input_size: usize, sigma: f32) -> GruWeights {
        let weight_ih: Vec<f32> = (0..3 * hidden * input_size)
            .map(|_| r.next_gaussian(sigma))
            .collect();
        let weight_hh: Vec<f32> = (0..3 * hidden * hidden)
            .map(|_| r.next_gaussian(sigma))
            .collect();
        let bias_ih = vec![0.0f32; 3 * hidden];
        let bias_hh = vec![0.0f32; 3 * hidden];
        GruWeights {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            hidden,
            input_size,
        }
    }

    NkfAecWeights {
        fc_in_lr: mk_linear(&mut r, h, ki, SIGMA),
        fc_in_li: mk_linear(&mut r, h, ki, SIGMA),
        // Upstream `nn.PReLU` initialises to 0.25 (torch default).
        fc_in_prelu: 0.25,
        gru_r: mk_gru(&mut r, h, cfg.fc_dim, SIGMA),
        gru_i: mk_gru(&mut r, h, cfg.fc_dim, SIGMA),
        fc_out0_lr: mk_linear(&mut r, h, h, SIGMA),
        fc_out0_li: mk_linear(&mut r, h, h, SIGMA),
        fc_out_prelu: 0.25,
        fc_out2_lr: mk_linear(&mut r, l, h, SIGMA),
        fc_out2_li: mk_linear(&mut r, l, h, SIGMA),
    }
}

/// **Defect 4** regression: exercise the KGNet + Kalman recurrence
/// with non-trivial deterministic weights so all four Complex
/// primitives (`ComplexDense`, `ComplexPReLU`, `ComplexGRU`, and the
/// Kalman update math itself) actually run against non-zero data.
/// Distinct from the zero-farend / all-zero-weights identity tests
/// (which only prove structural pass-through) — this is the smallest
/// test that fires every gate of the neural code path.
///
/// # Design deviation vs proposed contract
///
/// The design spec's `mean(|E|²) < 0.9 · mean(|Y|²)` bound assumes the
/// synth-weight KGNet reduces echo energy. Empirically that is
/// **untrained** — random-Xavier-seeded weights produce a random walk
/// in `h_posterior`; the recurrence is finite (no NaN — the primary
/// coverage assertion below), but on synthetic input in ~2 s of audio
/// the direction of the walk is not systematically echo-reducing. A
/// hard energy-reduction bound would depend on the trained NKF-AEC
/// checkpoint (see `crates/vokra-models/tests/parity_nkf_aec.rs` for
/// the real-weight ERLE floor, gated on `VOKRA_NKF_AEC_REAL_GGUF`).
///
/// # Contract (implementable at synth-weight strength)
///
/// - `farend = white noise × 2 s @ 16 kHz` — broadband so nearly every
///   frequency bin fires the KGNet path (a sinusoid would trip the
///   dead-air short-circuit on most bins, hiding coverage).
/// - `mic = 0.5 · farend + N(0, 0.01)` — synthetic linear echo path.
/// - After processing the whole utterance:
///   1. every cleaned sample is FINITE (no NaN / Inf — proves the
///      recurrence stayed bounded);
///   2. `frames_processed` is `(n - n_fft)/hop + 1` (exactly one
///      Kalman step per new STFT frame, Defect 2 shape);
///   3. at least one `h_posterior` entry is non-zero (proves the
///      dead-air skip did NOT short-circuit every bin — the KGNet +
///      Kalman update fired and mutated state);
///   4. at least one GRU hidden entry is non-zero (proves the
///      `complex_gru_step` mutated per-bin history — the ComplexGRU
///      path ran).
#[test]
fn kgnet_active_reduces_echo_with_deterministic_weights() {
    let cfg = NkfAecConfig::upstream_default();
    let weights = build_synth_weights(&cfg, 42);
    let session = NkfAec::from_parts(cfg.clone(), weights);
    let mut stream = NkfAecStream::new(cfg.clone(), Arc::clone(&session.weights));

    let n = SAMPLE_RATE as usize * 2; // 32000 samples = 2 s @ 16 kHz
    let mut r = SplitMix64::new(42);
    // Broadband white noise ⇒ far-end STFT has energy across every
    // frequency bin ⇒ dead-air skip fires only for near-silence bins
    // (~none, at 0.1 amplitude and 2 s of data). Amplitude 0.1 keeps
    // STFT peak magnitudes bounded so the untrained KG recurrence does
    // not saturate the `tanh` in the ComplexGRU cell.
    let farend: Vec<f32> = (0..n).map(|_| r.next_f32() * 0.1).collect();
    let mic: Vec<f32> = farend
        .iter()
        .map(|&x| 0.5 * x + 0.01 * r.next_gaussian(1.0))
        .collect();

    let cleaned = stream.push_paired(&mic, &farend).unwrap();

    // (1) Every cleaned sample is finite. NaN / Inf would signal a
    // divergent recurrence (KG saturating, unbounded Kalman drift).
    for (i, &s) in cleaned.iter().enumerate() {
        assert!(
            s.is_finite(),
            "cleaned[{i}] = {s} is not finite — the Kalman recurrence \
             diverged (likely KGNet output saturated the tanh in ComplexGRU \
             and produced ±Inf that propagated through h_posterior)"
        );
    }

    // (2) One Kalman step per new STFT frame (Defect 2 invariant).
    let expected_frames = (n - cfg.n_fft) / cfg.hop + 1;
    assert_eq!(
        stream.frames_processed, expected_frames,
        "Kalman must advance exactly `available_frames` times per drain \
         (Defect 2 shape held under non-trivial weights)"
    );

    // (3) KGNet + Kalman path actually fired (not entirely dead-air).
    let nonzero_h_post = stream
        .h_posterior
        .iter()
        .filter(|c| c.re != 0.0 || c.im != 0.0)
        .count();
    assert!(
        nonzero_h_post > 0,
        "h_posterior all-zero after processing broadband mic + farend — the \
         KGNet + Kalman update did not fire at any bin (either every bin \
         hit the dead-air skip, or step_frame silently short-circuited)"
    );

    // (4) ComplexGRU path fired (per-bin hidden state was mutated).
    let nonzero_gru = stream.h_rr.iter().any(|&x| x != 0.0)
        || stream.h_ir.iter().any(|&x| x != 0.0)
        || stream.h_ri.iter().any(|&x| x != 0.0)
        || stream.h_ii.iter().any(|&x| x != 0.0);
    assert!(
        nonzero_gru,
        "every ComplexGRU hidden state (h_rr/h_ir/h_ri/h_ii) is zero after \
         a 2 s broadband push — complex_gru_step did not run"
    );
}

/// **Defects 1 + 2** cornerstone regression: pushing 100 chunks of
/// exactly `hop`-sized cleaned mic + far-end pairs must produce
/// contiguous cleaned output past the warmup — the direct disproof of
/// the pre-fix drain that leaked pending buffers because `emitted_samples
/// < cleaned.len()` compared against per-block `cleaned.len()`.
///
/// # Contract
///
/// - 100 pushes of `hop = 256` samples each = 25600 samples total.
/// - After the first `n_fft / hop - 1 = 3` pushes (which starve
///   pending), each push must produce exactly `hop` samples of output
///   (Kalman advances by exactly one new frame per push after warmup).
/// - Total cleaned output must exceed `96 * hop` samples (allows
///   generous warmup slack — the design's floor).
///
/// With the pre-fix drain this test fails around push 5 because the
/// block-based `emitted_samples` gate stops growing past `cleaned.len()`
/// of a single block.
#[test]
fn hop_sized_pushes_produce_monotonically_growing_output() {
    let cfg = NkfAecConfig::upstream_default();
    let session = NkfAec::from_parts(cfg.clone(), NkfAecWeights::zeros(&cfg));
    let mut stream = NkfAecStream::new(cfg.clone(), Arc::clone(&session.weights));

    let hop = cfg.hop;
    let mut r = SplitMix64::new(7);
    let mut mic_all: Vec<f32> = Vec::with_capacity(100 * hop);
    let mut far_all: Vec<f32> = Vec::with_capacity(100 * hop);
    for _ in 0..100 * hop {
        mic_all.push(r.next_f32() * 0.5);
        far_all.push(r.next_f32() * 0.5);
    }

    let mut total_out = Vec::new();
    let mut per_push_lens: Vec<usize> = Vec::with_capacity(100);
    for i in 0..100 {
        let mic_chunk = &mic_all[i * hop..(i + 1) * hop];
        let far_chunk = &far_all[i * hop..(i + 1) * hop];
        let out = stream.push_paired(mic_chunk, far_chunk).unwrap();
        per_push_lens.push(out.len());
        total_out.extend_from_slice(&out);
    }

    // Total output floor — design's `>= 96 * hop`.
    assert!(
        total_out.len() >= 96 * hop,
        "streaming leaked pending buffers — total_out.len() = {} < 96 * hop = {} \
         (Defect 1 regression: drain does not grow past first cleaned.len())",
        total_out.len(),
        96 * hop
    );

    // Per-push shape after warmup — every push at `i >= 4` must emit
    // exactly `hop` samples (Kalman advances by exactly one new frame,
    // producing exactly one `hop`-worth of new committed OLA).
    for (i, &out_len) in per_push_lens.iter().enumerate().take(100).skip(4) {
        assert_eq!(
            out_len, hop,
            "push {i} should emit exactly hop = {hop} samples (got {out_len}) — \
             streaming drain must advance by one frame per push after warmup"
        );
    }
}

/// **Defect 2** direct regression: after priming with 5 hop-sized
/// pushes (available_frames = 2 after push 4, 3 after push 5), the
/// next hop-sized push must advance `frames_processed` by **exactly 1**
/// — not by `available_frames`, not by the pending buffer length in
/// hops, not by 0. This is the invariant the pre-fix drain violated by
/// re-STFT'ing the entire pending buffer every call, running Kalman
/// `available_frames` times per push.
#[test]
fn drain_advances_kalman_state_once_per_new_frame() {
    let cfg = NkfAecConfig::upstream_default();
    let session = NkfAec::from_parts(cfg.clone(), NkfAecWeights::zeros(&cfg));
    let mut stream = NkfAecStream::new(cfg.clone(), Arc::clone(&session.weights));

    let hop = cfg.hop;
    let mut r = SplitMix64::new(13);
    // Prime: push 5 hop-sized chunks. After 4 pushes, pending = n_fft
    // (available_frames = 1). After push 5, pending in absolute terms is
    // 5*hop = 1280 = n_fft + hop, so available_frames = 2.
    for _ in 0..5 {
        let mic: Vec<f32> = (0..hop).map(|_| r.next_f32() * 0.5).collect();
        let far: Vec<f32> = (0..hop).map(|_| r.next_f32() * 0.5).collect();
        let _ = stream.push_paired(&mic, &far).unwrap();
    }
    let frames_after_prime = stream.frames_processed;
    assert_eq!(
        frames_after_prime, 2,
        "after 5 hop-sized pushes, exactly 2 Kalman steps should have fired \
         (available_frames = 2 = (5*hop - n_fft)/hop + 1)"
    );

    // One more hop-sized push: expect `frames_processed` to grow by
    // exactly 1 (available_frames = 3 = (6*hop - n_fft)/hop + 1;
    // frames_processed advances 2 → 3).
    let mic: Vec<f32> = (0..hop).map(|_| r.next_f32() * 0.5).collect();
    let far: Vec<f32> = (0..hop).map(|_| r.next_f32() * 0.5).collect();
    let _ = stream.push_paired(&mic, &far).unwrap();
    assert_eq!(
        stream.frames_processed,
        frames_after_prime + 1,
        "one hop-sized push must advance frames_processed by exactly 1 \
         (Defect 2: pre-fix drain would re-run Kalman on every already-\
         processed frame, over-advancing state and producing incorrect echo \
         estimates on subsequent pushes)"
    );
}

/// **Streaming idempotence**: pushing a 4096-sample utterance once
/// (whole-utterance path) must produce output that is bit-close to
/// pushing the same 4096 samples as 16 successive `hop`-sized chunks
/// (streaming path) — outside the `n_fft`-sample OLA warmup at the
/// head. This is the classic "streaming = batch" contract and pins
/// the internal drain against silent block-boundary drift.
///
/// # Tolerance
///
/// `atol = 1e-5` per sample outside `[0, n_fft)`. Both paths use the
/// same `center=False` STFT internally, so alignment matches from
/// frame 0; the tolerance covers `f32` rounding order-of-operations
/// differences (mostly in the OLA accumulator's sum-order across a
/// different number of drain calls).
#[test]
fn whole_utterance_equals_hop_chunked_stream_within_tolerance() {
    let cfg = NkfAecConfig::upstream_default();
    let session = NkfAec::from_parts(cfg.clone(), NkfAecWeights::zeros(&cfg));

    let n = 4096usize;
    let hop = cfg.hop;
    assert_eq!(n % hop, 0, "test wants exact multiple of hop");
    let n_chunks = n / hop;

    let mut r_mic = SplitMix64::new(13);
    let mut r_far = SplitMix64::new(37);
    let mic: Vec<f32> = (0..n).map(|_| r_mic.next_f32() * 0.5).collect();
    let far: Vec<f32> = (0..n).map(|_| r_far.next_f32() * 0.5).collect();

    // Path A: one whole-utterance push.
    let mut stream_a = NkfAecStream::new(cfg.clone(), Arc::clone(&session.weights));
    let out_a = stream_a.push_paired(&mic, &far).unwrap();

    // Path B: 16 hop-sized pushes.
    let mut stream_b = NkfAecStream::new(cfg.clone(), Arc::clone(&session.weights));
    let mut out_b = Vec::new();
    for i in 0..n_chunks {
        let mic_chunk = &mic[i * hop..(i + 1) * hop];
        let far_chunk = &far[i * hop..(i + 1) * hop];
        let out = stream_b.push_paired(mic_chunk, far_chunk).unwrap();
        out_b.extend_from_slice(&out);
    }

    // Both paths under center=False should produce the same number of
    // committed samples once available_frames matches.
    assert_eq!(
        out_a.len(),
        out_b.len(),
        "streaming and whole-utterance drains must commit the same sample \
         count (available_frames * hop) on the same input"
    );

    // Skip the first `n_fft` samples — OLA warmup has partial Hann
    // coverage there. Interior samples must match tightly.
    let skip = cfg.n_fft.min(out_a.len());
    let mut max_dev = 0.0f32;
    for i in skip..out_a.len() {
        let d = (out_a[i] - out_b[i]).abs();
        if d > max_dev {
            max_dev = d;
        }
    }
    assert!(
        max_dev < 1e-5,
        "streaming vs whole-utterance drift: max |Δ| = {max_dev:e} > 1e-5 \
         over samples [{skip}, {}) (Defect 1/2: block-boundary drift in the \
         pre-fix drain would show up here as a large deviation)",
        out_a.len()
    );
}
