//! M4-13-T12 — Whisper base Vulkan parity harness, per-kernel differential.
//!
//! Two-stage posture (M3-02-T32 / M4-13-T12):
//!
//! 1. **Primary anchor — PyTorch reference**: the CPU backend's kernels are
//!    the differential oracle here, and those kernels are themselves
//!    PyTorch-anchored by the M0-06 fixtures + M0-08 parity suite
//!    (`tests/parity/whisper_base/`). Direct fixture comparison at the
//!    *model* level additionally needs the real Whisper base weights
//!    (GGUF — not committed; converted locally by the owner), so that leg
//!    lives with the owner flip-the-switch runs; the kernel-level truth
//!    is fully checkable without weights.
//! 2. **Differential — Vulkan vs CPU**: every SPIR-V kernel is compared
//!    against the CPU kernel of identical shape/semantics at **Whisper
//!    base dimensions** (d_model = 512, n_head = 8, head_dim = 64,
//!    n_mels = 80, MLP = 2048, conv k=3 with the s=1/p=1 + s=2/p=1
//!    envelope), FP32 atol = 0.01 (NFR-QL-01). Time axes are shortened
//!    (ctx = 64) so lavapipe CI stays fast — the channel dimensions, which
//!    drive the reduction depths and therefore the FP32 error budget, are
//!    the real ones. Full-length runs are the owner's Android soak
//!    (M4-13-T17).
//!
//! Gating (fabricated-pass prevention): every test skips cleanly with a
//! logged reason when (a) no Vulkan device is present (Apple authoring
//! host), or (b) the kernel's glslc `.spv` has not been committed yet
//! (owner M4-13-T16) — `spirv::has_blob` is the gate, so the whole file
//! lights up automatically after the blob commit. See
//! `tests/parity_whisper_chain_vulkan.rs` (M4-13-T13) for the chained
//! model-level counterpart.

use vokra_backend_vulkan::plan::Conv1dDims;
use vokra_backend_vulkan::{GemmPipelinePreference, VulkanBackend, spirv};

/// FP32 parity gate (NFR-QL-01).
const ATOL: f32 = 0.01;

// Whisper base dimensions (M0-06; encoder/decoder d_model 512, 8 heads).
const D_MODEL: usize = 512;
const HEAD_DIM: usize = 64;
const N_MELS: usize = 80;
const MLP: usize = 2048;
/// Shortened time axis (real encoder ctx is 1500) — reductions run over the
/// channel axes, so the FP32 error budget is exercised at full depth.
const CTX: usize = 64;

fn splitmix_f32s(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            ((z >> 40) as f32) / ((1u64 << 23) as f32) - 1.0
        })
        .collect()
}

/// Xavier-ish scaling for weight-like tensors so deep reductions stay in a
/// well-conditioned range (the M3-09 synthesized-weights pattern).
fn splitmix_weights(seed: u64, len: usize, fan_in: usize) -> Vec<f32> {
    let scale = 1.0 / (fan_in as f32).sqrt();
    splitmix_f32s(seed, len).iter().map(|v| v * scale).collect()
}

fn backend_or_skip(what: &str) -> Option<VulkanBackend> {
    match VulkanBackend::new() {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skip {what}: no Vulkan on this host ({e})");
            None
        }
    }
}

/// `true` iff every named blob is committed; logs the missing set otherwise
/// (clean skip, visible in the CI log — never a fabricated pass).
fn blobs_or_skip(what: &str, names: &[&str]) -> bool {
    let missing: Vec<&&str> = names.iter().filter(|n| !spirv::has_blob(n)).collect();
    if missing.is_empty() {
        true
    } else {
        eprintln!("skip {what}: .spv not committed yet for {missing:?} (owner M4-13-T16)");
        false
    }
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let mut max_abs = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        assert!(
            d <= ATOL,
            "{what}: diverged at {i}: got {g}, want {w} (|Δ| = {d} > atol {ATOL})"
        );
        max_abs = max_abs.max(d);
    }
    eprintln!("{what}: max |Δ| = {max_abs:.3e} (atol {ATOL})");
}

/// Attention QKV / out projections: `[ctx, 512] @ [512, 512]` on BOTH GEMM
/// variants (the probe-selected one and the forced Android baseline).
#[test]
fn whisper_gemm_projection_parity() {
    let Some(backend) = backend_or_skip("whisper_gemm_projection") else {
        return;
    };
    let x = splitmix_f32s(100, CTX * D_MODEL);
    let w = splitmix_weights(101, D_MODEL * D_MODEL, D_MODEL);
    let mut want = vec![0.0f32; CTX * D_MODEL];
    vokra_backend_cpu::kernels::gemm_f32(CTX, D_MODEL, D_MODEL, &x, &w, None, &mut want)
        .expect("CPU reference");

    for (pref, shader, label) in [
        (
            GemmPipelinePreference::ForceSubgroup,
            "gemm_subgroup",
            "subgroup",
        ),
        (
            GemmPipelinePreference::default(),
            backend
                .select_gemm_pipeline_variant(GemmPipelinePreference::default())
                .expect("default never errors")
                .shader_name(),
            "probe-selected",
        ),
    ] {
        if !blobs_or_skip("whisper_gemm_projection", &[shader]) {
            continue;
        }
        let got = backend
            .gemm_f32(pref, CTX, D_MODEL, D_MODEL, &x, &w)
            .expect("blob committed; gemm must dispatch");
        assert_close(
            &got,
            &want,
            &format!("gemm[{label}] {CTX}x{D_MODEL}x{D_MODEL}"),
        );
    }
}

/// MLP up/down projections: `[ctx, 512] @ [512, 2048]` and
/// `[ctx, 2048] @ [2048, 512]` — the deepest reductions in Whisper base.
#[test]
fn whisper_gemm_mlp_parity() {
    let Some(backend) = backend_or_skip("whisper_gemm_mlp") else {
        return;
    };
    if !blobs_or_skip("whisper_gemm_mlp", &["gemm_subgroup"]) {
        return;
    }
    // Up: 512 → 2048.
    let x = splitmix_f32s(110, CTX * D_MODEL);
    let w_up = splitmix_weights(111, D_MODEL * MLP, D_MODEL);
    let got = backend
        .gemm_f32(
            GemmPipelinePreference::ForceSubgroup,
            CTX,
            MLP,
            D_MODEL,
            &x,
            &w_up,
        )
        .expect("gemm up");
    let mut want = vec![0.0f32; CTX * MLP];
    vokra_backend_cpu::kernels::gemm_f32(CTX, MLP, D_MODEL, &x, &w_up, None, &mut want)
        .expect("CPU reference");
    assert_close(&got, &want, "gemm mlp-up");

    // Down: 2048 → 512 (k = 2048, the deepest FP32 accumulation).
    let h = splitmix_f32s(112, CTX * MLP);
    let w_down = splitmix_weights(113, MLP * D_MODEL, MLP);
    let got = backend
        .gemm_f32(
            GemmPipelinePreference::ForceSubgroup,
            CTX,
            D_MODEL,
            MLP,
            &h,
            &w_down,
        )
        .expect("gemm down");
    let mut want = vec![0.0f32; CTX * D_MODEL];
    vokra_backend_cpu::kernels::gemm_f32(CTX, D_MODEL, MLP, &h, &w_down, None, &mut want)
        .expect("CPU reference");
    assert_close(&got, &want, "gemm mlp-down");
}

/// Decoder-step logits head: `emb[v, 512] @ h[512]` via GEMV (the tied
/// embedding GEMV is THE decode hot path). Vocab slice of 8192 rows keeps
/// lavapipe fast; the real 51865-row head fits the same one-workgroup-per-
/// row dispatch (< 65535).
#[test]
fn whisper_gemv_logits_head_parity() {
    let Some(backend) = backend_or_skip("whisper_gemv_logits") else {
        return;
    };
    if !blobs_or_skip("whisper_gemv_logits", &["gemv"]) {
        return;
    }
    let vocab_slice = 8192usize;
    let emb = splitmix_weights(120, vocab_slice * D_MODEL, D_MODEL);
    let h = splitmix_f32s(121, D_MODEL);
    let got = backend
        .gemv_f32(vocab_slice, D_MODEL, &emb, &h, None)
        .expect("gemv logits");
    let mut want = vec![0.0f32; vocab_slice];
    vokra_backend_cpu::kernels::gemv_f32(vocab_slice, D_MODEL, &emb, &h, None, &mut want)
        .expect("CPU reference");
    assert_close(&got, &want, "gemv logits head (8192-row vocab slice)");
}

/// Encoder attention probabilities: non-causal softmax over `[ctx, ctx]`
/// score rows.
#[test]
fn whisper_softmax_attention_parity() {
    let Some(backend) = backend_or_skip("whisper_softmax") else {
        return;
    };
    if !blobs_or_skip("whisper_softmax", &["softmax"]) {
        return;
    }
    // Scores scaled like Whisper (Q·K^T / sqrt(64)) stay in a moderate
    // range; ±4 stretches the exp() spread.
    let scores: Vec<f32> = splitmix_f32s(130, CTX * CTX)
        .iter()
        .map(|v| v * 4.0)
        .collect();
    let got = backend
        .softmax_f32(CTX, CTX, &scores)
        .expect("softmax scores");
    let mut want = vec![0.0f32; CTX * CTX];
    vokra_backend_cpu::kernels::softmax_f32(&scores, &mut want, CTX, CTX).expect("CPU reference");
    assert_close(&got, &want, "softmax attention probs");
}

/// Decoder self-attention probabilities: causal softmax vs a host-masked
/// CPU softmax (`exp(-inf) = 0` equivalence), masked cols exactly 0.
#[test]
fn whisper_softmax_causal_attention_parity() {
    let Some(backend) = backend_or_skip("whisper_softmax_causal") else {
        return;
    };
    if !blobs_or_skip("whisper_softmax_causal", &["softmax_causal"]) {
        return;
    }
    let scores: Vec<f32> = splitmix_f32s(131, CTX * CTX)
        .iter()
        .map(|v| v * 4.0)
        .collect();
    let got = backend
        .softmax_causal_f32(CTX, CTX, &scores)
        .expect("softmax_causal scores");
    let mut masked = scores.clone();
    for i in 0..CTX {
        for j in (i + 1)..CTX {
            masked[i * CTX + j] = f32::NEG_INFINITY;
        }
    }
    let mut want = vec![0.0f32; CTX * CTX];
    vokra_backend_cpu::kernels::softmax_f32(&masked, &mut want, CTX, CTX).expect("CPU reference");
    assert_close(&got, &want, "softmax_causal attention probs");
    for i in 0..CTX {
        for j in (i + 1)..CTX {
            assert_eq!(got[i * CTX + j].to_bits(), 0.0f32.to_bits());
        }
    }
}

/// Pre-attention / pre-MLP layer norms: `[ctx, 512]` rows with the
/// PyTorch-default eps Whisper uses (1e-5) passed through verbatim.
#[test]
fn whisper_layer_norm_parity() {
    let Some(backend) = backend_or_skip("whisper_layer_norm") else {
        return;
    };
    if !blobs_or_skip("whisper_layer_norm", &["layer_norm"]) {
        return;
    }
    let eps = 1e-5f32;
    let x = splitmix_f32s(140, CTX * D_MODEL);
    let gamma = splitmix_f32s(141, D_MODEL);
    let beta = splitmix_f32s(142, D_MODEL);
    let got = backend
        .layer_norm_f32(CTX, D_MODEL, eps, &x, &gamma, &beta)
        .expect("layer_norm");
    let mut want = vec![0.0f32; CTX * D_MODEL];
    vokra_backend_cpu::kernels::layer_norm_f32(&x, &mut want, CTX, D_MODEL, &gamma, &beta, eps)
        .expect("CPU reference");
    assert_close(&got, &want, "layer_norm [ctx, 512]");
}

/// MLP GELU at Whisper width: `[ctx, 2048]` — exact/erf form on both sides
/// (shared A&S 7.1.26 coefficients).
#[test]
fn whisper_gelu_parity() {
    let Some(backend) = backend_or_skip("whisper_gelu") else {
        return;
    };
    if !blobs_or_skip("whisper_gelu", &["gelu"]) {
        return;
    }
    let x: Vec<f32> = splitmix_f32s(150, CTX * MLP)
        .iter()
        .map(|v| v * 4.0)
        .collect();
    let got = backend.gelu_f32(&x).expect("gelu");
    let mut want = vec![0.0f32; x.len()];
    vokra_backend_cpu::kernels::gelu_f32(&x, &mut want).expect("CPU reference");
    assert_close(&got, &want, "gelu [ctx, 2048]");
}

/// Conv stem at Whisper channel counts: conv1 (80 → 512, k3 s1 p1) and
/// conv2 (512 → 512, k3 s2 p1), batch = 1 like the real front-end.
#[test]
fn whisper_conv_stem_parity() {
    let Some(backend) = backend_or_skip("whisper_conv_stem") else {
        return;
    };
    if !blobs_or_skip("whisper_conv_stem", &["conv1d"]) {
        return;
    }
    let in_len = 2 * CTX; // conv2's s=2 halves it back to CTX
    // conv1: 80 → 512, s=1, p=1.
    let conv1 = Conv1dDims {
        batch: 1,
        in_ch: N_MELS,
        out_ch: D_MODEL,
        in_len,
        kernel_len: 3,
        stride: 1,
        padding: 1,
    };
    let mel = splitmix_f32s(160, N_MELS * in_len);
    let w1 = splitmix_weights(161, D_MODEL * N_MELS * 3, N_MELS * 3);
    let b1 = splitmix_f32s(162, D_MODEL);
    let got1 = backend
        .conv1d_f32(&conv1, &mel, &w1, Some(&b1))
        .expect("conv1");
    let mut want1 = vec![0.0f32; D_MODEL * in_len];
    vokra_backend_cpu::kernels::conv1d_f32(
        &mel,
        N_MELS,
        in_len,
        &w1,
        D_MODEL,
        3,
        Some(&b1),
        1,
        1,
        &mut want1,
    )
    .expect("CPU reference");
    assert_close(&got1, &want1, "conv1 (80→512, s1 p1)");

    // conv2: 512 → 512, s=2, p=1 (halves the time axis).
    let conv2 = Conv1dDims {
        in_ch: D_MODEL,
        stride: 2,
        ..conv1
    };
    let out_len2 = conv2.out_len().expect("valid dims");
    assert_eq!(out_len2, CTX, "s=2 halves 2*CTX to CTX");
    let w2 = splitmix_weights(163, D_MODEL * D_MODEL * 3, D_MODEL * 3);
    let b2 = splitmix_f32s(164, D_MODEL);
    let got2 = backend
        .conv1d_f32(&conv2, &want1, &w2, Some(&b2))
        .expect("conv2");
    let mut want2 = vec![0.0f32; D_MODEL * out_len2];
    vokra_backend_cpu::kernels::conv1d_f32(
        &want1,
        D_MODEL,
        in_len,
        &w2,
        D_MODEL,
        3,
        Some(&b2),
        2,
        1,
        &mut want2,
    )
    .expect("CPU reference");
    assert_close(&got2, &want2, "conv2 (512→512, s2 p1)");
}

/// Attention `K^T` at per-head shape: `[ctx, 64] → [64, ctx]`; pure data
/// movement, bit-verbatim.
#[test]
fn whisper_transpose_head_parity() {
    let Some(backend) = backend_or_skip("whisper_transpose") else {
        return;
    };
    if !blobs_or_skip("whisper_transpose", &["transpose"]) {
        return;
    }
    let k = splitmix_f32s(170, CTX * HEAD_DIM);
    let got = backend.transpose_f32(CTX, HEAD_DIM, &k).expect("transpose");
    for i in 0..CTX {
        for j in 0..HEAD_DIM {
            assert_eq!(
                got[j * CTX + i].to_bits(),
                k[i * HEAD_DIM + j].to_bits(),
                "K^T must be a verbatim move at ({i},{j})"
            );
        }
    }
}

/// Token-embedding gather at Whisper d_model: rows come back bit-verbatim.
#[test]
fn whisper_gather_token_embedding_parity() {
    let Some(backend) = backend_or_skip("whisper_gather") else {
        return;
    };
    if !blobs_or_skip("whisper_gather", &["gather"]) {
        return;
    }
    let vocab_slice = 1024usize;
    let table = splitmix_weights(180, vocab_slice * D_MODEL, D_MODEL);
    let indices: Vec<u32> = vec![0, 1023, 50, 50, 256, 7];
    let got = backend
        .gather_f32(vocab_slice, D_MODEL, &table, &indices)
        .expect("gather");
    for (row, &idx) in indices.iter().enumerate() {
        let g = &got[row * D_MODEL..(row + 1) * D_MODEL];
        let w = &table[idx as usize * D_MODEL..(idx as usize + 1) * D_MODEL];
        for (c, (gv, wv)) in g.iter().zip(w).enumerate() {
            assert_eq!(gv.to_bits(), wv.to_bits(), "row {row} col {c} verbatim");
        }
    }
}
