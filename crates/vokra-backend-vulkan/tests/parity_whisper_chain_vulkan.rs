//! M4-13-T13 — Whisper base **model-level** parity: the encoder + decoder
//! kernel chain executed end-to-end on the Vulkan backend's SPIR-V kernels
//! and compared against the CPU backend running the *identical* chain.
//!
//! # What "model-level" means here (honest scope)
//!
//! The graph `OpKind` enum cannot express Whisper (no LayerNorm / Gelu /
//! Conv1D / Gemv / SoftmaxCausal variants), so — exactly like the Metal
//! (M2-01) and CUDA (M2-03) model paths — the encoder/decoder is an
//! **imperative chain of per-kernel dispatches** (surface 2 of the
//! M4-13-T01 two-surface distinction): conv stem → (transpose, +pos) →
//! [ln → attention(gemm/transpose/mul/softmax) → residual → ln →
//! MLP(gemm/gelu/gemm) → residual] → ln_post, then a decoder step with
//! causal self-attention, cross-attention over the encoder output, and the
//! tied-embedding GEMV logits head.
//!
//! **Weights are synthesized** (SplitMix64, Xavier-ish 1/sqrt(fan_in) — the
//! M3-09 pattern): the real Whisper base GGUF is not committed to the repo,
//! so PyTorch-fixture-level verification of the *weights* is the owner's
//! flip-the-switch run (convert checkpoint → GGUF → the vokra-models parity
//! suite). What THIS test proves is the M4-13 claim that matters for the
//! backend: the Vulkan kernel chain at Whisper base dimensions computes the
//! same function as the CPU chain — whose kernels ARE PyTorch-anchored by
//! M0-06/M0-08 — within FP32 atol = 0.01 (NFR-QL-01) end-to-end, error
//! accumulation included.
//!
//! # Oracle-drift prevention
//!
//! Both executions share ONE chain function, generic over a small `Ops`
//! trait with a Vulkan impl (SPIR-V kernels) and a CPU impl
//! (`vokra-backend-cpu` kernels + trivial host moves). The two cannot fall
//! out of step structurally; only the kernel arithmetic differs.
//!
//! # Gating
//!
//! Skips cleanly (with the missing list logged) unless a Vulkan device is
//! present AND every required `.spv` is committed (owner M4-13-T16) —
//! never a fabricated pass. lavapipe CI exercises it after the blob commit.

use vokra_backend_vulkan::plan::{Conv1dDims, ElementwiseOp};
use vokra_backend_vulkan::{GemmPipelinePreference, VulkanBackend, spirv};
use vokra_core::Result;

/// FP32 parity gate (NFR-QL-01) applied to the CHAIN outputs (accumulated
/// FP32 divergence across every stage stays far below this in practice;
/// the max |Δ| is logged for visibility).
const ATOL: f32 = 0.01;

// Whisper base dimensions (M0-06) with a shortened time axis.
const D_MODEL: usize = 512;
const N_HEAD: usize = 8;
const HEAD_DIM: usize = 64; // 512 / 8
const N_MELS: usize = 80;
const MLP: usize = 2048;
const AUDIO_LEN: usize = 64; // mel frames fed to conv1 (real: 3000)
const ENC_CTX: usize = 32; // after conv2's stride 2 (real: 1500)
const DEC_CTX: usize = 4; // decoder prefill length
const VOCAB: usize = 256; // vocab slice for the tied-embedding head
const LN_EPS: f32 = 1e-5; // PyTorch nn.LayerNorm default, Whisper's value
const QK_SCALE: f32 = 0.125; // 1 / sqrt(HEAD_DIM)

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

fn splitmix_weights(seed: u64, len: usize, fan_in: usize) -> Vec<f32> {
    let scale = 1.0 / (fan_in as f32).sqrt();
    splitmix_f32s(seed, len).iter().map(|v| v * scale).collect()
}

// ---------------------------------------------------------------------------
// The Ops trait — one chain, two kernel providers.
// ---------------------------------------------------------------------------

trait Ops {
    fn conv1d(
        &self,
        dims: &Conv1dDims,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
    ) -> Result<Vec<f32>>;
    fn gelu(&self, x: &[f32]) -> Result<Vec<f32>>;
    fn layer_norm(&self, rows: usize, x: &[f32], gamma: &[f32], beta: &[f32]) -> Result<Vec<f32>>;
    fn gemm(&self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>>;
    fn softmax(&self, rows: usize, cols: usize, x: &[f32]) -> Result<Vec<f32>>;
    fn softmax_causal(&self, rows: usize, cols: usize, x: &[f32]) -> Result<Vec<f32>>;
    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>>;
    fn mul(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>>;
    fn transpose(&self, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>>;
    fn gather(&self, vocab: usize, dim: usize, table: &[f32], idx: &[u32]) -> Result<Vec<f32>>;
    fn gemv(&self, m: usize, n: usize, a: &[f32], x: &[f32]) -> Result<Vec<f32>>;
}

/// Vulkan provider — every op is a SPIR-V kernel dispatch.
struct VulkanOps<'a>(&'a VulkanBackend);

impl Ops for VulkanOps<'_> {
    fn conv1d(
        &self,
        dims: &Conv1dDims,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
    ) -> Result<Vec<f32>> {
        self.0.conv1d_f32(dims, input, weight, Some(bias))
    }
    fn gelu(&self, x: &[f32]) -> Result<Vec<f32>> {
        self.0.gelu_f32(x)
    }
    fn layer_norm(&self, rows: usize, x: &[f32], gamma: &[f32], beta: &[f32]) -> Result<Vec<f32>> {
        self.0.layer_norm_f32(rows, D_MODEL, LN_EPS, x, gamma, beta)
    }
    fn gemm(&self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        // ForceSubgroup: the Android-baseline pipeline is the parity target;
        // the coop-matrix variant is covered per-kernel in T12.
        self.0
            .gemm_f32(GemmPipelinePreference::ForceSubgroup, m, n, k, a, b)
    }
    fn softmax(&self, rows: usize, cols: usize, x: &[f32]) -> Result<Vec<f32>> {
        self.0.softmax_f32(rows, cols, x)
    }
    fn softmax_causal(&self, rows: usize, cols: usize, x: &[f32]) -> Result<Vec<f32>> {
        self.0.softmax_causal_f32(rows, cols, x)
    }
    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.0.elementwise_f32(ElementwiseOp::Add, a, b)
    }
    fn mul(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.0.elementwise_f32(ElementwiseOp::Mul, a, b)
    }
    fn transpose(&self, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>> {
        self.0.transpose_f32(m, n, x)
    }
    fn gather(&self, vocab: usize, dim: usize, table: &[f32], idx: &[u32]) -> Result<Vec<f32>> {
        self.0.gather_f32(vocab, dim, table, idx)
    }
    fn gemv(&self, m: usize, n: usize, a: &[f32], x: &[f32]) -> Result<Vec<f32>> {
        self.0.gemv_f32(m, n, a, x, None)
    }
}

/// CPU provider — the PyTorch-anchored differential oracle (M0-06/M0-08);
/// data movement (transpose / gather) is trivially host-side.
struct CpuOps;

impl Ops for CpuOps {
    fn conv1d(
        &self,
        dims: &Conv1dDims,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
    ) -> Result<Vec<f32>> {
        assert_eq!(dims.batch, 1, "chain runs batch 1 like the real front-end");
        let out_len = dims.out_len()?;
        let mut out = vec![0.0f32; dims.out_ch * out_len];
        vokra_backend_cpu::kernels::conv1d_f32(
            input,
            dims.in_ch,
            dims.in_len,
            weight,
            dims.out_ch,
            dims.kernel_len,
            Some(bias),
            dims.stride,
            dims.padding,
            &mut out,
        )?;
        Ok(out)
    }
    fn gelu(&self, x: &[f32]) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; x.len()];
        vokra_backend_cpu::kernels::gelu_f32(x, &mut out)?;
        Ok(out)
    }
    fn layer_norm(&self, rows: usize, x: &[f32], gamma: &[f32], beta: &[f32]) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; x.len()];
        vokra_backend_cpu::kernels::layer_norm_f32(
            x, &mut out, rows, D_MODEL, gamma, beta, LN_EPS,
        )?;
        Ok(out)
    }
    fn gemm(&self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; m * n];
        vokra_backend_cpu::kernels::gemm_f32(m, n, k, a, b, None, &mut out)?;
        Ok(out)
    }
    fn softmax(&self, rows: usize, cols: usize, x: &[f32]) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; x.len()];
        vokra_backend_cpu::kernels::softmax_f32(x, &mut out, rows, cols)?;
        Ok(out)
    }
    fn softmax_causal(&self, rows: usize, cols: usize, x: &[f32]) -> Result<Vec<f32>> {
        // Host mask + softmax — the exp(-inf) = 0 equivalence the GPU
        // kernels implement natively (CLAUDE.md causal contract).
        let mut masked = x.to_vec();
        for i in 0..rows {
            for j in (i + 1)..cols {
                masked[i * cols + j] = f32::NEG_INFINITY;
            }
        }
        self.softmax(rows, cols, &masked)
    }
    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; a.len()];
        vokra_backend_cpu::kernels::add_f32(a, b, &mut out)?;
        Ok(out)
    }
    fn mul(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; a.len()];
        vokra_backend_cpu::kernels::mul_f32(a, b, &mut out)?;
        Ok(out)
    }
    fn transpose(&self, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = x[i * n + j];
            }
        }
        Ok(out)
    }
    fn gather(&self, _vocab: usize, dim: usize, table: &[f32], idx: &[u32]) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(idx.len() * dim);
        for &i in idx {
            out.extend_from_slice(&table[i as usize * dim..(i as usize + 1) * dim]);
        }
        Ok(out)
    }
    fn gemv(&self, m: usize, n: usize, a: &[f32], x: &[f32]) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; m];
        vokra_backend_cpu::kernels::gemv_f32(m, n, a, x, None, &mut out)?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Synthetic Whisper-base-shaped weights (shared by both providers).
// ---------------------------------------------------------------------------

struct BlockWeights {
    ln1_g: Vec<f32>,
    ln1_b: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    ln2_g: Vec<f32>,
    ln2_b: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
}

impl BlockWeights {
    fn synthesize(seed: u64) -> Self {
        BlockWeights {
            // γ near 1, β near 0 (LN affine in a realistic band).
            ln1_g: splitmix_f32s(seed, D_MODEL)
                .iter()
                .map(|v| 1.0 + 0.1 * v)
                .collect(),
            ln1_b: splitmix_f32s(seed + 1, D_MODEL)
                .iter()
                .map(|v| 0.1 * v)
                .collect(),
            wq: splitmix_weights(seed + 2, D_MODEL * D_MODEL, D_MODEL),
            wk: splitmix_weights(seed + 3, D_MODEL * D_MODEL, D_MODEL),
            wv: splitmix_weights(seed + 4, D_MODEL * D_MODEL, D_MODEL),
            wo: splitmix_weights(seed + 5, D_MODEL * D_MODEL, D_MODEL),
            ln2_g: splitmix_f32s(seed + 6, D_MODEL)
                .iter()
                .map(|v| 1.0 + 0.1 * v)
                .collect(),
            ln2_b: splitmix_f32s(seed + 7, D_MODEL)
                .iter()
                .map(|v| 0.1 * v)
                .collect(),
            w_up: splitmix_weights(seed + 8, D_MODEL * MLP, D_MODEL),
            w_down: splitmix_weights(seed + 9, MLP * D_MODEL, MLP),
        }
    }
}

/// Host-side per-head column slice `[t, D_MODEL] → [t, HEAD_DIM]` (data
/// staging, not kernel math — same role the Metal/CUDA hosts play).
fn head_slice(x: &[f32], t: usize, head: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(t * HEAD_DIM);
    for row in 0..t {
        let base = row * D_MODEL + head * HEAD_DIM;
        out.extend_from_slice(&x[base..base + HEAD_DIM]);
    }
    out
}

/// Inverse of [`head_slice`]: writes a `[t, HEAD_DIM]` head block back into
/// the `[t, D_MODEL]` concat buffer.
fn head_unslice(dst: &mut [f32], src: &[f32], t: usize, head: usize) {
    for row in 0..t {
        let base = row * D_MODEL + head * HEAD_DIM;
        dst[base..base + HEAD_DIM].copy_from_slice(&src[row * HEAD_DIM..(row + 1) * HEAD_DIM]);
    }
}

/// Multi-head attention: `q_input` attends over `kv_input` (`t_q` × `t_kv`
/// rows), causal or not. Q·K^T scaling is folded into Q via `mul` with a
/// constant tensor (exercises the elementwise-mul kernel).
fn attention<O: Ops>(
    ops: &O,
    w: &BlockWeights,
    q_input: &[f32],
    kv_input: &[f32],
    t_q: usize,
    t_kv: usize,
    causal: bool,
) -> Result<Vec<f32>> {
    let q = ops.gemm(t_q, D_MODEL, D_MODEL, q_input, &w.wq)?;
    let k = ops.gemm(t_kv, D_MODEL, D_MODEL, kv_input, &w.wk)?;
    let v = ops.gemm(t_kv, D_MODEL, D_MODEL, kv_input, &w.wv)?;
    let scale = vec![QK_SCALE; t_q * HEAD_DIM];
    let mut concat = vec![0.0f32; t_q * D_MODEL];
    for h in 0..N_HEAD {
        let qh = ops.mul(&head_slice(&q, t_q, h), &scale)?;
        let kh_t = ops.transpose(t_kv, HEAD_DIM, &head_slice(&k, t_kv, h))?;
        let scores = ops.gemm(t_q, t_kv, HEAD_DIM, &qh, &kh_t)?;
        let probs = if causal {
            ops.softmax_causal(t_q, t_kv, &scores)?
        } else {
            ops.softmax(t_q, t_kv, &scores)?
        };
        let ctx = ops.gemm(t_q, HEAD_DIM, t_kv, &probs, &head_slice(&v, t_kv, h))?;
        head_unslice(&mut concat, &ctx, t_q, h);
    }
    ops.gemm(t_q, D_MODEL, D_MODEL, &concat, &w.wo)
}

/// One pre-norm transformer block (self-attention only — the encoder
/// shape); returns the residual-updated activations.
fn encoder_block<O: Ops>(ops: &O, w: &BlockWeights, x: &[f32], t: usize) -> Result<Vec<f32>> {
    let normed = ops.layer_norm(t, x, &w.ln1_g, &w.ln1_b)?;
    let attn = attention(ops, w, &normed, &normed, t, t, false)?;
    let x = ops.add(x, &attn)?;
    let normed = ops.layer_norm(t, &x, &w.ln2_g, &w.ln2_b)?;
    let up = ops.gemm(t, MLP, D_MODEL, &normed, &w.w_up)?;
    let act = ops.gelu(&up)?;
    let down = ops.gemm(t, D_MODEL, MLP, &act, &w.w_down)?;
    ops.add(&x, &down)
}

struct ChainWeights {
    conv1_w: Vec<f32>,
    conv1_b: Vec<f32>,
    conv2_w: Vec<f32>,
    conv2_b: Vec<f32>,
    enc_pos: Vec<f32>,
    enc_block: BlockWeights,
    ln_post_g: Vec<f32>,
    ln_post_b: Vec<f32>,
    token_emb: Vec<f32>,
    dec_pos: Vec<f32>,
    dec_self: BlockWeights,
    dec_cross: BlockWeights,
    ln_final_g: Vec<f32>,
    ln_final_b: Vec<f32>,
}

impl ChainWeights {
    fn synthesize() -> Self {
        ChainWeights {
            conv1_w: splitmix_weights(1000, D_MODEL * N_MELS * 3, N_MELS * 3),
            conv1_b: splitmix_f32s(1001, D_MODEL)
                .iter()
                .map(|v| 0.1 * v)
                .collect(),
            conv2_w: splitmix_weights(1002, D_MODEL * D_MODEL * 3, D_MODEL * 3),
            conv2_b: splitmix_f32s(1003, D_MODEL)
                .iter()
                .map(|v| 0.1 * v)
                .collect(),
            enc_pos: splitmix_weights(1004, ENC_CTX * D_MODEL, D_MODEL),
            enc_block: BlockWeights::synthesize(2000),
            ln_post_g: splitmix_f32s(1005, D_MODEL)
                .iter()
                .map(|v| 1.0 + 0.1 * v)
                .collect(),
            ln_post_b: splitmix_f32s(1006, D_MODEL)
                .iter()
                .map(|v| 0.1 * v)
                .collect(),
            token_emb: splitmix_weights(1007, VOCAB * D_MODEL, D_MODEL),
            dec_pos: splitmix_weights(1008, DEC_CTX * D_MODEL, D_MODEL),
            dec_self: BlockWeights::synthesize(3000),
            dec_cross: BlockWeights::synthesize(4000),
            ln_final_g: splitmix_f32s(1009, D_MODEL)
                .iter()
                .map(|v| 1.0 + 0.1 * v)
                .collect(),
            ln_final_b: splitmix_f32s(1010, D_MODEL)
                .iter()
                .map(|v| 0.1 * v)
                .collect(),
        }
    }
}

/// Whisper base encoder chain: conv stem (k3 s1p1 → gelu → k3 s2p1 → gelu)
/// → time-major transpose → +pos → transformer block → ln_post.
fn encoder_chain<O: Ops>(ops: &O, w: &ChainWeights, mel: &[f32]) -> Result<Vec<f32>> {
    let conv1 = Conv1dDims {
        batch: 1,
        in_ch: N_MELS,
        out_ch: D_MODEL,
        in_len: AUDIO_LEN,
        kernel_len: 3,
        stride: 1,
        padding: 1,
    };
    let x = ops.conv1d(&conv1, mel, &w.conv1_w, &w.conv1_b)?;
    let x = ops.gelu(&x)?;
    let conv2 = Conv1dDims {
        in_ch: D_MODEL,
        stride: 2,
        ..conv1
    };
    let x = ops.conv1d(&conv2, &x, &w.conv2_w, &w.conv2_b)?;
    let x = ops.gelu(&x)?;
    // [D_MODEL, ENC_CTX] channel-major → [ENC_CTX, D_MODEL] time-major.
    let x = ops.transpose(D_MODEL, ENC_CTX, &x)?;
    let x = ops.add(&x, &w.enc_pos)?;
    let x = encoder_block(ops, &w.enc_block, &x, ENC_CTX)?;
    ops.layer_norm(ENC_CTX, &x, &w.ln_post_g, &w.ln_post_b)
}

/// Whisper base decoder prefill step over `tokens`, cross-attending the
/// encoder output; returns the tied-embedding logits of the LAST position.
fn decoder_chain<O: Ops>(
    ops: &O,
    w: &ChainWeights,
    tokens: &[u32],
    enc_out: &[f32],
) -> Result<Vec<f32>> {
    let x = ops.gather(VOCAB, D_MODEL, &w.token_emb, tokens)?;
    let x = ops.add(&x, &w.dec_pos)?;
    // Self-attention (causal) block.
    let normed = ops.layer_norm(DEC_CTX, &x, &w.dec_self.ln1_g, &w.dec_self.ln1_b)?;
    let attn = attention(ops, &w.dec_self, &normed, &normed, DEC_CTX, DEC_CTX, true)?;
    let x = ops.add(&x, &attn)?;
    // Cross-attention block (queries from the decoder, keys/values from the
    // encoder output — non-causal).
    let normed = ops.layer_norm(DEC_CTX, &x, &w.dec_cross.ln1_g, &w.dec_cross.ln1_b)?;
    let attn = attention(ops, &w.dec_cross, &normed, enc_out, DEC_CTX, ENC_CTX, false)?;
    let x = ops.add(&x, &attn)?;
    // MLP block (weights from dec_self's MLP half).
    let normed = ops.layer_norm(DEC_CTX, &x, &w.dec_self.ln2_g, &w.dec_self.ln2_b)?;
    let up = ops.gemm(DEC_CTX, MLP, D_MODEL, &normed, &w.dec_self.w_up)?;
    let act = ops.gelu(&up)?;
    let down = ops.gemm(DEC_CTX, D_MODEL, MLP, &act, &w.dec_self.w_down)?;
    let x = ops.add(&x, &down)?;
    let x = ops.layer_norm(DEC_CTX, &x, &w.ln_final_g, &w.ln_final_b)?;
    // Tied-embedding logits head for the last position (the decode hot
    // path): logits[v] = emb[v, :] · h_last.
    let h_last = &x[(DEC_CTX - 1) * D_MODEL..DEC_CTX * D_MODEL];
    ops.gemv(VOCAB, D_MODEL, &w.token_emb, h_last)
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

/// Every SPIR-V kernel the chain dispatches (activation is not part of the
/// Whisper path — it is covered per-kernel by T12 / kernel_dispatch).
const REQUIRED_BLOBS: [&str; 9] = [
    "conv1d",
    "gelu",
    "layer_norm",
    "gemm_subgroup",
    "softmax",
    "softmax_causal",
    "elementwise",
    "transpose",
    "gather",
];

#[test]
fn whisper_base_encoder_and_decoder_chain_matches_cpu() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("skip whisper chain: no Vulkan on this host");
        return;
    };
    let missing: Vec<&&str> = REQUIRED_BLOBS
        .iter()
        .filter(|n| !spirv::has_blob(n))
        .collect();
    if !missing.is_empty() {
        eprintln!("skip whisper chain: .spv not committed yet for {missing:?} (owner M4-13-T16)");
        return;
    }

    let w = ChainWeights::synthesize();
    let mel = splitmix_f32s(5000, N_MELS * AUDIO_LEN);
    let tokens: Vec<u32> = vec![0, 17, 255, 42];

    // Vulkan chain vs CPU chain — the SAME chain function, two providers.
    let vk = VulkanOps(&backend);
    let enc_vk = encoder_chain(&vk, &w, &mel).expect("Vulkan encoder chain");
    let enc_cpu = encoder_chain(&CpuOps, &w, &mel).expect("CPU encoder chain");
    assert_eq!(enc_vk.len(), ENC_CTX * D_MODEL);
    assert_close(&enc_vk, &enc_cpu, "encoder output [32, 512]");

    // Decoder consumes ITS OWN backend's encoder output, exactly like a
    // real inference (no cross-feeding, so divergence compounds honestly).
    let logits_vk = decoder_chain(&vk, &w, &tokens, &enc_vk).expect("Vulkan decoder chain");
    let logits_cpu = decoder_chain(&CpuOps, &w, &tokens, &enc_cpu).expect("CPU decoder chain");
    assert_eq!(logits_vk.len(), VOCAB);
    assert_close(
        &logits_vk,
        &logits_cpu,
        "decoder last-position logits [256]",
    );

    // The two chains must also agree on the argmax token — the actual
    // decode decision (a rank flip within atol would be caught here).
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap()
    };
    assert_eq!(
        argmax(&logits_vk),
        argmax(&logits_cpu),
        "greedy argmax must agree between the Vulkan and CPU chains"
    );
}

/// The CPU oracle chain itself must be well-formed on EVERY host (this is
/// what the Vulkan chain is compared against, so a shape/plumbing bug here
/// would poison the parity signal). Runs on the Apple authoring host too —
/// no Vulkan needed — and pins shapes, finiteness and determinism.
#[test]
fn cpu_oracle_chain_is_self_consistent_on_any_host() {
    let w = ChainWeights::synthesize();
    let mel = splitmix_f32s(5000, N_MELS * AUDIO_LEN);
    let tokens: Vec<u32> = vec![0, 17, 255, 42];

    let enc = encoder_chain(&CpuOps, &w, &mel).expect("CPU encoder chain");
    assert_eq!(enc.len(), ENC_CTX * D_MODEL, "encoder output shape");
    assert!(
        enc.iter().all(|v| v.is_finite()),
        "encoder output must be finite (weight scaling keeps the chain conditioned)"
    );

    let logits = decoder_chain(&CpuOps, &w, &tokens, &enc).expect("CPU decoder chain");
    assert_eq!(logits.len(), VOCAB, "logits shape");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits must be finite"
    );

    // Determinism: the full chain is a pure function of its inputs.
    let enc2 = encoder_chain(&CpuOps, &w, &mel).expect("CPU encoder chain rerun");
    let logits2 = decoder_chain(&CpuOps, &w, &tokens, &enc2).expect("CPU decoder chain rerun");
    for (a, b) in enc.iter().zip(&enc2) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "encoder chain must be deterministic"
        );
    }
    for (a, b) in logits.iter().zip(&logits2) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "decoder chain must be deterministic"
        );
    }
}
