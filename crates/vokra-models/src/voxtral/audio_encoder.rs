//! Voxtral audio encoder — Whisper-derived pre-norm transformer.
//!
//! Structurally identical to `WhisperEncoder` (see
//! [`super::super::whisper::encoder`]); the upstream reference is HF
//! `transformers/models/voxtral/modeling_voxtral.py` (transformers 4.57.6,
//! `VoxtralEncoder.forward`, lines 343–366):
//!
//! 1. `conv1` (`n_mels → d`, kernel 3, stride 1, pad 1) + GELU (line 350);
//! 2. `conv2` (`d → d`, kernel 3, stride 2, pad 1) + GELU (line 351, halves
//!    the time axis: `2·n_ctx` mel frames → `n_ctx` positions);
//! 3. transpose (`permute(0, 2, 1)`, line 352) + **full learned positional
//!    embedding table** (`embed_pos = self.embed_positions.weight`, lines
//!    354–355 — the whole `[n_ctx, d]` table is added, which is why the
//!    input length is checked strictly, see below);
//! 4. `n_layer` pre-norm self-attention blocks (`VoxtralEncoderLayer.forward`,
//!    lines 202–216: `h += attn(ln(h))` then `h += fc2(gelu(fc1(ln(h))))`);
//! 5. final LayerNorm (`self.layer_norm`, line 305).
//!
//! Attention details mirrored from `VoxtralAttention` (== `WhisperAttention`):
//! `k_proj` has **no bias** (line 113), q/v/out do; the query is scaled by
//! `head_dim^-0.5` **before** the score matmul (line 138) — exactly what
//! [`super::super::whisper::nn::attention_from_kv_into`] does, so the block
//! math is driven through the *same* audited Whisper building blocks (one
//! implementation, two models).
//!
//! Note: upstream defines an `avg_pooler` on the encoder (line 307) but its
//! `forward` **never calls it** — the ×4 time reduction happens outside the
//! tower as a frame-stacking reshape (`get_audio_features`, line 452), which
//! this runtime implements as [`super::adapter::AdapterKind::FrameStackMlp`].
//!
//! # Strict input-length contract (FR-EX-08)
//!
//! Upstream hard-errors unless the mel input is exactly
//! `max_source_positions * conv1.stride * conv2.stride` frames
//! (`expected_seq_length` check, lines 343–347) — the positional table is
//! added without slicing. This runtime mirrors that: `forward` rejects any
//! log-mel whose post-conv length differs from `cfg.audio.n_ctx` instead of
//! silently zero-padding the positional rows (the pre-M4 conv-stem stub
//! tolerated arbitrary lengths; that lenient path is gone with the real
//! transformer stack).
//!
//! All matmul / conv / norm / activation goes through the shared compute
//! seam ([`crate::compute::Compute`]) so a GPU backend dispatches the same
//! kernels without a second implementation, and the M5-14 packed-GEMM CPU
//! driver accelerates this stack automatically.

use vokra_backend_cpu::kernels::LAYER_NORM_DEFAULT_EPS;
use vokra_core::gguf::{FrontendPolicy, FrontendSpec, GgufFile};
use vokra_core::{PrenormLayer, Result, VokraError};

use super::VoxtralConfig;
use crate::whisper::encoder::{encoder_block, prenorm_view};
use crate::whisper::nn::layer_norm_into;
use crate::whisper::scratch::EncoderScratch;
use crate::whisper::weights::{self as ww, EncoderLayer, LayerNorm};

/// Encoder hidden states, `[n_ctx, hidden_dim]` row-major.
#[derive(Debug, Clone)]
pub struct AudioEncoderOutput {
    /// Row-major `[n_ctx, hidden_dim]`.
    pub hidden: Vec<f32>,
    /// Number of audio context positions.
    pub n_ctx: usize,
    /// Encoder hidden width.
    pub hidden_dim: usize,
}

/// Parsed audio-encoder weights.
///
/// Every tensor the forward path touches is bound at load — a missing tensor
/// is surfaced by name (FR-EX-08), never silently absent.
pub struct AudioEncoder {
    /// Convolution stem, both convs stored as flat `Vec<f32>` in their
    /// safetensors shape (`[out, in, k]`).
    pub(crate) conv1_w: Vec<f32>,
    pub(crate) conv1_b: Vec<f32>,
    pub(crate) conv2_w: Vec<f32>,
    pub(crate) conv2_b: Vec<f32>,
    /// Learned positional embedding `[n_ctx, hidden_dim]` (sinusoidal
    /// releases have no tensor here; the load path falls through to an
    /// all-zero placeholder that the runtime later rebuilds — recorded in
    /// [`AudioEncoder::has_learned_pos_emb`] for the caller).
    pub(crate) pos_emb: Vec<f32>,
    /// Whether the checkpoint carries a learned positional embedding.
    pub(crate) has_learned_pos_emb: bool,
    /// The `n_layer` pre-norm transformer blocks
    /// (`audio_tower.layers.{i}.*`). Same struct the Whisper encoder binds —
    /// the block forward is shared (see the module docs).
    pub(crate) layers: Vec<EncoderLayer>,
    /// Final LayerNorm (`audio_tower.layer_norm.{weight,bias}`).
    pub(crate) ln_post: LayerNorm,
}

impl AudioEncoder {
    /// Binds every audio-encoder weight tensor the forward path needs.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] naming the offending tensor if it is
    /// missing, has an unsupported dtype, or has an unexpected shape.
    pub fn load(file: &GgufFile, cfg: &VoxtralConfig) -> Result<Self> {
        // A `hidden_dim == 0` audio config means "no encoder in this
        // GGUF" — the shape-only converter path uses this for
        // text-decoder-only re-encodes. Skip the load in that case; the
        // forward entry point will return an explicit error if invoked.
        if cfg.audio.hidden_dim == 0 || cfg.audio.n_layer == 0 {
            return Ok(Self {
                conv1_w: Vec::new(),
                conv1_b: Vec::new(),
                conv2_w: Vec::new(),
                conv2_b: Vec::new(),
                pos_emb: Vec::new(),
                has_learned_pos_emb: false,
                layers: Vec::new(),
                ln_post: LayerNorm {
                    gamma: Vec::new(),
                    beta: Vec::new(),
                },
            });
        }
        let d = cfg.audio.hidden_dim;
        let n_mels = cfg.audio.n_mels;
        let ff = cfg.audio.ffn_dim;
        if n_mels == 0 {
            return Err(bad("audio_encoder.n_mels must be non-zero".to_owned()));
        }
        if ff == 0 {
            return Err(bad(
                "audio_encoder.ffn_dim is 0 (shape-only converter sentinel) — the transformer \
                 stack needs the real fc1/fc2 width. Re-convert with a converter that writes \
                 vokra.voxtral.audio_encoder.ffn_dim (FR-EX-08 — no silent default)."
                    .to_owned(),
            ));
        }
        let conv1_w = tensor(file, "audio_tower.conv1.weight", &[d, n_mels, 3])?;
        let conv1_b = tensor(file, "audio_tower.conv1.bias", &[d])?;
        let conv2_w = tensor(file, "audio_tower.conv2.weight", &[d, d, 3])?;
        let conv2_b = tensor(file, "audio_tower.conv2.bias", &[d])?;
        // Positional embedding is optional — sinusoidal releases skip it.
        let (pos_emb, has_learned_pos_emb) =
            match file.tensor_info("audio_tower.embed_positions.weight") {
                Some(_) => (
                    tensor(
                        file,
                        "audio_tower.embed_positions.weight",
                        &[cfg.audio.n_ctx, d],
                    )?,
                    true,
                ),
                None => (vec![0.0f32; cfg.audio.n_ctx * d], false),
            };

        // The 32-layer (n_layer, generally) pre-norm transformer stack. The
        // per-layer tensor names are the upstream HF names verbatim (identity
        // converter map) — the same sub-names Whisper uses under the
        // `audio_tower.` prefix, so the Whisper weight loaders bind them
        // directly (checked shapes, k_proj bias-less — see
        // `whisper::weights::attention`).
        let mut layers = Vec::with_capacity(cfg.audio.n_layer);
        for i in 0..cfg.audio.n_layer {
            let p = format!("audio_tower.layers.{i}");
            layers.push(EncoderLayer {
                attn_ln: ww::layer_norm(file, &format!("{p}.self_attn_layer_norm"), d)?,
                attn: ww::attention(file, &format!("{p}.self_attn"), d)?,
                mlp_ln: ww::layer_norm(file, &format!("{p}.final_layer_norm"), d)?,
                fc1: ww::linear(file, &format!("{p}.fc1"), d, ff, true)?,
                fc2: ww::linear(file, &format!("{p}.fc2"), ff, d, true)?,
            });
        }
        let ln_post = ww::layer_norm(file, "audio_tower.layer_norm", d)?;

        Ok(Self {
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            pos_emb,
            has_learned_pos_emb,
            layers,
            ln_post,
        })
    }

    /// True iff the checkpoint carried a learned positional embedding
    /// (`audio_tower.embed_positions.weight`).
    #[must_use]
    pub fn has_learned_pos_emb(&self) -> bool {
        self.has_learned_pos_emb
    }

    /// Number of bound transformer blocks (0 on the shape-only sentinel
    /// path).
    #[must_use]
    pub fn n_layer(&self) -> usize {
        self.layers.len()
    }
}

/// The Voxtral runtime front-end spec (Whisper-derived, parameterised by
/// `n_mels`). Same values `vokra-convert::models::voxtral::write_frontend_spec`
/// wrote — a mismatched GGUF is rejected at load (FR-LD-03).
#[must_use]
pub fn runtime_frontend_spec(n_mels: usize) -> FrontendSpec {
    FrontendSpec {
        n_fft: 400,
        hop: 160,
        win_length: 400,
        window_type: "hann".to_owned(),
        mel_norm: "slaney".to_owned(),
        htk_mode: false,
        fmin: 0.0,
        fmax: 8000.0,
        n_mels: n_mels as u32,
        pad_mode: "reflect".to_owned(),
        dc_offset_removal: false,
        pre_emphasis: 0.0,
        sample_rate: 16_000,
    }
}

/// Validates the GGUF's `vokra.frontend.*` chunk against the runtime spec
/// at [`FrontendPolicy`].
pub fn check_frontend_spec(file: &GgufFile, n_mels: usize, policy: FrontendPolicy) -> Result<()> {
    let model_spec = FrontendSpec::from_gguf(file)?;
    model_spec.check_against(&runtime_frontend_spec(n_mels), policy)
}

/// Encodes a `[n_mels, n_frames]` log-mel feature tensor into `[n_ctx,
/// hidden_dim]` audio hidden states — the full Voxtral tower (conv stem +
/// positional embedding + `n_layer` pre-norm blocks + final LayerNorm).
///
/// # Input-length contract
///
/// `n_frames` must satisfy `conv_out_len(n_frames) == cfg.audio.n_ctx`
/// (i.e. `n_frames == 2 * n_ctx` for the shipping stride-2 stem — 3000 mel
/// frames → 1500 positions on the real checkpoint). Upstream enforces the
/// same bound (`expected_seq_length` raise, `modeling_voxtral.py` lines
/// 343–347) because the full positional table is added unsliced; a mismatch
/// here is an explicit [`VokraError::InvalidArgument`], never a silent
/// zero-position fabrication.
///
/// # Errors
///
/// - [`VokraError::ModelLoad`] when the config is the `0`-sentinel shape-only
///   path (audio encoder missing / head split absent) — never a silent
///   substitution;
/// - [`VokraError::InvalidArgument`] on log-mel shape mismatch or a downstream
///   kernel shape rejection.
pub fn forward(
    compute: &crate::compute::Compute,
    cfg: &VoxtralConfig,
    weights: &AudioEncoder,
    log_mel: &[f32],
    n_frames: usize,
) -> Result<AudioEncoderOutput> {
    let mut hidden = stem_forward(compute, cfg, weights, log_mel, n_frames)?;
    let t = cfg.audio.n_ctx;
    let d = cfg.audio.hidden_dim;
    let ff = cfg.audio.ffn_dim;
    let n_head = cfg.audio.n_head;

    // Fused device-resident stack (GPU backends): same gate + same seam the
    // Whisper encoder uses (`Compute::encode_prenorm_encoder`), bit-identical
    // to the per-op loop below by that seam's contract. The CPU always takes
    // the per-op loop (no silent fall back, FR-EX-08).
    if compute.prenorm_stack_is_fused() {
        let layers: Vec<PrenormLayer<'_>> = weights.layers.iter().map(prenorm_view).collect();
        let mut normed = vec![0.0f32; t * d];
        compute.encode_prenorm_encoder(
            t,
            d,
            ff,
            n_head,
            LAYER_NORM_DEFAULT_EPS,
            &hidden,
            &layers,
            &weights.ln_post.gamma,
            &weights.ln_post.beta,
            &mut normed,
        )?;
        return Ok(AudioEncoderOutput {
            hidden: normed,
            n_ctx: t,
            hidden_dim: d,
        });
    }

    // Per-op pre-norm block loop — the Whisper encoder block driven verbatim
    // (see module docs: `VoxtralEncoderLayer.forward` == the Whisper block).
    let mut scratch = EncoderScratch::with_reserve(t, d, ff, n_head);
    for layer in &weights.layers {
        encoder_block(
            compute,
            &mut scratch.block,
            &mut hidden,
            t,
            d,
            ff,
            n_head,
            layer,
        )?;
    }

    // Final LayerNorm (`audio_tower.layer_norm`, modeling_voxtral.py L305).
    let mut normed = Vec::new();
    layer_norm_into(compute, &mut normed, &hidden, t, &weights.ln_post)?;

    Ok(AudioEncoderOutput {
        hidden: normed,
        n_ctx: t,
        hidden_dim: d,
    })
}

/// Per-tap capture from [`forward_with_layer_taps`].
#[doc(hidden)]
#[derive(Debug)]
pub struct EncoderTaps {
    /// Hidden state after conv stem + positional add, before block 0
    /// (`[n_ctx, hidden_dim]`).
    pub pre_blocks: Vec<f32>,
    /// `(layer_index, hidden_after_that_block)` for every requested index,
    /// in ascending layer order (duplicates in the request capture once).
    pub after_layer: Vec<(usize, Vec<f32>)>,
}

/// [`forward`] with intermediate taps for the real-checkpoint parity harness
/// (`tests/voxtral_tower_parity.rs`). NOT a stable public surface
/// (`#[doc(hidden)]`).
///
/// Always drives the per-op CPU block loop (the fused GPU stack exposes no
/// intermediate hiddens; its seam contract is bit-identity with this loop,
/// so a CPU tap run is representative). `tap_after_layers` are 0-based block
/// indices whose *output* hidden is captured; out-of-range indices are an
/// explicit error, never silently skipped.
#[doc(hidden)]
pub fn forward_with_layer_taps(
    compute: &crate::compute::Compute,
    cfg: &VoxtralConfig,
    weights: &AudioEncoder,
    log_mel: &[f32],
    n_frames: usize,
    tap_after_layers: &[usize],
) -> Result<(AudioEncoderOutput, EncoderTaps)> {
    for &i in tap_after_layers {
        if i >= weights.layers.len() {
            return Err(VokraError::InvalidArgument(format!(
                "voxtral audio encoder taps: layer index {i} out of range (n_layer = {})",
                weights.layers.len()
            )));
        }
    }
    let mut hidden = stem_forward(compute, cfg, weights, log_mel, n_frames)?;
    let t = cfg.audio.n_ctx;
    let d = cfg.audio.hidden_dim;
    let ff = cfg.audio.ffn_dim;
    let n_head = cfg.audio.n_head;

    let mut taps = EncoderTaps {
        pre_blocks: hidden.clone(),
        after_layer: Vec::with_capacity(tap_after_layers.len()),
    };
    let mut scratch = EncoderScratch::with_reserve(t, d, ff, n_head);
    for (i, layer) in weights.layers.iter().enumerate() {
        encoder_block(
            compute,
            &mut scratch.block,
            &mut hidden,
            t,
            d,
            ff,
            n_head,
            layer,
        )?;
        if tap_after_layers.contains(&i) {
            taps.after_layer.push((i, hidden.clone()));
        }
    }
    let mut normed = Vec::new();
    layer_norm_into(compute, &mut normed, &hidden, t, &weights.ln_post)?;
    Ok((
        AudioEncoderOutput {
            hidden: normed,
            n_ctx: t,
            hidden_dim: d,
        },
        taps,
    ))
}

/// Conv stem + transpose + positional add → `[n_ctx, hidden_dim]` hidden
/// state ready for block 0. Shared by [`forward`] and
/// [`forward_with_layer_taps`] so the two entry points cannot drift.
fn stem_forward(
    compute: &crate::compute::Compute,
    cfg: &VoxtralConfig,
    weights: &AudioEncoder,
    log_mel: &[f32],
    n_frames: usize,
) -> Result<Vec<f32>> {
    if cfg.audio.hidden_dim == 0 || cfg.audio.n_layer == 0 {
        return Err(VokraError::ModelLoad(
            "voxtral audio encoder: config carries 0 layers / hidden_dim — the shape-only \
             converter path was used. Re-convert with a full VoxtralConfig (FR-EX-08)."
                .to_owned(),
        ));
    }
    let d = cfg.audio.hidden_dim;
    let n_mels = cfg.audio.n_mels;
    let n_head = cfg.audio.n_head;
    if n_head == 0 || d % n_head != 0 {
        return Err(VokraError::ModelLoad(format!(
            "voxtral audio encoder: n_head {n_head} must be non-zero and divide hidden_dim {d} \
             (shape-only converter sentinel?) — re-convert with a full VoxtralConfig (FR-EX-08)."
        )));
    }
    if weights.layers.len() != cfg.audio.n_layer {
        return Err(VokraError::ModelLoad(format!(
            "voxtral audio encoder: {} bound transformer blocks != config n_layer {} — the \
             weights and config disagree (FR-EX-08).",
            weights.layers.len(),
            cfg.audio.n_layer
        )));
    }
    if log_mel.len() != n_mels * n_frames {
        return Err(VokraError::InvalidArgument(format!(
            "voxtral audio encoder: log-mel len {} != n_mels*n_frames {}",
            log_mel.len(),
            n_mels * n_frames
        )));
    }

    // conv1: [n_mels, n_frames] -> [d, n_frames], stride 1 pad 1.
    let len1 = conv_out_len(n_frames, 3, 1, 1);
    let mut c1 = vec![0.0f32; d * len1];
    compute.conv1d_f32(
        log_mel,
        n_mels,
        n_frames,
        &weights.conv1_w,
        d,
        3,
        Some(&weights.conv1_b),
        1,
        1,
        &mut c1,
    )?;
    gelu_inplace(compute, &mut c1)?;

    // conv2: [d, len1] -> [d, len2], stride 2 pad 1.
    let len2 = conv_out_len(len1, 3, 2, 1);
    let mut c2 = vec![0.0f32; d * len2];
    compute.conv1d_f32(
        &c1,
        d,
        len1,
        &weights.conv2_w,
        d,
        3,
        Some(&weights.conv2_b),
        2,
        1,
        &mut c2,
    )?;
    gelu_inplace(compute, &mut c2)?;

    // Strict upstream length contract: the FULL positional table is added
    // (modeling_voxtral.py L354–355), so the post-conv length must equal
    // n_ctx exactly (upstream `expected_seq_length` raise, L343–347).
    let t = len2;
    if t != cfg.audio.n_ctx {
        return Err(VokraError::InvalidArgument(format!(
            "voxtral audio encoder: post-conv length {t} != n_ctx {} — upstream requires the mel \
             input to be exactly {} frames (max_source_positions * conv strides; \
             modeling_voxtral.py expected_seq_length check). Pad / window the log-mel to the \
             full context before encoding (FR-EX-08 — no silent positional zero-fill).",
            cfg.audio.n_ctx,
            2 * cfg.audio.n_ctx,
        )));
    }

    // Transpose [d, t] -> [t, d] and add the positional embedding.
    let mut hidden = vec![0.0f32; t * d];
    for c in 0..d {
        for i in 0..t {
            hidden[i * d + c] = c2[c * t + i] + weights.pos_emb[i * d + c];
        }
    }
    Ok(hidden)
}

fn gelu_inplace(compute: &crate::compute::Compute, x: &mut [f32]) -> Result<()> {
    let mut out = vec![0.0f32; x.len()];
    compute.gelu_f32(x, &mut out)?;
    x.copy_from_slice(&out);
    Ok(())
}

fn conv_out_len(in_len: usize, kernel: usize, stride: usize, pad: usize) -> usize {
    (in_len + 2 * pad - kernel) / stride + 1
}

fn bad(msg: String) -> VokraError {
    VokraError::ModelLoad(format!("voxtral audio_encoder: {msg}"))
}

fn tensor(file: &GgufFile, name: &str, want: &[usize]) -> Result<Vec<f32>> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| bad(format!("`{name}` missing from GGUF")))?;
    let got: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    if got != want {
        return Err(bad(format!("`{name}` shape {got:?} != expected {want:?}")));
    }
    file.tensor_f32(name)
        .map_err(|e| bad(format!("`{name}`: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::Compute;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// Deterministic pseudo-random fill: distinct small values so GEMMs mix
    /// rows (all-equal weights would collapse the oracle).
    fn fill(seed: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 31 + seed * 17) % 13) as f32 - 6.0) * 0.02)
            .collect()
    }

    /// A minimal encoder GGUF at (d=4, n_mels=2, n_ctx=8, layers=2). `zero`
    /// selects all-zero payloads (the load/forward zero oracle) vs the
    /// deterministic `fill` pattern (the Whisper-equivalence oracle).
    fn tiny_encoder_gguf(cfg: &VoxtralConfig, zero: bool) -> GgufFile {
        let mut b = GgufBuilder::new();
        let d = cfg.audio.hidden_dim;
        let n_mels = cfg.audio.n_mels;
        let ff = cfg.audio.ffn_dim;
        let val = |seed: usize, n: usize| -> Vec<u8> {
            if zero {
                vec![0u8; n * 4]
            } else {
                f32_bytes(&fill(seed, n))
            }
        };
        b.add_tensor(
            "audio_tower.conv1.weight",
            GgmlType::F32,
            vec![d as u64, n_mels as u64, 3],
            val(1, d * n_mels * 3),
        )
        .unwrap();
        b.add_tensor(
            "audio_tower.conv1.bias",
            GgmlType::F32,
            vec![d as u64],
            val(2, d),
        )
        .unwrap();
        b.add_tensor(
            "audio_tower.conv2.weight",
            GgmlType::F32,
            vec![d as u64, d as u64, 3],
            val(3, d * d * 3),
        )
        .unwrap();
        b.add_tensor(
            "audio_tower.conv2.bias",
            GgmlType::F32,
            vec![d as u64],
            val(4, d),
        )
        .unwrap();
        b.add_tensor(
            "audio_tower.embed_positions.weight",
            GgmlType::F32,
            vec![cfg.audio.n_ctx as u64, d as u64],
            val(5, cfg.audio.n_ctx * d),
        )
        .unwrap();
        for i in 0..cfg.audio.n_layer {
            let p = format!("audio_tower.layers.{i}");
            let s = 100 * (i + 1);
            // LayerNorms: gamma near 1 (fill offset) so the block is not a
            // degenerate all-zero pass in the filled case.
            let ln = |seed: usize| -> Vec<u8> {
                if zero {
                    vec![0u8; d * 4]
                } else {
                    f32_bytes(&fill(seed, d).iter().map(|v| 1.0 + v).collect::<Vec<_>>())
                }
            };
            b.add_tensor(
                &format!("{p}.self_attn_layer_norm.weight"),
                GgmlType::F32,
                vec![d as u64],
                ln(s + 1),
            )
            .unwrap();
            b.add_tensor(
                &format!("{p}.self_attn_layer_norm.bias"),
                GgmlType::F32,
                vec![d as u64],
                val(s + 2, d),
            )
            .unwrap();
            for (j, proj) in ["q_proj", "v_proj", "out_proj"].iter().enumerate() {
                b.add_tensor(
                    &format!("{p}.self_attn.{proj}.weight"),
                    GgmlType::F32,
                    vec![d as u64, d as u64],
                    val(s + 10 + j, d * d),
                )
                .unwrap();
                b.add_tensor(
                    &format!("{p}.self_attn.{proj}.bias"),
                    GgmlType::F32,
                    vec![d as u64],
                    val(s + 20 + j, d),
                )
                .unwrap();
            }
            // k_proj: weight only (bias-less upstream).
            b.add_tensor(
                &format!("{p}.self_attn.k_proj.weight"),
                GgmlType::F32,
                vec![d as u64, d as u64],
                val(s + 30, d * d),
            )
            .unwrap();
            b.add_tensor(
                &format!("{p}.final_layer_norm.weight"),
                GgmlType::F32,
                vec![d as u64],
                ln(s + 41),
            )
            .unwrap();
            b.add_tensor(
                &format!("{p}.final_layer_norm.bias"),
                GgmlType::F32,
                vec![d as u64],
                val(s + 42, d),
            )
            .unwrap();
            b.add_tensor(
                &format!("{p}.fc1.weight"),
                GgmlType::F32,
                vec![ff as u64, d as u64],
                val(s + 50, ff * d),
            )
            .unwrap();
            b.add_tensor(
                &format!("{p}.fc1.bias"),
                GgmlType::F32,
                vec![ff as u64],
                val(s + 51, ff),
            )
            .unwrap();
            b.add_tensor(
                &format!("{p}.fc2.weight"),
                GgmlType::F32,
                vec![d as u64, ff as u64],
                val(s + 52, d * ff),
            )
            .unwrap();
            b.add_tensor(
                &format!("{p}.fc2.bias"),
                GgmlType::F32,
                vec![d as u64],
                val(s + 53, d),
            )
            .unwrap();
        }
        let ln_post_g = if zero {
            vec![0u8; d * 4]
        } else {
            f32_bytes(&fill(7, d).iter().map(|v| 1.0 + v).collect::<Vec<_>>())
        };
        b.add_tensor(
            "audio_tower.layer_norm.weight",
            GgmlType::F32,
            vec![d as u64],
            ln_post_g,
        )
        .unwrap();
        b.add_tensor(
            "audio_tower.layer_norm.bias",
            GgmlType::F32,
            vec![d as u64],
            val(8, d),
        )
        .unwrap();
        GgufFile::parse(b.to_bytes().unwrap()).unwrap()
    }

    fn tiny_config() -> VoxtralConfig {
        VoxtralConfig {
            audio: super::super::config::AudioEncoderConfig {
                n_layer: 2,
                n_head: 2,
                hidden_dim: 4,
                n_ctx: 8,
                n_mels: 2,
                ffn_dim: 8,
            },
            text: super::super::config::TextDecoderConfig {
                n_layer: 1,
                n_head_q: 2,
                n_head_kv: 1,
                head_dim: 0,
                hidden_dim: 4,
                ffn_dim: 8,
                vocab_size: 8,
                n_ctx: 8,
                rope_base: 10_000.0,
                rms_norm_eps: 1e-5,
            },
            cross_attn_hidden_dim: 4,
            mode: "asr".to_owned(),
            s2s_codec_type: "none".to_owned(),
        }
    }

    /// n_frames that satisfies the strict post-conv == n_ctx contract for
    /// the stride-2 stem (`2 * n_ctx`).
    fn full_frames(cfg: &VoxtralConfig) -> usize {
        2 * cfg.audio.n_ctx
    }

    #[test]
    fn load_binds_conv_blocks_and_final_ln_from_synthetic_gguf() {
        let cfg = tiny_config();
        let file = tiny_encoder_gguf(&cfg, true);
        let ae = AudioEncoder::load(&file, &cfg).unwrap();
        assert_eq!(
            ae.conv1_w.len(),
            cfg.audio.hidden_dim * cfg.audio.n_mels * 3
        );
        assert_eq!(ae.conv1_b.len(), cfg.audio.hidden_dim);
        assert_eq!(
            ae.conv2_w.len(),
            cfg.audio.hidden_dim * cfg.audio.hidden_dim * 3
        );
        assert!(ae.has_learned_pos_emb);
        assert_eq!(ae.n_layer(), cfg.audio.n_layer);
        // k_proj bound bias-less; q/v/out carry biases (upstream contract).
        assert!(ae.layers[0].attn.k.bias.is_none());
        assert!(ae.layers[0].attn.q.bias.is_some());
        assert_eq!(ae.ln_post.gamma.len(), cfg.audio.hidden_dim);
    }

    #[test]
    fn load_missing_conv_tensor_is_model_load_error() {
        let cfg = tiny_config();
        // Empty GGUF → tensor missing.
        let file = GgufFile::parse(GgufBuilder::new().to_bytes().unwrap()).unwrap();
        assert!(matches!(
            AudioEncoder::load(&file, &cfg),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn load_missing_layer_tensor_names_the_offender() {
        // Build a GGUF with the conv stem + pos emb but NO layer tensors:
        // the load must fail naming the first missing layer tensor, not
        // silently bind a truncated stack (FR-EX-08).
        let cfg = tiny_config();
        let d = cfg.audio.hidden_dim;
        let n_mels = cfg.audio.n_mels;
        let mut b = GgufBuilder::new();
        b.add_tensor(
            "audio_tower.conv1.weight",
            GgmlType::F32,
            vec![d as u64, n_mels as u64, 3],
            vec![0u8; d * n_mels * 3 * 4],
        )
        .unwrap();
        b.add_tensor(
            "audio_tower.conv1.bias",
            GgmlType::F32,
            vec![d as u64],
            vec![0u8; d * 4],
        )
        .unwrap();
        b.add_tensor(
            "audio_tower.conv2.weight",
            GgmlType::F32,
            vec![d as u64, d as u64, 3],
            vec![0u8; d * d * 3 * 4],
        )
        .unwrap();
        b.add_tensor(
            "audio_tower.conv2.bias",
            GgmlType::F32,
            vec![d as u64],
            vec![0u8; d * 4],
        )
        .unwrap();
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        // `AudioEncoder` carries no `Debug` (weights), so destructure the
        // error instead of `unwrap_err`.
        let Err(err) = AudioEncoder::load(&file, &cfg) else {
            panic!("load must fail on the missing layer tensors");
        };
        assert!(
            matches!(err, VokraError::ModelLoad(ref m) if m.contains("audio_tower.layers.0")),
            "{err:?}"
        );
    }

    #[test]
    fn forward_all_zero_weights_produces_zero_hidden() {
        // With all-zero weights, the stem yields 0 (bias 0, pos 0,
        // GELU(0)=0); each pre-norm block adds attn/mlp of LN(0)·0-weights
        // = 0; the final LN of a zero row with gamma=0, beta=0 is 0. This
        // proves the FULL stack wiring is live (2 blocks + final LN) without
        // pinning numbers only a real checkpoint could supply.
        let cfg = tiny_config();
        let file = tiny_encoder_gguf(&cfg, true);
        let ae = AudioEncoder::load(&file, &cfg).unwrap();
        let n_frames = full_frames(&cfg);
        let log_mel = vec![1.0f32; cfg.audio.n_mels * n_frames];
        let out = forward(&Compute::cpu(), &cfg, &ae, &log_mel, n_frames).unwrap();
        assert_eq!(out.hidden_dim, cfg.audio.hidden_dim);
        assert_eq!(out.n_ctx, cfg.audio.n_ctx);
        assert!(out.hidden.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn forward_matches_whisper_encoder_bit_for_bit_on_shared_weights() {
        // The Voxtral tower IS the Whisper encoder (upstream
        // `VoxtralEncoder` == `WhisperEncoder` — see module docs). Mirror
        // the loaded Voxtral weights into a `whisper::weights::EncoderWeights`
        // and require the two `encode` paths to agree BIT-FOR-BIT on the
        // same log-mel. Whisper's encoder has real-checkpoint parity (M2-06
        // JFK byte-identical transcription), so this transitively anchors
        // the Voxtral block order/math to a validated implementation.
        use crate::whisper::config::WhisperConfig;
        use crate::whisper::weights::{Attention, EncoderWeights, Linear};

        let cfg = tiny_config();
        let file = tiny_encoder_gguf(&cfg, false);
        let ae = AudioEncoder::load(&file, &cfg).unwrap();

        let clone_linear = |l: &Linear| Linear {
            w_t: l.w_t.clone(),
            in_features: l.in_features,
            out_features: l.out_features,
            bias: l.bias.clone(),
        };
        let wcfg = WhisperConfig {
            n_mels: cfg.audio.n_mels,
            d_model: cfg.audio.hidden_dim,
            n_audio_ctx: cfg.audio.n_ctx,
            n_audio_head: cfg.audio.n_head,
            n_audio_layer: cfg.audio.n_layer,
            n_text_ctx: 8,
            n_text_head: 1,
            n_text_layer: 1,
            n_vocab: 8,
            ffn_dim: cfg.audio.ffn_dim,
            eot: 0,
            decoder_start_ids: Vec::new(),
            alignment_heads: Vec::new(),
        };
        let ww = EncoderWeights {
            conv1_w: ae.conv1_w.clone(),
            conv1_b: ae.conv1_b.clone(),
            conv2_w: ae.conv2_w.clone(),
            conv2_b: ae.conv2_b.clone(),
            pos_emb: ae.pos_emb.clone(),
            layers: ae
                .layers
                .iter()
                .map(|l| crate::whisper::weights::EncoderLayer {
                    attn_ln: crate::whisper::weights::LayerNorm {
                        gamma: l.attn_ln.gamma.clone(),
                        beta: l.attn_ln.beta.clone(),
                    },
                    attn: Attention {
                        q: clone_linear(&l.attn.q),
                        k: clone_linear(&l.attn.k),
                        v: clone_linear(&l.attn.v),
                        out: clone_linear(&l.attn.out),
                    },
                    mlp_ln: crate::whisper::weights::LayerNorm {
                        gamma: l.mlp_ln.gamma.clone(),
                        beta: l.mlp_ln.beta.clone(),
                    },
                    fc1: clone_linear(&l.fc1),
                    fc2: clone_linear(&l.fc2),
                })
                .collect(),
            ln_post: crate::whisper::weights::LayerNorm {
                gamma: ae.ln_post.gamma.clone(),
                beta: ae.ln_post.beta.clone(),
            },
        };

        let n_frames = full_frames(&cfg);
        let log_mel: Vec<f32> = fill(9, cfg.audio.n_mels * n_frames);
        let compute = Compute::cpu();
        let vox = forward(&compute, &cfg, &ae, &log_mel, n_frames).unwrap();
        let whi =
            crate::whisper::encoder::encode(&compute, &wcfg, &ww, &log_mel, n_frames).unwrap();
        assert_eq!(vox.n_ctx, whi.n_ctx);
        assert_eq!(
            vox.hidden, whi.hidden,
            "voxtral tower must be bit-identical to the whisper encoder on shared weights"
        );
        // Sanity: non-degenerate oracle (all-zero output would prove nothing).
        assert!(vox.hidden.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn forward_with_layer_taps_matches_forward_and_captures_layers() {
        let cfg = tiny_config();
        let file = tiny_encoder_gguf(&cfg, false);
        let ae = AudioEncoder::load(&file, &cfg).unwrap();
        let n_frames = full_frames(&cfg);
        let log_mel: Vec<f32> = fill(9, cfg.audio.n_mels * n_frames);
        let compute = Compute::cpu();
        let plain = forward(&compute, &cfg, &ae, &log_mel, n_frames).unwrap();
        let (tapped, taps) =
            forward_with_layer_taps(&compute, &cfg, &ae, &log_mel, n_frames, &[0, 1]).unwrap();
        assert_eq!(
            plain.hidden, tapped.hidden,
            "taps must not perturb the forward"
        );
        assert_eq!(taps.after_layer.len(), 2);
        assert_eq!(taps.after_layer[0].0, 0);
        assert_eq!(taps.after_layer[1].0, 1);
        assert_eq!(
            taps.pre_blocks.len(),
            cfg.audio.n_ctx * cfg.audio.hidden_dim
        );
        // Blocks with non-zero weights must actually transform the hidden.
        assert_ne!(taps.pre_blocks, taps.after_layer[0].1);
        // Out-of-range tap index is an explicit error.
        assert!(matches!(
            forward_with_layer_taps(&compute, &cfg, &ae, &log_mel, n_frames, &[99]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn forward_rejects_zero_config() {
        // Shape-only converter path leaves n_layer=0 → forward must reject
        // rather than silently substitute (FR-EX-08).
        let mut cfg = tiny_config();
        cfg.audio.n_layer = 0;
        let file = GgufFile::parse(GgufBuilder::new().to_bytes().unwrap()).unwrap();
        let ae = AudioEncoder::load(&file, &cfg).unwrap(); // sentinel skip-load
        let n_frames = 16;
        let log_mel = vec![1.0f32; cfg.audio.n_mels * n_frames];
        assert!(matches!(
            forward(&Compute::cpu(), &cfg, &ae, &log_mel, n_frames),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn forward_rejects_zero_head_split() {
        let mut cfg = tiny_config();
        let file = tiny_encoder_gguf(&cfg, true);
        let ae = AudioEncoder::load(&file, &cfg).unwrap();
        cfg.audio.n_head = 0;
        let n_frames = full_frames(&cfg);
        let log_mel = vec![0.0f32; cfg.audio.n_mels * n_frames];
        assert!(matches!(
            forward(&Compute::cpu(), &cfg, &ae, &log_mel, n_frames),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn forward_rejects_log_mel_shape_mismatch() {
        let cfg = tiny_config();
        let file = tiny_encoder_gguf(&cfg, true);
        let ae = AudioEncoder::load(&file, &cfg).unwrap();
        let log_mel = vec![1.0f32; 3]; // wrong length.
        assert!(matches!(
            forward(&Compute::cpu(), &cfg, &ae, &log_mel, 16),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn forward_rejects_short_window_instead_of_zero_padding_positions() {
        // Upstream requires exactly `2 * n_ctx` mel frames (the full
        // positional table is added unsliced — expected_seq_length check).
        // A shorter window must be an explicit error, not a silent
        // zero-position fabrication (the pre-M4 stub behaviour).
        let cfg = tiny_config();
        let file = tiny_encoder_gguf(&cfg, true);
        let ae = AudioEncoder::load(&file, &cfg).unwrap();
        let n_frames = 8; // conv → 4 positions != n_ctx 8.
        let log_mel = vec![0.5f32; cfg.audio.n_mels * n_frames];
        let err = forward(&Compute::cpu(), &cfg, &ae, &log_mel, n_frames).unwrap_err();
        assert!(
            matches!(err, VokraError::InvalidArgument(ref m) if m.contains("n_ctx")),
            "{err:?}"
        );
    }

    #[test]
    fn runtime_frontend_spec_matches_converter() {
        let s = runtime_frontend_spec(128);
        assert_eq!(s.n_fft, 400);
        assert_eq!(s.hop, 160);
        assert_eq!(s.mel_norm, "slaney");
        assert_eq!(s.n_mels, 128);
        assert_eq!(s.sample_rate, 16_000);
    }
}
