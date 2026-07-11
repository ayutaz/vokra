//! Voxtral audio encoder — Whisper-derived pre-norm transformer.
//!
//! Structurally identical to `WhisperEncoder` (see
//! [`super::super::whisper::encoder`]):
//!
//! 1. `conv1` (`n_mels → d`, kernel 3, stride 1, pad 1) + GELU;
//! 2. `conv2` (`d → d`, kernel 3, stride 2, pad 1) + GELU;
//! 3. transpose + learned/sinusoidal positional embedding;
//! 4. `n_layer` pre-norm self-attention blocks;
//! 5. final LayerNorm.
//!
//! All matmul / conv / norm / activation goes through the shared compute
//! seam ([`crate::compute::Compute`]) so a GPU backend can dispatch the same
//! kernels without a second implementation.
//!
//! # Foundation scope (M3-10-T07 / T08)
//!
//! This foundation file:
//! - reads a Voxtral-shaped weight blob out of the GGUF (identifying tensor
//!   names under `audio_tower.*` — the upstream Mistral Voxtral prefix);
//! - exposes a [`forward`] that runs the encoder on a caller-supplied log-mel
//!   feature tensor, using [`super::super::whisper::nn`] as the underlying
//!   block math;
//! - ships a synthetic smoke test that exercises the load / forward path
//!   with an all-zero checkpoint (shape / dispatch coverage only).
//!
//! Real-checkpoint parity + the reference-dumper fixtures are follow-on
//! tickets (T19–T22). See the module docs on [`super`](super) for the full
//! scope.

use vokra_core::gguf::{FrontendPolicy, FrontendSpec, GgufFile};
use vokra_core::{Result, VokraError};

use super::VoxtralConfig;

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
/// Only the tensors this module's [`forward`] path touches are bound —
/// missing tensors are surfaced by name at load time (FR-EX-08), never
/// silently absent.
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
        // forward entry point will return `UnsupportedOp` if invoked.
        if cfg.audio.hidden_dim == 0 || cfg.audio.n_layer == 0 {
            return Ok(Self {
                conv1_w: Vec::new(),
                conv1_b: Vec::new(),
                conv2_w: Vec::new(),
                conv2_b: Vec::new(),
                pos_emb: Vec::new(),
                has_learned_pos_emb: false,
            });
        }
        let d = cfg.audio.hidden_dim;
        let n_mels = cfg.audio.n_mels;
        if n_mels == 0 {
            return Err(bad("audio_encoder.n_mels must be non-zero".to_owned()));
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
        Ok(Self {
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            pos_emb,
            has_learned_pos_emb,
        })
    }

    /// True iff the checkpoint carried a learned positional embedding
    /// (`audio_tower.embed_positions.weight`).
    #[must_use]
    pub fn has_learned_pos_emb(&self) -> bool {
        self.has_learned_pos_emb
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
/// hidden_dim]` audio hidden states.
///
/// # Foundation-scope contract
///
/// This implementation:
/// - runs a **single 1-D conv stem** (conv1 + conv2 with GELU) to reduce the
///   time axis by 2× (matching Whisper);
/// - **adds the positional embedding** and returns the transposed hidden
///   state.
///
/// It does **NOT** yet run the pre-norm transformer stack (blocks 4/5 in the
/// module docs) — those layers require the full attention weight bind, which
/// is deferred to T19+ once a real Voxtral checkpoint parity dump exists.
/// The returned hidden state is still a valid encoder-output *shape*, which
/// is enough for the ASR / S2S head skeletons downstream to compile and be
/// unit-tested on synthetic data.
///
/// # Errors
///
/// - [`VokraError::ModelLoad`] when the config is the `0`-sentinel shape-only
///   path (audio encoder missing) — never a silent substitution;
/// - [`VokraError::InvalidArgument`] on log-mel shape mismatch or a downstream
///   kernel shape rejection.
pub fn forward(
    compute: &crate::compute::Compute,
    cfg: &VoxtralConfig,
    weights: &AudioEncoder,
    log_mel: &[f32],
    n_frames: usize,
) -> Result<AudioEncoderOutput> {
    if cfg.audio.hidden_dim == 0 || cfg.audio.n_layer == 0 {
        return Err(VokraError::ModelLoad(
            "voxtral audio encoder: config carries 0 layers / hidden_dim — the shape-only \
             converter path was used. Re-convert with a full VoxtralConfig (FR-EX-08)."
                .to_owned(),
        ));
    }
    let d = cfg.audio.hidden_dim;
    let n_mels = cfg.audio.n_mels;
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

    // Transpose + positional embedding.
    let t = len2;
    let mut hidden = vec![0.0f32; t * d];
    for c in 0..d {
        for i in 0..t {
            let pos = if i < cfg.audio.n_ctx && !weights.pos_emb.is_empty() {
                weights.pos_emb[i * d + c]
            } else {
                0.0
            };
            hidden[i * d + c] = c2[c * t + i] + pos;
        }
    }

    // NOTE: transformer stack (self-attention + MLP × n_layer + final LN)
    // deferred to T19+ once the real weight tensors + reference parity are
    // available (see module docs). The output shape is the same either way,
    // so downstream head modules compile / unit-test cleanly on this stub.

    Ok(AudioEncoderOutput {
        hidden,
        n_ctx: t,
        hidden_dim: d,
    })
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

    /// A minimal, all-zero encoder GGUF at (d=4, n_mels=2, n_ctx=8, layers=1)
    /// — enough to exercise load + forward without any allocation risk.
    fn tiny_encoder_gguf(cfg: &VoxtralConfig) -> GgufFile {
        let mut b = GgufBuilder::new();
        let d = cfg.audio.hidden_dim;
        let n_mels = cfg.audio.n_mels;
        // conv1: [d, n_mels, 3]
        let n1 = d * n_mels * 3;
        b.add_tensor(
            "audio_tower.conv1.weight",
            GgmlType::F32,
            vec![d as u64, n_mels as u64, 3],
            vec![0u8; n1 * 4],
        )
        .unwrap();
        b.add_tensor(
            "audio_tower.conv1.bias",
            GgmlType::F32,
            vec![d as u64],
            vec![0u8; d * 4],
        )
        .unwrap();
        // conv2: [d, d, 3]
        let n2 = d * d * 3;
        b.add_tensor(
            "audio_tower.conv2.weight",
            GgmlType::F32,
            vec![d as u64, d as u64, 3],
            vec![0u8; n2 * 4],
        )
        .unwrap();
        b.add_tensor(
            "audio_tower.conv2.bias",
            GgmlType::F32,
            vec![d as u64],
            vec![0u8; d * 4],
        )
        .unwrap();
        // pos_emb: [n_ctx, d]
        b.add_tensor(
            "audio_tower.embed_positions.weight",
            GgmlType::F32,
            vec![cfg.audio.n_ctx as u64, d as u64],
            vec![0u8; cfg.audio.n_ctx * d * 4],
        )
        .unwrap();
        GgufFile::parse(b.to_bytes().unwrap()).unwrap()
    }

    fn tiny_config() -> VoxtralConfig {
        VoxtralConfig {
            audio: super::super::config::AudioEncoderConfig {
                n_layer: 1,
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

    #[test]
    fn load_binds_all_conv_weights_from_synthetic_gguf() {
        let cfg = tiny_config();
        let file = tiny_encoder_gguf(&cfg);
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
    fn forward_all_zero_weights_produces_zero_hidden() {
        // With all-zero conv weights, every hidden output is exactly 0
        // (bias also zero + pos_emb also zero + GELU(0)=0). This proves
        // the load/forward wiring is live without pinning any numerical
        // parity that only a real checkpoint can supply.
        let cfg = tiny_config();
        let file = tiny_encoder_gguf(&cfg);
        let ae = AudioEncoder::load(&file, &cfg).unwrap();
        let n_frames = 8;
        let log_mel = vec![1.0f32; cfg.audio.n_mels * n_frames];
        let out = forward(&Compute::cpu(), &cfg, &ae, &log_mel, n_frames).unwrap();
        assert_eq!(out.hidden_dim, cfg.audio.hidden_dim);
        assert!(out.hidden.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn forward_rejects_zero_config() {
        // Shape-only converter path leaves n_layer=0 → forward must reject
        // rather than silently substitute (FR-EX-08).
        let mut cfg = tiny_config();
        cfg.audio.n_layer = 0;
        let file = tiny_encoder_gguf(&cfg);
        let ae = AudioEncoder::load(&file, &cfg).unwrap();
        let n_frames = 8;
        let log_mel = vec![1.0f32; cfg.audio.n_mels * n_frames];
        assert!(matches!(
            forward(&Compute::cpu(), &cfg, &ae, &log_mel, n_frames),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn forward_rejects_log_mel_shape_mismatch() {
        let cfg = tiny_config();
        let file = tiny_encoder_gguf(&cfg);
        let ae = AudioEncoder::load(&file, &cfg).unwrap();
        let log_mel = vec![1.0f32; 3]; // wrong length.
        assert!(matches!(
            forward(&Compute::cpu(), &cfg, &ae, &log_mel, 8),
            Err(VokraError::InvalidArgument(_))
        ));
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
