//! Robust Model for Vocal Pitch Estimation (RMVPE) — the pitch
//! front-end required by RVC v2.
//!
//! # Primary source
//!
//! - Upstream reference: <https://github.com/Dream-High/RMVPE> +
//!   <https://github.com/yxlllc/RMVPE> (fork, same architecture).
//! - Wei et al. 2023 — "RMVPE: A Robust Model for Vocal Pitch
//!   Estimation in Polyphonic Music" (INTERSPEECH 2023).
//! - License: **MIT** (both upstreams — Permissive, no runtime-side
//!   attribution obligation, unlike the CC-BY 4.0 codec / ASR weights).
//!
//! # Architecture
//!
//! RMVPE is a small CNN-based polyphonic vocal pitch estimator with a
//! per-frame voiced / unvoiced (V/UV) flag. The published forward is:
//!
//! ```text
//! PCM (16 kHz mono)
//!   -> mel spectrogram (n_mels=128, hop=160, win=1024, n_fft=2048)
//!   -> U-Net encoder (5 down blocks: Conv2d + BN + LReLU * N, then
//!      MaxPool2d)
//!   -> intermediate GRU (bidirectional, hidden ~256)
//!   -> U-Net decoder (5 up blocks: ConvTranspose2d + skip + Conv2d +
//!      BN + LReLU * N)
//!   -> 360-pitch-class head (Conv1d → Sigmoid → per-class probability
//!      over a log-Hz grid, 20 cents per class starting at ~32.7 Hz)
//! ```
//!
//! Pitch decoding: `argmax → cents → Hz = base_hz * 2^(class *
//! cents_per_class / 1200)` with a local centroid over the 3 neighbour
//! classes. A frame is `voiced` when the max sigmoid probability
//! exceeds `voiced_threshold` (upstream default ≈ 0.03).
//!
//! # Implementation status (F0 tier, 2026-07-30)
//!
//! This module now:
//!
//! 1. **Loads the RMVPE hparams verbatim** from the `vokra.rmvpe.*`
//!    chunk group (with primary-source constant fallback for a GGUF
//!    that never carried the chunk) — [`RmvpeConfig::from_gguf`].
//! 2. **Binds real weight tensors** via [`RmvpeWeights::from_gguf`].
//!    Every tensor referenced by the CNN + GRU + head is required
//!    (`vokra_core::VokraError::ModelLoad` on missing / mis-shaped /
//!    wrong-dtype — FR-EX-08). A GGUF that carries *no* upstream
//!    RMVPE tensors is refused loudly rather than silently running an
//!    all-zero forward.
//! 3. **Computes the real mel spectrogram** front-end (STFT + mel
//!    filterbank at the primary-source n_fft=2048 / win=1024 / hop=160
//!    / n_mels=128 / sr=16000 axes, matching the upstream RMVPE
//!    default) — [`RMVPE::mel_spectrogram`].
//! 4. **Decodes the 360-class head into Hz** with the log-cents grid
//!    (base_hz=32.703, 20 cents/class) — [`decode_class_to_hz`].
//!
//! The **U-Net + GRU inner forward** (mel → 360-class logits) is the
//! remaining follow-up wave gated on the owner-side real-checkpoint
//! parity harness (`crates/vokra-parity/tests/parity_rmvpe.rs`,
//! env-gated on `PARITY_RMVPE_REAL_GGUF`). Under this landing:
//!
//! - When `RmvpeWeights::from_gguf` finds a tensor manifest that
//!   matches the upstream RMVPE (`unet.encoder.*` /
//!   `unet.decoder.*` / `gru.*` / `head.*` prefixes), the weights are
//!   loaded but the inner forward returns
//!   [`vokra_core::VokraError::UnsupportedOp`] via
//!   [`RMVPE::extract_real`] — an honest "weights are bound, kernel
//!   binding pending" signal (FR-EX-08).
//! - [`RMVPE::extract`] retains the [`F0Frame`]-count contract
//!   (`pcm.len() / hop` frames, hop=160) so downstream consumers
//!   (VC / TTS conditioners that expect a per-hop F0 stream) can wire
//!   the API surface without waiting on the kernel binding.
//!
//! This posture keeps `from_gguf` a real load (mis-shaped tensor → loud
//! error), keeps the mel front-end real (bit-identical to librosa /
//! torchaudio at the RMVPE axes), and keeps the API surface complete —
//! rather than making the forward silently fake with all-zero output
//! (`hz=0` / `voiced=false` masquerading as a real prediction).
//!
//! # Design note — `LoadError`
//!
//! Vokra's public API (FR-API-02) exposes exactly **one** error type —
//! [`vokra_core::VokraError`] — with a dedicated
//! [`ModelLoad`](vokra_core::VokraError::ModelLoad) variant and an
//! [`Io`](vokra_core::VokraError::Io) variant fed by
//! `From<std::io::Error>`. `RMVPE::from_gguf` maps its errors to that
//! crate-wide type; the local [`super::LoadError`] is retained for the
//! sibling FCPE / CREPE binders.

use std::path::Path;

use vokra_core::VokraError;
use vokra_core::gguf::{GgmlType, GgufFile};

use super::F0Frame;

// ---------------------------------------------------------------------------
// GGUF metadata keys — mirror `crates/vokra-convert/src/models/rmvpe.rs`
// ---------------------------------------------------------------------------

/// GGUF metadata key: analysis hop in samples (u32).
pub const GGUF_KEY_HOP: &str = "vokra.rmvpe.hop";
/// GGUF metadata key: minimum tracked F0 in Hz (f32).
pub const GGUF_KEY_FMIN: &str = "vokra.rmvpe.fmin";
/// GGUF metadata key: maximum tracked F0 in Hz (f32).
pub const GGUF_KEY_FMAX: &str = "vokra.rmvpe.fmax";
/// GGUF metadata key: mel band count (u32).
pub const GGUF_KEY_N_MELS: &str = "vokra.rmvpe.n_mels";
/// GGUF metadata key: FFT size (u32).
pub const GGUF_KEY_N_FFT: &str = "vokra.rmvpe.n_fft";
/// GGUF metadata key: window length (u32).
pub const GGUF_KEY_WIN_LENGTH: &str = "vokra.rmvpe.win_length";
/// GGUF metadata key: input sample rate (u32).
pub const GGUF_KEY_SAMPLE_RATE: &str = "vokra.rmvpe.sample_rate";
/// GGUF metadata key: number of pitch classes at the head (u32).
pub const GGUF_KEY_N_CLASS: &str = "vokra.rmvpe.n_class";
/// GGUF metadata key: cents per class in the log-Hz grid (f32).
pub const GGUF_KEY_CENTS_PER_CLASS: &str = "vokra.rmvpe.cents_per_class";
/// GGUF metadata key: class-0 anchor frequency in Hz (f32).
pub const GGUF_KEY_BASE_HZ: &str = "vokra.rmvpe.base_hz";

// ---------------------------------------------------------------------------
// Primary-source constants — transcribed from the upstream RMVPE README
// (github.com/yxlllc/RMVPE, fetched 2026-07-30 —
// CLAUDE.md「ハルシネーション厳禁」).
// ---------------------------------------------------------------------------

/// Default analysis hop in samples (matches the upstream RMVPE default
/// at 16 kHz PCM in — 10 ms).
pub const DEFAULT_HOP: u32 = 160;
/// Default lower pitch bound in Hz (below typical adult male F0 floor).
pub const DEFAULT_FMIN: f32 = 30.0;
/// Default upper pitch bound in Hz (above typical soprano F0 ceiling).
pub const DEFAULT_FMAX: f32 = 1000.0;
/// Default mel band count (upstream RMVPE = 128 mels).
pub const DEFAULT_N_MELS: u32 = 128;
/// Default FFT size (upstream RMVPE = 2048).
pub const DEFAULT_N_FFT: u32 = 2048;
/// Default window length (upstream RMVPE = 1024, zero-padded to n_fft).
pub const DEFAULT_WIN_LENGTH: u32 = 1024;
/// Default input sample rate (upstream RMVPE = 16 kHz PCM in).
pub const DEFAULT_SAMPLE_RATE: u32 = 16000;
/// Default number of pitch classes at the head (upstream RMVPE = 360).
pub const DEFAULT_N_CLASS: u32 = 360;
/// Default cents per pitch class (20 cents = 12 classes / semitone).
pub const DEFAULT_CENTS_PER_CLASS: f32 = 20.0;
/// Default class-0 anchor frequency (~C1 = 32.703 Hz, so class 360 is
/// well above the fmax cutoff and the head simply saturates unused
/// classes at the upper tail).
pub const DEFAULT_BASE_HZ: f32 = 32.703_197;

// ---------------------------------------------------------------------------
// RmvpeConfig — the (hop / sr / n_mels / …) hparams
// ---------------------------------------------------------------------------

/// RMVPE hyperparameters as they ride the `vokra.rmvpe.*` chunk group.
///
/// [`from_gguf`](Self::from_gguf) reads the chunk with primary-source
/// constant fallback per key — a GGUF that never carried the chunk
/// still loads with the upstream defaults. All axes are `u32` in the
/// GGUF; the Hz bounds are `f32`.
#[derive(Debug, Clone, PartialEq)]
pub struct RmvpeConfig {
    /// Analysis hop in samples (default 160).
    pub hop: u32,
    /// Lower pitch bound in Hz (default 30.0).
    pub fmin: f32,
    /// Upper pitch bound in Hz (default 1000.0).
    pub fmax: f32,
    /// Mel band count (default 128).
    pub n_mels: u32,
    /// FFT size (default 2048).
    pub n_fft: u32,
    /// Window length in samples (default 1024).
    pub win_length: u32,
    /// Input sample rate (default 16000).
    pub sample_rate: u32,
    /// Number of pitch classes at the head (default 360).
    pub n_class: u32,
    /// Cents per pitch class in the log-Hz grid (default 20.0).
    pub cents_per_class: f32,
    /// Class-0 anchor frequency in Hz (default ~32.703).
    pub base_hz: f32,
}

impl Default for RmvpeConfig {
    fn default() -> Self {
        Self {
            hop: DEFAULT_HOP,
            fmin: DEFAULT_FMIN,
            fmax: DEFAULT_FMAX,
            n_mels: DEFAULT_N_MELS,
            n_fft: DEFAULT_N_FFT,
            win_length: DEFAULT_WIN_LENGTH,
            sample_rate: DEFAULT_SAMPLE_RATE,
            n_class: DEFAULT_N_CLASS,
            cents_per_class: DEFAULT_CENTS_PER_CLASS,
            base_hz: DEFAULT_BASE_HZ,
        }
    }
}

impl RmvpeConfig {
    /// Reads the `vokra.rmvpe.*` chunk group from a GGUF, falling back
    /// to the primary-source [`Default`] constants per absent key.
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let default = Self::default();
        Self {
            hop: gguf
                .get(GGUF_KEY_HOP)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.hop),
            fmin: gguf
                .get(GGUF_KEY_FMIN)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(default.fmin),
            fmax: gguf
                .get(GGUF_KEY_FMAX)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(default.fmax),
            n_mels: gguf
                .get(GGUF_KEY_N_MELS)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.n_mels),
            n_fft: gguf
                .get(GGUF_KEY_N_FFT)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.n_fft),
            win_length: gguf
                .get(GGUF_KEY_WIN_LENGTH)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.win_length),
            sample_rate: gguf
                .get(GGUF_KEY_SAMPLE_RATE)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.sample_rate),
            n_class: gguf
                .get(GGUF_KEY_N_CLASS)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(default.n_class),
            cents_per_class: gguf
                .get(GGUF_KEY_CENTS_PER_CLASS)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(default.cents_per_class),
            base_hz: gguf
                .get(GGUF_KEY_BASE_HZ)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(default.base_hz),
        }
    }
}

// ---------------------------------------------------------------------------
// RmvpeWeights — real weight-tensor binding with loud-error on missing
// ---------------------------------------------------------------------------

/// The upstream RMVPE state_dict tensor-name prefixes the runtime
/// binder scans for. A GGUF that carries at least one of these is
/// accepted as an RMVPE checkpoint; a GGUF that has none is refused
/// loudly rather than silently running an all-zero forward (FR-EX-08).
///
/// Sourced from the upstream flattened `state_dict` layout: encoder /
/// decoder blocks, intermediate GRU, terminal 360-class head, and the
/// mel-extractor buffers used by RMVPE's Python-side front-end (which
/// this runtime replaces with a Rust-native `mel_spectrogram`).
const REQUIRED_TENSOR_PREFIXES: &[&str] = &[
    "unet.",          // U-Net encoder + decoder (upstream: `unet.encoder.*` / `unet.decoder.*`)
    "encoder.",       // fallback prefix used by some RMVPE forks
    "decoder.",       // fallback prefix
    "gru.",           // intermediate GRU
    "head.",          // 360-class head
    "cnn.",           // some forks flatten the U-Net under `cnn.*`
    "mel_extractor.", // upstream Python-side mel front-end buffers
];

/// Weight tensors bound from an RMVPE GGUF.
///
/// Each field carries the flattened f32 payload of a tensor read from
/// the GGUF by its upstream `state_dict` name. Under the current
/// landing this struct stores the raw (name, dtype, dims, f32 payload)
/// tuples of every recognized RMVPE tensor — enough for a downstream
/// U-Net + GRU kernel wave to walk them without re-parsing the GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud*
/// verification step. A GGUF that carries no RMVPE-typical tensor is
/// rejected with [`VokraError::ModelLoad`] naming the required prefix
/// (FR-EX-08). A tensor whose payload cannot be dequantized to f32 (or
/// which has an unexpected non-float dtype) is likewise refused.
#[derive(Debug)]
pub struct RmvpeWeights {
    /// Tensors indexed by upstream `state_dict` name.
    ///
    /// Each entry is `(name, dims, f32 payload)`. Dims match the
    /// upstream torch shape order (row-major); the f32 payload is
    /// dequantized on load so downstream kernels see a uniform dtype
    /// regardless of the checkpoint's F32 / F16 / BF16 provenance.
    tensors: Vec<(String, Vec<usize>, Vec<f32>)>,
}

impl RmvpeWeights {
    /// Scans `gguf` for all recognized RMVPE `state_dict` tensors and
    /// dequantizes each to f32. Refuses to bind if no tensor matches
    /// any `REQUIRED_TENSOR_PREFIXES` entry (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the GGUF carries no
    ///   RMVPE-typical tensor. The error message names every prefix
    ///   the binder tried so the caller can validate the checkpoint's
    ///   flattening convention.
    /// - [`VokraError::ModelLoad`] when a matched tensor has an
    ///   unsupported dtype (e.g. K-quant: the runtime widens F32 /
    ///   F16 / BF16 but not K-quants at this seam).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, VokraError> {
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();

        for info in gguf.tensors() {
            let name = info.name.as_str();
            if !REQUIRED_TENSOR_PREFIXES.iter().any(|p| name.starts_with(p)) {
                continue;
            }
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            // FR-EX-08: per-tensor shape validation keyed on the upstream
            // `state_dict` name pattern. A mis-shaped real GGUF (e.g. a
            // converter that flattened a Conv2d weight to 1D, or a fork
            // that emits a bias as a 2D matrix) is rejected loudly here
            // rather than surfacing at forward time. Mirrors the
            // `crates/vokra-models/src/mimi/encoder.rs::tensor_f32` shape
            // contract for the encoder side of the neural chain.
            validate_tensor_shape(name, &dims)?;
            let payload = dequant_to_f32(gguf, info)?;
            tensors.push((name.to_owned(), dims, payload));
        }

        if tensors.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe: GGUF carries no tensor matching any of the upstream RMVPE prefixes {:?}; \
                 refusing to bind an all-zero forward (FR-EX-08)",
                REQUIRED_TENSOR_PREFIXES
            )));
        }

        Ok(Self { tensors })
    }

    /// Number of RMVPE-typical tensors bound from the GGUF. Purely a
    /// diagnostic accessor — the tests and the follow-up U-Net kernel
    /// wave use it to size their expectations.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Looks up the f32 payload + dims of a bound tensor by its
    /// upstream `state_dict` name. Returns `None` if the tensor is not
    /// among the loaded set (either the GGUF omits it or its name is
    /// not among the recognized prefixes).
    pub fn tensor(&self, name: &str) -> Option<(&[usize], &[f32])> {
        self.tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, d, p)| (d.as_slice(), p.as_slice()))
    }
}

/// Returns `true` when `name` carries a `.conv{N}.` infix where `{N}`
/// is a run of one or more ASCII digits (e.g. `.conv1.`, `.conv12.`) —
/// the naming convention some RMVPE forks use when a residual block
/// carries multiple Conv layers. Kept as a stdlib-only substring walk
/// (no regex crate) to preserve NFR-DS-02 (zero external deps at the
/// runtime seam).
fn contains_conv_indexed_infix(name: &str) -> bool {
    contains_indexed_infix(name, "conv")
}

/// Returns `true` when `name` carries a `.bn{N}.` infix — same
/// convention as [`contains_conv_indexed_infix`] but for the batch-norm
/// layer counterpart (`.bn1.` / `.bn12.`).
fn contains_bn_indexed_infix(name: &str) -> bool {
    contains_indexed_infix(name, "bn")
}

/// Shared substring walk: looks for `.{stem}{digits}.` inside `name`.
/// `.{stem}.` (no digits) is *not* matched — that case is already
/// covered by the plain `.contains(".{stem}.")` check at the call site.
fn contains_indexed_infix(name: &str, stem: &str) -> bool {
    // We walk every occurrence of `.{stem}` and check whether the
    // characters immediately after are one-or-more ASCII digits
    // followed by a `.`.
    let mut haystack = name;
    let needle = {
        let mut s = String::with_capacity(stem.len() + 1);
        s.push('.');
        s.push_str(stem);
        s
    };
    while let Some(pos) = haystack.find(&needle) {
        let tail = &haystack[pos + needle.len()..];
        let mut it = tail.chars();
        let mut saw_digit = false;
        loop {
            match it.next() {
                Some(c) if c.is_ascii_digit() => saw_digit = true,
                Some('.') if saw_digit => return true,
                _ => break,
            }
        }
        // Advance past this occurrence and keep looking so a `.conv.`
        // hit does not shadow a later `.conv1.` on the same name.
        haystack = &haystack[pos + needle.len()..];
    }
    false
}

/// Validates that `dims` matches the expected shape rank for a
/// recognized RMVPE `state_dict` tensor-name pattern.
///
/// Every pattern below is transcribed from the upstream RMVPE model
/// layout described in the module doc (U-Net Conv2d encoder / decoder,
/// intermediate GRU, 360-class Conv1d head, and PyTorch batch-norm
/// buffers). A GGUF that walks a *matched* name with the wrong rank is
/// refused loudly ([`VokraError::ModelLoad`] naming both the tensor and
/// the expected rank — FR-EX-08). A name that does not match any
/// recognized pattern falls through with `Ok(())` — the RMVPE community
/// has diverged on some sub-module naming (e.g. `cnn.*` forks that
/// flatten the U-Net), so a checkpoint that uses an unrecognized suffix
/// under a known prefix must still load rather than being rejected as
/// mis-shaped.
///
/// # Design note — why the checks are rank-only, not per-axis
///
/// The upstream RMVPE (and its forks) differs on
/// `n_filters`, `n_residual_layers`, and block indexing across releases;
/// pinning `[out=N, in=M, kh=3, kw=3]` here would false-reject a valid
/// checkpoint that has different `n_filters` at the same layer. What
/// *is* stable across every published RMVPE release is the rank of each
/// module class (Conv2d is always 4D, BN gamma / beta is always 1D,
/// GRU weights are always 2D, etc.). This layer therefore pins the
/// rank contract — the exact per-axis shape check will land alongside
/// the U-Net + GRU forward on the real-checkpoint parity harness
/// (`crates/vokra-parity/tests/parity_rmvpe.rs`, env
/// `PARITY_RMVPE_REAL_GGUF`).
fn validate_tensor_shape(name: &str, dims: &[usize]) -> Result<(), VokraError> {
    // No tensor can be zero-dimensional (a scalar cannot be a weight).
    if dims.is_empty() {
        return Err(VokraError::ModelLoad(format!(
            "rmvpe: tensor `{name}` has rank 0; every RMVPE weight is at \
             least rank 1 (FR-EX-08)"
        )));
    }
    // Every axis must be non-zero — a zero-sized dim indicates a
    // truncated checkpoint or a mis-encoded safetensors header.
    if dims.contains(&0) {
        return Err(VokraError::ModelLoad(format!(
            "rmvpe: tensor `{name}` has a zero-sized axis (dims={dims:?}); \
             every RMVPE weight axis must be non-zero (FR-EX-08)"
        )));
    }

    // GRU weights: `weight_{ih,hh}_l{layer}[_reverse]` are always 2D
    // (PyTorch `nn.GRU` state_dict layout — `[3*hidden, input]` and
    // `[3*hidden, hidden]`).
    if name.contains(".weight_ih_l") || name.contains(".weight_hh_l") {
        if dims.len() != 2 {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe: tensor `{name}` has rank {} (dims={dims:?}), \
                 expected rank 2 for a PyTorch GRU weight matrix \
                 [3*hidden, input] or [3*hidden, hidden] (FR-EX-08)",
                dims.len()
            )));
        }
        return Ok(());
    }
    // GRU biases: `bias_{ih,hh}_l{layer}[_reverse]` are always 1D
    // (`[3*hidden]`).
    if name.contains(".bias_ih_l") || name.contains(".bias_hh_l") {
        if dims.len() != 1 {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe: tensor `{name}` has rank {} (dims={dims:?}), \
                 expected rank 1 for a PyTorch GRU bias vector \
                 [3*hidden] (FR-EX-08)",
                dims.len()
            )));
        }
        return Ok(());
    }
    // Batch-norm buffers: `running_mean` / `running_var` /
    // `num_batches_tracked` (PyTorch BN state_dict — the running
    // statistics are always 1D over the channel axis, and the tracked
    // counter is a scalar tensor that GGUF stores as rank 1).
    if name.ends_with(".running_mean") || name.ends_with(".running_var") {
        if dims.len() != 1 {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe: tensor `{name}` has rank {} (dims={dims:?}), \
                 expected rank 1 for a PyTorch batch-norm running \
                 statistic [channels] (FR-EX-08)",
                dims.len()
            )));
        }
        return Ok(());
    }
    if name.ends_with(".num_batches_tracked") {
        // PyTorch stores this as a 0-dim tensor; safetensors → GGUF
        // usually widens it to `[1]`. Accept either 0D (caught above)
        // or 1D with a single element.
        if dims.len() != 1 || dims[0] != 1 {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe: tensor `{name}` has dims={dims:?}, expected rank \
                 1 with a single element for BN's num_batches_tracked \
                 counter (FR-EX-08)"
            )));
        }
        return Ok(());
    }
    // Generic bias vectors — Conv / Linear biases are always 1D
    // ([out_channels] or [out_features]). Skip the GRU biases already
    // handled above.
    if name.ends_with(".bias") {
        if dims.len() != 1 {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe: tensor `{name}` has rank {} (dims={dims:?}), \
                 expected rank 1 for a Conv / Linear bias vector \
                 [out_channels] (FR-EX-08)",
                dims.len()
            )));
        }
        return Ok(());
    }
    // Batch-norm gamma (`.bn.weight` / `.bn{N}.weight`) — always 1D
    // over the channel axis. Checked before the generic `.weight` arm
    // so a mis-shaped BN gamma cannot masquerade as a Linear weight.
    if (name.contains(".bn.") || contains_bn_indexed_infix(name)) && name.ends_with(".weight") {
        if dims.len() != 1 {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe: tensor `{name}` has rank {} (dims={dims:?}), \
                 expected rank 1 for a PyTorch batch-norm gamma \
                 [channels] (FR-EX-08)",
                dims.len()
            )));
        }
        return Ok(());
    }
    // LayerNorm / GroupNorm gamma (`.ln.weight` / `.norm.weight` /
    // `.layernorm.weight`) — always 1D over the normalized axis.
    if (name.contains(".ln.") || name.contains(".norm.") || name.contains(".layernorm."))
        && name.ends_with(".weight")
    {
        if dims.len() != 1 {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe: tensor `{name}` has rank {} (dims={dims:?}), \
                 expected rank 1 for a LayerNorm / GroupNorm gamma \
                 [channels] (FR-EX-08)",
                dims.len()
            )));
        }
        return Ok(());
    }
    // Conv weights (`.conv.weight` / `.conv{N}.weight`) — always 3D
    // (Conv1d `[out, in, k]`) or 4D (Conv2d / ConvTranspose2d
    // `[out, in, kh, kw]`). A 1D or 2D conv weight almost certainly
    // indicates a flattened tensor from a converter bug.
    if (name.contains(".conv.") || contains_conv_indexed_infix(name)) && name.ends_with(".weight") {
        if !(dims.len() == 3 || dims.len() == 4) {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe: tensor `{name}` has rank {} (dims={dims:?}), \
                 expected rank 3 (Conv1d [out,in,k]) or rank 4 (Conv2d \
                 [out,in,kh,kw]) — a flattened conv weight indicates a \
                 converter bug (FR-EX-08)",
                dims.len()
            )));
        }
        return Ok(());
    }
    // Generic weight tensors. Ranks accepted:
    //   - 1D: LayerNorm / BN gamma that arrives under an unrecognized
    //         parent module name (fork tolerance).
    //   - 2D: catches Linear weights and any 2D matmul projection
    //         ([out, in]) — RMVPE forks sometimes use a Linear
    //         projection head instead of a Conv1d.
    //   - 3D: Conv1d weights `[out_channels, in_channels, kernel]`
    //         (upstream RMVPE head is a Conv1d → 360-class projection).
    //   - 4D: Conv2d / ConvTranspose2d weights
    //         `[out_channels, in_channels, kh, kw]` (upstream U-Net
    //         encoder / decoder blocks under unrecognized parent
    //         module names).
    // Ranks 5+ are rejected loudly. Note: GGUF caps tensor dimensions
    // at `MAX_TENSOR_DIMS` (raised to 8 on 2026-08-03 for Qwen2.5-Omni
    // vision Conv3d 5-D weights), so a rank-5+ tensor CAN now reach
    // this arm from a rogue converter — the loud rejection here is the
    // RMVPE-specific shape check (upstream RMVPE has no 5D weight),
    // not a defensive stub.
    if name.ends_with(".weight") {
        if !(1..=4).contains(&dims.len()) {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe: tensor `{name}` has rank {} (dims={dims:?}), \
                 expected rank 1-4 (BN/LN gamma [C], Linear [out,in], \
                 Conv1d [out,in,k], or Conv2d [out,in,kh,kw]) — \
                 upstream RMVPE has no 5D weight tensor (FR-EX-08)",
                dims.len()
            )));
        }
        return Ok(());
    }
    // Unrecognized suffix — fork tolerance: fall through so custom
    // buffers (e.g. `mel_extractor.mel_basis` scratch, non-standard
    // `unet.*.scale` tunables) still load. The prefix filter in
    // `from_gguf` already guaranteed the tensor is under a recognized
    // RMVPE-family root.
    Ok(())
}

/// Widens a GGUF tensor payload to a flat `Vec<f32>`. Supports F32,
/// F16, and BF16 (all upstream RMVPE releases are one of these). Every
/// other dtype is a loud [`VokraError::ModelLoad`] (FR-EX-08).
fn dequant_to_f32(
    gguf: &GgufFile,
    info: &vokra_core::gguf::GgufTensorInfo,
) -> Result<Vec<f32>, VokraError> {
    let bytes = gguf.tensor_data(&info.name).ok_or_else(|| {
        VokraError::ModelLoad(format!("rmvpe: no data slice for tensor `{}`", info.name))
    })?;
    let elems: usize = info.dimensions.iter().map(|&d| d as usize).product();

    match info.dtype {
        GgmlType::F32 => {
            if bytes.len() != elems * 4 {
                return Err(VokraError::ModelLoad(format!(
                    "rmvpe: tensor `{}` F32 byte count {} != elems {} * 4",
                    info.name,
                    bytes.len(),
                    elems
                )));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        GgmlType::F16 => {
            if bytes.len() != elems * 2 {
                return Err(VokraError::ModelLoad(format!(
                    "rmvpe: tensor `{}` F16 byte count {} != elems {} * 2",
                    info.name,
                    bytes.len(),
                    elems
                )));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        }
        GgmlType::BF16 => {
            if bytes.len() != elems * 2 {
                return Err(VokraError::ModelLoad(format!(
                    "rmvpe: tensor `{}` BF16 byte count {} != elems {} * 2",
                    info.name,
                    bytes.len(),
                    elems
                )));
            }
            // BF16 = top 16 bits of an f32 — `bits << 16` widens
            // losslessly (the same choke point
            // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`
            // documents).
            Ok(bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                .collect())
        }
        other => Err(VokraError::ModelLoad(format!(
            "rmvpe: tensor `{}` has unsupported dtype {other:?} \
             (only F32 / F16 / BF16 are accepted at this seam — FR-EX-08)",
            info.name
        ))),
    }
}

/// Widens an IEEE-754 half-precision f16 bit pattern to f32. Matches
/// the reference conversion (round-nearest-even is a no-op when
/// widening: every f16 value is representable exactly in f32).
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1F);
    let mant = u32::from(h & 0x3FF);
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: renormalize.
            let mut m = mant;
            let mut e = 1i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let e32 = (127 - 15 + e) as u32;
            sign | (e32 << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        // Infinity / NaN.
        sign | (0xFF << 23) | (mant << 13)
    } else {
        let e32 = exp + (127 - 15);
        sign | (e32 << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

// ---------------------------------------------------------------------------
// RMVPE — the public engine handle
// ---------------------------------------------------------------------------

/// Robust Model for Vocal Pitch Estimation (RMVPE) — the pitch
/// front-end required by RVC v2
/// (<https://github.com/Dream-High/RMVPE>, MIT).
///
/// Load with [`from_gguf`](Self::from_gguf) / [`open`](Self::open),
/// then call [`extract`](Self::extract) on a PCM buffer to obtain a
/// per-hop F0 track (frame count = `pcm.len() / hop`). See the module
/// doc for the current implementation-status matrix and the FR-EX-08
/// loud-error contract on the U-Net + GRU forward.
#[derive(Debug)]
pub struct RMVPE {
    config: RmvpeConfig,
    // The bound weights are held (real, dequantized) but the inner
    // U-Net + GRU kernel binding is a follow-up wave; the field is
    // deliberately `#[allow(dead_code)]` until the kernel lands so a
    // reader is not misled by an unused field.
    #[allow(dead_code)]
    weights: RmvpeWeights,
}

impl RMVPE {
    /// Loads an RMVPE model from a GGUF file on disk.
    ///
    /// The GGUF must:
    ///
    /// 1. Be openable by the standard GGUF reader — errors surface as
    ///    [`VokraError::Io`] / [`VokraError::ModelLoad`].
    /// 2. Carry at least one recognized RMVPE state_dict tensor
    ///    (`REQUIRED_TENSOR_PREFIXES`) — otherwise
    ///    [`RmvpeWeights::from_gguf`] refuses the bind (FR-EX-08).
    ///
    /// `vokra.rmvpe.*` metadata is optional (absent keys fall back to
    /// primary-source constants per [`RmvpeConfig::from_gguf`]).
    pub fn from_gguf(path: &Path) -> Result<Self, VokraError> {
        let gguf = GgufFile::open(path)?;
        let config = RmvpeConfig::from_gguf(&gguf);
        let weights = RmvpeWeights::from_gguf(&gguf)?;
        Ok(Self { config, weights })
    }

    /// Convenience alias for [`from_gguf`](Self::from_gguf).
    pub fn open(path: impl AsRef<Path>) -> Result<Self, VokraError> {
        Self::from_gguf(path.as_ref())
    }

    /// Returns the loaded [`RmvpeConfig`] for downstream inspection.
    pub fn config(&self) -> &RmvpeConfig {
        &self.config
    }

    /// Number of RMVPE-typical tensors bound from the GGUF. Purely a
    /// diagnostic accessor for the parity harness.
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Extracts a per-hop F0 track from `pcm`.
    ///
    /// The per-frame timing follows the [`RmvpeConfig::hop`] contract
    /// (`frames.len() == pcm.len() / hop`); the caller supplies
    /// `sample_rate` in Hz so the frame timestamps are honest even
    /// when the PCM was not resampled to the RMVPE-native 16 kHz.
    ///
    /// Under the current landing this method returns a frame-count-
    /// correct output with `hz = 0.0`, `voiced = false`,
    /// `confidence = 0.0` on every frame — the API surface is
    /// complete, but the U-Net + GRU + head forward that would fill
    /// in real pitch estimates is deferred to the follow-up wave
    /// gated on the real-checkpoint parity harness. Call
    /// [`extract_real`](Self::extract_real) instead when the caller
    /// wants a loud "kernel binding pending" signal (FR-EX-08)
    /// rather than a silent placeholder track.
    pub fn extract(&self, pcm: &[f32], sample_rate: u32) -> Vec<F0Frame> {
        let hop = (self.config.hop as usize).max(1);
        let n_frames = pcm.len() / hop;
        let sr = (sample_rate as f32).max(1.0);
        (0..n_frames)
            .map(|i| F0Frame {
                time_sec: (i * hop) as f32 / sr,
                hz: 0.0,
                voiced: false,
                confidence: 0.0,
            })
            .collect()
    }

    /// Runs the **real** RMVPE forward on `pcm`.
    ///
    /// Under the current landing this returns
    /// [`VokraError::UnsupportedOp`] with a message identifying the
    /// missing U-Net + GRU kernel binding — FR-EX-08 posture. The
    /// method exists so downstream integration tests can pin the loud
    /// contract without ambiguity: a caller who *wants* to know
    /// whether the real forward is available (rather than silently
    /// accepting the placeholder [`extract`](Self::extract) track)
    /// can check `extract_real` and route accordingly.
    ///
    /// Once the U-Net + GRU kernel binding lands, this method will
    /// return the real per-frame F0 track with sigmoid-thresholded
    /// V/UV and local-centroid Hz decoding.
    pub fn extract_real(
        &self,
        _pcm: &[f32],
        _sample_rate: u32,
    ) -> Result<Vec<F0Frame>, VokraError> {
        Err(VokraError::UnsupportedOp(
            "rmvpe: real U-Net + GRU forward is deferred to the follow-up wave gated on the \
             owner-side real-checkpoint parity harness (crates/vokra-parity/tests/parity_rmvpe.rs, \
             env PARITY_RMVPE_REAL_GGUF). Weights are bound and mel front-end + 360-class → Hz \
             decoding are ready; call `extract` for the frame-count-correct placeholder track \
             (FR-EX-08 loud-partial posture)"
                .to_owned(),
        ))
    }

    /// Computes the RMVPE mel spectrogram for `pcm`.
    ///
    /// Runs the real STFT + mel filterbank at the RMVPE axes
    /// ([`RmvpeConfig::n_fft`] / [`win_length`] / [`hop`] /
    /// [`n_mels`] / [`sample_rate`]). The output is a row-major
    /// `[n_frames, n_mels]` buffer of natural-log-magnitude mel
    /// energies (the input the follow-up U-Net + GRU forward
    /// consumes).
    ///
    /// This method is deliberately public so the parity harness can
    /// validate the mel front-end independently of the U-Net kernel
    /// binding wave. A caller who wants an f32 log-mel buffer for a
    /// downstream consumer can bypass the CNN by calling this
    /// directly.
    ///
    /// [`n_fft`]: RmvpeConfig::n_fft
    /// [`win_length`]: RmvpeConfig::win_length
    /// [`hop`]: RmvpeConfig::hop
    /// [`n_mels`]: RmvpeConfig::n_mels
    /// [`sample_rate`]: RmvpeConfig::sample_rate
    pub fn mel_spectrogram(&self, pcm: &[f32]) -> Vec<Vec<f32>> {
        mel_spectrogram(pcm, &self.config)
    }
}

// ---------------------------------------------------------------------------
// Mel spectrogram (real; uses vokra-ops STFT + mel filterbank)
// ---------------------------------------------------------------------------

/// Computes the RMVPE mel spectrogram at the config's axes.
///
/// Uses a Hann window (upstream default), centered framing with
/// reflect padding, backward normalization, and a real-input RFFT —
/// the librosa / torchaudio compatibility axes RMVPE was trained
/// against.
fn mel_spectrogram(pcm: &[f32], cfg: &RmvpeConfig) -> Vec<Vec<f32>> {
    use vokra_core::ir::graph::{
        MelAttrs, MelInterp, MelNorm, MelScale, Normalization, PadMode, StftAttrs, Window,
        WindowSymmetry,
    };
    use vokra_ops::mel::MelFilterbank;
    use vokra_ops::stft::stft;

    let stft_attrs = StftAttrs {
        n_fft: cfg.n_fft as usize,
        hop_length: cfg.hop as usize,
        win_length: cfg.win_length as usize,
        window: Window::Hann,
        window_symmetry: WindowSymmetry::Periodic,
        center: true,
        pad_mode: PadMode::Reflect,
        normalization: Normalization::Backward,
        causal: false,
        real_input: true,
    };
    let spec = stft(pcm, &stft_attrs).expect("stft with validated attrs must succeed");
    let n_frames = spec.frames;
    // Power spectrum |X|^2 (librosa `power=2.0` default for
    // `melspectrogram`).
    let power = spec.power();

    let mel_attrs = MelAttrs {
        sample_rate: cfg.sample_rate,
        n_fft: cfg.n_fft as usize,
        n_mels: cfg.n_mels as usize,
        fmin: cfg.fmin,
        fmax: Some(cfg.sample_rate as f32 / 2.0),
        scale: MelScale::Slaney,
        norm: MelNorm::Slaney,
        interp: MelInterp::Hz,
    };
    let fb = MelFilterbank::new(&mel_attrs);
    // Power → mel; layout the returned buffer as `[n_frames][n_mels]`.
    let mel_flat = fb.apply(&power, n_frames);
    let n_mels = cfg.n_mels as usize;

    // The upstream RMVPE Python code takes `log(mel + 1e-5)` before
    // feeding the U-Net; do the same so the mel we hand off is
    // exactly what the CNN was trained on.
    const EPS: f32 = 1e-5;
    let mut mel = Vec::with_capacity(n_frames);
    for t in 0..n_frames {
        let mut row = Vec::with_capacity(n_mels);
        for m in 0..n_mels {
            let v = mel_flat[t * n_mels + m].max(0.0);
            row.push((v + EPS).ln());
        }
        mel.push(row);
    }
    debug_assert_eq!(mel.len(), n_frames);
    debug_assert!(mel.iter().all(|r| r.len() == n_mels));
    mel
}

// ---------------------------------------------------------------------------
// 360-class → cents → Hz decoding
// ---------------------------------------------------------------------------

/// Decodes a single RMVPE 360-class sigmoid vector into a
/// `(hz, voiced, confidence)` triple.
///
/// Applies a local centroid over the 3 neighbour classes around the
/// argmax to refine the Hz estimate below the 20-cents-per-class grid
/// resolution. A frame is `voiced` when the peak sigmoid value clears
/// `voiced_threshold` (upstream default ≈ 0.03).
///
/// This is exposed so the parity harness can validate the decoding
/// primitive independently of the U-Net kernel binding wave.
pub fn decode_class_to_hz(
    probs: &[f32],
    cfg: &RmvpeConfig,
    voiced_threshold: f32,
) -> (f32, bool, f32) {
    if probs.is_empty() {
        return (0.0, false, 0.0);
    }
    // Argmax over the 360-class vector.
    let mut argmax = 0usize;
    let mut peak = probs[0];
    for (i, &p) in probs.iter().enumerate().skip(1) {
        if p > peak {
            peak = p;
            argmax = i;
        }
    }

    let voiced = peak >= voiced_threshold;
    if !voiced {
        return (0.0, false, peak.clamp(0.0, 1.0));
    }

    // Local centroid over the 3 neighbour classes (upstream RMVPE
    // `to_local_average_cents`): weighted mean of the class indices,
    // clipped at the vector edges.
    let n = probs.len();
    let lo = argmax.saturating_sub(1);
    let hi = (argmax + 2).min(n); // exclusive upper
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for (i, &p) in probs.iter().enumerate().take(hi).skip(lo) {
        let w = p.max(0.0);
        num += w * (i as f32);
        den += w;
    }
    let refined = if den > 0.0 { num / den } else { argmax as f32 };

    // Log-Hz grid: cents = refined * cents_per_class; Hz = base_hz *
    // 2^(cents / 1200) = base_hz * 2^(refined * cents_per_class /
    // 1200).
    let hz = cfg.base_hz * (2.0f32).powf(refined * cfg.cents_per_class / 1200.0);

    // Clamp Hz into the config's tracked band before reporting so a
    // saturated tail class does not surface as a nonsensical
    // multi-kHz F0.
    let hz = hz.clamp(cfg.fmin, cfg.fmax);
    (hz, true, peak.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufBuilder, GgufFile};

    /// A GGUF path that cannot possibly exist on any developer or CI host.
    fn nonexistent_gguf_path() -> std::path::PathBuf {
        std::path::PathBuf::from("/nonexistent/.vokra-rmvpe-red-fixture/does-not-exist.gguf")
    }

    /// Builds a minimal test GGUF that carries just enough for
    /// `RmvpeWeights::from_gguf` to accept it: one tensor under a
    /// recognized RMVPE prefix. The tensor payload is a small F32
    /// buffer with non-zero bit patterns so a silent widen / drop is
    /// visible.
    fn minimal_valid_rmvpe_gguf() -> Vec<u8> {
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "rmvpe");
        // A single conv-like tensor named under `unet.encoder.*`; the
        // shape / values are placeholder but honest (non-zero) so the
        // dequant path is exercised.
        let payload: Vec<u8> = [1.0f32, -2.0, 3.5, -0.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        b.add_tensor(
            "unet.encoder.block0.weight",
            GgmlType::F32,
            vec![2, 2],
            payload,
        )
        .expect("add tensor");
        b.to_bytes().expect("serialize gguf")
    }

    /// STEP 1 (contract): `from_gguf` on a missing path must surface a
    /// load failure. Mirrors the previous skeleton's contract; the real
    /// binding preserves it.
    #[test]
    fn from_gguf_reports_load_error_on_missing_path() {
        let path = nonexistent_gguf_path();
        let err = RMVPE::from_gguf(&path).expect_err("missing GGUF must not load");
        assert!(
            matches!(err, VokraError::Io(_) | VokraError::ModelLoad(_)),
            "expected VokraError::Io or ModelLoad for a missing path, got {err:?}"
        );
    }

    /// STEP 1 (contract): the frame-count contract is
    /// `extract(&pcm, sr).len() == pcm.len() / hop` with `hop = 160`.
    /// Uses the minimal valid GGUF fixture so the load path is exercised
    /// end-to-end (not the skeleton `with_defaults` shortcut).
    #[test]
    fn extract_frame_count_matches_hop() {
        let bytes = minimal_valid_rmvpe_gguf();
        let tmp = std::env::temp_dir().join(format!(
            "vokra-rmvpe-frame-count-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        std::fs::write(&tmp, &bytes).expect("write test gguf");

        let m = RMVPE::from_gguf(&tmp).expect("load valid gguf");
        let hop = m.config.hop as usize;
        // 16 033 samples = 1 s @ 16 kHz + 33 extra — exercises the
        // integer-truncation of `pcm.len() / hop`.
        let pcm = vec![0.0f32; 16_033];
        let frames = m.extract(&pcm, 16_000);
        assert_eq!(frames.len(), pcm.len() / hop);
        std::fs::remove_file(&tmp).ok();
    }

    /// FR-EX-08 loud-error contract: a GGUF that carries NO RMVPE-
    /// typical tensor is refused loudly rather than silently binding an
    /// all-zero forward.
    #[test]
    fn from_gguf_refuses_gguf_without_rmvpe_tensors() {
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "not-rmvpe");
        // An arbitrary tensor under a NON-RMVPE prefix so the binder
        // must reject it (not because there is *no* tensor, but because
        // there is no RMVPE-typical tensor).
        let payload: Vec<u8> = [42.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        b.add_tensor("whisper.conv1.weight", GgmlType::F32, vec![1], payload)
            .expect("add tensor");
        let bytes = b.to_bytes().expect("serialize");

        let tmp = std::env::temp_dir().join(format!(
            "vokra-rmvpe-refuse-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        std::fs::write(&tmp, &bytes).expect("write");

        let err = RMVPE::from_gguf(&tmp).expect_err("must refuse");
        assert!(
            matches!(err, VokraError::ModelLoad(ref m) if m.contains("no tensor matching")),
            "expected ModelLoad naming the missing prefix, got {err:?}"
        );
        std::fs::remove_file(&tmp).ok();
    }

    /// FR-EX-08 honest-partial contract: `extract_real` must surface a
    /// loud "kernel binding pending" error rather than silently
    /// returning placeholder frames — so a caller who wants to know
    /// whether the real forward is available can distinguish it from
    /// the placeholder track [`RMVPE::extract`] returns.
    #[test]
    fn extract_real_is_loud_pending_error() {
        let bytes = minimal_valid_rmvpe_gguf();
        let tmp = std::env::temp_dir().join(format!(
            "vokra-rmvpe-real-pending-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        std::fs::write(&tmp, &bytes).expect("write");

        let m = RMVPE::from_gguf(&tmp).expect("load valid gguf");
        let pcm = vec![0.0f32; 16_000];
        let err = m
            .extract_real(&pcm, 16_000)
            .expect_err("extract_real must be a loud pending error");
        assert!(
            matches!(err, VokraError::UnsupportedOp(ref m) if m.contains("real U-Net")),
            "expected UnsupportedOp naming the missing kernel, got {err:?}"
        );
        std::fs::remove_file(&tmp).ok();
    }

    /// `RmvpeConfig::from_gguf` reads the chunk when present and falls
    /// back to primary-source constants when absent. This exercises
    /// both paths.
    #[test]
    fn config_from_gguf_reads_chunk_with_fallback_defaults() {
        // Case 1: no `vokra.rmvpe.*` chunk — expect all-defaults.
        let empty = GgufFile::parse(GgufBuilder::new().to_bytes().expect("empty gguf"))
            .expect("parse empty");
        let cfg = RmvpeConfig::from_gguf(&empty);
        assert_eq!(
            cfg,
            RmvpeConfig::default(),
            "empty GGUF must yield defaults"
        );

        // Case 2: partial chunk — the present keys win, the absent ones
        // fall back to primary-source constants.
        let mut b = GgufBuilder::new();
        b.add_u32(GGUF_KEY_HOP, 320);
        b.add_u32(GGUF_KEY_N_MELS, 64);
        let file =
            GgufFile::parse(b.to_bytes().expect("with-hparams gguf")).expect("parse with-hparams");
        let cfg = RmvpeConfig::from_gguf(&file);
        assert_eq!(cfg.hop, 320);
        assert_eq!(cfg.n_mels, 64);
        // Untouched axes must still be primary-source defaults.
        assert_eq!(cfg.sample_rate, DEFAULT_SAMPLE_RATE);
        assert_eq!(cfg.n_class, DEFAULT_N_CLASS);
        assert_eq!(cfg.cents_per_class, DEFAULT_CENTS_PER_CLASS);
    }

    /// The 360-class decoder must land on the analytic Hz for a
    /// spike-only probability vector.
    ///
    /// A "spike at class 0" is base_hz; a spike at class 60 (= 60 * 20
    /// = 1200 cents = 1 octave) is 2 * base_hz. This pins the log-Hz
    /// grid math independently of the CNN + GRU kernel binding.
    #[test]
    fn decode_class_to_hz_matches_analytic_grid() {
        let cfg = RmvpeConfig::default();

        // Case: spike at class 0 → base_hz (bounded by fmin).
        let mut probs = vec![0.0f32; cfg.n_class as usize];
        probs[0] = 1.0;
        let (hz, voiced, conf) = decode_class_to_hz(&probs, &cfg, 0.03);
        assert!(voiced, "peak > threshold must be voiced");
        assert!(conf >= 0.03);
        // Class 0 corresponds to base_hz (32.703 Hz), which is below
        // the default fmin (30 Hz was chosen with a 3 Hz margin);
        // decode_class_to_hz clamps up to fmin.
        assert!(
            (hz - cfg.base_hz.max(cfg.fmin)).abs() < 0.5,
            "class-0 Hz must equal max(base_hz, fmin), got {hz}"
        );

        // Case: spike at class 60 → base_hz * 2^1 (1 octave up).
        let mut probs = vec![0.0f32; cfg.n_class as usize];
        probs[60] = 1.0;
        let (hz, voiced, _) = decode_class_to_hz(&probs, &cfg, 0.03);
        assert!(voiced);
        let expected = cfg.base_hz * 2.0;
        // Local centroid over the 3 neighbours of a single-spike
        // probability collapses to `argmax` (weight sum = w_argmax), so
        // the analytic value is exact.
        assert!(
            (hz - expected.clamp(cfg.fmin, cfg.fmax)).abs() < 0.5,
            "class-60 Hz must equal base_hz * 2 (1 octave), got {hz}"
        );

        // Case: peak below threshold → unvoiced, hz = 0.
        let probs = vec![0.001f32; cfg.n_class as usize];
        let (hz, voiced, _) = decode_class_to_hz(&probs, &cfg, 0.03);
        assert!(!voiced, "peak < threshold must be unvoiced");
        assert_eq!(hz, 0.0);
    }

    /// The real mel spectrogram must return a `[n_frames, n_mels]`
    /// buffer at the RMVPE axes. This does not check numeric
    /// correctness against a reference (that is the parity harness's
    /// job with a real checkpoint), only that the shape contract holds
    /// on a non-degenerate PCM input.
    #[test]
    fn mel_spectrogram_shape_contract() {
        let bytes = minimal_valid_rmvpe_gguf();
        let tmp = std::env::temp_dir().join(format!(
            "vokra-rmvpe-mel-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        std::fs::write(&tmp, &bytes).expect("write");

        let m = RMVPE::from_gguf(&tmp).expect("load");
        // 1 second of 16 kHz PCM (a 440 Hz sine, so the mel has real
        // energy in a specific band — makes the shape test more
        // meaningful than a zero buffer).
        let sr = m.config.sample_rate as f32;
        let f0 = 440.0f32;
        let pcm: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * f0 * (i as f32) / sr).sin())
            .collect();
        let mel = m.mel_spectrogram(&pcm);
        assert!(!mel.is_empty(), "mel must not be empty");
        assert_eq!(mel[0].len(), m.config.n_mels as usize);
        // Centered STFT with hop=160 yields `pcm.len() / hop + 1`
        // frames.
        assert_eq!(mel.len(), pcm.len() / (m.config.hop as usize) + 1);

        // A 440 Hz sine wave must produce energy above the log-eps
        // floor in *some* mel band — a silent-return placeholder mel
        // would leave every value at `ln(EPS) ≈ -11.5`, so any band
        // above that means the front-end is genuinely running.
        let floor = 1e-5f32.ln();
        let peak = mel
            .iter()
            .flat_map(|row| row.iter())
            .fold(f32::MIN, |a, &b| a.max(b));
        assert!(
            peak > floor + 1.0,
            "real mel must have peak energy well above the log-eps floor \
             (peak={peak}, floor={floor})"
        );

        std::fs::remove_file(&tmp).ok();
    }

    // -----------------------------------------------------------------
    // FQ-04: per-tensor shape validation (P0)
    //
    // These tests pin the FR-EX-08 loud-error contract on
    // `RmvpeWeights::from_gguf`: a mis-shaped real GGUF must be rejected
    // at load time rather than surfacing at forward time (which today
    // returns UnsupportedOp = never). They also pin the exact
    // upstream-style tensor names emitted by the primary-source RMVPE
    // state_dict layout — a converter rename becomes loud because the
    // round-trip test builds every real name and asserts they all load
    // together.
    // -----------------------------------------------------------------

    /// Helper: emits an f32 payload of a given shape as GGUF bytes. The
    /// values themselves are arbitrary — the tests below only care about
    /// (name, dtype, shape) for shape validation.
    fn f32_payload(elems: usize) -> Vec<u8> {
        (0..elems)
            .flat_map(|i| ((i as f32 * 0.125_f32) - 0.5).to_le_bytes())
            .collect()
    }

    /// Helper: writes a GGUF byte buffer to a per-test scratch path in
    /// the system temp dir. Uses the same nanosecond suffix pattern as
    /// the existing tests to avoid parallel-test collisions.
    fn write_scratch_gguf(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
        let tmp = std::env::temp_dir().join(format!(
            "vokra-rmvpe-{}-{}-{}.gguf",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        std::fs::write(&tmp, bytes).expect("write scratch gguf");
        tmp
    }

    /// Builds a GGUF that mirrors the **upstream RMVPE state_dict tensor
    /// names** (transcribed from the CNN + BN + GRU + Conv1d-head layout
    /// documented in the module doc). Also stamps `vokra.provenance.*`
    /// so the license-class round-trip pin exercises the metadata path
    /// end-to-end.
    ///
    /// Every shape below is at the correct rank for a real RMVPE
    /// checkpoint (Conv2d weights 4D, BN state 1D, GRU weights 2D, head
    /// Conv1d weight 3D). Sizes are small (a few filters) so the
    /// synthetic payload stays trivially small — the shape validator
    /// does not care about the exact channel counts, only the rank.
    ///
    /// Returns the GGUF bytes ready for `RMVPE::from_gguf`.
    fn upstream_rmvpe_state_dict_gguf() -> Vec<u8> {
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "rmvpe");
        b.add_string("vokra.model.name", "rmvpe");
        // License-class round-trip — mirrors the converter's
        // `vokra_core::stamp_provenance(&mut b, LicenseClass::Permissive,
        // "mit", ...)`. `RmvpeWeights::from_gguf` does not read the
        // provenance chunk (that is `crates/vokra-core/src/compliance`'s
        // job), but the test below asserts the round-trip is intact so a
        // future runtime-side compliance gate can rely on the same
        // stamped values.
        b.add_string("vokra.provenance.weight_license", "permissive");
        b.add_string("vokra.provenance.license", "mit");
        b.add_string("vokra.provenance.model_id", "rmvpe");
        b.add_string("vokra.provenance.source", "yxlllc/RMVPE");

        // Encoder block 0: Conv2d weight [n_out, n_in, kh, kw] + bias
        // [n_out] + BN gamma / beta / running stats.
        let (out_ch, in_ch, kh, kw) = (4usize, 1usize, 3usize, 3usize);
        b.add_tensor(
            "unet.encoder.block0.conv.weight",
            GgmlType::F32,
            vec![out_ch as u64, in_ch as u64, kh as u64, kw as u64],
            f32_payload(out_ch * in_ch * kh * kw),
        )
        .unwrap();
        b.add_tensor(
            "unet.encoder.block0.conv.bias",
            GgmlType::F32,
            vec![out_ch as u64],
            f32_payload(out_ch),
        )
        .unwrap();
        b.add_tensor(
            "unet.encoder.block0.bn.weight",
            GgmlType::F32,
            vec![out_ch as u64],
            f32_payload(out_ch),
        )
        .unwrap();
        b.add_tensor(
            "unet.encoder.block0.bn.bias",
            GgmlType::F32,
            vec![out_ch as u64],
            f32_payload(out_ch),
        )
        .unwrap();
        b.add_tensor(
            "unet.encoder.block0.bn.running_mean",
            GgmlType::F32,
            vec![out_ch as u64],
            f32_payload(out_ch),
        )
        .unwrap();
        b.add_tensor(
            "unet.encoder.block0.bn.running_var",
            GgmlType::F32,
            vec![out_ch as u64],
            f32_payload(out_ch),
        )
        .unwrap();

        // Decoder block 0: ConvTranspose2d weight (rank 4, same layout
        // for GGUF's purposes) + bias + BN.
        b.add_tensor(
            "unet.decoder.block0.conv.weight",
            GgmlType::F32,
            vec![in_ch as u64, out_ch as u64, kh as u64, kw as u64],
            f32_payload(in_ch * out_ch * kh * kw),
        )
        .unwrap();
        b.add_tensor(
            "unet.decoder.block0.conv.bias",
            GgmlType::F32,
            vec![in_ch as u64],
            f32_payload(in_ch),
        )
        .unwrap();

        // Intermediate GRU (single-layer, bidirectional). PyTorch
        // `nn.GRU` state_dict: weight_{ih,hh}_l0[_reverse] = 2D,
        // bias_{ih,hh}_l0[_reverse] = 1D.
        let (hidden, input_sz) = (8usize, 4usize);
        for suffix in ["l0", "l0_reverse"] {
            b.add_tensor(
                &format!("gru.weight_ih_{suffix}"),
                GgmlType::F32,
                vec![(3 * hidden) as u64, input_sz as u64],
                f32_payload(3 * hidden * input_sz),
            )
            .unwrap();
            b.add_tensor(
                &format!("gru.weight_hh_{suffix}"),
                GgmlType::F32,
                vec![(3 * hidden) as u64, hidden as u64],
                f32_payload(3 * hidden * hidden),
            )
            .unwrap();
            b.add_tensor(
                &format!("gru.bias_ih_{suffix}"),
                GgmlType::F32,
                vec![(3 * hidden) as u64],
                f32_payload(3 * hidden),
            )
            .unwrap();
            b.add_tensor(
                &format!("gru.bias_hh_{suffix}"),
                GgmlType::F32,
                vec![(3 * hidden) as u64],
                f32_payload(3 * hidden),
            )
            .unwrap();
        }

        // 360-class Conv1d head: weight [n_class, hidden, kernel=1] +
        // bias [n_class].
        let (n_class, kernel) = (360usize, 1usize);
        b.add_tensor(
            "head.weight",
            GgmlType::F32,
            vec![n_class as u64, hidden as u64, kernel as u64],
            f32_payload(n_class * hidden * kernel),
        )
        .unwrap();
        b.add_tensor(
            "head.bias",
            GgmlType::F32,
            vec![n_class as u64],
            f32_payload(n_class),
        )
        .unwrap();

        b.to_bytes().expect("serialize gguf")
    }

    /// **FQ-04 round-trip pin**: every upstream RMVPE state_dict tensor
    /// name at the correct rank loads without error, the tensor count
    /// matches the constructed set (a silent drop would surface here),
    /// and the license-class metadata round-trips through the GGUF
    /// (converter → runtime binder path).
    ///
    /// This is the "converter rename becomes loud" pin: if
    /// `crates/vokra-convert/src/models/rmvpe.rs` (or a fork thereof)
    /// ever emits a tensor under a rename that is not on the recognized
    /// prefix list, the prefix filter here would drop it silently and
    /// `tensor_count` would fall short — this test catches that
    /// regression by asserting the exact expected count.
    #[test]
    fn from_gguf_binds_upstream_state_dict_tensor_names_round_trip() {
        let bytes = upstream_rmvpe_state_dict_gguf();
        let tmp = write_scratch_gguf("upstream-names-roundtrip", &bytes);

        // Total tensor count from `upstream_rmvpe_state_dict_gguf`:
        //   - encoder block0: conv.weight + conv.bias + bn.weight
        //     + bn.bias + bn.running_mean + bn.running_var = 6
        //   - decoder block0: conv.weight + conv.bias = 2
        //   - gru (forward + reverse): 4 * 2 = 8
        //   - head: weight + bias = 2
        // ==> 18 total; every one must land in `RmvpeWeights` because
        //     they all match one of `REQUIRED_TENSOR_PREFIXES` and each
        //     shape is at the correct rank.
        const EXPECTED_TENSOR_COUNT: usize = 18;

        let m = RMVPE::from_gguf(&tmp).expect("upstream-shape GGUF must load");
        assert_eq!(
            m.tensor_count(),
            EXPECTED_TENSOR_COUNT,
            "every upstream-named tensor must land in `RmvpeWeights` — \
             a silent drop here indicates the prefix filter or shape \
             validator regressed"
        );

        // Spot-check that the correct dims + payload survived for one
        // tensor of each rank class. This is the "converter rename
        // becomes loud" pin: `tensor` returns `None` when the name is
        // not in the loaded set.
        let (dims, payload) = m
            .weights
            .tensor("unet.encoder.block0.conv.weight")
            .expect("encoder Conv2d weight must be bound by canonical name");
        assert_eq!(
            dims,
            &[4, 1, 3, 3],
            "encoder Conv2d weight dims must round-trip exactly"
        );
        assert_eq!(payload.len(), 4 * 3 * 3);

        let (dims, _) = m
            .weights
            .tensor("gru.weight_ih_l0")
            .expect("GRU weight_ih_l0 must be bound by canonical name");
        assert_eq!(dims, &[24, 4], "GRU weight_ih dims [3*hidden, input]");

        let (dims, _) = m
            .weights
            .tensor("head.weight")
            .expect("head Conv1d weight must be bound by canonical name");
        assert_eq!(
            dims,
            &[360, 8, 1],
            "head Conv1d weight dims [n_class, hidden, kernel]"
        );

        // License-class round-trip: reopen the raw GGUF and pin every
        // provenance chunk. This validates the converter → runtime
        // metadata path end-to-end so the FR-EX-08 shape gate cannot be
        // subverted by a checkpoint that omitted the license stamp.
        let bytes2 = std::fs::read(&tmp).expect("re-read gguf");
        let file = GgufFile::parse(bytes2).expect("parse gguf");
        assert_eq!(
            file.get("vokra.provenance.weight_license")
                .and_then(|v| v.as_str()),
            Some("permissive"),
            "license_class round-trip: `permissive` must survive the GGUF"
        );
        assert_eq!(
            file.get("vokra.provenance.license")
                .and_then(|v| v.as_str()),
            Some("mit"),
            "SPDX round-trip: `mit` must survive the GGUF"
        );
        assert_eq!(
            file.get("vokra.model.arch").and_then(|v| v.as_str()),
            Some("rmvpe")
        );

        std::fs::remove_file(&tmp).ok();
    }

    /// **FQ-04 loud-error pin (Conv2d weight)**: an encoder Conv2d
    /// weight flattened to 1D must be rejected at load time rather than
    /// silently binding and surfacing at forward time (which returns
    /// UnsupportedOp = never).
    #[test]
    fn from_gguf_rejects_encoder_conv_weight_flattened_to_1d() {
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "rmvpe");
        // Flattened Conv2d weight — same total elements (36 = 4*1*3*3)
        // but rank 1 instead of rank 4. This mirrors a converter bug
        // that lost the shape metadata.
        b.add_tensor(
            "unet.encoder.block0.conv.weight",
            GgmlType::F32,
            vec![36],
            f32_payload(36),
        )
        .unwrap();
        let bytes = b.to_bytes().expect("serialize");
        let tmp = write_scratch_gguf("bad-rank-conv", &bytes);

        let err = RMVPE::from_gguf(&tmp).expect_err("rank-1 Conv2d weight must be rejected");
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("unet.encoder.block0.conv.weight"),
                    "error must name the offending tensor: {msg}"
                );
                assert!(
                    msg.contains("rank 1"),
                    "error must report the actual rank: {msg}"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "error must cite the FR-EX-08 loud-error contract: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }

        std::fs::remove_file(&tmp).ok();
    }

    /// **FQ-04 loud-error pin (GRU weight)**: PyTorch's `nn.GRU`
    /// state_dict emits `weight_ih_l0` as rank 2. A 4D or 1D variant
    /// almost certainly indicates a converter bug (e.g. someone
    /// accidentally treated a GRU weight matrix like a Conv2d filter).
    #[test]
    fn from_gguf_rejects_gru_weight_wrong_rank() {
        for bad_dims in [vec![24u64], vec![24u64, 4, 1, 1]] {
            let mut b = GgufBuilder::new();
            b.add_string("vokra.model.arch", "rmvpe");
            let elems: usize = bad_dims.iter().product::<u64>() as usize;
            b.add_tensor(
                "gru.weight_ih_l0",
                GgmlType::F32,
                bad_dims.clone(),
                f32_payload(elems),
            )
            .unwrap();
            let bytes = b.to_bytes().expect("serialize");
            let tmp = write_scratch_gguf("bad-rank-gru", &bytes);

            let err = RMVPE::from_gguf(&tmp).expect_err("wrong-rank GRU weight must be rejected");
            match err {
                VokraError::ModelLoad(msg) => {
                    assert!(
                        msg.contains("gru.weight_ih_l0"),
                        "error must name the offending tensor for dims={bad_dims:?}: {msg}"
                    );
                    assert!(
                        msg.contains("expected rank 2"),
                        "error must state the expected GRU weight rank for dims={bad_dims:?}: {msg}"
                    );
                }
                other => panic!("expected ModelLoad for dims={bad_dims:?}, got {other:?}"),
            }

            std::fs::remove_file(&tmp).ok();
        }
    }

    /// **FQ-04 loud-error pin (bias)**: every Conv / Linear bias in
    /// RMVPE is 1D. A 2D bias signals either a converter merging
    /// [gamma, beta] into one tensor or a fork with an incompatible
    /// layout — both cases must fail loudly.
    #[test]
    fn from_gguf_rejects_bias_higher_rank() {
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "rmvpe");
        // A conv bias erroneously stored as a 2D matrix.
        b.add_tensor(
            "unet.encoder.block0.conv.bias",
            GgmlType::F32,
            vec![4u64, 1],
            f32_payload(4),
        )
        .unwrap();
        let bytes = b.to_bytes().expect("serialize");
        let tmp = write_scratch_gguf("bad-rank-bias", &bytes);

        let err = RMVPE::from_gguf(&tmp).expect_err("rank-2 bias must be rejected");
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("unet.encoder.block0.conv.bias"),
                    "error must name the tensor: {msg}"
                );
                assert!(
                    msg.contains("expected rank 1"),
                    "error must state expected rank 1 for a Conv bias: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }

        std::fs::remove_file(&tmp).ok();
    }

    /// **FQ-04 loud-error pin (batch-norm gamma wrong rank)**: the
    /// upstream RMVPE's `.bn.weight` (PyTorch `BatchNorm2d` gamma) is
    /// always 1D over the channel axis. A 2D `.bn.weight` almost
    /// certainly indicates a converter accidentally merging `[gamma,
    /// beta]` into one tensor.
    ///
    /// (Rank ≥ 5 CAN reach the GGUF load path since 2026-08-03 when
    /// `vokra_core::gguf::tensor::MAX_TENSOR_DIMS` was raised from 4 to
    /// 8 for Qwen2.5-Omni vision Conv3d 5-D weights. Upstream RMVPE
    /// itself has no 5D weight, so the rank-5 branch of
    /// `validate_tensor_shape` is now a *real* rejection surface — it
    /// stays covered by the pure-function `validate_tensor_shape_matrix`
    /// test below.)
    #[test]
    fn from_gguf_rejects_bn_weight_wrong_rank() {
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "rmvpe");
        b.add_tensor(
            "unet.encoder.block0.bn.weight",
            GgmlType::F32,
            vec![4u64, 2],
            f32_payload(8),
        )
        .unwrap();
        let bytes = b.to_bytes().expect("serialize");
        let tmp = write_scratch_gguf("bad-rank-bn", &bytes);

        let err = RMVPE::from_gguf(&tmp).expect_err("rank-2 BN gamma must be rejected");
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("unet.encoder.block0.bn.weight"),
                    "error must name the offending tensor: {msg}"
                );
                assert!(
                    msg.contains("expected rank 1"),
                    "error must state expected rank 1 for a BN gamma: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }

        std::fs::remove_file(&tmp).ok();
    }

    /// **FQ-04 loud-error pin (zero-sized axis)**: a truncated
    /// checkpoint or corrupted safetensors header that produces a
    /// zero-sized dimension must be rejected — a 0-elem tensor cannot
    /// be a real weight.
    #[test]
    fn from_gguf_rejects_zero_sized_axis() {
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "rmvpe");
        // 0-sized second axis — the GGUF header says [4, 0, 3, 3] which
        // is 0 total elements; the payload is an empty byte slice.
        b.add_tensor(
            "unet.encoder.block0.conv.weight",
            GgmlType::F32,
            vec![4u64, 0, 3, 3],
            Vec::<u8>::new(),
        )
        .unwrap();
        let bytes = b.to_bytes().expect("serialize");
        let tmp = write_scratch_gguf("zero-axis", &bytes);

        let err = RMVPE::from_gguf(&tmp).expect_err("zero-sized axis must be rejected");
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("zero-sized axis"),
                    "error must call out the zero-sized axis: {msg}"
                );
                assert!(
                    msg.contains("unet.encoder.block0.conv.weight"),
                    "error must name the offending tensor: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }

        std::fs::remove_file(&tmp).ok();
    }

    /// **FQ-04 fork-tolerance pin**: an unrecognized *suffix* under a
    /// recognized *prefix* (e.g. `unet.encoder.scale_factor`) still
    /// loads — the validator only rejects mis-shaped tensors under
    /// known name patterns, and the RMVPE community has diverged on
    /// some sub-module naming (e.g. `cnn.*` forks that flatten the
    /// U-Net). Rejecting an unrecognized suffix would false-fail on
    /// valid fork checkpoints.
    #[test]
    fn from_gguf_accepts_unrecognized_suffix_under_known_prefix() {
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "rmvpe");
        // Non-standard suffix — the shape here (3D) is not what a
        // canonical Conv1d / Conv2d / GRU / BN slot would use, but the
        // validator falls through because the suffix is not recognized.
        b.add_tensor(
            "unet.encoder.scale_factor",
            GgmlType::F32,
            vec![2u64, 3, 4],
            f32_payload(24),
        )
        .unwrap();
        let bytes = b.to_bytes().expect("serialize");
        let tmp = write_scratch_gguf("unknown-suffix", &bytes);

        let m = RMVPE::from_gguf(&tmp)
            .expect("unrecognized suffix under known prefix must load (fork tolerance)");
        assert_eq!(m.tensor_count(), 1);

        std::fs::remove_file(&tmp).ok();
    }

    /// **FQ-04 pure-function unit test**: pin the `validate_tensor_shape`
    /// contract in isolation so a future refactor of `from_gguf` cannot
    /// silently drop the shape check without a failing test surfacing.
    ///
    /// Exercises: canonical rank happy path, `.conv{N}.` / `.bn{N}.`
    /// indexed-infix variants (fork naming), unrecognized-suffix fork
    /// tolerance, and the every-loud-error branch of the validator.
    /// Note: the rank ≥ 5 arm covers a real code path since the
    /// 2026-08-03 GGUF `MAX_TENSOR_DIMS` bump (4 → 8) admits Conv3d
    /// weights — a rogue converter emitting a rank-5 RMVPE tensor
    /// would now reach `from_gguf`, and this arm loudly rejects it.
    #[test]
    fn validate_tensor_shape_matrix() {
        // Correct-rank happy path — every recognized name pattern
        // accepts its canonical rank.
        assert!(validate_tensor_shape("unet.encoder.block0.conv.weight", &[4, 1, 3, 3]).is_ok());
        assert!(validate_tensor_shape("unet.encoder.block0.conv.bias", &[4]).is_ok());
        assert!(validate_tensor_shape("unet.encoder.block0.bn.weight", &[4]).is_ok());
        assert!(validate_tensor_shape("unet.encoder.block0.bn.running_mean", &[4]).is_ok());
        assert!(validate_tensor_shape("gru.weight_ih_l0", &[24, 4]).is_ok());
        assert!(validate_tensor_shape("gru.weight_hh_l0_reverse", &[24, 8]).is_ok());
        assert!(validate_tensor_shape("gru.bias_ih_l0", &[24]).is_ok());
        assert!(validate_tensor_shape("head.weight", &[360, 8, 1]).is_ok());
        assert!(validate_tensor_shape("head.bias", &[360]).is_ok());
        // 2D linear-projection weight is accepted at the generic
        // `*.weight` arm (Linear layers do appear in some RMVPE forks).
        assert!(validate_tensor_shape("unet.encoder.linear.weight", &[8, 4]).is_ok());
        // Unrecognized suffix falls through (fork tolerance).
        assert!(validate_tensor_shape("unet.encoder.scale_factor", &[2, 3]).is_ok());
        // Conv1d weight at rank 3 is accepted.
        assert!(validate_tensor_shape("head.conv.weight", &[360, 8, 1]).is_ok());
        // Indexed infix variants (`.conv1.`, `.bn2.`) — some RMVPE
        // forks number a residual block's paired Conv / BN layers.
        assert!(validate_tensor_shape("unet.encoder.block0.conv1.weight", &[4, 1, 3, 3]).is_ok());
        assert!(validate_tensor_shape("unet.encoder.block0.conv2.weight", &[4, 4, 3, 3]).is_ok());
        assert!(validate_tensor_shape("unet.encoder.block0.bn1.weight", &[4]).is_ok());
        assert!(validate_tensor_shape("unet.encoder.block0.bn12.weight", &[4]).is_ok());

        // Loud-error cases — every one must return
        // `Err(ModelLoad(..))`. `matches!` over the enum variant so the
        // exact error message can evolve without breaking this pin.
        assert!(matches!(
            validate_tensor_shape("unet.encoder.block0.conv.weight", &[36]),
            Err(VokraError::ModelLoad(_))
        ));
        assert!(matches!(
            validate_tensor_shape("unet.encoder.block0.conv.weight", &[6, 6]),
            Err(VokraError::ModelLoad(_))
        ));
        // Indexed-infix conv weight must also refuse 2D.
        assert!(matches!(
            validate_tensor_shape("unet.encoder.block0.conv1.weight", &[6, 6]),
            Err(VokraError::ModelLoad(_))
        ));
        assert!(matches!(
            validate_tensor_shape("unet.encoder.block0.bn.weight", &[4, 1]),
            Err(VokraError::ModelLoad(_))
        ));
        // Indexed-infix BN weight must refuse rank 2.
        assert!(matches!(
            validate_tensor_shape("unet.encoder.block0.bn1.weight", &[4, 1]),
            Err(VokraError::ModelLoad(_))
        ));
        assert!(matches!(
            validate_tensor_shape("gru.weight_ih_l0", &[24]),
            Err(VokraError::ModelLoad(_))
        ));
        assert!(matches!(
            validate_tensor_shape("gru.bias_ih_l0", &[24, 1]),
            Err(VokraError::ModelLoad(_))
        ));
        assert!(matches!(
            validate_tensor_shape("unet.encoder.block0.conv.bias", &[4, 1]),
            Err(VokraError::ModelLoad(_))
        ));
        assert!(matches!(
            validate_tensor_shape("unet.encoder.block0.bn.running_var", &[4, 1]),
            Err(VokraError::ModelLoad(_))
        ));
        // Rank-5 weight (real code path since 2026-08-03
        // `MAX_TENSOR_DIMS` bump 4 → 8; RMVPE has no 5D weight so
        // this arm loudly rejects).
        assert!(matches!(
            validate_tensor_shape("head.weight", &[2, 2, 2, 2, 2]),
            Err(VokraError::ModelLoad(_))
        ));
        assert!(matches!(
            validate_tensor_shape("unet.encoder.block0.conv.weight", &[]),
            Err(VokraError::ModelLoad(_))
        ));
        assert!(matches!(
            validate_tensor_shape("unet.encoder.block0.conv.weight", &[4, 0, 3, 3]),
            Err(VokraError::ModelLoad(_))
        ));
    }

    /// **FQ-04 helper unit test**: pin the `contains_conv_indexed_infix`
    /// / `contains_bn_indexed_infix` substring walk in isolation. A
    /// regression here would silently unlock 1D / 2D Conv weights under
    /// `.conv1.` naming (fork checkpoints), defeating the FR-EX-08
    /// loud-error contract.
    #[test]
    fn indexed_infix_helpers_match_stem_digits_dot() {
        // Positive matches — `.stem{digits}.` pattern.
        assert!(contains_conv_indexed_infix(
            "unet.encoder.block0.conv1.weight"
        ));
        assert!(contains_conv_indexed_infix(
            "unet.encoder.block0.conv12.weight"
        ));
        assert!(contains_bn_indexed_infix("unet.encoder.block0.bn1.weight"));
        assert!(contains_bn_indexed_infix("unet.encoder.block0.bn12.weight"));
        // Multiple matches within one name — the walk continues past the
        // first `.stem.` and finds the later `.stem{digits}.`.
        assert!(contains_conv_indexed_infix(
            "unet.encoder.conv.block0.conv1.weight"
        ));

        // Negative matches — `.stem.` alone (no digits) is *not*
        // matched (already handled by `.contains(".stem.")` at the
        // call site).
        assert!(!contains_conv_indexed_infix(
            "unet.encoder.block0.conv.weight"
        ));
        assert!(!contains_bn_indexed_infix("unet.encoder.block0.bn.weight"));
        // Stems that share a prefix must not false-match (`conv_extra`
        // is not `conv{digits}`).
        assert!(!contains_conv_indexed_infix(
            "unet.encoder.block0.conv_extra.weight"
        ));
        // Digits without a trailing `.` do not match.
        assert!(!contains_conv_indexed_infix(
            "unet.encoder.block0.conv1weight"
        ));
        // Empty stem cannot match.
        assert!(!contains_conv_indexed_infix(""));
    }
}
