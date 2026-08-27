//! Robust Model for Vocal Pitch Estimation (RMVPE), used by RVC v2.
//!
//! # Fixed source and public artifact
//!
//! The production contract is transcribed from `yxlllc/RMVPE` commit
//! `0aabafba18289ca938a73af0b0297686abf4922d`. The public
//! `vokra/rmvpe` GGUF at revision
//! `3eb5fa8946f1074ba3959074c5cde95ec22b8c91` carries that E2E0
//! state-dict: 623 inference tensors plus 118 scalar BatchNorm counters.
//! The runtime validates that complete manifest before executing it; it does
//! not discover a partial topology or silently omit layers.
//!
//! `yxlllc/RMVPE` has no LICENSE file. The historical public GGUF is therefore
//! incorrectly stamped `MIT` / `permissive`; the strict loader rejects that
//! stamp and requires provenance-corrected `unknown` terms. `Dream-High/RMVPE`
//! is Apache-2.0, but it is not a GitHub fork relationship and does not prove
//! terms for the exact `yxlllc` checkpoint. See `docs/license-audit.md`.
//!
//! # Exact forward
//!
//! ```text
//! 16 kHz mono PCM
//!   -> magnitude log-mel (n_fft=win=1024, hop=160, 128 HTK mels)
//!   -> initial BatchNorm
//!   -> 5 encoder layers (4 residual Conv-BN-ReLU blocks + AvgPool2d)
//!   -> 4 intermediate layers (4 residual Conv-BN-ReLU blocks each)
//!   -> 5 decoder layers (ConvTranspose-BN-ReLU + paired skip concat
//!      + 4 residual Conv-BN-ReLU blocks)
//!   -> Conv2d 16 -> 3, collapse to 384 features/frame
//!   -> bidirectional GRU (384 -> 256 per direction)
//!   -> Linear 512 -> 360, sigmoid
//!   -> nine-bin local-average pitch decode
//! ```
//!
//! Class zero is exactly 31.7 Hz and adjacent classes are 20 cents apart.
//! The decoder does not clamp voiced estimates to `fmin`/`fmax`, matching
//! `src/utils.py`. The old public header values (`n_fft=2048`,
//! `base_hz=32.703197`) are recognized only as one exact historical pair and
//! normalized to the fixed source values before inference.
//!
//! Conv2d and ConvTranspose2d are lowered to GEMM; GRU gates use GEMV.
//! [`Compute::for_backend`] validates all learned operations before execution,
//! so a requested Metal backend never falls back to CPU. The legacy reduced
//! topology remains reachable only through explicit
//! [`CnnChainPolicy::Optional`] structural fixtures.
//!
//! # Real-checkpoint parity
//!
//! The env-gated harness in `crates/vokra-models/tests/parity_rmvpe.rs`
//! consumes fixtures produced by the independent fixed-upstream importer in
//! `tools/parity/rmvpe/dump_reference.py`. Until a recorded VAST CPU/Metal run
//! exists, the exact source transcription is implemented but real-weight
//! numerical parity remains unverified:
//!
//! - `VOKRA_RMVPE_REAL_GGUF` — points at the real Vokra GGUF and
//!   exercises the end-to-end forward on a 1 s 440 Hz sine (shape /
//!   finite / sigmoid range).
//! - `VOKRA_RMVPE_REAL_HIDDEN` — bypasses the CNN and feeds a pre-
//!   dumped hidden state directly into the BiGRU + head, so the
//!   argmax-match-rate gate isolates the numerical parity of the
//!   deterministic post-CNN primitives from any topology drift in the
//!   CNN chain. The buffer is a **raw little-endian f32 blob with no
//!   `.npy` header**, so `VOKRA_RMVPE_REAL_HIDDEN_FEATURE_DIM` must be
//!   exported alongside it (the blob carries no shape), together with
//!   `VOKRA_RMVPE_REAL_ARGMAX` (raw little-endian `u32`).
//!
//! The dumper emits `hidden.f32`, `argmax.u32`, and `meta.json`; exporting the
//! documented variables flips the test from a clean skip to a hard parity
//! gate. See `tools/parity/rmvpe/README.md` for the VAST-only recipe.
//!
//! Absent either env var the harness skips cleanly — never a fabricated
//! pass. See [`RMVPE::forward_from_hidden`] for the env-gated entry
//! point and the parity_rmvpe module doc for the fixture recipe.
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
use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgmlType, GgufFile};

use crate::compute::{Compute, HotOp};

use super::F0Frame;

mod native;

/// Backend-dispatched learned operators used by RMVPE.
///
/// The 2-D convolutions are lowered to row-major GEMM (im2col for Conv2d,
/// contribution projection for ConvTranspose2d); the recurrent GRU gates use
/// GEMV. Batch-norm, pooling, activations, layout transforms and pitch decoding
/// remain host glue. [`Compute::for_backend`] validates the complete set before
/// the first learned operator runs, so selecting Metal never falls back to CPU.
pub const RMVPE_HOT_OPS: &[HotOp] = &[HotOp::Gemm, HotOp::Gemv];

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
/// GGUF metadata key: fixed upstream source revision (string).
pub const GGUF_KEY_UPSTREAM_REVISION: &str = "vokra.rmvpe.upstream_revision";

// ---------------------------------------------------------------------------
// Primary-source constants — transcribed from the fixed upstream source at
// `native::UPSTREAM_REVISION` (`src/inference.py`, `src/spec.py`, and
// `src/constants.py`).
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
/// Default FFT size. `src/spec.py` uses `n_fft = win_length` when the
/// caller passes `None`; the fixed E2E0 inference path does exactly that.
pub const DEFAULT_N_FFT: u32 = 1024;
/// Default window length (upstream RMVPE = 1024).
pub const DEFAULT_WIN_LENGTH: u32 = 1024;
/// Default input sample rate (upstream RMVPE = 16 kHz PCM in).
pub const DEFAULT_SAMPLE_RATE: u32 = 16000;
/// Default number of pitch classes at the head (upstream RMVPE = 360).
pub const DEFAULT_N_CLASS: u32 = 360;
/// Default cents per pitch class (20 cents = 12 classes / semitone).
pub const DEFAULT_CENTS_PER_CLASS: f32 = 20.0;
/// Default class-0 anchor. Upstream uses
/// `10 * 2^(1997.3794084376191 / 1200)`, which is exactly 31.7 Hz.
pub const DEFAULT_BASE_HZ: f32 = 31.7;

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
    /// FFT size (default 1024).
    pub n_fft: u32,
    /// Window length in samples (default 1024).
    pub win_length: u32,
    /// Input sample rate (default 16000).
    pub sample_rate: u32,
    /// Number of pitch classes at the head (default 360).
    pub n_class: u32,
    /// Cents per pitch class in the log-Hz grid (default 20.0).
    pub cents_per_class: f32,
    /// Class-0 anchor frequency in Hz (default 31.7).
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

/// Broad tensor-name prefixes used by the low-level payload binder and legacy
/// [`CnnChainPolicy::Optional`] fixtures. The production loader additionally
/// applies `native::validate_contract`, which requires the exact E2E0 manifest;
/// matching one prefix is never sufficient for production execution.
///
/// Sourced from the upstream flattened `state_dict` layout: encoder /
/// decoder blocks, intermediate GRU, terminal 360-class head, and the
/// mel-extractor buffers used by RMVPE's Python-side front-end (which
/// this runtime replaces with a Rust-native `mel_spectrogram`).
///
/// # This is a payload filter, not the production topology contract
///
/// The broad names remain only so historical small fixtures can exercise
/// primitive kernels. The strict path checks all canonical
/// `unet.{encoder,intermediate,decoder}.*`, `cnn.*`, and `fc.*` names/shapes
/// before entering the exact native forward.
const REQUIRED_TENSOR_PREFIXES: &[&str] = &[
    "unet.",          // U-Net encoder + decoder (upstream: `unet.encoder.*` / `unet.decoder.*`)
    "encoder.",       // fallback prefix used by some RMVPE forks
    "decoder.",       // fallback prefix
    "gru.",           // intermediate GRU
    "head.",          // 360-class head
    "cnn.",           // some forks flatten the U-Net under `cnn.*`
    "fc.",            // canonical E2E0 BiGRU + 360-class Linear head
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
///
/// **What it does not check**: the complete production manifest. The default
/// constructor performs that separate exact header validation; only an
/// explicitly optional structural fixture can retain broad-prefix behavior.
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

    /// The `REQUIRED_TENSOR_PREFIXES` entries this artifact actually
    /// carries, in table order.
    ///
    /// Diagnostic only. When the forward cannot find its U-Net under
    /// the literal upstream scheme, this tells the caller which
    /// flattening convention the checkpoint *does* appear to use —
    /// a fork checkpoint reports e.g. `["cnn.", "gru.", "head."]`,
    /// which points straight at the re-conversion that would fix it.
    fn present_prefixes(&self) -> Vec<&'static str> {
        REQUIRED_TENSOR_PREFIXES
            .iter()
            .copied()
            .filter(|p| self.tensors.iter().any(|(n, _, _)| n.starts_with(p)))
            .collect()
    }

    /// Up to `max` bound tensor names in GGUF order, for error
    /// messages. Bounded so a 500-tensor checkpoint cannot turn one
    /// `ModelLoad` string into a screenful.
    fn sample_tensor_names(&self, max: usize) -> Vec<&str> {
        self.tensors
            .iter()
            .take(max)
            .map(|(n, _, _)| n.as_str())
            .collect()
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
    // PyTorch stores `num_batches_tracked` as a scalar. The historical public
    // GGUF preserves that rank-0 shape, while some safetensors bridges widen
    // it to `[1]`. It is an unused inference counter, but its manifest entry
    // is validated so a malformed file still fails loudly.
    if name.ends_with(".num_batches_tracked") {
        if !(dims.is_empty() || dims == [1]) {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe: tensor `{name}` has dims={dims:?}, expected scalar or \
                 [1] for BN's num_batches_tracked counter (FR-EX-08)"
            )));
        }
        return Ok(());
    }
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
    // Canonical E2E0 stores BatchNorm inside Sequential containers, so their
    // names use numeric slots rather than a `.bn.` stem. These must be handled
    // before the generic `.conv{N}.weight` arm below.
    if name.ends_with(".conv.1.weight")
        || name.ends_with(".conv.4.weight")
        || name.ends_with(".conv1.1.weight")
    {
        if dims.len() != 1 {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe: canonical BatchNorm tensor `{name}` has rank {} \
                 (dims={dims:?}), expected rank 1 [channels] (FR-EX-08)",
                dims.len()
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
// Forward primitives — inline PyTorch-native ops used by extract_real
//
// These live at file scope (module-private, `pub(super)` unnecessary) so
// the forward can use them directly without threading a Compute seam.
// They are self-contained (no vokra-ops dependency) which keeps the
// RMVPE forward legible at a single reading — the RMVPE-specific
// weight-discovery logic on `RmvpeWeights` calls out to these
// primitives, not the other way around.
// ---------------------------------------------------------------------------

/// Cap on how many `unet.encoder.block{i}` / `unet.decoder.block{i}`
/// blocks the discoverable forward will walk. Upstream RMVPE releases
/// max out at 5-6 encoder/decoder pairs; the cap here is a safety net
/// so a rogue checkpoint cannot cause an unbounded walk.
const MAX_UNET_BLOCKS: usize = 8;

/// Slope for the `LeakyReLU` used between every Conv2d / BN and after
/// every ConvTranspose2d. The upstream RMVPE Python uses 0.01 (PyTorch
/// default); pinned here as a `const` so a silent drift shows up in the
/// parity harness.
const LRELU_SLOPE: f32 = 0.01;

/// PyTorch `BatchNorm2d` default `eps` (`1e-5`) — pinned as a `const`
/// so a silent drift shows up in the parity harness.
const BN_EPS: f32 = 1e-5;

/// PyTorch `Conv2d` with `padding = 'same'` (zero-pad, unit stride) on
/// an `[C_in, H, W]` NCHW plane. `weight` is `[C_out, C_in, KH, KW]`
/// row-major; `bias` is `[C_out]`.
///
/// The output has the input's spatial size (SAME padding). Odd kernel
/// sizes place the extra padding on the trailing edge (matching
/// PyTorch's `padding='same'` semantics).
#[allow(clippy::too_many_arguments)] // conv parameter set
fn conv2d_pad_same(
    input: &[f32],
    in_c: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    out_c: usize,
    kh: usize,
    kw: usize,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    debug_assert_eq!(input.len(), in_c * h * w);
    debug_assert_eq!(weight.len(), out_c * in_c * kh * kw);
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), out_c);
    }
    let pad_h = kh / 2;
    let pad_w = kw / 2;
    let mut out = vec![0.0f32; out_c * h * w];
    for oc in 0..out_c {
        let bias_v = bias.map(|b| b[oc]).unwrap_or(0.0);
        for oy in 0..h {
            for ox in 0..w {
                let mut acc = bias_v;
                for ic in 0..in_c {
                    for ky in 0..kh {
                        let iy = oy as isize + ky as isize - pad_h as isize;
                        if iy < 0 || (iy as usize) >= h {
                            continue;
                        }
                        for kx in 0..kw {
                            let ix = ox as isize + kx as isize - pad_w as isize;
                            if ix < 0 || (ix as usize) >= w {
                                continue;
                            }
                            let w_idx = ((oc * in_c + ic) * kh + ky) * kw + kx;
                            let i_idx = (ic * h + iy as usize) * w + ix as usize;
                            acc += weight[w_idx] * input[i_idx];
                        }
                    }
                }
                out[(oc * h + oy) * w + ox] = acc;
            }
        }
    }
    out
}

/// Backend-dispatched counterpart of [`conv2d_pad_same`]. The host builds the
/// zero-padded im2col matrix and restores NCHW layout; every learned
/// multiply-accumulate, including bias addition, is executed by one GEMM.
#[allow(clippy::too_many_arguments)] // mirrors the scalar convolution contract
fn conv2d_pad_same_with_compute(
    compute: &Compute,
    input: &[f32],
    in_c: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    out_c: usize,
    kh: usize,
    kw: usize,
    bias: Option<&[f32]>,
) -> Result<Vec<f32>, VokraError> {
    debug_assert_eq!(input.len(), in_c * h * w);
    debug_assert_eq!(weight.len(), out_c * in_c * kh * kw);
    let spatial = h * w;
    let kernel = in_c * kh * kw;
    let pad_h = kh / 2;
    let pad_w = kw / 2;

    let mut patches = vec![0.0f32; spatial * kernel];
    for oy in 0..h {
        for ox in 0..w {
            let row = (oy * w + ox) * kernel;
            for ic in 0..in_c {
                for ky in 0..kh {
                    let iy = oy as isize + ky as isize - pad_h as isize;
                    if iy < 0 || iy as usize >= h {
                        continue;
                    }
                    for kx in 0..kw {
                        let ix = ox as isize + kx as isize - pad_w as isize;
                        if ix < 0 || ix as usize >= w {
                            continue;
                        }
                        let col = (ic * kh + ky) * kw + kx;
                        patches[row + col] = input[(ic * h + iy as usize) * w + ix as usize];
                    }
                }
            }
        }
    }

    // GEMM consumes B as [kernel, out_c], while PyTorch stores Conv2d
    // weights as [out_c, kernel].
    let mut weight_t = vec![0.0f32; kernel * out_c];
    for oc in 0..out_c {
        for k in 0..kernel {
            weight_t[k * out_c + oc] = weight[oc * kernel + k];
        }
    }
    let mut spatial_major = vec![0.0f32; spatial * out_c];
    compute.gemm_f32(
        spatial,
        out_c,
        kernel,
        &patches,
        &weight_t,
        bias,
        &mut spatial_major,
    )?;

    let mut out = vec![0.0f32; out_c * spatial];
    for p in 0..spatial {
        for oc in 0..out_c {
            out[oc * spatial + p] = spatial_major[p * out_c + oc];
        }
    }
    Ok(out)
}

/// PyTorch `BatchNorm2d` (inference mode) — in-place affine over an
/// `[C, H, W]` NCHW plane:
///
/// `y = (x - running_mean) / sqrt(running_var + eps) * gamma + beta`
///
/// (Inference-mode BN uses the running statistics, not the batch's own
/// mean / var; `training = False` in every deployed RMVPE checkpoint.)
#[allow(clippy::too_many_arguments)] // BN affine + running stats + eps
fn batchnorm2d_apply(
    x: &mut [f32],
    c: usize,
    h: usize,
    w: usize,
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
) {
    debug_assert_eq!(x.len(), c * h * w);
    debug_assert_eq!(gamma.len(), c);
    debug_assert_eq!(beta.len(), c);
    debug_assert_eq!(mean.len(), c);
    debug_assert_eq!(var.len(), c);
    let spatial = h * w;
    for ci in 0..c {
        let scale = gamma[ci] / (var[ci] + eps).sqrt();
        let shift = beta[ci] - mean[ci] * scale;
        let base = ci * spatial;
        for v in &mut x[base..base + spatial] {
            *v = *v * scale + shift;
        }
    }
}

/// PyTorch `MaxPool2d(kernel_size = 2, stride = 2)` — `[C, H, W]` →
/// `[C, H/2, W/2]` (integer truncation on odd axes; trailing edge is
/// dropped as PyTorch's default `ceil_mode = False` requires).
///
/// Returns `(out, h_out, w_out)`; a caller who needs the pool axes for
/// a downstream shape check reads them from the return tuple rather
/// than recomputing.
fn maxpool2d_2x2(x: &[f32], c: usize, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    debug_assert_eq!(x.len(), c * h * w);
    let h_out = h / 2;
    let w_out = w / 2;
    let mut out = vec![f32::NEG_INFINITY; c * h_out * w_out];
    for ci in 0..c {
        for oy in 0..h_out {
            for ox in 0..w_out {
                let mut m = f32::NEG_INFINITY;
                for ky in 0..2 {
                    for kx in 0..2 {
                        let iy = oy * 2 + ky;
                        let ix = ox * 2 + kx;
                        let v = x[(ci * h + iy) * w + ix];
                        if v > m {
                            m = v;
                        }
                    }
                }
                out[(ci * h_out + oy) * w_out + ox] = m;
            }
        }
    }
    (out, h_out, w_out)
}

/// PyTorch `ConvTranspose2d(kernel_size = KH x KW, stride = 2)` on an
/// `[C_in, H, W]` NCHW plane. `weight` is `[C_in, C_out, KH, KW]`
/// row-major (PyTorch stores ConvT2d weights transposed vs Conv2d).
/// `bias` is `[C_out]`.
///
/// Output size = `H_out = (H - 1) * 2 + KH`, likewise for `W`. The
/// convolution accumulates into the output tensor by scattering
/// `input[c_in, iy, ix] * weight[c_in, c_out, ky, kx]` into
/// `out[c_out, iy*2 + ky, ix*2 + kx]`. No output padding, no dilation
/// (matching upstream RMVPE's decoder blocks).
#[allow(clippy::too_many_arguments)] // conv parameter set
fn conv_transpose2d_stride2(
    input: &[f32],
    in_c: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    out_c: usize,
    kh: usize,
    kw: usize,
    bias: Option<&[f32]>,
) -> (Vec<f32>, usize, usize) {
    debug_assert_eq!(input.len(), in_c * h * w);
    debug_assert_eq!(weight.len(), in_c * out_c * kh * kw);
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), out_c);
    }
    let stride = 2usize;
    let h_out = (h - 1) * stride + kh;
    let w_out = (w - 1) * stride + kw;
    let mut out = vec![0.0f32; out_c * h_out * w_out];
    for ic in 0..in_c {
        for iy in 0..h {
            for ix in 0..w {
                let v = input[(ic * h + iy) * w + ix];
                for oc in 0..out_c {
                    for ky in 0..kh {
                        let oy = iy * stride + ky;
                        for kx in 0..kw {
                            let ox = ix * stride + kx;
                            let w_idx = ((ic * out_c + oc) * kh + ky) * kw + kx;
                            let o_idx = (oc * h_out + oy) * w_out + ox;
                            out[o_idx] += v * weight[w_idx];
                        }
                    }
                }
            }
        }
    }
    if let Some(b) = bias {
        let spatial = h_out * w_out;
        for (oc, &bv) in b.iter().enumerate().take(out_c) {
            let base = oc * spatial;
            for v in &mut out[base..base + spatial] {
                *v += bv;
            }
        }
    }
    (out, h_out, w_out)
}

/// Backend-dispatched counterpart of [`conv_transpose2d_stride2`]. One GEMM
/// projects every input pixel from `in_c` to all `(out_c, kh, kw)` kernel
/// contributions. The host then performs only the fixed stride-2 scatter/add
/// and NCHW layout work; no weight multiply runs outside the selected backend.
#[allow(clippy::too_many_arguments)] // mirrors the scalar convolution contract
fn conv_transpose2d_stride2_with_compute(
    compute: &Compute,
    input: &[f32],
    in_c: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    out_c: usize,
    kh: usize,
    kw: usize,
    bias: Option<&[f32]>,
) -> Result<(Vec<f32>, usize, usize), VokraError> {
    debug_assert_eq!(input.len(), in_c * h * w);
    debug_assert_eq!(weight.len(), in_c * out_c * kh * kw);
    let spatial = h * w;
    let projected = out_c * kh * kw;
    let h_out = (h - 1) * 2 + kh;
    let w_out = (w - 1) * 2 + kw;

    let mut input_spatial = vec![0.0f32; spatial * in_c];
    for iy in 0..h {
        for ix in 0..w {
            let p = iy * w + ix;
            for ic in 0..in_c {
                input_spatial[p * in_c + ic] = input[(ic * h + iy) * w + ix];
            }
        }
    }
    // ConvTranspose2d weights are [in_c, out_c, kh, kw], exactly the
    // [in_c, projected] B matrix needed here.
    let mut contributions = vec![0.0f32; spatial * projected];
    compute.gemm_f32(
        spatial,
        projected,
        in_c,
        &input_spatial,
        weight,
        None,
        &mut contributions,
    )?;

    let mut out = vec![0.0f32; out_c * h_out * w_out];
    if let Some(b) = bias {
        let out_spatial = h_out * w_out;
        for (oc, &value) in b.iter().enumerate().take(out_c) {
            out[oc * out_spatial..(oc + 1) * out_spatial].fill(value);
        }
    }
    for iy in 0..h {
        for ix in 0..w {
            let row = (iy * w + ix) * projected;
            for oc in 0..out_c {
                for ky in 0..kh {
                    let oy = iy * 2 + ky;
                    for kx in 0..kw {
                        let ox = ix * 2 + kx;
                        let col = (oc * kh + ky) * kw + kx;
                        out[(oc * h_out + oy) * w_out + ox] += contributions[row + col];
                    }
                }
            }
        }
    }
    Ok((out, h_out, w_out))
}

/// In-place `LeakyReLU(slope)`.
fn leaky_relu_inplace(x: &mut [f32], slope: f32) {
    for v in x {
        if *v < 0.0 {
            *v *= slope;
        }
    }
}

/// In-place logistic sigmoid `1 / (1 + exp(-x))`.
fn sigmoid_inplace(x: &mut [f32]) {
    for v in x {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

/// Numerically-stable sigmoid on a single scalar (used inside the GRU
/// cell body where a fused fold + sigmoid would otherwise inline `exp`
/// on every gate).
#[inline]
fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Row-major matrix-vector product `y = W . x + b`. `W` is
/// `[out_size, in_size]`; every row is a contiguous slice.
fn linear_forward(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    in_size: usize,
    out_size: usize,
) -> Vec<f32> {
    debug_assert_eq!(x.len(), in_size);
    debug_assert_eq!(weight.len(), out_size * in_size);
    debug_assert_eq!(bias.len(), out_size);
    let mut y = Vec::with_capacity(out_size);
    for o in 0..out_size {
        let row = &weight[o * in_size..(o + 1) * in_size];
        let mut acc = bias[o];
        for (wv, xv) in row.iter().zip(x.iter()) {
            acc += wv * xv;
        }
        y.push(acc);
    }
    y
}

/// PyTorch `nn.GRU` single-step cell (one time step of one direction).
///
/// Weight layout mirrors PyTorch:
/// - `w_ih` = `[3 * hidden_size, input_size]`, row groups
///   `[reset | update | new]`.
/// - `w_hh` = `[3 * hidden_size, hidden_size]`, same grouping.
/// - `b_ih`, `b_hh` = `[3 * hidden_size]` each — PyTorch keeps the two
///   biases split so the "add r · b_hn" arm of the new-gate matches
///   `torch.nn.GRU` byte-for-byte.
///
/// Equations (PyTorch reference — same as `torch.nn.GRUCell`):
/// ```text
/// r_t = sigmoid(W_ir . x + b_ir + W_hr . h + b_hr)
/// z_t = sigmoid(W_iz . x + b_iz + W_hz . h + b_hz)
/// n_t = tanh   (W_in . x + b_in + r_t * (W_hn . h + b_hn))
/// h_t = (1 - z_t) * n_t + z_t * h_{t-1}
/// ```
///
/// The recurrent bias split is what distinguishes PyTorch from the
/// RNNoise "type 2" GRU (which uses a single bias) — pinned here so a
/// reader diagnosing a numerical drift does not confuse the two.
#[allow(clippy::too_many_arguments)] // GRU parameter set
fn gru_cell_step(
    x: &[f32],
    h_prev: &[f32],
    w_ih: &[f32],
    w_hh: &[f32],
    b_ih: &[f32],
    b_hh: &[f32],
    hidden_size: usize,
    input_size: usize,
) -> Vec<f32> {
    debug_assert_eq!(x.len(), input_size);
    debug_assert_eq!(h_prev.len(), hidden_size);
    debug_assert_eq!(w_ih.len(), 3 * hidden_size * input_size);
    debug_assert_eq!(w_hh.len(), 3 * hidden_size * hidden_size);
    debug_assert_eq!(b_ih.len(), 3 * hidden_size);
    debug_assert_eq!(b_hh.len(), 3 * hidden_size);
    let h = hidden_size;

    // Pre-compute (W_ih . x + b_ih) and (W_hh . h + b_hh) once — same
    // sizes as PyTorch's fused matmul, split into the 3 gate rows.
    let mut ih = vec![0.0f32; 3 * h];
    let mut hh = vec![0.0f32; 3 * h];
    for i in 0..(3 * h) {
        let mut acc = b_ih[i];
        let row = &w_ih[i * input_size..(i + 1) * input_size];
        for (wv, xv) in row.iter().zip(x.iter()) {
            acc += wv * xv;
        }
        ih[i] = acc;
    }
    for i in 0..(3 * h) {
        let mut acc = b_hh[i];
        let row = &w_hh[i * h..(i + 1) * h];
        for (wv, hv) in row.iter().zip(h_prev.iter()) {
            acc += wv * hv;
        }
        hh[i] = acc;
    }

    let mut h_new = vec![0.0f32; h];
    for i in 0..h {
        let r = sigmoid_scalar(ih[i] + hh[i]);
        let z = sigmoid_scalar(ih[h + i] + hh[h + i]);
        let n = (ih[2 * h + i] + r * hh[2 * h + i]).tanh();
        h_new[i] = (1.0 - z) * n + z * h_prev[i];
    }
    h_new
}

/// Backend-dispatched GRU cell. Both learned projections execute as GEMV;
/// gate activations and the fixed element-wise recurrence remain host glue.
#[allow(clippy::too_many_arguments)] // mirrors the scalar GRU-cell contract
fn gru_cell_step_with_compute(
    compute: &Compute,
    x: &[f32],
    h_prev: &[f32],
    w_ih: &[f32],
    w_hh: &[f32],
    b_ih: &[f32],
    b_hh: &[f32],
    hidden_size: usize,
    input_size: usize,
) -> Result<Vec<f32>, VokraError> {
    let h = hidden_size;
    let mut ih = vec![0.0f32; 3 * h];
    let mut hh = vec![0.0f32; 3 * h];
    compute.gemv_f32(3 * h, input_size, w_ih, x, Some(b_ih), &mut ih)?;
    compute.gemv_f32(3 * h, h, w_hh, h_prev, Some(b_hh), &mut hh)?;

    let mut h_new = vec![0.0f32; h];
    for i in 0..h {
        let r = sigmoid_scalar(ih[i] + hh[i]);
        let z = sigmoid_scalar(ih[h + i] + hh[h + i]);
        let n = (ih[2 * h + i] + r * hh[2 * h + i]).tanh();
        h_new[i] = (1.0 - z) * n + z * h_prev[i];
    }
    Ok(h_new)
}

/// Projects every frame through a row-major `[out, in]` weight using GEMM.
fn linear_frames_with_compute(
    compute: &Compute,
    input: &[f32],
    n_frames: usize,
    in_features: usize,
    weight: &[f32],
    bias: &[f32],
    out_features: usize,
) -> Result<Vec<f32>, VokraError> {
    let mut weight_t = vec![0.0f32; in_features * out_features];
    for o in 0..out_features {
        for i in 0..in_features {
            weight_t[i * out_features + o] = weight[o * in_features + i];
        }
    }
    let mut out = vec![0.0f32; n_frames * out_features];
    compute.gemm_f32(
        n_frames,
        out_features,
        in_features,
        input,
        &weight_t,
        Some(bias),
        &mut out,
    )?;
    Ok(out)
}

/// Collapses an `[C, H, W]` NCHW buffer to `[H, C * W]` row-major
/// per-frame features. Used by the CNN → BiGRU seam: after the U-Net
/// runs, every time step needs one flat feature vector to feed the GRU.
fn collapse_nchw_to_frames(input: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    debug_assert_eq!(input.len(), c * h * w);
    let feature = c * w;
    let mut out = vec![0.0f32; h * feature];
    for ci in 0..c {
        for hi in 0..h {
            for wi in 0..w {
                let src = (ci * h + hi) * w + wi;
                let dst = hi * feature + ci * w + wi;
                out[dst] = input[src];
            }
        }
    }
    out
}

/// Extracts `(out_features, in_features)` from a `head.weight` tensor's
/// dims. Accepts:
///
/// - **rank 2** `[out, in]` — plain Linear head.
/// - **rank 3** `[out, in, kernel=1]` — Conv1d with kernel 1 collapses
///   to a Linear along the feature axis.
/// - **rank 4** `[out, in, 1, 1]` — Conv2d with 1x1 kernel likewise.
///
/// Every other rank / non-unit trailing dim is a loud
/// [`VokraError::ModelLoad`] because the collapse would silently drop
/// information (a kernel > 1 head cannot be treated as a Linear).
fn head_shape(dims: &[usize]) -> Result<(usize, usize), VokraError> {
    match dims.len() {
        2 => Ok((dims[0], dims[1])),
        3 => {
            if dims[2] != 1 {
                return Err(VokraError::ModelLoad(format!(
                    "rmvpe: head.weight rank-3 kernel dim {} != 1 (Conv1d with a \
                     non-1 kernel cannot be treated as a Linear head — FR-EX-08)",
                    dims[2]
                )));
            }
            Ok((dims[0], dims[1]))
        }
        4 => {
            if dims[2] != 1 || dims[3] != 1 {
                return Err(VokraError::ModelLoad(format!(
                    "rmvpe: head.weight rank-4 kernel dims {}x{} != 1x1 (Conv2d with \
                     a non-1x1 kernel cannot be treated as a Linear head — FR-EX-08)",
                    dims[2], dims[3]
                )));
            }
            Ok((dims[0], dims[1]))
        }
        other => Err(VokraError::ModelLoad(format!(
            "rmvpe: head.weight rank {other} unsupported — expected rank 2 (Linear \
             [out, in]) or rank 3/4 (Conv1d/Conv2d with kernel=1) (FR-EX-08)"
        ))),
    }
}

// ---------------------------------------------------------------------------
// RmvpeWeights forward helpers — weight discovery + block assembly
// ---------------------------------------------------------------------------

/// A single U-Net encoder / decoder block discovered from bound
/// tensors. The block carries every slice the primitive needs in one
/// borrow so the forward chain does not have to re-look up tensors on
/// every call.
struct RmvpeBlock<'a> {
    /// Output channel count (used by the caller to track the current
    /// C dimension between blocks).
    n_out: usize,
    /// Input channel count.
    n_in: usize,
    /// Kernel height.
    kh: usize,
    /// Kernel width.
    kw: usize,
    /// Conv weight, row-major `[out, in, kh, kw]`.
    conv_w: &'a [f32],
    /// Conv bias, `[out]`. May be `None` if the checkpoint omits it
    /// (`Conv2d(bias=False)` fed straight into BN).
    conv_b: Option<&'a [f32]>,
    /// BN gamma, `[out]`. `None` when the block has no BN.
    bn_gamma: Option<&'a [f32]>,
    /// BN beta, `[out]`.
    bn_beta: Option<&'a [f32]>,
    /// BN running mean, `[out]`.
    bn_mean: Option<&'a [f32]>,
    /// BN running variance, `[out]`.
    bn_var: Option<&'a [f32]>,
}

impl<'a> RmvpeBlock<'a> {
    /// Runs the encoder path: `Conv2d(pad=same)` + optional BN +
    /// `LeakyReLU(0.01)` + `MaxPool2d(2, 2)`.
    fn apply_encoder(
        &self,
        input: &[f32],
        in_c: usize,
        h: usize,
        w: usize,
    ) -> Result<(Vec<f32>, usize, usize), VokraError> {
        if in_c != self.n_in {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe encoder: input channels {in_c} != block.n_in {} (FR-EX-08)",
                self.n_in
            )));
        }
        let mut out = conv2d_pad_same(
            input,
            in_c,
            h,
            w,
            self.conv_w,
            self.n_out,
            self.kh,
            self.kw,
            self.conv_b,
        );
        if let (Some(g), Some(b), Some(m), Some(v)) =
            (self.bn_gamma, self.bn_beta, self.bn_mean, self.bn_var)
        {
            batchnorm2d_apply(&mut out, self.n_out, h, w, g, b, m, v, BN_EPS);
        }
        leaky_relu_inplace(&mut out, LRELU_SLOPE);
        // MaxPool2d halves the spatial axes; skipped when either axis is
        // already 1 (avoids a degenerate 0-size output on very short
        // PCM inputs).
        if h >= 2 && w >= 2 {
            let (pooled, h_out, w_out) = maxpool2d_2x2(&out, self.n_out, h, w);
            Ok((pooled, h_out, w_out))
        } else {
            Ok((out, h, w))
        }
    }

    /// Backend-dispatched encoder path. Learned convolution runs through
    /// `compute`; BN, activation and pooling retain the scalar reference glue.
    fn apply_encoder_with_compute(
        &self,
        compute: &Compute,
        input: &[f32],
        in_c: usize,
        h: usize,
        w: usize,
    ) -> Result<(Vec<f32>, usize, usize), VokraError> {
        if in_c != self.n_in {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe encoder: input channels {in_c} != block.n_in {} (FR-EX-08)",
                self.n_in
            )));
        }
        let mut out = conv2d_pad_same_with_compute(
            compute,
            input,
            in_c,
            h,
            w,
            self.conv_w,
            self.n_out,
            self.kh,
            self.kw,
            self.conv_b,
        )?;
        if let (Some(g), Some(b), Some(m), Some(v)) =
            (self.bn_gamma, self.bn_beta, self.bn_mean, self.bn_var)
        {
            batchnorm2d_apply(&mut out, self.n_out, h, w, g, b, m, v, BN_EPS);
        }
        leaky_relu_inplace(&mut out, LRELU_SLOPE);
        if h >= 2 && w >= 2 {
            let (pooled, h_out, w_out) = maxpool2d_2x2(&out, self.n_out, h, w);
            Ok((pooled, h_out, w_out))
        } else {
            Ok((out, h, w))
        }
    }

    /// Runs the decoder path: `ConvTranspose2d(stride = 2)` + optional
    /// BN (after the ConvT if bound) + `LeakyReLU(0.01)`. Skip-concat
    /// with the paired encoder is a follow-up wave (would need paired
    /// encoder cache); the shape axes still round-trip so downstream
    /// consumers can align.
    fn apply_decoder(
        &self,
        input: &[f32],
        in_c: usize,
        h: usize,
        w: usize,
    ) -> Result<(Vec<f32>, usize, usize), VokraError> {
        if in_c != self.n_in {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe decoder: input channels {in_c} != block.n_in {} (FR-EX-08)",
                self.n_in
            )));
        }
        let (mut out, h_out, w_out) = conv_transpose2d_stride2(
            input,
            in_c,
            h,
            w,
            self.conv_w,
            self.n_out,
            self.kh,
            self.kw,
            self.conv_b,
        );
        if let (Some(g), Some(b), Some(m), Some(v)) =
            (self.bn_gamma, self.bn_beta, self.bn_mean, self.bn_var)
        {
            batchnorm2d_apply(&mut out, self.n_out, h_out, w_out, g, b, m, v, BN_EPS);
        }
        leaky_relu_inplace(&mut out, LRELU_SLOPE);
        Ok((out, h_out, w_out))
    }

    /// Backend-dispatched decoder path. The fixed scatter/add after the GEMM
    /// is layout glue; all weight multiplication runs through `compute`.
    fn apply_decoder_with_compute(
        &self,
        compute: &Compute,
        input: &[f32],
        in_c: usize,
        h: usize,
        w: usize,
    ) -> Result<(Vec<f32>, usize, usize), VokraError> {
        if in_c != self.n_in {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe decoder: input channels {in_c} != block.n_in {} (FR-EX-08)",
                self.n_in
            )));
        }
        let (mut out, h_out, w_out) = conv_transpose2d_stride2_with_compute(
            compute,
            input,
            in_c,
            h,
            w,
            self.conv_w,
            self.n_out,
            self.kh,
            self.kw,
            self.conv_b,
        )?;
        if let (Some(g), Some(b), Some(m), Some(v)) =
            (self.bn_gamma, self.bn_beta, self.bn_mean, self.bn_var)
        {
            batchnorm2d_apply(&mut out, self.n_out, h_out, w_out, g, b, m, v, BN_EPS);
        }
        leaky_relu_inplace(&mut out, LRELU_SLOPE);
        Ok((out, h_out, w_out))
    }
}

impl RmvpeWeights {
    /// Discovers the `i`-th encoder block if present. Returns `None`
    /// when the block's conv weight is missing (signals the end of the
    /// discoverable encoder chain to the caller's loop).
    fn encoder_block(&self, i: usize) -> Option<RmvpeBlock<'_>> {
        self.discover_block("unet.encoder", i, /*conv2d=*/ true)
    }

    /// How many encoder blocks are discoverable under the literal
    /// upstream scheme, walking from `block0` and stopping at the first
    /// gap. `0` means the U-Net is invisible to this runtime — see
    /// [`RMVPE::extract_real`], which refuses that case rather than
    /// skipping the CNN.
    fn encoder_block_count(&self) -> usize {
        (0..MAX_UNET_BLOCKS)
            .take_while(|&i| self.encoder_block(i).is_some())
            .count()
    }

    /// Decoder-side counterpart of `encoder_block_count`.
    fn decoder_block_count(&self) -> usize {
        (0..MAX_UNET_BLOCKS)
            .take_while(|&i| self.decoder_block(i).is_some())
            .count()
    }

    /// Discovers the `i`-th decoder block if present.
    fn decoder_block(&self, i: usize) -> Option<RmvpeBlock<'_>> {
        // For ConvTranspose2d the weight layout is [in, out, kh, kw],
        // not [out, in, kh, kw]. Signal that to `discover_block` so the
        // returned `n_in` / `n_out` are correctly assigned.
        self.discover_block("unet.decoder", i, /*conv2d=*/ false)
    }

    fn discover_block(&self, prefix: &str, i: usize, conv2d: bool) -> Option<RmvpeBlock<'_>> {
        let conv_w_name = format!("{prefix}.block{i}.conv.weight");
        let (conv_w_dims, conv_w) = self.tensor(&conv_w_name)?;
        if conv_w_dims.len() != 4 {
            // A rank-3 Conv1d in a U-Net position is a real red flag —
            // return None so the caller stops walking rather than
            // silently applying a mis-shaped kernel.
            return None;
        }
        let (n_out, n_in) = if conv2d {
            (conv_w_dims[0], conv_w_dims[1])
        } else {
            // ConvTranspose2d: weight is [in, out, kh, kw].
            (conv_w_dims[1], conv_w_dims[0])
        };
        let kh = conv_w_dims[2];
        let kw = conv_w_dims[3];
        let conv_b = self
            .tensor(&format!("{prefix}.block{i}.conv.bias"))
            .map(|(_, p)| p);
        let bn_gamma = self
            .tensor(&format!("{prefix}.block{i}.bn.weight"))
            .map(|(_, p)| p);
        let bn_beta = self
            .tensor(&format!("{prefix}.block{i}.bn.bias"))
            .map(|(_, p)| p);
        let bn_mean = self
            .tensor(&format!("{prefix}.block{i}.bn.running_mean"))
            .map(|(_, p)| p);
        let bn_var = self
            .tensor(&format!("{prefix}.block{i}.bn.running_var"))
            .map(|(_, p)| p);
        Some(RmvpeBlock {
            n_out,
            n_in,
            kh,
            kw,
            conv_w,
            conv_b,
            bn_gamma,
            bn_beta,
            bn_mean,
            bn_var,
        })
    }

    /// Discovers the bidirectional GRU input / hidden dims from the
    /// bound `gru.weight_ih_l0` / `gru.weight_hh_l0` tensors.
    ///
    /// Layout: PyTorch `nn.GRU` state_dict
    /// - `weight_ih_l0` = `[3 * hidden, input]`
    /// - `weight_hh_l0` = `[3 * hidden, hidden]`
    ///
    /// The two dims cross-check: `hidden` derived from `w_hh.dims[1]`
    /// must equal `w_ih.dims[0] / 3` and `w_hh.dims[0] / 3`.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] when any of the four GRU tensors is
    /// missing, mis-ranked, or shape-inconsistent (FR-EX-08).
    fn discover_gru_shape(&self) -> Result<(usize, usize), VokraError> {
        let (ih_dims, _) = self.tensor("gru.weight_ih_l0").ok_or_else(|| {
            VokraError::ModelLoad(
                "rmvpe: gru.weight_ih_l0 missing — bidirectional GRU is required for \
                 a real forward (FR-EX-08)"
                    .into(),
            )
        })?;
        let (hh_dims, _) = self.tensor("gru.weight_hh_l0").ok_or_else(|| {
            VokraError::ModelLoad("rmvpe: gru.weight_hh_l0 missing (FR-EX-08)".into())
        })?;
        if ih_dims.len() != 2 || hh_dims.len() != 2 {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe: GRU weights must be rank 2; got ih={ih_dims:?}, hh={hh_dims:?} \
                 (FR-EX-08)"
            )));
        }
        let input_size = ih_dims[1];
        let hidden = hh_dims[1];
        if ih_dims[0] != 3 * hidden || hh_dims[0] != 3 * hidden {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe: GRU shape inconsistent — ih={ih_dims:?}, hh={hh_dims:?}; \
                 expected [3*hidden, input] / [3*hidden, hidden] with hidden={hidden} \
                 (FR-EX-08)"
            )));
        }
        // The reverse direction must also be bound (this is a
        // bidirectional GRU) with the same shape.
        let (ih_r_dims, _) = self.tensor("gru.weight_ih_l0_reverse").ok_or_else(|| {
            VokraError::ModelLoad(
                "rmvpe: gru.weight_ih_l0_reverse missing — bidirectional GRU is \
                 required (FR-EX-08)"
                    .into(),
            )
        })?;
        let (hh_r_dims, _) = self.tensor("gru.weight_hh_l0_reverse").ok_or_else(|| {
            VokraError::ModelLoad("rmvpe: gru.weight_hh_l0_reverse missing (FR-EX-08)".into())
        })?;
        if ih_r_dims != ih_dims || hh_r_dims != hh_dims {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe: BiGRU forward / reverse shape mismatch — \
                 fwd_ih={ih_dims:?}, rev_ih={ih_r_dims:?}, \
                 fwd_hh={hh_dims:?}, rev_hh={hh_r_dims:?} (FR-EX-08)"
            )));
        }
        Ok((input_size, hidden))
    }

    /// Runs a bidirectional GRU over `input` of shape
    /// `[n_frames, input_size]`. Returns a `[n_frames, 2 * hidden]`
    /// concatenation of the forward and reverse hidden states at each
    /// time step (matching `torch.nn.GRU(bidirectional=True)`).
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] when any of the eight bias / weight
    /// tensors is missing or shape-inconsistent (the discover_gru_shape
    /// call guarantees ih/hh consistency; this method additionally
    /// checks the four biases).
    fn apply_bigru(
        &self,
        input: &[f32],
        n_frames: usize,
        input_size: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, VokraError> {
        debug_assert_eq!(input.len(), n_frames * input_size);
        let (_, w_ih_f) = self
            .tensor("gru.weight_ih_l0")
            .expect("checked by discover");
        let (_, w_hh_f) = self
            .tensor("gru.weight_hh_l0")
            .expect("checked by discover");
        let (_, w_ih_r) = self
            .tensor("gru.weight_ih_l0_reverse")
            .expect("checked by discover");
        let (_, w_hh_r) = self
            .tensor("gru.weight_hh_l0_reverse")
            .expect("checked by discover");

        // Biases: PyTorch splits into b_ih + b_hh so the "add r · b_hn"
        // arm of the new-gate matches `torch.nn.GRU` byte-for-byte.
        let (_, b_ih_f) = self.tensor("gru.bias_ih_l0").ok_or_else(|| {
            VokraError::ModelLoad("rmvpe: gru.bias_ih_l0 missing (FR-EX-08)".into())
        })?;
        let (_, b_hh_f) = self.tensor("gru.bias_hh_l0").ok_or_else(|| {
            VokraError::ModelLoad("rmvpe: gru.bias_hh_l0 missing (FR-EX-08)".into())
        })?;
        let (_, b_ih_r) = self.tensor("gru.bias_ih_l0_reverse").ok_or_else(|| {
            VokraError::ModelLoad("rmvpe: gru.bias_ih_l0_reverse missing (FR-EX-08)".into())
        })?;
        let (_, b_hh_r) = self.tensor("gru.bias_hh_l0_reverse").ok_or_else(|| {
            VokraError::ModelLoad("rmvpe: gru.bias_hh_l0_reverse missing (FR-EX-08)".into())
        })?;
        for (name, len) in [
            ("bias_ih_l0", b_ih_f.len()),
            ("bias_hh_l0", b_hh_f.len()),
            ("bias_ih_l0_reverse", b_ih_r.len()),
            ("bias_hh_l0_reverse", b_hh_r.len()),
        ] {
            if len != 3 * hidden {
                return Err(VokraError::ModelLoad(format!(
                    "rmvpe: gru.{name} len {len} != 3 * hidden {} (FR-EX-08)",
                    3 * hidden
                )));
            }
        }

        // Forward pass — h_t for t = 0..n_frames.
        let mut fwd_states = vec![0.0f32; n_frames * hidden];
        let mut h_prev = vec![0.0f32; hidden];
        for t in 0..n_frames {
            let x = &input[t * input_size..(t + 1) * input_size];
            let h_new = gru_cell_step(
                x, &h_prev, w_ih_f, w_hh_f, b_ih_f, b_hh_f, hidden, input_size,
            );
            fwd_states[t * hidden..(t + 1) * hidden].copy_from_slice(&h_new);
            h_prev = h_new;
        }

        // Reverse pass — h_t for t = n_frames-1..=0.
        let mut rev_states = vec![0.0f32; n_frames * hidden];
        let mut h_prev = vec![0.0f32; hidden];
        for t in (0..n_frames).rev() {
            let x = &input[t * input_size..(t + 1) * input_size];
            let h_new = gru_cell_step(
                x, &h_prev, w_ih_r, w_hh_r, b_ih_r, b_hh_r, hidden, input_size,
            );
            rev_states[t * hidden..(t + 1) * hidden].copy_from_slice(&h_new);
            h_prev = h_new;
        }

        // Concatenate [fwd | rev] per time step -> [n_frames, 2*hidden].
        let mut out = vec![0.0f32; n_frames * 2 * hidden];
        for t in 0..n_frames {
            let base = t * 2 * hidden;
            out[base..base + hidden].copy_from_slice(&fwd_states[t * hidden..(t + 1) * hidden]);
            out[base + hidden..base + 2 * hidden]
                .copy_from_slice(&rev_states[t * hidden..(t + 1) * hidden]);
        }
        Ok(out)
    }

    /// Backend-dispatched bidirectional GRU. This deliberately mirrors
    /// [`Self::apply_bigru`] so the CPU path remains the established scalar
    /// oracle while Metal routes both gate projections through GEMV.
    fn apply_bigru_with_compute(
        &self,
        compute: &Compute,
        input: &[f32],
        n_frames: usize,
        input_size: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, VokraError> {
        debug_assert_eq!(input.len(), n_frames * input_size);
        let (_, w_ih_f) = self
            .tensor("gru.weight_ih_l0")
            .expect("checked by discover");
        let (_, w_hh_f) = self
            .tensor("gru.weight_hh_l0")
            .expect("checked by discover");
        let (_, w_ih_r) = self
            .tensor("gru.weight_ih_l0_reverse")
            .expect("checked by discover");
        let (_, w_hh_r) = self
            .tensor("gru.weight_hh_l0_reverse")
            .expect("checked by discover");
        let (_, b_ih_f) = self.tensor("gru.bias_ih_l0").ok_or_else(|| {
            VokraError::ModelLoad("rmvpe: gru.bias_ih_l0 missing (FR-EX-08)".into())
        })?;
        let (_, b_hh_f) = self.tensor("gru.bias_hh_l0").ok_or_else(|| {
            VokraError::ModelLoad("rmvpe: gru.bias_hh_l0 missing (FR-EX-08)".into())
        })?;
        let (_, b_ih_r) = self.tensor("gru.bias_ih_l0_reverse").ok_or_else(|| {
            VokraError::ModelLoad("rmvpe: gru.bias_ih_l0_reverse missing (FR-EX-08)".into())
        })?;
        let (_, b_hh_r) = self.tensor("gru.bias_hh_l0_reverse").ok_or_else(|| {
            VokraError::ModelLoad("rmvpe: gru.bias_hh_l0_reverse missing (FR-EX-08)".into())
        })?;
        for (name, len) in [
            ("bias_ih_l0", b_ih_f.len()),
            ("bias_hh_l0", b_hh_f.len()),
            ("bias_ih_l0_reverse", b_ih_r.len()),
            ("bias_hh_l0_reverse", b_hh_r.len()),
        ] {
            if len != 3 * hidden {
                return Err(VokraError::ModelLoad(format!(
                    "rmvpe: gru.{name} len {len} != 3 * hidden {} (FR-EX-08)",
                    3 * hidden
                )));
            }
        }

        let mut fwd_states = vec![0.0f32; n_frames * hidden];
        let mut h_prev = vec![0.0f32; hidden];
        for t in 0..n_frames {
            let x = &input[t * input_size..(t + 1) * input_size];
            let h_new = gru_cell_step_with_compute(
                compute, x, &h_prev, w_ih_f, w_hh_f, b_ih_f, b_hh_f, hidden, input_size,
            )?;
            fwd_states[t * hidden..(t + 1) * hidden].copy_from_slice(&h_new);
            h_prev = h_new;
        }

        let mut rev_states = vec![0.0f32; n_frames * hidden];
        let mut h_prev = vec![0.0f32; hidden];
        for t in (0..n_frames).rev() {
            let x = &input[t * input_size..(t + 1) * input_size];
            let h_new = gru_cell_step_with_compute(
                compute, x, &h_prev, w_ih_r, w_hh_r, b_ih_r, b_hh_r, hidden, input_size,
            )?;
            rev_states[t * hidden..(t + 1) * hidden].copy_from_slice(&h_new);
            h_prev = h_new;
        }

        let mut out = vec![0.0f32; n_frames * 2 * hidden];
        for t in 0..n_frames {
            let base = t * 2 * hidden;
            out[base..base + hidden].copy_from_slice(&fwd_states[t * hidden..(t + 1) * hidden]);
            out[base + hidden..base + 2 * hidden]
                .copy_from_slice(&rev_states[t * hidden..(t + 1) * hidden]);
        }
        Ok(out)
    }

    /// Discovered `head.weight` / `head.bias` view — the terminal
    /// 360-class Linear projection's shape (`out_features` /
    /// `in_features`) plus the two flat slices the forward feeds into
    /// [`linear_forward`].
    ///
    /// See [`RmvpeWeights::head_shape_and_slices`] for the discovery /
    /// shape-gate contract.
    fn head_shape_and_slices(&self) -> Result<RmvpeHead<'_>, VokraError> {
        let (head_w_dims, head_w) = self
            .tensor("head.weight")
            .ok_or_else(|| VokraError::ModelLoad("rmvpe: head.weight missing (FR-EX-08)".into()))?;
        let (_head_b_dims, head_b) = self
            .tensor("head.bias")
            .ok_or_else(|| VokraError::ModelLoad("rmvpe: head.bias missing (FR-EX-08)".into()))?;
        let (out_features, in_features) = head_shape(head_w_dims)?;
        if head_b.len() != out_features {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe: head.bias len {} != head out_features {out_features} (FR-EX-08)",
                head_b.len()
            )));
        }
        Ok(RmvpeHead {
            out_features,
            in_features,
            weight: head_w,
            bias: head_b,
        })
    }
}

/// Discovered `head.weight` / `head.bias` view. Factored out from a
/// bare `(usize, usize, &[f32], &[f32])` tuple to satisfy clippy's
/// `type_complexity` lint and to keep the forward chain's call site
/// self-documenting.
struct RmvpeHead<'a> {
    /// Number of pitch classes at the head (== `config.n_class`).
    out_features: usize,
    /// Head input width (== `2 * gru_hidden` for a bidirectional GRU).
    in_features: usize,
    /// Head weight slice, row-major `[out_features, in_features]`.
    weight: &'a [f32],
    /// Head bias slice, `[out_features]`.
    bias: &'a [f32],
}

// ---------------------------------------------------------------------------
// RMVPE — the public engine handle
// ---------------------------------------------------------------------------

/// Selects the exact production contract or the historical structural-fixture
/// path. It is an explicit constructor policy; a strict-load failure never
/// changes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CnnChainPolicy {
    /// **Default.** Require the complete fixed E2E0 manifest and run all five
    /// encoder/decoder layers with paired skips.
    #[default]
    Required,
    /// Opt-in: tolerate a checkpoint with no discoverable U-Net,
    /// handing the mel plane to the BiGRU unchanged.
    ///
    /// This exists for **structural smoke fixtures** — a hand-built
    /// GGUF that binds only a GRU + head, used to exercise the
    /// mel → collapse → BiGRU → head → decode chain without
    /// synthesizing fake CNN weights. Point it at a real checkpoint and
    /// the track it returns is not a pitch measurement: it is the
    /// output of a model missing most of its layers. Nothing downstream
    /// can detect that, which is the whole reason this is opt-in and
    /// not a fallback.
    ///
    /// Blocks that *are* discoverable still run — this relaxes the
    /// requirement, it does not disable the CNN.
    Optional,
}

/// Robust Model for Vocal Pitch Estimation (RMVPE) — the pitch
/// front-end required by RVC v2. The exact production contract is the fixed
/// `yxlllc/RMVPE` source named in the module docs; its code and checkpoint
/// terms are unresolved and therefore fail-closed.
///
/// Load with [`from_gguf`](Self::from_gguf) / [`open`](Self::open),
/// then call [`extract`](Self::extract) on a PCM buffer to obtain a
/// per-hop F0 track (frame count = `pcm.len() / hop`). Both loaders use
/// [`CnnChainPolicy::Required`]; the CNN-less path has its own
/// constructor, [`from_gguf_with_cnn_policy`](Self::from_gguf_with_cnn_policy).
///
/// [`extract`](Self::extract) and [`extract_real`](Self::extract_real)
/// are the same fallible forward — every method that names pitch
/// measures it. [`frame_times`](Self::frame_times) returns bare
/// timestamps and no pitch column at all, so no call on this type can
/// hand back a track that reads as a measurement. The sibling
/// [`super::crepe::CREPE`] and [`super::fcpe::FCPE`] binders expose the
/// same three names with the same meanings, so the F0 family reads the
/// same way at every call site. See the module doc for the current
/// implementation-status matrix.
///
/// Those two siblings also carry a `has_real_weights()` probe and this
/// type does not — deliberately, not by oversight. Their weight bundles
/// are `Option`al, so a metadata-only GGUF loads with none bound and a
/// caller may want to branch before calling. [`from_gguf`](Self::from_gguf)
/// refuses a GGUF carrying no RMVPE tensor at all, so a loaded `RMVPE`
/// always has weights: the probe would be a constant `true`, which is a
/// worse thing to expose than nothing.
#[derive(Debug)]
pub struct RMVPE {
    config: RmvpeConfig,
    /// Selected execution backend. CPU preserves the established scalar
    /// oracle; non-CPU backends are validated against [`RMVPE_HOT_OPS`]
    /// before forward execution.
    backend: BackendKind,
    // Bound (real, dequantized) upstream RMVPE state_dict tensors —
    // consumed by [`RMVPE::extract_real`] at forward time. The
    // `RmvpeWeights` shape gate in `from_gguf` guarantees every tensor
    // is at the correct rank before it reaches the forward.
    weights: RmvpeWeights,
    // Whether the forward may run with no discoverable U-Net. Set once
    // at load time by the constructor the caller chose, never inferred.
    cnn_policy: CnnChainPolicy,
}

impl RMVPE {
    /// Loads an RMVPE model from a GGUF file on disk under
    /// [`CnnChainPolicy::Required`].
    ///
    /// The GGUF must:
    ///
    /// 1. Be openable by the standard GGUF reader — errors surface as
    ///    [`VokraError::Io`] / [`VokraError::ModelLoad`].
    /// 2. Match the complete fixed E2E0 tensor manifest (623 inference
    ///    tensors and, when present, the 118 BatchNorm counters).
    /// 3. Carry the exact RMVPE metadata and fail-closed provenance described
    ///    in the module documentation.
    ///
    /// The historical public metadata pair is normalized as documented, but
    /// incomplete or differently named topologies are rejected during load.
    pub fn from_gguf(path: &Path) -> Result<Self, VokraError> {
        Self::from_gguf_with_cnn_policy(path, CnnChainPolicy::Required)
    }

    /// Loads an RMVPE model under an explicit [`CnnChainPolicy`].
    ///
    /// Prefer [`from_gguf`](Self::from_gguf). Pass
    /// [`CnnChainPolicy::Optional`] only for a hand-built structural
    /// fixture that deliberately carries no U-Net — read that variant's
    /// documentation first, because on a real checkpoint it converts a
    /// loud load failure into a silently wrong pitch track.
    ///
    /// # Errors
    ///
    /// Under [`CnnChainPolicy::Required`], the complete public contract is
    /// checked during load. [`CnnChainPolicy::Optional`] deliberately bypasses
    /// that contract for small in-tree structural fixtures only.
    pub fn from_gguf_with_cnn_policy(
        path: &Path,
        cnn_policy: CnnChainPolicy,
    ) -> Result<Self, VokraError> {
        let gguf = GgufFile::open(path)?;
        // Bind/shape-check first so a malformed tensor is reported by name
        // even when the complete public manifest is also incomplete.
        let weights = RmvpeWeights::from_gguf(&gguf)?;
        let mut config = RmvpeConfig::from_gguf(&gguf);
        if cnn_policy == CnnChainPolicy::Required {
            let flavor = native::validate_contract(&gguf)?;
            native::normalize_config(&mut config, flavor);
        }
        Ok(Self {
            config,
            backend: BackendKind::Cpu,
            weights,
            cnn_policy,
        })
    }

    /// Convenience alias for [`from_gguf`](Self::from_gguf).
    pub fn open(path: impl AsRef<Path>) -> Result<Self, VokraError> {
        Self::from_gguf(path.as_ref())
    }

    /// Selects the backend used by subsequent forward calls.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Returns the selected execution backend.
    #[must_use]
    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    /// The [`CnnChainPolicy`] this handle was loaded under.
    pub fn cnn_policy(&self) -> CnnChainPolicy {
        self.cnn_policy
    }

    /// Number of U-Net encoder layers. A strict model has the fixed five-layer
    /// E2E0 topology; an optional structural fixture reports the historical
    /// discoverable-block count.
    pub fn encoder_block_count(&self) -> usize {
        if self.cnn_policy == CnnChainPolicy::Required {
            5
        } else {
            self.weights.encoder_block_count()
        }
    }

    /// Decoder-side counterpart of [`encoder_block_count`](Self::encoder_block_count).
    pub fn decoder_block_count(&self) -> usize {
        if self.cnn_policy == CnnChainPolicy::Required {
            5
        } else {
            self.weights.decoder_block_count()
        }
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

    /// Extracts a per-hop F0 track from `pcm` by running the real
    /// RMVPE forward.
    ///
    /// This is a straight delegation to
    /// [`extract_real`](Self::extract_real) — identical behaviour,
    /// identical errors. It exists so the obvious name on a loaded
    /// model is the one that measures pitch, matching
    /// [`super::crepe::CREPE::extract`] and
    /// [`super::fcpe::FCPE::extract`].
    ///
    /// # History
    ///
    /// Before 2026-08-15 this name returned `Vec<F0Frame>` and handed
    /// back `hz = 0.0` / `voiced = false` / `confidence = 0.0` on every
    /// frame **regardless of whether weights were bound** — the real
    /// forward was reachable only as `extract_real`. A caller who
    /// reached for the obviously-named method on a fully loaded model
    /// got a frame-count-correct zero track, which downstream reads as
    /// "this audio is entirely unvoiced" rather than "no measurement
    /// was made". The timebase-only half now lives in
    /// [`frame_times`](Self::frame_times), which returns timestamps and
    /// nothing that could be read as a pitch estimate.
    ///
    /// # Errors
    ///
    /// Propagates [`extract_real`](Self::extract_real)'s errors
    /// verbatim — see that method for the list.
    pub fn extract(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<F0Frame>, VokraError> {
        self.extract_real(pcm, sample_rate)
    }

    /// Returns the analysis timestamps [`extract`](Self::extract) will
    /// emit for a PCM buffer of `pcm_len` samples, in seconds from the
    /// start of the buffer.
    ///
    /// `result.len()` is the frame-count contract (`pcm_len / hop`,
    /// integer-truncated per [`RmvpeConfig::hop`]); `result[i]` is the
    /// hop-aligned left edge of frame `i`. A `sample_rate` of `0` is
    /// clamped to `1` so the column stays finite rather than `NaN` /
    /// `±inf`.
    ///
    /// This runs no weights and cannot fail: it is pure arithmetic over
    /// the config, for callers that need to size or align a buffer
    /// before (or without) running the forward. It deliberately does
    /// **not** return [`F0Frame`]: a frame carries `hz` / `voiced` /
    /// `confidence` columns this method has no evidence for, and
    /// emitting zeros there is exactly the fabricated track the
    /// 2026-08-15 fix removed. Mirrors
    /// [`super::crepe::CREPE::frame_times`] and
    /// [`super::fcpe::FCPE::frame_times`].
    pub fn frame_times(&self, pcm_len: usize, sample_rate: u32) -> Vec<f32> {
        let hop = (self.config.hop as usize).max(1);
        let sr = (sample_rate as f32).max(1.0);
        (0..pcm_len / hop).map(|i| (i * hop) as f32 / sr).collect()
    }

    /// Builds the loud [`VokraError::ModelLoad`] for "the U-Net is not
    /// where I looked". `half` is `"encoder"` or `"decoder"`.
    ///
    /// The message names the literal scheme that was searched, the
    /// prefixes the artifact actually carries, and a bounded sample of
    /// its tensor names — because the caller's next move differs
    /// completely between "this is not an RMVPE at all" and "this is a
    /// fork that flattens its U-Net somewhere else", and only the
    /// artifact's own names can tell those apart.
    fn no_unet_error(&self, half: &str) -> VokraError {
        let present = self.weights.present_prefixes();
        let sample = self.weights.sample_tensor_names(8);
        let bound = self.weights.tensor_count();
        VokraError::ModelLoad(format!(
            "rmvpe: no U-Net {half} block found. The forward discovers its CNN by the \
             literal upstream scheme `unet.{half}.block{{i}}.conv.weight` (i = \
             0..{MAX_UNET_BLOCKS}); this artifact carries none of them, so the {half} \
             half of the U-Net would be skipped entirely and the returned track would \
             read as a pitch measurement without being one. Bound tensors: {bound}; \
             prefixes present: {present:?}; first bound names: {sample:?}. Either this \
             checkpoint flattens its U-Net under a fork convention (re-convert it to the \
             upstream names), or this runtime's scheme is wrong for the canonical \
             checkpoint and the walker needs fixing — the names above settle which. If \
             the CNN-less path is genuinely what you want, load with \
             `RMVPE::from_gguf_with_cnn_policy(path, CnnChainPolicy::Optional)` \
             (FR-EX-08: never a silent fallback)."
        ))
    }

    /// Runs the exact fixed E2E0 RMVPE forward on `pcm`. The complete
    /// Conv-BN-ReLU residual U-Net, paired skip connections, 384-feature
    /// collapse, BiGRU, 360-way head, sigmoid, and nine-bin decoder are
    /// described in the module documentation. The returned frame count is
    /// `pcm.len() / hop`; the internal waveform is padded to a multiple of 32
    /// frames as upstream does, and the extra predictions are discarded.
    ///
    /// A model loaded under [`CnnChainPolicy::Optional`] instead takes the
    /// explicitly requested historical structural-fixture path. Failure of the
    /// strict topology never selects that path implicitly.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] for non-16-kHz PCM; resampling is
    ///   never implicit.
    /// - [`VokraError::ModelLoad`] when validated weights cannot compose.
    /// - [`VokraError::UnsupportedOp`] when a selected non-CPU backend does
    ///   not cover the complete [`RMVPE_HOT_OPS`] set. There is no per-op CPU
    ///   fallback.
    pub fn extract_real(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<F0Frame>, VokraError> {
        if self.cnn_policy == CnnChainPolicy::Required {
            return native::forward(self, pcm, sample_rate);
        }
        self.extract_legacy_optional(pcm, sample_rate)
    }

    /// Historical reduced-topology path retained only for explicitly opted-in
    /// structural fixtures. Production loaders use the exact native E2E0
    /// forward above and can never enter this function by failed discovery.
    fn extract_legacy_optional(
        &self,
        pcm: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<F0Frame>, VokraError> {
        let compute = if self.backend == BackendKind::Cpu {
            None
        } else {
            Some(Compute::for_backend(self.backend, RMVPE_HOT_OPS)?)
        };

        // 1. Mel front-end (real STFT + mel filterbank).
        let mel = self.mel_spectrogram(pcm);
        let hop = (self.config.hop as usize).max(1);
        let expected_frames = pcm.len() / hop;

        // Early-exit on degenerate PCM (mel produces 0 frames when the
        // input is shorter than n_fft).
        if mel.is_empty() {
            return Ok(Vec::new());
        }

        // 2. Flatten mel [T, F] into NCHW [1, 1, T, F] for the CNN.
        //    Real RMVPE treats each mel frame as a single-channel row
        //    over the frequency axis — the first Conv2d lifts C=1 → 32.
        let t_in = mel.len();
        let f_in = self.config.n_mels as usize;
        debug_assert!(mel.iter().all(|r| r.len() == f_in));
        let mut feature: Vec<f32> = Vec::with_capacity(t_in * f_in);
        for row in &mel {
            feature.extend_from_slice(row);
        }
        let mut c_cur = 1usize;
        let mut h_cur = t_in;
        let mut w_cur = f_in;

        // 3. Encoder blocks. The walk stops at the first gap, so a
        //    checkpoint whose U-Net is named under a fork convention
        //    yields zero blocks — which must NOT fall through to the
        //    BiGRU (that would run ~none of the model and return a
        //    track indistinguishable from a real one). Under the
        //    default policy zero blocks is a loud load error.
        let mut encoder_blocks = 0usize;
        for i in 0..MAX_UNET_BLOCKS {
            let Some(block) = self.weights.encoder_block(i) else {
                break;
            };
            let (out, h_out, w_out) = if let Some(compute) = compute.as_ref() {
                block.apply_encoder_with_compute(compute, &feature, c_cur, h_cur, w_cur)?
            } else {
                block.apply_encoder(&feature, c_cur, h_cur, w_cur)?
            };
            feature = out;
            c_cur = block.n_out;
            h_cur = h_out;
            w_cur = w_out;
            encoder_blocks += 1;
        }
        if encoder_blocks == 0 && self.cnn_policy == CnnChainPolicy::Required {
            return Err(self.no_unet_error("encoder"));
        }

        // 4. Decoder blocks — same pattern, mirrored via
        //    ConvTranspose2d(stride=2), and the same zero-block gate.
        //    NOTE: even when every block IS found, this chain is not
        //    upstream's: the skip-concat is dropped (see the method's
        //    "Where this forward is not upstream RMVPE" section). The
        //    shape axes round-trip so a consumer aligning against
        //    frame_times() works; the values do not match upstream.
        let mut decoder_blocks = 0usize;
        for i in 0..MAX_UNET_BLOCKS {
            let Some(block) = self.weights.decoder_block(i) else {
                break;
            };
            let (out, h_out, w_out) = if let Some(compute) = compute.as_ref() {
                block.apply_decoder_with_compute(compute, &feature, c_cur, h_cur, w_cur)?
            } else {
                block.apply_decoder(&feature, c_cur, h_cur, w_cur)?
            };
            feature = out;
            c_cur = block.n_out;
            h_cur = h_out;
            w_cur = w_out;
            decoder_blocks += 1;
        }
        if decoder_blocks == 0 && self.cnn_policy == CnnChainPolicy::Required {
            return Err(self.no_unet_error("decoder"));
        }

        // 5. Collapse the CNN output into a per-frame feature vector.
        //    Layout: input is [C, H, W] stored as c*H*W + h*W + w.
        //    Output: [H, C * W] stored as h*(C*W) + c*W + w.
        //    When the CNN chain is absent the layout is [1, 1, T, F]
        //    with C=1, so this degenerates to a [T, F] view of the mel
        //    (the BiGRU consumes it directly).
        let feature_per_frame = collapse_nchw_to_frames(&feature, c_cur, h_cur, w_cur);
        let feature_dim = c_cur * w_cur;
        let n_frames = h_cur;

        // 6. BiGRU — required for a real forward.
        let (gru_input_size, gru_hidden) = self.weights.discover_gru_shape()?;
        if gru_input_size != feature_dim {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe: BiGRU input_size {gru_input_size} != post-CNN feature width \
                 {feature_dim} (C_last={c_cur} * F_last={w_cur}); either the CNN chain \
                 or the GRU hparams are mis-configured (FR-EX-08)"
            )));
        }
        let bigru_out = if let Some(compute) = compute.as_ref() {
            self.weights.apply_bigru_with_compute(
                compute,
                &feature_per_frame,
                n_frames,
                gru_input_size,
                gru_hidden,
            )?
        } else {
            self.weights
                .apply_bigru(&feature_per_frame, n_frames, gru_input_size, gru_hidden)?
        };
        let bigru_out_width = 2 * gru_hidden;

        // 7. Head projection (Conv1d/Conv2d with kernel=1 or plain
        //    Linear) + sigmoid.
        let head = self.weights.head_shape_and_slices()?;
        if head.out_features != self.config.n_class as usize {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe: head.weight out_features {} != n_class {} (FR-EX-08)",
                head.out_features, self.config.n_class
            )));
        }
        if head.in_features != bigru_out_width {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe: head.weight in_features {} != BiGRU output width \
                 {bigru_out_width} (2 * hidden={gru_hidden}) (FR-EX-08)",
                head.in_features
            )));
        }

        let mut all_probs = if let Some(compute) = compute.as_ref() {
            linear_frames_with_compute(
                compute,
                &bigru_out,
                n_frames,
                head.in_features,
                head.weight,
                head.bias,
                head.out_features,
            )?
        } else {
            let mut all_probs = Vec::with_capacity(n_frames * head.out_features);
            for t in 0..n_frames {
                let frame_in = &bigru_out[t * bigru_out_width..(t + 1) * bigru_out_width];
                all_probs.extend(linear_forward(
                    frame_in,
                    head.weight,
                    head.bias,
                    head.in_features,
                    head.out_features,
                ));
            }
            all_probs
        };
        sigmoid_inplace(&mut all_probs);

        // 8. Decode per-frame -> F0Frame. Honor the frame_times()
        //    frame-count contract by truncating any center-STFT extra
        //    frame.
        let sr = (sample_rate as f32).max(1.0);
        const VOICED_THRESHOLD: f32 = 0.03;
        let mut frames = Vec::with_capacity(n_frames);
        for t in 0..n_frames {
            let probs = &all_probs[t * head.out_features..(t + 1) * head.out_features];
            let (hz, voiced, confidence) =
                decode_class_to_hz(probs, &self.config, VOICED_THRESHOLD);
            frames.push(F0Frame {
                time_sec: (t * hop) as f32 / sr,
                hz,
                voiced,
                confidence,
            });
        }
        frames.truncate(expected_frames);
        Ok(frames)
    }

    /// Env-gated alternate entry point — bypasses the mel + CNN and
    /// feeds a pre-computed hidden-state buffer `[n_frames, feature]`
    /// straight into the BiGRU + head. Used by the parity harness
    /// (`VOKRA_RMVPE_REAL_HIDDEN`) to isolate the numerical parity of
    /// the deterministic post-CNN primitives from any topology drift
    /// in the CNN chain (once the owner-side dumper lands the .npy
    /// bridge). Also used by the synthetic weight structural smoke
    /// test so a fixture that omits the CNN weights can exercise the
    /// forward + sigmoid + decoder chain without a fake CNN.
    ///
    /// Skipping the CNN is this method's *signature*, not a fallback —
    /// the caller supplies the post-CNN state — so it does not consult
    /// [`CnnChainPolicy`] and never raises the missing-U-Net error.
    /// What comes out is a pitch track for the hidden state handed in,
    /// which is a claim about the caller's buffer and not about any
    /// audio this model has seen.
    ///
    /// # Errors
    ///
    /// Same [`VokraError::ModelLoad`] surface as
    /// [`extract_real`](Self::extract_real): GRU shape mismatch, head
    /// missing / mis-sized, or bias width off (FR-EX-08).
    pub fn forward_from_hidden(
        &self,
        hidden: &[f32],
        n_frames: usize,
        feature_dim: usize,
        sample_rate: u32,
    ) -> Result<Vec<F0Frame>, VokraError> {
        if self.cnn_policy == CnnChainPolicy::Required {
            return native::forward_from_hidden(self, hidden, n_frames, feature_dim, sample_rate);
        }
        self.forward_from_hidden_legacy_optional(hidden, n_frames, feature_dim, sample_rate)
    }

    fn forward_from_hidden_legacy_optional(
        &self,
        hidden: &[f32],
        n_frames: usize,
        feature_dim: usize,
        sample_rate: u32,
    ) -> Result<Vec<F0Frame>, VokraError> {
        let compute = if self.backend == BackendKind::Cpu {
            None
        } else {
            Some(Compute::for_backend(self.backend, RMVPE_HOT_OPS)?)
        };
        if hidden.len() != n_frames * feature_dim {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe::forward_from_hidden: hidden len {} != n_frames {n_frames} * \
                 feature_dim {feature_dim} (FR-EX-08)",
                hidden.len()
            )));
        }
        let (gru_input_size, gru_hidden) = self.weights.discover_gru_shape()?;
        if gru_input_size != feature_dim {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe::forward_from_hidden: BiGRU input_size {gru_input_size} != \
                 supplied feature_dim {feature_dim} (FR-EX-08)"
            )));
        }
        let bigru_out = if let Some(compute) = compute.as_ref() {
            self.weights.apply_bigru_with_compute(
                compute,
                hidden,
                n_frames,
                gru_input_size,
                gru_hidden,
            )?
        } else {
            self.weights
                .apply_bigru(hidden, n_frames, gru_input_size, gru_hidden)?
        };
        let bigru_out_width = 2 * gru_hidden;

        let head = self.weights.head_shape_and_slices()?;
        if head.in_features != bigru_out_width || head.out_features != self.config.n_class as usize
        {
            return Err(VokraError::ModelLoad(format!(
                "rmvpe::forward_from_hidden: head [{}, {}] does not match \
                 [n_class {}, 2*hidden {bigru_out_width}] (FR-EX-08)",
                head.out_features, head.in_features, self.config.n_class
            )));
        }

        let hop = (self.config.hop as usize).max(1);
        let sr = (sample_rate as f32).max(1.0);
        const VOICED_THRESHOLD: f32 = 0.03;
        let mut all_probs = if let Some(compute) = compute.as_ref() {
            linear_frames_with_compute(
                compute,
                &bigru_out,
                n_frames,
                head.in_features,
                head.weight,
                head.bias,
                head.out_features,
            )?
        } else {
            let mut all_probs = Vec::with_capacity(n_frames * head.out_features);
            for t in 0..n_frames {
                let frame_in = &bigru_out[t * bigru_out_width..(t + 1) * bigru_out_width];
                all_probs.extend(linear_forward(
                    frame_in,
                    head.weight,
                    head.bias,
                    head.in_features,
                    head.out_features,
                ));
            }
            all_probs
        };
        sigmoid_inplace(&mut all_probs);

        let mut frames = Vec::with_capacity(n_frames);
        for t in 0..n_frames {
            let probs = &all_probs[t * head.out_features..(t + 1) * head.out_features];
            let (hz, voiced, confidence) =
                decode_class_to_hz(probs, &self.config, VOICED_THRESHOLD);
            frames.push(F0Frame {
                time_sec: (t * hop) as f32 / sr,
                hz,
                voiced,
                confidence,
            });
        }
        Ok(frames)
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
    // `src/spec.py` multiplies the mel basis by STFT magnitude, not power.
    let magnitude = spec.magnitude();

    let mel_attrs = MelAttrs {
        sample_rate: cfg.sample_rate,
        n_fft: cfg.n_fft as usize,
        n_mels: cfg.n_mels as usize,
        fmin: cfg.fmin,
        fmax: Some(cfg.sample_rate as f32 / 2.0),
        scale: MelScale::Htk,
        norm: MelNorm::Slaney,
        interp: MelInterp::Hz,
    };
    let fb = MelFilterbank::new(&mel_attrs);
    // Magnitude → mel; layout the returned buffer as `[n_frames][n_mels]`.
    let mel_flat = fb.apply(&magnitude, n_frames);
    let n_mels = cfg.n_mels as usize;

    // Upstream uses `log(clamp(mel, min=1e-5))`.
    const EPS: f32 = 1e-5;
    let mut mel = Vec::with_capacity(n_frames);
    for t in 0..n_frames {
        let mut row = Vec::with_capacity(n_mels);
        for m in 0..n_mels {
            let v = mel_flat[t * n_mels + m].max(EPS);
            row.push(v.ln());
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
/// Applies a local centroid over the nine classes (`argmax ± 4`) around the
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

    // `src/utils.py::to_local_average_cents` uses argmax ± 4 (nine bins).
    let n = probs.len();
    let lo = argmax.saturating_sub(4);
    let hi = (argmax + 5).min(n); // exclusive upper
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

    (hz, true, peak.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufBuilder, GgufFile};

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= tolerance,
                "element {i}: actual={a}, expected={e}, abs_diff={} > {tolerance}",
                (a - e).abs()
            );
        }
    }

    #[test]
    fn compute_cpu_matches_scalar_learned_primitives() {
        let compute = Compute::cpu();

        let input = vec![0.2, -0.3, 0.7, 0.1, -0.4, 0.8, 0.5, -0.6];
        let conv_w: Vec<f32> = (0..36).map(|i| (i as f32 - 17.0) * 0.013).collect();
        let conv_b = [0.1, -0.2];
        let scalar = conv2d_pad_same(&input, 2, 2, 2, &conv_w, 2, 3, 3, Some(&conv_b));
        let dispatched = conv2d_pad_same_with_compute(
            &compute,
            &input,
            2,
            2,
            2,
            &conv_w,
            2,
            3,
            3,
            Some(&conv_b),
        )
        .expect("Compute CPU Conv2d");
        assert_close(&dispatched, &scalar, 1e-5);

        let trans_w: Vec<f32> = (0..24).map(|i| (i as f32 - 11.0) * 0.017).collect();
        let trans_b = [0.03, -0.04, 0.05];
        let (scalar, sh, sw) =
            conv_transpose2d_stride2(&input, 2, 2, 2, &trans_w, 3, 2, 2, Some(&trans_b));
        let (dispatched, dh, dw) = conv_transpose2d_stride2_with_compute(
            &compute,
            &input,
            2,
            2,
            2,
            &trans_w,
            3,
            2,
            2,
            Some(&trans_b),
        )
        .expect("Compute CPU ConvTranspose2d");
        assert_eq!((dh, dw), (sh, sw));
        assert_close(&dispatched, &scalar, 1e-5);

        let x = [0.25, -0.4, 0.75];
        let h_prev = [-0.2, 0.3];
        let w_ih: Vec<f32> = (0..18).map(|i| (i as f32 - 8.0) * 0.021).collect();
        let w_hh: Vec<f32> = (0..12).map(|i| (i as f32 - 5.0) * -0.019).collect();
        let b_ih = [0.01, -0.02, 0.03, -0.04, 0.05, -0.06];
        let b_hh = [-0.03, 0.02, -0.01, 0.04, -0.05, 0.06];
        let scalar = gru_cell_step(&x, &h_prev, &w_ih, &w_hh, &b_ih, &b_hh, 2, 3);
        let dispatched =
            gru_cell_step_with_compute(&compute, &x, &h_prev, &w_ih, &w_hh, &b_ih, &b_hh, 2, 3)
                .expect("Compute CPU GRU");
        assert_close(&dispatched, &scalar, 1e-6);
    }

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
    /// `frame_times(pcm.len(), sr).len() == pcm.len() / hop` with
    /// `hop = 160`, and each timestamp is the hop-aligned left edge of
    /// its frame. Uses the minimal valid GGUF fixture so the load path
    /// is exercised end-to-end (not the skeleton `with_defaults`
    /// shortcut).
    #[test]
    fn frame_times_matches_hop() {
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

        let m = RMVPE::from_gguf_with_cnn_policy(&tmp, CnnChainPolicy::Optional)
            .expect("load structural fixture");
        let hop = m.config.hop as usize;
        // 16 033 samples = 1 s @ 16 kHz + 33 extra — exercises the
        // integer-truncation of `pcm.len() / hop`.
        let pcm_len = 16_033usize;
        let times = m.frame_times(pcm_len, 16_000);
        assert_eq!(times.len(), pcm_len / hop);
        for (i, t) in times.iter().enumerate() {
            assert!(t.is_finite(), "frame {i}: time_sec {t} is not finite");
            let expected = (i * hop) as f32 / 16_000.0;
            assert!(
                (t - expected).abs() < 1e-6,
                "frame {i}: time_sec {t} is not the hop-aligned left edge {expected}"
            );
        }
        std::fs::remove_file(&tmp).ok();
    }

    /// A zero `sample_rate` must not produce a `NaN` / `±inf`
    /// timestamp column — the clamp to `1` keeps the output finite.
    #[test]
    fn frame_times_clamps_zero_sample_rate() {
        let bytes = minimal_valid_rmvpe_gguf();
        let tmp = std::env::temp_dir().join(format!(
            "vokra-rmvpe-zero-sr-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        std::fs::write(&tmp, &bytes).expect("write test gguf");

        let m = RMVPE::from_gguf_with_cnn_policy(&tmp, CnnChainPolicy::Optional)
            .expect("load structural fixture");
        let times = m.frame_times(16_000, 0);
        assert!(!times.is_empty(), "16 000 samples must yield frames");
        assert!(
            times.iter().all(|t| t.is_finite()),
            "a zero sample_rate must be clamped, not divided by"
        );
        std::fs::remove_file(&tmp).ok();
    }

    /// Regression pin for the 2026-08-15 honesty fix: `extract` must be
    /// the REAL forward, so on a GGUF whose bound tensors do not compose
    /// into a walkable forward it reports the missing weight loudly.
    /// Before the fix this name unconditionally returned an all-zero
    /// `F0Frame` track that a caller could not tell apart from a real
    /// measurement of silence.
    ///
    /// The assertion is deliberately "is an `Err`" rather than "is a zero
    /// track": a future refactor that quietly restores the fabricated
    /// path fails here.
    #[test]
    fn extract_is_the_real_forward_not_a_fabricated_track() {
        let bytes = minimal_valid_rmvpe_gguf();
        let tmp = std::env::temp_dir().join(format!(
            "vokra-rmvpe-extract-real-path-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        std::fs::write(&tmp, &bytes).expect("write test gguf");

        let m = RMVPE::from_gguf_with_cnn_policy(&tmp, CnnChainPolicy::Optional)
            .expect("load structural fixture");
        let pcm = vec![0.0f32; 16_000];
        let err = m
            .extract(&pcm, 16_000)
            .expect_err("extract must refuse a GGUF without a walkable forward");
        assert!(
            matches!(err, VokraError::ModelLoad(_)),
            "expected ModelLoad naming the missing weight, got {err:?}"
        );
        std::fs::remove_file(&tmp).ok();
    }

    /// `extract` and `extract_real` are the same code path, so on a
    /// fixture whose weights DO compose they must agree frame for frame.
    /// Pins the delegation so the two names can never drift into two
    /// behaviours again — and pins that what they return is a real
    /// measurement rather than the old fabricated column.
    ///
    /// Uses the CNN-less structural fixture, so it must opt into
    /// [`CnnChainPolicy::Optional`]; under the default policy this GGUF
    /// is (correctly) refused.
    #[test]
    fn extract_agrees_with_extract_real_on_a_real_forward() {
        let bytes = smoke_no_cnn_gguf(/*n_mels=*/ 128, /*gru_hidden=*/ 8);
        let tmp = std::env::temp_dir().join(format!(
            "vokra-rmvpe-alias-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        std::fs::write(&tmp, &bytes).expect("write");

        let m = RMVPE::from_gguf_with_cnn_policy(&tmp, CnnChainPolicy::Optional)
            .expect("load smoke gguf");
        let sr = m.config().sample_rate as f32;
        let pcm: Vec<f32> = (0..m.config().sample_rate as usize)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / sr).sin())
            .collect();

        let via_extract = m.extract(&pcm, 16_000).expect("real forward must run");
        let via_extract_real = m.extract_real(&pcm, 16_000).expect("real forward must run");
        assert_eq!(
            via_extract, via_extract_real,
            "the two names must not drift"
        );

        // The result is a REAL track, not the old fabricated one. The
        // fabricated path pinned `confidence` at exactly 0.0 on EVERY
        // frame, so a single positive frame is enough to tell the two
        // apart — and that is all this pins, deliberately: `confidence`
        // is the peak sigmoid over the 360 classes
        // (`decode_class_to_hz`), and asserting it is positive on every
        // frame would be a claim about the synthetic fixture's logits
        // that this test has no basis to make.
        assert!(
            via_extract.iter().any(|f| f.confidence > 0.0),
            "a real forward must produce some positive confidence, \
             not the fabricated all-zero column"
        );

        // The frame count matches what `frame_times` promised, so a
        // consumer can still pre-size from the timebase.
        assert_eq!(
            via_extract.len(),
            m.frame_times(pcm.len(), 16_000).len(),
            "extract must honor the frame_times frame-count contract"
        );
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

    /// FR-EX-08 loud-error contract: `extract_real` must refuse a GGUF
    /// whose bound tensors do not compose into a walkable forward
    /// (`minimal_valid_rmvpe_gguf` binds one 2x2 tensor named
    /// `unet.encoder.block0.weight` — note the missing `.conv.` infix,
    /// so not even the U-Net walker recognizes it, and there is no GRU
    /// and no head).
    ///
    /// The explicit optional-CNN route skips the unrecognized tensor and must
    /// still fail on the first required learned stage, the bidirectional GRU.
    #[test]
    fn extract_real_refuses_gguf_missing_required_tensors() {
        let bytes = minimal_valid_rmvpe_gguf();
        let tmp = std::env::temp_dir().join(format!(
            "vokra-rmvpe-real-refuse-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        std::fs::write(&tmp, &bytes).expect("write");

        let m = RMVPE::from_gguf_with_cnn_policy(&tmp, CnnChainPolicy::Optional)
            .expect("load structural fixture");
        let pcm = vec![0.0f32; 16_000];
        let Err(VokraError::ModelLoad(msg)) = m.extract_real(&pcm, 16_000) else {
            panic!("extract_real must refuse a GGUF without a walkable forward");
        };
        assert!(
            msg.contains("gru.weight_ih_l0 missing"),
            "expected the missing-GRU error from the first required learned stage, got {msg}"
        );
        std::fs::remove_file(&tmp).ok();
    }

    /// Positive smoke: `extract_real` returns real per-frame F0 rows
    /// with a self-consistent synthetic GGUF (no CNN, mel goes
    /// straight into the BiGRU + head + sigmoid + decoder chain). The
    /// smoke test asserts:
    ///
    /// - Frame count matches the [`RMVPE::frame_times`] contract
    ///   (`pcm.len() / hop`).
    /// - Every `hz` / `confidence` value is finite (no NaN / Inf).
    /// - Every `confidence` value lies in `[0, 1]` (sigmoid range).
    /// - Every `voiced` frame's `hz` lies in the `[fmin, fmax]` band.
    ///
    /// This is the "real forward runs cleanly" pin, and nothing more:
    /// the fixture carries no U-Net, so it must opt into
    /// [`CnnChainPolicy::Optional`] and what it exercises is
    /// mel → collapse → BiGRU → head → sigmoid → decode. Bit-exact
    /// parity against upstream RMVPE is env-gated on
    /// `VOKRA_RMVPE_REAL_GGUF` / `VOKRA_RMVPE_REAL_HIDDEN` — see
    /// `crates/vokra-models/tests/parity_rmvpe.rs`.
    #[test]
    fn extract_real_returns_real_frames_with_synthetic_weights() {
        // 128 mels (RMVPE default), BiGRU hidden = 8 (small so the
        // synthetic weights stay trivially small), head input = 16.
        let bytes = smoke_no_cnn_gguf(/*n_mels=*/ 128, /*gru_hidden=*/ 8);
        let tmp = std::env::temp_dir().join(format!(
            "vokra-rmvpe-smoke-real-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        std::fs::write(&tmp, &bytes).expect("write");

        let m = RMVPE::from_gguf_with_cnn_policy(&tmp, CnnChainPolicy::Optional)
            .expect("load smoke gguf");
        // 1 s of 440 Hz sine at 16 kHz — real signal, not zeros, so a
        // deterministic mel front-end propagates real energy through
        // the forward chain.
        let sr = m.config().sample_rate as f32;
        let pcm: Vec<f32> = (0..m.config().sample_rate as usize)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / sr).sin())
            .collect();

        let frames = m.extract_real(&pcm, 16_000).expect("real forward must run");

        let hop = m.config().hop as usize;
        assert_eq!(
            frames.len(),
            pcm.len() / hop,
            "extract_real must honor the frame_times() frame-count contract"
        );
        for (i, f) in frames.iter().enumerate() {
            assert!(f.hz.is_finite(), "frame {i}: hz {} is not finite", f.hz);
            assert!(
                f.confidence.is_finite(),
                "frame {i}: confidence {} is not finite",
                f.confidence
            );
            assert!(
                (0.0..=1.0).contains(&f.confidence),
                "frame {i}: confidence {} outside sigmoid range [0, 1]",
                f.confidence
            );
            if f.voiced {
                let cfg = m.config();
                assert!(
                    f.hz >= cfg.fmin && f.hz <= cfg.fmax,
                    "frame {i}: voiced hz {} outside [{}, {}]",
                    f.hz,
                    cfg.fmin,
                    cfg.fmax
                );
            }
        }
        std::fs::remove_file(&tmp).ok();
    }

    /// Positive smoke: `forward_from_hidden` bypasses the mel + CNN
    /// and runs the BiGRU + head + sigmoid + decoder chain on a
    /// caller-supplied hidden buffer. Shape / finite / sigmoid-range
    /// contract must hold — same set of pins as `extract_real`.
    ///
    /// This mirrors the env-gated `VOKRA_RMVPE_REAL_HIDDEN` parity
    /// path (`crates/vokra-models/tests/parity_rmvpe.rs`); the smoke
    /// version substitutes a deterministic `sin`-driven hidden buffer
    /// so the mechanism is unit-testable without an owner-side dumper.
    #[test]
    fn forward_from_hidden_returns_real_frames_with_synthetic_weights() {
        let feature_dim = 128usize;
        let bytes = smoke_no_cnn_gguf(feature_dim as u32, /*gru_hidden=*/ 8);
        let tmp = std::env::temp_dir().join(format!(
            "vokra-rmvpe-hidden-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        std::fs::write(&tmp, &bytes).expect("write");

        let m = RMVPE::from_gguf_with_cnn_policy(&tmp, CnnChainPolicy::Optional)
            .expect("load structural fixture");
        // 100 frames of a deterministic-but-non-trivial hidden buffer
        // — mixing sin + a slow trend so the classifier lands on
        // varying argmax positions across frames.
        let n_frames = 100usize;
        let mut hidden = vec![0.0f32; n_frames * feature_dim];
        for t in 0..n_frames {
            for f in 0..feature_dim {
                hidden[t * feature_dim + f] =
                    ((t as f32 * 0.05 + f as f32 * 0.02).sin() + t as f32 * 0.001).tanh();
            }
        }
        let frames = m
            .forward_from_hidden(&hidden, n_frames, feature_dim, 16_000)
            .expect("hidden-driven forward must run");
        assert_eq!(frames.len(), n_frames);
        for (i, f) in frames.iter().enumerate() {
            assert!(f.hz.is_finite(), "frame {i}: hz {} is not finite", f.hz);
            assert!(f.confidence.is_finite());
            assert!((0.0..=1.0).contains(&f.confidence));
            if f.voiced {
                let cfg = m.config();
                assert!(f.hz >= cfg.fmin && f.hz <= cfg.fmax);
            }
        }
        std::fs::remove_file(&tmp).ok();
    }

    /// FR-EX-08 loud-error contract on `forward_from_hidden`: a
    /// mis-sized `hidden` buffer must refuse loudly (`ModelLoad`) —
    /// silently truncating or padding would surface as a wrong-frame
    /// F0 track downstream.
    #[test]
    fn forward_from_hidden_refuses_wrong_length() {
        let feature_dim = 128usize;
        let bytes = smoke_no_cnn_gguf(feature_dim as u32, 8);
        let tmp = std::env::temp_dir().join(format!(
            "vokra-rmvpe-hidden-refuse-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        std::fs::write(&tmp, &bytes).expect("write");

        let m = RMVPE::from_gguf_with_cnn_policy(&tmp, CnnChainPolicy::Optional)
            .expect("load structural fixture");
        // Buffer length declares 10 frames but supplies 9 * 128 = 1152.
        let hidden = vec![0.0f32; 9 * feature_dim];
        let err = m
            .forward_from_hidden(&hidden, 10, feature_dim, 16_000)
            .expect_err("wrong-length hidden must refuse");
        assert!(
            matches!(err, VokraError::ModelLoad(ref msg) if msg.contains("hidden len")),
            "expected ModelLoad naming the mismatched hidden length, got {err:?}"
        );
        std::fs::remove_file(&tmp).ok();
    }

    /// Builds a **self-consistent** smoke GGUF: no CNN weights (so the
    /// forward skips the encoder / decoder chain and hands mel or the
    /// caller-supplied hidden buffer straight into the BiGRU), BiGRU
    /// with `input = n_mels` and `hidden = gru_hidden`, and a plain
    /// Linear head with `[n_class = 360, in = 2 * gru_hidden]`.
    ///
    /// All weight values are small deterministic fills — this is a
    /// **structural** smoke fixture (validates that the forward runs
    /// cleanly, produces finite outputs, and honors the sigmoid range),
    /// NOT a numeric parity fixture. Real parity is env-gated on
    /// `VOKRA_RMVPE_REAL_GGUF` / `VOKRA_RMVPE_REAL_HIDDEN`.
    fn smoke_no_cnn_gguf(n_mels: u32, gru_hidden: usize) -> Vec<u8> {
        smoke_gguf_builder(n_mels, n_mels as usize, gru_hidden)
            .to_bytes()
            .expect("serialize smoke gguf")
    }

    /// Small deterministic f32 fill, little-endian, for fixture
    /// tensors. Magnitudes stay near 0.05 so a sigmoid downstream lands
    /// inside `(0, 1)` rather than saturating.
    fn f32_small(elems: usize, seed_offset: f32) -> Vec<u8> {
        (0..elems)
            .flat_map(|i| {
                let v = ((i as f32) * 0.001 + seed_offset).sin() * 0.05;
                v.to_le_bytes()
            })
            .collect()
    }

    /// Shared fixture spine: RMVPE metadata + a bidirectional GRU
    /// (`gru_input` → `gru_hidden`) + a `[360, 2 * gru_hidden]` Linear
    /// head. Carries **no** CNN tensors; callers add whatever U-Net
    /// naming convention they are exercising.
    ///
    /// `gru_input` is separate from `n_mels` because it must equal the
    /// *post-CNN* feature width: `n_mels` for a CNN-less fixture,
    /// `C_last * W_last` when a U-Net chain is present.
    fn smoke_gguf_builder(n_mels: u32, gru_input: usize, gru_hidden: usize) -> GgufBuilder {
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "rmvpe");
        // Pin the mel / hop axes to the primary-source defaults so
        // `RmvpeConfig::from_gguf` yields exactly the shape the forward
        // expects.
        b.add_u32(GGUF_KEY_N_MELS, n_mels);
        b.add_u32(GGUF_KEY_HOP, DEFAULT_HOP);
        b.add_u32(GGUF_KEY_SAMPLE_RATE, DEFAULT_SAMPLE_RATE);
        b.add_u32(GGUF_KEY_N_FFT, DEFAULT_N_FFT);
        b.add_u32(GGUF_KEY_WIN_LENGTH, DEFAULT_WIN_LENGTH);
        b.add_u32(GGUF_KEY_N_CLASS, DEFAULT_N_CLASS);
        b.add_f32(GGUF_KEY_FMIN, DEFAULT_FMIN);
        b.add_f32(GGUF_KEY_FMAX, DEFAULT_FMAX);
        b.add_f32(GGUF_KEY_CENTS_PER_CLASS, DEFAULT_CENTS_PER_CLASS);
        b.add_f32(GGUF_KEY_BASE_HZ, DEFAULT_BASE_HZ);
        b.add_string(GGUF_KEY_UPSTREAM_REVISION, native::UPSTREAM_REVISION);
        b.add_string("vokra.provenance.source", "yxlllc/RMVPE");
        b.add_string("vokra.provenance.license", "unknown");
        b.add_string("vokra.provenance.weight_license", "unknown");

        // GRU weights: bidirectional single-layer, input = gru_input,
        // hidden = gru_hidden.
        let input_size = gru_input;
        let three_h = 3 * gru_hidden;
        for (suffix, seed) in [("l0", 0.10f32), ("l0_reverse", 0.20)] {
            b.add_tensor(
                &format!("gru.weight_ih_{suffix}"),
                GgmlType::F32,
                vec![three_h as u64, input_size as u64],
                f32_small(three_h * input_size, seed + 0.01),
            )
            .expect("add ih");
            b.add_tensor(
                &format!("gru.weight_hh_{suffix}"),
                GgmlType::F32,
                vec![three_h as u64, gru_hidden as u64],
                f32_small(three_h * gru_hidden, seed + 0.02),
            )
            .expect("add hh");
            b.add_tensor(
                &format!("gru.bias_ih_{suffix}"),
                GgmlType::F32,
                vec![three_h as u64],
                f32_small(three_h, seed + 0.03),
            )
            .expect("add b_ih");
            b.add_tensor(
                &format!("gru.bias_hh_{suffix}"),
                GgmlType::F32,
                vec![three_h as u64],
                f32_small(three_h, seed + 0.04),
            )
            .expect("add b_hh");
        }

        // Head: Linear [n_class=360, in=2*gru_hidden].
        let n_class = DEFAULT_N_CLASS as usize;
        let head_in = 2 * gru_hidden;
        b.add_tensor(
            "head.weight",
            GgmlType::F32,
            vec![n_class as u64, head_in as u64],
            f32_small(n_class * head_in, 0.30),
        )
        .expect("add head.weight");
        b.add_tensor(
            "head.bias",
            GgmlType::F32,
            vec![n_class as u64],
            f32_small(n_class, 0.40),
        )
        .expect("add head.bias");

        b
    }

    /// A GGUF that looks like an RMVPE fork: its U-Net is flattened
    /// under `cnn.*` (one of `REQUIRED_TENSOR_PREFIXES`, so the loader
    /// accepts it) and carries **nothing** under
    /// `unet.encoder.block{i}.conv.weight`, so the U-Net walker sees
    /// zero blocks.
    ///
    /// Critically, the GRU is sized for the *bypass* width
    /// (`input = n_mels`), which is what makes this a regression
    /// fixture rather than a shape-error fixture: before the
    /// zero-block gate landed, this artifact ran to completion and
    /// returned a full-length, finite, in-band pitch track produced by
    /// a model with its entire CNN missing. Every structural assertion
    /// in this module passed on that track.
    fn fork_prefix_cnn_gguf(n_mels: u32, gru_hidden: usize) -> Vec<u8> {
        let mut b = smoke_gguf_builder(n_mels, n_mels as usize, gru_hidden);
        // Fork convention: `cnn.layers.{i}.conv.weight` — a plausible
        // flattening, and none of it matches the scheme the forward
        // walks.
        b.add_tensor(
            "cnn.layers.0.conv.weight",
            GgmlType::F32,
            vec![16, 1, 3, 3],
            f32_small(16 * 9, 0.50),
        )
        .expect("add fork conv");
        b.add_tensor(
            "cnn.layers.0.bn.weight",
            GgmlType::F32,
            vec![16],
            f32_small(16, 0.60),
        )
        .expect("add fork bn");
        b.to_bytes().expect("serialize fork gguf")
    }

    /// A GGUF carrying a real one-block U-Net under the exact upstream
    /// scheme, sized so the chain composes end to end at the RMVPE mel
    /// axes:
    ///
    /// ```text
    /// mel [1, 1, 101, 128]
    ///   encoder.block0 Conv2d [2, 1, 3, 3] pad=same -> [2, 101, 128]
    ///                  MaxPool2d(2,2)               -> [2,  50,  64]
    ///   decoder.block0 ConvT2d [2, 2, 2, 2] stride=2 -> [2, 100, 128]
    ///   collapse [H, C*W]                            -> [100, 256]
    ///   BiGRU(256 -> 4) -> [100, 8] -> head [360, 8] -> 100 frames
    /// ```
    ///
    /// 100 frames is exactly the `frame_times` contract for 1 s at
    /// 16 kHz with hop 160, so the frame-count pin stays meaningful.
    fn one_block_unet_gguf() -> Vec<u8> {
        // C_last = 2, W_last = 128 => post-CNN feature width 256.
        let mut b = smoke_gguf_builder(128, 256, /*gru_hidden=*/ 4);
        b.add_tensor(
            "unet.encoder.block0.conv.weight",
            GgmlType::F32,
            vec![2, 1, 3, 3],
            f32_small(2 * 9, 0.70),
        )
        .expect("add encoder conv");
        b.add_tensor(
            "unet.encoder.block0.conv.bias",
            GgmlType::F32,
            vec![2],
            f32_small(2, 0.71),
        )
        .expect("add encoder bias");
        // ConvTranspose2d weight layout is [in, out, kh, kw].
        b.add_tensor(
            "unet.decoder.block0.conv.weight",
            GgmlType::F32,
            vec![2, 2, 2, 2],
            f32_small(2 * 2 * 2 * 2, 0.80),
        )
        .expect("add decoder conv");
        b.add_tensor(
            "unet.decoder.block0.conv.bias",
            GgmlType::F32,
            vec![2],
            f32_small(2, 0.81),
        )
        .expect("add decoder bias");
        b.to_bytes().expect("serialize unet gguf")
    }

    /// Writes `bytes` to a uniquely named temp GGUF and returns the
    /// path. Collapses the `process::id()` + nanos boilerplate every
    /// fixture test was repeating.
    fn write_temp_gguf(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
        let tmp = std::env::temp_dir().join(format!(
            "vokra-rmvpe-{tag}-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        std::fs::write(&tmp, bytes).expect("write temp gguf");
        tmp
    }

    /// **The regression this whole gate exists for.** A fork-convention
    /// checkpoint binds cleanly (its `cnn.*` tensors match a recognized
    /// prefix) but exposes no U-Net under the scheme the forward walks.
    ///
    /// Before the zero-block gate, `extract_real` skipped the entire
    /// U-Net, pushed the raw mel plane into the BiGRU, and returned 100
    /// finite in-band frames — a pitch track that no downstream
    /// consumer, and no assertion in this file, could tell apart from a
    /// real measurement.
    ///
    /// The error must name the scheme that was searched, because
    /// "ModelLoad happened" does not tell the owner of a fork
    /// checkpoint what to re-convert.
    #[test]
    fn extract_real_refuses_fork_prefix_checkpoint_naming_the_scheme() {
        let tmp = write_temp_gguf("fork-prefix", &fork_prefix_cnn_gguf(128, 8));

        let Err(VokraError::ModelLoad(msg)) = RMVPE::from_gguf(&tmp) else {
            panic!("the production loader must reject a non-public RMVPE topology");
        };
        assert!(
            msg.contains("public-contract tensor") || msg.contains("tensor manifest"),
            "the error must identify the strict public contract, got {msg}"
        );
        std::fs::remove_file(&tmp).ok();
    }

    /// The CNN-less path is reachable **only** by naming it. Same
    /// artifact, two policies, two outcomes — which is the property
    /// that makes "no blocks found" unusable as an implicit fallback.
    #[test]
    fn cnn_less_path_requires_an_explicit_opt_in() {
        let tmp = write_temp_gguf("policy-optin", &fork_prefix_cnn_gguf(128, 8));

        let sr = DEFAULT_SAMPLE_RATE as f32;
        let pcm: Vec<f32> = (0..DEFAULT_SAMPLE_RATE as usize)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / sr).sin())
            .collect();

        assert!(
            RMVPE::from_gguf(&tmp).is_err(),
            "the default loader must reject a checkpoint outside the exact E2E0 manifest"
        );

        let lenient = RMVPE::from_gguf_with_cnn_policy(&tmp, CnnChainPolicy::Optional)
            .expect("load with opt-in");
        assert_eq!(lenient.cnn_policy(), CnnChainPolicy::Optional);
        let frames = lenient
            .extract_real(&pcm, 16_000)
            .expect("the opt-in path must still run");
        assert_eq!(
            frames.len(),
            pcm.len() / (lenient.config().hop as usize),
            "the opt-in path keeps the frame_times contract"
        );
        std::fs::remove_file(&tmp).ok();
    }

    /// The historical one-block structural fixture remains runnable only
    /// through the explicit optional policy. It is not the fixed E2E0 public
    /// contract and therefore must never pass the production loader.
    ///
    /// `encoder_block_count` / `decoder_block_count` are what separate
    /// "the U-Net ran" from "the U-Net was skipped" — frame count,
    /// finiteness and sigmoid range hold either way, which is precisely
    /// why they were not enough on their own.
    #[test]
    fn extract_real_runs_a_legacy_discoverable_unet_only_when_opted_in() {
        let tmp = write_temp_gguf("unet-one-block", &one_block_unet_gguf());

        assert!(
            RMVPE::from_gguf(&tmp).is_err(),
            "the production loader must reject a reduced one-block topology"
        );
        let m = RMVPE::from_gguf_with_cnn_policy(&tmp, CnnChainPolicy::Optional)
            .expect("load explicitly opted-in one-block structural fixture");
        assert_eq!(m.cnn_policy(), CnnChainPolicy::Optional);
        assert_eq!(m.encoder_block_count(), 1, "encoder block0 must be found");
        assert_eq!(m.decoder_block_count(), 1, "decoder block0 must be found");

        let sr = m.config().sample_rate as f32;
        let pcm: Vec<f32> = (0..m.config().sample_rate as usize)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / sr).sin())
            .collect();

        let frames = m
            .extract_real(&pcm, 16_000)
            .expect("a discoverable U-Net must run, not trip the zero-block gate");
        assert_eq!(
            frames.len(),
            pcm.len() / (m.config().hop as usize),
            "extract_real must honor the frame_times() frame-count contract"
        );
        // These checks catch non-finite output and invalid probabilities. They
        // are structural only; no reference is involved in this fixture.
        for (i, f) in frames.iter().enumerate() {
            assert!(f.hz.is_finite(), "frame {i}: hz {} is not finite", f.hz);
            assert!(
                (0.0..=1.0).contains(&f.confidence),
                "frame {i}: confidence {} is NaN or outside [0, 1]",
                f.confidence
            );
        }
        std::fs::remove_file(&tmp).ok();
    }

    /// An encoder-only historical topology is incomplete and the strict loader
    /// rejects it during the complete manifest check, before any forward can
    /// silently drop the decoder.
    #[test]
    fn production_loader_refuses_a_unet_with_no_decoder() {
        // Encoder halves both axes, so the post-CNN width is
        // C_last(2) * W_last(64) = 128 — which happens to equal n_mels,
        // so the GRU and head compose and the forward reaches the end
        // with the whole up path dropped. The only trace left is a
        // halved frame count (50 where frame_times promises 100), which
        // no error surfaces and no caller is obliged to check.
        let mut b = smoke_gguf_builder(128, 128, /*gru_hidden=*/ 4);
        b.add_tensor(
            "unet.encoder.block0.conv.weight",
            GgmlType::F32,
            vec![2, 1, 3, 3],
            f32_small(2 * 9, 0.70),
        )
        .expect("add encoder conv");
        let bytes = b.to_bytes().expect("serialize");
        let tmp = write_temp_gguf("unet-no-decoder", &bytes);

        let Err(VokraError::ModelLoad(msg)) = RMVPE::from_gguf(&tmp) else {
            panic!("an incomplete E2E0 manifest must be refused during load");
        };
        assert!(
            msg.contains("public-contract tensor") || msg.contains("tensor manifest"),
            "the error must identify the strict public contract, got {msg}"
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

        // Case: spike at class 0 -> base_hz. Upstream does not clamp voiced
        // estimates to the configured fmin/fmax display range.
        let mut probs = vec![0.0f32; cfg.n_class as usize];
        probs[0] = 1.0;
        let (hz, voiced, conf) = decode_class_to_hz(&probs, &cfg, 0.03);
        assert!(voiced, "peak > threshold must be voiced");
        assert!(conf >= 0.03);
        assert!(
            (hz - cfg.base_hz).abs() < 0.5,
            "class-0 Hz must equal base_hz, got {hz}"
        );

        // Case: spike at class 60 → base_hz * 2^1 (1 octave up).
        let mut probs = vec![0.0f32; cfg.n_class as usize];
        probs[60] = 1.0;
        let (hz, voiced, _) = decode_class_to_hz(&probs, &cfg, 0.03);
        assert!(voiced);
        let expected = cfg.base_hz * 2.0;
        // Local centroid over the nine-bin window of a single-spike
        // probability collapses to `argmax` (weight sum = w_argmax), so
        // the analytic value is exact.
        assert!(
            (hz - expected).abs() < 0.5,
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

        let m = RMVPE::from_gguf_with_cnn_policy(&tmp, CnnChainPolicy::Optional)
            .expect("load structural fixture");
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
        // `vokra_core::stamp_provenance(&mut b, LicenseClass::Unknown,
        // "unknown", ...)`. `RmvpeWeights::from_gguf` does not read the
        // provenance chunk (that is `crates/vokra-core/src/compliance`'s
        // job), but the test below asserts the round-trip is intact so a
        // future runtime-side compliance gate can rely on the same
        // stamped values.
        b.add_string("vokra.provenance.weight_license", "unknown");
        b.add_string("vokra.provenance.license", "unknown");
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

        let m = RMVPE::from_gguf_with_cnn_policy(&tmp, CnnChainPolicy::Optional)
            .expect("legacy-name structural GGUF must load");
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
            Some("unknown"),
            "license_class round-trip: fail-closed `unknown` must survive the GGUF"
        );
        assert_eq!(
            file.get("vokra.provenance.license")
                .and_then(|v| v.as_str()),
            Some("unknown"),
            "license round-trip: fail-closed `unknown` must survive the GGUF"
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

        let m = RMVPE::from_gguf_with_cnn_policy(&tmp, CnnChainPolicy::Optional)
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
        assert!(validate_tensor_shape("unet.encoder.layers.0.conv.0.conv.1.weight", &[16]).is_ok());
        assert!(validate_tensor_shape("unet.decoder.layers.0.conv1.1.weight", &[256]).is_ok());
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
