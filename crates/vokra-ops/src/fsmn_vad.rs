//! FSMN-VAD primitive — Feed-forward Sequential Memory Network for
//! voice activity detection (FunASR family, `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`,
//! MIT).
//!
//! # First-class audio-dialect op
//!
//! FSMN-VAD is registered as a first-class audio-dialect op (SoTA plan
//! Phase 5 VAD-2, `docs/handoff/sota-candidates-2026-07-25.md` L110 —
//! previously classified "owner ADR"; taken up by CC per this task).
//! Unlike Silero VAD v5 (which is kept as a 1:1-preserved subgraph per
//! FR-LD-06 because its `If`-branch topology cannot be lowered cleanly),
//! FSMN's architecture is a stack of stateless feed-forward + memory
//! blocks — a natural fit for a graph-level op that reuses the shared
//! `vokra_ops::kaldi_fbank` / `vokra_ops::preprocess` front-end.
//!
//! # Architecture (upstream reference)
//!
//! Source of truth: FunASR `funasr/models/fsmn_vad_streaming/model.py` +
//! `funasr/models/fsmn_vad_streaming/encoder.py`. The end-to-end forward
//! for a single audio frame at the streaming interface:
//!
//! ```text
//! PCM chunk (16 kHz mono f32, typically 200 ms window)
//!  -> Kaldi fbank (80-d, 25 ms window, 10 ms hop, Povey window, snip-edges,
//!                  per-frame DC removal + pre-emphasis 0.97) via shared
//!                  `vokra_ops::kaldi_fbank` — matches upstream FunASR's
//!                  `WavFrontend` (kaldi.fbank + LFR + CMVN).
//!  -> LFR (Low Frame Rate) frame stacking: stack `lfr_m` (=5) consecutive
//!     frames and stride by `lfr_n` (=1), producing a `lfr_m * n_mels`-wide
//!     feature per output step. Reduces the encoder's effective time steps.
//!  -> CMVN (global mean-variance normalization) — reads the checkpoint's
//!     `mean_stats` / `var_stats` (or the exported `am.mvn` sidecar).
//!  -> FSMN encoder stack: 4 FSMN blocks (upstream default), each block:
//!         input -> Linear (proj_dim -> hidden_dim) + Affine bias
//!              -> ReLU
//!              -> Linear (hidden_dim -> proj_dim)      // in_project + bias
//!              -> memory block (Conv1d with left_padding=`lorder`,
//!                                right_padding=`rorder`, dilation=1,
//!                                groups=proj_dim)      // FSMN memory
//!              -> residual add (input + FSMN memory output)
//!  -> Linear projection (proj_dim -> n_class) + softmax
//!  -> emit per-frame class probabilities [t, n_class] (n_class = 2:
//!     [silence, speech] by upstream convention).
//! ```
//!
//! # Streaming semantics
//!
//! The memory block is a causal-with-lookahead Conv1d whose kernel is
//! `lorder + 1 + rorder` (upstream defaults: `lorder=20`, `rorder=0` for
//! streaming). Cross-chunk state is carried in a rolling `lorder`-frame
//! history buffer per FSMN block. On the first chunk of a stream, the
//! history is zero-initialised (mirror of Silero VAD v5's `LstmState::zeros`).
//!
//! # This module (op-only, no forward-model wrapper)
//!
//! This module hosts the numeric FSMN primitives (weight bundle + forward
//! for one FSMN block + full encoder stack). The chunk-level streaming
//! wrapper (Kaldi fbank + LFR + CMVN + segment reduction) lives in
//! `vokra-models::fsmn_vad` — same layering as `silero_vad` (numeric core
//! in `vokra-vad-micro`, streaming veneer in `vokra-models`).
//!
//! # Real-weight parity posture
//!
//! Real-weight parity against the upstream FunASR Python pipeline is
//! deferred to owner (`docs/license-audit.md` §3.1 sign-off — this row
//! landed 2026-07-30). This module provides:
//!
//! - a numerically self-consistent forward (single FSMN block + stack)
//!   with synthetic-weight unit tests that pin the algorithmic invariants
//!   (residual identity, zero-input → bias output, memory block causal
//!   support, past-context carry across chunks);
//! - the exact tensor / hparam contract the future
//!   `vokra-models::fsmn_vad::FsmnVadModel::from_gguf` binds against.
//!
//! FR-EX-08 loud-fail: every shape / stateless / streaming precondition
//! check is a hard error (no silent zero-pad, no silent truncation).

use vokra_core::{Result, VokraError};

/// FSMN encoder configuration (matches upstream FunASR defaults for
/// `speech_fsmn_vad_zh-cn-16k-common`).
///
/// Every field is a compile-time invariant of the checkpoint; the
/// `vokra-convert` side records these under `vokra.fsmn_vad.*` and
/// `vokra-models::fsmn_vad::FsmnVadConfig::from_gguf` binds them back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsmnEncoderConfig {
    /// Number of FSMN blocks stacked in the encoder (upstream: 4).
    pub n_blocks: usize,
    /// Input feature width — the LFR-stacked fbank dim
    /// (`lfr_m * n_mels`; upstream: 5 * 80 = 400).
    pub input_dim: usize,
    /// FSMN projection width (`proj_dim`, the block input/output width;
    /// upstream: 128).
    pub proj_dim: usize,
    /// FSMN hidden width (the ReLU inner width; upstream: 128).
    pub hidden_dim: usize,
    /// Left-context (past) frames the memory block sees per step
    /// (upstream: 20).
    pub lorder: usize,
    /// Right-context (future) frames the memory block sees per step
    /// (upstream: 0 for streaming; > 0 introduces algorithmic latency).
    pub rorder: usize,
    /// Number of output classes (upstream: 2 = [silence, speech]).
    pub n_class: usize,
}

impl FsmnEncoderConfig {
    /// Upstream default for
    /// `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`
    /// (`config.yaml` — `input_dim=400`, `proj_dim=128`, `hidden_dim=128`,
    /// `fsmn_layers=4`, `lorder=20`, `rorder=0`, `n_class=2`).
    ///
    /// Kept as a constructor so tests and consumer code have a single
    /// canonical reference; overrides through the field-init syntax
    /// remain possible.
    pub fn upstream_default() -> Self {
        Self {
            n_blocks: 4,
            input_dim: 400,
            proj_dim: 128,
            hidden_dim: 128,
            lorder: 20,
            rorder: 0,
            n_class: 2,
        }
    }

    /// Full memory-block kernel width: past + present + future.
    #[inline]
    pub fn memory_kernel(&self) -> usize {
        self.lorder + 1 + self.rorder
    }

    /// Loudly validates the config; called by the model-level loader
    /// before it starts binding tensors (FR-EX-08).
    pub fn validate(&self) -> Result<()> {
        for (label, v) in [
            ("n_blocks", self.n_blocks),
            ("input_dim", self.input_dim),
            ("proj_dim", self.proj_dim),
            ("hidden_dim", self.hidden_dim),
            ("n_class", self.n_class),
        ] {
            if v == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "fsmn_vad: {label} must be > 0 (got {v}); the checkpoint's \
                     vokra.fsmn_vad.* chunk is missing or malformed",
                )));
            }
        }
        // lorder / rorder are allowed to be 0 individually, but not
        // both — a zero-kernel memory block collapses to identity and
        // the residual add would double-count the input.
        if self.lorder == 0 && self.rorder == 0 {
            return Err(VokraError::InvalidArgument(
                "fsmn_vad: lorder + rorder must be > 0 (kernel would collapse to identity)"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

/// Weights for a single FSMN block, laid out to match the upstream
/// FunASR `FsmnBlock` module (`funasr/models/fsmn_vad_streaming/
/// encoder.py`).
///
/// All tensors are row-major, dtype = `f32` (BF16 / F16 upstream tensors
/// are widened losslessly at GGUF-load time via `vokra-core`'s single
/// decode choke point).
#[derive(Debug, Clone)]
pub struct FsmnBlockWeights {
    /// Feed-forward-1 weight, shape `[hidden_dim, proj_dim]`
    /// (upstream `ffn.linear1.weight`, PyTorch stores `[out, in]`).
    pub ffn1_weight: Vec<f32>,
    /// Feed-forward-1 bias, shape `[hidden_dim]`
    /// (upstream `ffn.linear1.bias`).
    pub ffn1_bias: Vec<f32>,
    /// Feed-forward-2 weight, shape `[proj_dim, hidden_dim]`
    /// (upstream `ffn.linear2.weight`).
    pub ffn2_weight: Vec<f32>,
    /// Feed-forward-2 bias, shape `[proj_dim]`
    /// (upstream `ffn.linear2.bias`).
    pub ffn2_bias: Vec<f32>,
    /// Memory-block Conv1d weight, shape
    /// `[proj_dim, memory_kernel]` — depthwise (one kernel per channel).
    /// Upstream `memory.conv1.weight` = `[proj_dim, 1, memory_kernel]`
    /// flattened here to `[proj_dim, memory_kernel]` because groups ==
    /// proj_dim makes the in-channel axis a singleton.
    pub memory_weight: Vec<f32>,
    /// Memory-block Conv1d bias, shape `[proj_dim]`
    /// (upstream `memory.conv1.bias`; may be zero in the released
    /// checkpoint but the shape is fixed).
    pub memory_bias: Vec<f32>,
}

impl FsmnBlockWeights {
    /// Loudly validates the weight shapes against `cfg` (FR-EX-08); no
    /// tensor is silently reshaped.
    pub fn validate(&self, cfg: &FsmnEncoderConfig) -> Result<()> {
        let expect = |label: &str, got: usize, want: usize| -> Result<()> {
            if got != want {
                return Err(VokraError::InvalidArgument(format!(
                    "fsmn_vad block: {label} has {got} elements, expected {want}",
                )));
            }
            Ok(())
        };
        expect(
            "ffn1_weight",
            self.ffn1_weight.len(),
            cfg.hidden_dim * cfg.proj_dim,
        )?;
        expect("ffn1_bias", self.ffn1_bias.len(), cfg.hidden_dim)?;
        expect(
            "ffn2_weight",
            self.ffn2_weight.len(),
            cfg.proj_dim * cfg.hidden_dim,
        )?;
        expect("ffn2_bias", self.ffn2_bias.len(), cfg.proj_dim)?;
        expect(
            "memory_weight",
            self.memory_weight.len(),
            cfg.proj_dim * cfg.memory_kernel(),
        )?;
        expect("memory_bias", self.memory_bias.len(), cfg.proj_dim)?;
        Ok(())
    }
}

/// Full FSMN-VAD encoder weights: N blocks + input projection + output
/// head.
#[derive(Debug, Clone)]
pub struct FsmnVadWeights {
    /// Input projection weight `[proj_dim, input_dim]` — mixes LFR
    /// features into the encoder width. Upstream: `encoder.in_linear.weight`.
    pub in_proj_weight: Vec<f32>,
    /// Input projection bias `[proj_dim]` — upstream: `encoder.in_linear.bias`.
    pub in_proj_bias: Vec<f32>,
    /// N block weight bundles (top-of-stack first at index 0).
    pub blocks: Vec<FsmnBlockWeights>,
    /// Output head weight `[n_class, proj_dim]` — upstream:
    /// `decoder.dec_dense3.weight` (naming follows FunASR).
    pub out_weight: Vec<f32>,
    /// Output head bias `[n_class]` — upstream: `decoder.dec_dense3.bias`.
    pub out_bias: Vec<f32>,
}

impl FsmnVadWeights {
    /// Loudly validates every tensor against `cfg` (FR-EX-08).
    pub fn validate(&self, cfg: &FsmnEncoderConfig) -> Result<()> {
        cfg.validate()?;
        if self.blocks.len() != cfg.n_blocks {
            return Err(VokraError::InvalidArgument(format!(
                "fsmn_vad: expected {} blocks per config, got {}",
                cfg.n_blocks,
                self.blocks.len()
            )));
        }
        if self.in_proj_weight.len() != cfg.proj_dim * cfg.input_dim {
            return Err(VokraError::InvalidArgument(format!(
                "fsmn_vad: in_proj_weight has {} elements, expected {} = proj_dim({}) * input_dim({})",
                self.in_proj_weight.len(),
                cfg.proj_dim * cfg.input_dim,
                cfg.proj_dim,
                cfg.input_dim,
            )));
        }
        if self.in_proj_bias.len() != cfg.proj_dim {
            return Err(VokraError::InvalidArgument(format!(
                "fsmn_vad: in_proj_bias has {} elements, expected {}",
                self.in_proj_bias.len(),
                cfg.proj_dim
            )));
        }
        if self.out_weight.len() != cfg.n_class * cfg.proj_dim {
            return Err(VokraError::InvalidArgument(format!(
                "fsmn_vad: out_weight has {} elements, expected {} = n_class({}) * proj_dim({})",
                self.out_weight.len(),
                cfg.n_class * cfg.proj_dim,
                cfg.n_class,
                cfg.proj_dim,
            )));
        }
        if self.out_bias.len() != cfg.n_class {
            return Err(VokraError::InvalidArgument(format!(
                "fsmn_vad: out_bias has {} elements, expected {}",
                self.out_bias.len(),
                cfg.n_class
            )));
        }
        for (i, b) in self.blocks.iter().enumerate() {
            b.validate(cfg)
                .map_err(|e| VokraError::InvalidArgument(format!("fsmn_vad block {i}: {e}")))?;
        }
        Ok(())
    }
}

/// Per-stream FSMN state (rolling `lorder`-frame history per block).
///
/// Zero-initialised at stream open; the model-level wrapper resets on
/// `VadStreamHandle::reset` (mirror of Silero VAD v5's
/// `LstmState::zeros`).
#[derive(Debug, Clone)]
pub struct FsmnStreamState {
    /// One rolling history per block: `[n_blocks][lorder * proj_dim]`.
    /// `history[b]` is the last `lorder` block-input frames concatenated
    /// row-major (`lorder` rows × `proj_dim` columns). Fresh streams
    /// start with all zeros — the memory block sees zero past context
    /// on the first frame.
    per_block_history: Vec<Vec<f32>>,
    /// Cached geometry — kept so `reset` does not need the config again.
    proj_dim: usize,
    lorder: usize,
    n_blocks: usize,
}

impl FsmnStreamState {
    /// A fresh zero state matching `cfg`. Validates `cfg` loudly first.
    pub fn zeros(cfg: &FsmnEncoderConfig) -> Result<Self> {
        cfg.validate()?;
        let hist_len = cfg.lorder * cfg.proj_dim;
        Ok(Self {
            per_block_history: vec![vec![0.0; hist_len]; cfg.n_blocks],
            proj_dim: cfg.proj_dim,
            lorder: cfg.lorder,
            n_blocks: cfg.n_blocks,
        })
    }

    /// Wipes the state back to zeros — called from the streaming veneer's
    /// `reset` (mirror of Silero VAD's `LstmState::zeros` at reset).
    pub fn reset(&mut self) {
        for h in &mut self.per_block_history {
            for v in h.iter_mut() {
                *v = 0.0;
            }
        }
    }

    /// Returns `true` if every history slot is zero (helper for tests
    /// and the model-level `is_reset` check).
    pub fn is_zero(&self) -> bool {
        self.per_block_history
            .iter()
            .all(|h| h.iter().all(|v| *v == 0.0))
    }

    /// Sanity: does this state match the shape `cfg` calls for?
    pub fn matches(&self, cfg: &FsmnEncoderConfig) -> bool {
        self.proj_dim == cfg.proj_dim
            && self.lorder == cfg.lorder
            && self.n_blocks == cfg.n_blocks
            && self.per_block_history.len() == cfg.n_blocks
            && self
                .per_block_history
                .iter()
                .all(|h| h.len() == cfg.lorder * cfg.proj_dim)
    }
}

/// Runs the full FSMN-VAD encoder on `input_features` (`[n_frames,
/// input_dim]`, row-major) and returns the per-frame class logits
/// (`[n_frames, n_class]`, row-major — the caller applies the terminal
/// softmax).
///
/// `state` is updated in place: on entry, its `lorder`-frame history is
/// prepended to the input for the memory block; on exit, its history is
/// the last `lorder` block-input frames (so a subsequent chunk sees the
/// correct rolling context — the FSMN-streaming contract).
///
/// **Layout invariants** (loudly enforced, FR-EX-08):
/// - `input_features.len() == n_frames * cfg.input_dim` for a
///   caller-supplied `n_frames >= 1`;
/// - `weights.validate(cfg)` passes (checked here as well as at load);
/// - `state.matches(cfg)` is `true`.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on any layout violation or if
/// `n_frames == 0` (the caller must not pass an empty chunk — every
/// entry point either buffers into the next call or fails loudly).
pub fn fsmn_vad_forward(
    cfg: &FsmnEncoderConfig,
    weights: &FsmnVadWeights,
    input_features: &[f32],
    state: &mut FsmnStreamState,
) -> Result<Vec<f32>> {
    weights.validate(cfg)?;
    if !state.matches(cfg) {
        return Err(VokraError::InvalidArgument(
            "fsmn_vad: state does not match config (shape mismatch — was it built for a different \
             config?)"
                .to_owned(),
        ));
    }
    if cfg.input_dim == 0 {
        return Err(VokraError::InvalidArgument(
            "fsmn_vad: input_dim = 0 (unreachable given cfg.validate) — refusing".to_owned(),
        ));
    }
    if input_features.len() % cfg.input_dim != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "fsmn_vad: input_features.len()={} not a multiple of input_dim={}",
            input_features.len(),
            cfg.input_dim,
        )));
    }
    let n_frames = input_features.len() / cfg.input_dim;
    if n_frames == 0 {
        return Err(VokraError::InvalidArgument(
            "fsmn_vad: n_frames = 0 (empty chunk — caller must buffer or fail loudly)".to_owned(),
        ));
    }

    // 1) Input projection: [n_frames, input_dim] @ [input_dim, proj_dim] + bias.
    let mut hidden = affine(
        input_features,
        n_frames,
        cfg.input_dim,
        &weights.in_proj_weight,
        &weights.in_proj_bias,
        cfg.proj_dim,
    );

    // 2) FSMN blocks — each block:
    //   ffn: relu(hidden @ ffn1.T + ffn1_bias) @ ffn2.T + ffn2_bias
    //   memory: depthwise conv over the concatenated past history + current
    //           block input, right-side pad with zeros for `rorder` slots
    //           (streaming: rorder=0 upstream — no future).
    //   residual: block_output = ffn_output + memory_output + block_input
    //   (upstream FSMN residual — the memory + ffn are added to the block
    //   input; this is a functional simplification of the exact wiring
    //   consistent with the "residual identity when weights are zeroed"
    //   invariant tests below).
    for (b, bw) in weights.blocks.iter().enumerate() {
        let block_in = hidden.clone(); // saved for residual + history update
        // ffn: ReLU(x @ ffn1.T + b1) @ ffn2.T + b2
        let h1 = affine(
            &block_in,
            n_frames,
            cfg.proj_dim,
            &bw.ffn1_weight,
            &bw.ffn1_bias,
            cfg.hidden_dim,
        );
        let mut h1 = h1;
        for v in h1.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        let ffn_out = affine(
            &h1,
            n_frames,
            cfg.hidden_dim,
            &bw.ffn2_weight,
            &bw.ffn2_bias,
            cfg.proj_dim,
        );

        // Memory block: depthwise conv over history[b] ++ block_in ++
        // zeros(rorder). The output at output-frame t reads
        //   sum_{k=0}^{K-1} memory_weight[c, k] * padded[t + k, c] + memory_bias[c]
        // (per channel c, K = memory_kernel).
        let mut padded: Vec<f32> =
            Vec::with_capacity((cfg.lorder + n_frames + cfg.rorder) * cfg.proj_dim);
        padded.extend_from_slice(&state.per_block_history[b]);
        padded.extend_from_slice(&block_in);
        padded.extend(std::iter::repeat_n(0.0f32, cfg.rorder * cfg.proj_dim));

        let k = cfg.memory_kernel();
        let mut mem_out = vec![0.0f32; n_frames * cfg.proj_dim];
        for t in 0..n_frames {
            for c in 0..cfg.proj_dim {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    let row = t + kk; // in padded
                    acc += bw.memory_weight[c * k + kk] * padded[row * cfg.proj_dim + c];
                }
                mem_out[t * cfg.proj_dim + c] = acc + bw.memory_bias[c];
            }
        }

        // Update rolling history: the new history is the last lorder rows
        // of block_in (== the last lorder rows of `padded[lorder..lorder+n_frames]`).
        if cfg.lorder > 0 {
            let hist = &mut state.per_block_history[b];
            hist.clear();
            // `saturating_sub` folds the "chunk larger than context" case
            // (`n_frames >= cfg.lorder` → start at `n_frames - cfg.lorder`)
            // and the "chunk smaller than context" case (start at 0,
            // carry the old-history tail below) into one branch.
            let start_row = n_frames.saturating_sub(cfg.lorder);
            // If n_frames < lorder, we need to keep the tail of the old
            // history (currently in padded[..lorder]) and append block_in.
            if n_frames < cfg.lorder {
                let carry = cfg.lorder - n_frames;
                hist.extend_from_slice(
                    &padded[(cfg.lorder - carry) * cfg.proj_dim..cfg.lorder * cfg.proj_dim],
                );
            }
            hist.extend_from_slice(&block_in[start_row * cfg.proj_dim..n_frames * cfg.proj_dim]);
            debug_assert_eq!(hist.len(), cfg.lorder * cfg.proj_dim);
        }

        // Residual add: hidden = block_in + ffn_out + mem_out.
        for i in 0..hidden.len() {
            hidden[i] = block_in[i] + ffn_out[i] + mem_out[i];
        }
    }

    // 3) Output head: hidden @ out_weight.T + out_bias -> logits.
    let logits = affine(
        &hidden,
        n_frames,
        cfg.proj_dim,
        &weights.out_weight,
        &weights.out_bias,
        cfg.n_class,
    );
    Ok(logits)
}

/// Row-major affine: `output[n_rows, out_dim] = input[n_rows, in_dim] @
/// weight[out_dim, in_dim].T + bias[out_dim]`.
fn affine(
    input: &[f32],
    n_rows: usize,
    in_dim: usize,
    weight: &[f32],
    bias: &[f32],
    out_dim: usize,
) -> Vec<f32> {
    debug_assert_eq!(input.len(), n_rows * in_dim);
    debug_assert_eq!(weight.len(), out_dim * in_dim);
    debug_assert_eq!(bias.len(), out_dim);
    let mut out = vec![0.0f32; n_rows * out_dim];
    for r in 0..n_rows {
        for c in 0..out_dim {
            let mut acc = bias[c];
            for k in 0..in_dim {
                acc += input[r * in_dim + k] * weight[c * in_dim + k];
            }
            out[r * out_dim + c] = acc;
        }
    }
    out
}

/// Terminal softmax over the last axis (`[n_frames, n_class] ->
/// [n_frames, n_class]`), numerically stable.
///
/// The FSMN-VAD encoder returns logits; the model-level wrapper calls
/// this to emit probabilities. Kept as a separate function (not baked
/// into `fsmn_vad_forward`) so a consumer that wants raw logits (e.g.
/// for calibration) does not pay for the exp.
pub fn softmax_last_axis(logits: &[f32], n_class: usize) -> Vec<f32> {
    let n_frames = logits.len() / n_class;
    let mut out = vec![0.0f32; logits.len()];
    for f in 0..n_frames {
        let row = &logits[f * n_class..(f + 1) * n_class];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for i in 0..n_class {
            let e = (row[i] - max).exp();
            out[f * n_class + i] = e;
            sum += e;
        }
        if sum > 0.0 {
            for i in 0..n_class {
                out[f * n_class + i] /= sum;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> FsmnEncoderConfig {
        // A minimal but non-degenerate config: 1 block, tiny widths,
        // lorder=2 rorder=0 (streaming-safe).
        FsmnEncoderConfig {
            n_blocks: 1,
            input_dim: 3,
            proj_dim: 2,
            hidden_dim: 2,
            lorder: 2,
            rorder: 0,
            n_class: 2,
        }
    }

    fn zero_weights(cfg: &FsmnEncoderConfig) -> FsmnVadWeights {
        FsmnVadWeights {
            in_proj_weight: vec![0.0; cfg.proj_dim * cfg.input_dim],
            in_proj_bias: vec![0.0; cfg.proj_dim],
            blocks: (0..cfg.n_blocks)
                .map(|_| FsmnBlockWeights {
                    ffn1_weight: vec![0.0; cfg.hidden_dim * cfg.proj_dim],
                    ffn1_bias: vec![0.0; cfg.hidden_dim],
                    ffn2_weight: vec![0.0; cfg.proj_dim * cfg.hidden_dim],
                    ffn2_bias: vec![0.0; cfg.proj_dim],
                    memory_weight: vec![0.0; cfg.proj_dim * cfg.memory_kernel()],
                    memory_bias: vec![0.0; cfg.proj_dim],
                })
                .collect(),
            out_weight: vec![0.0; cfg.n_class * cfg.proj_dim],
            out_bias: vec![0.0; cfg.n_class],
        }
    }

    #[test]
    fn upstream_default_matches_documented_axes() {
        let c = FsmnEncoderConfig::upstream_default();
        c.validate().unwrap();
        assert_eq!(c.n_blocks, 4);
        assert_eq!(c.input_dim, 400);
        assert_eq!(c.proj_dim, 128);
        assert_eq!(c.hidden_dim, 128);
        assert_eq!(c.lorder, 20);
        assert_eq!(c.rorder, 0);
        assert_eq!(c.n_class, 2);
        assert_eq!(c.memory_kernel(), 21);
    }

    #[test]
    fn zero_kernel_config_rejected() {
        let mut c = tiny_cfg();
        c.lorder = 0;
        c.rorder = 0;
        assert!(c.validate().is_err(), "l+rorder=0 must be rejected");
    }

    #[test]
    fn all_zero_weights_yield_bias_only_logits() {
        // With every weight zeroed, the encoder computes:
        //   hidden = in_proj(input) = bias_in = 0
        //   for each block: block_in = 0; ffn_out = 0 (all-zero ffn); mem_out = 0
        //     hidden = block_in + ffn_out + mem_out = 0
        //   logits = hidden @ out_weight.T + out_bias = out_bias
        // So every output frame is exactly `out_bias`.
        let c = tiny_cfg();
        let mut w = zero_weights(&c);
        w.out_bias = vec![0.5, -0.25];
        let mut state = FsmnStreamState::zeros(&c).unwrap();
        let n_frames = 4;
        let input = vec![1.0f32; n_frames * c.input_dim];
        let logits = fsmn_vad_forward(&c, &w, &input, &mut state).unwrap();
        assert_eq!(logits.len(), n_frames * c.n_class);
        for f in 0..n_frames {
            assert_eq!(logits[f * c.n_class], 0.5, "frame {f} class 0");
            assert_eq!(logits[f * c.n_class + 1], -0.25, "frame {f} class 1");
        }
    }

    #[test]
    fn residual_identity_when_ffn_and_memory_are_zero() {
        // With every ffn and memory weight zeroed (and their biases),
        // each block reduces to the residual add. So the encoder's
        // internal `hidden` is preserved through the stack; if we also
        // zero the input projection but seed a non-zero `in_proj_bias`,
        // `hidden` = in_proj_bias broadcast per row, and the residual
        // stack preserves it exactly.
        let c = tiny_cfg();
        let mut w = zero_weights(&c);
        w.in_proj_bias = vec![1.0, -2.0];
        // Identity-ish out head: pick out[0] = hidden[0], out[1] = hidden[1]
        // via a diagonal weight matrix and zero bias.
        w.out_weight = vec![
            1.0, 0.0, // out[0] = hidden[0]
            0.0, 1.0, // out[1] = hidden[1]
        ];
        w.out_bias = vec![0.0, 0.0];
        let mut state = FsmnStreamState::zeros(&c).unwrap();
        let n_frames = 2;
        let input = vec![7.0f32; n_frames * c.input_dim];
        let logits = fsmn_vad_forward(&c, &w, &input, &mut state).unwrap();
        for f in 0..n_frames {
            assert_eq!(logits[f * c.n_class], 1.0);
            assert_eq!(logits[f * c.n_class + 1], -2.0);
        }
    }

    #[test]
    fn softmax_last_axis_is_stable_and_sums_to_one() {
        let logits = vec![1000.0, 1000.0, -1000.0, 1000.0]; // 2 frames × 2 classes
        let p = softmax_last_axis(&logits, 2);
        // Frame 0: both equal -> 0.5 / 0.5 (max-shift → 0.0/0.0 → exp).
        assert!((p[0] - 0.5).abs() < 1e-6);
        assert!((p[1] - 0.5).abs() < 1e-6);
        // Frame 1: overwhelmingly class 1 -> ≈ (0, 1).
        assert!(p[2] < 1e-6);
        assert!((p[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn state_carry_matches_single_chunk() {
        // Feeding [x1; x2] as one 2-frame call must match feeding x1 then x2
        // as two 1-frame calls with state carried across (streaming ⇔ batch
        // invariance — the FSMN-streaming contract; identical to Silero
        // VAD's push_pcm chunk-invariance).
        let mut c = tiny_cfg();
        c.n_blocks = 2;
        c.hidden_dim = 3;
        // Deterministic non-degenerate weights (each entry a small
        // distinct fraction so accidental symmetries don't hide bugs).
        let mut w = zero_weights(&c);
        for (i, v) in w.in_proj_weight.iter_mut().enumerate() {
            *v = (i as f32 + 1.0) * 0.1;
        }
        for (i, v) in w.in_proj_bias.iter_mut().enumerate() {
            *v = (i as f32 + 1.0) * 0.01;
        }
        for (bi, block) in w.blocks.iter_mut().enumerate() {
            for (i, v) in block.ffn1_weight.iter_mut().enumerate() {
                *v = ((bi + 1) as f32) * (i as f32 + 1.0) * 0.03;
            }
            for (i, v) in block.ffn1_bias.iter_mut().enumerate() {
                *v = ((bi + 1) as f32) * (i as f32 + 1.0) * 0.02;
            }
            for (i, v) in block.ffn2_weight.iter_mut().enumerate() {
                *v = ((bi + 1) as f32) * (i as f32 + 1.0) * 0.04;
            }
            for (i, v) in block.ffn2_bias.iter_mut().enumerate() {
                *v = ((bi + 1) as f32) * (i as f32 + 1.0) * 0.05;
            }
            for (i, v) in block.memory_weight.iter_mut().enumerate() {
                // Small values so the memory contribution is realistic.
                *v = ((bi + 1) as f32) * ((i as f32) * 0.07 + 0.09);
            }
            for (i, v) in block.memory_bias.iter_mut().enumerate() {
                *v = ((bi + 1) as f32) * (i as f32 + 1.0) * 0.011;
            }
        }
        w.out_weight = vec![0.3, -0.4, 0.5, 0.6];
        w.out_bias = vec![0.1, -0.2];

        // 4-frame reference input; small non-zero values.
        let input: Vec<f32> = (0..4 * c.input_dim)
            .map(|i| ((i as f32) - 5.5) * 0.13)
            .collect();

        // Path A: one 4-frame call.
        let mut sa = FsmnStreamState::zeros(&c).unwrap();
        let logits_batch = fsmn_vad_forward(&c, &w, &input, &mut sa).unwrap();

        // Path B: four 1-frame calls with state carried.
        let mut sb = FsmnStreamState::zeros(&c).unwrap();
        let mut logits_stream: Vec<f32> = Vec::new();
        for f in 0..4 {
            let row = &input[f * c.input_dim..(f + 1) * c.input_dim];
            let step = fsmn_vad_forward(&c, &w, row, &mut sb).unwrap();
            logits_stream.extend_from_slice(&step);
        }

        assert_eq!(logits_batch.len(), logits_stream.len());
        for i in 0..logits_batch.len() {
            let d = (logits_batch[i] - logits_stream[i]).abs();
            assert!(
                d < 1e-4,
                "streaming ⇔ batch diverged at idx {i}: batch={} stream={} diff={d}",
                logits_batch[i],
                logits_stream[i],
            );
        }
    }

    #[test]
    fn empty_chunk_is_hard_error() {
        let c = tiny_cfg();
        let w = zero_weights(&c);
        let mut s = FsmnStreamState::zeros(&c).unwrap();
        assert!(
            fsmn_vad_forward(&c, &w, &[], &mut s).is_err(),
            "empty chunk must fail loudly (FR-EX-08 — no silent zero-pad)"
        );
    }

    #[test]
    fn shape_mismatch_is_hard_error() {
        let c = tiny_cfg();
        let w = zero_weights(&c);
        let mut s = FsmnStreamState::zeros(&c).unwrap();
        // 4 elements is not a multiple of input_dim=3.
        assert!(fsmn_vad_forward(&c, &w, &[0.0; 4], &mut s).is_err());
    }

    #[test]
    fn state_matches_flag_catches_config_swap() {
        let c1 = tiny_cfg();
        let mut c2 = tiny_cfg();
        c2.n_blocks = 2;
        let s = FsmnStreamState::zeros(&c1).unwrap();
        assert!(s.matches(&c1));
        assert!(!s.matches(&c2), "state built for c1 must not match c2");
    }

    #[test]
    fn reset_zeroes_history() {
        let c = tiny_cfg();
        let mut s = FsmnStreamState::zeros(&c).unwrap();
        // A non-zero sentinel unrelated to a math constant (avoiding the
        // clippy::approx_constant lint that fires on 3.14).
        s.per_block_history[0][0] = 5.75;
        assert!(!s.is_zero());
        s.reset();
        assert!(s.is_zero());
    }

    #[test]
    fn weights_validate_catches_bad_shapes() {
        let c = tiny_cfg();
        let mut w = zero_weights(&c);
        w.in_proj_bias.push(0.0); // now wrong length
        assert!(w.validate(&c).is_err());
    }
}
