//! TEN-VAD (`TEN-framework/ten-vad`) — runtime binder for the `ten_vad`
//! converter arch (Wave B 2026-08-15, loud-partial per the nisqa / utmosv2 /
//! emotion2vec / panns / redimnet / wavlm / storm precedent — CLAUDE.md
//! 教訓 (a)「loud-partial は fake-complete より honest」).
//!
//! # The gap this closes
//!
//! `crates/vokra-convert/src/models/ten_vad.rs` (landed
//! coverage-audit-2026-08-03 Wave A permissive continuation, 2026-08-04)
//! produces a GGUF stamped `vokra.model.arch = "ten_vad"` that **nothing in
//! the workspace read back**, so every converted checkpoint was unloadable.
//! This module is that consumer: the arch tag now resolves to a real binder
//! with a real weight manifest, a real streaming frame accumulator and a real
//! [`VadEngine`] / [`VadStreamHandle`] surface, so a caller can swap
//! Silero / FSMN-VAD / TEN-VAD without rewriting call sites.
//!
//! # Primary sources
//!
//! - Upstream repository (the primary redistribution source — there is no HF
//!   mirror of the release): <https://github.com/TEN-framework/ten-vad>
//! - Upstream licence: <https://github.com/TEN-framework/ten-vad/blob/main/LICENSE>
//!   — **Apache-2.0** for the main project. The **LPCNet-derived DSP
//!   front-end** bundled in the upstream distribution is **BSD-3-Clause**;
//!   NOTICE attribution for the LPCNet copyright is required when
//!   redistributing runtime binaries that embed the front-end. Both facts are
//!   transcribed verbatim from the converter's module docstring (the
//!   cross-crate contract this binder mirrors) — see "Licensing" below.
//! - Offline ONNX bridge actually used for the published artefact:
//!   `tools/parity/onnx_to_safetensors.py` (generic ONNX-initializer →
//!   safetensors extractor). **No ONNX ever enters the runtime**
//!   (FR-LD-05 / NFR-DS-02).
//!
//! ## Sidecar-name discrepancy (recorded so the follow-up wave is not
//! ## sent hunting for a file that does not exist)
//!
//! The converter's module docstring names a per-model sidecar
//! `tools/parity/ten_vad_prepare_checkpoint.py`. **That file has never been
//! written.** The artefact was in fact produced through the *generic* bridge
//! [`PRIMARY_SOURCE_BRIDGE`] (`tools/parity/onnx_to_safetensors.py`), which
//! extracts float ONNX initializers and fail-closed-skips the integer
//! graph-metadata entries (shape / axes / steps). This binder therefore cites
//! the generic bridge — the tool that actually exists and actually produced
//! the weights — everywhere it points a reader at the recovery path. Same
//! class of finding as the `nisqa` binder's MISSING SIDECAR blocker.
//!
//! # Architecture (as far as the primary sources commit to it)
//!
//! ```text
//! PCM (mono f32, 16 kHz)                      ← the rate the upstream release targets
//!   -> LPCNet-derived DSP feature front-end        ← **loud-partial**
//!        (the upstream distribution bundles a
//!         BSD-3-Clause LPCNet-derived front-end;
//!         its exact feature stack — band count,
//!         pitch parameters, framing geometry — is
//!         NOT transcribable from the converter
//!         contract, which stamps no topology axes.)
//!   -> small recurrent backbone                    ← **loud-partial**
//!        (the converter docstring says "small
//!         LSTM/GRU backbone" — it deliberately does
//!         NOT commit to which. Picking one would be
//!         a coin flip whose wrong side is
//!         silent-wrong, not loud-wrong.)
//!   -> per-frame speech probability                ← the `VadStreamHandle` contract
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! **Real (this WP)** — everything that can be grounded in a primary source:
//!
//! - [`TenVad::from_gguf`] with **strict** `vokra.model.arch == "ten_vad"`
//!   verification. A sibling `vad-kws` GGUF handed here by mistake
//!   (`silero-vad` / `fsmn-vad` / `openwakeword` / `openwakeword_op`) fails
//!   with a specific mis-route [`VokraError::ModelLoad`] naming the expected
//!   and actual tags plus the whole sibling fleet, instead of a downstream
//!   missing-tensor error (FR-EX-08).
//! - [`TenVadWeights::from_gguf`]: a real tensor manifest walk with two loud
//!   gates — a non-empty gate (a zero-tensor GGUF is never a valid TEN-VAD
//!   checkpoint) and a converter-contract dtype gate (`convert_ten_vad_file`
//!   has **no quantization arm** — it passes F32 / F16 / BF16 through
//!   verbatim — so a K-quant tensor proves the GGUF did not come from that
//!   converter, the same class of signal as a wrong arch tag).
//! - [`TenVadWeights::require_tensor`]: the loud name-resolution primitive the
//!   follow-up real-forward wave binds against — a missing tensor is a
//!   [`VokraError::ModelLoad`] **naming the tensor** and previewing what the
//!   GGUF actually carries.
//! - [`TenVadWeights::bf16_count`]: the runtime-side mirror of the converter's
//!   `TenVadReport::bf16_passthrough` observability counter. BF16 widens to
//!   f32 losslessly at load through the single choke point
//!   `vokra-core/src/gguf/quant/mod.rs decode_bf16` (`bits << 16` is exact);
//!   a silent widen / downcast regression surfaces as this counter drifting.
//! - [`TenVadConfig`]: the **optional, all-or-nothing** `vokra.ten_vad.*`
//!   topology group. The converter does not stamp it today, so its absence is
//!   the normal case and is recorded as `None` rather than silently defaulted
//!   to invented axes; a *partially* stamped group is a loud
//!   [`VokraError::ModelLoad`] naming exactly which keys are missing.
//! - [`TenVadStream`]: a real hop-based frame accumulator with a loud sample-
//!   rate gate — a mismatched rate is a [`VokraError::InvalidArgument`],
//!   **never a silent resample** (FR-EX-08).
//! - Weight-license surfacing, fail-closed to [`LicenseClass::Unknown`] when
//!   the stamp is absent.
//!
//! **Loud-partial (this WP)** — [`TenVad::frame_probability`] and
//! [`VadStreamHandle::push_pcm`] return [`VokraError::UnsupportedOp`] naming
//! four concrete blockers, because a best-guess forward here would be
//! *silent*-wrong (a plausible-looking probability stream) rather than
//! loud-wrong:
//!
//! 1. **Missing tensor-name manifest.** The published artefact was produced by
//!    the generic `tools/parity/onnx_to_safetensors.py` bridge, which passes
//!    ONNX **initializer names through verbatim**. Nothing in this repository
//!    records what those names are, so no weight walk can map a tensor to a
//!    role (input-gate vs hidden-gate vs bias, per layer).
//! 2. **Missing topology axes.** The converter stamps `vokra.model.*` and
//!    `vokra.provenance.*` only — there is no `vokra.ten_vad.*` group — so the
//!    hop size, feature width, hidden width and layer count are all unknown at
//!    load time.
//! 3. **Unresolved backbone family.** The converter docstring says "small
//!    LSTM/GRU backbone" without committing to either. `vokra-ops` *does*
//!    carry a shape-generic GRU cell ([`vokra_ops::rnnoise::gru_forward`],
//!    TensorFlow `GRUCell` gate convention) that a real manifest could drive —
//!    so the blocker is the manifest and the axes, not the arithmetic.
//! 4. **Missing LPCNet-derived front-end.** `vokra-ops` carries RNNoise's
//!    *fixed* 22-band Bark filterbank ([`vokra_ops::rnnoise::bark_filterbank`],
//!    `BARK_BAND_EDGES` hard-coded for RNNoise's 481-bin RFFT space) — same
//!    Xiph lineage, different fixed table — but not the LPCNet-derived feature
//!    stack TEN-VAD's front-end implements, and the front-end spec is not
//!    transcribable from the converter contract.
//!
//! **No fabricated speech probabilities are ever emitted** (FR-EX-08 — no
//! silent partial output).
//!
//! # Sibling family distinctness (`category = "vad-kws"` neighbourhood)
//!
//! [`ARCH`] = `"ten_vad"` is **deliberately distinct** from every sibling
//! voice-activity / wake-word arch tag. All four siblings answer a related
//! question ("is there speech / my wake-word in this frame?") with completely
//! different topologies and load paths:
//!
//! - `silero-vad` — Silero VAD v5 / v6.2.1 as a **1:1-preserved dedicated
//!   subgraph** (FR-LD-06): a learned pseudo-STFT `Conv1d` front-end, a 4-conv
//!   encoder and an LSTM(128,128), with the forward living in the `#![no_std]`
//!   crate `vokra-vad-micro`. Two independently-trained rate branches
//!   (`sr8k.*` / `sr16k.*`) in one artefact.
//! - `fsmn-vad` — FunASR FSMN-VAD as a **first-class op stack**: Kaldi fbank →
//!   LFR frame stacking → global CMVN → feed-forward + memory blocks →
//!   2-class softmax, with CMVN stats carried in `vokra.fsmn_vad.cmvn_*`
//!   metadata chunks.
//! - `openwakeword` / `openwakeword_op` — keyword spotting, not VAD: a shared
//!   embedding extractor plus per-wake-word classifier MLPs, surfaced through
//!   [`vokra_core::engines::KwsEngine`] with `(wakeword_name, probability)`
//!   pairs rather than a single speech probability.
//!
//! Silently sharing an arch tag would let runtime dispatch mis-route a TEN-VAD
//! checkpoint onto a Silero subgraph loader (which would hunt for `sr16k.*`
//! tensors), an FSMN loader (which would demand a `vokra.fsmn_vad.*` chunk) or
//! a KWS loader (which would demand a wake-word name array) — three
//! downstream missing-key errors instead of one specific arch-mismatch
//! message. FR-EX-08 forbids the silent misroute.
//!
//! # Licensing
//!
//! - SPDX: **Apache-2.0** ([`LicenseClass::Permissive`]) — the upstream repo
//!   LICENSE, mirrored by the converter's [`DEFAULT_LICENSE_SPDX`].
//! - The bundled **LPCNet-derived DSP front-end is BSD-3-Clause**: NOTICE
//!   attribution for the LPCNet copyright is required when redistributing
//!   runtime binaries that embed the front-end. This obligation attaches to
//!   the *front-end port*, which is precisely the piece deferred here — so it
//!   becomes actionable in the follow-up wave, not this one.
//! - `docs/license-audit.md` §3.1 sign-off stays **BLANK** (owner-only per
//!   `[[feedback-license-signoff-primary-source]]` — CC does NOT sign).
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_URL`] /
//! [`DEFAULT_LICENSE_SPDX`] are a **mirror of the converter's** constants —
//! the same string-handshake rule the sibling binders (`nisqa` / `emotion2vec`
//! / `panns` / `redimnet` / `snac` / `hifigan`) use so `vokra-models` does not
//! gain a dependency edge onto `vokra-convert`, preserving the layered
//! convention `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`,
//! `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`. The
//! `contract_constants_mirror_the_converter` test pins every string.
//!
//! # No ONNX / no pickle (permanent)
//!
//! The upstream TEN-VAD release ships an ONNX file (~306 KB). The offline
//! bridge `tools/parity/onnx_to_safetensors.py` flattens its float
//! initializers to safetensors so the runtime never touches ONNX
//! (FR-LD-05, NFR-DS-02). This binder consumes only the resulting GGUF.

use std::sync::Arc;

use vokra_core::engines::{VadEngine, VadStreamHandle};
use vokra_core::gguf::{GgmlType, GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Contract constants — mirror of `crates/vokra-convert/src/models/ten_vad.rs`.
// See the module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model ten-vad`.
///
/// Distinct from every sibling `vad-kws` arch tag — `silero-vad` (1:1
/// pseudo-STFT + LSTM subgraph), `fsmn-vad` (fbank + LFR + CMVN op stack),
/// `openwakeword` / `openwakeword_op` (keyword spotting). Silent aliasing
/// would misroute runtime dispatch to a loader that hunts for a different
/// tensor namespace entirely (FR-EX-08 — see the module docstring's "Sibling
/// family distinctness" section).
pub const ARCH: &str = "ten_vad";

/// Expected `vokra.model.name` value written by the converter for the
/// canonical `TEN-framework/ten-vad` release.
pub const NAME: &str = "ten_vad";

/// Expected `vokra.model.category` value — the `vad-kws` umbrella shared with
/// the `silero-vad` / `fsmn-vad` / `openwakeword` siblings. Consumed by the
/// model-card generator and the zoo-manifest tier gate so a VAD is not
/// accidentally advertised as an ASR / TTS release.
pub const CATEGORY: &str = "vad-kws";

/// Primary redistribution source — the author's GitHub repository. There is
/// no HF mirror of the upstream release, which is why the converter stamps
/// [`KEY_PROVENANCE_UPSTREAM_URL`] rather than the HF-hosted sibling key
/// `vokra.provenance.upstream_hf`.
pub const UPSTREAM_URL: &str = "github.com/TEN-framework/ten-vad";

/// Default upstream weight licence (SPDX) the converter stamps —
/// **Apache-2.0** for the main project. The bundled LPCNet-derived DSP
/// front-end is separately **BSD-3-Clause** (NOTICE attribution required when
/// redistributing binaries that embed it); see the module docstring's
/// "Licensing" section.
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

/// SPDX identifier of the LPCNet-derived DSP front-end bundled in the
/// upstream distribution. Recorded as a named constant (rather than buried in
/// prose) so the follow-up front-end-port wave has a machine-greppable anchor
/// for the NOTICE-attribution obligation.
pub const FRONTEND_LICENSE_SPDX: &str = "bsd-3-clause";

// ---------------------------------------------------------------------------
// Metadata keys.
// ---------------------------------------------------------------------------

/// `vokra.model.category` metadata key. Kept module-local per the same
/// convention the converter uses (not yet centralized in
/// `vokra-core::gguf::chunks`).
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_url` metadata key — the primary redistribution
/// source URL for models whose canonical release is **not** on the Hugging
/// Face hub. Parallel to `vokra.provenance.upstream_hf`.
pub const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

/// `vokra.ten_vad.sample_rate` — the PCM rate the checkpoint was trained for
/// (Hz). Part of the **optional, all-or-nothing** topology group; see
/// [`TenVadConfig::from_gguf`].
pub const KEY_SAMPLE_RATE: &str = "vokra.ten_vad.sample_rate";

/// `vokra.ten_vad.hop_size` — samples per VAD frame at
/// [`KEY_SAMPLE_RATE`]. Part of the optional topology group.
pub const KEY_HOP_SIZE: &str = "vokra.ten_vad.hop_size";

/// `vokra.ten_vad.n_features` — front-end feature width handed to the
/// recurrent backbone per frame. Part of the optional topology group.
pub const KEY_N_FEATURES: &str = "vokra.ten_vad.n_features";

/// `vokra.ten_vad.hidden_dim` — recurrent-backbone hidden width. Part of the
/// optional topology group.
pub const KEY_HIDDEN_DIM: &str = "vokra.ten_vad.hidden_dim";

/// Every key of the optional `vokra.ten_vad.*` topology group, in the order
/// [`TenVadConfig::from_gguf`] reports them. The group is **all-or-nothing**:
/// none present → [`TenVadConfig`] is `None`; all present → a validated
/// config; some present → a loud [`VokraError::ModelLoad`].
pub const TOPOLOGY_GROUP_KEYS: [&str; 4] = [
    KEY_SAMPLE_RATE,
    KEY_HOP_SIZE,
    KEY_N_FEATURES,
    KEY_HIDDEN_DIM,
];

// ---------------------------------------------------------------------------
// Primary-source anchors — cited in the loud-partial errors so a reader
// diagnosing the gap has fully specified places to walk.
// ---------------------------------------------------------------------------

/// Primary-source anchor for the upstream repository.
pub const PRIMARY_SOURCE_REPO: &str = "github.com/TEN-framework/ten-vad";
/// Primary-source anchor for the upstream LICENSE (Apache-2.0 main project).
pub const PRIMARY_SOURCE_LICENSE: &str = "github.com/TEN-framework/ten-vad/blob/main/LICENSE";
/// The offline ONNX → safetensors bridge that produced the published
/// artefact. Named in the loud-partial error because it is the exact place a
/// follow-up wave recovers the tensor-name manifest from.
pub const PRIMARY_SOURCE_BRIDGE: &str = "tools/parity/onnx_to_safetensors.py";

// ---------------------------------------------------------------------------
// TenVadConfig — the optional, all-or-nothing `vokra.ten_vad.*` group.
// ---------------------------------------------------------------------------

/// TEN-VAD topology axes, transcribed from the **optional**
/// `vokra.ten_vad.*` GGUF metadata group.
///
/// # Why there is no `upstream_default()`
///
/// Every sibling VAD config in this crate offers an `upstream_default()`
/// constructor because its axes are stated in a primary source
/// (`FsmnVadConfig::upstream_default` transcribes the FunASR `WavFrontend`
/// config; Silero's rate branches are pinned in `silero_vad/SPEC.md`).
/// TEN-VAD has no such source available here: the converter contract stamps
/// no axes at all, and its docstring commits only to "~306 KB", "LSTM/GRU"
/// and "16 kHz". Shipping an `upstream_default()` would therefore mean
/// **inventing** a hop size and a hidden width — numbers that would look
/// authoritative, propagate into callers, and be silently wrong. Fail-closed
/// (`None`, plus a loud error at the point of use) is the honest posture
/// (CLAUDE.md「ハルシネーション厳禁」).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TenVadConfig {
    /// PCM sample rate the checkpoint expects (Hz).
    pub sample_rate: u32,
    /// Samples per VAD frame at [`Self::sample_rate`].
    pub hop_size: usize,
    /// Front-end feature width handed to the recurrent backbone per frame.
    pub n_features: usize,
    /// Recurrent-backbone hidden width.
    pub hidden_dim: usize,
}

impl TenVadConfig {
    /// Validates the axes loudly (FR-EX-08): a `0` on any axis means the
    /// `vokra.ten_vad.*` group was written by a broken producer, and a
    /// zero hop size in particular would make the streaming accumulator spin
    /// forever.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending axis.
    pub fn validate(&self) -> Result<()> {
        for (label, v) in [
            ("sample_rate", self.sample_rate as usize),
            ("hop_size", self.hop_size),
            ("n_features", self.n_features),
            ("hidden_dim", self.hidden_dim),
        ] {
            if v == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "ten_vad config: {label} must be > 0 (got 0 — the GGUF's \
                     `vokra.ten_vad.*` topology group is malformed; expected keys: \
                     {keys:?})",
                    keys = TOPOLOGY_GROUP_KEYS,
                )));
            }
        }
        Ok(())
    }

    /// Reads the **optional, all-or-nothing** `vokra.ten_vad.*` topology
    /// group.
    ///
    /// - No key present → `Ok(None)`. This is the **normal** case today: the
    ///   converter (`convert_ten_vad_file`) stamps `vokra.model.*` and
    ///   `vokra.provenance.*` only. `None` is recorded honestly rather than
    ///   substituted with invented axes; the loud error surfaces later, at
    ///   the point where an axis is actually needed.
    /// - All keys present → a validated [`TenVadConfig`].
    /// - Some keys present → a loud [`VokraError::ModelLoad`] naming exactly
    ///   which keys are missing. A half-stamped group is a producer bug, and
    ///   silently defaulting the rest would bury it (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] on a partially stamped group, on a key
    ///   whose value is not an unsigned integer, on a value that does not fit
    ///   its target width, or on a validation failure.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Option<Self>> {
        let missing: Vec<&str> = TOPOLOGY_GROUP_KEYS
            .iter()
            .copied()
            .filter(|&k| gguf.get(k).is_none())
            .collect();

        // Nothing stamped at all — the documented normal case.
        if missing.len() == TOPOLOGY_GROUP_KEYS.len() {
            return Ok(None);
        }
        // Half-stamped — loud, naming exactly what is absent.
        if !missing.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "ten_vad: the `vokra.ten_vad.*` topology group is ALL-OR-NOTHING but \
                 this GGUF stamps it only partially — missing {missing:?} out of \
                 {all:?}. Refusing to default the remainder: invented axes would look \
                 authoritative and be silently wrong (FR-EX-08). Either stamp the whole \
                 group or none of it.",
                all = TOPOLOGY_GROUP_KEYS
            )));
        }

        let get_u64 = |key: &str| -> Result<u64> {
            gguf.get(key).and_then(|v| v.as_u64()).ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "ten_vad: metadata `{key}` is present but is not an unsigned \
                     integer (the `vokra.ten_vad.*` topology group is u32-valued)"
                ))
            })
        };
        let as_usize = |key: &str, v: u64| -> Result<usize> {
            usize::try_from(v).map_err(|_| {
                VokraError::ModelLoad(format!(
                    "ten_vad: metadata `{key}` = {v} does not fit in usize on this target"
                ))
            })
        };

        let sample_rate_raw = get_u64(KEY_SAMPLE_RATE)?;
        let sample_rate = u32::try_from(sample_rate_raw).map_err(|_| {
            VokraError::ModelLoad(format!(
                "ten_vad: metadata `{KEY_SAMPLE_RATE}` = {sample_rate_raw} does not fit \
                 in u32"
            ))
        })?;
        let hop_raw = get_u64(KEY_HOP_SIZE)?;
        let feat_raw = get_u64(KEY_N_FEATURES)?;
        let hidden_raw = get_u64(KEY_HIDDEN_DIM)?;

        let cfg = Self {
            sample_rate,
            hop_size: as_usize(KEY_HOP_SIZE, hop_raw)?,
            n_features: as_usize(KEY_N_FEATURES, feat_raw)?,
            hidden_dim: as_usize(KEY_HIDDEN_DIM, hidden_raw)?,
        };
        cfg.validate()
            .map_err(|e| VokraError::ModelLoad(e.to_string()))?;
        Ok(Some(cfg))
    }
}

// ---------------------------------------------------------------------------
// TenVadWeights — real manifest walk with two loud gates.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a TEN-VAD GGUF.
///
/// The published artefact is produced by the generic
/// [`PRIMARY_SOURCE_BRIDGE`] ONNX bridge, which passes **ONNX initializer
/// names through verbatim**. This repository records no manifest of those
/// names, so this struct deliberately stores *what is actually on disk*
/// (name + dims) rather than pretending to know a fixed schema. The follow-up
/// real-forward wave resolves roles through
/// [`require_tensor`](Self::require_tensor) once the manifest is recovered.
#[derive(Debug)]
pub struct TenVadWeights {
    /// Tensors discovered on disk, in GGUF order, with their GGUF-side dims.
    tensors: Vec<(String, Vec<usize>)>,
    /// How many of [`Self::tensors`] were stored as BF16 — the runtime-side
    /// mirror of the converter's `TenVadReport::bf16_passthrough` counter.
    bf16_count: usize,
}

impl TenVadWeights {
    /// Walks the GGUF tensor manifest with two loud gates.
    ///
    /// 1. **Non-empty gate** — a GGUF that carries zero tensors is refused
    ///    rather than binding an all-zero forward (FR-EX-08).
    /// 2. **Converter-contract dtype gate** — `convert_ten_vad_file` has no
    ///    quantization arm (it passes F32 / F16 / BF16 through verbatim), so a
    ///    K-quant tensor proves the GGUF did not come from that converter.
    ///    That is the same class of signal as a wrong arch tag, and is
    ///    reported the same way: loudly, naming the tensor and its dtype.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors.
    /// - [`VokraError::ModelLoad`] when any tensor is stored in a dtype the
    ///   converter never emits.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let mut tensors: Vec<(String, Vec<usize>)> = Vec::new();
        let mut bf16_count = 0usize;

        for info in gguf.tensors() {
            match info.dtype {
                GgmlType::F32 | GgmlType::F16 => {}
                GgmlType::BF16 => bf16_count += 1,
                other => {
                    return Err(VokraError::ModelLoad(format!(
                        "ten_vad: tensor `{name}` is stored as {other:?}, but the \
                         `{ARCH}` converter (`convert_ten_vad_file`) has NO quantization \
                         arm — it passes F32 / F16 / BF16 through verbatim. A quantized \
                         tensor therefore proves this GGUF was not produced by that \
                         converter, the same class of signal as a wrong arch tag. \
                         Refusing to bind (FR-EX-08 — no silent partial load). \
                         Re-run `vokra-cli convert --model ten-vad` against an upstream \
                         `{UPSTREAM_URL}` checkpoint bridged through `{bridge}`.",
                        name = info.name,
                        bridge = PRIMARY_SOURCE_BRIDGE,
                    )));
                }
            }
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            tensors.push((info.name.clone(), dims));
        }

        if tensors.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "ten_vad: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). A legitimate TEN-VAD checkpoint carries the \
                 float initializers of the upstream ~306 KB ONNX bundle \
                 (arch={ARCH}, name={NAME}); zero tensors always signals a \
                 mis-produced GGUF. Re-run `vokra-cli convert --model ten-vad` \
                 against an upstream `{UPSTREAM_URL}` checkpoint bridged through \
                 `{bridge}`.",
                bridge = PRIMARY_SOURCE_BRIDGE,
            )));
        }

        Ok(Self {
            tensors,
            bf16_count,
        })
    }

    /// Number of tensors bound from the GGUF.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// How many bound tensors were stored as BF16 — the runtime-side mirror
    /// of the converter's `TenVadReport::bf16_passthrough` counter. BF16
    /// widens to f32 losslessly at load (`bits << 16`); a silent widen /
    /// downcast regression surfaces as this counter drifting away from the
    /// converter's.
    #[inline]
    #[must_use]
    pub fn bf16_count(&self) -> usize {
        self.bf16_count
    }

    /// Tensor names present in the GGUF, in file order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|(n, _)| n.as_str())
    }

    /// Whether a tensor with `name` is present.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.iter().any(|(n, _)| n == name)
    }

    /// Resolves `name` to its GGUF-side dims, or fails loudly **naming the
    /// tensor**.
    ///
    /// This is the name-resolution primitive the follow-up real-forward wave
    /// binds against: because the TEN-VAD manifest is ONNX-initializer-derived
    /// and unrecorded, the first thing that wave does is probe names, and a
    /// probe that misses must say which name missed and what the GGUF does
    /// carry — never `None` swallowed into a default (FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] naming the missing tensor and previewing the
    /// available names.
    pub fn require_tensor(&self, name: &str) -> Result<&[usize]> {
        if let Some((_, dims)) = self.tensors.iter().find(|(n, _)| n == name) {
            return Ok(dims);
        }
        // Preview enough of the real manifest to diagnose a typo / a
        // wrong-model GGUF without dumping a huge list into the message.
        const PREVIEW: usize = 12;
        let preview: Vec<&str> = self.names().take(PREVIEW).collect();
        let elided = self.tensors.len().saturating_sub(preview.len());
        let tail = if elided > 0 {
            format!(" (+{elided} more)")
        } else {
            String::new()
        };
        Err(VokraError::ModelLoad(format!(
            "ten_vad: required tensor `{name}` is absent from the GGUF. This GGUF \
             carries {count} tensor(s): {preview:?}{tail}. Note that the `{ARCH}` \
             artefact's tensor names are the upstream ONNX **initializer names \
             verbatim** (passed through by `{bridge}`), so a name mismatch usually \
             means the expected-name list was guessed rather than read off a real \
             checkpoint. Refusing to substitute a default (FR-EX-08).",
            count = self.tensors.len(),
            bridge = PRIMARY_SOURCE_BRIDGE,
        )))
    }
}

// ---------------------------------------------------------------------------
// TenVad — the runtime binder handle.
// ---------------------------------------------------------------------------

/// TEN-VAD (`TEN-framework/ten-vad`, Apache-2.0 + BSD-3-Clause front-end)
/// runtime binder.
///
/// Load with [`from_gguf`](Self::from_gguf) / [`open`](Self::open), then obtain
/// a stateful stream through the [`VadEngine`] trait
/// ([`open_stream`](VadEngine::open_stream)) exactly as with Silero VAD v5 and
/// FSMN-VAD — the three VAD binders share the
/// [`VadEngine`] / [`VadStreamHandle`] shape so a caller can swap between them
/// without rewriting call sites.
///
/// The model itself is immutable and shareable; all mutable state (the frame
/// accumulator, and the recurrent state the follow-up wave adds) lives in the
/// stream handle (FR-LD-06).
///
/// See the module docs for the loud-partial contract on
/// [`frame_probability`](Self::frame_probability) /
/// [`VadStreamHandle::push_pcm`].
#[derive(Debug)]
pub struct TenVad {
    weights: Arc<TenVadWeights>,
    /// `None` when the GGUF does not stamp the optional `vokra.ten_vad.*`
    /// topology group — the normal case for artefacts produced by today's
    /// converter. Never silently defaulted (see [`TenVadConfig`]).
    cfg: Option<TenVadConfig>,
    weight_license: LicenseClass,
}

impl TenVad {
    /// Binds a TEN-VAD GGUF: validates the arch tag, walks the tensor
    /// manifest, reads the optional topology group and surfaces the stamped
    /// weight-license class for the compliance-gate cross-checks.
    ///
    /// Every failure is a distinct [`VokraError::ModelLoad`] naming the
    /// missing / wrong key so a reader diagnosing a mis-produced GGUF has
    /// exactly one place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent or is not
    ///   `"ten_vad"` (a sibling `vad-kws` GGUF handed here by mistake —
    ///   `silero-vad` / `fsmn-vad` / `openwakeword` / `openwakeword_op` —
    ///   fails with a specific message instead of a downstream missing-tensor
    ///   error).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors or any
    ///   tensor in a dtype the converter never emits
    ///   ([`TenVadWeights::from_gguf`]).
    /// - [`VokraError::ModelLoad`] when the `vokra.ten_vad.*` topology group
    ///   is stamped only partially or malformed ([`TenVadConfig::from_gguf`]).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check FIRST so a mis-typed model handed here fails with a
        //    specific message instead of a downstream missing-tensor error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "ten_vad: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF \
                     produced by `vokra-cli convert --model ten-vad`? Note that the \
                     sibling `category={CATEGORY}` arch tags — `silero-vad` (Silero VAD \
                     v5/v6.2.1 as a 1:1-preserved pseudo-STFT + LSTM subgraph with \
                     per-rate `sr8k.*` / `sr16k.*` branches), `fsmn-vad` (FunASR fbank + \
                     LFR + CMVN op stack requiring a `vokra.fsmn_vad.*` chunk), \
                     `openwakeword` and `openwakeword_op` (keyword spotting, which \
                     reports `(wakeword_name, probability)` pairs rather than a single \
                     speech probability) — all answer a related question with a \
                     completely different topology and tensor namespace. Silently \
                     aliasing arch would misroute the runtime dispatch (FR-EX-08 — no \
                     silent partial load)."
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "ten_vad: GGUF is missing `vokra.model.arch` — this is not a \
                     Vokra-native ten_vad GGUF (was it produced by `vokra-cli convert \
                     --model ten-vad`?)"
                        .to_owned(),
                ));
            }
        }

        // 2. Tensor manifest, with the non-empty + converter-contract dtype
        //    gates.
        let weights = TenVadWeights::from_gguf(file)?;

        // 3. Optional, all-or-nothing topology group.
        let cfg = TenVadConfig::from_gguf(file)?;

        // 4. Provenance surfacing for the M2-13 compliance gate. The TEN-VAD
        //    converter stamps `Permissive` (apache-2.0); a GGUF missing the
        //    stamp reads back as `Unknown` (fail-closed default per
        //    `[[feedback-license-signoff-primary-source]]`).
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        Ok(Self {
            weights: Arc::new(weights),
            cfg,
            weight_license,
        })
    }

    /// Opens and binds the model from a GGUF file on disk.
    ///
    /// # Errors
    ///
    /// Propagates the GGUF read/parse error, then every
    /// [`from_gguf`](Self::from_gguf) failure mode.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// The topology axes read from the optional `vokra.ten_vad.*` group, or
    /// `None` when the GGUF does not stamp it (the normal case for today's
    /// converter output). Never silently defaulted — see [`TenVadConfig`].
    #[inline]
    #[must_use]
    pub fn config(&self) -> Option<&TenVadConfig> {
        self.cfg.as_ref()
    }

    /// The bound weight manifest.
    #[inline]
    #[must_use]
    pub fn weights(&self) -> &TenVadWeights {
        &self.weights
    }

    /// Number of tensors bound from the GGUF.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// How many bound tensors were stored as BF16 (see
    /// [`TenVadWeights::bf16_count`]).
    #[inline]
    #[must_use]
    pub fn bf16_count(&self) -> usize {
        self.weights.bf16_count()
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. The TEN-VAD converter stamps
    /// [`LicenseClass::Permissive`] (apache-2.0, [`DEFAULT_LICENSE_SPDX`]); a
    /// GGUF missing the stamp reads back as [`LicenseClass::Unknown`]
    /// (fail-closed at the M2-13 compliance gate).
    ///
    /// Note that the bundled LPCNet-derived DSP front-end carries a separate
    /// [`FRONTEND_LICENSE_SPDX`] (BSD-3-Clause) NOTICE-attribution obligation
    /// that this single class cannot express — see the module docstring's
    /// "Licensing" section.
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Speech probability for **one** frame of PCM.
    ///
    /// The one-shot analogue of pushing a single frame through
    /// [`VadStreamHandle::push_pcm`]; it carries no recurrent state.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — see the module docs for the
    /// four blockers (missing tensor-name manifest / missing topology axes /
    /// unresolved LSTM-vs-GRU backbone / missing LPCNet-derived front-end).
    /// **No fabricated probability is ever emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] when the GGUF stamps no
    ///   `vokra.ten_vad.*` topology group (the frame geometry is unknown, so
    ///   even the shape check below cannot run).
    /// - [`VokraError::InvalidArgument`] when `frame.len()` is not exactly
    ///   [`TenVadConfig::hop_size`] — the frame size is load-bearing and this
    ///   binder never silently pads or truncates.
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred forward.
    pub fn frame_probability(&self, frame: &[f32]) -> Result<f32> {
        let cfg = self.cfg.ok_or_else(missing_topology_axes)?;
        if frame.len() != cfg.hop_size {
            return Err(VokraError::InvalidArgument(format!(
                "ten_vad: frame has {got} samples but the checkpoint's frame size is \
                 {want} (`{KEY_HOP_SIZE}`). Refusing to pad / truncate silently — the \
                 frame geometry is load-bearing for a VAD (FR-EX-08).",
                got = frame.len(),
                want = cfg.hop_size,
            )));
        }
        Err(forward_loud_partial())
    }
}

impl VadEngine for TenVad {
    fn open_stream(&self) -> Box<dyn VadStreamHandle + Send> {
        Box::new(TenVadStream {
            cfg: self.cfg,
            weights: Arc::clone(&self.weights),
            pending_pcm: Vec::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// TenVadStream — real hop-based accumulator, loud-partial forward.
// ---------------------------------------------------------------------------

/// Stateful TEN-VAD stream: push PCM, get per-frame speech probabilities.
///
/// Mirrors `FsmnVadStream` / the Silero `VadStream`: the handle owns every
/// mutable buffer (FR-LD-06) and [`reset`](VadStreamHandle::reset) returns it
/// to its initial state.
///
/// # What is real here
///
/// The **sample-rate gate** and the **hop-based frame accumulator**. A
/// mismatched rate is a loud [`VokraError::InvalidArgument`], never a silent
/// resample. Sub-frame pushes are buffered and answered with an empty vector
/// — the same "starving" semantics `FsmnVadStream::push_pcm` has when its
/// front-end has not yet closed a frame.
///
/// # What is loud-partial here
///
/// The moment a push would **close a complete frame**, [`push_pcm`] returns
/// [`VokraError::UnsupportedOp`] rather than a fabricated probability. On that
/// path the pushed samples are deliberately **not** appended to the internal
/// buffer: the call failed, so the caller still owns its data, and the buffer
/// cannot grow without bound while a caller ignores the error (it is capped at
/// `hop_size - 1` samples by construction).
///
/// [`push_pcm`]: VadStreamHandle::push_pcm
pub struct TenVadStream {
    /// `None` when the GGUF stamped no topology group — every push is then a
    /// loud [`VokraError::UnsupportedOp`], because the frame geometry that
    /// framing needs is simply unknown.
    cfg: Option<TenVadConfig>,
    /// Shared with the parent model — never mutated. Held so the follow-up
    /// real-forward wave has its weights in hand without a re-bind.
    #[allow(dead_code)]
    weights: Arc<TenVadWeights>,
    /// Samples buffered toward the next complete frame. Always strictly
    /// shorter than `cfg.hop_size` (see the struct docs).
    pending_pcm: Vec<f32>,
}

impl TenVadStream {
    /// Samples currently buffered toward the next complete frame. Always
    /// `< hop_size` — a diagnostic accessor the tests use to prove the
    /// accumulator is real and bounded.
    #[inline]
    #[must_use]
    pub fn pending_samples(&self) -> usize {
        self.pending_pcm.len()
    }

    /// The topology axes this stream was opened against (`None` when the GGUF
    /// stamped no `vokra.ten_vad.*` group).
    #[inline]
    #[must_use]
    pub fn config(&self) -> Option<&TenVadConfig> {
        self.cfg.as_ref()
    }
}

impl VadStreamHandle for TenVadStream {
    fn push_pcm(&mut self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        // Without the topology group there is no frame geometry at all, so
        // even the rate check has nothing to compare against.
        let cfg = self.cfg.ok_or_else(missing_topology_axes)?;

        // The rate is a load-bearing invariant of a VAD's front-end (it was
        // trained at one rate); refuse a mismatch loudly rather than
        // resampling silently (FR-EX-08).
        if sample_rate != cfg.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "ten_vad: sample rate mismatch — pushed {sample_rate} Hz but the \
                 checkpoint expects {want} Hz (`{KEY_SAMPLE_RATE}`). Resample upstream, \
                 or open a stream on the matching rate; this binder never resamples \
                 silently (FR-EX-08).",
                want = cfg.sample_rate,
            )));
        }

        // Would this push close a complete frame? Compute first, mutate only
        // on the success path — so a rejected push leaves the caller's data
        // with the caller and keeps `pending_pcm` bounded by `hop_size - 1`.
        let would_be = self.pending_pcm.len().saturating_add(pcm.len());
        if would_be < cfg.hop_size {
            self.pending_pcm.extend_from_slice(pcm);
            // Starving: no frame boundary crossed, so there are honestly zero
            // probabilities to report. Same semantics as
            // `FsmnVadStream::push_pcm` before its front-end closes a frame.
            return Ok(Vec::new());
        }

        Err(forward_loud_partial())
    }

    fn reset(&mut self) {
        self.pending_pcm.clear();
    }
}

// ---------------------------------------------------------------------------
// Loud-partial error constructors.
// ---------------------------------------------------------------------------

/// The [`VokraError::UnsupportedOp`] returned when the GGUF stamps no
/// `vokra.ten_vad.*` topology group, so the frame geometry is unknown.
///
/// This is a *distinct* blocker from [`forward_loud_partial`]: it is the one a
/// follow-up wave can close **without** touching the forward, simply by
/// teaching `convert_ten_vad_file` to stamp the group.
fn missing_topology_axes() -> VokraError {
    VokraError::UnsupportedOp(format!(
        "ten_vad (loud-partial): this GGUF stamps no `vokra.ten_vad.*` topology \
         group, so the frame geometry is unknown — expected keys {keys:?}. The \
         converter (`crates/vokra-convert/src/models/ten_vad.rs` \
         `convert_ten_vad_file`) currently stamps only `vokra.model.*` and \
         `vokra.provenance.*`; it is a verbatim float pass-through that transcribes \
         no axes. Refusing to assume a hop size / feature width: invented axes would \
         look authoritative and be silently wrong (FR-EX-08, CLAUDE.md \
         ハルシネーション厳禁). Primary sources: repo {repo}, licence {license}, \
         offline bridge {bridge}.",
        keys = TOPOLOGY_GROUP_KEYS,
        repo = PRIMARY_SOURCE_REPO,
        license = PRIMARY_SOURCE_LICENSE,
        bridge = PRIMARY_SOURCE_BRIDGE,
    ))
}

/// The [`VokraError::UnsupportedOp`] returned by every TEN-VAD forward entry
/// point until the real forward lands.
///
/// Names **four** concrete blockers and **three** primary-source anchors so a
/// reader diagnosing the gap has exactly where to walk. Mirror of the nisqa /
/// utmosv2 / emotion2vec / panns / redimnet / storm loud-partial-message
/// precedent (CLAUDE.md 教訓 (a)).
fn forward_loud_partial() -> VokraError {
    VokraError::UnsupportedOp(format!(
        "ten_vad frame forward (loud-partial): the real forward is deferred; four \
         concrete blockers must be closed before an honest speech probability can be \
         emitted. \
         (1) MISSING TENSOR-NAME MANIFEST — the published artefact was produced by \
         the generic ONNX bridge `{bridge}`, which passes ONNX initializer names \
         through VERBATIM, and this repository records no manifest of those names, so \
         no weight walk can map a tensor to a role (input-gate vs hidden-gate vs bias, \
         per layer); recover it by re-running that bridge against the upstream release \
         and reading `TenVadWeights::names()`. \
         (2) MISSING TOPOLOGY AXES — the converter stamps no `vokra.ten_vad.*` group \
         (expected keys {keys:?}), so hop size, feature width, hidden width and layer \
         count are all unknown at load time. \
         (3) UNRESOLVED BACKBONE FAMILY — the converter docstring says \"small \
         LSTM/GRU backbone\" without committing to either, and picking one would be a \
         coin flip whose wrong side is SILENT-wrong; note `vokra-ops` already carries a \
         shape-generic GRU cell (`vokra_ops::rnnoise::gru_forward`, TensorFlow \
         `GRUCell` gate convention), so the blocker is the manifest and the axes, not \
         the arithmetic. \
         (4) MISSING LPCNet-DERIVED FRONT-END — `vokra-ops` carries RNNoise's FIXED \
         22-band Bark filterbank (`vokra_ops::rnnoise::bark_filterbank`, same Xiph \
         lineage, different hard-coded table) but not the LPCNet-derived feature stack \
         TEN-VAD bundles, whose spec is not transcribable from the converter contract; \
         that front-end is separately {frontend_spdx} and carries a NOTICE-attribution \
         obligation. \
         Primary sources: repo {repo}, licence {license} ({main_spdx} main project), \
         offline bridge {bridge}. The runtime cannot fabricate a speech probability \
         (FR-EX-08 — no silent partial output).",
        bridge = PRIMARY_SOURCE_BRIDGE,
        keys = TOPOLOGY_GROUP_KEYS,
        frontend_spdx = FRONTEND_LICENSE_SPDX,
        repo = PRIMARY_SOURCE_REPO,
        license = PRIMARY_SOURCE_LICENSE,
        main_spdx = DEFAULT_LICENSE_SPDX,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the TEN-VAD runtime binder — contract-constant pins,
    //! metadata round-trip, negative-space round-trip on every loud gate, and
    //! the real (non-deferred) streaming accumulator semantics.
    //!
    //! # What "round-trip" means here
    //!
    //! On a real checkpoint this would be `push_pcm(...)` returning a speech
    //! probability per frame, but the forward is deferred for the four
    //! blockers the module docs enumerate. Fabricating probabilities would
    //! violate CLAUDE.md 教訓 (a)「loud-partial は fake-complete より honest」.
    //! The semantics that CAN be tested honestly, and are:
    //!
    //! 1. **Contract-constant pin** — `ARCH` / `NAME` / `CATEGORY` /
    //!    `UPSTREAM_URL` / `DEFAULT_LICENSE_SPDX` match the converter's values
    //!    exactly, so a converter drift without a binder follow-through fails
    //!    here.
    //! 2. **Metadata round-trip** — arch + category + provenance URL + license
    //!    stamp + tensor manifest bind with the documented surface semantics
    //!    (including the fail-closed `Unknown` license fallback).
    //! 3. **BF16 round-trip** — a BF16 tensor is counted AND widens back to
    //!    the exact f32 values it was built from (`bits << 16` is exact), so a
    //!    silent widen / downcast regression fails here.
    //! 4. **Loud negative-space round-trip** — every stated blocker (missing
    //!    arch / wrong arch / empty manifest / quantized tensor / missing
    //!    tensor / half-stamped topology group / zero axis / rate mismatch /
    //!    frame-size mismatch / deferred forward) fires at its documented
    //!    surface point, in the documented error variant.
    //! 5. **Real accumulator semantics** — sub-frame pushes buffer and return
    //!    empty; the buffer stays bounded; `reset` clears it.

    use super::*;
    use vokra_core::gguf::GgufBuilder;

    /// 16-bit BF16 payload for `values` (top 16 bits of the f32 pattern —
    /// mirror of the converter test module's `bf16_bytes` helper).
    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// Builds a legitimate TEN-VAD GGUF: arch + name + category + upstream
    /// URL, an optional weight-license stamp, an optional fully-stamped
    /// topology group, and one representative F32 tensor so the non-empty
    /// gate passes.
    ///
    /// The tensor name is a **placeholder**: the real artefact's names are
    /// upstream ONNX initializer names, which this repository does not record
    /// (that is blocker (1) of the loud-partial). Nothing in the binder
    /// depends on the name, so a placeholder is honest here.
    fn ten_vad_gguf(
        weight_license_class: Option<LicenseClass>,
        topology: Option<TenVadConfig>,
    ) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        if let Some(cfg) = topology {
            b.add_u32(KEY_SAMPLE_RATE, cfg.sample_rate);
            b.add_u32(KEY_HOP_SIZE, cfg.hop_size as u32);
            b.add_u32(KEY_N_FEATURES, cfg.n_features as u32);
            b.add_u32(KEY_HIDDEN_DIM, cfg.hidden_dim as u32);
        }
        b.add_tensor(
            "ten_vad.placeholder.weight",
            GgmlType::F32,
            vec![2, 3],
            vec![0u8; 2 * 3 * 4],
        )
        .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    /// A synthetic topology group. These numbers are **test fixtures**, not a
    /// claim about the upstream checkpoint — the whole point of the
    /// loud-partial is that the real axes are unknown (see `TenVadConfig`'s
    /// "Why there is no `upstream_default()`").
    fn synthetic_topology() -> TenVadConfig {
        TenVadConfig {
            sample_rate: 16_000,
            hop_size: 8,
            n_features: 4,
            hidden_dim: 6,
        }
    }

    // -----------------------------------------------------------------------
    // Test 1 — contract-constant pin (cross-crate handshake with the
    //          converter) + sibling arch distinctness
    // -----------------------------------------------------------------------

    #[test]
    fn contract_constants_mirror_the_converter() {
        // Mirror of `vokra_convert::models::ten_vad::{ARCH, NAME, CATEGORY,
        // UPSTREAM_URL, DEFAULT_LICENSE_SPDX}` — a converter-side rename
        // without a binder-side follow-through lands here or fails.
        assert_eq!(ARCH, "ten_vad");
        assert_eq!(NAME, "ten_vad");
        assert_eq!(CATEGORY, "vad-kws");
        assert_eq!(UPSTREAM_URL, "github.com/TEN-framework/ten-vad");
        assert_eq!(DEFAULT_LICENSE_SPDX, "apache-2.0");
        // The bundled DSP front-end is separately BSD-3-Clause (NOTICE
        // attribution), per the converter's module docstring.
        assert_eq!(FRONTEND_LICENSE_SPDX, "bsd-3-clause");
        // Metadata key strings the converter writes.
        assert_eq!(KEY_MODEL_CATEGORY, "vokra.model.category");
        assert_eq!(KEY_PROVENANCE_UPSTREAM_URL, "vokra.provenance.upstream_url");

        // Distinct from every sibling `vad-kws` arch tag — silently sharing
        // one would misroute the runtime dispatch (FR-EX-08).
        for sibling in ["silero-vad", "fsmn-vad", "openwakeword", "openwakeword_op"] {
            assert_ne!(ARCH, sibling, "ten_vad arch must not alias `{sibling}`");
        }

        // The topology group must list exactly the four documented keys, in
        // the documented order (the loud errors quote this array).
        assert_eq!(
            TOPOLOGY_GROUP_KEYS,
            [
                "vokra.ten_vad.sample_rate",
                "vokra.ten_vad.hop_size",
                "vokra.ten_vad.n_features",
                "vokra.ten_vad.hidden_dim",
            ]
        );
    }

    // -----------------------------------------------------------------------
    // Test 2 — missing arch is loud
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "some-other-name");
        b.add_tensor("some.tensor", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = TenVad::from_gguf(&file) else {
            panic!("expected ModelLoad when `vokra.model.arch` is absent");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`vokra.model.arch`"),
                    "message must name the missing key, got `{m}`"
                );
                assert!(
                    m.contains("not a Vokra-native ten_vad GGUF"),
                    "message must name the missing-arch surface, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 3 — wrong arch is loud and names both tags plus the sibling fleet
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_wrong_arch() {
        // An `fsmn-vad` GGUF handed to the TEN-VAD binder by mistake. Both
        // are `vad-kws` and both emit per-frame speech probabilities, but
        // FSMN demands a `vokra.fsmn_vad.*` chunk and an entirely different
        // tensor namespace — silent aliasing would surface as a confusing
        // downstream missing-key error.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "fsmn-vad");
        b.add_string(chunks::KEY_MODEL_NAME, "fsmn-vad-zh-cn-16k-common");
        b.add_tensor(
            "encoder.in_linear.bias",
            GgmlType::F32,
            vec![4],
            vec![0u8; 16],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = TenVad::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`fsmn-vad`") && m.contains("`ten_vad`"),
                    "message must name both the actual and expected arch tags, got `{m}`"
                );
                for sibling in ["silero-vad", "fsmn-vad", "openwakeword", "openwakeword_op"] {
                    assert!(
                        m.contains(sibling),
                        "expected sibling `{sibling}` disambiguation in error: {m}"
                    );
                }
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 4 — a well-formed GGUF binds (metadata round-trip)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_binds_well_formed_gguf() {
        let file = ten_vad_gguf(Some(LicenseClass::Permissive), None);
        let m = TenVad::from_gguf(&file).expect("valid TEN-VAD GGUF must bind");

        assert_eq!(
            m.weight_license(),
            LicenseClass::Permissive,
            "apache-2.0 must surface as Permissive (what the converter stamps)"
        );
        assert_eq!(m.tensor_count(), 1, "the manifest walk must see the tensor");
        assert_eq!(m.bf16_count(), 0, "an F32-only GGUF has no BF16 tensors");
        assert!(
            m.weights().contains("ten_vad.placeholder.weight"),
            "the bound manifest must expose the tensor by name"
        );
        assert_eq!(
            m.weights()
                .require_tensor("ten_vad.placeholder.weight")
                .unwrap(),
            &[2usize, 3],
            "require_tensor must return the GGUF-side dims"
        );
        // Today's converter stamps no topology group, so `None` is the
        // documented normal case — recorded honestly, never defaulted.
        assert!(
            m.config().is_none(),
            "an un-stamped topology group must read back as None, not invented axes"
        );
    }

    // -----------------------------------------------------------------------
    // Test 5 — missing license stamp fails CLOSED to Unknown
    // -----------------------------------------------------------------------

    #[test]
    fn missing_license_stamp_fails_closed_to_unknown() {
        let file = ten_vad_gguf(None, None);
        let m = TenVad::from_gguf(&file).expect("arch + tensors are the bind gates");
        assert_eq!(
            m.weight_license(),
            LicenseClass::Unknown,
            "an absent provenance stamp must fail CLOSED to Unknown, never to Permissive"
        );
    }

    // -----------------------------------------------------------------------
    // Test 6 — empty tensor manifest is loud
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_manifest() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, "permissive");
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = TenVad::from_gguf(&file) else {
            panic!("expected ModelLoad on an empty tensor manifest");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("zero tensors"),
                    "message must name the empty-manifest gap, got `{m}`"
                );
                assert!(m.contains("FR-EX-08"), "message must cite FR-EX-08: {m}");
                assert!(
                    m.contains("vokra-cli convert --model ten-vad"),
                    "message must include the repro command, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 7 — a quantized tensor violates the converter contract, loudly
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_quantized_tensor() {
        // `convert_ten_vad_file` has no quantization arm — it passes
        // F32 / F16 / BF16 through verbatim — so a K-quant tensor proves the
        // GGUF came from somewhere else. Q4_K is a 256-element super-block of
        // 144 bytes.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_tensor(
            "ten_vad.quantized.weight",
            GgmlType::Q4K,
            vec![256],
            vec![0u8; 144],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = TenVad::from_gguf(&file) else {
            panic!("expected ModelLoad on a quantized tensor");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("ten_vad.quantized.weight"),
                    "message must name the offending tensor, got `{m}`"
                );
                assert!(
                    m.contains("Q4K"),
                    "message must name the offending dtype, got `{m}`"
                );
                assert!(
                    m.contains("no quantization arm") || m.contains("NO quantization arm"),
                    "message must explain the converter contract, got `{m}`"
                );
                assert!(m.contains("FR-EX-08"), "message must cite FR-EX-08: {m}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 8 — a missing tensor is loud AND names the tensor
    // -----------------------------------------------------------------------

    #[test]
    fn require_tensor_names_the_missing_tensor() {
        let file = ten_vad_gguf(Some(LicenseClass::Permissive), None);
        let m = TenVad::from_gguf(&file).expect("bind");

        let Err(err) = m.weights().require_tensor("ten_vad.absent.weight") else {
            panic!("expected ModelLoad for an absent tensor");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("ten_vad.absent.weight"),
                    "message must NAME the missing tensor, got `{msg}`"
                );
                assert!(
                    msg.contains("ten_vad.placeholder.weight"),
                    "message must preview what the GGUF does carry, got `{msg}`"
                );
                assert!(
                    msg.contains(PRIMARY_SOURCE_BRIDGE),
                    "message must point at the ONNX bridge that fixes the name gap: {msg}"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite FR-EX-08: {msg}"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 9 — BF16 counts AND widens back losslessly
    // -----------------------------------------------------------------------

    #[test]
    fn bf16_tensor_counts_and_widens_losslessly() {
        // Values chosen to be exactly representable in BF16 (they have at
        // most 7 mantissa bits), so the widen is bit-exact and the assertion
        // is a real round-trip rather than an approximate one.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let payload = bf16_bytes(&values);
        assert_eq!(payload.len(), 12, "6 elements x 2 bytes BF16");

        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_tensor("ten_vad.bf16.weight", GgmlType::BF16, vec![2, 3], payload)
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let m = TenVad::from_gguf(&file).expect("BF16 must ride the float arm");
        assert_eq!(
            m.bf16_count(),
            1,
            "the BF16 observability counter must mirror the converter's \
             TenVadReport::bf16_passthrough"
        );
        assert_eq!(m.tensor_count(), 1);

        // Round-trip through the single widening choke point: BF16 is the top
        // 16 bits of the f32 pattern, so `bits << 16` is exact.
        let widened = file.tensor_f32("ten_vad.bf16.weight").expect("widen BF16");
        assert_eq!(
            widened,
            values.to_vec(),
            "BF16 -> f32 must be bit-exact (no silent downcast / rounding)"
        );
    }

    // -----------------------------------------------------------------------
    // Test 10 — the topology group is all-or-nothing
    // -----------------------------------------------------------------------

    #[test]
    fn topology_group_partial_stamp_is_loud() {
        // Stamp two of the four keys — a producer bug that silent defaulting
        // would bury.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_u32(KEY_SAMPLE_RATE, 16_000);
        b.add_u32(KEY_HOP_SIZE, 8);
        b.add_tensor(
            "ten_vad.placeholder.weight",
            GgmlType::F32,
            vec![2],
            vec![0u8; 8],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = TenVad::from_gguf(&file) else {
            panic!("expected ModelLoad on a half-stamped topology group");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("ALL-OR-NOTHING"),
                    "message must state the all-or-nothing rule, got `{m}`"
                );
                // Pin the *missing* list precisely, not merely "the key
                // string appears somewhere": the message also quotes the full
                // expected-key array, so a substring check alone would pass
                // even if the missing-key computation were wrong.
                let expected_missing = format!("missing [{KEY_N_FEATURES:?}, {KEY_HIDDEN_DIM:?}]");
                assert!(
                    m.contains(&expected_missing),
                    "message must list exactly the absent keys (`{expected_missing}`), got `{m}`"
                );
                assert!(m.contains("FR-EX-08"), "message must cite FR-EX-08: {m}");
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn topology_group_round_trips_when_fully_stamped() {
        let cfg = synthetic_topology();
        let file = ten_vad_gguf(Some(LicenseClass::Permissive), Some(cfg));
        let m = TenVad::from_gguf(&file).expect("bind");
        assert_eq!(
            m.config().copied(),
            Some(cfg),
            "a fully stamped topology group must round-trip verbatim"
        );
    }

    #[test]
    fn topology_group_zero_axis_is_loud() {
        // A zero hop size would make the streaming accumulator spin forever;
        // the validator must refuse it at load time.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_u32(KEY_SAMPLE_RATE, 16_000);
        b.add_u32(KEY_HOP_SIZE, 0);
        b.add_u32(KEY_N_FEATURES, 4);
        b.add_u32(KEY_HIDDEN_DIM, 6);
        b.add_tensor(
            "ten_vad.placeholder.weight",
            GgmlType::F32,
            vec![2],
            vec![0u8; 8],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = TenVad::from_gguf(&file) else {
            panic!("expected ModelLoad on a zero axis");
        };
        match err {
            VokraError::ModelLoad(m) => assert!(
                m.contains("hop_size") && m.contains("must be > 0"),
                "message must name the offending axis, got `{m}`"
            ),
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 11 — sample-rate mismatch is a loud InvalidArgument, never a
    //           silent resample
    // -----------------------------------------------------------------------

    #[test]
    fn push_pcm_rejects_sample_rate_mismatch() {
        let cfg = synthetic_topology();
        let file = ten_vad_gguf(Some(LicenseClass::Permissive), Some(cfg));
        let m = TenVad::from_gguf(&file).expect("bind");
        let mut stream = m.open_stream();

        // 8 kHz into a 16 kHz checkpoint. A resampling VAD would silently
        // halve every frame's duration and produce plausible-but-wrong
        // probabilities — exactly the failure FR-EX-08 forbids.
        let Err(err) = stream.push_pcm(&[0.0; 8], 8_000) else {
            panic!("expected InvalidArgument on a sample-rate mismatch");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("8000") && msg.contains("16000"),
                    "message must name both the pushed and expected rate, got `{msg}`"
                );
                assert!(
                    msg.contains("never resamples silently") || msg.contains("Resample upstream"),
                    "message must state the no-silent-resample rule, got `{msg}`"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite FR-EX-08: {msg}"
                );
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }
        // The rejected push must not have been buffered.
        assert_eq!(
            stream.push_pcm(&[0.0; 1], 16_000).expect("sub-frame push"),
            Vec::<f32>::new(),
        );
    }

    // -----------------------------------------------------------------------
    // Test 12 — the frame accumulator is real: starving pushes buffer and
    //           return empty; a complete frame is the loud-partial
    // -----------------------------------------------------------------------

    #[test]
    fn push_pcm_buffers_sub_frame_then_loud_partials_on_a_complete_frame() {
        let cfg = synthetic_topology(); // hop_size = 8
        let file = ten_vad_gguf(Some(LicenseClass::Permissive), Some(cfg));
        let m = TenVad::from_gguf(&file).expect("bind");

        // `open_stream` hands back a `Box<dyn VadStreamHandle>`, which cannot
        // expose the `pending_samples` diagnostic; build the concrete stream
        // exactly as `open_stream` does so the accumulator can be observed.
        // (The trait-object path itself is covered by
        // `binds_as_a_vad_engine_trait_object`.)
        let mut stream = TenVadStream {
            cfg: Some(cfg),
            weights: Arc::clone(&m.weights),
            pending_pcm: Vec::new(),
        };

        // Three sub-frame pushes: 3 + 3 = 6 < 8, so both buffer and report no
        // completed frames (honest "starving", not a fabricated probability).
        assert!(stream.push_pcm(&[0.1; 3], 16_000).unwrap().is_empty());
        assert_eq!(stream.pending_samples(), 3);
        assert!(stream.push_pcm(&[0.2; 3], 16_000).unwrap().is_empty());
        assert_eq!(stream.pending_samples(), 6);

        // The next push would close a complete frame (6 + 2 == 8) -> loud.
        let Err(err) = stream.push_pcm(&[0.3; 2], 16_000) else {
            panic!("a complete frame must hit the loud-partial gate");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(msg.contains("loud-partial"), "posture label: {msg}");
                assert!(
                    msg.contains("MISSING TENSOR-NAME MANIFEST"),
                    "blocker (1) must be named: {msg}"
                );
                assert!(
                    msg.contains("MISSING TOPOLOGY AXES"),
                    "blocker (2) must be named: {msg}"
                );
                assert!(
                    msg.contains("UNRESOLVED BACKBONE FAMILY"),
                    "blocker (3) must be named: {msg}"
                );
                assert!(
                    msg.contains("MISSING LPCNet-DERIVED FRONT-END"),
                    "blocker (4) must be named: {msg}"
                );
                for anchor in [
                    PRIMARY_SOURCE_REPO,
                    PRIMARY_SOURCE_LICENSE,
                    PRIMARY_SOURCE_BRIDGE,
                ] {
                    assert!(
                        msg.contains(anchor),
                        "primary source `{anchor}` cited: {msg}"
                    );
                }
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite FR-EX-08: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }

        // The rejected push left the buffer untouched: the caller still owns
        // its samples and the buffer stays bounded below one frame.
        assert_eq!(stream.pending_samples(), 6);
        assert!(stream.pending_samples() < cfg.hop_size);

        stream.reset();
        assert_eq!(
            stream.pending_samples(),
            0,
            "reset must clear the accumulator"
        );
    }

    // -----------------------------------------------------------------------
    // Test 13 — without the topology group, every entry point is loud
    // -----------------------------------------------------------------------

    #[test]
    fn missing_topology_group_loud_partials_every_entry_point() {
        // The documented normal case for today's converter output.
        let file = ten_vad_gguf(Some(LicenseClass::Permissive), None);
        let m = TenVad::from_gguf(&file).expect("bind");
        assert!(m.config().is_none());

        let expect_axes_error = |err: VokraError, surface: &str| match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains("`vokra.ten_vad.*`"),
                    "{surface}: message must name the missing metadata group: {msg}"
                );
                for key in TOPOLOGY_GROUP_KEYS {
                    assert!(msg.contains(key), "{surface}: key `{key}` listed: {msg}");
                }
                assert!(
                    msg.contains("FR-EX-08"),
                    "{surface}: message must cite FR-EX-08: {msg}"
                );
            }
            other => panic!("{surface}: expected VokraError::UnsupportedOp, got {other:?}"),
        };

        let Err(err) = m.frame_probability(&[0.0; 8]) else {
            panic!("frame_probability must loud-partial without topology axes");
        };
        expect_axes_error(err, "frame_probability");

        let mut stream = m.open_stream();
        let Err(err) = stream.push_pcm(&[0.0; 8], 16_000) else {
            panic!("push_pcm must loud-partial without topology axes");
        };
        expect_axes_error(err, "push_pcm");
    }

    // -----------------------------------------------------------------------
    // Test 14 — frame-size mismatch is a loud InvalidArgument, and a
    //           correctly-sized frame reaches the deferred forward
    // -----------------------------------------------------------------------

    #[test]
    fn frame_probability_rejects_wrong_frame_size_then_loud_partials() {
        let cfg = synthetic_topology(); // hop_size = 8
        let file = ten_vad_gguf(Some(LicenseClass::Permissive), Some(cfg));
        let m = TenVad::from_gguf(&file).expect("bind");

        // Too short: never zero-padded up to the frame size.
        let Err(err) = m.frame_probability(&[0.0; 5]) else {
            panic!("expected InvalidArgument on a short frame");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains('5') && msg.contains('8'),
                    "message must name both the got and expected frame size, got `{msg}`"
                );
                assert!(
                    msg.contains("pad / truncate"),
                    "message must state the no-silent-pad rule, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }

        // Too long: never truncated down to the frame size either.
        assert!(
            matches!(
                m.frame_probability(&[0.0; 9]),
                Err(VokraError::InvalidArgument(_))
            ),
            "an over-long frame must be rejected loudly, never truncated"
        );

        // Exactly one frame: passes the shape gate, hits the deferred forward.
        let Err(VokraError::UnsupportedOp(msg)) = m.frame_probability(&[0.0; 8]) else {
            panic!("a correctly-sized frame must reach the loud-partial forward");
        };
        assert!(
            msg.contains("ten_vad frame forward"),
            "surface named: {msg}"
        );
        assert!(msg.contains("loud-partial"), "posture label: {msg}");
    }

    // -----------------------------------------------------------------------
    // Test 15 — the binder plugs into the shared VadEngine seam, so callers
    //           can swap Silero / FSMN-VAD / TEN-VAD without rewriting
    // -----------------------------------------------------------------------

    #[test]
    fn binds_as_a_vad_engine_trait_object() {
        let cfg = synthetic_topology();
        let file = ten_vad_gguf(Some(LicenseClass::Permissive), Some(cfg));
        let m = TenVad::from_gguf(&file).expect("bind");

        // Exercising through `&dyn VadEngine` proves the object-safety and
        // the `Send` bound the shared seam requires.
        let engine: &dyn VadEngine = &m;
        let mut handle: Box<dyn VadStreamHandle + Send> = engine.open_stream();

        // A sub-frame push through the trait object behaves as documented.
        assert!(handle.push_pcm(&[0.0; 1], 16_000).unwrap().is_empty());
        handle.reset();
        // ...and a mismatched rate is still loud through the trait object.
        assert!(matches!(
            handle.push_pcm(&[0.0; 1], 44_100),
            Err(VokraError::InvalidArgument(_))
        ));
    }
}
