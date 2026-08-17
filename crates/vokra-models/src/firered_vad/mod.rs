//! **FireRedVAD** (Xiaohongshu FireRedTeam) — runtime binder for the
//! `firered_vad` converter arch (Wave B 2026-08-15 audit follow-up,
//! loud-partial per the RMVPE / NISQA / emotion2vec / panns / redimnet
//! precedent — CLAUDE.md 教訓 (a): 「loud-partial は fake-complete より
//! honest」).
//!
//! # The gap this closes
//!
//! `crates/vokra-convert/src/models/firered_vad.rs` (TIER 1 F wave,
//! 2026-07-30) writes a GGUF stamped `vokra.model.arch = "firered_vad"`,
//! `vokra.model.name = "firered-vad"`, `vokra.model.category = "vad"` and
//! `vokra.provenance.upstream_hf = "FireRedTeam/FireRedVAD"` — but a
//! workspace-wide grep proved that **nothing anywhere read that arch
//! string back**. Every converted FireRedVAD checkpoint was therefore
//! unloadable: the bytes were on disk in the right container with the
//! right provenance stamps, and no code path could turn them into a model
//! handle. This module is that missing consumer.
//!
//! # Primary sources
//!
//! - Upstream release: <https://huggingface.co/FireRedTeam/FireRedVAD>
//!   (recorded by the converter as fetched 2026-07-30 —
//!   CLAUDE.md「ハルシネーション厳禁」).
//! - Family reference code: <https://github.com/FireRedTeam/FireRedASR>
//!   (the FireRedTeam speech family that FireRedVAD ships alongside —
//!   FireRedASR / FireRedTTS).
//! - In-repo contract: `crates/vokra-convert/src/models/firered_vad.rs`
//!   — the GGUF writer whose `ARCH` / `NAME` / `CATEGORY` /
//!   `UPSTREAM_HF` constants this module mirrors verbatim.
//!
//! Note the provenance key: the converter stamps
//! `vokra.provenance.upstream_hf` (a HuggingFace slug), **not**
//! `vokra.provenance.upstream_url` — unlike the GitHub-only siblings
//! (NISQA / NSNet2 / RNNoise / DNSMOS) which have no HF mirror.
//!
//! # Architecture posture — why the forward is a loud-partial
//!
//! The converter's docstring describes FireRedVAD as a "transformer-based
//! streaming VAD" and stops there. It copies every F32 / F16 / BF16
//! tensor under its **verbatim upstream safetensors name** and stamps
//! **no** `vokra.firered_vad.*` hyper-parameter chunk at all. There is
//! consequently no in-repo transcription of
//!
//! - the front-end (sample rate, window / hop length, mel-band count),
//! - the encoder geometry (layer count, model width, head count, FFN
//!   width, whether the streaming mask is chunked or fully causal),
//! - the output-head width or its class ordering (which column is
//!   "speech"),
//!
//! and no `tools/parity/firered_vad_*` sidecar or parity harness exists
//! to recover them from a real checkpoint. Writing a "plausible"
//! transformer-VAD forward on top of that would be exactly the
//! silent-wrong failure mode CLAUDE.md 教訓 (a) forbids: a mis-guessed
//! head-count or a swapped speech column produces a *shape-valid*
//! probability vector that is quietly wrong, with no crash to catch it.
//!
//! The blocker here is **geometry, not kernels**: once the topology is
//! transcribed from a real checkpoint the encoder body composes from
//! transformer primitives Vokra already carries. That is why this module
//! ships the whole surround as real and defers exactly one thing.
//!
//! # Loud-partial classification (CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**:
//!   - [`FireredVad::from_gguf`] with **strict** `vokra.model.arch ==
//!     "firered_vad"` verification. A foreign GGUF — including every
//!     sibling `category = "vad"` arch tag — is refused loudly with the
//!     whole VAD fleet enumerated (see "Sibling family distinctness").
//!   - [`FireredVadWeights::from_gguf`] with a non-empty tensor-manifest
//!     gate: a GGUF carrying zero tensors is refused rather than bound
//!     into an all-zero forward (FR-EX-08).
//!   - The optional all-or-nothing [`FireredVadConfig`] group
//!     (`vokra.firered_vad.*`) — the hyper-parameter contract a future
//!     converter extension must stamp. Absent → [`FireredVad::config`]
//!     is `None` and the checkpoint still binds (that is the state of
//!     every GGUF today's converter produces, and refusing it would
//!     re-open the very gap this module closes). Partially stamped →
//!     loud [`VokraError::ModelLoad`] naming the missing key.
//!   - The optional [`KEY_REQUIRED_TENSORS`] manifest gate — when a
//!     producer declares the tensor names it wrote, a truncated or
//!     mis-merged GGUF fails at **load** time naming the first missing
//!     tensor, instead of surprising a forward halfway through.
//!   - [`FireredVadWeights::dims`] — a by-name tensor lookup that names
//!     the absent tensor in its error rather than returning `None` for a
//!     caller to swallow.
//!   - Sample-rate guarding: [`FireredVad::speech_probabilities`] and the
//!     [`VadEngine`] stream refuse a mismatched rate with
//!     [`VokraError::InvalidArgument`] — Vokra never silently resamples
//!     (FR-EX-08).
//!   - Weight-license surfacing, fail-closed to
//!     [`LicenseClass::Unknown`] when the stamp is absent.
//!
//! - **Loud-partial (this WP)**: [`FireredVad::speech_probabilities`] and
//!   [`FireredVadStream::push_pcm`] return [`VokraError::UnsupportedOp`]
//!   naming three concrete blockers — the missing topology transcription,
//!   the missing `vokra.firered_vad.*` metadata group, and the missing
//!   parity sidecar — plus both primary-source URLs so a reader
//!   diagnosing the gap has exactly two places to walk. **No fabricated
//!   speech probabilities are ever emitted** (FR-EX-08 — no silent
//!   partial output).
//!
//! # Sibling family distinctness (`category = "vad"` neighbourhood)
//!
//! [`ARCH`] = `"firered_vad"` is **deliberately distinct** from every
//! sibling VAD arch tag in the converter tree. They all answer the same
//! question ("is this frame speech?") through completely different
//! topologies, front-ends and output conventions:
//!
//! - `silero-vad` — Silero VAD v5, a 1:1-preserved dedicated subgraph
//!   (FR-LD-06): its ONNX `If`-branch topology plus internal h/c/context
//!   state cannot be lowered into graph-level ops at all;
//! - `fsmn-vad` — FunASR FSMN-VAD: Kaldi fbank + LFR frame stacking +
//!   global CMVN front-end feeding a feed-forward sequential memory
//!   network, output `[t_lfr, 2]`;
//! - `pyannote-segmentation` — pyannote/segmentation-3.0: a *speaker
//!   segmentation* model whose powerset output is reduced to a VAD
//!   signal, not a native binary speech head;
//! - `smart_turn` — pipecat-ai/smart-turn-v2: **end-of-turn** prediction
//!   (has the speaker finished?), a different question from frame-level
//!   speech presence;
//! - `ten_vad` — TEN VAD, which additionally sits in the distinct
//!   `category = "vad-kws"` bucket.
//!
//! Silently sharing an arch tag would let runtime dispatch mis-route a
//! FireRedVAD checkpoint onto one of those loaders; the tensor-name walk
//! would then fail with a downstream missing-tensor error (or, worse for
//! `pyannote-segmentation` / `smart_turn`, produce a shape-valid vector
//! answering a different question). FR-EX-08 forbids the silent misroute.
//!
//! # Licensing (owner-only sign-off — CC never signs)
//!
//! The converter stamps [`DEFAULT_LICENSE_SPDX`] = `apache-2.0` →
//! [`LicenseClass::Permissive`], and `vokra-core`'s license-class
//! hard-map carries a matching `firered-vad` / `fireredteam/firered`
//! entry. That SPDX is recorded by the converter on the stated basis
//! that the FireRedTeam family LICENSE pins Apache-2.0 across the team's
//! releases and that the FireRedVAD card inherits it — i.e. it is an
//! **inherited-family** determination, not a direct transcription of a
//! FireRedVAD-specific licence file. This binder therefore only
//! *surfaces* whatever class the GGUF carries and fail-closes to
//! [`LicenseClass::Unknown`] when nothing is stamped. The
//! `docs/license-audit.md` §3.1 sign-off row stays **BLANK** — owner-only
//! per memory `[[feedback-license-signoff-primary-source]]`; CC does not
//! sign it and does not treat the converter's default as a sign-off.
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_HF`] /
//! [`DEFAULT_LICENSE_SPDX`] are mirrors of the converter's own constants
//! — the same rule every sibling binder (`emotion2vec` / `nisqa` /
//! `panns` / `redimnet` / `sortformer_diar_4spk_v1` / `snac` / …) follows
//! so `vokra-models` does not gain a dependency edge onto
//! `vokra-convert`. The layered convention holds: `vokra-ops → nothing
//! GGUF-aware`, `vokra-core → GGUF reader`, `vokra-models → GGUF binder`,
//! `vokra-convert → GGUF writer`. The handshake is a plain string, pinned
//! on both sides by tests.
//!
//! # No ONNX / no pickle (permanent)
//!
//! The converter consumes safetensors only and this runtime never touches
//! an ONNX graph or a Python pickle (FR-LD-05 / NFR-DS-02). Any upstream
//! artefact that is not already flat safetensors must be bridged by an
//! offline `tools/parity/` sidecar (uv-managed Python 3.12 per memory
//! `[[feedback-python-uses-uv]]` + `[[feedback-python-3-12]]`), which is
//! never shipped as part of the `vokra-*` runtime.

use vokra_core::engines::{VadEngine, VadStreamHandle};
use vokra_core::gguf::{GgufFile, GgufMetadataValue, GgufValueType, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Contract constants — mirror of `crates/vokra-convert/src/models/firered_vad.rs`.
// See the module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model firered-vad`.
///
/// Deliberately distinct from every sibling `category = "vad"` arch tag
/// (`silero-vad`, `fsmn-vad`, `pyannote-segmentation`, `smart_turn`) and
/// from the `vad-kws` neighbour `ten_vad` — see the module docstring's
/// "Sibling family distinctness" section for why aliasing would be an
/// FR-EX-08 violation.
pub const ARCH: &str = "firered_vad";

/// Expected `vokra.model.name` value written by the converter.
///
/// Note the hyphen: the *name* is `firered-vad` while the *arch* is
/// `firered_vad` (underscore). Both spellings are load-bearing on the
/// wire, so they are pinned separately by a test.
pub const NAME: &str = "firered-vad";

/// Expected `vokra.model.category` value — `"vad"`, shared with the
/// sibling voice-activity detectors (`silero-vad`, `fsmn-vad`,
/// `pyannote-segmentation`, `smart_turn`). The VAD load-path selector
/// reads this rather than the arch tag.
pub const CATEGORY: &str = "vad";

/// Upstream HuggingFace slug recorded under
/// [`KEY_PROVENANCE_UPSTREAM_HF`].
pub const UPSTREAM_HF: &str = "FireRedTeam/FireRedVAD";

/// Default upstream weight licence (SPDX), mirrored from the converter.
/// Resolves to [`LicenseClass::Permissive`].
///
/// See the module docstring's "Licensing" section: this is the
/// converter's *inherited-family* determination, not a sign-off. The
/// `docs/license-audit.md` §3.1 row stays blank (owner-only).
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

/// GGUF metadata key: model category tag (mirror of the converter's
/// module-private const).
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// GGUF metadata key: upstream HuggingFace slug (mirror of the
/// converter's module-private const). FireRedVAD ships on HF, so
/// provenance rides `upstream_hf` rather than `upstream_url`.
pub const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

// ---------------------------------------------------------------------------
// Primary-source anchors — cited verbatim in the loud-partial message so a
// reader diagnosing the gap has fully specified places to walk.
// ---------------------------------------------------------------------------

/// Primary-source anchor: the upstream HuggingFace release.
pub const PRIMARY_SOURCE_HF: &str = "huggingface.co/FireRedTeam/FireRedVAD";

/// Primary-source anchor: the FireRedTeam speech-family reference code
/// that FireRedVAD ships alongside (FireRedASR / FireRedTTS).
pub const PRIMARY_SOURCE_FAMILY_CODE: &str = "github.com/FireRedTeam/FireRedASR";

/// In-repo contract anchor: the converter this binder mirrors.
pub const CONVERTER_PATH: &str = "crates/vokra-convert/src/models/firered_vad.rs";

/// The offline sidecar that does not exist yet. It is the place a real
/// checkpoint's topology must be transcribed from and the
/// [`FIREREDVAD_SPEC_KEYS`] group emitted, mirroring the sibling
/// `tools/parity/*_prepare_checkpoint.py` bridges. Never shipped inside
/// the `vokra-*` runtime (NFR-DS-02).
pub const SIDECAR_PATH: &str = "tools/parity/firered_vad_prepare_checkpoint.py";

// ---------------------------------------------------------------------------
// `vokra.firered_vad.*` — the optional, all-or-nothing hyper-parameter group.
//
// NOT stamped by today's converter. These keys are the contract a future
// converter extension (or `SIDECAR_PATH`) must satisfy; declaring them here
// is what lets `from_gguf` verify a stamped group instead of silently
// defaulting one half of it (FR-EX-08).
// ---------------------------------------------------------------------------

/// Sample rate the checkpoint expects, in Hz. Load-bearing: the binder
/// refuses PCM pushed at any other rate rather than resampling silently.
pub const KEY_SAMPLE_RATE: &str = "vokra.firered_vad.sample_rate";
/// Front-end mel-band count per analysis frame.
pub const KEY_N_MELS: &str = "vokra.firered_vad.n_mels";
/// Front-end analysis window length, in **samples**.
pub const KEY_WINDOW_LENGTH: &str = "vokra.firered_vad.window_length";
/// Front-end hop (frame shift), in **samples**. Together with
/// [`KEY_SAMPLE_RATE`] this fixes the per-frame output rate that
/// [`FireredVadConfig::frame_rate_hz`] reports.
pub const KEY_HOP_LENGTH: &str = "vokra.firered_vad.hop_length";
/// Transformer encoder depth (block count).
pub const KEY_N_LAYERS: &str = "vokra.firered_vad.n_layers";
/// Transformer encoder model width.
pub const KEY_D_MODEL: &str = "vokra.firered_vad.d_model";
/// Transformer encoder attention-head count. Invisible in the weight
/// shapes whenever QKV is packed into one projection, which is why it
/// needs a metadata key of its own.
pub const KEY_N_HEADS: &str = "vokra.firered_vad.n_heads";
/// Transformer encoder feed-forward inner width.
pub const KEY_FFN_DIM: &str = "vokra.firered_vad.ffn_dim";
/// Output-head class count (e.g. `2` for a `[silence, speech]` softmax
/// head, `1` for a single sigmoid logit). Load-bearing for how a
/// per-frame speech probability is read out of the head.
pub const KEY_N_CLASS: &str = "vokra.firered_vad.n_class";

/// The hyper-parameter group in canonical read order — **all-or-nothing**.
///
/// [`FireredVadConfig::from_gguf`] returns `Ok(None)` when *no* key is
/// present and a loud [`VokraError::ModelLoad`] when only *some* are:
/// silently defaulting the missing half would produce a wrong-shaped
/// front-end or encoder with no crash to catch it (FR-EX-08).
pub const FIREREDVAD_SPEC_KEYS: [&str; 9] = [
    KEY_SAMPLE_RATE,
    KEY_N_MELS,
    KEY_WINDOW_LENGTH,
    KEY_HOP_LENGTH,
    KEY_N_LAYERS,
    KEY_D_MODEL,
    KEY_N_HEADS,
    KEY_FFN_DIM,
    KEY_N_CLASS,
];

/// Optional `Array<String>` metadata key: the exact tensor names the
/// producer wrote.
///
/// When present, [`FireredVad::from_gguf`] verifies every listed name is
/// in the manifest and fails loud naming the first absent one. This turns
/// a truncated / mis-merged / partially-uploaded GGUF into a **load-time**
/// failure instead of a surprise halfway through a forward. Absent →
/// skipped entirely (today's converter does not stamp it), because
/// requiring it would re-open the unloadable-checkpoint gap this module
/// exists to close.
pub const KEY_REQUIRED_TENSORS: &str = "vokra.firered_vad.required_tensors";

// ---------------------------------------------------------------------------
// Metadata read helpers.
// ---------------------------------------------------------------------------

/// `true` when **any** key of an all-or-nothing group is present.
fn group_present(gguf: &GgufFile, keys: &[&str]) -> bool {
    keys.iter().any(|k| gguf.get(k).is_some())
}

/// Reads a required unsigned-integer key, refusing a wrong value type
/// rather than coercing it (FR-EX-08).
fn read_u32_key(gguf: &GgufFile, key: &str) -> Result<u32> {
    let raw = gguf.get(key).and_then(|v| v.as_u64()).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "firered-vad: GGUF metadata `{key}` is missing or is not an unsigned \
             integer. The `vokra.firered_vad.*` group is all-or-nothing — a \
             partially stamped group is a bug in `{CONVERTER_PATH}` (or in \
             `{SIDECAR_PATH}`), and silently defaulting the missing half would \
             produce a wrong-shaped front-end / encoder with no crash to catch it \
             (FR-EX-08)."
        ))
    })?;
    u32::try_from(raw).map_err(|_| {
        VokraError::ModelLoad(format!(
            "firered-vad: GGUF metadata `{key}` = {raw} does not fit in u32"
        ))
    })
}

/// Reads the optional [`KEY_REQUIRED_TENSORS`] declaration.
///
/// Returns `Ok(None)` when the key is absent. Refuses a wrong container
/// type, a wrong element type, a non-string element, or an empty list —
/// an empty declaration asserts nothing and is always a producer bug.
fn read_required_tensors(gguf: &GgufFile) -> Result<Option<Vec<String>>> {
    let Some(value) = gguf.get(KEY_REQUIRED_TENSORS) else {
        return Ok(None);
    };
    let arr = value.as_array().ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "firered-vad: GGUF metadata `{KEY_REQUIRED_TENSORS}` is not an array \
             (expected Array<String> naming the tensors the producer wrote), got \
             {:?}",
            value.value_type()
        ))
    })?;
    if arr.element_type != GgufValueType::String {
        return Err(VokraError::ModelLoad(format!(
            "firered-vad: GGUF metadata `{KEY_REQUIRED_TENSORS}` has element_type \
             {:?}, expected String",
            arr.element_type
        )));
    }
    let mut out = Vec::with_capacity(arr.values.len());
    for (i, v) in arr.values.iter().enumerate() {
        match v {
            GgufMetadataValue::String(s) => out.push(s.clone()),
            other => {
                return Err(VokraError::ModelLoad(format!(
                    "firered-vad: GGUF metadata `{KEY_REQUIRED_TENSORS}[{i}]` is not \
                     a string (got {:?})",
                    other.value_type()
                )));
            }
        }
    }
    if out.is_empty() {
        return Err(VokraError::ModelLoad(format!(
            "firered-vad: GGUF metadata `{KEY_REQUIRED_TENSORS}` is an empty list — \
             an empty required-tensor declaration asserts nothing, so stamping it is \
             always a producer bug. Omit the key entirely, or list the tensor names \
             `{CONVERTER_PATH}` actually wrote (FR-EX-08)."
        )));
    }
    Ok(Some(out))
}

// ---------------------------------------------------------------------------
// FireredVadConfig — the optional `vokra.firered_vad.*` hyper-parameter group.
// ---------------------------------------------------------------------------

/// FireRedVAD hyper-parameters, read from the optional all-or-nothing
/// `vokra.firered_vad.*` group.
///
/// **Absent from every GGUF today's converter produces.** The struct is
/// the flip-the-switch contract: once the converter ([`CONVERTER_PATH`])
/// or [`SIDECAR_PATH`] transcribes a real checkpoint's topology and stamps
/// these nine keys, [`FireredVad::config`] starts returning `Some` and the
/// sample-rate guard becomes enforceable.
///
/// Every field is a plain positive integer, validated by
/// [`Self::validate`]; no field carries a default, because Vokra has no
/// primary-source transcription of FireRedVAD's real values and inventing
/// one would be the exact silent-wrong failure this module refuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FireredVadConfig {
    /// Sample rate the checkpoint expects, in Hz ([`KEY_SAMPLE_RATE`]).
    pub sample_rate: u32,
    /// Front-end mel-band count ([`KEY_N_MELS`]).
    pub n_mels: u32,
    /// Front-end analysis window length in samples
    /// ([`KEY_WINDOW_LENGTH`]).
    pub window_length: u32,
    /// Front-end hop in samples ([`KEY_HOP_LENGTH`]).
    pub hop_length: u32,
    /// Transformer encoder depth ([`KEY_N_LAYERS`]).
    pub n_layers: u32,
    /// Transformer encoder model width ([`KEY_D_MODEL`]).
    pub d_model: u32,
    /// Transformer encoder attention-head count ([`KEY_N_HEADS`]).
    pub n_heads: u32,
    /// Transformer encoder feed-forward inner width ([`KEY_FFN_DIM`]).
    pub ffn_dim: u32,
    /// Output-head class count ([`KEY_N_CLASS`]).
    pub n_class: u32,
}

impl FireredVadConfig {
    /// Validates the group loudly (FR-EX-08).
    ///
    /// Two classes of check, both universal to any transformer VAD and
    /// therefore safe to assert without a FireRedVAD-specific
    /// transcription:
    ///
    /// 1. every field must be `> 0` — a `0` is the classic
    ///    half-populated-metadata sentinel, and a zero window / hop /
    ///    width collapses the whole pipeline;
    /// 2. `d_model % n_heads == 0` — multi-head attention splits the
    ///    model width across heads, so an indivisible pair can only come
    ///    from a mis-stamp.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] naming the offending key.
    pub fn validate(&self) -> Result<()> {
        for (key, value) in [
            (KEY_SAMPLE_RATE, self.sample_rate),
            (KEY_N_MELS, self.n_mels),
            (KEY_WINDOW_LENGTH, self.window_length),
            (KEY_HOP_LENGTH, self.hop_length),
            (KEY_N_LAYERS, self.n_layers),
            (KEY_D_MODEL, self.d_model),
            (KEY_N_HEADS, self.n_heads),
            (KEY_FFN_DIM, self.ffn_dim),
            (KEY_N_CLASS, self.n_class),
        ] {
            if value == 0 {
                return Err(VokraError::ModelLoad(format!(
                    "firered-vad: `{key}` = 0 — every `vokra.firered_vad.*` \
                     hyper-parameter must be positive. A zero is the classic \
                     half-populated-metadata sentinel; accepting it would build a \
                     collapsed front-end / encoder that still runs (FR-EX-08)."
                )));
            }
        }
        if self.d_model % self.n_heads != 0 {
            return Err(VokraError::ModelLoad(format!(
                "firered-vad: `{KEY_D_MODEL}` = {d} is not divisible by \
                 `{KEY_N_HEADS}` = {h} — multi-head attention splits the model \
                 width evenly across heads, so an indivisible pair can only come \
                 from a mis-stamped group. Refusing rather than truncating the \
                 per-head width (FR-EX-08).",
                d = self.d_model,
                h = self.n_heads,
            )));
        }
        Ok(())
    }

    /// Per-head attention width (`d_model / n_heads`).
    ///
    /// Every field of this struct is public, so a caller can hand-build a
    /// config that never went through [`Self::validate`]. The `n_heads ==
    /// 0` arm returns `0` rather than dividing — a diagnostic accessor
    /// must never be the thing that panics, least of all inside
    /// [`forward_loud_partial`], whose whole job is to report a problem.
    /// A validated config can never take that arm.
    #[inline]
    #[must_use]
    pub const fn head_dim(&self) -> u32 {
        // `match` rather than `unwrap_or`, which is not const-callable.
        match self.d_model.checked_div(self.n_heads) {
            Some(v) => v,
            None => 0,
        }
    }

    /// The per-frame output rate in Hz (`sample_rate / hop_length`) — the
    /// rate at which [`FireredVad::speech_probabilities`] will emit
    /// probabilities once the forward lands.
    #[inline]
    #[must_use]
    pub fn frame_rate_hz(&self) -> f32 {
        self.sample_rate as f32 / self.hop_length as f32
    }

    /// Reads the group from a parsed GGUF.
    ///
    /// Returns `Ok(None)` when **no** key of the group is present — the
    /// state of every GGUF today's converter produces. Returns a loud
    /// [`VokraError::ModelLoad`] when the group is only partially
    /// stamped, when a value has the wrong type, or when
    /// [`Self::validate`] fails.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] on a partial group, a wrong value
    ///   type, or a failed validation.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Option<Self>> {
        if !group_present(gguf, &FIREREDVAD_SPEC_KEYS) {
            return Ok(None);
        }
        let cfg = Self {
            sample_rate: read_u32_key(gguf, KEY_SAMPLE_RATE)?,
            n_mels: read_u32_key(gguf, KEY_N_MELS)?,
            window_length: read_u32_key(gguf, KEY_WINDOW_LENGTH)?,
            hop_length: read_u32_key(gguf, KEY_HOP_LENGTH)?,
            n_layers: read_u32_key(gguf, KEY_N_LAYERS)?,
            d_model: read_u32_key(gguf, KEY_D_MODEL)?,
            n_heads: read_u32_key(gguf, KEY_N_HEADS)?,
            ffn_dim: read_u32_key(gguf, KEY_FFN_DIM)?,
            n_class: read_u32_key(gguf, KEY_N_CLASS)?,
        };
        cfg.validate()?;
        Ok(Some(cfg))
    }
}

// ---------------------------------------------------------------------------
// FireredVadWeights — tensor manifest with a non-empty gate.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a FireRedVAD GGUF.
///
/// Names + GGUF-side dims only. The forward is a loud-partial, so
/// preloading every tensor into RAM would buy nothing; the follow-up wave
/// that lands the real forward picks its own caching shape. What this
/// struct *does* buy today is the two loud gates the module docstring
/// promises: the non-empty manifest check and the by-name lookup that
/// names an absent tensor.
#[derive(Debug)]
pub struct FireredVadWeights {
    tensors: Vec<(String, Vec<usize>)>,
}

impl FireredVadWeights {
    /// Scans `gguf` for the FireRedVAD `state_dict` tensors, refusing an
    /// empty manifest (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let mut tensors: Vec<(String, Vec<usize>)> = Vec::new();
        for info in gguf.tensors() {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            tensors.push((info.name.clone(), dims));
        }
        if tensors.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "firered-vad: GGUF carries zero tensors — refusing to bind an \
                 all-zero forward (FR-EX-08). A legitimate FireRedVAD checkpoint \
                 carries the front-end plus every transformer encoder block's \
                 attention / feed-forward / normalisation parameters (arch={ARCH}, \
                 name={NAME}); zero tensors always signals a mis-produced GGUF. \
                 Re-run `vokra-cli convert --model firered-vad` against an upstream \
                 `{UPSTREAM_HF}` safetensors checkpoint."
            )));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// The bound tensor names with their GGUF-side dims — a diagnostic
    /// accessor for the follow-up forward wave.
    #[inline]
    #[must_use]
    pub fn tensors(&self) -> &[(String, Vec<usize>)] {
        &self.tensors
    }

    /// `true` when `name` is present in the bound manifest.
    #[must_use]
    pub fn has(&self, name: &str) -> bool {
        self.tensors.iter().any(|(n, _)| n.as_str() == name)
    }

    /// Looks a tensor's GGUF-side dims up by name.
    ///
    /// Returns a loud [`VokraError::ModelLoad`] **naming the absent
    /// tensor** rather than `None`, so a caller cannot swallow the miss
    /// with `unwrap_or_default()` and silently proceed on an implicit
    /// zero shape (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `name` is not in the manifest.
    pub fn dims(&self, name: &str) -> Result<&[usize]> {
        self.tensors
            .iter()
            .find(|(n, _)| n.as_str() == name)
            .map(|(_, d)| d.as_slice())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "firered-vad: tensor `{name}` is absent from the GGUF manifest \
                     ({count} tensors present). FireRedVAD GGUFs carry the upstream \
                     safetensors names verbatim (see `{CONVERTER_PATH}`), so a miss \
                     means either a mis-produced GGUF or a stale name in the caller \
                     (FR-EX-08 — no silent zero-shape fallback).",
                    count = self.tensors.len()
                ))
            })
    }

    /// Verifies every name in a [`KEY_REQUIRED_TENSORS`] declaration is
    /// present, failing loud on the **first** absent one.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] naming the first missing tensor.
    pub fn require_all(&self, names: &[String]) -> Result<()> {
        for name in names {
            if !self.has(name) {
                return Err(VokraError::ModelLoad(format!(
                    "firered-vad: required tensor `{name}` is declared in \
                     `{KEY_REQUIRED_TENSORS}` but absent from the GGUF manifest \
                     ({count} tensors present, {declared} declared). The producer \
                     asserted it wrote this tensor, so the GGUF is truncated, \
                     mis-merged or partially uploaded — refusing at load time \
                     rather than surprising a forward halfway through (FR-EX-08).",
                    count = self.tensors.len(),
                    declared = names.len(),
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FireredVad — the runtime binder handle.
// ---------------------------------------------------------------------------

/// FireRedVAD (`FireRedTeam/FireRedVAD`) runtime binder — the consumer of
/// the `firered_vad` converter arch.
///
/// Bind with [`from_gguf`](Self::from_gguf) / [`from_path`](Self::from_path),
/// then obtain per-frame speech probabilities either one-shot through
/// [`speech_probabilities`](Self::speech_probabilities) or streaming
/// through the [`VadEngine`] trait — the same API shape the sibling VAD
/// binders expose, so the `stream::open_stream` glue in `vokra-core` sees
/// no FireRedVAD-vs-FSMN-vs-Silero asymmetry.
///
/// Both entry points are loud-partials today; see the module doc for
/// exactly which three pieces are missing and why guessing them would be
/// silent-wrong.
#[derive(Debug)]
pub struct FireredVad {
    /// The `vokra.firered_vad.*` group when stamped. `None` for every
    /// GGUF today's converter produces — see [`FireredVadConfig`].
    cfg: Option<FireredVadConfig>,
    weights: FireredVadWeights,
    weight_license: LicenseClass,
}

impl FireredVad {
    /// Binds a FireRedVAD GGUF: verifies the arch tag strictly, binds the
    /// tensor manifest, honours an optional required-tensor declaration,
    /// reads the optional `vokra.firered_vad.*` group, and surfaces the
    /// stamped weight-license class.
    ///
    /// Every failure is a distinct [`VokraError::ModelLoad`] naming the
    /// missing / wrong key or tensor, so a reader diagnosing a
    /// mis-produced GGUF has exactly one place to walk (FR-EX-08 — never
    /// a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent or is
    ///   not [`ARCH`] (a sibling `category = "vad"` GGUF handed here by
    ///   mistake fails with a specific message rather than a downstream
    ///   missing-tensor error);
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors;
    /// - [`VokraError::ModelLoad`] when a [`KEY_REQUIRED_TENSORS`]
    ///   declaration names a tensor that is not in the manifest;
    /// - [`VokraError::ModelLoad`] when the `vokra.firered_vad.*` group is
    ///   partially stamped or fails [`FireredVadConfig::validate`].
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check first, always — a `silero-vad` / `fsmn-vad` /
        //    `pyannote-segmentation` / `smart_turn` / `ten_vad` GGUF
        //    handed here by mistake must fail with a clear message, not a
        //    downstream shape error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "firered-vad: GGUF arch is `{other}`, expected `{ARCH}` (was \
                     this GGUF produced by `vokra-cli convert --model \
                     firered-vad`? The sibling `category = \"vad\"` arch tags — \
                     `silero-vad` (Silero VAD v5, a 1:1-preserved subgraph whose \
                     ONNX If-branch topology and h/c/context state cannot be \
                     lowered, FR-LD-06), `fsmn-vad` (FunASR feed-forward \
                     sequential memory network over a Kaldi fbank + LFR + CMVN \
                     front-end), `pyannote-segmentation` (pyannote/segmentation-3.0 \
                     speaker segmentation reduced to a VAD signal, not a native \
                     binary speech head), `smart_turn` (pipecat-ai/smart-turn-v2 \
                     end-of-turn prediction — a different question from frame-level \
                     speech presence) — plus the `vad-kws` neighbour `ten_vad`, all \
                     live in the same neighbourhood but have completely different \
                     topologies, front-ends and output conventions. FireRedVAD's \
                     transformer streaming topology has no analog in any sibling; \
                     silently aliasing arch would mis-route runtime dispatch \
                     (FR-EX-08 — no silent partial load)."
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(format!(
                    "firered-vad: GGUF is missing `vokra.model.arch` — this is not \
                     a Vokra-native firered_vad GGUF (was it produced by \
                     `vokra-cli convert --model firered-vad`? that converter, \
                     `{CONVERTER_PATH}`, stamps `vokra.model.arch = {ARCH}`)."
                )));
            }
        }

        // 2. Tensor manifest with the non-emptiness gate, then the
        //    optional producer-declared required-tensor check.
        let weights = FireredVadWeights::from_gguf(file)?;
        if let Some(required) = read_required_tensors(file)? {
            weights.require_all(&required)?;
        }

        // 3. The optional all-or-nothing hyper-parameter group. `None`
        //    here is the normal state today (the converter stamps no
        //    `vokra.firered_vad.*` keys) and must NOT be a load failure —
        //    refusing it would re-open the unloadable-checkpoint gap this
        //    module closes.
        let cfg = FireredVadConfig::from_gguf(file)?;

        // 4. Provenance surfacing. The converter stamps `apache-2.0` →
        //    Permissive by default; a GGUF with no stamp fail-closes to
        //    Unknown (memory `[[feedback-license-signoff-primary-source]]`).
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        Ok(Self {
            cfg,
            weights,
            weight_license,
        })
    }

    /// Opens and binds the model from a GGUF file on disk.
    ///
    /// # Errors
    ///
    /// - Whatever [`GgufFile::open`] returns, plus every error of
    ///   [`Self::from_gguf`].
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// The `vokra.firered_vad.*` hyper-parameter group, when stamped.
    ///
    /// `None` for every GGUF today's converter produces — see
    /// [`FireredVadConfig`] for the flip-the-switch contract.
    // Deliberately not `const fn`: `Option::as_ref` in a const context is
    // newer than this workspace's MSRV floor is worth betting on, and no
    // caller needs a const config accessor.
    #[inline]
    #[must_use]
    pub fn config(&self) -> Option<&FireredVadConfig> {
        self.cfg.as_ref()
    }

    /// The bound tensor manifest.
    #[inline]
    #[must_use]
    pub const fn weights(&self) -> &FireredVadWeights {
        &self.weights
    }

    /// Number of tensors bound from the GGUF.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// The stamped weight-license class.
    ///
    /// The converter stamps [`DEFAULT_LICENSE_SPDX`] (`apache-2.0`) →
    /// [`LicenseClass::Permissive`] by default; a GGUF without the stamp
    /// reads back as [`LicenseClass::Unknown`] (fail-closed at the M2-13
    /// compliance gate). This is a *surface*, not a sign-off — see the
    /// module docstring's "Licensing" section.
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// `true` when the bound weights may only be used behind a research
    /// flag. An unstamped GGUF answers `true` (fail-closed).
    #[inline]
    #[must_use]
    pub fn is_research_only(&self) -> bool {
        self.weight_license.requires_research_flag()
    }

    /// Per-frame speech probabilities for a mono `f32` PCM clip at
    /// `sample_rate` Hz — the one-shot analogue of the streaming
    /// [`VadEngine`] path, and the same API shape the sibling VAD binders
    /// expose.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`]. FireRedVAD's topology is
    /// not primary-source-transcribable from anything in this repository
    /// — see [`forward_loud_partial`] for the full message and the
    /// flip-the-switch recipe. **No fabricated speech probabilities are
    /// ever emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] when `pcm` is empty;
    /// - [`VokraError::InvalidArgument`] when the `vokra.firered_vad.*`
    ///   group is stamped and `sample_rate` differs from its
    ///   [`FireredVadConfig::sample_rate`] — Vokra never silently
    ///   resamples (FR-EX-08);
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate.
    pub fn speech_probabilities(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "firered-vad: empty PCM slice — a voice-activity decision needs at \
                 least one sample. Returning an empty probability vector would be \
                 indistinguishable from 'no speech detected' (FR-EX-08)."
                    .to_owned(),
            ));
        }
        check_sample_rate(self.cfg.as_ref(), sample_rate)?;
        // The gate fires BEFORE any front-end work so a caller can never
        // observe a partial computation that looks like a real forward.
        Err(forward_loud_partial(self.cfg.as_ref()))
    }
}

impl VadEngine for FireredVad {
    fn open_stream(&self) -> Box<dyn VadStreamHandle + Send> {
        Box::new(FireredVadStream { cfg: self.cfg })
    }
}

/// Stateful FireRedVAD stream — the streaming half of the VAD API, shaped
/// exactly like [`crate::fsmn_vad::FsmnVadStream`] and the Silero v5
/// handle so `vokra-core`'s `stream::open_stream` glue sees one shape.
///
/// It carries **no recurrent state yet**: the encoder's streaming caches
/// only exist once the topology transcription lands (see the module
/// doc). Every [`push_pcm`](VadStreamHandle::push_pcm) therefore either
/// rejects the sample rate or returns the loud-partial — it never returns
/// a plausible-looking empty vector, because an empty return is
/// indistinguishable from "no frame completed" and would let a caller
/// loop forever believing the VAD is running.
#[derive(Debug)]
pub struct FireredVadStream {
    cfg: Option<FireredVadConfig>,
}

impl VadStreamHandle for FireredVadStream {
    fn push_pcm(&mut self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        // Rate first: a mismatched rate is a caller bug worth reporting
        // even while the forward itself is deferred, and reporting it
        // here keeps the guard identical to the one-shot path.
        check_sample_rate(self.cfg.as_ref(), sample_rate)?;
        let _ = pcm;
        Err(forward_loud_partial(self.cfg.as_ref()))
    }

    fn reset(&mut self) {
        // No recurrent state exists yet — the encoder's streaming caches
        // land with the topology transcription. Deliberately a no-op
        // rather than `unimplemented!()`: `reset` is infallible in the
        // trait, and panicking on a handle whose forward already fails
        // loudly would add nothing but a crash.
    }
}

/// Shared sample-rate guard for both VAD entry points.
///
/// A `None` config means the `vokra.firered_vad.*` group is unstamped, so
/// there is no declared rate to compare against — the caller falls
/// through to the loud-partial, whose message names the missing group as
/// blocker (2). Inventing a rate to compare against would be a fabricated
/// hyper-parameter.
fn check_sample_rate(cfg: Option<&FireredVadConfig>, sample_rate: u32) -> Result<()> {
    let Some(cfg) = cfg else {
        return Ok(());
    };
    if sample_rate != cfg.sample_rate {
        return Err(VokraError::InvalidArgument(format!(
            "firered-vad: sample rate mismatch — pushed {sample_rate} Hz but the \
             checkpoint declares `{KEY_SAMPLE_RATE}` = {expected} Hz. The front-end \
             window / hop ({KEY_WINDOW_LENGTH} = {win}, {KEY_HOP_LENGTH} = {hop} \
             samples) are sample counts fixed against that rate, so feeding another \
             rate silently rescales every analysis window. Resample upstream, or \
             open a stream on the matching rate — Vokra never resamples implicitly \
             (FR-EX-08).",
            expected = cfg.sample_rate,
            win = cfg.window_length,
            hop = cfg.hop_length,
        )));
    }
    Ok(())
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned by
/// every FireRedVAD forward path until the missing pieces land.
///
/// The message names all three blockers so a reader (or the follow-up
/// wave) knows exactly where to flip the switch:
///
/// 1. **The missing topology transcription** — the converter describes
///    FireRedVAD only as a "transformer-based streaming VAD"; there is no
///    in-repo transcription of the front-end, encoder geometry or output
///    head, and no parity harness to recover one. A best-guess topology
///    would emit a shape-valid probability vector that is quietly wrong.
/// 2. **The missing metadata** — the [`FIREREDVAD_SPEC_KEYS`] group,
///    which [`CONVERTER_PATH`] does not stamp today. Head count in
///    particular is invisible in the weight shapes whenever QKV is
///    packed, so it cannot be recovered from the manifest.
/// 3. **The missing sidecar** — [`SIDECAR_PATH`], the offline bridge that
///    must transcribe a real checkpoint and emit the group. It has never
///    been written, and no Python ever enters the runtime (FR-LD-05 /
///    NFR-DS-02).
#[must_use]
pub fn forward_loud_partial(cfg: Option<&FireredVadConfig>) -> VokraError {
    let spec_status = match cfg {
        Some(c) => format!(
            "the `vokra.firered_vad.*` group IS stamped on this GGUF \
             (sample_rate={sr} Hz, n_mels={mels}, window_length={win}, \
             hop_length={hop} -> {rate:.2} frames/s, n_layers={layers}, \
             d_model={dm}, n_heads={nh} -> head_dim={hd}, ffn_dim={ffn}, \
             n_class={ncls}), so blocker (2) is already cleared for it — \
             blockers (1) and (3) still stand",
            sr = c.sample_rate,
            mels = c.n_mels,
            win = c.window_length,
            hop = c.hop_length,
            rate = c.frame_rate_hz(),
            layers = c.n_layers,
            dm = c.d_model,
            nh = c.n_heads,
            hd = c.head_dim(),
            ffn = c.ffn_dim,
            ncls = c.n_class,
        ),
        None => format!(
            "the `vokra.firered_vad.*` group is NOT stamped on this GGUF (the \
             normal state today — `{CONVERTER_PATH}` writes only arch / name / \
             category / provenance), so blocker (2) applies in full"
        ),
    };
    VokraError::UnsupportedOp(format!(
        "firered-vad speech_probabilities (loud-partial): the FireRedVAD forward is \
         deferred; three pieces must land before real per-frame speech \
         probabilities can be emitted. \
         (1) MISSING TOPOLOGY TRANSCRIPTION: `{CONVERTER_PATH}` describes the model \
         only as a `transformer-based streaming VAD` and copies every tensor under \
         its verbatim upstream safetensors name. Nothing in this repository \
         transcribes the front-end (window / hop / mel bands), the encoder geometry \
         (layer count, model width, head count, FFN width, chunked-vs-causal \
         streaming mask) or the output head width and class ordering. A best-guess \
         topology would produce a SHAPE-VALID probability vector that is quietly \
         wrong — a mis-guessed head count or a swapped speech column never crashes \
         (CLAUDE.md loud-partial precedent; the blocker is geometry, not kernels — \
         the encoder body composes from transformer primitives Vokra already \
         carries once the geometry is known). \
         (2) MISSING METADATA: the all-or-nothing `vokra.firered_vad.*` group \
         ({keys:?}); {spec_status}. `{KEY_N_HEADS}` in particular is invisible in \
         the weight shapes whenever QKV is packed into one projection, so it cannot \
         be recovered from the tensor manifest at all. \
         (3) MISSING SIDECAR: `{SIDECAR_PATH}` does not exist — it must transcribe a \
         real checkpoint's topology and emit the group; no Python ever enters the \
         runtime (FR-LD-05 / NFR-DS-02). \
         Output once real: one speech probability per analysis frame, at \
         `{KEY_SAMPLE_RATE}` / `{KEY_HOP_LENGTH}` frames per second — the same \
         per-frame contract the sibling `fsmn-vad` and `silero-vad` binders honour. \
         Primary sources: HF release {hf}, family reference code {code}, in-repo \
         converter contract {CONVERTER_PATH}. Runtime cannot fabricate speech \
         probabilities (FR-EX-08 — no silent partial output).",
        keys = FIREREDVAD_SPEC_KEYS,
        hf = PRIMARY_SOURCE_HF,
        code = PRIMARY_SOURCE_FAMILY_CODE,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the FireRedVAD runtime binder — contract-constant pins
    //! (the cross-crate string handshake with the converter), metadata
    //! round-trip, and negative-space round-trip on every loud gate.
    //!
    //! # What "round-trip" means here
    //!
    //! On a real checkpoint this would be `speech_probabilities(...)`
    //! returning one probability per analysis frame, but FireRedVAD's
    //! topology is not primary-source-transcribable from anything in this
    //! repository (see the module doc). Fabricating a probability vector
    //! would violate CLAUDE.md 教訓 (a)「loud-partial は fake-complete
    //! より honest」.
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Contract-constant pin** — `ARCH` / `NAME` / `CATEGORY` /
    //!    `UPSTREAM_HF` / `DEFAULT_LICENSE_SPDX` match the converter's
    //!    values exactly, and the arch tag is distinct from every sibling
    //!    VAD arch tag.
    //! 2. **Metadata round-trip** — a synthetic GGUF binds, its category /
    //!    provenance / licence stamps read back, and the optional
    //!    `vokra.firered_vad.*` group round-trips field-for-field.
    //! 3. **Loud negative space** — missing arch, wrong arch, empty
    //!    manifest, partial group, zero sentinel, indivisible
    //!    `d_model % n_heads`, declared-but-absent tensor, by-name lookup
    //!    miss, sample-rate mismatch, and the loud-partial forward each
    //!    fire at their documented surface point in their documented
    //!    error variant.
    //!
    //! **On fixture values**: the `vokra.firered_vad.*` numbers below are
    //! SYNTHETIC. They are deliberately NOT a claim about the real
    //! FireRedVAD release — no in-repo transcription of its topology
    //! exists, which is precisely why the forward is a loud-partial. They
    //! exist only to exercise the reader and the validator.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufArray, GgufBuilder};

    /// Synthetic `vokra.firered_vad.*` fixture in canonical key order.
    ///
    /// NOT FireRedVAD's real hyper-parameters — see the module-test doc.
    /// `d_model = 256` / `n_heads = 4` is chosen divisible so the happy
    /// path passes `validate`.
    const FIXTURE_SPEC: [(&str, u32); 9] = [
        (KEY_SAMPLE_RATE, 16_000),
        (KEY_N_MELS, 80),
        (KEY_WINDOW_LENGTH, 400),
        (KEY_HOP_LENGTH, 160),
        (KEY_N_LAYERS, 4),
        (KEY_D_MODEL, 256),
        (KEY_N_HEADS, 4),
        (KEY_FFN_DIM, 1024),
        (KEY_N_CLASS, 2),
    ];

    /// A representative tensor name. FireRedVAD GGUFs carry the upstream
    /// safetensors names verbatim; this mirrors the sample the converter's
    /// own test module uses so the two files stay legible together.
    const FIXTURE_TENSOR: &str = "encoder.layer0.attn.qkv.weight";

    /// Builds a base FireRedVAD GGUF: arch + name + category + upstream
    /// slug, an optional weight-license stamp, and one representative
    /// tensor so the non-emptiness gate passes.
    fn base_builder(weight_license_class: Option<LicenseClass>) -> GgufBuilder {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        b.add_tensor(
            FIXTURE_TENSOR,
            GgmlType::F32,
            vec![2, 3],
            vec![0u8; 2 * 3 * 4],
        )
        .expect("add_tensor");
        b
    }

    fn finish(b: &GgufBuilder) -> GgufFile {
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    /// A GGUF in the state today's converter actually produces: arch +
    /// provenance + tensors, and NO `vokra.firered_vad.*` group.
    fn converter_shaped_gguf() -> GgufFile {
        finish(&base_builder(Some(LicenseClass::Permissive)))
    }

    /// A GGUF with the full synthetic hyper-parameter group stamped.
    fn spec_stamped_gguf() -> GgufFile {
        let mut b = base_builder(Some(LicenseClass::Permissive));
        for (key, value) in FIXTURE_SPEC {
            b.add_u32(key, value);
        }
        finish(&b)
    }

    // -----------------------------------------------------------------------
    // Test 1 — Contract-constant pin (cross-crate handshake with the
    //          converter) + sibling arch-tag distinctness.
    // -----------------------------------------------------------------------

    #[test]
    fn contract_constants_pin_matches_converter() {
        // Mirrors of `crates/vokra-convert/src/models/firered_vad.rs`. A
        // converter-side drift without a binder-side follow-through lands
        // here in the same commit or fails this test.
        assert_eq!(ARCH, "firered_vad", "arch tag pin (underscore)");
        assert_eq!(NAME, "firered-vad", "model name pin (hyphen)");
        assert_eq!(CATEGORY, "vad", "category pin");
        assert_eq!(
            UPSTREAM_HF, "FireRedTeam/FireRedVAD",
            "upstream HF slug pin"
        );
        assert_eq!(DEFAULT_LICENSE_SPDX, "apache-2.0", "default SPDX pin");

        // The arch / name spellings genuinely differ — a "helpful"
        // normalisation of either would break the on-the-wire handshake.
        assert_ne!(
            ARCH, NAME,
            "arch uses `_`, name uses `-` — both are on-wire values"
        );

        // Distinct from every sibling VAD arch tag (module doc "Sibling
        // family distinctness"); aliasing any of these would mis-route
        // dispatch (FR-EX-08).
        for sibling in [
            "silero-vad",
            "fsmn-vad",
            "pyannote-segmentation",
            "smart_turn",
            "ten_vad",
        ] {
            assert_ne!(ARCH, sibling, "arch must stay distinct from `{sibling}`");
        }

        // The spec-key group is exactly nine keys, all under the arch's
        // own `vokra.firered_vad.` namespace (no cross-arch collision).
        assert_eq!(FIREREDVAD_SPEC_KEYS.len(), 9);
        for key in FIREREDVAD_SPEC_KEYS {
            assert!(
                key.starts_with("vokra.firered_vad."),
                "spec key `{key}` must live under the arch's own namespace"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 2 — A GGUF shaped exactly like today's converter output binds,
    //          with `config() == None`.
    // -----------------------------------------------------------------------

    #[test]
    fn converter_shaped_gguf_binds_without_spec_group() {
        // This is the whole point of the module: before it existed, this
        // GGUF was unloadable. It must bind, and the absent
        // `vokra.firered_vad.*` group must NOT be a load failure.
        let file = converter_shaped_gguf();
        let m = FireredVad::from_gguf(&file).expect("converter-shaped GGUF must bind");
        assert!(
            m.config().is_none(),
            "today's converter stamps no `vokra.firered_vad.*` group, so config() must be None"
        );
        assert_eq!(m.tensor_count(), 1, "the fixture carries one tensor");
        assert_eq!(
            m.weight_license(),
            LicenseClass::Permissive,
            "the `apache-2.0` stamp must round-trip as Permissive"
        );
        assert!(
            !m.is_research_only(),
            "a Permissive stamp must not be flagged research-only"
        );

        // Metadata round-trip of the two converter-private keys this
        // module mirrors.
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF),
            "FireRedVAD ships on HF, so provenance rides `upstream_hf` not `upstream_url`"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3 — The optional hyper-parameter group round-trips field-for-field.
    // -----------------------------------------------------------------------

    #[test]
    fn spec_group_round_trips() {
        let file = spec_stamped_gguf();
        let m = FireredVad::from_gguf(&file).expect("spec-stamped GGUF must bind");
        let cfg = m
            .config()
            .copied()
            .expect("stamped group must read back as Some");

        assert_eq!(cfg.sample_rate, 16_000);
        assert_eq!(cfg.n_mels, 80);
        assert_eq!(cfg.window_length, 400);
        assert_eq!(cfg.hop_length, 160);
        assert_eq!(cfg.n_layers, 4);
        assert_eq!(cfg.d_model, 256);
        assert_eq!(cfg.n_heads, 4);
        assert_eq!(cfg.ffn_dim, 1024);
        assert_eq!(cfg.n_class, 2);

        // Derived quantities.
        assert_eq!(cfg.head_dim(), 64, "d_model / n_heads");
        // 16000 / 160 = 100 frames per second, exactly representable.
        assert!(
            (cfg.frame_rate_hz() - 100.0).abs() < 1e-6,
            "frame_rate_hz = sample_rate / hop_length, got {}",
            cfg.frame_rate_hz()
        );
        cfg.validate().expect("the fixture group must validate");
    }

    // -----------------------------------------------------------------------
    // Test 4 — Missing arch tag is loud.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "some-other-name");
        b.add_tensor("some.tensor", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = finish(&b);

        let Err(err) = FireredVad::from_gguf(&file) else {
            panic!("expected ModelLoad when `vokra.model.arch` is absent");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("not a Vokra-native firered_vad GGUF"),
                    "message must name the missing-arch surface, got `{m}`"
                );
                assert!(
                    m.contains("`vokra.model.arch`"),
                    "message must name the missing key, got `{m}`"
                );
                assert!(
                    m.contains(CONVERTER_PATH),
                    "message must point at the converter that stamps it, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 5 — A sibling VAD GGUF is refused, with the whole fleet named.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_wrong_arch() {
        // An `fsmn-vad` GGUF handed here by mistake. Both models answer
        // "is this frame speech?" and both stamp `category = "vad"`, so
        // the category tag alone cannot disambiguate them — only the arch
        // tag can, and mis-binding would walk a completely different
        // tensor namespace (FR-EX-08).
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "fsmn-vad");
        b.add_string(chunks::KEY_MODEL_NAME, "fsmn-vad-zh-cn-16k-common");
        b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
        b.add_tensor(
            "encoder.in_linear.weight",
            GgmlType::F32,
            vec![4, 4],
            vec![0u8; 64],
        )
        .expect("add_tensor");
        let file = finish(&b);

        let Err(err) = FireredVad::from_gguf(&file) else {
            panic!("expected ModelLoad on a foreign arch tag");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`fsmn-vad`") && m.contains("`firered_vad`"),
                    "message must name both the got and the expected arch, got `{m}`"
                );
                for sibling in [
                    "silero-vad",
                    "fsmn-vad",
                    "pyannote-segmentation",
                    "smart_turn",
                    "ten_vad",
                ] {
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
    // Test 6 — Empty tensor manifest is loud (never an all-zero forward).
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_manifest() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, "permissive");
        // NO tensors added.
        let file = finish(&b);

        let Err(err) = FireredVad::from_gguf(&file) else {
            panic!("expected ModelLoad on an empty tensor manifest");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("zero tensors"),
                    "message must name the empty-manifest gap, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{m}`"
                );
                assert!(
                    m.contains("vokra-cli convert --model firered-vad"),
                    "message must include the repro command, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 7 — A partially stamped hyper-parameter group is loud, naming
    //          the missing key (all-or-nothing).
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_partially_stamped_spec_group() {
        let mut b = base_builder(Some(LicenseClass::Permissive));
        for (key, value) in FIXTURE_SPEC {
            if key == KEY_N_HEADS {
                continue; // the one key that is invisible in weight shapes
            }
            b.add_u32(key, value);
        }
        let file = finish(&b);

        let Err(err) = FireredVad::from_gguf(&file) else {
            panic!("expected ModelLoad on a partially stamped group");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(KEY_N_HEADS),
                    "message must name the missing key `{KEY_N_HEADS}`, got `{m}`"
                );
                assert!(
                    m.contains("all-or-nothing"),
                    "message must state the all-or-nothing contract, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 8 — A `0` sentinel anywhere in the group is loud, naming the key.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_zero_sentinel_in_spec_group() {
        let mut b = base_builder(Some(LicenseClass::Permissive));
        for (key, value) in FIXTURE_SPEC {
            let v = if key == KEY_HOP_LENGTH { 0 } else { value };
            b.add_u32(key, v);
        }
        let file = finish(&b);

        let Err(err) = FireredVad::from_gguf(&file) else {
            panic!("expected ModelLoad on a zero-sentinel hyper-parameter");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(KEY_HOP_LENGTH),
                    "message must name the offending key `{KEY_HOP_LENGTH}`, got `{m}`"
                );
                assert!(
                    m.contains("must be positive"),
                    "message must state the positivity requirement, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 9 — `d_model` indivisible by `n_heads` is loud.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_indivisible_d_model() {
        let mut b = base_builder(Some(LicenseClass::Permissive));
        for (key, value) in FIXTURE_SPEC {
            // 250 % 4 != 0 — multi-head attention cannot split it evenly.
            let v = if key == KEY_D_MODEL { 250 } else { value };
            b.add_u32(key, v);
        }
        let file = finish(&b);

        let Err(err) = FireredVad::from_gguf(&file) else {
            panic!("expected ModelLoad when d_model % n_heads != 0");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(KEY_D_MODEL) && m.contains(KEY_N_HEADS),
                    "message must name both keys, got `{m}`"
                );
                assert!(
                    m.contains("250"),
                    "message must echo the offending width, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 10 — A declared-but-absent required tensor is loud AT LOAD TIME,
    //           naming the tensor.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_declared_but_absent_tensor() {
        const MISSING: &str = "encoder.layer3.ffn.linear2.weight";

        let mut b = base_builder(Some(LicenseClass::Permissive));
        b.add_metadata(
            KEY_REQUIRED_TENSORS,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::String,
                values: vec![
                    // present in the fixture manifest
                    GgufMetadataValue::String(FIXTURE_TENSOR.to_owned()),
                    // NOT present — the truncation this gate exists to catch
                    GgufMetadataValue::String(MISSING.to_owned()),
                ],
            }),
        );
        let file = finish(&b);

        let Err(err) = FireredVad::from_gguf(&file) else {
            panic!("expected ModelLoad when a declared tensor is absent");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(MISSING),
                    "message must name the absent tensor `{MISSING}`, got `{m}`"
                );
                assert!(
                    m.contains(KEY_REQUIRED_TENSORS),
                    "message must name the declaration key, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 11 — An empty required-tensor declaration is itself loud, and a
    //           fully satisfied one binds.
    // -----------------------------------------------------------------------

    #[test]
    fn required_tensor_declaration_edge_cases() {
        // (a) Satisfied declaration → binds.
        let mut ok = base_builder(Some(LicenseClass::Permissive));
        ok.add_metadata(
            KEY_REQUIRED_TENSORS,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::String,
                values: vec![GgufMetadataValue::String(FIXTURE_TENSOR.to_owned())],
            }),
        );
        FireredVad::from_gguf(&finish(&ok)).expect("a satisfied declaration must bind");

        // (b) Empty declaration asserts nothing → always a producer bug.
        let mut empty = base_builder(Some(LicenseClass::Permissive));
        empty.add_metadata(
            KEY_REQUIRED_TENSORS,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::String,
                values: Vec::new(),
            }),
        );
        let Err(err) = FireredVad::from_gguf(&finish(&empty)) else {
            panic!("expected ModelLoad on an empty required-tensor declaration");
        };
        match err {
            VokraError::ModelLoad(m) => assert!(
                m.contains("empty list"),
                "message must name the empty-declaration bug, got `{m}`"
            ),
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 12 — The by-name tensor lookup names an absent tensor rather
    //           than returning `None`.
    // -----------------------------------------------------------------------

    #[test]
    fn tensor_lookup_names_the_missing_tensor() {
        let file = converter_shaped_gguf();
        let m = FireredVad::from_gguf(&file).expect("bind");

        // Present → real dims, straight from the GGUF header.
        assert_eq!(
            m.weights().dims(FIXTURE_TENSOR).expect("present tensor"),
            &[2usize, 3usize],
            "dims must come from the GGUF header, not a guess"
        );
        assert!(m.weights().has(FIXTURE_TENSOR));

        // Absent → loud, naming it.
        const ABSENT: &str = "encoder.layer9.attn.out.weight";
        assert!(!m.weights().has(ABSENT));
        let Err(err) = m.weights().dims(ABSENT) else {
            panic!("expected ModelLoad on an absent tensor lookup");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains(ABSENT),
                    "message must name the absent tensor, got `{msg}`"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite the no-silent-fallback clause, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 13 — Sample-rate mismatch is a loud `InvalidArgument`, and a
    //           matching rate falls through to the loud-partial instead.
    // -----------------------------------------------------------------------

    #[test]
    fn sample_rate_mismatch_is_loud_invalid_argument() {
        let file = spec_stamped_gguf();
        let m = FireredVad::from_gguf(&file).expect("bind");
        let pcm = vec![0.0_f32; 1_600];

        // 8 kHz against a 16 kHz checkpoint — never resample silently.
        let Err(err) = m.speech_probabilities(&pcm, 8_000) else {
            panic!("expected InvalidArgument on a sample-rate mismatch");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("8000") && msg.contains("16000"),
                    "message must echo both the pushed and the expected rate, got `{msg}`"
                );
                assert!(
                    msg.contains(KEY_SAMPLE_RATE),
                    "message must name the declaring key, got `{msg}`"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite the no-implicit-resample clause, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }

        // The matching rate passes the guard and reaches the loud-partial
        // — proving the guard is a separate, earlier gate.
        let Err(err) = m.speech_probabilities(&pcm, 16_000) else {
            panic!("expected the loud-partial once the rate matches");
        };
        assert!(
            matches!(err, VokraError::UnsupportedOp(_)),
            "a matching rate must reach the loud-partial, got {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 14 — Empty PCM is a loud `InvalidArgument` (an empty
    //           probability vector reads as "no speech").
    // -----------------------------------------------------------------------

    #[test]
    fn empty_pcm_is_loud_invalid_argument() {
        let file = spec_stamped_gguf();
        let m = FireredVad::from_gguf(&file).expect("bind");
        let Err(err) = m.speech_probabilities(&[], 16_000) else {
            panic!("expected InvalidArgument on empty PCM");
        };
        match err {
            VokraError::InvalidArgument(msg) => assert!(
                msg.contains("empty PCM"),
                "message must name the empty-input surface, got `{msg}`"
            ),
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 15 — The loud-partial forward names all three blockers, both
    //           primary sources and the FR-EX-08 rationale.
    // -----------------------------------------------------------------------

    #[test]
    fn speech_probabilities_is_loud_partial() {
        // No licence stamp: the binder must still bind (arch + manifest
        // are the load gates; licence is a compliance surface) and must
        // fail-close the class to Unknown.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
        b.add_tensor(FIXTURE_TENSOR, GgmlType::F32, vec![2, 3], vec![0u8; 24])
            .expect("add_tensor");
        let file = finish(&b);

        let m = FireredVad::from_gguf(&file).expect("valid arch must bind");
        assert_eq!(
            m.weight_license(),
            LicenseClass::Unknown,
            "a missing licence stamp must fail-close to Unknown"
        );
        assert!(
            m.is_research_only(),
            "Unknown must be treated as research-only (fail-closed)"
        );

        // 1 s of silence at 16 kHz. The group is unstamped here, so the
        // rate guard cannot fire and we land on the loud-partial.
        let pcm = vec![0.0_f32; 16_000];
        let Err(err) = m.speech_probabilities(&pcm, 16_000) else {
            panic!("speech_probabilities must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains("firered-vad speech_probabilities"),
                    "the surface must be named: {msg}"
                );
                assert!(msg.contains("loud-partial"), "posture label: {msg}");

                // All three blockers, by their headline labels.
                for blocker in [
                    "MISSING TOPOLOGY TRANSCRIPTION",
                    "MISSING METADATA",
                    "MISSING SIDECAR",
                ] {
                    assert!(msg.contains(blocker), "expected blocker `{blocker}`: {msg}");
                }

                // The specific anchors a follow-up wave needs.
                assert!(
                    msg.contains(SIDECAR_PATH),
                    "the sidecar path must be named: {msg}"
                );
                assert!(
                    msg.contains(CONVERTER_PATH),
                    "the converter contract must be named: {msg}"
                );
                assert!(
                    msg.contains(KEY_N_HEADS),
                    "the head-count key (invisible in packed-QKV shapes) must be named: {msg}"
                );
                assert!(
                    msg.contains("is NOT stamped"),
                    "the unstamped-group state must be reported: {msg}"
                );

                // Both primary sources.
                for url in [PRIMARY_SOURCE_HF, PRIMARY_SOURCE_FAMILY_CODE] {
                    assert!(msg.contains(url), "expected primary source `{url}`: {msg}");
                }

                assert!(
                    msg.contains("FR-EX-08"),
                    "expected the FR-EX-08 no-fabrication rationale: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 16 — With the group stamped, the loud-partial reports blocker (2)
    //           as cleared and echoes the real geometry.
    // -----------------------------------------------------------------------

    #[test]
    fn loud_partial_reports_a_stamped_group_as_cleared() {
        let file = spec_stamped_gguf();
        let m = FireredVad::from_gguf(&file).expect("bind");
        let Err(VokraError::UnsupportedOp(msg)) = m.speech_probabilities(&[0.0; 320], 16_000)
        else {
            panic!("expected the loud-partial");
        };
        assert!(
            msg.contains("IS stamped"),
            "a stamped group must be reported as such: {msg}"
        );
        assert!(
            msg.contains("blocker (2) is already cleared"),
            "the message must narrow the remaining work: {msg}"
        );
        assert!(
            msg.contains("head_dim=64"),
            "the echoed geometry must be derived, not guessed: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 17 — The `VadEngine` streaming path mirrors the one-shot guards.
    // -----------------------------------------------------------------------

    #[test]
    fn vad_engine_stream_mirrors_the_one_shot_guards() {
        let file = spec_stamped_gguf();
        let m = FireredVad::from_gguf(&file).expect("bind");
        let mut stream = m.open_stream();
        let pcm = vec![0.0_f32; 1_600];

        // Wrong rate → the same loud InvalidArgument as the one-shot path.
        let Err(err) = stream.push_pcm(&pcm, 44_100) else {
            panic!("expected InvalidArgument on a sample-rate mismatch");
        };
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "stream rate guard must be InvalidArgument, got {err:?}"
        );

        // Right rate → the loud-partial. Critically NOT `Ok(vec![])`:
        // an empty return is indistinguishable from "no frame completed"
        // and would let a caller loop forever believing the VAD runs.
        let Err(err) = stream.push_pcm(&pcm, 16_000) else {
            panic!("expected the loud-partial, never an empty Ok");
        };
        assert!(
            matches!(err, VokraError::UnsupportedOp(_)),
            "stream forward must be the loud-partial, got {err:?}"
        );

        // `reset` is infallible in the trait and a documented no-op here
        // (no recurrent state exists yet); it must not change the posture.
        stream.reset();
        let Err(err) = stream.push_pcm(&pcm, 16_000) else {
            panic!("expected the loud-partial after reset");
        };
        assert!(
            matches!(err, VokraError::UnsupportedOp(_)),
            "reset must not turn the loud-partial into a silent success, got {err:?}"
        );
    }
}
