//! **UTMOSv2** (`sarulab-speech/UTMOSv2`, MIT) — runtime binder for the
//! `utmosv2` converter arch (Wave A 2026-08-15, loud-partial per the
//! `dnsmos_p808_p835` / `emotion2vec` / RMVPE precedent — CLAUDE.md 教訓 (a):
//! 「loud-partial は fake-complete より honest」).
//!
//! # Why this module exists
//!
//! `crates/vokra-convert/src/models/utmosv2.rs` has been able to *produce* a
//! `utmosv2` GGUF since 2026-08-04, but nothing in the workspace could
//! *read* one back: no code anywhere bound the `utmosv2` arch string, so a
//! converted checkpoint was a write-only artifact. That gap blocks the
//! NFR-QL-02 5 % quality gate (an M5 / v1.0 GA DoD item), because the gate
//! needs a reference-free MOS instrument it can actually load. This module
//! closes the *load* half of that gap; the *score* half is an explicit
//! loud-partial (see below) rather than a fabricated number.
//!
//! # Primary sources
//!
//! - In-repo conversion contract (the authority for every tensor name and
//!   every `vokra.*` key this binder reads):
//!   `crates/vokra-convert/src/models/utmosv2.rs`.
//! - Reference code + licence:
//!   <https://github.com/sarulab-speech/UTMOSv2> — the converter docstring
//!   records the licence as **standard MIT**, verified against
//!   `github.com/sarulab-speech/UTMOSv2/blob/main/LICENSE`
//!   ([`LicenseClass::Permissive`]). This binder does not re-derive that
//!   claim; it mirrors the converter's own recorded verification.
//! - Paper: Baba et al., *"UTMOSv2: UTokyo-SaruLab MOS Prediction System for
//!   VoiceMOS Challenge 2024"*, <https://arxiv.org/abs/2409.09305>
//!   (cited by the converter docstring).
//!
//! # Runtime layout
//!
//! ```text
//! PCM (mono f32)
//!   -> spectrogram-domain branch                       <- **loud-partial**
//!        (UTMOSv2 is described upstream as a multi-modal system that fuses
//!         a spectrogram branch with an SSL branch; the in-repo conversion
//!         contract is a verbatim float pass-through that stamps NO
//!         `vokra.utmosv2.*` topology axes, so this branch's backbone is not
//!         primary-source-transcribable from the GGUF.)
//!   -> wav2vec2-large SSL encoder                      <- **loud-partial**
//!        (converter docstring characterisation; layer count / hidden width
//!         are unstamped.)
//!   -> listener / domain conditioning                  <- **loud-partial**
//!        (converter docstring characterisation; the conditioning fan-in is
//!         unstamped.)
//!   -> Regressor head fusion -> 1 MOS scalar           <- **loud-partial**
//!   -> clamp to the ACR scale [1, 5]                   <- REAL
//!        ([`clamp_to_mos_range`] — model-independent, the same [1, 5] ACR
//!         range the sibling `dnsmos_p808_p835` binder documents.)
//! ```
//!
//! # Loud-partial classification (CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**:
//!   - [`Utmosv2::from_gguf`] with *strict* `vokra.model.arch == "utmosv2"`
//!     verification that enumerates every sibling eval-family arch tag
//!     ([`SIBLING_EVAL_ARCH_TAGS`]) so a mis-routed GGUF fails with a
//!     specific message instead of a downstream missing-tensor error
//!     (FR-EX-08);
//!   - [`Utmosv2Config::from_gguf`] parsing **exactly** the metadata the
//!     converter writes — `vokra.model.name`, `vokra.model.category`,
//!     `vokra.provenance.upstream_hf`, plus the `stamp_provenance` group
//!     (`weight_license` / `license` / `model_id` / `source`) — with a
//!     fail-closed [`LicenseClass::Unknown`] when the class stamp is absent;
//!   - [`Utmosv2Weights::from_gguf`] binding **every** tensor the converter
//!     emits with real per-tensor shape checks: dtype ∈ [`ACCEPTED_DTYPES`]
//!     (the converter's pass-through arm emits F32 / F16 / BF16 and nothing
//!     else), rank ≥ 1, no zero-extent dimension, on-disk payload length
//!     equal to the dtype-derived length, no duplicate names, and at least
//!     one rank ≥ 2 tensor (a Regressor head cannot exist without a single
//!     Linear weight matrix);
//!   - [`Utmosv2Weights::require`] / [`Utmosv2Weights::require_shape`] /
//!     [`Utmosv2Weights::load_f32`] — the named-tensor accessors the
//!     follow-up forward wave binds against, each failing loud and naming
//!     the tensor;
//!   - [`clamp_to_mos_range`] — the terminal ACR clamp, real today.
//!
//! - **Loud-partial (this WP)**: [`Utmosv2::predict_mos`] returns
//!   [`VokraError::UnsupportedOp`] naming (a) all three deferred stages,
//!   (b) the concrete `vokra.utmosv2.*` metadata keys the converter must
//!   start stamping, (c) the sidecar that does not exist in-tree yet
//!   ([`SIDECAR_PATH`]), (d) the re-conversion command
//!   ([`CONVERT_COMMAND`]), and (e) both primary-source URLs. **No
//!   fabricated MOS scalar is ever returned** — a silently invented `0.0`
//!   or `3.0` would corrupt the NFR-QL-02 5 % gate it feeds (FR-EX-08).
//!
//! # `vokra.*` chunk group read here
//!
//! Written by `vokra-convert::models::utmosv2::convert_utmosv2_file`:
//!
//! - `vokra.model.arch` (String) — `"utmosv2"`, strictly verified;
//! - `vokra.model.name` (String) — `"utmosv2"`;
//! - `vokra.model.category` (String) — `"eval"`, strictly verified;
//! - `vokra.provenance.upstream_hf` (String) — `"sarulab-speech/UTMOSv2"`;
//! - `vokra.provenance.weight_license` (String) — [`LicenseClass`] wire name;
//! - `vokra.provenance.license` (String) — raw SPDX (`"mit"` by default);
//! - `vokra.provenance.model_id` (String) / `vokra.provenance.source`
//!   (String) — free-text provenance.
//!
//! The converter writes **no** `vokra.utmosv2.*` group at all. Any key under
//! that prefix found in a GGUF is surfaced by
//! [`Utmosv2Config::topology_keys_present`] (and echoed in the loud-partial
//! message) so the follow-up wave can tell "metadata half landed, runtime
//! half did not" from "neither landed" without guessing.
//!
//! # Sibling eval-family distinctness
//!
//! [`ARCH`] = `"utmosv2"` is deliberately distinct from every sibling
//! eval-family arch tag. All five are reference-free quality instruments but
//! none share a topology or an output contract:
//!
//! - `utmos` — SaruLab **UTMOS22-strong** (wav2vec2-**base** SSL + regression
//!   head). The direct predecessor; the runtime skeleton for it lives in
//!   `vokra-eval::metrics::utmos`, not here. UTMOSv2's SSL axis is
//!   wav2vec2-**large** and its head layout differs, so sharing the arch tag
//!   would silently mis-route the loader;
//! - `dnsmos` — Microsoft DNSMOS P.808 / P.835 CNN predictors (1 or 3 scalar
//!   outputs on a 9.01 s window, not a single utterance MOS);
//! - `nisqa_v2_weight` — NISQA v2 multi-dimensional speech-quality model;
//! - `torchaudio_squim` — torchaudio SQUIM objective / subjective metrics.
//!
//! Silently aliasing any of these would mis-route runtime dispatch to a
//! loader with a different tensor walk and a different output width — the
//! failure would surface as a confusing missing-tensor error rather than a
//! specific arch mismatch (FR-EX-08).
//!
//! # Why [`vokra_core::engines::MosScorerEngine`] is *not* implemented here
//!
//! That trait's [`vokra_core::engines::MosScore`] payload is DNSMOS-shaped —
//! `p808` / `sig` / `bak` / `ovrl`, all ITU-T P.808 / P.835 terms. UTMOSv2
//! predicts a single utterance-level MOS with no P.808 / P.835 semantics, so
//! folding it into `p808` would attach an ITU-T claim the model does not
//! make. The binder therefore exposes a plain [`Utmosv2::predict_mos`]
//! instead. Widening `MosScore` with a generic scalar field is a
//! deliberately separate decision (it is a `vokra-core` engine-surface
//! change) and is not made here.
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_HF`] /
//! [`DEFAULT_LICENSE_SPDX`] and the two raw metadata keys are **mirrors** of
//! the converter's `pub const` surface — the same rule every sibling binder
//! (`dnsmos_p808_p835` / `emotion2vec` / `sensevoicesmall_runtime` / `panns`
//! / `redimnet` / …) follows so `vokra-models` never gains a dependency edge
//! onto `vokra-convert`. The layering stays
//! `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`,
//! `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! The upstream UTMOSv2 release ships PyTorch pickle files. The pickle is
//! flattened to safetensors **offline** by a sidecar before the converter
//! runs; neither a pickle nor an ONNX graph ever enters the runtime
//! (FR-LD-05 / NFR-DS-02).

use vokra_core::gguf::{GgmlType, GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Contract constants — mirrors of
// `crates/vokra-convert/src/models/utmosv2.rs`. See the module docstring's
// "Cross-crate constant duplication" section for the rationale (vokra-models
// must NOT depend on vokra-convert).
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model utmosv2`.
///
/// **Mirror** of the converter's `ARCH`. Deliberately distinct from every
/// sibling eval-family arch tag ([`SIBLING_EVAL_ARCH_TAGS`]) — see the module
/// docstring's "Sibling eval-family distinctness" section.
pub const ARCH: &str = "utmosv2";

/// Expected `vokra.model.name` value written by the converter.
///
/// **Mirror** of the converter's `NAME`.
pub const NAME: &str = "utmosv2";

/// Expected `vokra.model.category` value — `"eval"` (the reference-free
/// quality-instrument tier shared with `utmos` / `dnsmos` /
/// `nisqa_v2_weight` / `torchaudio_squim`).
///
/// **Mirror** of the converter's `CATEGORY`.
pub const CATEGORY: &str = "eval";

/// Canonical upstream Hugging Face / GitHub slug recorded under
/// `vokra.provenance.upstream_hf`.
///
/// **Mirror** of the converter's `UPSTREAM_HF`. A GGUF that carries a
/// *different* slug still loads (a legitimate mirror or fork is not an
/// error); [`Utmosv2Config::is_canonical_upstream`] reports the difference.
pub const UPSTREAM_HF: &str = "sarulab-speech/UTMOSv2";

/// Default upstream weight licence (raw SPDX) the converter stamps.
///
/// **Mirror** of the converter's `DEFAULT_LICENSE_SPDX`. The converter
/// docstring records this as verified against
/// `github.com/sarulab-speech/UTMOSv2/blob/main/LICENSE` (standard MIT →
/// [`LicenseClass::Permissive`]). This binder mirrors that recorded
/// verification rather than re-deriving it.
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

/// `vokra.model.category` metadata key.
///
/// **Mirror** of the converter's `KEY_MODEL_CATEGORY` — not (yet)
/// centralised in `vokra_core::gguf::chunks`, per the established
/// `sensevoicesmall` / `nkf_aec` / `funcodec` convention.
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_hf` metadata key.
///
/// **Mirror** of the converter's `KEY_PROVENANCE_UPSTREAM_HF`.
pub const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

// ---------------------------------------------------------------------------
// Future `vokra.utmosv2.*` topology axes.
//
// The converter writes NONE of these today. They are declared here so the
// loud-partial diagnostic can name them verbatim (a rename must therefore
// land in the same commit as the message, and the test suite pins that).
// ---------------------------------------------------------------------------

/// Prefix of the (currently unwritten) UTMOSv2 topology-axis metadata group.
///
/// [`Utmosv2Config::from_gguf`] collects every metadata key under this prefix
/// into [`Utmosv2Config::topology_keys_present`] so the follow-up wave can
/// distinguish "the converter half of the flip has landed" from "neither half
/// has landed" without guessing.
pub const KEY_UTMOSV2_PREFIX: &str = "vokra.utmosv2.";

/// Future metadata key: PCM sample rate the checkpoint expects (u32 Hz).
///
/// Not written by the converter today — named verbatim in the loud-partial
/// recipe.
pub const KEY_UTMOSV2_SAMPLE_RATE: &str = "vokra.utmosv2.sample_rate";

/// Future metadata key: SSL encoder block count (u32).
///
/// Not written by the converter today — named verbatim in the loud-partial
/// recipe.
pub const KEY_UTMOSV2_SSL_N_LAYER: &str = "vokra.utmosv2.ssl.n_layer";

/// Future metadata key: SSL encoder hidden width (u32).
///
/// Not written by the converter today — named verbatim in the loud-partial
/// recipe.
pub const KEY_UTMOSV2_SSL_HIDDEN_DIM: &str = "vokra.utmosv2.ssl.hidden_dim";

/// Future metadata key: mel-bin count of the spectrogram-domain branch (u32).
///
/// Not written by the converter today — named verbatim in the loud-partial
/// recipe.
pub const KEY_UTMOSV2_SPEC_N_MELS: &str = "vokra.utmosv2.spec.n_mels";

/// Future metadata key: Regressor head linear output dims (`Array<u32>`, last
/// element must be 1).
///
/// Not written by the converter today — named verbatim in the loud-partial
/// recipe.
pub const KEY_UTMOSV2_HEAD_DIMS: &str = "vokra.utmosv2.head.dims";

// ---------------------------------------------------------------------------
// Primary-source + recipe anchors (cited verbatim in the loud-partial error).
// ---------------------------------------------------------------------------

/// Primary-source anchor: the UTMOSv2 reference implementation.
pub const PRIMARY_SOURCE_CODE: &str = "github.com/sarulab-speech/UTMOSv2";

/// Primary-source anchor: the upstream `LICENSE` file the converter docstring
/// records as verified (standard MIT).
pub const PRIMARY_SOURCE_LICENSE: &str = "github.com/sarulab-speech/UTMOSv2/blob/main/LICENSE";

/// Primary-source anchor: Baba et al., VoiceMOS Challenge 2024 system paper.
pub const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2409.09305";

/// The offline pickle → safetensors sidecar the converter docstring names.
///
/// **It does not exist in the tree today** — that absence is one of the
/// reasons the forward is a loud-partial, and the diagnostic says so.
pub const SIDECAR_PATH: &str = "tools/parity/utmosv2_prepare_checkpoint.py";

/// The re-conversion command a reader must run after the converter starts
/// stamping the `vokra.utmosv2.*` axes.
pub const CONVERT_COMMAND: &str = "vokra-cli convert --model utmosv2";

/// Every sibling eval-family arch tag, enumerated in the wrong-arch
/// diagnostic so a mis-routed GGUF names its own family.
///
/// Order is stable for diagnostics only (it carries no dispatch meaning).
pub const SIBLING_EVAL_ARCH_TAGS: [&str; 4] =
    ["utmos", "dnsmos", "nisqa_v2_weight", "torchaudio_squim"];

/// Top-level `state_dict` module prefixes seen in the converter's own test
/// fixtures (`ssl_encoder.` / `listener_head.` / `mos_head.`).
///
/// **Diagnostic hints only — never a load requirement.** The converter is a
/// verbatim float pass-through: it neither renames nor validates upstream
/// keys, so a real checkpoint (or a fork) may legitimately use different
/// top-level module names. [`Utmosv2Weights::module_prefix_inventory`]
/// reports the per-prefix counts for triage; nothing rejects a GGUF for
/// carrying zero tensors under any of them (fabricating a required manifest
/// from test fixtures would be exactly the silent-wrong failure CLAUDE.md
/// 教訓 (a) warns about).
pub const KNOWN_MODULE_PREFIXES: [&str; 3] = ["ssl_encoder.", "listener_head.", "mos_head."];

/// Lower bound of the ACR mean-opinion-score scale.
///
/// The same `[1, 5]` range the sibling `dnsmos_p808_p835` binder documents
/// for its per-chunk MOS.
pub const MOS_MIN: f32 = 1.0;

/// Upper bound of the ACR mean-opinion-score scale (see [`MOS_MIN`]).
pub const MOS_MAX: f32 = 5.0;

/// The only tensor dtypes a `utmosv2` GGUF may carry.
///
/// Transcribed from the converter's pass-through match arm
/// (`GgmlType::F32 | GgmlType::F16 | GgmlType::BF16`): every other dtype is
/// counted as `skipped_non_float` and never written. A K-quant tensor in a
/// `utmosv2` GGUF therefore means the file was re-quantised *after*
/// conversion — which would silently shift the calibration of an instrument
/// whose whole job is to measure quality, so [`Utmosv2Weights::from_gguf`]
/// refuses it loudly.
pub const ACCEPTED_DTYPES: [GgmlType; 3] = [GgmlType::F32, GgmlType::F16, GgmlType::BF16];

// ---------------------------------------------------------------------------
// Real helper: terminal ACR clamp.
// ---------------------------------------------------------------------------

/// Clamps a raw regressor output onto the ACR scale bounded by [`MOS_MIN`]
/// and [`MOS_MAX`].
///
/// This is the one genuinely model-independent stage of the UTMOSv2
/// pipeline, so it lands real today and the follow-up forward wave calls it
/// unchanged.
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] when `raw` is NaN or infinite. A
///   non-finite regressor output must never be silently clamped to a
///   plausible-looking `1.0` / `5.0` — that would hide a poisoned forward
///   behind a valid-looking MOS (FR-EX-08).
pub fn clamp_to_mos_range(raw: f32) -> Result<f32> {
    if !raw.is_finite() {
        return Err(VokraError::InvalidArgument(format!(
            "utmosv2: regressor emitted a non-finite value ({raw}) — refusing to clamp it \
             onto the ACR scale [{MOS_MIN}, {MOS_MAX}]. Silently clamping NaN / inf would \
             hide a poisoned forward behind a plausible-looking MOS (FR-EX-08)."
        )));
    }
    Ok(raw.clamp(MOS_MIN, MOS_MAX))
}

// ---------------------------------------------------------------------------
// Utmosv2Config — parsed from exactly the metadata the converter writes.
// ---------------------------------------------------------------------------

/// UTMOSv2 runtime config, transcribed from the `vokra.model.*` /
/// `vokra.provenance.*` chunks the converter actually writes.
///
/// There is deliberately no topology field here: the converter stamps no
/// `vokra.utmosv2.*` axes, and inventing defaults for layer counts / widths
/// would be exactly the fabrication the loud-partial exists to avoid. What a
/// GGUF *does* advertise under that prefix is preserved verbatim in
/// [`Self::topology_keys_present`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Utmosv2Config {
    /// `vokra.model.name` (required). The canonical converter value is
    /// [`NAME`]; a fork may legitimately stamp its own name, so only
    /// non-emptiness is enforced.
    pub name: String,
    /// `vokra.model.category` (required). Must equal [`CATEGORY`].
    pub category: String,
    /// `vokra.provenance.upstream_hf` (required). Compared against
    /// [`UPSTREAM_HF`] by [`Self::is_canonical_upstream`], never enforced.
    pub upstream_hf: String,
    /// `vokra.provenance.license` — the raw SPDX string
    /// ([`DEFAULT_LICENSE_SPDX`] for a stock conversion). `None` when the
    /// converter was handed an empty licence override.
    pub license_spdx: Option<String>,
    /// `vokra.provenance.weight_license` resolved through
    /// [`LicenseClass::from_class_str`]. **Fail-closed**: an absent or
    /// unrecognised stamp yields [`LicenseClass::Unknown`], never a
    /// permissive default.
    pub weight_license: LicenseClass,
    /// `vokra.provenance.model_id` (free text, optional).
    pub model_id: Option<String>,
    /// `vokra.provenance.source` (free text, optional).
    pub source: Option<String>,
    /// Every metadata key found under [`KEY_UTMOSV2_PREFIX`], sorted.
    ///
    /// Empty for every GGUF the current converter produces. A non-empty
    /// vector means someone has already extended the converter with topology
    /// axes — the loud-partial message calls that out so the reader knows
    /// only the runtime half of the flip is outstanding.
    pub topology_keys_present: Vec<String>,
}

impl Utmosv2Config {
    /// Validates the parsed config loudly (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.name` is empty.
    /// - [`VokraError::ModelLoad`] when `vokra.model.category` is not
    ///   [`CATEGORY`].
    /// - [`VokraError::ModelLoad`] when `vokra.provenance.upstream_hf` is
    ///   empty.
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "utmosv2: `{key}` is empty — a Vokra-native UTMOSv2 GGUF always carries a \
                 model name (the stock converter stamps `{NAME}`). Re-run \
                 `{CONVERT_COMMAND}`.",
                key = chunks::KEY_MODEL_NAME,
            )));
        }
        if self.category != CATEGORY {
            return Err(VokraError::ModelLoad(format!(
                "utmosv2: `{KEY_MODEL_CATEGORY}` is `{got}`, expected `{CATEGORY}` — the \
                 UTMOSv2 converter stamps the reference-free quality-instrument tier. A \
                 different category means this GGUF came from another converter arm and \
                 would be mis-advertised by the model-card generator and the zoo manifest \
                 tier gate (FR-EX-08).",
                got = self.category,
            )));
        }
        if self.upstream_hf.trim().is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "utmosv2: `{KEY_PROVENANCE_UPSTREAM_HF}` is empty — the redistribution \
                 source slug is required so a downstream can trace the artifact back to \
                 `{UPSTREAM_HF}` without parsing the free-text \
                 `vokra.provenance.source`. Re-run `{CONVERT_COMMAND}`."
            )));
        }
        Ok(())
    }

    /// Reads the config from a parsed GGUF.
    ///
    /// Only the keys the converter actually writes are consulted; nothing is
    /// defaulted except the two genuinely optional provenance fields and the
    /// fail-closed licence class.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.name`,
    ///   `vokra.model.category` or `vokra.provenance.upstream_hf` is missing
    ///   or is not a String.
    /// - Whatever [`Self::validate`] rejects.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let name = required_string(gguf, chunks::KEY_MODEL_NAME)?;
        let category = required_string(gguf, KEY_MODEL_CATEGORY)?;
        let upstream_hf = required_string(gguf, KEY_PROVENANCE_UPSTREAM_HF)?;

        let license_spdx = gguf
            .get(chunks::KEY_PROVENANCE_LICENSE)
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        // Fail-closed: an absent or unrecognised class stamp is `Unknown`,
        // never a permissive default (memory
        // `[[feedback-license-signoff-primary-source]]`).
        let weight_license = gguf
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);
        let model_id = gguf
            .get(chunks::KEY_PROVENANCE_MODEL_ID)
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        let source = gguf
            .get(chunks::KEY_PROVENANCE_SOURCE)
            .and_then(|v| v.as_str())
            .map(str::to_owned);

        let mut topology_keys_present: Vec<String> = gguf
            .metadata()
            .iter()
            .map(|(k, _)| k.as_str())
            .filter(|k| k.starts_with(KEY_UTMOSV2_PREFIX))
            .map(str::to_owned)
            .collect();
        topology_keys_present.sort();

        let cfg = Self {
            name,
            category,
            upstream_hf,
            license_spdx,
            weight_license,
            model_id,
            source,
            topology_keys_present,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Whether [`Self::upstream_hf`] is the canonical [`UPSTREAM_HF`] slug.
    ///
    /// A mirror or a fork is *not* rejected — this is a triage accessor, not
    /// a gate.
    #[inline]
    #[must_use]
    pub fn is_canonical_upstream(&self) -> bool {
        self.upstream_hf == UPSTREAM_HF
    }
}

/// Reads a required `String` metadata chunk, refusing to coerce a
/// wrong-typed value (FR-EX-08).
fn required_string(gguf: &GgufFile, key: &str) -> Result<String> {
    let value = gguf.get(key).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "utmosv2 GGUF is missing required String metadata `{key}` — was this file \
             produced by `{CONVERT_COMMAND}`?"
        ))
    })?;
    value.as_str().map(str::to_owned).ok_or_else(|| {
        let got = value.value_type();
        VokraError::ModelLoad(format!(
            "utmosv2 GGUF metadata `{key}` has type {got:?}, expected String — the runtime \
             will not silently coerce a mis-typed provenance chunk (FR-EX-08)."
        ))
    })
}

// ---------------------------------------------------------------------------
// Utmosv2Tensor / Utmosv2Weights — the real tensor manifest bind.
// ---------------------------------------------------------------------------

/// One tensor bound from a `utmosv2` GGUF.
///
/// The payload is *not* held: the converter emits the upstream state-dict
/// verbatim (a ~500 MB checkpoint), and the forward that would consume it is
/// a loud-partial, so eagerly dequantising every tensor into RAM would buy
/// nothing. The follow-up wave pulls payloads on demand through
/// [`Utmosv2Weights::load_f32`]. Same posture as the sibling
/// `dnsmos_p808_p835` binder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Utmosv2Tensor {
    /// Upstream `state_dict` key, verbatim (the converter never renames).
    pub name: String,
    /// Dimensions, innermost first, exactly as stored on disk.
    pub dims: Vec<usize>,
    /// On-disk element type — always one of [`ACCEPTED_DTYPES`].
    pub dtype: GgmlType,
}

impl Utmosv2Tensor {
    /// Number of dimensions (rank). Always ≥ 1 for a bound tensor.
    #[inline]
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Product of [`Self::dims`]. Never zero for a bound tensor (the
    /// zero-extent gate in [`Utmosv2Weights::from_gguf`] rejects those).
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.dims.iter().product()
    }
}

/// The full tensor manifest of a `utmosv2` GGUF.
///
/// **Contract**: [`Self::from_gguf`] is a *loud* verification step. Every
/// tensor the converter emitted is bound and shape-checked; the first
/// violation aborts the load with a message naming the offending tensor
/// (FR-EX-08 — never a silent partial bind).
#[derive(Debug, Clone)]
pub struct Utmosv2Weights {
    tensors: Vec<Utmosv2Tensor>,
}

impl Utmosv2Weights {
    /// Binds and shape-checks every tensor in `gguf`.
    ///
    /// Per-tensor gates, in order:
    ///
    /// 1. dtype ∈ [`ACCEPTED_DTYPES`] — the converter's pass-through arm
    ///    emits nothing else, so a K-quant here means post-conversion
    ///    re-quantisation of a measuring instrument;
    /// 2. rank ≥ 1 — a rank-0 scalar is never a UTMOSv2 weight;
    /// 3. no zero-extent dimension — a `[512, 0]` tensor would dequantise to
    ///    an empty buffer and read as "all weights are zero";
    /// 4. every dimension fits `usize` (32-bit targets);
    /// 5. the on-disk payload length equals the dtype-derived length. This
    ///    one is a **defensive** check that cannot fire against today's
    ///    `vokra-core` reader (`GgufFile::tensor_bytes` lends a slice sized
    ///    by `byte_len`, and bounds were validated at parse time) — it is
    ///    kept for the same reason the converter keeps its
    ///    `skipped_non_float` counter: a tensor reaching it would signal a
    ///    reader change upstream, and the assertion is free.
    ///
    /// Manifest-level gates:
    ///
    /// 6. at least one tensor (an empty GGUF is never a valid checkpoint);
    /// 7. no duplicate tensor names (a duplicate makes every
    ///    `tensor_info` lookup ambiguous — defence against a non-Vokra
    ///    writer, since `GgufBuilder` already refuses duplicates);
    /// 8. at least one rank ≥ 2 tensor — a Regressor head cannot exist
    ///    without a single Linear weight matrix, and a manifest of nothing
    ///    but 1-D biases / norms always means a truncated flatten.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] naming the offending tensor for gates
    ///   1-5 and 7, or naming the structural offence for gates 6 and 8.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let mut tensors: Vec<Utmosv2Tensor> = Vec::with_capacity(gguf.tensors().len());

        for info in gguf.tensors() {
            let name = info.name.as_str();
            let dtype = info.dtype;

            // (1) dtype gate.
            if !ACCEPTED_DTYPES.contains(&dtype) {
                return Err(VokraError::ModelLoad(format!(
                    "utmosv2: tensor `{name}` has dtype {dtype:?}, but a `{ARCH}` GGUF may \
                     only carry F32 / F16 / BF16 — the converter's pass-through arm writes \
                     nothing else and counts every other dtype as `skipped_non_float`. A \
                     quantised tensor here means the file was re-quantised AFTER \
                     conversion, which would silently shift the calibration of the very \
                     instrument the NFR-QL-02 5% quality gate trusts (FR-EX-08). Re-run \
                     `{CONVERT_COMMAND}` from the upstream checkpoint instead."
                )));
            }

            // (2) rank gate.
            if info.dimensions.is_empty() {
                return Err(VokraError::ModelLoad(format!(
                    "utmosv2: tensor `{name}` is rank-0 (no dimensions) — a UTMOSv2 \
                     state-dict entry is always at least a 1-D bias / norm vector. A \
                     rank-0 entry signals a mis-produced GGUF (FR-EX-08)."
                )));
            }

            // (3) + (4) per-dimension gates.
            let mut dims: Vec<usize> = Vec::with_capacity(info.dimensions.len());
            for (axis, &d) in info.dimensions.iter().enumerate() {
                if d == 0 {
                    return Err(VokraError::ModelLoad(format!(
                        "utmosv2: tensor `{name}` has a zero-extent dimension at axis \
                         {axis} (dims = {shape:?}) — it would dequantise to an empty \
                         buffer and read downstream as `all weights are zero`. Refusing \
                         the load (FR-EX-08).",
                        shape = info.dimensions,
                    )));
                }
                let d = usize::try_from(d).map_err(|_| {
                    VokraError::ModelLoad(format!(
                        "utmosv2: tensor `{name}` dimension {d} at axis {axis} does not fit \
                         in `usize` on this target — refusing the load rather than \
                         truncating a shape (FR-EX-08)."
                    ))
                })?;
                dims.push(d);
            }

            // (5) payload-length gate.
            let expected = info.byte_len()?;
            let actual = gguf.tensor_bytes(info).len() as u64;
            if expected != actual {
                return Err(VokraError::ModelLoad(format!(
                    "utmosv2: tensor `{name}` payload is {actual} bytes but its dtype \
                     {dtype:?} and shape {dims:?} imply {expected} bytes — the GGUF tensor \
                     table and its data region disagree. Refusing the load (FR-EX-08)."
                )));
            }

            tensors.push(Utmosv2Tensor {
                name: name.to_owned(),
                dims,
                dtype,
            });
        }

        // (6) non-empty manifest gate.
        if tensors.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "utmosv2: GGUF carries zero tensors — refusing to bind an all-zero forward \
                 (FR-EX-08). A legitimate `{UPSTREAM_HF}` checkpoint carries the whole \
                 wav2vec2-large SSL encoder plus the listener / domain conditioning and \
                 Regressor head parameters (arch={ARCH}, name={NAME}); zero tensors always \
                 signals a mis-produced GGUF. Re-run `{CONVERT_COMMAND}` against an \
                 upstream safetensors checkpoint (flattened offline by `{SIDECAR_PATH}`)."
            )));
        }

        // (7) duplicate-name gate.
        check_duplicate_names(&tensors)?;

        // (8) structural gate — at least one weight matrix.
        if !tensors.iter().any(|t| t.rank() >= 2) {
            return Err(VokraError::ModelLoad(format!(
                "utmosv2: GGUF carries {n} tensors but not one of rank >= 2 — a Regressor \
                 head cannot exist without a single Linear weight matrix, so a manifest of \
                 nothing but 1-D biases / norm vectors always means a truncated flatten. \
                 Re-run `{SIDECAR_PATH}` and then `{CONVERT_COMMAND}` (FR-EX-08).",
                n = tensors.len(),
            )));
        }

        Ok(Self { tensors })
    }

    /// The bound tensor manifest, in GGUF order.
    #[inline]
    #[must_use]
    pub fn tensors(&self) -> &[Utmosv2Tensor] {
        &self.tensors
    }

    /// Number of bound tensors. Always ≥ 1.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Looks a tensor up by its verbatim upstream `state_dict` key.
    ///
    /// This is the accessor the follow-up forward wave binds against: every
    /// miss is a loud [`VokraError::ModelLoad`] that names the tensor asked
    /// for and lists nearby names actually present, so a `state_dict` key
    /// drift is diagnosable in one step (FR-EX-08 — never a zero-fill).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when no tensor carries that name.
    pub fn require(&self, name: &str) -> Result<&Utmosv2Tensor> {
        self.tensors.iter().find(|t| t.name == name).ok_or_else(|| {
            let near = self.nearest_names(name);
            let hint = if near.is_empty() {
                String::from("(the manifest is empty)")
            } else {
                near.join(", ")
            };
            VokraError::ModelLoad(format!(
                "utmosv2: required tensor `{name}` is missing from the GGUF ({n} \
                     tensors bound). Nearby names present: {hint}. The converter emits \
                     upstream `state_dict` keys verbatim, so a miss means either the \
                     offline flatten (`{SIDECAR_PATH}`) dropped the entry or the upstream \
                     key was renamed. Refusing to zero-fill (FR-EX-08).",
                n = self.tensors.len(),
            ))
        })
    }

    /// [`Self::require`] plus an exact shape assertion.
    ///
    /// # Errors
    ///
    /// - Whatever [`Self::require`] rejects.
    /// - [`VokraError::ModelLoad`] naming the tensor, the expected shape and
    ///   the actual shape when they differ.
    pub fn require_shape(&self, name: &str, expected: &[usize]) -> Result<&Utmosv2Tensor> {
        let t = self.require(name)?;
        if t.dims != expected {
            return Err(VokraError::ModelLoad(format!(
                "utmosv2: tensor `{name}` has shape {actual:?}, expected {expected:?} — a \
                 shape mismatch means the checkpoint variant does not match the topology \
                 this forward was wired for. Refusing to reinterpret the buffer \
                 (FR-EX-08).",
                actual = t.dims,
            )));
        }
        Ok(t)
    }

    /// Dequantises one tensor's payload into owned `f32`.
    ///
    /// The name is validated through [`Self::require`] first so a miss
    /// produces the rich manifest-aware diagnostic rather than a bare
    /// `MissingTensor`.
    ///
    /// # Errors
    ///
    /// - Whatever [`Self::require`] rejects.
    /// - [`VokraError::ModelLoad`] when the GGUF payload fails to decode.
    pub fn load_f32(&self, gguf: &GgufFile, name: &str) -> Result<Vec<f32>> {
        let _ = self.require(name)?;
        gguf.tensor_f32(name).map_err(|e| {
            VokraError::ModelLoad(format!(
                "utmosv2: tensor `{name}` is present in the manifest but its payload failed \
                 to decode: {e}"
            ))
        })
    }

    /// Number of bound tensors whose name starts with `prefix`.
    #[must_use]
    pub fn count_with_prefix(&self, prefix: &str) -> usize {
        self.tensors
            .iter()
            .filter(|t| t.name.starts_with(prefix))
            .count()
    }

    /// Per-prefix counts over [`KNOWN_MODULE_PREFIXES`], for triage.
    ///
    /// A zero count is informational, **not** an error — see the
    /// [`KNOWN_MODULE_PREFIXES`] doc for why those prefixes are hints rather
    /// than a required manifest.
    #[must_use]
    pub fn module_prefix_inventory(&self) -> Vec<(&'static str, usize)> {
        KNOWN_MODULE_PREFIXES
            .iter()
            .map(|p| (*p, self.count_with_prefix(p)))
            .collect()
    }

    /// Up to five manifest names sharing the requested name's first
    /// dot-segment; falls back to the first five names when nothing matches.
    fn nearest_names(&self, wanted: &str) -> Vec<&str> {
        let head = wanted.split('.').next().unwrap_or(wanted);
        let hits: Vec<&str> = self
            .tensors
            .iter()
            .map(|t| t.name.as_str())
            .filter(|n| n.starts_with(head))
            .take(5)
            .collect();
        if hits.is_empty() {
            self.tensors
                .iter()
                .map(|t| t.name.as_str())
                .take(5)
                .collect()
        } else {
            hits
        }
    }
}

/// Rejects duplicate tensor names.
///
/// `GgufBuilder::add_tensor` already refuses duplicates, so this can only
/// fire for a GGUF written by a non-Vokra producer — it is defence in depth,
/// kept because a duplicate would make every `tensor_info` lookup silently
/// resolve to whichever copy the reader indexed first.
fn check_duplicate_names(tensors: &[Utmosv2Tensor]) -> Result<()> {
    let mut names: Vec<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
    names.sort_unstable();
    for pair in names.windows(2) {
        if pair[0] == pair[1] {
            let dup = pair[0];
            return Err(VokraError::ModelLoad(format!(
                "utmosv2: tensor name `{dup}` appears more than once — every `tensor_info` \
                 lookup for it would silently resolve to whichever copy was indexed first. \
                 Refusing the load (FR-EX-08)."
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Utmosv2 — the runtime binder handle.
// ---------------------------------------------------------------------------

/// UTMOSv2 (`sarulab-speech/UTMOSv2`, MIT) reference-free MOS predictor.
///
/// Bind with [`Self::from_gguf`] / [`Self::from_path`], then call
/// [`Self::predict_mos`]. See the module doc for the implementation-status
/// matrix and the FR-EX-08 loud-partial contract on the deferred multi-modal
/// forward.
#[derive(Debug, Clone)]
pub struct Utmosv2 {
    cfg: Utmosv2Config,
    weights: Utmosv2Weights,
}

impl Utmosv2 {
    /// Binds a UTMOSv2 GGUF: verifies the arch tag, parses the config, and
    /// shape-checks the whole tensor manifest.
    ///
    /// The arch check runs **first** so a sibling eval-family GGUF handed
    /// here by mistake fails with a specific message instead of a downstream
    /// missing-tensor error.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent, is not
    ///   a String, or is not [`ARCH`].
    /// - Whatever [`Utmosv2Config::from_gguf`] rejects.
    /// - Whatever [`Utmosv2Weights::from_gguf`] rejects.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        match gguf.get(chunks::KEY_MODEL_ARCH) {
            Some(v) => match v.as_str() {
                Some(a) if a == ARCH => {}
                Some(other) => return Err(wrong_arch(other)),
                None => {
                    let got = v.value_type();
                    return Err(VokraError::ModelLoad(format!(
                        "utmosv2: GGUF metadata `{key}` has type {got:?}, expected String — \
                         the runtime will not coerce a mis-typed arch tag (FR-EX-08).",
                        key = chunks::KEY_MODEL_ARCH,
                    )));
                }
            },
            None => {
                return Err(VokraError::ModelLoad(format!(
                    "utmosv2: GGUF is missing `{key}` — this is not a Vokra-native UTMOSv2 \
                     GGUF (was it produced by `{CONVERT_COMMAND}`?).",
                    key = chunks::KEY_MODEL_ARCH,
                )));
            }
        }

        let cfg = Utmosv2Config::from_gguf(gguf)?;
        let weights = Utmosv2Weights::from_gguf(gguf)?;
        Ok(Self { cfg, weights })
    }

    /// Opens and binds a UTMOSv2 GGUF from disk.
    ///
    /// # Errors
    ///
    /// - [`VokraError`] on I/O or GGUF parse failure, plus everything
    ///   [`Self::from_gguf`] rejects.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// The parsed config.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &Utmosv2Config {
        &self.cfg
    }

    /// The bound tensor manifest.
    #[inline]
    #[must_use]
    pub fn weights(&self) -> &Utmosv2Weights {
        &self.weights
    }

    /// The stamped weight-licence class.
    ///
    /// A stock conversion stamps [`LicenseClass::Permissive`] (MIT — recorded
    /// as verified against [`PRIMARY_SOURCE_LICENSE`] by the converter
    /// docstring). A GGUF missing the stamp reads back as
    /// [`LicenseClass::Unknown`] (fail-closed at the M2-13 compliance gate).
    #[inline]
    #[must_use]
    pub fn weight_license(&self) -> LicenseClass {
        self.cfg.weight_license
    }

    /// Number of bound tensors.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// The ACR scale this predictor's output lives on.
    #[inline]
    #[must_use]
    pub const fn mos_range() -> (f32, f32) {
        (MOS_MIN, MOS_MAX)
    }

    /// Predicts a reference-free MOS for a mono `f32` PCM clip.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Always returns [`VokraError::UnsupportedOp`]. The full UTMOSv2 forward
    /// needs the spectrogram-domain branch, the wav2vec2-large SSL encoder
    /// with its listener / domain conditioning, and the Regressor head that
    /// fuses them — and **none** of those topologies is transcribable from
    /// the artifacts this runtime can see: the conversion contract is a
    /// verbatim float pass-through that stamps no `vokra.utmosv2.*` axes, and
    /// the offline sidecar [`SIDECAR_PATH`] does not exist in-tree yet.
    /// Best-guessing the stack would be silent-wrong; fabricating a MOS
    /// scalar would corrupt the NFR-QL-02 5 % gate that consumes it.
    ///
    /// The gate fires **before** any input inspection so a caller can never
    /// confuse "clip too short" with "not implemented"; the real input
    /// contract (sample rate, minimum length, finiteness) lands with the
    /// forward, driven by [`KEY_UTMOSV2_SAMPLE_RATE`].
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate, naming every
    ///   deferred stage, the metadata keys to stamp, the sidecar to write,
    ///   the re-conversion command and both primary sources.
    pub fn predict_mos(&self, pcm: &[f32]) -> Result<f32> {
        // Bind explicitly so an unused-parameter warning cannot mask an
        // accidental future removal of the argument (the `emotion2vec` /
        // `dnsmos_p808_p835` loud-partial signature discipline).
        let _ = pcm;
        Err(predict_mos_loud_partial(&self.cfg.topology_keys_present))
    }
}

/// Builds the wrong-arch [`VokraError::ModelLoad`], enumerating the sibling
/// eval-family arch tags so a mis-routed GGUF names its own family.
fn wrong_arch(other: &str) -> VokraError {
    let siblings = SIBLING_EVAL_ARCH_TAGS.join("`, `");
    VokraError::ModelLoad(format!(
        "utmosv2: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF produced by \
         `{CONVERT_COMMAND}`?). The sibling eval-family arch tags — `{siblings}` — all \
         name reference-free quality instruments, but none shares UTMOSv2's topology or \
         output contract: `utmos` is UTMOS22-strong on a wav2vec2-BASE SSL encoder with a \
         different head layout (UTMOSv2 is wav2vec2-LARGE), `dnsmos` emits 1 or 3 ITU-T \
         P.808 / P.835 scalars over a fixed 9.01 s window rather than one utterance MOS, \
         `nisqa_v2_weight` is a multi-dimensional NISQA v2 model, and \
         `torchaudio_squim` is the torchaudio SQUIM objective / subjective pair. Silently \
         aliasing any of them would mis-route runtime dispatch onto a different tensor \
         walk and a different output width (FR-EX-08 — no silent partial load)."
    ))
}

/// Builds the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`Utmosv2::predict_mos`] until the multi-modal forward lands.
///
/// The message is the flip-the-switch recipe: it names all three deferred
/// stages, the concrete `vokra.utmosv2.*` keys the converter must stamp
/// (verbatim, from the `KEY_UTMOSV2_*` constants, so a rename cannot silently
/// drift the recipe), the absent sidecar, the re-conversion command and both
/// primary sources. `topology_keys_present` is echoed so a reader can tell
/// whether the converter half of the flip has already landed.
fn predict_mos_loud_partial(topology_keys_present: &[String]) -> VokraError {
    let topology_note = if topology_keys_present.is_empty() {
        format!(
            "This GGUF advertises no `{KEY_UTMOSV2_PREFIX}` topology axes, so neither half \
             of the flip has landed."
        )
    } else {
        format!(
            "This GGUF already advertises {n} `{KEY_UTMOSV2_PREFIX}` topology axes ({keys}) \
             — the converter half of the flip has landed but the runtime forward has not, \
             so wiring this function is the only remaining step.",
            n = topology_keys_present.len(),
            keys = topology_keys_present.join(", "),
        )
    };

    VokraError::UnsupportedOp(format!(
        "utmosv2 predict_mos (loud-partial): the multi-modal MOS forward is deferred; three \
         stages must land before a real MOS in [{MOS_MIN}, {MOS_MAX}] can be emitted: \
         (1) the spectrogram-domain branch — UTMOSv2 is described upstream as a \
         multi-modal system that fuses a spectrogram branch with an SSL branch, but the \
         in-repo conversion contract \
         (`crates/vokra-convert/src/models/utmosv2.rs`) is a verbatim float pass-through \
         that stamps no `{KEY_UTMOSV2_PREFIX}` topology axes, so the branch's layer stack \
         is not primary-source-transcribable from the GGUF and would be silent-wrong if \
         best-guessed; \
         (2) the wav2vec2-large SSL encoder walk plus the listener / domain conditioning \
         (the converter docstring's characterisation) — layer count, hidden width and \
         conditioning fan-in are all unstamped; \
         (3) the Regressor head that fuses both branches into one scalar, followed by the \
         ACR clamp (already real here as `clamp_to_mos_range`). \
         Flip-the-switch recipe: write the offline flatten sidecar `{SIDECAR_PATH}` (it \
         does not exist in the tree today), extend the converter to stamp \
         `{KEY_UTMOSV2_SAMPLE_RATE}` / `{KEY_UTMOSV2_SSL_N_LAYER}` / \
         `{KEY_UTMOSV2_SSL_HIDDEN_DIM}` / `{KEY_UTMOSV2_SPEC_N_MELS}` / \
         `{KEY_UTMOSV2_HEAD_DIMS}`, re-run `{CONVERT_COMMAND}`, then wire this forward \
         against those axes using `Utmosv2Weights::require` / `require_shape` / \
         `load_f32`. {topology_note} \
         Primary sources: reference code {PRIMARY_SOURCE_CODE}, paper \
         {PRIMARY_SOURCE_PAPER}. The runtime cannot fabricate a MOS scalar (FR-EX-08 — no \
         silent partial output; an invented score would silently corrupt the NFR-QL-02 5% \
         quality gate this instrument feeds)."
    ))
}
