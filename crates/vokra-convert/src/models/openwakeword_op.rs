//! **openWakeWord op wiring** (`dscripka/openWakeWord`, Apache-2.0
//! code): safetensors → GGUF conversion (coverage-audit-2026-08-03
//! Wave A permissive continuation, 2026-08-04; metadata handshake
//! repaired 2026-08-15).
//!
//! Input: user-provided openWakeWord model checkpoints. This converter
//! is deliberately a **runtime-op wiring companion** to the existing
//! `Openwakeword` ModelKind (2026-08-02 Wave residual, custom-KWS
//! MLP/CNN over precomputed melspec). The `_op` suffix signals that
//! the model kind primarily exists so the first-class `kws` op family
//! (CLAUDE.md audio-dialect §Streaming / VAD / KWS, `FR-OP kws`,
//! Porcupine-compatible) has a distinct runtime-dispatch anchor that
//! is decoupled from the base `openwakeword` converter's arch tag —
//! user-provided weights (either official CC-BY-NC-SA-4.0 downloads
//! the user obtains under their own compliance judgement OR
//! self-trained Apache-2.0 weights) route through this op-wiring path
//! and reach the runtime [`OpenwakewordSession::from_gguf`] binder
//! without silently masquerading as the base official-checkpoint
//! converter. Callers pre-flatten the upstream ONNX to safetensors
//! offline via `tools/parity/openwakeword_prepare_checkpoint.py`.
//!
//! [`OpenwakewordSession::from_gguf`]: https://docs.rs/vokra-models
//!
//! Output: a GGUF carrying every float tensor plus the `vokra.model.*`,
//! `vokra.provenance.*` **and `vokra.openwakeword.*`** metadata chunks
//! the runtime binder reads.
//!
//! # The 2026-08-15 handshake repair
//!
//! Until 2026-08-15 this converter stamped only `vokra.model.*` +
//! `vokra.provenance.*`, and the module doc claimed those were "the
//! chunks the runtime `kws` op binds against". They are not.
//! `OpenwakewordConfig::from_gguf` reads **seven** `vokra.openwakeword.*`
//! keys and treats every one as required, erroring with `ModelLoad` on
//! absence. So every GGUF this converter produced failed to load in the
//! binder written for it, and the owner recipe documented in
//! `crates/vokra-models/tests/parity_openwakeword.rs` dead-ended at the
//! first load.
//!
//! Nothing in the suite could see it: the binder's unit tests hand-build
//! their GGUF with `GgufBuilder` rather than running this converter, and
//! the parity harness is env-gated and skips. Tensor names matched all
//! along — only the metadata group was missing. The integration test
//! `crates/vokra-convert/tests/openwakeword_op_roundtrip.rs` (and the
//! convert→bind test in `vokra-models`) now pin the pair together.
//!
//! # Where each of the seven axes comes from
//!
//! Two are **derived from the tensors themselves**, so they cannot drift
//! away from the weights they describe:
//!
//! - `n_wakewords` — the length of the contiguous run of
//!   `openwakeword.classifier.{i}.*` groups. A gap (0, 1, 3) is a hard
//!   error, not a silent truncation.
//! - `embedding_dim` — dim 1 of `openwakeword.classifier.0.linear1.weight`,
//!   cross-checked against every other classifier. A disagreement is a
//!   hard error.
//!
//! One is **required from a `--config` side-car** because it exists
//! nowhere else:
//!
//! - `wakeword_names` — the per-wake-word labels the runtime returns
//!   from `KwsEngine::wakeword_names()` and in its `(name, prob)` pairs.
//!   These are **not in the safetensors at all**: the prepare script
//!   writes tensors under a positional index (`classifier.{idx}.…`) and
//!   keeps the names only in its own reference JSON, having explicitly
//!   refused to infer them ("no silent path-basename inference" —
//!   `openwakeword_prepare_checkpoint.py::_parse_wakeword_spec`).
//!   Synthesising `wakeword_0` here would be strictly worse than the
//!   inference that script already rejected, so this converter refuses
//!   the plain path instead (the [`ModelKind::Crepe`] precedent).
//!
//! Four are **mirrors of named constants in the runtime binder**, each
//! overridable from the same side-car:
//!
//! - `window_frames` (76), `mel_bins` (32), `sample_rate` (16000),
//!   `hop_samples` (160) — see [`DEFAULT_WINDOW_FRAMES`] and siblings
//!   for the citation and the safety argument on each.
//!
//! # License
//!
//! - SPDX default: **Apache-2.0** ([`vokra_core::LicenseClass::Permissive`])
//!   — the Apache-2.0 code license of the upstream openWakeWord
//!   project. Official weights on the release page are
//!   CC-BY-NC-SA-4.0, so a caller who has downloaded them must
//!   override at the CLI boundary (`--license cc-by-nc-sa-4.0`); the
//!   fail-closed disposition then flips to NonCommercialShareAlike
//!   and publish gate refuses without `--allow-noncommercial`.
//! - Category: **vad-kws** (keyword-spotting / wake-word — sibling of
//!   `silero-vad` / `fsmn_vad` / `ten_vad` under the shared `vad-kws`
//!   umbrella covering VAD + KWS families; distinct from the base
//!   `openwakeword` converter's arch tag).
//! - Notes: **Vokra does not redistribute openWakeWord official
//!   weights** — the upstream repo's release-page CC-BY-NC-SA-4.0
//!   term is not compatible with Vokra's default commercial-mode
//!   redistribution policy. The `_op` runtime-wiring path is for
//!   user-provided weights only; no §3.1 sign-off is required for the
//!   op-wiring converter itself because Vokra does not publish
//!   op-wiring artefacts.
//!
//! # BF16 pass-through (mirror of sensevoicesmall / neucodec /
//! # ecapa_tdnn / speaker_3d)
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm on the same
//! match arm — no convert-time widening. BF16 is emitted as GGUF type
//! 30 ([`GgmlType::BF16`]); the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the keys the prepare script emits, verbatim.
//! Native v0.5.1 artifacts carry execution-order classifier tensors under
//! `openwakeword.classifier.{i}.linear.{j}.{weight,bias}`, canonical learned
//! DFT/mel tensors under `openwakeword.melspec.*`, and 20 canonical Conv2d
//! groups under `openwakeword.embedding.conv.{0..19}.*`. The converter
//! validates all fixed shapes before writing. The older
//! `linear{1,2}` classifier-only layout remains readable for compatibility.
//!
//! # Arch tag distinctness
//!
//! `vokra.model.arch = "openwakeword_op"` is intentionally distinct
//! from the sibling `openwakeword` (base official-checkpoint
//! converter, 2026-08-02 Wave residual). The `_op` variant is the
//! runtime-op-wiring anchor that user-provided weights route through
//! — silently sharing an arch tag with the base ModelKind would
//! blur the op-wiring vs published-artefact boundary and hide the
//! distinct license-override contract from the runtime dispatch.
//!
//! # No ONNX (permanent) in the runtime
//!
//! The upstream openWakeWord release ships ONNX + TFLite;
//! `tools/parity/openwakeword_prepare_checkpoint.py` flattens the graph
//! initializers to safetensors offline so the runtime never touches the
//! ONNX (FR-LD-05, NFR-DS-02).
//!
//! # Wiring status
//!
//! The native v0.5.1 load and streaming forward are complete. Official
//! release weights remain user-provided and non-redistributed; the gated
//! real-weight harness compares hop probabilities against ONNX Runtime.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};
use vokra_core::json::{self, JsonValue};

use crate::ConvertError;
use crate::safetensors::{SafeTensorInfo, SafetensorsFile};

/// `vokra.model.arch` value for openWakeWord op-wiring GGUFs.
/// Intentionally distinct from the sibling base `openwakeword` arch
/// tag (2026-08-02 Wave residual) — the `_op` variant is the runtime-
/// op-wiring anchor, decoupled from the base converter's arch tag so
/// the runtime dispatch sees the two as different topologies with
/// different license-override contracts.
pub const ARCH: &str = "openwakeword_op";

/// `vokra.model.name` value written for the canonical
/// `dscripka/openWakeWord` op-wiring release.
pub const NAME: &str = "openwakeword_op";

/// `vokra.model.category` value written for every openWakeWord op
/// GGUF. Sibling of `silero-vad` / `fsmn_vad` / `ten_vad` under the
/// shared `vad-kws` umbrella covering VAD + KWS families.
pub const CATEGORY: &str = "vad-kws";

/// Upstream HF repository slug (`org/name`) — canonical HF mirror of
/// the openWakeWord family. Note: Vokra does NOT redistribute the
/// upstream official CC-BY-NC-SA-4.0 weights; this slug is recorded
/// as provenance only.
pub const UPSTREAM_HF: &str = "dscripka/openWakeWord";

/// Default upstream code licence (SPDX) — Apache-2.0 code. Official
/// weights are CC-BY-NC-SA-4.0, but Vokra does not redistribute
/// them, and callers who supply their own Apache-2.0 self-trained
/// weights can keep the Permissive default; callers who redistribute
/// the CC-BY-NC-SA-4.0 official weights must override to
/// `--license cc-by-nc-sa-4.0` which flips the fail-closed publish
/// gate to NonCommercialShareAlike.
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication
// convention the sibling BF16-passthrough converters use applies).

/// `vokra.model.category` metadata key. Local per the established
/// sensevoicesmall / nkf_aec / funcodec convention.
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_hf` metadata key. Local per the same
/// convention as [`KEY_MODEL_CATEGORY`].
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

// ---- vokra.openwakeword.* metadata keys ---------------------------------
//
// Duplicated from `vokra-models::kws::openwakeword` rather than imported:
// `vokra-convert` depends only on `vokra-core` / `vokra-ops` / `vokra-mmap`,
// and adding a converter → models edge would invert the dependency
// direction the whole crate exists to keep clean. This is the same
// cross-crate constant-duplication convention every sibling converter
// uses; the round-trip test in `crates/vokra-convert/tests/` is what
// keeps the two copies honest, since a typo here produces a GGUF the
// binder rejects and the test fails on the missing key.

/// GGUF metadata key: number of per-wake-word classifiers (u32).
pub const KEY_N_WAKEWORDS: &str = "vokra.openwakeword.n_wakewords";
/// GGUF metadata key: shared embedding width (u32).
pub const KEY_EMBEDDING_DIM: &str = "vokra.openwakeword.embedding_dim";
/// GGUF metadata key: rolling melspec window in frames (u32).
pub const KEY_WINDOW_FRAMES: &str = "vokra.openwakeword.window_frames";
/// GGUF metadata key: mel-bin count per frame (u32).
pub const KEY_MEL_BINS: &str = "vokra.openwakeword.mel_bins";
/// GGUF metadata key: PCM sample rate the checkpoint expects (u32 Hz).
pub const KEY_SAMPLE_RATE: &str = "vokra.openwakeword.sample_rate";
/// GGUF metadata key: analysis hop between melspec frames (u32 samples).
pub const KEY_HOP_SAMPLES: &str = "vokra.openwakeword.hop_samples";
/// GGUF metadata key: per-wake-word names (`Array<String>`).
pub const KEY_WAKEWORD_NAMES: &str = "vokra.openwakeword.wakeword_names";
/// GGUF metadata key: classifier tensor topology identifier (string).
pub const KEY_CLASSIFIER_FORMAT: &str = "vokra.openwakeword.classifier_format";
/// GGUF metadata key: rolling embedding frames consumed per prediction (u32).
pub const KEY_CLASSIFIER_INPUT_FRAMES: &str = "vokra.openwakeword.classifier_input_frames";
/// GGUF metadata key: dense-layer count for each wake-word (`Array<U32>`).
pub const KEY_CLASSIFIER_LAYER_COUNTS: &str = "vokra.openwakeword.classifier_layer_counts";
/// GGUF metadata key: PCM samples consumed per upstream prediction (u32).
pub const KEY_PREDICT_CHUNK_SAMPLES: &str = "vokra.openwakeword.predict_chunk_samples";

const CLASSIFIER_FORMAT_DNN: &str = "dnn-relu-sigmoid-v1";
const CLASSIFIER_FORMAT_LEGACY: &str = "legacy-two-layer-v1";

// ---- mirrored front-end defaults ----------------------------------------
//
// WHY MIRRORING IS SAFE FOR THESE FOUR, AND ONLY THESE FOUR
//
// Each value below is transcribed from the runtime binder's own module
// documentation (`crates/vokra-models/src/kws/openwakeword/mod.rs`,
// the `vokra.openwakeword.*` chunk-group section), which is an in-tree
// constant this converter can cite rather than a number invented here.
// Two independent in-tree corroborations exist, both in
// `crates/vokra-models/tests/parity_openwakeword.rs`: it hard-asserts
// `cfg.sample_rate == 16_000`, and it sizes its melspec window fixture
// as `76 * 32`.
//
// The safety argument is narrower than "the numbers look right":
//
//   1. These are FRONT-END framing axes (how PCM becomes a melspec
//      window), not weight-shape axes. A wrong value here cannot
//      silently reshape a tensor — the two axes that do constrain
//      tensor shapes (`n_wakewords`, `embedding_dim`) are derived from
//      the tensors themselves, never mirrored.
//   2. A wrong `sample_rate` cannot run silently: the binder's
//      `push_pcm16k` refuses any rate other than 16 kHz outright
//      (FR-EX-08), so the failure is loud.
//   3. Native v0.5.1 artifacts consume all three axes and validate their
//      fixed values at load. Legacy classifier-only artifacts retain the
//      override surface for compatibility.
//   4. Every one is overridable from the `--config` side-car, so a
//      self-trained checkpoint with a different front-end is expressible
//      without touching this file.
//
// `wakeword_names` is deliberately absent from this list: it is a
// per-checkpoint label with no constant to mirror, which is why the
// side-car is required rather than optional.

/// Default `vokra.openwakeword.window_frames` — rolling melspec window
/// the embedding extractor consumes, in frames.
///
/// Mirror of the runtime binder's documented value (76 frames). Override
/// with `"window_frames"` in the `--config` side-car.
pub const DEFAULT_WINDOW_FRAMES: u32 = 76;

/// Default `vokra.openwakeword.mel_bins` — melspec width per frame.
///
/// Mirror of the runtime binder's documented value (32 bins). Override
/// with `"mel_bins"` in the `--config` side-car.
pub const DEFAULT_MEL_BINS: u32 = 32;

/// Default `vokra.openwakeword.sample_rate` — PCM sample rate in Hz.
///
/// Mirror of the runtime binder's documented value (16 000 Hz), which
/// `OpenwakewordSession::push_pcm16k` additionally enforces at runtime.
/// Override with `"sample_rate"` in the `--config` side-car.
pub const DEFAULT_SAMPLE_RATE: u32 = 16_000;

/// Default `vokra.openwakeword.hop_samples` — analysis hop between
/// melspec frames, in samples.
///
/// Mirror of the runtime binder's documented value (160 samples = 10 ms
/// at 16 kHz). Note this is the **mel analysis hop**, not the 1280-sample
/// (80 ms) chunk `openwakeword.Model.predict` consumes per call — the
/// prepare script's reference JSON records that larger figure under its
/// own `hop_samples` key and the two must not be conflated. The parity
/// harness corroborates the 160 reading by taking `hop * 8` samples and
/// describing the slice as "8 hops = ~80 ms". Override with
/// `"hop_samples"` in the `--config` side-car.
pub const DEFAULT_HOP_SAMPLES: u32 = 160;

/// Tensor-name prefix shared by every per-wake-word classifier group.
pub const CLASSIFIER_PREFIX: &str = "openwakeword.classifier.";

/// Formats the per-wake-word first-linear weight tensor name
/// (row-major `[hidden_dim, embedding_dim]`).
fn tensor_l1_weight(idx: usize) -> String {
    format!("openwakeword.classifier.{idx}.linear1.weight")
}
/// Formats the per-wake-word first-linear bias tensor name (`[hidden_dim]`).
fn tensor_l1_bias(idx: usize) -> String {
    format!("openwakeword.classifier.{idx}.linear1.bias")
}
/// Formats the per-wake-word output-linear weight tensor name
/// (row-major `[1, hidden_dim]`).
fn tensor_l2_weight(idx: usize) -> String {
    format!("openwakeword.classifier.{idx}.linear2.weight")
}
/// Formats the per-wake-word output-linear bias tensor name (`[1]`).
fn tensor_l2_bias(idx: usize) -> String {
    format!("openwakeword.classifier.{idx}.linear2.bias")
}

fn tensor_dnn_weight(idx: usize, layer: usize) -> String {
    format!("openwakeword.classifier.{idx}.linear.{layer}.weight")
}

fn tensor_dnn_bias(idx: usize, layer: usize) -> String {
    format!("openwakeword.classifier.{idx}.linear.{layer}.bias")
}

/// Outcome of an openWakeWord op-wiring conversion.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct OpenwakewordOpReport {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16.
    pub bf16_passthrough: usize,
    /// Per-wake-word classifier groups discovered in the input, and
    /// therefore the value stamped into [`KEY_N_WAKEWORDS`].
    pub n_wakewords: usize,
    /// Shared embedding width derived from the classifier weights, and
    /// therefore the value stamped into [`KEY_EMBEDDING_DIM`].
    pub embedding_dim: usize,
}

/// Parsed openWakeWord op-wiring config side-car.
///
/// Only `wakeword_names` is required; the four front-end axes fall back
/// to the mirrored [`DEFAULT_WINDOW_FRAMES`] family when omitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OpenwakewordOpConvertConfig {
    /// Per-wake-word labels, in the same order as the positional
    /// `openwakeword.classifier.{i}.*` tensor groups.
    pub(crate) wakeword_names: Vec<String>,
    /// Rolling melspec window in frames.
    pub(crate) window_frames: u32,
    /// Mel-bin count per frame.
    pub(crate) mel_bins: u32,
    /// PCM sample rate in Hz.
    pub(crate) sample_rate: u32,
    /// Analysis hop between melspec frames, in samples.
    pub(crate) hop_samples: u32,
    /// Classifier tensor topology.
    pub(crate) classifier_format: String,
    /// Rolling embedding frames flattened into each DNN.
    pub(crate) classifier_input_frames: u32,
    /// Dense-layer count for every classifier group.
    pub(crate) classifier_layer_counts: Vec<u32>,
    /// PCM samples consumed for one prediction.
    pub(crate) predict_chunk_samples: u32,
}

impl OpenwakewordOpConvertConfig {
    /// Parses the JSON side-car.
    ///
    /// Schema:
    ///
    /// ```json
    /// {
    ///   "wakeword_names": ["alexa", "hey_jarvis"],
    ///   "window_frames": 76,
    ///   "mel_bins": 32,
    ///   "sample_rate": 16000,
    ///   "hop_samples": 160
    /// }
    /// ```
    ///
    /// Every field but `wakeword_names` is optional and defaults to the
    /// mirrored constant of the same name.
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, ConvertError> {
        let root = json::parse(bytes).map_err(|e| ConvertError::Parse(e.to_string()))?;

        let names_json = root
            .get("wakeword_names")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| {
                ConvertError::Parse(
                    "openwakeword-op config: required array field `wakeword_names` is missing \
                     or not an array. These labels exist nowhere in the safetensors (the \
                     prepare script writes classifier tensors under a positional index and \
                     keeps the names only in its reference JSON), so the converter cannot \
                     derive them and will not invent them."
                        .to_owned(),
                )
            })?;

        let mut wakeword_names: Vec<String> = Vec::with_capacity(names_json.len());
        for (i, v) in names_json.iter().enumerate() {
            let s = v.as_str().ok_or_else(|| {
                ConvertError::Parse(format!(
                    "openwakeword-op config: `wakeword_names[{i}]` is not a string"
                ))
            })?;
            if s.trim().is_empty() {
                return Err(ConvertError::Parse(format!(
                    "openwakeword-op config: `wakeword_names[{i}]` is empty — the runtime \
                     returns these labels to callers, so a blank one is never intended"
                )));
            }
            if wakeword_names.iter().any(|prev| prev == s) {
                return Err(ConvertError::Parse(format!(
                    "openwakeword-op config: `wakeword_names[{i}]` = `{s}` is a duplicate — \
                     the runtime keys its `(name, prob)` output on these labels, so two \
                     classifiers sharing one name would be indistinguishable"
                )));
            }
            wakeword_names.push(s.to_owned());
        }
        if wakeword_names.is_empty() {
            return Err(ConvertError::Parse(
                "openwakeword-op config: `wakeword_names` is empty — a GGUF with zero \
                 wake-words has nothing for the runtime to classify"
                    .to_owned(),
            ));
        }

        let opt_u32 = |key: &str, default: u32| -> Result<u32, ConvertError> {
            let Some(v) = root.get(key) else {
                return Ok(default);
            };
            let raw = v.as_u64().ok_or_else(|| {
                ConvertError::Parse(format!(
                    "openwakeword-op config: `{key}` must be a positive integer"
                ))
            })?;
            let narrowed = u32::try_from(raw).map_err(|_| {
                ConvertError::Parse(format!(
                    "openwakeword-op config: `{key}` = {raw} does not fit in u32"
                ))
            })?;
            if narrowed == 0 {
                return Err(ConvertError::Parse(format!(
                    "openwakeword-op config: `{key}` must be > 0 (the runtime binder's \
                     `OpenwakewordConfig::validate` refuses a 0-sentinel on every hparam)"
                )));
            }
            Ok(narrowed)
        };

        let classifier_format = root
            .get("classifier_format")
            .and_then(JsonValue::as_str)
            .unwrap_or(CLASSIFIER_FORMAT_LEGACY)
            .to_owned();
        if classifier_format != CLASSIFIER_FORMAT_LEGACY
            && classifier_format != CLASSIFIER_FORMAT_DNN
        {
            return Err(ConvertError::Parse(format!(
                "openwakeword-op config: unsupported `classifier_format` `{classifier_format}`"
            )));
        }
        let classifier_input_frames = opt_u32(
            "classifier_input_frames",
            if classifier_format == CLASSIFIER_FORMAT_DNN {
                16
            } else {
                1
            },
        )?;
        let classifier_layer_counts = match root.get("classifier_layer_counts") {
            Some(value) => value
                .as_array()
                .ok_or_else(|| {
                    ConvertError::Parse(
                        "openwakeword-op config: `classifier_layer_counts` must be an array"
                            .to_owned(),
                    )
                })?
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let raw = value.as_u64().ok_or_else(|| {
                        ConvertError::Parse(format!(
                            "openwakeword-op config: `classifier_layer_counts[{index}]` must be a positive integer"
                        ))
                    })?;
                    let count = u32::try_from(raw).map_err(|_| {
                        ConvertError::Parse(format!(
                            "openwakeword-op config: `classifier_layer_counts[{index}]` overflows u32"
                        ))
                    })?;
                    if count == 0 {
                        return Err(ConvertError::Parse(format!(
                            "openwakeword-op config: `classifier_layer_counts[{index}]` must be > 0"
                        )));
                    }
                    Ok(count)
                })
                .collect::<Result<Vec<_>, ConvertError>>()?,
            None if classifier_format == CLASSIFIER_FORMAT_DNN => {
                return Err(ConvertError::Parse(
                    "openwakeword-op config: DNN format requires `classifier_layer_counts`"
                        .to_owned(),
                ));
            }
            None => vec![2; wakeword_names.len()],
        };
        if classifier_layer_counts.len() != wakeword_names.len() {
            return Err(ConvertError::Parse(format!(
                "openwakeword-op config: classifier_layer_counts has {} entries, expected {} wake-word entries",
                classifier_layer_counts.len(),
                wakeword_names.len()
            )));
        }

        Ok(Self {
            wakeword_names,
            window_frames: opt_u32("window_frames", DEFAULT_WINDOW_FRAMES)?,
            mel_bins: opt_u32("mel_bins", DEFAULT_MEL_BINS)?,
            sample_rate: opt_u32("sample_rate", DEFAULT_SAMPLE_RATE)?,
            hop_samples: opt_u32("hop_samples", DEFAULT_HOP_SAMPLES)?,
            classifier_format,
            classifier_input_frames,
            classifier_layer_counts,
            predict_chunk_samples: opt_u32("predict_chunk_samples", 1_280)?,
        })
    }
}

/// Classifier axes derived from the tensors, never from a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ClassifierAxes {
    n_wakewords: usize,
    embedding_dim: usize,
}

/// True when the dtype rides the verbatim pass-through arm, i.e. when
/// the runtime's `GgufFile::tensor_f32` can widen it back to f32.
fn is_passthrough_float(dtype: GgmlType) -> bool {
    matches!(dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
}

/// Looks a required classifier tensor up, failing loudly with the
/// binder-facing reason when it is absent or non-float.
///
/// A non-float classifier tensor is treated as an error rather than a
/// skip: the generic pass-through loop would silently drop it, and the
/// GGUF would then fail in the binder with a confusing "missing tensor"
/// for a tensor that was present in the input all along.
fn require_classifier_tensor<'a>(
    st: &'a SafetensorsFile,
    name: &str,
) -> Result<&'a SafeTensorInfo, ConvertError> {
    let info = st.tensor_info(name).ok_or_else(|| {
        ConvertError::Parse(format!(
            "openwakeword-op: required classifier tensor `{name}` is missing from the \
             safetensors. The runtime binder loads all four of \
             `linear{{1,2}}.{{weight,bias}}` per wake-word; re-run \
             tools/parity/openwakeword_prepare_checkpoint.py to regenerate a complete input."
        ))
    })?;
    if !is_passthrough_float(info.dtype) {
        return Err(ConvertError::Parse(format!(
            "openwakeword-op: classifier tensor `{name}` has dtype {:?}, which is not one of \
             F32 / F16 / BF16. The pass-through arm would skip it and the emitted GGUF would \
             fail to load with a misleading `missing tensor` error.",
            info.dtype
        )));
    }
    Ok(info)
}

/// Derives `n_wakewords` + `embedding_dim` from the classifier tensors
/// and validates every per-wake-word group against the exact shape
/// contract `OpenwakewordSession::from_gguf` enforces.
///
/// Doing the binder's own checks here means a conversion either produces
/// a loadable GGUF or fails at convert time with a message naming the
/// offending tensor — the failure never gets deferred to the operator's
/// first `from_gguf`.
fn derive_classifier_axes(st: &SafetensorsFile) -> Result<ClassifierAxes, ConvertError> {
    // Length of the contiguous run of groups starting at index 0.
    let mut n_wakewords = 0usize;
    while st.tensor_info(&tensor_l1_weight(n_wakewords)).is_some() {
        n_wakewords += 1;
    }
    if n_wakewords == 0 {
        return Err(ConvertError::Parse(format!(
            "openwakeword-op: no `{CLASSIFIER_PREFIX}0.linear1.weight` tensor found. This \
             converter expects the classifier layout that \
             tools/parity/openwakeword_prepare_checkpoint.py emits \
             (`{CLASSIFIER_PREFIX}{{i}}.linear{{1,2}}.{{weight,bias}}`), which is also what \
             the runtime binder reads back."
        )));
    }

    // A gap (0, 1, 3) would otherwise be silently truncated to the run
    // length, dropping a wake-word the operator supplied.
    for t in st.tensors() {
        let Some(rest) = t.name.strip_prefix(CLASSIFIER_PREFIX) else {
            continue;
        };
        let idx_str = rest.split('.').next().unwrap_or("");
        let Ok(idx) = idx_str.parse::<usize>() else {
            return Err(ConvertError::Parse(format!(
                "openwakeword-op: tensor `{}` sits under the classifier prefix but `{idx_str}` \
                 is not a group index",
                t.name
            )));
        };
        if idx >= n_wakewords {
            return Err(ConvertError::Parse(format!(
                "openwakeword-op: tensor `{}` carries classifier index {idx}, but the \
                 contiguous run of groups ends at {n_wakewords}. The indices must be dense \
                 from 0 — otherwise this wake-word would be dropped without a word.",
                t.name
            )));
        }
    }

    // `embedding_dim` from group 0, then cross-checked against the rest.
    let mut embedding_dim: Option<usize> = None;
    for i in 0..n_wakewords {
        let l1w_name = tensor_l1_weight(i);
        let l1w = require_classifier_tensor(st, &l1w_name)?;
        if l1w.shape.len() != 2 {
            return Err(ConvertError::Parse(format!(
                "openwakeword-op: tensor `{l1w_name}` has rank {}, expected rank 2 \
                 [hidden_dim, embedding_dim]",
                l1w.shape.len()
            )));
        }
        let hidden = l1w.shape[0];
        let embed = l1w.shape[1];
        if hidden == 0 || embed == 0 {
            return Err(ConvertError::Parse(format!(
                "openwakeword-op: tensor `{l1w_name}` has dims [{hidden}, {embed}]; both must \
                 be > 0"
            )));
        }

        match embedding_dim {
            None => embedding_dim = Some(embed as usize),
            Some(prev) if prev as u64 != embed => {
                return Err(ConvertError::Parse(format!(
                    "openwakeword-op: tensor `{l1w_name}` implies embedding_dim={embed}, but \
                     earlier classifiers implied {prev}. Every wake-word head consumes the one \
                     shared embedding, so a disagreement means the inputs were merged from \
                     incompatible releases."
                )));
            }
            Some(_) => {}
        }

        let l1b_name = tensor_l1_bias(i);
        let l1b = require_classifier_tensor(st, &l1b_name)?;
        if l1b.shape.as_slice() != [hidden] {
            return Err(ConvertError::Parse(format!(
                "openwakeword-op: tensor `{l1b_name}` has dims {:?}, expected [{hidden}] to \
                 match `{l1w_name}`'s hidden_dim",
                l1b.shape
            )));
        }

        let l2w_name = tensor_l2_weight(i);
        let l2w = require_classifier_tensor(st, &l2w_name)?;
        if l2w.shape.as_slice() != [1, hidden] {
            return Err(ConvertError::Parse(format!(
                "openwakeword-op: tensor `{l2w_name}` has dims {:?}, expected [1, {hidden}] \
                 row-major (one binary output class per wake-word)",
                l2w.shape
            )));
        }

        let l2b_name = tensor_l2_bias(i);
        let l2b = require_classifier_tensor(st, &l2b_name)?;
        if l2b.shape.as_slice() != [1] {
            return Err(ConvertError::Parse(format!(
                "openwakeword-op: tensor `{l2b_name}` has dims {:?}, expected [1]",
                l2b.shape
            )));
        }
    }

    let embedding_dim = embedding_dim.ok_or_else(|| {
        ConvertError::Parse(
            "openwakeword-op: embedding_dim could not be derived — GGUF invariant broken"
                .to_owned(),
        )
    })?;

    Ok(ClassifierAxes {
        n_wakewords,
        embedding_dim,
    })
}

fn derive_dnn_classifier_axes(
    st: &SafetensorsFile,
    cfg: &OpenwakewordOpConvertConfig,
) -> Result<ClassifierAxes, ConvertError> {
    let n_wakewords = cfg.classifier_layer_counts.len();
    if n_wakewords == 0 {
        return Err(ConvertError::Parse(
            "openwakeword-op: DNN classifier list is empty".to_owned(),
        ));
    }
    let input_frames = cfg.classifier_input_frames as usize;
    let first_name = tensor_dnn_weight(0, 0);
    let first = require_classifier_tensor(st, &first_name)?;
    if first.shape.len() != 2 || first.shape[0] == 0 || first.shape[1] == 0 {
        return Err(ConvertError::Parse(format!(
            "openwakeword-op: tensor `{first_name}` has dims {:?}, expected non-empty [out, frames*embedding]",
            first.shape
        )));
    }
    let flattened = first.shape[1] as usize;
    if !flattened.is_multiple_of(input_frames) {
        return Err(ConvertError::Parse(format!(
            "openwakeword-op: tensor `{first_name}` input width {flattened} is not divisible by classifier_input_frames={input_frames}"
        )));
    }
    let embedding_dim = flattened / input_frames;

    for (classifier, &layer_count) in cfg.classifier_layer_counts.iter().enumerate() {
        let mut expected_input = input_frames * embedding_dim;
        for layer in 0..layer_count as usize {
            let weight_name = tensor_dnn_weight(classifier, layer);
            let bias_name = tensor_dnn_bias(classifier, layer);
            let weight = require_classifier_tensor(st, &weight_name)?;
            let bias = require_classifier_tensor(st, &bias_name)?;
            if weight.shape.len() != 2
                || weight.shape[1] as usize != expected_input
                || weight.shape[0] == 0
            {
                return Err(ConvertError::Parse(format!(
                    "openwakeword-op: tensor `{weight_name}` has dims {:?}, expected [out, {expected_input}]",
                    weight.shape
                )));
            }
            if bias.shape.as_slice() != [weight.shape[0]] {
                return Err(ConvertError::Parse(format!(
                    "openwakeword-op: tensor `{bias_name}` has dims {:?}, expected [{}]",
                    bias.shape, weight.shape[0]
                )));
            }
            expected_input = weight.shape[0] as usize;
        }
        if expected_input != 1 {
            return Err(ConvertError::Parse(format!(
                "openwakeword-op: classifier {classifier} final width is {expected_input}, expected 1"
            )));
        }
        let unexpected = tensor_dnn_weight(classifier, layer_count as usize);
        if st.tensor_info(&unexpected).is_some() {
            return Err(ConvertError::Parse(format!(
                "openwakeword-op: classifier {classifier} has undeclared layer `{unexpected}`; update classifier_layer_counts"
            )));
        }
    }

    for tensor in st.tensors() {
        let Some(rest) = tensor.name.strip_prefix(CLASSIFIER_PREFIX) else {
            continue;
        };
        let index = rest
            .split('.')
            .next()
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| {
                ConvertError::Parse(format!(
                    "openwakeword-op: malformed classifier tensor `{}`",
                    tensor.name
                ))
            })?;
        if index >= n_wakewords {
            return Err(ConvertError::Parse(format!(
                "openwakeword-op: tensor `{}` carries classifier index {index}, but config declares {n_wakewords} groups",
                tensor.name
            )));
        }
    }

    Ok(ClassifierAxes {
        n_wakewords,
        embedding_dim,
    })
}

fn require_native_tensor_shape(
    st: &SafetensorsFile,
    name: &str,
    expected: &[u64],
) -> Result<(), ConvertError> {
    let tensor = require_classifier_tensor(st, name)?;
    if tensor.shape.as_slice() != expected {
        return Err(ConvertError::Parse(format!(
            "openwakeword-op: tensor `{name}` has dims {:?}, expected {expected:?}",
            tensor.shape
        )));
    }
    Ok(())
}

fn validate_native_frontend_bundle(st: &SafetensorsFile) -> Result<(), ConvertError> {
    require_native_tensor_shape(st, "openwakeword.melspec.dft_real", &[257, 512])?;
    require_native_tensor_shape(st, "openwakeword.melspec.dft_imag", &[257, 512])?;
    require_native_tensor_shape(st, "openwakeword.melspec.mel", &[257, 32])?;

    const CONVS: [(u64, u64, u64, u64); 20] = [
        (1, 24, 3, 3),
        (24, 24, 1, 3),
        (24, 24, 3, 1),
        (24, 48, 1, 3),
        (48, 48, 3, 1),
        (48, 48, 1, 3),
        (48, 48, 3, 1),
        (48, 72, 1, 3),
        (72, 72, 3, 1),
        (72, 72, 1, 3),
        (72, 72, 3, 1),
        (72, 96, 1, 3),
        (96, 96, 3, 1),
        (96, 96, 1, 3),
        (96, 96, 3, 1),
        (96, 96, 1, 3),
        (96, 96, 3, 1),
        (96, 96, 1, 3),
        (96, 96, 3, 1),
        (96, 96, 3, 1),
    ];
    for (index, (input, output, kh, kw)) in CONVS.into_iter().enumerate() {
        require_native_tensor_shape(
            st,
            &format!("openwakeword.embedding.conv.{index}.weight"),
            &[output, input, kh, kw],
        )?;
        let bias_name = format!("openwakeword.embedding.conv.{index}.bias");
        if index == 19 {
            if st.tensor_info(&bias_name).is_some() {
                return Err(ConvertError::Parse(format!(
                    "openwakeword-op: final embedding convolution must not carry `{bias_name}`"
                )));
            }
        } else {
            require_native_tensor_shape(st, &bias_name, &[output])?;
        }
    }
    Ok(())
}

/// Converts a parsed safetensors buffer plus its config side-car into a
/// populated [`GgufBuilder`].
///
/// Split out from the file-level entry point the way `models::crepe`
/// splits its own `convert`, so the caller owns the I/O boundary.
pub(crate) fn convert(
    bytes: Vec<u8>,
    cfg: &OpenwakewordOpConvertConfig,
    license: Option<&str>,
) -> Result<(GgufBuilder, OpenwakewordOpReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;
    let axes = if cfg.classifier_format == CLASSIFIER_FORMAT_DNN {
        validate_native_frontend_bundle(&st)?;
        derive_dnn_classifier_axes(&st, cfg)?
    } else {
        derive_classifier_axes(&st)?
    };

    if cfg.wakeword_names.len() != axes.n_wakewords {
        return Err(ConvertError::Usage(format!(
            "openwakeword-op: the config side-car lists {} wake-word name(s) but the \
             safetensors carries {} classifier group(s). The runtime binder requires \
             `wakeword_names.len() == n_wakewords` and refuses the load otherwise, so \
             stamping the mismatch would only move the failure downstream.",
            cfg.wakeword_names.len(),
            axes.n_wakewords
        )));
    }

    let n_wakewords_u32 = u32::try_from(axes.n_wakewords).map_err(|_| {
        ConvertError::Parse(format!(
            "openwakeword-op: n_wakewords={} does not fit in u32",
            axes.n_wakewords
        ))
    })?;
    let embedding_dim_u32 = u32::try_from(axes.embedding_dim).map_err(|_| {
        ConvertError::Parse(format!(
            "openwakeword-op: embedding_dim={} does not fit in u32",
            axes.embedding_dim
        ))
    })?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // The `vokra.openwakeword.*` chunk group the runtime binder reads.
    // All seven keys are required by `OpenwakewordConfig::from_gguf`;
    // omitting any one of them produces a GGUF that cannot load.
    b.add_u32(KEY_N_WAKEWORDS, n_wakewords_u32);
    b.add_u32(KEY_EMBEDDING_DIM, embedding_dim_u32);
    b.add_u32(KEY_WINDOW_FRAMES, cfg.window_frames);
    b.add_u32(KEY_MEL_BINS, cfg.mel_bins);
    b.add_u32(KEY_SAMPLE_RATE, cfg.sample_rate);
    b.add_u32(KEY_HOP_SAMPLES, cfg.hop_samples);
    b.add_metadata(
        KEY_WAKEWORD_NAMES,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: cfg
                .wakeword_names
                .iter()
                .map(|s| GgufMetadataValue::String(s.clone()))
                .collect(),
        }),
    );
    if cfg.classifier_format == CLASSIFIER_FORMAT_DNN {
        b.add_string(KEY_CLASSIFIER_FORMAT, &cfg.classifier_format);
        b.add_u32(KEY_CLASSIFIER_INPUT_FRAMES, cfg.classifier_input_frames);
        b.add_u32(KEY_PREDICT_CHUNK_SAMPLES, cfg.predict_chunk_samples);
        b.add_metadata(
            KEY_CLASSIFIER_LAYER_COUNTS,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::U32,
                values: cfg
                    .classifier_layer_counts
                    .iter()
                    .copied()
                    .map(GgufMetadataValue::U32)
                    .collect(),
            }),
        );
    }

    let effective_spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let effective_class = LicenseClass::from_license_str(effective_spdx);
    vokra_core::stamp_provenance(
        &mut b,
        effective_class,
        effective_spdx,
        Some(NAME),
        Some(
            "dscripka/openWakeWord op-wiring (custom-KWS MLP/CNN over precomputed melspec, \
             Apache-2.0 code / CC-BY-NC-SA-4.0 official weights — Vokra does not \
             redistribute official weights; user-provided weights only, override to \
             --license cc-by-nc-sa-4.0 when distributing official CC-BY-NC-SA-4.0)",
        ),
    );

    let mut report = OpenwakewordOpReport {
        n_wakewords: axes.n_wakewords,
        embedding_dim: axes.embedding_dim,
        ..OpenwakewordOpReport::default()
    };
    for t in st.tensors() {
        report.read += 1;
        if is_passthrough_float(t.dtype) {
            b.add_tensor(
                &t.name,
                t.dtype,
                t.shape.clone(),
                st.tensor_bytes(t).to_vec(),
            )?;
            report.written += 1;
            if t.dtype == GgmlType::BF16 {
                report.bf16_passthrough += 1;
            }
        } else {
            report.skipped_non_float += 1;
        }
    }

    Ok((b, report))
}

/// The plain (`--config`-less) path, which **refuses**.
///
/// `vokra.openwakeword.wakeword_names` cannot be honestly sourced
/// without the side-car: the labels are not in the safetensors, and the
/// runtime binder requires them. Synthesising `wakeword_0` would emit a
/// GGUF that loads but reports invented labels to every caller, which is
/// worse than refusing — so this mirrors the [`crate::ModelKind::Crepe`]
/// precedent and routes the caller to
/// [`crate::convert_openwakeword_op_file_with_config`].
///
/// # Errors
///
/// Always [`ConvertError::Usage`].
pub fn convert_openwakeword_op_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<OpenwakewordOpReport, ConvertError> {
    let _ = (input, output, license);
    Err(ConvertError::Usage(
        "openwakeword-op needs a --config config.json carrying `wakeword_names` (plus \
         optional window_frames / mel_bins / sample_rate / hop_samples overrides). The \
         per-wake-word labels are not present in the safetensors — the prepare script \
         indexes classifier tensors positionally and keeps the names in its reference \
         JSON — and the runtime binder requires them, so this converter will not invent \
         them; use convert_openwakeword_op_file_with_config"
            .to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Builds a one-wake-word safetensors whose classifier group matches
    /// the layout the prepare script emits and the binder reads:
    /// `hidden_dim = 2`, `embedding_dim = 3`.
    fn one_wakeword_safetensors() -> Vec<u8> {
        let l1w = f32_bytes(&[0.1, 0.2, -0.1, 0.05, -0.05, 0.1]); // [2, 3]
        let l1b = f32_bytes(&[0.01, -0.02]); // [2]
        let l2w = f32_bytes(&[0.5, -0.3]); // [1, 2]
        let l2b = f32_bytes(&[0.02]); // [1]

        let o1 = l1w.len();
        let o2 = o1 + l1b.len();
        let o3 = o2 + l2w.len();
        let o4 = o3 + l2b.len();
        let header = format!(
            r#"{{"openwakeword.classifier.0.linear1.weight":{{"dtype":"F32","shape":[2,3],"data_offsets":[0,{o1}]}},"openwakeword.classifier.0.linear1.bias":{{"dtype":"F32","shape":[2],"data_offsets":[{o1},{o2}]}},"openwakeword.classifier.0.linear2.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[{o2},{o3}]}},"openwakeword.classifier.0.linear2.bias":{{"dtype":"F32","shape":[1],"data_offsets":[{o3},{o4}]}}}}"#
        );
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&l1w);
        buf.extend_from_slice(&l1b);
        buf.extend_from_slice(&l2w);
        buf.extend_from_slice(&l2b);
        buf
    }

    fn one_wakeword_config() -> OpenwakewordOpConvertConfig {
        OpenwakewordOpConvertConfig::parse(br#"{"wakeword_names":["alexa"]}"#)
            .expect("minimal config parses")
    }

    fn native_dnn_safetensors() -> Vec<u8> {
        let mut tensors: Vec<(String, Vec<u64>)> = vec![
            ("openwakeword.melspec.dft_real".to_owned(), vec![257, 512]),
            ("openwakeword.melspec.dft_imag".to_owned(), vec![257, 512]),
            ("openwakeword.melspec.mel".to_owned(), vec![257, 32]),
        ];
        const CONVS: [(u64, u64, u64, u64); 20] = [
            (1, 24, 3, 3),
            (24, 24, 1, 3),
            (24, 24, 3, 1),
            (24, 48, 1, 3),
            (48, 48, 3, 1),
            (48, 48, 1, 3),
            (48, 48, 3, 1),
            (48, 72, 1, 3),
            (72, 72, 3, 1),
            (72, 72, 1, 3),
            (72, 72, 3, 1),
            (72, 96, 1, 3),
            (96, 96, 3, 1),
            (96, 96, 1, 3),
            (96, 96, 3, 1),
            (96, 96, 1, 3),
            (96, 96, 3, 1),
            (96, 96, 1, 3),
            (96, 96, 3, 1),
            (96, 96, 3, 1),
        ];
        for (index, (input, output, kh, kw)) in CONVS.into_iter().enumerate() {
            tensors.push((
                format!("openwakeword.embedding.conv.{index}.weight"),
                vec![output, input, kh, kw],
            ));
            if index != 19 {
                tensors.push((
                    format!("openwakeword.embedding.conv.{index}.bias"),
                    vec![output],
                ));
            }
        }
        for (layer, (output, input)) in [(128, 1_536), (128, 128), (1, 128)].into_iter().enumerate()
        {
            tensors.push((
                format!("openwakeword.classifier.0.linear.{layer}.weight"),
                vec![output, input],
            ));
            tensors.push((
                format!("openwakeword.classifier.0.linear.{layer}.bias"),
                vec![output],
            ));
        }

        let mut offset = 0usize;
        let mut entries = Vec::with_capacity(tensors.len());
        for (name, shape) in tensors {
            let elements = shape.iter().product::<u64>() as usize;
            let end = offset + elements * 4;
            entries.push(format!(
                "\"{name}\":{{\"dtype\":\"F32\",\"shape\":{shape:?},\"data_offsets\":[{offset},{end}]}}"
            ));
            offset = end;
        }
        let header = format!("{{{}}}", entries.join(","));
        let mut bytes = Vec::with_capacity(8 + header.len() + offset);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + offset, 0);
        bytes
    }

    #[test]
    fn native_dnn_bundle_stamps_additive_streaming_contract() {
        let cfg = OpenwakewordOpConvertConfig::parse(
            br#"{"wakeword_names":["alexa"],"classifier_format":"dnn-relu-sigmoid-v1","classifier_input_frames":16,"classifier_layer_counts":[3],"predict_chunk_samples":1280}"#,
        )
        .expect("native config parses");
        let (builder, report) = convert(native_dnn_safetensors(), &cfg, Some("cc-by-nc-sa-4.0"))
            .expect("native bundle converts");
        assert_eq!(report.read, 48);
        assert_eq!(report.n_wakewords, 1);
        assert_eq!(report.embedding_dim, 96);
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert_eq!(
            file.get(KEY_CLASSIFIER_FORMAT)
                .and_then(|value| value.as_str()),
            Some(CLASSIFIER_FORMAT_DNN)
        );
        assert_eq!(
            file.get(KEY_CLASSIFIER_INPUT_FRAMES)
                .and_then(|value| value.as_u64()),
            Some(16)
        );
        assert_eq!(
            file.get(KEY_PREDICT_CHUNK_SAMPLES)
                .and_then(|value| value.as_u64()),
            Some(1_280)
        );
        let counts = file
            .get(KEY_CLASSIFIER_LAYER_COUNTS)
            .and_then(|value| value.as_array())
            .expect("layer counts array");
        assert_eq!(counts.element_type, GgufValueType::U32);
        assert!(matches!(
            counts.values.as_slice(),
            [GgufMetadataValue::U32(3)]
        ));
    }

    /// The regression this whole wave exists for: every key the runtime
    /// binder requires must be stamped. If any one is dropped the GGUF
    /// cannot load, and until 2026-08-15 all seven were missing.
    #[test]
    fn all_seven_runtime_metadata_keys_are_stamped() {
        let (b, report) = convert(one_wakeword_safetensors(), &one_wakeword_config(), None)
            .expect("convert with config");
        assert_eq!(report.n_wakewords, 1);
        assert_eq!(report.embedding_dim, 3);

        let file = GgufFile::parse(b.to_bytes().expect("to_bytes")).expect("parse GGUF");
        for key in [
            KEY_N_WAKEWORDS,
            KEY_EMBEDDING_DIM,
            KEY_WINDOW_FRAMES,
            KEY_MEL_BINS,
            KEY_SAMPLE_RATE,
            KEY_HOP_SAMPLES,
            KEY_WAKEWORD_NAMES,
        ] {
            assert!(file.get(key).is_some(), "missing required key `{key}`");
        }
        assert_eq!(file.get(KEY_N_WAKEWORDS).and_then(|v| v.as_u64()), Some(1));
        assert_eq!(
            file.get(KEY_EMBEDDING_DIM).and_then(|v| v.as_u64()),
            Some(3)
        );
        assert_eq!(
            file.get(KEY_WINDOW_FRAMES).and_then(|v| v.as_u64()),
            Some(u64::from(DEFAULT_WINDOW_FRAMES))
        );
        assert_eq!(
            file.get(KEY_MEL_BINS).and_then(|v| v.as_u64()),
            Some(u64::from(DEFAULT_MEL_BINS))
        );
        assert_eq!(
            file.get(KEY_SAMPLE_RATE).and_then(|v| v.as_u64()),
            Some(u64::from(DEFAULT_SAMPLE_RATE))
        );
        assert_eq!(
            file.get(KEY_HOP_SAMPLES).and_then(|v| v.as_u64()),
            Some(u64::from(DEFAULT_HOP_SAMPLES))
        );
        let names = file
            .get(KEY_WAKEWORD_NAMES)
            .and_then(|v| v.as_array())
            .expect("wakeword_names is an array");
        assert_eq!(names.element_type, GgufValueType::String);
        assert_eq!(names.values.len(), 1);
        assert_eq!(names.values[0].as_str(), Some("alexa"));
    }

    /// `embedding_dim` is derived from the weights, not mirrored: a
    /// checkpoint with a different width must stamp that width.
    #[test]
    fn embedding_dim_is_derived_from_the_classifier_weight() {
        // hidden_dim = 1, embedding_dim = 5 — nothing like the reference
        // release's 96, so a mirrored constant would be caught here.
        let l1w = f32_bytes(&[0.1, 0.2, 0.3, 0.4, 0.5]);
        let l1b = f32_bytes(&[0.0]);
        let l2w = f32_bytes(&[1.0]);
        let l2b = f32_bytes(&[0.0]);
        let (o1, o2, o3, o4) = (
            l1w.len(),
            l1w.len() + l1b.len(),
            l1w.len() + l1b.len() + l2w.len(),
            l1w.len() + l1b.len() + l2w.len() + l2b.len(),
        );
        let header = format!(
            r#"{{"openwakeword.classifier.0.linear1.weight":{{"dtype":"F32","shape":[1,5],"data_offsets":[0,{o1}]}},"openwakeword.classifier.0.linear1.bias":{{"dtype":"F32","shape":[1],"data_offsets":[{o1},{o2}]}},"openwakeword.classifier.0.linear2.weight":{{"dtype":"F32","shape":[1,1],"data_offsets":[{o2},{o3}]}},"openwakeword.classifier.0.linear2.bias":{{"dtype":"F32","shape":[1],"data_offsets":[{o3},{o4}]}}}}"#
        );
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&l1w);
        buf.extend_from_slice(&l1b);
        buf.extend_from_slice(&l2w);
        buf.extend_from_slice(&l2b);

        let (b, report) = convert(buf, &one_wakeword_config(), None).expect("convert");
        assert_eq!(report.embedding_dim, 5);
        let file = GgufFile::parse(b.to_bytes().expect("to_bytes")).expect("parse");
        assert_eq!(
            file.get(KEY_EMBEDDING_DIM).and_then(|v| v.as_u64()),
            Some(5)
        );
    }

    /// BF16 still rides the verbatim pass-through arm on a real
    /// classifier tensor name.
    #[test]
    fn bf16_classifier_weight_passes_through_verbatim() {
        let l1w_vals: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let l1w = bf16_bytes(&l1w_vals); // [2, 3] BF16 = 12 bytes
        assert_eq!(l1w.len(), 12);
        let l1b = f32_bytes(&[0.01, -0.02]);
        let l2w = f32_bytes(&[0.5, -0.3]);
        let l2b = f32_bytes(&[0.02]);
        let (o1, o2, o3, o4) = (
            l1w.len(),
            l1w.len() + l1b.len(),
            l1w.len() + l1b.len() + l2w.len(),
            l1w.len() + l1b.len() + l2w.len() + l2b.len(),
        );
        let header = format!(
            r#"{{"openwakeword.classifier.0.linear1.weight":{{"dtype":"BF16","shape":[2,3],"data_offsets":[0,{o1}]}},"openwakeword.classifier.0.linear1.bias":{{"dtype":"F32","shape":[2],"data_offsets":[{o1},{o2}]}},"openwakeword.classifier.0.linear2.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[{o2},{o3}]}},"openwakeword.classifier.0.linear2.bias":{{"dtype":"F32","shape":[1],"data_offsets":[{o3},{o4}]}}}}"#
        );
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&l1w);
        buf.extend_from_slice(&l1b);
        buf.extend_from_slice(&l2w);
        buf.extend_from_slice(&l2b);

        let (b, report) = convert(buf, &one_wakeword_config(), None).expect("convert BF16");
        assert_eq!(report.read, 4);
        assert_eq!(report.written, 4);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);

        let file = GgufFile::parse(b.to_bytes().expect("to_bytes")).expect("parse");
        let info = file
            .tensor_info("openwakeword.classifier.0.linear1.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), l1w.as_slice());
    }

    /// Default license resolves to Permissive; the model / provenance
    /// stamps still land alongside the new chunk group.
    #[test]
    fn default_license_is_permissive_and_model_stamps_land() {
        let (b, _) =
            convert(one_wakeword_safetensors(), &one_wakeword_config(), None).expect("convert");
        let file = GgufFile::parse(b.to_bytes().expect("to_bytes")).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "apache-2.0 must resolve to Permissive (T1 tier)"
        );
    }

    /// The license override boundary — a caller redistributing the
    /// upstream CC-BY-NC-SA-4.0 official weights flips the fail-closed
    /// publish gate to NonCommercialShareAlike.
    #[test]
    fn license_override_replaces_default() {
        let (b, _) = convert(
            one_wakeword_safetensors(),
            &one_wakeword_config(),
            Some("cc-by-nc-sa-4.0"),
        )
        .expect("convert with override");
        let file = GgufFile::parse(b.to_bytes().expect("to_bytes")).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("cc-by-nc-sa-4.0"),
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::NonCommercialShareAlike.as_str()),
            "cc-by-nc-sa-4.0 override flips the class from Permissive to NonCommercialShareAlike",
        );
    }

    /// The plain path refuses rather than inventing wake-word labels.
    #[test]
    fn plain_path_refuses_and_names_the_config_route() {
        let err = convert_openwakeword_op_file(
            Path::new("/nonexistent/in.safetensors"),
            Path::new("/nonexistent/out.gguf"),
            None,
        )
        .expect_err("plain path must refuse");
        let msg = err.to_string();
        assert!(
            msg.contains("--config"),
            "message must name --config: {msg}"
        );
        assert!(
            msg.contains("wakeword_names"),
            "message must name the missing axis: {msg}"
        );
        assert!(
            msg.contains("convert_openwakeword_op_file_with_config"),
            "message must route the caller: {msg}"
        );
    }

    /// A name-count / group-count mismatch is refused at convert time,
    /// not deferred to the binder.
    #[test]
    fn name_count_must_match_classifier_group_count() {
        let cfg =
            OpenwakewordOpConvertConfig::parse(br#"{"wakeword_names":["alexa","hey_jarvis"]}"#)
                .expect("config parses");
        let err = convert(one_wakeword_safetensors(), &cfg, None)
            .expect_err("2 names vs 1 group must refuse");
        let msg = err.to_string();
        assert!(msg.contains("2 wake-word name"), "{msg}");
        assert!(msg.contains("1 classifier group"), "{msg}");
    }

    /// A gap in the classifier indices is refused rather than silently
    /// truncating the run to the contiguous prefix.
    #[test]
    fn gapped_classifier_indices_are_refused() {
        // Groups 0 and 2 present, 1 absent. Only group 0's four tensors
        // plus a stray group-2 weight — enough to trip the gap check.
        let l1w = f32_bytes(&[0.1, 0.2, -0.1, 0.05, -0.05, 0.1]);
        let l1b = f32_bytes(&[0.01, -0.02]);
        let l2w = f32_bytes(&[0.5, -0.3]);
        let l2b = f32_bytes(&[0.02]);
        let stray = f32_bytes(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let (o1, o2, o3, o4, o5) = (
            l1w.len(),
            l1w.len() + l1b.len(),
            l1w.len() + l1b.len() + l2w.len(),
            l1w.len() + l1b.len() + l2w.len() + l2b.len(),
            l1w.len() + l1b.len() + l2w.len() + l2b.len() + stray.len(),
        );
        let header = format!(
            r#"{{"openwakeword.classifier.0.linear1.weight":{{"dtype":"F32","shape":[2,3],"data_offsets":[0,{o1}]}},"openwakeword.classifier.0.linear1.bias":{{"dtype":"F32","shape":[2],"data_offsets":[{o1},{o2}]}},"openwakeword.classifier.0.linear2.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[{o2},{o3}]}},"openwakeword.classifier.0.linear2.bias":{{"dtype":"F32","shape":[1],"data_offsets":[{o3},{o4}]}},"openwakeword.classifier.2.linear1.weight":{{"dtype":"F32","shape":[2,3],"data_offsets":[{o4},{o5}]}}}}"#
        );
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&l1w);
        buf.extend_from_slice(&l1b);
        buf.extend_from_slice(&l2w);
        buf.extend_from_slice(&l2b);
        buf.extend_from_slice(&stray);

        let err = convert(buf, &one_wakeword_config(), None).expect_err("gap must refuse");
        let msg = err.to_string();
        assert!(msg.contains("classifier index 2"), "{msg}");
    }

    /// A missing sibling tensor names itself instead of surfacing later
    /// as a confusing binder-side `missing tensor`.
    #[test]
    fn missing_classifier_sibling_is_named_at_convert_time() {
        let l1w = f32_bytes(&[0.1, 0.2, -0.1, 0.05, -0.05, 0.1]);
        let header = format!(
            r#"{{"openwakeword.classifier.0.linear1.weight":{{"dtype":"F32","shape":[2,3],"data_offsets":[0,{}]}}}}"#,
            l1w.len()
        );
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&l1w);

        let err =
            convert(buf, &one_wakeword_config(), None).expect_err("missing sibling must refuse");
        let msg = err.to_string();
        assert!(msg.contains("linear1.bias"), "{msg}");
    }

    /// An input with no classifier group at all is refused with a
    /// message naming the expected layout.
    #[test]
    fn input_without_classifier_groups_is_refused() {
        let payload = f32_bytes(&[1.0, 2.0]);
        let header = format!(
            r#"{{"melspec_model.embedding.weight":{{"dtype":"F32","shape":[2],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&payload);

        let err = convert(buf, &one_wakeword_config(), None).expect_err("no groups must refuse");
        assert!(err.to_string().contains("linear1.weight"), "{err}");
    }

    /// Config parsing: the required field, the optional overrides, and
    /// each fail-closed rejection.
    #[test]
    fn config_parse_contract() {
        let full = OpenwakewordOpConvertConfig::parse(
            br#"{"wakeword_names":["a","b"],"window_frames":40,"mel_bins":16,"sample_rate":8000,"hop_samples":80}"#,
        )
        .expect("full config parses");
        assert_eq!(full.wakeword_names, vec!["a".to_owned(), "b".to_owned()]);
        assert_eq!(full.window_frames, 40);
        assert_eq!(full.mel_bins, 16);
        assert_eq!(full.sample_rate, 8_000);
        assert_eq!(full.hop_samples, 80);

        let minimal =
            OpenwakewordOpConvertConfig::parse(br#"{"wakeword_names":["a"]}"#).expect("minimal");
        assert_eq!(minimal.window_frames, DEFAULT_WINDOW_FRAMES);
        assert_eq!(minimal.mel_bins, DEFAULT_MEL_BINS);
        assert_eq!(minimal.sample_rate, DEFAULT_SAMPLE_RATE);
        assert_eq!(minimal.hop_samples, DEFAULT_HOP_SAMPLES);

        for (bad, needle) in [
            (&br#"{}"#[..], "wakeword_names"),
            (&br#"{"wakeword_names":[]}"#[..], "empty"),
            (&br#"{"wakeword_names":["a",""]}"#[..], "empty"),
            (&br#"{"wakeword_names":["a","a"]}"#[..], "duplicate"),
            (&br#"{"wakeword_names":[1]}"#[..], "not a string"),
            (
                &br#"{"wakeword_names":["a"],"mel_bins":0}"#[..],
                "must be > 0",
            ),
        ] {
            let err = OpenwakewordOpConvertConfig::parse(bad)
                .expect_err("malformed config must be refused");
            assert!(
                err.to_string().contains(needle),
                "expected `{needle}` in: {err}"
            );
        }
    }
}
