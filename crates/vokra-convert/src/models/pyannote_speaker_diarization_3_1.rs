//! **pyannote/speaker-diarization-3.1** (Bredin, CNRS, MIT): pipeline
//! `config.yaml` → GGUF metadata (2026-08-01 Wave 5).
//!
//! # What this converter is (and is not)
//!
//! Unlike every other converter in this tree, this module handles a
//! **pipeline orchestration definition**, not a weight checkpoint. The
//! upstream `pyannote/speaker-diarization-3.1` HF repo ships **only** a
//! ~2 KB `config.yaml`; there are no `.bin` / `.safetensors` /
//! `.ckpt` weight files — the pipeline delegates every forward-pass
//! computation to two sibling weight repos:
//!
//! - **`pyannote/segmentation-3.0`** (PyanNet VAD / speaker-segmentation
//!   backbone, MIT — already covered by
//!   `super::pyannote_segmentation`).
//! - **`pyannote/wespeaker-voxceleb-resnet34-LM`** (WeSpeaker speaker
//!   embedding, MIT — already covered by `super::wespeaker`).
//!
//! Both dependencies are Vokra-published (2026-08-01 wave §3.1
//! sign-off queue). This converter emits a **weightless GGUF** that
//! carries the pipeline's clustering + batch + threshold parameters
//! under the `vokra.pyannote_pipeline.*` chunk group so a future
//! runtime pipeline dispatch (`crates/vokra-models/src/pyannote/
//! pipeline.rs`) can wire the two sibling GGUFs together with the
//! correct clustering knobs.
//!
//! # Zero-tensor GGUF is intentional
//!
//! The `WeightlessPipeline` posture is a deliberate design choice, not
//! a scaffold gap:
//!
//! - **NFR-DS-02 zero-dep**: no YAML parser enters the runtime tree.
//!   The pipeline hparams are transcribed from primary source
//!   (`https://huggingface.co/pyannote/speaker-diarization-3.1/
//!   blob/main/config.yaml`, verified 2026-08-01 — CLAUDE.md
//!   「ハルシネーション厳禁」) as Rust compile-time constants and
//!   emitted verbatim.
//! - **FR-EX-08 loud contract**: if the upstream config diverges in a
//!   future release, the runtime pipeline dispatch fails loudly on the
//!   sub-model reference chunks not resolving (rather than silently
//!   applying stale defaults from a Python side-car).
//! - **Sibling weight repos are separately signed**: `docs/license-
//!   audit.md` §3.1 has independent rows for `pyannote/segmentation-3.0`
//!   (row 268) and `pyannote/wespeaker-voxceleb-resnet34-LM` — the
//!   pipeline GGUF only pins their **model_id references**, it does
//!   not embed their weights.
//!
//! # HF / licence / category
//!
//! - Upstream HF: `pyannote/speaker-diarization-3.1` (stamped under
//!   `vokra.provenance.upstream_hf`).
//! - SPDX: **`mit`** (`LicenseClass::Permissive`). Primary source
//!   verified via authenticated HF API
//!   `api/models/pyannote/speaker-diarization-3.1` = `license: mit,
//!   gated: auto` (2026-07-30 CC direct fetch, `docs/license-audit.md`
//!   §3.1 row 268 = `pyannote (speaker diarization)` yousan sign,
//!   `gated: auto` = access control only, no extra obligations).
//! - Model category: `diarize` (stamped under
//!   `vokra.model.category`). Distinct from the sibling
//!   `pyannote-segmentation` `vad` category — this pipeline **consumes**
//!   a VAD backbone (via `segmentation.model_id`) plus a speaker
//!   encoder (via `embedding.model_id`) plus a clusterer to emit
//!   `(start, end, speaker_id)` intervals.
//!
//! # Primary source config
//!
//! Source: `https://huggingface.co/pyannote/speaker-diarization-3.1/
//! blob/main/config.yaml`, fetched 2026-08-01 — CLAUDE.md「ハル
//! シネーション厳禁」.
//!
//! ```yaml
//! version: 3.1.0
//!
//! pipeline:
//!   name: pyannote.audio.pipelines.SpeakerDiarization
//!   params:
//!     clustering: AgglomerativeClustering
//!     embedding: pyannote/wespeaker-voxceleb-resnet34-LM
//!     embedding_batch_size: 32
//!     embedding_exclude_overlap: true
//!     segmentation: pyannote/segmentation-3.0
//!     segmentation_batch_size: 32
//!
//! params:
//!   clustering:
//!     method: centroid
//!     min_cluster_size: 12
//!     threshold: 0.7045654963945799
//!   segmentation:
//!     min_duration_off: 0.0
//! ```
//!
//! # A note on `onset` / `offset`
//!
//! Older pyannote 2.x pipelines carried `onset` + `offset` activation
//! thresholds on the segmentation head. **pyannote 3.x no longer has
//! them**: PyanNet-3.0 emits a 7-class powerset multiclass posterior
//! (3 speakers × 2 overlap slots) that is argmax-decoded per frame,
//! so there is no scalar activation threshold to tune. The primary
//! source config above has no `onset` / `offset` keys; we deliberately
//! do NOT stamp fake defaults for them (CLAUDE.md「ハルシネーション
//! 厳禁」). If a caller wants the 2.x-style scalar activation gate,
//! that lives in a different pipeline (`pyannote/voice-activity-
//! detection-3.0` etc.) and would be a distinct future converter.
//!
//! # No ONNX (permanent)
//!
//! FR-LD-05: the runtime never grows an ONNX parser. This converter
//! reads a raw byte buffer (typically the upstream `config.yaml` text
//! or an empty side-car), performs a lightweight sanity check that the
//! input plausibly identifies as the pyannote 3.1 speaker-diarization
//! pipeline (not a mis-routed different pipeline), and emits the
//! weightless GGUF from primary-source compile-time constants. No
//! YAML parser, no pickle parser, no ONNX parser — zero-dep preserved.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgufBuilder, chunks};

use crate::ConvertError;

/// `vokra.model.arch` for pyannote speaker-diarization pipelines.
/// Distinct arch tag from every sibling (including
/// `super::pyannote_segmentation` `pyannote-segmentation` — this is
/// a *pipeline orchestrator* over that VAD backbone plus a WeSpeaker
/// embedding backbone plus a clusterer, not the VAD backbone itself.
/// Silently sharing an arch tag would misroute the future runtime
/// pipeline dispatch (a caller who binds this arch would try to run
/// PyanNet's SincNet frontend against a config that has no waveform
/// input path).
pub const ARCH: &str = "pyannote-speaker-diarization";

/// `vokra.model.name` value written for the canonical
/// speaker-diarization-3.1 pipeline GGUF.
pub const NAME: &str = "pyannote-speaker-diarization-3.1";

/// `vokra.model.category` value — first `"diarize"` category in the
/// converter tree, sibling of the M5-residual anchor `DIARIZE_OP =
/// "diarize"` in `crates/vokra-core/src/m5_residual_ops.rs` (FR-OP-82).
/// The pipeline GGUF is what the future `diarize` op consumes at
/// runtime: it reads the clustering thresholds + sub-model references
/// and orchestrates them into an interval-labelling forward pass.
pub const CATEGORY: &str = "diarize";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf`. Preserves upstream casing.
pub const UPSTREAM_HF: &str = "pyannote/speaker-diarization-3.1";

/// Canonical weight license SPDX (`mit`). Overrides via the
/// [`convert_pyannote_speaker_diarization_3_1_file`] `license`
/// parameter — the standing mechanism for "implementation is
/// clean-room MIT but the upstream distributed checkpoint is another
/// license" scenarios (mirror of `convert_file_licensed` in `lib.rs`).
pub const DEFAULT_LICENSE: &str = "mit";

/// One-line free-text description written under
/// [`chunks::KEY_PROVENANCE_SOURCE`].
const SOURCE_DESCRIPTION: &str = "pyannote/speaker-diarization-3.1 (SpeakerDiarization pipeline: \
     segmentation-3.0 VAD + wespeaker-voxceleb-resnet34-LM embedding + \
     AgglomerativeClustering, MIT)";

/// Ad-hoc metadata key for the model category. Same key
/// `super::pyannote_segmentation` / `wespeaker` / `rmvpe` use.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

// -------------------------------------------------------------------------
// vokra.pyannote_pipeline.* chunk group — primary source constants
// transcribed from the upstream `config.yaml` at
// `https://huggingface.co/pyannote/speaker-diarization-3.1/blob/main/
// config.yaml` (fetched 2026-08-01, verified per CLAUDE.md
// 「ハルシネーション厳禁」). Every value below is a direct read of
// the config file — no defaults are invented, no keys are added
// that the upstream config does not carry.
// -------------------------------------------------------------------------

/// `vokra.pyannote_pipeline.type` — short discriminator (`"SpeakerDiarization"`)
/// so a future runtime binder can pick the correct pipeline dispatch
/// arm without parsing the free-text `vokra.pyannote_pipeline.name`
/// Python-class path.
pub(crate) const KEY_PIPELINE_TYPE: &str = "vokra.pyannote_pipeline.type";
/// `vokra.pyannote_pipeline.name` — full upstream Python class path
/// (`pyannote.audio.pipelines.SpeakerDiarization`). Preserved verbatim
/// for traceability against upstream source.
pub(crate) const KEY_PIPELINE_NAME: &str = "vokra.pyannote_pipeline.name";
/// `vokra.pyannote_pipeline.version` — upstream `version:` field from
/// config.yaml (`3.1.0` for this release).
pub(crate) const KEY_PIPELINE_VERSION: &str = "vokra.pyannote_pipeline.version";

/// `vokra.pyannote_pipeline.segmentation.model` — upstream HF slug of the
/// segmentation weight repo the runtime pipeline dispatch loads.
pub(crate) const KEY_SEGMENTATION_MODEL: &str = "vokra.pyannote_pipeline.segmentation.model";
/// `vokra.pyannote_pipeline.segmentation.batch_size` — upstream
/// `segmentation_batch_size` (32) — the per-forward chunk count the
/// pipeline pushes through the segmentation backbone.
pub(crate) const KEY_SEGMENTATION_BATCH_SIZE: &str =
    "vokra.pyannote_pipeline.segmentation.batch_size";
/// `vokra.pyannote_pipeline.segmentation.min_duration_off` — upstream
/// `params.segmentation.min_duration_off` (0.0 s) — the merge-adjacent
/// non-speech-gap floor used when reducing the PyanNet powerset
/// posterior to intervals.
pub(crate) const KEY_SEGMENTATION_MIN_DURATION_OFF: &str =
    "vokra.pyannote_pipeline.segmentation.min_duration_off";

/// `vokra.pyannote_pipeline.embedding.model` — upstream HF slug of the
/// speaker-embedding weight repo the runtime pipeline dispatch loads.
pub(crate) const KEY_EMBEDDING_MODEL: &str = "vokra.pyannote_pipeline.embedding.model";
/// `vokra.pyannote_pipeline.embedding.batch_size` — upstream
/// `embedding_batch_size` (32).
pub(crate) const KEY_EMBEDDING_BATCH_SIZE: &str = "vokra.pyannote_pipeline.embedding.batch_size";
/// `vokra.pyannote_pipeline.embedding.exclude_overlap` — upstream
/// `embedding_exclude_overlap` (true) — whether to drop overlap
/// regions before feeding a segment to the speaker encoder.
pub(crate) const KEY_EMBEDDING_EXCLUDE_OVERLAP: &str =
    "vokra.pyannote_pipeline.embedding.exclude_overlap";

/// `vokra.pyannote_pipeline.clustering.algorithm` — upstream
/// `pipeline.params.clustering` (`AgglomerativeClustering`).
pub(crate) const KEY_CLUSTERING_ALGORITHM: &str = "vokra.pyannote_pipeline.clustering.algorithm";
/// `vokra.pyannote_pipeline.clustering.method` — upstream
/// `params.clustering.method` (`centroid`) — the linkage rule used
/// inside AgglomerativeClustering.
pub(crate) const KEY_CLUSTERING_METHOD: &str = "vokra.pyannote_pipeline.clustering.method";
/// `vokra.pyannote_pipeline.clustering.min_cluster_size` — upstream
/// `params.clustering.min_cluster_size` (12).
pub(crate) const KEY_CLUSTERING_MIN_CLUSTER_SIZE: &str =
    "vokra.pyannote_pipeline.clustering.min_cluster_size";
/// `vokra.pyannote_pipeline.clustering.threshold` — upstream
/// `params.clustering.threshold` (0.7045654963945799) — the cosine-
/// distance cut used to stop the agglomerative merge.
pub(crate) const KEY_CLUSTERING_THRESHOLD: &str = "vokra.pyannote_pipeline.clustering.threshold";

// -------------------------------------------------------------------------
// Primary-source constants (values below transcribed directly from the
// upstream config.yaml, primary source verified 2026-08-01).
// -------------------------------------------------------------------------

/// Upstream `pipeline.name` — short discriminator.
pub const DEFAULT_PIPELINE_TYPE: &str = "SpeakerDiarization";
/// Upstream `pipeline.name` — full Python class path (preserved
/// verbatim, no rewriting).
pub const DEFAULT_PIPELINE_NAME: &str = "pyannote.audio.pipelines.SpeakerDiarization";
/// Upstream `version:` field.
pub const DEFAULT_PIPELINE_VERSION: &str = "3.1.0";

/// Upstream `pipeline.params.segmentation`.
pub const DEFAULT_SEGMENTATION_MODEL: &str = "pyannote/segmentation-3.0";
/// Upstream `pipeline.params.segmentation_batch_size`.
pub const DEFAULT_SEGMENTATION_BATCH_SIZE: u32 = 32;
/// Upstream `params.segmentation.min_duration_off` (seconds).
pub const DEFAULT_SEGMENTATION_MIN_DURATION_OFF: f32 = 0.0;

/// Upstream `pipeline.params.embedding`.
pub const DEFAULT_EMBEDDING_MODEL: &str = "pyannote/wespeaker-voxceleb-resnet34-LM";
/// Upstream `pipeline.params.embedding_batch_size`.
pub const DEFAULT_EMBEDDING_BATCH_SIZE: u32 = 32;
/// Upstream `pipeline.params.embedding_exclude_overlap`.
pub const DEFAULT_EMBEDDING_EXCLUDE_OVERLAP: bool = true;

/// Upstream `pipeline.params.clustering`.
pub const DEFAULT_CLUSTERING_ALGORITHM: &str = "AgglomerativeClustering";
/// Upstream `params.clustering.method`.
pub const DEFAULT_CLUSTERING_METHOD: &str = "centroid";
/// Upstream `params.clustering.min_cluster_size`.
pub const DEFAULT_CLUSTERING_MIN_CLUSTER_SIZE: u32 = 12;
/// Upstream `params.clustering.threshold`. Emitted as `f32` — the
/// upstream value is a 17-digit `f64` literal, but the pipeline
/// dispatch compares against cosine distances that carry ≤ 6
/// significant digits at the sizes used, so an `f32` round-trip is
/// lossless in practice.
pub const DEFAULT_CLUSTERING_THRESHOLD: f32 = 0.704_565_5;

/// Outcome of a pyannote-speaker-diarization-3.1 conversion.
///
/// Weightless — the pipeline GGUF carries no tensors, only metadata
/// chunks referencing the sibling weight repos. Counter fields are
/// kept for parity with sibling converter reports so the shared
/// verify surface in `main.rs` can treat this converter uniformly.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PyannoteSpeakerDiarization31Report {
    /// Whether the input file was present + readable. The pipeline
    /// converter accepts any raw byte buffer (typically the upstream
    /// `config.yaml` text) but does **not** parse it — the sanity
    /// check below only confirms the buffer is plausibly the
    /// speaker-diarization-3.1 pipeline (not a mis-routed different
    /// pipeline config or a completely unrelated file).
    pub input_read: bool,
    /// Whether the input passed the plausibility sanity check.
    /// `false` means the buffer did not contain the tell-tale
    /// `SpeakerDiarization` + `3.1` markers so the caller likely
    /// pointed at the wrong file (a segmentation-3.0 config, an
    /// arbitrary yaml, etc.). The converter still emits the primary-
    /// source-verified pipeline GGUF, but the counter records the
    /// mismatch so a downstream audit can flag it.
    pub input_recognized: bool,
    /// Always 0 — no float weights are converted; the pipeline is
    /// weightless by upstream design (the config.yaml references
    /// sibling weight repos, no tensors of its own).
    pub written: usize,
}

/// Reads an optional `config.yaml` sanity buffer at `input` and writes
/// a pyannote-speaker-diarization-3.1 pipeline GGUF to `output`.
///
/// The `input` file is read for sanity only — the pipeline hparams
/// are transcribed from the primary source `config.yaml` at
/// `https://huggingface.co/pyannote/speaker-diarization-3.1/blob/
/// main/config.yaml` (verified 2026-08-01 — CLAUDE.md「ハルシネーション
/// 厳禁」) and emitted verbatim from Rust compile-time constants. No
/// YAML parser enters the runtime tree (NFR-DS-02 zero-dep).
///
/// The emitted GGUF carries:
/// - `vokra.model.{arch,name,category}` = `pyannote-speaker-
///   diarization` / `pyannote-speaker-diarization-3.1` / `diarize`.
/// - `vokra.provenance.*` = MIT provenance (or the caller's
///   `license` override) plus `upstream_hf` back-reference.
/// - `vokra.pyannote_pipeline.*` = pipeline type / name / version +
///   sub-model references (segmentation + embedding) + clustering
///   knobs (algorithm / method / min_cluster_size / threshold) + one
///   segmentation param (`min_duration_off`).
/// - **Zero tensors** — the pipeline GGUF orchestrates two sibling
///   weight repos ([`DEFAULT_SEGMENTATION_MODEL`] +
///   [`DEFAULT_EMBEDDING_MODEL`]), it does not embed their weights.
///
/// `license` overrides `DEFAULT_LICENSE` (`"mit"`) — the same
/// mechanism `lib.rs::convert_file_licensed` uses when the
/// implementation is clean-room but the redistributed pipeline
/// carries a different SPDX.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Gguf`] if the GGUF writer rejects the
/// metadata-only builder (should never trigger — the metadata layout
/// is fixed).
pub fn convert_pyannote_speaker_diarization_3_1_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<PyannoteSpeakerDiarization31Report, ConvertError> {
    // Read the input file for a *sanity* check only. This is not a
    // parse — the byte buffer is inspected as raw bytes; we deliberately
    // do NOT bring a YAML parser into the runtime tree (NFR-DS-02
    // zero-dep, FR-LD-05 no-parser posture). An input file must be
    // present so the caller cannot accidentally publish a
    // "unconditional-constants" GGUF without pointing at a real
    // upstream config; an empty file is accepted so a scripted publish
    // path can synthesise a zero-byte placeholder if the config.yaml is
    // separately staged.
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let input_read = true;
    let input_recognized = is_speaker_diarization_3_1_config(&bytes);

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // Provenance — MIT end-to-end (primary source verified 2026-07-30
    // via authenticated HF API `api/models/pyannote/speaker-
    // diarization-3.1` = `license: mit, gated: auto`; `gated: auto`
    // is access control only, no extra obligations). The `license`
    // override lets a downstream repackager stamp a different SPDX
    // if they redistribute under stricter terms.
    let effective_license = license.unwrap_or(DEFAULT_LICENSE);
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        effective_license,
        Some(NAME),
        Some(SOURCE_DESCRIPTION),
    );
    // `vokra.provenance.upstream_hf` back-reference. `stamp_provenance`
    // does not write this key today; every sibling converter adds it
    // as a plain `add_string` so the artifact carries the upstream
    // slug for the model-card generator + the CI compliance gate
    // (`scripts/publish/check-catalog-reality.sh`).
    b.add_string("vokra.provenance.upstream_hf", UPSTREAM_HF);

    // vokra.pyannote_pipeline.* chunk group — every value below is a
    // direct read of the upstream config.yaml (primary source URL in
    // the module docstring). No defaults are invented; no keys are
    // added that the upstream config does not carry (see the module
    // docstring note on why `onset` / `offset` are deliberately
    // absent — pyannote 3.x uses powerset multiclass, not scalar
    // activation gates).
    b.add_string(KEY_PIPELINE_TYPE, DEFAULT_PIPELINE_TYPE);
    b.add_string(KEY_PIPELINE_NAME, DEFAULT_PIPELINE_NAME);
    b.add_string(KEY_PIPELINE_VERSION, DEFAULT_PIPELINE_VERSION);
    b.add_string(KEY_SEGMENTATION_MODEL, DEFAULT_SEGMENTATION_MODEL);
    b.add_u32(KEY_SEGMENTATION_BATCH_SIZE, DEFAULT_SEGMENTATION_BATCH_SIZE);
    b.add_f32(
        KEY_SEGMENTATION_MIN_DURATION_OFF,
        DEFAULT_SEGMENTATION_MIN_DURATION_OFF,
    );
    b.add_string(KEY_EMBEDDING_MODEL, DEFAULT_EMBEDDING_MODEL);
    b.add_u32(KEY_EMBEDDING_BATCH_SIZE, DEFAULT_EMBEDDING_BATCH_SIZE);
    b.add_bool(
        KEY_EMBEDDING_EXCLUDE_OVERLAP,
        DEFAULT_EMBEDDING_EXCLUDE_OVERLAP,
    );
    b.add_string(KEY_CLUSTERING_ALGORITHM, DEFAULT_CLUSTERING_ALGORITHM);
    b.add_string(KEY_CLUSTERING_METHOD, DEFAULT_CLUSTERING_METHOD);
    b.add_u32(
        KEY_CLUSTERING_MIN_CLUSTER_SIZE,
        DEFAULT_CLUSTERING_MIN_CLUSTER_SIZE,
    );
    b.add_f32(KEY_CLUSTERING_THRESHOLD, DEFAULT_CLUSTERING_THRESHOLD);

    // Zero tensors — the pipeline GGUF is weightless by upstream
    // design (see the module docstring "Zero-tensor GGUF is
    // intentional" section). The runtime pipeline dispatch loads the
    // sibling weight GGUFs (`vokra/pyannote-segmentation-3.0` +
    // `vokra/wespeaker-voxceleb-resnet34-lm`) via the sub-model
    // reference chunks above; this artifact only carries the
    // orchestration parameters.

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, out_bytes).map_err(ConvertError::Io)?;

    Ok(PyannoteSpeakerDiarization31Report {
        input_read,
        input_recognized,
        written: 0,
    })
}

/// Lightweight sanity check that the input buffer plausibly identifies
/// as a pyannote speaker-diarization-3.1 pipeline `config.yaml`.
/// Not a parser — just a substring scan for the two tell-tale
/// upstream markers (`SpeakerDiarization` in `pipeline.name` and the
/// `3.1` version marker). An input that fails the check still runs
/// through the primary-source-verified emit path; the counter in the
/// report records the mismatch so a downstream audit can flag it.
///
/// A completely empty input is accepted (returns `false` — recognized
/// as "no config supplied" rather than "config supplied but not
/// pyannote-diarization"); the emit path still runs. This mirrors
/// `pyannote_segmentation`'s `zero_tensor_input_returns_empty_report`
/// behaviour: a scripted publish path can pass a placeholder input.
fn is_speaker_diarization_3_1_config(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    // ASCII substring scan — YAML files are UTF-8 text; a byte-level
    // match on `SpeakerDiarization` + `3.1` catches both the raw
    // upstream config.yaml text and callers who staged the config as
    // a UTF-8 buffer through another path.
    contains_ascii(bytes, b"SpeakerDiarization") && contains_ascii(bytes, b"3.1")
}

/// ASCII / UTF-8 needle-in-haystack substring match. Standalone so it
/// does not pull in any external dependency (NFR-DS-02).
fn contains_ascii(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() {
        return needle.is_empty();
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Per-process, per-test scratch path in the system temp dir
    /// (pyannote_segmentation / rmvpe pattern — no external `tempfile`
    /// dep, preserving zero-dep NFR-DS-02). The nanosecond suffix
    /// separates the tests in this module so a parallel `cargo test`
    /// cannot clobber files across them.
    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-pyannote-diar31-{}-{}-{}.bin",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        p
    }

    /// The primary-source upstream `config.yaml` text, transcribed
    /// verbatim from `https://huggingface.co/pyannote/speaker-
    /// diarization-3.1/blob/main/config.yaml` (fetched 2026-08-01 —
    /// CLAUDE.md「ハルシネーション厳禁」). Used as the plausibility
    /// fixture for the sanity-recognized round-trip below.
    const UPSTREAM_CONFIG_YAML: &[u8] = b"version: 3.1.0

pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    embedding: pyannote/wespeaker-voxceleb-resnet34-LM
    embedding_batch_size: 32
    embedding_exclude_overlap: true
    segmentation: pyannote/segmentation-3.0
    segmentation_batch_size: 32

params:
  clustering:
    method: centroid
    min_cluster_size: 12
    threshold: 0.7045654963945799
  segmentation:
    min_duration_off: 0.0
";

    /// STEP 1 (primary-source round-trip): the upstream config.yaml text
    /// survives the pipeline converter round-trip. Every metadata
    /// chunk lands with the primary-source-transcribed value; zero
    /// tensors are emitted (pipeline is weightless by upstream design);
    /// provenance carries the MIT SPDX + `upstream_hf` back-reference.
    #[test]
    fn upstream_config_yaml_round_trips_and_stamps_land() {
        let input = scratch_path("upstream-in");
        let output = scratch_path("upstream-out");
        std::fs::write(&input, UPSTREAM_CONFIG_YAML).expect("write upstream config.yaml");

        let report =
            convert_pyannote_speaker_diarization_3_1_file(&input, &output, None).expect("convert");

        assert!(report.input_read, "input file present + read");
        assert!(
            report.input_recognized,
            "the upstream config must pass the SpeakerDiarization + 3.1 sanity check"
        );
        assert_eq!(
            report.written, 0,
            "pipeline GGUF is weightless — zero tensors emitted by design"
        );

        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");

        // Model triple.
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

        // Provenance.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
        assert_eq!(
            file.get("vokra.provenance.upstream_hf")
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );

        // Pipeline chunk group — every key must be present with the
        // primary-source-transcribed value.
        assert_eq!(
            file.get(KEY_PIPELINE_TYPE).and_then(|v| v.as_str()),
            Some(DEFAULT_PIPELINE_TYPE)
        );
        assert_eq!(
            file.get(KEY_PIPELINE_NAME).and_then(|v| v.as_str()),
            Some(DEFAULT_PIPELINE_NAME)
        );
        assert_eq!(
            file.get(KEY_PIPELINE_VERSION).and_then(|v| v.as_str()),
            Some(DEFAULT_PIPELINE_VERSION)
        );
        assert_eq!(
            file.get(KEY_SEGMENTATION_MODEL).and_then(|v| v.as_str()),
            Some(DEFAULT_SEGMENTATION_MODEL)
        );
        assert_eq!(
            file.get(KEY_SEGMENTATION_BATCH_SIZE)
                .and_then(|v| v.as_u64()),
            Some(DEFAULT_SEGMENTATION_BATCH_SIZE as u64)
        );
        assert!(
            (file
                .get(KEY_SEGMENTATION_MIN_DURATION_OFF)
                .and_then(|v| v.as_f64())
                .unwrap_or(f64::NAN)
                - DEFAULT_SEGMENTATION_MIN_DURATION_OFF as f64)
                .abs()
                < 1e-6,
            "min_duration_off must round-trip losslessly through f32"
        );
        assert_eq!(
            file.get(KEY_EMBEDDING_MODEL).and_then(|v| v.as_str()),
            Some(DEFAULT_EMBEDDING_MODEL)
        );
        assert_eq!(
            file.get(KEY_EMBEDDING_BATCH_SIZE).and_then(|v| v.as_u64()),
            Some(DEFAULT_EMBEDDING_BATCH_SIZE as u64)
        );
        assert_eq!(
            file.get(KEY_EMBEDDING_EXCLUDE_OVERLAP)
                .and_then(|v| v.as_bool()),
            Some(DEFAULT_EMBEDDING_EXCLUDE_OVERLAP)
        );
        assert_eq!(
            file.get(KEY_CLUSTERING_ALGORITHM).and_then(|v| v.as_str()),
            Some(DEFAULT_CLUSTERING_ALGORITHM)
        );
        assert_eq!(
            file.get(KEY_CLUSTERING_METHOD).and_then(|v| v.as_str()),
            Some(DEFAULT_CLUSTERING_METHOD)
        );
        assert_eq!(
            file.get(KEY_CLUSTERING_MIN_CLUSTER_SIZE)
                .and_then(|v| v.as_u64()),
            Some(DEFAULT_CLUSTERING_MIN_CLUSTER_SIZE as u64)
        );
        let stamped = file
            .get(KEY_CLUSTERING_THRESHOLD)
            .and_then(|v| v.as_f64())
            .expect("clustering.threshold must be present as f32/f64");
        assert!(
            (stamped - DEFAULT_CLUSTERING_THRESHOLD as f64).abs() < 1e-5,
            "clustering.threshold must round-trip through f32 close to the \
             upstream 0.7045654963945799 (f32 gives 0.70456547... which is \
             lossless at the sizes cosine distance uses); got {stamped}",
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// STEP 2 (unrecognized-input still emits primary-source GGUF): a
    /// caller who passes an arbitrary text file (e.g. an unrelated
    /// yaml, or the segmentation-3.0 config which shares the pyannote
    /// prefix but is not a speaker-diarization pipeline) triggers the
    /// `input_recognized = false` flag but still produces a valid
    /// pipeline GGUF (the hparams come from primary-source Rust
    /// constants, not the input file). Mirrors `pyannote_segmentation`'s
    /// zero-input tolerance.
    #[test]
    fn unrecognized_input_still_emits_primary_source_pipeline_gguf() {
        let input = scratch_path("unrecognized-in");
        let output = scratch_path("unrecognized-out");
        // A plausible-but-wrong input: the sibling segmentation config
        // (no `SpeakerDiarization` marker, no `3.1` version).
        std::fs::write(
            &input,
            b"# a completely different pipeline\nfoo: bar\nbaz: quux\n",
        )
        .expect("write unrecognized input");

        let report = convert_pyannote_speaker_diarization_3_1_file(&input, &output, None)
            .expect("convert should still succeed on unrecognized input");

        assert!(report.input_read);
        assert!(
            !report.input_recognized,
            "input without SpeakerDiarization + 3.1 must NOT be recognized"
        );
        assert_eq!(report.written, 0);

        // The emitted GGUF still carries every primary-source constant.
        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        assert_eq!(
            file.get(KEY_PIPELINE_TYPE).and_then(|v| v.as_str()),
            Some(DEFAULT_PIPELINE_TYPE),
            "primary-source pipeline type must land even on unrecognized input",
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// STEP 3 (license override path): a caller who obtained the config
    /// under a different SPDX (unlikely for pyannote which is MIT
    /// end-to-end, but the mechanism exists for all converters via
    /// `convert_file_licensed` in `lib.rs`) can stamp that SPDX
    /// through the `license` parameter. Guards against silent
    /// override / SPDX drift.
    #[test]
    fn license_override_stamps_supplied_spdx() {
        let input = scratch_path("license-in");
        let output = scratch_path("license-out");
        std::fs::write(&input, UPSTREAM_CONFIG_YAML).expect("write upstream config.yaml");

        // Hypothetical downstream override — apache-2.0 instead of
        // upstream MIT (both permissive, both round-trip through
        // `stamp_provenance` cleanly).
        convert_pyannote_speaker_diarization_3_1_file(&input, &output, Some("apache-2.0"))
            .expect("convert");

        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
            "license override must reach the provenance chunk"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// STEP 4 (constants pin): the primary-source values must not
    /// silently drift. A future edit that changes a default must also
    /// bump the module-doc primary source URL + the corresponding
    /// row in `docs/license-audit.md` (or the CC-verified date),
    /// not sneak past a stale check.
    #[test]
    #[allow(clippy::assertions_on_constants)] // Compile-time drift guards are intentional.
    fn primary_source_constants_do_not_drift() {
        // Upstream pipeline metadata (from `pipeline:` block of the
        // config.yaml).
        assert_eq!(DEFAULT_PIPELINE_TYPE, "SpeakerDiarization");
        assert_eq!(
            DEFAULT_PIPELINE_NAME,
            "pyannote.audio.pipelines.SpeakerDiarization"
        );
        assert_eq!(DEFAULT_PIPELINE_VERSION, "3.1.0");
        // Sub-model references.
        assert_eq!(DEFAULT_SEGMENTATION_MODEL, "pyannote/segmentation-3.0");
        assert_eq!(
            DEFAULT_EMBEDDING_MODEL,
            "pyannote/wespeaker-voxceleb-resnet34-LM"
        );
        // Batch axes.
        assert_eq!(DEFAULT_SEGMENTATION_BATCH_SIZE, 32);
        assert_eq!(DEFAULT_EMBEDDING_BATCH_SIZE, 32);
        // Boolean toggle.
        const { assert!(DEFAULT_EMBEDDING_EXCLUDE_OVERLAP) };
        // Clustering knobs.
        assert_eq!(DEFAULT_CLUSTERING_ALGORITHM, "AgglomerativeClustering");
        assert_eq!(DEFAULT_CLUSTERING_METHOD, "centroid");
        assert_eq!(DEFAULT_CLUSTERING_MIN_CLUSTER_SIZE, 12);
        // Upstream threshold is 0.7045654963945799 (f64 literal); f32
        // rounds this to 0.70456547... which is well under the 1e-5
        // tolerance the runtime uses for cosine distance cut. If a
        // future upstream release changes this value, the model card
        // must be re-verified.
        assert!(
            (DEFAULT_CLUSTERING_THRESHOLD - 0.704_565_5_f32).abs() < 1e-7,
            "clustering threshold constant drifted"
        );
        // Segmentation min_duration_off.
        assert_eq!(DEFAULT_SEGMENTATION_MIN_DURATION_OFF, 0.0);
    }

    /// STEP 5 (sanity heuristic): the `is_speaker_diarization_3_1_config`
    /// helper must catch mis-routed inputs (an unrelated YAML, a
    /// segmentation-3.0 config) while accepting the true upstream
    /// config.yaml plus reasonable text variants.
    #[test]
    fn sanity_heuristic_distinguishes_pipeline_from_siblings() {
        // Empty buffer is deliberately NOT recognized (represents "no
        // config supplied" rather than "config supplied but wrong").
        assert!(!is_speaker_diarization_3_1_config(b""));

        // True upstream config passes.
        assert!(is_speaker_diarization_3_1_config(UPSTREAM_CONFIG_YAML));

        // Sibling segmentation config (also pyannote 3.x, but a
        // different pipeline) is NOT recognized as a diarization
        // pipeline — the marker set is disjoint.
        let segmentation_only =
            b"version: 3.0.0\npipeline:\n  name: pyannote.audio.pipelines.VoiceActivityDetection\n";
        assert!(!is_speaker_diarization_3_1_config(segmentation_only));

        // Arbitrary yaml with neither marker.
        assert!(!is_speaker_diarization_3_1_config(b"foo: bar\nbaz: quux\n"));

        // Version-only match without SpeakerDiarization is not enough.
        assert!(!is_speaker_diarization_3_1_config(b"version: 3.1.0\n"));

        // SpeakerDiarization-only match without 3.1 is not enough
        // (guards against a hypothetical 3.2 that would need a new
        // converter with a distinct primary-source verification).
        assert!(!is_speaker_diarization_3_1_config(
            b"pipeline:\n  name: pyannote.audio.pipelines.SpeakerDiarization\nversion: 4.0.0\n"
        ));
    }
}
