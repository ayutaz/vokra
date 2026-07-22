//! GA **Definition-of-Done item 2** runner (M5-12-T03 / T07).
//!
//! DoD item 2 (`docs/milestones.md` §9) says the official model zoo's models
//! must **all** pass the NFR-QL-02 5 % degradation gate and numerical-parity
//! CI. This runner turns that into a machine check: it iterates the
//! [`ZooManifest`](crate::zoo::ZooManifest), dispatches each gated model to the
//! right quality axis ([`crate::gate`]), and — crucially — reports what it
//! *could not* measure as honestly as what it could.
//!
//! # What this runner does NOT do (the honest boundary)
//!
//! - **It does not recompute numerical-parity CI.** Item 2's "parity CI 通過"
//!   is a CI fact the owner records as a run-URL in the DoD template; the
//!   runner only carries the `parity_gate` pointer per model.
//! - **It does not fabricate coverage.** A model with no `VOKRA_*_GGUF` is
//!   [`RecordOutcome::SkippedNoWeight`]; a model with no eval corpus is
//!   [`RecordOutcome::SkippedNoCorpus`] — a **distinct** skip, because a
//!   missing corpus is not fixed by adding a weight (spec 位置付け, FR-EX-08).
//!   Neither is a pass.
//! - **It does not silently pass UTMOS.** When no [`AudioMosMetric`] scorer is
//!   injected (the real UTMOS weights are owner-sourced even though the M5-15
//!   native lands), generative-audio models are scored mel-loss-only and the
//!   report says `UTMOS leg: not run` — never "UTMOS passed"
//!   ([`crate::degradation`] `mel_loss_only`).
//! - **It never declares "item 2 satisfied".** The strongest verdict the
//!   runner returns is [`Item2RunnerVerdict::MeasuredGreen`] — "the measurable
//!   half is green"; the final item-2 call (parity CI + owner judgment) is
//!   [T11] owner work.
//!
//! # Testability
//!
//! The scoring core takes a [`ZooRunEnv`] (weight availability, corpus loading,
//! and the optional MOS scorer) so tests drive it deterministically with inline
//! corpora and never touch env vars or the filesystem. [`EnvZooRunEnv`] is the
//! thin real adapter (`std::env` + `std::fs` + `wav` decode + `Utmos`).

use crate::gate::{QualityGateReport, gate_asr_text, gate_generative_audio};
use crate::metrics::AudioMosMetric;
use crate::zoo::{GatedSpec, QualityMetric, ZooManifest, ZooModel};
use vokra_core::{Result, VokraError};

/// The NFR-QL-02 5 % gate threshold — the item-2 bound. Named so a change is a
/// single-site edit, never a scattered literal.
pub const DOD_ITEM2_THRESHOLD: f64 = 0.05;

/// A single eval input for one model, already decoded — the unit the scoring
/// core consumes. Producing these from a corpus file (decoding WAVs, reading
/// transcripts) is [`ZooRunEnv::load_corpus`]'s job, kept separate so the
/// dispatch/gating logic is testable without fixtures.
#[derive(Debug, Clone, PartialEq)]
pub enum CorpusItem {
    /// ASR: ground truth + the upstream-reference transcript + the Vokra
    /// transcript. Scored by [`gate_asr_text`] (WER/CER increase).
    AsrTriple {
        /// The true transcript.
        ground_truth: String,
        /// The upstream (PyTorch/ORT) reference transcript.
        reference_hyp: String,
        /// The Vokra-native transcript under test.
        hyp: String,
    },
    /// Audio: an upstream-reference waveform + the Vokra output waveform at a
    /// shared rate. Scored by [`gate_generative_audio`] (mel-loss ± UTMOS for
    /// TTS/S2S; mel-loss-only roundtrip Δ for codecs).
    AudioPair {
        /// Upstream reference (or, for a codec, the original) waveform.
        reference: Vec<f32>,
        /// Vokra output (or codec roundtrip) waveform.
        hypothesis: Vec<f32>,
        /// Shared sample rate.
        sample_rate: u32,
    },
}

/// The environment a run reads: weights, corpora, and the optional MOS scorer.
///
/// A trait so tests inject a deterministic fake and the real run uses
/// [`EnvZooRunEnv`]. No default impl guesses anything — availability is a fact,
/// not a heuristic.
pub trait ZooRunEnv {
    /// `true` when the model's `VOKRA_*_GGUF` weight is available.
    fn weight_available(&self, gguf_env: &str) -> bool;

    /// Loads the eval corpus for a model.
    ///
    /// - `Ok(None)` — no corpus (empty `eval_manifest`, or the file is absent):
    ///   surfaced as [`RecordOutcome::SkippedNoCorpus`], never a pass.
    /// - `Ok(Some(items))` — the decoded corpus (may be empty → also no-corpus).
    /// - `Err(_)` — a real load error on an existing corpus (malformed record,
    ///   undecodable WAV, sample-rate mismatch): surfaced loudly, FR-EX-08.
    ///
    /// # Errors
    ///
    /// Propagates any corpus-loading error.
    fn load_corpus(&self, model: &ZooModel, spec: &GatedSpec) -> Result<Option<Vec<CorpusItem>>>;

    /// The injected UTMOS scorer, or `None` when no `vokra.utmos.*` GGUF is
    /// available (the honest mel-loss-only path).
    fn mos_scorer(&self) -> Option<&dyn AudioMosMetric>;
}

/// The outcome for one zoo model — every non-pass state is distinct and named.
#[derive(Debug, Clone, PartialEq)]
pub enum RecordOutcome {
    /// The model was scored. Holds one [`QualityGateReport`] per corpus item;
    /// the model passes iff **every** item passed (and there was ≥1 item).
    Scored {
        /// Model name.
        name: String,
        /// Per-corpus-item gate reports.
        reports: Vec<QualityGateReport>,
        /// `true` when this model needs a **hard** UTMOS gate (a
        /// `tts_synthesis`-domain `mel_loss+utmos` model). Codec roundtrips and
        /// codec-streaming (advisory-UTMOS) models are `false` — running them
        /// mel-only is by design, not an unmeasured gate.
        utmos_required: bool,
    },
    /// A `parity_only` model (VAD): not scored at runtime — its verdict is a
    /// dedicated parity harness's green, which the **owner** records. A pointer,
    /// never a runner pass.
    ParityPointer {
        /// Model name.
        name: String,
        /// The parity gate whose green the owner records.
        parity_gate: String,
    },
    /// No `VOKRA_*_GGUF` weight — measurement impossible; not a pass.
    SkippedNoWeight {
        /// Model name.
        name: String,
        /// The env var that was unset.
        gguf_env: String,
    },
    /// No eval corpus — a **distinct** skip from a missing weight; not a pass.
    SkippedNoCorpus {
        /// Model name.
        name: String,
        /// The (empty or absent) eval-manifest pointer.
        eval_manifest: String,
    },
    /// A `★`/`⚠` zoo row that is not a mel/UTMOS quality-gated model (or is
    /// license-held) — not iterated, carried for completeness.
    Excluded {
        /// Model name.
        name: String,
        /// Why it is excluded.
        reason: String,
    },
    /// A scoring error surfaced (not swallowed): the run continues but this
    /// model is clearly not a pass and the message is printed (FR-EX-08).
    Errored {
        /// Model name.
        name: String,
        /// The error message.
        message: String,
    },
}

impl RecordOutcome {
    /// The model's name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Scored { name, .. }
            | Self::ParityPointer { name, .. }
            | Self::SkippedNoWeight { name, .. }
            | Self::SkippedNoCorpus { name, .. }
            | Self::Excluded { name, .. }
            | Self::Errored { name, .. } => name,
        }
    }

    /// `true` only when the model was scored on ≥1 item and every item passed.
    /// Skips, parity pointers, exclusions, and errors are never passes.
    #[must_use]
    pub fn is_pass(&self) -> bool {
        match self {
            Self::Scored { reports, .. } => {
                !reports.is_empty() && reports.iter().all(QualityGateReport::passed)
            }
            _ => false,
        }
    }

    /// `true` when the model was scored and at least one item failed.
    #[must_use]
    pub fn is_measured_failure(&self) -> bool {
        matches!(self, Self::Scored { reports, .. } if reports.iter().any(|r| !r.passed()))
    }
}

/// The runner's verdict on the **measurable half** of item 2. The final item-2
/// decision (parity CI green + owner judgment) is owner work (T11); this verdict
/// deliberately tops out at "measured green".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Item2RunnerVerdict {
    /// At least one measured model failed the 5 % gate or errored — item 2
    /// cannot pass.
    MeasuredFailures,
    /// Coverage is incomplete: some models were skipped (no weight / no corpus),
    /// UTMOS did not run for a gating generative-audio model, or a parity
    /// pointer awaits the owner. No conclusion is possible.
    Incomplete,
    /// Every model that *could* be measured was measured and passed. This is
    /// the strongest the runner asserts — item 2 still needs the parity-CI
    /// evidence and the owner's call.
    MeasuredGreen,
}

impl Item2RunnerVerdict {
    /// Stable label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MeasuredFailures => "MEASURED-FAILURES",
            Self::Incomplete => "INCOMPLETE",
            Self::MeasuredGreen => "MEASURED-GREEN (owner finalizes item 2)",
        }
    }
}

/// The item-2 run report: every model's outcome plus the honest tallies.
#[derive(Debug, Clone)]
pub struct DodItem2Report {
    /// One outcome per catalogued model (gated + excluded), in file order.
    pub outcomes: Vec<RecordOutcome>,
    /// The 5 % threshold used.
    pub threshold: f64,
    /// `true` when a UTMOS scorer was injected (the UTMOS leg ran); `false`
    /// means generative-audio models were scored mel-loss-only.
    pub utmos_leg_available: bool,
}

impl DodItem2Report {
    /// Number of models that were scored and passed every corpus item.
    #[must_use]
    pub fn passed(&self) -> usize {
        self.outcomes.iter().filter(|o| o.is_pass()).count()
    }

    /// Number of scored models with at least one failing item.
    #[must_use]
    pub fn failed(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|o| o.is_measured_failure())
            .count()
    }

    /// Count of a given non-scored outcome kind.
    fn count<F: Fn(&RecordOutcome) -> bool>(&self, f: F) -> usize {
        self.outcomes.iter().filter(|o| f(o)).count()
    }

    /// Models skipped for a missing weight.
    #[must_use]
    pub fn skipped_no_weight(&self) -> usize {
        self.count(|o| matches!(o, RecordOutcome::SkippedNoWeight { .. }))
    }

    /// Models skipped for a missing corpus (distinct from missing weight).
    #[must_use]
    pub fn skipped_no_corpus(&self) -> usize {
        self.count(|o| matches!(o, RecordOutcome::SkippedNoCorpus { .. }))
    }

    /// Parity-pointer models (owner records the parity CI green).
    #[must_use]
    pub fn parity_pointers(&self) -> usize {
        self.count(|o| matches!(o, RecordOutcome::ParityPointer { .. }))
    }

    /// Excluded (non-quality-gated / license-held) models.
    #[must_use]
    pub fn excluded(&self) -> usize {
        self.count(|o| matches!(o, RecordOutcome::Excluded { .. }))
    }

    /// Models whose scoring errored.
    #[must_use]
    pub fn errored(&self) -> usize {
        self.count(|o| matches!(o, RecordOutcome::Errored { .. }))
    }

    /// `true` when at least one scored model that needs a **hard** UTMOS gate
    /// (a `tts_synthesis` `mel_loss+utmos` model) ran mel-loss-only — i.e. the
    /// UTMOS half of item 2 is unmeasured for a model that requires it. Codec
    /// roundtrips and codec-streaming (advisory-UTMOS) models are excluded:
    /// running them mel-only is by design, not an unmeasured gate.
    #[must_use]
    pub fn has_unmeasured_utmos(&self) -> bool {
        self.outcomes.iter().any(|o| match o {
            RecordOutcome::Scored {
                reports,
                utmos_required,
                ..
            } => *utmos_required && reports.iter().any(QualityGateReport::audio_is_mel_only),
            _ => false,
        })
    }

    /// The runner verdict on the measurable half (never the final item-2 call).
    #[must_use]
    pub fn verdict(&self) -> Item2RunnerVerdict {
        if self.failed() > 0 || self.errored() > 0 {
            return Item2RunnerVerdict::MeasuredFailures;
        }
        let incomplete = self.skipped_no_weight() > 0
            || self.skipped_no_corpus() > 0
            || self.parity_pointers() > 0
            || self.has_unmeasured_utmos();
        if incomplete {
            Item2RunnerVerdict::Incomplete
        } else {
            Item2RunnerVerdict::MeasuredGreen
        }
    }

    /// A human-readable, honest report block for the DoD template / CLI.
    #[must_use]
    pub fn render(&self) -> String {
        let mut s = String::from("GA DoD item 2 — zoo mel/UTMOS/WER/roundtrip degradation run\n");
        s.push_str(&format!("threshold: {:.4}\n", self.threshold));
        s.push_str(&format!(
            "UTMOS leg: {}\n",
            if self.utmos_leg_available {
                "active (scorer injected)"
            } else {
                "not run — no vokra.utmos.* scorer injected (M5-15 native landed; real weights \
                 owner-sourced). Generative-audio models scored mel-loss-only."
            }
        ));
        for o in &self.outcomes {
            match o {
                RecordOutcome::Scored { name, reports, .. } => {
                    let verdict = if o.is_pass() { "PASS" } else { "FAIL" };
                    s.push_str(&format!("[{verdict}] {name} ({} item(s))\n", reports.len()));
                    for r in reports {
                        for line in r.summary().lines() {
                            s.push_str(&format!("    {line}\n"));
                        }
                    }
                }
                RecordOutcome::ParityPointer { name, parity_gate } => s.push_str(&format!(
                    "[PARITY-POINTER] {name} — owner records green of: {parity_gate}\n"
                )),
                RecordOutcome::SkippedNoWeight { name, gguf_env } => s.push_str(&format!(
                    "[SKIP:no-weight] {name} — set {gguf_env} to measure (NOT a pass)\n"
                )),
                RecordOutcome::SkippedNoCorpus {
                    name,
                    eval_manifest,
                } => s.push_str(&format!(
                    "[SKIP:no-corpus] {name} — eval corpus '{}' absent (distinct from no-weight; \
                     NOT a pass)\n",
                    if eval_manifest.is_empty() {
                        "(unset)"
                    } else {
                        eval_manifest
                    }
                )),
                RecordOutcome::Excluded { name, reason } => {
                    s.push_str(&format!("[EXCLUDED] {name} — {reason}\n"));
                }
                RecordOutcome::Errored { name, message } => {
                    s.push_str(&format!("[ERROR] {name} — {message}\n"));
                }
            }
        }
        s.push_str(&format!(
            "coverage: passed={} failed={} skipped_no_weight={} skipped_no_corpus={} \
             parity_pointer={} excluded={} errored={}\n",
            self.passed(),
            self.failed(),
            self.skipped_no_weight(),
            self.skipped_no_corpus(),
            self.parity_pointers(),
            self.excluded(),
            self.errored(),
        ));
        s.push_str(&format!("runner verdict: {}\n", self.verdict().as_str()));
        if self.verdict() != Item2RunnerVerdict::MeasuredGreen {
            s.push_str(
                "note: this is NOT an item-2 pass. Skips/parity-pointers/UTMOS-gaps mean the zoo \
                 was not fully measured; the final item-2 decision (parity CI green + owner \
                 judgment) is owner work (M5-12-T11).\n",
            );
        } else {
            s.push_str(
                "note: the measurable half is green. Item 2 still requires the numerical-parity \
                 CI evidence (run URLs) and the owner's final judgment (M5-12-T11).\n",
            );
        }
        s
    }
}

/// Runs the DoD item-2 gate over every model in `manifest` using `env`.
///
/// Never returns `Err`: per-model scoring errors are captured as
/// [`RecordOutcome::Errored`] so one bad corpus does not hide the rest of the
/// zoo — but they are counted as non-passes and printed (FR-EX-08). A caller
/// that wants the verdict asks [`DodItem2Report::verdict`].
#[must_use]
pub fn run_dod_item2(
    manifest: &ZooManifest,
    env: &dyn ZooRunEnv,
    threshold: f64,
) -> DodItem2Report {
    let utmos_leg_available = env.mos_scorer().is_some();
    let mut outcomes = Vec::with_capacity(manifest.models.len());
    for model in &manifest.models {
        let spec = match model.gated() {
            Some(g) => g,
            None => {
                // Excluded — carried for completeness, never scored.
                let reason = match &model.kind {
                    crate::zoo::ZooKind::Excluded { reason } => reason.clone(),
                    crate::zoo::ZooKind::Gated(_) => unreachable!("gated() was None"),
                };
                outcomes.push(RecordOutcome::Excluded {
                    name: model.name.clone(),
                    reason,
                });
                continue;
            }
        };

        // parity_only (VAD): a pointer to a separate parity harness — never a
        // runtime score here (spec 位置付け).
        if spec.quality_metric == QualityMetric::ParityOnly {
            outcomes.push(RecordOutcome::ParityPointer {
                name: model.name.clone(),
                parity_gate: spec.parity_gate.clone(),
            });
            continue;
        }

        // Weight gate: no GGUF -> skipped_no_weight (never a pass).
        if !env.weight_available(&spec.gguf_env) {
            outcomes.push(RecordOutcome::SkippedNoWeight {
                name: model.name.clone(),
                gguf_env: spec.gguf_env.clone(),
            });
            continue;
        }

        // Corpus gate: no corpus -> skipped_no_corpus (distinct skip).
        let items = match env.load_corpus(model, spec) {
            Ok(Some(items)) if !items.is_empty() => items,
            Ok(_) => {
                outcomes.push(RecordOutcome::SkippedNoCorpus {
                    name: model.name.clone(),
                    eval_manifest: spec.eval_manifest.clone(),
                });
                continue;
            }
            Err(e) => {
                outcomes.push(RecordOutcome::Errored {
                    name: model.name.clone(),
                    message: format!("corpus load: {e}"),
                });
                continue;
            }
        };

        // Score each item on the dispatched axis.
        let mut reports = Vec::with_capacity(items.len());
        let mut errored: Option<String> = None;
        for item in &items {
            match score_item(model, spec, item, threshold, env.mos_scorer()) {
                Ok(r) => reports.push(r),
                Err(e) => {
                    errored = Some(format!("{e}"));
                    break;
                }
            }
        }
        match errored {
            Some(message) => outcomes.push(RecordOutcome::Errored {
                name: model.name.clone(),
                message,
            }),
            None => outcomes.push(RecordOutcome::Scored {
                name: model.name.clone(),
                reports,
                // A hard UTMOS gate is required only for in-distribution TTS
                // synthesis; codec roundtrips and codec-streaming (advisory)
                // models legitimately run mel-only.
                utmos_required: spec.quality_metric == QualityMetric::MelLossUtmos
                    && spec.mos_domain == crate::zoo::ZooMosDomain::TtsSynthesis,
            }),
        }
    }
    DodItem2Report {
        outcomes,
        threshold,
        utmos_leg_available,
    }
}

/// Scores one corpus item on the axis its model's `quality_metric` selects.
///
/// A corpus item whose shape disagrees with the metric (e.g. an `AudioPair`
/// fed to a WER model) is a corpus-authoring bug and errors loudly rather than
/// being coerced (FR-EX-08).
fn score_item(
    model: &ZooModel,
    spec: &GatedSpec,
    item: &CorpusItem,
    threshold: f64,
    mos: Option<&dyn AudioMosMetric>,
) -> Result<QualityGateReport> {
    match spec.quality_metric {
        QualityMetric::Wer => match item {
            CorpusItem::AsrTriple {
                ground_truth,
                reference_hyp,
                hyp,
            } => gate_asr_text(&model.name, ground_truth, reference_hyp, hyp, threshold),
            CorpusItem::AudioPair { .. } => Err(mismatch(model, "wer", "an AudioPair")),
        },
        QualityMetric::MelLossUtmos => match item {
            CorpusItem::AudioPair {
                reference,
                hypothesis,
                sample_rate,
            } => {
                // UTMOS gates only when a scorer is injected AND the domain is
                // in-distribution; codec_streaming stays advisory. No scorer =
                // honest mel-loss-only (mel_loss_only = true).
                let mos_arg = match (mos, spec.mos_domain.to_degradation()) {
                    (Some(scorer), Some(domain)) => Some((scorer, domain)),
                    _ => None,
                };
                gate_generative_audio(
                    &model.name,
                    reference,
                    hypothesis,
                    *sample_rate,
                    threshold,
                    mos_arg,
                    false,
                )
            }
            CorpusItem::AsrTriple { .. } => Err(mismatch(model, "mel_loss+utmos", "an AsrTriple")),
        },
        QualityMetric::Roundtrip => match item {
            CorpusItem::AudioPair {
                reference,
                hypothesis,
                sample_rate,
            } => {
                // Roundtrip is a waveform pair scored mel-loss-only (no UTMOS —
                // a codec roundtrip is not a MOS-graded synthesis).
                gate_generative_audio(
                    &model.name,
                    reference,
                    hypothesis,
                    *sample_rate,
                    threshold,
                    None,
                    false,
                )
            }
            CorpusItem::AsrTriple { .. } => Err(mismatch(model, "roundtrip", "an AsrTriple")),
        },
        QualityMetric::ParityOnly => Err(VokraError::InvalidArgument(format!(
            "dod: {} is parity_only and must not reach score_item (handled as a pointer)",
            model.name
        ))),
    }
}

fn mismatch(model: &ZooModel, metric: &str, got: &str) -> VokraError {
    VokraError::InvalidArgument(format!(
        "dod: model '{}' has quality_metric '{metric}' but its corpus item is {got} — the corpus \
         shape does not match the metric (a corpus-authoring bug, not coerced; FR-EX-08)",
        model.name
    ))
}

/// The real environment: weights via `std::env`, corpora via `std::fs` + the
/// `wav` decoder, and an optional [`Utmos`](crate::metrics::Utmos) scorer.
///
/// The corpus loader resolves `eval_manifest` relative to `corpus_root`; an
/// empty pointer or an absent file is `Ok(None)` (no-corpus, not an error). The
/// WAV-decode path is exercised by owner-supplied corpora (none ship in-repo
/// yet — 未確定 (7)); this crate's tests drive the scoring core through a fake
/// env instead.
pub struct EnvZooRunEnv {
    corpus_root: std::path::PathBuf,
    utmos: Option<crate::metrics::Utmos>,
}

impl EnvZooRunEnv {
    /// Builds a real environment.
    ///
    /// `corpus_root` is the base for resolving `eval_manifest` paths.
    /// `utmos_gguf` — when `Some`, a `vokra.utmos.*` GGUF is loaded once and the
    /// UTMOS leg activates; when `None`, the run is mel-loss-only.
    ///
    /// # Errors
    ///
    /// Propagates a [`Utmos::from_path`](crate::metrics::Utmos::from_path)
    /// error (a broken/incompatible UTMOS GGUF is loud, never silently skipped).
    pub fn new(
        corpus_root: impl Into<std::path::PathBuf>,
        utmos_gguf: Option<&std::path::Path>,
    ) -> Result<Self> {
        let utmos = match utmos_gguf {
            Some(p) => Some(crate::metrics::Utmos::from_path(p)?),
            None => None,
        };
        Ok(Self {
            corpus_root: corpus_root.into(),
            utmos,
        })
    }
}

impl ZooRunEnv for EnvZooRunEnv {
    fn weight_available(&self, gguf_env: &str) -> bool {
        std::env::var_os(gguf_env).is_some()
    }

    fn load_corpus(&self, _model: &ZooModel, spec: &GatedSpec) -> Result<Option<Vec<CorpusItem>>> {
        if spec.eval_manifest.is_empty() {
            return Ok(None);
        }
        let path = self.corpus_root.join(&spec.eval_manifest);
        if !path.exists() {
            // A specified-but-absent corpus is still "no corpus", not an error:
            // the owner has not supplied it yet (未確定 (7)).
            return Ok(None);
        }
        let man = crate::manifest::Manifest::load(&path).map_err(VokraError::Io)?;
        let mut items = Vec::with_capacity(man.records.len());
        for rec in &man.records {
            let item = match spec.quality_metric {
                QualityMetric::Wer => CorpusItem::AsrTriple {
                    ground_truth: corpus_field(rec, "ground_truth", spec)?,
                    reference_hyp: corpus_field(rec, "reference_hyp", spec)?,
                    hyp: corpus_field(rec, "hyp", spec)?,
                },
                QualityMetric::MelLossUtmos | QualityMetric::Roundtrip => {
                    let ref_path = self.corpus_root.join(corpus_field(rec, "ref_wav", spec)?);
                    let hyp_path = self.corpus_root.join(corpus_field(rec, "hyp_wav", spec)?);
                    let ref_wav =
                        crate::wav::read_wav(&ref_path).map_err(VokraError::InvalidArgument)?;
                    let hyp_wav =
                        crate::wav::read_wav(&hyp_path).map_err(VokraError::InvalidArgument)?;
                    if ref_wav.sample_rate != hyp_wav.sample_rate {
                        return Err(VokraError::InvalidArgument(format!(
                            "dod corpus: {} vs {} sample-rate mismatch ({} != {}) — no silent \
                             resample (FR-EX-08)",
                            ref_path.display(),
                            hyp_path.display(),
                            ref_wav.sample_rate,
                            hyp_wav.sample_rate
                        )));
                    }
                    CorpusItem::AudioPair {
                        reference: ref_wav.samples,
                        hypothesis: hyp_wav.samples,
                        sample_rate: ref_wav.sample_rate,
                    }
                }
                QualityMetric::ParityOnly => {
                    // parity_only never loads a corpus (handled as a pointer).
                    return Ok(None);
                }
            };
            items.push(item);
        }
        Ok(Some(items))
    }

    fn mos_scorer(&self) -> Option<&dyn AudioMosMetric> {
        self.utmos.as_ref().map(|u| u as &dyn AudioMosMetric)
    }
}

fn corpus_field(rec: &crate::manifest::Record, key: &str, spec: &GatedSpec) -> Result<String> {
    match rec.get(key) {
        Some(v) if !v.is_empty() => Ok(v.to_owned()),
        _ => Err(VokraError::InvalidArgument(format!(
            "dod corpus (line {}): a '{}' record is missing field '{key}'",
            rec.line,
            spec.quality_metric.as_str()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{Direction, Metric};
    use std::cell::RefCell;
    use std::collections::HashSet;

    // A fully deterministic fake env: which weights exist, what corpus each
    // model returns, and whether a MOS scorer is present.
    struct FakeEnv {
        weights: HashSet<String>,
        corpora: std::collections::HashMap<String, Result<Option<Vec<CorpusItem>>>>,
        mos: Option<Box<dyn AudioMosMetric>>,
    }

    impl FakeEnv {
        fn new() -> Self {
            Self {
                weights: HashSet::new(),
                corpora: std::collections::HashMap::new(),
                mos: None,
            }
        }
        fn with_weight(mut self, env: &str) -> Self {
            self.weights.insert(env.to_owned());
            self
        }
        fn with_corpus(mut self, model: &str, c: Result<Option<Vec<CorpusItem>>>) -> Self {
            self.corpora.insert(model.to_owned(), c);
            self
        }
        fn with_mos(mut self, m: Box<dyn AudioMosMetric>) -> Self {
            self.mos = Some(m);
            self
        }
    }

    impl ZooRunEnv for FakeEnv {
        fn weight_available(&self, gguf_env: &str) -> bool {
            self.weights.contains(gguf_env)
        }
        fn load_corpus(
            &self,
            model: &ZooModel,
            _spec: &GatedSpec,
        ) -> Result<Option<Vec<CorpusItem>>> {
            match self.corpora.get(&model.name) {
                Some(Ok(v)) => Ok(v.clone()),
                Some(Err(e)) => Err(VokraError::InvalidArgument(format!("{e}"))),
                None => Ok(None),
            }
        }
        fn mos_scorer(&self) -> Option<&dyn AudioMosMetric> {
            self.mos.as_deref()
        }
    }

    struct ConstMos(f64);
    impl Metric for ConstMos {
        fn name(&self) -> &str {
            "const-mos"
        }
        fn direction(&self) -> Direction {
            Direction::HigherIsBetter
        }
    }
    impl AudioMosMetric for ConstMos {
        fn eval_mos(&self, _a: &[f32], _sr: u32) -> Result<f64> {
            Ok(self.0)
        }
    }

    struct ScriptedMos(RefCell<Vec<f64>>);
    impl Metric for ScriptedMos {
        fn name(&self) -> &str {
            "scripted"
        }
        fn direction(&self) -> Direction {
            Direction::HigherIsBetter
        }
    }
    impl AudioMosMetric for ScriptedMos {
        fn eval_mos(&self, _a: &[f32], _sr: u32) -> Result<f64> {
            self.0
                .borrow_mut()
                .pop()
                .ok_or_else(|| VokraError::InvalidArgument("exhausted".into()))
        }
    }

    fn tone(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin())
            .collect()
    }

    // ---- the builtin zoo, no weights: everything honest, nothing green ------

    #[test]
    fn builtin_zoo_with_no_weights_is_incomplete_never_green() {
        let m = ZooManifest::builtin().unwrap();
        let env = FakeEnv::new(); // no weights, no corpora, no MOS
        let rep = run_dod_item2(&m, &env, DOD_ITEM2_THRESHOLD);

        // Every gated non-VAD model is skipped_no_weight; VAD is a pointer;
        // excluded are carried. Nothing is a pass — this is the current repo
        // reality and it must read as INCOMPLETE, not green.
        assert_eq!(rep.passed(), 0, "no model can pass without weights");
        assert_eq!(rep.failed(), 0);
        assert!(
            rep.skipped_no_weight() >= 13,
            "most gated models skip on weight"
        );
        assert_eq!(
            rep.parity_pointers(),
            1,
            "Silero VAD is the one parity pointer"
        );
        assert_eq!(rep.excluded(), 12);
        assert_eq!(rep.verdict(), Item2RunnerVerdict::Incomplete);
        assert!(!rep.utmos_leg_available);
        let out = rep.render();
        assert!(out.contains("UTMOS leg: not run"), "{out}");
        assert!(out.contains("NOT an item-2 pass"), "{out}");
    }

    // ---- the two skip categories are distinct ------------------------------

    #[test]
    fn no_weight_and_no_corpus_are_counted_separately() {
        // whisper-base: weight present but no corpus -> skipped_no_corpus.
        // whisper-small: no weight -> skipped_no_weight. Distinct counters.
        let m = ZooManifest::builtin().unwrap();
        let env = FakeEnv::new()
            .with_weight("VOKRA_WHISPER_BASE_GGUF")
            .with_corpus("whisper-base", Ok(None));
        let rep = run_dod_item2(&m, &env, DOD_ITEM2_THRESHOLD);
        let base = rep
            .outcomes
            .iter()
            .find(|o| o.name() == "whisper-base")
            .unwrap();
        assert!(matches!(base, RecordOutcome::SkippedNoCorpus { .. }));
        let small = rep
            .outcomes
            .iter()
            .find(|o| o.name() == "whisper-small")
            .unwrap();
        assert!(matches!(small, RecordOutcome::SkippedNoWeight { .. }));
        assert!(rep.skipped_no_corpus() >= 1 && rep.skipped_no_weight() >= 1);
    }

    // ---- ASR scoring path (WER) --------------------------------------------

    #[test]
    fn asr_scores_and_passes_on_matching_transcripts() {
        let m = ZooManifest::builtin().unwrap();
        let env = FakeEnv::new()
            .with_weight("VOKRA_WHISPER_BASE_GGUF")
            .with_corpus(
                "whisper-base",
                Ok(Some(vec![CorpusItem::AsrTriple {
                    ground_truth: "the quick brown fox".into(),
                    reference_hyp: "the quick brown fox".into(),
                    hyp: "the quick brown fox".into(),
                }])),
            );
        let rep = run_dod_item2(&m, &env, DOD_ITEM2_THRESHOLD);
        let base = rep
            .outcomes
            .iter()
            .find(|o| o.name() == "whisper-base")
            .unwrap();
        assert!(base.is_pass(), "identical transcripts pass: {base:?}");
    }

    #[test]
    fn asr_fails_on_a_real_wer_regression() {
        let m = ZooManifest::builtin().unwrap();
        let env = FakeEnv::new()
            .with_weight("VOKRA_WHISPER_BASE_GGUF")
            .with_corpus(
                "whisper-base",
                Ok(Some(vec![CorpusItem::AsrTriple {
                    ground_truth: "the quick brown fox".into(),
                    reference_hyp: "the quick brown fox".into(),
                    hyp: "the quick brown box".into(), // one word wrong
                }])),
            );
        let rep = run_dod_item2(&m, &env, DOD_ITEM2_THRESHOLD);
        assert_eq!(rep.failed(), 1);
        assert_eq!(rep.verdict(), Item2RunnerVerdict::MeasuredFailures);
    }

    // ---- corpus-shape mismatch errors loudly (FR-EX-08) --------------------

    #[test]
    fn wrong_corpus_shape_for_metric_errors() {
        let m = ZooManifest::builtin().unwrap();
        let env = FakeEnv::new()
            .with_weight("VOKRA_WHISPER_BASE_GGUF")
            .with_corpus(
                "whisper-base",
                Ok(Some(vec![CorpusItem::AudioPair {
                    reference: tone(16_000),
                    hypothesis: tone(16_000),
                    sample_rate: 16_000,
                }])),
            );
        let rep = run_dod_item2(&m, &env, DOD_ITEM2_THRESHOLD);
        let base = rep
            .outcomes
            .iter()
            .find(|o| o.name() == "whisper-base")
            .unwrap();
        assert!(matches!(base, RecordOutcome::Errored { .. }), "{base:?}");
        assert_eq!(rep.errored(), 1);
        assert_eq!(rep.verdict(), Item2RunnerVerdict::MeasuredFailures);
    }

    #[test]
    fn corpus_load_error_is_surfaced_not_swallowed() {
        let m = ZooManifest::builtin().unwrap();
        let env = FakeEnv::new()
            .with_weight("VOKRA_WHISPER_BASE_GGUF")
            .with_corpus(
                "whisper-base",
                Err(VokraError::InvalidArgument("boom".into())),
            );
        let rep = run_dod_item2(&m, &env, DOD_ITEM2_THRESHOLD);
        let base = rep
            .outcomes
            .iter()
            .find(|o| o.name() == "whisper-base")
            .unwrap();
        match base {
            RecordOutcome::Errored { message, .. } => assert!(message.contains("boom")),
            other => panic!("expected Errored, got {other:?}"),
        }
    }

    // ---- TTS mel-only vs UTMOS-injected (T03 vs T07) -----------------------

    #[test]
    fn tts_without_scorer_is_mel_only_and_leaves_utmos_unmeasured() {
        // kokoro (tts_synthesis): with a weight + corpus but NO scorer, the
        // model is scored mel-loss-only. The report must flag the UTMOS half as
        // unmeasured — INCOMPLETE, never a green that hides the missing UTMOS.
        let m = ZooManifest::builtin().unwrap();
        let x = tone(16_000);
        let env = FakeEnv::new().with_weight("VOKRA_KOKORO_GGUF").with_corpus(
            "kokoro-82m",
            Ok(Some(vec![CorpusItem::AudioPair {
                reference: x.clone(),
                hypothesis: x.clone(),
                sample_rate: 16_000,
            }])),
        );
        let rep = run_dod_item2(&m, &env, DOD_ITEM2_THRESHOLD);
        let k = rep
            .outcomes
            .iter()
            .find(|o| o.name() == "kokoro-82m")
            .unwrap();
        assert!(k.is_pass(), "mel is identical -> mel half passes");
        assert!(rep.has_unmeasured_utmos(), "UTMOS half is unmeasured");
        // Even though kokoro's mel passed, the run is INCOMPLETE (others skip +
        // UTMOS unmeasured) — never MeasuredGreen with an unmeasured UTMOS.
        assert_eq!(rep.verdict(), Item2RunnerVerdict::Incomplete);
    }

    #[test]
    fn tts_with_scorer_runs_the_utmos_half() {
        let m = ZooManifest::builtin().unwrap();
        let x = tone(16_000);
        // Constant MOS => identical ref/quant => zero decrease => passes.
        let env = FakeEnv::new()
            .with_weight("VOKRA_KOKORO_GGUF")
            .with_corpus(
                "kokoro-82m",
                Ok(Some(vec![CorpusItem::AudioPair {
                    reference: x.clone(),
                    hypothesis: x.clone(),
                    sample_rate: 16_000,
                }])),
            )
            .with_mos(Box::new(ConstMos(4.0)));
        let rep = run_dod_item2(&m, &env, DOD_ITEM2_THRESHOLD);
        assert!(rep.utmos_leg_available);
        let k = rep
            .outcomes
            .iter()
            .find(|o| o.name() == "kokoro-82m")
            .unwrap();
        assert!(k.is_pass());
        assert!(
            !rep.has_unmeasured_utmos(),
            "scorer present -> UTMOS measured"
        );
        match k {
            RecordOutcome::Scored { reports, .. } => {
                let r = &reports[0];
                assert!(!r.audio_is_mel_only(), "UTMOS ran, not mel-only");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn codec_streaming_utmos_is_advisory_and_does_not_gate() {
        // moshi (codec_streaming): a big UTMOS drop must NOT fail the gate —
        // it is advisory-only for the codec domain. The mel half is identical.
        let m = ZooManifest::builtin().unwrap();
        let x = tone(16_000);
        let env = FakeEnv::new()
            .with_weight("VOKRA_MOSHI_GGUF")
            .with_corpus(
                "moshi",
                Ok(Some(vec![CorpusItem::AudioPair {
                    reference: x.clone(),
                    hypothesis: x.clone(),
                    sample_rate: 16_000,
                }])),
            )
            // ref 4.0, quant 3.0 = 25% drop, but advisory for codec domain.
            .with_mos(Box::new(ScriptedMos(RefCell::new(vec![3.0, 4.0]))));
        let rep = run_dod_item2(&m, &env, DOD_ITEM2_THRESHOLD);
        let moshi = rep.outcomes.iter().find(|o| o.name() == "moshi").unwrap();
        assert!(
            moshi.is_pass(),
            "advisory UTMOS drop must not fail: {moshi:?}"
        );
    }

    // ---- codec roundtrip is mel-only, never UTMOS --------------------------

    #[test]
    fn codec_roundtrip_scores_mel_only_even_with_a_scorer() {
        let m = ZooManifest::builtin().unwrap();
        let x = tone(16_000);
        let env = FakeEnv::new()
            .with_weight("VOKRA_DAC_GGUF")
            .with_corpus(
                "dac",
                Ok(Some(vec![CorpusItem::AudioPair {
                    reference: x.clone(),
                    hypothesis: x.clone(),
                    sample_rate: 16_000,
                }])),
            )
            .with_mos(Box::new(ConstMos(4.0)));
        let rep = run_dod_item2(&m, &env, DOD_ITEM2_THRESHOLD);
        let dac = rep.outcomes.iter().find(|o| o.name() == "dac").unwrap();
        match dac {
            RecordOutcome::Scored { reports, .. } => {
                // Roundtrip never runs UTMOS: the audio half is mel-only.
                assert!(
                    reports[0].audio_is_mel_only(),
                    "codec roundtrip is mel-only"
                );
            }
            other => panic!("{other:?}"),
        }
    }

    // ---- EnvZooRunEnv honest no-corpus / weight-absent branches ------------
    // (the WAV-decode path needs owner corpora — 未確定 (7) — so it is not
    // fixture-tested here; these cover the skip branches that do not.)

    #[test]
    fn env_run_env_empty_eval_manifest_is_no_corpus() {
        let env = EnvZooRunEnv::new(".", None).unwrap();
        assert!(env.mos_scorer().is_none(), "no GGUF -> no UTMOS leg");
        let m = ZooManifest::builtin().unwrap();
        let model = m.gated().find(|x| x.name == "kokoro-82m").unwrap();
        let spec = model.gated().unwrap();
        // The builtin eval_manifest is empty -> Ok(None) (no corpus), never an
        // error and never a pass.
        assert!(env.load_corpus(model, spec).unwrap().is_none());
    }

    #[test]
    fn env_run_env_absent_corpus_file_is_no_corpus_not_error() {
        let env = EnvZooRunEnv::new("/nonexistent-root-xyz-m5-12", None).unwrap();
        let text = "\
name = x
family = x
audit_name = X
license = MIT
task = tts
gguf_env = E
eval_manifest = does/not/exist.txt
quality_metric = mel_loss+utmos
mos_domain = tts_synthesis
parity_gate = g
since_version = v1
";
        let m = ZooManifest::parse(text).unwrap();
        let model = &m.models[0];
        let spec = model.gated().unwrap();
        // A specified-but-absent corpus is no-corpus (owner has not supplied it),
        // not a load error.
        assert!(env.load_corpus(model, spec).unwrap().is_none());
    }

    #[test]
    fn env_run_env_weight_absent_is_false() {
        let env = EnvZooRunEnv::new(".", None).unwrap();
        assert!(!env.weight_available("VOKRA_DEFINITELY_UNSET_ENV_VAR_M5_12"));
    }
}
