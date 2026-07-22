//! Machine-readable **official model-zoo catalog** (M5-12-T02; GA DoD item 2).
//!
//! The GA Definition-of-Done item 2 (`docs/milestones.md` §9) asserts that the
//! official model zoo's Apache-2.0 / MIT models all pass the NFR-QL-02 5 %
//! degradation gate and numerical-parity CI. Asserting "**all** models" needs
//! the set of models to be *enumerable by a machine*, not scattered across a
//! prose table. This module parses [`crate::manifest`]'s `key = value` record
//! format (reusing [`Manifest::parse`](crate::manifest::Manifest::parse) — no
//! second parser) into a typed catalog the DoD item-2 runner ([`crate::dod`])
//! iterates.
//!
//! # Two record kinds, because "not a quality-gated model" ≠ "dropped"
//!
//! A [`ZooModel`] is either [`ZooKind::Gated`] (a model the item-2 runner
//! scores) or [`ZooKind::Excluded`] (a zoo row that is **not** a mel/UTMOS
//! quality-gated model — a KWS spotter, speaker-embedding, enhancement,
//! watermarker, or vocoder component — or one held pending license sign-off).
//! An excluded record is *not* iterated, but it stays in the catalog so
//! `scripts/check-zoo-manifest-complete.sh` can prove every `★ 公式 zoo` row in
//! `docs/license-audit.md` is *accounted for*. A model that silently
//! disappears would make item 2 vacuously true for it — the exact failure this
//! catalog exists to prevent.
//!
//! # Errors (FR-EX-08 — unknown enums are loud, never guessed)
//!
//! An unknown `task`, `quality_metric`, or `mos_domain`, a missing required
//! field, a task/metric inconsistency (e.g. an `asr` model tagged
//! `mel_loss+utmos`), or an empty `excluded_reason` is a hard
//! [`VokraError::InvalidArgument`] carrying the offending record's line — a
//! manifest that half-parses is a manifest that lies about coverage.

use crate::manifest::Manifest;
use vokra_core::{Result, VokraError};

/// The embedded official-zoo manifest (`data/zoo/manifest.txt`), compiled in so
/// the catalog is available with zero runtime file dependency (NFR-DS-02).
pub const BUILTIN_MANIFEST: &str = include_str!("../data/zoo/manifest.txt");

/// The task a gated zoo model performs — decides which quality metric gates it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZooTask {
    /// Automatic speech recognition (Whisper, Voxtral) — gated on WER.
    Asr,
    /// Text-to-speech (piper-plus, Kokoro, CosyVoice2) — gated on mel/UTMOS.
    Tts,
    /// Speech-to-speech (CSM, Moshi) — gated on mel/UTMOS.
    S2s,
    /// Voice activity detection (Silero) — gated on segment-span parity.
    Vad,
    /// Neural audio codec (DAC, Mimi, WavTokenizer) — gated on roundtrip parity.
    Codec,
}

impl ZooTask {
    fn parse(s: &str, line: usize) -> Result<Self> {
        Ok(match s {
            "asr" => Self::Asr,
            "tts" => Self::Tts,
            "s2s" => Self::S2s,
            "vad" => Self::Vad,
            "codec" => Self::Codec,
            other => {
                return Err(VokraError::InvalidArgument(format!(
                    "zoo manifest (line {line}): unknown task '{other}' \
                     (expected one of: asr tts s2s vad codec)"
                )));
            }
        })
    }

    /// Stable lowercase identifier (round-trips [`Self::parse`]).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Asr => "asr",
            Self::Tts => "tts",
            Self::S2s => "s2s",
            Self::Vad => "vad",
            Self::Codec => "codec",
        }
    }
}

/// Which quality metric decides a gated model's item-2 verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityMetric {
    /// Word-error-rate increase vs the upstream reference transcript (ASR).
    Wer,
    /// Log-mel L1 loss + UTMOS decrease vs the upstream reference (TTS / S2S).
    /// The UTMOS half is only computable once a scorer is injected; without
    /// one the runner reports `mel_loss_only` — never a silent UTMOS pass.
    MelLossUtmos,
    /// Codec decode(encode(x)) vs x, measured as a mel-loss delta.
    Roundtrip,
    /// Not scored at runtime here — gated by a dedicated parity harness whose
    /// green the owner records (VAD segment-span parity).
    ParityOnly,
}

impl QualityMetric {
    fn parse(s: &str, line: usize) -> Result<Self> {
        Ok(match s {
            "wer" => Self::Wer,
            "mel_loss+utmos" => Self::MelLossUtmos,
            "roundtrip" => Self::Roundtrip,
            "parity_only" => Self::ParityOnly,
            other => {
                return Err(VokraError::InvalidArgument(format!(
                    "zoo manifest (line {line}): unknown quality_metric '{other}' \
                     (expected one of: wer  mel_loss+utmos  roundtrip  parity_only)"
                )));
            }
        })
    }

    /// Stable identifier (round-trips [`Self::parse`]).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Wer => "wer",
            Self::MelLossUtmos => "mel_loss+utmos",
            Self::Roundtrip => "roundtrip",
            Self::ParityOnly => "parity_only",
        }
    }
}

/// A model's acoustic domain relative to the UTMOS training distribution —
/// selects [`crate::degradation::MosDomain`], or marks the model as one where
/// UTMOS does not apply at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZooMosDomain {
    /// In-distribution TTS synthesis — UTMOS gates.
    TtsSynthesis,
    /// Mimi-codec / streaming output — UTMOS advisory-only (out-of-distribution
    /// until the owner calibration study; `degradation.rs`).
    CodecStreaming,
    /// UTMOS does not apply (ASR / VAD / codec roundtrip).
    NotApplicable,
}

impl ZooMosDomain {
    fn parse(s: &str, line: usize) -> Result<Self> {
        Ok(match s {
            "tts_synthesis" => Self::TtsSynthesis,
            "codec_streaming" => Self::CodecStreaming,
            "n_a" => Self::NotApplicable,
            other => {
                return Err(VokraError::InvalidArgument(format!(
                    "zoo manifest (line {line}): unknown mos_domain '{other}' \
                     (expected one of: tts_synthesis  codec_streaming  n_a)"
                )));
            }
        })
    }

    /// Maps to the [`crate::degradation::MosDomain`] the UTMOS gate takes, or
    /// `None` when UTMOS does not apply to this model.
    #[must_use]
    pub const fn to_degradation(self) -> Option<crate::degradation::MosDomain> {
        match self {
            Self::TtsSynthesis => Some(crate::degradation::MosDomain::TtsSynthesis),
            Self::CodecStreaming => Some(crate::degradation::MosDomain::CodecStreaming),
            Self::NotApplicable => None,
        }
    }

    /// Stable identifier (round-trips [`Self::parse`]).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TtsSynthesis => "tts_synthesis",
            Self::CodecStreaming => "codec_streaming",
            Self::NotApplicable => "n_a",
        }
    }
}

/// The gating parameters of a [`ZooKind::Gated`] model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatedSpec {
    /// What the model does (selects the metric).
    pub task: ZooTask,
    /// Env var whose presence gates the eval; absent = `skipped_no_weight`.
    pub gguf_env: String,
    /// Path to a hyp/ref [`Manifest`] of eval inputs; empty = `skipped_no_corpus`
    /// (a *distinct* skip from a missing weight).
    pub eval_manifest: String,
    /// The metric that decides the verdict.
    pub quality_metric: QualityMetric,
    /// UTMOS domain selection (or not-applicable).
    pub mos_domain: ZooMosDomain,
    /// The numerical-parity gate whose green the owner records as evidence.
    pub parity_gate: String,
}

/// A zoo model is either scored by the item-2 runner or explicitly excluded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZooKind {
    /// A model the item-2 runner iterates and gates.
    Gated(GatedSpec),
    /// A `★`/`⚠` zoo row that is not a mel/UTMOS quality-gated model, or is
    /// held pending license sign-off. Carries the reason so "excluded" is
    /// never mistaken for "forgotten".
    Excluded {
        /// Why this row is not iterated (non-empty by construction).
        reason: String,
    },
}

/// One catalog entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZooModel {
    /// Stable kebab id (e.g. `whisper-large-v3`).
    pub name: String,
    /// Model family (e.g. `whisper`).
    pub family: String,
    /// EXACT model-name column of the matching `docs/license-audit.md` row
    /// (`**` stripped) — the join key for the completeness gate. Whisper's five
    /// size records all carry the one bundle string.
    pub audit_name: String,
    /// Weight license (distribution-relevant), cross-referenced against
    /// `docs/license-audit.md`.
    pub license: String,
    /// First Vokra version the model shipped in.
    pub since_version: String,
    /// Free-text notes.
    pub notes: String,
    /// Gated or excluded.
    pub kind: ZooKind,
}

impl ZooModel {
    /// `true` when this is a [`ZooKind::Gated`] model.
    #[must_use]
    pub fn is_gated(&self) -> bool {
        matches!(self.kind, ZooKind::Gated(_))
    }

    /// The gated spec, if this model is gated.
    #[must_use]
    pub fn gated(&self) -> Option<&GatedSpec> {
        match &self.kind {
            ZooKind::Gated(g) => Some(g),
            ZooKind::Excluded { .. } => None,
        }
    }
}

/// The parsed official-zoo catalog.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ZooManifest {
    /// Models in file order.
    pub models: Vec<ZooModel>,
}

impl ZooManifest {
    /// Parses the [`BUILTIN_MANIFEST`].
    ///
    /// # Errors
    ///
    /// Propagates any schema error in the embedded data file.
    pub fn builtin() -> Result<Self> {
        Self::parse(BUILTIN_MANIFEST)
    }

    /// Parses a zoo manifest from text (reusing [`Manifest::parse`] for the
    /// `key = value` record mechanism).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a missing required field, unknown
    /// enum value, task/metric inconsistency, or empty `excluded_reason`
    /// (FR-EX-08 — a half-valid catalog is rejected, never partially trusted).
    pub fn parse(text: &str) -> Result<Self> {
        let raw = Manifest::parse(text);
        let mut models = Vec::with_capacity(raw.records.len());
        for rec in &raw.records {
            let line = rec.line;
            let req = |key: &str| -> Result<String> {
                match rec.get(key) {
                    Some(v) if !v.is_empty() => Ok(v.to_owned()),
                    _ => Err(VokraError::InvalidArgument(format!(
                        "zoo manifest (line {line}): missing required field '{key}'"
                    ))),
                }
            };
            let name = req("name")?;
            let family = req("family")?;
            let audit_name = req("audit_name")?;
            let license = req("license")?;
            let since_version = req("since_version")?;
            let notes = rec.get("notes").unwrap_or("").to_owned();

            // An excluded record is authoritative on `excluded_reason`; a
            // present-but-empty reason is rejected so "excluded" always says why.
            let kind = match rec.get("excluded_reason") {
                Some(reason) if !reason.trim().is_empty() => ZooKind::Excluded {
                    reason: reason.trim().to_owned(),
                },
                Some(_) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "zoo manifest (line {line}): 'excluded_reason' is present but empty — an \
                         excluded model must state why (FR-EX-08)"
                    )));
                }
                None => {
                    let task = ZooTask::parse(&req("task")?, line)?;
                    let quality_metric = QualityMetric::parse(&req("quality_metric")?, line)?;
                    let mos_domain = ZooMosDomain::parse(&req("mos_domain")?, line)?;
                    validate_task_metric(task, quality_metric, mos_domain, line)?;
                    ZooKind::Gated(GatedSpec {
                        task,
                        gguf_env: req("gguf_env")?,
                        // eval_manifest is optional: absent/empty = no corpus.
                        eval_manifest: rec.get("eval_manifest").unwrap_or("").trim().to_owned(),
                        quality_metric,
                        mos_domain,
                        parity_gate: req("parity_gate")?,
                    })
                }
            };

            models.push(ZooModel {
                name,
                family,
                audit_name,
                license,
                since_version,
                notes,
                kind,
            });
        }
        Ok(Self { models })
    }

    /// Iterator over the gated models (item-2 runner input).
    pub fn gated(&self) -> impl Iterator<Item = &ZooModel> {
        self.models.iter().filter(|m| m.is_gated())
    }

    /// Iterator over the excluded models.
    pub fn excluded(&self) -> impl Iterator<Item = &ZooModel> {
        self.models
            .iter()
            .filter(|m| matches!(m.kind, ZooKind::Excluded { .. }))
    }

    /// Count of records claiming a given `audit_name` (the completeness gate's
    /// bundle-expansion check — Whisper's bundle string must be claimed 5×).
    #[must_use]
    pub fn count_for_audit_name(&self, audit_name: &str) -> usize {
        self.models
            .iter()
            .filter(|m| m.audit_name == audit_name)
            .count()
    }
}

/// Enforces the task↔metric↔domain consistency a runner relies on to dispatch.
///
/// A mismatch (an `asr` record tagged `mel_loss+utmos`, a TTS record tagged
/// `n_a` domain, …) is a manifest-authoring bug that would make the runner
/// score the wrong axis — caught here, loudly, rather than producing a
/// confidently-wrong number (FR-EX-08).
fn validate_task_metric(
    task: ZooTask,
    metric: QualityMetric,
    domain: ZooMosDomain,
    line: usize,
) -> Result<()> {
    let expected = match task {
        ZooTask::Asr => QualityMetric::Wer,
        ZooTask::Tts | ZooTask::S2s => QualityMetric::MelLossUtmos,
        ZooTask::Vad => QualityMetric::ParityOnly,
        ZooTask::Codec => QualityMetric::Roundtrip,
    };
    if metric != expected {
        return Err(VokraError::InvalidArgument(format!(
            "zoo manifest (line {line}): task '{}' requires quality_metric '{}', got '{}'",
            task.as_str(),
            expected.as_str(),
            metric.as_str()
        )));
    }
    // mel/UTMOS models must state a UTMOS domain; everything else must be n_a.
    let domain_ok = match metric {
        QualityMetric::MelLossUtmos => matches!(
            domain,
            ZooMosDomain::TtsSynthesis | ZooMosDomain::CodecStreaming
        ),
        _ => domain == ZooMosDomain::NotApplicable,
    };
    if !domain_ok {
        return Err(VokraError::InvalidArgument(format!(
            "zoo manifest (line {line}): quality_metric '{}' is inconsistent with mos_domain \
             '{}' (mel_loss+utmos needs tts_synthesis|codec_streaming; all others need n_a)",
            metric.as_str(),
            domain.as_str()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- builtin data file -------------------------------------------------

    #[test]
    fn builtin_manifest_parses_and_is_internally_consistent() {
        let m = ZooManifest::builtin().expect("builtin zoo manifest must parse");
        // Sanity: the catalog is non-trivial and every record round-trips its
        // enums, and validate_task_metric passed for all gated records.
        assert!(
            m.models.len() >= 20,
            "catalog too small: {}",
            m.models.len()
        );
        for model in &m.models {
            assert!(!model.name.is_empty());
            assert!(!model.audit_name.is_empty());
            match &model.kind {
                ZooKind::Gated(g) => {
                    assert!(!g.gguf_env.is_empty(), "{} has empty gguf_env", model.name);
                    assert!(!g.parity_gate.is_empty());
                }
                ZooKind::Excluded { reason } => assert!(!reason.is_empty()),
            }
        }
    }

    #[test]
    fn builtin_whisper_bundle_expands_to_exactly_five_sizes() {
        // The one true bundle in license-audit.md: 5 Whisper sizes share one
        // audit row. The completeness gate asserts this count; assert it here
        // too so the crate carries the invariant.
        let m = ZooManifest::builtin().unwrap();
        assert_eq!(
            m.count_for_audit_name("Whisper base/small/medium/large-v3/turbo"),
            5,
            "Whisper bundle must expand to 5 size records"
        );
        let names: Vec<_> = m
            .models
            .iter()
            .filter(|x| x.family == "whisper")
            .map(|x| x.name.as_str())
            .collect();
        for want in [
            "whisper-base",
            "whisper-small",
            "whisper-medium",
            "whisper-large-v3",
            "whisper-turbo",
        ] {
            assert!(names.contains(&want), "missing {want}: {names:?}");
        }
    }

    #[test]
    fn builtin_has_the_expected_excluded_and_gated_split() {
        let m = ZooManifest::builtin().unwrap();
        let gated = m.gated().count();
        let excluded = m.excluded().count();
        // 15 gated (1 vad + 5 whisper + piper + kokoro + cosyvoice2 + csm +
        // moshi + voxtral + 3 codecs) / 12 excluded (xcodec2 + titanet +
        // pyannote license/embedding holds + 4 non-quality + 3 enhancement +
        // audioseal + vocos).
        assert_eq!(gated, 15, "gated count drifted");
        assert_eq!(excluded, 12, "excluded count drifted");
    }

    #[test]
    fn xcodec2_is_present_as_an_excluded_license_hold_not_dropped() {
        let m = ZooManifest::builtin().unwrap();
        let x = m
            .models
            .iter()
            .find(|r| r.name == "xcodec2")
            .expect("X-Codec 2 must be catalogued, not silently dropped");
        match &x.kind {
            ZooKind::Excluded { reason } => {
                assert!(
                    reason.contains("license-sign-off-pending"),
                    "reason: {reason}"
                );
            }
            other => panic!("X-Codec 2 must be Excluded (license hold), got {other:?}"),
        }
    }

    #[test]
    fn builtin_carries_no_cc_by_nc_weight() {
        // deliverables.md §3.5 "含めないもの" — no NC weight in the official zoo.
        let m = ZooManifest::builtin().unwrap();
        for model in &m.models {
            let up = model.license.to_uppercase();
            assert!(
                !up.contains("-NC") && !up.contains("NON-COMMERCIAL"),
                "{} has an NC license '{}' — must not be in the official zoo",
                model.name,
                model.license
            );
        }
    }

    // ---- parser happy paths ------------------------------------------------

    #[test]
    fn parses_a_gated_and_an_excluded_record() {
        let text = "\
name = whisper-base
family = whisper
audit_name = Whisper base/small/medium/large-v3/turbo
license = MIT
task = asr
gguf_env = VOKRA_WHISPER_BASE_GGUF
eval_manifest =
quality_metric = wer
mos_domain = n_a
parity_gate = parity_whisper.rs
since_version = v0.1
notes = hi

name = vocos
family = vocos
audit_name = Vocos
license = MIT
excluded_reason = vocoder-head-component-not-a-standalone-zoo-model
since_version = v1.0
notes =
";
        let m = ZooManifest::parse(text).unwrap();
        assert_eq!(m.models.len(), 2);
        let w = &m.models[0];
        assert_eq!(w.name, "whisper-base");
        let g = w.gated().expect("gated");
        assert_eq!(g.task, ZooTask::Asr);
        assert_eq!(g.quality_metric, QualityMetric::Wer);
        assert_eq!(g.mos_domain, ZooMosDomain::NotApplicable);
        assert_eq!(g.eval_manifest, "", "empty eval_manifest -> no corpus");
        assert_eq!(w.notes, "hi");
        assert!(matches!(m.models[1].kind, ZooKind::Excluded { .. }));
        assert_eq!(m.gated().count(), 1);
        assert_eq!(m.excluded().count(), 1);
    }

    #[test]
    fn codec_streaming_domain_maps_to_advisory() {
        let text = "\
name = moshi
family = moshi
audit_name = Moshi (Helium + Mimi)
license = CC-BY-4.0-attribution
task = s2s
gguf_env = VOKRA_MOSHI_GGUF
eval_manifest = corpus/moshi.txt
quality_metric = mel_loss+utmos
mos_domain = codec_streaming
parity_gate = moshi parity
since_version = v1.0-rc
";
        let m = ZooManifest::parse(text).unwrap();
        let g = m.models[0].gated().unwrap();
        assert_eq!(g.mos_domain, ZooMosDomain::CodecStreaming);
        assert_eq!(g.eval_manifest, "corpus/moshi.txt");
        let d = g.mos_domain.to_degradation().unwrap();
        assert!(
            d.is_advisory_only(),
            "codec_streaming must be advisory-only"
        );
    }

    // ---- parser error paths (FR-EX-08) -------------------------------------

    #[test]
    fn unknown_task_is_a_hard_error() {
        let text = "\
name = x
family = x
audit_name = X
license = MIT
task = translation
gguf_env = E
quality_metric = wer
mos_domain = n_a
parity_gate = g
since_version = v1
";
        let err = ZooManifest::parse(text).expect_err("unknown task must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
        assert!(format!("{err}").contains("unknown task"), "{err}");
    }

    #[test]
    fn unknown_quality_metric_is_a_hard_error() {
        let text = "\
name = x
family = x
audit_name = X
license = MIT
task = tts
gguf_env = E
quality_metric = pesq
mos_domain = tts_synthesis
parity_gate = g
since_version = v1
";
        let err = ZooManifest::parse(text).expect_err("unknown metric must fail");
        assert!(format!("{err}").contains("unknown quality_metric"), "{err}");
    }

    #[test]
    fn unknown_mos_domain_is_a_hard_error() {
        let text = "\
name = x
family = x
audit_name = X
license = MIT
task = tts
gguf_env = E
quality_metric = mel_loss+utmos
mos_domain = studio
parity_gate = g
since_version = v1
";
        let err = ZooManifest::parse(text).expect_err("unknown domain must fail");
        assert!(format!("{err}").contains("unknown mos_domain"), "{err}");
    }

    #[test]
    fn task_metric_mismatch_is_rejected() {
        // asr must be wer, not mel_loss+utmos — a runner would otherwise try to
        // score a text model on a waveform axis.
        let text = "\
name = x
family = x
audit_name = X
license = MIT
task = asr
gguf_env = E
quality_metric = mel_loss+utmos
mos_domain = tts_synthesis
parity_gate = g
since_version = v1
";
        let err = ZooManifest::parse(text).expect_err("mismatch must fail");
        assert!(
            format!("{err}").contains("requires quality_metric"),
            "{err}"
        );
    }

    #[test]
    fn tts_with_na_domain_is_rejected() {
        let text = "\
name = x
family = x
audit_name = X
license = MIT
task = tts
gguf_env = E
quality_metric = mel_loss+utmos
mos_domain = n_a
parity_gate = g
since_version = v1
";
        let err = ZooManifest::parse(text).expect_err("na domain on tts must fail");
        assert!(
            format!("{err}").contains("inconsistent with mos_domain"),
            "{err}"
        );
    }

    #[test]
    fn missing_required_field_is_a_hard_error() {
        let text = "\
name = x
task = asr
gguf_env = E
quality_metric = wer
mos_domain = n_a
parity_gate = g
since_version = v1
";
        let err = ZooManifest::parse(text).expect_err("missing family/audit_name/license");
        assert!(format!("{err}").contains("missing required field"), "{err}");
    }

    #[test]
    fn empty_excluded_reason_is_rejected() {
        let text = "\
name = x
family = x
audit_name = X
license = MIT
excluded_reason =
since_version = v1
";
        let err = ZooManifest::parse(text).expect_err("empty excluded_reason must fail");
        assert!(format!("{err}").contains("present but empty"), "{err}");
    }
}
