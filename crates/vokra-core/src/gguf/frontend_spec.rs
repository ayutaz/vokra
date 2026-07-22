//! Typed view of the `vokra.frontend.*` metadata chunk (`frontend_spec`), plus
//! its bit-exact inspection against the runtime front-end (FR-LD-03, M1-03).
//!
//! [`FrontendSpec`] has one field per key defined in [`super::chunks`]; it can
//! be written into a [`GgufBuilder`](super::GgufBuilder) and read back out of a
//! [`GgufFile`](super::GgufFile).
//!
//! # Inspection (M1-03)
//!
//! [`FrontendSpec::diff`] reports every field in which a model's declared
//! front-end differs from the runtime's, and [`FrontendSpec::check_against`]
//! turns that report into either a hard error or a stderr warning per a
//! [`FrontendPolicy`]. The comparison is **bit-exact** — the three `f32` fields
//! (`fmin` / `fmax` / `pre_emphasis`) are compared bit-for-bit (`f32::to_bits`),
//! not with a tolerance, because they are carried verbatim from the upstream
//! model: any difference is a genuine front-end mismatch (reviewer C note #2),
//! not floating-point noise.
//!
//! The check is **per-model conditional**: a model that owns an STFT / mel
//! front-end (Whisper) writes the chunk and is checked at load; models whose
//! front-end Vokra does not control (Silero VAD, piper-plus) write no
//! `vokra.frontend.*` keys and their loaders never invoke the check. Gating is
//! therefore by *which loader calls* [`FrontendSpec::check_against`], not by a
//! global pass — see `vokra_models::whisper::mel::check_frontend_spec`.

use super::chunks;
use super::value::{GgufMetadataValue, GgufValueType};
use super::{GgufBuilder, GgufError, GgufFile};
use crate::error::VokraError;

/// The front-end feature-extraction parameters stored under `vokra.frontend.*`.
///
/// The 13 fields are transcribed from CLAUDE.md / FR-LD-03; the GGUF value type
/// used for each is documented in [`super::chunks`]. Values are carried
/// verbatim from the upstream model's pre-processing configuration — the
/// converter must never invent them (frontend bit-exactness, reviewer C
/// note #2).
#[derive(Debug, Clone, PartialEq)]
pub struct FrontendSpec {
    /// FFT window size (`vokra.frontend.n_fft`).
    pub n_fft: u32,
    /// Hop length between successive frames (`vokra.frontend.hop`).
    pub hop: u32,
    /// Analysis window length (`vokra.frontend.win_length`).
    pub win_length: u32,
    /// Window function name, e.g. `"hann"` (`vokra.frontend.window_type`).
    pub window_type: String,
    /// Mel filterbank normalization, e.g. `"slaney"` (`vokra.frontend.mel_norm`).
    pub mel_norm: String,
    /// HTK mel scale if `true`, else Slaney (`vokra.frontend.htk_mode`).
    pub htk_mode: bool,
    /// Lowest mel band edge in Hz (`vokra.frontend.fmin`).
    pub fmin: f32,
    /// Highest mel band edge in Hz (`vokra.frontend.fmax`).
    pub fmax: f32,
    /// Number of mel bands (`vokra.frontend.n_mels`).
    pub n_mels: u32,
    /// Signal padding mode, e.g. `"reflect"` (`vokra.frontend.pad_mode`).
    pub pad_mode: String,
    /// Whether DC offset is removed before framing (`vokra.frontend.dc_offset_removal`).
    pub dc_offset_removal: bool,
    /// Pre-emphasis coefficient, `0.0` = disabled (`vokra.frontend.pre_emphasis`).
    pub pre_emphasis: f32,
    /// Input sample rate in Hz (`vokra.frontend.sample_rate`).
    pub sample_rate: u32,
}

impl FrontendSpec {
    /// Returns the 13 `vokra.frontend.*` key/value pairs for this spec.
    pub fn to_gguf_kv(&self) -> Vec<(String, GgufMetadataValue)> {
        vec![
            (
                chunks::KEY_FRONTEND_N_FFT.to_owned(),
                GgufMetadataValue::U32(self.n_fft),
            ),
            (
                chunks::KEY_FRONTEND_HOP.to_owned(),
                GgufMetadataValue::U32(self.hop),
            ),
            (
                chunks::KEY_FRONTEND_WIN_LENGTH.to_owned(),
                GgufMetadataValue::U32(self.win_length),
            ),
            (
                chunks::KEY_FRONTEND_WINDOW_TYPE.to_owned(),
                GgufMetadataValue::String(self.window_type.clone()),
            ),
            (
                chunks::KEY_FRONTEND_MEL_NORM.to_owned(),
                GgufMetadataValue::String(self.mel_norm.clone()),
            ),
            (
                chunks::KEY_FRONTEND_HTK_MODE.to_owned(),
                GgufMetadataValue::Bool(self.htk_mode),
            ),
            (
                chunks::KEY_FRONTEND_FMIN.to_owned(),
                GgufMetadataValue::F32(self.fmin),
            ),
            (
                chunks::KEY_FRONTEND_FMAX.to_owned(),
                GgufMetadataValue::F32(self.fmax),
            ),
            (
                chunks::KEY_FRONTEND_N_MELS.to_owned(),
                GgufMetadataValue::U32(self.n_mels),
            ),
            (
                chunks::KEY_FRONTEND_PAD_MODE.to_owned(),
                GgufMetadataValue::String(self.pad_mode.clone()),
            ),
            (
                chunks::KEY_FRONTEND_DC_OFFSET_REMOVAL.to_owned(),
                GgufMetadataValue::Bool(self.dc_offset_removal),
            ),
            (
                chunks::KEY_FRONTEND_PRE_EMPHASIS.to_owned(),
                GgufMetadataValue::F32(self.pre_emphasis),
            ),
            (
                chunks::KEY_FRONTEND_SAMPLE_RATE.to_owned(),
                GgufMetadataValue::U32(self.sample_rate),
            ),
        ]
    }

    /// Writes all 13 `vokra.frontend.*` keys into `builder`.
    pub fn write_into(&self, builder: &mut GgufBuilder) {
        for (key, value) in self.to_gguf_kv() {
            builder.add_metadata(&key, value);
        }
    }

    /// Reads a [`FrontendSpec`] from a parsed GGUF file.
    ///
    /// Returns [`GgufError::MissingKey`] if any of the 13 keys is absent or
    /// [`GgufError::WrongType`] if a key holds an unexpected value type. This
    /// is a plain deserialization — it does **not** validate the values
    /// against the runtime (that is FR-LD-03 / M1-03).
    pub fn from_gguf(file: &GgufFile) -> Result<Self, GgufError> {
        Ok(Self {
            n_fft: get_u32(file, chunks::KEY_FRONTEND_N_FFT)?,
            hop: get_u32(file, chunks::KEY_FRONTEND_HOP)?,
            win_length: get_u32(file, chunks::KEY_FRONTEND_WIN_LENGTH)?,
            window_type: get_string(file, chunks::KEY_FRONTEND_WINDOW_TYPE)?,
            mel_norm: get_string(file, chunks::KEY_FRONTEND_MEL_NORM)?,
            htk_mode: get_bool(file, chunks::KEY_FRONTEND_HTK_MODE)?,
            fmin: get_f32(file, chunks::KEY_FRONTEND_FMIN)?,
            fmax: get_f32(file, chunks::KEY_FRONTEND_FMAX)?,
            n_mels: get_u32(file, chunks::KEY_FRONTEND_N_MELS)?,
            pad_mode: get_string(file, chunks::KEY_FRONTEND_PAD_MODE)?,
            dc_offset_removal: get_bool(file, chunks::KEY_FRONTEND_DC_OFFSET_REMOVAL)?,
            pre_emphasis: get_f32(file, chunks::KEY_FRONTEND_PRE_EMPHASIS)?,
            sample_rate: get_u32(file, chunks::KEY_FRONTEND_SAMPLE_RATE)?,
        })
    }

    /// Lists every field in which `self` (a model's declared front-end) differs
    /// from `runtime` (what the consuming model actually computes).
    ///
    /// An empty result means the two specs are **bit-exact**. Integer, boolean
    /// and string fields are compared for equality; the three `f32` fields are
    /// compared bit-for-bit via [`f32::to_bits`] (so `+0.0` and `-0.0` differ,
    /// and two `NaN`s of the same bit pattern match) — a tolerance would hide a
    /// real front-end change (reviewer C note #2).
    pub fn diff(&self, runtime: &FrontendSpec) -> Vec<FieldMismatch> {
        let mut out = Vec::new();
        macro_rules! cmp {
            ($field:ident) => {
                if self.$field != runtime.$field {
                    out.push(FieldMismatch {
                        field: stringify!($field),
                        model: format!("{:?}", self.$field),
                        runtime: format!("{:?}", runtime.$field),
                    });
                }
            };
        }
        macro_rules! cmp_f32 {
            ($field:ident) => {
                if self.$field.to_bits() != runtime.$field.to_bits() {
                    out.push(FieldMismatch {
                        field: stringify!($field),
                        model: format!("{:?}", self.$field),
                        runtime: format!("{:?}", runtime.$field),
                    });
                }
            };
        }
        cmp!(n_fft);
        cmp!(hop);
        cmp!(win_length);
        cmp!(window_type);
        cmp!(mel_norm);
        cmp!(htk_mode);
        cmp_f32!(fmin);
        cmp_f32!(fmax);
        cmp!(n_mels);
        cmp!(pad_mode);
        cmp!(dc_offset_removal);
        cmp_f32!(pre_emphasis);
        cmp!(sample_rate);
        out
    }

    /// Validates `self` (a model's declared front-end) against `runtime` under
    /// `policy` (FR-LD-03).
    ///
    /// - a bit-exact match ([`diff`](Self::diff) empty) is always `Ok(())`;
    /// - under [`FrontendPolicy::Fail`] (the load-time default) any mismatch is
    ///   a [`VokraError::FrontendMismatch`] whose message lists the differing
    ///   fields;
    /// - under [`FrontendPolicy::Warn`] the same report is written to **stderr**
    ///   (the zero-dependency runtime has no logging crate — NFR-DS-02) and
    ///   loading continues with `Ok(())`.
    pub fn check_against(
        &self,
        runtime: &FrontendSpec,
        policy: FrontendPolicy,
    ) -> crate::error::Result<()> {
        let diffs = self.diff(runtime);
        if diffs.is_empty() {
            return Ok(());
        }
        let report = format_mismatch_report(&diffs);
        match policy {
            FrontendPolicy::Fail => Err(VokraError::FrontendMismatch(report)),
            FrontendPolicy::Warn => {
                eprintln!("vokra: frontend_spec mismatch (continuing, Warn policy): {report}");
                Ok(())
            }
        }
    }
}

/// How a [`FrontendSpec`] mismatch is handled at model load (FR-LD-03).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrontendPolicy {
    /// Bit-exact match is required; any mismatch is a hard
    /// [`VokraError::FrontendMismatch`]. The default, so a silently
    /// mis-configured front-end cannot corrupt features unnoticed.
    #[default]
    Fail,
    /// Report the mismatch to stderr and continue loading. For lenient callers
    /// that knowingly accept a non-matching front-end.
    Warn,
}

/// One field that differs between a model's [`FrontendSpec`] and the runtime's,
/// as reported by [`FrontendSpec::diff`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldMismatch {
    /// The `frontend_spec` field name (e.g. `"htk_mode"`).
    pub field: &'static str,
    /// The model file's value, `Debug`-formatted.
    pub model: String,
    /// The runtime's expected value, `Debug`-formatted.
    pub runtime: String,
}

/// Renders a non-empty list of field mismatches into a one-line-per-field report.
fn format_mismatch_report(diffs: &[FieldMismatch]) -> String {
    let mut s = format!(
        "{} field(s) differ from the runtime front-end:",
        diffs.len()
    );
    for d in diffs {
        s.push_str(&format!(
            " [{}: model={} runtime={}]",
            d.field, d.model, d.runtime
        ));
    }
    s
}

fn get_u32(file: &GgufFile, key: &str) -> Result<u32, GgufError> {
    match file.get(key) {
        Some(GgufMetadataValue::U32(v)) => Ok(*v),
        Some(_) => Err(wrong_type(key, GgufValueType::U32)),
        None => Err(GgufError::MissingKey(key.to_owned())),
    }
}

fn get_f32(file: &GgufFile, key: &str) -> Result<f32, GgufError> {
    match file.get(key) {
        Some(GgufMetadataValue::F32(v)) => Ok(*v),
        Some(_) => Err(wrong_type(key, GgufValueType::F32)),
        None => Err(GgufError::MissingKey(key.to_owned())),
    }
}

fn get_bool(file: &GgufFile, key: &str) -> Result<bool, GgufError> {
    match file.get(key) {
        Some(GgufMetadataValue::Bool(v)) => Ok(*v),
        Some(_) => Err(wrong_type(key, GgufValueType::Bool)),
        None => Err(GgufError::MissingKey(key.to_owned())),
    }
}

fn get_string(file: &GgufFile, key: &str) -> Result<String, GgufError> {
    match file.get(key) {
        Some(GgufMetadataValue::String(v)) => Ok(v.clone()),
        Some(_) => Err(wrong_type(key, GgufValueType::String)),
        None => Err(GgufError::MissingKey(key.to_owned())),
    }
}

fn wrong_type(key: &str, expected: GgufValueType) -> GgufError {
    GgufError::WrongType {
        key: key.to_owned(),
        expected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> FrontendSpec {
        // Values mirror Whisper's front-end (openai/whisper whisper/audio.py):
        // used here only to exercise round-tripping, not as an assertion of
        // model behaviour.
        FrontendSpec {
            n_fft: 400,
            hop: 160,
            win_length: 400,
            window_type: "hann".to_owned(),
            mel_norm: "slaney".to_owned(),
            htk_mode: false,
            fmin: 0.0,
            fmax: 8000.0,
            n_mels: 80,
            pad_mode: "reflect".to_owned(),
            dc_offset_removal: false,
            pre_emphasis: 0.0,
            sample_rate: 16_000,
        }
    }

    #[test]
    fn roundtrip_all_thirteen_fields() {
        let spec = sample();
        let mut b = GgufBuilder::new();
        spec.write_into(&mut b);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        // All 13 frontend keys present and no others, plus the 2 unconditional
        // `vokra.schema.*` stamps the writer adds to every GGUF.
        assert_eq!(file.metadata().len(), 13 + 2);
        let read = FrontendSpec::from_gguf(&file).unwrap();
        assert_eq!(read, spec);
    }

    #[test]
    fn missing_key_is_reported() {
        let mut b = GgufBuilder::new();
        // Only one of the 13 keys is present.
        b.add_u32(chunks::KEY_FRONTEND_N_FFT, 400);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(matches!(
            FrontendSpec::from_gguf(&file),
            Err(GgufError::MissingKey(_))
        ));
    }

    #[test]
    fn wrong_type_is_reported() {
        let spec = sample();
        let mut b = GgufBuilder::new();
        spec.write_into(&mut b);
        // Clobber one key with the wrong type.
        b.add_string(chunks::KEY_FRONTEND_N_FFT, "not-a-number");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(matches!(
            FrontendSpec::from_gguf(&file),
            Err(GgufError::WrongType { .. })
        ));
    }

    // --- inspection (M1-03) --------------------------------------------------

    #[test]
    fn identical_specs_have_no_diff_and_pass_every_policy() {
        let a = sample();
        let b = sample();
        assert!(a.diff(&b).is_empty());
        assert!(a.check_against(&b, FrontendPolicy::Fail).is_ok());
        assert!(a.check_against(&b, FrontendPolicy::Warn).is_ok());
    }

    #[test]
    fn default_policy_is_fail() {
        assert_eq!(FrontendPolicy::default(), FrontendPolicy::Fail);
    }

    #[test]
    fn single_field_mismatch_is_pinpointed() {
        let runtime = sample();
        let mut model = sample();
        model.htk_mode = true; // the model claims HTK; runtime computes Slaney.
        let diffs = model.diff(&runtime);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].field, "htk_mode");
        assert_eq!(diffs[0].model, "true");
        assert_eq!(diffs[0].runtime, "false");
    }

    #[test]
    fn fail_policy_errors_and_warn_policy_continues_on_mismatch() {
        let runtime = sample();
        let mut model = sample();
        model.n_fft = 512;
        // Fail: hard error naming the field.
        match model.check_against(&runtime, FrontendPolicy::Fail) {
            Err(VokraError::FrontendMismatch(msg)) => {
                assert!(msg.contains("n_fft"), "report should name the field: {msg}");
                assert!(msg.contains("512") && msg.contains("400"), "{msg}");
            }
            other => panic!("expected FrontendMismatch, got {other:?}"),
        }
        // Warn: the same mismatch is tolerated (report goes to stderr).
        assert!(model.check_against(&runtime, FrontendPolicy::Warn).is_ok());
    }

    #[test]
    fn f32_fields_are_compared_bit_exact_not_by_tolerance() {
        let runtime = sample(); // fmin = 0.0
        let mut model = sample();
        // A minuscule but non-zero fmax change is a real front-end difference.
        model.fmax = 8000.5;
        assert_eq!(model.diff(&runtime).len(), 1);
        assert_eq!(model.diff(&runtime)[0].field, "fmax");

        // -0.0 differs from +0.0 bit-for-bit even though they are `==`.
        let mut neg_zero = sample();
        neg_zero.fmin = -0.0;
        assert_eq!(neg_zero.fmin, runtime.fmin); // `==` says equal ...
        assert_eq!(neg_zero.diff(&runtime).len(), 1); // ... but bits differ.
        assert_eq!(neg_zero.diff(&runtime)[0].field, "fmin");
    }

    #[test]
    fn multiple_field_mismatches_are_all_reported() {
        let runtime = sample();
        let mut model = sample();
        model.hop = 200;
        model.mel_norm = "none".to_owned();
        model.pre_emphasis = 0.97;
        let diffs = model.diff(&runtime);
        assert_eq!(diffs.len(), 3);
        let fields: Vec<&str> = diffs.iter().map(|d| d.field).collect();
        assert!(fields.contains(&"hop"));
        assert!(fields.contains(&"mel_norm"));
        assert!(fields.contains(&"pre_emphasis"));
    }
}
