//! Typed view of the `vokra.frontend.*` metadata chunk (`frontend_spec`).
//!
//! [`FrontendSpec`] has one field per key defined in [`super::chunks`]; it can
//! be written into a [`GgufBuilder`](super::GgufBuilder) and read back out of a
//! [`GgufFile`](super::GgufFile).
//!
//! # Scope: read/write only (M0), not inspection (M1)
//!
//! This type deliberately contains **no validation and no bit-exact match
//! check**. Enforcing that a model's front-end matches the runtime's — warning
//! or failing on mismatch — is FR-LD-03, a v0.1 MVP concern owned by **M1-03**
//! (milestones.md §4.2 table note 1). Keeping inspection out of M0 prevents
//! scope creep; a future `FrontendSpec::check_against` (or similar) belongs to
//! that later work package.

use super::chunks;
use super::value::{GgufMetadataValue, GgufValueType};
use super::{GgufBuilder, GgufError, GgufFile};

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

        // All 13 keys present and no others.
        assert_eq!(file.metadata().len(), 13);
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
}
